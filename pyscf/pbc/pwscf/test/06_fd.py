import unittest
import tempfile
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.pwscf import khf, kuhf, krks, kuks
import pyscf.pbc
from numpy.testing import assert_allclose
pyscf.pbc.DEBUG = False


def setUpModule():
    global CELL, KPTS, ATOM, KPT1
    CELL = pbcgto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
    )
    CELL.mesh = [13, 13, 13]
    # CELL.mesh = [27, 27, 27]
    CELL.build()

    kmesh = [3, 1, 1]
    KPTS = CELL.make_kpts(kmesh)

    ATOM = pbcgto.Cell(
        atom = "C 0 0 0",
        a = np.eye(3) * 4,
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
        spin=2,
    )
    ATOM.mesh = [25, 25, 25]
    ATOM.build()
    ATOM.verbose = 6

    nk = 1
    kmesh = (nk,)*3
    KPT1 = ATOM.make_kpts(kmesh)


def tearDownModule():
    global CELL, ATOM
    del CELL, ATOM


class KnownValues(unittest.TestCase):

    def _get_calc(self, cell, kpts, spinpol=False, xc=None, run=True, **kwargs):
        if xc is None:
            if not spinpol:
                mf = khf.PWKRHF(cell, kpts)
            else:
                mf = kuhf.PWKUHF(cell, kpts)
        else:
            if not spinpol:
                mf = krks.PWKRKS(cell, kpts, xc=xc)
            else:
                mf = kuks.PWKUKS(cell, kpts, xc=xc)
        mf.__dict__.update(**kwargs)
        if run:
            mf.kernel()
        return mf

    def _check_rhf_uhf(self, cell, kpts, xc=None, rtol=1e-7, atol=1e-7):
        rmf = self._get_calc(cell, kpts, spinpol=False, xc=xc)
        umf = self._get_calc(cell, kpts, spinpol=True, xc=xc)
        assert_almost_equal(rmf.e_tot, umf.e_tot, rtol=rtol, atol=atol)

    def _check_fd(self, mf):
        if not mf.converged:
            mf.kernel()
            assert mv.converged
        mo_energy, mo_occ = mf.get_mo_energy(mf.mo_coeff, mf.mo_occ)
        # mo_energy, mo_occ = mf.mo_energy, mf.mo_occ
        delta = 1e-5
        cell = mf.cell
        mesh = cell.mesh
        Gv = cell.get_Gv(mesh)

        spinpol = isinstance(mf, kuhf.PWKUHF)
        if spinpol:
            nkpts = len(mf.mo_coeff[0])
        else:
            nkpts = len(mf.mo_coeff)

        def _update(Ct_ks):
            mf.update_pp(Ct_ks)
            mf.update_k(Ct_ks, mo_occ)

        def _transform(C_ks, mocc_ks, k, s=None):
            if s is not None:
                Ctspin_ks, vbm, cbm = _transform(C_ks[s], mocc_ks[s], k)
                Ct_ks = [[C.copy() for C in Cspin] for Cspin in C_ks]
                Ct_ks[s] = Ctspin_ks
                return Ct_ks, vbm, cbm
            vbm = np.max(np.where(mocc_ks[k] > 0.9))
            cbm = np.min(np.where(mocc_ks[k] < 0.1))
            transform = np.identity(C_ks[k].shape[0])
            transform[vbm, vbm] = np.sqrt(0.5)
            transform[vbm, cbm] = np.sqrt(0.5)
            transform[cbm, cbm] = np.sqrt(0.5)
            transform[cbm, vbm] = -np.sqrt(0.5)
            Ct_k = transform.dot(C_ks[k])
            Ct_ks = [C_k.copy() for C_k in C_ks]
            Ct_ks[k] = Ct_k.copy()
            return Ct_ks, vbm, cbm

        def _eig_subspace_ham(Ct_ks, k, s=None):
            if s is not None:
                Ctt_ks = [[C.copy() for C in Cspin] for Cspin in Ct_ks]
            else:
                Ctt_ks = [C.copy() for C in Ct_ks]
            Ctt_ks, moett_ks = mf.eig_subspace(
                Ctt_ks, mo_occ, Gv=Gv, mesh=mesh
            )[:2]
            if s is not None:
                moett_ks = moett_ks[s]
                Ct_ks = Ct_ks[s]
                Ctt_ks = Ctt_ks[s]
            ham1 = np.einsum("ig,jg->ij", Ctt_ks[k], Ct_ks[k].conj())
            ham2 = np.einsum("ki,i,ij->kj", ham1.conj().T, moett_ks[k], ham1)
            return ham2

        def _new_vbms(Ct_ks, vbm, cbm, k, s=None):
            vj_R = mf.get_vj_R(Ct_ks, mo_occ)
            if s is not None:
                Ct_ks = Ct_ks[s]
            new_vbm = Ct_ks[k][vbm].copy()
            new_cbm = Ct_ks[k][cbm].copy()
            new_vbm_p = new_vbm + 0.5 * delta * new_cbm
            new_vbm_m = new_vbm - 0.5 * delta * new_cbm
            return new_vbm_p, new_vbm_m

        def _run_test(s=None):
            for k in range(nkpts):
                Ct_ks, vbm, cbm = _transform(mf.mo_coeff, mo_occ, k, s=s)
                _update(Ct_ks)
                ham2 = _eig_subspace_ham(Ct_ks, k, s=s)
                new_ham = mf.get_mo_energy(Ct_ks, mo_occ, full_ham=True)
                if s is not None:
                    new_ham = new_ham[s]
                expected_de = new_ham[k][vbm, cbm] + new_ham[k][cbm, vbm]
                if hasattr(mf, "xc"):
                    if not mf._numint.libxc.is_hybrid_xc(mf.xc):
                        ham2_term = ham2[vbm, cbm] + ham2[cbm, vbm],
                        assert_allclose(ham2_term, expected_de)
                new_vbm_p, new_vbm_m = _new_vbms(Ct_ks, vbm, cbm, k, s=s)

                if s is None:
                    Ct_ks[k][vbm] = new_vbm_m
                else:
                    Ct_ks[s][k][vbm] = new_vbm_m
                _update(Ct_ks)
                em = mf.energy_elec(Ct_ks, mo_occ, Gv=Gv, mesh=mesh)

                if s is None:
                    Ct_ks[k][vbm] = new_vbm_p
                else:
                    Ct_ks[s][k][vbm] = new_vbm_p
                _update(Ct_ks)
                ep = mf.energy_elec(Ct_ks, mo_occ, Gv=Gv, mesh=mesh)
                fd = (ep - em) / delta

                # NOTE need to understand the factor of 2 a bit better
                # but the factor of nkpts is just because the fd energy
                # is per unit cell, but the gap is the energy derivative
                # for the supercell with respect to perturbing the orbital
                expected_de = expected_de * 2 / nkpts
                if spinpol:
                    # TODO why?
                    expected_de /= 2
                print(expected_de, fd)
                assert_allclose(expected_de, fd, atol=1e-8, rtol=1e-8)
        
        if not spinpol:
            _run_test()
        else:
            _run_test(s=0)
            _run_test(s=1)


    def test_fd_hf(self):
        rmf = self._get_calc(CELL, KPTS, nvir=2)
        umf = self._get_calc(CELL, KPTS, nvir=2, spinpol=True)
        assert_allclose(rmf.e_tot, umf.e_tot)
        assert_allclose(rmf.mo_energy, umf.mo_energy[0])
        assert_allclose(rmf.mo_energy, umf.mo_energy[1])
        assert_allclose(rmf.mo_occ, umf.mo_occ[0])
        assert_allclose(rmf.mo_occ, umf.mo_occ[1])
        self._check_fd(rmf)
        self._check_fd(umf)

    def _check_fd_ks(self, xc, mesh=None):
        if mesh is None:
            cell = CELL
        else:
            cell = CELL.copy()
            cell.mesh = mesh
            cell.build()
        rmf = self._get_calc(cell, KPTS, nvir=2, xc=xc, spinpol=False,
                             damp_type="simple", damp_factor=0.7)
        umf = self._get_calc(cell, KPTS, nvir=2, xc=xc, spinpol=True,
                             damp_type="simple", damp_factor=0.7)
        assert_allclose(rmf.e_tot, umf.e_tot)
        assert_allclose(rmf.mo_energy, umf.mo_energy[0])
        assert_allclose(rmf.mo_energy, umf.mo_energy[1])
        assert_allclose(rmf.mo_occ, umf.mo_occ[0])
        assert_allclose(rmf.mo_occ, umf.mo_occ[1])
        self._check_fd(rmf)
        self._check_fd(umf)

    def test_fd_ks_lda(self):
        self._check_fd_ks("LDA")

    def test_fd_ks_gga(self):
        self._check_fd_ks("PBE")

    def test_fd_ks_mgga(self):
        self._check_fd_ks("R2SCAN", mesh=[21, 21, 21])

    def test_fd_ks_hyb(self):
        self._check_fd_ks("PBE0")


if __name__ == "__main__":
    print("Finite difference for pbc.pwscf -- khf, kuhf, krks, kuks")
    unittest.main()

