import unittest
import tempfile
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.pwscf.smearing import smearing_
from pyscf.pbc.pwscf import khf, kuhf, krks, kuks
from pyscf.pbc.pwscf.ncpp_cell import NCPPCell
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
        verbose=0,
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
        spin=-2,
        verbose=0,
    )
    ATOM.mesh = [25, 25, 25]
    ATOM.build()

    nk = 1
    kmesh = (nk,)*3
    KPT1 = ATOM.make_kpts(kmesh)


def tearDownModule():
    global CELL, ATOM, KPTS, KPT1
    del CELL, ATOM, KPTS, KPT1


class KnownValues(unittest.TestCase):
    def _get_calc(self, cell, kpts, spinpol=False, xc=None, run=True, **kwargs):
        """
        Helper function to make an SCF calculation for a test
        """
        ecut_wf = kwargs.pop("ecut_wf", None)
        ecut_rho = kwargs.pop("ecut_rho", None)
        if xc is None:
            if not spinpol:
                mf = khf.PWKRHF(cell, kpts, ecut_wf=ecut_wf, ecut_rho=ecut_rho)
            else:
                mf = kuhf.PWKUHF(
                    cell, kpts, ecut_wf=ecut_wf, ecut_rho=ecut_rho
                )
        else:
            if not spinpol:
                mf = krks.PWKRKS(
                    cell, kpts, xc=xc, ecut_wf=ecut_wf, ecut_rho=ecut_rho
                )
            else:
                mf = kuks.PWKUKS(
                    cell, kpts, xc=xc, ecut_wf=ecut_wf, ecut_rho=ecut_rho
                )
        mf.conv_tol = 1e-8
        mf.__dict__.update(**kwargs)
        if run:
            mf.kernel()
        return mf

    def _check_fd(self, mf):
        """
        Check a bunch of properties of the mean-field calculation:
        - that get_mo_energy matches the mo_energy from SCF
        - that energy_tot with moe_ks gives same output as SCF e_tot
          (also tests energy_elec for this consistency implicitly)
        - that eig_subspace, get_mo_energy(full_ham=True), and
          finite difference all give the same prediction for the
          change in total energy upon perturbation of the orbitals.
          This implicitly tests all routines for constructing
          the effective Hamiltonian, especially the XC potential.
        """
        if not mf.converged:
            mf.kernel()
            assert mf.converged
        mo_energy, mo_occ = mf.get_mo_energy(mf.mo_coeff, mf.mo_occ)
        if mf.istype("KRHF"):
            assert_allclose(mo_energy, mf.mo_energy, rtol=1e-8, atol=1e-8)
        else:
            assert_allclose(mo_energy[0], mf.mo_energy[0], rtol=1e-6, atol=1e-6)
            assert_allclose(mo_energy[1], mf.mo_energy[1], rtol=1e-6, atol=1e-6)
        etot_ref = mf.e_tot
        etot_check = mf.energy_tot(mf.mo_coeff, mf.mo_occ,
                                   moe_ks=mo_energy)
        assert_allclose(etot_check, etot_ref, atol=1e-9)
        delta = 1e-5
        cell = mf.cell
        mesh = mf.wf_mesh
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
                assert_allclose(expected_de, fd, atol=1e-8, rtol=1e-8)
        
        if not spinpol:
            _run_test()
        else:
            _run_test(s=0)
            _run_test(s=1)

    def test_fd_hf(self):
        """
        Run the _check_fd tests for spin-restricted and unrestricted
        Hartree-Fock.
        """
        ref = -10.649288588747416
        rmf = self._get_calc(CELL, KPTS, nvir=2)
        umf = self._get_calc(CELL, KPTS, nvir=2, spinpol=True)
        assert_allclose(rmf.e_tot, ref, atol=1e-7, rtol=0)
        assert_allclose(rmf.e_tot, umf.e_tot, atol=1e-7, rtol=0)
        assert_allclose(rmf.mo_energy, umf.mo_energy[0])
        assert_allclose(rmf.mo_energy, umf.mo_energy[1])
        assert_allclose(rmf.mo_occ, umf.mo_occ[0])
        assert_allclose(rmf.mo_occ, umf.mo_occ[1])
        self._check_fd(rmf)
        self._check_fd(umf)
        umf = self._get_calc(ATOM, KPT1, nvir=2, spinpol=True,
                             damp_type="anderson", ecut_wf=15)
        self._check_fd(umf)

    def _check_fd_ks(self, xc, mesh=None, ref=None, run_atom=False):
        """
        Run the _check_fd tests for spin-restricted and unrestricted
        Kohn-Sham DFT.
        """
        if mesh is None:
            cell = CELL
            atom = ATOM
        else:
            cell = CELL.copy()
            cell.mesh = mesh
            atom = ATOM
            cell.build()
        rmf = self._get_calc(cell, KPTS, nvir=2, xc=xc, spinpol=False,
                             damp_type="simple", damp_factor=0.7)
        umf = self._get_calc(cell, KPTS, nvir=2, xc=xc, spinpol=True,
                             damp_type="simple", damp_factor=0.7)
        if ref is not None:
            assert_allclose(rmf.e_tot, ref, atol=1e-7, rtol=0)
        assert_allclose(rmf.e_tot, umf.e_tot, atol=1e-7, rtol=0)
        assert_allclose(rmf.mo_energy, umf.mo_energy[0])
        assert_allclose(rmf.mo_energy, umf.mo_energy[1])
        assert_allclose(rmf.mo_occ, umf.mo_occ[0])
        assert_allclose(rmf.mo_occ, umf.mo_occ[1])
        self._check_fd(rmf)
        self._check_fd(umf)
        if run_atom:
            umf = self._get_calc(atom, KPT1, nvir=2, xc=xc, spinpol=True,
                                 damp_type="anderson", ecut_wf=15,
                                 ecut_rho=200)
            self._check_fd(umf)

    def test_fd_ks_lda(self):
        self._check_fd_ks("LDA", ref=-10.453600311477887, run_atom=True)

    def test_fd_ks_gga(self):
        self._check_fd_ks("PBE", ref=-10.931960348543591, run_atom=True)

    def test_fd_ks_mgga(self):
        self._check_fd_ks("R2SCAN", mesh=[21, 21, 21], ref=-10.881956126701505)

    def test_fd_ks_hyb(self):
        self._check_fd_ks("PBE0", ref=-10.940602656908139)

    def test_smearing(self):
        """
        Make sure that smearing is working (should give similar energy
        to the non-smearing calculation and have mf.mo_energy matching
        get_mo_energy, e_tot matching energy_tot(..., moe_ks), etc.)
        """
        xc = "LDA,VWN"
        rmf = self._get_calc(
            CELL, KPTS, nvir=6, xc=xc, run=False, ecut_wf=15
        )
        umf1 = self._get_calc(
            CELL, KPTS, nvir=6, spinpol=True, xc=xc, run=False, ecut_wf=15,
        )
        umf2 = self._get_calc(
            ATOM, KPT1, nvir=2, spinpol=True, xc=xc, run=False, ecut_wf=15
        )
        assert_allclose(umf1.e_tot, rmf.e_tot, atol=1e-7)
        check = True
        sigmas = [0.05, 0.05, 0.01]
        new_mfs = []
        for mf, sigma in zip([rmf, umf1, umf2], sigmas):
            mf.kernel()
            etot_nosmear = mf.e_tot
            mf = smearing_(mf, sigma=sigma, method="gauss")
            mf.kernel()
            etot_ref = mf.e_tot
            # energy with and without smearing doesn't change too much
            assert_allclose(etot_ref, etot_nosmear, atol=1e-2)
            moe_tst = mf.mo_energy
            mo_energy, mo_occ = mf.get_mo_energy(mf.mo_coeff, mf.mo_occ)
            if check:
                check = False
                assert_allclose(mo_energy, moe_tst, rtol=1e-8, atol=1e-8)
            etot_check = mf.energy_tot(mf.mo_coeff, mf.mo_occ, mo_energy)
            assert_allclose(etot_check, etot_ref, atol=1e-8)
            new_mfs.append(mf)
        assert_allclose(new_mfs[1].e_tot, new_mfs[0].e_tot, atol=1e-7)

    def test_init_guesses(self):
        """
        Test a bunch of initial guesses for the SCF methods to make sure
        they give consistent results and don't crash.
        """
        for spinpol in [False, True]:
            mf = self._get_calc(
                CELL, KPTS, nvir=2, xc="LDA,VWN", spinpol=spinpol,
                ecut_wf=15, run=False
            )
            mf.init_guess = "hcore"
            mf.conv_tol = 1e-8
            e_ref = mf.kernel()
            e_tots = []
            for ig in ["h1e", "cycle1", "scf"]:
                mf.init_guess = ig
                e_tots.append(mf.kernel())
            e_tots.append(mf.kernel(C0=mf.mo_coeff))
            mf2 = self._get_calc(
                CELL, KPTS, nvir=2, xc="LDA,VWN", spinpol=spinpol,
                ecut_wf=15, run=False
            )
            e_tots.append(mf2.kernel(chkfile=mf.chkfile))
            C_ks, mocc_ks = mf.from_chk(mf.chkfile)
            e_tots.append(mf2.energy_tot(C_ks, mocc_ks))
            assert_allclose(np.array(e_tots) - e_ref, 0, atol=1e-7)

    def test_meshes(self):
        """
        Make sure that modifying the choices of meshes works
        """
        mf = self._get_calc(
            CELL, KPTS, nvir=2, xc="LDA,VWN", spinpol=False,
            ecut_wf=15, run=False
        )
        mf2 = self._get_calc(
            CELL, KPTS, nvir=2, xc="LDA,VWN", spinpol=False, run=False
        )
        orig_wf_mesh = mf.wf_mesh
        orig_xc_mesh = mf.xc_mesh
        e1 = mf.kernel()
        assert (mf2.wf_mesh == mf2.xc_mesh).all()
        assert (mf2.wf_mesh == CELL.mesh).all()
        e2 = mf2.kernel()
        # energy doesn't change because default wf_mesh avoids aliasing
        mf.set_meshes(wf_mesh=[m+5 for m in orig_wf_mesh], xc_mesh=orig_xc_mesh)
        e3 = mf.kernel()
        assert_allclose(e1, e3, atol=1e-7)
        mf.set_meshes(wf_mesh=orig_wf_mesh, xc_mesh=orig_wf_mesh)
        e4 = mf.kernel()
        # energy changes a bit bit the XC integration precision changes
        assert_allclose(e1, e3, atol=1e-5)


if __name__ == "__main__":
    print("Finite difference for pbc.pwscf -- khf, kuhf, krks, kuks")
    unittest.main()

