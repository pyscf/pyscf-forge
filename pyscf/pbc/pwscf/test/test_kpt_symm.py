import unittest
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.pwscf import khf, krks, kuks, jk, kpt_symm
from pyscf.pbc.pwscf.smearing import smearing_
from pyscf.pbc.pwscf.pw_helper import wf_ifft
import numpy as np
from pyscf.pbc import tools
from numpy.testing import assert_almost_equal
import time


ECUT_WF = 20
PRINT_TIMES = False


def get_mf_and_kpts():
    cell = pbcgto.Cell(
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
    cell.mesh = [20, 20, 20]
    cell.build()
    kmesh = (3, 3, 3)
    kpts = cell.make_kpts(kmesh)

    cell_sym = cell.copy()
    cell_sym.space_group_symmetry = True
    cell_sym.symmorphic = True
    cell_sym.build()
    kpts_sym = cell_sym.make_kpts(
        kmesh,
        time_reversal_symmetry=True,
        space_group_symmetry=True,
    )

    mf = krks.PWKRKS(cell, kpts, xc="LDA,VWN", ecut_wf=ECUT_WF)
    mf = smearing_(mf, sigma=0.01, method='gauss')
    mf.kernel()
    return mf, cell, kpts_sym, cell_sym


def setUpModule():
    global mf, cell, kpts_sym, cell_sym
    mf, cell, kpts_sym, cell_sym = get_mf_and_kpts()


def tearDownModule():
    global mf, cell, kpts_sym
    del mf
    del cell
    del kpts_sym


class KnownValues(unittest.TestCase):
    def test_get_rho(self):
        global mf, cell, kpts_sym
        C_ks = [coeff.copy() for coeff in mf.mo_coeff]
        mocc_ks = mf.mo_occ

        mf2 = kpt_symm.KsymAdaptedPWKRKS(cell, kpts_sym, ecut_wf=ECUT_WF)

        t0 = time.monotonic()
        rho_R = jk.get_rho_R(C_ks, mocc_ks, mf.wf_mesh, basis_ks=mf._basis_data)
        t1 = time.monotonic()
        Csym_ks = [C_ks[k_bz].copy() for k_bz in kpts_sym.ibz2bz]
        moccsym_ks = [mocc_ks[k_bz] for k_bz in kpts_sym.ibz2bz]
        t2 = time.monotonic()
        rhosym_R = kpt_symm.get_rho_R_ksym(
            Csym_ks, moccsym_ks, mf2.wf_mesh, kpts_sym, basis_ks=mf2._basis_data
        )
        t3 = time.monotonic()
        assert np.max(np.abs(rhosym_R - rho_R)) / np.mean(rho_R) < 1e-4
        if PRINT_TIMES:
            print("TIMES", t1 - t0, t3 - t2, len(C_ks), len(Csym_ks))

        mf2 = smearing_(mf2, sigma=0.01, method='gauss')
        mf2.init_jk()
        mf2.init_pp()
        eref = mf.energy_elec(C_ks, mocc_ks)
        epred = mf2.energy_elec(Csym_ks, moccsym_ks)

        rho1 = mf.get_rho_for_xc("LDA", C_ks, mocc_ks)
        rho2 = mf2.get_rho_for_xc("LDA", Csym_ks, moccsym_ks)
        assert_almost_equal(np.abs(rho1 - rho2).mean() * cell.vol, 0, 6)
        assert_almost_equal(epred, eref, 6)
    
    def test_get_wf(self):
        global mf, cell, kpts_sym
        C_ks = [coeff.copy() for coeff in mf.mo_coeff]
        Csym_ks = [C_ks[k_bz].copy() for k_bz in kpts_sym.ibz2bz]
        if mf._basis_data is not None:
            def _ecut2grid_(basis_ks, C_ks):
                for k, (basis, C_k) in enumerate(zip(basis_ks, C_ks)):
                    nmo = C_k.shape[0]
                    ngrid = np.prod(mf.wf_mesh)
                    newC_k = np.zeros((nmo, ngrid), C_k.dtype)
                    newC_k[:, basis.indexes] = C_k
                    C_ks[k] = newC_k
            _ecut2grid_(mf._basis_data, C_ks)
            _ecut2grid_([mf._basis_data[k_bz] for k_bz in kpts_sym.ibz2bz], Csym_ks)
        Cpred_ks = kpt_symm.get_C_from_C_ibz(Csym_ks, mf.wf_mesh, kpts_sym)
        k = 0
        for moe, Cref, Cpred in zip(mf.mo_energy, C_ks, Cpred_ks):
            dot1 = np.einsum("ig,jg->ij", Cref.conj(), Cref)
            dot2 = np.einsum("ig,jg->ij", Cref.conj(), Cpred)
            dot3 = np.einsum("ig,jg->ij", Cpred.conj(), Cpred)
            rdot1 = np.abs(dot1)
            rdot2 = np.abs(dot2)
            rdot3 = np.abs(dot3)
            assert_almost_equal(rdot1[:2, :2], rdot2[:2, :2], 6)
            assert_almost_equal(rdot1[:2], rdot2[:2], 4)
            assert_almost_equal(rdot1, rdot3, 6)
            k += 1
    
    def test_get_wf_real(self):
        global mf, cell, kpts_sym
        C_ks = [coeff.copy() for coeff in mf.mo_coeff]
        if mf._basis_data is None:
            C_ks_R = [tools.ifft(C_k, mf.wf_mesh) for C_k in C_ks]
        else:
            C_ks_R = [wf_ifft(C_k, mf.wf_mesh, basis)
                      for C_k, basis in zip(C_ks, mf._basis_data)]
        Csym_ks_R = [C_ks_R[k_bz].copy() for k_bz in kpts_sym.ibz2bz]
        Cpred_ks_R = kpt_symm.get_C_from_C_ibz(Csym_ks_R, mf.wf_mesh, kpts_sym,
                                               realspace=True) 
        k = 0
        norm = C_ks[0].shape[-1]
        for moe, Cref, Cpred in zip(mf.mo_energy, C_ks_R, Cpred_ks_R):
            dot1 = norm * np.einsum("ig,jg->ij", Cref.conj(), Cref)
            dot2 = norm * np.einsum("ig,jg->ij", Cref.conj(), Cpred)
            dot3 = norm * np.einsum("ig,jg->ij", Cpred.conj(), Cpred)
            rdot1 = np.abs(dot1)
            rdot2 = np.abs(dot2)
            rdot3 = np.abs(dot3)
            assert_almost_equal(rdot1[:2, :2], rdot2[:2, :2], 6)
            assert_almost_equal(rdot1[:2], rdot2[:2], 4)
            assert_almost_equal(rdot1, rdot3, 6)
            k += 1

    def test_hf_symm(self):
        global cell, cell_sym

        import time

        kmesh = (3, 3, 3)
        kpts = cell.make_kpts(kmesh)

        kpts_sym = cell_sym.make_kpts(
            kmesh,
            time_reversal_symmetry=True,
            space_group_symmetry=True,
        )
        mf = khf.PWKRHF(cell, kpts, ecut_wf=10)
        t0 = time.monotonic()
        mf.kernel()
        t1 = time.monotonic()
        mf_sym = kpt_symm.KsymAdaptedPWKRHF(cell_sym, kpts_sym, ecut_wf=10)
        t2 = time.monotonic()
        mf_sym.kernel()
        t3 = time.monotonic()
        assert_almost_equal(mf_sym.e_tot, mf.e_tot, 5)
        if PRINT_TIMES:
            print(t1 - t0, t3 - t2)


if __name__ == "__main__":
    unittest.main()

