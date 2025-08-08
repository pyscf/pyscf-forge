import unittest
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.pwscf import khf, krks, jk, kpt_symm
from pyscf.pbc.pwscf.smearing import smearing_
import numpy as np
from pyscf.pbc import tools
from numpy.testing import assert_almost_equal
import time


def get_mf_and_kpts():
    cell = pbcgto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        # atom = "C 0 0 0; C 0.8 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
    )
    # cell.mesh = [28, 28, 28]
    cell.mesh = [24, 24, 24]
    cell.build()
    # kmesh = (4, 4, 4)
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

    mf = krks.PWKRKS(cell, kpts, xc="LDA,VWN")
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
        
        t0 = time.monotonic()
        rho_R = jk.get_rho_R(C_ks, mocc_ks, cell.mesh)
        t1 = time.monotonic()
        Csym_ks = [C_ks[k_bz].copy() for k_bz in kpts_sym.ibz2bz]
        print(len(Csym_ks))
        moccsym_ks = [mocc_ks[k_bz] for k_bz in kpts_sym.ibz2bz]
        t2 = time.monotonic()
        rhosym_R = kpt_symm.get_rho_R_ksym(Csym_ks, moccsym_ks, cell.mesh, kpts_sym)
        t3 = time.monotonic()
        print(rho_R.sum())
        print(rhosym_R.sum())
        print(np.linalg.norm(rhosym_R - rho_R))
        print(np.abs(rhosym_R - rho_R).sum() / rho_R.sum())
        print(np.max(np.abs(rhosym_R - rho_R)) / np.mean(rho_R))
        assert np.max(np.abs(rhosym_R - rho_R)) / np.mean(rho_R) < 1e-4
        print(t1 - t0, t3 - t2, len(C_ks), len(Csym_ks))
        print("DONE")
        print()

        mf2 = kpt_symm.KsymAdaptedPWKRKS(cell, kpts_sym)
        mf2 = smearing_(mf2, sigma=0.01, method='gauss')
        mf2.init_jk()
        mf2.init_pp()
        eref = mf.energy_elec(C_ks, mocc_ks)
        epred = mf2.energy_elec(Csym_ks, moccsym_ks)

        rho1 = mf.get_rho_for_xc("LDA", C_ks, mocc_ks)
        rho2 = mf2.get_rho_for_xc("LDA", Csym_ks, moccsym_ks)
        print(rho1.sum() * cell.vol, rho2.sum() * cell.vol)
        print(np.abs(rho1 - rho2).mean() * cell.vol)

        print(mf.scf_summary, mf2.scf_summary)
        print(eref, epred)
        assert_almost_equal(np.abs(rho1 - rho2).mean() * cell.vol, 0, 6)
        assert_almost_equal(epred, eref, 6)
    
    def test_get_wf(self):
        global mf, cell, kpts_sym
        C_ks = [coeff.copy() for coeff in mf.mo_coeff]
        Csym_ks = [C_ks[k_bz].copy() for k_bz in kpts_sym.ibz2bz]
        Cpred_ks = kpt_symm.get_C_from_C_ibz(Csym_ks, cell.mesh, kpts_sym)
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
            print(k)
            print(moe)
            print(np.abs(dot1).sum(), np.abs(np.diag(dot1))**2)
            print(np.abs(dot2).sum(), np.abs(np.diag(dot2))**2)
            print(np.abs(dot3).sum(), np.abs(np.diag(dot3))**2)
            print()
            k += 1
    
    def test_get_wf_real(self):
        global mf, cell, kpts_sym
        C_ks = [coeff.copy() for coeff in mf.mo_coeff]
        C_ks_R = [tools.ifft(C_k, mf.wf_mesh) for C_k in C_ks]
        Csym_ks_R = [C_ks_R[k_bz].copy() for k_bz in kpts_sym.ibz2bz]
        Cpred_ks_R = kpt_symm.get_C_from_C_ibz(Csym_ks_R, cell.mesh, kpts_sym,
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
            print(k)
            print(moe)
            print(np.abs(dot1).sum(), np.abs(np.diag(dot1))**2)
            print(np.abs(dot2).sum(), np.abs(np.diag(dot2))**2)
            print(np.abs(dot3).sum(), np.abs(np.diag(dot3))**2)
            assert_almost_equal(rdot1[:2, :2], rdot2[:2, :2], 6)
            assert_almost_equal(rdot1[:2], rdot2[:2], 4)
            assert_almost_equal(rdot1, rdot3, 6)
            print()
            k += 1

    def test_hf_symm(self):
        global cell, cell_sym

        import time

        kmesh = (3, 3, 3)
        # kmesh = (2, 2, 2)
        kpts = cell.make_kpts(kmesh)

        kpts_sym = cell_sym.make_kpts(
            kmesh,
            time_reversal_symmetry=True,
            space_group_symmetry=True,
        )
        mf = khf.PWKRHF(cell, kpts)
        t0 = time.monotonic()
        mf.kernel()
        t1 = time.monotonic()
        mf_sym = kpt_symm.KsymAdaptedPWKRHF(cell_sym, kpts_sym)
        t2 = time.monotonic()
        mf_sym.kernel()
        t3 = time.monotonic()
        print(mf.scf_summary)
        print(mf_sym.scf_summary)
        print(mf.e_tot, mf_sym.e_tot, mf.e_tot - mf_sym.e_tot)
        assert_almost_equal(mf_sym.e_tot, mf.e_tot, 5)
        print(t1 - t0, t3 - t2)


if __name__ == "__main__":
    unittest.main()

