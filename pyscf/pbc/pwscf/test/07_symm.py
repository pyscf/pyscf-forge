import unittest
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.pwscf import khf, krks, jk
import numpy as np
import time


class TestSymmetry(unittest.TestCase):
    def test_get_rho(self):
        #cell = pbcgto.Cell(
        #    atom = "He 0.5 0.5 0.5",
        #    a = np.eye(3) * 3,
        #    basis="gth-szv",
        #    ke_cutoff=50,
        #    pseudo="gth-pade",
        #)
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
        cell.mesh = [42, 42, 42]
        cell.build()
        kmesh = (4, 4, 4)
        kpts = cell.make_kpts(kmesh)

        cell_sym = cell.copy()
        cell_sym.space_group_symmetry=True
        cell_sym.symmorphic=True
        cell_sym.build()
        kpts_sym = cell_sym.make_kpts(
            kmesh,
            time_reversal_symmetry=True,
            space_group_symmetry=True,
        )

        mf = krks.PWKRKS(cell, kpts, xc="LDA")
        mf.kernel()

        C_ks = mf.mo_coeff
        mocc_ks = mf.mo_occ
        
        t0 = time.monotonic()
        rho_R = jk.get_rho_R(C_ks, mocc_ks, cell.mesh)
        t1 = time.monotonic()
        Csym_ks = [C_ks[k_bz] for k_bz in kpts_sym.ibz2bz]
        print(len(Csym_ks))
        moccsym_ks = [mocc_ks[k_bz] for k_bz in kpts_sym.ibz2bz]
        t2 = time.monotonic()
        rhosym_R = jk.get_rho_R_ksym(Csym_ks, moccsym_ks, cell.mesh, kpts_sym)
        t3 = time.monotonic()
        print(rho_R.sum())
        print(rhosym_R.sum())
        print(np.linalg.norm(rhosym_R - rho_R))
        print(np.abs(rhosym_R - rho_R).sum() / rho_R.sum())
        print(np.max(np.abs(rhosym_R - rho_R)) / np.mean(rho_R))
        print(t1 - t0, t3 - t2, len(C_ks), len(Csym_ks))


if __name__ == "__main__":
    unittest.main()

