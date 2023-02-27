import numpy as np
from pyscf import gto, scf 
from pyscf import mcpdft
from pyscf.mcpdft import lpdft
import unittest


def get_lih (r, n_states=2, functional='ftLDA,VWN3', basis='sto3g'):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis=basis,
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    if n_states == 2:
        mc = mcpdft.CASSCF (mf, functional, 2, 2, grids_level=1)

    else:
        mc = mcpdft.CASSCF(mf, functional, 5, 2, grids_level=1)

    mc.fix_spin_(ss=0)
    weights = [1.0/float(n_states), ] * n_states
    
    mc = mc.multi_state(weights, "lin")
    mc = mc.run()
    return mol, mf, mc

def setUpModule():
    global mol, mf, mc, mc_4
    mol, mf, mc = get_lih(1.5)
    mol, mf, mc_4 = get_lih(1.5, n_states=4, basis="6-31G")

def tearDownModule():
    global mol, mf, mc, mc_4
    mol.stdout.close()
    del mol, mf, mc, mc_4

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_lih_2_states_adiabat(self):
        e_mcscf_avg = np.dot (mc.e_mcscf, mc.weights)
        hcoup = abs(mc.lpdft_ham[1,0])
        hdiag = mc.get_lpdft_diag()

        e_states = mc.e_states

        # Reference values from OpenMolcas v22.02, tag 177-gc48a1862b
        E_MCSCF_AVG_EXPECTED = -7.78902185
        
        # Below reference values from 
        #   - PySCF commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #   - mrh   commit c5fc02f1972c1c8793061f20ed6989e73638fc5e
        HCOUP_EXPECTED = 0.016636807982732867 
        HDIAG_EXPECTED = [-7.878489930907849, -7.729844823595374] 

        E_STATES_EXPECTED = [-7.88032921, -7.72800554]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 7)

    def test_lih_4_states_adiabat(self):
        e_mcscf_avg = np.dot(mc_4.e_mcscf, mc_4.weights)
        hdiag = mc_4.get_lpdft_diag()
        hcoup = mc_4.lpdft_ham[np.triu_indices(4, k=1)]
        e_states = mc_4.e_states

        # References values from
        #     - PySCF       commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #     - PySCF-forge commit 00183c314ebbf541f8461e7b7e5ee9e346fd6ff5
        E_MCSCF_AVG_EXPECTED = -7.881123865044279
        HDIAG_EXPECTED = [-7.997842598062071, -7.84720560226191, -7.80476518947314, -7.804765211915506]
        HCOUP_EXPECTED = [-0.01479405057250327,0,0,0,0,0]
        E_STATES_EXPECTED = [-7.999281764601187, -7.8457664246019005, -7.804765192541955, -7.804765192508891]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(hcoup, HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 7)

    def test_lih_hybrid_adiabat(self):
        e_states, _ = mc.hybrid_kernel(0.25)

        E_TPBE0_EXPECTED = [-7.874005199186947, -7.726756781565399]

        self.assertListAlmostEqual(e_states, E_TPBE0_EXPECTED, 7)

        
if __name__ == "__main__":
    print("Full Tests for Linearized-PDFT")
    unittest.main()
