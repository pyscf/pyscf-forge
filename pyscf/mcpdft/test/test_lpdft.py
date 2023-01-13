import numpy as np
from pyscf import gto, scf 
from pyscf import mcpdft
from pyscf.mcpdft import lpdft
import unittest


def get_lih (r, functional='ftLDA,VWN3', quasi=False):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, functional, 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    n_states = 2
    weights = [1.0/float(n_states), ] * n_states
    mc = mc.state_average(weights)
    mc = lpdft.lmspdft(mc, quasi=quasi).run()
    return mol, mf, mc

def setUpModule():
    global mol, mf, mc, mc_quasi
    mol, mf, mc = get_lih(1.5)
    _, _, mc_quasi = get_lih(1.5, quasi=True)

def tearDownModule():
    global mol, mf, mc, mc_quasi
    mol.stdout.close()
    mc_quasi.mol.stdout.close()
    del mol, mf, mc, mc_quasi

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_lih_adiabat(self):
        e_mcscf_avg = np.dot (mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_lin[1,0])
        hdiag = [mc.heff_lin[0,0], mc.heff_lin[1,1]] 

        e_states = mc.e_states

        # Reference values from OpenMolcas v22.02, tag 177-gc48a1862b
        E_MCSCF_AVG_EXPECTED = -7.78902185
        
        # Below reference values from 
        #   - PySCF commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #   - mrh   commit c5fc02f1972c1c8793061f20ed6989e73638fc5e
        HCOUP_EXPECTED = 0.016636807982732867 
        HDIAG_EXPECTED = [-7.878489930907849, -7.729844823595374] 

        # Always check LMS-PDFT and QLPDFT (thought this may eventually be deprecated!)
        E_STATES_EXPECTED = [-7.88032921, -7.72800554]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_STATES_EXPECTED, 7)

    def test_lih_quasi_adiabat(self):
        e_mcscf_avg = np.dot (mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_lin[1,0])
        hdiag = [mc.heff_lin[0,0], mc.heff_lin[1,1]] 

        e_states = mc_quasi.e_states
        
        # Reference values from OpenMolcas v22.02, tag 177-gc48a1862b
        E_MCSCF_AVG_EXPECTED = -7.78902185
        
        # Below reference values from 
        #   - PySCF commit 71fc2a41e697fec76f7f9a5d4d10fd2f2476302c
        #   - mrh   commit c5fc02f1972c1c8793061f20ed6989e73638fc5e
        HCOUP_EXPECTED = 0.016636807982732867 
        HDIAG_EXPECTED = [-7.878489930907849, -7.729844823595374] 
        E_QLPDFT_EXPECTED = [-7.86311503, -7.7070019]
        
        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 7)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 7)
        self.assertListAlmostEqual(hdiag, HDIAG_EXPECTED, 7)
        self.assertListAlmostEqual(e_states, E_QLPDFT_EXPECTED, 7)

    
    def test_lih_hybrid_adiabat(self):
        e_states, _ = mc.hybrid_kernel(0.25)

        E_TPBE0_EXPECTED = [-7.874005199186947, -7.726756781565399]

        self.assertListAlmostEqual(e_states, E_TPBE0_EXPECTED, 7)


if __name__ == "__main__":
    print("Full Tests for Linear-PDFT")
    unittest.main()
