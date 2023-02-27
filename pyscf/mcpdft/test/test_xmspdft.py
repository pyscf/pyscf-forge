
import numpy as np
from pyscf import gto, scf, mcpdft

import unittest

def get_lih (r):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    mc = mc.multi_state ([0.5,0.5], 'xms').run (conv_tol=1e-8)
    return mol, mf, mc

def setUpModule():
    global mol, mf, lih
    mol, mf, lih = get_lih (1.5)

def tearDownModule():
    global mol, mf, lih
    mol.stdout.close ()
    del mol, mf, lih

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_lih_diabats (self):
        # Reference values from OpenMolcas v22.02, tag 177-gc48a1862b
        # Ignoring the PDFT energies and final states because of grid nonsense
        e_mcscf_avg = np.dot (lih.e_mcscf, lih.weights)

        hcoup = abs (lih.heff_mcscf[1,0])
        ct_mcscf = abs (lih.si_mcscf[0,0])

        E_MCSCF_EXPECTED = -7.78902185
        E_STATES_EXPECTED = [-7.85862852, -7.6998051]
        HCOUP_EXPECTED = 0.019960046518608488

        self.assertAlmostEqual (e_mcscf_avg, E_MCSCF_EXPECTED, 7)
        self.assertListAlmostEqual(lih.e_states, E_STATES_EXPECTED, 7)
        self.assertAlmostEqual (hcoup, HCOUP_EXPECTED, 7)
        self.assertAlmostEqual (ct_mcscf, 0.9886771524332543, 7)

if __name__ == "__main__":
    print("Full Tests for XMS-PDFT")
    unittest.main()