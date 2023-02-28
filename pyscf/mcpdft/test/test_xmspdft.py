
import numpy as np
from math import cos, sin, pi
from pyscf import gto, scf, mcpdft

import unittest

def get_lih (r):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    mc = mc.multi_state ([0.5,0.5], 'xms').run (conv_tol=1e-8)
    return mol, mf, mc

def get_h2(r):
    mol = gto.M(atom='H 0 0 0\n H {} 0 0'.format(r), basis='sto3g', output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf, 'tPBE', 2, 2, grids_level=6)
    mc.fix_spin_(ss=0, shift=1)
    mc = mc.multi_state([0.5, 0.5], 'xms').run(conv_tol=1e-8)
    return mol, mf, mc


def setUpModule():
    global mol, mf, lih, h2
    mol, mf, lih = get_lih (1.5)
    _, _, h2 = get_h2(0.8)

def tearDownModule():
    global mol, mf, lih, h2
    mol.stdout.close ()
    del mol, mf, lih, h2

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    def test_lih_adiabats(self):
        e_mcscf_avg = np.dot (lih.e_mcscf, lih.weights)

        hcoup = abs (lih.heff_mcscf[1,0])

        E_MCSCF_EXPECTED = -7.78902185
        E_STATES_EXPECTED = [-7.9108699492171475, -7.7549375652733685]
        HCOUP_EXPECTED = 0.019960046518608488

        self.assertAlmostEqual (e_mcscf_avg, E_MCSCF_EXPECTED, 7)
        self.assertListAlmostEqual(lih.e_states, E_STATES_EXPECTED, 7)
        self.assertAlmostEqual (hcoup, HCOUP_EXPECTED, 7)

    def test_h2_adiabats(self):
        e_mcscf_avg = np.dot(h2.e_mcscf, h2.weights)

        # Reference Values from OpenMolcas v22.10
        # tag 462-g00b34a15f
        E_MCSCF_EXPECTED = -0.68103595
        E_STATES_EXPECTED = [-1.15358469, -0.48247557]
        HCOUP_EXPECTED = 0

        hcoup = abs(h2.heff_mcscf[1, 0])

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_EXPECTED, 5)
        self.assertListAlmostEqual(h2.e_states, E_STATES_EXPECTED, 5)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 5)


if __name__ == "__main__":
    print("Full Tests for XMS-PDFT")
    unittest.main()