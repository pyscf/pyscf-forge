
import numpy as np
from pyscf import gto, scf, fci
from pyscf import mcpdft
import unittest


def get_h2(r, n_states=2, functional='ftLDA,VWN3', basis='sto3g'):
    mol = gto.M(atom='H 0 0 0\nH {} 0 0'.format(r), basis=basis,
                output='/dev/null', verbose=0)
    mf = scf.RHF(mol).run()
    if n_states == 2:
        mc = mcpdft.CASSCF(mf, functional, 2, 2, grids_level=1)

    else:
        mc = mcpdft.CASSCF(mf, functional, 5, 2, grids_level=1)

    mc.fix_spin_(ss=0)
    weights = [1.0 / float(n_states), ] * n_states

    mc = mc.multi_state(weights, "lin")
    mc = mc.run()
    return mc

def setUpModule():
    global h2
    h2 = get_h2(1.5)

def tearDownModule():
    global h2
    h2.mol.stdout.close()
    del h2

class KnownValues(unittest.TestCase):
    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)


    def test_h2(self):
        print(h2.get_pdft_feff())

if __name__ == "__main__":
    print("Full Tests for pdft_feff")
    unittest.main()