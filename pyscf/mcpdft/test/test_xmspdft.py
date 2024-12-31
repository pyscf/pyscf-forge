
import numpy as np
from pyscf import gto, scf, dft, mcpdft
from pyscf.mcpdft import xmspdft

import unittest

def get_lih(r, weights=None):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    if weights is None: weights = [0.5,0.5]
    mc = mc.multi_state (weights, 'xms').run (conv_tol=1e-8)
    return mc

def setUpModule():
    global mc, mc_unequal, original_grids
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
    mc = get_lih(1.5)
    mc_unequal = get_lih(1.5, weights=[0.9, 0.1])

def tearDownModule():
    global mc, mc_unequal, original_grids
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    mc.mol.stdout.close()
    mc_unequal.mol.stdout.close()
    del mc, mc_unequal, original_grids

class KnownValues(unittest.TestCase):

    def assertListAlmostEqual(self, first_list, second_list, expected):
        self.assertTrue(len(first_list) == len(second_list))
        for first, second in zip(first_list, second_list):
            self.assertAlmostEqual(first, second, expected)

    # Reference values from
    # - PySCF        hash 8ae2bb2eefcd342c52639097517b1eda7ca5d1cd
    # - PySCF-forge  hash 5b8ab86a31917ca1a6b414f7a590c4046b9a8994
    #
    # Implementation with those hashes verified with OpenMolcas
    # (tag 462-g00b34a15f)
    def test_lih_diabats(self):
        e_mcscf_avg = np.dot(mc.e_mcscf, mc.weights)
        hcoup = abs(mc.heff_mcscf[1,0])
        ct_mcscf = abs(mc.si_mcscf[0,0])

        HCOUP_EXPECTED = 0.01996004651860848
        CT_MCSCF_EXPECTED = 0.9886771524332543
        E_MCSCF_AVG_EXPECTED = -7.789021830554006

        self.assertAlmostEqual (e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 9)
        self.assertAlmostEqual (hcoup, HCOUP_EXPECTED, 9)
        self.assertAlmostEqual (ct_mcscf, CT_MCSCF_EXPECTED, 9)

    def test_lih_safock(self):
        safock = xmspdft.make_fock_mcscf(mc, ci=mc.get_ci_adiabats(uci="MCSCF"))
        EXPECTED_SA_FOCK_DIAG = [-4.207598506457942, -3.88169762424571]
        EXPECTED_SA_FOCK_OFFDIAG = 0.05063053788053997

        self.assertListAlmostEqual(safock.diagonal(), EXPECTED_SA_FOCK_DIAG, 9)
        self.assertAlmostEqual(abs(safock[0,1]),EXPECTED_SA_FOCK_OFFDIAG, 9)

    def test_lih_safock_unequal(self):
        safock = xmspdft.make_fock_mcscf(mc, ci=mc.get_ci_adiabats(uci="MCSCF"), weights=[1,0])
        EXPECTED_SA_FOCK_DIAG = [-4.194714957289011, -3.8317682977263754]
        EXPECTED_SA_FOCK_OFFDIAG = 0.006987847963283834

        self.assertListAlmostEqual(safock.diagonal(), EXPECTED_SA_FOCK_DIAG, 9)
        self.assertAlmostEqual(abs(safock[0, 1]), EXPECTED_SA_FOCK_OFFDIAG, 9)

    def test_lih_adiabats(self):
        E_STATES_EXPECTED = [-7.858628517291297, -7.69980510010583]
        self.assertListAlmostEqual(mc.e_states, E_STATES_EXPECTED, 9)

    def test_lih_unequal(self):
        e_mcscf_avg = np.dot(mc_unequal.e_mcscf, mc_unequal.weights)
        hcoup = abs(mc_unequal.heff_mcscf[1, 0])
        ct_mcscf = abs(mc_unequal.si_mcscf[0, 0])

        HCOUP_EXPECTED = 0.006844540922301437
        CT_MCSCF_EXPECTED = 0.9990626861718451
        E_MCSCF_AVG_EXPECTED = -7.8479729055935685
        E_STATES_EXPECTED = [-7.869843475893866, -7.696820527732333]

        self.assertAlmostEqual(e_mcscf_avg, E_MCSCF_AVG_EXPECTED, 9)
        self.assertAlmostEqual(hcoup, HCOUP_EXPECTED, 9)
        self.assertAlmostEqual(ct_mcscf, CT_MCSCF_EXPECTED, 9)
        self.assertListAlmostEqual(mc_unequal.e_states, E_STATES_EXPECTED, 9)


if __name__ == "__main__":
    print("Full Tests for XMS-PDFT")
    unittest.main()
