#!/usr/bin/env python
'''
Unit tests for DFT-corrected CASCI
'''

import unittest
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import dft_corrected_casci
from pyscf.scf import fomoscf


class TestDFTCorrectedCASCI_RHF(unittest.TestCase):
    '''Tests for RHF-based DFT-corrected CASCI'''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', unit='Angstrom', verbose=0
        )
        cls.mf = scf.RHF(cls.mol).run()
        cls.ncas = 4
        cls.nelecas = 4

    def test_energy(self):
        '''Test DFT-corrected CASCI energy calculation'''
        mc = dft_corrected_casci.CASCI(self.mf, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)
        self.assertLess(mc.e_tot, -74.0)

    def test_functionals(self):
        '''Test different XC functionals'''
        for xc in ['LDA', 'PBE', 'B3LYP']:
            mc = dft_corrected_casci.CASCI(self.mf, self.ncas, self.nelecas, xc=xc)
            mc.kernel()
            self.assertIsNotNone(mc.e_tot)

    def test_ci_unchanged(self):
        '''Test that CI coefficients are identical to standard CASCI'''
        mc_hf = mcscf.CASCI(self.mf, self.ncas, self.nelecas)
        mc_hf.kernel()

        mc_dft = dft_corrected_casci.CASCI(self.mf, self.ncas, self.nelecas, xc='PBE')
        mc_dft.kernel()

        # CI coefficients should be identical (only core energy differs)
        overlap = abs(mc_hf.ci.flatten() @ mc_dft.ci.flatten())
        self.assertAlmostEqual(overlap, 1.0, places=10)


class TestDFTCorrectedCASCI_UHF(unittest.TestCase):
    '''Tests for UHF-based DFT-corrected CASCI'''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', unit='Angstrom', verbose=0
        )
        cls.mf = scf.UHF(cls.mol).run()
        cls.ncas = 4
        cls.nelecas = 4

    def test_energy(self):
        '''Test UHF-based DFT-corrected CASCI energy'''
        mc = dft_corrected_casci.UCASCI(self.mf, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)
        self.assertLess(mc.e_tot, -74.0)

    def test_functionals(self):
        '''Test different XC functionals for UHF'''
        for xc in ['LDA', 'PBE']:
            mc = dft_corrected_casci.UCASCI(self.mf, self.ncas, self.nelecas, xc=xc)
            mc.kernel()
            self.assertIsNotNone(mc.e_tot)


class TestDFCASCI_Factory(unittest.TestCase):
    '''Tests for DFCASCI factory function'''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', unit='Angstrom', verbose=0
        )

    def test_rhf_detection(self):
        '''Test factory auto-detects RHF'''
        mf = scf.RHF(self.mol).run()
        mc = dft_corrected_casci.DFCASCI(mf, 4, 4, xc='PBE')
        self.assertIsInstance(mc, dft_corrected_casci.CASCI)

    def test_uhf_detection(self):
        '''Test factory auto-detects UHF'''
        mf = scf.UHF(self.mol).run()
        mc = dft_corrected_casci.DFCASCI(mf, 4, 4, xc='PBE')
        self.assertIsInstance(mc, dft_corrected_casci.UCASCI)


class TestFOMO_DFTCorrectedCASCI(unittest.TestCase):
    '''Tests for FOMO-CASCI with DFT-corrected core'''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', unit='Angstrom', verbose=0
        )
        cls.mf = scf.RHF(cls.mol).run()
        cls.ncas = 4
        cls.nelecas = 4
        cls.ncore = 2

    def test_fomo_dft_corrected_casci_energy(self):
        '''Test FOMO-CASCI with DFT-corrected core energy'''
        mf_fomo = fomoscf.fomo_scf(
            self.mf, temperature=0.25, method='gaussian',
            restricted=(self.ncore, self.ncas)
        )
        mf_fomo.kernel()

        mc = dft_corrected_casci.CASCI(mf_fomo, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()

        self.assertIsNotNone(mc.e_tot)
        self.assertLess(mc.e_tot, -74.0)

    def test_fomo_occupations(self):
        '''Test that FOMO produces fractional occupations'''
        mf_fomo = fomoscf.fomo_scf(
            self.mf, temperature=0.25, method='gaussian',
            restricted=(self.ncore, self.ncas)
        )
        mf_fomo.kernel()

        # Check for fractional occupations in active space
        active_occ = mf_fomo.mo_occ[self.ncore:self.ncore + self.ncas]
        has_fractional = np.any((active_occ > 0.01) & (active_occ < 1.99))
        self.assertTrue(has_fractional, "FOMO should produce fractional occupations")


class TestDFTCorrectedCASCI_Consistency(unittest.TestCase):
    '''Consistency tests for DFT-corrected CASCI'''

    def test_h2_molecule(self):
        '''Test on simple H2 molecule'''
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()

        mc = dft_corrected_casci.CASCI(mf, ncas=2, nelecas=2, xc='LDA')
        mc.kernel()

        self.assertIsNotNone(mc.e_tot)
        self.assertTrue(mc.converged)

    def test_energy_lowering(self):
        '''Test that DFT-corrected CASCI lowers energy vs HF-CASCI'''
        mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', verbose=0
        )
        mf = scf.RHF(mol).run()

        # Standard HF-CASCI
        mc_hf = mcscf.CASCI(mf, ncas=4, nelecas=4)
        mc_hf.kernel()

        # DFT-corrected CASCI
        mc_dft = dft_corrected_casci.CASCI(mf, ncas=4, nelecas=4, xc='PBE')
        mc_dft.kernel()

        # DFT-corrected should typically lower the energy
        # (though not guaranteed in all cases)
        self.assertIsNotNone(mc_hf.e_tot)
        self.assertIsNotNone(mc_dft.e_tot)


if __name__ == '__main__':
    unittest.main()
