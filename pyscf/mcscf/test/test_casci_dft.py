#!/usr/bin/env python
"""
Unit tests for DFT-corrected CASCI (casci_dft)
"""

import unittest
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import casci_dft, addons


class TestCASCI_DFT_RHF(unittest.TestCase):
    """Tests for RHF-based DFT-CASCI"""
    
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
        mc = casci_dft.CASCI(self.mf, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)
        self.assertLess(mc.e_tot, -74.0)
    
    def test_functionals(self):
        for xc in ['LDA', 'PBE', 'B3LYP']:
            mc = casci_dft.CASCI(self.mf, self.ncas, self.nelecas, xc=xc)
            mc.kernel()
            self.assertIsNotNone(mc.e_tot)
    
    def test_ci_unchanged(self):
        mc_hf = mcscf.CASCI(self.mf, self.ncas, self.nelecas)
        mc_hf.kernel()
        mc_dft = casci_dft.CASCI(self.mf, self.ncas, self.nelecas, xc='PBE')
        mc_dft.kernel()
        # CI coefficients should be identical
        self.assertAlmostEqual(abs(mc_hf.ci.flatten() @ mc_dft.ci.flatten()), 1.0, places=10)


class TestCASCI_DFT_UHF(unittest.TestCase):
    """Tests for UHF-based DFT-CASCI"""
    
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
        mc = casci_dft.UCASCI(self.mf, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)
        self.assertLess(mc.e_tot, -74.0)
    
    def test_functionals(self):
        for xc in ['LDA', 'PBE']:
            mc = casci_dft.UCASCI(self.mf, self.ncas, self.nelecas, xc=xc)
            mc.kernel()
            self.assertIsNotNone(mc.e_tot)


class TestDFCASCI_Factory(unittest.TestCase):
    """Tests for DFCASCI factory function"""
    
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', unit='Angstrom', verbose=0
        )
    
    def test_rhf_detection(self):
        mf = scf.RHF(self.mol).run()
        mc = casci_dft.DFCASCI(mf, 4, 4, xc='PBE')
        self.assertIsInstance(mc, casci_dft.CASCI)
    
    def test_uhf_detection(self):
        mf = scf.UHF(self.mol).run()
        mc = casci_dft.DFCASCI(mf, 4, 4, xc='PBE')
        self.assertIsInstance(mc, casci_dft.UCASCI)


class TestFOMO_CASCI(unittest.TestCase):
    """Tests for FOMO-CASCI"""
    
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
    
    def test_fomo_casci_energy(self):
        mf_fomo = addons.fomo_scf(
            self.mf, temperature=0.25, method='gaussian',
            restricted=(self.ncore, self.ncas)
        )
        mf_fomo.kernel()
        mc = casci_dft.CASCI(mf_fomo, self.ncas, self.nelecas, xc='LDA')
        mc.kernel()
        self.assertIsNotNone(mc.e_tot)


if __name__ == '__main__':
    unittest.main()
