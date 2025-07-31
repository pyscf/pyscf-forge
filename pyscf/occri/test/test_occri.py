#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf
from pyscf.pbc.df import fft
from pyscf.occri import OCCRI

def setUpModule():
    global cell, cell_kpts, kpts, mf_rhf, mf_uhf, mf_krhf, mf_kuhf
    
    # Setup basic cell for gamma point tests
    cell = pgto.Cell()
    cell.atom = '''
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.a = numpy.eye(3) * 3.5607
    cell.mesh = [16] * 3
    cell.build()
    
    # Setup cell for k-point tests 
    cell_kpts = pgto.Cell()
    cell_kpts.atom = '''
    C 0.000000 0.000000 0.000000
    C 0.890186 0.890186 0.890186
    '''
    cell_kpts.basis = 'gth-szv'
    cell_kpts.pseudo = 'gth-pbe'
    cell_kpts.a = numpy.eye(3) * 3.5607
    cell_kpts.mesh = [25] * 3  # Larger mesh for k-point tests
    # Note that in general, the accuracy of occRI requires adequate cell mesh.
    # For sparse mesh, FFTDF and OCCRI converge to different states.
    cell_kpts.build()
    
    # Setup proper k-point mesh
    kpts = cell_kpts.make_kpts([2,2,2])
    
    # Setup mean-field objects
    mf_rhf = scf.RHF(cell)
    mf_uhf = scf.UHF(cell)
    mf_krhf = scf.KRHF(cell_kpts, kpts)
    mf_kuhf = scf.KUHF(cell_kpts, kpts)

def tearDownModule():
    global cell, cell_kpts, kpts, mf_rhf, mf_uhf, mf_krhf, mf_kuhf
    del cell, cell_kpts, kpts, mf_rhf, mf_uhf, mf_krhf, mf_kuhf

class TestOCCRI(unittest.TestCase):
    
    def test_rhf_gamma_point(self):
        """Test RHF at gamma point against FFTDF reference"""
        # Reference calculation with FFTDF
        mf_ref = scf.RHF(cell)
        mf_ref.kernel()
        e_ref = mf_ref.e_tot
        
        # OCCRI calculation
        mf_occri = scf.RHF(cell)
        mf_occri.with_df = OCCRI(mf_occri)
        mf_occri.kernel()
        e_occri = mf_occri.e_tot
        
        # Check convergence and energy agreement
        self.assertTrue(mf_occri.converged)
        self.assertAlmostEqual(e_ref, e_occri, places=8)
        
    def test_uhf_gamma_point(self):
        """Test UHF at gamma point against FFTDF reference"""
        # Reference calculation with FFTDF
        mf_ref = scf.UHF(cell)  
        mf_ref.kernel()
        e_ref = mf_ref.e_tot
        
        # OCCRI calculation
        mf_occri = scf.UHF(cell)
        mf_occri.with_df = OCCRI(mf_occri)
        mf_occri.kernel()
        e_occri = mf_occri.e_tot
        
        # Check convergence and energy agreement
        self.assertTrue(mf_occri.converged)
        self.assertAlmostEqual(e_ref, e_occri, places=8)
        
    def test_krhf(self):
        """Test KRHF with k-points against FFTDF reference"""
        # Reference calculation with FFTDF
        mf_ref = scf.KRHF(cell_kpts, kpts)
        mf_ref.kernel()
        e_ref = mf_ref.e_tot
        
        # OCCRI calculation - use same initial guess as reference
        mf_occri = scf.KRHF(cell_kpts, kpts)
        mf_occri.with_df = OCCRI(mf_occri, kmesh=[2,2,2])
        # Start from reference MO coefficients to ensure same state
        mf_occri.kernel(dm0=mf_ref.make_rdm1())
        e_occri = mf_occri.e_tot
        
        # Check convergence and energy agreement
        self.assertTrue(mf_ref.converged)
        self.assertTrue(mf_occri.converged)
        self.assertAlmostEqual(e_ref, e_occri, places=8)  # High accuracy with proper mesh size
        
    def test_kuhf(self):
        """Test KUHF with k-points against FFTDF reference"""
        # Reference calculation with FFTDF
        mf_ref = scf.KUHF(cell_kpts, kpts)
        mf_ref.kernel()
        e_ref = mf_ref.e_tot
        
        # OCCRI calculation
        mf_occri = scf.KUHF(cell_kpts, kpts)
        mf_occri.with_df = OCCRI(mf_occri, kmesh=[2,2,2])
        mf_occri.kernel()
        e_occri = mf_occri.e_tot
        
        # Check convergence and energy agreement
        self.assertTrue(mf_occri.converged)  
        self.assertAlmostEqual(e_ref, e_occri, places=8)  # High accuracy with proper mesh size
        
    def test_get_jk_rhf(self):
        """Test get_jk method for RHF"""
        mf_ref = scf.RHF(cell)
        dm = mf_ref.get_init_guess()
        
        # Reference FFTDF calculation
        df_ref = fft.FFTDF(cell)
        _, vk_ref = df_ref.get_jk(dm, exxdiv=None, with_k=True)
        
        # OCCRI calculation
        df_occri = OCCRI(mf_ref)
        _, vk_occri = df_occri.get_jk(dm, exxdiv=None, with_k=True)
        
        # Check K matrix agreement  
        self.assertTrue(numpy.allclose(vk_ref, vk_occri, atol=1.e-10, rtol=1.e-10))
        
    def test_get_jk_uhf(self):
        """Test get_jk method for UHF with spin"""
        mf_ref = scf.UHF(cell)
        dm = mf_ref.get_init_guess()
        
        # Reference FFTDF calculation
        df_ref = fft.FFTDF(cell)
        _, vk_ref = df_ref.get_jk(dm, exxdiv=None, with_k=True)
        
        # OCCRI calculation
        df_occri = OCCRI(mf_ref)
        _, vk_occri = df_occri.get_jk(dm, exxdiv=None, with_k=True)
        
        # Check K matrices agreement
        self.assertTrue(numpy.allclose(vk_ref, vk_occri, atol=1.e-10, rtol=1.e-10))
        
    def test_get_jk_kpts(self):
        """Test get_jk method with k-points"""
        mf_ref = scf.KRHF(cell_kpts, kpts)
        dms = mf_ref.get_init_guess().astype(numpy.complex128)
        
        # Reference FFTDF calculation
        df_ref = fft.FFTDF(cell_kpts)
        _, vk_ref = df_ref.get_jk(dms, kpts=kpts, exxdiv=None, with_k=True)
        
        # OCCRI calculation
        df_occri = OCCRI(mf_ref, kmesh=[2,2,2])
        _, vk_occri = df_occri.get_jk(dms, kpts=kpts, exxdiv=None, with_k=True)
        
        # Check data types
        self.assertTrue(vk_occri.dtype == numpy.complex128)
        
        # Check K matrices agreement
        e_ref = sum(numpy.einsum("ij,ji->", vi, di).real * 0.5 for vi, di in zip(vk_ref, dms))
        e_occri = sum(numpy.einsum("ij,ji->", vi, di).real * 0.5 for vi, di in zip(vk_occri, dms))
        self.assertTrue(numpy.allclose(e_ref, e_occri, atol=1.e-10, rtol=1.e-10))
        
    def test_ewald_correction(self):
        """Test Ewald correction for exchange divergence"""
        mf_ref = scf.RHF(cell)
        dm = mf_ref.get_init_guess()
        
        # OCCRI with and without Ewald correction
        df_occri = OCCRI(mf_ref)
        _, vk1 = df_occri.get_jk(dm, exxdiv=None, with_k=True)
        _, vk2 = df_occri.get_jk(dm, exxdiv='ewald', with_k=True)
        
        # K matrices should be different due to Ewald correction
        self.assertFalse(numpy.allclose(vk1, vk2, atol=1.e-10))
        
        
    def test_natural_orbitals(self):
        """Test natural orbital construction"""
        from pyscf.occri.occri_k import make_natural_orbitals
        
        mf = scf.RHF(cell)
        mf.kernel()
        nao = cell.nao
        dm = mf.make_rdm1().reshape(-1,nao,nao)
        
        # Construct natural orbitals
        mo_coeff, mo_occ = make_natural_orbitals(cell, dm)
        
        # Check occupation numbers are sorted in descending order
        self.assertTrue(numpy.all(mo_occ[:-1] >= mo_occ[1:]))
        
        # Check that natural orbitals give reasonable total occupation
        # (Close to expected number of electrons)
        s1e = mf.get_ovlp(cell)
        ne = numpy.einsum("xij,ji->x", dm, s1e).real
        nelec = cell.nelectron
        self.assertAlmostEqual(ne, nelec, delta=0.01)
        self.assertAlmostEqual(numpy.sum(mo_occ), nelec, delta=0.01)
        
    def test_natural_orbitals_kpts(self):
        """Test natural orbital construction with k-points"""
        from pyscf.occri.occri_k_kpts import make_natural_orbitals_kpts
        
        mf = scf.KRHF(cell_kpts, kpts)
        mf.kernel()
        dm = mf.make_rdm1().reshape(-1, kpts.shape[0], cell_kpts.nao, cell_kpts.nao)
        
        # Construct natural orbitals
        mo_coeff, mo_occ = make_natural_orbitals_kpts(cell_kpts, dm, kpts)
        
        # Check dimensions
        nao = cell_kpts.nao
        nk = len(kpts)
        self.assertEqual(mo_coeff.shape, (1, nk, nao, nao))  # 1 for RHF
        self.assertEqual(mo_occ.shape, (1, nk, nao))
        
        # Check occupation numbers are sorted in descending order for each k-point
        for k in range(nk):
            occ_k = mo_occ[0, k]
            self.assertTrue(numpy.all(occ_k[:-1] >= occ_k[1:]))
            
    def test_disable_c_extension(self):
        """Test fallback to Python implementation"""
        # Test with C extension disabled
        mf_python = scf.RHF(cell)
        mf_python.with_df = OCCRI(mf_python, disable_c=True)
        mf_python.kernel()
        e_python = mf_python.e_tot
        
        # Test with C extension enabled (if available)
        mf_c = scf.RHF(cell)
        mf_c.with_df = OCCRI(mf_c, disable_c=False)
        mf_c.kernel()
        e_c = mf_c.e_tot
        
        # Results should be the same
        self.assertAlmostEqual(e_python, e_c, places=8)
        
    def test_memory_usage(self):
        """Test memory usage estimation"""
        from pyscf.occri import log_mem
        
        # Create OCCRI object
        mf = scf.RHF(cell)
        df = OCCRI(mf)
        
        # This should not raise an exception
        log_mem(df)
        
    def test_build_full_exchange(self):
        """Test build_full_exchange function"""
        from pyscf.occri import build_full_exchange
        
        # Setup test matrices
        nao = 10
        nocc = 5
        numpy.random.seed(42)
        
        S = numpy.random.random((nao, nao))
        S = S + S.T  # Make symmetric
        Kao = numpy.random.random((nao, nocc)) + 1j * numpy.random.random((nao, nocc))
        mo_coeff = Kao.T 
        
        # Build exchange matrix
        K = build_full_exchange(S, Kao, mo_coeff)
        
        # Check dimensions
        self.assertEqual(K.shape, (nao, nao))
        
        # Exchange matrix should be symmetric
        self.assertTrue(numpy.allclose(K, K.T.conj(), atol=1.e-10))
        
    def test_consistency_rhf_uhf(self):
        """Test that RHF and UHF give same results for closed shell"""
        # RHF calculation
        mf_rhf = scf.RHF(cell)
        mf_rhf.with_df = OCCRI(mf_rhf)
        mf_rhf.kernel()
        e_rhf = mf_rhf.e_tot
        
        # UHF calculation with same density
        mf_uhf = scf.UHF(cell)
        mf_uhf.with_df = OCCRI(mf_uhf)
        # Initialize with RHF density
        dm_rhf = mf_rhf.make_rdm1()
        mf_uhf.init_guess = 'hcore'
        mf_uhf.kernel(dm0=[dm_rhf/2, dm_rhf/2])
        e_uhf = mf_uhf.e_tot
        
        # Should give similar energies for closed shell
        self.assertAlmostEqual(e_rhf, e_uhf, places=8)
        
    def test_small_system(self):
        """Test on very small system (H2)"""
        h2_cell = pgto.Cell()
        h2_cell.atom = '''
        H 0. 0. 0.
        H 1. 0. 0.
        '''
        h2_cell.basis = 'sto-3g'
        h2_cell.a = numpy.eye(3) * 4.0
        h2_cell.mesh = [12] * 3
        h2_cell.build()
        
        # Reference calculation
        mf_ref = scf.RHF(h2_cell)
        mf_ref.kernel()
        e_ref = mf_ref.e_tot
        
        # OCCRI calculation
        mf_occri = scf.RHF(h2_cell)
        mf_occri.with_df = OCCRI(mf_occri)
        mf_occri.kernel()
        e_occri = mf_occri.e_tot
        
        # Check agreement
        self.assertTrue(mf_occri.converged)
        self.assertAlmostEqual(e_ref, e_occri, places=8)

if __name__ == '__main__':
    print("Running OCCRI test suite...")
    unittest.main()