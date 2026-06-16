# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
import numpy as np
# import cupy as cp
from pyscf import gto, lib
from pyscf.dft import rks
import pyscf.tdscf.ris as ris

PLACES = 3


def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    nov = nocc * nvir
    a = a.reshape(nov, nov)
    b = b.reshape(nov, nov)
    h = np.block([[a        , b       ],
                     [-b.conj(),-a.conj()]])
    e, xy = np.linalg.eig(np.asarray(h))
    sorted_indices = np.argsort(e)

    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]

    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # methanol
        atom = '''
        C         -4.89126        3.29770        0.00029
        H         -5.28213        3.05494       -1.01161
        O         -3.49307        3.28429       -0.00328
        H         -5.28213        2.58374        0.75736
        H         -5.23998        4.31540        0.27138
        H         -3.22959        2.35981       -0.24953
        '''
        mol = gto.M(atom=atom, basis='def2-svp',
                    output = '/dev/null',  # Suppress excessive log output
                    verbose=3)
        cls.mol = mol

        # Initialize DFT calculations with different functionals
        cls.mf_pbe = rks.RKS(mol, xc='pbe').density_fit().run()
        cls.mf_pbe0 = rks.RKS(mol, xc='pbe0').density_fit().run()
        cls.mf_wb97x = rks.RKS(mol, xc='wb97x').density_fit().run()
        cls.nstates = 5  # Test the first 5 excited states

    @classmethod
    def tearDownClass(cls):
        # Close the molecule's output stream
        cls.mol.stdout.close()

    ''' TDA '''
    def test_tda_pbe(self):
        """Test TDA-ris method with PBE functional"""
        mf = self.mf_pbe
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-5)
        td.kernel()  
        energies = td.energies
        fosc     = td.oscillator_strength

        # Reference energies  (in eV) and oscillator strengths
        ref_energies = [6.404665, 7.756897, 8.352699, 8.825232, 9.068202]  
        ref_fosc     = [0.002239, 0.025867, 0.004578, 0.024548, 0.046676]  

        print('tda_pbe energies', ', '.join(f'{e:.6f}' for e in energies))
        print('ref_energies    ', ', '.join(f'{e:.6f}' for e in ref_energies))
        print('tda_pbe fosc    ', ', '.join(f'{f:.6f}' for f in fosc))
        print('ref_fosc        ', ', '.join(f'{f:.6f}' for f in ref_fosc))

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(), 0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tda_pbe0(self):
        """Test TDA-ris method with PBE0 functional"""
        mf = self.mf_pbe0
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-5)
        td.kernel()  
        energies = td.energies
        fosc     = td.oscillator_strength
        ref_energies = [7.113308, 8.836901, 9.116842, 9.872045, 10.122341]  
        ref_fosc     = [00.001698, 0.052238, 0.004957, 0.027982, 0.047371]  

        print('tda_pbe0 energies', ', '.join(f'{e:.6f}' for e in energies))
        print('ref_energies    ', ', '.join(f'{e:.6f}' for e in ref_energies))
        print('tda_pbe0 fosc    ', ', '.join(f'{f:.6f}' for f in fosc))
        print('ref_fosc        ', ', '.join(f'{f:.6f}' for f in ref_fosc))

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(), 0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tda_wb97x(self):
        """Test TDA-ris method with wB97x functional"""
        mf = self.mf_wb97x
        td = ris.TDA(mf=mf, nstates=self.nstates, spectra=False, 
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies
        fosc = td.oscillator_strength
        ref_energies = [7.417226, 9.551203, 9.561623, 10.499912, 10.824862]  
        ref_fosc     = [ 0.001173, 0.003632, 0.075188, 0.026098, 0.051708]  
        print('tda_wb97x energies', ', '.join(f'{e:.6f}' for e in energies))
        print('ref_energies      ', ', '.join(f'{e:.6f}' for e in ref_energies))
        print('tda_wb97x fosc    ', ', '.join(f'{f:.6f}' for f in fosc))
        print('ref_fosc          ', ', '.join(f'{f:.6f}' for f in ref_fosc))

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    ''' TDDFT '''
    def test_tddft_pbe(self):
        """Test TDDFT-ris method with PBE functional"""
        mf = self.mf_pbe
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies
        fosc     = td.oscillator_strength

        # Reference energies  (in eV) and oscillator strengths
        ref_energies = [6.401243, 7.750149, 8.326640, 8.816009, 9.038981]  
        ref_fosc     = [0.001780, 0.024351, 0.003966, 0.021896, 0.043143]  

        print('tddft_pbe energies', ', '.join(f'{e:.6f}' for e in energies))
        print('ref_energies      ', ', '.join(f'{e:.6f}' for e in ref_energies))
        print('tddft_pbe fosc    ', ', '.join(f'{f:.6f}' for f in fosc))
        print('ref_fosc          ', ', '.join(f'{f:.6f}' for f in ref_fosc))


        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tddft_pbe0(self):
        """Test TDDFT-ris method with PBE0 functional"""
        mf = self.mf_pbe0
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies
        fosc     = td.oscillator_strength

        ref_energies = [7.109694, 8.827497, 9.092700, 9.863771, 10.095211]  
        ref_fosc     = [0.001341, 0.049354, 0.004444, 0.025127, 0.046499]  

        print('tddft_pbe0 energies', ', '.join(f'{e:.6f}' for e in energies))
        print('ref_energies       ', ', '.join(f'{e:.6f}' for e in ref_energies))
        print('tddft_pbe0 fosc    ', ', '.join(f'{f:.6f}' for f in fosc))
        print('ref_fosc           ', ', '.join(f'{f:.6f}' for f in ref_fosc))

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tddft_wb97x(self):
        """Test TDDFT-ris method with wB97x functional"""
        mf = self.mf_wb97x
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=40, J_fit='sp', K_fit='s', gram_schmidt=True, single=True, conv_tol=1e-3)
        td.kernel()  
        energies = td.energies
        fosc = td.oscillator_strength

        ref_energies = [7.413729, 9.525444, 9.551607, 10.492725, 10.796021]  
        ref_fosc     = [0.000904, 0.003295, 0.071938, 0.023767, 0.054760]  

        print('tddft_wb97x energies', ', '.join(f'{e:.6f}' for e in energies))   
        print('ref_energies        ', ', '.join(f'{e:.6f}' for e in ref_energies))
        print('tddft_wb97x fosc    ', ', '.join(f'{f:.6f}' for f in fosc))
        print('ref_fosc            ', ', '.join(f'{f:.6f}' for f in ref_fosc))

        self.assertAlmostEqual(abs(energies[:len(ref_energies)] - ref_energies).max(),0, PLACES)
        self.assertAlmostEqual(abs(fosc[:len(ref_fosc)] - ref_fosc).max(),0, PLACES)

    def test_tddft_pbe_get_ab(self):
        """Test TDDFT-ris get_ab method with PBE0 functional"""
        mf = self.mf_pbe
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', gram_schmidt=True, single=False, conv_tol=1e-7)
        td.kernel()  
        energies = td.energies
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]*27.21138602

        self.assertAlmostEqual(abs(e_ab-np.array(energies)).max(),0, PLACES)

    def test_tddft_pbe0_get_ab(self):
        """Test TDDFT-ris get_ab method with PBE0 functional"""
        mf = self.mf_pbe0
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='s', gram_schmidt=True, single=False, conv_tol=1e-7)
        td.kernel()  
        energies = td.energies
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]*27.21138602

        self.assertAlmostEqual(abs(e_ab-np.array(energies)).max(),0, PLACES)

    def test_tddft_wb97x_get_ab(self):
        """Test TDDFT-ris get_ab method with wb97x functional"""
        mf = self.mf_wb97x
        td = ris.TDDFT(mf=mf, nstates=self.nstates, spectra=False,
                      Ktrunc=0, J_fit='sp', K_fit='sp', gram_schmidt=True, single=False, conv_tol=1e-7)
        td.kernel()  
        energies = td.energies
        a,b = td.get_ab()
        e_ab = diagonalize(a, b, self.nstates)[0]*27.21138602

        self.assertAlmostEqual(abs(e_ab-np.array(energies)).max(),0, 2) # TODO: change to PLACES


if __name__ == "__main__":
    print("Full Tests for TDDFT-RIS with PBE, PBE0, and wB97x")
    unittest.main()

