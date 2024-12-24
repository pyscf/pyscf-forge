#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

'''
Test code for
k-point spin-restricted periodic MP2 calculation using the staggered mesh method
Author: Xin Xing (xxing@berkeley.edu)
Reference: Staggered Mesh Method for Correlation Energy Calculations of Solids: Second-Order
        Møller–Plesset Perturbation Theory, J. Chem. Theory Comput. 2021, 17, 8, 4733-4745
'''

import unittest
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df
from pyscf.pbc.scf.khf_stagger import KHF_stagger

def build_h2_fftdf_cell():
    cell = pbcgto.Cell()
    cell.atom='''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        '''
    cell.unit = 'B'
    cell.pseudo = 'gth-pbe'
    cell.basis = 'gth-szv'
    cell.mesh = [12]*3
    cell.verbose = 4
    cell.output = '/dev/null'
    cell.build()
    return cell

def build_h2_gdf_cell():
    cell = pbcgto.Cell()
    cell.atom='''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        '''
    cell.unit = 'B'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.build()
    return cell

def run_kcell_fftdf(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    gdf = df.GDF(cell, abs_kpts).build()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()
    
    kmf_stagger = KHF_stagger(kmf)
    kmf_stagger.kernel()
    etot = kmf_stagger.e_tot
    ek_stagger = kmf_stagger.ek
    
    return etot, ek_stagger

def run_kcell_gdf(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    gdf = df.GDF(cell, abs_kpts).build()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()
    
    kmf_stagger = KHF_stagger(kmf,stagger_type='non-scf')
    kmf_stagger.kernel()
    etot = kmf_stagger.e_tot
    ek_stagger = kmf_stagger.ek
    
    return etot, ek_stagger

class KnownValues(unittest.TestCase):

    def test_222_h2_gdf(self):
        cell = build_h2_gdf_cell()
        nk = [2,2,2]
        etot, ek_stagger = run_kcell_gdf(cell,nk)
        self.assertAlmostEqual(etot, -0.40854731697431584, 7)
        self.assertAlmostEqual(ek_stagger, -0.04014371773827328, 7)


if __name__ == '__main__':
    print("Staggered mesh for exact exchange test")
    unittest.main()
