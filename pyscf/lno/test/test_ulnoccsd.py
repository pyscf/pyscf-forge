#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
#

import pytest
import numpy as np
from pyscf import gto, lo, scf
from pyscf.lno import tools
from pyscf.lno.ulnoccsd import ULNOCCSD_T

@pytest.fixture
def h2o2():
    # S22-2: water dimer
    atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        O    1.416468653903    0.111264435953    0.000000000000
        H    1.746241653903   -0.373945564047   -0.758561000000
        H    1.746241653903   -0.373945564047    0.758561000000
    '''
    basis = 'cc-pvdz'
    mol = gto.M(atom=atom, basis=basis, spin=0, verbose=0, max_memory=8000)
    yield mol

def test_ulnoccsd_1o(h2o2):
    mol = h2o2
    mf = scf.UHF(mol).density_fit()
    mf.kernel()

    frozen = 0
    # PM
    orbocca = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
    orbloca = lo.PipekMezey(mol, orbocca).kernel()
    orboccb = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
    orblocb = lo.PipekMezey(mol, orboccb).kernel()
    orbloc = [orbloca, orblocb]

    lno_type = ['1h'] * 2
    lno_thresh = [1e-4] * 2
    oa = [[[i],[]] for i in range(mol.nelectron//2)]
    ob = [[[],[i]] for i in range(mol.nelectron//2)]
    frag_lolist = oa + ob

    mlno = ULNOCCSD_T(mf, orbloc, frag_lolist, lno_type=lno_type, lno_thresh=lno_thresh, frozen=frozen)
    mlno.lo_proj_thresh_active = None
    mlno.kernel()
    ecc = mlno.e_corr_ccsd
    ecc_t = mlno.e_corr_ccsd_t

    assert abs(ecc - -0.3942038878) < 1e-5
    assert abs(ecc_t - -0.3956640879) < 1e-5

def test_ulnoccsd_frag(h2o2):
    mol = h2o2
    mf = scf.UHF(mol).density_fit()
    mf.kernel()

    frozen = 0
    # PM
    orbocca = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
    orbloca = lo.PipekMezey(mol, orbocca).kernel()
    orboccb = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
    orblocb = lo.PipekMezey(mol, orboccb).kernel()
    orbloc = [orbloca, orblocb]

    lno_type = ['1h'] * 2
    lno_thresh = [1e-4] * 2
    frag_atmlist = tools.autofrag_atom(mol, True)
    frag_lolist = tools.map_lo_to_frag(mol, orbloc, frag_atmlist)

    mlno = ULNOCCSD_T(mf, orbloc, frag_lolist, lno_type=lno_type, lno_thresh=lno_thresh, frozen=frozen)
    mlno.lo_proj_thresh_active = None
    mlno.kernel()
    ecc = mlno.e_corr_ccsd
    ecc_t = mlno.e_corr_ccsd_t

    assert abs(ecc - -0.42647563905653474) < 1e-5
    assert abs(ecc_t - -0.4325496978856983) < 1e-5
