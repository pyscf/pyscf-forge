#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
# Authors:
#          Shuhang Li <shuhangli98@gmail.com>
#          Zijun Zhao <brian.zhaozijun@gmail.com>
#

import unittest
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci
from pyscf import dsrg_mrpt2

def setUpModule():
    global moln2, molhf, molo2, molo2_df, molh2o, molbeh, molh2op, mfn2, mfhf, mfo2, mfo2_df, mfh2o, mfbeh, mfh2op

    moln2 = gto.M(
        verbose = 2,
        atom = '''
    N 0 0 0
    N 0 1.4 0
    ''',
        basis = '6-31g', spin=0, charge=0, symmetry=False, output = '/dev/null',
    )
    mfn2 = scf.RHF(moln2)
    mfn2.kernel()

    molhf = gto.M(
        verbose = 2,
        atom = '''
    H 0 0 0
    F 0 0 1.5
    ''',
        basis = 'cc-pvdz', spin=0, charge=0, output = '/dev/null',
    )
    mfhf = scf.RHF(molhf)
    mfhf.kernel()

    molo2 = gto.M(
        verbose = 2,
        atom = '''
    O 0 0 0
    O 0 0 1.251
    ''',
        basis = 'cc-pvdz', spin=2, charge=0, symmetry='d2h', output = '/dev/null',
    )
    mfo2 = scf.ROHF(molo2)
    mfo2.kernel()

    molo2_df = gto.M(
        verbose = 2,
        atom = '''
    O 0 0 0
    O 0 0 1.251
    ''',
        basis = 'cc-pvdz', spin=0, charge=0, symmetry='d2h', output = '/dev/null',
    )
    mfo2_df = scf.RHF(molo2_df).density_fit('cc-pvdz-jkfit')
    mfo2_df.kernel()

    molh2o = gto.M(
        verbose = 2,
        atom='''
    O     0.    0.000    0.1174
    H     0.    0.757   -0.4696
    H     0.   -0.757   -0.4696
        ''',
        basis = '6-31g', spin=0, charge=0, symmetry=True, output = '/dev/null',
    )

    mfh2o = scf.RHF(molh2o)
    mfh2o.kernel()

    molh2op = gto.M(
        atom = '''
    O        0.000000    0.000000    0.117790
    H        0.000000    0.755453   -0.471161
    H        0.000000   -0.755453   -0.471161''',
        basis = 'cc-pvdz', charge = 1, spin = 1, symmetry = 'c2v', output = '/dev/null',
    )

    mfh2op = scf.ROHF(molh2op)
    mfh2op.kernel()

    molbeh = gto.M(
        verbose = 2,
        atom = '''
    Be 0 0 0
    H 0 0 1.0
    ''',
        basis = '6-31g', spin=1, charge=0, symmetry='c2v', output='/dev/null',
    )
    mfbeh = scf.ROHF(molbeh)
    mfbeh.kernel()

def tearDownModule():
    global moln2, molhf, molo2, molo2_df, molh2o, molbeh, molh2op, mfn2, mfhf, mfo2, mfo2_df, mfh2o, mfbeh, mfh2op
    moln2.stdout.close()
    molhf.stdout.close()
    molo2.stdout.close()
    molo2_df.stdout.close()
    molh2o.stdout.close()
    molbeh.stdout.close()
    molh2op.stdout.close()
    del moln2, molhf, molo2, molo2_df, molh2o, molbeh, molh2op, mfn2, mfhf, mfo2, mfo2_df, mfh2o, mfbeh, mfh2op

class KnownValues(unittest.TestCase):
    def test_n2_casci_nosym(self):
        mc = mcscf.CASCI(mfn2, 6, 6)
        mc.kernel()
        e = dsrg_mrpt2.DSRG_MRPT2(mc).kernel()
        self.assertAlmostEqual(e, -108.99538785901142, delta=1.0e-6)
# Forte: -108.995383133239699

    def test_hf_casci_iterative_relaxation(self):
        mc = mcscf.CASCI(mfhf, 4, 6)
        mc.kernel()
        pt = dsrg_mrpt2.DSRG_MRPT2(mc, relax='iterate')
        pt.kernel()
        self.assertAlmostEqual(pt.e_corr, -0.19737539, delta=1.0e-6)
# Forte: -0.197374740048

    def test_triplet_o2_casscf_iterative_relaxation(self):
        mc = mcscf.CASSCF(mfo2, 6, 8)
        mc.fix_spin_(ss=2) # triplet state
        mc.mc2step()
        pt = dsrg_mrpt2.DSRG_MRPT2(mc, s=1.0, relax='iterate')
        pt.kernel()
        self.assertAlmostEqual(pt.e_corr, -0.26173265, delta=1.0e-6)
# Forte: -0.261732560387

    def test_triplet_o2_sa_casscf_iterative(self):
        mc = mcscf.CASSCF(mfo2, 6, 8).state_average_([.5,.5],wfnsym='B1g')
        mc.fix_spin_(ss=2) # triplet state
        mc.mc2step()
        pt = dsrg_mrpt2.DSRG_MRPT2(mc, s=1.0, relax='iterate')
        pt.kernel()
        e_sa = pt.e_relax_eigval_shifted
        self.assertAlmostEqual(e_sa[0], -149.9777576166627, delta=1.0e-6)
        self.assertAlmostEqual(e_sa[1], -149.46392257544233, delta=1.0e-6)
# Forte: -149.977757019742
# Forte: -149.463920553410


    def test_singlet_o2_casscf_df(self):
        mc = mcscf.CASSCF(mfo2_df, 6, 8)
        mc.fix_spin_(ss=0) # singlet state
        mc.mc2step()
        pt = dsrg_mrpt2.DSRG_MRPT2(mc, s=0.5, relax='once')
        pt.kernel()
        self.assertAlmostEqual(pt.e_tot, -149.93467822, delta=1.0e-6)
# Forte: -149.934676874344

    def test_water_sa_casscf(self):
        mc = mcscf.CASSCF(mfh2o, 4, 4).state_average_([.5,.5],wfnsym='A1')
        mc.fix_spin_(ss=0)
        ncore = {'A1':2, 'B1':1}
        ncas = {'A1':2, 'B1':1,'B2':1}
        mo = mcscf.sort_mo_by_irrep(mc, mfh2o.mo_coeff, ncas, ncore)
        #mc.kernel(mo)
        mc.mc2step(mo)
        pt = dsrg_mrpt2.DSRG_MRPT2(mc, relax='once')
        pt.kernel()
        e_sa = pt.e_relax_eigval_shifted
        self.assertAlmostEqual(e_sa[0], -76.11686746968063, delta=1.0e-6)
        self.assertAlmostEqual(e_sa[1], -75.71394328285785, delta=1.0e-6)
# Forte: -76.116867427126
# Forte: -75.713943178963

    def test_water_cation_doublet_casscf(self):
        mc = mcscf.CASSCF(mfh2op, 4, 3)
        mc.mc2step()
        pt = dsrg_mrpt2.DSRG_MRPT2(mc, s=1.0, relax='iterate')
        pt.kernel()
        self.assertAlmostEqual(pt.e_tot, -75.788112389449551, delta=1.0e-6)
# Forte: -75.788112856007814

    def test_beh_doublet_casscf(self):
        mc = mcscf.CASSCF(mfbeh, 5, 3)
        mc.fix_spin_(ss=0.75)
        mc.mc2step() 
        pt = dsrg_mrpt2.DSRG_MRPT2(mc, s=1.0, relax='twice')
        pt.kernel()
        self.assertAlmostEqual(pt.e_tot, -15.104778782361588, delta=1.0e-6)
# Forte: -15.104777031629652

if __name__ == "__main__":
    print("Full Tests for DSRG-MRPT2")
    unittest.main()

