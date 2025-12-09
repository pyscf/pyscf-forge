#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import __config__
setattr(__config__, 'lnocc_DEBUG_BLKSIZE', True)    # debug outcore mode
from pyscf import gto, scf, mp, cc, lo
from pyscf.cc.ccsd_t import kernel as CCSD_T
from pyscf.lno import LNOCCSD, LNOCCSD_T
from pyscf.lno.tools import autofrag_iao


class WaterDimer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        O    1.416468653903    0.111264435953    0.000000000000
        H    1.746241653903   -0.373945564047   -0.758561000000
        H    1.746241653903   -0.373945564047    0.758561000000
        '''
        mol.basis = 'cc-pvdz'
        mol.precision = 1e-10
        mol.build()
        mf = scf.RHF(mol).density_fit().run()

        # canonical
        frozen = 2
        mymp = mp.MP2(mf, frozen=frozen)
        mymp.kernel(with_t2=False)
        efull_mp2 = mymp.e_corr

        mycc = cc.CCSD(mf, frozen=frozen)
        eris = mycc.ao2mo()
        mycc.kernel(eris=eris)
        efull_ccsd = mycc.e_corr

        efull_t = CCSD_T(mycc, eris=eris, verbose=mycc.verbose)
        efull_ccsd_t = efull_ccsd + efull_t

        cls.mol = mol
        cls.mf = mf
        cls.frozen = frozen
        cls.ecano = [efull_mp2, efull_ccsd, efull_ccsd_t]
    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf, cls.ecano, cls.frozen

    def test_lno_pm_by_thresh(self):
        mol = self.mol
        mf = self.mf
        frozen = self.frozen

        # PM localization
        orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
        mlo = lo.PipekMezey(mol, orbocc)
        lo_coeff = mlo.kernel()
        for i in range(100): # always performing jacobi sweep to avoid trapping in local minimum/saddle point
            stable, lo_coeff1 = mlo.stability_jacobi()
            if stable:
                break
            mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            lo_coeff = mlo.kernel()

        # Fragment list: for PM, every orbital corresponds to a fragment
        frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

        gamma = 10
        threshs = [1e-5,1e-6,1e-100]
        refs = [
            [-0.4044781783,-0.4231598372,-0.4292049721],
            [-0.4058765086,-0.4244510794,-0.4307864928],
            self.ecano
        ]
        for thresh,ref in zip(threshs,refs):
            mcc = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
            mcc.lno_thresh = [thresh*gamma,thresh]
            mcc.kernel()
            emp2 = mcc.e_corr_pt2
            eccsd = mcc.e_corr_ccsd
            eccsd_t = mcc.e_corr_ccsd_t
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [emp2,eccsd,eccsd_t]])))
            self.assertAlmostEqual(emp2, ref[0], 6)
            self.assertAlmostEqual(eccsd, ref[1], 6)
            self.assertAlmostEqual(eccsd_t, ref[2], 6)


        # force outcore ao2mo for generating ovL
        for thresh,ref in zip(threshs,refs):
            mcc = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
            mcc.force_outcore_ao2mo = True
            mcc.lno_thresh = [thresh*gamma,thresh]
            mcc.kernel()
            emp2 = mcc.e_corr_pt2
            eccsd = mcc.e_corr_ccsd
            eccsd_t = mcc.e_corr_ccsd_t
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [emp2,eccsd,eccsd_t]])))
            self.assertAlmostEqual(emp2, ref[0], 6)
            self.assertAlmostEqual(eccsd, ref[1], 6)
            self.assertAlmostEqual(eccsd_t, ref[2], 6)

    def test_lno_iao_by_thresh(self):
        mol = self.mol
        mf = self.mf
        frozen = self.frozen

        # IAO localization
        orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
        iao_coeff = lo.iao.iao(mol, orbocc)
        lo_coeff = lo.orth.vec_lowdin(iao_coeff, mf.get_ovlp())
        moliao = lo.iao.reference_mol(mol)

        # Fragment list: all IAOs belonging to same atom form a fragment
        frag_lolist = autofrag_iao(moliao)

        gamma = 10
        threshs = [1e-5,1e-6,1e-100]
        refs = [
            [-0.4054784012,-0.4240686326,-0.4303996712],
            [-0.4060479828,-0.4245745223,-0.4309965749],
            self.ecano
        ]
        for thresh,ref in zip(threshs,refs):
            mcc = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
            mcc.lno_thresh = [thresh*gamma,thresh]
            mcc.kernel()
            emp2 = mcc.e_corr_pt2
            eccsd = mcc.e_corr_ccsd
            eccsd_t = mcc.e_corr_ccsd_t
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [emp2,eccsd,eccsd_t]])))
            self.assertAlmostEqual(emp2, ref[0], 6)
            self.assertAlmostEqual(eccsd, ref[1], 6)
            self.assertAlmostEqual(eccsd_t, ref[2], 6)



if __name__ == "__main__":
    print("Full Tests for LNO-CCSD and LNO-CCSD(T)")
    unittest.main()
