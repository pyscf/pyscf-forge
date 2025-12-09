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

import unittest
import numpy as np
from pyscf import gto, scf, mp, cc, lo
from pyscf.cc.uccsd_t import kernel as CCSD_T
from pyscf.lno import tools
from pyscf.lno.ulnoccsd import ULNOCCSD_T


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
        mol.build()
        mf = scf.UHF(mol).density_fit().run()

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

    def test_ulno_pm_by_thresh(self):
        mol = self.mol
        mf = self.mf
        frozen = self.frozen

        # PM localization
        orbocc = list()
        lo_coeff = list()
        for s in range(2):
            orbocc.append(mf.mo_coeff[s][:,frozen:np.count_nonzero(mf.mo_occ[s])])
            mlo = lo.PipekMezey(mol, orbocc[s])
            lo_coeff_s = mlo.kernel()
            for i in range(100): # always performing jacobi sweep to avoid trapping in local minimum/saddle point
                stable, lo_coeff1_s = mlo.stability_jacobi()
                if stable:
                    break
                mlo = lo.PipekMezey(mf.mol, lo_coeff1_s).set(verbose=4)
                mlo.init_guess = None
                lo_coeff_s = mlo.kernel()
            lo_coeff.append(lo_coeff_s)

        # Fragment list: for PM, every orbital corresponds to a fragment
        oa = [[[i],[]] for i in range(orbocc[0].shape[1])]
        ob = [[[],[i]] for i in range(orbocc[1].shape[1])]
        frag_lolist = oa + ob

        gamma = 10
        threshs = [1e-5*2,1e-6*2,1e-100*2]
        refs = [
            [-0.3995407761,-0.4185382023,-0.4231105742],
            [-0.4052089997,-0.4238689186,-0.4300854290],
            self.ecano
        ]
        for thresh,ref in zip(threshs,refs):
            mcc = ULNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
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
