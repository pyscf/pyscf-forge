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

from functools import reduce
import unittest
import numpy as np
from pyscf import gto, scf, cc, lo
from pyscf.lno.ulnoccsd_t import kernel as ULNOCCSD_T_kernel
from pyscf.lno.ulnoccsd_t_slow import kernel as ULNOCCSD_T_slow_kernel


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
        mol.spin = 2
        mol.basis = '631g'
        mol.build()
        mf = scf.UHF(mol).density_fit().run()

        frozen = 2

        cls.mol = mol
        cls.mf = mf
        cls.frozen = frozen

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf, cls.frozen

    def test_ulno_t(self):
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

        s1e = mf.get_ovlp()
        uocc_loc = (reduce(np.dot, (orbocc[0].T.conj(), s1e, lo_coeff[0])),
                    reduce(np.dot, (orbocc[1].T.conj(), s1e, lo_coeff[1])))

        # Fragment list: for PM, every orbital corresponds to a fragment
        oa = [[[i],[]] for i in range(orbocc[0].shape[1])]
        ob = [[[],[i]] for i in range(orbocc[1].shape[1])]
        frag_lolist = oa + ob

        mcc = cc.CCSD(mf, frozen=frozen)
        eris = mcc.ao2mo()
        _, t1, t2 = mcc.kernel(eris=eris)

        for frag in frag_lolist:
            prjlo = uocc_loc[0][:, frag[0]].T.conj(), uocc_loc[1][:, frag[1]].T.conj()
            slow_et = ULNOCCSD_T_slow_kernel(mcc, eris, prjlo, t1=t1, t2=t2)
            fast_et = ULNOCCSD_T_kernel(mcc, eris, prjlo, t1=t1, t2=t2)
            self.assertAlmostEqual(slow_et, fast_et, 10)

if __name__ == "__main__":
    print("Full Tests for LNO-CCSD(T)")
    unittest.main()
