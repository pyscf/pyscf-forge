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
from pyscf.pbc import gto, scf, tools as pbctools
from pyscf.lno import LNOCCSD, LNOCCSD_T
from pyscf.pbc.lno import KLNOCCSD, KLNOCCSD_T
from pyscf.lno.tools import autofrag_iao
from pyscf.pbc.lno.tools import k2s_scf, k2s_iao
from pyscf import lo


class WaterDimer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        O    1.416468653903    0.111264435953    0.000000000000
        H    1.746241653903   -0.373945564047   -0.758561000000
        H    1.746241653903   -0.373945564047    0.758561000000
        '''
        cell.a = np.eye(3) * 5
        cell.basis = 'def2-svp'
        cell.precision = 1e-10
        cell.build()

        kmesh = [2,1,1]
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)
        scell = pbctools.super_cell(cell, kmesh)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit().run()
        smf = k2s_scf(kmf)

        cls.cell = cell
        cls.kmf = kmf
        cls.scell = scell
        cls.smf = smf
        cls.frozen = 2 * nkpts

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        cls.scell.stdout.close()
        del cls.cell, cls.kmf, cls.frozen
        del cls.scell, cls.smf

    def test_lno_iao_by_thresh(self):
        cell = self.cell
        kmf = self.kmf
        smf = self.smf
        scell = smf.cell
        frozen = self.frozen
        kpts = kmf.kpts
        nkpts = len(kpts)

        # IAO localization in supercell
        kocc_coeff = [kmf.mo_coeff[k][:,kmf.mo_occ[k]>1e-6] for k in range(nkpts)]
        lo_coeff = k2s_iao(cell, kocc_coeff, kpts, orth=True)

        # k-point LNO: only need to compute fragments within a unit cell
        cell_iao = lo.iao.reference_mol(cell)
        frag_lolist = autofrag_iao(cell_iao)

        # Supercell LNO:
        ''' In principle, one needs to treat all fragments within the supercell. But here
            the supercell SCF object is from `k2s_scf`. As a result, the MOs, the IAOs and
            the fragments are all translationally invariant. We only need to treat fragments
            from the first unit cell, i.e., same as k-point LNO.
        '''
        sfrag_lolist = frag_lolist

        gamma = 10
        threshs = [1e-5,1e-6,1e-100]
        for thresh in threshs:
            # mcc = LNOCCSD_T(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
            kmcc = KLNOCCSD(kmf, lo_coeff, frag_lolist, mf=smf, frozen=frozen).set(verbose=5)
            kmcc.lno_thresh = [thresh*gamma,thresh]
            kmcc.kernel()
            kemp2 = kmcc.e_corr_pt2
            keccsd = kmcc.e_corr_ccsd
            keccsd_t = kmcc.e_corr_ccsd_t

            mcc = LNOCCSD(smf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
            mcc.lno_thresh = [thresh*gamma,thresh]
            mcc.kernel()

            semp2 = mcc.e_corr_pt2
            seccsd = mcc.e_corr_ccsd
            seccsd_t = mcc.e_corr_ccsd_t
            # print('[%s],' % (','.join([f'{x:.10f}' for x in [semp2,seccsd,seccsd_t]])))
            self.assertAlmostEqual(kemp2, semp2, 6)
            self.assertAlmostEqual(keccsd, seccsd, 6)
            # self.assertAlmostEqual(keccsd_t, seccsd_t, 6)


class H4Chain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 0
        cell.atom = 'H 0 0 0; H 0 0 0.74; H 1.85 0 0; H 1.85 0 0.74'
        cell.a = np.diag([3.7, 4.0, 4.0])
        cell.basis = '6-31g'
        cell.precision = 1e-10
        cell.build()
        kmf = scf.KRHF(cell, kpts=cell.make_kpts([2,1,1])).density_fit().run(conv_tol=1e-11)
        cls.cell = cell
        cls.kmf = kmf

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kmf

    def test_klnoccsd_t_outcore(self):
        ''' Outcore ovL for the LNO step, alone and with the DF impurity CCSD integrals
            (vvL), against incore energies. The aux blocks are split and every q block is
            zero padded (naux > naux_by_q) so the blocked and padded writes are exercised.
        '''
        from pyscf import lib
        from pyscf.pbc.lno import klno, klnoccsd
        from pyscf.pbc.lno import tools as ktools
        kmf = self.kmf
        smf = k2s_scf(kmf)
        kocc_coeff = [c[:,o>1e-6] for c,o in zip(kmf.mo_coeff, kmf.mo_occ)]
        lo_coeff = k2s_iao(self.cell, kocc_coeff, kmf.kpts, minao='sto-3g', orth=True)
        frag_lolist = autofrag_iao(lo.iao.reference_mol(self.cell, 'sto-3g'))
        # incore KLNO-CCSD(T): e_corr_pt2, e_corr_ccsd, e_corr_ccsd_t
        ref = [-0.0335002580, -0.0484248396, -0.0484581416]
        naux_pad = property(lambda k2sdf: k2sdf.get_naoaux().max() + 3)
        for force_dfkcc in (False, True):
            with self.subTest(force_dfkcc=force_dfkcc):
                with lib.temporary_env(klno, DEBUG_BLKSIZE=True), \
                     lib.temporary_env(klnoccsd, DEBUG_BLKSIZE=True, FORCE_DFKCC=force_dfkcc), \
                     lib.temporary_env(ktools.K2SDF, naux=naux_pad):
                    mcc = KLNOCCSD_T(kmf, lo_coeff, frag_lolist, mf=smf)
                    mcc.lno_thresh = [1e-5, 1e-6]
                    mcc.kwargs_imp = {'conv_tol': 1e-10, 'conv_tol_normt': 1e-8}
                    mcc.force_outcore_ao2mo = True
                    mcc.kernel()
                self.assertAlmostEqual(mcc.e_corr_pt2, ref[0], 8)
                self.assertAlmostEqual(mcc.e_corr_ccsd, ref[1], 8)
                self.assertAlmostEqual(mcc.e_corr_ccsd_t, ref[2], 8)



if __name__ == "__main__":
    print("Full Tests for LNO-CCSD and LNO-CCSD(T)")
    unittest.main()
