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
from functools import reduce

from pyscf import __config__
setattr(__config__, 'lnocc_DEBUG_BLKSIZE', True)    # debug outcore mode
from pyscf import gto, scf, lo, lib
from pyscf.lno.make_lno_rdm1 import *
from pyscf.lno.lno import _LNODFINCOREERIS


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

        cls.mol = mol
        cls.mf = mf
        cls.frozen = frozen

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol, cls.mf, cls.frozen

    def test_make_rdm1(self):
        mol = self.mol
        mf = self.mf
        frozen = self.frozen

        orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
        orbvir = mf.mo_coeff[:,np.count_nonzero(mf.mo_occ):]

        moeocc = mf.mo_energy[frozen:np.count_nonzero(mf.mo_occ)]
        moevir = mf.mo_energy[np.count_nonzero(mf.mo_occ):]

        # occ PM localization
        mlo = lo.PipekMezey(mol, orbocc)
        occloc = mlo.kernel()
        for i in range(100): # always performing jacobi sweep to avoid trapping in local minimum/saddle point
            stable, lo_coeff1 = mlo.stability_jacobi()
            if stable:
                break
            mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            occloc = mlo.kernel()

        uocc_loc = reduce(np.dot, (occloc.T.conj(), mf.get_ovlp(), orbocc))

        # vir PM localization
        mlo = lo.PipekMezey(mol, orbvir)
        virloc = mlo.kernel()
        for i in range(100): # always performing jacobi sweep to avoid trapping in local minimum/saddle point
            stable, lo_coeff1 = mlo.stability_jacobi()
            if stable:
                break
            mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            virloc = mlo.kernel()

        uvir_loc = reduce(np.dot, (virloc.T.conj(), mf.get_ovlp(), orbvir))


        eris=_LNODFINCOREERIS(mf.with_df, orbocc ,orbvir, mf.max_memory,
                              verbose=mol.verbose,stdout=mol.output)
        eris.build()

        # arr = make_full_rdm1_occ(eris, moeocc, moevir)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, 0.02856113869995006, 7)

        # arr = make_full_rdm1_vir(eris, moeocc, moevir)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, -0.01925614120035, 7)

        # arr = make_lo_rdm1_occ_1h(eris, moeocc, moevir, uocc_loc)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, -0.02856113869994, 7)

        # arr = make_lo_rdm1_occ_1p(eris, moeocc, moevir, uvir_loc)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, -0.02856113869994, 7)

        # arr = make_lo_rdm1_occ_2p(eris, moeocc, moevir, uvir_loc)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, -0.02856113869994, 7)

        # arr = make_lo_rdm1_vir_1h(eris, moeocc, moevir, uocc_loc)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, -0.01925614120035, 7)

        # arr = make_lo_rdm1_vir_1p(eris, moeocc, moevir, uvir_loc)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, -0.01925614120035, 7)

        # arr = make_lo_rdm1_vir_2h(eris, moeocc, moevir, uocc_loc)
        # fp = lib.fp(arr)
        # self.assertAlmostEqual(fp, -0.01925614120035, 7)


        # NOTE: RDM1 validation is performed in the AO basis to avoid numerical ambiguities arising from MO degeneracies.
        arr = make_full_rdm1_occ(eris, moeocc, moevir)
        ao_arr = reduce(np.dot, (orbocc, arr, orbocc.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, 0.008582041501554553, 7)

        arr = make_full_rdm1_vir(eris, moeocc, moevir)
        ao_arr = reduce(np.dot, (orbvir, arr, orbvir.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, 0.044855160932431, 7)

        arr = make_lo_rdm1_occ_1h(eris, moeocc, moevir, uocc_loc)
        ao_arr = reduce(np.dot, (orbocc, arr, orbocc.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, -0.008582041501554828, 7)

        arr = make_lo_rdm1_occ_1p(eris, moeocc, moevir, uvir_loc)
        ao_arr = reduce(np.dot, (orbocc, arr, orbocc.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, -0.008582041501554807, 7)

        arr = make_lo_rdm1_occ_2p(eris, moeocc, moevir, uvir_loc)
        ao_arr = reduce(np.dot, (orbocc, arr, orbocc.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, -0.008582041501554796, 7)

        arr = make_lo_rdm1_vir_1h(eris, moeocc, moevir, uocc_loc)
        ao_arr = reduce(np.dot, (orbvir, arr, orbvir.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, 0.04485516093243097, 7)

        arr = make_lo_rdm1_vir_1p(eris, moeocc, moevir, uvir_loc)
        ao_arr = reduce(np.dot, (orbvir, arr, orbvir.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, 0.04485516093243104, 7)

        arr = make_lo_rdm1_vir_2h(eris, moeocc, moevir, uocc_loc)
        ao_arr = reduce(np.dot, (orbvir, arr, orbvir.T.conj()))
        fp = lib.fp(ao_arr)
        self.assertAlmostEqual(fp, 0.04485516093243097, 7)

if __name__ == "__main__":
    print("TESTS FOR MAKE RDM1")
    unittest.main()
