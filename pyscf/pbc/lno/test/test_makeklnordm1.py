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
from pyscf import lo, lib
from pyscf.pbc import gto, scf, tools as pbctools
from pyscf.pbc.lno.tools import k2s_scf, sort_orb_by_cell
from pyscf.pbc.lno.klno import _KLNODFINCOREERIS_REAL, _KLNODFINCOREERIS_COMPLEX
from pyscf.pbc.lno.make_lno_rdm1 import *


class Water_REAL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        '''
        cell.a = np.eye(3) * 5
        cell.basis = 'cc-pvdz'
        cell.precision = 1e-10
        cell.build()

        kmesh = [1,1,1]
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)
        scell = pbctools.super_cell(cell, kmesh)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.conv_tol=1e-12
        kmf.kernel()
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

    def test_make_lno_real(self):
        cell = self.cell
        kmf = self.kmf
        mf = self.smf
        kpts = kmf.kpts

        orbocc = mf.mo_coeff[:,mf.mo_occ>0]
        orbvir = mf.mo_coeff[:,mf.mo_occ==0]
        moeocc =  mf.mo_energy[mf.mo_occ>0]
        moevir =  mf.mo_energy[mf.mo_occ==0]

        # PM localized occ orbs
        mlo = lo.PipekMezey(mf.cell, orbocc)
        lo_coeff = mlo.kernel()
        for i in range(100):
            lo_coeff1 = mlo.stability_jacobi()[1]
            if lo_coeff1 is lo_coeff:
                break
            mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            lo_coeff = mlo.kernel()

        # sort LOs by unit cell
        s1e = mf.get_ovlp()
        Nk = len(kpts)
        lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)
        uocc_loc = reduce(np.dot, (lo_coeff.T.conj(), s1e, orbocc))

        # PM localized vir orbs
        mlo = lo.PipekMezey(mf.cell, orbvir)
        lo_coeff = mlo.kernel()
        for i in range(100):
            lo_coeff1 = mlo.stability_jacobi()[1]
            if lo_coeff1 is lo_coeff:
                break
            mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            lo_coeff = mlo.kernel()

        # sort LOs by unit cell
        s1e = mf.get_ovlp()
        Nk = len(kpts)
        lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)
        uvir_loc = reduce(np.dot, (lo_coeff.T.conj(), s1e, orbvir))

        eris = _KLNODFINCOREERIS_REAL(kmf.with_df, orbocc, orbvir, kmf.max_memory,
                              verbose=cell.verbose,stdout=cell.stdout)
        eris.build()

        arr1,arr2 = make_full_rdm1(eris, moeocc, moevir)
        fp = lib.fp(arr1)
        self.assertAlmostEqual(fp, 0.06405023903998112, 7)

        fp = lib.fp(arr2)
        self.assertAlmostEqual(fp, -0.02656427018284405, 7)

        arr = make_lo_rdm1_occ_1h_real(eris, moeocc, moevir, uocc_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.06405023903998112, 7)

        arr = make_lo_rdm1_occ_1p_real(eris, moeocc, moevir, uvir_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.06405023903998112, 7)

        arr = make_lo_rdm1_occ_2p_real(eris, moeocc, moevir, uvir_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.06405023903998112, 7)

        arr = make_lo_rdm1_vir_1h_real(eris, moeocc, moevir, uocc_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.026564270182844, 7)

        arr = make_lo_rdm1_vir_1p_real(eris, moeocc, moevir, uvir_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.026564270182844, 7)

        arr = make_lo_rdm1_vir_2h_real(eris, moeocc, moevir, uocc_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.026564270182844, 7)


class Water_COMPLEX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O   -1.485163346097   -0.114724564047    0.000000000000
        H   -1.868415346097    0.762298435953    0.000000000000
        H   -0.533833346097    0.040507435953    0.000000000000
        '''
        cell.a = np.eye(3) * 5
        cell.basis = 'cc-pvdz'
        cell.precision = 1e-10
        cell.build()

        kmesh = [1,1,1]
        kpts = np.array([[0.1,0.1,0.1]])
        nkpts = len(kpts)
        scell = pbctools.super_cell(cell, kmesh)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.conv_tol=1e-12
        kmf.kernel()
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

    def test_make_lno_complex(self):
        cell = self.cell
        kmf = self.kmf
        mf = self.smf
        kpts = kmf.kpts

        orbocc = mf.mo_coeff[:,mf.mo_occ>0]
        orbvir = mf.mo_coeff[:,mf.mo_occ==0]
        moeocc =  mf.mo_energy[mf.mo_occ>0]
        moevir =  mf.mo_energy[mf.mo_occ==0]

        # PM localized occ orbs
        mlo = lo.PipekMezey(mf.cell, orbocc)
        lo_coeff = mlo.kernel()
        for i in range(100):
            lo_coeff1 = mlo.stability_jacobi()[1]
            if lo_coeff1 is lo_coeff:
                break
            mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            lo_coeff = mlo.kernel()

        # sort LOs by unit cell
        s1e = mf.get_ovlp()
        Nk = len(kpts)
        lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)
        uocc_loc = reduce(np.dot, (lo_coeff.T.conj(), s1e, orbocc))

        # PM localized vir orbs
        mlo = lo.PipekMezey(mf.cell, orbvir)
        lo_coeff = mlo.kernel()
        for i in range(100):
            lo_coeff1 = mlo.stability_jacobi()[1]
            if lo_coeff1 is lo_coeff:
                break
            mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
            mlo.init_guess = None
            lo_coeff = mlo.kernel()

        # sort LOs by unit cell
        s1e = mf.get_ovlp()
        Nk = len(kpts)
        lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)
        uvir_loc = reduce(np.dot, (lo_coeff.T.conj(), s1e, orbvir))

        eris = _KLNODFINCOREERIS_COMPLEX(kmf.with_df, orbocc, orbvir, kmf.max_memory,
                              verbose=cell.verbose,stdout=cell.stdout)
        eris.build()

        arr = make_lo_rdm1_occ_1h_complex(eris, moeocc, moevir, uocc_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.06422719794290856+2.5105071595401036e-05j, 7)

        arr = make_lo_rdm1_occ_1p_complex(eris, moeocc, moevir, uvir_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.06422719794290856+2.5105071595401036e-05j, 7)

        arr = make_lo_rdm1_occ_2p_complex(eris, moeocc, moevir, uvir_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.06422719794290856+2.5105071595401036e-05j, 7)

        arr = make_lo_rdm1_vir_1h_complex(eris, moeocc, moevir, uocc_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.002766848360192287-0.034349212190162834j, 7)

        arr = make_lo_rdm1_vir_1p_complex(eris, moeocc, moevir, uvir_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.002766848360192287-0.034349212190162834j, 7)

        arr = make_lo_rdm1_vir_2h_complex(eris, moeocc, moevir, uocc_loc)
        fp = lib.fp(arr)
        self.assertAlmostEqual(fp, -0.002766848360192287-0.034349212190162834j, 7)



if __name__ == "__main__":
    print("TESTS FOR MAKE RDM1")
    unittest.main()
