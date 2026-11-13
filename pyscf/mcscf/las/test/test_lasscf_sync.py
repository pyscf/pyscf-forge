#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
import os
import copy
import unittest
import numpy as np
from scipy import linalg
from pyscf import lib, gto, scf, dft, fci, mcscf, df
from pyscf.tools import molden
from c2h4n4_struct import structure as struct
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF
topdir = os.path.abspath (os.path.join (__file__, '..'))

def setUpModule():
    global mol, mf, las, ugg, h_op, x, nmo, ncore, nocc, offs_ci1, offs_ci2, xorb, xci, xci0
    dr_nn = 2.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG 
    mol.output = '/dev/null'
    mol.spin = 0 
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,4), ((3,1),(1,3)), spin_sub=(3,3))
    las.max_cycle_macro = 1
    las.kernel ()
    # TODO: fix?
    #assert (las.converged)
    las.mo_coeff = np.loadtxt (os.path.join (topdir, 'lasscf_sync_mo.dat'))
    las.ci = [[np.loadtxt (os.path.join (topdir, 'lasscf_sync_ci0.dat'))],
              [-np.loadtxt (os.path.join (topdir, 'lasscf_sync_ci1.dat')).T]]

    ugg = las.get_ugg ()
    h_op = las.get_hop (ugg=ugg)
    nmo, ncore, nocc = h_op.nmo, h_op.ncore, h_op.nocc
    np.random.seed (0)
    x = np.random.rand (ugg.nvar_tot)
    offs_ci1 = ugg.nvar_orb
    offs_ci2 = offs_ci1 + np.squeeze (ugg.ncsf_sub)[0]
    xorb, xci = ugg.unpack (x)
    xci0 = [[np.zeros (16),], [np.zeros (16),]]


def tearDownModule():
    global mol, mf, las, ugg, h_op, x
    mol.stdout.close ()
    del mol, mf, las, ugg, h_op, x

sectors = ["ac","aa","vc","va","ci1","ci2"]
def itsec (vorb, vci):
    yield vorb[ncore:nocc,:ncore]
    yield vorb[ncore:nocc,ncore:nocc]
    yield vorb[nocc:,:ncore]
    yield vorb[nocc:,ncore:nocc]
    yield vci[0][0]
    yield vci[1][0]

# Regression fingerprints of the Newton-CG operators. test_hess_<out>_<in> checks the <out> block
# of H x for x nonzero only in <in>; test_hess_from_<sector> checks every block of H x for x
# nonzero only in that orbital-rotation sector.
class KnownValues(unittest.TestCase):
    def test_grad_fingerprint (self):
        gorb0, gci0, gx0 = las.get_grad (ugg=ugg)
        grad0 = np.append (gorb0, gci0)
        grad1 = h_op.get_grad ()
        gx1 = h_op.get_gx () # "gx" is not even defined in this context
        self.assertAlmostEqual (lib.fp (grad0), -0.1547272159441664, 7)
        self.assertAlmostEqual (lib.fp (grad1), -0.1547272159441664, 7)
        self.assertAlmostEqual (lib.fp (gx0), 0.0, 9)
        self.assertAlmostEqual (lib.fp (gx1), 0.0, 9) 

    def test_hess_fingerprint (self):
        hx = h_op._matvec (x)
        self.assertAlmostEqual (lib.fp (hx), 88.38116591527076, 7)

    def test_hdiag_orb_fingerprint (self):
        hdiag = itsec (*ugg.unpack (h_op._get_Hdiag ()))
        #refs = np.zeros_like (x)
        #xp = x.copy ()
        #for i in range (len (x)):
        #    xp[:] = 0
        #    xp[i] = 1
        #    refs[i] = h_op (xp)[i]
        #refs = itsec (*ugg.unpack (refs))
        refs = [1.8702702560469948,
                0.33670049827960724,
                7.600706629982519,
                -1.2913164221503841]
        for sec, test, ref in zip (sectors, hdiag, refs):
            with self.subTest (sector=sec):
                self.assertAlmostEqual (lib.fp (test), ref, 6)

    def test_hess_ci2_ci2 (self):
        xp = x.copy ()
        xp[:offs_ci2] = 0.0
        hx = h_op._matvec (xp)[offs_ci2:]
        self.assertAlmostEqual (lib.fp (hx), 1.0889277847142012, 9)

    def test_hess_ci1_ci2 (self):
        xp = x.copy ()
        xp[:offs_ci2] = 0.0
        hx = h_op._matvec (xp)[offs_ci1:offs_ci2]
        self.assertAlmostEqual (lib.fp (hx), -0.00010213222170035406, 9)

    def test_hess_ci_orb (self):
        xp = x.copy ()
        xp[offs_ci1:] = 0.0
        hx = h_op._matvec (xp)[offs_ci1:]
        self.assertAlmostEqual (lib.fp (hx), 0.5539469718844992, 9)

    def test_hess_orb_ci (self):
        xp = x.copy ()
        xp[:offs_ci1] = 0.0
        hx = h_op._matvec (xp)[:offs_ci1]
        self.assertAlmostEqual (lib.fp (hx), -0.3198013307575741, 9)

    def test_hess_orb_orb (self):
        xp = x.copy ()
        xp[offs_ci1:] = 0.0
        hx = h_op._matvec (xp)[:offs_ci1]
        self.assertAlmostEqual (lib.fp (hx), 89.20958173580503, 7)

    def test_hess_from_core_virtual (self):
        xorb0 = xorb.copy ()
        xorb0[ncore:nocc,:] = xorb0[:,ncore:nocc] = 0.0
        xp = ugg.pack (xorb0, xci0)
        hxorb, hxci = ugg.unpack (h_op._matvec (xp))
        refs = [-0.03272038396141794,0.10246438926696706,-38.2707435632963,0.0007154936336222634,-0.023324408608589215,0.011004024379865151]
        for sec, test, ref in zip (sectors, itsec (hxorb,hxci), refs):
            with self.subTest (sector=sec):
                self.assertAlmostEqual (lib.fp (test), ref, 7)

    def test_hess_from_active_active (self):
        xorb0 = np.zeros_like (xorb)
        xorb0[ncore:nocc,ncore:nocc] = xorb[ncore:nocc,ncore:nocc] 
        xp = ugg.pack (xorb0, xci0)
        hxorb, hxci = ugg.unpack (h_op._matvec (xp))
        refs = [0.11842732747281545,0.5180261540565698,0.06792211271902979,0.045360463403717376,0.0015574081837203407,-0.0028940907410843902]
        for sec, test, ref in zip (sectors, itsec (hxorb,hxci), refs):
            with self.subTest (sector=sec):
                self.assertAlmostEqual (lib.fp (test), ref, 9)

    def test_hess_from_active_any (self):
        xorb0 = np.zeros_like (xorb)
        xorb0[ncore:nocc,:] = xorb[ncore:nocc,:] 
        xorb0[:,ncore:nocc] = xorb[:,ncore:nocc] 
        xp = ugg.pack (xorb0, xci0)
        hxorb, hxci = ugg.unpack (h_op._matvec (xp))
        refs = [16.297432789939652,0.4252776476293889,-0.789515435859487,-1.382514838664916,-0.09445253579668796,0.12861532657269764]
        for sec, test, ref in zip (sectors, itsec (hxorb,hxci), refs):
            with self.subTest (sector=sec):
                self.assertAlmostEqual (lib.fp (test), ref, 8)

    def test_prec_fingerprint (self):
        M_op = h_op.get_prec ()
        Mx = M_op._matvec (x)
        self.assertAlmostEqual (lib.fp (Mx), 4.4223425940525, 6)


if __name__ == "__main__":
    print("Full Tests for LASSCF Newton-CG module functions")
    unittest.main()

