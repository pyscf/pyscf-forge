#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

# Guards for behavior of lasscf_rdm that lassqd_code relies on, which isn't otherwise part of any
# documented API. lassqd replaces each fragment's solver with a custom kernel
#   kernel (norb, nelec, h0, h1s, h2) -> (e, dm1s, dm2)
# and enters las.kernel twice per cycle:
#   1. "capture": record every fragment's Hamiltonian, then raise out of the kernel from the last
#      fragment. This works only because every fragment's Hamiltonian is built before any
#      fragment's result is used.
#   2. "replay": with max_cycle_macro=1 and max_cycle_rdmjk=0, solve each fragment exactly once,
#      then take one orbital step and canonicalize. The step is taken only because the RDM-jk loop
#      reports "not converged" after zero cycles.
# lassqd also computes LAS-PDFT energies by monkeypatching the RDM makers of
# get_mcpdft_child_class (las, ot). If any of this has to change, give lassqd an API first.

import unittest
import tempfile
import h5py
import numpy as np
from pyscf import gto, scf, lib
from pyscf.mcscf.las import lasscf_rdm, _DFLASCI
from pyscf.mcscf.las.lasscf import LASSCF as LASSCF_CI
from pyscf.mcpdft.laspdft import get_mcpdft_child_class

def setUpModule():
    global mol, mf, mo, dm1, dm2
    xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0
             H 0.2 3.9 0.1
             H 1.159166 4.1 -0.1'''
    mol = gto.M (atom=xyz, basis='sto-3g', verbose=0, output='/dev/null')
    mf = scf.RHF (mol).run ()
    las = make_las ()
    mo = las.localize_init_guess (((0,1),(2,3)), mf.mo_coeff)
    dm1, dm2 = lasscf_rdm.get_init_guess_rdm (las, mo, las.get_h2eff (mo))
    dm1 = [np.stack (d) for d in dm1]
    dm2 = [np.stack (d) for d in dm2]

def tearDownModule():
    global mol, mf, mo, dm1, dm2
    mol.stdout.close ()
    del mol, mf, mo, dm1, dm2

def make_las ():
    return lasscf_rdm.LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))

def set_fragment_kernels (las, kernels):
    '''As in lassqd: keep each fragment's spin and smult (which dump_chk needs)'''
    las.fciboxes = [lasscf_rdm.make_fcibox (las.mol, kernel=kernel,
                                            spin=fcibox.fcisolvers[0].spin,
                                            smult=fcibox.fcisolvers[0].smult)
                    for kernel, fcibox in zip (kernels, las.fciboxes)]

def default_solver ():
    return lasscf_rdm.make_fcibox (mol, spin=0, smult=1).fcisolvers[0]

def kernel_from_guess (las):
    las.kernel (mo.copy (), casdm1frs=[d.copy () for d in dm1], casdm2fr=[d.copy () for d in dm2])

class _Captured (Exception):
    pass

def capture (garbage_frag0=False):
    '''Record every fragment's (h0, h1s, h2); stop in the last fragment. Optionally have
    fragment 0 return garbage, to check that fragment 1 doesn't see it.'''
    las = make_las ()
    hams = [None, None]
    solver = default_solver ()
    def make_kernel (ifrag):
        def kernel (norb, nelec, h0, h1s, h2):
            hams[ifrag] = (h0, np.array (h1s), np.array (h2))
            if ifrag == 1: raise _Captured
            if garbage_frag0:
                return 0.0, np.zeros ((2,norb,norb)), np.zeros ((norb,)*4)
            return solver.kernel (norb, nelec, h0, h1s, h2)
        return kernel
    set_fragment_kernels (las, [make_kernel (0), make_kernel (1)])
    try:
        kernel_from_guess (las)
    except _Captured:
        pass
    else:
        raise RuntimeError ("las.kernel returned without calling every fragment kernel")
    return hams

class KnownValues (unittest.TestCase):

    def test_factory (self):
        self.assertIsInstance (lasscf_rdm.LASSCF (mol, (2,2), (2,2)), lasscf_rdm.LASSCFNoSymm)
        las_df = lasscf_rdm.LASSCF (scf.RHF (mol).density_fit (), (2,2), (2,2))
        self.assertIsInstance (las_df, _DFLASCI)

    def test_capture (self):
        hams = capture ()
        for ifrag, (h0, h1s, h2) in enumerate (hams):
            with self.subTest (frag=ifrag):
                self.assertEqual (h0, 0.0)
                self.assertEqual (h1s.shape, (2,2,2))
                self.assertEqual (h2.shape, (2,2,2,2)) # ao2mo.restore (1, ...)
        # Every fragment's Hamiltonian is built before any fragment's result is used
        hams_garbage = capture (garbage_frag0=True)
        self.assertEqual (lib.fp (hams[1][1]), lib.fp (hams_garbage[1][1]))
        self.assertEqual (lib.fp (hams[1][2]), lib.fp (hams_garbage[1][2]))

    def test_replay (self):
        las = make_las ()
        las.max_cycle_macro, las.max_cycle_rdmjk = 1, 0
        las.chkfile = tempfile.NamedTemporaryFile (suffix='.chk').name
        visits = [0, 0]
        solved = [None, None]
        solver = default_solver ()
        def make_kernel (ifrag):
            def kernel (norb, nelec, h0, h1s, h2):
                visits[ifrag] += 1
                solved[ifrag] = solver.kernel (norb, nelec, h0, h1s, h2)
                return solved[ifrag]
            return kernel
        set_fragment_kernels (las, [make_kernel (0), make_kernel (1)])
        kernel_from_guess (las)
        with self.subTest ('each fragment solved exactly once'):
            self.assertEqual (visits, [1, 1])
        with self.subTest ('RDM-jk loop reports not converged after zero cycles'):
            self.assertFalse (las.converged)
        with self.subTest ('one orbital step was taken'):
            casdm1frs = [np.asarray (s[1])[None] for s in solved]
            casdm2fr = [np.asarray (s[2])[None] for s in solved]
            e_before = las.energy_nuc () + las.energy_elec (mo_coeff=mo, casdm1frs=casdm1frs,
                                                            casdm2fr=casdm2fr)
            self.assertLess (las.e_tot, e_before)
            self.assertAlmostEqual (las.e_tot, -2.2022821864333215, 8)
        with self.subTest ('canonicalized RDMs are consistent with the returned orbitals'):
            e_after = las.energy_nuc () + las.energy_elec (mo_coeff=las.mo_coeff,
                                                           casdm1frs=las.casdm1frs,
                                                           casdm2fr=las.casdm2fr)
            self.assertAlmostEqual (e_after, las.e_tot, 10)
        with self.subTest ('dump_chk works with custom fragment kernels'):
            with h5py.File (las.chkfile, 'r') as f:
                self.assertIn ('las', f)
                self.assertIn ('mo_coeff', f['las'])

    def test_pdft_monkeypatch (self):
        las_ci = LASSCF_CI (mf, (2,2), (2,2), spin_sub=(1,1))
        las_ci.kernel (las_ci.localize_init_guess (((0,1),(2,3)), mf.mo_coeff))
        self.assertTrue (las_ci.converged)
        ref = get_mcpdft_child_class (las_ci, 'tPBE', grids_level=1).compute_pdft_energy_ (
            las_ci.mo_coeff)
        pdft = get_mcpdft_child_class (make_las (), 'tPBE', grids_level=1)
        casdm1s, casdm2 = las_ci.make_casdm1s (), las_ci.make_casdm2 ()
        pdft.make_one_casdm1s = lambda ci=None, state=0, **kwargs: casdm1s
        pdft.make_one_casdm2 = lambda ci=None, state=0, **kwargs: casdm2
        e_tot, e_ot, e_states = pdft.compute_pdft_energy_ (las_ci.mo_coeff)
        self.assertAlmostEqual (e_tot, ref[0], 9)
        self.assertAlmostEqual (e_ot, ref[1], 9)
        self.assertAlmostEqual (e_states[0], ref[2][0], 9)

if __name__ == "__main__":
    print("Full Tests for lasscf_rdm behavior that lassqd_code relies on")
    unittest.main()
