#!/usr/bin/env python
#
# Author: Shreya Verma <shreyav@uchicago.edu>

# The tests are performed for the following purposes:
#   1. Check accuracy of spin-summed LASSCF 2-RDMs against spin-summed 2-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   2. Check accuracy of spin-separated LASSCF 2-RDMs against spin-separated 2-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   3. Check accuracy of spin-summed LASSCF 3-RDMs against spin-summed 3-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   4. Check accuracy of spin-separated LASSCF 3-RDMs against spin-separated 3-RDMs created by LAS ci vectors using the PySCF rdm generator- only for H8 with 4 fragments
#   5. Check accuracy of spin-separated LASSCF 3-RDMs by tracing them down to 2-RDMs
#   6. Check all of the above for different sub-spins of LASSCF fragments
# The spin-separated 3-RDM check (4) is in separate _slow methods: building its reference
# with fci.direct_spin1.make_rdm123s takes ~50 s per case, while everything else takes ~2 s.

import unittest

from pyscf import scf, gto, df, lib, fci
from pyscf import mcscf
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF
from pyscf.mcscf.las.addons import las2cas_civec
import numpy as np

SPIN_CASES = {1: (1,1,1,1), 2: (1,3,1,3), 3: (1,3,3,3), 4: (3,3,3,3)}

def generate_geom (natom = 8, dist = 1.0):
    """Generate H chain of natoms each with dist distance linearly"""
    coords = []
    for i in range(natom):
        coords.append(f"H 0.0 0.0 {i*dist}")
    return "\n".join(coords)

def setUpModule():
    global mf, _cache
    xyz = generate_geom (natom = 8, dist = 1.0)
    mol = gto.M(atom=xyz, basis='sto-3g', verbose=0, output='/dev/null')
    mf = scf.RHF(mol).run()
    _cache = {}

def tearDownModule():
    global mf, _cache
    mf.mol.stdout.close ()
    del mf, _cache

def run_las (spin_sub, nelecas_sub=(2,2,2,2)):
    """H8 with 4 fragments. Returns the LASSCF object and the equivalent CAS CI vector and
    nelec. Cached, so the fast and slow tests of one spin case share a kernel."""
    key = (tuple (spin_sub), str (nelecas_sub))
    if key not in _cache:
        las = LASSCF (mf, (2,2,2,2), nelecas_sub, spin_sub=spin_sub)
        mo_loc = las.localize_init_guess (((0, 1), (2, 3), (4, 5), (6,7)), mf.mo_coeff)
        las.kernel(mo_loc)
        # This often won't converge, but the system is nonphysical and it's just a consistency
        # check so who cares?
        ci, nelec = las2cas_civec(las)
        _cache[key] = (las, ci, nelec)
    return _cache[key]

class KnownValues(unittest.TestCase):

    def trace_down_test(self, rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,  rdm1, rdm2, rdm3, rdm1s, rdm2s, rdm3s):
        """ Test for tracing down spin-sep 3-RDMs down to 2-RDMs"""
        nelec = 8
        fac = 1. / (nelec//2-1)
        rdm2s_1a1 = np.einsum('ijkk->ij',rdm2s_las[0])*fac
        rdm2s_1a2 = np.einsum('kkij->ij',rdm2s_las[0])*fac
        rdm2s_1a3 = np.einsum('ijkk->ij',rdm2s_las[1])*(1.0/(nelec//2))
        rdm2s_1b1 = np.einsum('ijkk->ij',rdm2s_las[2])*fac
        rdm2s_1b2 = np.einsum('kkij->ij',rdm2s_las[2])*fac
        rdm2s_1b3 = np.einsum('ijkk->ij',rdm2s_las[1])*(1.0/(nelec//2))

        self.assertAlmostEqual (lib.fp (rdm2s_1a1), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1a2), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1a3), lib.fp (rdm1s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b1), lib.fp (rdm1s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b2), lib.fp (rdm1s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_1b3), lib.fp (rdm1s_las[1]), 12)

        # Tracing down ss 3-RDM to 2-RDM
        nelec = 8

        rdm3s_2a1 = np.einsum('ijklmm->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a2 = np.einsum('ijmmkl->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a3 = np.einsum('mmijkl->ijkl',rdm3s_las[0])*(1.0/(nelec//2-2))
        rdm3s_2a4 = np.einsum('ijklmm->ijkl',rdm3s_las[1])*(1.0/(nelec//2))

        rdm3s_2b1 = np.einsum('ijklmm->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b2 = np.einsum('ijmmkl->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b3 = np.einsum('mmijkl->ijkl',rdm3s_las[3])*(1.0/(nelec//2-2))
        rdm3s_2b4 = np.einsum('mmijkl->ijkl',rdm3s_las[2])*(1.0/(nelec//2))

        rdm3s_2ab1 = np.einsum('ijmmkl->ijkl',rdm3s_las[1])*(1.0/(nelec//2-1))
        rdm3s_2ab2 = np.einsum('mmijkl->ijkl',rdm3s_las[1])*(1.0/(nelec//2-1))
        rdm3s_2ab3 = np.einsum('ijklmm->ijkl',rdm3s_las[2])*(1.0/(nelec//2-1))
        rdm3s_2ab4 = np.einsum('ijmmkl->ijkl',rdm3s_las[2])*(1.0/(nelec//2-1))

        self.assertAlmostEqual (lib.fp (rdm3s_2a1), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a2), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a3), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2a4), lib.fp (rdm2s_las[0]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b1), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b2), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b3), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2b4), lib.fp (rdm2s_las[2]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab1), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab2), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab3), lib.fp (rdm2s_las[1]), 12)
        self.assertAlmostEqual (lib.fp (rdm3s_2ab4), lib.fp (rdm2s_las[1]), 12)

    def _check_fast (self, spin_sub, nelecas_sub=(2,2,2,2), trace_down=True):
        las, ci, nelec = run_las (spin_sub, nelecas_sub)
        norb = 8
        rdm1_las, rdm1s_las = las.make_casdm1(), las.make_casdm1s()
        rdm2_las, rdm2s_las = las.make_casdm2(), las.make_casdm2s()
        rdm3_las, rdm3s_las = las.make_casdm3(), las.make_casdm3s()
        rdm1, rdm2, rdm3 = fci.rdm.reorder_dm123 (*fci.rdm.make_dm123 (
            'FCI3pdm_kern_sf', ci, ci, norb, nelec[0])) # normal ordered RDMs
        rdm1s, rdm2s = fci.direct_spin1.make_rdm12s (np.array (ci), norb, nelec[0])

        # Direct tests with ci generated RDMs
        self.assertAlmostEqual (lib.fp (rdm1_las), lib.fp (rdm1), 12)
        self.assertAlmostEqual (lib.fp (rdm1s_las), lib.fp (rdm1s), 12)
        self.assertAlmostEqual (lib.fp (rdm2_las), lib.fp (rdm2), 12)
        self.assertAlmostEqual (lib.fp (rdm2s_las), lib.fp (rdm2s), 12)
        self.assertAlmostEqual (lib.fp (rdm3_las), lib.fp (rdm3), 12)
        # Tracing down tests
        if trace_down:
            self.trace_down_test(rdm1_las, rdm2_las, rdm3_las, rdm1s_las, rdm2s_las, rdm3s_las,
                                 None, None, None, None, None, None)

    def _check_rdm3s (self, spin_sub, nelecas_sub=(2,2,2,2)):
        las, ci, nelec = run_las (spin_sub, nelecas_sub)
        rdm3s = fci.direct_spin1.make_rdm123s (np.array (ci), 8, nelec[0])[2]
        self.assertAlmostEqual (lib.fp (las.make_casdm3s ()), lib.fp (rdm3s), 12)

    def test_rdm3_h8_sto3g_spin1 (self):
        """Spin cas 1"""
        self._check_fast (SPIN_CASES[1])

    def test_rdm3s_h8_sto3g_spin1_slow (self):
        """Spin cas 1: spin-separated 3-RDM against FCI"""
        self._check_rdm3s (SPIN_CASES[1])

    def test_rdm3_h8_sto3g_spin2 (self):
        """Spin cas 2"""
        self._check_fast (SPIN_CASES[2])

    def test_rdm3s_h8_sto3g_spin2_slow (self):
        """Spin cas 2: spin-separated 3-RDM against FCI"""
        self._check_rdm3s (SPIN_CASES[2])

    def test_rdm3_h8_sto3g_spin3 (self):
        """Spin cas 3"""
        self._check_fast (SPIN_CASES[3])

    def test_rdm3s_h8_sto3g_spin3_slow (self):
        """Spin cas 3: spin-separated 3-RDM against FCI"""
        self._check_rdm3s (SPIN_CASES[3])

    def test_rdm3_h8_sto3g_spin4 (self):
        """Spin cas 4"""
        self._check_fast (SPIN_CASES[4])

    def test_rdm3s_h8_sto3g_spin4_slow (self):
        """Spin cas 4: spin-separated 3-RDM against FCI"""
        self._check_rdm3s (SPIN_CASES[4])

# Every case above has M=0 fragments, so N_alpha = N_beta and the hard-coded trace-down factors
# never see a high-spin case. Here N_alpha = 6, N_beta = 2. (Until fixed, the inter-fragment
# Coulomb ab block of make_casdm2s and two exchange terms of the aab block of make_casdm3s used
# the wrong fragment's alpha/beta density when fragments differed in spin polarization.)
HIGH_SPIN = dict (spin_sub=(3,1,3,1), nelecas_sub=((2,0),(1,1),(2,0),(1,1)))

def _trace_down_checks (las):
    na, nb = las.nelecas
    dm1a, dm1b = las.make_casdm1s ()
    dm2aa, dm2ab, dm2bb = las.make_casdm2s ()
    dm3aaa, dm3aab, dm3abb, dm3bbb = las.make_casdm3s ()
    same = [('aa', 'ijkk->ij', dm2aa, (na-1)*dm1a), ('aa', 'kkij->ij', dm2aa, (na-1)*dm1a),
            ('bb', 'ijkk->ij', dm2bb, (nb-1)*dm1b), ('bb', 'kkij->ij', dm2bb, (nb-1)*dm1b)]
    same += [('aaa', sub, dm3aaa, (na-2)*dm2aa) for sub in ('ijklmm->ijkl', 'ijmmkl->ijkl',
                                                            'mmijkl->ijkl')]
    same += [('bbb', sub, dm3bbb, (nb-2)*dm2bb) for sub in ('ijklmm->ijkl', 'ijmmkl->ijkl',
                                                            'mmijkl->ijkl')]
    mixed = [('ab', 'ijkk->ij', dm2ab, nb*dm1a), ('ab', 'kkij->ij', dm2ab, na*dm1b),
             ('aab', 'ijklmm->ijkl', dm3aab, nb*dm2aa),
             ('aab', 'ijmmkl->ijkl', dm3aab, (na-1)*dm2ab),
             ('aab', 'mmijkl->ijkl', dm3aab, (na-1)*dm2ab),
             ('abb', 'mmijkl->ijkl', dm3abb, na*dm2bb),
             ('abb', 'ijklmm->ijkl', dm3abb, (nb-1)*dm2ab),
             ('abb', 'ijmmkl->ijkl', dm3abb, (nb-1)*dm2ab)]
    return same, mixed

class HighSpin (unittest.TestCase):
    _check_fast = KnownValues._check_fast
    _check_rdm3s = KnownValues._check_rdm3s

    def _check_traces (self, checks):
        errs = {(name, sub): np.abs (np.einsum (sub, dm_hi) - ref).max ()
                for name, sub, dm_hi, ref in checks}
        self.assertEqual ({k: v for k, v in errs.items () if v > 1e-10}, {})

    def test_rdm_h8_sto3g_high_spin_nelec (self):
        las = run_las (**HIGH_SPIN)[0]
        self.assertEqual (tuple (las.nelecas), (6, 2))

    def test_rdm_h8_sto3g_high_spin_trace_down_same_spin (self):
        self._check_traces (_trace_down_checks (run_las (**HIGH_SPIN)[0])[0])

    def test_rdm_h8_sto3g_high_spin_trace_down_mixed_spin (self):
        self._check_traces (_trace_down_checks (run_las (**HIGH_SPIN)[0])[1])

    def test_rdm_h8_sto3g_high_spin (self):
        self._check_fast (HIGH_SPIN['spin_sub'], HIGH_SPIN['nelecas_sub'], trace_down=False)

    def test_rdm3s_h8_sto3g_high_spin_slow (self):
        self._check_rdm3s (HIGH_SPIN['spin_sub'], HIGH_SPIN['nelecas_sub'])

class MixedSpinH6 (unittest.TestCase):
    '''Spin-separated 2- and 3-RDMs against FCI for every assignment of (2,0), (1,1) and (0,2) to
    three fragments. A 3-RDM involves at most three fragments, so this reaches every term.'''

    def test_rdm23s_h6_sto3g_all_polarizations (self):
        import itertools
        mol6 = gto.M (atom=generate_geom (natom=6), basis='sto-3g', verbose=0, output='/dev/null')
        mf6 = scf.RHF (mol6).run ()
        smult = {(2,0): 3, (1,1): 1, (0,2): 3}
        for nelecas_sub in itertools.product (smult, repeat=3):
            las = LASSCF (mf6, (2,2,2), nelecas_sub, spin_sub=[smult[n] for n in nelecas_sub])
            # Only consistency with the CI vectors matters, so skip orbital optimization
            las.lasci (las.localize_init_guess (((0,1),(2,3),(4,5)), mf6.mo_coeff))
            ci, nelec = las2cas_civec (las)
            ref2, ref3 = fci.direct_spin1.make_rdm123s (np.array (ci), 6, nelec[0])[1:]
            dm2s, dm3s = las.make_casdm2s (), las.make_casdm3s ()
            with self.subTest (nelecas_sub=nelecas_sub):
                for spin in range (3):
                    self.assertAlmostEqual (np.abs (dm2s[spin] - ref2[spin]).max (), 0, 10)
                for spin in range (4):
                    self.assertAlmostEqual (np.abs (dm3s[spin] - ref3[spin]).max (), 0, 10)
        mol6.stdout.close ()

class StateAverage (unittest.TestCase):
    '''State-averaged 2- and 3-RDMs against the weighted average of the FCI RDMs of each
    rootspace's product state. The rootspaces differ in fragment spin from the reference and from
    each other, and the weights are unequal. (Until fixed, make_casdm3 and make_casdm3s built the
    diagonal blocks from the reference nelecas_sub without averaging over rootspaces, so they
    raised an exception for any state average, and contracted the weights twice in the
    off-diagonal blocks.)'''

    def test_rdm23_h6_sto3g_state_average (self):
        mol6 = gto.M (atom=generate_geom (natom=6), basis='sto-3g', verbose=0, output='/dev/null')
        mf6 = scf.RHF (mol6).run ()
        las = LASSCF (mf6, (2,2,2), (2,2,2))
        las.state_average_(weights=[0.5,0.3,0.2], spins=[[0,0,0],[2,0,0],[0,-2,2]],
                           smults=[[1,1,1],[3,1,1],[1,3,3]])
        # Only consistency with the CI vectors matters, so skip orbital optimization
        las.lasci (las.localize_init_guess (((0,1),(2,3),(4,5)), mf6.mo_coeff))
        ref = {'dm2': 0, 'dm2s': 0, 'dm3': 0, 'dm3s': 0}
        for r, w in enumerate (las.weights):
            las1 = las.get_single_state_las (state=r)
            las1.ci = [[ci_f[r]] for ci_f in las.ci]
            ci, nelec = las2cas_civec (las1)
            ci, nelec = ci[0], tuple (int (n) for n in nelec[0])
            dm2s, dm3s = fci.direct_spin1.make_rdm123s (ci, 6, nelec)[1:]
            dm3 = fci.direct_spin1.make_rdm123 (ci, 6, nelec)[2]
            ref['dm2'] += w * (dm2s[0] + dm2s[1] + dm2s[1].transpose (2,3,0,1) + dm2s[2])
            ref['dm2s'] += w * np.stack (dm2s)
            ref['dm3'] += w * dm3
            ref['dm3s'] += w * np.stack (dm3s)
        for key, dm in (('dm2', las.make_casdm2 ()), ('dm2s', las.make_casdm2s ()),
                        ('dm3', las.make_casdm3 ()), ('dm3s', las.make_casdm3s ())):
            with self.subTest (key):
                self.assertEqual (np.shape (dm), ref[key].shape)
                self.assertAlmostEqual (np.abs (dm - ref[key]).max (), 0, 10)
        mol6.stdout.close ()

if __name__ == "__main__":
    print("Full Tests for LASSCF 3-RDMs")
    unittest.main()
