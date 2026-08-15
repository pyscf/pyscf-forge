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

'''
Test Description: In this test, we validate the functionality of 
SISO model spaces.
Test-1: Check accumulation of symmetry sectors having the same spin.
Test-2: Check grouping of model-space entries by spin.
Test-3: Check aggregation of symmetry-resolved sectors by spin.
Test-4: Check the symmetry-resolved solver interface.
Test-5: Check a symmetry-resolved SISO calculation.
Test-6: Reject invalid model-space entries.
Test-7: Reject a model-space root count inconsistent with the MC object.
Test-8: Validate SISO integral and Hamiltonian options.
Test-9: Check state-weight assignment across the model space.
Test-10: Check model-space NEVPT2 energies and automatic MC updates.
'''

import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.siso import siso, socaddons


class FakeFCISolver:
    def __init__(self, ss):
        self.ss = np.asarray(ss)

    def states_spin_square(self, ci, ncas, nelecas):
        return self.ss, None


def make_mc(mol, ss):
    nroots = len(ss)
    return SimpleNamespace(
        _scf=SimpleNamespace(mol=mol),
        ncas=4,
        nelecas=(1, 0),
        ci=[np.zeros((4, 1)) for _ in range(nroots)],
        e_states=np.arange(nroots, dtype=float),
        fcisolver=FakeFCISolver(ss),
        stdout=io.StringIO(),
        verbose=0,
    )


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='Li 0 0 0', spin=1, basis='sto-3g',
            symmetry='D2h', verbose=0)

    def test_symmetry_sectors_with_same_spin_are_accumulated(self):
        modelspace = [
            (1, 2, 'Ag'),
            (1, 2, 'B1u'),
            (1, 2, 'B2u'),
            (1, 2, 'B3u'),
        ]
        mc = make_mc(self.mol, [0.75] * 4)
        mysiso = siso.SISO(mc, modelspace)
        self.assertEqual(mysiso.modelspace, modelspace)
        self.assertEqual(mysiso.statelis, [0, 4])
        self.assertEqual(mysiso.twoslst.tolist(), [1])

    def test_modelspace_is_grouped_by_spin(self):
        states = socaddons._validate_modelspace(
            [(1, 3, 'Ag'), (2, 1, 'B1g')], mol=self.mol,
            ncas=4, nelecas=(2, 0))
        self.assertEqual(states, [(2, 1, 'B1g'), (1, 3, 'Ag')])

    def test_symmetry_sectors_are_aggregated_by_spin(self):
        states = [(1, 1, 'Ag'), (2, 1, 'B1g'), (1, 3, 'B2g')]
        self.assertEqual(
            socaddons._aggregate_modelspace(states), [(3, 1), (1, 3)])

    def test_symmetry_resolved_solver_api(self):
        modelspace = [(1, 2, 'Ag'), (1, 2, 'B1u')]
        mc = mcscf.CASSCF(scf.ROHF(self.mol), 2, 1)
        mc = socaddons.sacasscf_solver(mc, modelspace)

        self.assertEqual(len(mc.fcisolver.fcisolvers), 2)
        self.assertEqual(
            [solver.wfnsym for solver in mc.fcisolver.fcisolvers],
            ['Ag', 'B1u'])
        self.assertEqual(
            [solver.spin for solver in mc.fcisolver.fcisolvers], [1, 1])
        self.assertTrue(np.allclose(mc.weights, 0.5))

    def test_symmetry_resolved_siso(self):
        mol = gto.M(
            atom='H 0 0 -0.7; H 0 0 0.7', basis='sto-3g',
            symmetry='D2h', verbose=0)
        mf = scf.RHF(mol).run()
        modelspace = [(1, 1, 'Ag'), (1, 1, 'B1u')]
        mc = mcscf.CASSCF(mf, 2, 2)
        mc = socaddons.sacasscf_solver(mc, modelspace).run()

        my_siso = siso.SISO(mc, modelspace, ham='BP')
        energies, si_vecs = my_siso.kernel()

        self.assertEqual(my_siso.statelis, [2])
        self.assertEqual(my_siso.imds.c[0].shape, (2, 2, 2))
        self.assertEqual(si_vecs.shape, (2, 2))
        self.assertTrue(np.allclose(energies, mc.e_states))

    def test_invalid_modelspace_entries(self):
        invalid = [
            ([], TypeError),
            ([(0, 2)], ValueError),
            ([(1.5, 2)], TypeError),
            ([(1, 0)], ValueError),
            ([(1, 1), (1, 2)], ValueError),
            ([(1, 2, 'not_an_irrep')], ValueError),
            ([(1, 2), (1, 2, 'Ag')], ValueError),
            ([(1, 2, 'Ag'), (1, 2, 'Ag')], ValueError),
        ]
        for modelspace, exception in invalid:
            with self.subTest(modelspace=modelspace):
                with self.assertRaises(exception):
                    socaddons._validate_modelspace(
                        modelspace, mol=self.mol, ncas=4, nelecas=(1, 0))

    def test_modelspace_root_count_must_match_mc(self):
        mc = make_mc(self.mol, [0.75])
        with self.assertRaisesRegex(ValueError, 'mc.e_states contains 1'):
            siso.SISO(mc, [(2, 2)])

    def test_siso_options_are_validated(self):
        invalid_options = [
            ({'ham': 'invalid'}, ValueError),
            ({'amf': False, 'mmf': False}, ValueError),
            ({'amf': True, 'mmf': True}, ValueError),
            ({'soc1e': False, 'soc2e': False}, ValueError),
            ({'somf': False}, NotImplementedError),
            ({'soc1e': 1}, TypeError),
        ]
        for options, exception in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(exception):
                    siso.SISO(make_mc(self.mol, [0.75]), [(1, 2)],
                              **options)

    def test_state_weights(self):
        weights = socaddons._validate_state_weights(None, 4)
        self.assertTrue(np.allclose(weights, 0.25))
        with self.assertRaisesRegex(ValueError, 'one value for each'):
            socaddons._validate_state_weights([0.5, 0.5], 3)
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            socaddons._validate_state_weights([0.2, 0.2, 0.2], 3)

    def test_compute_nevpt2_energies_updates_mc(self):
        casci_objects = []

        class FakeCASCI:
            def __init__(self, mf, ncas, nelecas, ncore=None):
                self.nelecas = nelecas
                casci_objects.append(self)

            def kernel(self, mo_coeff):
                nroots = self.fcisolver.nroots
                energies = 10 * self.fcisolver.spin + np.arange(nroots)
                return energies, None

        class FakeNEVPT:
            def __init__(self, casci, root):
                self.root = root

            def kernel(self):
                return 0.1 * (self.root + 1)

        mc = SimpleNamespace(
            _scf=SimpleNamespace(mol=self.mol),
            ncas=5,
            nelecas=(3, 2),
            ncore=0,
            mo_coeff=np.eye(5),
            e_states=np.zeros(3),
        )
        original_e_states = mc.e_states
        modelspace = [(1, 4, 'Ag'), (2, 2, 'B1u')]

        with mock.patch.object(socaddons.mcscf, 'CASCI', FakeCASCI), \
                mock.patch.object(
                    socaddons, 'csf_solver',
                    side_effect=lambda mol, smult: SimpleNamespace()), \
                mock.patch.object(socaddons.mrpt, 'NEVPT', FakeNEVPT):
            energies = socaddons.compute_nevpt2_energies(mc, modelspace)

        self.assertTrue(np.allclose(energies, [10.1, 11.2, 30.1]))
        self.assertIs(mc.e_states, original_e_states)
        self.assertIs(energies, original_e_states)
        self.assertEqual([obj.nelecas for obj in casci_objects],
                         [(3, 2), (4, 1)])
        self.assertEqual([obj.fcisolver.wfnsym for obj in casci_objects],
                         ['B1u', 'Ag'])


if __name__ == '__main__':
    unittest.main()
