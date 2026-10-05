#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Tests for the GASSCF module."""

from contextlib import ExitStack
from functools import reduce
import io
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

import numpy
import scipy.linalg

from pyscf import ao2mo
from pyscf import df
from pyscf import dft
from pyscf import gto
from pyscf import lib
from pyscf import mcscf
from pyscf import scf
from pyscf import solvent
from pyscf.tools import molden
from pyscf.fci import addons as fci_addons
from pyscf.fci import direct_spin1
from pyscf.fci import spin_op
from pyscf.mcscf import addons
from pyscf.mcscf import addons_gas
from pyscf.mcscf import df as mcdf
from pyscf.mcscf import fci_gas
from pyscf.mcscf import gasci
from pyscf.mcscf import mc1step
from pyscf.mcscf import newton_casscf
from pyscf.mcscf import gasscf
from pyscf.mcscf.test._gasscf_n2_fixture import N2_GASSCF_MOL, N2_GASSCF_MO_COEFF


class _N2Fixture:
    """Shared N2 builders and assertions; intentionally contains no tests."""

    N2_REGRESSION_TOL = 1e-7

    N2_OPENMOLCAS_GASSCF_ENERGY = -109.0174765

    N2_REF_FIXED_ENERGY = -109.017476499975

    N2_REF_SS_ENERGY = -109.01747649998

    N2_REF_SA_HALF_HALF_ENERGY = -108.782370704347

    N2_REF_DF_ENERGY = -109.017467767868

    N2_REF_SCANNER_ENERGY = -109.033632423671

    def _n2_regression_fixture(self):
        mol = gto.loads(N2_GASSCF_MOL)
        mol.verbose = 0
        mo = numpy.array(N2_GASSCF_MO_COEFF, copy=True)

        if mo.shape[0] != mol.nao_nr() or mo.shape[1] <= 10:
            raise AssertionError("embedded N2 GASSCF fixture lost virtual orbitals")

        s = mol.intor_symmetric("int1e_ovlp")
        numpy.testing.assert_allclose(
            mo.T @ s @ mo,
            numpy.eye(mo.shape[1]),
            atol=1e-9,
            rtol=0,
        )

        mf = scf.RHF(mol)
        self.addCleanup(mf._chkfile.close)
        mf.conv_tol = 1e-12
        mf.max_cycle = 100
        mf.kernel()
        if not mf.converged:
            raise AssertionError("N2 regression RHF did not converge")
        mf.mo_coeff = mo.copy()
        return mol, mf, mo

    def _n2_regression_mc(self, mf):
        mc = gasscf.GASSCF(
            mf, 8, (5, 5),
            gas_orbs=(2, 4, 2),
            gas_restr=((2, 4), (7, 9), (10, 10)),
            gas_restr_type="cumulative-occ",
            ncore=2,
        )
        self.addCleanup(mc.close)
        mc.verbose = 0
        mc.canonicalization = False
        mc.max_cycle_macro = 50
        mc.max_cycle_micro = 10
        mc.conv_tol = 1e-10
        mc.conv_tol_grad = 1e-5
        mc.fcisolver.spin = 0
        mc.fcisolver.max_cycle = 300
        mc.fcisolver.max_space = 30
        mc.fcisolver.conv_tol = 1e-10
        return mc

    def _n2_property_pair(self):
        _, mf, mo = self._n2_regression_fixture()
        mc = self._n2_regression_mc(mf)
        ref = gasci.GASCI(
            mf, mc.ncas, mc.nelecas, ncore=mc.ncore,
            gas_orbs=mc.gas_orbs, gas_restr=mc.gas_restr,
            gas_restr_type=mc.gas_restr_type)
        # Keep the embedded MO order. Orthonormal CI probes isolate property
        # dispatch from eigensolver convergence and give distinct root DMs.
        ndet = mc.gas_space_info()["core"]["ndet"]
        probes = numpy.random.default_rng(211).normal(size=(ndet, 2))
        probes = numpy.linalg.qr(probes)[0]
        roots = [probes[:, i].copy() for i in range(2)]
        for obj in (mc, ref):
            obj.mo_coeff = mo.copy()
            obj.ci = [ci.copy() for ci in roots]
        return mc, ref, roots

    def _assert_property_close(self, actual, expected, *, atol=2e-11):
        if isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, e in zip(actual, expected):
                self._assert_property_close(a, e, atol=atol)
        else:
            numpy.testing.assert_allclose(actual, expected, atol=atol, rtol=0)


class TestInputs(unittest.TestCase):
    """Construction, supported inputs and early capability rejection."""

    def test_constructor_gasci_model_and_defaults(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        self.addCleanup(mf._chkfile.close)
        model = (2, (1, 1), (1, 1), ((1, 1), (2, 2)), 'cumulative-occ')
        reference = gasci.GASCI(mf, *model, ncore=0)
        cases = (
            ('positional', lambda: gasscf.GASSCF(mf, *model, ncore=0, frozen=[0])),
            ('keyword', lambda: gasscf.GASSCF(
                mf=mf, ncas=2, nelecas=(1, 1), gas_orbs=(1, 1),
                gas_restr=((1, 1), (2, 2)), gas_restr_type='cumulative-occ', ncore=0)),
            ('list', lambda: gasscf.GASSCF(
                mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
                gas_restr_type='cumulative-occ', ncore=0)))
        ref_orbs, ref_blocks = reference._normalized_restriction()
        for name, construct in cases:
            with self.subTest(form=name):
                obj = construct()
                self.addCleanup(obj.close)
                self.assertEqual((obj.ncas, obj.nelecas, obj.ncore, obj.ngas),
                                 (reference.ncas, reference.nelecas, 0, 2))
                self.assertIsInstance(obj.fcisolver, fci_gas.FCISolver)
                self.assertTrue(obj.cache_plans)
                for holder in (obj, obj.fcisolver):
                    self.assertEqual(holder.gas_orbs, (1, 1))
                    self.assertEqual(holder.gas_restr_type, 'cumulative-occ')
                    restriction = [[1, 1], [2, 2]] if name == 'list' else model[3]
                    self.assertEqual(holder.gas_restr, restriction)
                actual_orbs, actual_blocks = obj._normalized_restriction()
                self.assertEqual(actual_orbs, ref_orbs)
                numpy.testing.assert_array_equal(actual_blocks, ref_blocks)
                if name == 'positional':
                    self.assertEqual(obj.frozen, [0])
        default = gasscf.GASSCF(mf, 2, (1, 1), gas_orbs=(2,), ncore=0,
                                cache_plans=False)
        self.addCleanup(default.close)
        self.assertEqual(default.gas_restr_type, addons_gas.GAS_RESTR_SPIN_SUPERGROUP)
        self.assertFalse(default.cache_plans)

    def test_constructor_default_single_gas_matches_cas(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        reference = newton_casscf.CASSCF(mf, 2, 2).run()
        mc = gasscf.GASSCF(mf, numpy.int64(2), 2).run()
        self.assertEqual(mc.gas_orbs, (2,))
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_tot, reference.e_tot, places=10)

    def test_constructor_rejects_invalid_or_missing_arguments(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        self.addCleanup(mf._chkfile.close)
        solver = fci_gas.FCISolver(mol, gas_orbs=(2,))
        cases = [((3, (1, 1)), model, ValueError, 'ncas.*sum')
                 for model in ({'gas_orbs': (1, 1)}, {'fcisolver': solver})]
        cases += [((bad, (1, 1)), {'gas_orbs': (2,)}, TypeError, 'ncas.*integer')
                  for bad in (True, numpy.bool_(True), 2.0, '2', (1, 1))]
        cases += [((bad, (1, 1)), {'gas_orbs': (2,)}, ValueError, 'ncas.*positive')
                  for bad in (0, -2)]
        cases += [((2,), {'gas_orbs': (2,)}, TypeError, 'requires nelecas'),
                  ((), {'nelecas': (1, 1), 'ncore': 0}, ValueError, 'gas_orbs is required')]
        for factory in (gasscf.GASSCF, gasscf.DFGASSCF):
            for args, kwargs, error, message in cases:
                with self.subTest(factory=factory.__name__, args=args, kwargs=kwargs):
                    with self.assertRaisesRegex(error, message):
                        factory(mf, *args, **kwargs)
        self.assertEqual(solver.gas_orbs, (2,))
        self.assertFalse(hasattr(solver, 'cache_plans'))

    def test_constructor_explicit_and_inferred_ncas(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        self.addCleanup(mf._chkfile.close)
        for factory, reference in ((gasscf.GASSCF, mf),
                                   (gasscf.DFGASSCF, mf.density_fit(auxbasis='weigend'))):
            cases = [
                ('explicit', (2, (1, 1)), dict(gas_orbs=(2,), ncore=0)),
                ('inferred', (), dict(gas_orbs=(2,), nelecas=(1, 1), ncore=0))]
            if factory is gasscf.DFGASSCF:
                cases.append(('default', (), dict(ncas=2, nelecas=(1, 1),
                                                  with_df=reference.with_df)))
            normalized = []
            for name, args, kwargs in cases:
                with self.subTest(factory=factory.__name__, form=name):
                    obj = factory(reference, *args, **kwargs)
                    self.addCleanup(obj.close)
                    self.assertEqual((obj.ncas, obj.nelecas, obj.gas_orbs),
                                     (2, (1, 1), (2,)))
                    normalized.append(obj._normalized_restriction())
                    if factory is gasscf.DFGASSCF:
                        self.assertIs(obj.with_df, reference.with_df)
            for orbs, blocks in normalized[1:]:
                self.assertEqual(orbs, normalized[0][0])
                numpy.testing.assert_array_equal(blocks, normalized[0][1])
        solver = fci_gas.FCISolver(mol, gas_orbs=(2,))
        adapted = gasscf.GASSCF(mf, nelecas=(1, 1), fcisolver=solver)
        self.addCleanup(adapted.close)
        self.assertEqual(adapted.ncas, 2)
        self.assertIsNot(adapted.fcisolver, solver)

    def test_gas_orbs_uses_strict_gasci_integer_validation(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75",
                    basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)

        for bad in ((2.5,), (True,)):
            with self.subTest(path="constructor", gas_orbs=bad):
                with self.assertRaisesRegex(TypeError, "gas_orbs.*integers"):
                    gasscf.GASSCF(
                        mf, 2, (1, 1), gas_orbs=bad, gas_restr=None,
                        ncore=0)

        solver = fci_gas.FCISolver(mol, gas_orbs=(2,))
        solver.gas_orbs = (2.5,)
        with self.assertRaisesRegex(TypeError, "gas_orbs.*integers"):
            gasscf.GASSCF(
                mf, 2, (1, 1), fcisolver=solver, ncore=0)

        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None,
            ncore=0)
        with self.assertRaisesRegex(TypeError, "gas_orbs.*integers"):
            mc.gas_orbs = (True,)

    def test_ras_zero_sized_edge_spaces_match_gasci_contract(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75",
                    basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        restriction = {"max_holes": 0, "max_particles": 0}

        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(0, 2, 0), gas_restr=restriction,
            gas_restr_type="ras", ncore=0)
        self.assertEqual(mc.gas_orbs, (0, 2, 0))
        self.assertEqual(mc.ncas, 2)
        kernel_orbs, _ = mc._normalized_restriction()
        self.assertEqual(kernel_orbs, (2,))
        self.assertIs(mc.validate_capabilities(), mc)
        self.assertFalse(mc.internal_rotation)

        solver = fci_gas.FCISolver(
            mol, gas_orbs=(0, 2, 0), gas_restr=restriction,
            gas_restr_type="ras")
        adapted = gasscf.GASSCF(
            mf, 2, (1, 1), fcisolver=solver, ncore=0)
        self.assertEqual(adapted.gas_orbs, (0, 2, 0))
        self.assertEqual(adapted._normalized_restriction()[0], (2,))

    def test_gasscf_space_info_uses_gasci_normalization(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", ncore=0)

        info = mc.gas_space_info()
        metadata = info["metadata"]

        self.assertEqual(metadata["gas_orbs"], (1, 1))
        self.assertEqual(metadata["kernel_gas_orbs"], (1, 1))
        self.assertEqual(metadata["gas_restr_type"], "cumulative-occ")
        numpy.testing.assert_array_equal(
            metadata["gas_restr"], numpy.asarray([[1, 1], [2, 2]]))
        numpy.testing.assert_array_equal(
            metadata["spin_supergroups"],
            numpy.asarray([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=numpy.int32))
        self.assertEqual(info["core"]["ndet"], 2)

    def test_shared_space_info_preserves_user_and_kernel_specs(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        blocks = numpy.array([[0, 1, 1, 0], [1, 0, 0, 1]])
        cases = (
            ((1, 1), blocks, 'spin-supergroup', 2, (1, 1)),
            ((1, 1), ((1, 1),), 'supergroup', 2, (1, 1)),
            ((1, 1), ((1, 1), (2, 2)), 'cumulative-occ', 2, (1, 1)),
            ((0, 2, 0), {'max_holes': 0, 'max_particles': 0}, 'ras', 4, (2,)),
            (None, None, 'spin-supergroup', 4, (2,)),
        )
        for sizes, restriction, kind, ndet, kernel_sizes in cases:
            for driver in (gasci.GASCI, gasscf.GASSCF):
                with self.subTest(driver=driver.__name__, kind=kind, sizes=sizes):
                    mc = driver(mf, 2, (1, 1), ncore=0, gas_orbs=sizes,
                                gas_restr=restriction, gas_restr_type=kind)
                    if isinstance(mc, gasscf.GASSCF):
                        self.addCleanup(mc.close)
                    report = mc.gas_space_info()
                    meta = report['metadata']
                    self.assertEqual(meta['gas_orbs'], (2,) if sizes is None else sizes)
                    self.assertEqual(meta['kernel_gas_orbs'], kernel_sizes)
                    self.assertEqual(meta['gas_restr_type'], kind)
                    self.assertEqual(report['core']['ndet'], ndet)
                    expected = blocks if ndet == 2 else numpy.array([[1, 1]])
                    numpy.testing.assert_array_equal(meta['spin_supergroups'], expected)
                    if kind == 'ras':
                        self.assertEqual(meta['gas_restr'],
                                         {'max_holes': 0, 'max_particles': 0})
                    elif restriction is None:
                        self.assertIsNone(meta['gas_restr'])
                    else:
                        numpy.testing.assert_array_equal(meta['gas_restr'], restriction)
                    # Reports own their arrays; changing one must not change
                    # either the user specification or the next kernel space.
                    meta['spin_supergroups'][:] = -1
                    if isinstance(meta['gas_restr'], numpy.ndarray):
                        meta['gas_restr'][:] = -1
                    self.assertEqual(mc.gas_space_info()['core']['ndet'], ndet)

        # GASCI stores a user specification separately from its solver.
        mc = gasci.GASCI(mf, 2, (1, 1), ncore=0)
        mc.gas_orbs = (1, 1)
        mc.gas_restr = ((1, 1), (2, 2))
        mc.gas_restr_type = 'cumulative-occ'
        self.assertEqual(mc.gas_space_info()['core']['ndet'], 2)
        self.assertEqual(mc.fcisolver.gas_orbs, (1, 1))

    def test_shared_integral_methods_preserve_df_dispatch(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=1, gas_orbs=(1, 1),
                          gas_restr=((1, 1), (2, 2)),
                          gas_restr_type='cumulative-occ')
        self.addCleanup(mc.close)
        fitted = mc.density_fit(auxbasis='weigend')
        self.addCleanup(fitted.close)
        rotation = numpy.zeros((4, 4))
        rotation[0, 2], rotation[2, 0] = .13, -.13
        mo = mf.mo_coeff @ scipy.linalg.expm(rotation)
        active = mo[:, 1:3]
        exact = ao2mo.kernel(mol, active)
        approximate = fitted.with_df.ao2mo(active)
        self.assertGreater(numpy.linalg.norm(exact - approximate), 1e-8)
        for obj, expected in ((mc, exact), (fitted, approximate)):
            with self.subTest(df=hasattr(obj, 'with_df')):
                numpy.testing.assert_allclose(obj.get_h2gas(mo), expected, atol=1e-12, rtol=0)
                for ncore, ncas in ((1, 2), (0, 1)):
                    core = mo[:, :ncore]
                    dm = 2 * core @ core.T
                    vj, vk = obj.get_jk(mol, dm)
                    potential = vj - .5 * vk
                    hcore = obj.get_hcore()
                    act = mo[:, ncore:ncore+ncas]
                    expected_h1 = act.T @ (hcore + potential) @ act
                    expected_core = (mol.energy_nuc() + numpy.einsum('ij,ji', dm, hcore)
                                     + .5 * numpy.einsum('ij,ji', dm, potential))
                    h1, energy = obj.get_h1gas(mo, ncas=ncas, ncore=ncore)
                    numpy.testing.assert_allclose(h1, expected_h1, atol=1e-12, rtol=0)
                    self.assertAlmostEqual(energy, expected_core, 12)

    def test_public_entries_reject_invalid_mo_before_integrals(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        bad = ((mf.mo_coeff * 1.01, ValueError),
               (mf.mo_coeff[:, :2], ValueError),
               (mf.mo_coeff[:2], ValueError),
               (numpy.ones(4), ValueError),
               (numpy.full((4, 4), numpy.nan), ValueError),
               (numpy.full((4, 4), numpy.inf), ValueError),
               (mf.mo_coeff.astype(complex), TypeError),
               (mf.mo_coeff.astype(str), TypeError))
        for kind in ('plain', 'df', 'sa', 'df-sa', 'sa-df'):
            mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=1)
            if kind in ('df', 'df-sa'):
                mc = mc.density_fit()
            if kind in ('sa', 'df-sa', 'sa-df'):
                mc = mc.state_average((.5, .5))
            if kind == 'sa-df':
                mc = mc.density_fit()
            self.addCleanup(mc.close)
            for method in ('gasci', 'kernel', 'get_grad'):
                for implicit in (False, True):
                    for mo, error in bad:
                        with self.subTest(kind=kind, method=method,
                                          implicit=implicit, shape=mo.shape):
                            mc.mo_coeff = mo if implicit else mf.mo_coeff
                            with mock.patch.object(mc, 'get_h1eff') as h1, \
                                    mock.patch.object(mc, 'ao2mo',
                                                      side_effect=AssertionError('AO2MO')) as ao:
                                with self.assertRaises(error):
                                    getattr(mc, method)(None if implicit else mo)
                                h1.assert_not_called()
                                ao.assert_not_called()
                            if not implicit:
                                self.assertIs(mc.mo_coeff, mf.mo_coeff)

    def test_public_entries_share_gasci_metric_thresholds(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        for method in ('gasci', 'kernel', 'get_grad'):
            for sa in (False, True):
                mc = gasscf.GASSCF(mf, 2, (1, 1))
                if sa:
                    mc = mc.state_average((.5, .5))
                self.addCleanup(mc.close)
                mc.canonicalization = False
                mc.stdout = io.StringIO()
                mc.verbose = 4
                with mock.patch.object(gasci, 'MO_ORTH_WARN_TOL', 1e-7), \
                        mock.patch.object(gasci, 'MO_ORTH_ERROR_TOL', 1e-5):
                    for error in (5e-8, 2e-6, 2e-5):
                        with self.subTest(method=method, sa=sa, error=error):
                            mo = mf.mo_coeff.copy()
                            mo[:, 0] *= numpy.sqrt(1 + error)
                            mc.stdout.seek(0)
                            mc.stdout.truncate(0)
                            if error > 1e-5:
                                with self.assertRaisesRegex(ValueError, 'orthonormal'):
                                    getattr(mc, method)(mo)
                            else:
                                result = getattr(mc, method)(mo)
                                value = result if method == 'get_grad' else result[0]
                                self.assertTrue(numpy.all(numpy.isfinite(value)))
                                self.assertEqual('WARN: MO orthonormality' in
                                                 mc.stdout.getvalue(), error > 1e-7)

    def test_public_entries_validate_solver_and_problem(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        for method in ('gasci', 'kernel'):
            for invalid in ('solver', 'nroots', 'natorb', 'gas-size', 'electrons'):
                with self.subTest(method=method, invalid=invalid):
                    mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=0)
                    self.addCleanup(mc.fcisolver.close)
                    if invalid == 'solver':
                        mc.fcisolver = direct_spin1.FCISolver(mol)
                        error = NotImplementedError
                    elif invalid == 'nroots':
                        mc.fcisolver.nroots = 2
                        error = ValueError
                    elif invalid == 'natorb':
                        mc.natorb = True
                        error = NotImplementedError
                    elif invalid == 'gas-size':
                        mc.gas_orbs = (3,)
                        error = ValueError
                    else:
                        mc.nelecas = (0, 0)
                        error = AssertionError  # Native CASCI.check_sanity contract.
                    with mock.patch.object(mc, 'get_h1eff') as h1, \
                            mock.patch.object(mc, 'ao2mo') as ao:
                        with self.assertRaises(error):
                            getattr(mc, method)()
                        h1.assert_not_called()
                        ao.assert_not_called()

    def test_public_entries_initialize_and_validate_missing_mo(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        ref = scf.RHF(mol).run()
        expected = gasci.GASCI(ref, 2, (1, 1)).kernel()[0]
        for method in ('gasci', 'kernel'):
            mf = scf.RHF(mol)
            if getattr(mf, '_chkfile', None) is not None:
                self.addCleanup(mf._chkfile.close)
            mc = gasscf.GASSCF(mf, 2, (1, 1))
            self.addCleanup(mc.close)
            mc.canonicalization = False
            with mock.patch.object(mf, 'run', wraps=mf.run) as run:
                self.assertAlmostEqual(getattr(mc, method)()[0], expected, places=11)
                run.assert_called_once()
            self.assertTrue(mc.converged)
            # Explicit orbitals must not trigger an unnecessary SCF solve.
            mc.mo_coeff = None
            with mock.patch.object(mf, 'run') as run:
                self.assertAlmostEqual(getattr(mc, method)(ref.mo_coeff)[0],
                                       expected, places=11)
                run.assert_not_called()
            # Orbitals generated by SCF receive the same validation.
            mc.mo_coeff = None
            mf.mo_coeff = ref.mo_coeff * 1.01
            with mock.patch.object(mf, 'run', return_value=mf) as run:
                with self.assertRaisesRegex(ValueError, 'orthonormal'):
                    getattr(mc, method)()
                run.assert_called_once()

    def test_newton_checks_mo_only_at_public_entry(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.2; H 0 0 3.1',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), ncore=1, gas_orbs=(1, 1),
            gas_restr=((1, 1), (2, 2)), gas_restr_type='cumulative-occ')
        self.addCleanup(mc.close)
        mc.max_cycle_macro = 2
        mc.canonicalization = False
        with mock.patch.object(mc, '_check_mo_orthonormality',
                               wraps=mc._check_mo_orthonormality) as check, \
                mock.patch.object(mc, 'casci', wraps=mc.casci) as internal:
            result = mc.kernel()
            self.assertTrue(numpy.isfinite(result[0]))
            self.assertGreater(internal.call_count, 1)
            check.assert_called_once()

    def test_kernel_rejects_unwrapped_multiroot(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)
        mc.fcisolver.nroots = 2

        with self.assertRaisesRegex(ValueError, "nroots>1"):
            mc.kernel(mf.mo_coeff)

    def test_unadapted_native_wrappers_fail_before_calculation(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        wrappers = (
            ('sa', lambda mc: addons.state_average(mc, (.5, .5))),
            ('sa-inplace', lambda mc: addons.state_average_(mc, (.5, .5))),
            ('df', mcdf.density_fit),
            ('native-sa-over-gas-df', lambda mc:
             addons.state_average(mc.density_fit(), (.5, .5))),
            ('native-df-over-gas-sa', lambda mc:
             mcdf.density_fit(mc.state_average((.5, .5)))),
            ('native-df-sa', lambda mc:
             addons.state_average(mcdf.density_fit(mc), (.5, .5))),
            ('native-sa-df', lambda mc:
             mcdf.density_fit(addons.state_average(mc, (.5, .5)))),
        )
        for kind, wrap in wrappers:
            source = gasscf.GASSCF(mf, 2, (1, 1))
            self.addCleanup(source.close)
            plan = source.fcisolver._get_rdm_plan(2, (1, 1))
            mc = wrap(source)
            self.addCleanup(mc.close)
            for method in ('validate_capabilities', 'kernel', 'gasci',
                           'newton', 'as_scanner', 'state_average', 'density_fit',
                           'get_grad'):
                with self.subTest(kind=kind, method=method):
                    with mock.patch.object(mc._scf, 'run') as scf_run, \
                            mock.patch.object(mc, 'get_h1eff') as h1, \
                            mock.patch.object(mc, 'ao2mo') as ao:
                        with self.assertRaisesRegex(NotImplementedError,
                                                    'unadapted PySCF.*original GASSCF'):
                            getattr(mc, method)()
                        scf_run.assert_not_called()
                        h1.assert_not_called()
                        ao.assert_not_called()
                    self.assertIsNotNone(plan._plan)

    def test_native_approx_hessian_wrapper_is_rejected(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for sa in (False, True):
            source = gasscf.GASSCF(mf, 2, (1, 1))
            if sa:
                source = source.state_average((.5, .5))
            self.addCleanup(source.close)
            mc = mcdf.approx_hessian(source)
            self.addCleanup(mc.close)
            for method in ('validate_capabilities', 'kernel', 'gasci',
                           'newton', 'as_scanner', 'state_average', 'density_fit',
                           'get_grad'):
                with self.subTest(sa=sa, method=method):
                    with mock.patch.object(mc._scf, 'run') as run, \
                            mock.patch.object(mc, 'ao2mo') as ao:
                        with self.assertRaisesRegex(NotImplementedError,
                                                    'approximate Hessian'):
                            getattr(mc, method)()
                        run.assert_not_called()
                        ao.assert_not_called()

    def test_scf_inputs_follow_native_casscf_conversion(self):
        for spin in (0, 2):
            mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                        basis='sto-3g', spin=spin, verbose=0)
            nelec = (1 + spin // 2, 1 - spin // 2)
            for factory in (scf.UHF, dft.UKS):
                base = factory(mol).run()
                self.addCleanup(base._chkfile.close)
                for kind in ('plain', 'df', 'newton', 'df-newton'):
                    source = base.copy()
                    if 'df' in kind:
                        source = source.density_fit(auxbasis='weigend')
                    if 'newton' in kind:
                        source = source.newton()
                    for orbitals in (True, False):
                        for key in ('mo_coeff', 'mo_occ', 'mo_energy'):
                            setattr(source, key, getattr(base, key) if orbitals else None)
                        original = {key: getattr(source, key)
                                    for key in ('mo_coeff', 'mo_occ', 'mo_energy')}
                        with self.subTest(spin=spin, scf=factory.__name__,
                                          kind=kind, orbitals=orbitals):
                            ref = mcscf.CASSCF(source, 2, nelec)
                            for constructor in (gasscf.GASSCF, gasscf.DFGASSCF):
                                mc = constructor(source, 2, nelec)
                                self.addCleanup(mc.close)
                                self.assertEqual(type(mc._scf), type(ref._scf))
                                self.assertIsNot(mc._scf, source)
                                self.assertEqual(isinstance(mc, mcdf._DFCAS),
                                                 'df' in kind or
                                                 constructor is gasscf.DFGASSCF)
                                for key, value in original.items():
                                    self.assertIs(getattr(source, key), value)
                                    expected = getattr(ref._scf, key)
                                    actual = getattr(mc._scf, key)
                                    if expected is None:
                                        self.assertIsNone(actual)
                                    else:
                                        numpy.testing.assert_array_equal(actual, expected)
                                if 'df' in kind:
                                    self.assertIs(mc.with_df, source.with_df)
                                if orbitals and kind == 'plain':
                                    native = (ref if constructor is gasscf.GASSCF
                                              else mcscf.DFCASSCF(source, 2, nelec))
                                    expected = native.casci(
                                        native.mo_coeff, eris=native.ao2mo(native.mo_coeff))[0]
                                    self.assertAlmostEqual(mc.gasci()[0], expected, places=9)

    def test_df_construction_options_and_molecule_inputs(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        for constructor, native in ((gasscf.GASSCF, mcscf.CASSCF),
                                    (gasscf.DFGASSCF, mcscf.DFCASSCF)):
            mc, ref = constructor(mol, 2, (1, 1)), native(mol, 2, (1, 1))
            self.addCleanup(mc.close)
            self.addCleanup(mc._scf._chkfile.close)
            self.addCleanup(ref._scf._chkfile.close)
            self.assertEqual(type(mc._scf), type(ref._scf))
            self.assertEqual(isinstance(mc, mcdf._DFCAS), isinstance(ref, mcdf._DFCAS))
        mf = scf.RHF(mol).density_fit()
        self.addCleanup(mf._chkfile.close)
        for kwargs in ({}, {'auxbasis': 'weigend'},
                       {'with_df': df.DF(mol)},
                       {'auxbasis': 'weigend', 'with_df': df.DF(mol, 'weigend')}):
            with self.subTest(options=tuple(kwargs)):
                mc = gasscf.DFGASSCF(mf, 2, (1, 1), **kwargs)
                self.addCleanup(mc.close)
                # Compare the explicit options on an unwrapped native object.
                ref = mcdf.density_fit(mc1step.CASSCF(mf, 2, (1, 1)), **kwargs)
                self.assertEqual(mc.with_df.auxbasis, ref.with_df.auxbasis)
                self.assertIs(mc._scf, mf)
                if 'with_df' in kwargs:
                    self.assertIs(mc.with_df, kwargs['with_df'])
                elif not kwargs:
                    self.assertIs(mc.with_df, mf.with_df)
                else:
                    self.assertIsNot(mc.with_df, mf.with_df)
        mf.with_df = None
        mc = gasscf.GASSCF(mf, 2, (1, 1))
        self.addCleanup(mc.close)
        self.assertNotIsInstance(mc, mcdf._DFCAS)

    def test_spinor_scf_references_rejected_before_integrals(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        for factory in (scf.GHF, dft.GKS, scf.DHF, dft.DKS):
            source = factory(mol)
            self.addCleanup(source._chkfile.close)
            for constructor in (gasscf.GASSCF, gasscf.DFGASSCF):
                with self.subTest(scf=factory.__name__, constructor=constructor.__name__):
                    with self.assertRaisesRegex(NotImplementedError, 'spinor SCF'):
                        constructor(source, 2, (1, 1))
            ordinary = scf.RHF(mol)
            self.addCleanup(ordinary._chkfile.close)
            mc = gasscf.GASSCF(ordinary, 2, (1, 1))
            self.addCleanup(mc.close)
            mc._scf = source
            with mock.patch.object(mc, 'ao2mo', side_effect=AssertionError('AO2MO')):
                with self.assertRaisesRegex(NotImplementedError, 'spinor SCF'):
                    mc.kernel()

        # Native UHF conversion must not erase an unsupported decoration.
        source = scf.UHF(mol)
        self.addCleanup(source._chkfile.close)
        for decorated, message in ((source.sfx2c1e(), 'X2C'),
                                   (solvent.ddCOSMO(source), 'solvent models')):
            with mock.patch.object(decorated, 'to_rhf',
                                   side_effect=AssertionError('SCF conversion')):
                for constructor in (gasscf.GASSCF, gasscf.DFGASSCF):
                    with self.assertRaisesRegex(NotImplementedError, message):
                        constructor(decorated, 2, (1, 1))

    def test_dfgasscf_factory_reuses_density_fit_scf_object(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).density_fit().run()
        self.addCleanup(mf._chkfile.close)
        for constructor in (gasscf.GASSCF, gasscf.DFGASSCF):
            mc = constructor(mf, 2, (1, 1), gas_orbs=(2,), ncore=1)
            self.addCleanup(mc.close)
            self.assertIsInstance(mc, gasscf.GASSCF)
            self.assertIsInstance(mc, mcdf._DFCAS)
            self.assertIs(mc.with_df, mf.with_df)
            self.assertEqual(mc.gas_orbs, (2,))
            self.assertIs(mc.validate_capabilities(), mc)
            plain = mc.undo_df()
            self.addCleanup(plain.close)
            self.assertIsInstance(plain, gasscf.GASSCF)
            self.assertNotIsInstance(plain, mcdf._DFCAS)
            self.assertIsNotNone(plain._scf.with_df)
            self.assertIs(plain.newton(), plain)
            self.assertNotIsInstance(plain, mcdf._DFCAS)

    def test_symmetry_rejected_during_construction(self):
        for symmetry in (True, 'C1'):
            mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g',
                        symmetry=symmetry, verbose=0)
            mf = scf.UHF(mol)
            self.addCleanup(mf._chkfile.close)
            with mock.patch.object(mf, 'to_rhf', side_effect=AssertionError('conversion')):
                for source in (mol, mf):
                    for constructor in (gasscf.GASSCF, gasscf.DFGASSCF):
                        with self.subTest(symmetry=symmetry, constructor=constructor.__name__):
                            with self.assertRaisesRegex(NotImplementedError, 'point-group'):
                                constructor(source, 2, (1, 1))
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        solver = fci_gas.FCISolver(mol, gas_orbs=(2,))
        for target, name, value in (
                (solver, 'orbsym', [0, 0]), (solver, 'wfnsym', 0),
                (mf, 'mo_coeff', lib.tag_array(mf.mo_coeff.copy(), orbsym=[0, 0]))):
            with mock.patch.object(target, name, value, create=True):
                for constructor in (gasscf.GASSCF, gasscf.DFGASSCF):
                    with self.assertRaisesRegex(NotImplementedError, 'symmetry'):
                        constructor(mf, 2, (1, 1), fcisolver=solver)

    def test_symmetry_settings_and_tagged_orbitals_rejected(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        tagged = lib.tag_array(mf.mo_coeff.copy(), orbsym=[0, 0])
        for kind in ('plain', 'df', 'sa', 'df-sa', 'sa-df'):
            mc = gasscf.GASSCF(mf, 2, (1, 1))
            self.addCleanup(mc.close)
            if kind in ('df', 'df-sa'):
                mc = mc.density_fit()
            if kind in ('sa', 'df-sa', 'sa-df'):
                mc = mc.state_average((.5, .5))
            if kind == 'sa-df':
                mc = mc.density_fit()
            self.addCleanup(mc.close)
            entries = ('gasci', 'casci', 'kernel', 'get_grad', 'get_fock',
                       'canonicalize', 'canonicalize_')
            settings = ((mc, 'extrasym', [0, 0]), (mc, 'wfnsym', 0),
                        (mc, 'orbsym', [0, 0]),
                        (mc.fcisolver, 'orbsym', [0, 0]),
                        (mc.fcisolver, 'wfnsym', 0),
                        (mc, 'mo_coeff', tagged), (mc._scf, 'mo_coeff', tagged),
                        (mc.mol, 'symmetry', True))
            with mock.patch.object(mc, 'ao2mo', side_effect=AssertionError('AO2MO')):
                for target, name, value in settings:
                    with mock.patch.object(target, name, value, create=True):
                        for entry in entries + ('newton', 'as_scanner'):
                            with self.subTest(kind=kind, setting=name, entry=entry):
                                with self.assertRaisesRegex(NotImplementedError, 'symmetry'):
                                    getattr(mc, entry)()
                        with self.assertRaisesRegex(NotImplementedError, 'symmetry'):
                            mc.uniq_var_indices(2, 0, 2, None)
                for entry in entries:
                    with self.subTest(kind=kind, explicit_mo_entry=entry):
                        with self.assertRaisesRegex(NotImplementedError, 'orbsym'):
                            getattr(mc, entry)(tagged)
                for entry in ('get_gas_natorb', 'get_gas_pseudo_natorb'):
                    with self.assertRaisesRegex(NotImplementedError, 'orbsym'):
                        getattr(mc, entry)(tagged, gasdm1=numpy.eye(2))
                with self.assertRaisesRegex(NotImplementedError, 'orbsym'):
                    mc.gen_g_hop(tagged, None, None)
                with self.assertRaisesRegex(NotImplementedError, 'orbsym'):
                    mc.rotate_mo(tagged, numpy.eye(2))
                with self.assertRaisesRegex(NotImplementedError, 'orbsym'):
                    mc.sort_mo(([0, 1],), tagged, base=0)
                with mock.patch.object(mc, 'make_gasdm1s',
                                       return_value=(numpy.eye(2), numpy.eye(2))):
                    for entry in ('make_rdm1', 'make_rdm1s'):
                        with self.assertRaisesRegex(NotImplementedError, 'orbsym'):
                            getattr(mc, entry)(mo_coeff=tagged)
            # Explicit densities do not bypass the canonicalization guard.
            with self.assertRaisesRegex(NotImplementedError, 'orbsym'):
                mc.canonicalize(tagged, gasdm1=numpy.eye(2), gas_pseudo_natorb=True)
        numpy.testing.assert_array_equal(tagged.orbsym, [0, 0])

    def test_unsupported_apis_through_wrappers(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        # No SCF/CI solve is needed: unsupported APIs must fail at entry,
        # independently of the availability of orbitals or GPU packages.
        for kind in ('plain', 'df', 'sa', 'df-sa', 'sa-df', 'undo-sa', 'undo-df'):
            for scanner in (False, True):
                mc = gasscf.GASSCF(
                    mf, 2, (1, 1), ncore=0, gas_orbs=(1, 1),
                    gas_restr=((1, 1), (2, 2)), gas_restr_type='cumulative-occ')
                self.addCleanup(mc.close)
                if kind in ('df', 'df-sa', 'undo-df'):
                    mc = mc.density_fit()
                if kind in ('sa', 'df-sa', 'sa-df', 'undo-sa'):
                    mc = mc.state_average((.5, .5))
                if kind == 'sa-df':
                    mc = mc.density_fit()
                if kind == 'undo-sa':
                    mc = mc.undo_state_average()
                if kind == 'undo-df':
                    mc = mc.undo_df()
                if scanner:
                    mc = mc.as_scanner()
                self.addCleanup(mc.close)
                calls = (
                    ('nuc_grad_method', {}, 'nuclear gradient'),
                    ('Gradients', {}, 'nuclear gradient'),
                    ('nuc_grad_method', {'state': 1}, 'nuclear gradient'),
                    ('Gradients', {'state': 1}, 'nuclear gradient'),
                    ('nac_method', {}, 'nonadiabatic coupling'),
                    ('NACs', {}, 'nonadiabatic coupling'),
                    ('to_gpu', {}, 'C/OpenMP backend'),
                    ('approx_hessian', {}, 'approximate Hessian'),
                    ('sfx2c1e', {}, 'X2C'),
                    ('x2c1e', {}, 'X2C'),
                    ('x2c', {}, 'X2C'),
                    ('ddCOSMO', {}, 'solvent models'),
                    ('DDCOSMO', {}, 'solvent models'),
                    ('ddPCM', {}, 'solvent models'),
                    ('DDPCM', {}, 'solvent models'),
                    ('PCM', {}, 'solvent models'),
                    ('mc2step', {}, 'two-step GASSCF kernel'),
                    ('mc1step', {}, 'use kernel\\(\\)'),
                    ('solve_approx_ci', dict(h1=None, h2=None, ci0=None,
                                             ecore=0., e_cas=0., envs={}),
                     'legacy mc1step'),
                    ('update_casdm', dict(mo=None, u=None, fcivec=None,
                                          e_cas=0., eris=None), 'legacy mc1step'),
                    ('rotate_orb_cc', dict(mo=None, fcivec=None, fcasdm1=None,
                                           fcasdm2=None, eris=None), 'legacy mc1step'),
                )
                for method, kwargs, message in calls:
                    with self.subTest(kind=kind, scanner=scanner,
                                      method=method, kwargs=kwargs):
                        with self.assertRaisesRegex(NotImplementedError, message):
                            getattr(mc, method)(**kwargs)

                # External decorations must also fail at calculation entry.
                decorators = (
                    ('X2C', lambda obj: obj.sfx2c1e(), ('scf',), 'X2C'),
                    ('ddCOSMO', solvent.ddCOSMO, ('scf', 'mc'), 'solvent models'),
                    ('ddPCM', solvent.ddPCM, ('scf', 'mc'), 'solvent models'),
                    ('PCM', solvent.PCM, ('scf', 'mc'), 'solvent models'),
                )
                for name, decorate, targets, message in decorators:
                    for target in targets:
                        trial = mc.copy()
                        if target == 'scf':
                            trial._scf = decorate(trial._scf)
                        else:
                            trial = decorate(trial)
                        self.addCleanup(trial.close)
                        for method in ('kernel', 'gasci', 'get_grad', 'newton'):
                            with self.subTest(kind=kind, scanner=scanner,
                                              model=name, target=target, method=method):
                                with self.assertRaisesRegex(NotImplementedError, message):
                                    getattr(trial, method)()
                        with self.subTest(kind=kind, scanner=scanner,
                                          model=name, target=target, method='scanner'):
                            with self.assertRaisesRegex(NotImplementedError, message):
                                if scanner:
                                    trial(mol)
                                else:
                                    trial.as_scanner()

class TestSolverAdapter(unittest.TestCase):
    """GAS solver dispatch, physical operators and reusable plans."""

    def test_cached_rdm_dispatch_matches_full_fci_for_ss_and_sa(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        norb, nelec = 3, (1, 1)
        def compare(actual, expected):
            if isinstance(expected, tuple):
                self.assertEqual(len(actual), len(expected))
                for a, e in zip(actual, expected):
                    compare(a, e)
            else:
                numpy.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        for cache in (False, True):
            for weights in (None, (.3, .7), (1., 0.)):
                mc = gasscf.GASSCF(
                    mf, norb, nelec, ncore=1, gas_orbs=(1, 2),
                    gas_restr=((0, 1), (2, 2)), gas_restr_type='cumulative-occ',
                    cache_plans=cache)
                if weights is not None:
                    mc = mc.state_average(weights)
                self.addCleanup(mc.close)
                solver = mc.fcisolver
                with solver.make_space(norb, nelec) as space:
                    rng = numpy.random.default_rng(903)
                    roots = list(numpy.linalg.qr(rng.normal(size=(space.ndet, 2)))[0].T)
                    full = [fci_gas.gas2fci(c, space) for c in roots]
                base = (solver if weights is None else
                        super(addons.StateAverageFCISolver, solver))
                with self.subTest(cache=cache, weights=weights):
                    for method in ('make_rdm123', 'make_rdm123s',
                                   'make_rdm1234', 'make_rdm1234s'):
                        with self.assertRaisesRegex(NotImplementedError, 'density matrices'):
                            getattr(solver, method)(None, norb, nelec)
                    with self.assertRaisesRegex(NotImplementedError, 'C/OpenMP backend'):
                        solver.to_gpu()
                    for method in ('make_rdm1', 'make_rdm1s', 'make_rdm12',
                                   'make_rdm12s', 'trans_rdm1', 'trans_rdm1s',
                                   'trans_rdm12', 'trans_rdm12s'):
                        transition = method.startswith('trans')
                        args = tuple(full) if transition else (full[0],)
                        expected = getattr(direct_spin1, method)(*args, norb, nelec)
                        # Native Newton may supply single-root CI lists.
                        args = tuple([c] for c in roots) if transition else ([roots[0]],)
                        compare(getattr(base, method)(*args, norb, nelec), expected)
                    expected_spin = [spin_op.spin_square(c, norb, nelec) for c in full]
                    compare(base.spin_square(roots[0], norb, nelec), expected_spin[0])
                    if weights is not None:
                        expected = [direct_spin1.make_rdm12s(c, norb, nelec) for c in full]
                        dm1s, dm2s = solver.make_rdm12s(roots, norb, nelec)
                        for s in range(2):
                            compare(dm1s[s], sum(w*d[0][s] for w, d in zip(weights, expected)))
                        for s in range(3):
                            compare(dm2s[s], sum(w*d[1][s] for w, d in zip(weights, expected)))
                        compare(solver.make_rdm2(roots, norb, nelec),
                                sum(w*direct_spin1.make_rdm12(c, norb, nelec)[1]
                                    for w, c in zip(weights, full)))
                        compare(solver.spin_square(roots, norb, nelec),
                                tuple(numpy.asarray(weights) @ numpy.asarray(expected_spin)))
                    plan = solver._rdm_plan
                    if cache:
                        self.assertIsNotNone(plan)
                        base.make_rdm12(roots[0], norb, nelec)
                        self.assertIs(solver._rdm_plan, plan)
                        with self.assertRaises(ValueError):
                            base.make_rdm12(roots[0][:-1], norb, nelec)
                        self.assertIsNotNone(plan._plan)
                        mc.close()
                        self.assertIsNone(plan._plan)
                        self.assertIsNone(plan.gas)
                    else:
                        self.assertIsNone(plan)

    def test_external_rdm_plan_bypasses_and_survives_newton_cache(self):
        solver = gasscf._GASFCISolver(gas_orbs=(2,))
        self.addCleanup(solver.close)
        ci = numpy.array([.5, .5, .5, .5])
        reference = fci_gas.FCISolver(gas_orbs=(2,))
        with reference.make_rdm_plan(2, (1, 1)) as external:
            for cache in (False, True):
                solver.cache_plans = cache
                with mock.patch.object(solver, '_get_rdm_plan',
                                       side_effect=AssertionError('borrowed plan cached')):
                    actual = solver.make_rdm1(ci, 2, (1, 1), plan=external)
                    numpy.testing.assert_allclose(
                        actual, direct_spin1.make_rdm1(ci.reshape(2, 2), 2, (1, 1)),
                        atol=1e-12, rtol=0)
                    with self.assertRaises(ValueError):
                        solver.make_rdm12(ci[:-1], 2, (1, 1), plan=external)
                self.assertIsNone(solver._rdm_plan)
                solver.close()
                self.assertIsNotNone(external._plan)
            # A populated owned cache also survives an external-plan call.
            solver.make_rdm1(ci, 2, (1, 1))
            owned = solver._rdm_plan
            solver.make_rdm12(ci, 2, (1, 1), plan=external)
            self.assertIs(solver._rdm_plan, owned)
            solver.close()
            self.assertIsNone(owned._plan)
            self.assertIsNotNone(external._plan)

    def test_cached_contraction_uses_base_entry_and_physical_operator(self):
        solver = gasscf._GASFCISolver(
            gas_orbs=(1, 2), gas_restr=((0, 1), (2, 2)),
            gas_restr_type='cumulative-occ')
        self.addCleanup(solver.close)
        solver.ss_penalty, solver.ss_value = .2, 0.
        rng = numpy.random.default_rng(904)
        eri = rng.normal(size=(6, 6))
        eri += eri.T
        with solver.make_space(3, (1, 1)) as space:
            ci = rng.normal(size=space.ndet)
            full = fci_gas.gas2fci(ci, space)
            expected = fci_gas.fci2gas(
                direct_spin1.contract_2e(eri, full, 3, (1, 1)), space)
        original = fci_gas.FCISolver.contract_2e
        seen = []
        def contract(obj, *args, **kwargs):
            seen.append(kwargs.get('plan'))
            return original(obj, *args, **kwargs)
        with mock.patch.object(fci_gas.FCISolver, 'contract_2e', new=contract):
            for cache in (True, False):
                solver.cache_plans = cache
                for integrals in (eri, eri.copy()):
                    numpy.testing.assert_allclose(solver.contract_2e(integrals, ci, 3, (1, 1)),
                                                  expected, atol=1e-12, rtol=0)
        self.assertEqual(len(seen), 4)
        self.assertIsInstance(seen[0], fci_gas.GasContractPlan)
        self.assertIs(seen[0], seen[1])
        self.assertEqual(seen[2:], [None, None])

        # Exercise eviction through the public entry with distinct Hamiltonians.
        solver.cache_plans = True
        plan = seen[0]
        for scale in (1., 2., 3.):
            solver.contract_2e(eri + numpy.eye(6) * scale, ci, 3, (1, 1))
        self.assertLessEqual(len(solver._contract_plans), solver._MAX_CONTRACT_PLANS)
        self.assertIsNone(plan._plan)
        remaining = list(solver._contract_plans.values())
        space = solver._contract_space
        solver.close()
        self.assertEqual(len(solver._contract_plans), 0)
        self.assertIsNone(solver._contract_space)
        self.assertIsNone(space._gas)
        for plan in remaining:
            self.assertIsNone(plan._plan)

    def test_explicit_gasci_solver_is_adapted_by_copy(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = fci_gas.FCISolver(
            mol, gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ")
        solver.nroots = 2
        solver.spin = 0

        mc = gasscf.GASSCF(
            mf, 2, (1, 1), fcisolver=solver, ncore=0,
            cache_plans=False)

        self.assertIsNot(mc.fcisolver, solver)
        self.assertIsInstance(mc.fcisolver, fci_gas.FCISolver)
        self.assertEqual(mc.ncas, 2)
        self.assertEqual(mc.ngas, 2)
        self.assertEqual(mc.fcisolver.gas_orbs, (1, 1))
        self.assertEqual(mc.fcisolver.gas_restr, [[1, 1], [2, 2]])
        self.assertEqual(mc.fcisolver.gas_restr_type, "cumulative-occ")
        self.assertEqual(mc.fcisolver.nroots, 2)
        self.assertEqual(mc.fcisolver.spin, 0)
        self.assertFalse(mc.cache_plans)
        self.assertFalse(hasattr(solver, "cache_plans"))

    def test_explicit_newton_gas_solver_is_copied(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        source = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0,
            cache_plans=False).fcisolver

        mc = gasscf.GASSCF(
            mf, 2, (1, 1), fcisolver=source, ncore=0)

        self.assertIsNot(mc.fcisolver, source)
        self.assertEqual(mc.gas_orbs, (2,))
        self.assertFalse(mc.cache_plans)
        source.cache_plans = True
        self.assertFalse(mc.cache_plans)

    def test_explicit_solver_model_arguments_are_rejected(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = fci_gas.FCISolver(mol, gas_orbs=(2,))

        with self.assertRaisesRegex(ValueError, "explicit fcisolver"):
            gasscf.GASSCF(
                mf, 2, (1, 1), gas_orbs=(2,), fcisolver=solver,
                ncore=0)

    def test_explicit_solver_requires_gas_orbs(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = fci_gas.FCISolver(mol)

        with self.assertRaisesRegex(ValueError, "explicit GASCI solver"):
            gasscf.GASSCF(
                mf, 2, (1, 1), fcisolver=solver, ncore=0)

    def test_rejects_external_fci_solver(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        solver = direct_spin1.FCISolver(mol)

        with self.assertRaisesRegex(NotImplementedError, "external/non-GASCI"):
            gasscf.GASSCF(
                mf, 2, (1, 1), fcisolver=solver, ncore=0)

    def test_public_rdm_and_spin_plan_lifecycle(self):
        reference = fci_gas.FCISolver(gas_orbs=(2,))
        bra, ket = numpy.random.default_rng(62).normal(size=(2, 4))
        spin_ci = numpy.random.default_rng(63).normal(size=4)
        for cached in (False, True):
            with self.subTest(cache=cached):
                solver = gasscf._GASFCISolver(gas_orbs=(2,), cache_plans=cached)
                self.addCleanup(solver.close)
                rdm_plan = None
                for method in ('trans_rdm1', 'trans_rdm1s', 'trans_rdm12', 'trans_rdm12s'):
                    with self.subTest(method=method):
                        actual = getattr(solver, method)(bra, ket, 2, (1, 1))
                        expected = getattr(reference, method)(bra, ket, 2, (1, 1))
                        if method in ('trans_rdm1', 'trans_rdm1s'):
                            actual, expected = (actual,), (expected,)
                        for a, e in zip(actual, expected):
                            numpy.testing.assert_allclose(a, e, atol=1e-12, rtol=0)
                        if rdm_plan is None:
                            rdm_plan = solver._rdm_plan
                        self.assertIs(solver._rdm_plan, rdm_plan)
                ss_ci = solver.contract_ss(spin_ci, 2, (1, 1))
                spin_plan = solver._spin_plan
                ss = solver.spin_square(spin_ci, 2, (1, 1))
                numpy.testing.assert_allclose(
                    ss_ci, reference.contract_ss(spin_ci, 2, (1, 1)), atol=1e-12, rtol=0)
                numpy.testing.assert_allclose(
                    ss, reference.spin_square(spin_ci, 2, (1, 1)), atol=1e-12, rtol=0)
                self.assertIs(solver._rdm_plan, rdm_plan)
                self.assertIs(solver._spin_plan, spin_plan)
                if cached:
                    self.assertIsNotNone(rdm_plan)
                    self.assertIsNotNone(spin_plan)
                    self.assertIs(rdm_plan, solver._get_rdm_plan(2, (1, 1)))
                    self.assertIs(spin_plan, solver._get_spin_plan(2, (1, 1)))
                    self.assertEqual(rdm_plan.ndet, spin_plan.ndet)
                    self.assertIsNotNone(solver._topology_key)
                else:
                    # All three execution paths must avoid persistent caches.
                    ci = numpy.random.default_rng(64).normal(size=4)
                    solver.contract_2e(numpy.zeros((3, 3)), ci, 2, (1, 1))
                    solver.make_rdm12(ci, 2, (1, 1))
                    solver.contract_ss(ci, 2, (1, 1))
                    self.assertIsNone(solver._contract_space)
                    self.assertEqual(len(solver._contract_plans), 0)
                    self.assertIsNone(solver._rdm_plan)
                    self.assertIsNone(solver._spin_plan)
                solver.close()
                solver.close()
                self.assertIsNone(solver._rdm_plan)
                self.assertIsNone(solver._spin_plan)
                self.assertIsNone(solver._topology_key)
                if cached:
                    self.assertIsNone(rdm_plan._plan)
                    self.assertIsNone(rdm_plan.gas)

    def test_topology_change_invalidates_owned_plans(self):
        solver = gasscf._GASFCISolver(gas_orbs=(2,))
        old_plan = solver._get_rdm_plan(2, (1, 1))

        solver.gas_orbs = (1, 1)
        solver.gas_restr = [[1, 1], [2, 2]]
        solver.gas_restr_type = "cumulative-occ"
        new_plan = solver._get_rdm_plan(2, (1, 1))

        self.assertIsNot(old_plan, new_plan)
        self.assertIsNone(old_plan._plan)
        self.assertIsNone(old_plan.gas)
        solver.close()

    def test_contract_cache_preserves_real_array_validation(self):
        eri = numpy.eye(3)
        ci = numpy.array([.5, .5, .5, .5])
        expected = direct_spin1.contract_2e(eri, ci.reshape(2, 2), 2, (1, 1)).ravel()
        for cached in (False, True):
            solver = gasscf._GASFCISolver(gas_orbs=(2,), cache_plans=cached)
            self.addCleanup(solver.close)
            for warm in (False, True):
                with self.subTest(cache=cached, warm=warm):
                    # Zero imaginary parts are also rejected by GASCI.
                    for imaginary in (0., 2.):
                        with self.assertRaisesRegex(TypeError, 'real-valued'):
                            solver.contract_2e(eri + 1j * imaginary, ci, 2, (1, 1))
                    if not warm:
                        self.assertIsNone(solver._contract_space)
                    numpy.testing.assert_allclose(
                        solver.contract_2e(eri, ci, 2, (1, 1)), expected,
                        atol=1e-12, rtol=0)
                    with self.assertRaisesRegex(TypeError, 'real-valued'):
                        solver.contract_2e(eri, ci.astype(complex), 2, (1, 1))
                    with self.assertRaisesRegex(ValueError, 'CI vector size'):
                        solver.contract_2e(eri, ci[:-1], 2, (1, 1))
            numpy.testing.assert_allclose(
                solver.contract_2e(eri, ci, 2, (1, 1)), expected,
                atol=1e-12, rtol=0)

    def test_external_contract_plan_validation_through_gasscf(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.2; H 0 0 3.1',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        donor = fci_gas.FCISolver(
            gas_orbs=(1, 2), gas_restr=((0, 0), (2, 2)),
            gas_restr_type='cumulative-occ')
        h2 = fci_gas.absorb_h1e(numpy.diag([1., 2., 4.]),
                                numpy.zeros((6, 6)), 3, (1, 1), .5)
        with donor.make_space(3, (1, 1), compress_links=True) as wrong_space:
            with fci_gas.GasContractPlan(wrong_space, h2) as wrong:
                ci = numpy.ones(wrong.ndet) / numpy.sqrt(wrong.ndet)
                for cached in (False, True):
                    for sa in (False, True):
                        with self.subTest(cache=cached, sa=sa):
                            mc = gasscf.GASSCF(
                                mf, 3, (1, 1), gas_orbs=(1, 2),
                                gas_restr=((1, 1), (2, 2)),
                                gas_restr_type='cumulative-occ', cache_plans=cached)
                            self.addCleanup(mc.close)
                            if sa:
                                mc = mc.state_average((1., 0.))
                                self.addCleanup(mc.close)
                            solver = mc.fcisolver
                            with mock.patch.object(solver, '_get_contract_plan',
                                                   side_effect=AssertionError('borrowed plan cached')):
                                with self.assertRaisesRegex(ValueError, 'GAS space'):
                                    solver.contract_2e(h2, ci, 3, (1, 1), plan=wrong)
                                self.assertIsNotNone(wrong._plan)
                                with solver.make_space(3, (1, 1), compress_links=True) as gas:
                                    with fci_gas.GasContractPlan(gas, h2) as plan:
                                        expected = plan.contract(ci)
                                        actual = solver.contract_2e(h2, ci, 3, (1, 1), plan=plan)
                                        numpy.testing.assert_array_equal(actual, expected)
                                        # A borrowed plan supplies its own Hamiltonian.
                                        numpy.testing.assert_array_equal(
                                            solver.contract_2e(h2.astype(complex), ci,
                                                               3, (1, 1), plan=plan), expected)
                                        mc.close()
                                        numpy.testing.assert_array_equal(plan.contract(ci), expected)
                                        self.assertIsNotNone(gas._gas)
                                    with self.assertRaisesRegex(RuntimeError, 'plan is closed'):
                                        solver.contract_2e(h2, ci, 3, (1, 1), plan=plan)
                            self.assertIsNone(solver._contract_space)
                            self.assertEqual(len(solver._contract_plans), 0)

    def test_newton_solver_hides_native_cas_only_hooks(self):
        base = fci_gas.FCISolver(gas_orbs=(2,))
        self.assertTrue(callable(getattr(base, "gen_linkstr", None)))
        self.assertTrue(callable(getattr(
            base, "transform_ci_for_orbital_rotation", None)))

        solver = gasscf._GASFCISolver(gas_orbs=(2,))
        self.assertIsNone(getattr(solver, "gen_linkstr", None))
        self.assertIsNone(getattr(
            solver, "transform_ci_for_orbital_rotation", None))

        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        explicit = fci_gas.FCISolver(mol, gas_orbs=(2,))
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), fcisolver=explicit, ncore=0)
        self.assertIsNone(getattr(mc.fcisolver, "gen_linkstr", None))
        self.assertIsNone(getattr(
            mc.fcisolver, "transform_ci_for_orbital_rotation", None))

    def test_native_singleton_ci_list_is_accepted_by_rdm_dispatch(self):
        solver = gasscf._GASFCISolver(gas_orbs=(2,))
        ci = numpy.random.default_rng(81).normal(size=4)

        dm1, dm2 = solver.make_rdm12([ci], 2, (1, 1))
        ref_dm1, ref_dm2 = solver.make_rdm12(ci, 2, (1, 1))

        numpy.testing.assert_allclose(dm1, ref_dm1, atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(dm2, ref_dm2, atol=1e-12, rtol=0)
        with self.assertRaisesRegex(NotImplementedError,
                                    "multiroot CI density dispatch"):
            solver.make_rdm12([ci, ci], 2, (1, 1))
        solver.close()

    def test_fixed_orbital_gasci_bridge_matches_gasci_object(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)
        ref = gasci.GASCI(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None)
        ref.canonicalization = False

        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.gasci(mf.mo_coeff)
        ref_e_tot, ref_e_gas, ref_ci, _, _ = ref.kernel(mf.mo_coeff)

        self.assertAlmostEqual(e_tot, ref_e_tot, places=11)
        self.assertAlmostEqual(e_gas, ref_e_gas, places=11)
        self.assertIs(mc.mo_coeff, mo_coeff)
        self.assertIs(mc.mo_energy, mo_energy)
        self.assertIs(mc.ci, ci)
        self.assertEqual(numpy.asarray(ci).shape, numpy.asarray(ref_ci).shape)
        self.assertTrue(mc.converged)

    def test_casci_bridge_reuses_native_newton_eris(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75",
                    basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None,
            ncore=0)

        reference = mc.gasci(mf.mo_coeff)[0]
        eris = mc.ao2mo(mf.mo_coeff)
        with mock.patch.object(
                mc, "get_h2eff",
                side_effect=AssertionError("redundant active AO2MO")):
            e_tot = mc.casci(mf.mo_coeff, eris=eris)[0]

        self.assertAlmostEqual(e_tot, reference, places=11)

    def test_casci_bridge_returns_native_three_tuple(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)

        result = mc.casci(mf.mo_coeff)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], mc.e_tot, places=12)
        self.assertAlmostEqual(result[1], mc.e_gas, places=12)
        self.assertIs(result[2], mc.ci)

    def test_casci_bridge_rejects_multiroot_energy(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)
        mc.fcisolver.nroots = 2

        with self.assertRaisesRegex(RuntimeError, "Multiple roots"):
            mc.casci(mf.mo_coeff)


class TestNewton(_N2Fixture, unittest.TestCase):
    """Orbital masks, joint derivatives and native Newton integration."""

    def _check_n2_orbital_gradient_finite_difference(self, weights=None, use_df=False):
        _, mf, mo = self._n2_regression_fixture()
        mc = self._n2_regression_mc(mf)
        if use_df:
            mc = mc.density_fit()
            self.addCleanup(mc.close)
        if weights is not None:
            mc = mc.state_average(weights)
            self.addCleanup(mc.close)
        ncore, nocc = mc.ncore, mc.ncore + mc.ncas
        mask = mc.uniq_var_indices(mo.shape[1], ncore, mc.ncas, mc.frozen)
        rows, cols = numpy.where(mask)
        sectors = {
            "core-active": (rows < nocc) & (cols < ncore),
            "core-virtual": (rows >= nocc) & (cols < ncore),
            "active-virtual": (rows >= nocc) & (cols >= ncore),
            "inter-GAS": (rows < nocc) & (cols >= ncore),
        }
        rng = numpy.random.default_rng(211)
        directions = {}
        for name, selected in sectors.items():
            self.assertTrue(numpy.any(selected))
            direction = numpy.zeros(rows.size)
            direction[selected] = rng.normal(size=numpy.count_nonzero(selected))
            directions[name] = direction / numpy.linalg.norm(direction)

        # The embedded orbitals are close to a stationary singlet solution.
        # Apply a small deterministic rotation so the finite differences probe
        # nonzero gradients, rather than cancellation at that stationary point.
        # Keep the fixture's MO columns/GAS assignment; do not sort orbitals.
        displacement = .04 * sum(directions.values())
        mo = mo @ scipy.linalg.expm(mc.unpack_uniq_var(displacement))
        mc.gasci(mo)
        self.assertTrue(mc.converged)
        ci = mc.ci
        dm1, dm2 = mc.make_gasdm12(ci=ci)
        eris = mc.ao2mo(mo)
        gradient = mc.gen_g_hop(mo, ci, eris)[0][:rows.size]
        self.assertGreater(numpy.linalg.norm(gradient), 1e-3)
        # Public get_grad follows mc1step's half-gradient convention.
        # Supplying densities must not solve CI or enter joint Newton.
        with mock.patch.object(mc, 'casci', side_effect=AssertionError('CI solve')), \
                mock.patch.object(mc, 'gen_g_hop',
                                  side_effect=AssertionError('joint Newton')):
            orbital_gradient = mc.get_grad(mo, (dm1, dm2), eris)
        numpy.testing.assert_allclose(2 * orbital_gradient, gradient,
                                      atol=2e-10, rtol=1e-10)

        def energy(orbitals):
            # Independent scalar expectation value at fixed CI coefficients.
            # Do not reoptimize CI at +/-h: gen_g_hop supplies the orbital
            # partial derivative of the joint orbital/CI objective.
            # Rebuild integrals at both displacements, including on the DF path.
            h1, ecore = mc.get_h1gas(orbitals)
            h2 = ao2mo.restore(1, mc.get_h2gas(orbitals), mc.ncas)
            return (ecore + numpy.einsum('pq,qp', h1, dm1)
                    + .5 * numpy.einsum('pqrs,pqrs', h2, dm2))

        self.assertAlmostEqual(energy(mo), float(mc.e_tot), delta=2e-8)
        combined = sum(directions.values())
        directions["combined"] = combined / numpy.linalg.norm(combined)
        for name, direction in directions.items():
            with self.subTest(weights=weights, df=use_df, sector=name):
                analytic = 2 * numpy.dot(orbital_gradient, direction)
                self.assertGreater(abs(analytic), 1e-6)
                kappa = mc.unpack_uniq_var(direction)
                errors = []
                for step in (2e-4, 1e-4):
                    plus = energy(mo @ scipy.linalg.expm(step * kappa))
                    minus = energy(mo @ scipy.linalg.expm(-step * kappa))
                    numerical = (plus - minus) / (2 * step)
                    errors.append(abs(numerical - analytic))
                self.assertLess(errors[0], 8e-7)
                self.assertLess(errors[1], 2e-7)
                # Central differences have O(h^2) truncation error. Allow for
                # roundoff once the absolute error is already below 1e-8.
                self.assertLess(errors[1], .4 * errors[0] + 1e-8)

    def test_n2_orbital_gradient_finite_difference(self):
        self._check_n2_orbital_gradient_finite_difference()

    def test_n2_sa_orbital_gradient_finite_difference(self):
        for weights in ((.3, .7), (1., 0.)):
            self._check_n2_orbital_gradient_finite_difference(weights=weights)

    def test_n2_df_orbital_gradient_finite_difference(self):
        for weights in (None, (.3, .7)):
            self._check_n2_orbital_gradient_finite_difference(
                weights=weights, use_df=True)

    def test_get_grad_matches_casscf_for_cas_space(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 2.2; H 0 0 3.4',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for use_df in (False, True):
            for weights in (None, (.3, .7), (1., 0.)):
                with self.subTest(df=use_df, weights=weights):
                    mc = gasscf.GASSCF(mf, 2, (1, 1), gas_orbs=(2,))
                    self.addCleanup(mc.close)
                    ref = mc1step.CASSCF(mf, 2, (1, 1))
                    if use_df:
                        mc = mc.density_fit()
                        self.addCleanup(mc.close)
                        ref = ref.density_fit()
                    if weights is not None:
                        mc = mc.state_average(weights)
                        self.addCleanup(mc.close)
                        ref = ref.state_average(weights)
                    # Compare away from an orbital stationary point.
                    kappa = mc.unpack_uniq_var(numpy.linspace(-.08, .06, 5))
                    mo = mf.mo_coeff @ scipy.linalg.expm(kappa)
                    actual = mc.get_grad(mo)
                    expected = ref.get_grad(mo)
                    self.assertGreater(numpy.linalg.norm(expected), 1e-3)
                    numpy.testing.assert_allclose(actual, expected,
                                                  atol=2e-9, rtol=1e-8)

    def test_get_grad_preserves_gas_mask_and_fixed_density(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 2.2; H 0 0 3.4',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for use_df in (False, True):
            mc = gasscf.GASSCF(
                mf, 2, (1, 1), gas_orbs=(1, 1),
                gas_restr=((1, 1), (2, 2)), gas_restr_type='cumulative-occ')
            self.addCleanup(mc.close)
            if use_df:
                mc = mc.density_fit()
                self.addCleanup(mc.close)
            mc.gasci()
            densities = mc.make_gasdm12()
            eris = mc.ao2mo()
            mo = mc.mo_coeff
            full_mask = mc.uniq_var_indices(4, 1, 2, None)
            self.assertTrue(full_mask[2, 1])  # inter-GAS rotation
            full = mc.get_grad(mo, densities, eris)
            mc.fix_spin_(shift=.2, ss=0.)
            # At fixed CI, the spin penalty has no orbital derivative.
            # Supplied densities/integrals remain usable without CI data.
            mc.ci = None
            with mock.patch.object(mc, 'casci', side_effect=AssertionError('CI solve')), \
                    mock.patch.object(mc, 'ao2mo',
                                      side_effect=AssertionError('AO2MO')):
                # Supplying densities and ERIs must not bypass MO validation.
                with self.assertRaisesRegex(ValueError, 'orthonormal'):
                    mc.get_grad(mo * 1.01, densities, eris)
                self.assertIs(mc.mo_coeff, mo)
                for frozen in (None, 1, [3]):
                    with self.subTest(df=use_df, frozen=frozen):
                        mc.frozen = frozen
                        mask = mc.uniq_var_indices(4, 1, 2, frozen)
                        actual = mc.get_grad(casdm1_casdm2=densities, eris=eris)
                        numpy.testing.assert_allclose(actual, full[mask[full_mask]],
                                                      atol=1e-12, rtol=1e-12)

    def test_n2_numerical_regression_ss_and_zero_weight_sa(self):
        _, mf, mo = self._n2_regression_fixture()
        ss = self._n2_regression_mc(mf)
        e_ss = float(ss.kernel(mo)[0])

        _, mf, mo = self._n2_regression_fixture()
        zw = self._n2_regression_mc(mf).state_average((1.0, 0.0))
        self.addCleanup(zw.close)
        zw.fcisolver.spin = 0
        zw.fcisolver.max_cycle = 300
        zw.fcisolver.max_space = 30
        zw.fcisolver.conv_tol = 1e-10
        e_zw = float(zw.kernel(mo)[0])

        self.assertTrue(ss.converged)
        self.assertTrue(zw.converged)
        self.assertAlmostEqual(
            e_ss, self.N2_REF_SS_ENERGY,
            delta=self.N2_REGRESSION_TOL)
        self.assertAlmostEqual(
            e_zw, self.N2_REF_SS_ENERGY,
            delta=self.N2_REGRESSION_TOL)
        self.assertAlmostEqual(
            e_zw, e_ss, delta=self.N2_REGRESSION_TOL)

    def test_n2_numerical_regression_state_average(self):
        _, mf, mo = self._n2_regression_fixture()
        mc = self._n2_regression_mc(mf).state_average((0.5, 0.5))
        mc.fcisolver.spin = 0
        mc.fcisolver.max_cycle = 300
        mc.fcisolver.max_space = 30
        self.addCleanup(mc.close)
        # This is an energy regression at 1e-7 Eh, not a test of a particular
        # Newton trajectory. Tighter outer thresholds can linger in a noisy
        # SA tail and occasionally exhaust the macro limit across platforms.
        # Keep the CI residual well below the orbital stopping threshold;
        # Davidson otherwise defaults to sqrt(conv_tol) for its residual.
        mc.conv_tol = 1e-9
        mc.conv_tol_grad = 1e-4
        mc.fcisolver.conv_tol = 1e-12
        mc.fcisolver.conv_tol_residual = 1e-7
        progress = {}

        def record_progress(env):
            for key in ('imacro', 'de', 'norm_gall'):
                progress[key] = env[key]

        e_tot = float(mc.kernel(mo, callback=record_progress)[0])
        weighted = float(numpy.dot(
            numpy.asarray(mc.weights, dtype=float),
            numpy.asarray(mc.e_states, dtype=float),
        ))

        self.assertTrue(mc.converged,
                        "N2 SA failed to converge: E=%.15f, last macro=%s" %
                        (e_tot, progress))
        self.assertTrue(numpy.all(mc.fcisolver.converged),
                        "N2 SA final CI roots did not converge")
        self.assertAlmostEqual(
            e_tot, self.N2_REF_SA_HALF_HALF_ENERGY,
            delta=self.N2_REGRESSION_TOL)
        self.assertAlmostEqual(
            e_tot, weighted, delta=self.N2_REGRESSION_TOL)

    def test_n2_numerical_regression_density_fit_entry_points(self):
        _, mf, mo = self._n2_regression_fixture()
        by_method = self._n2_regression_mc(mf).density_fit()
        self.addCleanup(by_method.close)
        by_method.fcisolver.spin = 0
        by_method.fcisolver.max_cycle = 300
        by_method.fcisolver.max_space = 30
        by_method.fcisolver.conv_tol = 1e-10
        e_method = float(by_method.kernel(mo)[0])

        _, mf, mo = self._n2_regression_fixture()
        by_factory = gasscf.DFGASSCF(
            mf, 8, (5, 5),
            gas_orbs=(2, 4, 2),
            gas_restr=((2, 4), (7, 9), (10, 10)),
            gas_restr_type="cumulative-occ",
            ncore=2,
        )
        self.addCleanup(by_factory.close)
        by_factory.verbose = 0
        by_factory.canonicalization = False
        by_factory.max_cycle_macro = 50
        by_factory.max_cycle_micro = 10
        by_factory.conv_tol = 1e-10
        by_factory.conv_tol_grad = 1e-5
        by_factory.fcisolver.spin = 0
        by_factory.fcisolver.max_cycle = 300
        by_factory.fcisolver.max_space = 30
        by_factory.fcisolver.conv_tol = 1e-10
        e_factory = float(by_factory.kernel(mo)[0])

        self.assertTrue(by_method.converged)
        self.assertTrue(by_factory.converged)
        self.assertAlmostEqual(
            e_method, self.N2_REF_DF_ENERGY,
            delta=self.N2_REGRESSION_TOL)
        self.assertAlmostEqual(
            e_factory, self.N2_REF_DF_ENERGY,
            delta=self.N2_REGRESSION_TOL)
        self.assertAlmostEqual(
            e_method, e_factory,
            delta=self.N2_REGRESSION_TOL)

    def test_gas_orbital_rotation_mask_enables_only_intergas_internal_rotations(self):
        mol = gto.M(
            atom="; ".join("H 0 0 %g" % value for value in range(6)),
            basis="sto-3g",
            verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 4, (2, 2), gas_orbs=(1, 2, 1), gas_restr=None,
            ncore=1)

        mask = mc.uniq_var_indices(6, 1, 4, None)

        self.assertEqual(mask.shape, (6, 6))
        self.assertEqual(numpy.count_nonzero(mask), 14)

        # Native CASSCF-like external/core rotations are retained.
        self.assertTrue(mask[1, 0])
        self.assertTrue(mask[4, 0])
        self.assertTrue(mask[5, 0])
        self.assertTrue(mask[5, 1])
        self.assertTrue(mask[5, 4])

        # Active-active rotations are enabled only between GAS subspaces in the
        # lower-triangular orbital-rotation convention used by pack_uniq_var.
        self.assertTrue(mask[2, 1])
        self.assertTrue(mask[3, 1])
        self.assertTrue(mask[4, 1])
        self.assertTrue(mask[4, 2])
        self.assertTrue(mask[4, 3])

        # Same-subspace active rotations and opposite triangular entries remain
        # inactive.
        self.assertFalse(mask[3, 2])
        self.assertFalse(mask[1, 2])
        self.assertFalse(mask[0, 1])
        self.assertFalse(mask[1, 5])

    def test_gas_as_cas_mask_has_no_active_internal_rotation(self):
        mol = gto.M(
            atom="; ".join("H 0 0 %g" % value for value in range(6)),
            basis="sto-3g",
            verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 4, (2, 2), gas_orbs=(4,), gas_restr=None, ncore=1)

        mask = mc.uniq_var_indices(6, 1, 4, None)

        self.assertEqual(numpy.count_nonzero(mask), 9)
        active = mask[1:5, 1:5]
        self.assertFalse(numpy.any(active[numpy.tril_indices(4, -1)]))

    def test_gas_orbital_rotation_mask_honors_frozen(self):
        mol = gto.M(
            atom="; ".join("H 0 0 %g" % value for value in range(6)),
            basis="sto-3g",
            verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 4, (2, 2), gas_orbs=(1, 2, 1), gas_restr=None,
            ncore=1)

        mask = mc.uniq_var_indices(6, 1, 4, [2])

        self.assertFalse(numpy.any(mask[2]))
        self.assertFalse(numpy.any(mask[:, 2]))
        self.assertTrue(mask[4, 1])
        self.assertTrue(mask[5, 4])
        self.assertTrue(mask[3, 1])
        self.assertTrue(mask[5, 3])

    def test_full_kernel_gas_as_cas_smoke_matches_newton_casscf(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=1)
        ref = newton_casscf.CASSCF(mf, 2, (1, 1), ncore=1)
        for obj in (mc, ref):
            obj.max_cycle_macro = 12
            obj.max_cycle_micro = 4
            obj.conv_tol = 1e-9
            obj.conv_tol_grad = 1e-5
            obj.canonicalization = False

        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.kernel(mf.mo_coeff)
        ref_e_tot, ref_e_cas, _, _, _ = ref.kernel(mf.mo_coeff)

        self.assertEqual(mo_coeff.shape, mf.mo_coeff.shape)
        self.assertIsNone(mo_energy)
        self.assertIs(mc.mo_coeff, mo_coeff)
        self.assertIs(mc.mo_energy, mo_energy)
        self.assertIs(mc.ci, ci)
        self.assertEqual(mc.converged, ref.converged)
        self.assertAlmostEqual(e_tot, ref_e_tot, places=7)
        self.assertAlmostEqual(e_gas, ref_e_cas, places=7)

    def test_restricted_gas_kernel_with_and_without_canonicalization(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.2; H 0 0 3.1',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        for canonicalization in (False, True):
            with self.subTest(canonicalization=canonicalization):
                mc = gasscf.GASSCF(
                    mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
                    gas_restr_type='cumulative-occ', ncore=1)
                self.addCleanup(mc.close)
                mc.max_cycle_macro = mc.max_cycle_micro = 1
                mc.conv_tol, mc.conv_tol_grad = 1e-8, 1e-4
                if not canonicalization:
                    mc.canonicalization = False
                mask = mc.uniq_var_indices(mf.mo_coeff.shape[1], mc.ncore, mc.ncas, mc.frozen)
                self.assertTrue(mask[mc.ncore + 1, mc.ncore])
                e_tot, e_gas, ci, mo, mo_energy = mc.kernel(mf.mo_coeff)
                ndet = mc.fcisolver.space_info(mc.ncas, mc.nelecas)['ndet_estimate']
                self.assertTrue(numpy.isfinite(e_tot))
                self.assertTrue(numpy.isfinite(e_gas))
                self.assertEqual(numpy.asarray(ci).shape, (ndet,))
                self.assertEqual(mo.shape, mf.mo_coeff.shape)
                if canonicalization:
                    self.assertEqual(mo_energy.shape, (mf.mo_coeff.shape[1],))
                else:
                    self.assertIsNone(mo_energy)

    def test_validate_capabilities_sets_internal_rotation_policy(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)

        gas_as_cas = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)
        gas_as_cas.internal_rotation = True
        self.assertIs(gas_as_cas.newton(), gas_as_cas)
        self.assertFalse(gas_as_cas.internal_rotation)

        restricted = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", ncore=0)
        self.assertIs(restricted.validate_capabilities(), restricted)
        self.assertTrue(restricted.internal_rotation)

    def test_full_active_kernel_preserves_ci_convergence(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for use_df in (False, True):
            for weights in (None, (.5, .5), (1., 0.)):
                for canonical in (False, True):
                    with self.subTest(df=use_df, weights=weights, canonical=canonical):
                        mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=0)
                        self.addCleanup(mc.close)
                        if use_df:
                            mc = mc.density_fit()
                            self.addCleanup(mc.close)
                        if weights is not None:
                            mc = mc.state_average(weights)
                            self.addCleanup(mc.close)
                        mc.canonicalization = canonical
                        solve = mc.fcisolver.kernel
                        # Keep real CI vectors and energies, but control the
                        # convergence flag independently of a tiny exact solve.
                        for converged in (False, True):
                            flags = converged if weights is None else [True, converged]

                            def controlled_solve(*args, **kwargs):
                                result = solve(*args, **kwargs)
                                mc.fcisolver.converged = flags
                                return result

                            with mock.patch.object(mc.fcisolver, 'kernel',
                                                   side_effect=controlled_solve):
                                result = mc.kernel()
                            self.assertIs(mc.converged, converged)
                            self.assertTrue(numpy.isfinite(result[0]))
                            self.assertEqual(result[4] is None, not canonical)

    def test_full_active_gas_as_cas_kernel_without_canonicalization(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)
        ref = gasci.GASCI(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None)
        for obj in (mc, ref):
            obj.canonicalization = False

        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.kernel(mf.mo_coeff)
        ref_e_tot, ref_e_gas, ref_ci, _, _ = ref.kernel(mf.mo_coeff)

        self.assertFalse(mc.internal_rotation)
        self.assertAlmostEqual(e_tot, ref_e_tot, places=11)
        self.assertAlmostEqual(e_gas, ref_e_gas, places=11)
        self.assertEqual(numpy.asarray(ci).shape, numpy.asarray(ref_ci).shape)
        self.assertEqual(mo_coeff.shape, mf.mo_coeff.shape)
        self.assertIsNone(mo_energy)

    def test_joint_newton_curvature_against_full_fci_energy(self):
        # Nonstationary restricted GAS with core, virtual and inter-GAS rotations.
        mol = gto.M(atom=';'.join('H 0 0 %s' % z for z in
                                 (0., .8, 1.8, 2.6, 3.7, 4.8)),
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        cases = ((False, None, None, None),
                 (True, None, None, [5]),
                 (False, (.3, .7), 0., None),
                 (True, (.3, .7), 2., None),
                 (False, (.4, 0., .6), 2., [5]),
                 (True, (.4, 0., .6), 0., [5]))
        for use_df, weights, target, frozen in cases:
            with self.subTest(df=use_df, weights=weights, target=target, frozen=frozen):
                with ExitStack() as resources:
                    mc = gasscf.GASSCF(
                        mf, 4, (2, 2), ncore=1, gas_orbs=(1, 2, 1),
                        gas_restr=((0, 1), (3, 4), (4, 4)),
                        gas_restr_type='cumulative-occ', frozen=frozen)
                    resources.callback(mc.close)
                    if use_df:
                        mc = mc.density_fit(auxbasis='weigend')
                        resources.callback(mc.close)
                    if weights is not None:
                        mc = mc.state_average(weights)
                        resources.callback(mc.close)
                    if target is not None:
                        mc.fix_spin_(shift=.2, ss=target)
                    mc.validate_capabilities()
                    space = resources.enter_context(
                        mc.fcisolver.make_space(mc.ncas, mc.nelecas))
                    self.assertEqual(space.ndet, 19)
                    rng = numpy.random.default_rng(62091)
                    columns = numpy.linalg.qr(rng.normal(
                        size=(space.ndet, mc.fcisolver.nroots)))[0]
                    roots = [column.copy() for column in columns.T]
                    ci = roots[0] if weights is None else roots
                    ngorb = int(mc.uniq_var_indices(6, 1, 4, frozen).sum())
                    mo = mf.mo_coeff @ scipy.linalg.expm(
                        mc.unpack_uniq_var(.05 * rng.normal(size=ngorb)))
                    gradient, update, hop, _ = mc.gen_g_hop(mo, ci, mc.ao2mo(mo))
                    vo = rng.normal(size=ngorb)
                    vo /= numpy.linalg.norm(vo)
                    vc = []
                    for c in roots:
                        v = rng.normal(size=c.size)
                        v -= c.dot(v) * c
                        vc.append(v / numpy.linalg.norm(v))
                    orbital = numpy.r_[vo, numpy.zeros(space.ndet * len(roots))]
                    ci_direction = numpy.r_[numpy.zeros(ngorb), numpy.concatenate(vc)]
                    joint = orbital + ci_direction
                    joint /= numpy.linalg.norm(joint)

                    def energy(x):
                        # No CI solve, tested HVP or GAS RDM in this reference.
                        orbitals = mo @ scipy.linalg.expm(mc.unpack_uniq_var(x[:ngorb]))
                        h1, core = mc.get_h1eff(orbitals)
                        h2 = mc.get_h2eff(orbitals)
                        hamiltonian = direct_spin1.absorb_h1e(h1, h2, 4, (2, 2), .5)
                        value = core
                        for i, (weight, c) in enumerate(zip(
                                (1.,) if weights is None else weights, roots)):
                            displaced = c + x[ngorb + i * space.ndet:
                                              ngorb + (i + 1) * space.ndet]
                            full = fci_gas.gas2fci(
                                displaced / numpy.linalg.norm(displaced), space)
                            hfull = direct_spin1.contract_2e(hamiltonian, full, 4, (2, 2))
                            if target is not None:
                                penalty = spin_op.contract_ss(full, 4, (2, 2)) - target * full
                                if target != 0.:
                                    penalty = (spin_op.contract_ss(penalty, 4, (2, 2))
                                               - target * penalty)
                                hfull += .2 * penalty
                            value += weight * numpy.vdot(full, hfull).real
                        return float(value)

                    center = energy(numpy.zeros_like(gradient))
                    for name, direction in (('orbital', orbital), ('ci', ci_direction),
                                            ('joint', joint)):
                        slope = gradient @ direction
                        curvature = direction @ hop(direction)
                        self.assertGreater(abs(curvature), 1e-4)
                        for step in (2e-4, 1e-4):
                            with self.subTest(direction=name, step=step):
                                plus, minus = energy(step * direction), energy(-step * direction)
                                self.assertAlmostEqual((plus - minus) / (2 * step),
                                                       slope, delta=3e-7)
                                self.assertAlmostEqual((plus - 2 * center + minus) / step**2,
                                                       curvature, delta=3e-6)
                    mixed = orbital @ hop(ci_direction)
                    self.assertGreater(abs(mixed), 1e-4)
                    for step in (2e-4, 1e-4):
                        with self.subTest(direction='mixed', step=step):
                            cross = (energy(step * (orbital + ci_direction))
                                     - energy(step * (orbital - ci_direction))
                                     - energy(step * (-orbital + ci_direction))
                                     + energy(-step * (orbital + ci_direction))) / (4 * step**2)
                            self.assertAlmostEqual(cross, mixed, delta=3e-6)
                    x, y = rng.normal(size=(2, gradient.size))
                    self.assertAlmostEqual(x @ hop(y), y @ hop(x), delta=2e-11)
                    numpy.testing.assert_allclose(update(numpy.eye(6), ci), gradient,
                                                  atol=2e-11, rtol=0)
                    if weights is not None and 0. in weights:
                        index = weights.index(0.)
                        block = slice(ngorb + index * space.ndet,
                                      ngorb + (index + 1) * space.ndet)
                        zero_root = numpy.zeros_like(gradient)
                        zero_root[block] = rng.normal(size=space.ndet)
                        for value in (gradient[block], hop(x)[block], hop(zero_root)):
                            numpy.testing.assert_allclose(value, 0., atol=2e-11, rtol=0)

    def _check_spin_penalty_newton_derivatives(self, use_df=False, weights=None):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
            gas_restr=((0, 1), (2, 2)), gas_restr_type='cumulative-occ')
        self.addCleanup(mc.close)
        if use_df:
            mc = mc.density_fit()
            self.addCleanup(mc.close)
        if weights is not None:
            mc = mc.state_average(weights)
            self.addCleanup(mc.close)
        mc.validate_capabilities()
        mo = mf.mo_coeff
        eris = mc.ao2mo(mo)
        h1, core = mc.get_h1gas(mo)
        h2 = mc.get_h2gas(mo)
        absorbed = direct_spin1.absorb_h1e(h1, h2, mc.ncas, mc.nelecas, .5)
        # Independent full-FCI operators projected into a genuine restricted
        # GAS space. Deliberately mix spin sectors: pure-spin eigenvectors can
        # hide a missing penalty in the CI gradient and curvature.
        with mc.fcisolver.make_space(mc.ncas, mc.nelecas) as space:
            eye = numpy.eye(space.ndet)
            def project_action(action, column):
                full = fci_gas.gas2fci(column, space)
                return fci_gas.fci2gas(action(full), space).ravel()
            hmat = numpy.column_stack([
                project_action(lambda c: direct_spin1.contract_2e(
                    absorbed, c, mc.ncas, mc.nelecas), c) for c in eye])
            s2 = numpy.column_stack([
                project_action(lambda c: spin_op.contract_ss(
                    c, mc.ncas, mc.nelecas), c) for c in eye])
        nroots = mc.fcisolver.nroots
        rng = numpy.random.default_rng(211)
        columns = numpy.linalg.qr(rng.normal(size=(eye.shape[0], nroots)))[0]
        roots = [columns[:, i].copy() for i in range(nroots)]
        directions = []
        for c in roots:
            v = rng.normal(size=c.size)
            v -= c.dot(v) * c
            directions.append(v / numpy.linalg.norm(v))
        ci = roots[0] if weights is None else roots
        w = (1.,) if weights is None else weights
        native = newton_casscf.gen_g_hop(mc, mo, ci, eris)
        ngorb = native[0].size - sum(c.size for c in roots)
        ci_direction = numpy.r_[numpy.zeros(ngorb), numpy.concatenate(directions)]
        orbital_direction = numpy.r_[rng.normal(size=ngorb),
                                    numpy.zeros(native[0].size-ngorb)]
        u = scipy.linalg.expm(.02 * mc.unpack_uniq_var(orbital_direction[:ngorb]))

        for target in (0., 2.):  # Linear minimum-spin and quadratic penalties.
            mc.fix_spin_(shift=.2, ss=target)
            delta = s2 - target * eye
            penalty = .2 * (delta if target == 0 else delta @ delta)
            operator = hmat + penalty
            def energy(angle):
                vectors = [numpy.cos(angle)*c + numpy.sin(angle)*v
                           for c, v in zip(roots, directions)]
                return core + sum(a * c.dot(operator @ c)
                                  for a, c in zip(w, vectors))
            previous = None
            for cache in (True, False):
                mc.cache_plans = cache
                with self.subTest(df=use_df, weights=weights,
                                  target=target, cache=cache):
                    g, update, hop, diagonal = mc.gen_g_hop(mo, ci, eris)
                    # Public contraction remains physical, even with fix_spin.
                    numpy.testing.assert_allclose(
                        mc.fcisolver.contract_2e(absorbed, roots[0],
                                                mc.ncas, mc.nelecas),
                        hmat @ roots[0], atol=1e-12, rtol=0)
                    first = numpy.dot(g, ci_direction)
                    second = numpy.dot(ci_direction, hop(ci_direction))
                    self.assertGreater(abs(first-native[0].dot(ci_direction)), 1e-3)
                    for step in (2e-4, 1e-4):
                        plus, minus, center = energy(step), energy(-step), energy(0)
                        self.assertAlmostEqual(first, (plus-minus)/(2*step), delta=2e-7)
                        self.assertAlmostEqual(second, (plus-2*center+minus)/step**2,
                                               delta=2e-6)
                    # Compare the full tangent CI response with an independent
                    # projected full-FCI matrix, not just one quadratic form.
                    response = hop(ci_direction)[ngorb:]
                    start = 0
                    for a, c, v in zip(w, roots, directions):
                        block = response[start:start+c.size]
                        expected = 2*a*(operator @ v - c.dot(operator @ c)*v)
                        numpy.testing.assert_allclose(
                            block-c*c.dot(block), expected-c*c.dot(expected),
                            atol=1e-11, rtol=0)
                        start += c.size
                    # P is orbital independent: no extra orbital or mixed block.
                    numpy.testing.assert_allclose(g[:ngorb], native[0][:ngorb], atol=0, rtol=0)
                    numpy.testing.assert_allclose(hop(orbital_direction),
                                                  native[2](orbital_direction), atol=1e-12, rtol=0)
                    numpy.testing.assert_allclose(hop(ci_direction)[:ngorb],
                                                  native[2](ci_direction)[:ngorb], atol=1e-12, rtol=0)
                    # Check the keyframe update at changed, unnormalized CI and
                    # rotated orbitals. Only the normalized CI penalty changes.
                    current = [(1.2+i)*(c+.13*v)
                               for i, (c, v) in enumerate(zip(roots, directions))]
                    current_arg = current[0] if weights is None else current
                    actual = update(u, current_arg)
                    expected = native[1](u, current_arg)
                    start = ngorb
                    for a, c in zip(w, current):
                        c = c / numpy.linalg.norm(c)
                        expected[start:start+c.size] += 2*a*(penalty @ c-c.dot(penalty @ c)*c)
                        start += c.size
                    numpy.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
                    # Quadratic diagonal is a documented preconditioner
                    # approximation, shared with Davidson; the HVP is exact.
                    pd = .2 * (s2.diagonal()-target)
                    if target != 0:
                        pd = .2 * (s2.diagonal()-target)**2
                    expected = native[3].copy()
                    start = ngorb
                    for a, c in zip(w, roots):
                        ep = c.dot(penalty @ c)
                        residual = penalty @ c-ep*c
                        expected[start:start+c.size] += 2*a*(pd-ep-2*residual*c)
                        start += c.size
                    numpy.testing.assert_allclose(diagonal, expected, atol=1e-12, rtol=0)
                    if weights is not None and weights[-1] == 0:
                        numpy.testing.assert_allclose(g[-roots[-1].size:], 0, atol=0, rtol=0)
                        numpy.testing.assert_allclose(hop(ci_direction)[-roots[-1].size:], 0, atol=0, rtol=0)
                    values = (g, hop(ci_direction+orbital_direction), diagonal)
                    if previous is not None:
                        for a, b in zip(values, previous):
                            numpy.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
                    previous = values
            mc.undo_fix_spin_()

    def test_spin_penalty_newton_derivatives(self):
        self._check_spin_penalty_newton_derivatives()

    def test_spin_penalty_sa_newton_derivatives(self):
        for weights in ((.3, .7), (1., 0.)):
            self._check_spin_penalty_newton_derivatives(weights=weights)

    def test_spin_penalty_df_newton_derivatives(self):
        for weights in (None, (.3, .7), (1., 0.)):
            self._check_spin_penalty_newton_derivatives(use_df=True, weights=weights)

    def test_automatic_df_energy_and_orbital_derivative(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis='weigend').run()
        self.addCleanup(mf._chkfile.close)
        mc = gasscf.GASSCF(mf, 2, (1, 1))
        self.addCleanup(mc.close)
        ref = mcscf.CASSCF(mf, 2, (1, 1))
        direction = numpy.random.default_rng(32).normal(size=5)
        direction /= numpy.linalg.norm(direction)
        kappa = mc.unpack_uniq_var(direction)
        mo = mf.mo_coeff @ scipy.linalg.expm(.05 * kappa)
        energy, _, ci = mc.gasci(mo)[:3]
        expected = ref.casci(mo, eris=ref.ao2mo(mo))[0]
        self.assertAlmostEqual(energy, expected, places=10)
        dm1, dm2 = mc.fcisolver.make_rdm12(ci, 2, (1, 1))
        analytic = 2 * mc.get_grad(mo, (dm1, dm2)).dot(direction)

        def fixed_ci_energy(orbitals):
            h1, ecore = mc.get_h1eff(orbitals)
            h2 = ao2mo.restore(1, mc.get_h2eff(orbitals), 2)
            return (ecore + numpy.einsum('pq,qp', h1, dm1)
                    + .5 * numpy.einsum('pqrs,pqrs', h2, dm2))

        for step in (2e-4, 1e-4):
            plus = mo @ scipy.linalg.expm(step * kappa)
            minus = mo @ scipy.linalg.expm(-step * kappa)
            numerical = (fixed_ci_energy(plus) - fixed_ci_energy(minus)) / (2 * step)
            self.assertAlmostEqual(analytic, numerical, delta=2e-8)


class TestStateAverageAndSpin(unittest.TestCase):
    """State averaging, spin penalty and physical versus objective energies."""

    def test_state_average_constructs_zero_weight_roots_and_undoes(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=1)

        sa = mc.state_average((1.0, 0.0))

        self.assertIsInstance(sa, addons.StateAverageMCSCF)
        self.assertIsInstance(sa, gasscf.GASSCF)
        numpy.testing.assert_allclose(sa.weights, (1.0, 0.0), atol=0, rtol=0)
        self.assertEqual(sa.fcisolver.nroots, 2)
        self.assertEqual(sa.gas_orbs, (2,))
        self.assertIs(sa.validate_capabilities(), sa)

        self.addCleanup(mc.close)
        self.addCleanup(sa.close)
        self.addCleanup(mf._chkfile.close)
        plan = sa.fcisolver._get_rdm_plan(2, (1, 1))
        undone = sa.undo_state_average()
        self.addCleanup(undone.close)
        self.assertIsInstance(undone, gasscf.GASSCF)
        self.assertNotIsInstance(undone, addons.StateAverageMCSCF)
        self.assertEqual(undone.fcisolver.nroots, 1)
        self.assertIs(undone.validate_capabilities(), undone)
        self.assertIsNotNone(plan._plan)
        # The trailing-underscore operation still releases the replaced owner.
        old_solver = sa.fcisolver
        self.assertIs(sa.state_average_((.3, .7)), sa)
        self.assertIsNone(plan._plan)
        self.assertIsNot(sa.fcisolver, old_solver)
        self.assertEqual(sa.weights, (.3, .7))
        self.assertIsNone(sa.fcisolver._rdm_plan)

    def test_state_average_rejects_invalid_weights_and_wfnsym(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)

        for weights in ((), (1.0,), ((.5, .5),), (0.7, 0.4), (1.1, -0.1),
                        (float("nan"), 1.0), (float("inf"), 0.),
                        (.5, .5 + 2e-10)):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "weights"):
                    mc.state_average(weights)
        for weights in ((1., 0.), (0., 1.), (.5, .5 + 5e-11)):
            result = mc._validate_weights(weights)
            self.assertIsInstance(result, tuple)
            self.assertEqual(result, weights)
        with self.assertRaisesRegex(NotImplementedError, "wfnsym"):
            mc.state_average((0.5, 0.5), wfnsym=0)

    def test_sa_then_df_keeps_physical_energy_adapters(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.2; H 0 0 3.1',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for weights in ((.4, .6), (1., 0.)):
            mc = gasscf.GASSCF(
                mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                gas_restr=((0, 1), (2, 2)), gas_restr_type='cumulative-occ')
            mc = mc.state_average(weights).density_fit()
            self.addCleanup(mc.close)
            mc.fix_spin_(shift=.001, ss=0.)
            mc.max_cycle_macro = mc.max_cycle_micro = 1
            mc.canonicalization = False
            self.assertIs(mc.validate_capabilities(), mc)
            self._check_physical_energy_result(mc, mc.gasci())
            self._check_physical_energy_result(mc, mc.kernel())

    def test_state_average_kernel_gas_as_cas_smoke(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=1)
        mc = mc.state_average((1.0, 0.0))
        mc.max_cycle_macro = 1
        mc.max_cycle_micro = 1
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-4
        mc.canonicalization = False

        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.kernel(mf.mo_coeff)

        self.assertTrue(numpy.isfinite(e_tot))
        self.assertTrue(numpy.isfinite(e_gas))
        self.assertEqual(len(ci), 2)
        self.assertEqual(len(mc.e_states), 2)
        self.assertEqual(mo_coeff.shape, mf.mo_coeff.shape)
        self.assertIsNone(mo_energy)

    def test_fix_spin_sets_gas_native_penalty_and_undoes(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)

        self.assertIs(mc.fix_spin_(shift=.15, ss=0), mc)

        self.assertFalse(isinstance(mc.fcisolver, fci_addons.SpinPenaltyFCISolver))
        self.assertEqual(mc.fcisolver.ss_penalty, .15)
        self.assertEqual(mc.fcisolver.ss_value, 0.0)
        self.assertIsNone(getattr(mc.fcisolver, "gen_linkstr"))
        self.assertIsNone(getattr(mc.fcisolver, "transform_ci_for_orbital_rotation"))
        self.assertIs(mc.validate_capabilities(), mc)

        self.assertIs(mc.undo_fix_spin_(), mc)
        self.assertFalse(hasattr(mc.fcisolver, "ss_penalty"))
        self.assertFalse(hasattr(mc.fcisolver, "ss_value"))

        copied = mc.copy().fix_spin(shift=.25, ss=0)
        self.assertIsNot(copied, mc)
        self.assertIsNot(copied.fcisolver, mc.fcisolver)
        self.assertTrue(hasattr(copied.fcisolver, "ss_penalty"))
        self.assertFalse(hasattr(mc.fcisolver, "ss_penalty"))

    def test_fix_spin_is_in_place_for_plain_df_and_sa_objects(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        for kind in ("plain", "df", "sa"):
            for name in ("fix_spin", "fix_spin_"):
                with self.subTest(kind=kind, method=name):
                    mc = gasscf.GASSCF(
                        mf, 2, (1, 1), gas_orbs=(2,), ncore=0)
                    if kind == "df":
                        mc = mc.density_fit()
                    elif kind == "sa":
                        mc = mc.state_average((.25, .75))
                    self.addCleanup(mc.close)
                    cls, solver = mc.__class__, mc.fcisolver
                    ci = numpy.array([1., 0., 0., 0.])
                    mc.make_gasdm1(ci=ci, state=0)
                    self.assertIsNotNone(solver._rdm_plan)

                    result = getattr(mc, name)(shift=.15, ss=0)
                    self.assertIs(result, mc)
                    self.assertIs(mc.__class__, cls)
                    self.assertIs(mc.fcisolver, solver)
                    self.assertEqual(solver.ss_penalty, .15)
                    self.assertEqual(solver.ss_value, 0.)
                    self.assertIsNone(solver._rdm_plan)
                    self.assertNotIsInstance(solver, fci_addons.SpinPenaltyFCISolver)
                    # A second call updates the same object even if its return
                    # value is ignored, matching the GASCI public convention.
                    getattr(mc, name)(shift=.25, ss=0)
                    self.assertEqual(solver.ss_penalty, .25)
                    self.assertIs(mc.undo_fix_spin_(), mc)
                    self.assertFalse(hasattr(solver, "ss_penalty"))
                    if kind == "sa":
                        numpy.testing.assert_array_equal(mc.weights, (.25, .75))

    def test_fix_spin_rejects_spin_incomplete_gas_and_invalid_target(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        incomplete = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 0, 0, 1]],
            gas_restr_type="spin-supergroup", ncore=0)

        with self.assertRaisesRegex(ValueError, "spin-complete"):
            incomplete.fix_spin_(shift=.2, ss=0)

        complete = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)
        with self.assertRaisesRegex(ValueError, "target S"):
            complete.fix_spin_(shift=.2, ss=.5)

    def test_spin_penalty_kernel_uses_local_newton_response(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        original = tuple(getattr(newton_casscf, name) for name in
                         ('kernel', 'update_orb_ci', 'gen_g_hop'))
        for weights in (None, (.3, .7), (1., 0.)):
            mc = gasscf.GASSCF(mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                              gas_restr=((0, 1), (2, 2)),
                              gas_restr_type='cumulative-occ')
            if weights is not None:
                mc = mc.state_average(weights)
            self.addCleanup(mc.close)
            mc.fix_spin_(shift=.2, ss=0.)
            mc.max_cycle_macro = mc.max_cycle_micro = 1
            seen = []
            def callback(env):
                # Native module functions stay unchanged even while the
                # calculation is running, not merely restored on return.
                for name, function in zip(('kernel', 'update_orb_ci', 'gen_g_hop'), original):
                    self.assertIs(getattr(newton_casscf, name), function)
                self.assertAlmostEqual(env['e_tot'], mc.spin_energy_report()['objective'], 10)
                seen.append(True)
            with mock.patch.object(gasscf, 'gen_g_hop', wraps=gasscf.gen_g_hop) as response:
                result = mc.kernel(callback=callback)
                self.assertGreater(response.call_count, 0)
            self.assertTrue(seen)
            self.assertAlmostEqual(result[0], mc.spin_energy_report()['physical'], 10)

    def _check_physical_energy_result(self, mc, result):
        report = mc.spin_energy_report()
        roots = mc.ci if isinstance(mc.ci, (list, tuple)) else [mc.ci]
        h1, core = mc.get_h1gas(mc.mo_coeff)
        h2 = ao2mo.restore(1, mc.get_h2gas(mc.mo_coeff), mc.ncas)
        physical = []
        with mc.fcisolver.make_rdm_plan(mc.ncas, mc.nelecas) as plan:
            for ci in roots:
                d1, d2 = plan.make_rdm12(ci, ci)
                physical.append(core + numpy.einsum("pq,qp", h1, d1)
                                + .5 * numpy.einsum("pqrs,pqrs", h2, d2))
        numpy.testing.assert_allclose(report["root_physical"], physical, atol=1e-9, rtol=0)
        self.assertGreater(max(report["root_penalty"]), 1e-3)
        expected = numpy.dot(getattr(mc, "weights", (1.,)), physical)
        self.assertAlmostEqual(result[0], expected, 9)
        self.assertAlmostEqual(result[1], expected-core, 9)
        self.assertAlmostEqual(mc.e_tot, result[0], 12)
        self.assertAlmostEqual(mc.e_cas, result[1], 12)
        numpy.testing.assert_allclose(report["root_objective"],
                                      numpy.array(physical)+report["root_penalty"],
                                      atol=1e-9, rtol=0)
        if len(roots) > 1:
            numpy.testing.assert_allclose(mc.e_states, physical, atol=1e-9, rtol=0)
            numpy.testing.assert_allclose(mc.fcisolver.e_states, report["root_objective"],
                                          atol=1e-9, rtol=0)
            self.assertAlmostEqual(mc.e_average, expected, 9)
        self.assertFalse(hasattr(mc, "e_tot_physical"))
        self.assertFalse(hasattr(mc, "e_gas_physical"))

    def test_spin_physical_energy_boundaries(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 .9; H 0 0 2.2; H 0 0 3.1",
                    basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        for use_df in (False, True):
            for weights in (None, (.4, .6), (1., 0.)):
                with self.subTest(df=use_df, weights=weights):
                    mc = gasscf.GASSCF(
                        mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                        gas_restr=((0, 1), (2, 2)), gas_restr_type="cumulative-occ")
                    if use_df:
                        mc = mc.density_fit()
                    if weights is not None:
                        mc = mc.state_average(weights)
                    self.addCleanup(mc.close)
                    # Intentionally nonzero penalty; one macro also exercises
                    # the public energy convention for an unconverged result.
                    mc.fix_spin_(shift=.001, ss=0.)
                    mc.max_cycle_macro = mc.max_cycle_micro = 1
                    mc.canonicalization = use_df
                    mc.chk_ci = True
                    result = mc.gasci(mf.mo_coeff)
                    self._check_physical_energy_result(mc, result)
                    seen = []
                    def callback(env):
                        report = mc.spin_energy_report()
                        self.assertAlmostEqual(env["e_tot"], report["objective"], 10)
                        self.assertAlmostEqual(mc.e_tot, report["physical"], 10)
                        seen.append(env["e_tot"])
                    for repeat in range(2):
                        result = mc.kernel(mc.mo_coeff, callback=callback)
                        self._check_physical_energy_result(mc, result)
                    self.assertTrue(seen)
                    saved = lib.chkfile.load(mc.chkfile, "mcscf")
                    self.assertAlmostEqual(saved["e_tot"], mc.e_tot, 10)
                    self.assertAlmostEqual(saved["e_cas"], mc.e_cas, 10)
                    # Restore into a fresh object, using PySCF's native loader.
                    # The standard file stores physical energies, orbitals and
                    # (with chk_ci=True) CI, not our private spin-energy report.
                    self.assertNotIn("gas_energy_convention", saved)
                    self.assertNotIn("_gas_energy_results", saved)
                    restored = gasscf.GASSCF(
                        mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                        gas_restr=((0, 1), (2, 2)), gas_restr_type="cumulative-occ")
                    if use_df:
                        restored = restored.density_fit()
                    if weights is not None:
                        restored = restored.state_average(weights)
                    restored.fix_spin_(shift=.001, ss=0.)
                    restored.canonicalization = use_df
                    restored.max_cycle_macro = restored.max_cycle_micro = 1
                    self.addCleanup(restored.close)
                    restored.update_from_chk(mc.chkfile)
                    self.assertAlmostEqual(restored.e_tot, result[0], 10)
                    self.assertAlmostEqual(restored.e_cas, result[1], 10)
                    numpy.testing.assert_array_equal(restored.mo_coeff, mc.mo_coeff)
                    numpy.testing.assert_array_equal(restored.ci, mc.ci)
                    with self.assertRaises(ValueError):
                        restored.spin_energy_report()
                    self._check_physical_energy_result(
                        restored, restored.kernel(restored.mo_coeff, restored.ci))
                    scanner = mc.as_scanner()
                    self.addCleanup(scanner.close)
                    value = scanner("H 0 0 0; H 0 0 .92; H 0 0 2.2; H 0 0 3.1")
                    self._check_physical_energy_result(scanner, (value, scanner.e_gas))
                    mc.undo_fix_spin_()
                    with self.assertRaises(ValueError):
                        mc.spin_energy_report()
                    mc.gasci(mc.mo_coeff)
                    self.assertIsNone(mc.e_spin_penalty)
                    if weights is not None:
                        numpy.testing.assert_allclose(mc.e_states, mc.fcisolver.e_states,
                                                      atol=1e-9, rtol=0)
                    mc.reset(mol)
                    with self.assertRaises(ValueError):
                        mc.spin_energy_report()

    def test_spin_physical_energy_full_active_shortcut_and_native_checkpoint(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 .75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(mf, 2, (1, 1), gas_orbs=(2,), ncore=0)
        self.addCleanup(mc.close)
        mc.canonicalization = False
        # Ground singlet pays 4*shift for a triplet target at insufficient shift.
        mc.fix_spin_(shift=.001, ss=2.)
        result = mc.kernel(mf.mo_coeff)
        self._check_physical_energy_result(mc, result)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "energy.chk")
            mc.dump_chk(filename)
            mc.update_from_chk(filename)
            self.assertEqual(mc.e_tot, result[0])
            self.assertEqual(mc.e_gas, result[1])
            with self.assertRaisesRegex(ValueError, 'no completed'):
                mc.spin_energy_report()
            self._check_physical_energy_result(mc, mc.gasci())
            # Native update() accepts ordinary PySCF fields without a marker.
            lib.chkfile.dump(filename, "mcscf", {
                "mo_coeff": mc.mo_coeff, "ci": mc.ci,
                "e_tot": result[0], "e_cas": result[1],
                "nelecas": numpy.asarray(mc.nelecas)})
            restored = gasscf.GASSCF(mf, 2, (1, 1), gas_orbs=(2,), ncore=0)
            self.addCleanup(restored.close)
            restored.canonicalization = False
            restored.fix_spin_(shift=.001, ss=2.)
            restored.update(filename)
            self.assertEqual(restored.e_tot, result[0])
            self.assertEqual(restored.e_cas, result[1])
            numpy.testing.assert_array_equal(restored.mo_coeff, mc.mo_coeff)
            numpy.testing.assert_array_equal(restored.ci, mc.ci)
            self._check_physical_energy_result(
                restored, restored.kernel(restored.mo_coeff, restored.ci))

    def test_fix_spin_kernel_smoke_tracks_physical_energy(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=1)
        mc.max_cycle_macro = 1
        mc.max_cycle_micro = 1
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-4
        mc.canonicalization = False
        mc.fix_spin(shift=.2, ss=0)

        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.kernel(mf.mo_coeff)
        report = mc.spin_energy_report()

        self.assertTrue(numpy.isfinite(e_tot))
        self.assertTrue(numpy.isfinite(e_gas))
        self.assertEqual(numpy.asarray(ci).ndim, 1)
        self.assertEqual(mo_coeff.shape, mf.mo_coeff.shape)
        self.assertIsNone(mo_energy)
        self.assertTrue(numpy.isfinite(report["physical"]))
        self.assertTrue(numpy.isfinite(report["penalty"]))
        self.assertAlmostEqual(report["physical"], e_tot, places=9)
        self.assertEqual(report["target_s2"], 0.0)
        self.assertIn(report["method"], (
            "exact-small-space", "projected-plus-global-davidson"))


class TestProperties(_N2Fixture, unittest.TestCase):
    """Densities, transition properties and analysis orbitals."""

    def test_effective_nelecas_honors_solver_spin(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 2, 2, gas_orbs=(2,), gas_restr=None, ncore=0)

        self.assertEqual(mc._effective_nelecas(), (1, 1))
        mc.fcisolver.spin = 2
        self.assertEqual(mc._effective_nelecas(), (2, 0))

    def test_density_transition_and_spin_wrapper_dispatch(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        self.addCleanup(mf._chkfile.close)
        mc = gasscf.GASSCF(mf, 2, (1, 1), gas_orbs=(2,), ncore=0)
        self.addCleanup(mc.close)
        solver = mc.fcisolver

        # Deliberately unnormalized: wrappers must forward the supplied vectors.
        ci = numpy.random.default_rng(71).normal(size=4)
        mc.ci = ci
        d1, d2 = solver.make_rdm12(ci, 2, (1, 1))
        d1s, d2s = solver.make_rdm12s(ci, 2, (1, 1))
        for name, expected in (('make_gasdm1', d1), ('make_gasdm1s', d1s),
                               ('make_gasdm12', (d1, d2)), ('make_gasdm12s', (d1s, d2s)),
                               ('make_gasdm2', d2)):
            with self.subTest(method=name):
                self._assert_property_close(getattr(mc, name)(), expected, atol=1e-12)
        rng = numpy.random.default_rng(72)
        bra, ket = rng.normal(size=(2, 4))
        mc.ci = ket
        d1, d2 = solver.trans_rdm12(bra, ket, 2, (1, 1))
        d1s, d2s = solver.trans_rdm12s(bra, ket, 2, (1, 1))
        for name, expected in (
                ('trans_gasdm1', solver.trans_rdm1(bra, ket, 2, (1, 1))),
                ('trans_gasdm1s', solver.trans_rdm1s(bra, ket, 2, (1, 1))),
                ('trans_gasdm12', (d1, d2)), ('trans_gasdm12s', (d1s, d2s)),
                ('trans_gasdm2', d2)):
            with self.subTest(method=name):
                self._assert_property_close(getattr(mc, name)(bra, ket), expected, atol=1e-12)
        self._assert_property_close(mc.spin_square(ket), solver.spin_square(ket, 2, (1, 1)),
                                    atol=1e-12)

    def test_gasscf_property_wrappers_require_ci_and_select_roots(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)

        with self.assertRaisesRegex(ValueError, "CI vector is not available"):
            mc.make_gasdm1()
        roots = [numpy.array([1., 0., 0., 0.]),
                 numpy.array([0., 0., 0., 1.])]
        mc.ci = roots
        for state in (0, 1):
            numpy.testing.assert_allclose(
                mc.make_gasdm1(state=state),
                mc.fcisolver.make_rdm1(roots[state], 2, (1, 1)), atol=1e-12, rtol=0)
        numpy.testing.assert_allclose(mc.make_gasdm1(), mc.make_gasdm1(state=0),
                                      atol=1e-12, rtol=0)
        sa = mc.state_average((.25, .75))
        with self.assertRaisesRegex(ValueError, "one CI vector per state"):
            sa.make_gasdm1(ci=roots[0])
        with self.assertRaisesRegex(ValueError, "root count"):
            sa.make_gasdm1(ci=roots[:1])

    def test_n2_property_densities_match_gasci_for_ss_sa_and_zero_weight(self):
        mc, ref, roots = self._n2_property_pair()
        names = ("make_gasdm1", "make_gasdm1s", "make_gasdm12",
                 "make_gasdm12s", "make_gasdm2", "make_rdm1", "make_rdm1s")
        overlap = mc._scf.get_ovlp()
        for weights in (None, (.25, .75), (1., 0.)):
            obj = mc if weights is None else mc.state_average(weights)
            reference = ref if weights is None else ref.state_average(weights)
            self.addCleanup(obj.close)
            mo_before = obj.mo_coeff.copy()
            for cache in (True, False):
                obj.close()
                obj.cache_plans = cache
                for name in names:
                    for state in (None, 0, 1):
                        with self.subTest(weights=weights, cache=cache,
                                          method=name, state=state):
                            actual = getattr(obj, name)(state=state)
                            expected = getattr(reference, name)(state=state)
                            self._assert_property_close(actual, expected)
                # Independently verify weighted AO density and electron count.
                dm0 = ref.make_rdm1(state=0)
                dm1 = ref.make_rdm1(state=1)
                expected = dm0 if weights is None else (
                    weights[0] * dm0 + weights[1] * dm1)
                numpy.testing.assert_allclose(obj.make_rdm1(), expected,
                                              atol=2e-11, rtol=0)
                self.assertAlmostEqual(
                    numpy.einsum("ij,ji->", obj.make_rdm1(), overlap),
                    obj.mol.nelectron, places=10)
                self.assertAlmostEqual(numpy.trace(obj.make_gasdm1()),
                                       sum(obj.nelecas), places=10)
                self._assert_property_close(
                    obj.make_gasdm1(None, obj.ncas, obj.nelecas, 1),
                    ref.make_gasdm1(state=1))
                numpy.testing.assert_array_equal(obj.mo_coeff, mo_before)
                for actual, original in zip(obj.ci, roots):
                    numpy.testing.assert_array_equal(actual, original)

    def test_n2_transition_properties_select_bra_and_ket(self):
        mc, ref, roots = self._n2_property_pair()
        sa = mc.state_average((.25, .75))
        self.addCleanup(sa.close)
        names = ("trans_gasdm1", "trans_gasdm1s", "trans_gasdm12",
                 "trans_gasdm12s", "trans_gasdm2")
        for cache in (True, False):
            sa.close()
            sa.cache_plans = cache
            for name in names:
                for bra, ket in ((0, 1), (1, 0), (1, 1)):
                    with self.subTest(cache=cache, method=name, bra=bra, ket=ket):
                        expected = getattr(ref, name)(roots[bra], roots[ket])
                        self._assert_property_close(
                            getattr(sa, name)(bra_state=bra, ket_state=ket),
                            expected)
                        self._assert_property_close(
                            getattr(sa, name)(roots[bra], roots[ket]), expected)
                self._assert_property_close(getattr(sa, name)(),
                                            getattr(ref, name)(roots[0], roots[1]))
            # Reversing a real bra/ket transposes the transition 1-RDM.
            numpy.testing.assert_allclose(
                sa.trans_gasdm1(bra_state=1, ket_state=0),
                sa.trans_gasdm1().T, atol=2e-11, rtol=0)
            numpy.testing.assert_allclose(
                sa.trans_gasdm1(bra_state=1, ket_state=1),
                sa.make_gasdm1(state=1), atol=2e-11, rtol=0)

    def test_n2_spin_properties_match_gasci_and_reuse_newton_plan(self):
        mc, ref, _ = self._n2_property_pair()
        root_spins = [ref.spin_square(state=i)[0] for i in (0, 1)]
        for weights in (None, (.25, .75), (1., 0.)):
            obj = mc if weights is None else mc.state_average(weights)
            self.addCleanup(obj.close)
            obj.close()
            obj.cache_plans = True
            with mock.patch.object(obj.fcisolver, "make_rdm_plan",
                                   wraps=obj.fcisolver.make_rdm_plan) as create:
                for state in (0, 1):
                    self._assert_property_close(obj.spin_square(state=state),
                                                ref.spin_square(state=state))
                ss, multiplicity = obj.spin_square()
                expected = root_spins[0] if weights is None else numpy.dot(
                    weights, root_spins)
                self.assertAlmostEqual(ss, expected, places=11)
                self.assertAlmostEqual(multiplicity, numpy.sqrt(4*ss+1), places=11)
                self.assertEqual(create.call_count, 1)
            obj.close()
            obj.cache_plans = False
            self.assertAlmostEqual(obj.spin_square()[0], expected, places=11)

    def test_n2_sort_mo_preserves_requested_gas_order(self):
        mc, ref, _ = self._n2_property_pair()
        before = mc.mo_coeff.copy()
        gaslst = [[4, 2], [7, 5, 6, 3], [9, 8]]
        order = [0, 1, 4, 2, 7, 5, 6, 3, 9, 8] + list(range(10, before.shape[1]))
        actual = mc.sort_mo(gaslst, base=0)
        numpy.testing.assert_array_equal(actual, before[:, order])
        numpy.testing.assert_array_equal(actual, ref.sort_mo(gaslst, base=0))
        one_based = [[i + 1 for i in block] for block in gaslst]
        numpy.testing.assert_array_equal(mc.sort_mo(one_based), actual)
        numpy.testing.assert_array_equal(mc.mo_coeff, before)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            mc.sort_mo([[2, 2], [4, 5, 6, 7], [8, 9]], base=0)
        with self.assertRaisesRegex(ValueError, "subspace lists"):
            mc.sort_mo(list(range(2, 10)), base=0)

    def test_n2_get_fock_supports_gas_density_and_native_calls(self):
        mc, _, _ = self._n2_property_pair()
        for use_df in (False, True):
            obj = mc.density_fit() if use_df else mc.copy()
            obj = obj.state_average((.25, .75))
            self.addCleanup(obj.close)
            dm = obj.make_gasdm1()
            dm_ao = obj.make_rdm1()
            vj, vk = obj._scf.get_jk(obj.mol, dm_ao)
            expected = obj.get_hcore() + vj - .5 * vk
            numpy.testing.assert_allclose(obj.get_fock(), expected,
                                          atol=2e-10, rtol=0)
            eris = obj.ao2mo(obj.mo_coeff)
            for density in (dm, obj.make_gasdm1(state=1)):
                with self.subTest(df=use_df), mock.patch.object(
                        obj._scf, "get_jk", side_effect=AssertionError("reuse ERIS")):
                    actual = obj.get_fock(eris=eris, gasdm1=density)
                    numpy.testing.assert_allclose(
                        obj.get_fock(eris=eris, casdm1=density), actual,
                        atol=2e-10, rtol=0)
                    numpy.testing.assert_allclose(
                        obj.get_fock(obj.mo_coeff, obj.ci, eris, density, 0),
                        actual, atol=2e-10, rtol=0)
            with self.assertRaisesRegex(ValueError, "only one"):
                obj.get_fock(gasdm1=dm, casdm1=dm)

    def test_n2_natural_orbitals_reconstruct_selected_and_average_density(self):
        mc, ref, roots = self._n2_property_pair()
        overlap = mc._scf.get_ovlp()
        ncore, nocc = mc.ncore, mc.ncore + mc.ncas
        active = slice(ncore, nocc)
        for weights in (None, (.25, .75), (1., 0.)):
            obj = mc.copy() if weights is None else mc.state_average(weights)
            reference = ref if weights is None else ref.state_average(weights)
            self.addCleanup(obj.close)
            before = obj.mo_coeff.copy()
            for state in (0, 1, None):
                if state is None:
                    if weights is None:
                        with self.assertRaisesRegex(ValueError, "state-average"):
                            obj.get_gas_average_natorb()
                        continue
                    mo, occ = obj.get_gas_average_natorb()
                    _, ref_occ = reference.get_gas_average_natorb()
                else:
                    mo, occ = obj.get_gas_natorb(state=state)
                    _, ref_occ = reference.get_gas_natorb(state=state)
                with self.subTest(weights=weights, state=state):
                    self.assertEqual(mo.shape, before.shape)
                    self.assertEqual(occ.shape, (obj.ncas,))
                    numpy.testing.assert_allclose(occ, ref_occ, atol=2e-12, rtol=0)
                    numpy.testing.assert_allclose(mo.T @ overlap @ mo,
                                                  numpy.eye(mo.shape[1]), atol=2e-9, rtol=0)
                    numpy.testing.assert_array_equal(mo[:, :ncore], before[:, :ncore])
                    numpy.testing.assert_array_equal(mo[:, nocc:], before[:, nocc:])
                    dm = 2 * mo[:, :ncore] @ mo[:, :ncore].T
                    dm += (mo[:, active] * occ) @ mo[:, active].T
                    numpy.testing.assert_allclose(dm, obj.make_rdm1(state=state),
                                                  atol=2e-11, rtol=0)
            # No orbital/CI replacement is allowed for analysis-only methods.
            numpy.testing.assert_array_equal(obj.mo_coeff, before)
            for original, actual in zip(roots, obj.ci):
                numpy.testing.assert_array_equal(actual, original)
            with self.assertRaisesRegex(ValueError, "explicit state"):
                obj.get_gas_natorb(state=None)

    def test_n2_pseudo_natural_orbitals_preserve_gas_subspaces(self):
        mc, ref, roots = self._n2_property_pair()
        obj = mc.state_average((.25, .75))
        self.addCleanup(obj.close)
        overlap = obj._scf.get_ovlp()
        before = obj.mo_coeff.copy()
        active = slice(obj.ncore, obj.ncore + obj.ncas)
        for state in (None, 0, 1):
            mo, occupations = obj.get_gas_pseudo_natorb(state=state)
            self.assertIsInstance(occupations, tuple)
            self.assertEqual(tuple(len(o) for o in occupations), obj.gas_orbs)
            self._assert_property_close(
                occupations, obj.get_gas_pseudo_natorb_occupations(state=state))
            dm = obj.make_gasdm1(state=state)
            rotation = before[:, active].T @ overlap @ mo[:, active]
            allowed = numpy.zeros_like(rotation, dtype=bool)
            offset = 0
            for size, occ in zip(obj.gas_orbs, occupations):
                block = slice(offset, offset + size)
                allowed[block, block] = True
                u = rotation[block, block]
                numpy.testing.assert_allclose(
                    u.T @ dm[block, block] @ u, numpy.diag(occ), atol=2e-9, rtol=0)
                offset += size
            numpy.testing.assert_allclose(rotation[~allowed], 0., atol=2e-9, rtol=0)
            numpy.testing.assert_allclose(mo.T @ overlap @ mo,
                                          numpy.eye(mo.shape[1]), atol=2e-9, rtol=0)
            numpy.testing.assert_array_equal(obj.mo_coeff, before)
            for original, actual in zip(roots, obj.ci):
                numpy.testing.assert_array_equal(actual, original)
        _, ref_occ = ref.get_gas_pseudo_natorb(state=1)
        self._assert_property_close(occupations, ref_occ)

    def test_analysis_orbitals_preserve_eigenvector_dtype(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6; '
                        'H 0 0 3.7; H 0 0 4.5', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        overlap = mf.get_ovlp()
        for driver in (gasci.GASCI, gasscf.GASSCF):
            for sa in (False, True):
                mc = driver(mf, 4, (2, 2), ncore=1, gas_orbs=(2, 2),
                            gas_restr=((2, 2), (4, 4)),
                            gas_restr_type='cumulative-occ')
                if sa:
                    mc = mc.state_average((.3, .7))
                if hasattr(mc, 'close'):
                    self.addCleanup(mc.close)
                before = mc.mo_coeff
                active = before[:, 1:5]
                methods = ['get_gas_natorb', 'get_gas_pseudo_natorb']
                if sa:
                    methods.append('get_gas_average_natorb')
                for dtype in (numpy.int64, numpy.float64, numpy.complex128):
                    block = numpy.array([[1, 1], [1, 1]], dtype=dtype)
                    if numpy.iscomplexobj(block):
                        block[0, 1], block[1, 0] = 1j, -1j
                    density = scipy.linalg.block_diag(block, block)
                    expected = active @ density @ active.conj().T
                    for method in methods:
                        with self.subTest(driver=driver.__name__, sa=sa,
                                          dtype=dtype, method=method):
                            mo, occ = getattr(mc, method)(gasdm1=density)
                            if isinstance(occ, tuple):
                                occ = numpy.concatenate(occ)
                            numpy.testing.assert_allclose(
                                mo.conj().T @ overlap @ mo, numpy.eye(6), atol=2e-12, rtol=0)
                            reconstructed = (mo[:, 1:5] * occ) @ mo[:, 1:5].conj().T
                            numpy.testing.assert_allclose(reconstructed, expected, atol=2e-12, rtol=0)
                            numpy.testing.assert_array_equal(mo[:, [0, 5]], before[:, [0, 5]])
                            self.assertIs(mc.mo_coeff, before)
                            self.assertIsNone(mc.ci)
                            if method == 'get_gas_pseudo_natorb':
                                rotation = active.conj().T @ overlap @ mo[:, 1:5]
                                numpy.testing.assert_allclose(rotation[:2, 2:], 0., atol=2e-12, rtol=0)
                                numpy.testing.assert_allclose(rotation[2:, :2], 0., atol=2e-12, rtol=0)
                    # Complex analysis orbitals are not computational GAS MOs.
                    if numpy.iscomplexobj(density):
                        with self.assertRaisesRegex(TypeError, 'real-valued orbitals'):
                            mc.kernel(mo.astype(complex))

    def test_n2_analysis_orbitals_can_be_exported_to_molden(self):
        mc, _, roots = self._n2_property_pair()
        obj = mc.state_average((.25, .75))
        self.addCleanup(obj.close)
        before = obj.mo_coeff.copy()
        orbitals = (
            obj.get_gas_natorb(state=1), obj.get_gas_average_natorb(),
            obj.get_gas_pseudo_natorb())
        with tempfile.TemporaryDirectory() as directory:
            for index, (mo, active_occ) in enumerate(orbitals):
                if isinstance(active_occ, tuple):
                    active_occ = numpy.concatenate(active_occ)
                occ = numpy.zeros(mo.shape[1])
                occ[:obj.ncore] = 2.
                occ[obj.ncore:obj.ncore + obj.ncas] = active_occ
                path = str(Path(directory) / ("gas_%d.molden" % index))
                molden.from_mo(obj.mol, path, mo, occ=occ,
                               ene=numpy.zeros(mo.shape[1]), ignore_h=False)
                loaded_mol, energies, loaded_mo, loaded_occ, _, _ = molden.load(path)
                self.assertEqual(loaded_mol.nao_nr(), obj.mol.nao_nr())
                numpy.testing.assert_allclose(loaded_mo, mo, atol=1e-12, rtol=0)
                # Molden writes occupation numbers with five decimal places.
                numpy.testing.assert_allclose(loaded_occ, occ, atol=5.1e-6, rtol=0)
                numpy.testing.assert_array_equal(energies, numpy.zeros(mo.shape[1]))
        numpy.testing.assert_array_equal(obj.mo_coeff, before)
        for original, actual in zip(roots, obj.ci):
            numpy.testing.assert_array_equal(actual, original)

    def test_n2_analyze_reports_gas_properties_and_returns_ao_density(self):
        mc, ref, roots = self._n2_property_pair()
        # Supply consistent fixed-orbital energies for the analysis report.
        h1, ecore = mc.get_h1gas()
        h2 = mc.get_h2gas()
        energies = [ref.fcisolver.energy(h1, h2, ci, mc.ncas, mc.nelecas)
                    + ecore for ci in roots]
        for weights in (None, (.25, .75), (1., 0.)):
            obj = mc.copy() if weights is None else mc.state_average(weights)
            self.addCleanup(obj.close)
            if weights is None:
                obj.ci = roots[0].copy()
                obj.e_tot = energies[0]
                states = (None,)
            else:
                obj.fcisolver.e_states = numpy.array(energies)
                obj.e_tot = numpy.dot(weights, energies)
                states = (None, 1)
            gasci._publish_energy_results(obj, obj.e_tot, obj.e_tot - ecore)
            before_mo = obj.mo_coeff.copy()
            before_ci = numpy.array(obj.ci, copy=True)
            before_energy = obj.e_tot
            for state in states:
                obj.stdout = io.StringIO()
                # Full population-analysis output, including the GAS labels.
                actual = obj.analyze(verbose=4, state=state, with_meta_lowdin=False)
                self._assert_property_close(actual, obj.make_rdm1s(state=state))
                log = obj.stdout.getvalue()
                self.assertIn("GASSCF analysis", log)
                self.assertIn("GASSCF state", log)
                self.assertIn("GAS subspace 3", log)
                self.assertIn("pseudo-natural occupations", log)
                self.assertIn("Largest GAS CI components", log)
            numpy.testing.assert_array_equal(obj.mo_coeff, before_mo)
            numpy.testing.assert_array_equal(numpy.asarray(obj.ci), before_ci)
            self.assertEqual(obj.e_tot, before_energy)
        ref.stdout = io.StringIO()
        ref.e_tot = numpy.array(energies)
        ref.analyze(verbose=4, state=0, with_meta_lowdin=False)
        self.assertIn("GASCI analysis", ref.stdout.getvalue())


class TestCanonicalization(_N2Fixture, unittest.TestCase):
    """Orbital/CI representation changes and invariant physical results."""

    def test_n2_numerical_regression_canonicalization(self):
        _, mf, mo = self._n2_regression_fixture()
        mc = self._n2_regression_mc(mf)

        e_before, _, ci_before, mo_before, _ = mc.gasci(mo)
        mo_after, ci_after, _ = mc.canonicalize(
            mo_before, ci_before, sort=False)
        e_after = float(mc.casci(mo_after, ci_after)[0])

        self.assertAlmostEqual(
            float(e_before), self.N2_REF_FIXED_ENERGY,
            delta=self.N2_REGRESSION_TOL)
        self.assertAlmostEqual(
            e_after, self.N2_REF_FIXED_ENERGY,
            delta=self.N2_REGRESSION_TOL)
        self.assertAlmostEqual(
            float(e_before), e_after,
            delta=self.N2_REGRESSION_TOL)

    def test_n2_pseudo_canonicalize_preserves_roots_and_density(self):
        mc, _, roots = self._n2_property_pair()
        for use_df in (False, True):
            for weights in (None, (.3, .7), (1., 0.)):
                obj = mc.density_fit() if use_df else mc.copy()
                self.addCleanup(obj.close)
                if weights is None:
                    obj.ci = roots[0].copy()
                else:
                    obj = obj.state_average(weights)
                    self.addCleanup(obj.close)
                before_mo = obj.mo_coeff.copy()
                before_ci = numpy.array(obj.ci, copy=True)
                before_energy = obj.mo_energy
                eris = obj.ao2mo(before_mo)
                fock = obj.get_fock(eris=eris)
                # Canonicalization changes representation, without solving CI.
                with mock.patch.object(obj.fcisolver, 'kernel',
                                       side_effect=AssertionError('no CI solve')):
                    mo, ci, mo_energy = obj.canonicalize(
                        eris=eris, gas_pseudo_natorb=True)
                with self.subTest(df=use_df, weights=weights):
                    numpy.testing.assert_array_equal(obj.mo_coeff, before_mo)
                    numpy.testing.assert_array_equal(numpy.asarray(obj.ci), before_ci)
                    self.assertIs(obj.mo_energy, before_energy)
                    numpy.testing.assert_allclose(
                        mo.T @ obj._scf.get_ovlp() @ mo,
                        numpy.eye(mo.shape[1]), atol=2e-9, rtol=0)
                    numpy.testing.assert_allclose(
                        mo_energy, numpy.einsum('pi,pi->i', mo, fock @ mo),
                        atol=2e-11, rtol=0)
                    dm = obj.make_gasdm1(ci=ci)
                    offset = 0
                    for size in obj.gas_orbs:
                        block = dm[offset:offset+size, offset:offset+size]
                        numpy.testing.assert_allclose(
                            block, numpy.diag(block.diagonal()), atol=2e-11, rtol=0)
                        self.assertTrue(numpy.all(numpy.diff(block.diagonal()) <= 1e-11))
                        offset += size
                    h1_old, core_old = obj.get_h1gas(before_mo)
                    h2_old = ao2mo.restore(1, obj.get_h2gas(before_mo), obj.ncas)
                    h1_new, core_new = obj.get_h1gas(mo)
                    h2_new = ao2mo.restore(1, obj.get_h2gas(mo), obj.ncas)
                    states = (0,) if weights is None else (0, 1)
                    for state in states:
                        old_dm = obj.make_rdm1(mo_coeff=before_mo, state=state)
                        new_dm = obj.make_rdm1(mo_coeff=mo, ci=ci, state=state)
                        numpy.testing.assert_allclose(new_dm, old_dm, atol=2e-11, rtol=0)
                        d1, d2 = obj.make_gasdm12(state=state)
                        old_e = (core_old + numpy.einsum('pq,qp', h1_old, d1)
                                 + .5 * numpy.einsum('pqrs,pqrs', h2_old, d2))
                        d1, d2 = obj.make_gasdm12(ci=ci, state=state)
                        new_e = (core_new + numpy.einsum('pq,qp', h1_new, d1)
                                 + .5 * numpy.einsum('pqrs,pqrs', h2_new, d2))
                        self.assertAlmostEqual(new_e, old_e, delta=2e-10)
                    old_roots = before_ci.reshape(len(states), -1)
                    new_roots = numpy.asarray(ci).reshape(len(states), -1)
                    numpy.testing.assert_allclose(new_roots @ new_roots.T,
                                                  old_roots @ old_roots.T,
                                                  atol=2e-12, rtol=0)
                    self.assertIsNone(obj.fcisolver.transform_ci_for_orbital_rotation)

    def test_n2_pseudo_canonicalize_writeback_and_selected_density(self):
        mc, _, roots = self._n2_property_pair()
        obj = mc.state_average((1., 0.))
        self.addCleanup(obj.close)
        density = obj.make_gasdm1(state=1)
        expected = obj.canonicalize(gas_pseudo_natorb=True, gasdm1=density)
        e_before = obj.e_tot
        actual = obj.canonicalize_(gas_pseudo_natorb=True, gasdm1=density)
        for returned, stored in zip(actual, (obj.mo_coeff, obj.ci, obj.mo_energy)):
            self.assertIs(returned, stored)
        # Near-degenerate external Fock eigenvectors can differ between calls;
        # compare their subspaces rather than individual virtual MO columns.
        active = slice(obj.ncore, obj.ncore + obj.ncas)
        numpy.testing.assert_allclose(actual[0][:, active], expected[0][:, active],
                                      atol=2e-11, rtol=0)
        self._assert_property_close(actual[1], expected[1])
        numpy.testing.assert_allclose(actual[2], expected[2], atol=1e-9, rtol=0)
        # Compare projectors in an orthonormal AO basis: S = R.T @ R.
        # Raw C @ C.T can be large in an ill-conditioned AO basis and
        # amplify roundoff under an otherwise invariant orbital rotation.
        metric = scipy.linalg.cholesky(obj._scf.get_ovlp(), lower=False)
        for block in (slice(0, obj.ncore), slice(obj.ncore + obj.ncas, None)):
            old = metric @ expected[0][:, block]
            new = metric @ actual[0][:, block]
            numpy.testing.assert_allclose(old @ old.T, new @ new.T,
                                          atol=2e-11, rtol=0)
        self.assertEqual(obj.e_tot, e_before)
        dm = obj.make_gasdm1(state=1)
        offset = 0
        for size in obj.gas_orbs:
            block = dm[offset:offset+size, offset:offset+size]
            numpy.testing.assert_allclose(block, numpy.diag(block.diagonal()),
                                          atol=2e-11, rtol=0)
            offset += size
        # The zero-weight root has also been transformed, not silently retained.
        self.assertGreater(numpy.linalg.norm(obj.ci[1] - roots[1]), 1e-3)

    def test_n2_pseudo_canonicalize_respects_frozen_and_gas_blocks(self):
        mc, _, roots = self._n2_property_pair()
        mc.ci = roots[0]
        mc.frozen = [0, 2]
        before = mc.mo_coeff.copy()
        mo, ci, _ = mc.canonicalize(gas_pseudo_natorb=True)
        numpy.testing.assert_array_equal(mo[:, mc.frozen], before[:, mc.frozen])
        active = slice(mc.ncore, mc.ncore + mc.ncas)
        rotation = before[:, active].T @ mc._scf.get_ovlp() @ mo[:, active]
        gas_labels = numpy.repeat(numpy.arange(mc.ngas), mc.gas_orbs)
        numpy.testing.assert_allclose(rotation[gas_labels[:, None] != gas_labels],
                                      0., atol=2e-9, rtol=0)
        numpy.testing.assert_allclose(
            mc.make_rdm1(mo_coeff=mo, ci=ci), mc.make_rdm1(), atol=2e-11, rtol=0)
        dm = mc.make_gasdm1(ci=ci)
        offset = 0
        for size in mc.gas_orbs:
            group = [i for i in range(offset, offset + size)
                     if mc.ncore + i not in mc.frozen]
            block = dm[numpy.ix_(group, group)]
            numpy.testing.assert_allclose(block, numpy.diag(block.diagonal()),
                                          atol=2e-11, rtol=0)
            offset += size

    def test_n2_pseudo_canonicalize_with_spin_penalty_and_restart(self):
        _, mf, mo = self._n2_regression_fixture()
        mc = self._n2_regression_mc(mf).fix_spin(shift=.2, ss=0)
        mc.gasci(mo)
        self.assertTrue(mc.converged)
        e_before = mc.e_tot
        dm_before = mc.make_rdm1()
        spin_before = mc.spin_square()
        mo_new, ci_new, _ = mc.canonicalize_(gas_pseudo_natorb=True)
        numpy.testing.assert_allclose(mc.make_rdm1(), dm_before, atol=2e-11, rtol=0)
        numpy.testing.assert_allclose(mc.spin_square(), spin_before, atol=2e-10, rtol=0)
        h1, core = mc.get_h1gas()
        e_new = mc.fcisolver.energy(h1, mc.get_h2gas(), ci_new, mc.ncas, mc.nelecas) + core
        self.assertAlmostEqual(e_new, e_before, delta=2e-10)
        # The returned CI is directly reusable by the unchanged Newton driver.
        e_restart = mc.kernel(mo_new, ci_new)[0]
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(e_restart, self.N2_REF_SS_ENERGY,
                               delta=self.N2_REGRESSION_TOL)

    def test_canonicalize_accepts_native_density_alias(self):
        mol = gto.M(atom=';'.join('H 0 0 %s' % z for z in
                                 (0., .9, 1.9, 3., 4.2, 5.5)),
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for use_df in (False, True):
            for pseudo in (False, True):
                with self.subTest(df=use_df, pseudo=pseudo):
                    mc = gasscf.GASSCF(
                        mf, 4, (2, 2), gas_orbs=(2, 2),
                        gas_restr=((2, 2), (4, 4)), gas_restr_type='cumulative-occ')
                    self.addCleanup(mc.close)
                    if use_df:
                        mc = mc.density_fit()
                        self.addCleanup(mc.close)
                    mc = mc.state_average((1., 0.))
                    self.addCleanup(mc.close)
                    mc.gasci()
                    density = mc.make_gasdm1(state=1)
                    self.assertGreater(numpy.linalg.norm(
                        density - mc.make_gasdm1()), 1e-3)
                    options = dict(gas_pseudo_natorb=pseudo)
                    expected = mc.canonicalize(gasdm1=density, **options)
                    with mock.patch.object(mc, 'make_gasdm1',
                                           side_effect=AssertionError('density recomputed')):
                        actual = mc.canonicalize(casdm1=density, **options)
                        for left, right in zip(actual, expected):
                            numpy.testing.assert_allclose(left, right, atol=2e-11, rtol=0)
                        for method in (mc.canonicalize, mc.canonicalize_):
                            with self.assertRaisesRegex(ValueError, 'only one'):
                                method(gasdm1=density, casdm1=density, **options)
                        actual = mc.canonicalize_(casdm1=density, **options)
                    for left, right, stored in zip(
                            actual, expected, (mc.mo_coeff, mc.ci, mc.mo_energy)):
                        numpy.testing.assert_allclose(left, right, atol=2e-11, rtol=0)
                        self.assertIs(left, stored)

    def test_canonicalize_sort_preserves_input_orbitals(self):
        mol = gto.M(atom=';'.join('H 0 0 %s' % z for z in
                                 (0., .9, 1.9, 3., 4.2, 5.5)),
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1),
            gas_restr=((1, 1), (2, 2)), gas_restr_type='cumulative-occ')
        self.addCleanup(mc.close)
        mo = mf.mo_coeff.copy()
        before = mo.copy()
        mc.gasci(mo)
        # A deterministic Fock matrix forces permutations of columns
        # in both core and virtual space, without relying on energy ordering.
        sm = mf.get_ovlp() @ mo
        fock = (sm * numpy.array([2., 1., 3., 4., 6., 5.])) @ sm.T
        order = [1, 0, 2, 3, 5, 4]
        for pseudo in (False, True):
            with self.subTest(pseudo=pseudo), \
                    mock.patch.object(mc, 'get_fock', return_value=fock):
                new, _, eps = mc.canonicalize(sort=True, gas_pseudo_natorb=pseudo)
                numpy.testing.assert_allclose(new, mo[:, order], atol=2e-11, rtol=0)
                numpy.testing.assert_allclose(eps, numpy.arange(1., 7.), atol=2e-11, rtol=0)
                numpy.testing.assert_array_equal(mo, before)
                self.assertFalse(numpy.shares_memory(new, mo))

    def test_canonicalize_preserves_frozen_density_and_energy(self):
        mol = gto.M(atom=';'.join('H 0 0 %s' % z for z in
                                 (0., .8, 1.7, 2.7, 3.8, 5., 6.3, 7.7, 9.2, 10.8)),
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for use_df in (False, True):
            for weights in (None, (1., 0.)):
                mc = gasscf.GASSCF(
                    mf, 4, (2, 2), gas_orbs=(2, 2),
                    gas_restr=((2, 2), (4, 4)), gas_restr_type='cumulative-occ')
                self.addCleanup(mc.close)
                if use_df:
                    mc = mc.density_fit()
                    self.addCleanup(mc.close)
                if weights is not None:
                    mc = mc.state_average(weights)
                    self.addCleanup(mc.close)
                # Deliberately mix core/virtual orbitals so
                # canonicalization and sort=True are both nontrivial.
                kappa = numpy.zeros((10, 10))
                for start in (0, 7):
                    kappa[start+1, start] = .4
                    kappa[start+2, start] = .3
                mo = mf.mo_coeff @ scipy.linalg.expm(kappa - kappa.T)
                mc.gasci(mo)
                before = mo.copy()
                ci_before = numpy.array(mc.ci, copy=True)
                energy_before = mc.mo_energy
                fock = mc.get_fock()
                states = (0,) if weights is None else (0, 1)
                old_dm = [mc.make_rdm1(state=state) for state in states]
                def energies(orbitals, ci):
                    h1, ecore = mc.get_h1gas(orbitals)
                    h2 = ao2mo.restore(1, mc.get_h2gas(orbitals), mc.ncas)
                    result = []
                    for state in states:
                        dm1, dm2 = mc.make_gasdm12(ci=ci, state=state)
                        result.append(ecore + numpy.einsum('pq,qp', h1, dm1)
                                      + .5 * numpy.einsum('pqrs,pqrs', h2, dm2))
                    return result
                old_e = energies(mo, mc.ci)
                for frozen in (None, [0, 9]):
                    mc.frozen = frozen
                    for pseudo in (False, True):
                        for sort in (False, True):
                            with self.subTest(df=use_df, weights=weights,
                                              frozen=frozen, pseudo=pseudo, sort=sort):
                                new, ci, eps = mc.canonicalize(
                                    sort=sort, gas_pseudo_natorb=pseudo)
                                if frozen is not None:
                                    numpy.testing.assert_array_equal(new[:, frozen], before[:, frozen])
                                if sort:
                                    for indices in (numpy.arange(3), numpy.arange(7, 10)):
                                        indices = numpy.array([i for i in indices
                                                               if frozen is None or i not in frozen])
                                        self.assertTrue(numpy.all(
                                            numpy.diff(eps[indices]) >= -2e-9))
                                numpy.testing.assert_allclose(
                                    eps, numpy.einsum('pi,pi->i', new, fock @ new), atol=2e-11, rtol=0)
                                for state, density in zip(states, old_dm):
                                    numpy.testing.assert_allclose(
                                        mc.make_rdm1(mo_coeff=new, ci=ci, state=state),
                                        density, atol=2e-11, rtol=0)
                                numpy.testing.assert_allclose(energies(new, ci), old_e, atol=2e-10, rtol=0)
                                numpy.testing.assert_array_equal(mc.mo_coeff, before)
                                numpy.testing.assert_array_equal(numpy.asarray(mc.ci), ci_before)
                                self.assertIs(mc.mo_energy, energy_before)

    def test_canonicalize_keeps_restricted_gas_active_block_unchanged(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", ncore=1)
        e_tot, e_gas, ci, mo_coeff, _ = mc.gasci(mf.mo_coeff)

        mo1, ci1, mo_energy = mc.canonicalize(mo_coeff, ci, sort=False)

        self.assertTrue(numpy.isfinite(e_tot))
        self.assertTrue(numpy.isfinite(e_gas))
        self.assertEqual(mo1.shape, mo_coeff.shape)
        self.assertEqual(mo_energy.shape, (mo_coeff.shape[1],))
        self.assertTrue(numpy.all(numpy.isfinite(mo_energy)))
        self.assertEqual(numpy.asarray(ci1).shape, numpy.asarray(ci).shape)
        overlap = mf.get_ovlp()
        active = slice(mc.ncore, mc.ncore + mc.ncas)
        active_metric = reduce(numpy.dot, (
            mo_coeff[:, active].T, overlap, mo1[:, active]))
        numpy.testing.assert_allclose(
            active_metric, numpy.eye(mc.ncas), atol=1e-10, rtol=0)

    def test_natural_orbital_rotations_are_guarded(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)

        with self.assertRaisesRegex(NotImplementedError, "natural-orbital"):
            mc.canonicalize(mf.mo_coeff, gas_natorb=True)
        with self.assertRaisesRegex(NotImplementedError, "natural-orbital"):
            mc.cas_natorb()

        mc.natorb = True
        with self.assertRaisesRegex(NotImplementedError, "natural-orbital"):
            mc.validate_capabilities()

class TestLifecycle(unittest.TestCase):
    """Object ownership, copies, implicit CI validity and checkpoints."""

    def test_solver_and_mc_copy_detach_owned_plan_caches(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        self.addCleanup(mf._chkfile.close)
        for level in ('solver', 'mc'):
            with self.subTest(copy=level):
                mc = gasscf.GASSCF(mf, 2, (1, 1), gas_orbs=(2,), ncore=0)
                self.addCleanup(mc.close)
                solver = mc.fcisolver
                rdm = solver._get_rdm_plan(2, (1, 1))
                spin = solver._get_spin_plan(2, (1, 1))
                contract = solver._get_contract_plan(numpy.zeros((3, 3)), 2, (1, 1))
                source = solver if level == 'solver' else mc
                copied = source.copy()
                self.addCleanup(copied.close)
                self.assertIsNot(copied, source)
                self.assertEqual(copied.gas_orbs, source.gas_orbs)
                target = copied if level == 'solver' else copied.fcisolver
                self.assertIsNot(target, solver)
                for name in ('_topology_key', '_rdm_plan', '_spin_plan', '_contract_space'):
                    self.assertIsNone(getattr(target, name))
                self.assertEqual(len(target._contract_plans), 0)
                self.assertIsNot(target._contract_plans, solver._contract_plans)
                copied.close()
                self.assertIs(solver._rdm_plan, rdm)
                self.assertIs(solver._spin_plan, spin)
                self.assertIsNotNone(rdm._plan)
                self.assertIsNotNone(contract._plan)
                mc.close()
                self.assertIsNone(rdm._plan)
                self.assertIsNone(contract._plan)

    def test_changed_gas_space_discards_implicit_ci(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.; H 0 0 3.2',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for method in ('gasci', 'kernel', 'get_grad', 'scanner'):
            # Both fixed occupations have four determinants, with different
            # meanings; the unrestricted variant changes the vector length.
            for bounds, ndet in (((1, 1), 4), ((0, 2), 9)):
                with self.subTest(method=method, bounds=bounds):
                    mc = gasscf.GASSCF(
                        mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                        gas_restr=((0, 0), (2, 2)), gas_restr_type='cumulative-occ')
                    self.addCleanup(mc.close)
                    mc.canonicalization = False
                    mc.max_cycle_macro = 1
                    mc.gasci()
                    self.assertEqual(mc.ci.size, 4)
                    old_ci = mc.ci
                    mc.gas_restr = (bounds, (2, 2))
                    if method == 'scanner':
                        source = mc
                        old_signature = source._gas_ci_signature
                        mc = source.as_scanner()
                        self.addCleanup(mc.close)
                        self.assertIsNone(mc.ci)
                        self.assertIs(source.ci, old_ci)
                        self.assertEqual(source._gas_ci_signature, old_signature)
                    with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                        if method == 'scanner':
                            mc(mol)
                        else:
                            getattr(mc, method)()
                    self.assertIsNone(solve.call_args_list[0].kwargs['ci0'])
                    if method == 'scanner':
                        self.assertEqual(mc.scan_info['CI_source'], 'native initial guess')
                        self.assertIs(source.ci, old_ci)
                    self.assertIsNot(mc.ci, old_ci)
                    self.assertEqual(mc.ci.size, ndet)
                    ref = gasci.GASCI(
                        mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                        gas_restr=(bounds, (2, 2)), gas_restr_type='cumulative-occ')
                    self.assertAlmostEqual(mc.e_tot, ref.kernel(mc.mo_coeff)[0], places=9)

    def test_equivalent_gas_space_retains_ci_and_explicit_guess(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.; H 0 0 3.2',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for method in ('gasci', 'kernel'):
            mc = gasscf.GASSCF(
                mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                gas_restr=((0, 0), (2, 2)), gas_restr_type='cumulative-occ')
            self.addCleanup(mc.close)
            mc.canonicalization = False
            mc.max_cycle_macro = 1
            mc.gasci()
            previous_ci = mc.ci
            _, blocks = mc._normalized_restriction()
            mc.gas_restr, mc.gas_restr_type = blocks, 'spin-supergroup'
            # A same-model reset/copy and equivalent syntax preserve the basis.
            mc.reset(mol)
            mc = mc.copy()
            self.addCleanup(mc.close)
            with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                getattr(mc, method)()
            self.assertIs(solve.call_args_list[0].kwargs['ci0'], previous_ci)
            # Explicit guesses are still accepted for a changed model.
            mc.gas_restr, mc.gas_restr_type = ((1, 1), (2, 2)), 'cumulative-occ'
            explicit = numpy.ones(4) / 2
            with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                getattr(mc, method)(ci0=explicit)
            self.assertIs(solve.call_args_list[0].kwargs['ci0'], explicit)

    def test_state_average_root_count_invalidates_implicit_ci(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for method in ('gasci', 'kernel'):
            mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=0)
            self.addCleanup(mc.close)
            mc.gasci()
            mc = mc.state_average((.5, .5))
            self.addCleanup(mc.close)
            with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                getattr(mc, method)()
            self.assertIsNone(solve.call_args_list[0].kwargs['ci0'])
            self.assertEqual(len(mc.ci), 2)
            previous_ci = mc.ci
            mc = mc.state_average((1., 0.))
            self.addCleanup(mc.close)
            with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                getattr(mc, method)()
            self.assertIs(solve.call_args_list[0].kwargs['ci0'], previous_ci)
            mc = mc.undo_state_average()
            self.addCleanup(mc.close)
            with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                getattr(mc, method)()
            self.assertIsNone(solve.call_args_list[0].kwargs['ci0'])
            self.assertEqual(numpy.asarray(mc.ci).ndim, 1)

    def test_checkpoint_ci_requires_explicit_guess_without_gas_signature(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=0)
        self.addCleanup(mc.close)
        mc.gasci()
        with tempfile.TemporaryDirectory() as tmpdir:
            mc.chkfile = str(Path(tmpdir) / 'gas.chk')
            mc.chk_ci = True
            mc.dump_chk(dict(e_tot=mc.e_tot, e_cas=mc.e_cas, fcivec=mc.ci,
                             mo_coeff=mc.mo_coeff, casdm1=mc.make_gasdm1()))
            for entry in ('kernel', 'new-scanner', 'existing-scanner'):
                for explicit in (False, True):
                    with self.subTest(entry=entry, explicit=explicit):
                        target = mc if entry != 'existing-scanner' else mc.as_scanner()
                        self.addCleanup(target.close)
                        target.update_from_chk()
                        self.assertIsNone(target._gas_ci_signature)
                        loaded_ci = target.ci
                        self.assertIsNotNone(loaded_ci)
                        if entry == 'new-scanner':
                            target = mc.as_scanner()
                            self.addCleanup(target.close)
                            self.assertIs(mc.ci, loaded_ci)
                        with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                            if entry == 'kernel':
                                target.kernel(ci0=loaded_ci if explicit else None)
                            else:
                                target(mol, ci0=loaded_ci if explicit else None)
                        initial = solve.call_args_list[0].kwargs['ci0']
                        if explicit:
                            numpy.testing.assert_array_equal(initial, loaded_ci)
                            if entry != 'kernel':
                                self.assertFalse(numpy.shares_memory(initial, loaded_ci))
                        else:
                            self.assertIsNone(initial)
                        if entry != 'kernel':
                            self.assertEqual(target.scan_info['CI_source'],
                                             'explicit' if explicit else 'native initial guess')
                        self.assertIsNotNone(target._gas_ci_signature)

    def test_checkpoint_reload_invalidates_old_energy_reports(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .9; H 0 0 2.2; H 0 0 3.1',
                    basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        kappa = numpy.zeros((4, 4))
        kappa[1, 0], kappa[0, 1] = .18, -.18
        moved = mf.mo_coeff @ scipy.linalg.expm(kappa)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'new_results.chk')
            for use_df in (False, True):
                for weights in (None, (.4, .6), (1., 0.)):
                    for penalty in (False, True):
                        source = gasscf.GASSCF(
                            mf, 3, (1, 1), ncore=1, gas_orbs=(1, 2),
                            gas_restr=((0, 1), (2, 2)), gas_restr_type='cumulative-occ')
                        self.addCleanup(source.close)
                        if use_df:
                            source = source.density_fit()
                            self.addCleanup(source.close)
                        if weights is not None:
                            source = source.state_average(weights)
                            self.addCleanup(source.close)
                            self.assertIsNone(source.e_average)
                        if penalty:
                            source.fix_spin_(shift=.001, ss=0.)
                        source.gasci(moved)
                        source.chkfile, source.chk_ci = filename, True
                        # Use the native writer's iteration interface to save CI
                        # as well as the physical energies and orbitals.
                        source.dump_chk(dict(
                            e_tot=source.e_tot, e_cas=source.e_cas,
                            mo_coeff=source.mo_coeff, fcivec=source.ci,
                            casdm1=source.make_gasdm1()))
                        saved = lib.chkfile.load(filename, 'mcscf')
                        self.assertNotIn('_gas_energy_results', saved)
                        target = source.copy()
                        self.addCleanup(target.close)
                        for method in ('update_from_chk', 'update'):
                            with self.subTest(df=use_df, weights=weights,
                                              penalty=penalty, method=method):
                                target.gasci(mf.mo_coeff)
                                self.assertIsNotNone(target._gas_energy_results)
                                self.assertGreater(abs(target.e_tot - source.e_tot), 1e-4)
                                if penalty:
                                    target.spin_energy_report()
                                with mock.patch.object(target.fcisolver, 'kernel',
                                                       side_effect=AssertionError('CI solve')):
                                    result = (target.update_from_chk(filename)
                                              if method == 'update_from_chk' else target.update())
                                self.assertIs(result, target)
                                self.assertEqual(target.e_tot, saved['e_tot'])
                                self.assertEqual(target.e_gas, saved['e_cas'])
                                numpy.testing.assert_array_equal(target.mo_coeff, saved['mo_coeff'])
                                numpy.testing.assert_array_equal(target.ci, saved['ci'])
                                self.assertIsNone(target._gas_energy_results)
                                self.assertIsNone(target.e_spin_penalty)
                                self.assertIsNone(target.spin_penalty_method)
                                with self.assertRaisesRegex(ValueError, 'no completed'):
                                    target.spin_energy_report()
                                if weights is not None:
                                    self.assertEqual(target.e_states, [None] * len(weights))
                                    self.assertIsNone(target.e_average)
                                target.gasci()
                                self.assertAlmostEqual(target.e_tot, source.e_tot, delta=1e-9)
                                self.assertAlmostEqual(target.e_gas, source.e_gas, delta=1e-9)
                                if weights is not None:
                                    numpy.testing.assert_allclose(target.e_states, source.e_states,
                                                                  atol=1e-9, rtol=0)
                                    self.assertAlmostEqual(target.e_average, target.e_tot, delta=1e-9)
                                if penalty:
                                    report = target.spin_energy_report()
                                    self.assertAlmostEqual(report['physical'], target.e_tot, delta=1e-9)
                                    numpy.testing.assert_allclose(
                                        report['root_objective'], source.spin_energy_report()['root_objective'],
                                        atol=1e-9, rtol=0)

    def test_checkpoint_read_failure_preserves_completed_results(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        mc = gasscf.GASSCF(mf, 2, (1, 1)).state_average((.4, .6))
        self.addCleanup(mc.close)
        mc.fix_spin_(shift=.001, ss=0.)
        mc.gasci()
        snapshot = mc._gas_energy_results
        energies = (mc.e_tot, mc.e_gas, mc.e_average)
        mo, ci = mc.mo_coeff, mc.ci
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing.chk')
            for method in (mc.update_from_chk, mc.update):
                with self.assertRaises(OSError):
                    method(missing)
                self.assertIs(mc._gas_energy_results, snapshot)
                self.assertEqual((mc.e_tot, mc.e_gas, mc.e_average), energies)
                self.assertIs(mc.mo_coeff, mo)
                self.assertIs(mc.ci, ci)
                self.assertEqual(mc.spin_energy_report()['physical'], mc.e_tot)

    def test_df_wrappers_own_solver_and_preserve_source_plans(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        for sa in (False, True):
            with self.subTest(sa=sa):
                source = gasscf.GASSCF(scf.RHF(mol), 2, (1, 1))
                if sa:
                    source = source.state_average((.5, .5))
                self.addCleanup(source.close)
                plan = source.fcisolver._get_rdm_plan(2, (1, 1))
                fitted = source.density_fit(auxbasis='weigend')
                self.addCleanup(fitted.close)
                self.assertIsNot(fitted.fcisolver, source.fcisolver)
                self.assertIsNone(fitted.fcisolver._rdm_plan)
                fitted.fix_spin_(shift=.2, ss=0)
                self.assertFalse(hasattr(source.fcisolver, 'ss_penalty'))
                self.assertIsNotNone(plan._plan)

                fitted_plan = fitted.fcisolver._get_rdm_plan(2, (1, 1))
                # Reusing the same DF setup preserves the native identity rule.
                self.assertIs(fitted.density_fit(auxbasis='weigend'), fitted)
                plain = fitted.undo_df()
                self.addCleanup(plain.close)
                self.assertNotIsInstance(plain, mcdf._DFCAS)
                self.assertIsNot(plain.fcisolver, fitted.fcisolver)
                self.assertIsNone(plain.fcisolver._rdm_plan)
                plain.undo_fix_spin_()
                self.assertEqual(fitted.fcisolver.ss_penalty, .2)
                self.assertIsNotNone(fitted_plan._plan)
                self.assertEqual(plain.fcisolver.nroots, 2 if sa else 1)
                fitted.close()
                self.assertIsNotNone(plan._plan)

    def test_derived_objects_preserve_source_resources_geometry_and_energy(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.6',
                    basis='sto-3g', verbose=0)
        moved = mol.set_geom_(
            'H 0 0 0; H 0 0 1.05; H 0 0 1.8; H 0 0 2.6', inplace=False)
        for kind in ('plain', 'mc-df', 'shared-df', 'separate-df', 'auto-df'):
            for sa in (False, True):
                with self.subTest(kind=kind, sa=sa):
                    mf = scf.RHF(mol)
                    if kind not in ('plain', 'mc-df'):
                        mf = mf.density_fit(auxbasis='weigend')
                    mf.run()
                    self.addCleanup(mf._chkfile.close)
                    mc = gasscf.GASSCF(
                        mf, 2, (1, 1), ncore=1, gas_orbs=(1, 1),
                        gas_restr=((1, 1), (2, 2)),
                        gas_restr_type='cumulative-occ')
                    if kind not in ('plain', 'auto-df'):
                        auxbasis = 'def2-svp-jkfit' if kind == 'separate-df' else 'weigend'
                        mc = mc.density_fit(auxbasis=auxbasis)
                    if sa:
                        mc = mc.state_average((.5, .5))
                    self.addCleanup(mc.close)
                    before = mc.gasci()[0]
                    source_df = getattr(mc, 'with_df', None)
                    source_scf_df = getattr(mc._scf, 'with_df', None)
                    source_cderi = None if source_df is None else source_df._cderi
                    solver = mc.fcisolver
                    rdm = solver._get_rdm_plan(mc.ncas, mc.nelecas)
                    contract = solver._get_contract_plan(numpy.eye(3), mc.ncas, mc.nelecas)
                    spin = solver._get_spin_plan(mc.ncas, mc.nelecas)
                    factories = [mc.copy, mc.as_scanner]
                    if sa:
                        factories += [mc.undo_state_average,
                                      lambda: mc.state_average((.3, .7))]
                    for factory in factories:
                        copied = factory()
                        self.addCleanup(copied.close)
                        self.assertIsNot(copied.fcisolver, solver)
                        self.assertIsNone(copied.fcisolver._rdm_plan)
                        self.assertIsNone(copied.fcisolver._contract_space)
                        self.assertIsNone(copied.fcisolver._spin_plan)
                        self.assertIsNotNone(rdm._plan)
                        self.assertIsNotNone(contract._plan)
                        self.assertIs(solver._spin_plan, spin)
                        if sa:
                            self.assertEqual(mc.weights, (.5, .5))
                            self.assertEqual(mc.fcisolver.nroots, 2)
                        self.assertIsNot(copied._scf, mc._scf)
                        if source_df is not None:
                            self.assertIsNot(copied.with_df, source_df)
                        if source_scf_df is not None:
                            self.assertIsNot(copied._scf.with_df, source_scf_df)
                            if source_df is source_scf_df:
                                self.assertIs(copied.with_df, copied._scf.with_df)
                            elif source_df is not None:
                                self.assertIsNot(copied.with_df, copied._scf.with_df)
                        copied.reset(moved)
                        self.assertIs(mc.mol, mol)
                        self.assertIs(mc._scf.mol, mol)
                        if source_df is not None:
                            self.assertIs(source_df.mol, mol)
                            self.assertIs(source_df._cderi, source_cderi)
                            self.assertIs(copied.with_df.mol, moved)
                            copied.with_df.build()
                        if source_scf_df is not None:
                            self.assertIs(source_scf_df.mol, mol)
                            self.assertIs(copied._scf.with_df.mol, moved)
                        copied.close()
                        copied.close()
                        self.assertIsNotNone(rdm._plan)
                        self.assertIsNotNone(contract._plan)
                        self.assertAlmostEqual(mc.gasci()[0], before, places=11)

    def test_df_copies_do_not_overwrite_source_integral_files(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        moved = mol.set_geom_('H 0 0 0; H 0 0 1.05', inplace=False)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        for storage, sa in (('named', False), ('temporary', False),
                            ('named', True), ('temporary', True)):
            with self.subTest(storage=storage, sa=sa), ExitStack() as files:
                tmp = files.enter_context(tempfile.TemporaryDirectory())
                mc = gasscf.GASSCF(mf, 2, (1, 1)).density_fit(auxbasis='weigend')
                if sa:
                    mc = mc.state_average((.5, .5))
                self.addCleanup(mc.close)
                source_df = mc.with_df
                source_df.max_memory = 0
                if storage == 'named':
                    target = str(Path(tmp) / 'source.h5')
                else:
                    target = files.enter_context(tempfile.NamedTemporaryFile(dir=tmp))
                source_df._cderi_to_save = target
                before = mc.gasci()[0]
                source_path = Path(target if storage == 'named' else target.name)
                contents = source_path.read_bytes()
                factories = [mc.copy, mc.as_scanner]
                if sa:
                    factories += [mc.undo_state_average,
                                  lambda: mc.state_average((.3, .7))]
                for factory in factories:
                    copied = factory()
                    self.addCleanup(copied.close)
                    self.assertIsNone(copied.with_df._cderi)
                    self.assertIsNone(copied.with_df._cderi_to_save)
                    copied.reset(moved)
                    copied.with_df.build()
                    owned = copied.with_df._cderi_to_save
                    files.enter_context(owned)
                    self.assertNotEqual(Path(owned.name), source_path)
                    self.assertEqual(source_path.read_bytes(), contents)
                    self.assertIs(source_df._cderi_to_save, target)
                    self.assertAlmostEqual(mc.gasci()[0], before, places=11)


class TestScanner(_N2Fixture, unittest.TestCase):
    """Fixed-model energy scans, initial guesses and failure recovery."""

    def test_n2_numerical_regression_scanner(self):
        mol, mf, mo = self._n2_regression_fixture()
        source = self._n2_regression_mc(mf)
        source.kernel(mo)
        self.assertTrue(source.converged)

        scanner = source.as_scanner()
        self.addCleanup(scanner.close)
        coords = numpy.asarray(mol.atom_coords(unit="Angstrom"), dtype=float)
        bond = coords[1] - coords[0]
        coords[1] += 0.01 * bond / numpy.linalg.norm(bond)
        moved = mol.set_geom_(coords, unit="Angstrom", inplace=False)
        e_scan = float(scanner(moved))

        self.assertTrue(scanner.converged)
        self.assertAlmostEqual(
            e_scan, self.N2_REF_SCANNER_ENERGY,
            delta=self.N2_REGRESSION_TOL)

    def test_scanner_uses_gasci_mo_validation_and_thresholds(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=0, gas_orbs=(1, 1),
                          gas_restr=((1, 1), (2, 2)),
                          gas_restr_type='cumulative-occ')
        self.addCleanup(mc.close)
        scanner = mc.as_scanner()
        self.addCleanup(scanner.close)
        scanner.verbose = 4
        scanner.stdout = io.StringIO()
        # Exercise the real scanner through its optimizer boundary. The MO
        # validation must accept the warning interval and reject bad guesses
        # before the orbital/CI optimizer is called.
        with mock.patch.object(gasci, 'MO_ORTH_WARN_TOL', 1e-7), \
                mock.patch.object(gasci, 'MO_ORTH_ERROR_TOL', 1e-5), \
                mock.patch.object(scanner, 'kernel', return_value=(-1.,)) as solve:
            for error in (5e-8, 2e-6):
                mo = mf.mo_coeff.copy()
                mo[:, 0] *= numpy.sqrt(1 + error)
                scanner.stdout.seek(0)
                scanner.stdout.truncate(0)
                solve.reset_mock()
                self.assertEqual(scanner(mol, mo_coeff=mo), -1.)
                solve.assert_called_once()
                self.assertAlmostEqual(scanner.scan_info['initial_MO_metric_error'],
                                       error, delta=1e-14)
                self.assertEqual('WARN' in scanner.stdout.getvalue(), error > 1e-7)
                numpy.testing.assert_array_equal(solve.call_args.args[0], mo)
            mo = mf.mo_coeff.copy()
            mo[:, 0] *= numpy.sqrt(1 + 2e-5)
            bad = (mo, mf.mo_coeff[:, :1], mf.mo_coeff[:1],
                   numpy.full((2, 2), numpy.nan), numpy.full((2, 2), numpy.inf),
                   numpy.ones(2))
            for mo in bad:
                with self.subTest(shape=mo.shape):
                    solve.reset_mock()
                    with self.assertRaises(ValueError):
                        scanner(mol, mo_coeff=mo)
                    solve.assert_not_called()
            with self.assertRaises(TypeError):
                scanner(mol, mo_coeff=mf.mo_coeff.astype(complex))
            with self.assertRaises(TypeError):
                scanner(mol, mo_coeff=mf.mo_coeff.astype(str))

    def test_scanner_reuses_compatible_ci_with_independent_storage(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        moved = mol.set_geom_('H 0 0 0; H 0 0 .8', inplace=False)
        mf = scf.RHF(mol).run()
        if getattr(mf, '_chkfile', None) is not None:
            self.addCleanup(mf._chkfile.close)
        for use_df, weights in ((False, None), (True, (.5, .5))):
            with self.subTest(df=use_df, weights=weights):
                mc = gasscf.GASSCF(mf, 2, (1, 1), ncore=0)
                self.addCleanup(mc.close)
                if use_df:
                    mc = mc.density_fit().state_average(weights)
                    self.addCleanup(mc.close)
                mc.gasci()
                original_ci = mc.ci
                original_values = numpy.array(original_ci, copy=True)
                original_energy = mc.e_tot
                _, blocks = mc._normalized_restriction()
                mc.gas_restr, mc.gas_restr_type = blocks, 'spin-supergroup'
                scanner = mc.as_scanner()
                self.addCleanup(scanner.close)
                numpy.testing.assert_array_equal(scanner.ci, original_values)
                pairs = (zip(original_ci, scanner.ci) if weights is not None
                         else ((original_ci, scanner.ci),))
                for left, right in pairs:
                    self.assertFalse(numpy.shares_memory(left, right))
                # A second geometry must reuse the newly solved CI as well.
                for geometry in (mol, moved):
                    previous_ci = scanner.ci
                    with mock.patch.object(gasci, 'kernel', wraps=gasci.kernel) as solve:
                        energy = scanner(geometry)
                    initial = solve.call_args_list[0].kwargs['ci0']
                    numpy.testing.assert_array_equal(initial, previous_ci)
                    pairs = (zip(initial, previous_ci) if weights is not None
                             else ((initial, previous_ci),))
                    for left, right in pairs:
                        self.assertFalse(numpy.shares_memory(left, right))
                    self.assertEqual(scanner.scan_info['CI_source'], 'previous GAS CI guess')
                    # A fresh solve at the resulting orbitals is an energy reference.
                    ref = scanner.copy()
                    self.addCleanup(ref.close)
                    ref._clear_ci_guess()
                    self.assertAlmostEqual(energy, ref.gasci()[0], places=10)
                self.assertIs(mc.ci, original_ci)
                numpy.testing.assert_array_equal(mc.ci, original_values)
                self.assertEqual(mc.e_tot, original_energy)

    def test_as_scanner_runs_energy_scan_with_fixed_gas_model(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", ncore=1)
        mc.max_cycle_macro = 1
        mc.max_cycle_micro = 1
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-4
        mc.canonicalization = False
        mc.kernel(mf.mo_coeff)

        scanner = mc.as_scanner()
        energy = scanner("H 0 0 0; H 0 0 0.92; H 0 0 2.2; H 0 0 3.1")

        self.assertTrue(numpy.isfinite(energy))
        self.assertTrue(scanner.scan_info["native_converged"] in (True, False))
        self.assertEqual(scanner.scan_info["problem"]["gas_orbs"], (1, 1))
        self.assertEqual(scanner.scan_info["problem"]["nroots"], 1)
        self.assertIn("GASSCF", scanner.__class__.__name__)
        self.assertIs(scanner.as_scanner(), scanner)

    def test_scanner_reset_and_preparation_failure_invalidate_scan_report(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        moved = mol.set_geom_('H 0 0 0; H 0 0 .95', inplace=False)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        for use_df, weights in ((False, None), (True, (.5, .5))):
            mc = gasscf.GASSCF(mf, 2, (1, 1))
            self.addCleanup(mc.close)
            if use_df:
                mc = mc.density_fit().state_average(weights)
                self.addCleanup(mc.close)
            scanner = mc.as_scanner()
            self.addCleanup(scanner.close)
            for stage in ('reset', 'scf', 'projection', 'mo-validation'):
                with self.subTest(df=use_df, stage=stage):
                    scanner(mol)
                    self.assertTrue(scanner.converged)
                    self.assertIn('energy', scanner.scan_info)
                    previous_mo = scanner.mo_coeff
                    previous_ci = scanner.ci
                    with ExitStack() as patches:
                        kwargs = {}
                        error = RuntimeError
                        if stage == 'scf':
                            patches.enter_context(mock.patch.object(
                                type(scanner._scf), '__call__',
                                side_effect=RuntimeError('SCF preparation')))
                        elif stage == 'projection':
                            patches.enter_context(mock.patch.object(
                                gasscf.addons, 'project_init_guess',
                                side_effect=RuntimeError('MO projection')))
                        elif stage == 'mo-validation':
                            kwargs['mo_coeff'] = numpy.zeros_like(mf.mo_coeff)
                            error = ValueError
                        if stage == 'reset':
                            scanner.reset(moved)
                        else:
                            with self.assertRaises(error):
                                scanner(moved, **kwargs)
                    self.assertIs(scanner.mol, moved)
                    self.assertIs(scanner.mo_coeff, previous_mo)
                    self.assertIs(scanner.ci, previous_ci)
                    self.assertIsNone(scanner.e_tot)
                    self.assertIsNone(scanner.scan_info)
                    self.assertFalse(scanner.converged)
                    # Preserve native reuse of old orbitals as a projection guess.
                    energy = scanner(moved)
                    self.assertIn('projected previous', scanner.scan_info['MO_source'])
                    self.assertEqual(scanner.scan_info['energy'], energy)
                    self.assertTrue(scanner.converged)
                    ref = scanner.copy()
                    self.addCleanup(ref.close)
                    ref._clear_ci_guess()
                    self.assertAlmostEqual(energy, ref.gasci()[0], places=10)

    def test_scanner_rejects_symmetry_before_reset_or_scf(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .75', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        self.addCleanup(mf._chkfile.close)
        mc = gasscf.GASSCF(mf, 2, (1, 1))
        self.addCleanup(mc.close)
        tagged = lib.tag_array(mf.mo_coeff.copy(), orbsym=[0, 0])
        symmetric = mol.copy()
        symmetric.symmetry = True
        symmetric.build()
        for df in (False, True):
            source = mc.density_fit() if df else mc
            self.addCleanup(source.close)
            scanner = source.as_scanner()
            self.addCleanup(scanner.close)
            with mock.patch.object(scanner, 'reset', side_effect=AssertionError('reset')):
                for geometry, mo in ((symmetric, None), (mol, tagged)):
                    with self.assertRaisesRegex(NotImplementedError, 'symmetry'):
                        scanner(geometry, mo_coeff=mo)
                with mock.patch.object(scanner, 'extrasym', [0, 0]):
                    with self.assertRaisesRegex(NotImplementedError, 'extrasym'):
                        scanner(mol)
                    with self.assertRaisesRegex(NotImplementedError, 'extrasym'):
                        scanner.as_scanner()
            with self.assertRaisesRegex(NotImplementedError, 'point-group'):
                scanner.reset(symmetric)
            self.assertIs(scanner.mol, mol)
            self.assertIs(scanner._scf.mol, mol)
            if df:
                self.assertIs(scanner.with_df.mol, mol)

    def test_df_scanner_reset_updates_with_df_molecule(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", ncore=1)
        scanner = mc.density_fit().as_scanner()
        moved = mol.set_geom_(
            "H 0 0 0; H 0 0 0.92; H 0 0 2.2; H 0 0 3.1",
            inplace=False)

        scanner.reset(moved)

        self.assertIs(scanner.with_df.mol, moved)
        numpy.testing.assert_allclose(
            scanner.with_df.mol.atom_coords(), moved.atom_coords(),
            atol=0, rtol=0)

    def test_as_scanner_rejects_changed_system_model(self):
        mol = gto.M(
            atom='H 0 0 0; H 0 0 .9; H 0 0 2.2; H 0 0 3.1',
            basis='sto-3g', verbose=0)
        # Model rejection must precede reset and SCF; no electronic solve needed.
        mf = scf.RHF(mol)
        self.addCleanup(mf._chkfile.close)
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=((1, 1), (2, 2)),
            gas_restr_type='cumulative-occ', ncore=1)
        self.addCleanup(mc.close)
        mc = mc.state_average((.5, .5)).fix_spin_(shift=.2, ss=0.)
        self.addCleanup(mc.close)
        scanner = mc.as_scanner()
        self.addCleanup(scanner.close)
        molecular_changes = (
            {'atom': 'H 0 0 0; H 0 0 .9; H 0 0 2.2', 'spin': 1},
            {'basis': '6-31g'}, {'charge': 2}, {'spin': 2}, {'cart': True})
        for options in molecular_changes:
            with self.subTest(molecular_change=options):
                settings = dict(atom=mol.atom, basis='sto-3g', verbose=0)
                settings.update(options)
                changed = gto.M(**settings)
                with mock.patch.object(scanner, 'reset') as reset, \
                        mock.patch.object(type(scanner._scf), '__call__') as scf_call:
                    with self.assertRaisesRegex(ValueError, 'same atoms'):
                        scanner(changed)
                    reset.assert_not_called()
                    scf_call.assert_not_called()
        model_changes = (
            ('frozen', {'frozen': [0]}, {}, 'create a new scanner'),
            ('weights', {'weights': (.3, .7)}, {}, 'create a new scanner'),
            ('roots', {}, {'nroots': 3}, 'nroots/weights mismatch'),
            ('penalty-shift', {}, {'ss_penalty': .3}, 'create a new scanner'),
            ('penalty-target', {}, {'ss_value': 2.}, 'create a new scanner'),
            ('GAS-size', {'gas_restr': ((0, 2), (2, 2))}, {}, 'create a new scanner'),
            ('GAS-basis', {'gas_restr_type': 'supergroup',
                           'gas_restr': ((0, 2), (2, 0))}, {}, 'create a new scanner'))
        for name, mc_settings, solver_settings, message in model_changes:
            with self.subTest(model_change=name):
                trial = scanner.copy()
                self.addCleanup(trial.close)
                for key, value in mc_settings.items():
                    setattr(trial, key, value)
                for key, value in solver_settings.items():
                    setattr(trial.fcisolver, key, value)
                if name == 'GAS-basis':
                    # Equal determinant counts do not imply compatible CI bases.
                    trial.validate_capabilities()
                    with scanner.fcisolver.make_space(2, (1, 1)) as old, \
                            trial.fcisolver.make_space(2, (1, 1)) as new:
                        self.assertEqual(old.ndet, 2)
                        self.assertEqual(new.ndet, old.ndet)
                        old_full = fci_gas.gas2fci(numpy.ones(old.ndet), old)
                        new_full = fci_gas.gas2fci(numpy.ones(new.ndet), new)
                        self.assertFalse(numpy.array_equal(old_full, new_full))
                with mock.patch.object(trial, 'reset') as reset, \
                        mock.patch.object(type(trial._scf), '__call__') as scf_call:
                    with self.assertRaisesRegex(ValueError, message):
                        trial(mol)
                    reset.assert_not_called()
                    scf_call.assert_not_called()


class TestLogging(unittest.TestCase):
    """GAS labels, warning filtering and isolated output streams."""

    def test_log_filter_relabels_native_newton_output(self):
        buf = io.StringIO()
        stream = gasscf._GASSCFLogFilter(buf)
        stream.write(
            "WARN: SO-CASSCF (Second order CASSCF) is an experimental "
            "feature. Its performance is bad for large systems.\n"
            "Start SO-CASSCF (newton CASSCF)\n"
            "newton CASSCF converged in 6 macro steps\n"
            "CASSCF canonicalization\n"
            "CASSCF energy = -1.0\n"
            "CASCI E = -1.0  E(CI) = -0.2\n"
            "CAS (1e+1e, 2o), ncore = 1\n")
        stream.flush()
        out = buf.getvalue()

        self.assertIn("Start SO-GASSCF", out)
        self.assertIn("GASSCF converged", out)
        self.assertIn("GASSCF canonicalization", out)
        self.assertIn("GASSCF energy", out)
        self.assertIn("GASCI E", out)
        self.assertIn("E(GASCI)", out)
        self.assertIn("GAS (1e+1e, 2o)", out)
        self.assertNotIn("SO-CASSCF", out)
        self.assertNotIn("experimental feature", out)
        self.assertNotIn("performance is bad for large systems", out)
        self.assertNotIn("newton CASSCF", out)
        self.assertNotIn("CASCI E", out)

    def test_log_filter_suppresses_native_warning_without_global_stderr(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75",
                    basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None,
            ncore=0)
        out = io.StringIO()
        mc.stdout = out
        mc.verbose = 4
        old_sys_stderr = sys.stderr

        stdout, restore_stdout = mc._push_gasscf_log_labels()
        try:
            self.assertIs(sys.stderr, old_sys_stderr)
            log = gasscf._GASSCFLogger(mc.stdout, mc.verbose)
            log.warn(
                "SO-CASSCF (Second order CASSCF) is an experimental "
                "feature. Its performance is bad for large systems.")
        finally:
            if restore_stdout:
                mc.stdout.flush()
                mc.stdout = stdout

        self.assertIs(sys.stderr, old_sys_stderr)
        self.assertNotIn("SO-CASSCF", out.getvalue())
        self.assertNotIn("experimental feature", out.getvalue())
        self.assertNotIn("performance is bad for large systems",
                         out.getvalue())

    def test_dump_flags_uses_gas_labels(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(2,), gas_restr=None, ncore=0)
        buf = io.StringIO()
        terminal = io.StringIO()
        original_solver_stdout = mc.fcisolver.stdout
        mc.stdout = buf
        mc.verbose = 4
        old_sys_stdout = sys.stdout

        try:
            sys.stdout = terminal
            returned = mc.dump_flags()
        finally:
            sys.stdout = old_sys_stdout
        out = buf.getvalue()

        self.assertIs(returned, mc)
        self.assertIn("GAS (1e+1e, 2o)", out)
        self.assertIn("gas_orbs = (2,)", out)
        self.assertIn("gas_restr_type = spin-supergroup", out)
        self.assertIn("cache GAS helper plans", out)
        self.assertIn("max. cycles = 100", out)
        self.assertNotIn("CAS (1e+1e, 2o)", out)
        self.assertEqual(terminal.getvalue(), "")
        self.assertIs(mc.stdout, buf)
        self.assertIs(mc.fcisolver.stdout, original_solver_stdout)

    def test_density_fit_method_keeps_gas_labels_and_runs(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        mc = gasscf.GASSCF(
            mf, 2, (1, 1), gas_orbs=(1, 1), gas_restr=[[1, 1], [2, 2]],
            gas_restr_type="cumulative-occ", ncore=1)
        mc = mc.density_fit()
        mc.max_cycle_macro = 1
        mc.max_cycle_micro = 1
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-4
        mc.canonicalization = False

        buf = io.StringIO()
        mc.stdout = buf
        mc.verbose = 4
        e_tot, e_gas, ci, mo_coeff, mo_energy = mc.kernel(mf.mo_coeff)
        out = buf.getvalue()

        self.assertIsInstance(mc, gasscf.GASSCF)
        self.assertIsInstance(mc, mcdf._DFCAS)
        self.assertTrue(hasattr(mc, "with_df"))
        self.assertIn("DFGASSCF", out)
        self.assertNotIn("DFCASSCF", out)
        self.assertNotIn("DFCASCI", out)
        self.assertTrue(numpy.isfinite(e_tot))
        self.assertTrue(numpy.isfinite(e_gas))
        self.assertEqual(numpy.asarray(ci).ndim, 1)
        self.assertEqual(mo_coeff.shape, mf.mo_coeff.shape)
        self.assertIsNone(mo_energy)

    def test_repeated_kernel_and_scanner_spin_results_pass_sanity(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.9; H 0 0 2.2; H 0 0 3.1",
            basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        for spin_penalty in (False, True):
            with self.subTest(spin_penalty=spin_penalty):
                mc = gasscf.GASSCF(
                    mf, 2, (1, 1), gas_orbs=(2,), ncore=1)
                mc.max_cycle_macro = 1
                mc.max_cycle_micro = 1
                mc.canonicalization = False
                if spin_penalty:
                    mc.fix_spin_(shift=.2, ss=0)
                mc.verbose = 4
                mc.stdout = io.StringIO()
                errors = io.StringIO()
                # Do not let earlier tests hide a warning through warn-once.
                with mock.patch.dict(gasscf.lib.misc._warn_once_registry,
                                     {}, clear=True), \
                        mock.patch.object(sys, "stderr", errors):
                    mc.kernel(mf.mo_coeff)
                    mc.kernel(mc.mo_coeff, mc.ci)
                    scanner = mc.as_scanner()
                    for distance in (.92, .94):
                        energy = scanner(
                            "H 0 0 0; H 0 0 %s; H 0 0 2.2; H 0 0 3.1"
                            % distance)
                        self.assertTrue(numpy.isfinite(energy))
                        penalty = scanner.e_spin_penalty
                        if spin_penalty:
                            self.assertIsNotNone(penalty)
                        else:
                            self.assertIsNone(penalty)
                            penalty = 0.
                        self.assertAlmostEqual(
                            scanner.e_tot, energy, 10)
                        self.assertAlmostEqual(
                            scanner.e_tot - scanner.get_h1gas(scanner.mo_coeff)[1],
                            scanner.e_gas, 10)
                self.assertNotIn("does not have attributes", errors.getvalue())
                self.assertNotIn("does not have attributes", mc.stdout.getvalue())


if __name__ == "__main__":
    print("Full Tests for GASSCF")
    unittest.main()
