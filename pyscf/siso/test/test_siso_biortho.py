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

import io
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg

from pyscf import fci, gto, mcscf, scf
from pyscf.fci import cistring
from pyscf.siso import siso
from pyscf.siso import siso_biortho


class KnownValues(unittest.TestCase):

    def test_siso_kernel_calls_finalize(self):
        my_siso = object.__new__(siso_biortho.SISO)
        energies = np.array([-1.0, -0.9])
        vectors = np.eye(2)

        with mock.patch.object(
                siso_biortho.SI, "kernel",
                return_value=(energies, vectors)) as si_kernel, \
             mock.patch.object(my_siso, "_finalize") as finalize:
            result = my_siso.kernel()

        si_kernel.assert_called_once_with()
        finalize.assert_called_once_with()
        assert_allclose(result[0], energies)
        assert_allclose(result[1], vectors)

    def test_siso_finalize_reports_energies(self):
        output = io.StringIO()
        my_siso = object.__new__(siso_biortho.SISO)
        my_siso.mc = SimpleNamespace(stdout=output, verbose=4)
        my_siso.si_energies = np.array([-1.0, -0.9])

        my_siso._finalize()

        text = output.getvalue()
        self.assertIn("Spin Orbit Coupling Energetics", text)
        self.assertIn("SO-CASSI State 1 Total Energy = -0.9000000000", text)
        self.assertIn("21947.46314", text)

    def test_siso_initialization_reports_flags_and_energies(self):
        output = io.StringIO()
        my_siso = object.__new__(siso_biortho.SISO)
        my_siso.mc = SimpleNamespace(stdout=output, verbose=4)
        my_siso.modelspace = ((1, 2, None), (1, 4, None))
        my_siso.ci = [np.ones(1), np.ones(1)]
        my_siso.energies = np.array([-0.9, -1.0])
        my_siso.state_twos = np.array([1, 3])
        my_siso.lu_threshold = 1e-6
        my_siso.linear_dep_threshold = 1e-9
        my_siso.somf = True
        my_siso.amf = True
        my_siso.mmf = False
        my_siso.soc1e = True
        my_siso.soc2e = True
        my_siso.ham = "BP"

        my_siso._dump_flags()
        my_siso._dump_soc_flags()
        my_siso._initialize()

        text = output.getvalue()
        self.assertIn("model space: ((1, 2, None), (1, 4, None))", text)
        self.assertIn("linear dependency threshold: 1.000e-09", text)
        self.assertIn("SOMF: True", text)
        self.assertIn("SOC Hamiltonian: BP", text)
        self.assertIn("Spin Orbit Free Energetics", text)
        self.assertIn("State 0 Total Energy = -1.0000000000 S^2 = 3.75", text)
        self.assertIn("State 1 Total Energy = -0.9000000000 S^2 = 0.75", text)
        self.assertIn("21947.46314", text)

    def test_scalar_integral_dispatch(self):
        mo = np.eye(2)
        mol = SimpleNamespace(energy_nuc=lambda: 0.5)
        mf = SimpleNamespace(
            mol=mol,
            _eri=np.arange(6.0),
            get_hcore=lambda mol: np.diag([1.0, 2.0]),
        )

        for use_df in (False, True):
            with self.subTest(use_df=use_df):
                mc = SimpleNamespace(
                    _scf=mf,
                    ncore=1,
                    ncas=1,
                    get_jk=mock.Mock(return_value=(
                        np.diag([0.4, 0.2]), np.diag([0.2, 0.1]))),
                )
                if use_df:
                    mc.with_df = mock.Mock()
                    mc.with_df.ao2mo.return_value = np.asarray([3.0])

                with mock.patch.object(
                        siso_biortho.ao2mo, "general",
                        return_value=np.asarray([4.0])) as general:
                    _, _, h2 = siso_biortho._mixed_scalar_integrals(
                        mc, mo, mo)

                mc.get_jk.assert_called_once_with(mol, mock.ANY, hermi=0)
                if use_df:
                    mc.with_df.ao2mo.assert_called_once()
                    general.assert_not_called()
                    assert_allclose(h2, [[[[3.0]]]])
                else:
                    general.assert_called_once()
                    assert_allclose(h2, [[[[4.0]]]])

    def test_soc_action_uses_contraction_interface(self):
        rng = np.random.default_rng(12)
        norb = nelec = 4
        zmat = (rng.standard_normal((3, norb, norb))
                + 1j * rng.standard_normal((3, norb, norb)))

        cases = (
            ("ss", 0, (6, 6), siso.contract_same_spin,
             cistring.gen_linkstr_index(range(norb), 2),
             cistring.gen_linkstr_index(range(norb), 2)),
            ("ssp", 0, (6, 6), siso.contract_spin_plus,
             cistring.gen_des_str_index(range(norb), 3),
             cistring.gen_cre_str_index(range(norb), 1)),
            ("ssm", 2, (4, 4), siso.contract_spin_minus,
             cistring.gen_cre_str_index(range(norb), 2),
             cistring.gen_des_str_index(range(norb), 2)),
        )
        for mode, twos, shape, contract, linka, linkb in cases:
            with self.subTest(mode=mode):
                ci = (rng.standard_normal(shape)
                      + 1j * rng.standard_normal(shape))
                reference = contract(zmat, ci[None], (linka, linkb))[:, 0]
                result = siso_biortho._soc_action(
                    mode, zmat, ci, norb, nelec, twos)
                assert_allclose(result, reference, atol=1e-14)


class ScalarSI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="Li 0 0 0; H 0 0 1.6",
            basis="sto-3g",
            verbose=0,
        )
        cls.mf = scf.RHF(cls.mol).run()
        cls.mc = mcscf.CASCI(cls.mf, 2, 2)
        cls.mc.fcisolver = fci.solver(cls.mol, singlet=True)
        cls.mc.fcisolver.nroots = 2
        cls.mc.kernel()

        cls.ci = []
        cls.mo = []
        active = slice(cls.mc.ncore, cls.mc.ncore + cls.mc.ncas)
        for ci, angle in zip(cls.mc.ci, (0.21, -0.34)):
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
            mo = cls.mc.mo_coeff.copy()
            mo[:, active] = mo[:, active] @ rotation
            cls.mo.append(mo)
            cls.ci.append(fci.addons.transform_ci_for_orbital_rotation(
                ci, cls.mc.ncas, cls.mc.nelecas, rotation))

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.mc, cls.ci, cls.mo

    def test_rotated_state_representations_recover_casci(self):
        si = siso_biortho.SI(
            self.mc,
            [(2, 1)],
            self.ci,
            self.mo,
        )
        energies, vectors = si.kernel()

        assert_allclose(energies, self.mc.e_tot, atol=1e-10)
        assert_allclose(si.overlap, np.eye(2), atol=1e-10)
        assert_allclose(si.hamiltonian, np.diag(self.mc.e_tot), atol=1e-10)
        assert_allclose(
            vectors.conj().T @ si.overlap @ vectors,
            np.eye(2),
            atol=1e-12,
        )

    def test_transition_density_includes_inactive_orbitals(self):
        si = siso_biortho.SI(
            self.mc, [(2, 1)], self.ci, self.mo).build()
        ao_overlap = self.mf.get_ovlp()
        total_electrons = 2 * self.mc.ncore + sum(self.mc.nelecas)

        for left, right in ((0, 0), (0, 1)):
            dm1_active = si.transition_rdm1(
                left, right, basis="biorthogonal")
            dm1_ao = si.transition_rdm1(left, right, basis="ao")
            pair_overlap = si._get_pair(left, right).overlap

            assert_allclose(
                np.trace(dm1_active),
                sum(self.mc.nelecas) * pair_overlap,
                atol=1e-11,
            )
            assert_allclose(
                np.einsum("uv,uv->", ao_overlap, dm1_ao),
                total_electrons * pair_overlap,
                atol=1e-11,
            )

        tdm1, tdm2 = si.transition_rdm12(0, 1)
        self.assertEqual(tdm1.shape, (self.mc.ncas,) * 2)
        self.assertEqual(tdm2.shape, (self.mc.ncas,) * 4)

    def test_linearly_dependent_model_states(self):
        si = siso_biortho.SI(
            self.mc,
            [(2, 1)],
            [self.ci[0], self.ci[0]],
            [self.mo[0], self.mo[0]],
            energies=[self.mc.e_tot[0], self.mc.e_tot[0]],
        )
        with self.assertRaises(linalg.LinAlgError):
            si.kernel()

    def test_modelspace_reorders_state_data_with_spin_blocks(self):
        mol = SimpleNamespace(symmetry=False)
        mf = SimpleNamespace(mol=mol, get_ovlp=lambda: np.eye(3))
        mc = SimpleNamespace(
            _scf=mf,
            ncore=0,
            ncas=3,
            nelecas=(2, 1),
            e_tot=np.array([4.0, 2.0]),
            stdout=sys.stdout,
            verbose=0,
        )
        quartet_ci = np.ones((1, 1))
        doublet_ci = np.zeros((3, 3))
        doublet_ci[0, 0] = 1.0

        si = siso_biortho.SI(
            mc,
            [(1, 4), (1, 2)],
            [quartet_ci, doublet_ci],
            [np.eye(3), np.eye(3)],
        )

        self.assertEqual(si.modelspace, ((1, 2, None), (1, 4, None)))
        assert_allclose(si.state_twos, [1, 3])
        assert_allclose(si.energies, [2.0, 4.0])
        assert_allclose(si.ci[0], doublet_ci)
        assert_allclose(si.ci[1], quartet_ci)
        self.assertIs(si.run(), si)
        with self.assertRaisesRegex(ValueError, "same spin"):
            si.transition_rdm1(0, 1)
        with self.assertRaisesRegex(ValueError, "same spin"):
            si.transition_rdm12(0, 1)

    def test_rejects_unnormalized_and_complex_inputs(self):
        with self.assertRaisesRegex(ValueError, "not normalized"):
            siso_biortho.SI(
                self.mc,
                [(1, 1)],
                [2.0 * self.ci[0]],
                [self.mo[0]],
                energies=[self.mc.e_tot[0]],
            )

        with self.assertRaisesRegex(NotImplementedError, "Complex CI"):
            siso_biortho.SI(
                self.mc,
                [(1, 1)],
                [self.ci[0].astype(complex) * (1.0 + 1.0j)],
                [self.mo[0]],
                energies=[self.mc.e_tot[0]],
            )


if __name__ == "__main__":
    unittest.main()
