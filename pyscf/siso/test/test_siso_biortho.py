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

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose

from pyscf.fci import cistring
from pyscf.siso import siso
from pyscf.siso import siso_biortho


class KnownValues(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
