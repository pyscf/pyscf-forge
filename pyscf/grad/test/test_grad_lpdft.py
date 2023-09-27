#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>

import numpy as np
import unittest

from pyscf import scf, gto, mcscf
from pyscf import mcpdft
from pyscf.mcpdft import _dms


def setUpModule():
    global h2, lih
    h2 = scf.RHF(gto.M(atom='H 0 0 0; H 1.2 0 0', basis='sto-3g',
                       #output='/dev/null', verbose=0)).run()
                        verbose=0)).run()
    lih = scf.RHF(gto.M(atom='Li 0 0 0; H 1.2 0 0', basis='sto-3g',
                        output='/dev/null', verbose=0)).run()


def tearDownModule():
    global h2, lih
    h2.mol.stdout.close()
    lih.mol.stdout.close()
    del h2, lih


def wfn_case(kv, mc_grad):
    mc = mc_grad.base
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nmo = mc.mo_coeff.shape[1]
    nocc, nvir = ncore + ncas, nmo - ncore - ncas
    ngorb = ncore * ncas + nocc * nvir

    fcasscf = mcscf.CASSCF(mc._scf, ncas, nelecas)
    fcasscf.__dict__.update(mc.__dict__)

    def get_energy(mo, ci):
        ci = [c.ravel() for c in ci]
        casdm1s_0, casdm2_0 = mc.get_casdm12_0(ci=ci)
        dm1s_0 = _dms.casdm1s_to_dm1s(mc, casdm1s=casdm1s_0, mo_coeff=mo)
        dm1_0 = dm1s_0[0] + dm1s_0[1]

        casdm1s = mc.make_one_casdm1s(ci=ci)
        casdm1 = casdm1s[0] + casdm1s[1]
        casdm2 = mc.make_one_casdm2(ci=ci)
        dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s, mo_coeff=mo)
        dm1 = dm1s[0] + dm1s[1]

        # these are not agreeing when I calculate it twice in a row... why
        veff1, veff2, E_ot = mc.get_pdft_veff(mo=mo, casdm1s=casdm1s_0, casdm2=casdm2_0, drop_mcwfn=True, incl_energy=True)

        # The constant term
        ref_e = mc.get_lpdft_hconst(E_ot, casdm1s_0, casdm2_0, veff1=veff1, veff2=veff2, mo_coeff=mo)

        # The 1-electron term
        h1e = mc.get_hcore() + veff1 + mc._scf.get_j(dm=dm1_0)
        ref_e += np.tensordot(h1e, dm1)

        ref_e += veff2.energy_core
        ref_e += np.tensordot(veff2.vhf_c[ncore:nocc, ncore:nocc], casdm1)
        ref_e += 0.5 * np.tensordot(veff2.papa[ncore:nocc, :, ncore:nocc, :],
                                    casdm2, axes=4)

        return ref_e

    ref_e = get_energy(mc.mo_coeff, mc.ci)
    g_all, hdiag_all = mc_grad.get_wfn_response(incl_diag=True, verbose=0)

    g_numzero = np.abs(g_all) < 1e-8
    hdiag_all[g_numzero] = 1
    x0 = -g_all / hdiag_all
    xorb_norm = np.linalg.norm(x0[:ngorb])
    xci_norm = np.linalg.norm(x0[ngorb:])
    x0 = g_all * np.random.rand(*x0.shape) - 0.5
    x0[g_numzero] = 0
    x0[:ngorb] *= xorb_norm / np.linalg.norm(x0[:ngorb])
    x0[ngorb:] *= xci_norm / (np.linalg.norm(x0[ngorb:]) or 1)
    err_tab = np.zeros((0, 2))

    def seminum(x):
        uorb, ci1 = mcscf.newton_casscf.extract_rotation(fcasscf, x, 1, mc.ci)
        mo1 = mc.rotate_mo(mc.mo_coeff, uorb)
        semi_num_e = get_energy(mo1, ci1)
        return semi_num_e - ref_e

    for ix, p in enumerate(range(2)):
        x1 = x0/(2**p)
        x1_norm = np.linalg.norm(x1)
        dg_test = np.dot(g_all, x1)
        dg_ref = seminum(x1)
        print(f"dg_test: {dg_test:.10f}\tdg_ref: {dg_ref:.10f}\tratio: {dg_test/dg_ref:.5f}")
        dg_err = abs((dg_test - dg_ref)/dg_ref)
        err_tab = np.append(err_tab, [[x1_norm, dg_err]], axis=0)
        if ix > 0:
            conv_tab = err_tab[1:ix+1, :] / err_tab[:ix, :]

        if ix > 1 and np.all(np.abs(conv_tab[-3:, -1] - 0.5) < 0.01) and abs(err_tab[-1, 1]) < 1e-3:
            break


    # with kv.subTest(q='x'):
    #     kv.assertAlmostEqual(conv_tab[-1, 0], 0.5, 9)
    #
    # with kv.subTest(q='de'):
    #     kv.assertLess(abs(err_tab[-1, 1]), 1e-3)
    #     kv.assertAlmostEqual(conv_tab[-1, 1], 0.5, delta=0.05)


class KnownValues(unittest.TestCase):

    def test_wfn_response(self):
        np.random.seed(1)
        for mol, mf in zip(("H2", "LiH"), (h2, lih)):
            for state, nel in zip(('Singlet', 'Triplet'), (2, (2, 0))):
                max_roots = 5 if state == 'Singlet' else 2
                for nroots in range(4, max_roots):
                    for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                        weights = [1.0/nroots, ] * nroots
                        mc = mcpdft.CASSCF(mf, fnal, 2, nel, grids_level=1).multi_state(weights, method="lin").run()
                        mc_grad = mc.nuc_grad_method(state=0)
                        print(f"{mol}\t{state}\t{nroots}\t{fnal}")
                        with self.subTest(mol=mol, state=state, fnal=fnal):
                            wfn_case(self, mc_grad)
                        print("============================================")

                        return


if __name__ == "__main__":
    print("Full Tests for L-PDFT gradients API")
    unittest.main()
