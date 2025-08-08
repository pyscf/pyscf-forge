""" Check if the PW code gives same MO energies as the GTO code for a given
wave function
"""

import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, df, scf, pwscf
from pyscf.pbc.pwscf import khf, pw_helper
from pyscf import lib
import pyscf.lib.parameters as param

import unittest


class KnownValues(unittest.TestCase):
    def _run_test(self, cell, kpts, exxdiv):
        # GTO
        gmf = scf.KRHF(cell, kpts)
        gmf.exxdiv = exxdiv
        gmf.kernel()

        vpp = lib.asarray(gmf.with_df.get_pp(kpts))
        vkin = lib.asarray(gmf.cell.pbc_intor('int1e_kin', 1, 1, kpts))
        dm = gmf.make_rdm1()
        vj, vk = df.FFTDF(cell).get_jk(dm, kpts=kpts)

        nkpts = len(kpts)
        moe_comp_ks = np.zeros((4,nkpts), dtype=np.complex128)
        for k in range(nkpts):
            moe_comp_ks[0,k] = np.einsum("ij,ji->", vkin[k], dm[k])
            moe_comp_ks[1,k] = np.einsum("ij,ji->", vpp[k], dm[k])
            moe_comp_ks[2,k] = np.einsum("ij,ji->", vj[k], dm[k]) * 0.5
            moe_comp_ks[3,k] = -np.einsum("ij,ji->", vk[k], dm[k]) * 0.25

        # PW (both vanilla and ACE)
        pmf = pwscf.KRHF(cell, kpts)
        pmf.init_pp()
        pmf.init_jk()
        pmf.exxdiv = exxdiv
        no_ks = pw_helper.get_nocc_ks_from_mocc(gmf.mo_occ)
        C_ks = pw_helper.get_C_ks_G(cell, kpts, gmf.mo_coeff, no_ks)
        mocc_ks = khf.get_mo_occ(cell, C_ks=C_ks)
        pmf.update_pp(C_ks)
        vj_R = pmf.get_vj_R(C_ks, mocc_ks)
        mesh = cell.mesh
        Gv = cell.get_Gv(mesh)

        pmf.with_jk.ace_exx = False
        pmf.update_k(C_ks, mocc_ks)
        moe_comp_ks_pw = np.zeros((4, nkpts), dtype=np.complex128)
        for k in range(nkpts):
            C_k = C_ks[k]
            kpt = kpts[k]
            moe = pmf.apply_Fock_kpt(C_k, kpt, mocc_ks, mesh, Gv, vj_R, exxdiv,
                                    ret_E=True)[1]
            moe_comp_ks_pw[0,k] = moe[0]
            moe_comp_ks_pw[1,k] = moe[1] + moe[2]
            moe_comp_ks_pw[2:,k] = moe[3:]

        pmf.with_jk.ace_exx = True
        pmf.update_k(C_ks, mocc_ks)
        ace_moe_comp_ks_pw = np.zeros((4, nkpts), dtype=np.complex128)
        for k in range(nkpts):
            C_k = C_ks[k]
            kpt = kpts[k]
            moe = pmf.apply_Fock_kpt(C_k, kpt, mocc_ks, mesh, Gv, vj_R, exxdiv,
                                    ret_E=True)[1]
            ace_moe_comp_ks_pw[0,k] = moe[0]
            ace_moe_comp_ks_pw[1,k] = moe[1] + moe[2]
            ace_moe_comp_ks_pw[2:,k] = moe[3:]

        maxe_real = np.max(np.abs(moe_comp_ks.real - moe_comp_ks_pw.real))
        maxe_imag = np.max(np.abs(moe_comp_ks.imag - moe_comp_ks_pw.imag))
        ace_maxe_real = np.max(np.abs(moe_comp_ks.real - ace_moe_comp_ks_pw.real))
        ace_maxe_imag = np.max(np.abs(moe_comp_ks.imag - ace_moe_comp_ks_pw.imag))
        print(maxe_real, maxe_imag)
        print(ace_maxe_real, ace_maxe_imag)

        assert(maxe_real < 1e-6)
        assert(maxe_imag < 1e-6)
        assert(ace_maxe_real < 1e-6)
        assert(ace_maxe_imag < 1e-6)

    def test_gto_vs_pw(self):
        nk = 2
        kmesh = [2,1,1]
        ke_cutoff = 150
        pseudo = "gth-pade"
        exxdiv = None
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
        a = np.asarray(
            [[0.       , 1.78339987, 1.78339987],
            [1.78339987, 0.        , 1.78339987],
            [1.78339987, 1.78339987, 0.        ]])

        # cell
        cell = gto.Cell(
            atom=atom,
            a=a,
            basis="gth-szv",
            pseudo=pseudo,
            ke_cutoff=ke_cutoff
        )
        cell.build()
        cell.verbose = 5
        kpts = cell.make_kpts(kmesh)
        self._run_test(cell, kpts, exxdiv)


if __name__ == "__main__":
    unittest.main()
