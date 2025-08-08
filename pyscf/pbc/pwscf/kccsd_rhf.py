#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <osirpt.sun@gmail.com>
#

""" Restricted CCSD in plane-wave basis
"""

import h5py
import numpy as np

from pyscf.pbc import cc
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, get_frozen_mask,
                               padded_mo_energy, padding_k_idx)
from pyscf.pbc.pwscf.pw_helper import get_kcomp
from pyscf.pbc.pwscf.ao2mo.molint import get_molint_from_C
from pyscf.pbc.pwscf.khf import THR_OCC
from pyscf import lib
from pyscf.lib import logger


def padded_mo_coeff(mp, mo_coeff):
    frozen_mask = get_frozen_mask(mp)
    padding_convention = padding_k_idx(mp, kind="joint")
    nkpts = mp.nkpts

    result = np.zeros((nkpts, mp.nmo, mo_coeff[0].shape[1]),
                      dtype=mo_coeff[0].dtype)
    for k in range(nkpts):
        result[np.ix_([k], padding_convention[k],
               np.arange(result.shape[2]))] = mo_coeff[k][frozen_mask[k], :]

    return result


class PWKRCCSD:
    def __init__(self, mf, frozen=None):
        self._scf = mf
        self.mo_occ = mf.mo_occ
        self.mo_energy = mf.mo_energy
        self.frozen = frozen
        self.kpts = mf.kpts
        self.mcc = None

        # not input options
        self._nmo = None
        self._nocc = None
        self.nkpts = len(self.kpts)

    def kernel(self, eris=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if eris is None: eris = self.ao2mo()
        cput0 = logger.timer(self._scf, 'CCSD init eri', *cput0)
        self.mcc = cc.kccsd_rhf.RCCSD(self._scf)
        self.mcc.kernel(eris=eris)

        return self.mcc

    def ao2mo(self):
        return _ERIS(self)

    @property
    def e_corr(self):
        if self.mcc is None:
            raise RuntimeError("kernel must be called first.")
        return self.mcc.e_corr

# mimic KMP2
    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask


class _ERIS:
    def __init__(self, cc):
        mf = cc._scf
        cell = mf.cell
        kpts = mf.kpts
        nkpts = len(kpts)

        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        with h5py.File(mf.chkfile, "r") as f:
            mo_coeff = [get_kcomp(f["mo_coeff"], k) for k in range(nkpts)]

# padding
        mo_coeff = padded_mo_coeff(cc, mo_coeff)
        mo_energy = padded_mo_energy(cc, mo_energy)
        mo_occ = padded_mo_energy(cc, mo_occ)

        self.e_hf = mf.e_tot
        self.mo_energy = np.asarray(mo_energy)
# remove ewald correction
        moe_noewald = np.zeros_like(self.mo_energy)
        for k in range(nkpts):
            moe = self.mo_energy[k].copy()
            moe[mo_occ[k]>THR_OCC] += mf._madelung
            moe_noewald[k] = moe
        self.fock = np.asarray([np.diag(moe.astype(np.complex128)) for moe in moe_noewald])

        eris = get_molint_from_C(cell, mo_coeff,
                                 kpts).transpose(0,2,1,3,5,4,6)

        no = cc.nocc
        self.oooo = eris[:,:,:,:no,:no,:no,:no]
        self.ooov = eris[:,:,:,:no,:no,:no,no:]
        self.oovv = eris[:,:,:,:no,:no,no:,no:]
        self.ovov = eris[:,:,:,:no,no:,:no,no:]
        self.voov = eris[:,:,:,no:,:no,:no,no:]
        self.vovv = eris[:,:,:,no:,:no,no:,no:]
        self.vvvv = eris[:,:,:,no:,no:,no:,no:]

        eris = None


if __name__ == "__main__":
    a0 = 1.78339987
    atom = "C 0 0 0; C %.10f %.10f %.10f" % (a0*0.5, a0*0.5, a0*0.5)
    a = np.asarray([
            [0., a0, a0],
            [a0, 0., a0],
            [a0, a0, 0.]])

    from pyscf.pbc import gto, scf, pwscf
    cell = gto.Cell(atom=atom, a=a, basis="gth-szv", pseudo="gth-pade",
                    ke_cutoff=50)
    cell.build()
    cell.verbose = 5

    kpts = cell.make_kpts([2,1,1])

    mf = scf.KRHF(cell, kpts)
    mf.kernel()

    mcc = cc.kccsd_rhf.RCCSD(mf)
    mcc.kernel()

    from pyscf.pbc.pwscf.pw_helper import gtomf2pwmf
    pwmf = gtomf2pwmf(mf)
    pwmcc = PWKRCCSD(pwmf).kernel()

    assert(np.abs(mcc.e_corr - pwmcc.e_corr) < 1e-5)
