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
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

""" Occupation smearing for SCF methods in plane-wave basis
"""

from pyscf.pbc.scf.addons import _SmearingKSCF
from pyscf.pbc.pwscf import khf
from pyscf.lib import logger
from pyscf import lib
from pyscf import __config__
import numpy as np


SMEARING_METHOD = getattr(__config__, 'pbc_scf_addons_smearing_method', 'fermi')


def smearing(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    '''Fermi-Dirac or Gaussian smearing'''
    if not isinstance(mf, khf.PWKSCF):
        raise ValueError("For PW mode only")

    if isinstance(mf, _SmearingPWKSCF):
        mf.sigma = sigma
        mf.smearing_method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    return lib.set_class(_SmearingPWKSCF(mf, sigma, method, mu0, fix_spin),
                         (_SmearingPWKSCF, mf.__class__))


def smearing_(mf, *args, **kwargs):
    mf1 = smearing(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf


def has_smearing(mf):
    return isinstance(mf, _SmearingPWKSCF)


class _SmearingPWKSCF(_SmearingKSCF):

    def get_mo_occ(self, moe_ks=None, C_ks=None, nocc=None):
        cell = self.cell
        if nocc is None:
            nocc = cell.nelectron / 2.0
        else:
            assert nocc == cell.nelectron / 2.0
        if moe_ks is None:
            raise NotImplementedError(
                "PWKSCF smearing without mo energy input"
            )
        mocc_ks = self.get_occ(mo_energy_kpts=moe_ks, mo_coeff_kpts=C_ks)
        if self.istype("PWKUHF"):
            mocc_ks = [
                [2 * occ for occ in mocc_ks[0]],
                [2 * occ for occ in mocc_ks[1]]
            ]

        return mocc_ks

    def energy_tot(self, C_ks, mocc_ks, moe_ks=None, mesh=None, Gv=None,
                   vj_R=None, exxdiv=None):
        e_tot = khf.PWKRHF.energy_tot(self, C_ks, mocc_ks, moe_ks=moe_ks,
                                      mesh=mesh, Gv=Gv, vj_R=vj_R,
                                      exxdiv=exxdiv)
        if self.sigma and self.smearing_method and self.entropy is not None:
            self.e_free = e_tot - self.sigma * self.entropy
            self.e_zero = e_tot - self.sigma * self.entropy * .5
            logger.info(self, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                        e_tot, self.e_free, self.e_zero)
        return e_tot

