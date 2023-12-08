#!/usr/bin/env python
# Copyright 2023 The PySCF Developers. All Rights Reserved.
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

from pyscf import lib
from pyscf.scf import _vhf
from pyscf.lib import logger
from pyscf.lrdf import lrdf
from pyscf.hessian import rhf as rhf_hess
from pyscf.df.hessian import rhf as df_rhf_hess

class Hessian(rhf_hess.Hessian):
    def __init__(self, mf):
        assert isinstance(mf.with_df, lrdf.LRDensityFitting)
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.auxbasis_response = 0
        rhf_hess.Hessian.__init__(self, mf)

    def partial_hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None,
                          atmlst=None, max_memory=4000, verbose=None):
        lrdf_obj = self.base.with_df
        omega = lrdf_obj.omega
        with self.mol.with_short_range_coulomb(omega):
            e1, ej, ek = rhf_hess._partial_hess_ejk(
                self, mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose, True)
            de2 = ej - ek

        with lrdf_obj.range_coulomb(omega):
            with lib.temporary_env(lrdf_obj, auxmol=lrdf_obj.lr_auxmol):
                de2 += df_rhf_hess.partial_hess_elec(
                    self, mo_energy, mo_coeff, mo_occ, atmlst, max_memory, verbose)
        return de2

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        mol = self.mol
        lrdf_obj = self.base.with_df
        omega = lrdf_obj.omega

        nao, nmo = mo_coeff.shape
        mocc = mo_coeff[:,mo_occ>0]
        dm0 = mocc.dot(mocc.T) * 2

        pmol = mol.copy()
        pmol.omega = -omega
        vhfopt = lrdf._VHFOpt(pmol, 'int2e_ip1', omega=omega)
        vhfopt._this.q_cond = lrdf_obj._vhfopt._this.q_cond
        vhfopt._this.dm_cond = lrdf_obj._vhfopt._this.dm_cond
        aoslices = mol.aoslice_by_atom()

        h1ao = [None] * mol.natm
        with lrdf_obj.range_coulomb(omega):
            with lib.temporary_env(lrdf_obj, auxmol=lrdf_obj.lr_auxmol):
                for ia, h1, vj1, vk1 in df_rhf_hess._gen_jk(
                        self, mo_coeff, mo_occ, chkfile, atmlst, verbose, True):
                    h1 += vj1 - vk1 * .5

                    shl0, shl1, p0, p1 = aoslices[ia]
                    shls_slice = (shl0, shl1) + (0, mol.nbas)*3
                    vj1, vj2, vk1, vk2 = rhf_hess._get_jk(
                        pmol, 'int2e_ip1', 3, 's2kl',
                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                         'lk->s1ij', -dm0         ,  # vj2
                         'li->s1kj', -dm0[:,p0:p1],  # vk1
                         'jk->s1il', -dm0         ], # vk2
                        shls_slice=shls_slice, vhfopt=vhfopt)
                    vhf = vj1 - vk1*.5
                    vhf[:,p0:p1] += vj2 - vk2*.5
                    h1 += vhf + vhf.transpose(0,2,1)

                    if chkfile is None:
                        h1ao[ia] = h1
                    else:
                        key = 'scf_f1ao/%d' % ia
                        lib.chkfile.save(chkfile, key, h1)

        if chkfile is None:
            return h1ao
        else:
            return chkfile
