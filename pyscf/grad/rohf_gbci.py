#!/usr/bin/env python
#
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
# Author: Jiseong Park <fark4308@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

"""
Analytic gradients for GBCI

This module provides nuclear gradients for GBCI calculations.

Author: Jiseong Park <fark4308@snu.ac.kr>
"""

import numpy
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.grad.mp2 import _shell_prange
from pyscf.grad.rohf_casci import (
    _build_rohf_response_data,
    _solve_rohf_adjoint,
    _build_rohf_response_ao,
)
from pyscf.grad.gbci import (
    mo_to_um,
    make_1rdm_list,
    make_2rdm_list,
    make_contracted_H_list,
    get_X,
    get_h1eff_for_grad,
    _bath_rotation_zvec,
)


def grad_elec(
    gbci_grad, ref_mo_coeff, ref_mo_energy, mo_list, moe_list,
    conf_info_list, dmet_core_list, ov_list, ecore_list, ci,
    atmlst=None, verbose=None,
):
    mc = gbci_grad.base
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(gbci_grad, verbose)
    mol = gbci_grad.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = ref_mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    bath = list(numpy.arange(0,ncore)) + list(numpy.arange(ncore+ncas, nao))
    num_group = mo_list.shape[0]

    s1e = mc._scf.get_ovlp(mol)
    um_list = mo_to_um(ncas, ncore, ref_mo_coeff, mo_list, s1e)

    mo_cas = ref_mo_coeff[:,ncore:nocc]
    ordm_list = make_1rdm_list(ci, ncas, nelecas, conf_info_list, ov_list)
    trdm_list = make_2rdm_list(ci, ncas, nelecas, conf_info_list, ov_list)
    h1eff = get_h1eff_for_grad(mc, ref_mo_coeff, mo_cas, dmet_core_list)
    eri = mc.get_h2eff(ref_mo_coeff)

    H_list = make_contracted_H_list(ci, ncas, nelecas, ncore, conf_info_list, h1eff, eri, ecore_list, ov_list)
    group_prob = numpy.zeros(num_group)
    conf_info_list = conf_info_list.reshape(-1)
    for i in range(num_group):
        ci = ci.reshape(-1)
        group_where = numpy.where(conf_info_list == i)
        group_prob[i] = (numpy.abs(ci)**2)[group_where].sum()
    Xa, Xx  = get_X(mc, h1eff, ov_list, ordm_list, trdm_list, um_list, H_list, group_prob)
    xzvec, xzvec_ao, zy, as_occ = _bath_rotation_zvec(
        mc, ref_mo_coeff, num_group, bath, mo_list, moe_list, um_list, conf_info_list, Xx)

    h1 = mc.get_hcore()
    Imat = Xa + zy
    orbital_gradient = Imat - Imat.T
    response_data = _build_rohf_response_data(mc._scf, mol, ref_mo_coeff, h1, ncore, nocc, include_active_active=True)
    zvec, g_ref, same_weighted_pairs= _solve_rohf_adjoint(mc._scf,ref_mo_coeff,orbital_gradient,response_data)
    # _build_rohf_response_ao follows the CASCI convention
    # orbital_gradient = 2 * (Imat - Imat.T).  GBCI's Imat is already in
    # the full orbital-gradient convention, so its energy-weighted overlap
    # matrix is one half of the CASCI-style input.
    (zvec_ao, overlap_ao, fock_seed_a_ao, fock_seed_b_ao) = _build_rohf_response_ao(
        ref_mo_coeff, Imat * .5, zvec, g_ref, same_weighted_pairs, response_data)

    dma = response_data.dma
    dmb = response_data.dmb
    wtot = fock_seed_a_ao + fock_seed_b_ao
    dmtot = dma + dmb

    casdm1 = ref_mo_coeff[:,ncore:ncore+ncas] @ ordm_list.sum(axis = (0,1)) @ ref_mo_coeff[:,ncore:ncore+ncas].T
    casdm2 = trdm_list.sum(axis = (0,1))
    hcore_deriv = gbci_grad.hcore_generator(mol)
    s1 = gbci_grad.get_ovlp(mol)

    diag_idx = numpy.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    mo_cores = numpy.stack([mo_list[p][:,:ncore] for p in range(num_group)], axis = 0)
    dm_cores = 2 * numpy.einsum('xpc, xqc -> xpq', mo_cores, mo_cores)
    dm_acts = numpy.einsum('pa, xa, qa -> xpq', mo_cas, as_occ, mo_cas)
    dms = dm_cores + dm_acts
    xz = numpy.asarray(xzvec_ao)
    ordm_aos = numpy.einsum('pa,xwab,qb->xwpq', mo_cas, ordm_list, mo_cas, optimize=True)

    max_memory = gbci_grad.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, casdm1)
        de[k] += numpy.einsum('xij,ij->x', h1ao, zvec_ao)
        de[k] += numpy.einsum('xij,pij->x', h1ao, xzvec_ao)
        for x in range(num_group):
            dm_core = dm_cores[x]
            de[k] += numpy.einsum('xij, ij -> x', h1ao, dm_core) * group_prob[x]
        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                                shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de[k] -= numpy.einsum('xijw,ijw->x', eri1, dm2_ao) * 2

            xz_pq_sym = xz[:,p0:p1, q0:q1] + xz[:,q0:q1, p0:p1].transpose(0,2,1)
            xz_p_sym = xz[:,p0:p1, :] + xz[:, :, p0:p1].transpose(0,2,1)
            xz_q_sym = xz[:,q0:q1, :] + xz[:, :, q0:q1].transpose(0,2,1)

            ordm_pq_sym = ordm_aos[:, :, p0:p1, q0:q1]+ ordm_aos[:, :, q0:q1, p0:p1].transpose(0, 1, 3, 2)
            dmet_pq_sym = dmet_core_list[:, :, p0:p1, q0:q1]+ dmet_core_list[:, :, q0:q1, p0:p1].transpose(0, 1, 3, 2)

            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf,-1))
                eri1tmp = eri1tmp.reshape(p1-p0,nf,nao,nao)
                de[k,i] -= 2 * numpy.einsum('ijkl, xlk, xij, x ->', eri1tmp, dm_cores,
                                            dm_cores[:,p0:p1, q0:q1], group_prob, optimize = True)
                de[k,i] += numpy.einsum('ijkl, xjk, xil, x ->', eri1tmp, dm_cores[:,q0:q1,:],
                                        dm_cores[:,p0:p1, :], group_prob, optimize = True)

                de[k,i] -= numpy.einsum('ijkl, xij, xkl ->', eri1tmp, xz_pq_sym, dms, optimize = True)
                de[k,i] -= 2 * numpy.einsum('ijkl, xkl, xij ->', eri1tmp, xz, dms[:,p0:p1, q0:q1], optimize = True)
                de[k,i] += 0.5 * numpy.einsum('ijkl, xil, xjk ->', eri1tmp, xz_p_sym, dms[:,q0:q1,:], optimize = True)
                de[k,i] += 0.5 * numpy.einsum('ijkl, xjl, xik ->', eri1tmp, xz_q_sym, dms[:,p0:p1,:], optimize = True)

                de[k,i] -= 2 * numpy.einsum('ijkl, xwij, xwkl ->', eri1tmp, ordm_pq_sym,
                                            dmet_core_list, optimize = True)
                de[k,i] -= 2 * numpy.einsum('ijkl, xwkl, xwij ->', eri1tmp, ordm_aos, dmet_pq_sym, optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwil, xwkj', eri1tmp, ordm_aos[:, :, p0:p1, :],
                                        dmet_core_list[:, :, :,q0:q1], optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwjl, xwki', eri1tmp, ordm_aos[:, :, q0:q1, :],
                                        dmet_core_list[:, :, :,p0:p1], optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwkj, xwil', eri1tmp, ordm_aos[:,:,:,q0:q1],
                                        dmet_core_list[:,:,p0:p1,:], optimize = True)
                de[k,i] += numpy.einsum('ijkl, xwli, xwjk', eri1tmp, ordm_aos[:,:,:,p0:p1],
                                        dmet_core_list[:,:,q0:q1,:], optimize = True)

                de[k,i] -= numpy.einsum("ijkl,ij,kl",eri1tmp,wtot[p0:p1, q0:q1],dmtot,optimize=True) * 2
                de[k,i] -= numpy.einsum("ijkl,kl,ij",eri1tmp,wtot,dmtot[p0:p1, q0:q1],optimize=True) * 2
                for response_density, reference_density in (
                    (fock_seed_a_ao, dma),
                    (fock_seed_b_ao, dmb),
                ):
                    de[k,i] += numpy.einsum("ijkl,il,jk", eri1tmp, response_density[p0:p1],
                                            reference_density[q0:q1], optimize=True) * 2
                    de[k,i] += numpy.einsum("ijkl,kj,li", eri1tmp, response_density[:, q0:q1],
                                            reference_density[:, p0:p1], optimize=True) * 2

            eri1 = eri1tmp = None

        de[k] -= numpy.einsum("xij,ij->x", s1[:, p0:p1], overlap_ao[p0:p1], optimize=True)
        de[k] -= numpy.einsum("xij,ji->x", s1[:, p0:p1], overlap_ao[:, p0:p1], optimize=True)

    log.timer('GBCI nuclear gradients', *time0)
    return de
