#/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

# This file can be merged into pyscf.dft.numint.py

import numpy
from pyscf import lib
from pyscf.dft.gen_grid import NBINS

from pyscf.dft.numint import _dot_ao_ao_sparse,_scale_ao_sparse,_tau_dot_sparse

MGGA_DENSITY_LAPL = False # just copy from pyscf.dft.numint.py

def nr_uks_fxc_sf(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,rho0=None,
                  vxc=None, fxc=None, extype=0, max_memory=2000, verbose=None):
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)

    xctype = ni._xc_type(xc_code)

    dmb2a,dma2b = dms
    nao = dmb2a.shape[-1]
    make_rhob2a, nset = ni._gen_rho_evaluator(mol, dmb2a, hermi, False, grids)[:2]
    make_rhoa2b       = ni._gen_rho_evaluator(mol, dma2b, hermi, False, grids)[0]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[...,p0:p1]
            for i in range(nset):
                rho1b2a = make_rhob2a(i, ao, mask, xctype)
                rho1a2b = make_rhoa2b(i, ao, mask, xctype)
                if xctype == 'LDA':
                    # *2.0 becausue kernel xx,yy parts.
                    wvA_b2a = rho1b2a * _fxc[0,0]*2.0 *weight
                    wvA_a2b = rho1a2b * _fxc[0,0]*2.0 *weight
                    wvB_b2a = rho1a2b.conj() * _fxc[0,0]*2.0 *weight
                    wvB_a2b = rho1b2a.conj() * _fxc[0,0]*2.0 *weight
                    wv = numpy.array((wvA_b2a,wvA_a2b,wvB_b2a,wvB_a2b))
                else:
                    # *2.0 becausue kernel xx,yy parts.
                    wvA_b2a = lib.einsum('bg,abg->ag',rho1b2a,_fxc*2.0)*weight
                    wvA_a2b = lib.einsum('bg,abg->ag',rho1a2b,_fxc*2.0)*weight
                    wvB_b2a = lib.einsum('bg,abg->ag',rho1a2b.conj(),_fxc*2.0)*weight
                    wvB_a2b = lib.einsum('bg,abg->ag',rho1b2a.conj(),_fxc*2.0)*weight
                    wv = numpy.array((wvA_b2a,wvA_a2b,wvB_b2a,wvB_a2b))
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((4,nset,nao,nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[0,i])
            _dot_ao_ao_sparse(ao, ao, wv[1], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[1,i])
            _dot_ao_ao_sparse(ao, ao, wv[2], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[2,i])
            _dot_ao_ao_sparse(ao, ao, wv[3], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[3,i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            aow = _scale_ao_sparse(ao, wv[0], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[0,i])
            aow = _scale_ao_sparse(ao, wv[1], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[1,i])
            aow = _scale_ao_sparse(ao, wv[2], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[2,i])
            aow = _scale_ao_sparse(ao, wv[3], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[3,i])

        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(4,nset,nao,nao)

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wv[:,4] *= .5
            aow = _scale_ao_sparse(ao[:4], wv[0,:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[0,i])
            _tau_dot_sparse(ao, ao, wv[0,4], nbins, mask, pair_mask, ao_loc, out=v1[0,i])

            aow = _scale_ao_sparse(ao[:4], wv[1,:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[1,i])
            _tau_dot_sparse(ao, ao, wv[1,4], nbins, mask, pair_mask, ao_loc, out=v1[1,i])

            aow = _scale_ao_sparse(ao[:4], wv[2,:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[2,i])
            _tau_dot_sparse(ao, ao, wv[2,4], nbins, mask, pair_mask, ao_loc, out=v1[2,i])

            aow = _scale_ao_sparse(ao[:4], wv[3,:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[3,i])
            _tau_dot_sparse(ao, ao, wv[3,4], nbins, mask, pair_mask, ao_loc, out=v1[3,i])

        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(4,nset,nao,nao)
        vmat += v1

    if isinstance(dmb2a, numpy.ndarray) and dmb2a.ndim == 2:
        vmat = vmat[:,0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat

def nr_uks_fxc_sf_tda(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,rho0=None,
                      vxc=None, fxc=None, extype=0, max_memory=2000, verbose=None):
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if hermi != 1 and dtype != numpy.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)

    nao = dms.shape[-1]
    make_rhosf, nset = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)[:2]

    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[...,p0:p1]
            for i in range(nset):
                rho1sf = make_rhosf(i, ao, mask, xctype)
                if xctype == 'LDA':
                    # *2.0 becausue kernel xx,yy parts.
                    wv = rho1sf * _fxc[0,0]*2.0 *weight
                else:
                    # *2.0 becausue kernel xx,yy parts.
                    wv = lib.einsum('bg,abg->ag',rho1sf,_fxc*2.0)*weight
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((nset,nao,nao))
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])
    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            aow = _scale_ao_sparse(ao, wv, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])

        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(nset,nao,nao)

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5
            wv[4] *= .5
            aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, ao_loc,
                              hermi=0, out=vmat[i])
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])

        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(nset,nao,nao)
        vmat += v1

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vmat = vmat[:,0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat
