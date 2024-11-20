#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

import ctypes
import numpy
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.gto.eval_gto import _get_intor_and_comp, BLKSIZE
from pyscf.pbc.gto import _pbcintor
from pyscf import __config__

EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-2)

libpbc  = _pbcintor.libpbc
libisdf = lib.load_library('libisdf')

def z2d_InPlace(z):
    '''Convert complex array to double array in-place'''
    assert(z.dtype == numpy.complex128)
    
    fn = getattr(libisdf, "NPz2d_InPlace")
    assert(fn is not None)
    fn(z.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_size_t(z.size))
    z_real = numpy.ndarray(shape=z.shape, dtype=numpy.double, buffer=z)
    return z_real

def _estimate_rcut(cell):
    '''Cutoff raidus, above which each shell decays to a value less than the
    required precsion'''
    log_prec = numpy.log(cell.precision * EXTRA_PREC)
    rcut = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        r[r < 1.] = 1.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        rcut.append(r.max())
    return numpy.array(rcut)

def ISDF_eval_gto(cell, eval_name=None, coords=None, comp=None, kpts=numpy.zeros((1,3)), kpt=None,
             shls_slice=None, non0tab=None, ao_loc=None, cutoff=None,
             out=None, Ls=None, rcut=None):
    r'''Evaluate PBC-AO function value on the given grids,

    Args:
        eval_name : str

            ==========================  =======================
            Function                    Expression
            ==========================  =======================
            "GTOval_sph"                \sum_T exp(ik*T) |AO>
            "GTOval_ip_sph"             nabla \sum_T exp(ik*T) |AO>
            "GTOval_cart"               \sum_T exp(ik*T) |AO>
            "GTOval_ip_cart"            nabla \sum_T exp(ik*T) |AO>
            ==========================  =======================

        atm : int32 ndarray
            libcint integral function argument
        bas : int32 ndarray
            libcint integral function argument
        env : float64 ndarray
            libcint integral function argument

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in cell will be evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`dft.gen_grid.make_mask`
        cutoff : float
            AO values smaller than cutoff will be set to zero. The default
            cutoff threshold is ~1e-22 (defined in gto/grid_ao_drv.h)
        out : ndarray
            If provided, results are written into this array.

    Returns:
        A list of 2D (or 3D) arrays to hold the AO values on grids. 

    WARNING : only support gamma point calculation !!!!

    '''

    if eval_name is None:
        if cell.cart:
            eval_name = 'GTOval_cart_deriv%d' % 0
        else:
            eval_name = 'GTOval_sph_deriv%d' % 0

    if eval_name[:3] == 'PBC':  # PBCGTOval_xxx
        eval_name, comp = _get_intor_and_comp(cell, eval_name[3:], comp)
    else:
        eval_name, comp = _get_intor_and_comp(cell, eval_name, comp)
    eval_name = 'PBC' + eval_name

    assert comp == 1

    atm = numpy.asarray(cell._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(cell._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(cell._env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    if kpts is None:
        if kpt is not None:
            raise RuntimeError('kpt should be a list of k-points')
            kpts_lst = numpy.reshape(kpt, (1,3))
        else:
            kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    ngrids = len(coords)

    assert kpts_lst.shape[0] == 1

    # print("kpts_lst = ", kpts_lst)

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE, nbas),
                              dtype=numpy.uint8)
# non0tab stores the number of images to be summed in real space.
# Initializing it to 255 means all images should be included
        non0tab[:] = 0xff

    if ao_loc is None:
        ao_loc = moleintor.make_loc(bas, eval_name)
    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]

    if out is None:
        out = numpy.empty((nkpts,comp,nao,ngrids), dtype=numpy.complex128)  # NOTE THE definition of the shape!
    else:
        # print("out is given")
        out = numpy.ndarray((nkpts,comp,nao,ngrids), dtype=numpy.complex128,
                             buffer=out)
    coords = numpy.asarray(coords, order='F')

    # For atoms near the boundary of the cell, it is necessary (even in low-
    # dimensional systems) to include lattice translations in all 3 dimensions.
    if Ls is None:
        if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
            Ls = cell.get_lattice_Ls(dimension=cell.dimension)
        else:
            Ls = cell.get_lattice_Ls(dimension=3)
        Ls = Ls[numpy.argsort(lib.norm(Ls, axis=1))]
    expLk = numpy.exp(1j * numpy.asarray(numpy.dot(Ls, kpts_lst.T), order='C'))
    if rcut is None:
        rcut = _estimate_rcut(cell)

    with cell.with_integral_screen(cutoff):
        drv = getattr(libpbc, eval_name)
        drv(ctypes.c_int(ngrids),
            (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
            Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(Ls)),
            expLk.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts),
            out.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            rcut.ctypes.data_as(ctypes.c_void_p),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p))
  
    out = out[0]
    out = z2d_InPlace(out)
    return out[0]


