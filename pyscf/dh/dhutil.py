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
#
# Authors:
#          Zhenyu Zhu <ajz34@outlook.com>
#          Shirong Wang <srwang20@fudan.edu.cn>
#

import tempfile

import h5py
from pyscf import lib, gto
from pyscf.lib.numpy_helper import HERMITIAN
from pyscf.ao2mo.outcore import balance_partition

import numpy as np
from time import time, process_time
from functools import wraps



class HybridDict(dict):
    """
    HybridDict: Inherited dictionary class

    A dictionary specialized to store data both in memory and in disk.
    """
    def __init__(self, chkfile_name=None, dir=None, **kwargs):
        super(HybridDict, self).__init__(**kwargs)
        if dir is None:
            dir = lib.param.TMPDIR
        if chkfile_name is None:
            self._chkfile = tempfile.NamedTemporaryFile(dir=dir)
            chkfile_name = self._chkfile.name
        self.chkfile_name = chkfile_name
        self.chkfile = h5py.File(self.chkfile_name, "r+")
        self._external = []

    def create(self, name, data=None, incore=True, shape=None, dtype=None, **kwargs):
        # create logic check
        if data is None and shape is None:
            raise ValueError("Provide either data or shape!")
        if data is not None and shape is not None:
            raise ValueError("Data and shape shouldn't be provided together!")
        if name in self:
            try:  # don't create a new space if tensor already exists
                # data provided or shape not aligned is not considered here
                if shape and isinstance(self[name], h5py.Dataset) == (not incore):
                    if self[name].shape == shape:
                        self[name][:] = 0
                        return self.get(name)
            except (ValueError, AttributeError):
                # ValueError -- in h5py.h5d.create: Unable to create dataset (name already exists)
                # AttributeError -- [certain other type] object has no attribute 'shape'
                pass
            self.delete(name)
        dtype = dtype if dtype is not None else np.float64
        if not incore:
            self.chkfile.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)
            self.setdefault(name, self.chkfile[name])
        elif data is not None:
            self.setdefault(name, data)
        elif data is None and shape is not None:
            self.setdefault(name, np.zeros(shape=shape, dtype=dtype))
        else:
            raise ValueError("Could not handle create!")
        return self.get(name)

    def consume(self, other):
        if isinstance(other, HybridDict):
            other._consumed = True
            if hasattr(other, '_chkfile'):
                self._external.append((other.chkfile, other._chkfile))
        self.update(dict(other))
        return self

    def delete(self, key):
        val = self.pop(key)
        if isinstance(val, h5py.Dataset):
            try:
                del self.chkfile[key]
            except KeyError:  # h5py.h5g.GroupID.unlink: Couldn't delete link
                # another key maps to the same h5py dataset value, and this value has been deleted
                pass

    def load(self, key):
        return np.asarray(self.get(key))

    def close(self):
        for chkfile, tmpfile in self._external:
            try:
                chkfile.close()
            except Exception:
                pass
            try:
                tmpfile.close()
            except Exception:
                pass
        self._external.clear()
        if hasattr(self, 'chkfile'):
            try:
                self.chkfile.close()
            except Exception:
                pass
        if hasattr(self, '_chkfile'):
            try:
                self._chkfile.close()
            except Exception:
                pass

    def __del__(self):
        if getattr(self, '_consumed', False):
            return
        try:
            self.close()
        except Exception:
            pass


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0, p0 = time(), process_time()
        result = f(*args, **kwargs)
        t1, p1 = time(), process_time()
        with open("tmp_timing.log", "a") as log:
            log.write(" {0:50s}, Wall: {1:10.3f} s, CPU: {2:10.3f} s, ratio {3:7.1f}%\n"
                  .format(f.__qualname__, t1-t0, p1-p0, (p1-p0)/(t1-t0) * 100))
        return result
    return wrapper


def gen_batch(minval, maxval, nbatch):
    return [slice(i, (i + nbatch) if i + nbatch < maxval else maxval) for i in range(minval, maxval, nbatch)]


def gen_shl_batch(mol, blksize, start_id=0, stop_id=None):
    ao_loc = mol.ao_loc
    lst = balance_partition(ao_loc, blksize, start_id, stop_id)
    return [(t[0], t[1], ao_loc[t[0]], ao_loc[t[1]]) for t in lst]


def available_memory(max_memory):
    from pyscf import lib
    return max(max_memory - lib.current_memory()[0], 500)


def calc_batch_size(unit_flop, mem_avail, pre_flop=0):
    # mem_avail: in MB
    max_memory = 0.8 * mem_avail - pre_flop * 8 / 1024 ** 2
    batch_size = int(max(max_memory // (unit_flop * 8 / 1024 ** 2), 1))
    return batch_size


def get_rho_from_dm_gga(ni, mol, grids, dm):
    dm_shape = dm.shape
    dm = dm.reshape((-1, dm_shape[-2], dm_shape[-1]))
    nset, nao, _ = dm.shape
    rho = np.empty((nset, 4, grids.weights.size))
    ip = 0
    for ao, mask, weight, _ in ni.block_loop(mol, grids, nao, deriv=1):
        ngrid = weight.size
        for i in range(nset):
            rho[i, :, ip:ip+ngrid] = ni.eval_rho(mol, ao, dm[i], mask, "GGA", hermi=1)
        ip += ngrid
    rho = rho.reshape(tuple(dm_shape[:-2]) + rho.shape[-2:])
    return rho


def tot_size(*args):
    size = 0
    for i in args:
        if isinstance(i, np.ndarray):
            size += i.size
        else:
            size += tot_size(*i)
    return size


def restricted_biorthogonalize(t_ijab, c_os, c_ss):
    coef_0 = c_os + c_ss
    coef_1 = -c_ss
    if abs(coef_1) < 1e-7:
        return coef_0 * t_ijab
    t_shape = t_ijab.shape
    t_ijab_flat = t_ijab.reshape(-1, t_ijab.shape[-2], t_ijab.shape[-1])
    res = lib.transpose(t_ijab_flat, axes=(0, 2, 1)).reshape(t_shape)
    t_ijab = t_ijab_flat.reshape(t_shape)
    res *= coef_1
    res += coef_0 * t_ijab
    return res


def hermi_sum_last2dim(tsr, inplace=True, hermi=HERMITIAN):
    # shameless call lib.hermi_sum, just for a tensor wrapper
    tsr_shape = tsr.shape
    tsr = tsr.reshape(-1, tsr.shape[-2], tsr.shape[-1])
    res = lib.hermi_sum(tsr, axes=(0, 2, 1), hermi=hermi, inplace=inplace)
    tsr = tsr.reshape(tsr_shape)
    res = res.reshape(tsr_shape)
    return res


def as_scanner_grad(mf: lib.StreamObject, consequent_dm_guess=True):
    # A very, very, very, very, very strange way to define gradient scanner function
    # Important issue need to be stressed five times!
    # TODO avoid directly calling __init__ to initialize a DFDH object

    if isinstance(mf, lib.GradScanner):
        return mf

    class Scanner(mf.__class__, lib.GradScanner):

        e_tot = 0

        def __init__(self, g):
            # this class is intended not to call lib.GradScanner initializer
            # which envokes mf.base, but dfdh does not use .base currently
            self.__dict__.update(g.__dict__)

        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            self.mol = mol.build()

            # If second integration grids are created for RKS and UKS
            # gradients
            if getattr(self, 'grids', None):
                self.grids.reset(mol).build()
                self.grids_cpks.reset(mol).build()

            self.mf_s.mol = self.mf_n.mol = self.mol

            mf.__class__.__init__(self,
                mol,
                xc=self.xc_dh,
                auxbasis_jk=self.auxbasis_jk,
                auxbasis_ri=self.auxbasis_ri,
                grids=self.grids,
                grids_cpks=self.grids_cpks,
                unrestricted=self.unrestricted)

            dm = None
            if consequent_dm_guess:
                dm = self.D
            if dm is None:
                dm = None

            self.tensors = HybridDict()
            self.kernel(dm=dm, **kwargs)
            return self.e_tot, self.de
    return Scanner(mf)


