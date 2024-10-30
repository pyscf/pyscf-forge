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

import numpy as np

from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.gto.mole import *

from   pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast
import pyscf.isdf.isdf_local   as isdf_local
import pyscf.isdf.isdf_local_k as isdf_local_k
from   pyscf.isdf.isdf_tools_local import flatten_aoR_holder

###############################################################

# debug code #

def dump_attributes(mydf, attr_lst:list[str], dtype=np.int32, filename:str=None):
    
    res = []
    
    for attr in attr_lst:
        assert hasattr(mydf, attr)
        tmp = getattr(mydf, attr)
        if isinstance(tmp, list):
            if all([isinstance(x, np.ndarray) for x in tmp]):
                tmp = np.concatenate([x.ravel() for x in tmp])
            else:
                tmp = np.asarray(tmp, dtype=dtype)
        else:
            tmp = np.asarray(tmp, dtype=dtype)
        res.append(tmp.flatten().astype(dtype))
    
    res = np.concatenate(res)
    print("rank = ", rank, res.shape)
    res.tofile(filename)

def dump_aoR(mydf, filename:str=None):
    
    res_int   = []
    res_float = []
    
    for attr in ["aoR", "aoR1", "aoRg", "aoRg"]:
        if hasattr(mydf, attr):
            tmp = getattr(mydf, attr)
            if tmp is None:
                print("%s is None" % (attr))
                continue
            tmp1, tmp2 = flatten_aoR_holder(tmp)
            res_int.append(tmp1)
            res_float.append(tmp2)
    
    res_int   = np.concatenate(res_int)
    res_float = np.concatenate(res_float)
    
    print("rank = ", rank, res_int.shape, res_float.shape)
    res_int.tofile(filename   + "_int.dat")
    res_float.tofile(filename + "_float.dat")
    

############## MPI version of PBC_ISDF_Info_Quad ##############

class PBC_ISDF_Info_Quad_MPI(isdf_local.PBC_ISDF_Info_Quad):
    ''' Interpolative separable density fitting (ISDF) for periodic systems with MPI.
    
    The locality is explored! 
    
    k-point sampling is not currently supported!
    
    '''

    # Quad stands for quadratic scaling
    
    def __init__(self, mol:Cell, 
                 kmesh             = None,
                 verbose           = None,
                 rela_cutoff_QRCP  = None,
                 aoR_cutoff        = 1e-8,
                 limited_memory    = False,
                 build_K_bunchsize = None):
        
        super().__init__(mol, True, kmesh, verbose, rela_cutoff_QRCP, aoR_cutoff, True, 
                         use_occ_RI_K      = False,
                         limited_memory    = limited_memory,
                         build_K_bunchsize = build_K_bunchsize)
        self.use_mpi = True
        assert self.use_aft_ao == False
    
    dump_attributes = dump_attributes
    dump_aoR = dump_aoR

###############################################################

############## MPI version of PBC_ISDF_Info_Quad_K ##############

class PBC_ISDF_Info_Quad_K_MPI(isdf_local_k.PBC_ISDF_Info_Quad_K):
    ''' Interpolative separable density fitting (ISDF) for periodic systems with MPI.
    
    The locality is explored! 
    
    k-point sampling is not currently supported!
    
    '''

    # Quad stands for quadratic scaling
    
    def __init__(self, mol:Cell, 
                 kmesh             = None,
                 verbose           = None,
                 rela_cutoff_QRCP  = None,
                 aoR_cutoff        = 1e-8,
                 limited_memory    = False,
                 build_K_bunchsize = None):
        
        super().__init__(mol, True, kmesh, verbose, rela_cutoff_QRCP, aoR_cutoff, True, 
                         # use_occ_RI_K      = False,
                         limited_memory    = limited_memory,
                         build_K_bunchsize = build_K_bunchsize)
        self.use_mpi = True
        assert self.use_aft_ao == False

    dump_attributes = dump_attributes
    dump_aoR = dump_aoR

#################################################################

if __name__ == '__main__':

    C = 15
    from pyscf.lib.parameters import BOHR
    from isdf_tools_cell import build_supercell, build_supercell_with_partition
    
    verbose = 6
    if rank != 0:
        verbose = 0
    
    prim_a = np.array(
                    [[14.572056092/2, 0.000000000, 0.000000000],
                     [0.000000000, 14.572056092/2, 0.000000000],
                     [0.000000000, 0.000000000,  6.010273939],]) * BOHR
    atm = [
['Cu1',	(1.927800,	1.927800,	1.590250)],
['O1',	(1.927800,	0.000000,	1.590250)],
['O1',	(0.000000,	1.927800,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
    ]
    from pyscf.gto.basis import parse_nwchem
    fbas="basis2.dat"  ## NOTE: you should copy it from examples/isdf to run this scripts
    atms = ['O', 'Cu', "Ca"]
    basis = {atm:parse_nwchem.load(fbas, atm) for atm in atms}
    pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
    ke_cutoff = 128 
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    KE_CUTOFF = 128
        
    prim_mesh = prim_cell.mesh    
    prim_partition = [[0], [1], [2], [3]]    
    
    Ls = [2, 2, 1]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    if rank == 0:
        print("group_partition = ", group_partition)
    
    pbc_isdf_info = PBC_ISDF_Info_Quad_MPI(cell, aoR_cutoff=1e-8, verbose=verbose, limited_memory=True, build_K_bunchsize=16)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition)
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    from pyscf.pbc import scf

    if comm_size > 1:
        comm.Barrier()

    mf = scf.RHF(cell)
    mf = scf.addons.smearing_(mf, sigma=0.2, method='fermi')
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 0.0
    
    dm = mf.init_guess_by_atom()
    
    if comm_size > 1:
        dm = bcast(dm, root=0)
    
    mf.kernel(dm)
    
    comm.Barrier()