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

############ sys module ############

import numpy, scipy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
libisdf = lib.load_library('libisdf')

############ isdf utils ############

from pyscf.isdf.isdf_tools_local import aoR_Holder
from pyscf.isdf.isdf_jk          import _benchmark_time 
from pyscf.isdf.isdf_local_k     import PBC_ISDF_Info_Quad_K


def _aoR_full_col(mydf):
    '''
    return aoR[:, :ngrid_prim] for the supercell system
    '''

    assert isinstance(mydf, PBC_ISDF_Info_Quad_K)

    fn_pack = getattr(libisdf, "_Pack_Matrix_SparseRow_DenseCol", None)
    assert fn_pack is not None
    
    prim_cell  = mydf.primCell
    prim_mesh  = prim_cell.mesh
    prim_ngrid = np.prod(prim_mesh)
    prim_natm  = mydf.natmPrim
    
    assert len(mydf.aoR) == prim_natm
    
    res = np.zeros((mydf.nao, prim_ngrid), dtype=np.float64)
    
    for i in range(prim_natm):
        aoR_i               = mydf.aoR[i]
        ao_involved_i       = aoR_i.ao_involved
        nao_i               = aoR_i.aoR.shape[0]
        global_grid_begin_i = aoR_i.global_gridID_begin
        ngrid_i             = aoR_i.aoR.shape[1]
                
        fn_pack(
            res.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(res.shape[0]),
            ctypes.c_int(res.shape[1]),
            aoR_i.aoR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_i),
            ctypes.c_int(ngrid_i),
            ao_involved_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(global_grid_begin_i),
            ctypes.c_int(global_grid_begin_i+ngrid_i)
        )
                
    return res

def _aoRg_full_col(mydf):
    '''
    return aoR[:, :ngrid_prim] for the supercell system
    '''

    assert isinstance(mydf, PBC_ISDF_Info_Quad_K)

    fn_pack = getattr(libisdf, "_Pack_Matrix_SparseRow_DenseCol", None)
    assert fn_pack is not None
    
    prim_cell  = mydf.primCell
    prim_mesh  = prim_cell.mesh
    prim_ngrid = np.prod(prim_mesh)
    prim_natm  = mydf.natmPrim
    prim_nIP   = mydf.nIP_Prim
    
    assert len(mydf.aoR) == prim_natm
    
    res = np.zeros((mydf.nao, prim_nIP), dtype=np.float64)
    
    for i in range(mydf.natmPrim):
        aoRg_i            = mydf.aoRg[i]
        ao_involved_i     = aoRg_i.ao_involved
        nao_i             = aoRg_i.aoR.shape[0]
        global_IP_begin_i = aoRg_i.global_gridID_begin
        nIP_i             = aoRg_i.aoR.shape[1]
                
        fn_pack(
            res.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(res.shape[0]),
            ctypes.c_int(res.shape[1]),
            aoRg_i.aoR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_i),
            ctypes.c_int(nIP_i),
            ao_involved_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(global_IP_begin_i),
            ctypes.c_int(global_IP_begin_i+nIP_i)
        )
                
    return res

######## copy from libdmet ########

def eri_restore(eri, symmetry, nemb):
    """
    Restore eri with given permutation symmetry.
    """
    spin_pair = eri.shape[0]
    if spin_pair == 1:
        eri_res = ao2mo.restore(symmetry, eri[0].real, nemb)
    else:
        if symmetry == 4:
            nemb_pair = nemb*(nemb+1) // 2
            if eri.size == spin_pair * nemb_pair * nemb_pair:
                eri_res = eri.real.reshape(spin_pair, nemb_pair, nemb_pair)
            else:
                eri_res = np.empty((spin_pair, nemb_pair, nemb_pair))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        elif symmetry == 1:
            if eri.size == spin_pair * nemb**4:
                eri_res = eri.real.reshape(spin_pair, nemb, nemb, nemb, nemb)
            else:
                eri_res = np.empty((spin_pair, nemb, nemb, nemb, nemb))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        else:
            raise ValueError("Spin unrestricted ERI does not support 8-fold symmetry.")
    eri_res = np.asarray(eri_res, order='C')
    return eri_res

def get_emb_eri_isdf(mydf, C_ao_emb:np.ndarray=None, symmetry=4):
    
    ''' 
    get eri for embedding system
    '''
    
    #### preprocess #### 
    
    assert isinstance(mydf, PBC_ISDF_Info_Quad_K)
    assert not mydf.direct
    
    if C_ao_emb.ndim == 2:
        C_ao_emb = C_ao_emb.reshape(1, *C_ao_emb.shape)
    assert C_ao_emb.ndim  == 3
    assert C_ao_emb.dtype == np.float64  ## supercell basis
    
    nspin, nao_full, nemb = C_ao_emb.shape
    
    print("nspin    = ", nspin)
    print("nao_full = ", nao_full)
    print("nemb     = ", nemb)
    
    supercell = mydf.cell
    print("supercell.nao = ", supercell.nao)
    assert supercell.nao == nao_full
    
    ngrid      = mydf.ngrids
    vol        = supercell.vol
    mesh_prim  = mydf.primCell.mesh
    ngrid_prim = np.prod(mesh_prim)
    nao_prim   = mydf.nao_prim
    nIP_prim   = mydf.nIP_Prim
    kmesh      = mydf.kmesh
    nkpts      = np.prod(kmesh)
    nIP        = mydf.naux
    
    with_robust_fitting = mydf.with_robust_fitting
    
    #eri = np.zeros((nspin*(nspin+1)//2, nemb, nemb, nemb, nemb), dtype=np.float64) ## the ordering of spin is aa, bb, ab
    eri = np.zeros((nspin*(nspin+1)//2, nemb**2, nemb**2), dtype=np.float64) ## the ordering of spin is aa, bb, ab
    
    ### emb values on grid and IPs ###
    
    emb_R = []
    emb_Rg= []
    for i in range(nspin):
        emb_R.append([])
        emb_Rg.append([])
    
    if with_robust_fitting:
        aoR_fullcol  = _aoR_full_col(mydf)
        assert aoR_fullcol.shape  == (nao_full, ngrid_prim)
    aoRg_fullcol = _aoRg_full_col(mydf)
    assert aoRg_fullcol.shape == (nao_full, nIP_prim)
    
    aoR_tmp  = np.zeros_like(aoR_fullcol)
    aoRg_tmp = np.zeros_like(aoRg_fullcol)
    
    for kx in range(kmesh[0]):
        for ky in range(kmesh[1]):
            for kz in range(kmesh[2]):
                                
                for ix in range(kmesh[0]):
                    for iy in range(kmesh[1]):
                        for iz in range(kmesh[2]):
                            
                            ILOC  = ix*kmesh[1]*kmesh[2] + iy*kmesh[2] + iz
                            ix_   = (ix + kx) % kmesh[0]
                            iy_   = (iy + ky) % kmesh[1]
                            iz_   = (iz + kz) % kmesh[2]
                            ILOC_ = ix_*kmesh[1]*kmesh[2] + iy_*kmesh[2] + iz_
                            
                            if with_robust_fitting:
                                aoR_tmp[ILOC_*nao_prim:(ILOC_+1)*nao_prim,:]  = aoR_fullcol[ILOC*nao_prim:(ILOC+1)*nao_prim,:]
                            aoRg_tmp[ILOC_*nao_prim:(ILOC_+1)*nao_prim,:] = aoRg_fullcol[ILOC*nao_prim:(ILOC+1)*nao_prim,:]
                            
                for i in range(nspin):
                    if with_robust_fitting:
                        emb_R[i].append(np.dot(C_ao_emb[i].T, aoR_tmp))
                    emb_Rg[i].append(np.dot(C_ao_emb[i].T, aoRg_tmp))
                            
    
    ### V_R term ###
    
    #V_R = mydf.V_R
    #assert V_R.shape == (nIP_prim, ngrid)
        
    tmp_V = np.zeros((nspin, nIP, nemb*nemb), dtype=np.float64)
    
    def _construct_tmp_V_W(Is_V=False):

        tmp_V.ravel()[:] = 0.0

        if Is_V:
            V = mydf.V_R
            ngrid_per_box = ngrid_prim
            _emb_R = emb_R
        else:
            V = mydf.W
            ngrid_per_box = nIP_prim
            _emb_R = emb_Rg
        
        for kx in range(kmesh[0]):
            for ky in range(kmesh[1]):
                for kz in range(kmesh[2]):
                    
                    ILOC = kx*kmesh[1]*kmesh[2] + ky*kmesh[2] + kz
                    
                    for i in range(nspin):
                        
                        _emb_pair = np.einsum('iP,jP->ijP', _emb_R[i][ILOC], _emb_R[i][ILOC])
                        _emb_pair = _emb_pair.reshape(nemb*nemb, ngrid_per_box)
                        # _tmp_V    = lib.ddot(V[:,ILOC*ngrid_per_box:(ILOC+1)*ngrid_per_box],_emb_pair.T)

                        ## another pass to account for the transposition ##
                        
                        for ix in range(kmesh[0]):
                            for iy in range(kmesh[1]):
                                for iz in range(kmesh[2]):
                                    
                                    ix_ = (kx-ix+kmesh[0]) % kmesh[0]
                                    iy_ = (ky-iy+kmesh[1]) % kmesh[1]
                                    iz_ = (kz-iz+kmesh[2]) % kmesh[2]
                                    
                                    ILOC_ = ix_*kmesh[1]*kmesh[2] + iy_*kmesh[2] + iz_
                                    ILOC  = ix *kmesh[1]*kmesh[2] + iy *kmesh[2] + iz
                                    
                                    lib.ddot(
                                        a=V[:,ILOC_*ngrid_per_box:(ILOC_+1)*ngrid_per_box],
                                        b=_emb_pair.T,
                                        alpha=1.0,
                                        c=tmp_V[i][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                        beta=1.0)
    
    def _the_last_pass(plus):
        
        if plus:
            alpha = 1
        else:
            alpha =-1
            
        for ix in range(kmesh[0]):
            for iy in range(kmesh[1]):
                for iz in range(kmesh[2]):
                
                    ILOC = ix*kmesh[1]*kmesh[2] + iy*kmesh[2] + iz
                
                    if nspin == 1:

                        emb_pair_Rg = np.einsum('iP,jP->ijP', emb_Rg[0][ILOC], emb_Rg[0][ILOC])
                        emb_pair_Rg = emb_pair_Rg.reshape(nemb*nemb, nIP_prim)

                        lib.ddot(
                            a = emb_pair_Rg,
                            b = tmp_V[0][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                            alpha = alpha,
                            c     = eri[0],
                            beta  = 1
                        )
                    else:
                        if nspin == 2:

                            emb_pair_Rg_alpha = np.einsum('iP,jP->ijP', emb_Rg[0][ILOC], emb_Rg[0][ILOC])
                            emb_pair_Rg_beta  = np.einsum('iP,jP->ijP', emb_Rg[1][ILOC], emb_Rg[1][ILOC])

                            lib.ddot(
                                a = emb_pair_Rg_alpha,
                                b = tmp_V[0][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                alpha = alpha,
                                c     = eri[0],
                                beta  = 1
                            )

                            lib.ddot(
                                a = emb_pair_Rg_beta,
                                b = tmp_V[1][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                alpha = alpha,
                                c     = eri[1],
                                beta  = 1
                            )

                            lib.ddot(
                                a = emb_pair_Rg_alpha,
                                b = tmp_V[1][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                alpha = alpha,
                                c     = eri[2],
                                beta  = 1
                            )

                        else:
                            raise ValueError("nspin > 2 is not supported")
    
    if with_robust_fitting:
        
        _construct_tmp_V_W(True)
        _the_last_pass(plus=True)
        nspinpair = nspin*(nspin+1)//2
        
        for i in range(nspinpair):
            eri[i] += eri[i].T
    
    ### W term ###
    
    _construct_tmp_V_W(False)
    if with_robust_fitting:
        _the_last_pass(plus=False)
    else:
        _the_last_pass(plus=True)
    
    #### post process ####
    
    # reshape the eri 
    
    eri = eri.reshape(nspin*(nspin+1)//2, nemb, nemb, nemb, nemb)
    eri = eri_restore(eri, symmetry, nemb)
    
    return eri * ngrid / vol
    
    
if __name__ == "__main__":

    from isdf_tools_cell import build_supercell, build_supercell_with_partition
    C = 25
    
    verbose = 10
    import pyscf.pbc.gto as pbcgto
    
    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 
    
    KE_CUTOFF = 70
    basis = 'gth-szv'
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], basis=basis, ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    # prim_partition = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # prim_partition = [[0,1,2,3,4,5,6,7]]
    prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    
    Ls = [1, 2, 2]
    kpts = prim_cell.make_kpts(Ls)
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, 
                                                     #pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    
    pbc_isdf_info = PBC_ISDF_Info_Quad_K(prim_cell, kmesh=Ls, with_robust_fitting=True, aoR_cutoff=1e-8, 
                                         # direct=True, 
                                         direct=False, 
                                         rela_cutoff_QRCP=1e-4,
                                         limited_memory=True, build_K_bunchsize=32)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=prim_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    pbc_isdf_info.verbose = 10    
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    # print("grid_segment = ", pbc_isdf_info.grid_segment)
    
    from pyscf.pbc import scf

    mf = scf.KRHF(prim_cell, kpts)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 1e-7
    
    mf.kernel()

    nao_full = pbc_isdf_info.cell.nao
    nao_emb  = nao_full // 5
    C_ao_emb = np.random.rand(nao_full, nao_emb)
    
    eri_emb = get_emb_eri_isdf(pbc_isdf_info, C_ao_emb, symmetry=4)
    
    supercell = pbc_isdf_info.cell
    
    from pyscf.isdf.isdf_local import PBC_ISDF_Info_Quad
    
    pbc_isdf_info2 = PBC_ISDF_Info_Quad(supercell, with_robust_fitting=True, 
                                        aoR_cutoff=1e-8, 
                                        direct=False, 
                                        # direct=True, 
                                        limited_memory=True, build_K_bunchsize=32,
                                        use_occ_RI_K=False, rela_cutoff_QRCP=1e-4)
        
    pbc_isdf_info2.build_IP_local(c=C, m=5, group=group_partition)
    pbc_isdf_info2.build_auxiliary_Coulomb()    
    
    eri_emb_benchmark = pbc_isdf_info2.ao2mo(C_ao_emb)
    
    assert eri_emb.shape == eri_emb_benchmark.shape
    
    diff = np.linalg.norm(eri_emb - eri_emb_benchmark)
    print("diff     = ", diff)
    max_diff = np.max(np.abs(eri_emb - eri_emb_benchmark))
    print("max_diff = ", max_diff)
    
    # print("eri_emb.shape = ", eri_emb.shape)
    # print("eri_emb           = ", eri_emb[0,0],eri_emb[0,1])
    # print("eri_emb_benchmark = ", eri_emb_benchmark[0,0], eri_emb_benchmark[0,1])
    # for i in range(eri_emb.shape[0]):
    #     for j in range(eri_emb.shape[1]):
    #         print(eri_emb[i,j], eri_emb_benchmark[i,j], eri_emb[i,j]/eri_emb_benchmark[i,j])
