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

import numpy
import numpy as np
import ctypes

############ pyscf module ############

import pyscf
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
import pyscf.pbc.gto as pbcgto
from pyscf.cc.rccsd import _ChemistsERIs, RCCSD 
libpbc = lib.load_library('libpbc')

############ isdf utils ############

from   isdf_jk         import _benchmark_time
import isdf_local      as ISDF
from   isdf_tools_cell import build_supercell, build_supercell_with_partition
from   isdf_ao2mo      import LS_THC, LS_THC_eri

####################################

### post-HF with ISDF ERIs (NOT THC-POSTHF!)

####################################

############ subroutines ---- deal with CC ############

def _make_isdf_eris_incore(mycc, my_isdf:ISDF.PBC_ISDF_Info_Quad, mo_coeff=None):
    
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    eri1 = my_isdf.ao2mo(mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    
    cput1 = (logger.process_clock(), logger.perf_counter())
    
    _benchmark_time(cput0, cput1, "CCSD integral transformation", my_isdf)
    
    return eris

def RCCSD_isdf(mf, frozen=0, mo_coeff=None, mo_occ=None, run=True, cc2=False):
    mycc = RCCSD(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    mycc.cc2 = cc2
    # eris = mycc.ao2mo(mo_coeff)
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    eris_ccsd = _make_isdf_eris_incore(mycc, mf.with_df, mo_coeff=mo_coeff)
    # mycc.eris = eris
    if run:
        mycc.kernel(eris=eris_ccsd)
    return mycc, eris_ccsd

if __name__ == '__main__':

    for c in [15]:
        for N in [1]:

            print("Testing c = ", c, "N = ", N, "...")

            cell   = pbcgto.Cell()
            boxlen = 3.5668
            cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
            
            cell.atom = [
                ['C', (0.     , 0.     , 0.    )],
                ['C', (0.8917 , 0.8917 , 0.8917)],
                ['C', (1.7834 , 1.7834 , 0.    )],
                ['C', (2.6751 , 2.6751 , 0.8917)],
                ['C', (1.7834 , 0.     , 1.7834)],
                ['C', (2.6751 , 0.8917 , 2.6751)],
                ['C', (0.     , 1.7834 , 1.7834)],
                ['C', (0.8917 , 2.6751 , 2.6751)],
            ] 

            cell.basis   = 'gth-szv'
            cell.pseudo  = 'gth-pade'
            cell.verbose = 10
            cell.ke_cutoff = 128
            cell.max_memory = 800  # 800 Mb
            cell.precision  = 1e-8  # integral precision
            cell.use_particle_mesh_ewald = True
            
            verbose = 10
            
            prim_cell = build_supercell(cell.atom, cell.a, Ls = [1,1,1], ke_cutoff=cell.ke_cutoff, basis=cell.basis, pseudo=cell.pseudo, verbose=10)   
            prim_partition = [[0,1,2,3], [4,5,6,7]]
            prim_mesh = prim_cell.mesh
            
            Ls = [1, 1, N]
            Ls = np.array(Ls, dtype=np.int32)
            mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
            mesh = np.array(mesh, dtype=np.int32)
                        
            cell, group_partition = build_supercell_with_partition(
                                    cell.atom, cell.a, mesh=mesh, 
                                    Ls=Ls,
                                    basis=cell.basis, 
                                    pseudo=cell.pseudo,
                                    partition=prim_partition, ke_cutoff=cell.ke_cutoff, verbose=verbose) 
        
            ####### bench mark MP2 ####### 
            
            import numpy
            from pyscf.pbc import gto, scf, mp

            mf = scf.RHF(cell)
            # mf.kernel()
            mypt = mp.RMP2(mf)
            # mypt.kernel()
            
            ####### isdf MP2 can perform directly! ####### 
            
            myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False)
            myisdf.verbose = 10
            myisdf.build_IP_local(c=c, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
            myisdf.build_auxiliary_Coulomb(debug=True)
            
            mf_isdf = scf.RHF(cell)
            myisdf.direct_scf = mf_isdf.direct_scf
            mf_isdf.with_df = myisdf
            mf_isdf.max_cycle = 8
            mf_isdf.conv_tol = 1e-8
            mf_isdf.kernel()
                        
            isdf_pt = mp.RMP2(mf_isdf)
            isdf_pt.kernel()
            
            mf_isdf.with_df.LS_THC_recompression(mf_isdf.with_df.aoRg_full()[0], force_LS_THC=False)
            isdf_pt = mp.RMP2(mf_isdf)
            isdf_pt.kernel()
                        
            ######################## CCSD ########################
            
            ## benchmark ##
            
            mycc = pyscf.cc.CCSD(mf)
            # mycc.kernel()
            
            mycc_isdf, eris_ccsd = RCCSD_isdf(mf_isdf, run=False, cc2=False)
            mycc_isdf.kernel(eris=eris_ccsd)
            
            eip,cip = mycc_isdf.ipccsd(nroots=2, eris=eris_ccsd)
            eea,cea = mycc_isdf.eaccsd(nroots=2, eris=eris_ccsd)

            print("eip = ", eip)
            print("eea = ", eea)
                    
            ####### THC-DF ####### 
            
            _myisdf = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False)
            _myisdf.build_IP_local(c=15, m=5, group=group_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
            R,_        = _myisdf.aoRg_full()
            Z          = LS_THC(myisdf, R)
            eri_LS_THC = LS_THC_eri(Z, R) 
            print("eri_LS_THC = ", eri_LS_THC[0,0,0,0])
            eri_benchmark = myisdf.get_eri(compact=False)
            print("eri_benchmark = ", eri_benchmark[0,0,0,0])
            diff          = np.linalg.norm(eri_LS_THC - eri_benchmark)
            print("diff = ", diff/np.sqrt(eri_benchmark.size))