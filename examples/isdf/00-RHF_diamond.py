import numpy as np
from pyscf import lib
from pyscf.gto.mole import *

from pyscf.isdf import isdf_tools_cell
from pyscf.isdf import isdf_local_k
from pyscf.isdf import isdf_jk
from pyscf.isdf import isdf_local

from pyscf.lib.parameters import BOHR

MOL_STRUCTURE = '''
                   C     0.      0.      0.
                   C     0.8917  0.8917  0.8917
                   C     1.7834  1.7834  0.
                   C     2.6751  2.6751  0.8917
                   C     1.7834  0.      1.7834
                   C     2.6751  0.8917  2.6751
                   C     0.      1.7834  1.7834
                   C     0.8917  2.6751  2.6751
                '''

#### NOTE: a full tests on combinations of parameters ####
                
C_ARRAY = [15, 15, 20, 25, 30, 30]
RELA_CUTOFF = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
SuperCell_ARRAY = [
    # [1, 1, 1],
    [1, 1, 2],
    # [1, 2, 2],
    # [2, 2, 2],
    # [3, 3, 3],
    # [2, 4, 4],
    # [3, 4, 4],
    # [5, 5, 5],
    # [6, 6, 6],
    # [1, 1, 4],
    # [1, 1, 8],
    # [1, 1, 16],
    # [1, 1, 32],
]


Ke_CUTOFF = [70]
boxlen = 3.5668
Basis = ['gth-dzvp']

PARTITION = [
    [[0,1],[2,3],[4,5],[6,7]],
    [[0,1,2,3],[4,5,6,7]],
    [[0,1,2,3,4,5,6,7]],
    [[0],[1],[2],[3],[4],[5],[6],[7]],
]

if __name__ == '__main__':

    boxlen = 3.57371000
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    atm = [
        ['C', (0.        , 0.        , 0.    )],
        ['C', (0.8934275 , 0.8934275 , 0.8934275)],
        ['C', (1.786855  , 1.786855  , 0.    )],
        ['C', (2.6802825 , 2.6802825 , 0.8934275)],
        ['C', (1.786855  , 0.        , 1.786855)],
        ['C', (2.6802825 , 0.8934275 , 2.6802825)],
        ['C', (0.        , 1.786855  , 1.786855)],
        ['C', (0.8934275 , 2.6802825 , 2.6802825)],
    ]
    
    for supercell in SuperCell_ARRAY:
        ke_cutoff = Ke_CUTOFF[0]
        for partition in PARTITION:   ## test different partition of atoms
            for basis in Basis:
                for c, rela_cutoff in zip(C_ARRAY, RELA_CUTOFF):
                # for c in C_ARRAY:
                    print('--------------------------------------------')
                    print('C = %.2e, supercell = %s, kc_cutoff = %d, basis = %s, partition = %s' % (
                        c, str(supercell), ke_cutoff, basis, partition))

                    prim_cell = isdf_tools_cell.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo="gth-pade", verbose=4)
                    prim_mesh = prim_cell.mesh
                    print("prim_mesh = ", prim_mesh)
            
                    mesh = [supercell[0] * prim_mesh[0], supercell[1] * prim_mesh[1], supercell[2] * prim_mesh[2]]
                    mesh = np.array(mesh, dtype=np.int32)
            
                    cell, supercell_group = isdf_tools_cell.build_supercell_with_partition(atm, prim_a, partition=partition, Ls = supercell, ke_cutoff=ke_cutoff, mesh=mesh, basis=basis, pseudo="gth-pade", verbose=4)

                    cell.incore_anyway = False
                    cell.max_memory    = 200   # force to call with_df.get_jk

                    t1 = (lib.logger.process_clock(),lib.logger.perf_counter())
                        
                    pbc_isdf_info = isdf_local.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, direct=False, rela_cutoff_QRCP=rela_cutoff)
                    pbc_isdf_info.build_IP_local(c=c, group=supercell_group, Ls=[supercell[0]*4, supercell[1]*4, supercell[2]*4])
                    print("pbc_isdf_info.naux = ", pbc_isdf_info.naux) 
                    print("effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao) 
                    pbc_isdf_info.build_auxiliary_Coulomb()
                                                
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    print(isdf_jk._benchmark_time(t1, t2, 'build_isdf', pbc_isdf_info))

                    # for bunch_size in BUNCHSIZE_IO:
                    ### perform scf ###

                    from pyscf.pbc import scf

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    mf = scf.RHF(cell)
                    mf.with_df = pbc_isdf_info
                    mf.max_cycle = 32
                    mf.conv_tol = 1e-7
                    pbc_isdf_info.direct_scf = mf.direct_scf
                    mf.kernel()
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    print(isdf_jk._benchmark_time(t1, t2, 'scf_isdf', pbc_isdf_info))
                        
                    del mf
                    del pbc_isdf_info
                exit(1)