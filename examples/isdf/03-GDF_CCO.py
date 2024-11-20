import numpy as np
from pyscf import lib
from pyscf.gto.mole import *

from pyscf.isdf import isdf_tools_cell
from pyscf.isdf import isdf_local_k
from pyscf.isdf import isdf_jk
from pyscf.isdf import isdf_local

from pyscf.lib.parameters import BOHR

#### NOTE: a full tests on combinations of parameters ####

prim_a = np.array(
                [[14.572056092, 0.000000000, 0.000000000],
                 [0.000000000, 14.572056092, 0.000000000],
                 [0.000000000, 0.000000000,  6.010273939],]) * BOHR
atm = [
['Cu',	(1.927800,	1.927800,	1.590250)],
['Cu',	(5.783400,	5.783400,	1.590250)],
['Cu',	(1.927800,	5.783400,	1.590250)],
['Cu',	(5.783400,	1.927800,	1.590250)],
['O',	(1.927800,	3.855600,	1.590250)],
['O',	(3.855600,	5.783400,	1.590250)],
['O',	(5.783400,	3.855600,	1.590250)],
['O',	(3.855600,	1.927800,	1.590250)],
['O',	(0.000000,	1.927800,	1.590250)],
['O',	(1.927800,	7.711200,	1.590250)],
['O',	(7.711200,	5.783400,	1.590250)],
['O',	(5.783400,	0.000000,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
['Ca',	(3.855600,	3.855600,	0.000000)],
['Ca',	(7.711200,	3.855600,	0.000000)],
['Ca',	(3.855600,	7.711200,	0.000000)],
]
   
C_ARRAY = [25, 30, 35]
RELA_CUTOFF = [1e-3, 3e-4, 1e-4]
SuperCell_ARRAY = [
    [1, 1, 1],
]
Ke_CUTOFF = [256]
Basis = ['gth-dzvp']

PARTITION = [
    [[0],  [1],  [2],  [3], 
     [4],  [5],  [6],  [7], 
     [8],  [9],  [10], [11], 
     [12], [13], [14], [15]]
]

if __name__ == '__main__':
    
    for supercell in SuperCell_ARRAY:
        ke_cutoff = Ke_CUTOFF[0]
        for partition in PARTITION:   ## test different partition of atoms
            for _basis_ in Basis:
                
                DM_CACHED = None
                
                from pyscf.gto.basis import parse_nwchem
                fbas="basis2.dat"
                atms = ['O', 'Cu', "Ca"]
                basis = {atm:parse_nwchem.load(fbas, atm) for atm in atms}
                # print("basis = ", basis)
                
                pseudo = {'Cu': 'gth-pbe-q19', 'O': 'gth-pbe', 'Ca': 'gth-pbe'}
                
                prim_cell = isdf_tools_cell.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo, verbose=4)
                prim_mesh = prim_cell.mesh
                # print("prim_mesh = ", prim_mesh)
            
                mesh = [supercell[0] * prim_mesh[0], supercell[1] * prim_mesh[1], supercell[2] * prim_mesh[2]]
                mesh = np.array(mesh, dtype=np.int32)
            
                cell, supercell_group = isdf_tools_cell.build_supercell_with_partition(atm, prim_a, 
                                                                                       partition = partition, 
                                                                                       Ls        = supercell, 
                                                                                       ke_cutoff = ke_cutoff, 
                                                                                       mesh      = mesh, 
                                                                                       basis     = basis, 
                                                                                       pseudo    = pseudo, 
                                                                                       verbose   = 4)

                cell.incore_anyway = False
                cell.max_memory    = 200   # force to call with_df.get_jk
                
                for c, rela_cutoff in zip(C_ARRAY, RELA_CUTOFF):
                    
                    print('--------------------------------------------')
                    print('C = %.2e, supercell = %s, kc_cutoff = %d, basis = %s, partition = %s' % (
                        c, str(supercell), ke_cutoff, basis, partition))

                    t1 = (lib.logger.process_clock(),lib.logger.perf_counter())
                        
                    pbc_isdf_info = isdf_local.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, direct=True, rela_cutoff_QRCP=rela_cutoff,
                                                                  limited_memory=True, build_K_bunchsize=128)
                    pbc_isdf_info.build_IP_local(c=c, group=supercell_group)
                    print("pbc_isdf_info.naux = ", pbc_isdf_info.naux) 
                    print("effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao) 
                    pbc_isdf_info.build_auxiliary_Coulomb()
                                                
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    print(isdf_jk._benchmark_time(t1, t2, 'build_isdf', pbc_isdf_info))

                    # for bunch_size in BUNCHSIZE_IO:
                    ### perform scf ###

                    from pyscf.pbc import scf

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    mf = scf.GHF(cell)
                    mf.with_df = pbc_isdf_info
                    mf.max_cycle = 64
                    mf.conv_tol = 1e-7
                    pbc_isdf_info.direct_scf = mf.direct_scf
                    if DM_CACHED is not None:
                        mf.kernel(DM_CACHED)
                    else:
                        mf.kernel()
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    print(isdf_jk._benchmark_time(t1, t2, 'scf_isdf', pbc_isdf_info))
                        
                    del mf
                    del pbc_isdf_info
                    
                ### GDF benchmark ###
                
                mf = scf.GHF(cell).density_fit()
                mf.max_cycle = 64
                mf.conv_tol = 1e-7
                # pbc_isdf_info.direct_scf = mf.direct_scf
                mf.kernel(DM_CACHED)
                    
                exit(1)