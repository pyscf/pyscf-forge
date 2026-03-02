import numpy as np
from pyscf import gto
from pyscf.gto import Mole
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc.gto import Cell
from .physicalconstants import PhysicalConstants
# Alternative for read_xyzf
#
# !! in read_xyzf, -outputunit- was implicit (defaulted to ANGS, since xyzf is in ANGS unit) 
# !! whereas in this subroutine, you MUST explicitly specify A or B.
# 
# !! outputunit (A or B) also applies to lattice vectors
# 
def read_mol_or_cell(mol_or_cell,dict=None,outputunit=None):
    assert (outputunit=='A' or outputunit=='B'), "PLS specify outputunit"
    spdm=3
    natm=mol_or_cell.natm
    BOHRinANGS=PhysicalConstants.BOHRinANGS()
    if(isinstance(mol_or_cell,Cell)):
        # 
        latticevectors_BOHR = np.array( mol_or_cell.lattice_vectors() )
        # print("latticevectors_BOHR",latticevectors_BOHR)
        if( outputunit == 'A' ):
            dict.update({'a':latticevectors_BOHR*BOHRinANGS})
        else:
            dict.update({'a':latticevectors_BOHR})
# We assume that _atom = [ row_1, row_2, ... ,row_nAtm ]
    assert len(mol_or_cell._atom)==natm,"unexpected size of mol_or_cell._atom"
    Sy=[];Rnuc_BOHR=[]
    for ia in range(natm):
        row=mol_or_cell._atom[ia]
        s=row[0];
        assert isinstance(s,str),""
        Sy.append(s)
        xyz=row[1]
        assert len(xyz)==spdm,""
        assert isinstance(xyz[0],float) or isinstance(xyz[0],np.float64),""
        Rnuc_BOHR.append(xyz)
    if(outputunit=='A'):
        Rnuc_ANGS=np.array(Rnuc_BOHR)*BOHRinANGS
        return Rnuc_ANGS,Sy
    else:
        return Rnuc_BOHR,Sy

mol_H2O = gto.Mole()
mol_H2O.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
mol_H2O.basis = 'sto-3g'
mol_H2O.build()

mol_CH4 = gto.M(atom = 'C      -0.00002631       0.00024432       0.00007638;'\
+'H       1.09719302      -0.00099542      -0.00067323;'\
+'H      -0.36460184       1.03512412       0.00149901;'\
+'H      -0.36573517      -0.51759524       0.89561804;'\
+'H      -0.36696120      -0.51556008      -0.89613718', basis="aug-ccpvTz")


cell_C2 = pbc_gto.M(
    atom = '''C 0.0000 0.0000 0.0000
              C 0.8917 0.8917 0.8917''',
    a = '''0.0000 1.7834 1.7834
           1.7834 0.0000 1.7834
           1.7834 1.7834 0.0000''',
    pseudo = 'gth-pade',
    basis = 'gth-szv'
)
cell_C8 = pbc_gto.M(
    verbose = 4,
    a = np.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'sto3g',
)

for m_or_c in [mol_H2O, mol_CH4, cell_C2, cell_C8]:
    Dic={}
    #R,Sy=read_mol_or_cell(m_or_c,dict=Dic,outputunit='B')
    R,Sy=read_mol_or_cell(m_or_c,dict=Dic,outputunit='A')
    print(m_or_c.atom_symbol(0))
    print("Rnuc:",str(R))
    print("Sy:",str(Sy))
    gbs=m_or_c.basis
    print("gbs:",str(gbs))
    if( isinstance(m_or_c,Cell) ):
        print("a:",str(Dic['a']))
        psp=m_or_c.pseudo
        print("psp:",str(psp))
    
# O
# Rnuc: [[0.0, 0.0, 0.0], [0.0, 1.8897261245650618, 0.0], [0.0, 0.0, 1.8897261245650618]]
# Sy: ['O', 'H', 'H']
# gbs: sto-3g
# C
# Rnuc: [[-4.971869433730678e-05, 0.00046169788675373593, 0.00014433728139427943], [2.073394313584436, -0.0018810711789145537, -0.0012722203188409366], [-0.6889976221124907, 1.95610109173142, 0.0028327183579842734], [-0.691139305421244, -0.9781132469785231, 1.6924728078197566], [-0.6934561663417446, -0.9742673519588532, -1.6934538402400632]]
# Sy: ['C', 'H', 'H', 'H', 'H']
# gbs: aug-ccpvTz
# C
# Rnuc: [[0.0, 0.0, 0.0], [1.6850687852746657, 1.6850687852746657, 1.6850687852746657]]
# Sy: ['C', 'C']
# gbs: gth-szv
# a: [[0.         3.37013757 3.37013757]
#  [3.37013757 0.         3.37013757]
#  [3.37013757 3.37013757 0.        ]]
# psp: gth-pade
# C
# Rnuc: [[0.0, 0.0, 0.0], [1.6850687852746657, 1.6850687852746657, 1.6850687852746657], [3.3701375705493315, 3.3701375705493315, 0.0], [5.055206355823997, 5.055206355823997, 1.6850687852746657], [3.3701375705493315, 0.0, 3.3701375705493315], [5.055206355823997, 1.6850687852746657, 5.055206355823997], [0.0, 3.3701375705493315, 3.3701375705493315], [1.6850687852746657, 5.055206355823997, 5.055206355823997]]
# Sy: ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
# gbs: sto3g
# a: [[6.74027514 0.         0.        ]
#  [0.         6.74027514 0.        ]
#  [0.         0.         6.74027514]]
# psp: None

# 
# O
# Rnuc: [[0. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
# Sy: ['O', 'H', 'H']
# gbs: sto-3g
# C
# Rnuc: [[-2.63100000e-05  2.44320000e-04  7.63799999e-05]
#  [ 1.09719302e+00 -9.95419999e-04 -6.73230000e-04]
#  [-3.64601840e-01  1.03512412e+00  1.49901000e-03]
#  [-3.65735170e-01 -5.17595240e-01  8.95618039e-01]
#  [-3.66961200e-01 -5.15560080e-01 -8.96137179e-01]]
# Sy: ['C', 'H', 'H', 'H', 'H']
# gbs: aug-ccpvTz
# C
# Rnuc: [[0.     0.     0.    ]
#  [0.8917 0.8917 0.8917]]
# Sy: ['C', 'C']
# gbs: gth-szv
# a: [[0.     1.7834 1.7834]
#  [1.7834 0.     1.7834]
#  [1.7834 1.7834 0.    ]]
# psp: gth-pade
# C
# Rnuc: [[0.     0.     0.    ]
#  [0.8917 0.8917 0.8917]
#  [1.7834 1.7834 0.    ]
#  [2.6751 2.6751 0.8917]
#  [1.7834 0.     1.7834]
#  [2.6751 0.8917 2.6751]
#  [0.     1.7834 1.7834]
#  [0.8917 2.6751 2.6751]]
# Sy: ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
# gbs: sto3g
# a: [[3.5668 0.     0.    ]
#  [0.     3.5668 0.    ]
#  [0.     0.     3.5668]]
# psp: None
