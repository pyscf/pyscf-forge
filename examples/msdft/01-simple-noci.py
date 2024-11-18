#!/usr/bin/env/python

# Author: Peng Bao <baopeng@iccas.ac.cn>
# Edited by: Qiming Sun <osirpt.sun@gmail.com>

from pyscf import gto, msdft

mol = gto.M(atom='''
H 1.080977 -2.558832 0.000000
H -1.080977 2.558832 0.000000
H 2.103773 -1.017723 0.000000
H -2.103773 1.017723 0.000000
H -0.973565 -1.219040 0.000000
H 0.973565 1.219040 0.000000
C 0.000000 0.728881 0.000000
C 0.000000 -0.728881 0.000000
C 1.117962 -1.474815 0.000000
C -1.117962 1.474815 0.000000
''', basis='sto-3g')

mf = msdft.NOCI(mol)
mf.xc = 'pbe0'

h = homo = mol.nelec[0] - 1
l = h + 1
# Single excitation orbital pair
mf.s = [[h,l],[h-1,l],[h,l+1],[h-1,l+1]]
# Double excitation orbital pair
mf.d = [[h,l]]

mf.run()
# reference:
#[-153.93158107 -153.8742658  -153.82198958 -153.69666086 -153.59511111
# -153.53734913 -153.5155775  -153.47367943 -153.40221993 -153.37353437]
