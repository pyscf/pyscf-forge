#!/usr/bin/env/python

'''
Test the spin_square calculation tool in sftda.tools_td.py.
'''

from pyscf import gto,dft,sftda,tdscf
from pyscf.sftda import tools

# ToDo : add the spin flip TDDFT parts.

mol = gto.Mole()
mol.verbose = 3
mol.output = None
mol.spin = 2
mol.atom = 'O 0 0 2.07; O 0 0 0'
mol.unit = 'B'
mol.verbose = 6
mol.basis = '6-31g'
mol.build()

mf = dft.UKS(mol)
mf.xc = 'svwn'
mf.kernel()

#
# 1. <S^2> for spin flip up TDA
#
mftd1 = sftda.TDA_SF(mf)
mftd1.extype=0
mftd1.nstates = 5
mftd1.kernel()

# extype serves for controling spin excitation type in the spin_square function.
# tdtype serves for TDA/ TDDFT object in using 'TDA' or 'TDDFT'.
ssI = tools.spin_square(mf,mftd1.xy[0],extype=0,tdtype='TDA')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 2. <S^2> for spin flip down TDA
#
mftd2 = sftda.TDA_SF(mf)
mftd2.extype=1
mftd2.nstates = 5
mftd2.kernel()

ssI = tools.spin_square(mf,mftd1.xy[0],extype=1,tdtype='TDA')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 3. <S^2> for spin conserving TDA.
#
mftd2 = tdscf.TDA(mf)
mftd2.nstates = 5
mftd2.kernel()

ssI = tools.spin_square(mf,mftd1.xy[0],extype=2,tdtype='TDA')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 4. <S^2> for spin conserving TDDFT.
#
mftd2 = tdscf.TDDFT(mf)
mftd2.nstates = 5
mftd2.kernel()

ssI = tools.spin_square(mf,mftd1.xy[0],extype=2,tdtype='TDDFT')
print('The spin square of the first excited state is : ' + str(ssI))
