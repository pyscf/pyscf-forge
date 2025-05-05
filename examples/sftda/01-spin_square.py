#!/usr/bin/env/python

'''
Test the spin_square calculation tool in sftda.tools_td.py.
'''

from pyscf import gto,dft,sftda,tdscf
from pyscf.sftda import tools_td

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

# extype serves for controling the spin excitation type in the spin_square fu-
# nction. 0 for spin flip up, 1 for spin flip down and 2 for spin conserving.
# tdtype serves for TDA/ TDDFT object in using 'TDA' or 'TDDFT'.
ssI = tools_td.spin_square(mf,mftd1.xy[0],extype=0,tdtype='TDA')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 2. <S^2> for spin flip down TDA
#
mftd2 = sftda.TDA_SF(mf)
mftd2.extype=1
mftd2.nstates = 5
mftd2.kernel()

ssI = tools_td.spin_square(mf,mftd2.xy[0],extype=1,tdtype='TDA')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 3. <S^2> for spin flip up TDDFT
#
mftd3 = sftda.TDDFT_SF(mf)
mftd3.extype=0
mftd3.nstates = 5
mftd3.kernel()

ssI = tools_td.spin_square(mf,mftd3.xy[0],extype=0,tdtype='TDDFT')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 4. <S^2> for spin flip down TDDFT
#
mftd4 = sftda.TDDFT_SF(mf)
mftd4.extype=1
mftd4.nstates = 5
mftd4.kernel()

ssI = tools_td.spin_square(mf,mftd4.xy[0],extype=1,tdtype='TDDFT')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 5. <S^2> for spin conserving TDA.
#
mftd5 = tdscf.TDA(mf)
mftd5.nstates = 5
mftd5.kernel()

ssI = tools_td.spin_square(mf,mftd5.xy[0],extype=2,tdtype='TDA')
print('The spin square of the first excited state is : ' + str(ssI))

#
# 6. <S^2> for spin conserving TDDFT.
#
mftd6 = tdscf.TDDFT(mf)
mftd6.nstates = 5
mftd6.kernel()

ssI = tools_td.spin_square(mf,mftd6.xy[0],extype=2,tdtype='TDDFT')
print('The spin square of the first excited state is : ' + str(ssI))
