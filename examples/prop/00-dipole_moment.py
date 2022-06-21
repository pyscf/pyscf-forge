#!/usr/bin/env python

'''
Response-theory calculation of the permanent dipole moment
'''

from pyscf import gto, scf, mcpdft
from pyscf.lib import logger

logger.TIMER_LEVEL = logger.INFO

# Energy calculation
h2co_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_xyz, basis = 'def2svp', symmetry = False,
             verbose = logger.INFO, output = '00-dipole_moment.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6)
mc.kernel ()

# Electric Dipole calculation
dipole = mc.dip_moment(unit='Debye')
print ("MC-PDFT electric dipole moment Debye")
print (" {:8.5f} {:8.5f} {:8.5f}".format (*dipole))
print ("Numerical MC-PDFT electric dipole moment from GAMESS [Debye]")
print (" 2.09361 0.00000 0.00000 ")

