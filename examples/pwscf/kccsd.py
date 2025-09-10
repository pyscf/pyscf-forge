import numpy as np
from pyscf.pbc.pwscf.pw_helper import gtomf2pwmf
from pyscf.pbc.pwscf.kccsd_rhf import PWKRCCSD

"""
Simple CCSD calculation
"""

a0 = 1.78339987
atom = "C 0 0 0; C %.10f %.10f %.10f" % (a0*0.5, a0*0.5, a0*0.5)
a = np.asarray([
        [0., a0, a0],
        [a0, 0., a0],
        [a0, a0, 0.]])

from pyscf.pbc import gto, scf, pwscf
cell = gto.Cell(atom=atom, a=a, basis="gth-szv", pseudo="gth-pade",
                ke_cutoff=50)
cell.build()
cell.verbose = 5

kpts = cell.make_kpts([2,1,1])

mf = scf.KRHF(cell, kpts)
mf.kernel()

mcc = cc.kccsd_rhf.RCCSD(mf)
mcc.kernel()

pwmf = gtomf2pwmf(mf)
pwmcc = PWKRCCSD(pwmf).kernel()

assert(np.abs(mcc.e_corr - pwmcc.e_corr) < 1e-5)