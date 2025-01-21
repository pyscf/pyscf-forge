from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dsrg_mrpt2
import numpy as np

molh2o = gto.M(
    verbose = 2,
    atom='''
O     0.    0.000    0.1174
H     0.    0.757   -0.4696
H     0.   -0.757   -0.4696
    ''',
    basis = '6-31g', spin=0, charge=0, symmetry=True,
)

mfh2o = scf.RHF(molh2o)
mfh2o.kernel()
mc = mcscf.CASSCF(mfh2o, 4, 4).state_average_([.5,.5],wfnsym='A1')
mc.fix_spin_(ss=0)
ncore = {'A1':2, 'B1':1}
ncas = {'A1':2, 'B1':1,'B2':1}
mo = mcscf.sort_mo_by_irrep(mc, mfh2o.mo_coeff, ncas, ncore)
mc.mc2step(mo)
pt = dsrg_mrpt2.DSRG_MRPT2(mc, relax='once')
pt.kernel()
e_sa = pt.e_relax_eigval_shifted

print('E0 = %.12f, ref = -76.11686746968063' % e_sa[0])
print('E1 = %.12f, ref = -75.71394328285785' % e_sa[1])