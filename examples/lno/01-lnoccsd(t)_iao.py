import numpy as np
from pyscf import gto, scf, lno, lo
from pyscf.lno import tools

atom = '''
O   0.000000 0.000000  0.000000
H   0.758602 0.000000  0.504284
H   0.260455 0.000000 -0.872893
O   3.000000 0.500000  0.000000
H   3.758602 0.500000  0.504284
H   3.260455 0.500000 -0.872893
   	'''
basis = 'cc-pvdz'
mol = gto.M(atom=atom, basis=basis, spin=0, verbose=4, max_memory=8000)
mf = scf.RHF(mol).density_fit().run()
frozen=0

# IAO localization
orbocc = mf.mo_coeff[:, mf.mo_occ > 1e-6]
iao_coeff = lo.iao.iao(mol, orbocc)
lo_coeff = lo.orth.vec_lowdin(iao_coeff, mf.get_ovlp())
moliao = lo.iao.reference_mol(mol)

# Fragment list: all IAOs belonging to same atom form a fragment
frag_lolist = tools.autofrag_iao(moliao)

lno_thresh = [1e-5,1e-6]
mlcc = lno.LNOCCSD_T(mf, lo_coeff, frag_lolist, lno_thresh=lno_thresh)
mlcc.lno_thresh = lno_thresh
mlcc.kernel()
   
eccsd = mlcc.e_corr_ccsd
eccsd_t = mlcc.e_corr_ccsd_t
   
eref = -0.4209369350372839 # reference CCSD energy
err = eccsd - eref

print()
print(('E_corr= % .10f (err= % .10f)  '%(eccsd, err)))
print()