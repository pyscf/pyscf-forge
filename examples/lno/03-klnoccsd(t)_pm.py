import numpy as np

from pyscf.pbc import gto, scf, mp
from pyscf import lo
from pyscf.pbc.lno.tools import k2s_scf
from pyscf.pbc.lno.tools import sort_orb_by_cell
from pyscf.pbc.lno import KLNOCCSD
from pyscf.lno import LNOCCSD

atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
'''
a = np.eye(3) * 4
basis = 'cc-pvdz'
kmesh = [3,1,1]

scaled_center = None

cell = gto.M(atom=atom, basis=basis, a=a).set(verbose=4)
kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)

kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.kernel()

mf = k2s_scf(kmf)

# KLNO with PM localized orbitals
# PM localization within the BvK supercell
orbocc = mf.mo_coeff[:,mf.mo_occ>1e-6]
mlo = lo.PipekMezey(mf.cell, orbocc)
lo_coeff = mlo.kernel()
while True: # always performing jacobi sweep to avoid trapping in local minimum/saddle point
    lo_coeff1 = mlo.stability_jacobi()[1]
    if lo_coeff1 is lo_coeff:
        break
    mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
    mlo.init_guess = None
    lo_coeff = mlo.kernel()

# sort LOs by unit cell
s1e = mf.get_ovlp()
Nk = len(kpts)
nlo = lo_coeff.shape[1]//Nk
lo_coeff = sort_orb_by_cell(mf.cell, lo_coeff, Nk, s=s1e)

frag_lolist = [[i] for i in range(nlo)]

# KLNOCCSD calculations
kmlno = KLNOCCSD(kmf, lo_coeff, frag_lolist, mf=mf).set(verbose=5)
kmlno.lno_thresh = [1e-4, 1e-5]
kmlno.kernel()

# Supercell LNOCCSD calculation (the two should match!)
frag_lolist = [[i] for i in range(nlo*Nk)]
mlno = LNOCCSD(mf, lo_coeff, frag_lolist)
mlno.lno_thresh = [1e-4, 1e-5]
mlno.kernel()

def print_compare(name, ek, es):
    print(f'{name:9s} Ecorr:  {ek: 14.9f}  {es: 14.9f}  diff: {es-ek: 14.9f}')

print()
print('Comparing KLNO with supercell LNO (normalized to per cell):')
print_compare('LNOMP2', kmlno.e_corr_pt2, mlno.e_corr_pt2/Nk)
print_compare('LNOCCSD', kmlno.e_corr_ccsd, mlno.e_corr_ccsd/Nk)