'''
For spin-flip TDDFT gradient, you can obtain and create a spin-flip TDDFT
object use sftda.uhf_df.get_ab_sf() function. And transfrom the excited
energy omega and transition vector x,y into the correct form. Then use
the gradient module.
'''
import numpy as np
from pyscf import lib, gto, dft
from pyscf import sftda
from pyscf.grad import tduks_sf # this import is necessary.
try:
    import mcfun # mcfun>=0.2.5 must be used.
except ImportError:
    mcfun = None

mol = gto.Mole()
mol.verbose = 6
mol.output = None
mol.spin = 2
mol.atom = 'O 0 0 2.07; O 0 0 0'
mol.unit = 'B'
mol.basis = '6-31g'
mol.build()

# UKS object
mf = dft.UKS(mol)
mf.xc = 'svwn' # blyp, b3lyp, tpss
mf.kernel()

mftd1 = sftda.TDDFT_SF(mf)
mftd1.nstates = 5 # the number of excited states
mftd1.extype = 1  # 1 for spin flip down excited energies
mftd1.collinear_samples=200
mftd1.kernel()

# print(mftd1.e)

mftdg1 = mftd1.Gradients()
g1 = mftdg1.kernel(state=2)

print(g1)


# If use get_ab_sf() to get response Matrix:
a, b = sftda.TDDFT_SF(mf).get_ab_sf()
A_baba, A_abab = a
B_baab, B_abba = b

n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]

A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b,n_occ_a*n_virt_b), order='C')
B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b,n_occ_b*n_virt_a), order='C')
B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a,n_occ_a*n_virt_b), order='C')
A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a,n_occ_b*n_virt_a), order='C')

Casida_matrix = np.block([[ A_abab_2d, B_abba_2d],
                          [-B_baab_2d,-A_baba_2d]])

# solve the response matrix
eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)

# sort the eigenvalues and eigenvectors
idxt = eigenvals.real.argsort()
eigenvals = eigenvals[idxt]
eigenvecs = eigenvecs[:, idxt]

# find the positive roots
idxp = np.where(eigenvals>-1e-4)[0]
eigenvals = eigenvals[idxp]
eigenvecs= eigenvecs[:, idxp].transpose(1,0)

# transfrom the form of transiton vector
def norm_xy(z):
    x = z[:n_occ_a*n_virt_b].reshape(n_occ_a,n_virt_b)
    y = z[n_occ_a*n_virt_b:].reshape(n_occ_b,n_virt_a)
    norm = lib.norm(x)**2 - lib.norm(y)**2
    norm = np.sqrt(1./norm)
    return ((0,x*norm), (y*norm,0))

# transfrom the form of transiton vector
mftd1 = sftda.TDA_SF(mf)
mftd1.nstates = 5 # the number of excited states
mftd1.extype = 1  # 1 for spin flip down excited energies
mftd1.collinear_samples=200
mftd1.e = eigenvals
mftd1.xy = [norm_xy(z) for z in eigenvecs[:5]]

# print(mftd1.e)

mftdg1 = mftd1.Gradients()
g1 = mftdg1.kernel(state=2)

print(g1)
