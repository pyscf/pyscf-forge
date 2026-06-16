#!/usr/bin/env python

'''
Spin-flip TDDFT/TDA examples.

Key features demonstrated:
1. Spin-flip down (High spin -> Low spin) setup.
2. Spin-flip up (High spin -> Higher spin) setup.
3. Critical parameters: extype, collinear_samples
4. Validation via explicit construction and diagonalization of the Casida matrix.
'''

import numpy as np
from pyscf import gto
from pyscf import sftda  # Necessary import

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

# -------------------------------------------------------------------
# 1. Initialization
# -------------------------------------------------------------------
atom = '''
H  0.000000  0.934473 -0.588078
H  0.000000 -0.934473 -0.588078
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.221104
'''
# Triplet reference state
mol = gto.M(atom=atom, basis='6-31G', charge=0, spin=2, symmetry=True)
mf = mol.UKS(xc='CAM-B3LYP')
mf.kernel()

print_header("Reference Ground State Analysis")
mf.analyze() # Check orbital symmetries of the reference

# -------------------------------------------------------------------
# 2. Spin-Flip-Down TDDFT (e.g., Triplet -> Singlet)
# -------------------------------------------------------------------
print_header("CALCULATION 1: Spin-Flip-Down TDDFT")

sfd_tddft = mf.TDDFT_SF()
sfd_tddft.nstates = 5

# --- Key Settings ---
# extype=0/1/2 for spin-flip-up/flip-down/conserving excitations
sfd_tddft.extype = 1 # 1 for spin-flip-down excitations

# collinear_samples: Grid points for multicollinear integration.
# 20 is a typical robust value. -1 for SF-TDDFT using collinear functional
sfd_tddft.collinear_samples = 20

sfd_tddft.kernel()
sfd_tddft.analyze(verbose=4) # Verbose=4 shows orbital composition

# -------------------------------------------------------------------
# 3. Validation: Full Diagonalization of Casida Matrix
# -------------------------------------------------------------------
print_header("VALIDATION: Explicit Casida Matrix Diagonalization")

a, b = sftda.uhf_sf.get_ab_sf(mf, collinear_samples=20)
A_baba, A_abab = a
B_baab, B_abba = b
n_occ_a, n_virt_b = A_abab.shape[0], A_abab.shape[1]
n_occ_b, n_virt_a = B_abba.shape[2], B_abba.shape[3]
A_abab_2d = A_abab.reshape((n_occ_a*n_virt_b, n_occ_a*n_virt_b), order='C')
B_abba_2d = B_abba.reshape((n_occ_a*n_virt_b, n_occ_b*n_virt_a), order='C')
B_baab_2d = B_baab.reshape((n_occ_b*n_virt_a, n_occ_a*n_virt_b), order='C')
A_baba_2d = A_baba.reshape((n_occ_b*n_virt_a, n_occ_b*n_virt_a), order='C')
Casida_matrix = np.block([[ A_abab_2d, B_abba_2d],
                          [-B_baab_2d,-A_baba_2d]])
eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)
idx = eigenvals.real.argsort()
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]
norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0)**2
norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0)**2
sfd_eigenvals = eigenvals[norms>0]
sfd_eigenvecs = eigenvecs[:, norms>0].transpose(1, 0)

def norm_xy(z):
    x = z[:n_occ_a*n_virt_b].reshape(n_occ_a, n_virt_b)
    y = z[n_occ_a*n_virt_b:].reshape(n_occ_b, n_virt_a)
    norm = np.linalg.norm(x)**2 - np.linalg.norm(y)**2
    norm = np.sqrt(1./norm)
    return (x*norm, y*norm)

# Create a dummy object
fake_td = mf.TDDFT_SF()
fake_td.nstates = 5
fake_td.extype = 1
fake_td.collinear_samples = 20
fake_td.e = sfd_eigenvals
fake_td.xy = [norm_xy(z) for z in sfd_eigenvecs]
fake_td.analyze()

# -------------------------------------------------------------------
# 4. Spin-Flip-Down TDA
# -------------------------------------------------------------------
print_header("CALCULATION 2: Spin-Flip-Down TDA")

sfd_tda = mf.TDA_SF()
sfd_tda.extype = 1
sfd_tda.nstates = 5
sfd_tda.collinear_samples = -1 # Use collinear functional
sfd_tda.kernel()
sfd_tda.analyze()

# -------------------------------------------------------------------
# 5. Spin-Flip-Up TDDFT (e.g. Triplet -> Quintet)
# -------------------------------------------------------------------
print_header("CALCULATION 3: Spin-Flip-Up TDDFT")

sfu_tddft = sftda.TDDFT_SF(mf)
sfu_tddft.extype = 0  # 0 for spin-flip-up excitations
sfu_tddft.nstates = 5
sfu_tddft.collinear_samples = 20
sfu_tddft.kernel()
sfu_tddft.analyze(verbose=4)

print("\n--- Compare with Casida Neg-Norm Roots ---")
# Spin-flip-up roots appear as negative eigenvalues in the full Casida matrix
sfu_evals_val = eigenvals[norms < 0]
print(f"Casida derived Spin-flip-up Energies (eV): \n{-sfu_evals_val[-5:][::-1] * 27.2114}")
