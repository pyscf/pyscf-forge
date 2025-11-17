#!/usr/bin/env/python

'''
Test transition_analyze function in sftda.tool_td.py.
'''

import numpy as np
from pyscf import lib, gto, dft
from pyscf import sftda
from pyscf.sftda import tools_td

mol = gto.Mole()
mol.verbose = 6
mol.output = None
mol.spin = 2
mol.atom = 'O 0 0 2.07; O 0 0 0'
mol.unit = 'B'
mol.basis = '631g'
mol.build()

mf = dft.UKS(mol)
mf.xc = 'svwn' # blyp, b3lyp, tpss
mf.kernel()

mftd1 = sftda.TDA_SF(mf)
mftd1.nstates = 5
mftd1.extype = 0
mftd1.collinear_samples=200
mftd1.kernel()

e = mftd1.e
xy = mftd1.xy

#
# 1. Print tansition analyze for the first spin-flip up excited state.
#
tools_td.transition_analyze(mf, mftd1, e[0], xy[0], tdtype='TDA')


mftd2 = sftda.uks_sf.TDDFT_SF(mf)
mftd2.nstates = 5
mftd2.extype = 1
mftd2.collinear_samples=200
mftd2.kernel()

e = mftd2.e
xy = mftd2.xy

#
# 2. Print tansition analyze for the first spin-flip down excited state.
#
for i in range(5):
    tools_td.transition_analyze(mf, mftd2, e[i], xy[i], tdtype='TDDFT')


#
# 3. Print tansition analyze for the first spin-flip down excited state using TDDFT.
#
a, b = sftda.TDDFT_SF(mf).get_ab_sf(collinear_samples=200)
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

eigenvals, eigenvecs = np.linalg.eig(Casida_matrix)

# sort the eigenvalues and eigenvectors
idxt = eigenvals.real.argsort()
eigenvals = eigenvals[idxt]
eigenvecs = eigenvecs[:, idxt]
# find the positive roots
idxp = np.where(eigenvals>-1e-4)[0]
eigenvals = eigenvals[idxp]
eigenvecs = eigenvecs[:, idxp].transpose(1,0)

def norm_xy(z):
    x = z[:n_occ_a*n_virt_b].reshape(n_occ_a,n_virt_b)
    y = z[n_occ_a*n_virt_b:].reshape(n_occ_b,n_virt_a)
    norm = lib.norm(x)**2 - lib.norm(y)**2
    norm = np.sqrt(1./norm)
    return ((0,x*norm), (y*norm,0))

class CasidaSolution:
    pass

mfcasida = CasidaSolution()
mfcasida.extype = 1
mfcasida.e = eigenvals
mfcasida.xy = [norm_xy(z) for z in eigenvecs[:5]]

for i in range(5):
    tools_td.transition_analyze(mf, mfcasida, mfcasida.e[i], mfcasida.xy[i], tdtype='TDDFT')
