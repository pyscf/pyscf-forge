#!/usr/bin/env python
'''
Analytic gradient spin-flip TDDFT/TDA examples.
'''

import numpy as np
from pyscf import gto
from pyscf import sftda # necessary import
from pyscf.grad import tduks_sf  # necessary import

atom = '''
H  0.000000  0.934473 -0.588078
H  0.000000 -0.934473 -0.588078
C  0.000000  0.000000  0.000000
O  0.000000  0.000000  1.221104
'''
mol = gto.M(atom=atom, charge=0, spin=2, basis='6-31g', verbose=3)
fun = 'CAM-B3LYP' # try also 'CAM-B3LYP', 'TPSS', 'SVWN', etc.
mf = mol.UKS(xc=fun)
mf.grids.level = 7 # increase grid size for accuracy
mf.kernel()
td = mf.TDDFT_SF() # mf.TDA_SF() for SF-TDA
td.extype = 1
td.collinear_samples = 20 # -1 for collinear SF-TDDFT
td.nstates = 5
td.kernel()

tdg = td.Gradients()
anal_grad = tdg.kernel(state=1) # 1 for first excited state

def numerical_gradient(f, mol, delta=1e-5):
    coords = mol.atom_coords()
    grad = np.zeros_like(coords)
    for i in range(mol.natm):
        for j in range(3):
            orig_val = coords[i, j]
            coords[i, j] = orig_val + delta
            mol.set_geom_(coords, unit='Bohr')
            f_plus = f(mol)
            coords[i, j] = orig_val - delta
            mol.set_geom_(coords, unit='Bohr')
            f_minus = f(mol)
            grad[i, j] = (f_plus - f_minus) / (2 * delta)
            coords[i, j] = orig_val
    mol.set_geom_(coords, unit='Bohr')
    return grad

def f(mol):
    mf = mol.UKS(xc=fun)
    mf.kernel()
    td = mf.TDDFT_SF()
    td.extype = 1
    td.nstates = 5
    td.collinear_samples = 20
    td.kernel()
    return td.e[0] + mf.e_tot

num_grad = numerical_gradient(f, mol)

print("Analytic Gradient:\n", anal_grad)
print("Numerical Gradient:\n", num_grad)
print(np.allclose(anal_grad, num_grad, atol=1e-5))
