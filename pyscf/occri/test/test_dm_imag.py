
import numpy as np
import pyscf
from pyscf.pbc import gto, scf
from pyscf.occri import OCCRI

# Create a simple 2D system with magnetic field
cell = gto.Cell()
cell.atom = '''
H 0.0 0.0 0.0
H 1.0 0.0 0.0
'''
cell.a = np.array([
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 10.0]  # Large vacuum in z direction
])
cell.basis = 'sto-3g'
cell.dimension = 2
cell.build()

# Apply external magnetic field
# This breaks time-reversal symmetry
magnetic_field = np.array([0.0, 0.0, 0.1])  # B field in z direction
cell.set_common_orig([0, 0, 0])

# Use UHF with magnetic field
kmesh = [2, 2, 1]
kpts = cell.make_kpts(kmesh)
mf = scf.KUHF(cell, kpts=kpts)

# Add magnetic field via vector potential A = (-By/2, Bx/2, 0)
# This gives B = curl(A) = (0, 0, B)
def add_magnetic_field(h1e, cell, kpts):
    h1e_new = h1e.copy()
    B = 0.1
    # Simple linear magnetic field implementation
    # In practice, you'd need proper gauge-invariant implementation
    for i, kpt in enumerate(kpts):
        # Add small imaginary perturbation that breaks time-reversal
        h1e_new[i] += 1j * B * 0.01 * np.random.random(h1e[i].shape)
    return h1e_new

# Monkey patch to add magnetic field
original_get_hcore = mf.get_hcore
def get_hcore_with_field(cell=None, kpts=None):
    h1e = original_get_hcore(cell, kpts)
    return add_magnetic_field(h1e, cell or mf.cell, kpts or mf.kpts)
mf.get_hcore = get_hcore_with_field

# Run SCF
dm_init = mf.get_init_guess()
print("Initial DM max imaginary part:", abs(dm_init.imag).max())

mf.kernel()
dm_final = mf.make_rdm1()
print("Final DM max imaginary part:", abs(dm_final.imag).max())

# Test with OCCRI
mf.with_df = OCCRI(mf, kmesh=kmesh)
vk = mf.get_k()
print("Exchange matrix computed successfully")

# #   Alternative: Simpler Spin-Polarized System

# #   If the magnetic field approach is too complex, try this simpler open-shell system:

# # Open-shell radical system
# cell = gto.Cell()
# cell.atom = '''
# Li 0.0 0.0 0.0
# '''
# cell.a = np.eye(3) * 4.0  # 4 Bohr cubic cell
# cell.basis = 'sto-3g'
# cell.build()

# kmesh = [2, 2, 2]
# kpts = cell.make_kpts(kmesh)
# mf = scf.KUHF(cell, kpts=kpts)

# # Force spin polarization
# mf.init_guess = 'atom'
# dm_init = mf.get_init_guess()

# # Break symmetry manually
# dm_init[0] += 0.1j * np.random.random(dm_init[0].shape) * 0.01
# dm_init[1] -= 0.1j * np.random.random(dm_init[1].shape) * 0.01

# mf.kernel(dm_init)
# dm_final = mf.make_rdm1()
# print("Final DM max imaginary part:", abs(dm_final.imag).max())
