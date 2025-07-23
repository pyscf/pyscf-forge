import pyscf
import numpy
import re
import time
from pyscf.pbc.gto import Cell
from ase.build import bulk
from pyscf.scf.uhf import mulliken_spin_pop
from pyscf.occri import OCCRI

A2B = 1.889725989

def get_ase_cell():
    ase_atom = bulk('NiO', 'rocksalt', a=4.17*A2B, cubic=True) # 10.1103/PhysRevB.74.155108
    atom = [[atom.symbol, atom.position] for atom in ase_atom]
    cell = Cell(
        a=ase_atom.cell[:],
        unit="B",
        atom=atom,
        ke_cutoff = 190,
        basis = "gth-dzvp-molopt-sr",
        pseudo = "gth-pbe",
        spin = 0,
        verbose = 4,
    )
    cell.build()
    return cell

def flip_spin(mf, dm, afm_guess, bias=0.):
    cell = mf.cell

    # Function to strictly match the full AO label
    def find_exact_ao_indices(cell, target_label):
        pattern = rf"^{re.escape(target_label)}$"  # Ensure exact match
        return [i for i, label in enumerate(cell.ao_labels()) if re.match(pattern, label)]

    # Find AO indices for 3dx2-y2 orbitals
    alpha_indices = []
    beta_indices = []

    for key, ao_labels in afm_guess.items():
        print(key)
        for label in ao_labels:
            ao_idx = find_exact_ao_indices(cell, label)  # Find indices of specific AOs
            print("flipping:", ao_idx, label, dm[0][ao_idx, ao_idx],dm[1][ao_idx, ao_idx])
            if key == "alpha":
                alpha_indices.extend(ao_idx)
            else:
                beta_indices.extend(ao_idx)

    # Apply AFM ordering by modifying the density matrix
    #DOI: 10.1103/PhysRevB.74.155108
    for ao_idx in alpha_indices:
        dm[0][ao_idx, ao_idx] += bias
        dm[1][ao_idx, ao_idx] *= 0.0

    for ao_idx in beta_indices:
        dm[0][ao_idx, ao_idx] *= 0.0
        dm[1][ao_idx, ao_idx] += bias

    s1e = cell.pbc_intor('int1e_ovlp', hermi=1).astype(numpy.float64)
    ne = numpy.einsum("xij,ji->x", dm, s1e).real
    nelec = cell.nelec
    if numpy.any(abs(ne - nelec) > 0.01):
        print(
            "Spin flip causes error in the electron number "
            "of initial guess density matrix (Ne/cell = %s)!\n"
            "  This can cause huge error in Fock matrix and "
            "lead to instability in SCF for low-dimensional "
            "systems.\n  DM is normalized wrt the number "
            "of electrons %s",
            ne,
            nelec,
        )
        dm *= (nelec / ne).reshape(2, 1, 1)

    return dm


def afm_guess_for_supercell(afm_guess, ncopies, natm):
    # New dictionary to store expanded values
    expanded_afm_guess = {"alpha": [], "beta": []}

    for spin in afm_guess:
        for entry in afm_guess[spin]:
            atom_index, element, orbital = entry.split(maxsplit=2)
            atom_index = int(atom_index)

            # Duplicate for the supercell
            for i in range(ncopies):
                new_index = atom_index + i * natm // ncopies
                expanded_afm_guess[spin].append(f"{new_index} {element} {orbital}")

    print(expanded_afm_guess)
    return expanded_afm_guess


if __name__ == "__main__":

    method = pyscf.pbc.scf.KUHF
    cell = get_ase_cell()

    ncopy = [1,1,1]
    cell = pyscf.pbc.tools.pbc.super_cell(cell, ncopy)
    cell.build()

    afm_guess = {
        "alpha": ["0 Ni 3dx2-y2", "2 Ni 3dx2-y2"],  # Spin-up on Ni 0 and Ni 2
        "beta":  ["4 Ni 3dx2-y2", "6 Ni 3dx2-y2"],  # Spin-down on Ni 1 and Ni 3
    }

    n = numpy.prod(ncopy)
    if n > 1:
        afm_guess = afm_guess_for_supercell(afm_guess, n, cell.natm)

    t0 = time.time()
    mf = method(cell)
    mf.init_guess = "minao"
    mf.conv_tol = 1.e-6
    mf.with_df = OCCRI(mf)

    dm = mf.get_init_guess()
    dm_shape = dm.shape
    dm = flip_spin(mf, dm.reshape(dm.shape[0], dm.shape[-2], dm.shape[-1]), afm_guess)
    mulliken_spin_pop(cell, dm)

    dm = dm.reshape(dm_shape)
    mf.kernel(dm0=dm)
    # dm = mf.make_rdm1()
    # mulliken_spin_pop(cell, dm)

    print("Walltime: ", time.time() -t0)

    # Walltime:  152.7448079586029