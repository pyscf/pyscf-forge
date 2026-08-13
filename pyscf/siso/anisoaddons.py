#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>


import numpy as np
import sympy as sp
from itertools import product
from functools import reduce
from numbers import Integral, Real
from os import PathLike
from scipy.linalg import block_diag

from pyscf.data import nist
from pyscf.prop.dip_moment.mcpdft import get_guage_origin
from pyscf.siso import socaddons


# Generate the required data used by ANISO to compute the static magnetic properties.

def _validate_component_matrix(mat, name='mat'):
    mat = np.asarray(mat)
    if mat.ndim != 3 or mat.shape[0] != 3 or mat.shape[1] != mat.shape[2]:
        raise ValueError(f"{name} must have shape (3, n, n)")
    if not np.issubdtype(mat.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    return mat

def _basis_transformation(mat, sivec):
    '''
    Convert the given matrix to spin orbit basis.
    '''
    return np.array([reduce(np.dot, (sivec.conj().T, m, sivec)) for m in mat])

def get_twice_ms_values(mult):
    """Return integer ``2*M_S`` values for a spin multiplicity."""
    return list(range(1 - mult, mult, 2))

def unpack_sos_basis(mat):
    """Combine component matrices for different spin groups block-diagonally."""
    if isinstance(mat, np.ndarray) and mat.ndim == 0:
        raise TypeError("mat must be a non-empty sequence of component matrices")
    try:
        blocks = list(mat)
    except TypeError as err:
        raise TypeError(
            "mat must be a non-empty sequence of component matrices") from err
    if not blocks:
        raise ValueError("mat must contain at least one component matrix")
    blocks = [
        _validate_component_matrix(block, f'mat[{index}]')
        for index, block in enumerate(blocks)
    ]
    return np.stack([
        block_diag(*(block[component] for block in blocks))
        for component in range(3)
    ])

unpack_sfs_basis = unpack_sos_basis

def generate_sos_basis(mat, mult):
    '''
    Convert nonrelativistic component matrices to the spin-orbit basis.
    args:
        mat: np.ndarray
            Matrix containing the NR integrals
        mult: int
            Multiplicity of the state
    returns:
        mat: np.ndarray of shape (k, n*deg, n*deg)
            deg = multiplicity
            k = 3 (x, y, z components)
    '''
    mat = _validate_component_matrix(mat)
    return np.array([np.kron(m, np.eye(mult)) for m in mat])

def spin_operators(S_val):
    """
    Return (Sx, Sy, Sz) as a numpy array of shape (3, n, n)
    for a given spin S (can be int or half-int).
    """
    if isinstance(S_val, (bool, np.bool_)) or not isinstance(S_val, Real):
        raise TypeError("S_val must be an integer or half-integer")
    if not np.isfinite(S_val) or S_val < 0:
        raise ValueError("S_val must be a nonnegative integer or half-integer")
    twice_spin = round(2 * float(S_val))
    if not np.isclose(2 * float(S_val), twice_spin, atol=1e-12, rtol=0.0):
        raise ValueError("S_val must be an integer or half-integer")

    S = sp.Rational(twice_spin, 2)
    hbar = sp.S(1)
    nstates = int(2 * S + 1)
    msvals = sorted([S - i for i in range(nstates)])
    dim = len(msvals)
    ms_index = {ms: i for i, ms in enumerate(msvals)}

    Sx = sp.Matrix.zeros(dim)
    Sy = sp.Matrix.zeros(dim)
    Sz = sp.Matrix.zeros(dim)

    for ms in msvals:
        i = ms_index[ms]
        Sz[i, i] = hbar * ms
        for delta in (+1, -1):
            ms_new = ms + delta
            if ms_new in ms_index:
                j = ms_index[ms_new]
                coeff = hbar * sp.sqrt(S * (S + 1) - ms * ms_new) / 2
                Sx[j, i] += coeff
                Sy[j, i] += -sp.I * coeff if delta == +1 else sp.I * coeff

    Sx_np = np.array(Sx.evalf(), dtype=np.complex128)
    Sy_np = np.array(Sy.evalf(), dtype=np.complex128)
    Sz_np = np.array(Sz.evalf(), dtype=np.complex128)
    return Sx_np, Sy_np, Sz_np

def _get_lxyz_integrals(mol, origin='CHARGE_CENTER'):
    '''
    Note these integrals are antisymm.
    '''
    center = get_guage_origin(mol,origin)
    with mol.with_common_orig(center):
        ints = mol.intor('int1e_cg_irxp', comp=3)
    return ints

def _get_dipole_integrals(mol, origin='CHARGE_CENTER'):
    center = get_guage_origin(mol,origin)
    with mol.with_common_orig(center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    return ao_dip

def _get_soc_integrals(mol, origin='CHARGE_CENTER', ham='DKH', somf=True,
                       amf=True, mmf=False, soc1e=True, soc2e=True, dm=None):
    hso = socaddons.socintegrals(mol, somf=somf, amf=amf, mmf=mmf,
                                 soc1e=soc1e, soc2e=soc2e, ham=ham, dm=dm)
    hso /= 1j
    return hso.real

def get_1e_prop(mc, modelspace, mysiso, origin='CHARGE_CENTER'):
    """
    Get the one-electron properties for the given model space.
    Basically it computes r"<Psi_i|O|Psi_j>" for the one electron
    operator O.
    args:
        mc: mcscf object
            SA-CAS or L-PDFT object
        modelspace: list
            List of tuples (nroots, mult, symm)
        mysiso: siso.SISO object
            SISO object containing sivec, sienergy, and ham
        origin: str
            Origin for the gauge-dependent integrals, default is
            'CHARGE_CENTER'.
    returns:
        orbangmoment: list
            List of orbital angular momentum matrices for each multiplicity group
        amfiinterac: list
            List of spin-orbit interaction matrices for each multiplicity group
        edipinterac: list
            List of electric dipole interaction matrices for each multiplicity group
    """
    if getattr(mysiso, 'mc', None) is not mc:
        raise ValueError("mysiso must be attached to mc")

    ncas = mc.ncas
    nelecas = mc.nelecas
    ham = mysiso.ham
    somf = mysiso.somf
    amf = mysiso.amf
    mmf = mysiso.mmf
    soc1e = mysiso.soc1e
    soc2e = mysiso.soc2e
    dm = None
    if mmf:
        dm = mc.make_rdm1()
    mo_cas = mc.mo_coeff[:, mc.ncore:mc.ncas+mc.ncore]
    ints_mo = _basis_transformation(
        _get_lxyz_integrals(mc._scf.mol, origin), mo_cas)
    ints_so = _basis_transformation(_get_soc_integrals(mc._scf.mol, origin, ham=ham,
                                                       somf=somf, amf=amf, mmf=mmf, soc1e=soc1e,
                                                       soc2e=soc2e, dm=dm), mo_cas)
    ints_dip = _basis_transformation(
        _get_dipole_integrals(mc._scf.mol, origin), mo_cas)

    symmetry_modelspace = socaddons._validate_modelspace(
        modelspace, mol=mc._scf.mol, ncas=mc.ncas, nelecas=mc.nelecas)
    modelspace = socaddons._aggregate_modelspace(symmetry_modelspace)

    expected_nroots = sum(state[0] for state in symmetry_modelspace)
    if isinstance(mc.ci, (list, tuple)):
        ci_roots = list(mc.ci)
    elif expected_nroots == 1:
        ci_roots = [mc.ci]
    else:
        raise ValueError("mc.ci must contain one CI vector per model-space root")
    if len(ci_roots) != expected_nroots:
        raise ValueError("modelspace root count does not match mc.ci")

    solvers = getattr(mc.fcisolver, 'fcisolvers', None)
    if solvers is None or len(solvers) != len(symmetry_modelspace):
        raise ValueError(
            "mc.fcisolver must contain one solver per modelspace entry")

    # A transition-density routine does not depend on the wfnsym used to obtain
    # the CI vector. Use one solver for each spin group so that matrix elements
    # between different irreps of the same spin are retained.
    solver_by_spin = {}
    for solver, (_, spinmult, _) in zip(solvers, symmetry_modelspace):
        solver_by_spin.setdefault(spinmult, solver)

    orbangmoment = []
    amfiinterac = []
    edipinterac = []

    nroot0 = 0
    for nroots, imult in modelspace:
        solver = solver_by_spin[imult]

        orbLmat = np.zeros((3, nroots, nroots), dtype=ints_mo.dtype)
        amfimat = np.zeros((3, nroots, nroots), dtype=ints_mo.dtype)
        edipmat = np.zeros((3, nroots, nroots), dtype=ints_mo.dtype)

        ci_slice = ci_roots[nroot0:nroot0+nroots]

        ijpairs = list(product(range(nroots), repeat=2))
        for i, j in ijpairs:
            tdm1 = solver.trans_rdm1(ci_slice[i], ci_slice[j],ncas, nelecas)
            orbLmat[:, i, j] = np.tensordot(ints_mo, tdm1, axes=([1, 2], [0, 1])).real
            amfimat[:, i, j] = np.tensordot(ints_so, tdm1, axes=([1, 2], [0, 1])).real
            edipmat[:, i, j] = -np.tensordot(ints_dip, tdm1, axes=([1, 2], [0, 1])).real

        orbangmoment.append(orbLmat)
        amfiinterac.append(amfimat)
        edipinterac.append(edipmat)

        nroot0 += nroots

    return orbangmoment, amfiinterac, edipinterac

def generate_aniso_data(mol, mc, modelspace, mysiso, origin='CHARGE_CENTER',
                        ham=None):
    '''
    args:
        mol: instance of mol.gto
            Molecule object containing the molecular geometry and basis set
        mc: mcscf object
            SA-CAS or L-PDFT
        modelspace: list
            List of tuples (nroots, mult, symm)
        mysiso: siso.SISO object
            SISO object containing sivec, sienergy, and ham
        origin: str
            Origin for the integrals, default is 'CHARGE_CENTER'
        ham: str or None
            SOC Hamiltonian, 'BP' or 'DKH'. If None, use ``mysiso.ham``.
            An explicitly supplied value must match ``mysiso.ham``.
    returns:
        data: dict
            Dictionary containing the required data for ANISO calculations
    '''
    if mol is not mc._scf.mol:
        raise ValueError("mol must be the molecule attached to mc._scf")
    if getattr(mysiso, 'mc', None) is not mc:
        raise ValueError("mysiso must be attached to mc")

    modelspace = socaddons._validate_modelspace(
        modelspace, mol=mc._scf.mol, ncas=mc.ncas, nelecas=mc.nelecas)
    aggregate_modelspace = socaddons._aggregate_modelspace(modelspace)

    nstate = sum(nroots for nroots, _, _ in modelspace)
    nss = sum(nroots * spinmult for nroots, spinmult, _ in modelspace)
    spin_free_energies = np.asarray(mc.e_states)
    si_energies = np.asarray(mysiso.si_energies)
    si_vecs = np.asarray(mysiso.si_vecs)
    if spin_free_energies.shape != (nstate,):
        raise ValueError(f"mc.e_states must have shape ({nstate},)")
    if si_energies.shape != (nss,):
        raise ValueError(f"mysiso.si_energies must have shape ({nss},)")
    if si_vecs.shape != (nss, nss):
        raise ValueError(f"mysiso.si_vecs must have shape ({nss}, {nss})")

    if ham is None:
        ham = mysiso.ham
    elif not isinstance(ham, str) or ham.upper() not in ('BP', 'DKH'):
        raise ValueError("ham must be 'BP' or 'DKH'")
    else:
        ham = ham.upper()
    if ham != mysiso.ham:
        raise ValueError(
            f"ham={ham!r} does not match the SISO Hamiltonian "
            f"{mysiso.ham!r}")

    # From the state vectors and energies construct the Hamiltonian
    # Spin-orbit Hamiltonian matrix
    hso = si_vecs @ np.diag(si_energies) @ si_vecs.conj().T
    data = {}

    # Basic headings
    heading = 'PySCF Interface to SINGLE_ANISO'
    data['source'] = heading
    data['format'] =  '2021'

    # Geometrical data
    data['natoms'] =  int(mol.natm)
    atomlabels = [mol.atom_symbol(i) for i in range(mol.natm)]
    data['atomlbl'] =  atomlabels
    coords = mol.atom_coords(unit='Angstrom')
    atom_list = [[i, label, coord[0], coord[1], coord[2]]
        for i, (label, coord) in enumerate(zip(atomlabels, coords), 1)]
    atom_list.insert(0, [mol.natm])
    data['coords (in angstrom)'] = atom_list

    # Model space data
    modelspacearr = np.asarray(aggregate_modelspace, dtype=int)
    nroots, imult = modelspacearr.T
    szproj = np.concatenate([
        np.tile(get_twice_ms_values(m), n) for n, m in modelspacearr
    ])
    multiplicity = np.array(np.repeat(imult, nroots), dtype=int)
    data['nss'] = nss
    data['nstate'] = nstate
    data['nmult'] = int(len(modelspacearr))
    data['imult'] = [int(x) for x in imult]
    data['nroot'] = [int(r) for r in nroots]
    data['szproj'] = [int(x) for x in szproj]
    data['multiplicity'] = [int(x) for x in multiplicity]

    # Energy data
    data['eso'] = si_energies
    data['esfs'] = spin_free_energies

    # Generate the required operators
    sfs_lmat, sfs_amfi, sfs_edip = get_1e_prop(mc, modelspace, mysiso, origin)

    sos_spin = []
    sos_magmom = []
    sos_edipmat = []
    for i, (nroots, mult) in enumerate(modelspacearr):
        spinstates = [(mult-1)/2 for _ in range(nroots)]
        spininter = [np.stack(spin_operators(spin), axis=0) for spin in spinstates]
        lmatsos = generate_sos_basis(sfs_lmat[i], mult)
        edipsos = generate_sos_basis(sfs_edip[i], mult)

        sos_edipmat.append(edipsos)

        sos_spin.append(np.stack([block_diag(*[a[i] for a in spininter]) for i in range(3)], axis=0))

        sos_magneticmoment_ = -nist.G_ELECTRON * np.stack([
            block_diag(*[a[i] for a in spininter]) for i in range(3)
        ], axis=0)
        sos_magneticmoment_ -= 1j * lmatsos
        sos_magmom.append(sos_magneticmoment_)

    # Spin orbit free data
    sfs_lmat = unpack_sfs_basis(sfs_lmat)
    sfs_amfi = unpack_sfs_basis(sfs_amfi)
    sfs_edip = unpack_sfs_basis(sfs_edip)

    data['angmom_x'] = sfs_lmat[0]
    data['angmom_y'] = sfs_lmat[1]
    data['angmom_z'] = sfs_lmat[2]
    data['amfi_x'] = sfs_amfi[0]
    data['amfi_y'] = sfs_amfi[1]
    data['amfi_z'] = sfs_amfi[2]
    data['edmom_x'] = sfs_edip[0]
    data['edmom_y'] = sfs_edip[1]
    data['edmom_z'] = sfs_edip[2]

    # Spin orbit coupled data
    sivec = si_vecs
    sos_edipmat = unpack_sos_basis(sos_edipmat)
    sos_spin = unpack_sos_basis(sos_spin)
    sos_magneticmoment = unpack_sos_basis(sos_magmom)

    sos_spin = _basis_transformation(sos_spin, sivec)
    sos_magneticmoment = _basis_transformation(sos_magneticmoment, sivec)
    sos_edipmat = _basis_transformation(sos_edipmat, sivec)

    data['magn_xr'] = sos_magneticmoment[0].real
    data['magn_xi'] = sos_magneticmoment[0].imag
    data['magn_yr'] = sos_magneticmoment[1].real
    data['magn_yi'] = sos_magneticmoment[1].imag
    data['magn_zr'] = sos_magneticmoment[2].real
    data['magn_zi'] = sos_magneticmoment[2].imag
    data['spin_xr'] = sos_spin[0].real
    data['spin_xi'] = sos_spin[0].imag
    data['spin_yr'] = sos_spin[1].real
    data['spin_yi'] = sos_spin[1].imag
    data['spin_zr'] = sos_spin[2].real
    data['spin_zi'] = sos_spin[2].imag
    data['edipm_xr'] = sos_edipmat[0].real
    data['edipm_xi'] = sos_edipmat[0].imag
    data['edipm_yr'] = sos_edipmat[1].real
    data['edipm_yi'] = sos_edipmat[1].imag
    data['edipm_zr'] = sos_edipmat[2].real
    data['edipm_zi'] = sos_edipmat[2].imag

    # Hamiltonian data
    data['eigenr'] = si_vecs.real
    data['eigeni'] = si_vecs.imag
    data['hsor'] = hso.real
    data['hsoi'] = hso.imag
    return data

class ANISOFileWriter:
    '''
    This class provides methods to write various sections of the ANISO file
    including source, format, number of atoms, atom labels, coordinates,
    and various properties related to the calculation.
    '''
    def __init__(self, filename, data):
        '''
        args:
            filename (str): The name of the file to write.
            data (dict): A dictionary containing the data to write.
                          The keys should match the expected ANISO file format.
        '''
        if not isinstance(filename, (str, PathLike)):
            raise TypeError("filename must be a string or path-like object")
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
        self.filename = filename
        self.data = data

    def write_general(self, ky, val):
        '''
        args:
            ky (str): The key for the data to write.
            val (any): The value associated with the key.
        returns:
            str: Formatted string in ASCII format.
        '''

        if not isinstance(ky, str) or not ky or '\n' in ky or ky.startswith('$'):
            raise ValueError("ANISO keys must be non-empty strings without '$' or newlines")

        if isinstance(val, (bool, np.bool_)):
            raise TypeError(f"Unsupported boolean value for key: {ky}")
        if isinstance(val, (Integral, np.integer)):
            return f"${ky}\n{val}\n\n"

        if isinstance(val, Real):
            if not np.isfinite(val):
                raise ValueError(f"Non-finite value for key: {ky}")
            return f"${ky}\n{val}\n\n"

        if isinstance(val, str):
            return f"${ky}\n{val}\n\n"

        if isinstance(val, list):
            if not val:
                raise ValueError(f"List for key {ky} must not be empty")
            if all(isinstance(i, Real) and not isinstance(i, (bool, np.bool_))
                   for i in val):
                if not np.all(np.isfinite(val)):
                    raise ValueError(f"Non-finite list value for key: {ky}")
                return f"${ky}\n{len(val)}\n{' '.join(map(str, val))}\n\n"

            if all(isinstance(i, list) and i for i in val):
                if not all(
                        isinstance(item, (str, Real)) and
                        not isinstance(item, (bool, np.bool_))
                        for sublist in val for item in sublist):
                    raise ValueError(f"Unsupported nested list for key: {ky}")
                numeric_items = [
                    item for sublist in val for item in sublist
                    if isinstance(item, Real)
                ]
                if numeric_items and not np.all(np.isfinite(numeric_items)):
                    raise ValueError(f"Non-finite list value for key: {ky}")
                val_str = '\n'.join(' '.join(map(str, sublist)) for sublist in val)
                return f"${ky}\n{val_str}\n\n"

            if all(isinstance(i, str) for i in val):
                return f"${ky}\n{len(val)}\n{' '.join(val)}\n\n"

            raise ValueError(f"Unsupported list format for key: {ky}")

        if isinstance(val, np.ndarray):
            if val.ndim not in (1, 2):
                raise ValueError(f"Unsupported ndarray dimension for key: {ky}")
            if val.size == 0:
                raise ValueError(f"Array for key {ky} must not be empty")
            if (not np.issubdtype(val.dtype, np.number) or
                    np.issubdtype(val.dtype, np.complexfloating)):
                raise TypeError(f"Array for key {ky} must be real numeric data")
            if not np.all(np.isfinite(val)):
                raise ValueError(f"Non-finite array value for key: {ky}")
            shape = list(val.shape)
            lines = []
            arr = val
            if arr.ndim == 1:
                for i in range(0, arr.shape[0], 5):
                    line = ' '.join(f"{v:22.14E}" for v in arr[i:i+5])
                    lines.append(line)
            elif arr.ndim == 2:
                for row in arr:
                    row_strs = []
                    for i in range(0, len(row), 5):
                        row_strs.append(' '.join(f"{v:22.14E}" for v in row[i:i+5]))
                    lines.append('\n'.join(row_strs))
            return f"${ky}\n{' '.join(map(str, shape))}\n" + '\n'.join(lines) + '\n\n'

        raise TypeError(f"Unsupported data type for key: {ky}")

    def save_to_file(self):
        with open(self.filename, 'w', encoding='ascii') as f:
            for ky, val in self.data.items():
                f.write(self.write_general(ky, val))

def write_aniso_file(filename, data, backend='OpenMolcas'):
    '''
    Based on the backend of the SINGLE_ANISO, write the ANISO file.
    args:
        filename: str
            Name of ANISO file
        data: dict
            Data to write to the ANISO file
        backend: str
            SINGLE_ANISO backend (OpenMolcas or Orca)
    '''
    if not isinstance(backend, str):
        raise TypeError("backend must be 'OpenMolcas' or 'Orca'")
    if backend == 'OpenMolcas':
        pass
    elif backend == 'Orca':
        rename_map = {"angmom_x": "angmom_xi", "angmom_y": "angmom_yi",
                      "angmom_z": "angmom_zi"}
        data = {rename_map.get(k, k): v for k, v in data.items()}

        _keys={'format','nss','nstate','nmult','imult','nroot','szproj',
               'multiplicity','eso','esfs','angmom_xi','angmom_yi','angmom_zi',
               'amfi_x','amfi_y','amfi_z','edmom_x','edmom_y','edmom_z','magn_xr',
               'magn_xi','magn_yr','magn_yi','magn_zr','magn_zi','spin_xr','spin_xi',
               'spin_yr','spin_yi','spin_zr','spin_zi','edipm_xr','edipm_xi','edipm_yr',
               'edipm_yi','edipm_zr','edipm_zi','eigenr','eigeni','hsor','hsoi'}
        data = {k: v for k, v in data.items() if k in _keys}
    else:
        raise ValueError(f"Unknown SINGLE_ANISO backend: {backend}")

    writer = ANISOFileWriter(filename, data)
    writer.save_to_file()
