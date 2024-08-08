#!/usr/bin/env python

'''
TREX-IO interface

References:
    https://trex-coe.github.io/trexio/trex.html
    https://github.com/TREX-CoE/trexio-tutorials/blob/ad5c60aa6a7bca802c5918cef2aeb4debfa9f134/notebooks/tutorial_benzene.md

Installation instruction:
    https://github.com/TREX-CoE/trexio/blob/master/python/README.md
'''

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import scf
import trexio

def to_trexio(obj, filename, backend='h5'):
    with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
        if isinstance(obj, gto.Mole):
            _mol_to_trexio(obj, tf)
        elif isinstance(obj, scf.hf.SCF):
            _scf_to_trexio(obj, tf)
        else:
            raise NotImplementedError(f'Conversion function for {obj.__class__}')

def _mol_to_trexio(mol, trexio_file):
    trexio.write_nucleus_num(trexio_file, mol.natm)

    labels = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    trexio.write_nucleus_label(trexio_file, labels)
    if mol._ecp:
        raise NotImplementedError
    trexio.write_nucleus_charge(trexio_file, mol.atom_charges())
    trexio.write_nucleus_coord(trexio_file, mol.atom_coords())

    trexio.write_basis_type(trexio_file, 'Gaussian')
    trexio.write_basis_shell_num(trexio_file, mol.nbas)
    trexio.write_basis_prim_num(trexio_file, int(mol._bas[:,gto.NPRIM_OF].sum()))
    trexio.write_basis_nucleus_index(trexio_file, mol._bas[:,gto.ATOM_OF])
    trexio.write_basis_shell_ang_mom(trexio_file, mol._bas[:,gto.ANG_OF])
    trexio.write_basis_shell_factor(trexio_file, np.ones(mol.nbas))
    prim2sh = [[ib]*nprim for ib, nprim in enumerate(mol._bas[:,gto.NPRIM_OF])]
    trexio.write_basis_shell_index(trexio_file, np.hstack(prim2sh))
    trexio.write_basis_exponent(trexio_file, np.hstack(mol.bas_exps()))
    assert all(mol._bas[:,gto.NCTR_OF] == 1)
    coef = [mol.bas_ctr_coeff(i).ravel() for i in range(mol.nbas)]
    trexio.write_basis_coefficient(trexio_file, np.hstack(coef))
    prim_norms = [gto.gto_norm(mol.bas_angular(i), mol.bas_exp(i))
                  for i in range(mol.nbas)]
    trexio.write_basis_prim_factor(trexio_file, np.hstack(prim_norms))

    if mol.symmetry:
        trexio.write_nucleus_point_group(trexio_file, mol.groupname)

    if mol.cart:
        trexio.write_ao_cartesian(trexio_file, 1)
    else:
        trexio.write_ao_cartesian(trexio_file, 0)

def _scf_to_trexio(mf, trexio_file):
    mol = mf.mol
    _mol_to_trexio(mol, trexio_file)
    if isinstance(mf, scf.uhf.UHF):
        mo_energy = np.ravel(mf.mo_energy)
        mo = np.hstack(*mf.mo_coeff)
        mo_occ = np.ravel(mf.mo_occ)
        spin = np.zeros(mo_energy.size, dtype=int)
        spin[mf.mo_energy[0].size:] = 1
        mo_type = 'UHF'
    else:
        mo_energy = mf.mo_energy
        mo = mf.mo_coeff
        mo_occ = mf.mo_occ
        spin = np.zeros(mo_energy.size, dtype=int)
        mo_type = 'RHF'
    trexio.write_mo_type(trexio_file, mo_type)
    idx = _order_ao_index(mf.mol)
    trexio.write_ao_num(trexio_file, int(mol.nao))
    trexio.write_mo_num(trexio_file, mo_energy.size)
    trexio.write_mo_coefficient(trexio_file, mo[idx].T.ravel())
    trexio.write_mo_energy(trexio_file, mo_energy)
    trexio.write_mo_occupation(trexio_file, mo_occ)
    trexio.write_mo_spin(trexio_file, spin)

def _cc_to_trexio(cc_obj, trexio_file):
    raise NotImplementedError

def _mcscf_to_trexio(cas_obj, trexio_file):
    raise NotImplementedError

def mol_from_trexio(filename, backend='h5'):
    mol = gto.Mole()
    with trexio.File(filename, 'r', back_end=_mode(backend)) as tf:
        assert trexio.read_basis_type(tf) == 'Gaussian'
        if trexio.has_ecp(tf):
            raise NotImplementedError
        labels = trexio.read_nucleus_label(tf)
        coords = trexio.read_nucleus_coord(tf)
        elements = [s+str(i) for i, s in enumerate(labels)]
        mol.atom = list(zip(elements, coords))
        mol.unit = 'Bohr'
        if trexio.has_ao_cartesian(tf):
            mol.cart = trexio.read_ao_cartesian(tf) == 1

        if trexio.has_nucleus_point_group(tf):
            mol.symmetry = trexio.read_nucleus_point_group(tf)

        nuc_idx = trexio.read_basis_nucleus_index(tf)
        ls = trexio.read_basis_shell_ang_mom(tf)
        prim2sh = trexio.read_basis_shell_index(tf)
        exps = trexio.read_basis_exponent(tf)
        coef = trexio.read_basis_coefficient(tf)

    basis = {}
    exps = _group_by(exps, prim2sh)
    coef = _group_by(coef, prim2sh)
    p1 = 0
    for ia, at_ls in enumerate(_group_by(ls, nuc_idx)):
        p0, p1 = p1, p1 + at_ls.size
        at_basis = [[l, *zip(e, c)]
                    for l, e, c in zip(ls[p0:p1], exps[p0:p1], coef[p0:p1])]
        basis[elements[ia]] = at_basis

    # To avoid the mol.build() sort the basis, disable mol.basis and set the
    # internal data _basis directly.
    mol.basis = {}
    mol._basis = basis
    return mol.build()

def scf_from_trexio(filename, backend='h5'):
    mol = mol_from_trexio(filename, backend)
    with trexio.File(filename, 'r', back_end=_mode(backend)) as tf:
        mo_energy = trexio.read_mo_energy(tf)
        mo        = trexio.read_mo_coefficient(tf)
        mo_occ    = trexio.read_mo_occupation(tf)

    nao = mol.nao
    assert mo.shape == (nao, nao) # RHF only
    mf = mol.RHF()
    mf.mo_coeff = np.empty_like(mo).T
    idx = _order_ao_index(mol)
    mf.mo_coeff[idx] = mo.T
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    return mf

def write_eri(eri, filename, backend='h5'):
    num_integrals = eri.size
    if eri.ndim == 4:
        n = eri.shape[0]
        idx = lib.cartesian_prod([np.arange(n, dtype=np.int32)]*4)
    elif eri.ndim == 2: # 4-fold symmetry
        npair = eri.shape[0]
        n = int((npair * 2)**.5)
        idx_pair = np.argwhere(np.arange(n)[:,None] >= np.arange(n))
        idx = np.empty((npair, npair, 4), dtype=np.int32)
        idx[:,:,:2] = idx_pair[:,None,:]
        idx[:,:,2:] = idx_pair[None,:,:]
        idx = idx.reshape(npair**2, 4)
    elif eri.ndim == 1: # 8-fold symmetry
        npair = int((eri.shape[0] * 2)**.5)
        n = int((npair * 2)**.5)
        idx_pair = np.argwhere(np.arange(n)[:,None] >= np.arange(n))
        idx = np.empty((npair, npair, 4), dtype=np.int32)
        idx[:,:,:2] = idx_pair[:,None,:]
        idx[:,:,2:] = idx_pair[None,:,:]
        idx = idx[np.tril_indices(npair)]

    with trexio.File(filename, 'w', back_end=_mode(backend)) as tf:
        trexio.write_mo_2e_int_eri(tf, 0, num_integrals, idx, eri.ravel())

def read_eri(filename, backend='h5'):
    '''Read ERIs in AO basis, 8-fold symmetry is assumed'''
    with trexio.File(filename, 'r', back_end=_mode(backend)) as tf:
        nmo = trexio.read_mo_num(tf)
        nao_pair = nmo * (nmo+1) // 2
        eri_size = nao_pair * (nao_pair+1) // 2
        idx, data, n_read, eof_flag = trexio.read_mo_2e_int_eri(tf, 0, eri_size)
    eri = np.zeros(eri_size)
    x = idx[:,0]*(idx[:,0]+1)//2 + idx[:,1]
    y = idx[:,2]*(idx[:,2]+1)//2 + idx[:,3]
    eri[x*(x+1)//2+y] = data
    return eri

def _order_ao_index(mol):
    if mol.cart:
        return np.arange(mol.nao)

    # reorder spherical functions to
    # Pz, Px, Py
    # D 0, D+1, D-1, D+2, D-2
    # F 0, F+1, F-1, F+2, F-2, F+3, F-3
    # G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
    # ...
    lmax = mol._bas[:,gto.ANG_OF].max()
    cache_by_l = [np.array([0]), np.array([2, 0, 1])]
    for l in range(2, lmax + 1):
        idx = np.empty(l*2+1, dtype=int)
        idx[ ::2] = l - np.arange(0, l+1)
        idx[1::2] = np.arange(l+1, l+l+1)
        cache_by_l.append(idx)

    idx = []
    off = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for n in range(mol.bas_nctr(ib)):
            idx.append(cache_by_l[l] + off)
            off += l * 2 + 1
    return np.hstack(idx)

def _mode(code):
    if code == 'h5':
        return 0
    else: # trexio text
        return 1

def _group_by(a, keys):
    '''keys must be sorted'''
    assert all(keys[:-1] <= keys[1:])
    idx = np.unique(keys, return_index=True)[1]
    return np.split(a, idx[1:])
