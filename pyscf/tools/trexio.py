#!/usr/bin/env python

'''
TREX-IO interface

References:
    https://trex-coe.github.io/trexio/trex.html
    https://github.com/TREX-CoE/trexio-tutorials/blob/ad5c60aa6a7bca802c5918cef2aeb4debfa9f134/notebooks/tutorial_benzene.md

Installation instruction:
    https://github.com/TREX-CoE/trexio/blob/master/python/README.md
'''

import re
import math
import numpy as np
import scipy.linalg
from collections import defaultdict

import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import pbc
from pyscf import mcscf
from pyscf import fci
from pyscf.pbc import gto as pbcgto

import trexio

def to_trexio(obj, filename, backend='h5', ci_threshold=None, chunk_size=None):
    with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
        if isinstance(obj, gto.Mole) or isinstance(obj, pbcgto.Cell):
            _mol_to_trexio(obj, tf)
        elif isinstance(obj, scf.hf.SCF):
            _scf_to_trexio(obj, tf)
        elif isinstance(obj, mcscf.casci.CASCI) or isinstance(obj, mcscf.CASSCF):
            ci_threshold = ci_threshold if ci_threshold is not None else 0.
            chunk_size = chunk_size if chunk_size is not None else 100000
            _mcscf_to_trexio(obj, tf, ci_threshold=ci_threshold, chunk_size=chunk_size)
        else:
            raise NotImplementedError(f'Conversion function for {obj.__class__}')

def _mol_to_trexio(mol, trexio_file):
    # 1 Metadata
    trexio.write_metadata_code_num(trexio_file, 1)
    trexio.write_metadata_code(trexio_file, [f'PySCF-v{pyscf.__version__}'])
    #trexio.write_metadata_package_version(trexio_file, f'TREXIO-v{trexio.__version__}')

    # 2 System
    trexio.write_nucleus_num(trexio_file, mol.natm)
    nucleus_charge = [mol.atom_charge(i) for i in range(mol.natm)]
    trexio.write_nucleus_charge(trexio_file, nucleus_charge)
    trexio.write_nucleus_coord(trexio_file, mol.atom_coords())
    labels = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    trexio.write_nucleus_label(trexio_file, labels)
    if mol.symmetry:
        trexio.write_nucleus_point_group(trexio_file, mol.groupname)

    # 2.3 Periodic boundary calculations
    if isinstance(mol, pbc.gto.Cell):
        # 2.3 Periodic boundary calculations (pbc group)
        trexio.write_pbc_periodic(trexio_file, True)
        # 2.2 Cell
        lattice = mol.lattice_vectors()
        a = lattice[0] # unit is Bohr
        b = lattice[1] # unit is Bohr
        c = lattice[2] # unit is Bohr
        trexio.write_cell_a(trexio_file, a)
        trexio.write_cell_b(trexio_file, b)
        trexio.write_cell_c(trexio_file, c)
    else:
        # 2.3 Periodic boundary calculations (pbc group)
        trexio.write_pbc_periodic(trexio_file, False)

    # 2.4 Electron (electron group)
    electron_up_num, electron_dn_num = mol.nelec
    trexio.write_electron_num(trexio_file, electron_up_num + electron_dn_num )
    trexio.write_electron_up_num(trexio_file, electron_up_num)
    trexio.write_electron_dn_num(trexio_file, electron_dn_num)

    # 3.1 Basis set
    trexio.write_basis_type(trexio_file, 'Gaussian')
    if any(mol._bas[:,gto.NCTR_OF] > 1):
        mol = _to_segment_contraction(mol)
    trexio.write_basis_shell_num(trexio_file, mol.nbas)
    trexio.write_basis_prim_num(trexio_file, int(mol._bas[:,gto.NPRIM_OF].sum()))
    trexio.write_basis_nucleus_index(trexio_file, mol._bas[:,gto.ATOM_OF])
    trexio.write_basis_shell_ang_mom(trexio_file, mol._bas[:,gto.ANG_OF])
    trexio.write_basis_shell_factor(trexio_file, np.ones(mol.nbas))
    prim2sh = [[ib]*nprim for ib, nprim in enumerate(mol._bas[:,gto.NPRIM_OF])]
    trexio.write_basis_shell_index(trexio_file, np.hstack(prim2sh))
    trexio.write_basis_exponent(trexio_file, np.hstack(mol.bas_exps()))
    coef = [mol.bas_ctr_coeff(i).ravel() for i in range(mol.nbas)]
    trexio.write_basis_coefficient(trexio_file, np.hstack(coef))
    prim_norms = [gto.gto_norm(mol.bas_angular(i), mol.bas_exp(i)) for i in range(mol.nbas)]
    trexio.write_basis_prim_factor(trexio_file, np.hstack(prim_norms))

    # 3.2 Effective core potentials (ecp group)
    if len(mol._pseudo) > 0:
        raise NotImplementedError(
            "TREXIO does not support 'pseudo' format. Please use 'ecp'"
        )

    if mol._ecp:
        # internal format of pyscf is described in
        # https://pyscf.org/pyscf_api_docs/pyscf.gto.html?highlight=ecp#module-pyscf.gto.ecp

        ecp_num = 0
        ecp_max_ang_mom_plus_1 = []
        ecp_z_core = []
        ecp_nucleus_index = []
        ecp_ang_mom = []
        ecp_coefficient = []
        ecp_exponent = []
        ecp_power = []

        chemical_symbol_list = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
        atom_symbol_list = [mol.atom_symbol(i) for i in range(mol.natm)]

        for nuc_index, (chemical_symbol, atom_symbol) in enumerate(
            zip(chemical_symbol_list, atom_symbol_list)
        ):
            if re.match(r"X-.*", chemical_symbol) or re.match(r"X-.*", atom_symbol):
                ecp_num += 1
                ecp_max_ang_mom_plus_1.append(1)
                ecp_z_core.append(0)
                ecp_nucleus_index.append(nuc_index)
                ecp_ang_mom.append(0)
                ecp_coefficient.append(0.0)
                ecp_exponent.append(1.0)
                ecp_power.append(0)
                continue

            # atom_symbol is superior to atom_pure_symbol
            try:
                z_core, ecp_list = mol._ecp[atom_symbol]
            except KeyError:
                z_core, ecp_list = mol._ecp[chemical_symbol]

            # ecp zcore
            ecp_z_core.append(z_core)

            # max_ang_mom
            max_ang_mom = max([ecp[0] for ecp in ecp_list])
            if max_ang_mom == -1:
                # Special treatments are needed for H and He.
                # PySCF database does not define the ul-s part for these elements.
                dummy_ul_s_part = True
                max_ang_mom = 0
                max_ang_mom_plus_1 = 1
            else:
                dummy_ul_s_part = False
                max_ang_mom_plus_1 = max_ang_mom + 1

            ecp_max_ang_mom_plus_1.append(max_ang_mom_plus_1)

            for ecp in ecp_list:
                ang_mom = ecp[0]
                if ang_mom == -1:
                    ang_mom = max_ang_mom_plus_1
                for r, exp_coeff_list in enumerate(ecp[1]):
                    for exp_coeff in exp_coeff_list:
                        exp, coeff = exp_coeff
                        ecp_num += 1
                        ecp_nucleus_index.append(nuc_index)
                        ecp_ang_mom.append(ang_mom)
                        ecp_coefficient.append(coeff)
                        ecp_exponent.append(exp)
                        ecp_power.append(r - 2)

            if dummy_ul_s_part:
                # A dummy ECP is put for the ul-s part here for H and He atoms.
                ecp_num += 1
                ecp_nucleus_index.append(nuc_index)
                ecp_ang_mom.append(0)
                ecp_coefficient.append(0.0)
                ecp_exponent.append(1.0)
                ecp_power.append(0)

        trexio.write_ecp_num(trexio_file, ecp_num)
        trexio.write_ecp_max_ang_mom_plus_1(trexio_file, ecp_max_ang_mom_plus_1)
        trexio.write_ecp_z_core(trexio_file, ecp_z_core)
        trexio.write_ecp_nucleus_index(trexio_file, ecp_nucleus_index)
        trexio.write_ecp_ang_mom(trexio_file, ecp_ang_mom)
        trexio.write_ecp_coefficient(trexio_file, ecp_coefficient)
        trexio.write_ecp_exponent(trexio_file, ecp_exponent)
        trexio.write_ecp_power(trexio_file, ecp_power)

    # 4.1 Atomic orbitals (ao group)
    if mol.cart:
        trexio.write_ao_cartesian(trexio_file, 1)
        ao_shell = []
        ao_normalization = []
        for i, ang_mom in enumerate(mol._bas[:,gto.ANG_OF]):
            ao_shell += [i] * int((ang_mom+1) * (ang_mom+2) / 2)
            # note: PySCF(libintc) normalizes s and p only.
            if ang_mom == 0 or ang_mom == 1:
                ao_normalization += [float(np.sqrt((2*ang_mom+1)/(4*np.pi)))] * int((ang_mom+1) * (ang_mom+2) / 2)
            else:
                ao_normalization += [1.0] * int((ang_mom+1) * (ang_mom+2) / 2)
    else:
        trexio.write_ao_cartesian(trexio_file, 0)
        ao_shell = []
        ao_normalization = []
        for i, ang_mom in enumerate(mol._bas[:,gto.ANG_OF]):
            ao_shell += [i] * (2*ang_mom+1)
            # note: TREXIO employs the solid harmonics notation,; thus, we need these factors.
            ao_normalization += [float(np.sqrt((2*ang_mom+1)/(4*np.pi)))] * (2*ang_mom+1)

    trexio.write_ao_num(trexio_file, int(mol.nao))
    trexio.write_ao_shell(trexio_file, ao_shell)
    trexio.write_ao_normalization(trexio_file, ao_normalization)

def _scf_to_trexio(mf, trexio_file):
    mol = mf.mol
    _mol_to_trexio(mol, trexio_file)

    # PBC
    if isinstance(mol, pbc.gto.Cell):
        kpts = np.array(mf.kpts)
        kpts = mol.get_scaled_kpts(kpts)
        nk = len(mf.kpts)
        weights = np.full(nk, 1.0/nk)

        if nk == 1:
            # 2.3 Periodic boundary calculations (pbc group)
            trexio.write_pbc_k_point_num(trexio_file, 1)
            trexio.write_pbc_k_point(trexio_file, kpts)
            trexio.write_pbc_k_point_weight(trexio_file, weights[np.newaxis])

            if isinstance(mf, (pbc.scf.uhf.UHF, pbc.dft.uks.UKS, pbc.scf.kuhf.KUHF, pbc.dft.kuks.KUKS)):
                mo_type = 'UHF'
                if isinstance(mf, (pbc.scf.uhf.UHF, pbc.dft.uks.UKS)):
                    mo_energy = np.ravel(mf.mo_energy)
                    mo_num = mo_energy.size
                    mo_up, mo_dn = mf.mo_coeff
                elif isinstance(mf, (pbc.scf.kuhf.KUHF, pbc.dft.kuks.KUKS)):
                    mo_energy = np.ravel(mf.mo_energy)
                    mo_num = mo_energy.size
                    mo_up, mo_dn = mf.mo_coeff
                    mo_up = mo_up[0]
                    mo_dn = mo_dn[0]
                else:
                    raise NotImplementedError(f'Conversion function for {mf.__class__}')
                idx = _order_ao_index(mf.mol)
                mo_up = mo_up[idx].T
                mo_dn = mo_dn[idx].T
                num_mo_up = len(mo_up)
                num_mo_dn = len(mo_dn)
                assert num_mo_up + num_mo_dn == mo_num
                mo=np.concatenate([mo_up, mo_dn], axis=0) # dim (num_mo, num_ao) but it is f-contiguous
                mo_coefficient = np.ascontiguousarray(mo) # dim (num_mo, num_ao) and it is c-contiguous
                if np.all(np.isreal(mo_coefficient)):
                    mo_coefficient_real = mo_coefficient
                    mo_coefficient_imag = None
                else:
                    mo_coefficient_real = mo_coefficient.real
                    mo_coefficient_imag = mo_coefficient.imag
                mo_occ = np.ravel(mf.mo_occ)
                mo_spin = np.zeros(num_mo_up+num_mo_dn, dtype=int)
                mo_spin[:num_mo_up] = 0
                mo_spin[num_mo_up:] = 1

            elif isinstance(mf, (pbc.scf.krhf.KRHF, pbc.dft.krks.KRKS, pbc.scf.rhf.RHF, pbc.dft.rks.RKS)):
                mo_type = 'RHF'
                if isinstance(mf, (pbc.scf.rhf.RHF, pbc.dft.rks.RKS)):
                    mo_energy = np.ravel(mf.mo_energy)
                    mo_num = mo_energy.size
                    mo = mf.mo_coeff
                elif isinstance(mf, (pbc.scf.krhf.KRHF, pbc.dft.krks.KRKS)):
                    mo_energy = np.ravel(mf.mo_energy)
                    mo_num = mo_energy.size
                    mo = mf.mo_coeff[0]
                else:
                    raise NotImplementedError(f'Conversion function for {mf.__class__}')
                idx = _order_ao_index(mf.mol)
                mo = mo[idx].T # dim (num_mo, num_ao) but it is f-contiguous
                mo_coefficient = np.ascontiguousarray(mo) # dim (num_mo, num_ao) and it is c-contiguous
                if np.all(np.isreal(mo_coefficient)):
                    mo_coefficient_real = mo_coefficient
                    mo_coefficient_imag = None
                else:
                    mo_coefficient_real = mo_coefficient.real
                    mo_coefficient_imag = mo_coefficient.imag
                mo_occ = np.ravel(mf.mo_occ)
                mo_spin = np.zeros(mo_num, dtype=int)
            else:
                raise NotImplementedError(f'Conversion function for {mf.__class__}')

            # 4.2 Molecular orbitals (mo group)
            trexio.write_mo_type(trexio_file, mo_type)
            trexio.write_mo_num(trexio_file, mo_num)
            trexio.write_mo_k_point(trexio_file, [0]*mo_num)
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real)
            if mo_coefficient_imag is not None:
                trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag)
            trexio.write_mo_energy(trexio_file, mo_energy)
            trexio.write_mo_occupation(trexio_file, mo_occ)
            trexio.write_mo_spin(trexio_file, mo_spin)

        else:
            # 2.3 Periodic boundary calculations (pbc group)
            trexio.write_pbc_k_point_num(trexio_file, nk)
            trexio.write_pbc_k_point(trexio_file, kpts)
            trexio.write_pbc_k_point_weight(trexio_file, weights)

            # stack k-dependent molecular orbitals
            mo_k_point_pbc = []
            mo_num_pbc = 0
            mo_energy_pbc = []
            mo_coefficient_real_pbc = []
            mo_coefficient_imag_pbc = []
            mo_occ_pbc= []
            mo_spin_pbc = []

            if isinstance(mf, (pbc.scf.uhf.UHF, pbc.dft.uks.UKS, pbc.scf.kuhf.KUHF, pbc.dft.kuks.KUKS)):
                mo_type = 'UHF'
                for i_k, _ in enumerate(kpts):
                    mo_energy = np.ravel(mf.mo_energy[i_k])
                    mo_num = mo_energy.size
                    mo_up, mo_dn = mf.mo_coeff[i_k]
                    idx = _order_ao_index(mf.mol)
                    mo_up = mo_up[idx].T
                    mo_dn = mo_dn[idx].T
                    mo=np.concatenate([mo_up, mo_dn], axis=0) # dim (num_mo, num_ao) but it is f-contiguous
                    mo_coefficient_real = np.ascontiguousarray(mo.real) # dim (num_mo, num_ao) and it is c-contiguous
                    mo_coefficient_imag = np.ascontiguousarray(mo.imag) # dim (num_mo, num_ao) and it is c-contiguous
                    mo_occ = np.ravel(mf.mo_occ[i_k])
                    mo_spin = np.zeros(mo_energy.size, dtype=int)
                    mo_spin[mf.mo_energy[i_k][0].size:] = 1

                    mo_k_point_pbc += [i_k] * mo_energy.size
                    mo_num_pbc += mo_energy.size
                    mo_energy_pbc.append(mo_energy)
                    mo_coefficient_real_pbc.append(mo_coefficient_real)
                    mo_coefficient_imag_pbc.append(mo_coefficient_imag)
                    mo_occ_pbc.append(mo_occ)
                    mo_spin_pbc.append(mo_spin)

            else:
                mo_type = 'RHF'
                for i_k, _ in enumerate(kpts):
                    mo_energy = mf.mo_energy[i_k]
                    mo = mf.mo_coeff[i_k]
                    idx = _order_ao_index(mf.mol)
                    mo = mo[idx].T
                    mo_coefficient_real = np.ascontiguousarray(mo.real) # dim (num_mo, num_ao) and it is c-contiguous
                    mo_coefficient_imag = np.ascontiguousarray(mo.imag) # dim (num_mo, num_ao) and it is c-contiguous
                    mo_occ = np.ravel(mf.mo_occ[i_k])
                    mo_spin = np.zeros(mo_energy.size, dtype=int)

                    mo_k_point_pbc += [i_k] * mo_energy.size
                    mo_num_pbc += mo_energy.size
                    mo_energy_pbc.append(mo_energy)
                    mo_coefficient_real_pbc.append(mo_coefficient_real)
                    mo_coefficient_imag_pbc.append(mo_coefficient_imag)
                    mo_occ_pbc.append(mo_occ)
                    mo_spin_pbc.append(mo_spin)

            # stack the results
            mo_k_point_pbc = np.array(mo_k_point_pbc)
            mo_energy_pbc = np.concatenate(mo_energy_pbc)
            mo_coefficient_real_pbc = np.ascontiguousarray(np.vstack(mo_coefficient_real_pbc)) # it is c-contiguous
            mo_coefficient_imag_pbc = np.ascontiguousarray(np.vstack(mo_coefficient_imag_pbc)) # it is c-contiguous
            mo_occ_pbc = np.concatenate(mo_occ_pbc)
            mo_spin_pbc = np.concatenate(mo_spin_pbc)

            # 4.2 Molecular orbitals (mo group)
            trexio.write_mo_type(trexio_file, mo_type)
            trexio.write_mo_num(trexio_file, mo_num_pbc)
            trexio.write_mo_k_point(trexio_file, mo_k_point_pbc)
            trexio.write_mo_energy(trexio_file, mo_energy_pbc)
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real_pbc)
            trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag_pbc)
            trexio.write_mo_occupation(trexio_file, mo_occ_pbc)
            trexio.write_mo_spin(trexio_file, mo_spin_pbc)

    # Open systems
    else:
        if isinstance(mf, (scf.uhf.UHF, dft.uks.UKS)):
            mo_type = 'UHF'
            mo_energy = np.ravel(mf.mo_energy)
            mo_num = mo_energy.size
            mo_up, mo_dn = mf.mo_coeff
            idx = _order_ao_index(mf.mol)
            mo_up = mo_up[idx].T
            mo_dn = mo_dn[idx].T
            mo=np.concatenate([mo_up, mo_dn], axis=0) # dim (num_mo, num_ao) but it is f-contiguous
            mo_coefficient = np.ascontiguousarray(mo) # dim (num_mo, num_ao) and it is c-contiguous
            mo_occ = np.ravel(mf.mo_occ)
            mo_spin = np.zeros(mo_energy.size, dtype=int)
            mo_spin[mf.mo_energy[0].size:] = 1
        else:
            mo_type = 'RHF'
            mo_energy = mf.mo_energy
            mo_num = mo_energy.size
            mo = mf.mo_coeff
            idx = _order_ao_index(mf.mol)
            mo = mo[idx].T # dim (num_mo, num_ao) but it is f-contiguous
            mo_coefficient = np.ascontiguousarray(mo) # dim (num_mo, num_ao) and it is c-contiguous
            mo_occ = mf.mo_occ
            mo_spin = np.zeros(mo_energy.size, dtype=int)

        # 4.2 Molecular orbitals (mo group)
        trexio.write_mo_type(trexio_file, mo_type)
        trexio.write_mo_num(trexio_file, mo_num)
        trexio.write_mo_coefficient(trexio_file, mo_coefficient)
        trexio.write_mo_energy(trexio_file, mo_energy)
        trexio.write_mo_occupation(trexio_file, mo_occ)
        trexio.write_mo_spin(trexio_file, mo_spin)

def _cc_to_trexio(cc_obj, trexio_file):
    raise NotImplementedError

def _mcscf_to_trexio(cas_obj, trexio_file, ci_threshold=0., chunk_size=100000):
    mol = cas_obj.mol
    _mol_to_trexio(mol, trexio_file)
    mo_energy_cas = cas_obj.mo_energy
    mo_cas = cas_obj.mo_coeff
    num_mo = mo_energy_cas.size
    spin_cas = np.zeros(mo_energy_cas.size, dtype=int)
    mo_type_cas = 'CAS'
    trexio.write_mo_type(trexio_file, mo_type_cas)
    idx = _order_ao_index(mol)
    trexio.write_mo_num(trexio_file, num_mo)
    trexio.write_mo_coefficient(trexio_file, mo_cas[idx].T.ravel())
    trexio.write_mo_energy(trexio_file, mo_energy_cas)
    trexio.write_mo_spin(trexio_file, spin_cas)

    ncore = cas_obj.ncore
    ncas = cas_obj.ncas
    mo_classes = np.array(["Virtual"] * num_mo, dtype=str)  # Initialize all MOs as Virtual
    mo_classes[:ncore] = "Core"
    mo_classes[ncore:ncore + ncas] = "Active"
    trexio.write_mo_class(trexio_file, list(mo_classes))

    occupation = np.zeros(num_mo)
    occupation[:ncore] = 2.0
    rdm1 = cas_obj.fcisolver.make_rdm1(cas_obj.ci, ncas, cas_obj.nelecas)
    natural_occ = np.linalg.eigh(rdm1)[0]
    occupation[ncore:ncore + ncas] = natural_occ[::-1]
    occupation[ncore + ncas:] = 0.0
    trexio.write_mo_occupation(trexio_file, occupation)

    total_elec_cas = sum(cas_obj.nelecas)

    _det_to_trexio(cas_obj, ncas, total_elec_cas, trexio_file, ci_threshold, chunk_size)

def mol_from_trexio(filename):
    with trexio.File(filename, 'r', back_end=trexio.TREXIO_AUTO) as tf:
        assert trexio.read_basis_type(tf) == 'Gaussian'
        pbc_periodic = trexio.read_pbc_periodic(tf)
        labels = trexio.read_nucleus_label(tf)
        coords = trexio.read_nucleus_coord(tf)
        elements = [s+str(i) for i, s in enumerate(labels)]

        if pbc_periodic:
            mol = pbcgto.Cell()
            mol.unit = 'Bohr'
            a = np.asarray(trexio.read_cell_a(tf), dtype=float)
            b = np.asarray(trexio.read_cell_b(tf), dtype=float)
            c = np.asarray(trexio.read_cell_c(tf), dtype=float)
            mol.a = np.vstack([a, b, c])
        else:
            mol = gto.Mole()
            mol.unit = 'Bohr'

        mol.atom = list(zip(elements, coords))
        up_num = trexio.read_electron_up_num(tf)
        dn_num = trexio.read_electron_dn_num(tf)
        spin = up_num - dn_num
        mol.spin = spin

        if trexio.has_ecp(tf):
            # --- read TREXIO ECP arrays ---
            z_core      = trexio.read_ecp_z_core(tf)                 # shape (natm,)
            max_l1_arr  = trexio.read_ecp_max_ang_mom_plus_1(tf)     # shape (natm,)
            nuc_idx     = trexio.read_ecp_nucleus_index(tf)          # shape (ecp_num,)
            ang_mom_enc = trexio.read_ecp_ang_mom(tf)                # shape (ecp_num,)
            coeff_arr   = trexio.read_ecp_coefficient(tf)            # shape (ecp_num,)
            exp_arr     = trexio.read_ecp_exponent(tf)               # shape (ecp_num,)
            power_arr   = trexio.read_ecp_power(tf)                  # shape (ecp_num,)

            # --- aggregate primitives per (nucleus, l); decode local channel to l = -1 ---
            per_atom = defaultdict(lambda: defaultdict(list))  # per_atom[n][l] -> [(power, exp, coeff), ...]
            for k in range(len(nuc_idx)):
                n   = int(nuc_idx[k])
                l_e = int(ang_mom_enc[k])
                # local channel was encoded as max_l1_arr[n]; decode to l = -1
                l_d = -1 if l_e == int(max_l1_arr[n]) else l_e
                p   = int(power_arr[k])                # stored as r-2 on write
                e   = float(exp_arr[k])
                c   = float(coeff_arr[k])
                per_atom[n][l_d].append((p, e, c))

            def _is_dummy_ul_s(items):
                # H/He dummy record injected at write time: (power=0, exp=1.0, coeff=0.0)
                return (len(items) == 1 and items[0][0] == 0
                        and abs(items[0][1] - 1.0) < 1e-12 and abs(items[0][2]) < 1e-300)

            ecp_dict = {}
            for n, raw_sym in enumerate(labels):
                # skip ghosts: ECP not applied to 'X-*' atoms
                if re.match(r"X-.*", raw_sym):
                    continue

                zc   = int(z_core[n]) if n < len(z_core) else 0
                lmap = dict(per_atom.get(n, {}))

                # drop intentional dummy ul-s for H/He if present
                if 0 in lmap and _is_dummy_ul_s(lmap[0]):
                    lmap.pop(0)

                # nothing to assign if no channels and no core reduction
                if not lmap and zc == 0:
                    continue

                # Build PySCF format for this atom:
                # for each l, buckets[r] holds (exp, coeff), where r = power + 2
                at_list = []
                # ordering is not strictly required, but keep nonlocal (l>=0) then local (l=-1) last
                for l in sorted(lmap.keys(), key=lambda x: (x == -1, x)):
                    items = lmap[l]
                    if not items:
                        continue
                    r_list = [p + 2 for p, _, _ in items]
                    max_r  = max(r_list)
                    buckets = [[] for _ in range(max_r + 1)]  # r in [0..max_r]
                    for (p, e, c), r in zip(items, r_list):
                        if r < 0:
                            # should not happen if power = r-2 and r>=0
                            continue
                        buckets[r].append((e, c))
                    at_list.append([int(l), buckets])

                # ★ Use the exact same per-atom label as in mol.atom / mol._basis
                key = elements[n]
                ecp_dict[key] = (zc, at_list)

            if ecp_dict:
                mol.ecp = ecp_dict

        if trexio.has_ao_cartesian(tf):
            mol.cart = trexio.read_ao_cartesian(tf) == 1

        if trexio.has_nucleus_point_group(tf):
            mol.symmetry = trexio.read_nucleus_point_group(tf)

        nuc_idx = trexio.read_basis_nucleus_index(tf).tolist()
        ls = trexio.read_basis_shell_ang_mom(tf).tolist()
        prim2sh = trexio.read_basis_shell_index(tf).tolist()
        exps = trexio.read_basis_exponent(tf).tolist()
        coef = trexio.read_basis_coefficient(tf).tolist()

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

def scf_from_trexio(filename):
    mol = mol_from_trexio(filename)

    if isinstance(mol, pbcgto.Cell):
        # PBC case
        with trexio.File(filename, 'r', back_end=trexio.TREXIO_AUTO) as tf:
            k_point_num = trexio.read_pbc_k_point_num(tf)

        # Single WF
        if k_point_num == 1:
            with trexio.File(filename, 'r', back_end=trexio.TREXIO_AUTO) as tf:
                kpts = trexio.read_pbc_k_point(tf)
                num_mo    = trexio.read_mo_num(tf)
                mo_energy = trexio.read_mo_energy(tf)
                mo_coeff  = trexio.read_mo_coefficient(tf)
                if trexio.has_mo_coefficient_im(tf):
                    mo_coeff_imag = trexio.read_mo_coefficient_im(tf)
                else:
                    mo_coeff_imag = None
                mo_occ    = trexio.read_mo_occupation(tf)
                mo_spin    = trexio.read_mo_spin(tf)

            nao = mol.nao
            idx = _order_ao_index(mol)
            uniq = np.unique(mo_spin)

            # UHF
            if set(uniq.tolist()) == {0, 1}:
                i_up = np.where(mo_spin == 0)[0]
                i_dn = np.where(mo_spin == 1)[0]

                if mo_coeff_imag is not None:
                    mo_up_out = mo_coeff[i_up, :]  # (nmo_up, nao)
                    mo_dn_out = mo_coeff[i_dn, :]  # (nmo_dn, nao)
                    mo_up_imag_out = mo_coeff_imag[i_up, :]  # (nmo_up, nao)
                    mo_dn_imag_out = mo_coeff_imag[i_dn, :]  # (nmo_dn, nao)
                    mo_up_pyscf = np.empty((nao, mo_up_out.shape[0]), dtype=(mo_coeff+1j*mo_coeff_imag).dtype)
                    mo_dn_pyscf = np.empty((nao, mo_dn_out.shape[0]), dtype=(mo_coeff+1j*mo_coeff_imag).dtype)
                    mo_up_pyscf[idx, :] = mo_up_out.T + 1j * mo_up_imag_out.T
                    mo_dn_pyscf[idx, :] = mo_dn_out.T + 1j * mo_dn_imag_out.T
                else:
                    mo_up_out = mo_coeff[i_up, :]  # (nmo_up, nao)
                    mo_dn_out = mo_coeff[i_dn, :]  # (nmo_dn, nao)
                    mo_up_pyscf = np.empty((nao, mo_up_out.shape[0]), dtype=mo_coeff.dtype)
                    mo_dn_pyscf = np.empty((nao, mo_dn_out.shape[0]), dtype=mo_coeff.dtype)
                    mo_up_pyscf[idx, :] = mo_up_out.T
                    mo_dn_pyscf[idx, :] = mo_dn_out.T

                mf = mol.UHF(kpt=kpts[0])
                mf.mo_coeff  = (mo_up_pyscf, mo_dn_pyscf)
                mf.mo_energy = (mo_energy[i_up], mo_energy[i_dn])
                mf.mo_occ    = (mo_occ[i_up],   mo_occ[i_dn])
                return mf

            # RHF
            elif set(uniq.tolist()) == {0} or set(uniq.tolist()) == {1}:
                mf = mol.RHF(kpt=kpts[0])
                if mo_coeff_imag is not None:
                    mf.mo_coeff = np.empty((nao, num_mo), dtype=(mo_coeff + 1j * mo_coeff_imag).dtype)
                    mf.mo_coeff[idx, :] = mo_coeff.T + 1j * mo_coeff_imag.T
                else:
                    mf.mo_coeff = np.empty((nao, num_mo), dtype=mo_coeff.dtype)
                    mf.mo_coeff[idx, :] = mo_coeff.T
                mf.mo_energy = mo_energy
                mf.mo_occ    = mo_occ
                return mf

            else:
                raise ValueError(f'Unknown spin multiplicity {uniq}')

        # Multi WFs
        else:
            with trexio.File(filename, 'r', back_end=trexio.TREXIO_AUTO) as tf:
                kpts = trexio.read_pbc_k_point(tf)
                num_mo    = trexio.read_mo_num(tf)
                mo_energy = trexio.read_mo_energy(tf)
                mo_coeff  = trexio.read_mo_coefficient(tf)
                mo_coeff_imag = trexio.read_mo_coefficient_im(tf)
                mo_occ    = trexio.read_mo_occupation(tf)
                mo_spin    = trexio.read_mo_spin(tf)
                mo_k_point = trexio.read_mo_k_point(tf)

            nao = mol.nao
            idx = _order_ao_index(mol)
            uniq = np.unique(mo_spin)

            # UHF
            if set(uniq.tolist()) == {0, 1}:
                mo_energy_all_k = []
                mo_occ_all_k = []
                mo_coeff_all_k = []

                for i_k in range(k_point_num):
                    mask = (mo_k_point == i_k)
                    mo_energy_k = mo_energy[mask]
                    mo_occ_k    = mo_occ[mask]
                    mo_spin_k   = mo_spin[mask]
                    mo_coeff_k = mo_coeff[mask, :]
                    mo_coeff_k_imag = mo_coeff_imag[mask, :]

                    mask_up = (mo_spin_k == 0)   # alpha
                    mask_dn = (mo_spin_k == 1)   # beta

                    mo_energy_k_up = mo_energy_k[mask_up]
                    mo_occ_k_up    = mo_occ_k[mask_up]
                    mo_coeff_k_up  = mo_coeff_k[mask_up, :]
                    mo_coeff_k_imag_up = mo_coeff_k_imag[mask_up, :]

                    mo_energy_k_dn = mo_energy_k[mask_dn]
                    mo_occ_k_dn    = mo_occ_k[mask_dn]
                    mo_coeff_k_dn  = mo_coeff_k[mask_dn, :]
                    mo_coeff_k_imag_dn = mo_coeff_k_imag[mask_dn, :]

                    mo_coeff_k_up_ = np.empty((nao, len(mo_energy_k_up)),
                                              dtype=(mo_coeff_k_up + 1j * mo_coeff_k_imag_up).dtype)
                    mo_coeff_k_up_[idx, :] = mo_coeff_k_up.T + 1j * mo_coeff_k_imag_up.T

                    mo_coeff_k_dn_ = np.empty((nao, len(mo_energy_k_dn)),
                                              dtype=(mo_coeff_k_dn + 1j * mo_coeff_k_imag_dn).dtype)
                    mo_coeff_k_dn_[idx, :] = mo_coeff_k_dn.T + 1j * mo_coeff_k_imag_dn.T

                    mo_energy_all_k.append((mo_energy_k_up, mo_energy_k_dn))
                    mo_occ_all_k.append((mo_occ_k_up, mo_occ_k_dn))
                    mo_coeff_all_k.append((mo_coeff_k_up_, mo_coeff_k_dn_))

                mf = mol.KUHF(kpts=kpts)
                mf.mo_coeff = mo_coeff_all_k
                mf.mo_energy = mo_energy_all_k
                mf.mo_occ = mo_occ_all_k
                return mf

            # RHF
            elif set(uniq.tolist()) == {0} or set(uniq.tolist()) == {1}:
                mo_energy_all_k = []
                mo_occ_all_k = []
                mo_coeff_all_k = []

                for i_k in range(k_point_num):
                    mask = (mo_k_point == i_k)
                    mo_energy_k = mo_energy[mask]
                    mo_occ_k    = mo_occ[mask]
                    mo_spin_k   = mo_spin[mask]
                    mo_coeff_k = mo_coeff[mask, :]
                    mo_coeff_k_imag = mo_coeff_imag[mask, :]

                    mo_coeff_k_ = np.empty((nao, len(mo_energy_k)), dtype=(mo_coeff_k + 1j * mo_coeff_k_imag).dtype)
                    mo_coeff_k_[idx, :] = mo_coeff_k.T + 1j * mo_coeff_k_imag.T

                    mo_energy_all_k.append(mo_energy_k)
                    mo_occ_all_k.append(mo_occ_k)
                    mo_coeff_all_k.append(mo_coeff_k_)

                mf = mol.KRHF(kpts=kpts)
                mf.mo_coeff = mo_coeff_all_k
                mf.mo_energy = mo_energy_all_k
                mf.mo_occ    = mo_occ_all_k
                return mf

            else:
                raise ValueError(f'Unknown spin multiplicity {uniq}')


    else:
        # Non-periodic case
        with trexio.File(filename, 'r', back_end=trexio.TREXIO_AUTO) as tf:
            num_mo    = trexio.read_mo_num(tf)
            mo_energy = trexio.read_mo_energy(tf)
            mo_coeff  = trexio.read_mo_coefficient(tf)
            mo_occ    = trexio.read_mo_occupation(tf)
            mo_spin    = trexio.read_mo_spin(tf)

        nao = mol.nao
        idx = _order_ao_index(mol)

        uniq = np.unique(mo_spin)

        # UHF
        if set(uniq.tolist()) == {0, 1}:
            i_up = np.where(mo_spin == 0)[0]
            i_dn = np.where(mo_spin == 1)[0]

            mo_up_out = mo_coeff[i_up, :]  # (nmo_up, nao)
            mo_dn_out = mo_coeff[i_dn, :]  # (nmo_dn, nao)

            mo_up_pyscf = np.empty((nao, mo_up_out.shape[0]), dtype=mo_coeff.dtype)
            mo_dn_pyscf = np.empty((nao, mo_dn_out.shape[0]), dtype=mo_coeff.dtype)
            mo_up_pyscf[idx, :] = mo_up_out.T
            mo_dn_pyscf[idx, :] = mo_dn_out.T

            mf = mol.UHF()
            mf.mo_coeff  = (mo_up_pyscf, mo_dn_pyscf)
            mf.mo_energy = (mo_energy[i_up], mo_energy[i_dn])
            mf.mo_occ    = (mo_occ[i_up],   mo_occ[i_dn])
            return mf

        # RHF
        elif set(uniq.tolist()) == {0} or set(uniq.tolist()) == {1}:
            mf = mol.RHF()
            mf.mo_coeff = np.empty((nao, num_mo), dtype=mo_coeff.dtype)
            mf.mo_coeff[idx, :] = mo_coeff.T
            mf.mo_energy = mo_energy
            mf.mo_occ    = mo_occ
            return mf

        else:
            raise ValueError(f'Unknown spin multiplicity {uniq}')

def write_ao_2e_int_eri(eri, filename, backend='h5'):
    raise NotImplementedError

def read_ao_2e_int_eri(filename):
    raise NotImplementedError

def write_mo_2e_int_eri(eri, filename, backend='h5'):
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
    else:
        raise ValueError(f'ERI array must be 1, 2 or 4-dimensional, got {eri.ndim}')

    # Physicist notation
    idx=idx.reshape((num_integrals,4))
    for i in range(num_integrals):
        idx[i,1],idx[i,2]=idx[i,2],idx[i,1]

    idx=idx.flatten()

    with trexio.File(filename, 'w', back_end=_mode(backend)) as tf:
        trexio.write_mo_2e_int_eri(tf, 0, num_integrals, idx, eri.ravel())

def read_mo_2e_int_eri(filename):
    with trexio.File(filename, 'r', back_end=trexio.TREXIO_AUTO) as tf:
        nmo = trexio.read_mo_num(tf)
        nao_pair = nmo * (nmo+1) // 2
        eri_size = nao_pair * (nao_pair+1) // 2
        idx, data, n_read, eof_flag = trexio.read_mo_2e_int_eri(tf, 0, eri_size)
    eri = np.zeros(eri_size)
    x = idx[:,0]*(idx[:,0]+1)//2 + idx[:,2]
    y = idx[:,1]*(idx[:,1]+1)//2 + idx[:,3]
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
    assert all(a <= b for a, b in zip(keys, keys[1:]))
    idx = np.unique(keys, return_index=True)[1]
    return np.split(a, idx[1:])

def _get_occsa_and_occsb(mcscf, norb, nelec, ci_threshold=0.):
    ci_coeff = mcscf.ci
    num_determinants = int(np.sum(np.abs(ci_coeff) > ci_threshold))
    occslst = fci.cistring.gen_occslst(range(norb), nelec // 2)
    selected_occslst = occslst[:num_determinants]

    occsa = []
    occsb = []
    ci_values = []

    for i in range(min(len(selected_occslst), mcscf.ci.shape[0])):
        for j in range(min(len(selected_occslst), mcscf.ci.shape[1])):
            ci_coeff = mcscf.ci[i, j]
            if np.abs(ci_coeff) > ci_threshold:  # Check if CI coefficient is significant compared to user defined value
                occsa.append(selected_occslst[i])
                occsb.append(selected_occslst[j])
                ci_values.append(ci_coeff)

    # Sort by the absolute value of the CI coefficients in descending order
    sorted_indices = np.argsort(-np.abs(ci_values))
    occsa_sorted = [occsa[idx] for idx in sorted_indices]
    occsb_sorted = [occsb[idx] for idx in sorted_indices]
    ci_values_sorted = [ci_values[idx] for idx in sorted_indices]

    return occsa_sorted, occsb_sorted, ci_values_sorted, num_determinants

def _det_to_trexio(mcscf, norb, nelec, trexio_file, ci_threshold=0., chunk_size=100000):
    ncore = mcscf.ncore
    int64_num = trexio.get_int64_num(trexio_file)

    occsa, occsb, ci_values, num_determinants = _get_occsa_and_occsb(mcscf, norb, nelec, ci_threshold)

    det_list = []
    for a, b, coeff in zip(occsa, occsb, ci_values):
        occsa_upshifted = [orb for orb in range(ncore)] + [orb+ncore for orb in a]
        occsb_upshifted = [orb for orb in range(ncore)] + [orb+ncore for orb in b]
        det_tmp = []
        det_tmp += trexio.to_bitfield_list(int64_num, occsa_upshifted)
        det_tmp += trexio.to_bitfield_list(int64_num, occsb_upshifted)
        det_list.append(det_tmp)

    if num_determinants > chunk_size:
        n_chunks = math.ceil(num_determinants / chunk_size)
    else:
        n_chunks = 1

    if trexio.has_determinant(trexio_file):
        trexio.delete_determinant(trexio_file)

    offset_file = 0
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_determinants)
        current_chunk_size = end - start

        if current_chunk_size > 0:
            trexio.write_determinant_list(trexio_file, offset_file, current_chunk_size, det_list[start:end])
            trexio.write_determinant_coefficient(trexio_file, offset_file, current_chunk_size, ci_values[start:end])
            offset_file += current_chunk_size

def read_det_trexio(filename):
    with trexio.File(filename, 'r', back_end=trexio.TREXIO_AUTO) as tf:
        offset_file = 0

        num_det = trexio.read_determinant_num(tf)
        coeff = trexio.read_determinant_coefficient(tf, offset_file, num_det)
        det = trexio.read_determinant_list(tf, offset_file, num_det)
        return num_det, coeff, det

def _to_segment_contraction(mol):
    '''transform generally contracted basis to segment contracted basis
    '''
    _bas = []
    for shell in mol._bas:
        nctr = shell[gto.NCTR_OF]
        if nctr == 1:
            _bas.append(shell)
            continue

        nprim = shell[gto.NPRIM_OF]
        pcoeff = shell[gto.PTR_COEFF]
        bs = np.repeat(shell[np.newaxis], nctr, axis=0)
        bs[:,gto.NCTR_OF] = 1
        bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr, nprim)
        _bas.append(bs)

    pmol = mol.copy()
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    return pmol
