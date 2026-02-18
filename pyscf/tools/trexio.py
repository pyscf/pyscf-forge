#!/usr/bin/env python

"""
TREX-IO interface

References:
    https://trex-coe.github.io/trexio/trex.html
    https://github.com/TREX-CoE/trexio-tutorials/blob/ad5c60aa6a7bca802c5918cef2aeb4debfa9f134/notebooks/tutorial_benzene.md

Installation instruction:
    https://github.com/TREX-CoE/trexio/blob/master/python/README.md
"""

import re
import math
import numpy as np
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
from pyscf import ao2mo
from pyscf.pbc import df as pbcdf, tools as pbctools

import trexio


def to_trexio(obj, filename, backend="h5", ci_threshold=None, chunk_size=None):
    with trexio.File(filename, "u", back_end=_mode(backend)) as tf:
        if isinstance(obj, gto.Mole) or isinstance(obj, pbcgto.Cell):
            _mol_to_trexio(obj, tf)
        elif isinstance(obj, scf.hf.SCF):
            _scf_to_trexio(obj, tf)
        elif isinstance(obj, mcscf.casci.CASCI) or isinstance(obj, mcscf.CASSCF):
            ci_threshold = ci_threshold if ci_threshold is not None else 0.0
            chunk_size = chunk_size if chunk_size is not None else 100000
            _mcscf_to_trexio(obj, tf, ci_threshold=ci_threshold, chunk_size=chunk_size)
        else:
            raise NotImplementedError(f"Conversion function for {obj.__class__}")


def _mol_to_trexio(mol, trexio_file):
    # 1 Metadata
    trexio.write_metadata_code_num(trexio_file, 1)
    trexio.write_metadata_code(trexio_file, [f"PySCF-v{pyscf.__version__}"])
    # trexio.write_metadata_package_version(trexio_file, f'TREXIO-v{trexio.__version__}')

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
        a = lattice[0]  # unit is Bohr
        b = lattice[1]  # unit is Bohr
        c = lattice[2]  # unit is Bohr
        trexio.write_cell_a(trexio_file, a)
        trexio.write_cell_b(trexio_file, b)
        trexio.write_cell_c(trexio_file, c)
    else:
        # 2.3 Periodic boundary calculations (pbc group)
        trexio.write_pbc_periodic(trexio_file, False)

    # 2.4 Electron (electron group)
    electron_up_num, electron_dn_num = mol.nelec
    trexio.write_electron_num(trexio_file, electron_up_num + electron_dn_num)
    trexio.write_electron_up_num(trexio_file, electron_up_num)
    trexio.write_electron_dn_num(trexio_file, electron_dn_num)

    # 2.5 Nuclear repulsion energy
    try:
        trexio.write_nucleus_repulsion(trexio_file, mol.energy_nuc())
    except trexio.Error:
        # TREXIO ver 2.6.0 bug. See #231. A negative value is not accepted.
        trexio.write_nucleus_repulsion(trexio_file, 0.0)

    # 3.1 Basis set
    trexio.write_basis_type(trexio_file, "Gaussian")
    if any(mol._bas[:, gto.NCTR_OF] > 1):
        mol = _to_segment_contraction(mol)
    trexio.write_basis_shell_num(trexio_file, mol.nbas)
    trexio.write_basis_prim_num(trexio_file, int(mol._bas[:, gto.NPRIM_OF].sum()))
    trexio.write_basis_nucleus_index(trexio_file, mol._bas[:, gto.ATOM_OF])
    trexio.write_basis_shell_ang_mom(trexio_file, mol._bas[:, gto.ANG_OF])
    trexio.write_basis_shell_factor(trexio_file, np.ones(mol.nbas))
    prim2sh = [[ib] * nprim for ib, nprim in enumerate(mol._bas[:, gto.NPRIM_OF])]
    trexio.write_basis_shell_index(trexio_file, np.hstack(prim2sh))
    trexio.write_basis_exponent(trexio_file, np.hstack(mol.bas_exps()))
    coef = [mol.bas_ctr_coeff(i).ravel() for i in range(mol.nbas)]
    trexio.write_basis_coefficient(trexio_file, np.hstack(coef))
    prim_norms = [
        gto.gto_norm(mol.bas_angular(i), mol.bas_exp(i)) for i in range(mol.nbas)
    ]
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
        for i, ang_mom in enumerate(mol._bas[:, gto.ANG_OF]):
            ao_shell += [i] * int((ang_mom + 1) * (ang_mom + 2) / 2)
            # note: PySCF(libintc) normalizes s and p only.
            if ang_mom == 0 or ang_mom == 1:
                ao_normalization += [
                    float(np.sqrt((2 * ang_mom + 1) / (4 * np.pi)))
                ] * int((ang_mom + 1) * (ang_mom + 2) / 2)
            else:
                ao_normalization += [1.0] * int((ang_mom + 1) * (ang_mom + 2) / 2)
    else:
        trexio.write_ao_cartesian(trexio_file, 0)
        ao_shell = []
        ao_normalization = []
        for i, ang_mom in enumerate(mol._bas[:, gto.ANG_OF]):
            ao_shell += [i] * (2 * ang_mom + 1)
            # note: TREXIO employs the solid harmonics notation,; thus, we need these factors.
            ao_normalization += [float(np.sqrt((2 * ang_mom + 1) / (4 * np.pi)))] * (
                2 * ang_mom + 1
            )

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
        madelung = pbctools.pbc.madelung(mol, kpts)

        if nk == 1:
            # 2.3 Periodic boundary calculations (pbc group)
            trexio.write_pbc_k_point_num(trexio_file, 1)
            trexio.write_pbc_k_point(trexio_file, kpts)
            trexio.write_pbc_k_point_weight(trexio_file, weights[np.newaxis])
            trexio.write_pbc_madelung(trexio_file, madelung)

            if isinstance(
                mf,
                (
                    pbc.scf.uhf.UHF,
                    pbc.dft.uks.UKS,
                    pbc.scf.kuhf.KUHF,
                    pbc.dft.kuks.KUKS,
                ),
            ):
                mo_type = "UHF"
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
                    raise NotImplementedError(f"Conversion function for {mf.__class__}")
                idx = _order_ao_index(mf.mol)
                mo_up = mo_up[idx].T
                mo_dn = mo_dn[idx].T
                num_mo_up = len(mo_up)
                num_mo_dn = len(mo_dn)
                assert num_mo_up + num_mo_dn == mo_num
                mo = np.concatenate(
                    [mo_up, mo_dn], axis=0
                )  # dim (num_mo, num_ao) but it is f-contiguous
                mo_coefficient = np.ascontiguousarray(
                    mo
                )  # dim (num_mo, num_ao) and it is c-contiguous
                if np.all(np.isreal(mo_coefficient)):
                    mo_coefficient_real = mo_coefficient
                    mo_coefficient_imag = None
                else:
                    mo_coefficient_real = mo_coefficient.real
                    mo_coefficient_imag = mo_coefficient.imag
                mo_occ = np.ravel(mf.mo_occ)
                mo_spin = np.zeros(num_mo_up + num_mo_dn, dtype=int)
                mo_spin[:num_mo_up] = 0
                mo_spin[num_mo_up:] = 1

            elif isinstance(
                mf,
                (
                    pbc.scf.krhf.KRHF,
                    pbc.dft.krks.KRKS,
                    pbc.scf.rhf.RHF,
                    pbc.dft.rks.RKS,
                ),
            ):
                mo_type = "RHF"
                if isinstance(mf, (pbc.scf.rhf.RHF, pbc.dft.rks.RKS)):
                    mo_energy = np.ravel(mf.mo_energy)
                    mo_num = mo_energy.size
                    mo = mf.mo_coeff
                elif isinstance(mf, (pbc.scf.krhf.KRHF, pbc.dft.krks.KRKS)):
                    mo_energy = np.ravel(mf.mo_energy)
                    mo_num = mo_energy.size
                    mo = mf.mo_coeff[0]
                else:
                    raise NotImplementedError(f"Conversion function for {mf.__class__}")
                idx = _order_ao_index(mf.mol)
                mo = mo[idx].T  # dim (num_mo, num_ao) but it is f-contiguous
                mo_coefficient = np.ascontiguousarray(
                    mo
                )  # dim (num_mo, num_ao) and it is c-contiguous
                if np.all(np.isreal(mo_coefficient)):
                    mo_coefficient_real = mo_coefficient
                    mo_coefficient_imag = None
                else:
                    mo_coefficient_real = mo_coefficient.real
                    mo_coefficient_imag = mo_coefficient.imag
                mo_occ = np.ravel(mf.mo_occ)
                mo_spin = np.zeros(mo_num, dtype=int)
            else:
                raise NotImplementedError(f"Conversion function for {mf.__class__}")

            # 4.2 Molecular orbitals (mo group)
            trexio.write_mo_type(trexio_file, mo_type)
            trexio.write_mo_num(trexio_file, mo_num)
            trexio.write_mo_k_point(trexio_file, [0] * mo_num)
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
            trexio.write_pbc_madelung(trexio_file, madelung)

            # stack k-dependent molecular orbitals
            mo_k_point_pbc = []
            mo_num_pbc = 0
            mo_energy_pbc = []
            mo_coefficient_real_pbc = []
            mo_coefficient_imag_pbc = []
            mo_occ_pbc = []
            mo_spin_pbc = []

            if isinstance(
                mf,
                (
                    pbc.scf.uhf.UHF,
                    pbc.dft.uks.UKS,
                    pbc.scf.kuhf.KUHF,
                    pbc.dft.kuks.KUKS,
                ),
            ):
                mo_type = "UHF"
                # Check for split structure (common in KUKS/KDF): ([up...], [dn...])
                is_split_spin = isinstance(mf.mo_coeff, tuple) and len(mf.mo_coeff) == 2

                for i_k, _ in enumerate(kpts):
                    if is_split_spin:
                        e_up = mf.mo_energy[0][i_k]
                        e_dn = mf.mo_energy[1][i_k]
                        mo_energy = np.concatenate([np.ravel(e_up), np.ravel(e_dn)])

                        mo_up = mf.mo_coeff[0][i_k]
                        mo_dn = mf.mo_coeff[1][i_k]

                        occ_up = mf.mo_occ[0][i_k]
                        occ_dn = mf.mo_occ[1][i_k]
                        mo_occ = np.concatenate([np.ravel(occ_up), np.ravel(occ_dn)])
                    else:
                        mo_energy = np.ravel(mf.mo_energy[i_k])
                        mo_up, mo_dn = mf.mo_coeff[i_k]
                        mo_occ = np.ravel(mf.mo_occ[i_k])

                    mo_num = mo_energy.size
                    idx = _order_ao_index(mf.mol)
                    mo_up = mo_up[idx].T
                    mo_dn = mo_dn[idx].T
                    mo = np.concatenate(
                        [mo_up, mo_dn], axis=0
                    )  # dim (num_mo, num_ao) but it is f-contiguous
                    mo_coefficient_real = np.ascontiguousarray(
                        mo.real
                    )  # dim (num_mo, num_ao) and it is c-contiguous
                    mo_coefficient_imag = np.ascontiguousarray(
                        mo.imag
                    )  # dim (num_mo, num_ao) and it is c-contiguous

                    mo_spin = np.zeros(mo_energy.size, dtype=int)
                    # Use size of up component to split spin
                    if is_split_spin:
                        mo_spin[e_up.size :] = 1
                    else:
                        # Assumes mo_energy was flattened from (up, dn)
                        # If mo_energy[i_k] is tuple (up, dn), ravel joins them.
                        # We need size of up.
                        if isinstance(mf.mo_energy[i_k], (tuple, list)):
                            mo_spin[mf.mo_energy[i_k][0].size :] = 1
                        else:
                            # If somehow not tuple?
                            # Fallback to half?
                            mo_spin[mo_energy.size // 2 :] = 1

                    mo_k_point_pbc += [i_k] * mo_energy.size
                    mo_num_pbc += mo_energy.size
                    mo_energy_pbc.append(mo_energy)
                    mo_coefficient_real_pbc.append(mo_coefficient_real)
                    mo_coefficient_imag_pbc.append(mo_coefficient_imag)
                    mo_occ_pbc.append(mo_occ)
                    mo_spin_pbc.append(mo_spin)

            else:
                mo_type = "RHF"
                for i_k, _ in enumerate(kpts):
                    mo_energy = mf.mo_energy[i_k]
                    mo = mf.mo_coeff[i_k]
                    idx = _order_ao_index(mf.mol)
                    mo = mo[idx].T
                    mo_coefficient_real = np.ascontiguousarray(
                        mo.real
                    )  # dim (num_mo, num_ao) and it is c-contiguous
                    mo_coefficient_imag = np.ascontiguousarray(
                        mo.imag
                    )  # dim (num_mo, num_ao) and it is c-contiguous
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
            mo_coefficient_real_pbc = np.ascontiguousarray(
                np.vstack(mo_coefficient_real_pbc)
            )  # it is c-contiguous
            mo_coefficient_imag_pbc = np.ascontiguousarray(
                np.vstack(mo_coefficient_imag_pbc)
            )  # it is c-contiguous
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
            mo_type = "UHF"
            mo_energy = np.ravel(mf.mo_energy)
            mo_num = mo_energy.size
            mo_up, mo_dn = mf.mo_coeff
            idx = _order_ao_index(mf.mol)
            mo_up = mo_up[idx].T
            mo_dn = mo_dn[idx].T
            mo = np.concatenate(
                [mo_up, mo_dn], axis=0
            )  # dim (num_mo, num_ao) but it is f-contiguous
            mo_coefficient = np.ascontiguousarray(
                mo
            )  # dim (num_mo, num_ao) and it is c-contiguous
            mo_occ = np.ravel(mf.mo_occ)
            mo_spin = np.zeros(mo_energy.size, dtype=int)
            mo_spin[mf.mo_energy[0].size :] = 1
        else:
            mo_type = "RHF"
            mo_energy = mf.mo_energy
            mo_num = mo_energy.size
            mo = mf.mo_coeff
            idx = _order_ao_index(mf.mol)
            mo = mo[idx].T  # dim (num_mo, num_ao) but it is f-contiguous
            mo_coefficient = np.ascontiguousarray(
                mo
            )  # dim (num_mo, num_ao) and it is c-contiguous
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


def _mcscf_to_trexio(cas_obj, trexio_file, ci_threshold=0.0, chunk_size=100000):
    mol = cas_obj.mol
    _mol_to_trexio(mol, trexio_file)
    mo_energy_cas = cas_obj.mo_energy
    mo_cas = cas_obj.mo_coeff
    num_mo = mo_energy_cas.size
    spin_cas = np.zeros(mo_energy_cas.size, dtype=int)
    mo_type_cas = "CAS"
    trexio.write_mo_type(trexio_file, mo_type_cas)
    idx = _order_ao_index(mol)
    trexio.write_mo_num(trexio_file, num_mo)
    trexio.write_mo_coefficient(trexio_file, mo_cas[idx].T.ravel())
    trexio.write_mo_energy(trexio_file, mo_energy_cas)
    trexio.write_mo_spin(trexio_file, spin_cas)

    ncore = cas_obj.ncore
    ncas = cas_obj.ncas
    mo_classes = np.array(
        ["Virtual"] * num_mo, dtype=str
    )  # Initialize all MOs as Virtual
    mo_classes[:ncore] = "Core"
    mo_classes[ncore : ncore + ncas] = "Active"
    trexio.write_mo_class(trexio_file, list(mo_classes))

    occupation = np.zeros(num_mo)
    occupation[:ncore] = 2.0
    rdm1 = cas_obj.fcisolver.make_rdm1(cas_obj.ci, ncas, cas_obj.nelecas)
    natural_occ = np.linalg.eigh(rdm1)[0]
    occupation[ncore : ncore + ncas] = natural_occ[::-1]
    occupation[ncore + ncas :] = 0.0
    trexio.write_mo_occupation(trexio_file, occupation)

    total_elec_cas = sum(cas_obj.nelecas)

    _det_to_trexio(cas_obj, ncas, total_elec_cas, trexio_file, ci_threshold, chunk_size)


def mol_from_trexio(filename):
    with trexio.File(filename, "r", back_end=trexio.TREXIO_AUTO) as tf:
        assert trexio.read_basis_type(tf) == "Gaussian"
        pbc_periodic = trexio.read_pbc_periodic(tf)
        labels = trexio.read_nucleus_label(tf)
        coords = trexio.read_nucleus_coord(tf)
        elements = [s + str(i) for i, s in enumerate(labels)]

        if pbc_periodic:
            mol = pbcgto.Cell()
            mol.unit = "Bohr"
            a = np.asarray(trexio.read_cell_a(tf), dtype=float)
            b = np.asarray(trexio.read_cell_b(tf), dtype=float)
            c = np.asarray(trexio.read_cell_c(tf), dtype=float)
            mol.a = np.vstack([a, b, c])
        else:
            mol = gto.Mole()
            mol.unit = "Bohr"

        mol.atom = list(zip(elements, coords))
        up_num = trexio.read_electron_up_num(tf)
        dn_num = trexio.read_electron_dn_num(tf)
        spin = up_num - dn_num
        mol.spin = spin

        if trexio.has_ecp(tf):
            # --- read TREXIO ECP arrays ---
            z_core = trexio.read_ecp_z_core(tf)  # shape (natm,)
            max_l1_arr = trexio.read_ecp_max_ang_mom_plus_1(tf)  # shape (natm,)
            nuc_idx = trexio.read_ecp_nucleus_index(tf)  # shape (ecp_num,)
            ang_mom_enc = trexio.read_ecp_ang_mom(tf)  # shape (ecp_num,)
            coeff_arr = trexio.read_ecp_coefficient(tf)  # shape (ecp_num,)
            exp_arr = trexio.read_ecp_exponent(tf)  # shape (ecp_num,)
            power_arr = trexio.read_ecp_power(tf)  # shape (ecp_num,)

            # --- aggregate primitives per (nucleus, l); decode local channel to l = -1 ---
            per_atom = defaultdict(
                lambda: defaultdict(list)
            )  # per_atom[n][l] -> [(power, exp, coeff), ...]
            for k in range(len(nuc_idx)):
                n = int(nuc_idx[k])
                l_e = int(ang_mom_enc[k])
                # local channel was encoded as max_l1_arr[n]; decode to l = -1
                l_d = -1 if l_e == int(max_l1_arr[n]) else l_e
                p = int(power_arr[k])  # stored as r-2 on write
                e = float(exp_arr[k])
                c = float(coeff_arr[k])
                per_atom[n][l_d].append((p, e, c))

            def _is_dummy_ul_s(items):
                # H/He dummy record injected at write time: (power=0, exp=1.0, coeff=0.0)
                return (
                    len(items) == 1
                    and items[0][0] == 0
                    and abs(items[0][1] - 1.0) < 1e-12
                    and abs(items[0][2]) < 1e-300
                )

            ecp_dict = {}
            for n, raw_sym in enumerate(labels):
                # skip ghosts: ECP not applied to 'X-*' atoms
                if re.match(r"X-.*", raw_sym):
                    continue

                zc = int(z_core[n]) if n < len(z_core) else 0
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
                    max_r = max(r_list)
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
            at_basis = [
                [l, *zip(e, c)] for l, e, c in zip(ls[p0:p1], exps[p0:p1], coef[p0:p1])
            ]
            basis[elements[ia]] = at_basis

        # To avoid the mol.build() sort the basis, disable mol.basis and set the
        # internal data _basis directly.
        mol.basis = {}
        mol._basis = basis
        return mol.build()

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

_REAL_ONLY_TOL = 1e-12


def _trexio_ensure_real(x, *, tol=_REAL_ONLY_TOL, what="Complex data encountered but the backend is real-only."):
    if np.iscomplexobj(x):
        if np.all(np.abs(np.imag(x)) <= tol):
            x = np.real(x)
        else:
            raise NotImplementedError(what)
    return x


def _trexio_is_gamma_single_k(obj) -> bool:
    if not hasattr(obj, 'cell'):
        return False
    if hasattr(obj, 'kpt'):
        return np.allclose(np.asarray(obj.kpt), 0.0)
    if hasattr(obj, 'kpts'):
        kpts = np.asarray(obj.kpts)
        return (kpts.shape[0] == 1) and np.allclose(kpts[0], 0.0)
    return True


def _trexio_get_uks_coeff_pair(mf_obj, *, expect_gamma=False):
    C = mf_obj.mo_coeff
    if isinstance(C, (list, tuple)) and len(C) == 2:
        Ca, Cb = C
    elif isinstance(C, np.ndarray) and C.ndim >= 3 and C.shape[0] == 2:
        if C.ndim == 3:
            Ca, Cb = C[0], C[1]
        elif C.ndim == 4:
            if expect_gamma and C.shape[1] == 1:
                Ca, Cb = C[0, 0], C[1, 0]
            else:
                raise NotImplementedError("Only single-k UKS/UHF is supported.")
        else:
            raise ValueError(f"Unexpected mo_coeff shape: {C.shape}")
    else:
        raise TypeError("Not a UKS/UHF object or unsupported mo_coeff layout.")

    if expect_gamma:
        if Ca.ndim == 3 and Ca.shape[0] == 1:
            Ca = Ca[0]
        if Cb.ndim == 3 and Cb.shape[0] == 1:
            Cb = Cb[0]
        if Ca.ndim != 2 or Cb.ndim != 2:
            raise NotImplementedError(
                "Only Gamma-point data are supported for combined MO coefficients."
            )

    if Ca.ndim != 2 or Cb.ndim != 2:
        raise ValueError(f"Unexpected UKS/UHF mo_coeff shapes: Ca {Ca.shape}, Cb {Cb.shape}")
    return Ca, Cb


def _trexio_concat_spin_coeff(Ca, Cb):
    if Ca.ndim != 2 or Cb.ndim != 2:
        raise ValueError(f"Unexpected UKS/UHF mo_coeff shapes: Ca {Ca.shape}, Cb {Cb.shape}")
    return np.concatenate([Ca, Cb], axis=1)

def write_2e_eri(
    mf, filename, backend='h5', basis='mo', df_engine='MDF', sym='s1',
):
    """Write two-electron repulsion integrals to a TREXIO file.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. PBC data must be
        Gamma-only.
    filename : str
        Path to the TREXIO file to create or update.
    backend : {'h5', 'text'}, optional
        TREXIO backend selector passed through to ``trexio.File``.
    basis : {'AO', 'MO'}, optional
        Basis in which integrals are written. AO is always spin-independent.
        MO concatenates alpha and beta MOs for UHF/UKS to include cross-spin
        blocks.
    df_engine : {'MDF', 'GDF'}, optional
        Density-fitting engine used for PBC ERIs (Gamma-only).
    sym : {'s1', 's4', 's8'}, optional
        ERI symmetry/packing. MO supports ``s1``/``s4``; AO supports
        ``s1``/``s4``/``s8``.

    Behavior
    --------
    - Real-only backend: complex ERIs are rejected unless the imaginary part
      is <=1e-12 (in which case it is discarded).
    - PBC: only single-k Gamma is supported; non-Gamma raises
      ``NotImplementedError``.
    - MO + UHF/UKS: constructs the combined coefficient matrix ``[Ca | Cb]``
      and writes all spin blocks.
    - Arrays are made C-contiguous before writing; dtype is preserved.

    Raises
    ------
    ValueError
        For invalid ``basis``/``sym`` combinations or unexpected shapes.
    NotImplementedError
        For complex ERIs, non-Gamma PBC data, or unsupported symmetry in MO.
    """

    basis = basis.upper()
    sym = sym.lower()
    is_pbc = hasattr(mf, 'cell')

    if sym not in ('s1', 's4', 's8'):
        raise ValueError("sym must be 's1', 's4', or 's8'")
    if basis == 'MO' and sym == 's8':
        raise NotImplementedError("MO ERI does not support s8 symmetry; use s1 or s4")

    # ---------- helpers ----------
    def _df_obj():
        """Construct a DF engine for PBC Γ-point."""
        if df_engine.upper() == 'MDF':
            return pbcdf.MDF(mf.cell).build()
        if df_engine.upper() == 'GDF':
            return pbcdf.GDF(mf.cell).build()
        raise ValueError("df_engine must be 'MDF' or 'GDF'.")

    # ---------------------
    # MO-basis ERI writing
    # ---------------------
    if basis == 'MO':
        # Molecular
        if not is_pbc:
            if sym == 's8':
                raise NotImplementedError("MO ERI does not support s8 symmetry")
            mo_compact = sym == 's4'
            if (isinstance(mf.mo_coeff, (list, tuple)) or
                (isinstance(mf.mo_coeff, np.ndarray) and mf.mo_coeff.ndim >= 3 and mf.mo_coeff.shape[0] == 2)):
                # UKS/UHF -> concatenate [alpha | beta], include cross-spin terms
                Ca, Cb = _trexio_get_uks_coeff_pair(mf)
                C = _trexio_concat_spin_coeff(Ca, Cb)  # (nao, nalpha+nbeta)
                if getattr(mf, '_eri', None) is not None:
                    eri_mo = ao2mo.incore.full(mf._eri, C, compact=mo_compact)
                else:
                    eri_mo = ao2mo.kernel(mf.mol, C, compact=mo_compact)
                eri_mo = _trexio_ensure_real(
                    eri_mo,
                    what=(
                        "Complex ERI encountered but the backend is real-only. "
                        "Use Gamma-point (k=0) or a complex-capable backend."
                    ),
                )
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), filename, backend, 'MO', sym=sym)
            else:  # RHF/RKS
                C = mf.mo_coeff
                if getattr(mf, '_eri', None) is not None:
                    eri_mo = ao2mo.incore.full(mf._eri, C, compact=mo_compact)
                else:
                    eri_mo = ao2mo.kernel(mf.mol, C, compact=mo_compact)
                eri_mo = _trexio_ensure_real(
                    eri_mo,
                    what=(
                        "Complex ERI encountered but the backend is real-only. "
                        "Use Gamma-point (k=0) or a complex-capable backend."
                    ),
                )
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), filename, backend, 'MO', sym=sym)

        # PBC (Gamma only)
        else:
            if not _trexio_is_gamma_single_k(mf):
                raise NotImplementedError("PBC MO-ERI: non-Gamma k-points are not supported (real-only backend).")
            if sym == 's8':
                raise NotImplementedError("PBC MO-ERI does not support s8 symmetry; use s1 or s4")
            dfobj = _df_obj()

            if (isinstance(mf.mo_coeff, (list, tuple)) or
                (isinstance(mf.mo_coeff, np.ndarray) and mf.mo_coeff.ndim >= 3 and mf.mo_coeff.shape[0] == 2)):
                # UKS/UHF @ Gamma: combined MO matrix [Ca | Cb]
                Ca, Cb = _trexio_get_uks_coeff_pair(mf, expect_gamma=True)
                C = _trexio_concat_spin_coeff(Ca, Cb)
                eri_mo = dfobj.get_mo_eri((C, C, C, C))
                eri_mo = _trexio_ensure_real(
                    eri_mo,
                    what=(
                        "Complex ERI encountered but the backend is real-only. "
                        "Use Gamma-point (k=0) or a complex-capable backend."
                    ),
                )
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim == 2:
                        eri_mo = ao2mo.restore(1, ao2mo.restore(4, eri_mo, nmo), nmo)
                    elif eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                elif sym == 's4':
                    if eri_mo.ndim == 4:
                        eri_mo = ao2mo.restore(4, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), filename, backend, 'MO', sym=sym)
            else:  # RHF/RKS @ Gamma
                C = mf.mo_coeff
                if C.ndim == 3 and C.shape[0] == 1:  # normalize (1,nao,nmo) -> (nao,nmo)
                    C = C[0]
                eri_mo = dfobj.get_mo_eri((C, C, C, C))
                eri_mo = _trexio_ensure_real(
                    eri_mo,
                    what=(
                        "Complex ERI encountered but the backend is real-only. "
                        "Use Gamma-point (k=0) or a complex-capable backend."
                    ),
                )
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim == 2:
                        eri_mo = ao2mo.restore(1, ao2mo.restore(4, eri_mo, nmo), nmo)
                    elif eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                elif sym == 's4':
                    if eri_mo.ndim == 4:
                        eri_mo = ao2mo.restore(4, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), filename, backend, 'MO', sym=sym)

    # ---------------------
    # AO-basis ERI writing (spin-independent even for UKS/UHF)
    # ---------------------
    else:  # basis == 'AO'
        if is_pbc:
            # PBC AO: Gamma only via DF (real-only)
            if not _trexio_is_gamma_single_k(mf):
                raise NotImplementedError("PBC AO-ERI: non-Gamma k-points are not supported (real-only backend).")
            dfobj = _df_obj()
            eri2 = pbcdf.df_ao2mo.get_eri(dfobj, compact=False)  # (nao^2, nao^2)
            nao = mf.cell.nao_nr()
            if eri2.shape != (nao * nao, nao * nao):
                raise RuntimeError(f"Unexpected ERI shape {eri2.shape}; expected ({nao*nao}, {nao*nao}) at Gamma.")
            eri_ao = eri2.reshape(nao, nao, nao, nao)
            eri_ao = _trexio_ensure_real(
                eri_ao,
                what=(
                    "Complex ERI encountered but the backend is real-only. "
                    "Use Gamma-point (k=0) or a complex-capable backend."
                ),
            )
            if sym == 's4':
                eri_ao = ao2mo.restore(4, eri_ao, nao)
            elif sym == 's8':
                eri_ao = ao2mo.restore(8, eri_ao, nao)
            _write_2e_int_eri(np.ascontiguousarray(eri_ao), filename, backend, 'AO', sym=sym)
        else:
            # Molecular AO
            eri_ao = None
            if sym == 's1':
                eri_ao = getattr(mf, '_eri', None)
                if eri_ao is None:
                    # 'int2e' follows mol.cart automatically (spherical vs Cartesian)
                    eri_ao = mf.mol.intor('int2e', aosym='s1')
                else:
                    # _eri may be stored in s4/s8; expand to full tensor for TREXIO write
                    if eri_ao.ndim < 4:
                        n_ao = mf.mol.nao
                        eri_ao = ao2mo.restore(1, eri_ao, n_ao)
            else:
                eri_ao = mf.mol.intor('int2e', aosym=sym)
            eri_ao = _trexio_ensure_real(
                eri_ao,
                what=(
                    "Complex ERI encountered but the backend is real-only. "
                    "Use Gamma-point (k=0) or a complex-capable backend."
                ),
            )
            _write_2e_int_eri(np.ascontiguousarray(eri_ao), filename, backend, 'AO', sym=sym)

def _write_2e_int_eri(eri, filename, backend='h5', basis='MO', sym='s1'):
    basis = basis.upper()
    sym = sym.lower()
    if basis not in ['MO', 'AO']:
        raise ValueError("basis must be 'MO' or 'AO'")
    if sym not in ('s1', 's4', 's8'):
        raise ValueError("sym must be 's1', 's4', or 's8'")

    def _pair_from_tril(n):
        i, j = np.tril_indices(n)
        return i.astype(np.int32), j.astype(np.int32)

    if sym == 's1':
        if eri.ndim != 4:
            raise ValueError(f'ERI array must be a full 4D tensor (p,q,r,s); got ndim={eri.ndim}')

        num_integrals = eri.size
        n = eri.shape[0]
        idx = lib.cartesian_prod([np.arange(n, dtype=np.int32)] * 4)

        # Physicist notation
        idx=idx.reshape((num_integrals,4))
        for i in range(num_integrals):
            idx[i,1],idx[i,2]=idx[i,2],idx[i,1]
        idx=idx.flatten()

        # write ERI
        with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
            if basis == 'AO':
                if not trexio.has_ao_num(tf):
                    trexio.write_ao_num(tf, n)
            else:
                if not trexio.has_mo_num(tf):
                    trexio.write_mo_num(tf, n)
            if basis == 'MO':
                trexio.write_mo_2e_int_eri(tf, 0, num_integrals, idx, eri.ravel())
            else:
                trexio.write_ao_2e_int_eri(tf, 0, num_integrals, idx, eri.ravel())
        return

    if sym == 's4':
        if eri.ndim != 2 or eri.shape[0] != eri.shape[1]:
            raise ValueError("s4 ERI must be a square 2D array (npair, npair)")
        npair = eri.shape[0]
        n = int(round((np.sqrt(8 * npair + 1) - 1) / 2))
        if n * (n + 1) // 2 != npair:
            raise ValueError('Invalid s4 ERI size for pair indexing')

        pair_i, pair_j = _pair_from_tril(n)
        total = npair * npair
        chunk = 100000

        with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
            if basis == 'AO':
                if not trexio.has_ao_num(tf):
                    trexio.write_ao_num(tf, n)
            else:
                if not trexio.has_mo_num(tf):
                    trexio.write_mo_num(tf, n)

            offset = 0
            while offset < total:
                end = min(offset + chunk, total)
                t = np.arange(offset, end, dtype=np.int64)
                ij = t // npair
                kl = t % npair

                # Physicist notation
                i = pair_i[ij]
                j = pair_j[ij]
                k = pair_i[kl]
                l = pair_j[kl]
                idx = np.stack([i, k, j, l], axis=1).astype(np.int32).ravel()
                val = eri[ij, kl].ravel()

                if basis == 'MO':
                    trexio.write_mo_2e_int_eri(tf, offset, end - offset, idx, val)
                else:
                    trexio.write_ao_2e_int_eri(tf, offset, end - offset, idx, val)
                offset = end
        return

    # sym == 's8'
    if eri.ndim != 1:
        raise ValueError('s8 ERI must be a 1D packed array')
    total = eri.size
    npair = int(round((np.sqrt(8 * total + 1) - 1) / 2))
    if npair * (npair + 1) // 2 != total:
        raise ValueError('Invalid s8 ERI size for pair indexing')
    n = int(round((np.sqrt(8 * npair + 1) - 1) / 2))
    if n * (n + 1) // 2 != npair:
        raise ValueError('Invalid s8 ERI size for AO/MO indexing')

    pair_i, pair_j = _pair_from_tril(n)
    tri_i, tri_j = np.tril_indices(npair)
    chunk = 100000

    with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
        if basis == 'AO':
            if not trexio.has_ao_num(tf):
                trexio.write_ao_num(tf, n)
        else:
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, n)

        offset = 0
        while offset < total:
            end = min(offset + chunk, total)
            ij = tri_i[offset:end]
            kl = tri_j[offset:end]

            # Physicist notation
            i = pair_i[ij]
            j = pair_j[ij]
            k = pair_i[kl]
            l = pair_j[kl]
            idx = np.stack([i, k, j, l], axis=1).astype(np.int32).ravel()
            val = eri[offset:end]

            if basis == 'MO':
                trexio.write_mo_2e_int_eri(tf, offset, end - offset, idx, val)
            else:
                trexio.write_ao_2e_int_eri(tf, offset, end - offset, idx, val)
            offset = end

def write_1e_eri(
    mf, filename, backend='h5', basis='AO', df_engine='MDF',
):
    """Write one-electron integrals to a TREXIO file.

    Stored quantities are overlap, kinetic, nuclear-electron potential, and
    the core Hamiltonian (plus ECP if present) in either AO or MO basis.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. PBC data must be
        Gamma-only.
    filename : str
        Path to the TREXIO file to create or update.
    backend : {'h5', 'text'}, optional
        TREXIO backend selector passed through to ``trexio.File``.
    basis : {'AO', 'MO'}, optional
        Basis in which integrals are written. For MO + UHF/UKS, alpha and beta
        coefficients are concatenated column-wise to form a single block.
    df_engine : {'MDF', 'GDF'}, optional
        Density-fitting engine for PBC integrals (Gamma-only).

    Behavior
    --------
    - Real-only backend: imaginary parts larger than 1e-12 raise
      ``NotImplementedError``; smaller parts are discarded.
    - PBC: only single-k Gamma calculations are supported.
    - All matrices are hermitized before writing and made C-contiguous; dtype
      is preserved.

    Raises
    ------
    ValueError
        For invalid ``basis`` or unexpected shapes.
    NotImplementedError
        For complex data, non-Gamma PBC calculations, or unsupported MO layout.
    """

    basis = basis.upper()
    if basis not in ('AO', 'MO'):
        raise ValueError("basis must be either 'AO' or 'MO'")

    def _as_matrix(mat, label):
        if isinstance(mat, (tuple, list)):
            if len(mat) == 0:
                raise ValueError(f"Empty data for {label}")
            if len(mat) > 1:
                raise NotImplementedError(
                    f"{label}: multiple blocks are not supported in this helper"
                )
            mat = mat[0]
        mat = np.asarray(mat)
        if mat.ndim == 3:
            if mat.shape[0] != 1:
                raise NotImplementedError(
                    f"{label}: Gamma-only support; received shape {mat.shape}"
                )
            mat = mat[0]
        if mat.ndim != 2:
            raise ValueError(f"{label} must be a 2D matrix, got shape {mat.shape}")
        return mat

    def _hermitize(mat):
        return 0.5 * (mat + mat.T.conj())

    is_pbc = hasattr(mf, 'cell')

    ecp_mat = None

    if is_pbc:
        if not _trexio_is_gamma_single_k(mf):
            raise NotImplementedError(
                "PBC one-electron integrals are implemented for Gamma-point only."
            )

        cell = mf.cell
        overlap = _as_matrix(mf.get_ovlp(), 'AO overlap')
        kinetic = _as_matrix(cell.pbc_intor('int1e_kin', 1, 1), 'AO kinetic')

        df_builder = getattr(mf, 'with_df', None)
        if df_builder is None:
            if df_engine.upper() == 'MDF':
                df_builder = pbcdf.MDF(cell).build()
            elif df_engine.upper() == 'GDF':
                df_builder = pbcdf.GDF(cell).build()
            else:
                raise ValueError("df_engine must be 'MDF' or 'GDF'")
        else:
            df_builder = df_builder.build()

        if cell.pseudo:
            potential = _as_matrix(df_builder.get_pp(), 'AO potential')
        else:
            potential = _as_matrix(df_builder.get_nuc(), 'AO potential')

        if len(getattr(cell, '_ecpbas', [])) > 0:
            from pyscf.pbc.gto import ecp
            ecp_mat = _as_matrix(ecp.ecp_int(cell), 'AO ECP potential')
            potential += ecp_mat

        core = kinetic + potential
    else:
        mol = mf.mol
        overlap = _as_matrix(mf.get_ovlp(), 'AO overlap')
        kinetic = _as_matrix(mol.intor('int1e_kin'), 'AO kinetic')
        potential = _as_matrix(mol.intor('int1e_nuc'), 'AO potential')
        if mol._ecp:
            ecp_mat = _as_matrix(mol.intor('ECPscalar'), 'AO ECP potential')
            potential += ecp_mat
        core = kinetic + potential

    overlap = np.ascontiguousarray(
        _trexio_ensure_real(
            _hermitize(overlap),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    kinetic = np.ascontiguousarray(
        _trexio_ensure_real(
            _hermitize(kinetic),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    potential = np.ascontiguousarray(
        _trexio_ensure_real(
            _hermitize(potential),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    core = np.ascontiguousarray(
        _trexio_ensure_real(
            _hermitize(core),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    if ecp_mat is not None:
        ecp_mat = np.ascontiguousarray(
            _trexio_ensure_real(
                _hermitize(ecp_mat),
                what="Complex one-electron integrals encountered but the backend is real-only.",
            )
        )

    if basis == 'AO':
        _write_1e_int_eri(overlap, kinetic, potential, core, filename, backend, 'AO')
        if ecp_mat is not None:
            with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
                trexio.write_ao_1e_int_ecp(tf, ecp_mat)
        return

    def _get_rhf_coeff(mf_obj):
        coeff = mf_obj.mo_coeff
        if isinstance(coeff, np.ndarray):
            if coeff.ndim == 2:
                return coeff
            if coeff.ndim == 3 and coeff.shape[0] == 1:
                return coeff[0]
        if isinstance(coeff, (list, tuple)) and len(coeff) == 1:
            arr = np.asarray(coeff[0])
            if arr.ndim == 2:
                return arr
        raise TypeError(
            "Unsupported mo_coeff layout for RHF/RKS object in MO one-electron integrals"
        )

    if (
        isinstance(mf.mo_coeff, (list, tuple)) and len(mf.mo_coeff) == 2
    ) or (
        isinstance(mf.mo_coeff, np.ndarray) and mf.mo_coeff.ndim >= 3 and mf.mo_coeff.shape[0] == 2
    ):
        Ca, Cb = _trexio_get_uks_coeff_pair(mf, expect_gamma=is_pbc)
        C = _trexio_concat_spin_coeff(Ca, Cb)
    else:
        C = _get_rhf_coeff(mf)

    if is_pbc and C.ndim == 3:
        if C.shape[0] != 1:
            raise NotImplementedError(
                "MO one-electron integrals currently support single-k Gamma calculations only."
            )
        C = C[0]

    if C.ndim != 2:
        raise ValueError(f"MO coefficient matrix must be 2D, got shape {C.shape}")

    def _ao_to_mo(mat, coeff):
        return _hermitize(coeff.conj().T @ mat @ coeff)

    mo_overlap = np.ascontiguousarray(
        _trexio_ensure_real(
            _ao_to_mo(overlap, C),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    mo_kinetic = np.ascontiguousarray(
        _trexio_ensure_real(
            _ao_to_mo(kinetic, C),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    mo_potential = np.ascontiguousarray(
        _trexio_ensure_real(
            _ao_to_mo(potential, C),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    mo_core = np.ascontiguousarray(
        _trexio_ensure_real(
            _ao_to_mo(core, C),
            what="Complex one-electron integrals encountered but the backend is real-only.",
        )
    )
    mo_ecp = None
    if ecp_mat is not None:
        mo_ecp = np.ascontiguousarray(
            _trexio_ensure_real(
                _ao_to_mo(ecp_mat, C),
                what="Complex one-electron integrals encountered but the backend is real-only.",
            )
        )

    _write_1e_int_eri(
        mo_overlap, mo_kinetic, mo_potential, mo_core, filename, backend, 'MO'
    )
    if mo_ecp is not None:
        with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
            trexio.write_mo_1e_int_ecp(tf, mo_ecp)

def _write_1e_int_eri(overlap, kinetic, potential, core, filename, backend='h5', basis='AO'):
    basis = basis.upper()
    if basis not in ('AO', 'MO'):
        raise ValueError("basis must be either 'AO' or 'MO'")

    overlap = np.ascontiguousarray(overlap)
    kinetic = np.ascontiguousarray(kinetic)
    potential = np.ascontiguousarray(potential)
    core = np.ascontiguousarray(core)

    with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
        if basis == 'AO':
            ao_dim = overlap.shape[0]
            if not trexio.has_ao_num(tf):
                trexio.write_ao_num(tf, ao_dim)
            trexio.write_ao_1e_int_overlap(tf, overlap)
            trexio.write_ao_1e_int_kinetic(tf, kinetic)
            trexio.write_ao_1e_int_potential_n_e(tf, potential)
            trexio.write_ao_1e_int_core_hamiltonian(tf, core)
        else:
            mo_dim = overlap.shape[0]
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, mo_dim)
            trexio.write_mo_1e_int_overlap(tf, overlap)
            trexio.write_mo_1e_int_kinetic(tf, kinetic)
            trexio.write_mo_1e_int_potential_n_e(tf, potential)
            trexio.write_mo_1e_int_core_hamiltonian(tf, core)

def write_1b_rdm(mf, filename, backend='h5'):
    """Write a one-body reduced density matrix in MO basis to TREXIO.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. Uses ``mo_occ`` to
        build diagonal MO densities. PBC data must be Gamma-only.
    filename : str
        Path to the TREXIO file to create or update.
    backend : {'h5', 'text'}, optional
        TREXIO backend selector passed through to ``trexio.File``.

    Behavior
    --------
    - MO basis is enforced (TREXIO convention).
    - RHF/RKS: writes a spin-summed diagonal density ``diag(mo_occ)``.
    - UHF/UKS: writes spin-blocked ``rdm_1e_up``/``rdm_1e_dn`` and a
      block-diagonal spin-summed matrix. Alpha/beta MO counts must match.
    - PBC: only single-k Gamma is supported; k-resolved occupations are
      squeezed to 1D first.
    - Real-only backend: imaginary parts larger than 1e-12 raise
      ``NotImplementedError``.

    Raises
    ------
    ValueError
        When alpha/beta dimensions differ or input shapes are inconsistent.
    NotImplementedError
        For complex densities or non-Gamma PBC calculations.
    """

    is_pbc = hasattr(mf, 'cell')
    if is_pbc and not _trexio_is_gamma_single_k(mf):
        raise NotImplementedError("RDM write supports Gamma-point only for PBC.")

    is_uhf_like = isinstance(
        mf,
        (
            scf.uhf.UHF,
            dft.uks.UKS,
            pbc.scf.uhf.UHF,
            pbc.dft.uks.UKS,
            pbc.scf.kuhf.KUHF,
            pbc.dft.kuks.KUKS,
        ),
    )

    # MO-basis density is diagonal in canonical orbitals
    if is_uhf_like and isinstance(mf.mo_occ, (tuple, list)) and len(mf.mo_occ) == 2:
        occ_a, occ_b = mf.mo_occ
        occ_a = np.asarray(occ_a)
        occ_b = np.asarray(occ_b)
    else:
        occ = np.asarray(mf.mo_occ)
        if is_uhf_like and occ.ndim == 2 and occ.shape[0] == 2:
            occ_a, occ_b = occ[0], occ[1]
        else:
            occ_a = occ
            occ_b = None

    if is_pbc and isinstance(occ_a, np.ndarray) and occ_a.ndim == 2:
        if occ_a.shape[0] != 1:
            raise NotImplementedError(
                "PBC RDM write: only single-k Gamma supported for MO basis.")
        occ_a = occ_a[0]
        if occ_b is not None and occ_b.ndim == 2:
            occ_b = occ_b[0]

    if occ_b is not None:
        dm_a = np.diag(occ_a)
        dm_b = np.diag(occ_b)
        dm_a = _trexio_ensure_real(
            np.asarray(dm_a),
            what="Complex RDM encountered; real-only backend.",
        )
        dm_b = _trexio_ensure_real(
            np.asarray(dm_b),
            what="Complex RDM encountered; real-only backend.",
        )
        dm_a = np.ascontiguousarray(dm_a)
        dm_b = np.ascontiguousarray(dm_b)
        with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
            nmo_up = dm_a.shape[0]
            nmo_dn = dm_b.shape[0]
            if nmo_dn != nmo_up:
                raise ValueError("Alpha/beta MO sizes do not match for RDM write.")
            nmo = nmo_up + nmo_dn
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, nmo)
            trexio.write_rdm_1e_up(tf, dm_a)
            trexio.write_rdm_1e_dn(tf, dm_b)
            dm_tot = np.zeros((nmo, nmo), dtype=dm_a.dtype)
            dm_tot[:nmo_up, :nmo_up] = dm_a
            dm_tot[nmo_up:, nmo_up:] = dm_b
            trexio.write_rdm_1e(tf, dm_tot)
    else:
        dm = np.diag(occ_a)
        dm = _trexio_ensure_real(
            np.asarray(dm),
            what="Complex RDM encountered; real-only backend.",
        )
        dm = np.ascontiguousarray(dm)
        with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
            nmo = dm.shape[0]
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, nmo)
            trexio.write_rdm_1e(tf, dm)

def write_2b_rdm(mf, filename, backend='h5', chunk_size=100000):
    """Write a two-body reduced density matrix in MO basis to TREXIO.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. Uses ``mo_occ`` to
        build diagonal MO densities. PBC data must be Gamma-only.
    filename : str
        Path to the TREXIO file to create or update.
    backend : {'h5', 'text'}, optional
        TREXIO backend selector passed through to ``trexio.File``.
    chunk_size : int, optional
        Number of elements streamed per block when writing flattened ERIs.

    Behavior
    --------
    - MO basis is enforced (TREXIO convention).
    - RHF/RKS: writes spin-summed 2-RDM using ``dm = diag(mo_occ)`` and
      ``G[pqrs] = dm[p,r]*dm[q,s] - 0.5*dm[p,s]*dm[q,r]``.
    - UHF/UKS: writes spin-resolved blocks ``G_uu``, ``G_dd``, and ``G_ud``
      constructed from ``diag(occ_a)`` and ``diag(occ_b)``; alpha/beta sizes
      must match.
    - PBC: only single-k Gamma is supported; k-resolved occupations are
      squeezed to 1D first.
    - Data are streamed in chunks to avoid holding the entire flattened array
      in memory at once; memory still scales as O(n^4).

    Raises
    ------
    ValueError
        When alpha/beta dimensions differ or input shapes are inconsistent.
    NotImplementedError
        For non-Gamma PBC calculations.
    """

    is_pbc = hasattr(mf, 'cell')
    if is_pbc and not _trexio_is_gamma_single_k(mf):
        raise NotImplementedError("RDM write supports Gamma-point only for PBC.")

    is_uhf_like = isinstance(
        mf,
        (
            scf.uhf.UHF,
            dft.uks.UKS,
            pbc.scf.uhf.UHF,
            pbc.dft.uks.UKS,
            pbc.scf.kuhf.KUHF,
            pbc.dft.kuks.KUKS,
        ),
    )

    # Spin-summed occupations or spin-separated for UHF/UKS
    if is_uhf_like and isinstance(mf.mo_occ, (tuple, list)) and len(mf.mo_occ) == 2:
        occ_a, occ_b = mf.mo_occ
        occ_a = np.asarray(occ_a)
        occ_b = np.asarray(occ_b)
    else:
        occ = np.asarray(mf.mo_occ)
        if is_uhf_like and occ.ndim == 2 and occ.shape[0] == 2:
            occ_a, occ_b = occ[0], occ[1]
        else:
            occ_a = occ
            occ_b = None

    if is_pbc and isinstance(occ_a, np.ndarray) and occ_a.ndim == 2:
        if occ_a.shape[0] != 1:
            raise NotImplementedError(
                "PBC RDM write: only single-k Gamma supported for MO basis.")
        occ_a = occ_a[0]
        if occ_b is not None and occ_b.ndim == 2:
            occ_b = occ_b[0]

    if occ_b is not None:
        dm_a = np.diag(occ_a)
        dm_b = np.diag(occ_b)
        dm_a = _trexio_ensure_real(
            np.asarray(dm_a),
            what="Complex RDM encountered; real-only backend.",
        )
        dm_b = _trexio_ensure_real(
            np.asarray(dm_b),
            what="Complex RDM encountered; real-only backend.",
        )
        dm_a = np.ascontiguousarray(dm_a)
        dm_b = np.ascontiguousarray(dm_b)

        g2_uu = (
            np.einsum('pr,qs->pqrs', dm_a, dm_a)
            - np.einsum('ps,qr->pqrs', dm_a, dm_a)
        )
        g2_dd = (
            np.einsum('pr,qs->pqrs', dm_b, dm_b)
            - np.einsum('ps,qr->pqrs', dm_b, dm_b)
        )
        g2_ud = np.einsum('pr,qs->pqrs', dm_a, dm_b)

        g2_uu = np.ascontiguousarray(g2_uu)
        g2_dd = np.ascontiguousarray(g2_dd)
        g2_ud = np.ascontiguousarray(g2_ud)

        nmo = dm_a.shape[0]
        if dm_b.shape[0] != nmo:
            raise ValueError("Alpha/beta MO sizes do not match for RDM write.")
        idx = lib.cartesian_prod([np.arange(nmo, dtype=np.int32)] * 4)
        flat_uu = g2_uu.reshape(-1)
        flat_dd = g2_dd.reshape(-1)
        flat_ud = g2_ud.reshape(-1)

        with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, nmo)

            total = idx.shape[0]
            offset = 0
            while offset < total:
                end = min(offset + chunk_size, total)
                trexio.write_rdm_2e_upup(
                    tf,
                    offset,
                    end - offset,
                    idx[offset:end],
                    flat_uu[offset:end],
                )
                trexio.write_rdm_2e_dndn(
                    tf,
                    offset,
                    end - offset,
                    idx[offset:end],
                    flat_dd[offset:end],
                )
                trexio.write_rdm_2e_updn(
                    tf,
                    offset,
                    end - offset,
                    idx[offset:end],
                    flat_ud[offset:end],
                )
                offset = end
    else:
        dm = np.diag(occ_a)
        dm = _trexio_ensure_real(
            np.asarray(dm),
            what="Complex RDM encountered; real-only backend.",
        )
        dm = np.ascontiguousarray(dm)

        # Build dense spin-summed 2-RDM in MO basis
        g2 = (
            np.einsum('pr,qs->pqrs', dm, dm)
            - 0.5 * np.einsum('ps,qr->pqrs', dm, dm)
        )
        g2 = np.ascontiguousarray(g2)

        nmo = dm.shape[0]
        idx = lib.cartesian_prod([np.arange(nmo, dtype=np.int32)] * 4)
        flat_g2 = g2.reshape(-1)

        with trexio.File(filename, 'u', back_end=_mode(backend)) as tf:
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, nmo)

            total = idx.shape[0]
            offset = 0
            while offset < total:
                end = min(offset + chunk_size, total)
                trexio.write_rdm_2e(
                    tf,
                    offset,
                    end - offset,
                    idx[offset:end],
                    flat_g2[offset:end],
                )
                offset = end


def _order_ao_index(mol):
    if mol.cart:
        return np.arange(mol.nao)

    # reorder spherical functions to
    # Pz, Px, Py
    # D 0, D+1, D-1, D+2, D-2
    # F 0, F+1, F-1, F+2, F-2, F+3, F-3
    # G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
    # ...
    lmax = mol._bas[:, gto.ANG_OF].max()
    cache_by_l = [np.array([0]), np.array([2, 0, 1])]
    for l in range(2, lmax + 1):
        idx = np.empty(l * 2 + 1, dtype=int)
        idx[::2] = l - np.arange(0, l + 1)
        idx[1::2] = np.arange(l + 1, l + l + 1)
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
    if code == "h5":
        return 0
    else:  # trexio text
        return 1


def _group_by(a, keys):
    """keys must be sorted"""
    assert all(a <= b for a, b in zip(keys, keys[1:]))
    idx = np.unique(keys, return_index=True)[1]
    return np.split(a, idx[1:])


def _get_occsa_and_occsb(mcscf, norb, nelec, ci_threshold=0.0):
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
            if (
                np.abs(ci_coeff) > ci_threshold
            ):  # Check if CI coefficient is significant compared to user defined value
                occsa.append(selected_occslst[i])
                occsb.append(selected_occslst[j])
                ci_values.append(ci_coeff)

    # Sort by the absolute value of the CI coefficients in descending order
    sorted_indices = np.argsort(-np.abs(ci_values))
    occsa_sorted = [occsa[idx] for idx in sorted_indices]
    occsb_sorted = [occsb[idx] for idx in sorted_indices]
    ci_values_sorted = [ci_values[idx] for idx in sorted_indices]

    return occsa_sorted, occsb_sorted, ci_values_sorted, num_determinants


def _det_to_trexio(
    mcscf, norb, nelec, trexio_file, ci_threshold=0.0, chunk_size=100000
):
    ncore = mcscf.ncore
    int64_num = trexio.get_int64_num(trexio_file)

    occsa, occsb, ci_values, num_determinants = _get_occsa_and_occsb(
        mcscf, norb, nelec, ci_threshold
    )

    det_list = []
    for a, b, coeff in zip(occsa, occsb, ci_values):
        occsa_upshifted = [orb for orb in range(ncore)] + [orb + ncore for orb in a]
        occsb_upshifted = [orb for orb in range(ncore)] + [orb + ncore for orb in b]
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
            trexio.write_determinant_list(
                trexio_file, offset_file, current_chunk_size, det_list[start:end]
            )
            trexio.write_determinant_coefficient(
                trexio_file, offset_file, current_chunk_size, ci_values[start:end]
            )
            offset_file += current_chunk_size


def _to_segment_contraction(mol):
    """transform generally contracted basis to segment contracted basis"""
    _bas = []
    for shell in mol._bas:
        nctr = shell[gto.NCTR_OF]
        if nctr == 1:
            _bas.append(shell)
            continue

        nprim = shell[gto.NPRIM_OF]
        pcoeff = shell[gto.PTR_COEFF]
        bs = np.repeat(shell[np.newaxis], nctr, axis=0)
        bs[:, gto.NCTR_OF] = 1
        bs[:, gto.PTR_COEFF] = np.arange(pcoeff, pcoeff + nprim * nctr, nprim)
        _bas.append(bs)

    pmol = mol.copy()
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    return pmol
