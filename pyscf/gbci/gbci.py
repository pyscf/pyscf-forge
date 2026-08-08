#!/usr/bin/env python
#
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#
# Authors: Jiseong Park <fark4308@snu.ac.kr>
#          Minseok Oh <msjeff2001@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

'''
Grouped-Bath Configuration Interaction (GBCI)

References:
[1] Efficient grouped-bath ansatz for spin-flip non-orthogonal configuration
    interaction in transition-metal charge-transfer complexes
    Jiseong Park and Seunghoon Lee
    J. Chem. Theory Comput. 2025
[2] Spin-flip non-orthogonal configuration interaction: a variational and
    almost black-box method for describing strongly correlated molecules
    Nicholas J. Mayhall, Paul R. Horn, Eric J. Sundstrom and Martin Head-Gordon
    Phys. Chem. Chem. Phys. 2014, 16, 22694
'''

import numpy
import numpy as np
from itertools import product

from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import rohf
from pyscf.mcscf.casci import CASBase, CASCI
from pyscf.fci import cistring
from pyscf.gbci.direct_gbci import (
    GBCISolver, SpinPenaltyGBCISolver, fix_spin_, str2occ)
from pyscf.gbci import rdm

TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)

def kernel(gbci, mo_coeff=None, ci0=None, verbose=logger.NOTE):
    '''GBCI solver.

    With ``group_a=None`` this follows the original GBCI bath construction.
    With ``group_a`` set, occupation patterns are grouped and the grouped-bath
    GBCI path is used.

    Args:
        gbci: GBCI object

        mo_coeff : ndarray
            orbitals to construct active space Hamiltonian
        ci0 : ndarray or custom types
            FCI sovler initial guess. For external FCI-like solvers, it can be
            overloaded different data type. For example, in the state-average
            FCI solver, ci0 is a list of ndarray. In other solvers such as
            DMRGCI solver, SHCI solver, ci0 are custom types.

    kwargs:
        envs: dict
            The variable envs is created (for PR 807) to passes MCSCF runtime
            environment variables to SHCI solver. For solvers which do not
            need this parameter, a kwargs should be created in kernel method
            and "envs" pop in kernel function
    '''
    if mo_coeff is None: mo_coeff = gbci.mo_coeff
    if ci0 is None: ci0 = gbci.ci

    log = logger.new_logger(gbci, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    grouped = gbci.group_a is not None
    method_name = 'GBCI'
    log.debug('Start %s', method_name)

    ncas = gbci.ncas
    nelecas = gbci.nelecas

    # FASSCF
    mo_list, mo_energy, po_list, group = gbci.optimize_mo(mo_coeff)
    log.debug('GBCI bath groups: %s', group)
    if grouped:
        conf_info_list = group_info_list(ncas, nelecas, po_list, group)
        svd_basis = group
    else:
        conf_info_list = group_info_list(ncas, nelecas, po_list)
        svd_basis = po_list
    t1 = log.timer('FASSCF', *t0)

    # SVD and core density matrix
    dmet_core_list, ov_list = gbci.get_svd_matrices(mo_list, svd_basis)
    t1 = log.timer('SVD and core density matrix', *t1)
    gbci._cache_gbci_intermediates(
        mo_coeff, ncas, nelecas, gbci.ncore,
        {
            "mo_list": mo_list,
            "mo_energy": mo_energy,
            "po_list": po_list,
            "group": group,
            "svd_basis": svd_basis,
            "conf_info_list": conf_info_list,
            "dmet_core_list": dmet_core_list,
            "ov_list": ov_list,
        })

    # 1e
    dmet_act_list = gbci.get_active_dm(mo_coeff)
    h1e, ecore_list = gbci.get_h1cas(dmet_act_list, mo_list, dmet_core_list)
    t1 = log.timer('effective 1e hamiltonians and core energies', *t1)

    # 2e
    eri = gbci.get_h2eff(mo_coeff)
    t1 = log.timer('effective 2e hamiltonian', *t1)

    max_memory = max(400, gbci.max_memory-lib.current_memory()[0])
    e_tot, fcivec = gbci.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                            conf_info_list, ov_list, ecore_list,
                                            ci0=ci0, verbose=log,
                                            max_memory=max_memory)

    log.timer('%s solver' % method_name, *t1)
    log.timer('All %s process' % method_name, *t0)

    if isinstance(e_tot, (float, numpy.float64)):
        e_cas = e_tot - ecore_list
        if not grouped:
            e_cas = [e_cas]
            e_tot = [e_tot]
    else:
        e_cas = [e - ecore_list for e in e_tot]

    return e_tot, e_cas, fcivec

def possible_occ(n_as, n_ase):
    spin_resolved = not numpy.isscalar(n_ase)
    if spin_resolved:
        neleca, nelecb = n_ase
        neleca = int(neleca)
        nelecb = int(nelecb)
        n_ase = neleca + nelecb
    else:
        n_ase = int(n_ase)

    possible_values = [0, 1, 2]
    result_arrays = []
    for combination in product(possible_values, repeat=n_as):
        if sum(combination) != n_ase:
            continue
        if spin_resolved:
            ndouble = combination.count(2)
            nsingle = combination.count(1)
            alpha_singles = neleca - ndouble
            if alpha_singles < 0 or alpha_singles > nsingle:
                continue
        result_arrays.append(numpy.array(combination))

    concantenated_array=numpy.array(result_arrays, order = 'C', dtype = numpy.int32)
    return concantenated_array

def mo_overlap(mo1, mo2, s1e):
    mo_overlap_list = lib.einsum('ai,bj,ij->ab', numpy.conjugate(mo1.T), mo2.T, s1e)
    return mo_overlap_list

def biorthogonalize(mo1, mo2, s1e):
    u, s, vt = numpy.linalg.svd(mo_overlap(mo1, mo2, s1e))
    mo1_bimo_coeff = mo1.dot(u)
    mo2_bimo_coeff = mo2.dot(vt.T)
    return s, mo1_bimo_coeff, mo2_bimo_coeff, u, vt

def group_info_list(ncas, nelecas, po_list, group=None):
    stringsa = cistring.make_strings(range(0, ncas), nelecas[0])
    stringsb = cistring.make_strings(range(0, ncas), nelecas[1])
    na = len(stringsa)
    nb = len(stringsb)

    occ_to_p = {tuple(row.tolist()): i for i, row in enumerate(po_list)}

    if group is not None:
        p_to_group = {}
        for ig, members in enumerate(group):
            for p in members:
                p_to_group[int(p)] = ig

    occs_a = numpy.asarray([str2occ(s, ncas) for s in stringsa], dtype=numpy.int8)
    occs_b = numpy.asarray([str2occ(s, ncas) for s in stringsb], dtype=numpy.int8)

    group_info = numpy.empty((na, nb), dtype=numpy.int32)

    for ia in range(na):
        oa = occs_a[ia]
        for ib in range(nb):
            occ = tuple((oa + occs_b[ib]).tolist())
            p = occ_to_p[occ]
            if group is not None:
                p = p_to_group[p]
            group_info[ia, ib] = p

    return group_info.astype(int)

def group_occ(po_list, group):
    best_row = None
    best_one_count = -1
    best_two_score = numpy.inf
    #best_one_score = numpy.inf
    best_zero_score = numpy.inf

    for idx in group:
        row = po_list[idx]
        one_count = numpy.count_nonzero(row==1)
        two_positions = numpy.where(row==2)[0]
        two_score = sum(two_positions) if len(two_positions) > 0 else numpy.inf
        zero_positions = numpy.where(row == 0)[0]
        zero_score = sum(len(row) - zero_positions) if len(zero_positions) > 0 else numpy.inf

        if one_count > best_one_count:
            best_row = idx
            best_one_count = one_count
            best_two_score = two_score
            best_zero_score = zero_score
        elif one_count == best_one_count:
            if two_score < best_two_score:
                best_row = idx
                best_two_score = two_score
                best_zero_score = zero_score
            if zero_score < best_zero_score:
                best_row = idx
                best_two_score = two_score
                best_zero_score = zero_score

    return po_list[best_row]

def group_by_occ(po_list, group_a):
    a = len(group_a)
    p = len(po_list)
    A_occ = numpy.zeros((p,a))
    for index, occ in enumerate(po_list):
        for i in range(a):
            A_occ[index][i] = numpy.sum(occ[group_a[i]])
    grouped_rows = {}
    for i, row in enumerate(A_occ):
        row_tuple = tuple(row)
        if row_tuple not in grouped_rows:
            grouped_rows[row_tuple]=[]
        grouped_rows[row_tuple].append(i)
    return list(grouped_rows.values())


def group_by_occ_index(po_list, group_a, label="occ"):
    """Group possible occupation patterns by explicit PO indices."""
    p = len(po_list)
    seen = []
    group = []
    for ig, members in enumerate(group_a):
        normalized = []
        for member in members:
            idx = int(member)
            if idx < 0 or idx >= p:
                raise IndexError(
                    f"Occupation group {ig} references PO index {idx}, "
                    f"but possible_occ has size {p}."
                )
            normalized.append(idx)
            seen.append(idx)
        group.append(normalized)

    expected = list(range(p))
    if sorted(seen) != expected:
        raise ValueError(
            f"group_a {{'{label}': ...}} must cover each possible occupation index "
            f"exactly once; expected {expected}, got {seen}."
        )
    return group


_GROUPA_MODE_KEYS = ("atom", "mo", "occ")


def _is_int_like(x):
    return isinstance(x, (int, numpy.integer))


def _normalize_mo_groups(groups):
    if isinstance(groups, numpy.ndarray):
        groups = groups.tolist()
    if groups is None:
        raise ValueError("group_a {'mo': ...} requires active-orbital groups.")
    if isinstance(groups, (str, bytes)) or isinstance(groups, dict):
        raise TypeError("group_a {'mo': ...} must be a list of MO-index groups.")
    groups = list(groups)
    if not groups:
        raise ValueError("group_a {'mo': ...} must not be empty.")
    if all(_is_int_like(x) for x in groups):
        # Preserve the old flat-list behavior: group_a=[0, 1] meant two
        # one-orbital occupation groups, not one combined [0, 1] group.
        return [[int(x)] for x in groups]
    out = []
    for i, group in enumerate(groups):
        if isinstance(group, numpy.ndarray):
            group = group.tolist()
        if _is_int_like(group):
            out.append([int(group)])
            continue
        if isinstance(group, (str, bytes)) or isinstance(group, dict):
            raise TypeError(f"MO group {i} must be a sequence of integer indices.")
        g = [int(x) for x in group]
        if not g:
            raise ValueError(f"MO group {i} must not be empty.")
        out.append(g)
    return out


def _normalize_occ_groups(groups, label="occ"):
    if isinstance(groups, numpy.ndarray):
        groups = groups.tolist()
    if groups is None:
        raise ValueError(
            f"group_a {{'{label}': ...}} requires occupation-index groups."
        )
    if isinstance(groups, (str, bytes)) or isinstance(groups, dict):
        raise TypeError(
            f"group_a {{'{label}': ...}} must be a list of "
            "possible-occupation index groups."
        )
    groups = list(groups)
    if not groups:
        raise ValueError(f"group_a {{'{label}': ...}} must not be empty.")
    if all(_is_int_like(x) for x in groups):
        return [[int(x)] for x in groups]
    out = []
    for i, group in enumerate(groups):
        if isinstance(group, numpy.ndarray):
            group = group.tolist()
        if _is_int_like(group):
            out.append([int(group)])
            continue
        if isinstance(group, (str, bytes)) or isinstance(group, dict):
            raise TypeError(
                f"Occupation group {i} must be a sequence of integer indices."
            )
        g = [int(x) for x in group]
        if not g:
            raise ValueError(f"Occupation group {i} must not be empty.")
        out.append(g)
    return out


def _normalize_atom_groups(groups):
    if isinstance(groups, numpy.ndarray):
        groups = groups.tolist()
    if groups is None:
        raise ValueError("group_a {'atom': ...} requires atom-index groups.")
    if isinstance(groups, dict):
        raise TypeError(
            "group_a {'atom': ...} must be atom indices, not named groups. "
            "Use {'atom': [0, 1]} for one multi-atom fragment or "
            "{'atom': [[0, 1], [2, 3]]} for multi-atom fragments."
        )
    if isinstance(groups, (str, bytes)):
        raise TypeError("group_a {'atom': ...} must be atom indices, not AO labels.")
    groups = list(groups)
    if not groups:
        raise ValueError("group_a {'atom': ...} must not be empty.")
    if all(_is_int_like(x) for x in groups):
        return [[int(x) for x in groups]]
    out = []
    for i, atoms in enumerate(groups):
        if isinstance(atoms, numpy.ndarray):
            atoms = atoms.tolist()
        if _is_int_like(atoms):
            out.append([int(atoms)])
            continue
        if isinstance(atoms, (str, bytes)) or isinstance(atoms, dict):
            raise TypeError(f"Atom group {i} must be a sequence of atom indices.")
        g = [int(x) for x in atoms]
        if not g:
            raise ValueError(f"Atom group {i} must not be empty.")
        out.append(g)
    return out


def normalize_group_a(group_a):
    """Return the canonical grouped-bath specification.

    Public forms:
        {"atom": [0, 1]}
        {"atom": [[0], [1]]}
        {"atom": [[0, 1], [2, 3]]}
        {"mo": [[0, 1], [2, 3]]}
        {"occ": [[0, 2], [1]]}
    """
    if group_a is None:
        return None

    if isinstance(group_a, dict):
        if "kind" in group_a:
            kind = str(group_a["kind"]).strip().lower()
            raw = group_a.get("groups")
        else:
            mode_keys = [key for key in _GROUPA_MODE_KEYS if key in group_a]
            if mode_keys:
                if len(mode_keys) != 1:
                    raise ValueError(f"group_a must specify exactly one of {_GROUPA_MODE_KEYS}, got {mode_keys}")
                kind = mode_keys[0]
                raw = group_a[kind]
            else:
                unsupported = [
                    key for key in ("ao", "aolabel", "bath", "initial", "initial_occ",
                                    "relax", "relax_groups")
                    if key in group_a
                ]
                if unsupported:
                    raise ValueError(
                        "Unsupported group_a keys %s; use {'atom': [[0], [1]]} "
                        "for atom-index grouping or {'occ': ...} for explicit "
                        "possible-occupation grouping." % unsupported
                    )
                raise ValueError(
                    "group_a must specify exactly one mode: "
                    "{'atom': ...}, {'mo': ...}, or {'occ': ...}."
                )

        spec = {"kind": kind}
        if "threshold" in group_a:
            spec["threshold"] = float(group_a["threshold"])
        elif "thres" in group_a:
            spec["threshold"] = float(group_a["thres"])

        if kind == "atom":
            spec["groups"] = _normalize_atom_groups(raw)
        elif kind == "mo":
            spec["groups"] = _normalize_mo_groups(raw)
        elif kind == "occ":
            spec["groups"] = _normalize_occ_groups(raw)
        else:
            raise ValueError(f"Unknown group_a kind {kind!r}; expected one of {_GROUPA_MODE_KEYS}.")
        return spec

    if isinstance(group_a, str):
        raise TypeError(
            "AO-label group_a strings are not supported. Use atom-index "
            "grouping such as {'atom': [[0], [1]]}."
        )

    if isinstance(group_a, (list, tuple, numpy.ndarray)):
        raise TypeError(
            "Bare group_a lists are ambiguous. Use {'atom': ...}, "
            "{'mo': ...}, or {'occ': ...}."
        )

    raise TypeError(
        "group_a must be None or a dict form such as {'atom': [[0], [1]]}, "
        "{'mo': [[0, 1]]}, or {'occ': [[0, 2], [1]]}."
    )


def fragment_aos_by_atoms(mol, atom_ids):
    sl = mol.aoslice_by_atom()
    aos = []
    for a in atom_ids:
        if a < 0 or a >= mol.natm:
            raise IndexError(
                f"Atom group references atom index {a}, but mol has "
                f"{mol.natm} atoms."
            )
        p0, p1 = sl[a, 2], sl[a, 3]
        aos.extend(range(p0, p1))
    return aos

def group_by_atom(mol, ac_mo_coeff, po_list, atom_groups, thres=0.2):
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    e, v = np.linalg.eigh(ova)
    s12 = (v * np.sqrt(e)) @ v.T.conj()
    atom_groups = _normalize_atom_groups(atom_groups)
    aolist = []
    for atoms in atom_groups:
        ao_frag=fragment_aos_by_atoms(mol, atoms)
        aolist.append(ao_frag)

    p = len(po_list)
    ao_elecnums = np.zeros((p, len(aolist)))
    for i in range(p):
        one_list = np.where(po_list[i] == 1)[0]
        two_list = np.where(po_list[i] == 2)[0]

        # density-like matrix in AO basis (active only)
        pT1 = ac_mo_coeff[:, one_list] @ ac_mo_coeff[:, one_list].T
        pT2 = ac_mo_coeff[:, two_list] @ ac_mo_coeff[:, two_list].T
        pT  = pT1 + 2.0 * pT2

        pTOAO = s12 @ pT @ s12

        # Löwdin population sum on selected AOs
        diagP = np.diag(pTOAO)
        for j, ao_frag in enumerate(aolist):
            ao_elecnums[i, j] = np.sum(diagP[ao_frag])

    # grouping by similarity
    groups = []
    visited = set()
    for i in range(p):
        if i in visited:
            continue
        g = [i]
        visited.add(i)
        for j in range(p):
            if j not in visited and np.all(np.abs(ao_elecnums[i] - ao_elecnums[j]) <= thres):
                g.append(j)
                visited.add(j)
        groups.append(g)
    return groups

def optimize_mo(gbci, mo_coeff=None, ncas=None, nelecas=None, ncore=None,
                group_a=None):
    if mo_coeff is None : mo_coeff = gbci.mo_coeff
    if ncas is None : ncas = gbci.ncas
    if nelecas is None : nelecas = gbci.nelecas
    if ncore is None : ncore = gbci.ncore
    log = logger.new_logger(gbci)
    group_spec = normalize_group_a(group_a)

    po_list = possible_occ(ncas, nelecas)

    N=mo_coeff.shape[0]
    # Keep optimized mo_energy for gradient code.
    if group_spec is None:
        group = [[i] for i in range(len(po_list))]
        optimized_mo = numpy.zeros((len(po_list), N, N))
        optimized_mo_energy = numpy.zeros((len(po_list), N))
        fasscf = gbci._get_fasscf(
            group_spec, mo_coeff, ncas, nelecas, ncore)
        for i, occ in enumerate(po_list):
            result = fasscf.mixed_routine(
                target=occ, **gbci._mixed_routine_options(fasscf))
            conv, et, moe, moce, moocc = result.as_tuple()
            log.debug("occupation pattern index: %d", i)
            log.debug("FASSCF converged=%s energy=%s", conv, et)
            optimized_mo[i] = moce
            if moe is None:
                moe = gbci.mo_energy
            optimized_mo_energy[i] = moe
        return optimized_mo, optimized_mo_energy, po_list, group

    if group_spec["kind"] == "atom":
        group = group_by_atom(gbci.mol, mo_coeff[:, ncore:ncore+ncas], po_list,
                              group_spec["groups"], thres=group_spec.get("threshold", gbci._thres))
    elif group_spec["kind"] == "mo":
        group = group_by_occ(po_list, group_spec["groups"])
    elif group_spec["kind"] == "occ":
        group = group_by_occ_index(po_list, group_spec["groups"], label="occ")
    else:
        raise NotImplementedError(f"Unsupported group_a kind {group_spec['kind']!r}")

    g = len(group)
    group_info_flat = group_info_list(ncas, nelecas, po_list, group).reshape(-1)
    optimized_mo = numpy.zeros((g,N,N))
    optimized_mo_energy = numpy.zeros((g,N))
    fasscf = gbci._get_fasscf(
        group_spec, mo_coeff, ncas, nelecas, ncore)
    for i in range(0,g):
        result = fasscf.mixed_routine(
            target_group=i, group_info_list=group_info_flat,
            **gbci._mixed_routine_options(fasscf))
        conv, et, moe, moce = result.as_tuple()
        optimized_mo[i]=moce
        if moe is None:
            moe = gbci.mo_energy
        optimized_mo_energy[i]=moe
        log.debug("group index: %d", i)
        log.debug("state_average_fasscf converged=%s energy=%s", conv, et)

    return optimized_mo, optimized_mo_energy, po_list, group

def h1e_for_gbci(gbci, dmet_act_list=None, mo_list=None, dmet_core_list=None,
                   ncas=None, ncore=None):
    ''' GBCI space one-electron hamiltonian

    Args:
        gbci : a GBCI object

    Returns:
        A tuple, A tuple, the first is the effective one-electron hamiltonian defined in GBCI space,
        the second is the list of electronic energy from baths.
    '''
    if ncas is None : ncas = gbci.ncas
    if ncore is None : ncore = gbci.ncore
    if mo_list is None:
        mo_list, mo_energy, po_list, group = gbci.optimize_mo(gbci.mo_coeff)
    if dmet_core_list is None:
        dmet_core_list, ov_list = gbci.get_svd_matrices(mo_list)
    if dmet_act_list is None:
        dmet_act_list = gbci.get_active_dm(gbci.mo_coeff)
    p = dmet_core_list.shape[0]
    mo_cas = mo_list[0][:,ncore:ncore+ncas]
    hcore = gbci.get_hcore()

    h1e = numpy.zeros((p,p,ncas,ncas))
    ecore_list = numpy.zeros(p)
    energy_nuc = gbci.energy_nuc()
    ha1e = lib.einsum('ai,ab,bj->ij',mo_cas,hcore,mo_cas)

    for i in range(0,p):
        for j in range(0,p):
            if i >= j:
                corevhf = gbci.get_veff(dm = 2*dmet_core_list[i,j],hermi = 0)
                h1e[i,j] = ha1e + lib.einsum('ijab,ba -> ij', dmet_act_list , corevhf)
                h1e[j,i] = h1e[i,j].T
            if i==j:
                ecore_list[i] += lib.einsum('ab,ab -> ', dmet_core_list[i,i],corevhf)
                ecore_list[i] += energy_nuc
                ecore_list[i] += 2*lib.einsum('ab,ab->', dmet_core_list[i,i], hcore)
    gbci.h1e = h1e
    gbci.core_energies = ecore_list
    return h1e, ecore_list

def spin_square(gbci, rdm1, rdm2ab,rdm2ba):
    M_s = gbci.spin/2
    mo = gbci.mo_coeff
    s1e = gbci.mol.intor('int1e_ovlp')
    rdm1mo = lib.einsum('qi,pl,kj,qp,lk->ij', mo, rdm1, mo,s1e,s1e)
    rdm2mo = lib.einsum('ai,bj,ck,dl,ap,bq,cr,ds,pqrs',mo,mo,mo,mo,s1e,s1e,s1e,s1e,rdm2ab+rdm2ba)

    return M_s**2 + 0.5*lib.einsum('ii ->',rdm1mo) - 0.5*lib.einsum('ijji ->', rdm2mo)

class GBCI(CASBase):
    '''GBCI

    Args:
        mf : SCF object
            SCF to define the problem size and SCF type of FASSCF.
            The ROHF object is recommended.
        ncas : int
            Number of active orbitals
        nelecas : a pair of int
            Number of electrons in active space

    Kwargs:
        ncore : int
            Number of doubly occupied core orbitals. If not presented, this
            parameter can be automatically determined.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        ncas : int
            Active space size.
        nelecas : tuple of int
            Active (nelec_alpha, nelec_beta)
        ncore : int or tuple of int
            Core electron number.
        fcisolver : an instance of :class:`FCISolver`
            The GBCISolver in pyscf.gbci.direct_gbci module must be used.
            Other moldules in pyscf.fci cannot be used.
            You can control FCIsolver by setting e.g.::

            >>> mc.fcisolver.max_cycle = 30
            >>> mc.fcisolver.conv_tol = 1e-7

    Key variables :
        N : The basis number

        po_list : A list of possible occupation patterns.
            for example, for (2e, 2o): po_list = [[0,2], [1,1], [2,0]]. It is 2D numpy array.

        mo_list : ndarray (nbath , N, N)
            The optimized molecular orbital set by FASSCF. the nbath is equal to length of po_list.

        conf_info_list : ndarray, (nstringsa, nstringsb)
            The optimized bath orbitals indices for each configuration.

        dmet_core_list : density matrix of core orbitals between different bath in atomic basis : (nbath, nbath, N, N)

        h1e : effective one electron hamiltonian : (nbath, nbath, ncas, ncas)

        ov_list : overlap between different bath : (nbath, nbath)

        dmet_act_list : density matrix between specific two active orbitals in atomic basis : (ncas, ncas, N, N)

        ecore_list : 1D numpy array of core energies for each bath : (ngroup)
  '''
    def __init__(self, mf, ncas, nelecas, ncore=None, group_a=None):
        mol = mf.mol
        self.mol = mol
        self._scf = mf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, (int, numpy.integer)):
            raise NotImplementedError
        else:
            self.nelecas = (nelecas[0], nelecas[1])
            self._spin = nelecas[0] - nelecas[1]
        self.ncore = ncore
        self.group_a = group_a

        self.fcisolver = GBCISolver(mol)
        self.fcisolver.lindep = getattr(__config__,
                                      'gbci_GBCI_fcisolver_lindep', 1e-14)
        self.fcisolver.max_cycle = getattr(__config__,
                                         'gbci_GBCI_fcisolver_max_cycle', 100)
        self.fcisolver.conv_tol = getattr(__config__,
                                        'gbci_GBCI_fcisolver_conv_tol', 5e-7)

################################################## don't modify the following attributes, they are not input options
        self.e_tot = 0
        self.e_cas = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy
        self.mo_occ = None
        self.converged = False
        self._thres = 0.2
        self._gbci_intermediates = None
        self._gbci_intermediates_cache = None
        self._fasscf = None
        self._get_fasscf(self._group_a, self.mo_coeff, self.ncas,
                         self.nelecas, self.ncore)

    @property
    def ncore(self):
        if self._ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            return ncorelec // 2
        else:
            return self._ncore
    @ncore.setter
    def ncore(self, x):
        assert x is None or isinstance(x, (int, numpy.integer))
        assert x is None or x >= 0
        self._ncore = x
        self._clear_gbci_intermediates()


    @property
    def group_a(self):
        return self._group_a

    @group_a.setter
    def group_a(self, x):
        self._group_a = normalize_group_a(x)
        if hasattr(self, '_fasscf'):
            self._fasscf = None
        self._clear_gbci_intermediates()

    @property
    def lowdin_thres(self):
        return self._thres

    @lowdin_thres.setter
    def lowdin_thres(self, x):
        self._thres = x
        self._clear_gbci_intermediates()

    @property
    def spin(self):
        if self._spin is None:
            return self.mol.spin
        else:
            return self._spin

    @spin.setter
    def spin(self,x):
        assert x is None or isinstance(x, (int, numpy.integer))
        self._spin = x
        nelecas = self.nelecas
        necas = nelecas[0] + nelecas[1]
        nelecb = (necas- x)//2
        neleca = necas - nelecb
        self.nelecas = (neleca,nelecb)
        self._clear_gbci_intermediates()

    def possible_occ(self):
        po_list = possible_occ(self.ncas, self.nelecas)
        return po_list

    def _clear_gbci_intermediates(self):
        if hasattr(self, '_gbci_intermediates_cache'):
            self._gbci_intermediates = None
            self._gbci_intermediates_cache = None

    def _gbci_intermediates_key(self, mo_coeff, ncas, nelecas, ncore):
        return (
            tuple(mo_coeff.shape), str(mo_coeff.dtype), int(ncas),
            tuple(nelecas), int(ncore), repr(self._group_a))

    def _cache_gbci_intermediates(self, mo_coeff, ncas, nelecas, ncore,
                                  intermediates):
        self._gbci_intermediates = intermediates
        self._gbci_intermediates_cache = {
            "key": self._gbci_intermediates_key(
                mo_coeff, ncas, nelecas, ncore),
            "mo_coeff": numpy.array(mo_coeff, copy=True),
            "data": intermediates,
        }

    def _get_cached_gbci_intermediates(self, mo_coeff, ncas, nelecas, ncore):
        cache = getattr(self, '_gbci_intermediates_cache', None)
        if cache is None:
            return None
        key = self._gbci_intermediates_key(mo_coeff, ncas, nelecas, ncore)
        if cache.get("key") != key:
            return None
        if not numpy.array_equal(cache.get("mo_coeff"), mo_coeff):
            return None
        return cache["data"]

    def _get_fasscf(self, group_a=None, mo_coeff=None, ncas=None,
                    nelecas=None, ncore=None):
        from pyscf.gbci.fasscf import FASSCF as FASSCFDriver, GroupAverageFASSCF
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore

        fasscf_class = GroupAverageFASSCF if group_a is not None else FASSCFDriver
        fasscf = getattr(self, '_fasscf', None)
        mixed_routine_options = getattr(fasscf, 'mixed_routine_options', None)
        if group_a is None:
            reuse_fasscf = (
                isinstance(fasscf, FASSCFDriver) and
                not isinstance(fasscf, GroupAverageFASSCF)
            )
        else:
            reuse_fasscf = isinstance(fasscf, GroupAverageFASSCF)
        if not reuse_fasscf:
            max_cycle = 200 if group_a is not None else self._scf.max_cycle
            if fasscf is not None:
                max_cycle = getattr(fasscf, 'max_cycle', max_cycle)
            fasscf = fasscf_class(
                self, mo_coeff=mo_coeff,
                conv_tol=getattr(fasscf, 'conv_tol', self._scf.conv_tol),
                conv_tol_grad=getattr(fasscf, 'conv_tol_grad', None),
                max_cycle=max_cycle,
                dump_chk=getattr(fasscf, 'fasscf_dump_chk', True),
                conv_check=getattr(fasscf, 'conv_check', True),
                callback=getattr(fasscf, 'callback', None),
                scf_options=getattr(fasscf, 'scf_options', None),
                restore_on_failure=getattr(fasscf, 'restore_on_failure', True))
            self._fasscf = fasscf

        if mixed_routine_options is None:
            mixed_routine_options = getattr(fasscf, 'mixed_routine_options', {})
        fasscf.mixed_routine_options = mixed_routine_options
        fasscf.gbci = self
        if getattr(fasscf, 'ncas', None) != ncas or getattr(fasscf, 'ncore', None) != ncore:
            fasscf.ncas = ncas
            fasscf.ncore = ncore
            fasscf.active_orbitals = list(range(ncore, ncore + ncas))
            fasscf.core_orbitals = list(range(ncore))
        fasscf.nelecas = nelecas
        fasscf.mo_coeff = mo_coeff
        fasscf.mo_energy = self.mo_energy
        return fasscf

    def _mixed_routine_options(self, fasscf=None):
        if fasscf is None:
            fasscf = self._fasscf
        mixed_routine_options = getattr(fasscf, 'mixed_routine_options', None)
        if mixed_routine_options is None:
            mixed_routine_options = {}
            fasscf.mixed_routine_options = mixed_routine_options
        return dict(mixed_routine_options)

    def fasscf(self, target, mo_coeff=None, ncas=None, ncore=None,
               nelecas=None, po_list=None, group=None, group_info_flat=None,
               conv_tol=1e-10, conv_tol_grad=None, max_cycle=100):
        """Run the FASSCF orbital optimization variant selected by ``group``.

        ``group is None`` keeps the original occupation-pattern path.  A
        non-None ``group`` selects the state-averaged grouped path.  Both paths
        return five values so callers can use one interface.
        """
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if ncas is None : ncas = self.ncas
        if ncore is None : ncore = self.ncore
        if nelecas is None : nelecas = self.nelecas
        if group is not None:
            if po_list is None:
                po_list = possible_occ(ncas, nelecas)
            if group_info_flat is None:
                group_info_flat = group_info_list(
                    ncas, nelecas, po_list, group).reshape(-1)
            fasscf = self._get_fasscf(
                self._group_a if self._group_a is not None else group,
                mo_coeff, ncas, nelecas, ncore)
            fasscf.conv_tol = conv_tol
            fasscf.conv_tol_grad = conv_tol_grad
            fasscf.max_cycle = max_cycle
            mixed_options = self._mixed_routine_options(fasscf)
            mixed_options.setdefault('max_cycle1', max_cycle)
            mixed_options.setdefault('max_cycle2', max_cycle)
            result = fasscf.mixed_routine(
                target_group=target, group_info_list=group_info_flat,
                **mixed_options)
            fasscf_conv, fasscf_e_tot, fasscf_mo_energy, fasscf_mo_coeff = (
                result.as_tuple())
            return (fasscf_conv, fasscf_e_tot, fasscf_mo_energy,
                    fasscf_mo_coeff, None)

        fasscf = self._get_fasscf(None, mo_coeff, ncas, nelecas, ncore)
        fasscf.conv_tol = conv_tol
        fasscf.conv_tol_grad = conv_tol_grad
        fasscf.max_cycle = max_cycle
        mixed_options = self._mixed_routine_options(fasscf)
        mixed_options.setdefault('max_cycle1', max_cycle)
        mixed_options.setdefault('max_cycle2', max_cycle)
        result = fasscf.mixed_routine(target=target, **mixed_options)
        return result.as_tuple()

    def fix_spin_(self, ss=0, shift=1):
        r'''Use level shift to control FCI solver spin.

        .. math::

            (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

        Kwargs:
            shift : float
                Energy penalty for states which have wrong spin
            ss : number
                S^2 expection value == s*(s+1)
        '''
        fix_spin_(self.fcisolver, shift, ss)
        return self
    fix_spin = fix_spin_

    def optimize_mo(self, mo_coeff=None, group_a=None, conv_tol=1e-10,
                    max_cycle=100):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if group_a is None:
            group_a = self._group_a
        mo_list, mo_energy, po_list, group = optimize_mo(
            self, mo_coeff, self.ncas, self.nelecas, self.ncore, group_a)
        return mo_list, mo_energy, po_list, group

    def get_svd_matrices(self, mo_list=None, po_list=None):
        if mo_list is None or po_list is None:
            mo_list, mo_energy, po_list, group = self.optimize_mo(self.mo_coeff)
        ncore = self.ncore
        s1e = self._scf.get_ovlp(self.mol)
        core_list = numpy.arange(ncore, dtype=int)
        N = mo_list.shape[1]
        p = len(po_list)
        dmet_core_list = numpy.zeros((p,p,N,N))
        ov_list = numpy.zeros((p,p))
        for i in range(0,p):
            xc_mo_coeff = mo_list[i][:,core_list]
            for j in range(0,p):
                wc_mo_coeff = mo_list[j][:,core_list]
                S, xc_bimo_coeff, wc_bimo_coeff, U, Vt = biorthogonalize(xc_mo_coeff, wc_mo_coeff, s1e)
                #ov_list[i,j] = numpy.prod(S[numpy.abs(S)>1e-10])*numpy.linalg.det(U)*numpy.linalg.det(Vt)
                ov_list[i,j] = numpy.prod(S[numpy.abs(S)>1e-10]**2)
                dmet_core_list[i,j] += xc_bimo_coeff @ numpy.diag(1/S) @ wc_bimo_coeff.T
        return dmet_core_list, ov_list

    def get_active_dm(self,mo_coeff = None):
        ncas = self.ncas
        ncore = self.ncore
        nocc = ncore + ncas
        if mo_coeff is None:
            ncore = self.ncore
            mo_coeff = self.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:,ncore:nocc]
        N = mo_coeff.shape[0]
        dmet_act_list = numpy.zeros((ncas,ncas,N,N))
        for i in range(0,ncas):
            for j in range(0,ncas):
                dmet_act_list[i,j] = numpy.outer(mo_coeff[:,i],mo_coeff[:,j])
        self.dmet_act_list = dmet_act_list
        return dmet_act_list

    def get_h1cas(self, dmet_act_list = None, mo_list = None, dmet_core_list = None, ncas = None, ncore = None):
        return self.get_h1e(dmet_act_list, mo_list, dmet_core_list, ncas, ncore)
    get_h1e = h1e_for_gbci

    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        return CASCI.get_h2eff(self,mo_coeff)

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        Returns:
          Three elements:
          total energy,
          active space CI energy,
          the active space CI wavefunction coefficients.

        They are attributes of the GBCI object, which can be accessed by
        .e_tot, .e_cas, .ci.
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        log = logger.new_logger(self, verbose)

        self.e_tot, self.e_cas, self.ci = \
              kernel(self, mo_coeff, ci0=ci0, verbose=log)

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = numpy.all(self.fcisolver.converged)
            if self.converged:
                log.info('GBCI converged')
            else:
                log.info('GBCI not converged')
        else:
            self.converged = True
        return self.e_tot, self.e_cas, self.ci

    def _prepare_noci_intermediates(self, mo_coeff, ncas, nelecas):
        mo_list, mo_energy, po_list, group = self.optimize_mo(mo_coeff)
        if getattr(self, '_group_a', None) is None:
            conf_info_list = group_info_list(ncas, nelecas, po_list)
            svd_basis = po_list
        else:
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
            svd_basis = group
        return mo_list, svd_basis, conf_info_list


    def get_gbci_intermediates(self, mo_coeff=None, ncas=None, nelecas=None):
        """Build GBCI intermediates for downstream methods such as GBPDFT."""
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if mo_coeff is None : mo_coeff = self.mo_coeff

        cached = self._get_cached_gbci_intermediates(
            mo_coeff, ncas, nelecas, self.ncore)
        if cached is not None:
            return cached

        mo_list, mo_energy, po_list, group = self.optimize_mo(mo_coeff)
        if getattr(self, '_group_a', None) is None:
            conf_info_list = group_info_list(ncas, nelecas, po_list)
            svd_basis = po_list
        else:
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
            svd_basis = group
        dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)
        intermediates = {
            "mo_list": mo_list,
            "mo_energy": mo_energy,
            "po_list": po_list,
            "group": group,
            "svd_basis": svd_basis,
            "conf_info_list": conf_info_list,
            "dmet_core_list": dmet_core_list,
            "ov_list": ov_list,
        }
        self._cache_gbci_intermediates(
            mo_coeff, ncas, nelecas, self.ncore, intermediates)
        return intermediates


    def precompute_rdm1s(self, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None,
                ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None or dmet_core_list is None or ov_list is None:
            mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                mo_coeff, ncas, nelecas)
            if conf_info_list is None:
                conf_info_list = default_conf_info
            if dmet_core_list is None or ov_list is None:
                dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)

        return rdm.precompute_rdm1s_data(
            mo_coeff, ncas, nelecas, ncore, dmet_core_list,
            conf_info_list, ov_list)


    def make_rdm1s(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                 ncore=None, dmet_core_list=None, conf_info_list=None,
                 ov_list=None, data=None):
        if data is None:
            data = self.precompute_rdm1s(
                mo_coeff, ncas, nelecas, ncore, dmet_core_list,
                conf_info_list, ov_list)
        rdm1a, rdm1b = rdm.make_rdm1s_precomputed(
            ci, data, dmet_core_list=dmet_core_list, mo_coeff=mo_coeff)

        return rdm1a, rdm1b

    def make_rdm1(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None,
                ov_list=None, data=None):
        rdm1a, rdm1b = self.make_rdm1s(
            ci, mo_coeff, ncas, nelecas, ncore, dmet_core_list,
            conf_info_list, ov_list, data=data)
        return rdm1a + rdm1b

    def trans_rdm1s(self, ci_bra, ci_ket, mo_coeff=None, ncas=None,
                 nelecas=None, ncore=None, dmet_core_list=None,
                 conf_info_list=None, ov_list=None, data=None):
        if data is None:
            data = self.precompute_rdm1s(
                mo_coeff, ncas, nelecas, ncore, dmet_core_list,
                conf_info_list, ov_list)
        rdm1a, rdm1b = rdm.trans_rdm1s_precomputed(
            ci_bra, ci_ket, data, dmet_core_list=dmet_core_list,
            mo_coeff=mo_coeff)
        return rdm1a, rdm1b

    def trans_rdm1(self, ci_bra, ci_ket, mo_coeff=None, ncas=None,
                nelecas=None, ncore=None, dmet_core_list=None,
                conf_info_list=None, ov_list=None, data=None):
        rdm1a, rdm1b = self.trans_rdm1s(
            ci_bra, ci_ket, mo_coeff, ncas, nelecas, ncore,
            dmet_core_list, conf_info_list, ov_list, data=data)
        return rdm1a + rdm1b

    def make_rdm1s_mo(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None,
                ov_list=None, data=None):
        if data is None:
            data = self.precompute_rdm1s(
                mo_coeff, ncas, nelecas, ncore, dmet_core_list,
                conf_info_list, ov_list)
        if mo_coeff is None:
            mo_coeff = data["mo_coeff"]
        rdm_ao_a, rdm_ao_b = rdm.make_rdm1s_precomputed(
            ci, data, dmet_core_list=dmet_core_list, mo_coeff=mo_coeff)
        s1e = self._scf.get_ovlp(self.mol)
        rdm_a = mo_coeff.T @ s1e @ rdm_ao_a @ s1e @ mo_coeff
        rdm_b = mo_coeff.T @ s1e @ rdm_ao_b @ s1e @ mo_coeff

        return rdm_a, rdm_b

    def make_rdm1_mo(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None,
                ov_list=None, data=None):
        rdm_a, rdm_b = self.make_rdm1s_mo(
            ci, mo_coeff, ncas, nelecas, ncore, dmet_core_list,
            conf_info_list, ov_list, data=data)
        return rdm_a + rdm_b

    def trans_rdm1s_mo(self, ci_bra, ci_ket, mo_coeff=None, ncas=None,
                nelecas=None, ncore=None, dmet_core_list=None,
                conf_info_list=None, ov_list=None, data=None):
        if data is None:
            data = self.precompute_rdm1s(
                mo_coeff, ncas, nelecas, ncore, dmet_core_list,
                conf_info_list, ov_list)
        if mo_coeff is None:
            mo_coeff = data["mo_coeff"]
        rdm_ao_a, rdm_ao_b = rdm.trans_rdm1s_precomputed(
            ci_bra, ci_ket, data, dmet_core_list=dmet_core_list,
            mo_coeff=mo_coeff)
        s1e = self._scf.get_ovlp(self.mol)
        rdm_a = mo_coeff.conj().T @ s1e @ rdm_ao_a @ s1e @ mo_coeff
        rdm_b = mo_coeff.conj().T @ s1e @ rdm_ao_b @ s1e @ mo_coeff

        return rdm_a, rdm_b

    def trans_rdm1_mo(self, ci_bra, ci_ket, mo_coeff=None, ncas=None,
                nelecas=None, ncore=None, dmet_core_list=None,
                conf_info_list=None, ov_list=None, data=None):
        rdm_a, rdm_b = self.trans_rdm1s_mo(
            ci_bra, ci_ket, mo_coeff, ncas, nelecas, ncore,
            dmet_core_list, conf_info_list, ov_list, data=data)
        return rdm_a + rdm_b

    def make_rdm2s(self, ci, mo_coeff=None , ncas=None, nelecas=None,
                 ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None or dmet_core_list is None or ov_list is None:
            mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                mo_coeff, ncas, nelecas)
            if conf_info_list is None:
                conf_info_list = default_conf_info
            if dmet_core_list is None or ov_list is None:
                dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)

        rdm2aa, rdm2ab, rdm2ba, rdm2bb = rdm.make_rdm2s(mo_coeff, ci, ncas, nelecas,
                                         ncore, dmet_core_list, conf_info_list, ov_list)

        return rdm2aa, rdm2ab, rdm2ba, rdm2bb

    def make_rdm2(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None or dmet_core_list is None or ov_list is None:
            mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                mo_coeff, ncas, nelecas)
            if conf_info_list is None:
                conf_info_list = default_conf_info
            if dmet_core_list is None or ov_list is None:
                dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)

        rdm2 = rdm.make_rdm2(mo_coeff, ci, ncas, nelecas,
                              ncore, dmet_core_list, conf_info_list, ov_list)
        return rdm2

    def make_rdm2s_mo_slow(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None or dmet_core_list is None or ov_list is None:
            mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                mo_coeff, ncas, nelecas)
            if conf_info_list is None:
                conf_info_list = default_conf_info
            if dmet_core_list is None or ov_list is None:
                dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)

        rdm_ao_aa, rdm_ao_ab, rdm_ao_ba, rdm_ao_bb = rdm.make_rdm2s(mo_coeff, ci, ncas, nelecas,
                                     ncore, dmet_core_list, conf_info_list, ov_list)
        s1e = self._scf.get_ovlp(self.mol)
        A = mo_coeff.T @ s1e
        B = s1e @ mo_coeff
        rdm2aa = np.einsum('pa, qb, abcd, cr,ds->pqrs', A,A, rdm_ao_aa,B,B,optimize=True)
        rdm2ab = np.einsum('pa, qb, abcd, cr,ds->pqrs', A,A, rdm_ao_ab,B,B,optimize=True)
        rdm2ba = np.einsum('pa, qb, abcd, cr,ds->pqrs', A,A, rdm_ao_ba,B,B,optimize=True)
        rdm2bb = np.einsum('pa, qb, abcd, cr,ds->pqrs', A,A, rdm_ao_bb,B,B,optimize=True)

        return rdm2aa, rdm2ab, rdm2ba, rdm2bb

    def make_rdm2_mo_slow(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None or dmet_core_list is None or ov_list is None:
            mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                mo_coeff, ncas, nelecas)
            if conf_info_list is None:
                conf_info_list = default_conf_info
            if dmet_core_list is None or ov_list is None:
                dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)

        rdm2aa, rdm2ab, rdm2ba, rdm2bb = self.make_rdm2s_mo_slow(ci,mo_coeff, ncas, nelecas,
                                         ncore, dmet_core_list, conf_info_list, ov_list)

        return rdm2aa + rdm2ab + rdm2ba + rdm2bb

    def precompute_rdm2s_mo(self, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None or dmet_core_list is None or ov_list is None:
            mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                mo_coeff, ncas, nelecas)
            if conf_info_list is None:
                conf_info_list = default_conf_info
            if dmet_core_list is None or ov_list is None:
                dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)

        s1e = self._scf.get_ovlp(self.mol)
        precompute_data = rdm.precompute_rdm2s_mo_data(s1e, mo_coeff, ncas, nelecas,
                          ncore, dmet_core_list, conf_info_list, ov_list)

        return precompute_data

    def make_rdm2s_mo(self, ci, data=None, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if data is None:
            if ncas is None : ncas = self.ncas
            if nelecas is None : nelecas = self.nelecas
            if ncore is None : ncore = self.ncore
            if mo_coeff is None : mo_coeff = self.mo_coeff
            if conf_info_list is None or dmet_core_list is None or ov_list is None:
                mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                    mo_coeff, ncas, nelecas)
                if conf_info_list is None:
                    conf_info_list = default_conf_info
                if dmet_core_list is None or ov_list is None:
                    dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)
            data = self.precompute_rdm2s_mo(mo_coeff, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list)

        rdm2aa, rdm2ab, rdm2ba, rdm2bb = rdm.make_rdm2s_mo(ci, data)

        return rdm2aa, rdm2ab, rdm2ba, rdm2bb

    def make_rdm2_mo(self, ci, data=None, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if data is None:
            if ncas is None : ncas = self.ncas
            if nelecas is None : nelecas = self.nelecas
            if ncore is None : ncore = self.ncore
            if mo_coeff is None : mo_coeff = self.mo_coeff
            if conf_info_list is None or dmet_core_list is None or ov_list is None:
                mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                    mo_coeff, ncas, nelecas)
                if conf_info_list is None:
                    conf_info_list = default_conf_info
                if dmet_core_list is None or ov_list is None:
                    dmet_core_list, ov_list = self.get_svd_matrices(mo_list, svd_basis)
            data = self.precompute_rdm2s_mo(mo_coeff, ncas, nelecas, ncore, dmet_core_list, conf_info_list, ov_list)

        rdm2aa, rdm2ab, rdm2ba, rdm2bb = self.make_rdm2s_mo(ci, data)

        return rdm2aa + rdm2ab + rdm2ba + rdm2bb

    def get_core_density(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                         ncore=None, dmet_core_list=None,
                         conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None :
            mo_coeff = self.mo_coeff
        if conf_info_list is None or dmet_core_list is None or ov_list is None:
            mo_list, svd_basis, default_conf_info = self._prepare_noci_intermediates(
                mo_coeff, ncas, nelecas)
            if conf_info_list is None:
                conf_info_list = default_conf_info
            if dmet_core_list is None or ov_list is None:
                dmet_core_list, _ = self.get_svd_matrices(mo_list, svd_basis)

        core_density  = rdm.get_core_density(mo_coeff, ci, ncas, nelecas, ncore, dmet_core_list, conf_info_list)

        return core_density
