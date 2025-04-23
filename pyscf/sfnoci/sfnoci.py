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
# Author: Jiseong Park <fark4308@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

'''
Spin Flip Non-Orthogonal Configuration Interaction (SF-NOCI)
and Grouped-Bath Ansatz for SF-NOCI (SF-GNOCI)

References:
[1] Spin-flip non-orthogonal configuration interaction: a variational and
    almost black-box method for describing strongly correlated molecules
    Nicholas J. Mayhall, Paul R. Horn, Eric J. Sundstrom and Martin Head-Gordon
    Phys. Chem. Chem. Phys. 2014, 16, 22694
[2] Efficient grouped-bath ansatz for spin-flip non-orthogonal configuration
    interaction (SF-GNOCI) in transition-metal charge-transfer complexes
    Jiseong Park and Seunghoon Lee
    J. Chem. Theory Comput. 2025
'''

import numpy
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import rohf
from pyscf import __config__
from itertools import product
from pyscf.mcscf.casci import CASBase, CASCI
from pyscf.fci import cistring
from pyscf.sfnoci.direct_sfnoci import SFNOCISolver

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')
MO_BASE = getattr(__config__, 'MO_BASE', 1)
TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)

def kernel(sfnoci, mo_coeff=None, ci0=None, verbose=logger.NOTE):
    '''SFNOCI solver

    Args:
        sfnoci: SFNOCI object

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
    if mo_coeff is None: mo_coeff = sfnoci.mo_coeff
    if ci0 is None: ci0 = sfnoci.ci

    log = logger.new_logger(sfnoci, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    if hasattr(sfnoci, '_groupA'): log.debug('Start SFGNOCI')
    else: log.debug('Start SFNOCI')


    ncas = sfnoci.ncas
    nelecas = sfnoci.nelecas

    # FASSCF
    mo_list, po_list, group = sfnoci.optimize_mo(mo_coeff)
    conf_info_list = group_info_list(ncas, nelecas, po_list, group)
    t1 = log.timer('FASSCF', *t0)

    # SVD and core density matrix
    if group is None:
        dmet_core_list, ov_list = sfnoci.get_svd_matrices(mo_list, po_list)
    else:
        dmet_core_list, ov_list = sfnoci.get_svd_matrices(mo_list, group)
    t1 = log.timer('SVD and core density matrix', *t1)

    # 1e
    dmet_act_list = sfnoci.get_active_dm(mo_coeff)
    h1e, ecore_list = sfnoci.get_h1cas(dmet_act_list , mo_list , dmet_core_list)
    t1 = log.timer('effective 1e hamiltonians and core energies', *t1)

    # 2e
    eri = sfnoci.get_h2eff(mo_coeff)
    t1 = log.timer('effective 2e hamiltonian', *t1)

    # FCI
    max_memory = max(400, sfnoci.max_memory-lib.current_memory()[0])
    e_tot, fcivec = sfnoci.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                            conf_info_list, ov_list, ecore_list,
                                            ci0=ci0, verbose=log,
                                            max_memory=max_memory)

    log.timer('SFNOCI solver', *t1)
    log.timer('All SFNOCI process', *t0)

    if isinstance(e_tot, (float, numpy.float64)):
        e_cas = e_tot - ecore_list
    else:
        e_cas = [e - ecore_list for e in e_tot]

    return e_tot, e_cas, fcivec

def possible_occ(n_as,n_ase):
    def find_arrays(n_as,n_ase):
        possible_values=[0,1,2]
        arrays=[]

        for combination in product(possible_values,repeat=n_as):
            if sum(combination)==n_ase:
                arrays.append(numpy.array(combination))
        return arrays
    result_arrays=find_arrays(n_as,n_ase)
    concantenated_array=numpy.array(result_arrays, order = 'C', dtype = numpy.int32)
    return concantenated_array

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

def grouping_by_occ(po_list, groupA):
    a = len(groupA)
    p = len(po_list)
    #n = len(po_list[0])
    A_occ = numpy.zeros((p,a))
    for index, occ in enumerate(po_list):
        for i in range(a):
            A_occ[index][i] = numpy.sum(occ[groupA[i]])
    grouped_rows = {}
    for i, row in enumerate(A_occ):
        row_tuple = tuple(row)
        if row_tuple not in grouped_rows:
            grouped_rows[row_tuple]=[]
        grouped_rows[row_tuple].append(i)
    return list(grouped_rows.values())

def grouping_by_lowdin(mol, ac_mo_coeff,po_list, aolabel, thres = 0.2):
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    e,v = numpy.linalg.eigh(ova)
    s12 = numpy.dot(v *numpy.sqrt(e), v.T.conj())

    aolist = mol.search_ao_label(aolabel)
    print(aolist)
    #a = len(aolist)
    p = len(po_list)
    #n = len(po_list[0])
    #N = ac_mo_coeff.shape[0]
    ao_elecnums = numpy.zeros(p)
    for i in range(p):
        one_list = numpy.where(po_list[i] == 1)[0]
        two_list = numpy.where(po_list[i] == 2)[0]
        pT1 = numpy.dot(ac_mo_coeff[:,one_list],ac_mo_coeff[:,one_list].T)
        pT2 = numpy.dot(ac_mo_coeff[:,two_list],ac_mo_coeff[:,two_list].T)
        pT = pT1 + pT2 * 2
        pTOAO = reduce(numpy.dot,(s12,pT,s12))
        for index, j in enumerate(aolist):
            ao_elecnums[i] += pTOAO[j,j]

    print(ao_elecnums)
    groups = []
    visited = set()

    for i in range(len(ao_elecnums)):
        if i in visited:
            continue

        current_group = [i]
        visited.add(i)

        for j in range(len(ao_elecnums)):
            if i != j and j not in visited and abs(ao_elecnums[i] - ao_elecnums[j]) <= thres:
                current_group.append(j)
                visited.add(j)

        groups.append(current_group)

    print(groups)
    return groups

def mo_overlap(mo1, mo2, s1e):
    mo_overlap_list = lib.einsum('ai,bj,ij->ab', numpy.conjugate(mo1.T), mo2.T, s1e)
    return mo_overlap_list

def biorthogonalize(mo1, mo2, s1e):
    u, s, vt = numpy.linalg.svd(mo_overlap(mo1, mo2, s1e))
    mo1_bimo_coeff = mo1.dot(u)
    mo2_bimo_coeff = mo2.dot(vt.T)
    return s, mo1_bimo_coeff, mo2_bimo_coeff, u, vt

def find_matching_rows(matrix, target_row):
    matching_rows = numpy.where((matrix == target_row).all(axis=1))[0]
    return matching_rows

def str2occ(str0,norb):
    occ=numpy.zeros(norb)
    for i in range(norb):
        if str0 & ( 1 << i ):
            occ[i]=1

    return occ

def num_to_group(groups,number):
    for i, group in enumerate(groups):
        if number in group:
            return i
    return None

def group_info_list(ncas, nelecas, PO, group = None):
    stringsa = cistring.make_strings(range(0,ncas), nelecas[0])
    stringsb = cistring.make_strings(range(0,ncas), nelecas[1])
    na = len(stringsa)
    nb = len(stringsb)
    group_info = numpy.zeros((na,nb))
    for stra, strsa in enumerate(stringsa):
        for strb, strsb in enumerate(stringsb):
            occa = str2occ(stringsa[stra],ncas)
            occb = str2occ(stringsb[strb],ncas)
            occ = occa + occb
            p = find_matching_rows(PO,occ)[0]
            if group is not None:
                p = num_to_group(group,p)
            group_info[stra, strb] = p
    return group_info.astype(int)

def FASSCF(mf,as_list,core_list,highspin_mo_energy,highspin_mo_coeff,
           as_occ,conv_tol=1e-10, conv_tol_grad=None, max_cycle = 100,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    mf.max_cycle = max_cycle
    n_as=len(as_list)
    N=highspin_mo_coeff.shape[1]
    cn=len(core_list)
    vir_list=numpy.array(range(numpy.max(as_list)+1,N))
    highspin_mo_occ=numpy.zeros(N)
    for i in core_list:
        highspin_mo_occ[i]=2
    for idx, value in zip(as_list,as_occ):
        highspin_mo_occ[idx]=value
    for i in vir_list:
        highspin_mo_occ[i]=0

    if'init_dm' in kwargs:
        raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.11.
Keyword argument "init_dm" is replaced by "dm0"''')
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    if dm0 is None:
        dm = mf.make_rdm1(highspin_mo_coeff,highspin_mo_occ)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    s1e = mf.get_ovlp(mol)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond)*1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    cput1 = logger.timer(mf, 'initialize scf', *cput0)

    mo_energy=highspin_mo_energy
    mo_coeff=highspin_mo_coeff
    mo_occ=highspin_mo_occ
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot


        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_basis_fock=(mo_coeff.T.dot(fock)).dot(mo_coeff)
        I=numpy.identity(N-n_as)
        reduced_mo_basis_fock=mo_basis_fock[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_fock.shape[0]),as_list),
                                                      ~numpy.isin(numpy.arange(mo_basis_fock.shape[1]),as_list))]
        new_mo_energy, mo_basis_new_mo_coeff=mf.eig(reduced_mo_basis_fock,I)
        reduced_mo_coeff=numpy.delete(mo_coeff,as_list,axis=1)
        new_mo_coeff=reduced_mo_coeff.dot(mo_basis_new_mo_coeff)



        for i in as_list:
            new_mo_coeff=numpy.insert(new_mo_coeff,i,highspin_mo_coeff[:,i],axis=1)
        mo_coeff=new_mo_coeff


        AS_fock_energy=lib.einsum('ai,aj,ij->a',numpy.conjugate(mo_coeff.T),mo_coeff.T,fock)
        for i in as_list:
            new_mo_energy=numpy.insert(new_mo_energy,i,AS_fock_energy[i])
        mo_energy=new_mo_energy
        dm = mf.make_rdm1(mo_coeff,mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.i
        fock_last=fock
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_ddm < numpy.sqrt(conv_tol):
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_basis_fock=(mo_coeff.T.dot(fock)).dot(mo_coeff)
        I=numpy.identity(N-n_as)
        reduced_mo_basis_fock=mo_basis_fock[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_fock.shape[0]),as_list),
                                                      ~numpy.isin(numpy.arange(mo_basis_fock.shape[1]),as_list))]

        new_mo_energy, mo_basis_new_mo_coeff=mf.eig(reduced_mo_basis_fock,I)
        reduced_mo_coeff=numpy.delete(mo_coeff,as_list,axis=1)
        new_mo_coeff=reduced_mo_coeff.dot(mo_basis_new_mo_coeff)



        for i in as_list:
            new_mo_coeff=numpy.insert(new_mo_coeff,i,mo_coeff[:,i],axis=1)

        mo_coeff=new_mo_coeff

        AS_fock_energy=lib.einsum('ai,aj,ij->a',numpy.conjugate(mo_coeff.T),mo_coeff.T,fock)
        for i in as_list:
            new_mo_energy=numpy.insert(new_mo_energy,i,AS_fock_energy[i])
        mo_energy=new_mo_energy
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm,dm_last,vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    if mf.disp is not None:
        e_disp = mf.get_dispersion()
        mf.scf_summary['dispersion'] = e_disp
        e_tot += e_disp

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    if not scf_conv:
        mo_coeff=highspin_mo_coeff

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

def StateAverage_FASSCF(sfnoci, target_group, po_list, group, mo_coeff = None,
                        ncas = None, nelecas = None, ncore = None, conv_tol=1e-10, conv_tol_grad=None, max_cycle = 100,
                        dump_chk=True, dm0=None, callback=None, conv_check=True, **kwarg):
    if mo_coeff is None : mo_coeff = sfnoci.mo_coeff
    if ncas is None: ncas = sfnoci.ncas
    if nelecas is None : nelecas = sfnoci.nelecas
    if ncore is None : ncore = sfnoci.ncore
    mf = sfnoci._scf
    assert isinstance(mf, rohf.ROHF), "The SCF class of SF-GNOCI must be ROHF class."

    cput0 = (logger.process_clock(), logger.perf_counter())
    stringsa = cistring.make_strings(range(ncas),nelecas[0])
    stringsb = cistring.make_strings(range(ncas),nelecas[1])
    #na = len(stringsa)
    nb = len(stringsb)
    N = mo_coeff.shape[0]
    as_list = numpy.array(range(ncore, ncore + ncas))
    asn = len(as_list)
    as_mo_coeff = mo_coeff[:,as_list]
    group_info = group_info_list(ncas, nelecas, po_list, group)
    group_info = group_info.reshape(-1)
    target_conf = numpy.where(group_info == target_group)[0]


    mol = mf.mol
    as_dm_a = numpy.zeros((N,N))
    as_dm_b = numpy.zeros((N,N))
    for conf in target_conf:
        stra = conf // nb
        strb = conf % nb
        mo_occa = str2occ(stringsa[stra], ncas)
        mo_occb = str2occ(stringsb[strb], ncas)
        mo_occ = (mo_occa, mo_occb)
        dm_a, dm_b = mf.make_rdm1(as_mo_coeff, mo_occ)
        as_dm_a += dm_a
        as_dm_b += dm_b
    as_dm_a = as_dm_a / len(target_conf)
    as_dm_b = as_dm_b / len(target_conf)
    if dm0 is None:
        core_mo_coeff = mo_coeff[:,:ncore]
        dm0_core = (core_mo_coeff ).dot(core_mo_coeff.conj().T)
        dm = numpy.asarray((dm0_core  + as_dm_a , dm0_core + as_dm_b))
    else:
        dm = dm0

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    s1e = mf.get_ovlp(mol)
    cput1 = logger.timer(mf, 'initialize scf', *cput0)

    for cycle in range(max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_basis_fock=(mo_coeff.T.dot(fock)).dot(mo_coeff)
        I=numpy.identity(N-asn)
        reduced_mo_basis_fock=mo_basis_fock[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_fock.shape[0]),as_list),
                                                      ~numpy.isin(numpy.arange(mo_basis_fock.shape[1]),as_list))]
        new_mo_energy, mo_basis_new_mo_coeff=mf.eig(reduced_mo_basis_fock,I)
        reduced_mo_coeff=numpy.delete(mo_coeff,as_list,axis=1)
        new_mo_coeff=reduced_mo_coeff.dot(mo_basis_new_mo_coeff)



        for i in as_list:
            new_mo_coeff=numpy.insert(new_mo_coeff,i, mo_coeff[:,i],axis=1)
        mo_coeff=new_mo_coeff


        as_fock_energy=lib.einsum('ai,aj,ij->a',numpy.conjugate(mo_coeff.T),mo_coeff.T,fock)
        for i in as_list:
            new_mo_energy=numpy.insert(new_mo_energy,i,as_fock_energy[i])
        mo_energy = new_mo_energy

        core_mo_coeff = mo_coeff[:,:ncore]
        dm_core = (core_mo_coeff).dot(core_mo_coeff.conj().T)
        dm =numpy.asarray((dm_core + as_dm_a, dm_core + as_dm_b))
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf)

        #fock_last = fock
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_ddm = numpy.linalg.norm(dm-dm_last)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g |ddm|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_ddm)
        if abs(e_tot-last_hf_e) < conv_tol and norm_ddm < numpy.sqrt(conv_tol):
            scf_conv = True

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_basis_fock=(mo_coeff.T.dot(fock)).dot(mo_coeff)
        I=numpy.identity(N-asn)
        reduced_mo_basis_fock=mo_basis_fock[numpy.ix_(~numpy.isin(numpy.arange(mo_basis_fock.shape[0]),as_list),
                                                      ~numpy.isin(numpy.arange(mo_basis_fock.shape[1]),as_list))]

        new_mo_energy, mo_basis_new_mo_coeff=mf.eig(reduced_mo_basis_fock,I)
        reduced_mo_coeff=numpy.delete(mo_coeff,as_list,axis=1)
        new_mo_coeff=reduced_mo_coeff.dot(mo_basis_new_mo_coeff)



        for i in as_list:
            new_mo_coeff=numpy.insert(new_mo_coeff,i,mo_coeff[:,i],axis=1)

        mo_coeff=new_mo_coeff

        AS_fock_energy=lib.einsum('ai,aj,ij->a',numpy.conjugate(mo_coeff.T),mo_coeff.T,fock)
        for i in as_list:
            new_mo_energy=numpy.insert(new_mo_energy,i,AS_fock_energy[i])
        mo_energy=new_mo_energy
        dm_last = dm
        core_mo_coeff = mo_coeff[:,:ncore]
        dm_core = (core_mo_coeff * 2).dot(core_mo_coeff.conj().T)
        dm = (dm_core / 2 + as_dm_a, dm_core / 2 + as_dm_b)

        vhf = mf.get_veff(mol, dm,dm_last,vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_ddm = numpy.linalg.norm(dm-dm_last)

        if abs(e_tot-last_hf_e) < conv_tol or norm_ddm < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g |ddm|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_ddm)
    logger.timer(mf, 'scf_cycle', *cput0)

    if not scf_conv:
        mo_coeff = sfnoci.mo_coeff


    return scf_conv, e_tot, mo_energy, mo_coeff


def optimize_mo(sfnoci, mo_coeff = None, ncas = None, nelecas = None, ncore = None, groupA = None, debug = False):
    if mo_coeff is None : mo_coeff = sfnoci.mo_coeff
    if ncas is None : ncas = sfnoci.ncas
    if nelecas is None : nelecas = sfnoci.nelecas
    if ncore is None : ncore = sfnoci.ncore
    po_list = possible_occ(ncas, nelecas[0] + nelecas[1])
    N=mo_coeff.shape[0]
    p = len(po_list)
    group = None
    if not hasattr(sfnoci, '_groupA'):
        optimized_mo=numpy.zeros((p,N,N))
        #SF-CAS
        if debug:
            for i, occ in enumerate(po_list):
                optimized_mo[i]=mo_coeff
        #SF-NOCI
        else:
            for i, occ in enumerate(po_list):
                conv, et, moe, moce, moocc = sfnoci.FASSCF(occ, mo_coeff, ncas, ncore,
                                                           conv_tol= sfnoci._scf.conv_tol,
                                                           max_cycle= sfnoci._scf.max_cycle)
                print(conv, et)
                optimized_mo[i]=moce
                print("occuped pattern index:")
                print(i)

    else:
        groupA = sfnoci._groupA
        if isinstance(groupA, str):
            group = grouping_by_lowdin(sfnoci.mol,mo_coeff[:,ncore:ncore+ncas],po_list, groupA, thres= sfnoci._thres)
        elif isinstance(groupA, list):
            group = grouping_by_occ(po_list,groupA)
        else: NotImplementedError
        g = len(group)
        optimized_mo = numpy.zeros((g,N,N))
        for i in range(0,g):
            #SF-CAS
            if debug:
                optimized_mo[i] = mo_coeff
            #SF-GNOCI
            else:
                conv, et, moe, moce = StateAverage_FASSCF(sfnoci, i, po_list, group, mo_coeff,
                                                          ncas, nelecas, ncore,
                                                          conv_tol= sfnoci._scf.conv_tol,
                                                          max_cycle= sfnoci._scf.max_cycle)
                print(conv, et)
                optimized_mo[i]=moce
                print("occuped pattern index:")
                print(i)

    return optimized_mo, po_list, group


def h1e_for_sfnoci(sfnoci, dmet_act_list=None, mo_list=None, dmet_core_list=None,
                   ncas=None, ncore=None):
    ''' SF-NOCI space one-electron hamiltonian

    Args:
        sfnoci : a SF-NOCI/SF-GNOCI object

    Returns:
        A tuple, A tuple, the first is the effective one-electron hamiltonian defined in SF-NOCI space,
        the second is the list of electronic energy from baths.
    '''
    if ncas is None : ncas = sfnoci.ncas
    if ncore is None : ncore = sfnoci.ncore
    if mo_list is None:
        mo_list, po_list, group = sfnoci.optimize_mo(sfnoci.mo_coeff)
    if dmet_core_list is None:
        dmet_core_list, ov_list = sfnoci.get_svd_matrices(mo_list, group)
    if dmet_act_list is None:
        dmet_act_list = sfnoci.get_active_dm(sfnoci.mo_coeff)
    p = dmet_core_list.shape[0]
    mo_cas = mo_list[0][:,ncore:ncore+ncas]
    hcore = sfnoci.get_hcore()
    h1e = numpy.zeros((p,p,ncas,ncas))
    ecore_list = numpy.zeros(p)
    energy_nuc = sfnoci.energy_nuc()
    ha1e = lib.einsum('ai,ab,bj->ij',mo_cas,hcore,mo_cas)

    for i in range(0,p):
        for j in range(0,p):
            corevhf = sfnoci.get_veff(dm = 2 * dmet_core_list[i,j])
            h1e[i,j] = ha1e + lib.einsum('ijab,ab -> ij', dmet_act_list , corevhf)
            if i==j:
                ecore_list[i] += lib.einsum('ab,ab -> ', dmet_core_list[i,i],corevhf)
                ecore_list[i] += energy_nuc
                ecore_list[i] += 2*lib.einsum('ab,ab->', dmet_core_list[i,i], hcore)
    sfnoci.h1e = h1e
    sfnoci.core_energies = ecore_list
    return h1e, ecore_list

def spin_square(sfnoci, rdm1, rdm2ab,rdm2ba):
    M_s = sfnoci.spin/2
    mo = sfnoci.mo_coeff
    s1e = sfnoci.mol.intor('int1e_ovlp')
    rdm1mo = lib.einsum('qi,pl,kj,qp,lk->ij', mo, rdm1, mo,s1e,s1e)
    rdm2mo = lib.einsum('ai,bj,ck,dl,ap,bq,cr,ds,pqrs',mo,mo,mo,mo,s1e,s1e,s1e,s1e,rdm2ab+rdm2ba)

    return M_s**2 + 0.5*lib.einsum('ii ->',rdm1mo) - 0.5*lib.einsum('ijji ->', rdm2mo)


class SFNOCI(CASBase):
    '''SF-NOCI

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
            The SFNOCISolver in pyscf.sfnoci.direct_sfnoci module must be used.
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
    def __init__(self, mf, ncas, nelecas, ncore=None):

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

        self.fcisolver = SFNOCISolver(mol)
        self.fcisolver.lindep = getattr(__config__,
                                      'sfnoci_SFNOCI_fcisolver_lindep', 1e-14)
        self.fcisolver.max_cycle = getattr(__config__,
                                         'sfnoci_SFNOCI_fcisolver_max_cycle', 100)
        self.fcisolver.conv_tol = getattr(__config__,
                                        'sfnoci_SFNOCI_fcisolver_conv_tol', 5e-7)

################################################## don't modify the following attributes, they are not input options
        self.e_tot = 0
        self.e_cas = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy
        self.mo_occ = None
        self.converged = False
        self._thres = 0.2

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

    def possible_occ(self):
        po_list = possible_occ(self.ncas, sum(self.nelecas))
        return po_list

    def FASSCF(self, occ, mo_coeff = None, ncas = None, ncore = None,
               conv_tol=1e-10, conv_tol_grad=None, max_cycle = 100):
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if ncas is None : ncas = self.ncas
        if ncore is None : ncore = self.ncore
        mf = self._scf
        as_list = numpy.array(range(ncore,ncore + ncas))
        core_list = numpy.array(range(0,ncore))
        FAS_scf_conv, FAS_e_tot, FAS_mo_energy, FAS_mo_coeff, FAS_mo_occ \
            = FASSCF(mf,as_list,core_list,mf.mo_energy,mo_coeff,occ, conv_tol= conv_tol , max_cycle= max_cycle)
        return FAS_scf_conv, FAS_e_tot, FAS_mo_energy, FAS_mo_coeff, FAS_mo_occ

    def optimize_mo(self, mo_coeff=None, debug=False, conv_tol=1e-10, max_cycle=100):
        if mo_coeff is None : mo_coeff = self.mo_coeff
        #mode = 0
        #if debug : mode = 1
        #mf = self._scf
        nelecas = self.nelecas
        ncore = self.ncore
        ncas = self.ncas
        mo_list, po_list, _ = optimize_mo(self, mo_coeff, ncas, nelecas, ncore, debug = debug)
        return mo_list, po_list, _

    def get_svd_matrices(self, mo_list=None, po_list_or_group=None):
        if mo_list is None or po_list_or_group is None:
            mo_list, po_list, _ = self.optimize_mo(self.mo_coeff)
            po_list_or_group = po_list
        ncore = self.ncore
        s1e = self._scf.get_ovlp(self.mol)
        core_list = numpy.array(range(0,ncore))
        N = mo_list.shape[1]
        p = len(po_list_or_group)
        dmet_core_list = numpy.zeros((p,p,N,N))
        ov_list = numpy.zeros((p,p))
        for i in range(0,p):
            xc_mo_coeff = mo_list[i][:,core_list]
            for j in range(0,p):
                wc_mo_coeff = mo_list[j][:,core_list]
                S, xc_bimo_coeff, wc_bimo_coeff, U, Vt = biorthogonalize(xc_mo_coeff, wc_mo_coeff, s1e)
                ov_list[i,j] = numpy.prod(S[numpy.abs(S)>1e-10])*numpy.linalg.det(U)*numpy.linalg.det(Vt)
                for c in range(0,ncore):
                    dmet_core_list[i,j] +=numpy.outer(xc_bimo_coeff[:,c],wc_bimo_coeff[:,c])/S[c]
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
    get_h1e = h1e_for_sfnoci

    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        return CASCI.get_h2eff(self,mo_coeff)

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        Returns:
          Five elements, they are
          total energy,
          active space CI energy,
          the active space FCI wavefunction coefficients,
          the MCSCF canonical orbital coefficients,
          the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        log = logger.new_logger(self, verbose)

        #self.check_sanity()
        #self.dump_flags(log)

        self.e_tot, self.e_cas, self.ci = \
              kernel(self, mo_coeff, ci0=ci0, verbose=log)

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = numpy.all(self.fcisolver.converged)
            if self.converged:
                log.info('SFNOCI converged')
            else:
                log.info('SFNOCI not converged')
        else:
            self.converged = True
        #self._finalize()
        return self.e_tot, self.e_cas, self.ci   # will provide group info


    def make_rdm1s(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                 ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, _ = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, po_list)

        rdm1a, rdm1b = \
            self.fcisolver.make_rdm1s(mo_coeff, ci, ncas, nelecas,
                                    ncore, dmet_core_list, conf_info_list, ov_list)
        return rdm1a, rdm1b

    def make_rdm1(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, _ = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, po_list)

        rdm = self.fcisolver.make_rdm1(mo_coeff, ci, ncas, nelecas,
                                     ncore, dmet_core_list, conf_info_list, ov_list)
        return rdm

    def make_rdm2s(self, ci, mo_coeff=None , ncas=None, nelecas=None,
                 ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, _ = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, po_list)

        rdm2aa, rdm2ab, rdm2ba, rdm2bb = \
            self.fcisolver.make_rdm2s(mo_coeff, ci, ncas, nelecas,
                                    ncore, dmet_core_list, conf_info_list, ov_list)
        return rdm2aa, rdm2ab, rdm2ba, rdm2bb

    def make_rdm2(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, _ = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, po_list)

        rdm = self.fcisolver.make_rdm2(mo_coeff, ci, ncas, nelecas,
                                     ncore, dmet_core_list, conf_info_list, ov_list)
        return rdm

class SFGNOCI(SFNOCI):
    '''SF-GNOCI
        groupA : str or list
            The critertion of grouping the configurations
            str : the name of atom to become the critertion of grouping
            list : Grouping the configurations by occupation number of molecular orbitals

            groupA = 'Li'

            or

            groupA = [[0,1],[2,3]]

        lowdin_thres : float
            The criterion of grouping the configurations if the lowdin basis is used.

        group : list
            The result of grouping.

    '''
    def __init__(self, mf, ncas, nelecas, ncore=None, groupA=None):
        super().__init__(mf, ncas, nelecas, ncore)
        self._groupA = groupA
        self._thres = 0.2

    @property
    def groupA(self):
        return self._groupA

    @groupA.setter
    def groupA(self,x):
        self._groupA = x

    @property
    def lowdin_thres(self):
        return self._thres

    @lowdin_thres.setter
    def lowdin_thres(self, x):
        self._thres = x

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        Returns:
          Five elements, they are
          total energy,
          active space CI energy,
          the active space FCI wavefunction coefficients,
          the MCSCF canonical orbital coefficients,
          the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        log = logger.new_logger(self, verbose)

        #self.check_sanity()
        #self.dump_flags(log)

        self.e_tot, self.e_cas, self.ci = \
              kernel(self, mo_coeff, ci0=ci0, verbose=log)

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = numpy.all(self.fcisolver.converged)
            if self.converged:
                log.info('SFGNOCI converged')
            else:
                log.info('SFGNOCI not converged')
        else:
            self.converged = True
        #self._finalize()
        return self.e_tot, self.e_cas, self.ci   # will provide group info

    def optimize_mo(self, mo_coeff=None, debug=False, groupA = None, conv_tol=1e-10, max_cycle=100):
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if groupA is None : groupA = self._groupA
        #mode = 0
        #if debug : mode = 1
        #mf = self._scf
        nelecas = self.nelecas
        ncore = self.ncore
        ncas = self.ncas
        mo_list, po_list, group = optimize_mo(self, mo_coeff, ncas, nelecas, ncore, groupA, debug)
        return mo_list, po_list, group

    def make_rdm1s(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                 ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, group = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, group)

        rdm1a, rdm1b = \
            self.fcisolver.make_rdm1s(mo_coeff, ci, ncas, nelecas,
                                    ncore, dmet_core_list, conf_info_list, ov_list)
        return rdm1a, rdm1b

    def make_rdm1(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, group = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, group)

        rdm = self.fcisolver.make_rdm1(mo_coeff, ci, ncas, nelecas,
                                     ncore, dmet_core_list, conf_info_list, ov_list)
        return rdm

    def make_rdm2s(self, ci, mo_coeff=None , ncas=None, nelecas=None,
                 ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, group = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, group)

        rdm2aa, rdm2ab, rdm2ba, rdm2bb = \
            self.fcisolver.make_rdm2s(mo_coeff, ci, ncas, nelecas,
                                    ncore, dmet_core_list,conf_info_list, ov_list)
        return rdm2aa, rdm2ab, rdm2ba, rdm2bb

    def make_rdm2(self, ci, mo_coeff=None, ncas=None, nelecas=None,
                ncore=None, dmet_core_list=None, conf_info_list=None, ov_list=None):
        if ncas is None : ncas = self.ncas
        if nelecas is None : nelecas = self.nelecas
        if ncore is None : ncore = self.ncore
        if mo_coeff is None : mo_coeff = self.mo_coeff
        if conf_info_list is None:
            mo_list, po_list, group = self.optimize_mo(mo_coeff)
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
        if dmet_core_list is None or ov_list is None:
            dmet_core_list, ov_list = self.get_svd_matrices(mo_list, group)

        rdm = self.fcisolver.make_rdm2(mo_coeff, ci, ncas, nelecas,
                                     ncore, dmet_core_list,conf_info_list, ov_list)
        return rdm

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [['Li', (0, 0, 0)],['F',(0,0,1.4)]]
    mol.basis = 'ccpvdz'

    x_list=[]
    e1_list=[]
    e2_list=[]
    e3_list=[]
    e4_list=[]
    e5_list=[]
    e6_list=[]
    e7_list=[]
    e8_list=[]
    ma=[]
    mode=0

    mol.spin=2
    mol.build(0,0)
    rm=scf.ROHF(mol)
    rm.kernel()

    molr = gto.Mole()
    molr.verbose = 5
    molr.output = None
    molr.atom = [['Li', (0, 0, 0)],['F',(0,0,1.3)]]
    molr.basis = 'ccpvdz'
    mr=scf.RHF(molr)
    mr.kernel()

    mo0=mr.mo_coeff
    occ=mr.mo_occ
    setocc=numpy.zeros((2,occ.size))
    setocc[:,occ==2]=1
    setocc[1][3]=0
    setocc[0][6]=1
    ro_occ=setocc[0][:]+setocc[1][:]
    dm_ro=rm.make_rdm1(mo0,ro_occ)
    rm=scf.addons.mom_occ(rm,mo0,setocc)
    rm.scf(dm_ro)
    mo=rm.mo_coeff
    as_list=[3,6,7,10]
    s1e = mol.intor('int1e_ovlp')
    mySFNOCI = SFNOCI(rm,4,(2,2))
    mySFNOCI.lowdin_thres= 0.5

    from pyscf.mcscf import addons
    mo = addons.sort_mo(mySFNOCI,rm.mo_coeff, as_list,1)
    reei, _, ci = mySFNOCI.kernel(mo)

    i=1
    while i <= 4:
        x_list.append(i)
        mol.atom=[['Li',(0,0,0)],['F',(0,0,i)]]
        mol.build(0,0)
        mol.spin=2
        m=scf.RHF(mol)
        m.kernel()
        m=scf.addons.mom_occ(m,mo0,setocc)
        m.scf(dm_ro)

        mySFNOCI = SFNOCI(m,4,(2,2))
        mySFNOCI.spin = 0
        mySFNOCI.fcisolver.nroots = 4
        mo = addons.sort_mo(mySFNOCI,m.mo_coeff,as_list,1)
        eigenvalues, _, eigenvectors = mySFNOCI.kernel(mo)

        e1_list.append(eigenvalues[0])
        e2_list.append(eigenvalues[1])
        e3_list.append(eigenvalues[2])
        e4_list.append(eigenvalues[3])

        i+=0.5

    print(e1_list)
    print(e2_list)
    print(e3_list)
    print(e4_list)

    import matplotlib.pyplot as plt
    from pyscf.data.nist import HARTREE2EV
    ref = e1_list[-1]
    e1_list = (numpy.array(e1_list) - ref) * HARTREE2EV
    e2_list = (numpy.array(e2_list) - ref) * HARTREE2EV
    e3_list = (numpy.array(e3_list) - ref) * HARTREE2EV
    e4_list = (numpy.array(e4_list) - ref) * HARTREE2EV

    plt.plot(x_list, e1_list, '-o', label='SF-NOCI $1{}^1\\Sigma^+$')
    plt.plot(x_list, e2_list, '-o', label='SF-NOCI $1{}^3\\Sigma^+$')
    plt.plot(x_list, e3_list, '-o', label='SF-NOCI $2{}^1\\Sigma^+$')
    plt.plot(x_list, e4_list, '-o', label='SF-NOCI $2{}^3\\Sigma^+$')

    #plt.xlabel("Li-F distance, $\\AA$")
    plt.xlabel("Li-F distance, A")
    plt.ylabel("Relative Energy, eV")
    plt.legend()
    plt.xlim(1, 4)
    plt.ylim(-4, 10)
    plt.show()
