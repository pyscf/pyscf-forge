#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Ardavan Farahvash
# Author: Xing Zhang

import numpy as np
from functools import reduce

from pyscf import gto, lib, lo, scf, df
from pyscf.lib import logger
from pyscf.gto.mole import inter_distance
from pyscf import lno

def prune_lno_basis(mf, mo_coeff, orbloc, uocc_loc, eris, frozen, s1e=None,
                    fock=None, bp_thr=0.98):
    """
    Create Boughton-Pulay domain by pruning basis functions unnecessary for
    LNO active space.

    Parameters
    ----------
    bp_thr : Float, optional
        Boughton-Pulay threshold parameter. The default is 0.98.
    """

    if type(orbloc)==list:
        return(
            _prune_ulno(mf, mo_coeff, orbloc, uocc_loc, eris, frozen, s1e=None,
                            fock=None, bp_thr=bp_thr))
    else:
        return(
            _prune_rlno(mf, mo_coeff, orbloc, uocc_loc, eris, frozen, s1e=None,
                            fock=None, bp_thr=bp_thr))


def _prune_rlno(mf, mo_coeff, orbloc, uocc_loc, eris, frozen, s1e=None,
                fock=None, bp_thr=0.98):

    mol = mf.mol
    #
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    if fock is None:
        fock = mf.get_fock()

    #
    frozen, maskact = lno.lnoccsd.get_maskact(frozen, mo_coeff.shape[1])
    orbact = mo_coeff[:, maskact]
    maskocc = mf.mo_occ > 1e-10
    N_occ_act = np.sum(maskocc & maskact)

    #
    domains = get_bp_domain(mol, orbact, bp_thr=bp_thr)
    atmlst = unique(domains)

    #
    fake_mol = fake_mol_by_atom(mol, atmlst)
    fake_mol.verbose=0
    _df = df.DF(fake_mol, auxbasis=mf.auxbasis)
    _df.build()
    fake_mol.verbose=mol.verbose

    #
    ao_idx = ao_index_by_atom(mol, atmlst)
    s21 = s1e[ao_idx]
    s22 = s1e[np.ix_(ao_idx, ao_idx)]
    fock22 = fock[np.ix_(ao_idx, ao_idx)]

    # project LMOs
    orbloc_new = []
    for i in range(orbloc.shape[1]):
        lmo_i = orbloc[:,i].reshape(-1,1)
        lmo_i_prj = project_mo(lmo_i, s21, s22)
        lmo_i_prj = lo.orth.vec_lowdin(lmo_i_prj, s=s22)
        orbloc_new.append(lmo_i_prj)
    orbloc_new = np.hstack(orbloc_new)

    s_lmo = orbloc_new.conj().T @ s21 @ orbloc
    uocc_loc_prj = uocc_loc@s_lmo


    # project LNOs
    lno_new = []
    for i in range(orbact.shape[1]):
        lno_i = orbact[:,i].reshape(-1,1)
        lno_i_prj = project_mo(lno_i, s21, s22)
        lno_i_prj = lo.orth.vec_lowdin(lno_i_prj, s=s22)
        lno_new.append(lno_i_prj)
    lno_new = np.hstack(lno_new)

    moE_new, moC_new = lno.lno.subspace_eigh(fock22, lno_new)

    _mo_occ = np.zeros((moE_new.size), dtype=np.int32)
    _mo_occ[0:N_occ_act] = 2

    #
    fake_mf = mf.copy()
    fake_mf.mol = fake_mol
    fake_mf.with_df = _df
    fake_mf.converged = True

    fake_mf.mo_coeff = moC_new
    fake_mf.mo_energy = moE_new
    fake_mf.mo_occ = _mo_occ

    return fake_mf, lno_new, uocc_loc_prj, eris, []

def _prune_ulno(mf, mo_coeff, orbloc, uocc_loc, eris, frozen, s1e=None,
                fock=None, bp_thr=0.98):

    mol = mf.mol
    #
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    if fock is None:
        fock = mf.get_fock()

    #
    _, maskact_a = lno.lnoccsd.get_maskact(frozen[0], mo_coeff[0].shape[1])
    _, maskact_b = lno.lnoccsd.get_maskact(frozen[1], mo_coeff[1].shape[1])

    orbact_a = mo_coeff[0][:, maskact_a]
    orbact_b = mo_coeff[1][:, maskact_b]
    orbact = [orbact_a,orbact_b]

    #
    domains = get_bp_domain(mol, np.hstack([orbact_a,orbact_b]), bp_thr=bp_thr)
    atmlst = unique(domains)

    #
    fake_mol = fake_mol_by_atom(mol, atmlst)
    fake_mol.verbose=0
    _df = df.DF(fake_mol, auxbasis=mf.auxbasis)
    _df.build()
    fake_mol.verbose=mol.verbose



    # basis projected matrices
    ao_idx = ao_index_by_atom(mol, atmlst)
    s21 = s1e[ao_idx]
    s22 = s1e[np.ix_(ao_idx, ao_idx)]

    fock22_a = fock[0][np.ix_(ao_idx, ao_idx)]
    fock22_b = fock[1][np.ix_(ao_idx, ao_idx)]

    # project LMOs
    orbloc_new = [[],[]]
    for n in range(2):
        if orbloc[n].shape[1] > 0:
            for i in range(orbloc[n].shape[1]):
                lmo_i = orbloc[n][:,i].reshape(-1,1)
                lmo_i_prj = project_mo(lmo_i, s21, s22)
                lmo_i_prj = lo.orth.vec_lowdin(lmo_i_prj, s=s22)
                orbloc_new[n].append(lmo_i_prj)
            orbloc_new[n] = np.hstack(orbloc_new[n])
        else:
            orbloc_new[n] = orbloc[n][ao_idx,:]

    s_lmo_a = orbloc_new[0].conj().T @ s21 @ orbloc[0]
    s_lmo_b = orbloc_new[1].conj().T @ s21 @ orbloc[1]

    uocc_loc_prj = [None,] * 2
    uocc_loc_prj[0] = uocc_loc[0]@s_lmo_a
    uocc_loc_prj[1] = uocc_loc[1]@s_lmo_b

    # project LNOs
    Nact = [maskact_a.sum(), maskact_b.sum()]
    Nact_max = max(Nact)

    # pad lnos so that both alpha/beta have the same number of orbitals
    lno_new = np.zeros((2,len(ao_idx),Nact_max))
    frozen = [[],[]]
    for n in range(2):
        for i in range(Nact_max):
            if i < Nact[n]:
                lno_i = orbact[n][:,i].reshape(-1,1)
                lno_i_prj = project_mo(lno_i, s21, s22)
                lno_i_prj = lo.orth.vec_lowdin(lno_i_prj, s=s22)
                lno_new[n,:,i] = lno_i_prj[:,0]
            else:
                frozen[n].append(i)

    moE_new_a, moC_new_a = lno.lno.subspace_eigh(fock22_a, lno_new[0])
    moE_new_b, moC_new_b = lno.lno.subspace_eigh(fock22_b, lno_new[1])

    # freeze extra orbitals and ensure they are virtual
    indxzero_a = np.argsort(np.abs(moE_new_a))[0:len(frozen[0])]
    indxzero_b = np.argsort(np.abs(moE_new_b))[0:len(frozen[1])]
    Nocc_act_a = np.sum( (mf.mo_occ[0] > 1e-10) & maskact_a)
    Nocc_act_b = np.sum( (mf.mo_occ[1] > 1e-10) & maskact_b)
    _mo_occ = np.zeros((2,Nact_max), dtype=np.int32)

    k = 0
    for i in range(Nact_max):
        if k >= Nocc_act_a:
            break
        if i not in indxzero_a:
            _mo_occ[0,i] = 1
            k+=1

    k = 0
    for i in range(Nact_max):
        if k >= Nocc_act_b:
            break
        if i not in indxzero_b:
            _mo_occ[1,i] = 1
            k+=1

    # construct new mf object
    fake_mf = mf.copy()
    fake_mf.mol = fake_mol
    fake_mf.with_df = _df
    fake_mf.converged = True

    fake_mf.mo_coeff = np.array([moC_new_a,moC_new_b])
    fake_mf.mo_energy = np.array([moE_new_a,moE_new_b])
    fake_mf.mo_occ = _mo_occ

    return fake_mf, lno_new, uocc_loc_prj, eris, frozen

def get_bp_domain(mol, mos, s1e=None, bp_thr=0.98,
                  q_thr=0.8, q_type='lowdin', atmlst=None):
    """
    BP domains based on partial Mulliken/Lowdin charges.

    Domains are constructed based a total bp_threshold of 0.98
    (98% of the each MO must be contained within the corresponding domain)

    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    if q_thr is None:
        q_thr = 0.8

    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1,1)

    assert mos.ndim == 2
    nao, nmo = mos.shape

    assert nao == mol.nao

    rr = atom_distance(mol, atmlst)
    aoslices = mol.aoslice_by_atom()[:,2:]
    bp_atmlst = []

    for i in range(nmo):

        # compute atomic charges
        orbi = mos[:,i]
        if q_type=='mulliken':
            GOP = orbi * np.dot(s1e, orbi)


        elif q_type=='lowdin':
            e, v = np.linalg.eigh(s1e)
            s_half = v @ np.diag(np.sqrt(np.abs(e))) @ v.T
            GOP = np.dot(s_half, orbi)**2

        q = np.asarray([GOP[aoslices[a,0]:aoslices[a,1]].sum() for a in atmlst])

        sorted_indices = q.argsort()[::-1]
        q_sorted = q[sorted_indices]
        cumsum = np.cumsum(q_sorted)
        icut = np.where(cumsum >= q_thr)[0][0] if len(np.where(cumsum >= q_thr)[0]) > 0 else len(sorted_indices)
        indx = sorted_indices[0:icut + 1]
        _atms = atmlst[indx]


        if len(_atms) > 0:
            domain_pop = _compute_bpvalue(mol, orbi, s1e, _atms)
        else:
            domain_pop = 0

        # if the domain does not cover bp_thr of the MO, loop over nearby atoms
        # and add them to the domain as necessary
        if domain_pop < bp_thr:
            center_id = np.argsort(-q)[0]
            _sorted_atm_idx = np.argsort(rr[center_id])

            for iatm in _sorted_atm_idx[0:]:
                a = atmlst[iatm]
                if a not in _atms:
                    _atms = np.append(_atms, a)
                    domain_pop = _compute_bpvalue(mol, orbi, s1e, _atms)
                    if domain_pop >= bp_thr:
                        break

        bp_atmlst.append(_atms)

    return bp_atmlst

def _compute_bpvalue(mol, mo, s1e=None, atmlst=None):
    """
    Compute Boughton-Pulay Value, the total completeness of the MO over the BP
    domain.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mo = np.asarray(mo)

    ao_idx = ao_index_by_atom(mol, atmlst)
    v = s1e[ao_idx] @ mo
    a = np.linalg.solve(s1e[np.ix_(ao_idx, ao_idx)], v)
    av = np.sum(a * v, axis=0)
    return av

def ao_index_by_atom(mol, atmlst):
    aoslices = mol.aoslice_by_atom()[:,2:]
    ao_idx_lst = map(lambda x: np.arange(*x), aoslices[atmlst].reshape(-1,2))
    ao_idx = reduce(np.union1d, ao_idx_lst)
    return ao_idx

def atom_distance(mol, atmlst=None):
    """Atomic distance array
    """
    if atmlst is None:
        atmlst = np.arange(mol.natm)
    coords = mol.atom_coords()[atmlst].reshape(-1,3)
    return inter_distance(mol, coords=coords)

def fake_mol_by_atom(mol, atmlst=None):
    if atmlst is not None:
        fake_mol = mol.copy(deep=False)
        fake_mol._atom = [mol._atom[a] for a in atmlst]
        fake_mol._atm, fake_mol._bas, fake_mol._env = \
            fake_mol.make_env(fake_mol._atom, fake_mol._basis,
                              mol._env[:gto.PTR_ENV_START])
        fake_mol._built = True
    else:
        fake_mol = mol
    return fake_mol

def unique(a):
    unique_items = list({item for sublist in a for item in sublist})
    return unique_items

def project_mo(mo1, s21, s22):
    return lib.cho_solve(s22, s21 @ mo1, strict_sym_pos=False)


#%%
if __name__ == '__main__':
    from pyscf import mp, cc
    from pyscf.cc.ccsd_t import kernel as CCSD_T
    from pyscf.data.elements import chemcore

    # water 4-mer
    atom = '''
    O   -1.485163346097   -0.114724564047    0.000000000000
    H   -1.868415346097    0.762298435953    0.000000000000
    H   -0.533833346097    0.040507435953    0.000000000000
    O    1.416468653903    0.111264435953    0.000000000000
    H    1.746241653903   -0.373945564047   -0.758561000000
    H    1.746241653903   -0.373945564047    0.758561000000
    O    4.485163346097   -0.114724564047    0.000000000000
    H    4.868415346097    0.762298435953    0.000000000000
    H    5.533833346097    0.040507435953    0.000000000000
    O    6.416468653903    0.111264435953    0.000000000000
    H    6.746241653903   -0.373945564047   -0.758561000000
    H    6.746241653903   -0.373945564047    0.758561000000
    '''
    basis = 'cc-pvdz'

    mol = gto.M(atom=atom, basis=basis)
    mol.verbose = 4
    frozen = chemcore(mol)

    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
    mlo = lo.PipekMezey(mol, orbocc)
    lo_coeff = mlo.kernel()
    frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

    # LNO-CCSD(T) calculation: here we scan over a list of thresholds
    mcc = lno.LNOCCSD(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=4)
    mcc.prune_lno_basis=True

    threshs=np.asarray([0.95,0.98,0.99,1.0])
    elno_ccsd = np.zeros_like(threshs)
    for i,thresh in enumerate(threshs):
        mcc.lno_basis_thresh=thresh

        mcc.lno_thresh = [1e-4, 1e-5]
        mcc.kernel()
        elno_ccsd[i] = mcc.e_corr_ccsd

    print(elno_ccsd)


