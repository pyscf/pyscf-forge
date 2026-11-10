import numpy as np
from pyscf.lo import orth
from pyscf.lib import tag_array, logger

def _get_minmax_occs (las, norb_f, nelec_f, smults_f, mo_occ):
    norb = sum (norb_f)
    nfrag = len (norb_f)
    rngs = -np.ones ((nfrag, 3, 2), dtype=int)
    if mo_occ is None: return rngs
    assert (len (mo_occ) == norb)
    ndomo = np.count_nonzero (mo_occ==2)
    nsomo = np.count_nonzero (mo_occ==1)
    nuomo = norb - ndomo - nsomo
    assert (nuomo >= 0)
    nelec = 2*ndomo + nsomo
    if (smults_f is not None) and ((smults_f.sum () - nfrag) <= nsomo):
        spins_f = smults_f - 1
        rngs[:,1,0] = smults_f - 1
        rngs[:,1,1] = -(rngs[:,1,0] - rngs[:,1,0].sum ())
        assert ((rngs[:,1,:]>=0).all ())
    if (nelec_f is not None) and (nelec_f.sum () == nelec):
        subt = 0
        if (rngs[:,1,1]>=0).all ():
            subt = rngs[:,1,1]
        rngs[:,2,0] = (nelec_f - subt) // 2
        rngs[:,2,1] = -(rngs[:,2,0] - rngs[:,2,0].sum ()) 
    return rngs

def _localize (las, frags_orbs, mo_coeff, spin, lo_coeff, fock, ao_ovlp, freeze_cas_spaces=False,
               mo_occ=None, smults_f=None, nelec_f=None):
    ''' Project active orbitals into sets of orthonormal "fragments" defined by lo_coeff
    and frags_orbs, and orthonormalize inactive and virtual orbitals in the orthogonal complement
    space. Beware that unless freeze_cas_spaces=True, frozen orbitals will not be preserved.

    Args:
        las: LASSCF or LASCI object
        frags_orbs: list of length nfrags
            Contains list of AO indices formally defining the fragments
            into which the active orbitals are to be localized

    Kwargs: (some of these are args here but kwargs in the actual caller)
        mo_coeff: ndarray of shape (nao, nmo)
            Molecular orbital coefficients containing active orbitals
            on columns ncore:ncore+ncas
        spin: integer
            Unused; retained for backwards compatibility I guess
        lo_coeff: ndarray of shape (nao, nao)
            Linear combinations of AOs that are localized and orthonormal
        fock: ndarray of shape (nmo, nmo)
            Effective 1-electron Hamiltonian matrix for recanonicalizing
            the inactive and external sectors after the latter are
            possibly distorted by the projection of the active orbitals
        ao_ovlp: ndarray of shape (nao, nao)
            Overlap matrix of the underlying AO basis
        freeze_cas_spaces: logical
            If true, then active orbitals are mixed only among themselves
            when localizing, which leaves the inactive and external sectors
            unchanged (to within numerical precision). Otherwise, active
            orbitals are projected into the localized-orbital space and
            the inactive and external orbitals are reconstructed as closely
            as possible using SVD.
        mo_occ: ndarray of shape (nmo)
            If passed, only orbitals with the same occupancies are mixed with
            each other to localize them and freeze_cas_spaces is automatically
            set to True.
        smults_f: ndarray of ints of shape (nfrag,)
            Used to constrain how many singly-occupied orbitals (per mo_occ)
            are assigned to each fragment, if possible. If las is a single-state
            calculation, its quantum numbers are used as default values.
        nelec_f: ndarray of ints of shape (nfrag,)
            Used to constrain how many doubly-occupied orbitals (per mo_occ)
            are assigned to each fragment, if possible. If las is a single-state
            calculation, its quantum numbers are used as default values.

    Returns:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients after localization of the active space;
            columns in the order (inactive,las1,las2,...,lasn,external)
    '''
    # For reasons that pass my understanding, mo_coeff sometimes can't be assigned symmetry
    # by PySCF's own code. Therefore, I'm going to keep the symmetry tags on mo_coeff
    # and make sure the SVD engine sees them and doesn't try to figure it out itself.
    # Hopefully this never becomes a problem with the lo_coeff.
    log = logger.new_logger (las, las.verbose)
    ncore, ncas, ncas_sub = las.ncore, las.ncas, las.ncas_sub
    nocc = ncore + ncas
    nfrags = len (frags_orbs)
    nao, nmo = mo_coeff.shape
    unused_aos = np.ones (nao, dtype=np.bool_)
    for frag_orbs in frags_orbs: unused_aos[frag_orbs] = False
    has_orbsym = hasattr (mo_coeff, 'orbsym')
    mo_orbsym = getattr (mo_coeff, 'orbsym', np.zeros (nmo))
    mo_coeff = mo_coeff.copy () # Safety
    if mo_occ is not None:
        freeze_cas_spaces = True
        mo_occ = mo_occ.copy ()
        mocc_cas = mo_occ[ncore:nocc]
    else:
        mo_occ = np.zeros (nmo, dtype=int)
        mo_occ[:ncore] = 2
        mo_occ[ncore:nocc] = 1
        mocc_cas = None
    rngs = _get_minmax_occs (las, las.ncas_sub, nelec_f, smults_f, mocc_cas)

    # Duplicate AO handling
    dupeAOerr = ValueError (("Cannot assign 1 AO to more than 1 fragment unless active orbitals "
                             "are built from the MO guess (freeze_cas_spaces=True)"))
    if np.count_nonzero (~unused_aos) < len (np.concatenate (frags_orbs)):
        dupes, cnts = np.unique (np.concatenate (frags_orbs), return_counts=True)
        dupes = dupes[cnts>1]
        for dupe in dupes:
            dupestr = 'AO "{}" in multiple frags: '.format (las.mol.ao_labels ()[dupe])
            for ix, frag_orb in enumerate (frags_orbs):
                if dupe in frag_orb:
                    dupestr = dupestr + '{}, '.format (ix)
            log.warn (dupestr[:-2])
        if not freeze_cas_spaces:
            raise (dupeAOerr)

    # SVD to pick active orbitals
    mo_cas = tag_array (mo_coeff[:,ncore:nocc], orbsym=mo_orbsym[ncore:nocc])
    mocc_cas = mo_occ[ncore:nocc]
    if freeze_cas_spaces:
        null_coeff = np.hstack ([mo_coeff[:,:ncore], mo_coeff[:,nocc:]])
    else:
        null_coeff = lo_coeff[:,unused_aos]
    for ix, (nlas, frag_orbs) in enumerate (zip (las.ncas_sub, frags_orbs)):
        if nlas > len (frag_orbs):
            inadAOs = '{} active orbitals using only {} AOs for fragment {}'.format (
                nlas, len (frag_orbs), ix)
            if freeze_cas_spaces:
                log.warn ('Trying to localize ' + inadAOs)
            else:
                raise ValueError ("Cannot make " + inadAOs)
        mo_proj, sval, mo_cas, mocc_cas = las._svd (lo_coeff[:,frag_orbs], mo_cas, s=ao_ovlp,
                                                    mo_occ=mocc_cas, rngs=rngs[ix])
        if len (sval) < nlas:
            log.error ("Too many active or too few fragment orbitals in fragment %d", ix)
            raise RuntimeError ("Can't localize active space")
        i, j = ncore + sum (las.ncas_sub[:ix]), ncore + sum (las.ncas_sub[:ix]) + nlas
        mo_las = mo_cas if freeze_cas_spaces else mo_proj
        mo_coeff[:,i:j] = mo_las[:,:nlas]
        if has_orbsym: mo_orbsym[i:j] = mo_las.orbsym[:nlas]
        if freeze_cas_spaces:
            mo_occ[i:nocc] = mocc_cas
            mocc_cas = mo_occ[j:nocc]
            if has_orbsym: orbsym = mo_cas.orbsym[nlas:]
            mo_cas = mo_cas[:,nlas:]
            if has_orbsym: mo_cas = tag_array (mo_cas, orbsym=orbsym)
        else:
            null_coeff = np.hstack ([null_coeff, mo_proj[:,nlas:]])

    # SVD of null space to pick inactive orbitals
    if ncore > 0:
        assert (null_coeff.shape[-1] + ncas == nmo)
        mo_core = tag_array (mo_coeff[:,:ncore], orbsym=mo_orbsym[:ncore])
        mocc_core = mo_occ[:ncore]
        mo_proj, sval, mo_core, mocc_core = las._svd (null_coeff, mo_core, s=ao_ovlp,
                                                      mo_occ=mocc_core)
        mo_coeff[:,:ncore], mo_coeff[:,nocc:] = mo_proj[:,:ncore], mo_proj[:,ncore:]
        if has_orbsym:
            mo_orbsym[:ncore] = mo_proj.orbsym[:ncore]
            mo_orbsym[nocc:] = mo_proj.orbsym[ncore:]
    elif nocc < mo_coeff.shape[1]:
        mo_virt = tag_array (mo_coeff[:,nocc:], orbsym=mo_orbsym[nocc:])
        mocc_virt = mo_occ[nocc:]
        mo_proj, sval, mo_virt, mocc_virt = las._svd (null_coeff, mo_virt, s=ao_ovlp,
                                                      mo_occ=mocc_virt)
        mo_coeff[:,nocc:] = mo_proj[:,:]
        if has_orbsym:
            mo_orbsym[nocc:] = mo_proj.orbsym[:]

    mo_coeff = tag_array (mo_coeff, orbsym=mo_orbsym)

    # Canonicalize for good init CI guess and visualization
    ranges = [(0,ncore),(nocc,nmo)]
    for ix, di in enumerate (ncas_sub):
        i = ncore + sum (ncas_sub[:ix])
        j = i + di
        idx = np.argsort (-mo_occ[i:j])
        mo_coeff[:,i:j] = mo_coeff[:,i:j][:,idx]
        mo_orbsym[i:j] = mo_orbsym[i:j][idx]
        mo_occ[i:j] = mo_occ[i:j][idx]
        for m in np.unique (mo_occ[i:j]):
            idx = np.where (mo_occ[i:j]==m)[0]
            k = idx[0]
            l = idx[-1]+1
            ranges.append ((i+k,i+l))
    fock = mo_coeff.conj ().T @ fock @ mo_coeff
    for i, j in ranges:
        if (j == i): continue
        e, c = las._eig (fock[i:j,i:j], i, j, mo_orbsym[i:j])
        idx = np.argsort (e)
        mo_coeff[:,i:j] = mo_coeff[:,i:j] @ c[:,idx]
        mo_orbsym[i:j] = mo_orbsym[i:j][idx]
    if has_orbsym: mo_coeff = tag_array (mo_coeff, orbsym=mo_orbsym)
    else: mo_coeff = np.array (mo_coeff) # remove spurious tag
    return mo_coeff

def localize_init_guess (las, frags_atoms, mo_coeff=None, spin=None, lo_coeff=None, fock=None,
                         mo_occ=None, freeze_cas_spaces=False, frags_by_AOs=False, smults_f=None,
                         nelec_f=None):
    ''' Project active orbitals into sets of orthonormal "fragments" defined by lo_coeff
    and frags_orbs, and orthonormalize inactive and virtual orbitals in the orthogonal complement
    space. Beware that unless freeze_cas_spaces=True, frozen orbitals will not be preserved.

    Args:
        frags_atoms: list of length nfrags
            Contains either lists of integer atom indices, or lists of
            strings which are passed to mol.search_ao_label, which define
            fragments into which the active orbitals are to be localized

    Kwargs:
        mo_coeff: ndarray of shape (nao, nmo)
            Molecular orbital coefficients containing active orbitals
            on columns ncore:ncore+ncas
        spin: integer
            Unused; retained for backwards compatibility I guess
        lo_coeff: ndarray of shape (nao, nao)
            Linear combinations of AOs that are localized and orthonormal
        fock: ndarray of shape (nmo, nmo)
            Effective 1-electron Hamiltonian matrix for recanonicalizing
            the inactive and external sectors after the latter are
            possibly distorted by the projection of the active orbitals
        mo_occ: ndarray of shape (nmo)
            If passed, only orbitals with the same occupancies are mixed with
            each other to localize them and freeze_cas_spaces is automatically
            set to True.
        freeze_cas_spaces: logical
            If true, then active orbitals are mixed only among themselves
            when localizing, which leaves the inactive and external sectors
            unchanged (to within numerical precision). Otherwise, active
            orbitals are projected into the localized-orbital space and
            the inactive and external orbitals are reconstructed as closely
            as possible using SVD.
        frags_by_AOs: logical
            If True, interpret integer frags_atoms as AOs rather than atoms
        smults_f: ndarray of ints of shape (nfrag,)
            Used to constrain how many singly-occupied orbitals (per mo_occ)
            are assigned to each fragment, if possible. If las is a single-state
            calculation, its quantum numbers are used as default values.
        nelec_f: ndarray of ints of shape (nfrag,)
            Used to constrain how many doubly-occupied orbitals (per mo_occ)
            are assigned to each fragment, if possible. If las is a single-state
            calculation, its quantum numbers are used as default values.

    Returns:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients after localization of the active space;
            columns in the order (inactive,las1,las2,...,lasn,external)
    '''
    log = logger.new_logger (las, las.verbose)
    if mo_coeff is None:
        mo_coeff = las.mo_coeff
    if lo_coeff is None:
        lo_coeff = orth.orth_ao (las.mol, 'meta_lowdin')
    if spin is None:
        spin = las.nelecas[0] - las.nelecas[1]
    assert (spin % 2 == sum (las.nelecas) % 2)
    assert (len (frags_atoms) == len (las.ncas_sub))
    frags_orbs = interpret_frags_atoms (las.mol, frags_atoms, frags_by_AOs=frags_by_AOs)
    if fock is None: fock = las._scf.get_fock ()
    ao_ovlp = las._scf.get_ovlp ()
    if mo_occ is not None:
        if np.count_nonzero (mo_occ==1) and (smults_f is None) and (las.nroots > 1):
            log.warn (("Can't enforce spin distributions for multiple states! Pass "
                       "smults_f kwarg to localize_init_guess to guarantee even "
                       "distribution of SOMOs!"))
        elif np.count_nonzero (mo_occ==1) and (smults_f is None):
            smults_f = las.get_smults_fr ()[:,0]
        if (nelec_f is None) and (las.nroots > 1):
            log.warn (("Can't enforce charge distributions for multiple states! Pass "
                       "nelec_f kwarg to localize_init_guess to guarantee even "
                       "distribution of DOMOs!"))
        elif (nelec_f is None):
            nelec_f = las.get_nelec_frs ()[:,0,:].sum (1)
    return _localize (las, frags_orbs, mo_coeff, spin, lo_coeff, fock, ao_ovlp,
                      mo_occ=mo_occ, freeze_cas_spaces=freeze_cas_spaces,
                      smults_f=smults_f, nelec_f=nelec_f)

def interpret_frags_atoms (mol, frags_atoms, frags_by_AOs=False):
    frags_atoms_int = all ([all ([isinstance (i, (int,np.integer)) for i in j]) for j in frags_atoms])
    frags_atoms_str = all ([all ([isinstance (i, (str,np.character)) for i in j]) for j in frags_atoms])
    if frags_atoms_int:
        if frags_by_AOs: return frags_atoms
        ao_offset = mol.offset_ao_by_atom ()
        frags_orbs = [[orb for atom in frags_atoms 
                       for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))]
                      for frags_atoms in frags_atoms]
    elif frags_atoms_str:
        frags_orbs = [list (mol.search_ao_label (i)) for i in frags_atoms]
    else:
        raise RuntimeError ('localize_init_guess requires either all integers or all strings to identify fragments')
    return frags_orbs

