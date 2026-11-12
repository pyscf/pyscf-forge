from pyscf.scf.rohf import get_roothaan_fock
from pyscf import fci
from pyscf.fci import cistring
from pyscf.mcscf import casci, casci_symm, df
from pyscf.tools import dump_mat
from pyscf import symm, gto, scf, ao2mo, lib
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.csf_fci import csf_solver
from pyscf.mcscf.las.addons import state_average_n_mix, get_h1e_zipped_fcisolver, las2cas_civec
from pyscf.mcscf.las import _DFLASCI, lasscf_guess, las_ao2mo
from pyscf.mcscf.las import chkfile
from pyscf.mcscf.las.productstate import ImpureProductStateFCISolver, state_average_fcisolver
from itertools import combinations, permutations, product
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg
import numpy as np
import copy

def LASCI (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASCISymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASCINoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las

def density_fit (las, auxbasis=None, with_df=None):
    ''' Here I ONLY need to attach the tag and the df object because I put conditionals in
        LASCINoSymm to make my life easier '''
    las_class = las.__class__
    if with_df is None:
        if (getattr(las._scf, 'with_df', None) and
            (auxbasis is None or auxbasis == las._scf.with_df.auxbasis)):
            with_df = las._scf.with_df
        else:
            with_df = df.DF(las.mol)
            with_df.max_memory = las.max_memory
            with_df.stdout = las.stdout
            with_df.verbose = las.verbose
            with_df.auxbasis = auxbasis
    class DFLASCI (las_class, _DFLASCI):
        def __init__(self, scf, ncas_sub, nelecas_sub):
            self.with_df = with_df
            self._keys = self._keys.union(['with_df'])
            las_class.__init__(self, scf, ncas_sub, nelecas_sub)
    new_las = DFLASCI (las._scf, las.ncas_sub, las.nelecas_sub)
    new_las.__dict__.update (las.__dict__)
    return new_las

def h1e_for_las (las, mo_coeff=None, ncas=None, ncore=None, nelecas=None, ci=None, ncas_sub=None,
                 nelecas_sub=None, veff=None, h2eff_sub=None, casdm1s_sub=None, casdm1frs=None,
                 eri_cas=None):
    ''' Effective one-body Hamiltonians (plural) for a LASCI problem

    Args:
        las: a LASCI object

    Kwargs:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients ordered on the columns as: 
            core orbitals, subspace 1, subspace 2, ..., external orbitals
        ncas: integer
            As in PySCF's existing CASCI/CASSCF implementation
        nelecas: sequence of 2 integers
            As in PySCF's existing CASCI/CASSCF implementation
        ci: list (length=nfrags) of list (length=nroots) of ndarrays
            Contains CI vectors
        ncas_sub: ndarray of shape (nsub)
            Number of active orbitals in each subspace
        nelecas_sub: ndarray of shape (nsub,2)
            na, nb in each subspace
        veff: ndarray of shape (2, nao, nao)
            Contains spin-separated, state-averaged effective potential
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        casdm1s_sub : list (length=nfrags) of ndarrays
            Contains state-averaged, spin-separated 1-RDMs in the localized active subspaces
        casdm1frs : list (length=nfrags) of list (length=nroots) of ndarrays
            Contains spin-separated 1-RDMs for each state in the localized active subspaces

    Returns:
        h1e_fr: list (length=nfrags) of list (length=nroots) of ndarrays
            Spin-separated 1-body Hamiltonian operator for each fragment and state
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ncas is None: ncas = las.ncas
    if ncore is None: ncore = las.ncore
    if ncas_sub is None: ncas_sub = las.ncas_sub
    if nelecas_sub is None: nelecas_sub = las.nelecas_sub
    if ncore is None: ncore = las.ncore
    if ci is None: ci = las.ci
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if casdm1frs is None: casdm1frs = las.states_make_casdm1s_sub (ci=ci)
    if casdm1s_sub is None: casdm1s_sub = [np.einsum ('rsij,r->sij',dm,las.weights)
                                           for dm in casdm1frs]
    if veff is None:
        veff = las.get_veff (dm = las.make_rdm1s (mo_coeff=mo_coeff, casdm1s_sub=casdm1s_sub))

    # First pass: split by root  
    nocc = ncore + ncas
    nmo = mo_coeff.shape[-1]
    mo_cas = mo_coeff[:,ncore:nocc]
    moH_cas = mo_cas.conj ().T 
    h1e = moH_cas @ (las.get_hcore ()[None,:,:] + veff) @ mo_cas
    h1e_r = np.empty ((las.nroots, 2, ncas, ncas), dtype=h1e.dtype)
    if eri_cas is None:
        eri_cas = lib.numpy_helper.unpack_tril (h2eff_sub.reshape (nmo*ncas,
            ncas*(ncas+1)//2)).reshape (nmo, ncas, ncas, ncas)[ncore:nocc,:,:,:]
    else:
        assert (eri_cas.shape==(ncas,ncas,ncas,ncas))
    h2e = eri_cas
    avgdm1s = np.stack ([linalg.block_diag (*[dm[spin] for dm in casdm1s_sub])
                         for spin in range (2)], axis=0)
    for state in range (las.nroots):
        statedm1s = np.stack ([linalg.block_diag (*[dm[state][spin] for dm in casdm1frs])
                               for spin in range (2)], axis=0)
        dm1s = statedm1s - avgdm1s 
        j = np.tensordot (dm1s, h2e, axes=((1,2),(2,3)))
        k = np.tensordot (dm1s, h2e, axes=((1,2),(2,1)))
        h1e_r[state] = h1e + j + j[::-1] - k


    # Second pass: split by fragment and subtract double-counting
    h1e_fr = []
    for ix, casdm1s_r in enumerate (casdm1frs):
        p = sum (las.ncas_sub[:ix])
        q = p + las.ncas_sub[ix]
        h1e = h1e_r[:,:,p:q,p:q]
        h2e = eri_cas[p:q,p:q,p:q,p:q]
        j = np.tensordot (casdm1s_r, h2e, axes=((2,3),(2,3)))
        k = np.tensordot (casdm1s_r, h2e, axes=((2,3),(2,1)))
        h1e_fr.append (h1e - j - j[:,::-1] + k)

    return h1e_fr

def get_fock (las, mo_coeff=None, ci=None, eris=None, casdm1s=None, verbose=None, veff=None,
              dm1s=None):
    ''' f_pq = h_pq + (g_pqrs - g_psrq/2) D_rs, AO basis
    Note the difference between this and h1e_for_las: h1e_for_las only has
    JK terms from electrons outside the "current" active subspace; get_fock
    includes JK from all electrons. This is also NOT the "generalized Fock matrix"
    of orbital gradients (but it can be used in calculating those if you do a
    semi-cumulant decomposition).
    The "eris" kwarg does not do anything and is retained only for backwards
    compatibility (also why I don't just call las.make_rdm1) '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if dm1s is None:
        if casdm1s is None: casdm1s = las.make_casdm1s (ci=ci)
        mo_cas = mo_coeff[:,las.ncore:][:,:las.ncas]
        moH_cas = mo_cas.conjugate ().T
        mo_core = mo_coeff[:,:las.ncore]
        moH_core = mo_core.conjugate ().T
        dm1s = [(mo_core @ moH_core) + (mo_cas @ d @ moH_cas) for d in list(casdm1s)]
    if veff is not None:
        fock = las.get_hcore()[None,:,:] + veff
        return get_roothaan_fock (fock, dm1s, las._scf.get_ovlp ())
    dm1 = dm1s[0] + dm1s[1]
    if isinstance (las, _DFLASCI):
        vj, vk = las.with_df.get_jk(dm1, hermi=1)
    else:
        vj, vk = las._scf.get_jk(las.mol, dm1, hermi=1)
    fock = las.get_hcore () + vj - (vk/2)
    return fock

def _unfrozen_inactive_virtual (las, nmo):
    '''Indices of the inactive and of the virtual orbitals which are not in las.frozen'''
    ncore = las.ncore
    nocc = ncore + las.ncas
    mask = np.ones (nmo, dtype=bool)
    frozen = getattr (las, 'frozen', None)
    if frozen is not None:
        if isinstance (frozen, (int, np.integer)):
            mask[:frozen] = False
        else:
            mask[frozen] = False
    return np.where (mask[:ncore])[0], np.where (mask[nocc:])[0] + nocc

def _eig_inactive_virtual (las, fock, orbsym=None):
    '''Generate the unitary matrix canonicalizing the inactive and virtual orbitals only.
    Orbitals in las.frozen are not mixed with any others.

    Args:
        las : object of :class:`LASCINoSymm`
        fock : ndarray of shape (nmo,nmo)
            Contains Fock matrix in MO basis

    Kwargs:
        orbsym : list of length nmo
        umat : ndarray of shape (nmo, nmo)

    Returns:
        ene : ndarray of shape (nmo,)
        umat : ndarray of shape (nmo, nmo)'''
    nmo = fock.shape[0]
    ene = np.zeros (nmo)
    umat = np.eye (nmo)
    for idx in _unfrozen_inactive_virtual (las, nmo):
        if not idx.size: continue
        orbsym_i = None if orbsym is None else np.asarray (orbsym)[idx]
        ene[idx], umat[np.ix_(idx,idx)] = las._eig (fock[np.ix_(idx,idx)], 0, 0, orbsym_i)
    return ene, umat

def canonicalize (las, mo_coeff=None, ci=None, casdm1fs=None, natorb_casdm1=None, veff=None,
                  h2eff_sub=None, orbsym=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci

    # In-place safety
    log = lib.logger.new_logger (las, las.verbose)
    t = (lib.logger.process_clock(), lib.logger.perf_counter())
    mo_coeff = mo_coeff.copy ()
    ci = copy.deepcopy (ci)
    t = log.timer_debug1 ('Canonicalize mo and ci copy', *t)

    # Temporary lroots safety
    # The desired behavior is that the inactive and external orbitals should
    # be canonicalized according to the density matrix used for orbital optimization
    # TODO: once orbital optimization with lroots is enabled, change this behavior
    # TODO: possibly move this logic to the make_casdm* functions
    if casdm1fs is None:
        ci_dm = []
        for i in range (len (ci)):
            ci_i = []
            for j in range (len (ci[i])):
                if ci[i][j].ndim>2:
                    ci_i.append (ci[i][j][0])
                else:
                    ci_i.append (ci[i][j])
            ci_dm.append (ci_i)
        casdm1fs = las.make_casdm1s_sub (ci=ci_dm)

    t = log.timer_debug1 ('Canonicalize make casdm1fs', *t)
    nao, nmo = mo_coeff.shape
    ncore = las.ncore
    nocc = ncore + las.ncas
    ncas_sub = las.ncas_sub
    nelecas_sub = las.nelecas_sub

    # Passing casdm1 or lasdm1 only affects the canonicalization of the active orbitals
    umat = np.zeros_like (mo_coeff)
    casdm1s = np.stack ([linalg.block_diag (*[dm[0] for dm in casdm1fs]),
                         linalg.block_diag (*[dm[1] for dm in casdm1fs])], axis=0)
    fock = mo_coeff.conjugate ().T @ las.get_fock (mo_coeff=mo_coeff, casdm1s=casdm1s, veff=veff)
    fock = fock @ mo_coeff
    if natorb_casdm1 is None: # State-average natural orbitals by default
        natorb_casdm1 = casdm1s.sum (0)

    t = log.timer_debug1 ('Canonicalize active orbitals', *t)
    # Inactive-inactive and virtual-virtual
    ene, umat = _eig_inactive_virtual (las, fock, orbsym=orbsym)
    idx = np.arange (nmo, dtype=int)
    for sub in _unfrozen_inactive_virtual (las, nmo): # Frozen orbitals keep their positions
        idx[sub] = sub[np.argsort (ene[sub])]
    umat = umat[:,idx]
    if orbsym is not None: orbsym = orbsym[idx]
    t = log.timer_debug1 ('Canonicalize inactive-inactive and virtual-virtual', *t)
    # Active-active
    check_diag = natorb_casdm1.copy ()
    for ix, ncas in enumerate (ncas_sub):
        i = sum (ncas_sub[:ix])
        j = i + ncas
        check_diag[i:j,i:j] = 0.0
    is_block_diag = np.amax (np.abs (check_diag)) < 1e-8
    t = log.timer_debug1 ('Canonicalize check block diag', *t)
    if is_block_diag:
        # No off-diagonal RDM elements -> extra effort to prevent diagonalizer from breaking frags
        for isub, (ncas, nelecas) in enumerate (zip (ncas_sub, nelecas_sub)):
            i = sum (ncas_sub[:isub])
            j = i + ncas
            dm1 = natorb_casdm1[i:j,i:j]
            i += ncore
            j += ncore
            orbsym_i = None if orbsym is None else orbsym[i:j]
            occ, umat[i:j,i:j] = las._eig (dm1, 0, 0, orbsym_i)
            idx = np.argsort (occ)[::-1]
            umat[i:j,i:j] = umat[i:j,i:j][:,idx]
            if orbsym_i is not None: orbsym[i:j] = orbsym[i:j][idx]
            t = log.timer_debug1 ('Canonicalize diag setup for fci box', *t)
            if ci is not None:
                t = log.timer_debug1 ('Canonicalize ci not none?', *t)
                fcibox = las.fciboxes[isub]
                t = log.timer_debug1 ('Canonicalize call fcibox', *t)
                ci[isub] = fcibox.states_transform_ci_for_orbital_rotation (
                    ci[isub], ncas, nelecas, umat[i:j,i:j])
            t = log.timer_debug1 ('Canonicalize diag setup for fci', *t)
    else: # You can't get proper LAS-type CI vectors w/out active space fragmentation
        ci = None 
        orbsym_cas = None if orbsym is None else orbsym[ncore:nocc]
        occ, umat[ncore:nocc,ncore:nocc] = las._eig (natorb_casdm1, 0, 0, orbsym_cas)
        idx = np.argsort (occ)[::-1]
        umat[ncore:nocc,ncore:nocc] = umat[ncore:nocc,ncore:nocc][:,idx]
        if orbsym_cas is not None: orbsym[ncore:nocc] = orbsym[ncore:nocc][idx]
        t = log.timer_debug1 ('Canonicalize non diag RDM', *t)

    t = log.timer_debug1 ('Canonicalize active-active', *t)
    # Final
    mo_occ = np.zeros (nmo, dtype=natorb_casdm1.dtype)
    if ncore: mo_occ[:ncore] = 2
    ucas = umat[ncore:nocc,ncore:nocc]
    mo_occ[ncore:nocc] = ((natorb_casdm1 @ ucas) * ucas).sum (0)
    mo_ene = ((fock @ umat) * umat.conjugate ()).sum (0)
    mo_ene[ncore:][:sum (ncas_sub)] = 0.0
    mo_coeff = mo_coeff @ umat
    if orbsym is not None:
        '''
        print ("This is the second call to label_orb_symm inside of canonicalize") 
        orbsym = symm.label_orb_symm (las.mol, las.mol.irrep_id,
                                      las.mol.symm_orb, mo_coeff,
                                      s=las._scf.get_ovlp ())
        #mo_coeff = las.label_symmetry_(mo_coeff)
        '''
        mo_coeff = lib.tag_array (mo_coeff, orbsym=orbsym)
    t = log.timer_debug1 ('Canonicalize final mo_coeff', *t)
    if h2eff_sub is not None:
        h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub.reshape (nmo*las.ncas, -1))
        h2eff_sub = h2eff_sub.reshape (nmo, las.ncas, las.ncas, las.ncas)
        h2eff_sub = np.tensordot (umat, h2eff_sub, axes=((0),(0)))
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(1))).transpose (1,0,2,3)
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(2))).transpose (1,2,0,3)
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(3))).transpose (1,2,3,0)
        h2eff_sub = h2eff_sub.reshape (nmo*las.ncas, las.ncas, las.ncas)
        h2eff_sub = lib.numpy_helper.pack_tril (h2eff_sub).reshape (nmo, -1)

    t = log.timer_debug1 ('Canonicalize final h2eff_update', *t)
    # I/O
    log = lib.logger.new_logger (las, las.verbose)
    label = las.mol.ao_labels()
    if las.verbose >= lib.logger.INFO and len (label) == mo_coeff.shape[0]:
        if is_block_diag:
            for isub, nlas in enumerate (ncas_sub):
                log.info ("Fragment %d natural orbitals", isub)
                i = ncore + sum (ncas_sub[:isub])
                j = i + nlas
                log.info ('Natural occ %s', str (mo_occ[i:j]))
                log.info ('Natural orbital (expansion on AOs) in CAS space')
                mo_las = mo_coeff[:,i:j]
                dump_mat.dump_rec(log.stdout, mo_las, label, start=1)
        else:
            log.info ("Delocalized natural orbitals do not reflect LAS fragmentation")
            log.info ('Natural occ %s', str (mo_occ[ncore:nocc]))
            log.info ('Natural orbital (expansion on AOs) in CAS space')
            mo_las = mo_coeff[:,ncore:nocc]
            dump_mat.dump_rec(log.stdout, mo_las, label, start=1)

    t = log.timer_debug1 ('Canonicalize I/O', *t)
    return mo_coeff, mo_ene, mo_occ, ci, h2eff_sub

def get_init_guess_ci (las, mo_coeff=None, h2eff_sub=None, ci0=None, eri_cas=None, gtype=None):
    if gtype is None: gtype = las.init_guess_ci
    if 'aufbau1' in gtype.lower ():
        return get_init_guess_ci_aufbau1 (las, mo_coeff=mo_coeff, h2eff_sub=h2eff_sub, ci0=ci0,
                                          eri_cas=eri_cas)
    elif 'aufbau' in gtype.lower ():
        return get_init_guess_ci_aufbau0 (las, mo_coeff=mo_coeff, ci0=ci0)
    elif 'vac' in gtype.lower ():
        return get_init_guess_ci_vac (las, mo_coeff=mo_coeff, h2eff_sub=h2eff_sub, ci0=ci0,
                                      eri_cas=eri_cas)
    else:
        raise NotImplementedError ("CI init guess of type {}".format (gtype))

def get_init_guess_ci_aufbau0 (las, mo_coeff=None, ci0=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci0 is None: ci0 = [[None for i in range (las.nroots)] for j in range (las.nfrags)]
    nmo = mo_coeff.shape[-1]
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    casdm1frs = []
    for ix, (fcibox, norb, nelecas) in enumerate (zip (las.fciboxes,las.ncas_sub,las.nelecas_sub)):
        i = sum (las.ncas_sub[:ix])
        j = i + norb
        orbsym = getattr (mo_coeff, 'orbsym', None)
        if orbsym is not None: orbsym=orbsym[ncore+i:ncore+j]
        ci0g = fcibox.get_aufbau_guess (norb, nelecas, orbsym=orbsym)
        for iy, solver in enumerate (fcibox.fcisolvers):
            nelec = fcibox._get_nelec (solver, nelecas)
            if hasattr (mo_coeff, 'orbsym'):
                solver.orbsym = mo_coeff.orbsym[ncore+i:ncore+j]
            ci0[ix][iy] = las._combine_init_guess_ci (ci0[ix][iy], ci0g[iy], norb, nelec,
                                                      solver.nroots)
    return ci0

def get_init_guess_ci_aufbau1 (las, mo_coeff=None, h2eff_sub=None, ci0=None, eri_cas=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci0 is None: ci0 = [[None for i in range (las.nroots)] for j in range (las.nfrags)]
    ci1 = [[ci0_ij for ci0_ij in ci0_i] for ci0_i in ci0]
    ci1 = get_init_guess_ci_aufbau0 (las, mo_coeff=mo_coeff, ci0=ci1)
    nmo = mo_coeff.shape[-1]
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    if eri_cas is None:
        if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
        eri_cas = lib.numpy_helper.unpack_tril (h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2))
        eri_cas = eri_cas.reshape (nmo, ncas, ncas, ncas)
        eri_cas = eri_cas[ncore:nocc]
    h1eff = las.get_h1eff (mo_coeff=mo_coeff, ci=ci1, eri_cas=eri_cas)
    for ix, (fcibox, norb, nelecas) in enumerate (zip (las.fciboxes,las.ncas_sub,las.nelecas_sub)):
        i = sum (las.ncas_sub[:ix])
        j = i + norb
        mo = mo_coeff[:,ncore+i:ncore+j]
        moH = mo.conj ().T
        h1e = h1eff[ix]
        eri = eri_cas[i:j,i:j,i:j,i:j]
        for iy, solver in enumerate (fcibox.fcisolvers):
            nelec = fcibox._get_nelec (solver, nelecas)
            if hasattr (mo_coeff, 'orbsym'):
                solver.orbsym = mo_coeff.orbsym[ncore+i:ncore+j]
            max_memory = max (400, las.max_memory - lib.current_memory()[0])
            hdiag_csf = solver.make_hdiag_csf (h1e, eri, norb, nelec, max_memory=max_memory)
            ci0g = solver.get_init_guess (norb, nelec, solver.nroots, hdiag_csf)
            ci0[ix][iy] = las._combine_init_guess_ci (ci0[ix][iy], ci0g, norb, nelec,
                                                      solver.nroots)
    return ci0

def get_init_guess_ci_vac (las, mo_coeff=None, h2eff_sub=None, ci0=None, eri_cas=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci0 is None: ci0 = [[None for i in range (las.nroots)] for j in range (las.nfrags)]
    nmo = mo_coeff.shape[-1]
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    if eri_cas is None:
        if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
        eri_cas = lib.numpy_helper.unpack_tril (h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2))
        eri_cas = eri_cas.reshape (nmo, ncas, ncas, ncas)
        eri_cas = eri_cas[ncore:nocc]
    dm1_core= 2 * mo_coeff[:,:ncore] @ mo_coeff[:,:ncore].conj ().T
    h1e_ao = las._scf.get_fock (dm=dm1_core)
    for ix, (fcibox, norb, nelecas) in enumerate (zip (las.fciboxes,las.ncas_sub,las.nelecas_sub)):
        i = sum (las.ncas_sub[:ix])
        j = i + norb
        mo = mo_coeff[:,ncore+i:ncore+j]
        moH = mo.conj ().T
        h1e = moH @ h1e_ao @ mo
        h1e = [h1e, h1e]
        eri = eri_cas[i:j,i:j,i:j,i:j]
        for iy, solver in enumerate (fcibox.fcisolvers):
            nelec = fcibox._get_nelec (solver, nelecas)
            if hasattr (mo_coeff, 'orbsym'):
                solver.orbsym = mo_coeff.orbsym[ncore+i:ncore+j]
            max_memory = max (400, las.max_memory - lib.current_memory()[0])
            hdiag_csf = solver.make_hdiag_csf (h1e, eri, norb, nelec, max_memory=max_memory)
            ci0g = solver.get_init_guess (norb, nelec, solver.nroots, hdiag_csf)
            ci0[ix][iy] = las._combine_init_guess_ci (ci0[ix][iy], ci0g, norb, nelec,
                                                      solver.nroots)
    return ci0


def _combine_init_guess_ci (las, ci0i, ci0g, norb, nelec, nroots):
    ''' Function to handle the combination of a generated set of guess CI vectors (ci0g) with
    existing CI vectors (ci0i) already stored. '''
    nroots0 = 0
    ndet = tuple ([cistring.num_strings (norb, n) for n in nelec])
    ci0g = np.asarray (ci0g)
    if isinstance (ci0g, np.ndarray) and ci0g.size % ndet[0]*ndet[1] == 0:
        ci0g = ci0g.reshape (-1, ndet[0], ndet[1])
    if isinstance (ci0i, np.ndarray) and ci0i.size % ndet[0]*ndet[1] == 0:
        ci0i = ci0i.reshape (-1, ndet[0], ndet[1])
        nroots0 = ci0i.shape[0]
    if nroots0 == 0:
        ci0i = ci0g
    elif nroots0 < nroots:
        cg = ci0g.reshape (nroots,-1)
        cs = ci0i.reshape (nroots0,-1)
        ovlp = cg.conj () @ cs.T
        cg -= ovlp @ cs
        ovlp = cg.conj () @ cg.T
        Q, R = linalg.qr (ovlp)
        ci0g = Q.T @ ci0g
        ci0i = np.append (ci0i, ci0g, axis=0)[:nroots]
    if nroots == 1: ci0i = ci0i[0]
    return ci0i

def get_space_info (las):
    ''' Retrieve the quantum numbers defining the states of a LASSCF calculation '''
    nfrags, nroots = las.nfrags, las.nroots
    charges = np.zeros ((nroots, nfrags), dtype=np.int32)
    wfnsyms, spins, smults = charges.copy (), charges.copy (), charges.copy ()
    for ifrag, fcibox in enumerate (las.fciboxes):
     for iroot, solver in enumerate (fcibox.fcisolvers):
        nelec = fcibox._get_nelec (solver, las.nelecas_sub[ifrag])
        charges[iroot,ifrag] = np.sum (las.nelecas_sub[ifrag]) - np.sum (nelec)
        spins[iroot,ifrag] = nelec[0]-nelec[1]
        smults[iroot,ifrag] = getattr (solver, 'smult', abs(spins[iroot,ifrag])+1)
        try:
            wfnsyms[iroot,ifrag] = getattr (solver, 'wfnsym', None) or 0
        except ValueError as e:
            wfnsyms[iroot,ifrag] = symm.irrep_name2id (las.mol.groupname, solver.wfnsym)
    return charges, spins, smults, wfnsyms
  
def get_smults_fr (las):
    return las.get_space_info ()[2].T

def get_nelec_frs (las):
    charges_rf, spins_rf = las.get_space_info ()[:2]
    nelec_f = np.asarray ([np.sum (nelec) for nelec in las.nelecas_sub])
    nelec_fr = nelec_f - charges_rf.T
    nelec_frs = np.stack ([nelec_fr + spins_rf.T,
                           nelec_fr - spins_rf.T],
                          axis=-1) // 2
    return nelec_frs

def assert_no_duplicates (las, tab=None):
    log = lib.logger.new_logger (las, las.verbose)
    if tab is None: tab = np.stack (get_space_info (las), axis=-1)
    tab_uniq, uniq_idx, uniq_inv, uniq_cnts = np.unique (tab, return_index=True,
        return_inverse=True, return_counts=True, axis=0)
    idx_dupe = uniq_cnts>1
    try:
        err_str = ('LAS state basis has duplicates; details in logfile for '
                   'verbose >= INFO (4) [more details for verbose > INFO].\n'
                   '(Disable this assertion by passing assert_no_dupes=False '
                   'to the kernel, lasci, and state_average(_) functions.)')
        assert (~np.any (idx_dupe)), err_str
    except AssertionError as e:
        dupe_idx = uniq_idx[idx_dupe]
        dupe_cnts = uniq_cnts[idx_dupe]
        for i, (ix, cnt, col) in enumerate (zip (uniq_idx, uniq_cnts, tab_uniq)):
            if cnt==1: continue
            log.info ('State %d appears %d times', ix, cnt)
            idx_thisdupe = np.where (uniq_inv==i)[0]
            row = col.T
            log.debug ('As states {}'.format (idx_thisdupe))
            log.debug ('Charges = {}'.format (row[0]))
            log.debug ('2M_S = {}'.format (row[1]))
            log.debug ('2S+1 = {}'.format (row[2]))
            log.debug ('Wfnsyms = {}'.format (row[3]))
        raise e from None

def state_average_(las, weights=[0.5,0.5], charges=None, spins=None,
        smults=None, wfnsyms=None, lroots=None, lweights=None,
        assert_no_dupes=True):
    ''' Transform LASCI/LASSCF object into state-average LASCI/LASSCF 

    Args:
        las: LASCI/LASSCF instance

    Kwargs:
        weights: list of float; required
            E_SA = sum_i weights[i] E[i] is used to optimize the orbitals
        charges: 2d ndarray or nested list of integers
        spins: 2d ndarray or nested list of integers
            For the jth fragment in the ith state,
            neleca = (sum(las.nelecas_sub[j]) - charges[i][j] + spins[i][j]) // 2
            nelecb = (sum(las.nelecas_sub[j]) - charges[i][j] - spins[i][j]) // 2
            Defaults to
            charges[i][j] = 0
            spins[i][j] = las.nelecas_sub[j][0] - las.nelecas_sub[j][1]
        smults: 2d ndarray or nested list of integers
            For the jth fragment in the ith state,
            smults[i][j] = (2*s)+1
            where "s" is the total spin quantum number,
            S^2|j,i> = s*(s+1)|j,i>
            Defaults to
            smults[i][j] = abs (spins[i][j]) + 1
        wfnsyms: 2d ndarray or nested list of integers or strings
            For the jth fragment of the ith state,
            wfnsyms[i][j]
            identifies the point-group irreducible representation
            Defaults to all zeros (i.e., the totally-symmetric irrep)
        lroots: ndarray of shape (nroots,nfrags)
            Number of local roots in each fragment for each global state. 
            The corresponding local weights are set to [1/n,1/n,1/n,...].
            Note that this is transposed compared to the kwarg of kernel,
            in order to be consistent with the other kwargs of this function!
        lweights: list of length nfrags of list of length nroots of sequence
            Weights of local roots in each fragment for each global state.
            Passing lweights is incompatible with passing lroots. Defaults
            to, i.e., np.ones (las.nfrags, las.nroots, 1).tolist ()

    Returns:
        las: LASCI/LASSCF instance
            The first positional argument, modified in-place into a
            state-averaged LASCI/LASSCF instance.

    '''
    # TODO: incorporate lroots counting into get_space_info and simplify this!
    old_states = np.stack (get_space_info (las), axis=-1)
    nroots = len (weights)
    nfrags = las.nfrags
    if charges is None: charges = np.zeros ((nroots, nfrags), dtype=np.int32)
    if wfnsyms is None: wfnsyms = np.zeros ((nroots, nfrags), dtype=np.int32)
    if spins is None: spins = np.asarray ([[n[0]-n[1] for n in las.nelecas_sub] for i in weights]) 
    if smults is None: smults = np.abs (spins)+1 
    if lroots is not None and lweights is not None:
        raise RuntimeError ("lroots sets lweights: pass either or none but not both")
    elif lweights is None:
        if lroots is None:
            lroots = np.ones ((nfrags, nroots), dtype=int)
        else:
            lroots = np.asarray (lroots).T
        lweights = []
        for i in range (nfrags):
            lwi = []
            for j in range (nroots):
                lwij = np.ones (lroots[i,j], dtype=float) / lroots[i,j]
                lwi.append (lwij)
            lweights.append (lwi)
    else:
        lroots = np.array ([[len (lwij) for lwij in lwi] for lwi in lweights])
    if np.any (lroots>1): raise NotImplementedError ("Local state-averaged LASSCF")

    charges = np.asarray (charges)
    wfnsyms = np.asarray (wfnsyms)
    spins = np.asarray (spins)
    smults = np.asarray (smults)
    if np.issubdtype (wfnsyms.dtype, np.str_):
        wfnsyms_str = wfnsyms
        wfnsyms = np.zeros (wfnsyms_str.shape, dtype=np.int32)
        for ix, wfnsym in enumerate (wfnsyms_str.flat):
            try:
                wfnsyms.flat[ix] = symm.irrep_name2id (las.mol.groupname, wfnsym)
            except (TypeError, KeyError) as e:
                wfnsyms.flat[ix] = int (wfnsym)
    if nfrags == 1:
        charges = np.atleast_2d (np.squeeze (charges)).T
        wfnsyms = np.atleast_2d (np.squeeze (wfnsyms)).T
        spins = np.atleast_2d (np.squeeze (spins)).T
        smults = np.atleast_2d (np.squeeze (smults)).T
    new_states = np.stack ([charges, spins, smults, wfnsyms], axis=-1)
    if assert_no_dupes: assert_no_duplicates (las, tab=new_states)

    las.fciboxes = [get_h1e_zipped_fcisolver (state_average_n_mix (
        las, [csf_solver (las.mol, smult=s2p1).set (charge=c, spin=m2, wfnsym=ir)
              for c, m2, s2p1, ir in zip (c_r, m2_r, s2p1_r, ir_r)], weights).fcisolver)
        for c_r, m2_r, s2p1_r, ir_r in zip (charges.T, spins.T, smults.T, wfnsyms.T)]
    las.e_states = np.zeros (nroots)
    las.nroots = nroots
    las.weights = weights
    for ifrag, fcibox in enumerate (las.fciboxes):
        for iroot in range (las.nroots):
            if lroots[ifrag][iroot] == 1: continue
            fcibox.fcisolvers[iroot] = state_average_fcisolver (
                fcibox.fcisolvers[iroot], weights=lweights[ifrag][iroot]
            )

    if las.ci is not None:
        log = lib.logger.new_logger(las, las.verbose)
        log.debug (("lasci.state_average: Cached CI vectors may be present.\n"
                    "Looking for matches between old and new LAS states..."))
        ci0 = [[None for i in range (nroots)] for j in range (nfrags)]
        new_states = np.stack ([charges, spins, smults, wfnsyms],
            axis=-1).reshape (nroots, nfrags*4)
        old_states = old_states.reshape (-1, nfrags*4)
        for iroot, row in enumerate (old_states):
            idx = np.all (new_states == row[None,:], axis=1)
            if np.count_nonzero (idx) == 1:
                jroot = np.where (idx)[0][0] 
                log.debug ("Old state {} -> New state {}".format (iroot, jroot))
                for ifrag in range (nfrags):
                    ci0[ifrag][jroot] = las.ci[ifrag][iroot]
            elif np.count_nonzero (idx) > 1:
                raise RuntimeError ("Duplicate states specified?\n{}".format (idx))
        las.ci = ci0 

    las.converged = False
    return las

@lib.with_doc(''' A version of lasci.state_average_ that creates a copy instead of modifying the 
    LASCI/LASSCF method instance in place.

    See lasci.state_average_ docstring below:\n\n''' + state_average_.__doc__)
def state_average (las, weights=[0.5,0.5], charges=None, spins=None,
        smults=None, wfnsyms=None, lroots=None, lweights=None, assert_no_dupes=True):
    is_scanner = isinstance (las, lib.SinglePointScanner)
    if is_scanner: las = las.undo_scanner ()
    new_las = las.__class__(las._scf, las.ncas_sub, las.nelecas_sub)
    new_las.__dict__.update (las.__dict__)
    new_las.mo_coeff = las.mo_coeff.copy ()
    if getattr (las.mo_coeff, 'orbsym', None) is not None:
        new_las.mo_coeff = lib.tag_array (new_las.mo_coeff,
            orbsym=las.mo_coeff.orbsym)
    new_las.ci = None
    if las.ci is not None:
        new_las.ci = [[c2.copy () if isinstance (c2, np.ndarray) else None
            for c2 in c1] for c1 in las.ci]
    las = state_average_(new_las, weights=weights, charges=charges, spins=spins,
                         smults=smults, wfnsyms=wfnsyms, lroots=lroots, lweights=lweights,
                         assert_no_dupes=assert_no_dupes)
    if is_scanner: las = las.as_scanner ()
    return las

def get_single_state_las (las, state=0):
    ''' Quickly extract a state-specific las calculation from a state-average one '''
    charges, spins, smults, wfnsyms = get_space_info (las)
    charges = charges[state:state+1]
    spins = spins[state:state+1]
    smults = smults[state:state+1]
    wfnsyms = wfnsyms[state:state+1]
    weights = [1,]
    return state_average (las, weights=weights, charges=charges, spins=spins, smults=smults,
                          wfnsyms=wfnsyms)

def kernel (las, mo_coeff=None, ci0=None, lroots=None, lweights=None, verbose=0,
               assert_no_dupes=False, _dry_run=False):
    '''Self-consistently optimize the CI vectors of a LAS state with 
    frozen orbitals using a fixed-point algorithm. "lasci_" (with the
    trailing underscore) sets self.mo_coeff from the kwarg if it is passed;
    "lasci" (without the trailing underscore) leaves self.mo_coeff unchanged.

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            MO coefficients; defaults to self.mo_coeff
        ci0 : list (length nfrags) of list (length nroots) of ndarrays
            Contains CI vectors for initial guess
        lroots : ndarray of shape (nfrags,nroots)
            Number of local roots in each fragment for each global state. 
            The corresponding local weights are set to [1,0,0,0,...].
        lweights : list of length nfrags of list of length nroots of sequence
            Weights of local roots in each fragment for each global state.
            Passing lweights is incompatible with passing lroots. Defaults
            to, i.e., np.ones (las.nfrags, las.nroots, 1).tolist ()
        verbose : integer
            See pyscf.lib.logger.
        assert_no_dupes : logical
            If True, checks state list for duplicate states
        _dry_run : logical
            If True, sets up the fcisolvers with the appropriate tags, but does
            not run fcisolver kernels.

    Returns:
        converged : list of length nroots of logical
            Stores whether the calculation for each state successfully converged
        e_tot : float
            (State-averaged) total energy
        e_states : list of length nroots
            List of each state energy
        e_cas : list of length nroots
            List of the CAS space energy of each state
        ci : list (length nfrags) of list (length nroots) of ndarrays
            Contains optimized CI vectors
    '''
    if assert_no_dupes: assert_no_duplicates (las)
    if lroots is not None and lweights is not None:
        raise RuntimeError ("lroots sets lweights: pass either or none but not both")
    elif lweights is None:
        if lroots is None: lroots = np.ones ((las.nfrags, las.nroots), dtype=int)
        lweights = []
        for i in range (las.nfrags):
            lwi = []
            for j in range (las.nroots):
                lwij = np.zeros (lroots[i,j])
                lwij[0] = 1
                lwi.append (lwij)
            lweights.append (lwi)
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    ncas_sub = las.ncas_sub
    nelecas_sub = las.nelecas_sub
    orbsym = getattr (mo_coeff, 'orbsym', None)
    if orbsym is not None: orbsym=orbsym[ncore:nocc]
    elif isinstance (las, LASCISymm):
        mo_coeff = las.label_symmetry_(mo_coeff)
        orbsym = mo_coeff.orbsym[ncore:nocc]
    log = lib.logger.new_logger (las, verbose)

    h1eff, energy_core = las.h1e_for_cas (mo_coeff=mo_coeff,
        ncas=las.ncas, ncore=las.ncore)
    eri_cas = las.get_h2cas (mo_coeff) 
    if (ci0 is None or any ([c is None for c in ci0]) or
            any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        ci0 = las.get_init_guess_ci (mo_coeff, ci0=ci0, eri_cas=eri_cas)

    e_cas = np.empty (las.nroots)
    e_states = np.empty (las.nroots)
    ci1 = [[None for c2 in c1] for c1 in ci0]
    converged = []
    t = (lib.logger.process_clock(), lib.logger.perf_counter())
    e_lexc = [[None for i in range (las.nroots)] for j in range (las.nfrags)]
    for state in range (las.nroots):
        fcisolvers = [b.fcisolvers[state] for b in las.fciboxes]
        ci0_i = [c[state] for c in ci0]
        solver = ImpureProductStateFCISolver (fcisolvers, stdout=las.stdout,
            lweights=[l[state] for l in lweights], verbose=verbose)
        for ix, s in enumerate (solver.fcisolvers):
            # Set the calling las objects local bottom-layer fcisolvers to the
            # locally-state-averaged ones I just made so that I can more easily get
            # locally-state-averaged density matrices after this function exits
            las.fciboxes[ix].fcisolvers[state] = s
            i = sum (ncas_sub[:ix])
            j = i + ncas_sub[ix]
            if orbsym is not None: s.orbsym = orbsym[i:j]
            s.norb = ncas_sub[ix]
            s.nelec = solver._get_nelec (s, nelecas_sub[ix])
            s.check_transformer_cache ()
        if _dry_run: continue
        conv, e_i, ci_i = solver.kernel (h1eff, eri_cas, ncas_sub, nelecas_sub,
            ecore=0, ci0=ci0_i, orbsym=orbsym, conv_tol_grad=las.conv_tol_grad,
            conv_tol_self=las.conv_tol_self, max_cycle_macro=las.max_cycle_macro)
        e_cas[state] = e_i
        e_states[state] = e_i + energy_core
        for frag, s in enumerate (solver.fcisolvers):
            e_loc = np.atleast_1d (getattr (s, 'e_states', e_i))
            e_lexc[frag][state] = e_loc - e_i
        for c1, c2, s, no, ne in zip (ci1, ci_i, solver.fcisolvers, ncas_sub, nelecas_sub):
            ne = solver._get_nelec (s, ne)
            ndeta, ndetb = [cistring.num_strings (no, n) for n in ne]
            shape = [s.nroots, ndeta, ndetb] if s.nroots>1 else [ndeta, ndetb]
            c1[state] = np.asarray (c2).reshape (*shape)
        if not conv: log.warn ('State %d LASCI not converged!', state)
        converged.append (conv)
        t = log.timer ('State {} LASCI'.format (state), *t)

    e_tot = np.dot (las.weights, e_states)
    return converged, e_tot, e_states, e_cas, e_lexc, ci1


def get_nelec_frs (las):
    ''' Getter function for all electron occupancies in all fragments in all rootspaces.

    Returns:
        nelec_frs: ndarray of shape (nfrags, nroots, 2)
            number of electrons of each spin in each rootspace in each fragment
    '''
    nelec_frs = []
    for fcibox, nelec in zip (las.fciboxes, las.nelecas_sub):
        nelec_rs = []
        for solver in fcibox.fcisolvers:
            nelec_rs.append (_unpack_nelec (fcibox._get_nelec (solver, nelec)))
        nelec_frs.append (nelec_rs)
    return np.asarray (nelec_frs)

def get_sym_fr (las):
    ''' Getter function for irrep IDs of all fragments in all rootspaces.

    Returns:
        sym_fr: ndarray of shape (nfrags, nroots)
            irrep IDs in each rootspace in each fragment
    '''
    sym_fr = []
    for fcibox in las.fciboxes:
        sym_r = []
        for solver in fcibox.fcisolvers:
            sym = getattr (solver, 'wfnsym', 0) or 0
            if isinstance (sym, str):
                sym = symm.irrep_name2id (solver.mol.groupname, sym)
            sym_r.append (sym)
        sym_fr.append (sym_r)
    return np.asarray (sym_fr)

def _svd_subspaces (s, lspace, rspace, zero_atol=1e-8):
    '''SVD of lspace^H s rspace, with the singular vectors in the original basis

    Args:
        s: ndarray of shape (nao,nao)
        lspace: ndarray of shape (nao,m)
        rspace: ndarray of shape (nao,n)

    Kwargs:
        zero_atol: float
            If every element of lspace^H s rspace, or its norm divided by its size, is
            below this, the matrix is treated as zero.

    Returns:
        lvecs: ndarray of shape (nao,m)
            lspace @ u, including the null space
        svals: ndarray of shape (min (m,n),)
            In descending order; all zero if the matrix is treated as zero
        rvecs: ndarray of shape (nao,n)
            rspace @ v, including the null space. If the matrix is treated as zero,
            lspace and rspace are returned unchanged.
    '''
    lspace, rspace = np.asarray (lspace), np.asarray (rspace)
    k = min (lspace.shape[1], rspace.shape[1])
    if k == 0: return lspace, np.zeros (0), rspace
    mat = lspace.conjugate ().T @ s @ rspace
    if ((np.amax (np.abs (mat)) < zero_atol)
            or (linalg.norm (mat) / mat.size < zero_atol)):
        return lspace, np.zeros (k), rspace
    u, svals, vh = linalg.svd (mat, full_matrices=True)
    idx = np.abs (svals).argsort ()[::-1]
    svals = svals[idx]
    u[:,:k] = u[:,:k][:,idx]
    v = vh.conjugate ().T
    v[:,:k] = v[:,:k][:,idx]
    return lspace @ u, svals, rspace @ v

def _svd_subspaces_symm (s, lspace, rspace, lsymm, rsymm, zero_atol=1e-8):
    '''Like _svd_subspaces, but block-diagonal in the irreps labeling the columns of lspace
    (lsymm) and rspace (rsymm); singular vectors couple only columns with the same label.

    Returns:
        lvecs, svals, rvecs: as in _svd_subspaces, except that the null-space columns
            of lvecs and rvecs come last, grouped by irrep, and that svals is padded with
            zeros to length min (m,n) if the blocks give fewer
        llabels: ndarray of shape (m,)
            Irrep of each column of lvecs
        rlabels: ndarray of shape (n,)
            Irrep of each column of rvecs
    '''
    lspace, rspace = np.asarray (lspace), np.asarray (rspace)
    k = min (lspace.shape[1], rspace.shape[1])
    if k == 0: return lspace, np.zeros (0), rspace, lsymm, rsymm
    mat = lspace.conjugate ().T @ s @ rspace
    if ((np.amax (np.abs (mat)) < zero_atol)
            or (linalg.norm (mat) / mat.size < zero_atol)):
        return lspace, np.zeros (k), rspace, lsymm, rsymm
    lsymm = np.asarray (lsymm).astype (int)
    rsymm = np.asarray (rsymm).astype (int)
    svals = []
    vecs = [[], []]
    vecs_null = [[], []]
    labels = [[], []]
    labels_null = [[], []]
    for lbl in np.unique (np.append (lsymm, rsymm)):
        vecs_blk = [None, None]
        vecs_blk[0], svals_blk, vecs_blk[1] = _svd_subspaces (
            s, lspace[:,lsymm==lbl], rspace[:,rsymm==lbl], zero_atol=zero_atol)
        nvals = len (svals_blk)
        svals.append (svals_blk)
        for i in range (2):
            vecs[i].append (vecs_blk[i][:,:nvals])
            vecs_null[i].append (vecs_blk[i][:,nvals:])
            labels[i].extend ([lbl,]*nvals)
            labels_null[i].extend ([lbl,]*(vecs_blk[i].shape[1]-nvals))
    svals = np.concatenate (svals)
    nvals = len (svals)
    idx = svals.argsort ()[::-1]
    svals = svals[idx]
    vecs = [np.concatenate (x, axis=1)[:,idx] for x in vecs]
    labels = [np.asarray (x)[idx] for x in labels]
    vecs = [np.append (x, np.concatenate (y, axis=1), axis=1) for x, y in zip (vecs, vecs_null)]
    labels = [np.append (x, np.asarray (y).astype (int)) for x, y in zip (labels, labels_null)]
    if nvals < k: svals = np.append (svals, np.zeros (k - nvals))
    return vecs[0], svals, vecs[1], labels[0], labels[1]

def _shift_svals (l, sv, r, rng):
    k = len (sv)
    if rng is not None:
        idx = np.argsort (-sv)
        l[:,:k] = l[:,:k][:,idx]
        sv = sv[idx]
        r[:,:k] = r[:,:k][:,idx]
        if rng[0] >= 0:
            sv[:min (len (sv), rng[0])] += 1
        if (rng[1] >=0) and (rng[1] < len (sv)):
            sv[rng[1]:] -= 1
        idx = np.argsort (-sv)
        l[:,:k] = l[:,:k][:,idx]
        sv = sv[idx]
        r[:,:k] = r[:,:k][:,idx]
    return l, sv, r

class LASCINoSymm (casci.CASCI):

    get_space_info = get_space_info
    get_smults_fr = get_smults_fr
    get_nelec_frs = get_nelec_frs

    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, frozen=None, frozen_ci=None, **kwargs):
        self.init_guess_ci = 'aufbau1'
        if isinstance(ncas,int):
            ncas = [ncas]
        ncas_tot = sum (ncas)
        nel_tot = [0, 0]
        new_nelecas = []
        for ix, nel in enumerate (nelecas):
            if isinstance (nel, (int, np.integer)):
                nb = nel // 2
                na = nb + (nel % 2)
            else:
                na, nb = nel
            new_nelecas.append ((na, nb))
            nel_tot[0] += na
            nel_tot[1] += nb
        nelecas = new_nelecas
        self.nroots = 1
        super().__init__(mf, ncas=ncas_tot, nelecas=nel_tot, ncore=ncore)
        self.chkfile = self._scf.chkfile
        if spin_sub is None: spin_sub = [1 + abs(ne[0]-ne[1]) for ne in nelecas]
        self.ncas_sub = np.asarray (ncas)
        self.nelecas_sub = np.asarray (nelecas)
        assert (len (self.nelecas_sub) == self.nfrags)
        self.frozen = frozen
        self.frozen_ci = frozen_ci
        self.conv_tol_grad = 1e-4
        self.conv_tol_self = 1e-10
        self.ah_level_shift = 1e-8
        self.max_cycle_macro = 50
        self.max_cycle_micro = 5
        self.min_cycle_macro = 0
        self.trust_radius = np.pi
        keys = set(('e_states', 'fciboxes', 'nroots', 'weights', 'ncas_sub', 'nelecas_sub',
                    'conv_tol_grad', 'conv_tol_self', 'max_cycle_macro', 'max_cycle_micro',
                    'ah_level_shift', 'states_converged', 'chkfile', 'e_lexc', 'trust_radius'))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.fciboxes = []
        if isinstance(spin_sub,int):
            self.fciboxes.append(self._init_fcibox(spin_sub,self.nelecas_sub[0]))
        else:
            assert (len (spin_sub) == self.nfrags)
            for smult, nel in zip (spin_sub, self.nelecas_sub):
                self.fciboxes.append (self._init_fcibox (smult, nel))
        self.weights = [1.0]
        self.e_states = [0.0]
        self.e_lexc = [[np.array ([0]),],]

    def copy (self):
        # semi-deep copy of nested lists
        mycopy = super().copy ()
        mycopy.ci = [[xij for xij in xi] for xi in self.ci]
        return mycopy

    def _init_fcibox (self, smult, nel):
        s = csf_solver (self.mol, smult=smult)
        s.spin = nel[0] - nel[1] 
        return get_h1e_zipped_fcisolver (state_average_n_mix (self, [s], [1.0]).fcisolver)

    @property
    def nfrags (self): return len (self.ncas_sub)

    def get_mo_slice (self, idx, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo = mo_coeff[:,self.ncore:]
        for offs in self.ncas_sub[:idx]:
            mo = mo[:,offs:]
        mo = mo[:,:self.ncas_sub[idx]]
        return mo

    get_sym_fr = get_sym_fr
    get_nelec_frs = get_nelec_frs
    get_h1eff = get_h1las = h1e_for_las = h1e_for_las
    get_h2eff = ao2mo = las_ao2mo.get_h2eff
    def _get_h2eff_df (self, mo_coeff):
        return las_ao2mo.get_h2eff_df (self, mo_coeff)
    get_h2cas = las_ao2mo.get_h2cas
    get_h2eff_slice = las_ao2mo.get_h2eff_slice
    '''
    def get_h2eff (self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if isinstance (self, _DFLASCI):
            mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
            return self.with_df.ao2mo (mo_cas)
        return self.ao2mo (mo_coeff)
    '''

    get_fock = get_fock
    canonicalize = canonicalize

    def _finalize(self, method='LASCI'):
        log = lib.logger.new_logger (self, self.verbose)
        nroots_prt = len (self.e_states)
        if self.verbose <= lib.logger.INFO:
            nroots_prt = min (nroots_prt, 100)
        if nroots_prt < len (self.e_states):
            log.info (("Printing a maximum of 100 state energies;"
                       " increase self.verbose to see them all"))
        if nroots_prt > 1:
            log.info ("%s state-average energy = %.15g", method, self.e_tot)
            for i, e in enumerate (self.e_states[:nroots_prt]):
                log.info ("%s state %d energy = %.15g", method, i, e)
        else:
            log.info ("%s energy = %.15g", method, self.e_tot)
        return

    def states_make_casdm1s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Get spin-separated, rootspace-separated, locally state-averaged 1-RDMs of the active
        orbitals for each fragment in sequence.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 

        Returns:
            casdm1frs: list of ndarray of shape (nroots, 2, ncas_sub[i], ncas_sub[i])
                Spin-separated 1-body reduced density matrices for each fragment for each
                rootspace, locally state-averaged if applicable.
        '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None:
            return [np.zeros ((self.nroots,2,ncas,ncas)) for ncas in ncas_sub] 
        casdm1s = []
        for fcibox, ci_i, ncas, nelecas in zip (self.fciboxes, ci, ncas_sub, nelecas_sub):
            if ci_i is None:
                dm1a = dm1b = np.zeros ((ncas, ncas))
            else:
                dm1a, dm1b = fcibox.states_make_rdm1s (ci_i, ncas, nelecas)
            casdm1s.append (np.stack ([dm1a, dm1b], axis=1))
        return casdm1s

    def make_casdm1s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm1frs=None, w=None, **kwargs):
        ''' Get spin-separated, rootspace- and state-averaged 1-RDMs of the active orbitals for
        each fragment in sequence.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 

        Returns:
            casdm1fs: list of ndarray of shape (2, ncas_sub[i], ncas_sub[i])
                Spin-separated 1-body reduced density matrices for each fragment,
                rootspace-averaged and locally state-averaged if applicable.
        '''
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        if w is None: w = self.weights
        return [np.einsum ('rspq,r->spq', dm1, w) for dm1 in casdm1frs]

    def states_make_casdm1s (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm1frs=None, **kwargs):
        ''' Get spin-separated, rootspace-separated, locally state-averaged 1-RDMs of all active
        orbitals collectively.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 
            casdm1frs: sequence of ndarrays of shape (nroots, 2, ncas_sub[i], ncas_sub[i])
                output of states_make_casdm1s_sub

        Returns:
            casdm1rs: ndarray of shape (nroots, 2, ncas, ncas)
                Spin-separated 1-body reduced density matrix for the active orbitals for each
                rootspace, locally state-averaged if applicable.
        '''
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        return np.stack ([np.stack ([linalg.block_diag (*[dm1rs[iroot][ispin] 
                                                          for dm1rs in casdm1frs])
                                     for ispin in (0, 1)], axis=0)
                          for iroot in range (self.nroots)], axis=0)

    def states_make_casdm2_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Get spin-summed, rootspace-separated, locally state-averaged 2-RDMs of the active
        orbitals for each fragment in sequence.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 

        Returns:
            casdm2fr: list of ndarray of shape [nroots,]+[ncas_sub[i]]*4
                Spin-summed 2-body reduced density matrices for each fragment for each
                rootspace, locally state-averaged if applicable.
        '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        casdm2 = []
        for fcibox, ci_i, ncas, nel in zip (self.fciboxes, ci, ncas_sub, nelecas_sub):
            casdm2.append (fcibox.states_make_rdm12 (ci_i, ncas, nel)[-1])
        return casdm2

    def make_casdm2_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, casdm2fr=None, **kwargs):
        ''' Get spin-summed, rootspace- and state-averaged 2-RDMs of the active orbitals for each
        fragment in sequence.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 
            casdm2fr: sequence of ndarrays of shape [nroots,] + [ncas_sub[i],]*4
                output of states_make_casdm2_sub

        Returns:
            casdm2f: list of ndarray of shape [nroots,]+[ncas_sub[i]]*4
                Spin-summed 2-body reduced density matrices for each fragment, rootspace- and
                state-averaged if applicable.
        '''
        if casdm2fr is None: casdm2fr = self.states_make_casdm2_sub (ci=ci, ncas_sub=ncas_sub,
            nelecas_sub=nelecas_sub, **kwargs)
        return [np.einsum ('rijkl,r->ijkl', dm2, box.weights)
                for dm2, box in zip (casdm2fr, self.fciboxes)]


    def make_casdm2s_sub(self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm2frs=None, w=None, **kwargs):
        if casdm2frs is None: casdm2frs = self.states_make_casdm2s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        if w is None: w = self.weights
        return [np.einsum ('rsijkl,r->sijkl', dm2, w) for dm2 in casdm2frs]

    def states_make_casdm2s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        '''Spin-separated 2-RDMs in the MO basis for each subspace in sequence'''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None:
            return [np.zeros ((self.nroots,2,ncas,ncas,ncas,ncas)) for ncas in ncas_sub]
        casdm2s = []
        for fcibox, ci_i, ncas, nelecas in zip (self.fciboxes, ci, ncas_sub, nelecas_sub):
            if ci_i is None:
                dm2aa = dm2ab, dm2bb = np.zeros ((ncas, ncas,ncas,ncas))
            else:
                dm2aa, dm2ab, dm2bb = fcibox.states_make_rdm12s (ci_i, ncas, nelecas)[-1]
            casdm2s.append (np.stack ([dm2aa, dm2ab, dm2bb], axis=1))
        return casdm2s
    

    def states_make_casdm3_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        ''' Spin-separated 3-RDMs in the MO basis for each subspace in sequence, currently this
        does not have weights so it's not really a STATES '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        casdm3 = []
        for ci_i, ncas, nel in zip (ci, ncas_sub, nelecas_sub):
            dm1_not_no, dm2_not_no, dm3_not_no = fci.rdm.make_dm123 (
                'FCI3pdm_kern_sf',ci_i,ci_i, ncas, nel) # not normal ordered
            dm3_no = fci.rdm.reorder_dm123(dm1_not_no, dm2_not_no, dm3_not_no)[-1]
            casdm3.append (dm3_no)
        return casdm3

    def make_casdm3s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        '''Spin-separated 3-RDMs in the MO basis for each subspace in sequence using the PySCF make_rdm123s'''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None:
            return [np.zeros ((self.nroots,2,ncas,ncas,ncas,ncas,ncas,ncas)) for ncas in ncas_sub]
        casdm3s = []
        for fcibox, ci_i, ncas, nelecas in zip (self.fciboxes, ci, ncas_sub, nelecas_sub):
            dm3aaa, dm3aab, dm3abb, dm3bbb = fci.direct_spin1.make_rdm123s (np.array(ci_i), ncas, tuple(nelecas))[-1]
            casdm3s.append (np.stack ([dm3aaa, dm3aab, dm3abb, dm3bbb], axis=0))
        return casdm3s

    def states_make_rdm1s (self, mo_coeff=None, ci=None, ncas_sub=None,
            nelecas_sub=None, casdm1rs=None, casdm1frs=None, **kwargs):
        ''' Get spin-separated, rootspace-separated, locally state-averaged 1-RDMs in the AO basis.

        Kwargs:
            mo_coeff: ndarray of shape (nao,nmo)
                Contains MO coefficients
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace
            casdm1rs: ndarray of shape (nroots,2,ncas,ncas)
                output of states_make_casdm1s
            casdm1frs: list of ndarray of shape (nroots,2,ncas_sub[i],ncas_sub[i])
                output of states_make_casdm1s_sub

        Returns:
            dm1rs: ndarray of shape (nroots,2,nao,nao)
                Rootspace-separated, locally state-averaged, spin-separated 1-RDMs
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1rs is None: casdm1rs = self.states_make_casdm1s (ci=ci, 
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, casdm1frs=casdm1frs, 
            **kwargs)
        mo_core = mo_coeff[:,:self.ncore]
        mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
        dm1rs = np.tensordot (mo_cas.conj (), np.dot (casdm1rs, mo_cas.conj ().T), axes=((1),(2)))
        dm1rs = dm1rs.transpose (1,2,0,3)
        dm1rs += (mo_core @ mo_core.conj ().T)[None,None,:,:]
        return dm1rs

    def make_rdm1s_sub (self, mo_coeff=None, ci=None, ncas_sub=None,
            nelecas_sub=None, include_core=False, casdm1s_sub=None, **kwargs):
        ''' Get spin-separated, rootspace- and state-averaged 1-RDMs of the active orbitals in the
        AO basis for each fragment in sequence.

        Kwargs:
            mo_coeff: ndarray of shape (nao,nmo)
                Contains MO coefficients
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 
            include_core: logical
                If true, density of inactive orbitals is included on return
            casdm1s_sub: list of ndarray of shape (2, ncas_sub[i], ncas_sub[i])
                Output of make_casdm1s_sub

        Returns:
            dm1fs: list of ndarray of shape (2, nao, nao)
                Spin-separated 1-body reduced density matrices for each fragment in the AO basis,
                rootspace-averaged and locally state-averaged if applicable.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1s_sub is None: casdm1s_sub = self.make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        rdm1s = []
        for idx, casdm1s in enumerate (casdm1s_sub):
            mo = self.get_mo_slice (idx, mo_coeff=mo_coeff)
            moH = mo.conjugate ().T
            rdm1s.append (np.tensordot (mo, np.dot (casdm1s,moH), axes=((1),(1))).transpose(1,0,2))
        if include_core and self.ncore:
            mo_core = mo_coeff[:,:self.ncore]
            moH_core = mo_core.conjugate ().T
            dm_core = mo_core @ moH_core
            rdm1s = [np.stack ([dm_core, dm_core], axis=0)] + rdm1s
        rdm1s = np.stack (rdm1s, axis=0)
        return rdm1s

    def make_rdm1_sub (self, **kwargs):
        ''' Get spin-summed, rootspace- and state-averaged 1-RDMs of the active orbitals in the
        AO basis for each fragment in sequence.

        Kwargs:
            mo_coeff: ndarray of shape (nao,nmo)
                Contains MO coefficients
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 
            include_core: logical
                If true, density of inactive orbitals is included on return
            casdm1fs: list of ndarray of shape (2, ncas_sub[i], ncas_sub[i])
                Output of make_casdm1s_sub

        Returns:
            dm1f: list of ndarray of shape (nao, nao)
                Spin-summed, 1-body reduced density matrices for each fragment in the AO basis,
                rootspace-averaged and locally state-averaged if applicable.
        '''
        return self.make_rdm1s_sub (**kwargs).sum (1)

    def make_rdm1s (self, mo_coeff=None, ncore=None, **kwargs):
        ''' Get spin-separated, rootspace- and locally state-averaged 1-RDMs in the AO basis.

        Kwargs:
            mo_coeff: ndarray of shape (nao,nmo)
                Contains MO coefficients
            ncore: integer
                Number of core orbitals

        Returns:
            dm1s: ndarray of shape (2,nao,nao)
                Rootspace- and locally state-averaged, spin-separated 1-RDM
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        mo = mo_coeff[:,:ncore]
        moH = mo.conjugate ().T
        dm_core = mo @ moH
        dm_cas = self.make_rdm1s_sub (mo_coeff=mo_coeff, **kwargs).sum (0)
        return dm_core[None,:,:] + dm_cas

    def make_rdm1 (self, mo_coeff=None, ci=None, **kwargs):
        ''' Get spin-summed, rootspace- and locally state-averaged 1-RDM in the AO basis.

        Kwargs:
            mo_coeff: ndarray of shape (nao,nmo)
                Contains MO coefficients
            ci: list of list of ndarrays
                Contains CI vectors

        Returns:
            dm1: ndarray of shape (nao,nao)
                Rootspace- and locally state-averaged, spin-summed 1-RDM
        '''
        return self.make_rdm1s (mo_coeff=mo_coeff, ci=ci, **kwargs).sum (0)

    def make_casdm1s (self, ci=None, **kwargs):
        ''' Get spin-separated, rootspace- and state-averaged 1-RDMs of all active
        orbitals collectively.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 

        Returns:
            casdm1s: ndarray of shape (2, ncas, ncas)
                Spin-separated, rootspace- and state-averaged 1-body reduced density matrix for the
                active orbitals.
        '''
        casdm1s_sub = self.make_casdm1s_sub (ci=ci, **kwargs)
        casdm1a = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        casdm1b = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        return np.stack ([casdm1a, casdm1b], axis=0)

    def make_casdm1 (self, ci=None, **kwargs):
        ''' Get spin-summed, rootspace- and state-averaged 1-RDMs of all active
        orbitals collectively.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 

        Returns:
            casdm1s: ndarray of shape (ncas, ncas)
                Spin-summed, rootspace- and state-averaged 1-body reduced density matrix for the
                active orbitals.
        '''
        return self.make_casdm1s (ci=ci, **kwargs).sum (0)

    def state_make_casdm1s(self, ci=None, state=0,  ncas_sub=None, nelecas_sub=None,
        casdm1frs=None, **kwargs):
        ''' Get spin-separated, locally state-averaged 1-RDM of all active orbitals collectively
        for a single rootspace.

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            state: integer
                Index of rootspace to return
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 
            casdm1frs: sequence of ndarrays of shape (nroots, 2, ncas_sub[i], ncas_sub[i])
                output of states_make_casdm1s_sub

        Returns:
            casdm1s: ndarray of shape (2, ncas, ncas)
                Spin-separated 1-body reduced density matrix for the active orbitals for the chosen
                rootspace, locally state-averaged if applicable.
        '''
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        casdm1s = np.stack([np.stack ([linalg.block_diag (*[dm1rs[iroot][ispin] 
                                                            for dm1rs in casdm1frs])
                                        for ispin in (0, 1)], axis=0)
                            for iroot in range (self.nroots)], axis=0)
        return casdm1s[state]
    
    def state_make_casdm2(self, ci=None, state=0, ncas_sub=None, nelecas_sub=None, 
            casdm1frs=None, casdm2fr=None, **kwargs):
        ''' Get spin-summed, locally state-averaged 2-RDM of all active orbitals for a single
        rootspace. The formulas for nonzero elements are (state=r):

        casdm2r[p1,p2,p3,p4] = casdm2fr[p][r,p1,p2,p3,p4]
        casdm2r[p1,p2,q1,q2] = casdm1fr[p][r,p1,p2] * casdm1fr[q][r,q1,q2]
        casdm2r[p1,q2,q1,p2] = -(casdm1frs[p][r,0,p1,p2] * casdm1frs[q][r,0,q1,q2]
                                 + casdm1frs[p][r,1,p1,p2] * casdm1frs[q][r,1,q1,q2])

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            state: integer
                Index of the chosen rootspace
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 
            casdm1frs: sequence of ndarrays of shape [nroots,] + [ncas_sub[i],]*2
                output of states_make_casdm1s_sub
            casdm2fr: sequence of ndarrays of shape [nroots,] + [ncas_sub[i],]*4
                output of states_make_casdm2_sub

        Returns:
            casdm2: ndarray of shape [ncas,]*4
                Spin-summed 2-body reduced density matrices for all active orbitals for the
                rootspace identified by "state", locally state-averaged if applicable.
        '''
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci)
        if casdm2fr is None: casdm2fr = self.states_make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        casdm2 = np.zeros ((ncas,ncas,ncas,ncas))
        # Diagonal 
        for isub, dm2_r in enumerate (casdm2fr):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            casdm2[i:j, i:j, i:j, i:j] = dm2_r[state]
        # Off-diagonal
        for (isub1, dm1s1_r), (isub2, dm1s2_r) in combinations (enumerate (casdm1frs), 2):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            dma1, dmb1 = dm1s1_r[state][0], dm1s1_r[state][1]
            dma2, dmb2 = dm1s2_r[state][0], dm1s2_r[state][1]
            # Coulomb slice
            casdm2[i:j, i:j, k:l, k:l] = np.multiply.outer (dma1+dmb1, dma2+dmb2)
            casdm2[k:l, k:l, i:j, i:j] = casdm2[i:j, i:j, k:l, k:l].transpose (2,3,0,1)
            # Exchange slice
            casdm2[i:j, k:l, k:l, i:j] = -(np.multiply.outer (dma1, dma2)
                                           +np.multiply.outer (dmb1, dmb2)).transpose (0,3,2,1)
            casdm2[k:l, i:j, i:j, k:l] = casdm2[i:j, k:l, k:l, i:j].transpose (1,0,3,2)
        return casdm2 
     
    def make_casdm2s (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm2rs=None, casdm2fs=None, casdm1frs=None, casdm2frs=None,
            **kwargs):

        if casdm2rs is not None:
            return np.einsum ('rsijkl,r->sijkl', casdm2rs, self.weights)
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2fs is None: casdm2fs = self.make_casdm2s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, casdm2frs=casdm2frs)
        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        weights = self.weights
        casdm2s = np.zeros ((3,ncas,ncas,ncas,ncas))

        # Diagonal of aa,ab,bb
        for isub, dm2 in enumerate (casdm2fs):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            for spin in [0,1,2]:
                casdm2s[spin][i:j, i:j, i:j, i:j] = dm2[spin]

        # Off-diagonal
        for (isub1, dm1rs1), (isub2, dm1rs2) in combinations (enumerate (casdm1frs), 2):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            dma1r, dmb1r = dm1rs1[:,0], dm1rs1[:,1]
            dma2r, dmb2r = dm1rs2[:,0], dm1rs2[:,1]
            # Coulomb slice
            casdm2s[0][i:j, i:j, k:l, k:l] = lib.einsum ('r,rij,rkl->ijkl', weights, dma1r, dma2r)
            casdm2s[1][i:j, i:j, k:l, k:l] = lib.einsum ('r,rij,rkl->ijkl', weights, dma1r, dmb2r)
            casdm2s[1][k:l, k:l, i:j, i:j] = lib.einsum ('r,rij,rkl->ijkl', weights, dma2r, dmb1r)
            casdm2s[2][i:j, i:j, k:l, k:l] = lib.einsum ('r,rij,rkl->ijkl', weights, dmb1r, dmb2r)
            for spin in [0,2]:
                casdm2s[spin][k:l, k:l, i:j, i:j] = casdm2s[spin][i:j, i:j, k:l, k:l].transpose (2,3,0,1)
            # Exchange slice
            d2exc_aa = lib.einsum ('rij,rkl->rilkj', dma1r, dma2r)
            d2exc_bb = lib.einsum ('rij,rkl->rilkj', dmb1r, dmb2r)
            casdm2s[0][i:j, k:l, k:l, i:j] -= np.tensordot (weights, d2exc_aa, axes=1)
            casdm2s[2][i:j, k:l, k:l, i:j] -= np.tensordot (weights, d2exc_bb, axes=1)
            for spin in [0,2]:
                # Making it compatible with complex CI.
                casdm2s[spin][k:l, i:j, i:j, k:l] = casdm2s[spin][i:j, k:l, k:l, i:j].conj ().transpose (1,0,3,2)
        return casdm2s


    def make_casdm2 (self, ci=None, ncas_sub=None, nelecas_sub=None, 
            casdm2r=None, casdm2f=None, casdm1frs=None, casdm2fr=None,
            **kwargs):
        ''' Get spin-summed, rootspace- and state-averaged 2-RDM of all active orbitals. The
        formulas for nonzero elements are

        casdm2r[r,p1,p2,p3,p4] = casdm2fr[p][r,p1,p2,p3,p4]
        casdm2r[r,p1,p2,q1,q2] = casdm1fr[p][r,p1,p2] * casdm1fr[q][r,q1,q2]
        casdm2r[r,p1,q2,q1,p2] = -(casdm1frs[p][r,0,p1,p2] * casdm1frs[q][r,0,q1,q2]
                                 + casdm1frs[p][r,1,p1,p2] * casdm1frs[q][r,1,q1,q2])
        casdm2 = sum_r casdm2r[r]

        Kwargs:
            ci: list of list of ndarrays
                Contains CI vectors
            ncas_sub: sequence of integers
                number of orbitals in each fragment
            nelecas_sub: sequence of integers
                number of orbitals in each fragment in the reference rootspace 
            casdm2r: ndarray of shape [nroots,] + [ncas,]*4
                rootspace-separated 2-RDMs of the whole active space
            casdm2f: sequence of ndarrays of shape [ncas_sub[i],]*4
                output of make_casdm2_sub
            casdm1frs: sequence of ndarrays of shape [nroots,] + [ncas_sub[i],]*2
                output of states_make_casdm1s_sub
            casdm2fr: sequence of ndarrays of shape [nroots,] + [ncas_sub[i],]*4
                output of states_make_casdm2_sub

        Returns:
            casdm2: ndarray of shape [ncas,]*4
                Spin-summed 2-body reduced density matrices for all active orbitals, rootspace-
                and state-averaged if applicable.
        '''
        if casdm2r is not None: 
            return np.einsum ('rijkl,r->ijkl', casdm2r, self.weights)
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2f is None: casdm2f = self.make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, casdm2fr=casdm2fr)
        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        weights = self.weights
        dtype = np.result_type(*casdm2f, *casdm1frs)
        casdm2 = np.zeros ((ncas,ncas,ncas,ncas), dtype=dtype)
        # Diagonal 
        for isub, dm2 in enumerate (casdm2f):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            casdm2[i:j, i:j, i:j, i:j] = dm2
        # Off-diagonal
        for (isub1, dm1rs1), (isub2, dm1rs2) in combinations (enumerate (casdm1frs), 2):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            dma1r, dmb1r = dm1rs1[:,0], dm1rs1[:,1]
            dma2r, dmb2r = dm1rs2[:,0], dm1rs2[:,1]
            dm1r = dma1r + dmb1r
            dm2r = dma2r + dmb2r
            # Coulomb slice
            casdm2[i:j, i:j, k:l, k:l] = lib.einsum ('r,rij,rkl->ijkl', weights, dm1r, dm2r)
            casdm2[k:l, k:l, i:j, i:j] = casdm2[i:j, i:j, k:l, k:l].transpose (2,3,0,1)
            # Exchange slice
            d2exc = (lib.einsum ('rij,rkl->rilkj', dma1r, dma2r)
                   + lib.einsum ('rij,rkl->rilkj', dmb1r, dmb2r))
            casdm2[i:j, k:l, k:l, i:j] -= np.tensordot (weights, d2exc, axes=1)
            # Making it compatible with complex CI.
            casdm2[k:l, i:j, i:j, k:l] = casdm2[i:j, k:l, k:l, i:j].conj ().transpose (1,0,3,2)
            # IDU my eqn says [2,3,0,1] --> it's same!
        return casdm2

    def make_casdm3 (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm3r=None, casdm3f=None, casdm2frs=None, casdm3fr=None, casdm2r=None, casdm2f=None,
            casdm1frs=None, casdm2fr=None, **kwargs):
        ''' Make the full-dimensional casdm3 spanning the collective active space '''
        if casdm3r is not None:
            return np.einsum ('rijklmn,r->ijklmn', casdm3r, self.weights)
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2frs is None: casdm2frs = self.states_make_casdm2s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2f is None: casdm2f = self.make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, casdm2fr=casdm2fr)
        if casdm3f is None: casdm3f = self.states_make_casdm3_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)

        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        weights = self.weights
        casdm3 = np.zeros ((ncas,ncas,ncas,ncas,ncas,ncas))

        # Diagonal 
        for isub, dm3 in enumerate (casdm3f):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            casdm3[i:j, i:j, i:j, i:j, i:j, i:j] = dm3

        def _transpose_dm3_(i0, i1, i2, i3, j0, j1, j2, j3, k0, k1, k2, k3):
            d = casdm3[i0:i1, i2:i3, j0:j1, j2:j3, k0:k1, k2:k3]
            casdm3[i0:i1, i2:i3, k0:k1, k2:k3, j0:j1, j2:j3] = d.transpose (0,1,4,5,2,3)
            casdm3[k0:k1, k2:k3, i0:i1, i2:i3, j0:j1, j2:j3] = d.transpose (4,5,0,1,2,3)
            casdm3[k0:k1, k2:k3, j0:j1, j2:j3, i0:i1, i2:i3] = d.transpose (4,5,2,3,0,1)
            casdm3[j0:j1, j2:j3, k0:k1, k2:k3, i0:i1, i2:i3] = d.transpose (2,3,4,5,0,1)
            casdm3[j0:j1, j2:j3, i0:i1, i2:i3, k0:k1, k2:k3] = d.transpose (2,3,0,1,4,5)

        # Off-diagonal
        # First 6 terms - combs of f1+f2+f3: dm1rs1=f1, dm1rs2=f2, dm1rs3=f3
        from itertools import permutations
        for (isub1,dm1rs1),(isub2,dm1rs2),(isub3,dm1rs3) in combinations (enumerate(casdm1frs), 3):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            m = ncas_cum[isub3]
            n = ncas_cum[isub3+1]
            dma1r, dmb1r = dm1rs1[:,0], dm1rs1[:,1]
            dma2r, dmb2r = dm1rs2[:,0], dm1rs2[:,1]
            dma3r, dmb3r = dm1rs3[:,0], dm1rs3[:,1]
            dm1r = dma1r + dmb1r
            dm2r = dma2r + dmb2r
            dm3r = dma3r + dmb3r
            
            # Term 1- 29a
            d1sigma = lib.einsum('r,rij,rkl,rmn->rijklmn',weights,dm1r,dm2r,dm3r)
            casdm3[i:j, i:j, k:l, k:l, m:n, m:n] += np.tensordot(weights, d1sigma, axes=1)
            #lib.einsum ('r,rij,rkl,rmn->ijklmn', weights, dm1r, dm2r, dm3r)
            _transpose_dm3_(i,j, i,j, k,l, k,l, m,n, m,n)
            
            # Term 2- 29b
            d2sigma = (lib.einsum('r,rij,rkl,rmn->rijknml',weights,dm1r,dma2r,dma3r)
                       +lib.einsum('r,rij,rkl,rmn->rijknml',weights,dm1r,dmb2r,dmb3r))
            casdm3[i:j, i:j, k:l, m:n, m:n, k:l] -= np.tensordot(weights, d2sigma, axes=1)
            _transpose_dm3_(i,j, i,j, k,l, m,n, m,n, k,l)
            
            # Term 3- 29c
            d3sigma = (lib.einsum('r,rij,rkl,rmn->rinkjml',weights,dma1r,dma2r,dma3r)
                       +lib.einsum('r,rij,rkl,rmn->rinkjml',weights,dmb1r,dmb2r,dmb3r))
            casdm3[i:j, m:n, k:l, i:j, m:n, k:l] += np.tensordot (weights, d3sigma, axes=1)
            _transpose_dm3_(i,j, m,n, k,l, i,j, m,n, k,l)

            # Term 4- 29d
            d4sigma = (lib.einsum('r,rij,rkl,rmn->rilkjmn',weights,dma1r,dma2r,dm3r)
                       +lib.einsum('r,rij,rkl,rmn->rilkjmn',weights,dmb1r,dmb2r,dm3r))
            casdm3[i:j, k:l, k:l, i:j, m:n, m:n] -= np.tensordot(weights, d4sigma, axes=1)
            _transpose_dm3_(i,j, k,l, k,l, i,j, m,n, m,n)

            # Term 5- 29e
            d5sigma = (lib.einsum('r,rij,rkl,rmn->rinklmj',weights,dma1r,dm2r,dma3r)
                       +lib.einsum('r,rij,rkl,rmn->rinklmj',weights,dmb1r,dm2r,dmb3r))
            casdm3[i:j, m:n, k:l, k:l, m:n, i:j] -= np.tensordot(weights, d5sigma, axes=1)
            _transpose_dm3_(i,j, m,n, k,l, k,l, m,n, i,j)

            # Term 6- 29f
            d6sigma = (lib.einsum('r,rij,rkl,rmn->rilknmj',weights,dma1r,dma2r,dma3r)
                       +lib.einsum('r,rij,rkl,rmn->rilknmj',weights,dmb1r,dmb2r,dmb3r))
            casdm3[i:j, k:l, k:l, m:n, m:n, i:j] += np.tensordot(weights, d6sigma, axes=1)
            _transpose_dm3_(i,j, k,l, k,l, m,n, m,n, i,j)

        #Last 2 terms - combs of f1+f1+f2: dm1rs1=f1, dm2rs2=f2

        for isub1, isub2 in permutations (range (self.nfrags), 2):
            dm2rs2 = casdm2frs[isub1]
            dm1rs1 = casdm1frs[isub2]

            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            
            dma1r, dmb1r = dm1rs1[:,0], dm1rs1[:,1] 
            dm1r = dma1r + dmb1r
            dmaa2r, dmab2r, dmbb2r = dm2rs2[:,0], dm2rs2[:,1], dm2rs2[:,2]

            # Term 7 - 29g
            d4sigma = (lib.einsum('r,rmn,rijkl->rijklmn',weights,dm1r,dmaa2r)
                       +lib.einsum('r,rmn,rijkl->rijklmn',weights,dm1r,dmab2r)
                       +lib.einsum('r,rmn,rijkl->rijklmn',weights,dm1r,dmab2r)
                       +lib.einsum('r,rmn,rijkl->rijklmn',weights,dm1r,dmbb2r))
            casdm3[i:j, i:j, i:j, i:j, k:l, k:l] += np.tensordot (weights, d4sigma, axes=1)
            d = casdm3[i:j, i:j, i:j, i:j, k:l, k:l]
            casdm3[i:j, i:j, k:l, k:l, i:j, i:j] = d.transpose(0,1,4,5,2,3)
            casdm3[k:l, k:l, i:j, i:j, i:j, i:j] = d.transpose(4,5,0,1,2,3)

            # Term 8 - 29h
            d5sigma = (lib.einsum('r,rmn,rijkl->rijknml',weights,dma1r,dmaa2r)
                       +lib.einsum('r,rmn,rijkl->rijknml',weights,dmb1r,dmab2r)
                       +lib.einsum('r,rmn,rijkl->rijknml',weights,dma1r,dmab2r)
                       +lib.einsum('r,rmn,rijkl->rijknml',weights,dmb1r,dmbb2r))
            casdm3[i:j, i:j, i:j, k:l, k:l, i:j] -= np.tensordot (weights, d5sigma, axes=1)
            _transpose_dm3_(i,j, i,j, i,j, k,l, k,l, i,j)

        return casdm3


    def make_casdm3s (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm3r=None, casdm3f=None, casdm2frs=None, casdm3fr=None, casdm2r=None, casdm2f=None,
            casdm1frs=None, casdm2fr=None, casdm3fs= None, **kwargs):
        ''' Make the full-dimensional casdm3 spanning the collective active space '''
        if casdm3r is not None:
            return np.einsum ('rijklmn,r->ijklmn', casdm3r, self.weights)
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2frs is None: casdm2frs = self.states_make_casdm2s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2f is None: casdm2f = self.make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, casdm2fr=casdm2fr)
        if casdm3f is None: casdm3f = self.states_make_casdm3_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm3fs is None: casdm3fs = self.make_casdm3s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)

        ncas = sum (ncas_sub)
        ncas_cum = np.cumsum ([0] + ncas_sub.tolist ())
        weights = self.weights
        casdm3s = np.zeros ((4,ncas,ncas,ncas,ncas,ncas,ncas))

        # Diagonal 
        for isub, dm3 in enumerate (casdm3fs):
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            for spin in range(4):
                casdm3s[spin][i:j, i:j, i:j, i:j, i:j, i:j] = dm3[spin]

        def _transpose_dm3_(dmt, i0, i1, i2, i3, j0, j1, j2, j3, k0, k1, k2, k3):
            d = dmt[i0:i1, i2:i3, j0:j1, j2:j3, k0:k1, k2:k3]
            dmt[i0:i1, i2:i3, k0:k1, k2:k3, j0:j1, j2:j3] = d.transpose (0,1,4,5,2,3)
            dmt[k0:k1, k2:k3, i0:i1, i2:i3, j0:j1, j2:j3] = d.transpose (4,5,0,1,2,3)
            dmt[k0:k1, k2:k3, j0:j1, j2:j3, i0:i1, i2:i3] = d.transpose (4,5,2,3,0,1)
            dmt[j0:j1, j2:j3, k0:k1, k2:k3, i0:i1, i2:i3] = d.transpose (2,3,4,5,0,1)
            dmt[j0:j1, j2:j3, i0:i1, i2:i3, k0:k1, k2:k3] = d.transpose (2,3,0,1,4,5)

        
        # Off-diagonal
        # First 6 terms - combs of f1+f2+f3: dm1rs1=f1, dm1rs2=f2, dm1rs3=f3
        from itertools import permutations
        for (isub1,dm1rs1),(isub2,dm1rs2),(isub3,dm1rs3) in combinations (enumerate(casdm1frs), 3):
            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]
            m = ncas_cum[isub3]
            n = ncas_cum[isub3+1]
            dma1r, dmb1r = dm1rs1[:,0], dm1rs1[:,1]
            dma2r, dmb2r = dm1rs2[:,0], dm1rs2[:,1]
            dma3r, dmb3r = dm1rs3[:,0], dm1rs3[:,1]
            dm1r = dma1r + dmb1r
            dm2r = dma2r + dmb2r
            dm3r = dma3r + dmb3r

            # Term 1- 29a
            d1sigma0 = lib.einsum('r,rij,rkl,rmn->rijklmn',weights,dma1r,dma2r,dma3r)
            casdm3s[0][i:j, i:j, k:l, k:l, m:n, m:n] += np.tensordot(weights, d1sigma0, axes=1)
            _transpose_dm3_(casdm3s[0],i,j, i,j, k,l, k,l, m,n, m,n)

            d1sigma11 = lib.einsum('r,rij,rkl,rmn->rijklmn',weights,dma1r,dma2r,dmb3r)
            d1sigma12 = lib.einsum('r,rij,rkl,rmn->rijmnkl',weights,dma1r,dmb2r,dma3r)
            d1sigma13 = lib.einsum('r,rij,rkl,rmn->rmnijkl',weights,dma1r,dmb2r,dma3r)
            d1sigma14 = lib.einsum('r,rij,rkl,rmn->rmnklij',weights,dmb1r,dma2r,dma3r)
            d1sigma15 = lib.einsum('r,rij,rkl,rmn->rklmnij',weights,dmb1r,dma2r,dma3r)
            d1sigma16 = lib.einsum('r,rij,rkl,rmn->rklijmn',weights,dma1r,dma2r,dmb3r)

            d1sigma21 = lib.einsum('r,rij,rkl,rmn->rijklmn',weights,dma1r,dmb2r,dmb3r)
            d1sigma22 = lib.einsum('r,rij,rkl,rmn->rijmnkl',weights,dma1r,dmb2r,dmb3r)
            d1sigma23 = lib.einsum('r,rij,rkl,rmn->rmnijkl',weights,dmb1r,dmb2r,dma3r)
            d1sigma24 = lib.einsum('r,rij,rkl,rmn->rmnklij',weights,dmb1r,dmb2r,dma3r)
            d1sigma25 = lib.einsum('r,rij,rkl,rmn->rklmnij',weights,dmb1r,dma2r,dmb3r)
            d1sigma26 = lib.einsum('r,rij,rkl,rmn->rklijmn',weights,dmb1r,dma2r,dmb3r)

            d1sigma3 = lib.einsum('r,rij,rkl,rmn->rijklmn',weights,dmb1r,dmb2r,dmb3r)
            casdm3s[3][i:j, i:j, k:l, k:l, m:n, m:n] += np.tensordot(weights, d1sigma3, axes=1)
            _transpose_dm3_(casdm3s[3],i,j, i,j, k,l, k,l, m,n, m,n)

            casdm3s[1][i:j, i:j, k:l, k:l, m:n, m:n] += np.tensordot(weights, d1sigma11, axes=1)
            casdm3s[1][i:j, i:j, m:n, m:n, k:l, k:l] += np.tensordot(weights, d1sigma12, axes=1)
            casdm3s[1][m:n, m:n, i:j, i:j, k:l, k:l] += np.tensordot(weights, d1sigma13, axes=1)
            casdm3s[1][m:n, m:n, k:l, k:l, i:j, i:j] += np.tensordot(weights, d1sigma14, axes=1)
            casdm3s[1][k:l, k:l, m:n, m:n, i:j, i:j] += np.tensordot(weights, d1sigma15, axes=1)
            casdm3s[1][k:l, k:l, i:j, i:j, m:n, m:n] += np.tensordot(weights, d1sigma16, axes=1)

            casdm3s[2][i:j, i:j, k:l, k:l, m:n, m:n] += np.tensordot(weights, d1sigma21, axes=1)
            casdm3s[2][i:j, i:j, m:n, m:n, k:l, k:l] += np.tensordot(weights, d1sigma22, axes=1)
            casdm3s[2][m:n, m:n, i:j, i:j, k:l, k:l] += np.tensordot(weights, d1sigma23, axes=1)
            casdm3s[2][m:n, m:n, k:l, k:l, i:j, i:j] += np.tensordot(weights, d1sigma24, axes=1)
            casdm3s[2][k:l, k:l, m:n, m:n, i:j, i:j] += np.tensordot(weights, d1sigma25, axes=1)
            casdm3s[2][k:l, k:l, i:j, i:j, m:n, m:n] += np.tensordot(weights, d1sigma26, axes=1)

            # Term 2- 29b
            d2sigma0 = lib.einsum('r,rij,rkl,rmn->rijknml',weights,dma1r,dma2r,dma3r)
            casdm3s[0][i:j, i:j, k:l, m:n, m:n, k:l] -= np.tensordot(weights, d2sigma0, axes=1)
            _transpose_dm3_(casdm3s[0], i,j, i,j, k,l, m,n, m,n, k,l)

            d2sigma14 = lib.einsum('r,rij,rkl,rmn->rmlknij',weights,dmb1r,dma2r,dma3r)
            d2sigma15 = lib.einsum('r,rij,rkl,rmn->rknmlij',weights,dmb1r,dma2r,dma3r)
            casdm3s[1][m:n, k:l, k:l, m:n, i:j, i:j] -= np.tensordot(weights, d2sigma14, axes=1)
            casdm3s[1][k:l, m:n, m:n, k:l, i:j, i:j] -= np.tensordot(weights, d2sigma15, axes=1)

            d2sigma21 = lib.einsum('r,rij,rkl,rmn->rijknml',weights,dma1r,dmb2r,dmb3r)
            d2sigma22 = lib.einsum('r,rij,rkl,rmn->rijmlkn',weights,dma1r,dmb2r,dmb3r)
            casdm3s[2][i:j, i:j, k:l, m:n, m:n, k:l] -= np.tensordot(weights, d2sigma21, axes=1)
            casdm3s[2][i:j, i:j, m:n, k:l, k:l, m:n] -= np.tensordot(weights, d2sigma22, axes=1)

            d2sigma3 = lib.einsum('r,rij,rkl,rmn->rijknml',weights,dmb1r,dmb2r,dmb3r)
            casdm3s[3][i:j, i:j, k:l, m:n, m:n, k:l] -= np.tensordot(weights, d2sigma3, axes=1)
            _transpose_dm3_(casdm3s[3], i,j, i,j, k,l, m,n, m,n, k,l)

            # Term 3- 29c
            d3sigma0 = lib.einsum('r,rij,rkl,rmn->rilkjmn',weights,dma1r,dma2r,dma3r)
            casdm3s[0][i:j, k:l, k:l, i:j, m:n, m:n] -= np.tensordot (weights, d3sigma0, axes=1)
            _transpose_dm3_(casdm3s[0], i,j, k,l, k,l, i,j, m,n, m,n)

            d3sigma11 = lib.einsum('r,rij,rkl,rmn->rilkjmn',weights,dma1r,dma2r,dmb3r)
            d3sigma16 = lib.einsum('r,rij,rkl,rmn->rkjilmn',weights,dma1r,dma2r,dmb3r)
            casdm3s[1][i:j, k:l, k:l, i:j, m:n, m:n] -= np.tensordot (weights, d3sigma11, axes=1)
            casdm3s[1][k:l, i:j, i:j, k:l, m:n, m:n] -= np.tensordot (weights, d3sigma16, axes=1)

            d3sigma23 = lib.einsum('r,rij,rkl,rmn->rmnilkj',weights,dmb1r,dmb2r,dma3r)
            d3sigma24 = lib.einsum('r,rij,rkl,rmn->rmnkjil',weights,dmb1r,dmb2r,dma3r)
            casdm3s[2][m:n, m:n, i:j, k:l, k:l, i:j] -= np.tensordot (weights, d3sigma23, axes=1)
            casdm3s[2][m:n, m:n, k:l, i:j, i:j, k:l] -= np.tensordot (weights, d3sigma24, axes=1)

            d3sigma3 = lib.einsum('r,rij,rkl,rmn->rilkjmn',weights,dmb1r,dmb2r,dmb3r)
            casdm3s[3][i:j, k:l, k:l, i:j, m:n, m:n] -= np.tensordot (weights, d3sigma3, axes=1)
            _transpose_dm3_(casdm3s[3], i,j, k,l, k,l, i,j, m,n, m,n)

            # Term 4- 29d
            d4sigma0 = lib.einsum('r,rij,rkl,rmn->rilknmj',weights,dma1r,dma2r,dma3r)
            casdm3s[0][i:j, k:l, k:l, m:n, m:n, i:j] += np.tensordot(weights, d4sigma0, axes=1)
            _transpose_dm3_(casdm3s[0], i,j, k,l, k,l, m,n, m,n, i,j)

            d4sigma3 = lib.einsum('r,rij,rkl,rmn->rilknmj',weights,dmb1r,dmb2r,dmb3r)
            casdm3s[3][i:j, k:l, k:l, m:n, m:n, i:j] += np.tensordot(weights, d4sigma3, axes=1)
            _transpose_dm3_(casdm3s[3], i,j, k,l, k,l, m,n, m,n, i,j)

            # Term 5- 29e
            d5sigma0 = lib.einsum('r,rij,rkl,rmn->rinklmj',weights,dma1r,dma2r,dma3r)
            casdm3s[0][i:j, m:n, k:l, k:l, m:n, i:j] -= np.tensordot(weights, d5sigma0, axes=1)
            _transpose_dm3_(casdm3s[0], i,j, m,n, k,l, k,l, m,n, i,j)

            d5sigma12 = lib.einsum('r,rij,rkl,rmn->rinmjkl',weights,dma1r,dmb2r,dma3r)
            d5sigma13 = lib.einsum('r,rij,rkl,rmn->rmjinkl',weights,dma1r,dmb2r,dma3r)
            casdm3s[1][i:j, m:n, m:n, i:j, k:l, k:l] -= np.tensordot(weights, d5sigma12, axes=1)
            casdm3s[1][m:n, i:j, i:j, m:n, k:l, k:l] -= np.tensordot(weights, d5sigma13, axes=1)

            d5sigma25 = lib.einsum('r,rij,rkl,rmn->rklmjin',weights,dmb1r,dma2r,dmb3r)
            d5sigma26 = lib.einsum('r,rij,rkl,rmn->rklinmj',weights,dmb1r,dma2r,dmb3r)
            casdm3s[2][k:l, k:l, m:n, i:j, i:j, m:n] -= np.tensordot(weights, d5sigma25, axes=1)
            casdm3s[2][k:l, k:l, i:j, m:n, m:n, i:j] -= np.tensordot(weights, d5sigma26, axes=1)

            d5sigma3 = lib.einsum('r,rij,rkl,rmn->rinklmj',weights,dmb1r,dmb2r,dmb3r)
            casdm3s[3][i:j, m:n, k:l, k:l, m:n, i:j] -= np.tensordot(weights, d5sigma3, axes=1)
            _transpose_dm3_(casdm3s[3], i,j, m,n, k,l, k,l, m,n, i,j)

            # Term 6- 29f
            d6sigma0 = lib.einsum('r,rij,rkl,rmn->rinkjml',weights,dma1r,dma2r,dma3r)
            casdm3s[0][i:j, m:n, k:l, i:j, m:n, k:l] += np.tensordot(weights, d6sigma0, axes=1)
            _transpose_dm3_(casdm3s[0], i,j, m,n, k,l, i,j, m,n, k,l)

            d6sigma3 = lib.einsum('r,rij,rkl,rmn->rinkjml',weights,dmb1r,dmb2r,dmb3r)
            casdm3s[3][i:j, m:n, k:l, i:j, m:n, k:l] += np.tensordot(weights, d6sigma3, axes=1)
            _transpose_dm3_(casdm3s[3], i,j, m,n, k,l, i,j, m,n, k,l)

        
        #Last 2 terms - combs of f1+f1+f2: dm1rs1=f1, dm2rs2=f2

        for isub1, isub2 in permutations (range (self.nfrags), 2):
            dm2rs2 = casdm2frs[isub1]
            dm1rs1 = casdm1frs[isub2]

            i = ncas_cum[isub1]
            j = ncas_cum[isub1+1]
            k = ncas_cum[isub2]
            l = ncas_cum[isub2+1]

            dma1r, dmb1r = dm1rs1[:,0], dm1rs1[:,1]
            dmaa2r, dmab2r, dmbb2r = dm2rs2[:,0], dm2rs2[:,1], dm2rs2[:,2]
        
            # Term 7 - 29g
            d4sigma0 = lib.einsum('r,rmn,rijkl->rijklmn',weights,dma1r,dmaa2r)

            d4sigma11 = lib.einsum('r,rmn,rijkl->rijklmn',weights,dmb1r,dmaa2r)
            d4sigma12 = lib.einsum('r,rmn,rijkl->rijmnkl',weights,dma1r,dmab2r)
            d4sigma13 = lib.einsum('r,rmn,rijkl->rmnijkl',weights,dma1r,dmab2r)
            d4sigma14 = lib.einsum('r,rmn,rklij->rmnklij',weights,dma1r,dmab2r)
            d4sigma15 = lib.einsum('r,rmn,rklij->rklmnij',weights,dma1r,dmab2r)
            d4sigma16 = lib.einsum('r,rmn,rklij->rklijmn',weights,dmb1r,dmaa2r)
            
            d4sigma21 = lib.einsum('r,rmn,rijkl->rijklmn',weights,dmb1r,dmab2r)
            d4sigma22 = lib.einsum('r,rmn,rijkl->rijmnkl',weights,dmb1r,dmab2r)
            d4sigma23 = lib.einsum('r,rmn,rijkl->rmnijkl',weights,dma1r,dmbb2r)
            d4sigma24 = lib.einsum('r,rmn,rklij->rmnklij',weights,dma1r,dmbb2r)
            d4sigma25 = lib.einsum('r,rmn,rklij->rklmnij',weights,dmb1r,dmab2r)
            d4sigma26 = lib.einsum('r,rmn,rklij->rklijmn',weights,dmb1r,dmab2r)

            d4sigma3 = lib.einsum('r,rmn,rijkl->rijklmn',weights,dmb1r,dmbb2r)

            casdm3s[0][i:j, i:j, i:j, i:j, k:l, k:l] += np.tensordot (weights, d4sigma0, axes=1)
            d0 = casdm3s[0][i:j, i:j, i:j, i:j, k:l, k:l]
            casdm3s[0][i:j, i:j, k:l, k:l, i:j, i:j] = d0.transpose(0,1,4,5,2,3)
            casdm3s[0][k:l, k:l, i:j, i:j, i:j, i:j] = d0.transpose(4,5,0,1,2,3)

            casdm3s[1][i:j, i:j, i:j, i:j, k:l, k:l] += np.tensordot (weights, d4sigma11, axes=1)
            casdm3s[1][i:j, i:j, k:l, k:l, i:j, i:j] += np.tensordot (weights, d4sigma12, axes=1)
            casdm3s[1][k:l, k:l, i:j, i:j, i:j, i:j] += np.tensordot (weights, d4sigma13, axes=1)

            casdm3s[2][i:j, i:j, i:j, i:j, k:l, k:l] += np.tensordot (weights, d4sigma21, axes=1)
            casdm3s[2][i:j, i:j, k:l, k:l, i:j, i:j] += np.tensordot (weights, d4sigma22, axes=1)
            casdm3s[2][k:l, k:l, i:j, i:j, i:j, i:j] += np.tensordot (weights, d4sigma23, axes=1)

            casdm3s[3][i:j, i:j, i:j, i:j, k:l, k:l] += np.tensordot (weights, d4sigma3, axes=1)
            d3 = casdm3s[3][i:j, i:j, i:j, i:j, k:l, k:l]
            casdm3s[3][i:j, i:j, k:l, k:l, i:j, i:j] = d3.transpose(0,1,4,5,2,3)
            casdm3s[3][k:l, k:l, i:j, i:j, i:j, i:j] = d3.transpose(4,5,0,1,2,3)

            # Term 8 - 29h
            d5sigma0 = lib.einsum('r,rmn,rijkl->rijknml',weights,dma1r,dmaa2r)
            
            d5sigma14 = lib.einsum('r,rmn,rklij->rmlknij',weights,dma1r,dmab2r)
            d5sigma15 = lib.einsum('r,rmn,rklij->rknmlij',weights,dma1r,dmab2r)

            d5sigma21 = lib.einsum('r,rmn,rijkl->rijknml',weights,dmb1r,dmab2r)
            d5sigma22 = lib.einsum('r,rmn,rijkl->rijmlkn',weights,dmb1r,dmab2r)

            d5sigma3 = lib.einsum('r,rmn,rijkl->rijknml',weights,dmb1r,dmbb2r)

            casdm3s[0][i:j, i:j, i:j, k:l, k:l, i:j] -= np.tensordot (weights, d5sigma0, axes=1)
            _transpose_dm3_(casdm3s[0],i,j, i,j, i,j, k,l, k,l, i,j)

            casdm3s[1][k:l, i:j, i:j, k:l, i:j, i:j] -= np.tensordot (weights, d5sigma14, axes=1)
            casdm3s[1][i:j, k:l, k:l, i:j, i:j, i:j] -= np.tensordot (weights, d5sigma15, axes=1)
            
            casdm3s[2][i:j, i:j, i:j, k:l, k:l, i:j] -= np.tensordot (weights, d5sigma21, axes=1)
            casdm3s[2][i:j, i:j, k:l, i:j, i:j, k:l] -= np.tensordot (weights, d5sigma22, axes=1)

            casdm3s[3][i:j, i:j, i:j, k:l, k:l, i:j] -= np.tensordot (weights, d5sigma3, axes=1)
            _transpose_dm3_(casdm3s[3],i,j, i,j, i,j, k,l, k,l, i,j)
        
        return casdm3s

    def get_veff (self, mol=None, dm=None, hermi=1, dm1s=None, **kwargs):
        '''Effective potential matrix from a density matrix 

        Kwargs:
            mol : instance of :class:`Mole`
            dm : one or two ndarrays of shape (nao,nao)
                If two, they are interpreted as spin-up and spin-down and a spin-separated
                potential is returned. If one, it is interpreted as spin-summed and a spinless
                potential is returned.
            hermi : int
                Whether J, K matrix is hermitian

                | 0 : no hermitian or symmetric
                | 1 : hermitian
                | 2 : anti-hermitian

        Returns:
            veff : ndarray of shape (nao,nao) or (2,nao,nao)
                Depending on shape of np.asarray (dm)
        '''
        if dm is None and dm1s is not None:
            import warnings
            kwarg_warn = "The kwarg on get_veff has been renamed from dm1s to dm"
            warnings.warn(kwarg_warn, stacklevel=3)
            dm = dm1s
        if mol is None: mol = self.mol
        nao = mol.nao_nr ()
        if dm is None: dm = self.make_rdm1 (include_core=True, **kwargs).reshape (nao, nao)
        dm = np.asarray (dm)
        if isinstance (self, _DFLASCI):
            vj, vk = self.with_df.get_jk(dm, hermi=hermi)
        else:
            vj, vk = self._scf.get_jk(mol, dm, hermi=hermi)
        if dm.ndim==3:
            assert (dm.shape[0] == 2)
            return vj.sum (0)[None,:,:] - vk
        elif dm.ndim==2:
            veff = np.stack ([j - k/2 for j, k in zip (vj, vk)], axis=0)
            return veff
        else:
            raise RuntimeError ("dm must have 2 or 3 dimensions")

    def states_energy_elec (self, mo_coeff=None, ncore=None, ncas=None,
            ncas_sub=None, nelecas_sub=None, ci=None, h2eff=None, veff=None, 
            casdm1frs=None, casdm2fr=None, veff_core=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        if ncas is None: ncas = self.ncas
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None: ci = self.ci
        if h2eff is None: h2eff = self.get_h2eff (mo_coeff)
        if casdm1frs is None: casdm1frs = self.states_make_casdm1s_sub (ci=ci, ncas_sub=ncas_sub,
                                                                        nelecas_sub=nelecas_sub)
        if casdm2fr is None: casdm2fr = self.states_make_casdm2_sub (ci=ci, ncas_sub=ncas_sub,
                                                                     nelecas_sub=nelecas_sub)
        nao, nmo = mo_coeff.shape
        nocc = ncore + ncas
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:nocc]
        dm_core = 2*mo_core @ mo_core.conj ().T
        if veff_core is None: veff_core = getattr (veff, 'c', None)
        if veff_core is None: veff_core = self.get_veff (dm=dm_core)
        h1eff = self.get_hcore () + veff_core
        e0 = 2*np.dot (((h1eff-(veff_core/2)) @ mo_core).ravel (), mo_core.conj().ravel ())
        h1eff = mo_cas.conj ().T @ h1eff @ mo_cas
        eri_cas = lib.numpy_helper.unpack_tril (h2eff.reshape (nmo*ncas, ncas*(ncas+1)//2))
        eri_cas = eri_cas.reshape (nmo, ncas, ncas, ncas)
        eri_cas = eri_cas[ncore:nocc]
        casdm1rs = self.states_make_casdm1s (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub,
                                             casdm1frs=casdm1frs)
        vj_r = np.tensordot (casdm1rs.sum (1), eri_cas, axes=2)
        vk_rs = np.tensordot (casdm1rs, eri_cas, axes=((2,3),(2,1)))
        veff_rs = vj_r[:,None,:,:] - vk_rs

        energy_elec = []
        for idx, (dm1s, v) in enumerate (zip (casdm1rs, veff_rs)):
            casdm1fs = [dm[idx] for dm in casdm1frs]
            casdm2f = [dm[idx] for dm in casdm2fr]
            
            # 1-body veff terms
            h1e = h1eff[None,:,:] + v/2
            e1 = np.dot (h1e.ravel (), dm1s.ravel ())

            # 2-body cumulant terms
            e2 = 0
            for isub, (dm1s, dm2) in enumerate (zip (casdm1fs, casdm2f)):
                dm1a, dm1b = dm1s[0], dm1s[1]
                dm1 = dm1a + dm1b
                cdm2 = dm2 - np.multiply.outer (dm1, dm1)
                cdm2 += np.multiply.outer (dm1a, dm1a).transpose (0,3,2,1)
                cdm2 += np.multiply.outer (dm1b, dm1b).transpose (0,3,2,1)
                eri = self.get_h2eff_slice (h2eff, isub)
                te2 = np.tensordot (eri, cdm2, axes=4) / 2
                e2 += te2
            energy_elec.append (e0 + e1 + e2)
            self._e1_ref = e0 + e1
            self._e2_ref = e2

        return energy_elec

    def energy_elec (self, mo_coeff=None, ncore=None, ncas=None,
            ncas_sub=None, nelecas_sub=None, ci=None, h2eff=None, veff=None,
            casdm1frs=None, casdm2fr=None, **kwargs):
        ''' Since the LASCI energy cannot be calculated as simply as ecas + ecore, I need this '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        if ncas is None: ncas = self.ncas
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None: ci = self.ci
        if h2eff is None: h2eff = self.get_h2eff (mo_coeff)
        casdm1s_sub = self.make_casdm1s_sub (ci=ci, ncas_sub=ncas_sub, nelecas_sub=nelecas_sub,
                                             casdm1frs=casdm1frs)
        if veff is None:
            veff = self.get_veff (dm = self.make_rdm1s (mo_coeff=mo_coeff,casdm1s_sub=casdm1s_sub))

        # 1-body veff terms
        h1e = self.get_hcore ()[None,:,:] + veff/2
        dm1s = self.make_rdm1s (mo_coeff=mo_coeff, ncore=ncore, ncas_sub=ncas_sub,
            nelecas_sub=nelecas_sub, casdm1s_sub=casdm1s_sub)
        e1 = np.dot (h1e.ravel (), dm1s.ravel ())

        # 2-body cumulant terms
        casdm1s = self.make_casdm1s (ci=ci, ncas_sub=ncas_sub, 
            nelecas_sub=nelecas_sub, casdm1frs=casdm1frs)
        casdm1 = casdm1s.sum (0)
        casdm2 = self.make_casdm2 (ci=ci, ncas_sub=ncas_sub,
            nelecas_sub=nelecas_sub, casdm1frs=casdm1frs, casdm2fr=casdm2fr)
        casdm2 -= np.multiply.outer (casdm1, casdm1)
        casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
        casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
        ncore, ncas, nocc = self.ncore, self.ncas, self.ncore + self.ncas
        eri = lib.numpy_helper.unpack_tril (h2eff[ncore:nocc].reshape (ncas*ncas, -1))
        eri = eri.reshape ([ncas,]*4)
        e2 = np.tensordot (eri, casdm2, axes=4)/2

        e0 = self.energy_nuc ()
        self._e1_test = e1
        self._e2_test = e2
        return e1 + e2

    def cderi_ao2mo (self, mo_i, mo_j, compact=False):
        assert (isinstance (self, _DFLASCI))
        nmo_i, nmo_j = mo_i.shape[-1], mo_j.shape[-1]
        if compact:
            assert (nmo_i == nmo_j)
            bPij = np.empty ((self.with_df.get_naoaux (), nmo_i*(nmo_i+1)//2), dtype=mo_i.dtype)
        else:
            bPij = np.empty ((self.with_df.get_naoaux (), nmo_i, nmo_j), dtype=mo_i.dtype)
        ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos (mo_i, mo_j, compact=compact)
        b0 = 0
        for eri1 in self.with_df.loop ():
            b1 = b0 + eri1.shape[0]
            eri2 = bPij[b0:b1]
            eri2 = ao2mo._ao2mo.nr_e2 (eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=eri2)
            b0 = b1
        return bPij

    @lib.with_doc(kernel.__doc__)
    def kernel (self, mo_coeff=None, ci0=None, lroots=None, lweights=None, verbose=None,
               assert_no_dupes=False, _dry_run=False):
        if mo_coeff is None:
            mo_coeff=self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None: ci0 = self.ci
        if verbose is None: verbose = self.verbose
        converged, e_tot, e_states, e_cas, e_lexc, ci = kernel (
            self, mo_coeff=mo_coeff, ci0=ci0, lroots=lroots, lweights=lweights,
            verbose=verbose, assert_no_dupes=assert_no_dupes, _dry_run=_dry_run)
        if _dry_run: return
        self.converged, self.ci = converged, ci
        self.e_tot, self.e_states, self.e_cas, self.e_lexc = e_tot, e_states, e_cas, e_lexc
        if mo_coeff is self.mo_coeff:
            self.dump_chk ()
        elif getattr (self, 'chkfile', None) is not None:
            lib.logger.warn (self, 'orbitals changed; chkfile not dumped!')
        self._finalize (method='LASCI')
        return self.converged, self.e_tot, self.e_states, self.e_cas, e_lexc, self.ci

    state_average = state_average
    state_average_ = state_average_
    get_single_state_las = get_single_state_las


    las2cas_civec = las2cas_civec
    assert_no_duplicates = assert_no_duplicates
    get_init_guess_ci = get_init_guess_ci
    _combine_init_guess_ci = _combine_init_guess_ci
    localize_init_guess=lasscf_guess.localize_init_guess

    def _svd (self, mo_lspace, mo_rspace, s=None, mo_occ=None, rngs=None, **kwargs):
        log = lib.logger.new_logger (self, self.verbose)
        # if mo_occ is nontrivial, then I don't care about the lvecs
        if mo_occ is None: mo_occ = np.ones (mo_rspace.shape[1], dtype=int)
        if rngs is None: rngs = [None,None,None]
        mo_occ1 = []
        mo_occ2 = []
        svals = []
        mo_rvecs = []
        mo_rnull = []
        for m in np.unique (mo_occ):
            idx = (mo_occ==m)
            l, sv, r = self._svd1 (mo_lspace, mo_rspace[:,idx], s=s, rng=rngs[int(round(m))],
                                   **kwargs)
            k = min (len (sv), np.count_nonzero (idx))
            assert (l.shape == mo_lspace.shape), '{} {} {} {}'.format (m, l.shape, mo_lspace.shape, k)
            mo_lvecs = l
            svals.append (sv[:k])
            mo_rvecs.append (r[:,:k])
            mo_occ1.append (np.asarray ([m,]*k))
            if k < np.count_nonzero (idx):
                assert (r.shape[1] == np.count_nonzero (idx))
                mo_rnull.append (r[:,k:])
                mo_occ2.append (np.asarray ([m,]*(np.count_nonzero (idx) - k)))
        mo_rvecs = np.concatenate (mo_rvecs + mo_rnull, axis=1)
        svals = np.concatenate (svals)
        mo_occ1 = np.concatenate (mo_occ1 + mo_occ2)
        idx = np.argsort (-svals)
        svals = svals[idx]
        k = len (idx)
        mo_occ1[:k] = mo_occ1[:k][idx]
        # k1 = len (mo_lvecs[:,:k])
        k1 = mo_lvecs[:,:k].shape[1]
        if k1 != k:
            log.warn ("Fewer AOs than MOs! Did you indicate the correct atoms?")
        mo_lvecs[:,:k] = mo_lvecs[:,:k][:,idx[:k1]]
        mo_rvecs[:,:k] = mo_rvecs[:,:k][:,idx]
        return mo_lvecs, svals, mo_rvecs, mo_occ1

    def _svd1 (self, mo_lspace, mo_rspace, s=None, rng=None, **kwargs):
        if s is None: s = self._scf.get_ovlp ()
        l, sv, r = _svd_subspaces (s, mo_lspace, mo_rspace)
        return _shift_svals (l, sv, r, rng)

    def dump_flags (self, verbose=None, _method_name='CI'):
        log = lib.logger.new_logger (self, verbose)
        log.info ('')
        log.info ('******** %s flags ********', _method_name)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        nfrags = len (self.nelecas_sub)
        log.info ('CAS (%de+%de, %do), ncore = %d, nvir = %d',
                  self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        log.info ('Divided into %d LAS spaces', nfrags)
        for i, (no, ne) in enumerate (zip (self.ncas_sub, self.nelecas_sub)):
            log.info ('LAS %d : (%de+%de, %do)', i, ne[0], ne[1], no)
        log.info ('nroots = %d', self.nroots)
        log.info ('max_cycle_macro = %d', self.max_cycle_macro)
        log.info ('max_cycle_micro = %d', self.max_cycle_micro)
        log.info ('conv_tol_grad = %s', self.conv_tol_grad)
        log.info ('max_memory %d MB (current use %d MB)', self.max_memory,
                  lib.current_memory()[0])
        for i, fcibox in enumerate (self.fciboxes):
            if getattr (fcibox, 'dump_flags', None):
                log.info ('fragment %d FCI solver flags:', i)
                fcibox.dump_flags (log.verbose)

    @property
    def converged (self):
        return all (self.states_converged)
    @converged.setter
    def converged (self, x):
        if hasattr (x, '__len__'):
            self.states_converged = list (x)
        else:
            self.states_converged = [x,]*self.nroots

    def dump_spaces (self, nroots=None, sort_energy=False):
        log = lib.logger.new_logger (self, self.verbose)
        log.info ("******** LAS space tables ********")
        ci = self.ci
        if nroots is None and self.verbose <= lib.logger.INFO:
            nroots = min (self.nroots, 100)
        elif nroots is None:
            nroots = self.nroots
        if nroots < self.nroots:
            log.warn ("Dumping only 100 of %d spaces", self.nroots)
            log.warn ("To see more, explicitly pass nroots to dump_spaces or increase verbosity")
        if sort_energy:
            idx = np.argsort (self.e_states)
        else:
            idx = range (nroots)
        for state in idx:
            neleca_f = []
            nelecb_f = []
            wfnsym_f = []
            wfnsym = 0
            m_f = []
            s_f = []
            lroots = []
            s2_tot = 0
            for ifrag, (fcibox, nelecas) in enumerate (zip (self.fciboxes, self.nelecas_sub)):
                solver = fcibox.fcisolvers[state]
                na, nb = _unpack_nelec (fcibox._get_nelec (solver, nelecas))
                neleca_f.append (na)
                nelecb_f.append (nb)
                m_f.append ((na-nb)/2)
                s_f.append ((solver.smult-1)/2)
                s2_tot += s_f[-1] * (s_f[-1] + 1)
                fragsym = getattr (solver, 'wfnsym', 0) or 0
                if isinstance (fragsym, str):
                    fragsym_str = fragsym
                    fragsym_id = symm.irrep_name2id (solver.mol.groupname, fragsym)
                else:
                    fragsym_id = fragsym
                    fragsym_str = symm.irrep_id2name (solver.mol.groupname, fragsym)
                wfnsym ^= fragsym_id
                wfnsym_f.append (fragsym_str)
                lroots_i = 0
                if ci is not None:
                    if ci[ifrag] is not None:
                        ci_i = ci[ifrag]
                        if ci_i[state] is not None:
                            ci_ij = ci_i[state]
                            lroots_i = 1 if ci_ij.ndim<3 else ci_ij.shape[0]
                lroots.append (lroots_i)
            s2_tot += sum ([2*m1*m2 for m1, m2 in combinations (m_f, 2)])
            s_f, m_f = np.asarray (s_f), np.asarray (m_f)
            if np.all (m_f<0): m_f *= -1
            s_pure = bool (np.all (s_f==m_f))
            wfnsym = symm.irrep_id2name (self.mol.groupname, wfnsym)
            neleca = sum (neleca_f)
            nelecb = sum (nelecb_f)
            log.info ("LAS space %d: (%de+%de,%do) wfynsm=%s", state, neleca, nelecb, self.ncas, wfnsym)
            log.info ("Converged? %s", self.states_converged[state])
            log.info ("E(LAS) = %.15g", self.e_states[state])
            log.info ("S^2 = %.7f (%s)", s2_tot, ('Impure','Pure')[s_pure])
            log.info ("Space table")
            log.info (" frag    (ae+be,no)  2S+1   ir   lroots")
            for i in range (self.nfrags):
                smult_f = int (round (2*s_f[i] + 1))
                tupstr = '({}e+{}e,{}o)'.format (neleca_f[i], nelecb_f[i], self.ncas_sub[i])
                log.info (" %4d %13s  %4d  %3s   %6d", i, tupstr, smult_f, wfnsym_f[i], lroots[i])

    def check_sanity (self):
        casci.CASCI.check_sanity (self)

    dump_chk = chkfile.dump_las
    load_chk = load_chk_ = chkfile.load_las_

class LASCISymm (casci_symm.CASCI, LASCINoSymm):

    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, wfnsym_sub=None, frozen=None,
                 **kwargs):
        LASCINoSymm.__init__(self, mf, ncas, nelecas, ncore=ncore, spin_sub=spin_sub,
                             frozen=frozen, **kwargs)
        if getattr (self.mol, 'groupname', None) in ('Dooh', 'Coov'):
            raise NotImplementedError ("LASSCF support for cylindrical point group {}".format (
                self.mol.groupname))
        if wfnsym_sub is None: wfnsym_sub = [0 for icas in self.ncas_sub]
        # TODO: guess wfnsym_sub intelligently (0 can be impossible for some multiplicities)
        for wfnsym, frag in zip (wfnsym_sub, self.fciboxes):
            if isinstance (wfnsym, (str, np.str_)):
                wfnsym = symm.irrep_name2id (self.mol.groupname, wfnsym)
            frag.fcisolvers[0].wfnsym = wfnsym

    make_rdm1s = LASCINoSymm.make_rdm1s
    make_rdm1 = LASCINoSymm.make_rdm1
    get_veff = LASCINoSymm.get_veff
    get_h1eff = get_h1las = h1e_for_las
    dump_flags = LASCINoSymm.dump_flags
    dump_spaces = LASCINoSymm.dump_spaces
    check_sanity = LASCINoSymm.check_sanity

    @property
    def wfnsym (self):
        ''' This now returns the product of the irreps of the subspaces '''
        wfnsym = [0,]*self.nroots
        for frag in self.fciboxes:
            for state, solver in enumerate (frag.fcisolvers):
                wfnsym[state] ^= solver.wfnsym
        if self.nroots == 1: wfnsym = wfnsym[0]
        return wfnsym
    @wfnsym.setter
    def wfnsym (self, ir):
        raise RuntimeError (("Cannot assign the whole-system symmetry of a LASCI wave function. "
                             "Address fciboxes[ifrag].fcisolvers[istate].wfnsym instead."))

    def kernel(self, mo_coeff=None, ci0=None, lroots=None, lweights=None, verbose=None,
               assert_no_dupes=False, _dry_run=False):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci0 is None:
            ci0 = self.ci

        # Initialize/overwrite mo_coeff.orbsym. Don't pass ci0 because it's not the right shape
        lib.logger.info (self, ("LASCI lazy hack note: lines below reflect the point-group "
                                "symmetry of the whole molecule but not of the individual "
                                "subspaces"))
        mo_coeff = self.mo_coeff = self.label_symmetry_(mo_coeff)
        return LASCINoSymm.kernel(self, mo_coeff=mo_coeff, ci0=ci0, lroots=lroots,
            lweights=lweights, verbose=verbose, assert_no_dupes=assert_no_dupes,
            _dry_run=_dry_run)

    def canonicalize (self, mo_coeff=None, ci=None, casdm1fs=None, natorb_casdm1=None, veff=None,
                      h2eff_sub=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        mo_coeff = self.label_symmetry_(mo_coeff)
        return canonicalize (self, mo_coeff=mo_coeff, ci=ci, casdm1fs=casdm1fs,
                             natorb_casdm1=natorb_casdm1,
                             h2eff_sub=h2eff_sub, orbsym=mo_coeff.orbsym)

    def label_symmetry_(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        ncore = self.ncore
        ncas_sub = self.ncas_sub
        nocc = ncore + sum (ncas_sub)
        if ncore > 0:
            mo_coeff[:,:ncore] = symm.symmetrize_space (self.mol, mo_coeff[:,:ncore])
        for isub, ncas in enumerate (ncas_sub):
            i = ncore + sum (ncas_sub[:isub])
            j = i + ncas
            mo_coeff[:,i:j] = symm.symmetrize_space (self.mol, mo_coeff[:,i:j])
        if mo_coeff.shape[1] > nocc:
            mo_coeff[:,nocc:] = symm.symmetrize_space (self.mol, mo_coeff[:,nocc:])
        orbsym = symm.label_orb_symm (self.mol, self.mol.irrep_id,
                                      self.mol.symm_orb, mo_coeff,
                                      s=self._scf.get_ovlp ())
        mo_coeff = lib.tag_array (mo_coeff, orbsym=orbsym)
        return mo_coeff
        
    @lib.with_doc(LASCINoSymm.localize_init_guess.__doc__)
    def localize_init_guess (self, frags_atoms, mo_coeff=None, spin=None, lo_coeff=None, fock=None,
                             mo_occ=None, freeze_cas_spaces=False, smults_f=None, nelec_f=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = casci_symm.label_symmetry_(self, mo_coeff)
        return LASCINoSymm.localize_init_guess (self, frags_atoms, mo_coeff=mo_coeff, spin=spin,
            lo_coeff=lo_coeff, fock=fock, mo_occ=mo_occ, freeze_cas_spaces=freeze_cas_spaces,
            smults_f=smults_f, nelec_f=nelec_f)

    def _svd (self, mo_lspace, mo_rspace, s=None, mo_occ=None, rngs=None, **kwargs):
        # if mo_occ is nontrivial, then I don't care about the lvecs
        if mo_occ is None: mo_occ = np.ones (mo_rspace.shape[1], dtype=int)
        if rngs is None: rngs = [None,None,None]
        lsymm = getattr (mo_lspace, 'orbsym', None)
        if lsymm is None:
            mo_lspace = symm.symmetrize_space (self.mol, mo_lspace)
            lsymm = symm.label_orb_symm(self.mol, self.mol.irrep_id,
                self.mol.symm_orb, mo_lspace, s=s)
        rsymm = getattr (mo_rspace, 'orbsym', None)
        if rsymm is None:
            mo_rspace = symm.symmetrize_space (self.mol, mo_rspace)
            rsymm = symm.label_orb_symm(self.mol, self.mol.irrep_id,
                self.mol.symm_orb, mo_rspace, s=s)
        mo_occ1 = []
        mo_occ2 = []
        svals = []
        mo_rvecs = []
        mo_rnull = []
        rsymm1 = []
        rsymm2 = []
        for m in np.unique (mo_occ):
            idx = (mo_occ==m)
            l, sv, r = self._svd1 (mo_lspace, mo_rspace[:,idx], lsymm, rsymm[idx], s=s, 
                                   rng=rngs[int(round(m))], **kwargs)
            mo_lvecs = l
            k = min (len (sv), np.count_nonzero (idx))
            svals.append (sv[:k])
            mo_rvecs.append (r[:,:k])
            mo_occ1.append (np.asarray ([m,]*k))
            rsymm1.append (r.orbsym[:k].copy ())
            if k < np.count_nonzero (idx):
                assert (r.shape[1] == np.count_nonzero (idx))
                mo_rnull.append (r[:,k:])
                mo_occ2.append (np.asarray ([m,]*(np.count_nonzero (idx) - k)))
                rsymm2.append (r.orbsym[k:].copy ())
        mo_rvecs = np.concatenate (mo_rvecs + mo_rnull, axis=1)
        svals = np.concatenate (svals)
        mo_occ1 = np.concatenate (mo_occ1 + mo_occ2)
        idx = np.argsort (-svals)
        k = len (idx)
        svals = svals[idx]
        mo_occ1[:k] = mo_occ1[:k][idx]
        mo_rvecs[:,:k] = mo_rvecs[:,:k][:,idx]
        lsymm = mo_lvecs.orbsym
        lsymm[:k] = lsymm[:k][idx]
        mo_lvecs[:,:k] = mo_lvecs[:,:k][:,idx]
        mo_rvecs = lib.tag_array (mo_rvecs, orbsym=np.concatenate (rsymm1 + rsymm2))
        mo_lvecs = lib.tag_array (mo_lvecs, orbsym=lsymm)
        return mo_lvecs, svals, mo_rvecs, mo_occ1

    def _svd1 (self, mo_lspace, mo_rspace, lsymm, rsymm, s=None, rng=None, **kwargs):
        if s is None: s = self._scf.get_ovlp ()
        mo_lvecs, svals, mo_rvecs, lsymm, rsymm = _svd_subspaces_symm (s, mo_lspace, mo_rspace,
                                                                       lsymm, rsymm)
        mo_lvecs = lib.tag_array (mo_lvecs, orbsym=lsymm)
        mo_rvecs = lib.tag_array (mo_rvecs, orbsym=rsymm)
        return _shift_svals (mo_lvecs, svals, mo_rvecs, rng)
     
