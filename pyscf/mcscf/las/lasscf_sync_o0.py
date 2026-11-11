from pyscf import gto, scf, lib, symm
from pyscf.mcscf import mc_ao2mo, mc1step
from pyscf.mcscf import df as mc_df
from pyscf.csf_fci.csfstring import ImpossibleCIvecError
from pyscf.mcscf.las import _DFLASCI, lasci, lasscf_guess
from scipy.sparse import linalg as sparse_linalg
from scipy import linalg 
import numpy as np

# This must be locked to CSF solver for the forseeable future, because I know of no other way to
# handle spin-breaking potentials while retaining spin constraint

# An implementation that carries out vLASSCF, but without utilizing Schmidt decompositions
# or "fragment" subspaces, so that the orbital-optimization part scales no better than
# CASSCF. Eventually to be modified into a true all-PySCF implementation of vLASSCF

localize_init_guess=lasscf_guess._localize

class MicroIterInstabilityException (Exception):
    pass

def get_level_shift (trust_radius, prec_op, g, tol=1e-8):
    x = prec_op (-g)
    sign = x.dot (g) / np.sqrt (linalg.norm (x) * linalg.norm (g))
    g = linalg.norm (g)
    x = linalg.norm (x)
    if x <= trust_radius: return 0
    x0 = trust_radius 
    shift = (g/x) + (g/x0)
    assert (shift>=0), "{} {} {} {}".format (g, x, x0, shift)
    return shift

def kernel (las, mo_coeff=None, ci0=None, casdm0_fr=None, conv_tol_grad=1e-4, 
        assert_no_dupes=False, verbose=lib.logger.NOTE):
    from pyscf.mcscf.las.lasci import _eig_inactive_virtual
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if assert_no_dupes: las.assert_no_duplicates ()
    log = lib.logger.new_logger(las, verbose)
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    log.debug('Start LASSCF')
    # NB: no DF guard here. The active-active pair subproblem built by
    # combine.combine_pair -> crunch.get_pair_laspscf (inherit_df=False) carries use_gpu
    # but deliberately has no with_df; get_h2eff falls back to exact ERIs for it.
    h2eff_sub = las.get_h2eff (mo_coeff)
    t1 = log.timer('integral transformation to LAS space', *t0)

    # In the first cycle, I may pass casdm0_fr instead of ci0.
    # Therefore, I need to work out this get_veff call separately.
    # This is only for compatibility with the "old" algorithm
    if ci0 is None and casdm0_fr is not None:
        casdm0_sub = [np.einsum ('rsij,r->sij', dm, las.weights) for dm in casdm0_fr]
        dm1_core = mo_coeff[:,:las.ncore] @ mo_coeff[:,:las.ncore].conjugate ().T
        dm1s_sub = [np.stack ([dm1_core, dm1_core], axis=0)]
        for idx, casdm1s in enumerate (casdm0_sub):
            mo = las.get_mo_slice (idx, mo_coeff=mo_coeff)
            moH = mo.conjugate ().T
            dm1s_sub.append (np.tensordot (mo, np.dot (casdm1s, moH), 
                                           axes=((1),(1))).transpose (1,0,2))
        dm1s_sub = np.stack (dm1s_sub, axis=0)
        dm1s = dm1s_sub.sum (0)
        veff = las.get_veff (dm=dm1s)
        casdm1s_sub = casdm0_sub
        casdm1frs = casdm0_fr
    else:
        if (ci0 is None or any ([c is None for c in ci0]) or
          any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
            ci0 = las.get_init_guess_ci (mo_coeff, h2eff_sub, ci0)
        if (ci0 is None or any ([c is None for c in ci0]) or
          any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
            raise RuntimeError ("failed to populate get_init_guess")
        casdm1frs = las.states_make_casdm1s_sub (ci=ci0)
        casdm1s_sub = las.make_casdm1s_sub (ci=ci0, casdm1frs=casdm1frs)
        dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci0, casdm1s_sub=casdm1s_sub)
        veff = las.get_veff (dm = dm1s)
    t1 = log.timer('LASSCF initial get_veff', *t1)

    ugg = None
    converged = False
    ci1 = ci0
    t2 = (t1[0], t1[1])
    it = 0
    for it in range (las.max_cycle_macro):
        e_cas, ci1 = ci_cycle (las, mo_coeff, ci1, veff, h2eff_sub, casdm1frs, log)
        if ugg is None: ugg = las.get_ugg (mo_coeff, ci1)
        log.info ('LASSCF subspace CI energies: {}'.format (e_cas))
        t1 = log.timer ('LASSCF ci_cycle', *t1)

        # Canonicalize inactive and virtual spaces to set many off-diagonal elements of the
        # orbital-rotation Hessian to zero, which should improve the microiteration below
        fock = las.get_hcore () + veff.sum (0)/2
        fock = mo_coeff.conj ().T @ fock @ mo_coeff
        orbsym = getattr (mo_coeff, 'orbsym', None)
        ene, umat = _eig_inactive_virtual (las, fock, orbsym=orbsym)
        mo_coeff = mo_coeff @ umat
        if orbsym is not None:
            mo_coeff = lib.tag_array (mo_coeff, orbsym=orbsym)
        h2eff_sub[:,:] = umat.conj ().T @ h2eff_sub

        casdm1s_new = las.make_casdm1s_sub (ci=ci1)
        if not isinstance (las, _DFLASCI):
            #veff = las.get_veff (mo_coeff=mo_coeff, ci=ci1)
            veff = las.get_veff (dm = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci1))
        if isinstance (las, _DFLASCI):
            dcasdm1s = [dm_new - dm_old for dm_new, dm_old in zip (casdm1s_new, casdm1s_sub)]
            bmPu = getattr (h2eff_sub, 'bmPu', None)
            veff += las.fast_veffa (dcasdm1s, bmPu, mo_coeff=mo_coeff, ci=ci1) 
        casdm1s_sub = casdm1s_new

        t1 = log.timer ('LASSCF get_veff after ci', *t1)
        H_op = las.get_hop (ugg=ugg, mo_coeff=mo_coeff, ci=ci1, h2eff_sub=h2eff_sub, veff=veff,
                            do_init_eri=False)
        g_vec = H_op.get_grad ()
        if las.verbose > lib.logger.INFO:
            g_orb_test, g_ci_test = las.get_grad (ugg=ugg, mo_coeff=mo_coeff, ci=ci1,
                                                  h2eff_sub=h2eff_sub, veff=veff)[:2]
            if ugg.nvar_orb:
                err = linalg.norm (g_orb_test - g_vec[:ugg.nvar_orb])
                log.debug ('GRADIENT IMPLEMENTATION TEST: |D g_orb| = %.15g', err)
                assert (err < 1e-5), '{}'.format (err)
            for isub in range (len (ugg.ncsf_sub)):
                # TODO: double-check that this code works in SA-LASSCF
                i = ugg.ncsf_sub[:isub].sum ()
                j = i + ugg.ncsf_sub[isub].sum ()
                k = i + ugg.nvar_orb
                l = j + ugg.nvar_orb
                log.debug ('GRADIENT IMPLEMENTATION TEST: |D g_ci({})| = %.15g'.format (isub), 
                           linalg.norm (g_ci_test[i:j] - g_vec[k:l]))
            # TODO: figure out why this fails in intermediate combined laspscfs in lasscf_async
            err = linalg.norm (g_ci_test - g_vec[ugg.nvar_orb:])
            assert (err < 1e-5), '{}'.format (err)
        gx = H_op.get_gx ()
        prec_op = H_op.get_prec ()
        floating_level_shift = get_level_shift (las.trust_radius, prec_op, g_vec)
        if floating_level_shift > 0:
            log.debug ('Applying a floating level shift of %e', floating_level_shift)
        prec_op.Hdiag += floating_level_shift
        H_op.level_shift += floating_level_shift
        err = get_level_shift (las.trust_radius, prec_op, g_vec)
        assert (err < 1e-8), '{} {}'.format (floating_level_shift, err)
        norm_gorb = linalg.norm (g_vec[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
        norm_gci = linalg.norm (g_vec[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
        norm_gx = linalg.norm (gx) if gx.size else 0.0
        x0 = prec_op._matvec (-g_vec)
        norm_xorb = linalg.norm (x0[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
        norm_xci = linalg.norm (x0[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
        lib.logger.info (
            las, 'LASSCF macro %d : E = %.15g ; |g_int| = %.15g ; |g_ci| = %.15g ; |g_x| = %.15g',
            it, H_op.e_tot, norm_gorb, norm_gci, norm_gx)
        #log.info (
        #    ('LASSCF micro init : E = %.15g ; |g_orb| = %.15g ; |g_ci| = %.15g ; |x0_orb| = %.15g '
        #    '; |x0_ci| = %.15g'), H_op.e_tot, norm_gorb, norm_gci, norm_xorb, norm_xci)
        las.dump_chk (mo_coeff=mo_coeff, ci=ci1)
        if (((norm_gorb<conv_tol_grad and norm_gci<conv_tol_grad)
             or ((norm_gorb+norm_gci)<norm_gx/10))
            and (it>=las.min_cycle_macro)):
                converged = True
                break
        H_op._init_eri_() 
        # ^ This is down here to save time in case I am already converged at initialization
        t1 = log.timer ('LASSCF Hessian constructor', *t1)
        microit = [0]
        last_x = [None]
        first_norm_x = [None]
        def my_callback (x):
            microit[0] += 1
            norm_xorb = linalg.norm (x[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
            norm_xci = linalg.norm (x[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
            addr_max = np.argmax (np.abs (x))
            id_max = ugg.addr2idstr (addr_max)
            x_max = x[addr_max]/np.pi
            log.debug ('Maximum step vector element x[{}] = {}*pi ({})'.format (addr_max, x_max, id_max))
            if las.verbose > lib.logger.INFO:
                Hx = H_op._matvec (x) # This doubles the price of each iteration!!
                resid = g_vec + Hx
                norm_gorb = linalg.norm (resid[:ugg.nvar_orb]) if ugg.nvar_orb else 0.0
                norm_gci = linalg.norm (resid[ugg.nvar_orb:]) if ugg.ncsf_sub.sum () else 0.0
                xorb, xci = ugg.unpack (x)
                xci = [[x_s * las.weights[iroot] for iroot, x_s in enumerate (x_rs)]
                       for x_rs in xci]
                xscale = ugg.pack (xorb, xci)
                Ecall = H_op.e_tot + xscale.dot (g_vec + (Hx/2))
                log.info (('LASSCF micro %d : E = %.15g ; |g_orb| = %.15g ; |g_ci| = %.15g ;'
                          '|x_orb| = %.15g ; |x_ci| = %.15g'), microit[0], Ecall, norm_gorb,
                          norm_gci, norm_xorb, norm_xci)
            else:
                log.info ('LASSCF micro %d : |x_orb| = %.15g ; |x_ci| = %.15g', microit[0],
                          norm_xorb, norm_xci)
            if abs(x_max)>.5: # Nonphysical step vector element
                if last_x[0] is None:
                    x[np.abs (x)>.5*np.pi] = 0
                    last_x[0] = x
                raise MicroIterInstabilityException ("|x[i]| > pi/2")
            norm_x = linalg.norm (x)
            if first_norm_x[0] is None:
                first_norm_x[0] = norm_x
            elif norm_x > 10*first_norm_x[0]:
                raise MicroIterInstabilityException ("||x(n)|| > 10*||x(0)||")
            last_x[0] = x.copy ()

        my_tol = max (conv_tol_grad, norm_gx/10)
        try:
            x = sparse_linalg.cg (H_op, -g_vec, x0=x0, atol=my_tol,
                                  maxiter=las.max_cycle_micro, callback=my_callback,
                                  M=prec_op)[0]
            t1 = log.timer ('LASSCF {} microcycles'.format (microit[0]), *t1)
            mo_coeff, ci1, h2eff_sub = H_op.update_mo_ci_eri (x, h2eff_sub)
            t1 = log.timer ('LASSCF Hessian update', *t1)

            #veff = las.get_veff (mo_coeff=mo_coeff, ci=ci1)
            veff = las.get_veff (dm = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci1))
            t1 = log.timer ('LASSCF get_veff after secondorder', *t1)
        except MicroIterInstabilityException as e:
            log.info ('Unstable microiteration aborted: %s', str (e))
            t1 = log.timer ('LASSCF {} microcycles'.format (microit[0]), *t1)
            x = last_x[0]
            for i in range (3): # Make up to 3 attempts to scale-down x if necessary
                mo2, ci2, h2eff_sub2 = H_op.update_mo_ci_eri (x, h2eff_sub)
                t1 = log.timer ('LASSCF Hessian update', *t1)
                veff2 = las.get_veff (dm = las.make_rdm1s (mo_coeff=mo2, ci=ci2))
                t1 = log.timer ('LASSCF get_veff after secondorder', *t1)
                e2 = las.energy_nuc () + las.energy_elec (mo_coeff=mo2, ci=ci2, h2eff=h2eff_sub2,
                                                          veff=veff2)
                if e2 < H_op.e_tot:
                    break
                log.info ('New energy ({}) is higher than keyframe energy ({})'.format (
                    e2, H_op.e_tot))
                log.info ('Attempt {} of 3 to scale down trial step vector'.format (i+1))
                x *= .5
            mo_coeff, ci1, h2eff_sub, veff = mo2, ci2, h2eff_sub2, veff2


        casdm1frs = las.states_make_casdm1s_sub (ci=ci1)
        casdm1s_sub = las.make_casdm1s_sub (ci=ci1)

    t2 = log.timer ('LASSCF {} macrocycles'.format (it), *t2)

    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub,
                                                 veff=veff)
    if log.verbose > lib.logger.INFO:
        e_tot_test = las.get_hop (ugg=ugg, mo_coeff=mo_coeff, ci=ci1, h2eff_sub=h2eff_sub,
                                  veff=veff, do_init_eri=False).e_tot
    dm_core = 2 * mo_coeff[:,:las.ncore] @ mo_coeff[:,:las.ncore].conj ().T
    bmPu = getattr (h2eff_sub, 'bmPu', None)
    veff_a = np.stack ([las.fast_veffa ([d[state] for d in casdm1frs], bmPu,
                                        mo_coeff=mo_coeff, ci=ci1)
                        for state in range (las.nroots)], axis=0)
    veff_c = las.get_veff (dm=dm_core)
    # veff's spin-summed component should be correct because I called get_veff with spin-summed rdm
    veff = veff_c[None,None,:,:] + veff_a 
    veff = lib.tag_array (veff, c=veff_c, sa=np.einsum ('rsij,r->sij', veff, las.weights))
    e_states = las.energy_nuc () + np.array (las.states_energy_elec (mo_coeff=mo_coeff, ci=ci1,
                                                                     h2eff=h2eff_sub, veff=veff))
    if log.verbose > lib.logger.INFO:
        e_tot_test = np.dot (las.weights, e_states)
        if abs (e_tot_test-e_tot) > 1e-8:
            log.warn ('order-of-operations disagreement of %e in state-averaged energy (%e)',
                      e_tot_test-e_tot, e_tot)

    # I need the true veff, with f^a_a and f^i_i spin-separated, in order to use the Hessian
    # Better to do it here with bmPu than in localintegrals

    log.info ('LASSCF %s after %d cycles', ('not converged', 'converged')[converged], it+1)
    log.info ('LASSCF E = %.15g ; |g_int| = %.15g ; |g_ci| = %.15g ; |g_ext| = %.15g', e_tot,
              norm_gorb, norm_gci, norm_gx)
    t1 = log.timer ('LASSCF wrap-up', *t1)

    if las.canonicalization:
        mo_coeff, mo_energy, mo_occ, ci1, h2eff_sub = las.canonicalize (
            mo_coeff, ci1, veff=veff.sa, h2eff_sub=h2eff_sub)
        t1 = log.timer ('LASSCF canonicalization', *t1)
    else:
        fock = mo_coeff.conjugate ().T @ las.get_fock (mo_coeff=mo_coeff, ci=ci1,
                                                       veff=veff.sa)
        mo_energy = (fock * mo_coeff).sum (0)

    t0 = log.timer ('LASSCF kernel function', *t0)

    las.dump_chk (mo_coeff=mo_coeff, ci=ci1)

    return converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, ci1, h2eff_sub, veff

def ci_cycle (las, mo, ci0, veff, h2eff_sub, casdm1frs, log):
    if ci0 is None: ci0 = [None for idx in range (las.nfrags)]
    frozen_ci = las.frozen_ci
    if frozen_ci is None: frozen_ci = []
    # CI problems
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    h1eff_sub = las.get_h1eff (mo, veff=veff, h2eff_sub=h2eff_sub, casdm1frs=casdm1frs)
    ncas_cum = np.cumsum ([0] + las.ncas_sub.tolist ()) + las.ncore
    e_cas = []
    ci1 = []
    e0 = 0.0 
    for isub, (fcibox, ncas, nelecas, h1e, fcivec) in enumerate (zip (las.fciboxes, las.ncas_sub,
                                                                      las.nelecas_sub, h1eff_sub,
                                                                      ci0)):
        eri_cas = las.get_h2eff_slice (h2eff_sub, isub, compact=8)
        max_memory = max(400, las.max_memory-lib.current_memory()[0])
        orbsym = getattr (mo, 'orbsym', None)
        if orbsym is not None:
            i = ncas_cum[isub]
            j = ncas_cum[isub+1]
            orbsym = orbsym[i:j]
            orbsym_io = orbsym.copy ()
            if np.issubdtype (orbsym.dtype, np.integer):
                orbsym_io = np.asarray ([symm.irrep_id2name (las.mol.groupname, x)
                                         for x in orbsym])
            log.info ("LASSCF subspace {} with orbsyms {}".format (isub, orbsym_io))
        else:
            log.info ("LASSCF subspace {} with no orbsym information".format (isub))
        if log.verbose > lib.logger.DEBUG: 
         for state, solver in enumerate (fcibox.fcisolvers):
            wfnsym = getattr (solver, 'wfnsym', None)
            if (wfnsym is not None) and (orbsym is not None):
                if isinstance (wfnsym, str):
                    wfnsym_str = wfnsym
                else:
                    wfnsym_str = symm.irrep_id2name (las.mol.groupname, wfnsym)
                log.debug1 ("LASSCF subspace {} state {} with wfnsym {}".format (isub, state,
                                                                                wfnsym_str))

        if isub not in frozen_ci:
            e_sub, fcivec = fcibox.kernel(h1e, eri_cas, ncas, nelecas,
                                          ci0=fcivec, verbose=log,
                                          max_memory = max_memory,
                                          ecore=e0, orbsym=orbsym)
        else:
            e_sub = 0 # TODO: proper energy calculation (probably doesn't matter tho)
        e_cas.append (e_sub)
        ci1.append (fcivec)
        t1 = log.timer ('FCI box for subspace {}'.format (isub), *t1)
    return e_cas, ci1

def all_nonredundant_idx (nmo, ncore, ncas_sub):
    ''' Generate a index mask array addressing all nonredundant, lower-triangular elements of an
    nmo-by-nmo orbital-rotation unitary generator amplitude matrix for a LASSCF or LASSCF problem
    with ncore inactive orbitals and len (ncas_sub) fragments with ncas_sub[i] active orbitals in
    the ith fragment:

        <--------------nmo--------------->
        <-ncore->|<-sum(ncas_sub)->|
        __________________________________
        | False  |False|False| ... |False|
        |  True  |False|False| ... |False|
        |  True  | True|False| ... |False|
        |  ...   | ... | ... | ... |False|
        |  True  | True| True| ....|False|
        ----------------------------------
    '''
    nocc = ncore + sum (ncas_sub)
    idx = np.zeros ((nmo, nmo), dtype=np.bool_)
    idx[ncore:,:ncore] = True # inactive -> everything
    idx[nocc:,ncore:nocc] = True # active -> virtual
    sub_slice = np.cumsum ([0] + ncas_sub.tolist ()) + ncore
    idx[sub_slice[-1]:,:sub_slice[0]] = True
    for ix1, i in enumerate (sub_slice[:-1]):
        j = sub_slice[ix1+1]
        for ix2, k in enumerate (sub_slice[:ix1]):
            l = sub_slice[ix2+1]
            idx[i:j,k:l] = True
    # active -> active
    return idx

class LASSCF_UnitaryGroupGenerators (object):
    ''' Object for `pack'ing (for root-finding algorithms) and `unpack'ing (for direct
    manipulation) the nonredundant variables ('unitary generator amplitudes') of a `LASSCF' problem.
    `LASSCF' here means that the CAS is frozen relative to inactive or external orbitals, but active
    orbitals from different fragments may rotate into one another, and inactive orbitals may rotate
    into virtual orbitals, and CI vectors may also evolve. Transforms between the nonredundant
    lower-triangular part ('x') of a skew-symmetric orbital rotation matrix ('kappa')
    and transforms CI transfer vectors between the determinant and configuration state function
    bases. Subclass me to apply point-group symmetry or to do a full LASSCF calculation.

    Attributes:
        nmo : int
            Number of molecular orbitals
        frozen : sequence of int or index mask array
            Identify orbitals which are frozen.
        frozen_ci : sequence of int
            Identify fragments whose CI vectors are frozen
        nfrz_orb_idx : index mask array
            Identifies all nonredundant orbital rotation amplitudes for non-frozen orbitals
        uniq_orb_idx : index mask array
            The same as nfrz_orb_idx, but omitting active<->(inactive,virtual) degrees of freedom.
            (In the LASSCF child class uniq_orb_idx == nfrz_orb_idx.)
        ci_transformer : sequence of shape (nfrags,nroots) of :class:`CSFTransformer`
            Element [i][j] transforms between single determinants and CSFs for the ith fragment in
            the jth state
        nvar_orb : int
            Total number of nonredundant orbital-rotation degrees of freedom
        ncsf_sub : ndarray of shape (nfrags,nroots)
            Number of CSF vector elements in each fragment and state.
        nvar_tot : int
            Total length of the packed vector - approximately the number of nonredundant degrees
            of freedom (the CSF vector representation of the CI part of the problem still contains
            some redundancy even in `packed' form; fixing this is more trouble than it's worth).
    '''

    def __init__(self, las, mo_coeff, ci):
        self.nmo = mo_coeff.shape[-1]
        self.frozen = las.frozen
        self.frozen_ci = las.frozen_ci
        self._init_orb (las, mo_coeff, ci)
        self._init_ci (las, mo_coeff, ci)

    def _init_nonfrozen_orb (self, las):
        nmo, ncore, ncas_sub = self.nmo, las.ncore, las.ncas_sub
        idx = all_nonredundant_idx (nmo, ncore, ncas_sub)
        if self.frozen is not None:
            idx[self.frozen,:] = idx[:,self.frozen] = False
        self.nfrz_orb_idx = idx

    def _init_orb (self, las, mo_coeff, ci):
        self._init_nonfrozen_orb (las)
        self.uniq_orb_idx = self.nfrz_orb_idx.copy ()
        # The distinction between "uniq_orb_idx" and "nfrz_orb_idx" is an
        # artifact of backwards-compatibility with the old LASSCF implementation

    def get_gx_idx (self):
        ''' Returns an index mask array identifying all nonredundant, nonfrozen orbital rotations
        which are not considered in the current phase of the phase of the problem:
        active<->inactive and active<->virtual for the LASSCF parent class; nothing (all elements
        False) in the LASSCF child class. '''
        return np.logical_and (self.nfrz_orb_idx, np.logical_not (self.uniq_orb_idx))

    def _init_ci (self, las, mo_coeff, ci):
        self.ci_transformers = []
        if self.frozen_ci is None: self.frozen_ci = []
        for i, fcibox in enumerate (las.fciboxes):
            norb, nelec = las.ncas_sub[i], las.nelecas_sub[i]
            tf_list = []
            for j, solver in enumerate (fcibox.fcisolvers):
                solver.norb = norb
                solver.nelec = fcibox._get_nelec (solver, nelec)
                try:
                    solver.check_transformer_cache ()
                except ImpossibleCIvecError as e:
                    lib.logger.error (las, 'impossible CI vector in LAS frag %d, state %d', i, j)
                    raise (e)
                tf_list.append (solver.transformer)
            self.ci_transformers.append (tf_list)

    def pack_orb (self, kappa):
        return np.asarray (kappa[self.uniq_orb_idx]).reshape (-1)

    def unpack_orb (self, x_orb):
        x_orb = np.asarray (x_orb).reshape (-1)
        if x_orb.size != self.nvar_orb:
            raise ValueError ("orbital vector has size {}; expected {}".format (
                x_orb.size, self.nvar_orb))
        kappa = np.zeros (self.uniq_orb_idx.shape, dtype=x_orb.dtype)
        kappa[self.uniq_orb_idx] = x_orb
        return kappa - kappa.conj ().T

    def _det2csf (self, transformer, ci):
        return transformer.vec_det2csf (ci, normalize=False)

    def _csf2det (self, transformer, ci):
        return transformer.vec_csf2det (ci, normalize=False)

    def _zero_ci (self, transformer, ci_ref, dtype):
        return np.zeros (transformer.ndeta * transformer.ndetb, dtype=dtype)

    def _format_ci (self, transformer, ci, ci_ref):
        return ci

    def pack_ci (self, ci_sub):
        vectors = []
        dtypes = []
        if len (ci_sub) != len (self.ci_transformers):
            raise ValueError ("CI input must contain one entry per fragment")
        for ix, (trans_frag, ci_frag) in enumerate (
                zip (self.ci_transformers, ci_sub)):
            if len (ci_frag) != len (trans_frag):
                raise ValueError ("fragment {} has {} CI vectors; expected {}".format (
                    ix, len (ci_frag), len (trans_frag)))
            dtypes.extend (np.asarray (ci).dtype for ci in ci_frag)
            if ix in self.frozen_ci: continue
            for transformer, ci in zip (trans_frag, ci_frag):
                vectors.append (np.asarray (
                    self._det2csf (transformer, ci)).reshape (-1))
        if vectors:
            return np.concatenate (vectors)
        dtype = np.result_type (*dtypes) if dtypes else float
        return np.empty (0, dtype=dtype)

    def unpack_ci (self, x_ci):
        x_ci = np.asarray (x_ci).reshape (-1)
        if x_ci.size != self.nvar_ci:
            raise ValueError ("CI vector has size {}; expected {}".format (
                x_ci.size, self.nvar_ci))
        ci_sub = []
        offset = 0
        ci_reference = getattr (self, 'ci', None)
        for ix, trans_frag in enumerate (self.ci_transformers):
            if ci_reference is None or ci_reference[ix] is None:
                ci_ref_frag = [None] * len (trans_frag)
            else:
                ci_ref_frag = ci_reference[ix]
            ci_frag = []
            for transformer, ci_ref in zip (trans_frag, ci_ref_frag):
                if ix in self.frozen_ci:
                    dtype = (x_ci.dtype if ci_ref is None else
                             np.result_type (ci_ref, x_ci.dtype))
                    ci_frag.append (self._zero_ci (
                        transformer, ci_ref, dtype))
                    continue
                ncsf = transformer.ncsf
                ci = self._csf2det (
                    transformer, x_ci[offset:offset+ncsf])
                ci_frag.append (self._format_ci (
                    transformer, ci, ci_ref))
                offset += ncsf
            ci_sub.append (ci_frag)
        if offset != x_ci.size:
            raise ValueError ("consumed {} CI variables from a vector of size {}".format (
                offset, x_ci.size))
        return ci_sub

    def pack (self, kappa, ci_sub):
        x_orb = self.pack_orb (kappa)
        x_ci = self.pack_ci (ci_sub)
        dtype = np.result_type (x_orb.dtype, x_ci.dtype)
        x = np.empty (self.nvar_tot, dtype=dtype)
        x[:self.nvar_orb] = x_orb
        x[self.nvar_orb:] = x_ci
        return x

    def unpack (self, x):
        x = np.asarray (x).reshape (-1)
        if x.size != self.nvar_tot:
            raise ValueError ("combined vector has size {}; expected {}".format (
                x.size, self.nvar_tot))
        return self.unpack_orb (x[:self.nvar_orb]), self.unpack_ci (x[self.nvar_orb:])

    def addr2idstr (self, addr):
        if addr<self.nvar_orb:
            probe_orb = np.argwhere (self.uniq_orb_idx)[addr]
            idstr = 'orb: {},{}'.format (*probe_orb)
        else:
            addr -= self.nvar_orb
            ncsf_frag = self.ncsf_sub.sum (1)
            for i, trans_frag in enumerate (self.ci_transformers):
                if i in self.frozen_ci: continue
                if addr >= ncsf_frag[i]:
                    addr -= ncsf_frag[i]
                    continue
                for j, trans in enumerate (trans_frag):
                    if addr >= trans.ncsf:
                        addr -= trans.ncsf
                        continue
                    idstr = 'CI({}): <{}|{}>'.format (
                        i, j, trans.printable_csfstring (addr))
                    break
                break
        return idstr

    @property
    def nvar_orb (self):
        return np.count_nonzero (self.uniq_orb_idx)

    @property
    def ncsf_sub (self):
        return np.asarray ([[transformer.ncsf for transformer in trans_frag]
                            for i,trans_frag in enumerate (self.ci_transformers)
                            if i not in self.frozen_ci])

    @property
    def nvar_ci (self):
        return int (self.ncsf_sub.sum ())

    @property
    def nvar_tot (self):
        return self.nvar_orb + self.nvar_ci

class LASSCFSymm_UnitaryGroupGenerators (LASSCF_UnitaryGroupGenerators):
    __doc__ = LASSCF_UnitaryGroupGenerators.__doc__ + '''

    Symmetry subclass forbids rotations between orbitals of different point groups or CSFs of
    other-than-specified point group -> sets many additional elements of nfrz_orb_idx and
    uniq_orb_idx to False and reduces the values of nvar_orb, ncsf_sub, and nvar_tot.
    '''

    def __init__(self, las, mo_coeff, ci): 
        self.nmo = mo_coeff.shape[-1]
        self.frozen = las.frozen
        self.frozen_ci = las.frozen_ci
        if getattr (mo_coeff, 'orbsym', None) is None:
            mo_coeff = las.label_symmetry_(mo_coeff)
        orbsym = mo_coeff.orbsym
        self._init_orb (las, mo_coeff, ci, orbsym)
        self._init_ci (las, mo_coeff, ci, orbsym)
    
    def _init_orb (self, las, mo_coeff, ci, orbsym):
        LASSCF_UnitaryGroupGenerators._init_orb (self, las, mo_coeff, ci)
        self.symm_forbid = (orbsym[:,None] ^ orbsym[None,:]).astype (np.bool_)
        self.uniq_orb_idx[self.symm_forbid] = False
        self.nfrz_orb_idx[self.symm_forbid] = False

    def _init_ci (self, las, mo_coeff, ci, orbsym):
        if self.frozen_ci is None: self.frozen_ci = []
        sub_slice = np.cumsum ([0] + las.ncas_sub.tolist ()) + las.ncore
        orbsym_sub = [orbsym[i:sub_slice[isub+1]] for isub, i in enumerate (sub_slice[:-1])]
        self.ci_transformers = []
        for norb, nelec, orbsym, fcibox in zip (las.ncas_sub, las.nelecas_sub, orbsym_sub,
                                                las.fciboxes):
            tf_list = []
            fcibox.orbsym = orbsym
            for solver in fcibox.fcisolvers:
                solver.norb = norb
                solver.nelec = fcibox._get_nelec (solver, nelec)
                solver.orbsym = orbsym
                solver.check_transformer_cache ()
                tf_list.append (solver.transformer)
            self.ci_transformers.append (tf_list)

# TODO: local state-average generalization
class LASSCF_HessianOperator (sparse_linalg.LinearOperator):
    ''' The Hessian-vector product for a `LASSCF' energy minimization, implemented as a linear
    operator from the scipy.sparse.linalg module. `LASSCF' here means that the CAS is frozen
    relative to inactive or external orbitals, but active orbitals from different fragments may
    rotate into one another, and inactive orbitals may rotate into virtual orbitals, and CI vectors
    may also evolve. Implements the get_grad (gradient of the energy), get_prec (preconditioner for
    conjugate-gradient iteration), get_gx (gradient along non-`LASSCF' degrees of freedom), and
    update_mo_ci_eri (apply a shift vector `x' to MO coefficients and CI vectors) in addition to
    _matvec and _rmatvec. For a shift vector `x', in terms of attributes and methods of this class,
    the second-order power series for the total (state-averaged) electronic energy is

    e = self.e_tot + np.dot (self.get_grad (), x) + (.5 * np.dot (self._matvec (x), x))

    Args:
        las : instance of :class:`LASSCFNoSymm`
        ugg : instance of :class:`LASSCF_UnitaryGroupGenerators`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Molecular orbitals for trial state(s)
        ci : list (length = nfrags) of lists (length = nroots) of ndarrays
            CI vectors of the trial state(s); element [i][j] describes the ith fragment in the jth
            state
        casdm1frs : list of length (nfrags) of ndarrays
            ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i])
            Contains spin-separated 1-RDMs for the active orbitals of each fragment in each state.
        casdm2fr : list of length (nfrags) of ndarrays
            ith element has shape [nroots,] + [ncas_sub[i],]*4
            Contains spin-summed 2-RDMs for the active orbitals of each fragment in each state.
        ncore : int
            Number of doubly-occupied inactive orbitals
        ncas_sub : list of length (nfrags)
            Number of active orbitals in each fragment
        nelecas_sub : list of list of length (2) of length (nfrags)
            Number of active electrons in each fragment
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices, where p1 is any MO
            and an is any active MO (in any fragment).
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        do_init_eri : logical
            If False, the bPpj attribute is not initialized until the _init_eri_ method is
            separately called.

    Attributes:
        level_shift : float
            Shift added to the diagonal of the Hessian to improve convergence. Default = 1e-8.
        ncas : int
            Total number of active orbitals
        nao : int
            Total number of atomic orbitals
        nmo : int
            Total number of molecular orbitals
        nocc : int
            Total number of inactive plus active orbitals
        nroots : int
            Total number of states whose energies are averaged
        weights : list of length (nroots)
            Weights of the different states in the state average
        fciboxes : list of length (nfrags) of instances of :class:`H1EZipFCISolver`
            Contains the FCISolver objects for each fragment which implement the CI vector
            manipulation methods
        bPpj : ndarray of shape (naux,nmo,nocc)
            MO-basis CDERI array; only used in combination with density fitting. If
            do_init_eri=False is passed to the constructor
        casdm(N=1,2)[f][r][s] : ndarray or list of ndarrays
            Various 1RDMs (if N==1) or 2RDMs (if N==2) of active orbitals, obtained by summing or
            averaging over the casdm1frs and casdm2fr kwargs.
            If `f' is present, it is a list of ndarrays of length nfrags, and the last 2*N
            dimensions of the ith element are ncas_sub[i]. Otherwise, it is a single ndarray, and
            the last 2*N dimensions are ncas.
            If `r' is present, density matrices are separated by state and the first dimension of
            the ndarray(s) is nroots. Otherwise, density matrices are state-averaged.
            If 's' is present, density matrices are spin-separated and the first dimension of
            the ndarray(s) is 1+N. Otherwise, density matrices are spin-summed.
        cascm2 : ndarray of shape (ncas,ncas,ncas,ncas)
            The cumulant of the state-averaged, spin-summed 2-RDM of the active orbitals.
        dm1s : ndarray of shape (2,nmo,nmo)
            State-averaged, spin-separated 1-RDM of the whole molecule in the MO basis.
        eri_paaa : ndarray of shape (nmo, ncas, ncas, ncas)
            Same as kwarg h2eff_sub, be reshaped to be more accessible
        eri_cas : ndarray of shape [ncas,]*4
            ERIs (a1a2|a3a4)
        h1s : ndarray of shape (2,nmo,nmo)
            Spin-separated, state-averaged effective 1-electron Hamiltonian elements in MO basis
        h1s_cas : ndarray of shape (2,nmo,ncas)
            Spin-separated effective 1-electron Hamiltonian experience by the CAS, including the
            mean-field potential generated by the inactive electrons but not by any active space
        h1frs : list of length nroots of ndarray
            ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i])
            Spin-separated effective 1-electron Hamiltonian experienced by each fragment in each
            state
        e_tot : float
            Total (state-averaged) electronic energy for the trial state(s) at x=0
        fock1 : ndarray of shape (nmo,nmo)
            State-averaged first-order effective Fock matrix
        hci0 : list (length = nfrags) of lists (length = nroots) of ndarrays
            (H(i,j) - e0[i][j]) |ci[i][j]>, where H(i,j) is the effective Hamiltonian experienced
            by the ith fragment in the jth state, stored as a CI vector
        e0 : list (length = nfrags) of lists (length = nroots) of floats
            <ci[i][j]|H(i,j)|ci[i][j]>, where H(i,j) is the effective Hamiltonian experienced by
            the ith fragment in the jth state
        linkstr[l] : list (length = nfrags) of lists (length = nroots)
            PySCF FCI module linkstr and linkstrl arrays, for accelerating CI manipulation
    '''

    def __init__(self, las, ugg, mo_coeff=None, ci=None, casdm1frs=None,
            casdm2fr=None, ncore=None, ncas_sub=None, nelecas_sub=None,
            h2eff_sub=None, veff=None, do_init_eri=True):
        if mo_coeff is None: mo_coeff = las.mo_coeff
        if ci is None: ci = las.ci
        if ncore is None: ncore = las.ncore
        if ncas_sub is None: ncas_sub = las.ncas_sub
        if nelecas_sub is None: nelecas_sub = las.nelecas_sub
        if casdm1frs is None: casdm1frs = las.states_make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if casdm2fr is None: casdm2fr = las.states_make_casdm2_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub)
        if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
        self.las = las
        self.level_shift = las.ah_level_shift
        self.ugg = ugg
        self.mo_coeff = mo_coeff
        sparse_linalg.LinearOperator.__init__(self, dtype=self.mo_coeff.dtype,
                                              shape=(ugg.nvar_tot, ugg.nvar_tot))
        self.ci = ci = [[c.ravel () for c in cr] for cr in ci] 
        self.ncore = ncore
        self.ncas_sub = ncas_sub
        self.nelecas_sub = nelecas_sub
        self.ncas = ncas = sum (ncas_sub)
        self.nao = nao = mo_coeff.shape[0]
        self.nmo = nmo = mo_coeff.shape[-1]
        self.nocc = nocc = ncore + ncas
        self.fciboxes = las.fciboxes
        self.nroots = las.nroots
        self.weights = las.weights
        self.bPpj = None

        self._init_dms_(casdm1frs, casdm2fr)
        self._init_ham_(h2eff_sub, veff)
        self._init_orb_()
        self._init_ci_()
        # turn this off for extra optimization in kernel
        if do_init_eri: self._init_eri_()

    def _init_df_(self):
        _init_df_(self)

    def _init_dms_(self, casdm1frs, casdm2fr):
        las, ncore, nocc = self.las, self.ncore, self.nocc
        self.casdm1frs = casdm1frs 
        self.casdm1fs = las.make_casdm1s_sub (casdm1frs=self.casdm1frs)
        self.casdm1rs = las.states_make_casdm1s (casdm1frs=self.casdm1frs)
        self.casdm2fr = casdm2fr
        casdm1a = linalg.block_diag (*[dm[0] for dm in self.casdm1fs])
        casdm1b = linalg.block_diag (*[dm[1] for dm in self.casdm1fs])
        self.casdm1s = np.stack ([casdm1a, casdm1b], axis=0)
        casdm1 = self.casdm1s.sum (0)
        self.casdm2 = las.make_casdm2 (casdm1frs=casdm1frs, casdm2fr=casdm2fr)
        self.cascm2 = self.casdm2 - np.multiply.outer (casdm1, casdm1)
        self.cascm2 += np.multiply.outer (casdm1a, casdm1a).transpose (0,3,2,1)
        self.cascm2 += np.multiply.outer (casdm1b, casdm1b).transpose (0,3,2,1)
        self.dm1s = np.stack ([np.eye (self.nmo, dtype=self.dtype),
                               np.eye (self.nmo, dtype=self.dtype)], axis=0)
        self.dm1s[0,ncore:nocc,ncore:nocc] = casdm1a
        self.dm1s[1,ncore:nocc,ncore:nocc] = casdm1b
        self.dm1s[:,nocc:,nocc:] = 0
        
    def _init_ham_(self, h2eff_sub, veff):
        las, mo_coeff, ncas_sub = self.las, self.mo_coeff, self.ncas_sub
        ncore, ncas, nocc = self.ncore, self.ncas, self.nocc
        nao, nmo, nocc = self.nao, self.nmo, ncore+ncas
        casdm1a, casdm1b = tuple (self.casdm1s)
        casdm1 = casdm1a + casdm1b
        moH_coeff = mo_coeff.conjugate ().T
        if veff is None:
            veff = las.get_veff (dm = np.dot (
                mo_coeff, np.dot (self.dm1s, moH_coeff)
            ).transpose (1,0,2))
        self.eri_paaa = eri_paaa = lib.numpy_helper.unpack_tril (
            h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)).reshape (nmo, ncas,
            ncas, ncas)
        self.eri_cas = eri_cas = eri_paaa[ncore:nocc,:,:,:]
        hcore = las.get_hcore ()
        h1s = hcore[None,:,:] + veff
        h1s = np.dot (h1s, mo_coeff)
        self.h1s = np.dot (moH_coeff, h1s).transpose (1,0,2)
        self.h1s_cas = self.h1s[:,:,ncore:nocc].copy ()
        self.h1s_cas -= np.tensordot (eri_paaa, casdm1, axes=2)[None,:,:]
        self.h1s_cas += np.tensordot (self.casdm1s, eri_paaa, axes=((1,2),(2,1)))
        self.hcore = moH_coeff @ hcore @ mo_coeff

        self.h1frs = [np.zeros ((self.nroots, 2, nlas, nlas)) for nlas in ncas_sub]
        for ix, h1rs in enumerate (self.h1frs):
            i = sum (ncas_sub[:ix])
            j = i + ncas_sub[ix]
            k, l = i + ncore, j + ncore
            for h1s_sub, casdm1s in zip (h1rs, self.casdm1rs):
                h1s_sub[:,:,:] = self.h1s[:,k:l,k:l].copy ()
                dm1s = casdm1s.copy ()
                dm1s[:,i:j,i:j] = 0.0 # No double-counting
                dm1s[0] -= casdm1a # No state-averaging
                dm1s[1] -= casdm1b # No state-averaging
                dm1 = dm1s[0] + dm1s[1]
                h1s_sub[:,:,:] += np.tensordot (dm1, eri_cas, axes=((0,1),(2,3)))[None,i:j,i:j]
                h1s_sub[:,:,:] -= np.tensordot (dm1s, eri_cas, axes=((1,2),(2,1)))[:,i:j,i:j]

        # Total energy (for callback)
        h1 = (self.h1s + self.hcore[None,:,:]) / 2
        self.e_tot = (las.energy_nuc ()
            + np.dot (h1.ravel (), self.dm1s.ravel ())
            + np.tensordot (self.eri_cas, self.cascm2, axes=4) / 2)

    def _init_orb_(self):
        eri_paaa, ncore, nocc = self.eri_paaa, self.ncore, self.nocc
        self.fock1 = sum ([f @ d for f,d in zip (list (self.h1s), list (self.dm1s))])
        self.fock1[:,ncore:nocc] += np.tensordot (eri_paaa, self.cascm2, axes=((1,2,3),(1,2,3)))

    def _init_ci_(self):
        ci, ncas_sub, nelecas_sub = self.ci, self.ncas_sub, self.nelecas_sub
        self.linkstrl = []
        self.linkstr = []
        for fcibox, no, ne in zip (self.fciboxes, ncas_sub, nelecas_sub):
            self.linkstrl.append (fcibox.states_gen_linkstr (no, ne, True)) 
            self.linkstr.append (fcibox.states_gen_linkstr (no, ne, False))
        self.hci0 = self.Hci_all (None, self.h1frs, self.eri_cas, ci)
        self.e0 = [[hc.dot (c) for hc, c in zip (hcr, cr)] for hcr, cr in zip (self.hci0, ci)]
        self.hci0 = [[hc - c*e for hc, c, e in zip (hcr, cr, er)]
                     for hcr, cr, er in zip (self.hci0, ci, self.e0)]

    def _init_eri_(self):
        self._init_df_()
        if isinstance (self.las, _DFLASCI):
            self.cas_type_eris = mc_df._ERIS (self.las, self.mo_coeff, self.with_df)
        else:
            self.cas_type_eris = mc_ao2mo._ERIS (self.las, self.mo_coeff,
                method='incore', level=2) # level=2 -> ppaa, papa only
                # level=1 computes more stuff; it's only useful if I
                # want the honest hdiag in get_prec ()
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas
        paaa_test = np.zeros_like (self.eri_paaa)
        for p in range (self.nmo):
            paaa_test[p] = self.cas_type_eris.ppaa[p][ncore:nocc]
        if not np.allclose (paaa_test, self.eri_paaa):
            lib.logger.warn(self.las, 'possible (pa|aa) inconsistency; max err = %e',
                         np.amax (np.abs (paaa_test-self.eri_paaa)))

    def Hci (self, fcibox, no, ne, h0r, h1rs, h2, ci, linkstrl=None):
        ''' For a single fragment, evaluate the FCI operation H(i)|ci[i]>, where H(i) is the
        effective Hamiltonian experienced by the fragment in the ith state

        Args:
            fcibox : instance of :class:`H1EZipFCISolver`
                The FCI solver method for the fragment
            no : integer
                Number of active orbitals in the fragment
            ne : list of length (2) of integers
                Number of spin-up and spin-down electrons in the fragment
            h0r : list of length nroots
                Constant part of the effective Hamiltonian for each state
            h1rs : ndarray of shape (nroots,2,no,no)
                Spin-separated 1-electron part of the effective Hamiltonian for each state
            h2 : ndarray of shape (no,no,no,no)
                Two-electron integrals
            ci : list of length nroots of ndarray
                CI vectors

        Kwargs:
            linkstrl : see pyscf.fci module documentation

        Returns:
            hcr : list of length nroots of ndarray
        '''
        hr = fcibox.states_absorb_h1e (h1rs, h2, no, ne, 0.5)
        hcr = fcibox.states_contract_2e (hr, ci, no, ne, link_index=linkstrl)
        hcr = [hc + (h0 * c) for hc, h0, c in zip (hcr, h0r, ci)]
        return hcr

    def Hci_all (self, h0fr, h1frs, h2, ci_sub):
        ''' For all fragments, evaluate the FCI operations H(i,j)|ci_sub[i][j]>, where H(i,j) is
        the effective Hamiltonian experienced by the ith fragment in the jth state.

        Args:
            h0fr : list of length nfrags of lists of length nroots
                Constant part of the effective Hamiltonian for each fragment and state
            h1frs : list of length nfrags of ndarrays
                Spin-separated 1-electron parts of the effective Hamiltonian for each fragment and
                state
            h2 : ndarray of shape (ncas,ncas,ncas,ncas)
                Two-electron integrals spanning the entire active space
            ci_sub : list of length nfrags of list of length nroots of ndarray
                CI vectors

        Returns:
            hcfr : list of length nfrags of list of length nroots of ndarray
        '''
        if h0fr is None: h0fr = [[0.0 for h1r in h1rs] for h1rs in h1frs]
        hcfr = []
        for isub, (fcibox, h0, h1rs, ci) in enumerate (zip (self.fciboxes, h0fr, h1frs, ci_sub)):
            if self.linkstrl is not None: linkstrl = self.linkstrl[isub] 
            ncas = self.ncas_sub[isub]
            nelecas = self.nelecas_sub[isub]
            i = sum (self.ncas_sub[:isub])
            j = i + ncas
            h2_i = h2[i:j,i:j,i:j,i:j]
            h1rs_i = h1rs
            hcfr.append (self.Hci (fcibox, ncas, nelecas, h0, h1rs_i, h2_i, ci, linkstrl=linkstrl))
        return hcfr

    def make_tdm1s2c_sub (self, ci1):
        ''' Make effective 1-body and 2-body cumulant density matrices to first order
        in a CI rotation vector. 

        Args:
            ci : list (length = nfrags) of lists (length = nroots) of ndarrays
                CI shift vectors

        Returns:
            tdm1s : ndarray of shape (nroots,2,ncas,ncas)
                Spin-separated effective 1-body density matrix
            tcm2 : ndarray of shape (ncas,ncas,ncas,ncas)
                Spin-summed state-averaged cumulant effective 2-body density matrix
        '''
        tdm1rs = np.zeros ((self.nroots, 2, self.ncas, self.ncas), dtype=self.dtype)
        tcm2 = np.zeros ([self.ncas,]*4, dtype=self.dtype)
        for isub, (fcibox, ncas, nelecas, c1, c0, casdm1rs, casdm1s, casdm2r) in enumerate (
          zip (self.fciboxes, self.ncas_sub, self.nelecas_sub, ci1, self.ci,
          self.casdm1frs, self.casdm1fs, self.casdm2fr)):
            s01 = [c1i.dot (c0i) for c1i, c0i in zip (c1, c0)]
            i = sum (self.ncas_sub[:isub])
            j = i + ncas
            linkstr = None if self.linkstr is None else self.linkstr[isub]
            dm1, dm2 = fcibox.states_trans_rdm12s (c1, c0, ncas, nelecas, link_index=linkstr)
            # Subtrahend: super important, otherwise the veff part of CI response is even worse
            # With this in place, I don't have to worry about subtracting an overlap times gradient
            tdm1rs[:,:,i:j,i:j] = np.stack ([np.stack (t, axis=0) - c * s
                                             for t, c, s in zip (dm1, casdm1rs, s01)], axis=0)
            dm2 = np.stack ([(sum (t) - (c*s)) / 2
                             for t, c, s, in zip (dm2, casdm2r, s01)], axis=0)
            dm2 = np.einsum ('rijkl,r->ijkl', dm2, fcibox.weights)
            #tdm1frs[isub,:,:,i:j,i:j] = tdm1rs 
            tcm2[i:j,i:j,i:j,i:j] = dm2

        # Cumulant decomposition so I only have to do one jk call for orbrot response
        # The only rules are 1) the sectors that you think are zero must really be zero, and
        #                    2) you subtract here what you add later
        tdm1s = np.einsum ('r,rspq->spq', self.weights, tdm1rs)
        cdm1s = np.einsum ('r,rsqp->spq', self.weights, self.casdm1rs)
        tcm2 -= np.multiply.outer (tdm1s[0] + tdm1s[1], cdm1s[0] + cdm1s[1])
        tcm2 += np.multiply.outer (tdm1s[0], cdm1s[0]).transpose (0,3,2,1)
        tcm2 += np.multiply.outer (tdm1s[1], cdm1s[1]).transpose (0,3,2,1)

        # Two transposes 
        tdm1rs += tdm1rs.transpose (0,1,3,2) 
        tcm2 += tcm2.transpose (1,0,3,2)        
        tcm2 += tcm2.transpose (2,3,0,1)        

        return tdm1rs, tcm2    

    def get_veff_Heff (self, odm1s, tdm1rs):
        ''' Compute first-order effective potential (relevant to the orbital-rotation sector of the
        Hessian-vector product) and first-order effective 1-body Hamiltonian operator (relevant to
        the CI-rotation sector of the Hessian-vector product) from first-order effective density
        matrices. "First-order" means proportional to one power of the orbital/CI-rotation step
        vector.

        Args:
            odm1s : ndarray of shape (2,nmo,nmo)
                Pre-symmetrization effective spin-separated state-averaged 1-RDM from the orbital
                rotation part of the step vector
            tdm1rs : ndarray of shape (nroots,2,ncas,ncas)
                Effective spin-separated 1-RDMs from the CI rotation part of the step vector,
                separated by root

        Returns:
            veff_mo : ndarray of shape (nmo,nmo)
                Spin-symmetric effective 1-body potential, including the effects of both the
                orbital and CI parts of the step vector
            h1frs : list of length nfrags of ndarrays
                ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i]).
                Effective one-electron Hamiltonian amplitudes for each state and fragment + h.c.
                Includes the effects of orbital rotation and CI rotation of other fragments (i.e.,
                omits double-counting).
        '''

        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        ncore, nocc, nroots = self.ncore, self.nocc, self.nroots
        tdm1s_sa = np.einsum ('rspq,r->spq', tdm1rs, self.weights)
        dm1s_mo = odm1s + odm1s.transpose (0,2,1)
        dm1s_mo[:,ncore:nocc,ncore:nocc] += 2*tdm1s_sa
        mo = self.mo_coeff
        moH = mo.conjugate ().T
        t1 = lib.logger.timer (self.las, 'LASSCF get_veff_Heff 1', *t0)
        
        # Overall veff for gradient: the one and only jk call per microcycle that I will allow.
        veff_mo = self.get_veff (dm1s_mo=dm1s_mo)
        t2 = lib.logger.timer (self.las, 'LASSCF get_veff_Heff 2', *t1)
        veff_mo = self.split_veff (veff_mo, dm1s_mo)
        t3 = lib.logger.timer (self.las, 'LASSCF get_veff_Heff 3', *t2)

        # Core-orbital-effect only for individual CI problems
        odm1s_core = np.copy (odm1s)
        odm1s_core[:,ncore:nocc,:] = 0.0
        odm1s_core += odm1s_core.transpose (0,2,1)
        err_dm1s = odm1s_core - dm1s_mo
        # Deal with nonsymmetric eri: Coulomb part
        err_dm1s = err_dm1s[:,:,ncore:nocc] * 2.0
        err_dm1s[:,ncore:nocc,:] /= 2.0
        veff_ci = np.tensordot (err_dm1s, self.eri_paaa, axes=2)
        veff_ci += veff_ci[::-1,:,:]
        veff_ci -= np.tensordot (err_dm1s, self.eri_paaa, axes=((1,2),(0,3)))
        # Deal with nonsymmetric eri: exchange part
        veff_ci += veff_ci.transpose (0,2,1)
        veff_ci /= 2.0
        veff_ci += veff_mo[:,ncore:nocc,ncore:nocc]
        t4 = lib.logger.timer (self.las, 'LASSCF get_veff_Heff 4', *t3)
        
        # SO, individual CI problems!
        # 1) There is NO constant term. Constant terms immediately drop out via the ugg defs!
        # 2) veff_ci is correctfor the orbrots, so long as I don't explicitly add h.c. at the end
        # 3) If I don't add h.c., then the (non-self) mf effect of the 1-tdms needs to be 2x strong
        # 4) Of course, self-interaction (from both 1-odms and 1-tdms) needs to be eliminated
        # 5) I do the latter by copying the eris, rather than the tdms, in case nroots is large
        h1frs = [np.zeros ((nroots, 2, nlas, nlas), dtype=self.dtype) for nlas in self.ncas_sub]
        eri_tmp = self.eri_cas.copy ()
        for isub, nlas in enumerate (self.ncas_sub):
            i = sum (self.ncas_sub[:isub])
            j = i + nlas
            h1frs[isub][:,:,:,:] = veff_ci[None,:,i:j,i:j]
            eri_tmp[:,:,:,:] = self.eri_cas[:,:,:,:]
            eri_tmp[i:j,i:j,:,:] = 0.0
            err_h1rs = 2.0 * np.tensordot (tdm1rs, eri_tmp, axes=2) 
            err_h1rs += err_h1rs[:,::-1] # ja + jb
            eri_tmp[:,:,:,:] = self.eri_cas[:,:,:,:]
            eri_tmp[i:j,:,:,i:j] = 0.0
            err_h1rs -= 2.0 * np.tensordot (tdm1rs, eri_tmp, axes=((2,3),(0,3)))
            #err_dm1rs = 2 * (tdm1frs.sum (0) - tdm1rs)
            #err_h1rs = np.tensordot (err_dm1rs, self.eri_cas, axes=2)
            #err_h1rs += err_h1rs[:,::-1] # ja + jb
            #err_h1rs -= np.tensordot (err_dm1rs, self.eri_cas, axes=((2,3),(0,3)))
            h1frs[isub][:,:,:,:] += err_h1rs[:,:,i:j,i:j]
        t5 = lib.logger.timer (self.las, 'LASSCF get_veff_Heff 5', *t4)
        
        return veff_mo, h1frs
        
    def get_veff (self, dm1s_mo=None):
        mo = self.mo_coeff
        moH = mo.conjugate ().T
        nmo = mo.shape[-1]
        dm1_mo = dm1s_mo.sum (0)
        dm1_ao = np.dot (mo, np.dot (dm1_mo, moH))
        veff_ao = np.squeeze (self.las.get_veff (dm=dm1_ao))
        return np.dot (moH, np.dot (veff_ao, mo))

    def split_veff (self, veff_mo, dm1s_mo):
        veff_c = veff_mo.copy ()
        ncore = self.ncore
        nocc = self.nocc
        sdm = dm1s_mo[0] - dm1s_mo[1]
        sdm_ra = sdm[:,ncore:nocc]
        sdm_ar = sdm[ncore:nocc,:].copy ()
        sdm_ar[:,ncore:nocc] = 0.0
        veff_s = np.zeros_like (veff_c)
        vk_pa = veff_s[:,ncore:nocc]
        for p, v1 in enumerate (vk_pa):
            praa = self.cas_type_eris.ppaa[p]
            para = self.cas_type_eris.papa[p]
            paaa = praa[ncore:nocc]
            v1[:]  = np.tensordot (sdm_ra, praa, axes=2)
            v1[:] += np.tensordot (sdm_ar, para, axes=2)
        veff_s[:,:] *= -0.5
        vk_aa = vk_pa[ncore:nocc]
        veff_s[ncore:nocc,:] = vk_pa.T
        assert (np.allclose (veff_s, veff_s.T)), vk_aa-vk_aa.T
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)

    def _matvec (self, x):
        log = lib.logger.new_logger (self.las, self.las.verbose)
#        extra_timing = getattr (self.las, '_extra_hessian_timing', False)
        extra_timing = getattr (self.las, '_extra_hessian_timing', True)
        extra_timer = log.timer if extra_timing else log.timer_debug1
        t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        kappa1, ci1 = self.ugg.unpack (x)
        t1 = extra_timer ('LASSCF sync Hessian operator 1: unpack', *t0)

        # Effective density matrices, veffs, and overlaps from linear response
        odm1s = -np.dot (self.dm1s, kappa1)
        ocm2 = -np.dot (self.cascm2, kappa1[self.ncore:self.nocc])
        tdm1rs, tcm2 = self.make_tdm1s2c_sub (ci1)
        t1 = extra_timer ('LASSCF sync Hessian operator 2: effective density matrices', *t1)
        veff_prime, h1s_prime = self.get_veff_Heff (odm1s, tdm1rs)
        t1 = extra_timer ('LASSCF sync Hessian operator 3: effective potentials', *t1)

        # Responses!
        kappa2 = self.orbital_response (kappa1, odm1s, ocm2, tdm1rs*2, tcm2*2, veff_prime)
        t1 = extra_timer ('LASSCF sync Hessian operator 4: (Hx)_orb', *t1)
        ci2 = self.ci_response_offdiag (kappa1, h1s_prime)
        t1 = extra_timer ('LASSCF sync Hessian operator 5: (Hx)_CI offdiag', *t1)
        ci2 = [[x+y for x,y in zip (xr, yr)] for xr, yr in zip (ci2, self.ci_response_diag (ci1))]
        t1 = extra_timer ('LASSCF sync Hessian operator 6: (Hx)_CI diag', *t1)

        # LEVEL SHIFT!!
        kappa3, ci3 = self.ugg.unpack (self.level_shift * x)
        kappa2 += kappa3
        ci2 = [[x+y for x,y in zip (xr, yr)] for xr, yr in zip (ci2, ci3)]
        t1 = extra_timer ('LASSCF sync Hessian operator 7: level shift', *t1)

        Hx = self.ugg.pack (kappa2/2, ci2)
        t1 = extra_timer ('LASSCF sync Hessian operator 8: pack', *t1)
        t0 = log.timer ('LASSCF sync Hessian operator total', *t0)
        return Hx

    _rmatvec = _matvec # Hessian is Hermitian in this context!

    def orbital_response (self, kappa, odm1s, ocm2, tdm1rs, tcm2, veff_prime):
        '''Compute the orbital-response sector of the Hessian-vector product. It's conceptually
        pretty simple:

        Hx_pq = F'_pq - F'_qp + .5*(F_pr k_rq - k_pr F_rq)
        F'_pq = h_pr D'_qr + g_prst d'_qrst

        Since we use the cumulant decomposition:

        d'_pqrs = l'_pqrs + D'_pq D_rs + D_pq D'_rs
                  - .5*(D[s]'_ps D[s]_qr + D[s]_ps D[s]'_qr)

        where [s] means spin index, we find that

        F'_pq = h_pr D_qr + veff[s]_pr D'[s]_qr + veff'[s]_pr D[s]_qr + g_prst l'_qrst

        where veff is the effective potential from the zeroth-order 1-RDMs and veff' is that from
        the first-order 1-RDMs.

        Args:
            kappa : ndarray of shape (nmo,nmo)
                Unpacked orbital-rotation step vector
            odm1s : ndarray of shape (2,nmo,nmo)
                Pre-symmetrization effective spin-separated state-averaged 1-RDM from the orbital
                rotation part of the step vector
            ocm2 : ndarray of shape (ncas,ncas,ncas,nmo)
                Pre-symmetrization spin-summed state-averaged effective cumulant of the 2-RDM
                from the orbital-rotation part of the step vector
            tdm1rs : ndarray of shape (nroots,2,ncas,ncas)
                Effective spin-separated 1-RDMs from the CI rotation part of the step vector,
                separated by root
            tcm2 : ndarray of shape (ncas,ncas,ncas,nncas)
                Spin-summed state-averaged effective cumulant of the 2-RDM from the CI rotation
                part of the step vector
            veff_prime : ndarray of shape (2,nmo,nmo)
                Spin-separated state-averaged effective potential proportional to the first power
                of the step vector (all sectors)

        Returns:
            kappa2 : ndarray of shape (nmo,nmo)
                Contains the unpacked orbital-rotation sector of the Hessian-vector product.
        '''
        gorb = self.orbital_response_1cum (kappa, odm1s, ocm2, tdm1rs, tcm2, veff_prime)
        gorb = self.orbital_response_2cum (kappa, odm1s, ocm2, tdm1rs, tcm2, veff_prime, gorb)
        return gorb

    def orbital_response_1cum (self, kappa, odm1s, ocm2, tdm1rs, tcm2, veff_prime):
        ncore, nocc = self.ncore, self.nocc
        # I put off + h.c. until now in order to make other things more natural
        odm1s += odm1s.transpose (0,2,1)
        ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)
        ocm2 += ocm2.transpose (2,3,0,1)
        # Effective density matrices
        edm1s = odm1s
        edm1s[:,ncore:nocc,ncore:nocc] += np.einsum ('rspq,r->spq', tdm1rs, self.weights)
        ecm2 = ocm2 + tcm2
        # Evaluate hx = (F2..x) - (F2..x).T + (F1.x) - (F1.x).T
        fock1  = self.h1s[0] @ edm1s[0] + self.h1s[1] @ edm1s[1]
        fock1 += veff_prime[0] @ self.dm1s[0] + veff_prime[1] @ self.dm1s[1]
        fock1[ncore:nocc,ncore:nocc] += np.tensordot (self.eri_cas, ecm2, axes=((1,2,3),(1,2,3)))
        fock1 += (np.dot (self.fock1, kappa) - np.dot (kappa, self.fock1)) / 2
        return fock1 - fock1.T

    def orbital_response_2cum (self, kappa, odm1s, ocm2, tdm1frs, tcm2, veff_prime, gorb):
        ''' 1cum does everything except va/ac degrees of freedom
        (c: closed; a: active; v: virtual; p: any) '''

        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        f1_prime = np.zeros ((self.nmo, self.nmo), dtype=self.dtype)
        # (H.x_va)_pp, (H.x_ac)_pp sector
        for p, f1 in enumerate (f1_prime):
            praa = self.cas_type_eris.ppaa[p]
            para = self.cas_type_eris.papa[p]
            paaa = praa[ncore:nocc]
            # g_pabc d_qabc + g_prab d_qrab + g_parb d_qarb + g_pabr d_qabr (Formal)
            #        d_cbaq          d_abqr          d_aqbr          d_qabr (Symmetry of ocm2)
            # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbcr d_qbcr (Relabel)
            #                                                 g_pbrc        (Symmetry of eri)
            # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbrc d_qbcr (Final)
            for i, j in ((0, ncore), (nocc, nmo)): # Don't double-count
                ra, ar, cm = praa[i:j], para[:,i:j], ocm2[:,:,:,i:j]
                f1[i:j] += np.tensordot (paaa, cm, axes=((0,1,2),(2,1,0))) # last index external
                f1[ncore:nocc] += np.tensordot (ra, cm, axes=((0,1,2),(3,0,1))) # third index external
                f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(0,3,2))) # second index external
                f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(1,3,2))) # first index external

        # (H.x_aa)_va, (H.x_aa)_ac
        ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)
        ocm2 += ocm2.transpose (2,3,0,1)
        ecm2 = ocm2 + tcm2
        f1_prime[:ncore,ncore:nocc] += np.tensordot (self.eri_paaa[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
        f1_prime[nocc:,ncore:nocc] += np.tensordot (self.eri_paaa[nocc:], ecm2, axes=((1,2,3),(1,2,3)))
        return gorb + (f1_prime - f1_prime.T)


    def ci_response_offdiag (self, kappa1, h1frs_prime):
        '''Compute part of the CI rotation sector of the Hessian-vector product corresponding
        to off-diagonal blocks of the Hessian matrix; i.e., for a given fragment block of the
        Hessian-vector product, the input step vector omits CI degrees of freedom of that fragment,
        but includes CI degrees of freedom for all other fragments as well as orbital-rotation
        degrees of freedom.

        Args:
            kappa1 : ndarray of shape (nmo,nmo)
                Unpacked orbital-rotation step vector
            h1frs : list of length nfrags of ndarrays
                ith element has shape (nroots,2,ncas_sub[i],ncas_sub[i]).
                Effective one-electron Hamiltonian amplitudes for each state and fragment + h.c.
                Includes the effects of orbital rotation and CI rotation of other fragments (i.e.,
                omits double-counting).

        Returns:
            Kci0 : list (length = nfrags) of lists (length = nroots) of ndarrays
                Contains unpacked CI sector of partial Hessian-vector product
        '''
        # Since h1frs contains + h.c., I do NOT explicitly add + h.c. in this function
        ncore, nocc, ncas_sub, nroots = self.ncore, self.nocc, self.ncas_sub, self.nroots
        kappa1_cas = kappa1[ncore:nocc,:]
        h1frs = [np.zeros_like (h1) for h1 in h1frs_prime]
        h1_core = -np.tensordot (kappa1_cas, self.h1s_cas, axes=((1),(1))).transpose (1,0,2)
        h1_core += h1_core.transpose (0,2,1)
        h2 = -np.tensordot (kappa1_cas, self.eri_paaa, axes=1)
        h2 += h2.transpose (2,3,0,1)
        h2 += h2.transpose (1,0,3,2)
        # ^ h2 should also include + h.c.
        for j, casdm1s in enumerate (self.casdm1rs):
            for i, (h1rs, h1rs_prime) in enumerate (zip (h1frs, h1frs_prime)):
                k = sum (ncas_sub[:i])
                l = k + ncas_sub[i]
                h1s, h1s_prime = h1rs[j], h1rs_prime[j]
                dm1s = casdm1s.copy ()
                dm1s[:,k:l,k:l] = 0.0 # no double-counting
                dm1 = dm1s.sum (0)
                h1s[:,:,:] = h1_core[:,k:l,k:l].copy ()
                h1s[:,:,:] += np.tensordot (h2, dm1, axes=2)[None,k:l,k:l]
                h1s[:,:,:] -= np.tensordot (dm1s, h2, axes=((1,2),(2,1)))[:,k:l,k:l]
                #h1s[:,:,:] += h1s.transpose (0,2,1)
                h1s[:,:,:] += h1s_prime[:,:,:]
        Kci0 = self.Hci_all (None, h1frs, h2, self.ci)
        Kci0 = [[Kc - c*(c.dot (Kc)) for Kc, c in zip (Kcr, cr)]
                for Kcr, cr in zip (Kci0, self.ci)]
        # ^ The definition of the unitary group generator compels you to do this always!!!
        return Kci0

    def ci_response_diag (self, ci1):
        '''Compute part of the CI response sector of the Hessian-vector product corresponding
        to diagonal blocks of the Hessian matrix; i.e., for a given fragment block of the Hessian-
        vector product, the input step vector includes ONLY CI degrees of freedom for THAT
        FRAGMENT.

        Args:
            ci1 : list (length = nfrags) of lists (length = nroots) of ndarrays
                Contains unpacked CI sector of input step vector

        Returns:
            ci2 : list (length = nfrags) of lists (length = nroots) of ndarrays
                Contains unpacked CI sector of partial Hessian-vector product
        '''
        # IMPORTANT: this disagrees with PySCF, but I still think it's right and PySCF is wrong
        ci1HmEci0 = [[c.dot (Hci) for c, Hci in zip (cr, Hcir)] 
                     for cr, Hcir in zip (ci1, self.hci0)]
        s01 = [[c1.dot (c0) for c1,c0 in zip (c1r, c0r)] for c1r, c0r in zip (ci1, self.ci)]
        ci2 = self.Hci_all ([[-e for e in er] for er in self.e0], self.h1frs, self.eri_cas, ci1)
        ci2 = [[x-(y*z) for x,y,z in zip (xr,yr,zr)] for xr,yr,zr in zip (ci2, self.ci, ci1HmEci0)]
        ci2 = [[x-(y*z) for x,y,z in zip (xr,yr,zr)] for xr,yr,zr in zip (ci2, self.hci0, s01)]
        return [[x*2 for x in xr] for xr in ci2]

    def get_prec (self):
        '''Obtain the preconditioner for conjugate-gradient descent using a second-order power
        series of the energy from a given LAS-state keyframe (a single "macrocycle"). In general,
        the preconditioner should approximate multiplication by the matrix-inverse of the Hessian.
        Here, however, we also use it to identify and mask degrees of freedom along which the
        quadratically-approximated energy is numerically unstable.

        N.B. to future developers: an "exact" inverted-Hessian preconditioner is actually not
        desirable, because a failure of optimization is more likely due to the unsuitability of a
        quadratic power series in fundamentally periodic variables. I.O.W., we can't get too hung
        up on solving Ax=b, because Ax=b is an approximate equation in the first place. The actual
        goal is to minimize successive keyframe (aka "macrocycle" aka "trial") energies.

        Returns:
            prec_op : LinearOperator
                Approximately the inverse of the Hessian
        '''
        log = lib.logger.new_logger (self.las, self.las.verbose)
        Hdiag = self._get_Hdiag () + self.level_shift
        Hdiag[np.abs (Hdiag)<1e-8] = 1e-8
        # The quadratic power series is a bad approximation if the magnitude of the gradient in
        # the current keyframe is such that we will tend to predict steps with magnitude greater
        # than pi (a step of exactly pi transposes two states). This preconditioner should
        # mask out the corresponding degrees of freedom
        g_vec = self.get_grad ()
        b = linalg.norm (g_vec)
        probe_x0 = b/Hdiag
        log.debug ('|probe_x0| / ndeg = %g', linalg.norm (probe_x0) / len (probe_x0))
        ndeg = len (probe_x0)
        idx_unstable = np.abs (probe_x0) > np.pi
        # We can't mask everything, because that behavior would obfuscate the problem
        # If NO stable D.O.F. exist, then keyframe is just bad and it has to be handled upstream
        ndeg_unstable = np.count_nonzero (idx_unstable)
        ndeg_stable = np.count_nonzero (~idx_unstable)
        g_unst = linalg.norm (g_vec[idx_unstable]) if ndeg_unstable else 0
        if ndeg_stable and (round (g_unst/b, 2) < 1):
            Hdiag[idx_unstable] = np.inf
            ndeg_unstable = ndeg - ndeg_stable
            log.debug ('%d/%d d.o.f. masked in LASSCF sync preconditioner (masked gradient = %g)',
                       ndeg_unstable, ndeg, g_unst)
        else:
            log.warn ('LASSCF encountered an unmaskable instability; calculation may not converge')
        return self.PrecOp (Hdiag, log)

    class PrecOp (sparse_linalg.LinearOperator):
        def __init__(self, Hdiag, log):
            self.Hdiag = Hdiag
            self.log = log
            sparse_linalg.LinearOperator.__init__(self, dtype=Hdiag.dtype,
                                                  shape=(len (Hdiag), len (Hdiag)))
        def _matvec (self, x):
            t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
            Mx = x/self.Hdiag
            self.log.timer ('LASSCF sync preconditioner call', *t0)
            return Mx

    def _get_Horb_diag_presymm_fock (self):
        ncore, nocc = self.ncore, self.nocc
        fock = np.stack ([np.diag (h) for h in list (self.h1s)], axis=0)
        num = np.stack ([np.diag (d) for d in list (self.dm1s)], axis=0)
        Horb_diag = sum ([np.multiply.outer (f,n) for f,n in zip (fock, num)])
        Horb_diag -= np.diag (self.fock1)[None,:]
        # This is where I stop unless I want to add the split-c and split-x terms
        # Split-c and split-x, for inactive-external rotations, requires I calculate a bunch
        # of extra eris (g^aa_ii, g^ai_ai)
        return Horb_diag

    def _get_Horb_diag_presymm_eri_F2aaaa (self):
        nmo, ncore, nocc = self.nmo, self.ncore, self.nocc
        h2 = self.eri_paaa[ncore:nocc]
        d1s = self.casdm1s
        d2 = self.casdm2
        fock2 = lib.einsum ('pqij,rsij->prqs', h2, d2)
        fock2 += lib.einsum ('piqj,risj->prqs', h2, d2)
        fock2 += lib.einsum ('pjiq,rjis->prqs', h2, d2)
        Horb_aa = np.diagonal (fock2, axis1=0, axis2=2).copy ()
        Horb_aa -= np.diagonal (fock2, axis1=0, axis2=3)
        Horb_aa = np.diagonal (Horb_aa, axis1=0, axis2=1).copy ()
        # We double-counted g Da Da. Gotta subtract
        v1s = -lib.einsum ('skl,kjil->sij', d1s, h2)
        v1s += lib.einsum ('ijkl,kl->ij', h2, d1s.sum (0))[None,:,:]
        fock = np.stack ([np.diag (h) for h in list (v1s)], axis=0)
        num = np.stack ([np.diag (d) for d in list (d1s)], axis=0)
        Horb_aa -= sum ([np.multiply.outer (f,n) for f,n in zip (fock, num)])
        Horb_diag = np.zeros ((nmo, nmo), dtype=self.dtype)
        Horb_diag[ncore:nocc,ncore:nocc] = Horb_aa
        return Horb_diag

    def _get_Horb_diag_presymm_eri_F2ujuj (self):
        nmo, ncore, nocc = self.nmo, self.ncore, self.nocc
        d1s = self.casdm1s
        d1 = d1s.sum (0)
        d1_ubub = 2*lib.einsum ('ab,ac->bca', d1, d1)
        d1_uubb = -lib.einsum ('sab,sac->bca', d1s, d1s)
        d1_ubub += d1_uubb
        d2 = self.cascm2
        d2T = d2 + d2.transpose (0,1,3,2)
        d2_aabb = np.diagonal (d2,axis1=0,axis2=1) + d1_uubb
        d2_abab = np.diagonal (d2T,axis1=0,axis2=2) + d1_ubub
        Horb_diag = np.zeros ((nmo, nmo), dtype=self.dtype)
        j_pc = self.cas_type_eris.j_pc
        k_pc = self.cas_type_eris.k_pc
        # F2pipi
        Horb_diag[:,:ncore] = 6*k_pc - 2*j_pc
        # F2uaua
        Horb_ua = Horb_diag[:,ncore:nocc]
        for u in range (nmo):
            if (u>=ncore) and (u<nocc): continue
            uubb = self.cas_type_eris.ppaa[u][u]
            ubub = self.cas_type_eris.papa[u][:,u]
            Horb_ua[u] = lib.einsum ('bc,bca->a',uubb,d2_aabb)
            Horb_ua[u] += lib.einsum ('bc,bca->a',ubub,d2_abab)
        return Horb_diag

    def _get_Horb_diag_presymm_eri_F2aiia (self):
        # Both indices must have nonzero density matrix for this term
        nmo, ncore, nocc = self.nmo, self.ncore, self.nocc
        Horb_diag = np.zeros ((nmo, nmo), dtype=self.dtype)
        dm1_cas = self.dm1s[:,ncore:nocc,ncore:nocc].sum (0)
        Horb_pa = Horb_diag[:,ncore:nocc]
        for i in range (ncore):
            iibb = self.cas_type_eris.ppaa[i][i]
            ibib = self.cas_type_eris.papa[i][:,i]
            # 1 factor of 2 from ERI permutations
            # 1 factor of 2 from rdm1 of core orbitals
            Horb_pa[i] += 4*lib.einsum ('ab,ab->a',ibib,dm1_cas)
            # 1 factor of 2 from ERI permutations
            Horb_pa[i] -= lib.einsum ('ab,ab->a',ibib+iibb,dm1_cas)
        # electron1 <-> electron2
        # not to be confused with the final p,q <-> q,p symmetrization
        # This is just to fill out the nonzero elements of the pre-symmetrized object
        Horb_diag += Horb_diag.T
        return Horb_diag

    def _get_Horb_diag_presymm (self):
        Horb_diag = self._get_Horb_diag_presymm_fock ()
        self._init_eri_()
        Horb_diag += self._get_Horb_diag_presymm_eri_F2aaaa ()
        Horb_diag += self._get_Horb_diag_presymm_eri_F2ujuj ()
        Horb_diag -= self._get_Horb_diag_presymm_eri_F2aiia ()
        return Horb_diag

    def _get_Horb_diag (self):
        Horb_diag = self._get_Horb_diag_presymm ()
        Horb_diag += Horb_diag.T
        return Horb_diag[self.ugg.uniq_orb_idx]*.5

    def _get_Hci_diag (self):
        Hci_diag = []
        for ix, (fcibox, norb, nelec, h1rs, csf_list) in enumerate (zip (self.fciboxes, 
         self.ncas_sub, self.nelecas_sub, self.h1frs, self.ugg.ci_transformers)):
            if ix in self.ugg.frozen_ci: continue
            i = sum (self.ncas_sub[:ix])
            j = i + norb
            h2 = self.eri_cas[i:j,i:j,i:j,i:j]
            hdiag_csf_list = fcibox.states_make_hdiag_csf (h1rs, h2, norb, nelec)
            for csf, hdiag_csf in zip (csf_list, hdiag_csf_list):
                Hci_diag.append (csf.pack_csf (hdiag_csf))
        return Hci_diag

    def _get_Hdiag (self):
        return np.concatenate ([self._get_Horb_diag ()] + self._get_Hci_diag ())

    def update_mo_ci_eri (self, x, h2eff_sub):
        log = lib.logger.new_logger(self.las, self.las.verbose)
        t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        kappa, dci = self.ugg.unpack (x)
        umat = linalg.expm (kappa/2)
        t0=log.timer('update_init',*t0)
        # The 1/2 here is because my actual variables are just the lower-triangular
        # part of kappa, or equivalently 1/2 k^p_q (E^p_q - E^q_p). I can simplify
        # this to k^p_q E^p_q when evaluating derivatives, but not when exponentiating,
        # because the operator part has to be anti-hermitian.
        mo1 = self._update_mo (umat)
        t0=log.timer('update_mo',*t0)
        ci1 = self._update_ci (dci)
        t0=log.timer('update_ci',*t0)
        h2eff_sub = self._update_h2eff_sub (mo1, umat, h2eff_sub)
        t0=log.timer('update_h2eff_sub',*t0)
        return mo1, ci1, h2eff_sub

    def _update_mo (self, umat):
        mo1 = self.mo_coeff @ umat
        if hasattr (self.mo_coeff, 'orbsym'):
            mo1 = lib.tag_array (mo1, orbsym=self.mo_coeff.orbsym)
        return mo1

    def _update_ci (self, dci):
        ci1 = []
        for c_r, dc_r in zip (self.ci, dci):
            ci1_r = []
            for c, dc in zip (c_r, dc_r):
                dc[:] -= c * c.dot (dc)
                phi = linalg.norm (dc)
                cosp = np.cos (phi)
                if np.abs (phi) > 1e-8: sinp = np.sin (phi) / phi
                else: sinp = 1 # as precise as it can be w/ 64 bits
                c1 = cosp*c + sinp*dc
                assert (np.isclose (linalg.norm (c1), 1))
                ci1_r.append (c1)
            ci1.append (ci1_r)
        return ci1

      

    def _update_h2eff_sub (self, mo1, umat, h2eff_sub):
        return self.las.ao2mo (mo1)

    def get_grad (self):
        gorb = self.fock1 - self.fock1.T
        gci = [[2*hci0 for hci0 in hci0r] for hci0r in self.hci0]
        return self.ugg.pack (gorb, gci)

    def get_gx (self):
        gorb = self.fock1 - self.fock1.T
        gx = gorb[self.ugg.get_gx_idx ()]
        return gx

def _init_df_(h_op):
    from pyscf.mcscf.las.lasci import _DFLASCI
    if isinstance (h_op.las, _DFLASCI):
        h_op.with_df = h_op.las.with_df
        if h_op.bPpj is None: h_op.bPpj = np.ascontiguousarray (
                h_op.las.cderi_ao2mo (h_op.mo_coeff, h_op.mo_coeff[:,:h_op.nocc],
                compact=False))


density_fit = lasci.density_fit
def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = density_fit (las, with_df = mf.with_df) 
    return las

def get_grad (las, mo_coeff=None, ci=None, ugg=None, h1eff_sub=None, h2eff_sub=None,
              veff=None, dm1s=None):
    '''Return energy gradient for orbital rotation and CI relaxation.

    Args:
        las : instance of :class:`LASSCFNoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        ugg : instance of :class:`LASSCF_UnitaryGroupGenerators`
        h1eff_sub : list (length=nfrags) of list (length=nroots) of ndarray
            Contains effective one-electron Hamiltonians experienced by each fragment
            in each state
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis

    Returns:
        gorb : ndarray of shape (ugg.nvar_orb,)
            Orbital rotation gradients as a flat array
        gci : ndarray of shape (sum(ugg.ncsf_sub),)
            CI relaxation gradients as a flat array
        gx : ndarray
            Orbital rotation gradients for temporarily frozen orbitals in the "LASSCF" problem
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if ugg is None: ugg = las.get_ugg (mo_coeff, ci)
    if dm1s is None: dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if veff is None:
        veff = las.get_veff (dm = dm1s)
    if h1eff_sub is None: h1eff_sub = las.get_h1eff (mo_coeff, ci=ci, veff=veff,
                                                     h2eff_sub=h2eff_sub)

    if callable (getattr (las, 'get_grad_orb', None)):
        gorb = las.get_grad_orb (mo_coeff=mo_coeff, ci=ci, h2eff_sub=h2eff_sub, veff=veff, dm1s=dm1s)
    else:
        gorb = get_grad_orb (las, mo_coeff=mo_coeff, ci=ci, h2eff_sub=h2eff_sub, veff=veff, dm1s=dm1s)
    if callable (getattr (las, 'get_grad_ci', None)):
        gci = las.get_grad_ci (mo_coeff=mo_coeff, ci=ci, h1eff_sub=h1eff_sub, h2eff_sub=h2eff_sub,
                               veff=veff)
    else:
        gci = get_grad_ci (las, mo_coeff=mo_coeff, ci=ci, h1eff_sub=h1eff_sub, h2eff_sub=h2eff_sub,
                           veff=veff)

    idx = ugg.get_gx_idx ()
    gx = gorb[idx]
    gint = ugg.pack (gorb, gci)
    gorb = gint[:ugg.nvar_orb]
    gci = gint[ugg.nvar_orb:]
    return gorb, gci, gx.ravel ()

def get_grad_orb (las, mo_coeff=None, ci=None, h2eff_sub=None, veff=None, dm1s=None, hermi=-1):
    '''Return energy gradient for orbital rotation.

    Args:
        las : instance of :class:`LASSCFNoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis
        hermi : integer
            Control (anti-)symmetrization. 0 means to return the effective Fock matrix,
            F1 = h.D + g.d. -1 means to return the true orbital-rotation gradient, which is skew-
            symmetric: gorb = F1 - F1.T. +1 means to return the symmetrized effective Fock matrix,
            (F1 + F1.T) / 2. The factor of 2 difference between hermi=-1 and the other two options
            is intentional and necessary.

    Returns:
        gorb : ndarray of shape (nmo,nmo)
            Orbital rotation gradients as a square antihermitian array
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if dm1s is None: dm1s = las.make_rdm1s (mo_coeff=mo_coeff, ci=ci)
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if veff is None:
        veff = las.get_veff (dm = dm1s)
    nao, nmo = mo_coeff.shape
    ncore = las.ncore
    ncas = las.ncas
    nocc = las.ncore + las.ncas
    smo_cas = las._scf.get_ovlp () @ mo_coeff[:,ncore:nocc]
    smoH_cas = smo_cas.conj ().T

    # The orbrot part
    h1s = las.get_hcore ()[None,:,:] + veff
    f1 = h1s[0] @ dm1s[0] + h1s[1] @ dm1s[1]
    f1 = mo_coeff.conjugate ().T @ f1 @ las._scf.get_ovlp () @ mo_coeff
    # ^ I need the ovlp there to get dm1s back into its correct basis
    casdm2 = las.make_casdm2 (ci=ci)
    casdm1s = np.stack ([smoH_cas @ d @ smo_cas for d in dm1s], axis=0)
    casdm1 = casdm1s.sum (0)
    casdm2 -= np.multiply.outer (casdm1, casdm1)
    casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
    casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)
    eri = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
    eri = lib.numpy_helper.unpack_tril (eri).reshape (nmo, ncas, ncas, ncas)
    f1[:,ncore:nocc] += np.tensordot (eri, casdm2, axes=((1,2,3),(1,2,3)))

    if hermi == -1:
        return f1 - f1.T
    elif hermi == 1:
        return .5*(f1+f1.T)
    elif hermi == 0:
        return f1
    else:
        raise ValueError ("kwarg 'hermi' must = -1, 0, or +1")

def get_grad_ci (las, mo_coeff=None, ci=None, h1eff_sub=None, h2eff_sub=None, veff=None):
    '''Return energy gradient for CI relaxation.

    Args:
        las : instance of :class:`LASSCFNoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        h1eff_sub : list (length=nfrags) of list (length=nroots) of ndarray
            Contains effective one-electron Hamiltonians experienced by each fragment
            in each state
        h2eff_sub : ndarray of shape (nmo,ncas**2*(ncas+1)/2)
            Contains ERIs (p1a1|a2a3), lower-triangular in the a2a3 indices
        veff : ndarray of shape (2,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis

    Returns:
        gci : list (length=nfrags) of list (length=nroots) of ndarray
            CI relaxation gradients in the shape of CI vectors
    '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
    if h1eff_sub is None: h1eff_sub = las.get_h1eff (mo_coeff, ci=ci, veff=veff,
                                                     h2eff_sub=h2eff_sub)
    gci = []
    for isub, (fcibox, h1e, ci0, ncas, nelecas) in enumerate (zip (
            las.fciboxes, h1eff_sub, ci, las.ncas_sub, las.nelecas_sub)):
        eri_cas = las.get_h2eff_slice (h2eff_sub, isub, compact=8)
        linkstrl = fcibox.states_gen_linkstr (ncas, nelecas, True)
        linkstr  = fcibox.states_gen_linkstr (ncas, nelecas, False)
        h2eff = fcibox.states_absorb_h1e(h1e, eri_cas, ncas, nelecas, .5)
        hc0 = fcibox.states_contract_2e(h2eff, ci0, ncas, nelecas, link_index=linkstrl)
        hc0 = [hc.ravel () for hc in hc0]
        ci0 = [c.ravel () for c in ci0]
        gci.append ([2.0 * (hc - c * (c.dot (hc))) for c, hc in zip (ci0, hc0)])
    return gci

class LASSCFNoSymm (lasci.LASCINoSymm):

    get_grad = get_grad
    get_grad_orb = get_grad_orb
    get_grad_ci = get_grad_ci
    as_scanner = mc1step.as_scanner
    _hop = LASSCF_HessianOperator
    _kern = kernel
    def get_hop (self, mo_coeff=None, ci=None, ugg=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ugg is None: ugg = self.get_ugg ()
        return self._hop (self, ugg, mo_coeff=mo_coeff, ci=ci, **kwargs)

    def kernel(self, mo_coeff=None, ci0=None, casdm0_fr=None, conv_tol_grad=None,
            assert_no_dupes=False, verbose=None, _kern=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None: ci0 = self.ci
        if verbose is None: verbose = self.verbose
        if conv_tol_grad is None: conv_tol_grad = self.conv_tol_grad
        if _kern is None: _kern = self._kern
        log = lib.logger.new_logger(self, verbose)

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        self.dump_flags(log)

        for fcibox in self.fciboxes:
            fcibox.verbose = self.verbose
            fcibox.stdout = self.stdout
            fcibox.nroots = self.nroots
            fcibox.weights = self.weights
        # TODO: local excitations and locally-impure states in LASSCF kernel
        do_warn=False
        if ci0 is not None:
            for i, ci0_i in enumerate (ci0):
                if ci0_i is None: continue
                for j, ci0_ij in enumerate (ci0_i):
                    if ci0_ij is None: continue
                    if np.asarray (ci0_ij).ndim>2:
                        do_warn=True
                        ci0_i[j] = ci0_ij[0]
        if do_warn: log.warn ("Discarding all but the first root of guess CI vectors!")

        self.converged, self.e_tot, self.e_states, self.mo_energy, self.mo_coeff, self.e_cas, \
                self.ci, h2eff_sub, veff = _kern(mo_coeff=mo_coeff, ci0=ci0, verbose=verbose, \
                casdm0_fr=casdm0_fr, conv_tol_grad=conv_tol_grad, assert_no_dupes=assert_no_dupes)

        self._finalize ()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy, h2eff_sub, veff

    _ugg = LASSCF_UnitaryGroupGenerators
    def get_ugg (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        return self._ugg (self, mo_coeff, ci)

    def fast_veffa (self, casdm1s_sub, bmPu, mo_coeff=None, ci=None):
        '''Compute the effective potential exerted by active electrons on the whole orbital space
        using integrals and density matrices stored in the MO basis. This only makes sense to
        do if density fitting is used and is not implemented with GPUs at present.

        Args:
            casdm1s_sub : list of ndarray of shape (2,nlas,nlas)
            bmPu : ndarray of shape (nao,naux,ncas) or None
                Cholesky vectors with one AO index transformed into active orbitals

        Kwargs:
            mo_coeff : ndarray of shape (nao,nmo)
            ci : nested list of ndarrays

        Returns:
            veff : ndarray of shape (2,nao,nao)
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        ncore = self.ncore
        ncas_sub = self.ncas_sub
        ncas = sum (ncas_sub)
        nocc = ncore + ncas
        nao, nmo = mo_coeff.shape
        mo_cas = mo_coeff[:,ncore:nocc]
        moH_cas = mo_cas.conjugate ().T
        moH_coeff = mo_coeff.conjugate ().T
        dma = linalg.block_diag (*[dm[0] for dm in casdm1s_sub])
        dmb = linalg.block_diag (*[dm[1] for dm in casdm1s_sub])
        casdm1s = np.stack ([dma, dmb], axis=0)
        escape = bmPu is None
        escape = escape or (not isinstance (self, _DFLASCI))
        escape = escape or (not isinstance (self.with_df._cderi, np.ndarray))
        if escape:
            dm1s = np.dot (mo_cas, np.dot (casdm1s, moH_cas)).transpose (1,0,2)
            return self.get_veff (dm = dm1s)
        casdm1 = casdm1s.sum (0)
        dm1 = np.dot (mo_cas, np.dot (casdm1, moH_cas))
        bPmn = self.with_df._cderi

        # vj
        dm_tril = dm1 + dm1.T - np.diag (np.diag (dm1.T))
        rho = np.dot (bPmn, lib.pack_tril (dm_tril))
        vj = lib.unpack_tril (np.dot (rho, bPmn))

        # vk
        vmPsu = np.dot (bmPu, casdm1s)
        vk = np.tensordot (vmPsu, bmPu, axes=((1,3),(1,2))).transpose (1,0,2)
        return vj[None,:,:] - vk

    lasci = lasci_ = lasci.LASCINoSymm.kernel

    def dump_flags (self, verbose=None, _method_name='LASSCF'):
        super().dump_flags (verbose=verbose, _method_name=_method_name)

    def check_sanity (self):
        super().check_sanity ()
        self.get_ugg () # constructor encounters impossible states and raises error

    #SV 
    def nuc_grad_method(self): 
        from pyscf.grad import lasscf 
        return lasscf.Gradients(self) 
 
    #SV 
    Gradients = nuc_grad_method 

class LASSCFSymm (lasci.LASCISymm, LASSCFNoSymm):

    as_scanner = mc1step.as_scanner
    get_veff = LASSCFNoSymm.get_veff
    _ugg = LASSCFSymm_UnitaryGroupGenerators

    def kernel(self, mo_coeff=None, ci0=None, casdm0_fr=None, verbose=None, assert_no_dupes=False):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci0 is None:
            ci0 = self.ci

        # Initialize/overwrite mo_coeff.orbsym. Don't pass ci0 because it's not the right shape
        lib.logger.info (self, ("LASSCF lazy hack note: lines below reflect the point-group "
                                "symmetry of the whole molecule but not of the individual "
                                "subspaces"))
        mo_coeff = self.mo_coeff = self.label_symmetry_(mo_coeff)
        return LASSCFNoSymm.kernel(self, mo_coeff=mo_coeff, ci0=ci0,
            casdm0_fr=casdm0_fr, verbose=verbose, assert_no_dupes=assert_no_dupes)

 
