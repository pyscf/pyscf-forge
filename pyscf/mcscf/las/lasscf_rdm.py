# RDM-based variant of LASSCF in which internal electronic structure of each
# localized active subspace is decoupled from orbital rotation and the kernel
# for obtaining RDMs given LAS Hamiltonians can be subclassed arbitrarily

import time
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg
from pyscf.mcscf.las import lasscf_sync_o0, _DFLASCI, addons, lasci
from pyscf.mcscf.las.lasscf_sync_o0 import MicroIterInstabilityException
from pyscf.csf_fci import csf_solver
from pyscf import lib, gto, scf, ao2mo
from pyscf.fci.direct_spin1 import _unpack_nelec

class LASSCF_UnitaryGroupGenerators (lasscf_sync_o0.LASSCF_UnitaryGroupGenerators):
    ''' spoof away CI degrees of freedom '''
    def __init__(self, las, mo_coeff, *args):
        lasscf_sync_o0.LASSCF_UnitaryGroupGenerators.__init__(
            self, las, mo_coeff, None)
    def _init_ci (self, las, mo_coeff, ci):
        pass
    def pack (self, kappa):
        return kappa[self.uniq_orb_idx]
    def unpack (self, x):
        kappa = np.zeros ((self.nmo, self.nmo), dtype=x.dtype)
        kappa[self.uniq_orb_idx] = x[:self.nvar_orb]
        kappa = kappa - kappa.T
        return kappa
    @property
    def ncsf_sub (self):
        return np.array ([0])

class LASSCFSymm_UnitaryGroupGenerators (LASSCF_UnitaryGroupGenerators):
    def __init__(self, las, mo_coeff, *args):
        lasscf_sync_o0.LASSCFSymm_UnitaryGroupGenerators.__init__(
            self, las, mo_coeff, None)
    _init_orb = lasscf_sync_o0.LASSCFSymm_UnitaryGroupGenerators._init_orb

class LASSCF_HessianOperator (lasscf_sync_o0.LASSCF_HessianOperator):
    def __init__(self, las, ugg, mo_coeff=None, casdm1frs=None, casdm2fr=None,
            ncore=None, ncas_sub=None, nelecas_sub=None, h2eff_sub=None, veff=None,
            do_init_eri=True, **kwargs):
        if mo_coeff is None: mo_coeff = las.mo_coeff
        if casdm1frs is None: casdm1frs = las.casdm1frs
        if casdm2fr is None: casdm2fr = las.casdm2fr
        if ncore is None: ncore = las.ncore
        if ncas_sub is None: ncas_sub = las.ncas_sub
        if nelecas_sub is None: nelecas_sub = las.nelecas_sub
        if h2eff_sub is None: h2eff_sub = las.get_h2eff (mo_coeff)
        self.las = las
        self.level_shift = las.ah_level_shift
        self.ugg = ugg
        self.mo_coeff = mo_coeff
        sparse_linalg.LinearOperator.__init__(self, dtype=self.mo_coeff.dtype,
                                              shape=(ugg.nvar_tot, ugg.nvar_tot))
        self.ncore = ncore
        self.ncas_sub = ncas_sub
        self.nelecas_sub = nelecas_sub
        self.ncas = ncas = sum (ncas_sub)
        self.nao = mo_coeff.shape[0]
        self.nmo = mo_coeff.shape[-1]
        self.nocc = ncore + ncas
        self.nroots = nroots = las.nroots
        self.weights = las.weights
        self.bPpj = None
        # Spoof away CI: fixed zeros
        self._tdm1rs = np.zeros ((nroots, 2, ncas, ncas))
        self._tcm2 = np.zeros ([ncas,]*4)

        self._init_dms_(casdm1frs, casdm2fr)
        self._init_ham_(h2eff_sub, veff)
        self._init_orb_()
        if do_init_eri: self._init_eri_()

    def _matvec (self, x):
        kappa1 = self.ugg.unpack (x)

        # Effective density matrices, veffs, and overlaps from linear response
        odm1s = -np.dot (self.dm1s, kappa1)
        ocm2 = -np.dot (self.cascm2, kappa1[self.ncore:self.nocc])
        veff_prime = self.get_veff_prime (odm1s)

        # Responses!
        kappa2 = self.orbital_response (kappa1, odm1s, ocm2, veff_prime)

        # LEVEL SHIFT!!
        kappa3 = self.ugg.unpack (self.level_shift * np.abs (x))
        kappa2 += kappa3
        return self.ugg.pack (kappa2/2)

    def get_veff_prime (self, odm1s):
        # Spoof away CI by wrapping call
        fn = lasscf_sync_o0.LASSCF_HessianOperator.get_veff_Heff
        return fn (self, odm1s, self._tdm1rs)[0]

    def orbital_response (self, kappa1, odm1s, ocm2, veff_prime):
        # Spoof away CI by wrapping call
        fn = lasscf_sync_o0.LASSCF_HessianOperator.orbital_response
        t1, t2 = self._tdm1rs, self._tcm2
        return fn (self, kappa1, odm1s, ocm2, t1, t2, veff_prime)

    def update_mo_eri (self, x, h2eff_sub):
        kappa = self.ugg.unpack (x)
        umat = linalg.expm (kappa/2)
        mo1 = self._update_mo (umat)
        h2eff_sub = self._update_h2eff_sub (mo1, umat, h2eff_sub)
        return mo1, h2eff_sub

    def get_grad (self):
        return self.ugg.pack (self.fock1 - self.fock1.T)

    def _get_Hdiag (self):
        return self._get_Horb_diag ()

def get_init_guess_rdm (las, mo_coeff=None, h2eff_sub=None):
    ''' fcibox.solver[i] members make_hdiag_csf and get_init_guess both have
        to be spoofed '''
    fakeci = las.get_init_guess_ci (mo_coeff=mo_coeff, h2eff_sub=h2eff_sub)
    casdm1frs = [[r[0] for r in f] for f in fakeci]
    casdm2fr = [[r[1] for r in f] for f in fakeci]
    return casdm1frs, casdm2fr


def rdm_cycle (las, mo_coeff, casdm1frs, veff, h2eff_sub, log, max_cycle_rdmjk=3, conv_tol_rdmjkddm=3e-4,
               conv_tol_rdmjkde=1e-8):
    ''' "fcibox.kernel" should return e_cas, (casdm1rs, casdm2r) '''
    def get_veff (my_casdm1frs):
        casdm1fs = las.make_casdm1s_sub (casdm1frs=my_casdm1frs)
        my_veff = las.get_veff (dm=las.make_rdm1s (mo_coeff=mo_coeff, casdm1s_sub=casdm1fs))
        return my_veff
    converged = False
    e_cas, fakeci = lasscf_sync_o0.ci_cycle (las, mo_coeff, None, veff, h2eff_sub, casdm1frs, log)
    casdm1frs = [f[0] for f in fakeci]
    casdm2fr = [f[1] for f in fakeci]
    veff = get_veff (casdm1frs)
    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, h2eff=h2eff_sub,
                                                 casdm1frs=casdm1frs, casdm2fr=casdm2fr)
    log.info ("LASSCF rdm-jk init : E = %.15g", e_tot)
    for it in range (max_cycle_rdmjk):
        casdm1frs_old = casdm1frs
        casdm2fr_old = casdm2fr
        e_old = e_tot
        e_cas, fakeci = lasscf_sync_o0.ci_cycle (las, mo_coeff, None, veff, h2eff_sub, casdm1frs, log)
        casdm1frs = [f[0] for f in fakeci]
        casdm2fr = [f[1] for f in fakeci]
        veff = get_veff (casdm1frs)
        e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, h2eff=h2eff_sub,
                                                     casdm1frs=casdm1frs, casdm2fr=casdm2fr)
        ddm = np.concatenate ([(d1 - d0).ravel () for d1, d0 in zip (casdm1frs, casdm1frs_old)])
        ddm = np.append (ddm, np.concatenate (
            [(d1 - d0).ravel () for d1, d0 in zip (casdm2fr, casdm2fr_old)]
        ))
        ddm = linalg.norm (ddm)
        log.info ("LASSCF rdm-jk %d : E = %.15g ; dE = %.15g ; |ddm| = %.15g", it+1, e_tot,
                  e_tot-e_old, ddm)
        if (ddm < conv_tol_rdmjkddm) and (abs(e_tot-e_old) < conv_tol_rdmjkde):
            converged = True
            break
    log.info ("LASSCF rdm-jk {}".format (('not converged','converged')[int(converged)]))
    return e_cas, casdm1frs, casdm2fr, converged

def kernel (las, mo_coeff=None, casdm1frs=None, casdm2fr=None, conv_tol_grad=1e-4, verbose=lib.logger.NOTE):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    conv_tol_rdmjkde = 1e-8
    conv_tol_rdmjkddm = las.conv_tol_rdmjkddm or 3*conv_tol_grad
    log = lib.logger.new_logger(las, verbose)
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    log.debug('Start LASSCF')

    h2eff_sub = las.get_h2eff (mo_coeff)
    t1 = log.timer('integral transformation to LAS space', *t0)

    if casdm1frs is None: casdm1frs, casdm2fr = get_init_guess_rdm (las, mo_coeff, h2eff_sub)
    casdm1fs = las.make_casdm1s_sub (casdm1frs=casdm1frs)
    dm1s = las.make_rdm1s (casdm1s_sub=casdm1fs)
    veff = las.get_veff (dm=dm1s)
    t1 = log.timer('LASSCF initial get_veff', *t1)

    ugg = None
    converged = False
    t2 = (t1[0], t1[1])
    it = 0
    for it in range (las.max_cycle_macro):
        e_cas, casdm1frs, casdm2fr, rdmjk_conv = rdm_cycle (las, mo_coeff, casdm1frs,
            veff, h2eff_sub, log, max_cycle_rdmjk=las.max_cycle_rdmjk,
            conv_tol_rdmjkddm=conv_tol_rdmjkddm,
            conv_tol_rdmjkde=conv_tol_rdmjkde)
        if ugg is None: ugg = las.get_ugg (mo_coeff)
        log.info ('LASSCF subspace CI energies: {}'.format (e_cas))
        t1 = log.timer ('LASSCF rdm_cycle', *t1)

        casdm1fs_new = las.make_casdm1s_sub (casdm1frs=casdm1frs)
        if not isinstance (las, _DFLASCI):
            dm1s = las.make_rdm1s (mo_coeff=mo_coeff, casdm1s_sub=casdm1fs_new)
            veff = las.get_veff (dm=dm1s)
        if isinstance (las, _DFLASCI):
            ddm = [dm_new - dm_old for dm_new, dm_old in zip (casdm1fs_new, casdm1fs)]
            bmPu = getattr (h2eff_sub, 'bmPu', None)
            veff += las.fast_veffa (ddm, bmPu, mo_coeff=mo_coeff)
        casdm1fs = casdm1fs_new

        t1 = log.timer ('LASSCF get_veff after ci', *t1)
        H_op = las.get_hop (ugg=ugg, mo_coeff=mo_coeff, casdm1frs=casdm1frs,
            casdm2fr=casdm2fr, h2eff_sub=h2eff_sub, veff=veff, do_init_eri=False)
        g_vec = H_op.get_grad ()
        gx = H_op.get_gx ()
        prec_op = H_op.get_prec ()
        prec_op (np.ones_like (g_vec)) # Check for divergences
        norm_gorb = linalg.norm (g_vec) if g_vec.size else 0.0
        norm_gx = linalg.norm (gx) if gx.size else 0.0
        x0 = prec_op._matvec (-g_vec)
        lib.logger.info (las, 'LASSCF macro %d : E = %.15g ; |g_int| = %.15g ; |g_x| = %.15g',
            it, H_op.e_tot, norm_gorb, norm_gx)
        if ((norm_gorb < conv_tol_grad) or (norm_gorb < norm_gx/10)) and rdmjk_conv:
            converged = True
            break
        las.dump_chk (mo_coeff=mo_coeff, ci=[casdm1frs, casdm2fr])
        # Take this part out of the true initialization b/c
        # if I'm already converged I don't want to waste the cycles
        H_op._init_eri_()
        t1 = log.timer ('LASSCF Hessian constructor', *t1)
        microit = [0]
        last_x = [0]
        first_norm_x = [None]
        def my_callback (x):
            microit[0] += 1
            norm_xorb = linalg.norm (x) if x.size else 0.0
            addr_max = np.argmax (np.abs (x))
            id_max = ugg.addr2idstr (addr_max)
            x_max = x[addr_max]/np.pi
            log.debug ('Maximum step vector element x[{}] = {}*pi ({})'.format (addr_max, x_max, id_max))
            if las.verbose > lib.logger.INFO:
                Hx = H_op._matvec (x) # This doubles the price of each iteration!!
                resid = g_vec + Hx
                norm_gorb = linalg.norm (resid) if resid.size else 0.0
                Ecall = H_op.e_tot + x.dot (g_vec + (Hx/2))
                log.info ('LASSCF micro %d : E = %.15g ; |g_orb| = %.15g ; |x_orb| = %.15g',
                    microit[0], Ecall, norm_gorb, norm_xorb)
            else:
                log.info ('LASSCF micro %d : |x_orb| = %.15g', microit[0], norm_xorb)
            if abs(x_max)>.5: # Nonphysical step vector element
                if last_x[0] == 0:
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
            x, info_int = sparse.linalg.cg (H_op, -g_vec, x0=x0, atol=my_tol,
                                            maxiter=las.max_cycle_micro,
                                            callback=my_callback, M=prec_op)
            t1 = log.timer ('LASSCF {} microcycles'.format (microit[0]), *t1)
            mo_coeff, h2eff_sub = H_op.update_mo_eri (x, h2eff_sub)
            t1 = log.timer ('LASSCF Hessian update', *t1)

            veff = las.get_veff (dm = las.make_rdm1s (mo_coeff=mo_coeff, casdm1s_sub=casdm1fs))
            t1 = log.timer ('LASSCF get_veff after secondorder', *t1)
        except MicroIterInstabilityException as e:
            log.info ('Unstable microiteration aborted: %s', str (e))
            t1 = log.timer ('LASSCF {} microcycles'.format (microit[0]), *t1)
            x = last_x[0]
            for i in range (3): # Make up to 3 attempts to scale-down x if necessary
                mo2, h2eff_sub2 = H_op.update_mo_eri (x, h2eff_sub)
                t1 = log.timer ('LASCF Hessian update', *t1)
                veff2 = las.get_veff (dm = las.make_rdm1s (mo_coeff=mo2, casdm1s_sub=casdm1fs))
                t1 = log.timer ('LASSCF get_veff after secondorder', *t1)
                e2 = las.energy_nuc () + las.energy_elec (mo_coeff=mo2, h2eff=h2eff_sub2,
                                                          casdm1frs=casdm1frs,
                                                          casdm2fr=casdm2fr,
                                                          veff=veff2)
                if e2 < H_op.e_tot:
                    break
                log.info ('New energy ({}) is higher than keyframe energy ({})'.format (
                    e2, H_op.e_tot))
                log.info ('Attempt {} of 3 to scale down trial step vector'.format (i+1))
                x *= .5
            mo_coeff, h2eff_sub, veff = mo2, h2eff_sub2, veff2


    t2 = log.timer ('LASSCF {} macrocycles'.format (it), *t2)

    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff,
        casdm1frs=casdm1frs, casdm2fr=casdm2fr, h2eff=h2eff_sub, veff=veff)
    e_tot_test = las.get_hop (ugg=ugg, mo_coeff=mo_coeff, casdm1frs=casdm1frs,
        casdm2fr=casdm2fr, h2eff_sub=h2eff_sub, veff=veff, do_init_eri=False).e_tot
    bmPu = getattr (h2eff_sub, 'bmPu', None)
    veff_a = np.stack ([las.fast_veffa ([d[state] for d in casdm1frs], bmPu, mo_coeff=mo_coeff)
        for state in range (las.nroots)], axis=0)
    veff_c = (veff.sum (0) - np.einsum ('rsij,r->ij', veff_a, las.weights))/2
    veff = veff_c[None,None,:,:] + veff_a
    veff = lib.tag_array (veff, c=veff_c, sa=np.einsum ('rsij,r->sij', veff, las.weights))
    e_states = las.energy_nuc () + np.array (las.states_energy_elec (
        mo_coeff=mo_coeff, h2eff=h2eff_sub, veff=veff, casdm1frs=casdm1frs,
        casdm2fr=casdm2fr))
    assert (np.allclose (np.dot (las.weights, e_states), e_tot)), '{} {} {} {}'.format (
        e_states, np.dot (las.weights, e_states), e_tot, e_tot_test)

    lib.logger.info (las, 'LASSCF %s after %d cycles', ('not converged', 'converged')[converged], it+1)
    lib.logger.info (las, 'LASSCF E = %.15g ; |g_int| = %.15g ; |g_ext| = %.15g', e_tot, norm_gorb, norm_gx)
    t1 = log.timer ('LASSCF wrap-up', *t1)

    mo_coeff, mo_energy, mo_occ, casdm1frs, casdm2fr, h2eff_sub = las.canonicalize (
        mo_coeff, casdm1frs, casdm2fr, veff=veff.sa, h2eff_sub=h2eff_sub)
    t1 = log.timer ('LASSCF canonicalization', *t1)

    t0 = log.timer ('LASSCF kernel function', *t0)

    return converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, casdm1frs, casdm2fr, h2eff_sub, veff

def canonicalize (las, mo_coeff=None, casdm1frs=None, casdm2fr=None, natorb_casdm1=None,
        veff=None, h2eff_sub=None, orbsym=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if casdm1frs is None: casdm1frs = las.casdm1frs
    if casdm2fr is None: casdm2fr = las.casdm2fr
    casdm1fs = las.make_casdm1s_sub (casdm1frs=casdm1frs)
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    moH_cas = mo_coeff[:,ncore:nocc].conj ().T.copy ()
    mo_coeff, mo_ene, mo_occ, _, h2eff_sub = lasci.canonicalize (las, mo_coeff=mo_coeff,
        ci=None, casdm1fs=casdm1fs, natorb_casdm1=natorb_casdm1, veff=veff,
        h2eff_sub=h2eff_sub, orbsym=orbsym)
    ovlp = las._scf.get_ovlp ()
    umat = moH_cas @ ovlp @ mo_coeff[:,ncore:nocc]
    for isub, (lasdm1rs, lasdm2r) in enumerate (zip (casdm1frs, casdm2fr)):
        i = sum (las.ncas_sub[:isub])
        j = i + las.ncas_sub[isub]
        u = umat[i:j,i:j]
        lasdm1rs[:,:,:,:] = np.einsum ('rsij,ia,jb->rsab', lasdm1rs, u.conj (), u)
        lasdm2r[:,:,:,:,:] = np.einsum ('rijkl,ia,jb,kc,ld->rabcd', lasdm2r,
            u.conj (), u, u.conj (), u)
    return mo_coeff, mo_ene, mo_occ, casdm1frs, casdm2fr, h2eff_sub


# From lasscf_sync_o0.ci_cycle and lasscf_sync_o0.get_init_guess ci, I deduce that
# the fcibox class should have the following members:
#   callable "_get_nelec"
#   callable "kernel"
#   list of solvers "fcisolvers"
# And that the solver class should have members:
#   callable "make_hdiag_csf"
#   callable "get_init_guess"
#   integer "spin"
#   integer "charge"
# It should probably also have callable "kernel" which fcibox.kernel
# calls. I can use the fcibox as syntactic sugar here

class RDMSolver (lib.StreamObject):

    nroots=1
    def __init__(self, mol, kernel=None, get_init_guess=None):
        self.mol = mol
        self.spin = None
        self.charge = None
        self.smult = None
        self.fci = None
        self._get_init_guess = get_init_guess
        self._kernel = kernel

    def make_hdiag_csf (self, h1s, h2, norb, nelec, max_memory):
        ''' spoof! '''
        return (h1s, h2)

    def get_init_guess (self, norb, nelec, nroots, ham):
        h1s, h2 = ham
        if callable (self._get_init_guess):
            dm1s, dm2 = self._get_init_guess (norb, nelec, nroots, h1s, h2)
        else:
            fci = self._get_csf_solver (nelec)
            hdiag = fci.make_hdiag_csf (h1s, h2, norb, nelec)
            ci = fci.get_init_guess (norb, nelec, nroots, hdiag)
            dm1s, dm2 = self._ci2rdm (fci, ci, norb, nelec)
        return dm1s, dm2

    def kernel (self, norb, nelec, h0, h1s, h2):
        h2 = ao2mo.restore (1, h2, norb)
        if callable (self._kernel):
            erdm, dm1s, dm2 = self._kernel (norb, nelec, h0, h1s, h2)
        else:
            fci = self._get_csf_solver (nelec)
            erdm, ci = fci.kernel (h1s, h2, norb, nelec, nroots=1, ecore=h0)
            dm1s, dm2 = self._ci2rdm (fci, ci, norb, nelec)
        return erdm, dm1s, dm2

    def _get_csf_solver (self, nelec):
        smult = getattr (self, 'smult', None)
        if smult is None:
            if (self.spin is None) or isinstance (nelec, (list, tuple, np.ndarray)):
                nelec = _unpack_nelec (nelec)
                smult = abs (nelec[0] - nelec[1]) + 1
            else:
                smult = self.spin + 1
        return csf_solver (self.mol, smult=smult)

    def _ci2rdm (self, fci, ci, norb, nelec):
        dm1s, dm2 = fci.make_rdm12s (ci, norb, nelec)
        dm1s = np.stack (dm1s, axis=0)
        dm2 = dm2[0] + dm2[1] + dm2[1].transpose (2,3,0,1) + dm2[2]
        return dm1s, dm2

class FCIBox (lib.StreamObject):

    def __init__(self, rdmsolvers):
        self.fcisolvers = rdmsolvers

    @property
    def nroots (self):
        return len (self.fcisolvers)

    @property
    def weights (self):
        return [1.0/self.nroots,]*self.nroots

    def kernel (self, h1rs, h2, norb, nelec, ci0=None, verbose=None,
            max_memory=None, ecore=0, orbsym=None):
        if isinstance (ecore, (int, float, np.integer, np.floating)):
            ecore = [ecore,] * len (h1rs)
        erdm = []
        dm1rs = []
        dm2r = []
        for h0, h1s, solver in zip (ecore, h1rs, self.fcisolvers):
            e, dm1s, dm2 = solver.kernel (norb, nelec, h0, h1s, h2)
            erdm.append (e)
            dm1rs.append (dm1s)
            dm2r.append (dm2)
        dm1rs = np.stack (dm1rs, axis=0)
        dm2r = np.stack (dm2r, axis=0)
        erdm = np.array (erdm)
        return erdm, (dm1rs, dm2r)

    def _get_nelec (self, solver, nelec):
        m = solver.spin if solver.spin is not None else 0
        c = getattr (solver, 'charge', 0) or 0
        if m or c:
            nelec = np.sum (nelec) - c
            nelec = (nelec+m)//2, (nelec-m)//2
        return nelec

    def get_aufbau_guess (self, norb, nelec0, orbsym=None):
        solvers_r = []
        nelec_r = []
        for rdmsolver in self.fcisolvers:
            nelec_r.append (self._get_nelec (rdmsolver, nelec0))
            solvers_r.append (rdmsolver._get_csf_solver (nelec_r[-1]))
        ci_r = addons.get_aufbau_guess (solvers_r, norb, nelec_r, orbsym=orbsym)
        ci0 = []
        for ci, nelec, rdmsolver in zip (ci_r, nelec_r, self.fcisolvers):
            fci = rdmsolver._get_csf_solver (nelec)
            dm1s, dm2 = rdmsolver._ci2rdm (fci, ci, norb, nelec)
            ci0.append ((dm1s, dm2))
        return ci0

    def states_make_rdm1s (self, ci, norb, nelec):
        dm1a = np.stack ([c[0][0,:,:] for c in ci], axis=0)
        dm1b = np.stack ([c[0][1,:,:] for c in ci], axis=0)
        return dm1a, dm1b

def make_fcibox (mol, kernel=None, get_init_guess=None, spin=None, smult=None):
    s = RDMSolver (mol, kernel=kernel, get_init_guess=get_init_guess)
    s.spin = spin
    s.smult = smult
    return FCIBox ([s])

def _combine_init_guess_ci (las, ci0i, ci0g, norb, nelec, nroots):
    if nroots>1: raise NotImplementedError ("Multiple local roots for lasscf_rdm")
    if getattr (ci0i, '__len__', None) is None: return ci0g
    if len (ci0i) != 2: return ci0g
    for n, ci0in in enumerate (ci0i):
        if not isinstance (ci0in, np.ndarray): return ci0g
        if ci0in.size != norb ** (2*n): return ci0g
    return ci0i

class LASSCFNoSymm (lasscf_sync_o0.LASSCFNoSymm):

    def __init__(self, *args, **kwargs):
        self.casdm1frs = None
        self.casdm2fr = None
        self.max_cycle_rdmjk = 3
        self.conv_tol_rdmjkddm = None
        self.conv_tol_rdmjkde = 1e-8
        lasscf_sync_o0.LASSCFNoSymm.__init__(self, *args, **kwargs)
        self.max_cycle_micro = 3

    _ugg = LASSCF_UnitaryGroupGenerators
    _hop = LASSCF_HessianOperator
    canonicalize = canonicalize
    _combine_init_guess_ci = _combine_init_guess_ci

    def _init_fcibox (self, smult, nel):
        return make_fcibox (self.mol, spin=nel[0]-nel[1], smult=smult)

    def kernel(self, mo_coeff=None, casdm1frs=None, casdm2fr=None, conv_tol_grad=None, verbose=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if verbose is None: verbose = self.verbose
        if conv_tol_grad is None: conv_tol_grad = self.conv_tol_grad
        log = lib.logger.new_logger(self, verbose)

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        self.dump_flags(log)

        # MRH: the below two lines are not the ideal solution to my problem...
        for fcibox in self.fciboxes:
            fcibox.verbose = self.verbose
            fcibox.stdout = self.stdout
        self.nroots = self.fciboxes[0].nroots
        self.weights = self.fciboxes[0].weights

        self.converged, self.e_tot, self.e_states, self.mo_energy, self.mo_coeff, \
            self.e_cas, self.casdm1frs, self.casdm2fr, h2eff_sub, veff = \
                kernel(self, mo_coeff, casdm1frs=casdm1frs, casdm2fr=casdm2fr,
                    verbose=verbose, conv_tol_grad=conv_tol_grad)

        return self.e_tot, self.e_cas, self.casdm1frs, self.casdm2fr, self.mo_coeff, self.mo_energy, h2eff_sub, veff

def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry:
        raise NotImplementedError ("point-group symmetry")
        #las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = lasci.density_fit (las, with_df = mf.with_df)
    return las


if __name__ == '__main__':
    from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF as LASSCFRef
    xyz = '''H 0.0 0.0 0.0
             H 1.0 0.0 0.0
             H 0.2 3.9 0.1
             H 1.159166 4.1 -0.1'''
    mol = gto.M (atom = xyz, basis = '6-31g', output='lasscf_rdm.log',
        verbose=lib.logger.INFO)
    mf = scf.RHF (mol).run ()
    las = LASSCFRef (mf, (2,2), (2,2), spin_sub=(1,1))
    frag_atom_list = ((0,1),(2,3))
    mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
    las.ah_level_shift = 1e-4
    #las.max_cycle_macro = 3
    las.kernel (mo_loc)

    mo = las.mo_coeff
    casdm1frs = las.states_make_casdm1s_sub ()
    casdm2fr = las.states_make_casdm2_sub ()

    ugg_test = LASSCF_UnitaryGroupGenerators (las, mo)
    hop_test = LASSCF_HessianOperator (las, ugg_test, casdm1frs=casdm1frs,
        casdm2fr=casdm2fr)

    ugg_ref = las.get_ugg ()
    hop_ref = las.get_hop (ugg=ugg_ref)

    g_test = hop_test.get_grad ()
    g_ref = hop_ref.get_grad ()[:g_test.size]
    print ('gradient test:', linalg.norm (g_test-g_ref), linalg.norm (g_ref))

    x_test = np.random.rand (ugg_test.nvar_tot)
    x_ref = np.zeros (ugg_ref.nvar_tot)
    x_ref[:ugg_ref.nvar_orb] = x_test[:]

    prec_test = hop_test.get_prec ()(x_test)
    prec_ref = hop_ref.get_prec ()(x_ref)[:prec_test.size]
    print ('preconditioner test:', linalg.norm (prec_test-prec_ref),
        linalg.norm (prec_ref))

    hx_test = hop_test._matvec (x_test)
    hx_ref = hop_ref._matvec (x_ref)[:hx_test.size]
    print ('hessian test:', linalg.norm (hx_test-hx_ref), linalg.norm (hx_ref))

    print ("CI algorithm total energy:", las.e_tot)
    las_test = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
    las_test.kernel (mo_loc)
    print ("RDM algorithm total energy:", las_test.e_tot)
