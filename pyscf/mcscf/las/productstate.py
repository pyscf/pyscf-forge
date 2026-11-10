import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.scf.addons import canonical_orth_
from pyscf.fci import cistring
from pyscf.mcscf.addons import state_average as state_average_mcscf
from pyscf.csf_fci.csf import CSFFCISolver
from pyscf.csf_fci.csfstring import CSFTransformer
from pyscf.mcscf.las.addons import StateAverageNMixFCISolver
from itertools import combinations


def _is_csf_solver(solver):
    '''Recognize real or complex CSF solvers without importing PBC modules.'''
    if isinstance(solver, CSFFCISolver):
        return True
    transformer = getattr(solver, 'transformer', None)
    return (transformer is not None
            and hasattr(transformer, 'ncsf')
            and callable(getattr(transformer, 'vec_det2csf', None))
            and callable(getattr(transformer, 'printable_largest_csf', None)))


# TODO: linkstr support
class ProductStateFCISolver (StateAverageNMixFCISolver, lib.StreamObject):
    r'''Minimize the energy of a wave function of the form

    |Psi> = A \prod_K |ci_K>

    Self-consistently over all ci_K.
    '''

    def __init__(self, fcisolvers, stdout=None, verbose=0, **kwargs):
        self.fcisolvers = fcisolvers
        self.verbose = verbose
        self.stdout = stdout
        self.log = lib.logger.new_logger (self, verbose)

    def kernel (self, h1, h2, norb_f, nelec_f, ecore=0, ci0=None, orbsym=None,
            conv_tol_grad=1e-4, conv_tol_self=1e-10, max_cycle_macro=50,
            serialfrag=False, **kwargs):
        log = self.log
        converged = False
        e_sigma = 0
        e = [0 for n in norb_f]
        ci1 = ci0
        log.info ('Entering product-state fixed-point CI iteration')
        for it in range (max_cycle_macro):
            ci0 = self.get_init_guess (ci1, norb_f, nelec_f, h1, h2)
            # Issue #86: put get_init_guess INSIDE the iteration in case _1shot below encounters
            # linear dependencies and can't populate all CI vectors for all fragments.
            h1eff, h0eff, ci0 = self.project_hfrag (h1, h2, ci0, norb_f, nelec_f,
                ecore=ecore, **kwargs)
            grad = self._get_grad (h1eff, h2, ci0, norb_f, nelec_f, **kwargs)
            grad_max = np.amax (np.abs (grad))
            solvers_converged = [np.all (np.asarray (s.converged)) for s in self.fcisolvers]
            nconv = sum ([int (c) for c in solvers_converged])
            log.info ('Cycle %d: max grad = %e ; sigma = %e ; %d/%d fragment CI solvers converged',
                      it, grad_max, np.real(e_sigma), nconv, len (self.fcisolvers))
            log.debug ('e vector = {}'.format (e))
            if nconv<len(self.fcisolvers): log.debug ('unconverged fragment CI solvers: {}'.format (
                list(np.where (np.logical_not (solvers_converged))[0])))
            if ((grad_max < conv_tol_grad) and (e_sigma < conv_tol_self)
                and all ([solvers_converged]) and it>0):
                converged = True
                break
            e, ci1 = self._1shot (it, h0eff, h1eff, h2, e, ci0, norb_f, nelec_f,
                orbsym=orbsym, serialfrag=serialfrag, **kwargs)
            e_sigma = np.amax (e) - np.amin (e)
        conv_str = ['NOT converged','converged'][int (converged)]
        log.info (('Product_state fixed-point CI iteration {} after {} '
                   'cycles').format (conv_str, it))
        if not converged:
            ci1 = self.get_init_guess (ci1, norb_f, nelec_f, h1, h2)
            # Issue #86: see above, same problem
            self._debug_csfs (log, ci0, ci1, norb_f, nelec_f, grad)
        energy_elec = self.energy_elec (h1, h2, ci1, norb_f, nelec_f,
            ecore=ecore, efinal=e, **kwargs)
        return converged, energy_elec, ci1

    def get_init_guess (self, ci0, norb_f, nelec_f, h1, h2, nroots=None):
        '''Generate CI guess vectors for all fragments.

        Args:
            ci0: list of length nfrag or None
                Contains either None or ndarrays of guess CI vectors. Any new guess CI vectors
                constructed by this function are constrained to be orthogonal to those already
                provided here, if any.
            norb_f: list of length nfrag of integers
                Number of orbitals in each fragment
            nelec_f: list of length nfrag of integers
                Number of electrons (in reference state) in each fragment
            h1: ndarray of shape (ncas,ncas) or (2,ncas,ncas)
                One-electron Hamiltonian amplitudes
            h2: ndarray of shape (ncas,ncas,ncas,ncas)
                Two-electron Hamiltonian amplitudes

        Returns:
            ci1: list of length nfrag of ndarrays
                Orthonormal guess CI vectors. Any vectors present in ci0 are preserved unaltered.
        '''
        if ci0 is None: ci0 = [None for i in range (len (norb_f))]
        ci1 = [c for c in ci0] # reference safety
        if h1.ndim < 3: h1 = np.stack ([h1, h1], axis=0)
        for ix, (no, ne, solver) in enumerate (zip (norb_f, nelec_f, self.fcisolvers)):
            solver.check_transformer_cache ()
            snroots = solver.nroots if nroots is None else min (nroots, solver.transformer.ncsf)
            nelec = self._get_nelec (solver, ne)
            i = sum (norb_f[:ix])
            j = i + norb_f[ix]
            hdiag_csf = solver.make_hdiag_csf (h1[:,i:j,i:j], h2[i:j,i:j,i:j,i:j],
                                               no, nelec)
            ci1_guess = solver.get_init_guess (no, nelec, snroots, hdiag_csf)
            na = cistring.num_strings (no, nelec[0])
            nb = cistring.num_strings (no, nelec[1])
            if ci1[ix] is None:
                ci1[ix] = ci1_guess
            elif np.asarray (ci1[ix]).reshape (-1,na*nb).shape[0] < snroots:
                ci1_inp = np.asarray (ci1[ix]).reshape (-1,na*nb)
                ci1_guess = np.asarray (ci1_guess).reshape (-1,na*nb)
                x = np.append (ci1_inp, ci1_guess, axis=0)
                x = canonical_orth_(x.conj () @ x.T).T @ x
                # ^ an orthonormal basis
                assert (x.shape[0] >= snroots)
                x2inp = x.conj () @ ci1_inp.T
                ninp = ci1_inp.shape[0]
                u, svals, vh = linalg.svd (x2inp, full_matrices=True)
                u[:,:ninp] = u[:,:ninp] @ vh
                ci1_new = u.T @ x
                nnew = ci1_new.shape[0]
                ovlp = ci1_new.conj () @ ci1_inp.T
                assert (np.all (np.abs (ovlp[:ninp,:ninp] - np.eye (ninp)) < 1e-3)), '{}'.format (ovlp)
                ovlp = (ci1_new.conj () @ ci1_new.T)
                assert (np.all (np.abs (ovlp - np.eye (nnew)) < 1e-3)), '{}'.format (ovlp)
                ci1[ix] = ci1_new[:snroots].reshape (snroots, na, nb)
        return self._check_init_guess (ci1, norb_f, nelec_f, nroots=nroots)

    def _check_init_guess (self, ci0, norb_f, nelec_f, nroots=None):
        ci1 = []
        if ci0 is None: ci0 = [None for i in range (len (norb_f))]
        for ix, (no, ne, solver) in enumerate (zip (norb_f, nelec_f, self.fcisolvers)):
            solver.check_transformer_cache ()
            snroots = solver.nroots if nroots is None else min (nroots, solver.transformer.ncsf)
            nelec = self._get_nelec (solver, ne)
            neleca, nelecb = nelec
            na = cistring.num_strings (no, neleca)
            nb = cistring.num_strings (no, nelecb)
            cguess = np.asarray (ci0[ix]).reshape (-1,na,nb)
            dtype = np.result_type(
                getattr(self, 'ci_dtype', float), cguess.dtype,
            )
            zguess = np.zeros ((snroots,na,nb), dtype=dtype)
            ngroots = min (zguess.shape[0], cguess.shape[0])
            zguess[:ngroots,:,:] = cguess[:ngroots,:,:]
            ci1.append (zguess)
            if snroots>na*nb:
                raise RuntimeError ("{} roots > {} determinants in fragment {}".format (
                    snroots, na*nb, ix))
            if _is_csf_solver(solver):
                solver.check_transformer_cache ()
                if snroots>solver.transformer.ncsf:
                    raise RuntimeError ("{} roots > {} CSFs in fragment {} (nelec={}, smult={})".format (
                        snroots, solver.transformer.ncsf, ix, solver.nelec, solver.smult))
        return ci1
                
    def _debug_csfs (self, log, ci0, ci1, norb_f, nelec_f, grad, nroots=None):
        if not all (_is_csf_solver(s) for s in self.fcisolvers):
            return
        if log.verbose < lib.logger.INFO: return
        transformers = [s.transformer for s in self.fcisolvers]
        grad_f = []
        for s,t in zip (self.fcisolvers, transformers):
            snroots = nroots if nroots is not None else s.nroots
            grad_f.append (grad[:t.ncsf*snroots].reshape (snroots, t.ncsf))
            offs = (t.ncsf*snroots) + (snroots*(snroots-1)//2)
            grad = grad[offs:]
        assert (len (grad) == 0)
        log.info ('Debugging CI and gradient vectors...')
        for ix, (grad, c0, c1, s, t) in enumerate (zip (grad_f, ci0, ci1, self.fcisolvers, transformers)):
            log.info ('Fragment %d', ix)
            c0_csf, c0_norm = t.vec_det2csf (c0, normalize=True, return_norm=True)
            c1_csf, c1_norm = t.vec_det2csf (c1, normalize=True, return_norm=True)
            log.info ('CI vector norm = %s', str(c1_norm))
            grad_norm = linalg.norm (grad)
            log.info ('Gradient norm = %e', grad_norm)
            c0_lbls, c0_coeffs = t.printable_largest_csf (c0_csf, 10)
            c1_lbls, c1_coeffs = t.printable_largest_csf (c1_csf, 10)
            g_lbls, g_coeffs = t.printable_largest_csf (grad, 10, normalize=False)
            nroots = len (c0_lbls)
            for i in range (nroots):
                log.info ('Previous CI vector leading components (%d/%d):', i, nroots)
                for l, c in zip (c0_lbls[i], c0_coeffs[i]):
                    log.info ('%s : %e', l, c)
                log.info ('Current CI vector leading components (%d/%d):', i, nroots)
                for l, c in zip (c1_lbls[i], c1_coeffs[i]):
                    log.info ('%s : %e', l, c)
                log.info ('Grad vector leading components (%d/%d):', i, nroots)
                for l, c in zip (g_lbls[i], g_coeffs[i]):
                    log.info ('%s : %e', l, c)

    def _1shot (self, it, h0eff, h1eff, h2, e0, ci0, norb_f, nelec_f, orbsym=None,
                serialfrag=False, **kwargs):
        nfrag = len (norb_f)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h0eff, h1eff, ci0, norb_f, nelec_f, self.fcisolvers, ni, nj]
        e1 = [e for e in e0]
        ci1 = [c for c in ci0]
        for ifrag, (h0e, h1e, c, no, ne, solver, i, j) in enumerate (zip (*zipper)):
            if serialfrag and it % nfrag != ifrag: continue
            h2e = h2[i:j,i:j,i:j,i:j]
            osym = getattr (solver, 'orbsym', None)
            if orbsym is not None: osym=orbsym[i:j]
            nelec = self._get_nelec (solver, ne)
            e, c1 = solver.kernel (h1e, h2e, no, nelec, ci0=c, ecore=h0e,
                orbsym=osym, **kwargs)
            e1[ifrag] = e
            ci1[ifrag] = c1
        return e1, ci1

    def _get_grad (self, h1eff, h2, ci, norb_f, nelec_f, orbsym=None,
            **kwargs):
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h1eff, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        grad = []
        for h1e, c, no, ne, solver, i, j in zip (*zipper):
            nelec = self._get_nelec (solver, ne)
            nroots = solver.nroots
            h2e = h2[i:j,i:j,i:j,i:j]
            h2e = solver.absorb_h1e (h1e, h2e, no, nelec, 0.5)
            if nroots==1: c=c[None,:]
            hc = [solver.contract_2e (h2e, col, no, nelec) for col in c]
            c, hc = np.asarray (c), np.asarray (hc)
            chc = np.dot (np.asarray (c).reshape (nroots,-1).conj (),
                          np.asarray (hc).reshape (nroots,-1).T)
            hc = hc - np.tensordot (chc, c, axes=1)
            if isinstance (solver, CSFFCISolver):
                hc = solver.transformer.vec_det2csf (hc, normalize=False)
            # External degrees of freedom: not weighted, because I want
            # to converge all of the roots even if they don't contribute
            # to the mean field
            assert (hc.size == nroots*solver.transformer.ncsf)
            grad.append (hc.ravel ())
            # Internal degrees of freedom: weighted and lower-triangular
            # TODO: confirm the sign choice below before using this gradient
            # for something more advanced than convergence checking
            if nroots>1 and getattr (solver, 'weights', None) is not None:
                chc *= np.asarray (solver.weights)[:,None]
                chc -= chc.T
                grad.append (chc[np.tril_indices (nroots,k=-1)])
        return np.concatenate (grad)

    def energy_elec (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        dm1s = np.stack (self.make_rdm1s (ci, norb_f, nelec_f), axis=0)
        if h1.ndim < 3: h1 = np.stack ([h1, h1], axis=0)
        dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1s, axes=3)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        return energy_tot

    def project_hfrag (self, h1, h2, ci, norb_f, nelec_f, ecore=0, dm1s=None, dm2=None, **kwargs):
        if dm1s is None: dm1s = np.stack (self.make_rdm1s (ci, norb_f, nelec_f), axis=0)
        if h1.ndim < 3: h1 = np.stack ([h1,h1], axis=0)
        if dm2 is None: dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1s, axes=3)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        v1  = np.tensordot (dm1s, h2, axes=2)
        v1 += v1[::-1] # ja + jb
        v1 -= np.tensordot (dm1s, h2, axes=((1,2),(2,1)))
        f1 = h1 + v1
        h1eff = []
        h0eff = []
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        for i, j in zip (ni, nj):
            dm1s_i = dm1s[:,i:j,i:j]
            dm2_i = dm2[i:j,i:j,i:j,i:j]
            # v1 self-interaction
            h2_i = h2[i:j,i:j,:,:]
            v1_i = np.tensordot (dm1s_i, h2_i, axes=2)
            v1_i += v1_i[::-1] # ja + jb
            h2_i = h2[:,i:j,i:j,:]
            v1_i -= np.tensordot (dm1s_i, h2_i, axes=((1,2),(2,1)))
            # cancel off-diagonal energy double-counting
            e_i = energy_tot - np.tensordot (dm1s, v1_i, axes=3) # overcorrects
            # cancel h1eff double-counting
            v1_i = v1_i[:,i:j,i:j] 
            h1eff.append (f1[:,i:j,i:j]-v1_i)
            # cancel diagonal energy double-counting
            h1_i = h1[:,i:j,i:j] - v1_i # v1_i fixes overcorrect
            h2_i = h2[i:j,i:j,i:j,i:j]
            e_i -= (np.tensordot (h1_i, dm1s_i, axes=3)
              + 0.5*np.tensordot (h2_i, dm2_i, axes=4))
            h0eff.append (e_i)
        return h1eff, h0eff, ci
        # A child class will modify ci in this function

    def make_rdm1s (self, ci, norb_f, nelec_f, **kwargs):
        norb = sum (norb_f)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        dm1s_f = []
        for ix, (i, j, c, no, ne, s) in enumerate (zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers)):
            nelec = self._get_nelec (s, ne)
            if getattr (c, 'ndim', 3) == 3: c = list (c)
            try:
                a, b = s.make_rdm1s (c, no, nelec)
            except AssertionError as e:
                print (type (c), np.asarray (c).shape, no, nelec, ix, type (s), getattr (s, 'weights', None))
                raise (e)
            except ValueError as e:
                print ("frag=",ix,"nroots=",s.nroots,"no=",no,"ne=",nelec,'c.shape=',np.asarray(c).shape)
                if _is_csf_solver(s):
                    print ("smult=",s.smult,"ncsf=",s.transformer.ncsf)
                raise (e)
            dm1s_f.append ((i, j, np.asarray(a), np.asarray(b)))
        dtype = np.result_type (float,
            *[dm for _, _, a, b in dm1s_f for dm in (a, b)])
        dm1a = np.zeros ((norb, norb), dtype=dtype)
        dm1b = np.zeros ((norb, norb), dtype=dtype)
        for i, j, a, b in dm1s_f:
            dm1a[i:j,i:j] = a[:,:]
            dm1b[i:j,i:j] = b[:,:]
        return dm1a, dm1b

    def make_rdm1 (self, ci, norb_f, nelec_f, **kwargs):
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        return dm1a + dm1b

    def make_rdm2 (self, ci, norb_f, nelec_f, **kwargs):
        norb = sum (norb_f)
        dm2 = np.zeros ([norb,]*4)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        for i, j, c, no, ne, s in zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers):
            nelec = self._get_nelec (s, ne)
            dm2[i:j,i:j,i:j,i:j] = s.make_rdm2 (c, no, nelec)
        dm1 = dm1a + dm1b
        for (i,j), (k,l) in combinations (zip (ni, nj), 2):
            d1_ij, d1a_ij, d1b_ij = dm1[i:j,i:j], dm1a[i:j,i:j], dm1b[i:j,i:j]
            d1_kl, d1a_kl, d1b_kl = dm1[k:l,k:l], dm1a[k:l,k:l], dm1b[k:l,k:l]
            d2 = np.multiply.outer (d1_ij, d1_kl)
            dm2[i:j,i:j,k:l,k:l] = d2
            dm2[k:l,k:l,i:j,i:j] = d2.transpose (2,3,0,1)
            d2  = np.multiply.outer (d1a_ij, d1a_kl)
            d2 += np.multiply.outer (d1b_ij, d1b_kl)
            dm2[i:j,k:l,k:l,i:j] = -d2.transpose (0,2,3,1)
            dm2[k:l,i:j,i:j,k:l] = -d2.transpose (2,0,1,3)
        return dm2

def state_average_fcisolver (solver, weights=(.5,.5), wfnsym=None):
    # hoo boy this is real dumb
    dummy = type ("dummy",(object,),{"_keys":set()}) ()
    dummy.fcisolver = solver
    return state_average_mcscf (dummy, weights=weights, wfnsym=wfnsym).fcisolver

class ImpureProductStateFCISolver (ProductStateFCISolver):
    r'''Minimize the energy of an impure state:

    E = \sum_n1 w_n1 \sum_n2 w_n2 \sum_n3 w_n3 ... <n1n2n3...|H|n1n2n3...>

    over orthonormal sets of CI vectors {nK} for fragment K.'''

    def __init__(self, fcisolvers, stdout=None, verbose=0, lroots=None, lweights=None, **kwargs):
        ProductStateFCISolver.__init__(self, fcisolvers, stdout=stdout, verbose=verbose, **kwargs)
        if lweights is None:
            if lroots is not None:
                lweights = []
                for lroot in lroots:
                    l = np.zeros (lroot)
                    l[0] = 1
                    lweights.append (l)
            else:
                lweights = [[.5,.5],]*len(fcisolvers)
        for ix, (fcisolver, weights) in enumerate (zip (self.fcisolvers, lweights)):
            if len (weights) > 1:
                self.fcisolvers[ix] = state_average_fcisolver (fcisolver, weights=weights)

