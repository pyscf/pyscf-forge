import numpy as np
from scipy import linalg, special
from scipy.sparse.linalg import LinearOperator
from pyscf.lib import logger, temporary_env
from pyscf.mcscf.addons import StateAverageMCSCFSolver, StateAverageMixFCISolver, state_average_mix
from pyscf.mcscf.addons import StateAverageMixFCISolver_state_args as _state_arg
from pyscf.mcscf.addons import StateAverageMixFCISolver_solver_args as _solver_arg
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci import cistring

class StateAverageNMixFCISolver (StateAverageMixFCISolver):
    def _get_nelec (self, solver, nelec):
        n = np.sum (nelec)
        m = solver.spin if solver.spin is not None else n%2
        c = getattr (solver, 'charge', 0) or 0
        n -= c
        nelec = (n+m)//2, (n-m)//2
        return nelec

def get_sanmix_fcisolver (samix_fcisolver):

    # Recursion protection
    if isinstance (samix_fcisolver, StateAverageNMixFCISolver):
        return samix_fcisolver

    fcisolvers, weights = samix_fcisolver.fcisolvers, samix_fcisolver.weights
    class FCISolver (samix_fcisolver.__class__, StateAverageNMixFCISolver):
        _get_nelec = StateAverageNMixFCISolver._get_nelec

    sanmix_fcisolver = FCISolver (fcisolvers, weights)
    sanmix_fcisolver.__dict__.update (samix_fcisolver.__dict__)
    return sanmix_fcisolver

def state_average_n_mix (casscf, fcisolvers, weights=(0.5,0.5)):
    sacasscf = state_average_mix (casscf, fcisolvers, weights=weights)
    sacasscf.fcisolver = get_sanmix_fcisolver (sacasscf.fcisolver)
    # Inject "charge" into "_keys" to suppress annoying warning msg
    keys = set(('charge',))
    for solver in sacasscf.fcisolver.fcisolvers:
        solver._keys = set (solver._keys).union (keys)
    return sacasscf

def state_average_n_mix_(casscf, fcisolvers, weights=(0.5,0.5)):
    sacasscf = state_average_n_mix (casscf, fcisolvers, weights)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__.update(sacasscf.__dict__)
    return casscf

class H1EZipFCISolver (object):
    pass

def get_aufbau_guess (solvers_r, norb, nelec_r, orbsym=None):
    '''Get the spin-adapted CI vector that corresponds to spin-alpha and spin-beta
    orbitals being separately filled in Aufbau order from the bottom up, under
    constraints of point-group symmetry if applicable. In the determinant basis
    and C1 point group this is just ci[0,0] = 1, but in the CSF basis electron
    configurations are not necessarily in that order so we have to look up some
    stuff.

    Args:
        solvers_r: list (length nfrag) of CSF FCI solvers
        norb : integer
        nelec_r : list (length nfrag) of integers
            Numbers of electrons in each fragment

    Kwargs:
        orbsym: sequence (length norb)
            Irreps of the orbitals

    Returns:
        ci: ndarray
            CI vector
    '''
    ci = []
    for solver, nelec in zip (solvers_r, nelec_r):
        solver.norb, solver.nelec = norb, nelec
        if orbsym is not None:
            solver.orbsym = orbsym
        solver.check_transformer_cache ()
        t = solver.transformer
        c = np.zeros ((t.ncsf,), dtype=float)
        # TODO: either reverse the order of configurations in csf_fci, or else
        # move this logic onto the CSFTransformer class
        for i in range (t.ndeta*t.ndetb):
            # Find the earliest configuration with correct symmetry
            j = t.econf_det_mask[i]
            if (t.wfnsym is None) or (t._orbsym is None) or (t.confsym[j]==t.wfnsym):
                break
        for ix in range (t.ncsf):
            # Find the first CSF of that configuration
            if t.econf_csf_mask[ix] == j:
                break
        c[ix] = 1.0
        ci.append (t.vec_csf2det (c, normalize=True))
    return ci

def get_h1e_zipped_fcisolver (fcisolver):
    ''' Wrap a state-average-mix FCI solver to take a list of h1es to apply to each solver. '''

    # Recursion protection
    if isinstance (fcisolver, H1EZipFCISolver):
        return fcisolver

    assert isinstance (fcisolver, StateAverageMixFCISolver), 'requires StateAverageMixFCISolver'
    has_spin_square = getattr(fcisolver, 'states_spin_square', None)

    class FCISolver (fcisolver.__class__, H1EZipFCISolver):

        def get_aufbau_guess (self, norb, nelec0, orbsym=None):
            solvers = self.fcisolvers
            nelec_r = [self._get_nelec (solver, nelec0) for solver in solvers]
            return get_aufbau_guess (solvers, norb, nelec_r, orbsym=orbsym)

        # This is a hack designed to, i.e., set
        # self.fcisolvers[i].nroots = 1 for all i
        # in the context of StateAverageMixFCISolver._loop_civecs, so that local state-averaging
        # in the ith Hilbert space can be totally abstracted away from the HIEZipFCISolver,
        # StateAverageNMixFCISolver layer
        def _loop_civecs (self, *args, **kwargs):
            _solver_args = self._solver_args
            _state_args = self._state_args
            for i, solver in enumerate (self.fcisolvers):
                my_args = []
                for arg in args:
                    if isinstance (arg, (_state_args, _solver_args)):
                        my_args.append (arg[i])
                    else:
                        my_args.append (arg)
                my_kwargs = {}
                for key, item in kwargs.items ():
                    if isinstance (arg, (_state_args, _solver_args)):
                        my_kwargs[key] = item[i]
                    else:
                        my_kwargs[key] = item
                yield solver, my_args, my_kwargs

        def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, ecore=0, orbsym=None, **kwargs):
            # Note self.orbsym is initialized lazily in mc1step_symm.kernel function
            log = logger.new_logger(self, verbose)
            es = []
            cs = []
            if isinstance (ecore, (int, float, np.integer, np.floating)):
                ecore = [ecore,] * len (h1)
            if orbsym is None: orbsym=self.orbsym
            for solver, my_args, my_kwargs in self._loop_solver(_state_arg (ci0), _solver_arg (h1), _state_arg (ecore)):
                c0 = my_args[0]
                h1e = my_args[1]
                e0 = my_args[2]
                e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                     orbsym=orbsym, verbose=log, ecore=e0, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            self.e_states = es
            self.converged = np.all(getattr(sol, 'converged', True)
                                       for sol in self.fcisolvers)

            if log.verbose >= logger.DEBUG:
                if has_spin_square:
                    ss = self.states_spin_square(cs, norb, nelec)[0]
                    for i, ei in enumerate(es):
                        if i>10 and log.verbose < logger.DEBUG1:
                            log.debug ('printout for %d more states truncated', len(es)-11)
                            break
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                else:
                    for i, ei in enumerate(es):
                        if i>10 and log.verbose < logger.DEBUG1:
                            log.debug ('printout for %d more states truncated', len(es)-11)
                            break
                        log.debug('state %d  E = %.15g', i, ei)
            return np.einsum('i,i', np.array(es), self.weights), cs

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, orbsym=None, **kwargs):
            es = []
            cs = []
            if orbsym is None: orbsym=self.orbsym
            for solver, my_args, _ in self._loop_solver (_state_arg (ci0), _solver_arg (h1)):
                c0 = my_args[0]
                h1e = my_args[1]
                try:
                    e, c = solver.approx_kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                                orbsym=orbsym, **kwargs)
                except AttributeError:
                    e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                         orbsym=orbsym, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            return np.einsum('i,i->', es, self.weights), cs

        def states_absorb_h1e (self, h1, h2, norb, nelec, fac):
            op = []
            for solver, my_args, _ in self._loop_solver (_solver_arg (h1)):
                h1e = my_args[0]
                op.append (solver.absorb_h1e (h1e, h2, norb, self._get_nelec (solver, nelec), fac) if h1 is not None else h2)
            return op

        def states_contract_2e (self, h2, ci, norb, nelec, link_index=None):
            hc = []
            for solver, my_args, _ in self._loop_solver (_state_arg (ci), _state_arg (h2), _solver_arg (link_index)):
                c0 = my_args[0]
                h2e = my_args[1]
                linkstr = my_args[2]
                hc.append (solver.contract_2e (h2e, c0, norb, self._get_nelec (solver, nelec), link_index=linkstr))
            return hc

        def states_make_hdiag (self, h1, h2, norb, nelec):
            hdiag = []
            for solver, my_args, _ in self._loop_solver (_solver_arg (h1)):
                h1e = my_args[0]
                hdiag.append (solver.make_hdiag (h1e, h2, norb, self._get_nelec (solver, nelec)))
            return hdiag

        def states_make_hdiag_csf (self, h1, h2, norb, nelec):
            hdiag = []
            for solver, my_args, _ in self._loop_solver (_solver_arg (h1)):
                h1e = my_args[0]
                with temporary_env (solver, orbsym=self.orbsym):
                    hdiag.append (solver.make_hdiag_csf (h1e, h2, norb, self._get_nelec (solver, nelec), max_memory=self.max_memory))
            return hdiag

        # The below can conceivably be added to pyscf.mcscf.addons.StateAverageMixFCISolver in future

        def states_gen_linkstr (self, norb, nelec, tril=True):
            linkstr = []
            for solver in self.fcisolvers:
                with temporary_env (solver, orbsym=self.orbsym):
                    linkstr.append (solver.gen_linkstr (norb, self._get_nelec (solver, nelec), tril=tril)
                        if getattr (solver, 'gen_linkstr', None) else None)
            return linkstr
                    
        def states_transform_ci_for_orbital_rotation (self, ci0, norb, nelec, umat):
            ci1 = []
            for solver, my_args, _ in self._loop_solver (_state_arg (ci0)):
                ne = self._get_nelec (solver, nelec)
                ndet = [special.comb (norb, n, exact=True) for n in ne]
                try:
                    ci0_i = my_args[0].reshape (ndet)
                    ci1.append (solver.transform_ci_for_orbital_rotation (ci0_i, norb, ne, umat))
                except ValueError as err:
                    ci0_i = my_args[0].reshape ([-1,]+ndet)
                    ci1.append (np.stack ([solver.transform_ci_for_orbital_rotation (c, norb, ne, umat)
                                           for c in ci0_i], axis=0))
            return ci1

        def states_trans_rdm12s (self, ci1, ci0, norb, nelec, link_index=None, **kwargs):
            ci1 = _state_arg (ci1)
            ci0 = _state_arg (ci0)
            link_index = _solver_arg (link_index)
            nelec = _solver_arg ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
            tdm1 = []
            tdm2 = []
            for dm1, dm2 in self._collect ('trans_rdm12s', ci1, ci0, norb, nelec, link_index=link_index, **kwargs):
                tdm1.append (dm1)
                tdm2.append (dm2)
            return tdm1, tdm2

        # TODO: remove this?
        absorb_h1e = states_absorb_h1e
        contract_2e = states_contract_2e
        make_hdiag = states_make_hdiag

    h1ezipped_fcisolver = FCISolver (fcisolver.fcisolvers, fcisolver.weights)
    h1ezipped_fcisolver.__dict__.update (fcisolver.__dict__)
    return h1ezipped_fcisolver

def addr_outer_product (norb_f, nelec_f):
    '''Build index arrays for reshaping a direct product of LAS CI
    vectors into the appropriate orbital ordering for a CAS CI vector'''
    norb = sum (norb_f)
    nelec = sum (nelec_f)
    # Must skip over cases where there are no electrons of a specific spin in a particular subspace
    norbrange = np.cumsum (norb_f)
    addrs = []
    for i in range (0, len (norbrange)):
        irange = range (norbrange[i]-norb_f[i], norbrange[i])
        new_addrs = cistring.sub_addrs (norb, nelec, irange, nelec_f[i]) if nelec_f[i] else []
        if len (addrs) == 0:
            addrs = new_addrs
        elif len (new_addrs) > 0:
            addrs = np.intersect1d (addrs, new_addrs)
    if not len (addrs): addrs=[0] # No beta electrons edge case
    return addrs

def _ci_outer_product (ci_f, norb_f, nelec_f):
    '''Compute outer-product CI vector for one space table from fragment LAS CI vectors.
    See "ci_outer_product"'''
    nfrags = len (norb_f)
    neleca_f = [ne[0] for ne in nelec_f]
    nelecb_f = [ne[1] for ne in nelec_f]
    lroots_f = [1 if ci.ndim<3 else ci.shape[0] for ci in ci_f]
    nprods = np.prod (lroots_f)
    shape_f = [(lroots, cistring.num_strings (norb, neleca), cistring.num_strings (norb, nelecb))
              for lroots, norb, neleca, nelecb in zip (lroots_f, norb_f, neleca_f, nelecb_f)]
    addrs_a = addr_outer_product (norb_f, neleca_f)
    addrs_b = addr_outer_product (norb_f, nelecb_f)
    idx = np.ix_(addrs_a,addrs_b)
    addrs_a = addrs_b = None
    neleca = sum (neleca_f)
    nelecb = sum (nelecb_f)
    nelec = tuple ((neleca, nelecb))
    ndet_a = cistring.num_strings (sum (norb_f), neleca)
    ndet_b = cistring.num_strings (sum (norb_f), nelecb)
    ci_dp = ci_f[-1].reshape (shape_f[-1])
    for ci_r, shape in zip (ci_f[-2::-1], shape_f[-2::-1]):
        lroots, ndeta, ndetb = ci_dp.shape
        ci_dp = np.multiply.outer (ci_dp, ci_r.reshape (shape))
        ci_dp = ci_dp.transpose (0,3,1,4,2,5).reshape (
            lroots*shape[0], ndeta*shape[1], ndetb*shape[2]
        )
    #norm_dp = linalg.norm (ci_dp.reshape (ci_dp.shape[0],-1), axis=1)
    #ci_dp /= norm_dp[:,None,None]
    def gen_ci_dp (buf=None):
        if buf is None:
            ci = np.empty ((ndet_a,ndet_b), dtype=ci_f[-1].dtype)
        else:
            ci = np.asarray (buf.flat[:ndet_a*ndet_b]).reshape (ndet_a, ndet_b)
        #ci_dp = ci_f[-1].reshape (shape_f[-1])
        #for ci_r, shape in zip (ci_f[-2::-1], shape_f[-2::-1]):
        #    lroots, ndeta, ndetb = ci_dp.shape
        #    ci_dp = np.multiply.outer (ci_dp, ci_r.reshape (shape))
        #    ci_dp = ci_dp.transpose (0,3,1,4,2,5).reshape (
        #        lroots*shape[0], ndeta*shape[1], ndetb*shape[2]
        #    )
        #norm_dp = linalg.norm (ci_dp.reshape (ci_dp.shape[0],-1), axis=1)
        #ci_dp /= norm_dp[:,None,None]
        for vec in ci_dp:
            ci[:,:] = 0.0
            ci[idx] = vec[:,:]
            yield ci
    def dotter (c1, nelec1, skip=None):
        if nelec1 != nelec: return np.zeros (nprods)
        if skip is None: return np.dot (ci_dp.reshape (nprods, -1),
                                        c1[idx].ravel ())
        try:
            skip = set (skip)
        except TypeError as e:
            skip = set ([skip])
        c1_dp = c1[idx][None,:]
        skipdims = 1
        for ifrag, (ci_r, shape) in enumerate (zip (ci_f, shape_f)):
            new_shape = [x for s,t in zip (c1_dp.shape[skipdims:],shape[1:]) for x in (s//t, t)]
            new_shape = list(c1_dp.shape[0:skipdims]) + new_shape
            c1_dp = c1_dp.reshape (*new_shape)
            if ifrag in skip:
                dimorder = [0,skipdims+1,skipdims+3,] + list (range(1, skipdims+1)) + [skipdims+2,]
                c1_dp = c1_dp.transpose (*dimorder)
                skipdims += 2
            else:
                c1_dp = np.tensordot (ci_r.reshape (shape), c1_dp,
                                      axes=((1,2),(skipdims+1,skipdims+3)))
                new_shape = [c1_dp.shape[0]*c1_dp.shape[1],] + list (c1_dp.shape[2:])
                c1_dp = c1_dp.reshape (*new_shape)
        c1_dp = c1_dp.reshape (c1_dp.shape[0:skipdims])
        return c1_dp
    return gen_ci_dp, nprods, dotter

def ci_outer_product_generator (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors and return
    result as a generator

    Args:
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (lroots[i,j],ndeta[i,j],ndetb[i,j])
        norb_f : list of length (nfrags)
            Number of orbitals in each fragment
        nelec_fr : ndarray-like of shape (nfrags, nroots, 2)
            Number of spin-up and spin-down electrons in each fragment
            and root

    Returns:
        ci_r_gen : callable that returns a generator of length (nprods)
            Generates all direct-product CAS CI vectors
        nelec_p : list of length (nprods) of tuple of length 2
            (neleca, nelecb) for each product state
        dotter : callable
            Performs the dot product in the outer product basis
            on a CAS CI vector, without explicitly constructing
            any direct-product CAS CI vectors (again).
            Args:
                ket : ndarray
                    contains CAS-CI vector
                nelec_ket : tuple of length 2
                    neleca, nelecb
            Kwargs:
                spinless2ss : callable
                    Converts ci back from the spinless representation
                    into the (neleca,nelecb) representation. Takes c1
                    and nelec1 and returns a new ci vector
                iket : integer
                    Used to filter zero sectors when passed with oporder
                oporder: integer
                    Used to filter zero sectors when passed with oporder
                    ket is identified as product number iket, acted on by
                    some operator of up to order oporder. Any rootspace
                    which differs by more than oporder electron hops
                    therefore has zero projection onto ket.
            Returns:
                ndarray of length (nprods)
                    Expansion coefficients for c1 in terms of direct-
                    product states of ci_fr
    '''

    norb = sum (norb_f)
    ndet = max ([cistring.num_strings (norb, ne[0]) * cistring.num_strings (norb, ne[1])
                for ne in np.sum (nelec_fr, axis=0)])
    gen_ci_r = []
    nelec_p = []
    dotter_r = []
    space_p = []
    for space in range (len (ci_fr[0])):
        ci_f = [ci[space] for ci in ci_fr]
        nelec_f = [nelec[space] for nelec in nelec_fr]
        nelec = (sum ([ne[0] for ne in nelec_f]), sum ([ne[1] for ne in nelec_f]))
        gen_ci, nprods, dotter = _ci_outer_product (ci_f, norb_f, nelec_f)
        gen_ci_r.append (gen_ci)
        nelec_p.extend ([nelec,]*nprods)
        space_p.extend ([space,]*nprods)
        dotter_r.append ([dotter, nelec, nprods])
    def ci_r_gen (buf=None):
        if buf is None:
            buf1 = np.empty (ndet, dtype=ci_fr[-1][0].dtype)
        else:
            buf1 = np.asarray (buf.flat[:ndet])
        for gen_ci in gen_ci_r:
            for x in gen_ci (buf=buf1):
                yield x
    def dotter (ket, nelec_ket, spinless2ss=None, iket=None, oporder=None, skip=None):
        vec = []
        if callable (spinless2ss):
            parse_nelec = lambda nelec: sum(nelec)
        else:
            parse_nelec = lambda nelec: nelec
            spinless2ss = lambda vec, ne: vec
        filtbra = lambda *args: False
        if iket is not None and oporder is not None:
            iket = space_p[iket]
            nelec_f_ket = nelec_fr[:,iket]
            def filtbra (ibra):
                nelec_f_bra = nelec_fr[:,ibra]
                nhop_f = nelec_f_bra-nelec_f_ket
                return np.sum (np.abs(nhop_f))>2*oporder

        for ibra, (dot, nelec_bra, nprods) in enumerate (dotter_r):
            if parse_nelec (nelec_bra) != parse_nelec (nelec_ket) or filtbra (ibra):
                vec.append (np.zeros (nprods))
                continue
            vec.append (dot (spinless2ss (ket, nelec_bra), nelec_bra, skip=skip))
        if skip is None: return np.concatenate (vec)
        else: return vec
    return ci_r_gen, nelec_p, dotter

def ci_outer_product (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors.

    Args:
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (lroots[i,j],ndeta[i,j],ndetb[i,j])
        norb_f : list of length (nfrags)
            Number of orbitals in each fragment
        nelec_fr : ndarray-like of shape (nfrags, nroots, 2)
            Number of spin-up and spin-down electrons in each fragment
            and root

    Returns:
        ci_r : list of length (nroots)
            Contains full CAS CI vector
        nelec_r : list of length (nroots) of tuple of length 2
            (neleca, nelecb) for each product state
    '''
    ci_r_gen, nelec_r, _ = ci_outer_product_generator (ci_fr, norb_f, nelec_fr)
    ci_r = [x.copy () for x in ci_r_gen ()]
    return ci_r, nelec_r

def las2cas_civec (las):
    norb_f = las.ncas_sub
    nelec_fr = [[_unpack_nelec (fcibox._get_nelec (solver, nelecas)) for solver in fcibox.fcisolvers] for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)]
    ci, nelec = ci_outer_product (las.ci, norb_f, nelec_fr)
    return ci, nelec



