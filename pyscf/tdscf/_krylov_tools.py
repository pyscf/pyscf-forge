# Copyright 2024 The PySCF Developers. All Rights Reserved.
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
# Author: Zehao Zhou  zehaozhoucase@gmail.com



import numpy as np
import sys, gc, json
import scipy.linalg


from pyscf.tdscf import math_helper
from pyscf.tdscf.math_helper import get_mem_info, get_avail_cpumem


from pyscf.lib import logger
from functools import partial

from pyscf.data.nist import HARTREE2EV


RIS_PRECOND_CITATION_INFO = '''
Please cite the TDDFT-ris preconditioning method if you are happy with the fast convergence:

    1.  Zhou, Zehao, and Shane M. Parker.
        Converging Time-Dependent Density Functional Theory Calculations in Five Iterations
        with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation
        20, no. 15 (2024): 6738-6746.

    2.  Zhou, Zehao, Fabio Della Sala, and Shane M. Parker.
        Minimal auxiliary basis set approach for the electronic excitation spectra
        of organic molecules. The Journal of Physical Chemistry Letters
        14, no. 7 (2023): 1968-1976.

    3.  Zhou, Zehao, and Shane M. Parker.
        Accelerating molecular property calculations with
        semiempirical preconditioning. The Journal of Chemical Physics 155, no. 20 (2021).


'''


def _time_add(log, t_total, t_start):
    ''' t_total: list
        t_start: tuple

        In-place revise t_total, add the time elapsed since t_start
    '''
    cpu1, wall1 = logger.process_clock(), logger.perf_counter()

    t_total[0] += cpu1 - t_start[0]
    t_total[1] += wall1 - t_start[1]


def _time_profiling(log, t_mvp, t_subgen, t_solve_sub, t_sub2full, t_precond, t_fill_holder, t_total):
    '''
    This function prints out the time and percentage of each submodule

    Args:
    t_xxxx: 2-element list, [<cpu time>, <wall time>]
            each t_xxxx is a timer, the time profiling for each submodule in krylov_solver
            for example, t_mvp is the time profiling for matrix vector product

    example output:

    Timing breakdown:
                            CPU(sec)  wall(sec)    | Percentage
    mat vec product            3.61       3.67     42.9   42.7   93.6
    proj subspace              0.00       0.00      0.0    0.0    0.0
    solve subspace             0.00       0.00     0.0    0.0    0.0
    proj fullspace             0.00       0.00     0.0    0.0    0.0
    precondition               0.37       0.43     428.91     4.4    5.0    5.0
    fill holder                0.01       0.01       9.50     0.1    0.1    0.1
    Sum                        4.00       4.11    8482.78    47.5   47.9   98.8
    Total                      8.42       8.59    8587.89   100.0  100.0  100.0
    '''
    time_labels = ["CPU(sec)", "wall(sec)"]
    labels = time_labels[:len(t_total)]

    log.info("Timing breakdown:")
    header_time = " ".join(f"{label:>10}" for label in labels)
    log.info(f"{'':<20}  {header_time} | Percentage ")

    t_sum = [t_mvp[i] + t_subgen[i] + t_solve_sub[i] + t_sub2full[i] + t_precond[i] + t_fill_holder[i]
                for i in range(len(t_total))]

    ''' also calculate the time percentage for each timer '''
    timers = {
        'mat vec product':t_mvp,
        'proj subspace':  t_subgen,
        'solve subspace': t_solve_sub,
        'proj fullspace': t_sub2full,
        'precondition':   t_precond,
        'fill holder':    t_fill_holder,
        'Sum':            t_sum,
        'Total':          t_total
    }
    for entry, cost in timers.items():
        time_str = " ".join(f"{x:>10.2f}" for x in cost)
        percent_str = " ".join(f"{(x/y*100 if y != .0 else 100):>6.1f}" for x, y in zip(cost, t_total))
        log.info(f"{entry:<20} {time_str}  {percent_str}")



def eigenvalue_diagonal(**kwargs):
    '''solve
        DX=XΩ
        D is diagonal matrix
    '''
    n_states = kwargs['n_states']
    hdiag = kwargs['hdiag']

    hdiag = hdiag.reshape(-1,)
    A_size = hdiag.shape[0]
    Dsort = hdiag.argsort()[:n_states].reshape(-1,)

    X = np.zeros((n_states, A_size),dtype=hdiag.dtype)
    for i in range(n_states):
        X[i, Dsort[i]] = hdiag.dtype.type(1.0)
    _converged, _energies = True, None
    # print('X norm', np.linalg.norm(X, axis=1))
    return _converged, _energies, X

def linear_diagonal(**kwargs):
    ''' solve  DX=rhs,
        where D is a diagonal matrix'''
    hdiag = kwargs['hdiag']
    rhs = kwargs['rhs']

    _converged = True
    return _converged, rhs / hdiag

def shifted_linear_diagonal(**kwargs):
    '''
    solve shifted linear system, where D is a diagonal matrix
    DX - XΩ = rhs
    X = r/(D-Ω)
    Args:
        rhs: 2D array
            right hand side of the linear system
        hdiag: 1D array
            diagonal of the Hamiltonian matrix
        omega: 1D array
            diagonal of the shift matrix
    return X (X is in-place modified rhs)
    '''

    rhs = kwargs['rhs']
    hdiag = kwargs['hdiag']
    omega = kwargs['omega_shift']

    rhs = rhs.astype(dtype=hdiag.dtype, copy=False)
    omega = omega.astype(dtype=hdiag.dtype, copy=False)

    n_states = rhs.shape[0]
    assert n_states == len(omega)
    t = hdiag.dtype.type(1e-14)

    # omega = omega.reshape(-1,1)
    # D = np.repeat(hdiag.reshape(1,-1), n_states, axis=0) - omega
    # '''
    # force all small values not in [-t,t]
    # '''
    # D = np.where( abs(D) < t, np.sign(D)*t, D)
    # X = rhs/D
    # del rhs, D

    X = np.empty_like(rhs)
    for i in range(n_states):
        Di = hdiag - omega[i]                    # 1D: len(hdiag)

        # Replace |Di| < t with sign(Di)*t  (avoid np.abs to save memory)
        mask = (Di > -t) & (Di < t)              # Boolean mask for near-zero
        Di = np.where(mask, np.sign(Di) * t, Di)  # In-place friendly

        X[i] = rhs[i] / Di                       # Element-wise division
        # rhs[i] /= Di # danger of modifying rhs in-place!!!

        # Optional: clean small intermediates early
        del Di, mask
    # X = rhs
    gc.collect()
    _converged = True
    return _converged, X

def shifted_linear_diagonal_inplace(**kwargs):
    '''
    solve shifted linear system, where D is a diagonal matrix
    DX - XΩ = rhs
    X = r/(D-Ω)
    Args:
        rhs: 2D array
            right hand side of the linear system
        hdiag: 1D array
            diagonal of the Hamiltonian matrix
        omega: 1D array
            diagonal of the shift matrix
    return X (X is in-place modified rhs)
    '''

    rhs = kwargs['rhs']
    hdiag = kwargs['hdiag']
    omega = kwargs['omega_shift']

    rhs = rhs.astype(dtype=hdiag.dtype, copy=False)
    omega = omega.astype(dtype=hdiag.dtype, copy=False)

    n_states = rhs.shape[0]
    assert n_states == len(omega)
    t = hdiag.dtype.type(1e-14)

    # X = np.empty_like(rhs)
    for i in range(n_states):
        Di = hdiag - omega[i]                    # 1D: len(hdiag)

        # Replace |Di| < t with sign(Di)*t  (avoid np.abs to save memory)
        mask = (Di > -t) & (Di < t)
        force = np.sign(Di) * t            # Boolean mask for near-zero
        Di = np.where(mask, force, Di)  # In-place friendly

        # X[i] = rhs[i] / Di                       # Element-wise division
        rhs[i] /= Di # danger of modifying rhs in-place!!!

        # Optional: clean small intermediates early
        del Di, force, mask
        gc.collect()
    X = rhs
    gc.collect()
    _converged = True
    return _converged, X


'''for each problem type, setup diagonal initial guess and preconitioner '''

'''eigenvalue problem'''
_eigenvalue_diagonal_initguess = eigenvalue_diagonal
# _eigenvalue_diagonal_precond  = shifted_linear_diagonal
_eigenvalue_diagonal_precond  = shifted_linear_diagonal_inplace


'''linear problem'''
_linear_diagonal_initguess = linear_diagonal
_linear_diagonal_precond   = linear_diagonal


'''shifted linear problem'''
# _shifted_linear_diagonal_initguess = shifted_linear_diagonal
# _shifted_linear_diagonal_precond   = shifted_linear_diagonal
_shifted_linear_diagonal_initguess = shifted_linear_diagonal
_shifted_linear_diagonal_precond   = shifted_linear_diagonal_inplace

def krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
                  initguess_fn=None, precond_fn=None, rhs=None,
                  omega_shift=None, n_states=20,conv_tol=1e-5,conv_tol_scaling=0.1,
                  max_iter=35, extra_init=8, gs_initial=False, gram_schmidt=True,
                  restart_subspace=None, single=False,
                  verbose=logger.NOTE):
    '''
        This solver is used to solve the following problems:
        (1) Eigenvalue problem, return Ω and X
                    AX = XΩ

        (2) Linear system, return X
                    AX = rhs.
            e.g. CPKS problem (A+B)Z = R in TDDFT gradient calculation

        (3) Shifted linear system , return X
                 AX - XΩ = rhs, where Ω is a diagonal matrix.
            e.g. preconditioning,  Green's function

    Theory:
    (1) Eigenvalue problem
           AX = XΩ
        A(Vx) = (Vx)Ω
        V.TAV x = V.TV xΩ
        ax = sxΩ,
        whehre basis overlap s=V.TV, W=AV
        residual r = AX - XΩ = Wx - XΩ

    (2) Linear system
          AX = P
        A(Vx) = P
        V.TAV x = V.TP
        ax = p,
        where p = V.TP (but note that P != Vp)
        residual r = AX - P = Wx - P

    (3) Shifted linear system
        AX - XΩ = P   (P denotes rhs)
        A(Vx) - (Vx) Ω = P
        V.TAV x - V.TV xΩ = V.TP
        ax - sxΩ = p
        residual r = AX - XΩ - P = Wx - XΩ - P

    Args:
        matrix_vector_product: function
            matrix vector product
            e.g. def mvp(X):
                    return A.dot(X)
        hdiag: 1D array
            diagonal of the Hamiltonian matrix
        problem_type: str
            'eigenvalue', 'linear' or 'shifted_linear'
        initguess_fn: function
            function to generate initial guess
        precond_fn: function
            function to apply preconditioner

        -- for eigenvalue problem:
            n_states: int
                number of states to be solved, required, default 20

        -- for linear and shifted_linear problem:
            rhs: 2D array
                right hand side of the linear system, required

        -- for shifted_linear problem:
            omega_shift: 1D array
                diagonal of the shift matrix, required

        conv_tol: float
            convergence tolerance
        max_iter: int
            maximum iterations
        extra_init: int
            extra number of states to be initialized
        restart_iter: int or None
            restart the Krylov solver periodically after this iteration. Default None, no restart.
        gs_initial: bool
            apply gram_schmidt procedure on the initial guess,
            only in the case of gram_schmidt = True, but given wired initial guess
        gram_schmidt: bool
            use Gram-Schmidt orthogonalization
        single: bool
            use single precision
        verbose: logger.Logger
            logger object

    Returns:
        converged: the index of converged states/vectors
        omega: 1D array
            eigenvalues
        X: 2D array  (in c-order, each row is a solution vector)
            eigenvectors or solution vectors
    '''

    if problem_type not in ['eigenvalue', 'linear', 'shifted_linear']:
        raise ValueError('Invalid problem type, please choose either eigenvalue, linear or shifted_linear.')

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if single:
        log.info('Using single precision')
        # assert hdiag.dtype == np.float32
        hdiag = hdiag.astype(np.float32, copy=False)
    else:
        log.info('Using double precision')
        # assert hdiag.dtype == np.float64
        hdiag = hdiag.astype(np.float64, copy=False)



    log.info(f'====== {problem_type.capitalize()} Krylov Solver Starts ======')
    log.TIMER_LEVEL = 4
    log.DEBUG1 = 4

    log.info(f'n_states={n_states}, conv_tol={conv_tol}, max_iter={max_iter}, extra_init={extra_init}')

    ''' detailed timing for each sub module
        cpu0 = (cpu time, wall time, gpu time)'''
    cpu0 = (logger.process_clock(), logger.perf_counter())
    t_mvp         = [.0, .0]
    t_subgen      = [.0, .0]
    t_solve_sub   = [.0, .0]
    t_sub2full    = [.0, .0]
    t_precond     = [.0, .0]
    t_fill_holder = [.0, .0]
    t_total       = [.0, .0]

    A_size = hdiag.shape[0]
    log.info(f'Size of A matrix = {A_size}')
    if problem_type == 'eigenvalue':
        if extra_init is None:
            extra_init = 8
            size_new = min([n_states + extra_init, 2 * n_states, A_size])
        else:
            size_new = min([n_states + extra_init, A_size])
    elif problem_type in ['linear','shifted_linear']:
        if rhs is None:
            raise ValueError('rhs is required for linear or shifted_linear problem.')

        size_new = rhs.shape[0]
        n_states = rhs.shape[0]

    # record the number of extra initial vectors
    n_extra_init = size_new - n_states

    log.info(f'single trial vector X_ia size: {A_size*hdiag.itemsize/1024**2:.2f} MB')
    if restart_subspace is None:
        ''' calculate the maximum number of vectors allowed by the memory'''
        restart_subspace = int((get_avail_cpumem() * 0.9) // (2*A_size*hdiag.itemsize))
        log.info(f'the maximum number of vectors allowed by the memory is {restart_subspace}')
    else:
        log.info(f'user specified the maximum number of vectors is {restart_subspace}')

    max_N_mv = min(size_new + max_iter * n_states, restart_subspace)
    log.info(f'the maximum number of vectors in V_holder and W_holder is {max_N_mv}')

    # Initialize arrays
    V_holder_mem = max_N_mv*A_size*hdiag.itemsize/1024**3

    rss = get_avail_cpumem() / 1024**3 # current memory usage in GB
    log.info(f'the maximum CPU memory usage throughout the Krylov solver is around {2*V_holder_mem + rss:.2f} GB')


    V_holder = np.empty((max_N_mv, A_size), dtype=hdiag.dtype)
    W_holder = np.empty_like(V_holder)

    log.info(f'V_holder {V_holder_mem:.2f} GB')
    log.info(f'W_holder {V_holder_mem:.2f} GB')
    log.info(f'dtype of V_holder & W_holder {V_holder.dtype}')

    sub_A_holder = np.empty((max_N_mv, max_N_mv), dtype=hdiag.dtype)

    if problem_type in ['linear','shifted_linear']:
        '''Normalize RHS for linear system'''
        rhs_norm = np.linalg.norm(rhs, axis=1, keepdims=True)
        rhs = rhs/rhs_norm
        sub_rhs_holder = np.empty((max_N_mv, rhs.shape[0]), dtype=hdiag.dtype)


    # Setup basis projection method
    if gram_schmidt:
        log.info('Using Gram-Schmidt orthogonalization')
        fill_holder = partial(math_helper.Gram_Schmidt_fill_holder, double=True)

    else:
        log.info('Using non-orthogonalized Krylov subspace (nKs) method.')
        nks_citation = '''
        Furche, Filipp, Brandon T. Krull, Brian D. Nguyen, and Jake Kwon.
        Accelerating molecular property calculations with nonorthonormal Krylov space methods.
        The Journal of Chemical Physics 144, no. 17 (2016).
        '''
        log.info(nks_citation)
        fill_holder = math_helper.nKs_fill_holder
        s_holder = np.empty_like(sub_A_holder)

    if initguess_fn and callable(initguess_fn):
        log.info(' use user-specified function to generate initial guess.')
    else:
        log.info(' use hdiag to generate initial guess.')

        initguess_functions = {
            'eigenvalue':     _eigenvalue_diagonal_initguess,
            'linear':         _linear_diagonal_initguess,
            'shifted_linear': _shifted_linear_diagonal_initguess,
        }
        initguess_fn = initguess_functions[problem_type]


    ''' Generate initial guess '''
    log.info('generating initial guess')
    cpu0 = (logger.process_clock(), logger.perf_counter())

    if problem_type == 'eigenvalue':
        _converged, _energies, init_guess_X = initguess_fn(n_states=size_new, hdiag=hdiag)

    elif problem_type == 'linear':
        _converged, init_guess_X = initguess_fn(hdiag=hdiag, rhs=rhs)

    elif problem_type =='shifted_linear':
        omega_shift = np.asarray(omega_shift, dtype=hdiag.dtype)
        _converged, init_guess_X = initguess_fn(hdiag=hdiag, rhs=rhs, omega_shift=omega_shift)
    log.timer(f' {problem_type.capitalize()} initguess_fn cost', *cpu0)


    cpu0 = (logger.process_clock(), logger.perf_counter())

    log.info(get_mem_info('before put initial guess into V_holder'))
    size_old = 0
    if gs_initial:
        '''initial guess were already orthonormalized'''
        log.info(' initial guess were already orthonormalized, no need Gram_Schmidt here')
        # size_new = math_helper.nKs_fill_holder(V_holder, 0, init_guess_X[n_states:, :])# first fill extra_init vectors
        # size_new = math_helper.nKs_fill_holder(V_holder, size_new, init_guess_X[:n_states, :]) # n_states vectors
        extra_init_X = init_guess_X[n_states:, :]
        V_holder[:n_extra_init, :] = extra_init_X
        del extra_init_X

        n_states_X = init_guess_X[:n_states, :]
        V_holder[n_extra_init:n_extra_init+n_states, :] = n_states_X
        del n_states_X
        size_new = init_guess_X.shape[0]

    else:
        log.info(' put initial guess into V_holder')
        if n_extra_init > 0:
            size_new = fill_holder(V_holder, 0, init_guess_X[n_states:, :])# first fill extra_init vectors
            size_new = fill_holder(V_holder, size_new, init_guess_X[:n_states, :]) # n_states vectors
        else:
            size_new = fill_holder(V_holder, size_old, init_guess_X)


    del init_guess_X
    gc.collect()

    log.timer(f' {problem_type.capitalize()} init_guess_X fill_holder cost', *cpu0)


    log.info('initial guess done')


    if precond_fn and callable(precond_fn):
        log.info(' use user-specified function for preconditioning.')
    else:
        log.info(' use hdiag for preconditioning.')
        precond_functions = {
            'eigenvalue':     _eigenvalue_diagonal_precond,
            'linear':         _linear_diagonal_precond,
            'shifted_linear': _shifted_linear_diagonal_precond,
        }
        precond_fn = precond_functions[problem_type]
        precond_fn = partial(precond_fn, hdiag=hdiag)

    eigenvalue_record = []
    residual_record = []
    n_mvp_record = []
    ''' Davidson iteration starts!
    '''
    for ii in range(max_iter):

        ''' Matrix-vector product '''
        t0 = (logger.process_clock(), logger.perf_counter())

        log.info(get_mem_info(f' ▶ ------- iter {ii+1:<3d} MVP starts, {size_new-size_old} vectors'))
        X = V_holder[size_old:size_new, :]


        log.info(f'     X {X.shape} {X.nbytes//1024**2} MB')
        log.info(f'     V_holder[:size_new, :] memory usage {V_holder[:size_new, :].nbytes/1024**3:.2f} GB')
        log.info(f'     subspace size / maximum subspace size: {size_new} / {max_N_mv}')


        mvp = matrix_vector_product(X)
        del X
        gc.collect()

        log.info(get_mem_info('     after MVP'))


        _time_add(log, t_mvp, t0)

        ''' Project into Krylov subspace '''
        t0 = (logger.process_clock(), logger.perf_counter())
        sub_A_holder = math_helper.gen_VW_symmetry(sub_A_holder, V_holder, mvp, size_old, size_new)
        log.info(get_mem_info('     sub_A_holder updated'))


        W_holder[size_old:size_new, :] = mvp
        del mvp
        gc.collect()
        log.info(get_mem_info('     MVP stored in W_holder'))


        sub_A = sub_A_holder[:size_new, :size_new]
        if problem_type in ['linear','shifted_linear']:
            sub_rhs_holder = math_helper.gen_VP(sub_rhs_holder, V_holder, rhs, size_old, size_new)
            sub_rhs = sub_rhs_holder[:size_new, :]

        _time_add(log, t_subgen, t0)

        ''' solve subsapce problem
            solution x is column-wise vectors
            each vetcor contains elements of linear combination coefficient of projection basis
        '''
        t0 = (logger.process_clock(), logger.perf_counter())
        if not gram_schmidt:
            ''' no Gram Schidmit procedure, need the overlap matrix of projection basis'''
            math_helper.gen_VW(s_holder, V_holder, V_holder, size_old, size_new, symmetry=True)
            overlap_s = s_holder[:size_new, :size_new]
            log.info(get_mem_info('     overlap_s calculated'))

        if problem_type == 'eigenvalue':
            if gram_schmidt:
                ''' solve ax=xΩ '''
                omega, x = np.linalg.eigh(sub_A)
            else:
                ''' solve ax=sxΩ
                # preconditioned solver: d^-1/2 s d^-1/2'''
                omega, x = math_helper.solve_AX_SX(sub_A, overlap_s)

            omega = omega[:n_states]
            x = x[:, :n_states]
            log.info(f' Energies (eV): {[round(e,3) for e in (omega*HARTREE2EV).tolist()]}')

        elif problem_type == 'linear':
            x = np.linalg.solve(sub_A, sub_rhs)

        elif problem_type == 'shifted_linear':
            if gram_schmidt:
                ''' solve ax - xΩ = sub_rhs '''
                x = math_helper.solve_AX_Xla_B(sub_A, omega_shift, sub_rhs)
            else:
                ''' solve ax - s xΩ = sub_rhs
                    => s^-1 ax - xΩ = s^-1 sub_rhs
                # TODO maybe need precondition step: s/d first'''
                s_inv = np.linalg.inv(overlap_s)
                x = scipy.linalg.solve_sylvester(s_inv.dot(sub_A), -np.diag(omega_shift), s_inv.dot(sub_rhs))

        _time_add(log, t_solve_sub, t0)
        log.info(get_mem_info('     after solving subspace'))
        t0 = (logger.process_clock(), logger.perf_counter())

        ''' compute the residual
            full_X is current guess solution
            AX is A.dot(full_X)'''

        xT = x.T
        log.info(get_mem_info('     before AX = xTW'))
        residual = xT.dot(W_holder[:size_new,:])

        log.info(get_mem_info('     after AX = xTW'))

        if problem_type == 'eigenvalue':
            ''' r = AX - XΩ
                  = AVx - VxΩ
                  = Wx - VxΩ '''
            X_omega = (omega[:,None] * xT).dot(V_holder[:size_new,:])
            residual -= X_omega
            del X_omega

        elif problem_type == 'linear':
            ''' r = AX - rhs '''
            residual -= rhs

        elif problem_type == 'shifted_linear':
            ''' r = AX - X omega_shift - rhs '''
            X_omega_shift = (omega_shift[:,None] * xT).dot(V_holder[:size_new,:])
            residual -= X_omega_shift
            del X_omega_shift
            residual -= rhs

        log.info(get_mem_info('     residual computed'))
        gc.collect()

        _time_add(log, t_sub2full, t0)

        ''' Check convergence '''
        r_norms = np.linalg.norm(residual, axis=1)

        if problem_type == 'eigenvalue':
            eigenvalue_record.append((omega*HARTREE2EV).tolist())
        residual_record.append(r_norms.tolist())

        if log.verbose >= 5:
            data = {
                "A_size": A_size,
                "n_states": n_states,
                "n_extra_init": n_extra_init,
                "conv_tol": conv_tol,
                "max_iter": max_iter,
                "restart_subspace": restart_subspace,
                "max_N_mv": max_N_mv,
                "in_ram": in_ram,
                "problem_type": problem_type,
                "n_iterations": ii+1,
                "eigenvalue_history": eigenvalue_record if problem_type == 'eigenvalue' else None,
                "residual_norms_history":residual_record,
                "n_mvp_history": n_mvp_record,
            }

            with open('iter_record.json', 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                log.info('iter_record.json saved')

        max_idx = np.argmax(r_norms)
        log.debug('              state :  |R|2  unconverged')
        for state in range(len(r_norms)):
            if r_norms[state] < conv_tol:
                log.debug(f'              {state+1:>5d} {r_norms[state]:.2e}')
            else:
                log.debug(f'              {state+1:>5d} {r_norms[state]:.2e} *')

        max_norm = np.max(r_norms)
        log.info(f'              max|R|: {max_norm:>12.2e}, state {max_idx+1}')

        if max_norm < conv_tol or ii == (max_iter - 1):
            break

        else:

            unconverged_idx = np.where(r_norms.ravel() > conv_tol_scaling * conv_tol)[0]
            log.info(f'              number of unconverged states: {unconverged_idx.size}')

            if size_new + unconverged_idx.size > max_N_mv:
                log.info(f'     !!! restart subspace (subspace {size_new+unconverged_idx.size} > {max_N_mv})')
                ''' fill N_state solution into the V_holder, but keep the extra initial guess vectors
                    W_holder is also restarted to fully remove the numerical noise
                '''
                del residual
                current_X = x.T.dot(V_holder[:size_new,:])

                size_old = n_extra_init
                size_new = fill_holder(V_holder, size_old, current_X)

                del current_X
                gc.collect()

            else:
                ''' Preconditioning step '''
                # index_bool = r_norms > conv_tol
                t0 = (logger.process_clock(), logger.perf_counter())
                log.info(get_mem_info('     ▸ Preconditioning starts'))

                # residual_unconv = residual[index_bool, :] with boolean indexing creates a copy,
                #  which costs extra memory
                # instead, manually move the unconverged residual vectors forehead,
                # use residual[:unconverged_idx.size, :] to save memory


                pos = 0
                for idx in unconverged_idx:
                    if idx != pos:
                        residual[pos,:] = residual[idx,:]
                    pos += 1

                residual_unconv = residual[:unconverged_idx.size, :]

                if problem_type == 'eigenvalue':
                    _converged, X_new = precond_fn(rhs=residual_unconv, omega_shift=omega[unconverged_idx])
                elif problem_type == 'linear':
                    _converged, X_new = precond_fn(rhs=residual_unconv)
                elif problem_type =='shifted_linear':
                    _converged, X_new = precond_fn(rhs=residual_unconv, omega_shift=omega_shift[unconverged_idx])
                log.timer('          preconditioning', *t0)
                del residual_unconv
                gc.collect()

                _time_add(log, t_precond, t0)

                ''' put the new guess X into the holder '''
                t0 = (logger.process_clock(), logger.perf_counter())
                log.info(get_mem_info('     ▸ Preconditioning ends'))

                # _V_holder, size_new = fill_holder(V_holder, size_old, X_new)
                log.info('     putting new guesses into the holder')

                size_old = size_new
                size_new = fill_holder(V_holder, size_old, X_new)
                log.timer('     new guesses put into the holder', *t0)

                del X_new, residual
                gc.collect()
                log.info(get_mem_info('     ▸ new guesses put into the holder'))

                # if gram_schmidt:
                #     log.info(f'V_holder orthonormality: {math_helper.check_orthonormal(V_holder[:size_new, :])}')
                if size_new == size_old:
                    log.info('All new guesses kicked out during filling holder !!!!!!!')
                    break
                _time_add(log, t_fill_holder, t0)

    if ii == max_iter - 1 and max_norm >= conv_tol:
        log.info(f'=== {problem_type.capitalize()} Krylov Solver not converged below {conv_tol:.2e} due to max iteration limit ! ===')
        log.info(f'Current residual norms: {r_norms.tolist()}')
        log.info(f'max residual norms {np.max(r_norms)}')

    converged = r_norms <= conv_tol

    log.info(f'Finished in {ii+1} steps')
    log.info(f'Maximum residual norm = {max_norm:.2e}')
    log.info(f'Final subspace size = {sub_A.shape[0]}')

    full_X = x.T.dot(V_holder[:size_new,:])

    if problem_type in['linear', 'shifted_linear']:
        full_X *= rhs_norm

    _time_add(log, t_total, cpu0)

    log.timer(f'{problem_type.capitalize()} Krylov Solver total cost', *cpu0)
    _time_profiling(log, t_mvp, t_subgen, t_solve_sub, t_sub2full, t_precond, t_fill_holder, t_total)

    log.info(f'========== {problem_type.capitalize()} Krylov Solver Done ==========')

    del V_holder, W_holder
    gc.collect()

    if problem_type == 'eigenvalue':
        return converged, omega, full_X
    elif problem_type in ['linear', 'shifted_linear']:
        return converged, full_X

def nested_krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
        rhs=None, omega_shift=None, n_states=20, conv_tol=1e-5,
        max_iter=8, gram_schmidt=True, single=False, verbose=logger.INFO,
        init_mvp=None, precond_mvp=None, extra_init=3, extra_init_diag=8,
        init_conv_tol=1e-3, init_max_iter=10,
        precond_conv_tol=1e-2, precond_max_iter=10,
        init_restart_iter=None, precond_restart_iter=None, restart_iter=None):
    '''
    Wrapper for Krylov solver to handle preconditioned eigenvalue, linear, or shifted linear problems.
    requires the non-diagonal approximation of A matrix, i.e., ris approximation.

    Args:
        matrix_vector_product: Callable, computes AX.
        hdiag: 1D cupy array, diagonal of the Hamiltonian matrix.
        problem_type: str, 'eigenvalue', 'linear', 'shifted_linear'.
        rhs: 2D cupy array, right-hand side for linear systems (default: None).
        omega_shift: Diagonal matrix for shifted linear systems (default: None).
        n_states: int, number of eigenvalues or vectors to solve.
        conv_tol: float, convergence tolerance.
        max_iter: int, maximum iterations.
        gram_schmidt: bool, use Gram-Schmidt orthogonalization.
        single: bool, use single precision.
        verbose: logger.Logger or int, logging verbosity.
        init_mvp: Callable, matrix-vector product for initial guess (default: None).
        precond_mvp: Callable, matrix-vector product for preconditioner (default: None).
        init_conv_tol: float, convergence tolerance for initial guess.
        init_max_iter: int, maximum iterations for initial guess.
        precond_conv_tol: float, convergence tolerance for preconditioner.
        precond_max_iter: int, maximum iterations for preconditioner.

    Returns:
        Output of krylov_solver.
    '''

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    dtype = np.float32 if single else np.float64
    log.info(f'precision {dtype}')
    if single:
        log.info('Using single precision')
        hdiag = hdiag.astype(np.float32, copy=False)
    else:
        log.info('Using double precision')
        hdiag = hdiag.astype(np.float64, copy=False)

    # Validate problem type
    if problem_type not in ['eigenvalue', 'linear', 'shifted_linear']:
        raise ValueError('Invalid problem type, please choose either eigenvalue, linear or shifted_linear.')

    # Define micro_init_precond mapping
    #    the problem_type of
    #    macro problem      intial guess      preconditioner
    micro_init_precond = {
        'eigenvalue':     ['eigenvalue',     'shifted_linear'],
        'linear':         ['linear',         'linear'        ],
        'shifted_linear': ['shifted_linear', 'shifted_linear']
    }

    # Setup initial guess
    if callable(init_mvp):
        log.info('Using iterative initial guess')

        init_problem_type = micro_init_precond[problem_type][0]
        initguess_fn = partial(
            krylov_solver,
            problem_type=init_problem_type, hdiag=hdiag,
            matrix_vector_product=init_mvp,
            conv_tol=init_conv_tol, max_iter=init_max_iter,
            restart_iter=init_restart_iter,
            gram_schmidt=gram_schmidt, single=single, verbose=log.verbose-2
        )
    else:
        log.info('Using diagonal initial guess')
        initguess_fn = None

    # Setup preconditioner
    if callable(precond_mvp):
        log.info('Using iterative preconditioner')

        precond_problem_type = micro_init_precond[problem_type][1]
        precond_fn = partial(
            krylov_solver,
            problem_type=precond_problem_type, hdiag=hdiag,
            matrix_vector_product=precond_mvp,
            conv_tol=precond_conv_tol, max_iter=precond_max_iter,
            restart_iter=precond_restart_iter,
            gram_schmidt=gram_schmidt, single=single, verbose=log.verbose-1
        )
    else:
        log.info('Using diagonal preconditioner')
        precond_fn = None

    if not init_mvp and not precond_mvp:
        log.warn(f'diagonal initial guess and preconditioner provided, using extra_init={extra_init_diag}')
        extra_init = extra_init_diag

    # Run solver
    output = krylov_solver(
        matrix_vector_product=matrix_vector_product, hdiag=hdiag,
        problem_type=problem_type, n_states=n_states,
        rhs=rhs, omega_shift=omega_shift, extra_init=extra_init,
        initguess_fn=initguess_fn, precond_fn=precond_fn,
        conv_tol=conv_tol, max_iter=max_iter,
        gram_schmidt=gram_schmidt, single=single, verbose=verbose,
        restart_iter=restart_iter,
    )
    log.info(RIS_PRECOND_CITATION_INFO)
    return output


def example_krylov_solver():

    np.random.seed(42)
    A_size = 1000
    n_vec = 5
    A = np.random.rand(A_size,A_size)*0.01
    A = A + A.T
    scaling = 30
    np.fill_diagonal(A, (np.random.rand(A_size)+2) * scaling)
    omega_shift = (np.random.rand(n_vec)+2) * scaling
    rhs = np.random.rand(n_vec, A_size) * scaling

    def matrix_vector_product(x):
        return x.dot(A)

    hdiag = np.diag(A)

    _converged, eigenvalues, eigenvecters = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='eigenvalue', n_states=5,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    _converged, solution_vectors = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='linear', rhs=rhs,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    _converged, solution_vectors_shifted = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='shifted_linear', rhs=rhs, omega_shift=omega_shift,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    return eigenvalues, eigenvecters, solution_vectors, solution_vectors_shifted