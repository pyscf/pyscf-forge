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
import scipy
import gc, os
from pyscf.lib import einsum
from pyscf.lib.misc import current_memory

import re

FALLBACK_MB = 32000

MAX_MEMORY_MB = int(os.environ.get('PYSCF_MAX_MEMORY', 0))

if MAX_MEMORY_MB <= 0:
    try:
        with open('/proc/meminfo') as f:
            content = f.read()
        m = re.search(r'^MemTotal:\s+(\d+)', content, re.M)
        if m:
            kb = int(m.group(1))
            MAX_MEMORY_MB = kb // 1024
        else:
            MAX_MEMORY_MB = FALLBACK_MB
    except OSError:
        MAX_MEMORY_MB = FALLBACK_MB

MAX_MEMORY = MAX_MEMORY_MB / 1024 # in GB

def get_mem_info(words):
    rss = current_memory()[0] / 1024  # MB to GB
    memory_info = f"{words:35s} *** mem info: {rss:5.1f} / {MAX_MEMORY:5.1f} GB "
    return memory_info

def get_avail_cpumem():
    rss = current_memory()[0]  # in MB
    free_mem = MAX_MEMORY*1024 - rss
    free_mem *= 1024**2  # in bytes
    return free_mem

def matrix_power(S,a, epsilon=None):
    '''X == S^a'''
    s,ket = np.linalg.eigh(S)
    # s = s**a
    if epsilon:
        if s[0] < epsilon:
            raise LinearDependencyError(f'Matrix is singular. Min eigen = {s[0]}')
        valid_indices = s >= epsilon
        s = s[valid_indices]
        ket = ket[:, valid_indices]

    s = s**a

    X = np.dot(ket*s,ket.T)

    return X



def gen_anisotropy(a):

    # an alternative formula for anisotropy:
    # anis = (xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2 + 6*(xz**2 + xy**2 + yz**2)
    # anis = 0.5*anis
    # anis = anis**0.5

    a = 0.5*(a.T + a)
    tr = (1.0/3.0)*np.trace(a)
    xx = a[0,0]
    yy = a[1,1]
    zz = a[2,2]

    xy = a[0,1]
    xz = a[0,2]
    yz = a[1,2]

    ssum = xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2)
    anis = (1.5 * abs(ssum - 3*tr**2))**0.5
    return float(tr), float(anis)

def utriangle_symmetrize(A):
    upper = np.triu_indices(n=A.shape[0], k=1)
    lower = (upper[1], upper[0])
    A[lower] = A[upper]
    return A


def gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new, symmetry=False):
    '''

    [ V_old ] [W_old.T, W_new.T] = [VW_old,        V_old W_new.T] = [VW_old,   V_old W_new.T  ]
    [ V_new ]                      [V_new W_old.T, V_new W_new.T]   [  V_new  W_current.T     ]

    symmetry: whether sub_A is symmatric

                        V_holder (W_holder is same set up)

            |------------------------------------------------|
            |      V_old                                     |
size_old    |------------------------------------------------|  [ V_current ]
            |      V_new                                     |
size_new    |------------------------------------------------|
            |                                                |
            |      empty                                     |
            |                                                |
            |------------------------------------------------|




    sub_A_holder

                            size_old            size_new
                |---------------|-----------------|-----------------|
                |               |                 |                 |
                |               |                 |                 |
                | V_old W_old.T |  V_old W_new.T  |                 |
                |  (=sub_A_old) |                 |                 |
                |               |                 |                 |
      size_old  |---------------(V_currentW_new.T)|-----------------|
                | if symmetry:  |                 |                 |
                | V_new W_old.T |  V_new W_new.T  |                 |
                |               |                 |                 |
                |               |                 |                 |
                |               |                 |                 |
      size_new  |---------------------------------|-----------------|
                |               |                 |                 |
                |               |                 |                 |
                |               |                 |                 |
                |               |                 |                 |
                |               |                 |                 |
                |               |                 |                 |
                |---------------|-----------------|-----------------|
    '''

    sub_A_tmp = einsum('mn,ln->ml', V_holder[:size_new,:],W_holder[size_old:size_new, :])

    sub_A_holder[:size_new, size_old:size_new] = sub_A_tmp
    del sub_A_tmp
    gc.collect()

    if size_old > 0:
        if symmetry:
            sub_A_tmp = sub_A_holder[:size_old, size_old:size_new].T

            sub_A_holder[size_old:size_new, :size_old] = sub_A_tmp
            del sub_A_tmp
            gc.collect()
        else:
            sub_A_tmp = einsum('mn,ln->lm', W_holder[:size_old, :], V_holder[size_old:size_new,:])
            sub_A_holder[size_old:size_new, :size_old] = sub_A_tmp
            del sub_A_tmp
            gc.collect()
    return sub_A_holder

def gen_VW_symmetry(sub_A_holder, V_holder, W_new, size_old, size_new):
    '''
    a symmetric version of gen_VW, W_new in GPU already. W_new is mvp
    '''

    sub_A_tmp = einsum('mn,ln->ml', V_holder[:size_new,:],W_new)

    sub_A_holder[:size_new, size_old:size_new] = sub_A_tmp
    del sub_A_tmp
    gc.collect()

    if size_old > 0:
        sub_A_tmp = sub_A_holder[:size_old, size_old:size_new].T

        sub_A_holder[size_old:size_new, :size_old] = sub_A_tmp
        del sub_A_tmp
        gc.collect()
    return sub_A_holder



def gen_VP(sub_rhs_holder, V_holder, rhs, size_old, size_new):
    '''
    [ V_old ] [rhs.T] = [V_old rhs.T]
    [ V_new ]           [V_new rhs.T]
    '''
    V_new = V_holder[size_old:size_new:,:]
    # sub_rhs_holder[size_old:size_new,:] = np.dot(V_new, rhs.T)
    sub_rhs_holder[size_old:size_new,:] = einsum('mn,ln->ml', V_new, rhs)
    return sub_rhs_holder


def gen_sub_pq(V_holder, W_holder, P, Q, VP_holder, WQ_holder, WP_holder, VQ_holder, size_old, size_new):
    '''
    [ V_old ] [rhs.T] = [V_old rhs.T]
    [ V_new ]           [V_new rhs.T]
    '''
    VP_holder = gen_VP(VP_holder, V_holder, P, size_old, size_new)

    VQ_holder = gen_VP(VQ_holder, V_holder, Q, size_old, size_new)

    WP_holder = gen_VP(WP_holder, W_holder, P, size_old, size_new)

    WQ_holder = gen_VP(WQ_holder, W_holder, Q, size_old, size_new)

    p = VP_holder[:size_new,:] + WQ_holder[:size_new,:]
    q = WP_holder[:size_new,:] + VQ_holder[:size_new,:]

    return p, q, VP_holder, WQ_holder, WP_holder, VQ_holder


def Gram_Schmidt_fill_holder(V, count, vecs, double = True):
    '''V is a vectors holder
       count is the amount of vectors that already sit in the holder
       nvec is amount of new vectors intended to fill in the V
       count will be final amount of vectors in V

       this version is io-efficeint
    '''
    # if count == 0:
    #     return V

    n_new_vectors, A_size = vecs.shape
    assert V.shape[1] == A_size
    assert n_new_vectors >=1
    # print('n_new_vectors', n_new_vectors)

    if count >= 1:
        ''' first GS all the vecs against V[:count, :]'''

        projections_coeff = einsum('ab,cb->ac', V[:count,:], vecs)  # (chunk_size, n_new_vectors)
        # vecs = einsum('ac,ab->cb', projections_coeff, V_chunk, -1 , 1, out=vecs)  # (n_new_vectors, A_size)

        tmp = einsum('ac,ab->cb', projections_coeff, V[:count,:])
        vecs -= tmp
        del tmp, projections_coeff

        if double:
            projections_coeff = einsum('ab,cb->ac', V[:count,:], vecs)  # (chunk_size, n_new_vectors)
            # vecs = einsum('ac,ab->cb', projections_coeff, V_chunk, -1 , 1, out=vecs)  # (n_new_vectors, A_size)
            tmp = einsum('ac,ab->cb', projections_coeff, V[:count,:])
            vecs -= tmp
            del tmp, projections_coeff
            gc.collect()

    ''' second GS vecs between themselves'''
    p0 = 0
    for i in range(n_new_vectors):
        vec = vecs[p0,:].reshape(1,-1)
        # print('vec.shape', vec.shape)
        norm = np.linalg.norm(vec)

        if norm > 1e-14:
            vec /= norm
            V[count,:] = vec

            count += 1
        else:
            p0 += 1
            continue

        if p0+1 == n_new_vectors:
            break

        other_vec = vecs[p0+1:,:]
        # print('other_vec.shape', other_vec.shape)

        projections_coeff = einsum('ab,cb->ac', vec, other_vec)
        # other_vec = einsum('ac,ab->cb',projections_coeff, vec, alpha=-1, beta=1, out=other_vec)
        tmp = einsum('ac,ab->cb', projections_coeff, vec)
        other_vec -= tmp
        del tmp, projections_coeff
        gc.collect()
        if double:
            projections_coeff = einsum('ab,cb->ac', vec, other_vec)
            # other_vec = einsum('ac,ab->cb',projections_coeff, vec, alpha=-1, beta=1, out=other_vec)
            tmp = einsum('ac,ab->cb', projections_coeff, vec)
            other_vec -= tmp
            del tmp, projections_coeff
            gc.collect()
        p0 += 1
        # return bvec
    new_count = count
    return new_count


def nKs_fill_holder(V, count, vecs, double=True):
    '''V is a vectors holder
       count is the amount of vectors that already sit in the holder
       nvec is amount of new vectors intended to fill in the V
       count will be final amount of vectors in V
    '''
    nvec = vecs.shape[0]
    for j in range(nvec):
        vec = vecs[j,:].reshape(1,-1)
        norm = np.linalg.norm(vec)
        if norm > 1e-14:
            vec = vec/norm
            V[count,:] = vec
            del vec
            gc.collect()
            count += 1

    new_count = count
    return new_count

# def S_symmetry_orthogonal(x,y):
#     '''symmetrically orthogonalize the vectors |x,y> and |y,x>
#        as close to original vectors as possible
#     '''
#     x_p_y = x + y
#     x_p_y_norm = np.linalg.norm(x_p_y)

#     x_m_y = x - y
#     x_m_y_norm = np.linalg.norm(x_m_y)

#     a = x_p_y_norm/x_m_y_norm

#     x_p_y = x_p_y/2
#     x_m_y = x_m_y * a/2

#     new_x = x_p_y + x_m_y
#     new_y = x_p_y - x_m_y

#     return new_x, new_y



def solve_AX_Xla_B(A, omega, Q):
    '''AX - XΩ  = Q
       A, Ω, Q are known, solve X
       Q is column-wise vectors

       Au = ua -> A = uau.T , u.Tu = uu.T = I,  u is column-wise vectors
       (u a u.T) X - XΩ  = Q
       u.T (u a u.T) X - u.T XΩ  = u.T Q
       a (u.T X) - (u.T X)Ω  = (u.T Q)
       au
    '''
    Qnorm = np.linalg.norm(Q, axis=0, keepdims=True)
    Q = Q/Qnorm
    N_vectors = len(omega)
    a, u = np.linalg.eigh(A)
    uq = np.dot(u.T, Q)
    ux = np.zeros_like(Q)
    for k in range(N_vectors):
        ux[:, k] = uq[:, k]/(a - omega[k])
    X = np.dot(u, ux)
    X *= Qnorm
    return X

def solve_AX_SX(A, S):
    '''                        AX = SXΩ
                               AX = (d^-1/2 S d^-1/2) d^1/2 XΩ
        d^-1/2 A (d^-1/2 d^1/2) X = L L.T d^1/2 X Ω
        d^-1/2 A d^-1/2 L^-1.T L.T d^1/2 X   = L L.T d^1/2 X Ω
        {L^-1 d^-1/2 A d^-1/2 L^-T} {L.T d^1/2 X}  = {L.T d^1/2 X} Ω
        M Z = Z Ω
        M = L^-1 d^-1/2 A d^-1/2 L^-T
        Z = L.T d^1/2 X
        X  = d^-1/2 L^-T Z
    '''
    A = np.asarray(A)
    S = np.asarray(S)

    d = np.diag(S)
    sqrt_d_inv = np.sqrt(1.0 / d)

    precond_S = sqrt_d_inv[:, None] * S * sqrt_d_inv[None, :]
    # np.fill_diagonal(precond_S, 1.0)

    L = np.linalg.cholesky(precond_S)
    L_inv = scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    L_invT = L_inv.T
    M = L_inv.dot( sqrt_d_inv[:, None] * A * sqrt_d_inv[None, :] ).dot( L_invT )

    omega, Z = np.linalg.eigh(M)
    X = sqrt_d_inv[:, None] * (L_invT.dot(Z))

    DEBUG = False
    if DEBUG:
        omega_scipy, x_scipy = scipy.linalg.eigh(A, S)
        x_scipy = np.array(x_scipy)
        assert np.linalg.norm(abs(X) - abs(x_scipy)) < 1e-10
    return omega, X

# def TDDFT_subspace_eigen_solver2(a, b, sigma, pi, nroots):
#     ''' [ a b ] x - [ σ   π] x  Ω = 0 '''
#     ''' [ b a ] y   [-π  -σ] y    = 0 '''
#     original_dtype = a.dtype
#     if original_dtype != np.float64:
#         a = a.astype(np.float64)
#         b = b.astype(np.float64)
#         sigma = sigma.astype(np.float64)
#         pi = pi.astype(np.float64)

#     d = abs(np.diag(sigma))
#     d_mh = d**(-0.5)

#     s_m_p = np.einsum('i,ij,j->ij', d_mh, sigma - pi, d_mh)

#     '''LU = d^−1/2 (σ − π) d^−1/2'''
#     ''' A = LU '''
#     L, U = scipy.linalg.lu(s_m_p, permute_l=True)
#     L_inv = np.linalg.inv(L)
#     U_inv = np.linalg.inv(U)


#     U_invT = U_inv.T
#     '''U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
#     d_amb_d = np.einsum('i,ij,j->ij', d_mh, a-b, d_mh)
#     GGT = np.dot(U_invT, np.dot(d_amb_d, U_inv))

#     G = np.linalg.cholesky(GGT)
#     if np.any(np.isnan(G)):
#         eig, eigv = np.linalg.eigh(GGT)
#         if eig[0] < -1e-4:
#             error_msg = (
#                 "GGT matrix is not positive definite.\n"
#                 "SCF not correctly converged is likely to cause this error.\n"
#                 "For example, scf converged to the wrong state.\n"
#             )
#             raise RuntimeError(error_msg)

#     G_inv = np.linalg.inv(G)

#     ''' M = G^T L^−1 d^−1/2 (a+b) d^−1/2 L^−T G '''
#     d_apb_d = np.einsum('i,ij,j->ij', d_mh, a+b, d_mh)
#     M = np.dot(G.T, np.dot(L_inv, np.dot(d_apb_d, np.dot(L_inv.T, G))))

#     omega2, Z = np.linalg.eigh(M)
#     if np.any(omega2 <= 0):
#         idx = np.nonzero(omega2 > 0)[0]
#         omega2 = omega2[idx[:nroots]]
#         Z = Z[:,idx[:nroots]]
#     else:
#         omega2 = omega2[:nroots]
#         Z = Z[:,:nroots]
#     omega = omega2**0.5

#     ''' It requires Z^T Z = 1/Ω '''
#     ''' x+y = d^−1/2 L^−T GZ Ω^-0.5 '''
#     ''' x−y = d^−1/2 U^−1 G^−T Z Ω^0.5 '''
#     x_p_y = np.einsum('i,ik,k->ik', d_mh, L_inv.T.dot(G.dot(Z)), omega**-0.5)
#     x_m_y = np.einsum('i,ik,k->ik', d_mh, U_inv.dot(G_inv.T.dot(Z)), omega**0.5)

#     x = (x_p_y + x_m_y)/2
#     y = x_p_y - x

#     if original_dtype != np.float64:
#         omega = omega.astype(original_dtype)
#         x = x.astype(original_dtype)
#         y = y.astype(original_dtype)
#     return omega, x, y

# def TDDFT_subspace_eigen_solver3(a, b, sigma, pi, k):
#     ''' [ a b ] x - [ σ   π] x  Ω = 0
#         [ b a ] y   [-π  -σ] y    = 0
#         AT=BTΩ
#         B^-1/2 A B^-1/2 B^1/2 T = B^1/2 T Ω
#         MZ = Z Ω
#         M = B^-1/2 A B^-1/2
#         Z = B^1/2 T
#     '''
#     half_size = a.shape[0]
#     A = np.empty((2*half_size,2*half_size))
#     A[:half_size,:half_size] = a[:,:]
#     A[:half_size,half_size:] = b[:,:]
#     A[half_size:,:half_size] = b[:,:]
#     A[half_size:,half_size:] = a[:,:]

#     B = np.empty_like(A)
#     B[:half_size,:half_size] = sigma[:,:]
#     B[:half_size,half_size:] = pi[:,:]
#     B[half_size:,:half_size] = -pi[:,:]
#     B[half_size:,half_size:] = -sigma[:,:]
#     #B^-1/2
#     B_neg_tmp = matrix_power(B, -0.5)
#     M = np.dot(B_neg_tmp, A)  # B^-1/2 A
#     M = np.dot(M, B_neg_tmp)  # B^-1/2 A B^-1/2
#     omega, Z = np.linalg.eigh(M)

#     omega = omega[half_size:k]
#     Z = Z[:, half_size:k]

#     T = np.dot(B_neg_tmp, Z)
#     x = T[:half_size,:]
#     y = T[half_size:,:]

#     return omega, x, y

# def TDDFT_subspace_eigen_solver(a, b, sigma, pi, k):
#     ''' [ a b ] x - [ σ   π] x  Ω = 0
#         [ b a ] y   [-π  -σ] y    = 0
#         AT=BTΩ
#         A^1/2 T = A^-1/2 B A^-1/2 A^1/2 T Ω
#         MZ = Z 1/Ω
#         M = A^-1/2 B A^-1/2 A^1/2
#         Z = A^1/2 T
#         Z is always returned as normlized vectors, which are not what we wanted
#         because Z^T Z = [x]^T A^1/2 A^1/2 [x] = [x]^T [ a b ] [x] =  [x]^T [ σ   π] x Ω = Ω
#                         [y]               [y]   [y]   [ b a ] [y]    [y]   [-π  -σ] y
#         therefore Z=Z*(Ω**0.5)
#         k: N_states
#     '''
#     half_size = a.shape[0]
#     A = np.empty((2*half_size,2*half_size))
#     A[:half_size,:half_size] = a[:,:]
#     A[:half_size,half_size:] = b[:,:]
#     A[half_size:,:half_size] = b[:,:]
#     A[half_size:,half_size:] = a[:,:]
#     B = np.empty_like(A)
#     B[:half_size,:half_size] = sigma[:,:]
#     B[:half_size,half_size:] = pi[:,:]
#     B[half_size:,:half_size] = -pi[:,:]
#     B[half_size:,half_size:] = -sigma[:,:]
#     #A^-1/2
#     A_neg_tmp = matrix_power(A, -0.5, 1e-14)
#     M = np.dot(A_neg_tmp, B)
#     M = np.dot(M,A_neg_tmp )
#     omega, Z = np.linalg.eigh(M)

#     omega = 1/omega[-k:][::-1]
#     Z = Z[:, -k:][:, ::-1]
#     Z = Z*(omega**0.5)

#     T = np.dot(A_neg_tmp, Z)
#     x = T[:half_size,:]
#     y = T[half_size:,:]

#     return omega, x, y



def TDDFT_subspace_linear_solver(a, b, sigma, pi, p, q, omega):
    '''[ a b ] x - [ σ   π] x  Ω = p
       [ b a ] y   [-π  -σ] y    = q
       normalize the right hand side first
    '''
    pq = np.vstack((p,q))
    pqnorm = np.linalg.norm(pq, axis=0, keepdims=True)

    p = p/pqnorm
    q = q/pqnorm

    d = abs(np.diag(sigma))
    d_mh = d**(-0.5)

    ''' TODO:replace LU decompose to cholesky '''
    '''LU = d^−1/2 (σ − π) d^−1/2 '''
    s_m_p = np.einsum('i,ij,j->ij', d_mh, sigma - pi, d_mh)
    L, U = scipy.linalg.lu(s_m_p, permute_l=True)
    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)
    # L_inv = scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    # U_inv = scipy.linalg.solve_triangular(U, np.eye(L.shape[0]), lower=True)
    U_invT = U_inv.T
    p_p_q_tilde = np.dot(L_inv, d_mh.reshape(-1,1)*(p+q))
    p_m_q_tilde = np.dot(U_invT, d_mh.reshape(-1,1)*(p-q))

    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T'''
    d_amb_d = np.einsum('i,ij,j->ij', d_mh, a-b, d_mh)
    GGT = np.dot(U_invT, np.dot(d_amb_d, U_inv))

    G = np.linalg.cholesky(GGT)
    if np.any(np.isnan(G)):
        eig, eigv = np.linalg.eigh(GGT)
        if eig[0] < -1e-4:
            error_msg = (
                "GGT matrix is not positive definite.\n"
                "SCF not correctly converged is likely to cause this error.\n"
                "For example, scf converged to the wrong state.\n"
            )
            raise RuntimeError(error_msg)
    G_inv = np.linalg.inv(G)

    '''a ̃+ b ̃= L^−1 d^−1/2 (a+b) d^−1/2 L^−T
       M = G^T (a ̃+ b ̃) G
    '''
    d_apb_d = np.einsum('i,ij,j->ij', d_mh, a+b, d_mh)
    a_p_b_tilde = np.dot(np.dot(L_inv, d_apb_d),  L_inv.T)

    M = np.dot(np.dot(G.T, a_p_b_tilde), G)

    T = np.dot(G.T, p_p_q_tilde)
    T += np.dot(G_inv, p_m_q_tilde * omega.reshape(1,-1))

    Z = solve_AX_Xla_B(M, omega**2, T)

    '''(x ̃+ y ̃) = GZ
       x + y = d^-1/2 L^-T (x ̃+ y ̃)
       x - y = d^-1/2 U^-1 (x ̃- y ̃)
    '''
    x_p_y_tilde = np.dot(G,Z)
    x_p_y = d_mh.reshape(-1,1) * np.dot(L_inv.T, x_p_y_tilde)

    x_m_y_tilde = (np.dot(a_p_b_tilde, x_p_y_tilde) - p_p_q_tilde)/omega
    x_m_y = d_mh.reshape(-1,1) * np.dot(U_inv, x_m_y_tilde)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x
    x *= pqnorm
    y *= pqnorm
    return x, y


def TDDFT_subspace_linear_solver1(a, b, sigma, pi, p, q, omega):
    ''' [ a b ] x - [ σ   π] x  Ω = p
        [ b a ] y   [-π  -σ] y    = q
        AT - BTΩ = P

        B^-1 AT - T Ω = B^-1 R
        MT - TΩ = P
        where
        M = B^-1 A
        P = B^-1 R
    '''
    half_size = a.shape[0]

    rhs = np.vstack((p, q))
    rhs_norm = np.linalg.norm(rhs, axis=0, keepdims=True)
    rhs = rhs/rhs_norm

    A = np.empty((2*half_size,2*half_size))
    A[:half_size,:half_size] = a[:,:]
    A[:half_size,half_size:] = b[:,:]
    A[half_size:,:half_size] = b[:,:]
    A[half_size:,half_size:] = a[:,:]

    B = np.empty_like(A)
    B[:half_size,:half_size] = sigma[:,:]
    B[:half_size,half_size:] = pi[:,:]
    B[half_size:,:half_size] = -pi[:,:]
    B[half_size:,half_size:] = -sigma[:,:]

    B_inv = np.linalg.inv(B)
    M = np.dot(B_inv, A)
    R = np.dot(B_inv,rhs)

    T = scipy.linalg.solve_sylvester(M, -np.diag(omega), R)
    T *= rhs_norm

    x = T[:half_size,:]
    y = T[half_size:,:]

    return x, y


def TDDFT_subspace_eigen_solver4(a_p_b, a_m_b, sigma_p_pi, nroots):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 '''
    ''' [ b a ] y   [-π  -σ] y    = 0

    a, b, sigma, pi, nroots

    no d_mh version of TDDFT_subspace_eigen_solver5

    '''

    # convert to float64 to avoid precision issues, very useful

    original_dtype = a_p_b.dtype
    if original_dtype != np.float64:
        a_p_b = a_p_b.astype(np.float64)
        a_m_b = a_m_b.astype(np.float64)
        sigma_p_pi = sigma_p_pi.astype(np.float64)
    sigma_m_pi = sigma_p_pi.T

    # s_m_p = np.einsum('i,ij,j->ij', d_mh, sigma_m_pi, d_mh)
    s_m_p = sigma_m_pi
    '''LU = (σ − π) '''
    ''' A = LU '''
    L, U = scipy.linalg.lu(s_m_p, permute_l=True)
    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    '''U^-T(a−b) U^-1 = GG^T '''
    GGT = np.dot(U_inv.T, np.dot(a_m_b, U_inv))

    G = np.linalg.cholesky(GGT)
    if np.any(np.isnan(G)):
        eig, eigv = np.linalg.eigh(GGT)
        if eig[0] < -1e-4:
            error_msg = (
                "GGT matrix is not positive definite.\n"
                "SCF not correctly converged is likely to cause this error.\n"
                "For example, scf converged to the wrong state.\n"
            )
            raise RuntimeError(error_msg)

    G_inv = np.linalg.inv(G)

    ''' M = G^T L^−1 (a+b) L^−T G '''
    M = np.dot(G.T, np.dot(L_inv, np.dot(a_p_b, np.dot(L_inv.T, G))))

    omega2, Z = np.linalg.eigh(M)
    if np.any(omega2 <= 0):
        error_msg = (
            "omega**2 is not positive.\n"
            "SCF not correctly converged is likely to cause this error.\n"
            f"Or the precision {original_dtype} is not enough."
        )
        raise RuntimeError(error_msg)
    else:
        omega2 = omega2[:nroots]
        Z = Z[:,:nroots]
    omega = omega2**0.5

    ''' It requires Z^T Z = 1/Ω '''
    ''' x+y = L^−T GZ Ω^-0.5 '''
    ''' x−y = U^−1 G^−T Z Ω^0.5 '''
    x_p_y = L_inv.T.dot(G.dot(Z)) * omega**-0.5
    x_m_y = U_inv.dot(G_inv.T.dot(Z)) * omega**0.5

    if original_dtype != np.float64:
        omega = omega.astype(original_dtype)
        x_p_y = x_p_y.astype(original_dtype)
        x_m_y = x_m_y.astype(original_dtype)

    # print('x_p_yT.shape', x_p_y.T.shape)
    # print('x_m_yT.shape', x_m_y.T.shape)
    if np.any(np.isnan(x_p_y)) or np.any(np.isnan(x_m_y)):
        # print('x_p_y', x_p_y)
        # print('x_m_y', x_m_y)
        raise ValueError('x_p_y or x_m_y is nan')
    if np.any(np.isinf(x_p_y)) or np.any(np.isinf(x_m_y)):
        # print('x_p_y', x_p_y)
        # print('x_m_y', x_m_y)
        raise ValueError('x_p_y or x_m_y is inf')
    return omega, x_p_y, x_m_y



def XmY_2_XY(Z, AmB_sq, omega):
    '''given Z, (A-B)^2, omega
       return X, Y

        X-Y = (A-B)^-1/2 Z
        X+Y = (A-B)^1/2 Z omega^-1
    '''
    AmB_sq = AmB_sq.reshape(1,-1)

    '''AmB = (A - B)'''
    AmB = AmB_sq**0.5

    XmY = AmB**(-0.5) * Z

    omega = omega.reshape(-1,1)
    XpY = (AmB * XmY)/omega

    X = (XpY + XmY)/2
    Y = (XpY - XmY)/2

    return X, Y


class LinearDependencyError(RuntimeError):
    pass

