#!/usr/bin/env python
#
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
# Author: Zehao Zhou
# Edited by: Qiming Sun <osirpt.sun@gmail.com>

'''
TDDFT-ris

This implementation is a simplified version of https://github.com/John-zzh/pyscf_TDDFT_ris

References:
[1] Minimal Auxiliary Basis Set Approach for the Electronic Excitation Spectra
    of Organic Molecules
    Zehao Zhou, Fabio Della Sala, and Shane M. Parker
    J. Phys. Chem. Lett. 2023, 14, 1968-1976

[2] Converging Time-Dependent Density Functional Theory Calculations in Five
    Iterations with Minimal Auxiliary Preconditioning
    Zehao Zhou and Shane M. Parker
    J. Chem. Theory Comput. 2024, 20, 6738-6746
'''

__all__ = [
    'RKS_TDA',
    'RKS_TDDFT',
    'UKS_TDA',
    'UKS_TDDFT',
]

from functools import lru_cache
import numpy as np
from pyscf.lib import logger, einsum
from pyscf.data.elements import _std_symbol_without_ghost
from pyscf.dft.rks import KohnShamDFT
from pyscf.dft.numint import NumInt
from pyscf.df.incore import aux_e2
from pyscf.tdscf import rhf, uhf, rks, uks
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig

'''
GB Radii
Ghosh, Dulal C and coworkers
The wave mechanical evaluation of the absolute radii of atoms.
Journal of Molecular Structure: THEOCHEM 865, no. 1-3 (2008): 60-67.
'''

elements_106 = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

radii = [0.5292, 0.3113, 1.6283, 1.0855, 0.8141, 0.6513, 0.5428, 0.4652, 0.4071, 0.3618,
2.165, 1.6711, 1.3608, 1.1477, 0.9922, 0.8739, 0.7808, 0.7056, 3.293, 2.5419,
2.4149, 2.2998, 2.1953, 2.1, 2.0124, 1.9319, 1.8575, 1.7888, 1.725, 1.6654,
1.4489, 1.2823, 1.145, 1.0424, 0.9532, 0.8782, 3.8487, 2.9709, 2.8224, 2.688,
2.5658, 2.4543, 2.352, 2.2579, 2.1711, 2.0907, 2.016, 1.9465, 1.6934, 1.4986,
1.344, 1.2183, 1.1141, 1.0263, 4.2433, 3.2753, 2.6673, 2.2494, 1.9447, 1.7129,
1.5303, 1.383, 1.2615, 1.1596, 1.073, 0.9984, 0.9335, 0.8765, 0.8261, 0.7812,
0.7409, 0.7056, 0.6716, 0.6416, 0.6141, 0.589, 0.5657, 0.5443, 0.5244, 0.506,
1.867, 1.6523, 1.4818, 1.3431, 1.2283, 1.1315, 4.4479, 3.4332, 3.2615, 3.1061,
2.2756, 1.9767, 1.7473, 1.4496, 1.2915, 1.296, 1.1247, 1.0465, 0.9785, 0.9188,
0.8659, 0.8188, 0.8086]
exp = [1/(i*1.8897259885789)**2 for i in radii]

ris_exp = dict(zip(elements_106,exp))

'''
range-separated hybrid functionals, (omega, alpha, beta)
'''
rsh_func = {}
rsh_func['wb97'] = (0.4, 0, 1.0)
rsh_func['wb97x'] = (0.3, 0.157706, 0.842294)  # wb97 family, a+b=100% Long-range HF exchange
rsh_func['wb97x-d'] = (0.2, 0.22, 0.78)
rsh_func['wb97x-d3'] = (0.25, 0.195728, 0.804272)
rsh_func['wb97x-v'] = (0.30, 0.167, 0.833)
rsh_func['wb97x-d3bj'] = (0.30, 0.167, 0.833)
rsh_func['cam-b3lyp'] = (0.33, 0.19, 0.46) # a+b=65% Long-range HF exchange
rsh_func['lc-blyp'] = (0.33, 0, 1.0)
rsh_func['lc-PBE'] = (0.47, 0, 1.0)

'''
hybrid functionals, hybrid component a_x
'''
hbd_func = {}
hbd_func['pbe'] = 0
hbd_func['pbe,pbe'] = 0
hbd_func['tpss'] = 0
hbd_func['tpssh'] = 0.1
hbd_func['b3lyp'] = 0.2
hbd_func['pbe0'] = 0.25
hbd_func['bhh-lyp'] = 0.5
hbd_func['m05-2x'] = 0.56
hbd_func['m06'] = 0.27
hbd_func['m06-2x'] = 0.54
hbd_func[None] = 1

def gen_auxmol(mol, theta=0.2, add_p=False):
    '''
    auxmol_basis_keys: (['C1', 'H2', 'O3', 'H4', 'H5', 'H6'])

    aux_basis:
    C1 [[0, [0.1320292535005648, 1.0]]]
    H2 [[0, [0.1999828038466018, 1.0]]]
    O3 [[0, [0.2587932305664396, 1.0]]]
    H4 [[0, [0.1999828038466018, 1.0]]]
    H5 [[0, [0.1999828038466018, 1.0]]]
    H6 [[0, [0.1999828038466018, 1.0]]]
    '''
    aux_basis = {}
    for atom_index in mol._basis:
        atom = _std_symbol_without_ghost(atom_index)
        '''
        exponent alpha = 1/R^2 * theta
        '''
        exp = ris_exp[atom] * theta
        if atom != 'H' and add_p:
            aux_basis[atom_index] = [[0, [exp, 1.0]],[1, [exp, 1.0]]]
        else:
            aux_basis[atom_index] = [[0, [exp, 1.0]]]

    auxmol = mol.copy()
    auxmol.basis = aux_basis
    auxmol.build(False, False)
    return auxmol

@lru_cache(100)
def _get_rsh_parameters(xc):
    if xc in rsh_func: # predefined functionals
        '''
        RSH functional, need omega, alpha, beta
        '''
        omega, alpha, beta = rsh_func[xc]
        hyb = alpha

    elif xc in hbd_func:
        hyb = hbd_func[xc]
        omega, alpha, beta = None, hyb, 0

    else: # internal libxc database
        ni = NumInt()
        hybrid = ni.libxc.is_hybrid_xc(xc)
        if hybrid:
            omega, alpha, beta = ni.rsh_coeff(xc)
            if omega:
                hyb = alpha + beta
            else:
                hyb = ni.hybrid_coeff(xc)
            alpha = hyb
        else:
            hyb, omega, alpha, beta = 0, None, 0, 0
    return hyb, omega, alpha, beta


class TDDFT_ris_Base:
    '''
    TDDFT-ris

    References:
    [1] Minimal Auxiliary Basis Set Approach for the Electronic Excitation Spectra
        of Organic Molecules
        Zehao Zhou, Fabio Della Sala, and Shane M. Parker
        J. Phys. Chem. Lett. 2023, 14, 1968-1976

    [2] Converging Time-Dependent Density Functional Theory Calculations in Five
        Iterations with Minimal Auxiliary Preconditioning
        Zehao Zhou and Shane M. Parker
        J. Chem. Theory Comput. 2024, 20, 6738-6746
    '''

    # to generate exponent alpha, theta/R^2
    theta = 0.2
    # add_p: whether add p orbital to aux basis
    add_p = False
    # XC functional, can be different to the ground state KS model
    xc = None
    # The norm(residual) tolerance in diagonalization
    conv_tol = 1e-5
    # Number of states to solve
    nroots = 5
    # Maximum number of diagonalization iteration
    max_iter = 25
    # Whether to execute single precision computation
    single = False

    def __init__(self, mf):
        self._scf = mf
        self.mol = mf.mol
        self.stdout = mf.stdout
        self.verbose = mf.verbose
        self.auxmol = None
        self.a_x = self.omega = self.alpha = self.beta = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('nroots = %s', self.nroots)
        log.info('theta = %s', self.theta)
        log.info('add_p = %s', self.add_p)
        log.info('xc = %s', self.xc)
        log.info('conv_tol = %s', self.conv_tol)
        log.info('nroots = %s', self.nroots)
        log.info('max_iter = %s', self.max_iter)
        log.info('single precision = %s', self.single)
        if self.omega is not None:
            log.info("range-separated hybrid XC functional")
            log.info(f"ω = {self.omega}, screening factor")
            log.info(f"α = {self.alpha}, fixed HF exchange contribution")
            log.info(f"β = {self.beta}, variable part")
        elif self.a_x is not None:
            log.info(f"a_x = {self.a_x}")
        return self

    def build(self):
        self.auxmol = gen_auxmol(self.mol, self.theta, self.add_p)
        if (self.a_x, self.omega, self.alpha, self.beta) == (None, None, None, None):
            xc = self._get_xc()
            self.a_x, self.omega, self.alpha, self.beta = _get_rsh_parameters(xc)
        self.dump_flags()
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.auxmol = None
        self.a_x = self.omega = self.alpha = self.beta = None
        return self

    def _get_xc(self):
        if self.xc is None:
            xc = getattr(self._scf, 'xc', 'HF').lower()
        else:
            xc = self.xc.lower()
        return xc

    def gen_eri2c_eri3c(self, mol, auxmol, omega=None):
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            eri2c = auxmol.intor('int2c2e')
            eri3c = aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
        if self.single:
            eri2c = eri2c.astype(np.float32)
            eri3c = eri3c.astype(np.float32)
        return eri2c, eri3c

    def gen_uvQL(self, eri3c, eri2c):
        '''
        (P|Q)^-1 = LL^T
        uvQL = Σ_Q (uv|Q)L_Q
        '''
        Lower = np.linalg.cholesky(np.linalg.inv(eri2c))
        uvQL = einsum("uvQ,QP->uvP", eri3c, Lower)
        return uvQL

    def gen_B(self, uvQL, n_occ, mo_coeff, calc=None):
        ''' B_pq^P = C_u^p C_v^q Σ_Q (uv|Q)L_Q '''
        # B = einsum("up,vq,uvP->pqP", mo_coeff, mo_coeff, uvQL)
        if self.single:
            mo_coeff = mo_coeff.astype(np.float32)
        tmp = einsum("vq,uvP->uqP", mo_coeff, uvQL)
        B = einsum("up,uqP->pqP", mo_coeff, tmp)


        '''
                     n_occ          n_vir
                -|-------------||-------------|
                 |             ||             |
           n_occ |   B_ij      ||    B_ia     |
                 |             ||             |
                 |             ||             |
                =|=============||=============|
                 |             ||             |
           n_vir |             ||    B_ab     |
                 |             ||             |
                 |             ||             |
                -|-------------||-------------|
        '''

        B_ia = B[:n_occ,n_occ:,:].copy(order='C')

        if calc == 'both' or calc == 'exchange_only':
            '''
            For common bybrid DFT, exchange and coulomb term use same set of B matrix
            For range-seperated bybrid DFT, (ij|ab) and (ib|ja) use different
            set of B matrix than (ia|jb), because of the RSH eri2c and eri3c
            B_ia_ex is for (ib|ja)
            '''
            B_ij = B[:n_occ,:n_occ,:].copy(order='C')
            B_ab = B[n_occ:,n_occ:,:].copy(order='C')
            return B_ia, B_ij, B_ab

        elif calc == 'coulomb_only':
            '''(ia|jb) coulomb term'''
            return B_ia

    def gen_B_cl_B_ex(self, mol, auxmol, uvQL, n_occ, mo_coeff, eri3c=None, eri2c=None):

        '''
        the 2c2e and 3c2e integrals with/without RSH
        (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab)  + (ij|alpha + beta*erf(omega)/r|ab)
        short-range part (ij|1-(alpha + beta*erf(omega))/r|ab) is treated by the
        DFT XC functional, thus not considered here
        long-range part  (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
        '''
        log = logger.new_logger(self)

        if self.a_x != 0 and self.omega is None:
            '''
            for usual hybrid functional, the Coulomb and Exchange ERI
            share the same eri2c and eri3c,
            '''

            B_ia, B_ij, B_ab = self.gen_B(uvQL=uvQL,
                                          n_occ=n_occ,
                                          mo_coeff=mo_coeff,
                                          calc='both')
            B_ia_cl = B_ia
            B_ia_ex, B_ij_ex, B_ab_ex = B_ia, B_ij, B_ab

        elif self.omega:
            '''
            in the RSH functional, the Exchange ERI splits into two parts
            (ij|ab) = (ij|1-(alpha + beta*erf(omega))/r|ab) + (ij|alpha + beta*erf(omega)/r|ab)
            -- The first part (ij|1-(alpha + beta*erf(omega))/r|ab) is short range,
                treated by the DFT XC functional, thus not considered here
            -- The second part is long range
                (ij|alpha + beta*erf(omega)/r|ab) = alpha (ij|r|ab) + beta*(ij|erf(omega)/r|ab)
            '''
            log.debug('2c2e and 3c2e for RSH RI-K (ij|ab)')
            eri2c_erf, eri3c_erf = self.gen_eri2c_eri3c(mol=mol, auxmol=auxmol, omega=self.omega)
            eri2c_ex = self.alpha*eri2c + self.beta*eri2c_erf
            eri3c_ex = self.alpha*eri3c + self.beta*eri3c_erf

            B_ia_cl = self.gen_B(uvQL=uvQL,
                                 n_occ=n_occ,
                                 mo_coeff=mo_coeff,
                                 calc='coulomb_only')
            uvQL_ex = self.gen_uvQL(eri2c=eri2c_ex, eri3c=eri3c_ex)
            B_ia_ex, B_ij_ex, B_ab_ex = self.gen_B(uvQL=uvQL_ex,
                                                   n_occ=n_occ,
                                                   mo_coeff=mo_coeff,
                                                   calc='exchange_only')
        log.debug('type(B_ia_cl)',B_ia_cl.dtype)
        return B_ia_cl, B_ia_ex, B_ij_ex, B_ab_ex

    def gen_hdiag_fly(self, mo_energy, n_occ, n_vir, sqrt=False):

        '''KS orbital energy difference, ε_a - ε_i
        '''
        vir = mo_energy[n_occ:].reshape(1,n_vir)
        occ = mo_energy[:n_occ].reshape(n_occ,1)
        delta_hdiag = np.repeat(vir, n_occ, axis=0) - np.repeat(occ, n_vir, axis=1)

        hdiag = delta_hdiag.reshape(n_occ*n_vir)
        if sqrt == False:
            '''standard diag(A)V
               preconditioner = diag(A)
            '''
            def hdiag_fly(V):
                delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag, V)
                return delta_hdiag_v
            return hdiag_fly, hdiag

        elif sqrt == True:
            '''diag(A)**0.5 V
               preconditioner = diag(A)**2
            '''
            delta_hdiag_sqrt = np.sqrt(delta_hdiag)
            hdiag_sq = hdiag**2
            def hdiag_sqrt_fly(V):
                delta_hdiag_v = einsum("ia,iam->iam", delta_hdiag_sqrt, V)
                return delta_hdiag_v
            return hdiag_sqrt_fly, hdiag_sq

    def gen_iajb_fly(self, B_left, B_right):
        def iajb_fly(V):
            '''
            (ia|jb) = Σ_Pjb (B_left_ia^P B_right_jb^P V_jb^m)
                    = Σ_P [ B_left_ia^P Σ_jb(B_right_jb^P V_jb^m) ]
            if B_left == B_right, then it is either
                (1) (ia|jb) in RKS
                or
                (2)(ia_α|jb_α) or (ia_β|jb_β) in UKS,
            else,
                it is (ia_α|jb_β) or (ia_β|jb_α) in UKS
            '''
            B_right_jb_V = einsum("jbP,jbm->Pm", B_right, V)
            iajb_V = einsum("iaP,Pm->iam", B_left, B_right_jb_V)
            # print('iajb_V.dtype', iajb_V.dtype)
            return iajb_V
        return iajb_fly

    def gen_ijab_fly(self, B_ij, B_ab):
        def ijab_fly(V):
            '''
            (ij|ab) = Σ_Pjb (B_ij^P B_ab^P V_jb^m)
                    = Σ_P [B_ij^P Σ_jb(B_ab^P V_jb^m)]
            '''
            B_ab_V = einsum("abP,jbm->jPam", B_ab, V)
            ijab_V = einsum("ijP,jPam->iam", B_ij, B_ab_V)
            # print('ijab_V.dtype', ijab_V.dtype)
            return ijab_V
        return ijab_fly

    def gen_ibja_fly(self, B_ia):
        def ibja_fly(V):
            '''
            the exchange (ib|ja) in B matrix
            (ib|ja) = Σ_Pjb (B_ib^P B_ja^P V_jb^m)
                    = Σ_P [B_ja^P Σ_jb(B_ib^P V_jb^m)]
            '''
            B_ib_V = einsum("ibP,jbm->Pijm", B_ia, V)
            ibja_V = einsum("jaP,Pijm->iam", B_ia, B_ib_V)
            # print('ibja_V.dtype',ibja_V.dtype)
            return ibja_V
        return ibja_fly

class RKS_TDA:
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf

        mol = self.mol
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        orbo = mo_coeff[:,mo_occ==2]
        orbv = mo_coeff[:,mo_occ==0]
        n_occ = orbo.shape[1]
        n_vir = orbv.shape[1]
        A_size = n_occ * n_vir
        mo_coeff = np.hstack([orbo, orbv])

        auxmol = self.auxmol
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=mol, auxmol=auxmol)
        uvQL = self.gen_uvQL(eri2c, eri3c)

        hdiag_fly, hdiag = self.gen_hdiag_fly(mo_energy=mo_energy,
                                              n_occ=n_occ,
                                              n_vir=n_vir)
        a_x = self.a_x
        if a_x != 0:
            '''hybrid RKS TDA'''
            B_ia_cl, _, B_ij_ex, B_ab_ex = self.gen_B_cl_B_ex(mol=mol,
                                                            auxmol=auxmol,
                                                            uvQL=uvQL,
                                                            eri3c=eri3c,
                                                            eri2c=eri2c,
                                                            n_occ=n_occ,
                                                            mo_coeff=mo_coeff)

            iajb_fly = self.gen_iajb_fly(B_left=B_ia_cl, B_right=B_ia_cl)
            ijab_fly = self.gen_ijab_fly(B_ij=B_ij_ex, B_ab=B_ab_ex)

            def RKS_TDA_hybrid_mv(X):
                ''' hybrid or range-sparated hybrid, a_x > 0
                    return AX
                    AV = hdiag_fly(V) + 2*iajb_fly(V) - a_x*ijab_fly(V)
                    for RSH, a_x = 1
                '''
                # print('a_x=', a_x)
                X = X.reshape(n_occ, n_vir, -1)
                AX = hdiag_fly(X) + 2*iajb_fly(X) - a_x*ijab_fly(X)
                AX = AX.reshape(A_size, -1)
                return AX
            return RKS_TDA_hybrid_mv, hdiag

        elif a_x == 0:
            '''pure RKS TDA'''
            B_ia = self.gen_B(uvQL=uvQL,
                                n_occ=n_occ,
                                mo_coeff=mo_coeff,
                                calc='coulomb_only')
            iajb_fly = self.gen_iajb_fly(B_left=B_ia, B_right=B_ia)
            def RKS_TDA_pure_mv(X):
                ''' pure functional, a_x = 0
                    return AX
                    AV = hdiag_fly(V) + 2*iajb_fly(V)
                    for RSH, a_x = 1
                '''
                # print('a_x=', a_x)
                X = X.reshape(n_occ, n_vir, -1)
                AX = hdiag_fly(X) + 2*iajb_fly(X)
                AX = AX.reshape(A_size, -1)
                return AX

        return RKS_TDA_pure_mv, hdiag

    kernel = rks.TDA.kernel

class RKS_TDDFT:
    def gen_RKS_TDDFT_mv(self, mf=None):
        if mf is None:
            mf = self._scf

        mol = self.mol
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        orbo = mo_coeff[:,mo_occ==2]
        orbv = mo_coeff[:,mo_occ==0]
        n_occ = orbo.shape[1]
        n_vir = orbv.shape[1]
        A_size = n_occ * n_vir
        mo_coeff = np.hstack([orbo, orbv])

        auxmol = self.auxmol
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=mol, auxmol=auxmol)
        uvQL = self.gen_uvQL(eri2c, eri3c)

        '''hdiag_fly will be used in both RKS and UKS'''
        hdiag_fly, hdiag = self.gen_hdiag_fly(mo_energy=mo_energy,
                                                n_occ=n_occ,
                                                n_vir=n_vir)
        a_x = self.a_x
        if a_x != 0:
            '''hybrid or range-separated RKS TDDFT'''

            B_ia_cl, B_ia_ex, B_ij_ex, B_ab_ex = self.gen_B_cl_B_ex(mol=mol,
                                                                    auxmol=auxmol,
                                                                    uvQL=uvQL,
                                                                    eri3c=eri3c,
                                                                    eri2c=eri2c,
                                                                    n_occ=n_occ,
                                                                    mo_coeff=mo_coeff)
            iajb_fly = self.gen_iajb_fly(B_left=B_ia_cl, B_right=B_ia_cl)
            ijab_fly = self.gen_ijab_fly(B_ij=B_ij_ex, B_ab=B_ab_ex)
            ibja_fly = self.gen_ibja_fly(B_ia=B_ia_ex)

            def RKS_TDDFT_hybrid_mv(X, Y):
                '''
                RKS
                [A B][X] = [AX+BY] = [U1]
                [B A][Y]   [AY+BX]   [U2]
                we want AX+BY and AY+BX
                instead of directly computing AX+BY and AY+BX
                we compute (A+B)(X+Y) and (A-B)(X-Y)
                it can save one (ia|jb)V tensor contraction compared to directly computing AX+BY and AY+BX

                (A+B)V = hdiag_fly(V) + 4*iajb_fly(V) - a_x * [ ijab_fly(V) + ibja_fly(V) ]
                (A-B)V = hdiag_fly(V) - a_x * [ ijab_fly(V) - ibja_fly(V) ]
                for RSH, a_x = 1, because the exchange component is defined by alpha+beta
                '''
                X = X.reshape(n_occ, n_vir, -1)
                Y = Y.reshape(n_occ, n_vir, -1)

                XpY = X + Y
                XmY = X - Y

                ApB_XpY = hdiag_fly(XpY) + 4*iajb_fly(XpY) - a_x*(ijab_fly(XpY) + ibja_fly(XpY))

                AmB_XmY = hdiag_fly(XmY) - a_x*(ijab_fly(XmY) - ibja_fly(XmY) )

                ''' (A+B)(X+Y) = AX + BY + AY + BX   (1)
                    (A-B)(X-Y) = AX + BY - AY - BX   (2)
                    (1) + (1) /2 = AX + BY = U1
                    (1) - (2) /2 = AY + BX = U2
                '''
                U1 = (ApB_XpY + AmB_XmY)/2
                U2 = (ApB_XpY - AmB_XmY)/2

                U1 = U1.reshape(A_size,-1)
                U2 = U2.reshape(A_size,-1)

                return U1, U2
            return RKS_TDDFT_hybrid_mv, hdiag

        elif a_x == 0:
            '''pure RKS TDDFT'''
            hdiag_sqrt_fly, hdiag_sq = self.gen_hdiag_fly(mo_energy=mo_energy,
                                                          n_occ=n_occ,
                                                          n_vir=n_vir,
                                                          sqrt=True)
            B_ia = self.gen_B(uvQL=uvQL,
                            n_occ=n_occ,
                            mo_coeff=mo_coeff,
                            calc='coulomb_only')
            iajb_fly = self.gen_iajb_fly(B_left=B_ia, B_right=B_ia)

            def RKS_TDDFT_pure_mv(Z):
                '''(A-B)^1/2(A+B)(A-B)^1/2 Z = Z w^2
                                        MZ = Z w^2
                    X+Y = (A-B)^1/2 Z
                    A+B = hdiag_fly(V) + 4*iajb_fly(V)
                    (A-B)^1/2 = hdiag_sqrt_fly(V)
                '''
                Z = Z.reshape(n_occ, n_vir, -1)
                AmB_sqrt_V = hdiag_sqrt_fly(Z)
                ApB_AmB_sqrt_V = hdiag_fly(AmB_sqrt_V) + 4*iajb_fly(AmB_sqrt_V)
                MZ = hdiag_sqrt_fly(ApB_AmB_sqrt_V)
                MZ = MZ.reshape(A_size, -1)
                return MZ

            return RKS_TDDFT_pure_mv, hdiag_sq


    kernel = rks.TDDFT.kernel

class UKS_TDA:
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf

        mol = self.mol
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        occidxa = mo_occ[0]>0
        occidxb = mo_occ[1]>0
        viridxa = mo_occ[0]==0
        viridxb = mo_occ[1]==0
        n_occ_a = len(occidxa)
        n_occ_b = len(occidxb)
        n_vir_a = len(viridxa)
        n_vir_b = len(viridxb)
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbvb = mo_coeff[1][:,viridxb]
        mo_coeff = (np.hstack([orboa, orbva]),
                    np.hstack([orbob, orbvb]))

        auxmol = self.auxmol
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=mol, auxmol=auxmol)
        uvQL = self.gen_uvQL(eri2c, eri3c)

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        hdiag_a_fly, hdiag_a = self.gen_hdiag_fly(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_fly, hdiag_b = self.gen_hdiag_fly(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = np.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        a_x = self.a_x
        if a_x != 0:
            ''' UKS TDA hybrid '''
            B_ia_cl_alpha, _, B_ij_ex_alpha, B_ab_ex_alpha = self.gen_B_cl_B_ex(mol=mol,
                                                                                auxmol=auxmol,
                                                                                uvQL=uvQL,
                                                                                eri3c=eri3c,
                                                                                eri2c=eri2c,
                                                                                n_occ=n_occ_a,
                                                                                mo_coeff=mo_coeff[0])

            B_ia_cl_beta, _, B_ij_ex_beta, B_ab_ex_beta  = self.gen_B_cl_B_ex(mol=mol,
                                                                              auxmol=auxmol,
                                                                              uvQL=uvQL,
                                                                              eri3c=eri3c,
                                                                              eri2c=eri2c,
                                                                              n_occ=n_occ_b,
                                                                              mo_coeff=mo_coeff[1])

            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_beta)

            ijab_aa_fly = self.gen_ijab_fly(B_ij=B_ij_ex_alpha, B_ab=B_ab_ex_alpha)
            ijab_bb_fly = self.gen_ijab_fly(B_ij=B_ij_ex_beta,  B_ab=B_ab_ex_beta)

            def UKS_TDA_hybrid_mv(X):
                '''
                UKS
                return AX
                A have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]
                    [ Aβα Aββ ]

                X = [ Xα ]
                    [ Xβ ]
                AX = [ Aαα Xα + Aαβ Xβ ]
                     [ Aβα Xα + Aββ Xβ ]

                Aαα Xα = hdiag_fly(Xα) + iajb_aa_fly(Xα) - a_x * ijab_aa_fly(Xα)
                Aββ Xβ = hdiag_fly(Xβ) + iajb_bb_fly(Xβ) - a_x * ijab_bb_fly(Xβ)
                Aαβ Xβ = iajb_ab_fly(Xβ)
                Aβα Xα = iajb_ba_fly(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_fly(X_a) + iajb_aa_fly(X_a) - a_x * ijab_aa_fly(X_a)
                Aab_Xb = iajb_ab_fly(X_b)

                Aba_Xa = iajb_ba_fly(X_a)
                Abb_Xb = hdiag_b_fly(X_b) + iajb_bb_fly(X_b) - a_x * ijab_bb_fly(X_b)

                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = np.vstack((U_a, U_b))
                return U
            return UKS_TDA_hybrid_mv, hdiag

        elif a_x == 0:
            ''' UKS TDA pure '''
            B_ia_alpha = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_a,
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            B_ia_beta = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_b,
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')

            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_beta)

            def UKS_TDA_pure_mv(X):
                '''
                Aαα Xα = hdiag_fly(Xα) + iajb_aa_fly(Xα)
                Aββ Xβ = hdiag_fly(Xβ) + iajb_bb_fly(Xβ)
                Aαβ Xβ = iajb_ab_fly(Xβ)
                Aβα Xα = iajb_ba_fly(Xα)
                '''
                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                Aaa_Xa = hdiag_a_fly(X_a) + iajb_aa_fly(X_a)
                Aab_Xb = iajb_ab_fly(X_b)

                Aba_Xa = iajb_ba_fly(X_a)
                Abb_Xb = hdiag_b_fly(X_b) + iajb_bb_fly(X_b)

                U_a = (Aaa_Xa + Aab_Xb).reshape(A_aa_size,-1)
                U_b = (Aba_Xa + Abb_Xb).reshape(A_bb_size,-1)

                U = np.vstack((U_a, U_b))
                return U
            return UKS_TDA_pure_mv, hdiag

    kernel = uks.TDA.kernel

class UKS_TDDFT:
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf

        mol = self.mol
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        occidxa = mo_occ[0]>0
        occidxb = mo_occ[1]>0
        viridxa = mo_occ[0]==0
        viridxb = mo_occ[1]==0
        n_occ_a = len(occidxa)
        n_occ_b = len(occidxb)
        n_vir_a = len(viridxa)
        n_vir_b = len(viridxb)
        orboa = mo_coeff[0][:,occidxa]
        orbob = mo_coeff[1][:,occidxb]
        orbva = mo_coeff[0][:,viridxa]
        orbvb = mo_coeff[1][:,viridxb]
        mo_coeff = (np.hstack([orboa, orbva]),
                    np.hstack([orbob, orbvb]))

        auxmol = self.auxmol
        eri2c, eri3c = self.gen_eri2c_eri3c(mol=mol, auxmol=auxmol)
        uvQL = self.gen_uvQL(eri2c, eri3c)

        A_aa_size = n_occ_a * n_vir_a
        A_bb_size = n_occ_b * n_vir_b

        '''
        _aa_fly means alpha-alpha spin
        _ab_fly means alpha-beta spin
        B_ia_alpha means B_ia matrix for alpha spin
        B_ia_beta means B_ia matrix for beta spin
        '''

        hdiag_a_fly, hdiag_a = self.gen_hdiag_fly(mo_energy=mo_energy[0], n_occ=n_occ_a, n_vir=n_vir_a)
        hdiag_b_fly, hdiag_b = self.gen_hdiag_fly(mo_energy=mo_energy[1], n_occ=n_occ_b, n_vir=n_vir_b)
        hdiag = np.vstack((hdiag_a.reshape(-1,1), hdiag_b.reshape(-1,1))).reshape(-1)

        a_x = self.a_x
        if a_x != 0:
            B_ia_cl_alpha, B_ia_ex_alpha, B_ij_ex_alpha, B_ab_ex_alpha = self.gen_B_cl_B_ex(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvQL=uvQL,
                                                                                            eri3c=eri3c,
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_a,
                                                                                            mo_coeff=mo_coeff[0])

            B_ia_cl_beta,  B_ia_ex_beta,  B_ij_ex_beta,  B_ab_ex_beta  = self.gen_B_cl_B_ex(mol=mol,
                                                                                            auxmol=auxmol,
                                                                                            uvQL=uvQL,
                                                                                            eri3c=eri3c,
                                                                                            eri2c=eri2c,
                                                                                            n_occ=n_occ_b,
                                                                                            mo_coeff=mo_coeff[1])

            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_cl_alpha, B_right=B_ia_cl_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_cl_beta,  B_right=B_ia_cl_beta)

            ijab_aa_fly = self.gen_ijab_fly(B_ij=B_ij_ex_alpha, B_ab=B_ab_ex_alpha)
            ijab_bb_fly = self.gen_ijab_fly(B_ij=B_ij_ex_beta,  B_ab=B_ab_ex_beta)

            ibja_aa_fly = self.gen_ibja_fly(B_ia=B_ia_ex_alpha)
            ibja_bb_fly = self.gen_ibja_fly(B_ia=B_ia_ex_beta)

            def UKS_TDDFT_hybrid_mv(X,Y):
                '''
                UKS
                [A B][X] = [AX+BY] = [U1]
                [B A][Y]   [AY+BX]   [U2]
                A B have 4 blocks, αα, αβ, βα, ββ
                A = [ Aαα Aαβ ]   B = [ Bαα Bαβ ]
                    [ Aβα Aββ ]       [ Bβα Bββ ]

                X = [ Xα ]        Y = [ Yα ]
                    [ Xβ ]            [ Yβ ]

                (A+B)αα, (A+B)αβ is shown below

                βα, ββ can be obtained by change α to β
                we compute (A+B)(X+Y) and (A-B)(X-Y)

                V:= X+Y
                (A+B)αα Vα = hdiag_fly(Vα) + 2*iaαjbα_fly(Vα) - a_x*[ijαabα_fly(Vα) + ibαjaα_fly(Vα)]
                (A+B)αβ Vβ = 2*iaαjbβ_fly(Vβ)

                V:= X-Y
                (A-B)αα Vα = hdiag_fly(Vα) - a_x*[ijαabα_fly(Vα) - ibαjaα_fly(Vα)]
                (A-B)αβ Vβ = 0

                A+B = [ Cαα Cαβ ]   x+y = [ Vα ]
                      [ Cβα Cββ ]         [ Vβ ]
                (A+B)(x+y) =   [ Cαα Vα + Cαβ Vβ ]  = ApB_XpY
                               [ Cβα Vα + Cββ Vβ ]

                A-B = [ Cαα  0  ]   x-y = [ Vα ]
                      [  0  Cββ ]         [ Vβ ]
                (A-B)(x-y) =   [ Cαα Vα ]    = AmB_XmY
                               [ Cββ Vβ ]
                '''

                X_a = X[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                X_b = X[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)
                Y_a = Y[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                Y_b = Y[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                XpY_a = X_a + Y_a
                XpY_b = X_b + Y_b

                XmY_a = X_a - Y_a
                XmY_b = X_b - Y_b

                '''============== (A+B) (X+Y) ================'''
                '''(A+B)aa(X+Y)a'''
                ApB_XpY_aa = hdiag_a_fly(XpY_a) + 2*iajb_aa_fly(XpY_a) - a_x*(ijab_aa_fly(XpY_a) + ibja_aa_fly(XpY_a))
                '''(A+B)bb(X+Y)b'''
                ApB_XpY_bb = hdiag_b_fly(XpY_b) + 2*iajb_bb_fly(XpY_b) - a_x*(ijab_bb_fly(XpY_b) + ibja_bb_fly(XpY_b))
                '''(A+B)ab(X+Y)b'''
                ApB_XpY_ab = 2*iajb_ab_fly(XpY_b)
                '''(A+B)ba(X+Y)a'''
                ApB_XpY_ba = 2*iajb_ba_fly(XpY_a)

                '''============== (A-B) (X-Y) ================'''
                '''(A-B)aa(X-Y)a'''
                AmB_XmY_aa = hdiag_a_fly(XmY_a) - a_x*(ijab_aa_fly(XmY_a) - ibja_aa_fly(XmY_a))
                '''(A-B)bb(X-Y)b'''
                AmB_XmY_bb = hdiag_b_fly(XmY_b) - a_x*(ijab_bb_fly(XmY_b) - ibja_bb_fly(XmY_b))

                ''' (A-B)ab(X-Y)b
                    AmB_XmY_ab = 0
                    (A-B)ba(X-Y)a
                    AmB_XmY_ba = 0
                '''

                ''' (A+B)(X+Y) = AX + BY + AY + BX   (1) ApB_XpY
                    (A-B)(X-Y) = AX + BY - AY - BX   (2) AmB_XmY
                    (1) + (1) /2 = AX + BY = U1
                    (1) - (2) /2 = AY + BX = U2
                '''
                ApB_XpY_alpha = (ApB_XpY_aa + ApB_XpY_ab).reshape(A_aa_size,-1)
                ApB_XpY_beta  = (ApB_XpY_ba + ApB_XpY_bb).reshape(A_bb_size,-1)
                ApB_XpY = np.vstack((ApB_XpY_alpha, ApB_XpY_beta))

                AmB_XmY_alpha = AmB_XmY_aa.reshape(A_aa_size,-1)
                AmB_XmY_beta  = AmB_XmY_bb.reshape(A_bb_size,-1)
                AmB_XmY = np.vstack((AmB_XmY_alpha, AmB_XmY_beta))

                U1 = (ApB_XpY + AmB_XmY)/2
                U2 = (ApB_XpY - AmB_XmY)/2

                return U1, U2

            return UKS_TDDFT_hybrid_mv, hdiag

        elif a_x == 0:
            ''' UKS TDDFT pure '''

            hdiag_a_sqrt_fly, hdiag_a_sq = self.gen_hdiag_fly(mo_energy=mo_energy[0],
                                                          n_occ=n_occ_a,
                                                          n_vir=n_vir_a,
                                                          sqrt=True)
            hdiag_b_sqrt_fly, hdiag_b_sq = self.gen_hdiag_fly(mo_energy=mo_energy[1],
                                                          n_occ=n_occ_b,
                                                          n_vir=n_vir_b,
                                                          sqrt=True)
            '''hdiag_sq: preconditioner'''
            hdiag_sq = np.vstack((hdiag_a_sq.reshape(-1,1), hdiag_b_sq.reshape(-1,1))).reshape(-1)

            B_ia_alpha = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_a,
                                    mo_coeff=mo_coeff[0],
                                    calc='coulomb_only')
            B_ia_beta = self.gen_B(uvQL=uvQL,
                                    n_occ=n_occ_b,
                                    mo_coeff=mo_coeff[1],
                                    calc='coulomb_only')

            iajb_aa_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_alpha)
            iajb_ab_fly = self.gen_iajb_fly(B_left=B_ia_alpha, B_right=B_ia_beta)
            iajb_ba_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_alpha)
            iajb_bb_fly = self.gen_iajb_fly(B_left=B_ia_beta,  B_right=B_ia_beta)

            def UKS_TDDFT_pure_mv(Z):
                '''       MZ = Z w^2
                    M = (A-B)^1/2(A+B)(A-B)^1/2
                    Z = (A-B)^1/2(X-Y)

                    X+Y = (A-B)^1/2 Z * 1/w
                    A+B = hdiag_fly(V) + 4*iajb_fly(V)
                    (A-B)^1/2 = hdiag_sqrt_fly(V)


                    M =  [ (A-B)^1/2αα    0   ] [ (A+B)αα (A+B)αβ ] [ (A-B)^1/2αα    0   ]            Z = [ Zα ]
                         [    0   (A-B)^1/2ββ ] [ (A+B)βα (A+B)ββ ] [    0   (A-B)^1/2ββ ]                [ Zβ ]
                '''
                Z_a = Z[:A_aa_size,:].reshape(n_occ_a, n_vir_a, -1)
                Z_b = Z[A_aa_size:,:].reshape(n_occ_b, n_vir_b, -1)

                AmB_aa_sqrt_Z_a = hdiag_a_sqrt_fly(Z_a)
                AmB_bb_sqrt_Z_b = hdiag_b_sqrt_fly(Z_b)

                ApB_aa_sqrt_V = hdiag_a_fly(AmB_aa_sqrt_Z_a) + 2*iajb_aa_fly(AmB_aa_sqrt_Z_a)
                ApB_ab_sqrt_V = 2*iajb_ab_fly(AmB_bb_sqrt_Z_b)
                ApB_ba_sqrt_V = 2*iajb_ba_fly(AmB_aa_sqrt_Z_a)
                ApB_bb_sqrt_V = hdiag_b_fly(AmB_bb_sqrt_Z_b) + 2*iajb_bb_fly(AmB_bb_sqrt_Z_b)

                MZ_a = hdiag_a_sqrt_fly(ApB_aa_sqrt_V + ApB_ab_sqrt_V).reshape(A_aa_size, -1)
                MZ_b = hdiag_b_sqrt_fly(ApB_ba_sqrt_V + ApB_bb_sqrt_V).reshape(A_bb_size, -1)

                MZ = np.vstack((MZ_a, MZ_b))
                # print(MZ.shape)
                return MZ

            return UKS_TDDFT_pure_mv, hdiag_sq

    kernel = uks.TDDFT.kernel
