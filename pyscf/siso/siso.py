#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

'''
Quasi-Degenerate Perturbation Theory Based Spin-Orbit Coupling Treatment
Also known as State-Interaction Spin-Orbit (SI-SO) method.
'''

import numpy as np
from sympy import symbols
from sympy.physics.quantum import cg
from itertools import product
from functools import reduce
from pyscf import scf, lib, fci, mcpdft
from pyscf.siso import socaddons
import ctypes

libsiso = lib.load_library('libsiso')
libsiso.SOCcompute_ss.restype = None
libsiso.SOCcompute_ssp.restype = None
libsiso.SOCcompute_ssm.restype = None

logger = lib.logger

au2ev = 27.21138602
au2cminv = 219474.6313705

# Forked from: https://github.com/IrisA144/liblan_preview

def calculate_zmat(siso, somf=True, amf=True, mmf=False, soc1e=True, soc2e=True, ham='DKH'):
    """
    Computing the SOC integrals.

    Args:
        siso:
            instance of class SISO
        somf:
            spin-orbit mean field integrals.
        amf:
            atomic mean field integrals.
        mmf:
            molecular mean field integrals.
        soc1e:
            include 1e SOC integrals.
        soc2e:
            include 2e SOC integrals.
        ham:
            SOC Hamiltonian (BP or DKH)
    Returns:
        zsoc:
            SOC integrals of dimension (3, ncas, ncas)
    """
    mc = siso.mc
    mol = mc._scf.mol
    ncas = mc.ncas
    ncore = mc.ncore

    dm = None
    if mmf:
        dm = mc.make_rdm1()

    # Generate the SOC integrals
    hso = socaddons.socintegrals(mol, somf=somf, amf=amf, mmf=mmf,
                                 soc1e=soc1e, soc2e=soc2e, ham=ham, dm=dm)

    # Basis transformation
    mo_cas = mc.mo_coeff[:, ncore:ncore+ncas]
    h1 = np.asarray([reduce(np.dot, (mo_cas.T, hso_, mo_cas)) for hso_ in hso])
    # Cartesian to Spherical tensor transformation
    zsoc = np.zeros((3, ncas, ncas), dtype=hso[0].dtype)
    zsoc[0] = 1./np.sqrt(2)  * (h1[0] - 1.j*h1[1])
    zsoc[1] = h1[2]
    zsoc[2] = -1./np.sqrt(2) * (h1[0] + 1.j*h1[1])

    del hso, h1

    return zsoc

def assemble_civecs(siso):
    '''
    Assemble the ci vectors for the model space.
    args:
        siso:
            instance of class SISO
    returns:
        cimat: list
    '''
    mc = siso.mc
    ci_mc = mc.ci
    twoslst = siso.twoslst
    statelst = siso.statelis

    cimat = []

    for i in range(len(twoslst)):
        start_idx = int(np.sum(statelst[:twoslst[i]]))
        try:
            end_idx = int(np.sum(statelst[:twoslst[i] + 1]))
        except IndexError:
            end_idx = int(np.sum(statelst))

        cimat.append(np.asarray(ci_mc[start_idx:end_idx]))

    imds = siso.imds
    imds.c = cimat

    return cimat

def assemble_energy(siso):
    '''
    Assemble the energy for the model space.
    '''
    mc = siso.mc
    e_states = mc.e_states
    twoslst = siso.twoslst
    tottwos = len(twoslst)
    statelst = siso.statelis

    e = []
    for i in range(tottwos):
        start_idx = int(np.sum(statelst[:twoslst[i]]))
        try:
            end_idx = int(np.sum(statelst[:twoslst[i] + 1]))
        except IndexError:
            end_idx = int(np.sum(statelst))
        e.append(np.asarray(e_states[start_idx:end_idx]))

    imds = siso.imds
    imds.e = e

    return e

def assemble_amat(siso):
    '''
    Assemble the coupling coefficients for the model space.
    Args:
        siso:
            instance of class SISO
    Returns:
        amat: list
    '''
    def _gen_linkstr_index(ms, ncasorb, alphae, betae):
        ms_map = {-1:fci.cistring.gen_des_str_index,
                 0:fci.cistring.gen_linkstr_index,
                 1:fci.cistring.gen_cre_str_index}
        return [ms_map.get(ms[0])(ncasorb, alphae),
                ms_map.get(ms[1])(ncasorb, betae)]

    def _get_alpha_beta(nelec, S):
        alphae = (nelec + S) // 2
        betae = nelec - alphae
        return alphae, betae

    stuples = siso.stuples
    mc = siso.mc
    ncasorb = list(range(mc.ncas))
    nelec = sum(mc.nelecas)

    amat = []
    for (s1, s2) in stuples:
        if s1 == s2:  # SS part
            nelec_a, nelec_b = _get_alpha_beta(nelec, s1)
            amat.append(_gen_linkstr_index((0,0), ncasorb, nelec_a, nelec_b))
        elif s1 + 2 == s2:  # SS+1 part
            nelec_a, nelec_b = _get_alpha_beta(nelec, s1+2)
            amat.append(_gen_linkstr_index((-1,1), ncasorb, nelec_a, nelec_b))
        elif s1 == s2 + 2: # SS-1 part
            nelec_a, nelec_b = _get_alpha_beta(nelec, s1-2)
            amat.append(_gen_linkstr_index((1,-1), ncasorb, nelec_a, nelec_b))
        else: # No connection
            amat.append(0)

    imds = siso.imds
    imds.a = amat
    return amat

def compute_cg_coefficients(S, Ms=0):
    """
    Compute the Clebsch-Gordan coefficients for the spin-orbit coupling.

    Args:
        siso:
            instance of class SISO
    Returns:
        cg_coeffs:
            a list of Clebsch-Gordan coefficients for each state
    """
    assert Ms in (0, 1, -1), "Ms should be either 0, 1 or -1."

    Ms_map = {0:S+1, 1:S+3, -1:S-1}

    ss = symbols('S')
    sbra = S + 1
    sket = Ms_map[Ms]

    g = np.zeros((3, sbra, sket), dtype='complex')

    for g0 in range(3):
        for g1 in range(sbra):
            phase = (-1) ** (g0 + S / 2 - (g1 - S / 2))
            for g2 in range(sket):
                g[g0, g1, g2] += phase * cg.Wigner3j(ss/2, -(g1 - ss/2),
                                                     1, 1 - g0,
                                                     ss/2 + Ms, g2 - ss/2 - Ms).subs(ss,S).doit()
    return g

def compute_dmat(siso):
    '''
    '''
    twoslst = siso.twoslst
    stuples = siso.stuples
    imds = siso.imds

    d_col = []

    def flatten_to_ptr(array, dtype):
        return np.ascontiguousarray(array.flatten()).ctypes.data_as(ctypes.POINTER(dtype))

    for i, (s1,s2) in enumerate(stuples):
        if s1 == s2:  # SS part
            S = s1
            iS = np.where(twoslst == S)[0][0]
            zmat = imds.z # SOC integrals
            civecket = imds.c[iS].astype(np.complex128) # CI vectors
            civecbra = civecket
            alphamat, betamat = imds.a[i] # coupling coefficients
            b = np.zeros((3, *civecket.shape), dtype='complex')

            # Instead of passing all of these shapes, I am passing a single shape array
            shapearray = np.array([*b.shape,
                                   alphamat.shape[1],
                                   betamat.shape[1],
                                   zmat.shape[1],
                                   civecket.shape[1],
                                   civecket.shape[2]],
                                   dtype=np.int32, order='C')

            b = np.ascontiguousarray(b.flatten())

            libsiso.SOCcompute_ss.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # zmat
                ctypes.POINTER(ctypes.c_double),  # civecs
                ctypes.POINTER(ctypes.c_double),  # bdata
                ctypes.POINTER(ctypes.c_int),     # alphadet
                ctypes.POINTER(ctypes.c_int),     # betadet
                ctypes.POINTER(ctypes.c_int)      # shapearray
            ]

            libsiso.SOCcompute_ss(
                flatten_to_ptr(zmat, ctypes.c_double),
                flatten_to_ptr(civecket, ctypes.c_double),
                b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                flatten_to_ptr(alphamat, ctypes.c_int),
                flatten_to_ptr(betamat, ctypes.c_int),
                flatten_to_ptr(shapearray, ctypes.c_int))

            b = b.reshape(3, *civecket.shape)

            # WE TDM
            w = np.einsum('kij, mnij->mnk', civecbra, b)

            # Multiply by the Clebsch-Gordan coefficients
            g = compute_cg_coefficients(S, Ms=0)
            d = np.einsum('mij, mkl->kilj', g, w)

            d_col.append(d)

        elif s1 + 2 == s2:  # SS+1 part
            S = s1
            iS = np.where(twoslst == S)[0][0]
            iSp = np.where(twoslst == S + 2)[0][0]
            zmat = imds.z # SOC integrals
            civecket = imds.c[iS].astype(np.complex128) # CI vectors
            civecbra = imds.c[iSp].astype(np.complex128)
            alphamat, betamat = imds.a[i] # coupling coefficients
            b = np.zeros((3, civecket.shape[0], civecbra.shape[1], civecbra.shape[2]), dtype='complex')

            # Instead of passing all of these shapes, I am passing a single shape array
            shapearray = np.array([*b.shape,
                                   alphamat.shape[1],
                                   betamat.shape[1],
                                   zmat.shape[1],
                                   civecket.shape[1],
                                   civecket.shape[2]],
                                   dtype=np.int32, order='C')

            b = np.ascontiguousarray(b.flatten())

            libsiso.SOCcompute_ssp.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # zmat
                ctypes.POINTER(ctypes.c_double),  # civecs
                ctypes.POINTER(ctypes.c_double),  # bdata
                ctypes.POINTER(ctypes.c_int),     # alphadet
                ctypes.POINTER(ctypes.c_int),     # betadet
                ctypes.POINTER(ctypes.c_int)      # shapearray
            ]

            libsiso.SOCcompute_ssp(
                flatten_to_ptr(zmat, ctypes.c_double),
                flatten_to_ptr(civecket, ctypes.c_double),
                b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                flatten_to_ptr(alphamat, ctypes.c_int),
                flatten_to_ptr(betamat, ctypes.c_int),
                flatten_to_ptr(shapearray, ctypes.c_int))

            b = b.reshape(3, civecket.shape[0], civecbra.shape[1], civecbra.shape[2])

            # WE TDM
            w = np.einsum('kij, mnij->mnk', civecbra, b)

            # Multiply by the Clebsch-Gordan coefficients
            g = compute_cg_coefficients(S, Ms=1)
            d = np.einsum('mij, mkl->kilj', g, w)
            d_col.append(d)

        elif s1 == s2 + 2: # SS-1 part
            S = s1
            iS = np.where(twoslst == S)[0][0]
            iSm = np.where(twoslst == S - 2)[0][0]
            zmat = imds.z # SOC integrals
            civecket = imds.c[iS].astype(np.complex128) # CI vectors
            civecbra = imds.c[iSm].astype(np.complex128)
            alphamat, betamat = imds.a[i] # coupling coefficients

            b = np.zeros((3, civecket.shape[0], civecbra.shape[1], civecbra.shape[2]), dtype='complex')

            # Instead of passing all of these shapes, I am passing a single shape array
            shapearray = np.array([*b.shape,
                                   alphamat.shape[1],
                                   betamat.shape[1],
                                   zmat.shape[1],
                                   civecket.shape[1],
                                   civecket.shape[2]],
                                   dtype=np.int32, order='C')

            b = np.ascontiguousarray(b.flatten())

            libsiso.SOCcompute_ssm.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # zmat
                ctypes.POINTER(ctypes.c_double),  # civecs
                ctypes.POINTER(ctypes.c_double),  # bdata
                ctypes.POINTER(ctypes.c_int),     # alphadet
                ctypes.POINTER(ctypes.c_int),     # betadet
                ctypes.POINTER(ctypes.c_int)      # shapearray
            ]
            libsiso.SOCcompute_ssm(
                flatten_to_ptr(zmat, ctypes.c_double),
                flatten_to_ptr(civecket, ctypes.c_double),
                b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                flatten_to_ptr(alphamat, ctypes.c_int),
                flatten_to_ptr(betamat, ctypes.c_int),
                flatten_to_ptr(shapearray, ctypes.c_int))

            b = b.reshape(3, civecket.shape[0], civecbra.shape[1], civecbra.shape[2])

            # WE TDM
            w = np.einsum('kij, mnij->mnk', civecbra, b)

            # Multiply by the Clebsch-Gordan coefficients
            g = compute_cg_coefficients(S, Ms=-1)
            d = np.einsum('mij, mkl->kilj', g, w)
            d_col.append(d)

        else: # No connection
            d_col.append(0)

    return d_col

def compute_hamiltonian(siso):
    """
    Compute the Hamiltonian matrix for the spin-orbit coupling.

    Args:
        siso:
            instance of class SISO
    Returns:
        hamiltonian:
            a list of Hamiltonian matrices for each state
    """
    statelst = siso.statelis
    twoslst = siso.twoslst
    totalspins = len(twoslst)
    stuples = siso.stuples
    imds = siso.imds
    d = imds.d
    e = imds.e
    h_col = [[0] * totalspins for _ in range(totalspins)]

    for i in range(totalspins):
        for j in range(totalspins):
            s1, s2 = twoslst[i], twoslst[j]
            indt = stuples.index((s1, s2))
            counter = [(i+1)*(x) for i, x in enumerate(statelst)]

            if s1 == s2:
                S = s1
                nstates = counter[S]
                if S!=0:
                    coeff = np.sqrt((S / 2 + 1) * (S + 1) / (S / 2)) / 2
                else:
                    coeff = 0.0

                h = coeff * d[indt].reshape(nstates, nstates, order='C')

                for ns in range(nstates):
                    n1 = divmod(ns, S+1)[0]
                    h[ns, ns] += e[i][n1]

            elif s1 + 2 == s2:
                S = s1
                nstatesa = counter[S]
                nstatesb = counter[S + 2]
                coeff = np.sqrt((S + 3) / 2)
                h = coeff * d[indt].reshape((nstatesa, nstatesb), order='C')

            elif s1 == s2 + 2:  # SS-1 part
                S = s1
                nstatesa = counter[S]
                nstatesb = counter[S - 2]
                coeff = -np.sqrt((S + 1) / 2)
                h = coeff * d[indt].reshape((nstatesa, nstatesb), order='C')

            else:  # No connection
                nstatesa = counter[s1]
                nstatesb = counter[s2]
                h = np.zeros((nstatesa, nstatesb), dtype=np.complex128)
            h_col[i][j] = h

    return np.block(h_col)

def build_imds(siso):
    imds = siso.imds
    somf = siso.somf
    amf = siso.amf
    mmf = siso.mmf
    soc1e = siso.soc1e
    soc2e = siso.soc2e
    ham = siso.ham

    imds.z = calculate_zmat(siso, somf=somf, amf=amf, mmf=mmf,
                            soc1e=soc1e, soc2e=soc2e, ham=ham)
    imds.a = assemble_amat(siso)
    imds.c = assemble_civecs(siso)
    imds.e = assemble_energy(siso)
    imds.d = compute_dmat(siso)
    return siso

def kernel(siso):
    """
    Driver function for SI-SO
    """
    logger.debug(siso, 'Starting SI-SO kernel')
    siso.build_imds()
    hso = siso.compute_hamiltonian()
    # Sanity checks
    assert np.allclose(hso, hso.conj().T),\
        f"Hamiltonian is not Hermitian, max deviation: {np.max(np.abs(hso - hso.conj().T))}"
    siso.si_energies, siso.si_vecs = np.linalg.eigh(hso)
    siso._finalize()
    return siso.si_energies, siso.si_vecs

class _IMDS:
    """
    SI-SO intermediates
    Attributes:
    ----------
        self.z   :   (3,ncas,ncas)
        self.a   :   (nSt,2,ncia||b,4)
        self.b   :   (3, nstates nciam ncib) for given nS
        self.c   :   (nS,nstates,ncia,ncib)
        self.d   :   (nS,nstates,nstates)
        self.e   :   (nS,nstates)
    """
    def __init__(self):
        self.zmat = None
        self.a = None
        self.c = None
        self.e = None
        self.d = None

class SISO(lib.StreamObject):
    """
    Quasi-Degenerate Perturbation Theory Based Spin-Orbit Coupling Treatment.:
    State interaction Spin-Orbit Coupling (SISO) class.
    """
    _keys = ['modelspace', 'statelst', 'twoslst', 'stuples',
             'si_energies', 'si_vecs']

    def __init__(self, mc, modelspace, somf=True, amf=True, mmf=False, soc1e=True, soc2e=True, ham='DKH'):
        self.mc = mc
        self.somf = somf
        self.amf = amf
        self.mmf = mmf
        self.soc1e = soc1e
        self.soc2e = soc2e
        self.ham = ham.upper() if isinstance(ham, str) else ham
        self.imds = _IMDS()
        self.initialize_(modelspace)
        self.sanity_checks()
        self.dump_flags()

    def initialize_(self, modelspace):
        self.modelspace = socaddons._validate_modelspace(
            modelspace, mol=self.mc._scf.mol, ncas=self.mc.ncas,
            nelecas=self.mc.nelecas)
        spin_indices = [state[1] - 1 for state in self.modelspace]
        statelis_ = np.zeros(max(spin_indices) + 1, dtype=int)
        for nroots, spinmult, _ in self.modelspace:
            # Multiple symmetry sectors can have the same spin multiplicity.
            statelis_[spinmult - 1] += nroots
        self.statelis = statelis_.tolist()
        self.twoslst = np.nonzero(self.statelis)[0]
        self.stuples = [x for x in product(self.twoslst, self.twoslst)]
        return self

    def sanity_checks(self):
        """
        Perform sanity checks on the input parameters.
        """
        flags = {'somf': self.somf, 'amf': self.amf, 'mmf': self.mmf,
                 'soc1e': self.soc1e, 'soc2e': self.soc2e}
        if any(not isinstance(value, (bool, np.bool_))
               for value in flags.values()):
            raise TypeError("somf, amf, mmf, soc1e, and soc2e must be boolean")
        if self.ham not in ('BP', 'DKH'):
            raise ValueError("ham must be 'BP' or 'DKH'")
        if not self.somf:
            raise NotImplementedError(
                "Explicit 2e SOC integrals are not implemented yet")
        if self.mc._scf.mol.has_ecp():
            raise NotImplementedError("ECP is not supported yet.")
        if not self.soc1e and not self.soc2e:
            raise ValueError("At least one of soc1e and soc2e must be enabled")
        if self.amf == self.mmf:
            raise ValueError("Exactly one of amf and mmf must be enabled")

        if isinstance(self.mc, mcpdft.MultiStateMCPDFTSolver):
            self.sort_eigenstates_forlpdft()

        expected_nroots = sum(state[0] for state in self.modelspace)
        solvers = getattr(self.mc.fcisolver, 'fcisolvers', None)
        if solvers is not None:
            if len(solvers) != len(self.modelspace):
                raise ValueError(
                    "modelspace must contain one entry for each state-average "
                    f"FCI solver ({len(self.modelspace)} entries for "
                    f"{len(solvers)} solvers)")
            for index, (solver, state) in enumerate(zip(solvers, self.modelspace)):
                nroots, spinmult, wfnsym = state
                solver_nroots = int(getattr(solver, 'nroots', 1))
                solver_spinmult = int(getattr(solver, 'spin', spinmult - 1)) + 1
                if (solver_nroots, solver_spinmult) != (nroots, spinmult):
                    raise ValueError(
                        f"modelspace entry {index} requests ({nroots}, {spinmult}), "
                        "but the corresponding FCI solver has "
                        f"({solver_nroots}, {solver_spinmult})")
                solver_wfnsym = getattr(solver, 'wfnsym', None)
                if wfnsym is not None and solver_wfnsym != wfnsym:
                    raise ValueError(
                        f"modelspace entry {index} requests wfnsym {wfnsym!r}, "
                        f"but the corresponding FCI solver uses {solver_wfnsym!r}")

        if not hasattr(self.mc, 'e_states') or self.mc.e_states is None:
            raise ValueError("mc must contain state energies before SISO is initialized")
        actual_nroots = np.atleast_1d(self.mc.e_states).size
        if actual_nroots != expected_nroots:
            raise ValueError(
                f"modelspace contains {expected_nroots} roots, but mc.e_states "
                f"contains {actual_nroots}")
        if isinstance(self.mc.ci, (list, tuple)) and len(self.mc.ci) != expected_nroots:
            raise ValueError(
                f"modelspace contains {expected_nroots} roots, but mc.ci "
                f"contains {len(self.mc.ci)}")

        ss = np.asarray(self.mc.fcisolver.states_spin_square(
            self.mc.ci, self.mc.ncas, self.mc.nelecas)[0])
        self.spin_contanimation_check(ss)
        actual_spin_indices = np.rint(np.sqrt(1.0 + 4.0 * ss) - 1.0).astype(int)
        expected_spin_indices = np.concatenate([
            np.repeat(spinmult - 1, nroots)
            for nroots, spinmult, _ in self.modelspace
        ])
        if not np.array_equal(actual_spin_indices, expected_spin_indices):
            actual_mults = (actual_spin_indices + 1).tolist()
            expected_mults = (expected_spin_indices + 1).tolist()
            raise ValueError(
                "modelspace spin multiplicities do not match the ordered mc states: "
                f"expected {expected_mults}, found {actual_mults}")
        self._state_s2 = ss
        return self

    def sort_eigenstates_forlpdft(self):
        '''
        Sorting the L-PDFT ci-vecs and energies accoarding to spin-states.
        '''
        ss = self.mc.fcisolver.states_spin_square(self.mc.ci, self.mc.ncas, self.mc.nelecas)[0]
        idx = np.argsort(ss)
        self.mc.ci = [self.mc.ci[i] for i in idx]
        self.mc.e_states = [self.mc.e_states[i] for i in idx]

    def dump_flags(self):
        log = logger.Logger(self.mc.stdout, self.mc.verbose)
        log.note(" ")
        log.info('******** %s ********', self.__class__)
        log.note("somf: %s", self.somf)
        log.note("amfi: %s", self.amf)
        log.note("mmfi: %s", self.mmf)
        log.note("1e soc: %s", self.soc1e)
        log.note("2e soc: %s", self.soc2e)
        log.note("ham: %s", self.ham)
        log.note("model space: %s", self.modelspace)
        log.note("speed of light: %.8f a.u.", lib.param.LIGHT_SPEED)
        log.note(" ")
        self._initialize()
        return self

    def build_imds(self):
        return build_imds(self)

    def compute_hamiltonian(self):
        return compute_hamiltonian(self)

    def kernel(self):
        return kernel(self)

    @staticmethod
    def spin_contanimation_check(ss, spinconttol=1e-4, max_spin=15):
        '''
        Checking the spin-purity of the modeal state wave functions.
        '''
        sslookup = [s/2*(s/2+1) for s in range(max_spin)]
        for s2 in ss:
            if not any(abs(s2 - ssideal) <= spinconttol for ssideal in sslookup):
                raise ValueError("model states are not spin-pure")

    def _initialize(self):
        '''
        Initial SOC Free Model Space
        '''
        mc = self.mc
        log = logger.Logger(mc.stdout, mc.verbose)
        nroots=len(self.mc.e_states)
        e_states = np.array(mc.e_states)
        sortedindx = np.argsort(e_states)
        e_states = e_states[sortedindx]
        ss = getattr(self, '_state_s2', None)
        if ss is None:
            ss = mc.fcisolver.states_spin_square(
                mc.ci, mc.ncas, mc.nelecas)[0]
        ss = np.asarray(ss)
        ss = ss[sortedindx]

        log.info('******** %s ********', "Spin Orbit Free Energetics")
        for i in range(nroots):
            log.note(" State %d Total Energy = %.10f S^2 = %.2f",
                    i,
                    e_states[i],
                    ss[i])

        log.note(" ")
        log.info('******** %s ********',"Relative Spin Orbit Free Energetics")
        log.note("State         Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)")
        for i in range(nroots):
            log.note(" {:<10} {:>20.9f} {:>20.5f} {:>20.5f}".format(
                i,
                e_states[i] - e_states[0],
                au2ev * (e_states[i] - e_states[0]),
                au2cminv * (e_states[i] - e_states[0])))

        self.spin_contanimation_check(ss)

    def _finalize(self):
        '''
        Final SOC States
        '''
        mc = self.mc

        log = logger.Logger(mc.stdout, mc.verbose)
        nroots=len(self.si_energies)
        log.note(" ")
        log.info('******** %s ********',"Spin Orbit Coupling Energetics")
        for i in range(nroots):
            log.note(" SO-CASSI State %d Total Energy = %.10f ",
                    i,
                    self.si_energies[i])

        log.note(" ")
        log.info('******** %s ********',"Relative Spin Orbit Coupling Energetics")
        log.note("SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm^-1)")
        for i in range(nroots):
            log.note(" {:<10} {:>20.9f} {:>20.5f} {:>20.5f}".format(
                i,
                self.si_energies[i] - self.si_energies[0],
                au2ev * (self.si_energies[i] - self.si_energies[0]),
                au2cminv * (self.si_energies[i] - self.si_energies[0])))
