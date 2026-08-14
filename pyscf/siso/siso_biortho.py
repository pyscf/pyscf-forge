from functools import reduce
import ctypes

import numpy as np
from scipy import linalg

from pyscf import ao2mo, lib, mcpdft
from pyscf.fci import cistring, direct_spin1
from pyscf.siso import siso as _siso
from pyscf.siso import socaddons
from pyscf.siso.biortho import (
    BiorthogonalPair,
    LU_THRESH,
    biorthogonalize,
)

logger = lib.logger

# Author: Bhavnesh Jangid

'''
Biorthogonal state-interaction and spin-orbit state-interaction interfaces
'''


def _mixed_scalar_integrals(mc, mo_left, mo_right):
    '''
    Computing the inactive energy, effective one-electron integrals, and
    two-electron integrals in the mixed biorthonormal MO basis.

    args:
        mc:
            CASCI/CASSCF-like object
        mo_left, mo_right: np.array
            biorthonormal MO coefficients for the left and right states
    returns:
        ecore: float
            inactive and nuclear energy
        h1: np.array
            effective one-electron active-space integrals
        h2: np.array
            two-electron active-space integrals
    '''
    mf = mc._scf
    mol = mf.mol
    ncore = mc.ncore
    ncas = mc.ncas
    left_core = mo_left[:, :ncore]
    right_core = mo_right[:, :ncore]
    left_active = mo_left[:, ncore:ncore + ncas]
    right_active = mo_right[:, ncore:ncore + ncas]

    # OpenMolcas DIMAT: D_inactive = 2 C_left C_right^T.
    dm_core = 2.0 * left_core @ right_core.T
    hcore = mf.get_hcore(mol)
    if ncore:
        vj, vk = mf.get_jk(mol, dm_core, hermi=0)
        veff = vj - 0.5 * vk
    else:
        veff = np.zeros_like(hcore)

    ecore = mol.energy_nuc()
    ecore += np.einsum("ij,ij->", hcore + 0.5 * veff, dm_core)
    h1 = reduce(np.dot, (left_active.T, hcore + veff, right_active))

    eri_source = getattr(mf, "_eri", None)
    if eri_source is None:
        eri_source = mol
    h2 = ao2mo.general(
        eri_source,
        (left_active, right_active, left_active, right_active),
        compact=False,
    ).reshape((ncas,) * 4)

    return ecore, h1, h2


def _spherical_soc(hso, mo_left, mo_right, ncore, ncas):
    '''
    Transforming Cartesian AO SOC integrals to spherical components in the
    mixed biorthonormal active MO basis.

    args:
        hso: np.array
            Cartesian AO SOC integrals
        mo_left, mo_right: np.array
            biorthonormal MO coefficients for the left and right states
        ncore: int
            number of inactive orbitals
        ncas: int
            number of active orbitals
    returns:
        zsoc: np.array
            spherical SOC integrals
    '''
    left = mo_left[:, ncore:ncore + ncas]
    right = mo_right[:, ncore:ncore + ncas]
    h1 = np.asarray([reduce(np.dot, (left.T, x, right)) for x in hso])
    zsoc = np.empty((3, ncas, ncas), dtype=np.complex128)
    zsoc[0] = (h1[0] - 1j * h1[1]) / np.sqrt(2.0)
    zsoc[1] = h1[2]
    zsoc[2] = -(h1[0] + 1j * h1[1]) / np.sqrt(2.0)
    return zsoc


def _flatten_to_ptr(array, dtype):
    '''
    Converting a NumPy array to a contiguous ctypes pointer.
    '''
    array = np.ascontiguousarray(array)
    return array, array.ctypes.data_as(ctypes.POINTER(dtype))


def _soc_action(mode, zmat, ciket, norb, nelec, twos):
    '''
    Applying the reduced triplet SOC operator using the libsiso kernels.

    args:
        mode: str
            spin-coupling case: ``ss``, ``ssp``, or ``ssm``
        zmat: np.array
            spherical SOC integrals
        ciket: np.array
            ket CI vector
        norb: int
            number of active orbitals
        nelec: int
            total number of active electrons
        twos: int
            twice the ket spin
    returns:
        b: np.array
            SOC-contracted CI vectors
    '''
    nalpha = (nelec + twos) // 2
    nbeta = nelec - nalpha
    orbitals = list(range(norb))
    if mode == "ss":
        alpha = cistring.gen_linkstr_index(orbitals, nalpha)
        beta = cistring.gen_linkstr_index(orbitals, nbeta)
        output_shape = ciket.shape
        function = _siso.libsiso.SOCcompute_ss
    elif mode == "ssp":
        # Ket has S; output determinant space has S+1.
        alpha_high = nalpha + 1
        beta_high = nbeta - 1
        alpha = cistring.gen_des_str_index(orbitals, alpha_high)
        beta = cistring.gen_cre_str_index(orbitals, beta_high)
        output_shape = (
            cistring.num_strings(norb, alpha_high),
            cistring.num_strings(norb, beta_high),
        )
        function = _siso.libsiso.SOCcompute_ssp
    elif mode == "ssm":
        # Ket has S; output determinant space has S-1.
        alpha_low = nalpha - 1
        beta_low = nbeta + 1
        alpha = cistring.gen_cre_str_index(orbitals, alpha_low)
        beta = cistring.gen_des_str_index(orbitals, beta_low)
        output_shape = (
            cistring.num_strings(norb, alpha_low),
            cistring.num_strings(norb, beta_low),
        )
        function = _siso.libsiso.SOCcompute_ssm
    else:
        raise ValueError(f"Unknown SOC action mode {mode!r}")

    ciket = np.asarray(ciket, dtype=np.complex128).reshape((1,) + ciket.shape)
    b = np.zeros((3, 1) + output_shape, dtype=np.complex128)
    shapes = np.array(
        [
            *b.shape,
            alpha.shape[1],
            beta.shape[1],
            norb,
            ciket.shape[1],
            ciket.shape[2],
        ],
        dtype=np.int32,
    )

    zmat, zptr = _flatten_to_ptr(zmat, ctypes.c_double)
    ciket, ciptr = _flatten_to_ptr(ciket, ctypes.c_double)
    alpha, aptr = _flatten_to_ptr(alpha, ctypes.c_int)
    beta, bptr = _flatten_to_ptr(beta, ctypes.c_int)
    shapes, sptr = _flatten_to_ptr(shapes, ctypes.c_int)
    function(
        zptr,
        ciptr,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        aptr,
        bptr,
        sptr,
    )

    return b[:, 0]


def _soc_block(zmat, ci_row, ci_col, twos_row, twos_col, norb, nelec):
    '''
    Building one state-pair SOC block in the ordering used by ``SISO``.

    args:
        zmat: np.array
            spherical SOC integrals
        ci_row, ci_col: np.array
            row and column CI vectors
        twos_row, twos_col: int
            twice the spins of the row and column states
        norb: int
            number of active orbitals
        nelec: int
            total number of active electrons
    returns:
        soc_block: np.array
            state-pair SOC matrix
    '''
    if twos_row == twos_col:
        action = _soc_action("ss", zmat, ci_row, norb, nelec, twos_row)
        reduced = np.einsum("ij,mij->m", ci_col, action)
        cg = _siso.compute_cg_coefficients(twos_row, Ms=0)
        block = np.einsum("mij,m->ij", cg, reduced)
        if twos_row:
            coefficient = (
                np.sqrt((twos_row / 2 + 1) * (twos_row + 1) / (twos_row / 2))
                / 2
            )
        else:
            coefficient = 0.0

        return coefficient * block

    if twos_row + 2 == twos_col:
        action = _soc_action("ssp", zmat, ci_row, norb, nelec, twos_row)
        reduced = np.einsum("ij,mij->m", ci_col, action)
        cg = _siso.compute_cg_coefficients(twos_row, Ms=1)
        return np.sqrt((twos_row + 3) / 2) * np.einsum(
            "mij,m->ij", cg, reduced
        )

    if twos_row == twos_col + 2:
        action = _soc_action("ssm", zmat, ci_row, norb, nelec, twos_row)
        reduced = np.einsum("ij,mij->m", ci_col, action)
        cg = _siso.compute_cg_coefficients(twos_row, Ms=-1)
        return -np.sqrt((twos_row + 1) / 2) * np.einsum(
            "mij,m->ij", cg, reduced
        )

    soc_block = np.zeros((twos_row + 1, twos_col + 1),
                         dtype=np.complex128)
    return soc_block


class SI(lib.StreamObject):
    '''
    State interaction for CAS wave functions represented in different orbital
    bases.

    args:
        mc:
            CASCI/CASSCF-like object
        modelspace: list of tuples
            model-space definition as
            ``(number_of_states, spin_multiplicity)``
        ci: list of np.array
            determinant-basis CI vectors
        mo_coeff: list of np.array
            MO coefficients associated with the CI vectors
        energies: np.array
            spin-free state energies; defaults to ``mc.e_states``
        lu_threshold: float
            pseudo-pivoting threshold
        linear_dep_threshold: float
            threshold for linear dependence in the model-state overlap

    Pair-specific transformed orbitals, CI vectors, transition RDMs, overlaps,
    and Hamiltonian elements are stored in ``pair_data[(left, right)]`` after
    calling :meth:`build`.
    '''

    _keys = {
        "ci",
        "mo_coeff",
        "energies",
        "state_twos",
        "hamiltonian",
        "overlap",
        "si_energies",
        "si_vecs",
        "pair_data",
    }

    def __init__(
        self,
        mc,
        modelspace,
        ci,
        mo_coeff,
        energies=None,
        *,
        lu_threshold=LU_THRESH,
        linear_dep_threshold=1e-9,
    ):
        self.mc = mc
        self.modelspace = tuple(modelspace)
        ordered = sorted(self.modelspace, key=lambda item: item[1])
        self.state_twos = np.asarray(
            [multiplicity - 1 for count, multiplicity, *_ in ordered
             for _ in range(count)],
            dtype=int,
        )
        self.ci = [np.asarray(x, dtype=np.float64, order="C") for x in ci]
        self.mo_coeff = [
            np.asarray(x, dtype=np.float64, order="C") for x in mo_coeff]

        nstates = len(self.state_twos)
        if len(self.ci) != nstates or len(self.mo_coeff) != nstates:
            raise ValueError(
                f"modelspace defines {nstates} states, but received "
                f"{len(self.ci)} CI vectors and "
                f"{len(self.mo_coeff)} MO sets")

        if energies is None:
            energies = mc.e_states
        self.energies = np.asarray(energies, dtype=float)
        if self.energies.shape != (nstates,):
            raise ValueError(f"Expected {nstates} state energies")

        if not isinstance(mc.ncore, (int, np.integer)):
            raise NotImplementedError(
                "Only restricted CAS wave functions are supported")

        self.nelec = int(sum(mc.nelecas))
        for i, twos in enumerate(self.state_twos):
            na = (self.nelec + twos) // 2
            nb = self.nelec - na
            expected = (
                cistring.num_strings(mc.ncas, na),
                cistring.num_strings(mc.ncas, nb),
            )
            if self.ci[i].shape != expected:
                if self.ci[i].size == np.prod(expected):
                    self.ci[i] = self.ci[i].reshape(expected)
                else:
                    raise ValueError(
                        f"CI vector {i} has shape {self.ci[i].shape}; expected "
                        f"{expected} for 2S={twos}")

        self.lu_threshold = lu_threshold
        self.linear_dep_threshold = linear_dep_threshold
        self.pair_data = {}
        self.hamiltonian = None
        self.overlap = None
        self.si_energies = None
        self.si_vecs = None

        self._sanity_checks()
        self._dump_flags()

    def _sanity_checks(self):
        '''
        Checking the input parameters and orbital dimensions.
        '''
        if isinstance(self.mc, mcpdft.MultiStateMCPDFTSolver):
            raise NotImplementedError(
                "State-specific MC-PDFT effective Hamiltonians are not yet "
                "supported by the biorthogonal interface")

        shape = self.mo_coeff[0].shape
        ao_overlap = self.mc._scf.get_ovlp()
        for i, mo in enumerate(self.mo_coeff):
            if mo.shape != shape:
                raise ValueError(
                    "All MO coefficient arrays must have equal shapes")
            metric = reduce(np.dot, (mo.T, ao_overlap, mo))
            if not np.allclose(metric, np.eye(metric.shape[0]), atol=1e-8):
                raise ValueError(f"MO coefficient set {i} is not orthonormal")

    def _dump_flags(self):
        '''
        Printing the biorthogonal SI options.
        '''
        log = logger.Logger(self.mc.stdout, self.mc.verbose)
        log.info("******** %s ********", self.__class__)
        log.info("number of independently represented states: %d",
                 len(self.ci))
        log.info("Malmqvist LU threshold: %.3e", self.lu_threshold)

    def _nelec_for_state(self, state):
        '''
        Computing the active alpha and beta electrons for a model state.
        '''
        twos = self.state_twos[state]
        na = (self.nelec + twos) // 2
        return na, self.nelec - na

    def _make_pair(self, left, right):
        '''
        Building the biorthogonal intermediates for one state pair.
        '''
        values = biorthogonalize(
            self.mo_coeff[left],
            self.mo_coeff[right],
            self.ci[left],
            self.ci[right],
            self.mc._scf.get_ovlp(),
            self.mc.ncore,
            self.mc.ncas,
            self._nelec_for_state(left),
            self._nelec_for_state(right),
            lu_threshold=self.lu_threshold,
        )
        pair = BiorthogonalPair(left, right, *values)
        self.pair_data[left, right] = pair
        return pair

    def _get_pair(self, left, right):
        '''
        Returning an existing pair or building it on demand.
        '''
        pair = self.pair_data.get((left, right))
        if pair is None:
            pair = self._make_pair(left, right)
        return pair

    def _build_scalar_pair(self, pair):
        '''
        Computing the overlap, TDMs, and scalar Hamiltonian for one state pair.
        '''
        if self.state_twos[pair.left] != self.state_twos[pair.right]:
            return

        nelec = self._nelec_for_state(pair.left)
        pair.overlap = np.vdot(pair.ci_left.ravel(), pair.ci_right.ravel())
        pair.tdm1, pair.tdm2 = direct_spin1.trans_rdm12(
            pair.ci_left,
            pair.ci_right,
            self.mc.ncas,
            nelec,
        )
        ecore, h1, h2 = _mixed_scalar_integrals(
            self.mc, pair.mo_left, pair.mo_right)
        pair.hamiltonian = (
            ecore * pair.overlap
            + np.einsum("pq,pq->", h1, pair.tdm1)
            + 0.5 * np.einsum("pqrs,pqrs->", h2, pair.tdm2))

    def build(self):
        '''
        Building the nonorthogonal scalar Hamiltonian and overlap matrices.

        returns:
            self:
                updated ``SI`` object
        '''
        nstates = len(self.ci)
        hamiltonian = np.diag(self.energies).astype(np.complex128)
        overlap = np.eye(nstates, dtype=np.complex128)
        self.pair_data = {}

        for left in range(nstates):
            for right in range(left + 1, nstates):
                if self.state_twos[left] != self.state_twos[right]:
                    continue

                pair = self._make_pair(left, right)
                self._build_scalar_pair(pair)
                hamiltonian[left, right] = pair.hamiltonian
                hamiltonian[right, left] = pair.hamiltonian.conjugate()
                overlap[left, right] = pair.overlap
                overlap[right, left] = pair.overlap.conjugate()

        self.hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        self.overlap = 0.5 * (overlap + overlap.conj().T)
        return self

    def kernel(self):
        '''
        Building and solving the generalized state-interaction eigenproblem.

        returns:
            si_energies: np.array
                state-interaction energies
            si_vecs: np.array
                state-interaction eigenvectors
        '''
        self.build()
        overlap_eigenvalues = linalg.eigvalsh(self.overlap)
        if overlap_eigenvalues[0] <= self.linear_dep_threshold:
            raise linalg.LinAlgError(
                "The model-state overlap matrix is singular or linearly "
                f"dependent; smallest eigenvalue is "
                f"{overlap_eigenvalues[0]:.3e}")

        self.si_energies, self.si_vecs = linalg.eigh(
            self.hamiltonian, self.overlap)
        return self.si_energies, self.si_vecs

    run = kernel

    def transition_rdm1(self, left, right, basis="biorthogonal"):
        '''
        Returning the scalar transition 1-RDM for a state pair.
        '''
        pair = self._get_pair(left, right)
        if pair.tdm1 is None:
            self._build_scalar_pair(pair)

        if basis == "biorthogonal":
            return pair.tdm1
        if basis == "ao":
            ncore, ncas = self.mc.ncore, self.mc.ncas
            left_active = pair.mo_left[:, ncore:ncore + ncas]
            right_active = pair.mo_right[:, ncore:ncore + ncas]
            return left_active @ pair.tdm1 @ right_active.T
        raise ValueError("basis must be 'biorthogonal' or 'ao'")

    def transition_rdm12(self, left, right):
        '''
        Returning pair-specific 1- and 2-TDMs in the biorthogonal CAS basis.
        '''
        pair = self._get_pair(left, right)
        if pair.tdm1 is None:
            self._build_scalar_pair(pair)
        return pair.tdm1, pair.tdm2


class SISO(SI):
    '''
    Spin-orbit state interaction for CAS wave functions represented in
    different orbital bases.

    The scalar state-interaction Hamiltonian and overlap are first built by
    :class:`SI`.  Pair-specific SOC integrals are then evaluated in the
    biorthonormal basis and added in the spin-projection basis.
    '''

    _keys = SI._keys | {
        "somf",
        "amf",
        "mmf",
        "soc1e",
        "soc2e",
        "ham",
    }

    def __init__(
        self,
        mc,
        modelspace,
        ci,
        mo_coeff,
        energies=None,
        *,
        somf=True,
        amf=True,
        mmf=False,
        soc1e=True,
        soc2e=True,
        ham="DKH",
        lu_threshold=LU_THRESH,
        linear_dep_threshold=1e-9,
    ):
        self.somf = somf
        self.amf = amf
        self.mmf = mmf
        self.soc1e = soc1e
        self.soc2e = soc2e
        self.ham = ham

        super().__init__(
            mc,
            modelspace,
            ci,
            mo_coeff,
            energies=energies,
            lu_threshold=lu_threshold,
            linear_dep_threshold=linear_dep_threshold,
        )
        self._sanity_checks_soc()
        self._dump_soc_flags()

    def _sanity_checks_soc(self):
        '''
        Checking the spin-orbit coupling options.
        '''
        if self.ham not in ("BP", "DKH"):
            raise ValueError("ham must be 'BP' or 'DKH'")
        if not self.somf:
            raise NotImplementedError(
                "Explicit two-electron SOC is not implemented")
        if self.mc._scf.mol.has_ecp():
            raise NotImplementedError("ECP SOC is not supported")
        if not (self.soc1e or self.soc2e):
            raise ValueError("At least one SOC contribution must be enabled")
        if self.mmf:
            raise NotImplementedError(
                "MMF SOC is ambiguous for state-specific orbitals; use AMFI")

    def _dump_soc_flags(self):
        '''
        Printing the biorthogonal SISO options.
        '''
        log = logger.Logger(self.mc.stdout, self.mc.verbose)
        log.info("SOC Hamiltonian: %s; AMFI: %s", self.ham, self.amf)

    def build(self):
        '''
        Building the scalar state interaction and adding spin-orbit coupling.

        returns:
            self:
                updated ``SISO`` object
        '''
        super().build()
        electronic_h = self.hamiltonian.copy()
        electronic_s = self.overlap.copy()
        nstates = len(self.ci)

        hso = socaddons.socintegrals(
            self.mc._scf.mol,
            somf=self.somf,
            amf=self.amf,
            mmf=self.mmf,
            soc1e=self.soc1e,
            soc2e=self.soc2e,
            ham=self.ham,
            dm=None,
        )

        offsets = np.cumsum([0] + [x + 1 for x in self.state_twos])
        dimension = offsets[-1]
        hamiltonian = np.zeros((dimension, dimension),
                               dtype=np.complex128)
        overlap = np.zeros_like(hamiltonian)

        for left in range(nstates):
            for right in range(left, nstates):
                if abs(self.state_twos[left]
                       - self.state_twos[right]) > 2:
                    continue

                pair = self._get_pair(left, right)
                pair.zsoc = _spherical_soc(
                    hso,
                    pair.mo_left,
                    pair.mo_right,
                    self.mc.ncore,
                    self.mc.ncas,
                )

                row = slice(offsets[left], offsets[left + 1])
                col = slice(offsets[right], offsets[right + 1])
                soc_block = _soc_block(
                    pair.zsoc,
                    pair.ci_left,
                    pair.ci_right,
                    self.state_twos[left],
                    self.state_twos[right],
                    self.mc.ncas,
                    self.nelec,
                )

                if left == right:
                    hamiltonian[row, col] += 0.5 * (
                        soc_block + soc_block.conj().T)
                else:
                    hamiltonian[row, col] += soc_block
                    hamiltonian[col, row] += soc_block.conj().T

        # Expand the spin-free electronic H and S over M_S components.
        for left in range(nstates):
            row = slice(offsets[left], offsets[left + 1])
            for right in range(nstates):
                if self.state_twos[left] != self.state_twos[right]:
                    continue

                col = slice(offsets[right], offsets[right + 1])
                identity = np.eye(self.state_twos[left] + 1)
                hamiltonian[row, col] += electronic_h[left, right] * identity
                overlap[row, col] = electronic_s[left, right] * identity

        self.electronic_hamiltonian = electronic_h
        self.electronic_overlap = electronic_s
        self.hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        self.overlap = 0.5 * (overlap + overlap.conj().T)
        return self
