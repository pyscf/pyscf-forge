from functools import reduce

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

    # Only possible because the left and right core orbitals are biorthonormal.
    dm_core = 2.0 * left_core @ right_core.conj().T
    hcore = mf.get_hcore(mol)
    if ncore:
        vj, vk = mc.get_jk(mol, dm_core, hermi=0)
        veff = vj - 0.5 * vk
    else:
        veff = np.zeros_like(hcore)

    ecore = mol.energy_nuc()
    ecore += np.einsum("ij,ij->", hcore + 0.5 * veff, dm_core)
    h1 = reduce(np.dot, (left_active.conj().T, hcore + veff, right_active))

    active_orbitals = (
        left_active, right_active, left_active, right_active)
    with_df = getattr(mc, "with_df", None)
    if with_df:
        h2 = with_df.ao2mo(active_orbitals, compact=False)
    else:
        eri_source = getattr(mf, "_eri", None)
        if eri_source is None:
            eri_source = mol
        h2 = ao2mo.general(
            eri_source, active_orbitals, compact=False)
    h2 = h2.reshape((ncas,) * 4)

    return ecore, h1, h2

def _spherical_soc_integrals(hso, mo_left, mo_right, ncore, ncas):
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
    h1 = np.asarray([reduce(np.dot, (left.conj().T, x, right)) for x in hso])
    zsoc = np.empty((3, ncas, ncas), dtype=np.complex128)
    zsoc[0] = (h1[0] - 1j * h1[1]) / np.sqrt(2.0)
    zsoc[1] = h1[2]
    zsoc[2] = -(h1[0] + 1j * h1[1]) / np.sqrt(2.0)
    return zsoc

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
        contract = _siso.contract_same_spin
    elif mode == "ssp":
        # Ket has S; output determinant space has S+1.
        alpha_high = nalpha + 1
        beta_high = nbeta - 1
        alpha = cistring.gen_des_str_index(orbitals, alpha_high)
        beta = cistring.gen_cre_str_index(orbitals, beta_high)
        contract = _siso.contract_spin_plus
    elif mode == "ssm":
        # Ket has S; output determinant space has S-1.
        alpha_low = nalpha - 1
        beta_low = nbeta + 1
        alpha = cistring.gen_cre_str_index(orbitals, alpha_low)
        beta = cistring.gen_des_str_index(orbitals, beta_low)
        contract = _siso.contract_spin_minus
    else:
        raise ValueError(f"Unknown SOC action mode {mode!r}")

    ciket = np.asarray(ciket, dtype=np.complex128)[None]
    return contract(zmat, ciket, (alpha, beta))[:, 0]

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
    """Nonorthogonal state interaction between state-specific CAS wavefunctions.

    Each model state is represented by its own orthonormal MO coefficients and
    determinant-basis CI vector.  For each pair of states with the same spin,
    the occupied CAS spaces are biorthogonalized and the CI vectors are
    counter-transformed before scalar overlap and Hamiltonian matrix elements
    are evaluated.

    The implementation currently supports real, restricted CAS wavefunctions
    that use the same AO basis, inactive-space size, active-space size, and
    total number of active electrons.  Entries in ``ci``, ``mo_coeff``, and
    ``energies`` must occur in the contiguous state blocks described by the
    input ``modelspace``.  These blocks are grouped by increasing spin
    multiplicity internally, together with all three state-dependent inputs.

    Args:
        mc: CASCI/CASSCF-like object defining the molecular integrals and CAS.
        modelspace: Non-empty sequence of ``(nroots, spin_multiplicity)`` or
            ``(nroots, spin_multiplicity, wfnsym)`` entries.
        ci: Normalized determinant-basis CI vector for every model state.
        mo_coeff: Orthonormal MO coefficient array associated with every CI
            vector.
        energies: Spin-free total energy of every model state.  If omitted,
            ``mc.e_states`` is used, with ``mc.e_tot`` as a CASCI fallback.
        lu_threshold: Minimum squared pivot-tail norm used by the simultaneous
            LU decomposition.
        linear_dep_threshold: Smallest allowed eigenvalue of the model-state
            overlap matrix.

    Attributes:
        pair_data: Pair-specific biorthogonal orbitals, counter-transformed CI
            vectors, transition RDMs, overlaps, and Hamiltonian elements.
        hamiltonian: Scalar model-space Hamiltonian after :meth:`build`.
        overlap: Model-state overlap matrix after :meth:`build`.
        si_energies: Generalized eigenvalues after :meth:`kernel`.
        si_vecs: Generalized eigenvectors after :meth:`kernel`.

    Note:
        :meth:`build` evaluates off-diagonal same-spin pairs.  Diagonal or
        reverse-order pair data are constructed on demand by the transition-RDM
        accessors.
    """

    _keys = {
        "modelspace",
        "ci",
        "mo_coeff",
        "energies",
        "state_twos",
        "lu_threshold",
        "linear_dep_threshold",
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
        input_modelspace = tuple(modelspace)
        validated_modelspace = socaddons._validate_modelspace(
            input_modelspace,
            mol=mc._scf.mol,
            ncas=mc.ncas,
            nelecas=mc.nelecas,
        )

        # The explicit state data follow the input model-space blocks.  Apply
        # the same stable spin ordering used by _validate_modelspace to avoid
        # assigning a CI vector to the wrong spin when the blocks are unsorted.
        block_offsets = np.cumsum(
            [0] + [int(entry[0]) for entry in input_modelspace])
        block_order = sorted(
            range(len(input_modelspace)),
            key=lambda index: input_modelspace[index][1],
        )
        state_order = np.concatenate([
            np.arange(block_offsets[index], block_offsets[index + 1])
            for index in block_order
        ]).astype(int)

        self.modelspace = tuple(validated_modelspace)
        self.state_twos = np.asarray(
            [multiplicity - 1
             for count, multiplicity, _ in self.modelspace
             for _ in range(count)],
            dtype=int,
        )

        nstates = len(self.state_twos)
        ci = list(ci)
        mo_coeff = list(mo_coeff)
        if len(ci) != nstates or len(mo_coeff) != nstates:
            raise ValueError(
                f"modelspace defines {nstates} states, but received "
                f"{len(ci)} CI vectors and {len(mo_coeff)} MO sets")

        ci = [ci[index] for index in state_order]
        mo_coeff = [mo_coeff[index] for index in state_order]
        for label, arrays in (("CI vectors", ci), ("MO coefficients", mo_coeff)):
            if any(np.iscomplexobj(array)
                   and np.any(np.abs(np.imag(array)) > 1e-12)
                   for array in arrays):
                raise NotImplementedError(
                    f"Complex {label} are not supported by biorthogonal SI")
        self.ci = [np.asarray(x, dtype=np.float64, order="C") for x in ci]
        self.mo_coeff = [
            np.asarray(x, dtype=np.float64, order="C") for x in mo_coeff]

        if energies is None:
            energies = getattr(mc, "e_states", None)
            if energies is None:
                energies = getattr(mc, "e_tot", None)
        if energies is None:
            raise ValueError(
                "energies must be supplied when mc has no e_states or e_tot")
        self.energies = np.asarray(energies, dtype=float)
        if self.energies.shape != (nstates,):
            raise ValueError(f"Expected {nstates} state energies")
        self.energies = self.energies[state_order]
        if not np.all(np.isfinite(self.energies)):
            raise ValueError("State energies must be finite")

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
        """Validate the wavefunction representation and numerical thresholds."""
        if isinstance(self.mc, mcpdft.MultiStateMCPDFTSolver):
            raise NotImplementedError(
                "Multi state MC-PDFT effective Hamiltonians are not yet "
                "supported by the biorthogonal interface")

        if self.lu_threshold <= 0:
            raise ValueError("lu_threshold must be positive")
        if self.linear_dep_threshold < 0:
            raise ValueError("linear_dep_threshold cannot be negative")

        shape = self.mo_coeff[0].shape
        if len(shape) != 2 or shape[1] < self.mc.ncore + self.mc.ncas:
            raise ValueError(
                "MO coefficient arrays must contain the full inactive and "
                "active orbital spaces")
        ao_overlap = self.mc._scf.get_ovlp()
        if ao_overlap.shape != (shape[0], shape[0]):
            raise ValueError(
                "AO overlap dimensions are inconsistent with MO coefficients")
        for i, mo in enumerate(self.mo_coeff):
            if mo.shape != shape:
                raise ValueError(
                    "All MO coefficient arrays must have equal shapes")
            metric = reduce(np.dot, (mo.T, ao_overlap, mo))
            if not np.allclose(
                    metric, np.eye(metric.shape[0]), atol=1e-8, rtol=0.0):
                raise ValueError(f"MO coefficient set {i} is not orthonormal")
            norm = np.vdot(self.ci[i].ravel(), self.ci[i].ravel()).real
            if not np.isclose(norm, 1.0, atol=1e-8, rtol=0.0):
                raise ValueError(
                    f"CI vector {i} is not normalized; squared norm is "
                    f"{norm:.12g}")

    def _dump_flags(self):
        '''
        Printing the biorthogonal SI options.
        '''
        log = logger.Logger(self.mc.stdout, self.mc.verbose)
        log.info("******** %s ********", self.__class__)
        log.info("number of independently represented states: %d",
                 len(self.ci))
        log.info("model space: %s", self.modelspace)
        log.info("LU threshold: %.3e", self.lu_threshold)
        log.info("linear dependency threshold: %.3e",
                 self.linear_dep_threshold)

    def _nelec_for_state(self, state):
        """Return active alpha and beta electron counts for one model state."""
        twos = self.state_twos[state]
        na = (self.nelec + twos) // 2
        return na, self.nelec - na

    def _make_pair(self, left, right):
        """Construct and cache biorthogonal intermediates for a state pair."""
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
        """Return a cached state pair, constructing it when necessary."""
        pair = self.pair_data.get((left, right))
        if pair is None:
            pair = self._make_pair(left, right)
        return pair

    def _build_scalar_pair(self, pair):
        """Populate scalar overlap, active-space TDMs, and Hamiltonian."""
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
        """Build the nonorthogonal scalar Hamiltonian and overlap matrices.

        The supplied state energies form the diagonal of the Hamiltonian.
        Off-diagonal elements are evaluated explicitly for same-spin pairs;
        scalar matrix elements between different total spins are zero.

        Returns:
            SI: This object with ``hamiltonian``, ``overlap``, and
            ``pair_data`` updated.
        """
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

        # Suppress insignificant numerical anti-Hermitian components.
        self.hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        self.overlap = 0.5 * (overlap + overlap.conj().T)
        return self

    def kernel(self):
        """Build and solve the generalized state-interaction eigenproblem.

        Returns:
            tuple: ``(si_energies, si_vecs)``, containing the scalar SI
            eigenvalues and overlap-normalized generalized eigenvectors.

        Raises:
            scipy.linalg.LinAlgError: If the model-state overlap is singular or
            has an eigenvalue no larger than ``linear_dep_threshold``.
        """
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

    def transition_rdm1(self, left, right, basis="biorthogonal"):
        """Return the spin-summed transition one-particle density matrix.

        Args:
            left: Bra-state index.
            right: Ket-state index.
            basis: ``"biorthogonal"`` returns the active-space transition RDM
                in the pair-specific biorthogonal orbitals.  ``"ao"`` returns
                the complete AO transition density, including the inactive
                doubly occupied orbitals.

        Returns:
            numpy.ndarray: Pair-specific transition one-particle density.

        Note:
            The biorthogonal result contains only active indices.  The AO
            result satisfies ``einsum('uv,uv', S_AO, dm1) = N * overlap``.
        """
        if self.state_twos[left] != self.state_twos[right]:
            raise ValueError(
                "Scalar transition RDMs require states with the same spin")
        pair = self._get_pair(left, right)
        if pair.tdm1 is None:
            self._build_scalar_pair(pair)

        if basis == "biorthogonal":
            return pair.tdm1
        if basis == "ao":
            ncore, ncas = self.mc.ncore, self.mc.ncas
            left_core = pair.mo_left[:, :ncore]
            right_core = pair.mo_right[:, :ncore]
            left_active = pair.mo_left[:, ncore:ncore + ncas]
            right_active = pair.mo_right[:, ncore:ncore + ncas]
            dm1 = left_active @ pair.tdm1 @ right_active.T
            if ncore:
                dm1 += (2.0 * pair.overlap
                        * left_core @ right_core.T)
            return dm1
        raise ValueError("basis must be 'biorthogonal' or 'ao'")

    def transition_rdm12(self, left, right):
        """Return active-space 1- and 2-TDMs in the biorthogonal basis.

        Args:
            left: Bra-state index.
            right: Ket-state index.

        Returns:
            tuple: Pair-specific spin-summed active-space ``(tdm1, tdm2)``.
        """
        if self.state_twos[left] != self.state_twos[right]:
            raise ValueError(
                "Scalar transition RDMs require states with the same spin")
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
        self._initialize()

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
        log.info("SOMF: %s", self.somf)
        log.info("AMFI: %s", self.amf)
        log.info("MMFI: %s", self.mmf)
        log.info("one-electron SOC: %s", self.soc1e)
        log.info("two-electron SOC: %s", self.soc2e)
        log.info("SOC Hamiltonian: %s", self.ham)
        log.info("speed of light: %.8f a.u.", lib.param.LIGHT_SPEED)

    def _initialize(self):
        '''
        Report the initial spin-free model-space energies.
        '''
        log = logger.Logger(self.mc.stdout, self.mc.verbose)
        order = np.argsort(self.energies)
        energies = self.energies[order]
        state_twos = self.state_twos[order]

        log.note(" ")
        log.info("******** %s ********", "Spin Orbit Free Energetics")
        for state, (energy, twos) in enumerate(zip(energies, state_twos)):
            spin = twos / 2.0
            spin_square = spin * (spin + 1.0)
            log.note(" State %d Total Energy = %.10f S^2 = %.2f",
                     state, energy, spin_square)

        log.note(" ")
        log.info("******** %s ********",
                 "Relative Spin Orbit Free Energetics")
        log.note("State          Relative Energy(au)   Relative Energy(eV)   "
                 "Relative Energy(cm^-1)")
        reference = energies[0]
        for state, energy in enumerate(energies):
            relative_energy = energy - reference
            log.note(" {:<10} {:>20.9f} {:>20.5f} {:>20.5f}".format(
                state,
                relative_energy,
                _siso.au2ev * relative_energy,
                _siso.au2cminv * relative_energy))

    def kernel(self):
        '''
        Build and solve the biorthogonal SISO eigenvalue problem.
        '''
        self.si_energies, self.si_vecs = super().kernel()
        self._finalize()
        return self.si_energies, self.si_vecs

    def _finalize(self):
        '''
        Report the final spin-orbit coupled energies.
        '''
        log = logger.Logger(self.mc.stdout, self.mc.verbose)
        log.note(" ")
        log.info("******** %s ********", "Spin Orbit Coupling Energetics")
        for state, energy in enumerate(self.si_energies):
            log.note(" SO-CASSI State %d Total Energy = %.10f ",
                     state, energy)

        log.note(" ")
        log.info("******** %s ********",
                 "Relative Spin Orbit Coupling Energetics")
        log.note("SO State       Relative Energy(au)   Relative Energy(eV)   "
                 "Relative Energy(cm^-1)")
        reference = self.si_energies[0]
        for state, energy in enumerate(self.si_energies):
            relative_energy = energy - reference
            log.note(" {:<10} {:>20.9f} {:>20.5f} {:>20.5f}".format(
                state,
                relative_energy,
                _siso.au2ev * relative_energy,
                _siso.au2cminv * relative_energy))

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
                pair.zsoc = _spherical_soc_integrals(
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
