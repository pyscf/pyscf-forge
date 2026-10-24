# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Naoki Kitamura
#

"""
Diagonalization-free Hartree-Fock density construction with Chebyshev filters.

This module provides an experimental SCF driver that approximates the
finite-temperature occupation projector by a Chebyshev polynomial expansion.
The density matrix is built from filtered basis vectors in an orthogonalized AO
representation, avoiding an explicit Fock diagonalization in each SCF cycle.
"""


import numpy
import numpy.polynomial.chebyshev as C
from pyscf.lo.orth import lowdin
from scipy.optimize import bisect
from scipy.sparse import block_diag, coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from pyscf import __config__
from pyscf.lib import logger
from pyscf.scf import diis, hf


def block_inv_sqrt(s, thresh=1e-7):
    """
    Compute a block-local inverse square root of an overlap matrix.

    Matrix elements whose absolute values are below ``thresh`` are treated as
    disconnected when identifying AO blocks.  Each connected block is
    diagonalized independently and transformed back to the original AO order.

    Parameters
    ----------
    s : ndarray
        Symmetric positive-definite AO overlap matrix.
    thresh : float, optional
        Connectivity threshold used to identify independent overlap blocks.

    Returns
    -------
    ndarray
        Approximate ``s**(-1/2)`` assembled from independently transformed
        overlap blocks.
    """
    s_sparse = csr_matrix(numpy.abs(s) > thresh)
    n_components, labels = connected_components(s_sparse)
    block_indices = [list(numpy.where(labels == i)[0]) for i in range(n_components)]

    s_inv_sqrt = numpy.zeros_like(s)

    for idx in block_indices:
        s_block = s[numpy.ix_(idx, idx)]
        X = lowdin(s_block)

        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                s_inv_sqrt[ii, jj] = X[i, j]

    return s_inv_sqrt


def inv_sqrt(s, thresh=1e-7, full_inv=False):
    """
    Compute an inverse square root of a symmetric positive-definite matrix.

    Parameters
    ----------
    s : ndarray
        Matrix to transform.
    thresh : float, optional
        Threshold passed to :func:`block_inv_sqrt` when ``full_inv`` is false.
    full_inv : bool, optional
        If true, diagonalize the full matrix.  If false, use the block-local
        approximation defined by ``thresh``.

    Returns
    -------
    ndarray
        The inverse square-root transformation matrix.
    """
    if full_inv:
        evals, evecs = numpy.linalg.eigh(s)
        X = evecs @ numpy.diag(evals**-0.5) @ evecs.conj().T
        return X
    else:
        return block_inv_sqrt(s, thresh=thresh)


def to_csr_matrix(matrix, threshold=1e-5):
    """
    Convert a dense array to CSR format after dropping small elements.

    Parameters
    ----------
    matrix : array_like
        Dense matrix to sparsify.
    threshold : float, optional
        Entries with absolute value below this value are set to zero.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse representation of the thresholded matrix.
    """
    matrix = numpy.where(numpy.abs(matrix) < threshold, 0.0, matrix)
    return csr_matrix(matrix)


class ChebyshevSCF(hf.SCF):
    """
    Restricted Hartree-Fock SCF driver using Chebyshev density filtering.

    The class follows the :class:`pyscf.scf.hf.SCF` interface but replaces the
    conventional diagonalization-based density update with a Chebyshev
    approximation to the Fermi-Dirac occupation function.  The expansion degree
    is controlled by ``chebyshev_degree`` and the filter sharpness by ``beta``,
    held fixed throughout the SCF iteration. We have not been able to strictly avoid diagonalization.
    """

    def __init__(self, mol):
        """
        Initialize the Chebyshev SCF object for a PySCF molecule.

        Parameters
        ----------
        mol : pyscf.gto.Mole
            Molecular system defining the AO basis, charge, spin, and
            electron count.
        """
        self.mol = mol
        hf.SCF.__init__(self, mol)
        self.chebyshev_degree = 30
        self.beta = 50
        self.degree_safety_factor = 2.0
        self.rng_seed = 0
        self.chebyshev_moments = None
        self.ovlp_inv = to_csr_matrix(self.get_ovlp_inv_sqrt(), threshold=1e-5)
        self.a = None
        self.b = None
        self.g = 1

    def build(self, mol=None):
        """
        Build the underlying PySCF SCF object and register Chebyshev options.

        Parameters
        ----------
        mol : pyscf.gto.Mole, optional
            Molecule to associate with this SCF object.

        Returns
        -------
        ChebyshevSCF
            The built SCF object.
        """
        hf.SCF.build(self, mol)
        self._keys = self._keys.union(
            [
                "chebyshev_degree",
                "chebyshev_filter_interval",
                "degree_safety_factor",
                "rng_seed",
                "chebyshev_moments",
                "ovlp_inv",
                "a",
                "b",
                "g",
            ]
        )
        return self

    def _normalize_fock(self, fock_ortho, a=None, b=None):
        if a is None:
            a = self.a
        if b is None:
            b = self.b

        identity = csr_matrix(
            (
                numpy.ones(fock_ortho.shape[0]),
                (numpy.arange(fock_ortho.shape[0]), numpy.arange(fock_ortho.shape[0])),
            ),
            shape=(fock_ortho.shape[0], fock_ortho.shape[0]),
        )
        normalized_fock = (fock_ortho - b * identity) / a
        normalized_fock = to_csr_matrix(normalized_fock.toarray(), threshold=1e-5)
        return normalized_fock

    def get_ovlp_inv_sqrt(self, thresh=1e-7, full_inv=False):
        """
        Return the inverse square-root transform of the AO overlap matrix.

        Parameters
        ----------
        thresh : float, optional
            Connectivity threshold for the block-local inverse square root.
        full_inv : bool, optional
            Use a full overlap diagonalization instead of block-local blocks.

        Returns
        -------
        ndarray
            AO overlap inverse square-root matrix.
        """
        ovlp = self.get_ovlp()
        ovlp_inv_sqrt = inv_sqrt(ovlp, thresh=thresh, full_inv=full_inv)
        return ovlp_inv_sqrt

    def get_chebyshev_moments(
        self, normalized_fock, degree=None, vec_threshold=None, nmo=None
    ):
        if degree is None:
            degree = self.chebyshev_degree
        if nmo is None:
            nmo = normalized_fock.shape[0]
        if vec_threshold is None:
            vec_threshold = getattr(self, "vec_threshold", 1e-6)
        # T0 v_
        normalized_fock = to_csr_matrix(normalized_fock.toarray(), threshold=1e-5)
        self.chebyshev_moments = numpy.zeros(self.chebyshev_degree + 1)
        for i in range(nmo):
            v = coo_matrix(([1.0], ([i], [0])), shape=(nmo, 1)).tocsr()
            v0 = v.copy()
            # T1 v
            v1 = normalized_fock @ v0
            v1 = csr_matrix(v1.reshape(-1, 1))
            self.chebyshev_moments[0] += v @ v0
            self.chebyshev_moments[1] += v @ v1
            if vec_threshold:
                v1 = to_csr_matrix(v1.toarray(), threshold=vec_threshold)
            for k in range(2, degree + 1):
                v2 = 2 * normalized_fock @ v1 - v0
                if vec_threshold:
                    v2 = to_csr_matrix(v2.toarray(), threshold=vec_threshold)
                v0, v1 = v1, v2
                self.chebyshev_moments[k] += v @ v2
        return self.chebyshev_moments

    def _chebyshev_poly_vec(
        self, v, normalized_fock, mu, degree, beta, a, vec_threshold=None
    ):
        """
        Apply the Chebyshev occupation polynomial to a basis vector.

        Parameters
        ----------
        v : scipy.sparse.spmatrix
            Column vector to filter.
        normalized_fock : scipy.sparse.spmatrix
            Normalized Fock matrix in the orthogonal AO representation.
        mu : float
            Chemical potential in the normalized spectral interval.
        degree : int
            Chebyshev expansion degree.
        beta : float
            Inverse-temperature parameter of the Fermi-Dirac filter.
        a : float
            Energy scaling factor used in the spectral normalization.
        vec_threshold : float, optional
            Elements of the recursion vectors ``v0``/``v1``/``v2`` with
            magnitude below this value are dropped (set to zero) after
            every Chebyshev step. Defaults to ``self.vec_threshold``.

        Returns
        -------
        scipy.sparse.spmatrix
            Filtered vector contribution to the density matrix.
        """
        if vec_threshold is None:
            vec_threshold = getattr(self, "vec_threshold", 1e-6)

        coeffs = self._calc_coeff(degree, beta=beta, mu=mu, a=a)
        # T0 v_
        v0 = v.copy()
        # T1 v
        v1 = normalized_fock @ v0
        v1 = csr_matrix(v1.reshape(-1, 1))
        if vec_threshold:
            v1 = to_csr_matrix(v1.toarray(), threshold=vec_threshold)
        y = coeffs[0] * v0 + coeffs[1] * v1
        for k in range(2, degree + 1):
            v2 = 2 * normalized_fock @ v1 - v0
            if vec_threshold:
                v2 = to_csr_matrix(v2.toarray(), threshold=vec_threshold)
            y += coeffs[k] * v2
            v0, v1 = v1, v2
        return y

    def _calc_coeff(self, n, beta=100.0, mu=0.0, a=1.0):
        """
        Compute Chebyshev coefficients for the Fermi-Dirac occupation filter.

        Parameters
        ----------
        n : int
            Highest Chebyshev polynomial order.
        beta : float, optional
            Inverse-temperature parameter.
        mu : float, optional
            Chemical potential in the normalized spectral interval.
        a : float, optional
            Energy scaling factor for the physical Fock spectrum.

        Returns
        -------
        ndarray
            Coefficients from order zero through ``n``.
        """
        beta_tilde = beta * a

        def occupation(x):
            return 1.0 / (1.0 + numpy.exp(beta_tilde * (x - mu)))

        # f = 1.0 / (1.0 + numpy.exp(beta_tilde * (numpy.cos(theta) - mu)))
        # coeffs = numpy.fft.fft(numpy.r_[f, f[-2:0:-1]]).real[: n + 1]
        # coeffs *= 2.0 / (N - 1)
        # coeffs[0] *= 0.5
        coeffs = C.chebinterpolate(occupation, n)
        return coeffs

    def _get_chebyshev_vec(self, coeffs):
        assert len(coeffs) == len(self.chebyshev_sequences)
        vec = self.chebyshev_sequences[0] * coeffs[0]
        for v in self.chebyshev_sequences[1:]:
            k = self.chebyshev_sequences.index(v)
            vec += v * coeffs[k]
        return vec

    def _orthogonalize_fock(self, fock):
        """
        Orthogonalize the AO-basis Fock matrix using the inverse square-root
        of the AO overlap matrix.

        Parameters
        ----------
        fock : ndarray
            AO-basis Fock matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Orthogonalized AO-basis Fock matrix.
        """
        X = self.ovlp_inv
        fock_ortho = X @ fock @ X.T
        fock_ortho = to_csr_matrix(fock_ortho.toarray(), threshold=1e-5)
        return fock_ortho

    def _spectral_bounds(self, fock_ortho):
        """
        Estimate the orthogonalized-Fock spectral bounds ``(a, b, emin, emax)``
        used to normalize the Chebyshev expansion interval.

        Factored out of :meth:`make_rdm1_chebyshev` so the same estimate can
        be obtained once, up front, to auto-size ``chebyshev_degree`` before
        the SCF loop starts (see :meth:`_auto_degree`), without duplicating
        the overlap-orthogonalization and eigenvalue-bound logic.

        Parameters
        ----------
        fock_ortho : ndarray
            Orthogonalized AO-basis Fock matrix.

        Returns
        -------
        tuple
            ``(a, b, emin, emax)`` where ``a`` and ``b`` are the spectral bounds,
            and ``emin`` and ``emax`` are the estimated minimum and maximum eigenvalues.
        """

        nmo = fock_ortho.shape[0]
        diag = numpy.asarray(fock_ortho.diagonal())
        precond = diag
        # Use a locally-seeded RandomState (rather than the unseeded global
        # numpy.random used previously) so that chebyshev_degree, the
        # filtered density matrix, and the final energy are exactly
        # reproducible for a given (mol, beta, rng_seed), not just
        # statistically similar run to run.
        rng = numpy.random.RandomState(getattr(self, "rng_seed", 0))
        random_vec = rng.rand(nmo)
        random_vec /= numpy.linalg.norm(random_vec)
        from pyscf.lib import davidson

        def fock_ortho_mv(v):
            return fock_ortho @ v

        emin, _ = davidson(fock_ortho_mv, random_vec, precond=precond)
        emin -= 0.1

        abs_row_sum = numpy.asarray(numpy.abs(fock_ortho).sum(axis=1)).ravel()
        row_sum = abs_row_sum - numpy.abs(diag)
        emax = numpy.max(diag + row_sum)

        delta = 0.1
        a = delta + (emax - emin) / 2.0
        b = (emax + emin) / 2.0
        return a, b, emin, emax

    def make_rdm1_chebyshev(
        self, normalized_fock, mu, nmo=None, degree=None, beta=None, a=None, b=None
    ):
        """
        Construct an AO-basis one-particle density matrix by Chebyshev filtering.

        Parameters
        ----------
        normalized_fock : ndarray
            Normalized AO-basis Fock matrix.
        mu : float
            Chemical potential in the normalized spectral interval.
        nmo : int, optional
            Number of filtered columns.  Defaults to the Fock dimension.
        degree : int, optional
            Chebyshev expansion degree.  Defaults to ``self.chebyshev_degree``.
        beta : float, optional
            Inverse-temperature parameter.  Defaults to ``self.beta``.

        Returns
        -------
        ndarray
            Approximate AO-basis density matrix.
        """
        if degree is None:
            degree = self.chebyshev_degree
        if nmo is None:
            nmo = normalized_fock.shape[0]
        if beta is None:
            beta = self.beta
        if a is None:
            a = self.a
        if b is None:
            b = self.b

        X = self.ovlp_inv

        normalized_fock = to_csr_matrix(normalized_fock.toarray(), threshold=1e-5)

        dm = numpy.zeros((normalized_fock.shape[0], normalized_fock.shape[0]))

        identity = csr_matrix(
            (
                numpy.ones(normalized_fock.shape[0]),
                (
                    numpy.arange(normalized_fock.shape[0]),
                    numpy.arange(normalized_fock.shape[0]),
                ),
            ),
            shape=(normalized_fock.shape[0], normalized_fock.shape[0]),
        )

        # for i in range(nmo):
        #     v = coo_matrix(([1.0], ([i], [0])), shape=(nmo, 1)).tocsr()
        #     y = self._chebyshev_poly_vec(v, normalized_fock, mu, degree, beta, a)
        #     dm[:, i] = y.toarray().flatten()
        y = self._chebyshev_poly_vec(identity, normalized_fock, mu, degree, beta, a)
        dm = y.toarray()

        dm = to_csr_matrix(dm, threshold=1e-5)
        dm_ao = X @ dm @ X.T
        return dm_ao.toarray()

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        """
        Return the AO-basis one-particle density matrix from the last
        completed Chebyshev SCF cycle.

        Overrides :meth:`pyscf.scf.hf.SCF.make_rdm1`, which expects
        ``mo_coeff``/``mo_occ`` from an explicit diagonalization.  This
        class never occupies orbitals directly, so the converged density
        matrix built by :meth:`make_rdm1_chebyshev` is cached on
        ``self._dm1`` during :meth:`kernel` and returned here instead.

        Parameters
        ----------
        mo_coeff, mo_occ : ignored
            Accepted only for interface compatibility with
            :meth:`pyscf.scf.hf.SCF.make_rdm1`; this driver does not use
            them to build the density matrix.

        Returns
        -------
        ndarray
            AO-basis one-particle density matrix.
        """
        if getattr(self, "_dm1", None) is None:
            raise RuntimeError(
                "ChebyshevSCF.make_rdm1() called before kernel() has run; "
                "no density matrix is available yet."
            )
        return self._dm1

    def _diff_ne(self, nelec, mu, moments):
        """
        Return the electron-number residual for a trial chemical potential.

        Parameters
        ----------
        nelec : float
            Target number of electrons.
        mu : float
            Trial chemical potential.

        Returns
        -------
        float
            Difference between filtered electron count and ``nelec``.
        """
        coeffs = self._calc_coeff(
            self.chebyshev_degree, beta=self.beta, mu=mu, a=self.a
        )
        ne = self.g * coeffs @ moments
        return ne - nelec

    def bisect_search_mu(self, nelectrons, moments=None):
        """
        Find the chemical potential that gives the target electron count.

        Parameters
        ----------
        nelectrons : float
            Target number of electrons.
        moments : ndarray or None
            Chebyshev moments.  If None, use ``self.chebyshev_moments``.

        Returns
        -------
        float
            Chemical potential found by bisection over ``[-1, 1]``.
        """
        mu_lo = -1.0
        mu_hi = 1.0
        if moments is None:
            moments = self.chebyshev_moments
        mu = bisect(
            lambda mu: self._diff_ne(nelectrons, mu, moments), mu_lo, mu_hi, xtol=1e-14
        )
        return mu

    def kernel(
        self, conv_dm=1e-3, conv_e=1e-6, max_cycle=100, mixing_beta=0.3, dm1=None
    ):
        """
        Run the Chebyshev-filtered SCF iteration.

        Parameters
        ----------
        conv_dm : float, optional
            Convergence threshold for the density-matrix change.
        conv_e : float, optional
            Convergence threshold for the total-energy change.
        max_cycle : int, optional
            Maximum number of SCF iterations.
        mixing_beta : float, optional
            Reserved mixing parameter for API experimentation.
        dm1 : ndarray, optional
            Initial AO-basis density matrix.

        Returns
        -------
        float
            Final total SCF energy.
        """
        self.build(self.mol)

        dm1 = self.get_init_guess(self.mol, dm1=dm1)
        fock = self.get_fock(dm=dm1)
        ovlp = self.get_ovlp()

        e_save = 1e9
        diis_obj = diis.ADIIS(self)
        fock_diis = fock

        converged = False
        dm_new = dm1
        for cycle in range(max_cycle):
            fock = fock_diis
            fock_ortho = self._orthogonalize_fock(fock)
            self.a, self.b, _, _ = self._spectral_bounds(fock_ortho)
            normalized_fock = self._normalize_fock(fock_ortho)
            moments = self.get_chebyshev_moments(normalized_fock)
            self.mu = self.bisect_search_mu(self.mol.nelectron, moments=moments)
            dm_new = self.make_rdm1_chebyshev(normalized_fock, self.mu)
            h1e = self.get_hcore()
            veff = self.get_veff(self.mol, dm_new)
            F = h1e + veff
            fock_diis = diis_obj.update(ovlp, dm_new, F, self, h1e, veff)
            ne = numpy.trace(dm_new @ ovlp)
            diff = numpy.linalg.norm(dm_new - dm1)
            e = self.energy_elec(dm_new, h1e, veff)[0] + self.mol.energy_nuc()
            logger.info(
                self,
                "Cycle %d  mu = %.12f  diff = %.8f e = %.8f  ne = %.8f",
                cycle,
                self.mu,
                diff,
                e,
                ne,
            )
            if diff < conv_dm or abs(e - e_save) < conv_e:
                converged = True
                break
            dm1 = dm_new
            e_save = e

        if not converged:
            logger.warn(
                self,
                "ChebyshevSCF did not converge after %d cycles "
                "(final |ddm|/|dE| still above conv_dm=%.3g / "
                "conv_e=%.3g). Try increasing max_cycle, "
                "chebyshev_degree, or beta.",
                max_cycle,
                conv_dm,
                conv_e,
            )

        e_tot = self.energy_elec(dm_new, h1e, veff)[0] + self.mol.energy_nuc()

        self._dm1 = dm_new
        self.e_tot = e_tot
        self.converged = converged

        mo_energy, mo_coeff = self.eig(fock_diis, ovlp)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ

        logger.info(self, "ChebyshevSCF converged = %s", self.converged)
        logger.info(self, "Converged SCF energy = %.12f", e_tot)
        return e_tot


if __name__ == "__main__":
    from pyscf import gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=4)
    mf = ChebyshevSCF(mol)
    mf.chebyshev_degree = 10
    mf.beta = 40.0
    mf.kernel()

    print("ChebyshevSCF Energy:", mf.e_tot)
