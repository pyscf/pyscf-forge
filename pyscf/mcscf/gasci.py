#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
# Author: Yi Deng <yideng@uchicago.edu>
#

"""PySCF-style generalized active-space configuration interaction."""

import hashlib
from functools import reduce

import numpy

from pyscf import __config__
from pyscf import fci
from pyscf import gto
from pyscf import lib
from pyscf.fci import addons as fci_addons
from pyscf.lib import logger
from pyscf.mcscf import addons
from pyscf.mcscf import casci

from pyscf.mcscf import addons_gas
from pyscf.mcscf import fci_gas


WITH_META_LOWDIN = getattr(
    __config__, "mcscf_analyze_with_meta_lowdin", True)
LARGE_CI_TOL = getattr(__config__, "mcscf_analyze_large_ci_tol", 0.1)
MO_ORTH_WARN_TOL = getattr(
    __config__, "mcscf_gasci_mo_orth_warn_tol", 1e-7)
MO_ORTH_ERROR_TOL = getattr(
    __config__, "mcscf_gasci_mo_orth_error_tol", 1e-5)


h1e_for_gas = casci.h1e_for_cas


def kernel(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE):
    """Run the fixed-orbital GASCI solver."""

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci0 is None:
        ci0 = mc.ci

    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug("Start GASCI")

    eri_gas = mc.get_h2eff(mo_coeff)
    t1 = log.timer("integral transformation to GAS space", *t0)

    h1eff, energy_core = mc.get_h1eff(mo_coeff)
    log.debug("core energy = %.15g", energy_core)
    t1 = log.timer("effective h1e in GAS space", *t1)

    if h1eff.shape[0] != mc.ncas:
        raise RuntimeError(
            "Active space size error. nmo=%d ncore=%d ncas=%d" %
            (mo_coeff.shape[1], mc.ncore, mc.ncas))

    max_memory = max(400, mc.max_memory - lib.current_memory()[0])
    e_tot, ci = mc.fcisolver.kernel(
        h1eff, eri_gas, mc.ncas, mc.nelecas,
        ci0=ci0, verbose=log, max_memory=max_memory, ecore=energy_core)

    log.timer("GASCI solver", *t1)
    return e_tot, e_tot - energy_core, ci


def as_scanner(mc):
    """Return a GASCI potential-energy-surface scanner.

    The previous CI solution is reused only when the normalized GAS problem,
    determinant count, and requested root count are unchanged.
    """

    if isinstance(mc, lib.SinglePointScanner):
        return mc
    logger.info(mc, "Create scanner for %s", mc.__class__)
    name = mc.__class__.__name__ + GASCIScanner.__name_mixin__
    return lib.set_class(
        GASCIScanner(mc), (GASCIScanner, mc.__class__), name)


class GASCIScanner(lib.SinglePointScanner):
    """Callable GASCI scanner following the PySCF CASCI scanner protocol."""

    __name_mixin__ = "Scanner"

    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)
        self._scf = mc._scf.as_scanner()

    def __call__(self, mol_or_geom, mo_coeff=None, ci0=None):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        for key in ("with_df", "with_x2c", "with_solvent", "with_dftd3"):
            submodule = getattr(self, key, None)
            if submodule:
                submodule.reset(mol)
        if mo_coeff is None:
            self._scf(mol)
            mo_coeff = self._scf.mo_coeff
        self.mol = mol
        if ci0 is None:
            # A compact GAS space can contain disconnected spin-supergroup
            # sectors.  Reusing only the preceding roots may then follow an
            # excited sector through a crossing.  For spaces covered by the
            # bounded exact pspace, solve for the lowest roots afresh; larger
            # production spaces retain PySCF-style CI reuse.
            try:
                ndet = self._gas_problem_signature()[-1]
            except (TypeError, ValueError, NotImplementedError):
                pass
            else:
                pspace_size = int(getattr(
                    self.fcisolver, "pspace_size", 400))
                if (not self.fcisolver.davidson_only and
                        ndet <= pspace_size and
                        ndet <= fci_gas.GAS_PSPACE_MATVEC_MAX):
                    self._clear_ci_guess()
        return self.kernel(mo_coeff, ci0)[0]


class GASCI(casci.CASCI):
    """Configuration interaction in a generalized active space.

    Args:
        mf : SCF object
            Molecular-orbital reference used to form active-space integrals.
        ncas : int
            Total number of active orbitals.
        nelecas : int or pair of ints
            Number of active electrons, optionally resolved as alpha/beta.
        gas_orbs : sequence of ints, optional
            Ordered numbers of orbitals in the GAS subspaces.
        gas_restr : object, optional
            Restriction in the format selected by ``gas_restr_type``.
        gas_restr_type : str
            ``spin-supergroup``, ``supergroup``, ``cumulative-occ``, or
            ``ras``.  Every representation is converted to canonical D.

    Notes:
        GAS CI vectors are one-dimensional arrays in canonical block order.
        ``e_gas`` is the public active-space energy name; the inherited
        ``e_cas`` attribute is retained only for PySCF compatibility.
    """

    _keys = casci.CASCI._keys | {
        "gas_orbs", "gas_restr", "gas_restr_type", "e_spin_penalty",
        "e_tot_physical", "e_gas_physical", "spin_penalty_method",
    }

    def __init__(self, mf, ncas, nelecas, gas_orbs=None, gas_restr=None,
                 gas_restr_type=addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
                 **kwargs):
        super().__init__(mf, ncas, nelecas, **kwargs)
        self.gas_orbs = tuple(gas_orbs) if gas_orbs is not None else None
        self.gas_restr = gas_restr
        self.gas_restr_type = gas_restr_type
        self.fcisolver = fci_gas.FCISolver(
            getattr(mf, "mol", None), gas_orbs=self.gas_orbs,
            gas_restr=self.gas_restr, gas_restr_type=self.gas_restr_type)
        self._gas_ci_signature = None
        self.e_spin_penalty = None
        self.e_tot_physical = None
        self.e_gas_physical = None
        self.spin_penalty_method = None

    @property
    def ngas(self):
        """Number of ordered GAS subspaces presented to the user."""

        return 1 if self.gas_orbs is None else len(self.gas_orbs)

    @property
    def e_gas(self):
        """Electronic energy of the GAS problem, excluding the core energy."""

        return self.e_cas

    def _sync_fcisolver(self):
        self.fcisolver.gas_orbs = self.gas_orbs
        self.fcisolver.gas_restr = self.gas_restr
        self.fcisolver.gas_restr_type = self.gas_restr_type

    def _effective_nelecas(self, nelecas=None):
        """Return active alpha/beta counts after applying ``fcisolver.spin``."""

        if nelecas is None:
            nelecas = self.nelecas
        return fci_addons._unpack_nelec(nelecas, self.fcisolver.spin)

    def _normalized_restriction(self, return_info=False):
        gas_orbs = (int(self.ncas),) if self.gas_orbs is None else self.gas_orbs
        return addons_gas.normalize_gas_spec(
            gas_orbs, self._effective_nelecas(),
            self.gas_restr, self.gas_restr_type,
            return_info=return_info)

    def gas_space_info(self):
        """Return normalized GAS metadata and compact C-space information.

        ``metadata`` describes the public GAS specification after its
        documented normalization.  ``core`` is read from the constructed
        ``gas_space_t`` object and therefore reports the space actually seen
        by the C kernels.
        """

        self._sync_fcisolver()
        gas_orbs, blocks, info = self._normalized_restriction(
            return_info=True)
        restriction_type = self.gas_restr_type
        if self.gas_restr is None:
            restriction = None
        elif restriction_type == addons_gas.GAS_RESTR_SPIN_SUPERGROUP:
            restriction = numpy.array(
                info["canonical_spin_supergroups"], copy=True)
        elif restriction_type == addons_gas.GAS_RESTR_SUPERGROUP:
            restriction = numpy.array(
                info["canonical_supergroups"], copy=True)
        elif restriction_type == addons_gas.GAS_RESTR_CUMULATIVE_OCC:
            restriction = numpy.array(info["cumulative_bounds"], copy=True)
        elif restriction_type == addons_gas.GAS_RESTR_RAS:
            restriction = {
                "max_holes": int(info["max_holes"]),
                "max_particles": int(info["max_particles"]),
            }
        else:  # normalize_gas_spec rejects this before reaching this branch.
            raise RuntimeError("unrecognized normalized GAS restriction type")

        user_gas_orbs = ((int(self.ncas),) if self.gas_orbs is None else
                         tuple(int(value) for value in self.gas_orbs))
        with fci_gas.GasSpace(
                gas_orbs, self._effective_nelecas(), blocks,
                lib=self.fcisolver.lib) as space:
            core = space.core_info()
        return {
            "metadata": {
                "gas_orbs": user_gas_orbs,
                "gas_restr_type": restriction_type,
                "gas_restr": restriction,
                "kernel_gas_orbs": tuple(int(value) for value in gas_orbs),
                "spin_supergroups": numpy.array(blocks, copy=True),
            },
            "core": core,
        }

    def _gas_problem_signature(self):
        """Return the normalized GAS definition associated with a CI vector."""

        gas_orbs, gas_restr = self._normalized_restriction()
        gas_restr = numpy.ascontiguousarray(gas_restr, dtype=numpy.int32)
        nelecas = self._effective_nelecas()
        limits = addons_gas.check_kernel_limits(
            gas_orbs, nelecas, gas_restr)
        digest = hashlib.sha256(gas_restr.tobytes()).digest()
        nroots = int(getattr(self.fcisolver, "nroots", 1))
        return (
            int(self.ncas), tuple(int(value) for value in nelecas),
            tuple(int(value) for value in gas_orbs),
            addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
            tuple(int(value) for value in gas_restr.shape), digest,
            nroots, int(limits["ndet_estimate"]),
        )

    def _ci_matches_signature(self, ci, signature):
        if ci is None or self._gas_ci_signature != signature:
            return False
        nroots, ndet = signature[-2:]
        if isinstance(ci, (list, tuple)):
            return (len(ci) == nroots and
                    all(numpy.asarray(root).size == ndet for root in ci))
        array = numpy.asarray(ci)
        if nroots == 1:
            return array.size == ndet
        return array.ndim == 2 and array.shape in {
            (nroots, ndet), (ndet, nroots),
        }

    def _clear_ci_guess(self):
        self.ci = None
        self._gas_ci_signature = None
        for name in ("ci", "eci"):
            if hasattr(self.fcisolver, name):
                setattr(self.fcisolver, name, None)
        self.e_spin_penalty = None
        self.e_tot_physical = None
        self.e_gas_physical = None
        self.spin_penalty_method = None

    def reset(self, mol=None):
        """Reset molecular data while retaining only a compatible GAS CI guess."""

        previous_signature = self._gas_ci_signature
        super().reset(mol)
        self._sync_fcisolver()
        if previous_signature is not None:
            try:
                current_signature = self._gas_problem_signature()
            except (TypeError, ValueError, NotImplementedError):
                self._clear_ci_guess()
            else:
                if current_signature != previous_signature:
                    self._clear_ci_guess()
        return self

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        ncore = self.ncore
        nvir = self.mo_coeff.shape[1] - ncore - self.ncas
        nelecas = self._effective_nelecas()
        log.info("GAS (%de+%de, %do), ncore = %d, nvir = %d",
                 nelecas[0], nelecas[1], self.ncas, ncore, nvir)
        if self.frozen is not None:
            log.info("frozen orbitals %s", str(self.frozen))
        if self.extrasym is not None:
            log.info("extra symmetry labels:\n%s", str(self.extrasym))
        log.info("gas_orbs = %s", self.gas_orbs)
        log.info("gas_restr_type = %s", self.gas_restr_type)
        try:
            gas_orbs, gas_restr, restr_info = self._normalized_restriction(
                return_info=True)
            limits = addons_gas.check_kernel_limits(
                gas_orbs, nelecas, gas_restr)
            log.info("number of kernel GAS spaces = %d", limits["ngas"])
            if self.gas_restr_type == addons_gas.GAS_RESTR_SPIN_SUPERGROUP:
                log.info("spin-supergroup ordering = alpha-sector "
                         "lexicographic, then beta-sector lexicographic")
                log.info("input legal spin-supergroups = %d",
                         restr_info["input_nrow"])
                log.info("duplicate spin-supergroups removed = %d",
                         restr_info["duplicates_removed"])
                log.info("spin-supergroup rows reordered = %s",
                         "yes" if restr_info["order_changed"] else "no")
            elif self.gas_restr_type == addons_gas.GAS_RESTR_SUPERGROUP:
                log.info("supergroup ordering = lexicographic")
                log.info("input legal supergroups = %d",
                         restr_info["input_nrow"])
                log.info("duplicate supergroups removed = %d",
                         restr_info["duplicates_removed"])
                log.info("supergroup rows reordered = %s",
                         "yes" if restr_info["order_changed"] else "no")
                log.info("canonical legal supergroups =\n%s",
                         restr_info["canonical_supergroups"])
            elif (self.gas_restr_type ==
                  addons_gas.GAS_RESTR_CUMULATIVE_OCC):
                log.info("cumulative occupation bounds =\n%s",
                         restr_info["cumulative_bounds"])
                log.info("number of generated legal supergroups = %d",
                         restr_info["canonical_supergroups"].shape[0])
            elif self.gas_restr_type == addons_gas.GAS_RESTR_RAS:
                log.info("RAS orbital spaces (RAS1, RAS2, RAS3) = %s",
                         restr_info["ras_orbs"])
                log.info("RAS maximum holes in RAS1 = %d",
                         restr_info["max_holes"])
                log.info("RAS maximum particles in RAS3 = %d",
                         restr_info["max_particles"])
                if restr_info["empty_spaces_removed"]:
                    log.info("empty RAS spaces omitted before C kernel = %s",
                             restr_info["empty_spaces_removed"])
                    log.info("kernel gas_orbs = %s", gas_orbs)
                log.info("equivalent cumulative occupation bounds =\n%s",
                         restr_info["cumulative_bounds"])
                log.info("number of generated legal supergroups = %d",
                         restr_info["canonical_supergroups"].shape[0])
            log.info("number of legal spin-supergroups = %d", limits["nblock"])
            log.info("estimated number of determinants = %d",
                     limits["ndet_estimate"])
            log.info("C kernel limit precheck = passed")
        except Exception as err:
            log.warn("GAS restriction precheck failed: %s", err)
        log.info("natorb = %s", self.natorb)
        log.info("canonicalization = %s", self.canonicalization)
        log.info("sorting_mo_energy = %s", self.sorting_mo_energy)
        log.info("max_memory %d MB (current use %d MB)",
                 self.max_memory, lib.current_memory()[0])
        self._sync_fcisolver()
        self.fcisolver.dump_flags(log.verbose)
        if self.mo_coeff is None:
            log.error("Orbitals for GASCI are not specified. The relevant SCF "
                      "object may not be initialized.")
        return self

    def check_sanity(self):
        super().check_sanity()
        if self.natorb:
            raise NotImplementedError(
                "automatic in-place GAS natural-orbital rotation is not "
                "implemented; use get_gas_natorb or "
                "get_gas_pseudo_natorb for analysis")
        gas_orbs, gas_restr = self._normalized_restriction()
        addons_gas.check_kernel_limits(
            gas_orbs, self._effective_nelecas(), gas_restr)
        return self

    def _check_mo_orthonormality(self, mo_coeff=None, verbose=None):
        """Validate the MO metric before integral transformation."""

        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        array = numpy.asarray(mo_coeff)
        if array.ndim != 2:
            raise ValueError("mo_coeff must be a two-dimensional array")
        if numpy.iscomplexobj(array):
            raise TypeError(
                "the libfci_gas C kernels require real-valued orbitals")
        if array.dtype.kind not in "fiu":
            raise TypeError("mo_coeff must contain numeric values")
        if array.shape[1] < self.ncore + self.ncas:
            raise ValueError(
                "mo_coeff has fewer columns than ncore + ncas")
        if not numpy.all(numpy.isfinite(array)):
            raise ValueError("mo_coeff must contain only finite values")
        overlap = numpy.asarray(self._scf.get_ovlp())
        if overlap.shape != (array.shape[0], array.shape[0]):
            raise ValueError(
                "AO overlap shape is incompatible with mo_coeff")
        metric = reduce(numpy.dot, (array.conj().T, overlap, array))
        error = float(numpy.max(numpy.abs(
            metric - numpy.eye(array.shape[1], dtype=metric.dtype))))
        if not numpy.isfinite(error):
            raise ValueError("MO orthonormality error is not finite")
        if error > MO_ORTH_ERROR_TOL:
            raise ValueError(
                "mo_coeff is not orthonormal in the AO metric: "
                "max|C^H S C - I| = %.6g exceeds %.6g" %
                (error, MO_ORTH_ERROR_TOL))
        log = logger.new_logger(self, verbose)
        if error > MO_ORTH_WARN_TOL:
            log.warn("MO orthonormality max|C^H S C - I| = %.6g", error)
        else:
            log.debug("MO orthonormality max|C^H S C - I| = %.6g", error)
        return error

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        """Run fixed-orbital GASCI and return PySCF-style results.

        Returns:
            Tuple ``(e_tot, e_gas, ci, mo_coeff, mo_energy)``.  For multiple
            roots, energies and CI vectors follow the selected PySCF solver
            convention.
        """

        self._sync_fcisolver()
        if mo_coeff is None:
            if self.mo_coeff is None and self._scf.mol.nelectron > 0:
                self._scf.run()
                self.mo_coeff = self._scf.mo_coeff
            mo_coeff = self.mo_coeff
        log = logger.new_logger(self, verbose)
        self._check_mo_orthonormality(mo_coeff, log)
        self.mo_coeff = mo_coeff
        self.check_sanity()
        self.dump_flags(log)
        signature = self._gas_problem_signature()
        if ci0 is None:
            if self._ci_matches_signature(self.ci, signature):
                ci0 = self.ci
            else:
                self._clear_ci_guess()
        self.e_tot, self.e_cas, self.ci = kernel(
            self, mo_coeff, ci0=ci0, verbose=log)
        self._gas_ci_signature = signature

        solver_physical = getattr(self.fcisolver, "e_physical", None)
        solver_penalty = getattr(self.fcisolver, "e_spin_penalty", None)
        self.spin_penalty_method = getattr(
            self.fcisolver, "spin_penalty_method", None)
        if solver_physical is None or solver_penalty is None:
            self.e_spin_penalty = None
            self.e_tot_physical = self.e_tot
            self.e_gas_physical = self.e_gas
        else:
            physical = numpy.asarray(solver_physical, dtype=numpy.float64)
            penalty = numpy.asarray(solver_penalty, dtype=numpy.float64)
            core = (numpy.asarray(self.e_tot, dtype=numpy.float64) -
                    numpy.asarray(self.e_gas, dtype=numpy.float64))
            gas_physical = physical - core
            if physical.ndim == 0:
                self.e_spin_penalty = float(penalty)
                self.e_tot_physical = float(physical)
                self.e_gas_physical = float(gas_physical)
            else:
                self.e_spin_penalty = penalty
                self.e_tot_physical = physical
                self.e_gas_physical = gas_physical

        if self.canonicalization:
            gasdm1 = None
            if isinstance(self.ci, (list, tuple)):
                gasdm1 = self.make_gasdm1(
                    state=None if self._has_state_weights() else 0)
            self.canonicalize_(
                mo_coeff, self.ci, sort=self.sorting_mo_energy,
                gas_natorb=False, gasdm1=gasdm1, verbose=log)

        if getattr(self.fcisolver, "converged", None) is not None:
            self.converged = bool(numpy.all(self.fcisolver.converged))
        else:
            self.converged = True
        log.info("GASCI converged" if self.converged else "GASCI not converged")
        self._finalize()
        return self.e_tot, self.e_gas, self.ci, self.mo_coeff, self.mo_energy

    def gasci(self, mo_coeff=None, ci0=None, verbose=None):
        """Alias of :meth:`kernel` using GAS terminology."""

        return self.kernel(mo_coeff, ci0, verbose)

    def as_scanner(self):
        return as_scanner(self)

    def get_h1gas(self, mo_coeff=None, ncas=None, ncore=None):
        """Return the effective one-electron Hamiltonian in the GAS space."""

        return self.get_h1eff(mo_coeff, ncas, ncore)

    def get_h2gas(self, mo_coeff=None):
        """Return active-space two-electron integrals for GASCI."""

        return self.get_h2eff(mo_coeff)

    def get_h1cas(self, *args, **kwargs):
        raise NotImplementedError(
            "get_h1cas is a CAS-specific name; use get_h1gas for GASCI")

    h1e_for_cas = get_h1cas

    def get_h2cas(self, *args, **kwargs):
        raise NotImplementedError(
            "get_h2cas is a CAS-specific name; use get_h2gas for GASCI")

    def sort_mo(self, gaslst, mo_coeff=None, base=1):
        """Pick orbitals for ordered GAS subspaces.

        ``gaslst`` must contain one orbital-index list per entry of
        ``gas_orbs``.  ``base`` follows PySCF: 0 selects C-style indices and
        1 selects Fortran-style indices.
        """

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        self._check_mo_orthonormality(mo_coeff)
        return addons_gas.sort_mo(self, mo_coeff, gaslst, base)

    def get_fock(self, mo_coeff=None, ci=None, eris=None, gasdm1=None,
                 verbose=None):
        """Build the generalized Fock matrix from a GAS one-particle DM."""

        if gasdm1 is None:
            ci_source = self.ci if ci is None else ci
            if isinstance(ci_source, (list, tuple)):
                gasdm1 = self.make_gasdm1(
                    ci_source,
                    state=None if self._has_state_weights() else 0)
        return casci.get_fock(
            self, mo_coeff, ci, eris, casdm1=gasdm1, verbose=verbose)

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, sort=False,
                     gas_natorb=False, gasdm1=None, verbose=None,
                     **kwargs):
        """Canonicalize core/external orbitals without rotating the GAS space."""

        if gas_natorb:
            raise NotImplementedError(
                "automatic in-place GAS natural-orbital rotation is not "
                "implemented; use get_gas_pseudo_natorb for analysis")
        return casci.canonicalize(
            self, mo_coeff, ci, eris, sort=sort, cas_natorb=False,
            casdm1=gasdm1, verbose=verbose, **kwargs)

    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      gas_natorb=False, gasdm1=None, verbose=None,
                      **kwargs):
        mo_coeff, ci, mo_energy = self.canonicalize(
            mo_coeff, ci, eris, sort=sort, gas_natorb=gas_natorb,
            gasdm1=gasdm1, verbose=verbose, **kwargs)
        self.mo_coeff = mo_coeff
        self.mo_energy = mo_energy
        return mo_coeff, ci, mo_energy

    def cas_natorb(self, *args, **kwargs):
        raise NotImplementedError(
            "CAS natural-orbital rotation is not valid for a restricted GAS")

    cas_natorb_ = cas_natorb

    def fix_spin_(self, shift=0.2, ss=None):
        """Use a PySCF-style energy penalty to target a GAS spin state.

        This optional numerical aid is only defined for a spin-complete GAS
        restriction.  Its success depends on the energy gaps between spin
        sectors and on a suitable finite shift; it is not a spin-adapted CI
        representation.  Small GAS spaces are solved exactly, while larger
        spaces use target-spin-projected guesses together with global probes
        so that an insufficient shift does not silently change the eigenproblem.
        """

        self._sync_fcisolver()
        gas_orbs, gas_restr = self._normalized_restriction()
        nelecas = self._effective_nelecas()
        if not addons_gas.is_spin_complete(
                gas_orbs, nelecas, gas_restr):
            raise ValueError(
                "fix_spin_ requires a spin-complete GAS restriction")
        fci.addons.fix_spin_(self.fcisolver, shift, ss)
        return self

    fix_spin = fix_spin_

    def state_average(self, weights=(0.5, 0.5), wfnsym=None):
        """Return a GASCI object carrying fixed-orbital state weights.

        The weights define averaged RDMs and generalized Fock matrices for
        fixed-orbital analysis.  GASCI does not optimize orbitals, and every
        root retains its own energy.
        """

        return addons_gas.state_average(self, weights, wfnsym)

    def state_average_(self, weights=(0.5, 0.5), wfnsym=None):
        addons_gas.state_average_(self, weights, wfnsym)
        return self

    def state_average_mix(self, *args, **kwargs):
        raise NotImplementedError(
            "state_average_mix is not implemented for GASCI")

    state_average_mix_ = state_average_mix

    def nuc_grad_method(self):
        raise NotImplementedError(
            "GASCI nuclear gradients are not implemented")

    def to_gpu(self, *args, **kwargs):
        raise NotImplementedError(
            "the libfci_gas C/OpenMP backend does not support GPU execution")

    def state_specific_(self, state=1, wfnsym=None):
        if wfnsym is not None:
            raise NotImplementedError(
                "GAS wavefunction symmetry filtering is not implemented")
        if isinstance(self, addons_gas.StateAverageGASCI):
            raise ValueError(
                "undo state weighting before selecting a state-specific root")
        addons.state_specific_(self, state=state, wfnsym=None)
        return self

    state_specific = state_specific_

    def _has_state_weights(self):
        weights = getattr(self.fcisolver, "weights", None)
        return (weights is not None and
                isinstance(self.fcisolver, addons.StateAverageFCISolver))

    def _state_weights(self):
        if not self._has_state_weights():
            return None
        return numpy.asarray(self.fcisolver.weights, dtype=numpy.float64)

    def _base_fcisolver_method(self, name):
        if self._has_state_weights():
            base = super(addons.StateAverageFCISolver, self.fcisolver)
            return getattr(base, name)
        return getattr(self.fcisolver, name)

    def _ci_for_rdm(self, ci, state):
        ci = self.ci if ci is None else ci
        if self._has_state_weights() and state is None:
            if not isinstance(ci, (list, tuple)):
                raise ValueError(
                    "weighted GAS RDMs require one CI vector per state")
            if len(ci) != len(self._state_weights()):
                raise ValueError("CI root count does not match state weights")
            return ci, True
        selected = 0 if state is None else int(state)
        return self._select_ci(ci, selected), False

    def _spin_square_for_ci(self, ci, ncas, nelecas):
        return self._spin_square_for_roots(
            [ci], ncas, nelecas)[0]

    def _spin_square_for_roots(self, roots, ncas, nelecas):
        nelecas = self._effective_nelecas(nelecas)
        values = []
        with self.fcisolver.make_rdm_plan(ncas, nelecas) as plan:
            for ci in roots:
                dm1s, dm2s = plan.make_rdm12s(ci, ci)
                values.append(
                    fci_gas.spin_square_from_rdm12s(
                        dm1s, dm2s, nelecas))
        return values

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        weights = self._state_weights()
        if weights is not None:
            totals = numpy.asarray(
                self.fcisolver.e_states, dtype=numpy.float64).reshape(-1)
            roots = self.ci if isinstance(self.ci, (list, tuple)) else [self.ci]
            energy_core = float(self.e_tot) - float(self.e_gas)
            log.note("GASCI weighted energy (fixed orbitals) = %#.15g",
                     self.e_tot)
            log.note("GASCI energy for each state")
            spin_values = self._spin_square_for_roots(
                roots, self.ncas, self.nelecas)
            for i, (weight, e_tot, spin) in enumerate(
                    zip(weights, totals, spin_values)):
                log.note("  State %d weight %g  E = %#.15g  E(CI) = %#.15g  "
                         "S^2 = %.7f  multiplicity = %.7f",
                         i, weight, e_tot, e_tot - energy_core,
                         spin[0], spin[1])
            return self

        energies = numpy.asarray(self.e_gas).reshape(-1)
        totals = numpy.asarray(self.e_tot).reshape(-1)
        roots = self.ci if isinstance(self.ci, (list, tuple)) else [self.ci]
        try:
            spin_values = self._spin_square_for_roots(
                roots, self.ncas, self.nelecas)
        except NotImplementedError:
            spin_values = [None] * len(roots)
        for i, (e_tot, e_gas) in enumerate(zip(totals, energies)):
            ss = None if spin_values[i] is None else spin_values[i][0]
            state = getattr(self.fcisolver, "state", None)
            if len(energies) == 1 and state is not None and ss is None:
                log.note("GASCI state %3d  E = %#.15g  E(CI) = %#.15g",
                         state, e_tot, e_gas)
            elif len(energies) == 1 and state is not None:
                log.note("GASCI state %3d  E = %#.15g  E(CI) = %#.15g  "
                         "S^2 = %.7f", state, e_tot, e_gas, ss)
            elif len(energies) == 1:
                if ss is None:
                    log.note("GASCI E = %#.15g  E(CI) = %#.15g", e_tot, e_gas)
                else:
                    log.note("GASCI E = %#.15g  E(CI) = %#.15g  S^2 = %.7f",
                             e_tot, e_gas, ss)
            elif ss is None:
                log.note("GASCI state %3d  E = %#.15g  E(CI) = %#.15g",
                         i, e_tot, e_gas)
            else:
                log.note("GASCI state %3d  E = %#.15g  E(CI) = %#.15g  S^2 = %.7f",
                         i, e_tot, e_gas, ss)
        if self.e_spin_penalty is not None:
            penalties = numpy.asarray(
                self.e_spin_penalty, dtype=numpy.float64).reshape(-1)
            physical = numpy.asarray(
                self.e_tot_physical, dtype=numpy.float64).reshape(-1)
            target = getattr(self.fcisolver, "ss_value", None)
            if target is None:
                nelecas = self._effective_nelecas()
                sz = 0.5 * abs(nelecas[0] - nelecas[1])
                target = sz * (sz + 1.0)
            state = getattr(self.fcisolver, "state", None)
            for i, (penalty, e_physical) in enumerate(
                    zip(penalties, physical)):
                label = state if len(penalties) == 1 and state is not None else i
                log.note("GASCI state %3d  E(physical) = %#.15g  "
                         "E(spin penalty) = %#.15g  target S^2 = %.7f  "
                         "method = %s", label, e_physical, penalty, target,
                         self.spin_penalty_method)
                actual = None if spin_values[i] is None else spin_values[i][0]
                if actual is not None and abs(actual - target) > 1e-6:
                    log.warn("GASCI state %d did not reach target S^2: "
                             "target = %.7f, actual = %.7f.  The spin "
                             "penalty shift may be insufficient.",
                             label, target, actual)
        return self

    def _select_ci(self, ci=None, state=0):
        if ci is None:
            ci = self.ci
        if isinstance(ci, (list, tuple)):
            return ci[int(state)]
        arr = numpy.asarray(ci)
        nroots = int(getattr(self.fcisolver, "nroots", 1))
        if arr.ndim == 2 and arr.shape[0] == nroots:
            return arr[int(state)]
        if arr.ndim == 2 and arr.shape[1] == nroots:
            return arr[:, int(state)]
        return ci

    def make_gasdm1(self, ci=None, ncas=None, nelecas=None, state=None):
        """Return a spin-summed active-space GAS 1-RDM."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ci_value, weighted = self._ci_for_rdm(ci, state)
        method = (self.fcisolver.make_rdm1 if weighted else
                  self._base_fcisolver_method("make_rdm1"))
        return method(ci_value, ncas, nelecas)

    def make_gasdm1s(self, ci=None, ncas=None, nelecas=None, state=None):
        """Return alpha and beta active-space GAS 1-RDMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ci_value, weighted = self._ci_for_rdm(ci, state)
        method = (self.fcisolver.make_rdm1s if weighted else
                  self._base_fcisolver_method("make_rdm1s"))
        return method(ci_value, ncas, nelecas)

    def make_gasdm12(self, ci=None, ncas=None, nelecas=None, state=None):
        """Return spin-summed active-space GAS 1- and 2-RDMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ci_value, weighted = self._ci_for_rdm(ci, state)
        method = (self.fcisolver.make_rdm12 if weighted else
                  self._base_fcisolver_method("make_rdm12"))
        return method(ci_value, ncas, nelecas)

    def make_gasdm2(self, ci=None, ncas=None, nelecas=None, state=None):
        """Return the spin-summed active-space GAS 2-RDM."""

        return self.make_gasdm12(ci, ncas, nelecas, state)[1]

    def make_gasdm12s(self, ci=None, ncas=None, nelecas=None, state=None):
        """Return spin-resolved active-space GAS 1- and 2-RDMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ci_value, weighted = self._ci_for_rdm(ci, state)
        method = (self.fcisolver.make_rdm12s if weighted else
                  self._base_fcisolver_method("make_rdm12s"))
        return method(ci_value, ncas, nelecas)

    def trans_gasdm1(self, cibra=None, ciket=None, ncas=None, nelecas=None,
                     bra_state=0, ket_state=1):
        """Return the spin-summed GAS transition 1-RDM."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._select_ci(cibra, bra_state)
        ket = self._select_ci(ciket, ket_state)
        return self._base_fcisolver_method("trans_rdm1")(
            bra, ket, ncas, nelecas)

    def trans_gasdm1s(self, cibra=None, ciket=None, ncas=None, nelecas=None,
                      bra_state=0, ket_state=1):
        """Return alpha and beta GAS transition 1-RDMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._select_ci(cibra, bra_state)
        ket = self._select_ci(ciket, ket_state)
        return self._base_fcisolver_method("trans_rdm1s")(
            bra, ket, ncas, nelecas)

    def trans_gasdm12(self, cibra=None, ciket=None, ncas=None, nelecas=None,
                      bra_state=0, ket_state=1):
        """Return spin-summed GAS transition 1- and 2-RDMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._select_ci(cibra, bra_state)
        ket = self._select_ci(ciket, ket_state)
        return self._base_fcisolver_method("trans_rdm12")(
            bra, ket, ncas, nelecas)

    def trans_gasdm2(self, cibra=None, ciket=None, ncas=None, nelecas=None,
                     bra_state=0, ket_state=1):
        """Return the spin-summed GAS transition 2-RDM."""

        return self.trans_gasdm12(
            cibra, ciket, ncas, nelecas, bra_state, ket_state)[1]

    def trans_gasdm12s(self, cibra=None, ciket=None, ncas=None, nelecas=None,
                       bra_state=0, ket_state=1):
        """Return spin-resolved GAS transition 1- and 2-RDMs."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        bra = self._select_ci(cibra, bra_state)
        ket = self._select_ci(ciket, ket_state)
        return self._base_fcisolver_method("trans_rdm12s")(
            bra, ket, ncas, nelecas)

    def make_rdm1s(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                   ncore=None, state=None, **kwargs):
        """Return alpha and beta AO-basis one-particle density matrices."""

        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = self.ncore if ncore is None else ncore
        gasdm1a, gasdm1b = self.make_gasdm1s(
            ci, ncas, nelecas, state)
        return self._gasdm1s_to_ao(
            (gasdm1a, gasdm1b), mo_coeff, ncas, ncore)

    def _gasdm1s_to_ao(self, gasdm1s, mo_coeff, ncas, ncore):
        gasdm1a, gasdm1b = gasdm1s
        mocore = mo_coeff[:, :ncore]
        mogas = mo_coeff[:, ncore:ncore + ncas]
        dm1b = numpy.dot(mocore, mocore.conj().T)
        dm1a = dm1b + reduce(numpy.dot, (mogas, gasdm1a, mogas.conj().T))
        dm1b = dm1b + reduce(numpy.dot, (mogas, gasdm1b, mogas.conj().T))
        return dm1a, dm1b

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, state=None, **kwargs):
        """Return the spin-summed AO-basis one-particle density matrix."""

        dm1a, dm1b = self.make_rdm1s(
            mo_coeff, ci, ncas, nelecas, ncore, state, **kwargs)
        return dm1a + dm1b

    def spin_square(self, ci=None, ncas=None, nelecas=None, state=None):
        """Return ``(<S^2>, 2S+1)`` for one root or a weighted state set."""

        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ci_value, weighted = self._ci_for_rdm(ci, state)
        if weighted:
            values = self._spin_square_for_roots(
                ci_value, ncas, nelecas)
            ss = float(numpy.dot(
                self._state_weights(), [value[0] for value in values]))
            return ss, numpy.sqrt(max(0.0, 4.0 * ss + 1.0))
        return self._spin_square_for_ci(ci_value, ncas, nelecas)

    def get_gas_pseudo_natorb_occupations(self, gasdm1=None, ci=None,
                                           state=None, sort=True):
        """Return subspace-resolved pseudo-natural occupation spectra.

        Following the OpenMolcas convention for restricted active spaces,
        each diagonal GAS-subspace block of the spin-summed one-particle
        density matrix is diagonalized separately.  These are pseudo-natural,
        not true natural, occupations because no rotation across GAS
        subspaces is performed.  Sorting only makes the reported spectra
        deterministic; it does not define an orbital transformation.
        """

        if gasdm1 is None:
            gasdm1 = self.make_gasdm1(ci=ci, state=state)
        gasdm1 = numpy.asarray(gasdm1)
        if gasdm1.shape != (self.ncas, self.ncas):
            raise ValueError("gasdm1 must have shape (ncas, ncas)")

        gas_orbs = ((self.ncas,) if self.gas_orbs is None else
                    tuple(int(value) for value in self.gas_orbs))
        occupations = []
        offset = 0
        for size in gas_orbs:
            block = gasdm1[offset:offset + size, offset:offset + size]
            values, _ = self._natural_eigensystem(block, sort)
            occupations.append(values)
            offset += size
        return tuple(occupations)

    @staticmethod
    def _natural_eigensystem(dm1, sort=True):
        dm1 = numpy.asarray(dm1)
        dm1 = 0.5 * (dm1 + dm1.conj().T)
        occupations, rotation = numpy.linalg.eigh(dm1)
        occupations = occupations.real
        if sort:
            order = numpy.argsort(occupations)[::-1]
            occupations = occupations[order]
            rotation = rotation[:, order]
        return occupations, rotation

    def _rotate_gas_orbitals(self, mo_coeff, rotation):
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        mo_coeff = numpy.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("mo_coeff must be a two-dimensional array")
        if mo_coeff.shape[1] < self.ncore + self.ncas:
            raise ValueError("mo_coeff does not contain the complete GAS space")
        if rotation.shape != (self.ncas, self.ncas):
            raise ValueError("GAS orbital rotation has an invalid shape")
        mo_natorb = mo_coeff.copy()
        gas = slice(self.ncore, self.ncore + self.ncas)
        mo_natorb[:, gas] = numpy.dot(mo_coeff[:, gas], rotation)
        return mo_natorb

    def get_gas_natorb(self, mo_coeff=None, gasdm1=None, ci=None,
                       state=0, sort=True):
        """Return root-specific true natural orbitals and occupations.

        The complete spin-summed GAS one-particle density matrix is
        diagonalized.  For a restricted GAS, the resulting orbitals can mix
        different GAS subspaces and are therefore analysis orbitals only.
        This method never modifies ``mo_coeff`` or transforms the CI vector.
        """

        if gasdm1 is None and state is None:
            raise ValueError(
                "get_gas_natorb requires an explicit state; use "
                "get_gas_average_natorb for a state-average density")
        if gasdm1 is None:
            gasdm1 = self.make_gasdm1(ci=ci, state=state)
        gasdm1 = numpy.asarray(gasdm1)
        if gasdm1.shape != (self.ncas, self.ncas):
            raise ValueError("gasdm1 must have shape (ncas, ncas)")
        occupations, rotation = self._natural_eigensystem(gasdm1, sort)
        return self._rotate_gas_orbitals(mo_coeff, rotation), occupations

    def get_gas_average_natorb(self, mo_coeff=None, gasdm1=None, ci=None,
                               sort=True):
        """Return natural orbitals of the explicitly state-averaged GAS DM.

        This method is available only on a state-average GASCI object.  Like
        :meth:`get_gas_natorb`, the returned orbitals are for analysis and do
        not replace the computational GAS orbitals.
        """

        if not self._has_state_weights():
            raise ValueError(
                "get_gas_average_natorb requires a state-average GASCI object")
        if gasdm1 is None:
            gasdm1 = self.make_gasdm1(ci=ci, state=None)
        return self.get_gas_natorb(
            mo_coeff=mo_coeff, gasdm1=gasdm1, sort=sort)

    def get_gas_pseudo_natorb(self, mo_coeff=None, gasdm1=None, ci=None,
                              state=None, sort=True):
        """Return subspace-preserving pseudo-natural orbitals.

        Each diagonal GAS-subspace block of the spin-summed one-particle
        density matrix is diagonalized independently.  The returned orbital
        coefficients preserve the GAS partition but are not installed on the
        calculation object, and the CI vector is left unchanged.
        """

        if gasdm1 is None:
            gasdm1 = self.make_gasdm1(ci=ci, state=state)
        gasdm1 = numpy.asarray(gasdm1)
        if gasdm1.shape != (self.ncas, self.ncas):
            raise ValueError("gasdm1 must have shape (ncas, ncas)")

        gas_orbs = ((self.ncas,) if self.gas_orbs is None else
                    tuple(int(value) for value in self.gas_orbs))
        rotation = numpy.zeros_like(gasdm1)
        occupations = []
        offset = 0
        for size in gas_orbs:
            block = gasdm1[offset:offset + size, offset:offset + size]
            values, vectors = self._natural_eigensystem(block, sort)
            rotation[offset:offset + size, offset:offset + size] = vectors
            occupations.append(values)
            offset += size
        return (self._rotate_gas_orbitals(mo_coeff, rotation),
                tuple(occupations))

    def analyze(self, mo_coeff=None, ci=None, verbose=None,
                large_ci_tol=LARGE_CI_TOL,
                with_meta_lowdin=WITH_META_LOWDIN, state=None,
                **kwargs):
        """Analyze GASCI roots without rotations across GAS subspaces."""

        log = logger.new_logger(self, verbose)
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        ci_source = self.ci if ci is None else ci
        if isinstance(ci_source, (list, tuple)):
            roots = list(ci_source)
            root_labels = list(range(len(roots)))
        else:
            roots = [ci_source]
            root_labels = [int(getattr(self.fcisolver, "state", 0))]
        if state is not None:
            selected = int(state)
            roots = [self._select_ci(ci_source, selected)]
            root_labels = [selected]

        log.info("")
        log.info("******** GASCI analysis ********")
        weights = self._state_weights()
        energies = numpy.asarray(self.e_states if weights is not None else
                                 self.e_tot).reshape(-1)
        spin_values = self._spin_square_for_roots(
            roots, self.ncas, self.nelecas)
        for label, spin in zip(root_labels, spin_values):
            ss, mult = spin
            energy_index = label if energies.size > 1 else 0
            if weights is None:
                log.note("GASCI state %3d  E = %#.15g  S^2 = %.7f  "
                         "multiplicity = %.7f", label,
                         energies[energy_index], ss, mult)
            else:
                log.note("GASCI state %3d  weight = %g  E = %#.15g  "
                         "S^2 = %.7f  multiplicity = %.7f",
                         label, weights[label], energies[energy_index],
                         ss, mult)

        density_state = None if weights is not None and state is None else (
            root_labels[0])
        gasdm1s = self.make_gasdm1s(ci_source, state=density_state)
        gasdm1 = gasdm1s[0] + gasdm1s[1]
        occupations = self.get_gas_pseudo_natorb_occupations(
            gasdm1=gasdm1)
        gas_orbs = ((self.ncas,) if self.gas_orbs is None else
                    tuple(int(value) for value in self.gas_orbs))
        offset = 0
        for igas, (size, values) in enumerate(zip(gas_orbs, occupations)):
            block = gasdm1[offset:offset + size, offset:offset + size]
            log.info("GAS subspace %d electron number = %.12g", igas + 1,
                     numpy.trace(block).real)
            log.info("GAS subspace %d pseudo-natural occupations %s",
                     igas + 1, numpy.array2string(values, precision=10))
            offset += size

        if log.verbose >= logger.INFO and ci_source is not None:
            log.info("** Largest GAS CI components **")
            large_ci = self._base_fcisolver_method("large_ci")
            for root, label in zip(roots, root_labels):
                log.info("  [alpha occ-orbitals] [beta occ-orbitals]  "
                         "state %-3d CI coefficient", label)
                for coeff, alpha, beta in large_ci(
                        root, self.ncas, self.nelecas, large_ci_tol,
                        return_strs=False):
                    log.info("  %-20s %-30s % .12f",
                             alpha, beta, coeff)

        dm1a, dm1b = self._gasdm1s_to_ao(
            gasdm1s, mo_coeff, self.ncas, self.ncore)
        if log.verbose >= logger.INFO:
            ovlp = self._scf.get_ovlp()
            dm1 = dm1a + dm1b
            spin_dm1 = dm1a - dm1b
            if with_meta_lowdin:
                self._scf.mulliken_meta(
                    self.mol, dm1, s=ovlp, verbose=log)
                log.info("Mulliken spin population analysis on "
                         "meta-Lowdin AOs:")
                self._scf.mulliken_meta(
                    self.mol, spin_dm1, s=ovlp, verbose=log)
            else:
                self._scf.mulliken_pop(
                    self.mol, dm1, s=ovlp, verbose=log)
                log.info("Mulliken spin population analysis on AOs:")
                self._scf.mulliken_pop(
                    self.mol, spin_dm1, s=ovlp, verbose=log)
        return dm1a, dm1b
