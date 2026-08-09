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

"""Small GAS helpers shared by the GASCI Python layer."""

import math
from collections.abc import Mapping
from functools import lru_cache

import numpy

from pyscf import lib
from pyscf.fci import addons as fci_addons
from pyscf.lib import logger
from pyscf.mcscf import addons


GAS_MAX_ORB = 63
GAS_MAX_LOCAL_ORB = 31
GAS_MAX_NGAS = 15
GAS_MAX_SECTOR = 65534
GAS_MAX_DET = 2**32 - 1
GAS_MAX_BLOCK = 2**31 - 1
GAS_RESTR_SPIN_SUPERGROUP = "spin-supergroup"
GAS_RESTR_SUPERGROUP = "supergroup"
GAS_RESTR_CUMULATIVE_OCC = "cumulative-occ"
GAS_RESTR_RAS = "ras"


def gas_restr_usage(gas_restr_type=None):
    """Return the strict public input contract for GAS restrictions."""

    header = (
        "GAS restriction input rules (only row order and duplicate rows are "
        "normalized):")
    rules = {
        GAS_RESTR_SPIN_SUPERGROUP: (
            "gas_restr_type='spin-supergroup': gas_restr is a non-empty "
            "array of shape (nrow, 2*ngas); each row is "
            "(alpha occupations, beta occupations)."),
        GAS_RESTR_SUPERGROUP: (
            "gas_restr_type='supergroup': gas_restr is a non-empty array "
            "of shape (nrow, ngas); each row contains total occupations."),
        GAS_RESTR_CUMULATIVE_OCC: (
            "gas_restr_type='cumulative-occ': gas_restr has shape "
            "(ngas, 2), with non-decreasing [minimum, maximum] cumulative "
            "occupations and a final bound equal to the active electrons."),
        GAS_RESTR_RAS: (
            "gas_restr_type='ras': gas_orbs has exactly three entries "
            "(RAS1, RAS2, RAS3), which may be zero, and gas_restr is exactly "
            "{'max_holes': integer, 'max_particles': integer}."),
    }
    numeric = (
        "All occupations, bounds, orbital counts, and orbital indices must "
        "be integers; floating-point and Boolean values are rejected.")
    if gas_restr_type in rules:
        return "\n".join((header, rules[gas_restr_type], numeric))
    return "\n".join((header,) + tuple(rules[key] for key in (
        GAS_RESTR_SPIN_SUPERGROUP, GAS_RESTR_SUPERGROUP,
        GAS_RESTR_CUMULATIVE_OCC, GAS_RESTR_RAS)) + (numeric,))


def _exact_integer_array(values, label):
    """Require integer input and convert it to the C-kernel int32 dtype."""

    raw = numpy.asarray(values, dtype=object)
    if not all(lib.isinteger(value) for value in raw.flat):
        raise TypeError("%s must contain integers" % label)
    arr = numpy.asarray(values)
    info = numpy.iinfo(numpy.int32)
    if numpy.any(arr < info.min) or numpy.any(arr > info.max):
        raise ValueError("%s values exceed int32 range" % label)
    return numpy.asarray(arr, dtype=numpy.int32)


def _integer_vector(values, label, length=None):
    arr = _exact_integer_array(values, label)
    if arr.ndim != 1:
        raise ValueError("%s must be a one-dimensional sequence" % label)
    if length is not None and arr.size != int(length):
        raise ValueError("%s must contain exactly %d entries" %
                         (label, int(length)))
    if arr.size == 0:
        raise ValueError("%s must not be empty" % label)
    return tuple(int(value) for value in arr)


def _validate_state_weights(weights):
    weights = numpy.asarray(weights, dtype=numpy.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if not numpy.all(numpy.isfinite(weights)):
        raise ValueError("weights must be finite")
    if numpy.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    if abs(float(weights.sum()) - 1.0) > 1e-10:
        raise ValueError("weights must sum to one")
    return weights


class StateAverageFCISolver(addons.StateAverageFCISolver):
    """PySCF state wrapper with GAS-specific multi-root RDM kernels."""

    __name_mixin__ = "StateAverage"

    def dump_flags(self, verbose=None):
        super(addons.StateAverageFCISolver, self).dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info("State-average over %d states with weights %s",
                 len(self.weights), self.weights)
        log.info("state weights define the averaged GAS RDMs and generalized "
                 "Fock matrix")
        return self

    def states_make_rdm1(self, ci0, norb, nelec, *args, **kwargs):
        with self.make_rdm_plan(norb, nelec) as plan:
            return [plan.make_rdm1(ci, ci) for ci in ci0]

    def states_make_rdm1s(self, ci0, norb, nelec, *args, **kwargs):
        dm1a = []
        dm1b = []
        with self.make_rdm_plan(norb, nelec) as plan:
            for ci in ci0:
                values = plan.make_rdm1s(ci, ci)
                dm1a.append(values[0])
                dm1b.append(values[1])
        return dm1a, dm1b

    def states_make_rdm12(self, ci0, norb, nelec, *args, **kwargs):
        dm1 = []
        dm2 = []
        with self.make_rdm_plan(norb, nelec) as plan:
            for ci in ci0:
                values = plan.make_rdm12(ci, ci)
                dm1.append(values[0])
                dm2.append(values[1])
        return dm1, dm2

    def states_make_rdm12s(self, ci0, norb, nelec, *args, **kwargs):
        dm1a = []
        dm1b = []
        dm2aa = []
        dm2ab = []
        dm2bb = []
        with self.make_rdm_plan(norb, nelec) as plan:
            for ci in ci0:
                dm1s, dm2s = plan.make_rdm12s(ci, ci)
                dm1a.append(dm1s[0])
                dm1b.append(dm1s[1])
                dm2aa.append(dm2s[0])
                dm2ab.append(dm2s[1])
                dm2bb.append(dm2s[2])
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

    def states_trans_rdm12(self, ci1, ci0, norb, nelec, *args, **kwargs):
        dm1 = []
        dm2 = []
        with self.make_rdm_plan(norb, nelec) as plan:
            for bra, ket in zip(ci1, ci0):
                values = plan.make_rdm12(bra, ket)
                dm1.append(values[0])
                dm2.append(values[1])
        return dm1, dm2

    def states_spin_square(self, ci0, norb, nelec, *args, **kwargs):
        from pyscf.mcscf import fci_gas

        nelec = fci_addons._unpack_nelec(nelec, self.spin)
        values = []
        with self.make_rdm_plan(norb, nelec) as plan:
            for ci in ci0:
                dm1s, dm2s = plan.make_rdm12s(ci, ci)
                values.append(
                    fci_gas.spin_square_from_rdm12s(dm1s, dm2s, nelec))
        return ([value[0] for value in values],
                [value[1] for value in values])

    def spin_square(self, ci0, norb, nelec, *args, **kwargs):
        ss = numpy.asarray(
            self.states_spin_square(ci0, norb, nelec, *args, **kwargs)[0])
        ss_average = float(numpy.dot(self.weights, ss))
        multiplicity = numpy.sqrt(max(0.0, 4.0 * ss_average + 1.0))
        return ss_average, multiplicity

    def undo_state_average(self):
        obj = lib.view(
            self, lib.drop_class(self.__class__, StateAverageFCISolver))
        del obj.weights
        del obj.e_states
        return obj


class StateAverageGASCI(addons.StateAverageMCSCFSolver):
    """GAS multistate object carrying weights and root-resolved results.

    The weights define fixed-orbital RDM and generalized-Fock
    postprocessing.  Individual root energies remain unchanged.
    """

    __name_mixin__ = "StateAverage"

    def __init__(self, mc, fcisolver):
        self.__dict__.update(mc.__dict__)
        self.fcisolver = fcisolver

    @property
    def weights(self):
        return self.fcisolver.weights

    @weights.setter
    def weights(self, value):
        self.fcisolver.weights = _validate_state_weights(value)

    @property
    def e_average(self):
        return float(numpy.dot(self.weights, self.fcisolver.e_states))

    @property
    def e_states(self):
        return self.fcisolver.e_states

    def undo_state_average(self):
        obj = lib.view(
            self, lib.drop_class(self.__class__, StateAverageGASCI))
        if isinstance(self.fcisolver, StateAverageFCISolver):
            obj.fcisolver = self.fcisolver.undo_state_average()
        return obj

    def _finalize(self):
        from pyscf.mcscf import gasci

        return gasci.GASCI._finalize(self)


def state_average(mc, weights=(0.5, 0.5), wfnsym=None):
    """Attach PySCF-style state weights to a GASCI calculation.

    GASCI orbitals remain fixed, and the individual root energies are
    unchanged.  The weights define averaged RDMs and generalized Fock
    matrices for postprocessing.
    """

    if wfnsym is not None:
        raise NotImplementedError(
            "GAS wavefunction symmetry filtering is not implemented")
    weights = _validate_state_weights(weights)

    if isinstance(mc, StateAverageGASCI):
        mc = mc.undo_state_average()
    fcisolver = mc.fcisolver
    if isinstance(fcisolver, addons.StateAverageFCISolver):
        fcisolver = fcisolver.undo_state_average()
    fcisolver = lib.set_class(
        StateAverageFCISolver(fcisolver, weights, None),
        (StateAverageFCISolver, fcisolver.__class__))
    return lib.set_class(
        StateAverageGASCI(mc, fcisolver),
        (StateAverageGASCI, mc.__class__))


def state_average_(mc, weights=(0.5, 0.5), wfnsym=None):
    """In-place version of :func:`state_average`."""

    weighted = state_average(mc, weights, wfnsym)
    mc.__class__ = weighted.__class__
    mc.__dict__ = weighted.__dict__
    return mc


def _integer_rows_with_info(values, ncolumn, label):
    """Return canonical integer rows and their normalization metadata."""

    arr = _exact_integer_array(values, label)
    if arr.ndim != 2 or arr.shape[1] != int(ncolumn):
        raise ValueError(
            "%s must have shape (nrow, %d)" % (label, int(ncolumn)))
    if arr.shape[0] == 0:
        raise ValueError("%s must contain at least one row" % label)
    original = numpy.ascontiguousarray(arr, dtype=numpy.int32)
    canonical = numpy.ascontiguousarray(
        numpy.unique(original, axis=0), dtype=numpy.int32)
    return canonical, {
        "input_nrow": int(original.shape[0]),
        "canonical_nrow": int(canonical.shape[0]),
        "duplicates_removed": int(original.shape[0] - canonical.shape[0]),
        "order_changed": not (
            original.shape == canonical.shape and
            numpy.array_equal(original, canonical)),
        "canonical_rows": canonical,
    }


def _integer_rows(values, ncolumn, label):
    """Return non-empty, unique, lexicographically sorted integer rows."""

    return _integer_rows_with_info(values, ncolumn, label)[0]


def normalize_blocks(blocks, ngas):
    """Return the canonical sorted spin-supergroup set D."""

    return _integer_rows(blocks, 2 * int(ngas), "spin-supergroups")


def normalize_supergroups(supergroups, ngas):
    """Return a canonical sorted spin-free supergroup set G."""

    return _integer_rows(supergroups, int(ngas), "supergroups")


def normalize_cumulative_occ(bounds, ngas):
    """Validate cumulative occupation bounds without reordering GAS rows."""

    arr = _exact_integer_array(bounds, "cumulative occupation bounds")
    if arr.ndim != 2 or arr.shape != (int(ngas), 2):
        raise ValueError(
            "cumulative occupation bounds must have shape (%d, 2)" %
            int(ngas))
    return numpy.ascontiguousarray(arr, dtype=numpy.int32)


def _supergroup_block_count(gas_orbs, na, row):
    """Count alpha partitions of one supergroup without materializing D."""

    counts = [0] * (int(na) + 1)
    counts[0] = 1
    for norb, total in zip(gas_orbs, row):
        lo = max(0, int(total) - int(norb))
        hi = min(int(norb), int(total))
        updated = [0] * (int(na) + 1)
        for occupied, count in enumerate(counts):
            if count == 0:
                continue
            for alpha in range(lo, hi + 1):
                if occupied + alpha <= na:
                    updated[occupied + alpha] += count
        counts = updated
    return counts[int(na)]


def blocks_from_supergroups(gas_orbs, nelec, supergroups):
    """Expand a spin-free supergroup set G into canonical block set D."""

    gas_orbs = _integer_vector(gas_orbs, "gas_orbs")
    na, nb = fci_addons._unpack_nelec(nelec)
    ngas = len(gas_orbs)
    if ngas <= 0 or ngas > GAS_MAX_NGAS:
        raise ValueError("number of GAS spaces exceeds C kernel limit")
    if any(n <= 0 or n > GAS_MAX_LOCAL_ORB for n in gas_orbs):
        raise ValueError("each GAS space must have 1..31 orbitals")
    if sum(gas_orbs) > GAS_MAX_ORB:
        raise ValueError("total active orbitals exceed 63")
    if na < 0 or nb < 0 or na > sum(gas_orbs) or nb > sum(gas_orbs):
        raise ValueError("invalid alpha/beta electron count")

    supergroups = normalize_supergroups(supergroups, ngas)
    capacities = 2 * numpy.asarray(gas_orbs, dtype=numpy.int32)
    if numpy.any(supergroups < 0):
        raise ValueError("supergroup occupations must be non-negative")
    if numpy.any(supergroups > capacities):
        raise ValueError("supergroup occupation exceeds local GAS capacity")
    if numpy.any(supergroups.sum(axis=1) != na + nb):
        raise ValueError(
            "every supergroup must sum to the requested electron count")

    nblock = sum(
        _supergroup_block_count(gas_orbs, na, row)
        for row in supergroups)
    if nblock <= 0:
        raise ValueError("supergroups generate no legal spin-supergroups")
    if nblock > GAS_MAX_BLOCK:
        raise ValueError("number of blocks exceeds C int32 input limit")

    blocks = numpy.empty((nblock, 2 * ngas), dtype=numpy.int32)
    cursor = 0
    for row in supergroups:
        lower = numpy.maximum(0, row - numpy.asarray(gas_orbs))
        upper = numpy.minimum(numpy.asarray(gas_orbs), row)
        suffix_lower = numpy.zeros(ngas + 1, dtype=numpy.int32)
        suffix_upper = numpy.zeros(ngas + 1, dtype=numpy.int32)
        for ig in range(ngas - 1, -1, -1):
            suffix_lower[ig] = suffix_lower[ig + 1] + lower[ig]
            suffix_upper[ig] = suffix_upper[ig + 1] + upper[ig]

        alpha = numpy.empty(ngas, dtype=numpy.int32)

        def expand(ig, remaining):
            nonlocal cursor
            if ig == ngas:
                if remaining == 0:
                    blocks[cursor, :ngas] = alpha
                    blocks[cursor, ngas:] = row - alpha
                    cursor += 1
                return
            lo = max(int(lower[ig]),
                     int(remaining - suffix_upper[ig + 1]))
            hi = min(int(upper[ig]),
                     int(remaining - suffix_lower[ig + 1]))
            for occupied in range(lo, hi + 1):
                alpha[ig] = occupied
                expand(ig + 1, remaining - occupied)

        expand(0, na)

    if cursor != nblock:
        raise RuntimeError("internal supergroup expansion count mismatch")
    blocks = normalize_blocks(blocks, ngas)
    check_kernel_limits(gas_orbs, nelec, blocks)
    return blocks


def supergroups_from_cumulative_occ(gas_orbs, nelec, bounds,
                                    allow_empty=False):
    """Expand cumulative occupation bounds into canonical supergroups G."""

    gas_orbs = _integer_vector(gas_orbs, "gas_orbs")
    ngas = len(gas_orbs)
    if ngas <= 0 or ngas > GAS_MAX_NGAS:
        raise ValueError("number of GAS spaces exceeds C kernel limit")
    minimum = 0 if allow_empty else 1
    if any(n < minimum or n > GAS_MAX_LOCAL_ORB for n in gas_orbs):
        if allow_empty:
            raise ValueError("each RAS space must have 0..31 orbitals")
        raise ValueError("each GAS space must have 1..31 orbitals")
    if sum(gas_orbs) <= 0:
        raise ValueError("at least one active orbital is required")
    if sum(gas_orbs) > GAS_MAX_ORB:
        raise ValueError("total active orbitals exceed 63")

    na, nb = fci_addons._unpack_nelec(nelec)
    nelectron = na + nb
    if na < 0 or nb < 0 or na > sum(gas_orbs) or nb > sum(gas_orbs):
        raise ValueError("invalid alpha/beta electron count")
    bounds = normalize_cumulative_occ(bounds, ngas)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    cumulative_capacity = 2 * numpy.cumsum(
        numpy.asarray(gas_orbs, dtype=numpy.int32))

    if numpy.any(lower < 0):
        raise ValueError("cumulative occupations must be non-negative")
    if numpy.any(lower > upper):
        raise ValueError("cumulative minimum exceeds maximum")
    if numpy.any(lower[1:] < lower[:-1]) or numpy.any(
            upper[1:] < upper[:-1]):
        raise ValueError("cumulative bounds must be non-decreasing")
    if numpy.any(upper > cumulative_capacity):
        raise ValueError("cumulative occupation exceeds orbital capacity")
    if numpy.any(upper > nelectron):
        raise ValueError("cumulative occupation exceeds active electrons")
    if int(lower[-1]) != nelectron or int(upper[-1]) != nelectron:
        raise ValueError(
            "the final cumulative bound must equal the active electron count")

    capacities = tuple(2 * int(value) for value in gas_orbs)
    suffix_capacity = [0] * (ngas + 1)
    for igas in range(ngas - 1, -1, -1):
        suffix_capacity[igas] = (
            suffix_capacity[igas + 1] + capacities[igas])

    def occupation_range(igas, cumulative):
        lo = max(0, int(lower[igas]) - cumulative,
                 nelectron - cumulative - suffix_capacity[igas + 1])
        hi = min(capacities[igas], int(upper[igas]) - cumulative,
                 nelectron - cumulative)
        return lo, hi

    @lru_cache(maxsize=None)
    def count(igas, cumulative):
        if igas == ngas:
            return int(cumulative == nelectron)
        lo, hi = occupation_range(igas, cumulative)
        if lo > hi:
            return 0
        return sum(count(igas + 1, cumulative + occupation)
                   for occupation in range(lo, hi + 1))

    nsupergroup = count(0, 0)
    if nsupergroup <= 0:
        raise ValueError("cumulative bounds generate no legal supergroups")
    if nsupergroup > GAS_MAX_BLOCK:
        raise ValueError("number of generated supergroups exceeds int32 limit")

    supergroups = numpy.empty((nsupergroup, ngas), dtype=numpy.int32)
    row = numpy.empty(ngas, dtype=numpy.int32)
    cursor = 0

    def expand(igas, cumulative):
        nonlocal cursor
        if igas == ngas:
            if cumulative == nelectron:
                supergroups[cursor] = row
                cursor += 1
            return
        lo, hi = occupation_range(igas, cumulative)
        for occupation in range(lo, hi + 1):
            if count(igas + 1, cumulative + occupation) == 0:
                continue
            row[igas] = occupation
            expand(igas + 1, cumulative + occupation)

    expand(0, 0)
    if cursor != nsupergroup:
        raise RuntimeError("internal cumulative expansion count mismatch")
    return normalize_supergroups(supergroups, ngas)


def blocks_from_cumulative_occ(gas_orbs, nelec, bounds):
    """Convert cumulative occupation bounds to canonical D."""

    supergroups = supergroups_from_cumulative_occ(
        gas_orbs, nelec, bounds)
    return blocks_from_supergroups(gas_orbs, nelec, supergroups)


def _normalize_ras_restr(gas_restr):
    if not isinstance(gas_restr, Mapping):
        raise TypeError(
            "ras gas_restr must be a mapping with max_holes and "
            "max_particles")
    required = {"max_holes", "max_particles"}
    missing = required.difference(gas_restr)
    extra = set(gas_restr).difference(required)
    if missing:
        raise ValueError("missing RAS restriction keys: %s" %
                         ", ".join(sorted(missing)))
    if extra:
        raise ValueError("unknown RAS restriction keys: %s" %
                         ", ".join(sorted(extra)))
    max_holes = gas_restr["max_holes"]
    max_particles = gas_restr["max_particles"]
    if not lib.isinteger(max_holes):
        raise TypeError("max_holes must be an integer")
    if not lib.isinteger(max_particles):
        raise TypeError("max_particles must be an integer")
    max_holes = int(max_holes)
    max_particles = int(max_particles)
    if max_holes < 0 or max_particles < 0:
        raise ValueError("RAS hole and particle limits must be non-negative")
    return max_holes, max_particles


def _ras_spec(gas_orbs, nelec, gas_restr):
    """Return non-empty kernel spaces, canonical D, and RAS metadata."""

    gas_orbs = _integer_vector(
        gas_orbs, "ras gas_orbs", length=3)
    if any(value < 0 or value > GAS_MAX_LOCAL_ORB for value in gas_orbs):
        raise ValueError("each RAS space must have 0..31 orbitals")
    if sum(gas_orbs) <= 0:
        raise ValueError("at least one RAS orbital is required")
    if sum(gas_orbs) > GAS_MAX_ORB:
        raise ValueError("total active orbitals exceed 63")

    max_holes, max_particles = _normalize_ras_restr(gas_restr)
    nras1, nras2, nras3 = gas_orbs
    if max_holes > 2 * nras1:
        raise ValueError("max_holes exceeds the RAS1 electron capacity")
    if max_particles > 2 * nras3:
        raise ValueError("max_particles exceeds the RAS3 electron capacity")

    nelectron = sum(fci_addons._unpack_nelec(nelec))
    cumulative = numpy.array([
        [max(0, 2 * nras1 - max_holes), min(2 * nras1, nelectron)],
        [max(0, nelectron - max_particles),
         min(2 * (nras1 + nras2), nelectron)],
        [nelectron, nelectron],
    ], dtype=numpy.int32)
    cumulative[1, 0] = max(cumulative[1, 0], cumulative[0, 0])
    supergroups = supergroups_from_cumulative_occ(
        gas_orbs, nelec, cumulative, allow_empty=True)

    nonempty = numpy.asarray(gas_orbs, dtype=numpy.int32) > 0
    kernel_orbs = tuple(int(value) for value in
                        numpy.asarray(gas_orbs)[nonempty])
    kernel_supergroups = normalize_supergroups(
        supergroups[:, nonempty], len(kernel_orbs))
    blocks = blocks_from_supergroups(
        kernel_orbs, nelec, kernel_supergroups)
    return kernel_orbs, blocks, {
        "ras_orbs": gas_orbs,
        "max_holes": max_holes,
        "max_particles": max_particles,
        "cumulative_bounds": cumulative,
        "canonical_supergroups": kernel_supergroups,
        "conceptual_supergroups": supergroups,
        "empty_spaces_removed": tuple(
            index + 1 for index, size in enumerate(gas_orbs) if size == 0),
    }


def _spin_supergroup_spec(gas_orbs, nelec, gas_restr):
    gas_orbs = tuple(int(value) for value in gas_orbs)
    blocks, info = _integer_rows_with_info(
        gas_restr, 2 * len(gas_orbs), "spin-supergroups")
    check_kernel_limits(gas_orbs, nelec, blocks)
    info["canonical_spin_supergroups"] = blocks
    return gas_orbs, blocks, info


def _supergroup_spec(gas_orbs, nelec, gas_restr):
    gas_orbs = tuple(int(value) for value in gas_orbs)
    supergroups, info = _integer_rows_with_info(
        gas_restr, len(gas_orbs), "supergroups")
    blocks = blocks_from_supergroups(gas_orbs, nelec, supergroups)
    info.update({
        "canonical_supergroups": supergroups,
        "canonical_spin_supergroups": blocks,
    })
    return gas_orbs, blocks, info


def _cumulative_occ_spec(gas_orbs, nelec, gas_restr):
    gas_orbs = tuple(int(value) for value in gas_orbs)
    bounds = normalize_cumulative_occ(gas_restr, len(gas_orbs))
    supergroups = supergroups_from_cumulative_occ(
        gas_orbs, nelec, bounds)
    blocks = blocks_from_supergroups(gas_orbs, nelec, supergroups)
    return gas_orbs, blocks, {
        "cumulative_bounds": bounds,
        "canonical_supergroups": supergroups,
        "canonical_spin_supergroups": blocks,
    }


_GAS_RESTR_CONVERTERS = {
    GAS_RESTR_SPIN_SUPERGROUP: _spin_supergroup_spec,
    GAS_RESTR_SUPERGROUP: _supergroup_spec,
    GAS_RESTR_CUMULATIVE_OCC: _cumulative_occ_spec,
    GAS_RESTR_RAS: _ras_spec,
}


def normalize_gas_spec(gas_orbs, nelec, gas_restr=None,
                       gas_restr_type=GAS_RESTR_SPIN_SUPERGROUP,
                       return_info=False):
    """Normalize one public GAS specification to kernel spaces and D."""

    try:
        if not isinstance(gas_restr_type, str):
            raise TypeError("gas_restr_type must be a string")
        if gas_restr_type not in _GAS_RESTR_CONVERTERS:
            supported = ", ".join(sorted(_GAS_RESTR_CONVERTERS))
            raise NotImplementedError(
                "unsupported gas_restr_type %r; supported types: %s" %
                (gas_restr_type, supported))

        expected_length = 3 if gas_restr_type == GAS_RESTR_RAS else None
        user_gas_orbs = _integer_vector(
            gas_orbs, "gas_orbs", length=expected_length)
        info = {"gas_restr_type": gas_restr_type,
                "user_gas_orbs": user_gas_orbs}

        if gas_restr is None:
            if gas_restr_type != GAS_RESTR_SPIN_SUPERGROUP:
                raise ValueError(
                    "gas_restr is required for gas_restr_type %r" %
                    gas_restr_type)
            if len(user_gas_orbs) != 1:
                raise ValueError("gas_restr is required for multi-space GAS")
            gas_orbs = user_gas_orbs
            blocks = numpy.asarray(
                [fci_addons._unpack_nelec(nelec)], dtype=numpy.int32)
            check_kernel_limits(gas_orbs, nelec, blocks)
            info.update({
                "kernel_gas_orbs": gas_orbs,
                "canonical_spin_supergroups": blocks,
                "input_nrow": 1,
                "canonical_nrow": 1,
                "duplicates_removed": 0,
                "order_changed": False,
            })
        else:
            converter = _GAS_RESTR_CONVERTERS[gas_restr_type]
            gas_orbs, blocks, converter_info = converter(
                user_gas_orbs, nelec, gas_restr)
            info.update(converter_info)
            info["kernel_gas_orbs"] = gas_orbs
    except (TypeError, ValueError, NotImplementedError) as err:
        usage = gas_restr_usage(
            gas_restr_type if isinstance(gas_restr_type, str) else None)
        raise type(err)("%s\n%s" % (err, usage)) from err

    if return_info:
        return gas_orbs, blocks, info
    return gas_orbs, blocks


def _normalize_gaslst(gaslst, gas_orbs, nmo, base):
    """Validate nested GAS orbital indices and return a zero-based flat list."""

    if not lib.isinteger(base):
        raise TypeError("base must be an integer")
    base = int(base)
    if base not in (0, 1):
        raise ValueError("base must be 0 or 1")
    if not isinstance(gaslst, (tuple, list)):
        raise TypeError(
            "gaslst must be a sequence containing one orbital list per GAS "
            "subspace")
    if len(gaslst) != len(gas_orbs):
        raise ValueError(
            "gaslst must contain %d subspace lists matching gas_orbs %s" %
            (len(gas_orbs), tuple(gas_orbs)))

    flat = []
    for igas, (indices, size) in enumerate(zip(gaslst, gas_orbs)):
        values = _exact_integer_array(
            indices, "gaslst[%d]" % igas)
        if values.ndim != 1:
            raise ValueError("gaslst[%d] must be one-dimensional" % igas)
        if values.size != int(size):
            raise ValueError(
                "gaslst[%d] must contain %d orbital indices" %
                (igas, int(size)))
        flat.extend(int(value) - int(base) for value in values)

    if len(set(flat)) != len(flat):
        raise ValueError("gaslst orbital indices must not contain duplicates")
    if any(index < 0 or index >= int(nmo) for index in flat):
        raise ValueError(
            "gaslst orbital indices are outside the valid range for base=%d" %
            int(base))
    return flat


def sort_mo(mc, mo_coeff, gaslst, base=1):
    """Pick and order orbitals for each GAS subspace.

    ``gaslst`` is nested, with one orbital-index list per GAS subspace.  The
    order of subspaces and the order within each list are preserved.  As in
    :func:`pyscf.mcscf.addons.sort_mo`, ``base`` selects zero- or one-based
    user indices; internal indices are always zero based.
    """

    mo_array = numpy.asarray(mo_coeff)
    if mo_array.ndim != 2:
        raise ValueError("mo_coeff must be a two-dimensional array")
    gas_orbs = ((int(mc.ncas),) if mc.gas_orbs is None else
                _integer_vector(mc.gas_orbs, "gas_orbs"))
    if sum(gas_orbs) != int(mc.ncas):
        raise ValueError("sum(gas_orbs) must equal ncas")
    active = _normalize_gaslst(
        gaslst, gas_orbs, mo_array.shape[1], base)
    return addons.sort_mo(mc, mo_coeff, active, base=0)


def normalize_gas_restr(gas_orbs, nelec, gas_restr=None,
                        gas_restr_type=GAS_RESTR_SPIN_SUPERGROUP):
    """Convert one public restriction type to canonical C-kernel set D.

    ``spin-supergroup`` accepts D directly.  ``supergroup`` accepts spin-free
    occupation rows G.  ``cumulative-occ`` generates G from accumulated
    electron bounds.  ``ras`` maps named hole/particle limits through the
    cumulative representation.  Every route expands to canonical D before
    any downstream GAS calculation.
    """

    return normalize_gas_spec(
        gas_orbs, nelec, gas_restr, gas_restr_type)[1]


def is_spin_complete(gas_orbs, nelec, blocks):
    """Return whether every retained spatial supergroup is spin complete.

    A spin-free supergroup fixes the total occupation of each GAS subspace.
    Its explicit D representation must contain every compatible alpha/beta
    partition before the determinant space is invariant under ``S^2``.
    """

    gas_orbs = _integer_vector(gas_orbs, "gas_orbs")
    blocks = normalize_blocks(blocks, len(gas_orbs))
    ngas = len(gas_orbs)
    supergroups = numpy.unique(
        blocks[:, :ngas] + blocks[:, ngas:], axis=0)
    complete = blocks_from_supergroups(gas_orbs, nelec, supergroups)
    actual_rows = {tuple(int(value) for value in row) for row in blocks}
    complete_rows = {tuple(int(value) for value in row) for row in complete}
    return actual_rows == complete_rows


def _sector_string_count(norb, occ):
    total = 1
    for n, k in zip(norb, occ):
        total *= math.comb(int(n), int(k))
    return total


def check_kernel_limits(norb, nelec, blocks):
    """Preflight the public C-kernel limits before allocating gas_space_t."""

    norb = _integer_vector(norb, "gas_orbs")
    na, nb = fci_addons._unpack_nelec(nelec)
    ngas = len(norb)
    if ngas <= 0 or ngas > GAS_MAX_NGAS:
        raise ValueError("number of GAS spaces exceeds C kernel limit")
    if any(n <= 0 or n > GAS_MAX_LOCAL_ORB for n in norb):
        raise ValueError("each GAS space must have 1..31 orbitals")
    if sum(norb) > GAS_MAX_ORB:
        raise ValueError("total active orbitals exceed 63")
    if na < 0 or nb < 0 or na > sum(norb) or nb > sum(norb):
        raise ValueError("invalid alpha/beta electron count")

    blocks = normalize_blocks(blocks, ngas)
    if blocks.shape[0] > GAS_MAX_BLOCK:
        raise ValueError("number of blocks exceeds C int32 input limit")
    alpha = blocks[:, :ngas]
    beta = blocks[:, ngas:]
    if numpy.any(blocks < 0):
        raise ValueError("block occupations must be non-negative")
    if numpy.any(alpha > numpy.asarray(norb)):
        raise ValueError("alpha block occupation exceeds local GAS orbitals")
    if numpy.any(beta > numpy.asarray(norb)):
        raise ValueError("beta block occupation exceeds local GAS orbitals")
    if numpy.any(alpha.sum(axis=1) != na):
        raise ValueError("every alpha block must sum to the requested nelec")
    if numpy.any(beta.sum(axis=1) != nb):
        raise ValueError("every beta block must sum to the requested nelec")

    sectors = {tuple(row) for row in alpha.tolist()}
    sectors.update(tuple(row) for row in beta.tolist())
    if len(sectors) > GAS_MAX_SECTOR:
        raise ValueError("number of GAS sectors exceeds uint16_t kernel limit")
    for occ in sectors:
        if _sector_string_count(norb, occ) > GAS_MAX_DET:
            raise ValueError("sector string count exceeds uint32_t kernel limit")

    ndet = 0
    for aocc, bocc in zip(alpha, beta):
        ndet += _sector_string_count(norb, aocc) * _sector_string_count(norb, bocc)
        if ndet > GAS_MAX_DET:
            raise ValueError("determinant count exceeds uint32_t kernel limit")

    return {
        "ngas": ngas,
        "norb_total": sum(norb),
        "na": na,
        "nb": nb,
        "nblock": int(blocks.shape[0]),
        "nsector_upper": len(sectors),
        "ndet_estimate": int(ndet),
    }


def pair_index(p, q):
    hi = max(int(p), int(q))
    lo = min(int(p), int(q))
    return hi * (hi + 1) // 2 + lo
