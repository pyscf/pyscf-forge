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

"""GAS-aware FCISolver bindings for the frozen GAS FCI C kernels."""

import ctypes

import numpy

from pyscf import ao2mo
from pyscf import lib as pyscf_lib
from pyscf.fci import addons as fci_addons
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.lib import logger

from pyscf.mcscf import _gaslib
from pyscf.mcscf import addons_gas


# A GAS pspace column currently costs one full C-kernel Hamiltonian product.
# Keep exact small-space diagonalization bounded until a dedicated C pspace
# builder exists.
GAS_PSPACE_MATVEC_MAX = 64
GAS_SPIN_PSPACE_MATVEC_MAX = 400
GAS_INIT_NOISE_NORM = 1e-3
GAS_INIT_NOISE_CHUNK = 1 << 20


def _lowest_hdiag_addresses(hdiag, count):
    """Return addresses of the lowest Hamiltonian diagonal elements."""

    hdiag = numpy.asarray(hdiag).reshape(-1)
    count = int(count)
    if count < 1 or count > hdiag.size:
        raise ValueError("count must be between 1 and the diagonal size")
    if hdiag.size <= count:
        return numpy.arange(hdiag.size, dtype=numpy.intp)
    try:
        return numpy.argpartition(hdiag, count - 1)[:count].copy()
    except AttributeError:
        return numpy.argsort(hdiag)[:count].copy()


def _add_distributed_noise(vector, root, noise_norm=GAS_INIT_NOISE_NORM):
    """Add a reproducible low-norm component over the entire GAS vector."""

    vector = numpy.asarray(vector)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("distributed noise requires a nonempty vector")
    noise_norm = float(noise_norm)
    if noise_norm < 0.0 or not numpy.isfinite(noise_norm):
        raise ValueError("noise_norm must be finite and nonnegative")
    if noise_norm == 0.0:
        return vector

    mask = (1 << 64) - 1
    seed_value = ((int(root) + 1) * 0x9E3779B97F4A7C15) & mask
    seed = numpy.uint64(seed_value)
    scale = noise_norm / numpy.sqrt(float(vector.size))
    for start in range(0, vector.size, GAS_INIT_NOISE_CHUNK):
        stop = min(vector.size, start + GAS_INIT_NOISE_CHUNK)
        values = numpy.arange(start, stop, dtype=numpy.uint64) + seed
        values ^= values >> numpy.uint64(30)
        values *= numpy.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> numpy.uint64(27)
        values *= numpy.uint64(0x94D049BB133111EB)
        values ^= values >> numpy.uint64(31)
        signs = numpy.where(
            values & numpy.uint64(1), 1.0, -1.0)
        vector[start:stop] += scale * signs
    return vector


def _as_c_double(array, shape=None):
    source = numpy.asarray(array)
    if numpy.iscomplexobj(source):
        raise TypeError("the GAS C kernels require real-valued arrays")
    arr = numpy.asarray(source, dtype=numpy.float64, order="C")
    if shape is not None and arr.shape != shape:
        raise ValueError("expected shape %s, got %s" % (shape, arr.shape))
    return numpy.ascontiguousarray(arr, dtype=numpy.float64)


def _as_pair_matrix(eri, norb):
    norb = int(norb)
    npair = norb * (norb + 1) // 2
    source = numpy.asarray(eri)
    if numpy.iscomplexobj(source):
        raise TypeError("the GAS C kernels require real-valued integrals")
    arr = numpy.asarray(source, dtype=numpy.float64, order="C")
    if arr.shape == (npair, npair):
        return numpy.ascontiguousarray(arr, dtype=numpy.float64)
    if arr.shape == (norb, norb, norb, norb):
        out = numpy.empty((npair, npair), dtype=numpy.float64)
        for p in range(norb):
            for q in range(p + 1):
                pq = addons_gas.pair_index(p, q)
                for r in range(norb):
                    for s in range(r + 1):
                        rs = addons_gas.pair_index(r, s)
                        out[pq, rs] = arr[p, q, r, s]
        return out
    restored = ao2mo.restore(4, arr, norb)
    restored = numpy.asarray(restored, dtype=numpy.float64, order="C")
    if restored.shape != (npair, npair):
        raise ValueError("could not convert ERI to pair-matrix layout")
    return numpy.ascontiguousarray(restored, dtype=numpy.float64)


def _check_status(status, name):
    if status != _gaslib.GAS_SUCCESS:
        raise RuntimeError("%s failed with status %d" % (name, status))


def absorb_h1e(h1e, eri, norb, nelec, fac=0.5):
    """Return the absorbed two-electron Hamiltonian in pair-matrix layout."""

    h1e = _as_c_double(h1e, (int(norb), int(norb)))
    eri = _as_pair_matrix(eri, norb)
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)
    return _as_pair_matrix(h2e, norb)


def _fci_block_addresses(gas, block):
    """Return full-FCI alpha/beta addresses for one GAS block."""

    alpha_strings = numpy.fromiter(
        (gas.addr2str(block["sa"], address)
         for address in range(block["na"])),
        dtype=numpy.int64, count=block["na"])
    beta_strings = numpy.fromiter(
        (gas.addr2str(block["sb"], address)
         for address in range(block["nb"])),
        dtype=numpy.int64, count=block["nb"])
    alpha_addresses = cistring.strs2addr(
        gas.norb_total, gas.nelec[0], alpha_strings)
    beta_addresses = cistring.strs2addr(
        gas.norb_total, gas.nelec[1], beta_strings)
    return alpha_addresses, beta_addresses


def gas2fci(fcivec, gas):
    """Embed a packed GAS CI vector in the full-FCI determinant tensor.

    The returned array uses the standard ``direct_spin1`` rectangular
    ``(nstr_alpha, nstr_beta)`` ordering.  Determinants outside the GAS space
    are represented by zero coefficients.

    This conversion is intended for interoperability and validation when the
    corresponding full-FCI tensor is small enough to construct explicitly.
    """

    ci = numpy.asarray(fcivec)
    if ci.size != gas.ndet:
        raise ValueError("CI vector size does not match GAS determinant count")
    ci = ci.reshape(-1)

    nstr_alpha = cistring.num_strings(gas.norb_total, gas.nelec[0])
    nstr_beta = cistring.num_strings(gas.norb_total, gas.nelec[1])
    fci = numpy.zeros((nstr_alpha, nstr_beta), dtype=ci.dtype)
    for block in gas.block_descriptors():
        alpha_addresses, beta_addresses = _fci_block_addresses(gas, block)
        size = block["na"] * block["nb"]
        gas_block = ci[block["offset"]:block["offset"] + size].reshape(
            block["na"], block["nb"])
        fci[numpy.ix_(alpha_addresses, beta_addresses)] = gas_block
    return fci


def fci2gas(fcivec, gas):
    """Project a full-FCI determinant tensor into packed GAS ordering.

    Coefficients belonging to determinants outside the GAS space are
    discarded.  The returned array is one-dimensional and follows the
    canonical GAS block ordering used by the C kernels.
    """

    nstr_alpha = cistring.num_strings(gas.norb_total, gas.nelec[0])
    nstr_beta = cistring.num_strings(gas.norb_total, gas.nelec[1])
    ci = numpy.asarray(fcivec)
    if ci.size != nstr_alpha * nstr_beta:
        raise ValueError("CI vector size does not match full FCI determinant count")
    ci = ci.reshape(nstr_alpha, nstr_beta)

    gas_ci = numpy.empty(gas.ndet, dtype=ci.dtype)
    for block in gas.block_descriptors():
        alpha_addresses, beta_addresses = _fci_block_addresses(gas, block)
        size = block["na"] * block["nb"]
        gas_ci[block["offset"]:block["offset"] + size] = ci[
            numpy.ix_(alpha_addresses, beta_addresses)].reshape(-1)
    return gas_ci


def _spin_penalty_parameters(solver, norb, nelec):
    """Return validated spin-penalty parameters or ``None``."""

    if not hasattr(solver, "ss_penalty"):
        return None
    shift = float(solver.ss_penalty)
    if not numpy.isfinite(shift) or shift < 0.0:
        raise ValueError("spin penalty shift must be finite and nonnegative")
    na, nb = fci_addons._unpack_nelec(nelec)
    sz = 0.5 * abs(na - nb)
    minimum = sz * (sz + 1.0)
    target_value = getattr(solver, "ss_value", None)
    target = minimum if target_value is None else float(target_value)
    if not numpy.isfinite(target):
        raise ValueError("target S^2 must be finite")
    if target < minimum - 1e-10:
        raise ValueError(
            "target S^2 is below the minimum allowed by Nalpha-Nbeta")
    nelectron = na + nb
    max_unpaired = min(nelectron, 2 * int(norb) - nelectron)
    maximum_spin = 0.5 * max_unpaired
    spin_values = numpy.arange(sz, maximum_spin + 0.5, 1.0)
    eigenvalues = spin_values * (spin_values + 1.0)
    if not numpy.any(numpy.abs(eigenvalues - target) < 1e-10):
        raise ValueError(
            "target S^2 is not an allowed S(S+1) value for this space")
    return shift, target, minimum, tuple(float(x) for x in eigenvalues)


class _GasSpinPlan:
    """Reusable block-sparse ``S^2`` contraction plan.

    Raw GAS one-electron links retain excitation direction.  Pairing an alpha
    ``q -> p`` link with the reverse beta ``p -> q`` link applies the
    opposite-spin exchange term without expanding the CI vector into the full
    CAS tensor.  The numerical gather/scatter is delegated to PySCF's C-backed
    ``take_2d`` and ``takebak_2d`` helpers.
    """

    def __init__(self, gas):
        if gas.links_are_compressed():
            raise ValueError("S^2 planning requires raw GAS link tables")
        self.ndet = gas.ndet
        na, nb = gas.nelec
        sz = 0.5 * (na - nb)
        self.diagonal = sz * (sz + 1.0) + nb
        self.tasks = self._build_tasks(gas)

    @staticmethod
    def _table_maps(table):
        maps = {}
        nsrc = int(table.nsrc)
        nlink = int(table.nlink)
        for source in range(nsrc):
            for k in range(nlink):
                entry = table.link[source * nlink + k]
                op = int(entry.op)
                item = maps.setdefault(op, ([], [], []))
                item[0].append(source)
                item[1].append(int(entry.addr))
                item[2].append(int(entry.sign))
        return {
            op: (
                numpy.asarray(values[0], dtype=numpy.int32),
                numpy.asarray(values[1], dtype=numpy.int32),
                numpy.asarray(values[2], dtype=numpy.float64),
            )
            for op, values in maps.items()
        }

    @classmethod
    def _build_tasks(cls, gas):
        cgas = gas._gas
        table_maps = {}
        outgoing = [[] for _ in range(gas.nsector)]
        for source in range(gas.nsector):
            row = cgas.T.row[source]
            for i in range(int(row.n)):
                tid = int(row.off) + i
                destination = int(cgas.T.dst[tid])
                outgoing[source].append((destination, tid))
                table_maps[tid] = cls._table_maps(cgas.table[tid])

        blocks = list(gas.block_descriptors())
        block_by_sectors = {
            (block["sa"], block["sb"]): block for block in blocks
        }
        tasks = []
        for source in blocks:
            for alpha_destination, alpha_tid in outgoing[source["sa"]]:
                alpha_maps = table_maps[alpha_tid]
                for beta_destination, beta_tid in outgoing[source["sb"]]:
                    destination = block_by_sectors.get(
                        (alpha_destination, beta_destination))
                    if destination is None:
                        continue
                    beta_maps = table_maps[beta_tid]
                    for op, alpha_map in alpha_maps.items():
                        p = op & 0xff
                        q = op >> 8
                        beta_map = beta_maps.get(q | (p << 8))
                        if beta_map is None:
                            continue
                        tasks.append((source, destination,
                                      alpha_map, beta_map))
        return tuple(tasks)

    def contract(self, fcivec):
        ci0 = _as_c_double(numpy.asarray(fcivec).reshape(-1))
        if ci0.size != self.ndet:
            raise ValueError(
                "CI vector size does not match GAS determinant count")
        ci1 = self.diagonal * ci0
        for source, destination, alpha, beta in self.tasks:
            source_matrix = ci0[
                source["offset"]:
                source["offset"] + source["na"] * source["nb"]
            ].reshape(source["na"], source["nb"])
            destination_matrix = ci1[
                destination["offset"]:
                destination["offset"] +
                destination["na"] * destination["nb"]
            ].reshape(destination["na"], destination["nb"])
            alpha_source, alpha_destination, alpha_sign = alpha
            beta_source, beta_destination, beta_sign = beta
            intermediate = pyscf_lib.take_2d(
                source_matrix, alpha_source, beta_source)
            intermediate *= alpha_sign[:, None]
            intermediate *= beta_sign
            pyscf_lib.takebak_2d(
                destination_matrix, -intermediate,
                alpha_destination, beta_destination)
        return ci1.reshape(numpy.asarray(fcivec).shape)

    def diagonal_vector(self):
        """Return the exact determinant-basis diagonal of ``S^2``."""

        diagonal = numpy.full(self.ndet, self.diagonal, dtype=numpy.float64)
        for source, destination, alpha, beta in self.tasks:
            if source["offset"] != destination["offset"]:
                continue
            alpha_source, alpha_destination, alpha_sign = alpha
            beta_source, beta_destination, beta_sign = beta
            alpha_mask = alpha_source == alpha_destination
            beta_mask = beta_source == beta_destination
            if not numpy.any(alpha_mask) or not numpy.any(beta_mask):
                continue
            ia = alpha_source[alpha_mask]
            ib = beta_source[beta_mask]
            values = -numpy.multiply.outer(
                alpha_sign[alpha_mask], beta_sign[beta_mask])
            block = diagonal[
                source["offset"]:
                source["offset"] + source["na"] * source["nb"]
            ].reshape(source["na"], source["nb"])
            numpy.add.at(
                block,
                (numpy.repeat(ia, ib.size), numpy.tile(ib, ia.size)),
                values.reshape(-1))
        return diagonal

    def matrix(self):
        """Build the explicit ``S^2`` matrix for a bounded GAS space."""

        matrix = numpy.empty((self.ndet, self.ndet), dtype=numpy.float64)
        basis = numpy.zeros(self.ndet, dtype=numpy.float64)
        for column in range(self.ndet):
            basis.fill(0.0)
            basis[column] = 1.0
            matrix[:, column] = self.contract(basis)
        return 0.5 * (matrix + matrix.T)

    def project(self, vector, target, eigenvalues):
        """Project a trial vector onto one allowed total-spin eigenspace."""

        projected = _as_c_double(numpy.asarray(vector).reshape(-1)).copy()
        for eigenvalue in eigenvalues:
            if abs(eigenvalue - target) < 1e-10:
                continue
            projected = (
                numpy.asarray(self.contract(projected)).reshape(-1) -
                eigenvalue * projected) / (target - eigenvalue)
        return projected


class GasSpace:
    """Python owner for the C ``gas_space_t`` object.

    Use this low-level object as a context manager.  Normal calculations
    should construct it through :meth:`FCISolver.make_space` so that the
    solver's normalized restriction and loaded C library are reused.
    """

    def __init__(self, norb, nelec, blocks=None, lib=None, compress_links=False):
        self.norb = tuple(int(x) for x in norb)
        self.nelec = fci_addons._unpack_nelec(nelec)
        if blocks is None:
            if len(self.norb) != 1:
                raise ValueError("blocks are required for multi-space GAS")
            blocks = numpy.asarray([self.nelec], dtype=numpy.int32)
        self.blocks = addons_gas.normalize_blocks(blocks, len(self.norb))
        self.limits = addons_gas.check_kernel_limits(self.norb, self.nelec,
                                                     self.blocks)
        self.lib = lib or _gaslib.load_library()
        self._gas = _gaslib.GasSpaceStruct()
        self._norb_arr = numpy.ascontiguousarray(self.norb, dtype=numpy.int32)
        self._blocks_arr = numpy.ascontiguousarray(self.blocks.reshape(-1),
                                                   dtype=numpy.int32)
        status = self.lib.gas_space_from_blocks(
            ctypes.byref(self._gas),
            len(self.norb),
            self._norb_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.nelec[0],
            self.nelec[1],
            self.blocks.shape[0],
            self._blocks_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        _check_status(status, "gas_space_from_blocks")
        if compress_links:
            self.compress_links()

    @property
    def c_ptr(self):
        return ctypes.byref(self._gas)

    @property
    def ndet(self):
        """Number of determinants in the canonical GAS ordering."""

        return int(self._gas.ndet)

    @property
    def nsector(self):
        """Number of distinct alpha/beta string sectors."""

        return int(self._gas.nsector)

    @property
    def nblock(self):
        """Number of legal spin-supergroup blocks in D."""

        return int(self._gas.nblock)

    @property
    def ntable(self):
        """Number of one-electron link tables."""

        return int(self._gas.ntable)

    @property
    def norb_total(self):
        return int(self._gas.norb_tot)

    @property
    def memory_bytes(self):
        """Total memory owned by the C GAS-space object, in bytes."""

        return int(self.lib.gas_memory_bytes(self.c_ptr))

    def core_info(self):
        """Return the compact core state represented by ``gas_space_t``."""

        if self._gas is None:
            raise RuntimeError("GAS space is closed")
        ngas = int(self._gas.ngas)
        return {
            "ngas": ngas,
            "norb": tuple(int(self._gas.norb[i]) for i in range(ngas)),
            "start": tuple(int(self._gas.start[i]) for i in range(ngas)),
            "norb_total": int(self._gas.norb_tot),
            "nelec": (int(self._gas.na), int(self._gas.nb)),
            "ndet": int(self._gas.ndet),
            "nsector": int(self._gas.nsector),
            "nblock": int(self._gas.nblock),
            "ntable": int(self._gas.ntable),
        }

    def memory_report(self):
        """Return the C-space memory breakdown as a dictionary of bytes."""

        report = _gaslib.GasMemoryReport()
        self.lib.gas_memory_report(self.c_ptr, ctypes.byref(report))
        return {name: int(getattr(report, name)) for name, _ in report._fields_}

    def compress_links(self):
        """Convert raw link entries to the frozen compressed representation."""

        status = self.lib.gas_space_compress_links(self.c_ptr)
        _check_status(status, "gas_space_compress_links")

    def links_are_compressed(self):
        return bool(self.lib.gas_space_links_are_compressed(self.c_ptr))

    def addr2str(self, sector, addr):
        return int(self.lib.gas_addr2str_sector(self.c_ptr, int(sector),
                                                int(addr)))

    def str2addr(self, sector, string):
        return int(self.lib.gas_str2addr_sector(self.c_ptr, int(sector),
                                                int(string)))

    def block_ndet(self, block_id):
        return int(self.lib.gas_block_ndet(self.c_ptr, int(block_id)))

    def block_descriptors(self):
        for b in range(self.nblock):
            block = self._gas.block[b]
            yield {
                "offset": int(block.offset),
                "sa": int(block.sa),
                "sb": int(block.sb),
                "na": int(self._gas.sector_nstr[block.sa]),
                "nb": int(self._gas.sector_nstr[block.sb]),
            }

    def close(self):
        """Release the C object; repeated calls are safe."""

        gas = getattr(self, "_gas", None)
        if gas is not None:
            self.lib.gas_space_free(self.c_ptr)
            self._gas = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GasContractPlan:
    """Reusable Hamiltonian contraction plan for one compressed GAS space.

    The GAS space is retained but not closed by this object.  The normalized
    ERI and gos arrays are retained because the C plan stores borrowed
    pointers to them for the full plan lifetime.
    """

    def __init__(self, gas, eri):
        if not gas.links_are_compressed():
            raise ValueError(
                "Hamiltonian contraction planning requires compressed links")
        self.lib = gas.lib
        self.gas = gas
        self.eri = _as_pair_matrix(eri, gas.norb_total)
        self.gos = numpy.ascontiguousarray(
            self.eri + self.eri.T, dtype=numpy.float64)
        self._plan = ctypes.c_void_p()
        status = self.lib.fci_contract_gas_plan_create(
            ctypes.byref(self._plan), gas.c_ptr,
            _gaslib.double_ptr(self.eri), _gaslib.double_ptr(self.gos))
        _check_status(status, "fci_contract_gas_plan_create")

    @property
    def ndet(self):
        return self.gas.ndet

    @property
    def norb(self):
        return self.gas.norb_total

    @property
    def nelec(self):
        return self.gas.nelec

    def contract(self, fcivec):
        """Contract the fixed absorbed Hamiltonian with one GAS CI vector."""

        if self._plan is None:
            raise RuntimeError("Hamiltonian contraction plan is closed")
        shape = numpy.asarray(fcivec).shape
        ci0 = _as_c_double(numpy.asarray(fcivec).reshape(-1))
        if ci0.size != self.ndet:
            raise ValueError("CI vector size does not match GAS determinant count")
        ci1 = numpy.zeros_like(ci0)
        status = self.lib.fci_contract_gas_plan_execute(
            self._plan, _gaslib.double_ptr(ci0), _gaslib.double_ptr(ci1))
        _check_status(status, "fci_contract_gas_plan_execute")
        return ci1.reshape(shape)

    def close(self):
        """Release the C plan; the borrowed GAS space remains caller-owned."""

        plan = getattr(self, "_plan", None)
        if plan is not None:
            self.lib.fci_contract_gas_plan_free(plan)
            self._plan = None
        self.eri = None
        self.gos = None
        self.gas = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class GasRDMPlan:
    """Reusable owner of a raw-link GAS space and its C RDM plan."""

    def __init__(self, solver, norb, nelec):
        self.lib = solver.lib
        self.gas = solver.make_space(norb, nelec, compress_links=False)
        self._plan = ctypes.c_void_p()
        try:
            status = self.lib.fci_rdm_gas_plan_create(
                ctypes.byref(self._plan), self.gas.c_ptr)
            _check_status(status, "fci_rdm_gas_plan_create")
        except Exception:
            self.gas.close()
            raise

    @property
    def ndet(self):
        return self.gas.ndet

    @property
    def norb(self):
        return self.gas.norb_total

    @property
    def task_count(self):
        return int(self.lib.fci_rdm_gas_plan_task_count(self._plan))

    @property
    def workspace_bytes(self):
        return int(self.lib.fci_rdm_gas_plan_workspace_bytes(self._plan))

    def _vectors(self, cibra, ciket):
        bra = _as_c_double(numpy.asarray(cibra).reshape(-1))
        ket = _as_c_double(numpy.asarray(ciket).reshape(-1))
        if bra.size != self.ndet or ket.size != self.ndet:
            raise ValueError("CI vector size does not match GAS determinant count")
        return bra, ket

    def make_rdm1s(self, cibra, ciket):
        """Return alpha and beta active-space transition 1-RDMs."""

        bra, ket = self._vectors(cibra, ciket)
        dm1a = numpy.zeros((self.norb, self.norb), dtype=numpy.float64)
        dm1b = numpy.zeros_like(dm1a)
        status = self.lib.fci_rdm_gas_plan_make_rdm1s(
            self._plan, _gaslib.double_ptr(bra), _gaslib.double_ptr(ket),
            _gaslib.double_ptr(dm1a), _gaslib.double_ptr(dm1b))
        _check_status(status, "fci_rdm_gas_plan_make_rdm1s")
        return dm1a, dm1b

    def make_rdm1(self, cibra, ciket):
        """Return the spin-summed active-space transition 1-RDM."""

        dm1a, dm1b = self.make_rdm1s(cibra, ciket)
        return dm1a + dm1b

    def make_rdm12s(self, cibra, ciket):
        """Return spin-resolved active-space transition 1- and 2-RDMs."""

        bra, ket = self._vectors(cibra, ciket)
        shape1 = (self.norb, self.norb)
        shape2 = shape1 + shape1
        dm1a = numpy.zeros(shape1, dtype=numpy.float64)
        dm1b = numpy.zeros(shape1, dtype=numpy.float64)
        dm2aa = numpy.zeros(shape2, dtype=numpy.float64)
        dm2ab = numpy.zeros(shape2, dtype=numpy.float64)
        dm2bb = numpy.zeros(shape2, dtype=numpy.float64)
        status = self.lib.fci_rdm_gas_plan_make_rdm12s(
            self._plan, _gaslib.double_ptr(bra), _gaslib.double_ptr(ket),
            _gaslib.double_ptr(dm1a), _gaslib.double_ptr(dm1b),
            _gaslib.double_ptr(dm2aa), _gaslib.double_ptr(dm2ab),
            _gaslib.double_ptr(dm2bb))
        _check_status(status, "fci_rdm_gas_plan_make_rdm12s")
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

    def make_rdm12(self, cibra, ciket):
        """Return spin-summed active-space transition 1- and 2-RDMs."""

        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = self.make_rdm12s(
            cibra, ciket)
        dm1 = dm1a + dm1b
        dm2 = dm2aa + dm2bb
        dm2 += dm2ab
        dm2 += dm2ab.transpose(2, 3, 0, 1)
        return dm1, dm2

    def close(self):
        """Release the RDM plan and its GAS space; repeated calls are safe."""

        plan = getattr(self, "_plan", None)
        if plan is not None:
            self.lib.fci_rdm_gas_plan_free(plan)
            self._plan = None
        gas = getattr(self, "gas", None)
        if gas is not None:
            gas.close()
            self.gas = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FCISolver(direct_spin1.FCISolver):
    """Determinant-space GAS solver using the frozen C/OpenMP backend.

    The Davidson driver and its convergence controls follow
    :class:`pyscf.fci.direct_spin1.FCISolver`.  CI vectors are one-dimensional
    arrays in canonical GAS block order rather than rectangular CAS arrays.
    """

    _keys = direct_spin1.FCISolver._keys | {
        "gas_orbs", "gas_restr", "gas_restr_type", "e_spin_penalty",
        "e_physical", "spin_penalty_method",
    }

    def __init__(self, mol=None, gas_orbs=None, gas_restr=None,
                 gas_restr_type=addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
                 lib=None):
        super().__init__(mol)
        self.gas_orbs = tuple(gas_orbs) if gas_orbs is not None else None
        self.gas_restr = gas_restr
        self.gas_restr_type = gas_restr_type
        self.lib = lib or _gaslib.load_library()
        self.davidson_only = False
        self.e_spin_penalty = None
        self.e_physical = None
        self.spin_penalty_method = None

    def _space_spec(self, norb, nelec):
        nelec = fci_addons._unpack_nelec(nelec, self.spin)
        if self.gas_orbs is None:
            gas_orbs = (int(norb),)
            gas_orbs, blocks = addons_gas.normalize_gas_spec(
                gas_orbs, nelec)
        else:
            gas_orbs = self.gas_orbs
            if sum(gas_orbs) != int(norb):
                raise ValueError("sum(gas_orbs) must equal norb")
            gas_orbs, blocks = addons_gas.normalize_gas_spec(
                gas_orbs, nelec, self.gas_restr, self.gas_restr_type)
        return gas_orbs, nelec, blocks

    def dump_flags(self, verbose=None):
        """Print PySCF FCI flags and GAS spin-penalty settings."""

        direct_spin1.FCISolver.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        if hasattr(self, "ss_penalty"):
            target = getattr(self, "ss_value", None)
            log.info("spin penalty shift = %g", self.ss_penalty)
            log.info("target S^2 = %s", "minimum" if target is None else target)
        return self

    def space_info(self, norb, nelec):
        """Estimate GAS dimensions and validate C-kernel limits."""

        gas_orbs, nelec, blocks = self._space_spec(norb, nelec)
        return addons_gas.check_kernel_limits(gas_orbs, nelec, blocks)

    def make_space(self, norb, nelec, compress_links=False):
        """Construct an owned :class:`GasSpace` for this solver."""

        gas_orbs, nelec, blocks = self._space_spec(norb, nelec)
        return GasSpace(gas_orbs, nelec, blocks, lib=self.lib,
                        compress_links=compress_links)

    def make_rdm_plan(self, norb, nelec):
        """Construct a reusable :class:`GasRDMPlan` context manager."""

        return GasRDMPlan(self, norb, nelec)

    def get_init_guess(self, norb, nelec, nroots, hdiag, gas=None):
        """Build PySCF-style determinant guesses in the GAS vector layout."""

        hdiag = _as_c_double(numpy.asarray(hdiag).reshape(-1))
        ndet = (self.space_info(norb, nelec)["ndet_estimate"]
                if gas is None else gas.ndet)
        if hdiag.size != ndet:
            raise ValueError("hdiag size does not match GAS determinant count")
        nroots = int(nroots)
        if nroots < 1 or nroots > hdiag.size:
            raise ValueError("nroots must be between 1 and the GAS dimension")
        addresses = _lowest_hdiag_addresses(hdiag, nroots)
        guesses = []
        for root, address in enumerate(addresses):
            guess = numpy.zeros(hdiag.size, dtype=numpy.float64)
            guess[int(address)] = 1.0
            # Independent globally supported probes are required for robust
            # multiroot convergence when the Hamiltonian contains invariant
            # sectors that are absent from the lowest diagonal determinants.
            _add_distributed_noise(guess, root)
            guesses.append(guess)
        return guesses

    def make_hdiag(self, h1e, eri, norb, nelec, *args, **kwargs):
        """Return the Hamiltonian diagonal in canonical GAS order."""

        compress_links = bool(kwargs.pop("compress_links", False))
        with self.make_space(norb, nelec, compress_links=compress_links) as gas:
            n = gas.norb_total
            h1e = _as_c_double(h1e, (n, n))
            eri = _as_pair_matrix(eri, n)
            out = numpy.empty(gas.ndet, dtype=numpy.float64)
            status = self.lib.fci_make_hdiag_gas(
                gas.c_ptr, _gaslib.double_ptr(h1e), _gaslib.double_ptr(eri),
                _gaslib.double_ptr(out))
            _check_status(status, "fci_make_hdiag_gas")
            return out

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    *args, **kwargs):
        """Contract an absorbed Hamiltonian with a GAS CI vector."""

        plan = kwargs.pop("plan", None)
        if plan is not None:
            if not isinstance(plan, GasContractPlan):
                raise TypeError("plan must be a GasContractPlan")
            expected_nelec = fci_addons._unpack_nelec(nelec, self.spin)
            if int(norb) != plan.norb or expected_nelec != plan.nelec:
                raise ValueError("contraction plan does not match norb/nelec")
            return plan.contract(fcivec)

        compress_links = bool(kwargs.pop("compress_links", True))
        with self.make_space(norb, nelec, compress_links=compress_links) as gas:
            with GasContractPlan(gas, eri) as plan:
                return plan.contract(fcivec)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None,
                    **kwargs):
        raise NotImplementedError(
            "GASCI uses the absorbed one-electron Hamiltonian and contract_2e")

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        """Evaluate the GAS electronic energy for a normalized CI vector."""

        h2e = absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
        ci = _as_c_double(numpy.asarray(fcivec).reshape(-1))
        hc = self.contract_2e(h2e, ci, norb, nelec, link_index)
        return float(numpy.dot(ci, numpy.asarray(hc).reshape(-1)))

    def pspace(self, h1e, eri, norb, nelec, hdiag=None, np=400):
        """Build a bounded GAS pspace Hamiltonian.

        Unlike CAS ``direct_spin1``, the frozen GAS kernel has no dedicated
        pspace builder.  Each column therefore uses one full Hamiltonian
        product and the requested dimension is capped at
        ``GAS_PSPACE_MATVEC_MAX``.
        """

        requested = int(np)
        if requested < 1:
            raise ValueError("np must be positive")
        h1e = _as_c_double(h1e, (int(norb), int(norb)))
        eri_phys = _as_pair_matrix(eri, norb)
        h2e = absorb_h1e(h1e, eri_phys, norb, nelec, fac=0.5)
        with self.make_space(norb, nelec, compress_links=True) as gas:
            if hdiag is None:
                diagonal = numpy.empty(gas.ndet, dtype=numpy.float64)
                status = self.lib.fci_make_hdiag_gas(
                    gas.c_ptr, _gaslib.double_ptr(h1e),
                    _gaslib.double_ptr(eri_phys),
                    _gaslib.double_ptr(diagonal))
                _check_status(status, "fci_make_hdiag_gas")
            else:
                diagonal = _as_c_double(numpy.asarray(hdiag).reshape(-1))
                if diagonal.size != gas.ndet:
                    raise ValueError(
                        "hdiag size does not match GAS determinant count")

            size = min(requested, gas.ndet, GAS_PSPACE_MATVEC_MAX)
            addresses = _lowest_hdiag_addresses(diagonal, size)
            with GasContractPlan(gas, h2e) as plan:
                h0 = self._pspace_with_plan(plan, addresses)
        return addresses, h0

    def gen_linkstr(self, norb, nelec, tril=True, spin=None):
        raise NotImplementedError(
            "GAS link tables are owned by GasSpace and are not CAS linkstr arrays")

    def contract_ss(self, fcivec, norb, nelec):
        """Contract the spin-square operator with a GAS CI vector."""

        gas_orbs, nelec, blocks = self._space_spec(norb, nelec)
        if not addons_gas.is_spin_complete(gas_orbs, nelec, blocks):
            raise ValueError(
                "contract_ss requires a spin-complete GAS restriction")
        with self.make_space(norb, nelec, compress_links=False) as gas:
            return _GasSpinPlan(gas).contract(fcivec)

    def transform_ci_for_orbital_rotation(self, fcivec, norb, nelec, u):
        raise NotImplementedError(
            "arbitrary active-orbital rotations do not preserve a GAS space")

    def large_ci(self, fcivec, norb, nelec, tol=0.1, return_strs=True):
        """Return the largest GAS CI coefficients and determinant strings."""

        ci = _as_c_double(numpy.asarray(fcivec).reshape(-1))
        with self.make_space(norb, nelec, compress_links=False) as gas:
            if ci.size != gas.ndet:
                raise ValueError("CI vector size does not match GAS determinant count")
            selected = numpy.flatnonzero(numpy.abs(ci) > float(tol))
            if selected.size == 0:
                selected = numpy.asarray([int(numpy.argmax(numpy.abs(ci)))])

            result = []
            blocks = list(gas.block_descriptors())
            iblock = 0
            for address in selected:
                address = int(address)
                while (iblock + 1 < len(blocks) and
                       address >= blocks[iblock + 1]["offset"]):
                    iblock += 1
                desc = blocks[iblock]
                local = address - desc["offset"]
                ia, ib = divmod(local, desc["nb"])
                alpha = gas.addr2str(desc["sa"], ia)
                beta = gas.addr2str(desc["sb"], ib)
                if return_strs:
                    alpha_out = bin(alpha)
                    beta_out = bin(beta)
                else:
                    alpha_out = numpy.asarray(
                        [p for p in range(int(norb)) if (alpha >> p) & 1],
                        dtype=numpy.int32)
                    beta_out = numpy.asarray(
                        [p for p in range(int(norb)) if (beta >> p) & 1],
                        dtype=numpy.int32)
                result.append((ci[address], alpha_out, beta_out))
        return result

    def _pspace_with_plan(self, plan, addresses):
        """Construct H over selected compact GAS determinant addresses."""

        addresses = numpy.asarray(addresses, dtype=numpy.intp).reshape(-1)
        h0 = numpy.empty((addresses.size, addresses.size),
                         dtype=numpy.float64)
        basis = numpy.zeros(plan.ndet, dtype=numpy.float64)
        for column, address in enumerate(addresses):
            basis.fill(0.0)
            basis[int(address)] = 1.0
            product = plan.contract(basis).reshape(-1)
            h0[:, column] = product[addresses]
        # Roundoff in independently accumulated columns can leave tiny
        # antisymmetric components.  LAPACK receives an explicitly symmetric H.
        return 0.5 * (h0 + h0.T)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0,
               nroots=None, **kwargs):
        """Solve the GASCI problem with a bounded pspace or Davidson.

        This method is intentionally low-level: h1e and eri are assumed to be
        active-space integrals, and GAS restrictions are supplied through
        ``gas_orbs`` and ``gas_restr`` on the solver.
        """

        nroots = int(nroots if nroots is not None
                     else kwargs.pop("nroots", getattr(self, "nroots", 1)))
        if nroots < 1:
            raise ValueError("nroots must be positive")

        h1e = _as_c_double(h1e, (int(norb), int(norb)))
        eri_phys = _as_pair_matrix(eri, norb)
        h2e = absorb_h1e(h1e, eri_phys, norb, nelec, fac=0.5)

        def option(name, default):
            value = kwargs.pop(name, None)
            return default if value is None else value

        conv_tol = option("tol", getattr(self, "conv_tol", 1e-10))
        conv_tol_residual = option(
            "tol_residual", getattr(self, "conv_tol_residual", None))
        max_cycle = option("max_cycle", getattr(self, "max_cycle", 50))
        max_space = option("max_space", getattr(self, "max_space", 12))
        lindep = option("lindep", getattr(self, "lindep", 1e-14))
        verbose = option("verbose", getattr(self, "verbose", None))
        level_shift = option("level_shift", getattr(self, "level_shift", 1e-3))
        max_memory = option("max_memory", getattr(self, "max_memory", 4000))
        lessio = bool(option("lessio", getattr(self, "lessio", False)))
        davidson_only = bool(option("davidson_only", self.davidson_only))
        pspace_size = int(option(
            "pspace_size", getattr(self, "pspace_size", 400)))
        orbsym = option("orbsym", None)
        wfnsym = option("wfnsym", None)
        kwargs.pop("envs", None)
        if pspace_size < 0:
            raise ValueError("pspace_size must be nonnegative")
        if orbsym is not None or wfnsym is not None:
            raise NotImplementedError(
                "orbital and wavefunction symmetry filtering is not implemented")

        self.norb = int(norb)
        self.nelec = fci_addons._unpack_nelec(nelec, self.spin)
        spin_penalty = _spin_penalty_parameters(self, norb, self.nelec)
        self.e_spin_penalty = None
        self.e_physical = None
        self.spin_penalty_method = None

        with self.make_space(norb, nelec, compress_links=False) as gas:
            if nroots > gas.ndet:
                raise ValueError("nroots exceeds GAS determinant count")
            if (spin_penalty is not None and
                    not addons_gas.is_spin_complete(
                        gas.norb, self.nelec, gas.blocks)):
                raise ValueError(
                    "fix_spin_ requires a spin-complete GAS restriction")
            spin_plan = (_GasSpinPlan(gas)
                         if spin_penalty is not None else None)
            gas.compress_links()
            hdiag = numpy.empty(gas.ndet, dtype=numpy.float64)
            status = self.lib.fci_make_hdiag_gas(
                gas.c_ptr, _gaslib.double_ptr(h1e), _gaslib.double_ptr(eri_phys),
                _gaslib.double_ptr(hdiag))
            _check_status(status, "fci_make_hdiag_gas")

            linear_spin_penalty = False
            if spin_penalty is not None:
                (spin_shift, spin_target, spin_minimum,
                 spin_eigenvalues) = spin_penalty
                linear_spin_penalty = spin_target < spin_minimum + 0.1

            plan = GasContractPlan(gas, h2e)
            try:
                full_pspace = (
                    spin_penalty is None and ci0 is None and
                    not davidson_only and
                    gas.ndet <= pspace_size and
                    gas.ndet <= GAS_PSPACE_MATVEC_MAX)
                full_spin_pspace = (
                    spin_penalty is not None and ci0 is None and
                    pspace_size > 0 and gas.ndet <= pspace_size and
                    gas.ndet <= GAS_SPIN_PSPACE_MATVEC_MAX)
                if full_pspace or full_spin_pspace:
                    addresses = _lowest_hdiag_addresses(hdiag, gas.ndet)
                    h0 = self._pspace_with_plan(plan, addresses)
                    if full_spin_pspace:
                        s2 = spin_plan.matrix()
                        s2 = s2[numpy.ix_(addresses, addresses)]
                        delta_s2 = s2 - spin_target * numpy.eye(gas.ndet)
                        if linear_spin_penalty:
                            h0 += spin_shift * delta_s2
                        else:
                            h0 += spin_shift * numpy.dot(delta_s2, delta_s2)
                        self.spin_penalty_method = "exact-small-space"
                    eigenvalues, eigenvectors = numpy.linalg.eigh(h0)
                    e = eigenvalues[:nroots]
                    c = []
                    for root in range(nroots):
                        vector = numpy.empty(gas.ndet, dtype=numpy.float64)
                        vector[addresses] = eigenvectors[:, root]
                        c.append(vector)
                    converged = numpy.ones(nroots, dtype=bool)
                else:
                    guess_count = nroots
                    if spin_penalty is not None:
                        guess_count = min(
                            gas.ndet, max(4 * nroots, nroots + 4))
                    default_guess = self.get_init_guess(
                        norb, nelec, guess_count, hdiag, gas=gas)
                    if ci0 is None:
                        guess = default_guess
                    else:
                        if isinstance(ci0, (list, tuple)):
                            guess = [
                                _as_c_double(numpy.asarray(x).reshape(-1))
                                for x in ci0]
                        else:
                            arr = numpy.asarray(ci0)
                            if arr.ndim == 2 and arr.shape[0] == nroots:
                                guess = [
                                    _as_c_double(arr[i].reshape(-1))
                                    for i in range(nroots)]
                            elif arr.ndim == 2 and arr.shape[1] == nroots:
                                guess = [
                                    _as_c_double(arr[:, i].reshape(-1))
                                    for i in range(nroots)]
                            else:
                                guess = [_as_c_double(arr.reshape(-1))]
                        for i, vec in enumerate(guess):
                            if vec.size != gas.ndet:
                                raise ValueError(
                                    "initial CI vector size does not match "
                                    "GAS determinant count")
                            norm = numpy.linalg.norm(vec)
                            if norm == 0:
                                raise ValueError(
                                    "initial CI vector must be nonzero")
                            guess[i] = vec / norm
                        # direct_spin1 extends an undersized user guess so that
                        # multiroot Davidson can generate enough trial vectors.
                        if len(guess) < guess_count:
                            guess.extend(default_guess[len(guess):guess_count])
                    if spin_penalty is not None:
                        projected_guess = []
                        for candidate in guess:
                            vector = spin_plan.project(
                                candidate, spin_target, spin_eigenvalues)
                            for previous in projected_guess:
                                vector -= numpy.dot(previous, vector) * previous
                            norm = numpy.linalg.norm(vector)
                            if norm > max(100.0 * lindep, 1e-12):
                                projected_guess.append(vector / norm)
                            if len(projected_guess) == nroots:
                                break
                        if len(projected_guess) < nroots:
                            raise RuntimeError(
                                "could not construct enough independent "
                                "target-spin initial guesses")
                        # Retain a small unprojected globally supported probe.
                        # The Davidson problem must still find the lowest
                        # eigenvalue of the penalized Hamiltonian when the
                        # requested shift is too small to change spin ordering.
                        exploration_guess = []
                        for candidate in guess[:max(1, nroots)]:
                            norm = numpy.linalg.norm(candidate)
                            if norm > 0.0:
                                exploration_guess.append(candidate / norm)
                        guess = projected_guess + exploration_guess
                        self.spin_penalty_method = (
                            "projected-plus-global-davidson")
                    if nroots == 1 and len(guess) == 1:
                        guess = guess[0]

                    def hop(vec):
                        result = self.contract_2e(
                            h2e, vec, norb, nelec, plan=plan).reshape(-1)
                        if spin_penalty is None:
                            return result
                        ss_vector = numpy.asarray(
                            spin_plan.contract(vec)).reshape(-1)
                        if linear_spin_penalty:
                            ss_vector -= spin_target * vec
                            result += spin_shift * ss_vector
                            return result
                        tmp = ss_vector - spin_target * vec
                        correction = numpy.asarray(
                            spin_plan.contract(tmp)).reshape(-1)
                        correction -= spin_target * tmp
                        result += spin_shift * correction
                        return result

                    preconditioner_diagonal = hdiag
                    if spin_penalty is not None:
                        spin_diagonal = spin_plan.diagonal_vector()
                        delta_diagonal = spin_diagonal - spin_target
                        if linear_spin_penalty:
                            penalty_diagonal = spin_shift * delta_diagonal
                        else:
                            # This is the inexpensive diagonal approximation to
                            # (S^2-target)^2.  The projected trial vectors
                            # improve access to the target sector; this term
                            # only improves Davidson conditioning.
                            penalty_diagonal = (
                                spin_shift * delta_diagonal * delta_diagonal)
                        preconditioner_diagonal = hdiag + penalty_diagonal

                    def precond(dx, e, *args):
                        denom = preconditioner_diagonal - e
                        small = numpy.abs(denom) < level_shift
                        denom = denom.copy()
                        denom[small] = level_shift
                        return dx / denom

                    converged, e, c = pyscf_lib.davidson1(
                        lambda vectors: [hop(vec) for vec in vectors],
                        guess, precond, tol=conv_tol, max_cycle=max_cycle,
                        max_space=max_space, lindep=lindep,
                        max_memory=max_memory, nroots=nroots, lessio=lessio,
                        verbose=verbose, tol_residual=conv_tol_residual,
                        follow_state=spin_penalty is None)
            finally:
                plan.close()

        e = numpy.asarray(e, dtype=numpy.float64).reshape(-1) + float(ecore)
        if isinstance(c, (list, tuple)):
            c_list = [numpy.asarray(x, dtype=numpy.float64).reshape(-1)
                      for x in c]
        else:
            c_arr = numpy.asarray(c, dtype=numpy.float64)
            if nroots == 1:
                c_list = [c_arr.reshape(-1)]
            elif c_arr.ndim == 2 and c_arr.shape[0] == nroots:
                c_list = [c_arr[i].reshape(-1) for i in range(nroots)]
            elif c_arr.ndim == 2 and c_arr.shape[1] == nroots:
                c_list = [c_arr[:, i].reshape(-1) for i in range(nroots)]
            else:
                c_list = [c_arr.reshape(-1)]
        if spin_penalty is not None:
            penalty_values = []
            # The contraction GasSpace above has left its context.  Rebuild a
            # short-lived raw-link space for post-solver diagnostics instead
            # of retaining a plan backed by released C memory.
            with self.make_space(
                    norb, nelec, compress_links=False) as diagnostic_gas:
                diagnostic_spin = _GasSpinPlan(diagnostic_gas)
                for vector in c_list[:nroots]:
                    ss_vector = numpy.asarray(
                        diagnostic_spin.contract(vector)).reshape(-1)
                    if linear_spin_penalty:
                        penalty = spin_shift * (
                            numpy.dot(vector, ss_vector) - spin_target)
                    else:
                        delta = ss_vector - spin_target * vector
                        penalty = spin_shift * numpy.dot(delta, delta)
                    penalty_values.append(float(penalty))
            penalty_values = numpy.asarray(
                penalty_values, dtype=numpy.float64)
            physical_values = e[:nroots] - penalty_values
            if nroots == 1:
                self.e_spin_penalty = float(penalty_values[0])
                self.e_physical = float(physical_values[0])
            else:
                self.e_spin_penalty = penalty_values
                self.e_physical = physical_values
        if nroots == 1:
            self.converged = bool(numpy.asarray(converged).reshape(-1)[0])
            self.eci = float(e[0])
            self.ci = c_list[0]
        else:
            self.converged = numpy.asarray(converged, dtype=bool)[:nroots]
            self.eci = e[:nroots]
            self.ci = c_list[:nroots]
        return self.eci, self.ci

    @pyscf_lib.with_doc(direct_spin1.make_rdm1s.__doc__)
    def make_rdm1s(self, ci, norb, nelec, link_index=None):
        return self.trans_rdm1s(ci, ci, norb, nelec, link_index)

    @pyscf_lib.with_doc(direct_spin1.make_rdm1.__doc__)
    def make_rdm1(self, ci, norb, nelec, link_index=None):
        return self.trans_rdm1(ci, ci, norb, nelec, link_index)

    @pyscf_lib.with_doc(direct_spin1.make_rdm12s.__doc__)
    def make_rdm12s(self, ci, norb, nelec, link_index=None, reorder=True):
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        with self.make_rdm_plan(norb, nelec) as plan:
            return plan.make_rdm12s(ci, ci)

    @pyscf_lib.with_doc(direct_spin1.make_rdm12.__doc__)
    def make_rdm12(self, ci, norb, nelec, link_index=None, reorder=True):
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        with self.make_rdm_plan(norb, nelec) as plan:
            return plan.make_rdm12(ci, ci)

    def make_rdm2(self, ci, norb, nelec, link_index=None, reorder=True):
        """Return the spin-traced GAS two-particle density matrix."""

        return self.make_rdm12(ci, norb, nelec, link_index, reorder)[1]

    @pyscf_lib.with_doc(direct_spin1.trans_rdm1s.__doc__)
    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None):
        with self.make_rdm_plan(norb, nelec) as plan:
            return plan.make_rdm1s(cibra, ciket)

    @pyscf_lib.with_doc(direct_spin1.trans_rdm1.__doc__)
    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None):
        with self.make_rdm_plan(norb, nelec) as plan:
            return plan.make_rdm1(cibra, ciket)

    @pyscf_lib.with_doc(direct_spin1.trans_rdm12s.__doc__)
    def trans_rdm12s(self, cibra, ciket, norb, nelec, link_index=None,
                     reorder=True):
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        with self.make_rdm_plan(norb, nelec) as plan:
            dm1s, (dm2aa, dm2ab, dm2bb) = plan.make_rdm12s(cibra, ciket)
            _, (_, dm2ba_ji, _) = plan.make_rdm12s(ciket, cibra)
        dm2ba = dm2ba_ji.transpose(3, 2, 1, 0)
        return dm1s, (dm2aa, dm2ab, dm2ba, dm2bb)

    @pyscf_lib.with_doc(direct_spin1.trans_rdm12.__doc__)
    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
        if not reorder:
            raise NotImplementedError("reorder=False is not supported")
        with self.make_rdm_plan(norb, nelec) as plan:
            return plan.make_rdm12(cibra, ciket)

    def spin_square(self, ci, norb, nelec, *args, **kwargs):
        """Return ``(<S^2>, 2S+1)`` from spin-resolved GAS RDMs."""

        nelec = fci_addons._unpack_nelec(nelec, self.spin)
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = self.make_rdm12s(
            ci, norb, nelec)
        return spin_square_from_rdm12s((dm1a, dm1b), (dm2aa, dm2ab, dm2bb),
                                       nelec)


def spin_square_from_rdm12s(dm1s, dm2s, nelec):
    """Compute <S^2> from spin-resolved GAS RDMs.

    The formula uses the opposite-spin exchange trace from dm2ab and is valid
    for fixed (Nalpha, Nbeta) CI vectors in a common spatial orbital basis.
    """

    na, nb = fci_addons._unpack_nelec(nelec)
    dm2ab = dm2s[1]
    sz = 0.5 * (na - nb)
    trans = numpy.einsum("pqqp->", dm2ab)
    ss = sz * (sz + 1.0) + nb - trans
    ss = float(numpy.real_if_close(ss))
    mult = (max(0.0, 4.0 * ss + 1.0)) ** 0.5
    return ss, mult
