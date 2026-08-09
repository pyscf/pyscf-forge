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

"""Private ctypes bridge for the GAS FCI C kernels."""

import ctypes
import os
from pathlib import Path


GAS_SUCCESS = 0
GAS_ERR_INVALID = -1
GAS_ERR_MEMORY = -2

GAS_LINK_RAW = 0
GAS_LINK_COMPRESSED = 1


gas_sid_t = ctypes.c_uint16
gas_bid_t = ctypes.c_uint32
gas_tid_t = ctypes.c_uint32


class GasBlock(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("sa", gas_sid_t),
        ("sb", gas_sid_t),
    ]


class GasRow(ctypes.Structure):
    _fields_ = [
        ("off", ctypes.c_uint32),
        ("n", ctypes.c_uint32),
    ]


class GasBlockIndex(ctypes.Structure):
    _fields_ = [
        ("by_alpha_row", ctypes.POINTER(GasRow)),
        ("by_beta_row", ctypes.POINTER(GasRow)),
        ("by_beta_sid", ctypes.POINTER(gas_sid_t)),
        ("by_beta_bid", ctypes.POINTER(gas_bid_t)),
    ]


class GasTableIndex(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.POINTER(GasRow)),
        ("dst", ctypes.POINTER(gas_sid_t)),
    ]


class GasTableRevIndex(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.POINTER(GasRow)),
        ("src", ctypes.POINTER(gas_sid_t)),
        ("tid", ctypes.POINTER(gas_tid_t)),
    ]


class GasLinkEntry(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint32),
        ("op", ctypes.c_uint16),
        ("sign", ctypes.c_int8),
        ("padding", ctypes.c_uint8),
    ]


class GasLinkTable(ctypes.Structure):
    _fields_ = [
        ("link", ctypes.POINTER(GasLinkEntry)),
        ("active_op", ctypes.POINTER(ctypes.c_uint16)),
        ("nsrc", ctypes.c_uint32),
        ("nlink", ctypes.c_uint32),
        ("nop", ctypes.c_uint16),
    ]


class GasSpaceStruct(ctypes.Structure):
    _fields_ = [
        ("ngas", ctypes.c_int),
        ("norb_tot", ctypes.c_int),
        ("norb", ctypes.POINTER(ctypes.c_int)),
        ("start", ctypes.POINTER(ctypes.c_int)),
        ("na", ctypes.c_int),
        ("nb", ctypes.c_int),
        ("ndet", ctypes.c_uint32),
        ("nsector", ctypes.c_uint16),
        ("sector_padding", ctypes.c_uint16),
        ("sector_nstr", ctypes.POINTER(ctypes.c_uint32)),
        ("sector_occ", ctypes.POINTER(ctypes.c_uint8)),
        ("sector_stride", ctypes.POINTER(ctypes.c_uint32)),
        ("nblock", ctypes.c_uint32),
        ("block", ctypes.POINTER(GasBlock)),
        ("D", GasBlockIndex),
        ("ntable", ctypes.c_uint32),
        ("link_format", ctypes.c_uint8),
        ("link_format_padding", ctypes.c_uint8 * 3),
        ("table", ctypes.POINTER(GasLinkTable)),
        ("T", GasTableIndex),
        ("R", GasTableRevIndex),
    ]


class GasMemoryReport(ctypes.Structure):
    _fields_ = [
        ("metadata", ctypes.c_uint64),
        ("sector", ctypes.c_uint64),
        ("block", ctypes.c_uint64),
        ("block_index", ctypes.c_uint64),
        ("link_table", ctypes.c_uint64),
        ("link_table_index", ctypes.c_uint64),
        ("total", ctypes.c_uint64),
    ]


def double_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _configure_library(lib):
    gas_p = ctypes.POINTER(GasSpaceStruct)
    gas_plan_pp = ctypes.POINTER(ctypes.c_void_p)
    rdm_plan_pp = ctypes.POINTER(ctypes.c_void_p)
    int_p = ctypes.POINTER(ctypes.c_int)
    double_p = ctypes.POINTER(ctypes.c_double)

    lib.gas_space_from_blocks.argtypes = [
        gas_p, ctypes.c_int, int_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, int_p,
    ]
    lib.gas_space_from_blocks.restype = ctypes.c_int

    lib.gas_space_compress_links.argtypes = [gas_p]
    lib.gas_space_compress_links.restype = ctypes.c_int

    lib.gas_space_links_are_compressed.argtypes = [gas_p]
    lib.gas_space_links_are_compressed.restype = ctypes.c_int

    lib.gas_space_free.argtypes = [gas_p]
    lib.gas_space_free.restype = None

    lib.gas_memory_bytes.argtypes = [gas_p]
    lib.gas_memory_bytes.restype = ctypes.c_uint64

    lib.gas_memory_report.argtypes = [gas_p, ctypes.POINTER(GasMemoryReport)]
    lib.gas_memory_report.restype = None

    lib.gas_addr2str_sector.argtypes = [gas_p, gas_sid_t, ctypes.c_uint32]
    lib.gas_addr2str_sector.restype = ctypes.c_uint64

    lib.gas_str2addr_sector.argtypes = [gas_p, gas_sid_t, ctypes.c_uint64]
    lib.gas_str2addr_sector.restype = ctypes.c_uint32

    lib.gas_block_ndet.argtypes = [gas_p, gas_bid_t]
    lib.gas_block_ndet.restype = ctypes.c_uint32

    lib.fci_make_hdiag_gas.argtypes = [gas_p, double_p, double_p, double_p]
    lib.fci_make_hdiag_gas.restype = ctypes.c_int

    lib.fci_contract_gas_omp_task_count.argtypes = [gas_p]
    lib.fci_contract_gas_omp_task_count.restype = ctypes.c_uint32

    lib.fci_contract_gas_parallel_units.argtypes = [gas_p]
    lib.fci_contract_gas_parallel_units.restype = ctypes.c_uint32

    lib.fci_contract_gas_plan_create.argtypes = [
        gas_plan_pp, gas_p, double_p, double_p,
    ]
    lib.fci_contract_gas_plan_create.restype = ctypes.c_int

    lib.fci_contract_gas_plan_execute.argtypes = [
        ctypes.c_void_p, double_p, double_p,
    ]
    lib.fci_contract_gas_plan_execute.restype = ctypes.c_int

    lib.fci_contract_gas_plan_free.argtypes = [ctypes.c_void_p]
    lib.fci_contract_gas_plan_free.restype = None

    lib.fci_rdm_gas_plan_create.argtypes = [rdm_plan_pp, gas_p]
    lib.fci_rdm_gas_plan_create.restype = ctypes.c_int

    lib.fci_rdm_gas_plan_make_rdm1s.argtypes = [
        ctypes.c_void_p, double_p, double_p, double_p, double_p,
    ]
    lib.fci_rdm_gas_plan_make_rdm1s.restype = ctypes.c_int

    lib.fci_rdm_gas_plan_make_rdm12s.argtypes = [
        ctypes.c_void_p, double_p, double_p, double_p, double_p,
        double_p, double_p, double_p,
    ]
    lib.fci_rdm_gas_plan_make_rdm12s.restype = ctypes.c_int

    lib.fci_rdm_gas_plan_free.argtypes = [ctypes.c_void_p]
    lib.fci_rdm_gas_plan_free.restype = None

    lib.fci_rdm_gas_plan_task_count.argtypes = [ctypes.c_void_p]
    lib.fci_rdm_gas_plan_task_count.restype = ctypes.c_uint32

    lib.fci_rdm_gas_plan_workspace_bytes.argtypes = [ctypes.c_void_p]
    lib.fci_rdm_gas_plan_workspace_bytes.restype = ctypes.c_uint64

    return lib


def _candidate_library_paths():
    value = os.environ.get("PYSCF_GAS_LIB")
    if value:
        yield Path(value)

    pyscf_dir = Path(__file__).resolve().parents[1]
    yield pyscf_dir / "lib" / "libfci_gas.so"


_LIB = None


def load_library(path=None, reload=False):
    """Load and configure libfci_gas.

    An explicit path or ``PYSCF_GAS_LIB`` can select a development build.
    The final fallback uses PySCF's native library loader.
    """

    global _LIB
    if _LIB is not None and path is None and not reload:
        return _LIB

    if path is not None:
        lib = ctypes.CDLL(str(path))
    else:
        lib = None
        for candidate in _candidate_library_paths():
            if candidate.exists():
                lib = ctypes.CDLL(str(candidate))
                break
        if lib is None:
            from pyscf import lib as pyscf_lib
            lib = pyscf_lib.load_library("libfci_gas")

    lib = _configure_library(lib)
    if path is None:
        _LIB = lib
    return lib
