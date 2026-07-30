#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Authors:
#          Shirong Wang <srwang20@fudan.edu.cn>
#

from pyscf.dh.xccode.xccode import XCDH, XCType, XCList
from pyscf.dh.xccode.xcjson import FUNCTIONALS_DICT
from pyscf.dft.libxc import XCFunctionalCache
from pyscf import dft
import numpy

_D3_VERSIONS = {
    "bj": "d3bj", "zero": "d3zero", "bjm": "d3bjm", "mbj": "d3mbj",
    "zerom": "d3zerom", "mzero": "d3mzero", "op": "d3op",
}


def _register_ext_params(xc_list):
    r"""Register custom libxc functionals for XC components with external parameters.

    Uses named-parameter dict format (``{key: value}``) for ext_params, supported
    by PySCF >= 2.14.
    Omega values are extracted and passed to ``register_custom_functional_``
    for proper RSH handling.
    """
    ext_params = {}
    omega_vals = []
    for info in xc_list:
        if not (info.type & XCType.WITH_EXT_PARAM):
            continue
        if not info.parameters_keyword and not info.parameters:
            continue
        name = info.name
        try:
            xc_fc = XCFunctionalCache(name, 0)
        except Exception:
            continue
        obj_by_id = xc_fc.obj_by_id()
        if len(obj_by_id) != 1:
            continue
        xc_id = next(iter(obj_by_id))
        if info.parameters_keyword:
            named_params = {k.lower(): float(v) for k, v in info.parameters_keyword.items()}
            ext_params[xc_id] = named_params
        else:
            ext_params[xc_id] = numpy.array(info.parameters, dtype=numpy.float64)
            info.parameters.clear()
        if isinstance(info.parameters_keyword, dict):
            for k, v in info.parameters_keyword.items():
                if k.upper() == '_OMEGA' and float(v):
                    omega_vals.append(float(v))
        info.parameters_keyword.clear()
    if not ext_params:
        return None
    base_token = xc_list.name_token
    # Build deterministic hash: for numpy arrays use tobytes(), for dicts use sorted items
    def _hash_val(xid):
        v = ext_params[xid]
        if isinstance(v, dict):
            return tuple(sorted(v.items()))
        return v.tobytes()
    reg_name = f"__dh_{abs(hash((base_token, tuple(_hash_val(xid) for xid in sorted(ext_params))))):x}"
    dft.libxc.register_custom_functional_(
        reg_name, base_token,
        ext_params=ext_params,
        omega=omega_vals if omega_vals else None)
    return reg_name.lower()


def xc_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return XCList(a, code_scf=True) == XCList(b, code_scf=True)


def _check_unsupported(name):
    key = name.upper().replace("-", "_")
    entry = FUNCTIONALS_DICT.get(key)
    if entry and entry.get("supported") is False:
        raise NotImplementedError(
            f"Functional '{name}' is not supported in this version of pyscf.dh."
        )


def _strip_d3_suffix(name: str) -> str:
    for s in ("_D3BJ", "_D3ZERO", "_D3BJM", "_D3ZEROM", "_D3OP",
              "D3BJ", "D3ZERO", "D3BJM", "D3ZEROM", "D3OP",
              "_D3", "D3", "-D3", "-D3BJ", "-D3ZERO"):
        if name.upper().endswith(s.upper()):
            return name[:-len(s)]
    return name


def _extract_components(xc_eng, xc_scf, name):
    xc = xc_scf
    low_rung = xc_eng.remove(
        xc_eng.extract_by_xctype(XCType.MP2 | XCType.DFTD3 | XCType.DFTD4),
        inplace=False)
    is_bdh = (low_rung.token == xc_scf)
    if is_bdh:
        reg = _register_ext_params(low_rung)
        if reg:
            xc = reg
        else:
            xc = low_rung.token
        xc_n = None
    else:
        scf_reg = _register_ext_params(XCList(xc_scf, code_scf=True))
        if scf_reg:
            xc = scf_reg
        non_reg = _register_ext_params(low_rung)
        if non_reg:
            xc_n = non_reg
        else:
            xc_n = low_rung.token if low_rung.token != xc_scf else None
    mp2_list = xc_eng.extract_by_xctype(XCType.MP2)
    if len(mp2_list) > 0:
        mp2 = mp2_list[0]
        add = mp2.parameters_keyword
        if "OS" in add or "SS" in add:
            c_os = mp2.fac * mp2.parameters[0] * add.get("OS", 1)
            c_ss = mp2.fac * mp2.parameters[1] * add.get("SS", 1)
        else:
            c_os, c_ss = mp2.fac * mp2.parameters[0], mp2.fac * mp2.parameters[1]
    else:
        c_os, c_ss = 0, 0
    xc_add = {}
    d3_list = xc_eng.extract_by_xctype(XCType.DFTD3)
    if len(d3_list) > 0:
        d3 = d3_list[0]
        add = d3.parameters_keyword
        if any(k.upper() != "XC" for k in add):
            raise NotImplementedError(
                "Explicit D3 parameters are unsupported. "
                "Use 'DFTD3(BJ)' or 'DFTD3(BJ, XC=name)' to look up defaults."
            )
        damp = d3.parameters[0].lower() if d3.parameters else "bj"
        version = _D3_VERSIONS.get(damp, "d3bj")
        if "XC" in add:
            d3_xc = add["XC"]
        else:
            d3_xc = _strip_d3_suffix(name)
            if "*" in d3_xc or "+" in d3_xc:
                raise NotImplementedError(
                    "D3 in code strings requires XC= parameter. "
                    "Use 'DFTD3(BJ, XC=PBE)' or a named functional instead."
                )
        xc_add["D3"] = {"xc": d3_xc, "version": version}
    d4_list = xc_eng.extract_by_xctype(XCType.DFTD4)
    if len(d4_list) > 0:
        d4 = d4_list[0]
        add = d4.parameters_keyword
        if any(k.upper() != "XC" for k in add):
            raise NotImplementedError(
                "Explicit D4 parameters are unsupported. "
                "Use 'DFTD4()' or 'DFTD4(XC=name)' to look up defaults."
            )
        if "XC" in add:
            d4_xc = add["XC"]
        else:
            d4_xc = _strip_d3_suffix(name)
            if "*" in d4_xc or "+" in d4_xc:
                raise NotImplementedError(
                    "D4 in code strings requires XC= parameter. "
                    "Use 'DFTD4(XC=PBE)' or a named functional instead."
                )
        xc_add["D4"] = {"xc": d4_xc, "version": "d4"}
    return xc, xc_n, c_os, c_ss, xc_add


def parse_xc_dh(xc_dh: str):
    if isinstance(xc_dh, tuple) and len(xc_dh) == 2:
        xc_scf = XCList(xc_dh[0], code_scf=True).token
        xc_eng = XCList(xc_dh[1], code_scf=False)
        xc, xc_n, c_os, c_ss, xc_add = _extract_components(xc_eng, xc_scf, "")
        return (xc, xc_n, c_os, c_ss), xc_add

    xcdh = XCDH(xc_dh)
    _check_unsupported(xc_dh)
    xc_scf = xcdh.xc_scf.token
    xc_eng = xcdh.xc_eng
    xc, xc_n, c_os, c_ss, xc_add = _extract_components(xc_eng, xc_scf, xc_dh)
    return (xc, xc_n, c_os, c_ss), xc_add


def describe_xc_dh(xc_dh, file=None):
    """Print a human-readable summary of a parsed DH functional."""
    xcdh = XCDH(xc_dh)
    _check_unsupported(xc_dh)
    xc_scf_orig = xcdh.xc_scf.token
    xc_eng = xcdh.xc_eng
    low_rung_orig = xc_eng.remove(
        xc_eng.extract_by_xctype(XCType.MP2 | XCType.DFTD3 | XCType.DFTD4),
        inplace=False)

    (xc, xc_n, c_os, c_ss), xc_add = parse_xc_dh(xc_dh)
    corr_list = xc_eng.extract_by_xctype(XCType.RUNG_HIGH)
    corr = corr_list[0].name if corr_list else ""

    if xc_n is not None:
        dh_type = "xDH"
    else:
        has_exch = any(i.name != "HF" and XCType.EXCH & i.type for i in low_rung_orig)
        dh_type = "bDH" if has_exch else "post-HF"

    lines = [str(xc_dh)]
    lines.append(f"  type:   {dh_type}")
    lines.append(f"  corr:   {corr}")
    lines.append(f"  xc:     {xc_scf_orig}")
    if xc_n is not None:
        lines.append(f"  xc_n:   {low_rung_orig.token}")
    lines.append(f"  c_os:   {c_os}")
    lines.append(f"  c_ss:   {c_ss}")
    if xc_add.get("D3"):
        d3 = xc_add["D3"]
        lines.append(f"  D3:     xc={d3['xc']} version={d3['version']}")
    if xc_add.get("D4"):
        d4 = xc_add["D4"]
        lines.append(f"  D4:     xc={d4['xc']} version={d4['version']}")
    print("\n".join(lines), file=file)
    print(file=file)
