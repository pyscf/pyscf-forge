from pyscf.dh.xccode.xccode import XCDH, XCType, XCList
from pyscf.dh.xccode.xcjson import FUNCTIONALS_DICT

_D3_VERSIONS = {
    "bj": "d3bj", "zero": "d3zero", "bjm": "d3bjm", "mbj": "d3mbj",
    "zerom": "d3zerom", "mzero": "d3mzero", "op": "d3op",
}


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
    xc_n = low_rung.token if low_rung.token != xc_scf else None
    mp2_list = xc_eng.extract_by_xctype(XCType.MP2)
    if len(mp2_list) > 0:
        mp2 = mp2_list[0]
        cc, c_os, c_ss = mp2.fac, mp2.parameters[0], mp2.parameters[1]
    else:
        cc, c_os, c_ss = 0, 0, 0
    xc_add = {}
    d3_list = xc_eng.extract_by_xctype(XCType.DFTD3)
    if len(d3_list) > 0:
        d3 = d3_list[0]
        add = d3.additional
        damp = d3.parameters[0].lower() if d3.parameters else "bj"
        version = _D3_VERSIONS.get(damp, "d3bj")
        if "XC" in add:
            d3_xc = add["XC"]
        else:
            d3_xc = _strip_d3_suffix(name)
        xc_add["D3"] = {"xc": d3_xc, "version": version}
    d4_list = xc_eng.extract_by_xctype(XCType.DFTD4)
    if len(d4_list) > 0:
        d4 = d4_list[0]
        d4_xc = d4.additional.get("XC", _strip_d3_suffix(name))
        xc_add["D4"] = {"xc": d4_xc, "version": "d4"}
    return xc, xc_n, cc, c_os, c_ss, xc_add


def parse_xc_dh(xc_dh: str):
    if isinstance(xc_dh, tuple) and len(xc_dh) == 2:
        xc_scf = XCList(xc_dh[0], code_scf=True).token
        xc_eng = XCList(xc_dh[1], code_scf=False)
        xc, xc_n, cc, c_os, c_ss, xc_add = _extract_components(xc_eng, xc_scf, "")
        return (xc, xc_n, cc, c_os, c_ss), xc_add

    xcdh = XCDH(xc_dh)
    _check_unsupported(xc_dh)
    xc_scf = xcdh.xc_scf.token
    xc_eng = xcdh.xc_eng
    xc, xc_n, cc, c_os, c_ss, xc_add = _extract_components(xc_eng, xc_scf, xc_dh)
    return (xc, xc_n, cc, c_os, c_ss), xc_add
