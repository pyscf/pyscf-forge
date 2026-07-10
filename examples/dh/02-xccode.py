"""Example: xccode — how XC strings work in double-hybrid calculations."""

from pyscf import gto
from pyscf.dh import DFDH
from pyscf.dh.xccode import XCList, XCDH, XCType, parse_xc_dh, xc_equal

mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ")

# 1. DH functionals can be any name from the JSON database
for name in ["B2PLYP", "XYG3", "PBE0-DH", "DSD-PBEP86-D3BJ", "TPSS0-DH"]:
    mf = DFDH(mol, xc=name).run()
    print(f"{name:25s} {mf.e_tot:.10f}")

# 2. xDH via 2-tuple (code_scf, code_eng) — no JSON entry needed
xc_xdh = ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP + 0.3211*MP2")
mf = DFDH(mol, xc=xc_xdh).run()
print(f"\ncustom XYG3 (2-tuple): {mf.e_tot:.10f}")

# 3. Inspect what the parser returns for any functional
print("\nFunc       xc_scf                            cc    c_os  c_ss")
for name in ["B2PLYP", "XYG3", "PBE0-DH", "DSD-PBEP86-D3BJ", "SCS-MP2"]:
    xc_list, _ = parse_xc_dh(name)
    print(f"{name:10s} {xc_list[0]:35s} {xc_list[2]:5g} {xc_list[3]:5g} {xc_list[4]:5g}")

# 4. xDH: separate SCF and non-consistent energy via XCDH
xcdh = XCDH("XYG3")
print(f"\nXYG3 SCF  = {xcdh.xc_scf.token}")
low_rung = xcdh.xc_eng.remove(
    xcdh.xc_eng.extract_by_xctype(XCType.MP2 | XCType.DFTD3), inplace=False)
print(f"XYG3 xc_n = {low_rung.token}  (non-consistent energy)")

# parse_xc_dh also handles the 2-tuple format
xc_list, _ = parse_xc_dh(("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP + 0.3211*MP2"))
print(f"2-tuple → xc=({xc_list[0]}, {xc_list[2]:.4f}, {xc_list[3]:g}, {xc_list[4]:g})")

# 5. XC token comparison via xc_equal
print(f"\nxc_equal('HF', 'HF,') = {xc_equal('HF', 'HF,')}")
print(f"xc_equal('B3LYPG', 'B3LYPg') = {xc_equal('B3LYPG', 'B3LYPg')}")
print(f"xc_equal('PBE0', 'B3LYPG') = {xc_equal('PBE0', 'B3LYPG')}")

# 6. Custom XC strings via XCList (requires code_scf boolean)
xc = XCList("0.5*HF + 0.5*PBE, 0.875*PBE + 0.125*MP2", code_scf=False)
print(f"\nXCList full:  {xc.token}")
xc_scf = XCList("0.5*HF + 0.5*PBE, 0.875*PBE + 0.125*MP2", code_scf=True)
print(f"XCList scf:   {xc_scf.token}")
