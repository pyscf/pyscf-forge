"""Example: xccode — how XC strings work in double-hybrid calculations."""

from pyscf import gto
from pyscf.dh import DFDH
from pyscf.dh.xccode import parse_xc_dh, xc_equal

mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ")

# 1. DH functionals can be any name from the JSON database
for name in ["B2PLYP", "XYG3", "PBE0-DH", "DSD-PBEP86-D3BJ"]:
    mf = DFDH(mol, xc=name).run()
    print(f"{name:25s} {mf.e_tot:.10f}")

# 2. xDH via 2-tuple (code_scf, code_eng) 
xc_xdh = ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP + 0.3211*MP2")
mf = DFDH(mol, xc=xc_xdh).run()
print(f"custom XYG3 (2-tuple): {mf.e_tot:.10f}")

# 3. Inspect what the parser returns for any functional
print("\nFunc       xc_scf                            c_os   c_ss")
for name in ["B2PLYP", "XYG3", "PBE0-DH", "DSD-PBEP86-D3BJ", "SCS-MP2"]:
    xc_list, _ = parse_xc_dh(name)
    print(f"{name:10s} {xc_list[0]:35s} {xc_list[2]:5g} {xc_list[3]:5g}")

# 4. All components returned by parse_xc_dh
for name in ["MP2", "B2PLYP", "XYG3", "DSD-PBEP86-D3BJ", "B2PLYP-D3BJ"]:
    (xc, xc_n, c_os, c_ss), xc_add = parse_xc_dh(name)
    d3 = xc_add.get("D3", None)
    d4 = xc_add.get("D4", None)
    print(f"\n{name}:")
    print(f"  xc    = {xc}")
    print(f"  xc_n  = {xc_n}")
    print(f"  c_os  = {c_os:.6g}   c_ss = {c_ss:.6g}")
    if d3:
        print(f"  D3    = {d3}")
        if d4:
            print(f"  D4    = {d4}")

# 5. XC token comparison via xc_equal
print(f"\nxc_equal('HF', 'HF,') = {xc_equal('HF', 'HF,')}")
print(f"xc_equal('B3LYPG', 'B3LYPg') = {xc_equal('B3LYPG', 'B3LYPg')}")
print(f"xc_equal('PBE0', 'B3LYPG') = {xc_equal('PBE0', 'B3LYPG')}")
