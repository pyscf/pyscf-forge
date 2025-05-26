#!/usr/bin/env python

"""
Example script for exact exchange with staggered mesh method.

Author: Stephen Quiton (sjquiton@gmail.com)

Reference: The Staggered Mesh Method: Accurate Exact Exchange Toward the
           Thermodynamic Limit for Solids, J. Chem. Theory Comput. 2024, 20,
           18, 7958-7968
"""

from pyscf.pbc import df, gto, scf
from pyscf.pbc.scf.khf_stagger import KHF_stagger
from pyscf.pbc import dft as pbcdft
from pyscf import dft

"""
Hydrogen dimer
"""
cell = gto.Cell()
cell.pseudo = "gth-pbe"
cell.basis = "gth-szv"
cell.ke_cutoff = 100
cell.atom = """
    H 3.00   3.00   2.10
    H 3.00   3.00   3.90
    """
cell.a = """
    6.0   0.0   0.0
    0.0   6.0   0.0
    0.0   0.0   6.0
    """
cell.unit = "B"
cell.verbose = 4
cell.build()

"""
For Non-SCF and Split-SCF, need to run a normal kpts SCF calculation first.
"""
nks = [2, 2, 2]
kpts = cell.make_kpts(nks, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
kmf.with_df = df.GDF(cell, kpts).build()
ehf = kmf.kernel()

"""
KHF Stagger, Non-SCF version
Compute densities at shifted kpt mesh non-self-consistently using the Fock
matrix at the unshifted mesh. Additional cost is ~ 1 extra K-build.
"""
kmf_stagger = KHF_stagger(kmf, "non-scf")
kmf_stagger.kernel()
etot = kmf_stagger.e_tot
ek_stagger = kmf_stagger.ek

print("Non-SCF Stagger")
print("Total energy: ", etot)
print("Exchange energy: ", ek_stagger)

assert abs(etot - -1.0915433999061728) < 1e-6
assert abs(ek_stagger - -0.5688182610550594) < 1e-6

"""
KHF Stagger, Split-SCF version
Converge densities at shifted kpt mesh self-conistently. Additional cost
is ~ 1 extra SCF kernel.
"""
kmf_stagger = KHF_stagger(kmf, "split-scf")
kmf_stagger.kernel()
etot = kmf_stagger.e_tot
ek_stagger = kmf_stagger.ek

print("Split-SCF Stagger")
print("Total energy: ", etot)
print("Exchange energy: ", ek_stagger)

assert abs(etot - -1.0907254038200516) < 1e-6
assert abs(ek_stagger - -0.5680002649689386) < 1e-6

"""
KHF Stagger, regular version
Converge all densities with combined unshifted + shifted mesh. Total estimated
cost is 4x normal SCF. No need for prior SCF calculation.
"""
nks = [2, 2, 2]
kpts = cell.make_kpts(nks, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv="ewald")
kmf.with_df = df.GDF(cell, kpts).build()
# No kernel needed.
kmf_stagger = KHF_stagger(kmf, "regular")
kmf_stagger.kernel()
etot = kmf_stagger.e_tot
ek_stagger = kmf_stagger.ek

print("Regular Stagger")
print("Total energy: ", etot)
print("Exchange energy: ", ek_stagger)


assert abs(etot - -1.0973224854862946 < 1e-6)
assert abs(ek_stagger - -0.5684614923801601) < 1e-6


"""
KHF Stagger Non-SCF with DFT
"""

nks = [2, 2, 2]
kpts = cell.make_kpts(nks, with_gamma_point=True)

dft.numint.NumInt.libxc = dft.xcfun
xc = "PBE0"
krks = pbcdft.KRKS(cell, kpts)
krks.xc = xc
krks.exxdiv = "ewald"
krks.with_df = df.GDF(cell, kpts).build()
edft = krks.kernel()

krks_stagger = KHF_stagger(krks, "non-scf")
krks_stagger.kernel()
etot = krks_stagger.e_tot
ek_stagger = krks_stagger.ek

print("Non-SCF Stagger with DFT")
print("Total energy: ", etot)
print("Exchange energy: ", ek_stagger)

assert abs(etot - -1.133718165281441) < 1e-6
assert abs(ek_stagger - -0.5678939393308997) < 1e-6
