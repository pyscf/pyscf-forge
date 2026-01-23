import numpy as np
from pyscf.pbc import cc
from pyscf.pbc.pwscf.pw_helper import gtomf2pwmf
from pyscf.pbc.pwscf.kccsd_rhf import PWKRCCSD
from pyscf.pbc import gto, scf, pwscf

"""
Simple CCSD calculation.

Plane-wave CCSD (and MP2) calculations can be performed
by initializing them from a plane-wave SCF calculation.
One can also convert a converged GTO SCF calculation
to PW using gtomf2pwmf() and then use it to initialize
CCSD.

Also, virtual plane-wave orbitals can be obtained by orthogonalizing
a GTO basis set with respect to the occupied plane-wave orbitals
using the get_cpw_virtual() routine.
"""

#####################################################
# PART 1: Simple CCSD using PW KSCF or gtomf2pwmf() #
#####################################################

a0 = 1.78339987
atom = "C 0 0 0; C %.10f %.10f %.10f" % (a0*0.5, a0*0.5, a0*0.5)
a = np.asarray([
        [0., a0, a0],
        [a0, 0., a0],
        [a0, a0, 0.]])

cell = gto.Cell(atom=atom, a=a, basis="gth-szv", pseudo="gth-pade",
                ke_cutoff=50)
cell.build()
cell.verbose = 0

kpts = cell.make_kpts([2,1,1])

# Create a plane-wave SCF calculation and run it.
# Number of virtual orbitals should be set explicitly
# and tested for convergence of the correlation energy.
# Too few virtual orbitals will result in underestimating
# the correlation energy.
pwmf = pwscf.khf.PWKRHF(cell, kpts)
pwmf.nvir = 8
pwmf.kernel()

# Use pwmf to initialize plane-wave CCSD and run it
pwmcc = PWKRCCSD(pwmf).kernel()
e_corr = pwmcc.e_corr
# e_corr ~ -0.152 Ha
print("PW CCSD correlation energy from PWKRHF:", e_corr)

# Alternatively, perform a GTO SCF calculation
mf = scf.KRHF(cell, kpts)
mf.kernel()

# GTO CCSD calculation
mcc = cc.kccsd_rhf.RCCSD(mf)
mcc.kernel()

# Convert the GTO SCF to PW SCF
pwmf = gtomf2pwmf(mf)
# Run PW CCSD from the PW SCF
pwmcc = PWKRCCSD(pwmf).kernel()

# e_corr ~ -0.155 Ha
print("GTO CCSD correlation energy:", mcc.e_corr)
print("PW CCSD correlation energy using gtomf2pwmf:", pwmcc.e_corr)
# Correlation energies fromt the GTO and PW versions should agree
# if the two basis sets are representing the same orbitals.
# mcc.e_corr ~ pwmcc.e_corr ~ -0.155 Ha
assert(np.abs(mcc.e_corr - pwmcc.e_corr) < 1e-5)


######################################################################
# PART 2: Using get_cpw_virtual() to get better virtuals for PW CCSD #
######################################################################

# NOTE that convergence of e_corr with respect to the number of virtual
# orbitals can be slower when PW virtual orbitals are used.
# To fix this, we can construct the PW
# virtuals from a GTO basis. There are two benefits to this:
# 1) No need to set nvir or compute virtual orbitals with PW SCF
# 2) (Possibly) more correlation with fewer virtuals in the post-SCF step

pwmf = pwscf.KRHF(cell, kpts)
# no need to set nvir
pwmf.kernel()

# call get_cpw_virtual() with a chosen GTO basis
# This adds virtuals to the mo_coeff attribute
# of the pwmf object.
# Here we use a very small basis.
moe_ks, mocc_ks = pwmf.get_cpw_virtual("gth-szv")
mmp = PWKRCCSD(pwmf)
mmp.kernel()
# e_corr ~ -0.153 Ha
print("PW CCSD correlation energy using GTO virtuals:", mmp.e_corr)

# Use a larger basis to get more realistic correlation energy
moe_ks, mocc_ks = pwmf.get_cpw_virtual("ccecp-cc-pvdz")
mmp = PWKRCCSD(pwmf)
mmp.kernel()
# e_corr ~ -0.245 Ha
print("PW CCSD using GTO virtuals with larger basis:", mmp.e_corr)


###################################################
# PART 3: Initialize PW KSCF from gamma-point GTO #
###################################################

# One can also initialize PW KSCF from a gamma-point GTO calculation,
# and it will automatically be converted to KSCF.
atom = "H 0 0 -0.35; H 0 0 0.35"
a = 3 * np.eye(3)
cell = gto.Cell(atom=atom, a=a, basis="gth-szv", pseudo="gth-pade",
                ke_cutoff=50)
cell.build()
cell.verbose = 0

mf = scf.RHF(cell)
mf.kernel()

mcc = cc.ccsd.RCCSD(mf)
mcc.kernel()

pwmf = gtomf2pwmf(mf)
pwmcc = PWKRCCSD(pwmf).kernel()

assert(np.abs(mcc.e_corr - pwmcc.e_corr) < 1e-5)

