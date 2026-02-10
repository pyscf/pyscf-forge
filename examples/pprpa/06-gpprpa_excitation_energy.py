from pyscf import gto, dft

# For ppRPA excitation energy of N-electron system in particle-particle channel
# mean field is (N-2)-electron

mol = gto.Mole()
mol.verbose = 4
mol.atom = [
    ["O", (0.0, 0.0, 0.0)],
    ["H", (0.0, -0.7571, 0.5861)],
    ["H", (0.0, 0.7571, 0.5861)]]
mol.basis = 'def2-svp'
# create a (N-2)-electron system for charged-neutral H2O
mol.charge = 2
mol.build()

# =====> Part I. Restricted and its real-valued generalized ppRPA <=====
# restricted KS-DFT calculation as starting point of RppRPA
rmf = dft.RKS(mol)
rmf.xc = "b3lyp"
rmf.kernel()
from pyscf.pprpa.rpprpa_davidson import RppRPADavidson
pp = RppRPADavidson(rmf, nocc_act=3, nvir_act=None, nroot=1)
# solve for singlet states
pp.kernel("s")
# solve for triplet states
pp.kernel("t")
pp.analyze()

# Davidson algorithm for GKS, equivalent to the above RKS case
from pyscf.pprpa.gpprpa_davidson import GppRPADavidson
gmf = rmf.to_gks()
pp = GppRPADavidson(gmf, nocc_act=6, nvir_act=None, nroot=4)
# solve for singlet and triplet states
pp.kernel()
pp.analyze()



# =====> Part II. Complex-valued generalized ppRPA <=====
gmf = dft.GKS(mol).x2c1e()
gmf.xc = "b3lyp"
gmf.kernel()
pp = GppRPADavidson(gmf, nocc_act=6, nvir_act=None, nroot=4)
# solve for "singlet" and "triplet" states
# triplets are no longer degenerate with SOC
pp.kernel()
pp.analyze()