r"""
    This is an example to calculate the molecular geometrical gradient of the ppRPA energy.
"""
from pyscf import gto, scf, dft
from pyscf.pprpa import rpprpa_davidson, rpprpa_direct
from pyscf.grad import rpprpa
import numpy as np

mult = "s"
istate = 0
nocc = 5
nvir = 10

def mfobj(dx):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [["O", (0.0, 0.0, 0.0)],
                ["H", (0.0, 1.0, 1.0+dx)],
                ["H", (0.0, -1.0, 1.0)]]
    mol.basis = "cc-pvdz"
    mol.charge = 0
    mol.unit = "Bohr"
    mol.build()
    
    # density fitting is required to ensure the numerical and analytical gradients are the same
    # mf = scf.RHF(mol).density_fit()

    # DFT grid response not implemented in pprpa
    # Higher grid level for better accuracy
    mf = dft.RKS(mol, xc="b3lyp").density_fit()
    mf.grids.level = 7
    mf.conv_tol = 1e-12
    return mf

def pprpaobj(mf, nocc_act, nvir_act):
    mp = rpprpa_davidson.RppRPADavidson(mf, nocc_act, nvir_act, channel="pp", nroot=3)
    mp.mu = 0.0
    return mp




dx = 0.001
mf1 = mfobj(dx)
mf2 = mfobj(-dx)

e1 = mf1.kernel()
# f1 = mf1.get_fock(dm=dm0_hf)
h1 = mf1.get_hcore()

e2 = mf2.kernel()
# f2 = mf2.get_fock(dm=dm0_hf)
h2 = mf2.get_hcore()

# dfock = (f1 - f2)/(2.0*dx)
dh = (h1 - h2)/(2.0*dx)
e_hf = (e1 - e2)/dx/2.0

mp1 = pprpaobj(mf1, nocc, nvir)
mp1.kernel(mult)
mp1.analyze()
if mult == "s":
    e1 += mp1.exci_s[istate] if mp1.channel == "pp" else -mp1.exci_s[istate]
else:
    e1 += mp1.exci_t[istate] if mp1.channel == "pp" else -mp1.exci_t[istate]
mp2 = pprpaobj(mf2, nocc, nvir)
mp2.kernel(mult)
mp2.analyze()
if mult == "s":
    e2 += mp2.exci_s[istate] if mp2.channel == "pp" else -mp2.exci_s[istate]
else:
    e2 += mp2.exci_t[istate] if mp2.channel == "pp" else -mp2.exci_t[istate]

e_rpa = (e1 - e2)/dx/2.0

mf = mfobj(0.0)
mf.kernel()
if mult == "s":
    oo_dim = nocc * (nocc + 1) // 2
else:
    oo_dim = nocc * (nocc - 1) // 2
mp = pprpaobj(mf, nocc, nvir)
mp.kernel(mult)
mp.analyze()

xy = mp.xy_s[istate] if mult == "s" else mp.xy_t[istate]
from lib_pprpa.grad import pprpa
mpg = mp.Gradients(mult,istate)
mpg.kernel()

print("analytical (Total):  ", mpg.de)
print("numerical (Total):  ", e_rpa)
print("diff: ", e_rpa - mpg.de[1,2])
# print(e1,e2)
print(mpg.de[0]+mpg.de[1]+mpg.de[2]) # Should be zero
