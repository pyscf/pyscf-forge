from pyscf.pbc import gto
from pyscf.pbc.pwscf.krks import PWKRKS
# from pyscf.pbc.pwscf.kpt_symm import KsymAdaptedPWKRKS as PWKRKS
from pyscf.pbc.pwscf.ncpp_cell import NCPPCell
import numpy as np

"""
This example demonstrates the convergence of the total
energy with respect to plane-wave energy cutoff for
GTH and SG15 pseudopotentials. The SG15 converges
faster, especially up to a 1000 eV cutoff (36.76 Ha),
because these potentials were designed to converge more
quickly.

NOTE: Before using this example, you must set
pbc_pwscf_ncpp_cell_sg15_path in your pyscf config file.
"""

kwargs = dict(
    atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
    a = np.asarray([
            [0.       , 1.78339987, 1.78339987],
            [1.78339987, 0.        , 1.78339987],
            [1.78339987, 1.78339987, 0.        ]]),
    basis="gth-szv",
    ke_cutoff=50,
    pseudo="gth-pade",
    verbose=0,
)

cell = gto.Cell(**kwargs)
cell.build()

kwargs.pop("pseudo")
nccell = NCPPCell(**kwargs)
nccell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)

ens1 = []
ens2 = []
# A larger set of ecuts below is provided in case it's useful.
# ecuts = [18.38235294, 22.05882353, 25.73529412, 29.41176471, 33.08823529,
#          36.76470588, 44.11764706, 55.14705882, 73.52941176, 91.91176471]
ecuts = [18.38235294, 25.73529412, 33.08823529, 36.76470588, 55.14705882]
for ecut in ecuts:
    print("ECUT", ecut)
    # Run the GTH calculations
    mf = PWKRKS(cell, kpts, xc="PBE", ecut_wf=ecut)
    mf.damp_type = "simple"
    mf.damp_factor = 0.7
    mf.nvir = 4 # converge first 4 virtual bands
    mf.kernel()
    ens1.append(mf.e_tot)

    # Run the SG15 calculations
    mf2 = PWKRKS(nccell, kpts, xc="PBE", ecut_wf=ecut)
    mf2.damp_type = "simple"
    mf2.damp_factor = 0.7
    mf2.nvir = 4 # converge first 4 virtual bands
    mf2.init_pp()
    mf2.init_jk()
    mf2.kernel()
    ens2.append(mf2.e_tot)
    print(mf.e_tot, mf2.e_tot)
    print()

print()
print("GTH Total Energies (Ha)")
print(ens1)
print("Energy cutoffs (Ha)")
print(ecuts[:-1])
print("Differences vs Max Cutoff (Ha)")
print(np.array(ens1[:-1]) - ens1[-1])
print()
print("SG15 Total Energies (Ha)")
print(ens2)
print("Energy cutoffs (Ha)")
print(ecuts[:-1])
print("Differences vs Max Cutoff (Ha)")
print(np.array(ens2[:-1]) - ens2[-1])
