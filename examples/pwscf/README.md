# Plane-Wave Mode

The `pyscf.pbc.pwscf` module provides experimental support for Hartree-Fock (HF),
density functional theory (DFT), second-order Møller-Plesset perturbation theory (MP2),
and coupled cluster singles doubles (CCSD) in a plane-wave basis.
The CCECP and GTH pseudopotentials are supported for these methods,
and SG15 pseudopotentials are supported for HF and DFT calculations.
Occupation smearing and symmetry reduction of k-point meshes are implemented for HF and DFT.

## Feature Overview

The following self-consistent field (SCF) calculations are supported:
* Hartree-fock (Restricted and Unrestricted)
* Kohn-Sham DFT (Restricted and Unrestricted), with LDA, GGA, MGGA, and global hybrid functionals

Currently, the Davidson algorithm is implemented for the effective Hamiltonian diagonalization.
There are two mixing schemes for the effective potential, "Simple" and "Anderson" (DIIS).
Symmetry reduction of k-points is supported for SCF calculations, along with occupation
smearing. The default plane-wave basis set and integration grid are determined by `cell.mesh`,
but these can be customized using the energy cutoffs `ecut_wf` and `ecut_rho` or by setting
meshes directly using `PWKSCF.set_meshes()`.

The following post-SCF calculations are supported:
* MP2 (Restricted and Unrestricted)
* CCSD (Restricted only)

K-point symmetry and occupation smearing are currently not supported for post-SCF
methods. The `PWKSCF.get_cpw_virtual()` method can be used to create virtual
molecular orbitals in a GTO basis for use in post-SCF calculations.

Plane-wave calculations can be performed with GTH or CCECP pseudopotentials
(or all-electron for very small atoms). There is also basic support for SG15
norm-conserving pseudopotentials. The post-SCF methods have been tested with GTH
and CCECP but not SG15, while the SCF methods have been tested with GTH, CCECP, and SG15.

## SG15 Pseudopotentials

SG15 pseudopotentials are not included in the PySCF soruce code, and they are required
to use the `pyscf.pbc.pwscf.ncpp_cell` module. The SG15 potentials can be downloaded
from http://www.quantum-simulation.org/potentials/sg15_oncv/. The UPF format is used,
so the `sg15_oncv_upf_2020-02-06.tar.gz` set is recommended.

When initializing an `NCPPCell` object, one must pass the argument `sg15_path`,
which should point to wherever `sg15_oncv_upf_2020-02-06.tar.gz` was extracted.
Alternatively, one can set `pbc_pwscf_ncpp_cell_sg15_path` in `.pyscf_conf.py`.
