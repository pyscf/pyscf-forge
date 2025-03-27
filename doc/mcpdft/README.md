# Multiconfiguration Pair-Density Functional Theory Module for PySCF

2025-02-18

- Version 1.0

## Install

- Install to python site-packages folder

```sh
pip install git+https://github.com/pyscf/pyscf-forge
```

- Install in a custom folder for development

```sh
git clone https://github.com/pyscf/pyscf-forge /home/abc/local/path

# Set pyscf extended module path
echo 'export PYSCF_EXT_PATH=/home/abc/local/path:$PYSCF_EXT_PATH' >> ~/.bashrc

# Compile libpdft.so
cd /home/abc/local/path/pyscf/lib
mkdir build; cd build
# Intel MKL BLAS may require 'cmake -DBLA_VENDOR=Intel10_64lp_seq ..' below
cmake ..
make
```

## Features

- Multi-configuration pair-density functional theory (MC-PDFT) total electronic
energy calculations for wave functions of various types.
  - CASCI
  - CASSCF
  - State-averaged CASSCF (including "mixed" solver with different spins and/or
  point groups)
  - Extended multi-state MC-PDFT (XMS-PDFT): [*Faraday Discuss* **2020**, 224,
  348-372]
  - Compressed multi-state MC-PDFT (CMS-PDFT): [*JCTC* **2020**, *16*, 7444]
  - Linearized PDFT (L-PDFT): [*JCTC* **2023**, *19*, 3172]
- On-the-fly generation of on-top density functionals from underlying KS-DFT
'LDA' or 'GGA' exchange-correlation functionals as defined in libxc.
  - Translated functionals: 
    1. LDA and GGA functionals [*JCTC* **2014**, *10*, 3669]
    1. meta-GGA functionals [*PNAS* **2024**, *122*]
  - Fully-translated functionals: 
    1. LDA and GGA functionals [*JCTC* **2015**, *11*, 4077]
  - Global hybrid functionals: [*JPCL* **2020**, *11*, 10158] and [*JCTC*
  **2020**, *16*, 2274]
  - Notes:
    1. Translation of 'meta' KS-DFT functionals which depend on the Laplacian
    of the density are not supported.
    1. Range-separated hybrid on-top functionals are not supported.
    1. Translation of functionals defined as global hybrids at the Libxc or
    PySCF level is not supported, except for 'MC23', 'tPBE0' and 'ftPBE0'.
    Other global hybrid functionals are specified using PySCF's [custom
    functional parser][custom functional parser]; see
    [examples/mcpdft/02-hybrid_functionals.py].
- Additional properties
  - Decomposition of total electronic energy into core, Coulomb, on-top
  components
  - Analytical nuclear gradients for:
    - non-hybrid functionals only
        1. CMS-PDFT: [*Mol Phys* **2022**, 120]
        1. L-PDFT (no meta-GGA): [*JCTC* **2024**, *20*, 3637]
    - hybrid and non-hybrid functionals:
        1. Single-state CASSCF wave function: [*JCTC* **2018**, *14*, 126]
        1. State-averaged CASSCF wave functions: [*JCP* **2020**, *153*, 014106]
  - Permanent electric dipole moment (non-hybrid functionals only) for:
    1. Single-state CASSCF wave function: [*JCTC* **2021**, *17*, 7586]
    1. State-averaged CASSCF wave functions
    1. CMS-PDFT [*JPC A* **2023**, *127*, 4194]
  - Transition electric dipole moment (non-hybrid functionals only) for:
    1. CMS-PDFT [*JPC A* **2023**, *127*, 4194]
  - Derivative couplings for:
    1. CMS-PDFT [*JPC A* **2024**, *128*, 1698]
- Multi-configuration density-coherence functional theory (MC-DCFT) total
energy: [*JCTC* **2021**, *17*, 2775]

[*faraday discuss* **2020**, 224, 348-372]: http://dx.doi.org/10.1039/D0FD00037J
[*jcp* **2020**, *153*, 014106]: http://dx.doi.org/10.1063/5.0007040
[*jctc* **2014**, *10*, 3669]: http://dx.doi.org/10.1021/ct500483t
[*jctc* **2015**, *11*, 4077]: http://dx.doi.org/10.1021/acs.jctc.5b00609
[*jctc* **2018**, *14*, 126]: http://dx.doi.org/10.1021/acs.jctc.7b00967
[*jctc* **2020**, *16*, 2274]: http://dx.doi.org/10.1021/acs.jctc.9b01178
[*jctc* **2020**, *16*, 7444]: http://dx.doi.org/10.1021/acs.jctc.0c00908
[*jctc* **2021**, *17*, 2775]: http://dx.doi.org/10.1021/acs.jctc.0c01346
[*jctc* **2021**, *17*, 7586]: http://dx.doi.org/10.1021/acs.jctc.1c00915
[*jctc* **2023**, *19*, 3172]: https://dx.doi.org/10.1021/acs.jctc.3c00207
[*jpcl* **2020**, *11*, 10158]: http://dx.doi.org/10.1021/acs.jpclett.0c02956
[*mol phys* **2022**, 120]: http://dx.doi.org/10.1080/00268976.2022.2110534
[*JCTC* **2024**, *20*, 3637]: https://dx.doi.org/10.1021/acs.jctc.4c00095
[*JPC A* **2024**, *128*, 1698]: https://dx.doi.org/10.1021/acs.jpca.3c07048
[*JPC A* **2023**, *127*, 4194]: https://dx.doi.org/10.1021/acs.jpca.3c01142
[*PNAS* **2024**, *122*]: https://dx.doi.org/10.1073/pnas.2419413121

[comment]: <> (Code hyperlinks)
[examples/mcpdft/02-hybrid_functionals.py]: examples/mcpdft/02-hybrid_functionals.py
[custom functional parser]: https://github.com/pyscf/pyscf/blob/master/examples/dft/24-custom_xc_functional.py
[examples/mcpdft/02-hybrid_functionals.py]: examples/mcpdft/02-hybrid_functionals.py
