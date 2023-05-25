Multi-configuration pair-density functional theory module for PySCF
=========================

2022-11-27

* Version 1.0

Install
-------
* Install to python site-packages folder
```
pip install git+https://github.com/pyscf/pyscf-forge
```

* Install in a custom folder for development
```
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

Features
-------
* Multi-configuration pair-density functional theory (MC-PDFT) total electronic
  energy calculations for wave functions of various types.
    - CASCI
    - CASSCF
    - State-averaged CASSCF (including "mixed" solver with different spins
      and/or point groups)
    - Extended multi-state MC-PDFT (XMS-PDFT): [*Faraday Discuss* **2020**, 224, 348-372]
    - Compressed multi-state MC-PDFT (CMS-PDFT): [*JCTC* **2020**, *16*, 7444]
    - Linearized PDFT (L-PDFT): [*JCTC* **2023**]
* On-the-fly generation of on-top density functionals from underlying KS-DFT
  'LDA' or 'GGA' exchange-correlation functionals as defined in Libxc.
    - Translated functionals: [*JCTC* **2014**, *10*, 3669]
    - Fully-translated functionals: [*JCTC* **2015**, *11*, 4077]
    - Global hybrid functionals: [*JPCL* **2020**, *11*, 10158] and
      [*JCTC* **2020**, *16*, 2274]
    - Notes:
        1. Translation of 'meta' KS-DFT functionals which depend on the
           kinetic energy density and/or Laplacian is not supported.
        2. Range-separated hybrid on-top functionals are not supported.
        3. Translation of functionals defined as global hybrids at the Libxc or
           PySCF level is not supported, except for 'tPBE0' and 'ftPBE0'.
           Other global hybrid functionals are specified using PySCF's [custom
           functional parser]; see [examples/mcpdft/02-hybrid_functionals.py].
* Additional properties
    - Decomposition of total electronic energy into core, Coulomb, on-top
      components
    - Analytical nuclear gradients (non-hybrid functionals only) for:
        1. Single-state CASSCF wave function: [*JCTC* **2018**, *14*, 126]
        2. State-averaged CASSCF wave functions: [*JCP* **2020**, *153*, 014106]
        3. CMS-PDFT: [*Mol Phys* **2022**, 120]
    - Permanent electric dipole moment (non-hybrid functionals only) for:
        1. Single-state CASSCF wave function: [*JCTC* **2021**, *17*, 7586]
        2. State-averaged CASSCF wave functions
        3. CMS-PDFT
    - Transition electric dipole moment (non-hybrid functionals only) for:
        1. CMS-PDFT
* Multi-configuration density-coherence functional theory (MC-DCFT)
  total energy: [*JCTC* **2021**, *17*, 2775]

[comment]: <> (Reference hyperlinks)
[*JCTC* **2020**, *16*, 7444]: http://dx.doi.org/10.1021/acs.jctc.0c00908
[*JCTC* **2014**, *10*, 3669]: http://dx.doi.org/10.1021/ct500483t
[*JCTC* **2015**, *11*, 4077]: http://dx.doi.org/10.1021/acs.jctc.5b00609
[*JPCL* **2020**, *11*, 10158]: http://dx.doi.org/10.1021/acs.jpclett.0c02956
[*JCTC* **2020**, *16*, 2274]: http://dx.doi.org/10.1021/acs.jctc.9b01178
[*JCTC* **2018**, *14*, 126]: http://dx.doi.org/10.1021/acs.jctc.7b00967
[*JCP* **2020**, *153*, 014106]: http://dx.doi.org/10.1063/5.0007040
[*JCTC* **2021**, *17*, 7586]: http://dx.doi.org/10.1021/acs.jctc.1c00915
[*JCTC* **2021**, *17*, 2775]: http://dx.doi.org/10.1021/acs.jctc.0c01346
[*Mol Phys* **2022**, 120]: http://dx.doi.org/10.1080/00268976.2022.2110534
[*Faraday Discuss* **2020**, 224, 348-372]: http://dx.doi.org/10.1039/D0FD00037J
[*JCTC* **2023**]: https://dx.doi.org/10.1021/acs.jctc.3c00207

[comment]: <> (Code hyperlinks)
[examples/mcpdft/02-hybrid_functionals.py]: examples/mcpdft/02-hybrid_functionals.py
[custom functional parser]: https://github.com/pyscf/pyscf/blob/master/examples/dft/24-custom_xc_functional.py
