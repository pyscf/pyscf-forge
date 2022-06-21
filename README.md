Multi-configuration pair-density functional theory module for PySCF
=========================

2022-06-21

* Version 0.1

Install
-------
* Install to python site-packages folder
```
pip install git+https://github.com/pyscf/mcpdft
```

* Install in a custom folder for development
```
git clone https://github.com/pyscf/mcpdft /home/abc/local/path

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
* On-the-fly generation of on-top density functionals from underlying KS-DFT exchange-correlation functionals defined in LibXC.
    - Translated functionals: [*JCTC* **2014**, *10*, 3669](http://dx.doi.org/10.1021/ct500483t)
    - Fully-translated functionals: [*JCTC* **2015**, *11*, 4077](http://dx.doi.org/10.1021/acs.jctc.5b00609)
    - Hybrid functionals: [*JPCL* **2020**, *11*, 10158](http://dx.doi.org/10.1021/acs.jpclett.0c02956)
and [*JCTC* **2020**, *16*, 2274](http://dx.doi.org/10.1021/acs.jctc.9b01178)
* Total electronic energy calculations for wave functions of various types.
    - CASCI, single or multiple states
    - CASSCF
    - State-averaged CASSCF (including "mixed" solver with different spins, point groups, etc.)
    - Compressed multi-state MC-PDFT (CMS-PDFT): [*JCTC* **2020**, *16*, 7444](http://dx.doi.org/10.1021/acs.jctc.0c00908)
* Properties in addition to total electronic energy.
    - Decomposition into core, Coulomb, on-top components
    - Analytical nuclear gradients (non-hybrid functionals only) for:
        1. Single-state CASSCF wave function: [*JCTC* **2018**, *14*, 126](http://dx.doi.org/10.1021/acs.jctc.7b00967)
        2. State-averaged CASSCF wave functions: [*JCP* **2020**, *153*, 014106](http://dx.doi.org/10.1063/5.0007040)
        3. CMS-PDFT: **in press**
    - Permanent electric dipole moment (non-hybrid functionals only) for:
        1. Single-state CASSCF wave function: [*JCTC* **2021**, *17*, 7586](http://dx.doi.org/10.1021/acs.jctc.1c00915)
        2. (**in testing**) State-averaged CASSCF wave functions
* Multi-configuration density-coherence functional theory (MC-DCFT)
total energy for CASSCF wave functions: [*JCTC* **2021**, *17*, 5733](http://dx.doi.org/10.1021/acs.jctc.1c00679)
