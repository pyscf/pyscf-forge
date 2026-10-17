#!/usr/bin/env python
'''
This file details the basic usage of the diagonalization-free Chebyshev filtering SCF method.
References:
    Goedecker and Teter, PRB 51, 9455 (1995)
    Goedecker, Reviews of Modern Physics 71, 1085-1123 (1999)
    Martin, R. M. (2020). Electronic structure: basic theory and practical methods. Cambridge university press.
'''

from pyscf import gto
from pyscf.scf.chebyshev_scf import ChebyshevSCF

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=4)
mf = ChebyshevSCF(mol)
mf.chebyshev_degree = 10
mf.beta = 40.0
mf.kernel()
print('ChebyshevSCF Energy:', mf.e_tot)