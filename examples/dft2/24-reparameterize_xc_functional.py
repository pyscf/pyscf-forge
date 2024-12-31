#!/usr/bin/env python

'''
Running DFT calculations with reparameterized functionals in Libxc.

See also
* Example 24-custom_xc_functional.py to customize XC functionals using the
  functionals provided by Libxc library.
'''

from pyscf import gto
from pyscf import dft, dft2
import numpy as np

# Patch the libxc module with the new implementation
dft.libxc = dft2.libxc
dft.numint.libxc = dft2.libxc
dft.numint.LibXCMixin.libxc = dft2.libxc

XC_ID_B97_2 = 410

param_b97_1 = np.array([0.789518, 0.573805, 0.660975, 0.0, 0.0,
                        0.0820011, 2.71681, -2.87103, 0.0, 0.0,
                        0.955689, 0.788552, -5.47869, 0.0, 0.0,
                        0.21
])

param_b97_3 = np.array([0.7334648, 0.292527, 3.338789, -10.51158, 10.60907,
                        0.5623649, -1.32298, 6.359191, -7.464002, 1.827082,
                        1.13383, -2.811967, 7.431302, -1.969342, -11.74423,
                        2.692880E-01
])

mol = gto.M(
    atom = '''
    O  0.   0.       0.
    H  0.   -0.757   0.587
    H  0.   0.757    0.587 ''',
    basis = 'ccpvdz')

# Run normal B97-1
print('Normal B97-1')
mf = dft.RKS(mol, 'B97-1')
e_b971 = mf.kernel()

# Run normal B97-2
print('\nNormal B97-2')
mf.xc = 'B97-2'
e_b972 = mf.kernel()

# Run normal B97-3
print('\nNormal B97-3')
mf.xc = 'B97-3'
e_b973 = mf.kernel()

# Construct XC named `myfunctional` based on B97-2, but set its parameter to be B97-1
print('\nReparameterized B97-2: will be the same as B97-1')
dft2.libxc.register_custom_functional_('myfunctional', 'B97-2',
                                       ext_params={XC_ID_B97_2: param_b97_1})
mf.xc = 'myfunctional'
e = mf.kernel()
print('difference:', e - e_b971)

# Update the parameter of `myfunctional` to be the same as B97-3 and rerun
print('\nReparameterized B97-2: will be the same as B97-3')
dft2.libxc.update_custom_functional_('myfunctional',
                                     ext_params={XC_ID_B97_2: param_b97_3})
e = mf.kernel()
print('difference:', e - e_b973)
print()

