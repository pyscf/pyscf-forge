#!/usr/bin/env python

'''
Example script for exact exchange with staggered mesh method.

Author: Stephen Quiton (sjquiton@gmail.com)

Reference: The Staggered Mesh Method: Accurate Exact Exchange Toward the 
           Thermodynamic Limit for Solids, J. Chem. Theory Comput. 2024, 20,
           18, 7958-7968

'''


from pyscf.pbc.mp.kmp2_stagger import KMP2_stagger
from pyscf.pbc import df, gto, scf, mp
from pyscf.pbc.scf.khf_stagger import KHF_stagger
import time
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft import numint as pbcnumint
from pyscf import dft

'''
Hydrogen dimer
'''
cell = gto.Cell()
cell.pseudo = 'gth-pbe'
cell.basis = 'gth-szv'
cell.ke_cutoff = 100
cell.atom = '''
    H 3.00   3.00   2.10
    H 3.00   3.00   3.90
    '''
cell.a = '''
    6.0   0.0   0.0
    0.0   6.0   0.0
    0.0   0.0   6.0
    '''
cell.unit = 'B'
cell.verbose = 4
cell.build()


# HF calcuation to base Non-SCF and Split-SCF staggered mesh calculations on.
nks = [2, 2, 2]
kpts = cell.make_kpts(nks, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf.with_df = df.GDF(cell, kpts).build()
ehf = kmf.kernel()

'''
KHF Stagger, Non-SCF version
Compute densities at shifted mesh non-SCF using F_unshifted. Additional cost 
is ~ 1 extra K-build. 
'''
kmf_stagger = KHF_stagger(kmf,"non-scf")
kmf_stagger.kernel()
etot = kmf_stagger.e_tot
ek_stagger = kmf_stagger.ek
assert((abs(etot - -1.0915433999061728) < 1e-6))
assert((abs(ek_stagger - -0.5688182610550594) < 1e-6))

'''
KHF Stagger, Split-SCF version
Converge densities at shifted with SCF. Additional cost is ~ 1x normal SCF.
'''
kmf_stagger = KHF_stagger(kmf,"split-scf")
kmf_stagger.kernel()
etot = kmf_stagger.e_tot
ek_stagger = kmf_stagger.ek
assert((abs(etot - -1.0980852331458024) < 1e-6))
assert((abs(ek_stagger -  -0.575360094294689) < 1e-6))

'''
KHF Stagger, regular version
Converge densities with combined unshifted + shifted mesh. Additional cost is
4x normal SCF. No need for prior SCF calculation. 
''' 
nks = [2, 2, 2]
kpts = cell.make_kpts(nks, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf.with_df = df.GDF(cell, kpts).build()
# No kernel needed. 
kmf_stagger = KHF_stagger(kmf,"regular")
kmf_stagger.kernel()
etot = kmf_stagger.e_tot
ek_stagger = kmf_stagger.ek
assert((abs(etot - -1.0911866312312735) < 1e-6))
assert((abs(ek_stagger - -0.5684614923801602) < 1e-6))


'''
KHF Stagger Non-SCF with DFT
'''

nks = [2, 2, 2]
kpts = cell.make_kpts(nks, with_gamma_point=True)
# kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
# kmf.with_df = df.GDF(cell, kpts).build()
# ehf = kmf.kernel()

# Setup DFT object
dft.numint.NumInt.libxc = dft.xcfun
xc = 'PBE0'
xc_pure = "PBE"
x = 'PBEx'
c = 'PBEc'

krks = pbcdft.KRKS(cell, kpts)
krks.xc = xc
krks.exxdiv = 'ewald'
krks.with_df = df.GDF(cell, kpts).build()
edft = krks.kernel()

krks_stagger = KHF_stagger(krks,"non-scf")
krks_stagger.kernel()
etot = krks_stagger.e_tot
ek_stagger = krks_stagger.ek




# '''
# Diamond system
# '''

# cell = gto.Cell()
# cell.pseudo = 'gth-pade'
# cell.basis = 'gth-szv'
# cell.ke_cutoff = 100
# cell.atom = '''
#     C     0.      0.      0.
#     C     1.26349729, 0.7294805 , 0.51582061
#     '''
# cell.a = '''
#     2.52699457, 0.        , 0.
#     1.26349729, 2.18844149, 0.
#     1.26349729, 0.7294805 , 2.06328243
#     '''
# cell.unit = 'angstrom'
# cell.verbose = 4
# cell.build()