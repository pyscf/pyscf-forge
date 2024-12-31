from pyscf import scf, gto, mcdcft

'''
This input file performs a single point energy calculation of H2 with DC24.
'''

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-tzvp',
      symmetry=False, verbose=3, unit='angstrom')
mf = scf.RHF(mol)
mf.kernel()
mc = mcdcft.CASSCF(mf, 'DC24', 2, 2, grids_level=(99, 590))
mc.chkfile = 'H2_DC24.chk'
mc.kernel()

