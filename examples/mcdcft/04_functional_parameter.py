from pyscf import scf, gto, mcdcft

'''
This input file performs a single point energy calculation of H2 with DC24.
It demonstrates how users can define custom density coherence functionals,
especially how functional parameters can be set.
'''

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-tzvp',
      symmetry=False, verbose=3, unit='angstrom')
mf = scf.RHF(mol)
mf.kernel()

# Define a preset for the DC24 functional, which contains functional parameters.
preset = {
    # Display name of the new functional. It will be used in displayed messages and chkfiles.
    'display_name': 'DC24_custom',
    # Ratio of the MCSCF exchange-correlation energy
    'hyb_x': 4.525671e-01,
    # Functional parameters. We will set the functional parameters of HCTH407 (LibXC
    #     functional 164) with the values listed here. The meaning of each parameter
    #     can be obtained from the `xc-info` utility provided by LibXC.
    'params': {164: [8.198942e-01, 4.106753e+00, -3.716774e+01, 1.100812e+02, -9.600026e+01,
                     1.352989e+01, -6.881959e+01, 2.371350e+02, -3.433615e+02, 1.720927e+02,
                     1.134169e+00, 1.148509e+01, -2.210990e+01, -1.006682e+02, 1.477906e+02]},
}

# DC24 is based on HCTH407 functional form. We put "HCTH407" as xc_code.
mc = mcdcft.CASSCF(mf, 'HCTH407', 2, 2, xc_preset=preset, grids_level=(99, 590))
e1 = mc.kernel()[0]

# Compare with the results from pre-defined DC24
e2 = mc.recalculate_with_xc('DC24')[0]
print("Energy difference:", e1 - e2)

