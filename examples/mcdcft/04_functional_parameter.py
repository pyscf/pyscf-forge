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
    'display_name': 'DC24,custom',
    # The Kohn-Sham exchange correlation functional the DC functional is based on.
    # Use the same notation as used in the KS module
    'xc_code': 'HCTH407',
    # Ratio of the MCSCF exchange-correlation energy
    'hyb_x': 4.525671e-01,
    # Functional parameters. We will set the functional parameters of HCTH407 (LibXC
    #     functional 164) with the values listed here. The meaning of each parameter
    #     can be obtained from the `xc-info` utility provided by LibXC.
    'params': {164: [8.198942e-01, 4.106753e+00, -3.716774e+01, 1.100812e+02, -9.600026e+01,
                     1.352989e+01, -6.881959e+01, 2.371350e+02, -3.433615e+02, 1.720927e+02,
                     1.134169e+00, 1.148509e+01, -2.210990e+01, -1.006682e+02, 1.477906e+02]},
}

# Register the functional preset to the dcfnal module with identifier 'DC24_custom'
mcdcft.dcfnal.register_dcfnal_('DC24_custom', preset)

# Evaluate MC-DCFT energy of the custom functional
mc = mcdcft.CASSCF(mf, 'DC24_custom', 2, 2, grids_level=(99, 590))
e1 = mc.kernel()[0]

# Compare with the results from pre-defined DC24
e2 = mc.recalculate_with_xc('DC24')[0]
print("Energy difference:", e1 - e2)

