from pyscf import gto, fci, scf, lib, ao2mo
from pyscf.csf_fci import csf_solver

mol = gto.M (atom='H 0 0 0; F 0 0 1.1', basis='sto3g', output='csf_fci.log',
             verbose=lib.logger.INFO)
mf = scf.RHF (mol).run ()

h0 = mf.energy_nuc ()
h1 = mf.get_hcore ()
h2 = ao2mo.restore (1, mf._eri, mol.nao_nr ())

cisolver = csf_solver (mol, smult=1)

# Print how many CSFs, and other information, to stdout. You need to set # of orbs and electrons
print ("Singlet configuration:")
cisolver.print_transformer_cache (norb=mol.nao_nr (), nelec=mol.nelec)
print ("Singlet energy:", cisolver.kernel (h1,h2,mol.nao_nr(),mol.nelec,ecore=h0)[0])

# Number of orbs and electrons as well as spin multiplicity will be stored after running kernel
e = cisolver.kernel (h1,h2,mol.nao_nr(),mol.nelec,smult=3,ecore=h0)[0]
print ("\nTriplet configuration:")
cisolver.print_transformer_cache ()
print ("Triplet energy:", e)


