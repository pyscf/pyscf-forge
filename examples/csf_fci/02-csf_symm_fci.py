from pyscf import gto, fci, scf, lib, ao2mo
from pyscf.csf_fci import csf_solver

mol = gto.M (atom='H 0 0 0; F 0 0 1.1', basis='sto3g', output='02-csf_symm_fci.log',
             symmetry=True, verbose=lib.logger.INFO)
mf = scf.RHF (mol).run ()

h0 = mf.energy_nuc ()
h1 = mf.mo_coeff.conj ().T @ mf.get_hcore () @ mf.mo_coeff
h2 = ao2mo.restore (1, ao2mo.full (mf._eri, mf.mo_coeff), mol.nao_nr ())

orbsym = mf.get_orbsym (mf.mo_coeff, mf.get_ovlp ())
cisolver = csf_solver (mol, smult=1).set (orbsym=orbsym)

# Print how many CSFs, and other information, to stdout. You need to set # of orbs and electrons
print ("Singlet A1 configuration (wfnsym = None means it's guessed):")
e = cisolver.kernel (h1,h2,mol.nao_nr(),mol.nelec,ecore=h0)[0]
cisolver.print_transformer_cache (norb=mol.nao_nr (), nelec=mol.nelec)
print ("Singlet A1 energy:", e)

# Number of orbs and electrons as well as spin multiplicity will be stored after running kernel
print ("\nTriplet A1 configuration:")
e = cisolver.kernel (h1,h2,mol.nao_nr(),mol.nelec,smult=3,ecore=h0)[0]
cisolver.print_transformer_cache ()
print ("Triplet A1 energy:", e)

# Number of orbs and electrons as well as spin multiplicity will be stored after running kernel
print ("\nTriplet A2 configuration:")
e = cisolver.kernel (h1,h2,mol.nao_nr(),mol.nelec,smult=3,wfnsym='A2',ecore=h0)[0]
cisolver.print_transformer_cache ()
print ("Triplet A2 energy:", e)


