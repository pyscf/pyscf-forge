#!/usr/bin/env python

'''
Reference NOCI, NOCISD and NOCISDT for stretched H6 with RHF and two
broken-symmetry UHF references, compared with FCI.  The excitations of every
reference up to the excitation level form the NOCI space; same-reference
pairs use the Slater-Condon rules and nonorthogonal pairs the extended
nonorthogonal Wick theorem.
'''

from pyscf import gto, scf, fci, nonorthogonal

mol = gto.M(atom=[('H', (0, 0, 1.8 * i)) for i in range(6)], basis='sto-3g', verbose=0)

mf = scf.RHF(mol).run()
refs = [mf]
# Antiferromagnetic UHF solutions with opposite spin patterns
for pattern in ((1, -1, 1, -1, 1, -1), (-1, 1, -1, 1, -1, 1)):
    uhf = scf.UHF(mol)
    dm = uhf.get_init_guess()
    for iatm, spin in enumerate(pattern):
        ao = mol.search_ao_label('%d H' % iatm)
        major = 0 if spin > 0 else 1
        dm[major][ao, ao] *= 1.8
        dm[1 - major][ao, ao] *= .2
    uhf.kernel(dm)
    refs.append(uhf)
for k, r in enumerate(refs):
    print('reference %d  E = %.10f' % (k, r.e_tot))

e_ref = nonorthogonal.NOCI(mol, refs).run().e_tot
print('reference NOCI  E = %.10f' % e_ref)

mynoci = nonorthogonal.NOCISD(mol, refs).run()
print('NOCISD (%d determinants, %d independent)  E = %.10f' %
      (len(mynoci.dets), mynoci.nindep, mynoci.e_tot))

# Any excitation level, here singles to triples (NOCISDT)
mynoci = nonorthogonal.NOCI(mol, refs, level=3).run()
print('NOCISDT (%d determinants, %d independent)  E = %.10f' %
      (len(mynoci.dets), mynoci.nindep, mynoci.e_tot))

e_fci = fci.FCI(mf).kernel()[0]
print('FCI  E = %.10f' % e_fci)
