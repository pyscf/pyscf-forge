#!/usr/bin/env python
#
# LiH GBCI nuclear-gradient example.

from pyscf import gbci, gto, scf


mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.5",
    basis="cc-pvdz",
    verbose=4,
)

# GBCI gradients currently require an RHF reference.
mf = scf.RHF(mol).run()

mc = gbci.gbci(
    mf,
    ncas=2,
    nelecas=(1, 1),
    group_a={"atom": [0]},
)
mc.run()

grad = mc.nuc_grad_method().kernel()
print("GBCI nuclear gradient (Eh/Bohr):")
print(grad)
