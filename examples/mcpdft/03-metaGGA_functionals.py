# patch dft2.libxc (must be done before loading mpcdft module)
from pyscf import dft
from pyscf import dft2
dft.libxc = dft2.libxc
#!/usr/bin/env/python
from pyscf import gto, scf, mcpdft

mol = gto.M (
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

mf = scf.RHF (mol).run ()

# MC23: meta-hybrid on-top functional [PNAS, 122, 1, 2025, e2419413121; https://doi.org/10.1073/pnas.2419413121]

# State-Specific
mc = mcpdft.CASCI(mf, 'MC23', 6, 8)
mc.kernel()

# State-average
nroots=2
mc = mcpdft.CASCI(mf, 'MC23', 6, 8)
mc.fcisolver.nroots=nroots
mc.kernel()[0]
