from pyscf import lib, gto, scf, dft
from pyscf import sftda,tdscf
from pyscf.grad import tduks
from pyscf.sftda_grad import tduks_sf
try:
    import mcfun
except ImportError:
    mcfun = None
    
mol = gto.Mole()
mol.verbose = 6
mol.output = None
mol.spin = 2
mol.atom = 'O 0 0 2.07; O 0 0 0'
mol.unit = 'B'
mol.basis = 'sto-3g'
mol.build()

mf = dft.UKS(mol)
mf.xc = 'svwn' # blyp, b3lyp, tpss
mf.kernel()

mftd1 = sftda.TDA_SF(mf)
# mftd1 = tdscf.TDA(mf)
mftd1.nstates = 5 # the number of excited states
mftd1.extype = 1 # 0 for spin flip up excited energies
mftd1.collinear_samples=200
mftd1.kernel()

mftdg1 = mftd1.Gradients()
# mftdg1 = mftd1.nuc_grad_method()
g1 = mftdg1.kernel(state=2)

print('***********************')
print(g1)