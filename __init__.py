from dh.rdfdh import RDFDH
from dh.udfdh import UDFDH
from pyscf import gto


def DFDH(mol: gto.Mole, *args, **kwargs):
    if mol.spin != 0:
        return UDFDH(mol, *args, **kwargs)
    else:
        return RDFDH(mol, *args, **kwargs)


