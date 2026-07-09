from . import rdfdh, udfdh, dhutil
from pyscf import gto


def DFDH(mol: gto.Mole, *args, **kwargs):
    if mol.spin != 0:
        return udfdh.UDFDH(mol, *args, **kwargs)
    else:
        return rdfdh.RDFDH(mol, *args, **kwargs)


