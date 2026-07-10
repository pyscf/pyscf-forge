from . import rdfdh, udfdh, dhutil
from .dh import to_dh
from pyscf import gto


def DFDH(mf_or_mol, *args, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        if mf_or_mol.spin != 0:
            return udfdh.UDFDH(mf_or_mol, *args, **kwargs)
        return rdfdh.RDFDH(mf_or_mol, *args, **kwargs)
    else:
        from pyscf import scf
        if isinstance(mf_or_mol, scf.rhf.RHF):
            return rdfdh.RDFDH(mf_or_mol, *args, **kwargs)
        return udfdh.UDFDH(mf_or_mol, *args, **kwargs)
