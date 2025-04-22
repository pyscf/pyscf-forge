from pyscf.csf_fci import csf, csf_symm
from pyscf.csf_fci.csfstring import CSFTransformer

CSFFCISolver = csf.CSFFCISolver

def csf_solver(mol=None, smult=None, symm=None):
    if mol and symm is None:
        symm = mol.symmetry
    if symm:
        return csf_symm.FCISolver (mol=mol, smult=smult)
    else:
        return csf.FCISolver (mol=mol, smult=smult)

