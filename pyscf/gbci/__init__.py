from pyscf.gbci.gbci import GBCI

def gbci(mf, ncas, nelecas, ncore=None, group_a=None):
    return GBCI(mf, ncas, nelecas, ncore=ncore, group_a=group_a)
