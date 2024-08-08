import pyscf
from pyscf.tools import trexio

def test_mol():
    filename = 'test.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g**')
    ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12

    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g**', cart=True)
    ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12

def test_mf():
    filename = 'test1.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*')
    mf = mol.RHF().run()
    trexio.to_trexio(mf, filename)
    mf1 = trexio.scf_from_trexio(filename)
    assert abs(mf1.mo_coeff - mf.mo_coeff).max() < 1e-12

    trexio.write_eri(mf._eri, filename)
    eri = trexio.read_eri(filename)
    assert abs(mf._eri - eri).max() < 1e-12
