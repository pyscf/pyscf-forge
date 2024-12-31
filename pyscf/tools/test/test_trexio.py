import pyscf
from pyscf.tools import trexio

def test_mol_ae_6_31g():
    filename = 'test_mol_ae_6_31g_sphe.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g**', cart=False)
    ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12

    filename = 'test_mol_ae_6_31g_cart.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g**', cart=True)
    ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12

def test_mol_general_contraction():
    filename = 'test_mol_general_contraction_sphe.h5'
    mol = pyscf.M(atom='C', basis='ano')
    ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12

def test_mol_ccecp_ccpvqz():
    filename = 'test_mol_ccecp_ccpvqz_sphe.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-cc-pVQZ', ecp='ccECP', cart=False)
    # ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    ''' Todo: mol_from_trexio for ecp is yet implemented.
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12
    '''

    filename = 'test_mol_ccecp_ccpvqz_cart.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-cc-pVQZ', ecp='ccECP', cart=True)
    # ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    ''' Todo: mol_from_trexio for ecp is yet implemented.
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12
    '''

def test_mol_ae_ccpvdz():
    filename = 'test_mol_ae_ccpvdz_sphe.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', cart=False)
    ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12

    filename = 'test_mol_ae_ccpvdz_cart.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', cart=True)
    ref = mol.intor('int1e_ovlp')
    trexio.to_trexio(mol, filename)
    mol = trexio.mol_from_trexio(filename)
    s1 = mol.intor('int1e_ovlp')
    assert abs(ref - s1).max() < 1e-12

def test_mf_ae_6_31g():
    filename = 'test_mf_ae_6_31g_sphe.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=False)
    mf = mol.RHF().run()
    #print(mf.mo_coeff)
    trexio.to_trexio(mf, filename)
    mf1 = trexio.scf_from_trexio(filename)
    assert abs(mf1.mo_coeff - mf.mo_coeff).max() < 1e-12
    trexio.write_eri(mf._eri, filename)
    eri = trexio.read_eri(filename)
    assert abs(mf._eri - eri).max() < 1e-12

    filename = 'test_mf_ae_6_31g_cart.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=True)
    mf = mol.RHF().run()
    #print(mf.mo_coeff)
    trexio.to_trexio(mf, filename)
    mf1 = trexio.scf_from_trexio(filename)
    assert abs(mf1.mo_coeff - mf.mo_coeff).max() < 1e-12
    trexio.write_eri(mf._eri, filename)
    eri = trexio.read_eri(filename)
    assert abs(mf._eri - eri).max() < 1e-12

def test_mf_ccecp_ccpvqz():
    filename = 'test_mf_ccecp_ccpvqz_sphe.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-cc-pVQZ', ecp='ccECP', cart=False)
    mf = mol.RHF().run()
    trexio.to_trexio(mf, filename)
    ''' Todo: scf_from_trexio for ecp is yet implemented.
    mf1 = trexio.scf_from_trexio(filename)
    assert abs(mf1.mo_coeff - mf.mo_coeff).max() < 1e-12
    trexio.write_eri(mf._eri, filename)
    eri = trexio.read_eri(filename)
    assert abs(mf._eri - eri).max() < 1e-12
    '''

    filename = 'test_mf_ccecp_ccpvqz_cart.h5'
    mol = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-cc-pVQZ', ecp='ccECP', cart=True)
    mf = mol.RHF().run()
    trexio.to_trexio(mf, filename)
    ''' Todo: scf_from_trexio for ecp is yet implemented.
    mf1 = trexio.scf_from_trexio(filename)
    assert abs(mf1.mo_coeff - mf.mo_coeff).max() < 1e-12
    trexio.write_eri(mf._eri, filename)
    eri = trexio.read_eri(filename)
    assert abs(mf._eri - eri).max() < 1e-12
    '''

if __name__ == "__main__":
    #test_mol_ae_6_31g()
    #test_mol_ccecp_ccpvqz()
    #test_mol_ae_ccpvdz()
    #test_mf_ae_6_31g()
    test_mf_ccecp_ccpvqz()
