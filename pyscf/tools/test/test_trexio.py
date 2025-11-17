import pyscf
from pyscf import df
from pyscf.tools import trexio
import os
import numpy as np
import tempfile
import pytest

DIFF_TOL = 1e-10

#################################################################
# reading/writing `mol` from/to trexio file
#################################################################

## molecule, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g**', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## molecule, general contraction (ccpv5z), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_ccpv5z(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='C', basis='ccpv5z', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## molecule, general contraction (ano), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_ano(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='C', basis='ano', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ccecp_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvqz', ecp='ccecp', cart=cart)
        s0 = mol0.intor('int1e_ovlp')
        t0 = mol0.intor('int1e_kin')
        v0 = mol0.intor('int1e_nuc')
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s1 = mol1.intor('int1e_ovlp')
        t1 = mol1.intor('int1e_kin')
        v1 = mol1.intor('int1e_nuc')
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        assert abs(v0 - v1).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kpt = np.zeros(3)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g**', a=np.diag([3.0, 3.0, 5.0]))
        s0 = cell0.pbc_intor('int1e_ovlp', kpts=kpt)
        t0 = cell0.pbc_intor('int1e_kin', kpts=kpt)
        v0 = cell0.pbc_intor('int1e_nuc', kpts=kpt)
        trexio.to_trexio(cell0, filename)
        cell1 = trexio.mol_from_trexio(filename)
        s1 = cell1.pbc_intor('int1e_ovlp', kpts=kpt)
        t1 = cell1.pbc_intor('int1e_kin', kpts=kpt)
        v1 = cell1.pbc_intor('int1e_nuc', kpts=kpt)
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        #assert abs(v0 - v1).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_grid_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g**', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        s0 = np.asarray(cell0.pbc_intor('int1e_ovlp', kpts=kpts0))
        t0 = np.asarray(cell0.pbc_intor('int1e_kin', kpts=kpts0))
        v0 = np.asarray(cell0.pbc_intor('int1e_nuc', kpts=kpts0))
        trexio.to_trexio(cell0, filename)
        cell1 = trexio.mol_from_trexio(filename)
        kpts1 = kpts0
        #kpts1 = cell1.make_kpts(kmesh)
        s1 = np.asarray(cell1.pbc_intor('int1e_ovlp', kpts=kpts1))
        t1 = np.asarray(cell1.pbc_intor('int1e_kin', kpts=kpts1))
        v1 = np.asarray(cell1.pbc_intor('int1e_nuc', kpts=kpts1))
        assert abs(s0 - s1).max() < DIFF_TOL
        assert abs(t0 - t1).max() < DIFF_TOL
        #assert abs(v0 - v1).max() < DIFF_TOL

#################################################################
# reading/writing `mf` from/to trexio file
#################################################################

## molecule, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g', cart=cart)
        mf0 = mol0.RHF().density_fit()
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## molecule, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='6-31g', spin=2, cart=cart)
        mf0 = mol0.UHF().density_fit()
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvdz', ecp='ccecp', cart=cart)
        mf0 = mol0.RHF().run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_gamma_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        mf0 = pyscf.pbc.scf.RKS(cell0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=general, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_general_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.RKS(cell0, kpt=kpt0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=single_grid, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_single_grid_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 1)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(np.asarray(mf1.mo_coeff) - np.asarray(mf0.mo_coeff)).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_grid_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(np.asarray(mf1.mo_coeff) - np.asarray(mf0.mo_coeff)).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_gamma_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        mf0 = pyscf.pbc.scf.UKS(cell0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=general, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_general_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.UKS(cell0, kpt=kpt0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(mf1.mo_coeff - mf0.mo_coeff).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_single_grid_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 1)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KUKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(np.ravel(mf1.mo_coeff) - np.ravel(mf0.mo_coeff)).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_grid_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KUKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = 'LDA'
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mf1 = trexio.scf_from_trexio(filename)
        assert abs(np.asarray(mf1.mo_coeff) - np.asarray(mf0.mo_coeff)).max() < DIFF_TOL

#################################################################
# reading/writing `mol` from/to trexio file + SCF run.
#################################################################
## molecule, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g', cart=cart)
        auxbasis = df.make_auxbasis(mol0)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, RKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.RKS(cell0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        mf1 = pyscf.pbc.scf.RKS(cell1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=gamma, segment contraction (6-31g), all-electron, UKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.UKS(cell0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        mf1 = pyscf.pbc.scf.UKS(cell1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL


## PBC, k=general, segment contraction (6-31g), all-electron, RKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_general_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.RKS(cell0, kpt=kpt0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpt1 = cell1.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf1 = pyscf.pbc.scf.RKS(cell1, kpt=kpt1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=general, segment contraction (6-31g), all-electron, UKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_general_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.UKS(cell0, kpt=kpt0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpt1 = cell1.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf1 = pyscf.pbc.scf.UKS(cell1, kpt=kpt1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## PBC, k=grid, segment contraction (6-31g), all-electron, RKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_grid_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, 'test.h5')
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g', a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = 'LDA'
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpts1 = cell1.make_kpts(kmesh)
        mf1 = pyscf.pbc.scf.KRKS(cell1, kpts=kpts1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = 'LDA'
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## molecule, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='6-31g', spin=2, cart=cart)
        auxbasis = df.make_auxbasis(mol0)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.UHF().density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.UHF().density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvdz', ecp='ccecp', cart=cart)
        auxbasis = df.make_auxbasis(mol0)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='F 0 0 0; F 0 0 1', basis='ccecp-ccpvdz', ecp='ccecp', cart=cart)
        auxbasis = df.make_auxbasis(mol0)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='ccecp-ccpvdz', ecp='ccecp', cart=cart)
        auxbasis = df.make_auxbasis(mol0)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.RHF().density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.RHF().density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL

## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='ccecp-ccpvdz', ecp='ccecp', spin=2, cart=cart)
        auxbasis = df.make_auxbasis(mol0)
        trexio.to_trexio(mol0, filename)
        mf0 = mol0.UHF().density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.run()
        e0 = mf0.e_tot
        mol1 = trexio.mol_from_trexio(filename)
        mf1 = mol1.UHF().density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL
