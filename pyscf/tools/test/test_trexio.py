import pyscf
from pyscf import df
from pyscf.tools import trexio
import os
import numpy as np
import tempfile
import pytest
import trexio as trexio_lib
from pyscf import ao2mo
from pyscf import pbc

DIFF_TOL = 1e-10

_write_2e_int_eri = trexio._write_2e_int_eri

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

#################################################################
# writing `1e_int` and `2e_int` to trexio file
#################################################################

def _trexio_pack_eri(eri, basis):
    basis = basis.upper()
    if basis not in ('AO', 'MO'):
        raise ValueError("basis must be 'AO' or 'MO'")

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'pack.h5')
        _write_2e_int_eri(eri, filename, backend='h5', basis=basis)
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            if basis == 'AO':
                size = trexio_lib.read_ao_2e_int_eri_size(tf)
                idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            else:
                size = trexio_lib.read_mo_2e_int_eri_size(tf)
                idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            return np.asarray(idx, dtype=np.int32).ravel(), np.asarray(val)


def _hermitize(mat):
    mat = np.asarray(mat)
    return 0.5 * (mat + mat.T.conj())


def _squeeze_k1(mat):
    mat = np.asarray(mat)
    return mat[0] if mat.ndim == 3 and mat.shape[0] == 1 else mat


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_to_trexio_rks(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals.h5')

        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            from pyscf.gto import ecp
            potential += _hermitize(ecp.ecp_int(mol0))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        coeff = mf0.mo_coeff
        mo_overlap = _hermitize(coeff.conj().T @ overlap @ coeff)
        mo_kinetic = _hermitize(coeff.conj().T @ kinetic @ coeff)
        mo_potential = _hermitize(coeff.conj().T @ potential @ coeff)
        mo_core = _hermitize(coeff.conj().T @ core @ coeff)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_overlap(tf), mo_overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_kinetic(tf), mo_kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_potential_n_e(tf), mo_potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_core_hamiltonian(tf), mo_core, atol=DIFF_TOL)

        ao_eri = mol0.intor('int2e', aosym='s8')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='AO', aosym='s8')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        mo_eri = ao2mo.kernel(mol0, coeff, compact=True)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='MO', aosym='s8')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_to_trexio_uks(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_uks_integrals.h5')

        mol0 = pyscf.M(atom='O 0 0 0', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UKS()
        mf0.xc = 'LDA'
        mf0.kernel()
        assert mf0.converged

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            from pyscf.gto import ecp
            potential += _hermitize(ecp.ecp_int(mol0))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(
                trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL
            )
            np.testing.assert_allclose(
                trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL
            )

        coeff_alpha, coeff_beta = mf0.mo_coeff
        coeff = np.concatenate([coeff_alpha, coeff_beta], axis=1)
        mo_overlap = _hermitize(coeff.conj().T @ overlap @ coeff)
        mo_kinetic = _hermitize(coeff.conj().T @ kinetic @ coeff)
        mo_potential = _hermitize(coeff.conj().T @ potential @ coeff)
        mo_core = _hermitize(coeff.conj().T @ core @ coeff)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_overlap(tf), mo_overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_kinetic(tf), mo_kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(
                trexio_lib.read_mo_1e_int_potential_n_e(tf), mo_potential, atol=DIFF_TOL
            )
            np.testing.assert_allclose(
                trexio_lib.read_mo_1e_int_core_hamiltonian(tf), mo_core, atol=DIFF_TOL
            )

        ao_eri = mol0.intor('int2e', aosym='s8')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='AO', aosym='s8')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        mo_eri = ao2mo.kernel(mol0, coeff, compact=True)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='MO', aosym='s8')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_cell_gamma_integrals_to_trexio_rks(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'cell_integrals.h5')

        cell0 = pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        gamma_kpt = np.zeros(3)
        mf0 = pbc.scf.RHF(cell0, kpt=gamma_kpt).density_fit()
        mf0.kernel()
        assert mf0.converged

        overlap = _squeeze_k1(_hermitize(mf0.get_ovlp()))
        kinetic = _squeeze_k1(_hermitize(cell0.pbc_intor('int1e_kin', 1, 1)))
        df_builder = (
            mf0.with_df.build()
            if mf0.with_df is not None
            else pbc.df.MDF(cell0, kpts=[gamma_kpt]).build()
        )
        potential = _squeeze_k1(_hermitize(df_builder.get_nuc()))
        if len(getattr(cell0, '_ecpbas', [])) > 0:
            from pyscf.pbc.gto import ecp
            potential += _squeeze_k1(_hermitize(ecp.ecp_int(cell0)))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        coeff = mf0.mo_coeff
        coeff = _squeeze_k1(coeff) if getattr(coeff, 'ndim', 0) == 3 else coeff
        mo_overlap = _hermitize(coeff.conj().T @ overlap @ coeff)
        mo_kinetic = _hermitize(coeff.conj().T @ kinetic @ coeff)
        mo_potential = _hermitize(coeff.conj().T @ potential @ coeff)
        mo_core = _hermitize(coeff.conj().T @ core @ coeff)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_overlap(tf), mo_overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_kinetic(tf), mo_kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_potential_n_e(tf), mo_potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_core_hamiltonian(tf), mo_core, atol=DIFF_TOL)

        df_obj = pbc.df.MDF(cell0).build()
        eri_ao = pbc.df.df_ao2mo.get_eri(df_obj, compact=False)
        nao = cell0.nao_nr()
        eri_ao = eri_ao.reshape(nao, nao, nao, nao)
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(eri_ao, 'AO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        df_obj_mo = pbc.df.MDF(cell0).build()
        mo_eri = df_obj_mo.get_mo_eri((coeff, coeff, coeff, coeff))
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(np.real_if_close (mo_eri), 'MO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_cell_gamma_integrals_to_trexio_uks(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'cell_uks_integrals.h5')

        cell0 = pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        gamma_kpt = np.zeros(3)
        mf0 = pbc.scf.UKS(cell0, kpt=gamma_kpt).density_fit()
        mf0.xc = 'LDA'
        mf0.kernel()
        assert mf0.converged

        overlap = _squeeze_k1(_hermitize(mf0.get_ovlp()))
        kinetic = _squeeze_k1(_hermitize(cell0.pbc_intor('int1e_kin', 1, 1)))
        df_builder = (
            mf0.with_df.build()
            if mf0.with_df is not None
            else pbc.df.MDF(cell0, kpts=[gamma_kpt]).build()
        )
        potential = _squeeze_k1(_hermitize(df_builder.get_nuc()))
        if len(getattr(cell0, '_ecpbas', [])) > 0:
            from pyscf.pbc.gto import ecp
            potential += _squeeze_k1(_hermitize(ecp.ecp_int(cell0)))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        coeff_alpha, coeff_beta = mf0.mo_coeff
        coeff_alpha = _squeeze_k1(coeff_alpha) if getattr(coeff_alpha, 'ndim', 0) == 3 else coeff_alpha
        coeff_beta = _squeeze_k1(coeff_beta) if getattr(coeff_beta, 'ndim', 0) == 3 else coeff_beta
        coeff = np.concatenate([coeff_alpha, coeff_beta], axis=1)
        mo_overlap = _hermitize(coeff.conj().T @ overlap @ coeff)
        mo_kinetic = _hermitize(coeff.conj().T @ kinetic @ coeff)
        mo_potential = _hermitize(coeff.conj().T @ potential @ coeff)
        mo_core = _hermitize(coeff.conj().T @ core @ coeff)

        trexio.write_scf_1e_int_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_overlap(tf), mo_overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_kinetic(tf), mo_kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_potential_n_e(tf), mo_potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_core_hamiltonian(tf), mo_core, atol=DIFF_TOL)

        df_obj = pbc.df.MDF(cell0).build()
        eri_ao = pbc.df.df_ao2mo.get_eri(df_obj, compact=False)
        nao = cell0.nao_nr()
        eri_ao = eri_ao.reshape(nao, nao, nao, nao)
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(eri_ao, 'AO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        df_obj_mo = pbc.df.MDF(cell0).build()
        mo_eri = df_obj_mo.get_mo_eri((coeff, coeff, coeff, coeff))
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(np.real_if_close(mo_eri), 'MO')
        trexio.write_scf_2e_int_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)
