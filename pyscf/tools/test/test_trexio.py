import pyscf
from pyscf import df
from pyscf.tools import trexio
import trexio as trexio_lib
import os
import numpy as np
import tempfile
import pytest
import trexio as trexio_lib
from pyscf import ao2mo
from pyscf import pbc
from pyscf.pbc import df as pbcdf

DIFF_TOL = 1e-10

_write_2e_int_eri = trexio._write_2e_int_eri

#################################################################
# reading/writing `mol` from/to trexio file
#################################################################


def _get_integrals(mol, kpts=None):
    if isinstance(mol, pyscf.pbc.gto.Cell):
        if kpts is None:
            kpts = np.zeros((1, 3))

        s = np.asarray(mol.pbc_intor("int1e_ovlp", kpts=kpts))
        t = np.asarray(mol.pbc_intor("int1e_kin", kpts=kpts))
        v = np.asarray(mol.pbc_intor("int1e_nuc", kpts=kpts))

    else:
        s = mol.intor("int1e_ovlp")
        t = mol.intor("int1e_kin")
        v = mol.intor("int1e_nuc")

    return s, t, v


def _assert_s_t_v_roundtrip(s0, t0, v0, s1, t1, v1):
    assert abs(s0 - s1).max() < DIFF_TOL
    assert abs(t0 - t1).max() < DIFF_TOL
    assert abs(v0 - v1).max() < DIFF_TOL


## molecule, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(atom="H 0 0 0; F 0 0 1", basis="6-31g**", cart=cart)
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s0, t0, v0 = _get_integrals(mol0)
        s1, t1, v1 = _get_integrals(mol1)
        _assert_s_t_v_roundtrip(s0, t0, v0, s1, t1, v1)


## molecule, general contraction (ccpv5z), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_ccpv5z(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(atom="C", basis="ccpv5z", cart=cart)
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s0, t0, v0 = _get_integrals(mol0)
        s1, t1, v1 = _get_integrals(mol1)
        _assert_s_t_v_roundtrip(s0, t0, v0, s1, t1, v1)


## molecule, general contraction (ano), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ae_ano(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(atom="C", basis="ano", cart=cart)
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s0, t0, v0 = _get_integrals(mol0)
        s1, t1, v1 = _get_integrals(mol1)
        _assert_s_t_v_roundtrip(s0, t0, v0, s1, t1, v1)


## molecule, segment contraction (ccecp-cc-pVQZ), ccecp
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_ccecp_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(
            atom="H 0 0 0; F 0 0 1", basis="ccecp-ccpvqz", ecp="ccecp", cart=cart
        )
        trexio.to_trexio(mol0, filename)
        mol1 = trexio.mol_from_trexio(filename)
        s0, t0, v0 = _get_integrals(mol0)
        s1, t1, v1 = _get_integrals(mol1)
        _assert_s_t_v_roundtrip(s0, t0, v0, s1, t1, v1)


## PBC, k=gamma, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kpt = np.zeros(3)
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(
            atom="H 0 0 0; H 0 0 1", basis="6-31g**", a=np.diag([3.0, 3.0, 5.0])
        )
        trexio.to_trexio(cell0, filename)
        cell1 = trexio.mol_from_trexio(filename)
        s0, t0, v0 = _get_integrals(cell0, kpts=kpt)
        s1, t1, v1 = _get_integrals(cell1, kpts=kpt)
        _assert_s_t_v_roundtrip(s0, t0, v0, s1, t1, v1)

## PBC, k=grid, segment contraction (6-31g), all-electron
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_grid_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(
            atom="H 0 0 0; H 0 0 1", basis="6-31g**", a=np.diag([3.0, 3.0, 5.0])
        )
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        cell1 = trexio.mol_from_trexio(filename)
        s0, t0, v0 = _get_integrals(cell0, kpts=kpts0)
        s1, t1, v1 = _get_integrals(cell1, kpts=kpts0)
        _assert_s_t_v_roundtrip(s0, t0, v0, s1, t1, v1)


#################################################################
# reading/writing `mf` from/to trexio file
#################################################################


def _mo_coeff_from_trexio(filename):
    mol = trexio.mol_from_trexio(filename)
    with trexio_lib.File(filename, "r", back_end=trexio_lib.TREXIO_AUTO) as tf:
        pbc_mode = trexio_lib.read_pbc_periodic(tf)

    mo_coeff_k = []

    if pbc_mode:
        with trexio_lib.File(filename, "r", back_end=trexio_lib.TREXIO_AUTO) as tf:
            k_point_num = trexio_lib.read_pbc_k_point_num(tf)
            kpts = trexio_lib.read_pbc_k_point(tf)
            mo_type = trexio_lib.read_mo_type(tf)
            mo_num = trexio_lib.read_mo_num(tf)
            mo_energy = trexio_lib.read_mo_energy(tf)
            mo_coeff = trexio_lib.read_mo_coefficient(tf)
            mo_coeff_im = (
                trexio_lib.read_mo_coefficient_im(tf)
                if trexio_lib.has_mo_coefficient_im(tf)
                else None
            )
            mo_occ = trexio_lib.read_mo_occupation(tf)
            mo_spin = trexio_lib.read_mo_spin(tf)
            mo_k_point = (
                trexio_lib.read_mo_k_point(tf)
                if trexio_lib.has_mo_k_point(tf)
                else np.zeros(mo_num, dtype=int)
            )

        mo_coeff = mo_coeff + 1j * mo_coeff_im if mo_coeff_im is not None else mo_coeff

        nao = mol.nao
        idx = trexio._order_ao_index(mol)
        uniq = set(np.unique(mo_spin).tolist())

        if k_point_num == 0:
            k_point_num = 1
            mo_k_point[:] = 0

        for ik in range(k_point_num):
            mask = mo_k_point == ik
            if not np.any(mask):
                mo_coeff_k.append(np.empty((nao, 0), dtype=mo_coeff.dtype))
                continue

            coeff_k = mo_coeff[mask, :]
            spin_k = mo_spin[mask]

            if uniq == {0, 1}:  # UHF
                up = coeff_k[spin_k == 0]
                dn = coeff_k[spin_k == 1]
                up_pyscf = np.empty((nao, up.shape[0]), dtype=mo_coeff.dtype)
                dn_pyscf = np.empty((nao, dn.shape[0]), dtype=mo_coeff.dtype)
                up_pyscf[idx, :] = up.T
                dn_pyscf[idx, :] = dn.T
                mo_coeff_k.append((up_pyscf, dn_pyscf))
            else:  # RHF
                coeff_pyscf = np.empty((nao, coeff_k.shape[0]), dtype=mo_coeff.dtype)
                coeff_pyscf[idx, :] = coeff_k.T
                mo_coeff_k.append(coeff_pyscf)

    else:
        # non-PBC
        with trexio_lib.File(filename, "r", back_end=trexio_lib.TREXIO_AUTO) as tf:
            mo_type = trexio_lib.read_mo_type(tf)
            mo_num = trexio_lib.read_mo_num(tf)
            mo_energy = trexio_lib.read_mo_energy(tf)
            mo_coeff = trexio_lib.read_mo_coefficient(tf)
            mo_coeff_im = (
                trexio_lib.read_mo_coefficient_im(tf)
                if trexio_lib.has_mo_coefficient_im(tf)
                else None
            )
            mo_occ = trexio_lib.read_mo_occupation(tf)
            mo_spin = trexio_lib.read_mo_spin(tf)

        mo_coeff = mo_coeff + 1j * mo_coeff_im if mo_coeff_im is not None else mo_coeff

        nao = mol.nao
        idx = trexio._order_ao_index(mol)
        uniq = set(np.unique(mo_spin).tolist())

        if uniq == {0, 1}:  # UHF
            i_up = np.where(mo_spin == 0)[0]
            i_dn = np.where(mo_spin == 1)[0]
            up = mo_coeff[i_up, :]
            dn = mo_coeff[i_dn, :]
            up_pyscf = np.empty((nao, up.shape[0]), dtype=mo_coeff.dtype)
            dn_pyscf = np.empty((nao, dn.shape[0]), dtype=mo_coeff.dtype)
            up_pyscf[idx, :] = up.T
            dn_pyscf[idx, :] = dn.T
            mo_coeff_k.append((up_pyscf, dn_pyscf))
        else:
            coeff = np.empty((nao, mo_num), dtype=mo_coeff.dtype)
            coeff[idx, :] = mo_coeff.T
            mo_coeff_k.append(coeff)

    if isinstance(mol, pyscf.pbc.gto.Cell):
        # UKS with k-points: ([up_k...], [dn_k...])
        if len(mo_coeff_k) == 1:
            if isinstance(mo_coeff_k[0], tuple):  # UHF 1-kpt: (u, d)
                return np.stack(mo_coeff_k[0])  # (2, N, M)
            return mo_coeff_k[0]  # RHF 1-kpt: coeff

        if mo_coeff_k and isinstance(mo_coeff_k[0], tuple):
            mo_coeff_k = ([x[0] for x in mo_coeff_k], [x[1] for x in mo_coeff_k])
        return mo_coeff_k
    return mo_coeff_k[0]


def _assert_mo_coeff_roundtrip(mc0, mc1):
    np.testing.assert_array_almost_equal(mc0, mc1, decimal=9)


## molecule, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(atom="H 0 0 0; F 0 0 1", basis="6-31g", cart=cart)
        mf0 = mol0.RHF().density_fit()
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## molecule, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(atom="H 0 0 0; H 0 0 1", basis="6-31g", spin=2, cart=cart)
        mf0 = mol0.UHF().density_fit()
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(
            atom="H 0 0 0; F 0 0 1", basis="ccecp-ccpvdz", ecp="ccecp", cart=cart
        )
        mf0 = mol0.RHF().run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## PBC, k=gamma, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_gamma_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        mf0 = pyscf.pbc.scf.RKS(cell0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## PBC, k=general, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_general_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.RKS(cell0, kpt=kpt0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## PBC, k=single_grid, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_single_grid_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 1)
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff[0], mo_coeff1)


## PBC, k=grid, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_grid_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## PBC, k=gamma, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_gamma_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        mf0 = pyscf.pbc.scf.UKS(cell0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## PBC, k=general, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_general_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.UKS(cell0, kpt=kpt0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


## PBC, k=grid, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_single_grid_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 1)
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KUKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(
            np.stack([mf0.mo_coeff[0][0], mf0.mo_coeff[1][0]]), mo_coeff1
        )


## PBC, k=grid, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mf_k_grid_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KUKS(cell0, kpts=kpts0).density_fit()
        mf0.xc = "LDA"
        mf0.run()
        trexio.to_trexio(mf0, filename)
        mo_coeff1 = _mo_coeff_from_trexio(filename)
        _assert_mo_coeff_roundtrip(mf0.mo_coeff, mo_coeff1)


#################################################################
# reading/writing `mol` from/to trexio file + SCF run.
#################################################################


def _assert_e_roundtrip(e0, e1):
    assert abs(e0 - e1).max() < DIFF_TOL


## molecule, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(atom="H 0 0 0; F 0 0 1", basis="6-31g", cart=cart)
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
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.RKS(cell0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = "LDA"
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        mf1 = pyscf.pbc.scf.RKS(cell1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = "LDA"
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL


## PBC, k=gamma, segment contraction (6-31g), all-electron, UKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_gamma_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.UKS(cell0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = "LDA"
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        mf1 = pyscf.pbc.scf.UKS(cell1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = "LDA"
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL


## PBC, k=general, segment contraction (6-31g), all-electron, RKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_general_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.RKS(cell0, kpt=kpt0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = "LDA"
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpt1 = cell1.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf1 = pyscf.pbc.scf.RKS(cell1, kpt=kpt1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = "LDA"
        mf1.run()
        e1 = mf1.e_tot
        assert abs(e0 - e1).max() < DIFF_TOL


## PBC, k=general, segment contraction (6-31g), all-electron, UKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_general_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        kfrac = (0.25, 0.25, 0.25)
        cell0 = pyscf.pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        trexio.to_trexio(cell0, filename)
        kpt0 = cell0.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf0 = pyscf.pbc.scf.UKS(cell0, kpt=kpt0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = "LDA"
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpt1 = cell1.make_kpts([1, 1, 1], scaled_center=kfrac)[0]
        mf1 = pyscf.pbc.scf.UKS(cell1, kpt=kpt1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = "LDA"
        mf1.run()
        e1 = mf1.e_tot
        _assert_e_roundtrip(e0, e1)


## PBC, k=grid, segment contraction (6-31g), all-electron, RKS
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_cell_k_grid_scf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        kmesh = (1, 1, 2)
        filename = os.path.join(d, "test.h5")
        cell0 = pyscf.pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom="H 0 0 0; H 0 0 1", basis="6-31g", a=np.diag([3.0, 3.0, 5.0]))
        auxbasis = df.make_auxbasis(cell0)
        kpts0 = cell0.make_kpts(kmesh)
        trexio.to_trexio(cell0, filename)
        mf0 = pyscf.pbc.scf.KRKS(cell0, kpts=kpts0).density_fit()
        mf0.with_df.auxbasis = auxbasis
        mf0.xc = "LDA"
        mf0.run()
        e0 = mf0.e_tot
        cell1 = trexio.mol_from_trexio(filename)
        kpts1 = cell1.make_kpts(kmesh)
        mf1 = pyscf.pbc.scf.KRKS(cell1, kpts=kpts1).density_fit()
        mf1.with_df.auxbasis = auxbasis
        mf1.xc = "LDA"
        mf1.run()
        e1 = mf1.e_tot
        _assert_e_roundtrip(e0, e1)


## molecule, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_scf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(atom="H 0 0 0; H 0 0 1", basis="6-31g", spin=2, cart=cart)
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
        _assert_e_roundtrip(e0, e1)


## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_rhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(
            atom="H 0 0 0; F 0 0 1", basis="ccecp-ccpvdz", ecp="ccecp", cart=cart
        )
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
        _assert_e_roundtrip(e0, e1)

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(
            atom="F 0 0 0; F 0 0 1", basis="ccecp-ccpvdz", ecp="ccecp", cart=cart
        )
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
        _assert_e_roundtrip(e0, e1)

    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(
            atom="H 0 0 0; H 0 0 1", basis="ccecp-ccpvdz", ecp="ccecp", cart=cart
        )
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
        _assert_e_roundtrip(e0, e1)


## molecule, segment contraction (ccecp-cc-pVQZ), ccecp, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mol_uhf_ccecp_ccpvqz(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.h5")
        mol0 = pyscf.M(
            atom="H 0 0 0; F 0 0 1",
            basis="ccecp-ccpvdz",
            ecp="ccecp",
            spin=2,
            cart=cart,
        )
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
        _assert_e_roundtrip(e0, e1)

#################################################################
# writing `1e_int` and `2e_int` to trexio file
#################################################################

def _trexio_pack_eri(eri, basis, sym='s1'):
    basis = basis.upper()
    sym = sym.lower()
    if basis not in ('AO', 'MO'):
        raise ValueError("basis must be 'AO' or 'MO'")

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'pack.h5')
        _write_2e_int_eri(eri, filename, backend='h5', basis=basis, sym=sym)
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            if basis == 'AO':
                size = trexio_lib.read_ao_2e_int_eri_size(tf)
                idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            else:
                size = trexio_lib.read_mo_2e_int_eri_size(tf)
                idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            return np.asarray(idx, dtype=np.int32).ravel(), np.asarray(val)


def _expand_trexio_eri(idx, val, n):
    idx = np.asarray(idx, dtype=np.int64).reshape(-1, 4)
    val = np.asarray(val)
    w_chem = np.zeros((n, n, n, n))
    for (p, r, q, s), v in zip(idx, val):
        i, j, k, l = p, q, r, s
        w_chem[i, j, k, l] = v
        w_chem[j, i, k, l] = v
        w_chem[i, j, l, k] = v
        w_chem[j, i, l, k] = v
        w_chem[k, l, i, j] = v
        w_chem[l, k, i, j] = v
        w_chem[k, l, j, i] = v
        w_chem[l, k, j, i] = v
    return w_chem.transpose(0, 2, 1, 3)


def _hermitize(mat):
    mat = np.asarray(mat)
    return 0.5 * (mat + mat.T.conj())


def _squeeze_k1(mat):
    mat = np.asarray(mat)
    return mat[0] if mat.ndim == 3 and mat.shape[0] == 1 else mat


def _take_gamma(mat):
    """Pick the gamma (first) k-block if present."""
    mat = np.asarray(mat)
    return mat[0] if mat.ndim >= 3 else mat


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_sym_s1_to_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals.h5')

        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            potential += _hermitize(mol0.intor('ECPscalar'))
        core = kinetic + potential


        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
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

        trexio.write_1e_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_overlap(tf), mo_overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_kinetic(tf), mo_kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_potential_n_e(tf), mo_potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_core_hamiltonian(tf), mo_core, atol=DIFF_TOL)

        ao_eri = mol0.intor('int2e', aosym='s1')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO')
        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        mo_eri = ao2mo.kernel(mol0, coeff, compact=False)
        nmo = coeff.shape[1]
        mo_eri = mo_eri.reshape(nmo, nmo, nmo, nmo)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_sym_s1_to_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_uhf_integrals.h5')

        mol0 = pyscf.M(atom='O 0 0 0', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        assert mf0.converged

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            potential += _hermitize(mol0.intor('ECPscalar'))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
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

        trexio.write_1e_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_overlap(tf), mo_overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_mo_1e_int_kinetic(tf), mo_kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(
                trexio_lib.read_mo_1e_int_potential_n_e(tf), mo_potential, atol=DIFF_TOL
            )
            np.testing.assert_allclose(
                trexio_lib.read_mo_1e_int_core_hamiltonian(tf), mo_core, atol=DIFF_TOL
            )

        ao_eri = mol0.intor('int2e', aosym='s1')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO')
        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        mo_eri = ao2mo.kernel(mol0, coeff, compact=False)
        nmo = coeff.shape[1]
        mo_eri = mo_eri.reshape(nmo, nmo, nmo, nmo)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_sym_s4_to_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_s4.h5')

        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()

        trexio.to_trexio(mf0, filename)

        ao_eri = mol0.intor('int2e', aosym='s4')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO', sym='s4')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s4')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        coeff = mf0.mo_coeff
        mo_eri = ao2mo.kernel(mol0, coeff, compact=True)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO', sym='s4')
        trexio.write_2e_eri(mf0, filename, basis='MO', sym='s4')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_sym_s4_to_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_uhf_integrals_s4.h5')

        mol0 = pyscf.M(atom='O 0 0 0', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)

        ao_eri = mol0.intor('int2e', aosym='s4')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO', sym='s4')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s4')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        coeff_alpha, coeff_beta = mf0.mo_coeff
        coeff = np.concatenate([coeff_alpha, coeff_beta], axis=1)
        mo_eri = ao2mo.kernel(mol0, coeff, compact=True)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO', sym='s4')
        trexio.write_2e_eri(mf0, filename, basis='MO', sym='s4')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_sym_s8_to_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_s8.h5')

        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()

        trexio.to_trexio(mf0, filename)

        ao_eri = mol0.intor('int2e', aosym='s8')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO', sym='s8')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s8')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_molecule_integrals_sym_s8_to_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_uhf_integrals_s8.h5')

        mol0 = pyscf.M(atom='O 0 0 0', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)

        ao_eri = mol0.intor('int2e', aosym='s8')
        ao_idx_exp, ao_val_exp = _trexio_pack_eri(ao_eri, 'AO', sym='s8')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s8')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_cell_gamma_integrals_sym_s1_to_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'cell_integrals.h5')

        cell0 = pbc.gto.Cell()
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        gamma_kpt = np.zeros(3)
        mf0 = pbc.scf.RHF(cell0, kpt=gamma_kpt).density_fit()
        mf0.kernel()
        assert mf0.converged

        overlap = _hermitize(_take_gamma(mf0.get_ovlp()))
        kinetic = _hermitize(_take_gamma(cell0.pbc_intor('int1e_kin', 1, 1)))
        df_builder = (
            mf0.with_df.build()
            if mf0.with_df is not None
            else pbc.df.MDF(cell0, kpts=[gamma_kpt]).build()
        )
        potential = _hermitize(_take_gamma(df_builder.get_nuc()))
        if len(getattr(cell0, '_ecpbas', [])) > 0:
            from pyscf.pbc.gto import ecp
            potential += _hermitize(_take_gamma(ecp.ecp_int(cell0)))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
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

        trexio.write_1e_eri(mf0, filename, basis='MO')
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
        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        df_obj_mo = pbc.df.MDF(cell0).build()
        mo_eri = df_obj_mo.get_mo_eri((coeff, coeff, coeff, coeff))
        mo_eri = np.real_if_close(mo_eri)
        nmo = coeff.shape[1]
        if mo_eri.ndim == 2:
            # Pair-pair (s4) -> expand to full tensor
            mo_eri = ao2mo.restore(1, ao2mo.restore(4, mo_eri, nmo), nmo)
        elif mo_eri.ndim < 4:
            mo_eri = ao2mo.restore(1, mo_eri, nmo)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_write_cell_gamma_integrals_sym_s1_to_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'cell_uhf_integrals.h5')

        cell0 = pbc.gto.Cell()
        cell0.spin = 2
        cell0.cart = cart
        cell0.build(atom='H 0 0 0; H 0 0 1', basis='6-31g*', a=np.diag([3.0, 3.0, 5.0]))
        gamma_kpt = np.zeros(3)
        mf0 = pbc.scf.UHF(cell0, kpt=gamma_kpt).density_fit()
        mf0.kernel()
        assert mf0.converged

        overlap = _hermitize(_take_gamma(mf0.get_ovlp()))
        kinetic = _hermitize(_take_gamma(cell0.pbc_intor('int1e_kin', 1, 1)))
        df_builder = (
            mf0.with_df.build()
            if mf0.with_df is not None
            else pbc.df.MDF(cell0, kpts=[gamma_kpt]).build()
        )
        potential = _hermitize(_take_gamma(df_builder.get_nuc()))
        if len(getattr(cell0, '_ecpbas', [])) > 0:
            from pyscf.pbc.gto import ecp
            potential += _hermitize(_take_gamma(ecp.ecp_int(cell0)))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        coeff_alpha, coeff_beta = mf0.mo_coeff
        coeff_alpha = _take_gamma(coeff_alpha)
        coeff_beta = _take_gamma(coeff_beta)
        coeff = np.concatenate([coeff_alpha, coeff_beta], axis=1)
        mo_overlap = _hermitize(coeff.conj().T @ overlap @ coeff)
        mo_kinetic = _hermitize(coeff.conj().T @ kinetic @ coeff)
        mo_potential = _hermitize(coeff.conj().T @ potential @ coeff)
        mo_core = _hermitize(coeff.conj().T @ core @ coeff)

        trexio.write_1e_eri(mf0, filename, basis='MO')
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
        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), ao_idx_exp)
            np.testing.assert_allclose(np.asarray(val), ao_val_exp, atol=DIFF_TOL)

        df_obj_mo = pbc.df.MDF(cell0).build()
        mo_eri = df_obj_mo.get_mo_eri((coeff, coeff, coeff, coeff))
        mo_eri = np.real_if_close(mo_eri)
        nmo = coeff.shape[1]
        if mo_eri.ndim == 2:
            mo_eri = ao2mo.restore(1, ao2mo.restore(4, mo_eri, nmo), nmo)
        elif mo_eri.ndim < 4:
            mo_eri = ao2mo.restore(1, mo_eri, nmo)
        mo_idx_exp, mo_val_exp = _trexio_pack_eri(mo_eri, 'MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_mo_2e_int_eri(tf)
            size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, size)
            assert n_read == size
            np.testing.assert_array_equal(np.asarray(idx, dtype=np.int32).ravel(), mo_idx_exp)
            np.testing.assert_allclose(np.asarray(val), mo_val_exp, atol=DIFF_TOL)


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s1_in_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy.h5')

        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            potential += _hermitize(mol0.intor('ECPscalar'))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size

        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            assert ao_eri_size == nao ** 4
            offset = 0
            W_ao_phys = np.zeros((nao, nao, nao, nao))  # raw order as stored
            while offset < ao_eri_size:
                bufsize = min(BUFSIZE, ao_eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_ao_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    p, r, q, s = np.asarray(buf_idx, dtype=np.int64).T  # stored as (p,r,q,s)
                    W_ao_phys[p, r, q, s] = buf_val
                if feof:
                    break

            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_ao)
            K = np.einsum('prqs,rs->pq', W_ao_chem, dm_ao)
            vhf = J - 0.5 * K
            e_ao = e_nn
            e_ao += np.einsum('pq,pq->', dm_ao, core_ao)
            e_ao += 0.5 * np.einsum('pq,pq->', dm_ao, vhf)

            # MO energy reconstruction
            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break
            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            offset = 0
            G_mo_phys = np.zeros((n, n, n, n))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn
            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * np.tensordot(G_mo_phys, W_mo_phys, axes=4)

        assert abs(e_ao - mf0.e_tot) < 1e-8
        assert abs(e_mo - mf0.e_tot) < 1e-8


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_crystal_integrals_sym_s1_in_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'pbc_gamma_rdm_energy.h5')

        cell = pyscf.pbc.gto.Cell()
        cell.cart = cart
        cell.unit = 'Bohr'
        cell.build(
            atom='H 0 0 0; H 0 0 1.4',
            basis='sto-3g',
            a=np.diag([3.0, 3.0, 5.0]),
        )

        mf0 = pyscf.pbc.scf.RHF(cell)
        mf0.with_df = pbcdf.MDF(cell).build()
        mf0.run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)
        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)
            if abs(e_nn) < 1e-12:
                e_nn_mol = cell.energy_nuc()
                if abs(e_nn_mol) > 1e-12:
                    e_nn = e_nn_mol
            madelung = 0.0
            try:
                madelung = trexio_lib.read_pbc_madelung(tf)
                madelung = float(np.asarray(madelung).ravel()[0])
            except trexio_lib.Error:
                madelung = 0.0

            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            offset = 0
            G_mo_phys = np.zeros((n, n, n, n))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn - madelung
            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * np.tensordot(G_mo_phys, W_mo_phys, axes=4)

        assert abs(e_mo - mf0.e_tot) < 1e-6


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_crystal_integrals_sym_s1_in_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'pbc_gamma_rdm_energy_uhf.h5')

        cell = pyscf.pbc.gto.Cell()
        cell.cart = cart
        cell.unit = 'Bohr'
        cell.spin = 2
        cell.build(
            atom='H 0 0 0; H 0 0 1.4',
            basis='sto-3g',
            a=np.diag([3.0, 3.0, 5.0]),
        )

        mf0 = pyscf.pbc.scf.UHF(cell)
        mf0.with_df = pbcdf.MDF(cell).build()
        mf0.run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)
        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)
            if abs(e_nn) < 1e-12:
                e_nn_mol = cell.energy_nuc()
                if abs(e_nn_mol) > 1e-12:
                    e_nn = e_nn_mol
            madelung = 0.0
            try:
                madelung = trexio_lib.read_pbc_madelung(tf)
                madelung = float(np.asarray(madelung).ravel()[0])
            except trexio_lib.Error:
                madelung = 0.0

            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            spin_rdm = (
                trexio_lib.has_rdm_1e_up(tf)
                and trexio_lib.has_rdm_2e_upup(tf)
                and trexio_lib.has_rdm_2e_dndn(tf)
                and trexio_lib.has_rdm_2e_updn(tf)
            )

            assert spin_rdm

            rdm2_size = trexio_lib.read_rdm_2e_upup_size(tf)
            n_up = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_up ** 4

            rdm2_size = trexio_lib.read_rdm_2e_dndn_size(tf)
            n_dn = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_dn ** 4

            rdm2_size = trexio_lib.read_rdm_2e_updn_size(tf)
            assert rdm2_size == (n_up * n_dn) ** 2

            offset = 0
            G_uu = np.zeros((n_up, n_up, n_up, n_up))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_upup(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_uu[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_dd = np.zeros((n_dn, n_dn, n_dn, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_dndn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_dd[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_ud = np.zeros((n_up, n_dn, n_up, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_updn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_ud[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn - madelung
            off = n_up

            W_uu = W_mo_phys[:n_up, :n_up, :n_up, :n_up]
            W_dd = W_mo_phys[off:off + n_dn, off:off + n_dn, off:off + n_dn, off:off + n_dn]
            W_ud = W_mo_phys[:n_up, off:off + n_dn, :n_up, off:off + n_dn]

            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * (
                np.tensordot(G_uu, W_uu, axes=4)
                + np.tensordot(G_dd, W_dd, axes=4)
                + 2.0 * np.tensordot(G_ud, W_ud, axes=4)
            )

        assert abs(e_mo - mf0.e_tot) < 1e-6


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_crystal_integrals_sym_s1_in_trexio_rhf_ccecp(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'pbc_gamma_rdm_energy_ccecp.h5')

        cell = pyscf.pbc.gto.Cell()
        cell.cart = cart
        cell.unit = 'Bohr'
        cell.exp_to_discard=0.2
        cell.build(
            atom='H 0 0 0; H 0 0 2.6',
            basis='ccecp-ccpvdz',
            ecp='ccecp',
            a=np.diag([6.0, 6.0, 6.0]),
        )

        mf0 = pyscf.pbc.scf.RHF(cell)
        mf0.with_df = pbcdf.MDF(cell).build()
        mf0.run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)
        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)
            if abs(e_nn) < 1e-12:
                e_nn_mol = cell.energy_nuc()
                if abs(e_nn_mol) > 1e-12:
                    e_nn = e_nn_mol
            madelung = 0.0
            try:
                madelung = trexio_lib.read_pbc_madelung(tf)
                madelung = float(np.asarray(madelung).ravel()[0])
            except trexio_lib.Error:
                madelung = 0.0

            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            offset = 0
            G_mo_phys = np.zeros((n, n, n, n))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn - madelung
            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * np.tensordot(G_mo_phys, W_mo_phys, axes=4)

        assert abs(e_mo - mf0.e_tot) < 1e-6


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_crystal_integrals_sym_s1_in_trexio_uhf_ccecp(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'pbc_gamma_rdm_energy_uhf_ccecp.h5')

        cell = pyscf.pbc.gto.Cell()
        cell.cart = cart
        cell.unit = 'Bohr'
        cell.spin = 2
        cell.exp_to_discard=0.2
        cell.build(
            atom='H 0 0 0; H 0 0 2.6',
            basis='ccecp-ccpvdz',
            ecp='ccecp',
            a=np.diag([6.0, 6.0, 6.0]),
        )

        mf0 = pyscf.pbc.scf.UHF(cell)
        mf0.with_df = pbcdf.MDF(cell).build()
        mf0.run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)
        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)
            if abs(e_nn) < 1e-12:
                e_nn_mol = cell.energy_nuc()
                if abs(e_nn_mol) > 1e-12:
                    e_nn = e_nn_mol
            madelung = 0.0
            try:
                madelung = trexio_lib.read_pbc_madelung(tf)
                madelung = float(np.asarray(madelung).ravel()[0])
            except trexio_lib.Error:
                madelung = 0.0

            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            spin_rdm = (
                trexio_lib.has_rdm_1e_up(tf)
                and trexio_lib.has_rdm_2e_upup(tf)
                and trexio_lib.has_rdm_2e_dndn(tf)
                and trexio_lib.has_rdm_2e_updn(tf)
            )

            assert spin_rdm

            rdm2_size = trexio_lib.read_rdm_2e_upup_size(tf)
            n_up = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_up ** 4

            rdm2_size = trexio_lib.read_rdm_2e_dndn_size(tf)
            n_dn = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_dn ** 4

            rdm2_size = trexio_lib.read_rdm_2e_updn_size(tf)
            assert rdm2_size == (n_up * n_dn) ** 2

            offset = 0
            G_uu = np.zeros((n_up, n_up, n_up, n_up))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_upup(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_uu[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_dd = np.zeros((n_dn, n_dn, n_dn, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_dndn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_dd[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_ud = np.zeros((n_up, n_dn, n_up, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_updn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_ud[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn - madelung
            off = n_up

            W_uu = W_mo_phys[:n_up, :n_up, :n_up, :n_up]
            W_dd = W_mo_phys[off:off + n_dn, off:off + n_dn, off:off + n_dn, off:off + n_dn]
            W_ud = W_mo_phys[:n_up, off:off + n_dn, :n_up, off:off + n_dn]

            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * (
                np.tensordot(G_uu, W_uu, axes=4)
                + np.tensordot(G_dd, W_dd, axes=4)
                + 2.0 * np.tensordot(G_ud, W_ud, axes=4)
            )

        assert abs(e_mo - mf0.e_tot) < 1e-6


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_crystal_integrals_sym_s4_in_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'pbc_gamma_rdm_energy_s4.h5')

        cell = pyscf.pbc.gto.Cell()
        cell.cart = cart
        cell.unit = 'Bohr'
        cell.build(
            atom='H 0 0 0; H 0 0 1.4',
            basis='sto-3g',
            a=np.diag([3.0, 3.0, 5.0]),
        )

        mf0 = pyscf.pbc.scf.RHF(cell)
        mf0.with_df = pbcdf.MDF(cell).build()
        mf0.run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)
        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s4')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO', sym='s4')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)
            if abs(e_nn) < 1e-12:
                e_nn_mol = cell.energy_nuc()
                if abs(e_nn_mol) > 1e-12:
                    e_nn = e_nn_mol
            madelung = 0.0
            try:
                madelung = trexio_lib.read_pbc_madelung(tf)
                madelung = float(np.asarray(madelung).ravel()[0])
            except trexio_lib.Error:
                madelung = 0.0

            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            mo_eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, mo_eri_size)
            assert n_read == mo_eri_size
            W_mo_phys = _expand_trexio_eri(idx, val, n)

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            offset = 0
            G_mo_phys = np.zeros((n, n, n, n))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn - madelung
            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * np.tensordot(G_mo_phys, W_mo_phys, axes=4)

        assert abs(e_mo - mf0.e_tot) < 1e-6


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_crystal_integrals_sym_s4_in_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'pbc_gamma_rdm_energy_uhf_s4.h5')

        cell = pyscf.pbc.gto.Cell()
        cell.cart = cart
        cell.unit = 'Bohr'
        cell.spin = 2
        cell.build(
            atom='H 0 0 0; H 0 0 1.4',
            basis='sto-3g',
            a=np.diag([3.0, 3.0, 5.0]),
        )

        mf0 = pyscf.pbc.scf.UHF(cell)
        mf0.with_df = pbcdf.MDF(cell).build()
        mf0.run()
        assert mf0.converged

        trexio.to_trexio(mf0, filename)
        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s4')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO', sym='s4')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)
            if abs(e_nn) < 1e-12:
                e_nn_mol = cell.energy_nuc()
                if abs(e_nn_mol) > 1e-12:
                    e_nn = e_nn_mol
            madelung = 0.0
            try:
                madelung = trexio_lib.read_pbc_madelung(tf)
                madelung = float(np.asarray(madelung).ravel()[0])
            except trexio_lib.Error:
                madelung = 0.0

            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            mo_eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, mo_eri_size)
            assert n_read == mo_eri_size
            W_mo_phys = _expand_trexio_eri(idx, val, n)

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            spin_rdm = (
                trexio_lib.has_rdm_1e_up(tf)
                and trexio_lib.has_rdm_2e_upup(tf)
                and trexio_lib.has_rdm_2e_dndn(tf)
                and trexio_lib.has_rdm_2e_updn(tf)
            )

            assert spin_rdm

            rdm2_size = trexio_lib.read_rdm_2e_upup_size(tf)
            n_up = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_up ** 4

            rdm2_size = trexio_lib.read_rdm_2e_dndn_size(tf)
            n_dn = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_dn ** 4

            rdm2_size = trexio_lib.read_rdm_2e_updn_size(tf)
            assert rdm2_size == (n_up * n_dn) ** 2

            offset = 0
            G_uu = np.zeros((n_up, n_up, n_up, n_up))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_upup(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_uu[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_dd = np.zeros((n_dn, n_dn, n_dn, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_dndn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_dd[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_ud = np.zeros((n_up, n_dn, n_up, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_updn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_ud[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn - madelung
            off = n_up

            W_uu = W_mo_phys[:n_up, :n_up, :n_up, :n_up]
            W_dd = W_mo_phys[off:off + n_dn, off:off + n_dn, off:off + n_dn, off:off + n_dn]
            W_ud = W_mo_phys[:n_up, off:off + n_dn, :n_up, off:off + n_dn]

            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * (
                np.tensordot(G_uu, W_uu, axes=4)
                + np.tensordot(G_dd, W_dd, axes=4)
                + 2.0 * np.tensordot(G_ud, W_ud, axes=4)
            )

        assert abs(e_mo - mf0.e_tot) < 1e-6


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s1_in_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy_uks.h5')

        mol0 = pyscf.M(atom='O 0 0 0', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        assert mf0.converged

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            potential += _hermitize(mol0.intor('ECPscalar'))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            nao = trexio_lib.read_ao_num(tf)
            assert size == nao ** 4

        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction (UKS)
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()
            dm_a = dm_b = None
            if isinstance(dm_ao, (tuple, list)) and len(dm_ao) == 2:
                dm_a, dm_b = dm_ao
                dm_tot = dm_a + dm_b
            elif isinstance(dm_ao, np.ndarray) and dm_ao.ndim == 3 and dm_ao.shape[0] == 2:
                dm_a, dm_b = dm_ao[0], dm_ao[1]
                dm_tot = dm_a + dm_b
            else:
                dm_tot = dm_ao

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            assert ao_eri_size == nao ** 4
            offset = 0
            W_ao_phys = np.zeros((nao, nao, nao, nao))  # raw order as stored
            while offset < ao_eri_size:
                bufsize = min(BUFSIZE, ao_eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_ao_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    p, r, q, s = np.asarray(buf_idx, dtype=np.int64).T  # stored as (p,r,q,s)
                    W_ao_phys[p, r, q, s] = buf_val
                if feof:
                    break

            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_tot)
            e_ao = e_nn
            if dm_a is not None and dm_b is not None:
                K_a = np.einsum('prqs,rs->pq', W_ao_chem, dm_a)
                K_b = np.einsum('prqs,rs->pq', W_ao_chem, dm_b)

                e_ao += np.einsum('pq,pq->', dm_a, core_ao)
                e_ao += np.einsum('pq,pq->', dm_b, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.5 * (
                    np.einsum('pq,pq->', dm_a, K_a)
                    + np.einsum('pq,pq->', dm_b, K_b)
                )
            else:
                K = np.einsum('prqs,rs->pq', W_ao_chem, dm_tot)

                e_ao += np.einsum('pq,pq->', dm_tot, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.25 * np.einsum('pq,pq->', dm_tot, K)

            # MO energy reconstruction
            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break
            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            spin_rdm = (
                trexio_lib.has_rdm_1e_up(tf)
                and trexio_lib.has_rdm_2e_upup(tf)
                and trexio_lib.has_rdm_2e_dndn(tf)
                and trexio_lib.has_rdm_2e_updn(tf)
            )

            assert spin_rdm

            rdm2_size = trexio_lib.read_rdm_2e_upup_size(tf)
            n_up = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_up ** 4

            rdm2_size = trexio_lib.read_rdm_2e_dndn_size(tf)
            n_dn = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_dn ** 4

            rdm2_size = trexio_lib.read_rdm_2e_updn_size(tf)
            assert rdm2_size == (n_up * n_dn) ** 2

            offset = 0
            G_uu = np.zeros((n_up, n_up, n_up, n_up))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_upup(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_uu[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_dd = np.zeros((n_dn, n_dn, n_dn, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_dndn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_dd[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_ud = np.zeros((n_up, n_dn, n_up, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_updn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_ud[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn
            off = n_up

            W_uu = W_mo_phys[:n_up, :n_up, :n_up, :n_up]
            W_dd = W_mo_phys[off:off + n_dn, off:off + n_dn, off:off + n_dn, off:off + n_dn]
            W_ud = W_mo_phys[:n_up, off:off + n_dn, :n_up, off:off + n_dn]

            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * (
                np.tensordot(G_uu, W_uu, axes=4)
                + np.tensordot(G_dd, W_dd, axes=4)
                + 2.0 * np.tensordot(G_ud, W_ud, axes=4)
            )

        assert abs(e_ao - mf0.e_tot) < 1e-8
        assert abs(e_mo - mf0.e_tot) < 1e-8


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s1_in_trexio_rhf_ecp(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy_ecp.h5')

        mol0 = pyscf.M(
            atom='H 0 0 0; F 0 0 1',
            basis='ccecp-ccpvdz',
            ecp='ccecp',
            cart=cart,
        )
        mf0 = mol0.RHF().run()

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            potential += _hermitize(mol0.intor('ECPscalar'))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, size)
            assert n_read == size

        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            assert ao_eri_size == nao ** 4
            offset = 0
            W_ao_phys = np.zeros((nao, nao, nao, nao))  # raw order as stored
            while offset < ao_eri_size:
                bufsize = min(BUFSIZE, ao_eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_ao_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    p, r, q, s = np.asarray(buf_idx, dtype=np.int64).T  # stored as (p,r,q,s)
                    W_ao_phys[p, r, q, s] = buf_val
                if feof:
                    break

            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_ao)
            K = np.einsum('prqs,rs->pq', W_ao_chem, dm_ao)
            vhf = J - 0.5 * K
            e_ao = e_nn
            e_ao += np.einsum('pq,pq->', dm_ao, core_ao)
            e_ao += 0.5 * np.einsum('pq,pq->', dm_ao, vhf)

            # MO energy reconstruction
            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break
            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            offset = 0
            G_mo_phys = np.zeros((n, n, n, n))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn
            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * np.tensordot(G_mo_phys, W_mo_phys, axes=4)

        assert abs(e_ao - mf0.e_tot) < 1e-8
        assert abs(e_mo - mf0.e_tot) < 1e-8


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s1_in_trexio_uhf_ecp(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy_uks_ecp.h5')

        mol0 = pyscf.M(
            atom='H 0 0 0; F 0 0 1',
            basis='ccecp-ccpvdz',
            ecp='ccecp',
            spin=2,
            cart=cart,
        )
        mf0 = mol0.UHF().run()
        assert mf0.converged

        overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            potential += _hermitize(mol0.intor('ECPscalar'))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_overlap(tf), overlap, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_kinetic(tf), kinetic, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_potential_n_e(tf), potential, atol=DIFF_TOL)
            np.testing.assert_allclose(trexio_lib.read_ao_1e_int_core_hamiltonian(tf), core, atol=DIFF_TOL)

        trexio.write_2e_eri(mf0, filename, basis='AO')
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            assert trexio_lib.has_ao_2e_int_eri(tf)
            size = trexio_lib.read_ao_2e_int_eri_size(tf)
            nao = trexio_lib.read_ao_num(tf)
            assert size == nao ** 4

        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction (UKS)
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()
            dm_a = dm_b = None
            if isinstance(dm_ao, (tuple, list)) and len(dm_ao) == 2:
                dm_a, dm_b = dm_ao
                dm_tot = dm_a + dm_b
            elif isinstance(dm_ao, np.ndarray) and dm_ao.ndim == 3 and dm_ao.shape[0] == 2:
                dm_a, dm_b = dm_ao[0], dm_ao[1]
                dm_tot = dm_a + dm_b
            else:
                dm_tot = dm_ao

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            assert ao_eri_size == nao ** 4
            offset = 0
            W_ao_phys = np.zeros((nao, nao, nao, nao))  # raw order as stored
            while offset < ao_eri_size:
                bufsize = min(BUFSIZE, ao_eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_ao_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    p, r, q, s = np.asarray(buf_idx, dtype=np.int64).T  # stored as (p,r,q,s)
                    W_ao_phys[p, r, q, s] = buf_val
                if feof:
                    break

            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_tot)
            e_ao = e_nn
            if dm_a is not None and dm_b is not None:
                K_a = np.einsum('prqs,rs->pq', W_ao_chem, dm_a)
                K_b = np.einsum('prqs,rs->pq', W_ao_chem, dm_b)

                e_ao += np.einsum('pq,pq->', dm_a, core_ao)
                e_ao += np.einsum('pq,pq->', dm_b, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.5 * (
                    np.einsum('pq,pq->', dm_a, K_a)
                    + np.einsum('pq,pq->', dm_b, K_b)
                )
            else:
                K = np.einsum('prqs,rs->pq', W_ao_chem, dm_tot)

                e_ao += np.einsum('pq,pq->', dm_tot, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.25 * np.einsum('pq,pq->', dm_tot, K)

            # MO energy reconstruction
            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            assert eri_size == n ** 4
            offset = 0
            W_mo_phys = np.zeros((n, n, n, n))
            while offset < eri_size:
                bufsize = min(BUFSIZE, eri_size - offset)
                buf_idx, buf_val, icount, feof = trexio_lib.read_mo_2e_int_eri(tf, offset, bufsize)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    W_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break
            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            spin_rdm = (
                trexio_lib.has_rdm_1e_up(tf)
                and trexio_lib.has_rdm_2e_upup(tf)
                and trexio_lib.has_rdm_2e_dndn(tf)
                and trexio_lib.has_rdm_2e_updn(tf)
            )

            assert spin_rdm

            rdm2_size = trexio_lib.read_rdm_2e_upup_size(tf)
            n_up = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_up ** 4

            rdm2_size = trexio_lib.read_rdm_2e_dndn_size(tf)
            n_dn = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_dn ** 4

            rdm2_size = trexio_lib.read_rdm_2e_updn_size(tf)
            assert rdm2_size == (n_up * n_dn) ** 2

            offset = 0
            G_uu = np.zeros((n_up, n_up, n_up, n_up))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_upup(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_uu[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_dd = np.zeros((n_dn, n_dn, n_dn, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_dndn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_dd[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_ud = np.zeros((n_up, n_dn, n_up, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_updn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_ud[i, j, k, l] = buf_val
                if feof:
                    break

            e_mo = e_nn
            off = n_up

            W_uu = W_mo_phys[:n_up, :n_up, :n_up, :n_up]
            W_dd = W_mo_phys[off:off + n_dn, off:off + n_dn, off:off + n_dn, off:off + n_dn]
            W_ud = W_mo_phys[:n_up, off:off + n_dn, :n_up, off:off + n_dn]

            e_mo += np.einsum('pq,pq->', dens, core)
            e_mo += 0.5 * (
                np.tensordot(G_uu, W_uu, axes=4)
                + np.tensordot(G_dd, W_dd, axes=4)
                + 2.0 * np.tensordot(G_ud, W_ud, axes=4)
            )

        assert abs(e_ao - mf0.e_tot) < 1e-8
        assert abs(e_mo - mf0.e_tot) < 1e-8


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s4_in_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy_s4.h5')

        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()

        #overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            from pyscf.gto import ecp
            potential += _hermitize(ecp.ecp_int(mol0))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s4')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO', sym='s4')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, ao_eri_size)
            assert n_read == ao_eri_size
            W_ao_phys = _expand_trexio_eri(idx, val, nao)

            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_ao)
            K = np.einsum('prqs,rs->pq', W_ao_chem, dm_ao)
            vhf = J - 0.5 * K
            e_ao = e_nn
            e_ao += np.einsum('pq,pq->', dm_ao, core_ao)
            e_ao += 0.5 * np.einsum('pq,pq->', dm_ao, vhf)

            # MO energy reconstruction
            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            mo_eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, mo_eri_size)
            assert n_read == mo_eri_size
            W_mo_phys = _expand_trexio_eri(idx, val, n)

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            offset = 0
            G_mo_phys = np.zeros((n, n, n, n))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_mo_phys[i, j, k, l] = buf_val
                if feof:
                    break

        e_mo = e_nn
        e_mo += np.einsum('pq,pq->', dens, core)
        e_mo += 0.5 * np.tensordot(G_mo_phys, W_mo_phys, axes=4)

        assert abs(e_ao - mf0.e_tot) < 1e-8
        assert abs(e_mo - mf0.e_tot) < 1e-8


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s4_in_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy_uks_s4.h5')

        mol0 = pyscf.M(atom='O 0 0 0', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        assert mf0.converged

        #overlap = _hermitize(mf0.get_ovlp())
        kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            from pyscf.gto import ecp
            potential += _hermitize(ecp.ecp_int(mol0))
        core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s4')
        trexio.write_1e_eri(mf0, filename, basis='MO')
        trexio.write_2e_eri(mf0, filename, basis='MO', sym='s4')
        trexio.write_1b_rdm(mf0, filename)
        trexio.write_2b_rdm(mf0, filename)

        BUFSIZE = 100000
        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction (UKS)
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()
            dm_a = dm_b = None
            if isinstance(dm_ao, (tuple, list)) and len(dm_ao) == 2:
                dm_a, dm_b = dm_ao
                dm_tot = dm_a + dm_b
            elif isinstance(dm_ao, np.ndarray) and dm_ao.ndim == 3 and dm_ao.shape[0] == 2:
                dm_a, dm_b = dm_ao[0], dm_ao[1]
                dm_tot = dm_a + dm_b
            else:
                dm_tot = dm_ao

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, ao_eri_size)
            assert n_read == ao_eri_size
            W_ao_phys = _expand_trexio_eri(idx, val, nao)
            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_tot)
            e_ao = e_nn
            if dm_a is not None and dm_b is not None:
                K_a = np.einsum('prqs,rs->pq', W_ao_chem, dm_a)
                K_b = np.einsum('prqs,rs->pq', W_ao_chem, dm_b)

                e_ao += np.einsum('pq,pq->', dm_a, core_ao)
                e_ao += np.einsum('pq,pq->', dm_b, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.5 * (
                    np.einsum('pq,pq->', dm_a, K_a)
                    + np.einsum('pq,pq->', dm_b, K_b)
                )
            else:
                K = np.einsum('prqs,rs->pq', W_ao_chem, dm_tot)

                e_ao += np.einsum('pq,pq->', dm_tot, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.25 * np.einsum('pq,pq->', dm_tot, K)

            # MO energy reconstruction
            n = trexio_lib.read_mo_num(tf)
            core = trexio_lib.read_mo_1e_int_core_hamiltonian(tf)
            core = np.asarray(core)
            if core.ndim == 1:
                core = core.reshape(n, n)

            mo_eri_size = trexio_lib.read_mo_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_mo_2e_int_eri(tf, 0, mo_eri_size)
            assert n_read == mo_eri_size
            W_mo_phys = _expand_trexio_eri(idx, val, n)

            dens = trexio_lib.read_rdm_1e(tf, doReshape=True)
            dens = np.asarray(dens)
            if dens.ndim == 1:
                dens = dens.reshape(n, n)

            spin_rdm = (
                trexio_lib.has_rdm_1e_up(tf)
                and trexio_lib.has_rdm_2e_upup(tf)
                and trexio_lib.has_rdm_2e_dndn(tf)
                and trexio_lib.has_rdm_2e_updn(tf)
            )

            assert spin_rdm

            rdm2_size = trexio_lib.read_rdm_2e_upup_size(tf)
            n_up = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_up ** 4

            rdm2_size = trexio_lib.read_rdm_2e_dndn_size(tf)
            n_dn = int(round(rdm2_size ** 0.25))
            assert rdm2_size == n_dn ** 4

            rdm2_size = trexio_lib.read_rdm_2e_updn_size(tf)
            assert rdm2_size == (n_up * n_dn) ** 2

            offset = 0
            G_uu = np.zeros((n_up, n_up, n_up, n_up))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_upup(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_uu[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_dd = np.zeros((n_dn, n_dn, n_dn, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_dndn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_dd[i, j, k, l] = buf_val
                if feof:
                    break

            offset = 0
            G_ud = np.zeros((n_up, n_dn, n_up, n_dn))
            while True:
                buf_idx, buf_val, icount, feof = trexio_lib.read_rdm_2e_updn(tf, offset, BUFSIZE)
                offset += icount
                if icount:
                    i, j, k, l = np.asarray(buf_idx, dtype=np.int64).T
                    G_ud[i, j, k, l] = buf_val
                if feof:
                    break

        e_mo = e_nn
        off = n_up

        W_uu = W_mo_phys[:n_up, :n_up, :n_up, :n_up]
        W_dd = W_mo_phys[off:off + n_dn, off:off + n_dn, off:off + n_dn, off:off + n_dn]
        W_ud = W_mo_phys[:n_up, off:off + n_dn, :n_up, off:off + n_dn]

        e_mo += np.einsum('pq,pq->', dens, core)
        e_mo += 0.5 * (
            np.tensordot(G_uu, W_uu, axes=4)
            + np.tensordot(G_dd, W_dd, axes=4)
            + 2.0 * np.tensordot(G_ud, W_ud, axes=4)
        )

        assert abs(e_ao - mf0.e_tot) < 1e-8
        assert abs(e_mo - mf0.e_tot) < 1e-8


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s8_in_trexio_rhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy_s8.h5')

        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g*', cart=cart)
        mf0 = mol0.RHF().run()

        #overlap = _hermitize(mf0.get_ovlp())
        #kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            from pyscf.gto import ecp
            potential += _hermitize(ecp.ecp_int(mol0))
        #core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s8')

        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, ao_eri_size)
            assert n_read == ao_eri_size
            W_ao_phys = _expand_trexio_eri(idx, val, nao)

            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_ao)
            K = np.einsum('prqs,rs->pq', W_ao_chem, dm_ao)
            vhf = J - 0.5 * K
            e_ao = e_nn
            e_ao += np.einsum('pq,pq->', dm_ao, core_ao)
            e_ao += 0.5 * np.einsum('pq,pq->', dm_ao, vhf)

        assert abs(e_ao - mf0.e_tot) < 1e-8


@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_energy_molecule_integrals_sym_s8_in_trexio_uhf_ae(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'mol_integrals_energy_uks_s8.h5')

        mol0 = pyscf.M(atom='O 0 0 0', basis='6-31g*', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        assert mf0.converged

        #overlap = _hermitize(mf0.get_ovlp())
        #kinetic = _hermitize(mol0.intor('int1e_kin'))
        potential = _hermitize(mol0.intor('int1e_nuc'))
        if mol0._ecp:
            from pyscf.gto import ecp
            potential += _hermitize(ecp.ecp_int(mol0))
        #core = kinetic + potential

        trexio.to_trexio(mf0, filename)

        trexio.write_1e_eri(mf0, filename, basis='AO')
        trexio.write_2e_eri(mf0, filename, basis='AO', sym='s8')

        with trexio_lib.File(filename, 'r', back_end=trexio_lib.TREXIO_AUTO) as tf:
            e_nn = trexio_lib.read_nucleus_repulsion(tf)

            # AO energy reconstruction (UKS)
            nao = trexio_lib.read_ao_num(tf)
            core_ao = trexio_lib.read_ao_1e_int_core_hamiltonian(tf)
            core_ao = np.asarray(core_ao)
            if core_ao.ndim == 1:
                core_ao = core_ao.reshape(nao, nao)

            dm_ao = mf0.make_rdm1()
            dm_a = dm_b = None
            if isinstance(dm_ao, (tuple, list)) and len(dm_ao) == 2:
                dm_a, dm_b = dm_ao
                dm_tot = dm_a + dm_b
            elif isinstance(dm_ao, np.ndarray) and dm_ao.ndim == 3 and dm_ao.shape[0] == 2:
                dm_a, dm_b = dm_ao[0], dm_ao[1]
                dm_tot = dm_a + dm_b
            else:
                dm_tot = dm_ao

            ao_eri_size = trexio_lib.read_ao_2e_int_eri_size(tf)
            idx, val, n_read, _ = trexio_lib.read_ao_2e_int_eri(tf, 0, ao_eri_size)
            assert n_read == ao_eri_size
            W_ao_phys = _expand_trexio_eri(idx, val, nao)
            W_ao_chem = W_ao_phys.transpose(0, 2, 1, 3)

            J = np.einsum('pqrs,rs->pq', W_ao_chem, dm_tot)
            e_ao = e_nn
            if dm_a is not None and dm_b is not None:
                K_a = np.einsum('prqs,rs->pq', W_ao_chem, dm_a)
                K_b = np.einsum('prqs,rs->pq', W_ao_chem, dm_b)

                e_ao += np.einsum('pq,pq->', dm_a, core_ao)
                e_ao += np.einsum('pq,pq->', dm_b, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.5 * (
                    np.einsum('pq,pq->', dm_a, K_a)
                    + np.einsum('pq,pq->', dm_b, K_b)
                )
            else:
                K = np.einsum('prqs,rs->pq', W_ao_chem, dm_tot)

                e_ao += np.einsum('pq,pq->', dm_tot, core_ao)
                e_ao += 0.5 * np.einsum('pq,pq->', dm_tot, J)
                e_ao -= 0.25 * np.einsum('pq,pq->', dm_tot, K)

        assert abs(e_ao - mf0.e_tot) < 1e-8
