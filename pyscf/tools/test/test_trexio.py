import pyscf
from pyscf import df
from pyscf.tools import trexio
import trexio as trexio_lib
import os
import numpy as np
import tempfile
import pytest

DIFF_TOL = 1e-10

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
def test_mol_rhf_ccecp_ccpvqz(cart):
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
