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
# reading/writing `mcscf` from/to trexio file
#################################################################

## molecule, segment contraction (6-31g), all-electron, RHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mcscf_rhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g', cart=cart)
        mf0 = mol0.RHF().run()
        mc0 = pyscf.mcscf.CASCI(mf0, 2, 2)
        mc0.kernel()
        trexio.to_trexio(mc0, filename)

        rdm1 = mc0.fcisolver.make_rdm1(mc0.ci, mc0.ncas, mc0.nelecas)
        natural_occ = np.linalg.eigvalsh(rdm1)[::-1]
        expected_occ = np.zeros(mc0.mo_energy.size)
        expected_occ[:mc0.ncore] = 2.0
        expected_occ[mc0.ncore:mc0.ncore + mc0.ncas] = natural_occ

        with trexio.trexio.File(filename, 'r', back_end=trexio.trexio.TREXIO_AUTO) as tf:
            occ1 = trexio.trexio.read_mo_occupation(tf)
            det_num = trexio.trexio.read_determinant_num(tf)
            det_list = trexio.trexio.read_determinant_list(tf, 0, det_num)
            int64_num = trexio.trexio.get_int64_num(tf)
            rdm1_read = trexio.trexio.read_rdm_1e(tf)
            h1_read = trexio.trexio.read_mo_1e_int_core_hamiltonian(tf)
            idx_rdm2, data_rdm2, nread_rdm2, _ = trexio.trexio.read_rdm_2e(tf, 0, mc0.ncas**4)
            idx_h2, data_h2, nread_h2, _ = trexio.trexio.read_mo_2e_int_eri(tf, 0, mc0.ncas**4)
        assert abs(np.asarray(occ1) - expected_occ).max() < DIFF_TOL

        calc_int64_num = (mc0.ncore + mc0.ncas + 63) // 64
        if int64_num <= 0:
            int64_num = calc_int64_num
        else:
            int64_num = max(int64_num, calc_int64_num)

        def _normalize_det_list(det_list_in):
            if isinstance(det_list_in, (tuple, list)):
                if len(det_list_in) > 0 and isinstance(det_list_in[0], np.ndarray):
                    if all(np.isscalar(x) for x in det_list_in[1:]):
                        det_list_in = det_list_in[0]
            if isinstance(det_list_in, np.ndarray):
                if det_list_in.ndim == 2:
                    return [np.asarray(row, dtype=np.int64) for row in det_list_in]
                if det_list_in.ndim == 1:
                    return [np.asarray(det_list_in, dtype=np.int64)]
            return [np.asarray(row, dtype=np.int64) for row in det_list_in]

        occsa, occsb, _, _ = trexio._get_occsa_and_occsb(mc0, mc0.ncas, mc0.nelecas, 0.0)
        expected_det_list = []
        for a, b in zip(occsa, occsb):
            occsa_upshifted = [orb for orb in range(mc0.ncore)] + [orb + mc0.ncore for orb in a]
            occsb_upshifted = [orb for orb in range(mc0.ncore)] + [orb + mc0.ncore for orb in b]
            det_a = np.asarray(trexio.trexio.to_bitfield_list(int64_num, occsa_upshifted), dtype=np.int64)
            det_b = np.asarray(trexio.trexio.to_bitfield_list(int64_num, occsb_upshifted), dtype=np.int64)
            expected_det_list.append(np.hstack([det_a, det_b]).astype(np.int64, copy=False))

        det_list_norm = _normalize_det_list(det_list)
        expected_det_list_norm = _normalize_det_list(expected_det_list)
        assert len(det_list_norm) == len(expected_det_list_norm)
        for det_row, expected_row in zip(det_list_norm, expected_det_list_norm):
            assert np.array_equal(det_row, expected_row)


        dm1_cas, dm2_cas = trexio._get_cas_rdm12(mc0, mc0.ncas)
        h1eff, _ = trexio._get_cas_h1eff(mc0)
        h2eff = trexio._get_cas_h2eff(mc0, mc0.ncas)

        active = slice(mc0.ncore, mc0.ncore + mc0.ncas)
        assert abs(rdm1_read[active, active] - dm1_cas).max() < DIFF_TOL
        assert abs(h1_read[active, active] - h1eff).max() < DIFF_TOL

        rdm2_read = np.zeros((mc0.ncas, mc0.ncas, mc0.ncas, mc0.ncas))
        idx_rdm2 = np.asarray(idx_rdm2[:nread_rdm2], dtype=int)
        # Invert TREXIO k,l,i,j ordering back to PySCF (i,j,k,l)
        idx_rdm2 = idx_rdm2[:, [2, 3, 0, 1]] - mc0.ncore
        data_rdm2 = np.asarray(data_rdm2[:nread_rdm2])
        rdm2_read[idx_rdm2[:, 0], idx_rdm2[:, 1], idx_rdm2[:, 2], idx_rdm2[:, 3]] = data_rdm2
        assert abs(rdm2_read - dm2_cas).max() < DIFF_TOL

        h2_read = np.zeros((mc0.ncas, mc0.ncas, mc0.ncas, mc0.ncas))
        idx_h2 = np.asarray(idx_h2[:nread_h2], dtype=int)
        idx_h2 = idx_h2[:, [2, 3, 0, 1]] - mc0.ncore
        data_h2 = np.asarray(data_h2[:nread_h2])
        h2_read[idx_h2[:, 0], idx_h2[:, 1], idx_h2[:, 2], idx_h2[:, 3]] = data_h2
        assert abs(h2_read - h2eff).max() < DIFF_TOL

        dm1_cas, dm2_cas = trexio._get_cas_rdm12(mc0, mc0.ncas)
        h1eff, ecore = trexio._get_cas_h1eff(mc0)
        h2eff = trexio._get_cas_h2eff(mc0, mc0.ncas)
        e_act = np.einsum("ij,ij->", h1eff, dm1_cas) + 0.5 * np.einsum("ijkl,ijkl->", h2eff, dm2_cas)
        e_reconstructed = ecore + e_act
        assert abs(e_reconstructed - mc0.e_tot) < 1e-8

## molecule, segment contraction (6-31g), all-electron, UHF
@pytest.mark.parametrize("cart", [False, True], ids=["cart=false", "cart=true"])
def test_mcscf_uhf_ae_6_31g(cart):
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, 'test.h5')
        mol0 = pyscf.M(atom='H 0 0 0; F 0 0 1', basis='6-31g', spin=2, cart=cart)
        mf0 = mol0.UHF().run()
        mc0 = pyscf.mcscf.CASCI(mf0, 2, 2)
        mc0.kernel()
        trexio.to_trexio(mc0, filename)

        ncas = mc0.ncas
        neleca, nelecb = mc0.nelecas
        if hasattr(mc0.fcisolver, 'make_rdm1s'):
            rdm1a, rdm1b = mc0.fcisolver.make_rdm1s(mc0.ci, ncas, mc0.nelecas)
        else:
            rdm1 = mc0.fcisolver.make_rdm1(mc0.ci, ncas, mc0.nelecas)
            total = neleca + nelecb
            if total > 0:
                rdm1a = rdm1 * (neleca / total)
                rdm1b = rdm1 * (nelecb / total)
            else:
                rdm1a = rdm1 * 0.0
                rdm1b = rdm1 * 0.0

        nat_occ_a = np.linalg.eigvalsh(rdm1a)[::-1]
        nat_occ_b = np.linalg.eigvalsh(rdm1b)[::-1]

        def _get_uhf_mo_pair(mo_coeff):
            if isinstance(mo_coeff, (tuple, list)) and len(mo_coeff) == 2:
                return mo_coeff[0], mo_coeff[1]
            if isinstance(mo_coeff, np.ndarray):
                if mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
                    return mo_coeff[0], mo_coeff[1]
                if mo_coeff.ndim == 2:
                    return mo_coeff, mo_coeff
            return None

        mo_pair = _get_uhf_mo_pair(mc0.mo_coeff)
        if mo_pair is None:
            mo_pair = _get_uhf_mo_pair(mc0._scf.mo_coeff)
        if mo_pair is None:
            mo_pair = _get_uhf_mo_pair(mf0.mo_coeff)
        if mo_pair is None:
            raise ValueError("Unable to determine UHF mo_coeff pair for expected occupations.")

        mo_up, mo_dn = mo_pair
        num_mo_up = mo_up.shape[1]
        num_mo_dn = mo_dn.shape[1]

        with trexio.trexio.File(filename, 'r', back_end=trexio.trexio.TREXIO_AUTO) as tf:
            occ1 = trexio.trexio.read_mo_occupation(tf)
            det_num = trexio.trexio.read_determinant_num(tf)
            det_list = trexio.trexio.read_determinant_list(tf, 0, det_num)
            int64_num = trexio.trexio.get_int64_num(tf)

        occ_alpha = np.zeros(num_mo_up)
        occ_beta = np.zeros(num_mo_dn)
        occ_alpha[:mc0.ncore] = 1.0
        occ_beta[:mc0.ncore] = 1.0
        occ_alpha[mc0.ncore:mc0.ncore + ncas] = nat_occ_a
        occ_beta[mc0.ncore:mc0.ncore + ncas] = nat_occ_b

        occ1_arr = np.asarray(occ1)
        rdm1_total = rdm1a + rdm1b
        nat_occ_total = np.linalg.eigvalsh(rdm1_total)[::-1]
        occ_total = np.zeros(num_mo_up)
        occ_total[:mc0.ncore] = 2.0
        occ_total[mc0.ncore:mc0.ncore + ncas] = nat_occ_total

        is_uhf = isinstance(mc0._scf, (
            pyscf.scf.uhf.UHF,
            pyscf.dft.uks.UKS,
            pyscf.pbc.scf.kuhf.KUHF,
            pyscf.pbc.dft.kuks.KUKS,
        ))

        if is_uhf:
            expected_occ = np.concatenate([occ_alpha, occ_beta])
            assert occ1_arr.size == num_mo_up + num_mo_dn
        else:
            expected_occ = occ_total
            assert occ1_arr.size == num_mo_up

        assert abs(occ1_arr - expected_occ).max() < DIFF_TOL

        calc_int64_num = (mc0.ncore + mc0.ncas + 63) // 64
        if int64_num <= 0:
            int64_num = calc_int64_num
        else:
            int64_num = max(int64_num, calc_int64_num)

        def _normalize_det_list(det_list_in):
            if isinstance(det_list_in, (tuple, list)):
                if len(det_list_in) > 0 and isinstance(det_list_in[0], np.ndarray):
                    if all(np.isscalar(x) for x in det_list_in[1:]):
                        det_list_in = det_list_in[0]
            if isinstance(det_list_in, np.ndarray):
                if det_list_in.ndim == 2:
                    return [np.asarray(row, dtype=np.int64) for row in det_list_in]
                if det_list_in.ndim == 1:
                    return [np.asarray(det_list_in, dtype=np.int64)]
            return [np.asarray(row, dtype=np.int64) for row in det_list_in]

        occsa, occsb, _, _ = trexio._get_occsa_and_occsb(mc0, mc0.ncas, mc0.nelecas, 0.0)
        expected_det_list = []
        for a, b in zip(occsa, occsb):
            occsa_upshifted = [orb for orb in range(mc0.ncore)] + [orb + mc0.ncore for orb in a]
            occsb_upshifted = [orb for orb in range(mc0.ncore)] + [orb + mc0.ncore for orb in b]
            det_a = np.asarray(trexio.trexio.to_bitfield_list(int64_num, occsa_upshifted), dtype=np.int64)
            det_b = np.asarray(trexio.trexio.to_bitfield_list(int64_num, occsb_upshifted), dtype=np.int64)
            expected_det_list.append(np.hstack([det_a, det_b]).astype(np.int64, copy=False))

        det_list_norm = _normalize_det_list(det_list)
        expected_det_list_norm = _normalize_det_list(expected_det_list)
        assert len(det_list_norm) == len(expected_det_list_norm)
        for det_row, expected_row in zip(det_list_norm, expected_det_list_norm):
            assert np.array_equal(det_row, expected_row)

        dm1_cas, dm2_cas = trexio._get_cas_rdm12(mc0, mc0.ncas)
        h1eff, ecore = trexio._get_cas_h1eff(mc0)
        h2eff = trexio._get_cas_h2eff(mc0, mc0.ncas)
        e_act = np.einsum("ij,ij->", h1eff, dm1_cas) + 0.5 * np.einsum("ijkl,ijkl->", h2eff, dm2_cas)
        e_reconstructed = ecore + e_act
        assert abs(e_reconstructed - mc0.e_tot) < 1e-8

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
