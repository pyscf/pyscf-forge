#!/usr/bin/env python

import unittest

import numpy as np

from pyscf import dft
from pyscf import gto
from pyscf import sftda
from pyscf.sftda.uhf_sf import get_ab_sf


HF_REF_ETF = np.array([
    [-3.95470206e-03, -3.73615571e-04, -9.69878457e-06],
    [ 2.30042868e-03,  1.82489975e-04,  3.11897170e-06],
    [-3.38391814e-04,  2.19689308e-02, -7.70962711e-04],
    [ 1.99266520e-03, -2.17778053e-02,  7.77542524e-04],
])

HF_REF_FULL = np.array([
    [ 4.08124376e-03,  3.85893263e-04,  8.26819990e-06],
    [-2.12115474e-03, -1.73151723e-04, -3.75968091e-06],
    [ 5.19067725e-04, -2.61169259e-02,  6.30842571e-04],
    [-2.25256655e-03,  2.59167244e-02, -6.37481239e-04],
])

B3LYP_REF_ETF = np.array([
    [ 2.50598126e-03,  3.33252101e-04,  3.17780136e-05],
    [-9.06812181e-04, -1.32553941e-04, -1.33029415e-05],
    [ 1.11124891e-03, -3.60578572e-02, -3.53475871e-04],
    [-2.71047750e-03,  3.58571905e-02,  3.35157983e-04],
])

B3LYP_REF_FULL = np.array([
    [-2.98849214e-03, -3.83745104e-04, -2.82205655e-05],
    [ 7.40404358e-04,  1.27644164e-04,  1.47500840e-05],
    [-1.53443210e-03,  4.54559163e-02,  6.15094567e-04],
    [ 3.28532155e-03, -4.52411741e-02, -5.96643805e-04],
])


from pyscf.nac import tduks_sf as packaged_nac


def solve_shared_tddft(mf, extype=1, collinear_samples=50):
    a, b = get_ab_sf(mf, collinear_samples=collinear_samples)
    A_baba, A_abab = a
    B_baab, B_abba = b

    mo_occ = mf.mo_occ
    n_occ_a = int((mo_occ[0] > 0).sum())
    n_virt_a = int((mo_occ[0] == 0).sum())
    n_occ_b = int((mo_occ[1] > 0).sum())
    n_virt_b = int((mo_occ[1] == 0).sum())

    A_abab_2d = A_abab.reshape((n_occ_a * n_virt_b, n_occ_a * n_virt_b))
    B_abba_2d = B_abba.reshape((n_occ_a * n_virt_b, n_occ_b * n_virt_a))
    B_baab_2d = B_baab.reshape((n_occ_b * n_virt_a, n_occ_a * n_virt_b))
    A_baba_2d = A_baba.reshape((n_occ_b * n_virt_a, n_occ_b * n_virt_a))

    casida_matrix = np.block([
        [A_abab_2d, B_abba_2d],
        [-B_baab_2d, -A_baba_2d],
    ])
    eigenvals, eigenvecs = np.linalg.eig(casida_matrix)
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx].real
    eigenvecs = eigenvecs[:, idx]

    norms = np.linalg.norm(eigenvecs[:n_occ_a * n_virt_b], axis=0) ** 2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b:], axis=0) ** 2
    valid_mask = norms > 0 if extype == 1 else norms < 0

    return (
        eigenvals[valid_mask],
        eigenvecs[:, valid_mask].T,
        n_occ_a,
        n_virt_a,
        n_occ_b,
        n_virt_b,
    )


def build_td_object(mf, solved_data, extype=1, collinear_samples=50, xy_format='new'):
    e, vecs, n_occ_a, n_virt_a, n_occ_b, n_virt_b = solved_data

    def norm_xy_old(z):
        x_flat = z[:n_occ_a * n_virt_b]
        y_flat = z[n_occ_a * n_virt_b:]
        norm_val = np.linalg.norm(x_flat) ** 2 - np.linalg.norm(y_flat) ** 2
        norm_val = np.sqrt(1.0 / norm_val)
        x = x_flat.reshape(n_occ_a, n_virt_b) * norm_val
        y = y_flat.reshape(n_occ_b, n_virt_a) * norm_val
        return ((0, x), (y, 0)) if extype == 1 else ((y, 0), (0, x))

    def norm_xy_new(z):
        x_flat = z[:n_occ_a * n_virt_b]
        y_flat = z[n_occ_a * n_virt_b:]
        norm_val = np.linalg.norm(x_flat) ** 2 - np.linalg.norm(y_flat) ** 2
        norm_val = np.sqrt(1.0 / norm_val)
        x = x_flat.reshape(n_occ_a, n_virt_b) * norm_val
        y = y_flat.reshape(n_occ_b, n_virt_a) * norm_val
        return (x, y) if extype == 1 else (y, x)

    td = sftda.uks_sf.TDDFT_SF(mf)
    td.e = e
    td.xy = [norm_xy_old(z) if xy_format == 'old' else norm_xy_new(z) for z in vecs]
    td.nstates = len(e)
    td.extype = extype
    td.collinear_samples = collinear_samples
    return td


def build_demo_molecule():
    mol = gto.Mole()
    mol.atom = '''
C      0.000000    0.000000    0.000000
O      0.000000    0.000000    1.205000
H     -0.937704    0.000000   -0.513544
H      0.937704    0.100000   -0.513544
'''
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.verbose = 0
    mol.output = '/dev/null'
    return mol.build()


def demo_call_flow():
    print('Demo: forge-native NAC call flow')
    mol = gto.Mole()
    mol.atom = '''
O     0.000000    0.000000    0.000000
H     0.000000   -0.757000    0.587000
H     0.000000    0.757000    0.587000
'''
    mol.basis = '631g'
    mol.spin = 2
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'B3LYP'
    mf.kernel()

    td = mf.TDDFT_SF().set(extype=1, collinear_samples=50, nstates=3).run()
    nac_obj = packaged_nac.NAC(td)
    nac = nac_obj.kernel(state_I=1, state_J=2, use_etfs=False, ediff=False)

    print('  molecule atoms :', mol.natm)
    print('  td states      :', len(td.e))
    print('  xy shapes      :', td.xy[0][0].shape, td.xy[0][1].shape)
    print('  nac shape      :', nac.shape)
    print('  finite         :', np.isfinite(nac).all())
    print('  nac sample     :')
    print(nac)
    assert nac.shape == (mol.natm, 3)
    assert np.isfinite(nac).all()


def check_against_cache_refs():
    print('Check: packaged NAC against cache reference values')
    print('  note: reference arrays were regenerated with PySCF 2.15.0 (SCF conv_tol=1e-12)')
    cases = [
        ('HF', True, HF_REF_ETF),
        ('HF', False, HF_REF_FULL),
        ('B3LYP', True, B3LYP_REF_ETF),
        ('B3LYP', False, B3LYP_REF_FULL),
    ]

    for xc, use_etfs, ref in cases:
        mol = build_demo_molecule()
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.conv_tol = 1e-12
        mf.kernel()
        solved_data = solve_shared_tddft(mf, extype=1, collinear_samples=50)

        td_new = build_td_object(mf, solved_data, extype=1, collinear_samples=50, xy_format='new')

        new_val = packaged_nac.NAC(td_new).kernel(state_I=1, state_J=3, use_etfs=use_etfs, ediff=False)

        diff_direct = np.max(np.abs(new_val - ref))
        diff_flipped = np.max(np.abs(-new_val - ref))
        if diff_flipped < diff_direct:
            new_val = -new_val

        np.testing.assert_allclose(new_val, ref, rtol=1e-7, atol=1e-7)

        print(f'  XC={xc:5s} use_etfs={str(use_etfs):5s} max_abs_diff={np.max(np.abs(new_val - ref)):.12e}')
        if diff_flipped < diff_direct:
            print('  phase note: compared after global sign flip')
        print('  cache ref:')
        print(ref)


class KnownValues(unittest.TestCase):
    def test_demo_call_flow(self):
        demo_call_flow()

    def test_against_cache_refs(self):
        check_against_cache_refs()


if __name__ == '__main__':
    print('Full Tests for spin-flip-TDDFT nonadiabatic couplings')
    unittest.main()
