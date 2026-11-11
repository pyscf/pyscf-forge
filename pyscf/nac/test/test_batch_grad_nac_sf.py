#!/usr/bin/env python

import pathlib
import sys

import numpy as np

try:
    import pytest
except ModuleNotFoundError:
    class _PytestMarkShim:
        @staticmethod
        def parametrize(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class _PytestShim:
        mark = _PytestMarkShim()

        @staticmethod
        def skip(message):
            raise RuntimeError(message)

    pytest = _PytestShim()


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYSCF_NS_PATH = REPO_ROOT / 'pyscf'

# Frozen from the legacy standalone SFTDA gradient/NAC implementation before it
# was archived.  Keeping numeric oracles here makes the test independent of the
# surrounding workspace layout and prevents archived code from becoming a
# runtime dependency.  Source SHA-256:
# gradient: 50c6545c3f51b2ad1cd85e5b7039d1150c3067821de3924e09ed32c13b30029d
# NAC: 9fc01c017c999440294e4b4d31cea12466e471cb555bc622f967f5c43cf0ca61
LEGACY_REFERENCE = {
    ('HF', 0, True): {
        'grad': np.array([
            [4.32732285714086e-17, -2.3016207548549628e-15, 0.352888943019134],
            [9.35944578251098e-16, 0.32334643844266475, -0.17644447150956566],
            [-9.792178068225057e-16, -0.32334643844266187, -0.176444471509565],
        ]),
        'nac': np.array([
            [1.0921448094601843e-14, 0.57294550798535, -1.797088590716371e-14],
            [-4.867369091884677e-15, -0.2864727539926854, -0.054353155550487774],
            [-6.054079002716927e-15, -0.28647275399267685, 0.054353155550503796],
        ]),
    },
    ('B3LYP', 1, False): {
        'grad': np.array([
            [-3.08063583295919e-16, 7.91647814972555e-15, -0.0029494221904484696],
            [-2.4940687641299026e-16, 0.011241681431405581, 0.001468703102752933],
            [2.324728609537041e-16, -0.011241681431415795, 0.0014687031027460495],
        ]),
        'nac': np.array([
            [0.044842929399466415, 7.442612845276809e-16, -2.431058108329938e-16],
            [-0.022594953998347177, -6.504399889705097e-16, 4.0788298679907015e-16],
            [-0.02259495399833889, -4.445500794083051e-16, -9.273534299958306e-17],
        ]),
    },
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyscf
import pyscf.grad
import pyscf.nac

if str(PYSCF_NS_PATH) in pyscf.__path__:
    pyscf.__path__.remove(str(PYSCF_NS_PATH))
pyscf.__path__.insert(0, str(PYSCF_NS_PATH))
if str(PYSCF_NS_PATH / 'grad') not in pyscf.grad.__path__:
    pyscf.grad.__path__.insert(0, str(PYSCF_NS_PATH / 'grad'))
if str(PYSCF_NS_PATH / 'nac') not in pyscf.nac.__path__:
    pyscf.nac.__path__.insert(0, str(PYSCF_NS_PATH / 'nac'))

from pyscf import dft
from pyscf import gto
from pyscf import sftda
from pyscf.grad import tduks_sf as packaged_grad
from pyscf.nac import batch_grad_nac
from pyscf.nac import tduks_sf as packaged_nac
from pyscf.sftda.uhf_sf import get_ab_sf


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
    # Since numpy 2.5, eig returns complex arrays even when all eigenvalues are real
    assert not eigenvecs.imag.any()
    eigenvals, eigenvecs = eigenvals.real, eigenvecs.real
    idx = eigenvals.real.argsort()
    eigenvals = eigenvals[idx].real
    eigenvecs = eigenvecs[:, idx]

    norms = np.linalg.norm(eigenvecs[: n_occ_a * n_virt_b], axis=0) ** 2
    norms -= np.linalg.norm(eigenvecs[n_occ_a * n_virt_b :], axis=0) ** 2

    if extype == 1:
        valid_mask = norms > 1e-3
        valid_e = eigenvals[valid_mask]
        valid_vecs = eigenvecs[:, valid_mask].T
    else:
        valid_mask = norms < -1e-3
        valid_e = -eigenvals[valid_mask]
        valid_vecs = eigenvecs[:, valid_mask][:, ::-1].T
        valid_e = valid_e[::-1]

    return (valid_e.real, valid_vecs, n_occ_a, n_virt_a, n_occ_b, n_virt_b)


def build_td_object(mf, solved_data, extype=1, collinear_samples=50, xy_format='new'):
    e, vecs, n_occ_a, n_virt_a, n_occ_b, n_virt_b = solved_data

    def norm_xy_new(z):
        if extype == 1:
            x_flat = z[: n_occ_a * n_virt_b]
            y_flat = z[n_occ_a * n_virt_b :]
            x = x_flat.reshape(n_occ_a, n_virt_b)
            y = y_flat.reshape(n_occ_b, n_virt_a)
        else:
            x_flat = z[n_occ_a * n_virt_b :]
            y_flat = z[: n_occ_a * n_virt_b]
            x = x_flat.reshape(n_occ_b, n_virt_a)
            y = y_flat.reshape(n_occ_a, n_virt_b)
        norm_val = np.linalg.norm(x) ** 2 - np.linalg.norm(y) ** 2
        norm_val = np.sqrt(1.0 / norm_val)
        return x * norm_val, y * norm_val

    def norm_xy_old(z):
        x, y = norm_xy_new(z)
        return ((0, x), (y, 0)) if extype == 1 else ((x, 0), (0, y))

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
O     0.000000    0.000000    0.000000
H     0.000000   -0.757000    0.587000
H     0.000000    0.757000    0.587000
'''
    mol.basis = '631g'
    mol.spin = 2
    mol.verbose = 0
    mol.output = '/dev/null'
    return mol.build()


def assert_nac_allclose(actual, expected, rtol=1e-7, atol=1e-8):
    diff_direct = np.max(np.abs(actual - expected))
    diff_flipped = np.max(np.abs(actual + expected))
    if diff_flipped < diff_direct:
        actual = -actual
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def assert_cache_expectation(cache_stats, xc):
    if xc.upper() == 'HF':
        assert cache_stats['xc_block_hits'] == 0
        return
    assert cache_stats['xc_block_hits'] > 0


@pytest.mark.parametrize('xc, extype', [('HF', 0), ('B3LYP', 1)])
def test_batch_matches_current_standalone(xc, extype):
    mol = build_demo_molecule()
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.kernel()
    solved_data = solve_shared_tddft(mf, extype=extype, collinear_samples=50)
    td = build_td_object(mf, solved_data, extype=extype, collinear_samples=50, xy_format='new')

    force, nacs, timings = batch_grad_nac.compute_fssh_data(
        td, states=[1, 2], active_state=1, use_etfs=True, ediff=True, verbose=0
    )

    grad_ref = packaged_grad.Gradients(td).kernel(state=1)
    nac_ref = packaged_nac.NAC(td).kernel(state_I=1, state_J=2, use_etfs=True, ediff=True)

    np.testing.assert_allclose(-force, grad_ref, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(nacs[(1, 2)], nac_ref, rtol=1e-7, atol=1e-8)
    assert_cache_expectation(timings['cache_stats'], xc)


def test_batch_matches_current_full_nac():
    mol = build_demo_molecule()
    mf = dft.UKS(mol)
    mf.xc = 'B3LYP'
    mf.kernel()
    solved_data = solve_shared_tddft(mf, extype=1, collinear_samples=50)
    td = build_td_object(mf, solved_data, extype=1, collinear_samples=50, xy_format='new')

    _, nacs, timings = batch_grad_nac.compute_fssh_data(td, states=[1, 2], active_state=1, use_etfs=False, ediff=True, verbose=0)
    nac_ref = packaged_nac.NAC(td).kernel(state_I=1, state_J=2, use_etfs=False, ediff=True)

    np.testing.assert_allclose(nacs[(1, 2)], nac_ref, rtol=1e-7, atol=1e-8)
    assert_cache_expectation(timings['cache_stats'], 'B3LYP')


@pytest.mark.parametrize('xc, extype, use_etfs', [('HF', 0, True), ('B3LYP', 1, False)])
def test_batch_matches_old_reference(xc, extype, use_etfs):
    mol = build_demo_molecule()
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.kernel()
    solved_data = solve_shared_tddft(mf, extype=extype, collinear_samples=50)
    td = build_td_object(mf, solved_data, extype=extype, collinear_samples=50, xy_format='new')

    force, nacs, timings = batch_grad_nac.compute_fssh_data(
        td, states=[1, 2], active_state=1, use_etfs=use_etfs, ediff=True, verbose=0
    )
    ref = LEGACY_REFERENCE[(xc, extype, use_etfs)]

    np.testing.assert_allclose(-force, np.asarray(ref['grad']), rtol=1e-6, atol=1e-5)
    assert_nac_allclose(nacs[(1, 2)], np.asarray(ref['nac']), rtol=1e-6, atol=1e-5)
    assert_cache_expectation(timings['cache_stats'], xc)
