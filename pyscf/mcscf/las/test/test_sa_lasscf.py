import unittest
import numpy as np
from scipy import linalg
from pyscf import scf, lib, tools, mcscf
from pyscf.mcscf.addons import state_average_mix
from pyscf.mcscf.las.lasscf_sync_o0 import LASSCF
from pyscf.mcscf.newton_casscf import gen_g_hop, _pack_ci_get_H
from c2h6n4_struct import structure as struct
from pyscf.csf_fci import csf_solver
from itertools import product
import os

def setUpModule():
    global mol, mf, mo_loc, las, ugg, hop, las_sing, ugg_sing, hop_sing, las_quin, ugg_quin, hop_quin, xorb, xci, nmo, ncore, nocc
    mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol.output = '/dev/null'
    mol.verbose = lib.logger.DEBUG
    mol.build ()
    mf = scf.RHF (mol).run ()
    mo_coeff = mf.mo_coeff.copy ()
    las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
    mo_loc = las.localize_init_guess ((list(range(3)),list(range(9,12))), mo_coeff=mo_coeff)
    las.state_average_(weights=[0.5,0.5], spins=[[0,0],[2,-2]])
    las.set (max_cycle_macro=1, max_cycle_micro=1, ah_level_shift=0).kernel ()

    np.random.seed (1)
    ugg = las.get_ugg ()
    ci0 = [np.random.rand (ncsf) for ncsf in ugg.ncsf_sub.ravel ()]
    ci0 = [c / linalg.norm (c) for c in ci0]
    x0 = np.concatenate ([np.zeros (ugg.nvar_orb),] + ci0)
    _, ci0_sa = ugg.unpack (x0)
    las.mo_coeff = mo_loc
    las.ci = ci0_sa
    hop = las.get_hop (ugg=ugg)

    ci0_sing = [[ci0_sa[0][0]], [ci0_sa[1][0]]]
    las_sing = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1)).set (mo_coeff=mo_loc, ci=ci0_sing)
    las_sing = las_sing.set (ah_level_shift=0, max_cycle_macro=1, max_cycle_micro=1).run ()
    las_sing = las_sing.set (mo_coeff=mo_loc, ci=ci0_sing)
    ugg_sing = las_sing.get_ugg ()
    hop_sing = las_sing.get_hop (ugg=ugg_sing)

    ci0_quin = [[ci0_sa[0][1]], [ci0_sa[1][1]]]
    las_quin = LASSCF (mf, (4,4), ((3,1),(1,3)), spin_sub=(3,3)).set (mo_coeff=mo_loc, ci=ci0_quin)
    las_quin = las_quin.set (ah_level_shift=0, max_cycle_macro=1, max_cycle_micro=1).run ()
    las_quin = las_quin.set (mo_coeff=mo_loc, ci=ci0_quin)
    ugg_quin = las_quin.get_ugg ()
    hop_quin = las_quin.get_hop (ugg=ugg_quin)

    xorb, xci = ugg.unpack (np.random.rand (ugg.nvar_tot))
    nao, nmo = mo_loc.shape
    ncore, nocc = las.ncore, las.ncore + las.ncas

def sa_2_ss_x (x_sa):
    xorb, xci_sa = ugg.unpack (x_sa)
    xci_sing = [[xci_sa[0][0]], [xci_sa[1][0]]]
    xci_quin = [[xci_sa[0][1]], [xci_sa[1][1]]]
    x_sing = ugg_sing.pack (xorb, xci_sing)
    x_quin = ugg_quin.pack (xorb, xci_quin)
    return x_sing, x_quin

def ss_2_sa_hx (hx_sing, hx_quin):
    hxorb_sing, hxci_sing = ugg_sing.unpack (hx_sing)
    hxorb_quin, hxci_quin = ugg_quin.unpack (hx_quin)
    hxorb_sa = (hxorb_sing + hxorb_quin) / 2
    hxci_sa = [[hxci_sing[0][0], hxci_quin[0][0]], [hxci_sing[1][0], hxci_quin[1][0]]]
    return ugg.pack (hxorb_sa, hxci_sa)

def examine_sector (internal):
    ij = {'core': (0,ncore),
          'active': (ncore,nocc),
          'virtual': (nocc,nmo)}
    my_xorb = np.zeros_like (xorb)
    my_xci = [[np.zeros_like (x) for x in xr] for xr in xci]
    if 'CI' in internal.upper ():
        i, j = [int (word) for word in internal.replace (',',' ').split () if word.isdigit ()]
        my_xci[i][j] = xci[i][j].copy ()
    else:
        bra, ket = internal.split ('-')
        i, j = ij[bra]
        k, l = ij[ket]
        my_xorb[i:j,k:l] = xorb[i:j,k:l]
        my_xorb[k:l,i:j] = xorb[k:l,i:j]
    x_sa = ugg.pack (my_xorb, my_xci)
    x_sing, x_quin = sa_2_ss_x (x_sa)
    
    hxorb_test, hxci_test = ugg.unpack (hop (x_sa))
    hxorb_check, hxci_check = ugg.unpack (ss_2_sa_hx (hop_sing (x_sing), hop_quin (x_quin)))

    test = []
    check = []
    for external in ('core-virtual', 'active-active', 'active-virtual', 'core-active'):
        bra, ket = external.split ('-')
        i, j = ij[bra]
        k, l = ij[ket]
        test.append (hxorb_test[i:j,k:l])
        check.append (hxorb_check[i:j,k:l])
    for i, j in ((0,0), (0,1), (1,0), (1,1)):
        test.append (hxci_test[i][j])
        check.append (hxci_check[i][j])

    return test, check

lbls = ('core-virtual', 'active-active', 'active-virtual', 'core-active', 'CI 0,0', 'CI 0,1', 'CI 1,0', 'CI 1,1')

def unroll_unpack (x):
    xorb, xci = ugg.unpack (x)
    xtuple = (xorb[:ncore,nocc:], xorb[ncore:nocc,ncore:nocc], xorb[ncore:nocc,nocc:], xorb[:ncore,ncore:nocc],
        xci[0][0], xci[0][1], xci[1][0], xci[1][1])
    return xtuple

def tearDownModule():
    global mol, mf, las, ugg, hop, las_sing, ugg_sing, hop_sing, las_quin, ugg_quin, hop_quin
    mol.stdout.close ()
    del mol, mf, las, ugg, hop, las_sing, ugg_sing, hop_sing, las_quin, ugg_quin, hop_quin


class KnownValues(unittest.TestCase):
    def test_grad (self):
        g_test = unroll_unpack (hop.get_grad ())
        g_ref = unroll_unpack (ss_2_sa_hx (hop_sing.get_grad (), hop_quin.get_grad ()))
        for lbl, g_t, g_r in zip (lbls, g_test, g_ref):
            with self.subTest (sector=lbl):
                self.assertAlmostEqual (lib.fp (g_t), lib.fp (g_r), 9)

    def test_hess (self):
        for j in lbls:
            hx_test, hx_ref = examine_sector (j)
            for i, ht, hr in zip (lbls, hx_test, hx_ref):
                with self.subTest (sector=(i,j)):
                    self.assertAlmostEqual (lib.fp (ht), lib.fp (hr), 9)
                
if __name__ == "__main__":
    print("Full Tests for LASSCF Newton-CG module functions state-averaging")
    unittest.main()


