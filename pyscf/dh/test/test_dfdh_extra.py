"""Known reference energy tests for DH functionals.
"""

import unittest
from pyscf import gto
from pyscf.dft.libxc import _itrf
from pyscf.dh import DFDH

_HAS_NAMED_PARAMS = hasattr(_itrf, 'LIBXC_xc_func_find_ext_params_name')


class TestKnownEnergy(unittest.TestCase):

    def test_DSD_PBEPBE_D3BJ(self):
        ref = -76.20415120  # QChem 5.1.1
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="DSD-PBEPBE-D3BJ").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_DSD_PBEB95_D3BJ(self):
        ref = -76.22012321  # QChem 5.1.1
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="DSD-PBEB95-D3BJ").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_wB2PLYP(self):
        ref = -76.223828450855  # ORCA 5.0.4
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="wB2PLYP").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_wB2GPPLYP(self):
        ref = -76.216969239009  # ORCA 5.0.4
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="wB2GPPLYP").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_wB88PP86(self):
        ref = -76.223203247789  # ORCA 5.0.4
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="wB88PP86").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_wPBEPP86(self):
        ref = -76.261632401477  # ORCA 5.0.4
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="wPBEPP86").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_RSX_0DH(self):
        ref = -76.223451307545  # ORCA 5.0.4
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="RSX-0DH").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_RSX_QIDH(self):
        ref = -76.208221215310  # ORCA 5.0.4
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="RSX-QIDH").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_PTPSS(self):
        ref = -76.28365901  # QChem 5.1.1
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="PTPSS").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    def test_PWPB95(self):
        ref = -76.30404653  # QChem 5.1.1
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="PWPB95").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    @unittest.skipUnless(_HAS_NAMED_PARAMS,
                         'Named-param ext_params not available in this PySCF')
    def test_wB97X_2_TQZ(self):
        ref = -76.24074263  # QChem 5.1.1
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="wB97X-2-TQZ").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)

    @unittest.skipUnless(_HAS_NAMED_PARAMS,
                         'Named-param ext_params not available in this PySCF')
    def test_wB97X_2_LP(self):
        ref = -76.28031455  # QChem 5.1.1
        mol = gto.M(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = DFDH(mol, xc="wB97X-2-LP").run()
        self.assertAlmostEqual(mf.e_tot, ref, places=4)
