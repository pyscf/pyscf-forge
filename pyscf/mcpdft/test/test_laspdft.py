import unittest
import numpy as np
from pyscf import gto, scf, dft, mcpdft
from pyscf.csf_fci import csf_solver
from pyscf.mcscf.las.lasscf import LASSCF, LASSCFNoSymm
from pyscf.mcscf.las.lasci import LASCINoSymm
from pyscf.mcpdft import laspdft

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_h4 (self):
        mol = gto.M()
        mol.atom='''H -5.10574 2.01997 0.00000;
                    H -4.29369 2.08633 0.00000;
                    H -3.10185 2.22603 0.00000;
                    H -2.29672 2.35095 0.00000'''
        mol.basis='sto3g'
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.build()
        mf = scf.ROHF(mol).newton().run()
        las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1,1))
        mc = laspdft.get_mcpdft_child_class(las, 'tPBE', grids_level=1)
        frag_atom_list = ([0, 1] , [2, 3])
        mo0 = mc.localize_init_guess (frag_atom_list)
        mc.conv_tol_grad = 1e-6
        mc.kernel(mo0)
        self.assertTrue (mc.converged)
        self.assertAlmostEqual (mc.e_tot, -2.285623046332528, 6)

        # PDFT on an already-converged LASSCF reproduces the energy
        las = LASSCF(mf, (2, 2), (2, 2), spin_sub=(1,1))
        las.lasci_(mc.mo_coeff, ci0=mc.ci)
        mc2 = laspdft.get_mcpdft_child_class(las, 'tPBE', grids_level=1)
        epdft2 = mc2.compute_pdft_energy_()[0]
        self.assertAlmostEqual (epdft2, mc.e_tot, 7)

    def test_h4_constructors (self):
        # laspdft.LASSCF and LASCI, also available as mcpdft.LASSCF and mcpdft.LASCI once laspdft
        # is imported
        self.assertIs (mcpdft.LASSCF, laspdft.LASSCFPDFT)
        self.assertIs (mcpdft.LASCI, laspdft.LASCIPDFT)
        mf = _h4_mf ()
        mc = mcpdft.LASSCF (mf, 'tPBE', (2, 2), (2, 2), spin_sub=(1,1), grids_level=1)
        self.assertIsInstance (mc, LASSCFNoSymm)
        mo0 = mc.localize_init_guess (([0, 1] , [2, 3]))
        mc.conv_tol_grad = 1e-6
        mc.kernel (mo0)
        self.assertTrue (mc.converged)
        self.assertAlmostEqual (mc.e_tot, -2.285623046332528, 6)
        with self.subTest ('from a LAS object'):
            las = LASSCF (mf, (2, 2), (2, 2), spin_sub=(1,1))
            las.lasci_(mc.mo_coeff, ci0=mc.ci)
            mc2 = mcpdft.LASSCF (las, 'tPBE', grids_level=1)
            self.assertEqual (list (mc2.ncas_sub), [2, 2])
            self.assertIsNot (mc2.mo_coeff, las.mo_coeff)
            self.assertAlmostEqual (mc2.compute_pdft_energy_()[0], mc.e_tot, 7)
        with self.subTest ('LASCI'):
            # At the converged LASSCF orbitals, LASCI-PDFT gives the LASSCF-PDFT energy
            mc3 = mcpdft.LASCI (mf, 'tPBE', (2, 2), (2, 2), spin_sub=(1,1), grids_level=1)
            self.assertIsInstance (mc3, LASCINoSymm)
            self.assertNotIsInstance (mc3, LASSCFNoSymm)
            mc3.kernel (mc.mo_coeff)
            self.assertTrue (mc3.converged)
            self.assertAlmostEqual (mc3.e_tot, mc.e_tot, 6)

    def test_lroots_not_implemented (self):
        mf = _h4_mf ()
        mc = mcpdft.LASCI (mf, 'tPBE', (2, 2), (2, 2), spin_sub=(1,1), grids_level=1)
        mo0 = mc.localize_init_guess (([0, 1] , [2, 3]))
        with self.assertRaises (NotImplementedError):
            mc.kernel (mo0, lroots=np.array ([[2],[1]]))

    def test_mcpdft_namespace_not_overwritten (self):
        saved = mcpdft.LASCI
        try:
            mcpdft.LASCI = sentinel = object ()
            laspdft._add_to_mcpdft_namespace ()
            self.assertIs (mcpdft.LASCI, sentinel)
        finally:
            mcpdft.LASCI = saved

    def test_h4_state_average_cas_limit (self):
        # With one fragment, SA-LASSCF is SA-CASSCF, so SA-LAS-PDFT must equal SA-MC-PDFT
        mc, mc_ref = _h4_sa_cas_limit ()
        self.assertTrue (mc.converged)
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (mc.e_mcscf[i], mc_ref.e_mcscf[i], 7)
                self.assertAlmostEqual (mc.e_states[i], mc_ref.e_states[i], 7)
                self.assertAlmostEqual (mc.e_ot[i], mc_ref.e_ot[i], 7)
        # As in SA-MC-PDFT, e_tot is the weighted average
        self.assertAlmostEqual (mc.e_tot, mc_ref.e_tot, 7)
        e_states = mc.compute_pdft_energy_ ()[2]
        for i in range (2):
            with self.subTest ('compute_pdft_energy_', state=i):
                self.assertAlmostEqual (e_states[i], mc_ref.e_states[i], 7)
        with self.subTest ('energy decomposition not implemented'):
            with self.assertRaises (NotImplementedError):
                mc.get_energy_decomposition ()

    def test_h4_state_average_e_states_attribute (self):
        mc, mc_ref = _h4_sa_cas_limit ()
        for i in range (2):
            with self.subTest (state=i):
                self.assertAlmostEqual (mc.e_states[i], mc_ref.e_states[i], 7)

def _h4_mf ():
    if 'h4_mf' not in _cache:
        mol = gto.M (atom='''H -5.10574 2.01997 0.00000;
                             H -4.29369 2.08633 0.00000;
                             H -3.10185 2.22603 0.00000;
                             H -2.29672 2.35095 0.00000''',
                     basis='sto3g', verbose=0, output='/dev/null')
        _cache['h4_mf'] = scf.ROHF (mol).newton ().run ()
    return _cache['h4_mf']

_cache = {}
def _h4_sa_cas_limit ():
    if 'h4_sa' not in _cache:
        mol = gto.M (atom='H 0 0 0; H 0.8 0 0; H 0.2 2.6 0.1; H 1.0 2.8 -0.1', basis='sto3g',
                     verbose=0, output='/dev/null')
        mf = scf.RHF (mol).run ()
        las = LASSCF (mf, (4,), (4,), spin_sub=(1,)).set (conv_tol_grad=1e-6)
        las.state_average_(weights=[0.5,0.5], spins=[0,2], smults=[1,3])
        mc = laspdft.get_mcpdft_child_class (las, 'tPBE', grids_level=1)
        mc.kernel ()
        mc_ref = mcpdft.CASSCF (mf, 'tPBE', 4, 4, grids_level=1).set (conv_tol=1e-10)
        mc_ref = mc_ref.state_average_mix_(
            [csf_solver (mol, smult=m2+1).set (spin=m2) for m2 in (0,2)], [0.5,0.5])
        mc_ref.kernel (mc.mo_coeff)
        _cache['h4_sa'] = (mc, mc_ref)
    return _cache['h4_sa']

if __name__ == "__main__":
    print("Full Tests for LAS-PDFT")
    unittest.main()
