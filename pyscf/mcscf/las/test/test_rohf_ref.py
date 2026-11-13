import unittest
from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf.las import lasscf as syn

def setUpModule():
    global mf_c2h6n4, mf_c2h4n4
    from c2h6n4_struct import structure as struct
    mol = struct (0.2, 0.2, '6-31g', symmetry=False)
    mol.spin = 4
    mol.output = '/dev/null'
    mol.verbose = 0
    mol.build ()
    mf_c2h6n4 = scf.RHF (mol).run (max_cycle=100)
    from c2h4n4_struct import structure as struct
    mol = struct (0.2, 0.2, '6-31g', symmetry='Cs')
    mol.spin = 4
    mol.output = '/dev/null'
    mol.verbose = 0
    mol.build ()
    mf_c2h4n4 = scf.RHF (mol).run (max_cycle=100)

def tearDownModule():
    global mf_c2h6n4, mf_c2h4n4
    mf_c2h6n4.mol.stdout.close ()
    mf_c2h4n4.mol.stdout.close ()
    del mf_c2h6n4, mf_c2h4n4

def _las_e_tot (mf, frags, lasci_first=False, **kwargs):
    las = syn.LASSCF (mf, (4, 4), ((4,2), (4,2)), spin_sub=(3,3))
    las.init_guess_ci = 'aufbau'
    las.mo_coeff = las.localize_init_guess (frags, mf.mo_coeff, **kwargs)
    if lasci_first: las.lasci_() # need to initialize some FCISolver attributes
    las.ci = las.get_init_guess_ci ()
    return las.energy_elec () + las.energy_nuc ()

class KnownValues (unittest.TestCase):
    # The aufbau CI guess fills alpha and beta in orbital order in each fragment, so the LAS
    # product state is a single ROHF-type determinant. If localize_init_guess keeps the ROHF
    # doubly- and singly-occupied orbitals apart (mo_occ), that determinant is the ROHF one and
    # the energy is E_ROHF. With only freeze_cas_spaces=True, each fragment's orbitals mix the two
    # occupation classes, so the energy lies above E_ROHF (here by ~1e-3 for C2H6N4 and ~1e-1 for
    # C2H4N4; it can't be below, by the variational principle).

    def test_c2h6n4_casci (self):
        mc = mcscf.CASCI (mf_c2h6n4, 4, 4).run ()
        self.assertAlmostEqual (mf_c2h6n4.e_tot, mc.e_tot, 6)

    def test_c2h6n4_freeze_cas_spaces (self):
        e_tot = _las_e_tot (mf_c2h6n4, [[1,2],[9,10]], freeze_cas_spaces=True)
        self.assertGreaterEqual (e_tot, mf_c2h6n4.e_tot - 1e-6)

    def test_c2h6n4_mo_occ (self):
        e_tot = _las_e_tot (mf_c2h6n4, [[1,2],[9,10]], mo_occ=mf_c2h6n4.mo_occ)
        self.assertAlmostEqual (e_tot, mf_c2h6n4.e_tot, 6)

    def test_c2h4n4_casci (self):
        mc = mcscf.CASCI (mf_c2h4n4, 4, 4).run ()
        self.assertAlmostEqual (mf_c2h4n4.e_tot, mc.e_tot, 6)

    def test_c2h4n4_freeze_cas_spaces (self):
        e_tot = _las_e_tot (mf_c2h4n4, [[1,2],[7,8]], lasci_first=True, freeze_cas_spaces=True)
        self.assertGreaterEqual (e_tot, mf_c2h4n4.e_tot - 1e-6)

    def test_c2h4n4_mo_occ (self):
        e_tot = _las_e_tot (mf_c2h4n4, [[1,2],[7,8]], lasci_first=True, mo_occ=mf_c2h4n4.mo_occ)
        self.assertAlmostEqual (e_tot, mf_c2h4n4.e_tot, 6)

    # The two smallbasis tests are smoke tests: they pass if nothing raises.
    def test_smallbasis (self):
        # From https://github.com/MatthewRHermes/mrh/issues/187
        # Initialization
        mol = gto.M(verbose=0, output='/dev/null')
        mol.atom='''H -5.10574 2.01997 0.00000;
                    H -4.29369 2.08633 0.00000;
                    H -3.10185 2.22603 0.00000;
                    H -2.29672 2.35095 0.00000'''
        mol.basis='STO-6G'
        mol.build()
        
        # Mean field calculation
        mf = scf.ROHF(mol).newton().run()
        
        # LASSCF Calculations
        las = syn.LASSCF(mf,(2, 2),(2, 2),spin_sub=(1, 1))
        frag_atom_list = ([0, 1], [2, 3])
        mo0 = las.localize_init_guess(frag_atom_list, freeze_cas_spaces=True)
        las.max_cycle_macro=1
        las.kernel(mo0)

    def test_smallbasis_symm (self):
        # From https://github.com/MatthewRHermes/mrh/issues/187
        # Initialization
        mol = gto.M(verbose=0, output='/dev/null')
        mol.atom='''H -5 -2 0.00000;
                    H -4 -2 0.00000;
                    H -2 2 0.00000;
                    H -1 2 0.00000'''
        mol.basis='STO-6G'
        mol.symmetry='Cs'
        mol.build()
        
        # Mean field calculation
        mf = scf.ROHF(mol).newton().run()
        
        # LASSCF Calculations
        las = syn.LASSCF(mf,(2, 2),(2, 2),spin_sub=(1, 1))
        frag_atom_list = ([0, 1], [2, 3])
        mo0 = las.localize_init_guess(frag_atom_list, freeze_cas_spaces=True)
        las.max_cycle_macro=1
        las.kernel(mo0)


if __name__ == "__main__":
    print ("Full Tests for LASSCF orbital localization from ROHF references")
    unittest.main()
