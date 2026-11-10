import numpy as np
import unittest
from pyscf import gto, scf, mcscf, lib
from pyscf.lib import logger
from pyscf.qmmm import add_mm_charges
from pyscf.csf_fci import csf_solver
from pyscf.mcscf.las.addons import state_average_n_mix, get_h1e_zipped_fcisolver
import time

def zip_kernel(casci, mo_coeff=None, ci0=None, verbose=logger.NOTE):
    '''CASCI solver
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    log = logger.new_logger(casci, verbose)
    t0 = (time.process_time(), time.time())
    log.debug('Start CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas

    # 2e
    eri_cas = casci.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to CAS space', *t0)

    # 1e
    h1eff, energy_core = casci.get_h1eff(mo_coeff)
    h1eff_hp, energy_core_hp = mc_hp.get_h1eff (mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    t1 = log.timer('effective h1e in CAS space', *t1)

    if h1eff.shape[0] != ncas:
        raise RuntimeError('Active space size error. nmo=%d ncore=%d ncas=%d' %
                           (mo_coeff.shape[1], casci.ncore, ncas))

    # H1E zip
    h1eff = [h1eff_hp, h1eff, h1eff]
    energy_core = [energy_core_hp, energy_core, energy_core]

    # FCI
    max_memory = max(400, casci.max_memory-lib.current_memory()[0])
    e_tot, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, verbose=log,
                                           max_memory=max_memory,
                                           ecore=energy_core)

    t1 = log.timer('FCI solver', *t1)
    energy_core = np.einsum ('i,i->', casci.fcisolver.weights, energy_core)
    e_cas = e_tot - energy_core
    return e_tot, e_cas, fcivec


def setUpModule():
    global mol, mf, mc, anion, rad1, rad2, mf_hp, mc_hp
    mol = gto.M (atom = 'O 0 0 0; H 1.145 0 0', basis='6-31g', symmetry=True, charge=-1, spin=0, verbose=0, output='/dev/null')
    mf = scf.RHF (mol).set (conv_tol=1e-10).run ()
    mc = mcscf.CASCI (mf, 8, 8).set (conv_tol=1e-10).run ()

    anion = csf_solver (mol, smult=1)
    anion.wfnsym = 'A1'

    rad1 = csf_solver (mol, smult=2)
    rad1.spin = 1
    rad1.charge = 1
    rad1.wfnsym = 'E1x'

    rad2 = csf_solver (mol, smult=2)
    rad2.spin = 1
    rad2.charge = 1
    rad2.wfnsym = 'E1y'

    mf_hp = add_mm_charges (mf, [[-3.0, 0.0, 0.0]], [1.0])
    mc_hp = mcscf.CASCI (mf_hp, 8, 8).set (fcisolver=anion).run ()
    mc = state_average_n_mix (mc, [anion, rad1, rad2], [1.0/3.0,]*3)
    mc.fcisolver = get_h1e_zipped_fcisolver (mc.fcisolver)
    mc.ci = None
    with lib.temporary_env (mcscf.casci, kernel=zip_kernel):
        mc.kernel ()

def tearDownModule():
    global mol, mf, mc, anion, rad1, rad2, mf_hp, mc_hp
    mol.stdout.close ()
    del mol, mf, mc, anion, rad1, rad2, mf_hp, mc_hp


class KnownValues(unittest.TestCase):
    def test_energies (self):
        self.assertAlmostEqual (mc.e_states[0], mc_hp.e_tot, 9)
        self.assertAlmostEqual (mc.e_states[1], mc.e_states[2], 9)

    def test_occ (self):
        dm1 = mc.fcisolver.make_rdm1 (mc.ci, mc.ncas, mc.nelecas)
        dm1_states = mc.fcisolver.states_make_rdm1 (mc.ci, mc.ncas, mc.nelecas)
        self.assertAlmostEqual (np.trace (dm1), 7.3333333333, 9)
        self.assertAlmostEqual (np.trace (dm1_states[0]), 8.0, 9)
        self.assertAlmostEqual (np.trace (dm1_states[1]), 7.0, 9)
        self.assertAlmostEqual (np.trace (dm1_states[2]), 7.0, 9)

if __name__ == "__main__":
    print("Full Tests for h1ezip")
    unittest.main()




