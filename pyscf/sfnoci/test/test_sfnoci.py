import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf.sfnoci.sfnoci import SFNOCI, SFGNOCI


def setUpModule():
    global mol, mo0, setocc, ro_occ
    mol = gto.Mole()
    mol.atom = [['Li', (0, 0, 0)],['F',(0,0,1.3)]]
    mol.basis = 'ccpvtz'
    mr=scf.RHF(mol)
    mr.kernel()

    mo0=mr.mo_coeff
    occ=mr.mo_occ
    setocc=numpy.zeros((2,occ.size))
    setocc[:,occ==2]=1
    setocc[1][3]=0
    setocc[0][6]=1
    ro_occ=setocc[0][:]+setocc[1][:]


def tearDownModule():
    global mol, mo0, setocc, ro_occ
    del mol, mo0, setocc, ro_occ

class KnownValues(unittest.TestCase):
    def test_sfnoci(self):
        mol = gto.Mole()
        mol.atom = [['Li', (0, 0, 0)],['F',(0,0,1.6)]]
        mol.basis = 'ccpvtz'
        mol.spin=2
        mol.build(0,0)
        rm=scf.ROHF(mol)
        dm_ro=rm.make_rdm1(mo0,ro_occ)
        rm=scf.addons.mom_occ(rm,mo0,setocc)
        rm.scf(dm_ro)
        rm.kernel()

        mo=rm.mo_coeff
        as_list=[3,6,7,10]
        mySFNOCI = SFNOCI(rm,4,(2,2))
        mySFNOCI.lowdin_thres= 0.5

        from pyscf.mcscf import addons
        mo = addons.sort_mo(mySFNOCI,mo, as_list,1)
        e, _, ci = mySFNOCI.kernel(mo)

        mol.atom = [['Li', (0, 0, 0)],['F',(0,0,4.5)]]
        mol.build(0,0)
        rm=scf.ROHF(mol)
        dm_ro=rm.make_rdm1(mo0,ro_occ)
        rm=scf.addons.mom_occ(rm,mo0,setocc)
        rm.scf(dm_ro)
        rm.kernel()

        mo=rm.mo_coeff
        mySFNOCI = SFNOCI(rm,4,(2,2))
        mySFNOCI.lowdin_thres= 0.5

        mo = addons.sort_mo(mySFNOCI,mo, as_list,1)
        re, _, ci = mySFNOCI.kernel(mo)

        self.assertAlmostEqual(e-re, -0.11279175858597057, 6)

    def test_sfgnoci(self):
        mol = gto.Mole()
        mol.atom = [['Li', (0, 0, 0)],['F',(0,0,1.6)]]
        mol.basis = 'ccpvtz'
        mol.spin=2
        mol.build(0,0)
        rm=scf.ROHF(mol)
        dm_ro=rm.make_rdm1(mo0,ro_occ)
        rm=scf.addons.mom_occ(rm,mo0,setocc)
        rm.scf(dm_ro)
        rm.kernel()

        mo=rm.mo_coeff
        as_list=[3,6,7,10]
        mySFNOCI = SFGNOCI(rm,4,(2,2), groupA = 'Li')
        mySFNOCI.lowdin_thres= 0.5

        from pyscf.mcscf import addons
        mo = addons.sort_mo(mySFNOCI,mo, as_list,1)
        e, _, ci = mySFNOCI.kernel(mo)

        mol.atom = [['Li', (0, 0, 0)],['F',(0,0,4.5)]]
        mol.build(0,0)
        rm=scf.ROHF(mol)
        dm_ro=rm.make_rdm1(mo0,ro_occ)
        rm=scf.addons.mom_occ(rm,mo0,setocc)
        rm.scf(dm_ro)
        rm.kernel()

        mo=rm.mo_coeff
        mySFNOCI = SFGNOCI(rm,4,(2,2), groupA = 'Li')
        mySFNOCI.lowdin_thres= 0.5

        mo = addons.sort_mo(mySFNOCI,mo, as_list,1)
        re, _, ci = mySFNOCI.kernel(mo)

        self.assertAlmostEqual(e-re, -0.1129230696617185, 6)



if __name__ == "__main__":
    print("Full Tests for SF(G)NOCI")
    unittest.main()
