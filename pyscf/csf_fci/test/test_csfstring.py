import unittest
import numpy as np
from scipy import special
from pyscf.csf_fci.csfstring import *
from pyscf import lib
from pyscf.csf_fci import csfstring

def case_spin_evecs (ks, nspin, ms):
    neleca = int (round (nspin/2 + ms))
    nelecb = int (round (nspin/2 - ms))
    na = (nspin + neleca - nelecb) // 2
    ndet = int (special.comb (nspin, na, exact=True))
    spinstrs = cistring.addrs2str (nspin, na, list (range (ndet)))

    S2mat = np.zeros ((ndet, ndet), dtype=np.float64)
    twoMS = int (round (2 * ms))
    libcsf.FCICSFmakeS2mat (S2mat.ctypes.data_as (ctypes.c_void_p),
                            spinstrs.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (ndet),
                            ctypes.c_int (nspin),
                            ctypes.c_int (twoMS))

    evals = []
    evecs = []
    for s in np.arange (abs (ms), (nspin/2)+0.1, 1):
        smult = int (round (2*s+1))
        umat = get_spin_evecs (nspin, neleca, nelecb, smult)
        evecs.append (umat)
        evals.append (s*(s+1)*np.ones (umat.shape[1]))
    evals = np.concatenate (evals)
    evecs = np.concatenate (evecs, axis=-1)
    ks.assertEqual (evecs.shape, S2mat.shape)
    ovlp = np.dot (evecs.T, evecs)
    S2op = reduce (np.dot, (evecs.T, S2mat, evecs))
    ovlperr = linalg.norm (ovlp - np.eye (evecs.shape[1]))
    diagerr = linalg.norm (S2op - np.diag (evals))
    ks.assertLess (ovlperr, 1e-8)
    ks.assertLess (diagerr, 1e-8)

class KnownValues(unittest.TestCase):

    def setUp (self):
        self.sweep_min_nspin = csfstring.SWEEP_MIN_NSPIN

    def tearDown (self):
        csfstring.SWEEP_MIN_NSPIN = self.sweep_min_nspin

    def test_addrs2str_strs2addr (self):
        for nspin in range (21):
            for s in np.arange ((nspin%2)/2, (nspin/2)+0.1, 1):
                with self.subTest (nspin=nspin, s=s):
                    smult = int (round (2*s+1))
                    ncsf = count_csfs (nspin, smult)
                    #print ("Supposedly {} csfs of {} spins with overall smult = {}".format (ncsf, nspin, smult))
                    rand_addrs = np.random.randint (0, high=ncsf, size=min (ncsf, 5), dtype=np.int32)
                    rand_strs = addrs2str (nspin, smult, rand_addrs)
                    rand_addrs_2 = strs2addr (nspin, smult, rand_strs)
                    self.assertTrue (np.all (rand_addrs == rand_addrs_2))

    def test_spin_evecs (self):
        for nspin in range (15):
            for ms in np.arange (-(nspin/2), (nspin/2)+0.1, 1):
                with self.subTest (nspin=nspin, ms=ms):
                    case_spin_evecs (self, nspin, ms)

    def test_spin_evecs_dot (self):
        # The sequential-coupling sweep (FCICSFspinevecs_dot) must reproduce the dense umat, both ways
        csfstring.SWEEP_MIN_NSPIN = 0
        rng = np.random.default_rng (0)
        for nspin in range (13):
            for twoS in range (nspin%2, nspin+1, 2):
                for twoM in range (-twoS, twoS+1, 2):
                    # paired electrons don't enter umat
                    neleca, nelecb = (nspin+twoM)//2 + 1, (nspin-twoM)//2 + 1
                    umat = get_spin_evecs (nspin, neleca, nelecb, twoS+1)
                    # 1 row: parallel over pairs; 17 rows: parallel over row blocks
                    for nrow in (1, 17):
                        with self.subTest (nspin=nspin, twoS=twoS, twoM=twoM, nrow=nrow):
                            x = rng.standard_normal ((nrow, umat.shape[0]))
                            y = rng.standard_normal ((nrow, umat.shape[1]))
                            self.assertAlmostEqual (lib.fp (spin_evecs_dot (x, nspin, neleca, nelecb, twoS+1)),
                                                    lib.fp (np.dot (x, umat)), 10)
                            self.assertAlmostEqual (lib.fp (spin_evecs_dot (y, nspin, neleca, nelecb, twoS+1,
                                                                            transpose=True)),
                                                    lib.fp (np.dot (y, umat.T)), 10)
        with self.subTest ('3-index input'):
            umat = get_spin_evecs (12, 9, 3, 7)
            x = rng.standard_normal ((2, 5, umat.shape[0]))
            self.assertAlmostEqual (lib.fp (spin_evecs_dot (x, 12, 9, 3, 7)),
                                    lib.fp (np.tensordot (x, umat, axes=1)), 10)

    def test_sweep_dispatch (self):
        # det <-> csf transformations agree whether every nspin goes through the sweep or none does
        norb, neleca, nelecb = 8, 4, 4
        rng = np.random.default_rng (1)
        for smult in (1, 3, 5):
            t = CSFTransformer (norb, neleca, nelecb, smult)
            x = rng.standard_normal (t.ndet)
            c = rng.standard_normal (t.ncsf)
            confs = np.arange (0, t.econf_det_mask.max ()+1, 7)
            ndet_confs = np.count_nonzero (np.isin (t.econf_det_mask, confs))
            op = rng.standard_normal ((ndet_confs, ndet_confs))
            op += op.T
            results = []
            for cutoff in (0, 99):
                csfstring.SWEEP_MIN_NSPIN = cutoff
                results.append ([t.vec_det2csf (x, normalize=False),
                                 t.vec_csf2det (c, normalize=False),
                                 csfstring._transform_det2csf (x[None,:], norb, neleca, nelecb, smult,
                                                               csd_mask=t.csd_mask, project=True),
                                 t.mat_det2csf_confspace (op, confs)[0]])
            for lbl, r0, r1 in zip (('det2csf', 'csf2det', 'project', 'confspace'), *results):
                with self.subTest (smult=smult, transform=lbl):
                    self.assertAlmostEqual (lib.fp (r0), lib.fp (r1), 10)

    def test_memory_management (self):
        # Currently only the get_spin_evecs function
        with self.assertRaises (MemoryError):
            get_spin_evecs (10, 5, 5, 1, max_memory=1)

    def test_many_determinants (self):
        # Prove that there is no integer overrun for 26 singly-occupied orbitals
        umat = get_spin_evecs (26, 13, 13, 27)
        ovlp = np.dot (umat.T, umat)
        ovlperr = linalg.norm (ovlp - np.eye (ovlp.shape[0]))
        self.assertLess (ovlperr, 1e-8)

if __name__=="__main__":
    print ("Full tests for csfstring")
    unittest.main () 

