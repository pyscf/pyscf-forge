import unittest
import numpy as np
from scipy import special
from pyscf.csf_fci.csfstring import *

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

