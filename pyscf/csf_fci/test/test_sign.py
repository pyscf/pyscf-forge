import unittest
import numpy as np
from scipy import linalg
from pyscf.csf_fci import spin_op
from pyscf.csf_fci.csfstring import CSFTransformer
import itertools

class KnownValues(unittest.TestCase):

    def test_sign (self):
        rng = np.random.default_rng ()
        for smult, ndocc, nvirt in itertools.product (range (1,6), range(3), range(3)):
            if smult==1 and ndocc==0: continue
            nelec = ((smult-1) + ndocc, ndocc)
            norb = (smult-1) + ndocc + nvirt
            trans0 = CSFTransformer (norb, nelec[0], nelec[1], smult)
            ci_csf = 2 * rng.random (trans0.ncsf) - 1
            ci_csf /= linalg.norm (ci_csf)
            ci0 = trans0.vec_csf2det (ci_csf)
            for spin in range (1-smult, smult-1, 2):
                nelec = sum (nelec)
                nelec = ((nelec + spin) // 2, (nelec - spin) // 2)
                trans1 = CSFTransformer (norb, nelec[0], nelec[1], smult)
                ci1 = spin_op.mup (trans1.vec_csf2det (ci_csf).reshape (trans1.ndeta,trans1.ndetb),
                                   norb,
                                   nelec,
                                   smult)
                with self.subTest (norb=norb, nelec=sum(nelec), smult=smult, spin=spin):
                    self.assertAlmostEqual (ci1.ravel ().dot (ci0.ravel ()), 1.0, 8)
                
if __name__=="__main__":
    print ("Full tests for csfstring sign")
    unittest.main () 

