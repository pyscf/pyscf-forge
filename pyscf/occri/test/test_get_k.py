import numpy
import pyscf
import sys, os
from pyscf.pbc import gto
from pyscf.occri import OCCRI

if __name__ == "__main__":
    sys.stdout = open(os.devnull, "w")
    refcell = gto.Cell()
    refcell.atom = """ 
        C 0.000000 0.000000 1.780373
        C 0.890186 0.890186 2.670559
        C 0.000000 1.780373 0.000000
        C 0.890186 2.670559 0.890186
        C 1.780373 0.000000 0.000000
        C 2.670559 0.890186 0.890186
        C 1.780373 1.780373 1.780373
        C 2.670559 2.670559 2.670559
    """
    refcell.a = numpy.array(
        [
            [3.560745, 0.000000, 0.000000],
            [0.000000, 3.560745, 0.000000],
            [0.000000, 0.000000, 3.560745],
        ]
    )
    refcell.basis = "gth-dzvp-molopt-sr"
    refcell.pseudo = "gth-pbe"
    refcell.spin = 0
    refcell.verbose = 0
    refcell.ke_cutoff = 70
    refcell.exp_to_discard = 0.2
    refcell.build()

    ############ occRI-FFTDF ############
    mf = pyscf.pbc.scf.RKS(refcell)
    mf.init_guess = "1e"
    mf.with_df = OCCRI(
        mf,
    )
    en = mf.kernel()
    sys.stdout = sys.__stdout__
    en_diff = abs(en - -43.8779860184878) / refcell.natm
    if en_diff < 1.0e-10:
        print("single grid occRI passed", en_diff)
    else:
        print("single grid occRI FAILED!!!", en_diff)
