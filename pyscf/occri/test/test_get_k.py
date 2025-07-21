import numpy
import pyscf
from pyscf.pbc import gto
from pyscf.occri import OCCRI

TOL = 1.e-8

if __name__ == "__main__":
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
    refcell.basis = "gth-cc-dzvp"
    refcell.pseudo = "gth-pbe"
    refcell.ke_cutoff = 70
    refcell.build()

    ############ RHF ############
    en_fftdf = -43.9399339901445
    mf = pyscf.pbc.scf.RHF(refcell)
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("RHF occRI passed", en_diff)
    else:
        print("RHF occRI FAILED!!!", en_diff)


    ############ UHF ############
    mf = pyscf.pbc.scf.UHF(refcell)
    en_fftdf = -43.9399339901445
    mf = pyscf.pbc.scf.UHF(refcell)
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("UHF occRI passed", en_diff)
    else:
        print("UHF occRI FAILED!!!", en_diff)    


    ############ RKS ############
    en_fftdf = -45.0265010261793
    mf = pyscf.pbc.scf.RKS(refcell)
    mf.xc = 'pbe0'
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("RKS occRI passed", en_diff)
    else:
        print("RKS occRI FAILED!!!", en_diff)


    ############ UKS ############
    en_fftdf = -45.0265009589458
    mf = pyscf.pbc.scf.UKS(refcell)
    mf.xc = 'pbe0'
    mf.with_df = OCCRI(mf)
    en = mf.kernel()
    en_diff = abs(en - en_fftdf) / refcell.natm
    if en_diff < TOL:
        print("UKS occRI passed", en_diff)
    else:
        print("UKS occRI FAILED!!!", en_diff)            