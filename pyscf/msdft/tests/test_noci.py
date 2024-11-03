import numpy as np
from pyscf import gto, scf, lib
from pyscf.msdft import noci

def test_hf_det_ovlp():
    mol = gto.M(atom='''
O   0.   0.      0.
H   0.   -1.51   1.17
H   0.   1.51    1.17
''', basis='6-31g')
    ms_ks = noci.NOCI(mol)
    mf0 = mol.UKS(xc='b3lyp').run(conv_tol=1e-14)
    mf1 = mf0.copy()
    occ = mf0.mo_occ.copy()
    occ[0][4] = 0
    occ[0][6] = 1
    mf1 = scf.addons.mom_occ(mf1, mf0.mo_coeff, occ).run(conv_tol=1e-14)
    h, s = noci.hf_det_ovlp(ms_ks, [mf0, mf1])
    ref = np.array([[-80.18565865644,  2.735e-8      ],
                    [ 2.735e-8      , -80.32146993896]])
    assert abs(abs(h) - abs(ref)).max() < 1e-8

def test_noci_e_tot():
    mol = gto.M(atom='''
O   0.   0.      0.
H   0.   -1.51   1.17
H   0.   1.51    1.17
''', basis='6-31g')
    mf = noci.NOCI(mol)
    mf.xc = 'pbe0'
    mf.s = [[4,5]]
    mf.d = [[4,5]]
    mf.run()
    assert abs(mf.e_tot[0] - -76.005766) < 1e-6
