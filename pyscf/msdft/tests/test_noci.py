from functools import reduce
import numpy
import numpy as np
import scipy.linalg
from pyscf import gto, scf, lib, fci, ao2mo
from pyscf.msdft import noci

def test_hf_det_ovlp():
    mol = gto.M(atom='''
O   0.   0.      0.
H   0.   -.757   .587
H   0.   .757    .587
H   0.5  0.1     -0.2
''', basis='6-31g', spin=1)
    ms_ks = noci.NOCI(mol)
    # Reduce iterations to prevent numerical instablity
    mf0 = mol.UKS(xc='b3lyp').run(max_cycle=1)
    mf1 = mf0.copy()
    occ = mf0.mo_occ.copy()
    occ[0][mf0.nelec[0]-1] = 0
    occ[0][mf0.nelec[0]+1] = 1
    mf1 = scf.addons.mom_occ(mf1, mf0.mo_coeff, occ).run(max_cycle=1, mo_coeff=None)
    h, s = noci.hf_det_ovlp(ms_ks, [mf0, mf1])
    ref = np.array([[-9.35176786e+01, -6.82503177e-02],
                    [-6.82503177e-02, -9.33368874e+01]])
    assert abs(h/ref - 1.).max() < 1e-7

def test_noci_e_tot():
    mol = gto.M(atom='''
N   0.   0.      0.
H   0.   -1.51   1.17
H   0.   1.51    1.17
H   1.5  0.1     -0.2
''', basis='6-31g')
    mf = noci.NOCI(mol)
    mf.xc = 'pbe0'
    mf.s = [[4,5], [4,6]]
    mf.d = [[4,5]]
    mf.sm_t = False
    mf.run()
    assert abs(mf.e_tot[0] - -56.161179917474) < 1e-8
    mf.sm_t = True
    mf.run()
    assert abs(mf.e_tot[0] - -56.161253460503) < 1e-8

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
    assert abs(mf.e_tot[0] - -76.0190855601) < 1e-7

def det_ovlp(mo1, mo2, occ1, occ2, ovlp):
    if numpy.sum(occ1) !=numpy.sum(occ2):
        raise RuntimeError('Electron numbers are not equal. Electronic coupling does not exist.')
    c1_a = mo1[0][:, occ1[0]>0]
    c1_b = mo1[1][:, occ1[1]>0]
    c2_a = mo2[0][:, occ2[0]>0]
    c2_b = mo2[1][:, occ2[1]>0]
    o_a = numpy.asarray(reduce(numpy.dot, (c1_a.conj().T, ovlp, c2_a)))
    o_b = numpy.asarray(reduce(numpy.dot, (c1_b.conj().T, ovlp, c2_b)))
    u_a, s_a, vt_a = scipy.linalg.svd(o_a)
    u_b, s_b, vt_b = scipy.linalg.svd(o_b)
    s_a = numpy.where(abs(s_a) > 1.0e-11, s_a, 1.0e-11)
    s_b = numpy.where(abs(s_b) > 1.0e-11, s_b, 1.0e-11)
    OV = numpy.linalg.det(u_a)*numpy.linalg.det(u_b) \
       * numpy.prod(s_a)*numpy.prod(s_b) \
       * numpy.linalg.det(vt_a)*numpy.linalg.det(vt_b)
    x_a = reduce(numpy.dot, (u_a*numpy.reciprocal(s_a), vt_a))
    x_b = reduce(numpy.dot, (u_b*numpy.reciprocal(s_b), vt_b))
    return OV, numpy.array((x_a, x_b))

def scoup_dml(mol, mo0, mo1, occ0, occ1):
    mf = scf.UHF(mol)
    # Calculate overlap between two determiant <I|F>
    s, x = det_ovlp(mo0, mo1, occ0, occ1, mf.get_ovlp())
    # Construct density matrix
    dm_01 = mf.make_asym_dm(mo0, mo1, occ0, occ1, x)
    # One-electron part contrbution
    h1e = mf.get_hcore(mol)
    # Two-electron part contrbution. D_{IF} is asymmetric
    #vhf_01 = get_veff(mf, dm_01, hermi=0)
    vj, vk = mf.get_jk(mol, dm_01, hermi=0)
    vhf_01 = vj[0] + vj[1] - vk
    # New total energy
    e_01 = mf.energy_elec(dm_01, h1e, vhf_01)
    return s, s * e_01[0], dm_01

def test_scoup_vs_fci():
    numpy.random.seed(4)
    coords = numpy.random.rand(6, 3)
    mol = gto.M(atom=[('H', r) for r in coords], verbose=1)
    mf = mol.UHF().run()
    nmo = mf.mo_coeff[0].shape[1]
    nelec = (3,3)
    u = np.linalg.svd(np.random.rand(nmo,nmo))[0]
    mo = mf.mo_coeff[0], mf.mo_coeff[1].dot(u)

    eri_aa = ao2mo.kernel(mf._eri, mo[0])
    eri_bb = ao2mo.kernel(mf._eri, mo[1])
    eri_ab = ao2mo.kernel(mf._eri, (mo[0], mo[0], mo[1], mo[1]))
    eri = eri_aa, eri_ab, eri_bb
    h1e_a = mo[0].T.dot(mf.get_hcore()).dot(mo[0])
    h1e_b = mo[1].T.dot(mf.get_hcore()).dot(mo[1])
    h1e = h1e_a, h1e_b
    h2e = fci.direct_uhf.absorb_h1e(h1e, eri, nmo, nelec, .5)
    s1e_a = mf.mo_coeff[0].T.dot(mf.get_ovlp()).dot(mo[0])
    s1e_b = mf.mo_coeff[1].T.dot(mf.get_ovlp()).dot(mo[1])
    s = (s1e_a, s1e_b)

    linki = fci.direct_spin1._unpack(nmo, nelec, None)
    na = linki[0].shape[0]
    nb = linki[1].shape[0]
    ket = np.zeros((na, nb))
    ket[0,0] = 1.
    bra = ket.copy()

    ci1 = fci.direct_uhf.contract_2e(h2e, ket, nmo, nelec, linki)
    ref = fci.addons.overlap(bra, ci1, nmo, nelec, s)
    scoup_out = scoup_dml(mol, mf.mo_coeff, mo, mf.mo_occ, mf.mo_occ)[1]
    assert abs(ref - scoup_out) < 1e-12
