import sys
import numpy as np
from functools import reduce

from pyscf.pbc.df.df import _load3c
from pyscf.ao2mo import _ao2mo
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import gamma_point, KPT_DIFF_TOL
from pyscf.pbc.tools import k2gamma
from pyscf import lo
from pyscf import __config__

MINAO = getattr(__config__, 'lo_iao_minao', 'minao')


def k2s_scf(kmf, fock_imag_tol=1e-6):
    from pyscf.scf.hf import eig
    from pyscf.pbc import scf

    if not kmf.istype('KRHF'):
        raise TypeError('k2s_scf only works for KRHF')

    dfobj = getattr(kmf, "with_df", None)
    if dfobj is None:
        raise RuntimeError("k2s_scf requires kmf to be density-fitted (kmf.with_df is None)")

    from pyscf.pbc.df.df import GDF
    from pyscf.pbc.df.rsdf import RSDF

    cell = kmf.cell
    kpts = kmf.kpts
    Nk = len(kpts)
    assert( abs((Nk**(1./3.))**3.-Nk) < 0.1 )

    kmesh = k2gamma.kpts_to_kmesh(cell, kpts-kpts[0])
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh)

    kmo_coeff = kmf.mo_coeff
    kmo_energy = kmf.mo_energy
    ks1e = kmf.get_ovlp()
    kh1e = kmf.get_hcore()

    ksc = [np.dot(s1e,mo_coeff) for s1e,mo_coeff in zip(ks1e,kmo_coeff)]
    kfock = np.asarray([np.dot(sc*moe,sc.T.conj()) for sc,moe in zip(ksc,kmo_energy)])

    s1e = _k2s_aoint(ks1e, kpts, phase, 's1e')
    h1e = _k2s_aoint(kh1e, kpts, phase, 'h1e')
    fock = _k2s_aoint(kfock, kpts, phase, 'fock')

    mo_energy, mo_coeff = eig(fock, s1e)

    if isinstance(dfobj, RSDF):
        mf = scf.RHF(scell, kpt=kpts[0]).rs_density_fit(auxbasis=dfobj.auxbasis)
    elif isinstance(dfobj, GDF):
        mf = scf.RHF(scell, kpt=kpts[0]).density_fit(auxbasis=dfobj.auxbasis)
    else:
        # NOTE: please add more DF types if needed
        raise TypeError(
            "k2s_scf requires kmf.with_df to be GDF or RSDF; got "
            f"{dfobj.__class__.__module__}.{dfobj.__class__.__name__}"
        )

    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mf.get_occ()
    mf.e_tot = kmf.e_tot * Nk
    mf.converged = True
    mf.get_ovlp = lambda *args: s1e
    mf.get_hcore = lambda *args: h1e

    return mf


def get_kpts_trsymm(cell, kmesh):
    # Nk = np.prod(kmesh).astype(int)
    # assert( Nk//2+Nk%2 == len(kpts_ibz) )

    return cell.make_kpts(kmesh, time_reversal_symmetry=True)


def kscf_remove_trsymm(kmf_trsymm, **kwargs):
    ''' Return a KSCF object where the translational symmetry is removed
    '''
    from pyscf.pbc import scf

    cell = kmf_trsymm.cell
    kpts = kmf_trsymm.kpts
    assert( gamma_point(kpts.kpts[0]) )

    nkpts = kpts.nkpts
    nkpts_ibz = kpts.nkpts_ibz
    mo_coeff = [None] * nkpts
    mo_energy = [None] * nkpts
    mo_occ = [None] * nkpts
    for ki in range(nkpts_ibz):
        ids = np.where(kpts.bz2ibz==ki)[0]
        if ids.size == 1:
            mo_coeff[ids[0]] = kmf_trsymm.mo_coeff[ki]
            mo_energy[ids[0]] = kmf_trsymm.mo_energy[ki]
            mo_occ[ids[0]] = kmf_trsymm.mo_occ[ki]
        elif ids.size == 2:
            mo_coeff[ids[0]] = kmf_trsymm.mo_coeff[ki].conj()
            mo_coeff[ids[1]] = kmf_trsymm.mo_coeff[ki]
            mo_energy[ids[0]] = mo_energy[ids[1]] = kmf_trsymm.mo_energy[ki]
            mo_occ[ids[0]] = mo_occ[ids[1]] = kmf_trsymm.mo_occ[ki]
        else:
            raise RuntimeError

    kmf = scf.KRHF(cell, kpts=kpts.kpts).rs_density_fit()
    kmf.set(**kwargs)

    kmf.mo_coeff = mo_coeff
    kmf.mo_energy = mo_energy
    kmf.mo_occ = mo_occ

    return kmf


def s2k_mo_coeff(cell, kpts, mo_coeff):
    r''' U(R,k) * C(R\mu,i) -> C(k\mu,i)

    Args:
        cell (pyscf.pbc.gto.Cell object):
            Unit cell.
        kpts (np.ndarray):
            Nk k-points.
        mo_coeff (np.ndarray):
            MO coeff matrx in the supercell AO basis, i.e.,
                mo_coeff.shape[0] == Nk * cell.nao_nr()

    Returns:
        kmo_coeff (np.ndarray):
            Shape (Nk,nao,Nmo).
    '''
    Nk = len(kpts)
    nao = cell.nao_nr()
    Nao,Nmo = mo_coeff.shape
    assert(Nk*nao == Nao)

    kmesh = k2gamma.kpts_to_kmesh(cell, kpts-kpts[0])
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh)

    kmo_coeff = lib.einsum('Rk,Rpi->kpi', phase.conj(), mo_coeff.reshape(Nk,nao,Nmo))
    return kmo_coeff


def k2s_aoint(cell, kpts, kA, name='aoint', phase=None):
    r''' U(R,k) * C(k,\mu,\nu) * U(S,k).conj() -> C(R\mu,S\nu)
    '''
    if phase is None:
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts-kpts[0])
        scell, phase = k2gamma.get_phase(cell, kpts, kmesh)
    return _k2s_aoint(kA, kpts, phase)

def _k2s_aoint(kA, kpts, phase, name='aoint'):
    Ncell = phase.shape[0]
    nao = kA[0].shape[0]
    Nao = Ncell*nao

    sA = lib.einsum('Rk,kpq,Sk->RpSq', phase, np.asarray(kA), phase.conj()).reshape(Nao,Nao)

    if gamma_point(kpts[0]):
        sAi = abs(sA.imag).max()
        if sAi > 1e-4:
            log = lib.logger.Logger(sys.stdout, 6)
            log.warn('Discard large imag part in k2s %s (%s). This may lead to error.',
                     name, sAi)
        sA = np.asarray(sA.real, order='C')

    return sA


class K2SDF(lib.StreamObject):
    def __init__(self, with_df, time_reversal_symmetry=True):
        self.with_df = with_df
        self.cell = with_df.cell
        self.kpts = with_df.kpts
        self.qpts = self.kpts - self.kpts[0]

        self.kikj_by_q = get_kikj_by_q(self.cell, self.kpts, self.qpts)
        self.qconserv = get_qconserv(self.cell, self.qpts)

        nqpts = len(self.qpts)
        if gamma_point(self.kpts[0]) and time_reversal_symmetry:    # time reversal symmetry
            find = np.zeros(len(self.qpts), dtype=bool)
            ibz2bz = []
            qpts_ibz_weights = []
            for q1,q2 in enumerate(self.qconserv):
                if find[q1] or find[q2]: continue
                ibz2bz.append(min(q1,q2))
                qpts_ibz_weights.append(1. if q1==q2 else 2**0.5)
                find[q2] = find[q2] = True
            self.ibz2bz = np.asarray(ibz2bz, dtype=int)
            self.qpts_ibz = self.qpts[self.ibz2bz]
            self.qpts_ibz_weights = np.asarray(qpts_ibz_weights) / nqpts**0.5
        else:
            self.ibz2bz = np.arange(nqpts)
            self.qpts_ibz = self.qpts
            self.qpts_ibz_weights = np.ones(nqpts) / nqpts**0.5

        self.kmesh = k2gamma.kpts_to_kmesh(self.cell, self.qpts)
        self.scell, self.phase = k2gamma.get_phase(self.cell, self.kpts, self.kmesh)

        self._naux = None
        self.blockdim = with_df.blockdim

    @property
    def naux_by_q(self):
        return self.get_naoaux()
    @property
    def naux(self):
        return self.get_naoaux().max()
    @property
    def Naux(self):
        return self.naux*len(self.qpts)
    @property
    def Naux_ibz(self):
        return self.naux*len(self.qpts_ibz)
    def get_naoaux(self):
        if self._naux is None:
            with_df = self.with_df
            kpts = self.kpts
            nqpts = len(self.qpts)
            naux = np.zeros(nqpts, dtype=int)
            for q in range(nqpts):
                ki,kj = self.kikj_by_q[q][0]
                kpti_kptj = np.asarray((kpts[ki],kpts[kj]))
                with _load3c(with_df._cderi, with_df._dataname, kpti_kptj=kpti_kptj) as j3c:
                    naux[q] = j3c.shape[0]
            self._naux = naux
        return self._naux

    def get_auxslice(self, q):
        p0 = self._naux[:q].sum()
        return p0, p0+self._naux[q]

    def s2k_mo_coeff(self, mo_coeff):
        nao = self.cell.nao_nr()
        Nk = len(self.qpts)
        Nao,Nmo = mo_coeff.shape
        assert(Nao == nao*Nk)
        kmo_coeff = lib.einsum('Rk,Rpi->kpi', self.phase.conj(), mo_coeff.reshape(Nk,nao,Nmo))
        return kmo_coeff

    def loop(self, q, auxslice=None):
        if auxslice is None: auxslice = (0,self._naux[q])
        p0,p1 = auxslice
        with_df = self.with_df
        kpts = self.kpts
        for ki,kj in self.kikj_by_q[q]:
            kpti_kptj = np.asarray((kpts[ki],kpts[kj]))
            with _load3c(with_df._cderi, with_df._dataname, kpti_kptj=kpti_kptj) as j3c:
                yield (ki,kj), np.asarray(j3c[p0:p1], order='C')

    def loop_ao2mo(self, q, mo1, mo2, buf=None, real_and_imag=False, auxslice=None):
        ''' Loop over all Lpq[k1,k2] s.t. kpts[k] = -kpts[k1] + kpts[k2]
        '''
        kmo1 = self.s2k_mo_coeff(mo1)
        kmo2 = self.s2k_mo_coeff(mo2)
        tao = []
        ao_loc = None
        nao = self.cell.nao_nr()
        for (ki,kj),Lpq_ao in self.loop(q, auxslice=auxslice):
            mo = np.asarray(np.hstack((kmo1[ki], kmo2[kj])), order='F')
            nmo1 = kmo1[ki].shape[1]
            nmo2 = kmo2[kj].shape[1]
            ijslice = (0,nmo1,nmo1,nmo1+nmo2)
            if Lpq_ao[1].size != nao**2:  # aosym = 's2'
                Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
            Lpq = _ao2mo.r_e2(Lpq_ao, mo, ijslice, tao, ao_loc, out=buf)
            if real_and_imag:
                yield (ki,kj), np.asarray(Lpq.real), np.asarray(Lpq.imag)
            else:
                yield (ki,kj), Lpq
            Lpq_ao = Lpq = None

    def get_eri_dtype_dsize(self, *arrs):
        ''' Get ERI dtype/size given arrays involved (e.g., mo_coeff).

        Explanation:
            While the 3c DF integrals is complex due to using kpts, the resulting
            ERI can be real if (a) kpt mesh is Gamma-inclusive and (b) MOs are real.
        '''
        return _guess_dtype_dsize(self.kpts, *arrs)

def get_kikj_by_q(cell, kpts, qpts):
    ''' Find map such that dot(qpts[q] - kpts[ki] + kpts[kj], alat) = 2pi * n

    Returns:
        kikj_by_q (nested list):
            Usage:
                for q,qpt in enumerate(qpts):
                    for ki,kj in kikj_by_q[q]:
                        kpti,kptj = kpts[ki], kpts[kj]
    '''
    nkpts = len(kpts)
    nqpts = len(qpts)
    kptijs = -kpts[:,None,:]+kpts
    qkikjs = lib.direct_sum('ax+bcx->abcx', qpts, kptijs)
    x = cell.get_scaled_kpts(qkikjs.reshape(-1,3)).reshape(nqpts,nkpts**2,3)
    qs, kijs = np.where(abs(x - x.round()).sum(axis=-1) < KPT_DIFF_TOL)
    kijs_by_q = [kijs[qs==q] for q in range(nqpts)]
    kikjs_by_q = []
    for q in range(nqpts):
        kijs = kijs_by_q[q]
        k1s = kijs // nkpts
        k2s = kijs % nkpts
        kikjs = np.vstack((k1s,k2s)).T
        kikjs_by_q.append(kikjs)
    return kikjs_by_q

def get_qconserv(cell, qpts):
    ''' Find map such that dot(qpts[q1] + qpts[q2], alat) = 2pi * n

    Returns:
        qconserv (list):
            Usage:
                for q1 in range(nqpts):
                    q2 = qconserv[q1]
    '''
    nqpts = len(qpts)
    qiqjs = lib.direct_sum('ax+bx->abx', qpts, qpts)
    x = cell.get_scaled_kpts(qiqjs.reshape(-1,3)).reshape(nqpts,nqpts,3)
    q1s, q2s = np.where(abs(x-x.round()).sum(axis=-1) < KPT_DIFF_TOL)
    assert( len(np.unique(q1s)) == nqpts )
    assert( len(q1s) == nqpts )
    qconserv = q2s[np.argsort(q1s)]
    return qconserv

def zdotCNtoR(aR, aI, bR, bI, alpha=1, cR=None, beta=0):
    '''c = (a.conj()*b).real'''
    cR = lib.ddot(aR, bR, alpha, cR, beta)
    cR = lib.ddot(aI, bI, alpha, cR, 1   )
    return cR
def zdotNNtoR(aR, aI, bR, bI, alpha=1, cR=None, beta=0):
    '''c = (a.conj()*b).real'''
    cR = lib.ddot(aR, bR,  alpha, cR, beta)
    cR = lib.ddot(aI, bI, -alpha, cR, 1  )
    return cR

def _guess_dtype_dsize(kpts, *arrs):
    dtype = np.float64 if gamma_point(kpts[0]) else np.complex128
    dtype = np.result_type(dtype, *arrs)
    dsize = 8 if dtype == np.float64 else 16
    return dtype, dsize


def sort_orb_by_cell(scell, lo_coeff, Ncell, s=None, kpt=np.zeros(3), pop_method='meta-lowdin',
                     Q_tol=1e-3):
    r''' Reorder LOs by unit cells in a supercel.

    Args:
        scell (pyscf.pbc.gto.Cell object):
            Supercell.
        lo_coeff (np.ndarray):
            AO coefficient matrix of LOs in supercell.
        Ncell (int):
            Number of cell in supercell.
        s (np.ndarray):
            AO overlap matrix in supercell. If not given, `s` is calculated by
                s = scell.pbc_intor('int1e_ovlp', kpts=kpt)
            where `kpt` needs to be specified (vide infra).
        kpt (np.ndarray):
            k-point for calculating the AO overlap matrix. Default is Gamma point.
        pop_method (str):
            Population method for assigning LOs to cells. Default is 'meta-lowdin'.
        Q_tol (float):
            Tolerance for determining degenerate LOs based on the `Q` value defined as:
                Q[alpha,i] := ( \sum_{A} pop[alpha,i,A]^2 )^0.5
            where `pop[alpha,i,A]` denotes the population of a LO `phi_alpha` on atom
            `A` in cell `i`. `Q[alpha,i]` \in [0,1] serves as a 'fingerprint' of a LO.
            The higher `Q` is, the more localized the corresponding LO is, and vice versa.
            Finding equal values in the `Q` matrix allows us to find
                1. degenerate LOs in the same cell (e.g., the 4 sp^3 C-C LOs in diamond)
                2. LOs in different cells related by lattice translation.
            This argument `Q_tol` determines how close is deemed "equal".

    Returns:
        lo_coeff (np.ndarray):
            Same LO coefficient matrix but reordered by cell. Let Nlo = Ncell*nlo.
            Then lo_coeff[:,i*nlo:(i+1)*nlo] gives the LOs in i-th cell.
    '''
    from pyscf.lo.orth import orth_ao

    Natm = scell.natm
    natm = Natm//Ncell
    assert( natm*Ncell == Natm )
    Nlo = lo_coeff.shape[1]
    nlo = Nlo//Ncell
    assert( nlo*Ncell == Nlo )

    if s is None: s = scell.pbc_intor('int1e_ovlp', kpts=kpt)
    csc = reduce(np.dot, (lo_coeff.conj().T, s, orth_ao(scell, pop_method, 'ANO', s=s)))
    pop_by_atom = np.asarray([lib.einsum('ix,ix->i', csc[:,p0:p1], csc[:,p0:p1].conj())
                              for i, (b0, b1, p0, p1) in enumerate(scell.offset_nr_by_atom())])
    Q_by_cell = np.asarray([(pop_by_atom[i*natm:(i+1)*natm]**2.).sum(axis=0)**0.5
                            for i in range(Ncell)])

    order0 = np.argsort(Q_by_cell[0])[::-1]
    assigned = np.zeros(Nlo, dtype=bool)
    reorder = [[] for i in range(Ncell)]
    for i in order0:
        if assigned[i]:
            continue
        Q_ref = Q_by_cell[0,i]
        mask = np.repeat(~assigned, Ncell).reshape(-1,Ncell).T  # LOs not already assigned
        idxcell, idxlo = np.where((abs(Q_by_cell-Q_ref)<Q_tol) & (mask))
        for ic,il in zip(idxcell,idxlo):
            reorder[ic].append(il)
        assigned[idxlo] = True
    reorder = np.hstack(reorder).astype(int)
    lo_coeff = np.asarray(lo_coeff[:,reorder], order='C')

    return lo_coeff


def k2s_iao(cell, kocc_coeff, kpts, minao=MINAO, orth=False, s1e=None):
    ''' Compute supercell IAO from k-point orbitals

    Args:
        cell (pyscf.pbc.gto.Cell):
            Unit cell.
        kocc_coeff (np.ndarray):
            k-point occupied MO coefficients.
        kpts (np.ndarray):
            k-points.
        minao (str or basis):
            minao for IAO projection.
        orth (boolean):
            Whether to symmetrically orthogonalize the IAOs.

    Returns:
        iao_coeff (np.ndarray):
            Supercell IAO coefficients of shape (nkpts*nao, nkpts*niao) where nao and niao
            are the number of AOs and IAOs within a unit cell. The IAOs are sorted by cell.
    '''
    from pyscf.lib import logger
    log = logger.new_logger(cell, cell.verbose)

    nkpts = len(kpts)
    kiao_coeff = np.asarray(lo.iao.iao(cell, kocc_coeff, kpts=kpts, minao=minao))
    if orth:
        if s1e is None: s1e = cell.pbc_intor('int1e_ovlp', kpts=kpts)
        kiao_coeff = np.asarray([lo.orth.vec_lowdin(kiao_coeff[k], s1e[k]) for k in range(nkpts)])
    cell_iao = lo.iao.reference_mol(cell, minao)
    nao = cell.nao_nr()
    niao = cell_iao.nao_nr()

    kmesh = k2gamma.kpts_to_kmesh(cell, kpts-kpts[0])
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh)
    iao_coeff = lib.einsum('Rk,kpq,Sk->RpSq', phase, kiao_coeff,
                           phase.conj()).reshape(nkpts*nao,nkpts*niao)

    if gamma_point(kpts[0]):
        iao_coeff_imag = abs(iao_coeff.imag).max()
        if iao_coeff_imag > 1e-10:
            log.warn('Discard large imag part in k2s_iao: %6.2e.', iao_coeff_imag)
        iao_coeff = iao_coeff.real

    return iao_coeff
