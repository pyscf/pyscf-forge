#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#


''' LNO-RCCSD and LNO-CCSD(T) (for both molecule and pbc w/ Gamma-point BZ sampling)

    - Original publication by Kállay and co-workers:
        Rolik and Kállay, J. Chem. Phys. 135, 104111 (2011)

    - Publication for this implementation by Ye and Berkelbach:
        Ye and Berkelbach, J. Chem. Theory Comput. 2024, 20, 20, 8948–8959
'''


import sys
import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib

from pyscf.lno import LNO

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum


r''' TODO's
'''

''' Beginning of modification of PySCF's (DF)CCSD class

    The following functions are modified from pyscf.cc module

    In PySCF, 1e integrals (s1e, h1e, vhf) are calculated whenever a CCSD object is
    initialized. In LNOCCSD, this means that the same set of 1e integrals are evaluated
    for every fragment. For PBC calculations, evaluating 1e integrals (especially h1e
    and vhf) can be very slow in PySCF's current implementation.

    The following modification forces the CCSD class to take precomputed 1e integrals
    and thus can lead to significant amount of time saving in PBC LNOCCSD calculations.
'''
from pyscf.cc import ccsd, dfccsd, rccsd
def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib
    from pyscf.soscf import newton_ah
    from pyscf import scf

    log = logger.new_logger(mf)

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    if getattr(mf, 'with_df', None):
        ''' auto-choose if using DFCCSD (storing Lvv) or CCSD (storing vvvv) by memory
        '''
        naux = mf.with_df.get_naoaux()
        if mo_occ is None: mo_occ = mf.mo_occ
        maskocc = mo_occ > 1e-10
        frozen, maskact = get_maskact(frozen, len(mo_occ))
        nvir = np.count_nonzero(~maskocc & maskact)
        nvir_pair = nvir*(nvir+1)//2
        mem_avail = mf.max_memory - lib.current_memory()[0]
        mem_need = nvir_pair**2*8/1024**2.
        log.debug1('naux= %d  nvir_pair= %d  mem_avail= %.1f  mem_vvvv= %.1f',
                   naux, nvir_pair, mem_avail, mem_need)

        if np.iscomplexobj(mf.mo_coeff):
            if naux > nvir_pair or mem_need < mem_avail * 0.7:
                log.debug1('Using complex CCSD')
                return MODIFIED_CCSD_complex(mf, frozen, mo_coeff, mo_occ)
            else:
                log.debug1('Using complex DFCCSD')
                raise NotImplementedError('LNO-DFCCSD not implemented for complex orbitals.')
                return MODIFIED_DFCCSD_complex(mf, frozen, mo_coeff, mo_occ)
        else:
            if naux > nvir_pair or mem_need < mem_avail * 0.7:
                log.debug1('Using CCSD')
                return MODIFIED_CCSD(mf, frozen, mo_coeff, mo_occ)
            else:
                log.debug1('Using DFCCSD')
                return MODIFIED_DFCCSD(mf, frozen, mo_coeff, mo_occ)
    else:
        raise NotImplementedError('LNO-CCSD not implemented for 4c eris. Use DF-SCF instead.')

def is_unitary_related(c1, c2, s=None, thresh=1e-8):
    if c1.shape != c2.shape:
        return False
    if s is None:
        u = fdot(c1.T.conj(), c2)
    else:
        u = np.linalg.multi_dot((c1.T.conj(), s, c2))
    return abs(fdot(u.T.conj(), u) - np.eye(u.shape[1])).max() < thresh

def get_e_hf(self, mo_coeff=None):
    ''' Fragment CC does not need HF energy. We here just return e_tot from SCF to avoid
        any recomputation of integrals.
    '''
    return self._scf.e_tot

class MODIFIED_CCSD(ccsd.CCSD):
    get_e_hf = get_e_hf

    def ao2mo(self, mo_coeff=None):
        # Pseudo code how eris are implemented:
        # nocc = self.nocc
        # nmo = self.nmo
        # nvir = nmo - nocc
        # eris = _ChemistsERIs()
        # eri = ao2mo.incore.full(self._scf._eri, mo_coeff)
        # eri = ao2mo.restore(1, eri, nmo)
        # eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
        # eris.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
        # eris.ovvo = eri[nocc:,:nocc,nocc:,:nocc].copy()
        # eris.ovov = eri[nocc:,:nocc,:nocc,nocc:].copy()
        # eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
        # ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
        # eris.ovvv = lib.pack_tril(ovvv.reshape(-1,nvir,nvir))
        # eris.vvvv = ao2mo.restore(4, eri[nocc:,nocc:,nocc:,nocc:], nvir)
        # eris.fock = np.diag(self._scf.mo_energy)
        # return eris

        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            return _make_df_eris_outcore(self, mo_coeff)

        else:
            raise NotImplementedError   # LNO without DF should never happen

class _ChemistsERIs(ccsd._ChemistsERIs):
    def _common_init_(self, mycc, mo_coeff=None):
        from pyscf.mp.mp2 import _mo_without_core

        mymf = mycc._scf

        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)

# Note: Recomputed fock matrix and HF energy since SCF may not be fully converged.
        ''' This block is modified to take precomputed 1e integrals
        '''
        s1e = getattr(mycc, '_s1e', None)
        h1e = getattr(mycc, '_h1e', None)
        vhf = getattr(mycc, '_vhf', None)
        dm = mymf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        if vhf is None: vhf = self.get_vhf(mymf, dm, h1e=h1e, s1e=s1e)
        fockao = mymf.get_fock(vhf=vhf, dm=dm, h1e=h1e, s1e=s1e)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.e_hf = mymf.energy_tot(dm=dm, vhf=vhf, h1e=h1e)
        nocc = self.nocc = mycc.nocc
        self.mol = mycc.mol

        # Note self.mo_energy can be different to fock.diagonal().
        # self.mo_energy is used in the initial guess function (to generate
        # MP2 amplitudes) and CCSD update_amps preconditioner.
        # fock.diagonal() should only be used to compute the expectation value
        # of Slater determinants.
        self.mo_energy = self.fock.diagonal().real
        # vhf is assumed to be computed with exxdiv=None and mo_energy is not
        # exxdiv-corrected. We add the correction back for MP2 energy if
        # mymf.exxdiv is 'ewald'.
        # FIXME: Should we correct it for other exxdiv options (e.g., 'vcut_sph')?
        if hasattr(mymf, 'exxdiv') and mymf.exxdiv == 'ewald':  # PBC HF object
            from pyscf.pbc.cc.ccsd import _adjust_occ
            from pyscf.pbc import tools
            madelung = tools.madelung(mymf.cell, mymf.kpt)
            self.mo_energy = _adjust_occ(self.mo_energy, self.nocc, -madelung)
        mo_e = self.mo_energy
        try:
            gap = abs(mo_e[:nocc,None] - mo_e[None,nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for CCSD.\n'
                            'CCSD may be difficult to converge. Increasing '
                            'CCSD Attribute level_shift may improve '
                            'convergence.', gap)
        except ValueError:  # gap.size == 0
            pass
        return self
    def get_vhf(self, mymf, dm, h1e=None, s1e=None):
        ''' Build vhf from input dm.

        NOTE 1:
            If the input dm is the same as the SCF dm, vhf is built directly from the SCF
            MO and MO energy; otherwise, scf.get_vhf is called.
        NOTE 2:
            For PBC, exxdiv = None will be used for building vhf.
        '''
        dm0 = mymf.make_rdm1()
        errdm = abs(dm0-dm).max()
        if errdm < 1e-6:
            if h1e is None: h1e = mymf.get_hcore()
            vhf = fock_from_mo(mymf, s1e=s1e, force_exxdiv_none=True) - h1e
        else:
            if hasattr(mymf, 'exxdiv'):    # PBC CC requires exxdiv=None
                with lib.temporary_env(mymf, exxdiv=None):
                    vhf = mymf.get_veff(mymf.mol, dm)
            else:
                vhf = mymf.get_veff(mymf.mol, dm)
        return vhf

def _make_eris_incore(mycc, mo_coeff=None):
    from pyscf import ao2mo

    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)

    if eri1.ndim == 4:
        eri1 = ao2mo.restore(4, eri1, nmo)

    nvir_pair = nvir * (nvir+1) // 2
    eris.oooo = np.empty((nocc,nocc,nocc,nocc))
    eris.ovoo = np.empty((nocc,nvir,nocc,nocc))
    eris.ovvo = np.empty((nocc,nvir,nvir,nocc))
    eris.ovov = np.empty((nocc,nvir,nocc,nvir))
    eris.ovvv = np.empty((nocc,nvir,nvir_pair))
    eris.vvvv = np.empty((nvir_pair,nvir_pair))

    ij = 0
    outbuf = np.empty((nmo,nmo,nmo))
    oovv = np.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        for j in range(i+1):
            eris.oooo[i,j] = eris.oooo[j,i] = buf[j,:nocc,:nocc]
            oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
        ij += i + 1
    eris.oovv = oovv
    oovv = None

    ij1 = 0
    for i in range(nocc,nmo):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        eris.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.ovvo[:,i-nocc] = buf[:nocc,nocc:,:nocc]
        eris.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
        eris.ovvv[:,i-nocc] = lib.pack_tril(buf[:nocc,nocc:,nocc:])
        dij = i - nocc + 1
        lib.pack_tril(buf[nocc:i+1,nocc:,nocc:],
                      out=eris.vvvv[ij1:ij1+dij])
        ij += i + 1
        ij1 += dij
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris
def _make_df_eris_outcore(mycc, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mo_coeff = np.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2

    naux = mycc._scf.with_df.get_naoaux()
    Loo = np.empty((naux,nocc,nocc))
    Lov = np.empty((naux,nocc,nvir))
    Lvo = np.empty((naux,nvir,nocc))
    Lvv = np.empty((naux,nvir_pair))
    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    p1 = 0
    for eri1 in mycc._scf.with_df.loop():
        Lpq = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo,nmo)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvo[p0:p1] = Lpq[:,nocc:,:nocc]
        Lvv[p0:p1] = lib.pack_tril(Lpq[:,nocc:,nocc:].reshape(-1,nvir,nvir))
    Loo = Loo.reshape(naux,nocc*nocc)
    Lov = Lov.reshape(naux,nocc*nvir)
    Lvo = Lvo.reshape(naux,nocc*nvir)

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8')
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir_pair,nvir_pair), 'f8')
    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    eris.oovv[:] = lib.unpack_tril(lib.ddot(Loo.T, Lvv)).reshape(nocc,nocc,nvir,nvir)
    eris.ovvo[:] = lib.ddot(Lov.T, Lvo).reshape(nocc,nvir,nvir,nocc)
    eris.ovov[:] = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovvv[:] = lib.ddot(Lov.T, Lvv).reshape(nocc,nvir,nvir_pair)
    eris.vvvv[:] = lib.ddot(Lvv.T, Lvv)
    log.timer('CCSD integral transformation', *cput0)
    return eris

class _ChemistsERIs_complex(_ChemistsERIs):
    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        if slices:
            return self.ovvv[slices]
        else:
            return self.ovvv

class MODIFIED_CCSD_complex(rccsd.RCCSD):
    get_e_hf = get_e_hf

    def ao2mo(self, mo_coeff=None):
        from pyscf.pbc import tools
        from pyscf.pbc import mp
        from pyscf.pbc.cc.ccsd import _adjust_occ
        ao2mofn = mp.mp2._gen_ao2mofn(self._scf)
        # _scf.exxdiv affects eris.fock. HF exchange correction should be
        # excluded from the Fock matrix.
        with lib.temporary_env(self._scf, exxdiv=None):
            eris = _make_eris_incore_complex(self, mo_coeff, ao2mofn=ao2mofn)

        # eris.mo_energy so far is just the diagonal part of the Fock matrix
        # without the exxdiv treatment. Here to add the exchange correction to
        # get better orbital energies. It is important for the low-dimension
        # systems since their occupied and the virtual orbital energies may
        # overlap which may lead to numerical issue in the CCSD iterations.
        #if mo_coeff is self._scf.mo_coeff:
        #    eris.mo_energy = self._scf.mo_energy[self.get_frozen_mask()]
        #else:

        # Add the HFX correction of Ewald probe charge method.
        # FIXME: Whether to add this correction for other exxdiv treatments?
        # Without the correction, MP2 energy may be largely off the
        # correct value.
        madelung = tools.madelung(self._scf.cell, self._scf.kpt)
        eris.mo_energy = _adjust_occ(eris.mo_energy, eris.nocc, -madelung)
        return eris

def _make_eris_incore_complex(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs_complex()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    if callable(ao2mofn):
        eri1 = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        from pyscf import ao2mo
        eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris


class MODIFIED_DFCCSD(dfccsd.RCCSD):
    get_e_hf = get_e_hf

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff)

class _DFChemistsERIs(_ChemistsERIs):
    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert(not direct)
        return dfccsd._contract_vvvv_t2(mycc, self.mol, self.vvL, t2, out, verbose)
def _make_df_eris(cc, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo

    eris = _DFChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2
    with_df = cc.with_df
    naux = eris.naux = with_df.get_naoaux()

    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovov = eris.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvo = eris.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.oovv = eris.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    # nrow ~ 4e9/8/blockdim to ensure hdf5 chunk < 4GB
    chunks = (min(nvir_pair,int(4e8/with_df.blockdim)), min(naux,with_df.blockdim))
    eris.vvL = eris.feri.create_dataset('vvL', (nvir_pair,naux), 'f8', chunks=chunks)

    Loo = np.empty((naux,nocc,nocc))
    Lov = np.empty((naux,nocc,nvir))
    mo = np.asarray(eris.mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0,nmo,nmo)
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])
        eris.vvL[:,p0:p1] = Lvv.T
    Lpq = Lvv = None
    Loo = Loo.reshape(naux,nocc**2)
    #Lvo = Lov.transpose(0,2,1).reshape(naux,nvir*nocc)
    Lov = Lov.reshape(naux,nocc*nvir)
    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov[:] = ovov
    eris.ovvo[:] = ovov.transpose(0,1,3,2)
    ovov = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now)
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-nocc**2*nvir_pair)/(nocc**2+naux)))
    oovv_tril = np.empty((nocc*nocc,nvir_pair))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        oovv_tril[:,p0:p1] = lib.ddot(Loo.T, _cp(eris.vvL[p0:p1]).T)
    eris.oovv[:] = lib.unpack_tril(oovv_tril).reshape(nocc,nocc,nvir,nvir)
    oovv_tril = Loo = None

    Lov = Lov.reshape(naux,nocc,nvir)
    vblk = max(nocc, int((max_memory*.15e6/8)/(nocc*nvir_pair)))
    vvblk = int(min(nvir_pair, 4e8/nocc, max(4, (max_memory*.8e6/8)/(vblk*nocc+naux))))
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8',
                                         chunks=(nocc,1,vvblk))
    for q0, q1 in lib.prange(0, nvir_pair, vvblk):
        vvL = _cp(eris.vvL[q0:q1])
        for p0, p1 in lib.prange(0, nvir, vblk):
            tmpLov = _cp(Lov[:,:,p0:p1]).reshape(naux,-1)
            eris.ovvv[:,p0:p1,q0:q1] = lib.ddot(tmpLov.T, vvL.T).reshape(nocc,p1-p0,q1-q0)
        vvL = None
    return eris

def _cp(a):
    return np.array(a, copy=False, order='C')

class MODIFIED_DFCCSD_complex:
    pass
''' End of modification of PySCF's CCSD class
'''

''' impurity solver for LNO-based CCSD/CCSD_T
'''
def impurity_solve(mcc, mo_coeff, uocc_loc, mo_occ, maskact, eris,
                   ccsd_t=False, log=None, verbose_imp=None,
                   max_las_size_ccsd=1000, max_las_size_ccsd_t=1000):
    r''' Solve impurity problem and calculate local correlation energy.

    Args:
        mo_coeff (np.ndarray):
            MOs where the impurity problem is solved.
        uocc_loc (np.ndarray):
            <i|I> where i is semi-canonical occ LNOs and I is LO.
        ccsd_t (bool):
            If True, CCSD(T) energy is calculated and returned as the third
            item (0 is returned otherwise).
        frozen (int or list; optional):
            Same syntax as `frozen` in MP2, CCSD, etc.

    Return:
        e_loc_corr_pt2, e_loc_corr_ccsd, e_loc_corr_ccsd_t:
            Local correlation energy at MP2, CCSD, and CCSD(T) level. Note that
            the CCSD(T) energy is 0 unless 'ccsd_t' is set to True.
    '''
    log = logger.new_logger(mcc if log is None else log)
    cput1 = (logger.process_clock(), logger.perf_counter())

    maskocc = mo_occ>1e-10
    nmo = mo_occ.size

    orbfrzocc = mo_coeff[:,~maskact &  maskocc]
    orbactocc = mo_coeff[:, maskact &  maskocc]
    orbactvir = mo_coeff[:, maskact & ~maskocc]
    orbfrzvir = mo_coeff[:,~maskact & ~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [orb.shape[1]
                                          for orb in [orbfrzocc,orbactocc,
                                                      orbactvir,orbfrzvir]]
    nlo = uocc_loc.shape[1]
    nactmo = nactocc + nactvir
    log.debug('    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir',
              nlo, nactmo, nmo, nactocc, nactvir)

    if nactocc == 0 or nactvir == 0:
        elcorr_pt2 = elcorr_cc = lib.tag_array(0., spin_comp=np.array((0., 0.)))
        elcorr_cc_t = 0.
    else:

        if nactmo > max_las_size_ccsd:
            log.warn('Number of active space orbitals (%d) exceed '
                     '`_max_las_size_ccsd` (%d). Impurity CCSD calculations '
                     'will NOT be performed.', nactmo, max_las_size_ccsd)
            elcorr_pt2 = elcorr_cc = lib.tag_array(0., spin_comp=np.array((0.,0.)))
            elcorr_cc_t = 0.
        else:
            # solve impurity problem
            imp_eris = mcc.ao2mo()
            if isinstance(imp_eris.ovov, np.ndarray):
                ovov = imp_eris.ovov
            else:
                ovov = imp_eris.ovov[()]
            oovv = ovov.reshape(nactocc,nactvir,nactocc,nactvir).transpose(0,2,1,3)
            ovov = None
            cput1 = log.timer_debug1('imp sol - eri    ', *cput1)

            # MP2 fragment energy
            t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
            cput1 = log.timer_debug1('imp sol - mp2 amp', *cput1)
            elcorr_pt2 = get_fragment_energy(oovv, t2, uocc_loc).real
            cput1 = log.timer_debug1('imp sol - mp2 ene', *cput1)

            # CCSD fragment energy
            t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
            if not mcc.converged:
                log.warn('Impurity CCSD did not converge, please be careful of the results.')

            cput1 = log.timer_debug1('imp sol - cc  amp', *cput1)
            t2 += einsum('ia,jb->ijab',t1,t1)
            elcorr_cc = get_fragment_energy(oovv, t2, uocc_loc)
            cput1 = log.timer_debug1('imp sol - cc  ene', *cput1)

            # CCSD(T) fragment energy
            if ccsd_t:
                if nactmo > max_las_size_ccsd_t:
                    log.warn('Number of active space orbitals (%d) exceed '
                             '`_max_las_size_ccsd_t` (%d). Impurity CCSD(T) calculations '
                             'will NOT be performed.', nactmo, max_las_size_ccsd_t)
                    elcorr_cc_t = 0.
                else:
                    from pyscf.lno.lnoccsd_t import kernel as CCSD_T
                    t2 -= einsum('ia,jb->ijab',t1,t1)   # restore t2
                    elcorr_cc_t = CCSD_T(mcc, imp_eris, uocc_loc, t1=t1, t2=t2, verbose=verbose_imp)
                    cput1 = log.timer_debug1('imp sol - cc  (T)', *cput1)
            else:
                elcorr_cc_t = 0.

        t1 = t2 = oovv = imp_eris = mcc = None

    frag_msg = '  '.join([f'E_corr(MP2) = {elcorr_pt2:.15g}',
                          f'E_corr(CCSD) = {elcorr_cc:.15g}',
                          f'E_corr(CCSD(T)) = {elcorr_cc_t:.15g}'])

    return (elcorr_pt2, elcorr_cc, elcorr_cc_t), frag_msg

def get_maskact(frozen, nmo):
    # Convert frozen to 0 bc PySCF solvers do not support frozen=None or empty list
    if frozen is None:
        frozen = 0
    elif isinstance(frozen, (list,tuple,np.ndarray)) and len(frozen) == 0:
        frozen = 0

    if isinstance(frozen, (int,np.integer)):
        maskact = np.hstack([np.zeros(frozen,dtype=bool),
                             np.ones(nmo-frozen,dtype=bool)])
    elif isinstance(frozen, (list,tuple,np.ndarray)):
        maskact = np.array([i not in frozen for i in range(nmo)])
    else:
        raise RuntimeError

    return frozen, maskact

def get_fragment_energy(oovv, t2, uocc_loc):
    m = fdot(uocc_loc, uocc_loc.T.conj())
    # return einsum('ijab,kjab,ik->',t2,2*oovv-oovv.transpose(0,1,3,2),m)
    ed = einsum('ijab,kjab,ik->', t2, oovv, m) * 2
    ex = -einsum('ijab,kjba,ik->', t2, oovv, m)
    ed = ed.real
    ex = ex.real
    ess = ed*0.5 + ex
    eos = ed*0.5
    return lib.tag_array(ess+eos, spin_comp=np.array((ess, eos)))



class LNOCCSD(LNO):
    # Use the following _max_las_size arguments to avoid calculations that have no
    # hope of finishing. This may ease scanning thresholds.
    _max_las_size_ccsd = 1000
    _max_las_size_ccsd_t = 1000

    # The following arguments set default scs coefficients
    pss=0.333
    pos=1.2

    def __init__(self, mf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None):

        super().__init__(mf, lo_coeff, frag_lolist, lno_type=lno_type,
                         lno_thresh=lno_thresh, frozen=frozen)

        self.efrag_cc = None
        self.efrag_pt2 = None
        self.efrag_cc_t = None
        self.efrag_cc_spin_comp = None
        self.efrag_pt2_spin_comp = None
        self.ccsd_t = False

        # args for impurity solver
        self.kwargs_imp = None
        self.verbose_imp = 2    # ERROR and WARNING

        # args for precompute
        self._h1e = None
        self._vhf = None

    @property
    def h1e(self):
        if self._h1e is None:
            self._h1e = self._scf.get_hcore()
        return self._h1e

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info('_max_las_size_ccsd = %s', self._max_las_size_ccsd)
        log.info('_max_las_size_ccsd_t = %s', self._max_las_size_ccsd_t)
        return self

    def _precompute(self):
        log = logger.new_logger(self)

        mf = self._scf
        s1e = self.s1e
        h1e = self.h1e
        if self._vhf is None:
            log.warn('Input vhf is not found. Building vhf from SCF MO.')
            self._vhf = fock_from_mo(mf, s1e=s1e, force_exxdiv_none=True) - h1e
        elif hasattr(mf, 'exxdiv'):
            log.warn('Input vhf is detected while using PBC HF. Make sure '
                     'that the input vhf was computed with exxdiv=None, or '
                     'the MP2 and CCSD energy can be both wrong when compared '
                     'to k-point MP2 and CCSD results.')

    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris, frozen=None, log=None):
        if log is None: log = logger.new_logger(self)
        mo_occ = self.mo_occ
        frozen, maskact = get_maskact(frozen, mo_occ.size)
        mcc = CCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=self.verbose_imp)
        mcc._s1e = self._s1e
        mcc._h1e = self._h1e
        mcc._vhf = self._vhf

        if self.kwargs_imp is not None:
            mcc = mcc.set(**self.kwargs_imp)

        return impurity_solve(mcc, mo_coeff, uocc_loc, mo_occ, maskact, eris, log=log,
                              ccsd_t=self.ccsd_t, verbose_imp=self.verbose_imp,
                              max_las_size_ccsd=self._max_las_size_ccsd,
                              max_las_size_ccsd_t=self._max_las_size_ccsd_t)

    def _post_proc(self, frag_res, frag_wghtlist):
        ''' Post processing results returned by `impurity_solve` collected in `frag_res`.
        '''
        # TODO: add spin-component for CCSD(T)
        nfrag = len(frag_res)
        efrag_pt2 = np.zeros(nfrag)
        efrag_cc = np.zeros(nfrag)
        efrag_cc_t = np.zeros(nfrag)
        efrag_pt2_spin_comp = np.zeros((nfrag,2))
        efrag_cc_spin_comp = np.zeros((nfrag,2))

        for i in range(nfrag):
            ept2, ecc, ecc_t = frag_res[i]
            efrag_pt2[i] = float(ept2)
            efrag_cc[i] = float(ecc)
            efrag_cc_t[i] = float(ecc_t)
            efrag_pt2_spin_comp[i] = ept2.spin_comp
            efrag_cc_spin_comp[i] = ecc.spin_comp
        self.efrag_pt2  = efrag_pt2  * frag_wghtlist
        self.efrag_cc   = efrag_cc   * frag_wghtlist
        self.efrag_cc_t = efrag_cc_t * frag_wghtlist
        self.efrag_pt2_spin_comp  = efrag_pt2_spin_comp  * frag_wghtlist[:,None]
        self.efrag_cc_spin_comp   = efrag_cc_spin_comp   * frag_wghtlist[:,None]

    def _finalize(self):
        r''' Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                    'LNOMP2', self.e_tot_pt2, self.e_corr_pt2)
        logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                    'LNOCCSD', self.e_tot, self.e_corr)
        if self.ccsd_t:
            logger.note(self, 'E(%s) = %.15g  E_corr = %.15g',
                        'LNOCCSD_T', self.e_tot_ccsd_t, self.e_corr_ccsd_t)
        logger.note(self, 'Summary by spin components')
        logger.note(self, 'LNOMP2   Ess = %.15g  Eos = %.15g  Escs = %.15g',
                    self.e_corr_pt2_ss, self.e_corr_pt2_os, self.e_corr_pt2_scs)
        logger.note(self, 'LNOCCSD  Ess = %.15g  Eos = %.15g  Escs = %.15g',
                    self.e_corr_ccsd_ss, self.e_corr_ccsd_os, self.e_corr_ccsd_scs)
        return self

    @property
    def e_tot_scf(self):
        return self._scf.e_tot

    @property
    def e_corr(self):
        return self.e_corr_ccsd

    @property
    def e_tot(self):
        return self.e_corr + self.e_tot_scf

    @property
    def e_corr_ccsd(self):
        e_corr = np.sum(self.efrag_cc)
        return e_corr

    @property
    def e_corr_ccsd_ss(self):
        e_corr = np.sum(self.efrag_cc_spin_comp[:,0])
        return e_corr

    @property
    def e_corr_ccsd_os(self):
        e_corr = np.sum(self.efrag_cc_spin_comp[:,1])
        return e_corr

    @property
    def e_corr_ccsd_scs(self):
        e_corr = self.pss*self.e_corr_ccsd_ss + self.pos*self.e_corr_ccsd_os
        return e_corr

    @property
    def e_corr_pt2(self):
        e_corr = np.sum(self.efrag_pt2)
        return e_corr

    @property
    def e_corr_pt2_ss(self):
        e_corr = np.sum(self.efrag_pt2_spin_comp[:,0])
        return e_corr

    @property
    def e_corr_pt2_os(self):
        e_corr = np.sum(self.efrag_pt2_spin_comp[:,1])
        return e_corr

    @property
    def e_corr_pt2_scs(self):
        e_corr = self.pss*self.e_corr_pt2_ss + self.pos*self.e_corr_pt2_os
        return e_corr

    @property
    def e_corr_ccsd_t(self):
        e_corr = np.sum(self.efrag_cc_t)
        return e_corr + self.e_corr_ccsd

    @property
    def e_tot_ccsd(self):
        return self.e_corr_ccsd + self.e_tot_scf

    @property
    def e_tot_ccsd_t(self):
        return self.e_corr_ccsd_t + self.e_tot_scf

    @property
    def e_tot_pt2(self):
        return self.e_corr_pt2 + self.e_tot_scf

    def e_corr_pt2corrected(self, ept2):
        return self.e_corr - self.e_corr_pt2 + ept2

    def e_tot_pt2corrected(self, ept2):
        return self.e_tot_scf + self.e_corr_pt2corrected(ept2)

    def e_corr_ccsd_pt2corrected(self, ept2):
        return self.e_corr_ccsd - self.e_corr_pt2 + ept2

    def e_tot_ccsd_pt2corrected(self, ept2):
        return self.e_tot_scf + self.e_corr_ccsd_pt2corrected(ept2)

    def e_corr_ccsd_t_pt2corrected(self, ept2):
        return self.e_corr_ccsd_t - self.e_corr_pt2 + ept2

    def e_tot_ccsd_t_pt2corrected(self, ept2):
        return self.e_tot_scf + self.e_corr_ccsd_t_pt2corrected(ept2)


class LNOCCSD_T(LNOCCSD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ccsd_t = True


def fock_from_mo(mymf, s1e=None, force_exxdiv_none=True):
    if s1e is None: s1e = mymf.get_ovlp()
    mo0 = np.dot(s1e, mymf.mo_coeff)
    moe0 = mymf.mo_energy
    nocc0 = np.count_nonzero(mymf.mo_occ)
    if force_exxdiv_none:
        if hasattr(mymf, 'exxdiv') and mymf.exxdiv == 'ewald': # remove madelung
            from pyscf.pbc.cc.ccsd import _adjust_occ
            from pyscf.pbc import tools
            madelung = tools.madelung(mymf.cell, mymf.kpt)
            moe0 = _adjust_occ(moe0, nocc0, madelung)
    fock = np.dot(mo0*moe0, mo0.T.conj())
    return fock


if __name__ == '__main__':
    from pyscf import gto, scf, mp, cc, lo
    from pyscf.cc.ccsd_t import kernel as CCSD_T
    from pyscf.data.elements import chemcore

    log = logger.Logger(sys.stdout, 6)

    # S22-2: water dimer
    atom = '''
    O   -1.485163346097   -0.114724564047    0.000000000000
    H   -1.868415346097    0.762298435953    0.000000000000
    H   -0.533833346097    0.040507435953    0.000000000000
    O    1.416468653903    0.111264435953    0.000000000000
    H    1.746241653903   -0.373945564047   -0.758561000000
    H    1.746241653903   -0.373945564047    0.758561000000
    '''
    basis = 'cc-pvdz'

    mol = gto.M(atom=atom, basis=basis)
    mol.verbose = 4
    frozen = chemcore(mol)

    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    # canonical
    mmp = mp.MP2(mf, frozen=frozen)
    mmp.kernel()
    efull_mp2 = mmp.e_corr

    mcc = cc.CCSD(mf, frozen=frozen).set(verbose=5)
    eris = mcc.ao2mo()
    mcc.kernel(eris=eris)
    efull_ccsd = mcc.e_corr

    efull_t = CCSD_T(mcc, eris=eris, verbose=mcc.verbose)
    efull_ccsd_t = efull_ccsd + efull_t

    # LNO with PM localized orbitals
    # PM localization
    orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
    mlo = lo.PipekMezey(mol, orbocc)
    lo_coeff = mlo.kernel()
    while True: # always performing jacobi sweep to avoid trapping in local minimum/saddle point
        lo_coeff1 = mlo.stability_jacobi()[1]
        if lo_coeff1 is lo_coeff:
            break
        mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
        mlo.init_guess = None
        lo_coeff = mlo.kernel()

    # Fragment list: for PM, every orbital corresponds to a fragment
    frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

    # LNO-CCSD(T) calculation: here we scan over a list of thresholds
    mcc = LNOCCSD(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
    gamma = 10  # thresh_occ / thresh_vir
    threshs = np.asarray([1e-1,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7])
    elno_ccsd_uncorr = np.zeros_like(threshs)
    elno_ccsd_t_uncorr = np.zeros_like(threshs)
    elno_mp2 = np.zeros_like(threshs)
    for i,thresh in enumerate(threshs):
        mcc.lno_thresh = [thresh*gamma, thresh]
        mcc.kernel()
        elno_ccsd_uncorr[i] = mcc.e_corr_ccsd
        elno_ccsd_t_uncorr[i] = mcc.e_corr_ccsd_t
        elno_mp2[i] = mcc.e_corr_pt2
    elno_ccsd = elno_ccsd_uncorr - elno_mp2 + efull_mp2
    elno_ccsd_t = elno_ccsd_t_uncorr - elno_mp2 + efull_mp2

    log.info('')
    log.info('Reference CCSD E_corr = %.15g', efull_ccsd)
    for i,thresh in enumerate(threshs):
        e0 = elno_ccsd_uncorr[i]
        e1 = elno_ccsd[i]
        log.info('thresh = %.3e  E_corr(LNO-CCSD) = %.15g  E_corr(LNO-CCSD+∆PT2) = %.15g',
                 thresh, e0, e1)

    # log.info('')
    # log.info('Reference CCSD(T) E_corr = %.15g', efull_ccsd_t)
    # for i,thresh in enumerate(threshs):
    #     e0 = elno_ccsd_t_uncorr[i]
    #     e1 = elno_ccsd_t[i]
    #     log.info('thresh = %.3e  E_corr(LNO-CCSD(T)) = %.15g  E_corr(LNO-CCSD(T)+∆PT2) = %.15g',
    #              thresh, e0, e1)
