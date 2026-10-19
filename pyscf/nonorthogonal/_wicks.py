#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

'''
Reference-pair intermediates of the extended nonorthogonal Wick theorem and
the ctypes binding to libnonorthwick.

For an ordered reference pair (x,w) and each spin, the occupied orbital
overlap is Lowdin paired by an SVD, the m zero-overlap pairs are separated
from the reduced overlap, and the fundamental contractions X^{(m_i)},
Y^{(m_i)} (m_i = 0, 1) are formed together with the one-column (F, V) and
two-column (J, II) operator intermediates.  All tensors of all pairs are
stored in one contiguous slab with the offset layout of nowick.h.

References:
    H. G. A. Burton, J. Chem. Phys. 154, 144109 (2021)
    H. G. A. Burton, J. Chem. Phys. 157, 204109 (2022)
'''

import ctypes
import numpy
from pyscf import lib

libwick = lib.load_library('libnonorthwick')

PAIRLEN = 96
MA, MB, NOCCA, NOCCB, NMO = 0, 1, 2, 3, 4
AA, BB, AB = 5, 32, 59
X, Y, FH, V, HCOL0, J = 0, 2, 4, 8, 16, 17
VAB, VBA, IIAB = 0, 8, 16

NSCAL = 24
SAA, SBB, SVAB0 = 0, 8, 16
SPHASE, STSP, SF0H, SV0 = 0, 1, 2, 4

OH_A, OH_B, OAA, OBB, OAB = 91, 92, 93, 94, 95

DHEAD = 5

# Symmetry-unique (m1, m2, m3, m4) branches of the same-spin J intermediate
J_BRANCHES = ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
              (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 0),
              (1, 0, 1, 1), (1, 1, 1, 1))


def _ao2mo(eri, a, b, c, d):
    '''sum_{mu nu lam sig} (mu nu|lam sig) a*_{mu p} b_{nu q} c*_{lam r} d_{sig s}'''
    n = eri.shape[0]
    np_, nq, nr, ns = a.shape[1], b.shape[1], c.shape[1], d.shape[1]
    t = eri.reshape(n**3, n).dot(d)                                  # mu nu lam s
    t = numpy.matmul(c.conj().T, t.reshape(n * n, n, ns))            # mu nu r s
    t = numpy.matmul(b.T, t.reshape(n, n, nr * ns))                  # mu q r s
    return a.conj().T.dot(t.reshape(n, -1)).reshape(np_, nq, nr, ns)


def _coulomb(eri, dm):
    n = dm.shape[0]
    return eri.reshape(n*n, n*n).dot(dm.ravel()).reshape(n, n)


def _exchange(eri, dm):
    '''K_{st} = sum_{mu nu} (s mu|nu t) M_{mu nu}'''
    return numpy.tensordot(eri, dm, axes=([1, 2], [0, 1]))


class _Pairing:
    '''Lowdin pairing of the occupied orbitals of one spin of (x,w)'''
    def __init__(self, s, cx, cw, nocc, tol):
        nmo = cx.shape[1]
        dtype = numpy.result_type(cx, cw)
        self.nocc = nocc
        self.nmo = nmo
        cxo = cx[:, :nocc]
        cwo = cw[:, :nocc]
        socc = cxo.conj().T.dot(s).dot(cwo)
        if nocc > 0:
            u, sv, vh = numpy.linalg.svd(socc)
            v = vh.conj().T
            phase = numpy.linalg.det(u) * numpy.linalg.det(v).conj()
        else:
            u = v = numpy.zeros((0, 0), dtype=dtype)
            sv = numpy.zeros(0)
            phase = 1.
        cxt = cxo.dot(u)
        cwt = cwo.dot(v)
        nz = abs(sv) > tol
        self.phase = phase
        self.tilde_s_prod = numpy.prod(sv[nz])
        self.m = int(nocc - nz.sum())
        self.singular_values = sv

        # M^{(0)} = W + P^{xw} + P^{ww}, M^{(1)} = P^{xw}
        cwz = cwt[:, ~nz]
        cxz = cxt[:, ~nz]
        m1 = cwz.dot(cxz.conj().T)
        m0 = (cwt[:, nz] / sv[nz]).dot(cxt[:, nz].conj().T)
        m0 = m0 + cwz.dot(cwz.conj().T) + m1
        self.mao = (m0, m1)

        # rows: V_x + O_w, columns: O_x + V_w
        rowc = numpy.hstack((cx[:, nocc:], cw[:, :nocc]))
        colc = numpy.hstack((cx[:, :nocc], cw[:, nocc:]))
        xs = []
        ys = []
        for mi, mm in enumerate(self.mao):
            xao = s.dot(mm).dot(s)
            yao = xao - s if mi == 0 else xao
            xs.append(rowc.conj().T.dot(xao).dot(colc))
            ys.append(rowc.conj().T.dot(yao).dot(colc))
        self.x = xs
        self.y = ys

        # Left (row) and right (column) contraction coefficients.  For m_i = 0
        # the direct orbital contribution is removed, giving Y^{(0)}.
        cxs = []
        xcs = []
        for mi, mm in enumerate(self.mao):
            mts = mm.conj().T.dot(s)
            ms = mm.dot(s)
            f = 1 - mi
            cxs.append(numpy.hstack((mts.dot(cx[:, nocc:]) - f * cx[:, nocc:],
                                     mts.dot(cw[:, :nocc]))))
            xcs.append(numpy.hstack((ms.dot(cx[:, :nocc]),
                                     ms.dot(cw[:, nocc:]) - f * cw[:, nocc:])))
        self.cx = cxs
        self.xc = xcs


def _same_spin(p, h, eri):
    '''One-body, and same-spin two-body intermediates of one spin'''
    out = {}
    m = p.m
    mao = p.mao
    cx, xc = p.cx, p.xc
    out['f0h'] = [numpy.einsum('ab,ba->', mm, h) for mm in mao]
    out['fh'] = [[cx[i].conj().T.dot(h).dot(xc[j]) for j in range(2)] for i in range(2)]
    jk = [_coulomb(eri, mm) - _exchange(eri, mm) for mm in mao]
    # V_0^{(m1,m2)} is symmetric in (m1,m2) and indexed by m1 + m2.  The
    # evaluator visits (0,1) and (1,0) separately, so the middle entry holds
    # one assignment only.
    v0 = [numpy.einsum('ab,ba->', jk[0], mao[0]), 0, 0]
    if m >= 1:
        v0[1] = numpy.einsum('ab,ba->', jk[0], mao[1])
    if m >= 2:
        v0[2] = numpy.einsum('ab,ba->', jk[1], mao[1])
    out['v0'] = v0
    # v[m1][m3][m2] = cx[m1]^+ (J-K)[m2] xc[m3]
    out['v'] = [[[cx[i].conj().T.dot(jk[k]).dot(xc[j]) for k in range(2)]
                 for j in range(2)] for i in range(2)]
    js = {}
    for slot, (mi, mj, mk, ml) in enumerate(J_BRANCHES):
        if mi + mj + mk + ml > m:
            continue
        blk = _ao2mo(eri, cx[mi], xc[mj], cx[mk], xc[ml])
        # exchange: row-side coefficients cx are conjugated, column-side xc not
        blk -= _ao2mo(eri, cx[mi], cx[mk].conj(), xc[mj].conj(),
                      xc[ml]).transpose(0, 2, 1, 3)
        js[slot] = blk
    out['j'] = js
    return out


def _diff_spin(pa, pb, eri):
    '''Different-spin two-body intermediates'''
    out = {}
    ja = [_coulomb(eri, mm) for mm in pa.mao]
    jb = [_coulomb(eri, mm) for mm in pb.mao]
    out['vab0'] = [[numpy.einsum('ab,ba->', ja[i], pb.mao[j]) for j in range(2)]
                   for i in range(2)]
    out['vab'] = [[[pa.cx[i].conj().T.dot(jb[j]).dot(pa.xc[k]) for k in range(2)]
                   for j in range(2)] for i in range(2)]
    out['vba'] = [[[pb.cx[j].conj().T.dot(ja[i]).dot(pb.xc[k]) for k in range(2)]
                   for i in range(2)] for j in range(2)]
    ii = {}
    for ma0 in range(2):
        for maz in range(2):
            for mb0 in range(2):
                for mby in range(2):
                    if ma0 + maz <= pa.m and mb0 + mby <= pb.m:
                        ii[ma0, maz, mb0, mby] = _ao2mo(eri, pa.cx[ma0], pa.xc[maz],
                                                        pb.cx[mb0], pb.xc[mby])
    out['ii'] = ii
    return out


def count_zero_pairs(s, cx, cw, nocc, tol):
    '''Number m of zero-overlap occupied orbital pairs'''
    if nocc == 0:
        return 0
    socc = cx[:, :nocc].conj().T.dot(s).dot(cw[:, :nocc])
    sv = numpy.linalg.svd(socc, compute_uv=False)
    return int((abs(sv) <= tol).sum())


def pair_layout(ma, mb, nmo, start):
    '''Offsets of one reference pair in the slab, pair-adaptive in (ma, mb).
    Returns (pairinfo, end).'''
    nn2 = nmo * nmo
    nn4 = nn2 * nn2
    info = numpy.full(PAIRLEN, -1, dtype=numpy.int64)
    i = start
    for blk, m in ((AA, ma), (BB, mb)):
        for k in range(FH + 4 + 8):     # x, y, fh, v
            info[blk+k] = i
            i += nn2
        for slot, br in enumerate(J_BRANCHES):
            if sum(br) <= m:
                info[blk+J+slot] = i
                i += nn4
    for k in range(16):                 # vab, vba
        info[AB+k] = i
        i += nn2
    if ma == 0 and mb == 0:
        info[AA+HCOL0] = i
        i += nn2
        info[BB+HCOL0] = i
        i += nn2
    for ma0 in range(2):
        for maz in range(2):
            for mb0 in range(2):
                for mby in range(2):
                    if ma0 + maz <= ma and mb0 + mby <= mb:
                        info[AB+IIAB+ma0*8+maz*4+mb0*2+mby] = i
                        i += nn4
    return info, i


def orthogonal_layout(info, nmo, start):
    '''Offsets of the same-parent MO integrals of a diagonal pair'''
    nn2 = nmo * nmo
    nn4 = nn2 * nn2
    i = start
    for k, size in ((OH_A, nn2), (OH_B, nn2), (OAA, nn4), (OBB, nn4), (OAB, nn4)):
        info[k] = i
        i += size
    return i


def write_orthogonal(slab, info, h, eri, ca, cb):
    '''MO integrals of one reference for the same-parent Slater-Condon rules'''
    nmo = ca.shape[1]
    nn2 = nmo * nmo
    nn4 = nn2 * nn2
    slab[info[OH_A]:info[OH_A]+nn2] = ca.conj().T.dot(h).dot(ca).ravel()
    slab[info[OH_B]:info[OH_B]+nn2] = cb.conj().T.dot(h).dot(cb).ravel()
    for k, c in ((OAA, ca), (OBB, cb)):
        g = _ao2mo(eri, c, c, c, c)
        slab[info[k]:info[k]+nn4] = (g - g.transpose(0, 3, 2, 1)).ravel()
    slab[info[OAB]:info[OAB]+nn4] = _ao2mo(eri, ca, ca, cb, cb).ravel()


def write_pair(slab, scal, info, pa, pb, sa, sb, ab):
    '''Write the intermediates of one ordered pair into slab and scal'''
    nmo = pa.nmo
    nn2 = nmo * nmo
    nn4 = nn2 * nn2

    def put2(off, a):
        slab[off:off+nn2] = a.ravel()

    def put2t(off, a):
        slab[off:off+nn2] = a.T.ravel()

    for blk, sblk, p, w in ((AA, SAA, pa, sa), (BB, SBB, pb, sb)):
        for mi in range(2):
            put2(info[blk+X+mi], p.x[mi])
            put2(info[blk+Y+mi], p.y[mi])
            for mj in range(2):
                put2t(info[blk+FH+mi*2+mj], w['fh'][mi][mj])
                for mk in range(2):
                    put2t(info[blk+V+mi*4+mj*2+mk], w['v'][mi][mj][mk])
        for slot, t in w['j'].items():
            off = info[blk+J+slot]
            # stored in [i,j,r,c] order: the fixed pair precedes the varying one
            slab[off:off+nn4] = t.transpose(2, 3, 0, 1).ravel()
        scal[sblk+SPHASE] = p.phase
        scal[sblk+STSP] = p.tilde_s_prod
        scal[sblk+SF0H:sblk+SF0H+2] = w['f0h']
        scal[sblk+SV0:sblk+SV0+3] = w['v0']
    for i in range(2):
        for j in range(2):
            for k in range(2):
                put2t(info[AB+VAB+i*4+j*2+k], ab['vab'][i][j][k])
                put2t(info[AB+VBA+i*4+j*2+k], ab['vba'][i][j][k])
            scal[SVAB0+i*2+j] = ab['vab0'][i][j]
    if info[AA+HCOL0] >= 0:
        hcol = (sa['fh'][0][0] + sa['v'][0][0][0] + ab['vab'][0][0][0])
        put2t(info[AA+HCOL0], hcol)
        hcol = (sb['fh'][0][0] + sb['v'][0][0][0] + ab['vba'][0][0][0])
        put2t(info[BB+HCOL0], hcol)
    for (ma0, maz, mb0, mby), t in ab['ii'].items():
        off = info[AB+IIAB+ma0*8+maz*4+mb0*2+mby]
        slab[off:off+nn4] = t.ravel()


def build(s, h, eri, refs, tol, dtype):
    '''Construct the slab of all ordered reference pairs.

    refs: list of (ca, cb, nocca, noccb), occupied orbitals first.
    Returns (pairinfo, slab, scal, nref).
    '''
    nref = len(refs)
    nmo = refs[0][0].shape[1]
    nocca = refs[0][2]
    noccb = refs[0][3]
    infos = []
    end = 0
    for x in range(nref):
        for w in range(nref):
            ma = count_zero_pairs(s, refs[x][0], refs[w][0], nocca, tol)
            mb = count_zero_pairs(s, refs[x][1], refs[w][1], noccb, tol)
            info, end = pair_layout(ma, mb, nmo, end)
            if x == w:
                end = orthogonal_layout(info, nmo, end)
            info[MA] = ma
            info[MB] = mb
            info[NOCCA] = nocca
            info[NOCCB] = noccb
            info[NMO] = nmo
            infos.append(info)
    pairinfo = numpy.asarray(infos)
    slab = numpy.zeros(end, dtype=dtype)
    scal = numpy.zeros((nref*nref, NSCAL), dtype=dtype)
    for x in range(nref):
        for w in range(nref):
            k = x * nref + w
            pa = _Pairing(s, refs[x][0], refs[w][0], nocca, tol)
            pb = _Pairing(s, refs[x][1], refs[w][1], noccb, tol)
            assert pa.m == pairinfo[k, MA] and pb.m == pairinfo[k, MB]
            sa = _same_spin(pa, h, eri)
            sb = _same_spin(pb, h, eri)
            ab = _diff_spin(pa, pb, eri)
            write_pair(slab, scal[k], pairinfo[k], pa, pb, sa, sb, ab)
            if x == w:
                write_orthogonal(slab, pairinfo[k], h, eri, refs[x][0], refs[x][1])
    return pairinfo, slab, scal, nref


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def overlap_pairs(env, dets, kmax, reqs):
    pairinfo, slab, scal, nref = env
    reqs = numpy.asarray(reqs, dtype=numpy.int32, order='C')
    nreq = reqs.shape[0]
    cplx = numpy.iscomplexobj(slab)
    out = numpy.empty(nreq, dtype=slab.dtype)
    libwick.NOWICKoverlap_pairs(
        _ptr(out), _ptr(reqs), ctypes.c_size_t(nreq), _ptr(dets), ctypes.c_int(kmax),
        _ptr(pairinfo), _ptr(slab), _ptr(scal), ctypes.c_int(nref), ctypes.c_int(cplx))
    return out


def hamiltonian_pairs(env, dets, kmax, reqs, enuc, general=False):
    pairinfo, slab, scal, nref = env
    reqs = numpy.asarray(reqs, dtype=numpy.int32, order='C')
    nreq = reqs.shape[0]
    cplx = numpy.iscomplexobj(slab)
    h = numpy.empty(nreq, dtype=slab.dtype)
    s = numpy.empty(nreq, dtype=slab.dtype)
    libwick.NOWICKhamiltonian_pairs(
        _ptr(h), _ptr(s), _ptr(reqs), ctypes.c_size_t(nreq), _ptr(dets), ctypes.c_int(kmax),
        _ptr(pairinfo), _ptr(slab), _ptr(scal), ctypes.c_int(nref),
        ctypes.c_double(enuc), ctypes.c_int(cplx), ctypes.c_int(general))
    return h, s


def hamiltonian_matrix(env, dets, kmax, enuc):
    pairinfo, slab, scal, nref = env
    ndet = dets.shape[0]
    cplx = numpy.iscomplexobj(slab)
    h = numpy.empty((ndet, ndet), dtype=slab.dtype)
    s = numpy.empty((ndet, ndet), dtype=slab.dtype)
    libwick.NOWICKhamiltonian_matrix(
        _ptr(h), _ptr(s), _ptr(dets), ctypes.c_int(ndet), ctypes.c_int(kmax),
        _ptr(pairinfo), _ptr(slab), _ptr(scal), ctypes.c_int(nref),
        ctypes.c_double(enuc), ctypes.c_int(cplx))
    return h, s
