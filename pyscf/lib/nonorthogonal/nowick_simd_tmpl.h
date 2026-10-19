/* Copyright 2026 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Packed m_a = m_b = 0 Hamiltonian/overlap kernels.  NLANE determinant
 * pairs sharing (L_a, L_b) are evaluated together: determinants, cofactors
 * and second minors are lane-parallel, intermediates are gathered per lane.
 *
 * Template parameters, defined by the including file:
 *   DT       scalar type
 *   PT       packet type of NLANE scalars
 *   FN(x)    function name
 *   PZERO, PONE, PADD, PSUB, PMUL, PSCALE(p, double), PBCAST(DT),
 *   PLOAD(DT *), PSTORE(DT *, PT)
 */

#define PFMA(acc, a, b)         PADD(acc, PMUL(a, b))
#define PFMS(acc, a, b)         PSUB(acc, PMUL(a, b))

ALWAYS_INLINE PT FN(pdet_cof)(PT *cof, const PT *a, const int n, const int want_cof)
{
        PT b[NOWICK_MAXL*NOWICK_MAXL];
        PT p[NOWICK_MAXL*NOWICK_MAXL];
        PT c, v;
        int i, j, k, l;
        if (n == 0) {
                return PONE;
        }
        if (n == 1) {
                if (want_cof) {
                        cof[0] = PONE;
                }
                return a[0];
        }
        if (n == 2) {
                if (want_cof) {
                        cof[0] = a[3];
                        cof[1] = PSUB(PZERO, a[2]);
                        cof[2] = PSUB(PZERO, a[1]);
                        cof[3] = a[0];
                }
                return PSUB(PMUL(a[0], a[3]), PMUL(a[1], a[2]));
        }
        c = PZERO;
        for (i = 0; i < n; i++) {
                c = PSUB(c, a[i*n+i]);
        }
        for (i = 0; i < n*n; i++) {
                b[i] = a[i];
        }
        for (i = 0; i < n; i++) {
                b[i*n+i] = PADD(b[i*n+i], c);
        }
        for (k = 2; k <= n; k++) {
                if (want_cof && k == n) {
                        for (i = 0; i < n; i++) {
                        for (j = 0; j < n; j++) {
                                cof[i*n+j] = (n & 1) ? b[j*n+i] : PSUB(PZERO, b[j*n+i]);
                        } }
                }
                if (k == n) {
                        c = PZERO;
                        for (i = 0; i < n; i++) {
                        for (l = 0; l < n; l++) {
                                c = PFMA(c, a[i*n+l], b[l*n+i]);
                        } }
                        c = PSCALE(c, -1. / n);
                        return (n & 1) ? PSUB(PZERO, c) : c;
                }
                for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                        v = PZERO;
                        for (l = 0; l < n; l++) {
                                v = PFMA(v, a[i*n+l], b[l*n+j]);
                        }
                        p[i*n+j] = v;
                } }
                c = PZERO;
                for (i = 0; i < n; i++) {
                        c = PADD(c, p[i*n+i]);
                }
                c = PSCALE(c, -1. / k);
                for (i = 0; i < n*n; i++) {
                        b[i] = p[i];
                }
                for (i = 0; i < n; i++) {
                        b[i*n+i] = PADD(b[i*n+i], c);
                }
        }
        return PZERO;
}

ALWAYS_INLINE PT FN(psecond_minor_det)(const PT *a, const int n, int r0, int r1, int c0, int c1)
{
        PT buf[NOWICK_MAXL*NOWICK_MAXL];
        int i, j, k = 0;
        for (i = 0; i < n; i++) {
                if (i == r0 || i == r1) continue;
                for (j = 0; j < n; j++) {
                        if (j == c0 || j == c1) continue;
                        buf[k++] = a[i*n+j];
                }
        }
        return FN(pdet_cof)(NULL, buf, n - 2, 0);
}

/* lane labels of one spin */
typedef struct {
        int rows[NLANE][NOWICK_MAXL];
        int cols[NLANE][NOWICK_MAXL];
} FN(Labels);

ALWAYS_INLINE void FN(pbuild_d)(PT *d, const DT *x, const DT *y, const FN(Labels) *lb,
                                const int l, size_t n)
{
        DT tmp[NLANE];
        int i, j, k;
        for (i = 0; i < l; i++) {
        for (j = 0; j < l; j++) {
                const DT *t = (i >= j) ? x : y;
                for (k = 0; k < NLANE; k++) {
                        tmp[k] = t[lb->rows[k][i] * n + lb->cols[k][j]];
                }
                d[i*l+j] = PLOAD(tmp);
        } }
}

/* same-spin second-minor J class, sum_{eta<xi,z<y} */
ALWAYS_INLINE PT FN(pjterm)(const PT *d, const DT *jsl, const FN(Labels) *lb,
                            const int l, size_t n)
{
        const size_t n2 = n * n;
        const size_t n3 = n2 * n;
        DT tmp[NLANE];
        PT acc = PZERO;
        PT m2;
        int eta, xi, z, y, k;
        for (eta = 0; eta < l; eta++) {
        for (xi = eta+1; xi < l; xi++) {
        for (z = 0; z < l; z++) {
        for (y = z+1; y < l; y++) {
                m2 = FN(psecond_minor_det)(d, l, eta, xi, z, y);
                for (k = 0; k < NLANE; k++) {
                        const int *r = lb->rows[k];
                        const int *c = lb->cols[k];
                        tmp[k] = jsl[r[eta]*n3 + c[z]*n2 + r[xi]*n + c[y]]
                               - jsl[r[eta]*n3 + c[y]*n2 + r[xi]*n + c[z]];
                }
                if ((eta + xi + z + y) & 1) {
                        acc = PFMS(acc, m2, PLOAD(tmp));
                } else {
                        acc = PFMA(acc, m2, PLOAD(tmp));
                }
        } } } }
        return acc;
}

ALWAYS_INLINE PT FN(prepl)(const PT *cof, const DT *hcol, const FN(Labels) *lb,
                           const int l, size_t n)
{
        DT tmp[NLANE];
        PT acc = PZERO;
        int eta, z, k;
        for (z = 0; z < l; z++) {
        for (eta = 0; eta < l; eta++) {
                for (k = 0; k < NLANE; k++) {
                        tmp[k] = hcol[lb->cols[k][z] * n + lb->rows[k][eta]];
                }
                acc = PFMA(acc, cof[eta*l+z], PLOAD(tmp));
        } }
        return acc;
}

ALWAYS_INLINE void FN(hs_m0_pkt)(DT *hout, DT *sout, const NOWICKEnv *env,
                                 const int64_t *info, const DT *scal, size_t n,
                                 const FN(Labels) *lba, const int la,
                                 const FN(Labels) *lbb, const int lb)
{
        const DT *slab = env->slab;
        const int64_t *offa = info + NOWICK_AA;
        const int64_t *offb = info + NOWICK_BB;
        const DT *sa = scal + NOWICK_SAA;
        const DT *sb = scal + NOWICK_SBB;
        const size_t n2 = n * n;
        const size_t n3 = n2 * n;
        PT da[NOWICK_MAXL*NOWICK_MAXL+1], cofa[NOWICK_MAXL*NOWICK_MAXL+1];
        PT db[NOWICK_MAXL*NOWICK_MAXL+1], cofb[NOWICK_MAXL*NOWICK_MAXL+1];
        PT deta, detb, ja, jb, repa, repb, ii, inner, core, g0;
        DT tmp[NLANE];
        const DT *iisl = slab + info[NOWICK_AB+NOWICK_IIAB];
        int eta, xi, z, y, k;

        FN(pbuild_d)(da, slab + offa[NOWICK_X], slab + offa[NOWICK_Y], lba, la, n);
        FN(pbuild_d)(db, slab + offb[NOWICK_X], slab + offb[NOWICK_Y], lbb, lb, n);
        deta = FN(pdet_cof)(cofa, da, la, 1);
        detb = FN(pdet_cof)(cofb, db, lb, 1);
        ja = FN(pjterm)(da, slab + offa[NOWICK_J], lba, la, n);
        jb = FN(pjterm)(db, slab + offb[NOWICK_J], lbb, lb, n);
        repa = FN(prepl)(cofa, slab + offa[NOWICK_HCOL0], lba, la, n);
        repb = FN(prepl)(cofb, slab + offb[NOWICK_HCOL0], lbb, lb, n);

        ii = PZERO;
        for (z = 0; z < la; z++) {
        for (eta = 0; eta < la; eta++) {
                inner = PZERO;
                for (y = 0; y < lb; y++) {
                for (xi = 0; xi < lb; xi++) {
                        for (k = 0; k < NLANE; k++) {
                                tmp[k] = iisl[lba->rows[k][eta]*n3 + lba->cols[k][z]*n2
                                              + lbb->rows[k][xi]*n + lbb->cols[k][y]];
                        }
                        inner = PFMA(inner, cofb[xi*lb+y], PLOAD(tmp));
                } }
                ii = PFMA(ii, cofa[eta*la+z], inner);
        } }

        g0 = PBCAST(env->enuc + sa[NOWICK_SF0H] + .5 * sa[NOWICK_SV0]
                    + sb[NOWICK_SF0H] + .5 * sb[NOWICK_SV0] + scal[NOWICK_SVAB0]);
        core = PMUL(g0, PMUL(deta, detb));
        core = PFMS(core, detb, repa);
        core = PFMS(core, deta, repb);
        core = PFMA(core, ja, detb);
        core = PFMA(core, jb, deta);
        core = PADD(core, ii);
        PSTORE(hout, core);
        PSTORE(sout, PMUL(deta, detb));
}

/* (L_a, L_b) with L_a + L_b <= NOWICK_MAXL */
#define PKT_CASE(LA, LB) \
        case LA * (NOWICK_MAXL + 1) + LB: \
                FN(hs_m0_pkt)(h, s, env, info, scal, n, lba, LA, lbb, LB); \
                break;

static void FN(hs_m0_dispatch)(DT *h, DT *s, const NOWICKEnv *env, const int64_t *info,
                               const DT *scal, size_t n, const FN(Labels) *lba, int la,
                               const FN(Labels) *lbb, int lb)
{
        switch (la * (NOWICK_MAXL + 1) + lb) {
        NOWICK_LALB_CASES(PKT_CASE)
        }
}
#undef PKT_CASE

/*
 * One row of the m_a = m_b = 0 Hamiltonian block: the bra dx against the
 * nw kets starting at dws, all with the same parent.  Kets with
 * L_a + L_b <= NOWICK_MAXL are binned by (L_a, L_b) into NLANE packets; a
 * partial packet is padded with copies of a valid lane.  Other kets fall
 * back to the scalar evaluator.
 */
void FN(hs_row)(DT *hrow, DT *srow, const NOWICKEnv *env, const int *dx,
                const int *dws, int nw)
{
        const int kmax = env->kmax;
        const int stride = env->dstride;
        const int px = dx[NOWICK_DPARENT];
        const int pw = dws[NOWICK_DPARENT];
        const int64_t *info = env->pairinfo + ((size_t)px * env->nref + pw) * NOWICK_PAIRLEN;
        const DT *scal = (const DT *)env->scal + ((size_t)px * env->nref + pw) * NOWICK_NSCAL;
        const int nocca = info[NOWICK_NOCCA];
        const int noccb = info[NOWICK_NOCCB];
        const size_t n = info[NOWICK_NMO];
        const int rxa = dx[NOWICK_DRA];
        const int rxb = dx[NOWICK_DRB];
        const DT refpref = scal[NOWICK_SAA+NOWICK_SPHASE] * scal[NOWICK_SAA+NOWICK_STSP]
                         * scal[NOWICK_SBB+NOWICK_SPHASE] * scal[NOWICK_SBB+NOWICK_STSP];
        int bins[(NOWICK_MAXL+1)*(NOWICK_MAXL+1)][NLANE];
        int counts[(NOWICK_MAXL+1)*(NOWICK_MAXL+1)];
        FN(Labels) lba, lbb;
        DT h[NLANE], s[NLANE];
        int j, k, key, la, lb, cnt;

        memset(counts, 0, sizeof(counts));
        for (j = 0; j <= nw; j++) {
                const int *dw;
                if (j < nw) {
                        dw = dws + (size_t)j * stride;
                        la = rxa + dw[NOWICK_DRA];
                        lb = rxb + dw[NOWICK_DRB];
                        if (la + lb > NOWICK_MAXL) {
                                HS_SCALAR(hrow+j, srow+j, env, dx, dw);
                                continue;
                        }
                        key = la * (NOWICK_MAXL + 1) + lb;
                        bins[key][counts[key]++] = j;
                        if (counts[key] < NLANE) {
                                continue;
                        }
                        cnt = NLANE;
                } else {
                        /* flush partial packets */
                        for (key = 0; key < (NOWICK_MAXL+1)*(NOWICK_MAXL+1); key++) {
                                if (counts[key] == 0) continue;
                                for (k = counts[key]; k < NLANE; k++) {
                                        bins[key][k] = bins[key][0];
                                }
                                break;
                        }
                        if (key == (NOWICK_MAXL+1)*(NOWICK_MAXL+1)) {
                                break;
                        }
                        cnt = counts[key];
                        la = key / (NOWICK_MAXL + 1);
                        lb = key % (NOWICK_MAXL + 1);
                        j--;    /* revisit the flush step */
                }
                for (k = 0; k < NLANE; k++) {
                        dw = dws + (size_t)bins[key][k] * stride;
                        nw_labels(lba.rows[k], lba.cols[k],
                                   dx+NOWICK_DHEAD, dx+NOWICK_DHEAD+kmax, rxa,
                                   dw+NOWICK_DHEAD, dw+NOWICK_DHEAD+kmax, dw[NOWICK_DRA],
                                   nocca, n);
                        nw_labels(lbb.rows[k], lbb.cols[k],
                                   dx+NOWICK_DHEAD+2*kmax, dx+NOWICK_DHEAD+3*kmax, rxb,
                                   dw+NOWICK_DHEAD+2*kmax, dw+NOWICK_DHEAD+3*kmax, dw[NOWICK_DRB],
                                   noccb, n);
                }
                FN(hs_m0_dispatch)(h, s, env, info, scal, n, &lba, la, &lbb, lb);
                for (k = 0; k < cnt; k++) {
                        int jj = bins[key][k];
                        DT pref = dx[NOWICK_DSIGN] * dws[(size_t)jj*stride+NOWICK_DSIGN] * refpref;
                        hrow[jj] = pref * h[k];
                        srow[jj] = pref * s[k];
                }
                counts[key] = 0;
        }
}

#undef PFMA
#undef PFMS
