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
 * Scalar ENOWT kernels.  This file is a template included by nowick_d.c
 * (DT = double) and nowick_z.c (DT = double complex).  The including file
 * defines DT, FN(name) and ABS2(x).
 *
 * For a bra excitation from reference x with rank L_x and a ket excitation
 * from w with rank L_w, the contraction determinant D has dimension
 * L = L_x + L_w.  Row labels lie in V_x + O_w and column labels in O_x + V_w:
 *      (r_k, c_k) = (a_k - nocc, i_k)          k < L_x
 *      (r_k, c_k) = (nvir + j_k, b_k)          L_x <= k < L
 * D_{ij} = X^{(m_j)}_{r_i c_j} for i >= j and Y^{(m_j)}_{r_i c_j} for i < j.
 * When the reference pair has m zero-overlap orbital pairs, matrix elements
 * are sums over the binary distributions sum_k m_k = m.
 */

#define ALWAYS_INLINE   static inline __attribute__((always_inline))

/*
 * Faddeev-LeVerrier recurrence for det(A) and, optionally, the cofactor
 * matrix cof[r*n+c] = adj(A)[c*n+r].  It is a polynomial in the entries of
 * A, so the cofactors are exact for singular A.  Inlined with a constant n,
 * the loops are fully unrolled.
 */
ALWAYS_INLINE DT FN(fl_det_cof)(DT *cof, const DT *a, const int n, const int want_cof)
{
        DT b[NOWICK_MAXL*NOWICK_MAXL];
        DT p[NOWICK_MAXL*NOWICK_MAXL];
        DT c, v;
        int i, j, k, l;
        if (n == 0) {
                return 1;
        }
        if (n == 1) {
                if (want_cof) {
                        cof[0] = 1;
                }
                return a[0];
        }
        if (n == 2) {
                if (want_cof) {
                        cof[0] = a[3];
                        cof[1] = -a[2];
                        cof[2] = -a[1];
                        cof[3] = a[0];
                }
                return a[0] * a[3] - a[1] * a[2];
        }
        if (n == 3 && !want_cof) {
                return a[0] * (a[4] * a[8] - a[5] * a[7])
                     - a[1] * (a[3] * a[8] - a[5] * a[6])
                     + a[2] * (a[3] * a[7] - a[4] * a[6]);
        }
        /* B_1 = A + c_1 I with c_1 = -tr(A) */
        c = 0;
        for (i = 0; i < n; i++) {
                c -= a[i*n+i];
        }
        for (i = 0; i < n*n; i++) {
                b[i] = a[i];
        }
        for (i = 0; i < n; i++) {
                b[i*n+i] += c;
        }
        for (k = 2; k <= n; k++) {
                if (want_cof && k == n) {
                        /* adj(A) = (-1)^{n-1} B_{n-1} */
                        for (i = 0; i < n; i++) {
                        for (j = 0; j < n; j++) {
                                cof[i*n+j] = (n & 1) ? b[j*n+i] : -b[j*n+i];
                        } }
                }
                if (k == n) {
                        /* only the trace of A B_{n-1} is needed */
                        c = 0;
                        for (i = 0; i < n; i++) {
                        for (l = 0; l < n; l++) {
                                c += a[i*n+l] * b[l*n+i];
                        } }
                        c *= -1. / n;
                        return (n & 1) ? -c : c;
                }
                for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                        v = 0;
                        for (l = 0; l < n; l++) {
                                v += a[i*n+l] * b[l*n+j];
                        }
                        p[i*n+j] = v;
                } }
                c = 0;
                for (i = 0; i < n; i++) {
                        c += p[i*n+i];
                }
                c *= -1. / k;
                for (i = 0; i < n*n; i++) {
                        b[i] = p[i];
                }
                for (i = 0; i < n; i++) {
                        b[i*n+i] += c;
                }
        }
        return 0;
}

/* Partial-pivot LU determinant, used above NOWICK_MAXL */
static DT FN(det_lu)(const DT *a, int n)
{
        DT *lu = malloc(sizeof(DT) * n * n);
        DT det = 1;
        DT f, t;
        int i, j, k, piv;
        double amax, aval;
        for (i = 0; i < n*n; i++) {
                lu[i] = a[i];
        }
        for (k = 0; k < n; k++) {
                piv = k;
                amax = ABS2(lu[k*n+k]);
                for (i = k+1; i < n; i++) {
                        aval = ABS2(lu[i*n+k]);
                        if (aval > amax) {
                                amax = aval;
                                piv = i;
                        }
                }
                if (amax == 0) {
                        free(lu);
                        return 0;
                }
                if (piv != k) {
                        for (j = 0; j < n; j++) {
                                t = lu[k*n+j];
                                lu[k*n+j] = lu[piv*n+j];
                                lu[piv*n+j] = t;
                        }
                        det = -det;
                }
                det *= lu[k*n+k];
                for (i = k+1; i < n; i++) {
                        f = lu[i*n+k] / lu[k*n+k];
                        for (j = k+1; j < n; j++) {
                                lu[i*n+j] -= f * lu[k*n+j];
                        }
                }
        }
        free(lu);
        return det;
}

static void FN(minor)(DT *out, const DT *a, int n, int r0, int c0)
{
        int i, j, k = 0;
        for (i = 0; i < n; i++) {
                if (i == r0) continue;
                for (j = 0; j < n; j++) {
                        if (j == c0) continue;
                        out[k++] = a[i*n+j];
                }
        }
}

#define FL_CASES(F) \
        F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8)

static DT FN(det)(const DT *a, int n)
{
        switch (n) {
#define CASE(N) case N: return FN(fl_det_cof)(NULL, a, N, 0);
        FL_CASES(CASE)
#undef CASE
        }
        return FN(det_lu)(a, n);
}

/* Determinant and cofactor matrix for arbitrary n */
static DT FN(det_cof)(DT *cof, const DT *a, int n)
{
        DT *mnr;
        int i, j;
        switch (n) {
#define CASE(N) case N: return FN(fl_det_cof)(cof, a, N, 1);
        FL_CASES(CASE)
#undef CASE
        }
        /* explicit minors keep the cofactors exact for singular a */
        mnr = malloc(sizeof(DT) * (n-1) * (n-1));
        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
                FN(minor)(mnr, a, n, i, j);
                cof[i*n+j] = ((i + j) & 1) ? -FN(det_lu)(mnr, n-1)
                                           :  FN(det_lu)(mnr, n-1);
        } }
        free(mnr);
        return FN(det_lu)(a, n);
}

/* Determinant of a with rows r0 < r1 and columns c0 < c1 removed */
ALWAYS_INLINE DT FN(second_minor_det)(DT *buf, const DT *a, const int n,
                                      int r0, int r1, int c0, int c1)
{
        int i, j, k = 0;
        for (i = 0; i < n; i++) {
                if (i == r0 || i == r1) continue;
                for (j = 0; j < n; j++) {
                        if (j == c0 || j == c1) continue;
                        buf[k++] = a[i*n+j];
                }
        }
        if (n - 2 <= NOWICK_MAXL) {
                switch (n - 2) {
#define CASE(N) case N: return FN(fl_det_cof)(NULL, buf, N, 0);
                FL_CASES(CASE)
#undef CASE
                }
        }
        return FN(det_lu)(buf, n-2);
}

/* Map the excitations of one spin to contraction-determinant labels */
ALWAYS_INLINE void FN(labels)(int *rows, int *cols,
                              const int *hx, const int *px, int rx,
                              const int *hw, const int *pw, int rw,
                              int nocc, int nmo)
{
        int i;
        int nvir = nmo - nocc;
        for (i = 0; i < rx; i++) {
                rows[i] = px[i] - nocc;
                cols[i] = hx[i];
        }
        for (i = 0; i < rw; i++) {
                rows[rx+i] = nvir + hw[i];
                cols[rx+i] = pw[i];
        }
}

ALWAYS_INLINE void FN(build_d)(DT *d, const DT *x, const DT *y,
                               const int *rows, const int *cols, const int l, int n)
{
        int i, j;
        for (i = 0; i < l; i++) {
                const DT *xr = x + (size_t)rows[i] * n;
                const DT *yr = y + (size_t)rows[i] * n;
                for (j = 0; j <= i; j++) {
                        d[i*l+j] = xr[cols[j]];
                }
                for (j = i+1; j < l; j++) {
                        d[i*l+j] = yr[cols[j]];
                }
        }
}

/* Select column k of mix from d1 when bit k of bits is set, else from d0 */
static void FN(mix_columns)(DT *mix, const DT *d0, const DT *d1, int l, uint64_t bits)
{
        int i, j;
        for (i = 0; i < l; i++) {
        for (j = 0; j < l; j++) {
                mix[i*l+j] = ((bits >> j) & 1) ? d1[i*l+j] : d0[i*l+j];
        } }
}

/* Next integer with the same popcount (Gosper) */
static inline uint64_t next_combination(uint64_t x)
{
        uint64_t c = x & -x;
        uint64_t r = x + c;
        return (((r ^ x) >> 2) / c) | r;
}

/* number of binary distributions of m over l contractions */
static inline int64_t ncombinations(int l, int m)
{
        int64_t c = 1;
        int i;
        if (m < 0 || m > l) return 0;
        for (i = 0; i < m; i++) {
                c = c * (l - i) / (i + 1);
        }
        return c;
}

typedef struct {
        const DT *slab;
        const int64_t *off;     /* same-spin offset block */
        const DT *scal;         /* same-spin scalar block */
        int m;
        int nocc;
        int nmo;
} FN(Spin);

static void FN(spin_view)(FN(Spin) *sp, const NOWICKEnv *env, const int64_t *info,
                          const DT *scal, int alpha)
{
        sp->slab = env->slab;
        sp->off = info + (alpha ? NOWICK_AA : NOWICK_BB);
        sp->scal = scal + (alpha ? NOWICK_SAA : NOWICK_SBB);
        sp->m = info[alpha ? NOWICK_MA : NOWICK_MB];
        sp->nocc = info[alpha ? NOWICK_NOCCA : NOWICK_NOCCB];
        sp->nmo = info[NOWICK_NMO];
}

#define SLAB(sp, k)     ((sp)->slab + (sp)->off[k])

/*
 * Same-spin overlap sum_{m_1..m_L} det D(m_1..m_L), without the reduced
 * overlap prefactor.
 */
static DT FN(spin_overlap)(const FN(Spin) *sp, const int *rows, const int *cols, int l)
{
        const int n = sp->nmo;
        DT dbuf[NOWICK_MAXL*NOWICK_MAXL];
        DT *d0, *d1, *mix;
        DT acc;
        uint64_t bits, last;
        if (sp->m > l) {
                return 0;
        }
        if (l == 0) {
                return 1;
        }
        if (sp->m == 0 && l <= NOWICK_MAXL) {
                FN(build_d)(dbuf, SLAB(sp, NOWICK_X), SLAB(sp, NOWICK_Y), rows, cols, l, n);
                return FN(det)(dbuf, l);
        }
        d0 = malloc(sizeof(DT) * l * l * 3);
        d1 = d0 + l * l;
        mix = d1 + l * l;
        FN(build_d)(d0, SLAB(sp, NOWICK_X), SLAB(sp, NOWICK_Y), rows, cols, l, n);
        FN(build_d)(d1, SLAB(sp, NOWICK_X+1), SLAB(sp, NOWICK_Y+1), rows, cols, l, n);
        acc = 0;
        last = (l == 64) ? ~0ULL : (1ULL << l);
        for (bits = (1ULL << sp->m) - 1; bits < last; bits = next_combination(bits)) {
                FN(mix_columns)(mix, d0, d1, l, bits);
                acc += FN(det)(mix, l);
                if (sp->m == 0) break;
        }
        free(d0);
        return acc;
}

static inline DT FN(spin_pref)(const FN(Spin) *sp)
{
        return sp->scal[NOWICK_SPHASE] * sp->scal[NOWICK_STSP];
}

#define PAIRINFO(env, k)        ((env)->pairinfo + (size_t)(k) * NOWICK_PAIRLEN)
#define PAIRSCAL(env, k)        ((const DT *)(env)->scal + (size_t)(k) * NOWICK_NSCAL)

/* Row of corner combination c of the pair whose primary row is info */
static inline size_t corner_row(const int64_t *info, size_t k, int64_t c)
{
        return c == 0 ? k : (size_t)(info[NOWICK_CSTART] + c - 1);
}

/* Overlap between two determinant records at one corner combination */
static DT FN(overlap_corner)(const NOWICKEnv *env, const int64_t *info, const DT *scal,
                             const int *dx, const int *dw)
{
        const int kmax = env->kmax;
        int rxa = dx[NOWICK_DRA], rxb = dx[NOWICK_DRB];
        int rwa = dw[NOWICK_DRA], rwb = dw[NOWICK_DRB];
        int la = rxa + rwa;
        int lb = rxb + rwb;
        int rows[2*kmax+1], cols[2*kmax+1];
        FN(Spin) sa, sb;
        DT ova, ovb;
        if (info[NOWICK_MA] > la || info[NOWICK_MB] > lb) {
                return 0;
        }
        FN(spin_view)(&sa, env, info, scal, 1);
        FN(spin_view)(&sb, env, info, scal, 0);
        FN(labels)(rows, cols, dx+NOWICK_DHEAD, dx+NOWICK_DHEAD+kmax, rxa,
                   dw+NOWICK_DHEAD, dw+NOWICK_DHEAD+kmax, rwa, sa.nocc, sa.nmo);
        ova = FN(spin_overlap)(&sa, rows, cols, la);
        if (ova == 0) {
                return 0;
        }
        FN(labels)(rows, cols, dx+NOWICK_DHEAD+2*kmax, dx+NOWICK_DHEAD+3*kmax, rxb,
                   dw+NOWICK_DHEAD+2*kmax, dw+NOWICK_DHEAD+3*kmax, rwb, sb.nocc, sb.nmo);
        ovb = FN(spin_overlap)(&sb, rows, cols, lb);
        return dx[NOWICK_DSIGN] * dw[NOWICK_DSIGN]
                * FN(spin_pref)(&sa) * FN(spin_pref)(&sb) * ova * ovb;
}

/* Overlap between two determinant records */
DT FN(overlap)(const NOWICKEnv *env, const int *dx, const int *dw)
{
        const size_t k = (size_t)dx[NOWICK_DPARENT] * env->nref + dw[NOWICK_DPARENT];
        const int64_t *info = PAIRINFO(env, k);
        DT out = 0;
        int64_t c;
        size_t kc;
        for (c = 0; c < info[NOWICK_NCORNER]; c++) {
                kc = corner_row(info, k, c);
                out += FN(overlap_corner)(env, PAIRINFO(env, kc), PAIRSCAL(env, kc), dx, dw);
        }
        return out;
}

/*
 * Map the four contraction assignments of J_{eta z, xi y} to one of the ten
 * stored symmetry-unique tensors.  swap marks the pair-exchanged access
 * J_{xi y, eta z}.
 */
static const int8_t JSLOT[16] = {
        0, 1, 2, 3,     /* (0,0,0,0) (0,0,0,1) (0,0,1,0) (0,0,1,1) */
        1, 4, 5, 6,     /* (0,1,0,0) (0,1,0,1) (0,1,1,0) (0,1,1,1) */
        2, 5, 7, 8,     /* (1,0,0,0) (1,0,0,1) (1,0,1,0) (1,0,1,1) */
        3, 6, 8, 9,     /* (1,1,0,0) (1,1,0,1) (1,1,1,0) (1,1,1,1) */
};
static const int8_t JSWAP[16] = {
        0, 0, 0, 0,
        1, 0, 0, 0,
        1, 1, 0, 0,
        1, 1, 1, 0,
};

/*
 * J entry for the fixed pair (a,b) and varying pair (c,d) of determinant
 * positions.  bits holds (m1, m2, m_col0, m_col1, ...).
 */
ALWAYS_INLINE DT FN(jget)(const FN(Spin) *sp, const int *rows, const int *cols,
                          uint64_t bits, int a, int b, int c, int d)
{
        const size_t n = sp->nmo;
        int mi = bits & 1;
        int mj = (bits >> 1) & 1;
        int mk = (bits >> (d + 2)) & 1;
        int ml = (bits >> (b + 2)) & 1;
        int key = mi * 8 + mj * 4 + mk * 2 + ml;
        const DT *t = sp->slab + sp->off[NOWICK_J + JSLOT[key]];
        size_t ri = rows[a], cj = cols[b], rr = rows[c], cc = cols[d];
        if (JSWAP[key]) {
                return t[((rr * n + cc) * n + ri) * n + cj];
        } else {
                return t[((ri * n + cj) * n + rr) * n + cc];
        }
}

/*
 * Same-spin two-column J contribution for one distribution:
 *   sum_{eta<xi, z<y} (-1)^{eta+xi+z+y} det D[eta,xi|z,y]
 *     * (J(eta z, xi y) + J(xi y, eta z) - J(eta y, xi z) - J(xi z, eta y))
 * The m = 0 J tensor is pair symmetric, so the bracket is twice the
 * direct-minus-exchange difference.
 */
static DT FN(jterm)(const FN(Spin) *sp, const DT *d, const int *rows, const int *cols,
                    int l, uint64_t bits)
{
        DT *buf = malloc(sizeof(DT) * (l > 2 ? (l-2)*(l-2) : 1));
        DT acc = 0;
        DT m2, val;
        int eta, xi, z, y;
        for (eta = 0; eta < l; eta++) {
        for (xi = eta+1; xi < l; xi++) {
        for (z = 0; z < l; z++) {
        for (y = z+1; y < l; y++) {
                m2 = FN(second_minor_det)(buf, d, l, eta, xi, z, y);
                if (m2 == 0) continue;
                val = FN(jget)(sp, rows, cols, bits, eta, z, xi, y)
                    + FN(jget)(sp, rows, cols, bits, xi, y, eta, z)
                    - FN(jget)(sp, rows, cols, bits, eta, y, xi, z)
                    - FN(jget)(sp, rows, cols, bits, xi, z, eta, y);
                if ((eta + xi + z + y) & 1) {
                        acc -= m2 * val;
                } else {
                        acc += m2 * val;
                }
        } } } }
        free(buf);
        return acc;
}

/* sum_{k,r} col_t[cols[k]*n + rows[r]] cof[r*l+k] for the column matrix col_t
 * chosen by the assignment of column k */
ALWAYS_INLINE DT FN(column_sum)(const DT *t0, const DT *t1, const DT *cof,
                                const int *rows, const int *cols, const int l,
                                size_t n, uint64_t colbits)
{
        DT acc = 0;
        int k, r;
        for (k = 0; k < l; k++) {
                const DT *t = (((colbits >> k) & 1) ? t1 : t0) + cols[k] * n;
                for (r = 0; r < l; r++) {
                        acc += t[rows[r]] * cof[r*l+k];
                }
        }
        return acc;
}

typedef struct {
        DT s;           /* sum of overlap distributions */
        DT h1;          /* one-electron */
        DT h2;          /* same-spin two-electron, without the 1/2 */
} FN(SpinTerms);

/*
 * General same-spin pass.  Enumerates sum_{k=1}^{L+2} m_k = m with m_1 and
 * m_2 the operator contractions.  m_2 = 0 selects the one-body terms and
 * m_1 = m_2 = 0 the overlap terms.
 */
static void FN(spin_terms)(FN(SpinTerms) *out, const FN(Spin) *sp,
                           const int *rows, const int *cols, int l)
{
        const size_t n = sp->nmo;
        const int m = sp->m;
        DT *d0, *d1, *mix, *cof;
        DT det, h;
        uint64_t bits, last, cb;
        int m1, m2;
        out->s = 0;
        out->h1 = 0;
        out->h2 = 0;
        if (m > l + 2) {
                return;
        }
        d0 = malloc(sizeof(DT) * (l * l * 4 + 1));
        d1 = d0 + l * l;
        mix = d1 + l * l;
        cof = mix + l * l;
        FN(build_d)(d0, SLAB(sp, NOWICK_X), SLAB(sp, NOWICK_Y), rows, cols, l, n);
        FN(build_d)(d1, SLAB(sp, NOWICK_X+1), SLAB(sp, NOWICK_Y+1), rows, cols, l, n);

        last = 1ULL << (l + 2);
        for (bits = (1ULL << m) - 1; bits < last; bits = (m ? next_combination(bits) : last)) {
                m1 = bits & 1;
                m2 = (bits >> 1) & 1;
                cb = bits >> 2;
                FN(mix_columns)(mix, d0, d1, l, cb);
                det = FN(det_cof)(cof, mix, l);
                if (m1 == 0 && m2 == 0) {
                        out->s += det;
                }
                if (m2 == 0) {
                        h = sp->scal[NOWICK_SF0H+m1] * det;
                        h -= FN(column_sum)(SLAB(sp, NOWICK_FH+m1*2), SLAB(sp, NOWICK_FH+m1*2+1),
                                            cof, rows, cols, l, n, cb);
                        out->h1 += h;
                }
                h = sp->scal[NOWICK_SV0+m1+m2] * det;
                h -= 2 * FN(column_sum)(SLAB(sp, NOWICK_V+m1*4+m2*2), SLAB(sp, NOWICK_V+m1*4+m2*2+1),
                                        cof, rows, cols, l, n, cb);
                if (l >= 2) {
                        h += FN(jterm)(sp, mix, rows, cols, l, bits);
                }
                out->h2 += h;
        }
        free(d0);
}

/*
 * Different-spin pass.  The alpha and beta distributions
 *   m_{a0} + sum_z m_{az} = m_a,      m_{b0} + sum_y m_{by} = m_b
 * are enumerated independently; bit 0 holds the operator contraction.
 */
static DT FN(diff_terms)(const NOWICKEnv *env, const int64_t *info, const DT *scal,
                         const FN(Spin) *sa, const int *rows_a, const int *cols_a, int la,
                         const FN(Spin) *sb, const int *rows_b, const int *cols_b, int lb)
{
        const DT *slab = env->slab;
        const int64_t *off = info + NOWICK_AB;
        const size_t n = sa->nmo;
        const int ma = sa->m;
        const int mb = sb->m;
        int64_t nca = ncombinations(la + 1, ma);
        int64_t ncb = ncombinations(lb + 1, mb);
        int64_t ia, ib;
        int sza = la * la + 1;
        int szb = lb * lb + 1;
        DT *buf, *deta, *detb, *cofa, *cofb, *mix, *d0, *d1;
        uint64_t *bitsa, *bitsb, bits, last;
        DT acc = 0;
        DT contrib, da, db, inner;
        int ma0, mb0, i, j, k, r;
        size_t ra, ca;

        if (nca == 0 || ncb == 0) {
                return 0;
        }
        buf = malloc(sizeof(DT) * ((nca + ncb) * 2 + nca * sza + ncb * szb
                                   + 3 * (sza + szb)));
        bitsa = malloc(sizeof(uint64_t) * (nca + ncb));
        bitsb = bitsa + nca;
        deta = buf;
        detb = deta + nca;
        cofa = detb + ncb;
        cofb = cofa + nca * sza;
        mix = cofb + ncb * szb;
        d0 = mix + sza + szb;
        d1 = d0 + sza + szb;

        FN(build_d)(d0, SLAB(sa, NOWICK_X), SLAB(sa, NOWICK_Y), rows_a, cols_a, la, n);
        FN(build_d)(d1, SLAB(sa, NOWICK_X+1), SLAB(sa, NOWICK_Y+1), rows_a, cols_a, la, n);
        last = 1ULL << (la + 1);
        for (ia = 0, bits = (1ULL << ma) - 1; ia < nca;
             ia++, bits = (ia < nca) ? next_combination(bits) : 0) {
                bitsa[ia] = bits;
                FN(mix_columns)(mix, d0, d1, la, bits >> 1);
                deta[ia] = FN(det_cof)(cofa + ia * sza, mix, la);
        }
        FN(build_d)(d0, SLAB(sb, NOWICK_X), SLAB(sb, NOWICK_Y), rows_b, cols_b, lb, n);
        FN(build_d)(d1, SLAB(sb, NOWICK_X+1), SLAB(sb, NOWICK_Y+1), rows_b, cols_b, lb, n);
        last = 1ULL << (lb + 1);
        for (ib = 0, bits = (1ULL << mb) - 1; ib < ncb;
             ib++, bits = (ib < ncb) ? next_combination(bits) : 0) {
                bitsb[ib] = bits;
                FN(mix_columns)(mix, d0, d1, lb, bits >> 1);
                detb[ib] = FN(det_cof)(cofb + ib * szb, mix, lb);
        }
        (void)last;

        for (ia = 0; ia < nca; ia++) {
        for (ib = 0; ib < ncb; ib++) {
                const DT *ca_ = cofa + ia * sza;
                const DT *cb_ = cofb + ib * szb;
                uint64_t ba = bitsa[ia];
                uint64_t bb = bitsb[ib];
                ma0 = ba & 1;
                mb0 = bb & 1;
                da = deta[ia];
                db = detb[ib];
                contrib = scal[NOWICK_SVAB0 + ma0*2 + mb0] * da * db;
                contrib -= db * FN(column_sum)(slab + off[NOWICK_VAB+ma0*4+mb0*2],
                                               slab + off[NOWICK_VAB+ma0*4+mb0*2+1],
                                               ca_, rows_a, cols_a, la, n, ba >> 1);
                contrib -= da * FN(column_sum)(slab + off[NOWICK_VBA+mb0*4+ma0*2],
                                               slab + off[NOWICK_VBA+mb0*4+ma0*2+1],
                                               cb_, rows_b, cols_b, lb, n, bb >> 1);
                for (i = 0; i < la; i++) {
                for (j = 0; j < la; j++) {
                        int maj = (ba >> (j + 1)) & 1;
                        ra = rows_a[i];
                        ca = cols_a[j];
                        inner = 0;
                        for (k = 0; k < lb; k++) {
                                int mbk = (bb >> (k + 1)) & 1;
                                const DT *t = slab + off[NOWICK_IIAB+ma0*8+maj*4+mb0*2+mbk]
                                        + ((ra * n + ca) * n) * n + cols_b[k];
                                for (r = 0; r < lb; r++) {
                                        inner += t[rows_b[r] * n] * cb_[r*lb+k];
                                }
                        }
                        acc += ca_[i*la+j] * inner;
                } }
                acc += contrib;
        } }
        free(bitsa);
        free(buf);
        return acc;
}

/*
 * Fixed-rank m_a = m_b = 0 Hamiltonian and overlap (no prefactors).  The
 * one-column classes use the precombined hcol0 = F + V + V^{ab}.
 */
ALWAYS_INLINE void FN(hs_m0_fixed)(DT *hout, DT *sout, const NOWICKEnv *env,
                                   const int64_t *info, const DT *scal,
                                   const FN(Spin) *sa, const int *rows_a, const int *cols_a, const int la,
                                   const FN(Spin) *sb, const int *rows_b, const int *cols_b, const int lb)
{
        const size_t n = sa->nmo;
        const size_t n2 = n * n;
        const size_t n3 = n2 * n;
        DT da[NOWICK_MAXL*NOWICK_MAXL+1], cofa[NOWICK_MAXL*NOWICK_MAXL+1];
        DT db[NOWICK_MAXL*NOWICK_MAXL+1], cofb[NOWICK_MAXL*NOWICK_MAXL+1];
        DT mnr[NOWICK_MAXL*NOWICK_MAXL+1];
        DT deta, detb, ja, jb, repa, repb, ii, inner, m2, g0, core;
        const DT *jsl, *hcol, *iisl;
        int eta, xi, z, y;

        FN(build_d)(da, SLAB(sa, NOWICK_X), SLAB(sa, NOWICK_Y), rows_a, cols_a, la, n);
        FN(build_d)(db, SLAB(sb, NOWICK_X), SLAB(sb, NOWICK_Y), rows_b, cols_b, lb, n);
        deta = FN(fl_det_cof)(cofa, da, la, 1);
        detb = FN(fl_det_cof)(cofb, db, lb, 1);

        ja = 0;
        jsl = SLAB(sa, NOWICK_J);
        for (eta = 0; eta < la; eta++) {
        for (xi = eta+1; xi < la; xi++) {
        for (z = 0; z < la; z++) {
        for (y = z+1; y < la; y++) {
                m2 = FN(second_minor_det)(mnr, da, la, eta, xi, z, y);
                m2 *= jsl[rows_a[eta]*n3 + cols_a[z]*n2 + rows_a[xi]*n + cols_a[y]]
                    - jsl[rows_a[eta]*n3 + cols_a[y]*n2 + rows_a[xi]*n + cols_a[z]];
                if ((eta + xi + z + y) & 1) {
                        ja -= m2;
                } else {
                        ja += m2;
                }
        } } } }
        jb = 0;
        jsl = SLAB(sb, NOWICK_J);
        for (eta = 0; eta < lb; eta++) {
        for (xi = eta+1; xi < lb; xi++) {
        for (z = 0; z < lb; z++) {
        for (y = z+1; y < lb; y++) {
                m2 = FN(second_minor_det)(mnr, db, lb, eta, xi, z, y);
                m2 *= jsl[rows_b[eta]*n3 + cols_b[z]*n2 + rows_b[xi]*n + cols_b[y]]
                    - jsl[rows_b[eta]*n3 + cols_b[y]*n2 + rows_b[xi]*n + cols_b[z]];
                if ((eta + xi + z + y) & 1) {
                        jb -= m2;
                } else {
                        jb += m2;
                }
        } } } }

        repa = 0;
        hcol = SLAB(sa, NOWICK_HCOL0);
        for (z = 0; z < la; z++) {
                const DT *t = hcol + cols_a[z] * n;
                for (eta = 0; eta < la; eta++) {
                        repa += cofa[eta*la+z] * t[rows_a[eta]];
                }
        }
        repb = 0;
        hcol = SLAB(sb, NOWICK_HCOL0);
        for (z = 0; z < lb; z++) {
                const DT *t = hcol + cols_b[z] * n;
                for (eta = 0; eta < lb; eta++) {
                        repb += cofb[eta*lb+z] * t[rows_b[eta]];
                }
        }

        ii = 0;
        iisl = (const DT *)env->slab + info[NOWICK_AB+NOWICK_IIAB];
        for (z = 0; z < la; z++) {
        for (eta = 0; eta < la; eta++) {
                const DT *t = iisl + rows_a[eta]*n3 + cols_a[z]*n2;
                inner = 0;
                for (y = 0; y < lb; y++) {
                for (xi = 0; xi < lb; xi++) {
                        inner += cofb[xi*lb+y] * t[rows_b[xi]*n + cols_b[y]];
                } }
                ii += cofa[eta*la+z] * inner;
        } }

        g0 = env->enuc + sa->scal[NOWICK_SF0H] + .5 * sa->scal[NOWICK_SV0]
           + sb->scal[NOWICK_SF0H] + .5 * sb->scal[NOWICK_SV0]
           + scal[NOWICK_SVAB0];
        core = g0 * deta * detb - detb * repa - deta * repb + ja * detb + jb * deta + ii;
        *hout = core;
        *sout = deta * detb;
}

/* General Hamiltonian and overlap (no prefactors) for any m_a, m_b, L */
static void FN(hs_general)(DT *hout, DT *sout, const NOWICKEnv *env,
                           const int64_t *info, const DT *scal,
                           const FN(Spin) *sa, const int *rows_a, const int *cols_a, int la,
                           const FN(Spin) *sb, const int *rows_b, const int *cols_b, int lb)
{
        FN(SpinTerms) ta, tb;
        DT hab;
        FN(spin_terms)(&ta, sa, rows_a, cols_a, la);
        FN(spin_terms)(&tb, sb, rows_b, cols_b, lb);
        hab = FN(diff_terms)(env, info, scal, sa, rows_a, cols_a, la,
                             sb, rows_b, cols_b, lb);
        *sout = ta.s * tb.s;
        *hout = ta.h1 * tb.s + tb.h1 * ta.s + .5 * (ta.h2 * tb.s + tb.h2 * ta.s) + hab
                + env->enuc * ta.s * tb.s;
}

#define HS_M0_CASE(LA, LB) \
        case LA * (NOWICK_MAXL + 1) + LB: \
                FN(hs_m0_fixed)(&h, &s, env, info, scal, &sa, rows_a, cols_a, LA, \
                                &sb, rows_b, cols_b, LB); \
                break;

/*
 * One spin of the excitation taking the ket occupation K to the bra
 * occupation B, both excitations of the same parent:  K\B are the holes and
 * B\K the particles.  Returns the rank, or -1 above maxrank.  odd receives
 * the parity of annihilating the holes (descending) and creating the
 * particles (ascending) on K.
 */
static int FN(relative_excitation)(int *holes, int *parts, int *odd, int nocc,
                                   const int *hb, const int *pb, int rb,
                                   const int *hk, const int *pk, int rk, int maxrank)
{
        int nh = 0, np = 0, i, j, k, below, found;
        /* holes: K\B = (hb\hk) + (pk\pb) */
        for (i = 0; i < rb; i++) {
                for (found = 0, j = 0; j < rk; j++) found |= (hk[j] == hb[i]);
                if (!found) {
                        if (nh == maxrank) return -1;
                        holes[nh++] = hb[i];
                }
        }
        for (i = 0; i < rk; i++) {
                for (found = 0, j = 0; j < rb; j++) found |= (pb[j] == pk[i]);
                if (!found) {
                        if (nh == maxrank) return -1;
                        holes[nh++] = pk[i];
                }
        }
        /* particles: B\K = (hk\hb) + (pb\pk) */
        for (i = 0; i < rk; i++) {
                for (found = 0, j = 0; j < rb; j++) found |= (hb[j] == hk[i]);
                if (!found) {
                        if (np == maxrank) return -1;
                        parts[np++] = hk[i];
                }
        }
        for (i = 0; i < rb; i++) {
                for (found = 0, j = 0; j < rk; j++) found |= (pk[j] == pb[i]);
                if (!found) {
                        if (np == maxrank) return -1;
                        parts[np++] = pb[i];
                }
        }
        if (nh != np) return -1;
        /* holes descending, particles ascending; holes and particles of
         * each list are already disjoint index ranges in each half */
        for (i = 0; i < nh; i++) {
        for (j = i+1; j < nh; j++) {
                if (holes[j] > holes[i]) { k = holes[i]; holes[i] = holes[j]; holes[j] = k; }
                if (parts[j] < parts[i]) { k = parts[i]; parts[i] = parts[j]; parts[j] = k; }
        } }
        /* occupation changes of K relative to {0..nocc-1}: -hk, +pk, then the
         * excitation's own removals (-1) and insertions (+1) */
        *odd = 0;
        for (i = 0; i < nh; i++) {
                /* occupied orbitals of the current string below holes[i] */
                below = holes[i] < nocc ? holes[i] : nocc;
                for (j = 0; j < rk; j++) below -= (hk[j] < holes[i]);
                for (j = 0; j < rk; j++) below += (pk[j] < holes[i]);
                for (j = 0; j < i; j++) below -= (holes[j] < holes[i]);
                *odd ^= below & 1;
        }
        for (i = 0; i < np; i++) {
                below = parts[i] < nocc ? parts[i] : nocc;
                for (j = 0; j < rk; j++) below -= (hk[j] < parts[i]);
                for (j = 0; j < rk; j++) below += (pk[j] < parts[i]);
                for (j = 0; j < nh; j++) below -= (holes[j] < parts[i]);
                for (j = 0; j < i; j++) below += (parts[j] < parts[i]);
                *odd ^= below & 1;
        }
        return nh;
}

/* occupied orbitals of record d (one spin) in the occupied-first order */
static int FN(occupied)(int *occ, int nocc, const int *h, const int *p, int r)
{
        int i, j, k = 0, skip;
        for (i = 0; i < nocc; i++) {
                for (skip = 0, j = 0; j < r; j++) skip |= (h[j] == i);
                if (!skip) occ[k++] = i;
        }
        for (j = 0; j < r; j++) {
                occ[k++] = p[j];
        }
        return k;
}

/*
 * Same-parent Hamiltonian and overlap with the Slater-Condon rules in the
 * parent's orthonormal MO basis (noci-rs elements/orthogonal).
 */
static void FN(hs_orth)(DT *hout, DT *sout, const NOWICKEnv *env,
                        const int64_t *info, const int *dx, const int *dw)
{
        const int kmax = env->kmax;
        const DT *slab = env->slab;
        const size_t n = info[NOWICK_NMO];
        const size_t n2 = n * n;
        const size_t n3 = n2 * n;
        const int nocca = info[NOWICK_NOCCA];
        const int noccb = info[NOWICK_NOCCB];
        const DT *ha = slab + info[NOWICK_OH_A];
        const DT *hb = slab + info[NOWICK_OH_B];
        const DT *aa = slab + info[NOWICK_OAA];
        const DT *bb = slab + info[NOWICK_OBB];
        const DT *ab = slab + info[NOWICK_OAB];
        int ha_[2], pa_[2], hb_[2], pb_[2];
        int ra, rb, odda, oddb;
        int oa[nocca + 1];
        int ob[noccb + 1];
        int na, nb, i, j;
        size_t p, q, r, s;
        DT h = 0;
        double sign = dx[NOWICK_DOSIGN] * dw[NOWICK_DOSIGN];

        ra = FN(relative_excitation)(ha_, pa_, &odda, nocca,
                                     dx+NOWICK_DHEAD, dx+NOWICK_DHEAD+kmax, dx[NOWICK_DRA],
                                     dw+NOWICK_DHEAD, dw+NOWICK_DHEAD+kmax, dw[NOWICK_DRA], 2);
        rb = FN(relative_excitation)(hb_, pb_, &oddb, noccb,
                                     dx+NOWICK_DHEAD+2*kmax, dx+NOWICK_DHEAD+3*kmax, dx[NOWICK_DRB],
                                     dw+NOWICK_DHEAD+2*kmax, dw+NOWICK_DHEAD+3*kmax, dw[NOWICK_DRB], 2);
        *sout = 0;
        *hout = 0;
        if (ra < 0 || rb < 0 || ra + rb > 2) {
                return;
        }
        if ((odda ^ oddb) & 1) {
                sign = -sign;
        }
        /* occupations of the ket K */
        na = FN(occupied)(oa, nocca, dw+NOWICK_DHEAD, dw+NOWICK_DHEAD+kmax, dw[NOWICK_DRA]);
        nb = FN(occupied)(ob, noccb, dw+NOWICK_DHEAD+2*kmax, dw+NOWICK_DHEAD+3*kmax, dw[NOWICK_DRB]);

        if (ra == 0 && rb == 0) {
                h = env->enuc;
                for (i = 0; i < na; i++) {
                        p = oa[i];
                        h += ha[p*n+p];
                        for (j = 0; j < na; j++) {
                                q = oa[j];
                                h += .5 * aa[p*n3+p*n2+q*n+q];
                        }
                        for (j = 0; j < nb; j++) {
                                q = ob[j];
                                h += ab[p*n3+p*n2+q*n+q];
                        }
                }
                for (i = 0; i < nb; i++) {
                        p = ob[i];
                        h += hb[p*n+p];
                        for (j = 0; j < nb; j++) {
                                q = ob[j];
                                h += .5 * bb[p*n3+p*n2+q*n+q];
                        }
                }
                *sout = sign;
        } else if (ra == 1 && rb == 0) {
                /* (a <- i): h_ai + sum_{j in K_a} <ai||jj> + sum_{j in K_b} (ai|jj) */
                p = pa_[0];
                q = ha_[0];
                h = ha[p*n+q];
                for (j = 0; j < na; j++) {
                        r = oa[j];
                        h += aa[p*n3+q*n2+r*n+r];
                }
                for (j = 0; j < nb; j++) {
                        r = ob[j];
                        h += ab[p*n3+q*n2+r*n+r];
                }
        } else if (ra == 0 && rb == 1) {
                p = pb_[0];
                q = hb_[0];
                h = hb[p*n+q];
                for (j = 0; j < nb; j++) {
                        r = ob[j];
                        h += bb[p*n3+q*n2+r*n+r];
                }
                for (j = 0; j < na; j++) {
                        r = oa[j];
                        h += ab[r*n3+r*n2+p*n+q];
                }
        } else if (ra == 2) {
                /* holes descending (i > j), particles ascending (a < b) */
                p = pa_[0]; q = ha_[1]; r = pa_[1]; s = ha_[0];
                h = aa[p*n3+q*n2+r*n+s];
        } else if (rb == 2) {
                p = pb_[0]; q = hb_[1]; r = pb_[1]; s = hb_[0];
                h = bb[p*n3+q*n2+r*n+s];
        } else {
                /* alpha (a <- i), beta (b <- j): (ai|bj) */
                h = ab[(size_t)pa_[0]*n3 + (size_t)ha_[0]*n2 + (size_t)pb_[0]*n + hb_[0]];
        }
        *hout = sign * h;
}

/* Hamiltonian and overlap between records of different parents at one
 * corner combination */
void FN(hamiltonian_corner)(DT *hout, DT *sout, const NOWICKEnv *env,
                            const int64_t *info, const DT *scal,
                            const int *dx, const int *dw)
{
        const int kmax = env->kmax;
        int rxa = dx[NOWICK_DRA], rxb = dx[NOWICK_DRB];
        int rwa = dw[NOWICK_DRA], rwb = dw[NOWICK_DRB];
        int la = rxa + rwa;
        int lb = rxb + rwb;
        int rows_a[2*kmax+1], cols_a[2*kmax+1];
        int rows_b[2*kmax+1], cols_b[2*kmax+1];
        FN(Spin) sa, sb;
        DT h, s, pref;
        /* a zero-overlap pair needs a contraction in the operator or in D */
        if (info[NOWICK_MA] > la + 2 || info[NOWICK_MB] > lb + 2
            || info[NOWICK_MA] + info[NOWICK_MB] > la + lb + 2) {
                *hout = 0;
                *sout = 0;
                return;
        }
        FN(spin_view)(&sa, env, info, scal, 1);
        FN(spin_view)(&sb, env, info, scal, 0);
        FN(labels)(rows_a, cols_a, dx+NOWICK_DHEAD, dx+NOWICK_DHEAD+kmax, rxa,
                   dw+NOWICK_DHEAD, dw+NOWICK_DHEAD+kmax, rwa, sa.nocc, sa.nmo);
        FN(labels)(rows_b, cols_b, dx+NOWICK_DHEAD+2*kmax, dx+NOWICK_DHEAD+3*kmax, rxb,
                   dw+NOWICK_DHEAD+2*kmax, dw+NOWICK_DHEAD+3*kmax, rwb, sb.nocc, sb.nmo);
        if (sa.m == 0 && sb.m == 0 && la + lb <= NOWICK_MAXL) {
                switch (la * (NOWICK_MAXL + 1) + lb) {
                NOWICK_LALB_CASES(HS_M0_CASE)
                }
        } else {
                FN(hs_general)(&h, &s, env, info, scal, &sa, rows_a, cols_a, la,
                               &sb, rows_b, cols_b, lb);
        }
        pref = dx[NOWICK_DSIGN] * dw[NOWICK_DSIGN] * FN(spin_pref)(&sa) * FN(spin_pref)(&sb);
        *hout = pref * h;
        *sout = pref * s;
}

/* Hamiltonian and overlap between two determinant records */
void FN(hamiltonian)(DT *hout, DT *sout, const NOWICKEnv *env, const int *dx, const int *dw)
{
        const size_t k = (size_t)dx[NOWICK_DPARENT] * env->nref + dw[NOWICK_DPARENT];
        const int64_t *info = PAIRINFO(env, k);
        DT h, s;
        int64_t c;
        size_t kc;
        if (dx[NOWICK_DPARENT] == dw[NOWICK_DPARENT]) {
                FN(hs_orth)(hout, sout, env, info, dx, dw);
                return;
        }
        *hout = 0;
        *sout = 0;
        for (c = 0; c < info[NOWICK_NCORNER]; c++) {
                kc = corner_row(info, k, c);
                FN(hamiltonian_corner)(&h, &s, env, PAIRINFO(env, kc), PAIRSCAL(env, kc),
                                       dx, dw);
                *hout += h;
                *sout += s;
        }
}

/* Hamiltonian between two records at one corner combination with the
 * general evaluator only */
static void FN(general_corner)(DT *hout, DT *sout, const NOWICKEnv *env,
                               const int64_t *info, const DT *scal,
                               const int *dx, const int *dw)
{
        const int kmax = env->kmax;
        int rxa = dx[NOWICK_DRA], rxb = dx[NOWICK_DRB];
        int rwa = dw[NOWICK_DRA], rwb = dw[NOWICK_DRB];
        int rows_a[2*kmax+1], cols_a[2*kmax+1];
        int rows_b[2*kmax+1], cols_b[2*kmax+1];
        FN(Spin) sa, sb;
        DT h, s, pref;
        FN(spin_view)(&sa, env, info, scal, 1);
        FN(spin_view)(&sb, env, info, scal, 0);
        FN(labels)(rows_a, cols_a, dx+NOWICK_DHEAD, dx+NOWICK_DHEAD+kmax, rxa,
                   dw+NOWICK_DHEAD, dw+NOWICK_DHEAD+kmax, rwa, sa.nocc, sa.nmo);
        FN(labels)(rows_b, cols_b, dx+NOWICK_DHEAD+2*kmax, dx+NOWICK_DHEAD+3*kmax, rxb,
                   dw+NOWICK_DHEAD+2*kmax, dw+NOWICK_DHEAD+3*kmax, rwb, sb.nocc, sb.nmo);
        FN(hs_general)(&h, &s, env, info, scal, &sa, rows_a, cols_a, rxa + rwa,
                       &sb, rows_b, cols_b, rxb + rwb);
        pref = dx[NOWICK_DSIGN] * dw[NOWICK_DSIGN] * FN(spin_pref)(&sa) * FN(spin_pref)(&sb);
        *hout = pref * h;
        *sout = pref * s;
}

/* Hamiltonian between two records with the general evaluator only */
void FN(hamiltonian_general)(DT *hout, DT *sout, const NOWICKEnv *env,
                             const int *dx, const int *dw)
{
        const size_t k = (size_t)dx[NOWICK_DPARENT] * env->nref + dw[NOWICK_DPARENT];
        const int64_t *info = PAIRINFO(env, k);
        DT h, s;
        int64_t c;
        size_t kc;
        *hout = 0;
        *sout = 0;
        for (c = 0; c < info[NOWICK_NCORNER]; c++) {
                kc = corner_row(info, k, c);
                FN(general_corner)(&h, &s, env, PAIRINFO(env, kc), PAIRSCAL(env, kc), dx, dw);
                *hout += h;
                *sout += s;
        }
}

#undef PAIRINFO
#undef PAIRSCAL
