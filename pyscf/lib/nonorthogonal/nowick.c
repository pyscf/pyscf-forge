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
 * Batched drivers for the ENOWT kernels
 */

#include <stdlib.h>
#include <complex.h>
#include "nowick.h"

double NOWICKdoverlap(const NOWICKEnv *env, const int *dx, const int *dw);
double complex NOWICKzoverlap(const NOWICKEnv *env, const int *dx, const int *dw);
void NOWICKdhamiltonian(double *h, double *s, const NOWICKEnv *env,
                        const int *dx, const int *dw);
void NOWICKzhamiltonian(double complex *h, double complex *s, const NOWICKEnv *env,
                        const int *dx, const int *dw);
void NOWICKdhs_orth_row(double *h, double *s, const NOWICKEnv *env, const int *dx,
                        const uint64_t *sx, const int *dws, const uint64_t *sws, int nw);
void NOWICKzhs_orth_row(double complex *h, double complex *s, const NOWICKEnv *env,
                        const int *dx, const uint64_t *sx, const int *dws,
                        const uint64_t *sws, int nw);
void NOWICKdhamiltonian_general(double *h, double *s, const NOWICKEnv *env,
                                const int *dx, const int *dw);
void NOWICKzhamiltonian_general(double complex *h, double complex *s, const NOWICKEnv *env,
                                const int *dx, const int *dw);

#if defined(NOWICK_X86_SIMD)
void NOWICKdhs_row_avx2(double *h, double *s, const NOWICKEnv *env, const int64_t *info,
                        const double *scal, const int *dx, const int *dws, int nw);
void NOWICKzhs_row_avx2(double complex *h, double complex *s, const NOWICKEnv *env,
                        const int64_t *info, const double complex *scal,
                        const int *dx, const int *dws, int nw);
void NOWICKdhs_row_avx512(double *h, double *s, const NOWICKEnv *env, const int64_t *info,
                          const double *scal, const int *dx, const int *dws, int nw);
void NOWICKzhs_row_avx512(double complex *h, double complex *s, const NOWICKEnv *env,
                          const int64_t *info, const double complex *scal,
                          const int *dx, const int *dws, int nw);
#endif

static int simd_override = -1;

/* SIMD level supported by the CPU and the build */
int NOWICKsimd_level()
{
        int level = NOWICK_SIMD_NONE;
#if defined(NOWICK_X86_SIMD)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
                level = NOWICK_SIMD_AVX2;
        }
        if (__builtin_cpu_supports("avx512f")) {
                level = NOWICK_SIMD_AVX512;
        }
#endif
        if (simd_override >= 0 && simd_override < level) {
                level = simd_override;
        }
        return level;
}

/* Cap the SIMD level (-1 restores automatic selection).  Returns the level
 * that will be used. */
int NOWICKset_simd(int level)
{
        simd_override = level;
        return NOWICKsimd_level();
}

/* Layout sizes for the check in _wicks.py: PAIRLEN, NSCAL, DHEAD, MAXMO */
void NOWICKlayout(int *sizes)
{
        sizes[0] = NOWICK_PAIRLEN;
        sizes[1] = NOWICK_NSCAL;
        sizes[2] = NOWICK_DHEAD;
        sizes[3] = NOWICK_MAXMO;
}

static void set_env(NOWICKEnv *env, int64_t *pairinfo, double *slab, double *scal,
                    int nref, int kmax, double enuc)
{
        env->nref = nref;
        env->kmax = kmax;
        env->dstride = NOWICK_DHEAD + 4 * kmax;
        env->simd = NOWICKsimd_level();
        env->enuc = enuc;
        env->pairinfo = pairinfo;
        env->slab = slab;
        env->scal = scal;
}

/*
 * Overlaps <dets[reqs[2i]] | dets[reqs[2i+1]]>
 */
void NOWICKoverlap_pairs(double *out, int *reqs, size_t nreq, int *dets, int kmax,
                         int64_t *pairinfo, double *slab, double *scal, int nref,
                         int cplx)
{
        NOWICKEnv env;
        set_env(&env, pairinfo, slab, scal, nref, kmax, 0);
#pragma omp parallel
{
        size_t i;
        const int *dx, *dw;
#pragma omp for schedule(dynamic, 64)
        for (i = 0; i < nreq; i++) {
                dx = dets + (size_t)reqs[2*i] * env.dstride;
                dw = dets + (size_t)reqs[2*i+1] * env.dstride;
                if (cplx) {
                        ((double complex *)out)[i] = NOWICKzoverlap(&env, dx, dw);
                } else {
                        out[i] = NOWICKdoverlap(&env, dx, dw);
                }
        }
}
}

/*
 * Hamiltonian and overlap <dets[reqs[2i]] | H | dets[reqs[2i+1]]>.
 * general != 0 forces the general evaluator (for testing).
 */
void NOWICKhamiltonian_pairs(double *hout, double *sout, int *reqs, size_t nreq,
                             int *dets, int kmax, int64_t *pairinfo, double *slab,
                             double *scal, int nref, double enuc, int cplx, int general)
{
        NOWICKEnv env;
        set_env(&env, pairinfo, slab, scal, nref, kmax, enuc);
#pragma omp parallel
{
        size_t i;
        const int *dx, *dw;
        double complex *zh = (double complex *)hout;
        double complex *zs = (double complex *)sout;
#pragma omp for schedule(dynamic, 64)
        for (i = 0; i < nreq; i++) {
                dx = dets + (size_t)reqs[2*i] * env.dstride;
                dw = dets + (size_t)reqs[2*i+1] * env.dstride;
                if (cplx) {
                        if (general) {
                                NOWICKzhamiltonian_general(zh+i, zs+i, &env, dx, dw);
                        } else {
                                NOWICKzhamiltonian(zh+i, zs+i, &env, dx, dw);
                        }
                } else {
                        if (general) {
                                NOWICKdhamiltonian_general(hout+i, sout+i, &env, dx, dw);
                        } else {
                                NOWICKdhamiltonian(hout+i, sout+i, &env, dx, dw);
                        }
                }
        }
}
}

#if defined(NOWICK_X86_SIMD)
/* Packed row of one corner combination (info, scal) */
static void hs_row(double *h, double *s, const NOWICKEnv *env, const int64_t *info,
                   const double *scal, const int *dx, const int *dws, int nw, int cplx)
{
        if (env->simd == NOWICK_SIMD_AVX512) {
                if (cplx) {
                        NOWICKzhs_row_avx512((double complex *)h, (double complex *)s, env,
                                             info, (const double complex *)scal, dx, dws, nw);
                } else {
                        NOWICKdhs_row_avx512(h, s, env, info, scal, dx, dws, nw);
                }
        } else {
                if (cplx) {
                        NOWICKzhs_row_avx2((double complex *)h, (double complex *)s, env,
                                           info, (const double complex *)scal, dx, dws, nw);
                } else {
                        NOWICKdhs_row_avx2(h, s, env, info, scal, dx, dws, nw);
                }
        }
}
#endif

static void hs_segment(double *h, double *s, const NOWICKEnv *env, const int *dx,
                       const uint64_t *sx, const int *dws, const uint64_t *sws,
                       int nw, int cplx)
{
        const size_t es = cplx ? 2 : 1;
        const size_t k = (size_t)dx[NOWICK_DPARENT] * env->nref + dws[NOWICK_DPARENT];
        const int64_t *info = env->pairinfo + k * NOWICK_PAIRLEN;
        const double *scal = (const double *)env->scal;
        double complex *zh = (double complex *)h;
        double complex *zs = (double complex *)s;
        int j;
        if (dx[NOWICK_DPARENT] == dws[NOWICK_DPARENT]) {
                if (cplx) {
                        NOWICKzhs_orth_row(zh, zs, env, dx, sx, dws, sws, nw);
                } else {
                        NOWICKdhs_orth_row(h, s, env, dx, sx, dws, sws, nw);
                }
                return;
        }
#if defined(NOWICK_X86_SIMD)
        if (info[NOWICK_MA] == 0 && info[NOWICK_MB] == 0
            && env->simd != NOWICK_SIMD_NONE) {
                /* sum the packed rows of every corner combination */
                double *buf = NULL;
                int64_t c;
                size_t kc, i;
                hs_row(h, s, env, info, scal + k * NOWICK_NSCAL * es, dx, dws, nw, cplx);
                if (info[NOWICK_NCORNER] > 1) {
                        buf = malloc(sizeof(double) * es * nw * 2);
                }
                for (c = 1; c < info[NOWICK_NCORNER]; c++) {
                        kc = info[NOWICK_CSTART] + c - 1;
                        hs_row(buf, buf + es * nw, env, env->pairinfo + kc * NOWICK_PAIRLEN,
                               scal + kc * NOWICK_NSCAL * es, dx, dws, nw, cplx);
                        for (i = 0; i < es * nw; i++) {
                                h[i] += buf[i];
                                s[i] += buf[es * nw + i];
                        }
                }
                free(buf);
                return;
        }
#endif
        for (j = 0; j < nw; j++) {
                const int *dw = dws + (size_t)j * env->dstride;
                if (cplx) {
                        NOWICKzhamiltonian(zh+j, zs+j, env, dx, dw);
                } else {
                        NOWICKdhamiltonian(h+j, s+j, env, dx, dw);
                }
        }
}

/*
 * Dense Hamiltonian and overlap matrices over ndet determinant records.
 * The upper triangle is evaluated and the lower triangle filled by
 * Hermiticity.  Records must be sorted by parent: each row is processed in
 * segments of kets sharing a parent, so packed kernels see one reference
 * pair at a time.
 */
void NOWICKhamiltonian_matrix(double *h, double *s, int *dets, int ndet, int kmax,
                              int64_t *pairinfo, double *slab, double *scal, int nref,
                              double enuc, int cplx)
{
        NOWICKEnv env;
        const size_t es = cplx ? 2 : 1;
        uint64_t *strs = malloc(sizeof(uint64_t) * NOWICK_NSTR * ndet);
        set_env(&env, pairinfo, slab, scal, nref, kmax, enuc);
#pragma omp parallel
{
        size_t i, j, j1;
#pragma omp for schedule(static)
        for (i = 0; i < ndet; i++) {
                NOWICKdet_strings(strs + i * NOWICK_NSTR, dets + i * env.dstride, kmax,
                                  pairinfo[NOWICK_NOCCA], pairinfo[NOWICK_NOCCB]);
        }
        const int *dx;
        double complex *zh = (double complex *)h;
        double complex *zs = (double complex *)s;
#pragma omp for schedule(dynamic, 1)
        for (i = 0; i < ndet; i++) {
                dx = dets + i * env.dstride;
                for (j = i; j < ndet; j = j1) {
                        int pw = dets[j * env.dstride + NOWICK_DPARENT];
                        for (j1 = j + 1; j1 < ndet; j1++) {
                                if (dets[j1 * env.dstride + NOWICK_DPARENT] != pw) break;
                        }
                        hs_segment(h + (i*ndet+j)*es, s + (i*ndet+j)*es, &env,
                                   dx, strs + i * NOWICK_NSTR,
                                   dets + j * env.dstride, strs + j * NOWICK_NSTR,
                                   j1 - j, cplx);
                }
        }
#pragma omp barrier
#pragma omp for schedule(static)
        for (i = 0; i < ndet; i++) {
                for (j = 0; j < i; j++) {
                        if (cplx) {
                                zh[i*ndet+j] = conj(zh[j*ndet+i]);
                                zs[i*ndet+j] = conj(zs[j*ndet+i]);
                        } else {
                                h[i*ndet+j] = h[j*ndet+i];
                                s[i*ndet+j] = s[j*ndet+i];
                        }
                }
        }
}
        free(strs);
}
