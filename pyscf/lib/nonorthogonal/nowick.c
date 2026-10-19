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
void NOWICKdhamiltonian_general(double *h, double *s, const NOWICKEnv *env,
                                const int *dx, const int *dw);
void NOWICKzhamiltonian_general(double complex *h, double complex *s, const NOWICKEnv *env,
                                const int *dx, const int *dw);

static void set_env(NOWICKEnv *env, int64_t *pairinfo, double *slab, double *scal,
                    int nref, int kmax, double enuc)
{
        env->nref = nref;
        env->kmax = kmax;
        env->dstride = NOWICK_DHEAD + 4 * kmax;
        env->simd = 0;
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

/*
 * Dense Hamiltonian and overlap matrices over ndet determinant records.
 * The upper triangle is evaluated and the lower triangle filled by
 * Hermiticity.  Records should be grouped by parent for locality.
 */
void NOWICKhamiltonian_matrix(double *h, double *s, int *dets, int ndet, int kmax,
                              int64_t *pairinfo, double *slab, double *scal, int nref,
                              double enuc, int cplx)
{
        NOWICKEnv env;
        set_env(&env, pairinfo, slab, scal, nref, kmax, enuc);
#pragma omp parallel
{
        size_t i, j;
        const int *dx, *dw;
        double complex *zh = (double complex *)h;
        double complex *zs = (double complex *)s;
#pragma omp for schedule(dynamic, 1)
        for (i = 0; i < ndet; i++) {
                dx = dets + i * env.dstride;
                for (j = i; j < ndet; j++) {
                        dw = dets + j * env.dstride;
                        if (cplx) {
                                NOWICKzhamiltonian(zh+i*ndet+j, zs+i*ndet+j, &env, dx, dw);
                        } else {
                                NOWICKdhamiltonian(h+i*ndet+j, s+i*ndet+j, &env, dx, dw);
                        }
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
}
