/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

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
 * Author: Hong-Zhou Ye <hzyechem@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

typedef struct {
        void *cache[6];
        short a;
        short b;
        short c;
        short _padding;
} CacheJob;

size_t _ccsd_t_gen_jobs(CacheJob *jobs, int nocc, int nvir,
                        int a0, int a1, int b0, int b1,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b, size_t stride);

void _make_permute_indices(int *idx, int n);

/* FIXME:
    reuse functions such as `_ccsd_t_gen_jobs` from pyscf/lib/cc/ccsd_t.c
*/



/*  For LNO-CCSD(T)

    CCSD(T) energy normalized to a set of localized orbitals (LOs).
    The canonical (T) energy
        E = 1/3 * \sum_{ijk,abc} (4 W_{ijk,abc} + W_{ijk,bca} + W_{ijk,cab}) *
                                    (V_{ijk,abc} - V_{ijk,cba}) / D_{ijk,abc}
    is first rewritten using w = W / sqrt(D), v = V / sqrt(D)
        E = 1/3 * \sum_{ijk,abc} (4 w_{ijk,abc} + w_{ijk,bca} + w_{ijk,cab}) *
                                    (v_{ijk,abc} - v_{ijk,cba})
    and then rewritten using 6-fold permutational symmetry and ij being dummy variables
        E = 1/3 * \sum_{a>=b>=c} \sum_{ijk}
                w[ijk] * ( +8*v[ijk] -5*v[ikj] -2*v[jik] +2*v[jki] +2*v[kij] -5*v[kji] ) +
                w[ikj] * ( -5*v[ijk] +8*v[ikj] +2*v[jik] -2*v[jki] -5*v[kij] +2*v[kji] ) +
                w[kij] * ( +2*v[ijk] -5*v[ikj] -5*v[jik] +2*v[jki] +8*v[kij] -2*v[kji] )
    (we omitted abc indices above, which are "abc" for all of them).
    Finally, the local (T) energy for a localized orbital K is
        E^K = 1/3 * \sum_{a>=b>=c} * \sum_{ij}
                w[ijK] * ( +8*v[ijK] -5*v[iKj] -2*v[jiK] +2*v[jKi] +2*v[Kij] -5*v[Kji] ) +
                w[iKj] * ( -5*v[ijK] +8*v[iKj] +2*v[jiK] -2*v[jKi] -5*v[Kij] +2*v[Kji] ) +
                w[Kij] * ( +2*v[ijK] -5*v[iKj] -5*v[jiK] +2*v[jKi] +8*v[Kij] -2*v[Kji] )
    where
        w_{ijK,abc} = \sum_{k} ulo_{Kk} * w_{ijk,abc}
        v_{ijK,abc} = \sum_{k} ulo_{Kk} * v_{ijk,abc}
*/

// copied from pyscf/lib/cc/ccsd_t.c
static void get_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const double DN1 =-1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double *pt2T;

        dgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        dgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static void sym_wv(double *w, double *v, double *cache,
                   double *fvohalf, double *vooo,
                   double *vv_op, double *t1Thalf, double *t2T,
                   int nocc, int nvir, int a, int b, int c, int nirrep,
                   int *o_ir_loc, int *v_ir_loc, int *oo_ir_loc, int *orbsym,
                   int *idx)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int a_irrep = orbsym[nocc+a];
        int b_irrep = orbsym[nocc+b];
        int c_irrep = orbsym[nocc+c];
        int ab_irrep = a_irrep ^ b_irrep;
        int bc_irrep = c_irrep ^ b_irrep;
        int i, j, k, n;
        int fr, f0, f1, df, mr, m0, m1, dm, mk0;
        int ir, i0, i1, di, kr, k0, k1, dk, jr;
        int ijr, ij0, ij1, dij, jkr, jk0, jk1, djk;
        double *pt2T;

/* symmetry adapted
 * w = numpy.einsum('if,fjk->ijk', ov, t2T[c]) */
        pt2T = t2T + c * nvoo;
        for (ir = 0; ir < nirrep; ir++) {
                i0 = o_ir_loc[ir];
                i1 = o_ir_loc[ir+1];
                di = i1 - i0;
                if (di > 0) {
                        fr = ir ^ ab_irrep;
                        f0 = v_ir_loc[fr];
                        f1 = v_ir_loc[fr+1];
                        df = f1 - f0;
                        if (df > 0) {
                                jkr = fr ^ c_irrep;
                                jk0 = oo_ir_loc[jkr];
                                jk1 = oo_ir_loc[jkr+1];
                                djk = jk1 - jk0;
                                if (djk > 0) {

        dgemm_(&TRANS_N, &TRANS_N, &djk, &di, &df,
               &D1, pt2T+f0*noo+jk0, &noo, vv_op+i0*nmo+nocc+f0, &nmo,
               &D0, cache, &djk);
        for (n = 0, i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
        for (jr = 0; jr < nirrep; jr++) {
                kr = jkr ^ jr;
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[idx[i*noo+j*nocc+k]] += cache[n];
                } }
        } }
                                }
                        }
                }
        }

/* symmetry adapted
 * w-= numpy.einsum('ijm,mk->ijk', eris_vooo[a], t2T[c,b]) */
        pt2T = t2T + c * nvoo + b * noo;
        vooo += a * nooo;
        mk0 = oo_ir_loc[bc_irrep];
        for (mr = 0; mr < nirrep; mr++) {
                m0 = o_ir_loc[mr];
                m1 = o_ir_loc[mr+1];
                dm = m1 - m0;
                if (dm > 0) {
                        kr = mr ^ bc_irrep;
                        k0 = o_ir_loc[kr];
                        k1 = o_ir_loc[kr+1];
                        dk = k1 - k0;
                        if (dk > 0) {
                                ijr = mr ^ a_irrep;
                                ij0 = oo_ir_loc[ijr];
                                ij1 = oo_ir_loc[ijr+1];
                                dij = ij1 - ij0;
                                if (dij > 0) {

        dgemm_(&TRANS_N, &TRANS_N, &dk, &dij, &dm,
               &D1, pt2T+mk0, &dk, vooo+ij0*nocc+m0, &nocc,
               &D0, cache, &dk);
        for (n = 0, ir = 0; ir < nirrep; ir++) {
                jr = ijr ^ ir;
                for (i = o_ir_loc[ir]; i < o_ir_loc[ir+1]; i++) {
                for (j = o_ir_loc[jr]; j < o_ir_loc[jr+1]; j++) {
                for (k = o_ir_loc[kr]; k < o_ir_loc[kr+1]; k++, n++) {
                        w[idx[i*noo+j*nocc+k]] -= cache[n];
                } }
        } }
                                }
                                mk0 += dm * dk;
                        }
                }
        }

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}

static void zget_wv(double complex *w, double complex *v,
                    double complex *cache, double complex *fvohalf,
                    double complex *vooo, double complex *vv_op,
                    double complex *t1Thalf, double complex *t2T,
                    int nocc, int nvir, int a, int b, int c, int *idx)
{
        const double complex D0 = 0;
        const double complex D1 = 1;
        const double complex DN1 =-1;
        const char TRANS_N = 'N';
        const int nmo = nocc + nvir;
        const int noo = nocc * nocc;
        const size_t nooo = nocc * noo;
        const size_t nvoo = nvir * noo;
        int i, j, k, n;
        double complex *pt2T;

        zgemm_(&TRANS_N, &TRANS_N, &noo, &nocc, &nvir,
               &D1, t2T+c*nvoo, &noo, vv_op+nocc, &nmo,
               &D0, cache, &noo);
        zgemm_(&TRANS_N, &TRANS_N, &nocc, &noo, &nocc,
               &DN1, t2T+c*nvoo+b*noo, &nocc, vooo+a*nooo, &nocc,
               &D1, cache, &nocc);

        pt2T = t2T + b * nvoo + a * noo;
        for (n = 0, i = 0; i < nocc; i++) {
        for (j = 0; j < nocc; j++) {
        for (k = 0; k < nocc; k++, n++) {
                w[idx[n]] += cache[n];
                v[idx[n]] +=(vv_op[i*nmo+j] * t1Thalf[c*nocc+k]
                           + pt2T[i*nocc+j] * fvohalf[c*nocc+k]);
        } } }
}
// end copy

static double _ccsd_t_get_energy_lo(double *w, double *v, double *mo_energy,
                                    double *cache, double *ulo, int nlo, // <--- extra args
                                    int nocc, int a, int b, int c, double fac)
{
        int i, j, k, mu;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double *tij1 = cache;
        double *tij2 = tij1 + nocc;
        double *tij3 = tij2 + nocc;
        double *vij1 = tij3 + nocc;
        double *vij2 = vij1 + nocc;
        double *vij3 = vij2 + nocc;
        double t3lo, vlo;
        double et = 0;

        int ijk, ikj, jik, jki, kij, kji;
        int n = nocc;
        int nn = n * n;
        double denom;
        double *ulo_mu;

        for (i = 0; i < nocc; i++) {
            for (j = 0; j < nocc; j++) {
                for (k = 0; k < nocc; k++) {
                    ijk = i*nn + j*n + k;
                    ikj = i*nn + k*n + j;
                    jik = j*nn + i*n + k;
                    jki = j*nn + k*n + i;
                    kij = k*nn + i*n + j;
                    kji = k*nn + j*n + i;
                    denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                    denom = 1/sqrt(denom);
                    tij1[k] = w[ijk] * denom;
                    tij2[k] = w[ikj] * denom;
                    tij3[k] = w[kij] * denom;
                    vij1[k] = (+8*v[ijk] -5*v[ikj] -2*v[jik] +2*v[jki] +2*v[kij] -5*v[kji]) * denom;
                    vij2[k] = (-5*v[ijk] +8*v[ikj] +2*v[jik] -2*v[jki] -5*v[kij] +2*v[kji]) * denom;
                    vij3[k] = (+2*v[ijk] -5*v[ikj] -5*v[jik] +2*v[jki] +8*v[kij] -2*v[kji]) * denom;
                }
                for (mu = 0; mu < nlo; mu++) {
                    ulo_mu = ulo + mu*nocc;

                    t3lo = vlo = 0.;
                    for (k = 0; k < nocc; k++) {
                        t3lo += tij1[k] * ulo_mu[k];
                        vlo += vij1[k] * ulo_mu[k];
                    }
                    et += t3lo * vlo;

                    t3lo = vlo = 0.;
                    for (k = 0; k < nocc; k++) {
                        t3lo += tij2[k] * ulo_mu[k];
                        vlo += vij2[k] * ulo_mu[k];
                    }
                    et += t3lo * vlo;

                    t3lo = vlo = 0.;
                    for (k = 0; k < nocc; k++) {
                        t3lo += tij3[k] * ulo_mu[k];
                        vlo += vij3[k] * ulo_mu[k];
                    }
                    et += t3lo * vlo;
                }
            }
        }
        et *= - fac / 6.;

        return et;
}
static double contract6_lo(int nocc, int nvir, int a, int b, int c,
                           double *mo_energy, double *t1T, double *t2T,
                           double *ulo, int nlo,    // <--- extra args
                           int nirrep, int *o_ir_loc, int *v_ir_loc,
                           int *oo_ir_loc, int *orbsym, double *fvo,
                           double *vooo, double *cache1, void **cache,
                           int *permute_idx, double fac)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double *v0 = cache1;
        double *w0 = v0 + nooo;
        double *z0 = w0 + nooo;
        double *cache2 = z0 + nooo;
        double *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        if (nirrep == 1) {
                get_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
                get_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);
        } else {
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx0);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx1);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx2);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx3);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx4);
                sym_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a,
                       nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym, idx5);
        }
        for (i = 0; i < nooo; i++) {
            v0[i] += w0[i];
        }

        double et;
        if (a == c) {
                et = _ccsd_t_get_energy_lo(w0, v0, mo_energy,
                                           cache2, ulo, nlo, // <--- extra args
                                           nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_get_energy_lo(w0, v0, mo_energy,
                                           cache2, ulo, nlo, // <--- extra args
                                           nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_get_energy_lo(w0, v0, mo_energy,
                                           cache2, ulo, nlo, // <--- extra args
                                           nocc, a, b, c, 1.);
        }
        return et;
}
void CCsd_t_contract_lo(double *e_tot,
                        double *mo_energy, double *t1T, double *t2T,
                        double *vooo, double *fvo,
                        double *ulo, int nlo, // <--- extra args
                        int nocc, int nvir, int a0, int a1, int b0, int b1,
                        int nirrep, int *o_ir_loc, int *v_ir_loc,
                        int *oo_ir_loc, int *orbsym,
                        void *cache_row_a, void *cache_col_a,
                        void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b, sizeof(double));
        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);
#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, nlo, ulo, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        // extra 6*nocc for tij and vij in :func:`_ccsd_t_get_energy_lo`
        double *cache1 = malloc(sizeof(double) * (nocc*nocc*nocc*3+2+6*nocc));
        double *t1Thalf = malloc(sizeof(double) * nvir*nocc * 2);
        double *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += contract6_lo(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                                  ulo, nlo, // <--- extra args
                                  nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                  fvohalf, vooo, cache1, jobs[k].cache, permute_idx,
                                  1.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}


static double _ccsd_t_zget_energy_lo(double complex *w, double complex *v, double *mo_energy,
                                     double complex *cache, double complex *ulo, int nlo,
                                     int nocc, int a, int b, int c, double fac)
{
        int i, j, k, mu;
        double abc = mo_energy[nocc+a] + mo_energy[nocc+b] + mo_energy[nocc+c];
        double complex *tij1 = cache;
        double complex *tij2 = tij1 + nocc;
        double complex *tij3 = tij2 + nocc;
        double complex *vij1 = tij3 + nocc;
        double complex *vij2 = vij1 + nocc;
        double complex *vij3 = vij2 + nocc;
        double complex t3lo, vlo;
        double et = 0;

        int ijk, ikj, jik, jki, kij, kji;
        int n = nocc;
        int nn = n * n;
        double denom;
        double complex *ulo_mu;

        for (i = 0; i < nocc; i++) {
            for (j = 0; j < nocc; j++) {
                for (k = 0; k < nocc; k++) {
                    ijk = i*nn + j*n + k;
                    ikj = i*nn + k*n + j;
                    jik = j*nn + i*n + k;
                    jki = j*nn + k*n + i;
                    kij = k*nn + i*n + j;
                    kji = k*nn + j*n + i;
                    denom = abc - (mo_energy[i] + mo_energy[j] + mo_energy[k]);
                    denom = 1/sqrt(denom);
                    tij1[k] = w[ijk] * denom;
                    tij2[k] = w[ikj] * denom;
                    tij3[k] = w[kij] * denom;
                    vij1[k] = (+8*v[ijk] -5*v[ikj] -2*v[jik] +2*v[jki] +2*v[kij] -5*v[kji]) * denom;
                    vij2[k] = (-5*v[ijk] +8*v[ikj] +2*v[jik] -2*v[jki] -5*v[kij] +2*v[kji]) * denom;
                    vij3[k] = (+2*v[ijk] -5*v[ikj] -5*v[jik] +2*v[jki] +8*v[kij] -2*v[kji]) * denom;
                }
                for (mu = 0; mu < nlo; mu++) {
                    ulo_mu = ulo + mu*nocc;

                    t3lo = vlo = 0;
                    for (k = 0; k < nocc; k++) {
                        t3lo += tij1[k] * conj(ulo_mu[k]);
                        vlo += vij1[k] * conj(ulo_mu[k]);
                    }
                    et += t3lo * conj(vlo);

                    t3lo = vlo = 0;
                    for (k = 0; k < nocc; k++) {
                        t3lo += tij2[k] * conj(ulo_mu[k]);
                        vlo += vij2[k] * conj(ulo_mu[k]);
                    }
                    et += t3lo * conj(vlo);

                    t3lo = vlo = 0;
                    for (k = 0; k < nocc; k++) {
                        t3lo += tij3[k] * conj(ulo_mu[k]);
                        vlo += vij3[k] * conj(ulo_mu[k]);
                    }
                    et += t3lo * conj(vlo);
                }
            }
        }
        et *= - fac / 6.;

        return et;
}
static double complex
zcontract6_lo(int nocc, int nvir, int a, int b, int c,
              double *mo_energy, double complex *t1T, double complex *t2T,
              double complex *ulo, int nlo,    // <--- extra args
              int nirrep, int *o_ir_loc, int *v_ir_loc,
              int *oo_ir_loc, int *orbsym, double complex *fvo,
              double complex *vooo, double complex *cache1, void **cache,
              int *permute_idx, double fac)
{
        int nooo = nocc * nocc * nocc;
        int *idx0 = permute_idx;
        int *idx1 = idx0 + nooo;
        int *idx2 = idx1 + nooo;
        int *idx3 = idx2 + nooo;
        int *idx4 = idx3 + nooo;
        int *idx5 = idx4 + nooo;
        double complex *v0 = cache1;
        double complex *w0 = v0 + nooo;
        double complex *z0 = w0 + nooo;
        double complex *cache2 = z0 + nooo;
        double complex *wtmp = z0;
        int i;

        for (i = 0; i < nooo; i++) {
                w0[i] = 0;
                v0[i] = 0;
        }

        zget_wv(w0, v0, wtmp, fvo, vooo, cache[0], t1T, t2T, nocc, nvir, a, b, c, idx0);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[1], t1T, t2T, nocc, nvir, a, c, b, idx1);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[2], t1T, t2T, nocc, nvir, b, a, c, idx2);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[3], t1T, t2T, nocc, nvir, b, c, a, idx3);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[4], t1T, t2T, nocc, nvir, c, a, b, idx4);
        zget_wv(w0, v0, wtmp, fvo, vooo, cache[5], t1T, t2T, nocc, nvir, c, b, a, idx5);
        for (i = 0; i < nooo; i++) {
            v0[i] += w0[i];
        }

        double complex et;
        if (a == c) {
                et = _ccsd_t_zget_energy_lo(w0, v0, mo_energy,
                                            cache2, ulo, nlo, // <--- extra args
                                            nocc, a, b, c, 1./6);
        } else if (a == b || b == c) {
                et = _ccsd_t_zget_energy_lo(w0, v0, mo_energy,
                                            cache2, ulo, nlo, // <--- extra args
                                            nocc, a, b, c, .5);
        } else {
                et = _ccsd_t_zget_energy_lo(w0, v0, mo_energy,
                                            cache2, ulo, nlo, // <--- extra args
                                            nocc, a, b, c, 1.);
        }
        return et;
}

void CCsd_t_zcontract_lo(double complex *e_tot,
                         double *mo_energy, double complex *t1T, double complex *t2T,
                         double complex *vooo, double complex *fvo,
                         double complex *ulo, int nlo, // <--- extra args
                         int nocc, int nvir, int a0, int a1, int b0, int b1,
                         int nirrep, int *o_ir_loc, int *v_ir_loc,
                         int *oo_ir_loc, int *orbsym,
                         void *cache_row_a, void *cache_col_a,
                         void *cache_row_b, void *cache_col_b)
{
        int da = a1 - a0;
        int db = b1 - b0;
        CacheJob *jobs = malloc(sizeof(CacheJob) * da*db*b1);
        size_t njobs = _ccsd_t_gen_jobs(jobs, nocc, nvir, a0, a1, b0, b1,
                                        cache_row_a, cache_col_a,
                                        cache_row_b, cache_col_b,
                                        sizeof(double complex));
        int *permute_idx = malloc(sizeof(int) * nocc*nocc*nocc * 6);
        _make_permute_indices(permute_idx, nocc);
#pragma omp parallel default(none) \
        shared(njobs, nocc, nvir, nlo, ulo, mo_energy, t1T, t2T, nirrep, o_ir_loc, \
               v_ir_loc, oo_ir_loc, orbsym, vooo, fvo, jobs, e_tot, permute_idx)
{
        int a, b, c;
        size_t k;
        // extra 6*nocc for tij and vij in :func:`_ccsd_t_zget_energy_lo`
        double complex *cache1 = malloc(sizeof(double complex) * (nocc*nocc*nocc*3+2+6*nocc));
        double complex *t1Thalf = malloc(sizeof(double complex) * nvir*nocc * 2);
        double complex *fvohalf = t1Thalf + nvir*nocc;
        for (k = 0; k < nvir*nocc; k++) {
                t1Thalf[k] = t1T[k] * .5;
                fvohalf[k] = fvo[k] * .5;
        }
        double complex e = 0;
#pragma omp for schedule (dynamic, 4)
        for (k = 0; k < njobs; k++) {
                a = jobs[k].a;
                b = jobs[k].b;
                c = jobs[k].c;
                e += zcontract6_lo(nocc, nvir, a, b, c, mo_energy, t1Thalf, t2T,
                                   ulo, nlo, // <--- extra args
                                   nirrep, o_ir_loc, v_ir_loc, oo_ir_loc, orbsym,
                                   fvohalf, vooo, cache1, jobs[k].cache, permute_idx,
                                   1.0);
        }
        free(t1Thalf);
        free(cache1);
#pragma omp critical
        *e_tot += e;
}
        free(permute_idx);
        free(jobs);
}
