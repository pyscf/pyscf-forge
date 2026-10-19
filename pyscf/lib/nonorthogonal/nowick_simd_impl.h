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
 * Packet types and operations for the SIMD kernels, included by
 * nowick_avx2.c and nowick_avx512.c with NLANE and ISA(name) defined.
 * Real packets are GCC vector extensions; complex packets hold separate
 * real and imaginary vectors.
 */

#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "nowick.h"

#define ALWAYS_INLINE   static inline __attribute__((always_inline))

typedef double ISA(vd) __attribute__((vector_size(8 * NLANE)));
typedef struct {
        ISA(vd) re;
        ISA(vd) im;
} ISA(vz);

void NOWICKdhamiltonian(double *h, double *s, const NOWICKEnv *env,
                        const int *dx, const int *dw);
void NOWICKzhamiltonian(double complex *h, double complex *s, const NOWICKEnv *env,
                        const int *dx, const int *dw);

static void nw_labels(int *rows, int *cols, const int *hx, const int *px, int rx,
                      const int *hw, const int *pw, int rw, int nocc, int nmo)
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

ALWAYS_INLINE ISA(vd) ISA(vd_load)(const double *p)
{
        ISA(vd) v;
        memcpy(&v, p, sizeof(v));
        return v;
}

ALWAYS_INLINE void ISA(vd_store)(double *p, ISA(vd) v)
{
        memcpy(p, &v, sizeof(v));
}

ALWAYS_INLINE ISA(vz) ISA(vz_add)(ISA(vz) a, ISA(vz) b)
{
        ISA(vz) c = {a.re + b.re, a.im + b.im};
        return c;
}

ALWAYS_INLINE ISA(vz) ISA(vz_sub)(ISA(vz) a, ISA(vz) b)
{
        ISA(vz) c = {a.re - b.re, a.im - b.im};
        return c;
}

ALWAYS_INLINE ISA(vz) ISA(vz_mul)(ISA(vz) a, ISA(vz) b)
{
        ISA(vz) c = {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
        return c;
}

ALWAYS_INLINE ISA(vz) ISA(vz_scale)(ISA(vz) a, double s)
{
        ISA(vz) c = {a.re * s, a.im * s};
        return c;
}

ALWAYS_INLINE ISA(vz) ISA(vz_bcast)(double complex x)
{
        ISA(vz) c = {(ISA(vd)){0} + creal(x), (ISA(vd)){0} + cimag(x)};
        return c;
}

ALWAYS_INLINE ISA(vz) ISA(vz_load)(const double complex *p)
{
        double re[NLANE], im[NLANE];
        ISA(vz) c;
        int k;
        for (k = 0; k < NLANE; k++) {
                re[k] = creal(p[k]);
                im[k] = cimag(p[k]);
        }
        c.re = ISA(vd_load)(re);
        c.im = ISA(vd_load)(im);
        return c;
}

ALWAYS_INLINE void ISA(vz_store)(double complex *p, ISA(vz) v)
{
        int k;
        for (k = 0; k < NLANE; k++) {
                p[k] = v.re[k] + I * v.im[k];
        }
}

/* real packets */
#define DT              double
#define PT              ISA(vd)
#define FN(name)        ISA(NOWICKd##name)
#define HS_SCALAR       NOWICKdhamiltonian
#define PZERO           ((ISA(vd)){0})
#define PONE            ((ISA(vd)){0} + 1.)
#define PADD(a, b)      ((a) + (b))
#define PSUB(a, b)      ((a) - (b))
#define PMUL(a, b)      ((a) * (b))
#define PSCALE(a, s)    ((a) * (s))
#define PBCAST(x)       ((ISA(vd)){0} + (x))
#define PLOAD(p)        ISA(vd_load)(p)
#define PSTORE(p, v)    ISA(vd_store)(p, v)
#include "nowick_simd_tmpl.h"
#undef DT
#undef PT
#undef FN
#undef HS_SCALAR
#undef PZERO
#undef PONE
#undef PADD
#undef PSUB
#undef PMUL
#undef PSCALE
#undef PBCAST
#undef PLOAD
#undef PSTORE

/* complex packets */
#define DT              double complex
#define PT              ISA(vz)
#define FN(name)        ISA(NOWICKz##name)
#define HS_SCALAR       NOWICKzhamiltonian
#define PZERO           ((ISA(vz)){(ISA(vd)){0}, (ISA(vd)){0}})
#define PONE            ((ISA(vz)){(ISA(vd)){0} + 1., (ISA(vd)){0}})
#define PADD(a, b)      ISA(vz_add)(a, b)
#define PSUB(a, b)      ISA(vz_sub)(a, b)
#define PMUL(a, b)      ISA(vz_mul)(a, b)
#define PSCALE(a, s)    ISA(vz_scale)(a, s)
#define PBCAST(x)       ISA(vz_bcast)(x)
#define PLOAD(p)        ISA(vz_load)(p)
#define PSTORE(p, v)    ISA(vz_store)(p, v)
#include "nowick_simd_tmpl.h"
