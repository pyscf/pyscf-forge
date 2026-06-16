// !/usr/bin/env python
//  Copyright 2014-2024 The PySCF Developers. All Rights Reserved.

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Authors:
//           Shuhang Li <shuhangli98@gmail.com>
//           Zijun Zhao <brian.zhaozijun@gmail.com>


#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "np_helper/np_helper.h"

#define MACHEPS 1e-9
#define TAYLOR_THRES 1e-3

double taylor_exp(double z)
{
        int n = (int)(0.5 * (15.0 / TAYLOR_THRES + 1)) + 1;
        if (n > 0) {
                double value = z;
                double tmp = z;
                for (int x = 0; x < n-1; x++) {
                        tmp *= -1.0 * z * z / (x + 2);
                        value += tmp;
                }
                return value;
        } else {
                return 0.0;
        }
}

double regularized_denominator(double x, double s)
{
        double z = sqrt(s) * x;
        if (fabs(z) <= MACHEPS) {
                return taylor_exp(z) * sqrt(s);
        } else {
                return (1. - exp(-s * x * x)) / x;
        }
}

void compute_T2_block(double *t2, double *ei, double *ej, double *ea, double *eb, double flow_param, int ni, int nj, int na, int nb)
{
#pragma omp parallel
{
        int i,j,a,b;
        double* pt2;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (i = 0; i < ni; i++) {
                for (j = 0; j < nj; j++) {
                        for (a = 0; a < na; a++) {
                                for (b = 0; b < nb; b++) {
                                        double denom = ei[i] + ej[j] - ea[a] - eb[b];
                                        pt2 = t2 + i * nj * na * nb + j * na * nb + a * nb + b;;
                                        *pt2 *= regularized_denominator(denom,flow_param);
                                }
                        }
                }
        }
}
}

void renormalize_V(double *V, double *ei, double *ej, double *ea, double *eb, double flow_param, int ni, int nj, int na, int nb)
{
#pragma omp parallel
{
        int i,j,a,b;
        double *pV;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (i = 0; i < ni; i++) {
                for (j = 0; j < nj; j++) {
                        for (a = 0; a < na; a++) {
                                for (b = 0; b < nb; b++) {
                                        double denom = ei[i] + ej[j] - ea[a] - eb[b];
                                        pV = V + i * nj * na * nb + j * na * nb + a * nb + b;
                                        *pV *= (1. + exp(-flow_param*denom*denom));
                                }
                        }
                }
        }
}
}

void renormalize_F(double *F, double *ei, double *ea, double flow_param, int ni, int na)
{
#pragma omp parallel
{
        int i,a;
        double *pF;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (a = 0; a < na; a++) {
                for (i = 0; i < ni; i++) {
                        double denom = ei[i] - ea[a];
                        pF = F + a*ni + i;
                        *pF *= exp(-flow_param*denom*denom);
                }
        }
}
}

void compute_T1(double *t1, double *ei, double *ea, double flow_param, int ni, int na)
{
#pragma omp parallel
{
        int i, a;
        double *pt1;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (i = 0; i < ni; i++) {
                for (a = 0; a < na; a++) {
                        double denom = ei[i] - ea[a];
                        pt1 = t1 + i * na + a;
                        *pt1 *= regularized_denominator(denom,flow_param);
                }
        }
}
}

void renormalize_CCVV_batch(double *Jcc, double ecc, double *ev, double flow_param, int nv)
{
#pragma omp parallel
{
        int r,s;
        double *pJcc;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (r = 0; r < nv; r++) {
                for (s = 0; s < nv; s++) {
                        double denom = ecc - ev[r] + ev[s];
                        pJcc = Jcc + r * nv + s;
                        *pJcc *= (1. + exp(-flow_param*denom*denom)) * regularized_denominator(denom,flow_param);
                }
        }
}
}

void renormalize_CCVV(double *Jvvc, double ec_, double *ev, double *ec, double flow_param, int nv, int nc)
{
#pragma omp parallel
{
        int q,r,s;
        double *pJvvc;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (q = 0; q < nv; q++) {
                for (r = 0; r < nv; r++) {
                        for (s = 0; s < nc; s++){
                                double denom = ec_ + ec[s] - ev[q] - ev[r];
                                pJvvc = Jvvc + q * nv * nc + r * nc + s;
                                *pJvvc *= (1. + exp(-flow_param*denom*denom)) * regularized_denominator(denom,flow_param);
                        }
                }
        }
}
}

void renormalize_CAVV(double *JKvva, double *Jvva, double e_, double *ev, double *ea, double flow_param, int nv, int na)
{
#pragma omp parallel
{
        int e,f,u;
        double *pJKvva, *pJvva;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (e = 0; e < nv; e++){
                for (f = 0; f < nv; f++){
                        for (u = 0; u < na; u++){
                                double denom = e_ + ea[u] - ev[e] - ev[f];
                                pJKvva = JKvva + e * nv * na + f * na + u;
                                pJvva = Jvva + e * nv * na + f * na + u;
                                *pJKvva *= regularized_denominator(denom,flow_param);
                                *pJvva *= (1. + exp(-flow_param*denom*denom));
                        }
                }
        }
}
}

void renormalize_CCAV(double *JKva, double *Jva, double e_, double *ev, double *ea, double flow_param, int nv, int na)
{
#pragma omp parallel
{
        int e,u;
        double *pJKva, *pJva;
#pragma omp for schedule(dynamic, 10) collapse(2)
        for (e = 0; e < nv; e++){
                for (u = 0; u < na; u++){
                        double denom = e_ - ea[u] - ev[e];
                        pJKva = JKva + e * na + u;
                        pJva = Jva + e * na + u;
                        *pJKva *= regularized_denominator(denom,flow_param);
                        *pJva *= (1. + exp(-flow_param*denom*denom));
                }
        }
}
}

