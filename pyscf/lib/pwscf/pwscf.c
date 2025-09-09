/* Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
  
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
 * Author: Hong-Zhou Ye <osirpt.sun@gmail.com>
 * Author: Kyle Bystrom <kylebystrom@gmail.com>
 */

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "config.h"


void fast_SphBsli0(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = sinh(x) / x;
    }
}
}

void fast_SphBsli1(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = (x*cosh(x) - sinh(x)) / (x*x);
    }
}
}

void fast_SphBsli2(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = ((x*x+3.)*sinh(x) - 3.*x*cosh(x)) / (x*x*x);
    }
}
}

void fast_SphBsli3(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = ((x*x*x+15.*x)*cosh(x) -
                (6.*x*x+15.)*sinh(x)) / (x*x*x*x);
    }
}
}

void fast_SphBslin(double * xs, int n, int l, double * out)
// n: size of xs; l: order
{
    if (l == 0)
        fast_SphBsli0(xs, n, out);
    else if (l == 1)
        fast_SphBsli1(xs, n, out);
    else if (l == 2)
        fast_SphBsli2(xs, n, out);
    else if (l == 3)
        fast_SphBsli3(xs, n, out);
}

inline static int modulo(int i, int j) {
    return (i % j + j) % j;
}

inline static size_t rotated_index(const int *c, const size_t *N,
                                   const int *shift,
                                   int xi, int yi, int zi)
{
    int xo = modulo(c[0] * xi + c[1] * yi + c[2] * zi + shift[0], N[0]);
    int yo = modulo(c[3] * xi + c[4] * yi + c[5] * zi + shift[1], N[1]);
    int zo = modulo(c[6] * xi + c[7] * yi + c[8] * zi + shift[2], N[2]);
    return zo + N[2] * (yo + N[1] * xo);
}

// f is the function shape (n[0], n[1], n[2])
// c is the 3x3 rotation matrix
// assumes that each coord in fin maps to 1 coord in fout,
// which should always be the case.
// Otherwise there will be race conditions.
// This function essentially applies
// fout += rot(wt * fin)
void add_rotated_realspace_func(const double *fin, double *fout, const int *n,
                                const int *c, const double wt)
{
#pragma omp parallel
{
    const size_t N[3] = {n[0], n[1], n[2]};
    const int shift[3] = {0, 0, 0};
    size_t indi;
    size_t indo;
    int xi, yi, zi;
    // int xo, yo, zo;
#pragma omp for schedule(static)
    for (xi = 0; xi < n[0]; xi++) {
        indi = xi * N[1] * N[2];
        for (yi = 0; yi < n[1]; yi++) {
            for (zi = 0; zi < n[2]; zi++) {
                //xo = modulo(c[0] * xi + c[1] * yi + c[2] * zi, n[0]);
                //yo = modulo(c[3] * xi + c[4] * yi + c[5] * zi, n[1]);
                //zo = modulo(c[6] * xi + c[7] * yi + c[8] * zi, n[2]);
                //indo = zo + N[2] * (yo + N[1] * xo);
                indo = rotated_index(c, N, shift, xi, yi, zi);
                fout[indo] += wt * fin[indi];
                indi++;
            }
        }
    }
}
}

void get_rotated_complex_func(const double complex *fin,
                              double complex *fout, const int *n,
                              const int *c, const int *shift) {
#pragma omp parallel
{
    const size_t N[3] = {n[0], n[1], n[2]};
    size_t indi;
    size_t indo;
    int xi, yi, zi;
#pragma omp for schedule(static)
    for (xi = 0; xi < n[0]; xi++) {
        indi = xi * N[1] * N[2];
        for (yi = 0; yi < n[1]; yi++) {
            for (zi = 0; zi < n[2]; zi++) {
                indo = rotated_index(c, N, shift, xi, yi, zi);
                fout[indo] = fin[indi];
                indi++;
            }
        }
    }
}
}

