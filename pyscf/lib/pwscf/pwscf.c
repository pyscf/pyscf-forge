#include <stdio.h>
#include <math.h>
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

// f is the function shape (n[0], n[1], n[2])
// c is the 3x3 rotation matrix
// assumes that each coord in fin maps to 1 coord in fout,
// which should always be the case.
// Otherwise there will be race conditions.
// This function essentially applies
// fout += rot(wt * fin)
void add_rotated_realspace_func(const double *fin, double *fout, const int *n,
                                const int *c, const double wt) {
#pragma omp parallel
{
    const size_t N[3] = {n[0], n[1], n[2]};
    size_t indi;
    size_t indo;
    int xi, yi, zi;
    int xo, yo, zo;
#pragma omp for schedule(static)
    for (xi = 0; xi < n[0]; xi++) {
        indi = xi * N[1] * N[2];
        for (yi = 0; yi < n[1]; yi++) {
            for (zi = 0; zi < n[2]; zi++) {
                xo = modulo(c[0] * xi + c[1] * yi + c[2] * zi, n[0]);
                yo = modulo(c[3] * xi + c[4] * yi + c[5] * zi, n[1]);
                zo = modulo(c[6] * xi + c[7] * yi + c[8] * zi, n[2]);
                indo = zo + N[2] * (yo + N[1] * xo);
                fout[indo] += wt * fin[indi];
                indi++;
            }
        }
    }
}
}

