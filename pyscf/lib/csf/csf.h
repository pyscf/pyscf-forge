#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#if defined __cplusplus
extern "C" {
#endif

void dsbmv_(const char *uplo,
            const int *n,
            const int *k,
            const double *alpha,
            const double *a,
            const int *lda,
            const double *x,
            const int *incx,
            const double *beta,
            double *y,
            const int *incy);

#if defined __cplusplus
} // end extern "C"
#endif

#ifndef MAX_PARTICLE
#define MAX_PARTICLE 64
#endif

void FCICSFmakeS2mat (double * S2mat, uint64_t * detstr, size_t ndet, int nspin, int twoMS);
unsigned int _count_set_bits (uint64_t str);
unsigned int _get_occ (unsigned int i, uint64_t dconfstr, uint64_t sconfstr);
unsigned int _get_spinindex (unsigned int i, uint64_t dconfstr, uint64_t sconfstr);
int _get_twoM (uint64_t sconfstr, uint64_t detstr);
unsigned int _get_twoS_running (uint64_t coupstr, unsigned int i, unsigned int nspin);
double _get_vcc (uint64_t coupstr, unsigned int i, unsigned int j, unsigned int nspin);
double _get_xdiag (uint64_t coupstr, unsigned int i, unsigned int j, unsigned int nspin);
double _get_x (uint64_t brastr, uint64_t ketstr,
               unsigned int t, unsigned int q, unsigned int r, unsigned int p,
               int nt, int nq, int nr, int np,
               unsigned int nspin);
double _get_xcore (unsigned int * twoSk, unsigned int * twoSb,
                   unsigned int r, unsigned int q,
                   int nr, int nq, bool pphh);

double _get_szfac (uint64_t coupstr, unsigned int i, unsigned int nspin, int twoM);
void FCICSFhdiag (double * hdiag_csf, double * hdiag_det,
                  double * h1e_s, double * eri,
                  uint64_t * dconfstrs, uint64_t * sconfstrs,
                  uint64_t * coupstrs, uint64_t * detstrs,
                  double * wrk,
                  size_t ndoub, size_t nsing,
                  size_t ncoup, size_t ndet,
                  unsigned int norb);

void FCICSFddstrs2csdstrs (uint64_t * csdstrs, uint64_t * ddstrs, size_t nstr, int norb, int neleca, int nelecb);
void FCICSFcsdstrs2ddstrs (uint64_t * ddstrs, uint64_t * csdstrs, size_t nstr, int norb, int neleca, int nelecb);
void FCICSFmakecsf (double * umat, uint64_t * detstr, uint64_t * coupstr, int nspin, size_t ndet, size_t ncoup, int twoS, int twoMS);
void FCICSFgetscstrs (uint64_t * scstrs, bool * mask, size_t nstr, int nspin);
void FCICSFstrs2addr (int * addrs, uint64_t * strings, size_t nstr, int * gentable_ravel, int nspin, int twoS);
void FCICSFaddrs2str (uint64_t * strings, int * addrs, size_t nstr, int * gentable_ravel, int nspin, int twoS);
