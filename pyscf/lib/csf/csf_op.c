#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include "vhf/fblas.h"
#include "csf.h"

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
#define MAX_PARTICLE = 64
#endif

void FCICSFmakeS2mat (double * S2mat, uint64_t * detstr, size_t ndet, int nspin, int twoMS)
{

    size_t idet, jdet;
    int nflip, iflip, osgn, sgn;
    uint64_t flipdet;
    double sz2 = (double) twoMS * twoMS / 4;
    double diag = sz2 + (double) nspin / 2;

    for (idet = 0; idet < ndet; idet++){ for (jdet = 0; jdet < ndet; jdet++){
        flipdet = detstr[idet] ^ detstr[jdet];
        nflip = 0;
        if (flipdet == 0ULL){
            S2mat[idet*ndet + jdet] = diag;
            continue;
        }
        osgn = -1;
        sgn = 1;
        for (iflip = 0; iflip < nspin; iflip++){
            osgn *= -1;
            if ((1ULL << iflip) & detstr[idet]){ sgn *= osgn; }
            if ((1ULL << iflip) & detstr[jdet]){ sgn *= osgn; }
            if ((1ULL << iflip) & flipdet){ nflip++; }
            if (nflip > 2){ break; }
        }
        if (nflip == 2){
            S2mat[idet*ndet + jdet] = sgn * 1.0;
        }
    }}

}

void FCICSFhdiag (double * hdiag, double * hdiag_det, double * eri, uint64_t * astrs, uint64_t * bstrs, unsigned int norb, size_t nconf, size_t ndet)
{

    size_t ndet_lt = ndet * (ndet+1) / 2;

#pragma omp parallel default(shared)
{

    size_t iconf, idetx, idety, idetconf;
    unsigned int iorb, nexc;
    uint64_t exc_str, somo_str, big_idx1, big_idx2, hdiag_idx_lt, hdiag_idx_ut;
    unsigned int exc[2];
    int sgn, esgn;

#pragma omp for schedule(static) 

    for (idetconf = 0; idetconf < nconf * ndet_lt; idetconf++){
        iconf = idetconf / ndet_lt;
        idety = idetconf % ndet_lt;
        for (idetx = 0; idetx < ndet; idetx++){
            if (idetx < idety){ idety -= idetx + 1; }
            else { break; }
        }
        // Careful with possible integer overflow
        hdiag_idx_lt = ndet;
        hdiag_idx_lt *= ndet;
        hdiag_idx_lt *= iconf;
        hdiag_idx_ut = hdiag_idx_lt;
        big_idx1 = ndet;
        big_idx1 *= idety;
        big_idx2 = ndet;
        big_idx2 *= idetx;
        hdiag_idx_lt += big_idx1;
        hdiag_idx_ut += big_idx2;
        hdiag_idx_lt += idetx;
        hdiag_idx_ut += idety;
        if (idetx == idety){ 
            hdiag[hdiag_idx_lt] = hdiag_det[(iconf*ndet)+idetx];
            continue;
        }
        // Fear of integer overflow is only reasonable for off-diagonal elements of a Hamiltonian matrix
        // It's not reasonable for anything else
        big_idx1 = (ndet*iconf) + idetx;
        big_idx2 = (ndet*iconf) + idety;
        exc_str  = astrs[big_idx1] ^ astrs[big_idx2];
        somo_str = astrs[big_idx1] ^ bstrs[big_idx1];
        nexc = 0; esgn = 1; sgn = -1;
        for (iorb = 0; iorb < norb; iorb++){
            if (somo_str & 1ULL << iorb){ esgn *= -1; }
            if (exc_str & 1ULL << iorb){
                if (nexc < 2){ exc[nexc] = iorb; }
                nexc++;
                if (nexc > 2){ break; }
                sgn *= esgn;
            }
        } 
        if (nexc > 2){ continue; }
        assert (nexc == 2);

        // Fear of integer overflow is only reasonable for off-diagonal elements of a Hamiltonian matrix
        // It's not reasonable for anything else
        big_idx1 = exc[0]*norb*norb*norb + exc[1]*norb*norb + exc[1]*norb + exc[0];
        hdiag[hdiag_idx_lt] = sgn * eri[big_idx1];
        hdiag[hdiag_idx_ut] = hdiag[hdiag_idx_lt];
    }

}
}

unsigned int _get_occ (unsigned int i, uint64_t dconfstr, uint64_t sconfstr)
{
    if ((1ULL << i) & dconfstr){ return 2; }
    unsigned int j = i;
    for (unsigned int k = 0; k < i; k++){
        if ((1ULL << k) & dconfstr){ j--; }
    }
    if ((1ULL << j) & sconfstr){ return 1; }
    else { return 0; }
}

unsigned int _get_spinindex (unsigned int i, uint64_t dconfstr, uint64_t sconfstr)
{
    unsigned int j = i;
    for (unsigned int k = 0; k < i; k++){
        if ((1ULL << k) & dconfstr){ j--; }
    }
    i = j;
    j = 0;
    for (unsigned int k = 0; k < MAX_PARTICLE; k++){
        if ((1ULL << k) & sconfstr){ j++; }
        if (j == i){ break; }
    }
    return j;
}

unsigned int _get_twoS_running (uint64_t coupstr, unsigned int i, unsigned int nspin)
{
    assert (nspin - i >= 0);
    unsigned int n = coupstr;
    unsigned int twoS = 0;
    // First determine twoS
    while (n){
        n &= (n - 1);
        twoS++;
    }
    // unset all bits i or more places from the left edge
    n = (coupstr & (1ULL << (nspin-i))) - 1;
    // Subtract 1/2 from S for every remaining set bit
    while (n){
        n &= (n - 1);
        twoS--;
    }
    return twoS;
}

double _get_vcc (uint64_t coupstr, unsigned int i, unsigned int j, unsigned int nspin)
{
    // Drake & Schlesinger ``reverse the order of counting'' so we have to do that here
    // If I just bitshift the coupstr instead I'm not sure that the CSFs mean the same thing
    // They have S0 = S and SN = 0
    i = nspin - i;
    j = nspin - j;
    // because the highest possible value of i is nspin - 1 and we want to start at 1 and go
    // through nspin inclusively.

    // I don't want to pass twoS
    unsigned int twoS = 0;
    for (unsigned int p = 0; p < nspin ; p++){
        if ((1ULL << p) & coupstr){ twoS++; }
    }

    return 1.0;
}

void FCICSFhdiag_o1 (double * hdiag_csf, double * hcoul_det, double * keri,
                     uint64_t * dconfstrs, uint64_t * sconfstrs,
                     uint64_t * coupstrs, uint64_t * detstrs, 
                     size_t nconf, size_t ncoup, size_t ndet,
                     unsigned int norb, unsigned int npair, unsigned int nspin,
                     unsigned int twoS, int twoMS, double * wrk)
{
    const int izero = 0;
    const int ione = 1;
    const double dzero = 0.0;
    const double done = 1.0;
#pragma omp parallel default(shared)
{

    size_t iconf, icoup, iconfcoup;
    size_t last_icoup = ncoup;
    unsigned int ithread = omp_get_thread_num ();
    double * cgbuf0 = wrk + (ithread * (2*ndet));
    double * cgbuf1 = cgbuf0 + ndet;
    double vcc;
    unsigned int ni, nj;
    int narg;

#pragma omp for schedule(static)
    for (iconfcoup = 0; iconfcoup < nconf * ncoup; iconfcoup++){
        icoup = iconfcoup / nconf;
        iconf = iconfcoup % nconf;
        if (last_icoup != icoup){
            FCICSFmakecsf (cgbuf0, detstrs, coupstrs+icoup, nspin, ndet, ione, twoS, twoMS);
        }
        narg = (int) ndet;
        dsbmv_("L", &narg, &izero, 
               &done, hcoul_det, &ione, // hcoul_det ...
               cgbuf0, &ione, // ... * cgbuf0 ...
               &dzero, cgbuf1, &ione); // ... -> cgbuf1
        hdiag_csf[iconfcoup] = ddot_(&narg, cgbuf0, &ione, cgbuf1, &ione);
        for (unsigned int i = 0; i < norb; i++){
            ni = _get_occ (i, dconfstrs[iconf], sconfstrs[iconf]);
            if (ni == 0){ continue; }
            // diagonal
            hdiag_csf[iconfcoup] -= .5 * ni * keri[i*norb + i];
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (j, dconfstrs[iconf], sconfstrs[iconf]);
                // There's almost certainly a sign I need to add here
                // On the other hand, maybe not, since this is hDIAG and the bra and the ket
                // would have the same sign
                vcc = 1.0;
                if (ni + nj == 2){
                    vcc = _get_vcc (
                        coupstrs[icoup],
                        _get_spinindex (i, dconfstrs[iconf], sconfstrs[iconf]),
                        _get_spinindex (j, dconfstrs[iconf], sconfstrs[iconf]),
                        nspin
                    );
                }
                hdiag_csf[iconfcoup] -= (ni+nj-2) * vcc * keri[i*norb + j];
            }
        }

    }
}
}



