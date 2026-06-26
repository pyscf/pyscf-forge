#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include "vhf/fblas.h"
#include "csf.h"

#ifndef MAX_PARTICLE
#define MAX_PARTICLE = 64
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
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

unsigned int _count_set_bits (uint64_t str)
{
    unsigned int n = 0;
    while (str){
        str &= (str - 1);
        n++;
    }
    return n;
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

unsigned int _get_spin (unsigned int i, uint64_t dconfstr, uint64_t sconfstr, uint64_t detstr)
{
    unsigned int ni = _get_occ (i, dconfstr, sconfstr);
    if (ni != 1){ return 0; }
    ni = _get_spinindex (i, dconfstr, sconfstr);
    ni = (detstr & (1ULL << ni)) >> ni;
    return ni;
}

unsigned int _get_spinindex (unsigned int i, uint64_t dconfstr, uint64_t sconfstr)
{
    unsigned int j = i;
    for (unsigned int k = 0; k < i; k++){
        if ((1ULL << k) & dconfstr){ j--; }
    }
    i = j;
    for (unsigned int k = 0; k < i; k++){
        if (((1ULL << k) & sconfstr) == 0){ j--; }
    }
    return j;
}

int _get_twoM (uint64_t sconfstr, uint64_t detstr)
{
    int nspin = _count_set_bits (sconfstr);
    int twoM = _count_set_bits (detstr);
    assert (twoM <= nspin);
    twoM = nspin - 2*twoM;
    return -twoM;
}

unsigned int _get_twoS_running (uint64_t coupstr, unsigned int i, unsigned int nspin)
{
    if (nspin == 0){ return 0; }
    assert (nspin - i >= 0);
    uint64_t n = coupstr;
    unsigned int twoS = _count_set_bits (coupstr);
    assert (nspin >= twoS); // S >= 0
    twoS = (2*twoS) - nspin;
    // in range 0 < j <= i;
    //    add 1/2 for every unset bit
    //    subtract 1/2 for every set bit
    twoS += i;
    // unset all bits i or more places from the edge
    n = coupstr ^ (coupstr & ((1ULL << (nspin-i)) - 1));
    n = _count_set_bits (n);
    assert (n <= i);
    twoS -= 2 * n;
    return twoS;
}

double _get_wigner_6j_j41h (unsigned int j1, unsigned int j2, unsigned int j3, unsigned int j5, unsigned int j6)
{
    /* Get Wigner 6j symbols where j4 is always 1/2
       i.e., eqs 19 and 20 of Drake & Schlesinger
       On input all js are multiplied by 2 */
    unsigned int j;
    int fac;
    double num = 0.0;
    unsigned int denom = 1;
    // Switch around j5 = j3 + 1/2 case
    if ((j5 == (j3+1)) && (j6 == (j2-1))){
        j = j2;
        j2 = j3;
        j3 = j;
        j = j5;
        j5 = j6;
        j6 = j;
    }
    fac = (j1+j2+j3 % 4 == 0) ? 1 : -1;
    if (j5 != (j3-1)){ fac = 0; }
    if (j6 == (j2+1)){
        num = (double) ((j1+j3-j2) * (j1+j2-j3+2)) / 2;
        denom = (j2+1) * (j2+2) * j3 * (j3+1);
    } else if (j6 == (j2-1)){
        num = (double) ((j1+j2+j3+2) * (j2+j3-j1)) / 2;
        denom = j2 * (j2+1) * j3 * (j3+1);
    }
    num = fac * num / denom;
    return num;
}

double _get_vcc (uint64_t coupstr, unsigned int i, unsigned int j, unsigned int nspin)
{
    /* Compute
        -<Eijji>
        = <Eii> - <Eij Eji>
        = 1 - (3xdiag - 1/2)
        = 1/2 - 3xdiag
       where i > j (although see below) using
       Drake & Schlesinger, PRA 15 1990 (1977) (DOI:10.1103/PhysRevA.15.1990)
       "xdiag" is the irreducible graph
    */
    return 0.5 - 3*_get_xdiag (coupstr, i, j, nspin);
}

double _get_szfac (uint64_t coupstr, unsigned int i, unsigned int nspin, int twoM)
{
    unsigned int twoS = _get_twoS_running (coupstr, 0, nspin);
    double szfac = _get_xdiag (coupstr, nspin, i, nspin);
    szfac *= sqrt (1.5);
    szfac *= twoM; // WHY? UNCLEAR FACTOR OF 2 * .5;
    if (twoS > 0){
        szfac /= sqrt (twoS*(twoS+2)*.25);
    }
    return szfac;
}

double _get_xdiag (uint64_t coupstr, unsigned int i, unsigned int j, unsigned int nspin)
{
    double xdiag = 1.0;
    unsigned int twoS1, twoS0, twoS;
    double rat;
    int parity = 0;
    unsigned int k;
    // Drake & Schlesinger ``reverse the order of counting'' so we have to do that here
    // If I just bitshift the coupstr instead I'm not sure that the CSFs mean the same thing
    // They have S0 = S and SN = 0
    assert (i <= nspin);
    assert (j < nspin);
    i = nspin - i;
    j = nspin - j;
    assert (j>=i);
    // because the highest possible value of i is nspin - 1 and we want to start at 1 and go
    // through nspin inclusively.

    // i
    //     -1**[S(i) + S'(i-1) - 1/2]
    //     *
    //     sqrt[2S'(i) + 1]
    //     *
    //     { S'(i)  S(i)  1      }
    //     { 1/2    1/2   S(i-1) }
    // We just skip this diagram when we are doing Sz, which corresponds to i == 0
    twoS0 = _get_twoS_running (coupstr, i, nspin);
    if (i>0){
        twoS1 = _get_twoS_running (coupstr, i-1, nspin);

        parity += twoS0 + twoS1 - 1; // graph signs
        parity += (2*twoS0 + 2); // Wigner 6j sign; the exponent is multiplied by 2
        parity = parity % 4; // remember everything is *2 until the very end

        if (twoS1 > twoS0){ // S(i-1) = S(i) + 1/2
            rat = ((double) (twoS0)) / ((twoS0+1) * (twoS0+2) * 6);
        } else { // S(i-1) = S(i) - 1/2
            rat = ((double) (twoS0+2)) / (twoS0 * (twoS0+1) * 6);
        }
        rat *= twoS0 + 1; // normalization
        xdiag *= sqrt (rat);
    }

    // i+1, i+2, ... j-2, j-1
    //
    //     -1**[S(k) + S'(k+1) - 1/2]
    //     *
    //     sqrt[(2S(k) + 1)(2S'(k+1) + 1)]
    //     *
    //     { 1    S'(k)   S(k)    }
    //     { 1/2  S(k+1)  S'(k+1) }
    for (k=i+1; k < j; k++){
        twoS1 = twoS0;
        twoS0 = _get_twoS_running (coupstr, k, nspin);
        twoS = MIN (twoS0, twoS1);

        parity += twoS0 + twoS1 - 1; // graph signs
        parity += 2*twoS; // Wigner 6j sign
        parity = parity % 4;

        rat = ((double) ((twoS+3) * twoS));
        rat *= twoS0+1; // normalization
        rat *= twoS1+1; // normalization
        rat = sqrt (rat);
        rat /= ((twoS+1)*(twoS+2));
        xdiag *= rat;
    }
    
    // j
    //
    //     -1**[S'(j-1) + S(j) - 1/2]
    //     *
    //     sqrt[2S(j-1) + 1]
    //     *
    //     { S'(j-1)  S(j-1)  1    }
    //     { 1/2      1/2     S(j) }
    twoS1 = twoS0;
    twoS0 = _get_twoS_running (coupstr, j, nspin);

    parity += twoS0 + twoS1 - 1; // graph signs
    parity += (2*twoS1 + 2); // Wigner 6j signs
    parity = parity % 4;

    if (twoS1 > twoS0){ // S(i-1) = S(i) + 1/2
        rat = ((double) (twoS1+2)) / (twoS1 * (twoS1+1) * 6);
    } else { // S(i-1) = S(i) - 1/2
        rat = ((double) (twoS1)) / ((twoS1+1) * (twoS1+2) * 6);
    }
    rat *= twoS1+1; // normalization
    xdiag *= sqrt (rat);

    // Final sign computation
    assert ((parity % 2)==0);
    parity = parity / 2;
    if ((parity % 2) == 1){ xdiag = -xdiag; }

    return xdiag;
}

void FCICSFhdiag (double * hdiag_csf, double * hdiag_det,
                  double * h1e_s, double * eri,
                  uint64_t * dconfstrs, uint64_t * sconfstrs,
                  uint64_t * coupstrs, uint64_t * detstrs,
                  double * wrk, 
                  size_t ndoub, size_t nsing,
                  size_t ncoup, size_t ndet,
                  unsigned int norb)
{
/* 
    Output:
        hdiag_csf : array of shape (ndoub,nsing,ncoup)

    Input:
        hdiag_det : array of shape (ndoub,nsing,ndet)
            1-electron + Coulomb energies of the determinants (i.e., excluding exchange)
        h1e_s : array of shape (norb)
            Diagonal elements of spin potential
        eri : array of shape (norb,norb,norb,norb)
            (ij|kl) two-electron integrals
        dconfstrs : array of shape (ndoub,)
            Strings for doubly-occupied orbitals
        sconfstrs : array of shape (nsing,)
            Strings for singly-occupied orbitals in the non-doubly-occupied subspace
        coupstrs : array of shape (ncoup,)
            Strings for S coupling in CSFs
        detstrs : array of shape (ndet,)
            Strings for M state in determinants

    Buffer:
        wrk : array of shape (ndoub,nsing)
*/
const size_t nconf = ndoub * nsing;
int twoM = _get_twoM (sconfstrs[0], detstrs[0]);
#pragma omp parallel default(shared)
{

    size_t iconf, idoub, ising, icoup, icoupconf;
    double fac;
    unsigned int ni, nj, si, sj, idxi, idx;
    unsigned int nspin;
    uint64_t dconfstr, sconfstr;

// Subtract SOMO exchange terms from hdiag_det
#pragma omp for schedule(static)
    for (iconf = 0; iconf < nconf; iconf++){
        idoub = iconf / nsing;
        ising = iconf % nsing;
        dconfstr = dconfstrs[idoub];
        sconfstr = sconfstrs[ising];
        wrk[iconf] = hdiag_det[iconf*ndet];
        for (unsigned int i = 0; i < norb; i++){
            ni = _get_occ (i, dconfstr, sconfstr);
            if (ni != 1){ continue ; }
            si = _get_spin (i, dconfstr, sconfstr, detstrs[0]);
            wrk[iconf] -= si ? h1e_s[i] : -h1e_s[i];
            idxi = i*norb*norb*norb + i;
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (j, dconfstr, sconfstr);
                if (nj != 1){ continue ; }
                sj = _get_spin (j, dconfstr, sconfstr, detstrs[0]);
                idx = idxi + j*norb*(norb+1);
                if (si==sj){
                    wrk[iconf] += eri[idx];
                }
            }
        } // All the other exchange terms should be identical for all determinants
    }

#pragma omp for schedule(static)
    for (icoupconf = 0; icoupconf < nconf*ncoup; icoupconf++){
        iconf = icoupconf / ncoup;
        icoup = icoupconf % ncoup;
        idoub = iconf / nsing;
        ising = iconf % nsing;
        dconfstr = dconfstrs[idoub];
        sconfstr = sconfstrs[ising];
        hdiag_csf[icoupconf] = wrk[iconf];
        nspin = _count_set_bits (sconfstr);
        for (unsigned int i = 0; i < norb; i++){
            ni = _get_occ (i, dconfstr, sconfstr);
            if (ni != 1){ continue; }
            si = _get_spinindex (i, dconfstr, sconfstr);
            fac = _get_szfac (coupstrs[icoup], si, nspin, twoM);
            hdiag_csf[icoupconf] += fac * h1e_s[i];
            idxi = i*norb*norb*norb + i;
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (j, dconfstr, sconfstr);
                if (nj != 1){ continue; }
                sj = _get_spinindex (j, dconfstr, sconfstr);
                fac = _get_vcc (coupstrs[icoup], si, sj, nspin);
                idx = idxi + j*norb*(norb+1);
                hdiag_csf[icoupconf] -= fac * eri[idx];
            }
        }
    }
}
}



