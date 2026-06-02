#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
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
    printf ("i = %d, dconfstr = %d, sconfstr = %d\n", i, dconfstr, sconfstr);
    unsigned int j = i;
    for (unsigned int k = 0; k < i; k++){
        if ((1ULL << k) & dconfstr){ j--; }
    }
    printf ("j = %d\n", j);
    i = j;
    for (unsigned int k = 0; k < i; k++){
        if (((1ULL << k) & sconfstr) == 0){ j--; }
    }
    printf ("j = %d\n", j);
    return j;
}

unsigned int _get_twoS_running (uint64_t coupstr, unsigned int i, unsigned int nspin)
{
    printf ("nspin = %d, i = %d\n", nspin, i);
    assert (nspin - i >= 0);
    uint64_t n = coupstr;
    unsigned int twoS = _count_set_bits (coupstr);
    printf ("twoS = %d\n", twoS);
    assert (nspin >= twoS); // S >= 0
    twoS = (2*twoS) - nspin;
    // in range 0 < j <= i;
    //    add 1/2 for every unset bit
    //    subtract 1/2 for every set bit
    twoS += i;
    // unset all bits i or more places from the edge
    n = coupstr ^ (coupstr & ((1ULL << (nspin-i)) - 1));
    n = _count_set_bits (n);
    printf ("set bits in range = %d\n", n);
    assert (n <= i);
    twoS -= 2 * n;
    return twoS;
}

double _get_vcc (uint64_t coupstr, unsigned int i, unsigned int j, unsigned int nspin)
{
    /* Compute
        1 + <CSF | Eij Eji | CSF> = 0.5 + "G2"
       where i > j (although see below) using
       Drake & Schlesinger, PRA 15 1990 (1977) (DOI:10.1103/PhysRevA.15.1990)
    */
    double vcc = 0.5;
    double g2 = 1.0;
    unsigned int k;
    // Drake & Schlesinger ``reverse the order of counting'' so we have to do that here
    // If I just bitshift the coupstr instead I'm not sure that the CSFs mean the same thing
    // They have S0 = S and SN = 0
    printf ("coupstr = %d\n", coupstr);
    printf ("i = %d, j = %d, nspin = %d\n", i, j, nspin);
    assert (i < nspin);
    assert (j < nspin);
    i = nspin - i;
    j = nspin - j;
    printf ("redefined as i = %d, j = %d\n", i, j);
    assert (j>=i);
    // because the highest possible value of i is nspin - 1 and we want to start at 1 and go
    // through nspin inclusively.

    // i
    unsigned int twoS1 = _get_twoS_running (coupstr, i-1, nspin);
    printf ("2S(%d) = %d\n", i-1, twoS1);
    unsigned int twoS0 = _get_twoS_running (coupstr, i, nspin);
    printf ("2S(%d) = %d\n", i, twoS0);
    unsigned int twoS;
    double rat = 1;

    int parity = twoS0 + twoS1 - 1; // graph signs
    parity += (2*twoS0 + 2); // Wigner 6j sign; the exponent is multiplied by 2
    parity = parity % 4; // remember everything is *2 until the very end

    if (twoS1 > twoS0){ // S(i-1) = S(i) + 1/2
        rat = ((double) (twoS0)) / ((twoS0+1) * (twoS0+2) * 6);
    } else { // S(i-1) = S(i) - 1/2
        rat = ((double) (twoS0+2)) / (twoS0 * (twoS0+1) * 6);
    }
    rat *= twoS0+1; // normalization
    g2 *= sqrt (rat);
    printf ("g2 = %f\n", g2);

    // i+1, i+2, ... j-2, j-1
    for (k=i+1; k < j; k++){
        twoS1 = twoS0;
        twoS0 = _get_twoS_running (coupstr, k, nspin);
        //printf ("2S(%d) = %d\n", k, twoS0);
        twoS = MAX (twoS0, twoS1);

        parity += twoS0 + twoS1 - 1; // graph signs
        parity += (2*twoS + 2); // Wigner 6j sign
        parity = parity % 4;

        rat = ((double) ((twoS+2) * (twoS-1))) / (twoS0 * (twoS0+1));
        rat *= twoS0+1; // normalization
        g2 *= sqrt (rat);
        printf ("g2 = %f\n", g2);
    }
    
    // j
    twoS1 = twoS0;
    twoS0 = _get_twoS_running (coupstr, j, nspin);
    printf ("2S(%d) = %d\n", j, twoS0);

    parity += twoS0 + twoS1 - 1; // graph signs
    parity += (2*twoS1 + 2); // Wigner 6j signs
    parity = parity % 4;

    if (twoS1 > twoS0){ // S(i-1) = S(i) + 1/2
        rat = ((double) (twoS1+2)) / (twoS1 * (twoS1+1) * 6);
    } else { // S(i-1) = S(i) - 1/2
        rat = ((double) (twoS1)) / ((twoS1+1) * (twoS1+2) * 6);
    }
    rat *= twoS0+1; // normalization
    g2 *= sqrt (rat);
    printf ("g2 = %f\n", g2);

    // Final sign computation
    printf ("parity = %d\n", parity);
    assert ((parity % 2)==0);
    parity = parity / 2;
    if ((parity % 2) == 1){ g2 = -g2; }

    //abort();
    return vcc+g2;
}

void FCICSFhdiag (double * hdiag_csf, double * hdiag_det, double * eri,
                  uint64_t * dconfstrs, uint64_t * sconfstrs,
                  uint64_t * coupstrs, uint64_t * detstrs,
                  double * wrk, 
                  size_t nconf, size_t ncoup, size_t ndet,
                  unsigned int norb)
{
/* 
    Output:
        hdiag_csf : array of shape (nconf,ncoup)

    Input:
        hdiag_det : array of shape (nconf,ndet)
            1-electron + Coulomb energies of the determinants (i.e., excluding exchange)
        eri : array of shape (norb,norb,norb,norb)
            (ij|kl) two-electron integrals
        dconfstrs : array of shape (nconf,)
            Strings for doubly-occupied orbitals
        sconfstrs : array of shape (nconf,)
            Strings for singly-occupied orbitals in the non-doubly-occupied subspace
        coupstrs : array of shape (ncoup,)
            Strings for S coupling in CSFs
        detstrs : array of shape (ndet,)
            Strings for M state in determinants

    Buffer:
        wrk : array of shape (nconf)
*/
#pragma omp parallel default(shared)
{

    size_t iconf, icoup, icoupconf;
    double fac;
    unsigned int ni, nj, si, sj, idx;
    unsigned int nspin;

// Subtract SOMO exchange terms from hdiag_det
#pragma omp for schedule(static)
    for (iconf = 0; iconf < nconf; iconf++){
        wrk[iconf] = hdiag_det[iconf*ndet];
        for (unsigned int i = 1; i < norb; i++){
            ni = _get_occ (i, dconfstrs[iconf], sconfstrs[iconf]);
            if (ni != 1){ continue ; }
            si = _get_spin (i, dconfstrs[iconf], sconfstrs[iconf], detstrs[0]);
            idx = i*norb*norb*norb + i;
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (j, dconfstrs[iconf], sconfstrs[iconf]);
                if (nj != 1){ continue ; }
                sj = _get_spin (j, dconfstrs[iconf], sconfstrs[iconf], detstrs[0]);
                idx += j*norb*(norb+1);
                if (si==sj){
                    wrk[iconf] += eri[idx];
                }
            }
        } // All the other exchange terms should be identical for all determinants
    }

#pragma omp for schedule(static)
    for (icoupconf = 0; icoupconf < nconf * ncoup; icoupconf++){
        iconf = icoupconf / ncoup;
        icoup = icoupconf % ncoup;
        hdiag_csf[icoupconf] = wrk[iconf];
        nspin = _count_set_bits (sconfstrs[iconf]);
        for (unsigned int i = 0; i < norb; i++){
            ni = _get_occ (i, dconfstrs[iconf], sconfstrs[iconf]);
            if (ni != 1){ continue; }
            si = _get_spinindex (i, dconfstrs[iconf], sconfstrs[iconf]);
            idx = i*norb*norb*norb + i;
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (j, dconfstrs[iconf], sconfstrs[iconf]);
                if (nj != 1){ continue; }
                sj = _get_spinindex (j, dconfstrs[iconf], sconfstrs[iconf]);
                printf ("i = %d, j = %d, si = %d, sj = %d\n",i,j,si,sj);
                fac = _get_vcc (coupstrs[icoup], si, sj, nspin);
                idx += j*norb*(norb+1);
                hdiag_csf[icoupconf] -= fac * eri[idx];
            }
        }
    }
}
}



