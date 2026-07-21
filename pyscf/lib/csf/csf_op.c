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
    // Permute 3rd column if necessary
    if ((abs (j2-j6) != 1) || (abs (j3-j5) != 1)){
        j = j6;
        j6 = j3;
        j3 = j;
    }
    // Swap second and 3rd columns if necessary
    if (((j5+1) != j3) && ((j6+1) == j2)){
        j = j2;
        j2 = j3;
        j3 = j;
        j = j5;
        j5 = j6;
        j6 = j;
    }
    // Last try: permute 2nd and 3rd columns. I need to keep track of j5 and j6 for the final check.
    if (((j5+1) != j3)){
        j = j6;
        j6 = j3;
        j3 = j;
        j = j5;
        j5 = j2;
        j2 = j;
    }
    if (((j5+1) != j3) || (abs (j2-j6) != 1)){
        return 0.0;
    }
    assert ((j1+j2+j3) % 2 == 0);
    fac = ((j1+j2+j3) % 4 == 0) ? 1 : -1;
    if (j6 == (j2+1)){
        num = ((double) ((j1+j3-j2) * (j1+j2-j3+2))) * .25;
        denom = (j2+1) * (j2+2) * j3 * (j3+1);
    } else if ((j6+1) == j2){
        num = ((double) ((j1+j2+j3+2) * (j2+j3-j1))) * .25;
        denom = j2 * (j2+1) * j3 * (j3+1);
    }
    num = fac * sqrt (num / denom);
    return num;
}

double CGC_2e_diag (uint64_t coupstr, unsigned int i, unsigned int j, unsigned int nspin)
{
    /* Compute
        <Eijij>
        = <Eij Eji> - <Eii>
        = (-1) * (3xdiag + 1/2) - 1
        = -3xdiag - 1/2
       where i > j (although see below) using
       Drake & Schlesinger, PRA 15 1990 (1977) (DOI:10.1103/PhysRevA.15.1990)
       "xdiag" is the irreducible graph
       -1 in line 2 comes from returning creation operators to their proper order at the end
       I think this accounts for the factor of -1 on the 1/2 in Drake & Schlesinger as well
       I think they arbitrarily put -1 on one of their terms in xcore to make it more
       symmetrical.
    */
    return -3*CGC_2e_X_diag (coupstr, i, j, nspin) - 0.5;
}

double CGC_1s_diag (uint64_t coupstr, unsigned int i, unsigned int nspin, int twoM)
{
    unsigned int twoS = _get_twoS_running (coupstr, 0, nspin);
    double szfac = CGC_2e_X_diag (coupstr, nspin, i, nspin);
    // Sigma_z = ( 1/2  1  1/2 ) * S_z
    //           ( m    0   -m )
    //         = sqrt (2/3) * S_z
    // S_z = sqrt (3/2) * Sigma_z
    szfac *= sqrt (1.5);

    //     -1**[S-M]
    //     *
    //     ( S  1  S )
    //     ( M  0 -M )
    szfac *= twoM; // WHY? UNCLEAR FACTOR OF 2 * .5;
    if (twoS > 0){
        szfac /= sqrt (twoS*(twoS+2)*.25);
    }
    return szfac;
}

double CGC_2e_X_diag (uint64_t coupstr, unsigned int t, unsigned int p, unsigned int nspin)
{
    return CGC_2e_X (coupstr, coupstr, t, t, p, p, 1, -1, 1, -1, nspin);
}

double CGC_2e_X_core (unsigned int * twoSk, unsigned int * twoSb,
                   unsigned int r, unsigned int q,
                   int nr, int nq, bool pphh)
{
    int parity = 0;
    double xdiag = 1.0;
    unsigned int twoS0;
    unsigned int i;

    // TODO: check that this still works for t = 1, q = 2 case
    if (q>0){
    if (pphh){
        if (nq > 0){
            parity += twoSb[q-1] + (3*twoSk[q-2]) - 1;
        } else {
            parity += twoSk[q-1] + (3*twoSb[q-2]) - 1;
        }
        if (nr > 0){
            parity += twoSk[r+1] + (3*twoSb[r]) + 1;
        } else {
            parity += twoSb[r+1] + (3*twoSk[r]) - 1;
        }
    }
    parity = parity % 4;

    // q
    //     -1**[S(q) + S"(q-1) + 1/2]
    //     *
    //     { S'(q)  S(q)  1       }
    //     { 1/2    1/2   S"(q-1) }
    //
    //     S"(q-1) = S(q-1)   if nq isin {-2,+1}
    //             = S'(q-1)  if nq isin {+2,-1}
    switch (nq) {
        case -2:
            twoS0 = twoSk[q-1];
        case -1:
            twoS0 = twoSb[q-1];
        case 1:
            twoS0 = twoSk[q-1];
        case 2:
            twoS0 = twoSb[q-1];
    }
    parity += twoSk[q] + twoS0 + 1; // graph signs
    parity = parity % 4; // remember everything is *2 until the very end
    xdiag *= _get_wigner_6j_j41h (twoSb[q], twoSk[q], 2, 1, twoS0);

    // 'hill' and 'self' parity: |nq| = 2
    switch (nq) {
        case -2:
            parity += twoSk[q] + 3*twoSb[q-1] - 1;
        case 2:
            parity += twoSb[q] + 3*twoSk[q-1] - 1;
    }
    parity = parity % 4;

    } // q>0

    // q+1, q+2, ... r-2, r-1
    //     ~T(i) =
    //     -1**[S'(i) + S(i-1) - 1/2]
    //     *
    //     { 1    S'(i)   S(i)    }
    //     { 1/2  S(i-1)  S'(i-1) }
    for (i=q+1; i < r; i++){
        parity += twoSb[i] + twoSk[i-1] - 1; // graph signs
        parity = parity % 4;
        xdiag *= _get_wigner_6j_j41h (2, twoSb[i], twoSk[i], twoSk[i-1], twoSb[i-1]);
    }

    // r
    //
    //     -1**[S'(r-1) + S"(r) - 1/2]
    //     *
    //     { S'(r-1)  S(r-1)  1     }
    //     { 1/2      1/2     S"(r) }
    //
    //     S"(r) = S(r)   if nr isin {-2,+1}
    //           = S'(r)  if nr isin {+2,-1}
    switch (nr) {
        case -2:
            twoS0 = twoSk[r];
        case -1:
            twoS0 = twoSb[r];
        case 1:
            twoS0 = twoSk[r];
        case 2:
            twoS0 = twoSb[r];
    }
    xdiag *= _get_wigner_6j_j41h (twoSb[r-1], twoSk[r-1], 2, 1, twoS0);
    parity += twoS0 + twoSb[r-1] - 1; // graph signs
    parity = parity % 4;

    // 'hill' and 'self' parity: |nr| = 2
    switch (nr) {
        case -2:
            parity += twoSk[r] + 3*twoSb[r-1] - 1;
        case 2:
            parity += twoSb[r] + 3*twoSk[r-1] - 1;
    }
    parity = parity % 4;

    // Final sign computation
    assert ((parity % 2)==0);
    parity = parity / 2;
    if ((parity % 2) == 1){ xdiag = -xdiag; }

    return xdiag;
}

double CGC_2e_X (uint64_t brastr, uint64_t ketstr,
                 unsigned int t, unsigned int q, unsigned int r, unsigned int p,
                 int nt, int nq, int nr, int np,
                 unsigned int nspin)
{
    // nt, nq, nr, np:
    // -2: doubly-occupied in the bra
    // -1: singly-occupied in the bra
    //  1: singly-occupied in the ket
    //  2: doubly-occupied in the ket

    assert ((t!=q) || ((nt+nq)==0));
    assert ((r!=p) || ((nr+np)==0));
    double xdiag = 1.0;
    unsigned int twoS0b, twoS0k;
    int offk = 0;
    int offb = 0;
    int parity = 0;
    unsigned int i;
    unsigned int * twoSk = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int * twoSb = malloc ((nspin+1) * sizeof (unsigned int));
    for (i=0; i <= nspin; i++){
        twoSk[i] = 0;
        twoSb[i] = 0;
    }

    // Drake & Schlesinger ``reverse the order of counting'' so we have to do that here
    // If I just bitshift the coupstr instead I'm not sure that the CSFs mean the same thing
    // They have S0 = S and SN = 0
    assert (t <= nspin);
    assert (q <= nspin);
    assert (r < nspin);
    assert (p < nspin);
    t = nspin - t;
    q = nspin - q;
    r = nspin - r;
    p = nspin - p;
    assert (p>=r);
    assert (r>=q);
    assert (q>=t);
    // because the highest possible value of i is nspin - 1 and we want to start at 1 and go
    // through nspin inclusively.
    assert ((t>0) || abs (nt) == 1); // undefined to have doubly-occupied dummy orbital
    assert ((q>0) || abs (nq) == 1); // undefined to have doubly-occupied dummy orbital
    bool qt_pp = ((nq > 0) && (nt > 0)) || ((nq < 0) && (nt < 0));
    bool pr_pp = ((np > 0) && (nr > 0)) || ((np < 0) && (nr < 0));
    assert (qt_pp == pr_pp);

    // A bunch of index sanity checks
    if (p==r){ // p'r, pr'
        assert (abs (np) == 1);
        assert (np == -nr);
    } else if (p==(r+1)){
        if (pr_pp==false){ // p'r, pr'
            assert (abs (np) + abs (nr) < 4);
        } else { // p'r', pr
            assert (abs (np+nr) < 3);
        }
    } else if (p==(r+2)){ // p'r', pr
        assert (abs (np+nr) < 4);
    }
    if (q==t){ // q't, qt'
        assert (abs (nq) == 1);
        assert (nq == -nt);
    } else if (q==(t+1)){
        if (qt_pp==false){ // q't, qt'
            assert (abs (nq) + abs (nt) < 4);
        } else { // q't', qt
            assert (abs (nq+nt) < 3);
        }
    } else if (q==(t+2)){ // q't', qt
        assert (abs (nq+nt) < 4);
    }

    // populate the actual S arrays   
    for (i=0; i <= p; i++){
        twoSk[i] = _get_twoS_running (ketstr, i, nspin);
        twoSb[i] = _get_twoS_running (brastr, i, nspin);
    }

    // Factorize out the damn norm! I'm not keeping track of this garbage in the subdiagrams anymore!
    for (i=t; i < p; i++){
        // 2S+1 for each CG coefficient
        twoS0k = twoSk[i];
        twoS0b = twoSb[i];
        // paired electrons don't have CG coefficients!
        if (((i==t) || (i==t+1)) && nt==2){ twoS0k = 0; }
        if (((i==q) || (i==q-1)) && nq==2){ twoS0k = 0; }
        if (((i==r) || (i==r+1)) && nr==2){ twoS0k = 0; }
        if (((i==p) || (i==p-1)) && np==2){ twoS0k = 0; }
        if (((i==t) || (i==t+1)) && nt==-2){ twoS0b = 0; }
        if (((i==q) || (i==q-1)) && nq==-2){ twoS0b = 0; }
        if (((i==r) || (i==r+1)) && nr==-2){ twoS0b = 0; }
        if (((i==p) || (i==p-1)) && np==-2){ twoS0b = 0; }
        // For the dummy electron, somehow, the ket CG survives to cancel something
        if (i==0){ twoS0b = 0; }
        xdiag = xdiag * (twoS0k+1) * (twoS0b+1);
    }
    xdiag = sqrt (xdiag);

    // We just skip all of these when we are doing Sz, which corresponds to t = q = 0
    if (q>0){

    // t
    //
    //     = T'(t)                  if |nt| = 1
    //       S'(t) - S'(t-1) + 1/2  if nt = 2
    //       S(t) - S(t-1) + 1/2    if nt = -2
    if (abs (nt) == 2){
        if (nt > 0){
            parity += twoSb[t] + 3*twoSb[t-1] + 1;
        } else {
            parity += twoSk[t] + 3*twoSk[t-1] + 1;
        }
    }
    parity = parity % 4;

    // T(i) and T'(i) strings
    //     T(i) =
    //     -1**[S'(i) + S(i) + 1]
    //     *
    //     { 1    S(i)   S'(i+1) }
    //     { 1/2  S'(i)  S(i-1)  }
    for (i=t; i < q; i++){
        if ((i==t) && (abs (nt) == 2)){
            continue;
        }
        if ((i==(q-1)) && (abs (nq) == 2)){
            continue;
        }
        if (nt < 0){ // T(i)
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSb[i+1], twoSb[i], twoSk[i-1]);
        } else { // T'(i)
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSk[i+1], twoSb[i], twoSb[i-1]);
        }
        parity += 2 + twoSk[i] + twoSb[i];
        parity = parity % 4;
    }
    }

    // Xcore
    if (qt_pp){
        if (nq > 0){
            offb = -2;
        } else {
            offk = -2;
        }
        assert (q>=t+(abs(nt)+abs(nq))-1);
    }
    xdiag *= CGC_2e_X_core (twoSk+offk, twoSb+offb,
                            r, q,
                            nr, nq,
                            qt_pp);

    // T(i) and T'(i) strings
    //     T(i) =
    //     -1**[S'(i) + S(i) + 1]
    //     *
    //     { 1    S(i)   S'(i+1) }
    //     { 1/2  S'(i)  S(i-1)  }
    for (i=r; i < p; i++){
        if ((i==r) && (abs (nr) == 2)){
            continue;
        }
        if ((i==(p-1)) && (abs (np) == 2)){
            continue;
        }
        if (np > 0){ // T(i)
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSb[i+1], twoSb[i], twoSk[i-1]);
        } else { // T'(i)
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSk[i+1], twoSb[i], twoSb[i-1]);
        }
        parity += 2 + twoSk[i] + twoSb[i];
        parity = parity % 4;
    }

    // p
    //
    //     = T(p-1)                 if |np| = 1
    //       S'(p-1) - S'(p) + 1/2  if np = 2
    //       S(p-1) - S(p) + 1/2    if np = -2
    if (abs (np) == 2){
        if (np > 0){
            parity += twoSb[p-1] + 3*twoSb[p] + 1;
        } else {
            parity += twoSk[p-1] + 3*twoSk[p] + 1;
        }
    }
    parity = parity % 4;

    // Final sign computation
    assert ((parity % 2)==0);
    parity = parity / 2;
    if ((parity % 2) == 1){ xdiag = -xdiag; }

    free (twoSk);
    free (twoSb);
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
            fac = CGC_1s_diag (coupstrs[icoup], si, nspin, twoM);
            hdiag_csf[icoupconf] += fac * h1e_s[i];
            idxi = i*norb*norb*norb + i;
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (j, dconfstr, sconfstr);
                if (nj != 1){ continue; }
                sj = _get_spinindex (j, dconfstr, sconfstr);
                fac = CGC_2e_diag (coupstrs[icoup], si, sj, nspin);
                idx = idxi + j*norb*(norb+1);
                hdiag_csf[icoupconf] += fac * eri[idx];
            }
        }
    }
}
}



