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

static int first1(uint64_t r)
{
#if defined(__builtin_ffsll)
        return __builtin_ffsll(r) - 1;
#elif defined(HAVE_FFS)
        return ffsll(r) - 1;
#else
        if (r==0ULL){ return -1; }
        int n = 0;
        if ((((1ULL<<32)-1) & (r>>n)) == 0){ n += 32; }
        if ((((1ULL<<16)-1) & (r>>n)) == 0){ n += 16; }
        if ((((1ULL<<8)-1) & (r>>n)) == 0){ n += 8; }
        if ((((1ULL<<4)-1) & (r>>n)) == 0){ n += 4; }
        if ((((1ULL<<2)-1) & (r>>n)) == 0){ n += 2; }
        if ((((1ULL<<1)-1) & (r>>n)) == 0){ n += 1; }
        return n;
#endif
}

static int last1 (uint64_t r)
{
    if (r==0ULL){ return -1; }
    int n = 0;
    if (r >> (n + 32)) n += 32;
    if (r >> (n + 16)) n += 16;
    if (r >> (n +  8)) n +=  8;
    if (r >> (n +  4)) n +=  4;        
    if (r >> (n +  2)) n +=  2;
    if (r >> (n +  1)) n +=  1;
    return n;
}

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

unsigned int _get_occ (Str3 * addr, unsigned int i)
{
    if ((1ULL << i) & addr->dconf){ return 2; }
    unsigned int j = i;
    for (unsigned int k = 0; k < i; k++){
        if ((1ULL << k) & addr->dconf){ j--; }
    }
    if ((1ULL << j) & addr->sconf){ return 1; }
    else { return 0; }
}

unsigned int _get_spin (Str3 * addr, unsigned int i)
{
    unsigned int ni = _get_occ (addr, i);
    if (ni != 1){ return 0; }
    ni = _get_spinindex (addr, i);
    ni = (addr->spin & (1ULL << ni)) >> ni;
    return ni;
}

unsigned int _get_spinindex (Str3 * addr, unsigned int i)
{
    unsigned int j = i;
    for (unsigned int k = 0; k < i; k++){
        if ((1ULL << k) & addr->dconf){ j--; }
    }
    i = j;
    for (unsigned int k = 0; k < i; k++){
        if (((1ULL << k) & addr->sconf) == 0){ j--; }
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

void _pad_Str3 (Str3 * addr, Str3 * addr_padded)
{
    // pad zero bits in the sconf and spin members of addr to make orbital indexing easier
    uint64_t dconf = addr->dconf;
    uint64_t sconf = addr->sconf;
    uint64_t spin = addr->spin;
    addr_padded->dconf = dconf;

    unsigned int n = _count_set_bits (dconf);
    unsigned int p;
    for (unsigned int ip=0; ip<n; ip++){
        p = first1 (dconf);
        sconf = ((((1ULL << p)-1) & sconf) |
                 ((sconf >> p) << (p+1)));
        dconf = dconf ^ (1ULL<<p);
    }
    assert (dconf==0ULL);
    addr_padded->sconf = sconf;

    sconf = ~sconf;
    n = _count_set_bits (sconf);
    for (unsigned int ip=0; ip<n; ip++){
        p = first1 (sconf);
        spin = ((((1ULL << p)-1) & spin) |
                ((spin >> p) << (p+1)));
        sconf = sconf ^ (1ULL<<p);
    }
    assert (sconf==0ULL);
    addr_padded->spin = spin;
}

int Str3_link (Str3 * bra, Str3 * ket,
               unsigned int * p, unsigned int * r,
               unsigned int * q, unsigned int * t)
{
    /* Return value:
        -1 : CSFs are guaranteed unlinked by a 2-body Hamiltonian
         0 : CSFs are identical
         1 : CSFs linked by a single excitation
         2 : CSFs are linked by a double excitation
        On return, arguments "p","t"/"q","r" are set to the indices of the particle/hole operators
        as relevant depending on the return value (i.e., "p" and "t" are undefined for return value
        < 1, and "r" and "q" are only defined for return value 2). Only counts charge hops as
        excitations (spin flips are hard to evaluate because the intermediate spin couplings
        can be very differnt for 2-electron interactions).
    */
    unsigned int nelec_bra = 2*_count_set_bits (bra->dconf) + _count_set_bits (bra->sconf);
    unsigned int nelec_ket = 2*_count_set_bits (ket->dconf) + _count_set_bits (ket->sconf);
    assert (nelec_bra == nelec_ket);
    Str3 brap, ketp;
    _pad_Str3 (bra, &brap);
    _pad_Str3 (ket, &ketp);

    uint64_t uket = ketp.dconf | ketp.sconf; // occ > 0
    uint64_t ubra = brap.dconf | brap.sconf; // occ > 0
    uint64_t dsig = (bra->dconf) ^ (ket->dconf); // occ=2 to occ=0,1
    uint64_t usig = uket ^ ubra; // occ=1,2 to occ=0
    uint64_t c1sig = dsig^usig; // Identifies orbitals in which one electron hops in or out
    uint64_t c2sig = dsig&usig; // Identifies orbitals in which two electrons hop in or out

    unsigned int nc2 = _count_set_bits (c2sig);
    unsigned int nc1 = _count_set_bits (c1sig);
    assert ((nc1%2) == 0);
    int n = (2*nc2 + nc1) / 2;
    if (n>2){ n = -1; }
    if (n>0){
        switch (nc2){
            case 2:
                *p = first1 (c2sig);
                *r = first1 (c2sig);
                *q = last1 (c2sig);
                *t = last1 (c2sig);
                break;
            case 1:
                if (first1 (c2sig) < first1 (c1sig)){
                    *p = first1 (c2sig);
                    *r = first1 (c2sig);
                    *q = first1 (c1sig);
                    *t = last1 (c1sig);
                } else {
                    *p = first1 (c1sig);
                    *r = last1 (c1sig);
                    *q = first1 (c2sig);
                    *t = first1 (c2sig);
                }
                break;
            case 0:
                if (nc1 > 0){
                    *p = first1 (c1sig);
                    *t = last1 (c1sig);
                    if (nc1 > 2){
                        c1sig = c1sig ^ (1ULL << *p);
                        c1sig = c1sig ^ (1ULL << *t);
                        *r = first1 (c1sig);
                        *q = last1 (c1sig);
                    }
                } 
        }
    }
    return n;
}

double CGC_2e_X_core (unsigned int * twoSk, unsigned int * twoSb,
                      unsigned int r, unsigned int q,
                      int nr, int nq, bool pphh)
{
    int parity = 0;
    double xdiag = 1.0;
    unsigned int twoS0;
    unsigned int i;

    printf ("\nCGC_2e_X_core\n");
    printf ("(r,nr), (q,nq), pphh = (%u,%d), (%u,%d), %d",
            r, nr, q, nq, pphh);
    printf ("\ntwoSk =");
    for (unsigned int i=MAX (0, ((int) q)-2); i<=r+1; i++){ printf (" %u", twoSk[i]); }
    printf ("\ntwoSb =");
    for (unsigned int i=MAX (0, ((int) q)-2); i<=r+1; i++){ printf (" %u", twoSb[i]); }
    printf ("\n");
    fflush (stdout);

    // TODO: check that this still works for t = 1, q = 2 case
    if (q>0){

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
            break;
        case -1:
            twoS0 = twoSb[q-1];
            break;
        case 1:
            twoS0 = twoSk[q-1];
            break;
        case 2:
            twoS0 = twoSb[q-1];
            break;
    }
    parity += twoSk[q] + twoS0 + 1; // graph signs
    // parity = parity % 4; // remember everything is *2 until the very end
    printf ("opening: twoSb[%u] = %u, twoSk[%u] = %u, twoS0 = %u\n",
            q, twoSb[q], q, twoSk[q], twoS0);
    fflush (stdout);
    xdiag *= _get_wigner_6j_j41h (twoSb[q], twoSk[q], 2, 1, twoS0);
    printf ("opening parity = %d\n", parity);

    // 'hill' and 'self' parity: |nq| = 2
    switch (nq) {
        case -2:
            parity += twoSk[q] + 3*twoSb[q-1] - 1;
            break;
        case 2:
            parity += twoSb[q] + 3*twoSk[q-1] - 1;
            break;
    }
    printf ("hill/shelf opening parity = %d\n", parity);
    // parity = parity % 4;

    } // q>0

    // q+1, q+2, ... r-2, r-1
    //     ~T(i) =
    //     -1**[S'(i) + S(i-1) - 1/2]
    //     *
    //     { 1    S'(i)   S(i)    }
    //     { 1/2  S(i-1)  S'(i-1) }
    for (i=q+1; i < r; i++){
        parity += twoSb[i] + twoSk[i-1] - 1; // graph signs
        printf ("~T(%d) parity = %d\n", i, parity);
        // parity = parity % 4;
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
            break;
        case -1:
            twoS0 = twoSb[r];
            break;
        case 1:
            twoS0 = twoSk[r];
            break;
        case 2:
            twoS0 = twoSb[r];
            break;
    }
    printf ("closing: twoSb[%u] = %u, twoSk[%u] = %u, twoS0 = %u\n",
            r-1, twoSb[r-1], r-1, twoSk[r-1], twoS0);
    fflush (stdout);
    xdiag *= _get_wigner_6j_j41h (twoSb[r-1], twoSk[r-1], 2, 1, twoS0);
    parity += twoS0 + twoSb[r-1] - 1; // graph signs
    printf ("closing parity = %d\n", parity);
    // parity = parity % 4;

    // 'hill' and 'self' parity: |nr| = 2
    switch (nr) {
        case -2:
            parity += twoSk[r] + 3*twoSb[r-1] - 1;
            break;
        case 2:
            parity += twoSb[r] + 3*twoSk[r-1] - 1;
            break;
    }
    // parity = parity % 4;
    printf ("hill/shelf closing parity = %d\n", parity);
    fflush (stdout);

    // Final sign computation
    assert ((parity % 2)==0);
    parity = parity / 2;
    if ((parity % 2) == 1){ xdiag = -xdiag; }

    return xdiag;
}

double CGC_2e_X (unsigned int * twoSk, unsigned int *twoSb,
                 unsigned int t, unsigned int q, unsigned int r, unsigned int p,
                 int nt, int nq, int nr, int np,
                 unsigned int nspin)
{
    // nt, nq, nr, np:
    // -2: doubly-occupied in the bra
    // -1: singly-occupied in the bra
    //  1: singly-occupied in the ket
    //  2: doubly-occupied in the ket

    assert ((t!=q) || (abs(nt+nq)<(abs(nt)+abs(nq))));
    assert ((r!=p) || (abs(nr+np)<(abs(nr)+abs(np))));
    double xdiag = 1.0;
    unsigned int twoS0b, twoS0k;
    int offk = 0;
    int offb = 0;
    int parity = 0;
    unsigned int i;

    printf ("\nCGC_2e_X\n");
    printf ("(p,np), (r,nr), (q,nq), (t,nt), nspin = (%u,%d), (%u,%d), (%u,%d), (%u,%d), %u",
            p, np, r, nr, q, nq, t, nt, nspin);
    printf ("\ntwoSk =");
    for (unsigned int i=0; i<=nspin; i++){ printf (" %u", twoSk[i]); }
    printf ("\ntwoSb =");
    for (unsigned int i=0; i<=nspin; i++){ printf (" %u", twoSb[i]); }
    printf ("\n");
    fflush (stdout);
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
            assert ((abs (np) + abs (nr)) < 4);
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
            // assert (abs (nq+nt) < 3);
            // This can't be enforced because of the sr = sq, |nr| = |nq| = 2 problem.
        }
    } else if (q==(t+2)){ // q't', qt
        assert (abs (nq+nt) < 4);
    }

    // populate the actual S arrays   

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
        printf ("T(%u) ", i); fflush (stdout);
        parity += 2 + twoSk[i] + twoSb[i];
        parity = parity % 4;
    }
    }

    // Xcore
    if (qt_pp){
        if (nq > 0){
            offb = -2;
            parity += twoSk[r+1+abs(nr)-1] + (3*twoSb[r+abs(nr)-1]);
            parity += twoSb[q-abs(nq)-1] + (3*twoSk[q-abs(nq)]);
        } else {
            offk = -2;
            assert (false); // I haven't rederived this case but I think it doesn't appear?
        }
        // assert (q>=t+(abs(nt)+abs(nq))-1);
        // This can't be enforced because of the sr = sq, |nr| = |nq| = 2 problem.
    }
    printf ("Xcore(%u,%u) ", r, q); fflush (stdout);
    xdiag *= CGC_2e_X_core (twoSk+offk, twoSb+offb,
                            r-offb, q-offk,
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
        printf ("T(%u) ", i); fflush (stdout);
    }
    printf ("\n"); fflush (stdout);

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

    return xdiag;
}


void exc1_sort (Str3 * bra, Str3 * ket, 
               unsigned int p, unsigned int q,
               unsigned int * a, unsigned int * i)
{
    int np = _get_occ (ket, p) - _get_occ (bra, p);
    if (np > 0){
        *i = p;
        *a = q;
    } else {
        *a = p;
        *i = q;
    }
}

void exc2_sort (Str3 * bra, Str3 * ket, 
               unsigned int p, unsigned int q,
               unsigned int r, unsigned int s,
               unsigned int * a, unsigned int * i,
               unsigned int * b, unsigned int * j)
{
    unsigned int as[2];
    unsigned int is[2];
    unsigned int aidx = 0;
    unsigned int iidx = 0;
    int n = _get_occ (ket, p) - _get_occ (bra, p);
    if (n > 0){
        is[iidx] = p;
        iidx++;
    } else {
        as[aidx] = p;
        aidx++;
    }
    n = _get_occ (ket, q) - _get_occ (bra, q);
    if (n > 0){
        is[iidx] = q;
        iidx++;
    } else {
        as[aidx] = q;
        aidx++;
    }
    n = _get_occ (ket, r) - _get_occ (bra, r);
    if (n > 0){
        is[iidx] = r;
        iidx++;
    } else {
        as[aidx] = r;
        aidx++;
    }
    n = _get_occ (ket, s) - _get_occ (bra, s);
    if (n > 0){
        is[iidx] = s;
        iidx++;
    } else {
        as[aidx] = s;
        aidx++;
    }
    assert (aidx == 2);
    assert (iidx == 2);
    (*a) = as[0];
    (*b) = as[1];
    (*i) = is[0];
    (*j) = is[1];
}

double CGC_1e (unsigned int * twoSk, unsigned int * twoSb,
               unsigned int p, unsigned int q,
               int np, int nq,
               unsigned int nspin)
{
    if (p < q){
        return CGC_1e (twoSk, twoSb, q, p, nq, np, nspin);
    }
    printf ("\nCGC_1e\n");
    printf ("(p,np), (q,nq), nspin = (%u,%d), (%u,%d), %u",
            p, np, q, nq, nspin);
    printf ("\ntwoSk =");
    for (unsigned int i=0; i<=nspin; i++){ printf (" %u", twoSk[i]); }
    printf ("\ntwoSb =");
    for (unsigned int i=0; i<=nspin; i++){ printf (" %u", twoSb[i]); }
    printf ("\n");
    fflush (stdout);
    int parity = 0;
    double xdiag = 1.0;
    unsigned int twoS0;
    unsigned int * twoSp1;
    unsigned int * twoSp0;
    bool pphh = false;

    // B(q) nq = 2
    if (abs (nq) == 2){
        twoS0 = nq < 0 ? twoSk[q] : twoSb[q];
        parity += twoS0 + 3*twoSk[q-1] + 1;
        // parity = parity % 4;
    }

    // pp, hh case
    if ((np>0) == (nq>0)){
        pphh = true;
        if (nq>0){
            parity += twoSb[q] + 3*twoSk[q+1] + 1;
        } else {
            parity += twoSk[q] + 3*twoSb[q+1] + 1;
        }
        // parity = parity % 4;
        twoSp1 = malloc (nspin * sizeof (unsigned int));
        twoSp0 = nq>0 ? twoSb : twoSk;
        for (unsigned int i=0; i<=(q-1); i++){
            twoSp1[i] = twoSp0[i];
        }
        for (unsigned int i=(q-1); i<=p-2; i++){
            twoSp1[i+2] = twoSp0[i];
        }
        if (nq>0){
            twoSp0 = twoSb;
            twoSb = twoSp1;
        } else {
            twoSp0 = twoSk;
            twoSk = twoSp1;
        }
        q++;
    }

    printf ("After handling pphh logic\n");
    printf ("(p,np), (q,nq), nspin = (%u,%d), (%u,%d), %u",
            p, np, q, nq, nspin);
    printf ("\ntwoSk =");
    for (unsigned int i=0; i<=nspin; i++){ printf (" %u", twoSk[i]); }
    printf ("\ntwoSb =");
    for (unsigned int i=0; i<=nspin; i++){ printf (" %u", twoSb[i]); }
    printf ("\ncurrent parity = %d\n", parity);
    fflush (stdout);

    // T(i) and T'(i) strings
    //     T'(i) =
    //     -1**[S'(i) + S(i) + 1]
    //     *
    //     { 1    S(i)   S(i-1) }
    //     { 1/2  S'(i)  S'(i+1)  }
    for (unsigned int i=q; i<p; i++){
        if ((i==q) && (abs (nq) == 2)){ continue; }
        if ((i==(p-1)) && (abs (np) == 2)){ continue; }
        if (np > 0){
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSb[i+1], twoSb[i], twoSk[i-1]);
        } else {
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSk[i+1], twoSb[i], twoSb[i-1]);
        }
        parity += 2 + twoSk[i] + twoSb[i];
        // parity = parity % 4;
    }

    // A(p) np = 2
    if (abs (np) == 2){
        if (np > 0){
            parity += twoSb[p-1] + 3*twoSk[p] + 1;
        } else {
            parity += twoSk[p-1] + 3*twoSb[p] + 1;
        }
        // parity = parity % 4;
    }

    if (pphh){
        if (nq>0){
            twoSb = twoSp0;
        } else {
            twoSk = twoSp0;
        }
        free (twoSp1); 
    }
    printf ("parity = %d, xdiag = %f\n", parity, xdiag);
    fflush (stdout);
    assert ((parity % 2) == 0);
    parity = parity / 2;
    if ((parity%2)==1){ xdiag = -xdiag; }
    return xdiag;
}

void _get_spinindices1 (Str3 * addr, unsigned int p, unsigned int nspin,
                        unsigned int * twoS, unsigned int * sp)
{
    unsigned int nspin0 = _count_set_bits (addr->sconf);
    for (unsigned int r=0; r <= nspin0; r++){
        twoS[r] = _get_twoS_running (addr->spin, r, nspin0);
    }
    *sp = nspin0 - _get_spinindex (addr, p);
    int np = _get_occ (addr, p);
    if (np==2){
        for (int r=nspin0; r>=((int) (*sp)); r--){
            assert ((int) (r+2) < (int) (nspin+1));
            twoS[r+2] = twoS[r];
        }
        (*sp)++;
    }
}

void _get_spinindices2 (Str3 * addr, unsigned int p, unsigned int q,
                        unsigned int nspin,
                        unsigned int * twoS, 
                        unsigned int * sp, unsigned int * sq)
{
    if (p > q){
        _get_spinindices2 (addr, q, p, nspin, twoS, sq, sp);
        return;
    }
    unsigned int nspin0 = _count_set_bits (addr->sconf);
    _get_spinindices1 (addr, q, nspin, twoS, sq);
    *sp = nspin0 - _get_spinindex (addr, p);
    int np = _get_occ (addr, p);
    int nq = _get_occ (addr, q);
    if (nq==2){ *sp += (2 - ((int) (p==q))); }
    if ((p!=q) && (np==2)){
        for (int r=nspin0; r>=((int) (*sp)); r--){
            assert ((int) (r+2) < (int) (nspin+1));
            twoS[r+2] = twoS[r];
        }
        (*sp)++;
    } 
}

double csf_Eai (Str3 * bra, Str3 * ket, unsigned int a, unsigned int i)
{
    if (a>i){
        return csf_Eai (ket, bra, i, a);
    }
    Str3 brap, ketp;
    _pad_Str3 (bra, &brap);
    _pad_Str3 (ket, &ketp);
    // CSF orthogonality
    if ((brap.spin & ((1ULL<<a)-1)) != (ketp.spin & ((1ULL<<a)-1))){
        return 0.0;
    }
    if ((brap.spin>>i) != (ketp.spin>>i)){
        return 0.0;
    }
    int ni = _get_occ (ket, i);
    int na = -_get_occ (bra, a);
    unsigned int nspin_ket = _count_set_bits (ket->sconf);
    unsigned int nspin_bra = _count_set_bits (bra->sconf);
    nspin_ket += (ni-1)*2;
    nspin_bra += (abs(na)-1)*2;
    assert (nspin_bra==nspin_ket);
    unsigned int nspin=nspin_ket;
    unsigned int * twoSk = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int * twoSb = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int si, sa;
    _get_spinindices1 (bra, a, nspin, twoSb, &sa);
    _get_spinindices1 (ket, i, nspin, twoSk, &si);
    if (abs (na) == 2){ sa++; }
    double fac = CGC_1e (twoSk, twoSb, sa, si, na, ni, nspin);
    free (twoSk);
    free (twoSb);
    // operator anticommutation
    if (((sa-si)%2) == 1){
        fac = -fac;
    }
    return fac;
}

double csf_Sai (Str3 * bra, Str3 * ket, unsigned int a, unsigned int i, int twoM)
{
    if (a>i){
        return csf_Sai (ket, bra, i, a, twoM);
    }
    Str3 brap, ketp;
    _pad_Str3 (bra, &brap);
    _pad_Str3 (ket, &ketp);
    // CSF orthogonality
    if ((brap.spin & ((1ULL<<a)-1)) != (ketp.spin & ((1ULL<<a)-1))){
        return 0.0;
    }
    int ni = _get_occ (ket, i);
    int na = -_get_occ (bra, a);
    unsigned int nspin_ket = _count_set_bits (ket->sconf);
    unsigned int nspin_bra = _count_set_bits (bra->sconf);
    nspin_ket += (ni-1)*2;
    nspin_bra += (abs(na)-1)*2;
    assert (nspin_bra==nspin_ket);
    unsigned int nspin=nspin_ket;
    unsigned int * twoSk = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int * twoSb = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int si, sa;
    _get_spinindices1 (bra, a, nspin, twoSb, &sa);
    _get_spinindices1 (ket, i, nspin, twoSk, &si);
    if (abs (na) == 2){ sa++; }
    double fac = CGC_2e_X (twoSk, twoSb, 0, 0, si, sa, 1, -1, ni, na, nspin);
    free (twoSk);
    free (twoSb);
    unsigned int twoS = _get_twoS_running (bra->spin, 0, nspin_bra);
    // Sigma_z = ( 1/2  1  1/2 ) * S_z
    //           ( m    0   -m )
    //         = sqrt (2/3) * S_z
    // S_z = sqrt (3/2) * Sigma_z
    //     -1**[S-M]
    //     *
    //     ( S  1  S )
    //     ( M  0 -M )
    // WHY? UNCLEAR FACTOR OF 2 * .5;
    fac *= sqrt (1.5) * twoM; 
    if (twoS > 0){
       fac /= sqrt (twoS*(twoS+2)*.25);
    }
    return fac;
}

double csf_EaiEaj (Str3 * bra, Str3 * ket,
                   unsigned int i, unsigned int j)
{
    if (i == j){ return 1.0; }
    if (i > j){
        return csf_EaiEaj (bra, ket, j, i);
    }
    printf ("\ncsf_EaiEaj\n");
    fflush (stdout);
    Str3 brap, ketp;
    _pad_Str3 (bra, &brap);
    _pad_Str3 (ket, &ketp);
    // CSF orthogonality
    if ((brap.spin & ((1ULL<<i)-1)) != (ketp.spin & ((1ULL<<i)-1))){
        return 0.0;
    }
    if ((brap.spin>>j) != (ketp.spin>>j)){
        return 0.0;
    }
    int ni = _get_occ (ket, i);
    int nj = _get_occ (ket, j);
    unsigned int nspin_ket = _count_set_bits (ket->sconf);
    unsigned int nspin_bra = _count_set_bits (bra->sconf);
    nspin_ket += (ni-1)*2;
    nspin_ket += (nj-1)*2;
    assert ((nspin_bra+2)==nspin_ket);
    unsigned int nspin=nspin_ket;
    unsigned int * twoSk = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int * twoSb = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int si, sj;
    _get_spinindices2 (ket, i, j, nspin, twoSk, &si, &sj);
    if (abs (ni) == 2){ si++; }
    for (unsigned int r=0; r <= nspin_bra; r++){
        twoSb[r] = _get_twoS_running (bra->spin, r, nspin_bra);
    }
    int parity = si + sj + 1;
    double fac = CGC_1e (twoSk, twoSb, si, sj, ni, nj, nspin);
    printf ("exiting CGC_1e\n");
    fflush (stdout);
    if ((parity%2)==1){ fac = -fac; }
    return fac;

}

double csf_EaiEbj (Str3 * bra, Str3 * ket,
                   unsigned int a, unsigned int i,
                   unsigned int b, unsigned int j)
{
    if (a==b){
        return csf_EaiEaj (bra, ket, i, j);
    } else if (i==j){
        return csf_EaiEaj (ket, bra, a, b);
    }
    if (MIN (a,b) > MIN (i,j)){
        return csf_EaiEbj (ket, bra, i, a, j, b);
    }
    if (a > b){
        return csf_EaiEbj (bra, ket, b, j, a, i);
    }
    Str3 brap, ketp;
    _pad_Str3 (bra, &brap);
    _pad_Str3 (ket, &ketp);
    // CSF orthogonality
    unsigned int p = a;
    unsigned int r = MIN (i, MIN (j, b));
    unsigned int t = MAX (i, MAX (j, b));
    unsigned int q = (i+j+b) - (r+t);
    if ((brap.spin & ((1ULL<<p)-1)) != (ketp.spin & ((1ULL<<p)-1))){
        return 0.0;
    }
    if ((brap.spin>>t) != (ketp.spin>>t)){
        return 0.0;
    }
    uint64_t bra_rq = ((bra->spin) & ((1ULL<<r)-1)) >> q;
    uint64_t ket_rq = ((ket->spin) & ((1ULL<<r)-1)) >> q;
    bool unlinked_orth = (bra_rq!=ket_rq);

    int ni = _get_occ (ket, i);
    int nj = _get_occ (ket, j);
    int na = -_get_occ (bra, a);
    int nb = -_get_occ (bra, b);
    printf ("\ncsf_EaiEbj\n");
    printf ("a,i,b,j = %d,%d,%d,%d\n", a, i, b, j);
    printf ("na,ni,nb,nj = %d,%d,%d,%d\n", na, ni, nb, nj);
    fflush (stdout);
    printf ("p,r,q,t = %u,%u,%u,%u\n", p, r, q, t);
    unsigned int nspin_ket = _count_set_bits (ket->sconf);
    unsigned int nspin_bra = _count_set_bits (bra->sconf);
    //printf ("nspin_bra = %u, nspin_ket = %u\n", nspin_bra, nspin_ket);
    if (i==j){
        nspin_ket += 2;
        assert (ni == 2);
        assert (nj == 2);
    } else {
        nspin_ket += (ni-1)*2;
        nspin_ket += (nj-1)*2;
    }
    if (a==b){
        nspin_bra += 2;
        assert (na == -2);
        assert (nb == -2);
    } else {
        nspin_bra += (abs(na)-1)*2;
        nspin_bra += (abs(nb)-1)*2;
    }
    assert (nspin_bra==nspin_ket);
    unsigned int nspin=nspin_ket;
    printf ("nspin = %u\n", nspin);
    unsigned int * twoSk = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int * twoSb = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int si, sa, sj, sb;
    _get_spinindices2 (bra, a, b, nspin, twoSb, &sa, &sb);
    _get_spinindices2 (ket, i, j, nspin, twoSk, &si, &sj);
    if (abs (na) == 2){ sa++; }
    printf ("sa,si,sb,sj = %u,%u,%u,%u\n", sa, si, sb, sj);
    fflush (stdout);
    int np = na;
    unsigned int sp=sa;
    int nr,nq,nt;
    unsigned int sr,sq,st;
    if (r==i){
        nr=ni; sr=si;
        if (q==b){
            nq=nb; sq=sb; nt=nj; st=sj;
        } else {
            nq=nj; sq=sj; nt=nb; st=sb;
        }
    } else if (r==b){
        nr=nb; sr=sb;
        if (q==i){
            nq=ni; sq=si; nt=nj; st=sj;
        } else {
            nq=nj; sq=sj; nt=ni; st=si;
        }
    } else {
        nr=nj; sr=sj;
        if (q==b){
            nq=nb; sq=sb; nt=ni; st=si;
        } else {
            nq=ni; sq=si; nt=nb; st=sb;
        }
    }
    // When acting on doubly-occupied orbitals, I can choose ~sq or ~sq+1 freely, because
    // the difference is formally just a factor of -1 that is canceled by the "parity" line
    // down there. But I need to choose consistently: sp >= sr >= sq >= st and ~sq+1; ~sr
    // are mutually contradictory if ~sq = ~sr and |nq| = |nr| = 2. 
    if (abs (nq) == 2){
        if (sr > sq){
            sq++; 
        } else if (abs (nr) == 1){
            assert (sr == sq);
            assert (abs (nr+nq) < (abs (nr) + abs (nq)));
            // Here just sort so that sr = ~sr, for which there is no contradiction
            int nx = nr;
            nr = nq; nq = nx;
        } else if (abs (nq+nt) == (abs (nq) + abs (nt))){
            sq++; // You should end up adding 2 to sr down the line
        } else {
            assert (nr == -nq); // other cases should have been handled by csf_EaiEaj
            // I don't know what to do!
        }
    }

    int parity = sp + sr + sq + st + (int) (i>j);
    double facl = (i==r) ? 0.0 : 3.0;
    double facu = unlinked_orth ? 0.0 : 1.0;
    if ((parity%2)==1){
        facl = -facl;
        facu = -facu;
    }

    printf ("sp,sr,sq,st = %u,%u,%u,%u\n", sp,sr,sq,st);
    printf ("np,nr,nq,nt = %d,%d,%d,%d\n", np,nr,nq,nt);
    printf ("linked term\n");
    fflush (stdout);
    // linked term
    if (i!=r){
        facl *= CGC_2e_X (twoSk, twoSb, st, sq, sr, sp, nt, nq, nr, np, nspin);
        printf ("completing CGC_2e_X\n");
        facu *= 0.5; // this comes from closure of two S=0 Wigner 3j matrices + 1 fermion swap
    }

    // unlinked term
    if (!unlinked_orth){
        printf ("unlinked term: (%u,%d), (%u,%d)\n", sp, np, sr, nr);
        fflush (stdout);
        facu *= CGC_1e (twoSk, twoSb, sp, sr, np, nr, nspin);
        printf ("unlinked term: (%u,%d), (%u,%d)\n", sq, nq, st, nt);
        fflush (stdout);
        facu *= CGC_1e (twoSk, twoSb, sq, st, nq, nt, nspin);
    }

    free (twoSk);
    free (twoSb);
    return facl+facu; 
}

void FCICSFpspace_h0tril(double *hmat,
                         double *h1e_c, double *h1e_s, double *g2e,
                         uint64_t * dconfstrs,
                         uint64_t * sconfstrs,
                         uint64_t * coupstrs,
                         size_t np,
                         unsigned int norb, int twoM)
{
#pragma omp parallel default(shared)
{
    unsigned int p,q,r,t;
    unsigned int a,b,i,j;
    int nch;
    Str3 bra,ket;
    size_t ihmat, ihop;
    double fac = 1.0;
    double hop;
#pragma omp for schedule(dynamic)
    for (size_t ibra=0; ibra<np; ibra++){
    bra.dconf = dconfstrs[ibra];
    bra.sconf = sconfstrs[ibra];
    bra.spin = coupstrs[ibra];
    for (size_t iket=0; iket<ibra; iket++){
    ihmat = (iket*np) + ibra;
    ket.dconf = dconfstrs[iket];
    ket.sconf = sconfstrs[iket];
    ket.spin = coupstrs[iket];
    nch = Str3_link (&bra, &ket, &p, &r, &q, &t);
    switch (nch) {
        case 0: // spin interaction only
            for (p=0; p<norb; p++){
                if (_get_occ(&bra, p) != 1){ continue ; }
                // Sz
                ihop = p * (norb+1);
                fac = csf_Sai (&bra, &ket, p, p, twoM);
                hmat[ihmat] += fac * h1e_s[ihop];
                // eri exchange
                for (q=p; q<norb; q++){
                    if (_get_occ(&bra, q) != 1){ continue ; }
                    ihop = p*((norb*norb*norb) + 1) + q*(norb+1)*norb;
                    fac = csf_EaiEbj (&bra, &ket, p, q, q, p);
                    hmat[ihmat] += fac * g2e[ihop];
                }
            }
            break;
        case 1:
            exc1_sort (&bra, &ket, p, t, &a, &i);
            // E^a_i
            // h^a_i + g^aa_ai + g^ap_ip np - g^ap_pi
            // The second term looks odd but I think it's right. It makes more sense if you 
            // isolate the cases p=a and p=i.
            ihop = (a*norb) + i;
            hop = h1e_c[ihop];
            for (p=0; p<norb; p++){
                if (_get_occ(&ket, p) == 0){ continue ; }
                ihop = (a*norb*norb*norb) + (i*norb*norb) + p*(norb+1);
                hop += g2e[ihop] * _get_occ (&ket, p);
                if (p!=a){
                    ihop = (a*norb*norb*norb) + p*((norb*norb) + norb) + i;
                    hop -= g2e[ihop];
                }
            }
            fac = csf_Eai (&bra, &ket, a, i);
            hmat[ihmat] += fac * hop;
            // S^a_i
            ihop = (a*norb) + i;
            fac = csf_Sai (&bra, &ket, a, i, twoM);
            hmat[ihmat] += fac * h1e_s[ihop];
            // E^a_p E^p_i
            for (p=0; p<norb; p++){
                if (_get_occ(&ket, p) == 2){ continue ; }
                if (_get_occ(&ket, p) == 0){ continue ; }
                if (p==a){ continue; }
                if (p==i){ continue; }
                ihop = (a*norb*norb*norb) + p*((norb*norb) + norb) + i;
                fac = csf_EaiEbj (&bra, &ket, a, p, p, i);
                hmat[ihmat] += g2e[ihop] * fac;
            }
            break;
        case 2:
            exc2_sort (&bra, &ket, p, r, q, t, &a, &i, &b, &j);
            // E^a_i E^b_j
            ihop = (a*norb*norb*norb) + (i*norb*norb) + (b*norb) + j;
            printf ("\nmain loop\n");
            printf ("bra = %lu,%lu,%lu\n", bra.dconf, bra.sconf, bra.spin);
            printf ("ket = %lu,%lu,%lu\n", ket.dconf, ket.sconf, ket.spin);
            printf ("a,i,b,j = %u,%u,%u,%u\n", a, i, b, j);
            fflush (stdout);
            fac = csf_EaiEbj (&bra, &ket, a, i, b, j); // abs (np+nr) < 3
            hmat[ihmat] += g2e[ihop] * fac;
            // E^a_j E^b_i
            ihop = (a*norb*norb*norb) + (j*norb*norb) + (b*norb) + i;
            // fac = csf_EaiEbj (&bra, &ket, a, j, b, i); // abs (np+nr) < 3
            hmat[ihmat] += g2e[ihop] * fac;
            break;
    }
    }
    }
}
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
    Str3 addr;

// Subtract SOMO exchange terms from hdiag_det
#pragma omp for schedule(static)
    for (iconf = 0; iconf < nconf; iconf++){
        idoub = iconf / nsing;
        ising = iconf % nsing;
        addr.dconf = dconfstrs[idoub];
        addr.sconf = sconfstrs[ising];
        addr.spin = detstrs[0];
        wrk[iconf] = hdiag_det[iconf*ndet];
        for (unsigned int i = 0; i < norb; i++){
            ni = _get_occ (&addr, i);
            if (ni != 1){ continue ; }
            si = _get_spin (&addr, i);
            wrk[iconf] -= si ? h1e_s[i] : -h1e_s[i];
            idxi = i*norb*norb*norb + i;
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (&addr, j);
                if (nj != 1){ continue ; }
                sj = _get_spin (&addr, j);
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
        addr.dconf = dconfstrs[idoub];
        addr.sconf = sconfstrs[ising];
        addr.spin = coupstrs[icoup];
        hdiag_csf[icoupconf] = wrk[iconf];
        for (unsigned int i = 0; i < norb; i++){
            ni = _get_occ (&addr, i);
            if (ni != 1){ continue; }
            fac = csf_Sai (&addr, &addr, i, i, twoM);
            hdiag_csf[icoupconf] += fac * h1e_s[i];
            idxi = i*norb*norb*norb + i;
            for (unsigned int j = 0; j < i; j++){
                nj = _get_occ (&addr, j);
                if (nj != 1){ continue; }
                fac = csf_EaiEbj (&addr, &addr, i, j, j, i);
                idx = idxi + j*norb*(norb+1);
                hdiag_csf[icoupconf] += fac * eri[idx];
            }
        }
    }
}
}

