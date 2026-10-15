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
#define MAX_PARTICLE 64
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef DEBUG_1E
#define DEBUG_1E false
#endif

#ifndef DEBUG_2E
#define DEBUG_2E false
#endif

#ifndef DEBUG_2E_CGC
#define DEBUG_2E_CGC false
#endif

// The running spin "between" the two electrons of a doubly-occupied orbital is formally undefined
// (Drake & Schlesinger Fig. 2: it drops out of every graph). It is nevertheless stored in the twoS
// arrays by _get_spinindices1/2. Nothing should depend on it; these hooks let dev_csf_op/probe.py
// set it to the other legal value or to a value of the wrong parity to detect any code that reads it.
#ifndef PAIR_MID1
#define PAIR_MID1(x) ((x)+1)
#endif
#ifndef PAIR_MID2
#define PAIR_MID2(x) ((x)+1)
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
    int nspin = count_set_bits (sconfstr);
    int twoM = count_set_bits (detstr);
    assert (twoM <= nspin);
    twoM = nspin - 2*twoM;
    return -twoM;
}

unsigned int _get_twoS_running (uint64_t coupstr, unsigned int i, unsigned int nspin)
{
    if (nspin == 0){ return 0; }
    assert (nspin - i >= 0);
    uint64_t n = coupstr;
    unsigned int twoS = count_set_bits (coupstr);
    assert (nspin >= twoS); // S >= 0
    assert ((2*twoS) >= nspin);
    twoS = (2*twoS) - nspin;
    // in range 0 < j <= i;
    //    add 1/2 for every unset bit
    //    subtract 1/2 for every set bit
    twoS += i;
    // unset all bits i or more places from the edge
    n = coupstr ^ (coupstr & ((1ULL << (nspin-i)) - 1));
    n = count_set_bits (n);
    assert (n <= i);
    assert (twoS >= 2*n);
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

    unsigned int n = count_set_bits (dconf);
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
    n = count_set_bits (sconf);
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
    unsigned int nelec_bra = 2*count_set_bits (bra->dconf) + count_set_bits (bra->sconf);
    unsigned int nelec_ket = 2*count_set_bits (ket->dconf) + count_set_bits (ket->sconf);
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

    unsigned int nc2 = count_set_bits (c2sig);
    unsigned int nc1 = count_set_bits (c1sig);
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
        default:
            assert (false);
            exit (1);
    }
    parity += twoSk[q] + twoS0 + 1; // graph signs
    // parity = parity % 4; // remember everything is *2 until the very end
    xdiag *= _get_wigner_6j_j41h (twoSb[q], twoSk[q], 2, 1, twoS0);

    // 'hill' and 'shelf' parity: |nq| = 2
    switch (nq) {
        case -2:
            parity += twoSk[q-1] + 3*twoSb[q] - 1;
            break;
        case 2:
            parity += twoSb[q-1] + 3*twoSk[q] - 1;
            break;
    }
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
    xdiag *= _get_wigner_6j_j41h (twoSb[r-1], twoSk[r-1], 2, 1, twoS0);
    parity += twoS0 + twoSb[r-1] - 1; // graph signs
    // parity = parity % 4;

    // 'hill' and 'shelf' parity: |nr| = 2
    switch (nr) {
        case -2:
            parity += twoSk[r] + 3*twoSb[r-1] - 1;
            break;
        case 2:
            parity += twoSb[r] + 3*twoSk[r-1] - 1;
            break;
    }
    // parity = parity % 4;

    // Final sign computation
    assert ((parity % 2)==0);
    parity = parity / 2;
    if ((parity % 2) == 1){ xdiag = -xdiag; }

    return xdiag;
}

// Number of doubly-occupied operator orbitals in the chain with sign sgn (+1: ket, -1: bra) that
// claim the CG norm index x (pairs carry no CG coefficients); see the norm loop of CGC_2e_X.
static int _pair_claims (int x, int t, int q, int r, int p,
                         int nt, int nq, int nr, int np, int sgn)
{
    int n = 0;
    if (((x==t) || (x==t+1)) && nt==2*sgn){ n++; }
    if (((x==q) || (x==q-1)) && nq==2*sgn){ n++; }
    if (((x==r) || (x==r-1)) && nr==2*sgn){ n++; }
    if (((x==(p-1)) || (x==(p-2))) && np==2*sgn){ n++; }
    return n;
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

    double xdiag = 1.0;
    unsigned int twoS0b, twoS0k;
    int offb = 0;
    int parity = 0;
    unsigned int i;
    if (DEBUG_2E_CGC){
        printf ("\ntwoSk: ");
        for (i = 0; i <= nspin; i++){ printf ("%u ", twoSk[i]); }
        printf ("\ntwoSb: ");
        for (i = 0; i <= nspin; i++){ printf ("%u ", twoSb[i]); }
        printf ("\n"); fflush (stdout);
    }

    // because the highest possible value of i is nspin - 1 and we want to start at 1 and go
    // through nspin inclusively.
    assert ((t>0) || abs (nt) == 1); // undefined to have doubly-occupied dummy orbital
    assert ((q>0) || abs (nq) == 1); // undefined to have doubly-occupied dummy orbital
    bool qt_pp = ((nq > 0) && (nt > 0)) || ((nq < 0) && (nt < 0));
    bool pr_pp = ((np > 0) && (nr > 0)) || ((np < 0) && (nr < 0));
    assert (qt_pp == pr_pp);

    // A bunch of index sanity checks
    assert (p>=r);
    // pphh (qt_pp): r is an index into the bra chain and q into the ket chain; between them the
    // bra has two more electrons, so r+2 is the ket index of the same position.
    assert (qt_pp ? (r+2 >= q) : (r >= q));
    assert (q>=t);
    assert ((t!=q) || (abs(nt+nq)<(abs(nt)+abs(nq))));
    assert ((r!=p) || (abs(nr+np)<(abs(nr)+abs(np))));
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

    // pphh boundary CGs. The norm below runs over [t,p) for both chains, and the T strings
    // implicitly leave out one bra CG at the j end (index t-1, outside the loop anyway) and one
    // ket CG at the a end (index p, also outside). In pphh a pair can sit on either of those:
    // a bra pair at b with nothing open between b and j, or a ket pair at i with nothing open
    // between a and i. Then the CG to leave out is the next one into the chain: the lowest
    // unclaimed bra index >= t-1, or the highest unclaimed ket index <= p. If that search runs
    // all the way through to the other boundary, the factor has to be divided out explicitly.
    int dropb = -1, dropk = -1;
    if (qt_pp){
        int x;
        for (x=((int) t)-1; x < (int) p; x++){
            if (!_pair_claims (x, t, q, r, p, nt, nq, nr, np, -1)){ break; }
        }
        if (x > ((int) t)-1){ dropb = x; }
        for (x=(int) p; x > ((int) t)-1; x--){
            if (!_pair_claims (x, t, q, r, p, nt, nq, nr, np, +1)){ break; }
        }
        if (x < (int) p){ dropk = x; }
    }
    if (dropb == (int) p){ xdiag /= (double) (twoSb[p]+1); }
    if ((dropk == ((int) t)-1) && (t>0)){ xdiag /= (double) (twoSk[t-1]+1); }

    // Factorize out the damn norm! I'm not keeping track of this garbage in the subdiagrams anymore!
    for (i=t; i < p; i++){
        // 2S+1 for each CG coefficient
        twoS0k = twoSk[i];
        twoS0b = twoSb[i];
        // paired electrons don't have CG coefficients!
        // Each doubly-occupied operator orbital removes two factors. If two pairs are adjacent in
        // the same chain their claims overlap on one index, and one factor is left over; since a
        // pair doesn't change the running spin, it equals (2S+1) at the shared index.
        int claimk = _pair_claims (i, t, q, r, p, nt, nq, nr, np, +1);
        int claimb = _pair_claims (i, t, q, r, p, nt, nq, nr, np, -1);
        if (claimk || ((int) i == dropk)){ twoS0k = 0; }
        bool bra_paired = false;
        if (claimb){ twoS0b = 0; bra_paired = true; }
        if ((int) i == dropb){ twoS0b = 0; }
        for (int c=1; c<claimk; c++){ xdiag /= (double) (twoSk[i]+1); }
        for (int c=1; c<claimb; c++){ xdiag /= (double) (twoSb[i]+1); }
        // This bugfix came from this session:
        // claude --resume 5c6acb8f-5f71-43d9-b6a5-02f8e5be130b
        // Exactly one of the two CGs at the left boundary is spurious. Normally it is the
        // bra's, but if the boundary is inside the bra's doubly-occupied gap that CG is
        // already gone, so the ket's is the one that drops.
        if (i==0){ if (bra_paired){ twoS0k = 0; } else { twoS0b = 0; } }
        xdiag = xdiag * (twoS0k+1) * (twoS0b+1);
    }
    xdiag = sqrt (xdiag);
    if (DEBUG_2E_CGC){ printf ("CGC_2e_X norm: %f\n", xdiag); fflush (stdout); }

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
    // parity = parity % 4;

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
        // pphh (qt_pp): q = i is a ket orbital, so the ket index right of i, where the T
        // string must stop, is q-1 (q-2 for a ket pair at i). In phph, q = b is a bra orbital
        // whose index is already the ket index right of b.
        if ((i==(q-1)) && ((abs (nq) == 2) || qt_pp)){
            continue;
        }
        if (qt_pp && (i==(q-2)) && (abs (nq) == 2)){
            continue;
        }
        if (nt < 0){ // T(i)
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSb[i+1], twoSb[i], twoSk[i-1]);
            if (DEBUG_2E_CGC){
                printf (" { 1 %d %d }\n { 1 %d %d }\n", twoSk[i], twoSb[i+1], twoSb[i], twoSk[i-1]);
                fflush (stdout);
            }
        } else { // T'(i)
            xdiag *= _get_wigner_6j_j41h (1, twoSk[i], twoSk[i+1], twoSb[i], twoSb[i-1]);
            if (DEBUG_2E_CGC){
                printf (" { 1 %d %d }\n { 1 %d %d }\n", twoSk[i], twoSb[i+1], twoSb[i], twoSk[i-1]);
                fflush (stdout);
            }
        }
        if (DEBUG_2E_CGC){ printf (" * T(%d) = %f\n", i, xdiag); fflush (stdout); }
        parity += 2 + twoSk[i] + twoSb[i];
        // parity = parity % 4;
    }
    }

    // Xcore
    if (qt_pp){
        // q = i is a ket creator after csf_EaiEbj's a<b canonicalization
        assert (nq > 0);
        offb = -2;
        parity += twoSk[r+1+abs(nr)-1] + (3*twoSb[r+abs(nr)-1]);
        parity += twoSb[q-abs(nq)-1] + (3*twoSk[q-abs(nq)]);
        // The core is the hphp core (nr < 0, nq > 0) read with the bra chain shifted by the
        // two extra bra electrons between b and i: twoSb-2, and r+2 for b in ket indices.
    }
    xdiag *= CGC_2e_X_core (twoSk, twoSb+offb,
                            r-offb, q,
                            nr, nq,
                            qt_pp);
    if (DEBUG_2E_CGC){ printf (" * Xcore = %f\n", xdiag); fflush (stdout); }

    // T(i) and T'(i) strings
    //     T(i) =
    //     -1**[S'(i) + S(i) + 1]
    //     *
    //     { 1    S(i)   S'(i+1) }
    //     { 1/2  S'(i)  S(i-1)  }
    for (i=r; i < p; i++){
        // pphh (qt_pp): r = b is a bra index, and between a and b the bra has one more
        // electron than the ket, so the T string starts at ket index r+1 (r+2 for a bra pair
        // at b, whose index r is the middle of the pair).
        if ((i==r) && ((abs (nr) == 2) || qt_pp)){
            continue;
        }
        if (qt_pp && (i==(r+1)) && (abs (nr) == 2)){
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
        if (DEBUG_2E_CGC){ printf (" * T(%d) = %f\n", i, xdiag); fflush (stdout); }
        // parity = parity % 4;
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
    // parity = parity % 4;

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
    // callers pass p left of q (csf_Eai sorts a<i; csf_EaiEbj passes sp>=sr, sq>=st)
    assert (p >= q);
    if (DEBUG_1E){
    printf ("<");
    for (unsigned int i=0; i<=nspin; i++){
        printf ("%u", twoSb[i]);
    }
    printf ("|%u(%d)'%u(%d)|", p, np, q, nq);
    for (unsigned int i=0; i<=nspin; i++){
        printf ("%u", twoSk[i]);
    }
    printf ("> = "); fflush (stdout);
    }
    int parity = 0;
    double xdiag = 1.0;
    unsigned int twoS0k, twoS0b;
    // one creator and one annihilator (pphh unlinked terms are evaluated by _pphh_unlinked)
    assert ((np>0) != (nq>0));

    // norm
    // In Drake & Schlesinger eq (14), the norm is associated with the spin going into a CG
    // coefficient: S(i) corresponds to the norm for the i+1 CG coefficient. However, in
    // eq (9'), the S(0) case is explicitly excluded from the overall norm, and S(N) is by
    // construction always zero.
    for (unsigned int i=q; i < p; i++){
        // 2S+1 for each CG coefficient
        twoS0k = twoSk[i];
        twoS0b = twoSb[i];
        // paired electrons don't have CG coefficients!
        if (((i==q) || (i==(q+1))) && nq==2){ twoS0k = 0; }
        if (((i==(p-1)) || (i==(p-2))) && np==2){ twoS0k = 0; }
        if (((i==q) || (i==(q+1))) && nq==-2){ twoS0b = 0; }
        if (((i==(p-1)) || (i==(p-2))) && np==-2){ twoS0b = 0; }
        xdiag = xdiag * (twoS0k+1) * (twoS0b+1);
    }
    xdiag = sqrt (xdiag);

    // The diagram has a factor of 1/(2S(q-1)+1) canceling the norm of the q CG coefficient.
    // Normally this is accounted for by just starting the product above at i=q. If q is paired,
    // skipping both S(q) and S(q+1) in the product below usually accounts for this cancellation.
    // Then exception is if p = q+1. Then sqrt (1/(2S(q-1)+1)) is "orphaned": it can't cancel the
    // S(q+1) normalization factor because that's already cancelled by the implicit 1/(S(q+1)+1)
    // from eq (7.33) of Brink & Satchler. We have to divide explicitly.
    if (p==q+1){
        assert ((abs (np) + abs (nq)) <= 3);
        if ((nq == 2) || (np == 2)){
            assert (twoSk[q-1]==twoSk[q+1]);
            xdiag /= sqrt ((double) (twoSk[q-1]+1));
        } else if ((nq == -2) || (np == -2)){
            assert (twoSb[q-1]==twoSb[q+1]);
            xdiag /= sqrt ((double) (twoSb[q-1]+1));
        }
    }

    // A(q) nq = 2
    if (abs (nq) == 2){
        if (nq > 0){
            parity += twoSb[q-1] + 3*twoSb[q] + 3;
        } else {
            parity += twoSk[q-1] + 3*twoSk[q] + 3;
        }
        // parity = parity % 4;
    }

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

    // B(p) np = 2
    if (abs (np) == 2){
        if (np > 0){
            parity += twoSb[p] + 3*twoSb[p-1] + 3;
        } else {
            parity += twoSk[p] + 3*twoSk[p-1] + 3;
        }
        // parity = parity % 4;
    }

    assert ((parity % 2) == 0);
    parity = parity / 2;
    if ((parity%2)==1){ xdiag = -xdiag; }
    if (DEBUG_1E){ printf ("%e\n", xdiag); fflush (stdout); }
    return xdiag;
}

void _get_spinindices1 (Str3 * addr, unsigned int p, unsigned int nspin,
                        unsigned int * twoS, unsigned int * sp)
{
    unsigned int nspin0 = count_set_bits (addr->sconf);
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
        twoS[(*sp)+1] = PAIR_MID1 (twoS[(*sp)]);
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
    unsigned int nspin0 = count_set_bits (addr->sconf);
    _get_spinindices1 (addr, q, nspin, twoS, sq);
    *sp = nspin0 - _get_spinindex (addr, p);
    int np = _get_occ (addr, p);
    int nq = _get_occ (addr, q);
    if (nq==2){ 
        *sp += (2 - ((int) (p==q))); 
        nspin0 += 2;
    }
    if ((p!=q) && (np==2)){
        for (int r=nspin0; r>=((int) (*sp)); r--){
            assert ((int) (r+2) < (int) (nspin+1));
            twoS[r+2] = twoS[r];
        }
        twoS[(*sp)+1] = PAIR_MID2 (twoS[(*sp)]);
        (*sp)++;
    } 
}

double csf_Eai (Str3 * bra, Str3 * ket, unsigned int a, unsigned int i)
{
    if (a>i){
        return csf_Eai (ket, bra, i, a);
    }
    if (DEBUG_1E){
    printf ("\n<%lu,%lu,%lu|%u'%u|%lu,%lu,%lu>\n",
            bra->dconf, bra->sconf, bra->spin,
            a, i,
            ket->dconf, ket->sconf, ket->spin);
    fflush (stdout);
    }
    Str3 brap, ketp;
    _pad_Str3 (bra, &brap);
    _pad_Str3 (ket, &ketp);
    // CSF orthogonality
    if (a>0){ if ((brap.spin & ((1ULL<<a)-1)) != (ketp.spin & ((1ULL<<a)-1))){
        if (DEBUG_1E){ printf ("right escape\n"); fflush (stdout); }
        return 0.0;
    }}
    if ((brap.spin>>(i+1)) != (ketp.spin>>(i+1))){
        if (DEBUG_1E){ printf ("left escape\n"); fflush (stdout); }
        return 0.0;
    }
    int ni = _get_occ (ket, i);
    int na = -_get_occ (bra, a);
    unsigned int nspin_ket = count_set_bits (ket->sconf);
    unsigned int nspin_bra = count_set_bits (bra->sconf);
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
    if (DEBUG_1E){ printf ("final value = %e\n\n", fac); fflush (stdout); }
    assert (fac==fac);
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
    if (a>0){ if ((brap.spin & ((1ULL<<a)-1)) != (ketp.spin & ((1ULL<<a)-1))){
        return 0.0;
    }}
    int ni = _get_occ (ket, i);
    int na = -_get_occ (bra, a);
    unsigned int nspin_ket = count_set_bits (ket->sconf);
    unsigned int nspin_bra = count_set_bits (bra->sconf);
    unsigned int twoS = _get_twoS_running (bra->spin, 0, nspin_bra);
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
    // Sigma_z = ( 1/2  1  1/2 ) * S_z
    //           ( m    0   -m )
    //         = sqrt (2/3) * S_z
    // S_z = sqrt (3/2) * Sigma_z
    //     -1**[S-M]
    //     *
    //     ( S  1  S )
    //     ( M  0 -M )
    // WHY? UNCLEAR FACTOR OF 2 * .5;
    assert (fac==fac);
    fac *= sqrt (1.5) * twoM; 
    assert (fac==fac);
    if (twoS > 0){
       fac /= sqrt (twoS*(twoS+2)*.25);
    }
    // operator anticommutation
    if (((sa-si)%2) == 1){
        fac = -fac;
    }
    assert (fac==fac);
    return fac;
}

// Move the closed pair of orbital src to orbital dst (dst must be empty). The spin coupling string
// is unchanged because a closed pair carries no spin and the set of open shells is unchanged.
static void _relocate_pair (Str3 * in, unsigned int src, unsigned int dst, Str3 * out)
{
    int occ[64];
    for (unsigned int p=0; p<64; p++){ occ[p] = (int) _get_occ (in, p); }
    assert (occ[src] == 2);
    assert (occ[dst] == 0);
    occ[src] = 0; occ[dst] = 2;
    out->dconf = 0ULL; out->sconf = 0ULL;
    unsigned int k = 0;
    for (unsigned int p=0; p<64; p++){
        if (occ[p] == 2){ out->dconf |= (1ULL << p); continue; }
        if (occ[p] == 1){ out->sconf |= (1ULL << k); }
        k++;
    }
    out->spin = (out->sconf == 0ULL) ? 0ULL : in->spin;
}

double csf_EaiEaj (Str3 * bra, Str3 * ket,
                   unsigned int i, unsigned int j)
{
    /* <bra|E_ai E_aj|ket>, where a is the orbital that is doubly occupied in bra and empty in ket.

       On the (i,j) electrons, E_ai E_aj and E_ji act identically: both give sqrt(2) times the
       singlet-coupled component and annihilate the triplet one. A closed pair is a spin scalar,
       so moving it between orbitals changes no spin coupling (and costs no fermion sign). Hence
       this is a single excitation between CSFs with one pair relocated, and csf_Eai (which reads
       no undefined intermediate spins) can evaluate it:
         nj = 1:      <bra(a->j)|E_ji|ket>
         ni = 1:      <bra(a->i)|E_ij|ket>
         ni = nj = 2: -<bra|E_ji|ket(j->a)>
       (signs verified numerically against the resolution-of-the-identity oracle csf_EaiEbj_ri).
       This replaces the old CGC_1e-based evaluation, which padded the bra chain with a fabricated
       pair and read the undefined intermediate spin of doubly-occupied i or j. */
    unsigned int a = 64;
    for (unsigned int p=0; p<64; p++){
        if (((int) _get_occ (bra, p) - (int) _get_occ (ket, p)) == 2){ a = p; break; }
    }
    assert (a < 64);
    if (a >= 64){ return 0.0; }
    Str3 x;
    if (i == j){
        // A whole pair moves i -> a: <bra|E_ai E_ai|ket> = 2 if the spin couplings are identical,
        // else 0. The caller's convention for this case includes the Hamiltonian's 1/2, so return 1.
        _relocate_pair (bra, a, i, &x);
        uint64_t sx = (x.sconf == 0ULL) ? 0ULL : x.spin;
        uint64_t sk = (ket->sconf == 0ULL) ? 0ULL : ket->spin;
        return ((x.dconf == ket->dconf) && (x.sconf == ket->sconf) && (sx == sk)) ? 1.0 : 0.0;
    }
    int ni = _get_occ (ket, i);
    int nj = _get_occ (ket, j);
    double fac;
    if (nj == 1){
        _relocate_pair (bra, a, j, &x);
        fac = csf_Eai (&x, ket, j, i);
    } else if (ni == 1){
        _relocate_pair (bra, a, i, &x);
        fac = csf_Eai (&x, ket, i, j);
    } else {
        _relocate_pair (ket, j, a, &x);
        fac = -csf_Eai (bra, &x, j, i);
    }
    if (DEBUG_2E){ printf ("csf_EaiEaj <%lu,%lu,%lu|%u,%u|%lu,%lu,%lu> = %e\n",
                           bra->dconf, bra->sconf, bra->spin, i, j, ket->dconf, ket->sconf, ket->spin, fac); }
    assert (fac==fac);
    return fac;
}


// Resolution-of-the-identity evaluation of two-electron spin-free operator products, built only
// from the (fully validated) one-electron csf_Eai. Because E_pq is spin-free, E_bj|ket> lies in the
// span of the CSFs of its occupation with the same S and M, so
//     <bra|E_ai E_bj|ket> = sum_m <bra|E_ai|m> <m|E_bj|ket>
// exactly, with m running over all genealogical couplings of the intermediate occupation.
// Brute-force enumeration of the couplings; meant as an oracle (-DCSF_2E_RI=1 uses it everywhere).
#ifndef CSF_2E_RI
#define CSF_2E_RI 0
#endif
#ifndef CSF_EAIEBJ_PART
#define CSF_EAIEBJ_PART 0
#endif
#ifndef CSF_PPHH_GRAPHICAL
#define CSF_PPHH_GRAPHICAL 1
#endif
#ifndef CSF_HPHP_GRAPHICAL
#define CSF_HPHP_GRAPHICAL 1
#endif
static void _occ_of (Str3 * s, unsigned int norb, int * occ){
    for (unsigned int p=0; p<norb; p++){ occ[p] = (int) _get_occ (s, p); }
}
static bool _str3_from_occ (int * occ, unsigned int norb, Str3 * s, unsigned int * nopen){
    s->dconf = 0ULL; s->sconf = 0ULL; s->spin = 0ULL;
    unsigned int k = 0; *nopen = 0;
    for (unsigned int p=0; p<norb; p++){
        if ((occ[p] < 0) || (occ[p] > 2)){ return false; }
        if (occ[p] == 2){ s->dconf |= (1ULL << p); continue; }
        if (occ[p] == 1){ s->sconf |= (1ULL << k); (*nopen)++; }
        k++;
    }
    return true;
}
// <bra|E_ai|ket>, a != i. The RI callers build every intermediate so that the occupations are
// consistent; csf_Eai does not check them (its escapes test only the spin couplings), so assert it.
static double _csf_E1 (Str3 * bra, Str3 * ket, unsigned int a, unsigned int i, unsigned int norb){
    assert (a != i);
#ifndef NDEBUG
    int ob[64], ok[64];
    _occ_of (bra, norb, ob); _occ_of (ket, norb, ok);
    ok[i]--; ok[a]++;
    for (unsigned int p=0; p<norb; p++){ assert (ok[p] == ob[p]); }
#endif
    return csf_Eai (bra, ket, a, i);
}
static unsigned int _norb_of (Str3 * x, Str3 * y){
    // orbitals actually needed to describe both strings
    unsigned int n = 1;
    for (unsigned int p=0; p<64; p++){ if ((_get_occ (x, p) > 0) || (_get_occ (y, p) > 0)){ n = p+1; } }
    return n;
}
double csf_EaiEbj_ri (Str3 * bra, Str3 * ket, unsigned int a, unsigned int i, unsigned int b, unsigned int j)
{
    unsigned int norb = MAX (_norb_of (bra, ket), MAX (MAX (a, i), MAX (b, j)) + 1);
    int om[64];
    _occ_of (ket, norb, om);
    if (b == j){ return ((double) om[j]) * _csf_E1 (bra, ket, a, i, norb); }
    om[j]--; om[b]++;
    Str3 m;
    unsigned int nopen;
    if (!_str3_from_occ (om, norb, &m, &nopen)){ return 0.0; }
    unsigned int nspin_ket = count_set_bits (ket->sconf);
    unsigned int twoS = (ket->sconf == 0ULL) ? 0 : _get_twoS_running (ket->spin, 0, nspin_ket);
    if ((nopen < twoS) || ((nopen - twoS) % 2)){ return 0.0; }
    unsigned int nup = (nopen + twoS) / 2;
    double sum = 0.0;
    if (nopen == 0){
        m.spin = 0ULL;
        return _csf_E1 (bra, &m, a, i, norb) * _csf_E1 (&m, ket, b, j, norb);
    }
    // iterate over all nopen-bit strings with nup set bits (Gosper's hack)
    uint64_t x = (nup == 0) ? 0ULL : ((1ULL << nup) - 1ULL);
    uint64_t lim = 1ULL << nopen;
    while (x < lim){
        // valid genealogical coupling: running spin never negative, counting from bit 0
        int run = 0; bool ok = true;
        for (unsigned int k=0; k<nopen; k++){ run += ((x >> k) & 1ULL) ? 1 : -1; if (run < 0){ ok = false; break; } }
        if (ok){
            m.spin = x;
            double f2 = _csf_E1 (&m, ket, b, j, norb);
            if (f2 != 0.0){ sum += _csf_E1 (bra, &m, a, i, norb) * f2; }
        }
        if (x == 0ULL){ break; }
        uint64_t c = x & (~x + 1ULL);
        uint64_t r = x + c;
        x = (((r ^ x) >> 2) / c) | r;
    }
    return sum;
}

// Same identity as csf_EaiEbj_ri, but enumerating only the intermediate couplings that can
// contribute: <m|E_bj|ket> vanishes unless m and ket have identical couplings outside the orbital
// range spanned by (b,j), and <bra|E_ai|m> unless m and bra agree outside the range of (a,i).
// So only the open shells of m inside both ranges are free.
static void _steps_of (Str3 * s, unsigned int norb, int * step){
    // step[p] = +1/-1 for the coupling step of open shell p, 0 if p is not singly occupied
    unsigned int k = 0;
    for (unsigned int p=0; p<norb; p++){
        step[p] = 0;
        if (_get_occ (s, p) == 1){ step[p] = ((s->spin >> k) & 1ULL) ? 1 : -1; k++; }
    }
}
double csf_EaiEbj_ri_local (Str3 * bra, Str3 * ket, unsigned int a, unsigned int i, unsigned int b, unsigned int j)
{
    unsigned int norb = MAX (_norb_of (bra, ket), MAX (MAX (a, i), MAX (b, j)) + 1);
    int om[64], sb[64], sk[64], sm[64], freep[64];
    _occ_of (ket, norb, om);
    if (b == j){ return ((double) om[j]) * _csf_E1 (bra, ket, a, i, norb); }
    om[j]--; om[b]++;
    Str3 m;
    unsigned int nopen;
    if (!_str3_from_occ (om, norb, &m, &nopen)){ return 0.0; }
    _steps_of (bra, norb, sb); _steps_of (ket, norb, sk);
    unsigned int lo1 = MIN (b, j), hi1 = MAX (b, j), lo2 = MIN (a, i), hi2 = MAX (a, i);
    int nfree = 0;
    for (unsigned int p=0; p<norb; p++){
        sm[p] = 0;
        if (om[p] != 1){ continue; }
        bool in1 = (p >= lo1) && (p <= hi1);
        bool in2 = (p >= lo2) && (p <= hi2);
        if (in1 && in2){ freep[nfree++] = (int) p; continue; }
        if (!in1 && !in2){
            if ((sk[p] == 0) || (sk[p] != sb[p])){ return 0.0; }
            sm[p] = sk[p];
        } else if (!in1){
            if (sk[p] == 0){ return 0.0; }
            sm[p] = sk[p];
        } else {
            if (sb[p] == 0){ return 0.0; }
            sm[p] = sb[p];
        }
    }
    unsigned int nspin_ket = count_set_bits (ket->sconf);
    int twoS = (ket->sconf == 0ULL) ? 0 : (int) _get_twoS_running (ket->spin, 0, nspin_ket);
    double sum = 0.0;
    for (uint64_t x=0; x < (1ULL << nfree); x++){
        for (int f=0; f<nfree; f++){ sm[freep[f]] = ((x >> f) & 1ULL) ? 1 : -1; }
        int run = 0; bool ok = true; uint64_t spin = 0ULL; unsigned int k = 0;
        for (unsigned int p=0; p<norb; p++){
            if (sm[p] == 0){ continue; }
            run += sm[p]; if (run < 0){ ok = false; break; }
            if (sm[p] > 0){ spin |= (1ULL << k); }
            k++;
        }
        if (!ok || (run != twoS)){ continue; }
        m.spin = (m.sconf == 0ULL) ? 0ULL : spin;
        double f2 = _csf_E1 (&m, ket, b, j, norb);
        if (f2 != 0.0){ sum += _csf_E1 (bra, &m, a, i, norb) * f2; }
    }
    return sum;
}

// Running 2S after each orbital: twoS[p] is the spin of the chain coupled through orbital p.
static void _twoS_after (Str3 * s, int * twoS){
    int run = 0;
    unsigned int k = 0;
    for (unsigned int p=0; p<64; p++){
        if (_get_occ (s, p) == 1){
            run += ((s->spin >> k) & 1ULL) ? 1 : -1;
            k++;
        }
        twoS[p] = run;
    }
}

// Unlinked pphh term (the singlet-pair channel, (<E_ai E_bj> + <E_aj E_bi>)/2) for a<b<i,j.
// Its diagram is a pair annihilation on (i,j) times a pair creation on (a,b), joined by an
// intermediate m = ket minus the (i,j) pair = bra minus the (a,b) pair. Pairs are spin scalars,
// so m is unique: its running spin is the ket's to the left of min(i,j) and the bra's from there
// on (if bra and ket don't agree between b and min(i,j), it is not a valid coupling and the term
// vanishes). Putting the pair on a spare orbital x turns each factor into a pair hop, which
// csf_EaiEaj evaluates as a single csf_Eai. Each pair hop carries sqrt(2) times the singlet-pair
// amplitude, hence the 1/2.
static double _pphh_unlinked (Str3 * bra, Str3 * ket,
                              unsigned int a, unsigned int i,
                              unsigned int b, unsigned int j)
{
    const unsigned int x = 63; // spare orbital (pspace is limited to norb <= 30)
    int om[64], sb[64], sk[64];
    for (unsigned int p=0; p<64; p++){ om[p] = (int) _get_occ (ket, p); }
    om[i]--; om[j]--;
#ifndef NDEBUG
    for (unsigned int p=0; p<64; p++){
        int ob = (int) _get_occ (bra, p) - ((p==a) ? 1 : 0) - ((p==b) ? 1 : 0);
        assert (ob == om[p]);
    }
    assert (om[x] == 0 && _get_occ (bra, x) == 0);
#endif
    _twoS_after (bra, sb);
    _twoS_after (ket, sk);
    unsigned int c = MIN (i, j);
    Str3 m;
    m.dconf = 1ULL << x; m.sconf = 0ULL; m.spin = 0ULL;
    unsigned int k = 0, kk = 0;
    int prev = 0;
    for (unsigned int p=0; p<64; p++){
        int cur = (p < c) ? sk[p] : sb[p];
        int step = cur - prev;
        prev = cur;
        if (om[p] == 2){
            if (step != 0){ return 0.0; }
            m.dconf |= 1ULL << p;
            continue;
        }
        if (om[p] == 1){
            m.sconf |= 1ULL << k;
            if (step == 1){ m.spin |= 1ULL << kk; }
            else if (step != -1){ return 0.0; }
            kk++;
        } else if (step != 0){
            return 0.0;
        }
        k++;
    }
    return 0.5 * csf_EaiEaj (&m, bra, b, a) * csf_EaiEaj (&m, ket, i, j);
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
    // nested hphp: after the a<b swap below, j < a < i < b
    bool hphp = CSF_HPHP_GRAPHICAL && ((a<b) ? ((j<a) && (a<i) && (i<b)) : ((i<b) && (b<j) && (j<a)));
    if ((MIN (a,b) > MIN (i,j)) && !hphp){
        return csf_EaiEbj (ket, bra, i, a, j, b);
    }
    if (a > b){
        return csf_EaiEbj (bra, ket, b, j, a, i);
    }
    // pphh (both creators below both annihilators) is evaluated graphically below: linked term by
    // CGC_2e_X (qt_pp), unlinked term by _pphh_unlinked. -DCSF_PPHH_GRAPHICAL=0 evaluates it
    // instead by the local resolution of the identity, which is exact and can serve as a check.
    if (!CSF_PPHH_GRAPHICAL && (b < i) && (b < j)){
        return csf_EaiEbj_ri_local (bra, ket, a, i, b, j);
    }
    if (DEBUG_2E){
    printf ("\n<%lu,%lu,%lu|%u'%u,%u'%u|%lu,%lu,%lu>\n",
            bra->dconf, bra->sconf, bra->spin,
            a, i, b, j,
            ket->dconf, ket->sconf, ket->spin);
    fflush (stdout);
    }
    Str3 brap, ketp;
    _pad_Str3 (bra, &brap);
    _pad_Str3 (ket, &ketp);
    // CSF orthogonality
    unsigned int p = hphp ? j : a;
    unsigned int r = hphp ? a : MIN (i, MIN (j, b));
    unsigned int t = MAX (i, MAX (j, b));
    unsigned int q = hphp ? i : (i+j+b) - (r+t);
    if (p>0){ if ((brap.spin & ((1ULL<<p)-1)) != (ketp.spin & ((1ULL<<p)-1))){
        if (DEBUG_2E){ printf ("csf_EaiEbj right escape\n"); fflush (stdout); }
        return 0.0;
    }}
    if ((brap.spin>>(t+1)) != (ketp.spin>>(t+1))){
        if (DEBUG_2E){ printf ("csf_EaiEbj left escape\n"); fflush (stdout); }
        return 0.0;
    }

    int ni = _get_occ (ket, i);
    int nj = _get_occ (ket, j);
    int na = -_get_occ (bra, a);
    int nb = -_get_occ (bra, b);
    // Note that at this point in the code we can't have a==b or i==j
    unsigned int sx;
    int nx;
    if (i==a){ assert (ni==-na); if (ni!=-na){ return 0.0; }}
    else if (i==b){ assert (ni==-nb); if (ni!=-nb){ return 0.0; }}
    else { 
        nx = _get_occ (bra, i) + 1; 
        assert (ni == nx);
        if (ni != nx){ return 0.0; }
    }
    if (j==a){ assert (nj==-na); if (nj!=-na){ return 0.0; }}
    else if (j==b){ assert (nj==-nb); if (nj!=-nb){ return 0.0; }}
    else { 
        nx = _get_occ (bra, j) + 1; 
        assert (nj == nx); 
        if (nj != nx){ return 0.0; }
    }
    if ((a!=i) && (a!=j)){ 
        nx = _get_occ (ket, a) + 1; 
        assert (-na == nx); 
        if (-na != nx){ return 0.0; }
    }
    if ((b!=i) && (b!=j)){ 
        nx = _get_occ (ket, b) + 1; 
        assert (-nb == nx); 
        if (-nb != nx){ return 0.0; }
    }
    if (!(ni>0 && nj>0 && na<0 && nb<0)){ 
        assert (false);
        return 0.0;
    }
    // According to the graphical logic in Drake & Schlesinger, a two-electron interaction which
    // includes a number operator that hits a doubly-occupied orbital in both the bra and the ket
    // is identical to a one-electron interaction with the number part dropped. I am not good
    // enough at coding to make the CGC functions handle that elegantly, but I can write up a
    // brute-force branch, although I already simplified in the caller so I shouldn't go here
    // in practice.
    if (a==i && na==-2 && ni==2){ assert (false); return csf_Eai (bra, ket, b, j); }
    if (a==j && na==-2 && nj==2){ assert (false); return csf_Eai (bra, ket, b, i); }
    if (b==i && nb==-2 && ni==2){ assert (false); return csf_Eai (bra, ket, a, j); }
    if (b==j && nb==-2 && nj==2){ assert (false); return csf_Eai (bra, ket, a, i); }
    unsigned int nspin_ket = count_set_bits (ket->sconf);
    unsigned int nspin_bra = count_set_bits (bra->sconf);
    nspin_ket += (ni-1)*2;
    nspin_ket += (nj-1)*2;
    nspin_bra += (abs(na)-1)*2;
    nspin_bra += (abs(nb)-1)*2;
    assert (nspin_bra==nspin_ket);
    unsigned int nspin=nspin_ket;
    unsigned int * twoSk = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int * twoSb = malloc ((nspin+1) * sizeof (unsigned int));
    unsigned int si, sa, sj, sb;
    _get_spinindices2 (bra, a, b, nspin, twoSb, &sa, &sb);
    _get_spinindices2 (ket, i, j, nspin, twoSk, &si, &sj);
    if (hphp){ if (abs (nj) == 2){ sj++; } }
    else if (abs (na) == 2){ sa++; }
    // The line above looks like double-counting in the case that a==b
    // But, of course, the case that a==b (or i==j) can't get here, because
    // we divert to csf_EaiEaj in that case.
    assert (hphp || (sa >= sb));
    assert (hphp || (sa >= si));
    assert (hphp || (sa >= sj));
    assert (si != sj);
    int np = hphp ? nj : na;
    unsigned int sp = hphp ? sj : sa;
    int nr,nq,nt;
    unsigned int sr,sq,st;
    if (hphp){
        nr=na; sr=sa; nq=ni; sq=si; nt=nb; st=sb;
    } else if (r==i){
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
    // pphh: both creators to the left of both annihilators
    bool pphh = ((np < 0) == (nr < 0));
    // sr >= sq is not optional* and that logic up there ^ sometimes breaks it
    // *exception: pphh, where sr is an index into the bra chain and sq into the ket chain.
    // Between b and i the bra has two more electrons than the ket, so the same position has
    // a bra index 2 lower than its ket index, and comparing sr with sq means nothing. (The
    // swap below used to fire for most pphh elements; it exchanged b with i or j, and when it
    // swapped in i the linked term was zeroed by the (i==r) check further down.)
    if ((sq > sr) && !pphh){
        sx = sq; sq = sr; sr = sx;
        nx = nq; nq = nr; nr = nx;
        sx = r; r = q; q = sx;
    }
    // When acting on doubly-occupied orbitals, I can choose ~sq or ~sq+1 freely, because
    // the difference is formally just a factor of -1 that is canceled by the "parity" line
    // down there. But I need to choose consistently: sp >= sr >= sq >= st and sq=~sq+1; sr=~sr
    // are mutually contradictory if ~sq = ~sr and |nq| = |nr| = 2. 
    // My sense that you ought to sq++ comes from my reading of the diagrams of
    // Drake & Schlesinger, literally the lines connecting the rectangles to each other
    // always crossing the lines connecting the rectangles to the floor/ceiling.
    if (pphh){
        // q = i is in the ket and r = b in the bra, so there is no contradiction: a ket pair
        // at i points to its left edge, the same position a singly-occupied i points to.
        if (abs (nq) == 2){ sq++; }
    } else if (abs (nq) == 2){
        if (sr > sq){
            sq++; 
        } else if (abs (nr) == 1){
            assert (sr == sq);
            assert (abs (nr+nq) < (abs (nr) + abs (nq)));
            // Here just sort so that sr = ~sr, for which there is no contradiction
            nx = nr; nr = nq; nq = nx;
            sx = r; r = q; q = sx;
        } else if (abs (nq+nt) == (abs (nq) + abs (nt))){
            assert (sr == sq);
            sq++;
            // This is the exception indicated above to the rule sr>=sq.
            // Going from CGC_2e_X to CGC_2e_X_core should have implicitly caused
            // sr += 2 or sq -= 2 which I think is what makes this OK.
        } else {
            assert (sr == sq);
            assert (nr == -nq);
            sr++;
            // This is the actual contradiction in terms.
            // Claude thinks I should sr++ here based on D&S.
            // other cases should have been handled by csf_EaiEaj
            assert (false);
            // Claude also tells me we never get to this branch through test_csf.py,
            // but it could in principle. Adding this here as a tripwire so that I can
            // investigate if and when we come this way.
            //
            // Update 09/30/2026: Now Claude says we ~can't~ get here, at least not as long as
            // I keep all those occupancy checks up there. I didn't quite follow its explanation but
            // I think it's true based on D&S graphical logic. Graphically, being here means
            //
            //            r                     t
            //         ____                  ____
            //  ...  --|__|--x-- ... --x--x--|__|
            //      ____     |         |  |  |  
            //  ... |__|--x--x-- ... --x--x--x
            //         q
            //
            // (nt == 2 above) or (nt == 1 below)
            //
            //            r                     t
            //         ____
            //  ...  --|__|--x-- ... --x--x--x--x
            //      ____     |         |  |  |  
            //  ... |__|--x--x-- ... --x--x--x
            //         q
            //
            // (q is always on the bra row rather than the ket row because it has to be opposite
            // both r and t, the implicit p is always on the bra row by construction, and E E
            // conserves total number of electrons.)
            // sr and sq are the index of the spins, increasing right to left, which are
            // the "x" symbols and the left and right sides of the boxes, so here sr == sq.
            // But the orbital positions, r and q, are the columns (i.e., the horizontal
            // positions), increasing from left to right. So here we have r > q, which is
            // impossible because I quite explicitly sorted it at the top of this function
            // so that r <= q,t.
        }
    }

    // The unlinked diagram vanishes if, at any spectator spin, the cumulative spin
    // magnitudes of bra and ket are unequal.
    // (pphh evaluates its unlinked term separately; see _pphh_unlinked)
    bool unlinked_orth = false;
    if (!pphh){
        for (sx = sq; sx < sr; sx++){
            if (twoSk[sx] != twoSb[sx]){ unlinked_orth = true; }
        }
    }

    int parity = sp + sr + sq + st + ((nr>0) || (i>j));
    // This amounts to:
    // a > i > ...:   sp + sr + sq + st + 1
    // a > j > ...:   sp + sr + sq + st + 1
    // a > b > i > j: sp + sr + sq + st + 1
    // a > b > j > i: sp + sr + sq + st
    // Which I suppose is just normal-ordering the fermion operators, as implied by
    // D&S figure 1 (except that Claude's _pphh_unlinked doesn't get this, but that's
    // almost certainly because it's already counted in the underlying csf_EaiEaj
    // mapping).
    double facl = (i==r) ? 0.0 : 3.0; 
    double facu = unlinked_orth ? 0.0 : -1.0;
    if ((parity%2)==1){
        facl = -facl;
        facu = -facu;
    }

    if (DEBUG_2E){
        printf ("sp,sr,sq,st: %u,%u,%u,%u\n", sp, sr, sq, st);
        printf ("np,nr,nq,nt: %d,%d,%d,%d\n", np, nr, nq, nt);
        fflush (stdout);
    }

    // linked term
    if (i!=r){
        if (DEBUG_2E){ printf ("evaluating a linked term\n"); fflush (stdout); }
        facl *= CGC_2e_X (twoSk, twoSb, st, sq, sr, sp, nt, nq, nr, np, nspin);
        facu *= -0.5; // this comes from closure of two S=0 Wigner 3j matrices + 1 fermion swap
    }

    // unlinked term
    if (pphh){
        if (DEBUG_2E){ printf ("evaluating a pphh unlinked term\n"); fflush (stdout); }
        facu = _pphh_unlinked (bra, ket, a, i, b, j);
    } else if (!unlinked_orth){
        if (DEBUG_2E){ printf ("evaluating an unlinked term\n"); fflush (stdout); }
        facu *= CGC_1e (twoSk, twoSb, sq, st, nq, nt, nspin);
        facu *= CGC_1e (twoSk, twoSb, sp, sr, np, nr, nspin);
    }

    free (twoSk);
    free (twoSb);
    assert (facl==facl);
    assert (facu==facu);
    if (DEBUG_2E){ printf ("final value = %e + %e\n\n", facl,facu); fflush (stdout); }
    // debugging hook: -DCSF_EAIEBJ_PART=1 (linked only) or 2 (unlinked only)
    if (CSF_EAIEBJ_PART == 1){ return facl; }
    if (CSF_EAIEBJ_PART == 2){ return facu; }
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
    if (bra.sconf == 0){ bra.spin = 0ULL; }
    for (size_t iket=0; iket<ibra; iket++){
    ihmat = (iket*np) + ibra;
    ket.dconf = dconfstrs[iket];
    ket.sconf = sconfstrs[iket];
    ket.spin = coupstrs[iket];
    if (ket.sconf == 0){ ket.spin = 0ULL; }
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
                for (q=p+1; q<norb; q++){
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
                if (DEBUG_2E){ printf (" %u %u %u %u %f\n", a,p,p,i,fac); fflush (stdout); }
                fac += csf_EaiEbj (&bra, &ket, a, i, p, p);
                if (DEBUG_2E){ printf (" %u %u %u %u %f\n", a,i,p,p,fac); fflush (stdout); }
                hmat[ihmat] += g2e[ihop] * fac;
            }
            break;
        case 2:
            exc2_sort (&bra, &ket, p, r, q, t, &a, &i, &b, &j);
            // E^a_i E^b_j
            ihop = (a*norb*norb*norb) + (i*norb*norb) + (b*norb) + j;
            if (CSF_2E_RI){
                fac = csf_EaiEbj_ri (&bra, &ket, a, i, b, j);
                // a==b and i==j: only one ordering, so the 1/2 of the Hamiltonian survives
                if ((a==b) && (i==j)){ fac *= 0.5; }
            } else {
                fac = csf_EaiEbj (&bra, &ket, a, i, b, j);
            }
            if (DEBUG_2E){ printf (" %u %u %u %u %f\n", a,i,b,j,fac); fflush (stdout); }
            hmat[ihmat] += g2e[ihop] * fac;
            // E^a_j E^b_i
            if ((a!=b) && (i!=j)){
                ihop = (a*norb*norb*norb) + (j*norb*norb) + (b*norb) + i;
                fac = CSF_2E_RI ? csf_EaiEbj_ri (&bra, &ket, a, j, b, i) : csf_EaiEbj (&bra, &ket, a, j, b, i);
                if (DEBUG_2E){ printf (" %u %u %u %u %f\n", a,j,b,i,fac); fflush (stdout); }
                hmat[ihmat] += g2e[ihop] * fac;
            }
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

