#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include "csf.h"

void FCICSFddstrs2csdstrs (uint64_t * csdstrs, uint64_t * ddstrs, size_t nstr, int norb, int neleca, int nelecb)
{

    size_t i;
    int iorb, isorb, ispin;
    uint64_t * astrs = ddstrs;
    uint64_t * bstrs = & ddstrs[nstr];
    uint64_t * npairs = csdstrs;
    uint64_t * dconf_strs = & csdstrs[nstr];
    uint64_t * sconf_strs = & csdstrs[2*nstr];
    uint64_t * spins_strs = & csdstrs[3*nstr];

    for (i = 0; i < nstr; i++){
        npairs[i] = 0;
        dconf_strs[i] = 0;
        sconf_strs[i] = 0;
        spins_strs[i] = 0;

        isorb = 0;
        ispin = 0;

        for (iorb = 0; iorb < norb; iorb++){
            if ((1ULL << iorb) & astrs[i] & bstrs[i]) { /* DOUBLY OCCUPIED */
                npairs[i]++;
                dconf_strs[i] |= 1ULL << iorb; /* This is adding 2 electrons at position iorb */
            } else if ((1ULL << iorb) & (astrs[i] | bstrs[i])) { /* SINGLY OCCUPIED */
                sconf_strs[i] |= 1ULL << isorb; /* This is adding 1 electron at non-double position isorb */
                isorb++;
                if ((1ULL << iorb) & astrs[i]) { spins_strs[i] |= 1ULL << ispin ; } /* This is adding 1 alpha spin state at spin ispin */
                ispin++;
            } else { isorb++; } /* VIRTUAL */
        }
    }
}

void FCICSFcsdstrs2ddstrs (uint64_t * ddstrs, uint64_t * csdstrs, size_t nstr, int norb, int neleca, int nelecb)
{

    size_t i;
    int iorb, isorb, ispin;
    uint64_t * astrs = ddstrs;
    uint64_t * bstrs = & ddstrs[nstr];
    uint64_t * dconf_strs = & csdstrs[nstr];
    uint64_t * sconf_strs = & csdstrs[2*nstr];
    uint64_t * spins_strs = & csdstrs[3*nstr];

    for (i = 0; i < nstr; i++){
        astrs[i] = 0;
        bstrs[i] = 0;

        isorb = 0;
        ispin = 0;

        for (iorb = 0; iorb < norb; iorb++){
            if ((1ULL << iorb) & dconf_strs[i]){
                astrs[i] |= 1ULL << iorb;
                bstrs[i] |= 1ULL << iorb;
            } else {
                if ((1ULL << isorb) & sconf_strs[i]){
                    if ((1ULL << ispin) & spins_strs[i]){
                        astrs[i] |= 1ULL << iorb;
                    } else {
                        bstrs[i] |= 1ULL << iorb;
                    }
                    ispin++;
                }
                isorb++;
            }
        }
    }
}


void FCICSFmakecsf (double * umat, uint64_t * detstr, uint64_t * coupstr, int nspin, size_t ndet, size_t ncoup, int twoS, int twoMS)
{


#pragma omp parallel default(shared)
{

    size_t idet, icoup, ispin, idetcoup, ndetcoup;
    int track2S, track2MS, sgn, osgn;
    uint64_t sup, msup;
    double numerator, denominator;

    ndetcoup = ndet * ncoup;

#pragma omp for schedule(static) 

    for (idetcoup = 0; idetcoup < ndetcoup; idetcoup++){
        /* This is a shitty way to do a nested loop but the gcc version that pyscf has to stay compatible with can't use "collapse" statements */
        idet = idetcoup / ncoup;
        icoup = idetcoup % ncoup;
        track2S = 1;
        track2MS = 1ULL & detstr[idet] ? 1 : -1;
        numerator = 1;
        denominator = 1;
        sgn = 1;
        // Commute each spin-down electron past each spin-up electron
        osgn = -track2MS;
        for (ispin = 1; ispin < nspin; ispin++){
            sup = (1ULL << ispin) & coupstr[icoup];
            msup = (1ULL << ispin) & detstr[idet];
            if (msup){ track2MS++; osgn *= -1; } else { track2MS--; sgn *= osgn; }
            /* Clebsch-Gordan coefficient <j1,j2,m1,m2|J,M=m1+m2> (j2 = 1/2, m2 = +-1/2)
 *              = sgn * sqrt (num / denom)
 *              sgn = sgn(J2-j1)^delta(m2,+1/2)
 *              num = j1 + 1/2 + sgn(J-j1)*sgn(m2)*M
 *              denom = 2*j1 + 1 
 *              All numbers are half-integers so multiply num and denom by 2
 *          */
            numerator *= (sup == msup) ? track2S + track2MS + 1 : track2S + 1 - track2MS;
            if (numerator == 0){ break; }
            denominator *= (track2S + 1);
            if (msup && !sup){ sgn *= -1; }

            /* All numbers are half-integers so num is *2 computed in this way.
 *          */
            numerator /= 2;

            if (sup){ track2S++; } else { track2S--; }
        } 
        if (numerator == 0) { umat[idetcoup] = 0.0; } else {
            umat[idetcoup] = sgn * sqrt ((double) numerator / denominator);
        }
    }

}
}

/* FCICSFspinevecs_dot: out = in . umat (transpose = 0) or out = in . umat^T (transpose = 1), where umat
 * is the (ndet, ncoup) array that FCICSFmakecsf would build, without ever building it.
 *
 * Each element of umat is a product of nspin Clebsch-Gordan coefficients, one per spin, so umat factors
 * into nspin sparse steps, each of which couples one more spin. Before step k, a basis state is labeled
 * by an nspin-bit key: bits 0..k-1 are the coupling path of the first k spins (running spin
 * 2S_k = 2*nup - k) and bits k..nspin-1 are the remaining det spins, which fix the running 2M_k because
 * the total 2M is fixed. Step k pairs the keys K (bit k = 0) and K | 1<<k (bit k = 1). Before the step
 * they mean m_k = -1/2 and +1/2; after it they mean S_{k+1} = S_k - 1/2 and S_k + 1/2. Each pair is
 * acted on by the 2x2 matrix c[J][m] = <S_k, M_{k+1}-m_k; 1/2, m_k | S_{k+1}, M_{k+1}>. States that can
 * no longer reach S = twoS/2 are dropped, so no step ever holds more than ndet states. The fermion sign
 * of FCICSFmakecsf is diagonal in the det basis and is applied before (or, for the transpose, after) the
 * sweep. The cost is O(nspin*ndet) per row instead of O(ndet*ncoup).
 */
// int32 indices cap ndet at INT32_MAX (e.g., nspin = 34, 2M = 0); spin_evecs_dot raises before calling
typedef struct {
    int32_t in[2];  // index of the m_k = -1/2, +1/2 states before the step (-1 if absent)
    int32_t out[2]; // index of the S_k -/+ 1/2 states after the step (-1 if absent)
    int16_t twoS;   // 2S_k
    int16_t twoM;   // 2M_{k+1}
} _sweep_pair;

typedef struct {
    uint64_t key;
    int64_t idx;
} _key_idx;

static int _cmp_key_idx (const void * a, const void * b)
{
    uint64_t x = ((const _key_idx *) a)->key;
    uint64_t y = ((const _key_idx *) b)->key;
    return (x > y) - (x < y);
}

static int64_t _find_key (uint64_t * keys, size_t nkey, uint64_t key)
{
    size_t lo = 0, hi = nkey;
    while (lo < hi){
        size_t mid = lo + (hi - lo) / 2;
        if (keys[mid] < key){ lo = mid + 1; } else { hi = mid; }
    }
    if ((lo < nkey) && (keys[lo] == key)){ return (int64_t) lo; }
    return -1;
}

/* <S, M-m; 1/2, m | S +/- 1/2, M> in doubled units (twoS = 2S, twoM = 2M, up_m: m = +1/2, up_J: J = S + 1/2),
 * with the same sign convention as FCICSFmakecsf */
static double _cg_half (int twoS, int twoM, int up_m, int up_J)
{
    int sm = up_m ? twoM : -twoM;
    int num = up_J ? (twoS + sm + 1) : (twoS - sm + 1);
    if (num <= 0){ return 0.0; }
    double c = sqrt ((double) num / (2.0 * (twoS + 1)));
    return (up_m && !up_J) ? -c : c;
}

#ifndef SWEEP_BLK_ELEMS
#define SWEEP_BLK_ELEMS (1<<17) // doubles per sweep buffer (two per thread)
#endif
#ifndef SWEEP_BLK_MAXROW
#define SWEEP_BLK_MAXROW 64
#endif

// y = c0*x0 + c1*x1, where a NULL x counts as zero
static void _sweep_comb (double * y, double * x0, double * x1, double c0, double c1, size_t n)
{
    size_t r;
    if (x0 && x1){ for (r = 0; r < n; r++){ y[r] = c0*x0[r] + c1*x1[r]; } }
    else if (x0){ for (r = 0; r < n; r++){ y[r] = c0*x0[r]; } }
    else if (x1){ for (r = 0; r < n; r++){ y[r] = c1*x1[r]; } }
    else { for (r = 0; r < n; r++){ y[r] = 0.0; } }
}

void FCICSFspinevecs_dot (double * out, double * in, uint64_t * detstr, uint64_t * coupstr,
                          int nspin, size_t ndet, size_t ncoup, int twoS, int twoMS,
                          size_t nrow, int transpose)
{
    assert (ndet < INT32_MAX);
    assert ((nspin >= 0) && (nspin < 64));
    int k;
    size_t i, j;

    // fermion sign: each spin-down electron commutes past every spin-up electron to its right
    double * fsgn = malloc (ndet * sizeof (double));
    for (i = 0; i < ndet; i++){
        int nup = 0;
        fsgn[i] = 1.0;
        for (k = 0; k < nspin; k++){
            if ((detstr[i] >> k) & 1ULL){ nup++; } else if (nup % 2){ fsgn[i] = -fsgn[i]; }
        }
    }

    // coefficient table, indexed [twoS_k][twoM_{k+1}+moff][up_J][up_m]
    int moff = 2*nspin + 1;
    int nm = 2*moff + 1;
    double * cgtab = malloc ((size_t) (nspin+1) * nm * 4 * sizeof (double));
    for (int s = 0; s <= nspin; s++){ for (int m = 0; m < nm; m++){
        for (int up_J = 0; up_J < 2; up_J++){ for (int up_m = 0; up_m < 2; up_m++){
            cgtab[((s*nm + m)*2 + up_J)*2 + up_m] = _cg_half (s, m - moff, up_m, up_J);
        }}
    }}

    // build the steps
    _sweep_pair ** pairs = malloc ((nspin+1) * sizeof (_sweep_pair *));
    size_t * npair = malloc ((nspin+1) * sizeof (size_t));
    size_t nkey = ndet;
    size_t maxkey = ndet;
    uint64_t * keys = malloc (ndet * sizeof (uint64_t));
    for (i = 0; i < ndet; i++){
        keys[i] = detstr[i];
        if (i > 0){ assert (keys[i] > keys[i-1]); }
    }
    for (k = 0; k < nspin; k++){
        uint64_t bit = 1ULL << k;
        uint64_t low = bit - 1ULL;
        int nrest = nspin - k - 1;
        // pair representatives (bit k cleared), sorted and unique
        _key_idx * kr = malloc ((nkey+1) * sizeof (_key_idx));
        for (i = 0; i < nkey; i++){ kr[i].key = keys[i] & ~bit; kr[i].idx = 0; }
        qsort (kr, nkey, sizeof (_key_idx), _cmp_key_idx);
        uint64_t * reps = malloc ((nkey+1) * sizeof (uint64_t));
        size_t nrep = 0;
        for (i = 0; i < nkey; i++){
            if ((nrep == 0) || (reps[nrep-1] != kr[i].key)){ reps[nrep++] = kr[i].key; }
        }
        free (kr);
        _sweep_pair * p = malloc (nrep * sizeof (_sweep_pair));
        _key_idx * newkeys = malloc (2 * nrep * sizeof (_key_idx));
        size_t nnew = 0;
        for (j = 0; j < nrep; j++){
            uint64_t r = reps[j];
            int twoSk = 2 * (int) count_set_bits (r & low) - k;
            int twoMk = twoMS - (2 * (int) count_set_bits (r >> (k+1)) - nrest);
            p[j].twoS = (int16_t) twoSk;
            p[j].twoM = (int16_t) twoMk;
            p[j].in[0] = (int32_t) _find_key (keys, nkey, r);
            p[j].in[1] = (int32_t) _find_key (keys, nkey, r | bit);
            for (int up_J = 0; up_J < 2; up_J++){
                int twoS1 = twoSk + (up_J ? 1 : -1);
                if ((twoS1 >= 0) && (twoS1 >= abs (twoMk)) && (abs (twoS1 - twoS) <= nrest)){
                    p[j].out[up_J] = (int32_t) nnew;
                    newkeys[nnew].key = up_J ? (r | bit) : r;
                    newkeys[nnew].idx = (int64_t) nnew;
                    nnew++;
                } else {
                    p[j].out[up_J] = -1;
                }
            }
        }
        free (reps);
        // sort the new keys and point the pairs' outputs at their sorted positions
        qsort (newkeys, nnew, sizeof (_key_idx), _cmp_key_idx);
        int32_t * rank = malloc ((nnew+1) * sizeof (int32_t));
        free (keys);
        keys = malloc ((nnew+1) * sizeof (uint64_t));
        for (i = 0; i < nnew; i++){
            rank[newkeys[i].idx] = (int32_t) i;
            keys[i] = newkeys[i].key;
        }
        for (j = 0; j < nrep; j++){
            for (int up_J = 0; up_J < 2; up_J++){
                if (p[j].out[up_J] >= 0){ p[j].out[up_J] = rank[p[j].out[up_J]]; }
            }
        }
        free (rank);
        free (newkeys);
        pairs[k] = p;
        npair[k] = nrep;
        nkey = nnew;
        if (nkey > maxkey){ maxkey = nkey; }
    }
    // the surviving keys are exactly the coupling strings; map them to csf addresses
    assert (nkey == ncoup);
    int64_t * csfaddr = malloc ((ncoup+1) * sizeof (int64_t));
    {
        _key_idx * ki = malloc ((ncoup+1) * sizeof (_key_idx));
        // (the nspin = 0 coupling string is stored as 1)
        uint64_t mask = (nspin > 0) ? (~0ULL >> (64 - nspin)) : 0ULL;
        for (j = 0; j < ncoup; j++){ ki[j].key = coupstr[j] & mask; ki[j].idx = (int64_t) j; }
        qsort (ki, ncoup, sizeof (_key_idx), _cmp_key_idx);
        for (j = 0; j < ncoup; j++){
            assert (ki[j].key == keys[j]);
            csfaddr[j] = ki[j].idx;
        }
        free (ki);
    }
    free (keys);

    // apply the steps to blocks of R rows at a time; the buffers are laid out [key][row in block] so that
    // each pair acts on contiguous memory. Parallel over blocks if there are enough of them, otherwise
    // over the pairs of each step.
    size_t ninp = transpose ? ncoup : ndet;
    size_t nout = transpose ? ndet : ncoup;
    size_t R = SWEEP_BLK_ELEMS / (maxkey+1);
    if (R > SWEEP_BLK_MAXROW){ R = SWEEP_BLK_MAXROW; }
    if (R > nrow){ R = nrow; }
    if (R < 1){ R = 1; }
    size_t nblk = (nrow + R - 1) / R;
    int blk_parallel = (nblk >= (size_t) omp_get_max_threads ());
#pragma omp parallel if (blk_parallel) default(shared) private(i,j,k)
{
    double * a = malloc ((maxkey+1) * R * sizeof (double));
    double * b = malloc ((maxkey+1) * R * sizeof (double));
    double * t;
    size_t iblk, r;
#pragma omp for schedule(dynamic)
    for (iblk = 0; iblk < nblk; iblk++){
        size_t r0 = iblk * R;
        size_t nr = ((r0 + R) > nrow) ? (nrow - r0) : R;
        double * x = in + r0 * ninp;
        double * y = out + r0 * nout;
        if (!transpose){
            for (r = 0; r < nr; r++){ for (i = 0; i < ndet; i++){
                a[i*R + r] = x[r*ninp + i] * fsgn[i];
            }}
            for (k = 0; k < nspin; k++){
                _sweep_pair * p = pairs[k];
#pragma omp parallel for if (!blk_parallel) schedule(static)
                for (j = 0; j < npair[k]; j++){
                    double * x0 = (p[j].in[0] >= 0) ? a + (size_t) p[j].in[0] * R : NULL;
                    double * x1 = (p[j].in[1] >= 0) ? a + (size_t) p[j].in[1] * R : NULL;
                    double * c = cgtab + ((size_t) p[j].twoS*nm + p[j].twoM + moff)*4;
                    if (p[j].out[0] >= 0){ _sweep_comb (b + (size_t) p[j].out[0] * R, x0, x1, c[0], c[1], nr); }
                    if (p[j].out[1] >= 0){ _sweep_comb (b + (size_t) p[j].out[1] * R, x0, x1, c[2], c[3], nr); }
                }
                t = a; a = b; b = t;
            }
            for (r = 0; r < nr; r++){ for (j = 0; j < ncoup; j++){
                y[r*nout + csfaddr[j]] = a[j*R + r];
            }}
        } else {
            for (r = 0; r < nr; r++){ for (j = 0; j < ncoup; j++){
                a[j*R + r] = x[r*ninp + csfaddr[j]];
            }}
            for (k = nspin-1; k >= 0; k--){
                _sweep_pair * p = pairs[k];
#pragma omp parallel for if (!blk_parallel) schedule(static)
                for (j = 0; j < npair[k]; j++){
                    double * y0 = (p[j].out[0] >= 0) ? a + (size_t) p[j].out[0] * R : NULL;
                    double * y1 = (p[j].out[1] >= 0) ? a + (size_t) p[j].out[1] * R : NULL;
                    double * c = cgtab + ((size_t) p[j].twoS*nm + p[j].twoM + moff)*4;
                    if (p[j].in[0] >= 0){ _sweep_comb (b + (size_t) p[j].in[0] * R, y0, y1, c[0], c[2], nr); }
                    if (p[j].in[1] >= 0){ _sweep_comb (b + (size_t) p[j].in[1] * R, y0, y1, c[1], c[3], nr); }
                }
                t = a; a = b; b = t;
            }
            for (r = 0; r < nr; r++){ for (i = 0; i < ndet; i++){
                y[r*nout + i] = a[i*R + r] * fsgn[i];
            }}
        }
    }
    free (a);
    free (b);
}

    for (k = 0; k < nspin; k++){ free (pairs[k]); }
    free (pairs);
    free (npair);
    free (csfaddr);
    free (cgtab);
    free (fsgn);
}


void FCICSFgetscstrs (uint64_t * scstrs, bool * mask, size_t nstr, int nspin)
{

    size_t istr;
    int ispin, srun;
    for (istr = 0; istr < nstr; istr++){
        srun = 0;
        for (ispin = 0; ispin < nspin; ispin++){
            if (1ULL << ispin & scstrs[istr]){ srun++; } else { srun--; }
            if (srun < 0){ mask[istr] = false; break; }
        }
    }

}

void FCICSFstrs2addr (int * addrs, uint64_t * strings, size_t nstr, int * gentable_ravel, int nspin, int twoS)
{

    /*  Example of a genealogical coupling table for 8 spins and s = 1 (triplet), counting from the final state
        back to the null state:

           28 28 19 10  4  1  .
            |  9  9  6  3  1  .
            |  |  3  3  2  1  .
            |  |  |  1  1  t  .
                        .  .  .
                           .  .
                              .

        n0 = 3 zero bits; n1 = 5 one bits -> 4x6 matrix. Going to the right is a 1 bit, going down is a 0 bit (bits sorted right to left).
        Notice how gen[i0,i1] = sum_(j0=i0)^n0 gen[j0,i1+1] (t == 1) except for the final column (gen[i0,n1] = 1).
        Example: 11001011: right twice, then down (instead of hitting 10 to the right), then right, then down twice (instead of hitting 3 or 2 to the right),
        then right twice. Address = 10 + 3 + 2 = 15.
        The next would be 01110011: right twice, then down twice (avoiding 10 and 6), then right three times, then down once (in the final column).
        Address = 10 + 6 + 0 = 16.
        Top left (0,0) is the null state (nspin = 0, s = 0).
    */
    size_t istr;
    int n0 = (nspin - twoS) / 2;
    int n1 = (nspin + twoS) / 2;
    int i0, i1;
    int ** gentable = malloc ((n0+1) * sizeof (int*));
    for (i0 = 0; i0 <= n0; i0++){
        gentable[i0] = & (gentable_ravel[i0*(n1+1)]);
    }
    int ispin;

    for (istr = 0; istr < nstr; istr++){
        addrs[istr] = 0;
        assert (1ULL & strings[istr]);
        i0 = 0, i1 = 0;
        for (ispin = 0; ispin < nspin; ispin++){
            if (1ULL << ispin & strings[istr]){  
                i1++;
            } else {
                if (i1 < n1) { addrs[istr] += gentable[i0][i1+1]; }
                i0++;
            }
        }
        assert (i0 == n0);
        assert (i1 == n1);
    }
    free (gentable);
}


void FCICSFaddrs2str (uint64_t * strings, int * addrs, size_t nstr, int * gentable_ravel, int nspin, int twoS)
{

    /*  Example of a genealogical coupling table for 8 spins and s = 1 (triplet), counting from the final state
        back to the null state:

           28 28 19 10  4  1  .
            |  9  9  6  3  1  .
            |  |  3  3  2  1  .
            |  |  |  1  1  t  .
                        .  .  .
                           .  .
                              .

        n0 = 3 zero bits; n1 = 5 one bits -> 4x6 matrix. Going to the right is a 1 bit, going down is a 0 bit (bits sorted right to left).
        Notice how gen[i0,i1] = sum_(j0=i0)^n0 gen[j0,i1+1] (t == 1) except for the final column (gen[i0,n1] = 1).
        Example: 11001011: right twice, then down (instead of hitting 10 to the right), then right, then down twice (instead of hitting 3 or 2 to the right),
        then right twice. Address = 10 + 3 + 2 = 15.
        The next would be 01110011: right twice, then down twice (avoiding 10 and 6), then right three times, then down once (in the final column).
        Address = 10 + 6 + 0 = 16.
    */
    size_t istr;
    int n0 = (nspin - twoS) / 2;
    int n1 = (nspin + twoS) / 2;
    int i0, i1;
    int ** gentable = malloc ((n0+1) * sizeof (int*));
    for (i0 = 0; i0 <= n0; i0++){
        gentable[i0] = &(gentable_ravel[i0*(n1+1)]);
    }
    int ispin, caddrs;

    for (istr = 0; istr < nstr; istr++){
        strings[istr] = 1ULL;
        caddrs = addrs[istr];
        i0 = 0, i1 = 0;
        for (ispin = 0; ispin < nspin; ispin++){
            if (i1 == n1){ break; }
            else if (gentable[i0][i1+1] <= caddrs){
                caddrs -= gentable[i0][i1+1];
                assert (i0 < i1);
                i0++;
            } else {
                strings[istr] |= 1ULL << ispin;
                i1++;
            }
            
        }
        assert (caddrs == 0);
    }
    free (gentable);
}

void FCICSFsignrule (int * sgns, uint64_t * dconfs, uint64_t * sconfs, size_t ndet, unsigned int norb)
{
/* Anticommute electron creation operators that participate in pairs out of normal order to before any
   unpaired electrons so that CSFs of different seniority can have comparable signs

   Output:
    sgns : array of shape (ndet,)
   Input:
    dconfs : array of shape (ndet,)
    sconfs : array of shape (ndet,)
*/

#pragma omp parallel default(shared)
{

    unsigned int npair;
    uint64_t dconf, sconf, tconf;
    int k;

#pragma omp for schedule(static) 

    for (size_t idet = 0; idet < ndet; idet++){
        dconf = dconfs[idet];
        sconf = sconfs[idet];
        npair = count_set_bits (dconf);
        sgns[idet] = 0;
        for (unsigned int ipair = 0; ipair < npair; ipair++){
            k = first1 (dconf);
            assert (k>=0);
            tconf = sconf & ((1ULL<<k)-1);
            sgns[idet] += count_set_bits (tconf);
            sconf >>= k;
            sconf <<= (k+1);
            sconf |= tconf;
            dconf &= dconf-1;
        }
        sgns[idet] = ((sgns[idet] & 1ULL) == 1) ? -1 : 1;
    }

}

}

