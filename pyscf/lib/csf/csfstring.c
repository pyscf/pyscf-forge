#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
//#include <stdio.h>
#include <omp.h>
//#include "config.h"
//#include "vhf/fblas.h"
//#include "fci.h"

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

