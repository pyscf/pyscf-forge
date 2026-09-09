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

int first1 (uint64_t r)
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

int last1 (uint64_t r)
{
#if defined(__builtin_flsll)
        return __builtin_flsll(r) - 1;
#elif defined(HAVE_FFS)
        return flsll(r) - 1;
#else
    if (r==0ULL){ return -1; }
    int n = 0;
    if (r >> (n + 32)) n += 32;
    if (r >> (n + 16)) n += 16;
    if (r >> (n +  8)) n +=  8;
    if (r >> (n +  4)) n +=  4;        
    if (r >> (n +  2)) n +=  2;
    if (r >> (n +  1)) n +=  1;
    return n;
#endif
}


