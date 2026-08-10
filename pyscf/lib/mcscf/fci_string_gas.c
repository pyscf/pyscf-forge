/* Copyright 2026 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

 *
 * Author: Yi Deng <yideng@uchicago.edu>
 */

#include "fci_gas.h"

#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
static const uint64_t LINK_BUILD_PARALLEL_MIN = 1048576u;
#endif

/* ========================================================================== */
/* 1. Low-level utilities                                                     */
/* ========================================================================== */

static inline int popcnt64(uint64_t x)
{
        return __builtin_popcountll(x);
}

static inline int ctz64(uint64_t x)
{
        return __builtin_ctzll(x);
}

static inline uint64_t lowbits(int n)
{
        return (1ULL << n) - 1ULL;
}

/*
 * Branchless C(n,k) lookup; valid for 0 <= n,k <= GAS_MAX_LOCAL_ORB
 */
static const uint32_t binom_tab[(GAS_MAX_LOCAL_ORB + 1) * 32] = {
        /* n = 0 */
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 1 */
        1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 2 */
        1u, 2u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 3 */
        1u, 3u, 3u, 1u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 4 */
        1u, 4u, 6u, 4u, 1u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 5 */
        1u, 5u, 10u, 10u, 5u, 1u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 6 */
        1u, 6u, 15u, 20u, 15u, 6u, 1u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 7 */
        1u, 7u, 21u, 35u, 35u, 21u, 7u, 1u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 8 */
        1u, 8u, 28u, 56u, 70u, 56u, 28u, 8u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 9 */
        1u, 9u, 36u, 84u, 126u, 126u, 84u, 36u,
        9u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 10 */
        1u, 10u, 45u, 120u, 210u, 252u, 210u, 120u,
        45u, 10u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 11 */
        1u, 11u, 55u, 165u, 330u, 462u, 462u, 330u,
        165u, 55u, 11u, 1u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 12 */
        1u, 12u, 66u, 220u, 495u, 792u, 924u, 792u,
        495u, 220u, 66u, 12u, 1u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 13 */
        1u, 13u, 78u, 286u, 715u, 1287u, 1716u, 1716u,
        1287u, 715u, 286u, 78u, 13u, 1u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 14 */
        1u, 14u, 91u, 364u, 1001u, 2002u, 3003u, 3432u,
        3003u, 2002u, 1001u, 364u, 91u, 14u, 1u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 15 */
        1u, 15u, 105u, 455u, 1365u, 3003u, 5005u, 6435u,
        6435u, 5005u, 3003u, 1365u, 455u, 105u, 15u, 1u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 16 */
        1u, 16u, 120u, 560u, 1820u, 4368u, 8008u, 11440u,
        12870u, 11440u, 8008u, 4368u, 1820u, 560u, 120u, 16u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 17 */
        1u, 17u, 136u, 680u, 2380u, 6188u, 12376u, 19448u,
        24310u, 24310u, 19448u, 12376u, 6188u, 2380u, 680u, 136u,
        17u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 18 */
        1u, 18u, 153u, 816u, 3060u, 8568u, 18564u, 31824u,
        43758u, 48620u, 43758u, 31824u, 18564u, 8568u, 3060u, 816u,
        153u, 18u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 19 */
        1u, 19u, 171u, 969u, 3876u, 11628u, 27132u, 50388u,
        75582u, 92378u, 92378u, 75582u, 50388u, 27132u, 11628u, 3876u,
        969u, 171u, 19u, 1u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 20 */
        1u, 20u, 190u, 1140u, 4845u, 15504u, 38760u, 77520u,
        125970u, 167960u, 184756u, 167960u, 125970u, 77520u, 38760u, 15504u,
        4845u, 1140u, 190u, 20u, 1u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 21 */
        1u, 21u, 210u, 1330u, 5985u, 20349u, 54264u, 116280u,
        203490u, 293930u, 352716u, 352716u, 293930u, 203490u, 116280u, 54264u,
        20349u, 5985u, 1330u, 210u, 21u, 1u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 22 */
        1u, 22u, 231u, 1540u, 7315u, 26334u, 74613u, 170544u,
        319770u, 497420u, 646646u, 705432u, 646646u, 497420u, 319770u, 170544u,
        74613u, 26334u, 7315u, 1540u, 231u, 22u, 1u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 23 */
        1u, 23u, 253u, 1771u, 8855u, 33649u, 100947u, 245157u,
        490314u, 817190u, 1144066u, 1352078u, 1352078u, 1144066u, 817190u, 490314u,
        245157u, 100947u, 33649u, 8855u, 1771u, 253u, 23u, 1u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 24 */
        1u, 24u, 276u, 2024u, 10626u, 42504u, 134596u, 346104u,
        735471u, 1307504u, 1961256u, 2496144u, 2704156u, 2496144u, 1961256u, 1307504u,
        735471u, 346104u, 134596u, 42504u, 10626u, 2024u, 276u, 24u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 25 */
        1u, 25u, 300u, 2300u, 12650u, 53130u, 177100u, 480700u,
        1081575u, 2042975u, 3268760u, 4457400u, 5200300u, 5200300u, 4457400u, 3268760u,
        2042975u, 1081575u, 480700u, 177100u, 53130u, 12650u, 2300u, 300u,
        25u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,

        /* n = 26 */
        1u, 26u, 325u, 2600u, 14950u, 65780u, 230230u, 657800u,
        1562275u, 3124550u, 5311735u, 7726160u, 9657700u, 10400600u, 9657700u, 7726160u,
        5311735u, 3124550u, 1562275u, 657800u, 230230u, 65780u, 14950u, 2600u,
        325u, 26u, 1u, 0u, 0u, 0u, 0u, 0u,

        /* n = 27 */
        1u, 27u, 351u, 2925u, 17550u, 80730u, 296010u, 888030u,
        2220075u, 4686825u, 8436285u, 13037895u, 17383860u, 20058300u, 20058300u, 17383860u,
        13037895u, 8436285u, 4686825u, 2220075u, 888030u, 296010u, 80730u, 17550u,
        2925u, 351u, 27u, 1u, 0u, 0u, 0u, 0u,

        /* n = 28 */
        1u, 28u, 378u, 3276u, 20475u, 98280u, 376740u, 1184040u,
        3108105u, 6906900u, 13123110u, 21474180u, 30421755u, 37442160u, 40116600u, 37442160u,
        30421755u, 21474180u, 13123110u, 6906900u, 3108105u, 1184040u, 376740u, 98280u,
        20475u, 3276u, 378u, 28u, 1u, 0u, 0u, 0u,

        /* n = 29 */
        1u, 29u, 406u, 3654u, 23751u, 118755u, 475020u, 1560780u,
        4292145u, 10015005u, 20030010u, 34597290u, 51895935u, 67863915u, 77558760u, 77558760u,
        67863915u, 51895935u, 34597290u, 20030010u, 10015005u, 4292145u, 1560780u, 475020u,
        118755u, 23751u, 3654u, 406u, 29u, 1u, 0u, 0u,

        /* n = 30 */
        1u, 30u, 435u, 4060u, 27405u, 142506u, 593775u, 2035800u,
        5852925u, 14307150u, 30045015u, 54627300u, 86493225u, 119759850u, 145422675u, 155117520u,
        145422675u, 119759850u, 86493225u, 54627300u, 30045015u, 14307150u, 5852925u, 2035800u,
        593775u, 142506u, 27405u, 4060u, 435u, 30u, 1u, 0u,

        /* n = 31 */
        1u, 31u, 465u, 4495u, 31465u, 169911u, 736281u, 2629575u,
        7888725u, 20160075u, 44352165u, 84672315u, 141120525u, 206253075u, 265182525u, 300540195u,
        300540195u, 265182525u, 206253075u, 141120525u, 84672315u, 44352165u, 20160075u, 7888725u,
        2629575u, 736281u, 169911u, 31465u, 4495u, 465u, 31u, 1u,
};

static inline uint32_t binom(int n, int k)
{
        return binom_tab[(n << 5) + k];
}

/*
 * Fermion sign for a_cre^+ a_des |I> = sign |J>
 */
static inline int8_t cd_sign(int p, int q, uint64_t str)
{
        int hi = p > q ? p : q;
        int lo = p > q ? q : p;
        uint64_t mask = (1ULL << hi) - (1ULL << (lo + 1));
        return (int8_t)(1 - 2 * (popcnt64(str & mask) & 1));
}

/*
 * Lexical address of a spin string using Cantor expansion
 */
static inline uint32_t str2addr_local(int nelec, uint64_t str)
{
        uint32_t addr = 0;

        for (int i = 1; i <= nelec; i++) {
                int p = ctz64(str);
                addr += binom(p, i);
                str &= str - 1;
        }
        return addr;
}

/*
 * Spin string from lexical address using inverse Cantor expansion
 */
static inline uint64_t addr2str_local(int norb, int nelec, uint32_t addr)
{
        uint64_t str = 0;
        uint32_t a = addr;
        int e = nelec;

        for (int p = norb - 1; e > 0; p--) {
                if (a == 0 || p < e) {
                        str |= lowbits(e);
                        break;
                }

                uint32_t c = binom(p, e);
                if (c <= a) {
                        str |= 1ULL << p;
                        a -= c;
                        e--;
                }
        }
        return str;
}

static inline int occ_eq_u8(const uint8_t *a, const uint8_t *b, int n)
{
        return memcmp(a, b, (size_t)n) == 0;
}

static void gas_zero(gas_space_t *gas)
{
        memset(gas, 0, sizeof(*gas));
}

static inline uint8_t *sector_occ_ptr(const gas_space_t *gas, gas_sid_t s)
{
        return gas->sector_occ + (size_t)s * gas->ngas;
}

static inline uint32_t *sector_stride_ptr(const gas_space_t *gas, gas_sid_t s)
{
        return gas->sector_stride + (size_t)s * gas->ngas;
}

static int occ_lex_cmp_width(const uint8_t *a, const uint8_t *b, int width)
{
        for (int i = 0; i < width; i++) {
                if (a[i] < b[i]) return -1;
                if (a[i] > b[i]) return 1;
        }
        return 0;
}

/*
 * qsort comparator context; used only in serial build stages
 */
static int qsort_occ_width;

static int cmp_occ_rows(const void *pa, const void *pb)
{
        const uint8_t *a = (const uint8_t *)pa;
        const uint8_t *b = (const uint8_t *)pb;
        return occ_lex_cmp_width(a, b, qsort_occ_width);
}

static int cmp_u32(const void *pa, const void *pb)
{
        uint32_t a = *(const uint32_t *)pa;
        uint32_t b = *(const uint32_t *)pb;
        return (a > b) - (a < b);
}

/* ========================================================================== */
/* 2. Temporary row-list helpers                                              */
/* ========================================================================== */

/*
 * Dynamic array for temporary storage and sort
 */
typedef struct {
        uint8_t *data;
        uint32_t n;
        uint32_t cap;
        int width;
} gas_u8_rows_t;

static inline uint8_t *u8_row(gas_u8_rows_t *R, uint32_t i)
{
        return R->data + (size_t)i * R->width;
}

static inline const uint8_t *u8_crow(const gas_u8_rows_t *R, uint32_t i)
{
        return R->data + (size_t)i * R->width;
}

static inline const uint8_t *block_alpha_occ(const gas_u8_rows_t *blocks,
                                             uint32_t b)
{
        return u8_crow(blocks, b);
}

static inline const uint8_t *block_beta_occ(const gas_space_t *gas,
                                            const gas_u8_rows_t *blocks,
                                            uint32_t b)
{
        return u8_crow(blocks, b) + gas->ngas;
}

static void free_u8_rows(gas_u8_rows_t *R)
{
        if (R != 0) {
                free(R->data);
                memset(R, 0, sizeof(*R));
        }
}

static int init_u8_rows(gas_u8_rows_t *R, int width)
{
        memset(R, 0, sizeof(*R));
        R->width = width;
        return 0;
}

static int reserve_u8_rows(gas_u8_rows_t *R, uint32_t need)
{
        if (need <= R->cap) {
                return 0;
        }

        uint32_t cap = R->cap ? R->cap : 128u;
        while (cap < need) {
                if (cap > UINT32_MAX / 2u) {
                        cap = need;
                        break;
                }
                cap *= 2u;
        }

        size_t bytes = (size_t)cap * (size_t)R->width * sizeof(*R->data);
        uint8_t *p = realloc(R->data, bytes);
        if (p == 0) {
                return -1;
        }

        R->data = p;
        R->cap = cap;
        return 0;
}

static int append_u8_row(gas_u8_rows_t *R, const uint8_t *row)
{
        if (reserve_u8_rows(R, R->n + 1u) != 0) {
                return -1;
        }
        memcpy(u8_row(R, R->n), row, (size_t)R->width);
        R->n++;
        return 0;
}

static void sort_unique_occ_rows(gas_u8_rows_t *R)
{
        if (R->n == 0) {
                return;
        }
        qsort_occ_width = R->width;
        qsort(R->data, R->n, (size_t)R->width, cmp_occ_rows);

        uint32_t w = 1;
        for (uint32_t r = 1; r < R->n; r++) {
                uint8_t *prev = u8_row(R, w - 1u);
                uint8_t *cur  = u8_row(R, r);
                if (!occ_eq_u8(prev, cur, R->width)) {
                        if (w != r) {
                                memcpy(u8_row(R, w), cur, (size_t)R->width);
                        }
                        w++;
                }
        }
        R->n = w;
}

/* ========================================================================== */
/* 3. Public address and lookup routines                                      */
/* ========================================================================== */

uint32_t gas_str2addr_sector(const gas_space_t *gas, gas_sid_t s, uint64_t str)
{
        const uint8_t *occ = sector_occ_ptr(gas, s);
        const uint32_t *stride = sector_stride_ptr(gas, s);
        uint64_t addr = 0;

        for (int g = 0; g < gas->ngas; g++) {
                uint64_t sub = (str >> gas->start[g]) & lowbits(gas->norb[g]);
                uint32_t a = str2addr_local(occ[g], sub);
                addr += (uint64_t)a * stride[g];
        }
        return (uint32_t)addr;
}

uint64_t gas_addr2str_sector(const gas_space_t *gas, gas_sid_t s, uint32_t addr)
{
        const uint8_t *occ = sector_occ_ptr(gas, s);
        const uint32_t *stride = sector_stride_ptr(gas, s);
        uint64_t str = 0;
        uint32_t rem = addr;

        for (int g = 0; g < gas->ngas; g++) {
                uint32_t q = stride[g] == 0 ? 0 : rem / stride[g];
                rem -= q * stride[g];
                str |= addr2str_local(gas->norb[g], occ[g], q) << gas->start[g];
        }
        return str;
}

uint32_t gas_block_ndet(const gas_space_t *gas, gas_bid_t b)
{
        const gas_block_t *blk = gas->block + b;
        return gas->sector_nstr[blk->sa] * gas->sector_nstr[blk->sb];
}

uint32_t gas_det_addr(const gas_space_t *gas, gas_bid_t b,
                      uint32_t ia, uint32_t ib)
{
        const gas_block_t *blk = gas->block + b;
        uint32_t nb = gas->sector_nstr[blk->sb];
        return blk->offset + ia * nb + ib;
}

gas_bid_t gas_find_block(const gas_space_t *gas, gas_sid_t sa, gas_sid_t sb)
{
        if (sa >= gas->nsector) {
                return GAS_INVALID_BID;
        }

        const gas_row_t *row = gas->D.by_alpha_row + sa;
        uint32_t off = row->off;
        uint32_t n = row->n;

        if (n <= 8) {
                for (uint32_t i = 0; i < n; i++) {
                        gas_bid_t b = off + i;
                        if (gas->block[b].sb == sb) {
                                return b;
                        }
                }
                return GAS_INVALID_BID;
        }

        uint32_t lo = 0;
        uint32_t hi = n;
        while (lo < hi) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                gas_sid_t x = gas->block[off + mid].sb;
                if (x < sb) {
                        lo = mid + 1;
                } else {
                        hi = mid;
                }
        }
        if (lo < n && gas->block[off + lo].sb == sb) {
                return off + lo;
        }
        return GAS_INVALID_BID;
}

gas_tid_t gas_find_table(const gas_space_t *gas, gas_sid_t src, gas_sid_t dst)
{
        if (src >= gas->nsector) {
                return GAS_INVALID_TID;
        }

        const gas_row_t *row = gas->T.row + src;
        const gas_sid_t *dsts = gas->T.dst + row->off;
        uint32_t n = row->n;

        if (n <= 8) {
                for (uint32_t i = 0; i < n; i++) {
                        if (dsts[i] == dst) {
                                return row->off + i;
                        }
                }
                return GAS_INVALID_TID;
        }

        uint32_t lo = 0;
        uint32_t hi = n;
        while (lo < hi) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                if (dsts[mid] < dst) {
                        lo = mid + 1;
                } else {
                        hi = mid;
                }
        }
        if (lo < n && dsts[lo] == dst) {
                return row->off + lo;
        }
        return GAS_INVALID_TID;
}

const gas_link_table_t *gas_get_link_table(const gas_space_t *gas,
                                           gas_sid_t src, gas_sid_t dst)
{
        gas_tid_t tid = gas_find_table(gas, src, dst);
        return tid == GAS_INVALID_TID ? 0 : gas->table + tid;
}

/* ========================================================================== */
/* 4. Raw block generation from D                                             */
/* ========================================================================== */

static int build_raw_blocks_from_input(gas_space_t *gas, int nblock,
                                       const int *block_occ,
                                       gas_u8_rows_t *blocks)
{
        if (nblock <= 0 || block_occ == 0) {
                return -1;
        }
        init_u8_rows(blocks, 2 * gas->ngas);
        if (reserve_u8_rows(blocks, (uint32_t)nblock) != 0 ||
            blocks->data == 0) {
                return -1;
        }

        for (int i = 0; i < nblock; i++) {
                const int *a = block_occ + (size_t)(2 * i) * gas->ngas;
                const int *b = a + gas->ngas;
                uint8_t *row = u8_row(blocks, blocks->n);
                int sum_a = 0;
                int sum_b = 0;

                for (int g = 0; g < gas->ngas; g++) {
                        if (a[g] < 0 || a[g] > gas->norb[g] ||
                            b[g] < 0 || b[g] > gas->norb[g]) {
                                return -1;
                        }
                        row[g] = (uint8_t)a[g];
                        row[gas->ngas + g] = (uint8_t)b[g];
                        sum_a += a[g];
                        sum_b += b[g];
                }
                if (sum_a != gas->na || sum_b != gas->nb) {
                        return -1;
                }
                if (blocks->n != 0u &&
                    occ_lex_cmp_width(u8_crow(blocks, blocks->n - 1u), row,
                                      2 * gas->ngas) >= 0) {
                        /* D must be strictly sorted in canonical alpha-major order. */
                        return -1;
                }
                blocks->n++;
        }
        return 0;
}

/* ========================================================================== */
/* 5. Sector collection and initialization                                    */
/* ========================================================================== */

static int append_sector_neighbors_occ(const gas_space_t *gas,
                                       const uint8_t *occ,
                                       gas_u8_rows_t *out,
                                       uint8_t *work)
{
        if (append_u8_row(out, occ) != 0) {       /* self-neighbor always exists */
                return -1;
        }

        for (int r = 0; r < gas->ngas; r++) {
                if (occ[r] == 0) {
                        continue;
                }
                for (int s = 0; s < gas->ngas; s++) {
                        if (r == s || occ[s] >= gas->norb[s]) {
                                continue;
                        }
                        memcpy(work, occ, (size_t)gas->ngas);
                        work[r]--;
                        work[s]++;
                        if (append_u8_row(out, work) != 0) {
                                return -1;
                        }
                }
        }
        return 0;
}

static int collect_sector_pool(gas_space_t *gas, const gas_u8_rows_t *blocks,
                               gas_u8_rows_t *sectors)
{
        init_u8_rows(sectors, gas->ngas);
        uint8_t *work = malloc(sizeof(*work) * (size_t)gas->ngas);
        if (work == 0) {
                return -1;
        }

        for (uint32_t b = 0; b < blocks->n; b++) {
                const uint8_t *a = block_alpha_occ(blocks, b);
                const uint8_t *bb = block_beta_occ(gas, blocks, b);

                if (append_u8_row(sectors, a) != 0 ||
                    append_u8_row(sectors, bb) != 0 ||
                    append_sector_neighbors_occ(gas, a, sectors, work) != 0 ||
                    append_sector_neighbors_occ(gas, bb, sectors, work) != 0) {
                        free(work);
                        return -1;
                }
        }

        free(work);
        sort_unique_occ_rows(sectors);
        if (sectors->n > GAS_INVALID_SID) {
                return -1;
        }
        return 0;
}

static int init_sectors_from_pool(gas_space_t *gas, const gas_u8_rows_t *sectors)
{
        if (sectors->n == 0u || sectors->n > UINT16_MAX) {
                return -1;
        }
        gas->nsector = (uint16_t)sectors->n;
        gas->sector_nstr = malloc(
                sizeof(*gas->sector_nstr) * (size_t)gas->nsector);
        gas->sector_occ = malloc(
                sizeof(*gas->sector_occ) * (size_t)gas->nsector * gas->ngas);
        gas->sector_stride = malloc(
                sizeof(*gas->sector_stride) * (size_t)gas->nsector * gas->ngas);
        if (gas->sector_nstr == 0 || gas->sector_occ == 0 ||
            gas->sector_stride == 0) {
                return -1;
        }

        for (uint32_t s = 0; s < gas->nsector; s++) {
                memcpy(gas->sector_occ + (size_t)s * gas->ngas,
                       u8_crow(sectors, s),
                       (size_t)gas->ngas * sizeof(uint8_t));
        }

        for (uint32_t s = 0; s < gas->nsector; s++) {
                uint8_t *occ = gas->sector_occ + (size_t)s * gas->ngas;
                uint32_t *stride = gas->sector_stride + (size_t)s * gas->ngas;
                uint64_t st = 1;
                for (int g = gas->ngas - 1; g >= 0; g--) {
                        stride[g] = (uint32_t)st;
                        st *= binom(gas->norb[g], occ[g]);
                        if (st > UINT32_MAX) {
                                return -1;
                        }
                }

                gas->sector_nstr[s] = (uint32_t)st;
        }
        return 0;
}

static int find_sector_occ(const gas_space_t *gas, const uint8_t *occ)
{
        uint32_t lo = 0;
        uint32_t hi = gas->nsector;
        while (lo < hi) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                const uint8_t *m = sector_occ_ptr(gas, (gas_sid_t)mid);
                int c = occ_lex_cmp_width(m, occ, gas->ngas);
                if (c < 0) {
                        lo = mid + 1;
                } else {
                        hi = mid;
                }
        }
        if (lo < gas->nsector &&
            occ_eq_u8(sector_occ_ptr(gas, (gas_sid_t)lo), occ, gas->ngas)) {
                return (int)lo;
        }
        return -1;
}

/* ========================================================================== */
/* 6. Legal block set D and two D indices                                     */
/* ========================================================================== */

static int fill_blocks_from_raw(gas_space_t *gas, const gas_u8_rows_t *blocks)
{
        gas->nblock = blocks->n;
        gas->block = malloc(sizeof(*gas->block) * (size_t)gas->nblock);
        if (gas->nblock && gas->block == 0) {
                return -1;
        }

        uint64_t off = 0;
        for (uint32_t b = 0; b < gas->nblock; b++) {
                const uint8_t *a = block_alpha_occ(blocks, b);
                const uint8_t *bb = block_beta_occ(gas, blocks, b);
                int sa = find_sector_occ(gas, a);
                int sb = find_sector_occ(gas, bb);
                if (sa < 0 || sb < 0) {
                        return -1;
                }

                uint64_t nda = gas->sector_nstr[sa];
                uint64_t ndb = gas->sector_nstr[sb];
                uint64_t ndet = nda * ndb;
                if (ndet > UINT32_MAX || off + ndet > UINT32_MAX) {
                        return -1;
                }

                gas->block[b].offset = (uint32_t)off;
                gas->block[b].sa = (gas_sid_t)sa;
                gas->block[b].sb = (gas_sid_t)sb;
                off += ndet;
        }
        gas->ndet = (uint32_t)off;
        return 0;
}

static int build_block_index(gas_space_t *gas)
{
        uint32_t nsec = gas->nsector;
        uint32_t nb = gas->nblock;

        gas->D.by_alpha_row = calloc((size_t)nsec, sizeof(*gas->D.by_alpha_row));
        gas->D.by_beta_row  = calloc((size_t)nsec, sizeof(*gas->D.by_beta_row));
        gas->D.by_beta_sid  = malloc(sizeof(*gas->D.by_beta_sid) * (size_t)nb);
        gas->D.by_beta_bid  = malloc(sizeof(*gas->D.by_beta_bid) * (size_t)nb);
        if ((nsec && (gas->D.by_alpha_row == 0 || gas->D.by_beta_row == 0)) ||
            (nb && (gas->D.by_beta_sid == 0 || gas->D.by_beta_bid == 0))) {
                return -1;
        }

        for (uint32_t b = 0; b < nb; b++) {
                gas->D.by_alpha_row[gas->block[b].sa].n++;
                gas->D.by_beta_row [gas->block[b].sb].n++;
        }

        uint32_t off = 0;
        for (uint32_t s = 0; s < nsec; s++) {
                uint32_t n = gas->D.by_alpha_row[s].n;
                gas->D.by_alpha_row[s].off = off;
                off += n;
                /* Keep n unchanged.  Since block[] is canonical alpha-major,
                 * this row points directly into block[off : off+n). */
        }

        off = 0;
        for (uint32_t s = 0; s < nsec; s++) {
                uint32_t n = gas->D.by_beta_row[s].n;
                gas->D.by_beta_row[s].off = off;
                off += n;
                gas->D.by_beta_row[s].n = 0;
        }

        /* by_beta is the only materialized secondary sparse view.  by_alpha is
         * implicit because canonical block order makes alpha rows contiguous. */
        for (uint32_t b = 0; b < nb; b++) {
                gas_sid_t sb = gas->block[b].sb;
                gas_row_t *rb = gas->D.by_beta_row + sb;
                uint32_t pb = rb->off + rb->n;

                gas->D.by_beta_sid[pb] = gas->block[b].sa;
                gas->D.by_beta_bid[pb] = b;
                rb->n++;
        }
        return 0;
}

/* ========================================================================== */
/* 7. Temporary sector-neighbor rows                                          */
/* ========================================================================== */

typedef struct {
        gas_row_t *row;                /* [nsector] */
        gas_sid_t *sid;                /* packed generated neighbors */
        uint32_t n;
        uint32_t cap;
} gas_sector_neighbors_tmp_t;

static void free_sector_neighbors_tmp(gas_sector_neighbors_tmp_t *S)
{
        if (S != 0) {
                free(S->row);
                free(S->sid);
                memset(S, 0, sizeof(*S));
        }
}

static int reserve_sid_tmp(gas_sector_neighbors_tmp_t *S, uint32_t need)
{
        if (need <= S->cap) {
                return 0;
        }
        uint32_t cap = S->cap ? S->cap : 1024u;
        while (cap < need) {
                if (cap > UINT32_MAX / 2u) {
                        cap = need;
                        break;
                }
                cap *= 2u;
        }
        gas_sid_t *p = realloc(S->sid, (size_t)cap * sizeof(gas_sid_t));
        if (p == 0) {
                return -1;
        }
        S->sid = p;
        S->cap = cap;
        return 0;
}

static int append_sector_neighbor_sids(const gas_space_t *gas,
                                      gas_sid_t src,
                                      gas_sector_neighbors_tmp_t *S,
                                      gas_u8_rows_t *nbr_occ,
                                      uint8_t *work)
{
        nbr_occ->n = 0;
        S->row[src].off = S->n;

        if (append_sector_neighbors_occ(gas, sector_occ_ptr(gas, src),
                                        nbr_occ, work) != 0) {
                return -1;
        }
        sort_unique_occ_rows(nbr_occ);

        for (uint32_t k = 0; k < nbr_occ->n; k++) {
                int sid = find_sector_occ(gas, u8_crow(nbr_occ, k));
                if (sid < 0) {
                        continue;
                }
                if (reserve_sid_tmp(S, S->n + 1u) != 0) {
                        return -1;
                }
                S->sid[S->n++] = (gas_sid_t)sid;
                S->row[src].n++;
        }
        return 0;
}

static int build_sector_neighbors_tmp(gas_space_t *gas,
                                      gas_sector_neighbors_tmp_t *S)
{
        uint32_t nsec = gas->nsector;
        uint8_t *is_legal_src = 0;
        uint8_t *work = 0;
        gas_u8_rows_t nbr_occ;
        int ret = -1;

        memset(S, 0, sizeof(*S));
        memset(&nbr_occ, 0, sizeof(nbr_occ));

        S->row = calloc((size_t)nsec, sizeof(*S->row));
        is_legal_src = calloc((size_t)nsec, sizeof(*is_legal_src));
        work = malloc(sizeof(*work) * (size_t)gas->ngas);
        init_u8_rows(&nbr_occ, gas->ngas);
        if ((nsec && (S->row == 0 || is_legal_src == 0)) || work == 0) {
                goto done;
        }

        for (uint32_t b = 0; b < gas->nblock; b++) {
                is_legal_src[gas->block[b].sa] = 1;
                is_legal_src[gas->block[b].sb] = 1;
        }

        for (uint32_t s = 0; s < nsec; s++) {
                S->row[s].off = S->n;
                if (is_legal_src[s] &&
                    append_sector_neighbor_sids(gas, (gas_sid_t)s, S,
                                                &nbr_occ, work) != 0) {
                        goto done;
                }
        }

        ret = 0;

done:
        free(is_legal_src);
        free(work);
        free_u8_rows(&nbr_occ);
        return ret;
}

/* ========================================================================== */
/* 8. Required table pair marker and table index                              */
/* ========================================================================== */

typedef struct {
        uint32_t *pair;                /* packed (src,dst) keys */
        uint32_t n;
        uint32_t cap;

        uint32_t *slot;                /* open-addressing set; stores pair_key + 1 */
        uint32_t nslot;
        uint32_t used;
} gas_pair_marker_t;

static inline uint32_t pair_key(gas_sid_t src, gas_sid_t dst)
{
        return ((uint32_t)src << 16) | (uint32_t)dst;
}

static inline gas_sid_t key_src(uint32_t key)
{
        return (gas_sid_t)(key >> 16);
}

static inline gas_sid_t key_dst(uint32_t key)
{
        return (gas_sid_t)(key & 0xffffu);
}

static uint32_t pair_key_hash(uint32_t x)
{
        x ^= x >> 16;
        x *= UINT32_C(0x7feb352d);
        x ^= x >> 15;
        x *= UINT32_C(0x846ca68b);
        x ^= x >> 16;
        return x;
}

static void free_pair_marker(gas_pair_marker_t *M)
{
        if (M != 0) {
                free(M->pair);
                free(M->slot);
                memset(M, 0, sizeof(*M));
        }
}

static int reserve_marker_pairs(gas_pair_marker_t *M, uint32_t need)
{
        if (need <= M->cap) {
                return 0;
        }
        uint32_t cap = M->cap ? M->cap : 1024u;
        while (cap < need) {
                if (cap > UINT32_MAX / 2u) {
                        cap = need;
                        break;
                }
                cap *= 2u;
        }
        uint32_t *p = realloc(M->pair, (size_t)cap * sizeof(uint32_t));
        if (p == 0) {
                return -1;
        }
        M->pair = p;
        M->cap = cap;
        return 0;
}

static int marker_rehash(gas_pair_marker_t *M, uint32_t new_nslot)
{
        uint32_t *old = M->slot;
        uint32_t old_n = M->nslot;

        M->slot = calloc((size_t)new_nslot, sizeof(uint32_t));
        if (M->slot == 0) {
                M->slot = old;
                return -1;
        }
        M->nslot = new_nslot;
        M->used = 0;

        for (uint32_t i = 0; i < old_n; i++) {
                uint32_t stored = old[i];
                if (stored == 0) {
                        continue;
                }
                uint32_t mask = new_nslot - 1u;
                uint32_t h = pair_key_hash(stored) & mask;
                while (M->slot[h] != 0) {
                        h = (h + 1u) & mask;
                }
                M->slot[h] = stored;
                M->used++;
        }
        free(old);
        return 0;
}

static int init_pair_marker(gas_pair_marker_t *M)
{
        memset(M, 0, sizeof(*M));
        return marker_rehash(M, 2048u);
}

static int mark_pair(gas_pair_marker_t *M, gas_sid_t src, gas_sid_t dst)
{
        if ((uint64_t)(M->used + 1u) * 3u >= (uint64_t)M->nslot * 2u) {
                if (M->nslot > UINT32_MAX / 2u) {
                        return -1;
                }
                if (marker_rehash(M, M->nslot * 2u) != 0) {
                        return -1;
                }
        }

        uint32_t key = pair_key(src, dst);
        uint32_t stored = key + 1u;
        uint32_t mask = M->nslot - 1u;
        uint32_t h = pair_key_hash(stored) & mask;

        while (M->slot[h] != 0) {
                if (M->slot[h] == stored) {
                        return 0;
                }
                h = (h + 1u) & mask;
        }

        if (reserve_marker_pairs(M, M->n + 1u) != 0) {
                return -1;
        }
        M->slot[h] = stored;
        M->used++;
        M->pair[M->n++] = key;
        return 0;
}

static int prescan_required_tables(const gas_space_t *gas,
                                   const gas_sector_neighbors_tmp_t *S,
                                   gas_pair_marker_t *M)
{
        if (init_pair_marker(M) != 0) {
                return -1;
        }

        /* Required table set is the symmetric closure of legal-block one-step
         * sector edges.  If legal block (A,B) has a one-step neighbor X, the
         * H*c contract may need both A -> X and X -> A: the forward direction
         * is used for one-step/first-step contributions, and the reverse
         * direction covers AA/BB second steps through illegal intermediates. */
        for (uint32_t b = 0; b < gas->nblock; b++) {
                gas_sid_t A = gas->block[b].sa;
                gas_sid_t B = gas->block[b].sb;
                gas_row_t ar = S->row[A];
                gas_row_t br = S->row[B];

                for (uint32_t i = 0; i < ar.n; i++) {
                        gas_sid_t X = S->sid[ar.off + i];
                        if (mark_pair(M, A, X) != 0 ||
                            mark_pair(M, X, A) != 0) {
                                return -1;
                        }
                }
                for (uint32_t i = 0; i < br.n; i++) {
                        gas_sid_t Y = S->sid[br.off + i];
                        if (mark_pair(M, B, Y) != 0 ||
                            mark_pair(M, Y, B) != 0) {
                                return -1;
                        }
                }
        }
        return 0;
}

static int build_table_index(gas_space_t *gas, gas_pair_marker_t *M)
{
        if (M->n > 1u) {
                qsort(M->pair, M->n, sizeof(uint32_t), cmp_u32);
        }
        gas->ntable = M->n;

        uint32_t nsec = gas->nsector;
        gas->T.row = calloc((size_t)nsec, sizeof(*gas->T.row));
        gas->R.row = calloc((size_t)nsec, sizeof(*gas->R.row));
        gas->T.dst = malloc(sizeof(*gas->T.dst) * (size_t)gas->ntable);
        gas->R.src = malloc(sizeof(*gas->R.src) * (size_t)gas->ntable);
        gas->R.tid = malloc(sizeof(*gas->R.tid) * (size_t)gas->ntable);
        gas->table = calloc((size_t)gas->ntable, sizeof(*gas->table));
        if ((nsec && (gas->T.row == 0 || gas->R.row == 0)) ||
            (gas->ntable && (gas->T.dst == 0 || gas->R.src == 0 ||
                             gas->R.tid == 0 || gas->table == 0))) {
                return -1;
        }

        for (uint32_t i = 0; i < gas->ntable; i++) {
                gas_sid_t src = key_src(M->pair[i]);
                gas_sid_t dst = key_dst(M->pair[i]);
                gas->T.row[src].n++;
                gas->R.row[dst].n++;
        }

        uint32_t off = 0;
        for (uint32_t s = 0; s < nsec; s++) {
                uint32_t n = gas->T.row[s].n;
                gas->T.row[s].off = off;
                off += n;
                gas->T.row[s].n = 0;
        }

        off = 0;
        for (uint32_t s = 0; s < nsec; s++) {
                uint32_t n = gas->R.row[s].n;
                gas->R.row[s].off = off;
                off += n;
                gas->R.row[s].n = 0;
        }

        for (uint32_t tid = 0; tid < gas->ntable; tid++) {
                gas_sid_t src = key_src(M->pair[tid]);
                gas_sid_t dst = key_dst(M->pair[tid]);

                gas_row_t *trow = gas->T.row + src;
                uint32_t tpos = trow->off + trow->n;
                gas->T.dst[tpos] = dst;
                trow->n++;

                gas_row_t *rrow = gas->R.row + dst;
                uint32_t rpos = rrow->off + rrow->n;
                gas->R.src[rpos] = src;
                gas->R.tid[rpos] = tid;
                rrow->n++;
        }
        return 0;
}

/* ========================================================================== */
/* 9. Sector-pair spin-string link tables                                     */
/* ========================================================================== */

static int sector_transfer(const gas_space_t *gas, gas_sid_t src, gas_sid_t dst,
                           int *r, int *s)
{
        const uint8_t *a = sector_occ_ptr(gas, src);
        const uint8_t *b = sector_occ_ptr(gas, dst);
        *r = -1;
        *s = -1;

        if (src == dst) {
                return 1;
        }

        int nr = 0;
        int ns = 0;
        for (int g = 0; g < gas->ngas; g++) {
                int d = (int)b[g] - (int)a[g];
                if (d == -1) {
                        *r = g;        /* source loses one electron in r */
                        nr++;
                } else if (d == 1) {
                        *s = g;        /* destination gains one electron in s */
                        ns++;
                } else if (d != 0) {
                        return 0;
                }
        }
        return nr == 1 && ns == 1;
}

static int nlink_pair(const gas_space_t *gas, gas_sid_t src, gas_sid_t dst,
                      uint32_t *out)
{
        const uint8_t *occ = sector_occ_ptr(gas, src);
        int r = -1;
        int s = -1;
        uint64_t n = 0;

        *out = 0;
        if (!sector_transfer(gas, src, dst, &r, &s)) {
                return -1;
        }

        if (src == dst) {
                for (int g = 0; g < gas->ngas; g++) {
                        uint64_t nocc = occ[g];
                        uint64_t nvir = (uint64_t)gas->norb[g] - occ[g];
                        n += nocc * (nvir + 1u);       /* p=q plus p!=q inside subspace */
                }
        } else {
                n = (uint64_t)occ[r] * ((uint64_t)gas->norb[s] - occ[s]);
        }
        if (n > UINT32_MAX) {
                return -1;
        }
        *out = (uint32_t)n;
        return 0;
}

static inline void set_link(gas_link_entry_t *e, uint32_t addr,
                            int p, int q, int8_t sign)
{
        e->addr = addr;
        gas_link_set_raw(e, p, q);
        e->sign = sign;
        e->padding = 0;
}

static int init_link_table(gas_space_t *gas, gas_tid_t tid,
                           gas_sid_t src, gas_sid_t dst,
                           uint64_t *work)
{
        gas_link_table_t *table = gas->table + tid;
        uint32_t nsrc = gas->sector_nstr[src];
        uint32_t nlink = 0;
        int remove = 0;
        int add = 0;

        if (nlink_pair(gas, src, dst, &nlink) != 0 ||
            !sector_transfer(gas, src, dst, &remove, &add)) {
                return -1;
        }
        if (src != dst && (remove >= gas->ngas || add >= gas->ngas)) {
                return -1;
        }

        table->link = 0;
        table->nsrc = nsrc;
        table->nlink = nlink;
        if (nsrc == 0 || nlink == 0) {
                return 0;
        }
        if (nsrc > UINT32_MAX / nlink) {
                return -1;
        }

        size_t nentry = (size_t)nsrc * nlink;
        table->link = malloc(nentry * sizeof(*table->link));
        if (table->link == 0) {
                return -1;
        }
        if (UINT64_MAX - *work < nentry) {
                return -1;
        }
        *work += nentry;
        return 0;
}

static int fill_link_table(const gas_space_t *gas, gas_tid_t tid,
                           gas_sid_t src, gas_sid_t dst)
{
        gas_link_table_t *table = gas->table + tid;
        uint32_t nsrc = table->nsrc;
        uint32_t nlink = table->nlink;
        int r = 0;
        int ss = 0;

        if (!sector_transfer(gas, src, dst, &r, &ss)) {
                return -1;
        }
        if (nsrc == 0 || nlink == 0) {
                return 0;
        }
        const int ngas = gas->ngas;
        const uint8_t *src_occ = sector_occ_ptr(gas, src);
        const uint8_t *dst_occ = sector_occ_ptr(gas, dst);
        const uint32_t *src_stride = sector_stride_ptr(gas, src);
        const uint32_t *dst_stride = sector_stride_ptr(gas, dst);

        for (uint32_t ia = 0; ia < nsrc; ia++) {
                uint32_t subaddr[GAS_MAX_NGAS];
                uint32_t rem = ia;
                uint64_t str = 0;
                gas_link_entry_t *row = table->link + (size_t)ia * nlink;
                uint32_t k = 0;

                for (int g = 0; g < ngas; g++) {
                        uint32_t q = rem / src_stride[g];
                        uint64_t sub;
                        rem -= q * src_stride[g];
                        subaddr[g] = q;
                        sub = addr2str_local(gas->norb[g], src_occ[g], q);
                        str |= sub << gas->start[g];
                }

                if (src == dst) {
                        for (int g = 0; g < ngas; g++) {
                                int beg = gas->start[g];
                                uint32_t stride = src_stride[g];
                                uint64_t sub = (str >> beg) & lowbits(gas->norb[g]);
                                uint64_t occbits = sub;
                                uint64_t vir0 = (~sub) & lowbits(gas->norb[g]);

                                while (occbits) {
                                        int qloc = ctz64(occbits);
                                        int q = beg + qloc;
                                        occbits &= occbits - 1;

                                        set_link(row + k, ia, q, q, 1);
                                        k++;

                                        uint64_t virbits = vir0;
                                        while (virbits) {
                                                int ploc = ctz64(virbits);
                                                int p = beg + ploc;
                                                uint64_t sub1;
                                                uint32_t olda;
                                                uint32_t newa;
                                                uint32_t addr1;

                                                virbits &= virbits - 1;
                                                sub1 = (sub ^ (1ULL << qloc)) |
                                                       (1ULL << ploc);
                                                olda = subaddr[g];
                                                newa = str2addr_local(src_occ[g], sub1);
                                                addr1 = ia;
                                                if (newa >= olda) {
                                                        addr1 += (newa - olda) * stride;
                                                } else {
                                                        addr1 -= (olda - newa) * stride;
                                                }

                                                set_link(row + k, addr1, p, q,
                                                         cd_sign(p, q, str));
                                                k++;
                                        }
                                }
                        }
                } else {
                        int begr = gas->start[r];
                        int begs = gas->start[ss];
                        uint64_t subr = (str >> begr) & lowbits(gas->norb[r]);
                        uint64_t subs = (str >> begs) & lowbits(gas->norb[ss]);
                        uint64_t occbits = subr;
                        uint64_t vir0 = (~subs) & lowbits(gas->norb[ss]);
                        uint32_t base_dst = 0;

                        for (int g = 0; g < ngas; g++) {
                                if (g != r && g != ss) {
                                        base_dst += subaddr[g] * dst_stride[g];
                                }
                        }

                        while (occbits) {
                                int qloc = ctz64(occbits);
                                int q = begr + qloc;
                                uint64_t subr1 = subr ^ (1ULL << qloc);
                                uint64_t virbits = vir0;
                                uint32_t ar = str2addr_local(dst_occ[r], subr1);
                                occbits &= occbits - 1;

                                while (virbits) {
                                        int ploc = ctz64(virbits);
                                        int p = begs + ploc;
                                        uint64_t subs1;
                                        uint32_t as;
                                        uint32_t addr1;

                                        virbits &= virbits - 1;
                                        subs1 = subs | (1ULL << ploc);
                                        as = str2addr_local(dst_occ[ss], subs1);
                                        addr1 = base_dst + ar * dst_stride[r]
                                                + as * dst_stride[ss];
                                        set_link(row + k, addr1, p, q,
                                                 cd_sign(p, q, str));
                                        k++;
                                }
                        }
                }
        }
        return 0;
}

static int build_all_link_tables(gas_space_t *gas)
{
        uint64_t work = 0;

        /* Allocate serially, then fill all tables in one OpenMP region. */

        for (gas_sid_t src = 0; src < gas->nsector; src++) {
                gas_row_t row = gas->T.row[src];
                for (uint32_t i = 0; i < row.n; i++) {
                        gas_tid_t tid = row.off + i;
                        gas_sid_t dst = gas->T.dst[tid];
                        if (init_link_table(gas, tid, src, dst, &work) != 0) {
                                return -1;
                        }
                }
        }

        int failed = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) reduction(|:failed) \
        if(work >= LINK_BUILD_PARALLEL_MIN)
#endif
        for (int64_t ss = 0; ss < (int64_t)gas->nsector; ss++) {
                gas_sid_t src = (gas_sid_t)ss;
                gas_row_t row = gas->T.row[src];

                for (uint32_t i = 0; i < row.n; i++) {
                        gas_tid_t tid = row.off + i;
                        gas_sid_t dst = gas->T.dst[tid];
                        if (fill_link_table(gas, tid, src, dst) != 0) {
                                failed = 1;
                        }
                }
        }
        return failed ? -1 : 0;
}

static int build_table_active_ops_raw(gas_space_t *gas)
{
        const uint32_t nnorb = (uint32_t)(gas->norb_tot * (gas->norb_tot + 1) / 2);
        if (nnorb == 0u) {
                return -1;
        }
        uint8_t *seen = malloc((size_t)nnorb * sizeof(*seen));
        if (seen == 0) {
                return -1;
        }

        for (uint32_t t = 0; t < gas->ntable; t++) {
                gas_link_table_t *tab = gas->table + t;
                size_t nentry = (size_t)tab->nsrc * tab->nlink;
                uint32_t nop = 0;

                memset(seen, 0, (size_t)nnorb * sizeof(*seen));
                for (size_t i = 0; i < nentry; i++) {
                        const gas_link_entry_t *e = tab->link + i;
                        uint16_t op = gas_link_pair_index(gas_link_cre(e), gas_link_des(e));
                        if (!seen[op]) {
                                seen[op] = 1u;
                                nop++;
                        }
                }
                if (nop > UINT16_MAX) {
                        free(seen);
                        return -1;
                }

                free(tab->active_op);
                tab->active_op = 0;
                tab->nop = (uint16_t)nop;
                if (nop != 0u) {
                        tab->active_op = malloc((size_t)nop * sizeof(*tab->active_op));
                        if (tab->active_op == 0) {
                                free(seen);
                                return -1;
                        }
                        uint32_t k = 0;
                        for (uint32_t op = 0; op < nnorb; op++) {
                                if (seen[op]) {
                                        tab->active_op[k++] = (uint16_t)op;
                                }
                        }
                }
        }
        free(seen);
        return 0;
}

int gas_space_compress_links(gas_space_t *gas)
{
        if (gas == 0) {
                return -1;
        }
        if (gas->link_format == GAS_LINK_COMPRESSED) {
                return 0;
        }

        /* Active operator sets are table properties.  Build them once from
         * the raw (p,q) fields before replacing those fields by pair indices. */
        if (build_table_active_ops_raw(gas) != 0) {
                return -1;
        }

        for (uint32_t t = 0; t < gas->ntable; t++) {
                gas_link_table_t *tab = gas->table + t;
                for (size_t i = 0, n = (size_t)tab->nsrc * tab->nlink; i < n; i++) {
                        gas_link_entry_t *e = tab->link + i;
                        gas_link_set_ia(e, gas_link_pair_index(gas_link_cre(e), gas_link_des(e)));
                }
        }
        gas->link_format = GAS_LINK_COMPRESSED;
        return 0;
}

int gas_space_links_are_compressed(const gas_space_t *gas)
{
        return gas != 0 && gas->link_format == GAS_LINK_COMPRESSED;
}

/* ========================================================================== */
/* 10. Memory reporting                                                       */
/* ========================================================================== */

void gas_memory_report(const gas_space_t *gas, gas_memory_report_t *r)
{
        memset(r, 0, sizeof(*r));

        r->metadata = sizeof(*gas)
                    + (uint64_t)gas->ngas * sizeof(*gas->norb)
                    + (uint64_t)gas->ngas * sizeof(*gas->start);
        r->sector = (uint64_t)gas->nsector * sizeof(*gas->sector_nstr)
                  + (uint64_t)gas->nsector * gas->ngas * sizeof(*gas->sector_occ)
                  + (uint64_t)gas->nsector * gas->ngas * sizeof(*gas->sector_stride);
        r->block = (uint64_t)gas->nblock * sizeof(*gas->block);
        r->block_index = (uint64_t)gas->nsector * sizeof(*gas->D.by_alpha_row)
                       + (uint64_t)gas->nsector * sizeof(*gas->D.by_beta_row)
                       + (uint64_t)gas->nblock  * sizeof(*gas->D.by_beta_sid)
                       + (uint64_t)gas->nblock  * sizeof(*gas->D.by_beta_bid);
        r->link_table_index = (uint64_t)gas->nsector * sizeof(*gas->T.row)
                            + (uint64_t)gas->ntable  * sizeof(*gas->T.dst)
                            + (uint64_t)gas->nsector * sizeof(*gas->R.row)
                            + (uint64_t)gas->ntable  * sizeof(*gas->R.src)
                            + (uint64_t)gas->ntable  * sizeof(*gas->R.tid);
        r->link_table = (uint64_t)gas->ntable * sizeof(*gas->table);
        for (uint32_t i = 0; i < gas->ntable; i++) {
                r->link_table += (uint64_t)gas->table[i].nsrc * gas->table[i].nlink
                               * sizeof(*gas->table[i].link);
                r->link_table += (uint64_t)gas->table[i].nop
                               * sizeof(*gas->table[i].active_op);
        }
        r->total = r->metadata
                 + r->sector
                 + r->block
                 + r->block_index
                 + r->link_table_index
                 + r->link_table;
}

uint64_t gas_memory_bytes(const gas_space_t *gas)
{
        gas_memory_report_t r;
        gas_memory_report(gas, &r);
        return r.total;
}

/* ========================================================================== */
/* 11. Constructors and destructor                                            */
/* ========================================================================== */

static int init_common(gas_space_t *gas, int ngas,
                       const int *norb, int na, int nb)
{
        int off = 0;
        if (gas == 0) {
                return -1;
        }
        gas_zero(gas);

        if (ngas <= 0 || ngas > GAS_MAX_NGAS || norb == 0 || na < 0 || nb < 0) {
                return -1;
        }

        gas->ngas = ngas;
        gas->na = na;
        gas->nb = nb;
        gas->norb = malloc(sizeof(*gas->norb) * (size_t)ngas);
        gas->start = malloc(sizeof(*gas->start) * (size_t)ngas);
        if (gas->norb == 0 || gas->start == 0) {
                return -1;
        }

        for (int g = 0; g < ngas; g++) {
                if (norb[g] <= 0 || norb[g] > GAS_MAX_LOCAL_ORB) {
                        return -1;
                }
                if (off + norb[g] > GAS_MAX_ORB) {
                        return -1;
                }
                gas->norb[g] = norb[g];
                gas->start[g] = off;
                off += norb[g];
        }
        gas->norb_tot = off;
        return 0;
}

static int finish_build_from_raw(gas_space_t *gas, const gas_u8_rows_t *blocks)
{
        gas_u8_rows_t sectors;
        gas_sector_neighbors_tmp_t S;
        gas_pair_marker_t M;
        int ret = -1;

        memset(&sectors, 0, sizeof(sectors));
        memset(&S, 0, sizeof(S));
        memset(&M, 0, sizeof(M));

        if (collect_sector_pool(gas, blocks, &sectors) != 0) {
                goto done;
        }
        if (init_sectors_from_pool(gas, &sectors) != 0) {
                goto done;
        }
        if (fill_blocks_from_raw(gas, blocks) != 0) {
                goto done;
        }
        if (build_block_index(gas) != 0) {
                goto done;
        }
        if (build_sector_neighbors_tmp(gas, &S) != 0) {
                goto done;
        }
        if (prescan_required_tables(gas, &S, &M) != 0) {
                goto done;
        }
        if (build_table_index(gas, &M) != 0) {
                goto done;
        }
        free_pair_marker(&M);

        if (build_all_link_tables(gas) != 0) {
                goto done;
        }
        gas->link_format = GAS_LINK_RAW;

        ret = 0;

done:
        free_u8_rows(&sectors);
        free_sector_neighbors_tmp(&S);
        free_pair_marker(&M);
        return ret;
}

int gas_space_from_blocks(gas_space_t *gas, int ngas, const int *norb,
                          int na, int nb,
                          int nblock, const int *block_occ)
{
        gas_u8_rows_t blocks;
        memset(&blocks, 0, sizeof(blocks));

        if (init_common(gas, ngas, norb, na, nb) != 0) {
                gas_space_free(gas);
                return -1;
        }

        if (build_raw_blocks_from_input(gas, nblock, block_occ, &blocks) != 0 ||
            finish_build_from_raw(gas, &blocks) != 0) {
                free_u8_rows(&blocks);
                gas_space_free(gas);
                return -1;
        }

        free_u8_rows(&blocks);
        return 0;
}

void gas_space_free(gas_space_t *gas)
{
        if (gas == 0) {
                return;
        }

        free(gas->norb);
        free(gas->start);
        free(gas->sector_nstr);
        free(gas->sector_occ);
        free(gas->sector_stride);

        free(gas->block);
        free(gas->D.by_alpha_row);
        free(gas->D.by_beta_row);
        free(gas->D.by_beta_sid);
        free(gas->D.by_beta_bid);

        free(gas->T.row);
        free(gas->T.dst);
        free(gas->R.row);
        free(gas->R.src);
        free(gas->R.tid);

        if (gas->table != 0) {
                for (uint32_t i = 0; i < gas->ntable; i++) {
                        free(gas->table[i].link);
                        free(gas->table[i].active_op);
                }
                free(gas->table);
        }

        gas_zero(gas);
}
