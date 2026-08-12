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

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
        RDM_TASK_FACTOR = 16,
        RDM_MAX_ALPHA_SPLIT = 64,
        RDM_MIN_ALPHA_TILE = 4,
        RDM_TRANSPOSE_TILE = 32,
        RDM_BB_TRANSPOSE_MIN_ALPHA = 4,
        RDM_AA_GROUP_SIZE = 8
};

static const size_t RDM_WORKSPACE_MAX_ELEMS = 67108864u;
static size_t rdm_abba_t1_target_bytes = 16u * 1024u * 1024u;

/*
 * BB kernels store consecutive link-table IDs.  AB kernels reuse the buffer
 * for (destination block ID, beta link-table ID).
 */
typedef struct {
        gas_tid_t first;
        gas_tid_t second;
} rdm_tid_pair_t;

/*
 * Alpha link landing in the current destination tile [a0,a1).
 * rel_addr is relative to a0; op_index selects the plan's active operator.
 */
typedef struct {
        uint32_t src_row;
        uint32_t rel_addr;
        uint16_t op_index;
        int8_t sign;
        uint8_t padding;
} rdm_alpha_hit_t;

typedef struct {
        uint32_t off;
        uint16_t n;
        uint16_t padding;
} rdm_table_ops_t;

typedef struct {
        double *data;
        size_t capacity;
        double *arena;
        size_t arena_capacity;
        rdm_tid_pair_t *pair;
        size_t pair_capacity;
        rdm_alpha_hit_t *alpha_hit;
        size_t alpha_hit_capacity;
        uint32_t alpha_hit_n;
        int32_t *op_map;
        size_t op_map_capacity;
} rdm_workspace_t;

typedef struct {
        uint64_t cost;
        gas_sid_t adst;
        uint32_t q0;
        uint32_t q1;
        uint32_t a0;
        uint32_t a1;
} rdm_task_t;

struct gas_rdm_plan {
        const gas_space_t *gas;
        rdm_workspace_t *workspace;
        uint32_t nworkspace;
        rdm_task_t *task;
        uint32_t ntask;
        uint32_t task_threads;
        rdm_table_ops_t *table_ops;
        uint16_t *active_op;
        uint32_t *active_row;
        size_t abba_t1_target_bytes;
};

static int rdm_arena_reserve(rdm_workspace_t *workspace, size_t need);
static int rdm_pair_reserve(rdm_workspace_t *workspace, size_t need);
static int rdm_hit_reserve(rdm_workspace_t *workspace, size_t need);
static int rdm_op_map_reserve(rdm_workspace_t *workspace, size_t need);

void fci_rdm_gas_set_abba_t1_bytes(uint64_t bytes)
{
        rdm_abba_t1_target_bytes = (size_t)bytes;
}

static inline uint32_t rdm_op(const gas_link_entry_t *e, uint32_t norb)
{
        return (uint32_t)gas_link_cre(e) * norb + gas_link_des(e);
}

static inline size_t rdm2_index(uint32_t p, uint32_t q,
                                uint32_t r, uint32_t s, uint32_t norb)
{
        return (((size_t)p * norb + q) * norb + r) * norb + s;
}

static inline double rdm_dot(const double *x, const double *y, uint32_t n)
{
        double value = 0.0;
        for (uint32_t i = 0; i < n; i++) {
                value += x[i] * y[i];
        }
        return value;
}

static inline void rdm_dot4(const double *restrict x,
                            const double *restrict y0,
                            const double *restrict y1,
                            const double *restrict y2,
                            const double *restrict y3,
                            uint32_t n, double value[4])
{
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        uint32_t i = 0;
#if defined(__AVX2__)
        __m256d v0 = _mm256_setzero_pd();
        __m256d v1 = _mm256_setzero_pd();
        __m256d v2 = _mm256_setzero_pd();
        __m256d v3 = _mm256_setzero_pd();
        for (; i + 4u <= n; i += 4u) {
                __m256d vx = _mm256_loadu_pd(x + i);
                v0 = _mm256_add_pd(v0, _mm256_mul_pd(vx, _mm256_loadu_pd(y0 + i)));
                v1 = _mm256_add_pd(v1, _mm256_mul_pd(vx, _mm256_loadu_pd(y1 + i)));
                v2 = _mm256_add_pd(v2, _mm256_mul_pd(vx, _mm256_loadu_pd(y2 + i)));
                v3 = _mm256_add_pd(v3, _mm256_mul_pd(vx, _mm256_loadu_pd(y3 + i)));
        }
        double lane[4];
        _mm256_storeu_pd(lane, v0);
        s0 = lane[0] + lane[1] + lane[2] + lane[3];
        _mm256_storeu_pd(lane, v1);
        s1 = lane[0] + lane[1] + lane[2] + lane[3];
        _mm256_storeu_pd(lane, v2);
        s2 = lane[0] + lane[1] + lane[2] + lane[3];
        _mm256_storeu_pd(lane, v3);
        s3 = lane[0] + lane[1] + lane[2] + lane[3];
#endif
        for (; i < n; i++) {
                double xi = x[i];
                s0 += xi * y0[i];
                s1 += xi * y1[i];
                s2 += xi * y2[i];
                s3 += xi * y3[i];
        }
        value[0] = s0;
        value[1] = s1;
        value[2] = s2;
        value[3] = s3;
}

static inline void rdm_dot2(const double *restrict x,
                            const double *restrict y0,
                            const double *restrict y1,
                            uint32_t n, double value[2])
{
        double s0 = 0.0, s1 = 0.0;
        uint32_t i = 0u;
#if defined(__AVX2__)
        __m256d v0 = _mm256_setzero_pd();
        __m256d v1 = _mm256_setzero_pd();
        for (; i + 4u <= n; i += 4u) {
                __m256d vx = _mm256_loadu_pd(x + i);
                v0 = _mm256_add_pd(v0, _mm256_mul_pd(
                        vx, _mm256_loadu_pd(y0 + i)));
                v1 = _mm256_add_pd(v1, _mm256_mul_pd(
                        vx, _mm256_loadu_pd(y1 + i)));
        }
        double lane[4];
        _mm256_storeu_pd(lane, v0);
        s0 = lane[0] + lane[1] + lane[2] + lane[3];
        _mm256_storeu_pd(lane, v1);
        s1 = lane[0] + lane[1] + lane[2] + lane[3];
#endif
        for (; i < n; i++) {
                double xi = x[i];
                s0 += xi * y0[i];
                s1 += xi * y1[i];
        }
        value[0] = s0;
        value[1] = s1;
}

static inline void rdm_dot3(const double *restrict x,
                            const double *restrict y0,
                            const double *restrict y1,
                            const double *restrict y2,
                            uint32_t n, double value[3])
{
        double s0 = 0.0, s1 = 0.0, s2 = 0.0;
        uint32_t i = 0u;
#if defined(__AVX2__)
        __m256d v0 = _mm256_setzero_pd();
        __m256d v1 = _mm256_setzero_pd();
        __m256d v2 = _mm256_setzero_pd();
        for (; i + 4u <= n; i += 4u) {
                __m256d vx = _mm256_loadu_pd(x + i);
                v0 = _mm256_add_pd(v0, _mm256_mul_pd(
                        vx, _mm256_loadu_pd(y0 + i)));
                v1 = _mm256_add_pd(v1, _mm256_mul_pd(
                        vx, _mm256_loadu_pd(y1 + i)));
                v2 = _mm256_add_pd(v2, _mm256_mul_pd(
                        vx, _mm256_loadu_pd(y2 + i)));
        }
        double lane[4];
        _mm256_storeu_pd(lane, v0);
        s0 = lane[0] + lane[1] + lane[2] + lane[3];
        _mm256_storeu_pd(lane, v1);
        s1 = lane[0] + lane[1] + lane[2] + lane[3];
        _mm256_storeu_pd(lane, v2);
        s2 = lane[0] + lane[1] + lane[2] + lane[3];
#endif
        for (; i < n; i++) {
                double xi = x[i];
                s0 += xi * y0[i];
                s1 += xi * y1[i];
                s2 += xi * y2[i];
        }
        value[0] = s0;
        value[1] = s1;
        value[2] = s2;
}

static inline void rdm_dot8_ptrs(const double *restrict x,
                                 const double *const y[8],
                                 uint32_t n, double value[8])
{
        for (uint32_t j = 0; j < 8u; j++) value[j] = 0.0;
        uint32_t i = 0u;
#if defined(__AVX2__)
        __m256d sum[8];
        for (uint32_t j = 0; j < 8u; j++) sum[j] = _mm256_setzero_pd();
        for (; i + 4u <= n; i += 4u) {
                __m256d vx = _mm256_loadu_pd(x + i);
                for (uint32_t j = 0; j < 8u; j++) {
                        sum[j] = _mm256_add_pd(
                                sum[j], _mm256_mul_pd(
                                        vx, _mm256_loadu_pd(y[j] + i)));
                }
        }
        for (uint32_t j = 0; j < 8u; j++) {
                double lane[4];
                _mm256_storeu_pd(lane, sum[j]);
                value[j] = lane[0] + lane[1] + lane[2] + lane[3];
        }
#endif
        for (; i < n; i++) {
                double xi = x[i];
                for (uint32_t j = 0; j < 8u; j++) value[j] += xi * y[j][i];
        }
}

static void rdm_transpose_pack(double *restrict packed,
                               const double *restrict block,
                               uint32_t block_nb, uint32_t na,
                               uint32_t nb, uint32_t block_a0)
{
        for (uint32_t a0 = 0; a0 < na; a0 += RDM_TRANSPOSE_TILE) {
                uint32_t a1 = na - a0 < RDM_TRANSPOSE_TILE ?
                              na : a0 + RDM_TRANSPOSE_TILE;
                for (uint32_t b0 = 0; b0 < nb; b0 += RDM_TRANSPOSE_TILE) {
                        uint32_t b1 = nb - b0 < RDM_TRANSPOSE_TILE ?
                                      nb : b0 + RDM_TRANSPOSE_TILE;
                        for (uint32_t b = b0; b < b1; b++) {
                                double *dst = packed + (size_t)b * na + a0;
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (uint32_t a = a0; a < a1; a++) {
                                        dst[a - a0] =
                                                block[(size_t)(block_a0 + a) *
                                                      block_nb + b];
                                }
                        }
                }
        }
}

static int rdm_validate(const gas_space_t *gas,
                        const double *bra, const double *ket)
{
        if (gas == 0 || bra == 0 || ket == 0 ||
            gas->link_format != GAS_LINK_RAW || gas->norb_tot <= 0) {
                return GAS_ERR_INVALID;
        }
        return GAS_SUCCESS;
}

static void rdm1_alpha(const gas_space_t *gas,
                       const double *bra, const double *ket, double *dm1a)
{
        uint32_t norb = (uint32_t)gas->norb_tot;

        for (gas_bid_t bdst = 0; bdst < gas->nblock; bdst++) {
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t incoming = gas->R.row[bd->sa];
                uint32_t nb = gas->sector_nstr[bd->sb];
                const double *dst = bra + bd->offset;

                for (uint32_t it = 0; it < incoming.n; it++) {
                        gas_sid_t asrc = gas->R.src[incoming.off + it];
                        gas_tid_t tid = gas->R.tid[incoming.off + it];
                        gas_bid_t bsrc = gas_find_block(gas, asrc, bd->sb);
                        if (bsrc == GAS_INVALID_BID) continue;

                        const gas_link_table_t *table = gas->table + tid;
                        const double *src = ket + gas->block[bsrc].offset;
                        for (uint32_t ia0 = 0; ia0 < table->nsrc; ia0++) {
                                const double *src_row = src + (size_t)ia0 * nb;
                                const gas_link_entry_t *row = table->link +
                                        (size_t)ia0 * table->nlink;
                                for (uint32_t k = 0; k < table->nlink; k++) {
                                        const gas_link_entry_t *e = row + k;
                                        const double *dst_row = dst + (size_t)e->addr * nb;
                                        uint32_t p = gas_link_cre(e);
                                        uint32_t q = gas_link_des(e);
                                        dm1a[(size_t)q * norb + p] +=
                                                (double)e->sign *
                                                rdm_dot(dst_row, src_row, nb);
                                }
                        }
                }
        }
}

static void rdm1_beta(const gas_space_t *gas,
                      const double *bra, const double *ket, double *dm1b)
{
        uint32_t norb = (uint32_t)gas->norb_tot;

        for (gas_bid_t bdst = 0; bdst < gas->nblock; bdst++) {
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t incoming = gas->R.row[bd->sb];
                uint32_t na = gas->sector_nstr[bd->sa];
                uint32_t nb1 = gas->sector_nstr[bd->sb];
                const double *dst = bra + bd->offset;

                for (uint32_t it = 0; it < incoming.n; it++) {
                        gas_sid_t bsrc_sid = gas->R.src[incoming.off + it];
                        gas_tid_t tid = gas->R.tid[incoming.off + it];
                        gas_bid_t bsrc = gas_find_block(gas, bd->sa, bsrc_sid);
                        if (bsrc == GAS_INVALID_BID) continue;

                        const gas_link_table_t *table = gas->table + tid;
                        uint32_t nb0 = gas->sector_nstr[bsrc_sid];
                        const double *src = ket + gas->block[bsrc].offset;
                        for (uint32_t ib0 = 0; ib0 < table->nsrc; ib0++) {
                                const gas_link_entry_t *row = table->link +
                                        (size_t)ib0 * table->nlink;
                                for (uint32_t k = 0; k < table->nlink; k++) {
                                        const gas_link_entry_t *e = row + k;
                                        double value = 0.0;
                                        for (uint32_t ia = 0; ia < na; ia++) {
                                                value += dst[(size_t)ia * nb1 + e->addr] *
                                                         src[(size_t)ia * nb0 + ib0];
                                        }
                                        uint32_t p = gas_link_cre(e);
                                        uint32_t q = gas_link_des(e);
                                        dm1b[(size_t)q * norb + p] +=
                                                (double)e->sign * value;
                                }
                        }
                }
        }
}

static void rdm2_aa_path(const gas_space_t *gas,
                         const double *bra, const double *ket,
                         gas_bid_t bsrc, gas_bid_t bdst,
                         gas_tid_t tid1, gas_tid_t tid2, double *ph2aa)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *t1 = gas->table + tid1;
        const gas_link_table_t *t2 = gas->table + tid2;
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t nop = norb * norb;
        uint32_t nb = gas->sector_nstr[bs->sb];
        const double *src = ket + bs->offset;
        const double *dst = bra + bd->offset;

        for (uint32_t ia0 = 0; ia0 < t1->nsrc; ia0++) {
                const double *src_row = src + (size_t)ia0 * nb;
                const gas_link_entry_t *r1 = t1->link +
                        (size_t)ia0 * t1->nlink;
                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                        const gas_link_entry_t *e1 = r1 + k1;
                        const gas_link_entry_t *r2 = t2->link +
                                (size_t)e1->addr * t2->nlink;
                        uint32_t op1 = rdm_op(e1, norb);
                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                const gas_link_entry_t *e2 = r2 + k2;
                                const double *dst_row = dst + (size_t)e2->addr * nb;
                                uint32_t op2 = rdm_op(e2, norb);
                                ph2aa[(size_t)op2 * nop + op1] +=
                                        (double)(e1->sign * e2->sign) *
                                        rdm_dot(dst_row, src_row, nb);
                        }
                }
        }
}

static void rdm2_aa(const gas_space_t *gas,
                    const double *bra, const double *ket, double *ph2aa)
{
        for (gas_bid_t bdst = 0; bdst < gas->nblock; bdst++) {
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t reverse = gas->R.row[bd->sa];
                gas_row_t sources = gas->D.by_beta_row[bd->sb];
                const gas_sid_t *middle_rev = gas->R.src + reverse.off;
                const gas_tid_t *second_tid = gas->R.tid + reverse.off;

                for (uint32_t q = 0; q < sources.n; q++) {
                        gas_sid_t asrc = gas->D.by_beta_sid[sources.off + q];
                        gas_bid_t bsrc = gas->D.by_beta_bid[sources.off + q];
                        gas_row_t forward = gas->T.row[asrc];
                        const gas_sid_t *middle_fwd = gas->T.dst + forward.off;
                        uint32_t i = 0;
                        uint32_t j = 0;

                        while (i < forward.n && j < reverse.n) {
                                if (middle_fwd[i] < middle_rev[j]) {
                                        i++;
                                } else if (middle_fwd[i] > middle_rev[j]) {
                                        j++;
                                } else {
                                        rdm2_aa_path(gas, bra, ket, bsrc, bdst,
                                                     forward.off + i, second_tid[j],
                                                     ph2aa);
                                        i++;
                                        j++;
                                }
                        }
                }
        }
}

static void rdm2_bb_path(const gas_space_t *gas,
                         const double *bra, const double *ket,
                         gas_bid_t bsrc, gas_bid_t bdst,
                         gas_tid_t tid1, gas_tid_t tid2, double *ph2bb)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *t1 = gas->table + tid1;
        const gas_link_table_t *t2 = gas->table + tid2;
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t nop = norb * norb;
        uint32_t na = gas->sector_nstr[bs->sa];
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        uint32_t nb2 = gas->sector_nstr[bd->sb];
        const double *src = ket + bs->offset;
        const double *dst = bra + bd->offset;

        for (uint32_t ib0 = 0; ib0 < t1->nsrc; ib0++) {
                const gas_link_entry_t *r1 = t1->link +
                        (size_t)ib0 * t1->nlink;
                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                        const gas_link_entry_t *e1 = r1 + k1;
                        const gas_link_entry_t *r2 = t2->link +
                                (size_t)e1->addr * t2->nlink;
                        uint32_t op1 = rdm_op(e1, norb);
                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                const gas_link_entry_t *e2 = r2 + k2;
                                double value = 0.0;
                                for (uint32_t ia = 0; ia < na; ia++) {
                                        value += dst[(size_t)ia * nb2 + e2->addr] *
                                                 src[(size_t)ia * nb0 + ib0];
                                }
                                uint32_t op2 = rdm_op(e2, norb);
                                ph2bb[(size_t)op2 * nop + op1] +=
                                        (double)(e1->sign * e2->sign) * value;
                        }
                }
        }
}

static void rdm2_bb(const gas_space_t *gas,
                    const double *bra, const double *ket, double *ph2bb)
{
        for (gas_bid_t bdst = 0; bdst < gas->nblock; bdst++) {
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t reverse = gas->R.row[bd->sb];
                gas_row_t sources = gas->D.by_alpha_row[bd->sa];
                const gas_sid_t *middle_rev = gas->R.src + reverse.off;
                const gas_tid_t *second_tid = gas->R.tid + reverse.off;

                for (uint32_t q = 0; q < sources.n; q++) {
                        gas_bid_t bsrc = sources.off + q;
                        gas_sid_t bsrc_sid = gas->block[bsrc].sb;
                        gas_row_t forward = gas->T.row[bsrc_sid];
                        const gas_sid_t *middle_fwd = gas->T.dst + forward.off;
                        uint32_t i = 0;
                        uint32_t j = 0;

                        while (i < forward.n && j < reverse.n) {
                                if (middle_fwd[i] < middle_rev[j]) {
                                        i++;
                                } else if (middle_fwd[i] > middle_rev[j]) {
                                        j++;
                                } else {
                                        rdm2_bb_path(gas, bra, ket, bsrc, bdst,
                                                     forward.off + i, second_tid[j],
                                                     ph2bb);
                                        i++;
                                        j++;
                                }
                        }
                }
        }
}

static void rdm2_ab_path(const gas_space_t *gas,
                         const double *bra, const double *ket,
                         gas_bid_t bsrc, gas_bid_t bdst,
                         gas_tid_t tida, gas_tid_t tidb, double *dm2ab)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *ta = gas->table + tida;
        const gas_link_table_t *tb = gas->table + tidb;
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        uint32_t nb1 = gas->sector_nstr[bd->sb];
        const double *src = ket + bs->offset;
        const double *dst = bra + bd->offset;

        for (uint32_t ia0 = 0; ia0 < ta->nsrc; ia0++) {
                const gas_link_entry_t *ra = ta->link +
                        (size_t)ia0 * ta->nlink;
                for (uint32_t ka = 0; ka < ta->nlink; ka++) {
                        const gas_link_entry_t *ea = ra + ka;
                        uint32_t p = gas_link_cre(ea);
                        uint32_t q = gas_link_des(ea);
                        for (uint32_t ib0 = 0; ib0 < tb->nsrc; ib0++) {
                                double c = src[(size_t)ia0 * nb0 + ib0];
                                const gas_link_entry_t *rb = tb->link +
                                        (size_t)ib0 * tb->nlink;
                                for (uint32_t kb = 0; kb < tb->nlink; kb++) {
                                        const gas_link_entry_t *eb = rb + kb;
                                        uint32_t r = gas_link_cre(eb);
                                        uint32_t s = gas_link_des(eb);
                                        dm2ab[rdm2_index(p, q, r, s, norb)] +=
                                                (double)(ea->sign * eb->sign) *
                                                dst[(size_t)ea->addr * nb1 + eb->addr] * c;
                                }
                        }
                }
        }
}

static void rdm2_ab(const gas_space_t *gas,
                    const double *bra, const double *ket, double *dm2ab)
{
        for (gas_bid_t bdst = 0; bdst < gas->nblock; bdst++) {
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t alpha_in = gas->R.row[bd->sa];

                for (uint32_t i = 0; i < alpha_in.n; i++) {
                        gas_sid_t asrc = gas->R.src[alpha_in.off + i];
                        gas_tid_t tida = gas->R.tid[alpha_in.off + i];
                        gas_row_t source_blocks = gas->D.by_alpha_row[asrc];

                        for (uint32_t q = 0; q < source_blocks.n; q++) {
                                gas_bid_t bsrc = source_blocks.off + q;
                                gas_sid_t bsrc_sid = gas->block[bsrc].sb;
                                gas_tid_t tidb = gas_find_table(gas, bsrc_sid, bd->sb);
                                if (tidb != GAS_INVALID_TID) {
                                        rdm2_ab_path(gas, bra, ket, bsrc, bdst,
                                                     tida, tidb, dm2ab);
                                }
                        }
                }
        }
}

static void rdm_task_bounds(const gas_space_t *gas, const rdm_task_t *task,
                            gas_row_t *row, uint32_t *q0, uint32_t *q1,
                            uint32_t *a0, uint32_t *a1)
{
        *row = gas->D.by_alpha_row[task->adst];
        uint32_t na = gas->sector_nstr[task->adst];
        *q0 = task->q0 < row->n ? task->q0 : row->n;
        *q1 = task->q1 < row->n ? task->q1 : row->n;
        if (*q1 < *q0) *q1 = *q0;
        *a0 = task->a0 < na ? task->a0 : na;
        *a1 = task->a1 < na ? task->a1 : na;
        if (*a1 < *a0) *a1 = *a0;
}

static void rdm1_alpha_task(const gas_space_t *gas,
                            const double *bra, const double *ket,
                            const rdm_task_t *task, double *dm1a)
{
        uint32_t norb = (uint32_t)gas->norb_tot;
        gas_row_t blocks;
        uint32_t q0, q1, a0, a1;
        rdm_task_bounds(gas, task, &blocks, &q0, &q1, &a0, &a1);

        for (uint32_t qb = q0; qb < q1; qb++) {
                gas_bid_t bdst = blocks.off + qb;
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t incoming = gas->R.row[bd->sa];
                uint32_t nb = gas->sector_nstr[bd->sb];
                const double *dst = bra + bd->offset;

                for (uint32_t it = 0; it < incoming.n; it++) {
                        gas_sid_t asrc = gas->R.src[incoming.off + it];
                        gas_tid_t tid = gas->R.tid[incoming.off + it];
                        gas_bid_t bsrc = gas_find_block(gas, asrc, bd->sb);
                        if (bsrc == GAS_INVALID_BID) continue;
                        const gas_link_table_t *table = gas->table + tid;
                        const double *src = ket + gas->block[bsrc].offset;

                        for (uint32_t ia0 = 0; ia0 < table->nsrc; ia0++) {
                                const double *src_row = src + (size_t)ia0 * nb;
                                const gas_link_entry_t *link = table->link +
                                        (size_t)ia0 * table->nlink;
                                for (uint32_t k = 0; k < table->nlink; k++) {
                                        const gas_link_entry_t *e = link + k;
                                        if (e->addr < a0 || e->addr >= a1) continue;
                                        const double *dst_row = dst + (size_t)e->addr * nb;
                                        uint32_t p = gas_link_cre(e);
                                        uint32_t q = gas_link_des(e);
                                        dm1a[(size_t)q * norb + p] +=
                                                (double)e->sign *
                                                rdm_dot(dst_row, src_row, nb);
                                }
                        }
                }
        }
}

static void rdm1_beta_task(const gas_space_t *gas,
                           const double *bra, const double *ket,
                           const rdm_task_t *task, double *dm1b)
{
        uint32_t norb = (uint32_t)gas->norb_tot;
        gas_row_t blocks;
        uint32_t q0, q1, a0, a1;
        rdm_task_bounds(gas, task, &blocks, &q0, &q1, &a0, &a1);

        for (uint32_t qb = q0; qb < q1; qb++) {
                gas_bid_t bdst = blocks.off + qb;
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t incoming = gas->R.row[bd->sb];
                uint32_t nb1 = gas->sector_nstr[bd->sb];
                const double *dst = bra + bd->offset;

                for (uint32_t it = 0; it < incoming.n; it++) {
                        gas_sid_t bsrc_sid = gas->R.src[incoming.off + it];
                        gas_tid_t tid = gas->R.tid[incoming.off + it];
                        gas_bid_t bsrc = gas_find_block(gas, bd->sa, bsrc_sid);
                        if (bsrc == GAS_INVALID_BID) continue;
                        const gas_link_table_t *table = gas->table + tid;
                        uint32_t nb0 = gas->sector_nstr[bsrc_sid];
                        const double *src = ket + gas->block[bsrc].offset;

                        for (uint32_t ib0 = 0; ib0 < table->nsrc; ib0++) {
                                const gas_link_entry_t *link = table->link +
                                        (size_t)ib0 * table->nlink;
                                for (uint32_t k = 0; k < table->nlink; k++) {
                                        const gas_link_entry_t *e = link + k;
                                        double value = 0.0;
                                        for (uint32_t ia = a0; ia < a1; ia++) {
                                                value += dst[(size_t)ia * nb1 + e->addr] *
                                                         src[(size_t)ia * nb0 + ib0];
                                        }
                                        uint32_t p = gas_link_cre(e);
                                        uint32_t q = gas_link_des(e);
                                        dm1b[(size_t)q * norb + p] +=
                                                (double)e->sign * value;
                                }
                        }
                }
        }
}

static void rdm2_aa_path_task(const gas_space_t *gas,
                              const double *bra, const double *ket,
                              gas_bid_t bsrc, gas_bid_t bdst,
                              gas_tid_t tid1, gas_tid_t tid2,
                              uint32_t a0, uint32_t a1, double *ph2aa)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *t1 = gas->table + tid1;
        const gas_link_table_t *t2 = gas->table + tid2;
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t nop = norb * norb;
        uint32_t nb = gas->sector_nstr[bs->sb];
        const double *src = ket + bs->offset;
        const double *dst = bra + bd->offset;

        for (uint32_t ia0 = 0; ia0 < t1->nsrc; ia0++) {
                const double *src_row = src + (size_t)ia0 * nb;
                const gas_link_entry_t *r1 = t1->link +
                        (size_t)ia0 * t1->nlink;
                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                        const gas_link_entry_t *e1 = r1 + k1;
                        const gas_link_entry_t *r2 = t2->link +
                                (size_t)e1->addr * t2->nlink;
                        uint32_t op1 = rdm_op(e1, norb);
                        const gas_link_entry_t *touched[8];
                        uint32_t ntouched = 0;
                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                const gas_link_entry_t *e2 = r2 + k2;
                                if (e2->addr < a0 || e2->addr >= a1) continue;
                                touched[ntouched++] = e2;
                                if (ntouched == RDM_AA_GROUP_SIZE) {
                                        const double *row[8];
                                        double value[8];
                                        for (uint32_t u = 0; u < 8u; u++) {
                                                row[u] = dst +
                                                        (size_t)touched[u]->addr * nb;
                                        }
                                        rdm_dot8_ptrs(src_row, row, nb, value);
                                        for (uint32_t u = 0; u < 8u; u++) {
                                                uint32_t op2 = rdm_op(touched[u], norb);
                                                ph2aa[(size_t)op2 * nop + op1] +=
                                                        (double)(e1->sign *
                                                        touched[u]->sign) * value[u];
                                        }
                                        ntouched = 0u;
                                }
                        }
                        if (ntouched >= 4u) {
                                double value[4];
                                rdm_dot4(src_row,
                                        dst + (size_t)touched[0]->addr * nb,
                                        dst + (size_t)touched[1]->addr * nb,
                                        dst + (size_t)touched[2]->addr * nb,
                                        dst + (size_t)touched[3]->addr * nb,
                                        nb, value);
                                for (uint32_t u = 0; u < 4u; u++) {
                                        uint32_t op2 = rdm_op(touched[u], norb);
                                        ph2aa[(size_t)op2 * nop + op1] +=
                                                (double)(e1->sign *
                                                touched[u]->sign) * value[u];
                                }
                                for (uint32_t u = 4u; u < ntouched; u++) {
                                        touched[u - 4u] = touched[u];
                                }
                                ntouched -= 4u;
                        }
                        if (ntouched == 3u) {
                                double value[3];
                                rdm_dot3(src_row,
                                        dst + (size_t)touched[0]->addr * nb,
                                        dst + (size_t)touched[1]->addr * nb,
                                        dst + (size_t)touched[2]->addr * nb,
                                        nb, value);
                                for (uint32_t u = 0; u < 3u; u++) {
                                        uint32_t op2 = rdm_op(touched[u], norb);
                                        ph2aa[(size_t)op2 * nop + op1] +=
                                                (double)(e1->sign *
                                                touched[u]->sign) * value[u];
                                }
                                ntouched = 0u;
                        } else if (ntouched == 2u) {
                                double value[2];
                                rdm_dot2(src_row,
                                        dst + (size_t)touched[0]->addr * nb,
                                        dst + (size_t)touched[1]->addr * nb,
                                        nb, value);
                                for (uint32_t u = 0; u < 2u; u++) {
                                        uint32_t op2 = rdm_op(touched[u], norb);
                                        ph2aa[(size_t)op2 * nop + op1] +=
                                                (double)(e1->sign *
                                                touched[u]->sign) * value[u];
                                }
                                ntouched = 0u;
                        }
                        for (uint32_t u = 0; u < ntouched; u++) {
                                const gas_link_entry_t *e2 = touched[u];
                                const double *dst_row = dst + (size_t)e2->addr * nb;
                                uint32_t op2 = rdm_op(e2, norb);
                                ph2aa[(size_t)op2 * nop + op1] +=
                                        (double)(e1->sign * e2->sign) *
                                        rdm_dot(dst_row, src_row, nb);
                        }
                }
        }
}

static void rdm2_aa_task(const gas_space_t *gas,
                         const double *bra, const double *ket,
                         const rdm_task_t *task, double *ph2aa)
{
        gas_row_t blocks;
        uint32_t q0, q1, a0, a1;
        rdm_task_bounds(gas, task, &blocks, &q0, &q1, &a0, &a1);

        for (uint32_t qb = q0; qb < q1; qb++) {
                gas_bid_t bdst = blocks.off + qb;
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t reverse = gas->R.row[bd->sa];
                gas_row_t sources = gas->D.by_beta_row[bd->sb];
                const gas_sid_t *middle_rev = gas->R.src + reverse.off;
                const gas_tid_t *second_tid = gas->R.tid + reverse.off;

                for (uint32_t q = 0; q < sources.n; q++) {
                        gas_sid_t asrc = gas->D.by_beta_sid[sources.off + q];
                        gas_bid_t bsrc = gas->D.by_beta_bid[sources.off + q];
                        gas_row_t forward = gas->T.row[asrc];
                        const gas_sid_t *middle_fwd = gas->T.dst + forward.off;
                        uint32_t i = 0, j = 0;
                        while (i < forward.n && j < reverse.n) {
                                if (middle_fwd[i] < middle_rev[j]) {
                                        i++;
                                } else if (middle_fwd[i] > middle_rev[j]) {
                                        j++;
                                } else {
                                        rdm2_aa_path_task(gas, bra, ket, bsrc, bdst,
                                                          forward.off + i,
                                                          second_tid[j], a0, a1,
                                                          ph2aa);
                                        i++;
                                        j++;
                                }
                        }
                }
        }
}

static void rdm2_bb_path_task(const gas_space_t *gas,
                              const double *bra, const double *ket,
                              gas_bid_t bsrc, gas_bid_t bdst,
                              gas_tid_t tid1, gas_tid_t tid2,
                              uint32_t a0, uint32_t a1, double *ph2bb)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *t1 = gas->table + tid1;
        const gas_link_table_t *t2 = gas->table + tid2;
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t nop = norb * norb;
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        uint32_t nb2 = gas->sector_nstr[bd->sb];
        const double *src = ket + bs->offset;
        const double *dst = bra + bd->offset;

        for (uint32_t ib0 = 0; ib0 < t1->nsrc; ib0++) {
                const gas_link_entry_t *r1 = t1->link +
                        (size_t)ib0 * t1->nlink;
                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                        const gas_link_entry_t *e1 = r1 + k1;
                        const gas_link_entry_t *r2 = t2->link +
                                (size_t)e1->addr * t2->nlink;
                        uint32_t op1 = rdm_op(e1, norb);
                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                const gas_link_entry_t *e2 = r2 + k2;
                                double value = 0.0;
                                for (uint32_t ia = a0; ia < a1; ia++) {
                                        value += dst[(size_t)ia * nb2 + e2->addr] *
                                                 src[(size_t)ia * nb0 + ib0];
                                }
                                uint32_t op2 = rdm_op(e2, norb);
                                ph2bb[(size_t)op2 * nop + op1] +=
                                        (double)(e1->sign * e2->sign) * value;
                        }
                }
        }
}

static void rdm2_bb_pair_list_task(const gas_space_t *gas,
                                   const double *bra, const double *ket,
                                   gas_bid_t bsrc, gas_bid_t bdst,
                                   const rdm_tid_pair_t *pair, uint32_t npair,
                                   uint32_t a0, uint32_t a1, double *ph2bb,
                                   rdm_workspace_t *workspace)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        uint32_t nb2 = gas->sector_nstr[bd->sb];
        uint32_t nat = a1 > a0 ? a1 - a0 : 0u;
        size_t src_elems = (size_t)nb0 * nat;
        size_t dst_elems = (size_t)nb2 * nat;

        if (nat >= RDM_BB_TRANSPOSE_MIN_ALPHA &&
            src_elems + dst_elems <= RDM_WORKSPACE_MAX_ELEMS &&
            rdm_arena_reserve(workspace, src_elems + dst_elems) == GAS_SUCCESS) {
                double *src_t = workspace->arena;
                double *dst_t = src_t + src_elems;
                rdm_transpose_pack(src_t, ket + bs->offset, nb0,
                                   nat, nb0, a0);
                rdm_transpose_pack(dst_t, bra + bd->offset, nb2,
                                   nat, nb2, a0);
                uint32_t norb = (uint32_t)gas->norb_tot;
                uint32_t nop = norb * norb;
                for (uint32_t ip = 0; ip < npair; ip++) {
                        const gas_link_table_t *t1 = gas->table + pair[ip].first;
                        const gas_link_table_t *t2 = gas->table + pair[ip].second;
                        for (uint32_t ib0 = 0; ib0 < t1->nsrc; ib0++) {
                                const double *src_vec = src_t + (size_t)ib0 * nat;
                                const gas_link_entry_t *r1 = t1->link +
                                        (size_t)ib0 * t1->nlink;
                                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                                        const gas_link_entry_t *e1 = r1 + k1;
                                        const gas_link_entry_t *r2 = t2->link +
                                                (size_t)e1->addr * t2->nlink;
                                        uint32_t op1 = rdm_op(e1, norb);
                                        uint32_t k2 = 0u;
                                        for (; k2 + 8u <= t2->nlink; k2 += 8u) {
                                                const double *row[8];
                                                double value[8];
                                                for (uint32_t u = 0; u < 8u; u++) {
                                                        row[u] = dst_t +
                                                                (size_t)r2[k2 + u].addr * nat;
                                                }
                                                rdm_dot8_ptrs(src_vec, row, nat, value);
                                                for (uint32_t u = 0; u < 8u; u++) {
                                                        const gas_link_entry_t *e2 = r2 + k2 + u;
                                                        uint32_t op2 = rdm_op(e2, norb);
                                                        ph2bb[(size_t)op2 * nop + op1] +=
                                                                (double)(e1->sign * e2->sign) *
                                                                value[u];
                                                }
                                        }
                                        if (k2 + 4u <= t2->nlink) {
                                                double value[4];
                                                rdm_dot4(src_vec,
                                                        dst_t + (size_t)r2[k2].addr * nat,
                                                        dst_t + (size_t)r2[k2 + 1u].addr * nat,
                                                        dst_t + (size_t)r2[k2 + 2u].addr * nat,
                                                        dst_t + (size_t)r2[k2 + 3u].addr * nat,
                                                        nat, value);
                                                for (uint32_t u = 0; u < 4u; u++) {
                                                        const gas_link_entry_t *e2 = r2 + k2 + u;
                                                        uint32_t op2 = rdm_op(e2, norb);
                                                        ph2bb[(size_t)op2 * nop + op1] +=
                                                                (double)(e1->sign * e2->sign) *
                                                                value[u];
                                                }
                                                k2 += 4u;
                                        }
                                        for (; k2 < t2->nlink; k2++) {
                                                const gas_link_entry_t *e2 = r2 + k2;
                                                uint32_t op2 = rdm_op(e2, norb);
                                                double value = rdm_dot(
                                                        dst_t + (size_t)e2->addr * nat,
                                                        src_vec, nat);
                                                ph2bb[(size_t)op2 * nop + op1] +=
                                                        (double)(e1->sign * e2->sign) *
                                                        value;
                                        }
                                }
                        }
                }
                return;
        }

        for (uint32_t ip = 0; ip < npair; ip++) {
                rdm2_bb_path_task(gas, bra, ket, bsrc, bdst,
                                  pair[ip].first, pair[ip].second,
                                  a0, a1, ph2bb);
        }
}

static void rdm2_bb_task(const gas_space_t *gas,
                         const double *bra, const double *ket,
                         const rdm_task_t *task, double *ph2bb,
                         rdm_workspace_t *workspace)
{
        gas_row_t blocks;
        uint32_t q0, q1, a0, a1;
        rdm_task_bounds(gas, task, &blocks, &q0, &q1, &a0, &a1);

        for (uint32_t qb = q0; qb < q1; qb++) {
                gas_bid_t bdst = blocks.off + qb;
                const gas_block_t *bd = gas->block + bdst;
                gas_row_t reverse = gas->R.row[bd->sb];
                gas_row_t sources = gas->D.by_alpha_row[bd->sa];
                const gas_sid_t *middle_rev = gas->R.src + reverse.off;
                const gas_tid_t *second_tid = gas->R.tid + reverse.off;

                for (uint32_t q = 0; q < sources.n; q++) {
                        gas_bid_t bsrc = sources.off + q;
                        gas_sid_t bsrc_sid = gas->block[bsrc].sb;
                        gas_row_t forward = gas->T.row[bsrc_sid];
                        const gas_sid_t *middle_fwd = gas->T.dst + forward.off;
                        uint32_t i = 0, j = 0;
                        uint32_t maxpair = forward.n < reverse.n ?
                                           forward.n : reverse.n;
                        if (rdm_pair_reserve(workspace, maxpair) != GAS_SUCCESS) {
                                maxpair = 0u;
                        }
                        uint32_t npair = 0;
                        while (i < forward.n && j < reverse.n) {
                                if (middle_fwd[i] < middle_rev[j]) {
                                        i++;
                                } else if (middle_fwd[i] > middle_rev[j]) {
                                        j++;
                                } else {
                                        if (npair < maxpair) {
                                                workspace->pair[npair].first =
                                                        forward.off + i;
                                                workspace->pair[npair].second =
                                                        second_tid[j];
                                                npair++;
                                        } else {
                                                rdm2_bb_path_task(
                                                        gas, bra, ket, bsrc, bdst,
                                                        forward.off + i,
                                                        second_tid[j], a0, a1,
                                                        ph2bb);
                                        }
                                        i++;
                                        j++;
                                }
                        }
                        if (npair != 0u) {
                                rdm2_bb_pair_list_task(gas, bra, ket, bsrc, bdst,
                                                       workspace->pair, npair,
                                                       a0, a1, ph2bb, workspace);
                        }
                }
        }
}

static void rdm2_ab_path_fallback(const gas_space_t *gas,
                                  const double *bra, const double *ket,
                                  gas_bid_t bsrc, gas_bid_t bdst,
                                  gas_tid_t tida, gas_tid_t tidb,
                                  uint32_t a0, uint32_t a1, double *dm2ab)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *ta = gas->table + tida;
        const gas_link_table_t *tb = gas->table + tidb;
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        uint32_t nb1 = gas->sector_nstr[bd->sb];
        const double *src = ket + bs->offset;
        const double *dst = bra + bd->offset;

        for (uint32_t ia0 = 0; ia0 < ta->nsrc; ia0++) {
                const gas_link_entry_t *ra = ta->link +
                        (size_t)ia0 * ta->nlink;
                for (uint32_t ka = 0; ka < ta->nlink; ka++) {
                        const gas_link_entry_t *ea = ra + ka;
                        if (ea->addr < a0 || ea->addr >= a1) continue;
                        uint32_t p = gas_link_cre(ea);
                        uint32_t q = gas_link_des(ea);
                        for (uint32_t ib0 = 0; ib0 < tb->nsrc; ib0++) {
                                double c = src[(size_t)ia0 * nb0 + ib0];
                                const gas_link_entry_t *rb = tb->link +
                                        (size_t)ib0 * tb->nlink;
                                for (uint32_t kb = 0; kb < tb->nlink; kb++) {
                                        const gas_link_entry_t *eb = rb + kb;
                                        uint32_t r = gas_link_cre(eb);
                                        uint32_t s = gas_link_des(eb);
                                        dm2ab[rdm2_index(p, q, r, s, norb)] +=
                                                (double)(ea->sign * eb->sign) *
                                                dst[(size_t)ea->addr * nb1 + eb->addr] * c;
                                }
                        }
                }
        }
}

#if defined(__AVX2__)
static inline double rdm_hsum256(__m256d value)
{
        double lane[4];
        _mm256_storeu_pd(lane, value);
        return lane[0] + lane[1] + lane[2] + lane[3];
}
#endif

static void rdm_dot8_streams(const double *restrict y,
                             const double *restrict base, size_t stride,
                             uint32_t n, double value[8])
{
        for (uint32_t j = 0; j < 8u; j++) value[j] = 0.0;
        uint32_t i = 0;
#if defined(__AVX2__)
        __m256d sum[8];
        for (uint32_t j = 0; j < 8u; j++) sum[j] = _mm256_setzero_pd();
        for (; i + 4u <= n; i += 4u) {
                __m256d vy = _mm256_loadu_pd(y + i);
                for (uint32_t j = 0; j < 8u; j++) {
                        __m256d vt = _mm256_loadu_pd(base + (size_t)j * stride + i);
                        sum[j] = _mm256_add_pd(sum[j], _mm256_mul_pd(vy, vt));
                }
        }
        for (uint32_t j = 0; j < 8u; j++) value[j] = rdm_hsum256(sum[j]);
#endif
        for (; i < n; i++) {
                double yi = y[i];
                for (uint32_t j = 0; j < 8u; j++) {
                        value[j] += yi * base[(size_t)j * stride + i];
                }
        }
}

static void rdm_dot2x4_streams(const double *restrict y0,
                               const double *restrict y1,
                               const double *restrict base, size_t stride,
                               uint32_t n, double value0[4], double value1[4])
{
        for (uint32_t j = 0; j < 4u; j++) {
                value0[j] = 0.0;
                value1[j] = 0.0;
        }
        uint32_t i = 0;
#if defined(__AVX2__)
        __m256d sum0[4], sum1[4];
        for (uint32_t j = 0; j < 4u; j++) {
                sum0[j] = _mm256_setzero_pd();
                sum1[j] = _mm256_setzero_pd();
        }
        for (; i + 4u <= n; i += 4u) {
                __m256d vy0 = _mm256_loadu_pd(y0 + i);
                __m256d vy1 = _mm256_loadu_pd(y1 + i);
                for (uint32_t j = 0; j < 4u; j++) {
                        __m256d vt = _mm256_loadu_pd(base + (size_t)j * stride + i);
                        sum0[j] = _mm256_add_pd(sum0[j], _mm256_mul_pd(vy0, vt));
                        sum1[j] = _mm256_add_pd(sum1[j], _mm256_mul_pd(vy1, vt));
                }
        }
        for (uint32_t j = 0; j < 4u; j++) {
                value0[j] = rdm_hsum256(sum0[j]);
                value1[j] = rdm_hsum256(sum1[j]);
        }
#endif
        for (; i < n; i++) {
                double a = y0[i], b = y1[i];
                for (uint32_t j = 0; j < 4u; j++) {
                        double t = base[(size_t)j * stride + i];
                        value0[j] += a * t;
                        value1[j] += b * t;
                }
        }
}

static inline size_t rdm_abba_row_offset(const uint32_t *active_row,
                                         uint32_t i)
{
        return active_row[i];
}

static void rdm_abba_fused_ops(double *dm2ab,
                               const uint32_t *active_row, uint32_t nactive,
                               uint32_t opb, double sign,
                               const double *y, const double *tbase,
                               size_t op_stride, uint32_t nat)
{
        uint32_t io = 0;
        for (; io + 8u <= nactive; io += 8u) {
                double value[8];
                rdm_dot8_streams(y, tbase + (size_t)io * op_stride,
                                 op_stride, nat, value);
                for (uint32_t j = 0; j < 8u; j++) {
                        dm2ab[rdm_abba_row_offset(active_row, io + j) + opb] +=
                                sign * value[j];
                }
        }
        for (; io + 4u <= nactive; io += 4u) {
                double value[4];
                const double *base = tbase + (size_t)io * op_stride;
                rdm_dot4(y, base, base + op_stride,
                         base + 2u * op_stride, base + 3u * op_stride,
                         nat, value);
                for (uint32_t j = 0; j < 4u; j++) {
                        dm2ab[rdm_abba_row_offset(active_row, io + j) + opb] +=
                                sign * value[j];
                }
        }
        for (; io < nactive; io++) {
                double value = rdm_dot(y, tbase + (size_t)io * op_stride, nat);
                dm2ab[rdm_abba_row_offset(active_row, io) + opb] += sign * value;
        }
}

static void rdm_abba_pair2x4(double *dm2ab,
                             const uint32_t *active_row, uint32_t nactive,
                             uint32_t opb0, uint32_t opb1,
                             double sign0, double sign1,
                             const double *y0, const double *y1,
                             const double *tbase, size_t op_stride,
                             uint32_t nat)
{
        uint32_t io = 0;
        for (; io + 4u <= nactive; io += 4u) {
                double value0[4], value1[4];
                rdm_dot2x4_streams(y0, y1,
                                   tbase + (size_t)io * op_stride,
                                   op_stride, nat, value0, value1);
                for (uint32_t j = 0; j < 4u; j++) {
                        size_t row = rdm_abba_row_offset(active_row, io + j);
                        dm2ab[row + opb0] += sign0 * value0[j];
                        dm2ab[row + opb1] += sign1 * value1[j];
                }
        }
        if (io < nactive) {
                rdm_abba_fused_ops(dm2ab, active_row + io,
                                   nactive - io, opb0, sign0, y0,
                                   tbase + (size_t)io * op_stride,
                                   op_stride, nat);
                rdm_abba_fused_ops(dm2ab, active_row + io,
                                   nactive - io, opb1, sign1, y1,
                                   tbase + (size_t)io * op_stride,
                                   op_stride, nat);
        }
}

static uint32_t rdm_abba_beta_tile(uint32_t nb0, uint32_t nactive,
                                   uint32_t nat, size_t target_bytes)
{
        if (nb0 == 0u || nactive == 0u || nat == 0u || target_bytes == 0u) {
                return nb0;
        }
        if ((size_t)nactive > SIZE_MAX / nat / sizeof(double)) return 1u;
        size_t denom = (size_t)nactive * nat * sizeof(double);
        size_t tile = denom ? target_bytes / denom : nb0;
        if (tile < 1u) tile = 1u;
        if (tile > nb0) tile = nb0;
        return (uint32_t)tile;
}

static int rdm_build_alpha_hitlist(const gas_space_t *gas,
                                   const gas_rdm_plan_t *plan,
                                   gas_tid_t tida, uint32_t a0, uint32_t a1,
                                   rdm_workspace_t *workspace)
{
        const gas_link_table_t *ta = gas->table + tida;
        const rdm_table_ops_t *meta = plan->table_ops + tida;
        uint32_t nopair = (uint32_t)gas->norb_tot * gas->norb_tot;
        if (rdm_op_map_reserve(workspace, nopair) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        for (uint32_t i = 0; i < nopair; i++) workspace->op_map[i] = -1;
        const uint16_t *active = plan->active_op + meta->off;
        for (uint32_t i = 0; i < meta->n; i++) workspace->op_map[active[i]] = (int32_t)i;

        uint32_t nhit = 0;
        for (uint32_t ia0 = 0; ia0 < ta->nsrc; ia0++) {
                const gas_link_entry_t *row = ta->link + (size_t)ia0 * ta->nlink;
                for (uint32_t k = 0; k < ta->nlink; k++) {
                        if (row[k].addr >= a0 && row[k].addr < a1) nhit++;
                }
        }
        if (rdm_hit_reserve(workspace, nhit) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        uint32_t out = 0;
        for (uint32_t ia0 = 0; ia0 < ta->nsrc; ia0++) {
                const gas_link_entry_t *row = ta->link + (size_t)ia0 * ta->nlink;
                for (uint32_t k = 0; k < ta->nlink; k++) {
                        const gas_link_entry_t *e = row + k;
                        if (e->addr < a0 || e->addr >= a1) continue;
                        int32_t io = workspace->op_map[rdm_op(e,
                                (uint32_t)gas->norb_tot)];
                        if (io < 0) continue;
                        workspace->alpha_hit[out].src_row = ia0;
                        workspace->alpha_hit[out].rel_addr = e->addr - a0;
                        workspace->alpha_hit[out].op_index = (uint16_t)io;
                        workspace->alpha_hit[out].sign = e->sign;
                        workspace->alpha_hit[out].padding = 0u;
                        out++;
                }
        }
        workspace->alpha_hit_n = out;
        return out == nhit ? GAS_SUCCESS : GAS_ERR_INVALID;
}

static int rdm_collect_ab_pairs(const gas_space_t *gas, gas_sid_t adst,
                                uint32_t q0, uint32_t q1,
                                gas_row_t beta_tables,
                                rdm_workspace_t *workspace,
                                uint32_t *npair_out)
{
        gas_row_t blocks = gas->D.by_alpha_row[adst];
        if (q0 > blocks.n) q0 = blocks.n;
        if (q1 > blocks.n) q1 = blocks.n;
        if (q1 < q0) q1 = q0;
        uint32_t maxpair = q1 - q0 < beta_tables.n ?
                           q1 - q0 : beta_tables.n;
        if (rdm_pair_reserve(workspace, maxpair) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        uint32_t q = q0, j = 0u, npair = 0u;
        while (q < q1 && j < beta_tables.n) {
                gas_bid_t bdst = blocks.off + q;
                gas_sid_t bdst_beta = gas->block[bdst].sb;
                gas_sid_t table_beta = gas->T.dst[beta_tables.off + j];
                if (bdst_beta < table_beta) {
                        q++;
                } else if (bdst_beta > table_beta) {
                        j++;
                } else {
                        workspace->pair[npair].first = bdst;
                        workspace->pair[npair].second = beta_tables.off + j;
                        npair++;
                        q++;
                        j++;
                }
        }
        *npair_out = npair;
        return GAS_SUCCESS;
}

static int rdm2_ab_batch_tiled(const gas_rdm_plan_t *plan,
                               const double *bra, const double *ket,
                               gas_bid_t bsrc, gas_tid_t tida,
                               const rdm_tid_pair_t *pair, uint32_t npair,
                               uint32_t a0, uint32_t a1, double *dm2ab,
                               rdm_workspace_t *workspace)
{
        const gas_space_t *gas = plan->gas;
        const gas_block_t *bs = gas->block + bsrc;
        const rdm_table_ops_t *meta = plan->table_ops + tida;
        uint32_t nactive = meta->n;
        uint32_t nat = a1 > a0 ? a1 - a0 : 0u;
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        if (nactive == 0u || nat == 0u || nb0 == 0u || npair == 0u) {
                return GAS_SUCCESS;
        }
        if (rdm_build_alpha_hitlist(gas, plan, tida, a0, a1,
                                    workspace) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }

        uint32_t btile = rdm_abba_beta_tile(nb0, nactive, nat,
                                            plan->abba_t1_target_bytes);
        if (btile == 0u) return GAS_SUCCESS;
        if ((size_t)btile > SIZE_MAX / nat) return GAS_ERR_MEMORY;
        uint32_t max_nb1 = 0u;
        size_t all_packed_elems = 0u;
        for (uint32_t ip = 0; ip < npair; ip++) {
                const gas_block_t *bd = gas->block + pair[ip].first;
                uint32_t nb1 = gas->sector_nstr[bd->sb];
                if (nb1 > max_nb1) max_nb1 = nb1;
                if ((size_t)nb1 > SIZE_MAX / nat ||
                    all_packed_elems > SIZE_MAX - (size_t)nb1 * nat) {
                        return GAS_ERR_MEMORY;
                }
                all_packed_elems += (size_t)nb1 * nat;
        }
        if ((size_t)max_nb1 > SIZE_MAX / nat) return GAS_ERR_MEMORY;
        size_t op_stride = (size_t)btile * nat;
        size_t one_packed_elems = (size_t)max_nb1 * nat;
        if ((size_t)nactive > SIZE_MAX / op_stride) return GAS_ERR_MEMORY;
        size_t t1_elems = (size_t)nactive * op_stride;
        int pack_once = 0;
        if (t1_elems <= RDM_WORKSPACE_MAX_ELEMS &&
            all_packed_elems <= RDM_WORKSPACE_MAX_ELEMS - t1_elems &&
            rdm_arena_reserve(workspace, t1_elems + all_packed_elems) ==
                    GAS_SUCCESS) {
                pack_once = 1;
        }
        if (!pack_once &&
            (t1_elems > RDM_WORKSPACE_MAX_ELEMS ||
             one_packed_elems > RDM_WORKSPACE_MAX_ELEMS - t1_elems ||
             rdm_arena_reserve(workspace, t1_elems + one_packed_elems) !=
                    GAS_SUCCESS)) {
                return GAS_ERR_MEMORY;
        }
        double *t1 = workspace->arena;
        double *packed = t1 + t1_elems;
        memset(t1, 0, t1_elems * sizeof(*t1));
        const double *src = ket + bs->offset;
        const uint32_t *active_row = plan->active_row + meta->off;

        if (pack_once) {
                size_t off = 0u;
                for (uint32_t ip = 0; ip < npair; ip++) {
                        const gas_block_t *bd = gas->block + pair[ip].first;
                        uint32_t nb1 = gas->sector_nstr[bd->sb];
                        rdm_transpose_pack(packed + off, bra + bd->offset, nb1,
                                           nat, nb1, a0);
                        off += (size_t)nb1 * nat;
                }
        }

        for (uint32_t b0 = 0; b0 < nb0; b0 += btile) {
                uint32_t b1 = nb0 - b0 < btile ? nb0 : b0 + btile;
                for (uint32_t ih = 0; ih < workspace->alpha_hit_n; ih++) {
                        const rdm_alpha_hit_t *h = workspace->alpha_hit + ih;
                        const double *src_row = src + (size_t)h->src_row * nb0;
                        double *stream = t1 + (size_t)h->op_index * op_stride +
                                         h->rel_addr;
                        double sign = (double)h->sign;
                        for (uint32_t ib = b0; ib < b1; ib++) {
                                stream[(size_t)(ib - b0) * nat] = sign * src_row[ib];
                        }
                }

                size_t packed_off = 0u;
                for (uint32_t ip = 0; ip < npair; ip++) {
                        gas_bid_t bdst = pair[ip].first;
                        gas_tid_t tidb = pair[ip].second;
                        const gas_block_t *bd = gas->block + bdst;
                        const gas_link_table_t *tb = gas->table + tidb;
                        uint32_t nb1 = gas->sector_nstr[bd->sb];
                        double *dst_t = pack_once ? packed + packed_off : packed;
                        if (!pack_once) {
                                rdm_transpose_pack(dst_t, bra + bd->offset, nb1,
                                                   nat, nb1, a0);
                        }
                        for (uint32_t ib = b0; ib < b1; ib++) {
                                const gas_link_entry_t *row = tb->link +
                                        (size_t)ib * tb->nlink;
                                const double *tbase = t1 +
                                        (size_t)(ib - b0) * nat;
                                uint32_t k = 0u;
                                for (; k + 1u < tb->nlink; k += 2u) {
                                        const gas_link_entry_t *e0 = row + k;
                                        const gas_link_entry_t *e1 = row + k + 1u;
                                        rdm_abba_pair2x4(
                                                dm2ab, active_row, nactive,
                                                rdm_op(e0, (uint32_t)gas->norb_tot),
                                                rdm_op(e1, (uint32_t)gas->norb_tot),
                                                (double)e0->sign,
                                                (double)e1->sign,
                                                dst_t + (size_t)e0->addr * nat,
                                                dst_t + (size_t)e1->addr * nat,
                                                tbase, op_stride, nat);
                                }
                                for (; k < tb->nlink; k++) {
                                        const gas_link_entry_t *e = row + k;
                                        rdm_abba_fused_ops(
                                                dm2ab, active_row, nactive,
                                                rdm_op(e, (uint32_t)gas->norb_tot),
                                                (double)e->sign,
                                                dst_t + (size_t)e->addr * nat,
                                                tbase, op_stride, nat);
                                }
                        }
                        packed_off += (size_t)nb1 * nat;
                }
        }
        return GAS_SUCCESS;
}

static void rdm2_ab_task(const gas_rdm_plan_t *plan,
                         const double *bra, const double *ket,
                         const rdm_task_t *task, double *dm2ab,
                         rdm_workspace_t *workspace)
{
        const gas_space_t *gas = plan->gas;
        gas_row_t blocks;
        uint32_t q0, q1, a0, a1;
        rdm_task_bounds(gas, task, &blocks, &q0, &q1, &a0, &a1);

        gas_row_t alpha_in = gas->R.row[task->adst];
        for (uint32_t i = 0; i < alpha_in.n; i++) {
                gas_sid_t asrc = gas->R.src[alpha_in.off + i];
                gas_tid_t tida = gas->R.tid[alpha_in.off + i];
                gas_row_t source_blocks = gas->D.by_alpha_row[asrc];
                for (uint32_t q = 0; q < source_blocks.n; q++) {
                        gas_bid_t bsrc = source_blocks.off + q;
                        gas_sid_t beta0 = gas->block[bsrc].sb;
                        gas_row_t beta_tables = gas->T.row[beta0];
                        uint32_t npair = 0u;
                        if (rdm_collect_ab_pairs(gas, task->adst, q0, q1,
                                beta_tables, workspace, &npair) != GAS_SUCCESS) {
                                for (uint32_t qb = q0; qb < q1; qb++) {
                                        gas_bid_t bdst = blocks.off + qb;
                                        gas_tid_t tidb = gas_find_table(
                                                gas, beta0, gas->block[bdst].sb);
                                        if (tidb == GAS_INVALID_TID) continue;
                                        rdm2_ab_path_fallback(
                                                gas, bra, ket, bsrc, bdst,
                                                tida, tidb, a0, a1, dm2ab);
                                }
                                continue;
                        }
                        if (npair == 0u) continue;
                        if (rdm2_ab_batch_tiled(
                                plan, bra, ket, bsrc, tida,
                                workspace->pair, npair, a0, a1,
                                dm2ab, workspace) != GAS_SUCCESS) {
                                for (uint32_t ip = 0; ip < npair; ip++) {
                                        rdm2_ab_path_fallback(
                                                gas, bra, ket, bsrc,
                                                workspace->pair[ip].first,
                                                tida, workspace->pair[ip].second,
                                                a0, a1, dm2ab);
                                }
                        }
                }
        }
}

static void rdm12_task_accumulate(const gas_rdm_plan_t *plan,
                                  const double *bra, const double *ket,
                                  const rdm_task_t *task,
                                  rdm_workspace_t *workspace,
                                  size_t n1, size_t n2)
{
        const gas_space_t *gas = plan->gas;
        double *work = workspace->data;
        double *dm1a = work;
        double *dm1b = dm1a + n1;
        double *dm2aa = dm1b + n1;
        double *dm2ab = dm2aa + n2;
        double *dm2bb = dm2ab + n2;
        rdm1_alpha_task(gas, bra, ket, task, dm1a);
        rdm1_beta_task(gas, bra, ket, task, dm1b);
        rdm2_aa_task(gas, bra, ket, task, dm2aa);
        rdm2_ab_task(plan, bra, ket, task, dm2ab, workspace);
        rdm2_bb_task(gas, bra, ket, task, dm2bb, workspace);
}

static uint64_t rdm_cost_add(uint64_t a, uint64_t b)
{
        return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static uint64_t rdm_cost_mul(uint64_t a, uint64_t b)
{
        return a != 0u && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static uint64_t rdm_destination_block_cost(const gas_space_t *gas,
                                           gas_bid_t bdst)
{
        const gas_block_t *bd = gas->block + bdst;
        uint64_t na1 = gas->sector_nstr[bd->sa];
        uint64_t nb1 = gas->sector_nstr[bd->sb];
        uint64_t cost = 0u;
        gas_row_t alpha_reverse = gas->R.row[bd->sa];
        gas_row_t beta_reverse = gas->R.row[bd->sb];

        /* One-body alpha and beta links. */
        for (uint32_t i = 0; i < alpha_reverse.n; i++) {
                gas_sid_t asrc = gas->R.src[alpha_reverse.off + i];
                if (gas_find_block(gas, asrc, bd->sb) != GAS_INVALID_BID) {
                        const gas_link_table_t *t = gas->table +
                                gas->R.tid[alpha_reverse.off + i];
                        cost = rdm_cost_add(cost, rdm_cost_mul(
                                rdm_cost_mul(t->nsrc, t->nlink), nb1));
                }
        }
        for (uint32_t i = 0; i < beta_reverse.n; i++) {
                gas_sid_t bsrc = gas->R.src[beta_reverse.off + i];
                if (gas_find_block(gas, bd->sa, bsrc) != GAS_INVALID_BID) {
                        const gas_link_table_t *t = gas->table +
                                gas->R.tid[beta_reverse.off + i];
                        cost = rdm_cost_add(cost, rdm_cost_mul(
                                rdm_cost_mul(t->nsrc, t->nlink), na1));
                }
        }

        /* Same-spin alpha paths through a common middle sector. */
        gas_row_t alpha_sources = gas->D.by_beta_row[bd->sb];
        const gas_sid_t *alpha_middle_reverse =
                gas->R.src + alpha_reverse.off;
        for (uint32_t q = 0; q < alpha_sources.n; q++) {
                gas_sid_t asrc = gas->D.by_beta_sid[alpha_sources.off + q];
                gas_row_t forward = gas->T.row[asrc];
                const gas_sid_t *middle = gas->T.dst + forward.off;
                uint32_t i = 0, j = 0;
                while (i < forward.n && j < alpha_reverse.n) {
                        if (middle[i] < alpha_middle_reverse[j]) {
                                i++;
                        } else if (middle[i] > alpha_middle_reverse[j]) {
                                j++;
                        } else {
                                const gas_link_table_t *t1 = gas->table + forward.off + i;
                                const gas_link_table_t *t2 = gas->table +
                                        gas->R.tid[alpha_reverse.off + j];
                                uint64_t w = rdm_cost_mul(t1->nsrc, t1->nlink);
                                w = rdm_cost_mul(w, t2->nlink);
                                w = rdm_cost_mul(w, nb1);
                                cost = rdm_cost_add(cost, w);
                                i++;
                                j++;
                        }
                }
        }

        /* Same-spin beta paths through a common middle sector. */
        gas_row_t beta_sources = gas->D.by_alpha_row[bd->sa];
        const gas_sid_t *beta_middle_reverse = gas->R.src + beta_reverse.off;
        for (uint32_t q = 0; q < beta_sources.n; q++) {
                gas_bid_t bsrc = beta_sources.off + q;
                gas_sid_t bsrc_sid = gas->block[bsrc].sb;
                gas_row_t forward = gas->T.row[bsrc_sid];
                const gas_sid_t *middle = gas->T.dst + forward.off;
                uint32_t i = 0, j = 0;
                while (i < forward.n && j < beta_reverse.n) {
                        if (middle[i] < beta_middle_reverse[j]) {
                                i++;
                        } else if (middle[i] > beta_middle_reverse[j]) {
                                j++;
                        } else {
                                const gas_link_table_t *t1 = gas->table + forward.off + i;
                                const gas_link_table_t *t2 = gas->table +
                                        gas->R.tid[beta_reverse.off + j];
                                uint64_t w = rdm_cost_mul(t1->nsrc, t1->nlink);
                                w = rdm_cost_mul(w, t2->nlink);
                                w = rdm_cost_mul(w, na1);
                                cost = rdm_cost_add(cost, w);
                                i++;
                                j++;
                        }
                }
        }

        /* Opposite-spin direct alpha/beta link products. */
        for (uint32_t i = 0; i < alpha_reverse.n; i++) {
                gas_sid_t asrc = gas->R.src[alpha_reverse.off + i];
                const gas_link_table_t *ta = gas->table +
                        gas->R.tid[alpha_reverse.off + i];
                gas_row_t source_blocks = gas->D.by_alpha_row[asrc];
                for (uint32_t q = 0; q < source_blocks.n; q++) {
                        gas_sid_t bsrc = gas->block[source_blocks.off + q].sb;
                        gas_tid_t tidb = gas_find_table(gas, bsrc, bd->sb);
                        if (tidb != GAS_INVALID_TID) {
                                const gas_link_table_t *tb = gas->table + tidb;
                                uint64_t wa = rdm_cost_mul(ta->nsrc, ta->nlink);
                                uint64_t wb = rdm_cost_mul(tb->nsrc, tb->nlink);
                                cost = rdm_cost_add(cost, rdm_cost_mul(wa, wb));
                        }
                }
        }
        return cost ? cost : 1u;
}

static int rdm_task_compare(const void *pa, const void *pb)
{
        const rdm_task_t *a = pa;
        const rdm_task_t *b = pb;
        if (a->cost != b->cost) return a->cost < b->cost ? 1 : -1;
        if (a->adst != b->adst) return (a->adst > b->adst) - (a->adst < b->adst);
        if (a->q0 != b->q0) return (a->q0 > b->q0) - (a->q0 < b->q0);
        return (a->a0 > b->a0) - (a->a0 < b->a0);
}

static int rdm_append_task(rdm_task_t **task, uint32_t *n, uint32_t *capacity,
                           gas_sid_t adst, uint32_t q0, uint32_t q1,
                           uint32_t a0, uint32_t a1, uint64_t cost)
{
        if (*n == *capacity) {
                uint32_t next = *capacity ? 2u * *capacity : 256u;
                if (next < *capacity) return GAS_ERR_MEMORY;
                rdm_task_t *p = realloc(*task, (size_t)next * sizeof(*p));
                if (p == 0) return GAS_ERR_MEMORY;
                *task = p;
                *capacity = next;
        }
        (*task)[*n].cost = cost ? cost : 1u;
        (*task)[*n].adst = adst;
        (*task)[*n].q0 = q0;
        (*task)[*n].q1 = q1;
        (*task)[*n].a0 = a0;
        (*task)[*n].a1 = a1;
        (*n)++;
        return GAS_SUCCESS;
}

static uint32_t rdm_max_threads(void)
{
#ifdef _OPENMP
        int n = omp_get_max_threads();
        return n > 0 ? (uint32_t)n : 1u;
#else
        return 1u;
#endif
}

static rdm_task_t *rdm_build_tasks(const gas_space_t *gas, uint32_t nthread,
                                   uint32_t *ntask)
{
        rdm_task_t *task = 0;
        uint32_t n = 0, capacity = 0;
        uint64_t total = 0;
        uint64_t *block_cost = malloc((size_t)gas->nblock * sizeof(*block_cost));
        if (block_cost == 0) return 0;

        for (gas_bid_t b = 0; b < gas->nblock; b++) {
                block_cost[b] = rdm_destination_block_cost(gas, b);
                total = rdm_cost_add(total, block_cost[b]);
        }
        uint64_t target = total /
                ((uint64_t)(nthread ? nthread : 1u) * RDM_TASK_FACTOR);
        if (target == 0u) target = 1u;

        for (gas_sid_t s = 0; s < gas->nsector; s++) {
                gas_row_t row = gas->D.by_alpha_row[s];
                uint32_t na = gas->sector_nstr[s];
                if (row.n == 0u || na == 0u) continue;
                uint32_t q0 = 0;
                uint64_t chunk = 0;

                for (uint32_t q = 0; q < row.n; q++) {
                        uint64_t w = block_cost[row.off + q];
                        uint64_t want64 = (w + target - 1u) / target;
                        uint32_t by_rows = (na + RDM_MIN_ALPHA_TILE - 1u) /
                                           RDM_MIN_ALPHA_TILE;
                        uint32_t split = want64 > UINT32_MAX ?
                                         UINT32_MAX : (uint32_t)want64;
                        if (split > RDM_MAX_ALPHA_SPLIT) {
                                split = RDM_MAX_ALPHA_SPLIT;
                        }
                        if (split > by_rows) split = by_rows;

                        if (split > 1u) {
                                if (q > q0 && rdm_append_task(&task, &n, &capacity,
                                        s, q0, q, 0u, na, chunk) != GAS_SUCCESS) {
                                        free(block_cost);
                                        free(task);
                                        return 0;
                                }
                                uint32_t a0 = 0;
                                for (uint32_t part = 0; part < split; part++) {
                                        uint32_t left = na - a0;
                                        uint32_t remain = split - part;
                                        uint32_t width = (left + remain - 1u) / remain;
                                        uint32_t a1 = a0 + width;
                                        uint64_t part_cost = rdm_cost_mul(w, width) / na;
                                        if (rdm_append_task(&task, &n, &capacity,
                                                s, q, q + 1u, a0, a1,
                                                part_cost) != GAS_SUCCESS) {
                                                free(block_cost);
                                                free(task);
                                                return 0;
                                        }
                                        a0 = a1;
                                }
                                q0 = q + 1u;
                                chunk = 0;
                                continue;
                        }
                        if (q > q0 && chunk != 0u &&
                            rdm_cost_add(chunk, w) > target) {
                                if (rdm_append_task(&task, &n, &capacity,
                                        s, q0, q, 0u, na, chunk) != GAS_SUCCESS) {
                                        free(block_cost);
                                        free(task);
                                        return 0;
                                }
                                q0 = q;
                                chunk = 0;
                        }
                        chunk = rdm_cost_add(chunk, w);
                }
                if (q0 < row.n && rdm_append_task(&task, &n, &capacity,
                        s, q0, row.n, 0u, na, chunk) != GAS_SUCCESS) {
                        free(block_cost);
                        free(task);
                        return 0;
                }
        }
        free(block_cost);
        qsort(task, n, sizeof(*task), rdm_task_compare);
        *ntask = n;
        return task;
}

/* Convert <E_ps E_rq> to <p^+ r^+ s q> for one spin species. */
static void reorder_same_spin(double *dm2, const double *dm1, uint32_t norb)
{
        for (uint32_t p = 0; p < norb; p++) {
                for (uint32_t r = 0; r < norb; r++) {
                        for (uint32_t q = 0; q < norb; q++) {
                                size_t d = rdm2_index(p, q, r, q, norb);
                                double old = dm2[d];
                                double delta = q == r ? dm1[(size_t)q * norb + p] : 0.0;
                                dm2[d] = delta - old;
                                for (uint32_t s = q + 1u; s < norb; s++) {
                                        size_t i = rdm2_index(p, q, r, s, norb);
                                        size_t j = rdm2_index(p, s, r, q, norb);
                                        double x = dm2[i];
                                        double y = dm2[j];
                                        double di = s == r ?
                                                dm1[(size_t)q * norb + p] : 0.0;
                                        double dj = q == r ?
                                                dm1[(size_t)s * norb + p] : 0.0;
                                        dm2[i] = di - y;
                                        dm2[j] = dj - x;
                                }
                        }
                }
        }
}

static int rdm_output_validate(const gas_space_t *gas,
                               const double *bra, const double *ket,
                               const double *dm1a, const double *dm1b,
                               const double *dm2aa, const double *dm2ab,
                               const double *dm2bb)
{
        if (rdm_validate(gas, bra, ket) != GAS_SUCCESS ||
            dm1a == 0 || dm1b == 0 || dm1a == dm1b ||
            dm1a == bra || dm1a == ket || dm1b == bra || dm1b == ket) {
                return GAS_ERR_INVALID;
        }
        if (dm2aa != 0 || dm2ab != 0 || dm2bb != 0) {
                if (dm2aa == 0 || dm2ab == 0 || dm2bb == 0 ||
                    dm2aa == dm2ab || dm2aa == dm2bb || dm2ab == dm2bb) {
                        return GAS_ERR_INVALID;
                }
                if (dm2aa == bra || dm2aa == ket ||
                    dm2ab == bra || dm2ab == ket ||
                    dm2bb == bra || dm2bb == ket ||
                    dm1a == dm2aa || dm1a == dm2ab || dm1a == dm2bb ||
                    dm1b == dm2aa || dm1b == dm2ab || dm1b == dm2bb) {
                        return GAS_ERR_INVALID;
                }
        }
        return GAS_SUCCESS;
}

static int rdm_workspace_reserve(rdm_workspace_t *workspace, size_t need)
{
        if (need <= workspace->capacity) return GAS_SUCCESS;
        if (need > SIZE_MAX / sizeof(*workspace->data)) return GAS_ERR_MEMORY;
        double *p = realloc(workspace->data, need * sizeof(*p));
        if (p == 0) return GAS_ERR_MEMORY;
        workspace->data = p;
        workspace->capacity = need;
        return GAS_SUCCESS;
}

static int rdm_arena_reserve(rdm_workspace_t *workspace, size_t need)
{
        if (need <= workspace->arena_capacity) return GAS_SUCCESS;
        if (need > SIZE_MAX / sizeof(*workspace->arena)) return GAS_ERR_MEMORY;
        double *p = realloc(workspace->arena, need * sizeof(*p));
        if (p == 0) return GAS_ERR_MEMORY;
        workspace->arena = p;
        workspace->arena_capacity = need;
        return GAS_SUCCESS;
}

static int rdm_pair_reserve(rdm_workspace_t *workspace, size_t need)
{
        if (need <= workspace->pair_capacity) return GAS_SUCCESS;
        if (need > SIZE_MAX / sizeof(*workspace->pair)) return GAS_ERR_MEMORY;
        rdm_tid_pair_t *p = realloc(workspace->pair, need * sizeof(*p));
        if (p == 0) return GAS_ERR_MEMORY;
        workspace->pair = p;
        workspace->pair_capacity = need;
        return GAS_SUCCESS;
}

static int rdm_hit_reserve(rdm_workspace_t *workspace, size_t need)
{
        if (need <= workspace->alpha_hit_capacity) return GAS_SUCCESS;
        if (need > SIZE_MAX / sizeof(*workspace->alpha_hit)) return GAS_ERR_MEMORY;
        rdm_alpha_hit_t *p = realloc(workspace->alpha_hit, need * sizeof(*p));
        if (p == 0) return GAS_ERR_MEMORY;
        workspace->alpha_hit = p;
        workspace->alpha_hit_capacity = need;
        return GAS_SUCCESS;
}

static int rdm_op_map_reserve(rdm_workspace_t *workspace, size_t need)
{
        if (need <= workspace->op_map_capacity) return GAS_SUCCESS;
        if (need > SIZE_MAX / sizeof(*workspace->op_map)) return GAS_ERR_MEMORY;
        int32_t *p = realloc(workspace->op_map, need * sizeof(*p));
        if (p == 0) return GAS_ERR_MEMORY;
        workspace->op_map = p;
        workspace->op_map_capacity = need;
        return GAS_SUCCESS;
}

static int rdm_plan_resize_workspaces(gas_rdm_plan_t *plan,
                                      uint32_t nthread, size_t size)
{
        if (nthread > plan->nworkspace) {
                rdm_workspace_t *p = realloc(
                        plan->workspace, (size_t)nthread * sizeof(*p));
                if (p == 0) return GAS_ERR_MEMORY;
                plan->workspace = p;
                for (uint32_t i = plan->nworkspace; i < nthread; i++) {
                        memset(plan->workspace + i, 0, sizeof(plan->workspace[i]));
                }
                plan->nworkspace = nthread;
        }
        for (uint32_t i = 0; i < nthread; i++) {
                if (rdm_workspace_reserve(plan->workspace + i, size) != GAS_SUCCESS) {
                        return GAS_ERR_MEMORY;
                }
        }
        return GAS_SUCCESS;
}

static int rdm_plan_build_active_ops(gas_rdm_plan_t *plan)
{
        const gas_space_t *gas = plan->gas;
        uint32_t nopair = (uint32_t)gas->norb_tot * gas->norb_tot;
        uint8_t *seen = calloc(nopair, sizeof(*seen));
        rdm_table_ops_t *meta = calloc(gas->ntable, sizeof(*meta));
        if ((seen == 0 && nopair != 0u) || (meta == 0 && gas->ntable != 0u)) {
                free(seen);
                free(meta);
                return GAS_ERR_MEMORY;
        }

        uint64_t total = 0u;
        for (gas_tid_t tid = 0; tid < gas->ntable; tid++) {
                const gas_link_table_t *table = gas->table + tid;
                if (nopair != 0u) {
                        memset(seen, 0, nopair * sizeof(*seen));
                }
                uint32_t count = 0u;
                size_t nentry = (size_t)table->nsrc * table->nlink;
                for (size_t i = 0; i < nentry; i++) {
                        uint32_t op = rdm_op(table->link + i,
                                             (uint32_t)gas->norb_tot);
                        if (!seen[op]) {
                                seen[op] = 1u;
                                count++;
                        }
                }
                if (count > UINT16_MAX || total + count > UINT32_MAX) {
                        free(seen);
                        free(meta);
                        return GAS_ERR_INVALID;
                }
                meta[tid].off = (uint32_t)total;
                meta[tid].n = (uint16_t)count;
                total += count;
        }

        if (total == 0u) {
                free(seen);
                plan->table_ops = meta;
                return GAS_SUCCESS;
        }
        uint16_t *active = malloc((size_t)total * sizeof(*active));
        uint32_t *active_row = malloc((size_t)total * sizeof(*active_row));
        if (active == 0 || active_row == 0) {
                free(seen);
                free(meta);
                free(active);
                free(active_row);
                return GAS_ERR_MEMORY;
        }
        for (gas_tid_t tid = 0; tid < gas->ntable; tid++) {
                const gas_link_table_t *table = gas->table + tid;
                if (nopair != 0u) {
                        memset(seen, 0, nopair * sizeof(*seen));
                }
                size_t nentry = (size_t)table->nsrc * table->nlink;
                for (size_t i = 0; i < nentry; i++) {
                        seen[rdm_op(table->link + i,
                                    (uint32_t)gas->norb_tot)] = 1u;
                }
                uint32_t out = meta[tid].off;
                for (uint32_t op = 0; op < nopair; op++) {
                        if (seen[op]) {
                                active[out] = (uint16_t)op;
                                active_row[out] =
                                        (uint32_t)((uint64_t)op * nopair);
                                out++;
                        }
                }
        }
        free(seen);
        plan->table_ops = meta;
        plan->active_op = active;
        plan->active_row = active_row;
        return GAS_SUCCESS;
}

static int rdm_plan_rebuild_tasks(gas_rdm_plan_t *plan)
{
        uint32_t nthread = rdm_max_threads();
        if (plan->task != 0 && plan->task_threads == nthread) {
                return GAS_SUCCESS;
        }
        uint32_t ntask = 0;
        rdm_task_t *task = rdm_build_tasks(plan->gas, nthread, &ntask);
        if (task == 0 && plan->gas->nblock != 0u) return GAS_ERR_MEMORY;
        free(plan->task);
        plan->task = task;
        plan->ntask = ntask;
        plan->task_threads = nthread;
        return GAS_SUCCESS;
}

static void rdm1_serial(const gas_space_t *gas,
                        const double *bra, const double *ket,
                        double *dm1a, double *dm1b)
{
        size_t n1 = (size_t)gas->norb_tot * gas->norb_tot;
        memset(dm1a, 0, n1 * sizeof(*dm1a));
        memset(dm1b, 0, n1 * sizeof(*dm1b));
        rdm1_alpha(gas, bra, ket, dm1a);
        rdm1_beta(gas, bra, ket, dm1b);
}

static void rdm12_serial(const gas_space_t *gas,
                         const double *bra, const double *ket,
                         double *dm1a, double *dm1b,
                         double *dm2aa, double *dm2ab, double *dm2bb)
{
        uint32_t norb = (uint32_t)gas->norb_tot;
        size_t n1 = (size_t)norb * norb;
        size_t n2 = n1 * n1;
        memset(dm1a, 0, n1 * sizeof(*dm1a));
        memset(dm1b, 0, n1 * sizeof(*dm1b));
        memset(dm2aa, 0, n2 * sizeof(*dm2aa));
        memset(dm2ab, 0, n2 * sizeof(*dm2ab));
        memset(dm2bb, 0, n2 * sizeof(*dm2bb));
        rdm1_alpha(gas, bra, ket, dm1a);
        rdm1_beta(gas, bra, ket, dm1b);
        rdm2_aa(gas, bra, ket, dm2aa);
        rdm2_ab(gas, bra, ket, dm2ab);
        rdm2_bb(gas, bra, ket, dm2bb);
        reorder_same_spin(dm2aa, dm1a, norb);
        reorder_same_spin(dm2bb, dm1b, norb);
}

static int rdm12_tasks_serial(gas_rdm_plan_t *plan,
                              const double *bra, const double *ket,
                              double *dm1a, double *dm1b,
                              double *dm2aa, double *dm2ab, double *dm2bb,
                              size_t n1, size_t n2)
{
        size_t local_size = 2u * n1 + 3u * n2;
        if (rdm_plan_resize_workspaces(plan, 1u, local_size) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        rdm_workspace_t *workspace = plan->workspace;
        memset(workspace->data, 0, local_size * sizeof(*workspace->data));
        workspace->alpha_hit_n = 0u;
        for (uint32_t tt = 0; tt < plan->ntask; tt++) {
                rdm12_task_accumulate(plan, bra, ket, plan->task + tt,
                                      workspace, n1, n2);
        }
        const double *local = workspace->data;
        memcpy(dm1a, local, n1 * sizeof(*dm1a));
        memcpy(dm1b, local + n1, n1 * sizeof(*dm1b));
        local += 2u * n1;
        memcpy(dm2aa, local, n2 * sizeof(*dm2aa));
        memcpy(dm2ab, local + n2, n2 * sizeof(*dm2ab));
        memcpy(dm2bb, local + 2u * n2, n2 * sizeof(*dm2bb));
        uint32_t norb = (uint32_t)plan->gas->norb_tot;
        reorder_same_spin(dm2aa, dm1a, norb);
        reorder_same_spin(dm2bb, dm1b, norb);
        return GAS_SUCCESS;
}

#ifdef _OPENMP
static void rdm1_parallel(gas_rdm_plan_t *plan,
                          const double *bra, const double *ket,
                          double *dm1a, double *dm1b, size_t n1)
{
        uint32_t active_threads = 1u;

#pragma omp parallel shared(active_threads)
        {
                uint32_t tid = (uint32_t)omp_get_thread_num();
                double *local = plan->workspace[tid].data;
#pragma omp single
                active_threads = (uint32_t)omp_get_num_threads();
                memset(local, 0, 2u * n1 * sizeof(*local));
#pragma omp barrier
#pragma omp for schedule(dynamic, 1)
                for (int64_t tt = 0; tt < (int64_t)plan->ntask; tt++) {
                        rdm1_alpha_task(plan->gas, bra, ket, plan->task + tt, local);
                        rdm1_beta_task(plan->gas, bra, ket, plan->task + tt,
                                      local + n1);
                }
#pragma omp for schedule(static)
                for (int64_t i = 0; i < (int64_t)n1; i++) {
                        double a = 0.0, b = 0.0;
                        for (uint32_t t = 0; t < active_threads; t++) {
                                const double *w = plan->workspace[t].data;
                                a += w[i];
                                b += w[n1 + (size_t)i];
                        }
                        dm1a[i] = a;
                        dm1b[i] = b;
                }
        }
}

static void rdm12_parallel(gas_rdm_plan_t *plan,
                           const double *bra, const double *ket,
                           double *dm1a, double *dm1b,
                           double *dm2aa, double *dm2ab, double *dm2bb,
                           size_t n1, size_t n2)
{
        uint32_t active_threads = 1u;
        size_t local_size = 2u * n1 + 3u * n2;

#pragma omp parallel shared(active_threads)
        {
                uint32_t tid = (uint32_t)omp_get_thread_num();
                rdm_workspace_t *workspace = plan->workspace + tid;
                double *local = workspace->data;
#pragma omp single
                active_threads = (uint32_t)omp_get_num_threads();
                memset(local, 0, local_size * sizeof(*local));
                workspace->alpha_hit_n = 0u;
#pragma omp barrier
#pragma omp for schedule(dynamic, 1)
                for (int64_t tt = 0; tt < (int64_t)plan->ntask; tt++) {
                        rdm12_task_accumulate(plan, bra, ket,
                                              plan->task + tt, workspace,
                                              n1, n2);
                }
#pragma omp for schedule(static)
                for (int64_t i = 0; i < (int64_t)n1; i++) {
                        double a = 0.0, b = 0.0;
                        for (uint32_t t = 0; t < active_threads; t++) {
                                const double *w = plan->workspace[t].data;
                                a += w[i];
                                b += w[n1 + (size_t)i];
                        }
                        dm1a[i] = a;
                        dm1b[i] = b;
                }
#pragma omp for schedule(static)
                for (int64_t i = 0; i < (int64_t)n2; i++) {
                        double aa = 0.0, ab = 0.0, bb = 0.0;
                        for (uint32_t t = 0; t < active_threads; t++) {
                                const double *w = plan->workspace[t].data + 2u * n1;
                                aa += w[i];
                                ab += w[n2 + (size_t)i];
                                bb += w[2u * n2 + (size_t)i];
                        }
                        dm2aa[i] = aa;
                        dm2ab[i] = ab;
                        dm2bb[i] = bb;
                }
        }
        uint32_t norb = (uint32_t)plan->gas->norb_tot;
        reorder_same_spin(dm2aa, dm1a, norb);
        reorder_same_spin(dm2bb, dm1b, norb);
}
#endif

int fci_rdm_gas_plan_create(gas_rdm_plan_t **out, const gas_space_t *gas)
{
        if (out == 0 || gas == 0 || gas->link_format != GAS_LINK_RAW ||
            gas->norb_tot <= 0) return GAS_ERR_INVALID;
        *out = 0;
        gas_rdm_plan_t *plan = calloc(1, sizeof(*plan));
        if (plan == 0) return GAS_ERR_MEMORY;
        plan->gas = gas;
        plan->abba_t1_target_bytes = rdm_abba_t1_target_bytes;
        int status = rdm_plan_build_active_ops(plan);
        if (status != GAS_SUCCESS) {
                fci_rdm_gas_plan_free(plan);
                return status;
        }
        if (rdm_plan_rebuild_tasks(plan) != GAS_SUCCESS) {
                fci_rdm_gas_plan_free(plan);
                return GAS_ERR_MEMORY;
        }
        *out = plan;
        return GAS_SUCCESS;
}

int fci_rdm_gas_plan_make_rdm1s(gas_rdm_plan_t *plan,
                                const double *bra, const double *ket,
                                double *dm1a, double *dm1b)
{
        if (plan == 0 || rdm_output_validate(plan->gas, bra, ket,
                dm1a, dm1b, 0, 0, 0) != GAS_SUCCESS) return GAS_ERR_INVALID;
        if (rdm_plan_rebuild_tasks(plan) != GAS_SUCCESS) return GAS_ERR_MEMORY;
#ifdef _OPENMP
        size_t n1 = (size_t)plan->gas->norb_tot * plan->gas->norb_tot;
        uint32_t nthread = rdm_max_threads();
        if (nthread > 1u && plan->ntask > 1u) {
                if (rdm_plan_resize_workspaces(plan, nthread, 2u * n1) != GAS_SUCCESS) {
                        return GAS_ERR_MEMORY;
                }
                rdm1_parallel(plan, bra, ket, dm1a, dm1b, n1);
                return GAS_SUCCESS;
        }
#endif
        rdm1_serial(plan->gas, bra, ket, dm1a, dm1b);
        return GAS_SUCCESS;
}

int fci_rdm_gas_plan_make_rdm12s(gas_rdm_plan_t *plan,
                                 const double *bra, const double *ket,
                                 double *dm1a, double *dm1b,
                                 double *dm2aa, double *dm2ab, double *dm2bb)
{
        if (plan == 0 || rdm_output_validate(plan->gas, bra, ket,
                dm1a, dm1b, dm2aa, dm2ab, dm2bb) != GAS_SUCCESS) {
                return GAS_ERR_INVALID;
        }
        if (plan->abba_t1_target_bytes != rdm_abba_t1_target_bytes) {
                return GAS_ERR_INVALID;
        }
        if (rdm_plan_rebuild_tasks(plan) != GAS_SUCCESS) return GAS_ERR_MEMORY;
        uint32_t norb = (uint32_t)plan->gas->norb_tot;
        size_t n1 = (size_t)norb * norb;
        size_t n2 = n1 * n1;
        if (plan->ntask == 0u) {
                rdm12_serial(plan->gas, bra, ket, dm1a, dm1b,
                             dm2aa, dm2ab, dm2bb);
                return GAS_SUCCESS;
        }
#ifdef _OPENMP
        uint32_t nthread = rdm_max_threads();
        if (nthread > 1u && plan->ntask > 1u) {
                size_t local_size = 2u * n1 + 3u * n2;
                if (rdm_plan_resize_workspaces(plan, nthread,
                                               local_size) != GAS_SUCCESS) {
                        return GAS_ERR_MEMORY;
                }
                rdm12_parallel(plan, bra, ket, dm1a, dm1b,
                               dm2aa, dm2ab, dm2bb, n1, n2);
                return GAS_SUCCESS;
        }
#endif
        return rdm12_tasks_serial(plan, bra, ket, dm1a, dm1b,
                                  dm2aa, dm2ab, dm2bb, n1, n2);
}

void fci_rdm_gas_plan_free(gas_rdm_plan_t *plan)
{
        if (plan == 0) return;
        for (uint32_t i = 0; i < plan->nworkspace; i++) {
                free(plan->workspace[i].data);
                free(plan->workspace[i].arena);
                free(plan->workspace[i].pair);
                free(plan->workspace[i].alpha_hit);
                free(plan->workspace[i].op_map);
        }
        free(plan->workspace);
        free(plan->task);
        free(plan->table_ops);
        free(plan->active_op);
        free(plan->active_row);
        free(plan);
}

uint32_t fci_rdm_gas_plan_task_count(const gas_rdm_plan_t *plan)
{
        return plan ? plan->ntask : 0u;
}

uint64_t fci_rdm_gas_plan_workspace_bytes(const gas_rdm_plan_t *plan)
{
        uint64_t bytes = 0;
        if (plan == 0) return 0u;
        for (uint32_t i = 0; i < plan->nworkspace; i++) {
                const rdm_workspace_t *w = plan->workspace + i;
                bytes = rdm_cost_add(bytes, rdm_cost_mul(
                        w->capacity, sizeof(*w->data)));
                bytes = rdm_cost_add(bytes, rdm_cost_mul(
                        w->arena_capacity, sizeof(*w->arena)));
                bytes = rdm_cost_add(bytes, rdm_cost_mul(
                        w->pair_capacity, sizeof(*w->pair)));
                bytes = rdm_cost_add(bytes, rdm_cost_mul(
                        w->alpha_hit_capacity, sizeof(*w->alpha_hit)));
                bytes = rdm_cost_add(bytes, rdm_cost_mul(
                        w->op_map_capacity, sizeof(*w->op_map)));
        }
        return bytes;
}
