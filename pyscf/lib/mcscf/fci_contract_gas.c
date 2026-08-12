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

/* ========================================================================== */
/* 1. Configuration and low-level utilities                                   */
/* ========================================================================== */

enum {
        BB_ALPHA_BLOCK = 64,
        BB_TRANSPOSE_MIN_DIM = 4,
        BB_TRANSPOSE_TILE = 32,
        ABBA_WRITEBACK_TILE = 32,
        OMP_TASK_FACTOR = 32,
        OMP_MAX_ALPHA_SPLIT = 64,
        OMP_MIN_ALPHA_TILE = 4
};

static const size_t BB_TRANSPOSE_MAX_ELEMS = 67108864u;
static const size_t ABBA_WORKSPACE_MAX_ELEMS = 67108864u;
static size_t gas_abba_t1_target_bytes = 64u * 1024u * 1024u;

void fci_contract_gas_set_abba_t1_bytes(uint64_t bytes)
{
        gas_abba_t1_target_bytes = (size_t)bytes;
}

static uint32_t abba_beta_tile_size(uint32_t nb0, uint32_t nop, uint32_t na1)
{
        if (nb0 == 0u || nop == 0u || na1 == 0u ||
            gas_abba_t1_target_bytes == 0u) {
                return nb0;
        }

        size_t denom = (size_t)nop * (size_t)na1 * sizeof(double);
        size_t tile = denom == 0u ? nb0 : gas_abba_t1_target_bytes / denom;
        if (tile < 1u) {
                tile = 1u;
        }
        if (tile > nb0) {
                tile = nb0;
        }
        return (uint32_t)tile;
}

static inline double abba_gvalue(const double *restrict g,
                                 uint32_t nnorb, uint16_t opb, uint16_t opa)
{
        return g[(uint64_t)opb * nnorb + opa];
}

/* Consecutive beta link tables b0->b1 and b1->b2. */
typedef struct {
        gas_tid_t first;
        gas_tid_t second;
} gas_tid_pair_t;

/*
 * Alpha link landing in the current destination tile [a0,a1).
 * rel_addr is relative to a0; op_index selects the active alpha operator.
 */
typedef struct {
        uint32_t src_row;
        uint32_t rel_addr;
        uint16_t op_index;
        int8_t sign;
        uint8_t pad;
} gas_alpha_hit_t;

typedef struct {
        double *arena;
        size_t arena_cap;
        gas_tid_pair_t *pair;
        size_t pair_cap;
        int32_t *op_map;
        size_t op_map_cap;
        uint16_t *op;
        size_t op_cap;

        size_t arena_zeroed_upto;

        /* A hit is an alpha link whose destination address lies inside the
         * current alpha tile [a0,a1).
         * Building the list once per ABBA batch lets each beta tile reuse it
         * instead of rescanning mostly-missing alpha links. */
        gas_alpha_hit_t *alpha_hit;
        size_t alpha_hit_cap;
        uint32_t alpha_hit_n;
} gas_contract_ws_t;

typedef enum {
        GAS_BATCH_DONE = 0,
        GAS_BATCH_FALLBACK = 1
} gas_batch_status_t;

#ifdef _OPENMP
typedef struct {
        uint64_t cost;
        gas_sid_t adst;
        uint32_t q0;
        uint32_t q1;
        uint32_t a0;
        uint32_t a1;
} gas_omp_task_t;

static int cmp_omp_task_desc(const void *pa, const void *pb)
{
        const gas_omp_task_t *a = (const gas_omp_task_t *)pa;
        const gas_omp_task_t *b = (const gas_omp_task_t *)pb;
        if (a->cost != b->cost) {
                return a->cost < b->cost ? 1 : -1;
        }
        return (a->adst > b->adst) - (a->adst < b->adst);
}
#endif

struct gas_contract_plan {
        const gas_space_t *gas;
        const double *eri;
        const double *gos;
        size_t abba_t1_target_bytes;
        gas_contract_ws_t *workspace;
        uint32_t nworkspace;
#ifdef _OPENMP
        gas_omp_task_t *task;
        uint32_t ntask;
        uint32_t task_threads;
#endif
};

static inline void contract_zero(double *x, uint32_t n)
{
        memset(x, 0, (size_t)n * sizeof(*x));
}

static inline void contract_axpy(double *restrict y,
                                 const double *restrict x,
                                 double a, uint32_t n)
{
#ifdef _OPENMP
#pragma omp simd
#endif
        for (uint32_t i = 0; i < n; i++) {
                y[i] += a * x[i];
        }
}

/* One AA source row updates several distinct destination rows.  Loading x once
 * for four destinations reduces the dominant source-row traffic without
 * changing the arithmetic order within any destination row. */
static inline void contract_axpy4(double *restrict y0,
                                  double *restrict y1,
                                  double *restrict y2,
                                  double *restrict y3,
                                  const double *restrict x,
                                  double a0, double a1,
                                  double a2, double a3,
                                  uint32_t n)
{
        uint32_t i = 0;
#if defined(__AVX2__)
        const __m256d va0 = _mm256_set1_pd(a0);
        const __m256d va1 = _mm256_set1_pd(a1);
        const __m256d va2 = _mm256_set1_pd(a2);
        const __m256d va3 = _mm256_set1_pd(a3);
        for (; i + 4u <= n; i += 4u) {
                const __m256d vx = _mm256_loadu_pd(x + i);
                __m256d vy0 = _mm256_loadu_pd(y0 + i);
                __m256d vy1 = _mm256_loadu_pd(y1 + i);
                __m256d vy2 = _mm256_loadu_pd(y2 + i);
                __m256d vy3 = _mm256_loadu_pd(y3 + i);
                vy0 = _mm256_add_pd(vy0, _mm256_mul_pd(va0, vx));
                vy1 = _mm256_add_pd(vy1, _mm256_mul_pd(va1, vx));
                vy2 = _mm256_add_pd(vy2, _mm256_mul_pd(va2, vx));
                vy3 = _mm256_add_pd(vy3, _mm256_mul_pd(va3, vx));
                _mm256_storeu_pd(y0 + i, vy0);
                _mm256_storeu_pd(y1 + i, vy1);
                _mm256_storeu_pd(y2 + i, vy2);
                _mm256_storeu_pd(y3 + i, vy3);
        }
#endif
        for (; i < n; i++) {
                const double xi = x[i];
                y0[i] += a0 * xi;
                y1[i] += a1 * xi;
                y2[i] += a2 * xi;
                y3[i] += a3 * xi;
        }
}

static inline void contract_axpy2(double *restrict y0,
                                  double *restrict y1,
                                  const double *restrict x,
                                  double a0, double a1,
                                  uint32_t n)
{
        uint32_t i = 0;
#if defined(__AVX2__)
        const __m256d va0 = _mm256_set1_pd(a0);
        const __m256d va1 = _mm256_set1_pd(a1);
        for (; i + 4u <= n; i += 4u) {
                const __m256d vx = _mm256_loadu_pd(x + i);
                __m256d vy0 = _mm256_loadu_pd(y0 + i);
                __m256d vy1 = _mm256_loadu_pd(y1 + i);
                vy0 = _mm256_add_pd(vy0, _mm256_mul_pd(va0, vx));
                vy1 = _mm256_add_pd(vy1, _mm256_mul_pd(va1, vx));
                _mm256_storeu_pd(y0 + i, vy0);
                _mm256_storeu_pd(y1 + i, vy1);
        }
#endif
        for (; i < n; i++) {
                const double xi = x[i];
                y0[i] += a0 * xi;
                y1[i] += a1 * xi;
        }
}

/* ========================================================================== */
/* 2. ABBA microkernels and workspace helpers                                 */
/* ========================================================================== */

static inline void abba_beta_kernel1(double *restrict y,
                                        const double *restrict t0,
                                        double g0, uint32_t n)
{
        for (uint32_t i = 0; i < n; i++) {
                y[i] += g0 * t0[i];
        }
}

static inline void abba_beta_kernel2(double *restrict y,
                                        const double *restrict t0,
                                        const double *restrict t1,
                                        double g0, double g1,
                                        uint32_t n)
{
        for (uint32_t i = 0; i < n; i++) {
                y[i] += g0 * t0[i] + g1 * t1[i];
        }
}

static inline void abba_beta_kernel4_scalar(double *restrict y,
                                               const double *restrict t0,
                                               const double *restrict t1,
                                               const double *restrict t2,
                                               const double *restrict t3,
                                               double g0, double g1,
                                               double g2, double g3,
                                               uint32_t n)
{
        for (uint32_t i = 0; i < n; i++) {
                y[i] += g0 * t0[i] + g1 * t1[i] +
                        g2 * t2[i] + g3 * t3[i];
        }
}

#if defined(__AVX2__)
static inline void abba_beta_kernel4_avx2(double *restrict y,
                                             const double *restrict t0,
                                             const double *restrict t1,
                                             const double *restrict t2,
                                             const double *restrict t3,
                                             double g0, double g1,
                                             double g2, double g3,
                                             uint32_t n)
{
        const __m256d vg0 = _mm256_set1_pd(g0);
        const __m256d vg1 = _mm256_set1_pd(g1);
        const __m256d vg2 = _mm256_set1_pd(g2);
        const __m256d vg3 = _mm256_set1_pd(g3);
        uint32_t i = 0;
        for (; i + 4u <= n; i += 4u) {
                __m256d vy = _mm256_loadu_pd(y + i);
                __m256d v0 = _mm256_loadu_pd(t0 + i);
                __m256d v1 = _mm256_loadu_pd(t1 + i);
                __m256d v2 = _mm256_loadu_pd(t2 + i);
                __m256d v3 = _mm256_loadu_pd(t3 + i);
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg0, v0));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg1, v1));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg2, v2));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg3, v3));
                _mm256_storeu_pd(y + i, vy);
        }
        for (; i < n; i++) {
                y[i] += g0 * t0[i] + g1 * t1[i] +
                        g2 * t2[i] + g3 * t3[i];
        }
}
#endif

static inline void abba_beta_kernel4(double *restrict y,
                                        const double *restrict t0,
                                        const double *restrict t1,
                                        const double *restrict t2,
                                        const double *restrict t3,
                                        double g0, double g1,
                                        double g2, double g3,
                                        uint32_t n)
{
#if defined(__AVX2__)
        abba_beta_kernel4_avx2(y, t0, t1, t2, t3, g0, g1, g2, g3, n);
#else
        abba_beta_kernel4_scalar(y, t0, t1, t2, t3, g0, g1, g2, g3, n);
#endif
}

/* Update two non-aliasing destination rows from the same four T1 streams.
 * Keeping four operators leaves enough AVX2 registers for both coefficient
 * sets and both Y accumulators.  The multiply/add order within each Y row is
 * identical to the first/second half of the established width8 kernel. */
static inline void abba_beta_kernel2dst4(
        double *restrict y0, double *restrict y1,
        const double *restrict t0, const double *restrict t1,
        const double *restrict t2, const double *restrict t3,
        double g00, double g01, double g02, double g03,
        double g10, double g11, double g12, double g13,
        uint32_t n)
{
#if defined(__AVX2__)
        const __m256d vg00 = _mm256_set1_pd(g00);
        const __m256d vg01 = _mm256_set1_pd(g01);
        const __m256d vg02 = _mm256_set1_pd(g02);
        const __m256d vg03 = _mm256_set1_pd(g03);
        const __m256d vg10 = _mm256_set1_pd(g10);
        const __m256d vg11 = _mm256_set1_pd(g11);
        const __m256d vg12 = _mm256_set1_pd(g12);
        const __m256d vg13 = _mm256_set1_pd(g13);
        uint32_t i = 0u;
        for (; i + 4u <= n; i += 4u) {
                __m256d vy0 = _mm256_loadu_pd(y0 + i);
                __m256d vy1 = _mm256_loadu_pd(y1 + i);
                __m256d vt = _mm256_loadu_pd(t0 + i);
                vy0 = _mm256_add_pd(vy0, _mm256_mul_pd(vg00, vt));
                vy1 = _mm256_add_pd(vy1, _mm256_mul_pd(vg10, vt));
                vt = _mm256_loadu_pd(t1 + i);
                vy0 = _mm256_add_pd(vy0, _mm256_mul_pd(vg01, vt));
                vy1 = _mm256_add_pd(vy1, _mm256_mul_pd(vg11, vt));
                vt = _mm256_loadu_pd(t2 + i);
                vy0 = _mm256_add_pd(vy0, _mm256_mul_pd(vg02, vt));
                vy1 = _mm256_add_pd(vy1, _mm256_mul_pd(vg12, vt));
                vt = _mm256_loadu_pd(t3 + i);
                vy0 = _mm256_add_pd(vy0, _mm256_mul_pd(vg03, vt));
                vy1 = _mm256_add_pd(vy1, _mm256_mul_pd(vg13, vt));
                _mm256_storeu_pd(y0 + i, vy0);
                _mm256_storeu_pd(y1 + i, vy1);
        }
        for (; i < n; i++) {
                y0[i] += g00 * t0[i] + g01 * t1[i] +
                         g02 * t2[i] + g03 * t3[i];
                y1[i] += g10 * t0[i] + g11 * t1[i] +
                         g12 * t2[i] + g13 * t3[i];
        }
#else
        for (uint32_t i = 0; i < n; i++) {
                y0[i] += g00 * t0[i] + g01 * t1[i] +
                         g02 * t2[i] + g03 * t3[i];
                y1[i] += g10 * t0[i] + g11 * t1[i] +
                         g12 * t2[i] + g13 * t3[i];
        }
#endif
}

static inline void abba_beta_kernel8(double *restrict y,
                                        const double *restrict t0,
                                        const double *restrict t1,
                                        const double *restrict t2,
                                        const double *restrict t3,
                                        const double *restrict t4,
                                        const double *restrict t5,
                                        const double *restrict t6,
                                        const double *restrict t7,
                                        double g0, double g1, double g2, double g3,
                                        double g4, double g5, double g6, double g7,
                                        uint32_t n)
{
#if defined(__AVX2__)
        const __m256d vg0 = _mm256_set1_pd(g0);
        const __m256d vg1 = _mm256_set1_pd(g1);
        const __m256d vg2 = _mm256_set1_pd(g2);
        const __m256d vg3 = _mm256_set1_pd(g3);
        const __m256d vg4 = _mm256_set1_pd(g4);
        const __m256d vg5 = _mm256_set1_pd(g5);
        const __m256d vg6 = _mm256_set1_pd(g6);
        const __m256d vg7 = _mm256_set1_pd(g7);
        uint32_t i = 0;
        for (; i + 4u <= n; i += 4u) {
                __m256d vy = _mm256_loadu_pd(y + i);
                __m256d v0 = _mm256_loadu_pd(t0 + i);
                __m256d v1 = _mm256_loadu_pd(t1 + i);
                __m256d v2 = _mm256_loadu_pd(t2 + i);
                __m256d v3 = _mm256_loadu_pd(t3 + i);
                __m256d v4 = _mm256_loadu_pd(t4 + i);
                __m256d v5 = _mm256_loadu_pd(t5 + i);
                __m256d v6 = _mm256_loadu_pd(t6 + i);
                __m256d v7 = _mm256_loadu_pd(t7 + i);
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg0, v0));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg1, v1));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg2, v2));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg3, v3));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg4, v4));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg5, v5));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg6, v6));
                vy = _mm256_add_pd(vy, _mm256_mul_pd(vg7, v7));
                _mm256_storeu_pd(y + i, vy);
        }
        for (; i < n; i++) {
                y[i] += g0 * t0[i] + g1 * t1[i] +
                        g2 * t2[i] + g3 * t3[i] +
                        g4 * t4[i] + g5 * t5[i] +
                        g6 * t6[i] + g7 * t7[i];
        }
#else
        for (uint32_t i = 0; i < n; i++) {
                y[i] += g0 * t0[i] + g1 * t1[i] +
                        g2 * t2[i] + g3 * t3[i] +
                        g4 * t4[i] + g5 * t5[i] +
                        g6 * t6[i] + g7 * t7[i];
        }
#endif
}

static inline void abba_fused_ops(double *restrict y,
                                     const double *restrict tbase,
                                     size_t op_stride,
                                     const double *restrict gos,
                                     uint32_t nnorb, uint16_t opb,
                                     const uint16_t *restrict opa,
                                     uint32_t nop, double sbeta, uint32_t n)
{
        uint32_t io = 0;
        for (; io + 8u <= nop; io += 8u) {
                const double *restrict t0 = tbase + (size_t)(io + 0u) * op_stride;
                const double *restrict t1 = tbase + (size_t)(io + 1u) * op_stride;
                const double *restrict t2 = tbase + (size_t)(io + 2u) * op_stride;
                const double *restrict t3 = tbase + (size_t)(io + 3u) * op_stride;
                const double *restrict t4 = tbase + (size_t)(io + 4u) * op_stride;
                const double *restrict t5 = tbase + (size_t)(io + 5u) * op_stride;
                const double *restrict t6 = tbase + (size_t)(io + 6u) * op_stride;
                const double *restrict t7 = tbase + (size_t)(io + 7u) * op_stride;
                double g0 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 0u]);
                double g1 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 1u]);
                double g2 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 2u]);
                double g3 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 3u]);
                double g4 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 4u]);
                double g5 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 5u]);
                double g6 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 6u]);
                double g7 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 7u]);
                abba_beta_kernel8(y, t0, t1, t2, t3, t4, t5, t6, t7,
                                      g0, g1, g2, g3, g4, g5, g6, g7, n);
        }
        for (; io + 4u <= nop; io += 4u) {
                const double *restrict t0 = tbase + (size_t)(io + 0u) * op_stride;
                const double *restrict t1 = tbase + (size_t)(io + 1u) * op_stride;
                const double *restrict t2 = tbase + (size_t)(io + 2u) * op_stride;
                const double *restrict t3 = tbase + (size_t)(io + 3u) * op_stride;
                double g0 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 0u]);
                double g1 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 1u]);
                double g2 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 2u]);
                double g3 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 3u]);
                abba_beta_kernel4(y, t0, t1, t2, t3, g0, g1, g2, g3, n);
        }
        for (; io + 2u <= nop; io += 2u) {
                const double *restrict t0 = tbase + (size_t)(io + 0u) * op_stride;
                const double *restrict t1 = tbase + (size_t)(io + 1u) * op_stride;
                double g0 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 0u]);
                double g1 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io + 1u]);
                abba_beta_kernel2(y, t0, t1, g0, g1, n);
        }
        for (; io < nop; io++) {
                const double *restrict t0 = tbase + (size_t)io * op_stride;
                double g0 = sbeta * abba_gvalue(gos, nnorb, opb, opa[io]);
                abba_beta_kernel1(y, t0, g0, n);
        }
}

static inline void abba_fused_ops_pair2x4(
        double *restrict y0, double *restrict y1,
        const double *restrict tbase, size_t op_stride,
        const double *restrict gos, uint32_t nnorb,
        uint16_t opb0, uint16_t opb1,
        const uint16_t *restrict opa, uint32_t nop,
        double sbeta0, double sbeta1, uint32_t n)
{
        uint32_t io = 0u;
        for (; io + 4u <= nop; io += 4u) {
                const double *restrict t0 = tbase + (size_t)(io + 0u) * op_stride;
                const double *restrict t1 = tbase + (size_t)(io + 1u) * op_stride;
                const double *restrict t2 = tbase + (size_t)(io + 2u) * op_stride;
                const double *restrict t3 = tbase + (size_t)(io + 3u) * op_stride;
                double g00 = sbeta0 * abba_gvalue(gos, nnorb, opb0, opa[io + 0u]);
                double g01 = sbeta0 * abba_gvalue(gos, nnorb, opb0, opa[io + 1u]);
                double g02 = sbeta0 * abba_gvalue(gos, nnorb, opb0, opa[io + 2u]);
                double g03 = sbeta0 * abba_gvalue(gos, nnorb, opb0, opa[io + 3u]);
                double g10 = sbeta1 * abba_gvalue(gos, nnorb, opb1, opa[io + 0u]);
                double g11 = sbeta1 * abba_gvalue(gos, nnorb, opb1, opa[io + 1u]);
                double g12 = sbeta1 * abba_gvalue(gos, nnorb, opb1, opa[io + 2u]);
                double g13 = sbeta1 * abba_gvalue(gos, nnorb, opb1, opa[io + 3u]);
                abba_beta_kernel2dst4(y0, y1, t0, t1, t2, t3,
                                          g00, g01, g02, g03,
                                          g10, g11, g12, g13, n);
        }
        for (; io + 2u <= nop; io += 2u) {
                const double *restrict t0 = tbase + (size_t)(io + 0u) * op_stride;
                const double *restrict t1 = tbase + (size_t)(io + 1u) * op_stride;
                double g00 = sbeta0 * abba_gvalue(gos, nnorb, opb0, opa[io + 0u]);
                double g01 = sbeta0 * abba_gvalue(gos, nnorb, opb0, opa[io + 1u]);
                double g10 = sbeta1 * abba_gvalue(gos, nnorb, opb1, opa[io + 0u]);
                double g11 = sbeta1 * abba_gvalue(gos, nnorb, opb1, opa[io + 1u]);
                abba_beta_kernel2(y0, t0, t1, g00, g01, n);
                abba_beta_kernel2(y1, t0, t1, g10, g11, n);
        }
        for (; io < nop; io++) {
                const double *restrict t0 = tbase + (size_t)io * op_stride;
                double g0 = sbeta0 * abba_gvalue(gos, nnorb, opb0, opa[io]);
                double g1 = sbeta1 * abba_gvalue(gos, nnorb, opb1, opa[io]);
                abba_beta_kernel1(y0, t0, g0, n);
                abba_beta_kernel1(y1, t0, g1, n);
        }
}

/* Blocked transpose from C[alpha,beta] to T[beta,alpha]. */
static void transpose_pack(double *restrict t,
                           const double *restrict c, uint32_t c_nb,
                           uint32_t na, uint32_t nb, uint32_t c_a0)
{
        for (uint32_t a0 = 0; a0 < na; a0 += BB_TRANSPOSE_TILE) {
                uint32_t a1 = na - a0 < BB_TRANSPOSE_TILE ?
                              na : a0 + BB_TRANSPOSE_TILE;
                for (uint32_t b0 = 0; b0 < nb; b0 += BB_TRANSPOSE_TILE) {
                        uint32_t b1 = nb - b0 < BB_TRANSPOSE_TILE ?
                                      nb : b0 + BB_TRANSPOSE_TILE;
                        for (uint32_t b = b0; b < b1; b++) {
                                double *restrict tr = t + (uint64_t)b * na + a0;
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (uint32_t a = a0; a < a1; a++) {
                                        tr[a - a0] = c[(uint64_t)(c_a0 + a) * c_nb + b];
                                }
                        }
                }
        }
}

/* Add T[beta,alpha] into the row-major block C[alpha,beta]. */
static void transpose_add(double *restrict c, uint32_t c_nb,
                          const double *restrict t,
                          uint32_t na, uint32_t nb, uint32_t c_a0,
                          uint32_t tile)
{
        for (uint32_t a0 = 0; a0 < na; a0 += tile) {
                uint32_t a1 = na - a0 < tile ? na : a0 + tile;
                for (uint32_t b0 = 0; b0 < nb; b0 += tile) {
                        uint32_t b1 = nb - b0 < tile ? nb : b0 + tile;
                        for (uint32_t a = a0; a < a1; a++) {
                                double *restrict cr = c +
                                        (uint64_t)(c_a0 + a) * c_nb + b0;
#ifdef _OPENMP
#pragma omp simd
#endif
                                for (uint32_t b = b0; b < b1; b++) {
                                        cr[b - b0] += t[(uint64_t)b * na + a];
                                }
                        }
                }
        }
}

static inline void abba_transpose_add(double *restrict dst, uint32_t dst_nb,
                                      const double *restrict y,
                                      uint32_t na, uint32_t nb,
                                      uint32_t dst_a0)
{
        transpose_add(dst, dst_nb, y, na, nb, dst_a0, ABBA_WRITEBACK_TILE);
}

static inline void contract_ws_init(gas_contract_ws_t *ws)
{
        memset(ws, 0, sizeof(*ws));
}

static inline void contract_ws_free(gas_contract_ws_t *ws)
{
        free(ws->arena);
        free(ws->pair);
        free(ws->op_map);
        free(ws->op);
        free(ws->alpha_hit);
        memset(ws, 0, sizeof(*ws));
}

static inline uint64_t contract_ws_bytes(const gas_contract_ws_t *ws)
{
        return (uint64_t)ws->arena_cap * sizeof(*ws->arena) +
               (uint64_t)ws->pair_cap * sizeof(*ws->pair) +
               (uint64_t)ws->op_map_cap * sizeof(*ws->op_map) +
               (uint64_t)ws->op_cap * sizeof(*ws->op) +
               (uint64_t)ws->alpha_hit_cap * sizeof(*ws->alpha_hit);
}

static size_t contract_grow_capacity(size_t cap, size_t need)
{
        size_t next = cap ? cap : 64u;
        while (next < need) {
                if (next > SIZE_MAX / 2u) {
                        return need;
                }
                next *= 2u;
        }
        return next;
}

static int reserve_arena(gas_contract_ws_t *ws, size_t n)
{
        if (n <= ws->arena_cap) {
                return GAS_SUCCESS;
        }
        size_t cap = contract_grow_capacity(ws->arena_cap, n);
        double *p = realloc(ws->arena, cap * sizeof(*p));
        if (p == 0) {
                return GAS_ERR_MEMORY;
        }
        ws->arena = p;
        ws->arena_cap = cap;
        if (ws->arena_zeroed_upto > cap) {
                ws->arena_zeroed_upto = cap;
        }
        return GAS_SUCCESS;
}

static int reserve_alpha_hitlist(gas_contract_ws_t *ws, size_t n)
{
        if (n > UINT32_MAX) {
                return GAS_ERR_MEMORY;
        }
        if (n <= ws->alpha_hit_cap) {
                return GAS_SUCCESS;
        }
        size_t cap = contract_grow_capacity(ws->alpha_hit_cap, n);
        gas_alpha_hit_t *p = realloc(ws->alpha_hit, cap * sizeof(*p));
        if (p == 0) {
                return GAS_ERR_MEMORY;
        }
        ws->alpha_hit = p;
        ws->alpha_hit_cap = cap;
        return GAS_SUCCESS;
}

static int reserve_pairs(gas_contract_ws_t *ws, size_t n)
{
        if (n <= ws->pair_cap) {
                return GAS_SUCCESS;
        }
        size_t cap = contract_grow_capacity(ws->pair_cap, n);
        gas_tid_pair_t *p = realloc(ws->pair, cap * sizeof(*p));
        if (p == 0) {
                return GAS_ERR_MEMORY;
        }
        ws->pair = p;
        ws->pair_cap = cap;
        return GAS_SUCCESS;
}

static int reserve_op_map(gas_contract_ws_t *ws, size_t n)
{
        if (n <= ws->op_map_cap) {
                return GAS_SUCCESS;
        }
        size_t cap = contract_grow_capacity(ws->op_map_cap, n);
        int32_t *p = realloc(ws->op_map, cap * sizeof(*p));
        if (p == 0) {
                return GAS_ERR_MEMORY;
        }
        ws->op_map = p;
        ws->op_map_cap = cap;
        return GAS_SUCCESS;
}

static int reserve_ops(gas_contract_ws_t *ws, size_t n)
{
        if (n <= ws->op_cap) {
                return GAS_SUCCESS;
        }
        size_t cap = contract_grow_capacity(ws->op_cap, n);
        uint16_t *p = realloc(ws->op, cap * sizeof(*p));
        if (p == 0) {
                return GAS_ERR_MEMORY;
        }
        ws->op = p;
        ws->op_cap = cap;
        return GAS_SUCCESS;
}

#ifdef _OPENMP
static gas_tid_t find_table_in_row(gas_sid_t dst,
                                   const gas_sid_t *restrict sid,
                                   gas_tid_t off, uint32_t n)
{
        if (n <= 8u) {
                for (uint32_t i = 0; i < n; i++) {
                        if (sid[i] == dst) {
                                return off + i;
                        }
                }
                return GAS_INVALID_TID;
        }

        uint32_t lo = 0;
        uint32_t hi = n;
        while (lo < hi) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                if (sid[mid] < dst) {
                        lo = mid + 1u;
                } else {
                        hi = mid;
                }
        }
        return lo < n && sid[lo] == dst ? off + lo : GAS_INVALID_TID;
}
#endif

/* ========================================================================== */
/* 3. Two-electron path kernels                                               */
/* ========================================================================== */

static void contract_aa_path_rows(const gas_space_t *gas,
                                  const double *restrict eri,
                                  const double *restrict ci0,
                                  double *restrict ci1,
                                  uint32_t nnorb,
                                  gas_bid_t bsrc, gas_bid_t bdst,
                                  gas_tid_t tid1, gas_tid_t tid2,
                                  uint32_t a0, uint32_t a1)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *t1 = gas->table + tid1;
        const gas_link_table_t *t2 = gas->table + tid2;
        const uint32_t nb = gas->sector_nstr[bs->sb];
        const uint32_t na1 = gas->sector_nstr[bd->sa];
        const double *src = ci0 + bs->offset;
        double *dst = ci1 + bd->offset;

        if (a0 > na1) a0 = na1;
        if (a1 > na1) a1 = na1;
        if (a1 <= a0) return;

        for (uint32_t ia0 = 0; ia0 < t1->nsrc; ia0++) {
                const double *src_row = src + (uint64_t)ia0 * nb;
                const gas_link_entry_t *r1 = t1->link + (uint64_t)ia0 * t1->nlink;
                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                        const gas_link_entry_t *e1 = r1 + k1;
                        const gas_link_entry_t *r2 = t2->link +
                                (uint64_t)e1->addr * t2->nlink;
                        uint16_t op1 = gas_link_ia(e1);
                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                const gas_link_entry_t *e2 = r2 + k2;
                                if (e2->addr < a0 || e2->addr >= a1) continue;
                                double fac = (double)(e1->sign * e2->sign) *
                                        eri[(uint64_t)gas_link_ia(e2) * nnorb + op1];
                                double *dst_row = dst + (uint64_t)e2->addr * nb;
                                contract_axpy(dst_row, src_row, fac, nb);
                        }
                }
        }
}

static void contract_aa_path(const gas_space_t *gas,
                             const double *restrict eri,
                             const double *restrict ci0,
                             double *restrict ci1,
                             uint32_t nnorb,
                             gas_bid_t bsrc, gas_bid_t bdst,
                             gas_tid_t tid1, gas_tid_t tid2)
{
        contract_aa_path_rows(gas, eri, ci0, ci1, nnorb, bsrc, bdst,
                              tid1, tid2, 0u,
                              gas->sector_nstr[gas->block[bdst].sa]);
}

#ifdef _OPENMP
static int contract_aa_path_rows_accum(const gas_space_t *gas,
                                       const double *restrict eri,
                                       const double *restrict ci0,
                                       double *restrict ci1,
                                       uint32_t nnorb,
                                       gas_bid_t bsrc, gas_bid_t bdst,
                                       gas_tid_t tid1, gas_tid_t tid2,
                                       uint32_t a0, uint32_t a1,
                                       gas_contract_ws_t *ws)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *t1 = gas->table + tid1;
        const gas_link_table_t *t2 = gas->table + tid2;
        const uint32_t nb = gas->sector_nstr[bs->sb];
        const uint32_t na1 = gas->sector_nstr[bd->sa];
        const double *src = ci0 + bs->offset;
        double *dst = ci1 + bd->offset;

        if (a0 > na1) a0 = na1;
        if (a1 > na1) a1 = na1;
        if (a1 <= a0) return GAS_SUCCESS;

        if (na1 > UINT16_MAX ||
            reserve_op_map(ws, na1) != GAS_SUCCESS ||
            reserve_ops(ws, na1) != GAS_SUCCESS ||
            reserve_arena(ws, na1) != GAS_SUCCESS ||
            ws->op_map == 0 || ws->op == 0 || ws->arena == 0) {
                return GAS_ERR_MEMORY;
        }
        for (uint32_t i = 0; i < na1; i++) {
                ws->op_map[i] = -1;
        }
        double *acc = ws->arena;

        /* Large-block AA kernel.  For a fixed source alpha string, the two
         * one-electron alpha paths can reach the same destination alpha string
         * through different intermediate strings/orderings.  A direct kernel would do
         * one full beta-row AXPY per path.  Here all coefficients landing on the
         * same destination alpha row are accumulated first, then one AXPY is
         * issued.  This mirrors the BB coefficient-accumulation kernel and
         * targets the large dense beta rows in grouped2x8. */
        for (uint32_t ia0 = 0; ia0 < t1->nsrc; ia0++) {
                const double *src_row = src + (uint64_t)ia0 * nb;
                const gas_link_entry_t *r1 = t1->link + (uint64_t)ia0 * t1->nlink;
                uint32_t ntouch = 0;

                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                        const gas_link_entry_t *e1 = r1 + k1;
                        const gas_link_entry_t *r2 = t2->link +
                                (uint64_t)e1->addr * t2->nlink;
                        uint16_t op1 = gas_link_ia(e1);
                        int s1 = e1->sign;
                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                const gas_link_entry_t *e2 = r2 + k2;
                                uint32_t ia2 = e2->addr;
                                if (ia2 < a0 || ia2 >= a1) continue;
                                double fac = (double)(s1 * e2->sign) *
                                        eri[(uint64_t)gas_link_ia(e2) * nnorb + op1];
                                int32_t pos = ws->op_map[ia2];
                                if (pos < 0) {
                                        pos = (int32_t)ntouch;
                                        ws->op_map[ia2] = pos;
                                        ws->op[ntouch++] = (uint16_t)ia2;
                                        acc[ia2] = fac;
                                } else {
                                        acc[ia2] += fac;
                                }
                        }
                }

                uint32_t it = 0;
                for (; it + 4u <= ntouch; it += 4u) {
                        uint32_t ia20 = ws->op[it + 0u];
                        uint32_t ia21 = ws->op[it + 1u];
                        uint32_t ia22 = ws->op[it + 2u];
                        uint32_t ia23 = ws->op[it + 3u];
                        contract_axpy4(dst + (uint64_t)ia20 * nb,
                                       dst + (uint64_t)ia21 * nb,
                                       dst + (uint64_t)ia22 * nb,
                                       dst + (uint64_t)ia23 * nb,
                                       src_row, acc[ia20], acc[ia21],
                                       acc[ia22], acc[ia23], nb);
                        ws->op_map[ia20] = -1;
                        ws->op_map[ia21] = -1;
                        ws->op_map[ia22] = -1;
                        ws->op_map[ia23] = -1;
                }
                if (it + 2u <= ntouch) {
                        uint32_t ia20 = ws->op[it + 0u];
                        uint32_t ia21 = ws->op[it + 1u];
                        contract_axpy2(dst + (uint64_t)ia20 * nb,
                                       dst + (uint64_t)ia21 * nb,
                                       src_row, acc[ia20], acc[ia21], nb);
                        ws->op_map[ia20] = -1;
                        ws->op_map[ia21] = -1;
                        it += 2u;
                }
                for (; it < ntouch; it++) {
                        uint32_t ia2 = ws->op[it];
                        contract_axpy(dst + (uint64_t)ia2 * nb,
                                      src_row, acc[ia2], nb);
                        ws->op_map[ia2] = -1;
                }
        }
        return GAS_SUCCESS;
}
#endif

static void contract_bb_path_rows(const gas_space_t *gas,
                                  const double *restrict eri,
                                  const double *restrict ci0,
                                  double *restrict ci1,
                                  uint32_t nnorb,
                                  gas_bid_t bsrc, gas_bid_t bdst,
                                  gas_tid_t tid1, gas_tid_t tid2,
                                  uint32_t a0, uint32_t a1)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *t1 = gas->table + tid1;
        const gas_link_table_t *t2 = gas->table + tid2;
        const uint32_t na = gas->sector_nstr[bs->sa];
        const uint32_t nb0 = gas->sector_nstr[bs->sb];
        const uint32_t nb2 = gas->sector_nstr[bd->sb];
        const double *src = ci0 + bs->offset;
        double *dst = ci1 + bd->offset;

        if (a0 > na) a0 = na;
        if (a1 > na) a1 = na;
        if (a1 <= a0) return;

        for (uint32_t aa0 = a0; aa0 < a1; aa0 += BB_ALPHA_BLOCK) {
                uint32_t aa1 = a1 - aa0 < BB_ALPHA_BLOCK ? a1 : aa0 + BB_ALPHA_BLOCK;
                for (uint32_t ib0 = 0; ib0 < t1->nsrc; ib0++) {
                        const gas_link_entry_t *r1 = t1->link +
                                (uint64_t)ib0 * t1->nlink;
                        for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                                const gas_link_entry_t *e1 = r1 + k1;
                                const gas_link_entry_t *r2 = t2->link +
                                        (uint64_t)e1->addr * t2->nlink;
                                uint16_t op1 = gas_link_ia(e1);
                                int s1 = e1->sign;
                                for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                        const gas_link_entry_t *e2 = r2 + k2;
                                        double fac = (double)(s1 * e2->sign) *
                                                eri[(uint64_t)gas_link_ia(e2) * nnorb + op1];
                                        uint32_t ib2 = e2->addr;
#ifdef _OPENMP
#pragma omp simd
#endif
                                        for (uint32_t ia = aa0; ia < aa1; ia++) {
                                                dst[(uint64_t)ia * nb2 + ib2] +=
                                                        fac * src[(uint64_t)ia * nb0 + ib0];
                                        }
                                }
                        }
                }
        }
}

static void contract_bb_path(const gas_space_t *gas,
                             const double *restrict eri,
                             const double *restrict ci0,
                             double *restrict ci1,
                             uint32_t nnorb,
                             gas_bid_t bsrc, gas_bid_t bdst,
                             gas_tid_t tid1, gas_tid_t tid2)
{
        contract_bb_path_rows(gas, eri, ci0, ci1, nnorb, bsrc, bdst,
                              tid1, tid2, 0u,
                              gas->sector_nstr[gas->block[bsrc].sa]);
}

static int use_bb_transpose(uint32_t na, uint32_t nb0,
                            uint32_t nb2, uint32_t npair)
{
        size_t elems = (size_t)na * nb0 + (size_t)na * nb2;
        return npair != 0 && na >= BB_TRANSPOSE_MIN_DIM &&
               nb0 >= BB_TRANSPOSE_MIN_DIM && nb2 >= BB_TRANSPOSE_MIN_DIM &&
               elems <= BB_TRANSPOSE_MAX_ELEMS;
}

static int contract_bb_blockpair_transpose_rows(
        const gas_space_t *gas,
        const double *restrict eri,
        const double *restrict ci0,
        double *restrict ci1,
        uint32_t nnorb,
        gas_bid_t bsrc, gas_bid_t bdst,
        const gas_tid_pair_t *pair,
        uint32_t npair,
        gas_contract_ws_t *ws,
        uint32_t a0, uint32_t a1)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const uint32_t na = gas->sector_nstr[bs->sa];
        const uint32_t nb0 = gas->sector_nstr[bs->sb];
        const uint32_t nb2 = gas->sector_nstr[bd->sb];
        if (a0 > na) a0 = na;
        if (a1 > na) a1 = na;
        if (a1 <= a0) return GAS_SUCCESS;
        const uint32_t nat = a1 - a0;
        const size_t nsrc = (size_t)nat * nb0;
        const size_t ndst = (size_t)nat * nb2;
        const double *src = ci0 + bs->offset;
        double *dst = ci1 + bd->offset;
        int use_accum = 0;
        if (nb2 <= UINT16_MAX &&
            reserve_op_map(ws, nb2) == GAS_SUCCESS &&
            reserve_ops(ws, nb2) == GAS_SUCCESS &&
            reserve_arena(ws, nsrc + ndst + (size_t)nb2) == GAS_SUCCESS) {
                use_accum = 1;
                for (uint32_t i = 0; i < nb2; i++) {
                        ws->op_map[i] = -1;
                }
        } else if (reserve_arena(ws, nsrc + ndst) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        double *src_t = ws->arena;
        double *dst_t = ws->arena + nsrc;
        double *acc = use_accum ? ws->arena + nsrc + ndst : 0;

        transpose_pack(src_t, src, nb0, nat, nb0, a0);
        memset(dst_t, 0, ndst * sizeof(*dst_t));
        if (use_accum) {
                /* Large-block BB kernel: for each source beta string, combine all
                 * path coefficients that land on the same destination beta string.
                 * This turns many repeated AXPYs over the same alpha tile into one
                 * AXPY with the accumulated coefficient.  The touched list keeps
                 * reset cost proportional to the number of destinations reached,
                 * not to nb2. */
                for (uint32_t ib0 = 0; ib0 < nb0; ib0++) {
                        const double *src_vec = src_t + (uint64_t)ib0 * nat;
                        uint32_t ntouch = 0;

                        for (uint32_t ip = 0; ip < npair; ip++) {
                                const gas_link_table_t *t1 = gas->table + pair[ip].first;
                                const gas_link_table_t *t2 = gas->table + pair[ip].second;
                                if (ib0 >= t1->nsrc) {
                                        continue;
                                }
                                const gas_link_entry_t *r1 = t1->link +
                                        (uint64_t)ib0 * t1->nlink;
                                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                                        const gas_link_entry_t *e1 = r1 + k1;
                                        const gas_link_entry_t *r2 = t2->link +
                                                (uint64_t)e1->addr * t2->nlink;
                                        uint16_t op1 = gas_link_ia(e1);
                                        int s1 = e1->sign;
                                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                                const gas_link_entry_t *e2 = r2 + k2;
                                                uint32_t ib2 = e2->addr;
                                                double fac = (double)(s1 * e2->sign) *
                                                        eri[(uint64_t)gas_link_ia(e2) *
                                                            nnorb + op1];
                                                int32_t pos = ws->op_map[ib2];
                                                if (pos < 0) {
                                                        pos = (int32_t)ntouch;
                                                        ws->op_map[ib2] = pos;
                                                        ws->op[ntouch++] = (uint16_t)ib2;
                                                        acc[ib2] = fac;
                                                } else {
                                                        acc[ib2] += fac;
                                                }
                                        }
                                }
                        }

                        for (uint32_t it = 0; it < ntouch; it++) {
                                uint32_t ib2 = ws->op[it];
                                contract_axpy(dst_t + (uint64_t)ib2 * nat,
                                             src_vec, acc[ib2], nat);
                                ws->op_map[ib2] = -1;
                        }
                }
        } else {
                for (uint32_t ip = 0; ip < npair; ip++) {
                        const gas_link_table_t *t1 = gas->table + pair[ip].first;
                        const gas_link_table_t *t2 = gas->table + pair[ip].second;
                        for (uint32_t ib0 = 0; ib0 < t1->nsrc; ib0++) {
                                const double *src_vec = src_t + (uint64_t)ib0 * nat;
                                const gas_link_entry_t *r1 = t1->link +
                                        (uint64_t)ib0 * t1->nlink;
                                for (uint32_t k1 = 0; k1 < t1->nlink; k1++) {
                                        const gas_link_entry_t *e1 = r1 + k1;
                                        const gas_link_entry_t *r2 = t2->link +
                                                (uint64_t)e1->addr * t2->nlink;
                                        uint16_t op1 = gas_link_ia(e1);
                                        int s1 = e1->sign;
                                        for (uint32_t k2 = 0; k2 < t2->nlink; k2++) {
                                                const gas_link_entry_t *e2 = r2 + k2;
                                                double fac = (double)(s1 * e2->sign) *
                                                        eri[(uint64_t)gas_link_ia(e2) *
                                                            nnorb + op1];
                                                double *dst_vec = dst_t +
                                                        (uint64_t)e2->addr * nat;
                                                contract_axpy(dst_vec, src_vec, fac, nat);
                                        }
                                }
                        }
                }
        }
        transpose_add(dst, nb2, dst_t, nat, nb2, a0, BB_TRANSPOSE_TILE);

        return GAS_SUCCESS;
}

static void contract_bb_path_list_rows(const gas_space_t *gas,
                                       const double *restrict eri,
                                       const double *restrict ci0,
                                       double *restrict ci1,
                                       uint32_t nnorb,
                                       gas_bid_t bsrc, gas_bid_t bdst,
                                       const gas_tid_pair_t *pair,
                                       uint32_t npair,
                                       uint32_t a0, uint32_t a1)
{
        for (uint32_t i = 0; i < npair; i++) {
                contract_bb_path_rows(gas, eri, ci0, ci1, nnorb, bsrc, bdst,
                                      pair[i].first, pair[i].second, a0, a1);
        }
}

static void contract_bb_blockpair_rows(const gas_space_t *gas,
                                       const double *restrict eri,
                                       const double *restrict ci0,
                                       double *restrict ci1,
                                       uint32_t nnorb,
                                       gas_bid_t bsrc, gas_bid_t bdst,
                                       const gas_tid_pair_t *pair,
                                       uint32_t npair,
                                       gas_contract_ws_t *ws,
                                       uint32_t a0, uint32_t a1)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        uint32_t na = gas->sector_nstr[bs->sa];
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        uint32_t nb2 = gas->sector_nstr[bd->sb];
        if (a0 > na) a0 = na;
        if (a1 > na) a1 = na;
        uint32_t nat = a1 > a0 ? a1 - a0 : 0u;

        if (use_bb_transpose(nat, nb0, nb2, npair) &&
            contract_bb_blockpair_transpose_rows(gas, eri, ci0, ci1, nnorb,
                                                  bsrc, bdst, pair, npair, ws,
                                                  a0, a1) == GAS_SUCCESS) {
                return;
        }
        contract_bb_path_list_rows(gas, eri, ci0, ci1, nnorb,
                                   bsrc, bdst, pair, npair, a0, a1);
}

static void contract_bb_blockpair(const gas_space_t *gas,
                                  const double *restrict eri,
                                  const double *restrict ci0,
                                  double *restrict ci1,
                                  uint32_t nnorb,
                                  gas_bid_t bsrc, gas_bid_t bdst,
                                  const gas_tid_pair_t *pair, uint32_t npair,
                                  gas_contract_ws_t *ws)
{
        contract_bb_blockpair_rows(gas, eri, ci0, ci1, nnorb,
                                   bsrc, bdst, pair, npair, ws, 0u,
                                   gas->sector_nstr[gas->block[bsrc].sa]);
}

static void contract_abba_path_rows(const gas_space_t *gas,
                                    const double *restrict eri,
                                    const double *restrict ci0,
                                    double *restrict ci1,
                                    uint32_t nnorb,
                                    gas_bid_t bsrc, gas_bid_t bdst,
                                    gas_tid_t tida, gas_tid_t tidb,
                                    uint32_t a0, uint32_t a1)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_block_t *bd = gas->block + bdst;
        const gas_link_table_t *ta = gas->table + tida;
        const gas_link_table_t *tb = gas->table + tidb;
        const uint32_t nb0 = gas->sector_nstr[bs->sb];
        const uint32_t nb1 = gas->sector_nstr[bd->sb];
        const uint32_t na1 = gas->sector_nstr[bd->sa];
        const double *src = ci0 + bs->offset;
        double *dst = ci1 + bd->offset;
        if (a0 > na1) a0 = na1;
        if (a1 > na1) a1 = na1;
        if (a1 <= a0) return;

        for (uint32_t ia0 = 0; ia0 < ta->nsrc; ia0++) {
                const double *src_row = src + (uint64_t)ia0 * nb0;
                const gas_link_entry_t *ra = ta->link + (uint64_t)ia0 * ta->nlink;
                for (uint32_t ka = 0; ka < ta->nlink; ka++) {
                        const gas_link_entry_t *ea = ra + ka;
                        if (ea->addr < a0 || ea->addr >= a1) continue;
                        double *dst_row = dst + (uint64_t)ea->addr * nb1;
                        uint16_t opa = gas_link_ia(ea);
                        int sa = ea->sign;
                        for (uint32_t ib0 = 0; ib0 < tb->nsrc; ib0++) {
                                double c = src_row[ib0];
                                const gas_link_entry_t *rb = tb->link +
                                        (uint64_t)ib0 * tb->nlink;
                                for (uint32_t kb = 0; kb < tb->nlink; kb++) {
                                        const gas_link_entry_t *eb = rb + kb;
                                        uint16_t opb = gas_link_ia(eb);
                                        dst_row[eb->addr] += (double)(sa * eb->sign) * c *
                                                (eri[(uint64_t)opb * nnorb + opa] +
                                                 eri[(uint64_t)opa * nnorb + opb]);
                                }
                        }
                }
        }
}

static void contract_abba_path(const gas_space_t *gas,
                               const double *restrict eri,
                               const double *restrict ci0,
                               double *restrict ci1,
                               uint32_t nnorb,
                               gas_bid_t bsrc, gas_bid_t bdst,
                               gas_tid_t tida, gas_tid_t tidb)
{
        contract_abba_path_rows(gas, eri, ci0, ci1, nnorb, bsrc, bdst,
                                tida, tidb, 0u,
                                gas->sector_nstr[gas->block[bdst].sa]);
}

/* ========================================================================== */
/* 4. Destination-driven two-electron traversal                               */
/* ========================================================================== */

static uint32_t aa_forward_scan_cost(const gas_space_t *gas,
                                     const gas_sid_t *as, uint32_t n)
{
        uint32_t cost = 0;
        for (uint32_t i = 0; i < n; i++) {
                cost += gas->T.row[as[i]].n;
        }
        return cost;
}

static void contract_aa_forward(const gas_space_t *gas,
                                const double *restrict eri,
                                const double *restrict ci0,
                                double *restrict ci1,
                                uint32_t nnorb, gas_bid_t bsrc)
{
        const gas_block_t *blk = gas->block + bsrc;
        gas_row_t ar = gas->T.row[blk->sa];
        const gas_sid_t *as = gas->T.dst + ar.off;

        for (uint32_t i = 0; i < ar.n; i++) {
                gas_row_t row = gas->T.row[as[i]];
                const gas_sid_t *dst = gas->T.dst + row.off;
                for (uint32_t j = 0; j < row.n; j++) {
                        gas_bid_t bd = gas_find_block(gas, dst[j], blk->sb);
                        if (bd != GAS_INVALID_BID) {
                                contract_aa_path(gas, eri, ci0, ci1, nnorb,
                                                 bsrc, bd, ar.off + i, row.off + j);
                        }
                }
        }
}

static uint32_t bb_forward_scan_cost(const gas_space_t *gas,
                                     const gas_sid_t *bs, uint32_t n)
{
        uint32_t cost = 0;
        for (uint32_t i = 0; i < n; i++) {
                cost += gas->T.row[bs[i]].n;
        }
        return cost;
}

static void contract_bb_forward(const gas_space_t *gas,
                                const double *restrict eri,
                                const double *restrict ci0,
                                double *restrict ci1,
                                uint32_t nnorb, gas_bid_t bsrc)
{
        const gas_block_t *blk = gas->block + bsrc;
        gas_row_t br = gas->T.row[blk->sb];
        const gas_sid_t *bs = gas->T.dst + br.off;

        for (uint32_t i = 0; i < br.n; i++) {
                gas_row_t row = gas->T.row[bs[i]];
                const gas_sid_t *dst = gas->T.dst + row.off;
                for (uint32_t j = 0; j < row.n; j++) {
                        gas_bid_t bd = gas_find_block(gas, blk->sa, dst[j]);
                        if (bd != GAS_INVALID_BID) {
                                contract_bb_path(gas, eri, ci0, ci1, nnorb,
                                                 bsrc, bd, br.off + i, row.off + j);
                        }
                }
        }
}

static void contract_aa_by_dst(const gas_space_t *gas,
                               const double *restrict eri,
                               const double *restrict ci0,
                               double *restrict ci1,
                               uint32_t nnorb, gas_bid_t bsrc)
{
        const gas_block_t *blk = gas->block + bsrc;
        gas_row_t ar = gas->T.row[blk->sa];
        const gas_sid_t *as = gas->T.dst + ar.off;
        gas_row_t qrow = gas->D.by_beta_row[blk->sb];
        const gas_sid_t *qa = gas->D.by_beta_sid + qrow.off;
        const gas_bid_t *qb = gas->D.by_beta_bid + qrow.off;

        for (uint32_t q = 0; q < qrow.n; q++) {
                gas_row_t rr = gas->R.row[qa[q]];
                const gas_sid_t *rs = gas->R.src + rr.off;
                const gas_tid_t *rt = gas->R.tid + rr.off;
                uint32_t i = 0;
                uint32_t j = 0;
                while (i < ar.n && j < rr.n) {
                        if (as[i] < rs[j]) {
                                i++;
                        } else if (as[i] > rs[j]) {
                                j++;
                        } else {
                                contract_aa_path(gas, eri, ci0, ci1, nnorb,
                                                 bsrc, qb[q], ar.off + i, rt[j]);
                                i++;
                                j++;
                        }
                }
        }
}

static void contract_bb_by_dst(const gas_space_t *gas,
                               const double *restrict eri,
                               const double *restrict ci0,
                               double *restrict ci1,
                               uint32_t nnorb, gas_bid_t bsrc,
                               gas_contract_ws_t *ws)
{
        const gas_block_t *blk = gas->block + bsrc;
        uint32_t na = gas->sector_nstr[blk->sa];
        uint32_t nb0 = gas->sector_nstr[blk->sb];
        gas_row_t br = gas->T.row[blk->sb];
        const gas_sid_t *bs = gas->T.dst + br.off;
        gas_row_t qrow = gas->D.by_alpha_row[blk->sa];

        for (uint32_t q = 0; q < qrow.n; q++) {
                gas_bid_t bdst = qrow.off + q;
                gas_sid_t b2 = gas->block[bdst].sb;
                gas_row_t rr = gas->R.row[b2];
                const gas_sid_t *rs = gas->R.src + rr.off;
                const gas_tid_t *rt = gas->R.tid + rr.off;
                uint32_t i = 0;
                uint32_t j = 0;
                uint32_t maxpair = br.n < rr.n ? br.n : rr.n;
                uint32_t nb2 = gas->sector_nstr[b2];

                if (use_bb_transpose(na, nb0, nb2, maxpair) &&
                    reserve_pairs(ws, maxpair) == GAS_SUCCESS) {
                        uint32_t npair = 0;
                        while (i < br.n && j < rr.n) {
                                if (bs[i] < rs[j]) {
                                        i++;
                                } else if (bs[i] > rs[j]) {
                                        j++;
                                } else {
                                        ws->pair[npair].first = br.off + i;
                                        ws->pair[npair].second = rt[j];
                                        npair++;
                                        i++;
                                        j++;
                                }
                        }
                        if (npair != 0) {
                                contract_bb_blockpair(gas, eri, ci0, ci1, nnorb,
                                                      bsrc, bdst, ws->pair, npair, ws);
                        }
                        continue;
                }

                while (i < br.n && j < rr.n) {
                        if (bs[i] < rs[j]) {
                                i++;
                        } else if (bs[i] > rs[j]) {
                                j++;
                        } else {
                                contract_bb_path(gas, eri, ci0, ci1, nnorb,
                                                 bsrc, bdst, br.off + i, rt[j]);
                                i++;
                                j++;
                        }
                }
        }
}

static int collect_alpha_ops(const gas_link_table_t *ta,
                             uint32_t nnorb, gas_contract_ws_t *ws,
                             uint32_t *nop_out)
{
        uint32_t nop = ta->nop;
        if (reserve_op_map(ws, nnorb) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        for (uint32_t i = 0; i < nnorb; i++) {
                ws->op_map[i] = -1;
        }
        if (reserve_ops(ws, nop) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        if (nop != 0u && ta->active_op == 0) {
                return GAS_ERR_INVALID;
        }
        for (uint32_t i = 0; i < nop; i++) {
                uint16_t op = ta->active_op[i];
                ws->op_map[op] = (int32_t)i;
                ws->op[i] = op;
        }
        *nop_out = nop;
        return GAS_SUCCESS;
}

static int build_alpha_hitlist(const gas_link_table_t *ta,
                               uint32_t a0, uint32_t a1,
                               gas_contract_ws_t *ws)
{
        uint32_t nhit = 0u;

        for (uint32_t ia0 = 0; ia0 < ta->nsrc; ia0++) {
                const gas_link_entry_t *restrict ra = ta->link +
                        (uint64_t)ia0 * ta->nlink;
                for (uint32_t ka = 0; ka < ta->nlink; ka++) {
                        const gas_link_entry_t *ea = ra + ka;
                        if (ea->addr >= a0 && ea->addr < a1) {
                                nhit++;
                        }
                }
        }

        if (reserve_alpha_hitlist(ws, nhit) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }

        uint32_t out = 0u;
        for (uint32_t ia0 = 0; ia0 < ta->nsrc; ia0++) {
                const gas_link_entry_t *restrict ra = ta->link +
                        (uint64_t)ia0 * ta->nlink;
                for (uint32_t ka = 0; ka < ta->nlink; ka++) {
                        const gas_link_entry_t *ea = ra + ka;
                        if (ea->addr < a0 || ea->addr >= a1) continue;
                        int32_t io = ws->op_map[gas_link_ia(ea)];
                        if (io < 0) continue;
                        ws->alpha_hit[out].src_row = ia0;
                        ws->alpha_hit[out].rel_addr = ea->addr - a0;
                        ws->alpha_hit[out].op_index = (uint16_t)io;
                        ws->alpha_hit[out].sign = ea->sign;
                        ws->alpha_hit[out].pad = 0u;
                        out++;
                }
        }
        ws->alpha_hit_n = out;

        return GAS_SUCCESS;
}

static int collect_abba_beta_pairs_range(const gas_space_t *gas,
                                         gas_sid_t adst,
                                         uint32_t q0, uint32_t q1,
                                         const gas_sid_t *restrict beta_sid,
                                         gas_tid_t beta_off, uint32_t nbeta,
                                         gas_contract_ws_t *ws,
                                         uint32_t *npair_out)
{
        gas_row_t brow = gas->D.by_alpha_row[adst];
        uint32_t nrange;
        uint32_t maxpair;
        uint32_t q = q0;
        uint32_t j = 0;
        uint32_t npair = 0;

        if (q0 > brow.n) q0 = brow.n;
        if (q1 > brow.n) q1 = brow.n;
        if (q1 < q0) q1 = q0;
        q = q0;
        nrange = q1 - q0;
        maxpair = nrange < nbeta ? nrange : nbeta;

        if (reserve_pairs(ws, maxpair) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }

        while (q < q1 && j < nbeta) {
                gas_bid_t bdst = brow.off + q;
                gas_sid_t b1 = gas->block[bdst].sb;
                gas_sid_t b2 = beta_sid[j];

                if (b1 < b2) {
                        q++;
                } else if (b1 > b2) {
                        j++;
                } else {
                        ws->pair[npair].first = bdst;
                        ws->pair[npair].second = beta_off + j;
                        npair++;
                        q++;
                        j++;
                }
        }

        *npair_out = npair;
        return GAS_SUCCESS;
}

static gas_batch_status_t contract_abba_batch_alpha_tile(
        const gas_space_t *gas,
        const double *restrict gos,
        const double *restrict ci0,
        double *restrict ci1,
        uint32_t nnorb, gas_bid_t bsrc,
        gas_sid_t adst, gas_tid_t tida,
        uint32_t q0, uint32_t q1,
        uint32_t a0, uint32_t a1,
        const gas_sid_t *restrict beta_sid,
        gas_tid_t beta_off, uint32_t nbeta,
        gas_contract_ws_t *ws)
{
        const gas_block_t *bs = gas->block + bsrc;
        const gas_link_table_t *ta = gas->table + tida;
        uint32_t nb0 = gas->sector_nstr[bs->sb];
        uint32_t na_full = gas->sector_nstr[adst];
        const double *src = ci0 + bs->offset;
        uint32_t nop = 0;
        uint32_t npair = 0;
        if (a0 > na_full) a0 = na_full;
        if (a1 > na_full) a1 = na_full;
        if (a1 <= a0) return GAS_BATCH_DONE;
        uint32_t nat = a1 - a0;

        if (collect_alpha_ops(ta, nnorb, ws, &nop) != GAS_SUCCESS) {
                return GAS_BATCH_FALLBACK;
        }

        if (nop == 0u) {
                return GAS_BATCH_DONE;
        }
        if (collect_abba_beta_pairs_range(gas, adst, q0, q1, beta_sid, beta_off,
                                          nbeta, ws, &npair) != GAS_SUCCESS) {
                return GAS_BATCH_FALLBACK;
        }
        if (npair == 0u) {
                return GAS_BATCH_DONE;
        }

        uint32_t btile = abba_beta_tile_size(nb0, nop, nat);
        if (btile == 0u) {
                return GAS_BATCH_DONE;
        }

        if (build_alpha_hitlist(ta, a0, a1, ws) != GAS_SUCCESS) {
                return GAS_BATCH_FALLBACK;
        }

        for (uint32_t b0 = 0; b0 < nb0; b0 += btile) {
                uint32_t b1lim = nb0 - b0 < btile ? nb0 : b0 + btile;
                /* Keep one invariant [op][full-btile][alpha] layout even for
                 * the final partial beta tile.  Unused beta rows are never
                 * read, while stable strides make earlier zero state reusable. */
                uint32_t t1_nbt = btile;
                size_t dense_t1_elems = (size_t)nop * t1_nbt * nat;
                size_t t1_elems = dense_t1_elems;
                if (t1_elems > ABBA_WORKSPACE_MAX_ELEMS ||
                    reserve_arena(ws, t1_elems) != GAS_SUCCESS) {
                        return GAS_BATCH_FALLBACK;
                }
                if (b0 == 0u) {
                        memset(ws->arena, 0, t1_elems * sizeof(*ws->arena));
                        ws->arena_zeroed_upto = t1_elems;
                }
                for (uint32_t ih = 0; ih < ws->alpha_hit_n; ih++) {
                        const gas_alpha_hit_t *h = ws->alpha_hit + ih;
                        const double *restrict src_row = src + (uint64_t)h->src_row * nb0;
                        double *t = ws->arena +
                                ((size_t)h->op_index * t1_nbt * nat + h->rel_addr);
                        double sgn = (double)h->sign;
                        for (uint32_t ib0 = b0; ib0 < b1lim; ib0++) {
                                t[(size_t)(ib0 - b0) * nat] = sgn * src_row[ib0];
                        }
                }

                for (uint32_t ip = 0; ip < npair; ip++) {
                        gas_bid_t bdst = (gas_bid_t)ws->pair[ip].first;
                        gas_tid_t tidb = ws->pair[ip].second;
                        const gas_link_table_t *tb = gas->table + tidb;
                        gas_sid_t bdst_sid = gas->block[bdst].sb;
                        uint32_t nb1 = gas->sector_nstr[bdst_sid];
                        double *dst = ci1 + gas->block[bdst].offset;
                        size_t y_elems = (size_t)nb1 * nat;
                        int use_tile = t1_elems + y_elems <= ABBA_WORKSPACE_MAX_ELEMS &&
                                       reserve_arena(ws, t1_elems + y_elems) == GAS_SUCCESS;
                        double *t1 = ws->arena;

                        if (use_tile) {
                                double *y = ws->arena + t1_elems;
                                memset(y, 0, y_elems * sizeof(*y));
                                for (uint32_t ib0 = b0; ib0 < b1lim; ib0++) {
                                        const gas_link_entry_t *rb = tb->link +
                                                (uint64_t)ib0 * tb->nlink;
                                        const double *tbase = t1 +
                                                (size_t)(ib0 - b0) * nat;
                                        uint32_t kb = 0u;
                                        while (kb < tb->nlink) {
                                                const gas_link_entry_t *eb0 =
                                                        rb + kb;
                                                if (kb + 1u < tb->nlink) {
                                                        const gas_link_entry_t *eb1 =
                                                                rb + kb + 1u;
                                                        if (eb0->addr != eb1->addr) {
                                                                double *restrict y0 = y +
                                                                        (size_t)eb0->addr * nat;
                                                                double *restrict y1 = y +
                                                                        (size_t)eb1->addr * nat;
                                                                abba_fused_ops_pair2x4(
                                                                        y0, y1, tbase,
                                                                        (size_t)t1_nbt * nat,
                                                                        gos, nnorb,
                                                                        gas_link_ia(eb0),
                                                                        gas_link_ia(eb1),
                                                                        ws->op, nop,
                                                                        (double)eb0->sign,
                                                                        (double)eb1->sign,
                                                                        nat);
                                                                kb += 2u;
                                                                continue;
                                                        }
                                                }
                                                double *restrict y0 = y + (size_t)eb0->addr * nat;
                                                abba_fused_ops(
                                                        y0, tbase, (size_t)t1_nbt * nat,
                                                        gos, nnorb, gas_link_ia(eb0),
                                                        ws->op, nop, (double)eb0->sign,
                                                        nat);
                                                kb++;
                                        }
                                }
                                abba_transpose_add(dst, nb1, y, nat, nb1, a0);
                                continue;
                        }

                        for (uint32_t ib0 = b0; ib0 < b1lim; ib0++) {
                                const gas_link_entry_t *rb = tb->link +
                                        (uint64_t)ib0 * tb->nlink;
                                for (uint32_t kb = 0; kb < tb->nlink; kb++) {
                                        const gas_link_entry_t *eb = rb + kb;
                                        uint16_t opb = gas_link_ia(eb);
                                        double sbeta = (double)eb->sign;
                                        for (uint32_t io = 0; io < nop; io++) {
                                                uint16_t opa = ws->op[io];
                                                double g = sbeta *
                                                        abba_gvalue(gos, nnorb, opb, opa);
                                                const double *t = t1 +
                                                        ((size_t)io * t1_nbt +
                                                         (ib0 - b0)) * nat;
                                                for (uint32_t ia = 0; ia < nat; ia++) {
                                                        dst[(uint64_t)(a0 + ia) * nb1 +
                                                            eb->addr] += g * t[ia];
                                                }
                                        }
                                }
                        }

                }
        }
        return GAS_BATCH_DONE;
}

static gas_batch_status_t contract_abba_batch_alpha_range(
        const gas_space_t *gas,
        const double *restrict gos,
        const double *restrict ci0,
        double *restrict ci1,
        uint32_t nnorb, gas_bid_t bsrc,
        gas_sid_t adst, gas_tid_t tida,
        uint32_t q0, uint32_t q1,
        const gas_sid_t *restrict beta_sid,
        gas_tid_t beta_off, uint32_t nbeta,
        gas_contract_ws_t *ws)
{
        return contract_abba_batch_alpha_tile(
                gas, gos, ci0, ci1, nnorb, bsrc, adst, tida,
                q0, q1, 0u, gas->sector_nstr[adst],
                beta_sid, beta_off, nbeta, ws);
}

static gas_batch_status_t contract_abba_batch_alpha(
        const gas_space_t *gas,
        const double *restrict gos,
        const double *restrict ci0,
        double *restrict ci1,
        uint32_t nnorb, gas_bid_t bsrc,
        gas_sid_t adst, gas_tid_t tida,
        const gas_sid_t *restrict beta_sid,
        gas_tid_t beta_off, uint32_t nbeta,
        gas_contract_ws_t *ws)
{
        gas_row_t brow = gas->D.by_alpha_row[adst];
        return contract_abba_batch_alpha_range(gas, gos, ci0, ci1, nnorb,
                                               bsrc, adst, tida, 0, brow.n,
                                               beta_sid, beta_off, nbeta, ws);
}

static void contract_abba_alpha_fallback(
        const gas_space_t *gas,
        const double *restrict eri,
        const double *restrict ci0,
        double *restrict ci1,
        uint32_t nnorb, gas_bid_t bsrc,
        gas_sid_t adst, gas_tid_t tida,
        const gas_sid_t *restrict beta_sid,
        gas_tid_t beta_off, uint32_t nbeta)
{
        for (uint32_t j = 0; j < nbeta; j++) {
                gas_bid_t bdst = gas_find_block(gas, adst, beta_sid[j]);
                if (bdst != GAS_INVALID_BID) {
                        contract_abba_path(gas, eri, ci0, ci1, nnorb,
                                           bsrc, bdst, tida, beta_off + j);
                }
        }
}

static void contract_abba_source_batch(const gas_space_t *gas,
                                       const double *restrict eri,
                                       const double *restrict gos,
                                       const double *restrict ci0,
                                       double *restrict ci1,
                                       uint32_t nnorb, gas_bid_t bsrc,
                                       gas_contract_ws_t *ws)
{
        const gas_block_t *blk = gas->block + bsrc;
        gas_row_t ar = gas->T.row[blk->sa];
        gas_row_t br = gas->T.row[blk->sb];
        const gas_sid_t *as = gas->T.dst + ar.off;
        const gas_sid_t *bs = gas->T.dst + br.off;

        for (uint32_t i = 0; i < ar.n; i++) {
                gas_tid_t tida = ar.off + i;
                gas_batch_status_t status = contract_abba_batch_alpha(
                        gas, gos, ci0, ci1, nnorb, bsrc,
                        as[i], tida, bs, br.off, br.n, ws);
                if (status == GAS_BATCH_FALLBACK) {
                        contract_abba_alpha_fallback(gas, eri, ci0, ci1, nnorb,
                                                     bsrc, as[i], tida,
                                                     bs, br.off, br.n);
                }
        }
}

static void contract_2e_block(const gas_space_t *gas,
                              const double *restrict eri,
                              const double *restrict gos,
                              const double *restrict ci0,
                              double *restrict ci1,
                              uint32_t nnorb, gas_bid_t b,
                              gas_contract_ws_t *ws)
{
        const gas_block_t *blk = gas->block + b;
        gas_row_t ar = gas->T.row[blk->sa];
        gas_row_t br = gas->T.row[blk->sb];
        const gas_sid_t *as = gas->T.dst + ar.off;
        const gas_sid_t *bs = gas->T.dst + br.off;

        if (gas->D.by_beta_row[blk->sb].n < aa_forward_scan_cost(gas, as, ar.n)) {
                contract_aa_by_dst(gas, eri, ci0, ci1, nnorb, b);
        } else {
                contract_aa_forward(gas, eri, ci0, ci1, nnorb, b);
        }

        if (gas->D.by_alpha_row[blk->sa].n < bb_forward_scan_cost(gas, bs, br.n)) {
                contract_bb_by_dst(gas, eri, ci0, ci1, nnorb, b, ws);
        } else {
                contract_bb_forward(gas, eri, ci0, ci1, nnorb, b);
        }

        contract_abba_source_batch(gas, eri, gos, ci0, ci1, nnorb, b, ws);
}

/* ========================================================================== */
/* 5. Destination alpha-sector traversal                                      */
/* ========================================================================== */

uint32_t fci_contract_gas_parallel_units(const gas_space_t *gas)
{
        uint32_t n = 0;

        if (gas == 0) {
                return 0;
        }
        for (gas_sid_t s = 0; s < gas->nsector; s++) {
                n += gas->D.by_alpha_row[s].n != 0;
        }
        return n;
}

#ifdef _OPENMP
static uint64_t cost_add_sat(uint64_t a, uint64_t b)
{
        return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static uint64_t cost_mul_sat(uint64_t a, uint64_t b)
{
        return a != 0u && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

static int append_omp_task(gas_omp_task_t **task, uint32_t *n, uint32_t *cap,
                           gas_sid_t adst, uint32_t q0, uint32_t q1,
                           uint32_t a0, uint32_t a1, uint64_t cost)
{
        if (*n == *cap) {
                uint32_t next = *cap ? *cap * 2u : 1024u;
                if (next < *cap) return GAS_ERR_MEMORY;
                gas_omp_task_t *p = realloc(
                        *task, (size_t)next * sizeof(*p));
                if (p == 0) return GAS_ERR_MEMORY;
                *task = p;
                *cap = next;
        }
        (*task)[*n].adst = adst;
        (*task)[*n].q0 = q0;
        (*task)[*n].q1 = q1;
        (*task)[*n].a0 = a0;
        (*task)[*n].a1 = a1;
        (*task)[*n].cost = cost;
        (*n)++;
        return GAS_SUCCESS;
}

static uint64_t destination_block_cost(const gas_space_t *gas, gas_bid_t bdst)
{
        const gas_block_t *bd = gas->block + bdst;
        const uint64_t na1 = gas->sector_nstr[bd->sa];
        const uint64_t nb1 = gas->sector_nstr[bd->sb];
        uint64_t cost = cost_mul_sat(na1, nb1); /* destination zero/write */

        /* AA: all legal sources with the same beta sector, intersected by the
         * directed intermediate alpha sector. */
        gas_row_t arr = gas->R.row[bd->sa];
        gas_row_t asrc_row = gas->D.by_beta_row[bd->sb];
        for (uint32_t q = 0; q < asrc_row.n; q++) {
                gas_sid_t asrc = gas->D.by_beta_sid[asrc_row.off + q];
                gas_row_t tr = gas->T.row[asrc];
                uint32_t i = 0, j = 0;
                while (i < tr.n && j < arr.n) {
                        gas_sid_t x = gas->T.dst[tr.off + i];
                        gas_sid_t y = gas->R.src[arr.off + j];
                        if (x < y) {
                                i++;
                        } else if (x > y) {
                                j++;
                        } else {
                                const gas_link_table_t *t1 = gas->table + tr.off + i;
                                const gas_link_table_t *t2 = gas->table + gas->R.tid[arr.off + j];
                                uint64_t w = cost_mul_sat(t1->nsrc, t1->nlink);
                                w = cost_mul_sat(w, t2->nlink);
                                w = cost_mul_sat(w, nb1);
                                cost = cost_add_sat(cost, w);
                                i++;
                                j++;
                        }
                }
        }

        /* BB: same alpha sector, beta common-middle intersections. */
        gas_row_t brr = gas->R.row[bd->sb];
        gas_row_t same_alpha = gas->D.by_alpha_row[bd->sa];
        for (uint32_t q = 0; q < same_alpha.n; q++) {
                gas_sid_t bsrc = gas->block[same_alpha.off + q].sb;
                gas_row_t tr = gas->T.row[bsrc];
                uint32_t i = 0, j = 0;
                while (i < tr.n && j < brr.n) {
                        gas_sid_t x = gas->T.dst[tr.off + i];
                        gas_sid_t y = gas->R.src[brr.off + j];
                        if (x < y) {
                                i++;
                        } else if (x > y) {
                                j++;
                        } else {
                                const gas_link_table_t *t1 = gas->table + tr.off + i;
                                const gas_link_table_t *t2 = gas->table + gas->R.tid[brr.off + j];
                                uint64_t w = cost_mul_sat(t1->nsrc, t1->nlink);
                                w = cost_mul_sat(w, t2->nlink);
                                w = cost_mul_sat(w, na1);
                                cost = cost_add_sat(cost, w);
                                i++;
                                j++;
                        }
                }
        }

        /* AB/BA: direct alpha and beta sector transfers.  The alpha gather is
         * included in each destination weight; this deliberately errs on the
         * side of splitting expensive, high-degree destination blocks. */
        for (uint32_t i = 0; i < arr.n; i++) {
                gas_sid_t asrc = gas->R.src[arr.off + i];
                const gas_link_table_t *ta = gas->table + gas->R.tid[arr.off + i];
                gas_row_t srow = gas->D.by_alpha_row[asrc];
                for (uint32_t q = 0; q < srow.n; q++) {
                        gas_sid_t bsrc = gas->block[srow.off + q].sb;
                        gas_tid_t tidb = gas_find_table(gas, bsrc, bd->sb);
                        if (tidb != GAS_INVALID_TID) {
                                const gas_link_table_t *tb = gas->table + tidb;
                                uint64_t nb0 = gas->sector_nstr[bsrc];
                                uint64_t gather = cost_mul_sat(ta->nsrc, ta->nlink);
                                gather = cost_mul_sat(gather, nb0);
                                uint64_t scatter = cost_mul_sat(tb->nsrc, tb->nlink);
                                scatter = cost_mul_sat(scatter, ta->nop ? ta->nop : 1u);
                                scatter = cost_mul_sat(scatter, na1);
                                cost = cost_add_sat(cost, gather);
                                cost = cost_add_sat(cost, scatter);
                                cost = cost_add_sat(cost, cost_mul_sat(na1, nb1));
                        }
                }
        }
        return cost;
}

static gas_omp_task_t *build_omp_tasks(const gas_space_t *gas, uint32_t *ntask)
{
        gas_omp_task_t *task = 0;
        uint32_t n = 0;
        uint32_t cap = 0;
        uint64_t total_cost = 0;
        uint32_t nthread = (uint32_t)omp_get_max_threads();
        uint64_t *block_cost = malloc(
                (size_t)gas->nblock * sizeof(*block_cost));

        if (block_cost == 0) {
                *ntask = 0;
                return 0;
        }
        for (gas_bid_t b = 0; b < gas->nblock; b++) {
                block_cost[b] = destination_block_cost(gas, b);
                total_cost = cost_add_sat(total_cost, block_cost[b]);
        }
        uint64_t target = total_cost /
                ((uint64_t)(nthread ? nthread : 1u) * OMP_TASK_FACTOR);
        if (target < 1u) target = 1u;

        for (gas_sid_t s = 0; s < gas->nsector; s++) {
                gas_row_t row = gas->D.by_alpha_row[s];
                uint32_t na = gas->sector_nstr[s];
                if (row.n == 0 || na == 0u) continue;
                uint32_t q0 = 0;
                uint64_t chunk = 0;

                for (uint32_t q = 0; q < row.n; q++) {
                        uint64_t w = block_cost[row.off + q];
                        uint64_t want64 = (w + target - 1u) / target;
                        uint32_t by_rows =
                                (na + OMP_MIN_ALPHA_TILE - 1u) /
                                OMP_MIN_ALPHA_TILE;
                        uint32_t split = want64 > UINT32_MAX ?
                                         UINT32_MAX : (uint32_t)want64;
                        if (split > OMP_MAX_ALPHA_SPLIT) {
                                split = OMP_MAX_ALPHA_SPLIT;
                        }
                        if (split > by_rows) split = by_rows;

                        if (split > 1u) {
                                if (q > q0 && append_omp_task(
                                        &task, &n, &cap, s, q0, q, 0u, na,
                                        chunk) != GAS_SUCCESS) {
                                        goto fail;
                                }
                                uint32_t a0 = 0;
                                for (uint32_t part = 0; part < split; part++) {
                                        uint32_t left = na - a0;
                                        uint32_t remain = split - part;
                                        uint32_t width =
                                                (left + remain - 1u) / remain;
                                        uint32_t a1 = a0 + width;
                                        uint64_t task_cost =
                                                cost_mul_sat(w, width);
                                        task_cost = na ? task_cost / na : w;
                                        if (task_cost == 0u) task_cost = 1u;
                                        if (append_omp_task(
                                                &task, &n, &cap, s,
                                                q, q + 1u, a0, a1,
                                                task_cost) != GAS_SUCCESS) {
                                                goto fail;
                                        }
                                        a0 = a1;
                                }
                                q0 = q + 1u;
                                chunk = 0;
                                continue;
                        }

                        if (q > q0 && chunk != 0u &&
                            cost_add_sat(chunk, w) > target) {
                                if (append_omp_task(&task, &n, &cap,
                                        s, q0, q, 0u, na,
                                        chunk) != GAS_SUCCESS) {
                                        goto fail;
                                }
                                q0 = q;
                                chunk = 0;
                        }
                        chunk = cost_add_sat(chunk, w);
                }
                if (q0 < row.n && append_omp_task(
                        &task, &n, &cap, s, q0, row.n, 0u, na,
                        chunk) != GAS_SUCCESS) {
                        goto fail;
                }
        }
        free(block_cost);
        if (n > 1u) {
                qsort(task, n, sizeof(*task), cmp_omp_task_desc);
        }
        *ntask = n;
        return task;

fail:
        free(block_cost);
        free(task);
        *ntask = 0;
        return 0;
}

uint32_t fci_contract_gas_omp_task_count(const gas_space_t *gas)
{
        uint32_t n = 0;
#ifdef _OPENMP
        gas_omp_task_t *task;
        if (gas == 0) return 0;
        task = build_omp_tasks(gas, &n);
        free(task);
#else
        (void)gas;
#endif
        return n;
}
static void zero_alpha_range(const gas_space_t *gas, double *ci1,
                             gas_sid_t adst, uint32_t q0, uint32_t q1,
                             uint32_t a0, uint32_t a1)
{
        gas_row_t row = gas->D.by_alpha_row[adst];
        uint32_t na = gas->sector_nstr[adst];
        if (q0 > row.n) q0 = row.n;
        if (q1 > row.n) q1 = row.n;
        if (q1 < q0) q1 = q0;
        if (a0 > na) a0 = na;
        if (a1 > na) a1 = na;
        if (a1 <= a0) return;
        for (uint32_t q = q0; q < q1; q++) {
                gas_bid_t b = row.off + q;
                uint32_t nb = gas->sector_nstr[gas->block[b].sb];
                contract_zero(ci1 + gas->block[b].offset + (uint64_t)a0 * nb,
                              (a1 - a0) * nb);
        }
}

static void contract_aa_alpha_range(const gas_space_t *gas,
                                    const double *restrict eri,
                                    const double *restrict ci0,
                                    double *restrict ci1,
                                    uint32_t nnorb, gas_sid_t adst,
                                    uint32_t q0, uint32_t q1,
                                    uint32_t a0, uint32_t a1,
                                    gas_contract_ws_t *ws)
{
        gas_row_t drow = gas->D.by_alpha_row[adst];
        gas_row_t rr = gas->R.row[adst];
        const gas_sid_t *middle_rev = gas->R.src + rr.off;
        const gas_tid_t *second_tid = gas->R.tid + rr.off;
        uint32_t na = gas->sector_nstr[adst];

        if (q0 > drow.n) q0 = drow.n;
        if (q1 > drow.n) q1 = drow.n;
        if (q1 < q0) q1 = q0;
        if (a0 > na) a0 = na;
        if (a1 > na) a1 = na;

        for (uint32_t d = q0; d < q1; d++) {
                gas_bid_t bdst = drow.off + d;
                gas_sid_t beta = gas->block[bdst].sb;
                gas_row_t srow = gas->D.by_beta_row[beta];
                const gas_sid_t *alpha_src = gas->D.by_beta_sid + srow.off;
                const gas_bid_t *source_block = gas->D.by_beta_bid + srow.off;

                for (uint32_t q = 0; q < srow.n; q++) {
                        gas_row_t tr = gas->T.row[alpha_src[q]];
                        const gas_sid_t *middle_fwd = gas->T.dst + tr.off;
                        uint32_t i = 0;
                        uint32_t j = 0;

                        while (i < tr.n && j < rr.n) {
                                if (middle_fwd[i] < middle_rev[j]) {
                                        i++;
                                } else if (middle_fwd[i] > middle_rev[j]) {
                                        j++;
                                } else {
                                        if (ws == 0 ||
                                            contract_aa_path_rows_accum(
                                                    gas, eri, ci0, ci1, nnorb,
                                                    source_block[q], bdst,
                                                    tr.off + i, second_tid[j],
                                                    a0, a1, ws) != GAS_SUCCESS) {
                                                contract_aa_path_rows(
                                                        gas, eri, ci0, ci1, nnorb,
                                                        source_block[q], bdst,
                                                        tr.off + i, second_tid[j], a0, a1);
                                        }
                                        i++;
                                        j++;
                                }
                        }
                }
        }
}

static void contract_bb_alpha_range(const gas_space_t *gas,
                                    const double *restrict eri,
                                    const double *restrict ci0,
                                    double *restrict ci1,
                                    uint32_t nnorb, gas_sid_t alpha,
                                    uint32_t q0, uint32_t q1,
                                    uint32_t a0, uint32_t a1,
                                    gas_contract_ws_t *ws)
{
        gas_row_t row = gas->D.by_alpha_row[alpha];
        uint32_t na = gas->sector_nstr[alpha];
        if (q0 > row.n) q0 = row.n;
        if (q1 > row.n) q1 = row.n;
        if (q1 < q0) q1 = q0;
        if (a0 > na) a0 = na;
        if (a1 > na) a1 = na;

        for (uint32_t d = q0; d < q1; d++) {
                gas_bid_t bdst = row.off + d;
                gas_sid_t b2 = gas->block[bdst].sb;
                gas_row_t rr = gas->R.row[b2];
                const gas_sid_t *rs = gas->R.src + rr.off;
                const gas_tid_t *rt = gas->R.tid + rr.off;
                uint32_t nb2 = gas->sector_nstr[b2];

                for (uint32_t q = 0; q < row.n; q++) {
                        gas_bid_t bsrc = row.off + q;
                        gas_sid_t b0 = gas->block[bsrc].sb;
                        gas_row_t br = gas->T.row[b0];
                        const gas_sid_t *bs = gas->T.dst + br.off;
                        uint32_t nb0 = gas->sector_nstr[b0];
                        uint32_t maxpair = br.n < rr.n ? br.n : rr.n;
                        uint32_t i = 0;
                        uint32_t j = 0;
                        uint32_t nat = a1 > a0 ? a1 - a0 : 0u;

                        if (use_bb_transpose(nat, nb0, nb2, maxpair) &&
                            reserve_pairs(ws, maxpair) == GAS_SUCCESS) {
                                uint32_t npair = 0;
                                while (i < br.n && j < rr.n) {
                                        if (bs[i] < rs[j]) {
                                                i++;
                                        } else if (bs[i] > rs[j]) {
                                                j++;
                                        } else {
                                                ws->pair[npair].first = br.off + i;
                                                ws->pair[npair].second = rt[j];
                                                npair++;
                                                i++;
                                                j++;
                                        }
                                }
                                if (npair != 0) {
                                        contract_bb_blockpair_rows(
                                                gas, eri, ci0, ci1, nnorb,
                                                bsrc, bdst, ws->pair, npair, ws,
                                                a0, a1);
                                }
                                continue;
                        }

                        while (i < br.n && j < rr.n) {
                                if (bs[i] < rs[j]) {
                                        i++;
                                } else if (bs[i] > rs[j]) {
                                        j++;
                                } else {
                                        contract_bb_path_rows(
                                                gas, eri, ci0, ci1, nnorb,
                                                bsrc, bdst, br.off + i, rt[j], a0, a1);
                                        i++;
                                        j++;
                                }
                        }
                }
        }
}

static void contract_abba_alpha_fallback_range(
        const gas_space_t *gas,
        const double *restrict eri,
        const double *restrict ci0,
        double *restrict ci1,
        uint32_t nnorb, gas_bid_t bsrc,
        gas_sid_t adst, gas_tid_t tida,
        uint32_t q0, uint32_t q1,
        uint32_t a0, uint32_t a1,
        const gas_sid_t *restrict beta_sid,
        gas_tid_t beta_off, uint32_t nbeta)
{
        gas_row_t brow = gas->D.by_alpha_row[adst];
        if (q0 > brow.n) q0 = brow.n;
        if (q1 > brow.n) q1 = brow.n;
        if (q1 < q0) q1 = q0;
        for (uint32_t q = q0; q < q1; q++) {
                gas_bid_t bdst = brow.off + q;
                gas_sid_t b1 = gas->block[bdst].sb;
                gas_tid_t tidb = find_table_in_row(b1, beta_sid, beta_off, nbeta);
                if (tidb != GAS_INVALID_TID) {
                        contract_abba_path_rows(gas, eri, ci0, ci1, nnorb,
                                                bsrc, bdst, tida, tidb, a0, a1);
                }
        }
}

static void contract_abba_alpha_range(const gas_space_t *gas,
                                      const double *restrict eri,
                                      const double *restrict gos,
                                      const double *restrict ci0,
                                      double *restrict ci1,
                                      uint32_t nnorb, gas_sid_t adst,
                                      uint32_t q0, uint32_t q1,
                                      uint32_t a0, uint32_t a1,
                                      gas_contract_ws_t *ws)
{
        gas_row_t ar = gas->R.row[adst];
        const gas_sid_t *alpha_src = gas->R.src + ar.off;
        const gas_tid_t *alpha_tid = gas->R.tid + ar.off;
        gas_row_t drow = gas->D.by_alpha_row[adst];
        uint32_t na = gas->sector_nstr[adst];

        if (q0 > drow.n) q0 = drow.n;
        if (q1 > drow.n) q1 = drow.n;
        if (q1 < q0) q1 = q0;
        if (a0 > na) a0 = na;
        if (a1 > na) a1 = na;

        for (uint32_t i = 0; i < ar.n; i++) {
                gas_row_t srow = gas->D.by_alpha_row[alpha_src[i]];

                for (uint32_t q = 0; q < srow.n; q++) {
                        gas_bid_t bsrc = srow.off + q;
                        gas_sid_t beta0 = gas->block[bsrc].sb;
                        gas_row_t br = gas->T.row[beta0];
                        const gas_sid_t *beta_dst = gas->T.dst + br.off;
                        gas_batch_status_t status;

                        status = contract_abba_batch_alpha_tile(
                                gas, gos, ci0, ci1, nnorb, bsrc,
                                adst, alpha_tid[i], q0, q1, a0, a1,
                                beta_dst, br.off, br.n, ws);

                        if (status == GAS_BATCH_FALLBACK) {
                                contract_abba_alpha_fallback_range(
                                        gas, eri, ci0, ci1, nnorb, bsrc,
                                        adst, alpha_tid[i], q0, q1, a0, a1,
                                        beta_dst, br.off, br.n);
                        }
                }
        }
}

static void contract_2e_alpha_range(const gas_space_t *gas,
                                    const double *restrict eri,
                                    const double *restrict gos,
                                    const double *restrict ci0,
                                    double *restrict ci1,
                                    uint32_t nnorb, gas_sid_t adst,
                                    uint32_t q0, uint32_t q1,
                                    uint32_t a0, uint32_t a1,
                                    gas_contract_ws_t *ws)
{
        contract_aa_alpha_range(gas, eri, ci0, ci1, nnorb,
                                adst, q0, q1, a0, a1, ws);
        contract_bb_alpha_range(gas, eri, ci0, ci1, nnorb,
                                adst, q0, q1, a0, a1, ws);
        contract_abba_alpha_range(gas, eri, gos, ci0, ci1, nnorb,
                                  adst, q0, q1, a0, a1, ws);
}

#endif

#ifndef _OPENMP
uint32_t fci_contract_gas_omp_task_count(const gas_space_t *gas)
{
        (void)gas;
        return 1u;
}
#endif

/* ========================================================================== */
/* 6. Reusable contraction plan                                               */
/* ========================================================================== */

static int contract_plan_config_matches(const gas_contract_plan_t *plan)
{
        return plan->abba_t1_target_bytes == gas_abba_t1_target_bytes;
}

static int contract_plan_resize_workspaces(gas_contract_plan_t *plan,
                                           uint32_t need)
{
        if (need <= plan->nworkspace) {
                return GAS_SUCCESS;
        }
        gas_contract_ws_t *p = realloc(
                plan->workspace, (size_t)need * sizeof(*p));
        if (p == 0) {
                return GAS_ERR_MEMORY;
        }
        plan->workspace = p;
        for (uint32_t i = plan->nworkspace; i < need; i++) {
                contract_ws_init(plan->workspace + i);
        }
        plan->nworkspace = need;
        return GAS_SUCCESS;
}

#ifdef _OPENMP
static int contract_plan_rebuild_tasks(gas_contract_plan_t *plan)
{
        uint32_t nthread = (uint32_t)omp_get_max_threads();
        if (nthread == 0u) nthread = 1u;
        if (contract_plan_resize_workspaces(plan, nthread) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        if (plan->task != 0 && plan->task_threads == nthread) {
                return GAS_SUCCESS;
        }

        free(plan->task);
        plan->task = build_omp_tasks(plan->gas, &plan->ntask);
        plan->task_threads = nthread;
        if (plan->task == 0 && plan->gas->nblock != 0u) {
                plan->ntask = 0u;
                return GAS_ERR_MEMORY;
        }
        return GAS_SUCCESS;
}

static void contract_gas_2e_parallel_plan(gas_contract_plan_t *plan,
                                          const double *ci0, double *ci1,
                                          uint32_t nnorb)
{
        const gas_space_t *gas = plan->gas;
        const double *abba_eri = plan->gos;

#pragma omp parallel
        {
                uint32_t tid = (uint32_t)omp_get_thread_num();
                gas_contract_ws_t *ws = plan->workspace + tid;

#pragma omp for schedule(dynamic, 1)
                for (int64_t tt = 0; tt < (int64_t)plan->ntask; tt++) {
                        gas_sid_t adst = plan->task[tt].adst;
                        uint32_t q0 = plan->task[tt].q0;
                        uint32_t q1 = plan->task[tt].q1;
                        uint32_t a0 = plan->task[tt].a0;
                        uint32_t a1 = plan->task[tt].a1;
                        zero_alpha_range(gas, ci1, adst, q0, q1, a0, a1);
                        contract_2e_alpha_range(gas, plan->eri, abba_eri,
                                                ci0, ci1, nnorb, adst,
                                                q0, q1, a0, a1, ws);
                }
        }
}
#endif

int fci_contract_gas_plan_create(gas_contract_plan_t **out,
                                const gas_space_t *gas,
                                const double *eri,
                                const double *gos)
{
        if (out == 0 || gas == 0 || eri == 0 || gos == 0 ||
            gas->link_format != GAS_LINK_COMPRESSED) {
                return GAS_ERR_INVALID;
        }
        *out = 0;

        gas_contract_plan_t *plan = calloc(1, sizeof(*plan));
        if (plan == 0) {
                return GAS_ERR_MEMORY;
        }
        plan->gas = gas;
        plan->eri = eri;
        plan->gos = gos;
        plan->abba_t1_target_bytes = gas_abba_t1_target_bytes;

#ifdef _OPENMP
        if (contract_plan_rebuild_tasks(plan) != GAS_SUCCESS) {
                fci_contract_gas_plan_free(plan);
                return GAS_ERR_MEMORY;
        }
#else
        if (contract_plan_resize_workspaces(plan, 1u) != GAS_SUCCESS) {
                fci_contract_gas_plan_free(plan);
                return GAS_ERR_MEMORY;
        }
#endif
        *out = plan;
        return GAS_SUCCESS;
}

int fci_contract_gas_plan_execute(gas_contract_plan_t *plan,
                                 const double *ci0, double *ci1)
{
        if (plan == 0 || ci0 == 0 || ci1 == 0 || ci0 == ci1 ||
            !contract_plan_config_matches(plan)) {
                return GAS_ERR_INVALID;
        }
        const gas_space_t *gas = plan->gas;
        uint32_t nnorb = (uint32_t)(gas->norb_tot * (gas->norb_tot + 1) / 2);
        const double *abba_eri = plan->gos;

#ifdef _OPENMP
        if (contract_plan_rebuild_tasks(plan) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        if (omp_get_max_threads() > 1 && plan->ntask > 1u) {
                contract_gas_2e_parallel_plan(plan, ci0, ci1, nnorb);
                return GAS_SUCCESS;
        }
#endif
        if (contract_plan_resize_workspaces(plan, 1u) != GAS_SUCCESS) {
                return GAS_ERR_MEMORY;
        }
        contract_zero(ci1, gas->ndet);
        for (gas_bid_t b = 0; b < gas->nblock; b++) {
                contract_2e_block(gas, plan->eri, abba_eri,
                                  ci0, ci1, nnorb, b, plan->workspace);
        }
        return GAS_SUCCESS;
}

void fci_contract_gas_plan_free(gas_contract_plan_t *plan)
{
        if (plan == 0) {
                return;
        }
        for (uint32_t i = 0; i < plan->nworkspace; i++) {
                contract_ws_free(plan->workspace + i);
        }
        free(plan->workspace);
#ifdef _OPENMP
        free(plan->task);
#endif
        free(plan);
}

uint32_t fci_contract_gas_plan_task_count(const gas_contract_plan_t *plan)
{
        if (plan == 0) return 0u;
#ifdef _OPENMP
        return plan->ntask;
#else
        return 1u;
#endif
}

uint64_t fci_contract_gas_plan_workspace_bytes(const gas_contract_plan_t *plan)
{
        uint64_t n = 0;
        if (plan == 0) return 0u;
        for (uint32_t i = 0; i < plan->nworkspace; i++) {
                n += contract_ws_bytes(plan->workspace + i);
        }
        return n;
}
