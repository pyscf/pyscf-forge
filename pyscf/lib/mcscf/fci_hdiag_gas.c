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
        HDIAG_POTENTIAL_BYTES = 64u * 1024u * 1024u,
        HDIAG_TASK_FACTOR = 32u,
        HDIAG_CACHE_CHUNK = 128u
};

typedef struct {
        uint64_t *mask;
        double *spin_diag;
        double *beta_potential;
} hdiag_sector_cache_t;

typedef struct {
        gas_bid_t block;
        uint32_t alpha_begin;
        uint32_t alpha_end;
} hdiag_task_t;

typedef struct {
        gas_sid_t sector;
        uint16_t orbital;
        uint32_t begin;
        uint32_t end;
} hdiag_cache_task_t;

typedef struct {
        hdiag_cache_task_t *mask_task;
        hdiag_cache_task_t *potential_task;
        uint32_t mask_task_count;
        uint32_t potential_task_count;
} hdiag_cache_work_t;

static inline int ctz64(uint64_t x)
{
        return __builtin_ctzll(x);
}

static inline double coulomb_value(const double *restrict eri,
                                   uint32_t npair, int p, int q)
{
        uint16_t pp = gas_link_pair_index(p, p);
        uint16_t qq = gas_link_pair_index(q, q);
        return eri[(size_t)pp * npair + qq];
}

static inline double exchange_value(const double *restrict eri,
                                    uint32_t npair, int p, int q)
{
        uint16_t pq = gas_link_pair_index(p, q);
        return eri[(size_t)pq * npair + pq];
}

static int memory_ranges_overlap(const void *left, size_t left_bytes,
                                 const void *right, size_t right_bytes)
{
        uintptr_t a = (uintptr_t)left;
        uintptr_t b = (uintptr_t)right;
        if (left_bytes > UINTPTR_MAX - a || right_bytes > UINTPTR_MAX - b) {
                return 1;
        }
        return a < b + right_bytes && b < a + left_bytes;
}

static int hdiag_validate(const gas_space_t *gas, const double *h1e,
                          const double *eri, const double *hdiag)
{
        if (gas == 0 || h1e == 0 || eri == 0 || hdiag == 0) {
                return GAS_ERR_INVALID;
        }
        if (gas->ngas <= 0 || gas->ngas > GAS_MAX_NGAS ||
            gas->norb_tot <= 0 || gas->norb_tot > GAS_MAX_ORB ||
            gas->nsector == 0u || gas->nblock == 0u || gas->ndet == 0u ||
            gas->norb == 0 || gas->start == 0 ||
            gas->sector_nstr == 0 || gas->sector_occ == 0 ||
            gas->sector_stride == 0 || gas->block == 0) {
                return GAS_ERR_INVALID;
        }
        if (gas->link_format != GAS_LINK_RAW &&
            gas->link_format != GAS_LINK_COMPRESSED) {
                return GAS_ERR_INVALID;
        }

        size_t norb = (size_t)gas->norb_tot;
        size_t npair = norb * (norb + 1u) / 2u;
        size_t output_bytes = (size_t)gas->ndet * sizeof(*hdiag);
        if (memory_ranges_overlap(hdiag, output_bytes,
                                  h1e, norb * norb * sizeof(*h1e)) ||
            memory_ranges_overlap(hdiag, output_bytes,
                                  eri, npair * npair * sizeof(*eri))) {
                return GAS_ERR_INVALID;
        }
        return GAS_SUCCESS;
}

static void free_sector_cache(hdiag_sector_cache_t *cache, uint32_t nsector)
{
        if (cache == 0) return;
        for (uint32_t s = 0; s < nsector; s++) {
                free(cache[s].mask);
                free(cache[s].spin_diag);
                free(cache[s].beta_potential);
        }
        free(cache);
}

static void free_cache_work(hdiag_cache_work_t *work)
{
        free(work->mask_task);
        free(work->potential_task);
        memset(work, 0, sizeof(*work));
}

static double spin_diagonal(uint64_t mask, const double *restrict h1e,
                            const double *restrict eri,
                            uint32_t norb, uint32_t npair)
{
        double value = 0.0;
        uint64_t remaining = mask;
        while (remaining != 0u) {
                int p = ctz64(remaining);
                remaining &= remaining - 1u;
                value += h1e[(size_t)p * norb + (uint32_t)p];

                uint64_t partners = remaining;
                while (partners != 0u) {
                        int q = ctz64(partners);
                        partners &= partners - 1u;
                        value += coulomb_value(eri, npair, p, q)
                               - exchange_value(eri, npair, p, q);
                }
        }
        return value;
}

static int prepare_sector_cache(const gas_space_t *gas,
                                hdiag_sector_cache_t **out,
                                hdiag_cache_work_t *work)
{
        uint32_t nsector = gas->nsector;
        uint32_t norb = (uint32_t)gas->norb_tot;
        unsigned char *used = calloc((size_t)nsector, sizeof(*used));
        hdiag_sector_cache_t *cache = calloc((size_t)nsector, sizeof(*cache));
        uint64_t nmask64 = 0u;
        uint64_t npotential64 = 0u;
        int status = GAS_ERR_MEMORY;
        if (used == 0 || cache == 0) goto done;

        for (gas_bid_t b = 0; b < gas->nblock; b++) {
                used[gas->block[b].sa] |= 1u;
                used[gas->block[b].sb] |= 2u;
        }

        size_t budget = (size_t)HDIAG_POTENTIAL_BYTES;
        for (uint32_t s = 0; s < nsector; s++) {
                if (used[s] == 0u) continue;
                uint32_t nstr = gas->sector_nstr[s];
                cache[s].mask = malloc((size_t)nstr * sizeof(*cache[s].mask));
                if (cache[s].mask == 0) goto done;
                cache[s].spin_diag = malloc(
                        (size_t)nstr * sizeof(*cache[s].spin_diag));
                if (cache[s].spin_diag == 0) goto done;

                uint64_t chunks = ((uint64_t)nstr + HDIAG_CACHE_CHUNK - 1u) /
                                  HDIAG_CACHE_CHUNK;
                nmask64 += chunks;
                if ((used[s] & 2u) == 0u ||
                    nstr > SIZE_MAX / norb / sizeof(double)) {
                        continue;
                }

                size_t bytes = (size_t)norb * nstr * sizeof(double);
                if (bytes > budget) continue;
                cache[s].beta_potential = malloc(bytes);
                if (cache[s].beta_potential == 0) continue;
                budget -= bytes;
                npotential64 += chunks * norb;
        }

        if (nmask64 == 0u || nmask64 > UINT32_MAX ||
            npotential64 > UINT32_MAX ||
            nmask64 > SIZE_MAX / sizeof(*work->mask_task) ||
            npotential64 > SIZE_MAX / sizeof(*work->potential_task)) {
                goto done;
        }

        work->mask_task = malloc((size_t)nmask64 * sizeof(*work->mask_task));
        if (npotential64 != 0u) {
                work->potential_task = malloc(
                        (size_t)npotential64 * sizeof(*work->potential_task));
        }
        if (work->mask_task == 0 ||
            (npotential64 != 0u && work->potential_task == 0)) {
                goto done;
        }

        uint32_t nmask = 0u;
        uint32_t npotential = 0u;
        for (uint32_t s = 0; s < nsector; s++) {
                if (used[s] == 0u) continue;
                uint32_t nstr = gas->sector_nstr[s];
                for (uint32_t begin = 0u; begin < nstr;
                     begin += HDIAG_CACHE_CHUNK) {
                        uint32_t end = nstr - begin < HDIAG_CACHE_CHUNK ?
                                       nstr : begin + HDIAG_CACHE_CHUNK;
                        work->mask_task[nmask++] = (hdiag_cache_task_t){
                                (gas_sid_t)s, 0u, begin, end};
                        if (cache[s].beta_potential == 0) continue;
                        for (uint32_t p = 0u; p < norb; p++) {
                                work->potential_task[npotential++] =
                                        (hdiag_cache_task_t){
                                                (gas_sid_t)s, (uint16_t)p,
                                                begin, end};
                        }
                }
        }

        work->mask_task_count = nmask;
        work->potential_task_count = npotential;
        *out = cache;
        cache = 0;
        status = GAS_SUCCESS;

done:
        free(used);
        if (status != GAS_SUCCESS) free_cache_work(work);
        free_sector_cache(cache, nsector);
        return status;
}

static void execute_mask_cache_task(const gas_space_t *gas,
                                    const double *restrict h1e,
                                    const double *restrict eri,
                                    hdiag_sector_cache_t *cache,
                                    const hdiag_cache_task_t *task)
{
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t npair = norb * (norb + 1u) / 2u;
        hdiag_sector_cache_t *sector = cache + task->sector;
        for (uint32_t i = task->begin; i < task->end; i++) {
                uint64_t mask = gas_addr2str_sector(gas, task->sector, i);
                sector->mask[i] = mask;
                sector->spin_diag[i] = spin_diagonal(
                        mask, h1e, eri, norb, npair);
        }
}

static void execute_potential_cache_task(const gas_space_t *gas,
                                         const double *restrict eri,
                                         hdiag_sector_cache_t *cache,
                                         const hdiag_cache_task_t *task)
{
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t npair = norb * (norb + 1u) / 2u;
        hdiag_sector_cache_t *sector = cache + task->sector;
        uint32_t nstr = gas->sector_nstr[task->sector];
        double *potential = sector->beta_potential +
                            (size_t)task->orbital * nstr;
        for (uint32_t i = task->begin; i < task->end; i++) {
                uint64_t mask = sector->mask[i];
                double value = 0.0;
                while (mask != 0u) {
                        int q = ctz64(mask);
                        mask &= mask - 1u;
                        value += coulomb_value(eri, npair, task->orbital, q);
                }
                potential[i] = value;
        }
}

static void execute_cache_work(const gas_space_t *gas,
                               const double *restrict h1e,
                               const double *restrict eri,
                               hdiag_sector_cache_t *cache,
                               const hdiag_cache_work_t *work)
{
#ifdef _OPENMP
#pragma omp parallel
        {
#pragma omp for schedule(static)
                for (int64_t k = 0;
                     k < (int64_t)work->mask_task_count; k++) {
                        execute_mask_cache_task(
                                gas, h1e, eri, cache, work->mask_task + k);
                }
#pragma omp for schedule(static)
                for (int64_t k = 0;
                     k < (int64_t)work->potential_task_count; k++) {
                        execute_potential_cache_task(
                                gas, eri, cache, work->potential_task + k);
                }
        }
#else
        for (uint32_t k = 0; k < work->mask_task_count; k++) {
                execute_mask_cache_task(
                        gas, h1e, eri, cache, work->mask_task + k);
        }
        for (uint32_t k = 0; k < work->potential_task_count; k++) {
                execute_potential_cache_task(
                        gas, eri, cache, work->potential_task + k);
        }
#endif
}

static int build_sector_cache(const gas_space_t *gas,
                              const double *restrict h1e,
                              const double *restrict eri,
                              hdiag_sector_cache_t **out)
{
        hdiag_cache_work_t work = {0};
        int status = prepare_sector_cache(gas, out, &work);
        if (status == GAS_SUCCESS) {
                execute_cache_work(gas, h1e, eri, *out, &work);
        }
        free_cache_work(&work);
        return status;
}

static int build_tasks(const gas_space_t *gas,
                       hdiag_task_t **out, uint32_t *task_count)
{
#ifdef _OPENMP
        uint32_t threads = (uint32_t)omp_get_max_threads();
#else
        uint32_t threads = 1u;
#endif
        uint64_t target_tasks = (uint64_t)threads * HDIAG_TASK_FACTOR;
        uint64_t target_dets = ((uint64_t)gas->ndet + target_tasks - 1u) /
                               target_tasks;
        if (target_dets == 0u) target_dets = 1u;

        uint64_t count = 0u;
        for (gas_bid_t b = 0; b < gas->nblock; b++) {
                const gas_block_t *block = gas->block + b;
                uint32_t na = gas->sector_nstr[block->sa];
                uint32_t nb = gas->sector_nstr[block->sb];
                uint64_t rows64 = target_dets / nb;
                uint32_t rows = rows64 == 0u ? 1u :
                                rows64 > na ? na : (uint32_t)rows64;
                count += ((uint64_t)na + rows - 1u) / rows;
        }
        if (count == 0u || count > UINT32_MAX ||
            count > SIZE_MAX / sizeof(hdiag_task_t)) {
                return GAS_ERR_MEMORY;
        }

        hdiag_task_t *task = malloc((size_t)count * sizeof(*task));
        if (task == 0) return GAS_ERR_MEMORY;

        uint32_t n = 0u;
        for (gas_bid_t b = 0; b < gas->nblock; b++) {
                const gas_block_t *block = gas->block + b;
                uint32_t na = gas->sector_nstr[block->sa];
                uint32_t nb = gas->sector_nstr[block->sb];
                uint64_t rows64 = target_dets / nb;
                uint32_t rows = rows64 == 0u ? 1u :
                                rows64 > na ? na : (uint32_t)rows64;
                for (uint32_t a0 = 0; a0 < na; a0 += rows) {
                        uint32_t a1 = na - a0 < rows ? na : a0 + rows;
                        task[n].block = b;
                        task[n].alpha_begin = a0;
                        task[n].alpha_end = a1;
                        n++;
                }
        }

        *out = task;
        *task_count = n;
        return GAS_SUCCESS;
}

static inline void initialize_row(double *restrict y,
                                  const double *restrict beta,
                                  double alpha, uint32_t n)
{
        uint32_t i = 0u;
#if defined(__AVX2__)
        __m256d va = _mm256_set1_pd(alpha);
        for (; i + 4u <= n; i += 4u) {
                __m256d vb = _mm256_loadu_pd(beta + i);
                _mm256_storeu_pd(y + i, _mm256_add_pd(va, vb));
        }
#endif
        uint32_t tail = i;
#ifdef _OPENMP
#pragma omp simd
#endif
        for (uint32_t j = tail; j < n; j++) y[j] = alpha + beta[j];
}

static inline void add_rows8(double *restrict y,
                             const double *restrict x0,
                             const double *restrict x1,
                             const double *restrict x2,
                             const double *restrict x3,
                             const double *restrict x4,
                             const double *restrict x5,
                             const double *restrict x6,
                             const double *restrict x7,
                             uint32_t n)
{
        uint32_t i = 0u;
#if defined(__AVX2__)
        for (; i + 4u <= n; i += 4u) {
                __m256d v = _mm256_loadu_pd(y + i);
                v = _mm256_add_pd(v, _mm256_loadu_pd(x0 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x1 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x2 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x3 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x4 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x5 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x6 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x7 + i));
                _mm256_storeu_pd(y + i, v);
        }
#endif
        uint32_t tail = i;
#ifdef _OPENMP
#pragma omp simd
#endif
        for (uint32_t j = tail; j < n; j++) {
                y[j] += x0[j] + x1[j] + x2[j] + x3[j] +
                        x4[j] + x5[j] + x6[j] + x7[j];
        }
}

static inline void add_rows4(double *restrict y,
                             const double *restrict x0,
                             const double *restrict x1,
                             const double *restrict x2,
                             const double *restrict x3,
                             uint32_t n)
{
        uint32_t i = 0u;
#if defined(__AVX2__)
        for (; i + 4u <= n; i += 4u) {
                __m256d v = _mm256_loadu_pd(y + i);
                v = _mm256_add_pd(v, _mm256_loadu_pd(x0 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x1 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x2 + i));
                v = _mm256_add_pd(v, _mm256_loadu_pd(x3 + i));
                _mm256_storeu_pd(y + i, v);
        }
#endif
        uint32_t tail = i;
#ifdef _OPENMP
#pragma omp simd
#endif
        for (uint32_t j = tail; j < n; j++) {
                y[j] += x0[j] + x1[j] + x2[j] + x3[j];
        }
}

static inline void add_rows2(double *restrict y,
                             const double *restrict x0,
                             const double *restrict x1, uint32_t n)
{
#ifdef _OPENMP
#pragma omp simd
#endif
        for (uint32_t i = 0; i < n; i++) y[i] += x0[i] + x1[i];
}

static inline void add_row(double *restrict y,
                           const double *restrict x, uint32_t n)
{
#ifdef _OPENMP
#pragma omp simd
#endif
        for (uint32_t i = 0; i < n; i++) y[i] += x[i];
}

static inline void set_fused_rows(double *restrict y,
                                  const double *restrict beta,
                                  double alpha,
                                  const double *const restrict *row,
                                  uint32_t nrow, uint32_t n)
{
        switch (nrow) {
        case 8:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i] + row[1][i] +
                               row[2][i] + row[3][i] + row[4][i] + row[5][i] +
                               row[6][i] + row[7][i];
                }
                break;
        case 7:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i] + row[1][i] +
                               row[2][i] + row[3][i] + row[4][i] + row[5][i] +
                               row[6][i];
                }
                break;
        case 6:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i] + row[1][i] +
                               row[2][i] + row[3][i] + row[4][i] + row[5][i];
                }
                break;
        case 5:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i] + row[1][i] +
                               row[2][i] + row[3][i] + row[4][i];
                }
                break;
        case 4:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i] + row[1][i] +
                               row[2][i] + row[3][i];
                }
                break;
        case 3:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i] + row[1][i] +
                               row[2][i];
                }
                break;
        case 2:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i] + row[1][i];
                }
                break;
        case 1:
#ifdef _OPENMP
#pragma omp simd
#endif
                for (uint32_t i = 0; i < n; i++) {
                        y[i] = alpha + beta[i] + row[0][i];
                }
                break;
        default:
                initialize_row(y, beta, alpha, n);
                break;
        }
}

static void fill_alpha_row_potential(const gas_space_t *gas,
                                     const hdiag_sector_cache_t *cache,
                                     const gas_block_t *block,
                                     uint32_t ia, double *restrict y)
{
        uint32_t nb = gas->sector_nstr[block->sb];
        const hdiag_sector_cache_t *alpha = cache + block->sa;
        const hdiag_sector_cache_t *beta = cache + block->sb;
        const double *row[GAS_MAX_ORB];
        uint32_t nrow = 0u;
        uint64_t mask = alpha->mask[ia];
        while (mask != 0u) {
                int p = ctz64(mask);
                mask &= mask - 1u;
                row[nrow++] = beta->beta_potential + (size_t)p * nb;
        }

        uint32_t i = nrow < 8u ? nrow : 8u;
        set_fused_rows(y, beta->spin_diag, alpha->spin_diag[ia], row, i, nb);
        for (; i + 8u <= nrow; i += 8u) {
                add_rows8(y, row[i], row[i + 1u], row[i + 2u], row[i + 3u],
                          row[i + 4u], row[i + 5u], row[i + 6u],
                          row[i + 7u], nb);
        }
        if (i + 4u <= nrow) {
                add_rows4(y, row[i], row[i + 1u], row[i + 2u], row[i + 3u],
                          nb);
                i += 4u;
        }
        if (i + 2u <= nrow) {
                add_rows2(y, row[i], row[i + 1u], nb);
                i += 2u;
        }
        if (i < nrow) add_row(y, row[i], nb);
}

static void fill_alpha_row_direct(const gas_space_t *gas,
                                  const double *restrict eri,
                                  const hdiag_sector_cache_t *cache,
                                  const gas_block_t *block,
                                  uint32_t ia, double *restrict y)
{
        uint32_t norb = (uint32_t)gas->norb_tot;
        uint32_t npair = norb * (norb + 1u) / 2u;
        uint32_t nb = gas->sector_nstr[block->sb];
        const hdiag_sector_cache_t *alpha = cache + block->sa;
        const hdiag_sector_cache_t *beta = cache + block->sb;
        double potential[GAS_MAX_ORB] = {0.0};

        uint64_t alpha_mask = alpha->mask[ia];
        while (alpha_mask != 0u) {
                int p = ctz64(alpha_mask);
                alpha_mask &= alpha_mask - 1u;
                for (uint32_t q = 0; q < norb; q++) {
                        potential[q] += coulomb_value(eri, npair, p, (int)q);
                }
        }

        double alpha_value = alpha->spin_diag[ia];
        for (uint32_t ib = 0; ib < nb; ib++) {
                double value = alpha_value + beta->spin_diag[ib];
                uint64_t beta_mask = beta->mask[ib];
                while (beta_mask != 0u) {
                        int q = ctz64(beta_mask);
                        beta_mask &= beta_mask - 1u;
                        value += potential[q];
                }
                y[ib] = value;
        }
}

static void execute_task(const gas_space_t *gas,
                         const double *restrict eri,
                         const hdiag_sector_cache_t *cache,
                         const hdiag_task_t *task,
                         double *restrict hdiag)
{
        const gas_block_t *block = gas->block + task->block;
        uint32_t nb = gas->sector_nstr[block->sb];
        int use_potential = cache[block->sb].beta_potential != 0;
        for (uint32_t ia = task->alpha_begin; ia < task->alpha_end; ia++) {
                double *row = hdiag + block->offset + (size_t)ia * nb;
                if (use_potential) {
                        fill_alpha_row_potential(gas, cache, block, ia, row);
                } else {
                        fill_alpha_row_direct(gas, eri, cache, block, ia, row);
                }
        }
}

int fci_make_hdiag_gas(const gas_space_t *gas,
                       const double *h1e, const double *eri,
                       double *hdiag)
{
        int status = hdiag_validate(gas, h1e, eri, hdiag);
        if (status != GAS_SUCCESS) return status;

        hdiag_sector_cache_t *cache = 0;
        hdiag_task_t *task = 0;
        uint32_t task_count = 0u;
        status = build_sector_cache(gas, h1e, eri, &cache);
        if (status != GAS_SUCCESS) goto done;

        status = build_tasks(gas, &task, &task_count);
        if (status != GAS_SUCCESS) goto done;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)task_count; i++) {
                execute_task(gas, eri, cache, task + i, hdiag);
        }
#else
        for (uint32_t i = 0; i < task_count; i++) {
                execute_task(gas, eri, cache, task + i, hdiag);
        }
#endif

done:
        free(task);
        free_sector_cache(cache, gas->nsector);
        return status;
}
