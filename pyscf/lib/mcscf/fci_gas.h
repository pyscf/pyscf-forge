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

#ifndef FCI_GAS_H
#define FCI_GAS_H

#include <stdint.h>

enum {
        GAS_MAX_ORB = 63,
        GAS_MAX_LOCAL_ORB = 31,
        GAS_MAX_NGAS = 15
};

#define GAS_INVALID_SID UINT16_MAX
#define GAS_INVALID_BID UINT32_MAX
#define GAS_INVALID_TID UINT32_MAX

typedef uint16_t gas_sid_t;
typedef uint32_t gas_bid_t;
typedef uint32_t gas_tid_t;

typedef enum {
        GAS_SUCCESS = 0,
        GAS_ERR_INVALID = -1,
        GAS_ERR_MEMORY = -2
} gas_status_t;

typedef enum {
        GAS_LINK_RAW = 0,
        GAS_LINK_COMPRESSED = 1
} gas_link_format_t;

/*
 * A legal determinant block couples one alpha sector to one beta sector.
 * Determinants occupy one contiguous alpha-major rectangle in the flattened
 * CI vector: offset + ia * nstr(beta) + ib.
 */
typedef struct {
        uint32_t offset;
        gas_sid_t sa;
        gas_sid_t sb;
} gas_block_t;

/* Slice [off, off+n) in a packed index array. */
typedef struct {
        uint32_t off;
        uint32_t n;
} gas_row_t;

/*
 * Index D for the legal block set.  Alpha rows address gas->block directly;
 * beta rows use the packed sector-ID and block-ID arrays because canonical
 * block order is alpha-major.
 */
typedef struct {
        gas_row_t *by_alpha_row;
        gas_row_t *by_beta_row;
        gas_sid_t *by_beta_sid;
        gas_bid_t *by_beta_bid;
} gas_block_index_t;

/* Forward table index T: source sector -> packed destination sectors. */
typedef struct {
        gas_row_t *row;
        gas_sid_t *dst;
} gas_table_index_t;

/*
 * Reverse table index R: destination sector -> packed (source sector,
 * link-table ID) entries.
 */
typedef struct {
        gas_row_t *row;
        gas_sid_t *src;
        gas_tid_t *tid;
} gas_table_rev_index_t;

/*
 * One spin-string excitation.  addr is the destination string address and
 * sign is the fermionic phase.  In a raw table, op packs creation orbital p
 * in the low byte and destruction orbital q in the high byte.  In a
 * compressed table, op is the lower-triangular pair index for (p,q).
 */
typedef struct {
        uint32_t addr;
        uint16_t op;
        int8_t sign;
        uint8_t padding;             /* completes the 8-byte entry */
} gas_link_entry_t;

/*
 * Directed source-sector -> destination-sector link table.  active_op is the
 * sorted list of triangular operator-pair indices present in the table.
 */
typedef struct {
        gas_link_entry_t *link;
        uint16_t *active_op;
        uint32_t nsrc;
        uint32_t nlink;
        uint16_t nop;
} gas_link_table_t;

/*
 * Reusable H*c execution plan.  A plan caches Hamiltonian-dependent data,
 * OpenMP tasks and per-thread workspaces across Davidson iterations.
 * The referenced gas space, eri array and gos array must outlive the plan.
 * A single plan must not be executed concurrently by multiple host threads.
 */
typedef struct gas_contract_plan gas_contract_plan_t;
typedef struct gas_rdm_plan gas_rdm_plan_t;

typedef struct {
        int ngas;
        int norb_tot;
        int *norb;
        int *start;
        int na;
        int nb;
        uint32_t ndet;
        uint16_t nsector;
        uint16_t sector_padding;     /* 2 + 2 bytes align sector_nstr */
        uint32_t *sector_nstr;
        uint8_t *sector_occ;
        uint32_t *sector_stride;

        uint32_t nblock;
        gas_block_t *block;
        gas_block_index_t D;          /* legal determinant-block index */

        uint32_t ntable;
        uint8_t link_format;
        uint8_t link_format_padding[3]; /* 1 + 3 bytes align table */
        gas_link_table_t *table;
        gas_table_index_t T;          /* forward link-table index */
        gas_table_rev_index_t R;      /* reverse link-table index */
} gas_space_t;

typedef struct {
        uint64_t metadata;
        uint64_t sector;
        uint64_t block;
        uint64_t block_index;
        uint64_t link_table;
        uint64_t link_table_index;
        uint64_t total;
} gas_memory_report_t;

_Static_assert(sizeof(void *) == 8, "fci_gas requires a 64-bit platform");
_Static_assert(sizeof(gas_link_entry_t) == 8, "gas_link_entry_t must be 8 bytes");
_Static_assert(sizeof(gas_block_t) == 8, "gas_block_t must be 8 bytes");
_Static_assert(sizeof(gas_row_t) == 8, "gas_row_t must be 8 bytes");
_Static_assert(sizeof(gas_link_table_t) == 32, "gas_link_table_t must be 32 bytes");
_Static_assert(sizeof(gas_space_t) == 168, "gas_space_t must be 168 bytes");

static inline uint16_t gas_link_pair_index(int p, int q)
{
        uint32_t hi = p > q ? (uint32_t)p : (uint32_t)q;
        uint32_t lo = p > q ? (uint32_t)q : (uint32_t)p;
        return (uint16_t)(hi * (hi + 1u) / 2u + lo);
}

static inline void gas_link_set_raw(gas_link_entry_t *e, int p, int q)
{
        uint32_t packed = (uint32_t)p | ((uint32_t)q << 8);
        e->op = (uint16_t)packed;
}

static inline uint8_t gas_link_cre(const gas_link_entry_t *e)
{
        return (uint8_t)(e->op & 0xffu);
}

static inline uint8_t gas_link_des(const gas_link_entry_t *e)
{
        return (uint8_t)(e->op >> 8);
}

static inline void gas_link_set_ia(gas_link_entry_t *e, uint16_t ia)
{
        e->op = ia;
}

static inline uint16_t gas_link_ia(const gas_link_entry_t *e)
{
        return e->op;
}

uint32_t gas_str2addr_sector(const gas_space_t *gas, gas_sid_t s, uint64_t str);
uint64_t gas_addr2str_sector(const gas_space_t *gas, gas_sid_t s, uint32_t addr);
uint32_t gas_block_ndet(const gas_space_t *gas, gas_bid_t b);
uint32_t gas_det_addr(const gas_space_t *gas, gas_bid_t b, uint32_t ia, uint32_t ib);
gas_bid_t gas_find_block(const gas_space_t *gas, gas_sid_t sa, gas_sid_t sb);
gas_tid_t gas_find_table(const gas_space_t *gas, gas_sid_t src, gas_sid_t dst);
const gas_link_table_t *gas_get_link_table(const gas_space_t *gas,
                                           gas_sid_t src, gas_sid_t dst);

int gas_space_from_blocks(gas_space_t *gas, int ngas, const int *norb,
                          int na, int nb, int nblock, const int *block_occ);
int gas_space_compress_links(gas_space_t *gas);
int gas_space_links_are_compressed(const gas_space_t *gas);
void gas_space_free(gas_space_t *gas);

void gas_memory_report(const gas_space_t *gas, gas_memory_report_t *report);
uint64_t gas_memory_bytes(const gas_space_t *gas);

/*
 * Diagonal of the physical GAS Hamiltonian in the determinant order defined
 * by gas->block.  h1e is a row-major (norb,norb) matrix.  eri is the original
 * (unabsorbed) chemist-notation two-electron integral tensor restored to the
 * four-fold pair-matrix layout (npair,npair), where
 * npair = norb * (norb + 1) / 2.  Raw and compressed link spaces are accepted.
 */
int fci_make_hdiag_gas(const gas_space_t *gas,
                       const double *h1e, const double *eri,
                       double *hdiag);

/* ABBA workspace target; the production default is 64 MiB. */
void fci_contract_gas_set_abba_t1_bytes(uint64_t bytes);
uint32_t fci_contract_gas_omp_task_count(const gas_space_t *gas);

/*
 * Contract the spin-free Hamiltonian after absorbing h1e into the
 * two-electron tensor.  eri_tril is a row-major (npair,npair) array in
 * PySCF's four-fold pair-matrix layout; each orbital pair uses
 * gas_link_pair_index.  ci0 and ci1 follow gas->block determinant order.
 * Contraction requires compressed link tables.
 */
int fci_contract_gas_plan_create(gas_contract_plan_t **out,
                                const gas_space_t *gas,
                                const double *eri_tril,
                                const double *gos);
int fci_contract_gas_plan_execute(gas_contract_plan_t *plan,
                                 const double *ci0, double *ci1);
void fci_contract_gas_plan_free(gas_contract_plan_t *plan);
uint32_t fci_contract_gas_plan_task_count(const gas_contract_plan_t *plan);
uint64_t fci_contract_gas_plan_workspace_bytes(const gas_contract_plan_t *plan);
uint32_t fci_contract_gas_parallel_units(const gas_space_t *gas);

/*
 * GAS reduced density matrices.  These entry points require raw link tables;
 * rebuild the gas space after the Davidson contraction stage rather than
 * passing a space that has been compressed for Hamiltonian contraction.
 *
 * All arrays are row-major.  The conventions match PySCF's public RDMs:
 *
 *   dm1[p,q]       = <bra| q^+ p |ket>
 *   dm2[p,q,r,s]   = <bra| p^+ r^+ s q |ket>
 *
 * dm2ab has alpha spin on the (p,q) operator pair and beta spin on
 * the (r,s) pair.  Spin-free RDMs are assembled from these arrays in
 * the Python layer.
 */

/*
 * Target size for each thread's tiled opposite-spin T1 workspace
 * (16 MiB production default).
 */
void fci_rdm_gas_set_abba_t1_bytes(uint64_t bytes);

/*
 * Reusable RDM plan.  The referenced raw-link gas space must outlive the plan.
 * A plan caches destination-alpha range tasks and grows thread-private RDM
 * workspaces as needed.  It must not be executed concurrently by multiple
 * host threads.  Changing omp_get_max_threads() between calls is supported.
 */
int fci_rdm_gas_plan_create(gas_rdm_plan_t **out, const gas_space_t *gas);
int fci_rdm_gas_plan_make_rdm1s(gas_rdm_plan_t *plan,
                                const double *bra, const double *ket,
                                double *dm1a, double *dm1b);
int fci_rdm_gas_plan_make_rdm12s(gas_rdm_plan_t *plan,
                                 const double *bra, const double *ket,
                                 double *dm1a, double *dm1b,
                                 double *dm2aa, double *dm2ab, double *dm2bb);
void fci_rdm_gas_plan_free(gas_rdm_plan_t *plan);
uint32_t fci_rdm_gas_plan_task_count(const gas_rdm_plan_t *plan);
uint64_t fci_rdm_gas_plan_workspace_bytes(const gas_rdm_plan_t *plan);

#endif /* FCI_GAS_H */
