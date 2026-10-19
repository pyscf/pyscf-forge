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
 * Extended nonorthogonal Wick theorem (ENOWT) matrix elements between
 * arbitrary excitations of nonorthogonal reference determinants.
 *
 * The reference-pair intermediates of Burton, J. Chem. Phys. 154, 144109
 * (2021); 157, 204109 (2022) are built in Python (pyscf/nonorthogonal/
 * _wicks.py) and passed here as one contiguous slab.  Every ordered
 * reference pair (x,w) owns NOWICK_PAIRLEN int64 entries of pairinfo and
 * NOWICK_NSCAL scalars.  Offsets of unallocated tensors are -1.
 */

#include <stdint.h>
#include <complex.h>

/* Largest L_a + L_b with a fixed-rank m_a = m_b = 0 kernel.  This covers
 * determinant pairs with up to quadruple excitations on each side. */
#define NOWICK_MAXL     8

/* Apply F(LA, LB) to every (L_a, L_b) with L_a + L_b <= NOWICK_MAXL */
#define NOWICK_LALB_CASES(F) \
        F(0,0) F(0,1) F(0,2) F(0,3) F(0,4) F(0,5) F(0,6) F(0,7) F(0,8) \
        F(1,0) F(1,1) F(1,2) F(1,3) F(1,4) F(1,5) F(1,6) F(1,7) \
        F(2,0) F(2,1) F(2,2) F(2,3) F(2,4) F(2,5) F(2,6) \
        F(3,0) F(3,1) F(3,2) F(3,3) F(3,4) F(3,5) \
        F(4,0) F(4,1) F(4,2) F(4,3) F(4,4) \
        F(5,0) F(5,1) F(5,2) F(5,3) \
        F(6,0) F(6,1) F(6,2) \
        F(7,0) F(7,1) \
        F(8,0)

/* pairinfo layout */
#define NOWICK_PAIRLEN  96
#define NOWICK_MA       0
#define NOWICK_MB       1
#define NOWICK_NOCCA    2
#define NOWICK_NOCCB    3
#define NOWICK_NMO      4
#define NOWICK_AA       5
#define NOWICK_BB       32
#define NOWICK_AB       59
/* orthogonal (same-parent) MO integrals, diagonal pairs (x,x) only */
#define NOWICK_OH_A     91      /* C_a^+ h C_a */
#define NOWICK_OH_B     92
#define NOWICK_OAA      93      /* (pq|rs) - (ps|rq), alpha */
#define NOWICK_OBB      94
#define NOWICK_OAB      95      /* (pq|rs), alpha pq, beta rs */
/* offsets inside one same-spin block */
#define NOWICK_X        0       /* X^{(mi)} */
#define NOWICK_Y        2       /* Y^{(mi)} */
#define NOWICK_FH       4       /* F^{(mi,mj)}, [z,r] order, mi*2+mj */
#define NOWICK_V        8       /* V^{(m1,m3,m2)}, [z,r] order, m1*4+m3*2+m2 */
#define NOWICK_HCOL0    16      /* F+V+V^{ab} for m_a=m_b=0, [z,r] order */
#define NOWICK_J        17      /* ten symmetry-unique J, [i,j,r,c] order */
/* offsets inside the different-spin block */
#define NOWICK_VAB      0       /* V^a, [z,r] order, ma0*4+mb0*2+maz */
#define NOWICK_VBA      8       /* V^b, [y,r] order, mb0*4+ma0*2+mby */
#define NOWICK_IIAB     16      /* II, ma0*8+maz*4+mb0*2+mby */

/* scalar layout */
#define NOWICK_NSCAL    24
#define NOWICK_SAA      0
#define NOWICK_SBB      8
#define NOWICK_SVAB0    16      /* V_ab0^{(ma0,mb0)}, ma0*2+mb0 */
/* inside one same-spin scalar block */
#define NOWICK_SPHASE   0       /* orbital pairing phase */
#define NOWICK_STSP     1       /* product of nonzero singular values */
#define NOWICK_SF0H     2       /* F_0^{(m)} */
#define NOWICK_SV0      4       /* V_0 by m1+m2 */

/* determinant record: header followed by
 * holes_a[kmax] parts_a[kmax] holes_b[kmax] parts_b[kmax]
 * holes and particles are indices in the occupied-first orbital order of
 * the parent, sorted ascending.  sign converts the excitation operator
 * string applied to the parent into the determinant (Wick evaluators);
 * osign converts the canonical determinant in the occupied-first order
 * into the original MO order (Slater-Condon evaluator). */
#define NOWICK_DPARENT  0
#define NOWICK_DRA      1
#define NOWICK_DRB      2
#define NOWICK_DSIGN    3
#define NOWICK_DOSIGN   4
#define NOWICK_DHEAD    5

typedef struct {
        int nref;
        int kmax;       /* largest per-spin excitation rank of a record */
        int dstride;    /* NOWICK_DHEAD + 4*kmax */
        int simd;       /* 0: scalar only; 1: allow SIMD packets */
        double enuc;
        const int64_t *pairinfo;
        const void *slab;
        const void *scal;
} NOWICKEnv;

#define NOWICK_SIMD_NONE        0
#define NOWICK_SIMD_AVX2        1
#define NOWICK_SIMD_AVX512      2

int NOWICKsimd_level();
