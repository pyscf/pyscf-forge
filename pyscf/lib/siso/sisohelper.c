/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
        Author: Bhavnesh Jangid
*/

#include <complex.h>
#include <stddef.h>

#include "siso.h"

#define SOC_NCOMP 3
#define LINK_STRIDE 4

static inline size_t h1e_addr(int component, int p, int q, int norb)
{
        return ((size_t)component * norb + p) * norb + q;
}

static inline size_t ci_addr(int root, int stra, int strb,
                             int nstra, int nstrb)
{
        return ((size_t)root * nstra + stra) * nstrb + strb;
}

static inline size_t output_addr(int component, int root, int stra, int strb,
                                 int nroots, int nstra, int nstrb)
{
        return (((size_t)component * nroots + root) * nstra + stra)
               * nstrb + strb;
}

/* Contract the rank-one SOC operator between states with equal total spin. */
void SISOcontract_same_spin(const double complex *h1e,
                            const double complex *ci0,
                            double complex *ci1,
                            int norb, int nroots,
                            int nstra1, int nstrb1,
                            int nstra0, int nstrb0,
                            int nlinka, int nlinkb,
                            const int *link_indexa,
                            const int *link_indexb)
{
#pragma omp parallel for collapse(4) schedule(static)
        for (int component = 0; component < SOC_NCOMP; component++) {
                for (int root = 0; root < nroots; root++) {
                        for (int stra = 0; stra < nstra1; stra++) {
                                for (int strb = 0; strb < nstrb1; strb++) {
                                        double complex value = 0.0;

                                        const int *linka = link_indexa
                                                + (size_t)stra * nlinka * LINK_STRIDE;
                                        for (int link = 0; link < nlinka; link++) {
                                                const int *entry = linka
                                                        + link * LINK_STRIDE;
                                                value += entry[3]
                                                        * h1e[h1e_addr(component,
                                                                      entry[0], entry[1],
                                                                      norb)]
                                                        * ci0[ci_addr(root, entry[2],
                                                                      strb, nstra0,
                                                                      nstrb0)];
                                        }

                                        const int *linkb = link_indexb
                                                + (size_t)strb * nlinkb * LINK_STRIDE;
                                        for (int link = 0; link < nlinkb; link++) {
                                                const int *entry = linkb
                                                        + link * LINK_STRIDE;
                                                value -= entry[3]
                                                        * h1e[h1e_addr(component,
                                                                      entry[0], entry[1],
                                                                      norb)]
                                                        * ci0[ci_addr(root, stra,
                                                                      entry[2], nstra0,
                                                                      nstrb0)];
                                        }

                                        ci1[output_addr(component, root, stra, strb,
                                                        nroots, nstra1, nstrb1)] = value;
                                }
                        }
                }
        }
}

/* Contract from a ket with spin S to a bra with spin S+1. */
void SISOcontract_spin_plus(const double complex *h1e,
                            const double complex *ci0,
                            double complex *ci1,
                            int norb, int nroots,
                            int nstra1, int nstrb1,
                            int nstra0, int nstrb0,
                            int nlinka, int nlinkb,
                            const int *link_indexa,
                            const int *link_indexb)
{
#pragma omp parallel for collapse(4) schedule(static)
        for (int component = 0; component < SOC_NCOMP; component++) {
                for (int root = 0; root < nroots; root++) {
                        for (int stra = 0; stra < nstra1; stra++) {
                                for (int strb = 0; strb < nstrb1; strb++) {
                                        double complex value = 0.0;
                                        const int *linka = link_indexa
                                                + (size_t)stra * nlinka * LINK_STRIDE;
                                        const int *linkb = link_indexb
                                                + (size_t)strb * nlinkb * LINK_STRIDE;

                                        for (int ia = 0; ia < nlinka; ia++) {
                                                const int *a = linka + ia * LINK_STRIDE;
                                                for (int ib = 0; ib < nlinkb; ib++) {
                                                        const int *b = linkb
                                                                + ib * LINK_STRIDE;
                                                        value += a[3] * b[3]
                                                                * h1e[h1e_addr(component,
                                                                              b[0], a[1],
                                                                              norb)]
                                                                * ci0[ci_addr(root, a[2],
                                                                              b[2], nstra0,
                                                                              nstrb0)];
                                                }
                                        }

                                        ci1[output_addr(component, root, stra, strb,
                                                        nroots, nstra1, nstrb1)] = value;
                                }
                        }
                }
        }
}

/* Contract from a ket with spin S to a bra with spin S-1. */
void SISOcontract_spin_minus(const double complex *h1e,
                             const double complex *ci0,
                             double complex *ci1,
                             int norb, int nroots,
                             int nstra1, int nstrb1,
                             int nstra0, int nstrb0,
                             int nlinka, int nlinkb,
                             const int *link_indexa,
                             const int *link_indexb)
{
#pragma omp parallel for collapse(4) schedule(static)
        for (int component = 0; component < SOC_NCOMP; component++) {
                for (int root = 0; root < nroots; root++) {
                        for (int stra = 0; stra < nstra1; stra++) {
                                for (int strb = 0; strb < nstrb1; strb++) {
                                        double complex value = 0.0;
                                        const int *linka = link_indexa
                                                + (size_t)stra * nlinka * LINK_STRIDE;
                                        const int *linkb = link_indexb
                                                + (size_t)strb * nlinkb * LINK_STRIDE;

                                        for (int ia = 0; ia < nlinka; ia++) {
                                                const int *a = linka + ia * LINK_STRIDE;
                                                for (int ib = 0; ib < nlinkb; ib++) {
                                                        const int *b = linkb
                                                                + ib * LINK_STRIDE;
                                                        value += a[3] * b[3]
                                                                * h1e[h1e_addr(component,
                                                                              a[0], b[1],
                                                                              norb)]
                                                                * ci0[ci_addr(root, a[2],
                                                                              b[2], nstra0,
                                                                              nstrb0)];
                                                }
                                        }

                                        ci1[output_addr(component, root, stra, strb,
                                                        nroots, nstra1, nstrb1)] = value;
                                }
                        }
                }
        }
}
