/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

/*
 * Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>
*/

#ifndef PYSCF_SISO_H
#define PYSCF_SISO_H

#include <complex.h>

void SISOcontract_same_spin(const double complex *h1e,
                            const double complex *ci0,
                            double complex *ci1,
                            int norb, int nroots,
                            int nstra1, int nstrb1,
                            int nstra0, int nstrb0,
                            int nlinka, int nlinkb,
                            const int *link_indexa,
                            const int *link_indexb);

void SISOcontract_spin_plus(const double complex *h1e,
                            const double complex *ci0,
                            double complex *ci1,
                            int norb, int nroots,
                            int nstra1, int nstrb1,
                            int nstra0, int nstrb0,
                            int nlinka, int nlinkb,
                            const int *link_indexa,
                            const int *link_indexb);

void SISOcontract_spin_minus(const double complex *h1e,
                             const double complex *ci0,
                             double complex *ci1,
                             int norb, int nroots,
                             int nstra1, int nstrb1,
                             int nstra0, int nstrb0,
                             int nlinka, int nlinkb,
                             const int *link_indexa,
                             const int *link_indexb);

#endif
