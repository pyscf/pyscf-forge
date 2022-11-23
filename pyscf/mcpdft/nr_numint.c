/* Copyright 2014-2020 The PySCF Developers. All Rights Reserved.

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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

/* Copied from pyscf/pyscf/lib/dft/nr_numint.c on 05/18/2020 - MRH
   VXC -> VOT; start with dot_ao_ao and ao_empty_blocks and throw 
   away everything else */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

// --------------------------------------------------------------
// -------------------- begin PySCF includes --------------------
// --------------------------------------------------------------

//#include "build/config.h"
#if defined _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

//#include "gto/grid_ao_drv.h"
#define BLKSIZE         104

//#include "np_helper/np_helper.h"
#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))

//#include "vhf/fblas.h"
void dgemm_(const char*, const char*,
            const int*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*,
            const double*, double*, const int*);

// --------------------------------------------------------------
// --------------------- end PySCF includes ---------------------
// --------------------------------------------------------------

#define BOXSIZE         56
/* 
MRH 06/10/2022: This function is defined in pyscf/lib/libdft.so
(dft/nr_numint.c) 
*/
int VXCao_empty_blocks(int8_t *empty, uint8_t *non0table, int *shls_slice,
                       int *ao_loc)
{
        if (non0table == NULL || shls_slice == NULL || ao_loc == NULL) {
                return 0;
        }

        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];

        int bas_id;
        int box_id = 0;
        int bound = BOXSIZE;
        int has0 = 0;
        empty[box_id] = 1;
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                if (ao_loc[bas_id] == bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = 1;
                }
                empty[box_id] &= !non0table[bas_id];
                if (ao_loc[bas_id+1] > bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = !non0table[bas_id];
                }
        }
        return has0;
}
// TODO: figure out how to link against the libdft.so definition of this function instead of copying it

/* vv[n,m] = ao[n,ngrids] * mo[m,ngrids] */
/* MRH 05/18/2020: dot_ao_ao -> dot_ao_mo requires
    1. New variable nmo
    2. nbox -> nboxi, nboxj 
    3. Never hermitian: I can handle mo, mo case using dot_ao_ao and null mask
    4. Second degree of freedom has nothing to do with mask: remove conditional 
Notice that the linear algebra in column-major order is
mo(ngrids,nmo).T @ ao(ngrids, nao) = vv(nmo,nao) */ 
static void dot_ao_mo(double *vv, double *ao, double *mo,
                      int nao, int nmo, int ngrids, int bgrids,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        int nboxi = (nao+BOXSIZE-1) / BOXSIZE;
        int nboxj = (nmo+BOXSIZE-1) / BOXSIZE;
        int8_t empty[nboxi];
        int has0 = VXCao_empty_blocks(empty, non0table, shls_slice, ao_loc);

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        if (has0) {
                int ib, jb, leni, lenj;
                size_t b0i, b0j;

                for (ib = 0; ib < nboxi; ib++) {
                if (!empty[ib]) {
                        b0i = ib * BOXSIZE;
                        leni = MIN(nao-b0i, BOXSIZE);
                        for (jb = 0; jb < nboxj; jb++) {
                                b0j = jb * BOXSIZE;
                                lenj = MIN(nmo-b0j, BOXSIZE);
                                dgemm_(&TRANS_T, &TRANS_N, &lenj, &leni, &bgrids, &D1,
                                       mo+b0j*ngrids, &ngrids, ao+b0i*ngrids, &ngrids,
                                       &D1, vv+b0i*nao+b0j, &nmo);
                        } 
                } }
        } else {
                dgemm_(&TRANS_T, &TRANS_N, &nmo, &nao, &bgrids,
                       &D1, mo, &ngrids, ao, &ngrids, &D1, vv, &nmo);
        }
}


/* vv[nao,nmo] = ao[i,nao] * mo[i,nmo] */
/* MRH 05/18/2020: dot_ao_ao -> dot_ao_mo straightforward because it's
    multithreaded over grid blocks. Just make sure the variable names
    and allocations all get changed and hermi is taken out. */
void VOTdot_ao_mo(double *vv, double *ao, double *mo,
                  int nao, int nmo, int ngrids, int nbas, 
                  unsigned char *non0table, int *shls_slice, int *ao_loc)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        memset(vv, 0, sizeof(double) * nao * nmo);

#pragma omp parallel
{
        int ip, ib;
        double *v_priv = calloc(nao*nmo+2, sizeof(double));
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                dot_ao_mo(v_priv, ao+ip, mo+ip,
                          nao, nmo, ngrids, MIN(ngrids-ip, BLKSIZE),
                          non0table+ib*nbas, shls_slice, ao_loc);
        }
#pragma omp critical
        {
                for (ip = 0; ip < nao*nmo; ip++) {
                        vv[ip] += v_priv[ip];
                }
        }
        free(v_priv);
}
}

