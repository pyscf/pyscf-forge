#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
#include "fblas.h"

#ifndef MINMAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MINMAX
#endif

void SINTKmatpermR (double eri, double * dense_vk, double * dense_dm, unsigned int iao, unsigned int jao, unsigned int kao, unsigned int lao, unsigned int nao)
{
    dense_vk[(iao*nao) + kao] += eri * dense_dm[(jao*nao) + lao];
}
void SDFKmatR_obsolete (double * sparse_cderi, double * dense_dm, double * dense_vk, int * nonzero_pair, int * tril_pair1, int * tril_pair2, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{
    /* Get exchange matrix from a sparse cderi with pairs indicated by nonzero_pair. tril_* are lower-triangular index lists.
       This is prone to huge rounding errors if I reduce it on the fly, so each thread should keep its own vk matrix, and the final reduction
       should be carried out post facto.
       sparse_cderi has the auxiliary basis on the faster-moving index (so the increment of the dot product is 1 and the offset to the address has a factor of naux in it)
       This is the worst implementation possible; never use this. I'm keeping it here only for reference.
    */

    unsigned int npp = npair * (npair + 1) / 2;
    const int one = 1;

#pragma omp parallel default(shared)
{

    unsigned int ithread = omp_get_thread_num (); // Thread index, used in final reduction
    unsigned int ipp; // Pair-of-pair index
    unsigned int ix_pair1, ix_pair2; // Pair index
    unsigned int pair1, pair2; // Pair identity
    unsigned int iao, jao, kao, lao; // AO indices
    double eri;
    double * my_vk = dense_vk + (nao*nao*ithread);

#pragma omp for schedule(static) 

    for (ipp = 0; ipp < npp; ipp++){

        // Pair-of-pair indexing
        ix_pair1 = tril_pair1[ipp];
        pair1 = nonzero_pair[ix_pair1];
        iao = tril_iao[pair1];
        jao = tril_jao[pair1];
        ix_pair2 = tril_pair2[ipp];
        pair2 = nonzero_pair[ix_pair2];
        kao = tril_iao[pair2];
        lao = tril_jao[pair2];

        // Dot product over auxbasis
        eri = ddot_(&naux, sparse_cderi + (ix_pair1*naux), &one, sparse_cderi + (ix_pair2*naux), &one);
        if (iao == jao){ eri /= 2; }
        if (kao == lao){ eri /= 2; }
        if (pair1 == pair2){ eri /= 2; }

        // (ij|kl) external index permutations: i <-> j, k <-> l, and ij <-> kl
        SINTKmatpermR (eri, my_vk, dense_dm, iao, jao, kao, lao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, jao, iao, kao, lao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, iao, jao, lao, kao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, jao, iao, lao, kao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, kao, lao, iao, jao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, lao, kao, iao, jao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, kao, lao, jao, iao, nao);
        SINTKmatpermR (eri, my_vk, dense_dm, lao, kao, jao, iao, nao);

    }

}

}


void SDFKmatR1 (double * sparse_cderi, double * dense_dm, double * dense_int, int * nonzero_pair, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{

    /* 
    Do the first contraction in the calculation of the exchange matrix using a sparse CDERI array and a density matrix:
    h(i,j,P) = (ik|P) D^k_j
    
    Input:
        sparse_cderi : array of shape (npair, naux); contains CDERIs
        dense_dm : array of shape (nao, nao); contains density matrix
        nonzero_pair : array of shape (npair) ; contains addresses of nao pairs with nonzero overlap
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        npair : number of pairs with nonzero overlap ; <= nao * (nao+1) / 2 ; propto nao in thermodynamic limit
        nao : number of AOs
        naux : number of auxiliary basis functions

    Output:
        dense_int : array of shape (nao, nao, naux); contains inner product of density matrix with CDERIs
           The first (slowest-changing) AO index is the CDERI index; the second is the density matrix index.
           This is necessary in order to use dger. I'll want to transpose this before doing the second contraction
    */

    const unsigned int i_one = 1;
    const unsigned int nao_naux = nao * naux;
    const double d_one = 1.0;

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int ipair, pair_id; // Pair index and identity
    unsigned int iao, jao; // AO indices
    // Array pointers for lapack
    double * out_ptr; 
    double * cderi_ptr;
    double * dm_ptr;

#pragma omp for schedule(static) 

    for (ipair = 0; ipair < npair; ipair++){
        cderi_ptr = sparse_cderi + (ipair * naux); // This points to a vector of length naux
        pair_id = nonzero_pair[ipair]; 
        iao = tril_iao[pair_id];
        jao = tril_jao[pair_id];
        if (iao % nthreads == ithread){ // Prevent race condition by assigning each CDERI output idx to 1 thread (output is NOT lower-triangular!)
            out_ptr = dense_int + (iao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dm_ptr = dense_dm + (jao * nao); // This points to a vector of length nao
            dger_(&naux, &nao, &d_one, cderi_ptr, &i_one, dm_ptr, &i_one, out_ptr, &naux);
            /* ^ I may have this backwards but if dger_ interprets everything in col-major form, this is what I need to get a
            row-major output of the type I want. Likewise below! */
        }
        if (iao != jao){ if (jao % nthreads == ithread){ 
            out_ptr = dense_int + (jao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dm_ptr = dense_dm + (iao * nao); // This points to a vector of length nao
            dger_(&naux, &nao, &d_one, cderi_ptr, &i_one, dm_ptr, &i_one, out_ptr, &naux);
        }}
    }

#pragma omp barrier

}
}

void SDFKmatRT (double * arr, double * wrk, int * tril_iao, int * tril_jao, int nao, int naux)
{

    /*
    Transpose the (dense, massive) intermediate from SDFKmatR1 in-place in preparation for the dgemv in the second step

    Input:
        wrk : array of shape (nthreads, naux); used to hold vectors during the transpose
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        nao : number of AOs
        naux : number of auxiliary basis functions

    Input/Output:
        arr : array of shape (nao, nao, naux); contains the output of SDFKmatR1
            On entry, slowest-moving index is the CDERI index
            On exit, slowest-moving index is the density-matrix index
    */

    const unsigned int i_one = 1;
    const unsigned int nao_naux = nao * naux;
    const unsigned int npair = nao * (nao + 1) / 2;

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int ipair, iao, jao; // AO indices
    // Array pointers for lapack
    double * my_wrk = wrk + (ithread * naux);
    double * vec_ij;
    double * vec_ji;

    for (ipair = 0; ipair < npair; ipair++){ // Race conditions are impossible because the pairs don't talk to each other
        iao = tril_iao[ipair];
        jao = tril_jao[ipair];
        if (iao > jao){ 
            vec_ij = arr + (iao * nao_naux) + (jao * naux);
            vec_ji = arr + (jao * nao_naux) + (iao * naux);
            dcopy_(&naux, vec_ij, &i_one, my_wrk, &i_one);
            dcopy_(&naux, vec_ji, &i_one, vec_ij, &i_one);
            dcopy_(&naux, my_wrk, &i_one, vec_ji, &i_one);
        }
    } 

}
}

void SDFKmatR2 (double * sparse_cderi, double * dense_int, double * dense_vk, int * nonzero_pair, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{

    /* 
    Do the second contraction in the calculation of the exchange matrix using a sparse CDERI array and a density matrix:
    K^i_j = (ik|P) h(j,k,P)
    
    Input:
        sparse_cderi : array of shape (npair, naux); contains CDERIs
        dense_int : array of shape (nao, nao, naux); contains contraction of CDERI with density matrix
            Slowest-moving index is the density matrix index
        nonzero_pair : array of shape (npair) ; contains addresses of nao pairs with nonzero overlap
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        npair : number of pairs with nonzero overlap ; <= nao * (nao+1) / 2 ; propto nao in thermodynamic limit
        nao : number of AOs
        naux : number of auxiliary basis functions

    Output:
        dense_vk : array of shape (nao,nao); contains the exchange matrix
    */

    const unsigned int i_one = 1;
    const unsigned int nao_naux = nao * naux;
    const double d_one = 1.0;

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int ipair, pair_id; // Pair index and identity
    unsigned int iao, jao; // AO indices
    const char trans = 'T';
    // Array pointers for lapack
    double * out_ptr;
    double * cderi_ptr;
    double * int_ptr; 

#pragma omp for schedule(static) 

    for (ipair = 0; ipair < npair; ipair++){
        cderi_ptr = sparse_cderi + (ipair * naux); // This points to a vector of length naux
        pair_id = nonzero_pair[ipair]; 
        iao = tril_iao[pair_id];
        jao = tril_jao[pair_id];
        if (iao % nthreads == ithread){ // Prevent race condition by assigning each CDERI output idx to 1 thread
            // I COULD make vk lower-triangular, but that speedup wouldn't parallel-scale unless I reworked the output thread assignment 
            out_ptr = dense_vk + (iao * nao); // This points to a vector of length nao
            int_ptr = dense_int + (jao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dgemv_(&trans, &naux, &nao, &d_one, int_ptr, &naux, cderi_ptr, &i_one, &d_one, out_ptr, &i_one);
            /* ^ I may have this backwards but if dgemv_ interprets everything in col-major form, this is what I need to get a
            row-major output of the type I want. Likewise below! */
        }
        if (iao != jao){ if (jao % nthreads == ithread){ 
            out_ptr = dense_vk + (jao * nao); // This points to a vector of length nao
            int_ptr = dense_int + (iao * nao_naux); // This points to a matrix of shape (nao, naux) in C or (naux, nao) in Fortran
            dgemv_(&trans, &naux, &nao, &d_one, int_ptr, &naux, cderi_ptr, &i_one, &d_one, out_ptr, &i_one);
        }}
    }
}
}

void SDFKmatR (double * sparse_cderi, double * dense_dm, double * dense_vk, double * large_int, double * small_int, int * nonzero_pair, int * tril_iao, int * tril_jao, int npair, int nao, int naux)
{

    /*
    Wrapper for the three steps of calculating the exchange matrix from the sparse CDERI array and a density matrix:
    first contraction, transpose, second contraction

    Input:
        sparse_cderi : array of shape (npair, naux); contains CDERIs
        dense_dm : array of shape (nao, nao); contains density matrix
        large_int : array of shape (nao, nao, naux); used to contain the large, dense intermediate created after the first step
        small_int : array of shape (nthreads, naux); used to contain vectors when transposing large_int
        nonzero_pair : array of shape (npair) ; contains addresses of nao pairs with nonzero overlap
        tril_iao : array of shape (nao*(nao+1)/2) ; first AO index for a given pair address
        tril_jao : array of shape (nao*(nao+1)/2) ; second AO index for a given pair address
        npair : number of pairs with nonzero overlap ; <= nao * (nao+1) / 2 ; propto nao in thermodynamic limit
        nao : number of AOs
        naux : number of auxiliary basis functions

    Output:
        dense_vk : array of shape (nao, nao); contains the exchange matrix
    */

    SDFKmatR1 (sparse_cderi, dense_dm, large_int, nonzero_pair, tril_iao, tril_jao, npair, nao, naux);
    SDFKmatRT (large_int, small_int, tril_iao, tril_jao, nao, naux);
    SDFKmatR2 (sparse_cderi, large_int, dense_vk, nonzero_pair, tril_iao, tril_jao, npair, nao, naux);

}

void SINT_SDCDERI_DDMAT (double * dense_cderi, double * dense_A, double * dense_prod, double * wrk,  
    int * iao_sort, int * iao_nent, int * iao_entlist, int nao, int naux, int nmo, int nent_max)
{
    /*
    Matrix-multiply a CDERI array using sparsity information from the CDERIs, but not the other multiplicand.

    Input:
        dense_cderi : array of shape (naux, nao*(nao+1)/2); contains CDERIs (dense, lower-triangular storage with auxiliary basis index first)
        dense_A : array of shape (nao, nmo); contains the multiplicand
        iao_sort : array of shape (nao); sorts the AOs according to iao_nent (for benefit of parallel scaling)
        iao_nent : array of shape (nao); lists how many nonvanishing pairs exist for each orbital
        iao_entlist : array of shape (nao, nent_max); lists the other orbital in nonvanishing pairs involving each orbital

    Input/output:
        wrk : array of length nthreads * nent_max * (naux + nmo); contains 0s on input and garbage on output
        
    Output:
        dense_prod : array of shape (nao, nmo, naux); contains result of multiplication (with auxbasis index placed last) 
        Transpose me BEFORE calling SINT_SDCDERI_VK!
    */

    
    const unsigned int i_one = 1;
    const unsigned int npair = nao * (nao + 1) / 2;
    const double d_one = 1.0;
    const char transCDERI = 'T';
    const char transA = 'N';

    
#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int iao_ix, iao, jao_ix, jao; // AO indices
    unsigned int my_nent;
    const char trans = 'T';
    // Array pointers for lapack
    double * my_cderi; // put nent on the faster-moving index: (naux, nent_max)
    double * my_A; // put nent on the faster-moving index: (nmo, nent_max) in C
    double * my_prod;
    double * my_cderi_wrk;
    double * my_A_wrk;
    int * my_entlist;
    my_cderi_wrk = wrk + (ithread * nent_max * (naux + nmo));
    my_A_wrk = my_cderi_wrk + (nent_max * naux);

#pragma omp for schedule(static) 

    for (iao_ix = 0; iao_ix < nao; iao_ix++){
        iao = iao_sort[iao_ix];
        my_nent = iao_nent[iao];
        my_entlist = iao_entlist + (iao*nent_max);
        my_prod = dense_prod + (iao * nmo * naux);
        for (jao_ix = 0; jao_ix < my_nent; jao_ix++){
            jao = my_entlist[jao_ix];
            if (iao > jao){ 
                my_cderi = dense_cderi + ((iao * (iao + 1) / 2) + jao);
            } else {
                my_cderi = dense_cderi + ((jao * (jao + 1) / 2) + iao);
            }
            my_A = dense_A + (jao * nmo);
            dcopy_(&naux, my_cderi, &npair, my_cderi_wrk + jao_ix, &my_nent);
            dcopy_(&nmo, my_A, &i_one, my_A_wrk + jao_ix, &my_nent);
        }
        dgemm_(&transCDERI, &transA, &naux, &nmo, &my_nent,
            &d_one, my_cderi_wrk, &my_nent, my_A_wrk, &my_nent,
            &d_one, my_prod, &naux);
        /* Remember, dgemm_ is Fortran and reads array indices backwards. So when I say
        I am transposing my_CDERI but not transposing my_A, it's actually the opposite. */
    }

}
}

void SINT_SDCDERI_VK (double * dense_cderi, double * dense_int, double * dense_vk, double * wrk,  
    int * iao_sort, int * iao_nent, int * iao_entlist, int nao, int naux, int nent_max)
{

    /* Compute the exchange matrix: k(m,n) = sum(p,r,s) CDERI(p,m,r) [CDERI(p,n,s) DM(r,s)]

    Input:
        dense_cderi : array of shape (nao*(nao+1)/2, naux); contains CDERIs (dense, lower-triangular storage with auxiliary basis index fast)
        dense_int : array of shape (nao, nao, naux); contains the intermediate I(r,n,p) = sum(s) CDERI(p,n,s) DM(r,s)
        iao_sort : array of shape (nao); sorts the AOs according to iao_nent (for benefit of parallel scaling)
        iao_nent : array of shape (nao); lists how many nonvanishing pairs exist for each orbital
        iao_entlist : array of shape (nao, nent_max); lists the other orbital in nonvanishing pairs involving each orbital

    Input/output:
        wrk : array of length (nthreads * nent_max * naux); contains 0s on input and garbage on output
        
    Output:
        dense_vk : array of shape (nao, nao); contains exchange matrix 
    */

    const unsigned int i_one = 1;
    const unsigned int npair = nao * (nao + 1) / 2;
    const unsigned int nao_naux = nao * naux;
    const double d_one = 1.0;
    const char transINT = 'T';

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int iao_ix, iao, jao_ix, jao; // AO indices
    unsigned int my_nent;
    const char trans = 'T';
    // Array pointers for lapack
    double * my_cderi; 
    double * my_int; 
    double * my_vk;
    double * my_cderi_wrk;
    int * my_entlist;
    int nent_naux;
    my_cderi_wrk = wrk + (ithread * nent_max * naux);

#pragma omp for schedule(static) 

    for (iao_ix = 0; iao_ix < nao; iao_ix++){
        iao = iao_sort[iao_ix];
        my_nent = iao_nent[iao];
        my_entlist = iao_entlist + (iao*nent_max);
        my_vk = dense_vk + (iao * nao);
        nent_naux = my_nent * naux;
        for (jao_ix = 0; jao_ix < my_nent; jao_ix++){
            jao = my_entlist[jao_ix];
            if (iao > jao){ 
                my_cderi = dense_cderi + ((iao * (iao + 1) / 2) + jao) * naux;
            } else {
                my_cderi = dense_cderi + ((jao * (jao + 1) / 2) + iao) * naux;
            }
            my_int = dense_int + (jao * nao * naux);
            jao = iao + 1; // Fill lower-triangular part only 
            dgemv_(&transINT, &naux, &jao, &d_one, my_int, &naux, my_cderi, &i_one, &d_one, my_vk, &i_one);
            //dcopy_(&naux, my_cderi, &npair, my_cderi_wrk + jao_ix, &my_nent); 
            //dcopy_(&nao_naux, my_int, &i_one, my_int_wrk + jao_ix, &my_nent);   
        }
        //dgemv_(&transINT, &nent_naux, &nao, &d_one, my_int_wrk, &nent_naux, my_cderi_wrk, &i_one, &d_one, my_vk, &i_one);
    }
}
}

void SINT_SDCDERI_MO_LVEC (double * dense_cderi, double * mo_coeff, double * cderi_out, double * mo_out,
    double * wrk, double sv_thresh, int * iao_sort, int * iao_nent, int * iao_entlist, int * imo_nent,
    int nao, int naux, int nmo, int nent_max)
{
    /*
    Perform the sparsest possible basis transformation of a sigle index of a CDERI array using repeated SVDs of
    an MO coefficient matrix. For each AO index, linear combinations of MOs which span the nent CDERI-coupled AOs are generated
    as left- and right-singular vectors of the coefficient array. CDERI rows transformed by the left-singular vectors are returned
    in cderi_out and the right-singular vectors times the singular values are returned in mo_out.

    Input:
        dense_cderi : array of shape (naux, nao*(nao+1)/2); contains CDERIs (dense, lower-triangular storage with auxiliary basis index first)
        mo_coeff : array of shape (nao, nmo); contains the MO coefficients
        iao_sort : array of shape (nao); sorts the AOs according to iao_nent (for benefit of parallel scaling)
        iao_nent : array of shape (nao); lists how many nonvanishing pairs exist for each orbital
        iao_entlist : array of shape (nao, nent_max); lists the other orbital in nonvanishing pairs involving each orbital
        sv_thresh : threshold at which to discard singular values

    Input/output:
        wrk : array of shape (nthreads*(lwork+global_K+nent_max[global_K*nmo*naux]); used to store intermediates
            where lwork and global_K are defined below

    Output:
        cderi_out : array of shape (nao, naux, global_K); contains the CDERI array with one AO index transformed
        mo_out : array of shape (nao, nent_max, nmo); contains the right-singular vectors multiplied by singular values
        imo_nent : array of shape (nao); contains number of singular values for each AO
    */
    const char svdjob = 'S';
    const char trans = 'T';
    const char notrans = 'N';
    const double d_one = 1.0;
    const unsigned int global_K = MIN(nent_max,nmo);
    const unsigned int lwork = (4*global_K*global_K) + (7*global_K);
    const unsigned int npair = nao * (nao + 1) / 2;
    const unsigned int i_one = 1;
    const unsigned int lfullwrk = lwork + global_K + nent_max*(global_K+naux+nmo);

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int iao_ix, iao, jao_ix, jao; // AO indices
    unsigned int nsv_p, nsv_n, nsv_nn, nsv;
    unsigned int my_k;
    unsigned int my_info;
    unsigned int uint_wrk;
    unsigned int my_nent;
    double * ptr_wrk;
    double * my_vt;
    int * my_entlist;
    // Partition out the wrk array
    double * my_svdwork = wrk + ithread*lfullwrk;
    double * my_mo = my_svdwork + lwork;
    double * my_singval = my_mo + nent_max*nmo;
    double * my_u = my_singval + global_K;
    double * my_cderi = my_u + global_K*nent_max;
    // Integer array allocation
    int * iwork = malloc (8 * global_K * sizeof(int));

#pragma omp for schedule(static) 

    for (iao_ix = 0; iao_ix < nao; iao_ix++){
        iao = iao_sort[iao_ix];
        my_nent = iao_nent[iao];
        my_entlist = iao_entlist + (iao*nent_max);
        my_vt = mo_out + (iao * nent_max * nmo);
        for (jao_ix = 0; jao_ix < my_nent; jao_ix++){
            jao = my_entlist[jao_ix];
            dcopy_(&nmo, &mo_coeff[jao], &i_one, &my_mo[jao_ix], &my_nent);
            // The source has AO index slower-moving; the dest has AO index faster-moving! Row- to col-major order, but it's the same shape!
            if (iao > jao){ 
                ptr_wrk = dense_cderi + ((iao * (iao + 1) / 2) + jao);
            } else {
                ptr_wrk = dense_cderi + ((jao * (jao + 1) / 2) + iao);
            }
            dcopy_(&naux, ptr_wrk, &npair, &my_cderi[jao_ix], &my_nent);
        }
        my_k = MIN (my_nent, nmo);
        dgesdd_(&svdjob, &my_nent, &nmo, my_mo, &my_nent, my_singval, my_u, &my_nent, my_vt, &my_k,
            my_svdwork, &lwork, iwork, &my_info);
        if (my_info != 0){ printf ("SVD return value = %d", my_info); }
        assert (my_info == 0);
        // Compress away zero singular values.  Singvals can be negative unfortunately because of mo gauge invariance
        nsv_p = 0; nsv_n = my_k;
        for (jao_ix = 0; jao_ix < my_k; jao_ix++){
            if (my_singval[jao_ix] < -sv_thresh){ break; }
            else { nsv_n--; }
            if (my_singval[jao_ix] > sv_thresh){ nsv_p++; }
        }
        nsv_nn = my_k - nsv_n;
        dcopy_(&nsv_n, &my_singval[nsv_nn], &i_one, &my_singval[nsv_p], &i_one);
        uint_wrk = my_nent * nsv_n;
        dcopy_(&uint_wrk, &my_u[nsv_nn*my_nent], &i_one, &my_u[nsv_p*my_nent], &i_one);
        ptr_wrk = my_vt + nsv_p;
        for (jao_ix = 0; jao_ix < my_k; jao_ix++){ // This one's tricky because I'm compressing away the faster-moving index
            uint_wrk = nsv_n + nsv_p;
            dcopy_(&uint_wrk, &my_vt[(jao_ix*my_k)+nsv_nn], &i_one, ptr_wrk, &i_one);
            ptr_wrk += uint_wrk;
        }
        // Scale by singular value! Do it on the right because the right has singvec index faster-moving!
        imo_nent[iao] = nsv_nn;
        for (jao_ix = 0; jao_ix < nsv_nn; jao_ix++){
            dscal_(&nmo, &my_singval[jao_ix], &my_vt[jao_ix], &nmo);
        }
        // Finally, matrix-multiply
        dgemm_(&trans, &notrans, imo_nent + iao, &naux, &my_nent,
            &d_one, my_u, &my_nent, my_cderi, &my_nent,
            &d_one, &cderi_out[iao*global_K*naux], &global_K);
    }
    free (iwork);

}
}

void SINT_SDCDERI_DDMAT_MOSVD (double * cderi_mo, double * mo_rvecs, double * vk, double * wrk,
    int * imo_nent, int nao, int nmo, int naux, int nent_max, int global_K)
{
    /*
    Calculate the vk matrix using the output of SINT_SDCDERI_MO_LVEC above.

    Input:
        cderi_mo : array of shape (nao, naux, global_K); contains cderi_out from SINT_SDCDERI_MO_LVEC
        mo_rvecs : array of shape (nao, nent_max, nmo); contains mo_out from SINT_SDCDERI_MO_LVEC
        imo_nent : array of shape (nao); contains imo_nent from SINT_SDCDERI_MO_LVEC

    Input/Output:
        wrk : array of shape (nthreads*global_K*[global_K+naux]); used for intermediates

    Output:
        vk : array of shape (nao, nao); contains exchange matrix
    */
    const char trans = 'T';
    const char notrans = 'N';
    const double d_one = 1.0;
    const unsigned int i_one = 1;
    const unsigned int npair = nao * (nao + 1) / 2;
    const unsigned int lfullwork = global_K*(global_K+naux);
    const unsigned int global_K_naux = global_K * naux;
    const unsigned int nent_max_nmo = nent_max * nmo;

#pragma omp parallel default(shared)
{

    unsigned int nthreads = omp_get_num_threads ();
    unsigned int ithread = omp_get_thread_num ();
    unsigned int ipair, iao, jao, kao, lao; // AO indices
    unsigned int iao_nent, jao_nent, kao_nent, lao_nent;
    unsigned int uint_wrk;
    double * ptr_wrk;
    double * my_vt;
    int * my_entlist;
    double * kao_rvecs;
    double * lao_rvecs;
    double * kao_cderi;
    double * lao_cderi;
    // Partition out the wrk array
    double * dm = wrk + ithread*lfullwork;
    double * vdm = dm + global_K*global_K;

#pragma omp for schedule(dynamic) 

    for (ipair = 0; ipair < npair; ipair++){
        uint_wrk = 0;
        iao = 0;
        jao = 0;
        while (uint_wrk + iao + 1 < ipair){
            iao++;
            uint_wrk += iao;
        }
        jao = ipair - uint_wrk;
        iao_nent = imo_nent[iao];
        jao_nent = imo_nent[jao];
        // Put longer-range index at faster-moving position
        if (iao_nent > jao_nent){
            kao = iao; kao_nent = iao_nent;
            lao = jao; lao_nent = jao_nent;
        } else {
            kao = jao; kao_nent = jao_nent;
            lao = iao; lao_nent = iao_nent;
        }
        kao_rvecs = mo_rvecs + (kao * nent_max_nmo);
        lao_rvecs = mo_rvecs + (lao * nent_max_nmo);
        kao_cderi = cderi_mo + (kao * global_K_naux);
        lao_cderi = cderi_mo + (lao * global_K_naux);
        // Make density matrix
        dgemm_(&trans, &notrans, &kao_nent, &lao_nent, &nmo,
            &d_one, kao_rvecs, &nmo, lao_rvecs, &nmo,
            &d_one, dm, &kao_nent);
        // Contract density matrix with first CDERI factor
        dgemm_(&trans, &notrans, &naux, &lao_nent, &kao_nent,
            &d_one, kao_cderi, &kao_nent, dm, &kao_nent,
            &d_one, vdm, &naux);
        // Final contraction
        uint_wrk = lao_nent * naux;
        vk[(iao*nao)+jao] = ddot_(&uint_wrk, vdm, &i_one, lao_cderi, &i_one);
        vk[(jao*nao)+iao] = vk[(iao*nao)+jao];
    }

}
}

