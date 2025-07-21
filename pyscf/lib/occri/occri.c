/*
 * OCCRI C backend for evaluating real-space integrals and performing FFT-based convolution
 *
 * This implementation uses FFTW's r2c and c2r interfaces to evaluate the Coulomb kernel in G-space.
 * For each occupied MO pair (i,j), constructs rho_{ij}(r), FFTs it to rho(G), multiplies by
 * Coulomb kernel (1/|G|^2), inverse transforms, and then contracts with MO_j and occupation number.
 *
 * Key optimizations:
 *  - Uses sqrt(Ng) normalization to match PySCF's conventions
 *  - OpenMP-parallel over outer-loop index i
 *  - Pre-allocates per-thread FFT buffers and plans for reuse
 *
 * Inputs:
 *  - ao_mos: flattened (nmo x ngrids) real-valued MO functions on grid
 *  - vR_dm: output buffer for accumulated potential (nmo x ngrids)
 *  - coulG: Coulomb kernel in G-space (real, length = nG_complex)
 *
 * Author: Kori Smyser
 * Date: 2025
 */

#include "occri.h"
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <omp.h>
#include "vhf/fblas.h"

FFTWBuffers *init_fftw_buffers(int mesh[3]) {
    const int ngrids = mesh[0] * mesh[1] * mesh[2];
    const int ncomplex = mesh[0] * mesh[1] * (mesh[2]/2 + 1);

    FFTWBuffers *buf = malloc(sizeof(FFTWBuffers));
    // Use FFTW_ALLOC_MAXIMUM_ALIGNMENT for better vectorization
    buf->rho = fftw_malloc(sizeof(double) * ngrids);
    buf->vG  = fftw_malloc(sizeof(fftw_complex) * ncomplex);
    buf->vR  = fftw_malloc(sizeof(double) * ngrids);

    // Use FFTW_PATIENT for first thread, then copy plans for others
    // This provides better performance while keeping setup cost reasonable
    buf->forward = fftw_plan_dft_r2c_3d(mesh[0], mesh[1], mesh[2],
                                        buf->rho, buf->vG, FFTW_PATIENT);

    buf->backward = fftw_plan_dft_c2r_3d(mesh[0], mesh[1], mesh[2],
                                         buf->vG, buf->vR, FFTW_PATIENT);
    return buf;
}

void destroy_fftw_buffers(FFTWBuffers *buf) {
    fftw_destroy_plan(buf->forward);
    fftw_destroy_plan(buf->backward);
    fftw_free(buf->rho);
    fftw_free(buf->vG);
    fftw_free(buf->vR);
    free(buf);
}

FFTWBuffers **allocate_thread_fftw_buffers(int mesh[3], int nthreads) {
    FFTWBuffers **buffers = (FFTWBuffers **)malloc(sizeof(FFTWBuffers *) * nthreads);
    
    // Create master plan with FFTW_PATIENT for optimal performance
    FFTWBuffers *master = init_fftw_buffers(mesh);
    buffers[0] = master;
    
    // For additional threads, create buffers but reuse plan wisdom
    const int ngrids = mesh[0] * mesh[1] * mesh[2];
    const int ncomplex = mesh[0] * mesh[1] * (mesh[2]/2 + 1);
    
    for (int i = 1; i < nthreads; ++i) {
        FFTWBuffers *buf = malloc(sizeof(FFTWBuffers));
        buf->rho = fftw_malloc(sizeof(double) * ngrids);
        buf->vG  = fftw_malloc(sizeof(fftw_complex) * ncomplex);
        buf->vR  = fftw_malloc(sizeof(double) * ngrids);
        
        // Use FFTW_ESTIMATE since we have wisdom from master plan
        buf->forward = fftw_plan_dft_r2c_3d(mesh[0], mesh[1], mesh[2],
                                            buf->rho, buf->vG, FFTW_ESTIMATE);
        buf->backward = fftw_plan_dft_c2r_3d(mesh[0], mesh[1], mesh[2],
                                             buf->vG, buf->vR, FFTW_ESTIMATE);
        buffers[i] = buf;
    }
    return buffers;
}

void free_thread_fftw_buffers(FFTWBuffers **buffers, int nthreads) {
    for (int i = 0; i < nthreads; ++i) {
        destroy_fftw_buffers(buffers[i]);
    }
    free(buffers);
}

void integrals_uu(int i, double *ao_mos, double *vR_dm, double *coulG, int nmo,
                  int ngrids, double *mo_occ, int mesh[3], FFTWBuffers *buf) {
    const int idx_i = i * ngrids;
    const double sqn = sqrt(ngrids);
    const double isqn = 1.0 / sqn;
    const int ncomplex = mesh[0] * mesh[1] * (mesh[2]/2 + 1);
    
    // Cache pointers for better compiler optimization
    double * const rho = buf->rho;
    fftw_complex * const vG = buf->vG;
    double * const vR = buf->vR;
    const double * const ao_mos_i = &ao_mos[idx_i];

    for (int j = 0; j < nmo; j++) {
        const int idx_j = j * ngrids;
        const double * const ao_mos_j = &ao_mos[idx_j];
        const double mo_occ_j_isqn = mo_occ[j] * isqn;  // Hoist multiplication

        // Vectorizable loop: compute density
        #pragma omp simd aligned(rho,ao_mos_i,ao_mos_j:64)
        for (int g = 0; g < ngrids; g++) {
            rho[g] = ao_mos_i[g] * ao_mos_j[g];
        }

        fftw_execute(buf->forward);

        // Vectorizable loop: apply Coulomb kernel
        #pragma omp simd aligned(vG,coulG:64)
        for (int g = 0; g < ncomplex; g++) {
            const double scale = coulG[g] * isqn;
            vG[g][0] *= scale;
            vG[g][1] *= scale;
        }

        fftw_execute(buf->backward);

        // Vectorizable loop: accumulate result with better memory access
        double * const vR_dm_i = &vR_dm[idx_i];
        #pragma omp simd aligned(vR_dm_i,vR,ao_mos_j:64)
        for (int g = 0; g < ngrids; g++) {
            vR_dm_i[g] += vR[g] * ao_mos_j[g] * mo_occ_j_isqn;
        }
    }
}


void occri_vR(double *vR_dm, double *mo_occ, double *coulG,
                 int mesh[3], double *ao_mos, int nmo) {

    const int ngrids = mesh[0] * mesh[1] * mesh[2];
    const int nthreads = omp_get_max_threads();

    // Initialize FFTW multithreading support
    if (!fftw_init_threads()) {
        fprintf(stderr, "FFTW thread init failed\n");
        exit(1);
    }

    // Tell FFTW how many threads to use per plan (1 per buffer for thread safety)
    fftw_plan_with_nthreads(1);

    // Pre-zero the output array for better cache behavior
    #pragma omp parallel for simd
    for (int i = 0; i < nmo * ngrids; i++) {
        vR_dm[i] = 0.0;
    }

    FFTWBuffers **fftw_buffers_array = allocate_thread_fftw_buffers(mesh, nthreads);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        FFTWBuffers *buf = fftw_buffers_array[tid];

        // Use static scheduling for better load balancing with irregular workloads
        #pragma omp for schedule(static)
        for (int i = 0; i < nmo; i++) {
            integrals_uu(i, ao_mos, vR_dm, coulG, nmo, ngrids, mo_occ, mesh, buf);
        }
    }

    free_thread_fftw_buffers(fftw_buffers_array, nthreads);
    
    // Clean up FFTW threading
    fftw_cleanup_threads();
}