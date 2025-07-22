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

double max(int *vals, int size)
{
    int maxVal = vals[0];
    for (int i = 0; i < size; i++)
        if (vals[i] > maxVal)
            maxVal = vals[i];
    return maxVal;
}

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


// Complex FFT buffer management for k-point calculations
FFTWBuffersComplex *init_fftw_buffers_complex(int mesh[3]) {
    const int ngrids = mesh[0] * mesh[1] * mesh[2];

    FFTWBuffersComplex *buf = malloc(sizeof(FFTWBuffersComplex));
    buf->rho_c = fftw_malloc(sizeof(fftw_complex) * ngrids);
    buf->vG_c  = fftw_malloc(sizeof(fftw_complex) * ngrids);
    buf->vR_c  = fftw_malloc(sizeof(fftw_complex) * ngrids);

    // Use complex-to-complex FFTs for k-point calculations
    buf->forward_c2c = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2],
                                        buf->rho_c, buf->vG_c, FFTW_FORWARD, FFTW_PATIENT);
    
    buf->backward_c2c = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2],
                                         buf->vG_c, buf->vR_c, FFTW_BACKWARD, FFTW_PATIENT);
    return buf;
}

void destroy_fftw_buffers_complex(FFTWBuffersComplex *buf) {
    fftw_destroy_plan(buf->forward_c2c);
    fftw_destroy_plan(buf->backward_c2c);
    fftw_free(buf->rho_c);
    fftw_free(buf->vG_c);
    fftw_free(buf->vR_c);
    free(buf);
}

FFTWBuffersComplex **allocate_thread_fftw_buffers_complex(int mesh[3], int nthreads) {
    FFTWBuffersComplex **buffers = (FFTWBuffersComplex **)malloc(sizeof(FFTWBuffersComplex *) * nthreads);
    
    // Create master plan with FFTW_PATIENT for optimal performance
    FFTWBuffersComplex *master = init_fftw_buffers_complex(mesh);
    buffers[0] = master;
    
    // For additional threads, create buffers but reuse plan wisdom
    const int ngrids = mesh[0] * mesh[1] * mesh[2];
    
    for (int i = 1; i < nthreads; ++i) {
        FFTWBuffersComplex *buf = malloc(sizeof(FFTWBuffersComplex));
        buf->rho_c = fftw_malloc(sizeof(fftw_complex) * ngrids);
        buf->vG_c  = fftw_malloc(sizeof(fftw_complex) * ngrids);
        buf->vR_c  = fftw_malloc(sizeof(fftw_complex) * ngrids);
        
        // Use FFTW_ESTIMATE since we have wisdom from master plan
        buf->forward_c2c = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2],
                                            buf->rho_c, buf->vG_c, FFTW_FORWARD, FFTW_ESTIMATE);
        buf->backward_c2c = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2],
                                             buf->vG_c, buf->vR_c, FFTW_BACKWARD, FFTW_ESTIMATE);
        buffers[i] = buf;
    }
    return buffers;
}

void free_thread_fftw_buffers_complex(FFTWBuffersComplex **buffers, int nthreads) {
    for (int i = 0; i < nthreads; ++i) {
        destroy_fftw_buffers_complex(buffers[i]);
    }
    free(buffers);
}

// Helper function for k-point exchange integrals
void integrals_uu_kpts(int j, int k, int k_prim, 
                       double *ao_k_j_real, double *ao_k_j_imag,
                       double *ao_kprim_real, double *ao_kprim_imag,
                       double *vR_dm_real, double *vR_dm_imag,
                       double *coulG, double *mo_occ_kprim,
                       double *expmikr_real, double *expmikr_imag, double *kpts, int mesh[3],
                       int *nmo, int ngrids, FFTWBuffersComplex *buf) {
    
    const double sqn = sqrt(ngrids);
    const double isqn = 1.0 / sqn;
    const int nmo_k_prim = nmo[k_prim];
    
    // Cache pointers for better compiler optimization
    fftw_complex * const rho_c = buf->rho_c;
    fftw_complex * const vG_c = buf->vG_c;
    fftw_complex * const vR_c = buf->vR_c;

    for (int i = 0; i < nmo_k_prim; i++) {
        double *ao_kprim_i_real = &ao_kprim_real[i * ngrids];
        double *ao_kprim_i_imag = &ao_kprim_imag[i * ngrids]; // take conjugate        
        const double mo_occ_i = mo_occ_kprim[i] * isqn;

        // Vectorizable loop: compute complex density with phase factor
        #pragma omp simd aligned(rho_c,expmikr_real,expmikr_imag:64)
        for (int g = 0; g < ngrids; g++) {
            // rho = conj(ao_mos_kprim[i]) * exp(-i k·r) * ao_mos_k[j]
            // Apply phase factor to ao_i: ao_i_phase = conj(ao_i) * exp(-i k·r)
            const double ao_i_phase_real = ao_kprim_i_real[g] * expmikr_real[g] - (-ao_kprim_i_imag[g]) * expmikr_imag[g];
            const double ao_i_phase_imag = ao_kprim_i_real[g] * expmikr_imag[g] + (-ao_kprim_i_imag[g]) * expmikr_real[g];
            
            // Complex multiplication: ao_i_phase * ao_j
            rho_c[g][0] = ao_i_phase_real * ao_k_j_real[g] - ao_i_phase_imag * ao_k_j_imag[g];
            rho_c[g][1] = ao_i_phase_real * ao_k_j_imag[g] + ao_i_phase_imag * ao_k_j_real[g];
        }

        fftw_execute(buf->forward_c2c);

        // Vectorizable loop: apply Coulomb kernel
        #pragma omp simd aligned(vG_c,coulG:64)
        for (int g = 0; g < ngrids; g++) {
            const double vG_real = vG_c[g][0];
            const double vG_imag = vG_c[g][1];
            const double scale = coulG[g] * isqn;
            
            // Complex multiplication: vG * coulG
            vG_c[g][0] = vG_real * scale;
            vG_c[g][1] = vG_imag * scale;
        }

        fftw_execute(buf->backward_c2c);

        // Vectorizable loop: accumulate result with phase factor
        const int idx_vR_j = j * ngrids;
        #pragma omp simd aligned(vR_c,expmikr_real,expmikr_imag:64)
        for (int g = 0; g < ngrids; g++) {
            // Apply phase factor to ao_i: ao_i_phase = ao_i * exp(+i k·r)
            // This matches i_Rg_exp.conj() in the Python version
            const double ao_i_phase_real = ao_kprim_i_real[g] *    expmikr_real[g]  - ao_kprim_i_imag[g] * (-expmikr_imag[g]);
            const double ao_i_phase_imag = ao_kprim_i_real[g] * (- expmikr_imag[g]) + ao_kprim_i_imag[g] *   expmikr_real[g];
            
            // vR_c * ao_i_phase * mo_occ (complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i)
            const double vR_real = vR_c[g][0];
            const double vR_imag = vR_c[g][1];
            
            const double contrib_real = (vR_real * ao_i_phase_real - vR_imag * ao_i_phase_imag) * mo_occ_i;
            const double contrib_imag = (vR_real * ao_i_phase_imag + vR_imag * ao_i_phase_real) * mo_occ_i;
            
            vR_dm_real[idx_vR_j + g] += contrib_real;
            vR_dm_imag[idx_vR_j + g] += contrib_imag;
        }
    }
}

void occri_vR_kpts(double *vR_dm_real, double *vR_dm_imag, 
                   double *mo_occ, double *coulG_all, int mesh[3], 
                   double *expmikr_all_real, double *expmikr_all_imag, 
                   double *kpts, double *ao_real, double *ao_imag,
                   int *nmo, int ngrids, int nk, int k_idx) {

    const int nmo_max = max(nmo, nk);
    const int nmo_k = nmo[k_idx];
    double *ao_k_real = &ao_real[k_idx * nmo_max * ngrids];
    double *ao_k_imag = &ao_imag[k_idx * nmo_max * ngrids];
    const int nthreads = omp_get_max_threads();
    
    // Initialize FFTW multithreading support
    if (!fftw_init_threads()) {
        fprintf(stderr, "FFTW thread init failed\n");
        exit(1);
    }
    
    fftw_plan_with_nthreads(1);

    // Pre-zero the output arrays
    #pragma omp parallel for simd
    for (int i = 0; i < nmo_k * ngrids; i++) {
        vR_dm_real[i] = 0.0;
        vR_dm_imag[i] = 0.0;
    }
    
    FFTWBuffersComplex **fftw_buffers_array = allocate_thread_fftw_buffers_complex(mesh, nthreads);
    
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        FFTWBuffersComplex *buf = fftw_buffers_array[tid];
        
        // Parallelize over j (orbital index)
        #pragma omp for schedule(dynamic)
        for (int j = 0; j < nmo_k; j++) {
            // Loop over all k-points for this j
            double *ao_k_j_real = &ao_k_real[j * ngrids];
            double *ao_k_j_imag = &ao_k_imag[j * ngrids];

            for (int k_prim = 0; k_prim < nk; k_prim++) {
                // Get Coulomb kernel for this k-point pair
                double *coulG = &coulG_all[k_prim * ngrids];
                double *expmikr_real = &expmikr_all_real[k_prim * ngrids];
                double *expmikr_imag = &expmikr_all_imag[k_prim * ngrids];
                
                double *ao_kprim_real = &ao_real[k_prim * nmo_max * ngrids];
                double *ao_kprim_imag = &ao_imag[k_prim * nmo_max * ngrids];
                
                double *mo_occ_kprim = &mo_occ[k_prim * nmo_max];
                
                integrals_uu_kpts(j, k_idx, k_prim, 
                                 ao_k_j_real, ao_k_j_imag,
                                 ao_kprim_real, ao_kprim_imag,
                                 vR_dm_real, vR_dm_imag,
                                 coulG, mo_occ_kprim,
                                 expmikr_real, expmikr_imag, kpts, mesh,
                                 nmo, ngrids, buf);
            }
        }
    }
    
    free_thread_fftw_buffers_complex(fftw_buffers_array, nthreads);
    fftw_cleanup_threads();
}