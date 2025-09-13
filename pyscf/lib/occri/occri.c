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
#include <stdint.h>
#include <string.h>
#include <fftw3.h>
#include <omp.h>
#include "vhf/fblas.h"

// Global plan caches
static FFTWBuffers **g_real_buffers = NULL;
static int g_real_cached_mesh[3] = {0, 0, 0};
static int g_real_cached_nthreads = 0;

static FFTWBuffersComplex **g_complex_buffers = NULL;
static int g_complex_cached_mesh[3] = {0, 0, 0};
static int g_complex_cached_nthreads = 0;

// Thread safety for cache access
static omp_lock_t g_cache_lock;

double max(int *vals, int size)
{
    int maxVal = vals[0];
    for (int i = 0; i < size; i++)
        if (vals[i] > maxVal)
            maxVal = vals[i];
    return maxVal;
}

void init_cache_lock() {
    static int lock_initialized = 0;
    if (!lock_initialized) {
        omp_init_lock(&g_cache_lock);
        lock_initialized = 1;
    }
}

// Get cached real FFTW buffers or create new ones
FFTWBuffers **get_cached_real_buffers(int mesh[3], int nthreads) {
    init_cache_lock();
    
    omp_set_lock(&g_cache_lock);
    
    // Check if we need to recreate buffers
    int need_new = (!g_real_buffers || 
                    memcmp(g_real_cached_mesh, mesh, 3 * sizeof(int)) != 0 ||
                    g_real_cached_nthreads != nthreads);
    
    if (need_new) {
        // Clean up old buffers
        if (g_real_buffers) {
            free_thread_fftw_buffers(g_real_buffers, g_real_cached_nthreads);
        }
        
        // Create new buffers
        g_real_buffers = allocate_thread_fftw_buffers(mesh, nthreads);
        memcpy(g_real_cached_mesh, mesh, 3 * sizeof(int));
        g_real_cached_nthreads = nthreads;
    }
    
    FFTWBuffers **result = g_real_buffers;
    omp_unset_lock(&g_cache_lock);
    return result;
}

// Get cached complex FFTW buffers or create new ones
FFTWBuffersComplex **get_cached_complex_buffers(int mesh[3], int nthreads) {
    init_cache_lock();
    
    omp_set_lock(&g_cache_lock);
    
    // Check if we need to recreate buffers
    int need_new = (!g_complex_buffers || 
                    memcmp(g_complex_cached_mesh, mesh, 3 * sizeof(int)) != 0 ||
                    g_complex_cached_nthreads != nthreads);
    
    if (need_new) {
        // Clean up old buffers
        if (g_complex_buffers) {
            free_thread_fftw_buffers_complex(g_complex_buffers, g_complex_cached_nthreads);
        }
        
        // Create new buffers  
        g_complex_buffers = allocate_thread_fftw_buffers_complex(mesh, nthreads);
        memcpy(g_complex_cached_mesh, mesh, 3 * sizeof(int));
        g_complex_cached_nthreads = nthreads;
    }
    
    FFTWBuffersComplex **result = g_complex_buffers;
    omp_unset_lock(&g_cache_lock);
    return result;
}

FFTWBuffers *init_fftw_buffers(int mesh[3]) {
    const int ngrids = mesh[0] * mesh[1] * mesh[2];
    const int ncomplex = mesh[0] * mesh[1] * (mesh[2]/2 + 1);

    FFTWBuffers *buf = malloc(sizeof(FFTWBuffers));
    buf->rho = fftw_malloc(sizeof(double) * ngrids);
    buf->vG  = fftw_malloc(sizeof(fftw_complex) * ncomplex);
    buf->vR  = fftw_malloc(sizeof(double) * ngrids);

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
    
    const int ngrids = mesh[0] * mesh[1] * mesh[2];
    const int ncomplex = mesh[0] * mesh[1] * (mesh[2]/2 + 1);
    
    // Use consistent FFTW_ESTIMATE strategy for all threads for better performance
    for (int i = 0; i < nthreads; ++i) {
        FFTWBuffers *buf = malloc(sizeof(FFTWBuffers));
        buf->rho = fftw_malloc(sizeof(double) * ngrids);
        buf->vG  = fftw_malloc(sizeof(fftw_complex) * ncomplex);
        buf->vR  = fftw_malloc(sizeof(double) * ngrids);
        
        // Use FFTW_ESTIMATE consistently for all threads
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
        #pragma omp simd
        for (int g = 0; g < ngrids; g++) {
            rho[g] = ao_mos_i[g] * ao_mos_j[g];
        }

        fftw_execute(buf->forward);

        // Vectorizable loop: apply Coulomb kernel
        #pragma omp simd
        for (int g = 0; g < ncomplex; g++) {
            const double scale = coulG[g] * isqn;
            vG[g][0] *= scale;
            vG[g][1] *= scale;
        }

        fftw_execute(buf->backward);

        // Vectorizable loop: accumulate result with better memory access
        double * const vR_dm_i = &vR_dm[idx_i];
        #pragma omp simd
        for (int g = 0; g < ngrids; g++) {
            vR_dm_i[g] += vR[g] * ao_mos_j[g] * mo_occ_j_isqn;
        }
    }
}


int occri_vR(double *vk_out, double *mo_coeff, double *mo_occ, double *aovals,
               double *coulG, double *overlap, int mesh[3], int nmo, int nao, 
               int ngrids, const double weight) {
    /*
     * Direct computation of exchange matrix with all operations in C.
     * 
     * This function computes:
     * 1. ao_mos = mo_coeff @ aovals (MO values on grid)
     * 2. Exchange integrals using FFT 
     * 3. Full exchange matrix construction (build_full_exchange equivalent)
     * 
     * Parameters:
     * -----------
     * vk_out : double* [nao x nao]
     *     Output exchange matrix in AO basis
     * mo_coeff : double* [nmo x nao] 
     *     MO coefficients (transposed from Python convention)
     * mo_occ : double* [nmo]
     *     MO occupation numbers
     * aovals : double* [nao x ngrids]
     *     AO values evaluated on real-space grid
     * coulG : double* [ncomplex]
     *     Coulomb kernel in G-space
     * overlap : double* [nao x nao]
     *     AO overlap matrix
     * mesh : int[3]
     *     FFT mesh dimensions [nx, ny, nz]
     * nmo, nao, ngrids : int
     *     Number of MOs, AOs, and grid points
     */
    
    const int nthreads = omp_get_max_threads();

    // Allocate aligned arrays for intermediate results
    double *ao_mos = fftw_malloc(sizeof(double) * nmo * ngrids);
    double *vR_dm = fftw_malloc(sizeof(double) * nmo * ngrids);
    double *Sa = fftw_malloc(sizeof(double) * nao * nmo);      // overlap @ mo_coeff.T
    double *vk_j = fftw_malloc(sizeof(double) * nao * nmo);    // aovals @ vR_dm.T
    double *Koo = fftw_malloc(sizeof(double) * nmo * nmo);     // mo_coeff.T @ vk_j
    
    if (!ao_mos || !vR_dm || !Sa || !vk_j || !Koo) {
        fprintf(stderr, "Memory allocation failed\n");
        return -2;
    }

    // Step 1: Compute ao_mos = mo_coeff @ aovals
    // mo_coeff is [nmo x nao], aovals is [nao x ngrids], result is [nmo x ngrids]
    #pragma omp parallel for
    for (int i = 0; i < nmo; i++) {
        for (int g = 0; g < ngrids; g++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int a = 0; a < nao; a++) {
                sum += mo_coeff[i * nao + a] * aovals[a * ngrids + g];
            }
            ao_mos[i * ngrids + g] = sum;
        }
    }

    // Step 2: Initialize vR_dm output array
    #pragma omp parallel for simd
    for (int i = 0; i < nmo * ngrids; i++) {
        vR_dm[i] = 0.0;
    }

    // Step 3: Use simple, safe buffer allocation
    // Use cached buffer allocation for better performance
    FFTWBuffers **fftw_buffers_array = get_cached_real_buffers(mesh, nthreads);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        FFTWBuffers *buf = fftw_buffers_array[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < nmo; i++) {
            integrals_uu(i, ao_mos, vR_dm, coulG, nmo, ngrids, mo_occ, mesh, buf);
        }
    }

    // Don't free cached buffers - they will be reused

    // Step 4: Compute vk_j = aovals @ vR_dm.T
    // aovals is [nao x ngrids], vR_dm is [nmo x ngrids], result is [nao x nmo]
    #pragma omp parallel for
    for (int a = 0; a < nao; a++) {
        for (int i = 0; i < nmo; i++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int g = 0; g < ngrids; g++) {
                sum += aovals[a * ngrids + g] * vR_dm[i * ngrids + g];
            }
            vk_j[a * nmo + i] = sum * weight;
        }
    }

    // Step 5: Compute Sa = overlap @ mo_coeff.T  
    // overlap is [nao x nao], mo_coeff.T is [nao x nmo], result is [nao x nmo]
    #pragma omp parallel for
    for (int a = 0; a < nao; a++) {
        for (int i = 0; i < nmo; i++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int b = 0; b < nao; b++) {
                sum += overlap[a * nao + b] * mo_coeff[i * nao + b];  // mo_coeff[i][b] = mo_coeff.T[b][i]
            }
            Sa[a * nmo + i] = sum;
        }
    }

    // Step 6: Compute Koo = mo_coeff @ vk_j
    // mo_coeff is [nmo x nao], vk_j is [nao x nmo], result is [nmo x nmo]
    #pragma omp parallel for
    for (int i = 0; i < nmo; i++) {
        for (int j = 0; j < nmo; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int a = 0; a < nao; a++) {
                sum += mo_coeff[i * nao + a] * vk_j[a * nmo + j];
            }
            Koo[i * nmo + j] = sum;
        }
    }

    // Step 7: Build full exchange matrix
    // K_uv = Sa_ui * vk_j_vi + Sa_vi * vk_j_ui - Sa_ui * Koo_ij * Sa_vj
    #pragma omp parallel for
    for (int u = 0; u < nao; u++) {
        for (int v = 0; v < nao; v++) {
            double Kuv = 0.0;
            
            // First two terms: Sa_ui * vk_j_vi + Sa_vi * vk_j_ui
            #pragma omp simd reduction(+:Kuv)
            for (int i = 0; i < nmo; i++) {
                Kuv += Sa[u * nmo + i] * vk_j[v * nmo + i];  // Sa[u][i] * vk_j[v][i]
                Kuv += Sa[v * nmo + i] * vk_j[u * nmo + i];  // Sa[v][i] * vk_j[u][i]
            }
            
            // Third term: -Sa_ui * Koo_ij * Sa_vj
            for (int i = 0; i < nmo; i++) {
                #pragma omp simd reduction(-:Kuv)
                for (int j = 0; j < nmo; j++) {
                    Kuv -= Sa[u * nmo + i] * Koo[i * nmo + j] * Sa[v * nmo + j];
                }
            }
            
            vk_out[u * nao + v] = Kuv;
        }
    }

    // Clean up
    fftw_free(ao_mos);
    fftw_free(vR_dm);
    fftw_free(Sa);
    fftw_free(vk_j);
    fftw_free(Koo);
    return 0;
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


void free_thread_fftw_buffers_complex(FFTWBuffersComplex **buffers, int nthreads) {
    for (int i = 0; i < nthreads; ++i) {
        destroy_fftw_buffers_complex(buffers[i]);
    }
    free(buffers);
}


FFTWBuffersComplex **allocate_thread_fftw_buffers_complex(int mesh[3], int nthreads) {
    FFTWBuffersComplex **buffers = (FFTWBuffersComplex **)malloc(sizeof(FFTWBuffersComplex *) * nthreads);
    
    const int ngrids = mesh[0] * mesh[1] * mesh[2];
    
    // Use consistent FFTW_ESTIMATE strategy for all threads for better performance
    for (int i = 0; i < nthreads; ++i) {
        FFTWBuffersComplex *buf = malloc(sizeof(FFTWBuffersComplex));
        buf->rho_c = fftw_malloc(sizeof(fftw_complex) * ngrids);
        buf->vG_c  = fftw_malloc(sizeof(fftw_complex) * ngrids);
        buf->vR_c  = fftw_malloc(sizeof(fftw_complex) * ngrids);
        
        // Use FFTW_ESTIMATE consistently for all threads
        buf->forward_c2c = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2],
                                            buf->rho_c, buf->vG_c, FFTW_FORWARD, FFTW_ESTIMATE);
        buf->backward_c2c = fftw_plan_dft_3d(mesh[0], mesh[1], mesh[2],
                                             buf->vG_c, buf->vR_c, FFTW_BACKWARD, FFTW_ESTIMATE);
        buffers[i] = buf;
    }
    return buffers;
}


/*
 * Compute k-point exchange integrals between occupied orbitals using complex FFT.
 * 
 * This function evaluates exchange integrals for periodic systems with k-point sampling.
 * It handles the complex phase factors arising from Bloch functions and applies the
 * Coulomb interaction in reciprocal space using complex-to-complex FFT transforms.
 * 
 * The algorithm implements:
 * 1. Form complex orbital pair density: ρ_ij(r) = conj(φ_i^{k'}(r)) * exp(-i(k-k')·r) * φ_j^k(r)
 * 2. Transform to reciprocal space: ρ̃_ij(G) = FFT[ρ_ij(r)]
 * 3. Apply Coulomb kernel: Ṽ_ij(G) = ρ̃_ij(G) * v_C(|G+k-k'|)
 * 4. Transform back to real space: V_ij(r) = IFFT[Ṽ_ij(G)]
 * 5. Contract with orbital and phase: vR_dm += V_ij(r) * conj(φ_i^{k'}(r)) * exp(+i(k-k')·r) * n_i
 *
 * Parameters:
 * -----------
 * j : int
 *     Orbital index j at k-point k
 * k : int  
 *     k-point index for orbital j
 * k_prim : int
 *     k-point index for orbital i (k')
 * ao_k_j_real/imag : double*
 *     Real/imaginary parts of orbital j at k-point k on grid
 * ao_kprim_real/imag : double*
 *     Real/imaginary parts of all orbitals at k-point k' on grid 
 * vR_dm_real/imag : double*
 *     Output arrays for exchange potential (modified in-place)
 * coulG : double*
 *     Coulomb kernel in G-space for k-point difference k-k'
 * mo_occ_kprim : double*
 *     Occupation numbers for orbitals at k-point k'
 * expmikr_real/imag : double*
 *     Phase factors exp(-i(k-k')·r) on grid
 * mesh : int[3]
 *     FFT mesh dimensions [nx, ny, nz]
 * nmo : int*
 *     Number of orbitals at each k-point
 * ngrids : int
 *     Number of real-space grid points
 * buf : FFTWBuffersComplex*
 *     Pre-allocated FFTW buffers for complex transforms
 *
 * Notes:
 * ------
 * - Uses complex-to-complex FFTs to handle k-point phase factors
 * - Applies proper normalization factors for FFTW c2c transforms
 * - Vectorized loops with OpenMP SIMD directives for performance
 * - Thread-safe when called with separate buffer instances
 */

void integrals_uu_kpts(int j, int k, int k_prim, 
                       double *ao_k_j_real, double *ao_k_j_imag,
                       double *ao_kprim_real, double *ao_kprim_imag,
                       double *vR_dm_real, double *vR_dm_imag,
                       double *coulG, double *mo_occ_kprim,
                       double *expmikr_real, double *expmikr_imag, double *kpts, int mesh[3],
                       int *nmo, int ngrids, FFTWBuffersComplex *buf) {
    
    const int nmo_k_prim = nmo[k_prim];
    
    // Cache pointers for better compiler optimization
    fftw_complex * const rho_c = buf->rho_c;
    fftw_complex * const vG_c = buf->vG_c;
    fftw_complex * const vR_c = buf->vR_c;

    for (int i = 0; i < nmo_k_prim; i++) {
        double *ao_kprim_i_real = &ao_kprim_real[i * ngrids];
        double *ao_kprim_i_imag = &ao_kprim_imag[i * ngrids]; // take conjugate        
        const double mo_occ_i = mo_occ_kprim[i];

        // Vectorizable loop: compute complex density with phase factor
        #pragma omp simd
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
        #pragma omp simd
        for (int g = 0; g < ngrids; g++) {
            const double vG_real = vG_c[g][0];
            const double vG_imag = vG_c[g][1];
            const double cG = coulG[g];
            
            // Complex multiplication: vG * coulG
            vG_c[g][0] = vG_real * cG;
            vG_c[g][1] = vG_imag * cG;
        }

        fftw_execute(buf->backward_c2c);

        // Vectorizable loop: accumulate result with phase factor
        const int idx_vR_j = j * ngrids;
        #pragma omp simd
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

/*
 * Main k-point exchange matrix evaluation using OCCRI method with OpenMP parallelization.
 *
 * This function computes exchange integrals for periodic systems with k-point sampling
 * using the Occupied Orbital Coulomb Resolution of Identity (OCCRI) method. It handles
 * complex Bloch functions, k-point phase factors, and multiple k-point interactions
 * efficiently using optimized C code with FFTW and OpenMP.
 *
 * Algorithm Overview:
 * ------------------
 * For each orbital j at k-point k:
 *   For each k-point k':
 *     For each orbital i at k':
 *       Compute exchange integral (j,k | i,k') using FFT-based Coulomb evaluation
 *       Apply k-point phase factors exp(±i(k-k')·r)
 *       Accumulate contributions to exchange potential
 *
 * Key Features:
 * -------------
 * - OpenMP parallelization over orbital indices
 * - Complex FFT handling for k-point phase factors  
 * - Efficient memory layout with pre-allocated buffers
 * - Vectorized inner loops for optimal performance
 * - Thread-safe FFTW usage with per-thread buffers
 *
 * Parameters:
 * -----------
 * vR_dm_real/imag : double*
 *     Output arrays for exchange potential in real/imaginary parts
 *     Shape: (nmo[k_idx] * ngrids)
 * mo_occ : double*
 *     Flattened occupation numbers for all k-points with nmo_max padding
 *     Layout: [k0: nmo_max values, k1: nmo_max values, ...]
 * coulG_all : double*
 *     Coulomb kernels for all k-point differences
 *     Shape: (nk * ngrids), coulG_all[k_prim*ngrids:(k_prim+1)*ngrids] = v_C(|G+k-k_prim|)
 * mesh : int[3]
 *     FFT mesh dimensions [nx, ny, nz]
 * expmikr_all_real/imag : double*
 *     Phase factors exp(-i(k-k')·r) for all k-point pairs
 *     Shape: (nk * ngrids)
 * kpts : double*
 *     k-point coordinates (currently unused but kept for interface compatibility)
 * ao_real/imag : double*
 *     Flattened orbital data for all k-points with nmo_max padding per k-point
 *     Layout: [k0: nmo_max*ngrids values, k1: nmo_max*ngrids values, ...]
 * nmo : int*
 *     Number of orbitals at each k-point, shape (nk,)
 * ngrids : int
 *     Number of real-space grid points
 * nk : int
 *     Number of k-points
 * k_idx : int
 *     Index of target k-point for which to compute exchange potential
 *
 * Memory Layout:
 * --------------
 * All input arrays use C-contiguous (row-major) ordering.
 * Orbital data is padded to nmo_max for consistent indexing across k-points.
 * Output vR_dm contains only nmo[k_idx] orbitals (no padding).
 *
 * Performance Notes:
 * ------------------
 * - Uses dynamic OpenMP scheduling for load balancing
 * - Each thread gets dedicated FFTW buffers to avoid conflicts  
 * - Memory access patterns optimized for cache efficiency
 * - SIMD vectorization applied to inner loops where possible
 *
 * Thread Safety:
 * --------------
 * Function is thread-safe when called from different threads with different
 * output arrays. Internal OpenMP parallelization handles synchronization.
 */
int occri_vR_kpts(double *vR_dm_real, double *vR_dm_imag, 
                   double *mo_occ, double *coulG_all, int mesh[3], 
                   double *expmikr_all_real, double *expmikr_all_imag, 
                   double *kpts, double *ao_real, double *ao_imag,
                   int *nmo, int ngrids, int nk, int k_idx) {

    const int nmo_max = max(nmo, nk);
    const int nmo_k = nmo[k_idx];
    double *ao_k_real = &ao_real[k_idx * nmo_max * ngrids];
    double *ao_k_imag = &ao_imag[k_idx * nmo_max * ngrids];
    const int nthreads = omp_get_max_threads();

    // Pre-zero the output arrays
    #pragma omp parallel for simd
    for (int i = 0; i < nmo_k * ngrids; i++) {
        vR_dm_real[i] = 0.0;
        vR_dm_imag[i] = 0.0;
    }
    
    // Get both real and complex buffers for mixed operations
    FFTWBuffersComplex **complex_buffers_array = get_cached_complex_buffers(mesh, nthreads);
    
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        FFTWBuffersComplex *complex_buf = complex_buffers_array[tid];
        
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
                                    nmo, ngrids, complex_buf);

            }
        }
    }
    
    // Don't free cached buffers - they will be reused
    return 0;
}

