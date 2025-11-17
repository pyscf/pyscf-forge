/*
 * Header for OCCRI C extension
 *
 * Defines the FFTWBuffers struct for thread-local FFT plans and storage.
 * Declares the public interface functions for allocation, deallocation,
 * and contraction of integrals.
 *
 * To be included in the shared object compiled and linked with PySCF.
 */

#ifndef OCCRI_H
#define OCCRI_H

#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    fftw_plan forward;
    fftw_plan backward;
    double *rho;
    fftw_complex *vG;
    double *vR;
} FFTWBuffers;

typedef struct {
    fftw_plan forward_c2c;
    fftw_plan backward_c2c;
    fftw_complex *rho_c;
    fftw_complex *vG_c;
    fftw_complex *vR_c;
} FFTWBuffersComplex;

FFTWBuffers *init_fftw_buffers(int mesh[3]);
void destroy_fftw_buffers(FFTWBuffers *buf);
FFTWBuffers **allocate_thread_fftw_buffers(int mesh[3], int nthreads);
void free_thread_fftw_buffers(FFTWBuffers **buffers, int nthreads);

FFTWBuffersComplex *init_fftw_buffers_complex(int mesh[3]);
void destroy_fftw_buffers_complex(FFTWBuffersComplex *buf);
FFTWBuffersComplex **allocate_thread_fftw_buffers_complex(int mesh[3], int nthreads);
void free_thread_fftw_buffers_complex(FFTWBuffersComplex **buffers, int nthreads);

int occri_vR(double *vk_out, double *mo_coeff, double *mo_occ, double *aovals,
              double *coulG, double *overlap, int mesh[3], int nmo, int nao, int ngrids, double weight);

int occri_vR_kpts(double *vR_dm_real, double *vR_dm_imag, 
                   double *mo_occ, double *coulG_all, int mesh[3], 
                   double *expmikr_all_real, double *expmikr_all_imag, 
                   double *kpts, double *ao_real, double *ao_imag,
                   int *nmo, int ngrids, int nk, int k_idx);



#ifdef __cplusplus
}
#endif

#endif  // OCCRI_H