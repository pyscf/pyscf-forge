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

FFTWBuffers *init_fftw_buffers(int mesh[3]);
void destroy_fftw_buffers(FFTWBuffers *buf);
FFTWBuffers **allocate_thread_fftw_buffers(int mesh[3], int nthreads);
void free_thread_fftw_buffers(FFTWBuffers **buffers, int nthreads);

#ifdef __cplusplus
}
#endif

#endif  // OCCRI_H