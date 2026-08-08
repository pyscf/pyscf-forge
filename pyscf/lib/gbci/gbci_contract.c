#include <stdint.h>
#include <stddef.h>


void gbci_contract_h_spin1(double *erieff, double *ci0, double *ci1,
int ncas, int nelecasa, int nelecasb,
int *conf_info_list, int na, uint64_t *stringsa, int nb, uint64_t *stringsb,
int num, int *t1a, int *t1a_nonzero, int t1a_nonzero_size,
int *t1b, int *t1b_nonzero, int t1b_nonzero_size,
int *t2aa, int *t2aa_nonzero, int t2aa_nonzero_size,
int *t2bb, int *t2bb_nonzero, int t2bb_nonzero_size,
double *TSc, double *energy_core)
{
    (void)nelecasa;
    (void)nelecasb;
    (void)stringsa;
    (void)stringsb;

    const size_t ncas2 = (size_t)ncas * ncas;
    const size_t ncas4 = ncas2 * ncas2;
    const size_t eri_p1_stride = (size_t)num * ncas4;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < t1a_nonzero_size; i++) {
        const int aa = t1a_nonzero[i*4+0];
        const int ia = t1a_nonzero[i*4+1];
        const int str1a = t1a_nonzero[i*4+2];
        const int str0a = t1a_nonzero[i*4+3];

        for (int j = 0; j < t1b_nonzero_size; j++) {
            const int ab = t1b_nonzero[j*4+0];
            const int ib = t1b_nonzero[j*4+1];
            const int str1b = t1b_nonzero[j*4+2];
            const int str0b = t1b_nonzero[j*4+3];
            const int p1 = conf_info_list[str1a * nb + str1b];
            const int p2 = conf_info_list[str0a * nb + str0b];

            const size_t ci1_idx = (size_t)str1a * nb + str1b;
            const size_t ci0_idx = (size_t)str0a * nb + str0b;
            const size_t eri_idx = (size_t)p1 * eri_p1_stride
                + (size_t)p2 * ncas4
                + (size_t)aa * ncas * ncas * ncas
                + (size_t)ia * ncas2
                + (size_t)ab * ncas
                + ib;
            const size_t t1a_idx = (size_t)aa * ncas * na * na
                + (size_t)ia * na * na
                + (size_t)str1a * na
                + str0a;
            const size_t t1b_idx = (size_t)ab * ncas * nb * nb
                + (size_t)ib * nb * nb
                + (size_t)str1b * nb
                + str0b;

            const double contrib = ci0[ci0_idx] * erieff[eri_idx]
                * t1a[t1a_idx] * t1b[t1b_idx]
                * TSc[p1 * num + p2] * 2.0;
#ifdef _OPENMP
#pragma omp atomic
#endif
            ci1[ci1_idx] += contrib;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < t2aa_nonzero_size; i++) {
        const int a1 = t2aa_nonzero[i*6+0];
        const int i1 = t2aa_nonzero[i*6+1];
        const int a2 = t2aa_nonzero[i*6+2];
        const int i2 = t2aa_nonzero[i*6+3];
        const int str1a = t2aa_nonzero[i*6+4];
        const int str0a = t2aa_nonzero[i*6+5];

        for (int str0b = 0; str0b < nb; str0b++) {
            const int p1 = conf_info_list[str1a * nb + str0b];
            const int p2 = conf_info_list[str0a * nb + str0b];

            const size_t ci1_idx = (size_t)str1a * nb + str0b;
            const size_t ci0_idx = (size_t)str0a * nb + str0b;
            const size_t eri_idx = (size_t)p1 * eri_p1_stride
                + (size_t)p2 * ncas4
                + (size_t)a1 * ncas * ncas * ncas
                + (size_t)i1 * ncas2
                + (size_t)a2 * ncas
                + i2;
            const size_t t2aa_idx = (size_t)a1 * ncas * ncas * ncas * na * na
                + (size_t)i1 * ncas * ncas * na * na
                + (size_t)a2 * ncas * na * na
                + (size_t)i2 * na * na
                + (size_t)str1a * na
                + str0a;

            const double contrib = ci0[ci0_idx] * erieff[eri_idx]
                * t2aa[t2aa_idx] * TSc[p1 * num + p2];
#ifdef _OPENMP
#pragma omp atomic
#endif
            ci1[ci1_idx] += contrib;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < t2bb_nonzero_size; i++) {
        const int a1 = t2bb_nonzero[i*6+0];
        const int i1 = t2bb_nonzero[i*6+1];
        const int a2 = t2bb_nonzero[i*6+2];
        const int i2 = t2bb_nonzero[i*6+3];
        const int str1b = t2bb_nonzero[i*6+4];
        const int str0b = t2bb_nonzero[i*6+5];

        for (int str0a = 0; str0a < na; str0a++) {
            const int p1 = conf_info_list[str0a * nb + str1b];
            const int p2 = conf_info_list[str0a * nb + str0b];

            const size_t ci1_idx = (size_t)str0a * nb + str1b;
            const size_t ci0_idx = (size_t)str0a * nb + str0b;
            const size_t eri_idx = (size_t)p1 * eri_p1_stride
                + (size_t)p2 * ncas4
                + (size_t)a1 * ncas * ncas * ncas
                + (size_t)i1 * ncas2
                + (size_t)a2 * ncas
                + i2;
            const size_t t2bb_idx = (size_t)a1 * ncas * ncas * ncas * nb * nb
                + (size_t)i1 * ncas * ncas * nb * nb
                + (size_t)a2 * ncas * nb * nb
                + (size_t)i2 * nb * nb
                + (size_t)str1b * nb
                + str0b;

            const double contrib = ci0[ci0_idx] * erieff[eri_idx]
                * t2bb[t2bb_idx] * TSc[p1 * num + p2];
#ifdef _OPENMP
#pragma omp atomic
#endif
            ci1[ci1_idx] += contrib;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int str0a = 0; str0a < na; str0a++) {
        for (int str0b = 0; str0b < nb; str0b++) {
            const int p = conf_info_list[str0a * nb + str0b];
            const size_t ci_idx = (size_t)str0a * nb + str0b;
            ci1[ci_idx] += energy_core[p] * ci0[ci_idx];
        }
    }
}
