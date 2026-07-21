#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif


void GBPDFTcontract_alpha_core(double *ci, int *conf_info_list, double *ov_list,
                             double *t1a, int *t1a_nz, int t1a_nz_size,
                             int ncas, int na, int nb, int ngroup,
                             double *Ka)
{
    int kpair = ngroup * ngroup;
    size_t ka_size = (size_t)ncas * ncas * kpair;
    for (size_t idx = 0; idx < ka_size; idx++) {
        Ka[idx] = 0.0;
    }

    for (int inz = 0; inz < t1a_nz_size; inz++) {
        int a = t1a_nz[inz * 4 + 0];
        int i = t1a_nz[inz * 4 + 1];
        int x = t1a_nz[inz * 4 + 2];
        int y = t1a_nz[inz * 4 + 3];
        double sign = t1a[((size_t)a * ncas + i) * na * na
                         + (size_t)x * na + y];

        for (int b = 0; b < nb; b++) {
            int p1 = conf_info_list[x * nb + b];
            int p2 = conf_info_list[y * nb + b];
            int pid = p1 * ngroup + p2;
            Ka[((size_t)a * ncas + i) * kpair + pid] +=
                sign * ci[x * nb + b] * ci[y * nb + b]
                * ov_list[p1 * ngroup + p2];
        }
    }
}


void GBPDFTcontract_beta_core(double *ci, int *conf_info_list, double *ov_list,
                            double *t1b, int *t1b_nz, int t1b_nz_size,
                            int ncas, int na, int nb, int ngroup,
                            double *Kb)
{
    int kpair = ngroup * ngroup;
    size_t kb_size = (size_t)ncas * ncas * kpair;
    for (size_t idx = 0; idx < kb_size; idx++) {
        Kb[idx] = 0.0;
    }

    for (int inz = 0; inz < t1b_nz_size; inz++) {
        int a = t1b_nz[inz * 4 + 0];
        int i = t1b_nz[inz * 4 + 1];
        int u = t1b_nz[inz * 4 + 2];
        int v = t1b_nz[inz * 4 + 3];
        double sign = t1b[((size_t)a * ncas + i) * nb * nb
                         + (size_t)u * nb + v];

        for (int x = 0; x < na; x++) {
            int p1 = conf_info_list[x * nb + u];
            int p2 = conf_info_list[x * nb + v];
            int pid = p1 * ngroup + p2;
            Kb[((size_t)a * ncas + i) * kpair + pid] +=
                sign * ci[x * nb + u] * ci[x * nb + v]
                * ov_list[p1 * ngroup + p2];
        }
    }
}


void GBPDFTcontract_active_pair(double *ci, int *conf_info_list, double *ov_list,
                              double *t1a, double *t1b,
                              int *t1a_nz, int t1a_nz_size,
                              int *t1b_nz, int t1b_nz_size,
                              int ncas, int na, int nb, int ngroup,
                              double *out)
{
    size_t out_size = (size_t)ncas * ncas * ncas * ncas;
    for (size_t idx = 0; idx < out_size; idx++) {
        out[idx] = 0.0;
    }

    for (int ia_nz = 0; ia_nz < t1a_nz_size; ia_nz++) {
        int a = t1a_nz[ia_nz * 4 + 0];
        int i = t1a_nz[ia_nz * 4 + 1];
        int x = t1a_nz[ia_nz * 4 + 2];
        int u = t1a_nz[ia_nz * 4 + 3];
        double sa = t1a[((size_t)a * ncas + i) * na * na
                       + (size_t)x * na + u];

        for (int ib_nz = 0; ib_nz < t1b_nz_size; ib_nz++) {
            int b = t1b_nz[ib_nz * 4 + 0];
            int j = t1b_nz[ib_nz * 4 + 1];
            int y = t1b_nz[ib_nz * 4 + 2];
            int v = t1b_nz[ib_nz * 4 + 3];
            double sb = t1b[((size_t)b * ncas + j) * nb * nb
                           + (size_t)y * nb + v];

            int p1 = conf_info_list[x * nb + y];
            int p2 = conf_info_list[u * nb + v];
            out[(((size_t)a * ncas + i) * ncas + b) * ncas + j] +=
                ci[x * nb + y] * ci[u * nb + v]
                * ov_list[p1 * ngroup + p2] * sa * sb;
        }
    }
}


void GBPDFTcontract_core_pair(double *M, double *Ma,
                            double *Dgroup_pack, int ndiag,
                            double *Dpair_pack, int kpair,
                            double *Ka, double *Kb, double *wgroup,
                            int ngrids, int nmo, int ncas, double *out)
{
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ngrids; g++) {
        double value = 0.0;

        for (int kd = 0; kd < ndiag; kd++) {
            double qval = 0.0;
            for (int p = 0; p < nmo; p++) {
                double mp = M[(size_t)g * nmo + p];
                for (int q = 0; q < nmo; q++) {
                    qval += mp * Dgroup_pack[((size_t)kd * nmo + p) * nmo + q]
                            * M[(size_t)g * nmo + q];
                }
            }
            value += wgroup[kd] * qval * qval;
        }

        for (int k = 0; k < kpair; k++) {
            double qval = 0.0;
            for (int p = 0; p < nmo; p++) {
                double mp = M[(size_t)g * nmo + p];
                for (int q = 0; q < nmo; q++) {
                    qval += mp * Dpair_pack[((size_t)k * nmo + p) * nmo + q]
                            * M[(size_t)g * nmo + q];
                }
            }

            double aval = 0.0;
            for (int a = 0; a < ncas; a++) {
                double ma = Ma[(size_t)g * ncas + a];
                for (int i = 0; i < ncas; i++) {
                    aval += ma * Ma[(size_t)g * ncas + i]
                            * (Ka[((size_t)a * ncas + i) * kpair + k]
                               + Kb[((size_t)a * ncas + i) * kpair + k]);
                }
            }
            value += qval * aval;
        }

        out[g] = value;
    }
}
