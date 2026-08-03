#ifndef _INCLUDE_xgaussope_H
#define _INCLUDE_xgaussope_H
#include "dcmplx.h"
#include "ushort.h"
#include "Gaussian.h"
#include "govlp.h"
dcmplx *calc_pbc_overlaps02(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, int MPIrank);
dcmplx *calc_pbc_overlaps02x(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank);
int calc_pbc_overlaps_f(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int filenumber);
dcmplx *calc_pbc_overlaps02_(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1,const int MPIrank,const char o_OR_x);

int write_z1buf(const char *path,const dcmplx *zbuf,const int Ld);
dcmplx **calc_pbc_overlaps(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1);
int fill_zbuf(dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin);
int fill_zbuf1(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax);
int fill_zbuf1b(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax);
int fill_zbuf1_(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr,
                Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax,
                dcmplx (*fnGaussian_ovlpNew)(Gaussian *,Gaussian *,const double *,const double *) );


int fill_zbuf1x(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
      const int ixyz,Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax,
      dcmplx ***zbuf_REFR,const int *Hj_REFR,const int *hj_REFR);
dcmplx **pbc_overlapx(int Iblock,int Ilhs,int Jrhs,Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,
                   int nKpoints,double *kvectors,
                   int *iargs,const int clear_buffer,const int Ldim_margin);
dcmplx *pbc_overlap(int Iblock,int Ilhs,int Jrhs,Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,
                   int nKpoints,double *kvectors,
                   int *iargs,const int clear_buffer,const int Ldim_margin);
dcmplx *calc_pbc_overlapsx(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1);

/*
int calc_pbc_overlaps_f(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int filenumber);
*/
Gaussian ***gen_bsetx3(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *n_cols,const int *npGTO,const double *alph,const double *cofs);
Gaussian **gen_bset(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *n_cols,const int *npGTO,const double *alph,const double *cofs, const int I_col);
Gaussian **gen_bset01(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *n_cols,const int *npGTO,const double *alph,const double *cofs,const int I_col,char verbose);
//double check_nrmz(const char *title,Gaussian **BS,int Nb,double fix_thr);
double check_nrmz01(const char *title,Gaussian **BS,int Nb,const char fix_nrmz);
double check_nrmz(const char *title,Gaussian **BS,int Nb);
double check_nrmzx(const char normalization_ForPorN,const char *title,Gaussian **BS,int Nb,double fix_thr,
                   const int spdm,const double *BravisVectors,const int Lx,const int Ly,const int Lz);
double pbc_overlap_GammaPoint(Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,int Lx,int Ly,int Lz);

/*
Gaussian **gen_bset(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *npGTO,const double *alph,const double *cofs,const int N_more);
Gaussian **gen_bset01(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *npGTO,const double *alph,const double *cofs,const int N_more,char verbose);
*/
/*
Gaussian **gen_bset01(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *npGTO,const double *alph,const double *cofs,int verbose);
Gaussian **gen_bset(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *npGTO,const double *alph,const double *cofs);

dcmplx *calc_pbc_overlaps(int nAtm,double *Rnuc,int *IZnuc,int nDa,int *distinctIZnuc,
                         const int *Nsh,const int *ell,const int *npGTO,const double *alph,const double *cofs,
                         int nAtmB,double *RnucB,int *IZnucB,int nDb,int *distinctIZnucB,
                         const int *NshB,const int *ellB,const int *n_cols,const int *npGTOB,const double *alphB,const double *cofsB,
                         int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,double *kvectors,
                         int Lx,int Ly,int Lz);
dcmplx *calc_pbc_overlaps02(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1);
dcmplx **calc_pbc_overlaps(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1);
//dcmplx *pbc_overlap(Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,int nKpoints,double *kvectors,
//                   int *iargs);
int fill_zbuf(dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin);
dcmplx *pbc_overlap(int iblock,int Ilhs,int Jrhs,Gaussian *lhs,Gaussian *rhs,int spdm,
                    double *BravisVectors,double *Vectorfield,int nKpoints,double *kvectors,
                    int *iargs,const int clear_buffer,const int Ldim_margin);*/

dcmplx Gaussian_overlap(Gaussian *lhs,Gaussian *rhs,const double *Vecfield,const double *Rdispl);
dcmplx calc_matrix( dcmplx (*ope)(ushort *,ushort *,double *,double *,ushort,double *,ushort),
       ushort *xe_pows,ushort *diffs, double *A,double *alph,ushort nga,double *args,ushort nargs);
dcmplx ope_xovlp1(ushort *diffs,ushort *Xnuc_Pows,double *A,double *alph,ushort Nga,double *args,ushort Nargs);
int calc_pbc_overlaps01(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1);
int fcountup(const char *fpath, const char incm);
int calc_pbc_overlaps_dbg(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int iflag);

int calc_pbc_overlaps_dbg_(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
       double *kvectors,  int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int iflag,const int filenumber);
dcmplx ****pbc_nabla(Gaussian **BS,const int nBS, int spdm,  double *BravaisVectors, int nKpoints,
                     double *kvectors,  int Lx,int Ly,int Lz);
int calc_nabla_f(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                 const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
								 int spdm,  double *BravaisVectors, int nKpoints,
								 double *kvectors,  int Lx,int Ly,int Lz,   const int nCGTO_1, const int MPIrank, const int filenumber);
double d1min(const double *A,int N);
Gaussian **Gaussian_nabla(const Gaussian *org);

#endif
