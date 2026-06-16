#ifndef _INCLUDE_govlp_H
#define _INCLUDE_govlp_H
dcmplx Gaussian_ovlpNew_B(Gaussian *lhs,Gaussian *rhs,const double *VecField,const double *displ);
dcmplx Gaussian_ovlpNew(Gaussian *lhs,Gaussian *rhs,const double *VecField,const double *displ);
dcmplx Gaussian_ovlpNewX(Gaussian *lhs,Gaussian *rhs,const int ixyz,const dcmplx overlap0,const double *VecField,const double *displ);
dcmplx pgovlp(ushort *xpow_A,const double alph,const double *A,
              ushort *xpow_B,const double beta,const double *B,const double *Vecfield);
dcmplx pgovlp_(ushort *xpow_A,const double alph,const double *A,
              ushort *xpow_B,const double beta,const double *B,const double *Vecfield,const int verbose);
void test_ovlpx(Gaussian **BS1,int nBS1, Gaussian **BS2, int nBS2, const double *BravaisVectors);
dcmplx nmrinteg_ovlp1(double h,Gaussian *lhs,Gaussian *rhs,const double scale,const int Ixyz_or_0,const double *VecField,const double *displ,
                      double *pWalltime, double *pEdge_max, dcmplx *pEdge_sum);
dcmplx gaussian_integ1D_vcfield_(const double gamma, const int ell, const double Vmu);
dcmplx Gaussian_ovlpNew_verbose(Gaussian *lhs,Gaussian *rhs,const double *VecField,const double *displ);
dcmplx gaussian_integ1D_vcfield_verbose_(const double gamma, const int ell, const double Vmu);
int dbg_gaussian_integ1D_vcfield(const double gamma,const int elmax,const double Vmu,const double *scalefac);
#endif
