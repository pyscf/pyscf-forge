#ifndef _INCLUDE_dcmplx_H
#define _INCLUDE_dcmplx_H
#include <complex.h>
#undef I
#define _I  _Complex_I

// Note: double cabs  dcmplx conj  are available
#define dcmplx   double complex

extern void zfdot4_(dcmplx *ret,dcmplx *F, dcmplx *f1,dcmplx *f2,double *w,int *N);  // see futils.f90

#define __RandI(z)  creal(z),cimag(z)
#define __CtoRI(z)  creal(z),cimag(z)
#define __dcmplx(x,y) ((x)+_I*(y))

#define ZabsSQR(v)  ( creal(v)*creal(v) + cimag(v)*cimag(v) )
#define ZsqrDist(a,b)  ( creal((a-b))*creal((a-b)) + cimag((a-b))*cimag((a-b)) ) 
dcmplx *z1alloc(int n);
dcmplx **z2alloc(int n,int m);
dcmplx ***z3alloc(int n,int m,int l);
dcmplx ***z3alloc_i(int N1,int N2,int N3, dcmplx val);
dcmplx ****z4alloc(int I,int J,int K,int L);
dcmplx *z1alloc_i(int n,const dcmplx val);
void z1free(dcmplx *z);
void z2free(dcmplx **z);
void z3free(dcmplx ***z);
void z4free(dcmplx ****v);
#endif
