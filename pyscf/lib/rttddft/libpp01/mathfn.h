#ifndef INCLUDE_MATHFNH
#define INCLUDE_MATHFNH
//#include "dCmplx.h"
#include "dcmplx.h"
int i1sum(const int *a,const int n);
double d1sum(const double *a,const int n);
long l1sum(const long *a,const int n);
int i1prod(const int *a,const int n);
double d1prod(const double *a,const int n);
long l1prod(const long *a,const int n);

double int_factorial(int n);
double hfodd_factorial(int odd);
double gamma_hfint(int arg);
double *xyz_to_polar(double *ret,const double *xyz);

double factorial(int n);
long l_nCk(int n,int k);
double r_nCk(int n,int k);
double factln(int n);
double gammln(double xx);

double gamma_hfodd(int odd);
double gammp(double a,double x);
void gser(double *gamser,double a,double x,double *gln);
void gcf(double *gammcf,double a,double x, double *gln);
double BoysFn(int n,double zeta);
double BoysFn1(int n,double zeta);

double Plgndr(int l,int m,double x);//,short CondonPhase);
double dlgndr(int l,double x);
double ddlgndr(int l,double x);

double glaguerre(int n, double alph, double x); // generalized Lauguerre

dcmplx zSpher(int l,int m,double theta,double phi);
double dSpher(int l,int m,double theta,double phi);

double *set_u_hydrogenlike(double *unl,double *rA, int Nr,int n, int l, double Z);

#endif
