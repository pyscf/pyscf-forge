#ifndef INCLUDE_GAUSSIANH
#define INCLUDE_GAUSSIANH
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ushort.h"
#include "dcmplx.h"
#include "mathfn.h"
double calc_Ng(const double alpha,const ushort *pows);
double calc_Nl(const double alpha,const int l);

typedef struct struct_gaussian{
	ushort N,Jnuc,**pows;
	double *alpha,*cofs,*R,*Ng;
	char *description;
} Gaussian;

typedef struct struct_spherical_gaussian{
	ushort N,Jnuc,l; int m,I;
	double *alpha,*cofs,*R,*Nl;
	char *description;
} SphericalGaussian;

void free_Bset(Gaussian **BS,int Nb);
void free_Gaussian01(Gaussian *this,const int verbose);
void free_Gaussian(Gaussian *this);
void free_SphericalGaussian(SphericalGaussian *this);
#define Gaussian_fprtout(msg,fptr,this) { fprintf(fptr,"#Gaussian_prtout:");fprintf msg;\
fprintf(fptr,"#Nsh=%d R=(%9.4f %9.4f %9.4f)\n",this->N,this->R[0],this->R[1],this->R[2]);\
int i;for(i=0;i<(this->N);i++){ fprintf(fptr,"%d: %d.%d.%d %f %f %f\n",\
i,this->pows[i][0],this->pows[i][1],this->pows[i][2],this->alpha[i],this->cofs[i]*sqrt(this->Ng[i]),this->cofs[i]);}}

#define BSet_fprtout(msg,fptr,BS,N) { fprintf msg;\
int ib;for(ib=0;ib<(N);ib++){Gaussian_fprtout((fptr,"#BS_%d",ib),fptr,BS[ib]);}}
#define BSet_fwrite(description,fpath,BS,N) { FILE *fptr1=fopen(fpath,"w");\
BSet_fprtout((fptr1,"#%s",description),fptr1,BS,N);fclose(fptr1); }

#define BSet_prtout(msg,BS,N) { printf msg;\
int ib;for(ib=0;ib<(N);ib++){Gaussian_prtout(("#BS_%d",ib),BS[ib]);}}

#define Gaussian_prtout(msg,this) { printf("#Gaussian_prtout:");printf msg;\
printf("#Nsh=%d R=(%9.4f %9.4f %9.4f)\n",this->N,this->R[0],this->R[1],this->R[2]);\
int i;for(i=0;i<(this->N);i++){ printf("%d: %d.%d.%d %f %f %f\n",\
i,this->pows[i][0],this->pows[i][1],this->pows[i][2],this->alpha[i],this->cofs[i]*sqrt(this->Ng[i]),this->cofs[i]);}}

Gaussian *new_Gaussian(const double *alpha,const double *R,const ushort *pw,int N,const double *cof,int Jnuc);
Gaussian *new_Gaussian_1(char divide_by_Ng,const double *alpha,const double *R,const ushort *pow1,ushort **pow_a,int N,const double *cof,int Jnuc);
SphericalGaussian *new_SphericalGaussian(const double *alpha, const double *R,int l,int m,int N, const double *cof,int Jnuc);
SphericalGaussian *new_SphericalGaussianX(const double *alpha,const double *R,int l,int m,int N,const double *cof,int Jnuc,int Ipow);
Gaussian *Spherical_to_Cartesian(SphericalGaussian *src);
double Gaussian_value(Gaussian *G,const double *R);
double SphericalGaussian_value(SphericalGaussian *spg,const double *R);
int Xlm_to_Cartesian(const int l,const int m,ushort ***pows,double **cofs);
#endif
