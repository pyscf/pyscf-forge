#ifndef DALLOCH
#define DALLOCH
#ifndef dmatr
 #define dmatr double
#endif
#ifndef imatr
 #define imatr int
#endif
#ifndef lmatr
 #define lmatr long
#endif
int ***i3alloc(int n,int m,int l);
int ***i3alloc_i(int n,int m,int l,int v);
double ***d3allocx(int Nat,const int *Nsh,int **NpGTO);
int **i2allocx(int L,const int *Nd);
/*
// #include "dCmplx.h"
dCmplx *z1alloc(int n);
void z1free(dCmplx *v);
dCmplx **z2alloc(int n,int m);
void z2free(dCmplx **v);
dCmplx ***z3alloc(int n,int m,int l);
void z3free(dCmplx ***v);
*/
dmatr **d2matr(int N);
void free_d2matr(dmatr **v,int N);
double ****d4alloc(int I,int J,int K,int L);
void d4free(double ****v);
int *i1clone(const int *org,int n);
double **d2realloc(double **old,int Lo,int No,int Ln,int Nn);

int *i1alloc_i(int n,int a);
double *d1alloc_i(int n,double a);
double *d1alloc(int n);
void d1free(double *v);
double **d2alloc(int n,int m);
double **d2alloc_i(int n,int m,double v);
void d2free(double **v);
//double ****d4wrapper(int n,int m,double ***d3);
//void d4wfree(double ****d4w);
double ***d3alloc(int n,int m,int l);
void d3free(double ***v);

long *l1alloc(int n);
void l1free(long *v);
long **l2alloc(int n,int m);
void l2free(long **v);

int *i1alloc(int n);
void i1free(int *v);
int **i2alloc(int n,int m);
void i2free(int **v);

char *ch1alloc(int n);
void ch1free(char *v);
char **ch2alloc(int n,int m);
void ch2free(char **v);

char ***ch3alloc(int n,int m,int l);
void ch3free(char ***v);

unsigned short *ush1alloc(int L);
void ush1free(unsigned short *v);

unsigned long *ul1alloc(int L);
void ul1free(unsigned long *v);

// unsigned long **ul2realloc(unsigned long **v, int Ln, int Nn);
//unsigned short **ush2realloc(unsigned short **v, int Ln, int Nn);

unsigned long ***ul3alloc(int n,int m,int l);
void ul3free(unsigned long ***v);

unsigned short **ush2alloc(int n,int m);
//unsigned long **ul2realloc(unsigned long **v, int Ln, int Nn);
void ush2free(unsigned short **v);
unsigned long **ul2alloc(int n,int m);
void ul2free(unsigned long **v);

unsigned long **ul2realloc(unsigned long **v, int Lo, int No, int Ln, int Nn);
unsigned short **ush2realloc(unsigned short **v, int Lo, int No, int Ln, int Nn);

double ***d3wrap2(double **core,int l,int m);
char **ch2alloc_i(int n,int m,char v);
char *ch1alloc_i(int n,char c);
double ***d3alloc_i(int L,int M,int N,double v);

short *sh1alloc(int n);
void sh1free(short *v);
short **sh2alloc(int n,int m);
void sh2free(short **v);
double **d2clone(double **src,int n,int m);
char *ch1clone(const char *org);
void **vo2alloc(int n,int len);
short *sh1clone(const short *a,int n);

short *sh1alloc_i(int n,short a);
double *d1clone(const double *org,int n);

#define __D1free(v) { if((v)!=NULL)d1free(v);v=NULL; }
#define __D2free(v) { if((v)!=NULL)d2free(v);v=NULL; }
#define __D3free(v) { if((v)!=NULL)d3free(v);v=NULL; }

#define __I1free(v) { if((v)!=NULL)i1free(v);v=NULL; }
#define __I2free(v) { if((v)!=NULL)i2free(v);v=NULL; }
#define __I3free(v) { if((v)!=NULL)i3free(v);v=NULL; }

#define __Ch1free(v) { if((v)!=NULL)ch1free(v);v=NULL; }
#define __Ch2free(v) { if((v)!=NULL)ch2free(v);v=NULL; }
#define __Ch3free(v) { if((v)!=NULL)ch3free(v);v=NULL; }

double **d2wrap(double *dat,int L, int N);

#endif
