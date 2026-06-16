#ifndef _INCLUDE_cstrutil_H
#define _INCLUDE_cstrutil_H
#include "dcmplx.h"
int fcountup(const char *fpath, const char incm);
double z3diff(dcmplx ***a,dcmplx ***b,const int *Ndim,int *maxdiffloc,const char *title);
dcmplx ***read_z3f(char *fpath,int *Ndim);
char *write_z3(const char *fpath,dcmplx ***zbuf,const int *Ndim);
int startswith(const char *str,const char *ptn);

char *get_datetime_x(char *sbuf,const char *format);
char *get_datetime();
char *z1toa(const dcmplx *a,const int N);
char *i1toa(const int *a,const int N);
char *d1toa(const double *a,const int N);

int read_file(const char *fpath,void **bufs,char *types,int *lengths,int bfsz);
double *parse_doubles(int *pnGot,char *org);
int *parse_ints(int *N,char *org); // or parseInts_all

#endif
