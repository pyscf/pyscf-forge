#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "dalloc.h"
#include "macro_util.h"
// #include "dcmplx.h"
// #define dCmplx double complex
#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

#ifdef NALLOCTST
#define ALLOCMSG(T,P) fprintf(stdout,"%%c-ALLOC:%s:  %p\n",T,P)
#define ALLOCMSG3(T,P,N) fprintf(stdout,"%%c-ALLOC:%s:  %p %d\n",T,P,N)
#define DELOCMSG(T,P) fprintf(stdout,"          %%c-FREE:%s:  %p\n",T,P)
#else
#define ALLOCMSG(T,P)
#define ALLOCMSG3(T,P,N)
#define DELOCMSG(T,P)
#endif
#define CABORT(MSG) fprintf(stdout,"!E c-program exitting due to error\n");fprintf(stdout,MSG);fflush(stdout);fflush(stderr);exit(1)
#define CABORT2(MSG,PMT) fprintf(stdout,"!E c-program exitting due to error\n");fprintf(stdout,MSG,PMT);fflush(stdout);fflush(stderr);exit(1)
#define CABORT3(MSG,PMT,ARG) fprintf(stdout,"!E c-program exitting due to error\n");fprintf(stdout,MSG,PMT,ARG);fflush(stdout);fflush(stderr);exit(1)

#define CKNULPO(v,WHERE) if((void *)v == NULL){CABORT2("null pointer at %s",WHERE);}
#define CKNULPO2(v,WHERE,PMT) if((void *)v == NULL){CABORT3("null pointer at %s %d",WHERE,PMT);}

static long __Nheap_D1=0, __Nheap_D2=0,__Nheap_D3=0, __Nheap_I1=0, __Nheap_I2=0,__Nheap_I3=0,
            __Nheap_L1=0, __Nheap_L2=0,__Nheap_L3=0, __Nheap_UL1=0, __Nheap_UL2=0,__Nheap_UL3=0,
           __Nheap_US1=0, __Nheap_US2=0,__Nheap_US3=0, __Nheap_C1=0, __Nheap_C2=0,__Nheap_C3=0; 
#define __d1alc_(N) __Nheap_D1++;
#define __d2alc_(N,L) __Nheap_D2++;
#define __d3alc_(N,L,M) __Nheap_D3++;
#define __i1alc_(N) __Nheap_I1++;
#define __i2alc_(N,L) __Nheap_I2++;
#define __i3alc_(N,L,M) __Nheap_I3++;
#define __l1alc_(N) __Nheap_L1++;
#define __l2alc_(N,L) __Nheap_L2++;
#define __l3alc_(N,L,M) __Nheap_L3++;

#define __ul1alc_(N) __Nheap_UL1++;
#define __ul2alc_(N,L) __Nheap_UL2++;
#define __ul3alc_(N,L,M) __Nheap_UL3++;
#define __ush1alc_(N) __Nheap_US1++;
#define __ush2alc_(N,L) __Nheap_US2++;
#define __ush3alc_(N,L,M) __Nheap_US3++;
#define __ch1alc_(N) __Nheap_C1++;
#define __ch2alc_(N,L) __Nheap_C2++;
#define __ch3alc_(N,L,M) __Nheap_C3++;

// 2021.05.19: we remove this assertion since it now appears that not necessarily all -alloc- subroutines counts up the allocation. #define __SBTRwCHK( N ) { N--;if(N<0){printf("!E Nheap reaches negative value\n");fflush(stdout);fflush(stderr);exit(1);} }
#define __SBTRwCHK( N )
#define __d1free_   __SBTRwCHK(__Nheap_D1 )
#define __d2free_  __SBTRwCHK(__Nheap_D2 )
#define __d3free_  __SBTRwCHK(__Nheap_D3 )
#define __i1free_  __SBTRwCHK(__Nheap_I1 )
#define __i2free_  __SBTRwCHK(__Nheap_I2 )
#define __i3free_  __SBTRwCHK(__Nheap_I3 )
#define __l1free_  __SBTRwCHK(__Nheap_L1 )
#define __l2free_  __SBTRwCHK(__Nheap_L2 )
#define __l3free_  __SBTRwCHK(__Nheap_L3 )

#define __ul1free_  __SBTRwCHK(__Nheap_UL1 )
#define __ul2free_  __SBTRwCHK(__Nheap_UL2 )
#define __ul3free_  __SBTRwCHK(__Nheap_UL3 )
#define __ush1free_  __SBTRwCHK(__Nheap_US1 )
#define __ush2free_  __SBTRwCHK(__Nheap_US2 )
#define __ush3free_  __SBTRwCHK(__Nheap_US3 )
#define __ch1free_  __SBTRwCHK(__Nheap_C1 )
#define __ch2free_  __SBTRwCHK(__Nheap_C2 )
#define __ch3free_  __SBTRwCHK(__Nheap_C3 )

#define __ush2realc_(N,L,NN,LL)
#define __ul2realc_(N,L,NN,LL)

dmatr **d2matr(int N){
	dmatr **ret=(dmatr **)malloc( N*sizeof(dmatr *));
	int i;
	for(i=0;i<N;i++) ret[i]=NULL;
	return ret;
}
void free_d2matr(dmatr **v,int N){
	int i;
	for(i=0;i<N;i++){ if(v[i]!=NULL){ d1free(v[i]);v[i]=NULL;} }
	v=NULL;
}
double **d2realloc(double **old,int Lo,int No,int Ln,int Nn){
	// old[0] should be reallocated from
	if(Nn==No){ // only if in this case you can realloc 
     // realloc v[0]
     double *p=(double *)realloc( old[0], (Ln*Nn)*sizeof(double) );
     // realloc v
     double **pp=(double **) realloc( old, (Ln)*sizeof(double *) );
     pp[0]=p;
     int i;
     for(i=1;i<Ln;i++) pp[i]=pp[i-1]+Nn;
     return pp;
   }
   double **pp=d2alloc(Ln,Nn);
   int i;
   for(i=0;i<Lo;i++) memcpy(pp[i],old[i],No*sizeof(double));
   d2free(old);old=NULL;
   return pp;
}
void c_heap_check_(char *msg8){
  int kk;
  char buf9[9]; for(kk=0;kk<8;kk++)buf9[kk]=msg8[kk];buf9[8]='\0';
  printf("#c_heap_check:%s:D %ld %ld %ld  I %ld %ld %ld  L %ld %ld %ld  UL %ld %ld %ld  USh %ld %ld %ld  Ch %ld %ld %ld\n",
         buf9,  __Nheap_D1, __Nheap_D2,__Nheap_D3, __Nheap_I1, __Nheap_I2,__Nheap_I3,
                __Nheap_L1, __Nheap_L2,__Nheap_L3, __Nheap_UL1, __Nheap_UL2,__Nheap_UL3,
                __Nheap_US1, __Nheap_US2,__Nheap_US3, __Nheap_C1, __Nheap_C2,__Nheap_C3 );
  fflush(stdout);            
}

// core core+m core+2m,... should belong to ret[0],ret[1],...
double ***d3wrap2(double **core,int l,int m){
  double ***ret=(double ***)malloc( sizeof(double **)*l );
  ret[0]=core;
  int j; for(j=1;j<l;j++) ret[j]=ret[j-1]+m;
  return ret;
}

/*
dCmplx *z1alloc(int n){
	dCmplx *v;
	v=(dCmplx *)malloc( (size_t) (n*sizeof(dCmplx)) );
	CKNULPO2(v,"z1alloc",n);ALLOCMSG3("z",v,n);
	return v;
}
void z1free(dCmplx *v){
	CKNULPO(v,"z1free");DELOCMSG("z",v);
	free(v);
}

dCmplx **z2alloc(int n,int m){
	dCmplx **v;
	int i;
	if( n<=0 || m<=0){ 
		fprintf(stdout,"!E z2alloc illegal arg:%d %d\n",n,m);
		fflush(stdout);
		exit(1);
	}

	v=(dCmplx **)malloc( (size_t) (n*sizeof(dCmplx *)) );
	CKNULPO(v,"z2alloc-1");ALLOCMSG3("z2",v,n*m);
	
	v[0]=(dCmplx *)malloc( (size_t) (n*m*sizeof(dCmplx)) );
	CKNULPO(v[0],"z2alloc-sub");ALLOCMSG("z1(sub)",v[0]);
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc

	return v;
}
void z2free(dCmplx **v){
	CKNULPO(v,"z2free-1");CKNULPO(v[0],"z2free-2");
	
	free(v[0]);DELOCMSG("z1(sub)",v[0]);
	free(v);DELOCMSG("z2",v);
}
*/

double *d1alloc_i(int n,double a){
	double *v=d1alloc(n);
	int i;for(i=0;i<n;i++) v[i]=a;
	return v;
}
double *d1clone(const double *org,int n){
	double *cpy=d1alloc(n);
	int i;for(i=0;i<n;i++) cpy[i]=org[i];
	return cpy;
}

int *i1clone(const int *org,int n){
	int *cpy=i1alloc(n);
	int i;for(i=0;i<n;i++) cpy[i]=org[i];
	return cpy;
}

char *ch1clone(const char *org){
	int n=strlen(org);
	char *cpy=ch1alloc( n+1 );
	int i;for(i=0;i<n;i++) cpy[i]=org[i];
	cpy[n]='\0';
	return cpy;
}

int *i1alloc_i(int n,int a){
	int *v=i1alloc(n);
	int i;for(i=0;i<n;i++) v[i]=a;
	return v;
}
short *sh1alloc_i(int n,short a){
	short *v=sh1alloc(n);
	int i;for(i=0;i<n;i++) v[i]=a;
	return v;
}


double *d1alloc(int n){
	double *v;
	v=(double *)malloc( (size_t) (n*sizeof(double)) );
  __d1alc_(n)
   if( v==NULL ){ printf("d1alloc:malloc returned error for allocation of %d doubles..\n",n);fflush(stdout);fflush(stderr); }
	CKNULPO2(v,"d1alloc",n);ALLOCMSG3("d",v,n);
	return v;
}

void d1free(double *v){
	CKNULPO(v,"d1free");DELOCMSG("d",v);
  __d1free_
	free(v);
}

double **d2alloc(int n,int m){
	double **v;
	int i;
	if( n<=0 || m<=0){ 
		fprintf(stdout,"!E d2alloc illegal arg:%d %d\n",n,m);
		fflush(stdout);
		exit(1);
	}
  __d2alc_(n,m)
	v=(double **)malloc( (size_t) (n*sizeof(double *)) );
	CKNULPO(v,"d2alloc-1");ALLOCMSG3("d2",v,n*m);
	
	v[0]=(double *)malloc( (size_t) (n*m*sizeof(double)) );
	CKNULPO(v[0],"d2alloc-sub");ALLOCMSG("d1(sub)",v[0]);
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc

	return v;
}
void d2free(double **v){
	CKNULPO(v,"d2free-1");CKNULPO(v[0],"d2free-2");
	__d2free_
	free(v[0]);DELOCMSG("d1(sub)",v[0]);
	free(v);DELOCMSG("d2",v);
}
double **d2clone(double **src,int n,int m){
	double **v=d2alloc(n,m);
	int i,j;
	for(i=0;i<n;i++){ for(j=0;j<m;j++){ v[i][j]=src[i][j]; } }
	return v;
}
double **d2alloc_i(int n,int m,double v){
	int i,j;
	double **ret=d2alloc(n,m);
	for(i=0;i<n;i++)for(j=0;j<m;j++) ret[i][j]=v;
	return ret;
}


long **l2alloc(int n,int m){
  
	long **v;
	int i;
	v=(long **)malloc( (size_t) (n*sizeof(long *)) );	
	v[0]=(long *)malloc( (size_t) (n*m*sizeof(long)) );
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc
	return v;
}
void l2free(long **v){
  free(v[0]); free(v);
}

unsigned short *ush1alloc(int L){
  unsigned short *v;
  __ush1alc_(L)
	v=(unsigned short *)malloc( (size_t) (L*sizeof(unsigned short)) );
	return v;
}
void ush1free(unsigned short *v){
  __ush1free_
  free(v);
}
unsigned long *ul1alloc(int L){
  unsigned long *v;
  __ul1alc_(L)
	v=(unsigned long *)malloc( (size_t) (L*sizeof(unsigned long)) );
	return v;
}
void ul1free(unsigned long *v){
  __ul1free_
  free(v);
}

unsigned short **ush2alloc(int n,int m){
	unsigned short **v;
	int i;
  __ush2alc_(n,m)
	v=(unsigned short **)malloc( (size_t) (n*sizeof(unsigned short *)) );	
	v[0]=(unsigned short *)malloc( (size_t) (n*m*sizeof(unsigned short)) );
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc
	return v;
}
void ush2free(unsigned short **v){
  __ush2free_
  free(v[0]); free(v);
}
unsigned long **ul2alloc(int n,int m){
  __ul2alc_(n,m)
	unsigned long **v;
	int i;
	v=(unsigned long **)malloc( (size_t) (n*sizeof(unsigned long *)) );	
	v[0]=(unsigned long *)malloc( (size_t) (n*m*sizeof(unsigned long)) );
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc
	return v;
}
void ul2free(unsigned long **v){
  __ul2free_
  free(v[0]); free(v);
}
unsigned long **ul2realloc(unsigned long **v, int Lo, int No, int Ln, int Nn){
  __ul2realc_(Lo,No,Ln,Mn)
   if(Nn==No){
     // realloc v[0]
     unsigned long *p=(unsigned long *)realloc( v[0], (Ln*Nn)*sizeof(unsigned long) );
     // realloc v
     unsigned long **pp=(unsigned long **) realloc( v, (Ln)*sizeof(unsigned long *) );
     pp[0]=p;
     int i;
     for(i=1;i<Ln;i++) pp[i]=pp[i-1]+Nn;
     return pp;
   }
   unsigned long **pp=ul2alloc(Ln,Nn);
   int i;
   for(i=0;i<Lo;i++) memcpy(pp[i],v[i],No*sizeof(unsigned long));
   ul2free(v);v=NULL;
   return pp;
   
}
unsigned short **ush2realloc(unsigned short **v, int Lo, int No, int Ln, int Nn){
  __ush2realc_(Lo,No,Ln,Mn)
   if(Nn==No){
     // realloc v[0]
     unsigned short *p=(unsigned short *)realloc( v[0], (Ln*Nn)*sizeof(unsigned short) );
     // realloc v
     unsigned short **pp=(unsigned short **) realloc( v, (Ln)*sizeof(unsigned short *) );
     pp[0]=p;
     int i;
     for(i=1;i<Ln;i++) pp[i]=pp[i-1]+Nn;
     return pp;
   }
   unsigned short **pp=ush2alloc(Ln,Nn);
   int i;
   for(i=0;i<Lo;i++) memcpy(pp[i],v[i],No*sizeof(unsigned short));
   ush2free(v);v=NULL;
   return pp;
   
}

/*
XXX 22.12.2014 : These routines fiales to reproduce proper arrangement 
unsigned long **ul2realloc(unsigned long **v, int Ln, int Nn){
   // realloc v[0]
   unsigned long *p=(unsigned long *)realloc( v[0], (Ln*Nn)*sizeof(unsigned long) );
   // realloc v
   unsigned long **pp=(unsigned long **) realloc( v, (Ln)*sizeof(unsigned long *) );
   pp[0]=p;
   int i;
   for(i=1;i<Ln;i++) pp[i]=pp[i-1]+Nn;
   return pp;
}
unsigned short **ush2realloc(unsigned short **v, int Ln, int Nn){
   // realloc v[0]
   unsigned short *p=(unsigned short *)realloc( v[0], (Ln*Nn)*sizeof(unsigned short) );
   // realloc v
   unsigned short **pp=(unsigned short **) realloc( v, (Ln)*sizeof(unsigned short *) );
   pp[0]=p;
   int i;
   for(i=1;i<Ln;i++) pp[i]=pp[i-1]+Nn;
   return pp;
}
*/


/*
double ****d4wrapper(int n,int m,double ***d3){
	double ****rtn=(double ****)malloc( (size_t) (n*sizeof(double ***)) );
	int i,ij=0;
	for(i=0;i<n;i++){ 
		rtn[i]=d3+ij;ij+=m;
		// fprintf(stdout,"#%d:%p(%p) ",i,(d3+ij),(&d3[ij]));
	}
	return rtn;
}
void d4wfree(double ****d4w){
	free(d4w);DELOCMSG("d4w",d4w);
}*/

double ***d3alloc(int n,int m,int l){
  __d3alc_(n,m,l)
	double ***v;
	int i,j;
	if( n<=0 || m<=0 || l<=0 ){ 
		fprintf(stdout,"!E d3alloc illegal arg:%d %d %d\n",n,m,l);
		fflush(stdout);
		exit(1);
	}

	v=(double ***)malloc( (size_t) (n*sizeof(double **)) );
	CKNULPO(v,"d3alloc-0");ALLOCMSG("d3-0",v);
	v[0]=(double **)malloc( (size_t) (n*m*sizeof(double *)) );
	CKNULPO(v[0],"d3alloc-1");ALLOCMSG("d3-1",v[0]);
	v[0][0]=(double *)malloc( (size_t) (n*m*l*sizeof(double)) );
	i=0;
	for(j=1;j<m;j++) v[0][j]=v[0][j-1]+l;

	for(i=1;i<n;i++){
		v[i]=v[i-1]+m;
		v[i][0]=v[i-1][0]+(l*m);
 		for(j=1;j<m;j++) v[i][j]=v[i][j-1]+l;   // v[i]: addr( v[i][0] ) etc
	}
	
	return v;
}
double ***d3allocx_redundant(int Nat,const int *Nsh,int **NpGTO){
	int NshMX=Nsh[0],NshSUM=0;
	int i;
	for(i=0;i<Nat;i++)NshMX=__MAX(NshMX,Nsh[i]);
	for(i=0;i<Nat;i++)NshSUM+=Nsh[i];
	int j;
	int npgtoMX=NpGTO[0][0];
	for(i=0;i<Nat;i++)for(j=0;j<Nsh[i];j++) npgtoMX=__MAX( npgtoMX,NpGTO[i][j] );
	return d3alloc(Nat,NshMX,npgtoMX);
}
double ***d3allocx(int Nat,const int *Nsh,int **NpGTO){
// #define _Prtout_d3allocx { int Iat,Jsh;for(Iat=0;Iat<Nat;Iat++){ printf("#d3allocx:%02d: Nsh=%d \t Npgto:",Iat,Nsh[Iat]);fflush(stdout);\
// for(Jsh=0;Jsh<Nsh[Iat];Jsh++){ printf("%d ",NpGTO[Iat][Jsh]);fflush(stdout);} printf("\n");} }//
//	_Prtout_d3allocx
	double ***v=(double ***)malloc( (size_t) (Nat*sizeof(double **)) );
	int NshSUM=0;int iat;
	for(iat=0;iat<Nat;iat++)NshSUM+=Nsh[iat];
	v[0]=(double **)malloc( (size_t) ( NshSUM*sizeof(double *)) );
	int jsh;int npGTOsum=0;
	for(iat=0;iat<Nat;iat++)for(jsh=0;jsh<Nsh[iat];jsh++) npGTOsum+=NpGTO[iat][jsh];
	
	v[0][0]=(double *)malloc( (size_t) (npGTOsum*sizeof(double)) );
	iat=0;
	for(jsh=1;jsh<Nsh[iat];jsh++) v[0][jsh]=v[0][jsh-1]+NpGTO[0][jsh-1]; // iat*jsh = jsh for iat=0
	int offset=Nsh[0];
	
	for(iat=1;iat<Nat;iat++){
		v[iat]=v[iat-1]+Nsh[iat-1];
		// npgto[iat-1][jsh] = npgto[ jsh + \sum(Nsh[<(iat-1)])
		int npgto1=0; for(jsh=0;jsh<Nsh[iat-1];jsh++) npgto1+=NpGTO[iat-1][jsh];
		v[iat][0]=v[iat-1][0]+npgto1;
 		for(jsh=1;jsh<Nsh[iat];jsh++) v[iat][jsh]=v[iat][jsh-1]+NpGTO[iat][jsh-1];   // v[i]: addr( v[i][0] ) etc
	}
	int dbgng=1;
	if(dbgng){
		for(iat=0;iat<Nat;iat++){
			double **p=v[iat];if(iat){ double **pp=v[iat-1]; long lp=((long)p),lpp=(long)pp;
																	__assertf( lp== lpp + Nsh[iat-1]*sizeof(double *),("E001:%ld / %ld",lp,lpp),-1);}
			for(jsh=0;jsh<Nsh[iat];jsh++){
				if(jsh){ double *p=v[iat][jsh],*pp=v[iat][jsh-1];long lp=((long)p),lpp=(long)pp;
																	__assertf( lp== lpp + NpGTO[iat][jsh-1]*sizeof(double ),("E002:%ld / %ld %d/%d",lp,lpp,NpGTO[iat][jsh-1],(lp-lpp)/sizeof(double )),-1);}
				else if(iat){
					if( Nsh[iat-1]<1 )continue; // 2021.07.22  Nsh[i] can be zero ...
					double *p=v[iat][0],*pp=v[iat-1][ Nsh[iat-1]-1 ];long lp=((long)p),lpp=(long)pp;
							__assertf( lp== lpp + NpGTO[iat-1][Nsh[iat-1]-1]*sizeof(double ),("E003:%ld / %ld",lp,lpp),-1);
				}
			}
		}
	}
	return v;
}


void d3free(double ***v){
  __d3free_

	CKNULPO(v,"d3free-1");CKNULPO(v[0],"d3free-2");CKNULPO(v[0][0],"d3free-3");
	free(v[0][0]);DELOCMSG("d3(sub2)",v[0][0]);	
	free(v[0]);DELOCMSG("d3(sub)",v[0]);
	free(v);DELOCMSG("d3",v);
}



unsigned long ***ul3alloc(int n,int m,int l){
  __ul3alc_(n,m,l)
	unsigned long  ***v;
	int i,j;
	if( n<=0 || m<=0 || l<=0 ){ 
		fprintf(stdout,"!E d3alloc illegal arg:%d %d %d\n",n,m,l);
		fflush(stdout);
		exit(1);
	}

	v=(unsigned long  ***)malloc( (size_t) (n*sizeof(unsigned long  **)) );
	v[0]=(unsigned long  **)malloc( (size_t) (n*m*sizeof(unsigned long  *)) );
	v[0][0]=(unsigned long  *)malloc( (size_t) (n*m*l*sizeof(unsigned long )) );
	i=0;
	for(j=1;j<m;j++) v[0][j]=v[0][j-1]+l;

	for(i=1;i<n;i++){
		v[i]=v[i-1]+m;
		v[i][0]=v[i-1][0]+(l*m);
 		for(j=1;j<m;j++) v[i][j]=v[i][j-1]+l;   // v[i]: addr( v[i][0] ) etc
	}
	
	return v;
}
void ul3free(unsigned long ***v){
  __ul3free_
	free(v[0][0]);	
	free(v[0]);
	free(v);
}

char ***ch3alloc(int n,int m,int l){
  __ch3alc_(n,m,l)
	char ***v;
	int i,j;
	if( n<=0 || m<=0 || l<=0 ){ 
		fprintf(stdout,"!E ch3alloc illegal arg:%d %d %d\n",n,m,l);
		fflush(stdout);
		exit(1);
	}

	v=(char ***)malloc( (size_t) (n*sizeof(char **)) );
	v[0]=(char **)malloc( (size_t) (n*m*sizeof(char *)) );
	v[0][0]=(char *)malloc( (size_t) (n*m*l*sizeof(char)) );
	i=0;
	for(j=1;j<m;j++) v[0][j]=v[0][j-1]+l;

	for(i=1;i<n;i++){
		v[i]=v[i-1]+m;
		v[i][0]=v[i-1][0]+(l*m);
 		for(j=1;j<m;j++) v[i][j]=v[i][j-1]+l;   // v[i]: addr( v[i][0] ) etc
	}
	
	return v;
}
void ch3free(char ***v){
  __ch3free_
	free(v[0][0]);DELOCMSG("ch3(sub2)",v[0][0]);	
	free(v[0]);DELOCMSG("ch3(sub)",v[0]);
	free(v);DELOCMSG("ch3",v);
}



long *l1alloc(int n){
  __l1alc_(n)
	long *v;
	v=(long *)malloc( (size_t) (n*sizeof(long)) );
	CKNULPO(v,"L1alloc");ALLOCMSG("L",v);
	return v;
}
void l1free(long *v){
  __l1free_
	CKNULPO(v,"L1free");DELOCMSG("L",v);
	free(v);
}

int *i1alloc(int n){
  __i1alc_(n)
	int *v;
	v=(int *)malloc( (size_t) (n*sizeof(int)) );
	CKNULPO(v,"i1alloc");ALLOCMSG("i",v);
	return v;
}
void i1free(int *v){
  __i1free_
	CKNULPO(v,"i1Free");DELOCMSG("i",v);
	free(v);
}
void **vo2alloc(int n,int len){
	void **v;
	int i;
	v=(void **)malloc( (size_t) (n*sizeof(void *)) );	
	v[0]=(void *)malloc( (size_t) (n*len) );
	for(i=1;i<n;i++) v[i]=v[i-1]+len;   // v[i]: addr( v[i][0] ) etc
	return v;
}

int **i2alloc(int n,int m){
  __i2alc_(n,m)
	int **v;
	int i;
	v=(int **)malloc( (size_t) (n*sizeof(int *)) );	
	v[0]=(int *)malloc( (size_t) (n*m*sizeof(int)) );
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc
	return v;
}
void i2free(int **v){
  __i2free_
  free(v[0]); free(v);
}
int **i2allocx_redundant(int L,const int *Nd){
	// redundant version
	int NdMX=Nd[0];
	int i;
	for(i=0;i<L;i++)NdMX=__MAX(NdMX,Nd[i]);
	return i2alloc(L,NdMX);
}

int **i2allocx(int L,const int *Nd){
	int **v;
	int i;
	v=(int **)malloc( (size_t) (L*sizeof(int *)) );//printf("#i2alloc:i2:%p\n",v);
	int Nsum=0;for(i=0;i<L;i++)Nsum+=Nd[i];
	v[0]=(int *)malloc( (size_t) (Nsum*sizeof(int)) );//printf("#i2alloc:i1:%p\n",v[0]);
	for(i=1;i<L;i++){ v[i]=v[i-1]+Nd[i-1]; }//printf("#i2alloc:i1_%d:%p\n",i,v[i]);fflush(stdout);}
	return v;
}

char *ch1alloc(int n){
  __ch1alc_(n)
	char *v;
	v=(char *)malloc( (size_t) (n*sizeof(char)) );
	return v;
}
void ch1free(char *v){
	free(v);
}
char *ch1alloc_i(int n,char c){
	char *v=ch1alloc(n);
	int i; for(i=0;i<n;i++) v[i]=c;
	return v;
}

char **ch2alloc_i(int n,int m,char v){
  char **ret=ch2alloc(n,m);
  int i,j;
  for(i=0;i<n;i++)for(j=0;j<m;j++)ret[i][j]=v;
  return ret;
}
char **ch2alloc(int n,int m){
  __ch2alc_(n,m)
	char **v;
	int i;
	if( n<=0 || m<=0){ 
		fprintf(stdout,"!E ch2alloc illegal arg:%d %d\n",n,m);
		fflush(stdout);
		exit(1);
	}

	v=(char **)malloc( (size_t) (n*sizeof(char *)) );
	CKNULPO(v,"ch2alloc-1");ALLOCMSG("ch2",v);
	
	v[0]=(char *)malloc( (size_t) (n*m*sizeof(char)) );
	CKNULPO(v[0],"ch2alloc-sub");ALLOCMSG("ch1(sub)",v[0]);
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc

	return v;
}
void ch2free(char **v){
  __ch2free_
	CKNULPO(v,"ch2free-1");CKNULPO(v[0],"ch2free-2");
	
	free(v[0]);DELOCMSG("ch1(sub)",v[0]);
	free(v);DELOCMSG("ch2",v);
}
// Caution for realloc of higher dim---
// do not do this...
//             01234...  |N.....
// original :  Apple     |Banana
// new         Apple      Bana|na...
//                         N+3 N+4
char **ch2realloc(char **ptr,int L0, int N0, int L,int N){
   if( N0 == N ) {
      char **ret=(char **)malloc( sizeof(char *)*L );
      ret[0]=(char *) realloc( ptr[0], sizeof(char)*L*N );
      int i; for(i=1;i<L;i++) ret[i]=ret[i-1]+N;
      free(ptr); return ret;
   } else {
      char **ret=ch2alloc(L,N);   
      int i,nszcpy=(N<N0 ? N:N0)*sizeof(char), Lcpy=(L<L0 ? L:L0); 
      for(i=0;i<Lcpy;i++)memcpy(ret[i],ptr[i],nszcpy);
      ch2free(ptr); return ret;
   }
}
double ***d3alloc_i(int L,int M,int N,double v){
  int i,j,k; double ***ret=d3alloc(L,M,N);
  for(i=0;i<L;i++)for(j=0;j<M;j++)for(k=0;k<N;k++) ret[i][j][k]=v;
  return ret;
}



short *sh1alloc(int n){
  //__i1alc_(n)
	short *v;
	v=(short *)malloc( (size_t) (n*sizeof(short)) );
	CKNULPO(v,"sh1alloc");ALLOCMSG("i",v);
	return v;
}
short *sh1clone(const short *a,int n){
	short *v=sh1alloc(n);
	int kk;
	for(kk=0;kk<n;kk++){ v[kk]=a[kk];}
	return v;
}
void sh1free(short *v){
  //__i1free_
	CKNULPO(v,"sh1Free");DELOCMSG("i",v);
	free(v);
}
short **sh2alloc(int n,int m){
  //__sh2alc_(n,m)
	short **v;
	int i;
	v=(short **)malloc( (size_t) (n*sizeof(short *)) );	
	v[0]=(short *)malloc( (size_t) (n*m*sizeof(short)) );
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc
	return v;
}
void sh2free(short **v){
  //__sh2free_
  free(v[0]); free(v);
}

double **d2wrap(double *dat,int L, int N){
	double **ret=(double **)malloc(L*sizeof(double *));
	ret[0]=dat;
	int i;for(i=1;i<L;i++) ret[i]=ret[i-1]+N;
	return ret;
}


/*
dCmplx *z1alloc(int n){
	dCmplx *v;
	v=(dCmplx *)malloc( (size_t) (n*sizeof(dCmplx)) );
	CKNULPO(v,"z1alloc");ALLOCMSG("Z",v);
	return v;
}
void z1free(dCmplx *v){
	CKNULPO(v,"z1free");DELOCMSG("Z",v);
	free(v);
}

dCmplx **z2alloc(int n,int m){
  
	dCmplx **v;
	int i;
	v=(dCmplx **)malloc( (size_t) (n*sizeof(dCmplx *)) );	
	v[0]=(dCmplx *)malloc( (size_t) (n*m*sizeof(dCmplx)) );
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc
	return v;
}
void z2free(dCmplx **v){
  free(v[0]); free(v);
}


dCmplx ***z3alloc(int n,int m,int l){
	dCmplx ***v;
	int i,j;
	if( n<=0 || m<=0 || l<=0 ){ 
		fprintf(stdout,"!E z3alloc illegal arg:%d %d %d\n",n,m,l);
		fflush(stdout);
		exit(1);
	}

	v=(dCmplx ***)malloc( (size_t) (n*sizeof(dCmplx **)) );
	v[0]=(dCmplx **)malloc( (size_t) (n*m*sizeof(dCmplx *)) );
	v[0][0]=(dCmplx *)malloc( (size_t) (n*m*l*sizeof(dCmplx)) );
	i=0;
	for(j=1;j<m;j++) v[0][j]=v[0][j-1]+l;

	for(i=1;i<n;i++){
		v[i]=v[i-1]+m;
		v[i][0]=v[i-1][0]+(l*m);
 		for(j=1;j<m;j++) v[i][j]=v[i][j-1]+l;   // v[i]: addr( v[i][0] ) etc
	}
	
	return v;
}
void z3free(dCmplx ***v){
	free(v[0][0]);DELOCMSG("z3(sub2)",v[0][0]);	
	free(v[0]);DELOCMSG("z3(sub)",v[0]);
	free(v);DELOCMSG("z3",v);
}
*/


double ****d4alloc(int I,int J,int K,int L){
/*
   v[I][J][K] : I*J*K double pointers. each separated by L
                v[i][j][k]=v[i][j][k-1]+L
                v[i][j][0]=v[i][j-1][0]+K*L
                v[i][0][0]=v[i-1][0][0]+J*K*L
   v[I][J]    :I*J double* pointers each separated by K             
                v[i][j] =v[i][j-1]+K
                v[i][0] =v[i-1][0]+J*K
   d3 construction applies for double* part             
	for(j=1;j<J;j++) v[0][j]=v[0][j-1]+K;
	for(i=1;i<I;i++){
		v[i]=v[i-1]+J;
		v[i][0]=v[i-1][0]+(J*K);
 		for(j=1;j<J;j++) v[i][j]=v[i][j-1]+K;}

 */
	double ****v;
	int i,j,k;
	if( I<=0 || J<=0 || K<=0 || L<=0 ){ 
		fprintf(stdout,"!E d4alloc illegal arg:%d %d %d\n",I,J,K);
		fflush(stdout);
		exit(1);
	}
	v=(double ****)malloc( (size_t) (I*sizeof(double ***)) );
	CKNULPO(v,"d4alloc-0");ALLOCMSG("d4-0",v);
	v[0]=(double ***)malloc( (size_t) (I*J*sizeof(double **)) );
	CKNULPO(v[0],"d4alloc-1");ALLOCMSG("d4-1",v[0]);
	v[0][0]=(double **)malloc( (size_t) (I*J*K*sizeof(double *)) );
	CKNULPO(v[0][0],"d4alloc-2");ALLOCMSG("d4-2",v[0][0]);
	v[0][0][0]=(double *)malloc( (size_t) (I*J*K*L*sizeof(double)) );
	CKNULPO(v[0][0][0],"d4alloc-3");ALLOCMSG("d4-3",v[0][0][0]);
	i=0;
	for(k=1;k<K;k++){
		v[0][0][k]=v[0][0][k-1]+L; //[0][0][*]
	}
	for(j=1;j<J;j++) {
		v[0][j]=v[0][j-1]+K;      //[0][*]
		v[0][j][0]=v[0][j-1][0]+(K*L); //[0][*][0]
		for(k=1;k<K;k++){
			v[0][j][k]=v[0][j][k-1]+L;   //[0][*][*]
		}
	}
	for(i=1;i<I;i++){
		// we need v[0], v[0][0], v[0][0][0]
		v[i]=v[i-1]+J;
		v[i][0]=v[i-1][0]+(J*K);         // we have I*  J*K double* s//[*][0]
		for(j=1;j<J;j++){
			v[i][j]=v[i][j-1]+K;           //[*][*]
		}	
		v[i][0][0]=v[i-1][0][0]+(J*K*L); // we have I   J*K*L double s		
																			//[*][0][0]
		for(k=1;k<K;k++) v[i][0][k]=v[i][0][k-1]+L;	//[*][0][*]
 		for(j=1;j<J;j++) {
			// we need v[i][0].  v[i][0][0]
 			v[i][j][0]=v[i][j-1][0]+(K*L);             //[*][*][0]
 			// we need v[i][j][0]
 			for(k=1;k<K;k++) v[i][j][k]=v[i][j][k-1]+L;//[*][*][*]
 		}
	}
	return v;
}

void d4free(double ****v){
	free(v[0][0][0]);
	free(v[0][0]);
	free(v[0]);
	free(v);
}

int ***i3alloc(int n,int m,int l){
	int ***v;
	int i,j;
	if( n<=0 || m<=0 || l<=0 ){ 
		fprintf(stdout,"!E d3alloc illegal arg:%d %d %d\n",n,m,l);
		fflush(stdout);
		exit(1);
	}

	v=(int ***)malloc( (size_t) (n*sizeof(int **)) );
	CKNULPO(v,"d3alloc-0");ALLOCMSG("d3-0",v);
	v[0]=(int **)malloc( (size_t) (n*m*sizeof(int *)) );
	CKNULPO(v[0],"d3alloc-1");ALLOCMSG("d3-1",v[0]);
	v[0][0]=(int *)malloc( (size_t) (n*m*l*sizeof(int)) );
	i=0;
	for(j=1;j<m;j++) v[0][j]=v[0][j-1]+l;

	for(i=1;i<n;i++){
		v[i]=v[i-1]+m;
		v[i][0]=v[i-1][0]+(l*m);
 		for(j=1;j<m;j++) v[i][j]=v[i][j-1]+l;   // v[i]: addr( v[i][0] ) etc
	}
	
	return v;
}
int ***i3alloc_i(int n,int m,int l,int v){
	int i,j,k;
	int ***ret=i3alloc(n,m,l);
	for(i=0;i<n;i++){ for(j=0;j<m;j++){ for(k=0;k<l;k++){ ret[i][j][k]=v;} } }
	return ret;
}
