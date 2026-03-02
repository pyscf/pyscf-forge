#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mathfn.h"
#include "macro_util.h"
#include "cConstants.h"
#include "dcmplx.h"
#include "dalloc.h"

int i1sum(const int *a,const int n){
	int ret=0;int i;for(i=0;i<n;i++)ret+=a[i]; return ret;
}
double d1sum(const double *a,const int n){
	double ret=0.0;int i;for(i=0;i<n;i++)ret+=a[i]; return ret;
}
long l1sum(const long *a,const int n){
	long ret=0;int i;for(i=0;i<n;i++)ret+=a[i]; return ret;
}
int i1prod(const int *a,const int n){
	int ret=0;int i;for(i=0;i<n;i++)ret*=a[i]; return ret;
}
double d1prod(const double *a,const int n){
	double ret=0.0;int i;for(i=0;i<n;i++)ret*=a[i]; return ret;
}
long l1prod(const long *a,const int n){
	long ret=0;int i;for(i=0;i<n;i++)ret*=a[i]; return ret;
}


double factorial(int n){
#define __nBuff 35
   static double buf[__nBuff]={1,1,2,6,24,120,720,5040,40320};
   static int iTop=8;
   if(n<2)return 1;
   if(n<__nBuff){
      while(iTop<n){iTop++; buf[iTop]=buf[iTop-1]*((double)iTop);}
      return buf[n];
   }
   return exp( gammln(n+1.0) );
#undef __nBuff
}
double int_factorial(int n){
	//                     0 1 2 3  4   5   6    7     8      9      10
#define N_fctrl 25
	static double retv[N_fctrl]={1,1,2,6,24,120,720,5040,40320,362880, 3628800,0,0,0,0,0,0,0,0,0, 0,0,0,0,0};
	static int n_last=10;
	if(n<=n_last) return retv[n];
	else {
		int i;// retv[10]:=10! so retv[n_last+1] = (n_last+1)*retv[n_last+1],...
		for(i=n_last+1;(i<N_fctrl && i<=n);i++){ retv[i]=retv[i-1]*i;n_last=i; }
		if( n<=n_last ){ return retv[n];}
		else { double val=retv[n_last]; 
		       for(i=n_last+1;i<=n;i++) val*=i; return val;}
	}
}
double hfodd_factorial(int odd){ // 1,3,
	if(odd%2==0){ __assertf(0,("hfodd_factorial:%d",odd),-1);}
#define sqrt_pi 1.7724538509055160272981674833411
	int p=(odd-1)/2;//1,3,5,... => 0,1,2,... and  retv[p]= Gamma((2p+1)/2) = ((2p-1)/2)*Gamma[p-1]
	static double retv[N_fctrl]={ sqrt_pi, 0.5*sqrt_pi, 0.75*sqrt_pi, 1.875*sqrt_pi, 6.5625*sqrt_pi,
	                              0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0 };
	static int n_last=4;int i;
	if(p<=n_last){ return retv[p];}
	for(i=n_last+1;(i<N_fctrl && i<=p);i++){ retv[i]=(i-0.5)*retv[i-1];n_last=i;} // retv[1]=Gamma(3/2)=(1/2)*Gamma(1/2) ...
	if(p<=n_last) return retv[p];
	double val=retv[n_last];
	for(i=n_last+1;i<=p;i++){ val=(i-0.5)*val;}
	return val;
}   
double gamma_hfint(int arg){
	if(arg%2==0){ // Gamma(n)=(n-1)!
		int i,p=arg/2;
		return int_factorial(p-1);
	} else {
		return hfodd_factorial(arg);
	}
}
void c_factorial_(double *ret,int *np){
	*ret=factorial(*np);
}
double factorial_2(int numr,int denom){
	int upl=20;
	//printf("#factorial_2:%d,%d\n",numr,denom);fflush(stdout);
	if( numr<upl && denom<upl ) return factorial(numr)/factorial(denom);
	return exp( factln(numr)-factln(denom) );
}
double *xyz_to_polar(double *ret,const double *xyz){
	double phi=atan2(xyz[1],xyz[0]);
	double rhoSQR=xyz[0]*xyz[0] + xyz[1]*xyz[1];  //$ x*x + y*y 
	double rho=sqrt( rhoSQR );
	double r=sqrt( rhoSQR + xyz[2]*xyz[2] );
	double theta=atan2(rho,xyz[2]);
	ret[0]=r;ret[1]=theta;ret[2]=phi;return ret;
}
	
long l_nCk(int n,int k){
	//printf("#l_nCk:%d,%d\n",n,k);fflush(stdout);
   return (long)( 0.50+exp( factln(n)-factln(k)-factln(n-k) ) );
}
double r_nCk(int n,int k){
	//printf("#r_nCk:%d,%d\n",n,k);fflush(stdout);
//printf("fn_nCK:%d %d:%f\n",n,k,floor( 0.50+exp( factln(n)-factln(k)-factln(n-k) ) ));
   return floor( 0.50+exp( factln(n)-factln(k)-factln(n-k) ) );
}
double factln(int n){
   static double a[101];
   __assertf(n>=0,("factln arg<0"),-1);
   if(n<=1) return 0.0;
   if(n<101) return a[n] ? a[n]:( a[n]=( gammln(n+1.0) ) );
   return gammln(n+1.0);
}
double gammln(double xx){
	const double cof[6]={76.18009172947146e+00,
	-86.50532032941677e+00,
	24.01409824083091e+00, -1.231739572450155e+00,
	0.1208650973866179e-02, -0.5395239384953e-05};
	int j;
	double x,y,tmp,ser;
	y=x=xx;
	tmp=x+5.5; tmp=tmp-(x+0.5)*log(tmp);
	ser=1.000000000190015;
	for(j=0;j<6;j++)ser+=cof[j]/++y;
	return -tmp+log(2.5066282746310005*ser/x);
}
void c_gammln_(double *ret,double *x){
	*ret=gammln(*x);
}
#define ITMAX 100
#define EPS   3.0e-07
#define FPMIN 1.0e-30
double gammp(double a,double x){
double gamser,gammcf,gln;
//if(x<0.0||a<=0)__assertf(0,("wrongArg"));
if(x<(a+1.0)){ gser(&gamser,a,x,&gln);return gamser;}
else         { gcf(&gammcf,a,x,&gln); return 1-gammcf;}
}
void gser(double *gamser,double a,double x,double *gln){
  int n; double sum,del,ap;
  *gln=gammln(a);
  if( x<=0.0 ){
    if(x<0.0){ __assertf(0,("x<0 in gser"),-1); }
    *gamser=0.0;return;
  } else {
     ap=a; del=sum=1.0/a;
     for(n=1;n<=ITMAX;n++){
        ++ap; del*=x/ap; sum+=del;
        if(fabs(del)<fabs(sum)*EPS){ *gamser=sum*exp(-x+a*log(x)-(*gln)); return;}
     }
     __assertf(0,("gser;did not reach convergence.."),-1);
  }
}
void gcf(double *gammcf,double a,double x, double *gln){
  int i;
  double an,b,c,d,del,h;
  *gln=gammln(a);   b=x+1.0-a;
  c=1.0/FPMIN;      d=1.0/b;
  h=d;
  for(i=1;i<=ITMAX;i++){
    an=-i*(i-a);b+=2.0;
    d=an*d+b; if(fabs(d)<FPMIN)d=FPMIN;
    c=b+an/c; if(fabs(c)<FPMIN)c=FPMIN;
    d=1.0/d;  del=d*c; h*=del;
    if( fabs(del-1.0)<EPS )break;
  }
  if(i>ITMAX){ __assertf(0,("gcf:did not reach convergence.."),-1); }
  *gammcf=exp(-x+a*log(x)-(*gln))*h;
}

double BoysFn(int n,double zeta){
   if( fabs(zeta)<FPMIN ) return (1.0/(2*n+1.0));
#define __nBuff 35
   //                              1/2       3/2            (i+1/2)
   static double buf[__nBuff]={ __sqrtPI, __sqrtPI*0.50, __sqrtPI*0.75, __sqrtPI*1.875};
   static int iTop=3;
   const double dhlf=0.50;
   double gamNhf;
   if(n<__nBuff){
      // Gamma( iTop+1/2 ) = Gamma( iTop-1/2 )*(iTop-1/2)
      // while(iTop<n) buf[++iTop]=buf[iTop-1]*(iTop-dhlf);
      while(iTop<n) { buf[iTop+1]=buf[iTop]*(iTop+dhlf);iTop++; }
      gamNhf=buf[n];
   } else gamNhf= exp( gammln(n+dhlf) );
   double fac; int i;
   fac=2.0*sqrt(zeta);
   for(i=0;i<n;i++)fac*=zeta;
   return gamNhf*gammp(n+dhlf,zeta)/fac;
#undef __nBuff
}

double BoysFn1(int n,double zeta){
   if( fabs(zeta)<FPMIN ) return (1.0/(2*n+1.0));
#define __nBuff 35
   //                              1/2       3/2            (i+1/2)
   static double buf[__nBuff]={ __sqrtPI, __sqrtPI*0.50, __sqrtPI*0.75, __sqrtPI*1.875};
   static int iTop=3;
   const double dhlf=0.50;
   double gamNhf;
   if(n<__nBuff){

      // Gamma( iTop+1/2 ) = Gamma( iTop-1/2 )*(iTop-1/2)
      // while(iTop<n) buf[++iTop]=buf[iTop-1]*(iTop-dhlf);
      while(iTop<n) { buf[iTop+1]=buf[iTop]*(iTop+dhlf);iTop++; }
      gamNhf=buf[n];
   } else gamNhf= exp( gammln(n+dhlf) );
   double fac; int i;
   fac=2.0*sqrt(zeta);
   for(i=0;i<n;i++)fac*=zeta;
// OLD -------------------------------------------------------
// BoysFn1:4,0.547311 :3.323351 0.132765 0.000813
// BoysFn1:4,0.547311 :3.323351 0.132765 0.000813
// #xope_Fpart_2:4 0.624473 1.264376 0.432871 0.020359  0.012713
// REVISED ------------------------------------------------------
//BoysFn1:4,0.547311 [4]:11.631728 0.132765 0.000813
//BoysFn1:4,0.547311 [4]:11.631728 0.132765 0.000813
//#xope_Fpart_2:4 0.624473 1.264376 0.432871 0.071255  0.044497

   printf("BoysFn1:%d,%f [%d]:%f %f %f\n",n,zeta, iTop, gamNhf,fac,gammp(n+dhlf,zeta));

   return gamNhf*gammp(n+dhlf,zeta)/fac;
#undef __nBuff
}

#undef ITMAX
#undef EPS  
#undef FPMIN



//#define pLgndr_O(l,m,x)   plgndr(l,m,x,0)
//#define pLgndr_P(l,m,x)   plgndr(l,m,x,1)

/*
 
*/
double Plgndr(int l,int m,double x){
	double fact,pll,pmm,pmmp1,somx2;
	int i,ll;
	double TINY=1.0E-08;
	if(m<0||m>l){
		__abort( ("illegal arg in Plgndr %d %d %f %d%d\n",l,m,x,0,0) );
	}
	if(fabs(x)>1.0){
		if(fabs(x)>1.0+TINY)
			__abort( ("illegal arg in Plgndr %d %d %f %d%d\n",l,m,x,0,0) );
		x=(x>0.0 ? x-TINY:x+TINY);
		// XXX XXX this frequently happens... XXX fprintf(stdout,"#Plgndr:!W arg modified:%f\n",x);
	}
/*	if(m<0||m>l||fabs(x)>1.0){
		__ABORT6("illegal arg in Plgndr %d %d %f\n",l,m,x,0,0)
	}*/
	pmm=1.0;
	if(m>0){
		somx2=sqrt((1.0-x)*(1.0+x));
		fact=1.0;
		for(i=1;i<=m;i++){
			// pmm*=-fact*somx2;    ! WW phase convention is different from XMOLECULE
			pmm*=fact*somx2;
			fact+=2.0;
		}
	}
	if(l==m)return pmm;
	else {
		pmmp1=x*(2*m+1)*pmm;
		if(l==(m+1))return pmmp1;
		else {
			for(ll=m+2;ll<=l;ll++){
				pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m);
				pmm=pmmp1;
				pmmp1=pll;
			}
			return pll;
		}
	}
}

double dlgndr(int l,double x){
	// recursion-1 : P'_{n+1}=xP_{n}'+(n+1)P_{n}
	double cur=0.0;
	int j;
	if(l==0) return 0.0;
	if(l==1) return 1.0;
	if(l==2) return 3.0*x;
	cur=3.0*x;
	for(j=2;j<l;j++){
		// solve for P'_{j+1}
		cur=x*cur+(j+1)*Plgndr(j,0,x);// m=0 so no distinction btw _P or _O
	}
	
	return cur;
}

double ddlgndr(int l,double x){
	// recursion-1 : P'_{n+1}=xP_{n}'+(n+1)P_{n}
	double cur=0.0;
	int j;
	if(l<2) return 0.0;
	if(l==2) return 3.0;
	if(l==3) return 15.0*x;
	cur=15.0*x;
	for(j=3;j<l;j++){
		// solve for P'_{j+1}
		cur=x*cur+(j+2)*dlgndr(j,x);
	}
	
	return cur;
}
dcmplx zSpher(int l,int m,double theta,double phi){
    double p = sqrt( ((double)(2*l+1.0)) / (4.0*PI) * factorial_2( l-abs(m), l+abs(m) ) ) * Plgndr(l,abs(m),cos(theta));
    //if( ((m+abs(m))/2)%2 )p=-p;
    if( m>0 && m%2 ) p=-p;
    return p*cos(m*phi) + _I*p*sin(m*phi);
    //return new_dCmplx( p*cos(m*phi), p*sin(m*phi));
}
/* !$ dSpher  m=0: sqrt((2l+1)/4PI) P_l(Cos) = Y_{l0}
   !$         m>0: sqrt((2l+1)/4PI) sqrt( (l-|m|)!/(l+|m|)! ) P_l^{|m|}(Cos) cos(|m|phi) = \sqrt{2} Re( Y_l^{-|m|} )    inc. (-)^{(m+|m|)/2} factor
   !$                                                                        sin(|m|phi) 
 */
double dSpher(int l,int m,double theta,double phi){
	// 
	// printf("dSpher:%d,%d\n",l,m);fflush(stdout);
	if(m==0) return sqrt( ((double)(2*l+1.0)) / (4.0*PI)                                     ) * Plgndr(l,0,cos(theta));
	double        p=sqrt( ((double)(2*l+1.0)) / (2.0*PI) * factorial_2( l-abs(m), l+abs(m) ) ) * Plgndr(l,abs(m),cos(theta));
	if(m<0) return p*sin(-m*phi);
	return p*cos(m*phi);
}


void c_dSpher_(double *ret,int *l,int *m,double *theta,double *ph){
	*ret=dSpher(*l,*m,*theta,*ph);
}


/*
void test_spher(){
	int l=2;int N=100;
	int j,k;
//	for(m=-l;m<=l;m++){
	double phi=0;
	double c,theta,c0=-1,c1=1,dc=2.0/N;
	double sum1[5]={0,0,0,0,0};double sum2[5]={0,0,0,0,0};
	for(j=0;j<=N;j++){ c=c0+dc*j;theta=acos(c);double s=sin(theta);
		dcmplx vals[5]={ zSpher(2,-2,theta,phi)[0], zSpher(2,-1,theta,phi)[0], zSpher(2,0,theta,phi)[0], zSpher(2,1,theta,phi)[0], zSpher(2,2,theta,phi)[0] };
		double comp[5]={ sqrt(15.0/(32*PI))*s*s*cos(2*phi), -sqrt(15.0/(8*PI))*s*c*cos(phi), sqrt(5.0/(16*PI))*(3*c*c-1.0),
						sqrt(15.0/(8*PI))*s*c*cos(phi),sqrt(15.0/(32*PI))*s*s*cos(2*phi)};
		printf("%lf %lf   ",theta,cos(theta));
		for(k=0;k<5;k++)printf("%lf %lf   ",vals[k],comp[k]);
		printf("\n");
		for(k=0;k<5;k++){ sum1[k]+=( (j==0 || j==N) ? vals[k]*vals[k]/2 : vals[k]*vals[k])*dc;
						  sum2[k]+=( (j==0 || j==N) ? comp[k]*comp[k]/2 : comp[k]*comp[k])*dc;
		}
	}
	printf("#Sum: ");
	for(k=0;k<5;k++)printf("%lf %lf   ",sum1[k]*2*PI,sum2[k]*2*PI);
	
}*/
void test_spher2(){
	int N=400,M=400,j,k,ib,Jb,ijb;
	double phi,dphi=2*PI/M;
	double c,theta,c0=-1,c1=1,dc=2.0/N;
	int l,m,lm,Nb=0,lmax=4;for(l=0;l<=lmax;l++){ Nb+=(l+1)*(l+1); }
	double *sum=d1alloc_i(Nb*(Nb+1)/2,0);
	double *vals=d1alloc(Nb);
	for(j=0;j<=N;j++){ c=c0+dc*j;theta=acos(c);double s=sin(theta);
		for(phi=0,k=0;k<M;k++,phi+=dphi){ 
			for(lm=0,l=0;l<=lmax;l++)for(m=-l;m<=l;m++,lm++){
				vals[lm]=dSpher(l,m,theta,phi);
			}
			for(ijb=0,ib=0;ib<Nb;ib++)for(Jb=0;Jb<=ib;Jb++,ijb++){ 
				sum[ijb]+=( (j==0||j==N) ? vals[ib]*vals[Jb]*dc*dphi/2 : vals[ib]*vals[Jb]*dc*dphi); 
			}
		}
	}
//	for(ijb=0,ib=0;ib<Nb;ib++){
	for(ijb=0,ib=0,l=0;l<=lmax;l++){ for(m=-l;m<=l;m++,ib++){  printf("%d %d:",l,m);
		for(Jb=0;Jb<=ib;Jb++,ijb++){ if(fabs(sum[ijb])<1e-6)printf("0 ");else printf("%lf ",sum[ijb]); }
		printf("\n");
	}}
}

double glaguerre(int n, double alph, double x){ // generalized Lauguerre
	double p1=1.0; // L_{0}^{a}=1
	double p2=0.0; // formally zero
	int j; double p3;
	for(j=1;j<=n;j++){
		p3=p2;
		p2=p1;
		p1=( (2*j-1+alph-x)*p2-(j-1+alph)*p3 )/j;
	}
	return p1;
}
double *set_u_hydrogenlike(double *unl,double *rA, int Nr,int n, int l, double Z){
	//
	double xi,fac;
	fac=(2*Z/n)*(2*Z/n)*(2*Z/n)*factorial(n-l-1)/(2*n*factorial(n+l));
	fac=sqrt(fac);
	printf("##set_u_hydlike:fac=%f\n",fac);
	int jr;
	double en=(double)n;
	double alph=(2*l+1);
	for(jr=0;jr<Nr;jr++){
		xi=2*Z*rA[jr]/en;
		unl[jr]=fac*pow(xi,l)*glaguerre(n-l-1,alph,xi)*exp(-0.5*xi)*rA[jr];
	}
	//for(jr=0;jr<Nr;jr++)printf("%f %f %f\n",rA[jr],unl[jr],2*rA[jr]*exp(-rA[jr]));
	return unl;
}
/*
0 0:1.000000
1 -1:0 0.999994
1 0:0 0 1.000012
1 1:0 0 0 0.999994
2 -2:0 0 0 0 1.000000
2 -1:0 0 0 0 0 0.999969
2 0:0.000014 0 0 0 0 0 1.000062
2 1:0 0 0 0 0 0 0 0.999969
2 2:0 0 0 0 0 0 0 0 1.000000
3 -3:0 0 0 0 0 0 0 0 0 1.000000
3 -2:0 0 0 0 0 0 0 0 0 0 1.000000
3 -1:0 -0.000023 0 0 0 0 0 0 0 0 0 0.999913
3 0:0 0 0.000067 0 0 0 0 0 0 0 0 0 1.000175
3 1:0 0 0 -0.000023 0 0 0 0 0 0 0 0 0 0.999913
3 2:0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000
3 3:0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000
4 -4:0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000
4 -3:0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000
4 -2:0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000
4 -1:0 0 0 0 0 -0.000077 0 0 0 0 0 0 0 0 0 0 0 0 0 0.999813
4 0:0.000062 0 0 0 0 0 0.000182 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000375
4 1:0 0 0 0 0 0 0 -0.000077 0 0 0 0 0 0 0 0 0 0 0 0 0 0.999813
4 2:0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000
4 3:0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000
4 4:0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.000000

*/
/*
#include "psGrid.h"
#include "angularGrid.h"
void test_spher3(int lmax){
	int l,m,lm,ib,Jb,ijb,Nb,Nb2;//for(l=0;l<=lmax;l++){ Nb+=(l+1)*(l+1); }
	Nb=(lmax+1)*(lmax+1);
	Nb2=Nb*(Nb+1)/2;
	double *sum= d1alloc_i(Nb2,0);
	double *fn = d1alloc(Nb);
	psGrid *grid=gen_LebedevGrid(lmax);
	int k,Ng=grid->N;
__DBGLGR( ("gen_LebedevGrid done") );
	for(k=0;k<Ng;k++){
		for(lm=0,l=0;l<=lmax;l++){ for(m=-l;m<=l;m++,lm++){ __assertf( lm<Nb,("001"),-1);
			fn[lm]=dSpher(l,m,grid->x[k],grid->y[k]);
		}}
		for(ijb=0,ib=0;ib<Nb;ib++){  __assertf( ijb<Nb2,("002"),-1);
			for(Jb=0;Jb<=ib;Jb++,ijb++)sum[ijb]+=grid->w[k]*fn[ib]*fn[Jb];
		}
	}
#define _ln10 2.3025850929940456840179914546844
#define __PrtDev(v,ref)  { if( fabs(v-ref)>1e-5 ) printf("%lf ",fabs(v-ref));\
                           else { printf("0(%d) ",(int) floor( log(fabs(v-ref))/_ln10 )+1 ); } }
                           //else { printf("%7.1e(%d) ",fabs(v-ref),(int) floor( log(fabs(v-ref))/_ln10 )+1 ); } }
    double devMx1=-1,devMx2=-1;
	for(ijb=0,ib=0;ib<Nb;ib++){
		for(Jb=0;Jb<=ib;Jb++,ijb++){ if(Jb==ib){  devMx1=__MAX( devMx1, fabs(sum[ijb]-1)); } 
		                             else      {  devMx2=__MAX( devMx2, fabs(sum[ijb]));} }//printf("%lf ",sum[ijb]);
		//printf("\n");
	}
    printf("#devMx:%d:%le %le\n",lmax,devMx1,devMx2);//#devMx:5.355638e-11 2.752572e-15
}
*/
/*
// >>> int main(int argn,char **args){
	//test_spher2();
	test_spher3(4);
	test_spher3(5);
	test_spher3(6);
	test_spher3(7);
	test_spher3(8);
	test_spher3(9);
	test_spher3(10);
	test_spher3(11);
	test_spher3(12);
	test_spher3(13);
	test_spher3(14);
	test_spher3(15);
	test_spher3(16);
	test_spher3(17);
	test_spher3(18);
	test_spher3(19);

	test_spher3(20);
	test_spher3(21);
	test_spher3(22);
	test_spher3(23);
	test_spher3(24);
	test_spher3(25);
	test_spher3(26);
	test_spher3(27);
	test_spher3(28);
	test_spher3(29);
	
}
*/
/*
ldXXXX doneDEV:0.000000: 1.000000 0.000000 0.000000 / 1.000000 0.000000 0.000000
DEV:0.000000: -1.000000 0.000000 0.000000 / -1.000000 0.000000 0.000000
DEV:0.000000: -0.888074 0.459701 0.000000 / -0.888074 0.459701 0.000000
gen_LebedevGrid done:0x600039f60 lmax=4 N=38,38,38gen_LebedevGrid done#devMx:4:1.554312e-15 5.932754e-16
ldXXXX donegen_LebedevGrid done:0x60004bfc0 lmax=5 N=50,50,50gen_LebedevGrid done#devMx:5:1.110223e-15 1.165734e-15
ldXXXX donegen_LebedevGrid done:0x60004f190 lmax=6 N=74,74,74gen_LebedevGrid done#devMx:6:1.998401e-15 1.106754e-15
ldXXXX doneDEV:0.000000: 0.369603 0.369603 -0.852518 / 0.369603 0.369603 -0.852518
gen_LebedevGrid done:0x600054320 lmax=7 N=86,86,86gen_LebedevGrid done#devMx:7:1.998401e-15 2.445960e-15
ldXXXX doneDEV:0.000000: 0.185116 0.185116 -0.965124 / 0.185116 0.185116 -0.965124
gen_LebedevGrid done:0x60005be30 lmax=8 N=110,110,110gen_LebedevGrid done#devMx:8:2.331468e-15 1.703498e-15
ldXXXX donegen_LebedevGrid done:0x600067470 lmax=9 N=146,146,146gen_LebedevGrid done#devMx:9:3.108624e-15 1.292369e-15
ldXXXX donegen_LebedevGrid done:0x600077ae0 lmax=10 N=170,170,170gen_LebedevGrid done#devMx:10:1.159783e-11 1.098080e-15
ldXXXX donegen_LebedevGrid done:0x60008e630 lmax=11 N=194,194,194gen_LebedevGrid done#devMx:11:1.473377e-11 1.400789e-15
ldXXXX donegen_LebedevGrid done:0x6000ad160 lmax=12 N=230,230,230gen_LebedevGrid done#devMx:12:1.801959e-11 1.039707e-14
ldXXXX donegen_LebedevGrid done:0x6000d5ea0 lmax=13 N=266,266,266gen_LebedevGrid done#devMx:13:2.138711e-11 4.100127e-15
ldXXXX doneDEV:0.000000: 0.096183 0.096183 0.990706 / 0.096183 0.096183 0.990706
DEV:0.000000: 0.096183 0.096183 -0.990706 / 0.096183 0.096183 -0.990706
gen_LebedevGrid done:0x60010b2b0 lmax=14 N=302,302,302gen_LebedevGrid done#devMx:14:2.480482e-11 1.553445e-15
ldXXXX donegen_LebedevGrid done:0x60010f3f0 lmax=15 N=350,350,350gen_LebedevGrid done#devMx:15:2.823164e-11 4.359794e-15
ldXXXX donegen_LebedevGrid done:0x600113f30 lmax=16 N=434,434,434gen_LebedevGrid done#devMx:16:3.162814e-11 2.719816e-15
ldXXXX donegen_LebedevGrid done:0x600119b50 lmax=17 N=590,590,590gen_LebedevGrid done#devMx:17:3.497891e-11 2.722540e-15
ldXXXX donegen_LebedevGrid done:0x6001215d0 lmax=18 N=770,770,770gen_LebedevGrid done#devMx:18:3.827128e-11 1.598006e-15
ldXXXX donegen_LebedevGrid done:0x60012b350 lmax=19 N=974,974,974gen_LebedevGrid done#devMx:19:4.149225e-11 1.904177e-15
ldXXXX donegen_LebedevGrid done:0x600137850 lmax=20 N=1202,1202,1202gen_LebedevGrid done#devMx:20:4.465173e-11 5.415048e-15
ldXXXX donegen_LebedevGrid done:0x600146970 lmax=21 N=1454,1454,1454gen_LebedevGrid done#devMx:21:4.769396e-11 2.924635e-15
ldXXXX doneDEV:0.000000: 0.028609 0.028609 0.999181 / 0.028609 0.028609 0.999181
gen_LebedevGrid done:0x600158b30 lmax=22 N=1730,1730,1730gen_LebedevGrid done#devMx:22:5.068579e-11 2.685677e-15
ldXXXX donegen_LebedevGrid done:0x60016e230 lmax=23 N=2030,2030,2030gen_LebedevGrid done#devMx:23:5.355638e-11 2.752572e-15
ldXXXX donegen_LebedevGrid done:0x6001872f0 lmax=24 N=2354,2354,2354gen_LebedevGrid done#devMx:24:5.635303e-11 3.331306e-15
ldXXXX donegen_LebedevGrid done:0x6001a4210 lmax=25 N=2702,2702,2702gen_LebedevGrid done#devMx:25:5.905143e-11 4.468431e-15
ldXXXX doneDEV:0.000000: 0.018861 0.018861 0.999644 / 0.018861 0.018861 0.999644
gen_LebedevGrid done:0x6001c5410 lmax=26 N=3074,3074,3074gen_LebedevGrid done#devMx:26:6.169043e-11 4.222722e-15
ldXXXX donegen_LebedevGrid done:0x6001ead90 lmax=27 N=3470,3470,3470gen_LebedevGrid done#devMx:27:6.422685e-11 4.174395e-15
ldXXXX donegen_LebedevGrid done:0x600215310 lmax=28 N=3890,3890,3890gen_LebedevGrid done#devMx:28:6.669365e-11 3.789802e-15
ldXXXX donegen_LebedevGrid done:0x600244930 lmax=29 N=4334,4334,4334gen_LebedevGrid done#devMx:29:6.902567e-11 4.344739e-15


*/
/*
int main(int argn,char **args){
int n,k,Nmx=40;double dum,pre=1.0;
for(n=0;n<Nmx;n++){ dum=factorial(n); printf("%d %f %f %lf\n",dum,pre*n,fabs(dum-n*pre));pre=dum;}

for(n=0;n<Nmx;n++){ for(k=0;k<=n;k++) printf("%f ",r_nCk(n,k));printf("\n");}
for(n=0;n<Nmx;n++){ for(k=0;k<=n;k++) printf("%ld ",l_nCk(n,k));printf("\n");}
}
*/
/*
  int n,Nmx=10;
  double buf[30];
  double z,z0=0.0,dz=0.001;int iz,Nz=100;
  z=z0;for(iz=0;iz<Nz;iz++,z+=dz){
     for(n=0;n<Nmx;n++){ buf[3*n]=BoysFn(10+n,z);   buf[3*n+1]=(exp(-z)+2*z*BoysFn(10+n+1,z))/(2*n+21);
                         buf[3*n+2]=fabs( buf[3*n+1]-buf[3*n]);}
     printf("%f ",z);for(n=0;n<3*Nmx;n++)printf("%19.10e ",buf[n]);printf("\n");
  }
  dz=0.10;
  for(iz=0;iz<Nz;iz++,z+=dz){
     for(n=0;n<Nmx;n++){ buf[3*n]=BoysFn(10+n,z);   buf[3*n+1]=(exp(-z)+2*z*BoysFn(10+n+1,z))/(2*n+21);
                         buf[3*n+2]=fabs( buf[3*n+1]-buf[3*n]);}
     printf("%f ",z);for(n=0;n<3*Nmx;n++)printf("%19.10e ",buf[n]);printf("\n");
  }
  dz=2.00;Nz=50;
  for(iz=0;iz<Nz;iz++,z+=dz){
     for(n=0;n<Nmx;n++){ buf[3*n]=BoysFn(10+n,z);   buf[3*n+1]=(exp(-z)+2*z*BoysFn(10+n+1,z))/(2*n+21);
                         buf[3*n+2]=fabs( buf[3*n+1]-buf[3*n]);}
     printf("%f ",z);for(n=0;n<3*Nmx;n++)printf("%19.10e ",buf[n]);printf("\n");
  }
  dz=10.00;Nz=10;
  for(iz=0;iz<Nz;iz++,z+=dz){
     for(n=0;n<Nmx;n++){ buf[3*n]=BoysFn(10+n,z);   buf[3*n+1]=(exp(-z)+2*z*BoysFn(10+n+1,z))/(2*n+21);
                         buf[3*n+2]=fabs( buf[3*n+1]-buf[3*n]);}
     printf("%f ",z);for(n=0;n<3*Nmx;n++)printf("%19.10e ",buf[n]);printf("\n");
  }
  dz=500.00;Nz=10;
  for(iz=0;iz<Nz;iz++,z+=dz){
     for(n=0;n<Nmx;n++){ buf[3*n]=BoysFn(10+n,z);   buf[3*n+1]=(exp(-z)+2*z*BoysFn(10+n+1,z))/(2*n+21);
                         buf[3*n+2]=fabs( buf[3*n+1]-buf[3*n]);}
     printf("%f ",z);for(n=0;n<3*Nmx;n++)printf("%19.10e ",buf[n]);printf("\n");
  }
}
*/

/*
int main(int argn,char **args){
  int n,k,dev;
  int nCk[30][30];
  for(n=1;n<30;n++){
     for(k=0;k<=n;k++) nCk[n][k]=floor(0.50+fn_nCk(n,k));
     for(k=0;k<=n;k++) printf("%d ",nCk[n][k]);
     printf("\n");
  }
  for(n=2;n<30;n++){
     for(k=1;k<=n-1;k++) {
        dev=(nCk[n-1][k-1]+nCk[n-1][k])-nCk[n][k];
        if( dev>0 ) printf("Error %d: (%d)%ld %ld   %d (%d)%ld\n",
                            n-1,k-1,nCk[n-1][k-1],nCk[n-1][k],n,k,nCk[n][k]);
     }
  }
  double p=1.0,f;
  for(n=0;n<30;n++){
     f=factorial(n);
     printf("%d %f %f\n",n,f,f/p); p=f;
  }
}*/
double gamma_hfodd(int odd){
	#define bfsz 30
	#define sqrtPI 1.7724538509055160272981674833411
	const double hf=0.5, qd3=0.75, oc15=1.875, qoc105=3.28125,ooc945=14.765625, dmy=-1.0, one=1.0, quad=0.25,
    sqrt2=1.4142135623730950488016887242097;
  // [i]: Gamma((2*i+1)/2)  so [i+1]=[i]*(i+0.5)
	static double gambuf[bfsz]={
		sqrtPI, sqrtPI*0.5, sqrtPI*0.75, sqrtPI*1.875, sqrtPI*6.5625,
		sqrtPI*29.53125,-1,       -1,        -1,         -1,
   -1,-1,-1,-1,-1,  -1,-1,-1,-1,-1,  -1,-1,-1,-1,-1,  -1,-1,-1,-1,-1};                                       
	static int nbuf=6;
 	int ia=(odd-1)/2;//$1->0 3->1 etc.
 	if(ia<nbuf) return gambuf[ia];
 	int j=nbuf-1; double dum=gambuf[nbuf-1]; // j<ia. we next multiply (2*(nbuf-1)+1)/2=j+0.5
 	for(;j<ia;j++) { dum*=(j+0.5);j++;if(j<bfsz){ gambuf[j]=dum;nbuf=j+1;}}
 	// odd=(2*nbuf+1)/2 and ia==nbuf : j=(nbuf-1) dum=gambuf[nbuf-1]*(nbuf-0.5)
 	return dum;
 	#undef bfsz
 	#undef sqrtPI
}