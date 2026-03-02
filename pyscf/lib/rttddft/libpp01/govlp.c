#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dalloc.h"
#include "dcmplx.h"
#include "macro_util.h"
#include "xgaussope.h"
#include "Gaussian.h"
#include "govlp.h"

void test_nrmz2(){
	const int N=2,Ipow=1;
	double alph[2]={1.192, 0.0645},A[3]={0,0,0},cofs[2]={0.7, 0.5};
	double beta[2]={1.868, 0.375 },B[3]={0.645,1.192,0.794};
	Gaussian *sA=Spherical_to_Cartesian( new_SphericalGaussianX(alph,A,0,0,N,cofs,0,Ipow) );
	Gaussian *xB=Spherical_to_Cartesian( new_SphericalGaussianX(beta,B,0,0,N,cofs,0,Ipow) );
	double VecField[3]={0.794,-1.192,0.8848},displ[3]={0,0,0};
	dcmplx v0=Gaussian_overlap(sA,xB,VecField,displ);
	dcmplx v1=Gaussian_ovlpNew(sA,xB,VecField,displ);
	printf("### %f+j%f / %f+j%f\n", creal(v1),cimag(v1),creal(v0),cimag(v0) );
}
void test_nrmz1(){
	printf("#test_nrmz1");fflush(stdout);//__assertf(0,(""),-1);
	double alph[3]={0.7122640246E+00, 0.2628702203E+00, 0.1160862609E+00};
	double cofs_s[3]={-0.3088441214E+00, 0.1960641165E-01, 0.1131034442E+01};
	double cofs_p[3]={-0.1215468600E+00,  0.5715227604E+00, 0.5498949471E+00};
	double *cofs=cofs_p;
	double R[3]={0,0,0};int N=3,Ipow=0;
	const double PI=3.1415926535897932384626433832795;
	int i,j;double cum=0;
	for(i=0;i<N;i++)for(j=0;j<N;j++){
		double nmz_i= sqrt( 0.5/(alph[i]+alph[i])*pow( sqrt( PI/(alph[i]+alph[i]) ), 3) );
		double nmz_j= sqrt( 0.5/(alph[j]+alph[j])*pow( sqrt( PI/(alph[j]+alph[j]) ), 3) );
		double ovlp=0.5/(alph[i]+alph[j])*pow( sqrt( PI/(alph[i]+alph[j]) ), 3);
		printf("%d %d %f %f\n",i,j,ovlp,ovlp/(nmz_i*nmz_j));
		cum+=ovlp*(cofs[i]/nmz_i)*(cofs[j]/nmz_j);
	}
	printf("analytic OVLP:%f\n",cum);
	int el=1;int em;
	for(em=-el;em<=el;em++){
		SphericalGaussian *spg=new_SphericalGaussianX(alph,R,el,em,N,cofs,0,Ipow);
		Gaussian *G=Spherical_to_Cartesian(spg);
		double dmy[3]={0,0,0};
		dcmplx ovlp=Gaussian_overlap(G,G,dmy,dmy);
		printf("\n#OVLP:%d.%d.%d %f+j%f\n",Ipow,el,em,creal(ovlp),cimag(ovlp));fflush(stdout);
	}
}

Gaussian *get_cGTO(double *A,int el,int em,int Ipow){
	double alcof_s[14*7]={
		1.642000E+05, 2.600000E-05, -6.000000E-06, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		2.459000E+04, 2.050000E-04, -4.600000E-05, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		5.592000E+03, 1.076000E-03, -2.440000E-04, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		1.582000E+03, 4.522000E-03, -1.031000E-03, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		5.161000E+02, 1.610800E-02, -3.688000E-03, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		1.872000E+02, 4.908500E-02, -1.151400E-02, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		7.393000E+01, 1.248570E-01, -3.043500E-02, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		3.122000E+01, 2.516860E-01, -6.814700E-02, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		1.381000E+01, 3.624200E-01, -1.203680E-01, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		6.256000E+00, 2.790510E-01, -1.482600E-01, 0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		2.776000E+00, 6.355200E-02, 9.905000E-03, 1.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 
		1.138000E+00, 1.063000E-03, 3.842860E-01, 0.000000E+00, 1.000000E+00, 0.000000E+00, 0.000000E+00, 
		4.600000E-01, 1.144000E-03, 5.368050E-01, 0.000000E+00, 0.000000E+00, 1.000000E+00, 0.000000E+00, 
		1.829000E-01, -4.000000E-05, 2.026870E-01, 0.000000E+00, 0.000000E+00, 0.000000E+00, 1.000000E+00};
	double alph[14],cofs[14];int N=14;
	double B[3]={0.645, 0.1192, -0.794};
	int i;
	for(i=0;i<14*7;i++){
		if(i%7==0) alph[i/7]=alcof_s[i];
		if(i%7==1 && el==0 ) cofs[i/7]=alcof_s[i];
		if(i%7==2 && el==1 ) cofs[i/7]=alcof_s[i];
		if(i%7==3 && el==2 ) cofs[i/7]=alcof_s[i];
	}
	//new_SphericalGaussian(double *alpha,double *R,int l,int m,int N,double *cof,int Jnuc)
	return Spherical_to_Cartesian(  new_SphericalGaussianX(alph,A,el,em,N,cofs,0,Ipow));
}

/*
int main(int argn,char **args){
	//test_nrmz1();__assertf(0,(""),-1);
	test_nrmz2(); __assertf(0,(""),-1);
	int Ipow,el,em;
	double A[3]={0.645, 1.192, -0.794};
	double B[3]={ 0.710,1.868, -0.894};
	double alph[3]={15.82, 1.603, 0.645};
	double cofs[3]={ 0.2,  0.7,   0.3};
	double beta[4]={14.67, 1.592, 0.794, 0.01192};
	double cofs4[4]={ 0.710,0.3776,1.192, 0.1338};
	double VecField[3]={0.794,-1.192,0.8848},displ[3]={0,0,0};
	const double zerovec[3]={0,0,0};
	//double VecField[3]={0,0,0},displ[3]={0,0,0};
	for(Ipow=0;Ipow<=2;Ipow++){
		Gaussian *sA=get_cGTO(A,0,0,Ipow),*xA=get_cGTO(A,1,1,Ipow),*yA=get_cGTO(A,1,-1,Ipow),*xxA=get_cGTO(A,2,2,Ipow),*zzA=get_cGTO(A,2,0,Ipow);
		Gaussian *sB=get_cGTO(B,0,0,Ipow),*yB=get_cGTO(B,1,-1,Ipow),*xyB=get_cGTO(B,2,-1,Ipow);
		Gaussian *BS[8]={ sA,xA,yA,xxA,zzA,sB,yB,xyB};
		if(1){
			dcmplx v0=Gaussian_overlap(sA,sA,VecField,displ);
			dcmplx v1=Gaussian_ovlpNew(sA,sA,VecField,displ);
			printf("### sA sA OVLP: %f+j%f / %f+j%f\n",creal(v1),cimag(v1), creal(v0),cimag(v0));
			//### sA sA OVLP: 1.000000+j0.000000 / 1.000000+j0.000000
		}
		int I,J;double t0,t1,t2,t_old,t_new;
		for(I=0;I<8;I++)for(J=0;J<8;J++){
			t0=__ctime;t1=t0;
			dcmplx v0=Gaussian_overlap(BS[I],BS[J],VecField,displ);
			dcmplx v0ZF=Gaussian_overlap(BS[I],BS[J],zerovec,zerovec);
			t2=t1;t1=__ctime;t_old=t1-t2;
			dcmplx v1=Gaussian_ovlpNew(BS[I],BS[J],VecField,displ);
			dcmplx v1ZF=Gaussian_ovlpNew(BS[I],BS[J],zerovec,zerovec);
			t2=t1;t1=__ctime;t_new=t1-t2;
			printf("###%d.%d.%d OVLP:%s * %s : %f+j%f (%f)/ %f+j%f (%f)  \t\t %f+j%f / %f+j%f\n",
				Ipow,I,J,BS[I]->description,BS[J]->description,creal(v1),cimag(v1),t_new,creal(v0),cimag(v0),t_old,
				creal(v1ZF),cimag(v1ZF), creal(v0ZF),cimag(v0ZF));
			double dev=ZabsSQR((v0-v1));dev=sqrt(dev);
			double devZF=ZabsSQR((v0ZF-v1ZF));devZF=sqrt(devZF);
			__assertf( (devZF<1.0e-6),("DEV=%e",devZF),-1);
		}
	}
	__assertf(0,("end"),-1);
	
	for(Ipow=0;Ipow<=2;Ipow++){
		for(el=0;el<=3;el++){
			for(em=-el;em<=el;em++){
				Gaussian *lhs=Spherical_to_Cartesian(  new_SphericalGaussianX(alph,A,el,em,3,cofs,0,Ipow));
					int Jpow,l,m;
					for(Jpow=0;Jpow<=2;Jpow++){
						for(l=0;l<=3;l++){
							double t0,t1,t2,t_old,t_new;
							for(m=-l;m<=l;m++){
								//Gaussian *rhs=Spherical_to_Cartesian(  new_SphericalGaussianX(beta,B,l,m,4,cofs,0,Jpow));
								Gaussian *rhs=Spherical_to_Cartesian(  new_SphericalGaussianX(beta,A,l,m,4,cofs,0,Jpow));
								t0=__ctime;t1=t0;
								dcmplx v0=Gaussian_overlap(lhs,rhs,VecField,displ);
								t2=t1;t1=__ctime;t_old=t1-t2;
								dcmplx v1=Gaussian_ovlpNew(lhs,rhs,VecField,displ);
								t2=t1;t1=__ctime;t_new=t1-t2;
								double dev=ZabsSQR((v0-v1));dev=sqrt(dev);
								printf("OVLP:%d %d %d * %d %d %d : %f+j%f (%f)/ %f+j%f (%f)\n",Ipow,el,em,Jpow,l,m,creal(v1),cimag(v1),t_new, creal(v0),cimag(v0),t_old);
								__assertf(dev<1.0e-6,("wrong OVLP %f / %f+j%f\n",v1,creal(v0),cimag(v0)),-1); 
							}
						}
	}}}}
	
}
*/
dcmplx Gaussian_ovlpNew_verbose(Gaussian *lhs,Gaussian *rhs,const double *VecField,const double *displ){
	int i,j;
	
	const int Nlhs=lhs->N,Nrhs=rhs->N;
	dcmplx zsum=0;
	for(i=0;i<Nlhs;i++)for(j=0;j<Nrhs;j++){
		double B[3]={ rhs->R[0]+displ[0], rhs->R[1]+displ[1], rhs->R[2]+displ[2]}; 
		dcmplx ovlp=pgovlp(lhs->pows[i],lhs->alpha[i],lhs->R, rhs->pows[j],rhs->alpha[j],B, VecField);
		double re=creal(ovlp),im=cimag(ovlp);
		if( __isNaNInf(re) || __isNaNInf(im) ){ printf("#pgovlp:NaNinf:%f %f\n",re,im);
		                                        pgovlp_(lhs->pows[i],lhs->alpha[i],lhs->R, rhs->pows[j],rhs->alpha[j],B, VecField,1);
		                                        __assertf(0,("NaNInf at Gaussian_ovlpNew_verbose"),-1);}
		zsum+=(lhs->cofs[i])*(rhs->cofs[j])*ovlp;
	}
	return zsum;
}
#define _Check_NaN(ztgt)   {if(__isNaN(ztgt)){ __assertf(0,("#check_NaN:%f %f ... isnan:%d,%d\n",creal(ztgt),cimag(ztgt),__isNaN(creal(ztgt)),__isNaN(cimag(ztgt))),1);\
__assertf(( !(__isNaN(creal(ztgt))) && !(__isNaN(cimag(ztgt)))),("_Check_NaN"),-1);} }  
// 2021.08.30 revision 
// 1:    < lhs(r-A) | rhs(r-(B+D)) e^{iA.r} > e^{iKD}
// 2: < lhs(r-(A-D))| rhs(r-B) e^{iA.r} > e^{i(K+A)D} 
/*
dcmplx Gaussian_ovlpNew_B(Gaussian *lhs,Gaussian *rhs,const double *VecField,const double *displ){
	int i,j;
	const int Nlhs=lhs->N,Nrhs=rhs->N;
	dcmplx zsum=0;
	for(i=0;i<Nlhs;i++)for(j=0;j<Nrhs;j++){
		double A[3]={ lhs->R[0]-displ[0], lhs->R[1]-displ[1], lhs->R[2]-displ[2]}; 
		dcmplx ovlp=pgovlp(lhs->pows[i],lhs->alpha[i],A, rhs->pows[j],rhs->alpha[j],rhs->R, VecField);
		zsum+=(lhs->cofs[i])*(rhs->cofs[j])*ovlp;
	}
	_Check_NaN(zsum);
	return zsum;
} */
dcmplx Gaussian_ovlpNew(Gaussian *lhs,Gaussian *rhs,const double *VecField,const double *displ){
	int i,j;
	
	const int Nlhs=lhs->N,Nrhs=rhs->N;
	dcmplx zsum=0;
	for(i=0;i<Nlhs;i++)for(j=0;j<Nrhs;j++){
		double B[3]={ rhs->R[0]+displ[0], rhs->R[1]+displ[1], rhs->R[2]+displ[2]}; 
		dcmplx ovlp=pgovlp(lhs->pows[i],lhs->alpha[i],lhs->R, rhs->pows[j],rhs->alpha[j],B, VecField);
		zsum+=(lhs->cofs[i])*(rhs->cofs[j])*ovlp;
	}
	_Check_NaN(zsum);
	return zsum;
}
dcmplx Gaussian_ovlpNewX(Gaussian *lhs,Gaussian *rhs,const int ixyz,const dcmplx overlap0,const double *VecField,const double *displ){
	int i,j;
	 
	const int Nlhs=lhs->N,Nrhs=rhs->N;
	dcmplx zsum=0;
	for(i=0;i<Nlhs;i++){
		ushort lhs_pows[3]={ lhs->pows[i][0], lhs->pows[i][1], lhs->pows[i][2] };
		lhs_pows[ixyz]+=1;
		for(j=0;j<Nrhs;j++){
			double B[3]={ rhs->R[0]+displ[0], rhs->R[1]+displ[1], rhs->R[2]+displ[2]}; 
			dcmplx ovlp=pgovlp( lhs_pows, lhs->alpha[i],lhs->R, rhs->pows[j],rhs->alpha[j],B, VecField);
			zsum+=(lhs->cofs[i])*(rhs->cofs[j])*ovlp;
		}
	}
	// we have calculated <lhs_0(r-A)|(x_i-A_i)|rhs_n(r-B-D)>
	zsum+= lhs->R[ixyz] * overlap0;
	_Check_NaN(zsum);
	return zsum;
}

#define _INTpow(a,n)  ((n)<2 ? ((n)==0 ? 1.0:(a)):((n)==2 ? ((a)*(a)):( (n)==3 ? (a)*(a)*(a):pow(a,n))))
//#define _INTpow_safe(a,n,wksp)  ((n)<2 ? ((n)==0 ? 1.0:(a)):((n)==2 ? ((a)*(a)):( (n)==3 ? (a)*(a)*(a):pow(a,n))))
dcmplx pgovlp(ushort *xpow_A,const double alph,const double *A,
              ushort *xpow_B,const double beta,const double *B,const double *Vecfield){
	return pgovlp_(xpow_A,alph,A,xpow_B,beta,B,Vecfield,0);
}
dcmplx pgovlp_(ushort *xpow_A,const double alph,const double *A,
              ushort *xpow_B,const double beta,const double *B,const double *Vecfield,const int verbose){
	double sqrDist=( (A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]) + (A[2]-B[2])*(A[2]-B[2]) );
	// a*b/(a+b) = a/(a/b+1.0)
	double gamma=( beta > alph ? ( alph/(alph/beta + 1.0)):(beta/((beta/alph)+1.0)) );
	
	double G[3]={ (alph*A[0]+beta*B[0])/(alph+beta), (alph*A[1]+beta*B[1])/(alph+beta), (alph*A[2]+beta*B[2])/(alph+beta) };
	double G_dot_A= G[0]*Vecfield[0] + G[1]*Vecfield[1] + G[2]*Vecfield[2];
	dcmplx zfac=( cos(G_dot_A)+_I*sin(G_dot_A) );
	
	const double Asqr=Vecfield[0]*Vecfield[0] + Vecfield[1]*Vecfield[1] + Vecfield[2]*Vecfield[2];
	const char zerofield=(Asqr<1.0e-20);
	// - 0.25*Asqr/(alph[0]+alph[1]));
	const double expofac=exp(-gamma*sqrDist - 0.25*Asqr/(alph+beta));
	char dbgng=1;
	int dir;
	dcmplx prod=1.0;
	if(verbose){ printf("#pgovlp: sqrDist:%f Asqr:%f gamma:%f G:%f,%f,%f zfac:%f %f\n",sqrDist,Asqr,gamma,G[0],G[1],G[2],creal(zfac),cimag(zfac));fflush(stdout);}
	for(dir=0;dir<3;dir++){
		dcmplx zsum=0;
		double Gd=(alph*A[dir]+beta*B[dir])/(alph+beta);
		double GA=Gd-A[dir], GB=Gd-B[dir];
		const int a=xpow_A[dir],b=xpow_B[dir];const int el=a+b;
		int j,k,m;
		if(fabs(GA)<1.0e-8){
			//printf("#Case GA==0..\n");fflush(stdout); // const double invsqrt_alph_plus_beta = sqrt(1/(alph+beta));
			for(m=a;m<=el;m++){ // l_nCk(a,j) * l_nCk(b,m-j) * pow( GA, a-j) * pow( GB, b-m+j ) 
				if(zerofield && m%2)continue;
				// printf("#dir=%d m=%d / a=%d b=%d\n",dir,m,a,b); 
				// j==a only.   here  MAX(0,m-b)<= j
				j=a;k=m-j;
				dcmplx zdum= l_nCk(b,(m-j)) * _INTpow( GB, b-k );   // pow(GA,a-j)==1
				// double argIpw=sqrt(1/(alph+beta));
				zdum*=gaussian_integ1D_vcfield_((alph+beta),m,Vecfield[dir]); //gamma_hfint(m+1)*_INTpow( argIpw, m+1 );
				if(verbose){ dcmplx cdmy=gaussian_integ1D_vcfield_verbose_((alph+beta),m,Vecfield[dir]);
					           printf("#pgovlp:m=%d:a=%d,b=%d, %d_C_%d:%ld  cdmy:%f %f\n",m,a,b, b,m-j,l_nCk(b,m-j), creal(cdmy),cimag(cdmy) );fflush(stdout);}
				zsum+=zdum;// printf("j=%d k=%d %f %f %f\n",j,k,dum,sum,l_nCk(b,(m-j)) * pow( GB, b-k )*pow(GA,(a-j))*pow(GB,b-k) );
			}
		} else {
			double GBoverGA=GB/GA;
			if(verbose){ printf("#pgovlp: GBoverGA:%f %f/%f\n",GBoverGA,GB,GA);fflush(stdout);}
			// expand (y+GA)**a ...
			const int m_skip=(zerofield ? 2:1); 
			for(m=0;m<=el;m+=m_skip){    // collecting term with y**m   j from a, m-j from b.. 
				const int jmin=__MAX( (m-b), 0);
				const int jmax=__MIN( a, m);
				double cum=0;
				double fac=pow(GA, a-jmin)*pow(GB, b-m+jmin);
				for(j=jmin;j<=jmax;j++){
					double dum=l_nCk(a,j) * l_nCk(b,m-j) * fac;
					cum+= dum;
					if(verbose){ printf("#pgovlp:%d: %d C %d:%ld  %d C %d:%ld  fac:%f\n",m, a,j,l_nCk(a,j), b,m-j,l_nCk(b,m-j), fac);fflush(stdout);}
					/* if(dbgng){
						double ref=l_nCk(a,j) * l_nCk(b,m-j) * pow(GA,(a-j)) * pow(GB,(b-m+j));
						double dev1=fabs(ref-dum);double scle=0.50*( fabs(ref)+fabs(dum) );
						if(scle>1.0) dev1=dev1/scle;
						// this fails if ref or dum is of size ~ 1e+10 __assertf( fabs(ref-dum)<1.0e-6,("#pgovlp: dum=%f / ref=%f  %d %d %d %d %f %f  fac=%f",dum,ref, a,j,b,m-j, pow(GA,(a-j)), pow(GB,(b-m+j)), fac),-1);
						__assertf( dev1<1.0e-6,("#pgovlp: dum=%f / ref=%f : dev=%e (scle=%e) %d %d %d %d %f %f  fac=%f",dum,ref,dev1,scle, a,j,b,m-j, pow(GA,(a-j)), pow(GB,(b-m+j)), fac),-1);
					} */
					fac*=GBoverGA; // for next step
				}
				zsum+= cum* gaussian_integ1D_vcfield_((alph+beta),m,Vecfield[dir]);  //gamma_hfint(m+1)*_INTpow( sqrt(1/(alph+beta)),m+1 );
				if(verbose){ printf("#pgovlp:%d: %f %f\n",m,creal(gaussian_integ1D_vcfield_((alph+beta),m,Vecfield[dir])),
				                                            cimag(gaussian_integ1D_vcfield_((alph+beta),m,Vecfield[dir])));fflush(stdout);}
				
			}
		}
		prod*=zsum;
	}
	dcmplx retv=prod*expofac*zfac;
	
/*	double absG=sqrt(G[0]*G[0] + G[1]*G[1] + G[2]*G[2]),absA=sqrt(Asqr);
	if( absG>1.0e-6 && absA>1.0e-6 && fabs(prod*expofac)>1.0e-7){ 
		printf("#pgovlp:|G|=%9.4f |A|=%9.4f GdotA=%9.4f AB=%f zf=%f,%f  ovlp=%f ret=%f,%f\n",
						 absG, absA, G_dot_A,sqrt(sqrDist),creal(zfac),cimag(zfac),prod*expofac,creal(retv),cimag(retv));} */
	//printf("#zfac:%f %f*j%f\n",G_dot_A,creal(zfac),cimag(zfac));
	return retv;
	//return prod*expofac*zfac;
}


void test_ovlpx(Gaussian **BS1,int nBS1, Gaussian **BS2, int nBS2, const double *BravaisVectors){
	const double *ax=BravaisVectors, *ay=BravaisVectors+3, *az=BravaisVectors+6;
	int ilh,jrh; double h,h0=0.05;int jh,nh=5;dcmplx vals[5];int Idspl;int Ixyz; double wc0,wc1;
	double Aref[3]={0.645, -0.3776, 0.1192};
	dcmplx refv[4];const char revision[3]=".1";//.1: we fixed Gaussian_ovlpNew and Gaussian_ovlpNewX ..
	char fpath[40];sprintf(fpath,"test_ovlpx%s.log",revision);int seqno=0;
	for(ilh=0,seqno=0;ilh<nBS1;ilh++){ for(jrh=0;jrh<nBS2;jrh++,seqno++){
		for(Idspl=0;Idspl<4;Idspl++){ const double tol=5.0e-6;
			const double zero[3]={0,0,0};
			const double *displ=(Idspl == 0 ? zero: BravaisVectors+3*(Idspl-1));
			for(Ixyz=0;Ixyz<4;Ixyz++){ refv[Ixyz]=(Ixyz==0 ? Gaussian_ovlpNew(BS1[ilh],BS2[jrh],Aref,displ):
			                                                 Gaussian_ovlpNewX(BS1[ilh],BS2[jrh],Ixyz-1,refv[0],Aref,displ));
			  const int dbgng=(seqno%10==0 && (Ixyz==0 || Ixyz==1) );
				dcmplx third=0,second=0,best=0;double scale=3.0; double walltime_00,ermax_00;dcmplx edgesum_00; double walltime_01,ermax_01;dcmplx edgesum_01;
				double walltime_10,ermax_10;dcmplx edgesum_10;  double walltime_11,ermax_11;dcmplx edgesum_11;
				scale=3.0;h=h0; dcmplx v00=nmrinteg_ovlp1(h,BS1[ilh],BS2[jrh],scale,Ixyz,Aref,displ,&walltime_00, &ermax_00, &edgesum_00);
				scale=4.0;h=h0; dcmplx v10=nmrinteg_ovlp1(h,BS1[ilh],BS2[jrh],scale,Ixyz,Aref,displ,&walltime_10, &ermax_10, &edgesum_10);
				scale=3.0;h=h0*0.7071; dcmplx v01=nmrinteg_ovlp1(h,BS1[ilh],BS2[jrh],scale,Ixyz,Aref,displ,&walltime_01, &ermax_01, &edgesum_01);
				dcmplx v11=0;
				// convergence wrt h
				const char cvgd_wrt_h=( ( cabs(v00-v01)<tol ) ? 1:0), cvgd_wrt_s=( ( cabs(v00-v10)<tol ) ? 1:0);
				if(dbgng || ((!cvgd_wrt_h)&&(!cvgd_wrt_s))){
					scale=4.0;h=h0*0.7071; v11=nmrinteg_ovlp1(h,BS1[ilh],BS2[jrh],scale,Ixyz,Aref,displ,&walltime_11, &ermax_11, &edgesum_11);
					best=v11;second=( cabs(v11-v10)<cabs(v11-v01) ? v10:v01 );third=( cabs(v11-v10)<cabs(v11-v01) ? v01:v10 );
				} else {
					if(cvgd_wrt_h){ best=v10;second=v01;third=v00;}
					else          { best=v01;second=v10;third=v00;}
				}
				if(dbgng){
					FILE *fp=fopen(fpath,"a");
					fprintf(fp,"%04d %04d %02d %d  00: %14.6f +j%14.6f  %e  %f %e %f+j%f\n",ilh,jrh,Idspl,Ixyz, 
					            creal(v00),cimag(v00),cabs(v00-v11),walltime_00,ermax_00,creal(edgesum_00),cimag(edgesum_00));
					fprintf(fp,"%04d %04d %02d %d  01: %14.6f +j%14.6f  %e  %f %e %f+j%f\n",ilh,jrh,Idspl,Ixyz, 
					            creal(v01),cimag(v01),cabs(v01-v11),walltime_01,ermax_01,creal(edgesum_01),cimag(edgesum_01));
					fprintf(fp,"%04d %04d %02d %d  10: %14.6f +j%14.6f  %e  %f %e %f+j%f\n",ilh,jrh,Idspl,Ixyz, 
					            creal(v10),cimag(v10),cabs(v10-v11),walltime_10,ermax_10,creal(edgesum_10),cimag(edgesum_10));
					fprintf(fp,"%04d %04d %02d %d  11: %14.6f +j%14.6f  %e  %f %e %f+j%f\n",ilh,jrh,Idspl,Ixyz, 
					            creal(v11),cimag(v11),cabs(v11-v11),walltime_11,ermax_11,creal(edgesum_11),cimag(edgesum_11));fclose(fp);
				}
				FILE *fp=fopen(fpath,"a");fprintf(fp,"%04d %04d %02d %d  %14.6f +j%14.6f      %14.6f +j%14.6f %e       %14.6f +j%14.6f\n", 
																	ilh,jrh,Idspl,Ixyz,creal(refv[Ixyz]),cimag(refv[Ixyz]),creal(best),cimag(best),cabs(best-refv[Ixyz]),creal(second),cimag(second));
				fclose(fp);
			}
		}
	}}
}
int irand(const int uplm){
	int idum=rand();
	return idum%uplm;
}
double drand(void){
	return ((double)rand())/(double)RAND_MAX;
}
double *d1rand(const int Ld,const double scale){
	double *ret=d1alloc(Ld);int i;
	for(i=0;i<Ld;i++) ret[i]=scale*(1.0-2.0*rand());
	return ret;
}
Gaussian *gen_randomGaussian(int *pEllEmm,char **description){
	int ell=pEllEmm[0],emm=pEllEmm[1];
	if(ell<0){ pEllEmm[0]=( ell=irand(4) );pEllEmm[1]=( emm=irand(2*ell+1)-ell );
	           printf("#gen_randomGaussian:ell=%d,emm=%d\n",ell,emm);}
	int npgto=irand(9);
	double *alph=d1alloc(npgto),*cofs=d1alloc(npgto),*R=d1alloc(3);double *A=d1alloc(3);
	int i;for(i=0;i<npgto;i++){ alph[i]=drand();cofs[i]=(1.0-2*drand());}
	for(i=0;i<3;i++){ R[i]=(1.0-2*drand())*2.0;}
	SphericalGaussian *spg=new_SphericalGaussian(alph, R,ell,emm,npgto,cofs,0);
	Gaussian *ret=Spherical_to_Cartesian(spg);free_SphericalGaussian(spg);spg=NULL;
	description[0]=ch1alloc(200);
	sprintf( description[0],"l=%d,m=%d R=%f,%f,%f npgto=%d al,cof=%f,%f ,..",ell,emm,R[0],R[1],R[2],npgto,alph[0],cofs[0]);
	return ret;
}
#define _D3dot(A,B)  (A[0]*B[0] + A[1]*B[1] + A[2]*B[2])

#define _Check_cvg(Ga_lhs,Ga_rhs,h_0,scale_0,I_xyz) {\
double walltime_10,edge_max_10;dcmplx edge_sum_10;\
dcmplx zovlp_10=nmrinteg_ovlp1((h_0),Ga_lhs,Ga_rhs,((scale_0)+1.0),I_xyz,VecField,displ,&walltime_10,&edge_max_10,&edge_sum_10);\
double walltime_01,edge_max_01;dcmplx edge_sum_01;\
dcmplx zovlp_01=nmrinteg_ovlp1((h_0)*0.7071,Ga_lhs,Ga_rhs,(scale_0),I_xyz,VecField,displ,&walltime_01,&edge_max_01,&edge_sum_01);\
double walltime_11,edge_max_11;dcmplx edge_sum_11;\
dcmplx zovlp_11=nmrinteg_ovlp1((h_0)*0.7071,Ga_lhs,Ga_rhs,((scale_0)+1.0),I_xyz,VecField,displ,&walltime_11,&edge_max_11,&edge_sum_11);\
printf("#Check_cvg: 10: %14.6f %14.6f  %e  %f  edge:%f  %f\n", _RandI(zovlp_10),cabs(zovlp_10-zovlp_11),walltime_10,edge_max_10,cabs(edge_sum_10));\
printf("#Check_cvg: 01: %14.6f %14.6f  %e  %f  edge:%f  %f\n", _RandI(zovlp_01),cabs(zovlp_01-zovlp_11),walltime_01,edge_max_01,cabs(edge_sum_01));\
printf("#Check_cvg: 11: %14.6f %14.6f  %e  %f  edge:%f  %f\n", _RandI(zovlp_11),cabs(zovlp_11-zovlp_11),walltime_11,edge_max_11,cabs(edge_sum_11));}//


void test_intg002(){
	int itest,ntest=200;char *description[2]={NULL,NULL};double VecField[3]={0,0,0};double displ[3]={0,0,0};int j;
	double walltime,edge_max;dcmplx edge_sum;
	int check_cvg=1; 
	int ell,emm;
	for(ell=2,itest=0;ell<8;ell++) for(emm=-ell;emm<=ell;emm++,itest++){
		int ellemm[2]={ell,emm};int ellemmdmy[2]={-1,-1};
		Gaussian *lhs=gen_randomGaussian(ellemm, description+0);
		Gaussian *rhs=gen_randomGaussian(ellemm, description+1);

		for(j=0;j<3;j++) VecField[j]=(1.0-2.0*drand());
		for(j=0;j<3;j++) displ[j]=(1.0-2.0*drand())*0.50;
		dcmplx zovlp0=Gaussian_ovlpNew(lhs,rhs,VecField,displ);printf("#zovlp0:%f %f\n",__RandI(zovlp0));fflush(stdout);
		dcmplx nmrintg=nmrinteg_ovlp1(0.05,lhs,rhs,3.50,0,VecField,displ,&walltime,&edge_max,&edge_sum);
		printf(" %04d: %14.6f %14.6f     %14.6f %14.6f    %e    NmrERR:%e %e     %s %s\n", 
					itest, __RandI(zovlp0), __RandI(nmrintg), cabs(zovlp0-nmrintg), edge_max, cabs(edge_sum), description[0],description[1]);
	//	if(ell){ _Check_cvg(lhs,rhs,0.05,3.50,0);}
		int dir;
		for(dir=0;dir<3;dir++){
			dcmplx zovlp=Gaussian_ovlpNewX(lhs,rhs,dir,zovlp0,VecField,displ);
			nmrintg=nmrinteg_ovlp1(0.05,lhs,rhs,3.50, dir+1, VecField, displ, &walltime,&edge_max,&edge_sum);
			printf(" %04d.%d: %14.6f %14.6f     %14.6f %14.6f    %e   %s %s\n", 
					itest,dir+1, __RandI(zovlp), __RandI(nmrintg), cabs(zovlp-nmrintg), description[0],description[1]);
			//if(ell){_Check_cvg(lhs,rhs,0.05,3.50,(dir+1));}
		}
		free(description[0]);free(description[1]);
		free_Gaussian(lhs);
		free_Gaussian(rhs);
	}
}
void test_intg001(){
	double alph=0.1192, beta=0.645;
	double A[3]={0.645, -0.794, 1.192},B[3]={0.894,-0.5, 1.603};
	double VecField[3]={ 0.8848, 1.467, -0.666};double zerovec[3]={0,0,0};
	ushort pwA[3]={0,0,0},pwB[3]={0,0,0};double cof1=1.00;
	Gaussian *lhs=new_Gaussian(&alph,A, pwA,1, &cof1,0);
	Gaussian *rhs=new_Gaussian(&beta, B, pwB,1, &cof1,0);
	dcmplx zovlp=Gaussian_ovlpNew(lhs,rhs,VecField,zerovec);
	double Asqr=_D3dot(VecField, VecField); double gamma=alph*beta/(alph + beta);
	double ABsqr= (A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]) + (A[2]-B[2])*(A[2]-B[2]);
	double C[3]={ (alph*A[0]+beta*B[0])/(alph+beta), (alph*A[1]+beta*B[1])/(alph+beta), (alph*A[2]+beta*B[2])/(alph+beta) };
	double VdotC=_D3dot(C, VecField);
	const double PI=3.1415926535897932384626433832795;
	dcmplx zfac=cos(VdotC) + _I*sin(VdotC);
	dcmplx refv=exp(-0.25*Asqr/(alph+beta) - gamma*ABsqr)*zfac*pow( sqrt(PI/(alph+beta)),3);
	double walltime,edge_max;dcmplx edge_sum;
	dcmplx nmrintg=nmrinteg_ovlp1(0.05,lhs,rhs,4.0,0,VecField,zerovec,&walltime,&edge_max,&edge_sum);
	printf("#test_intg001 : %f+j%f  %f+j%f  %f+j%f\n", __RandI(zovlp), __RandI(refv),__RandI(nmrintg) );
}
/*
int main(int narg,char **args){
	test_intg002();
	//test_intg001();
}*/
// scale : scale/sqrt(min-alph) is the integration range ...
dcmplx nmrinteg_ovlp1(double h,Gaussian *lhs,Gaussian *rhs,const double scale,const int Ixyz_or_0,const double *VecField,const double *displ,
                      double *pWalltime, double *pEdge_max, dcmplx *pEdge_sum){
	double wc0=__ctime;
	const double *A=lhs->R;double Rlb[3],Rrt[3];
	double B[3]={ rhs->R[0]+displ[0],rhs->R[1]+displ[1], rhs->R[2]+displ[2]};
	
	int i; double bmin,amin;
	bmin=rhs->alpha[0];for(i=1;i<rhs->N;i++) bmin=__MIN(rhs->alpha[i],bmin);
	amin=lhs->alpha[0];for(i=1;i<lhs->N;i++) amin=__MIN(lhs->alpha[i],amin);
	int kk;for(kk=0;kk<3;kk++){ Rlb[kk]=__MIN( B[kk]-scale/sqrt(bmin), A[kk]-scale/sqrt(amin));
	                            Rrt[kk]=__MAX( B[kk]+scale/sqrt(bmin), A[kk]+scale/sqrt(amin)); }
	double dvol=h*h*h;
	int N[3];for(kk=0;kk<3;kk++){ N[kk]=(int) ceil( (Rrt[kk]-Rlb[kk])/h );}
	double R[3];double re=0.0,im=0.0;double AxR[3]={0,0,0};int I,J,K;int on_edge=0; double edge_max=0.0;
	double edgesum_re=0,edgesum_im=0;
	printf("#nmrinteg gridsize:%d,%d,%d\n",N[0],N[1],N[2]);
	if(Ixyz_or_0==0){
		for(I=0;I<N[0];I++){ R[0]=Rlb[0]+h*I; AxR[0]=R[0]*VecField[0]; on_edge=(I==0||I==N[0]-1);
			for(J=0;J<N[1];J++){ R[1]=Rlb[1]+h*J; AxR[1]=R[1]*VecField[1]; on_edge=( on_edge || (J==0||I==N[1]-1) );
				for(K=0;K<N[2];K++){ R[2]=Rlb[2]+h*K; AxR[2]=R[2]*VecField[2]; on_edge=( on_edge || (K==0||K==N[2]-1) );
					double vlh=Gaussian_value(lhs, R);
					double Rrhs[3]={ R[0]-displ[0], R[1]-displ[1], R[2]-displ[2] }; // R-(rhs->B+displ)
					double vrh=Gaussian_value(rhs, Rrhs); double arg=AxR[0]+AxR[1]+AxR[2];
					double prd=vlh*vrh;  double xi=prd*cos(arg),eta=prd*sin(arg);
					re+= xi; im+= eta;
					if(on_edge){ edge_max=__MAX(edge_max,prd); edgesum_re+=xi; edgesum_im+=eta;}
		}}}
	} else {
		const int ixyz=Ixyz_or_0-1;
		for(I=0;I<N[0];I++){ R[0]=Rlb[0]+h*I; AxR[0]=R[0]*VecField[0]; on_edge=(I==0||I==N[0]-1);
			for(J=0;J<N[1];J++){ R[1]=Rlb[1]+h*J; AxR[1]=R[1]*VecField[1]; on_edge=( on_edge || (J==0||I==N[1]-1) );
				for(K=0;K<N[2];K++){ R[2]=Rlb[2]+h*K; AxR[2]=R[2]*VecField[2]; on_edge=( on_edge || (K==0||K==N[2]-1) );
					double vlh=Gaussian_value(lhs, R);
					double Rrhs[3]={ R[0]-displ[0], R[1]-displ[1], R[2]-displ[2] }; // R-(rhs->B+displ)
					double vrh=Gaussian_value(rhs, Rrhs); double arg=AxR[0]+AxR[1]+AxR[2];
					double prd=vlh*vrh*R[ixyz]; double xi=prd*cos(arg),eta=prd*sin(arg);
					re+= xi; im+= eta;
					if(on_edge){ edge_max=__MAX(edge_max,prd); edgesum_re+=xi; edgesum_im+=eta;}
		}}}
	}
	double wc1=__ctime;
	*pWalltime=(wc1-wc0);*pEdge_max=edge_max;*pEdge_sum=( edgesum_re*dvol+_I*(edgesum_im*dvol) );
	printf("#nmrinteg gridsize:%d,%d,%d elapsed:%f\n",N[0],N[1],N[2],wc1-wc0);
	return re*dvol+_I*(im*dvol);
	
}
// \int dx  (x**ell) exp(-gamma*(x**2))e^{iVx}  without exp(-0.25*(V**2)/gamma )
// this replaces (Gamma_hfint(m+1)*pow( sqrt(1/gamma),m+1 ) in presence of Vector field ...
// !!! NOTE WE DO NOT INCLUDE exp(-0.25*(A-dot-A)/gamma ) here !!!
// ell==0    sqrt(pi)*sqrt(1/gamma)
// ell==1    sqrt(pi)*sqrt(1/gamma) * (iV/2gamma)
dcmplx gaussian_integ1D_vcfield_(const double gamma, const int ell, const double Vmu){
	const double sqrt_pi=1.7724538509055160272981674833411;
	const double Vmu_TINY=1.0e-20;
	if(ell<2){ if(ell==0) return sqrt_pi*sqrt(1.0/gamma);
	           else       return sqrt_pi*sqrt(1.0/gamma)*(0.5*Vmu/gamma)*_I;}
	 if(fabs(Vmu)<Vmu_TINY){ if(ell%2!=0) return 0; // \int x**ell e^{-\gamma (x**2)}
	                         double argIpw=sqrt(1/gamma);return gamma_hfint(ell+1)*_INTpow( argIpw, (ell+1) );}
//	if(fabs(Vmu)<1.0e-9){ double argIpw=sqrt(1/gamma); return gamma_hfint(ell+1)*_INTpow( argIpw, (ell+1) );}
	int j;
	double inv_gamma = 1.0/gamma;
	
	// overall factor _I**ell ...
	dcmplx zfac=1;
	switch (ell%4){
		case(1):zfac=_I;break;
		case(2):zfac=(-1.0);break;
		case(3):zfac=(-_I);break;
	}
	double arg_intpow=0.5*Vmu*inv_gamma;
	double dum = sqrt_pi * sqrt( inv_gamma ) * _INTpow( arg_intpow, ell);
	double cum = dum;
	double fac = -4.0*gamma/(Vmu*Vmu); double hfint=0.50;   //  -(0.5*Vmu*inv_gamma)^{-2}/gamma;
	for(j=2;j<=ell;j+=2){
		dum*= fac * hfint;    
		cum += l_nCk(ell,j)*dum;  // ell_C_j * dum*(hfint*sqrt_pi*(
		hfint+=1.0;
	}
	return cum*zfac;
} 

dcmplx nmr_gaussian_integ1D_vcfield(const double gamma, const int ell, const double Vmu,const double *scalefac){
	double Rmax=scalefac[0]/sqrt(gamma);
	double dR=scalefac[1]/sqrt(gamma);
	int N=__NINT( Rmax/dR );
	int i; double X=-Rmax; dcmplx cum=0;
	for(i=-N;i<N;i++,X+=dR){
		double arg=Vmu*X;
		dcmplx zfac=cos(arg)+_I*sin(arg);
		cum+= pow(X,ell)*exp(-gamma*X*X)*zfac;
	}
	return cum*dR;
}
int dbg_gaussian_integ1D_vcfield(const double gamma,const int elmax,const double Vmu,const double *scalefac){
	int ell;int nerr=0;
	FILE *fp=fopen("dbg_gaussian_integ1D_vcfield.log","a");
	fprintf(fp,"#gamma=%e Vmu=%f scle:%f,%f\n",gamma,Vmu,scalefac[0],scalefac[1]);
	const double TOL=1.0e-7;
	double expfac=exp(-0.25*Vmu*Vmu/gamma);
	for(ell=0;ell<=elmax;ell++){
		dcmplx testee=gaussian_integ1D_vcfield_(gamma, ell, Vmu);
		testee*=expfac;
		dcmplx refr=nmr_gaussian_integ1D_vcfield(gamma, ell, Vmu, scalefac);
		double dev=cabs(refr-testee);
		if( dev > TOL ){
			double scalefac2[2]={scalefac[0]*1.50, scalefac[1]*0.66666};
			dcmplx refr2=nmr_gaussian_integ1D_vcfield(gamma, ell, Vmu, scalefac2);
			double dev2=cabs(refr2-testee);
			fprintf(fp," %3d    %16.8f %16.8f    %16.8f %16.8f    %14.4e    %16.8f %16.8f    %14.4e\n",
			            ell, creal(testee),cimag(testee), creal(refr2),cimag(refr2),dev2,
			            creal(refr),cimag(refr),dev,( dev2<TOL ? "":"ERROR")); if(dev2>=TOL)nerr++;
		} else {
			fprintf(fp," %3d    %16.8f %16.8f    %16.8f %16.8f    %14.4e\n",
			            ell, creal(testee),cimag(testee), creal(refr),cimag(refr),dev);
		}
	}
	fclose(fp);
	system("ls -ltrh dbg_gaussian_integ1D_vcfield.log");
	return nerr;
}
	
// \int dx x^{ell} e^{-\gamma x**2 } e^{iVmu x}
// = 
// obviously, ell == 0 : \sqrt(pi/gamma) etc
// for Vmu==0 : Gamma_hfint(ell+1) \sqrt(1/gamma)**(ell+1)
dcmplx gaussian_integ1D_vcfield_verbose_(const double gamma, const int ell, const double Vmu){
	printf("#gaussian_integ1D_vcfield_verbose_:%f %d %f\n",gamma,ell,Vmu);fflush(stdout);
	const double sqrt_pi=1.7724538509055160272981674833411;
	if(ell<2){ if(ell==0) return sqrt_pi*sqrt(1.0/gamma);
	           else       return sqrt_pi*sqrt(1.0/gamma)*(0.5*Vmu/gamma)*_I;}
	if(fabs(Vmu)<1.0e-9){ double argIpw=sqrt(1/gamma); return gamma_hfint(ell+1)*_INTpow( argIpw, (ell+1) );}
	int j;
	double inv_gamma = 1.0/gamma;
	
	// overall factor _I**ell ...
	dcmplx zfac=1;
	switch (ell%4){
		case(1):zfac=_I;break;
		case(2):zfac=(-1.0);break;
		case(3):zfac=(-_I);break;
	}
	double arg_intpow=0.5*Vmu*inv_gamma;
	double dum = sqrt_pi * sqrt( inv_gamma ) * _INTpow( arg_intpow, ell);
	printf("#gaussian_integ1D_vcfield_verbose_:arg:%f dum:%f\n",arg_intpow,dum);fflush(stdout);
				//#gaussian_integ1D_vcfield_verbose_:arg:0.000000 dum:0.000000
	double cum = dum;
	double fac = -4.0*gamma/(Vmu*Vmu); double hfint=0.50;   //  -(0.5*Vmu*inv_gamma)^{-2}/gamma;
	for(j=2;j<=ell;j+=2){
		dum*= fac * hfint;    
		printf("#gaussian_integ1D_vcfield_verbose_:j:%d dum:%f\n",j,dum);fflush(stdout);
		cum += l_nCk(ell,j)*dum;  // ell_C_j * dum*(hfint*sqrt_pi*(
		hfint+=1.0;
	}
	return cum*zfac;
} 