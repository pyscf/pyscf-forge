#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ushort.h"
#include "mathfn.h"
#include "macro_util.h"
#include "dcmplx.h"
#include "dalloc.h"
#include "Gaussian.h"
double calc_Ng(const double alpha,const ushort *pw){
	double prod=1.0;int i;
  for(i=0;i<3;i++){
      prod=prod* gamma_hfint(2*pw[i]+1) * pow( ( sqrt(1.0/(2.0*alpha)) ), 2.0*pw[i]+1.0 );}
  return prod;
}
double calc_Nl(const double alpha,const int l){
	return 0.5*gamma_hfint(2*l+3)/ pow( sqrt( 2*alpha ), 2*l+3 );
	//prod=HALF*gamma_hfint(2*l+3)/( (sqrt(TWO*alpha))** (2*l+3) )
}
double calc_Nlx(const double alpha,const int l,const int I){
	return 0.5*gamma_hfint(2*(l+2*I)+3)/ pow( sqrt( 2*alpha ), 2*(l+2*I)+3 );
	//prod=HALF*gamma_hfint(2*l+3)/( (sqrt(TWO*alpha))** (2*l+3) )
}
/*
see macros in Gaussian.h ...
void Gaussian_toString(Gaussian *G){
	char *ret=ch1alloc(1000+1);
	int i,off=0;
	for(i=0;i<G->N;i++){
	}
}
void print_Bset(Gaussian **BS,const int Nb,const char *fpath,const char *description){
	FILE *fp=fopen(fpath,"w");
	int ib;
	for(ib=0;ib<Nb;ib++){
		
	}
}
*/
Gaussian *new_Gaussian(const double *alpha,const double *R,const ushort *pw,int N,const double *cof,int Jnuc){
  return new_Gaussian_1(1,alpha,R,pw,NULL,N,cof,Jnuc);
}
Gaussian *new_Gaussian_1(char divide_by_Ng,const double *alpha,const double *R,const ushort *pow1,ushort **pow_a,int N,const double *cof,int Jnuc){
	//printf("#new_Gaussian_1:%p %p %p %p %d %p",alpha,R,pow1,pow_a,N,cof);fflush(stdout);
  Gaussian *this=(Gaussian *)malloc(sizeof(Gaussian));
  this->description=NULL;
  this->N=N;this->Jnuc=Jnuc;
  // this->pows=(ushort **)malloc(N*sizeof(ushort *));
  // this->pows[0]=(ushort *)malloc(N*3*sizeof(ushort));
  this->pows=ush2alloc(N,3);
  int j,k;
  
  if(pow_a !=NULL){
    for(j=0;j<N;j++)for(k=0;k<3;k++) this->pows[j][k]=pow_a[j][k];
  } else {
    for(j=0;j<N;j++)for(k=0;k<3;k++) this->pows[j][k]=pow1[k];
  }  
  this->alpha=d1clone(alpha,N);
  this->cofs=d1clone(cof,N);
  this->R=d1clone(R,3);
  this->Ng=d1alloc(N);
  for(j=0;j<N;j++)this->Ng[j]=calc_Ng(alpha[j],this->pows[j]);
  if(divide_by_Ng){ 
  	for(j=0;j<N;j++)this->cofs[j]/=sqrt(this->Ng[j]);
  }
  /*
	int i;
	for(i=0;i<this->N;i++){
		printf("#new_Gaussian:%p %d %f %f (%f)\n",this,i,this->alpha[i],this->cofs[i],cof[i]);
	}*/
	//printf("#new_Gaussian:%p %d,%d,%d %f,%f,%f\n",this,this->pows[0],this->pows[1],this->pows[2],this->R[0],this->R[1],this->R[2]);
  return this;
}
SphericalGaussian *new_SphericalGaussian(const double *alpha, const double *R,int l,int m,int N, const double *cof,int Jnuc){
	return new_SphericalGaussianX(alpha,R,l,m,N,cof,Jnuc,0);
}
SphericalGaussian *new_SphericalGaussianX(const double *alpha,const double *R,int l,int m,int N,const double *cof,int Jnuc,int Ipow){
	
	SphericalGaussian *this=(SphericalGaussian *)malloc(sizeof(SphericalGaussian));
	this->description=(char *)malloc(60*N);
	int jj,off=0;
	this->I=Ipow;
	this->N=N;this->Jnuc=Jnuc;
	this->l=(ushort)l;this->m=m; //printf("#new_SphericalGaussian:%d,%d\n",this->l,this->m);fflush(stdout);
	
	this->alpha=d1clone(alpha,N);
	this->cofs=d1clone(cof,N);
	//printf("#new_SphericalGaussian:%d,%d\n",this->l,this->m);fflush(stdout);
	__assertf( (this->m>=-(this->l) &&  this->m<=(this->l)),("wrong m:%d/l=%d",this->m,this->l),-1)
	
	this->R=d1clone(R,3);
	this->Nl=d1alloc(N); int k;
	for(k=0;k<N;k++)this->Nl[k]=calc_Nlx(alpha[k],this->l,this->I);// 0.5 Gamma_hfint(2l+3)/
  for(k=0;k<N;k++)this->cofs[k]/=sqrt(this->Nl[k]);
	sprintf(this->description,"l=%d,m=%d,I=%d,N=%d",l,m,Ipow,N);off=strlen(this->description);
	int verbose=0;
	if(verbose){
		for(jj=0;jj<N && off<(60*N);jj++){ 
			sprintf( this->description + off,"%12.6f %12.6f %12.6f\n",alpha[jj],cof[jj],this->cofs[jj]);off=strlen(this->description);}
	}
	//printf("#new_SphericalGaussian:%d,%d\n",this->l,this->m);fflush(stdout);
	int i;
	if(verbose){
		for(i=0;i<this->N;i++){
			printf("#new_SphericalGaussian:%p %d %f %f %f\n",this,i,this->alpha[i],this->cofs[i],this->cofs[i]*sqrt(this->Nl[i]));
		}
	}
	//printf("#new_SphericalGaussian:%p %d,%d %f,%f,%f\n",this,this->l,this->m,this->R[0],this->R[1],this->R[2]);
	return this;
}
Gaussian *Spherical_to_Cartesian(SphericalGaussian *src){
	ushort **Ppows[1]={NULL};double *Pcofs[1]={NULL};
	int Nmult=Xlm_to_Cartesian(src->l,src->m,Ppows,Pcofs);
	//printf("#Xlm_to_Cartesian:%d,%d >> %d\n",src->l,src->m,Nmult);fflush(stdout);
	ushort **xlmpws=Ppows[0];double *xlmcofs=Pcofs[0];
	// Gaussian with Npgto * N elements
	// sh_0.0 : pows[0] cofs[0]*alph[0]
	// sh_0.1 : pows[1] cofs[1]*alph[0] ...
	const int N_More_A[3]={1,3,6};
	const int N_more=N_More_A[src->I];
	__assertf( (src->I<3),("wrong Ipow:%d",src->I),-1); // here I=0,1,2 (:=i-1)
	const int Npgto0 = src->N;
	const int npgto1 = Npgto0 * Nmult;
	const int Npgto2 = Npgto0 * Nmult * N_more;
	
	double *alph=d1alloc(Npgto2),*cofs=d1alloc(Npgto2);
	ushort **pows=ush2alloc(Npgto2,3);
	int I_more; int off=0;
	// printf("#%d N_more=%d\n",src->I,N_More_A[src->I]);fflush(stdout);
	for(I_more=0;I_more<N_more;I_more++){
		int Ipw=-1,Jpw=-1,n_add=0;int fac=1;
		     if(src->I==1){ Ipw=I_more;n_add=2;fac=1;}  // add 2 to x,y,z ...
		else if(src->I==2){ int iarr[12]={ 0,0, 1,1, 2,2, 0,1, 0,2, 1,2};
												Ipw=iarr[2*I_more];Jpw=iarr[2*I_more+1]; fac=( (Ipw==Jpw) ? 1:2 );n_add=2;}
		int Ipgto,Jmult,IxJ;
		for(Ipgto=0,IxJ=0;Ipgto<Npgto0;Ipgto++){
			for(Jmult=0;Jmult<Nmult;Jmult++,IxJ++){ int k;
				for(k=0;k<3;k++) pows[ npgto1*I_more + IxJ ][k] = xlmpws[Jmult][k]; //printf("#%d.%d :%d %d %d\n",Ipgto,Jmult,pows[IxJ][0],pows[IxJ][1],pows[IxJ][2]);
				if(Ipw>=0){ pows[ npgto1*I_more + IxJ ][Ipw]+=2;}
				if(Jpw>=0){ pows[ npgto1*I_more + IxJ ][Jpw]+=2;}
				
				double Ng=calc_Ng(src->alpha[Ipgto],pows[IxJ]);
				alph[ npgto1*I_more + IxJ] = src->alpha[Ipgto];
				cofs[ npgto1*I_more + IxJ] = src->cofs[Ipgto] * xlmcofs[Jmult]*fac;
			/*	printf("#%d:%d.%d.%d  %f %f",npgto1*I_more + IxJ, 
				        pows[ npgto1*I_more + IxJ ][0], pows[ npgto1*I_more + IxJ ][1], pows[ npgto1*I_more + IxJ ][2],  
				        alph[ npgto1*I_more + IxJ ], cofs[ npgto1*I_more + IxJ ]);*/
			}
		}
	}
	//printf("#generating new..");fflush(stdout);
	Gaussian *Ga=new_Gaussian_1( 0, alph,src->R,NULL,pows,Npgto2,cofs,src->Jnuc);
	d1free(alph);d1free(cofs);ush2free(pows);alph=NULL;cofs=NULL;pows=NULL;
	ush2free(Ppows[0]);d1free(Pcofs[0]);Ppows[0]=NULL;Pcofs[0]=NULL;
	Ga->description=ch1clone(src->description);
	//printf("#generating new..%p\n",Ga);fflush(stdout);
	return Ga;
}
#define SQRDIST(A,B)  ( (A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]) + (A[2]-B[2])*(A[2]-B[2]) )
double Gaussian_value(Gaussian *G, const double *R){
	int i,N=G->N;
	double sqrdist=SQRDIST(R,G->R);
	double xA[3]={ R[0]-G->R[0], R[1]-G->R[1], R[2]-G->R[2]};
	double retv=0.0;
	for(i=0;i<N;i++){
		double prod=1.0; int dir,j;
		for(dir=0;dir<3;dir++)for(j=0;j<(G->pows[i][dir]);j++)prod*=xA[dir];
		
		double dum= prod * (G->cofs[i])*exp(-(G->alpha[i])*sqrdist ); // XXX /sqrt(G->Ng[i]);
		retv+=dum;
		//printf("#Gaussian_value:%d %f %f %f %f\n",i,dum*sqrt(G->Ng[i]),dum,retv,sqrdist);
	}
	return retv;
}
void free_Bset(Gaussian **BS,int Nb){
	int i;
	// printf("#Freeing Bset:%d....",Nb);fflush(stdout);
	for(i=0;i<Nb;i++){
		//printf("#Freeing Bset:%d/%d\n",i,Nb);fflush(stdout);
		free_Gaussian01(BS[i],0);
	}
	free(BS);//printf("... free_Bset done\n");fflush(stdout);
}
void free_Gaussian01(Gaussian *this,const int verbose){
	if(verbose){
		printf("#freeing Gaussian01:%p %p %p\n",this->pows,this->alpha,this->cofs);fflush(stdout);
	}
	free(this->pows[0]);free(this->pows);this->pows=NULL;
	if(this->description !=NULL) {free(this->description);this->description=NULL;}
	d1free(this->alpha);
	d1free(this->cofs);
	d1free(this->R);
	d1free(this->Ng);
	free(this);this=NULL;
	if(verbose){
		printf("#freeing Gaussian01:...END\n");fflush(stdout);
	}
	
}
void free_Gaussian(Gaussian *this){
	free_Gaussian01(this,0);
}
void free_SphericalGaussian(SphericalGaussian *this){
	d1free(this->alpha);
	if(this->description !=NULL){free(this->description);this->description=NULL;}
	d1free(this->cofs);
	d1free(this->R);
	d1free(this->Nl);
	free(this);this=NULL;
	
}
double SphericalGaussian_value(SphericalGaussian *G,const double *R){
	int i,N=G->N;
	double polar[3];
	double sqrdist=SQRDIST(R,G->R);
	 
	double xA[3]={ R[0]-G->R[0], R[1]-G->R[1], R[2]-G->R[2]};
	xyz_to_polar(polar,xA);double rxxel=pow(polar[0],G->l);
	//printf("#SphericalGaussian:%d,%d\n",G->l,G->m);fflush(stdout);
	double Ylm=dSpher(G->l,G->m,polar[1],polar[2]);
	double fac =rxxel * Ylm; 
	double retv=0.0;
	for(i=0;i<N;i++){
		double dum= G->cofs[i]*exp(-(G->alpha[i])*sqrdist ); // XXX /sqrt(G->Nl[i]);
		retv+=dum;
		//printf("#sphGaussian_value:%d %f %f %f %f %f\n",i,dum*sqrt(G->Nl[i]),dum,retv,retv*fac,sqrdist);
	}
	if(G->I){
		double Rsqr=polar[0]*polar[0];
		for(i=0;i<G->I;i++)retv*=Rsqr; //R^{2*I}
	}
	return retv*fac;
}

// 
#define PUSH3(arr,p,q,r)  { arr[0]=p;arr[1]=q;arr[2]=r; }
int Xlm_to_Cartesian(const int l,const int m,ushort ***pows,double **cofs){
	// on return, ***pows[0]=i2alloc(N,3)
	//             **cofs[0]=d1alloc(N)    and returns N
	const double PI=3.1415926535897932384626433832795;
	int N;
	if(l==0){ N=1;
		pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
		PUSH3(pows[0][0],0,0,0);
		cofs[0][0]=sqrt(1.0/(4.0*PI)); return N;
	} 
	if(l==1){ N=1;
		pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N); double dum=sqrt(3.0/((4.0d+00)*PI));
			   if(m==-1){ PUSH3(pows[0][0],0,1,0); cofs[0][0]=dum;  return N;}
		else if(m==0){  PUSH3(pows[0][0],0,0,1); cofs[0][0]=dum;  return N;}
		else if(m==1){  PUSH3(pows[0][0],1,0,0); cofs[0][0]=dum; return N;}
	} else if(l==2){ 
		const int Nret[5]={1,1,3,1,2};int N=Nret[m+2];
		pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
		double dum=sqrt( 15.0/(4.0*PI) );
           if(m==-2){ PUSH3(pows[0][0],1,1,0); cofs[0][0]=dum; return N;
	  } else if(m==-1){ PUSH3(pows[0][0],0,1,1); cofs[0][0]=dum;return N;
    } else if(m== 0){ dum=sqrt(5.0/(4.0*PI)); 
                      PUSH3(pows[0][0],0,0,2); cofs[0][0]=dum;
                      PUSH3(pows[0][1],2,0,0); cofs[0][1]=-0.5*dum;
                      PUSH3(pows[0][2],0,2,0); cofs[0][2]=-0.5*dum;  return N;
	  } else if(m== 1){ PUSH3(pows[0][0],1,0,1); cofs[0][0]=dum;      return N; 
    } else if(m== 2){ PUSH3(pows[0][0],2,0,0); cofs[0][0]=0.5*dum;
                      PUSH3(pows[0][1],0,2,0); cofs[0][1]=-0.5*dum;  return N;}
	} else if(l==3){
		double dum=sqrt(35.0/(32.0*PI));
		if(m==-3){ N=2;pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                      PUSH3(pows[0][0],2,1,0); cofs[0][0]= 3*dum;
                      PUSH3(pows[0][1],0,3,0); cofs[0][1]=   -dum;  return N;
		} else if(m== 3){ N=2;pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
			                PUSH3(pows[0][0],3,0,0); cofs[0][0]=    dum;
                      PUSH3(pows[0][1],1,2,0); cofs[0][1]= -3*dum;  return N;
		} else {
			N=(m==-2 ? 1:(m==2 ? 2:3));
			pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
			dum=sqrt(105.0/(4.0*PI));
			if(m==-2){        PUSH3(pows[0][0],1,1,1); cofs[0][0]=dum;     return N;
			} else if(m== 2){ PUSH3(pows[0][0],2,0,1); cofs[0][0]=0.5*dum;
                      PUSH3(pows[0][1],0,2,1); cofs[0][1]=-0.5*dum;return N;
			} else {
				dum=sqrt(21.0/(32.0*PI));
				if(m==-1){ PUSH3(pows[0][0],0,1,2); cofs[0][0]=4*dum;
                 PUSH3(pows[0][1],2,1,0); cofs[0][1]=-dum;
                 PUSH3(pows[0][2],0,3,0); cofs[0][2]=-dum;return N;
				} if(m== 1){ PUSH3(pows[0][0],1,0,2); cofs[0][0]=4*dum;
                   PUSH3(pows[0][1],3,0,0); cofs[0][1]=-dum;
                   PUSH3(pows[0][2],1,2,0); cofs[0][2]=-dum;return N;
				} else {
           dum=sqrt(7.0/(16.0*PI));
           PUSH3(pows[0][0],0,0,3); cofs[0][0]=2*dum;
           PUSH3(pows[0][1],2,0,1); cofs[0][1]=-3*dum;
           PUSH3(pows[0][2],0,2,1); cofs[0][2]=-3*dum; return N;
				}
			}
		}
	}else if(l==4){
		double dum;
    switch (abs(m)){
       // sqrt(2) times Ylm-prefactor (with sign (-)^{|m|})  ... except m==0
       case(4): dum= (3/16.0)*sqrt(35.0/PI);break;
       case(3): dum= (3/8.0)*sqrt(70.0/PI);break;
       case(2): dum= (15.0/8.0)*sqrt( 1.0/(5*PI) );break;
       case(1): dum= (15.0/8.0)*sqrt( 2.0/(5*PI) );break;
       case(0): dum= sqrt(9.0/(4*PI))/8.0;break;
    }
    switch(m){
        case(-4):  N=2; pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],1,3,0);cofs[0][0]=-4*dum;
                   PUSH3(pows[0][1],3,1,0);cofs[0][1]= 4*dum; return N;
        case( 4):  N=3; pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],0,4,0);cofs[0][0]= dum;
                   PUSH3(pows[0][1],2,2,0);cofs[0][1]=-6*dum;
                   PUSH3(pows[0][2],4,0,0);cofs[0][2]= dum; return N;
        case(-3):  N=2; pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],2,1,1);cofs[0][0]= 3*dum;
                   PUSH3(pows[0][1],0,3,1);cofs[0][1]=  -dum; return N;
        case( 3):  N=2; pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],3,0,1);cofs[0][0]=   dum;
                   PUSH3(pows[0][1],1,2,1);cofs[0][1]=-3*dum; return N;

        case(-2):  N=3; pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],1,1,2);cofs[0][0]= 12*dum;
                   PUSH3(pows[0][1],3,1,0);cofs[0][1]=-2*dum;
                   PUSH3(pows[0][2],1,3,0);cofs[0][2]=-2*dum; return N;
        case( 2):  N=4; pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],2,0,2);cofs[0][0]= 6*dum;
                   PUSH3(pows[0][1],4,0,0);cofs[0][1]=  -dum;
                   PUSH3(pows[0][2],0,2,2);cofs[0][2]=-6*dum;
                   PUSH3(pows[0][3],0,4,0);cofs[0][3]=   dum; return N;

        case(-1):  N=3;pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],0,1,3);cofs[0][0]= 4*dum;
                   PUSH3(pows[0][1],2,1,1);cofs[0][1]=-3*dum;
                   PUSH3(pows[0][2],0,3,1);cofs[0][2]=-3*dum; return N;
        case( 1):  N=3;pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],1,0,3);cofs[0][0]= 4*dum;
                   PUSH3(pows[0][1],3,0,1);cofs[0][1]=-3*dum;
                   PUSH3(pows[0][2],1,2,1);cofs[0][2]=-3*dum; return N;
        case( 0):  N=6;pows[0]=ush2alloc(N,3);cofs[0]=d1alloc(N);
                   PUSH3(pows[0][0],0,0,4);cofs[0][0]=  8*dum;
                   PUSH3(pows[0][1],0,4,0);cofs[0][1]=  3*dum;
                   PUSH3(pows[0][2],4,0,0);cofs[0][2]=  3*dum;

                   PUSH3(pows[0][3],2,0,2);cofs[0][3]=-24*dum;
                   PUSH3(pows[0][4],0,2,2);cofs[0][4]=-24*dum;
                   PUSH3(pows[0][5],2,2,0);cofs[0][5]=  6*dum; return N;
    }
  }               
	__assertf(0,("Xlm_to_Cartesian:%d.%d",l,m),-1);
  return -1;
}
