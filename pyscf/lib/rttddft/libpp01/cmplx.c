#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#include "macro_util.h"
#include "dcmplx.h"

dcmplx *z1alloc(int n){
	dcmplx *v;
	v=(dcmplx *)malloc( (size_t) (n*sizeof(dcmplx)) );
	return v;
}
dcmplx *z1alloc_i(int n,const dcmplx val){
	dcmplx *ret=z1alloc(n);
	int i;
	for(i=0;i<n;i++) ret[i]=val;
	return ret;
}
void z1free(dcmplx *v){
	free(v);
}

dcmplx **z2alloc(int n,int m){
	dcmplx **v;
	int i;
	v=(dcmplx **)malloc( (size_t) (n*sizeof(dcmplx *)) );	
	v[0]=(dcmplx *)malloc( (size_t) (n*m*sizeof(dcmplx)) );
	for(i=1;i<n;i++) v[i]=v[i-1]+m;   // v[i]: addr( v[i][0] ) etc
	return v;
}
void z2free(dcmplx **v){
	free(v[0]);
	free(v);
}

dcmplx **z2clone(dcmplx **src,int n,int m){
	dcmplx **v=z2alloc(n,m);
	int i,j;
	for(i=0;i<n;i++){ for(j=0;j<m;j++){ v[i][j]=src[i][j]; } }
	return v;
}

dcmplx ***z3alloc(int n,int m,int l){
	dcmplx ***v;
	int i,j;
	if( n<=0 || m<=0 || l<=0 ){ 
		fprintf(stdout,"!E d3alloc illegal arg:%d %d %d\n",n,m,l);
		fflush(stdout);
		exit(1);
	}

	v=(dcmplx ***)malloc( (size_t) (n*sizeof(dcmplx **)) );
	v[0]=(dcmplx **)malloc( (size_t) (n*m*sizeof(dcmplx *)) );
	v[0][0]=(dcmplx *)malloc( (size_t) (n*m*l*sizeof(dcmplx)) );
	i=0;
	for(j=1;j<m;j++) v[0][j]=v[0][j-1]+l;

	for(i=1;i<n;i++){
		v[i]=v[i-1]+m;
		v[i][0]=v[i-1][0]+(l*m);
 		for(j=1;j<m;j++) v[i][j]=v[i][j-1]+l;   // v[i]: addr( v[i][0] ) etc
	}
	
	return v;
}
void z3free(dcmplx ***v){
	free(v[0][0]);	
	free(v[0]);
	free(v);
}
void z4free(dcmplx ****v){
	free(v[0][0][0]);
	free(v[0][0]);
	free(v[0]);
	free(v);
}
dcmplx ***z3alloc_i(int N1,int N2,int N3, dcmplx val){
	dcmplx ***ret=z3alloc(N1,N2,N3);
	int i,j,k;
	for(i=0;i<N1;i++){ for(j=0;j<N2;j++){ for(k=0;k<N3;k++){ 
		ret[i][j][k]=val;
	}}}
	return ret;
}

dcmplx ****z4alloc(int I,int J,int K,int L){
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
	dcmplx ****v;
	int i,j,k;
	if( I<=0 || J<=0 || K<=0 || L<=0 ){ 
		fprintf(stdout,"!E d4alloc illegal arg:%d %d %d\n",I,J,K);
		fflush(stdout);
		exit(1);
	}
	v=(dcmplx ****)malloc( (size_t) (I*sizeof(dcmplx ***)) );
	v[0]=(dcmplx ***)malloc( (size_t) (I*J*sizeof(dcmplx **)) );
	v[0][0]=(dcmplx **)malloc( (size_t) (I*J*K*sizeof(dcmplx *)) );
	v[0][0][0]=(dcmplx *)malloc( (size_t) (I*J*K*L*sizeof(dcmplx)) );
	printf("#z4alloc:%p %d\n",v[0][0][0], I*J*K*L);fflush(stdout);
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
		v[i][0]=v[i-1][0]+(J*K);         // we have I*  J*K dcmplx* s//[*][0]
		for(j=1;j<J;j++){
			v[i][j]=v[i][j-1]+K;           //[*][*]
		}	
		v[i][0][0]=v[i-1][0][0]+(J*K*L); // we have I   J*K*L dcmplx s		
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
