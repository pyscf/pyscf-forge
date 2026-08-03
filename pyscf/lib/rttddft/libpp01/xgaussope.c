#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dcmplx.h"
#include "ushort.h"
#include "Gaussian.h"
#include "cstrutil.h"
#include "xgaussope.h"
#include "macro_util.h"
#include "dalloc.h"
static char Ope_='\0';  //'o' or 'x'
static int N_call_=0;  //0:default,
static int MPIrank_=0; //caller's threadno...
#define _Printf(args)  { if(MPIrank_==0){ printf args;} }
#define _Fprintf(args) { if(MPIrank_==0){ fprintf args;} }
#define _Fwrite(fpath,Append,dtme,dmyfp,args) { FILE *dmyfp=fopen(fpath, ((Append) ? "a":"w") );fprintf args;\
if(dtme){__fprtdtme(dmyfp,"\t\t ","\n") } fclose(dmyfp);}

#define Dot_3d(A,B)  (A[0]*B[0] + A[1]*B[1] + A[2]*B[2])
#define dist3D(A,B)  (sqrt( (A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]) + (A[2]-B[2])*(A[2]-B[2]) ))

unsigned short *get_ush1(unsigned short L){
    return (unsigned short *)malloc(L*sizeof(unsigned short));
} 
void free_ush1(unsigned short *p){
    free(p);
}

#define _get_ush1( ret, L, lret){\
ret=get_ush1(L);}
#define _free_ush1( ptr, L, laddr){\
free(ptr);}
#define _clone_ush1( org, ret, L, lret, kk) {\
_get_ush1( ret,L,lret);for(kk=0;kk<L;kk++) ret[kk]=org[kk];}
#define _sqrdist1(A,B)  ( (A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]) + (A[2]-B[2])*(A[2]-B[2]) )

void test_ovlp(){
	double A[3]={1,0,-2},B[3]={1.2,0.1,-2.1},C[3]={2,1,-1};
	double alph[4]={1.192, 18.68, 0.0645, 0.894};
	double cofs[4]={1.0,    0.5,  -0.7,   1.0};
	double alphB[2]={0.794, 15.82};
	double cofsB[2]={0.7,    0.4};
	double zeroField[3]={0,0,0};
	double zeroDispl[3]={0,0,0};
	
	const ushort es[3]={0,0,0},px[3]={1,0,0},py[3]={0,1,0};
	Gaussian *s1=new_Gaussian(alph,A,es,1,cofs,0);
	Gaussian *s2=new_Gaussian(alph+1,B,es,1,cofs,1);
	Gaussian *s3=new_Gaussian(alph+2,C,es,1,cofs,2);
	Gaussian *x1=new_Gaussian(alph+3,A,es,1,cofs,0);
	dcmplx s11=Gaussian_overlap(s1,s1,zeroField,zeroDispl);printf("#Ovlp:s11:%f+j%f\n",creal(s11),cimag(s11));
	dcmplx s12=Gaussian_overlap(s1,s2,zeroField,zeroDispl);printf("#Ovlp:s12:%f+j%f\n",creal(s12),cimag(s12));
}


/*
Nsh[C]=6
Npgto[\sum(Nsh[:])]
shell_0 : el[0]=0   .... C    S     >> nCGTO+=1
   4563.240                  0.00196665
    682.0240                 0.0152306
    154.9730                 0.0761269
     44.45530                0.2608010
     13.02900                0.6164620
      1.827730               0.2210060

shell_1 : C    SP                     nCGTO+=(
shell_2 :                            nCGTO+=(2*l+1)
     20.96420                0.114660               0.0402487
      4.803310               0.919999               0.237594
      1.459330              -0.00303068             0.815854
shell_3 : C    SP                     nCGTO+=(
shell_4 :                            nCGTO+=(2*l+1)
C    SP
      0.4834560              1.000000               1.000000
C    SP
      0.1455850              1.000000               1.000000
*/
// ell[ \sum(Nsh[:]) ] npGTO[ \sum(Nsh[:]) ] alph[ \sum(npGTO[:]) 
// 
Gaussian ***gen_bsetx3(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *n_cols,const int *npGTO,const double *alph,const double *cofs){
	static int n_call_loc=0;n_call_loc++;
	Gaussian ***ret=(Gaussian ***)malloc(3*sizeof(Gaussian **));
	int n0=0,n1=0,n2=0;const int verbose=0;
	__assertf(n_cols!=NULL,("gen_bsetx3.000"),-1);
	ret[0]=gen_bset01(&n0,nAtm,Rnuc,IZnuc, nDa, distinctIZnuc, Nsh, ell, n_cols, npGTO, alph, cofs, 0, verbose);
	ret[1]=gen_bset01(&n1,nAtm,Rnuc,IZnuc, nDa, distinctIZnuc, Nsh, ell, n_cols, npGTO, alph, cofs, 1, verbose);
	ret[2]=NULL;
	const int NshSUM=i1sum(Nsh,nDa);
	int i;int nMx=-1; for(i=0;i<NshSUM;i++) nMx=__MAX(n_cols[i],nMx);
	if(nMx>2){ ret[2]=gen_bset01(&n2,nAtm,Rnuc,IZnuc, nDa, distinctIZnuc, Nsh, ell, n_cols, npGTO, alph, cofs, 2, (verbose||(n_call_loc==1 && N_call_==1 && MPIrank_==0)));
	           if( (N_call_==1||N_call_==2) && MPIrank_==0 ){ printf("#gen_bsetX3 allocating 02-th bset %d...\n",n2);fflush(stdout);} }
	pNcGTO[0]=n0;
	pNcGTO[1]=n1;
	pNcGTO[2]=n2;
	return ret;
}
Gaussian **gen_bset(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *n_cols,const int *npGTO,const double *alph,const double *cofs, const int I_col){
	const int verbose=0;
	return gen_bset01(pNcGTO, nAtm, Rnuc, IZnuc, nDa, distinctIZnuc, Nsh, ell, n_cols, npGTO, alph, cofs,  I_col, verbose);
}
Gaussian **gen_bset01(int *pNcGTO,const int nAtm,double *Rnuc,const int *IZnuc,const int nDa,const int *distinctIZnuc,
    const int *Nsh,const int *ell,const int *n_cols,const int *npGTO,const double *alph,const double *cofs,const int I_col,char verbose){
	if(verbose){ printf("#gen_bset01:I_col=%d %p %d %p %p %d %p %p\n",I_col,pNcGTO,nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh);
	             fflush(stdout);
	             int NshSUM=i1sum(Nsh,nDa);
	             ___prtD1(("#gen_bset01:Rnuc"),Rnuc,(nAtm*3))
	             ___prtI1(("#gen_bset01:IZnuc"),IZnuc,(nAtm))
	             ___prtI1(("#gen_bset01:distinctIZnuc"),distinctIZnuc,nDa)
	             ___prtI1(("#gen_bset01:Nsh"),Nsh,nDa);
	             ___prtI1(("#gen_bset01:ell"),ell,NshSUM);
	             if(n_cols !=NULL ){
									___prtI1(("#gen_bset01:n_cols"),n_cols,NshSUM);
								}
	             ___prtI1(("#gen_bset01:npGTO"),npGTO,NshSUM);
	             int npgtoSUM=i1sum(npGTO,NshSUM);
	             ___prtD1(("#gen_bset01:alph"),alph,npgtoSUM);
	             ___prtD1(("#gen_bset01:cofs"),cofs,npgtoSUM);
	             }
#define ___fprtD1(fptr,msg,A,L)        { int kprtd1;   fprintf msg;\
for(kprtd1=0;kprtd1<L;kprtd1++) fprintf(fptr,"%f ",A[kprtd1]); fprintf(fptr,"\n");}
#define ___fprtI1(fptr,msg,A,L) { fprintf msg;int kkprti1;\
for(kkprti1=0;kkprti1<L;kkprti1++) fprintf(fptr,"%d ",A[kkprti1]); fprintf(fptr,"\n");}

	if(1){
		FILE *fp1=fopen("gen_bset01.log","w");
     ___fprtD1(fp1, (fp1,"#gen_bset01:Rnuc"),Rnuc,(nAtm*3));
     ___fprtI1(fp1, (fp1,"#gen_bset01:IZnuc"),IZnuc,(nAtm));
     ___fprtI1(fp1, (fp1,"#gen_bset01:distinctIZnuc"),distinctIZnuc,nDa);
     ___fprtI1(fp1, (fp1,"#gen_bset01:Nsh"),Nsh,nDa);
     int NshSUM=i1sum(Nsh,nDa);
     ___fprtI1(fp1, (fp1,"#gen_bset01:ell"),ell,NshSUM);
     ___fprtI1(fp1, (fp1,"#gen_bset01:npGTO"),npGTO,NshSUM);
     int npgtoSUM=i1sum(npGTO,NshSUM);
     ___fprtD1(fp1, (fp1,"#gen_bset01:alph"),alph,npgtoSUM);
     ___fprtD1(fp1, (fp1,"#gen_bset01:cofs"),cofs,npgtoSUM);
		fclose(fp1);
	}
	if(I_col>0){ __assertf( (n_cols !=NULL),("n_cols %d",I_col),-1);}
	int idbgng=3;
	int **Ncols=NULL;
	int **Ell=i2allocx(nDa,Nsh);
	if(verbose && idbgng){ printf("#gen_bset01.DBGNG:0010\n");fflush(stdout);}
	int **Npgto=i2allocx(nDa,Nsh);
	if(verbose && idbgng){ printf("#gen_bset01.DBGNG:0020\n");fflush(stdout);}
	int ida,jsh,ixj,ko,ixjxk;
	for(ida=0,ixj=0;ida<nDa;ida++)for(jsh=0;jsh<Nsh[ida];jsh++,ixj++){ Ell[ida][jsh]=ell[ixj];
                                                                     Npgto[ida][jsh]=npGTO[ixj];}
  if(n_cols !=NULL){
		Ncols=i2allocx(nDa,Nsh);
		for(ida=0,ixj=0;ida<nDa;ida++)for(jsh=0;jsh<Nsh[ida];jsh++,ixj++){ Ncols[ida][jsh]=n_cols[ixj];printf("Ncols_%d:%d",ixj,n_cols[ixj]);}
	}
	int append_logf=0;
	if(verbose && idbgng){ printf("#gen_bset01.DBGNG:0030");fflush(stdout);}
  double ***Alph=d3allocx(nDa,Nsh,Npgto);
  double ***Cofs=d3allocx(nDa,Nsh,Npgto);
	if(verbose && idbgng){ printf("#gen_bset01.DBGNG:0040");fflush(stdout);}
  for(ida=0,ixjxk=0;ida<nDa;ida++){ for(jsh=0;jsh<Nsh[ida];jsh++){ for(ko=0;ko<Npgto[ida][jsh];ko++,ixjxk++){ 
		Alph[ida][jsh][ko]=alph[ixjxk];Cofs[ida][jsh][ko]=cofs[ixjxk];
	} } }
	if(verbose && idbgng){ printf("#gen_bset01.DBGNG:0050");fflush(stdout);}
	Gaussian **BS=NULL;int loop;
	FILE *fpBSLOG=NULL;
	if(N_call_==1 && MPIrank_==0){ fpBSLOG=fopen("gen_bset01_BS.log","a");__fprtdtme(fpBSLOG,"#gen_bset01:","\n");}
	for(loop=0;loop<2;loop++){
		int nCGTO=0;
		
		int iat;
		for(iat=0;iat<nAtm;iat++){
			//printf("#top of iat loop:iat=%d",iat);fflush(stdout);
			for(ida=0;ida<nDa;ida++){ if(distinctIZnuc[ida]==IZnuc[iat])break;} __assertf(ida<nDa,("ida=%d/%d",ida,nDa),-1);
			if(verbose && idbgng){ printf("#gen_bset01.DBGNG:BS_%d %d %d\n",iat,ida,Nsh[ida]);fflush(stdout);}
			for(jsh=0;jsh<Nsh[ida];jsh++){
				const int el=Ell[ida][jsh], npgto=Npgto[ida][jsh];
				if(I_col){
					//FILE *fptr=fopen("gen_bset01.log",( append_logf ? "a":"w"));append_logf=1;
					//fprintf(fptr,"ida=%d,jsh=%d  l=%d  alph=%f, cofs=%f : Ncols:%d\n",ida,jsh,Ell[ida][jsh],Alph[ida][jsh][0],Cofs[ida][jsh][0],Ncols[ida][jsh]);
					//fclose(fptr);
					if(I_col>=Ncols[ida][jsh]){ /*printf("gen_bset:skipping.. el=%d",el);*/ continue;} // I_col==1 && Ncols[ida][jsh]<=1 skip
				}
				const int nmult=2*el+1; int m;
				if( BS != NULL ){
					const int p_to_xyz[3]={1,-1,0}; // somehow p-type AOs are aligned in x,y,z order
					for(m=-el;m<=el;m++){ const int em=(el==1 ? p_to_xyz[m+el]:m);
						if(verbose){ printf("#gen_bset01:BS:%d:el=%d nMult=%d\n",nCGTO+1,el,nmult);fflush(stdout);} 
						SphericalGaussian *spg=new_SphericalGaussianX(Alph[ida][jsh],Rnuc+3*iat,el,em,Npgto[ida][jsh],Cofs[ida][jsh],iat,I_col);
						if(verbose){ printf("#gen_bset01:BS:%d:el=%d nMult=%d\n",nCGTO+1,el,nmult);fflush(stdout);}
						//if(I_col){
						//	FILE *fptr=fopen("gen_bset01.log",( append_logf ? "a":"w"));append_logf=1;
						//	fprintf(fptr,"#%d:%s\n",nCGTO,spg->description);fclose(fptr);
						//} 
						BS[nCGTO]=Spherical_to_Cartesian(spg); free_SphericalGaussian(spg);spg=NULL; 
						double zerovec[3]={0,0,0};
						dcmplx ovlp=Gaussian_overlap(BS[nCGTO],BS[nCGTO],zerovec,zerovec);
						// FILE *fp2=fopen("gen_bset01_BS.log","a");
						if( fpBSLOG !=NULL ){ Gaussian_fprtout((fpBSLOG,"#gen_bset01:%d %d,%d ovlp:%f",nCGTO,el,m,creal(ovlp)),fpBSLOG,BS[nCGTO]);}
						nCGTO++;
					}
				} else {
					nCGTO+=nmult;
					if(verbose){ printf("#gen_bset01:prep:Counting up:el=%d nMult=%d %d\n",el,nmult,nCGTO);fflush(stdout);}
				}
			}
			//printf("#end of jsh loop:iat=%d",iat);fflush(stdout);
		}
		if(BS==NULL){ /* printf("#gen_bset01;allocating %d\n",nCGTO);fflush(stdout); */
			            BS=(Gaussian **)malloc(nCGTO*sizeof(Gaussian *)); *pNcGTO=nCGTO;}
  }
  if( fpBSLOG !=NULL ) fclose(fpBSLOG);
  int ngot=*pNcGTO;
#ifdef PRTOUT_BSET
  if(verbose){ BSet_prtout(("gen_bset:%d",ngot),BS,ngot);}
#endif
	//if(1){FILE *fp1=fopen("xgaussope_check_free.txt","a");fprintf(fp1,"BF:%p %p %p %p %p\n",Alph,Cofs,Ell,Npgto,Ncols);fclose(fp1);}
	d3free(Alph);Alph=NULL; d3free(Cofs);Cofs=NULL;
	//if(1){FILE *fp1=fopen("xgaussope_check_free.txt","a");fprintf(fp1,"#1:%p %p %p %p %p\n",Alph,Cofs,Ell,Npgto,Ncols);fclose(fp1);}
	i2free(Ell);Ell=NULL;i2free(Npgto);Npgto=NULL;
	//if(1){FILE *fp1=fopen("xgaussope_check_free.txt","a");fprintf(fp1,"#2:%p %p %p %p %p\n",Alph,Cofs,Ell,Npgto,Ncols);fclose(fp1);}
	if(Ncols!=NULL){ i2free(Ncols);Ncols=NULL; }
	//FILE *fp2=fopen("xgaussope_check_free.txt","a");fprintf(fp2,"AF:%p %p %p %p %p\n",Alph,Cofs,Ell,Npgto,Ncols);fclose(fp2);
  //__assertf(0,("BSet"),-1)
  return BS;
}
double check_nrmz(const char *title,Gaussian **BS,int Nb){
	return check_nrmz01(title,BS,Nb,0);
}
double check_nrmz01(const char *title,Gaussian **BS,int Nb,const char fix_nrmz){
	int i;
	double displ[3]={0.0, 0.0, 0.0};
	double vectorfield[3]={0.0, 0.0, 0.0};
	double maxdev=-1,devsum=0;
	dcmplx as=0;int at=-1;
	int prtout=0;
	FILE *fperr=stdout,*fpOUT=NULL;
	double devTOL=(fix_nrmz ? 1.0e-6:1.0e-3);
	int idbgng=2;char strbuf50[50];
#define PRTOUT_NRMZ01 // XXX XXX
#ifdef PRTOUT_NRMZ01
	if( fix_nrmz || idbgng ){ sprintf(strbuf50,"xgaussope_fix_nrmz_%s.log",title);fperr=( fpOUT=fopen(strbuf50,"w") ); }
#endif

	for(i=0;i<Nb;i++){
		dcmplx ovlp=Gaussian_overlap(BS[i],BS[i],vectorfield,displ);
		double sqrdev=(creal(ovlp)-1.0)*(creal(ovlp)-1.0) + cimag(ovlp)*cimag(ovlp);
		double dev=sqrt(sqrdev);
		if(dev>devTOL){
#ifdef PRTOUT_NRMZ01
			fprintf(fperr,"#un-normalized Gaussian:%s.%03d:ovlp=%f+j%f Ga:%s\n",title,i,creal(ovlp),cimag(ovlp),BS[i]->description);
#else
			if(!fix_nrmz){ printf("#un-normalized Gaussian:%s.%03d:ovlp=%f+j%f Ga:%s\n",title,i,creal(ovlp),cimag(ovlp),BS[i]->description);}
#endif
			if(fix_nrmz){
				__assertf( fabs(cimag(ovlp))<1.0e-7,(""),-1);
				double fac=1.0/sqrt( creal(ovlp) );
				int j; for(j=0;j<(BS[i]->N);j++) BS[i]->cofs[j]*=fac;
				ovlp=Gaussian_overlap(BS[i],BS[i],vectorfield,displ);
				double dev2=sqrt( (creal(ovlp)-1.0)*(creal(ovlp)-1.0) + cimag(ovlp)*cimag(ovlp) );
				__assertf(dev2<1.0e-7,(""),-1);
			}
/*
#un-normalized Gaussian:ovlp=1.205603+j0.000000 Ga:    0.712264    -0.308844
    0.262870     0.019606
    0.116086     1.131034

*/
		}
		if(dev>maxdev){ maxdev=dev;at=i;as=ovlp;}
		devsum+=sqrdev;
	}
#ifdef PRTOUT_NRMZ01
	fprintf(fperr,"#check_nrmz:%s:maxdev=%e [%d]:S=%f+j%f",title,maxdev,at,creal(as),cimag(as));
	if(fpOUT!=NULL){ fclose(fpOUT);} fperr=NULL;
#else
	if( (!fix_nrmz) && (maxdev>devTOL) ){ printf("#check_nrmz:%s:maxdev=%e [%d]:S=%f+j%f",title,maxdev,at,creal(as),cimag(as));}
#endif
	if(maxdev>1.0e-4 && (!fix_nrmz) ){
		int j;Gaussian *G=BS[at];
		printf("#BS_%04d",at);for(j=0;j<G->N;j++){ printf("%d.%d.%d  %f \t %f\n", G->pows[j][0],G->pows[j][1],G->pows[j][2],G->alpha[j],G->cofs[j]);}
	}
	return maxdev;
	//__assertf( maxdev<1.0e-3,("wrong ovlp:%e [%d]:S=%f+j%f",maxdev,at,creal(as),cimag(as)),-1)
}
dcmplx *calc_pbc_overlaps_dmy(const int nAtm,      const double *Rnuc,  const int *IZnuc, const int nDa,      const int *distinctIZnuc, //1,2,3,4,5
                              const int *Nsh,      const int *ell,      const int *npGTO, const double *alph, const double *cofs,// 6,7,8,9,10
                              const int nAtmB,     const double *RnucB, const int *IZnucB,const int nDb,      const int *distinctIZnucB,//11,12,13,14,15
                              const int *NshB,     const int *ellB,     const int *n_cols,const int *npGTOB,  const double *alphB,
                              const double *cofsB, const int spdm,      const double *BravisVectors, const double *Vectorfield,const int nKpoints,
                              const double *kvectors, int Lx,int Ly,int Lz, const int nCGTO_2,
                              const int nCGTO_1){
	dcmplx ****ret=z4alloc(3,nKpoints,nCGTO_2,nCGTO_1);
	int i,j,k,l;
	for(i=0;i<3;i++)for(j=0;j<nKpoints;j++)for(k=0;k<nCGTO_2;k++)for(l=0;l<nCGTO_1;l++)ret[i][j][k][l]=0.1*(i+1) + 0.0001*(j+1)+ _I*( 0.1*(k+1) + 0.001*(l+1));
	return ret[0][0][0];
}
// distinctIZnuc[nDa]
// Nsh[nDa]
//   ell[ \sum( Nsh[:] ) ]  1 per each shell
// npGTO[ \sum( Nsh[:] ) ] 
// alph [ \sum( \npGTO[:] ) ]
// cofs [ \sum( \npGTO[:] ) ]
// BravisVectors[spdm,3] ... 3d vector for each dimension
// kvectors[nKpoints,3]
// Vectorfield[3]        ... A_over_c to be applied as  -i q_e/(hbar c) A.r =  i A_over_c .r
int calc_pbc_overlaps01(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1){
	dcmplx *zbuf= calc_pbc_overlaps02(nAtm,        Rnuc,    IZnuc,            nDa,            distinctIZnuc,
                          Nsh,  ell,  npGTO,      alph, cofs,
                          nAtmB,       RnucB,   IZnucB,           nDb,            distinctIZnucB,
                          NshB, ellB, n_cols,     npGTOB,  alphB,
                          cofsB,    spdm, BravisVectors, Vectorfield,nKpoints,
                          kvectors,       Lx,Ly,Lz,  nCGTO_2, nCGTO_1, 0);
  free(zbuf);zbuf=NULL;
  return 0;
}
//$ 
//$ calc_pbc_overlaps02: library interface, main function : returns zbuf[3,nKpoints,nCGTO_2,nCGTO_1]
//$
dcmplx *calc_pbc_overlaps02(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, int MPIrank){
	dcmplx *z1a=calc_pbc_overlaps02_(nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,npGTO,alph,cofs,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,
	                            NshB,ellB,n_cols,npGTOB, alphB,cofsB,spdm,BravaisVectors, Vectorfield,nKpoints,
	                            kvectors,Lx,Ly,Lz,nCGTO_2,nCGTO_1,MPIrank, 'o');
	return z1a;
}
//$ 
//$ calc_pbc_overlaps02x: library interface, main function : returns zbuf[4,3,nKpoints,nCGTO_2,nCGTO_1]
//$                       returns <\mu_B | x^{i} V(x,y) |\nu_1>   i=0,1,2,3
dcmplx *calc_pbc_overlaps02x(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank){
	return calc_pbc_overlaps02_(nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,npGTO,alph,cofs,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,
	                            NshB,ellB,n_cols,npGTOB, alphB,cofsB,spdm,BravaisVectors, Vectorfield,nKpoints,
	                            kvectors,Lx,Ly,Lz,nCGTO_2,nCGTO_1,MPIrank, 'x');
}
int calc_pbc_overlaps02x_f(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int filenumber){
	dcmplx *zbuf=calc_pbc_overlaps02_(nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,npGTO,alph,cofs,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,
	                            NshB,ellB,n_cols,npGTOB, alphB,cofsB,spdm,BravaisVectors, Vectorfield,nKpoints,
	                            kvectors,Lx,Ly,Lz,nCGTO_2,nCGTO_1,MPIrank, 'x');
	int Ld=4*3*nKpoints*nCGTO_2*nCGTO_1; // 3*nKpoints*nBS2a[0]*nBS1;
	char filename[50];
	sprintf(filename,"calc_pbc_overlapsx_%03d.retf",filenumber);
	write_z1buf(filename,zbuf,Ld);
	z1free(zbuf);
	return Ld;
}
int calc_pbc_overlaps_f(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int filenumber){
	dcmplx *z1a=calc_pbc_overlaps02_(nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,npGTO,alph,cofs,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,
	                            NshB,ellB,n_cols,npGTOB, alphB,cofsB,spdm,BravaisVectors, Vectorfield,nKpoints,
	                            kvectors,Lx,Ly,Lz,nCGTO_2,nCGTO_1,MPIrank, 'o');
	int Ld=3*nKpoints*nCGTO_2*nCGTO_1; // 3*nKpoints*nBS2a[0]*nBS1;
	char filename[50];
	sprintf(filename,"calc_pbc_overlaps_%03d.retf",filenumber);
	write_z1buf(filename,z1a,Ld);
	z1free(z1a);z1a=NULL;
	return Ld;
}
#define _GEN_BS1 1
#define _GEN_BS2 2
#define _CALC_OVLPo 4
#define _WRITE_OVLPo 8
#define _CALC_OVLPx 16
#define _WRITE_OVLPx 32

int calc_pbc_overlaps_f01(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank,const int filenumber){
	int iflag=_GEN_BS1+_GEN_BS2+_CALC_OVLPo+_WRITE_OVLPo;
	calc_pbc_overlaps_dbg_(nAtm, Rnuc, IZnuc, nDa, distinctIZnuc, Nsh, ell, npGTO, alph, cofs, nAtmB, RnucB, IZnucB, nDb, distinctIZnucB,
                        NshB, ellB, n_cols, npGTOB, alphB, cofsB, spdm, BravaisVectors, Vectorfield, nKpoints, kvectors, Lx, Ly, Lz,
                        nCGTO_2, nCGTO_1, MPIrank, iflag, filenumber);
  return 0;
}
int calc_pbc_overlaps02x_f01(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank,const int filenumber){
	int iflag=_GEN_BS1+_GEN_BS2+_CALC_OVLPx+_WRITE_OVLPx;
	calc_pbc_overlaps_dbg_(nAtm, Rnuc, IZnuc, nDa, distinctIZnuc, Nsh, ell, npGTO, alph, cofs, nAtmB, RnucB, IZnucB, nDb, distinctIZnucB,
                        NshB, ellB, n_cols, npGTOB, alphB, cofsB, spdm, BravaisVectors, Vectorfield, nKpoints, kvectors, Lx, Ly, Lz,
                        nCGTO_2, nCGTO_1, MPIrank, iflag, filenumber);
  return 0;
}
// for backward compatibility....
int calc_pbc_overlaps_dbg(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int iflag){
	int filenumber=MPIrank;
	calc_pbc_overlaps_dbg_(nAtm, Rnuc, IZnuc, nDa, distinctIZnuc, Nsh, ell, npGTO, alph, cofs, nAtmB, RnucB, IZnucB, nDb, distinctIZnucB,
                        NshB, ellB, n_cols, npGTOB, alphB, cofsB, spdm, BravaisVectors, Vectorfield, nKpoints, kvectors, Lx, Ly, Lz,
                        nCGTO_2, nCGTO_1, MPIrank, iflag, filenumber);
  return 0;
}
int calc_pbc_overlaps_dbg_(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
       double *kvectors,  int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int iflag,const int filenumber){
	int nBS1=0, nBS2a[3]={0,0,0}, verbose=0;
	int fix_nrmz=1;char *sdum=getenv("calc_pbc_overlaps.fix_nrmz");
	if(sdum!=NULL){ long ldum=0;int iretv=sscanf(sdum,"%ld",&ldum);if(iretv){ fix_nrmz=(int)ldum;} }
	Gaussian **BS=NULL,***BS2=NULL;
	if( (iflag & _GEN_BS1 )!=0 ){ BS =gen_bset01(&nBS1,nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,NULL,npGTO,alph,cofs,0,verbose);
	                              check_nrmz01("BS1",BS,nBS1,( (fix_nrmz&1)==0 ? 0:1 )); }
	if( (iflag & _GEN_BS2 )!=0 ){ BS2=gen_bsetx3(nBS2a,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,NshB,ellB,n_cols,npGTOB,alphB,cofsB);
	                              check_nrmz01("BS2",BS2[0],nBS2a[0], ( (fix_nrmz&2)==0 ? 0:1 ) ); }
  int i2LH,j1RH,I_cols,kp;int iargs[3]={Lx,Ly,Lz};
  int Ldim_margin=5;
	if( (iflag & _CALC_OVLPo )!=0 && (BS!=NULL && BS2!=NULL) ){
		dcmplx **ret=(dcmplx **)malloc(3*sizeof(dcmplx *));
		ret[0]=z1alloc_i( 3*nKpoints*nBS2a[0]*nBS1, 0.0); ret[1]= ret[0] + (nKpoints*nBS2a[0]*nBS1); ret[2]= ret[1] + (nKpoints*nBS2a[0]*nBS1);
		for(I_cols=0;I_cols<3;I_cols++){ if(nBS2a[I_cols]<=0)continue;
			for(i2LH=0;i2LH<nBS2a[I_cols];i2LH++){ for(j1RH=0;j1RH<nBS1;j1RH++){
				dcmplx *ovlps=pbc_overlap(I_cols,i2LH,j1RH, BS2[I_cols][i2LH], BS[j1RH],spdm, BravaisVectors,
				                          Vectorfield, nKpoints, kvectors, iargs, 0, Ldim_margin);
				for(kp=0;kp<nKpoints;kp++) ret[I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]=ovlps[kp];
				z1free(ovlps);
			}}
		}
		if( (iflag & _WRITE_OVLPo)!=0 ){
			int Ld=3*nKpoints*nCGTO_2*nCGTO_1; // 3*nKpoints*nBS2a[0]*nBS1;
			char filename[50];
			sprintf(filename,"calc_pbc_overlaps_%03d.retf",filenumber);
			write_z1buf(filename,ret[0],Ld);
		}
		z2free(ret);
	}
	if( (iflag & _CALC_OVLPx)!=0 && (BS!=NULL && BS2!=NULL)){ const int N_I=4;
		dcmplx ***zbuf=z3alloc_i( N_I,3,(nKpoints*nBS2a[0]*nBS1), 0.0 );
		for(I_cols=0;I_cols<3;I_cols++){
			if(nBS2a[I_cols]<=0)continue;
			for(i2LH=0;i2LH<nBS2a[I_cols];i2LH++){ for(j1RH=0;j1RH<nBS1;j1RH++){ int I;
				dcmplx **ovlps=pbc_overlapx(I_cols,i2LH,j1RH, BS2[I_cols][i2LH], BS[j1RH],spdm, BravaisVectors,Vectorfield, nKpoints, kvectors, iargs, 0, Ldim_margin);
				for(I=0;I<N_I;I++) for(kp=0;kp<nKpoints;kp++) zbuf[I][I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]=ovlps[I][kp];
				z2free(ovlps);
			}}
		}
		if( (iflag & _WRITE_OVLPx)!=0 ){
			int Ld=N_I*3*(nKpoints*nBS2a[0]*nBS1); // 3*nKpoints*nBS2a[0]*nBS1;
			char filename[50];
			sprintf(filename,"calc_pbc_overlapsx_%03d.retf",filenumber);
			write_z1buf(filename,zbuf[0][0],Ld);
		}
		z3free(zbuf);
	}
	if(BS!=NULL){ free_Bset(BS,nBS1);}
	if(BS2!=NULL){ printf("#calc_pbc_overlaps_DBG:$%02d freeing nBS2:%d,%d,%d ...\n",MPIrank_,nBS2a[0],nBS2a[1],nBS2a[2]);fflush(stdout);
		free_Bset(BS2[0],nBS2a[0]);
		free_Bset(BS2[1],nBS2a[1]);
		if(nBS2a[2]>0){ free_Bset(BS2[2],nBS2a[2]);}
	}
	//printf("#calc_pbc_overlaps_DBG:$%02d END\n",MPIrank_);
	return 0;
}
#undef _GEN_BS1
#undef _GEN_BS2
#undef _CALC_OVLPo
#undef _WRITE_OVLPo
#undef _CALC_OVLPx
#undef _WRITE_OVLPx

dcmplx *calc_pbc_overlaps02_(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1,const int MPIrank,const char o_OR_x){
	MPIrank_=MPIrank;
	if(MPIrank_==0){ N_call_=fcountup("xgaussope_c", 1);}
	Ope_=o_OR_x;
	int verbose=1;
	static int append_logf=0;
	if( MPIrank_==0 && append_logf==0 ){ int narg=0;
		char *str=NULL;
		FILE *fptr=fopen("calc_pbc_overlaps_input.log",(append_logf ? "a":"w"));append_logf=1;
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nAtm);
		fprintf(fptr,"#pbc_overlaps01:%f:%f\n",Rnuc[0],Rnuc[1]);
		fprintf(fptr,"#pbc_overlaps01:%d:%d\n",IZnuc[0],IZnuc[1]);
		fprintf(fptr,"#pbc_overlaps01:%02d:Rnuc:%s\n",(narg++), (str=d1toa(Rnuc,3*nAtm)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(IZnuc,nAtm)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nDa);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(distinctIZnuc,nDa)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(Nsh,nDa)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(ell,nDa)));free(str);
		int NshSUM=i1sum(Nsh,nDa);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(npGTO,NshSUM)));free(str);
		int Npgto=i1sum(npGTO,NshSUM);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=d1toa(alph,Npgto)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=d1toa(cofs,Npgto)));free(str);

		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nAtmB);
		fprintf(fptr,"#pbc_overlaps01:%02d:RnucB:%s\n",(narg++), (str=d1toa(RnucB,3*nAtmB)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:IZnucB:%s\n",(narg++), (str=i1toa(IZnucB,nAtmB)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nDb);
		fprintf(fptr,"#pbc_overlaps01:%02d:distinctIZnucB:%s\n",(narg++), (str=i1toa(distinctIZnucB,nDb)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:NshB:%s\n",(narg++), (str=i1toa(NshB,nDb)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:ellB:%s\n",(narg++), (str=i1toa(ellB,nDb)));free(str);
		NshSUM=i1sum(NshB,nDb);
		fprintf(fptr,"#pbc_overlaps01:%02d:npGTOB:%s\n",(narg++), (str=i1toa(npGTOB,NshSUM)));free(str);
		if(n_cols!=NULL){
			fprintf(fptr,"#pbc_overlaps01:%02d:n_cols:%s\n",(narg++), (str=i1toa(n_cols,NshSUM)));free(str);
		}
		Npgto=i1sum(npGTOB,NshSUM);
		fprintf(fptr,"#pbc_overlaps01:%02d:alphB:%s\n",(narg++), (str=d1toa(alphB,Npgto)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:cofsB:%s\n",(narg++), (str=d1toa(cofsB,Npgto)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:spdm:%d\n",(narg++), spdm);
		fprintf(fptr,"#pbc_overlaps01:%02d:BravisVectors:%s\n",(narg++), (str=d1toa(BravisVectors,3*spdm)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:Vectorfield:%s\n",(narg++), (str=d1toa(Vectorfield,3)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:kvectors:%s\n",(narg++), (str=d1toa(kvectors,3*spdm)));free(str);
		fprintf(fptr,"#pbc_overlaps01:%02d:spdm:%d,%d %d %d %d\n",(narg++), Lx,Ly,Lz,nCGTO_2,nCGTO_1);
		fclose(fptr);
	}
	dcmplx *ret=NULL;
	if( o_OR_x == 'o'){
		dcmplx **zbuf=calc_pbc_overlaps(nAtm, Rnuc,    IZnuc,   nDa,  distinctIZnuc,
                    Nsh, ell, npGTO, alph, cofs,          nAtmB, RnucB, IZnucB, nDb,  distinctIZnucB,
                    NshB, ellB, n_cols, npGTOB, alphB,    cofsB, spdm, BravisVectors, Vectorfield,nKpoints,
                    kvectors,       Lx,Ly,Lz,  nCGTO_2, nCGTO_1);
     ret=zbuf[0]; free(zbuf);
  } else { __assertf( o_OR_x=='x',(""),-1);
		ret=calc_pbc_overlapsx(nAtm, Rnuc,    IZnuc,   nDa,  distinctIZnuc,
                    Nsh,  ell,  npGTO,  alph, cofs,      nAtmB, RnucB, IZnucB, nDb, distinctIZnucB,
                    NshB, ellB, n_cols, npGTOB, alphB,    cofsB, spdm, BravisVectors, Vectorfield,nKpoints,
                    kvectors,       Lx,Ly,Lz,  nCGTO_2, nCGTO_1);
	}
  //printf("#clib:calc_pbc_overlaps01 END...\n");fflush(stdout);
  return ret;
}

double *calc_pbc_overlaps03(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1){
	//printf("#calc_pbc_overlaps:\n");fflush(stdout);
	int verbose=0;
	int nBS1=0,nBS2a[3]={0,0,0};
	Gaussian **BS =gen_bset01(&nBS1,nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,NULL,npGTO,alph,cofs,0,verbose);
  Gaussian ***BS2=gen_bsetx3(nBS2a,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,NshB,ellB,n_cols,npGTOB,alphB,cofsB);
	printf("#calc_pbc_overlaps:\n");fflush(stdout);
	free_Bset(BS,nBS1);
	printf("#calc_pbc_overlaps:END.010:\n");fflush(stdout);
	
	free_Bset(BS2[0],nBS2a[0]);
	printf("#calc_pbc_overlaps:END.020:\n");fflush(stdout);
	free_Bset(BS2[1],nBS2a[1]);
	printf("#calc_pbc_overlaps:END.030\n");fflush(stdout);
	//return ret[0][0][0];
	double *ret=d1alloc( 3*nKpoints*nBS2a[0]*nBS1*2);
	int i,kp,j2,k1,ijk,ri;
	for(i=0,ijk=0;i<3;i++){ for(kp=0;kp<nKpoints;kp++){ for(j2=0;j2<nBS2a[0];j2++){ for(k1=0;k1<nBS1;k1++){ for(ri=0;ri<2;ri++,ijk++){
		ret[ijk]=(100*(i+1) + 1*(kp+1) + 0.01*(j2+1) + 0.0001*(k1+1))*(1-2*(ri%2));
	}}}}}
	return ret;
}
dcmplx **calc_pbc_overlaps(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1){
  //da stands for distinct atom
  //test_ovlp();
  //__assertf(0,("test_ovlp"),-1);
  static int nCall_loc_=0;nCall_loc_++;
  int fix_nrmz=1; // 2^{1}: fix nrmz of BS2  2^{0} fix nrmz of BS1
  char sbuf10[10];
  char *sdum=getenv("calc_pbc_overlaps.fix_nrmz");
  if(sdum!=NULL){ long ldum=0;int iretv=sscanf(sdum,"%ld",&ldum);if(iretv){ fix_nrmz=(int)ldum;} }
  //printf("#calc_pbc_overlaps:entering:%d %d %d fix_nrmz=%d",ellB[0],ellB[1],ellB[2],fix_nrmz);fflush(stdout);
  int nBS1=0;int verbose=0;
  int iargs[3]={Lx,Ly,Lz};
//printf("#calc_pbc_overlaps:gen_bset01:nAtm:%d IZnuc:%d\n",nAtm,IZnuc[0]);fflush(stdout);
  Gaussian **BS =gen_bset01(&nBS1,nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,NULL,npGTO,alph,cofs,0,verbose);
  //printf("#gen_BS1");fflush(stdout);
  check_nrmz01("BS1",BS,nBS1,( (fix_nrmz&1)==0 ? 0:1 ));
  // printf("#check_Nrmz.1");fflush(stdout);
  // printf("#gen_bset_2...");fflush(stdout);
  int nBS2a[3]={0,0,0};
//printf("#calc_pbc_overlaps:gen_bsetx3:nAtm:%d Nsh:%d,%d IZnuc:%d\n",nAtmB,NshB[0],NshB[1],IZnucB[0]);fflush(stdout);
  Gaussian ***BS2=gen_bsetx3(nBS2a,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,NshB,ellB,n_cols,npGTOB,alphB,cofsB);
  // printf("#gen_BS2");fflush(stdout);
  check_nrmz01("BS2",BS2[0],nBS2a[0], ( (fix_nrmz&2)==0 ? 0:1 ) );
  __assertf( (nCGTO_2==nBS2a[0]) && (nCGTO_1==nBS1),("%d,%d / %d,%d",nBS2a[0],nBS1,nCGTO_2,nCGTO_1),-1);
  //__assertf(0,("calc_pbc_overlaps check_nrmz"),-1);
  // printf("#calc_pbc_overlaps:295");fflush(stdout);
  int i2LH,j1RH;
  char dbgng=1;
  int Ldim_margin=5;
  //dcmplx ***ret[3]={ z3alloc(nKpoints,nBS2,nBS1), z3alloc(nKpoints,nBS2,nBS1), z3alloc(nKpoints,nBS2,nBS1) }; 
  //dcmplx ****ret=z4alloc(3,nKpoints,nBS2a[0],nBS1); //[N_cols][nKpoints][nBS2a[0]][nBS1]
  dcmplx **ret=(dcmplx **)malloc(3*sizeof(dcmplx *));
  //printf("#clib:z1alloc:3*%d*%d*%d:%d\n",nKpoints,nBS2a[0],nBS1, 3*nKpoints*nBS2a[0]*nBS1);
#ifdef ZALLOC_ADD_MARGIN
	ret[0]=z1alloc_i( 3*nKpoints*nBS2a[0]*nBS1*2, 0.0);
	if( MPIrank_==0 && (nCall_loc_<2) ) printf("#ZALLOC_ADD_MARGIN\n");
#else
  ret[0]=z1alloc_i( 3*nKpoints*nBS2a[0]*nBS1, 0.0); // 2021.0519 fixed :: *sizeof(dcmplx);
#endif
  ret[1]= ret[0] + (nKpoints*nBS2a[0]*nBS1);
  ret[2]= ret[1] + (nKpoints*nBS2a[0]*nBS1);
  double clock00=__ctime; double clock01=clock00, clock02=clock00;
  int n=0; int append_1=0;
  int I_cols; char prtout1=0, prtout2=0;
  static int n_logf_=0;n_logf_++;
  for(I_cols=0;I_cols<3;I_cols++){
		if(nBS2a[I_cols]<=0)continue;
		FILE *fp1=NULL;char fnme[100];
		if(prtout1){ sprintf(fnme,"calc_pbc_overlaps02_%02d_I%d.dat",MPIrank_,I_cols); fp1=fopen(fnme,(n_logf_==1 ? "w":"a")); }
		int ijk=0;
		// printf("#calc_pbc_overlaps:Cols:%d %d\n",I_cols,nBS2a[I_cols]);fflush(stdout);
		double wct_00=__ctime;
	  for(i2LH=0;i2LH<nBS2a[I_cols];i2LH++){
			/*if(I_cols){
				Gaussian *G2=BS2[I_cols][i2LH];double zerovec[3]={0,0,0};
				FILE *fpx=fopen("calc_pbc_overlaps.log",(append_1 ? "a":"w"));append_1=1;
				fprintf(fpx,"%d.%d  %s  %d,%d,%d:nrmz=%f ",I_cols,i2LH,G2->description, G2->pows[0][0],G2->pows[0][1],G2->pows[0][2],
				                                           Gaussian_overlap(G2,G2,zerovec,zerovec));
				fclose(fpx);
			}*/
	    for(j1RH=0;j1RH<nBS1;j1RH++){
				//printf("#calc_pbc_overlaps:303:%d.%d.%d",I_cols,i2LH,j1RH);fflush(stdout);
	  		// if(i2LH%20==0 && j1RH%20==0 ){ printf("#calc_pbc_overlaps:303:%d.%d.%d",I_cols,i2LH,j1RH);fflush(stdout);}
	  		// XXX 20210530:if(i2LH%20==0 && j1RH%20==0 ){ printf("#calc_pbc_overlaps:357:%d.%d.%d",I_cols,i2LH,j1RH);fflush(stdout);}
	  		if( i2LH==1 && j1RH==0 ){ double wct1=(__ctime)-wct_00; printf("#calc_pbc_overlaps:357:%d %d/%d %d:%f",I_cols,i2LH,nBS2a[I_cols],nBS1,wct1);fflush(stdout);}
				//printf("#calc_pbc_overlaps:%d,%d:",i2LH,j1RH);
				//Gaussian_prtout(("LHS"),BS[i2LH]); Gaussian_prtout(("RHS"),BS[j1RH]);
				dcmplx *ovlps=pbc_overlap(I_cols,i2LH,j1RH, BS2[I_cols][i2LH], BS[j1RH],spdm, BravisVectors,
				                          Vectorfield, nKpoints, kvectors, iargs, 0, Ldim_margin); // 0:clear buffer
				clock02=clock01;clock01=__ctime;n++; 
				//if(n==1 || n==10||clock01-clock02>10.0 ) printf("pbc_overlap:LH=%d:I=%d RH=%d elapsed step=%f avg=%f total=%f\n",i2LH,I_cols,j1RH,clock01-clock02,(clock01-clock00)/n,(clock01-clock00));
				int kp;
				if(prtout1){ fprintf(fp1,"%d %d   ",i2LH,j1RH);
										for(kp=0;kp<nKpoints;kp++) fprintf(fp1,"%14.6f %14.6f         ",creal(ovlps[kp]),cimag(ovlps[kp]));
										fprintf(fp1,"\n"); 
				}
				for(kp=0;kp<nKpoints;kp++) ret[I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]=ovlps[kp];
				// for(kp=0;kp<nKpoints;kp++) ret[I_cols][kp][i2LH][j1RH]=ovlps[kp];
				z1free(ovlps);
				//printf("#calc_pbc_overlaps %d %d / %d %d\n",iLHS,jRHS,nBS2a[0],nBS1);fflush(stdout);
		}}
		if(prtout1) fclose(fp1);
		pbc_overlap(-1,-1,-1, NULL, NULL,-1, NULL, NULL,-1,NULL,NULL,-1, -1); // clear buffer

		if(prtout2){
			FILE *fp=fopen("calc_pbc_overlaps01.dat",(I_cols==0 ? "w":"a"));
			fprintf(fp,"#%d %d %d\n",nKpoints,nBS2a[I_cols],nBS1);
			for(i2LH=0;i2LH<nBS2a[I_cols];i2LH++){
				for(j1RH=0;j1RH<nBS1;j1RH++){ int kp;
					for(kp=0;kp<nKpoints;kp++){
						fprintf(fp,"%18.10f %18.10f ",creal(ret[I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]),
							                              cimag(ret[I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]));
			}}}
			fprintf(fp,"\n");fclose(fp);
		}
	}
	//printf("#calc_pbc_overlaps.327:END\n");fflush(stdout);
	
	if(0){
		FILE *fp=fopen("calc_pbc_overlaps.dat","w");
		fprintf(fp,"%d %d %d\n",nKpoints,nBS2a[0],nBS1);
		int kp;
		for(I_cols=0;I_cols<3;I_cols++){
			printf("#printing out I_cols=%d N=%d %p\n",I_cols,nBS2a[I_cols],ret[I_cols]);
			if(nBS2a[I_cols]<=0)continue;
			
		  for(kp=0;kp<nKpoints;kp++){
		  	for(i2LH=0;i2LH<nBS2a[I_cols];i2LH++){
		    	for(j1RH=0;j1RH<nBS1;j1RH++){
						fprintf(fp,"%18.10f %18.10f ",creal(ret[I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]),
						                              cimag(ret[I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]));
//						fprintf(fp,"%18.10f %18.10f ",creal(ret[I_cols][kp][i2LH][j1RH]),cimag(ret[I_cols][kp][i2LH][j1RH]));
					} fprintf(fp,"\n");
			}}
		}
		fclose(fp);
		printf("#calc_pbc_overlaps.343:END\n");fflush(stdout);
	}
	//printf("#calc_pbc_overlaps:END.000:%p\n",ret);fflush(stdout);
	free_Bset(BS,nBS1);
	//printf("#calc_pbc_overlaps:END.010:%p\n",ret);fflush(stdout);
	
	free_Bset(BS2[0],nBS2a[0]);BS2[0]=NULL;
	//printf("#calc_pbc_overlaps:END.020:%p\n",ret);fflush(stdout);
	free_Bset(BS2[1],nBS2a[1]);BS2[1]=NULL;
	if( nBS2a[2]>0 && BS2[2]!=NULL ){
		printf("#calc_pbc_overlaps:END.030:%p\n",ret);fflush(stdout);
		free_Bset(BS2[2],nBS2a[2]);BS2[2]=NULL;
	}
	//printf("#calc_pbc_overlaps:END.030:%p\n",ret);fflush(stdout);
	//return ret[0][0][0];
	return ret;
}

int fill_zbuf(dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin){
	int hj[3]={1,1,1};double ermax=0;
	return fill_zbuf1(hj,p_zbuf,p_Hj,abs_thr,lhs,rhs,spdm,BravisVectors,Vectorfield,margin,&ermax);
}

//  p_Hj (INOUT) defines buffer size and offset :  zbuf[ 2Hj[0]+1 ][ ... ]
//                and istep[0] displ is stored in the manner zbuf[ istep[0] + Hj[0] ][ ... ]
//             it can change when realloc occurs in this subroutine
//  hj (INOUT) defines the range you actually filled the buffer ...
//             (INPUT)  we first fill [ -hj[0]:hj[0] ][ ... ] and check convergence
//  returns cvgd (1 or 0)
/*
int fill_zbuf1b(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax){
	return fill_zbuf1_(hj,p_zbuf,p_Hj,abs_thr,lhs,rhs,spdm,BravisVectors,Vectorfield,margin,ermax, Gaussian_ovlpNew_B);
}*/
int fill_zbuf1(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax){
	return fill_zbuf1_(hj,p_zbuf,p_Hj,abs_thr,lhs,rhs,spdm,BravisVectors,Vectorfield,margin,ermax, Gaussian_ovlpNew);
}
int fill_zbuf1_(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr,
                Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax,
                dcmplx (*fnGaussian_ovlpNew)(Gaussian *,Gaussian *,const double *,const double *) ){
	double *ax=BravisVectors,*ay=BravisVectors+3,*az=BravisVectors+6;
	*ermax=-1.0;
	int *Hj_ini=p_Hj[0];
	static int hj_0[3]={1,1,1};
	static int hj_step[3]={1,1,1};
	__assertf(hj[0]>=1 && hj[1]>=0 && hj[2]>=1,("wrong ini hj 001"),-1);
	__assertf(hj[0]<=Hj_ini[0] && hj[1]<=Hj_ini[1] && hj[2]<=Hj_ini[2],("wrong ini hj 002"),-1);
	// hj[0]=__MIN(Hj_ini[0],hj_0[0]);hj[1]=__MIN(Hj_ini[1],hj_0[1]);hj[2]=__MIN(Hj_ini[2],hj_0[2]);
	int cvgd=0;
	int Ia[3]={0,0,0};
	int nloop=0;int hjLAST[3]={-1,-1,-1};
	int n_extra_loop=0;int N_cum=0;
	while( (nloop++)<999 ){ int kk;
		dcmplx ***zbuf=p_zbuf[0];
		int *Hj=p_Hj[0];
		// printf("#fill_zbuf:loop%d %d %d %d",nloop,hj[0],hj[1],hj[2]);fflush(stdout);
		double abs_max=-1;int maxat[3]={0,0,0};int N_eff=0;
		//printf("#loop:%d %d,%d,%d / %d,%d,%d  prev:%d,%d,%d\n",nloop,hj[0],hj[1],hj[2], Hj[0],Hj[1],Hj[2], hjLAST[0],hjLAST[1],hjLAST[2]);
		for(Ia[0]=-hj[0];Ia[0]<=hj[0];Ia[0]++){   int o0=( abs(Ia[0])<=hjLAST[0]);
			for(Ia[1]=-hj[1];Ia[1]<=hj[1];Ia[1]++){  int o1=(o0 && abs(Ia[1])<=hjLAST[1]);
				for(Ia[2]=-hj[2];Ia[2]<=hj[2];Ia[2]++){ int o2=(o1 && abs(Ia[2])<=hjLAST[2]);
					if(o2) { 
										continue;}
					N_eff++;
					double displ[3]={ ax[0]*Ia[0] + ay[0]*Ia[1] + az[0]*Ia[2],
														ax[1]*Ia[0] + ay[1]*Ia[1] + az[1]*Ia[2], ax[2]*Ia[0] + ay[2]*Ia[1] + az[2]*Ia[2]};
					dcmplx cdmy=zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ];
					dcmplx ovlp=fnGaussian_ovlpNew(lhs,rhs,Vectorfield,displ);
					
					zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]=ovlp;
					double dum=cabs(ovlp);
					if( dum>abs_max ){ abs_max=dum; for(kk=0;kk<3;kk++) maxat[kk]=Ia[kk];}
		}}}
		int prod=(2*hj[0]+1)*(2*hj[1]+1)*(2*hj[2]+1);
		N_cum+=N_eff;__assertf( (N_cum==prod),("N_cum:%d/%d",N_cum,prod),-1);
		for(kk=0;kk<3;kk++) hjLAST[kk]=hj[kk];
		
		// (ia) expands outward if any of f(x) at boundary is larger than abs_thr ...
		int ninc=0;char iInc[3]={0,0,0};int I;
		for(I=0;I<3;I++){
			int J=(I+1)%3,K=(I+2)%3;
			Ia[J]=0;Ia[K]=0;
			Ia[I]=-hj[I]; double dum1=cabs(zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]);
			Ia[I]=hj[I];  double dum2=cabs(zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]);
			if( dum1>=abs_thr || dum2>=abs_thr ){ hj[I]+=hj_step[I]; ninc+=1; iInc[I]=1;}
		}
		*ermax=abs_max;
		// (ib) any of f(x) in the final shell exceeds abs_thr expand in all direction by 1 ...
		// (ii) if ( margin != NULL )  expands in each direction by margin[:] 
		if( ninc == 0 ){
			if(abs_max>abs_thr){
				hj[0]+=1;hj[1]+=1;hj[2]+=1;
			} else {
				if( (margin != NULL)  && (n_extra_loop==0) ){
					hj[0]+=margin[0];hj[1]+=margin[1];hj[2]+=margin[2];n_extra_loop++;
				} else {
					cvgd=1; break;
				}
			}
		} 
		//printf("#fill_zbuf:loop%d %d %d %d DONE  ninc:%d absmax:%e/thr:%e\n",nloop,hj[0],hj[1],hj[2],ninc,abs_max,abs_thr);fflush(stdout);
		int Noverflow=0,ioverflow[3]={0,0,0};
		for (I=0;I<3;I++) { if(hj[I]>Hj[I]) Noverflow++;ioverflow[I]=1;}
		if( Noverflow==0 ) continue;
		int margin=2;
		int Hj_new[3]={0,0,0};
		for(I=0;I<3;I++){ 
			if(iInc[I]==0){ if(hj[I]+margin<=Hj[I]){ Hj_new[I]=Hj[I];       }
			                else                   { Hj_new[I]=hj[I]+margin;} }
			else { Hj_new[I]=hj[I]+margin;}
		}
		dcmplx ***zbuf_new=z3alloc_i( 2*Hj_new[0]+1, 2*Hj_new[1]+1, 2*Hj_new[2]+1, 0.0 );
		for(Ia[0]=-Hj[0];Ia[0]<=Hj[0];Ia[0]++){
			for(Ia[1]=-Hj[1];Ia[1]<=Hj[1];Ia[1]++){
				for(Ia[2]=-Hj[2];Ia[2]<=Hj[2];Ia[2]++){
					dcmplx cdum=zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ];
					zbuf_new[ Ia[0]+Hj_new[0] ][ Ia[1]+Hj_new[1] ][ Ia[2]+Hj_new[2] ] = cdum;
		}}}
		for(kk=0;kk<3;kk++) p_Hj[0][kk]=Hj_new[kk];
		p_zbuf[0]=zbuf_new;
		//printf("#zbuf switches from %p %p  to %p %p Hj %d,%d,%d\n",zbuf,zbuf[0][0],
		//				p_zbuf[0],p_zbuf[0][0][0], p_Hj[0][0], p_Hj[0][1], p_Hj[0][2]);
		z3free(zbuf);zbuf=NULL;
		continue;
	}
	return cvgd;
}

int fill_zbuf1x(int *hj,dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              const int ixyz,Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin,double *ermax,
              dcmplx ***zbuf_REFR,const int *Hj_REFR,const int *hj_REFR){
	double *ax=BravisVectors,*ay=BravisVectors+3,*az=BravisVectors+6;
	*ermax=-1.0;
	int *Hj_ini=p_Hj[0];
	static int hj_0[3]={1,1,1};
	static int hj_step[3]={1,1,1};
	__assertf(hj[0]>=1 && hj[1]>=1 && hj[2]>=1,("wrong ini hj 001"),-1);
	__assertf(hj[0]<=Hj_ini[0] && hj[1]<=Hj_ini[1] && hj[2]<=Hj_ini[2],("wrong ini hj 002"),-1);
	// hj[0]=__MIN(Hj_ini[0],hj_0[0]);hj[1]=__MIN(Hj_ini[1],hj_0[1]);hj[2]=__MIN(Hj_ini[2],hj_0[2]);
	int cvgd=0;
	int Ia[3]={0,0,0};
	int nloop=0;int hjLAST[3]={-1,-1,-1};
	int n_extra_loop=0;int N_cum=0;
	while( (nloop++)<999 ){ int kk;
		dcmplx ***zbuf=p_zbuf[0];
		int *Hj=p_Hj[0];
		// printf("#fill_zbuf:loop%d %d %d %d",nloop,hj[0],hj[1],hj[2]);fflush(stdout);
		double abs_max=-1;int maxat[3]={0,0,0};int N_eff=0;
		int in_REFR=0; if( hj[0]<=hj_REFR[0] && hj[1]<=hj_REFR[1] && hj[2]<=hj_REFR[2] ) in_REFR=1;
		//printf("#loop:%d %d,%d,%d / %d,%d,%d  prev:%d,%d,%d\n",nloop,hj[0],hj[1],hj[2], Hj[0],Hj[1],Hj[2], hjLAST[0],hjLAST[1],hjLAST[2]);
		for(Ia[0]=-hj[0];Ia[0]<=hj[0];Ia[0]++){   int o0=( abs(Ia[0])<=hjLAST[0]);
			for(Ia[1]=-hj[1];Ia[1]<=hj[1];Ia[1]++){  int o1=(o0 && abs(Ia[1])<=hjLAST[1]);
				for(Ia[2]=-hj[2];Ia[2]<=hj[2];Ia[2]++){ int o2=(o1 && abs(Ia[2])<=hjLAST[2]);
					if(o2) { 
										continue;}
					N_eff++;
					double displ[3]={ ax[0]*Ia[0] + ay[0]*Ia[1] + az[0]*Ia[2],
														ax[1]*Ia[0] + ay[1]*Ia[1] + az[1]*Ia[2], ax[2]*Ia[0] + ay[2]*Ia[1] + az[2]*Ia[2]};
					//dcmplx cdmy=zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ];
					dcmplx ovlp00;
					if( in_REFR || ( abs(Ia[0])<=hj_REFR[0] && abs(Ia[1])<=hj_REFR[1] && abs(Ia[2])<=hj_REFR[2] ) ){
						ovlp00 = zbuf_REFR[ Ia[0]+Hj_REFR[0] ][ Ia[1]+Hj_REFR[1] ][ Ia[2]+Hj_REFR[2] ];
					} else { ovlp00 = Gaussian_ovlpNew(lhs,rhs,Vectorfield,displ);}
					
					dcmplx ovlp=Gaussian_ovlpNewX(lhs,rhs,ixyz,ovlp00,Vectorfield,displ);
					
					zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]=ovlp;
					double dum=cabs(ovlp);
					if( dum>abs_max ){ abs_max=dum; for(kk=0;kk<3;kk++) maxat[kk]=Ia[kk];}
		}}}
		int prod=(2*hj[0]+1)*(2*hj[1]+1)*(2*hj[2]+1);
		N_cum+=N_eff;__assertf( (N_cum==prod),("N_cum:%d/%d",N_cum,prod),-1);
		for(kk=0;kk<3;kk++) hjLAST[kk]=hj[kk];
		
		// (ia) expands outward if any of f(x) at boundary is larger than abs_thr ...
		int ninc=0;char iInc[3]={0,0,0};int I;
		for(I=0;I<3;I++){
			int J=(I+1)%3,K=(I+2)%3;
			Ia[J]=0;Ia[K]=0;
			Ia[I]=-hj[I]; double dum1=cabs(zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]);
			Ia[I]=hj[I];  double dum2=cabs(zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]);
			if( dum1>=abs_thr || dum2>=abs_thr ){ hj[I]+=hj_step[I]; ninc+=1; iInc[I]=1;}
		}
		*ermax=abs_max;
		// (ib) any of f(x) in the final shell exceeds abs_thr expand in all direction by 1 ...
		// (ii) if ( margin != NULL )  expands in each direction by margin[:] 
		if( ninc == 0 ){
			if(abs_max>abs_thr){
				hj[0]+=1;hj[1]+=1;hj[2]+=1;
			} else {
				if( (margin != NULL)  && (n_extra_loop==0) ){
					hj[0]+=margin[0];hj[1]+=margin[1];hj[2]+=margin[2];n_extra_loop++;
				} else {
					cvgd=1; break;
				}
			}
		} 
		//printf("#fill_zbuf:loop%d %d %d %d DONE  ninc:%d absmax:%e/thr:%e\n",nloop,hj[0],hj[1],hj[2],ninc,abs_max,abs_thr);fflush(stdout);
		int Noverflow=0,ioverflow[3]={0,0,0};
		for (I=0;I<3;I++) { if(hj[I]>Hj[I]) Noverflow++;ioverflow[I]=1;}
		if( Noverflow==0 ) continue;
		int margin=2;
		int Hj_new[3]={0,0,0};
		for(I=0;I<3;I++){ 
			if(iInc[I]==0){ if(hj[I]+margin<=Hj[I]){ Hj_new[I]=Hj[I];       }
			                else                   { Hj_new[I]=hj[I]+margin;} }
			else { Hj_new[I]=hj[I]+margin;}
		}
		dcmplx ***zbuf_new=z3alloc_i( 2*Hj_new[0]+1, 2*Hj_new[1]+1, 2*Hj_new[2]+1, 0.0 );
		for(Ia[0]=-Hj[0];Ia[0]<=Hj[0];Ia[0]++){
			for(Ia[1]=-Hj[1];Ia[1]<=Hj[1];Ia[1]++){
				for(Ia[2]=-Hj[2];Ia[2]<=Hj[2];Ia[2]++){
					dcmplx cdum=zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ];
					zbuf_new[ Ia[0]+Hj_new[0] ][ Ia[1]+Hj_new[1] ][ Ia[2]+Hj_new[2] ] = cdum;
		}}}
		for(kk=0;kk<3;kk++) p_Hj[0][kk]=Hj_new[kk];
		p_zbuf[0]=zbuf_new;
		//printf("#zbuf switches from %p %p  to %p %p Hj %d,%d,%d\n",zbuf,zbuf[0][0],
		//				p_zbuf[0],p_zbuf[0][0][0], p_Hj[0][0], p_Hj[0][1], p_Hj[0][2]);
		z3free(zbuf);zbuf=NULL;
		continue;
	}
	return cvgd;
}

//ovlp=Gaussian_ovlpNew(lhs,rhs,Vectorfield,displ);
int fill_zbuf_oldver(dcmplx ****p_zbuf,int **p_Hj,const double abs_thr, 
              Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,const int *margin){
	double *ax=BravisVectors,*ay=BravisVectors+3,*az=BravisVectors+6;
	
	int *Hj_ini=p_Hj[0];
	static int hj_0[3]={1,1,1};
	static int hj_step[3]={1,1,1};
	int hj[3]={__MIN(Hj_ini[0],hj_0[0]), __MIN(Hj_ini[1],hj_0[1]), __MIN(Hj_ini[2],hj_0[2])};
	int cvgd=0;
	int Ia[3]={0,0,0};
	int nloop=0;int hjLAST[3]={-1,-1,-1};
	int n_extra_loop=0;int N_cum=0;
	while( (nloop++)<999 ){ int kk;
		dcmplx ***zbuf=p_zbuf[0];
		int *Hj=p_Hj[0];
		double abs_max=-1;int maxat[3]={0,0,0};int N_eff=0;
		//printf("#loop:%d %d,%d,%d / %d,%d,%d  prev:%d,%d,%d\n",nloop,hj[0],hj[1],hj[2], Hj[0],Hj[1],Hj[2], hjLAST[0],hjLAST[1],hjLAST[2]);
		for(Ia[0]=-hj[0];Ia[0]<=hj[0];Ia[0]++){   int o0=( abs(Ia[0])<=hjLAST[0]);
			for(Ia[1]=-hj[1];Ia[1]<=hj[1];Ia[1]++){  int o1=(o0 && abs(Ia[1])<=hjLAST[1]);
				for(Ia[2]=-hj[2];Ia[2]<=hj[2];Ia[2]++){ int o2=(o1 && abs(Ia[2])<=hjLAST[2]);
					if(o2) { /*printf("#skipping %d,%d,%d:%f+j%f\n",Ia[0],Ia[1],Ia[2],
													creal( zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ] ),
													cimag( zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ] ));*/
									continue;}
					N_eff++;
					double displ[3]={ ax[0]*Ia[0] + ay[0]*Ia[1] + az[0]*Ia[2],
														ax[1]*Ia[0] + ay[1]*Ia[1] + az[1]*Ia[2], ax[2]*Ia[0] + ay[2]*Ia[1] + az[2]*Ia[2]};
					dcmplx cdmy=zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ];
					//printf("#Gaussian_ovlpNew... %d %d %d / %d %d %d\n",Ia[0],Ia[1],Ia[2], Hj[0],Hj[1],Hj[2]);fflush(stdout);
					dcmplx ovlp=Gaussian_ovlpNew(lhs,rhs,Vectorfield,displ);
				//	if( (Ia[0]==0 && Ia[1]==0 && Ia[2]==0) || cabs(ovlp)>1.0e-3  ){
				//		printf("%d,%d,%d displ=%f %f %f  ovlp:%f %f\n",Ia[0],Ia[1],Ia[2],displ[0],displ[1],displ[2],creal(ovlp),cimag(ovlp));
				//	}
					zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]=ovlp;
					double dum=cabs(ovlp);
					if( dum>abs_max ){ abs_max=dum; for(kk=0;kk<3;kk++) maxat[kk]=Ia[kk];}
		}}}
		int prod=(2*hj[0]+1)*(2*hj[1]+1)*(2*hj[2]+1);
		N_cum+=N_eff;__assertf( (N_cum==prod),("N_cum:%d/%d",N_cum,prod),-1);
		//printf("#end loop:%d %d,%d,%d / %d,%d,%d  prev:%d,%d,%d  N_cum=%d absMAX=%e\n",
		//			nloop,hj[0],hj[1],hj[2], Hj[0],Hj[1],Hj[2], hjLAST[0],hjLAST[1],hjLAST[2],N_cum,abs_max);
		for(kk=0;kk<3;kk++) hjLAST[kk]=hj[kk];
		
		// check values at boundary...
		int ninc=0;char iInc[3]={0,0,0};int I;
		for(I=0;I<3;I++){
			int J=(I+1)%3,K=(I+2)%3;
			Ia[J]=0;Ia[K]=0;
			Ia[I]=-hj[I]; double dum1=cabs(zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]);
			Ia[I]=hj[I];  double dum2=cabs(zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ]);
			if( dum1>=abs_thr || dum2>=abs_thr ){ hj[I]+=hj_step[I]; ninc+=1; iInc[I]=1;}
		}
		if( ninc == 0 ){
			if(abs_max>abs_thr){
				// printf("#! abs_max=%e at %d,%d,%d / %d,%d,%d\n",abs_max,maxat[0],maxat[1],maxat[2], hj[0],hj[1],hj[2]);
				hj[0]+=1;hj[1]+=1;hj[2]+=1;
			} else {
				// printf("#Loop converges at %d,%d,%d: abs_max=%e at %d,%d,%d    ",hj[0],hj[1],hj[2],abs_max,maxat[0],maxat[1],maxat[2]);
				if( (margin != NULL)  && (n_extra_loop==0) ){
					hj[0]+=margin[0];hj[1]+=margin[1];hj[2]+=margin[2];n_extra_loop++;
					// printf("....... adding to extra loop:%d,%d,%d\n",hj[0],hj[1],hj[2]);
				} else {
					// if( margin != NULL ) printf("OK n_extra:%d\n",n_extra_loop);
					// else                 printf("   no extra loop\n");
					cvgd=1; break;
				}
			}
		} /*else {
			printf("#Loop expands in xyz:%d,%d,%d\n",iInc[0],iInc[1],iInc[2]);
		}*/
		int Noverflow=0,ioverflow[3]={0,0,0};
		for (I=0;I<3;I++) { if(hj[I]>Hj[I]) Noverflow++;ioverflow[I]=1;}
		if( Noverflow==0 ) continue;
		int margin=2;
		int Hj_new[3]={0,0,0};
		for(I=0;I<3;I++){ 
			if(iInc[I]==0){ if(hj[I]+margin<=Hj[I]){ Hj_new[I]=Hj[I];       }
			                else                   { Hj_new[I]=hj[I]+margin;} }
			else { Hj_new[I]=hj[I]+margin;}
		}
		dcmplx ***zbuf_new=z3alloc_i( 2*Hj_new[0]+1, 2*Hj_new[1]+1, 2*Hj_new[2]+1, 0.0 );
		for(Ia[0]=-Hj[0];Ia[0]<=Hj[0];Ia[0]++){
			for(Ia[1]=-Hj[1];Ia[1]<=Hj[1];Ia[1]++){
				for(Ia[2]=-Hj[2];Ia[2]<=Hj[2];Ia[2]++){
					dcmplx cdum=zbuf[ Hj[0]+Ia[0] ][ Hj[1]+Ia[1] ][ Hj[2]+Ia[2] ];
					zbuf_new[ Ia[0]+Hj_new[0] ][ Ia[1]+Hj_new[1] ][ Ia[2]+Hj_new[2] ] = cdum;
		}}}
		for(kk=0;kk<3;kk++) p_Hj[0][kk]=Hj_new[kk];
		p_zbuf[0]=zbuf_new;
		//printf("#zbuf switches from %p %p  to %p %p Hj %d,%d,%d\n",zbuf,zbuf[0][0],
		//				p_zbuf[0],p_zbuf[0][0][0], p_Hj[0][0], p_Hj[0][1], p_Hj[0][2]);
		z3free(zbuf);zbuf=NULL;
		continue;
	}
	if(0){
		int *Hjf=p_Hj[0];dcmplx ***zbuf=p_zbuf[0];
		int hjp[3]={1,1,1};int Ia[3]={0,0,0};
		for(Ia[0]=-hjp[0];Ia[0]<=hjp[0];Ia[0]++){
			printf("#zbuf:%02d\n",Ia[0]);
			for(Ia[1]=-hjp[1];Ia[1]<=hjp[1];Ia[1]++){
				for(Ia[2]=-hjp[2];Ia[2]<=hjp[2];Ia[2]++){
					dcmplx cdum=zbuf[ Ia[0]+Hjf[0] ][ Ia[1]+Hjf[1] ][ Ia[2]+Hjf[2] ];
					printf("%10.4f+j%10.4f    ",creal(cdum),cimag(cdum));
				}
				printf("\n");
			}
			printf("\n\n");
		}
	}
	return cvgd;
}
#define FPATH_checkFTx "checkFTx_pbcoverlap.dat"
#define FPATH_FTwarningx "checkFTx_pbcoverlap_warning.log"

dcmplx **pbc_overlapx(int Iblock,int Ilhs,int Jrhs,Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,
                   int nKpoints,double *kvectors,
                   int *iargs,const int clear_buffer,const int Ldim_margin){
	double *ax=BravisVectors,*ay=BravisVectors+3,*az=BravisVectors+6;
	const int Idbgng=0;
	const int O_=0,X_=1,Y_=2,Z_=3;int I;
#ifdef STATIC_ZBUFX
	static dcmplx ***p_zbuf[4]={NULL,NULL,NULL,NULL}; static int *p_Hj[4]={NULL,NULL,NULL,NULL};
	if( clear_buffer < 0 ){ 
		for(I=0;I<4;I++){	z3free(p_zbuf[I]);p_zbuf[I]=NULL;i1free(p_Hj[I]);p_Hj[I]=NULL;} return NULL;}
#else
  dcmplx ***p_zbuf[4]={NULL,NULL,NULL,NULL}; int *p_Hj[4]={NULL,NULL,NULL,NULL};
  if( clear_buffer < 0 ) return NULL;
#endif
	static double buf_ermax_[4]={0,0,0,0}, FT_ermax_[4]={0,0,0,0}, walltimeAVG[4]={0,0,0,0};const double FT_errthr=1.0e-6;
	static double FTavgERR[4]={0,0,0,0}; static long FTavgERR_count[4]={0,0,0,0},walltimeAVG_count[4]={0,0,0,0};
	static double FTmaxERR[4]={-1,-1,-1,-1};static int FTmaxERR_Hjx[4]={-1,-1,-1,-1},FTmaxERR_Hjy[4]={-1,-1,-1,-1},FTmaxERR_Hjz[4]={-1,-1,-1,-1};
#define upd_FTmaxERR(Idx,val,hjs)  { if((val)>FTmaxERR[(Idx)]){ FTmaxERR[(Idx)]=val; FTmaxERR_Hjx[(Idx)]=hjs[0]; FTmaxERR_Hjy[(Idx)]=hjs[1]; FTmaxERR_Hjz[(Idx)]=hjs[2];} }

	if( p_zbuf[0]==NULL ){
		const int Hx=3,Hy=3,Hz=3;
		for(I=0;I<4;I++){
			p_Hj[I]=i1alloc(3);	p_Hj[I][0]=Hx; p_Hj[I][1]=Hy; p_Hj[I][2]=Hz;
			p_zbuf[I]=z3alloc_i(2*Hx+1,2*Hy+1,2*Hz+1, 0.0);}
		/*	printf("#pbc_overlap:zbuf allocates:%p %p Hj[%d]:%p:%d,%d,%d\n",p_zbuf[I],p_zbuf[I][0][0],I,p_Hj[I], 
			                                                                p_Hj[I][0],p_Hj[I][1],p_Hj[I][2]);   }*/
	}
	__assertf((p_zbuf[0]!=NULL && p_Hj[0]!=NULL),("pbc_overlap.0000"),-1);
	static int n_call=0; n_call++;
	double abs_thr=1.0e-7; int kp;
	double **kXa=d2alloc(nKpoints,3),*kvec=NULL;
	for(kp=0,kvec=kvectors;kp<nKpoints;kp++,kvec=kvec+3){ // kvector[kp] \cdot ax, ay, az
		kXa[kp][0]= Dot_3d( kvec,ax ); kXa[kp][1]= Dot_3d( kvec,ay ); kXa[kp][2]= Dot_3d( kvec,az );
	}
	dcmplx **retbuf=z2alloc(4,nKpoints);double **FT_errbuf=d2alloc(4,nKpoints);
	double Vabs=sqrt( Vectorfield[0]*Vectorfield[0] + Vectorfield[1]*Vectorfield[1] + Vectorfield[2]*Vectorfield[2] );	
	int *margin=NULL;
	if( Ldim_margin > 0 ) { margin=i1alloc(3); margin[0]=Ldim_margin;margin[1]=Ldim_margin;margin[2]=Ldim_margin;}
	int hj_O[3]={1,1,1};
	for(I=0;I<4;I++){ const int Ixyz=I-1;
	  int hj_1[3]={1,1,1}; // requirement: hj[:]<Hj[:]  we'd better start from hj[:]=1 ...
		const char ctype=(I==0 ? 'O':('x'+(char)(Ixyz)));
		double wct01=__ctime; double ermx=0;
		if(I==0) fill_zbuf1(hj_O, p_zbuf+I,p_Hj+I, abs_thr, lhs, rhs, spdm, BravisVectors, Vectorfield, margin, &ermx);
		else     fill_zbuf1x(hj_1, p_zbuf+I, p_Hj+I, abs_thr, Ixyz, lhs,rhs,spdm, BravisVectors, Vectorfield, margin, &ermx, 
		                     p_zbuf[0], p_Hj[0], hj_O);
		int *hj=(I==0 ? hj_O:hj_1); // here I wanted to keep hj_O in loop I>=1 
		double wct02=__ctime;
		if(I==0 || I==3){
			if(n_call==1 || n_call==20 || n_call == 100 || buf_ermax_[I] < ermx){ buf_ermax_[I]=__MAX(ermx,buf_ermax_[I]);
				_Printf( ("#%06d:pbc_overlap%c: hj=%d,%d,%d  ermx=%e  elapsed:%f buf_ermax_:%e\n",n_call,ctype,hj[0],hj[1],hj[2],ermx,wct02-wct01,buf_ermax_[I]) );}
		}
		
		const double *ax=BravisVectors,*ay=BravisVectors+3,*az=BravisVectors+6;
		for(kp=0;kp<nKpoints;kp++) retbuf[I][kp]=0.0;
		const int *Hj_I=p_Hj[I];
		//printf("#Hj[%d]:%p %d,%d,%d\n",I,Hj_I,Hj_I[0],Hj_I[1],Hj_I[2]);fflush(stdout);
		const int Hx=p_Hj[I][0], Hy=p_Hj[I][1], Hz=p_Hj[I][2];
		double ermax=-1;int kp_ermax=-1;dcmplx values[2]={0.0, 0.0};
		double max_absIM=-1;int kp_maxabsIM=-1;
		for(kp=0;kp<nKpoints;kp++){
			//printf("#kp%02d: Hj[%d]:%p %d,%d,%d\n",kp,I,Hj,Hj[0],Hj[1],Hj[2]);fflush(stdout);
			int Ij[3],kk;
			int hj0[3],nstep,istep;int hjLAST[3]={-1,-1,-1};int hjSTEP[3]={1,1,1};
			if( Ldim_margin == 0 ){ nstep=2;for(kk=0;kk<3;kk++)hj0[kk]=Hj_I[kk]-1;}
			else                  { nstep=Ldim_margin; for(kk=0;kk<3;kk++)hj0[kk]=Hj_I[kk]-Ldim_margin;}
			//printf("#I=%d Ldim_margin=%d hj0 set:%d,%d,%d  / Hj_I:%d,%d,%d\n",I,Ldim_margin,hj0[0],hj0[1],hj0[2],Hj_I[0],Hj_I[1],Hj_I[2]);fflush(stdout);
			int hj[3];for(kk=0;kk<3;kk++)hj[kk]=hj0[kk];
			double Dummy_Value=9.99e+20;
			dcmplx p4rev=Dummy_Value,p3rev=Dummy_Value,p2rev=Dummy_Value,prev=Dummy_Value,cur=Dummy_Value;int Neff=0;
			cur=0.0; // accumulates over n steps ...
			for(istep=0;istep<nstep;istep++){
				p4rev=p3rev,p3rev=p2rev;p2rev=prev;prev=cur;// xxx xxx fixed :cur=0.0;
				for(Ij[0]=-hj[0];Ij[0]<=hj[0];Ij[0]++){  int o0=(abs(Ij[0])<=hjLAST[0]);
					for(Ij[1]=-hj[1];Ij[1]<=hj[1];Ij[1]++){ int o1=( o0 && (abs(Ij[1])<=hjLAST[1]) );
						for(Ij[2]=-hj[2];Ij[2]<=hj[2];Ij[2]++){ int o2=( o1 && (abs(Ij[2])<=hjLAST[2]) );
							if(o2) continue;
							dcmplx ovlp=p_zbuf[I][ Ij[0]+Hj_I[0] ][ Ij[1]+Hj_I[1] ][ Ij[2]+Hj_I[2] ];
							double arg=kXa[kp][0]*Ij[0] + kXa[kp][1]*Ij[1] + kXa[kp][2]*Ij[2];
							dcmplx zfac=cos(arg)+_I*sin(arg);
							dcmplx cdum=zfac*ovlp;
							cur+=cdum; Neff++;
				}}}
				__assertf( Neff==(2*hj[0]+1)*(2*hj[1]+1)*(2*hj[2]+1), ("Neff=%d L981 hj=%d,%d,%d",Neff,hj[0],hj[1],hj[2]), -1);
				for(kk=0;kk<3;kk++) hjLAST[kk]=hj[kk];
				if( istep < nstep-1 ){
					int inc=0;
					for(kk=0;kk<3;kk++){ if(hj[kk]<Hj_I[kk]){ hj[kk]=__MIN( hj[kk]+hjSTEP[kk], Hj_I[kk]);inc+=1;} }
					if(inc) continue; 
					else { __assertf(0,("pbc_overlapx:%d,%d,%d/%d,%d,%d c:%f+j%f p:%f+j%f pp:%f+j%f ...",hj[0],hj[1],hj[2],Hj_I[0],Hj_I[1],Hj_I[2],\
					                         creal(cur),cimag(cur), creal(prev),cimag(prev), creal(p2rev),cimag(p2rev)),1);break;                    }
				} else break;
			}
			retbuf[I][kp]=cur; double err=cabs(cur-prev);FT_errbuf[I][kp]=err;
			upd_FTmaxERR(I,err,hj); FTavgERR[I]+=err*err; FTavgERR_count[I]+=1;
			if( FT_errbuf[I][kp] > FT_errthr ){ 
				if(MPIrank_==0){
					int N_error=fcountup("FTerror_warn__countup.txt", 1);
					if(N_error<50){ _Fwrite( "FTerror_warn.log", 1, 1,fpdmy,
															(fpdmy,"#pbc_overlapx:%02d.%03d.%03d %c [%02d]:ERR=%12.4e  %16.8f+j%16.8f / %16.8f+j%16.8f %d,%d,%d ",
															Iblock,Ilhs,Jrhs,kp,ctype,FT_errbuf[I][kp],creal(cur),cimag(cur),creal(prev),cimag(prev),Hj_I[0],Hj_I[1],Hj_I[2] ) );} }
			}
		}
		double wct03=__ctime; walltimeAVG[I]+=(wct03-wct01);walltimeAVG_count[I]++;
	}
	if( margin != NULL ) {i1free(margin);margin=NULL;}
	//int N_call=fcountup("FTerror__countup.txt", 1);
	if( FTavgERR_count[0] == 1000 || FTavgERR_count[0] == 10000 || FTavgERR_count[0] == 20000 ){
		double avgERR[4],tavg[4];
		for(I=0;I<4;I++) { avgERR[I]=sqrt( FTavgERR[I]/(double) FTavgERR_count[I]); tavg[I]=walltimeAVG[I]/(double) walltimeAVG_count[I];
		_Fwrite( "FTerror.log",1,1,fpdmy,(fpdmy,"#pbc_overlapx:#$%02d:avgERR:%e,%e,%e,%e maxERR:%e (%d,%d,%d) avgTIME:%f %f %f %f LdMARGIN:%d ",\
						 MPIrank_,avgERR[0],avgERR[1],avgERR[2],avgERR[3], FTmaxERR[0],FTmaxERR_Hjx[0],FTmaxERR_Hjy[0],FTmaxERR_Hjz[0],\
						 tavg[0],tavg[1],tavg[2],tavg[3], Ldim_margin) );}
	}
  d2free(kXa);kXa=NULL;d2free(FT_errbuf);FT_errbuf=NULL;
#ifdef STATIC_ZBUFX
	if( clear_buffer ){
		for(I=0;I<4;I++){  z3free(p_zbuf[I]);p_zbuf[I]=NULL; i1free(p_Hj[I]); p_Hj[I]=NULL;  }   }
#else
	for(I=0;I<4;I++){ z3free(p_zbuf[I]); p_zbuf[I]=NULL; i1free(p_Hj[I]); p_Hj[I]=NULL; }
#endif
  return retbuf;
#undef upd_FTmaxERR
}
#define _Dbglogger(fptr1,msg,Append)
#define _Dbglogger_Gaussian(fptr1,msg,Append)
#define _Dbglogger_zbuf(fptr1,msg, hxyz,Append)

int free_buf(void *buf){
	printf("#free_buf:releasing mem:%p %ld..",buf,(long)buf);fflush(stdout);
	free(buf);
	printf("#free_buf:released mem\n");fflush(stdout);
	return 0;
	
}
int free_zbuf(dcmplx *zbuf){
	printf("#free_zbuf:releasing mem:%p %ld..",zbuf,(long)zbuf);fflush(stdout);
	z1free(zbuf);
	printf("#free_zbuf:released mem\n");fflush(stdout);
	return 0;
}
dcmplx *test_alloc(const int N1,const int N2,const int N3){
	dcmplx ***zbuf=z3alloc_i(N1,N2,N3,1.192+_I*1.868);dcmplx *retv=zbuf[0][0];free(zbuf[0]);free(zbuf);
	long laddr=(long)retv;
	printf("#test_alloc:returning:%p(%ld) %d*%d*%d\n",retv,laddr,N1,N2,N3);
	return retv;
}
/*
#define _Dbglogger(fptr1,msg,Append)  {char dbgfnme[50];sprintf(dbgfnme,"pbc_overlap_%02d_%02d_%02d_%04d.log",MPIrank_,Ilhs,Jrhs,n_loop);\
FILE *fptr1=fopen(dbgfnme,( (Append) ? "w":"a"));fprintf msg;fclose(fptr1);}//
#define _Dbglogger_Gaussian(fptr1,msg,Append) {char dbgfnme[50];sprintf(dbgfnme,"pbc_overlap_%02d_%02d_%02d_%04d.log",MPIrank_,Ilhs,Jrhs,n_loop);\
FILE *fptr1=fopen(dbgfnme,( (Append) ? "w":"a"));fprintf msg;Gaussian_fprtout((fptr1,"#LHS:%02d:",Ilhs),fptr1,lhs);\
Gaussian_fprtout((fptr1,"#RHS:%02d:",Jrhs),fptr1,rhs);fclose(fptr1);}//

#define _Dbglogger_zbuf(fptr1,msg, hxyz,Append)    {char dbgfnme[50];sprintf(dbgfnme,"pbc_overlap_%02d_%02d_%02d_%04d.log",MPIrank_,Ilhs,Jrhs,n_loop);\
FILE *fptr1=fopen(dbgfnme,"a");int ii,jj,kk;fprintf msg;\
for(ii=-hxyz[0];ii<=hxyz[0];ii++){ for(jj=-hxyz[1];jj<=hxyz[1];jj++){ for(kk=-hxyz[2];kk<=hxyz[2];kk++){\
fprintf(fptr1,"%12.6f %12.6f          ", creal( p_zbuf[0][ ii+p_Hj[0][0] ][ jj+p_Hj[0][1] ][ kk+p_Hj[0][2] ] ),\
                                         cimag( p_zbuf[0][ ii+p_Hj[0][0] ][ jj+p_Hj[0][1] ][ kk+p_Hj[0][2] ] ) );}}} fclose(fptr1); }//
*/                                       

dcmplx *pbc_overlap(int Iblock,int Ilhs,int Jrhs,Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,
                   int nKpoints,double *kvectors,
                   int *iargs,const int clear_buffer,const int Ldim_margin){
	const int Idbgng=0;
	static int n_loop=0; if(Ilhs==0 && Jrhs==0)n_loop++;char Append_dbglogf=(n_loop==1);
	static int n_call=0; n_call++;
	static dcmplx ***p_zbuf[1]={NULL};
	static int *p_Hj[1]={NULL};
	if( clear_buffer < 0 ){
		z3free(p_zbuf[0]);p_zbuf[0]=NULL; i1free(p_Hj[0]);  p_Hj[0]=NULL;  return NULL; }
	if( p_zbuf[0]==NULL ){
		//int Lx=iargs[0],Ly=iargs[1],Lz=iargs[2];
		//int Hx=Lx/2,Hy=Ly/2,Hz=Lz/2;
		const int Hx=3,Hy=3,Hz=3;
		p_Hj[0]=i1alloc(3);
		p_Hj[0][0]=Hx; p_Hj[0][1]=Hy; p_Hj[0][2]=Hz;
		p_zbuf[0]=z3alloc_i(2*Hx+1,2*Hy+1,2*Hz+1, 0.0);
		_Dbglogger(dbgfp1,(dbgfp1,"#zbuf allocates:%p:%d,%d,%d\n",p_zbuf[0],Hx,Hy,Hz),Append_dbglogf);Append_dbglogf=1;
		printf("#pbc_overlap:zbuf allocates:%p %p Hj:%d,%d,%d\n",p_zbuf[0],p_zbuf[0][0][0],p_Hj[0][0],p_Hj[0][1],p_Hj[0][2]);
	}
	_Dbglogger_Gaussian(dbgfp1,(dbgfp1,""),Append_dbglogf);Append_dbglogf=1;
	__assertf((p_zbuf[0]!=NULL && p_Hj[0]!=NULL),("pbc_overlap.0000"),-1);
	double abs_thr=1.0e-7;
	int *margin=NULL;
	if( Ldim_margin > 0 ) { margin=i1alloc(3); margin[0]=Ldim_margin;margin[1]=Ldim_margin;margin[2]=Ldim_margin;}
	int hj[3]={1,1,1}; double ermax=0;
	static double error_max=-1;
	double wct01=__ctime;
	
/*	if(Ilhs==1 && Jrhs==2){ double displ[3]={0,0,0};
		dcmplx zdum=Gaussian_ovlpNew_verbose(lhs,rhs,Vectorfield,displ);printf("#zovlp:%f %f\n",__RandI(zdum));
		__assertf( 0,("pbc_overlap.0000"),-1);
	}*/
	fill_zbuf1(hj, p_zbuf,p_Hj,abs_thr, lhs, rhs, spdm, BravisVectors, Vectorfield, margin, &ermax);
	_Dbglogger_zbuf(dbgfp1,(dbgfp1,"fill_zbuf1:%d,%d,%d/%d,%d,%d",hj[0],hj[1],hj[2],p_Hj[0][0],p_Hj[0][1],p_Hj[0][2]), hj, Append_dbglogf);Append_dbglogf=1;
	double wct02=__ctime;
	if(n_call==1 || n_call==20 || n_call == 100 || error_max < ermax){ error_max=__MAX(ermax,error_max);
		printf("#%06d:pbc_overlap: hj=%d,%d,%d  ermax=%e  elapsed:%f error_max:%e\n",n_call,hj[0],hj[1],hj[2],ermax,wct01-wct01,error_max);}
	if( margin != NULL ) {i1free(margin);margin=NULL;}
  double *ax=BravisVectors,*ay=BravisVectors+3,*az=BravisVectors+6;
  int kp; 
  double **kXa=d2alloc(nKpoints,3),*kvec=NULL;
  for(kp=0,kvec=kvectors;kp<nKpoints;kp++,kvec=kvec+3){ // kvector[kp] \cdot ax, ay, az
		// k - q_e A/c = k+A/c
  	kXa[kp][0]= Dot_3d( kvec,ax ); kXa[kp][1]= Dot_3d( kvec,ay ); kXa[kp][2]= Dot_3d( kvec,az );
  }
	double Vabs=sqrt( Vectorfield[0]*Vectorfield[0] + Vectorfield[1]*Vectorfield[1] + Vectorfield[2]*Vectorfield[2] );

  dcmplx *ret=z1alloc(nKpoints);double *err=d1alloc(nKpoints);
  for(kp=0;kp<nKpoints;kp++) ret[kp]=0.0;
  
  const int *Hj=p_Hj[0];
  const int Hx=p_Hj[0][0], Hy=p_Hj[0][1], Hz=p_Hj[0][2];
	FILE *fpOUT=NULL; FILE *fp=NULL;
	//FILE *fp=(Idbgng ? (fpOUT=fopen("check_pp_FT.dat","a")):stdout);
#define FPATH_checkFT "checkFT_pbcoverlap.dat"
#define FPATH_FTwarning "checkFT_pbcoverlap_warning.log"
	if(Idbgng) fp=(fpOUT=fopen(FPATH_checkFT,"a"));
	else { int fexist=0; FILE *fp1=fopen(FPATH_checkFT,"r"); if(fp1!=NULL){ fexist=1;fclose(fp1);fp1=NULL;}
				 if(fexist==0) { fp=(fpOUT=fopen(FPATH_checkFT,"w"));} }
	if(fp!=NULL){
		fprintf(fp,"#%02d.%03d.%03d H:%d %d %d\n",Iblock,Ilhs,Jrhs,Hx,Hy,Hz);
		fprintf(fp,"\n\n\n## Vecfield:%10.5f %10.5f %10.5f\n",Vectorfield[0],Vectorfield[1],Vectorfield[2]);}
	double ft_ermax=-1;int kp_ermax=-1;dcmplx values[2]={0.0, 0.0};
	double max_absIM=-1;int kp_maxabsIM=-1;
	for(kp=0;kp<nKpoints;kp++){
		int Ij[3],kk;
		int hj0[3],nstep,istep;int hjLAST[3]={-1,-1,-1};int hjSTEP[3]={1,1,1};
		if( Ldim_margin == 0 ){ nstep=2;for(kk=0;kk<3;kk++)hj0[kk]=Hj[kk]-1;}
		else                  { nstep=Ldim_margin; for(kk=0;kk<3;kk++)hj0[kk]=Hj[kk]-Ldim_margin;}
		int hj[3];for(kk=0;kk<3;kk++)hj[kk]=hj0[kk];
		double Dummy_Value=9.99e+20;
		dcmplx p4rev=Dummy_Value,p3rev=Dummy_Value,p2rev=Dummy_Value,prev=Dummy_Value,cur=Dummy_Value;int Neff=0;
		cur=0.0; // accumulates over n steps ...
		for(istep=0;istep<nstep;istep++){
			p4rev=p3rev,p3rev=p2rev;p2rev=prev;prev=cur;// xxx xxx fixed :cur=0.0;
			for(Ij[0]=-hj[0];Ij[0]<=hj[0];Ij[0]++){  int o0=(abs(Ij[0])<=hjLAST[0]);
				for(Ij[1]=-hj[1];Ij[1]<=hj[1];Ij[1]++){ int o1=( o0 && (abs(Ij[1])<=hjLAST[1]) );
					for(Ij[2]=-hj[2];Ij[2]<=hj[2];Ij[2]++){ int o2=( o1 && (abs(Ij[2])<=hjLAST[2]) );
						if(o2) continue;
						dcmplx ovlp=p_zbuf[0][ Ij[0]+Hj[0] ][ Ij[1]+Hj[1] ][ Ij[2]+Hj[2] ];
						double arg=kXa[kp][0]*Ij[0] + kXa[kp][1]*Ij[1] + kXa[kp][2]*Ij[2];
						dcmplx zfac=cos(arg)+_I*sin(arg);
						dcmplx cdum=zfac*ovlp;
						cur+=cdum; Neff++;
			}}}
			//fprintf(fp,"#Neff=%d << %d,%d,%d %d\n",Neff,hj[0],hj[1],hj[2],(2*hj[0]+1)*(2*hj[1]+1)*(2*hj[2]+1));
			__assertf( Neff==(2*hj[0]+1)*(2*hj[1]+1)*(2*hj[2]+1), ("Neff=%d assertion1094: %d,%d,%d",Neff,hj[0],hj[1],hj[2]), -1);
			for(kk=0;kk<3;kk++) hjLAST[kk]=hj[kk];
			if( istep < nstep-1 ){
				for(kk=0;kk<3;kk++) hj[kk]+=hjSTEP[kk];
				__assertf( hj[0]<=Hj[0] && hj[1]<=Hj[1] && hj[2]<=Hj[2],("hj exceeds Hj:%d,%d,%d / %d,%d,%d\n",hj[0],hj[1],hj[2],Hj[0],Hj[1],Hj[2]),-1);
				continue;
			} else break;
		}
		ret[kp]=cur;err[kp]=cabs(cur-prev);
		
		if(fp!=NULL){
			fprintf(fp,"%02d.%03d.%03d.%02d: %14.8f %14.8f %dx%dx%d    \t prev:%12.6f %12.6f (%12.4e)",
							Iblock,Ilhs,Jrhs,kp,creal(cur),cimag(cur),Hj[0],Hj[1],Hj[2],creal(prev),cimag(prev),cabs(cur-prev));
			if(nstep>2) fprintf(fp,"    \t p2rev:%12.6f %12.6f (%12.4e)",creal(p2rev),cimag(p2rev),cabs(cur-p2rev));
			if(nstep>3) fprintf(fp,"    \t p2rev:%12.6f %12.6f (%12.4e)",creal(p3rev),cimag(p3rev),cabs(cur-p3rev));
			if(nstep>4) fprintf(fp,"    \t p2rev:%12.6f %12.6f (%12.4e)",creal(p4rev),cimag(p4rev),cabs(cur-p4rev));
			fprintf(fp,"\n");
		}
		if(ft_ermax<err[kp]){ ft_ermax=err[kp];kp_ermax=kp;values[0]=ret[kp];values[1]=prev;}
		if( err[kp] > 1.0e-6 ){ fprintf((fp==NULL ? stdout:fp),
															"#!W pbc_overlap:%02d.%03d.%03d [%02d]:ERR=%12.4e  %16.8f+j%16.8f / %16.8f+j%16.8f %d,%d,%d\n",
															Iblock,Ilhs,Jrhs,kp,err[kp],creal(cur),cimag(cur),creal(prev),cimag(prev),Hj[0],Hj[1],Hj[2] );}
		double dum=fabs(cimag(cur));if(dum>max_absIM){ max_absIM=dum;kp_maxabsIM=kp;}
	}
	if(fp!=NULL) fprintf(fp,"### pbc_overlap:%02d.%03d.%03d estimated ermax:%12.4e  @[%02d]  %16.8f+j%16.8f / %16.8f+j%16.8f %d,%d,%d\n",
															Iblock,Ilhs,Jrhs,err[kp],kp,creal(values[0]),cimag(values[0]),creal(values[1]),cimag(values[1]),Hx,Hy,Hz);
  if(fp!=NULL) fprintf(fp,"### MAX abs IM:%14.6f as [%d] %14.6f %14.6f\n",max_absIM,kp_maxabsIM,creal(ret[kp_maxabsIM]),cimag(ret[kp_maxabsIM]));
  if( ft_ermax > 1.0e-6 ){ FILE *fpA[2]={NULL,NULL};fpA[0]=(fp==NULL ? stdout:fp);fpA[1]=fopen(FPATH_FTwarning,"a");int kk;
												for(kk=0;kk<2;kk++){  fprintf(fpA[kk],
															"#!W pbc_overlap:%02d.%03d.%03d ermax [%02d]:ERR=%12.4e  cur: %16.8f+j%16.8f / prev_step: %16.8f+j%16.8f Hj:%d,%d,%d\n",
															Iblock,Ilhs,Jrhs,kp_ermax,err[kp_ermax],creal(values[0]),cimag(values[0]),creal(values[1]),cimag(values[1]),Hj[0],Hj[1],Hj[2]);}
												fclose(fpA[1]); }
  if( fpOUT!=NULL )fclose(fpOUT);
  //printf("#pbc_overlap:%f+j%f\n",creal(ret[0]),cimag(ret[0]));fflush(stdout);
  d2free(kXa);kXa=NULL;d1free(err);err=NULL;
	if( clear_buffer ){
		z3free(p_zbuf[0]);p_zbuf[0]=NULL;
		i1free(p_Hj[0]);  p_Hj[0]=NULL;  
	}
  return ret;
}

dcmplx *pbc_overlap_old(Gaussian *lhs,Gaussian *rhs,int spdm,double *BravisVectors,double *Vectorfield,int nKpoints,double *kvectors,
                   int *iargs){
	const int Lx=iargs[0],Ly=iargs[1],Lz=iargs[2];
  double *ax=BravisVectors,*ay=BravisVectors+3,*az=BravisVectors+6;
  int kp,Ix,Jy,Kz; const int Hx=Lx/2, Hy=Ly/2, Hz=Lz/2, Qx=Lx/4, Qy=Ly/4, Qz=Lz/4;
  double **kXa=d2alloc(nKpoints,3),*kvec=NULL;
  for(kp=0,kvec=kvectors;kp<nKpoints;kp++,kvec=kvec+3){ // kvector[kp] \cdot ax, ay, az
  	kXa[kp][0]= Dot_3d( kvec,ax ); kXa[kp][1]= Dot_3d( kvec,ay ); kXa[kp][2]= Dot_3d( kvec,az );
  }
  dcmplx *ret=z1alloc(nKpoints);
  for(kp=0;kp<nKpoints;kp++) ret[kp]=0.0;
  
  for(Ix=-Hx;Ix<=Hx;Ix++){ for(Jy=-Hy;Jy<=Hy;Jy++){ for(Kz=-Hz;Kz<=Hz;Kz++){
		double displ[3]={ ax[0]*Ix + ay[0]*Jy + az[0]*Kz,
		                  ax[1]*Ix + ay[1]*Jy + az[1]*Kz, ax[2]*Ix + ay[2]*Jy + az[2]*Kz};
		//dcmplx ovlp=Gaussian_overlap(lhs,rhs,Vectorfield,displ);
		dcmplx ovlp=Gaussian_ovlpNew(lhs,rhs,Vectorfield,displ);
		for(kp=0;kp<nKpoints;kp++){ 
			double arg=kXa[kp][0]*Ix + kXa[kp][1]*Jy + kXa[kp][2]*Kz;
			dcmplx zfac=cos(arg)+_I*sin(arg);
			ret[kp]+=zfac*ovlp;
		}
		//if(Ix==0 && Jy==0 && Kz==0) printf("#pbc_overlap:%d %d %d %f   %f+j%f   %f+j%f\n",Ix,Jy,Kz,dist3D(lhs->R,rhs->R),creal(ovlp),cimag(ovlp),creal(ret[0]),cimag(ret[0]));
  }}}
  //printf("#pbc_overlap:%f+j%f\n",creal(ret[0]),cimag(ret[0]));fflush(stdout);
  d2free(kXa);kXa=NULL;
  return ret;
}


dcmplx Gaussian_overlap(Gaussian *lhs,Gaussian *rhs,const double *Vecfield,const double *Rdispl){
	int i,j; const int Nlhs=lhs->N, Nrhs=rhs->N;
	double R[6]={lhs->R[0],lhs->R[1],lhs->R[2], rhs->R[0] + Rdispl[0],rhs->R[1]+ Rdispl[1],rhs->R[2]+ Rdispl[2]};
	double alpha[2]={0,0}; const int Nga=2;
	ushort xe_pows[6];
	dcmplx ret=0.0;
	for(i=0;i<Nlhs;i++){ alpha[0]=lhs->alpha[i];
	                     xe_pows[0]=lhs->pows[i][0]; xe_pows[1]=lhs->pows[i][1]; xe_pows[2]=lhs->pows[i][2];
		for(j=0;j<Nrhs;j++){ alpha[1]=rhs->alpha[j];
		                     xe_pows[3]=rhs->pows[j][0]; xe_pows[4]=rhs->pows[j][1]; xe_pows[5]=rhs->pows[j][2];
		  ushort diffs[6]={0,0,0, 0,0,0};
			dcmplx ovlp=calc_matrix( ope_xovlp1, xe_pows, diffs, R, alpha, Nga, Vecfield, 3);
			ret+= ovlp* lhs->cofs[i]* rhs->cofs[j];
		}
	}
	return ret;
}
#define NgaMax_3 36
#define NnucPowsMax 6   // in fact 3
// matrix elmt of normalized Cartesian Gaussian
// e.g. overlap = sqrt(__PI/(alph[0]+alph[1]))**3 *exp(-(al*bt/(al+bt))*( (A-B)**2 ) 
// @param xe_pows[3+3]
// @param diffs  [3+3]
// @param A      [3+3]
// @param alph   [1+1]
// @param nga   :=2  for ope overlap
// @param args    A_over_c[3]  
dcmplx calc_matrix( dcmplx (*ope)(ushort *,ushort *,double *,double *,ushort,double *,ushort),
       ushort *xe_pows,ushort *diffs, double *A,double *alph,ushort nga,double *args,ushort nargs){
  int i,j,ij,kk;
  double dF,F_2,ret=0.0;
  ushort diffs1[NgaMax_3],xe_pows1[NgaMax_3];
  ushort Xnuc_pows[NnucPowsMax];
  ij=0; for(i=0;i<nga;i++){ for(j=0;j<3;j++,ij++){
    if(xe_pows[ij]){
      __a1cpy(xe_pows,xe_pows1,3*nga,kk);
      __a1cpy(diffs,  diffs1,  3*nga,kk);
      xe_pows1[ij]--;diffs1[ij]++;
      dF=calc_matrix(ope,xe_pows1,diffs1,A,alph,nga,args,nargs);
      ret=0.50*dF/alph[i];
      if( xe_pows[ij]>1){
         xe_pows1[ij]=xe_pows[ij]-2;
         F_2=calc_matrix(ope,xe_pows1,diffs,A,alph,nga,args,nargs);
         ret=ret+( xe_pows[ij]-1)*0.50*F_2/alph[i];
      }
      return ret;
    }
  }}
  __a1clr(Xnuc_pows,NnucPowsMax,kk); // empty array
  // printf("at bottom\n"); fflush(stdout);
  ret=ope(diffs,Xnuc_pows,A,alph,nga,args,nargs);
  // printf("at bottom %f\n",ret); fflush(stdout);
  return ret;
}

// diffs[6]      xdiff_LHS,ydiff_LHS,... 
// Xnuc_Pows[3]  power of (A-B)^{\mu}    should be [0,0,0] at first call
dcmplx ope_xovlp1(ushort *diffs,ushort *Xnuc_Pows,double *A,double *alph,ushort Nga,double *args,ushort Nargs){
 double ret=0.0, dum,prod;
 double gam=alph[0]*alph[1]/(alph[0]+alph[1]);
 ushort i,j,ij,kk;
 ushort *newdiffs,*newpows; long lnwdiffs,lnwpows;
 short sgn=1;
 double *AoverC=args; //printf("#VectorPotential:%f %f %f",AoverC[0],AoverC[1],AoverC[2]);
 ij=0;for(i=0;i<2;i++){
   sgn=1-i*2; double fac=(alph[i]/(alph[0]+alph[1]));
   for(j=0;j<3;j++,ij++){
     if(diffs[ij]){
       _clone_ush1( diffs, newdiffs, 6, lnwdiffs, kk);
       _clone_ush1( Xnuc_Pows, newpows, 3, lnwpows, kk);
       newdiffs[ij]--;
       // reduce diffs by one and multiply  i VecF^{\mu} al/(al+bt)
       ret = _I * AoverC[j] * fac * ope_xovlp1(newdiffs,newpows,A,alph,Nga,args,Nargs);
       newpows[j]++; // further multiply (A-B)
       ret += -sgn*2.0*gam*ope_xovlp1(newdiffs,newpows,A,alph,Nga,args,Nargs);
       if( Xnuc_Pows[j]>0 ){
         newpows[j]=Xnuc_Pows[j]-1;
         dum=ope_xovlp1(newdiffs,newpows,A,alph,Nga,args,Nargs);
         ret=ret+sgn*Xnuc_Pows[j]*dum;
       }
       _free_ush1( newpows, 3, lnwpows);
       _free_ush1( newdiffs, 6, lnwdiffs);
       
       return ret;
    }
   }
 }
 double sqrDist,*B=A+3, expofac;
 sqrDist=_sqrdist1(A,B);   //sum( (A(:)-B(:))**2 )
 double Asqr=AoverC[0]*AoverC[0] + AoverC[1]*AoverC[1] + AoverC[2]*AoverC[2];
 expofac=exp(-gam*sqrDist - 0.25*Asqr/(alph[0]+alph[1]));
 double G[3]={ (alph[0]*A[0]+alph[1]*B[0])/(alph[0]+alph[1]), (alph[0]*A[1]+alph[1]*B[1])/(alph[0]+alph[1]), (alph[0]*A[2]+alph[1]*B[2])/(alph[0]+alph[1]) };
 double G_dot_A=G[0]*AoverC[0] + G[1]*AoverC[1] + G[2]*AoverC[2];
 dcmplx zfac=( cos(G_dot_A)+_I*sin(G_dot_A) );
 dum= sqrt(__PI/(alph[0]+alph[1]));
 ret= (dum*dum*dum) * expofac;
//printf("#cGaussian:ope_ovlp:dum:%f %f %f",dum,expofac,ret);
 for(i=0;i<3;i++){ 
    prod=1.0;for(kk=0;kk<Xnuc_Pows[i];kk++)prod*=A[i]-A[i+3];
    ret*=prod;
 }
 return ret*zfac;
}
/*
int calc_pbc_overlaps_f(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int filenumber){
	int verbose=0;
	static int append_logf=0;
	if(verbose){ int narg=0;
		char *str=NULL;
		FILE *fptr=fopen("calc_pbc_overlaps_input.log",(append_logf ? "a":"w"));append_logf=1;
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nAtm);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%f:%f\n",Rnuc[0],Rnuc[1]);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%d:%d\n",IZnuc[0],IZnuc[1]);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:Rnuc:%s\n",(narg++), (str=d1toa(Rnuc,3*nAtm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(IZnuc,nAtm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nDa);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(distinctIZnuc,nDa)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(Nsh,nDa)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(ell,nDa)));free(str);fflush(fptr);
		int NshSUM=i1sum(Nsh,nDa);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(npGTO,NshSUM)));free(str);fflush(fptr);
		int Npgto=i1sum(npGTO,NshSUM);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=d1toa(alph,Npgto)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=d1toa(cofs,Npgto)));free(str);fflush(fptr);

		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nAtmB);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:RnucB:%s\n",(narg++), (str=d1toa(RnucB,3*nAtmB)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:IZnucB:%s\n",(narg++), (str=i1toa(IZnucB,nAtmB)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nDb);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:distinctIZnucB:%s\n",(narg++), (str=i1toa(distinctIZnucB,nDb)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:NshB:%s\n",(narg++), (str=i1toa(NshB,nDb)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:ellB:%s\n",(narg++), (str=i1toa(ellB,nDb)));free(str);fflush(fptr);
		NshSUM=i1sum(NshB,nDb);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:npGTOB:%s\n",(narg++), (str=i1toa(npGTOB,NshSUM)));free(str);fflush(fptr);
		if(n_cols!=NULL){
			fprintf(fptr,"#pbc_overlaps01:%02d:n_cols:%s\n",(narg++), (str=i1toa(n_cols,NshSUM)));free(str);fflush(fptr);
		}
		Npgto=i1sum(npGTOB,NshSUM);
		fprintf(fptr,"#pbc_overlaps01:%02d:alphB:%s\n",(narg++), (str=d1toa(alphB,Npgto)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:cofsB:%s\n",(narg++), (str=d1toa(cofsB,Npgto)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:spdm:%d\n",(narg++), spdm);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:BravisVectors:%s\n",(narg++), (str=d1toa(BravisVectors,3*spdm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:Vectorfield:%s\n",(narg++), (str=d1toa(Vectorfield,3)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:kvectors:%s\n",(narg++), (str=d1toa(kvectors,3*spdm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:spdm:%d,%d %d %d %d\n",(narg++), Lx,Ly,Lz,nCGTO_2,nCGTO_1);fflush(fptr);
		fclose(fptr);
	}

	dcmplx **zbuf=calc_pbc_overlaps(nAtm, Rnuc,    IZnuc,   nDa,  distinctIZnuc,
                    Nsh,  ell,  npGTO,      alph, cofs,
                    nAtmB,       RnucB,   IZnucB,           nDb,            distinctIZnucB,
                    NshB, ellB, n_cols,     npGTOB,  alphB,
                    cofsB,    spdm, BravisVectors, Vectorfield,nKpoints,
                    kvectors,       Lx,Ly,Lz,  nCGTO_2, nCGTO_1);
	dcmplx *z1a=zbuf[0];
	int Ld=3*nKpoints*nCGTO_2*nCGTO_1; // 3*nKpoints*nBS2a[0]*nBS1;
	char filename[50];
	sprintf(filename,"calc_pbc_overlaps_%03d.retf",filenumber);
	write_z1buf(filename,z1a,Ld);
	z2free(zbuf);zbuf=NULL;
	return Ld;
}*/
int write_z1buf(const char *path,const dcmplx *zbuf,const int Ld){
	FILE *fp=fopen(path,"w");
	int i,n;const int ncol=6;
	fprintf(fp,"z\n");
	for(i=0,n=0;i<Ld;i++){
		fprintf(fp,"%18.10e %18.10e ",creal(zbuf[i]),cimag(zbuf[i]));
		n++;
		if(n==ncol){ fprintf(fp,"\n");n=0;}
	}
	fclose(fp);
	return Ld;
}

int test_overlaps(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                  const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                  int spdm, double *BravisVectors, double *Vectorfield,    int nKpoints,
                  double *kvectors, const int nCGTO_1){
  int fix_nrmz=1; // 2^{1}: fix nrmz of BS2  2^{0} fix nrmz of BS1
  char sbuf10[10];
  char *sdum=getenv("calc_pbc_overlaps.fix_nrmz");
  if(sdum!=NULL){ long ldum=0;int iretv=sscanf(sdum,"%ld",&ldum);if(iretv){ fix_nrmz=(int)ldum;} }
  int nBS1=0;int verbose=0;
  Gaussian **BS =gen_bset01(&nBS1,nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,NULL,npGTO,alph,cofs,0,verbose);
  check_nrmz01("BS1",BS,nBS1,( (fix_nrmz&1)==0 ? 0:1 ));
	int i,j; double zerovec[3]={0,0,0};
	FILE *fp0=fopen("xgaussian_bset.log","w");
	int ibs;for(ibs=0;ibs<nBS1;ibs++){ 
		dcmplx cdum=Gaussian_overlap(BS[ibs],BS[ibs],zerovec,zerovec);
		Gaussian_fprtout( (fp0,"#BS%03d: %12.6f",ibs,creal(cdum)),fp0,BS[ibs]);
	}
	fclose(fp0);
	FILE *fp1=fopen("gaussope_SAO.log","w");
	for(i=0;i<nBS1;i++){ 
		for(j=0;j<nBS1;j++){ 
			dcmplx ovlp=Gaussian_overlap(BS[i],BS[j],zerovec,zerovec);
			fprintf(fp1,"%16.8f %16.8f       ",creal(ovlp),cimag(ovlp));}
		fprintf(fp1,"\n");
	}
	fclose(fp1);
	FILE *fp2=fopen("gaussope_pbcSAO.log","w");
	for(i=0;i<nBS1;i++){ for(j=0;j<nBS1;j++){ 
		int iargs_dmy[1]={0};int Ldim_margin=5;
		dcmplx *arr=pbc_overlap(0,i,j,BS[i],BS[j],3,BravisVectors,Vectorfield,nKpoints,kvectors, iargs_dmy,
									((i==nBS1-1 && j==nBS1-1) ? 1:0),Ldim_margin);
		fprintf(fp2,"%16.8f %16.8f       ",creal(arr[0]),cimag(arr[0]));free(arr);arr=NULL;
  }} fclose(fp2);
	
	void prtout_Bset(const char *fpath,const char *title,Gaussian **BS,const int Nb);
	prtout_Bset("xGaussianBset.log","",BS,nBS1);
	free_Bset(BS,nBS1);
	return 0;
}

int test001(int nAtm,  double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
            const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
            int spdm, double *BravisVectors, double *Vectorfield,    int nKpoints,
            double *kvectors, const int nCGTO_1){
	int verbose=1;
	static int append_logf=0;
	if(verbose){ int narg=0;
		char *str=NULL;
		FILE *fptr=fopen("test001_input.log",(append_logf ? "a":"w"));append_logf=1;
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nAtm);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%f:%f\n",Rnuc[0],Rnuc[1]);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%d:%d\n",IZnuc[0],IZnuc[1]);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:Rnuc:%s\n",(narg++), (str=d1toa(Rnuc,3*nAtm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(IZnuc,nAtm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%d\n",(narg++),nDa);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(distinctIZnuc,nDa)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(Nsh,nDa)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(ell,nDa)));free(str);fflush(fptr);
		int NshSUM=i1sum(Nsh,nDa);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=i1toa(npGTO,NshSUM)));free(str);fflush(fptr);
		int Npgto=i1sum(npGTO,NshSUM);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=d1toa(alph,Npgto)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nAtm:%s\n",(narg++), (str=d1toa(cofs,Npgto)));free(str);fflush(fptr);
		
		fprintf(fptr,"#pbc_overlaps01:%02d:spdm:%d\n",(narg++), spdm);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:BravisVectors:%s\n",(narg++), (str=d1toa(BravisVectors,3*spdm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:Vectorfield:%s\n",(narg++), (str=d1toa(Vectorfield,3)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:kvectors:%s\n",(narg++), (str=d1toa(kvectors,3*spdm)));free(str);fflush(fptr);
		fprintf(fptr,"#pbc_overlaps01:%02d:nCGTO1:%d\n",(narg++), nCGTO_1);fflush(fptr);
		fclose(fptr);
	}

	test_overlaps(nAtm, Rnuc,    IZnuc,   nDa,  distinctIZnuc,
                Nsh,  ell,  npGTO,      alph, cofs,
                spdm, BravisVectors, Vectorfield,nKpoints, kvectors, nCGTO_1);
  return 0;
}
void prtout_Bset(const char *fpath,const char *title,Gaussian **BS,const int Nb){
	FILE *fp=fopen(fpath,"w");
	int i,j;
	for(i=0;i<Nb;i++){
		fprintf(fp,"#Gaussian_prtout:%s Nuc:%d %f %f %f\n",title, BS[i]->Jnuc,BS[i]->R[0],BS[i]->R[1],BS[i]->R[2]);
		fprintf(fp,"#Gaussian_prtout:%s %d nPGTO:%d\n",title,i+1,BS[i]->N);
		fprintf(fp,"##sub.pows:");
		for(j=0;j<BS[i]->N;j++){ 
			fprintf(fp,"      %1d %1d %1d ",BS[i]->pows[j][0],BS[i]->pows[j][1],BS[i]->pows[j][2]);} fprintf(fp,"\n");
		fprintf(fp,"##sub.alph:");
		for(j=0;j<BS[i]->N;j++){ 
			fprintf(fp,"%12.4e ",BS[i]->alpha[j]);} fprintf(fp,"\n");
		fprintf(fp,"##sub.cofs:");
		for(j=0;j<BS[i]->N;j++){ fprintf(fp,"%12.6f ",BS[i]->cofs[j]);} fprintf(fp,"\n");
	}
	fclose(fp);
}

// [spdm+1,3,nKpoints,nCGTO_2,nCGTO_1,2]
dcmplx *calc_pbc_overlapsx(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1){
	__assertf( spdm==3,("calc_pbc_overlapsx:spdm=%d",spdm),-1);
  int fix_nrmz=1; // 2^{1}: fix nrmz of BS2  2^{0} fix nrmz of BS1
  char sbuf10[10];
  char *sdum=getenv("calc_pbc_overlaps.fix_nrmz");
  if(sdum!=NULL){ long ldum=0;int iretv=sscanf(sdum,"%ld",&ldum);if(iretv){ fix_nrmz=(int)ldum;} }
  //printf("#calc_pbc_overlaps:entering:%d %d %d fix_nrmz=%d",ellB[0],ellB[1],ellB[2],fix_nrmz);fflush(stdout);
  int nBS1=0;int verbose=0;
  int iargs[3]={Lx,Ly,Lz};
//printf("#calc_pbc_overlapsx:gen_bset01:nAtm:%d IZnuc:%d\n",nAtm,IZnuc[0]);fflush(stdout);
  Gaussian **BS =gen_bset01(&nBS1,nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,NULL,npGTO,alph,cofs,0,verbose);
  check_nrmz01("BS1",BS,nBS1,( (fix_nrmz&1)==0 ? 0:1 ));
  int nBS2a[3]={0,0,0};
//printf("#calc_pbc_overlapsx:gen_bsetx3:nAtm:%d Nsh:%d,%d IZnuc:%d\n",nAtmB,NshB[0],NshB[1],IZnucB[0]);fflush(stdout);
  Gaussian ***BS2=gen_bsetx3(nBS2a,nAtmB,RnucB,IZnucB,nDb,distinctIZnucB,NshB,ellB,n_cols,npGTOB,alphB,cofsB);
  check_nrmz01("BS2",BS2[0],nBS2a[0], ( (fix_nrmz&2)==0 ? 0:1 ) );
  __assertf( (nCGTO_2==nBS2a[0]) && (nCGTO_1==nBS1),("%d,%d / %d,%d",nBS2a[0],nBS1,nCGTO_2,nCGTO_1),-1);

//test_ovlpx( BS, nBS1, BS2[0], nBS2a[0], BravisVectors); XXX XXX XXX 

  int i2LH,j1RH;
  char dbgng=1;
  int Ldim_margin=5;int I; const int N_I=(spdm+1);
  dcmplx ***zbuf=z3alloc_i(N_I,3,(nKpoints*nBS2a[0]*nBS1), 0.0);
	double clock00=__ctime; double clock01=clock00, clock02=clock00;
	int n=0; int append_1=0; int I_cols;
	for(I_cols=0;I_cols<3;I_cols++){
		if(nBS2a[I_cols]<=0)continue;
		int ijk=0;
		double wct_00=__ctime;
		for(i2LH=0;i2LH<nBS2a[I_cols];i2LH++){
			for(j1RH=0;j1RH<nBS1;j1RH++){
  			if( i2LH==1 && j1RH==0 ){ double wct1=(__ctime)-wct_00; printf("#calc_pbc_overlapsx:357:%d %d/%d %d:%f",I_cols,i2LH,nBS2a[I_cols],nBS1,wct1);fflush(stdout);}
				dcmplx **ovlps=pbc_overlapx(I_cols,i2LH,j1RH, BS2[I_cols][i2LH], BS[j1RH],spdm, BravisVectors,Vectorfield, nKpoints, kvectors, iargs, 0, Ldim_margin);
				clock02=clock01;clock01=__ctime;n++;
				int kp;
				for(I=0;I<N_I;I++) for(kp=0;kp<nKpoints;kp++) zbuf[I][I_cols][ kp*(nBS2a[0]*nBS1) + i2LH*(nBS1) + j1RH]=ovlps[I][kp];
				z2free(ovlps);
		}}
		pbc_overlapx(-1,-1,-1, NULL, NULL,-1, NULL, NULL,-1,NULL,NULL,-1, -1); // clear buffer
	}
	free_Bset(BS,nBS1);
	free_Bset(BS2[0],nBS2a[0]);
	free_Bset(BS2[1],nBS2a[1]);
	if( nBS2a[2]>0 && BS2[2]!=NULL ){ free_Bset(BS2[2],nBS2a[2]);}
	//printf("#calc_pbc_overlaps:END.030:%p\n",zbuf);fflush(stdout);
	dcmplx *ret=zbuf[0][0];free(zbuf[0]);zbuf[0]=NULL; free(zbuf); zbuf=NULL;
	//printf("#calc_pbc_overlaps:returning:%p\n",zbuf);fflush(stdout);
	return ret;
}
/*
int fcountup(const char *fpath, const char incm){
	int retv=0;
	FILE *fp=fopen(fpath,"r");
	if(fp==NULL) { retv=0;}
	else         { char sbuf[21];int ngot=0;long ldum=-1;
		while(fgets(sbuf, 20, fp)!=NULL){ int le=strlen(sbuf);
			if(le<1)continue;if(sbuf[le-1]=='\n'){ sbuf[le-1]='\0';le--; if(le<1)continue;}
			ngot=sscanf(sbuf,"%ld",&ldum);if(ngot){ retv=(int)ldum;break;}
		} fclose(fp);
		__assertf( ngot,("fcountup:read from %s failed",fpath),1);
	}
	if(incm){
		retv+=1;fp=fopen(fpath,"w");fprintf(fp,"%d",retv);fclose(fp);
	}
	return retv;
}*/
double d1min(const double *A,int N){
	int i,at=0;double val=A[at];
	for(i=1;i<N;i++){ if(A[i]<val){ at=i;val=A[at];}}
	return val;
}
double d1max(const double *A,int N){
	int i,at=0;double val=A[at];
	for(i=1;i<N;i++){ if(A[i]>val){ at=i;val=A[at];}}
	return val;
}
int calc_nabla_f(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
                 const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
								 int spdm,  double *BravaisVectors, int nKpoints,
								 double *kvectors,  int Lx,int Ly,int Lz,   const int nCGTO_1, const int MPIrank, const int filenumber){
	printf("#calc_nabla_f:%d ... %d,%d,%d\n",nAtm,nCGTO_1,MPIrank,filenumber);fflush(stdout);
	int nBS1=0;	const int verbose=0;
	Gaussian **BS =gen_bset01(&nBS1,nAtm,Rnuc,IZnuc,nDa,distinctIZnuc,Nsh,ell,NULL,npGTO,alph,cofs,0,verbose);
	dcmplx ****zbuf=pbc_nabla(BS,nBS1, spdm, BravaisVectors, nKpoints, kvectors, Lx,Ly,Lz);
	char fnme[50];sprintf(fnme,"calc_nabla_f%03d.retf",filenumber);
	write_z1buf(fnme, zbuf[0][0][0], 3*nKpoints*nBS1*nBS1 );
	z4free(zbuf);free_Bset(BS,nBS1);
	return 0;
}
// returns <I| \nabla |J>  as Matr[:][I][J] ....
dcmplx ****pbc_nabla(Gaussian **BS,const int nBS, int spdm,  double *BravaisVectors, int nKpoints,
                     double *kvectors,  int Lx,int Ly,int Lz){
	dcmplx ****ret=z4alloc(nKpoints,3,nBS,nBS); const double Vectorfield[3]={0,0,0};
	int kp,dir,iRH,jLH;int iargs[3]={Lx,Ly,Lz};int Ldim_margin=5;
	for(iRH=0;iRH<nBS;iRH++){ Gaussian **gradG=Gaussian_nabla(BS[iRH]);
		for(jLH=0;jLH<nBS;jLH++){ 
			for(dir=0;dir<3;dir++) {
				dcmplx *ovlps= pbc_overlap(0,jLH, iRH, BS[jLH], gradG[dir], spdm, BravaisVectors,
				                          Vectorfield, nKpoints, kvectors, iargs, 0, Ldim_margin);
				for(kp=0;kp<nKpoints;kp++) ret[kp][dir][jLH][iRH] =ovlps[kp];
				z1free(ovlps);
		}}
		for(dir=0;dir<3;dir++) free_Gaussian01(gradG[dir],0); free(gradG);
	}
	return ret;	
}
Gaussian **Gaussian_nabla(const Gaussian *org){
	static int dbgng=3;
	Gaussian **ret=(Gaussian **)malloc(3*sizeof(Gaussian *));
	int I,dir;const int N=org->N;
#define _Cpy3( src, dst )  { dst[0]=src[0];dst[1]=src[1];dst[2]=src[2];}
	for(dir=0;dir<3;dir++){
		int Nnew=0;
		for(I=0;I<N;I++){ Nnew++; if( org->pows[I][dir]>0 )Nnew++;  }
		double *cofs=d1alloc(Nnew),*alph=d1alloc(Nnew);ushort **pws=ush2alloc(Nnew,3);
		int Inew=0;
		for(I=0;I<N;I++){
			if( org->pows[I][dir]>0 ){ cofs[Inew]=org->cofs[I]*( org->pows[I][dir] ); alph[Inew]=org->alpha[I];
			                           _Cpy3( org->pows[I], pws[Inew] );         pws[Inew][dir]--;      Inew++;}
			cofs[Inew]=org->cofs[I]*(-2*(org->alpha[I])); alph[Inew]=org->alpha[I];
			_Cpy3( org->pows[I], pws[Inew] );        pws[Inew][dir]++;      Inew++;
		}
		ret[dir]=new_Gaussian_1( 0, alph,org->R,NULL,pws,Nnew,cofs,org->Jnuc);
		d1free(cofs);d1free(alph);ush2free(pws);
	}
	/* dir=0;
	if(1){
		FILE *fp=fopen("Gaussian_nabla_x.log","a");
		int Inew=0;
		for(I=0;I<N;I++){ int jj;
			for(jj=0;jj<2;jj++){
				fprintf(fp,"%03d: %d,%d,%d  %12.4e %12.6f   \t\t  %03d: %d,%d,%d  %12.4e %12.6f\n",
					I, org->pows[I][0],org->pows[I][1],org->pows[I][2],org->alpha[I],org->cofs[I], 
					Inew, ret[dir]->pows[Inew][0],ret[dir]->pows[Inew][1],ret[dir]->pows[Inew][2],ret[dir]->alpha[Inew],ret[dir]->cofs[Inew]); Inew++;
				if( org->pows[I][0]==0 )break;}
		} fclose(fp);
	}*/
	if(dbgng){
		double almin=d1min(org->alpha,org->N);double sqrt_almin=sqrt(almin);double fac=2.0;
		double Rlb[3]={ org->R[0]-fac/sqrt_almin, org->R[1]-fac/sqrt_almin, org->R[2]-fac/sqrt_almin};
		double Rrt[3]={ org->R[0]+fac/sqrt_almin, org->R[1]+fac/sqrt_almin, org->R[2]+fac/sqrt_almin};
		double Ndim[3]={ 16,16,16 }; const double h=0.02;
		double dR[3]={ (Rrt[0]-Rlb[0])/Ndim[0], (Rrt[1]-Rlb[1])/Ndim[1], (Rrt[2]-Rlb[2])/Ndim[2] };
		double devmx[3]={-1,-1,-1},lhs,rhs;const double devTOL=1.0e-5; int ix,jy,kz; double R[3]={0,0,0};
		for(ix=0,R[0]=Rlb[0];ix<Ndim[0];ix++,R[0]+=dR[0]){
			for(jy=0,R[1]=Rlb[1];jy<Ndim[1];jy++,R[1]+=dR[1]){
				for(kz=0,R[2]=Rlb[2];kz<Ndim[2];kz++,R[2]+=dR[2]){
					for(dir=0;dir<3;dir++){
						double R1[3]={R[0],R[1],R[2]}; R1[dir]-=h; double v1=Gaussian_value(org,R1);
						double R2[3]={R[0],R[1],R[2]}; R2[dir]+=h; double v2=Gaussian_value(org,R2); 
						double nmdif=0.50*(v2-v1)/h, testee=Gaussian_value(ret[dir],R);
						double dev=fabs(nmdif-testee);
						if(dev>devTOL){ printf("#nmdiff:%f testee:%f dev=%e @%f,%f,%f\n",nmdif,testee,dev,R[0],R[1],R[2]);}
						if(dev>devmx[dir]) { devmx[dir]=dev;lhs=nmdif; rhs=testee;}
					}
		}}}
		printf("#Gaussian_nabla:compto nmdiff:devmax=%e ( nmdiff:%f / val:%f)\n",devmx[dir],lhs,rhs);
	}
//	if(dbgng>0) dbgng--;
	
	return ret;
}

// scheme 1 (lshift==1): < lh(r+T) | rh(r) e^{iAr} > ( ... to be summed over T with factor e^{ikT})
// scheme 0 (lshift==0): < lh(r) | rh(r-T) e^{iAr} > ( ... to be summed over T with factor e^{ikT})
dcmplx nmrovlp1(Gaussian *lh,Gaussian *rh,const double *VecField, const double *T, const char Lshift,
              const double *scalefacs,const char verbose){
	double almin1=d1min(lh->alpha,lh->N),almin2=d1min(rh->alpha,rh->N);
	double almax1=d1max(lh->alpha,lh->N),almax2=d1max(rh->alpha,rh->N);
	double almin=__MIN( almin1, almin2);
	double sqrt_almin=sqrt(almin); const double fac=scalefacs[0];//double fac=2.0;
	double almax=__MIN( almax1, almax2);
	double sqrt_almax=sqrt(almax);
	double Rlb[3],Rrt[3]; int jj;
	if(Lshift){ //lh center is shifted by -T
		double Rlb1[3]={ lh->R[0]-T[0] -fac/sqrt_almin, lh->R[1]-T[1] -fac/sqrt_almin, lh->R[2]-T[2] -fac/sqrt_almin};
		double Rrt1[3]={ lh->R[0]-T[0] +fac/sqrt_almin, lh->R[1]-T[1] +fac/sqrt_almin, lh->R[2]-T[2] +fac/sqrt_almin};
		double Rlb2[3]={ rh->R[0]-fac/sqrt_almin, rh->R[1]-fac/sqrt_almin, rh->R[2]-fac/sqrt_almin};
		double Rrt2[3]={ rh->R[0]+fac/sqrt_almin, rh->R[1]+fac/sqrt_almin, rh->R[2]+fac/sqrt_almin};
		for(jj=0;jj<3;jj++){ Rlb[jj]=__MIN(Rlb1[jj],Rlb2[jj]);Rrt[jj]=__MAX(Rrt1[jj],Rrt2[jj]);}
	} else {
		double Rlb1[3]={ rh->R[0]+T[0] -fac/sqrt_almin, rh->R[1]+T[1] -fac/sqrt_almin, rh->R[2]+T[2] -fac/sqrt_almin};
		double Rrt1[3]={ rh->R[0]+T[0] +fac/sqrt_almin, rh->R[1]+T[1] +fac/sqrt_almin, rh->R[2]+T[2] +fac/sqrt_almin};
		double Rlb2[3]={ lh->R[0]-fac/sqrt_almin, lh->R[1]-fac/sqrt_almin, lh->R[2]-fac/sqrt_almin};
		double Rrt2[3]={ lh->R[0]+fac/sqrt_almin, lh->R[1]+fac/sqrt_almin, lh->R[2]+fac/sqrt_almin};
		for(jj=0;jj<3;jj++){ Rlb[jj]=__MIN(Rlb1[jj],Rlb2[jj]);Rrt[jj]=__MAX(Rrt1[jj],Rrt2[jj]);}
	}
	const double h=__MIN( 0.02, scalefacs[1]/sqrt_almax);
	double dR[3]={ h,h,h };
	int Ndim[3]={ __NINT( (Rrt[0]-Rlb[0])/dR[0] ),  __NINT( (Rrt[1]-Rlb[1])/dR[1] ), __NINT( (Rrt[2]-Rlb[2])/dR[2] ) };
	
	if( verbose ){
		printf("#nmrovlp1: R:[%f,%f]x[%f,%f]x[%f,%f]/%fx%fx%f %d,%d,%d\n",Rlb[0],Rrt[0],Rlb[1],Rrt[1],Rlb[2],Rrt[2],h,h,h,Ndim[0],Ndim[1],Ndim[2]);
		fflush(stdout);
	}
	int ix,jy,kz;dcmplx cum=0.0; double dVol=dR[0]*dR[1]*dR[2];
	double Rminus[3],Rplus[3],lhv,rhv,R[3];
	for(ix=0,R[0]=Rlb[0],Rminus[0]=Rlb[0]-T[0],Rplus[0]=Rlb[0]+T[0];ix<Ndim[0];ix++,R[0]+=dR[0],Rminus[0]+=dR[0],Rplus[0]+=dR[0]){
		for(jy=0,R[1]=Rlb[1],Rminus[1]=Rlb[1]-T[1],Rplus[1]=Rlb[1]+T[1];jy<Ndim[1];jy++,R[1]+=dR[1],Rminus[1]+=dR[1],Rplus[1]+=dR[1]){
			dcmplx cum1=0.0;
			for(kz=0,R[2]=Rlb[2],Rminus[2]=Rlb[2]-T[2],Rplus[2]=Rlb[2]+T[2];kz<Ndim[2];kz++,R[2]+=dR[2],Rminus[2]+=dR[2],Rplus[2]+=dR[2]){
				if(Lshift){ lhv=Gaussian_value(lh, Rplus);rhv=Gaussian_value(rh,R);}
				else      { lhv=Gaussian_value(lh, R);rhv=Gaussian_value(rh,Rminus);}
				double AdotR= VecField[0]*R[0] + VecField[1]*R[1] + VecField[2]*R[2];
				dcmplx zfac= cos(AdotR) + _I*sin(AdotR);
				dcmplx val=rhv*lhv*zfac;
				cum1+=val;
			}
			cum+=cum1*dVol;
		}
	}
	return cum;
}
// Lshift scheme * exp(-
dcmplx nmrpbcovlp1(Gaussian *lh,Gaussian *rh,const char Lshift,const double *kvector,const double *BravaisVectors,
                 const double *VecField,const double *scalefacs){
	const double *ax=BravaisVectors,*ay=BravaisVectors+3,*az=BravaisVectors+6;
	const double TOL=1.0e-4;
	int nloop=0,extra_loop=1;
	int hj[3]={2,2,2},hjLAST[3]={-1,-1,-1},Ia[3]={0,0,0};
	dcmplx cum=0;
	double kDOTaj[3]={ Dot_3d(kvector,ax), Dot_3d(kvector,ay), Dot_3d(kvector,az) };
	char ifirst=1;
	double wtSTT=__ctime;
	while( (nloop++)<999 ){ int kk;double absmx=-1;
		for(Ia[0]=-hj[0];Ia[0]<=hj[0];Ia[0]++){  int o0=( abs(Ia[0])<=hjLAST[0]); double kDOTa0=kDOTaj[0]*Ia[0];
			for(Ia[1]=-hj[1];Ia[1]<=hj[1];Ia[1]++){  int o1=(o0 && abs(Ia[1])<=hjLAST[1]); double kDOTa1=kDOTaj[1]*Ia[1];
				for(Ia[2]=-hj[2];Ia[2]<=hj[2];Ia[2]++){  int o2=(o1 && abs(Ia[2])<=hjLAST[2]); double kDOTa2=kDOTaj[2]*Ia[2];
					if(o2) continue;
					double T[3]={ax[0]*Ia[0] + ay[0]*Ia[1] + az[0]*Ia[2],
											 ax[1]*Ia[0] + ay[1]*Ia[1] + az[1]*Ia[2], ax[2]*Ia[0] + ay[2]*Ia[1] + az[2]*Ia[2]};
					double kdota= kDOTa0 + kDOTa1 + kDOTa2;
					dcmplx zfac=cos(kdota)+_I*sin(kdota);
					dcmplx ovlp1 = nmrovlp1(lh,rh,VecField, T, Lshift,scalefacs,ifirst); cum+=zfac * ovlp1;
					dcmplx refv1 = Gaussian_ovlpNew(lh,rh,VecField,T);
					double wtNOW = __ctime;
					if( 1 ){ printf("#nmrovlp1:%d,%d,%d  %f %f / %f %f \t %f\n",Ia[0],Ia[1],Ia[2],creal(ovlp1),cimag(ovlp1),creal(refv1),cimag(refv1),wtNOW-wtSTT);} ifirst=0;
					double dum=cabs(ovlp1);
					if(dum>absmx){ absmx=dum;}
		}}}
		printf("#nmrpbcovlp1:loop%d:h:%d,%d,%d absmx:%e cum:%f,%f\n",nloop,hj[0],hj[1],hj[2],absmx,creal(cum),cimag(cum));
		if( absmx < TOL ){
			if( extra_loop>0 ){ hj[0]+=1;hj[1]+=1;hj[2]+=1;extra_loop--;continue;}
			break;
		} else {
			hj[0]+=1;hj[1]+=1;hj[2]+=1;continue;
		}
	}
	return cum;
}
void testovlp03(){
	double scalefacs_1[2]={2.6, 0.30};
	double scalefacs_2[2]={3.2, 0.15};
#define Ng_LH 2
#define Ng_RH 3
	const char Lshift=0;
	double BravaisVectors[9] = { 1.582, 0, 0,  0,1.333,0, 0,0,1.543 };
	double kvector[3]={ -0.3, 0.2, 0.05 };
	double VecField[3]={ 0.3776, -0.08848, 0.0 };
	int ell1,ell2,emm1,emm2;
	const double dev_TINY=1.0e-7;
	double displ[3]={0,0,0};char verbose=1;
	double alph_LH[Ng_LH]={ 0.20, 4.00 },cofs_LH[Ng_LH]={ 0.30, 0.85};
	double alph_RH[Ng_RH]={ 0.3776, 0.8848, 1.868 },cofs_RH[Ng_LH]={ 0.645, 0.794, 1.192};
	double R_LH[3]={ 0.0894, -0.0710, 0.1333 }, R_RH[3]={ 0.0710, -0.0645, -0.1467};
/*	FILE *fp=fopen("testovlp03b.002.dat","w"); FILE *fpA[2]={fp,stdout};int I;
	fprintf(fp,"#displ:%f %f %f VF:%f %f %f\n", displ[0], displ[1], displ[2], VecField[0], VecField[1], VecField[2]);
	for(ell1=0;ell1<4;ell1++){ for(emm1=-ell1;emm1<=ell1;emm1++){
		Gaussian *lh= Spherical_to_Cartesian( new_SphericalGaussian(alph_LH, R_LH,ell1,emm1,Ng_LH, cofs_LH,0) );
		for(ell2=0;ell2<4;ell2++){ for(emm2=-ell2;emm2<=ell2;emm2++){
			Gaussian *rh= Spherical_to_Cartesian( new_SphericalGaussian(alph_RH, R_RH,ell2,emm2,Ng_RH, cofs_RH,1) );
				dcmplx ovlpO=Gaussian_ovlpNew(lh,rh,VecField,displ);
				dcmplx ovlpN=nmrovlp1(lh,rh,VecField, displ, Lshift, scalefacs_1,verbose);verbose=0;
				double dev=cabs(ovlpO-ovlpN);
				if(dev>dev_TINY){
					dcmplx ovlpN2=nmrovlp1(lh,rh,VecField, displ, Lshift, scalefacs_2,verbose);
					double dev2=cabs(ovlpO-ovlpN2);
					for(I=0;I<2;I++)
						fprintf(fpA[I],"%d,%d x %d,%d : %16.8f %16.8f     %16.8f %16.8f  %14.4e     %16.8f %16.8f  %14.4e  %s\n", 
							ell1,emm1,ell2,emm2,creal(ovlpO),cimag(ovlpO), creal(ovlpN2),cimag(ovlpN2),dev2, creal(ovlpN),cimag(ovlpN),dev,
							(dev2<dev_TINY ? "":"XXX") );
				} else {
					for(I=0;I<2;I++)
						fprintf(fpA[I],"%d,%d x %d,%d : %16.8f %16.8f     %16.8f %16.8f\n", ell1,emm1,ell2,emm2,creal(ovlpO),cimag(ovlpO), creal(ovlpN),cimag(ovlpN));
				}
		}}
	}}
	__assertf( 0,(""),-1);*/
	// double VecField[3]={ 0,0,0};
	int pw_lh[3]={0,1,1},pw_rh[3]={2,1,2}; int i,j;
	ushort **pw_LH=ush2alloc(Ng_LH,3),**pw_RH=ush2alloc(Ng_RH,3);
	for(i=0;i<Ng_LH;i++)for(j=0;j<3;j++) pw_LH[i][j]=pw_lh[j];
	for(i=0;i<Ng_RH;i++)for(j=0;j<3;j++) pw_RH[i][j]=pw_rh[j];
	//new_Gaussian_1(char divide_by_Ng,const double *alpha,const double *R,const ushort *pow1,ushort **pow_a,int N,const double *cof,int Jnuc){
	Gaussian *lh=new_Gaussian_1(1,alph_LH, R_LH, NULL, pw_LH, Ng_LH,cofs_LH,0);
	Gaussian *rh=new_Gaussian_1(1,alph_RH, R_RH, NULL, pw_RH, Ng_RH,cofs_RH,1);
	printf("#testovlp03:Gaussian lh,rh\n");fflush(stdout);
	int iargs[1]={0},Ldim_margin=5;
	dcmplx *ovlps=pbc_overlap(0,0,0, lh, rh,3, BravaisVectors, VecField, 1, kvector,iargs,0,Ldim_margin);
	dcmplx *ovlpsB=NULL; // XXX XXX this function no longer exists --> pbc_overlap_B(0,0,0, lh, rh,3, BravaisVectors, VecField, 1, kvector,iargs,0,Ldim_margin);
	dcmplx *ovlpsO=pbc_overlap_old(lh,rh,3, BravaisVectors, VecField, 1, kvector,iargs);
	printf(" %16.8f %16.8f     %16.8f %16.8f     %16.8f %16.8f\n", 
	       creal(ovlps[0]),cimag(ovlps[0]), creal(ovlpsB[0]),cimag(ovlpsB[0]),creal(ovlpsO[0]),cimag(ovlpsO[0]) );
	
	printf("#testovlp03:start nmrintg...\n");fflush(stdout);
	dcmplx nmrovlp=nmrpbcovlp1( lh, rh, Lshift, kvector, BravaisVectors, VecField, scalefacs_1);
	
	FILE *fp1=fopen("testovlp03b.log","a");
	fprintf(fp1," %16.8f %16.8f      %16.8f %16.8f     %16.8f %16.8f     %16.8f %16.8f\n", 
	        creal(nmrovlp),cimag(nmrovlp), creal(ovlps[0]),cimag(ovlps[0]), creal(ovlpsB[0]),cimag(ovlpsB[0]),
	        creal(ovlpsO[0]),cimag(ovlpsO[0]) );
	fclose(fp1);
}
void testovlp04(){
	const double dev_TINY=1.0e-7;const double dev_TOL=1.0e-7;
	double alph_s[10]={8236.0, 1235.0, 280.8, 79.27, 25.59, 8.997, 3.319, 0.9059, 0.3643, 0.1285};
	double cofs_s[10]={0.000531, 0.004108, 0.021087, 0.081853, 0.234817, 0.434401, 0.346129, 0.039378, -0.008983, 0.002385};
	double alph_d[9]={403.83, 121.17, 46.345, 19.721, 8.8624, 3.9962, 1.7636, 0.70619, 0.2639};
	double cofs_d[9]={ 0.0014732, 0.0126725, 0.0580451, 0.1705103, 0.3185958, 0.3845023, 0.2737737, 0.0743967, 0.0};
	double alph_p[13]={8676.5, 2055.9, 666.23, 253.1, 106.12, 47.242, 21.825, 9.9684, 4.5171, 1.9982, 0.70988, 0.28145, 0.10204};
  double cofs_p1[13]={0.0004357, 0.0037815, 0.0204782, 0.0792834, 0.2178473, 0.3878585, 0.359435, 0.1121995, 0.0043874, 0.0017809, -0.0004576, 0.0002122, -7.34e-05};
  double cofs_p2[13]={-0.0001748, -0.0015263, -0.0083399, -0.0332203, -0.095418, -0.1824026, -0.1558308, 0.1867899, 0.5427733, 0.3873309, 0.0453069, -0.0043784, 0.0018111};
	double cofs_p3[13]={4.51e-05, 0.0003964, 0.0021555, 0.008672, 0.024868, 0.0485472, 0.0396156, -0.0605749, -0.1871699, -0.1377757, 0.2928021, 0.5760896, 0.3078617};
	
	double alph_sB[20]={10639000.0, 1593400.0, 362610.0, 102700.0, 33501.0, 12093.0, 4715.9, 1955.6, 852.61, 387.67, 182.68, 88.245, 39.263, 19.234, 9.4057, 4.1601, 1.8995, 0.60472, 0.30114, 0.12515};
	double cofs_s1[20]={5.9e-06, 4.61e-05, 0.0002422, 0.0010226, 0.0037113, 0.0119785, 0.0346927, 0.0891239, 0.1934557, 0.3209019, 0.3299233, 0.1494121, 0.0149938, -0.0009165, 0.000438, -0.0002398, 7.36e-05, -3.67e-05, 2.39e-05, -5.6e-06};
	double cofs_s2[20]={-1.9e-06, -1.45e-05, -7.61e-05, -0.000321, -0.0011709, -0.0037968, -0.0112307, -0.0299277, -0.0712706, -0.1403136, -0.2030763, -0.0960985, 0.3558086, 0.5921792, 0.2215977, 0.0137648, 0.0008395, -4.51e-05, -8.5e-06, -1.24e-05};
	double cofs_s3[20]={7e-07, 5.7e-06, 3.03e-05, 0.0001275, 0.0004659, 0.0015096, 0.0044852, 0.0119835, 0.0289571, 0.0581566, 0.0888133, 0.0445244, -0.2060387, -0.5127017, -0.1509349, 0.6789203, 0.5817697, 0.0467555, -0.0111825, 0.0024402};
	double cofs_s4[20]={-2e-07, -1.8e-06, -9.3e-06, -3.91e-05, -0.0001428, -0.0004628, -0.001375, -0.0036784, -0.0088981, -0.0179529, -0.0275732, -0.0140953, 0.0672561, 0.1766928, 0.0528861, -0.3075955, -0.4700658, 0.2558761, 0.6980341, 0.2967256};
	double R_1[3]={ -0.645, -0.894, 0.1192};
	double R_2[3]={ -0.1467, 0.375, 0.3776};
	//(const double *alpha, const double *R,int l,int m,int N, const double *cof,int Jnuc)
	int ell1[12]={ 0,  1,1,1,  1,1,1,  2,2,2,2,2};
	int emm1[12]={ 0, -1,0,1, -1,0,1, -2,-1,0,1,2};
	Gaussian *BS1[12]={ Spherical_to_Cartesian( new_SphericalGaussian( alph_s, R_1, 0, 0, 10, cofs_s, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_1, 1, -1, 13, cofs_p1, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_1, 1,  0, 13, cofs_p1, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_1, 1,  1, 13, cofs_p1, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_1, 1, -1, 13, cofs_p3, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_1, 1,  0, 13, cofs_p3, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_1, 1,  1, 13, cofs_p3, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_1, 2, -2, 9, cofs_d, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_1, 2, -1, 9, cofs_d, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_1, 2,  0, 9, cofs_d, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_1, 2,  1, 9, cofs_d, 0) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_1, 2,  2, 9, cofs_d, 0) ) };
	int ell2[10]={ 0,0,  1,1,1,   2,2,2,2,2};
	int emm2[10]={ 0,0, -1,0,1,  -2,-1,0,1,2};
	Gaussian *BS2[10]={ Spherical_to_Cartesian( new_SphericalGaussian( alph_sB, R_2, 0, 0, 20, cofs_s1, 0 ) ),
                     Spherical_to_Cartesian( new_SphericalGaussian( alph_sB, R_2, 0, 0, 20, cofs_s4, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_2, 1, -1, 13, cofs_p2, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_2, 1,  0, 13, cofs_p2, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_p, R_2, 1,  1, 13, cofs_p2, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_2, 2, -2, 9, cofs_d, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_2, 2, -1, 9, cofs_d, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_2, 2,  0, 9, cofs_d, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_2, 2,  1, 9, cofs_d, 0 ) ),
	                  Spherical_to_Cartesian( new_SphericalGaussian( alph_d, R_2, 2,  2, 9, cofs_d, 0 ) ) };
	double scalefacs_1[2]={2.6, 0.30};
	double scalefacs_2[2]={3.2, 0.15};int verbose=1;
	const int Lshift=0;
	double V_al[3*4]={ 0,0,0,  0.1192,0.0645,0.0,  1.467,-1.333,1.868, 64.58,37.76,8.848}; const int nV=4;
	double D_al[3*2]={ 0.5,-0.2,0.3,  2.4,1.8,1.6 };                                       const int nD=2;
	int iV,jD,Ib,Jb,I; double *V,*D; const int Nb1=12,Nb2=10;
	FILE *fpA[2]={ fopen("testovlp04.log","w"),stdout };
	for(iV=0,V=V_al;iV<nV;iV++,V=V+3){
		for(jD=0,D=D_al;jD<nD;jD++,D=D+3){
			for(I=0;I<2;I++)fprintf(fpA[I],"\n\n#%d,%d:\n",iV,jD);
			for(Ib=0;Ib<Nb1;Ib++)for(Jb=0;Jb<Nb2;Jb++){
				dcmplx ovlpO=Gaussian_ovlpNew(BS1[Ib],BS2[Jb],V,D);
				dcmplx ovlpN=nmrovlp1(BS1[Ib],BS2[Jb],V,D, Lshift, scalefacs_1,verbose);verbose=0; double dev=cabs(ovlpO-ovlpN);
				if(dev>dev_TOL){ dcmplx ovlpN2=nmrovlp1(BS1[Ib],BS2[Jb],V,D, Lshift, scalefacs_2,verbose);double dev2=cabs(ovlpO-ovlpN2);
					for(I=0;I<2;I++)
						fprintf(fpA[I],"%d,%d x %d,%d : %16.8f %16.8f     %16.8f %16.8f  %14.4e     %16.8f %16.8f  %14.4e  %s\n", 
							ell1[Ib],emm1[Ib],ell2[Jb],emm2[Jb],creal(ovlpO),cimag(ovlpO), creal(ovlpN2),cimag(ovlpN2),dev2, creal(ovlpN),cimag(ovlpN),dev,
							(dev2<dev_TOL ? "":"XXX") );
				} else {
					for(I=0;I<2;I++)
						fprintf(fpA[I],"%d,%d x %d,%d : %16.8f %16.8f     %16.8f %16.8f\n", ell1[Ib],emm1[Ib],ell2[Jb],emm2[Jb],creal(ovlpO),cimag(ovlpO), creal(ovlpN),cimag(ovlpN));
				}
			}
			for(Ib=0;Ib<Nb1;Ib++)for(Jb=0;Jb<Nb1;Jb++){
				dcmplx ovlpO=Gaussian_ovlpNew(BS1[Ib],BS1[Jb],V,D);
				dcmplx ovlpN=nmrovlp1(BS1[Ib],BS1[Jb],V,D, Lshift, scalefacs_1,verbose);verbose=0; double dev=cabs(ovlpO-ovlpN);
				if(dev>dev_TOL){ dcmplx ovlpN2=nmrovlp1(BS1[Ib],BS1[Jb],V,D, Lshift, scalefacs_2,verbose);double dev2=cabs(ovlpO-ovlpN2);
					for(I=0;I<2;I++)
						fprintf(fpA[I],"%d,%d x %d,%d : %16.8f %16.8f     %16.8f %16.8f  %14.4e     %16.8f %16.8f  %14.4e  %s\n", 
							ell1[Ib],emm1[Ib],ell1[Jb],emm1[Jb],creal(ovlpO),cimag(ovlpO), creal(ovlpN2),cimag(ovlpN2),dev2, creal(ovlpN),cimag(ovlpN),dev,
							(dev2<dev_TINY ? "":"XXX") );
				} else {
					for(I=0;I<2;I++)
						fprintf(fpA[I],"%d,%d x %d,%d : %16.8f %16.8f     %16.8f %16.8f\n", ell1[Ib],emm1[Ib],ell1[Jb],emm1[Jb],creal(ovlpO),cimag(ovlpO), creal(ovlpN),cimag(ovlpN));
				}
			}
		}
	}
}
/*
int main(int narg,char **args){
	testovlp03();
	__assertf(0,(""),-1);
	double gamma=1.192;int elmax=32;double Vmu=0.8848;double scalefac[2]={10.0,1.0e-7};
	gamma=1.868e-3;Vmu=0.0;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
__assertf(0,(""),-1);	
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	Vmu=0.0;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	gamma=186.81603;Vmu=0.8848;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	gamma=186.81603;Vmu=-88.48;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	gamma=186.81603;Vmu=0.0;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	
	
	gamma=1.868e-3;Vmu=0.8848;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	gamma=1.868e-3;Vmu=-88.48;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	
	gamma=1.868e-3;Vmu=0.0;
	dbg_gaussian_integ1D_vcfield(gamma, elmax, Vmu, scalefac);
	
}
*/
/*
#nmrovlp1:-2,-2,-2  0.022791 0.000000 / 0.022791 0.000000        93.953125
#nmrovlp1:-2,-2,-1  0.014275 0.000000 / 0.014275 0.000000        177.406250
#nmrovlp1:-2,-2,0  -0.000766 0.000000 / -0.000766 0.000000       255.015625
#nmrovlp1:-2,-2,1  -0.005592 0.000000 / -0.005592 0.000000       336.000000
#nmrovlp1:-2,-2,2  -0.023420 0.000000 / -0.023420 0.000000       426.531250
#nmrovlp1:-2,-1,-2  -0.023266 0.000000 / -0.023266 0.000000      512.203125
#nmrovlp1:-2,-1,-1  -0.012829 0.000000 / -0.012829 0.000000      589.812500
#nmrovlp1:-2,-1,0  -0.001904 0.000000 / -0.001904 0.000000       659.687500
#nmrovlp1:-2,-1,1  0.007771 0.000000 / 0.007771 0.000000         733.093750
#nmrovlp1:-2,-1,2  0.021964 0.000000 / 0.021964 0.000000         814.343750
#nmrovlp1:-2,0,-2  -0.063879 0.000000 / -0.063879 0.000000       889.781250
#nmrovlp1:-2,0,-1  -0.039368 0.000000 / -0.039368 0.000000       957.437500
#nmrovlp1:-2,0,0  0.003706 0.000000 / 0.003706 0.000000          1019.968750
#nmrovlp1:-2,0,1  0.014125 0.000000 / 0.014125 0.000000          1085.515625
#nmrovlp1:-2,0,2  0.065280 0.000000 / 0.065280 0.000000          1154.546875
#nmrovlp1:-2,1,-2  -0.022656 0.000000 / -0.022656 0.000000       1237.468750
#nmrovlp1:-2,1,-1  -0.012453 0.000000 / -0.012453 0.000000       1307.984375
#nmrovlp1:-2,1,0  -0.001941 0.000000 / -0.001941 0.000000        1373.656250
#nmrovlp1:-2,1,1  0.007646 0.000000 / 0.007646 0.000000          1443.859375
#nmrovlp1:-2,1,2  0.021333 0.000000 / 0.021333 0.000000          1524.281250
#nmrovlp1:-2,2,-2  0.022991 0.000000 / 0.022991 0.000000         1613.875000
#nmrovlp1:-2,2,-1  0.014383 0.000000 / 0.014383 0.000000         1694.921875
#nmrovlp1:-2,2,0  -0.000736 0.000000 / -0.000736 0.000000        1762.500000
#nmrovlp1:-2,2,1  -0.005677 0.000000 / -0.005677 0.000000        1840.359375
#nmrovlp1:-2,2,2  -0.023599 0.000000 / -0.023599 0.000000        1932.046875
#nmrovlp1:-1,-2,-2  0.042703 0.000000 / 0.042703 0.000000        2016.187500
#nmrovlp1:-1,-2,-1  0.031863 0.000000 / 0.031863 0.000000        2093.140625
#nmrovlp1:-1,-2,0  -0.007493 0.000000 / -0.007493 0.000000       2161.937500
#nmrovlp1:-1,-2,1  -0.009055 0.000000 / -0.009055 0.000000       2235.312500
#nmrovlp1:-1,-2,2  -0.046791 0.000000 / -0.046791 0.000000       2315.062500
#nmrovlp1:-1,-1,-2  -0.037669 0.000000 / -0.037669 0.000000      2389.781250
#nmrovlp1:-1,-1,-1  0.003321 0.000000 / 0.003321 0.000000        2458.375000
#nmrovlp1:-1,-1,0  -0.025266 0.000000 / -0.025266 0.000000       2520.093750
#nmrovlp1:-1,-1,1  0.010825 0.000000 / 0.010825 0.000000         2585.593750
#nmrovlp1:-1,-1,2  0.031164 0.000000 / 0.031164 0.000000         2657.687500
#nmrovlp1:-1,0,-2  -0.119981 0.000000 / -0.119981 0.000000       2723.968750
#nmrovlp1:-1,0,-1  -0.136456 0.000000 / -0.136456 0.000000       2790.734375
#nmrovlp1:-1,0,0  0.059864 0.000000 / 0.059864 0.000000          2846.437500
#nmrovlp1:-1,0,1  0.027047 0.000000 / 0.027047 0.000000          2905.656250
#nmrovlp1:-1,0,2  0.139761 0.000000 / 0.139761 0.000000          2971.671875
#nmrovlp1:-1,1,-2  -0.036511 0.000000 / -0.036511 0.000000       3043.562500
#nmrovlp1:-1,1,-1  0.004328 0.000000 / 0.004328 0.000000         3110.500000
#nmrovlp1:-1,1,0  -0.025717 0.000000 / -0.025717 0.000000        3171.812500
#nmrovlp1:-1,1,1  0.010481 0.000000 / 0.010481 0.000000          3236.921875
#nmrovlp1:-1,1,2  0.029861 0.000000 / 0.029861 0.000000          3307.421875
#nmrovlp1:-1,2,-2  0.043001 0.000000 / 0.043001 0.000000         3386.843750
S 0
s 0       0.72567863       0.02733438           0.72567863       0.02733438
p-1      -0.00304181      -0.03121054          -0.00304181      -0.03121054
  0       0.17875212       0.00695143           0.17875212       0.00695143
  1       0.00701893       0.13296319           0.00701893       0.13296319
d-2       0.00986801      -0.00042223           0.00986801      -0.00042223
 -1      -0.00096694      -0.00469092          -0.00096694      -0.00469092
  0       0.04115833       0.00076271           0.04115833       0.00076271
  1       0.00249087       0.01995382           0.00249087       0.01995382
  2      -0.01994569       0.00050037          -0.01994569       0.00050037
f-3       0.00008330       0.00220569           0.00008330       0.00220569
 -2       0.00115139      -0.00009060           0.00115139      -0.00009060
 -1      -0.00018604      -0.00112178          -0.00018604      -0.00112178
  0       0.20310247       0.00693717           0.00520830       0.00001604  **********
  1       0.00050565       0.00477813           0.00050565       0.00477813
  2      -0.00233431       0.00012245          -0.00233431       0.00012245
  3      -0.00006275      -0.00267191          -0.00006275      -0.00267191

P-1
s 0       0.00325329      -0.03448750           0.00325329      -0.03448750
p-1      0.51566032       0.01931503           0.51566032       0.01931503
  0      0.00057914      -0.00810458           0.00057914      -0.00810458
  1      0.00732141       0.00019252           0.00732141       0.00019252
d-2      0.00331473       0.11054805           0.00331473       0.11054805
 -1      0.10969378       0.00430635           0.10969378       0.00430635
  0      0.00101278       0.01298341           0.00101278       0.01298341
  1      0.00106475      -0.00006868           0.00106475      -0.00006868
  2      0.00157633       0.02723146           0.00157633       0.02723146
f-3      -0.02161418       0.00032277          -0.02161418       0.00032277
 -2      0.00108465       0.01457141           0.00108465       0.01457141
 -1      0.02241526       0.00050502           0.02241526       0.00050502
  0      0.00139059      -0.00908310           0.00033732       0.00237287 ********
  1     -0.00250758       0.00006088          -0.00250758       0.00006088
  2      0.00044513       0.00357502           0.00044513       0.00357502
  3     -0.01092929       0.00030729          -0.01092929       0.00030729

P 0
s 0     -0.08471402      -0.00316966          -0.08471402      -0.00316966
p-1       0.00010108       0.00471321           0.00010108       0.00471321
  0      0.50588344       0.01883013           0.50588344       0.01883013
  1     -0.00004786      -0.02009866          -0.00004786      -0.02009866
d-2      -0.00168824       0.00001299          -0.00168824       0.00001299
 -1     -0.00167295      -0.02563798          -0.00167295      -0.02563798
  0      0.49403704       0.01827873           0.12539355       0.00494652   **********
  1      0.00343563       0.10927018           0.00343563       0.10927018
  2      0.00340212       0.00000638           0.00340212       0.00000638
f-3      -0.00001047      -0.00038538          -0.00001047      -0.00038538
 -2      0.00861869      -0.00027181           0.00861869      -0.00027181
 -1     -0.00093033      -0.03244343          -0.00060091      -0.00427762   ***********
  0      0.07601652       0.00217679           0.03480891       0.00050854   ***********
  1      0.00099172       0.13834101           0.00147269       0.01821131
  2     -0.01740387       0.00028638          -0.01740387       0.00028638
  3      0.00000693       0.00046601           0.00000693       0.00046601

P 1
s 0      -0.01105814       0.14728564          -0.01105814       0.14728564
p-1       0.00733735      -0.00023502           0.00733735      -0.00023502
  0     -0.00208807       0.03460294          -0.00208807       0.03460294
  1      0.48613695       0.01914558           0.48613695       0.01914558
d-2      -0.00165312      -0.02362096          -0.00165312      -0.02362096
 -1      0.00106828      -0.00015953           0.00106828      -0.00015953
  0     -0.00215288      -0.05532657          -0.00215288      -0.05532657
  1      0.10539519       0.00462722           0.10539519       0.00462722
  2      0.00341963       0.10617335           0.00341963       0.10617335
f-3       0.01021204      -0.00031902           0.01021204      -0.00031902
 -2     -0.00043718      -0.00314934          -0.00043718      -0.00314934
 -1     -0.00250714       0.00004284          -0.00250714       0.00004284
  0     -0.00437577       0.03882128          -0.00081715      -0.01010067
  1      0.03253814       0.00042890           0.03253814       0.00042890
  2      0.00108047       0.01405359           0.00108047       0.01405359
  3     -0.02106882       0.00033376          -0.02106882       0.00033376

D -2
s 0       0.00726549       0.00103395           0.00726549       0.00103395
p-1      -0.00771883       0.11029269          -0.00771883       0.11029269
  0      0.00158274       0.00019130           0.00158274       0.00019130
  1      0.00209738      -0.02412707           0.00209738      -0.02412707
d-2       0.34673723       0.01330298           0.34673723       0.01330298
 -1     -0.00101528       0.02296711          -0.00101528       0.02296711
  0     -0.00684684      -0.00007756          -0.00684684      -0.00007756
  1      0.00025051      -0.00514239           0.00025051      -0.00514239
  2     -0.00031264       0.00028207          -0.00031264       0.00028207
f-3       0.00129047       0.10930867           0.00129047       0.10930867
 -2      0.06035373       0.00268466           0.06035373       0.00268466
 -1     -0.00029456      -0.02429097          -0.00029456      -0.02429097
  0      0.00159200       0.00049161          -0.00123508       0.00009312
  1      0.00026051       0.00443403           0.00026051       0.00443403
  2     -0.00003799       0.00005075          -0.00003799       0.00005075
  3      0.00085189       0.02558858           0.00085189       0.02558858
*/