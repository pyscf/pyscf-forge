#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "macro_util.h"
#include "cstrutil.h"
#include "dalloc.h"
#include "dcmplx.h"
#include "xgaussope.h"
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
}

double z3diff(dcmplx ***a,dcmplx ***b,const int *Ndim,int *maxdiffloc,const char *title){
	int i,j,k;double maxdiff=cabs(a[0][0][0]-b[0][0][0]);int at[3]={0,0,0};double sqresum=0;
	for(i=0;i<Ndim[0];i++){for(j=0;j<Ndim[1];j++){for(k=0;k<Ndim[2];k++){
		double diff=cabs(a[i][j][k]-b[i][j][k]);if(diff>maxdiff){ maxdiff=diff;at[0]=i;at[1]=j;at[2]=k;}
		sqresum+=diff*diff;
	}}}
	if( maxdiffloc != NULL ){
		maxdiffloc[0]=at[0];maxdiffloc[1]=at[1];maxdiffloc[2]=at[2];
	}
	if( title !=NULL){
		printf("#z3diff:%s sqrediff %16.6e dist %16.6e [%d][%d][%d] %f+j%f / %f+j%f\n",
						title,sqresum,sqrt(sqresum),at[0],at[1],at[2], __CtoRI( a[at[0]][at[1]][at[2]] ),
						__CtoRI( b[at[0]][at[1]][at[2]] ) );                                            
	}
	return maxdiff;
}
dcmplx ***read_z3f(char *fpath,int *Ndim){
	FILE *fp=fopen(fpath,"r");__assertf( fp!=NULL,("fopen:%s fails",fpath),-1);int kk;
	const int llenmax=4096; char *line=NULL,*sbuf=(char *)malloc( (llenmax+1)*sizeof(char) );
	if( Ndim[0]<=0 ){
		while( (line=fgets(sbuf, llenmax, fp))!=NULL ){
			int le=strlen(line);if(le<1)continue;
			int off=0; while(off<le){ if(line[off]!=' ')break;off++;}
			if(off){ line+=off;le-=off;}
			if(startswith(line,"#Ndim:")){ int rank=0;
				int *Nd1=parse_ints(&rank,line+6);for(kk=0;kk<3;kk++)Ndim[kk]=Nd1[kk];i1free(Nd1); break;
			} 
			if(startswith(line,"#Z") && '0'<=line[2] && line[2]<='9'){ int rank=0;
				int *Nd1=parse_ints(&rank,line+2);for(kk=0;kk<3;kk++)Ndim[kk]=Nd1[kk];i1free(Nd1); break;
			}
		} fclose(fp);
		fp=fopen(fpath,"r");
	}
	__assertf( Ndim!=NULL,(""),-1);
	dcmplx *buf=z1alloc( Ndim[0]*Ndim[1]*Ndim[2]);
	int nGot=0;int nblank=0;int Iblc=0;
	while( (line=fgets(sbuf, llenmax, fp))!=NULL ){
		int le=strlen(line);
		if(le<1)continue;
		int off=0;
		while(off<le){ if(line[off]!=' ')break;off++;}
		if(off){ line=line+off;le-=off; if(le<1)continue;}
		if(line[0]=='#'){ printf("## reading off:%s\n",line);continue;}
		if(nblank>1)Iblc++; nblank=0;
		int ngot1=0,jj;
		double *dbuf=parse_doubles(&ngot1,line);__assertf( (ngot1%2==0),(""),-1);
    ngot1/=2;
		for(jj=0;jj<ngot1;jj++){ buf[nGot+jj]=dbuf[2*jj]+ _I*dbuf[2*jj+1];}
		nGot+=ngot1; free(dbuf);dbuf=NULL;
	}
	fclose(fp);__assertf((nGot==Ndim[0]*Ndim[1]*Ndim[2]),("nGot=%d / %dx%dx%d",nGot,Ndim[0],Ndim[1],Ndim[2]),-1);
	free(sbuf);sbuf=NULL;
	dcmplx ***ret=z3alloc( Ndim[0], Ndim[1], Ndim[2] );
	int i,j,k,ijk;
	for(i=0,ijk=0;i<Ndim[0];i++){ for(j=0;j<Ndim[1];j++){ for(k=0;k<Ndim[2];k++,ijk++){ ret[i][j][k]=buf[ijk];}}}
	z1free(buf);buf=NULL;
	return ret;
}


char *get_datetime_x(char *sbuf,const char *format){
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  // 4+2+2+1+2+2+2=15
  sprintf(sbuf,format,tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  return sbuf;
}
char *get_datetime(){
	                       //012345678901234567890123
	const char format[25]="%04d%02d%02d_%02d%02d%02d";
	char *sbuf=(char *)malloc( (15+1)*sizeof(char) );
	char *ret=get_datetime_x(sbuf,format);
	//printf("len=%d [13]=%c [14]=%c\n",strlen(sbuf),sbuf[13],sbuf[14]);
	return ret;
}

char *d1toa(const double *a,const int N){
	char *ret=(char *)malloc( N*(12+1) + 100);
	int off=0,i;
	for(i=0;i<N;i++) { sprintf(ret+off,"%12.6f ",a[i]);off+=13;}
	return ret;
}
char *write_z3(const char *fpath,dcmplx ***zbuf,const int *Ndim){
	FILE *fp=fopen(fpath,"w");char *sdum=NULL; int i,j,k;
	fprintf(fp,"### write_z3@cstrutil.c %s\n",(sdum=get_datetime()));free(sdum);
	fprintf(fp,"#Z%d,%d,%d\n",Ndim[0],Ndim[1],Ndim[2]);
	for(i=0;i<Ndim[0];i++){ for(j=0;j<Ndim[1];j++){ for(k=0;k<Ndim[2];k++){ 
		fprintf(fp,"%24.10e %24.10e      ",creal( zbuf[i][j][k] ),cimag( zbuf[i][j][k] ));
	} fprintf(fp,"\n");} fprintf(fp,"\n\n");}
	fclose(fp);
	return fpath;
}
char *z1toa(const dcmplx *a,const int N){
	char *ret=(char *)malloc( N*(27) + 200);
	int off=0,i;
	for(i=0;i<N;i++) { sprintf(ret+off,"%12.6f+j%12.6f ",creal(a[i]),cimag(a[i]));off+=27;}
	return ret;
}
char *i1toa(const int *a,const int N){
	char *ret=(char *)malloc( N*(10+1) + 100);
	int off=0,i;
	for(i=0;i<N;i++) { sprintf(ret+off,"%10d ",a[i]);off+=11;}
	return ret;
}

int read_file(const char *fpath,void **bufs,char *types,int *lengths,int bfsz){
	FILE *fp=fopen(fpath,"r");
	const int verbose=1;
	const int llenmax=20000;
	char *sbuf=(char *)malloc((llenmax+1)*sizeof(char));
	char *line=NULL;
	int nbuf=0;int nline=0;
	while( (line=fgets(sbuf, llenmax, fp))!=NULL ){
		int le=strlen(line);
		int at=0;char top=line[at];
		while(top==' ' && at<le){ at++;top=line[at];}
		if(1){ int offset=at+1;
			int i;int n;
			if(top=='D' || top=='Z'||top=='I'){
				for(i=offset;i<le;i++){ if(line[i]=='['){ offset=i+1;break;} }
				for(i=le-1;i>=0;i--){ if(line[i]==']'){ le=i;break;} }
			} else {
				for(i=offset;i<le;i++){ if(line[i]=='\t'){ offset=i+1;break;} }
			}
			if(top=='I' || top=='i'){
				int *ibuf=parse_ints(&n,line+offset);__assertf(n>0,("parse_ints;%d for %s",n,line+offset),-1);
				types[nbuf]=top;lengths[nbuf]=n;bufs[nbuf]=(void *)ibuf;nbuf++;
				if(verbose){ printf("#read_file:I:%d:%d %s\n",(nline++),n,i1toa(ibuf,n));}
		} else {
				double *dbuf=parse_doubles(&n,line+offset);
				if( top=='D' || top=='d' ){
					types[nbuf]=top;lengths[nbuf]=n;bufs[nbuf]=(void *)dbuf;nbuf++;
					if(verbose){ printf("#read_file:D:%d:%d %s\n",(nline++),n,d1toa(dbuf,n));}
				} else if( top=='Z' || top=='z'){
					__assertf( n%2==0,("read_file:odd n=%d",n),-1);
					n=n/2;
					dcmplx *zbuf=z1alloc(n);
					int k;for(k=0;k<n;k++){ zbuf[k]=dbuf[2*k]+_I*dbuf[2*k+1];}
					d1free(dbuf);dbuf=NULL;
					types[nbuf]=top;lengths[nbuf]=n;bufs[nbuf]=(void *)zbuf;nbuf++;
					if(verbose){ printf("#read_file:Z:%d:%d %s\n",(nline++),n,z1toa(zbuf,n));}
				} else if( top=='s' ){
					types[nbuf]=top; n=le-offset;
					char *sbuf=(char *)malloc( (n+1)*sizeof(char) );
					int k;for(k=0;k<n;k++)sbuf[k]=line[offset+k]; sbuf[n]='\0';
					types[nbuf]=top;lengths[nbuf]=(le-offset); nbuf++;
				}
			}
		}
	}
	fclose(fp);
	free(sbuf);sbuf=NULL;line=NULL;
	return nbuf;
}
int startswith(const char *str,const char *ptn){
	const int ls=strlen(str),lp=strlen(ptn);
	if(ls<lp) return 0;
	int j;
	for(j=0;j<lp;j++){ if(str[j]!=ptn[j])break;}
	if(j>=lp) return 1;
	return 0;
}
double *parse_doubles(int *pnGot,char *org){
	double *dbuf=NULL;
	int loop;int N=0;
	for(loop=0;loop<2;loop++){
	  int i=0, len=strlen(org), nGot=0;
	  while( i<len && !(__isDnumeric(org[i])) )i++;
	  // while( nGot<N && i<len ){
	  while( i<len ){
	    int j, igot; char c; double dum;
	    j=i; while( __isDnumeric(org[j]) && j<len )j++;
	    if( j<len ){ c=org[j]; org[j]='\0'; }
	    igot=sscanf(org+i,"%lf",&dum);   //fprintf(stderr,"%d : %d--%d >> %lf\n",nGot,i,j,dbuf[nGot]);
	    if(dbuf!=NULL){ dbuf[nGot]=dum; }
	    if(igot>0)nGot++;
	    if( j<len )org[j]=c;
	    i=j; while( i<len && !(__isDnumeric(org[i])) )i++;
	  }
	  if(loop==0){
  		dbuf=d1alloc(nGot);N=nGot;*pnGot=nGot;continue;
  	} 
	}
  return dbuf;

}
int *parse_ints(int *N,char *org){ // or parseInts_all
	int loop, i=0, len=strlen(org);
	long ldum=0;
	int *ibuf=NULL;
	int nGot=0;
	for(loop=0;loop<2;loop++){
		nGot=0;i=0;
		while( i<len && !(__isInumeric(org[i])) )i++;
		while( i<len ){
			int j, igot; char c;
			j=i; while( __isInumeric(org[j]) && j<len )j++;
			if( j<len ){ c=org[j]; org[j]='\0'; }
			if( loop ){
				igot=sscanf(org+i,"%ld",&ldum); 
				__assertf( igot,("genInts:000"),-1); 
				ibuf[nGot]=(int)ldum;//printf("#parse_ints:%d %d\n",nGot,ibuf[nGot]);
			}
			nGot++;//printf("#parse_ints:counting up:%d\n",nGot);
			if( j<len )org[j]=c;
			i=j; while( i<len && !(__isInumeric(org[i])) )i++;
		}
  	if( loop==0 ) { ibuf=i1alloc(nGot);*N=nGot;}
  }
	
	return ibuf;
}


void calcone(const char *fpath){
	int nAtm; double *Rnuc=NULL;    int *IZnuc=NULL;            int nDa;            int *distinctIZnuc=NULL;
  int *Nsh=NULL;  int *ell=NULL;  int *npGTO=NULL;      double *alph=NULL; double *cofs=NULL;
  int nAtmB;       double *RnucB=NULL;   int *IZnucB=NULL;           int nDb;            int *distinctIZnucB=NULL;
  int *NshB=NULL; int *ellB=NULL; int *n_cols=NULL;     int *npGTOB=NULL;  double *alphB=NULL;
  double *cofsB=NULL;    int spdm; double *BravisVectors=NULL; double *Vectorfield=NULL;int nKpoints;
  double *kvectors=NULL;       int Lx;int Ly;int Lz;  int nCGTO_2; int nCGTO_1;
  int verbose=1;
	FILE *fp=fopen(fpath,"r");
	int bflen=20000;
	char *line=NULL,*sbuf=(char *)malloc(bflen*sizeof(char));
	int nbuf=0;
	while( (line=fgets(sbuf, bflen, fp))!=NULL ){
		int le=strlen(line);
		int *ibuf=NULL;double *dbuf=NULL;dcmplx *zbuf=NULL;
		int at=0;char top=line[at];
		while(top==' ' && at<le){ at++;top=line[at];}
		int i,n,offset=at+1,nline=0;
		if(top=='D' || top=='Z'||top=='I'){
			for(i=offset;i<le;i++){ if(line[i]=='['){ offset=i+1;break;} }
			for(i=le-1;i>=0;i--){ if(line[i]==']'){ le=i;break;} }
		} else {
			for(i=offset;i<le;i++){ if(line[i]=='\t'){ offset=i+1;break;} }
		}
		if(top=='I' || top=='i'){
			ibuf=parse_ints(&n,line+offset);__assertf(n>0,("parse_ints;%d for %s",n,line+offset),-1);
			if(verbose){ printf("#read_file:I:%d:%d %s\n",(nline++),n,i1toa(ibuf,n));}
		} else {
			dbuf=parse_doubles(&n,line+offset);
			if( top=='D' || top=='d' ){
				if(verbose){ printf("#read_file:D:%d:%d %s\n",(nline++),n,d1toa(dbuf,n));}
			}
		}
		printf("#%d i:%p d:%p z:%p n=%d\n",nbuf,ibuf,dbuf,zbuf,n);fflush(stdout);
		     if(nbuf==0) { nAtm=ibuf[0];__assertf(ibuf!=NULL,("nAtm"),-1);}
		else if(nbuf==1) { Rnuc=dbuf;   __assertf(dbuf!=NULL,("Rnuc..%d",nbuf),-1);}
		else if(nbuf==2) { IZnuc=ibuf;  __assertf(ibuf!=NULL,("IZnuc..%d",nbuf),-1);}
		else if(nbuf==3) { nDa=ibuf[0]; __assertf(ibuf!=NULL,("nDa..%d",nbuf),-1);}
		else if(nbuf==4) { distinctIZnuc=ibuf;__assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==5) { Nsh=ibuf;   __assertf(ibuf!=NULL,("Nsh..%d",nbuf),-1);}
		else if(nbuf==6) { ell=ibuf;   __assertf(ibuf!=NULL,("ell..%d",nbuf),-1);}
		else if(nbuf==7) { npGTO=ibuf; __assertf(ibuf!=NULL,("npGTO..%d",nbuf),-1);}
		else if(nbuf==8) { alph=dbuf;  __assertf(dbuf!=NULL,("alph..%d",nbuf),-1);}
		else if(nbuf==9) { cofs=dbuf;   __assertf(dbuf!=NULL,("cofs..%d",nbuf),-1);}
		else if(nbuf==10) { nAtmB=ibuf[0]; __assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==11) { RnucB=dbuf;  __assertf(dbuf!=NULL,("RnucB..%d",nbuf),-1);}
		else if(nbuf==12) { IZnucB=ibuf; __assertf(ibuf!=NULL,("IZnucB..%d",nbuf),-1);}
		else if(nbuf==13) { nDb=ibuf[0]; __assertf(ibuf!=NULL,("nDb..%d",nbuf),-1);}
		
		else if(nbuf==14) { distinctIZnucB=ibuf;__assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==15) { NshB=ibuf;  __assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==16) { ellB=ibuf;  __assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==17) { n_cols=ibuf;__assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==18) { npGTOB=ibuf;__assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==19) { alphB=dbuf; __assertf(dbuf!=NULL,("Rnuc..%d",nbuf),-1);}
		else if(nbuf==20) { cofsB=dbuf; __assertf(dbuf!=NULL,("cofsB..%d",nbuf),-1);}
		else if(nbuf==21) { spdm=ibuf[0];__assertf(ibuf!=NULL,("spdm..%d",nbuf),-1);}
		else if(nbuf==22) { BravisVectors=dbuf;__assertf(dbuf!=NULL,("BravisVectors..%d",nbuf),-1);}
		else if(nbuf==23) { Vectorfield=dbuf;__assertf(dbuf!=NULL,("Vectorfield..%d",nbuf),-1);}
		
		else if(nbuf==24) { nKpoints=ibuf[0];__assertf(ibuf!=NULL,("nKpoints%d",nbuf),-1);}
		else if(nbuf==25) { kvectors=dbuf; __assertf(dbuf!=NULL,("kvectors..%d",nbuf),-1);}
		else if(nbuf==26) { Lx=ibuf[0];__assertf(ibuf!=NULL,("Lx:nbuf=%d",nbuf),-1);}
		else if(nbuf==27) { Ly=ibuf[0];__assertf(ibuf!=NULL,("Ly:nbuf=%d",nbuf),-1);}
		else if(nbuf==28) { Lz=ibuf[0];__assertf(ibuf!=NULL,("Lz:nbuf=%d",nbuf),-1);}
		else if(nbuf==29) { nCGTO_2=ibuf[0];__assertf(ibuf!=NULL,("nCGTO_1:nbuf=%d",nbuf),-1);}
		else if(nbuf==30) { nCGTO_1=ibuf[0];__assertf(ibuf!=NULL,("nCGTO_1:nbuf=%d",nbuf),-1);}
		nbuf++;
	}
	fclose(fp);fp=NULL;
	free(sbuf);sbuf=NULL;
	calc_pbc_overlaps01(nAtm,  Rnuc,    IZnuc,            nDa,            distinctIZnuc,
                      Nsh,  ell,  npGTO,      alph, cofs,
                      nAtmB,       RnucB,   IZnucB,           nDb,            distinctIZnucB,
                      NshB, ellB, n_cols,     npGTOB,  alphB,
                      cofsB,    spdm, BravisVectors, Vectorfield,nKpoints,
                      kvectors,       Lx,Ly,Lz,  nCGTO_2, nCGTO_1);
}


/*
int main(int narg,char **args){
	char fpath[160]="calc_pbc_overlaps01.in";
	calcone(fpath);
	printf("#calcone ends here\n");fflush(stdout);
	return 0;
	__assertf(0,("end here"),-1);
	int bfsz=31;
	void **bufs=(void **)malloc( bfsz*sizeof(void *) );
	int *lengths=i1alloc(bfsz);
	char *types=ch1alloc(bfsz);
	read_file(fpath, bufs,types,lengths,bfsz);
	int *ibuf=(int *)bufs[0];
	
	printf("i0:%d",ibuf[0]);fflush(stdout);
	int i0=ibuf[0];
	int *i3=(int *)bufs[3],*i4=(int *)bufs[4],*i10=(int *)bufs[10], *i26=(int *)bufs[26], *i27=(int *)bufs[27] , *i28=(int *)bufs[28];
	int *i13=(int *)bufs[13];
	int *i21=(int *)bufs[21], *i24=(int *)bufs[24], *i29=(int *)bufs[29], *i30=(int *)bufs[30];
	calc_pbc_overlaps01( i0,  (double *)bufs[1],   (int *)bufs[2],  i3[0], i4[0],
	                     (int *)bufs[5], (int *)bufs[6], (int *)bufs[7],   (double *)bufs[8],  (double *)bufs[9],
	                i10[0], (double *)bufs[11], (int *)bufs[12], i13[0],(int *)bufs[14],
	                (int *)bufs[15],     (int *) bufs[16],  (int *)bufs[17], (int *)bufs[18], (double *)bufs[19],
	                (double *)bufs[20],  i21[0], (double *)bufs[22], (double *)bufs[23], i24[0],
	                (double *)bufs[25],  i26[0], i27[0], i28[0],
	                i29[0], i30[0] );
}*/
/*                          const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
                          int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
                          const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
                          const double *cofsB,    int spdm, double *BravisVectors, double *Vectorfield,int nKpoints,
                          double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1){*/
