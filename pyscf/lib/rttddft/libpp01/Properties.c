#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "cstrutil.h"
#include "dalloc.h"
#include "dcmplx.h"
#include "macro_util.h"

//#include "futils.h"
//#include "flib.h"
//#include "utils.h"
//#include "cutils.h"
//#include "FortInt.h"
#include "Properties.h"
Properties *new_Properties(const char *fpath,int size){
	Properties *ret=(Properties *)malloc(sizeof(Properties));
	ret->size=size; ret->N=0;
	ret->keylist=(char **)malloc(size*sizeof(char *));int i;for(i=0;i<size;i++)ret->keylist[i]=NULL;
	ret->values=(char **)malloc(size*sizeof(char *));for(i=0;i<size;i++)ret->values[i]=NULL;
	ret->fpath=ch1clone(fpath);
	return ret;
}
void Properties_update(Properties *this, const char *key,const char *value){
	int i;
	for(i=0;i<this->N;i++){
		if(strcmp(this->keylist[i],key)==0)break;
	}
	if(i>=this->N){
		if(this->N>=this->size){ int newsize=this->size+20;
			char **newK=(char **)realloc(this->keylist,(newsize*sizeof(char *)));__assertf(newK!=NULL,("newK"),-1);
			this->keylist=newK;
			char **newV=(char **)realloc(this->values,(newsize*sizeof(char *)));__assertf(newV!=NULL,("newV"),-1);
			this->values=newV;
			this->size=newsize;
		}
		i=this->N; this->N=this->N+1;
		this->keylist[i]=ch1clone(key);
	}
	this->values[i]=ch1clone(value);
	return i;
}
char *Properties_getvalue(Properties *this,const char *key){
	int i;
	for(i=0;i<this->N;i++){
		if(strcmp(this->keylist[i],key)==0)break;
	}
	if(i>=this->N) return NULL;
	return this->values[i];
}
void Properties_setvalue(Properties *this, const char *key,const char *type,const void *value){
	char sbuf[50+1]="";
	if(type[0]=='i') { const int *p=(int *)value; sprintf(sbuf,"%d",p[0]);Properties_update(this,key,sbuf);return;}
	if(type[0]=='d') { const double *p=(double *)value; sprintf(sbuf,"%24.10e",p[0]);Properties_update(this,key,sbuf);return;}
	if(type[0]=='c') { const char *p=(char *)value; sprintf(sbuf,"%c",p[0]);Properties_update(this,key,sbuf);return;}
	if(type[0]=='z') { const dcmplx *p=(dcmplx *)value; sprintf(sbuf,"%24.10e %24.10e",__RandI(p[0]));Properties_update(this,key,sbuf);return;}
	if(type[0]=='s') { const char *p=(char *)value; Properties_update(this,key,p);return;}
	if(type[0]=='I') { int rank=0,*Ndim=parse_ints(&rank,type+1);__assertf(rank==1,(""),-1);
										 char *sdum=i1toa((int *)value,Ndim[0]);Properties_update(this,key,sdum);free(sdum);return;}
	if(type[0]=='D') { int rank=0,*Ndim=parse_ints(&rank,type+1);__assertf(rank==1,(""),-1);
										 char *sdum=d1toa((double *)value,Ndim[0]);Properties_update(this,key,sdum);free(sdum);return;}
	if(type[0]=='Z') { int rank=0,*Ndim=parse_ints(&rank,type+1);__assertf(rank==1,(""),-1);
										 char *sdum=z1toa((dcmplx *)value,Ndim[0]);Properties_update(this,key,sdum);free(sdum);return;}
	__assertf(0,("type:%s",type),-1);
}
int Properties_load(Properties *this, const char *fpath){
	const char *path=( fpath==NULL ? this->fpath:fpath );
	int llenmax=2048;char *sbuf=ch1alloc(llenmax+1),*line=NULL;
	FILE *fp=fopen(path,"r");int Nret=0;
	while( (line=fgets(sbuf,llenmax,fp))!=NULL ){
		int le=strlen(line);
		if(le<1)continue;
		int off=0;
		while(off<le){ if(line[off]!=' ')break; off++;}
		if(off){ le-=off;line+=off;}
		if(le<1)continue;
		if(line[0]=='#')continue;
		int k;
		for(k=0;k<le;k++)if(line[k]==':')break;
		__assertf(k<le,(""),-1);
		line[k]='\0';
		Properties_update(this, line, line+(k+1));Nret++;
	}
	fclose(fp);free(sbuf);sbuf=NULL;line=NULL;
	return Nret;
}
int Properties_save(Properties *this, const char *fpath){
	const char *path=( fpath==NULL ? this->fpath:fpath );
	FILE *fp=fopen(path,"w");
	int i;
	for(i=0;i<(this->N);i++){
		fprintf(fp,"%s:%s\n",this->keylist[i],this->values[i]);
	}
	fclose(fp);return this->N;
} 


