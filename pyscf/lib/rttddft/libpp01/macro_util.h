#ifndef MACRO_UTILH
#define MACRO_UTILH
#include <time.h>

#define _I1Free(buf)  {i1free(buf);buf=NULL;}
#define _D1Free(buf)  {d1free(buf);buf=NULL;}
#define _Z1Free(buf)  {z1free(buf);buf=NULL;}
#define _Ch1Free(buf)  {ch1free(buf);buf=NULL;}

#define _I2Free(buf)  {i2free(buf);buf=NULL;}
#define _D2Free(buf)  {d2free(buf);buf=NULL;}
#define _Z2Free(buf)  {z2free(buf);buf=NULL;}
#define _Ch2Free(buf)  {ch2free(buf);buf=NULL;}

#define __fprtdtme(fptr1732,strheader,strtrailer)  {time_t time_t_instance_2718 = time(NULL); struct tm struct_tm_instance_31415 = *localtime(&time_t_instance_2718);\
fprintf(fptr1732,"%s%04d%02d%02d%c%02d%02d%02d%s", strheader,struct_tm_instance_31415.tm_year + 1900, struct_tm_instance_31415.tm_mon + 1, struct_tm_instance_31415.tm_mday, '_', struct_tm_instance_31415.tm_hour, struct_tm_instance_31415.tm_min, struct_tm_instance_31415.tm_sec,strtrailer);}

//extern void fflush_(void);
// cimag(PBCovlps[kkp]) for kkp in range(nKpoints)
#define ___prtI2xx(msg,A,Ld,Ndim)   {printf("#prtI2xx:");printf msg;int ii,iixjj;for(ii=0,iixjj=0;ii<(Ld);ii++){int jj;for(jj=0;jj<Ndim[ii];jj++,iixjj++){ printf("%d ",A[ii][jj]);\
                                     int *jp=A[ii]+jj;__assertf( A[0]+iixjj == jp,("%p %p A[0]:%p",A[0]+iixjj,jp,A[0]),-1);} printf("\n");}}
#define ___prtD3xx(msg,A,Ld,Ndim,Mdim) {printf("#prtD3xx:");printf msg;int ii,jj,kk,iixjj;for(ii=0,iixjj=0;ii<(Ld);ii++){\
for(jj=0;jj<Ndim[ii];jj++,iixjj++){printf("#%d.%d:",ii,jj);for(kk=0;kk<Mdim[iixjj];kk++){ printf("%f ",A[ii][jj][kk]);} printf("\n");}}}                               
#define __maxval(retv,jjat,itrt,val_at_jj,jjINI,jjUPL)  {int itrt=jjINI; retv=val_at_jj; jjat=(jjINI);\
for( itrt=(jjINI)+1; itrt<(jjUPL); itrt++ ){ if( (val_at_jj) > retv){ jjat=itrt;retv=(val_at_jj);}}}
#define __minval(retv,jjat,itrt,val_at_jj,jjINI,jjUPL)  {int itrt=jjINI; retv=val_at_jj; jjat=(jjINI);\
for( itrt=(jjINI)+1; itrt<(jjUPL); itrt++ ){ if( (val_at_jj) < retv){ jjat=itrt;retv=(val_at_jj);}}}

#ifndef dmatr
 #define dmatr double
#endif
#ifndef imatr
 #define imatr int
#endif
#define __FLOOR(a) ((a)<0 ? (-((int)(-a))-1):(int)(a))
#define ___DBGLGR(msg)  {printf("#DBGLGR:"); printf msg;fflush(stdout);}
#define __spdf_to_el(c) ((c)=='s'?0:((c)=='p'?1:((c)=='d'?2:((c)=='f'?3:((c)=='g'?4:(printf("macro_util.h.spdf_to_el:illegal arg for spdf:%c\n",(c)),fflush(stdout),exit(1),-999) )))))
//#define __spdf_to_el(c) ((c)=='s'?0:((c)=='p'?1:((c)=='d'?2: (int)((c)-'f')+3)))
#define __el_to_spdf(l) ((l)<3 ? ((l)==0 ? 's':((l)==1 ? 'p':'d') ): (char)('f'+(char)((l)-3)))
/* #define __Enuc( ret, Z, R, Na, dum,ja,ka,kk) {\
   ret=0.0;for(ja=0;ja<(Na);ja++){ for(ka=0;ka<ja;ka++){ __sqrtdist(,A,B,L,kk)*/
#define __POWI( ret, v, N, kk)  { ret=1.0;for(kk=0;kk<N;kk++)ret*=v; }

#define __write_file(fnme,append,fpt,msg)  {FILE *fpt=fopen(fnme,((append) ? "a":"w"));fprintf msg;fclose(fpt);}
#define __lXm(l,m) ((l)*(l)+(m)+(l))
#define ___prtD1x2(msg,A,B,L)   { int jj;for(jj=0;jj<(L);jj++){printf msg;printf(":%d %f %f",jj,A[jj],B[jj]);} }
#define __seekNaNinf(ret,A,N)   { ret=0;for(ret=0;ret<(N);ret++){if(__isNaNInf(A[ ret ]))break;}if( ret >=(N)){ret=-1;} } 
#define __hasNaNinf(ret,A,N)   { ret=0;int k_hasnaninf;for(k_hasnaninf=0;k_hasnaninf<(N);k_hasnaninf++){if(__isNaNInf(A[ k_hasnaninf ]))break;}if( k_hasnaninf <(N)){ret=1;} } 
#define __d1dff_max(ret,at,A,B,N,kk) { at=0;ret=fabs(A[at]-B[at]);for(kk=0;kk<(N);kk++){ if(fabs(A[kk]-B[kk])>ret){ ret=fabs(A[kk]-B[kk]);at=kk;} } }
#define __count(ret,b_kk,N,kk) { ret=0; for(kk=0;kk<(N);kk++) { if( b_kk ) ret++; } }
#define __ctime  ((double)clock())/CLOCKS_PER_SEC
#define __isNaN(v) ( (v)!=(v) )
#define __isInf(v) ( ((v)+(v))==(v) && ( (v)>1 || (v)<(-1))  )
#define __isNaNInf(v)  ( __isNaN(v) || __isInf(v) )
#define __opt_dot(dum,A,B,N) { int __opt_dot_k; dum=0; for(__opt_dot_k=0;__opt_dot_k<(N);__opt_dot_k++){ dum+=A[__opt_dot_k]*B[__opt_dot_k]; } }
#define __isDigit(c)  ('0'<=(c) && (c)<='9') 
#define __isDnumeric(c)  ( __isDigit(c) || (c)=='-' || (c)=='+'|| (c)=='.' ||(c)=='E' ||(c)=='e')
#define __isDnumeric1(c,nsgn,ndot,nexpo)  ( __isDigit(c) || ((c)=='-' && (++nsgn)==1) || ((c)=='+'&& (++nsgn)==1)|| ((c)=='.' && (--ndot)==1) ||( (c)=='E' && (++nexpo)==1 && ( (nsgn=0)==0 ) ) || ( (c)=='e' && (++nexpo)==1 && ((nsgn=0)==0) ) )

#define __isInumeric(c)  ( __isDigit(c) || (c)=='-' || (c)=='+')
#define __isAlphabet(c)  ( 'a'<=(c)<='z' || 'A'<=(c)<='Z' )

#define __sumof(ret,Ajj,N,jj) { ret=0; for(jj=0;jj<(N);jj++){ ret+=Ajj; } }
#define __a1sum(ret,A,N,kk) { ret=0; for(kk=0;kk<(N);kk++) ret+=A[kk]; }
#define __maxv(ret,Akk,N,kk) { kk=0;ret=(Akk);for(kk=1;kk<(N);kk++){ if( (Akk) > ret )ret=(Akk); } }
#define __maxvAt(ret,at,Akk,N,kk) { kk=0;at=0;ret=(Akk);for(kk=1;kk<(N);kk++){ if( (Akk) > ret ){ ret=(Akk);at=kk; } } }
#define __minv(ret,Akk,N,kk) { kk=0;ret=(Akk);for(kk=1;kk<(N);kk++){ if( (Akk) < ret )ret=(Akk); } }
#define __minvAt(ret,at,Akk,N,kk) { kk=0;at=0;ret=(Akk);for(kk=1;kk<(N);kk++){ if( (Akk) < ret ){ ret=(Akk);at=kk; } } }

#define __c_toUpperCase(c)  ( ((c)>='a' && (c)<='z') ? (c)+('A'-'a'):(c) )
#define __c_toLowerCase(c)  ( ((c)>='A' && (c)<='Z') ? (c)-('A'-'a'):(c) )
#define __toUpperCaseN(ret,s,kk) { for(kk=0;kk<strlen(s);kk++)ret[kk]=__c_toUpperCase(s[kk]); ret[kk]='\0';}
#define __toLowerCaseN(ret,s,kk) { for(kk=0;kk<strlen(s);kk++)ret[kk]=__c_toLowerCase(s[kk]); ret[kk]='\0';}

#define __DbgFlush2(msg,arg)  {fprintf(stdout,msg,arg);fprintf(stderr,msg,arg);fflush(stdout);fflush(stderr);}

#define __sqrtNorm(S,A,L,kk)  {S=0;for(kk=0;kk<(L);kk++)S+=A[kk]*A[kk];S=sqrt(S);}
#define __sqreNorm(S,A,L,kk)  {S=0;for(kk=0;kk<(L);kk++)S+=A[kk]*A[kk];}
#define __PI     3.141592653589793238
#define __sqrtPI 1.772453850905516027

#define __foreach( ope_kk, N, kk )  for(kk=0;kk<(N);kk++){ ope_kk; }
#define __D1Free_Safe(F) { if(F!=NULL){ d1free(F); F=NULL;} }
#define __I1Free_Safe(F) { if(F!=NULL){ i1free(F); F=NULL;} }
#define __I2Free_Safe(F) { if(F!=NULL){ i2free(F); F=NULL;} }
#define __Free_Safe(F)   { if(F!=NULL){ free(F); F=NULL; }  } 
#define __d1dot(R,A,B,L,kk) { R=0;for(kk=0;kk<L;kk++)R+=A[kk]*B[kk];}
#define __d1sub(R,A,B,L,kk) { for(kk=0;kk<L;kk++)R[kk]=A[kk]-B[kk];}
#define __d1add(R,A,B,L,kk) { for(kk=0;kk<L;kk++)R[kk]=A[kk]+B[kk];}
#define __d1sqrdiff(R,A,B,L,kk) { R=0; for(kk=0;kk<L;kk++)R+=(A[kk]-B[kk])*(A[kk]-B[kk]);}

#define __sqredist(R,A,B,L,kk) { R=0; for(kk=0;kk<L;kk++)R+=(A[kk]-B[kk])*(A[kk]-B[kk]);}
#define __sqrtdist(R,A,B,L,kk) { R=0; for(kk=0;kk<L;kk++)R+=(A[kk]-B[kk])*(A[kk]-B[kk]); R=sqrt(R);}

#define __assertX( bool, msg, foo )  if( !(bool) ){ printf msg; foo; }

#define __assert_Dev0( v, shdb, w, e, msg ) { if( fabs((v)-(shdb))>(w) ){ printf("#assert_Dev0;%lf deviates from %lf by %e\n",v,shdb,fabs(v-shdb) );__assertf(0,msg,( ((e)>=0 && fabs((v)-(shdb))>(e)) ? (-1):1) ); } }
#define __DBGLGR( msg ) { printf msg; fflush(stdout); }
#define __seekMaxAbs(R,J,A,L,kk) { R=fabs(A[0]);J=0;for(kk=0;kk<L;kk++){ if(fabs(A[kk]>R)){J=kk;R=fabs(A[kk]);} } }
//                                   1676581944693513195484
//static char _abortAt_cbuf[32];
//#define __abortAt(msg) { for(kk=0;kk<999;kk++){ getarg_(&kk,-abortAT_);
#define __abort(msg) { printf("#c abort:Aborting..");printf msg; fflush(stdout); fflush(stderr); exit(1); }
#define __a1eqlAt(A,B,L,iat)  { for(iat=0;iat<L;iat++)if(A[iat]==B[iat])break; if(iat>=L)iat=-1; }
#define __a1NeqAt(A,B,L,iat)  { for(iat=0;iat<L;iat++)if(A[iat]!=B[iat])break; if(iat>=L)iat=-1; }
#define __checkNaNinf( lvl, ttl, A, L, kk )  { for(kk=0;kk<(L);kk++) if(__isNaNInf(A[kk])){ __assertf(0,("NaNinf@%s,%d:%e\n",ttl,kk,A[kk]),lvl)} }

#define __checkNaNinfx( lvl, ttl, A, L, kk )  { for(kk=0;kk<(L);kk++) if(__isNaNInf(A[kk])){\
 printf("#checkNaNinfx:%f @%d..",A[kk],kk);printf ttl;__assertf(0,("NaNinf@%d:%e\n",kk,A[kk]),lvl)} }

#define __CheckPoint(key,msg)  { char __CheckPoint_buffer=ch1alloc(64);\
                                if( cseekarg("-AbortAt",__CheckPoint_buffer,64) ){\
                                if(strcmp(key,__CheckPoint_buffer)==0 ){ printf("AbortAt CheckPoint:%s\n",key);__abort(msg);}\
                                else printf("#CheckPoint: passing through %s\n");}}
#define __CheckPt(key) __CheckPoint(key,(""));
#define ___prtI1(msg,A,L) { printf msg;int kkprti1; for(kkprti1=0;kkprti1<L;kkprti1++) printf("%d ",A[kkprti1]); printf("\n");}
#define __prtD1c( msg, A, N, kk ) { printf msg; for(kk=0;kk<(N);kk++) printf("%f ",A[kk]); printf("\n"); }
#define __prtD1cum( cum, msg, Ak, N, kk ) { cum=0.0; printf("#prtD1cum:#");printf msg;\
for(kk=0;kk<N;kk++){ cum+=Ak; printf("#prtD1cum:%f %f\n",Ak,cum);}}
#define __prtD1(msg,A,L,kk) { printf msg; for(kk=0;kk<L;kk++) printf("%f ",A[kk]); printf("\n");}
#define __prtD1x(msg,A,I0,I1,kk) { printf msg; for(kk=I0;kk<I1;kk++) printf("%f ",A[kk]); printf("\n");}
#define __prtD1r(msg,A,L,M,kk) { printf msg; for(kk=0;kk<L && kk<M;kk++) printf("%f ",A[kk]); if(L>M)printf("...%d-th:%f ",L-1,A[L-1]);printf("\n");}
#define __prtI1(msg,A,L,kk) { printf msg; for(kk=0;kk<L;kk++) printf("%d ",A[kk]); printf("\n");}
#define __prtI1D1(msg,frmt,A,D,L,kk) { printf msg; for(kk=0;kk<L;kk++) printf(frmt,A[kk],D[kk]); printf("\n");}
#define __prtD1_noEndl(msg,A,L,kk) { printf msg; for(kk=0;kk<L;kk++) printf("%f ",A[kk]); }
#define __prtI1_noEndl(msg,A,L,kk) { printf msg; for(kk=0;kk<L;kk++) printf("%d ",A[kk]); }
#define __fprtD1f(fp,msg,frmt,A,L,kk) { fprintf msg; for(kk=0;kk<L;kk++) fprintf(fp,frmt,A[kk]); fprintf(fp,"\n");}
#define __fprtD1(fp,msg,A,L,kk) { fprintf msg; for(kk=0;kk<L;kk++) fprintf(fp,"%f ",A[kk]); fprintf(fp,"\n");}
#define __fprtD1_2(fp,msg,A,L,B,N,kk) { fprintf msg; for(kk=0;kk<L;kk++)fprintf(fp,"%f ",A[kk]); for(kk=0;kk<N;kk++) fprintf(fp,"%f ",B[kk]); fprintf(fp,"\n");}
#define __fprtI1(fp,msg,A,L,kk) { fprintf msg; for(kk=0;kk<L;kk++) fprintf(fp,"%d ",A[kk]); fprintf(fp,"\n");}
#define __fprtC1(fp,msg,A,L,kk) { fprintf msg; for(kk=0;kk<L;kk++) fprintf(fp,"%c ",A[kk]); fprintf(fp,"\n");}
#define __assertf( b, msg, lvl )  if(!(b)){ printf("!%c c Assertion Failed:",(lvl<0 ? 'E':'W')); printf msg;  fflush(stdout); if( lvl<0){fprintf(stderr,"!E c Assertion Failed:");fflush(stderr);} if(lvl<0) exit(1); }
#define __a1clr(A,L,ii) { for(ii=0;ii<L;ii++)A[ii]=0; }
#define __a2clr(A,L,N,ii,jj) { for(ii=0;ii<L;ii++)for(jj=0;jj<N;jj++)A[ii][jj]=0; }
#define __a1cpy(A,B,L,ii) { for(ii=0;ii<L;ii++)B[ii]=A[ii]; }
#define __a2cpy(A,B,L,M,ii,jj) { for(ii=0;ii<(L);ii++){ for(jj=0;jj<(M);jj++){ B[ii][jj]=A[ii][jj];} } }
#define __prtFD2(msg,A,L,M,jj,kk) { printf("#prtFD2:");printf msg;printf("#>>>\n"); for(jj=0;jj<L;jj++){ printf msg; for(kk=0;kk<M;kk++)printf("%f ",A[jj][kk]); printf("\n"); } }
#define __fprtFD2(fp,msg,A,L,M,jj,kk) { fprintf(fp,"#prtFD2:");fprintf msg;fprintf(fp,"#>>>\n"); for(jj=0;jj<L;jj++){ fprintf msg; for(kk=0;kk<M;kk++)fprintf(fp,"%f ",A[jj][kk]); fprintf(fp,"\n"); } }
#define ___prtD1(msg,A,L)        { int kprtd1;   printf msg; for(kprtd1=0;kprtd1<L;kprtd1++) printf("%f ",A[kprtd1]); printf("\n");}
#define ___prtFD2(msg,A,L,M)     { int jprtfd2,kprtfd2;printf("#prtFD2:");printf msg;printf("#>>>\n"); for(jprtfd2=0;jprtfd2<L;jprtfd2++){ printf msg; for(kprtfd2=0;kprtfd2<M;kprtfd2++)printf("%f ",A[jprtfd2][kprtfd2]); printf("\n"); } }
#define ___fprtFD2(fp,msg,A,L,M) { int jprtfd2,kprtfd2;fprintf(fp,"#prtFD2:");fprintf msg;fprintf(fp,"#>>>\n"); for(jprtfd2=0;jprtfd2<L;jprtfd2++){ fprintf msg; for(kprtfd2=0;kprtfd2<M;kprtfd2++)fprintf(fp,"%f ",A[jprtfd2][kprtfd2]); fprintf(fp,"\n"); } }
#define __assertx( bool, msg, todo )  if( !(bool) ){\
fprintf(stderr,"!WW c assertX: assertion failed\n");\
printf("!WW c assertX: assertion failed\n");printf msg;\
todo;}
#define __Sum( Ret, Akk, N, kk)  { Ret=0;for(kk=0;kk<(N);kk++){ Ret+=(Akk); }}


#define __sumA1(R,A,L,kk) { R=0;for(kk=0;kk<(L);kk++)R+=A[kk];}
#define __updMin(i,v,iat,vmin) ( (vmin)>v ? (vmin=v,iat=i, 1):0 )
#define __updMax(i,v,iat,vmx) ( (vmx)<v ? (vmx=v,iat=i, 1):0 )
#define __prtSD2r(ttl,A,L,N,jj,kk,jjk) { printf ttl; for(jj=0,jjk=0;jj<L && jj<N;jj++){ for(kk=0;kk<=jj;kk++,jjk++)printf("%f ",A[jjk]); printf("\n");} }
#define __prtSD2(ttl,A,L,jj,kk,jjk) { printf ttl; for(jj=0,jjk=0;jj<L;jj++){ for(kk=0;kk<=jj;kk++,jjk++)printf("%f ",A[jjk]); printf("\n");} }
#define __checkRange(msg, a,i,b ) { __assertf( ( ((a)<=(i)) && ((i)<(b)) ), msg, -1); } 
#define __seekA1(k,va,A,L) { for(k=0;k<L;k++){if(A[k]==va)break;}if(k>=L)k=-1; }
#define __A1comp(k,A,B,L) { k=L-1;while( A[k]==B[k] && k>=0)k--; k=( k<0 ? 0:( A[k]<B[k] ? -1:1) );}
#define __A1eqv(iret,A,B,L,k) { for(k=0;k<L;k++)if(A[k]!=B[k])break; iret=(k>=L); }
#define __seekA2(i,vec,U,Nv,Ld,kk,idum) { for(i=0;i<Nv;i++){ __A1eqv(idum,vec,U[i],Ld,kk); if(idum)break;} if(i>=Nv)i=-1; }
#define __append_A2(TYPE, elmt,Ld, buf,len,bufsz,inc,dflt,jj,kk)  { \
if(len>=bufsz){ TYPE tmpbf1=NULL; tmpbf1=(TYPE)realloc(buf[0],(bufsz+inc)*Ld); free(buf); buf=(TYPE *)malloc( (bufsz+inc)*sizeof(TYPE)); \
                buf[0]=tmpbf1;for(jj=1;jj<(bufsz+inc);jj++)buf[jj]=buf[jj-1]+Ld;\
                for(jj=bufsz;jj<(bufsz+inc);jj++)for(kk=0;kk<Ld;kk++)buf[jj][kk]=dflt; bufsz=bufsz+inc; }\
for(kk=0;kk<Ld;kk++)buf[len][kk]=elmt[kk];len++;}
#define __print_A2( header, frmt, buf,len,Ld, jj, kk) { for(jj=0;jj<len;jj++) { printf header;for(kk=0;kk<Ld;kk++) printf( frmt, buf[jj][kk] );printf("\n"); } }
#define __NINT(v)  ( (v)>0 ? (int)((v)+0.50):(int)((v)-0.50) )
#define __cvt_fstr(dir,org,len,kk) { if(dir<0){ for(kk=len-1;kk>=0;kk--){if(org[kk]=='\0')org[kk]=' ';break;} }\
else{ for(kk=len-1;kk>=0;kk--)if(org[kk]!=' ')break; if(kk<len-1)org[kk+1]='\0'; } }

#define __a2fill(A,v,L,N,ii,jj) { for(ii=0;ii<L;ii++)for(jj=0;jj<N;jj++)A[ii][jj]=v; }
#define __a1fill(A,v,L,jj) { for(jj=0;jj<L;jj++)A[jj]=v; }
//#define __buffer_add(bf,sz,at,T,v) { if(bf==NULL){sz=100;at=0;bf=(T *)malloc(sz*sizeof(T));}if( at==sz ){bf=(T *)realloc( bf,(sz+100)*sizeof(T));} bf[at++]=v;} 
#define __MAX(a,b) ((a)<(b) ? (b):(a) )
#define __MIN(a,b) ((a)>(b) ? (b):(a) )
#define __ijAsym(i,j,sg)  ((j)<(i) ? (sg=1,((i)-1)*(i)/2+(j)) : (sg=-1,((j)-1)*(j)/2+(i)) )
#define __ijSyn(i,j) ( (j)<(i) ? (i)*((i)+1)/2+(j) : (j)*((j)+1)/2 + (i) )
#define __CHECK_DEV(msg,a,b) { if(fabs((a)-(b))>1.0e-07){ printf msg;printf("#CHECK_DEV:%f %f\n",a,b); } }
#define __DBGPRNT(msg) {  printf("###DBGPRNT>>>>\n");printf msg; fflush(stdout); }
#define __print_legend( content, bool )  if( bool ){ printf content;printf("\n"); }
#define __countChar( ret, src, c, kk) { ret=0; kk=-1; while( src[++kk]!='\0'){ if(src[kk]==c)ret++; } }
#define __pythag(a,b) (fabs(a)>fabs(b) ? fabs(a)*sqrt(1.0+(b/a)*(b/a)):\
((b)==0.0 ? 0.0:fabs(b)*sqrt(1.0+(a/b)*(a/b))) )

#endif
