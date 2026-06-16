import os
import sys
import numpy as np
from numpy import random
from .utils import i1eqb,i1toa,toString,a1maxdiff,d1toa,z1toa,d2toa,parse_ints
from .serialize import load_fromlist,parse_slist,read_strtype,serialize
from .Logger import Logger
from .rttddft_common import rttddft_common
import math
from mpi4py import MPI ### XXX idbgng_threadsync

# DM   'R'   [nkpt,nAO,nAO] / [nAO,nAO]
#      else  [2,nkpt,nAO,nAO]/[2,nAO,nAO]
# Fck  'U'   [2,nkpt,nAO,nAO]/[2,nAO,nAO]
#      else  
# S    [nkpt,nAO,nAO] / [nAO,nAO]
def dot_AOmatrices(a,b,Smat,spinrestriction,DorF,title=None):

    assert DorF=='D' or DorF=='F',""
    nmult=( (1 if(spinrestriction=='R') else 2) if(DorF=='D') else \
            (2 if(spinrestriction=='U') else 1) )
    Ndim_S=np.shape(Smat);le_Ndim_S=len(Ndim_S)
    pbc=(le_Ndim_S==3)
    nAO=Ndim_S[1]
    assert (pbc and Ndim_S[1]==Ndim_S[2]) or ((not pbc) and Ndim_S[0]==Ndim_S[1]), ""
    nkpt=(1 if(not pbc) else Ndim_S[0])
    Ndim_DorF=( (   [nkpt,nAO,nAO] if(pbc) else   [nAO,nAO]) if(nmult==1) else \
              ( [2,nkpt,nAO,nAO] if(pbc) else [2,nAO,nAO]) )
    A=np.reshape(a,Ndim_DorF);B=np.reshape(b,Ndim_DorF)
    cum=0.0;buf=[]
    for sp in range(nmult):
        As=(A if(nmult==1) else A[sp])
        Bs=(B if(nmult==1) else B[sp])
        dum=0.0
        for kp in range(nkpt):
            Ask=(As if(not pbc) else As[kp])
            Bsk=(Bs if(not pbc) else Bs[kp])
            Sk= (Smat if(not pbc) else Smat[kp])
            ASBS=np.matmul(Ask,np.matmul(Sk,np.matmul(Bsk,Sk)))
            tr=0.0
            for I in range(nAO):
                tr+=ASBS[I][I].real
            dum+=tr;buf.append(tr)
        if(pbc):
            dum=dum/float(nkpt)
        cum+=dum
    if(title is not None):
        print("dot_AOmatrices:%s "%(title)+str(Ndim_DorF))
        print("dot_AOmatrices:A:%f+j%f %f+j%f ..."%(A[0][0][0].real,A[0][0][0].imag,A[0][0][1].real,A[0][0][1].imag))
        print("dot_AOmatrices:B:%f+j%f %f+j%f ..."%(B[0][0][0].real,B[0][0][0].imag,B[0][0][1].real,B[0][0][1].imag))
        print("dot_AOmatrices:Smat:"+str(Ndim_S))
        print("dot_AOmatrices:Smat:%f %f %f ..."%( Smat[0][0][0].real, Smat[0][0][1].real, Smat[0][0][2].real))
        print("dot_AOmatrices:Retv:%f+j%f sqrt:%f ..."%(cum.real,cum.imag,np.sqrt(abs(cum.real))) + str(buf))
    return cum


def a1maxdiff(a,b):
    Ld=len(a)
    ret=abs( a[0]-b[0] );at=0
    for I in range(1,Ld):
        dum=abs(a[I]-b[I])
        if( ret<dum ):
            ret=dum; at=I
    return ret
def a1maxval(a,dict=None):
    Ld=len(a)
    ret=abs( a[0] );at=0
    for I in range(1,Ld):
        dum=abs(a[I])
        if( ret<dum ):
            ret=dum; at=I
    if( dict is not None ):
        dict["at"]=I;dict["dev"]=ret;
    return ret
def i1eqb(a,b):
    Na=len(a);Nb=len(b)
    if(Na==Nb):
        for I in range(Na):
            if(a[I]!=b[I]):
                return False
        return True
    return False
##  dot_product(self,'ervecs',a,b)
## diis.update_AOparams({'AOrep':True,'SAO':sao,'spinrestriction':,'DorF':'D'})
def dot_product(this,type,a,b):
    if(type=='ervecs'):
        if(this.AOparams_ is not None ):
            if( this.AOparams_['AOrep'] ):
                return dot_AOmatrices(a,b, this.AOparams_['SAO'], this.AOparams_['spinrestriction'], this.AOparams_['DorF'])
    return np.vdot(a,b)
def z1toa(A,format="%8.3f+j%8.3f "):
    ret=""
    N=len(A)
    for J in range(N):
        ret=ret+format%(A[J].real, A[J].imag)
    return ret;
def diag_tostring(matr):
    sh=np.shape(matr)
    N=min(sh[0],sh[1])
    ret="";
    if ( (type(matr[0][0]) is np.complex128) or (type(matr[0][0]) is complex) ):
        for J in range(N):
            ret+="%8.3f+j%8.3f "%(matr[J][J].real, matr[J][J].imag);
    else:
        for J in range(N):
            ret+="%8.3f "%(matr[J][J]);

    return ret
def sd2toa(matr,Ld=None,format="%10.4f ",delimiter="\n"):
    N=Ld;
    if( Ld is None ):
        N=len(matr)
    ret=""
    for I in range(N):
        for J in range(I+1):
            ret+=format%(matr[I][J])
        ret+=delimiter
    return ret

class DIIS01:
    @staticmethod
    def serialize_DIIS01(this,delimiter=';'):
        ret="type:"+str(type(this))
        ret+=delimiter+ serialize(this,delimiter=delimiter,verbose=0)
    @staticmethod
    def construct_DIIS01(string, delimiter=';'):
        if( string=="None" ):
            return None
        retv=None
        sbuf=string.split(delimiter)
        col0=sbuf.pop(0)
        sA=col0.split(':');
        key=sA[0].strip()
        assert (key=="type"),""
        strtype=sA[1].strip();
        ty=read_strtype(strtype)
        assert ty=="DIIS01",""
        types,values=parse_slist(sbuf)
        Ndim_vec=parse_ints( values["Ndim_vec"], delimiter=None )
        bfsz=int( values["bfsz"] )
        dtype=values["dtype"]
        ervec_thr=float( values["ervec_thr"] )
        retv=DIIS01(Ndim_vec,bfsz,dtype,ervec_thr)
        return load_fromlist(retv, sbuf)
    # suggested = new * (1 - self.mixing_factor) + old * self.mixing_factor
    def __init__(self,Ndim_vec,bfsz,dtype,ervec_thr=1.0e-3,nstep_thr=-1,mixing_factor=0.50,nvect_thr=5):
        self.Ndim_vec=tuple(Ndim_vec) + tuple()
        Ldvec=Ndim_vec[0]
        for I in range(1,len(Ndim_vec)):
            Ldvec=Ldvec*Ndim_vec[I]
        self.Ldvec=Ldvec
        self.ibuf=np.zeros([bfsz],dtype=int)
        self.vcbuf=np.zeros([bfsz,Ldvec],dtype=dtype)
        self.erbuf=np.zeros([bfsz,Ldvec],dtype=dtype)
        self.bfsz=bfsz
        self.nbuf=0
        self.matrix=np.zeros( [bfsz+1,bfsz+1], dtype=dtype )
            # at each step [I][J] with I>=J are filled 
        self.ervec_thr=ervec_thr
        self.nstep_thr=nstep_thr
        self.dtype=dtype
        self.started=False
        self.mixing_factor=mixing_factor
        self.nvect_thr = nvect_thr
        self.nSleep=0
        self.seqno=0
        self.writer=None   ##::Logger  must implement .append(text) and Logger.info( .writer, ...)
        self.iter=0
        self.check_linear_dependency=False
        self.AOparams_=None;
        self.dbgbuffer=[];
        self.retv_last=None
    def update_AOparams(self,dic,clear=True):
        if( self.AOparams_ is None ):
            self.AOparams_ = {}
        elif(clear):
            self.AOparams_.clear()
        for key in dic:
            self.AOparams_.update({key:dic[key]})
        string="";
        for key in self.AOparams_:
            if(key != 'SAO'):
                string+="%s:%s "%( str(key), str( self.AOparams_[key] ))
        rttddft_common.Write_once('DIIS.update_AOparams',"#DIIS_params set:"+string)

    def get_recent_sqrerrors(self, Nwant, nstepbefore):
        Iv0=self.nbuf-1-nstepbefore
        if( Iv0<0 ):
            return None
        nupl=min( Nwant, Iv0+1)
        ret=[]
        for J in range(nupl):
            Iv=Iv0-J
            Imod=Iv%self.bfsz
            ret.append( self.matrix[Imod][Imod] )
        return ret

    def check_buffer_02(self,Nkeep,order='E',nSleep=6):
        if( self.nbuf<2 ):
            print("#DIIS:check_buffer too few vecs:%d"%(self.nbuf));
            return 0
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        Nremov=nvect-Nkeep
        if( Nremov<=0 ):
            return 0
        removed=[];N=0
        if( order == 'E' ):
            erveclen =np.zeros( [nvect])
            for I in range(nvect):
                erveclen[I]=self.matrix[I][I]
            jthTOiv=np.argsort(erveclen)
            for jth in range(nvect):
                Iv=jthTOiv[nvect-jth-1]
                removed.append(Iv);
                N+=1
                if(N>=Nremov):
                    break
        else:
            nk=0
            Iv=(self.nbuf-1) % self.bfsz
            for I in range(nvect):
                if(nk>=Nkeep):
                    removed.append(Iv);
                else:
                    nk+=1
                Iv=( Iv - 1 + self.bfsz )%self.bfsz
        self.remove_vectors( removed, nSleep=nSleep, update_attributes=True)
        return len(removed)

    ## default=0,1,2.. : force apply Ith
    ##          -1,-2,               Nbranch+default th
    def check_buffer(self,dicts,N_recent=10,N_latest=4,logger=None,default=None):

        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        ## DEFAULT : N_recent=10; N_latest=4
        if( nvect < N_recent+N_latest ):
            return 0
        Iv=(self.bfsz + self.nbuf-1) % self.bfsz
        recent_min=self.matrix[Iv][Iv]
        for K in range(1,N_latest):
            Iv=(self.bfsz + self.nbuf-1-K)%self.bfsz
            recent_min= min( recent_min, self.matrix[Iv][Iv] )

        Iv=(self.bfsz + self.nbuf-1-N_recent) % self.bfsz
        old_min=self.matrix[Iv][Iv]
        for K in range(1,N_latest):           ## nbuf - 1-(N_latest-1) - N_recent = nbuf-(N_recent+N_latest) 
            Iv=(self.bfsz + self.nbuf-1-K-N_recent)%self.bfsz
            old_min= min( old_min, self.matrix[Iv][Iv] )
        recent_min=np.sqrt(recent_min);old_min=np.sqrt(old_min)

        Nbranch=len(dicts);branch=None
        for I in range(Nbranch-1):
            if( dicts[I]["condition"]( recent_min, old_min ) ):
                branch=I;break
        if( branch is None ):
            if( default is None ):
                return None
            Ith=(default if(default>=0) else Nbranch+default)
            if( Ith<0 or Ith>=Nbranch ):
                return None
            branch=Ith  ## last one
        Nkeep=dicts[branch]["Nkeep"];nSleep=dicts[branch]["nSleep"];order=dicts[branch]["order"];
        ratio=recent_min/old_min        
        Logger.write(logger,"#DIIS:check_buffer:step=%d:%d recent_min/old_min:%e (%e/%e) >> keep %d sleep %d ordering:%s"%(
                     self.iter,branch,ratio,recent_min,old_min,Nkeep,nSleep,order))
        Nremov=self.check_buffer_02(Nkeep,order=order,nSleep=nSleep)
        if( logger is not None):
            nvect=min( self.nbuf, self.bfsz) 
            logger.info("#DIIS.%d nbuf=%d"%(self.iter, nvect))
            strbuf=""
            for jv in range(nvect):
                vsqr=self.matrix[jv][jv]
                vsqr2=np.vdot( self.erbuf[jv], self.erbuf[jv] )
                strbuf+="#%d: %e %e %d\n"%( jv, np.sqrt(vsqr.real), np.sqrt(vsqr2.real),self.ibuf[jv] )
                assert (abs(vsqr-vsqr2)<1.0e-6),"check vsqr:"+str( [vsqr,vsqr2])
            strbuf+="\n\n"
            logger.info(strbuf)
        return Nremov

    def check_buffer_old(self,vnorm_abs_thr=1.0e-5,vnorm_relt_thr=1.0e-3,verbose=False,dicts=None,
                    nSleep=6,N_keep=None,N_remov=None,N_recent=10,N_latest=4, logger=None):
        ## check if error is decreasing 
        ## n=4  N=10
        ## [min. of --N_latest-- errors]   -  [ the same quant. N steps before ] 
        ## (i) decreasing by ratio > R_1 (:2)  >>> keep going on.. but we want to reduce IRRELEVANT ones if nvect is too large
        ## (ii)                    < R_1 (:2)  >>> reduce # of vectors 
        ## 
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        ## DEFAULT : N_recent=10; N_latest=4
        if( nvect < N_recent+N_latest ):
            return 0
        Iv=(self.bfsz + self.nbuf-1) % self.bfsz
        recent_min=self.matrix[Iv][Iv]
        for K in range(1,N_latest):
            Iv=(self.bfsz + self.nbuf-1-K)%self.bfsz
            recent_min= min( recent_min, self.matrix[Iv][Iv] )

        Iv=(self.bfsz + self.nbuf-1-N_recent) % self.bfsz
        old_min=self.matrix[Iv][Iv]
        for K in range(1,N_latest):           ## nbuf - 1-(N_latest-1) - N_recent = nbuf-(N_recent+N_latest) 
            Iv=(self.bfsz + self.nbuf-1-K-N_recent)%self.bfsz
            old_min= min( old_min, self.matrix[Iv][Iv] )
        recent_min=np.sqrt(recent_min);old_min=np.sqrt(old_min)
#12.08.001        diff_small=1.0e-3; diff_fair=1.0e-2; 
        diff_small=3.0e-3; diff_fair=1.0e-2; 
        order='E';Scheme_02=True
        ratio = recent_min / old_min
        Nremov=0

        if( Scheme_02 ):
            if( (ratio < 0.50 and recent_min<diff_fair) or recent_min<diff_small ):
#12.08.001            if( (ratio < 0.50 and recent_min<diff_fair) or (ratio<1.0 and recent_min<diff_small) ):
                Nkeep=30;nSleep=0
                Logger.write(logger,"#DIIS:check_buffer:step=%d recent_min/old_min:%e (%e/%e) >> keep %d by %s"%(self.nbuf,ratio,recent_min,old_min,Nkeep,order))
                Nremov=self.check_buffer_02(Nkeep,order=order,nSleep=nSleep)
#12.08.001            elif( ratio<0.50 or recent_min<diff_small ):
            elif( ratio<0.50 or recent_min<diff_fair ):
                Nkeep=20;nSleep=3
                Logger.write(logger,"#DIIS:check_buffer:step=%d recent_min/old_min:%e (%e/%e) >> keep %d by %s"%(self.nbuf,ratio,recent_min,old_min,Nkeep,order))
                Nremov=self.check_buffer_02(Nkeep,order=order,nSleep=nSleep)
            elif( ratio < 1.0 or recent_min<diff_fair ):
                Nkeep=10;nSleep=6
                Logger.write(logger,"#DIIS:check_buffer:step=%d recent_min/old_min:%e (%e/%e) >> keep %d by %s"%(self.nbuf,ratio,recent_min,old_min,Nkeep,order))
                Nremov=self.check_buffer_02(Nkeep,order=order,nSleep=nSleep)
            else:
                Nkeep=5;nSleep=8
                Logger.write(logger,"#DIIS:check_buffer:step=%d recent_min/old_min:%e (%e/%e) >> keep %d by %s"%(self.nbuf,ratio,recent_min,old_min,Nkeep,order))
                Nremov=self.check_buffer_02(Nkeep,order=order,nSleep=nSleep)
        else:
            if( ratio < 0.50 or (recent_min<diff_small) ):
                Logger.write(logger,"#DIIS:check_buffer:step=%d recent_min/old_min:%e (%e/%e) >> keep %d latest"%(self.nbuf,ratio,recent_min,old_min,N_recent))
                Nremov=self.check_buffer_01(vnorm_abs_thr=vnorm_abs_thr, vnorm_relt_thr=vnorm_relt_thr, verbose=verbose,\
                                        nSleep=nSleep,latestN_excluded=N_recent,N_keep=N_keep,N_remov=N_remov)
            elif( ratio < 1.0 or (recent_min<diff_fair)):
                Logger.write(logger,"#DIIS:check_buffer:step=%d recent_min/old_min:%e (%e/%e) >> keep %d latest"%(self.nbuf,ratio,recent_min,old_min,3))
                Nremov=self.check_buffer_01(vnorm_abs_thr=vnorm_abs_thr, vnorm_relt_thr=vnorm_relt_thr, verbose=verbose,\
                                        nSleep=nSleep,latestN_excluded=3,N_keep=N_keep,N_remov=N_remov)
            else:
                ## We'd better restart ??
                Logger.write(logger,"#DIIS:check_buffer:step=%d recent_min/old_min:%e (%e/%e) >> restart.."%(self.nbuf,ratio,recent_min,old_min))

                Nremov=self.check_buffer_01(vnorm_abs_thr=vnorm_abs_thr,vnorm_relt_thr=vnorm_relt_thr, verbose=verbose,\
                                            nSleep=nSleep,latestN_excluded=0,N_keep=N_latest,N_remov=None,err_upl=0.10,err_lwl=1.0e-3,Nkeep_upl=15)
##            Nremov=self.check_buffer_01(vnorm_abs_thr=vnorm_abs_thr,vnorm_relt_thr=vnorm_relt_thr, verbose=verbose,\
##                                        nSleep=nSleep,latestN_excluded=0,N_keep=N_latest,N_remov=None)
        if( logger is not None):
            nvect=min( self.nbuf, self.bfsz) 
            logger.info("#DIIS.%d nbuf=%d"%(self.iter, nvect))
            strbuf=""
            for jv in range(nvect):
                vsqr=self.matrix[jv][jv]
                vsqr2=np.vdot( self.erbuf[jv], self.erbuf[jv] )
                strbuf+="#%d: %e %e %d\n"%( jv, np.sqrt(vsqr.real), np.sqrt(vsqr2.real),self.ibuf[jv] )
                assert (abs(vsqr-vsqr2)<1.0e-6),"check vsqr:"+str( [vsqr,vsqr2])
            strbuf+="\n\n"
            logger.info(strbuf)
        return Nremov

    def check_linear_independency1(self,vect,vnorm_abs_thr=-1,vnorm_relt_thr=1.0e-3,logger=None):
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        vc1=np.ravel(vect).copy()
        vnorm_org= np.sqrt( np.vdot(vc1,vc1).real )
        cofs=[]
        for ivect in range(nvect):
            ref=self.vcbuf[ivect]
            prd = np.vdot( ref, vc1); cofs.append(prd)
            vsqr = np.vdot(ref,ref)
            vnorm= math.sqrt( vsqr.real )
            vc1 = vc1 - (prd/vnorm)*ref
        vsqr=np.vdot(vc1,vc1)
        vnorm=math.sqrt( vsqr.real );
        vnratio=vnorm/vnorm_org
        if( ( vnorm_abs_thr>=0  and vnorm<vnorm_abs_thr ) or 
            ( vnorm_relt_thr>=0 and vnratio<vnorm_relt_thr) ):
            Logger.write(logger,"#check_linear_independency1:vnorm %e <- %e LINDP."%(vnorm,vnorm_org));
            return 0
        else:
            return 1

    def remove_linear_dependency(self, vnorm_relt_thr=1.0e-3, vnorm_abs_thr=-1, err_upl=None, logger=None):
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        excl=[]

        erveclen =np.zeros( [nvect])
        for I in range(nvect):
            erveclen[I]=self.matrix[I][I]

        jthTOiv=np.argsort(erveclen)

        vNORM_min=-1.0; vRELT_min=-1.0
        vcs=[]; n_tooLarge=0; jth_removed=[]; ## Schmidt orth
        jth=0; Iv=jthTOiv[jth]; vcs.append( self.vcbuf[Iv] ) 
        while(jth<nvect-1):
            jth+=1;Iv=jthTOiv[jth];
            vc1=self.vcbuf[Iv].copy() ## copy
            vnorm_org= np.sqrt( np.vdot(vc1,vc1).real )
            if( err_upl is not None ):
                if( vnorm_org >err_upl):
                    n_tooLarge+=1;jth_removed.append(jth);continue
            cofs=[]
            for ref in vcs:
                prd = np.vdot( ref, vc1); cofs.append(prd)
                vsqr = np.vdot(ref,ref)
                vnorm= math.sqrt( vsqr.real )
                vc1 = vc1 - (prd/vnorm)*ref
            vsqr=np.vdot(vc1,vc1)
            vnorm=math.sqrt( vsqr.real );
            vnratio=vnorm/vnorm_org
            if( ( vnorm_abs_thr>=0  and vnorm<vnorm_abs_thr ) or 
                ( vnorm_relt_thr>=0 and vnratio<vnorm_relt_thr) ):
                jth_removed.append(jth)
            else:
                if (vNORM_min<0): 
                    vNORM_min=vnorm
                else:
                    vNORM_min=min(vNORM_min,vnorm);
                if(vRELT_min<0):
                    vRELT_min=vnratio
                else:
                    vRELT_min=min(vRELT_min,vnratio)
        Nremov=len(jth_removed)
        if( Nremov <=0 ):
            Logger.write(logger,"#DIIS_check_buffer:removing:0 vectors:vNORM_min:%e vRELT_min:%e:"%(vNORM_min,vRELT_min))
            return 0;
        removed=[]
        for jth in jth_removed:
            Iv=jthTOiv[jth]
            removed.append(Iv)
        self.remove_vectors(removed)
        return Nremov
    def check_if_stuck4(self, pmset4=[ {"width":3,"nstep_back":10, "relative_diff_thr":1.0e-2}],text=None,strbuf=None):
        if( strbuf is not None ):
            strbuf.clear()
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        istuck=0;Ipms=0
        for pms4 in pmset4:
            Ipms+=1
            if( nvect < pms4["width"] + pms4["nstep_back"]):
                continue
            ## ermin of most recent 
            Iv =(self.nbuf-1 + self.bfsz) % self.bfsz
            ermax=self.matrix[Iv][Iv].real; ermin=self.matrix[Iv][Iv].real;Iv_at=Iv
            for I in range(1, pms4["width"] ):
                Iv=(self.nbuf-1-I + self.bfsz) % self.bfsz
                ermax=max(ermax,self.matrix[Iv][Iv].real); 
                if( ermin>self.matrix[Iv][Iv].real):
                    ermin=self.matrix[Iv][Iv].real; Iv_at=Iv
            ermin=np.sqrt(ermin)

            Iv =(self.nbuf-1 -pms4["nstep_back"] + self.bfsz) % self.bfsz
            ermax_2=self.matrix[Iv][Iv].real; ermin_2=self.matrix[Iv][Iv].real;Iv2_at=Iv;
            for I in range(1, pms4["width"] ):
                ## index (bf. modulo) runs up to nbuf-pms4["width"]-pms4["nstep_back"], which must be >=0 ...
                Iv=(self.nbuf-1-I-pms4["nstep_back"] + self.bfsz) % self.bfsz
                ermax_2=max(ermax_2,self.matrix[Iv][Iv].real); 
                if( ermin_2>self.matrix[Iv][Iv].real):
                    ermin_2=self.matrix[Iv][Iv].real; Iv2_at=Iv
            ermin_2=np.sqrt(ermin_2)

            if( "relative_diff_thr" in pms4 ):
                test = (ermin_2 - ermin)/ 0.50*(ermin + ermin_2)  ## >0:improving
                
                string="#check_if_stuck4:"+("!W STUCK" if(test< pms4[ "relative_diff_thr" ]) else "")\
                        +("" if(text is None) else text+":")\
                        +" recent_min:[%d]%e / %d steps before:[%d]%e"%( Iv_at, ermin, pms4["nstep_back"], Iv2_at, ermin_2 )
                if( strbuf is not None ):
                    strbuf.append(string)
                if( test< pms4[ "relative_diff_thr" ] ):
                    istuck=Ipms;
                    print(string);
                    break
                if(text is not None):
                    print(string);
        return istuck;
 
    def check_if_stuck( self, pmset1=[ {"nstep_thr":6,"relative_diff_thr":1.0e-2} ],\
                              pmset2=[ {"nstep_avg":4, "nstep_back":10, "relative_diff_thr":1.0e-2} ],\
                              pmset3=[ {"nstep_thr":50,"diff_thr":5.0e-5} ]):
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        ## {"nstep_thr":6,"relative_diff_thr":1.0e-2}
        istuck=0
        Ipms1=0
        for pms1 in pmset1:
            Ipms1+=1
            Iv=(self.nbuf-1 + self.bfsz) % self.bfsz
            ermax=self.matrix[Iv][Iv].real; ermin=self.matrix[Iv][Iv].real
            for I in range(1, pms1["nstep_thr"] ):
                Iv=(self.nbuf-1-I + self.bfsz) % self.bfsz
                ##for J in range(I):
                ##    Jv = (self.nbuf-1-J + self.bfsz) % self.bfsz
                ermax=max( ermax, self.matrix[Iv][Iv].real )
                ermin=min( ermin, self.matrix[Iv][Iv].real )
            ermax=np.sqrt(ermax); ermin=np.sqrt(ermin)
            relative_diff = (ermax-ermin)/ 0.50*(ermax + ermin)
            if( relative_diff < pms1["relative_diff_thr"] ):
                istuck = Ipms1; break
        if( istuck > 0 ):
            return istuck
        ## Test of avg-err : {"nstep_avg":4, "nstep_back":10, "relative_diff_thr":1.0e-2 }     
        Ipms2=0
        for pms2 in pmset2:
            Ipms2+=1
            if( nvect < pms2["nstep_avg"] + pms2["nstep_back"]):
                continue
            else:
                avgC=0.0;navgC=0
                for I in range(1, pms2["nstep_avg"] ):
                    Iv=(self.nbuf-1-I + self.bfsz) % self.bfsz
                    avgC+=self.matrix[Iv][Iv];navgC+=1
                avgC=avgC/float(navgC)

                avgP=0.0;navgP=0
                for I in range(1, pms2["nstep_avg"] ):
                    Iv=(self.nbuf-1-I -pms2["nstep_back"] + self.bfsz) % self.bfsz
                    avgP+=self.matrix[Iv][Iv];navgP+=1
                avgP=avgP/float(navgP)
                relative_diff = abs(avgC-avgP)/ 0.50*(abs(avgC) + abs(avgP))
                if( relative_diff < pms2["relative_diff_thr"] ):
                    istuck = (pms2+1)*10; break
                    logger.info("#check_2:%d_avg_err current:%e  %d steps ago:%e relative diff:%e < thr:%e"%(\
                         pms2["nstep_avg"],avgC,pms2["nstep_back"],avgP,relative_diff,pms2["relative_diff_thr"]))
                    istuck = Ipms2*100; break
        if( istuck > 0 ):
            return istuck
        ## check if you've reached almost converging result long ago but are still iterating
        ## {"nstep_thr":50,"diff_thr":5.0e-5}
        Ipms3=0
        for pms3 in pmset3:
            Ipms3+=1
            if( (self.bfsz < pms3["nstep_thr"]) or (nvect < pms3["nstep_thr"]) ):
                continue
            else:
                Istep1st=None;IstepLast=None;Ibuf=[]
                err1st=None;errLast=None
                sqrTHR=pms3["diff_thr"]**2
                for I in range(nvect):
                    J=nvect-I-1
                    Jv=(self.nbuf-1-J + self.bfsz) % self.bfsz
                    errSQR=self.matrix[Iv][Iv];
                    if( errSQR < sqrTHR):
                        if(Istep1st is None):
                            Istep1st=I; err1st=errSQR
                        IstepLast=I;errLast=errSQR; Ibuf.append(I)
                if( Istep1st is not None):
                    if( IstepLast-Istep1st >= pms3["nstep_thr"]):
                        logger.info("#check_3:iteration first reaches %e at %d:(%d):%e but is still %e at %d:(%d) and for %d more steps..."%(\
                             pms3["diff_thr"],Istep1st,np.sqrt(err1st),np.sqrt(errLast),IstepLast,len(Ibuf)-2))
                        istuck = Ipms3*100; break
        return istuck
    ## scheme-01b: removes those with error > err_upl (~0.05)
    ##             keep all vecs with error < err_lwl (~2e-3) up to  N_keep                         
    ## Scheme-01: (i) removes linearly dependent ones TAKING ACCOUNT of smallness of error
    ##            (ii) if(N_remov is not None): removes further up to N from those with largest errors 
    def check_buffer_01(self,vnorm_abs_thr=1.0e-5,vnorm_relt_thr=1.0e-3,verbose=False,nSleep=6,N_keep=20,Nkeep_upl=None,\
                        latestN_excluded=0,N_remov=None, err_upl=None, err_lwl=None):
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)

        excl=[]
        Iv=(self.nbuf-1) % self.bfsz
        for I in range(latestN_excluded):
            excl.append(Iv);
            Iv=( Iv - 1 + self.bfsz )%self.bfsz
            #for I in range(nvect):
            #    if( self.matrix[I][I].real < err_lwl )
        if( self.nbuf<2 ):
            print("#DIIS:check_buffer too few vecs:%d"%(self.nbuf));return 0
        erveclen =np.zeros( [nvect])
        for I in range(nvect):
            erveclen[I]=self.matrix[I][I]
        jthTOiv=np.argsort(erveclen)

        Nexcl_MAX=nvect
        if( N_remov is not None ):
            Nexcl_MAX=nvect - N_remov
        elif( Nkeep_upl is not None ):
            Nexcl_MAX=Nkeep_upl
        Nexcl_MAX-=len(excl)

        if( err_lwl is not None):
            sqr_lwl=err_lwl**2
            nlwr=0;nadd=0
            for jth in range(nvect):
                iv=jthTOiv[jth]
                vsqr = erveclen[ iv ]
                if( vsqr >sqr_lwl ):
                    break
                if( iv in excl ):
                    continue
                nlwr+=1
                if( nlwr<= Nexcl_MAX):
                    excl.append(iv);nadd+=1
            Logger.write(self.writer,"#err_lwl:%e nlwr:%d add:%d"%(err_lwl,nlwr,nadd)) 

        vNORM_min=-1.0;vRELT_min=-1.0
        removed=[];vnorms=[]
        vcs=[]; ## Schmidt orth
        jth=0; Iv=jthTOiv[jth]; vcs.append( self.vcbuf[Iv] ) 
        n_tooLarge=0
        while(jth<nvect-1):
            jth+=1;Iv=jthTOiv[jth];
            vc1=self.vcbuf[Iv].copy() ## copy
            vnorm_org= np.sqrt( np.vdot(vc1,vc1).real )
            if( err_upl is not None ):
                if( vnorm_org >err_upl):
                    n_tooLarge+=1;removed.append(jth);continue
            cofs=[]
            for ref in vcs:
                prd = np.vdot( ref, vc1); cofs.append(prd)
                vsqr = np.vdot(ref,ref)
                vnorm= math.sqrt( vsqr.real )
                vc1 = vc1 - (prd/vnorm)*ref
            vsqr=np.vdot(vc1,vc1)
            vnorm=math.sqrt( vsqr.real );vnorms.append(vnorm)
            vnratio=vnorm/vnorm_org
            assert (vnratio<=1.0),"vnratio"
            if( ( vnorm_abs_thr>=0  and vnorm<vnorm_abs_thr ) or 
                ( vnorm_relt_thr>=0 and vnratio<vnorm_relt_thr) ):
                if( not (Iv in excl) ):
                    if(verbose):
                        print("#Vector:%d vnorm AF:%e <- BF:%e ratio:%e"%(Iv,vnorm,vnorm_org,vnratio))
                    removed.append(jth)
                else:
                    if(verbose):
                        print("#Vector:%d vnorm AF:%e <- BF:%e ratio:%e ... but we KEEP THIS"%(Iv,vnorm,vnorm_org,vnratio))
            else:
                if (vNORM_min<0): 
                    vNORM_min=vnorm
                else:
                    vNORM_min=min(vNORM_min,vnorm);
                if(vRELT_min<0):
                    vRELT_min=vnratio
                else:
                    vRELT_min=min(vRELT_min,vnratio)

        if(n_tooLarge>0):
            Logger.write(self.writer,"#err_upl:%e nExcl:%d"%(err_upl,n_tooLarge))

        Ndel=len(removed)
        Ndel_lnDP=Ndel
        if( N_keep is not None ):
            assert ( (N_remov is None) or (N_remov<0) ),""
            N_remov = nvect - N_keep
        Ndel_2=0
        if( N_remov is not None):
            if( Ndel < N_remov ):
                for kth in range(nvect):
                    jth=nvect-1-kth
                    if( jth in removed ):
                        continue
                    Iv=jthTOiv[jth]
                    if( Iv in excl ):
                        continue
                    removed.append(jth);Ndel=Ndel+1;Ndel_2+=1
                    if( Ndel>=N_remov ):
                        break
        if( Ndel>0 ):
            text="#DIIS_check_buffer:removing:%d(lnDP:%d add:%d) vectors:jth:"%(Ndel,Ndel_lnDP,Ndel_2)+i1toa(removed)
            print(text)
            if( self.writer is not None):
                self.writer.append(text)
            Iv_removed=[]
            for jth in removed:
                Iv_removed.append( jthTOiv[jth] )
            self.remove_vectors(Iv_removed)
            return Ndel
        else:
            text="#DIIS_check_buffer:removing:0 vectors:vNORM_min:%e vRELT_min:%e:"%(vNORM_min,vRELT_min)
            print(text)
            if( self.writer is not None):
                self.writer.append(text)
            return 0
    def remove_vectors(self,removed,nSleep=6,update_attributes=True):
        if( len(removed)<=0 ):
            return 0
        vcbuf=np.zeros([self.bfsz,self.Ldvec],dtype=self.dtype)
        erbuf=np.zeros([self.bfsz,self.Ldvec],dtype=self.dtype)
        ibuf=np.zeros([self.bfsz],dtype=int)
        nINDP=0
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        for I in range(nvect):
            if( I in removed ):
                ## print("removing vec:%d .."%(I))
                continue
            for K in range(self.Ldvec):
                vcbuf[nINDP][K]=self.vcbuf[I][K]
                erbuf[nINDP][K]=self.erbuf[I][K]
            ibuf[nINDP]=self.ibuf[I]
            ## print("Keeping vec:%d as %d th"%(I,nINDP))
            nINDP+=1
        print("#check_buffer:checked:%d removed:%d Nremaining:%d:"%(nvect,len(removed),nINDP)+i1toa(ibuf,N=nINDP))
        self.vcbuf=vcbuf
        self.erbuf=erbuf
        self.ibuf =ibuf
        self.nbuf=nINDP
        for I in range(nINDP):
            for j in range(I+1):
                self.matrix[I][j]=dot_product(self,'ervecs',self.erbuf[I], self.erbuf[j])
        ## print("#AF:"+sd2toa( self.matrix, Ld=nINDP, format="%8.2e ",))
        if( update_attributes ):
            self.started=False;self.nSleep=nSleep
        return len(removed)
    ## Here we remove linearly dependent vectors..
    def check_buffer_00(self,vnorm_thr=1.0e-5,debug=False,vnorm_relt_thr=1.0e-3,verbose=False,nSleep=6):
        print("#DIIS:check_buffer nBuf:%d / size:%d"%(self.nbuf,self.bfsz))
        if( self.nbuf<2 ):
            print("#DIIS:check_buffer too few vecs:%d"%(self.nbuf));return 0
        nvect=(self.nbuf if( self.nbuf< self.bfsz) else self.bfsz)
        removed=[];vnorms=[]
        vcs=[];
        jth=0;Iv=nvect-jth-1;Iv=jth; vcs.append( self.vcbuf[Iv] ) ## zero-th
        while (jth<nvect-1):
            jth+=1;Iv=nvect-jth-1;Iv=jth;   ## runs from 1 to nvect-1 ...
            vc1=self.vcbuf[Iv].copy() ## copy
            vnorm_org= np.sqrt( np.vdot(vc1,vc1) )
            cofs=[]
            for ref in vcs:
                prd = np.vdot( ref, vc1); cofs.append(prd)
                vsqr = np.vdot(ref,ref)
                vnorm= math.sqrt( vsqr.real )
                vc1 = vc1 - (prd/vnorm)*vc1
            vsqr=np.vdot(vc1,vc1)
            vnorm=math.sqrt( vsqr.real );vnorms.append(vnorm)
            if( vnorm<vnorm_thr ):
                if(verbose):
                    print("#Vector:%d vnorm AF:%e <- BF:%e ratio:%e"%(Iv,vnorm,vnorm_org,vnorm/vnorm_org))
                if( vnorm/vnorm_org < vnorm_relt_thr ):
                    if(verbose):
                        print("#Vector:%d is linearly dependent:Norm after projecting out %d vecs is %e"%(Iv,len(vcs),vnorm),end="")
                    print(cofs);removed.append(Iv)
                    continue
            else:
                vcs.append( self.vcbuf[Iv] )
        ### if( debug and (len(removed)==0) ):
        ###    print("removing 0 and 3 ...");removed.append(0);
        ###    if( nvect>3 ):
        ###        removed.append(3);
        ###    print("#BF:"+sd2toa( self.matrix, Ld=nvect))
        if( len(removed)==0 ):
            print("#check_buffer:checked %d vnorm_min:%e"%(nvect,min(vnorms)) );return 0
        else:
            self.remove_vectors(removed,nSleep=nSleep)
        return len(removed)

# USAGE :
# H(dm0)->dm1 :  dm1' = update(dm0, dm1-dm0)  ... this returns  dm0 + (1-mixing)*(dm1-dm0)
# 
    def update(self,vect,ervec,dict=None,Iter=-1,ervec_mixing=None,oldvec_mixing=None,verbose=False,strbuf=None,ervec_norm=None):

        comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        idbgng_threadsync=0 ## 2021.05.21 fixed.. ## idbgng_threadsync=2
        if( MPIsize < 2 ):
            idbgng_threadsync=0

        self.iter=Iter
        #if( self.iter>50 ):
        #    verbose=True
        self.seqno+=1
        #if( self.seqno>15):
        #    print("#DIIS.update:%d 

        assert (i1eqb(self.Ndim_vec,np.shape(vect))),""
        if(ervec is not None):
            assert (i1eqb(self.Ndim_vec,np.shape(ervec))),""

        if( self.nSleep > 0 ):
            if( self.check_linear_dependency ):
                if( self.check_linear_independency1(vect)):
                    Logger.write(self.writer,"linearly INdependent reducing nSleep:%d"%(self.nSleep-1))
                    self.nSleep=self.nSleep-1
                else:
                    Logger.write(self.writer,"linearly dependent vector..")
            else:
                self.nSleep=self.nSleep-1
            print("#DIIS:nSleep:%d"%(self.nSleep))
        start_diis=False
        if( self.nbuf == 0 ):
            assert (ervec is not None),"set ervec explicitly at the 1st step"
        nprev=(self.nbuf-1)%self.bfsz
        if( ervec is None ):
            ervec_1d = np.reshape( vect.copy(), [self.Ldvec]) ## you need to copy this
            ervec_1d[:]=ervec_1d[:]-self.vcbuf[nprev][:]
        else:
            ervec_1d = np.reshape( ervec, [self.Ldvec] ) ## you do not have to copy this
        dict1={"at":-1,"dev":-1}
        ermax=a1maxval(ervec_1d,dict=dict1)
        # print("#diis:ermax:%d %e"%(dict1["at"],dict1["dev"]));
        if( not self.started and self.nbuf>0 and self.nSleep<=0 ):
            if( ermax<self.ervec_thr and  self.nbuf>= self.nvect_thr ):
                ### print("start DIIS:%d/%d %e"%(self.nbuf,self.nvect_thr,ermax)); 
                self.started=True
            elif( self.nstep_thr>0 and self.nbuf>self.nstep_thr and self.nbuf>= self.nvect_thr):
                print("force start DIIS:%d/%d %d %e nvthr:%d"%(self.nbuf,self.nstep_thr,self.nvect_thr,ermax,self.nvect_thr)); self.started=True
        # fd1=open("DIIS_forcestart.log","a");
        # print("%05d %d %d %r %d/%d"%( self.seqno, Iter, self.bfsz, self.started,self.nbuf,self.nvect_thr),file=fd1)
        # fd1.close()
        ncur=self.nbuf%self.bfsz

        ## --- a. store error vectors 
        self.erbuf[ncur][:]=ervec_1d[:]             ## substitution into existing buffer
        self.vcbuf[ncur][:]=np.reshape( vect, [self.Ldvec]) ## substitution into existing buffer
        self.ibuf[ncur]=Iter
        ## --- b. update matrix [I][j] (only the LOWER triangle, I>=j are filled)
        if( self.nbuf>=self.bfsz ):
            # update all [ncur][*] [*][ncur].. note we fill [I]>=[j]
            for J in range(ncur):
                self.matrix[ncur][J]=dot_product(self,'ervecs',ervec_1d, self.erbuf[J])
            for I in range(ncur,self.bfsz):
                self.matrix[I][ncur]=dot_product(self,'ervecs',self.erbuf[I], ervec_1d)
        else:
            for J in range(ncur+1):
                self.matrix[ncur][J]=dot_product(self,'ervecs',ervec_1d, self.erbuf[J])
                ### print("#diis:filling into %d,%d:%f+j%f"%(ncur,J,self.matrix[ncur][J].real,self.matrix[ncur][J].imag))

        ### print(self.AOparams_['AOrep'])
        ### print(self.AOparams_['spinrestriction']);print(self.AOparams_['DorF'])
        if( ervec_norm is not None and self.AOparams_ is not None ):
            ervcSQREnrm=dot_AOmatrices(ervec_1d,ervec_1d,self.AOparams_['SAO'],
                                    self.AOparams_['spinrestriction'],self.AOparams_['DorF'],title="ervecnorm")
            ervcnrm=np.sqrt( abs(ervcSQREnrm) )
            print( "#ervecnorm:"+str(ervcnrm) +"/"+str( np.sqrt( abs(self.matrix[ncur][ncur].real) ) ))
            assert abs(ervec_norm-np.sqrt(abs(self.matrix[ncur][ncur])))<1e-8,"%e / %e"%( ervec_norm, np.sqrt( abs(self.matrix[ncur][ncur]) ) )

        self.nbuf+=1 ## --- c. finally we increment nbuf ----
        if( not self.started ):
            if( ervec_mixing is not None):
                if(verbose):
                    print("#diis_retv:vect:mixed:%d:%d:"%(ncur,self.ibuf[ncur])+toString(vect,Nmax=6) )
                    print("#diis_retv:ervec:%d:%d:"%(ncur,self.ibuf[ncur])+toString(ervec,Nmax=6) )
                return vect[:] - ervec_mixing * ervec[:]   ### new - 0.5*(new-old)
            elif( oldvec_mixing is not None):
                if( self.nbuf < 2 ):
                    if(verbose):
                        print("#diis_retv:vect:default:%d:%d:"%(ncur,self.ibuf[ncur])+toString(vect,Nmax=6) )
                    return vect[:]  ## nbuf==1 at first time
                else:
                    ndim_vect=np.shape(vect)
                    Iold=( self.nbuf - 2 )%self.bfsz 
                    if(verbose):
                        print("#diis_retv:vect:mixed:%d:%d:"%(ncur,self.ibuf[ncur])+toString(vect,Nmax=6) )
                        print("#diis_retv:prev_vect:%d:%d:"%(Iold,self.ibuf[Iold])+toString(self.vcbuf[Iold],Nmax=6) )
                    return vect[:]*(1.0-oldvec_mixing) + oldvec_mixing * np.reshape( self.vcbuf[Iold][:], ndim_vect)
            else:
                return vect;
        else:
            Neff= self.nbuf if(self.nbuf<self.bfsz) else self.bfsz;
            LdMat= Neff+1
            
            for I in range(1,Neff):
                for J in range (I):
                    self.matrix[J][I] = np.conj( self.matrix[I][J] )
            submatr=np.zeros([LdMat,LdMat],dtype=self.dtype)
            for I in range(LdMat-1):
                for J in range(LdMat-1):
                    submatr[I][J]=self.matrix[I][J];
            for J in range(LdMat-1):
                submatr[LdMat-1][J]=1.0
                submatr[J][LdMat-1]=1.0
            submatr[LdMat-1][LdMat-1]=0.0
            
            w=np.zeros([LdMat],dtype=self.dtype)
            for J in range(LdMat-1):
                w[J]=0.0
            w[LdMat-1]=1.0
            
            strdiff_sync="";strADD=""
            if(idbgng_threadsync>0):
                sqrediff_IN,maxabsdiff_IN,At_IN,vals_IN = mpi_aNdiff(submatr,"DIIS01%06d.%03d_submatr"%(self.seqno,self.iter))
                strdiff_sync="submatr:"+ mpidiffs_tostring_(sqrediff_IN,maxabsdiff_IN,At_IN,vals_IN,Datf=False)

            DEBUG_matrix=False
            if( DEBUG_matrix ):
                dev=-1;at=[-1,-1];vals=[0,0]
                for I in range(LdMat-1):
                    for J in range(LdMat-1):
                        cdum=dot_product(self,'ervecs',self.erbuf[I],self.erbuf[J])
                        diff=abs(cdum-submatr[I][J]);
                        if(diff>dev):
                            dev=diff;at=[I,J];vals=[submatr[I][J],cdum]
                if( dev>1.0e-6 ):
                    print("wrong mat:%f+j%f / %f+j%f"%( vals[0].real,vals[0].imag, vals[1].real,vals[1].imag ))
                    for I in range(LdMat-1):
                        ref="";string=""
                        for J in range(LdMat-1):
                            cdum=dot_product(self,'ervecs',self.vcbuf[I],self.vcbuf[J])
                            ref=ref+"%10.3f %10.3f "%(cdum.real,cdum.imag)
                            string=string+"%10.3f %10.3f "%(submatr[I][J].real, submatr[I][J].imag)
                        print("#matr:%d:"%(I)+string)
                        print("#refr:%d:"%(I)+ref)

                    assert False,"wrong mat"
            ### coef1,res1,rank1,sv1=np.linalg.lstsq(submatr,w,rcond=None)
            coef,res,rank,sv=np.linalg.lstsq(submatr,w,rcond=None) ## this is equivalent to svdsolv
            ### coef,res,rank,sv=np.linalg.lstsq(submatr,w,rcond=-1)

            #svdsolv:  List_err=[]
            #svdsolv:  coef,res = svdsolv(submatr,w,List_errors=List_err)
            #if( coef is None ):
            #    printout("DIIS:svdsolve failed:"+List_err[0],warning=-1)
            #    coef=coef1;res=res1

            if(idbgng_threadsync>0 and MPIsize>1 ):
                sqrediff_OU1,maxabsdiff_OU1,At_OU1,vals_OU1 = mpi_aNdiff(coef,"DIIS01%06d.%03d_cof1"%(self.seqno,self.iter))
                sqrediff_OUT,maxabsdiff_OUT,At_OUT,vals_OUT = mpi_aNdiff(coef,"DIIS01%06d.%03d_coef"%(self.seqno,self.iter))
                strdiff_sync+="coef:"+ mpidiffs_tostring_(sqrediff_OUT,maxabsdiff_OUT,At_OUT,vals_OUT,Datf=False)
                strADD="#np.linalg.lstsq: %16.6e  with res="%(maxabsdiff_OU1)+str(res)
                ###   +"  \t svdsolv: %16.6e  with res="%(maxabsdiff_OUT)+str(res) 
            if( dict is not None ):
                if("coef" in dict):
                    dict["coef"]=coef
                if("ibuf" in dict):
                    dict["ibuf"]=self.ibuf
            DEBUG_soln=False
            if(DEBUG_soln):
                wrk=np.zeros([LdMat],dtype=self.dtype)
                wrk=np.matmul( submatr,coef)
                print("#diis:AxSOLN:"+z1toa(wrk))
                print("#diis:w:"+z1toa(w))

                lcERR=np.zeros([self.Ldvec],dtype=self.dtype)
                lcERR[:]=coef[0]*self.erbuf[0][:]
                for J in range(1,Neff):
                    lcERR[:]+=coef[J]*self.erbuf[J][:]
                cdum=dot_product(self,'ervecs', lcERR, lcERR)
                lcNORM=( cdum.real if (type(cdum) is complex) else cdum)
                print("#LC norm:%f / ervcs:"+diag_tostring(self.matrix))

            retv_1d=np.zeros([self.Ldvec],dtype=self.dtype)
            retv_1d[:]=coef[0]*self.vcbuf[0][:]
            for J in range(1,Neff):
                retv_1d[:]+=coef[J]*self.vcbuf[J][:]

            if(idbgng_threadsync>0):
                sqrediff_IN2,maxabsdiff_IN2,At_IN2,vals_IN2 = mpi_aNdiff(self.vcbuf[0:Neff],"DIIS01%06d.%03d_vcbuf"%(self.seqno,self.iter))
                strdiff_sync+=" vecs:"+ mpidiffs_tostring_(sqrediff_IN2,maxabsdiff_IN2,At_IN2,vals_IN2,Datf=False)
                sqrediff_OUT2,maxabsdiff_OUT2,At_OUT2,vals_OUT2 = mpi_aNdiff(retv_1d,"DIIS01%06d.%03d_retv"%(self.seqno,self.iter))
                strdiff_sync+=" retv_1d:"+ mpidiffs_tostring_(sqrediff_OUT2,maxabsdiff_OUT2,At_OUT2,vals_OUT2,Datf=False)

                fd01=open("diis_DBGsynchthreads_%02d.log"%(MPIrank),'a')
                print("# DIIS%06d.%03d: retv:%16.4e  submatr:%16.4e  vecs:%16.4e  coefs:%16.4e"%(\
                    self.seqno, self.iter, maxabsdiff_OUT2, maxabsdiff_IN, maxabsdiff_IN2, maxabsdiff_OUT),file=fd01);
                print("## DIIS%06d.%03d: "%(self.seqno,self.iter) + strdiff_sync, file=fd01); 
                print(strADD, file=fd01);fd01.close()
## -------------------------------------------------------------------------------------
            dbgng_DIIScoefs=False
            if( dbgng_DIIScoefs and self.AOparams_ is not None):
                diff_to_prev=None
                if( self.retv_last is not None ):
                    cdum= dot_AOmatrices( retv_1d-self.retv_last, retv_1d-self.retv_last, self.AOparams_['SAO'], 
                                           self.AOparams_['spinrestriction'], self.AOparams_['DorF'])
                    diff_to_prev = np.sqrt( abs(cdum.real) )
                self.retv_last=retv_1d.copy()
                err=np.vdot( coef[0:LdMat-1], np.matmul(submatr[0:LdMat-1,0:LdMat-1],coef[0:LdMat-1]) )
                err=np.sqrt( abs(err.real) )
                self.dbgbuffer.append({'ncur':ncur, 'cofs':coef.copy(), 'err':err, 'diff_to_prev':diff_to_prev})
                lebf=len(self.dbgbuffer)
                if( lebf>= 6 ):
                    fpath1="DIIS_coefs.log"
                    fd1=open(fpath1,"a");
                    print("#DIIS:iter=%d nbuf=%d/%d ncur=%d"%(Iter,self.nbuf,self.bfsz,ncur),file=fd1)
                    
                    for k in range(Neff):
                        I=ncur-Neff+1+k
                        Imod=(I+self.bfsz)%self.bfsz; err=np.sqrt( abs(self.matrix[Imod][Imod].real))
                        string="";
                        for j in range(lebf):
                            arr=self.dbgbuffer[j]['cofs']
                            if( len(arr)>k ):
                                string+="%10.4f %10.4f     "%(arr[k].real,arr[k].imag)
                            else:
                                string+=(' '*26)
                        print(" %5d  %3d  %14.4e  "%(I,self.ibuf[Imod],err) +string,file=fd1)
                    string="";
                    for j in range(lebf):
                        err_est=self.dbgbuffer[j]['err']
                        jnxt =self.dbgbuffer[j]['ncur']+1
                        if( jnxt < self.nbuf ):
                            jmod=(jnxt+self.bfsz)%self.bfsz
                            err_true=np.sqrt( abs(self.matrix[jmod][jmod].real) )
                            string+="%10.4e %10.4e     "%(err_est,err_true)
                        else:
                            string+="%10.4e "%(err_est)+(' '*15)
                    print("#%5s  %3s  %14s  "%("err","","")+string,file=fd1)
                    string="";
                    for j in range(lebf):
                        diff=self.dbgbuffer[j]['diff_to_prev']
                        if( diff is None ):
                            string+=(' '*26)
                        else:
                            string+="%14.4e "%(diff)+' '*11
                    print("#%5s  %3s  %14s  "%("diff","","")+string,file=fd1)
                    fd1.close()
                    self.dbgbuffer.clear();
                    os.system("fopen "+fpath1);
##------------------------------------------------------------------------------------
            if(verbose):
                for J in range(Neff):
                    Logger.write(self.writer,"#diis_retv:%d:LC:%d(%d)  %f+%fj"%(self.iter, J, self.ibuf[J],coef[J].real,coef[J].imag))
#            if(self.iter>80):
#                assert False,""
###+toString(retv_1d,Nmax=6)
            return np.reshape(retv_1d, self.Ndim_vec)

def z1rand(N):
    ret=np.zeros([N],dtype=complex)
    foo=random.rand(N)
    boo=random.rand(N)
    for I in range(N):
        ret[I]=complex( foo[I],boo[I])
    return ret;

def d1toa(vect,N_upl=None,format="%5.2f "):
    ret=""
    N=len(vect);
    if(N_upl is not None):
        N=min( N_upl, N)
    for j in range(N):
        ret=ret+ format%vect[j]
    return ret;
def dbgwrited1(vect):
    ret=""
    Ndim=np.shape(vect)
    dim=len(Ndim)
    if(dim==1):
        return d1toa(vect,N_upl=20)
    else:
        Ld=Ndim[0];
        for k in range(1,dim):
            Ld=Ld*Ndim[k]
        return d1toa(np.reshape(vect,[Ld]),N_upl=20)

#Ldvec=10;bfsz=10;
#diis=DIIS01(Ldvec,bfsz,dtype=complex,ervec_thr=1.0e-3,nstep_thr=2,mixing_factor=0.50)
#v0=z1rand(Ldvec)
#v1=z1rand(Ldvec)
#diis.update(v1,v1[:]-v0[:])
#v2=z1rand(Ldvec)
#diis.update(v2)
#v3=z1rand(Ldvec)
#diis.update(v3)
#v4=z1rand(Ldvec)
#diis.update(v4)

def aNdiff(A,B):
    from .utils import ixj_to_IandJ
    a=np.ravel( np.array(A) )
    b=np.ravel( np.array(B) )
    d=a-b
    sqrediff=( np.vdot( d, d) ).real

    maxabsdiff=abs(a[0]-b[0]);at=0;vals=[a[0],b[0]];
    N=len(a)
    for I in range(1,N):
        dum=abs(a[I]-b[I])
        if( dum>maxabsdiff ):
            maxabsdiff=dum; at=I; vals=[a[I],b[I]]
    Ndim_A=np.shape(A)
    At=ixj_to_IandJ(at,Ndim_A)
    return sqrediff,maxabsdiff,At,vals

def mpidiffs_tostring_(sqrediff,maxabsdiff,At,vals,Datf=False,Legend=False):
    retv=None;legend=""
    if( not Datf ):
        retv="%16.8f %16.6e %10.4f+j%10.4f / %10.4f+j%10.4f at "%(\
            np.sqrt(sqrediff), maxabsdiff,vals[0].real,vals[0].imag,vals[1].real,vals[1].imag)\
            + str(At)
    else:
        retv="%16.8f    %16.6e    %10.4f %10.4f    %10.4f %10.4f"%(\
            np.sqrt(sqrediff), maxabsdiff,vals[0].real,vals[0].imag,vals[1].real,vals[1].imag)
    if( not Legend ):
        return retv
    else:
        legend=( "%16s    %16s    %21s    %21s"%('dist','maxabsdiff','lhs','rhs') if(Datf) else 
             "%16s %16s %22s / %22s at %s"%('dist','maxabsdiff','lhs','rhs','ijk') )
    return retv,legend
# sqrediff,maxabsdiff,At,vals = mpi_aNdiff(md.tdMO,"tdMO_eldyn%03d.%03d"%(I_step,iter))
def mpi_aNdiff(buf,key,compto=0,sync=False):
    from .utils import arrayclone
    from mpi4py import MPI
    from .mpiutils import mpi_Bcast
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()    
    if( MPIsize < 2 ):
        return
    assert compto>=0,"" ## (-1) may mean comp to MPIrank-1 (mod size) th thread but..
    assert compto<MPIsize,"" ## (-1) may mean comp to MPIrank-1 (mod size) th thread but..
    wks=arrayclone(buf)
    mpi_Bcast("diff_"+key,wks,root=compto)
    sqrediff,maxabsdiff,At,vals=aNdiff(buf,wks)
    
    if(sync and (MPIrank != compto)):
        if( maxabsdiff>diff_THR ):
            Ndim=np.shape(buf);leNdim=len(Ndim)
            if(leNdim==1):
                for I in range(Ndim[0]):
                    buf[I]=wks[I]
            elif(leNdim==2):
                for I in range(Ndim[0]):
                    for J in range(Ndim[1]):
                        buf[I][J]=wks[I][J]
            elif(leNdim==3):
                for I in range(Ndim[0]):
                    for J in range(Ndim[1]):
                        for K in range(Ndim[2]):
                            buf[I][J][K]=wks[I][J][K]
            elif(leNdim==4):
                for I in range(Ndim[0]):
                    for J in range(Ndim[1]):
                        for K in range(Ndim[2]):
                            for L in range(Ndim[3]):
                                buf[I][J][K][L]=wks[I][J][K][L]
            elif(leNdim==5):
                for I in range(Ndim[0]):
                    for J in range(Ndim[1]):
                        for K in range(Ndim[2]):
                            for L in range(Ndim[3]):
                                for M in range(Ndim[4]):
                                    buf[I][J][K][L][M]=wks[I][J][K][L][M]
            else:
                assert False,""
    return sqrediff,maxabsdiff,At,vals

def svdsolv(A,b,eps=None,rcond=None,List_errors=None,Dict_logs=None):
    from scipy import linalg as scipy_linalg
    from .utils import arrayclone
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    dbgng_sync=True  # XXX XXX
    if( MPIsize < 2 ):
        dbgng_sync= False
    dbgng=True # XXX XXX
    if( eps is None ):eps=np.finfo(float).eps
    Ndim=np.shape(A)
    
    if(dbgng_sync):
        sqrediff_IN,maxabsdiff_IN,At_IN,vals_IN = mpi_aNdiff(A,"svdsolv_INPUT")
        sqrediff_b,maxabsdiff_b,At_b,vals_b = mpi_aNdiff(b,"svdsolv_INPUTb")

    U,w,Vh=scipy_linalg.svd(A)
    if(dbgng_sync):
        sqrediff_U,maxabsdiff_U,At_U,vals_U = mpi_aNdiff(U,"svdsolv_Umat")
        sqrediff_w,maxabsdiff_w,At_w,vals_w = mpi_aNdiff(w,"svdsolv_w")
        sqrediff_Vh,maxabsdiff_Vh,At_Vh,vals_Vh = mpi_aNdiff(Vh,"svdsolv_Vh")

    N=len(w)
    absmXw=max( abs(w) )
    absmNw=min( abs(w) )
    if( absmXw < eps ):
        string="svdsolv:!E:all max|singular value|(=%e) < eps(:=%e)"%(absmXw,eps)
        if( List_errors is not None):
            List_errors.append(string);
            return None,None
        else:
            assert False,""
    condition_number=absmNw/absmXw
    if( Dict_logs is not None):
        warn=( condition_number < 1.0e-12 )
        key=('warning' if(warn) else 'info')
        string="svdsolv:%s condition number %e"%( ('!W' if(warn) else ''),condition_number )
        if( key not in Dict_logs ):
            Dict_logs.update({key:[]})
        Dict_logs[key].append(string)
    
    thr=eps
    if( rcond is None ):
        thr=eps*absmXw
##  A   = U w Vh
## [M,N] [M,N][N,N][N,N]  
## here, Vh[i][:] is the Ith vector
    cofs=(np.zeros([N],dtype=(np.array(A)).dtype) if(dbgng_sync) else None)
    bcpy=arrayclone(b)
    ret=np.zeros(Ndim[1]);Nskip=0
    for I in range(N):
        if( abs(w[I])<thr ):
            Nskip+=1;continue
        a=np.vdot( U[:,I], b )
        cof=a/w[I]
        if(cofs is not None):
            cofs[I]=cof
        ret=ret+Vh[I]*cof
        bcpy=bcpy-a*U[:,I]
    res=np.sqrt( abs(np.vdot(bcpy,bcpy).real))
    if(dbgng_sync):
        sqrediff_c,maxabsdiff_c,At_c,vals_c = mpi_aNdiff(cofs,"svdsolv_cofs")
        sqrediff_ret,maxabsdiff_ret,At_ret,vals_ret = mpi_aNdiff(ret,"svdsolv_ret")
        fd=open("diis_svdsolv_%02d.log"%(MPIrank),'a')
        print("#diis01.svdsolv:  ret:%e  INPUT:%e,%e  coefs:%e([%d] w=%e coef=%s/%s) w:%e U:%e Vh:%e"%(\
            maxabsdiff_ret, maxabsdiff_IN,maxabsdiff_b,maxabsdiff_c,At_c[0],w[ At_c[0] ],str(vals_c[0]),str(vals_c[1]),
            maxabsdiff_w, maxabsdiff_U, maxabsdiff_Vh),file=fd)
        fd.close()

    if(dbgng):
        y=np.matmul(A,ret)
        ref=b-bcpy
        dev=max( abs( y-ref ))
        dist=np.sqrt( abs( np.vdot(y-ref,y-ref).real ) )
        print("#svdsolv:|Ax-b|=%16.6e |Ax-(b-res)|= %16.8f  %16.6e"%(res,dist, dev))
        assert dev<1.0e-8 and dist<1.0e-8,""
    return ret,res
