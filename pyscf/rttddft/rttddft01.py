import numpy as np
import os
import time
import sys
import math
import cmath
import datetime
from scipy import linalg
from mpi4py import MPI

from pyscf.pbc.gto import intor_cross
from pyscf.pbc.dft.krks import KRKS
### from krks01 import KRKS01
from pyscf.dft.rks import RKS
from scipy import linalg
from pyscf.lib import logger as pyscflib_logger
from .rttddft_common import rttddft_common
from pyscf.lib import logger as pyscflib_logger
from pyscf import lib
import pyscf.gto as molgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.scf.khf import get_occ as khf_get_occ
from pyscf.pbc.scf.kuhf import get_occ as kuhf_get_occ
from pyscf.pbc.scf.khf import make_rdm1 as khf_make_rdm1
from pyscf.pbc.dft import multigrid
from pyscf.pbc.dft import gen_grid
from pyscf.gto.mole import nao_cart
from pyscf import __config__
from pyscf.pbc.dft import rks
from pyscf.pbc import df
from pyscf.lib import logger as pyscf_logger

from pyscf.dft.rks import _dft_common_init_

from .Loglv import printout
from .heapcheck import heapcheck

from .futils import futils
from .laserfield import *
from .Dbglogger import Dbglogger
from .diis01 import DIIS01
from .Logger import Logger
from .GEAR import initz_rGEARvec, rGEAR2ndOne
from .serialize import is_method_or_function,serialize_listedfields,serialize_tdfield,svld_aNbuf

# tostring,save_zbuf,write_aNbuf,fwrite_zNbuf,aNdiff,d3diff,read_matr,a1maxdiff,d2toa,z2toa,parse_xyzstring,write_xyzstring,
from .utils import parse_doubles,parse_dict,check_Hermicity_xKaa,calc_3Dvolume,update_dict,printout_dict,print_Hmatrices,\
d1dist,i1eqb,z2diff,dNtoa,d1toa,z1toa,arrayclone,d1x2toa,aNmaxdiff,toComplexArray,write_once,print_z2array,iNtoa,d1diff,sNtoa,zNtoa,dist3D,\
check_equivalence
from .Constants import Constants
from .physicalconstants import PhysicalConstants
from .gto_ps_pp_int01 import with_df_get_pp,with_df_get_ppxx
from .gto_ps_pp_int01_bf20210608 import with_df_get_pp as with_df_get_pp_bf20210608
from .with_fftdf_get_pp import with_fftdf_get_pp

def obj_to_string(val):
    ## this is taken from my serialize.py ...
    typ='';strval=""
    if( val is None ):
        typ='o';strval="None"
    elif( isinstance(val,str) ):
        Stdlogger.printout(2,key+":str",verbose=verbose)
        typ='s';strval=str(val)
    elif( isinstance(val,bool) ):   ## << check bool first:  isinstance( BOOLEAN, int ) returns -True- .. 
        Stdlogger.printout(2,key+":bool",verbose=verbose)
        typ='b';strval=str(val)
    elif( isinstance(val,int) ):
        Stdlogger.printout(2,key+":int",verbose=verbose)
        typ='i';strval="%d"%(val)
    elif( isinstance(val,float) or isinstance(val,np.float64) ):
        Stdlogger.printout(2,key+":float",verbose=verbose)
        typ='d';strval="%24.12e"%(val)
    elif( isinstance(val,complex) or isinstance(val,np.complex128) ):
        Stdlogger.printout(2,key+":complex",verbose=verbose)
        typ='z';strval="%24.12e %24.12e"%(val.real, val.imag)
    elif( isinstance(val,list) or isinstance(val,np.ndarray) ):
        Stdlogger.printout(2,key+":list_or_ndarray:"+str(val),verbose=verbose)
        Ndim=np.shape(val)
        a1D =np.ravel( np.array(val) );dty=a1D.dtype
        if( len(a1D) == 0 ):
            if( isinstance(val,list) ):
                typ='I0';strval=""
            else:
                typ=('I0' if(val.dtype==int) else ('D0' if(val.dtype==float or val.dtype==np.float64) else 
                    ('Z0' if(val.dtype==complex or val.dtype==np.complex128) else 'S0')));strval=""
        else:
            if( type(a1D[0])==str or type(a1D[0])==np.str_ ):
                typ='S'+iNtoa(Ndim,delimiter=',');strval=sNtoa(val,delimiter="  ")
            else:
                if( dty==int ):
                    typ='I'+iNtoa(Ndim,delimiter=',');strval=iNtoa(val,delimiter="  ")
                elif( dty==float or dty==np.float64 ):
                    typ='D'+iNtoa(Ndim,delimiter=',');strval=dNtoa(val,format="%24.12e",delimiter="  ")
                elif( dty==complex or dty==np.complex128 ):
                    typ='Z'+iNtoa(Ndim,delimiter=',');strval=zNtoa(val,format="%24.12e %24.12e", delimiter="     ")
                ### elif( dty==str ):
                ###    typ='S'+iNtoa(Ndim,delimiter=',');strval=sNtoa(val,delimiter="  ")
                else:
                    strval=str(val);
                    print("#serialize:"+"dty:"+str(dty)+" KEY:"+str(key)+" type:"+str(type(val)))
                    print("#serialize:"+str(val))
                    Stdlogger.printout(1,"#Serialize:"+header+str(type(obj))+":"+str(obj),verbose=verbose)
                    Stdlogger.printout(1,"key:"+str(key)+" type:"+str(type(val))+" dtype:"+str(np.array(val).dtype)+" val:"+str(val),verbose=verbose)
                    ### assert False,"dty:"+str(dty)+" KEY:"+str(key)+" type:"+str(type(val))
    elif( isinstance(val,dict) ):
        if( len(val) == 0 ):
            typ='dict';strval="{}"
        else:
            typ='dict';strval=None
            for ky in val:
                if(strval is None):
                    strval="{"+ str(ky).strip() + ","+ str( val[ky] ).strip()
                else:
                    strval+=","+ str(ky).strip() + ","+ str( val[ky] ).strip()
            strval+="}"
    return strval
def get_Ssqrt_invsqrt_sINV(S):
    Ssqrt,Invrt,sINV = get_Ssqrt_invsqrt_sINV_1(S)
    N=len(S)
    dum=np.zeros([N,N],dtype=np.complex128)
    dum=np.matmul( Ssqrt, Ssqrt)
    dev1=z2diff( S, dum);### , title="|Ssqrt**2-S|");
    Logger.write_maxv(None,"Ssqrt_dev",dev1);
    dum=np.matmul( Invrt, Ssqrt )
    dev2=z2diff( dum, np.eye(N,dtype=np.complex128));###,title="|Invrt*Ssqrt-1|")
    Logger.write_maxv(None,"invSsqrt_dev",dev2);
## 
    dum=np.matmul( Invrt, np.matmul( S, Invrt) )
    dev3=z2diff( dum, np.eye(N,dtype=np.complex128));
    Logger.write_maxv(None,"invSsqrt_dev3",dev3); ### print("#invSsqrt_dev3:",dev3)
    return Ssqrt,Invrt,sINV

# S V = V \Lambda   so  S = V \Lambda V^{\dagger}
# S[I][:] = V[I][k] Lambda[k] conjg( V[:][k] )
def get_Ssqrt_invsqrt_sINV_1(S):
    N=len(S)
    dtype=np.array(S).dtype
    ### print("get_Ssqrt_invsqrt:",end="");print(dtype)
    iscomplex=False
    if( dtype == complex ):
        iscomplex=True
        eigvals,vecs,info=linalg.lapack.zheev(S)
    else:
        eigvals,vecs,info=linalg.lapack.dsyev(S)
    assert (info==0),"dsyev/zheev failed"
    # print("S_eigvals:",end="");print(eigvals)
    Ssqrt=np.zeros([N,N],dtype=dtype)
    InvSqrt=np.zeros([N,N],dtype=dtype)
    sINV=np.zeros([N,N],dtype=dtype)
    for I in range(N):
        for J in range(N):
            cdum=( np.complex128(0.0) if(iscomplex) else 0.0)
            for k in range(N):
                cdum+= vecs[I][k]*math.sqrt(eigvals[k])*np.conj( vecs[J][k])
            Ssqrt[I][J]=cdum

            cdum=( np.complex128(0.0) if(iscomplex) else 0.0)
            for k in range(N):
                cdum+= vecs[I][k]*(1.0/math.sqrt(eigvals[k]))*np.conj( vecs[J][k])
            InvSqrt[I][J]=cdum

            cdum=( np.complex128(0.0) if(iscomplex) else 0.0)
            for k in range(N):
                cdum+= vecs[I][k]*(1.0/eigvals[k])*np.conj( vecs[J][k])
            sINV[I][J]=cdum

    return Ssqrt,InvSqrt,sINV;

def suppress_prtout():
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();
    return ( MPIrank!=0 )

def list_to_array(src):
    dbgng=True;dbglog=""
    if( dbgng ):
        dbglog+="#list_to_array:INPUT:"+str(type(src))+str(np.shape(src))+"\n"
    retv=np.array(src)
    if( dbgng ):
        dbglog+="#list_to_array:OUTPUT:"+str(type(retv))+str(np.shape(retv))+"\n"
        Ndim=np.shape(retv);rank=len(Ndim)
        if( rank > 1 ):
            for I in range(Ndim[0]):
                dbglog+="#list_to_array:OUTPUT:[%d]:"%(I)+str(type(retv[I]))+str(np.shape(retv[I]))+"\n"
        printout(dbglog,fpath="list_to_array_DBG.log",Append=True)
    return retv

class Dipoles_static:
    Diff_moldip_eldip_=None


def dmtrace_xKaa(matr,comp,dmat,pbc,spinrestriction,dict=None):
    if( spinrestriction == 'R' ):
        return dmtrace_xKaa1(matr,comp,dmat,pbc,dict=dict)
    else:
        return dmtrace_xKaa1(matr,comp,dmat[0],pbc,dict=dict),dmtrace1(matr,comp,dmat[1],pbc,dict=dict)
def dmtrace_xKaa1(matr,comp,dmat,pbc,dict=None):
    ndim=np.shape(dmat)
    nkpt=1;nAO=ndim[0];   # dmat[nAO][nAO]
    if( pbc ):            # dmat[nKpt][nAO][nAO] 
        nkpt=ndim[0];nAO=ndim[1]
    wks=np.zeros([nAO,nAO],dtype=np.complex128)
    
    ndim_matr=np.shape(matr)
    if( not pbc ):           ## non-pbc [comp][nAO][nAO] 
        assert( ndim_matr[0]==comp and ndim_matr[1]==nAO),""+str(ndim_matr)
    else:                    ## pbc     [comp][nKpt][nAO][nAO] 
        assert( ndim_matr[0]==comp and ndim_matr[1]==nkpt and ndim_matr[2]==nAO),""+str(ndim_matr)

    if( not pbc ):
        ret=[]
        for dir in range(comp):
            wks=np.matmul( matr[dir],dmat)
            cum=wks[0][0]
            for I in range(1,nAO):
                cum+=wks[I][I]
            ret.append(cum)
        return np.array(ret)
    else:
        ## print("#dmtrace:DMshape",end="");print(np.shape(dmat))
        ## print("#dmtrace:MATshape",end="");print(np.shape(matr))
        ret=[]
        for dir in range(comp):
            buf=[]
            for kp in range(nkpt):
                wks=np.matmul( matr[dir][kp],dmat[kp])
                cum=wks[0][0]
                for I in range(1,nAO):
                    cum+=wks[I][I]
                buf.append(cum)
            tot=sum(buf)
            avg=tot/float(nkpt)
            ## print("#dipole_%d:"%(dir)+z1toa(buf)+" avg(/%d):%f+j%f"%(nkpt,avg.real, avg.imag))
            ret.append(avg)
        return np.array(ret)

def dmtrace(matr,comp,dmat,pbc,spinrestriction,dict=None):
    # spin-sym:
    # non-pbc : matr[comp][nAO][nAO]        dmat[nAO][nAO]         >> ret[comp]
    # pbc     : matr[nkpt][comp][nAO][nAO]  dmat[nkpt][nAO][nAO]   >> ret[comp] ## avgd over kpts
    # spin-pol 
    # non-pbc : matr[comp][nAO][nAO]        dmat[2][nAO][nAO]      >> ret[spin][comp] 
    # pbc     : matr[nkpt][comp][nAO][nAO]  dmat[2][nkpt][nAO][nAO]>> ret[spin][comp]
    if( spinrestriction == 'R' ):
        return dmtrace1(matr,comp,dmat,pbc,dict=dict)
    else:
        return dmtrace1(matr,comp,dmat[0],pbc,dict=dict),dmtrace1(matr,comp,dmat[1],pbc,dict=dict)

def dmtrace1(matr,comp,dmat,pbc,dict=None):
    ndim=np.shape(dmat)
    nkpt=1;nAO=ndim[0];
    if( pbc ):
        nkpt=ndim[0];nAO=ndim[1]
    wks=np.zeros([nAO,nAO],dtype=np.complex128)
    
    ndim_matr=np.shape(matr)
    if( not pbc ):
        assert( ndim_matr[0]==comp and ndim_matr[1]==nAO),""+str(ndim_matr)
    else:
        assert( ndim_matr[0]==nkpt and ndim_matr[1]==comp and ndim_matr[2]==nAO),""+str(ndim_matr)
        
    if( not pbc ):
        ret=[]
        for dir in range(comp):
            wks=np.matmul( matr[dir],dmat)
            cum=wks[0][0]
            for I in range(1,nAO):
                cum+=wks[I][I]
            ret.append(cum)
        return np.array(ret)
    else:
        ## print("#dmtrace:DMshape",end="");print(np.shape(dmat))
        ## print("#dmtrace:MATshape",end="");print(np.shape(matr))
        ret=[]
        for dir in range(comp):
            buf=[]
            for kp in range(nkpt):
                wks=np.matmul( matr[kp][dir],dmat[kp])
                cum=wks[0][0]
                for I in range(1,nAO):
                    cum+=wks[I][I]
                buf.append(cum)
            tot=sum(buf)
            avg=tot/float(nkpt)
            ## print("#dipole_%d:"%(dir)+z1toa(buf)+" avg(/%d):%f+j%f"%(nkpt,avg.real, avg.imag))
            ret.append(avg)
        return np.array(ret)

def check_hermicity_dm(this, densitymatrix, sqrSums=None, DAGthr=-1.0, OFDthr=-1.0, title="", verbose=0):

    spinrestriction=this._spinrestriction
    pbc=this._pbc
    nspin=(1 if(spinrestriction=='R') else 2)
    OFDdev=-1;OFDvals=None;OFDindcs=None;sqrsum=0.0
    DAGdev=-1;DAGval=None; DAGindex=None
    max_Im=-1;val=None;index=None
    sqrsum_Re=0.0; sqrsum_Im=0.0
    S1e=None
    if( not pbc ):
        S1e=this.get_ovlp()
    for spin in range(nspin):
        dmat = ( densitymatrix if(nspin==1) else densitymatrix[spin] )
        Ndim=np.shape(dmat)
        nKpt=(1 if(not pbc) else Ndim[0])
        nAO=(Ndim[0] if(not pbc) else Ndim[1])
        kvectors=(None if(not this._pbc) else np.reshape( this.kpts, (-1,3)) )
            
        for kp in range(nKpt):
            if(pbc):
                S1e =this.get_ovlp( this.cell, kvectors[kp])
            dm=( dmat if(not pbc) else dmat[kp])
            for mu in range(nAO):
                for nu in range(mu):
                    dum=abs( dm[mu][nu] - np.conj( dm[nu][mu] ))
                    if( dum > OFDdev):
                        OFDdev=dum;OFDvals=[ dm[mu][nu], dm[nu][mu] ];OFDindcs=[mu,nu]
                    if( max_Im < abs(dm[mu][nu].imag) ):
                        max_Im = abs(dm[mu][nu].imag); val=dm[mu][nu]; index=[mu,nu]
                    sqrsum_Re+=0.50* (dm[mu][nu].real + dm[nu][mu].real)*S1e[mu][nu]
                    sqrsum_Im+=0.50* (dm[mu][nu].imag - dm[nu][mu].imag)*S1e[mu][nu]
                dum=abs( dm[mu][mu].imag )
                if(dum>DAGdev):
                    DAGdev=dum; DAGval= dm[mu][mu];DAGindex=mu
                sqrsum_Re+= dm[mu][mu].real * S1e[mu][mu]
                sqrsum_Im+= dm[mu][mu].imag * S1e[mu][mu]
                #    sqrsum_Re+= dm[mu][mu].real**2
                #    sqrsum_Im+= dm[mu][mu].imag**2
        if(sqrSums is not None):
            sqrSums.clear()
            sqrSums.update({'real':sqrsum_Re,'imag':sqrsum_Im})
        prtout=(verbose>1);warn=""
        if( DAGdev>1.0e-4 or OFDdev>1.0e-4 or (DAGthr>0 and DAGdev>0.1*DAGthr) or (OFDthr>0 and OFDdev>0.1*OFDthr) ):
            prtout=True;warn="W!:"
        if(prtout):
            mu=OFDindcs[0];nu=OFDindcs[1]
            printout("#check_hermicity_DM:"+warn+title+" dagdev=%16.6e [%d][%d]:%14.6f+j%14.6f  ofddev=%16.6e  [%d][%d]:%14.6f+j%14.6f / %14.6f+j%14.6f"%(
                DAGdev,DAGindex,DAGindex,DAGval.real,DAGval.imag,
                OFDdev,mu,nu, dm[mu][nu].real,dm[mu][nu].imag, dm[nu][mu].real,dm[nu][mu].imag))
            printout("#check_hermicity_DM:"+warn+title+" max_Im:%16.6e [%d][%d]:%14.6f+j%14.6f sqrSUM:%14.6f, %14.6f"%(
                max_Im,index[0],index[1], dm[index[0]][index[1]].real, dm[index[0]][index[1]].imag,sqrsum_Re, sqrsum_Im))
        if(DAGthr > 0):
            assert DAGdev<DAGthr,""
        if(OFDthr > 0):
            assert OFDdev<OFDthr,""
        
        return DAGdev,OFDdev

def trace_dmat(this,DMAT,spinrestriction,pbc,verbose=False,dev_abort_thr=None):
#        pbc
#  U,O   T    [2,nKpt,nAO,nAO]
#        F    [2,nAO,nAO
#  R     T    [nKpt,nAO,nAO]
#        F    [nAO,nAO
    nKpt=1;nAO=None;nspin=1
    Ndim=np.shape(DMAT); ### print("DMshape:",end="");print(Ndim)
    S1e=None; kvectors=None
    if( not pbc ):
        S1e=this.get_ovlp()
    else:
        kvectors=np.reshape( this.kpts, (-1,3))

    if( spinrestriction != 'R' ):
        assert Ndim[0]==2,""
        if( pbc ):
            nKpt=Ndim[1];nAO=Ndim[2];assert Ndim[3]==nAO,""
        else:
            nKpt=1; nAO=Ndim[1];assert Ndim[2]==nAO,""
    else:
        if( pbc ):
            nKpt=Ndim[0];nAO=Ndim[1];assert Ndim[2]==nAO,""
        else:
            nKpt=1; nAO=Ndim[0];assert Ndim[1]==nAO,""
    
    
    nspin=(1 if(spinrestriction=='R') else 2)
    ret=[]
    for spin in range(nspin):
        dma=( DMAT if(nspin==1) else DMAT[spin]);
        kp_sum=0.0;
        for kp in range(nKpt):
            if( pbc ):
                S1e=this.get_ovlp( this.cell,kvectors[kp] )
            dm=( dma if(not pbc) else dma[kp] );
            Sxdm=np.matmul(S1e, dm)
            cum=0.0
            for ao in range(nAO):
                cum+=Sxdm[ao][ao]
            ### print("#DMtrace:sub:s=%d kp=%d/%d subtrace=%16.8f"%(spin,kp,nKpt,cum))
            # at EACH kpoint, the subtrace equals N_{ele} (RKS) or N_{sigma} (UKS)
            kp_sum+=cum;
        if(pbc):
            kp_sum = kp_sum/float(nKpt)  # .. so sum over kpoints and divide by nkpt ..
        ret.append(kp_sum);
    if(nspin==1):
        ret=ret[0]
    ### print("ret:",end="");print(ret)
    dev=-1.0
    N_ele=get_Nele(this)
    for spin in range(nspin):
        strspin=("" if(nspin==1) else "spin%02d"%(spin))
        v=( ret if(nspin==1) else ret[spin] );  ### print("#v:",end="");print(v);
        occ=(this._mo_occ if(nspin==1) else this._mo_occ[spin])
        n_e=(N_ele if(nspin==1) else N_ele[spin]); ### print("#occ:",end="");print(occ)  ## RHF:Nele_{up+dn}  UHF:Nele_sigma
        dev=max( dev, abs(v-n_e))
        if(verbose):
            printout("#trace_dmat:"+strspin+":%14.6f / occsum=%14.6f"%(v,n_e))
    if(dev_abort_thr is not None):
        if(dev_abort_thr>0 and dev>=dev_abort_thr):
            assert False,"too large deviation from Nele:"+str(ret)+"/"+str(N_ele)
    return ret;                

def get_HOMO(this,nth=0):
    return get_LHomo(this,-1-nth)
def get_SOMO(this):
    return get_LHomo(this,0)
def get_LUMO(this,nth=0):
    return get_LHomo(this,1+nth)

def krks_get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''
    modification of KRKS.get_veff
    somehow we need a type cast onto complex
    '''
    import time
    from .utils import printout_dict,update_dict
    
    if( ks.nkpt is not None ):
        assert ks.nkpt == len(ks.kpts),"%d/%s"%(ks.nkpt,str(np.shape(ks.kpts)));
    fncnme='get_veff';fncdepth=1   ## fncdepth ~ SCF:0 vhf:1 get_j:2
    Wctm000=time.time();Wctm010=Wctm000;dic1={}
    Dic=rttddft_common.Dict_getv('timing_'+fncnme, default=None)
    if( Dic is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});
        Dic=rttddft_common.Dict_getv('timing_'+fncnme, default=None)

    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    N_call=rttddft_common.Countup("krks_get_veff");
    check_timing_veff=(N_call<3 or N_call==20) # xxx xxx
    t0 = (time.process_time(), time.perf_counter())
    AUinFS=2.418884326058678e-2
    rtTDDFT_Istep=rttddft_common.Dict_getv("rtTDDFT_Istep",0);rtTDDFT_Iter=rttddft_common.Dict_getv("rtTDDFT_Iter",0)
    time_AU=(0.0 if(not hasattr(ks,"_time_AU")) else ks._time_AU); time_FS=time_AU*AUinFS
    wt_010= rttddft_common.Start_timer("setup");wt_000=wt_010;timing={"misc":0.0}
    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10 or abs(alpha) > 1e-10
    wt_020=wt_010;wt_010=rttddft_common.Stop_timer("setup");timing.update({"setup":wt_010-wt_020})
    ### wt_020=wt_010; wt_010=time.time(); Logger.UpdTiming('setup',wt_010-wt_020)
    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1,Dic,"setup",Wctm010-Wctm020,depth=fncdepth)  
    
    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_rks(ks.with_df, ks.xc, dm, hermi=hermi,
                                       kpts=kpts, kpts_band=kpts_band,
                                       with_j=True, return_j=False)
        pyscflib_logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = pyscflib_logger.timer(ks, 'vxc', *t0)
        return vxc

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm, np.ndarray) and dm.ndim == 3 and
                    kpts_band is None)
# For UniformGrids, grids.coords does not indicate whehter grids are initialized
    if ks.grids.non0tab is None:
        wt_020=wt_010;wt_010=rttddft_common.Start_timer("setup_grids");timing['misc']+=wt_010-wt_020
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm, ks.grids, kpts)
        ### wt_020=wt_010; wt_010=time.time(); Logger.UpdTiming('setup_grids',wt_010-wt_020)
        wt_020=wt_010;wt_010=rttddft_common.Stop_timer("setup_grids");timing.update({"setup_grids":wt_010-wt_020})
        t0 = pyscflib_logger.timer(ks, 'setting up grids', *t0)
        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1,Dic,"setup_grids",Wctm010-Wctm020,depth=fncdepth)  

    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        ### print("#KRKS.veff:hermi(%d)/=2"%(hermi))  # 1
        wt_020=wt_010;wt_010=rttddft_common.Start_timer("nr_rks");timing['misc']+=wt_010-wt_020
        N_nr_rks=rttddft_common.Countup("krks_get_veff.nr_rks")
        wt_x0=time.time();
        n, exc, vxc = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, hermi=0,
                                        kpts=kpts, kpts_band=kpts_band)
        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1,Dic,"nr_rks",Wctm010-Wctm020,depth=fncdepth)  
        wt_020=wt_010;wt_010=rttddft_common.Stop_timer("nr_rks");timing.update({"nr_rks":wt_010-wt_020})
        wt_x1=time.time();wt_cur=wt_x1-wt_x0
        pyscflib_logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = pyscflib_logger.timer(ks, 'vxc', *t0)

    weight = 1./len(kpts)
    if not hybrid:
        wt_020=wt_010;wt_010=rttddft_common.Start_timer("get_j");timing['misc']+=wt_010-wt_020
        
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        wt_020=wt_010;wt_010=rttddft_common.Stop_timer("get_j");timing.update({"get_j":wt_010-wt_020})
        
        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1,Dic,"get_j",Wctm010-Wctm020,depth=fncdepth)  

        if( (np.array(vxc).dtype == np.float64 or np.array(vxc).dtype == float) and 
            (np.array(vj).dtype == np.complex128 or np.array(vj).dtype == complex) ):
            vxc=tocomplex(vxc)
        vxc += vj
    else:
        ### print("#KRKS.veff:Hybrid")
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk * .5

        if ground_state:
            exc -= np.einsum('Kij,Kji', dm, vk).real * .5 * .5 * weight

    if ground_state:
        wt_020=wt_010;wt_010=rttddft_common.Start_timer("ecoul");timing['misc']+=wt_010-wt_020

        ecoul = np.einsum('Kij,Kji', dm, vj).real * .5 * weight
        wt_020=wt_010;wt_010=rttddft_common.Stop_timer("ecoul");timing.update({"ecoul":wt_010-wt_020})
        ## wt_020=wt_010; wt_010=time.time(); Logger.UpdTiming('ecoul',wt_010-wt_020)
        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1,Dic,"ecoul",Wctm010-Wctm020,depth=fncdepth)  

    else:
        ecoul = None

    wt_020=wt_010; wt_010=rttddft_common.Get_time();

    wt_100=rttddft_common.Get_time()
    if( check_timing_veff ):
        fpath1="rttddft_get_veff_timing.log"
        legend="#%06s %6s %4s %14s \t\t %14s "%("N_call", "Istep", "Iter", "time_FS", "walltime")
        text=" %06d %6d %4d %14.4f \t\t %14.4f   "%(N_call, rtTDDFT_Istep, rtTDDFT_Iter, time_FS,wt_100-wt_000)\
             +str(timing)
        printout("#rttddft.krks_get_veff:elapsed:%14.4f "%(wt_010-wt_000)+text)
        if(N_call==1):
            printout(legend,fpath=fpath1,Append=True,Threads=[0])
        printout(text,fpath=fpath1,Append=True,Threads=[0])
        ### fdOUT.close()
    rttddft_common.Print_timing('Get_Veff',['setup',"setup_grids", "nr_rks","get_j_1st","get_j_2nd", "ecoul"],
                              walltime=wt_100-wt_000);timing.update({"Get_Veff":wt_100-wt_000})
    Wctm010=time.time()
    printout_dict(fncnme,dic1,Dic,Wctm010-Wctm000,depth=fncdepth)

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def get_Nele(this,Occ=None):
    # Occ:[spin][nKpt][nMO]
    if( Occ is None ):
        Occ=this._mo_occ

    #print("#get_Nele:Occ:"+str(np.shape(Occ)))
    #print("#get_Nele:Occ:"+str(Occ))

    nspin=1; nKpt=1
## 2021.05.24 fix    if( this._spinrestriction == 'U'):
    if( this._spinrestriction != 'R'): 
        assert len(Occ)==2,"";nspin=2
        Ndim1=np.shape(Occ[0]);
        if( this._pbc ): ## [nKpt][nMO]
            assert len(Ndim1)==2,""; nKpt=Ndim1[0];
    else:
        Ndim1=np.shape(Occ);
        if( this._pbc ):
            assert len(Ndim1)==2,""; nKpt=Ndim1[0];

    ret=[]
    for spin in range(nspin):
        occ1=( Occ if(nspin==1) else Occ[spin])
        tot=0.0
        for kp in range(nKpt):
            occ11=( occ1 if(not this._pbc) else occ1[kp]);
            cum=sum(occ11);
            tot+=cum
        if(this._pbc):
            tot=tot/float(nKpt)
        ret.append(tot)
    if(nspin==1):
        ret=ret[0];
    return ret

def set_nOrbs(this,nkpt=None,nAO=None,nMO=None):
    if( nkpt is not None ):
        this.nkpt=nkpt
    if( nAO is not None ):
        this.nAO=nAO
    if( nMO is not None ):
        this.nMO=nMO

def update_nOrbs(this,force=False):
    ##        non-PBC       PBC 
    ##  O,R   nAO,nMO       nKPT,nAO,nMO
    ##  U     2,nAO,nMO     2,nKPT,nAO,nMO
    if( not force ):
        if( (this.nAO is not None) and (this.nkpt is not None) and (this.nMO is not None) ):
            return 0
    if( this.mo_coeff is None ):
        assert False,""
        Logger.info(this._logger,"get_len but mo_coeff is None");
        return -1
    Ndim=np.shape(this.mo_coeff)
    if( this._spinrestriction=='U'):
        
        if( this._pbc ):
            assert Ndim[0]==2,""; this.nkpt=Ndim[1]; this.nAO=Ndim[2]; this.nMO=Ndim[3]
        else:
            assert Ndim[0]==2,""; this.nkpt=1;       this.nAO=Ndim[1]; this.nMO=Ndim[2]
        printout("#UKS %d %d %d mo_coeff:"%(this.nkpt,this.nAO,this.nMO)+str(Ndim))
    else:
        if( this._pbc ):
            this.nkpt=Ndim[0]; this.nAO=Ndim[1]; this.nMO=Ndim[2];
        else:
            this.nkpt=1;       this.nAO=Ndim[0]; this.nMO=Ndim[1];
    return 1


def get_LHomo(this,nth):
    occ=this._mo_occ
    if( occ is None ):
        printout("#get_LHOMO:recalculating occ..")
        occ=this.get_occ()
    occ_a=(occ if(this._spinrestriction != 'U') else occ[0])
    occ_a0=(occ_a if(not this._pbc) else occ_a[0])
    this.update_nOrbs()
    nMO=this.nMO
    thr=1.0;LUMO=-1
    for i in range(nMO):
        if( occ_a0[i]<thr ):
            LUMO=i;printout("#occ[%d]:%f /[%d]:%f"%(i,occ_a0[i],max([0,i-1]),occ_a0[ max([0,i-1]) ]));break
    if( LUMO< 0):
        printout("OCC:"+str(occ_a0))
        assert False,"get_LHOMO"
    if( occ_a0[LUMO]>0.5):
        ## this is SOMO..
        SOMO=LUMO
        if( nth==0 ):
            retv= SOMO
        else:
            retv= SOMO+nth
        if( retv<0 or retv>nMO ):
            printout("!W get_LHomo:nth=%d >> retv=%d (SOMO=%d nMO=%d)"%(nth,retv,SOMO,nMO) )
        return retv
    else:
        ## this is LUMO..
        retv= LUMO+nth-1   # 1=LUMO 2=LUMO+1

        if( retv<0 or retv>nMO ):
            printout("!W get_LHomo:nth=%d >> retv=%d (LUMO=%d nMO=%d)"%(nth,retv,LUMO,nMO) )
        return retv

def print_calc_gs(this,Egnd,CPUtime=None,Niter=None,legend=False,fname=None,appendtofile=False):
    if(suppress_prtout()):
        return

    string=" ";strlegend="#"
    if(this.nAO is not None):
        string+=" %5d "%(this.nAO);strlegend+=" %5s "%("Nao")
    if(this._pbc):
        if( getattr(this.cell,"mesh",None) is not None ):
            mesh=getattr(this.cell,"mesh",None);
            meshsz=mesh[0]*mesh[1]*mesh[2];
            string+="  %8d %4d %4d %4d  "%(meshsz,mesh[0],mesh[1],mesh[2])
            strlegend+="  %8s %4s %4s %4s  "%("meshsz","mesh","mesh","mesh")
    string+=" %16.8f "%(Egnd);strlegend+=" %16s "%("Egnd")
    if( CPUtime is not None):
        string+=" %14.4f "%(CPUtime);strlegend+=" %14s "%("CPUtime")
        if( Niter is not None and Niter>0 ):
            string+=" %5d  %14.4f "%(Niter,CPUtime/Niter);strlegend+=" %5s %14s "%("Niter","time_per_Iter")
    if( rttddft_common.get_job(False) is not None ):
        string+=rttddft_common.get_job(False)
    string+="  "+str(datetime.datetime.now())

    fdlist=[sys.stdout];fdout=None
    if( fname is not None ):
        fdout=open(fname,("a" if(appendtofile) else "w"))
        fdlist.append(fdout  )
    for fd1 in fdlist:
        if(legend):
            print(strlegend,file=fd1);
        print(string,file=fd1)
    if(fdout is not None):
        fdout.close()

def calc_gs(this,update_Matrices=True,verbose=False,dm0=None):
    assert (not this._fix_occ),""
    t0=time.time()
    En=this.kernel(dm0=dm0)
    t1=time.time()

    check_time=True
    if(check_time):
        print_calc_gs(this,En,CPUtime=t1-t0,Niter=None,legend=True,fname="calc_gs.log",appendtofile=True)

    this._Egnd=En
    print("#rttddft:calc_gs:%16.8f N_ele:%f"%(En,get_Nele(this,Occ=this.mo_occ)))
    if( update_Matrices ):
        this.update_Sinv(refresh=True)
    return En

def set_mo_occ(this,mo_occ,fix_occ,clone=False):
    if( not clone ):
        this._mo_occ=mo_occ
    else:
        this._mo_occ=arrayclone(mo_occ)
    this._fix_occ=fix_occ
    this.get_logger(True).Info("set_mo_occ:"+d1toa(np.ravel(np.array(mo_occ))))

def update_Sinv(this,refresh=False):
    need_update=refresh
    if( not need_update ):
        if( this._sINV is not None):
            return False
    pbc=this._pbc
    if( not pbc ):
        S1e_1=this.get_ovlp()
        this._Ssqrt,this._Sinvrt,this._sINV=get_Ssqrt_invsqrt_sINV(S1e_1)
    else:
        kvectors=np.reshape( this.kpts, (-1,3))
        nkpt=len(kvectors)
        this._Ssqrt=[];this._Sinvrt=[];this._sINV=[]
        for k in range(nkpt):
            S1e_k=this.get_ovlp(this.cell,kvectors[k])
            Ssqrt,Sinvrt,sINV = get_Ssqrt_invsqrt_sINV(S1e_k)
            this._Ssqrt.append(Ssqrt);this._Sinvrt.append(Sinvrt);this._sINV.append(sINV);
    return True

def energy_tot(mf, dm=None, h1e=None, vhf=None,verbose=False):
    nuc = mf.energy_nuc()
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + nuc
    mf.scf_summary['nuc'] = nuc.real
    return e_tot

def energy_elec_krks(this, dm_kpts=None, h1e_kpts=None, vhf=None,verbose=False):
    logger=this.get_logger(set=True)
    if(verbose):
        logger.info("#rttddftPBC:energy_elec_krks ---")
    if( dm_kpts is None ):
        if( verbose ):
            logger.info("#rttddftPBC:MOcofs:calculating dm_kpts from Cofs..")
            moCofs=this.mo_coeff[0]
            HOMO=get_HOMO(this)
            logger.info("#MO_0:"+z1toa(moCofs[:,0]))
            logger.info("#MO_%d(HOMO):"%(HOMO)+z1toa(moCofs[:,HOMO]))
        dm_kpts = this.make_rdm1()
        if(verbose):
            logger.info("#dm_00:%f  _01:%f+j%f _11 %f ..."%(
                dm_kpts[0][0][0].real,dm_kpts[0][0][1].real,dm_kpts[0][0][1].imag,dm_kpts[0][1][1].real) )
    retv = this.get_super().energy_elec(dm_kpts=dm_kpts,h1e_kpts=h1e_kpts,vhf=vhf)
    if(verbose):
        Enuc=this.energy_nuc().real
        Etot = retv + Enuc
        logger.info("#E_el:%f  / E_tot:%f / E_GS:%f..."%(retv,Etot,this._Egnd) )
    return retv
    
def energy_elec_rks(this, dm=None, h1e=None, vhf=None, verbose=False):
    logger=this.get_logger(set=True)
    if(verbose):
        logger.info("#rttddftMOL:energy_elec_rks ---")
    if( dm is None ):
        if( verbose ):
            logger.info("#rttddftMOL:MOcofs:calculating dm from Cofs..")
            moCofs=this.mo_coeff
            HOMO=get_HOMO(this)
            logger.info("#MO_0:"+z1toa(moCofs[:,0]))
            logger.info("#MO_%d(HOMO):"%(HOMO)+z1toa(moCofs[:,HOMO]))
        dm = this.make_rdm1()
        if(verbose):
            logger.info("#dm_00:%f  _01:%f+j%f _11 %f ..."%(dm[0][0].real,dm[0][1].real,dm[0][1].imag,dm[1][1].real) )
    retv = this.get_super().energy_elec(dm=dm,h1e=h1e,vhf=vhf)
    if(verbose):
        Enuc=this.energy_nuc().real
        Etot = retv + Enuc
        logger.info("#E_el:%f  / E_tot:%f / E_GS:%f..."%(retv,Etot,this.Egnd) )
    return retv

def get_Rnuc(this,unit):
    fac=1.0
    assert (unit=='A' or unit=='ANGS' or unit=='B' or unit=='BOHR'),"unit:"+unit
    if( unit=='A' or unit=='ANGS'):
        fac=physicalconstants.BOHRinANGS
    pbc=this._pbc
    if( not pbc ):
        return np.array( this.mol.atom_coords() )*fac
    else:
        return np.array( this.cell.atom_coords() )*fac


def get_logger(self,set=False):
    if( (self._logger is None) and set ):
        printout("get_logger:self._logger is None..."+str(id(self)));
        self.set_logger(filename="rttddft_dbg.log");
    return self._logger;

def set_logger(mf,logger=None,filename=None,replace=True):
    if( not replace ):
        if( mf._logger is not None):
            printout("#rttddftPBC already has a logger");
            return False
    if( logger is None ):
        assert (filename is not None),""
        printout("set_logger:setting Logger..."+str(id(mf)) );
        mf._logger = Logger(filename)
    else:
        mf._logger = logger
    return True

def get_populations(this,dm_kpts, title="", kpts=None):
    ## pop[\mu] = D_{\mu,k'} S_{k',\mu} 
    ##            non-pbc           pbc 
    ## RKS        [nAO]             [nkpt][nAO]
    ## UKS,ROKS   [2][nAO]          [2][nkpt][nAO]
    if( this._spinrestriction == 'R'):
        return get_populations_1(this,dm_kpts, title=title, kpts=kpts)
    else:
        return get_populations_1(this,dm_kpts[0], title=title, kpts=kpts),\
               get_populations_1(this,dm_kpts[1], title=title, kpts=kpts)

def get_populations_1(this,dm_kpts, title="", kpts=None):
    pbc=this._pbc
    
    nkpt=(1 if (not pbc) else len(dm_kpts))
 
    nAO=( len(dm_kpts) if (not pbc) else len(dm_kpts[0]) )

    kvectors=(None if (not pbc) else np.reshape( this.kpts, (-1,3)))

    wks=None; ## np.zeros([nAO,nAO],dtype=np.complex128 )

    ret=[]
    for kp in range(nkpt):
        if( not pbc ):
            S1e_k=this.get_ovlp()
            wks=np.matmul( S1e_k, dm_kpts)
        else:
            if( kpts is not None ):
                if( not (kp in kpts) ):
                    continue
            S1e_k=this.get_ovlp(this.cell,kvectors[kp])
            wks=np.matmul( S1e_k, dm_kpts[kp])
        pop=[]
        for ao in range(nAO):
            pop.append( wks[ao][ao].real )
        popsum=sum(pop)
        printout("#AOpop_%d:%f "%(kp,popsum)+ title +":"+d1toa(pop,format="%5.2f "))
        if( not pbc ):
            return np.array(pop)
        ret.append( pop.copy() )
    if(pbc):
        return np.array( ret )

def calc_phases(mf,time_AU,mocoefs,mocoefs_REF,eorbs_REF,fpath_format,append=False):  #,gnuplot=False):
    if( suppress_prtout() ):
        return

    assert (this._spinrestriction=='R'),"otherwise unimplemented"
    pbc=mf._pbc
    eorbs=( eorbs_REF if(eorbs_REF is not None) else mf.mo_energy )
    ndim=np.shape(mocoefs)

    kvectors=None;
    if( pbc ):
        nkp=ndim[0];nAO=ndim[1];nMO=ndim[2];
        kvectors=(None if (not pbc) else np.reshape( mf.kpts, (-1,3)))
    else:
        nkp=1;nAO=ndim[0];nMO=ndim[1]

    SxC=np.zeros([nAO,nMO],dtype=np.complex128)
    pop=np.zeros([nMO],dtype=float)
    angl=np.zeros([nMO],dtype=float)
    col=np.zeros([nAO],dtype=np.complex128)
    ref=np.zeros([nAO],dtype=np.complex128)
    
    for kp in range(nkp):
        fd=futils.fopen( fpath_format%(kp),("a" if append else "w")) ## OK
        if( not pbc ):
            S1e=mf.get_ovlp()
        else:
            S1e=mf.get_ovlp(mf.cell,kvectors[kp])

        SxC=np.matmul( S1e, mocoefs[kp] )
        print("%f "%(time_AU),end="",file=fd)
        for mo in range(nMO):
            for kao in range(nAO):
                col[kao]=SxC[kao][mo]
            if( not pbc ):
                for kao in range(nMO):
                    ref[kao]=mocoefs_REF[kao][mo]
            else:
                for kao in range(nMO):
                    ref[kao]=mocoefs_REF[kp][kao][mo]
            
            cdum = np.vdot(ref,col)
            pop[mo]= (cdum.real)**2 + (cdum.imag)**2
            angl[mo]=math.atan2( cdum.imag, cdum.real )
            print("%f %f %f %f     "%(angl[mo],eorbs[kp][mo],angl[mo]/eorbs[kp][mo],pop[mo]),end="",file=fd)
        print("",file=fd);
        futils.fclose(fd)

def print_eorbocc(this,title="",file=None):
    if( this._spinrestriction =='U'):
        this.print_eorbocc1(this._mo_occ[0], this.mo_energy[0], title=title,file=file, end="")
        this.print_eorbocc1(this._mo_occ[1], this.mo_energy[1], title=title,file=file)
    else:
        this.print_eorbocc1(this._mo_occ, this.mo_energy, title=title,file=file)

def print_eorbocc1(this,occ,eorbs,title="",file=None,end="\n"):
    if( suppress_prtout() ):
        return
    fd=(sys.stdout if (file is None) else file)
    pbc=this._pbc

    if( not pbc ):
        print("#print_eorbocc_"+title+":"+d1x2toa(this._mo_occ,this.mo_energy,format="%3.1f %10.4f  "),end=end,file=fd )
    else:
        nkpt=len(this.kpts)
        mo_energy_kpts=this.mo_energy
        for kp in range(nkpt):
            print("#K%d:%10.4f %10.4f %10.4f :"%( \
                kp, this.kpts[kp][0],this.kpts[kp][1],this.kpts[kp][2]) \
                +d1x2toa(this._mo_occ_kpts[kp],this.mo_energy[kp],format="%3.1f %10.4f  "),end=end, file=fd)


def get_hcore(this,cell_or_mol=None,kpts=None,tm_AU=None, dm_kOR1=None, tmAU_dmat=None, Dict_hc=None):
    idbgng=0
    diff_ZF_and_fF=False 
    ncall=rttddft_common.count("get_hcore",inc=True)
    pbc=this._pbc
    
    if(pbc and (this.nkpt is not None) ):
        assert this.nkpt == len(this.kpts),"%d/%s"%(this.nkpt,str(np.shape(this.kpts)));

    dbgng_constantfield=True ## TODO 
    if(this._fixednucleiapprox):
        if( this._constantfield and (this._td_field is not None) ):
            ncall_cf =rttddft_common.count("get_hcore.cf",inc=True)
            do_calc_hc=( this._hcore is None )
            print("#get_hcore.constantfield:%05d:start:%r"%(ncall,do_calc_hc))
            if( this._hcore is None ):
                print("#get_hcore.constantfield:%05d:get_hcore new"%(ncall_cf) )
            if( dbgng_constantfield ):
                # we assume 1st:CALC 2nd:cpy, ...
                if(ncall_cf==3 or ncall_cf==20 or ncall_cf==100 or ncall_cf==3000):
                    print("#get_hcore.constantfield:%05d:get_hcore for DBG.."%(ncall_cf))
                    do_calc_hc=True
            if( (not do_calc_hc) and (this._hcore is not None) ):
                return arrayclone( this._hcore )
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    fncnme="get_hcore";fncdepth=1   ## fncdepth ~ SCF:0 vhf:1 get_j:2
    Wctm000=time.time();Wctm010=Wctm000;N_call=rttddft_common.Countup("get_hcore");dic1_timing={}
    Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)
    if( Dic_timing is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});
        Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)
    AUinFS=PhysicalConstants.aut_in_femtosec()
    cLIGHT_AU=PhysicalConstants.CLIGHTinAU()
    assert ( abs( cLIGHT_AU - 137.035999)<2.0e-6),"check cLIGHT_AU:%f"%(cLIGHT_AU)

    A_over_c=None
    if( this._calc_Aind !=0 ):
        assert this._dt_AU is not None,"PLS set dt_AU ..."
        assert dm_kOR1 is not None,"PLS set dm_kOR1 ...:%d"%(this._calc_Aind)
        Aind_over_c = calc_Aind_over_c( this, (tm_AU if(tm_AU is not None) else this._time_AU), this._dt_AU, 
                                       dm_kOR1, tm_dmat=tmAU_dmat,caller="get_hcore")
        Aext_over_c = this._td_field.get_vectorfield(tm_AU)/cLIGHT_AU
        A_over_c = Aext_over_c + Aind_over_c
    else:
        tmAU_refr=( tm_AU if(tm_AU is not None) else this._time_AU)
        Aext_over_c = (np.zeros([3],dtype=np.float64) if(this._td_field is None) else this._td_field.get_vectorfield(tmAU_refr)/cLIGHT_AU);
        A_over_c=Aext_over_c

    if( this._iflag_prtout_field!=0 ):
        headline=""
        if( this._iflag_prtout_field < 0 ):
            headline=" %6s %14s    %16s %16s %16s\n"%("step","tm_AU","A-over-c_x","A-over-c_y","A-over-c_z")
        printout( headline+" %6d %14.6f    %16.8f %16.8f %16.8f"%(abs(this._iflag_prtout_field),tm_AU, A_over_c[0], A_over_c[1], A_over_c[2]),\
                  fpath=rttddft_common.get_job(True)+"_hcore_AoverC.log",Append=(this._iflag_prtout_field>0));
        this._iflag_prtout_field=0;


    if( Dict_hc is not None ):
        Dict_hc.update({"A_over_c":arrayclone(A_over_c)})
    nkpt=1;kvectors=None
    if(pbc):
        nkpt=len(this.kpts)
        kvectors=np.reshape( this.kpts, (-1,3))
    
    if( tm_AU is None ):
        tm_AU = this._time_AU
    
    ## hcore ([nkpt])[nAO][nAO]   for all types of spin restriction 
    pseudo=False
    if( pbc and this.cell.pseudo):
        pseudo=True
    calc_hcZF=0;calc_hcA0=0;calc_hcSF=0; hcZF=None;hcA0=None; hcSF=None; tdF=this._td_field;
    if( this._td_field is not None ):
        if( pseudo ):
            if(pbc):
                Dic1={}
                hcore= krhf_get_hcore_pp(this, cell_or_mol, kpts=kpts,tm_AU=tm_AU,Dict_DBG=Dic1, A_over_c=A_over_c)
                                                                                                #A_over_c is automatically calculated (as a purely external field) even if it is None
                if( Dict_hc is not None ):
                    Dict_hc.update({"hcore_00":arrayclone(hcore)})

                Wctm020=Wctm010;Wctm010=time.time()                                                      
                update_dict(fncnme,dic1_timing,Dic_timing,"get_hcore_pp",Wctm010-Wctm020,depth=fncdepth)
                assert Dic1['get_pp']=='new',""
            else:
                assert False,""
        else:
            if( not pbc ):
                hcore = this.get_super().get_hcore(cell_or_mol)

                Wctm020=Wctm010;Wctm010=time.time()
                update_dict(fncnme,dic1_timing,Dic_timing,"super.get_hcore",Wctm010-Wctm020,depth=fncdepth)
            else:
                hcore= this.get_super().get_hcore(cell_or_mol,kpts);  ##  wrong:(cell=this.cell, kpts=kvectors) 2020.11.12 fixed
                assert nkpt==len(hcore),"nkpt:%s / len(hcore):%d hcore:"%(str(nkpt),len(hcore))+str(np.shape(hcore))

                Wctm020=Wctm010;Wctm010=time.time()
                update_dict(fncnme,dic1_timing,Dic_timing,"super.get_hcore_PBC",Wctm010-Wctm020,depth=fncdepth)
            if( Dict_hc is not None ):
                Dict_hc.update({"hcore_00":arrayclone(hcore)})

        if( this.dipolegauge ):
            ## - q_e r:E = +|e| r:E
            Efield = this._td_field.get_electricfield(tm_AU)
            if( not pbc ):
                dipole_kpt = this.mol.intor('int1e_r', comp=3)
                for dir in range(3):
                    hcore[:][:] = hcore[:][:] + Efield[dir] * dipole_kpt[dir][:][:]
                if( Dict_hc is not None ):
                    Dict_hc.update({"hcore_field":arrayclone( Efield[0]*dipole_kpt[0][:][:] \
                                                  + Efield[1]*dipole_kpt[1][:][:] + Efield[2]*dipole_kpt[2][:][:] )})
            else:
                dipole_kpt = this.cell.pbc_intor('int1e_r', comp=3, hermi=1, kpts=kvectors)
                assert (len(elgrad_kpt)==nkpt),"shape:"
                for kp in range(nkpt):
                    for dir in range(3):
                        hcore[kp][:][:] = hcore[kp][:][:] + Efield[dir] * dipole_kpt[kp][dir][:][:]
                if( Dict_hc is not None ):
                    Dict_hc.update({"hcore_field":arrayclone( hcore - Dict_hc["hcore_00"] )})
                        
            return hcore
        elif( this.velocitygauge ):
            ## print("#get_hcore.constantfield:velocitygauge...") 
            hcore=toComplexArray(hcore)
            ## - p: q_e A/c = p|e|A/c = -i (nabla) A/c (a.u.)
            if( A_over_c is None ):
                A_over_c = this._td_field.get_vectorfield(tm_AU)/cLIGHT_AU

            if( (tm_AU < 0.10/AUinFS) or int( round( (tm_AU/AUinFS)/0.008 ) )%100==0 ):
                if( MPIrank == 0 ):
                    fd01=open(rttddft_common.get_job(True)+"_tdfield.log","a");
                    print("%14.6f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f"%(tm_AU,A_over_c[0],A_over_c[1],A_over_c[2],
                        Aext_over_c[0],Aext_over_c[1],Aext_over_c[2]),file=fd01);fd01.close()

            if( not pbc ):
                elgrad_kpt = this.mol.intor('int1e_ipovlp',comp=3, hermi=2)
                # comp : Components of the integrals, e.g. int1e_ipovlp_sph has 3 components
                # hermi: Hermitian:1  anti-Hermitian:2
            else:
                elgrad_kpt = this.cell.pbc_intor('int1e_ipovlp',comp=3, hermi=2, kpts=kvectors)
                assert (len(elgrad_kpt)==nkpt),"shape:"

            if( not pbc ):
                for dir in range(3):
                    hcore[:][:] = hcore[:][:] - 1j * Constants.sgn_int1e_ipovlp * A_over_c[dir] * elgrad_kpt[dir][:][:]

                S1e_1=this.get_ovlp()
                quiver_energy = 0.50 * np.dot( A_over_c, A_over_c )
                hcore[:][:] = hcore[:][:] + quiver_energy * S1e_1[:][:]

                if( Dict_hc is not None ):
                    Dict_hc.update({"hcore_field":arrayclone( - 1j * Constants.sgn_int1e_ipovlp * ( A_over_c[0]*elgrad_kpt[0][:][:] \
                                                                + A_over_c[1]*elgrad_kpt[1][:][:] + A_over_c[2]*elgrad_kpt[2][:][:] ) \
                                                              + quiver_energy * S1e_1[:][:])})

            else:
                # - p:(e/mc)A = + (\hbar/i) p:A/c

                for kp in range(nkpt):
                    for dir in range(3):
                        hcore[kp][:][:] = hcore[kp][:][:] - 1j * Constants.sgn_int1e_ipovlp * A_over_c[dir] * elgrad_kpt[kp][dir][:][:]
                for kp in range(nkpt):
                    S1e_k=this.get_ovlp( this.cell, kvectors[kp])
                    quiver_energy = 0.50 * np.dot( A_over_c, A_over_c )
                    hcore[kp][:][:] = hcore[kp][:][:] + quiver_energy * S1e_k[:][:]

                    if( (hcZF is not None) and diff_ZF_and_fF and (np.sqrt(2*quiver_energy)>1.0e-8) ):
                        ttl1="rttddft01_get_hcore_hcZF_and_hcfF_K%d"%(kp)
                        print_Hmatrices(ttl1,hcore[kp],hcZF[kp],Nref=7,fpath=ttl1+".dat")
                                
                if( Dict_hc is not None ):
                    Dict_hc.update({"hcore_field":arrayclone( hcore - Dict_hc["hcore_00"] )})
            Wctm020=Wctm010;Wctm010=time.time()
            update_dict(fncnme,dic1_timing,Dic_timing,"dipvel",Wctm010-Wctm020,depth=fncdepth)
            # Wctm010=time.time()
            # printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth)

            if(this._fixednucleiapprox):
                if(this._constantfield):
                    if( this._hcore is not None ):
                        diff=aNmaxdiff( this._hcore, hcore )
                        if(MPIrank==0):
                            fd1=open("constantfield.log","a");
                            print("#get_hcore.%05d diff:%e elapsed:%f"%(ncall_cf,diff,Wctm010-Wctm000),file=fd1)
                            fd1.close()
                        assert diff<1e-7,"diff:%e"%(diff)
                    else:
                        if(MPIrank==0):
                            fd1=open("constantfield.log","a");
                            print("#get_hcore.%05d set afresh:%s elapsed:%f"%(ncall_cf, str(np.shape(hcore)),Wctm010-Wctm000),file=fd1)
                            fd1.close()
                    this._hcore=arrayclone( hcore )
                    print("#get_hcore.constantfield:set hcore:",np.shape(this._hcore))
            return hcore
        else:
            assert False,"wrong gauge %r,%r"%(this.dipolegauge,this.velocitygauge)
    else:
        if( not pbc ):
            return this.get_super().get_hcore(cell_or_mol)
        else:
            return this.get_super().get_hcore(cell_or_mol,kpts); ## wrong:(cell=this.cell, kpts=this.kpts)

def rttddft_sget_Aind_args_(SorG,this,Dic,errorlogfnme=None,verbose=False):
    #keys_Aind_over_c= ["_Aind_over_c_ini:D3",     "_calc_Aind:i","_dt_AU:d",     "BravaisVectors_au:D3,3","cell_volume:d",
    #      "_d2Aind_over_c_dt2:d","_d2Aind_over_c_dt2_tmAU:d",    "_Aind_tmAU:d","_Aind_Gearvc_C:D5,3","_Aind_Gearvc_P:D5,3",
    #      "_Aind_Gearvc_C_nxt:D5,3","_Aind_Gearvc_P_nxt:D5,3","_Aind_tmNxt:d"]
    assert this._pbc,"are you going to set Aind_over_c for system without pbc ??"
    Threads=[0];comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();MPIsize=comm.Get_size()
    if(errorlogfnme is None):
        errorlogfnme="rttddft_%set_Aind_args_error.log"%(SorG);
    
    dbglogfnme="rttddft_%set_Aind_args.log"%(SorG);
    def append_log1(ith,k,t,v):
        if( MPIsize<2 or (MPIrank in Threads) ):
            fdL=open(dbglogfnme,('a' if(ith>0) else 'w'))
            print("%03d:"%(ith)+ k+":"+str(t)+":"+obj_to_string(v));fdL.close()

    def print_error1(msg):
        if( MPIsize<2 or (MPIrank in Threads) ):
            msg+="  \t\t\t"+str(datetime.datetime.now())
            fdE=open(errorlogfnme,"a");print(msg,file=fdE);print(msg);fdE.close()
    assert SorG=='S' or SorG=='G',""
    set_args=(SorG=='S' or SorG=='s')
    nerror=0;ith=0
    for item in rttddftPBC.Keys_Aind_over_c_:
        sA=item.split(':');nA=len(sA);assert nA==2,"wrong format:"+str(item)
        ky=sA[0].strip(); ty=sA[1];
        if(set_args):
            if(ky not in Dic):
                print_error1(ky+" not in Dic");nerror+=1;continue
            setattr(this,ky,Dic[ky]);
            if(verbose):
                append_log1(ith,ky,ty,Dic[ky])
        else:
            val=getattr(this,ky,None)
            if(val is None):
                if( ky in rttddftPBC.Keys_Aind_over_c_allow_None_ ):
                    printout("#rttddft_sget_Aind_args:%s allows None"%(ky)); ## go down and set {ky:None} ...
                else:
                    print_error1(ky+" does not exist in this rttddft instance");nerror+=1;continue
            Dic.update({ky:val})
            if(verbose):
                append_log1(ith,ky,ty,val)
        ith+=1

    if(nerror>0):
        print("%set args failed:"%(SorG),Dic);

    return nerror

def printout_matrix(path,Matrix,Append=False,fopen=False,job=True,dtme=True,step=None,time_AU=None):
    def fnformat_Z(arg):
        return "%12.6f,%12.6f"%(arg.real,arg.imag)
    def fnformat_R(arg):
        return "%14.6f"%(arg)
    def fprtD2(mat,fd,format="%10.5f",delimiter=" ",end="\n\n"):
        ndim=np.shape(mat)
        for i in range(ndim[0]):
            print(delimiter.join( [ format%(mat[i][j]) for j in range(ndim[1]) ]),file=fd)
        print(end,file=fd)
    def fprtZ2(mat,fd,format="%9.4f,%9.4f",delimiter="  ",end="\n\n"):
        ndim=np.shape(mat)
        for i in range(ndim[0]):
            print(delimiter.join( [ format%(mat[i][j].real,mat[i][j].imag) for j in range(ndim[1]) ]),file=fd)
        print(end,file=fd)

    AUinFS=physicalconstants.aut_in_femtosec
    fd=open(path,("a" if(Append) else "w"))
    Ndim=np.shape(Matrix);rank=len(Ndim)
    dtype=( np.array( Matrix[0]).dtype if( (not isinstance(Matrix,np.ndarray)) and rank>1) else np.array(Matrix).dtype )
    fnformat=( fnformat_Z if(dtype==np.complex128 or dtype==complex) else fnformat_R )
    fprtA2 =( fprtZ2 if(dtype==np.complex128 or dtype==complex) else fprtD2 )

    print("# xVVx matrix %s %s dim:%s    job:%s  %s"%(\
        ("at step %04d"%(step) if(step is not None) else ""),\
        ("t= %12.6f fs"%(time_AU*AUinFS) if(time_AU is not None) else ""),\
        str(Ndim), ( rttddft_common.get_job(True) if(job) else ""), (str(datetime.datetime.now()) if(dtme) else "")),file=fd)
    
    ia=np.zeros([rank],dtype=int)        
    for ia[0] in range(Ndim[0]):
        if(rank==1):
            print(" %3d:"%(ia[0])+fnformat(Matrix[ ia[0] ]),file=fd);continue
        if(rank==3):
            print("#%3d:"%(ia[0]),file=fd);fprtA2(Matrix[ia[0]],fd);continue
            
        for ia[1] in range(Ndim[1]):
            if(rank==2):
                print(" %3d,%3d:"%(ia[0],ia[1])+fnformat(Matrix[ ia[0] ][ ia[1] ]),file=fd);continue
            if(rank==4):
                print("#%3d,%3d:"%(ia[0],ia[1]),file=fd); fprtA2(Matrix[ ia[0] ][ ia[1] ],fd);continue

            for ia[2] in range(Ndim[2]):
                if(rank==3):
                    print(" %3d,%3d,%3d:"%(ia[0],ia[1],ia[2])+fnformat(Matrix[ ia[0] ][ ia[1] ][ ia[2] ]),file=fd);continue
                if(rank==5):
                    print("#%3d,%3d,%3d:"%(ia[0],ia[1],ia[2]),file=fd); fprtA2(Matrix[ ia[0] ][ ia[1] ][ ia[2] ],fd);continue

                for ia[3] in range(Ndim[3]):
                    if(rank==4):
                        print(" %3d,%3d,%3d,%3d:"%(ia[0],ia[1],ia[2],ia[3])+fnformat(Matrix[ ia[0] ][ ia[1] ][ ia[2] ][ ia[3] ]),file=fd);continue
                    for ia[4] in range(Ndim[3]):
                        if(rank==5):
                            print(" %3d,%3d,%3d,%3d,%3d:"%(ia[0],ia[1],ia[2],ia[3],ia[4])+fnformat(Matrix[ ia[0] ][ ia[1] ][ ia[2] ][ ia[3] ][ ia[4] ]),file=fd);continue
                        else:
                            assert False,"%d"%(rank)+str(Ndim)
    fd.close()
    if(fopen):
        os.system("fopen "+path)

def load_Aind_over_c(this,tm_AU,dt_AU,fpath):
    def d1diff(lhs,rhs):
        le=len(lhs)
        if(len(rhs)!=le):
            return -1
        ret=0.0
        for j in range(le):
            ret=max( ret, abs(lhs[j]-rhs[j]))
        return ret
    IpK=0; Indcsrtn=[2,3,4]
    dtTINY=dt_AU*0.01;diff_TINY=1.0e-7
    fd=open(fpath,"r")
    Dic={};pKmaxv=-1;spKmax=None;erbuf=[]
    spKwant=None;N_dupl=0
    for line in fd:
        line=line.strip();le=len(line)
        if(le<1):
            continue
        sA=line.split();nA=len(sA);
        spK=sA[IpK]
        arr=np.array( [ float(sA[k]) for k in range(nA) ] )
        pK=arr[IpK]
        if( pKmaxv<0 or pK>= pKmaxv + dt_AU - dtTINY):
            Dic.update({spK:arr})
        else:
            if(spK in Dic):
                old=Dic[spK];
                dff=d1diff(old,arr);
                if(dff < diff_TINY ):
                    N_dupl+=1
                else:
                    strwrn="#Conflicting data:#diff=%16.6e\n#Conflicting data:%s_1:%s\n#Conflicting data:%s_2:%s"%(
                            dff,spK,str(old),spK,str(arr))
                    erbuf.append(strwrn)
            else:
                strwrn="#check pK is correct: pKlast=%f / pKnew=%f but %s is not in Dic.."%(pKlast,pK,spK)
                erbuf.append(strwrn)
                Dic.update({spK:arr})
        if(pKmaxv<pK):
            pKmaxv=pK;spKmax=spK
        if(abs(pK-tm_AU)<dtTINY):
            spKwant=spK
    fd.close()

    print("#load_Aind_over_c:Ndupl:%d Nerror:%d %f/%f\n"%(N_dupl,len(erbuf),pKmaxv,tm_AU),spKwant)
    assert spKwant is not None,""
    if( len(erbuf) > 0 ):
        fd=open("load_Aind_over_c_err.log","w")
        for item in erbuf:
            print(item,file=fd);
        fd.close()
    if(spKwant is not None):
        arr=Dic[spKwant]
        fd=open("load_Aind_over_c.log","a")
        print( [ arr[I] for I in Indcsrtn ],file=fd);fd.close()
        return np.array( [ arr[I] for I in Indcsrtn ] );
    return None

def get_current(this,densitymatrix=None):
    if( densitymatrix is None ):
        densitymatrix = this.make_rdm1()
    ## \sum_{mu>nu} Im(DM_{\mu\nu}) * [ mu GRAD nu - nu GRAD mu ]
    ##                                  
    ##
    dbgng=True
    if(dbgng):
        if( rttddft_common.count("rttddft01.get_current",inc=True)%10==1 ):
            sqrSums={};
            dagDEV,ofdDEV=check_hermicity_dm(this, densitymatrix, sqrSums=sqrSums,title="DM");
    matrix=None
    kvectors=(None if(not this._pbc) else np.reshape( this.kpts, (-1,3)) )
    if( not this._pbc ):
        matrix = this.mol.intor('int1e_ipovlp',comp=3, hermi=2)
    else:
        matrix = this.cell.pbc_intor('int1e_ipovlp',comp=3, hermi=2, kpts=kvectors)
    Ndim_matrix=np.shape(matrix)
    ### print("Ndim_matrix:",end="");print(Ndim_matrix)
    ### print("should_be:",end="");print([3,this.nAO,this.nAO] if(not this._pbc) else [this.nkpt,3,this.nAO,this.nAO]) 
    if( not this._pbc ):
        assert i1eqb(Ndim_matrix,[3,this.nAO,this.nAO],verbose=True),""+str(Ndim_matrix)
    else:
        assert i1eqb(Ndim_matrix,[this.nkpt,3,this.nAO,this.nAO],verbose=True),""+str(Ndim_matrix)
        
    nspin=(1 if(this._spinrestriction=='R') else 2)
    for spin in range(nspin):
        dmat = ( densitymatrix if(nspin==1) else densitymatrix[spin] )
        Ndim=np.shape(dmat)
        nKpt=(1 if(not this._pbc) else Ndim[0])
        nAO=(Ndim[0] if(not this._pbc) else Ndim[1])
        ret=[0.0, 0.0, 0.0]
        for kp in range(nKpt):
            dm=( dmat if(not this._pbc) else dmat[kp])
            mat=( matrix if(not this._pbc) else matrix[kp]) ## [3,nAO,nAO]
            cum=[ 0.0, 0.0, 0.0 ]
            for mu in range(1,nAO):
                for nu in range(mu):
                    for dir in range(3):
                        cum[dir] += dm[mu][nu].imag * Constants.sgn_int1e_ipovlp*(mat[dir][mu][nu]-mat[dir][nu][mu])
            for dir in range(3):
                ret[dir]+=cum[dir]
        if(this._pbc):
            for dir in range(3):
                ret[dir]=ret[dir]/float(nKpt)
    return ret;

def get_dipole(this,dm_kpts,mode,filepath=None,header=None,trailer="", molecular_dipole=None, 
               headline=False,current=None,Aind_over_c=None, Aext_over_c=None, caller="", tmrefAU_Aind=None,
               xVVx_A_over_c_diff_TOL=1.0e-6):
    # dict_retv={'dipole':None, 'dipole_velocity':None}
    # Note your dm[\mu\nu] = C^{^mu}_n w_n { C^{\nu}_n }^{\ast}
    # .. so  TR( matmul(O,dm) ) gives the expectation value.
    # the dipole velocity : TR(...) should give pure-imaginary and -i * value is to be stored..
    AUinFS=0.02418884326058678
    if( tmrefAU_Aind is None ):
        tmrefAU_Aind = this._time_AU
    tmAU_TINY=1.0e-7
    STRwalltime=" \t\t walltime: %12.4f"%( time.time()-this._wt000 )
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();
    dbgng_calcAind=False ## True;
    pbc=this._pbc
    check_DMtrace=True
    if(check_DMtrace):
        trDM=trace_dmat(this, dm_kpts, this._spinrestriction, this._pbc)
        N_ele=get_Nele(this)
        if( this._spinrestriction == 'R' ):
            assert abs(trDM-N_ele)<1.0e-7,""
        else:
            assert ( abs(trDM[0]-N_ele[0])<1.0e-7 and abs(trDM[1]-N_ele[1])<1.0e-7),""

    if(Aind_over_c is None):
        if( this._calc_Aind != 0 ):
            Aind_over_c=get_Aind_over_c(this,this._time_AU)

    kvectors = (None if (not pbc) else np.reshape( this.kpts, (-1,3)))
    nkpt= (None if (not pbc) else len(kvectors) )
     
    cLIGHT_AU = PhysicalConstants.CLIGHTinAU()
    this.update_nOrbs()

    assert ( mode == 'R' or mode=='B' or mode=='V'),""
    ret_RandV=[None, None]
    nmult_DM=(1 if(this._spinrestriction == 'R') else 2)

    dipvel_0=None; dipvel_1=None; dipvel_2=None; dipvel_3=None; dipvel_x0=None
    for RorV in ["R","V"]:
        matrix = None
        if( RorV=='R' and (mode == 'B' or mode==RorV) ):
            if( not pbc ):
                matrix = this.mol.intor('int1e_r',comp=3, hermi=1)
            else:
                matrix = this.cell.pbc_intor('int1e_r', comp=3, hermi=1, kpts=kvectors)

            ndim_matrix=np.shape(matrix)
            if( not pbc ):
                # this should be [3, nAO, nAO] 
                assert ( ndim_matrix[1]==this.nAO and ndim_matrix[2]==this.nAO ),"matrix dimension"+str(ndim_matrix)
            else:
                # this should be [nKpoints, 3, nAO, nAO] 
                assert ( ndim_matrix[0]==this.nkpt and ndim_matrix[2]==this.nAO and ndim_matrix[3]==this.nAO ),"matrix dimension:"+str(ndim_matrix)

            if( nmult_DM==1 ):
                cdum = dmtrace(matrix,3,dm_kpts,pbc,this._spinrestriction)
                ret_RandV[0] = [ cdum[0].real, cdum[1].real, cdum[2].real ]
            else:
                cdumA,cdumB = dmtrace(matrix,3,dm_kpts,pbc,this._spinrestriction)
                ret_RandV[0]=[ cdumA[0].real+cdumB[0].real, cdumA[1].real+cdumB[1].real, cdumA[2].real+cdumB[2].real ] 
        elif( RorV=='V' and (mode == 'B' or mode==RorV) ):
            if( not pbc ):
                matrix = this.mol.intor('int1e_ipovlp',comp=3, hermi=2)
            else:
                matrix = this.cell.pbc_intor('int1e_ipovlp',comp=3, hermi=2, kpts=kvectors)

            if( Constants.sgn_int1e_ipovlp<0 ):
                matrix = - np.array( matrix )

            if( this._spinrestriction == 'R'):
                cdum = dmtrace(matrix,3,dm_kpts,pbc,this._spinrestriction,dict=None)
                ret_RandV[1] = [ cdum[0].imag, cdum[1].imag, cdum[2].imag ]
            else:
                cdumA,cdumB = dmtrace(matrix,3,dm_kpts,pbc,this._spinrestriction,dict=None)
                ret_RandV[1] = [ cdumA[k].imag + cdumB[k].imag for k in range(3) ]

            dipvel_0 =[ ret_RandV[1][0], ret_RandV[1][1], ret_RandV[1][2] ]  ## dipvel_0 : < \hat{v} >

            if( this.velocitygauge and this._td_field is not None ):
                if( Aext_over_c is not None ):
                    A_over_c=np.array( [ Aext_over_c[kk] for kk in range(3) ] )
                else:
                    A_over_c=this._td_field.get_vectorfield(this._time_AU)/cLIGHT_AU

                if( this._spinrestriction == 'R'):
                    for kk in range(3):
                        ret_RandV[1][kk]+=A_over_c[kk]*N_ele    ### electronic charge being -1 ...
                    ### print("#adding Gauge term*N_ele(%f): %f %f %f"%(N_ele, A_over_c[0]*N_ele, A_over_c[1]*N_ele, A_over_c[2]*N_ele))
                else:
                    for kk in range(3):
                        ret_RandV[1][kk]+=A_over_c[kk]*( N_ele[0]+N_ele[1] )    ### electronic charge being -1 ...
            else:
                A_over_c=np.zeros([3])  ## external field

            dipvel_1=[ ret_RandV[1][0], ret_RandV[1][1], ret_RandV[1][2] ]   ## dipvel_1 : < \hat{v} - q_e/c A_ext >
            if( Aind_over_c is not None ):   ## always true if calc_Aind!=0 (see above) 
                for kk in range(3):
                        ret_RandV[1][kk]+=Aind_over_c[kk]*( N_ele if(this._spinrestriction == 'R') else (N_ele[0]+N_ele[1]) )
                        A_over_c[kk]+=Aind_over_c[kk]
            dipvel_2=[ ret_RandV[1][0], ret_RandV[1][1], ret_RandV[1][2] ]   ## dipvel_1 : < \hat{v} - q_e/c (A_ext + A_ind) >

            A_over_c_diff=-1
            N_ele_sum=( N_ele if(this._spinrestriction == 'R') else (N_ele[0]+N_ele[1]) )
                ## recall that we are calculating electronic --velocity-- i.e. [ r,H]/ih
                ## we therefore calculate DMtrace of <xV-Vx> and divide by ih and ADD to velocity 
## 2022.02.08 XXX XXX we always include xVVx -->   if( this._calc_Aind and this.cell.pseudo):
            calc_xVVx=(this._pbc and this.cell.pseudo)
            if( calc_xVVx ):
                Dbgng_xVVx=False  ## TODO 
                Ncall_xVVx=rttddft_common.Countup("xVVx")
                reCalc_xVVx=( Dbgng_xVVx and (Ncall_xVVx==5 or Ncall_xVVx==20 or Ncall_xVVx==200 or\
                              Ncall_xVVx%1000==0 ) )
                reuse_xVVx=True
                if( rttddft_common.Params_get("reuse_xVVx") is not None ):
                    reuse_xVVx = rttddft_common.Params_get("reuse_xVVx"); ## pyscf_common.params["reuse_xVVx"]

                is_fixednucleiapprox=False
                if( hasattr(this,"_fixednucleiapprox") ):
                    if( this._fixednucleiapprox is not None ):
                        is_fixednucleiapprox=this._fixednucleiapprox
                if( not is_fixednucleiapprox ):
                    reuse_xVVx=False;
                    rttddft_common.Write_once("reuse_xVVx","reuse_xVVx set False job:"+str(rttddft_common.get_job(True)),
                                            fpath="reuse_xVVx.log",Append=True)
                else:
                    rttddft_common.Write_once("reuse_xVVx","reuse_xVVx set default, %r job:"%(reuse_xVVx)+str(rttddft_common.get_job(True)),
                                            fpath="reuse_xVVx.log",Append=True)

                istep=int( round(this._time_AU/this._dt_AU) )
                wct_010=time.time()
                xVVx_matr=None;wct_xVVx=0
                if( reuse_xVVx and (this._xVVx is not None) ):
                    A_over_c_diff=max( [  abs(A_over_c[0]-this._Aeff_over_c_xVVx[0]),
                                          abs(A_over_c[1]-this._Aeff_over_c_xVVx[1]),
                                          abs(A_over_c[2]-this._Aeff_over_c_xVVx[2]) ] )
                    if( A_over_c_diff < xVVx_A_over_c_diff_TOL ):
                        xVVx_matr=this._xVVx;
                fd01=open("xVVx.log","a");print("#xVVx:%05d:"%(Ncall_xVVx),reuse_xVVx,reCalc_xVVx,file=fd01);fd01.close()

                if( xVVx_matr is None  or reCalc_xVVx ):
                    xVVx_matr= with_df_get_ppxx(this.with_df,A_over_c,kpts=this.kpts,rttddft=this)
                    printout_matrix("xVVx_matr.dat",xVVx_matr,step=this._step,time_AU=this._time_AU)
                    wct_020=time.time(); wct_xVVx=wct_020-wct_010
                    if( A_over_c_diff > 0 and this._xVVx is not None):
                        xVVx_diff=aNmaxdiff(xVVx_matr,this._xVVx) 
                        printout("#get_dipole:RECALC xVVx:%f %f %f / %f %f %f Adiff=%e tmAU=%f/%f xVVx diff:%e /walltime:%f"%(\
                             A_over_c[0], A_over_c[1], A_over_c[2],
                             this._Aeff_over_c_xVVx[0], this._Aeff_over_c_xVVx[1], this._Aeff_over_c_xVVx[2],
                             A_over_c_diff,tmrefAU_Aind, this._tmrefAU_xVVx, xVVx_diff, wct_xVVx),fpath="xVVx.log",Append=True)
                    this._xVVx = arrayclone( xVVx_matr);this._tmrefAU_xVVx = tmrefAU_Aind;
                    this._Aeff_over_c_xVVx = [ A_over_c[0], A_over_c[1], A_over_c[2] ]
                Dic001={};
                n_dev = check_Hermicity_xKaa('A',xVVx_matr,3,len(this.kpts),Dic=Dic001,title="xVVx_matr")
                absmaxv= max( abs( np.ravel(xVVx_matr) ) )
                fd_xVVxlog =open("xVVx_consistency.log","a");
                print(" step %d, time %f (fs),  absmax:%e  anti Hermicity:%s    job:%s  time:%s"%(\
                        (-1 if(this._step is None) else this._step),\
                        (-1.0 if(this._time_AU is None) else this._time_AU*AUinFS),\
                        absmaxv, str(Dic001),str(rttddft_common.get_job(True)),\
                        str(datetime.datetime.now())),file=fd_xVVxlog)
                ### 2021.08.24 :: fwrite_zNbuf(xVVx_matr,"xVVx_matr_%04d.dat"%(istep),Append=False,description="xVVx",format="%16.8f %16.8f",delimiter=" ",Threads=[0])
                ### print("#rttddft01:check_Hermicity_xKaa returns:%d"%(n_dev),flush=True);
                v_add = -1j*dmtrace_xKaa( xVVx_matr,3,dm_kpts,pbc,this._spinrestriction)

                # v_add := [x, V]/(i\hbar) the real part is to be added to the electronic velocity
                #                          the imaginary part should be zero in principle because of the Hermicity of xVVx/i\hbar .

                dum=np.sqrt( v_add[0].imag**2 + v_add[1].imag**2 + v_add[2].imag**2 )
                print("xVVx:v_add imag:%e"%(dum))
                print("#v_add: %15.5e %15.5e     %15.5e %15.5e    %15.5e %15.5e"%(\
                    v_add[0].real,v_add[0].imag,  v_add[1].real,v_add[1].imag, v_add[2].real,v_add[2].imag),file=fd_xVVxlog)
                fd_xVVxlog.close()
                assert dum<1e-4, "check anti hermicity or dmtrace"

                for kk in range(3):
                    ret_RandV[1][kk]+= v_add[kk].real
                dipvel_3 = [ ret_RandV[1][0], ret_RandV[1][1], ret_RandV[1][2] ]

                dirZ=[0.0, 0.0, 1.0 ]
                if( rttddft_common.Params_get("polarization") is not None ):
                    ##  dirZ= np.array( parse_doubles(pyscf_common.params["polarization"],delimiter=',') );
                    dirZ= np.array( parse_doubles(rttddft_common.Params_get("polarization"),delimiter=',') );
                    if( Ncall_xVVx == 1 ):
                        dum=np.sqrt( np.vdot( dirZ,dirZ ) );assert abs(dum-1)<1e-5,"polarization vector"
                if( Ncall_xVVx == 1 ):
                    printout("#%14s    %16s %16s %16s %16s     %16s %16s %16s %16s     %16s %16s %16s %16s     %16s %16s %16s %16s     %16s %16s %16s %16s        %16s %16s %16s %16s      %s"%(
                             "time_AU", "xVVx_x","xVVx_y","xVVx_z","xVVx_n",  "dipvel3_x","dipvel3_y","dipvel3_z","dipvel3_n",
                             "dipvel0_x","dipvel0_y","dipvel0_z","dipvel0_n", "Aext_x","Aext_y","Aext_z","Aext_n",
                             "Aind_x","Aind_y","Aind_z","Aind_n",     "xVVx_x","xVVx_y","xVVx_z","xVVx_n",  "walltime"),
                             dtme=True,fpath=rttddft_common.get_job(True)+"_dipvel.dat",Append=False );

                printout(" %14.6f    %16.6e %16.6e %16.6e %16.6e     %16.6e %16.6e %16.6e %16.6e     %16.6e %16.6e %16.6e %16.6e     %16.6e %16.6e %16.6e %16.6e     %16.6e %16.6e %16.6e %16.6e        %16.6e %16.6e %16.6e %16.6e      %f"%(
                         this._time_AU,\
                         v_add[0].real, v_add[1].real, v_add[2].real,  np.vdot(dirZ, np.array([ v_add[0].real, v_add[1].real, v_add[2].real ])),\
                         dipvel_3[0], dipvel_3[1], dipvel_3[2],  np.vdot(dirZ, dipvel_3),\
                         dipvel_0[0], dipvel_0[1], dipvel_0[2],  np.vdot(dirZ, dipvel_0),\
                         dipvel_1[0]-dipvel_0[0], dipvel_1[1]-dipvel_0[1], dipvel_1[2]-dipvel_0[2], np.vdot(dirZ, np.array(dipvel_1)-np.array(dipvel_0)),\
                         dipvel_2[0]-dipvel_1[0], dipvel_2[1]-dipvel_1[1], dipvel_2[2]-dipvel_1[2], np.vdot(dirZ, np.array(dipvel_2)-np.array(dipvel_1)),\
                         dipvel_3[0]-dipvel_2[0], dipvel_3[1]-dipvel_2[1], dipvel_3[2]-dipvel_2[2], np.vdot(dirZ, np.array(dipvel_3)-np.array(dipvel_2)),\
                         wct_xVVx),dtme=True,fpath=rttddft_common.get_job(True)+"_dipvel.dat",Append=True );
                dipvel_x0 =[ dipvel_0[0]+v_add[0].real, dipvel_0[1]+v_add[1].real, dipvel_0[2]+v_add[2].real ]
                
                ### print("#rttddft01:dmtrace_xKaa returns",flush=True);
                if( dbgng_calcAind and MPIrank==0 ):
                    tm_fs=this._time_AU*( physicalconstants.aut_in_femtosec );n_rcd=rttddft_common.Countup("calc_Aind_dipvel.dat")
                    fd01a=open("calc_Aind_dipvel.dat",("a" if(n_rcd>1) else "w"));
                    if(n_rcd==1):
                        print("#%14s %14s    %50s   %50s   %50s   %50s   %50s   %50s"%(\
                            "time_au","time_fs","dipvel_0","dipvel-Aext","dipvel-Aeff","dipvel-Aeff_corrected","Aind_over_c","v_add"),file=fd01a)
                    strWRN=""
                    Aind_REF=( Aind_over_c if(Aind_over_c is not None) else [ 0,0,0] );strWRN+=(" Aind_over_c_is_None" if(Aind_over_c is None) else "");
                    vadd_REF=( v_add  if(v_add is not None) else [0.0+1j*0.0, 0.0+1j*0.0, 0.0+1j*0.0]);strWRN+=(" v_add_is_None" if(v_add is None) else "");
                    Aeff_REF=[ A_over_c[0] + Aind_REF[0] + vadd_REF[0], A_over_c[1] + Aind_REF[1] + vadd_REF[1], A_over_c[2] + Aind_REF[2] + vadd_REF[2] ]
                    print(" %14.6f %14.6f    %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.6e %16.6e %16.6e   %16.8f %16.8f %16.8f"%(\
                        this._time_AU, tm_fs,     dipvel_0[0], dipvel_0[1], dipvel_0[2],  dipvel_1[0], dipvel_1[1], dipvel_1[2],\
                        dipvel_2[0], dipvel_2[1], dipvel_2[2], ret_RandV[1][0], ret_RandV[1][1], ret_RandV[1][2],
                        Aind_REF[0], Aind_REF[1], Aind_REF[2],  vadd_REF[0].real, vadd_REF[1].real, vadd_REF[2].real,
                        Aeff_REF[0], Aeff_REF[1], Aeff_REF[2] )+strWRN )
                    print(" %14.6f %14.6f    %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f   %16.6e %16.6e %16.6e   %16.8f %16.8f %16.8f"%(\
                        this._time_AU, tm_fs,     dipvel_0[0], dipvel_0[1], dipvel_0[2],  dipvel_1[0], dipvel_1[1], dipvel_1[2],\
                        dipvel_2[0], dipvel_2[1], dipvel_2[2], ret_RandV[1][0], ret_RandV[1][1], ret_RandV[1][2],
                        Aind_REF[0], Aind_REF[1], Aind_REF[2],  vadd_REF[0].real, vadd_REF[1].real, vadd_REF[2].real,
                        Aeff_REF[0], Aeff_REF[1], Aeff_REF[2] )+strWRN,file=fd01a)
                    fd01a.close()
                    if(n_rcd==1):
                        ### os.system("gnuf.sh calc_Aind_dipvel");
                        fd01a=open("calc_Aind_dipvel.plt","w");
                        print("set term postscript color\nset output \"calc_Aind_dipvel.ps\"\n",file=fd01a);
                        print("plot \"calc_Aind_dipvel.dat\" using 2:5 title \"v_0\" with lines ls 2,\\\n" \
                             +"\"\" using 2:8 title \"v-Aext\" with lines ls 4,\\\n" \
                             +"\"\" using 2:11 title \"v-Aeff\" with lines ls 6,\\\n" \
                             +"\"\" using 2:14 title \"v-Aeff-fix\" with lines ls 8",file=fd01a);
                        print("plot \"calc_Aind_dipvel.dat\" using 2:17 title \"A_{ind}\" with lines ls 102,\\\n" \
                             +"\"\" using 2:20 title \"v-add\" with lines ls 4",file=fd01a);fd01a.close()
                        
                
        else:
            continue

    dict_retv={'dipole':ret_RandV[0], 'dipole_velocity':ret_RandV[1]};strdiff=""
    if( dipvel_0 is not None ):
        dict_retv.update( {'dipvel_0':dipvel_0} )
    if( dipvel_x0 is not None ):
        dict_retv.update( {'dipvel_x0':dipvel_x0} )
    if( dipvel_1 is not None ):
        dict_retv.update( {'dipvel_1':dipvel_1, 'dipvel_2':dipvel_2} )
    if( dipvel_3 is not None ):
        dict_retv.update( {'dipvel_3':dipvel_3} )
    check_dip_ref=(molecular_dipole is not None)
    calc_MOLdip=( (molecular_dipole is not None) and (not this._pbc) )
    if(calc_MOLdip):
        dip_ref=None
        if(this._pbc):
            dip_ref= this.dip_moment( this.cell, dm_kpts, unit='au', verbose=pyscflib_logger.WARN)
        else:
            dip_ref= this.dip_moment( this.mol, dm_kpts, unit='au', verbose=pyscflib_logger.WARN)
                     ## takes summation over spin 
        dict_retv.update({'molecular_dipole':dip_ref})
        dip_dist=d1dist( dip_ref, ret_RandV[0])
        dip_dist1=dip_dist
        charge_center = None
        if(this._pbc):
            charges = this.cell.atom_charges()
            coords  = this.cell.atom_coords()
            charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
        else:
            charges = this.mol.atom_charges()
            coords  = this.mol.atom_coords()
            charge_center = np.einsum('z,zr->r', charges, coords)/charges.sum()

        if( molecular_dipole is not None ):
            molecular_dipole.clear()
            molecular_dipole.append( dip_ref )
        else:
            assert dip_dist1<1.0e-7,"check dipoles.."

        strdiff=""
        diff_moldip_eldip=[ ret_RandV[0][kk]-dip_ref[kk] for kk in range(3) ];Ddiff=[0.0, 0.0, 0.0]
        if( Dipoles_static.Diff_moldip_eldip_ is None):
            Dipoles_static.Diff_moldip_eldip_ = diff_moldip_eldip
            strdiff="   %14.6f  %14.6f  %14.6f       "%(
                diff_moldip_eldip[0],diff_moldip_eldip[1],diff_moldip_eldip[2])
                ### Ddiff[0],Ddiff[1],Ddiff[2])
        else:
            for k in range(3):
                Ddiff[k]= diff_moldip_eldip[k] - Dipoles_static.Diff_moldip_eldip_[k]
            strdiff="   %14.6f  %14.6f  %14.6f       %14.6f %14.6f %14.6f"%(
                diff_moldip_eldip[0],diff_moldip_eldip[1],diff_moldip_eldip[2],
                Ddiff[0],Ddiff[1],Ddiff[2])
    elif( molecular_dipole is not None ):
        molecular_dipole.append( [0.0, 0.0, 0.0])

    if( (filepath is not None) and (not suppress_prtout()) ):
        nmult_dipole=1 ### (1 if(this._spinrestriction == 'R') else 2)
        fd=futils.fopen(filepath,"a");## OK
        if(headline):
            strbuf="#1:%11s 2:%16s 3:%16s 4:%16s    5:%16s 6:%16s 7:%16s"%("tm_AU","x","y","z","Vx","Vy","Vz")
            ncol=7
            if( dipvel_0 is not None ):
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"v^{0}_x", ncol+2,"v^{0}_y", ncol+3,"v^{0}_z");ncol+=3
            if( dipvel_1 is not None ):
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"p-qA^{ext}_x", ncol+2,"p-qA^{ext}_y", ncol+3,"p-qA^{ext}_z");
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+4,"p-qA^{tot}_x", ncol+5,"p-qA^{tot}_y", ncol+6,"p-qA^{tot}_z");ncol+=6
            if( current is not None):
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"Jx", ncol+2,"Jy", ncol+3,"Jz"); ncol+=3

            if( this._calc_Aind < 0 ):
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"C-dAdt/c.x",ncol+2,"C-dAdt/c.y", ncol+3,"C-dAdt/c.z");ncol+=3
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"Aind/c.x",ncol+2,"Aind/c.y", ncol+3,"Aind/c.z");ncol+=3
            elif( this._calc_Aind > 0 ):
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"P-dAdt/c.x",ncol+2,"P-dAdt/c.y", ncol+3,"P-dAdt/c.z");ncol+=3
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"Aind/c.x",ncol+2,"Aind/c.y", ncol+3,"Aind/c.z");ncol+=3

            if( molecular_dipole is not None):
                strbuf+="      %2d:%15s %2d:%15s %2d:%15s"%( ncol+1,"Moldip.x", ncol+2,"Moldip.y", ncol+3,"Moldip.z"); ncol+=3

            print(strbuf,file=fd)

        print(header,end="",file=fd)
        if(mode == 'B' or mode=='R'):
            for m in range(nmult_dipole):
                ref=( ret_RandV[0] if(nmult_dipole==1) else ret_RandV[0][m] )
                print("  %18.10f %18.10f %18.10f  "%( ref[0], ref[1], ref[2]), end="", file=fd)
        if(mode == 'B' or mode=='V'):
            for m in range(nmult_dipole):
                ref=( ret_RandV[1] if(nmult_dipole==1) else ret_RandV[1][m] )
                print("  %18.10f %18.10f %18.10f  "%( ref[0], ref[1], ref[2]), end="", file=fd)
        if( dipvel_0 is not None ):
            print("  %18.10f %18.10f %18.10f  "%( dipvel_0[0], dipvel_0[1], dipvel_0[2]), end="", file=fd)
        if( dipvel_1 is not None ):
            print("  %18.10f %18.10f %18.10f  "%( dipvel_1[0], dipvel_1[1], dipvel_1[2]), end="", file=fd)
            print("  %18.10f %18.10f %18.10f  "%( dipvel_2[0], dipvel_2[1], dipvel_2[2]), end="", file=fd)

        if(current is not None):
            print("  %18.10f %18.10f %18.10f  "%( current[0], current[1], current[2]), end="", file=fd)

        if( this._calc_Aind <0 ):
            if( this._Aind_Gearvc_C is None ):
                print("  %18.10f %18.10f %18.10f    "%( 0.0, 0.0, 0.0),end="",file=fd)
                print("  %18.10f %18.10f %18.10f    "%( 0.0, 0.0, 0.0),end="",file=fd)
            else:
                print("  %18.10f %18.10f %18.10f    "%( this._Aind_Gearvc_C[1][0]/this._dt_AU, this._Aind_Gearvc_C[1][1]/this._dt_AU, this._Aind_Gearvc_C[1][2]/this._dt_AU),end="",file=fd)
                print("  %18.10f %18.10f %18.10f    "%( this._Aind_Gearvc_C[0][0], this._Aind_Gearvc_C[0][1], this._Aind_Gearvc_C[0][2]/this._dt_AU),end="",file=fd)
        elif( this._calc_Aind >0 ):
            if( this._Aind_Gearvc_P is None ):
                print("  %18.10f %18.10f %18.10f    "%( 0.0, 0.0, 0.0),end="",file=fd)
                print("  %18.10f %18.10f %18.10f    "%( 0.0, 0.0, 0.0),end="",file=fd)
            else:
                print("  %18.10f %18.10f %18.10f    "%( this._Aind_Gearvc_P[1][0]/this._dt_AU, this._Aind_Gearvc_P[1][1]/this._dt_AU, this._Aind_Gearvc_P[1][2]/this._dt_AU),end="",file=fd)
                print("  %18.10f %18.10f %18.10f    "%( this._Aind_Gearvc_P[0][0], this._Aind_Gearvc_P[0][1], this._Aind_Gearvc_P[0][2]/this._dt_AU),end="",file=fd)

        if(molecular_dipole is not None):
            ndim1=np.shape( molecular_dipole[0] );rank1=len(ndim1);
            ## [3] or [2][3]
            nmult_moldip=1 ### (1 if(this._spinrestriction == 'R' or rank1==1) else 2)
            for m in range(nmult_moldip):
                ref=( molecular_dipole[0] if(nmult_moldip==1) else molecular_dipole[0][m] )
            print("  %18.10f %18.10f %18.10f    %s"%( molecular_dipole[0][0],
                molecular_dipole[0][1],molecular_dipole[0][2],strdiff),end="",file=fd) 
        print(STRwalltime,end="",file=fd)
        print(trailer,file=fd)
        futils.fclose(fd)
    return dict_retv;

def update_Aind_over_c(this):
    n_call=rttddft_common.Countup("update_Aind_over_c")
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    assert this._Aind_tmNxt is not None,""
    assert this._Aind_Gearvc_C_nxt is not None,""

    if( this._Aind_over_c_Nintpl is not None ):
        assert this._veff_latest is not None,"this._veff_latest is None tm:%f"%(this._Aind_tmNxt)
        this._veff_LAST=np.array( [ this._veff_latest[kk] for kk in range(3) ] )
        this._veff_tmAU=this._tmAU_veff_latest
        this._veff_latest=None; this._tmAU_veff_latest=None

    AUinFS=0.02418884326058678
    this._Aind_tmAU=this._Aind_tmNxt
    for ii in range(5):
        for jj in range(3):
            this._Aind_Gearvc_C[ii][jj]=this._Aind_Gearvc_C_nxt[ii][jj]
            this._Aind_Gearvc_P[ii][jj]=this._Aind_Gearvc_P_nxt[ii][jj]
    this._Aind_Gearvc_C_nxt=None; this._Aind_Gearvc_P_nxt=None; this._Aind_tmNxt=None

def rttddft_set_Aind_args(this,Dic,errorlogfnme=None,verbose=False):
    nerror=rttddft_sget_Aind_args_('S',this,Dic,errorlogfnme=errorlogfnme,verbose=verbose);
    assert nerror==0,""

def rttddft_get_Aind_args(this,Dic,errorlogfnme=None,verbose=False):
    nerror=rttddft_sget_Aind_args_('G',this,Dic,errorlogfnme=errorlogfnme,verbose=verbose);
    assert nerror==0,""

def check_nao_nr(mol, cart=None):
    ANG_OF     = 1
    NCTR_OF    = 3
    '''Total number of contracted GTOs for the given :class:`Mole` object'''
    if cart is None:
        cart = mol.cart
    if cart:
        ### print("#nao_nr:cart...",flush=True)
        return nao_cart(mol)
    else:
        ### print("#nao_nr:_ANG:",end="");print(mol._bas[:,ANG_OF])
        ### print("#nao_nr:NCTR:",end="");print(mol._bas[:,NCTR_OF])
        return ((mol._bas[:,ANG_OF]*2+1) * mol._bas[:,NCTR_OF]).sum()

def update_timing(this,label,cputime,Nskip_prtout=20,logfpath=None,logfappend=True,logtext=""):
    if(this.timing is None):
        this.timing={}
    if( not (label in this.timing)):
        dic1={"count":0,"time":0.0}
        this.timing.update({label:dic1})
    dic1=this.timing[label]
    dic1["count"]+=1;dic1["time"]+=(cputime)
    prtou=(dic1["count"]==1)
    if(Nskip_prtout>0):
        if( dic1["count"]%Nskip_prtout==0):
            prtou=True
    if(prtou and (not suppress_prtout()) ):
        fdA=[ sys.stdout ];fdOUTF=None
        if( logfpath is not None ):
            fdOUTF =futils.fopen(logfpath,("a" if(logfappend) else "w"));
            fdA.append(fdOUTF)
        for fdx in fdA:
            print("#"+label+":%f %s: %s sum:%f(%d) avg:%f"%(
                cputime, str(label),str(logtext), dic1["time"], dic1["count"], dic1["time"]/dic1["count"]),file=fdx)
        if(fdOUTF is not None):
            futils.fclose(fdOUTF)

def tocomplex(src):
    ndim=np.shape(src)
    ### print("#toComplex:",end="");print(ndim)
    rank=len(ndim)
    if( rank<5):
        ret=np.zeros(ndim,dtype=np.complex128)
        for i in range(ndim[0]):
            if(rank<2):
                ret[i]=src[i];continue
            for j in range(ndim[1]):
                if(rank<3):
                    ret[i][j]=src[i][j];continue
                for k in range(ndim[2]):
                    if(rank<4):
                        ret[i][j][k]=src[i][j][k];continue
                    for l in range(ndim[3]):
                        ret[i][j][k][l]=src[i][j][k][l];
        return ret;
    else:
        Ld=1;
        for j in range(rank):
            Ld=Ld*ndim[j]
        s1d=np.ravel(src)
        r1d=np.zeros([Ld],dtype=np.complex128)
        for k in range(Ld):
            r1d[k]=s1d[k]
        return np.reshape(r1d,ndim)

def calc_FieldEng_cell(this,tmAU=None,Dict=None):
    if(tmAU is None ):
        tmAU=this._time_AU

    if( this.cell_volume is None ):
        assert this.BravaisVectors_au is not None,""
        this.cell_volume=calc_3Dvolume( this.BravaisVectors_au )
    PIx8=25.132741228718345907701147066236
    fac=this.cell_volume/PIx8

    E_tot=[ 0.0, 0.0, 0.0 ]
    E_ext=[ 0.0, 0.0, 0.0 ]
    if( this._td_field is not None ):
        if( not isinstance( this._td_field, kickfield ) ):
            E_ext=this._td_field.get_electricfield( tmAU )
    if( this._calc_Aind == 0 ):
        Esqr= E_ext[0]**2 + E_ext[1]**2 + E_ext[2]**2
        return 0.0, Esqr*fac
    if( Dict is not None ):
        Dict.update({"E_ext":[ E_ext[0],E_ext[1],E_ext[2] ] })
### 2021.08.25        assert False,""
### 2021.08.25       return 0
    if( this._Aind_Gearvc_C is None ):
        assert False,"_Aind_Gearvc_C"
        return 0
    Implicit=(this._calc_Aind < 0 )
    Adot_over_c=( this._Aind_Gearvc_C[1]/this._dt_AU if(Implicit) else this._Aind_Gearvc_P[1]/this._dt_AU )
    ret0= ( Adot_over_c[0]**2 + Adot_over_c[1]**2 + Adot_over_c[2]**2 )*fac
    ### printout("#cell_volume:",this.cell_volume,fac,this._Aind_Gearvc_C[1])
    ### printout("#Adot_over_c:%e %e %e"%(Adot_over_c[0],Adot_over_c[1],Adot_over_c[2]))
    E_tot=[ E_ext[k] - Adot_over_c[k] for k in range(3) ]
    Esqr= E_tot[0]**2 + E_tot[1]**2 + E_tot[2]**2
    ret1=Esqr*fac
    if( Dict is not None ):
        Dict.update({"E_ind":[ -Adot_over_c[0],-Adot_over_c[1],-Adot_over_c[2] ] })
        Dict.update({"E_tot":[ E_tot[0],E_tot[1],E_tot[2] ]})
    return ret0,ret1

def krhf_get_hcore_pp(this, cell=None, kpts=None,tm_AU=None,Dict_DBG=None, A_over_c=None):
    import time
    from .utils import aNmaxdiff,print_z2array
    from .update_dict import printout_dict,update_dict
    
    ''' Copy of khf.py#get_hcore
    Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    assert (this._pbc),"never come here otherwise"
    cLIGHT_AU=PhysicalConstants.CLIGHTinAU()
    if tm_AU is None: tm_AU=this._time_AU
    if cell is None: cell = this.cell
    if kpts is None: kpts = this.kpts
    
    n_call=rttddft_common.Countup("krhf_get_hcore_pp")
    fncnme='krhf_get_hcore_pp';fncdepth=2   ## fncdepth ~ SCF:0 vhf:1 get_j:2
    Wctm000=time.time();Wctm010=Wctm000;dic1={}
    Dic=rttddft_common.Dict_getv('timing_'+fncnme, default=None)
    if( Dic is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});
        Dic=rttddft_common.Dict_getv('timing_'+fncnme, default=None)

    dbgng=False
    ## see UT/with_fftdf_get_pp
    ## dbgng_pp=True;pp_dbgfnme=str(rttddft_common.get_job(True))+"_pp.dat"
    
    if( this._td_field is not None ):
        A_div_c = (A_over_c if(A_over_c is not None) else this._td_field.get_vectorfield(tm_AU)/cLIGHT_AU )
        Logger.write_once(logger=None,title="get_hcore_AoverC",content="AoverC:%f,%f,%f"%(A_div_c[0],A_div_c[1],A_div_c[2]),fnme="get_hcore.log")
        if cell.pseudo:
            details=None
            if( ( n_call == 1 or n_call == 5 or n_call == 20 or n_call == 50 or n_call == 200 ) and 
                isinstance( this.with_df, df.GDF ) ):
                details={}
            if( Dict_DBG is not None ):
                Dict_DBG.update({'get_pp':'new'})
            if( isinstance( this.with_df, df.FFTDF ) ):
                nuc=lib.asarray( with_fftdf_get_pp(this.with_df, A_div_c, kpts=kpts) )
                #if( dbgng_pp ):
                #    svld_aNbuf("S",pp_dbgfnme,buf=nuc,
                #      comment="A_over_c:"+str(A_over_c)+"Rnuc:"+tostring(this.get_Rnuc(unit='ANGS')))
            else:
                nuc=lib.asarray( with_df_get_pp( this.with_df,A_div_c,kpts,rttddft=this, details=details ) ) ### this.with_df.get_pp01(kpts,A_div_c))
                #if( dbgng_pp ):
                #    svld_aNbuf("S",pp_dbgfnme,buf=nuc,
                #      comment="A_over_c:"+str(A_over_c)+"Rnuc:"+tostring(this.get_Rnuc(unit='ANGS')))
            Wctm020=Wctm010;Wctm010=time.time()
            update_dict(fncnme,dic1,Dic,"get_pp_A",Wctm010-Wctm020,depth=fncdepth)  
            Wctm_par=Wctm010-Wctm020

            ## n_call = rttddft_common.Countup("krhf_get_hcore_pp")
            if( ( n_call == 1 or n_call == 5 or n_call == 20 or n_call == 50 or n_call == 200 ) and 
                isinstance( this.with_df, df.GDF ) ):
                ##
                ## 20210609MPIvalidation: When you remove this validation, you might want to comment-in dbgng_mpi in gto_ps_pp_int01.py line 955 .. 
                ##
                details_bf20210608={}
                nuc_refr=lib.asarray( with_df_get_pp_bf20210608( this.with_df,A_div_c,kpts,details=details_bf20210608 ) )
                
                Wctm020=Wctm010;Wctm010=time.time()
                update_dict(fncnme,dic1,Dic,"get_pp_singlethread",Wctm010-Wctm020,depth=fncdepth)
                diff=aNmaxdiff( nuc, nuc_refr )
                diffs={};
                for ky in details:
                    diffs.update({ky:aNmaxdiff( details[ky],details_bf20210608[ky])}) 
                printout("with_df_get_pp %d:%e %s walltime par:%f / ser:%f"%(n_call,diff, str(diffs),Wctm_par, Wctm010-Wctm020),
                        fpath="ppnl_half.log",Append=True,Threads=[0])

                if(diff>=1.0e-7):
                    zerovec=np.zeros([3],dtype=np.float64)
                    ppZF_new=with_df_get_pp( this.with_df,zerovec,kpts,rttddft=this )
                    ppZF_old=with_df_get_pp_bf20210608( this.with_df,zerovec,kpts)
                    ppZF_refr=this.with_df.get_pp(kpts)
                    dev_r_n=aNmaxdiff( ppZF_refr, ppZF_new)
                    dev_r_o=aNmaxdiff( ppZF_refr, ppZF_old)
                    dev_n_o=aNmaxdiff( ppZF_new,  ppZF_old)
                    printout("#ppZF:%e %e %e"%(dev_r_n,dev_r_o,dev_n_o),fpath="ppZF_comp.log",Append=False,dtme=True)
                assert diff<1.0e-7,"new/old deviates"
            ### print("#with_df_get_pp returns",flush=True)
        else:
            assert False,""
    else:
        if cell.pseudo:
            if( Dict_DBG is not None ):
                Dict_DBG.update({'get_pp':'old'})
            nuc = lib.asarray(this.with_df.get_pp(kpts))

            #if( dbgng_pp ):
            #    svld_aNbuf("S",pp_dbgfnme,
            #               buf=nuc,comment=tostring(this.get_Rnuc(unit='ANGS')))
            Wctm020=Wctm010;Wctm010=time.time()
            update_dict(fncnme,dic1,Dic,"get_pp",Wctm010-Wctm020,depth=fncdepth)  
        else:
            nuc = lib.asarray(this.with_df.get_nuc(kpts))
            Wctm020=Wctm010;Wctm010=time.time()
            update_dict(fncnme,dic1,Dic,"get_nuc",Wctm010-Wctm020,depth=fncdepth)  
        if len(cell._ecpbas) > 0:
            nuc += lib.asarray(ecp.ecp_int(cell, kpts))
    #if(dbgng_pp):
    #    os.system("fopen "+pp_dbgfnme)
    #    assert False,"dbgng_pp"
    t=None
    if( hasattr(this,"_fixednucleiapprox") ):
        if( this._fixednucleiapprox is not None ):
            if( this._fixednucleiapprox ):
                if( this.int1e_kin_ is None ):
                    this.int1e_kin_=lib.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
                    Wctm020=Wctm010;Wctm010=time.time() ## XXX XXX
                    update_dict(fncnme,dic1,Dic,"int1e_kin",Wctm010-Wctm020,depth=fncdepth)  ## XXX XXX
                t=arrayclone( this.int1e_kin_)
                if( n_call == 2 or n_call == 10 or n_call == 50 ):
                    t_ref=lib.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts)) 
                    Wctm020=Wctm010;Wctm010=time.time() ## XXX XXX
                    update_dict(fncnme,dic1,Dic,"int1e_kin_DBG",Wctm010-Wctm020,depth=fncdepth)  ## XXX XXX
                    diff=aNmaxdiff(t,t_ref)
                    printout("int1e_kin:diff=%e time=%f"%(diff,Wctm010-Wctm020),
                            fnme_format="Save_vloc_%02d.log",Append=True)
                    assert (diff<1.0e-10),"t deviates"
    t = lib.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1,Dic,"int1e_kin",Wctm010-Wctm020,depth=fncdepth)  
    printout_dict(fncnme,dic1,Dic,Wctm010-Wctm000,depth=fncdepth)
    return nuc + t





def get_Aind_over_c(this,tm_AU,errlv=-1):
    ### test_Aind_over_c(this) ## XXX XXX
    tmAU_TINY=1.0e-8
    if( this._calc_Aind == 0 ):
        return None
    Implicit_scheme=( this._calc_Aind < 0 )
    if( this._Aind_Gearvc_C is None ):
        if( abs(tm_AU)<tmAU_TINY ):
            return np.array( [ this._Aind_over_c_ini[0], this._Aind_over_c_ini[1], this._Aind_over_c_ini[2] ] )
        if(errlv<0):
            assert False,"#get_Aind_over_c:Aind_Gearvc is not ready:consider calc_Aind_over_c.."
        return None
    if( abs( tm_AU - this._Aind_tmAU ) < tmAU_TINY ):
        return ( np.array( [ this._Aind_Gearvc_C[0][0], this._Aind_Gearvc_C[0][1], this._Aind_Gearvc_C[0][2] ] ) if(Implicit_scheme) else \
                 np.array( [ this._Aind_Gearvc_P[0][0], this._Aind_Gearvc_P[0][1], this._Aind_Gearvc_P[0][2] ] ) )
    else:
        if( errlv < 0 ):
            assert False,"#get_Aind_over_c: this._Aind_tmAU=%f tm_AU=%f diff=%e"%(this._Aind_tmAU,tm_AU,abs(this._Aind_tmAU-tm_AU))
        return None


def calc_Aind_over_c(this,tm_AU,dt_AU, dm_kOR1, tm_dmat=None, Aind_over_c_ini=None,caller=""):
    tmAU_TINY=1.0e-8
    AUinFS=0.02418884326058678
    Ld=3
    Implicit_Scheme=( this._calc_Aind < 0 )
    if( rttddft_common.Params_get("Aind_over_c_Nintpl") is not None ):
        return propagate_Aind_over_c_interpol(this,tm_AU, dt_AU, dm_kOR1, tm_dmat=tm_dmat, Aind_over_c_ini=Aind_over_c_ini,caller="calc_Aind_over_c<-"+caller)

def calc_d2Aind_over_c_dt2_(this, flag, dm_kOR1, Aind_over_c, tm_AU):
    dbgng_Aind_over_c=True ## XXX XXX 2021.08.03 
    PIx4=12.566370614359172953850573533118
    if( this.cell_volume is None ):
        assert this.BravaisVectors_au is not None,""
        this.cell_volume=calc_3Dvolume( this.BravaisVectors_au )
    dic=get_dipole(this, dm_kOR1, mode='V', Aind_over_c=Aind_over_c,caller="calc_d2Aind_over_c", tmrefAU_Aind=tm_AU)
    velo=None
    if( flag == 1):
        velo=dic['dipvel_0']  ## bare
    elif( flag == 2):
        velo=dic['dipvel_1']  ## v- q_e A_ext
    elif( flag == 3):
        velo=dic['dipvel_2']  ## v- q_e (A_ext+A_ind) 
    elif( flag == 4):
        assert this.cell.pseudo,""; velo=dic['dipvel_3']
    else:
        assert False,"%d,%d"%(flag,this._calc_Aind)

    ## 20210803 : Ulrich suggests --2-- with \alpha = 0.2
    ##          : Yabana 1 or 2 ??      with \alpha = 1.0   

    assert velo is not None,""
    fac= PIx4/this.cell_volume
    if( this._alpha_Aind_over_c is not None ):
        fac=fac*this._alpha_Aind_over_c

    retv=np.array([ -velo[0]*fac, -velo[1]*fac, -velo[2]*fac ]) ## electric charge being -1...
    if( dbgng_Aind_over_c ):
        comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
        if( MPIrank == 0 ):
            string=" %14.6f   %12.6f   %14.6f %14.6f %14.6f   %14.6f %14.6f %14.6f    %14.6f %14.6f %14.6f     %14.6f %14.6f %14.6f     %14.6f %14.6f %14.6f"%(\
                   tm_AU, this._alpha_Aind_over_c,  retv[0],retv[1],retv[2],  velo[0],velo[1],velo[2], 
                   dic['dipvel_0'][0], dic['dipvel_0'][1], dic['dipvel_0'][2], dic['dipvel_2'][0],dic['dipvel_2'][1],dic['dipvel_2'][2],\
                   dic['dipvel_3'][0], dic['dipvel_3'][1], dic['dipvel_3'][2])
            if( this._step is None or this._step==0 ):
                legend="#%14s   %12s   %44s   %44s    %44s     %44s     %44s"%(\
                      "tm_AU","alpha","d2Aind_over_c/dt2","v^{eff}","v^{0}","v^{0}+Atot","v^{0}+Atot+xVVx")\
                       + " vol:%14.4e fac:%14.4e"%(this.cell_volume,fac)
                printout(legend, fpath=rttddft_common.get_job(True)+"_Aind_over_c_integ.log",Append=True,dtme=True)
            printout(string, fpath=rttddft_common.get_job(True)+"_Aind_over_c_integ.log",Append=True)

    return retv



class rttddftPBC(KRKS):
    Keys_Aind_over_c_= ["_Aind_over_c_ini:D3",     "_calc_Aind:i","_dt_AU:d",     "BravaisVectors_au:D3,3","cell_volume:d",
          "_alpha_Aind_over_c:d",
          "_d2Aind_over_c_dt2:D3","_d2Aind_over_c_dt2_tmAU:d",    "_Aind_tmAU:d","_Aind_Gearvc_C:D5,3","_Aind_Gearvc_P:D5,3",
          "_Aind_over_c_Nintpl:i","_veff_LAST:D3", "_veff_tmAU:d", "_veff_latest:D3", "_tmAU_veff_latest:d"];
    Keys_Aind_over_c_list_not_ndarray_ =[ ] ## put those args you want to keep it as a list instance ... otherwise we cast all -D- type fields to np.ndarray
    Keys_Aind_over_c_allow_None_=[ "_veff_latest", "_tmAU_veff_latest",]

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',td_field=None, dipolegauge=False, 
                 calc_Aind=0, Aind_over_c_ini=None, alpha_Aind_over_c=None):
        self._pbc=True
        self._spinrestriction='R'
        self._td_field=td_field
        self._time_AU=0.0
        self._step=None
        Logger.write_once( None, "rttddft_xc:"+xc, 
                content="rttddftPBC:"+xc+" \t\t "+str(datetime.datetime.now()),
                fnme="rttddft_xc.log", append=True )
        self._fix_occ=False
        self._mo_occ=None
        self._Egnd = None

        self._logger=None
        if( dipolegauge ):
            assert (td_field is not None),"field is missing"
            self.dipolegauge=(td_field is not None); 
            self.velocitygauge=False
        else:
            self.velocitygauge=(td_field is not None); 
            self.dipolegauge=False;
        self.timing=None
        self._sINV=None
        self._Sinvrt=None
        self._Ssqrt=None
        self.nkpt=None; self.nAO=None; self.nMO=None;

        self._constantfield=False; ## default.. 
        self._hcore=None    
        ## Aind_over_c   since 2021.07.16 ----------------------------------
        ##               2021.07.29 : see below rttddft_set_Aind_args / rttddft_get_Aind_args have to set/get all fields
        self._Aind_over_c_ini=(None if(calc_Aind==0) else \
                               ( np.array([ Aind_over_c_ini[0],Aind_over_c_ini[1],Aind_over_c_ini[2]]) if(Aind_over_c_ini is not None) else \
                                np.zeros([3],dtype=np.float64) ) )
        self._alpha_Aind_over_c=( None if(calc_Aind==0) else \
                                  alpha_Aind_over_c if(alpha_Aind_over_c is not None) else 1.0);
        self._calc_Aind= calc_Aind;   self._dt_AU=None        ## PBC, MOL 
        self.BravaisVectors_au = None;self.cell_volume=None   ## PBC only   values are set automatically in calc_d2Aind_over_c_dt2_
        self._d2Aind_over_c_dt2 = None; self._d2Aind_over_c_dt2_tmAU=None  ## FOR DEBUGGING ---
        ## 2021.08.21
        self._Aind_over_c_Nintpl=None; self._veff_LAST=None; self._veff_tmAU=None
        self._veff_latest=None; self._tmAU_veff_latest=None

        if( calc_Aind != 0 ):
            printout("#rttddft01:_calc_Aind:%d _alpha_Aind_over_c:%f"%(self._calc_Aind, self._alpha_Aind_over_c),
                      fpath=rttddft_common.get_job(True)+"_Aind_over_c_setting.log",Append=True,dtme=True)
        ## _Aind_Gearvc_C/P[5][3]  ... 
        ##  Explicit : uses \ddot{A} at time time_AU-dt_AU to get PREDICTOR vector at time_AU ...
        ##  Implicit : uses \ddot{A} at time time_AU       to get CORRECTED vector at time_AU
        self._Aind_tmAU=None;         self._Aind_Gearvc_C=None;     self._Aind_Gearvc_P=None    ##  see calc_Aind_over_c ... do not fill/allocate these fields
        self._Aind_Gearvc_C_nxt=None; self._Aind_Gearvc_P_nxt=None; self._Aind_tmNxt=None       ##  see calc_Aind_over_c ... do not fill/allocate these fields

        self._xVVx=None; self._Aeff_over_c_xVVx=None; self._tmrefAU_xVVx=None

        self._fixednucleiapprox=None  ## 20210610:fixednucleiapprox
        self.vloc1_=None              ## 20210610:fixednucleiapprox
        self.vloc2_=None              ## 20210610:fixednucleiapprox
        self.int1e_kin_=None          ## 20210610b:fixednucleiapprox
        self._wt000=time.time()
        self._iflag_prtout_field=0    ## 20210828:DBG_ZnOeldyn
        KRKS.__init__(self,cell,kpts,xc)
        
    def set_constantfield(self, constantfield):
        self._constantfield=constantfield
        rttddft_common.Printout_warning("constantfield","rttddft set constantfield:%r"%(self._constantfield),
                                   fpath="constantfield.log")
    def set_calc_Aind(self,calc_Aind, alpha_Aind_over_c, Aind_over_c_ini=None):
        self._Aind_over_c_ini=(None if(calc_Aind==0) else \
                               ( np.array([ Aind_over_c_ini[0],Aind_over_c_ini[1],Aind_over_c_ini[2]]) if(Aind_over_c_ini is not None) else \
                                np.zeros([3],dtype=np.float64) ) )
        if( calc_Aind != 0 ):
            assert alpha_Aind_over_c is not None,"PLS set it explicitly.."
        self._alpha_Aind_over_c=alpha_Aind_over_c
        self._calc_Aind= calc_Aind;
        if( calc_Aind != 0 ):
            printout("#rttddft01:_calc_Aind:%d _alpha_Aind_over_c:%f"%(self._calc_Aind, self._alpha_Aind_over_c),
                      fpath=rttddft_common.get_job(True)+"_Aind_over_c_setting.log",Append=True,dtme=True)

    get_HOMO = get_HOMO
    get_SOMO = get_SOMO
    get_LUMO = get_LUMO

    update_nOrbs = update_nOrbs        
    set_nOrbs = set_nOrbs

    calc_gs = calc_gs

    set_mo_occ = set_mo_occ

    update_Sinv = update_Sinv

    energy_tot = energy_tot

    energy_elec = energy_elec_krks

    get_Rnuc = get_Rnuc

    get_logger = get_logger
    set_logger = set_logger

    get_populations = get_populations

    calc_phases = calc_phases

    print_eorbocc = print_eorbocc

    def get_super(self):
        return super()


    #> Matrix Calculators ---------------------------------------------------------
    get_hcore = get_hcore

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, kpts=None, kpts_band=None):
        cput0=time.time();
        retv= krks_get_veff(self,cell=cell,dm=dm,dm_last=dm_last,vhf_last=vhf_last,hermi=hermi,kpts=kpts,kpts_band=kpts_band)
        cput1=time.time();
        if(self.timing is None):
            self.timing={}
        if( not ("get_veff" in self.timing)):
            Dic={"count":0,"time":0.0}
            self.timing.update({"get_veff":Dic})
        self.timing["get_veff"]["count"]+=1;self.timing["get_veff"]["time"]+=(cput1-cput0)
        dic=self.timing["get_veff"]
        if( (dic["count"]==1 or dic["count"]%20==0) and (not suppress_prtout()) ):
            fdlog=futils.fopen("get_veff.log","a");Ndim=np.shape(retv); ## OK 
            for fd in [fdlog,sys.stdout]:
                print("#get_veff:%f step %s sum:%f(%d) avg:%f   "%( 
                    cput1-cput0, str(Ndim), dic["time"], dic["count"], dic["time"]/dic["count"])
                    +" \t\t" + rttddft_common.get_job(True)+ " \t\t" + str(datetime.datetime.now()), file=fd )
            futils.fclose(fdlog)
        return retv

    def energy_nuc(self,Rnuc_au=None):
        if(Rnuc_au is not None):
            ref_1d=np.ravel( self.mol.atom_coords() )
            inp_1d=np.ravel( Rnuc_au )
            dev=max( abs(ref_1d-inp_1d) )
            assert dev<1e-6,"dev:%e:"%(dev)+str(inp_1d)+"/"+str(ref_1d)
            
        retv=super().energy_nuc()
        return retv

    def get_occ(self, mo_energy_kpts=None, mo_coeff_kpts=None):
        if( self._fix_occ ):
            assert (self._mo_occ is not None),""
            return self._mo_occ
        else:
            ret = khf_get_occ(self,mo_energy_kpts,mo_coeff_kpts)
            return ret;


class rttddftMOL(RKS):
    seqno_=0
    def __init__(self, mol, xc='LDA,VWN',td_field=None,dipolegauge=False,logger=None):
        self._pbc=False
        self._spinrestriction='R'
        rttddftMOL.seqno_+=1
        Logger.write_once( None, "rttddft_xc:"+xc, 
                content="rttddftMOL:"+xc+" \t\t "+str(datetime.datetime.now()),
                fnme="rttddft_xc.log", append=True )
        self._td_field=td_field
        self._time_AU=0.0
        self._fix_occ=False
        self._logger=logger
        self._step=None
        self._mo_occ=None
        self._iflag_prtout_field=0
        if( dipolegauge ):
            assert (td_field is not None),"field is missing"
            self.dipolegauge=(td_field is not None); 
            self.velocitygauge=False
        else:
            self.velocitygauge=(td_field is not None); 
            self.dipolegauge=False;

        self._calc_Aind=0
        self._constantfield=False;
        self._wt000=time.time()
        self._sINV=None
        self._Sinvrt=None
        self._Ssqrt=None
        self.nkpt=None; self.nAO=None; self.nMO=None;
        self._fixednucleiapprox=None  ## 20210610:fixednucleiapprox
        RKS.__init__(self,mol,xc)

    get_HOMO = get_HOMO
    get_SOMO = get_SOMO
    get_LUMO = get_LUMO

    update_nOrbs = update_nOrbs        
    set_nOrbs = set_nOrbs

    calc_gs = calc_gs

    set_mo_occ = set_mo_occ

    update_Sinv = update_Sinv

    energy_tot = energy_tot

    energy_elec = energy_elec_rks

    get_Rnuc = get_Rnuc

    get_logger = get_logger
    set_logger = set_logger

    get_populations = get_populations

    calc_phases = calc_phases

    print_eorbocc = print_eorbocc

    def get_super(self):
        return super()

    #> Matrix Calculators ---------------------------------------------------------
    get_hcore = get_hcore

    def energy_nuc(self,Rnuc_au=None):
        if(Rnuc_au is not None):
            ref_1d=np.ravel( self.mol.atom_coords() )
            inp_1d=np.ravel( Rnuc_au )
            dev=max( abs(ref_1d-inp_1d) )
            assert dev<1e-6,"dev:%e:"%(dev)+str(inp_1d)+"/"+str(ref_1d)
            
        retv=super().energy_nuc()
        return retv

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if( self._fix_occ ):
            assert (self._mo_occ is not None),""
            return self._mo_occ
        else:
            return super().get_occ(mo_energy,mo_coeff)

def construct_Aind_args(strIN):
    ### print("## construct_Aind_args:"+strIN,flush=True)
    Dic=parse_dict(strIN)
    ### print("## construct_Aind_args:Dic:"+str(Dic),flush=True)
    Ret={}
    for item in rttddftPBC.Keys_Aind_over_c_:
        sA=item.split(':');nA=len(sA);assert nA==2,"wrong format:"+str(item)
        ky=sA[0].strip(); ty=sA[1]; 
        if( ky not in Dic ):
            errorstring="#construct_Aind_args:ky:%s missing in Dic"%(ky);
            printout(errorstring,fpath=rttddft_common.get_job(True)+"_Aind_over_c_setting.log",Append=True,dtme=True,warning=1);continue
        val=Dic[ky]
        if( val is not None and val=='None' ):
            val=None
        printout("#construct_Aind_args:"+ky+":"+ty+":"+str(val),fpath=rttddft_common.get_job(True)+"_Aind_over_c_setting.log",Append=True,dtme=True)
        print("#construct_Aind_args:%s "%(ky),val,type(val));
        if( ky in rttddftPBC.Keys_Aind_over_c_allow_None_):
            if( val is None ):
                print("#construct_Aind_args:%s:skipping None... "%(ky),val);
                Ret.update({ky:val}); continue  ## never forget to set None 
        elif( val is None ):
            print("%s is not in allow_None"%(ky),val)
        if(ty[0]=='D'):
            if( ky in rttddftPBC.Keys_Aind_over_c_list_not_ndarray_ ):
                assert isinstance(val,list),""  ### or convert ndarray to list 
            else:
                if( not isinstance(val,np.ndarray) ):
                    val=list_to_array(val)
                assert isinstance(val,np.ndarray),""
        else:
            if(ty[0]=='d'):
                if( val is not None ):
                    val=float(val);
            elif(ty[0]=='i'):
                if( val is not None ):
                    val=int(val)
            else:
                assert False,"key:"+ky+"  type:"+ty
        Ret.update({ky:val})
    return Ret;

# if( pyscf_common.params["Aind_over_c_Nintpl"] is not None ):
#       assuming 
##            tm_AU  tm_dmat   this._Aind_tmAU  ._dt_AU and GEARvecs  _Aind_tmNxt and _Aind_Gearvc_C_nxt
## DEFAULT  :                  None
## 1st call : 0.0    0.0       None->0.0        -> initialized
#                    h/Nintpl          
#                   2h/Nintpl          
#                   ...                
#         (Nintpl-1)h/Nintpl 
## 2nd call : h      h                          --                        h               rGEAR2ndOne(GEARvecs)
## 3rd call : h      h                          --                        h               rGEAR2ndOne(GEARvecs)
## AF-eldyn :                  h   _Aind_Gearvc_C_nxt 
##    ... here C is at this._Aind_tmAU,  P is at this._Aind_tmAU+dt_AU
def propagate_Aind_over_c_interpol(this,tm_AU,dt_AU, dm_kOR1, tm_dmat=None, Aind_over_c_ini=None,caller=""):
    dbgng_Aind_over_c_interpol=True;fdDBG=None
    flag_Aind=abs(this._calc_Aind)
    if( dbgng_Aind_over_c_interpol ):
        Istep= int( round(tm_AU/dt_AU) );Imod=Istep%2;
        if( Istep == 1 ):
            os.system("cat propagate_Aind_over_c_interpol_%d.wks > propagate_Aind_over_c_interpol_DBG.log"%(1-Imod) )
        elif( Istep > 0 ):
            idum=os.system("cat propagate_Aind_over_c_interpol_%d.wks >> propagate_Aind_over_c_interpol_DBG.log"%(1-Imod) )
            ### print("cat propagate_Aind_over_c_interpol_%d.wks >> propagate_Aind_over_c_interpol_DBG.log %d"%(1-Imod,idum) )
            ### os.system("cat propagate_Aind_over_c_interpol_%d.wks"%(1-Imod))
            ### os.system("fopen propagate_Aind_over_c_interpol_DBG.log")
        fdDBG= open("propagate_Aind_over_c_interpol_%d.wks"%(Imod),"w")
    
    def SQRdist3D(a,b):
        return  ( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 )
    def get_dipvel0_eff(this,dmat,tmAU_dmat,caller=""):   ## dmat-dependent part of dip-vel 
        dbgng=1
        flag=abs(this._calc_Aind)
        PIx4=12.566370614359172953850573533118;tmAU_TINY=1.0e-8
        if( this.cell_volume is None ):
            assert this.BravaisVectors_au is not None,""
            this.cell_volume=calc_3Dvolume( this.BravaisVectors_au )
        # ------
        # Strictly speaking, v^{0} (or paramagnetic current times cell_volume) depends on A through xVVx term
        # We however assume dependence of xVVx term on A (as well as ampd of xVVx term) to be small and use  
        # A_{ext+ind}(t) and DM^{guess}(t+h) to calculate v^{0}(t+h)
        # v^{phys} or v^{0}(t+h)+A(t+h)/c  on the other hand uses A carefully propagated in this subroutine
        # ------
        cLIGHT_AU=PhysicalConstants.CLIGHTinAU()
        tmAU_refr=this._time_AU
        assert (abs(tmAU_dmat)<tmAU_TINY) or abs(tmAU_dmat - (this._time_AU + this._dt_AU))<tmAU_TINY,"%f %f"%(tmAU_dmat,this._time_AU)
        Aext_over_c_t1 = ( np.zeros([3],dtype=np.float64) if( this._td_field is None ) else \
                                 this._td_field.get_vectorfield( tmAU_refr)/cLIGHT_AU )
        Aind_over_c_t1 = ( np.zeros([3],dtype=np.float64) if( this._Aind_Gearvc_C is None ) else this._Aind_Gearvc_C[0] )
        assert isinstance( Aext_over_c_t1, np.ndarray ) and isinstance( Aind_over_c_t1, np.ndarray ),"sum and scalar division ??"
        Atot_over_c_t1 = Aext_over_c_t1 + Aind_over_c_t1
        dic=get_dipole(this, dmat, mode='V', Aind_over_c=Aind_over_c_t1,Aext_over_c=Aext_over_c_t1,\
                       caller="get_dipvel0_eff:Aeff_t1<-"+caller, tmrefAU_Aind=tmAU_refr)
        N_ele=get_Nele(this); N_ele_sum=( N_ele if(this._spinrestriction == 'R') else (N_ele[0]+N_ele[1]) )
        if( dbgng > 0 ):
            # v1 = v0 + Aext * N_ele_sum 
            assert SQRdist3D( dic['dipvel_0'] + Aext_over_c_t1 * N_ele_sum, dic['dipvel_1'] ) < 1.0e-8,""
            assert SQRdist3D( dic['dipvel_0'] + Atot_over_c_t1 * N_ele_sum, dic['dipvel_2'] ) < 1.0e-8,""
            if(flag==4):
                assert SQRdist3D( dic['dipvel_3'] - Atot_over_c_t1 * N_ele_sum, dic['dipvel_x0'] ) < 1.0e-8,""
        if( flag != 4 ):
            return dic['dipvel_0']
        else:
            return dic['dipvel_x0']
    def make_d2Aind_over_c_dt2_(this, flag, time_au, veff0, Aind_over_c=None):
        N_ele=get_Nele(this)
        PIx4=12.566370614359172953850573533118; 
        cLIGHT_AU=PhysicalConstants.CLIGHTinAU()

        if( this.cell_volume is None ):
            assert this.BravaisVectors_au is not None,""
            this.cell_volume=calc_3Dvolume( this.BravaisVectors_au )
        fac= PIx4/this.cell_volume; q_e=-1
        if( this._alpha_Aind_over_c is not None ):
            fac=fac*this._alpha_Aind_over_c
        if( flag == 1 ):
            return np.array([ q_e*veff0[kk]*fac for kk in range(3) ])
        A_over_c = this._td_field.get_vectorfield(time_au)/cLIGHT_AU 
        if( flag == 2 ):
            return np.array([ q_e*(veff0[kk]-q_e*A_over_c[kk]*N_ele)*fac for kk in range(3) ])
        assert Aind_over_c is not None,""
        return np.array([ q_e*(veff0[kk]-q_e*( A_over_c[kk] + Aind_over_c[kk])*N_ele )*fac for kk in range(3) ])
            
    assert this._calc_Aind < 0,""
    tmAU_TINY=1.0e-8; AUinFS=0.02418884326058678; Ld=3; Implicit_Scheme=True; PIx4=12.566370614359172953850573533118;

    if( this._dt_AU is None ):
        this._dt_AU = ( dt_AU if(dt_AU is not None) else tm_AU-this._Aind_tmAU)
    else:
        assert abs( this._dt_AU - dt_AU )<tmAU_TINY,""

    if( this._Aind_over_c_Nintpl is None ):
        v=rttddft_common.Params_get("Aind_over_c_Nintpl")
        if( v is not None):
            this._Aind_over_c_Nintpl = int( v )
    t_eps = this._dt_AU/float( this._Aind_over_c_Nintpl )
    if( tm_dmat is not None ):
        assert  (  (this._Aind_tmAU is None and abs(tm_AU)<tmAU_TINY) or   (abs( tm_AU - this._Aind_tmAU)< tmAU_TINY ) )  or \
                ( Implicit_Scheme and abs(tm_dmat-tm_AU)<tmAU_TINY )                                                      or \
                ( (not Implicit_Scheme) and abs(tm_dmat-(tm_AU-dt_AU))<tmAU_TINY ), \
              "Implicit:%r tm_AU:%f tm_DMAT:%f t+dt:%f %e"%(Implicit_Scheme,tm_AU,tm_dmat,tm_AU-dt_AU,abs(tm_dmat-(tm_AU-dt_AU)))
    else:
        tm_dmat=( tm_AU-dt_AU if(not Implicit_Scheme) else tm_AU )
    this._d2Aind_over_c_dt2_tmAU = tm_dmat
    ### print("#make_d2Aind_over_c:",tm_AU,this._Aind_tmAU,this._Aind_Gearvc_C)

    if( this._Aind_tmAU is None or this._Aind_Gearvc_C is None ):
        assert (abs(tm_AU)<tmAU_TINY or Aind_over_c_ini is not None),""
        this._Aind_tmAU=tm_AU
        Aind_over_c = ( np.zeros([3],dtype=np.float64) if( Aind_over_c_ini is None ) else np.array( Aind_over_c_ini.copy() ) )        
        veff_now=np.array( get_dipvel0_eff( this, dm_kOR1, tmAU_dmat=tm_AU,caller="propagate_Aind_over_c_interpolL227<-"+caller ) ); ## paramagnetic current(*cell volume) + xVVx 
        this._d2Aind_over_c_dt2 = make_d2Aind_over_c_dt2_(this, flag_Aind, tm_AU, veff_now, Aind_over_c=Aind_over_c)
        this._Aind_Gearvc_C, this._Aind_Gearvc_P =initz_rGEARvec( t_eps, Aind_over_c, np.zeros([Ld],dtype=np.float64),
                                                                 this._d2Aind_over_c_dt2, Ld)
        this._veff_LAST=np.array( veff_now );this._veff_tmAU=tm_dmat
        ### print("#veff_LAST set:",this._veff_LAST)
        if( fdDBG is not None ):
            fctr= PIx4/this.cell_volume; q_e=-1
            if( this._alpha_Aind_over_c is not None ):
                fctr=fctr*this._alpha_Aind_over_c
            print(" %14.6f   %16.8f %16.8f %16.8f    %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f"%(tm_AU,\
                 this._Aind_Gearvc_C[0][0], this._Aind_Gearvc_C[0][1], this._Aind_Gearvc_C[0][2], \
                 this._d2Aind_over_c_dt2[0], this._d2Aind_over_c_dt2[1], this._d2Aind_over_c_dt2[2],\
                 0.0, 0.0, 0.0),file=fdDBG)
        if( fdDBG is not None ):
            fdDBG.close()

        return Aind_over_c;
    else:
        if( abs( tm_AU - this._Aind_tmAU )<tmAU_TINY ):
            return np.array( [ this._Aind_Gearvc_C[0][k] for k in range(3)] )
        else:
            ## I: propagate using DM(tm_AU) to fix the correction vector    E: propagate using  DM(tm_AU-h) to get the prediction vector
            if( this._Aind_Gearvc_C_nxt is None ):
                this._Aind_Gearvc_C_nxt=np.zeros([5,3],dtype=np.float64); this._Aind_Gearvc_P_nxt=np.zeros([5,3],dtype=np.float64);
            this._Aind_tmNxt=tm_AU
            for ii in range(5):
                for jj in range(3):
                    this._Aind_Gearvc_C_nxt[ii][jj]=this._Aind_Gearvc_C[ii][jj];
                    this._Aind_Gearvc_P_nxt[ii][jj]=this._Aind_Gearvc_P[ii][jj];
            veff_now=np.array( get_dipvel0_eff( this,dm_kOR1,tmAU_dmat=tm_AU,caller="propagate_Aind_over_c_interpolL257<-"+caller ) );
            this._veff_latest=np.array( [ veff_now[kk] for kk in range(3) ])
            this._tmAU_veff_latest=tm_AU

            ### print("#now:%f veff_tmAU:%f"%(tm_AU,this._veff_tmAU))
            assert (abs( this._veff_tmAU-(tm_AU-this._dt_AU) )<tmAU_TINY),"#now:%f veff_tmAU:%f"%(tm_AU,this._veff_tmAU)
            tmau=this._veff_tmAU
            for jstep in range(this._Aind_over_c_Nintpl):
                tmau=tmau + t_eps
                f=(jstep+1.0)/float( this._Aind_over_c_Nintpl )
                veff_j= f*veff_now + (1.0-f)*this._veff_LAST
                this._d2Aind_over_c_dt2 = make_d2Aind_over_c_dt2_(this, flag_Aind, tmau, veff_j, Aind_over_c=this._Aind_Gearvc_P_nxt[0])
                this._Aind_Gearvc_C_nxt, this._Aind_Gearvc_P_nxt= rGEAR2ndOne(t_eps,this._Aind_Gearvc_C_nxt,this._Aind_Gearvc_P_nxt,this._d2Aind_over_c_dt2,Ld)
                if( fdDBG is not None ):
                    fctr= PIx4/this.cell_volume; q_e=-1
                    if( this._alpha_Aind_over_c is not None ):
                        fctr=fctr*this._alpha_Aind_over_c
                    print(" %14.6f   %16.8f %16.8f %16.8f    %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f"%(tmau,\
                         this._Aind_Gearvc_C_nxt[0][0], this._Aind_Gearvc_C_nxt[0][1], this._Aind_Gearvc_C_nxt[0][2], \
                         this._d2Aind_over_c_dt2[0], this._d2Aind_over_c_dt2[1], this._d2Aind_over_c_dt2[2],\
                         fctr*q_e*veff_j[0], fctr*q_e*veff_j[1], fctr*q_e*veff_j[2]),file=fdDBG)
            if( fdDBG is not None ):
                fdDBG.close()
            return np.array( [ this._Aind_Gearvc_C_nxt[0][k] for k in range(3)] )
