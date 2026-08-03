#!/usr/bin/env python
# revision of pbc/gto/pseudo/pp_int.py
# 
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Analytic PP integrals.  See also pyscf/pbc/gto/pesudo/pp.py

For GTH/HGH PPs, see:
    Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
    Hartwigsen, Goedecker, and Hutter, PRB 58, 3641 (1998)
'''

import ctypes
import pathlib
import copy
import numpy
import os
import sys
import math
import scipy.special
from pyscf import lib
from pyscf import gto
import pathlib
from .filter_by_exponent import filter_by_exponent
from .futils import futils
from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vnl, _int_vnl
from pyscf.pbc.df.aft import _get_pp_loc_part1 as get_pp_loc_part1
from .utils import write_filex,update_dict,printout_dict,readwrite_xbuf,arrayclone,print_a2maxdiff,aNmaxdiff,print_Hmatrices,atomicsymbol_to_atomicnumber,parse_xyzstring,write_file,i1eqb,list_to_a1,gamma_hfint,atomicnumber_to_atomicsymbol
from pyscf.gto.mole import format_basis
import ctypes
from .rttddft_common import rttddft_common
from pyscf.pbc.gto.pseudo.pp_int import get_pp_nl
from pyscf.pbc.gto import pseudo
from numpy.ctypeslib import ndpointer
from .Logger import Logger
from .physicalconstants import physicalconstants
from .Loglv import printout
from mpi4py import MPI
from .mpiutils import mpi_Bcast
from .heapcheck import heapcheck
from .MPIutils01 import MPIutils01
import datetime
libpbc = lib.load_library('libpbc')

def suppress_prtout():
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    return ( MPIrank!=0 )


def fprintf(fpath,text,Append=False,Threads=[0],flush=False,stdout=False,dtme=True):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();MPIsize=comm.Get_size()
    if( MPIsize>1 and (MPIrank not in Threads) ):
        return
    fd=open(fpath,('a' if(Append) else 'w'))
    if( dtme ):
        text=text+"  \t\t"+str(datetime.datetime.now())
    print(text,file=fd);fd.close()
    if(stdout):
        printout(text)
def print_kvectors(kvectors,BravaisVectors,fpath=None,Append=False):
    PIx2=6.283185307179586476925286766559;
    if( suppress_prtout() ):
        return
    fd=sys.stdout;
    if(fpath is not None ):
        fd=open(fpath,('a' if(Append) else 'w'))
    ## B=numpy.transpose( numpy.linalg.inv(BravaisVectors) )*PIx2
    ## for I in range(3):
    ##    print("#check_BdotA:"+str( [ numpy.vdot(B[I],BravaisVectors[J]) for J in range(3) ] ))
    Nv=len(kvectors)
    print("#print_kvectors: Kx,Ky,Kz  projection_to_Gvectors",file=fd); ## rank=0 only
    for I in range(Nv):
        print(" %03d: %9.4f %9.4f %9.4f    %9.4f %9.4f %9.4f"%(I, kvectors[I][0],kvectors[I][1],kvectors[I][2],
            numpy.vdot(kvectors[I],BravaisVectors[0])/PIx2, numpy.vdot(kvectors[I],BravaisVectors[1])/PIx2,
            numpy.vdot(kvectors[I],BravaisVectors[2])/PIx2),file=fd ) ## rank=0 only
    if( fpath is not None ):
        fd.close()
def absmax_imaginary_part(zbuf):
    buf=numpy.ravel( numpy.array(zbuf) )
    le=len(buf)
    if(le<1):
        return 0
    I=0; maxv=abs( buf[I].imag );at=I
    for I in range(1,le):
        dum=abs( buf[I].imag )
        if(dum>maxv):
            maxv=dum;at=I
    return maxv
def modify_normalization(zbuf,nCGTO_2A,nKpoints,nCGTO_1,n_cols, IZnuc_B,distinct_IZnuc,ell_B,alpha_B,Nsh):
    # Input: zbuf[3][nKpoints][nCGTO_2A[:]][nCGTO_1]
    # fix normalization 
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    Nat=len(IZnuc_B)
    nDa=len(distinct_IZnuc)
    info=[]
    norb_AL=[0,0,0]
    nAO=0
    for Iat in range(Nat):
        IZ=IZnuc_B[Iat]
        for iDa in range(nDa):
            if(distinct_IZnuc[iDa]==IZ):
                break
        assert (distinct_IZnuc[iDa]==IZ),""
        offset=0
        for jda in range(iDa):
            offset+=Nsh[jda]
        for ksh in range(Nsh[iDa]):
            el=ell_B[offset+ksh]
            alph=alpha_B[offset+ksh]
            rLsqr=0.5/alph
            norb=2*el+1
            
            for m in range(-el,el+1):
                info.append({"indx":"%d.%d.%d"%(Iat,iDa,ksh),"alph":alph,"l":el,"m":m,"n_cols":n_cols[offset+ksh]})
            ##
            ## we now fix normalization...
            ##
            norb_AL[0]+=norb
            for Icol in range(1,3):
                if( nCGTO_2A[Icol]<1 ):
                    continue  ## skip as if there is no input
#
# I normalized p^{l}_n by 1.0/[ sqrt(0.5 Gamma( (2l+4I+3)/2 )) * rL**(l+(4I+3)/2) ]
# but here pySCF expects it to be normalized by I=0 value
# so I multiply with  sqrt( Gamma( (2l+4I+3)/2 )/Gamma( (2l+3)/2 ) )* rL**2I
                if( Icol< n_cols[offset+ksh]): ## 1 < 2 etc. 
                    fac=math.sqrt( gamma_hfint( 2*el+4*Icol+3 )/gamma_hfint( 2*el + 3) )*(rLsqr**Icol)
                    printout("#zbuf_%d:"%(Icol)+str(numpy.shape(zbuf[Icol])))
                    printout("#iorb [%d:%d+%d) / %d:"%(norb_AL[Icol], norb_AL[Icol], norb, nCGTO_2A[Icol]))
                    for kp in range(nKpoints):
                        for kAO in range(norb):
                            for mu1 in range(nCGTO_1):
                                ### printout("%d %d %d %d"%(Icol,kp,norb_AL[Icol]+kAO,mu1),flush=True)
                                zbuf[Icol][kp][ norb_AL[Icol]+kAO ][mu1]*=fac
                    norb_AL[Icol] +=norb
    ### fd=futils.fopen("modify_nrmz.log","w")  ## fd.close @line 96
    fpath1="modify_nrmz.log";  ## fd.close @line 96
    printout(info,fpath=fpath1,Threads=[0],Append=False);### futils.fclose(fd);
        
    ### print("norb_AL_%02d:"%(MPIrank)+str(norb_AL))
    ### print("nCGTO_2A_%02d:"%(MPIrank)+str(nCGTO_2A))
    assert ( norb_AL[0]==nCGTO_2A[0] and norb_AL[1]==nCGTO_2A[1] and
            ( norb_AL[2]==0 or norb_AL[2]==nCGTO_2A[2])),"norb_AL:"+str(norb_AL)
    return zbuf

def count_nCGTO(IZnuc,distinct_IZnuc,Nsh,ell,n_cols=None,title=None,verbose=0):
    # Nshell[iDa]
    # Ell[iDa][jsh]
    if( verbose > 0 ):
        printout("#count_nCGTO:INPUT:Nsh="+str(Nsh)+" for distinct_IZnuc:"+str(distinct_IZnuc))
    retv=0
    nDa=len(distinct_IZnuc)
    Ell=[]
    ixj=0
    ## Ell[ nDA ][ Nsh[*] ]
    for iDa in range(nDa):
        el_1=numpy.zeros([Nsh[iDa]],dtype=int)
        for ksh in range(Nsh[iDa]):
            el_1[ksh]=ell[ ixj + ksh ]
        ixj+=Nsh[iDa]

        Ell.append( el_1 )

    ret_Array=None
    info=[];
    NshSUM=0
    Nat=len(IZnuc)
    for iat in range(Nat):
        IZ=IZnuc[iat]
        for iDa in range(nDa):
            if(IZ==distinct_IZnuc[iDa]):
                break
        assert (iDa<nDa),""
        nOrb=0; arr=[]
        for ksh in range(Nsh[iDa]):
            nOrb+= 2*Ell[iDa][ksh]+1
            arr.append(Ell[iDa][ksh])
        dict={"Z":IZ,"Nsh":Nsh[iDa],"ell":str(arr)}
        # n_cols[0,,,Nsh[0]-1]
        #       [Nsh[0],...,Nsh[0]+Nsh[1]-1]
        #
        if(n_cols is not None):
            nco=[]
            off=0
            for jDb in range(iDa):
                off+=Nsh[jDb]
            for kk in range(Nsh[iDa]):
                nco.append(n_cols[off+kk])
            dict.update({"n_cols":str(nco)})
            if(ret_Array is None):
                ret_Array=[0,0,0]
            for Ico in range(3):
                if(Ico==0):
                    ret_Array[Ico]+=nOrb
                else:
                    for kk in range(Nsh[iDa]):
                        ## add Ico=1 if nco[shell]>1 ...
                        if(nco[kk]>Ico):
                            ret_Array[Ico]+=(2*Ell[iDa][kk]+1)

        NshSUM+=Nsh[iDa]
        retv+=nOrb
        ### printout("#Count_nCGTO:iat=%d iDa=%d nOrb=%d %d"%(iat,iDa,nOrb,retv));
    if( title is not None):
        printout("#Count_nCGTO:shell_info:"+str(title)+":"+str(info))
    ### if( n_cols is not None):
    ###    le=len(n_cols)
    ###    assert (le==NshSUM),"len:%d / NshSUM:%d"%(le,NshSUM)
    if(ret_Array is not None):
        assert retv==ret_Array[0],"%d/%d"%(retv,ret_Array[0])
    if(n_cols is None):
        return retv
    else:
        return ret_Array
def print_bset(IZnuc,distinct_IZnuc,Nsh,ell):
    Nat=len(IZnuc)
    Nsh_sum=sum(Nsh)
    printout("#IZnuc:%d"%(len(IZnuc)))

def gen_bset_info(cell,IZnuc):
    ## generalization of format_basis({'Ti':'sto-3g','O':'sto-3g'}) ..
    Nat=len(IZnuc)
    if( isinstance(cell.basis,str) ):
        dict={}
        for iat in range(Nat):
            Z=IZnuc[iat]
            Sy=atomicnumber_to_atomicsymbol(Z)
            dict.update({Sy:cell.basis})
        ### printout("gen_bset_info:INPUT:",end="");printout(dict)
        ret=format_basis(dict)
        ### printout("gen_bset_info:OUTPUT:",end="");printout(ret)
        return ret
    elif( isinstance(cell.basis,dict) ):
        ### print("gen_bset_info:INPUT:",end="");print(cell.basis)
        ret=format_basis(cell.basis)
        ### print("gen_bset_info:OUTPUT:",end="");print(ret)
        return ret
    else:
        assert False,"check cell.basis and its type.."+str(type(cell.basis))+" "+str(cell.basis)        
        return None

# alph,cofs,Ell,distinct_IZnuc,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_infox(cell,Sy)
def reorg_bset_infox(cell,Sy):
    Nat=len(Sy)
    IZnuc=[];
    for i in range(Nat):
        IZnuc.append( atomicsymbol_to_atomicnumber( Sy[i] ) )
    bset_info=gen_bset_info(cell,IZnuc)
    
    if( cell.exp_to_discard is not None ):
        filtered =filter_by_exponent(bset_info, cell.exp_to_discard )
        bset_info=filtered

    return reorg_bset_info(bset_info)
# alph,cofs,Ell,distinct_IZnuc,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_info(bset_info)
# nDa:=len(distinct_IZnuc)
# Nsh [ nDa ]   # of shell for each atom
# Ell [ sum(Nsh[:]) ]
# Npgto[ sum(Nsh[:]) ]
def reorg_bset_info(bset_info):
    distinct_IZnuc=[]
    nDa=len(bset_info)
    alph=[];cofs=[];
    Nsh=[]; Npgto_sum=0
    Npgto=[]
    Ell=[]
    for A in bset_info:
        infoA=bset_info[A];
        nsh1=len(infoA)
        
        distinct_IZnuc.append( atomicsymbol_to_atomicnumber(A) )
        for jsh in range(nsh1):
            shell=infoA[jsh]
            el=shell[0]; Ell.append(el);
            npgto=0; alpha_col2=None; cofs_col2=None; alpha_col3=None; cofs_col3=None
            alpha_col4=None; cofs_col4=None; alpha_col5=None; cofs_col5=None; alpha_col6=None; cofs_col6=None
            alpha_col7=None; cofs_col7=None; alpha_col8=None; cofs_col8=None
            for j in range(1,len(shell)):
                row=shell[j]
                alph.append(row[0]);cofs.append(row[1]);npgto+=1
                if(len(row)>2):
                    if(len(row)>=3):
                        if( alpha_col2 is None):
                            alpha_col2=[]; cofs_col2=[]
                        alpha_col2.append(row[0]);cofs_col2.append(row[2]);
                    if(len(row)>=4):
                        if( alpha_col3 is None):
                            alpha_col3=[]; cofs_col3=[]
                        alpha_col3.append(row[0]);cofs_col3.append(row[3]);
                    if(len(row)>=5):
                        if( alpha_col4 is None):
                            alpha_col4=[]; cofs_col4=[]
                        alpha_col4.append(row[0]);cofs_col4.append(row[4]);
                    if(len(row)>=6):
                        if( alpha_col5 is None):
                            alpha_col5=[]; cofs_col5=[]
                        alpha_col5.append(row[0]);cofs_col5.append(row[5]);
                    if(len(row)>=7):
                        if( alpha_col6 is None):
                            alpha_col6=[]; cofs_col6=[]
                        alpha_col6.append(row[0]);cofs_col6.append(row[6]);
                    if(len(row)>=8):
                        if( alpha_col7 is None):
                            alpha_col7=[]; cofs_col7=[]
                        alpha_col7.append(row[0]);cofs_col7.append(row[7]);
                    if(len(row)>=9):
                        if( alpha_col8 is None):
                            alpha_col8=[]; cofs_col8=[]
                        alpha_col8.append(row[0]);cofs_col8.append(row[8]);
                    if(len(row)>=10):
                        assert False,"unimplemnted"
                
            Npgto.append(npgto); Npgto_sum+=npgto
            if( alpha_col2 is not None):
                Ell.append(el); ## Ell[\sum Nsh[:]]
                npgto_2=len(alpha_col2)
                Npgto.append(npgto_2)
                for k in range(npgto_2):
                    ### print("#adding to alph:%d %f"%(k,alpha_col2[k]))
                    alph.append( alpha_col2[k] )
                    cofs.append( cofs_col2[k] )
                nsh1+=1; Npgto_sum+=npgto_2
            if( alpha_col3 is not None):
                Ell.append(el); ## Ell[\sum Nsh[:]]
                npgto_3=len(alpha_col3)
                Npgto.append(npgto_3)
                for k in range(npgto_3):
                    alph.append( alpha_col3[k] )
                    cofs.append( cofs_col3[k] )
                nsh1+=1; Npgto_sum+=npgto_3

            if( alpha_col4 is not None):
                Ell.append(el); ## Ell[\sum Nsh[:]]
                npgto_4=len(alpha_col4)
                Npgto.append(npgto_4)
                for k in range(npgto_4):
                    alph.append( alpha_col4[k] )
                    cofs.append( cofs_col4[k] )
                nsh1+=1; Npgto_sum+=npgto_4
            if( alpha_col5 is not None):
                Ell.append(el); ## Ell[\sum Nsh[:]]
                npgto_5=len(alpha_col5)
                Npgto.append(npgto_5)
                for k in range(npgto_5):
                    alph.append( alpha_col5[k] )
                    cofs.append( cofs_col5[k] )
                nsh1+=1; Npgto_sum+=npgto_5
            if( alpha_col6 is not None):
                Ell.append(el); ## Ell[\sum Nsh[:]]
                npgto_6=len(alpha_col6)
                Npgto.append(npgto_6)
                for k in range(npgto_6):
                    alph.append( alpha_col6[k] )
                    cofs.append( cofs_col6[k] )
                nsh1+=1; Npgto_sum+=npgto_6
            if( alpha_col7 is not None):
                Ell.append(el); ## Ell[\sum Nsh[:]]
                npgto_7=len(alpha_col7)
                Npgto.append(npgto_7)
                for k in range(npgto_7):
                    alph.append( alpha_col7[k] )
                    cofs.append( cofs_col7[k] )
                nsh1+=1; Npgto_sum+=npgto_7
            if( alpha_col8 is not None):
                Ell.append(el); ## Ell[\sum Nsh[:]]
                npgto_8=len(alpha_col8)
                Npgto.append(npgto_8)
                for k in range(npgto_8):
                    alph.append( alpha_col8[k] )
                    cofs.append( cofs_col8[k] )
                nsh1+=1; Npgto_sum+=npgto_8
        Nsh.append(nsh1);        
    Nsh_sum=sum(Nsh)
    return alph,cofs,Ell,distinct_IZnuc,Nsh,Npgto,Npgto_sum,Nsh_sum

# retv[0] [ nKpoints ][ nCGTO_2A[0] ][ nCGTO_1 ]
# retv[1] [ nKpoints ][ nCGTO_2A[1] ][ nCGTO_1 ]  *** truncated if nCGTO_2A[1]==0 ***
# retv[2] [ nKpoints ][ nCGTO_2A[1] ][ nCGTO_1 ]
def calc_pp_overlaps(cell,A_over_c,kpts, singlethread=None,dbgng=False,Zbuf_Refr=None):
    import math
    import time
    comm=MPI.COMM_WORLD
    kvectors=numpy.reshape(kpts,(-1,3))
    nKpoints=len(kvectors)
    MPIrank=comm.Get_rank();MPIsize=comm.Get_size()
    Rnuc_au,Sy=parse_xyzstring(cell.atom,unit_BorA='B')
    Nat_too_small=3
    Nat=len(Sy)
    dbgng_mpi=False
    if( singlethread is None ):
        singlethread = ( MPIsize < 2 or Nat <= Nat_too_small )
    if( MPIsize < 2 ):
        singlethread = True

    if( not MPIutils01.Multithreading() ):
        MPIutils01.DebugTrace("calc_pp_overlaps","#calc_pp_overlaps:singlethread");
        singlethread = True

    fncnme="calc_pp_overlaps";fncdepth=4      ## XXX XXX  fncdepth ~ SCF:0 vhf:1 get_j:2
    Wctm000=time.perf_counter();Wctm010=Wctm000;N_call=rttddft_common.Countup(fncnme);dic1_timing={} ## XXX XXX
    Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)                      ## XXX XXX
    if( Dic_timing is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});
        Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)
    f_write("calc_pp_overlaps_DBGNG.log",Append=True,msg="calc_pp_overlaps Nat:%d "%(len(Sy))+str(Sy))
    if( singlethread ):
        ### printout("calling calc_pp_overlaps1... %r"%(dbgng),fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
        retv=calc_pp_overlaps1(cell, A_over_c, kpts, Rnuc_au, Sy, Rnuc_au, Sy, dbgng=dbgng, Zbuf_Refr=Zbuf_Refr)
        ### printout("calling calc_pp_overlaps1... returns",fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
        update_dict(fncnme,dic1_timing,Dic_timing,"singlethread",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
        Wctm010=time.perf_counter()  # XXX XXX
        printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth) # XXX XXX
        return retv

    ## divide task over threads...
    ## Here I take the SIMPLEST strategy, which is not necessarily optimal
    ## better choice is to evenly distribute task taking account of N_shells/N_pgtos of each atom..
    ## but in such case you have to broadcast matrix dimension etc.
    ## NOTE we here assume (i)  numpy.shape( ppnl_half_sub ) [0] and [1] are the same
    ##                     (ii) numpy.shape( ppnl_half_sub ) [2] is largest 
    Nat_per_thread=int( math.ceil( Nat / float(MPIsize) ) )
    iatSTT= min( MPIrank*Nat_per_thread, Nat)
    iatUPL= min( (MPIrank+1)*Nat_per_thread, Nat) 

    f_write("calc_pp_overlaps_DBGNG.log",Append=True,msg="calc_pp_overlaps iatSTT:%d %d "%(iatSTT,iatUPL),Threads=None)
    ppnl_half_sub=None; le_ppnl_half_sub=0
    if( iatSTT < iatUPL ):
        Rnuc_sub=Rnuc_au[iatSTT:iatUPL]
        Sy_sub=Sy[iatSTT:iatUPL]
        ## [3][ nKpoints ][ n
        ppnl_half_sub=calc_pp_overlaps1(cell, A_over_c, kpts, Rnuc_sub, Sy_sub, Rnuc_au, Sy, dbgng=dbgng )
        le_ppnl_half_sub = len( ppnl_half_sub )
        ### if( dbgng_mpi ):
        ###    mpi_prtout_zbuf("ppnl_half_sub%02d.dat",ppnl_half_sub,
        ###        description="Nat:%d:%d"%(iatSTT,iatUPL)+str(Rnuc_sub),Append=False)
    Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"multithread_ppnl",Wctm010-Wctm020,depth=fncdepth) # XXX XXX

    IZnuc=[ atomicsymbol_to_atomicnumber( Sy[i] ) for i in range(Nat) ]
    alph,cofs,Ell,distinct_IZnuc,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_infox(cell,Sy)    

    ## METADATA : nCGTO1
    nCGTO1_A=[]
    for r in range(MPIsize):
        iatSTT= min( r*Nat_per_thread, Nat)
        iatUPL= min( (r+1)*Nat_per_thread, Nat) 
        if( iatSTT < iatUPL ):
            IZnuc_sub=IZnuc[iatSTT:iatUPL]
            ## this applies IZnuc_sub 
            ncgto=count_nCGTO(IZnuc_sub,distinct_IZnuc,Nsh,Ell)
            nCGTO1_A.append(ncgto)
        else:
            nCGTO1_A.append(0)
    
    nCGTO1_sum=sum(nCGTO1_A)

    ## METADATA : nCGTO2
    nCGTO2_A=numpy.zeros([3], dtype='i')

    for Icol in range(3):
        if( (ppnl_half_sub is None) or (Icol >=le_ppnl_half_sub) ):
            nCGTO2_A[Icol]=0
        else:
            ndim=numpy.shape( ppnl_half_sub[Icol] )
            nCGTO2_A[Icol]=ndim[1]  ## [0]:nKpoints [1]:nCGTO2 [2]:nCGTO1
    if( MPIrank == 0 ):
        for Icol in range(len(ppnl_half_sub)):
            print("#00:numpy.shape(ppnl_half_sub[Icol=%d])"%(Icol)\
                  +str(numpy.shape(ppnl_half_sub[Icol])) )
    print("#%02d:nCGTO2_A:"%(MPIrank)+str(nCGTO2_A))
    print("#%02d:nCGTO1_A:"%(MPIrank)+str(nCGTO1_A))
    mpi_Bcast("calc_pp_overlaps.nCGTO2",nCGTO2_A,root=0)
    
    ## final buffer
    ppnl_half=[]
    for Icol in range(3):
        if( nCGTO2_A[Icol] < 1 ):
            continue
        zbuf1=numpy.zeros( [ nKpoints, nCGTO2_A[Icol],nCGTO1_sum ],dtype=numpy.complex128)
        ioff=0
        for r in range(MPIsize):
            if( nCGTO1_A[r] < 1 ):
                continue
            iupl=ioff + nCGTO1_A[r]
            if(r==0):
                if(MPIrank==0):
                    zbuf1[:,:, ioff:iupl]=ppnl_half_sub[Icol][:,:,0:nCGTO1_A[r]]
                    ### mpi_prtout_zbuf("zbuf1_%d_0"%(Icol)+"_%02d",zbuf1[:,:, ioff:iupl],description="[%d:%d)"%(ioff,iupl),Append=False)
                    ### mpi_prtout_zbuf("ppnl_half_sub_%d"%(Icol)+"_%02d",ppnl_half_sub[Icol],description="",Append=False)
            else:
                cwks=numpy.zeros([nKpoints, nCGTO2_A[Icol],nCGTO1_A[r]], dtype=numpy.complex128)
                if( MPIrank==0 ):
                    comm.Recv(cwks, source=r, tag=( MPIsize*Icol + r ))
                    zbuf1[:,:,ioff:iupl] =  cwks
                    ### mpi_prtout_zbuf("zbuf1_%d_%d"%(Icol,r)+"_%02d",zbuf1[:,:, ioff:iupl],description="[%d:%d)"%(ioff,iupl),Append=False)
                    
                elif( MPIrank == r ):
                    comm.Send( numpy.array(ppnl_half_sub[Icol]), dest=0,  tag=( MPIsize*Icol + r ))
                    ### mpi_prtout_zbuf("ppnl_half_sub_%d_%d"%(Icol,r)+"_%02d",ppnl_half_sub[Icol],description="",Append=False)
                

            ioff=iupl ## <<< never remove this >>>
        ppnl_half.append( zbuf1 )

    Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"mpi_send-recv",Wctm010-Wctm020,depth=fncdepth) # XXX XXX

    ### mpi_prtout_zbuf("ppnl_half_retv0a_%02d",ppnl_half[0],description="singlethread:%r"%(singlethread),Append=True)
    ## then broadcast the result..
    le_ppnl_half=len(ppnl_half)
    for Icol in range(le_ppnl_half):
        if( nCGTO2_A[Icol] < 1 ):
            continue
        mpi_Bcast("calc_pp_overlaps.ppnl_half", ppnl_half[Icol], root=0)
    Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"mpi_Bcast",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
    Wctm010=time.perf_counter()  # XXX XXX
    printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth) # XXX XXX
    ### mpi_prtout_zbuf("ppnl_half_retv0b_%02d",ppnl_half[0],description="singlethread:%r"%(singlethread),Append=True)

    if( MPIrank==0 and get_pp_nl01_static.prtout_calc_pbc_overlaps_retv_ ):
        get_pp_nl01_static.fnme_prtout_calc_pbc_overlaps_retv_="calc_pbc_overlaps_pyretv.dat"
        get_pp_nl01_static.AoverC_prtout_calc_pbc_overlaps_retv_=numpy.array( [ A_over_c[0], A_over_c[1], A_over_c[2] ] )
        fd=futils.fopen( get_pp_nl01_static.fnme_prtout_calc_pbc_overlaps_retv_, "w");# OK;
        for Icol in range(3):
            nCGTO_i=nCGTOb_Array[Icol]
            if( nCGTO_i < 1 ):
                continue
            print("###%d %d %d"%(nKpoints,nCGTOb_Array[Icol],nCGTO_1),file=fd)
            for kp in range(nKpoints):
                for j2 in range(nCGTOb_Array[Icol]):
                    print("#%d,%d,%d:\t "%(Icol,kp,j2),file=fd)
                    for k1 in range(nCGTO_1):
                        print("%16.8f j%16.8f        "%( zbuf[Icol][kp][j2][k1].real, zbuf[Icol][kp][j2][k1].imag ),file=fd)
        futils.fclose(fd);

    return ppnl_half

def f_write(path,Append,msg,Threads=[0]):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    if(Threads is not None):
        if(MPIrank not in Threads):
            return
    fd=open(path,("a" if(Append) else "w"))
    print(msg,file=fd);fd.close()

def calc_pp_overlaps1(cell, A_over_c, kpts, RnucAU_sub, Sy_sub, RnucAU_full, Sy_full, dbgng=False, Zbuf_Refr=None):
    #libname=pathlib.Path().absolute() / "libpp01.so"    
    #c_lib = ctypes.CDLL(libname)
    c_lib = lib.load_library("libpp01")

    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();

    kvectors=numpy.reshape( kpts, (-1,3))
    nKpoints=len(kvectors)
    kvecs=[]
    for v in numpy.ravel(kvectors):
        kvecs.append(v)
    kvectors_1=( ctypes.c_double*(nKpoints*3) )(*kvecs)

    Nat_sub=len(Sy_sub)
    IZnuc_sub=[];
    for i in range(Nat_sub):
        IZnuc_sub.append( atomicsymbol_to_atomicnumber( Sy_sub[i] ) )
    Nat_full=len(Sy_full)
    IZnuc_full=[];
    for i in range(Nat_full):
        IZnuc_full.append( atomicsymbol_to_atomicnumber( Sy_full[i] ) )

    bset_info=gen_bset_info(cell,IZnuc_sub)
    
    if( cell.exp_to_discard is not None ):
        filtered =filter_by_exponent(bset_info, cell.exp_to_discard )
        bset_info=filtered

    ## 
    alph,cofs,Ell,distinctIZnuc_sub,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_info(bset_info)
    nDa_sub=len(bset_info)
    
    IZnuc_B=IZnuc_full.copy(); Nat_B=len(IZnuc_full)
    alph_B=[];cofs_B=[];Nsh_B=[];NpGTO_B=[];ell_B=[]
    distinct_IZnuc_B=[];N_cols=[];n_cum=0
    for symb in cell._pseudo:
        n_th=0;n_cum=0
        distinct_IZnuc_B.append( atomicsymbol_to_atomicnumber(symb) )
        pp = cell._pseudo[symb]
        nproj_types = pp[4]
        nsh=0
        for l, (rl, nl, hl) in enumerate(pp[5:]):
            n_th+=1;n_cum+=(2*l+1)
            ### print("#ppshell_"+symb+":%d l=%d n_cum=%d"%(n_th,l,n_cum))
            if nl > 0:
                alpha = .5 / rl**2
                alph_B.append(alpha); cofs_B.append(1.0);    # for each pGTO
                ell_B.append(l); NpGTO_B.append(1); nsh+=1;  # for each shell
        Nsh_B.append(nsh)                                    # for each nDa_sub
#
    n_call=Logger.Countup("pp_ncols");
    verbose_pp=(0 if(n_call!=1) else 2)
    n_cols=get_ncols(cell,IZnuc_full,distinct_IZnuc_B,Nsh_B,title="pp_ncols",verbose=verbose_pp )
    n_cols=list_to_a1(n_cols)
    BravisVectors= numpy.array( cell.a )/physicalconstants.BOHRinANGS
## note: cell.a this is given in ANGS unit...
## 
    ### print("#BravisVectors","");print(BravisVectors);### assert False,""
    spdm=3;
    BravisVectors_1=( ctypes.c_double*(spdm*3) )( *numpy.ravel(BravisVectors) )
    Vectorfield_1=( ctypes.c_double*(3) )( *A_over_c )
    nKpoints=len(kpts);### print(kpts);print(nKpoints);### assert False,""

    Npgto_sum_B=len(alph_B)
    Nsh_sum_B=sum(Nsh_B)
    nDa_B=len(distinct_IZnuc_B)
    le=len(n_cols)
    ncols_2=( ctypes.c_int*(le))(*n_cols)
    RnucAU_sub =list(numpy.ravel(RnucAU_sub))
    Rnuc_1 =( ctypes.c_double*(3*Nat_sub) )(*RnucAU_sub)
    IZnuc_1=( ctypes.c_int*(Nat_sub) )( *IZnuc_sub )
    Nsh_1  =( ctypes.c_int*(nDa_sub) )( *Nsh ) 
    Ell_1  =( ctypes.c_int*(Nsh_sum) )( *Ell )
    Npgto_1=( ctypes.c_int*(Nsh_sum) )(*Npgto)
    distinct_IZnuc_1=( ctypes.c_int*(nDa_sub) )( *distinctIZnuc_sub )
    alph_1=( ctypes.c_double*(Npgto_sum) )(*alph)
    cofs_1=( ctypes.c_double*(Npgto_sum) )(*cofs) 

    nCGTO_1=count_nCGTO(IZnuc_sub,distinctIZnuc_sub,Nsh,Ell); nCGTO_1=int(nCGTO_1)
    nCGTOb_Array=count_nCGTO(IZnuc_B,distinct_IZnuc_B,Nsh_B,ell_B,n_cols=n_cols,verbose=verbose_pp);
    nCGTO_2=nCGTOb_Array[0]
    ### print(nCGTO_1,end="");print(nCGTO_1)
    ### print(nCGTO_1,end="");print(type(nCGTO_1))

    ### print(nCGTO_2,end="");print(nCGTO_2)
    ### print(nCGTO_2,end="");print(type(nCGTO_2))
    nCGTO_2=int(nCGTO_2)
###    nCGTO_2=count_nCGTO(IZnuc_B,distinct_IZnuc_B,Nsh_B,ell_B,n_cols=n_cols,title="pseudo");nCGTO_2=int(nCGTO_2)

    IZnuc_2=( ctypes.c_int*(Nat_B) )( *IZnuc_B )
    RnucAU_full =list(numpy.ravel(RnucAU_full))
    Rnuc_2 =( ctypes.c_double*(3*Nat_full) )(*RnucAU_full)

    Nsh_2  =( ctypes.c_int*(nDa_B) )( *Nsh_B ) 
    Ell_2  =( ctypes.c_int*(Nsh_sum_B) )( *ell_B )
    Npgto_2=( ctypes.c_int*(Nsh_sum_B) )(*NpGTO_B)
    distinct_IZnuc_2=( ctypes.c_int*(nDa_B) )( *distinct_IZnuc_B )
    alph_2=( ctypes.c_double*(Npgto_sum_B) )(*alph_B)
    cofs_2=( ctypes.c_double*(Npgto_sum_B) )(*cofs_B) 
    Ndim_retv=[3,nKpoints,nCGTO_2,nCGTO_1,2]

    if( nKpoints*nCGTO_2*nCGTO_1 <=0 ):
        printout("zero dimension of PSP:"+str( [nKpoints,nCGTO_2,nCGTO_1] ),warning=1)
        kpts_lst = ( numpy.zeros((1,3)) if(kpts is None) else numpy.reshape(kpts, (-1,3)) )
        fakecell, hl_blocks = fake_cell_vnl(cell)
        ppnl_half_o = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
        printout("ppnl_half_o:SHAPE:"+str(numpy.shape(ppnl_half_o))+"  /"+str( Ndim_retv ),flush=True)
        printout("ppnl_half_o:VALS:"+str( ppnl_half_o ) );
        ### print("ppnl_half_o:SHAPE:"+str(numpy.shape(ppnl_half_o))+"  /"+str( Ndim_retv ),flush=True)
        ### print("ppnl_half_o:VALS:"+str( ppnl_half_o ) );
        ### assert False,"PLS check result"
        return ppnl_half_o
    assert (nKpoints*nCGTO_2*nCGTO_1>0),"%d %d %d"%(nKpoints,nCGTO_2,nCGTO_1)
    Lx=20;Ly=20;Lz=20
    Lx=8;Ly=8;Lz=8
    piX2=6.283185307179586476925286766559
    Lj=[-1,-1,-1];
    dbgng_Lx=True; dbgfpath="Lx_for_FT.log"
    if( dbgng_Lx ):
        print_kvectors(kvectors,BravisVectors,fpath=dbgfpath)
    for dir in range(3):
        abs_aDOTk=[ abs(numpy.vdot(BravisVectors[dir],kvectors[j])) for j in range(nKpoints) ]
        ceil_2pi_over_aDOTk=[ ( 1 if(abs_aDOTk[j]<1e-6) else int(math.ceil( piX2/abs_aDOTk[j])) ) for j in range(nKpoints) ]
        Lj[dir]=max( ceil_2pi_over_aDOTk )
        fprintf(dbgfpath,"%02d:"%(dir)+str(abs_aDOTk)+" > "+str(ceil_2pi_over_aDOTk),Append=True)
    Lx=Lj[0];Ly=Lj[1];Lz=Lj[2]

    fileIO=False;both=False
    key='calc_pbc_overlaps_output'
    if( rttddft_common.Params_get(key) is not None ):
        dum=rttddft_common.Params_get(key)
        if( dum is not None):
            if( dum == 'F' or dum=='f'):
                fileIO=True;both=False
            elif( dum=='B' or dum=='b'):
                fileIO=True;both=True
            elif( dum=='D' or dum=='d'):
                fileIO=False;both=False
            else:
                assert False, key+":"+str(dum)
    zbuf=None;zbuf_refr=None;

    ### nc=get_pp_nl01_static.Countup(fnme)
    prtout_retv1=False; ## ( nc==1 )
    prtout_retv2=False; ## ( nc==1 )

    if(dbgng):                                   ## XXX XXX 
        both=True;fileIO=True;prtout_retv1=True  ## XXX XXX 
    ### printout("dbgng:%r both:%r"%(dbgng,both),fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)

    N_call_dbgclib = rttddft_common.Countup("Call_calc_pbc_overlaps_dbg")
    dbgng_clib=False
    if(dbgng_clib):
        if( N_call_dbgclib==1 or N_call_dbgclib==5 ):
            flag=1+2; N_test=8000;I_skip=400
            for I_test in range(N_test):
                c_lib.calc_pbc_overlaps_dbg( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                        Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                        Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                        Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                        cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                        kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank, flag);
                if( I_test==0 or I_test==5 or (I_test+1)%I_skip==0 ):
                    if( MPIrank==0 ):
                        fprintf("calc_pbc_overlaps_dbg_python.log","#c_lib.calc_pbc_overlaps_dbg test%4d END"%(I_test+1),Append=True)
        print("#c_lib.calc_pbc_overlaps_dbg.1 END",flush=True)
        flag=1+2+4+8+16+32; N_test=200; I_skip=20
        if( N_call_dbgclib==1 ):
            N_test=20;I_skip=5
        elif( N_call_dbgclib==3 or N_call_dbgclib==4):
            N_test=4000;I_skip=400
        else:
            N_test=0;I_skip=1
        for I_test in range(N_test):
            c_lib.calc_pbc_overlaps_dbg( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                    Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                    Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                    Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                    cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                    kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank, flag);
            if( I_test==0 or I_test==5 or (I_test+1)%I_skip==0 ):
                if( MPIrank==0 ):
                    fprintf("calc_pbc_overlaps_dbg_python.log","#c_lib.calc_pbc_overlaps_dbg FULL test%4d / %4d"%(I_test+1,N_test),Append=True)
        print("#c_lib.calc_pbc_overlaps_dbg.2 END",flush=True)

    n_call=rttddft_common.Countup("calc_pp_overlaps");
    dbgng_libf01o=( n_call==1 or n_call==5)    ## XXX XXX PLS. remove this in the expanse cluster ...
    ### printout("dbgng_libf01o:%r n_call:%d"%(dbgng_libf01o,n_call),fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
    if( fileIO ):
#int calc_pbc_overlaps_f(int nAtm,  double *Rnuc, int *IZnuc,  int nDa,  int *distinctIZnuc,
#    const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
#    int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
#    const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
#    const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
#    double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank, const int filenumber)
        filenumber=MPIrank
        heapcheck("calc_pbc_overlaps-BF")
        c_lib.calc_pbc_overlaps_f01( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                    Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                    Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                    Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                    cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                    kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank, filenumber);
        heapcheck("calc_pbc_overlaps-AF")
        retf="calc_pbc_overlaps_%03d.retf"%(filenumber)
        ### printout("FILE:%s"%(retf),fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
        retv=numpy.reshape( readwrite_xbuf('R',retf), [3,nKpoints,nCGTO_2,nCGTO_1])
        zbuf=[]
        for Icol in range(3):
            nCGTO_i=nCGTOb_Array[Icol]
            if( nCGTO_i<1 ):
                continue
            array=numpy.zeros([nKpoints,nCGTO_i,nCGTO_1],dtype=numpy.complex128)
            for kp in range(nKpoints):
                for j2 in range(nCGTOb_Array[Icol]):
                    for k1 in range(nCGTO_1):
                        array[kp][j2][k1] = retv[Icol][kp][j2][k1]
            zbuf.append(array)
        if( dbgng_libf01o ):
            filenumber=MPIrank+400
            c_lib.calc_pbc_overlaps_f( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                        Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                        Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                        Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                        cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                        kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank, filenumber);
            retf="calc_pbc_overlaps_%03d.retf"%(filenumber)
            printout("flib01o:%s"%(retf),fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
            retv_ref=numpy.reshape( readwrite_xbuf('R',retf), [3,nKpoints,nCGTO_2,nCGTO_1])
            zbuf_ref=[]
            for Icol in range(3):
                nCGTO_i=nCGTOb_Array[Icol]
                if( nCGTO_i<1 ):
                    continue
                array=numpy.zeros([nKpoints,nCGTO_i,nCGTO_1],dtype=numpy.complex128)
                for kp in range(nKpoints):
                    for j2 in range(nCGTOb_Array[Icol]):
                        for k1 in range(nCGTO_1):
                            array[kp][j2][k1] = retv_ref[Icol][kp][j2][k1]
                zbuf_ref.append(array)
                zbuf_diff.append( aNmaxdiff( zbuf[Icol], zbuf_ref[Icol]) )
            if(MPIrank==0):
                fdDBG=open("dbgng_libf01o.log","a")
                print(zbuf_diff,file=fdDBG);fdDBG.close()
            verbose_DBG=False
            if(verbose_DBG):
                fd=futils.fopen("dbgng_libf01o_zbuf.log","a")  ## OK
                print("#A_over_c:%f,%f,%f"%(A_over_c[0],A_over_c[1],A_over_c[2]),file=fd);
                for kp in range(nKpoints):
                    print("#%03d:K=%d\n"%(kp,kp),file=fd);
                    for j2 in range(nCGTO_2):
                        string="";strng2=""
                        for k1 in range(nCGTO_1):
                            string+="%16.8f %16.8f        "%(zbuf[Icol][kp][j2][k1][0].real,zbuf[Icol][kp][j2][k1][1].imag)
                            strng2+="%16.8f %16.8f        "%(zbuf_ref[Icol][kp][j2][k1][0].real,zbuf_ref[Icol][kp][j2][k1][1].imag)
                        print(string,file=fd);
                        print(strng2,file=fd);
                        ### print("%d %d %d %d  %f %f"%(Icol,kp,j2,k1,dbuf[Icol][kp][j2][k1][0],dbuf[Icol][kp][j2][k1][1]),file=fd)
                futils.fclose(fd);

            dev=max( zbuf_diff );devTOL=1.0e-7
            if( dev >= devTOL or MPIrank==0 ):
                print("#dbgng_libf01o:diff:%e"%(dev)+ str(zbuf_diff))
            assert dev<devTOL,""
            
    if(both):
        ### printout("copying zbuf to zbuf_refr",fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
        zbuf_refr=[ arrayclone(zbuf[J]) for J in range( len(zbuf) ) ]
        
    if(not fileIO or both):
        c_lib.calc_pbc_overlaps02.restype= ndpointer(dtype=ctypes.c_double,shape=Ndim_retv )
        print("#clib:restype:Ndim_retv:"+str(Ndim_retv));
        ### Here is alternative choice: parameters are handed over files
        write_file("calc_pbc_overlaps01.in", Nat_sub,RnucAU_sub,IZnuc_sub,nDa_sub,distinctIZnuc_sub, 
                                      Nsh, numpy.ravel(Ell), numpy.ravel(Npgto) ,numpy.ravel(alph),numpy.ravel(cofs),
                                      Nat_full,   numpy.ravel(RnucAU_full),numpy.ravel(IZnuc_B),nDa_B,
                                      numpy.ravel(distinct_IZnuc_B),  Nsh_B,    numpy.ravel(ell_B), n_cols, NpGTO_B,
                                      alph_B,   cofs_B,   spdm,    numpy.ravel(BravisVectors), A_over_c,  
                                      nKpoints, kvecs,    Lx,      Ly,                         Lz, 
                                      nCGTO_2,  nCGTO_1)
        ### os.system("./testread.x");
        ### zbuf=read_zbuf("calc_pbc_overlaps.dat")
        
        dbuf=c_lib.calc_pbc_overlaps02( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                        Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                        Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                        Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                        cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                        kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank)
        ### printout("c_lib.calc_pbc_overlaps02 returns..."+str(dbuf),fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
        ### prtout_retv1=(MPIrank==0)
        if(prtout_retv1): # xxx xxx remove after debugging xxx xxx 
            for Icol in range(3):
                str_AoverC=("0" if( abs(A_over_c[0])<1.0e-7) else ("%6.3f"%(A_over_c[0])).strip() )+"_"+\
                           ("0" if( abs(A_over_c[1])<1.0e-7) else ("%6.3f"%(A_over_c[1])).strip() )+"_"+\
                           ("0" if( abs(A_over_c[2])<1.0e-7) else ("%6.3f"%(A_over_c[2])).strip() )
#                fnme="calc_pbc_overlaps_A"+str_AoverC
                fnme="calc_pbc_overlaps_%02d"%(MPIrank)
                fpath01=fnme+"I%d.dat"%(Icol);
                fd=futils.fopen(fpath01,"w")  ## OK
                print("#A_over_c:%f,%f,%f"%(A_over_c[0],A_over_c[1],A_over_c[2]),file=fd);
                for kp in range(nKpoints):
                    print("#%03d:K=%d\n"%(kp,kp),file=fd);
                    for j2 in range(nCGTO_2):
                        string=""
                        for k1 in range(nCGTO_1):
                            string+="%16.8f %16.8f        "%(dbuf[Icol][kp][j2][k1][0],dbuf[Icol][kp][j2][k1][1])
                        print(string,file=fd);
                        ### print("%d %d %d %d  %f %f"%(Icol,kp,j2,k1,dbuf[Icol][kp][j2][k1][0],dbuf[Icol][kp][j2][k1][1]),file=fd)
                futils.fclose(fd);
        zbuf=[]
        for Icol in range(3):
            nCGTO_i=nCGTOb_Array[Icol]
            if( nCGTO_i<1 ):
                continue
            array=numpy.zeros([nKpoints,nCGTO_i,nCGTO_1],dtype=numpy.complex128)
            for kp in range(nKpoints):
                for j2 in range(nCGTOb_Array[Icol]):
                    for k1 in range(nCGTO_1):
                        array[kp][j2][k1] = dbuf[Icol][kp][j2][k1][0] + 1j*dbuf[Icol][kp][j2][k1][1]
            zbuf.append(array)
        ## print("calc_pbc_overlaps:direct IO:"+str(zbuf[0][0][0][0]))
    
    if(both):
        leL=len(zbuf);leR=len(zbuf_refr);le=min(leL,leR)
        devs=[];maxdev=-1;
        for J in range(le):
            dev1=aNmaxdiff(zbuf[J],zbuf_refr[J]);
            devs.append(dev1);maxdev=max(maxdev,dev1)
        printout("checking diff..."+str(devs),fpath="gto_ps_pp_int01_py.log",dtme=True,Threads=[0],Append=True)
        assert maxdev<1.0e-6,""  #XXX XXX

        fdDBG=open("comp_pbc_overlaps.dat","a")
        print("maxdev:%e "%(maxdev)+str(devs),file=fdDBG)                        
        fdDBG.close();
        #readwrite_xbuf('W',"comp_pbc_overlaps_LHS.dat",data=zbuf)        2021.08.02 commented out..
        #readwrite_xbuf('W',"comp_pbc_overlaps_RHS.dat",data=zbuf_refr)   2021.08.02 commented out..
        
    zbuf=modify_normalization(zbuf,nCGTOb_Array,nKpoints,nCGTO_1,n_cols, IZnuc_full,distinct_IZnuc_B,ell_B,alph_B,Nsh_B)

    if(Zbuf_Refr is not None):
        devs_1=[ aNmaxdiff(Zbuf_Refr[J],zbuf[J]) for J in range(le) ]
        printout("CompToRefr..."+str(devs_1),fpath="gto_ps_pp_int01_py.log",dtme=True,Threads=[0],Append=True)
    return zbuf

def with_df_get_pp(mydf,A_over_c,kpts=None,rttddft=None,details=None):
    import time
    from .update_dict import printout_dict,update_dict
    ## Copy of pyscf.pbc.df.aft.py ---
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    idbgng=2 # XXX XXX
    diff_ZF_to_fF=False

    # fncnme|fncdepth|Wctm000|Wctm010|N_call|Dic_timing|dic1_timing
    fncnme="with_df_get_pp";fncdepth=3   ## fncdepth ~ SCF:0 vhf:1 get_j:2 XXX XXX
    Wctm000=time.perf_counter();Wctm010=Wctm000;dic1_timing={}                    # XXX XXX
    Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)     # XXX XXX
    if( Dic_timing is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});                     # XXX XXX
        Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None) # XXX XXX

    N_call=rttddft_common.Countup("with_df_get_pp");
    n_call=N_call
###    n_call=rttddft_common.Countup("with_df_get_pp")
    cell = mydf.cell
    if kpts is None:
        assert False,"please set kpts explicitly..";
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    dbgng_vloc=True
    vloc1=None;vloc2=None
    if( rttddft is not None):
        if( hasattr(rttddft,"_fixednucleiapprox")):
            if(rttddft._fixednucleiapprox is not None):
                if( rttddft._fixednucleiapprox ):
                    if( rttddft.vloc1_ is None ):
                        rttddft.vloc1_ = get_pp_loc_part1(mydf, kpts_lst)
                        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
                        update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part1",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
                    if( rttddft.vloc2_ is None ):
                        rttddft.vloc2_ = pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
                        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
                        update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part2",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
                    vloc1=arrayclone( rttddft.vloc1_ )
                    vloc2=arrayclone( rttddft.vloc2_ )
                    Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
                    update_dict(fncnme,dic1_timing,Dic_timing,"arrayclone",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
                    if( dbgng_vloc ):
                        if( N_call == 2 or N_call == 8 or N_call==20 or N_call%50==0 ):
                            vloc1ref = get_pp_loc_part1(mydf, kpts_lst)
                            Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
                            update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part1dbg",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
                            Wctm_vloc1=Wctm010-Wctm020
                            vloc2ref = pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
                            Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
                            update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part2dbg",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
                            Wctm_vloc2=Wctm010-Wctm020
                            dev1=aNmaxdiff( vloc1, vloc1ref )
                            dev2=aNmaxdiff( vloc2, vloc2ref )
                            printout("vloc:maxdiff:%e %e %f %f"%(dev1,dev2,Wctm_vloc1,Wctm_vloc2),fnme_format="Save_vloc_%02d.log",Append=True)
                            assert dev1<1.0e-10 and dev2<1.0e-10,"vloc1 or vloc2 deviates:see Save_vloc_xx.log"
    if( vloc1 is None ):
        vloc1 = get_pp_loc_part1(mydf, kpts_lst)
        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
        update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part1",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
    if( vloc2 is None ):
        vloc2 = pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
        update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part2",Wctm010-Wctm020,depth=fncdepth) # XXX XXX

    vpp_org=None;vpp_ZF=None
    if( (idbgng > 0) and (n_call==1 or n_call==20 or n_call==200) ):
        vpp_org = pseudo.pp_int.get_pp_nl(cell, kpts_lst)
        fakecell, hl_blocks = fake_cell_vnl(cell)
        ppnl_half_o = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
        update_dict(fncnme,dic1_timing,Dic_timing,"DBG_get_pp_nl_org",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
        
        zerovec=[ 0.0, 0.0, 0.0]
        get_ppnl_half=[]
        vpp_ZF= get_pp_nl01(cell, zerovec, kpts_lst, comp_to_singlethread=True, get_ppnl_half=get_ppnl_half)
        ppnl_half_ZF= get_ppnl_half[0] 
        diff_vpp = max( abs( numpy.ravel( vpp_org ) - numpy.ravel( vpp_ZF ) ))

        diff_ppnl_half = -1; le=len(ppnl_half_ZF)
        for I in range(le):
            if( len(ppnl_half_ZF[I])==0 or len(ppnl_half_o[I]) ):
                fdOU=open("ppnl_half.log","a");
                print("#skipping %d %s %s"%(I,str(numpy.shape(ppnl_half_ZF[I])),str(numpy.shape(ppnl_half_o))),file=fdOU)
                fdOU.close();continue
            diff1 = max( abs( numpy.ravel( ppnl_half_ZF[I] ) - numpy.ravel( ppnl_half_o[I] ) ) )
            diff_ppnl_half = max( diff_ppnl_half, diff1 )
        fdOU=open("ppnl_half.log","a");
        for fd in [ fdOU, sys.stdout ]: 
            print("#diff_ppnl_half:%e"%(diff_ppnl_half), file=fd)
            print("#diff_vpp :%e"%(diff_vpp), file=fd)
        fdOU.close()
        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
        update_dict(fncnme,dic1_timing,Dic_timing,"DBG_get_pp_nl_ZF",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
        if(idbgng>1):
            dev=-1
            for kp in range(nkpts):
                diff=print_a2maxdiff(vpp_org[kp],vpp_ZF[kp],
                        "validate_get_pp_nl01_"+str(rttddft_common.get_job(True))+".rcd",
                        Append=(kp>0 or n_call>1),
                        description="ZF pp (lhs:mine,rhs:pySCForg) n_call=%d kp=%02d/%2d"%(n_call,kp,nkpts))
                dev=max(diff,dev)
        else:
            dev=aNmaxdiff(vpp_org,vpp_ZF,comment="vpp",verbose=True,title="vpp_ZF")
        assert (dev<1.0e-6),"diff:||vpp_org-vpp_ZF||=%e"%(dev)
    ### print("vloc1;",end="");print( numpy.shape(vloc1))  ## vloc1;(8, 56, 56)
    ### print("vloc2;",end="");print( numpy.shape(vloc2))  ## vloc1;(8, 56, 56)
    ### print("vpp_org;",end="");print( numpy.shape(vpp_org)) ## (8, 56, 56)

    vpp = get_pp_nl01(cell, A_over_c, kpts_lst)
    Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_nl01",Wctm010-Wctm020,depth=fncdepth) # XXX XXX

    if( details is not None ):
        details.update({"vloc1":arrayclone(vloc1),"vloc2":arrayclone(vloc2),
                        "vpp":arrayclone(vpp)})
        
    if( diff_ZF_to_fF and (vpp_org is not None) ):
        abs_AoverC=numpy.sqrt( A_over_c[0]**2 + A_over_c[1]**2 + A_over_c[2]**2 )
        if(abs_AoverC>1.0e-6):
            n1=rttddft_common.Countup("with_df_get_pp.diff_ZF_to_fF")
            if(n1==1):
                diff=aNmaxdiff(vpp,vpp_org,comment="vpp/vpp_org",verbose=True,title="vpp")
                print_Hmatrices("vpp",vpp,vpp_org,fpath="vpp_and_vppZF.dat")
        
    for k in range(nkpts):
        vpp[k] += vloc1[k] + vloc2[k]

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    Wctm010=time.perf_counter()  # XXX XXX
    printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth) # XXX XXX

    return vpp

class get_pp_nl01_static:
    count=0
    Dict_={}
    prtout_calc_pbc_overlaps_retv_=False       ## XXX XXX
    AoverC_prtout_calc_pbc_overlaps_retv_=None ## XXX XXX
    fnme_prtout_calc_pbc_overlaps_retv_=None   ## XXX XXX
    @staticmethod
    def Countup(key,inc=True):
        if( key not in get_pp_nl01_static.Dict_ ):
            get_pp_nl01_static.Dict_.update({key:0});
        if( inc ):
            get_pp_nl01_static.Dict_[key]+=1
        return get_pp_nl01_static.Dict_[key]

def get_pp_nl01(cell, A_over_c, kpts=None, comp_to_singlethread=False, get_ppnl_half=None):
    ### fd01=open("get_pp_nl01_dbg.log","w");print("start:"+str( datetime.datetime.now() ),file=fd01);fd01.close()
    ### test_zalloc(5,3,4,1,prtout=True)
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();

    Logger.write_once(None,"get_pp_nl01:calc_pp_overlaps","calculate pp for A="+str(A_over_c))
    if kpts is None:
        assert False,"please set kpts explicitly..";
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)   ## pyscf.pbc.gto.pseudo.pp_int

    #if( hasattr(fakecell,"atom") ):
    #    pyscf_common.Check_cellRnuc(fakecell,"get_pp_nl01.fakecell")

    force_singlethread=None  ## default ...
    if( not MPIutils01.Multithreading() ):
        ## MPIutils01.DebugLog("#calc_pp_overlaps:singlethread");
        comp_to_singlethread = False
        force_singlethread = True
        MPIutils01.DebugTrace("get_pp_nl01","get_pp_nl01:SINGLETHREAD")
    dbgng=1;strict=1;
    if(dbgng>0):
        get_pp_nl01_static.count=get_pp_nl01_static.count+1
        if( get_pp_nl01_static.count==1 or get_pp_nl01_static.count==20 or comp_to_singlethread):
            ppnl_half_o = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
###             fd001=open("ppnl_half_o_01.log","w");
###             print("#ppnl_half_o:"+str(datetime.datetime.now()),file=fd001);
###             ### print(ppnl_half_o,file=fd001);### fd001.close()
###             print( len(ppnl_half_o),file=fd001 );
###             for I in range( len(ppnl_half_o) ):
###                 print(numpy.shape(ppnl_half_o[I]),file=fd001)
###                 ## (1, 72, 180), (1, 16, 180), (1, 4, 180) ... so this cannot be treated as ndarray
###                 ## whereas  A[0],A[1],A[2] are ndarrays
###             fd001.close()
            
###             datf01="ppnl_half.dat"
###             if( not suppress_prtout() ):
###                 fd=futils.fopen(datf01,("w" if(get_pp_nl01_static.count==1) else "a")) ## OK
###                 for col in ppnl_half_o:
###                     line=""
###                     print("#"+str(numpy.shape(col)),file=fd)
###                     for y in numpy.ravel(col):
###                        line+=str(y)
###                     print(line,file=fd)
###                 futils.fclose(fd);

            ppnl_half_ZF = calc_pp_overlaps(cell,[0,0,0],kpts_lst,dbgng=True,Zbuf_Refr=ppnl_half_o)    ## New subroutine
            comp_datf="ppnl_half_comp_n%06d.dat"%(get_pp_nl01_static.count); maxdev=-1;Ndev=0
            if( not suppress_prtout() ):
                fd=futils.fopen(comp_datf,"w") ## OK
                maxdev=-1;vals=[];Ndev=0
                for Icol in range(3):
                    Ndim2=numpy.shape(ppnl_half_o[Icol]); ### print("#%d:shape:"%(Icol),end="");print(Ndim2)
                    if(Ndim2[0]<1):
                        continue
                    Ndim1=numpy.shape(ppnl_half_ZF[Icol]);### print("#%d:shape:"%(Icol),end="");print(Ndim1)
                    Ndim=[ min( Ndim1[0], Ndim2[0]), min( Ndim1[1], Ndim2[1]), min( Ndim1[2], Ndim2[2]) ]
                    for Kp in range( Ndim[0] ):
                        for j2 in range(Ndim[1]):
                            for k1 in range(Ndim[2]):
                                dev=abs( ppnl_half_ZF[Icol][Kp][j2][k1] - ppnl_half_o[Icol][Kp][j2][k1] )
                                if(dev>maxdev):
                                    maxdev=dev; vals=[ ppnl_half_ZF[Icol][Kp][j2][k1], ppnl_half_o[Icol][Kp][j2][k1] ];
                                if(dev<1.0e-6):
                                    print("##  ",end="",file=fd)
                                else:
                                    Ndev+=1
                                ratio=ppnl_half_ZF[Icol][Kp][j2][k1]/(1.0 if( abs(ppnl_half_o[Icol][Kp][j2][k1])<1.0e-20 ) else ppnl_half_o[Icol][Kp][j2][k1]); 
                                print("%d %d %d %d  %f+j%f  %f+j%f  %e  val/ref=%14.8f + j%14.8f"%(Icol,Kp,j2,k1,
                                      ppnl_half_ZF[Icol][Kp][j2][k1].real,   ppnl_half_ZF[Icol][Kp][j2][k1].imag,
                                      ppnl_half_o[Icol][Kp][j2][k1].real, ppnl_half_o[Icol][Kp][j2][k1].imag, dev,
                                      ratio.real, ratio.imag),file=fd)
                        fd2=futils.fopen("comp_vals.dat",("w" if(Kp==0 and Icol==0) else "a"))
                        print("%d %d %e  %f+j%f / %f+j%f"%(Icol,Kp,maxdev, vals[0].real,vals[0].imag, vals[1].real,vals[1].imag),file=fd2)
                        futils.fclose(fd2);
                        
                futils.fclose(fd);
                if( maxdev > 1.0e-6 or Ndev > 0 ):
                    assert False,"Check results %02d %e %d:check:"%(MPIrank,maxdev,Ndev) +comp_datf
                if( maxdev > 1.0e-7 or Ndev > 0 ):
                    if(strict>1):
                        assert False,"Check results %02d %e %d:check:"%(MPIrank,maxdev,Ndev)+comp_datf
                    else:
                        Logger.Warning("ppnl_half_ZF","ppnl_half_ZF:maxdev=%e:"%(maxdev)+str(vals))
            assert (maxdev<1.0e-6),"check"+comp_datf
    if(get_pp_nl01_static.count%20==1):
        printout("#calculating ppnl_half with Field:%f"%(math.sqrt(A_over_c[0]*A_over_c[0] + A_over_c[1]*A_over_c[1] + A_over_c[2]*A_over_c[2])))
    
    abs_A_over_c= numpy.sqrt( A_over_c[0]**2 + A_over_c[1]**2 + A_over_c[2]**2 )

    dbgng_mpi=False; ppnl_half_refr=None  ## 20210609MPIvalidation: we compare final results to those of with_df_get_pp_bf20210608 .. 
    if( dbgng_mpi or comp_to_singlethread ):
        n_count = rttddft_common.Countup("get_pp_nl01_testMPI")
        if(n_count==1 or comp_to_singlethread):
            ppnl_half_refr=calc_pp_overlaps(cell,A_over_c,kpts_lst,singlethread=True)
    ### printout("calling calc_pp_overlaps... to get ppnl_half (main)",fpath="gto_ps_pp_int01_py.log",Threads=[0],Append=True)
    ppnl_half = calc_pp_overlaps(cell,A_over_c,kpts_lst, singlethread=force_singlethread)
    if( force_singlethread ):
        print("#multithreading:$%02d:ppnl_half:SINGLETHREAD"%(MPIrank) )
    else:
        print("#multithreading:$%02d:ppnl_half:MULTITHREAD"%(MPIrank) )

    if( ppnl_half_refr is not None ):
        Ld=len(ppnl_half);dev=-1;diffs=[]
        for I in range(Ld):
            diffs.append( aNmaxdiff(ppnl_half_refr[I], ppnl_half[I]) )
        printout("ppnl_half:maxdiff:%e"%( max(diffs) ),fpath="ppnl_half.log",Append=True,Threads=[0])
        print("ppnl_half:maxdiff:%e"%( max(diffs) ) )
        assert max(diffs)<1.0e-7,"$%02d:|ppnl_half - ppnl_half_SINGLETHREAD|=%e"(MPIrank,diffs)


    if( get_ppnl_half is not None ):
        print("#cloning ppnl_half:",[ numpy.shape( buf ) for buf in ppnl_half ])
        get_ppnl_half.append( [ arrayclone(buf) for buf in ppnl_half ])
##ppnl_half:(8, 22, 56)
##ppnl_half:(8, 8, 56)
##ppnl_half:(0,)
#nao:56

    nao = cell.nao_nr() # 56 
    ### print("nao:",end="");print(nao)
    buf = numpy.empty((3*9*nao), dtype=numpy.complex128)

    # We set this equal to zeros in case hl_blocks loop is skipped
    # and ppnl is returned
    n_cum=0
    ppnl = numpy.zeros((nkpts,nao,nao), dtype=numpy.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        for ib, hl in enumerate(hl_blocks):
            l = fakecell.bas_angular(ib)
            if(k==0):
                n_cum+=(2*l+1)
                ### print("#ppShell:%d el=%d nmult=%d %d"%(ib,l,2*l+1,n_cum)) #ppShell:0 el=0 nmult=1 1
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ### print("hl_dim;",end="");print(hl_dim)  ## 2 or 1
            ilp = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.complex128, buffer=buf)
            for i in range(hl_dim):
                p0 = offset[i]
                ilp[i] = ppnl_half[i][k][p0:p0+nd]

                ### print("ilp_%d:"%(i),end="");print(ppnl_half[i][k][p0:p0+nd])
                ### print("ilp_%d:"%(i),end="");print(ilp[i]);

                offset[i] = p0 + nd
            ppnl[k] += numpy.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
        ### if(k==0):
        ###    print("offset_sum:%d %d %d"%(offset[0],offset[1],offset[2])) (8,0,0) 
    if abs(kpts_lst).sum() < 1e-9:  # gamma_point:
        if( abs_A_over_c < 1e-10 ):
            ppnl = ppnl.real
        else:
            # dum=absmax_imaginary_part(ppnl)
            printout("#finite A_over_c.. keep as a complex array: imag:%e"%(absmax_imaginary_part(ppnl)))
            # assert (dum<1.0e-10),""
    if kpts is None or numpy.shape(kpts) == (3,):
        ppnl = ppnl[0]
    return ppnl

def get_ncols(cell,IZnuc, distinct_IZnuc, Nsh_B=None,verbose=0,title=""):

    fakecell, hl_blocks = fake_cell_vnl(cell)
    if(verbose>1):
        printout("#get_ncols:"+title+":cell:"+str(cell))
        printout("#get_ncols:"+title+":fakecell:"+str(fakecell))
        printout("#get_ncols:"+title+":hl_blocks:"+str(hl_blocks))
    pre=[]
    for hl in hl_blocks:
        if(verbose>1):
            printout("#get_ncols:"+title+":hl_blocks:"+str(hl))
        nd=len(hl)
        pre.append(nd)
    ### print("#n_cols_pre:",end="");print(pre)
    nDa=len(distinct_IZnuc)
    assert nDa>0,"wrong nDa:"+str(distinct_IZnuc)
    ret=[]
    for iDa in range(nDa):
        ret.append(None)

    
    Nat=len(IZnuc)
    ksh_to_Iat=[]
    for Iat in range(Nat):
        IZ=IZnuc[Iat]
        for iDa in range(nDa):
            if(IZ==distinct_IZnuc[iDa]):
                break
        assert(distinct_IZnuc[iDa]==IZ),""
        nsh1=Nsh_B[iDa]
        for k in range(nsh1):
            ksh_to_Iat.append(Iat)

    if(verbose>0):
        printout("#get_ncols:"+title+":ksh_to_Iat:"+str(ksh_to_Iat));
    if(verbose>0):
        printout("#get_ncols:"+title+":pre:"+str(pre));
#
# 20210608: Nat is not necs. full size...     
#    assert len(ksh_to_Iat)==len(pre),""
#
    le=len(ksh_to_Iat)
    append_to_logfile=False
    for Iat in range(Nat):
        IZ=IZnuc[Iat]
        for iDa in range(nDa):
            if(IZ==distinct_IZnuc[iDa]):
                break
        assert(distinct_IZnuc[iDa]==IZ),""
        if( ret[iDa] is None):
            ret[iDa]=[];kA=[];
            for k in range(le):
                if( ksh_to_Iat[k] != Iat ):
                    continue
                else:
                    ret[iDa].append( pre[k] );kA.append(k)
                    #if(verbose>1):
                    #    print("#get_ncols:"+title+":ret:"+str(pre[k]));
            if( not suppress_prtout() ):
                fd=futils.fopen("gen_ncols.log",("a" if(append_to_logfile) else "w"));append_to_logfile=True ## OK
                print("#%d: %d k={"%(Iat,iDa)+str(kA)+"}  ret:"+str(ret[iDa]),file=fd);
                futils.fclose(fd);
        else:
            refr=[];kA=[];
            for k in range(le):
                if( ksh_to_Iat[k] != Iat ):
                    continue
                else:
                    refr.append( pre[k] );kA.append(k)
            assert  i1eqb(refr, ret[iDa], verbose=False,title="gen_ncols.001"),""
    ### print("#n_cols_ret:",end="");print(ret)
    # NshSUM=sum(Nsh_B)
    # leRET=len(ret)
    # assert leRET==NshSUM,"len:%d / NshSUM:%d ("%(leRET,NshSUM) + str(Nsh_B)
    return ret;
def test_pp(cell):
    fakecell, hl_blocks = fake_cell_vnl(cell)
    printout("#fakecell:"+str(fakecell))
    
    n=0
    for x in vars(fakecell):
        y=getattr(fakecell,x,None)
        printout("#fakecell_%s"%(str(x))+str(y))
        n+=1
    n=0
    for hl in hl_blocks:
        printout("#hl_blocks_%d:"%(n)+str(hl))
        n+=1
#fakecell_ke_cutoff None
#fakecell_pseudo gth-pade
#fakecell_dimension 3
#fakecell_low_dim_ft_type None
#fakecell__mesh [969 969 625]
#fakecell__mesh_from_build True
#fakecell__ew_eta 2.7513252105444863
#fakecell__ew_from_build True
#fakecell__ew_cut 2.5435279375048974
#fakecell__rcut 20.291655858262136
#fakecell__rcut_from_build True
#fakecell_0:[[  6.9257397    5.62514728]
#            [  5.62514728 -67.13819583]]
#fakecell_1:[[   5.0790865    15.27691303]
#            [  15.27691303 -207.97142959]]
#fakecell_2:[[-9.12589591]]
#fakecell_3:[[  6.9257397    5.62514728]
#            [  5.62514728 -67.13819583]]
#fakecell_4:[[   5.0790865    15.27691303]
#            [  15.27691303 -207.97142959]]
#fakecell_5:[[-9.12589591]]
#fakecell_6:[[18.26691718]]
#fakecell_7:[[18.26691718]]
#fakecell_8:[[18.26691718]]
#fakecell_9:[[18.26691718]]
def mpi_prtout_zbuf(fnme_format,zbuf,description="",Append=False):
    comm=MPI.COMM_WORLD;MPIrank=comm.Get_rank();MPIsize=comm.Get_size()
    fnme=fnme_format%(MPIrank)
    fd=open(fnme,('a' if(Append) else 'w'))
    Ld=len(zbuf)
    print("##"+description,file=fd);
    for I in range(Ld):
        cbuf=zbuf[I];ndim=numpy.shape(cbuf)
        print("#%03d:"%(I) +str(ndim))
        if(len(ndim)==2):
            for J in range(ndim[0]):
                string=""
                for K in range(ndim[1]):
                    string+="%14.6f %14.6f      "%(cbuf[J][K].real, cbuf[J][K].imag)
                print(string,file=fd)
        elif( len(ndim)==3 ):
            for J in range(ndim[0]):
                print("#%02d"%(J),file=fd)
                for K in range(ndim[1]):
                    string=""
                    for L in range(ndim[2]):
                        string+="%14.6f %14.6f      "%(cbuf[J][K][L].real, cbuf[J][K][L].imag)
                    print(string,file=fd)
                print("\n\n",file=fd)
        else:
            cbuf=numpy.ravel(cbuf);le=len(cbuf)
            string=""
            for K in range(le):
                string+="%14.6f %14.6f      "%(cbuf[K].real, cbuf[K].imag)
            print(string,file=fd)
    fd.close()

def uniquify_i1a(org):
    le=len(org)
    ret=[]
    for I in range(le):
        if( org[I] in ret ):
            continue
        ret.append( org[I] )
    return ret


def prtout_bsetx(rttddft,filename=None):
    kpts=( None if(not rttddft._pbc) else rttddft.kpts)
    mol_or_cell = (rttddft.mol if(not rttddft._pbc) else rttddft.cell)
    Rnuc_au,Sy=parse_xyzstring( mol_or_cell.atom, unit_BorA='B')
    A_over_c=[0.0, 0.0, 0.0]
    prtout_bset1(rttddft, mol_or_cell, A_over_c, kpts, Rnuc_au, Sy, filename=filename)

def prtout_bset1(rttddft, mol_or_cell, A_over_c, kpts, RnucAU_sub, Sy_sub, filename=None):
    #libnme=pathlib.Path().absolute() / "libpp01.so"
    #testlib=ctypes.CDLL(libnme)
    testlib = lib.load_library("libpp01")

    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();

    if( kpts is None ):
        kpts=[ 0.0, 0.0, 0.0]

    kvectors=numpy.reshape( kpts, (-1,3))
    nKpoints=len(kvectors)
    kvecs=[]
    for v in numpy.ravel(kvectors):
        kvecs.append(v)
    kvectors_1=( ctypes.c_double*(nKpoints*3) )(*kvecs)

    Nat_sub=len(Sy_sub)
    IZnuc_sub=[];
    for i in range(Nat_sub):
        IZnuc_sub.append( atomicsymbol_to_atomicnumber( Sy_sub[i] ) )

    bset_info=gen_bset_info(mol_or_cell,IZnuc_sub)
    
    strbset=None
    if( isinstance(mol_or_cell.basis,str) ):
        strbset=(mol_or_cell.basis).replace('(','_').replace(')','').replace(',','')
    elif( isinstance(mol_or_cell.basis, dict) ):
        delimiter=""
        for elmt in mol_or_cell.basis:
            strbset+=delimiter+elmt+ str( mol_or_cell.basis[elmt] ).replace('(','_').replace(')','').replace(',','')
            delimiter="_"
    
    if( hasattr(mol_or_cell,"exp_to_discard")):
        if( mol_or_cell.exp_to_discard is not None ):
            filtered =filter_by_exponent(bset_info, mol_or_cell.exp_to_discard )
            bset_info=filtered

    ## 
    spdm=3
    alph,cofs,Ell,distinctIZnuc_sub,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_info(bset_info)
    nDa_sub=len(bset_info)
    BravisVectors= ( numpy.array([0.0 for k in range(9)]) if( not rttddft._pbc ) else \
                     numpy.array( mol_or_cell.a )/physicalconstants.BOHRinANGS )
    BravisVectors_1=( ctypes.c_double*(spdm*3) )( *numpy.ravel(BravisVectors) )
    Vectorfield_1=( ctypes.c_double*(3) )( *A_over_c )
    nKpoints=len(kpts);### print(kpts);print(nKpoints);### assert False,""
    
    RnucAU_sub =list(numpy.ravel(RnucAU_sub))
    Rnuc_1 =( ctypes.c_double*(3*Nat_sub) )(*RnucAU_sub)
    IZnuc_1=( ctypes.c_int*(Nat_sub) )( *IZnuc_sub )
    Nsh_1  =( ctypes.c_int*(nDa_sub) )( *Nsh ) 
    Ell_1  =( ctypes.c_int*(Nsh_sum) )( *Ell )
    Npgto_1=( ctypes.c_int*(Nsh_sum) )(*Npgto)
    distinct_IZnuc_1=( ctypes.c_int*(nDa_sub) )( *distinctIZnuc_sub )
    alph_1=( ctypes.c_double*(Npgto_sum) )(*alph)
    cofs_1=( ctypes.c_double*(Npgto_sum) )(*cofs) 
    nCGTO_1=count_nCGTO(IZnuc_sub,distinctIZnuc_sub,Nsh,Ell); nCGTO_1=int(nCGTO_1)
    Nsh_sum=sum( Nsh ); nPGTO_sum=sum(Npgto)

    fnme=filename
    if( fnme is None ):
        fnme=str(rttddft_common.get_job(True))+"_basisset.dat"
        if( strbset is not None ):
            if( rttddft_common.Params_get("name") is not None ):
                fnme=rttddft_common.Params_get("name")+strbset+"_basisset.dat"
    write_filex(fnme,
        Natm=Nat_sub, Rnuc=RnucAU_sub, IZnuc=IZnuc_sub, nDa=nDa_sub, distinctIZnuc=distinctIZnuc_sub,
        Nsh=Nsh,   Nsh_sum=Nsh_sum,    ell=Ell,   nPGTO=Npgto,
        nPGTO_sum=nPGTO_sum, alph=alph, cofs=cofs, spdm=3,
        BravisVectors=BravisVectors,    Vectorfield=A_over_c, nKpoints=nKpoints,
        kvectors=kvectors,  nCGTO=nCGTO_1);
    testlib.test001( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                   Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                   spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                   kvectors_1,   nCGTO_1);

## returns Complex [spdm][nKpoints][nAO][nAO] matrix
## corresponding to < \mu | x V_NL(x,x') - V_NL(x,x') x'| \nu >
## 
def with_df_get_ppxx(mydf,A_over_c,kpts=None,rttddft=None):
    import time
    from .update_dict import printout_dict,update_dict

    fncnme="with_df_get_ppxx";fncdepth=3   ## fncdepth ~ SCF:0 vhf:1 get_j:2 XXX XXX
    Wctm000=time.perf_counter();Wctm010=Wctm000;dic1_timing={}                    # XXX XXX
    Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)     # XXX XXX
    if( Dic_timing is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});                     # XXX XXX
        Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None) # XXX XXX
    N_call=rttddft_common.Countup("with_df_get_ppxx");
    cell = mydf.cell
    if kpts is None:
        assert False,"please set kpts explicitly..";
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    ## in any sense this is <\mu | xV(x,x') - V(x,x')x' |\nu>
    ## [3][nAO][nAO] 
    vpp = get_pp_nl01xx(cell, A_over_c, kpts_lst)
    ### Dbgtrace("#with_df_get_ppxx:get_pp_nl01xx returns");
    Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_nl01xx",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
    printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth) # XXX XXX
    ### Dbgtrace("#with_df_get_ppxx:END");
    return vpp
def Dbgtrace(msg,flush=None):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    msg+="     \t\t "+str( datetime.datetime.now() )
    print("#gto_ps_pp_int01.Dbgtrace:#$%02d/%d:"%(MPIrank,MPIsize)+str(msg),flush=True);
    fd1=open("gto_ps_pp_int01_Dbgtrace%02d.log"%(MPIrank),"a");print(msg,file=fd1);fd1.close()

def test_zalloc(N1,N2,N3,N4,prtout=True):
    ### XXX XXX TODO: this stupid test takes very very long time 
    ###               it looks that the memory allocated in c library is not fully released by python -del- method
    ###       
    #libname=pathlib.Path().absolute() / "libpp01.so"    
    #c_lib = ctypes.CDLL(libname)
    c_lib = lib.load_library("libpp01")

    Ndim_retv=[N1,N2,N3,2]
    zbuf=None
    for I in range(N4):
        c_lib.test_alloc.restype= ndpointer(dtype=ctypes.c_double,shape=Ndim_retv )
        dbuf=c_lib.test_alloc(N1,N2,N3)
        ### print(numpy.shape(dbuf))
        zbuf=numpy.zeros([N1,N2,N3],dtype=numpy.complex128)
        for i in range(N1):
            for j in range(N2):
                for k in range(N3):
                    zbuf[i][j][k]= dbuf[i][j][k][0] + 1j*dbuf[i][j][k][1]
        ##if(I%2==0):
        ##    c_lib.free_zbuf(dbuf); ## ctypes.ArgumentError: argument 1: <class 'TypeError'>: Don't know how to convert parameter 1
        ##else:
        ##    c_lib.free_buf(dbuf);  ## ctypes.ArgumentError: argument 1: <class 'TypeError'>: Don't know how to convert parameter 1
        ### dbuf=None
        if(I<20 or (I<1000 and I%20==0) or (I<10000 and I%200==0) or I%2000==0 ):
            print("#test_alloc:%d.."%(I))
        ### del zbuf;zbuf=None
    if(prtout):
        fd1=open("test_zalloc.log","w")
        for i in range(N1):
            print("#%02d"%(i),file=fd1)
            for j in range(N2):
                strbuf=""
                for k in range(N3):
                    strbuf+="%12.6f %12.6f     "%(zbuf[i][j][k].real,zbuf[i][j][k].imag)
                print(strbuf,file=fd1)
        fd1.close()

    print("#test_alloc:DONE:%d   %d,%d,%d"%(N4, N1,N2,N3))
            
def get_pp_nl01xx(cell, A_over_c, kpts=None):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    if kpts is None:
        assert False,"please set kpts explicitly..";
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    fakecell, hl_blocks = fake_cell_vnl(cell)   ## pyscf.pbc.gto.pseudo.pp_int
    ## this invokes lib.calc_pbc_overlapsx and possiblly collects distributed results
    ## [spdm+1,3,nKpoints,nCGTO_2,nCGTO_1,2]
    ### Dbgtrace("#get_pp_nl01xx:calc_pp_overlapsxx start");
    ppnl_half = calc_pp_overlapsxx(cell,A_over_c,kpts_lst)  # [1+3][N_col][nkpt][nBS2][nBS1]

    ### for dir in range(4):  ## xxx xxx xxx 
    ###     readwrite_xbuf('W',"calc_pp_overlapsxx_ppnl_half%02d.dat"%(dir), data=ppnl_half[dir])## xxx xxx xxx 
    
    nao = cell.nao_nr()
    buf1 = numpy.empty((3*9*nao), dtype=numpy.complex128)
    buf2 = numpy.empty((3*9*nao), dtype=numpy.complex128)

    dbgng=False; ppnl_DBG=None
    if(dbgng):
        ppnl_DBG=numpy.zeros((3,nkpts,nao,nao), dtype=numpy.complex128)
    Dbgng2=1
    ppnl = numpy.zeros((3,nkpts,nao,nao), dtype=numpy.complex128)
    for dir in range(3):
        ### Dbgtrace("#get_pp_nl01xx:calc_pp_overlapsxx dir:%d"%(dir));
        for k, kpt in enumerate(kpts_lst):
            offset = [0] * 3
            for ib, hl in enumerate(hl_blocks):
                ### Dbgtrace("#get_pp_nl01xx:calc_pp_overlapsxx dir,ib:%d,%d"%(dir,ib));
                l = fakecell.bas_angular(ib)
                nd = 2 * l + 1
                hl_dim = hl.shape[0]
                ilpO = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.complex128, buffer=buf1)
                ilpX = numpy.ndarray((hl_dim,nd,nao), dtype=numpy.complex128, buffer=buf2)
                for i in range(hl_dim):
                    p0 = offset[i]
                    ilpO[i] = ppnl_half[0][i][k][p0:p0+nd]    # [1+3][N_col][nkpt][nBS2][nBS1] -> [nd \in nBS2][nBS1]
                    ilpX[i] = ppnl_half[1+dir][i][k][p0:p0+nd]
                    offset[i] = p0 + nd
                XhO=None;OhX=None
                if(Dbgng2>1 or (Dbgng2==1 and k==0) ):
                    XhO= numpy.einsum('ilp,ij,jlq->pq', ilpX.conj(), hl, ilpO)
                    OhX= numpy.einsum('ilp,ij,jlq->pq', ilpO.conj(), hl, ilpX)
                    ## originally they are ppnlxx.log 
                    printout_zmatrices(rttddft_common.get_job(True)+"_xVandVx.log",XhO,OhX,index=[dir,k],fopen=False) ## (dir==0 and ib==0 and i==0))
                ppnl[dir][k] += numpy.einsum('ilp,ij,jlq->pq', ilpX.conj(), hl, ilpO)\
                                - numpy.einsum('ilp,ij,jlq->pq', ilpO.conj(), hl, ilpX)
### this is anti-Hermitian Array
                
#DbgCode: see gto_ps_pp_int01_20220214DBG.py
#         results : DBG_einsum.log  ppnlx_%d_%03d_DIAG/OFD.log
#         results show XhO or OhX summations are both non-zero but difference is ZERO..                                                                                                                      ilpX[i][l][0]                             ilpO[i][l][4]
#000 004:    0   0   0       -0.000000     0.000000       -0.000000     0.000000         -0.000000     0.000000             0.000000    -0.000000     -14.277462       0.000000     0.000000
#000 004:    0   0   1       -0.000000    -0.000000       -0.000000     0.000000         -0.000000     0.000000             0.000000    -0.000000     -14.277462       0.000000     0.000000
#000 004:    0   0   2        0.000000     0.000005        0.000000     0.000005          0.000000     0.000005             0.029784    -0.000000     -14.277462      -0.000000    -0.000013
#
#                                                                                                                           ilpX[i][l][4]                              ilpO[i][l][0]
#004 000:    0   0   0       -0.000000    -0.000000       -0.000000    -0.000000         -0.000000    -0.000000             0.000000    -0.000000     -14.277462       0.000000     0.000000
#004 000:    0   0   1       -0.000000     0.000000       -0.000000    -0.000000         -0.000000    -0.000000             0.000000    -0.000000     -14.277462       0.000000     0.000000
#004 000:    0   0   2        0.000000    -0.000005        0.000000    -0.000005          0.000000    -0.000005            -0.000000     0.000013     -14.277462       0.029784     0.000000


    if kpts is None or numpy.shape(kpts) == (3,):
        ppnl = ppnl[0]

    if(dbgng):
        n_call=rttddft_common.Countup("get_pp_nl01xx")
        for dir in range(3):
            fnm1="get_pp_nl01xx_%02d_ppnl.log"%(dir)
            fd1=open(fnm1,('w' if(n_call==1) else 'a'))
            fd2=open("get_pp_nl01xx_%02d_ilp.log"%(dir+1),('w' if(n_call==1) else 'a'))
            for k, kpt in enumerate(kpts_lst):
                print("#%06d %03d   "%(n_call,k)+str(datetime.datetime.now()),file=fd1)
                print("#%06d %03d   "%(n_call,k)+str(datetime.datetime.now()),file=fd2)
                for jj in range(nao):
                    strbuf="";
                    for kk in range(nao):
                        strbuf+="%14.8f %14.8f       "%( ppnl_DBG[dir][k][jj][kk].real, ppnl_DBG[dir][k][jj][kk].imag )
                    print(strbuf,file=fd1)
                print("\n\n",file=fd1)

                offset = [0] * 3
                for ib, hl in enumerate(hl_blocks):
                    str2="";str3=""
                    l = fakecell.bas_angular(ib)
                    nd = 2 * l + 1
                    hl_dim = hl.shape[0]
                    for i in range(hl_dim):
                        p0 = offset[i]
                        str2+= str( ppnl_half[0][i][k][p0:p0+nd] )+" \t\t "
                        str3+= str( ppnl_half[1+dir][i][k][p0:p0+nd] )+" \t\t "
                        offset[i] = p0 + nd
                    print(str2,file=fd2);
                    print(str3+"\n",file=fd2);
            fd1.close();fd2.close()
            ### os.system("fopen "+fnm1);
    ### Dbgtrace("#get_pp_nl01xx:calc_pp_overlapsxx dir:%d"%(dir));
    return ppnl

def calc_pp_overlapsxx(cell,A_over_c,kpts, singlethread=None):
    import math
    import time
    comm=MPI.COMM_WORLD
    kvectors=numpy.reshape(kpts,(-1,3))
    nKpoints=len(kvectors)
    MPIrank=comm.Get_rank();MPIsize=comm.Get_size()
    Rnuc_au,Sy=parse_xyzstring(cell.atom,unit_BorA='B')
    Nat_too_small=3
    Nat=len(Sy)
    if( singlethread is None ):
        singlethread = ( MPIsize < 2 or Nat <= Nat_too_small )
    if( MPIsize < 2 ):
        singlethread = True
    fncnme="calc_pp_overlapsxx";fncdepth=4      ## XXX XXX  fncdepth ~ SCF:0 vhf:1 get_j:2
    Wctm000=time.perf_counter();Wctm010=Wctm000;N_call=rttddft_common.Countup(fncnme);dic1_timing={} ## XXX XXX
    Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)                      ## XXX XXX
    if( Dic_timing is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});
        Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)
    if( singlethread ):
        retv=calc_pp_overlapsxx1(cell, A_over_c, kpts, Rnuc_au, Sy, Rnuc_au, Sy)
        ### Dbgtrace(".calc_pp_overlapsxx:singlethread returns.... ",flush=True);
        Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
        update_dict(fncnme,dic1_timing,Dic_timing,"singlethread",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
        Wctm010=time.perf_counter()  # XXX XXX
        printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth) # XXX XXX
        ### Dbgtrace(".calc_pp_overlapsxx:singlethread END",flush=True);
        return retv
    Nat_per_thread=int( math.ceil( Nat / float(MPIsize) ) )
    iatSTT= min( MPIrank*Nat_per_thread, Nat)
    iatUPL= min( (MPIrank+1)*Nat_per_thread, Nat) 
    ppnl_half_sub=None; ncol_ppnl_half_sub=0
    if( iatSTT < iatUPL ):
        Rnuc_sub=Rnuc_au[iatSTT:iatUPL]
        Sy_sub=Sy[iatSTT:iatUPL]
        ppnl_half_sub=calc_pp_overlapsxx1(cell, A_over_c, kpts, Rnuc_sub, Sy_sub, Rnuc_au, Sy )
        ### Dbgtrace(".calc_pp_overlapsxx:thread %d returns.... "%(MPIrank),flush=True);
        ncol_ppnl_half_sub = len( ppnl_half_sub[0] )
    Wctm020=Wctm010;Wctm010=time.perf_counter() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"multithread_ppnl",Wctm010-Wctm020,depth=fncdepth) # XXX XXX

    ## We have to recalculate matrix size(s) here
    IZnuc=[ atomicsymbol_to_atomicnumber( Sy[i] ) for i in range(Nat) ]
    alph,cofs,Ell,distinct_IZnuc,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_infox(cell,Sy)    

    ## nCGTO sizes
    nCGTO1_A=[]
    for r in range(MPIsize):
        iatSTT= min( r*Nat_per_thread, Nat)
        iatUPL= min( (r+1)*Nat_per_thread, Nat) 
        if( iatSTT < iatUPL ):
            IZnuc_sub=IZnuc[iatSTT:iatUPL]
            ## this applies IZnuc_sub 
            ncgto=count_nCGTO(IZnuc_sub,distinct_IZnuc,Nsh,Ell)
            nCGTO1_A.append(ncgto)
        else:
            nCGTO1_A.append(0)
    nCGTO1_sum=sum(nCGTO1_A)
    nCGTO2_A=numpy.zeros([3], dtype='i')
    ### Dbgtrace(".calc_pp_overlapsxx:collecting.... "+str(nCGTO1_A),flush=True);
    for Icol in range(3):
        if( (ppnl_half_sub is None) or (Icol >=ncol_ppnl_half_sub) ):
            nCGTO2_A[Icol]=0
        else:
            ndim=numpy.shape( ppnl_half_sub[0][Icol] )
            nCGTO2_A[Icol]=ndim[1]  ## [0]:nKpoints [1]:nCGTO2 [2]:nCGTO1
    mpi_Bcast("calc_pp_overlapsxx.nCGTO2",nCGTO2_A,root=0)

    ## make ppnl_half [spdm+1,3,nKpoints,nCGTO_2,nCGTO_1]
    ## out of chunks  [spdm+1,3,nKpoints,nCGTO_2,nCGTO1_A[r]]
    ppnl_half=[ [], [], [], [] ]
    for dir in range(4):
        for Icol in range(3):
            if( nCGTO2_A[Icol] < 1 ):
                continue
            zbuf1=numpy.zeros( [ nKpoints, nCGTO2_A[Icol],nCGTO1_sum ],dtype=numpy.complex128)
            ioff=0
            for r in range(MPIsize):
                if( nCGTO1_A[r] < 1 ):
                    continue
                iupl=ioff + nCGTO1_A[r]
                if(r==0):
                    if(MPIrank==0):
                        zbuf1[:,:, ioff:iupl]=ppnl_half_sub[dir][Icol][:,:,0:nCGTO1_A[r]]
                        ### mpi_prtout_zbuf("zbuf1_%d_0"%(Icol)+"_%02d",zbuf1[:,:, ioff:iupl],description="[%d:%d)"%(ioff,iupl),Append=False)
                        ### mpi_prtout_zbuf("ppnl_half_sub_%d"%(Icol)+"_%02d",ppnl_half_sub[Icol],description="",Append=False)
                else:
                    cwks=numpy.zeros([nKpoints, nCGTO2_A[Icol],nCGTO1_A[r]], dtype=numpy.complex128)
                    if( MPIrank==0 ):
                        comm.Recv(cwks, source=r, tag=( MPIsize*Icol + r ))
                        zbuf1[:,:,ioff:iupl] =  cwks
                        ### mpi_prtout_zbuf("zbuf1_%d_%d"%(Icol,r)+"_%02d",zbuf1[:,:, ioff:iupl],description="[%d:%d)"%(ioff,iupl),Append=False)
                        
                    elif( MPIrank == r ):
                        comm.Send( numpy.array(ppnl_half_sub[dir][Icol]), dest=0,  tag=( MPIsize*Icol + r ))
                        ### mpi_prtout_zbuf("ppnl_half_sub_%d_%d"%(Icol,r)+"_%02d",ppnl_half_sub[Icol],description="",Append=False)
                    

                ioff=iupl ## <<< never remove this >>>
            ppnl_half[dir].append( zbuf1 ) ## up to 3 collection of [ nKpoints, nCGTO2_A[Icol],nCGTO1_sum ]

##     if( gto_ps_pp_int01_static.prtout_calc_pbc_overlaps_retv_ ):
##         z1a=[]
##         fdIN=open(gto_ps_pp_int01_static.fnme_prtout_calc_pbc_overlaps_retv_,"r")
##         for line in fdIN:
##             line=line.strip();le=len(line);
##             if(le<1):
##                 continue
##             if(line[0]=='#'):
##                 continue
##             sA=line.split();nA=len(sA);assert nA%2==0,"";nZ=nA//2
##             for j in range(nZ):
##                 z1a.append( float(sA[2*j]) + 1j*float(sA[2*j+1]) )
##         fclose(fdIN)
##         z1b=[]
##         for Icol in range(3):
##             nCGTO_i=nCGTOb_Array[Icol]
##             if( nCGTO_i < 1 ):
##                 continue
##             for kp in range(nKpoints):
##                 for j2 in range(nCGTOb_Array[Icol]):
##                     for k1 in range(nCGTO_1):
##                         z1b.append( zbuf[0][Icol][kp][j2][k1] )
##         dev=z1diff( z1a, z1b, "calc_pbc_overlapsxx")
##         assert dev<1.0e-7,""
    return ppnl_half

def calc_pp_overlapsxx1(cell, A_over_c, kpts, RnucAU_sub, Sy_sub, RnucAU_full, Sy_full):
    #libname=pathlib.Path().absolute() / "libpp01.so"    
    #c_lib = ctypes.CDLL(libname)
    c_lib = lib.load_library("libpp01")

    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();

    kvectors=numpy.reshape( kpts, (-1,3))
    nKpoints=len(kvectors)
    kvecs=[]
    for v in numpy.ravel(kvectors):
        kvecs.append(v)
    kvectors_1=( ctypes.c_double*(nKpoints*3) )(*kvecs)

    Nat_sub=len(Sy_sub)
    IZnuc_sub=[];
    for i in range(Nat_sub):
        IZnuc_sub.append( atomicsymbol_to_atomicnumber( Sy_sub[i] ) )
    Nat_full=len(Sy_full)
    IZnuc_full=[];
    for i in range(Nat_full):
        IZnuc_full.append( atomicsymbol_to_atomicnumber( Sy_full[i] ) )

    bset_info=gen_bset_info(cell,IZnuc_sub)
    
    if( cell.exp_to_discard is not None ):
        filtered =filter_by_exponent(bset_info, cell.exp_to_discard )
        bset_info=filtered

    ## 
    alph,cofs,Ell,distinctIZnuc_sub,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_info(bset_info)
    nDa_sub=len(bset_info)
    
    IZnuc_B=IZnuc_full.copy(); Nat_B=len(IZnuc_full)
    alph_B=[];cofs_B=[];Nsh_B=[];NpGTO_B=[];ell_B=[]
    distinct_IZnuc_B=[];N_cols=[];n_cum=0
    for symb in cell._pseudo:
        n_th=0;n_cum=0
        distinct_IZnuc_B.append( atomicsymbol_to_atomicnumber(symb) )
        pp = cell._pseudo[symb]
        nproj_types = pp[4]
        nsh=0
        for l, (rl, nl, hl) in enumerate(pp[5:]):
            n_th+=1;n_cum+=(2*l+1)
            ### print("#ppshell_"+symb+":%d l=%d n_cum=%d"%(n_th,l,n_cum))
            if nl > 0:
                alpha = .5 / rl**2
                alph_B.append(alpha); cofs_B.append(1.0);    # for each pGTO
                ell_B.append(l); NpGTO_B.append(1); nsh+=1;  # for each shell
        Nsh_B.append(nsh)                                    # for each nDa_sub
#
    n_call=Logger.Countup("pp_ncols");
    verbose_pp=(0 if(n_call!=1) else 2); verbose_pp=3;
    print({"IZnuc_full":IZnuc_full, "distinct_IZnuc_B":distinct_IZnuc_B,"Nsh_B":Nsh_B})
    n_cols=get_ncols(cell,IZnuc_full,distinct_IZnuc_B,Nsh_B,title="pp_ncols",verbose=verbose_pp )
    n_cols=list_to_a1(n_cols)
    BravisVectors= numpy.array( cell.a )/physicalconstants.BOHRinANGS
## note: cell.a this is given in ANGS unit...
## 
    ### print("#BravisVectors","");print(BravisVectors);### assert False,""
    spdm=3;
    BravisVectors_1=( ctypes.c_double*(spdm*3) )( *numpy.ravel(BravisVectors) )
    Vectorfield_1=( ctypes.c_double*(3) )( *A_over_c )
    nKpoints=len(kpts);### print(kpts);print(nKpoints);### assert False,""

    Npgto_sum_B=len(alph_B)
    Nsh_sum_B=sum(Nsh_B)
    nDa_B=len(distinct_IZnuc_B)
    le=len(n_cols)
    ncols_2=( ctypes.c_int*(le))(*n_cols)
    RnucAU_sub =list(numpy.ravel(RnucAU_sub))
    Rnuc_1 =( ctypes.c_double*(3*Nat_sub) )(*RnucAU_sub)
    IZnuc_1=( ctypes.c_int*(Nat_sub) )( *IZnuc_sub )
    Nsh_1  =( ctypes.c_int*(nDa_sub) )( *Nsh ) 
    Ell_1  =( ctypes.c_int*(Nsh_sum) )( *Ell )
    Npgto_1=( ctypes.c_int*(Nsh_sum) )(*Npgto)
    distinct_IZnuc_1=( ctypes.c_int*(nDa_sub) )( *distinctIZnuc_sub )
    alph_1=( ctypes.c_double*(Npgto_sum) )(*alph)
    cofs_1=( ctypes.c_double*(Npgto_sum) )(*cofs) 

    nCGTO_1=count_nCGTO(IZnuc_sub,distinctIZnuc_sub,Nsh,Ell); nCGTO_1=int(nCGTO_1)
    nCGTOb_Array=count_nCGTO(IZnuc_B,distinct_IZnuc_B,Nsh_B,ell_B,n_cols=n_cols,verbose=verbose_pp);
    nCGTO_2=nCGTOb_Array[0]
    nCGTO_2=int(nCGTO_2)

    IZnuc_2=( ctypes.c_int*(Nat_B) )( *IZnuc_B )
    RnucAU_full =list(numpy.ravel(RnucAU_full))
    Rnuc_2 =( ctypes.c_double*(3*Nat_full) )(*RnucAU_full)

    Nsh_2  =( ctypes.c_int*(nDa_B) )( *Nsh_B ) 
    Ell_2  =( ctypes.c_int*(Nsh_sum_B) )( *ell_B )
    Npgto_2=( ctypes.c_int*(Nsh_sum_B) )(*NpGTO_B)
    distinct_IZnuc_2=( ctypes.c_int*(nDa_B) )( *distinct_IZnuc_B )
    alph_2=( ctypes.c_double*(Npgto_sum_B) )(*alph_B)
    cofs_2=( ctypes.c_double*(Npgto_sum_B) )(*cofs_B) 
    Ndim_retv=[4,3,nKpoints,nCGTO_2,nCGTO_1,2]

    if( nKpoints*nCGTO_2*nCGTO_1 <=0 ):
        assert nCGTO_2==0,""
        printout("zero dimension of PSP:"+str( [nKpoints,nCGTO_2,nCGTO_1] ),warning=1)
        return [ [],[],[],[] ];
#        assert False,"%d %d %d"%(nKpoints,nCGTO_2,nCGTO_1)
        ### kpts_lst = ( numpy.zeros((1,3)) if(kpts is None) else numpy.reshape(kpts, (-1,3)) )
        ### fakecell, hl_blocks = fake_cell_vnl(cell)
        ### ppnl_half_o = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
        ### printout("ppnl_half_o:SHAPE:"+str(numpy.shape(ppnl_half_o))+"  /"+str( Ndim_retv ),flush=True)
        ### printout("ppnl_half_o:VALS:"+str( ppnl_half_o ) );
        ### return ppnl_half_o
    assert (nKpoints*nCGTO_2*nCGTO_1>0),"%d %d %d"%(nKpoints,nCGTO_2,nCGTO_1)
    Lx=8;Ly=8;Lz=8
    piX2=6.283185307179586476925286766559
    Lj=[-1,-1,-1];
    for dir in range(3):
        abs_aDOTk=[ abs(numpy.vdot(BravisVectors[dir],kvectors[j])) for j in range(nKpoints) ]
        ceil_2pi_over_aDOTk=[ ( 1 if(abs_aDOTk[j]<1e-6) else int(math.ceil( piX2/abs_aDOTk[j])) ) for j in range(nKpoints) ]
        Lj[dir]=max( ceil_2pi_over_aDOTk )
        ### fprintf(dbgfpath,"%02d:"%(dir)+str(abs_aDOTk)+" > "+str(ceil_2pi_over_aDOTk),Append=True)
    Lx=Lj[0];Ly=Lj[1];Lz=Lj[2]

    fileIO=False;both=False
    
    key='calc_pbc_overlaps_output'
    if( rttddft_common.Params_get(key) is not None ):
        dum=rttddft_common.Params_get(key)
        if( dum is not None):
            if( dum == 'F' or dum=='f'):
                fileIO=True;both=False
            elif( dum=='B' or dum=='b'):
                fileIO=True;both=True
            elif( dum=='D' or dum=='d'):
                fileIO=False;both=False
            else:
                assert False, key+":"+str(dum)
    zbuf=None;zbuf_refr=None;

    n_call=rttddft_common.Countup("calc_pp_overlapsxx1");
    dbgng_libf01x=( n_call==1 or n_call==5)    ## XXX XXX PLS. remove this in the expanse cluster ...
    prtout_retv1=False; ## ( nc==1 )
    prtout_retv2=False; ## ( nc==1 )
    if( fileIO ):
        filenumber=MPIrank
        c_lib.calc_pbc_overlaps02x_f01( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                    Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                    Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                    Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                    cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                    kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank, filenumber);
        retf="calc_pbc_overlapsx_%03d.retf"%(filenumber)
        retv=numpy.reshape( readwrite_xbuf('R',retf), [4,3,nKpoints,nCGTO_2,nCGTO_1])
        zbuf=[ [], [], [], [] ]
        for dir in range(4):
            for Icol in range(3):
                nCGTO_i=nCGTOb_Array[Icol]
                if( nCGTO_i<1 ):
                    continue
                array=numpy.zeros([nKpoints,nCGTO_i,nCGTO_1],dtype=numpy.complex128)
                for kp in range(nKpoints):
                    for j2 in range(nCGTOb_Array[Icol]):
                        for k1 in range(nCGTO_1):
                            array[kp][j2][k1] = retv[dir][Icol][kp][j2][k1]
                zbuf[dir].append(array)
        ### print_zbufs("calc_pp_overlapx",False,zbuf) ## TODO 
        if( dbgng_libf01x ):
            filenumber=MPIrank+300;
            c_lib.calc_pbc_overlaps02x_f( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                        Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                        Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                        Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                        cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                        kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank, filenumber);
            retf="calc_pbc_overlapsx_%03d.retf"%(filenumber)
            retv_ref=numpy.reshape( readwrite_xbuf('R',retf), [4,3,nKpoints,nCGTO_2,nCGTO_1])
            zbuf_ref=[ [], [], [], [] ]; zbuf_diff=[]
            for dir in range(4):
                for Icol in range(3):
                    nCGTO_i=nCGTOb_Array[Icol]
                    if( nCGTO_i<1 ):
                        continue
                    array=numpy.zeros([nKpoints,nCGTO_i,nCGTO_1],dtype=numpy.complex128)
                    for kp in range(nKpoints):
                        for j2 in range(nCGTOb_Array[Icol]):
                            for k1 in range(nCGTO_1):
                                array[kp][j2][k1] = retv_ref[dir][Icol][kp][j2][k1]
                    zbuf_ref[dir].append(array)
                    zbuf_diff.append( aNmaxdiff( zbuf[dir][Icol], zbuf_ref[dir][Icol] ) )
            dev=max( zbuf_diff ); devTOL=1.0e-7
            if( dev >= devTOL or MPIrank==0 ):
                print("#dbgng_libf01x:diff %e"%(dev)+str(zbuf_diff))
            assert dev<devTOL,""

    if(both):
        zbuf_refr=[ [ arrayclone(zbuf[dir][J]) for J in range( len(zbuf[dir]) ) ] for dir in range(4) ]
        
    if(not fileIO or both):
        ### Ndim_retv=[4,3,nKpoints,nCGTO_2,nCGTO_1,2]
        c_lib.calc_pbc_overlaps02x.restype= ndpointer(dtype=ctypes.c_double,shape=Ndim_retv )
        print("#clib:restype:Ndim_retv:"+str(Ndim_retv));
        ### Here is alternative choice: parameters are handed over files
        write_file("calc_pbc_overlaps02x.in", Nat_sub,RnucAU_sub,IZnuc_sub,nDa_sub,distinctIZnuc_sub, 
                                      Nsh, numpy.ravel(Ell), numpy.ravel(Npgto) ,numpy.ravel(alph),numpy.ravel(cofs),
                                      Nat_full,   numpy.ravel(RnucAU_full),numpy.ravel(IZnuc_B),nDa_B,
                                      numpy.ravel(distinct_IZnuc_B),  Nsh_B,    numpy.ravel(ell_B), n_cols, NpGTO_B,
                                      alph_B,   cofs_B,   spdm,    numpy.ravel(BravisVectors), A_over_c,  
                                      nKpoints, kvecs,    Lx,      Ly,                         Lz, 
                                      nCGTO_2,  nCGTO_1, MPIrank)
### dcmplx *calc_pbc_overlaps02x(int nAtm,        double *Rnuc,    int *IZnuc,            int nDa,            int *distinctIZnuc,
###      const int *Nsh,  const int *ell,  const int *npGTO,      const double *alph, const double *cofs,
###      int nAtmB,       double *RnucB,   int *IZnucB,           int nDb,            int *distinctIZnucB,
###      const int *NshB, const int *ellB, const int *n_cols,     const int *npGTOB,  const double *alphB,
###      const double *cofsB,    int spdm, double *BravaisVectors, double *Vectorfield,int nKpoints,
###      double *kvectors,       int Lx,int Ly,int Lz,  const int nCGTO_2, const int nCGTO_1, const int MPIrank){
        
        dbuf=c_lib.calc_pbc_overlaps02x( Nat_sub,   Rnuc_1, IZnuc_1, nDa_sub,     distinct_IZnuc_1, 
                                        Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                        Nat_full,   Rnuc_2, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                        Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                        cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                        kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank)
        ### Dbgtrace(":clib returns",flush=True);
        zbuf=[ [], [], [], [] ]
        for dir in range(4):
            for Icol in range(3):
                nCGTO_i=nCGTOb_Array[Icol]
                if( nCGTO_i<1 ):
                    continue
                array=numpy.zeros([nKpoints,nCGTO_i,nCGTO_1],dtype=numpy.complex128)
                for kp in range(nKpoints):
                    for j2 in range(nCGTOb_Array[Icol]):
                        for k1 in range(nCGTO_1):
                            array[kp][j2][k1] = dbuf[dir][Icol][kp][j2][k1][0] + 1j*dbuf[dir][Icol][kp][j2][k1][1]
                zbuf[dir].append(array)
        ### print_zbufs("calc_pp_overlapx",False,zbuf) ## TODO 
        ## print("calc_pbc_overlaps:direct IO:"+str(zbuf[0][0][0][0]))
        ### Dbgtrace(".calc_pp_overlapsxx1:clib returns.... %r"%(both),flush=True);
    
    if(both):
        for dir in range(4):
            leL=len(zbuf[dir]);leR=len(zbuf_refr[dir]);le=min(leL,leR)
            devs=[];maxdev=-1;
            for J in range(le):
                dev1=aNmaxdiff(zbuf[dir][J],zbuf_refr[dir][J]);
                devs.append(dev1);maxdev=max(maxdev,dev1)
            assert maxdev<1.0e-6,""

            fdDBG=open("comp_pbc_overlapsx.dat","a")
            print("maxdev:%e "%(maxdev)+str(devs),file=fdDBG)                        
            fdDBG.close();
            readwrite_xbuf('W',"comp_pbc_overlaps_LHS%02d.dat"%(dir), data=zbuf[dir])
            readwrite_xbuf('W',"comp_pbc_overlaps_RHS%02d.dat"%(dir), data=zbuf_refr[dir])
    for dir in range(4):    
        ### Dbgtrace(".calc_pp_overlapsxx1:modify_normalization.... %d"%(dir),flush=True);
        zbuf[dir]=modify_normalization(zbuf[dir],nCGTOb_Array,nKpoints,nCGTO_1,n_cols, IZnuc_full,distinct_IZnuc_B,ell_B,alph_B,Nsh_B)

    ### for dir in range(4):    
    ###    readwrite_xbuf('W',"pbc_overlaps_af-modif%02d.dat"%(dir), data=zbuf[dir])
    ### Dbgtrace(".calc_pp_overlapsxx1:END.... "+str(numpy.shape(zbuf)),flush=True);
    return zbuf

def comp_pbc_nabla(cell, kpts, RnucAU, Sy):
    #libname=pathlib.Path().absolute() / "libpp01.so"    
    #c_lib = ctypes.CDLL(libname)
    c_lib = lib.load_library("libpp01")

    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();
    kvectors=numpy.reshape( kpts, (-1,3))
    nKpoints=len(kvectors)
    kvecs=[]
    for v in numpy.ravel(kvectors):
        kvecs.append(v)
    kvectors_1=( ctypes.c_double*(nKpoints*3) )(*kvecs)

    Nat=len(Sy)
    IZnuc=[];
    for i in range(Nat):
        IZnuc.append( atomicsymbol_to_atomicnumber( Sy[i] ) )
    bset_info=gen_bset_info(cell,IZnuc)
    if( cell.exp_to_discard is not None ):
        filtered =filter_by_exponent(bset_info, cell.exp_to_discard )
        bset_info=filtered
    alph,cofs,Ell,distinctIZnuc,Nsh,Npgto,Npgto_sum,Nsh_sum=reorg_bset_info(bset_info)
    BravisVectors= numpy.array( cell.a )/physicalconstants.BOHRinANGS
    spdm=3;Lx=8;Ly=8;Lz=8
    BravaisVectors_1=( ctypes.c_double*(spdm*3) )( *numpy.ravel(BravisVectors) )
    nKpoints=len(kpts);### print(kpts);print(nKpoints);### assert False,""
    nDa=len(bset_info)
    RnucAU =list(numpy.ravel(RnucAU))
    Rnuc_1 =( ctypes.c_double*(3*Nat) )(*RnucAU)
    IZnuc_1=( ctypes.c_int*(Nat) )( *IZnuc )
    Nsh_1  =( ctypes.c_int*(nDa) )( *Nsh ) 
    Ell_1  =( ctypes.c_int*(Nsh_sum) )( *Ell )
    Npgto_1=( ctypes.c_int*(Nsh_sum) )(*Npgto)
    distinct_IZnuc_1=( ctypes.c_int*(nDa) )( *distinctIZnuc )
    alph_1=( ctypes.c_double*(Npgto_sum) )(*alph)
    cofs_1=( ctypes.c_double*(Npgto_sum) )(*cofs) 
    filenumber=200+MPIrank
    nCGTO_1=count_nCGTO(IZnuc,distinctIZnuc,Nsh,Ell); nCGTO_1=int(nCGTO_1)
    c_lib.calc_nabla_f(Nat, Rnuc_1, IZnuc_1, nDa, distinct_IZnuc_1, Nsh_1, Ell_1, Npgto_1,
                       alph_1, cofs_1,spdm, BravaisVectors_1, nKpoints, kvectors_1, Lx,Ly,Lz, nCGTO_1, MPIrank, filenumber)
    retf="calc_nabla_f%03d.retf"%(filenumber)
    retv=numpy.reshape( readwrite_xbuf('R',retf), [nKpoints,3,nCGTO_1,nCGTO_1])

    matrix = cell.pbc_intor('int1e_ipovlp',comp=3, hermi=2, kpts=kvectors)
    if(MPIrank==0):
        fd=open("comp_ipovlp.dat","w");iblk=0
        for kp in range(nKpoints):
            for dir in range(3):
                print("#%05d  %03d,%d"%(iblk,kp,dir),file=fd)
                for il in range(nCGTO_1):
                    for jr in range(nCGTO_1):
                        print("%04d %04d    %16.6f %16.6f    %16.6f %16.6f   %e %e"%(il,jr, 
                            retv[kp][dir][il][jr].real, retv[kp][dir][il][jr].imag, matrix[kp][dir][il][jr].real, matrix[kp][dir][il][jr].imag,
                            abs( retv[kp][dir][il][jr]-matrix[kp][dir][il][jr] ), abs( retv[kp][dir][il][jr]+matrix[kp][dir][il][jr] )),file=fd)
                print("\n",file=fd)
        fd.close()
def check_ppnlx(ppnl,dir,k,TINY=1e-7):
    Ndim=numpy.shape(ppnl);rank=len(Ndim); ## [dir][k][nAO][nAO]
    Absmax=-1;At=None;Vals=None;Nonzero=0
    assert rank==4,""
    for I in range(Ndim[0]):
        for J in range(I,Ndim[1]):
            absmax=-1;at=None;val=None;nonzero=0
            for K in range(I,Ndim[2]):
                for L in range(I,Ndim[3]):
                    if( rank==4 ):
                        dum=abs( ppnl[I][J][K][L] )
                        if( dum > absmax):
                            absmax=dum;at=[K,L];val=ppnl[I][J][K][L]
                        if( dum > TINY ):
                            nonzero+=1
            if( absmax > Absmax ):
                Absmax=absmax;At=[I,J,at[0],at[1]];Val=val;Nonzero+=nonzero
    if( maxdev > TINY ):
        fnme=rttddft_common.get_job(True) + "_ppnlx.log"; fd1=open(fnme,"a");
        print("#check_ppnlx:nonzero matrix elmt:dir=%d,k=%03d: [%d][%d] %16.8f,%16.8f  Nnonzero=%d"%(\
              At[0],At[1],At[2],At[3],  Val.real,Val.imag, Nonzero),file=fd1); fd1.close()
        fd1.close(); os.system("fopen "+fnme);
def check_xV(xV,dir,k,TINY=1e-7):
    ## Hermicity of xV... part of ppnl[dir][k] [nAO][nAO]
    Ndim=numpy.shape(xV);rank=len(Ndim);
    if(rank!=2):
        fd=open("pySCFwarning.log","w");print("check_xV:rank is not 2.. "+str(Ndim),file=fd);fd.close()
        os.system("fopen pySCFwarning.log"); return
    if(Ndim[0]!=Ndim[1]):
        fd=open("pySCFwarning.log","w");print("check_xV:is not SQUARE.. "+str(Ndim),file=fd);fd.close()
        os.system("fopen pySCFwarning.log"); return
    maxdev=-1;at=None;vals=None
    for I in range(Ndim[0]):
        for J in range(I,Ndim[1]):
            lhs=xV[I][J];rhs=xV[J][I].conj()
            dev=abs(lhs-rhs)
            if( dev > maxdev ):
                maxdev=dev;at=[I,J]; vals=[lhs,rhs]
    if( maxdev > TINY ):
        fnme=rttddft_common.get_job(True) + "_xVVx.log"; fd1=open(fnme,"a");
        print("#check_xV:antiHermicity:dir=%d,k=%03d: %18.6e   [%d][%d] %16.8f,%16.8f  %16.8f,%16.8f"%(\
              dir,k,maxdev,at[0],at[1],vals[0].real,vals[0].imag, vals[1].real, vals[1].imag),file=fd1); fd1.close()
        fd1.close(); os.system("fopen "+fnme);  ## only in exceptional case (xVVx!=0)
    return maxdev

def printout_zmatrices(path,Mat1,Mat2,index,Append=False,fopen=False, Ndim_uplm=None):
    def fnformat(arg1,arg2):
        return "%14.8f,%14.8f        %14.8f,%14.8f       %16.6e"%(arg1.real,arg1.imag, arg2.real,arg2.imag, abs(arg1-arg2))

    N_call=rttddft_common.Countup(path)
    fd=open(path,("a" if(Append or N_call>1) else "w"))
    Ndim=numpy.shape(Mat1);rank=len(Ndim)
    Ia=numpy.zeros([rank],dtype=int)
    strNdim_o=str(Ndim)

    if( Ndim_uplm is not None ):
        rank_uplm=len(Ndim_uplm)
        for j in range(rank_uplm):
            Ndim[j]=min( Ndim[j], Ndim_uplm[j] )
        print("%s#%04d:%s  Ndim:%s / %s"%( ("" if(N_call==1) else "\n\n\n"), (N_call-1), str(index), str(Ndim), strNdim_o ),file=fd)
    else:
        print("%s#%04d:%s  Ndim:%s"%( ("" if(N_call==1) else "\n\n\n"), (N_call-1), str(index), strNdim_o ),file=fd)

    for Ia[0] in range(Ndim[0]):
        if(rank==1):
            print(" %3d:"%(Ia[0])+fnformat(Mat1[Ia[0]],Mat2[Ia[0]]),file=fd);continue
        for Ia[1] in range(Ndim[1]):
            if(rank==2):
                print(" %3d,%3d:"%(Ia[0],Ia[1])+fnformat( Mat1[Ia[0]][Ia[1]],Mat2[Ia[0]][Ia[1]] ),file=fd);continue
            for Ia[2] in range(Ndim[2]):
                if(rank==3):
                    print(" %3d,%3d,%3d:"%(Ia[0],Ia[1],Ia[2])+fnformat( Mat1[Ia[0]][Ia[1]][Ia[2]],Mat2[Ia[0]][Ia[1]][Ia[2]] ),file=fd);continue
                for Ia[3] in range(Ndim[3]):
                    if(rank==4):
                        print(" %3d,%3d,%3d,%3d:"%(Ia[0],Ia[1],Ia[2],Ia[3])+fnformat( Mat1[Ia[0]][Ia[1]][Ia[2]][Ia[3]],Mat2[Ia[0]][Ia[1]][Ia[2]][Ia[3]] ),file=fd);continue
                    for Ia[4] in range(Ndim[3]):
                        if(rank==5):
                            print(" %3d,%3d,%3d,%3d,%3d:"%(Ia[0],Ia[1],Ia[2],Ia[3],Ia[4])+fnformat( Mat1[Ia[0]][Ia[1]][Ia[2]][Ia[3]][Ia[4]],Mat2[Ia[0]][Ia[1]][Ia[2]][Ia[3]][Ia[4]] ),file=fd);continue
                        else:
                            assert False,"%d"%(rank)+str(Ndim)
    fd.close()
    if(fopen):
        os.system("fopen "+path)

def print_zbufs(name,Append,zbuf,fopen=False):
    # zbuf[dir][Icol][kp][j2][k1] => FILE_Icol,kp: j2,k1,[0][1][2][3] ...
    zbuf_0=zbuf[0];
    N_col=len(zbuf_0)
    Ndim=numpy.shape(zbuf_0[0]);
    nkpt=Ndim[0];nBS2=Ndim[1];nBS1=Ndim[2];
    fnmes=[]
    for Icol in range(N_col):
        for kp in range(nkpt):
            fnme="%s_%d_%03d.dat"%(name,Icol,kp);fd=open(fnme,("a" if(Append) else "w"))
            fnmes.append(fnme)
            for j2 in range(nBS2):
                for k1 in range(nBS1):
                    print(" %3d,%3d    %16.8f,%16.8f    %16.8f,%16.8f    %16.8f,%16.8f    %16.8f,%16.8f"%(\
                        j2,k1,  zbuf[0][Icol][kp][j2][k1].real,zbuf[0][Icol][kp][j2][k1].imag,\
                        zbuf[1][Icol][kp][j2][k1].real,zbuf[1][Icol][kp][j2][k1].imag,\
                        zbuf[2][Icol][kp][j2][k1].real,zbuf[2][Icol][kp][j2][k1].imag,\
                        zbuf[3][Icol][kp][j2][k1].real,zbuf[3][Icol][kp][j2][k1].imag),file=fd)
            fd.close()
    if(fopen):
        os.system("fopen "+fnmes[0]);
    for F in fnmes:
        os.system("ls -ltrh "+F)

        

            
