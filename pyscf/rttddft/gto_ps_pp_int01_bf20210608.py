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
from .utils import comp_zbufs,readwrite_xbuf,arrayclone,print_a2maxdiff,print_aNmatr,aNmaxdiff,print_Hmatrices,prtaNx2,atomicsymbol_to_atomicnumber,parse_xyzstring,write_file,read_zbuf,i1eqb,list_to_a1,gamma_hfint,atomicnumber_to_atomicsymbol
from pyscf.gto.mole import format_basis
import ctypes
from pyscf.pbc.gto.pseudo.pp_int import get_pp_nl
from pyscf.pbc.gto import pseudo
from numpy.ctypeslib import ndpointer
from .Logger import Logger
from .physicalconstants import physicalconstants
from .rttddft_common import rttddft_common
from .Loglv import printout
from mpi4py import MPI
import datetime
libpbc = lib.load_library('libpbc')

def suppress_prtout():
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();
    return ( MPIrank!=0 )


def fprintf(fpath,text,Append=False,stdout=True):
    if( suppress_prtout() ):
        return
    fd=open(fpath,('a' if(Append) else 'w'))
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
        
    ### print("#check norb:"+str(norb_AL)+"/"+str(nCGTO_2A))
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

def calc_pp_overlaps(cell,A_over_c,kpts):
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
    ### print(kvectors_1)
    IZnuc=[];
    Rnuc_au,Sy=parse_xyzstring(cell.atom,unit_BorA='B')
    Nat=len(Sy)
    IZnuc=[];
    for i in range(Nat):
        IZnuc.append( atomicsymbol_to_atomicnumber( Sy[i] ) )
    bset_info=gen_bset_info(cell,IZnuc)
    
    if( cell.exp_to_discard is not None ):
        filtered =filter_by_exponent(bset_info, cell.exp_to_discard )
        bset_info=filtered

    ### bset_info=format_basis({'Ti':'sto-3g','O':'sto-3g'})
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
    
    IZnuc_B=IZnuc.copy(); Nat_B=len(IZnuc_B)
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
        Nsh_B.append(nsh)                                    # for each nDa
#ppshell_Ti:1 l=0 n_cum=1
#ppshell_Ti:2 l=1 n_cum=4
#ppshell_Ti:3 l=2 n_cum=9
#ppshell_O:1 l=0 n_cum=1
#ppshell_O:2 l=1 n_cum=4
#
    n_call=Logger.Countup("pp_ncols");
    verbose_pp=(0 if(n_call!=1) else 2)
    n_cols=get_ncols(cell,IZnuc,distinct_IZnuc_B,Nsh_B,title="pp_ncols",verbose=verbose_pp )
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
    Rnuc_au =list(numpy.ravel(Rnuc_au))
    Rnuc_1 =( ctypes.c_double*(3*Nat) )(*Rnuc_au)
    IZnuc_1=( ctypes.c_int*(Nat) )( *IZnuc )
    Nsh_1  =( ctypes.c_int*(nDa) )( *Nsh ) 
    Ell_1  =( ctypes.c_int*(Nsh_sum) )( *Ell )
    Npgto_1=( ctypes.c_int*(Nsh_sum) )(*Npgto)
    distinct_IZnuc_1=( ctypes.c_int*(nDa) )( *distinct_IZnuc )
    alph_1=( ctypes.c_double*(Npgto_sum) )(*alph)
    cofs_1=( ctypes.c_double*(Npgto_sum) )(*cofs) 

    nCGTO_1=count_nCGTO(IZnuc,distinct_IZnuc,Nsh,Ell); nCGTO_1=int(nCGTO_1)
    nCGTOb_Array=count_nCGTO(IZnuc_B,distinct_IZnuc_B,Nsh_B,ell_B,n_cols=n_cols,verbose=verbose_pp);
    nCGTO_2=nCGTOb_Array[0]
    ### print(nCGTO_1,end="");print(nCGTO_1)
    ### print(nCGTO_1,end="");print(type(nCGTO_1))

    ### print(nCGTO_2,end="");print(nCGTO_2)
    ### print(nCGTO_2,end="");print(type(nCGTO_2))
    nCGTO_2=int(nCGTO_2)
###    nCGTO_2=count_nCGTO(IZnuc_B,distinct_IZnuc_B,Nsh_B,ell_B,n_cols=n_cols,title="pseudo");nCGTO_2=int(nCGTO_2)

    IZnuc_2=( ctypes.c_int*(Nat_B) )( *IZnuc_B )
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
    if( fileIO ):
        filenumber=MPIrank+100
        c_lib.calc_pbc_overlaps_f( Nat,   Rnuc_1, IZnuc_1, nDa,     distinct_IZnuc_1, 
                                    Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                    Nat,   Rnuc_1, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                    Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                    cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                    kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank,filenumber);
        retf="calc_pbc_overlaps_%03d.retf"%(filenumber)
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

    if(both):
        zbuf_refr=[ arrayclone(zbuf[J]) for J in range( len(zbuf) ) ]
        
    if(not fileIO or both):
        c_lib.calc_pbc_overlaps02.restype= ndpointer(dtype=ctypes.c_double,shape=Ndim_retv )
        print("#clib:restype:Ndim_retv:"+str(Ndim_retv));
        ### Here is alternative choice: parameters are handed over files
        write_file("calc_pbc_overlaps01.in", Nat,Rnuc_au,IZnuc,nDa,distinct_IZnuc, 
                                      Nsh, numpy.ravel(Ell), numpy.ravel(Npgto) ,numpy.ravel(alph),numpy.ravel(cofs),
                                      Nat,   numpy.ravel(Rnuc_au),numpy.ravel(IZnuc_B),nDa_B,
                                      numpy.ravel(distinct_IZnuc_B),  Nsh_B,    numpy.ravel(ell_B), n_cols, NpGTO_B,
                                      alph_B,   cofs_B,   spdm,    numpy.ravel(BravisVectors), A_over_c,  
                                      nKpoints, kvecs,    Lx,      Ly,                         Lz, 
                                      nCGTO_2,  nCGTO_1)
        ### os.system("./testread.x");
        ### zbuf=read_zbuf("calc_pbc_overlaps.dat")
        
        dbuf=c_lib.calc_pbc_overlaps02( Nat,   Rnuc_1, IZnuc_1, nDa,     distinct_IZnuc_1, 
                                        Nsh_1, Ell_1,  Npgto_1, alph_1,  cofs_1,
                                        Nat,   Rnuc_1, IZnuc_2, nDa_B,   distinct_IZnuc_2,
                                        Nsh_2, Ell_2,  ncols_2, Npgto_2, alph_2,
                                        cofs_2,spdm,   BravisVectors_1, Vectorfield_1, nKpoints, 
                                        kvectors_1,   Lx, Ly, Lz, nCGTO_2, nCGTO_1, MPIrank)
        
        if(prtout_retv1): # xxx xxx remove after debugging xxx xxx 
            for Icol in range(3):
                #str_AoverC=("0" if( abs(A_over_c[0])<1.0e-7) else ("%6.3f"%(A_over_c[0])).strip() )+"_"+\
                #           ("0" if( abs(A_over_c[1])<1.0e-7) else ("%6.3f"%(A_over_c[1])).strip() )+"_"+\
                #           ("0" if( abs(A_over_c[2])<1.0e-7) else ("%6.3f"%(A_over_c[2])).strip() )
                fnme="calc_pbc_overlaps_bf20210608"; ##+str_AoverC
                fpath01=fnme+"I%d.dat"%(Icol);
                fd=futils.fopen(fpath01,"w")  ## OK
                print("#A_over_c:%f,%f,%f"%(A_over_c[0],A_over_c[1],A_over_c[2]),file=fd);
                for kp in range(nKpoints):
                    print("#%03d:K=%d\n"%(kp,kp),file=fd);
                    for j2 in range(nCGTO_2):
                        string=""
                        for k1 in range(nCGTO_1):
                            string+="%12.6f %12.6f        "%(dbuf[Icol][kp][j2][k1][0],dbuf[Icol][kp][j2][k1][1])
                        print(string,file=fd);
                        ### print("%d %d %d %d  %f %f"%(Icol,kp,j2,k1,dbuf[Icol][kp][j2][k1][0],dbuf[Icol][kp][j2][k1][1]),file=fd)
                futils.fclose(fd);
                ### os.system("fopen "+fpath01);
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

        if(maxdev>=1.0e-6):
            comp_zbufs(zbuf,zbuf_refr,"comp_pbc_overlaps.diff",Append=False,description="zbuf,zbuf_refr")

        assert maxdev<1.0e-6,"maxdev=%e"%(maxdev)

        # fdDBG=open("comp_pbc_overlaps.dat","a")
        # print("maxdev:%e "%(maxdev)+str(devs),file=fdDBG)                        
        # fdDBG.close();
        # readwrite_xbuf('W',"comp_pbc_overlaps_LHS.dat",data=zbuf)
        # readwrite_xbuf('W',"comp_pbc_overlaps_RHS.dat",data=zbuf_refr)
        
    zbuf=modify_normalization(zbuf,nCGTOb_Array,nKpoints,nCGTO_1,n_cols, IZnuc,distinct_IZnuc_B,ell_B,alph_B,Nsh_B)
    if(prtout_retv2):
        fd=futils.fopen("calc_pbc_overlaps_pyretv.dat","w")#OK
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
    return zbuf

def with_df_get_pp(mydf,A_over_c,kpts=None):
    import time
    from .update_dict import printout_dict,update_dict
    
    ## Copy of pyscf.pbc.df.aft.py ---
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    idbgng=0 # XXX XXX
    diff_ZF_to_fF=False

    # fncnme|fncdepth|Wctm000|Wctm010|N_call|Dic_timing|dic1_timing
    fncnme="with_df_get_pp";fncdepth=3   ## fncdepth ~ SCF:0 vhf:1 get_j:2 XXX XXX
    Wctm000=time.time();Wctm010=Wctm000;dic1_timing={}                    # XXX XXX
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

    vloc1 = get_pp_loc_part1(mydf, kpts_lst)
    Wctm020=Wctm010;Wctm010=time.time() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part1",Wctm010-Wctm020,depth=fncdepth) # XXX XXX

    vloc2 = pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
    Wctm020=Wctm010;Wctm010=time.time() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_loc_part2",Wctm010-Wctm020,depth=fncdepth) # XXX XXX

    vpp_org=None;vpp_ZF=None
    if( (idbgng > 0) and (n_call==1 or n_call==20 or n_call==200) ):
        vpp_org = pseudo.pp_int.get_pp_nl(cell, kpts_lst)
        
        zerovec=[ 0.0, 0.0, 0.0]
        vpp_ZF= get_pp_nl01(cell, zerovec, kpts_lst)
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
        assert (dev<1.0e-6),"diff=%e"%(dev)
    ### print("vloc1;",end="");print( numpy.shape(vloc1))  ## vloc1;(8, 56, 56)
    ### print("vloc2;",end="");print( numpy.shape(vloc2))  ## vloc1;(8, 56, 56)
    ### print("vpp_org;",end="");print( numpy.shape(vpp_org)) ## (8, 56, 56)

    vpp = get_pp_nl01(cell, A_over_c, kpts_lst)
    Wctm020=Wctm010;Wctm010=time.time() # XXX XXX
    update_dict(fncnme,dic1_timing,Dic_timing,"get_pp_nl01",Wctm010-Wctm020,depth=fncdepth) # XXX XXX
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
    Wctm010=time.time()  # XXX XXX
    printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth) # XXX XXX

    return vpp

class get_pp_nl01_static:
    count=0
    Dict_={}
    @staticmethod
    def Countup(key,inc=True):
        if( key not in get_pp_nl01_static.Dict_ ):
            get_pp_nl01_static.Dict_.update({key:0});
        if( inc ):
            get_pp_nl01_static.Dict_[key]+=1
        return get_pp_nl01_static.Dict_[key]

def get_pp_nl01(cell, A_over_c, kpts=None):
    ### fd01=open("get_pp_nl01_dbg.log","w");print("start:"+str( datetime.datetime.now() ),file=fd01);fd01.close()

    Logger.write_once(None,"get_pp_nl01:calc_pp_overlaps","calculate pp for A="+str(A_over_c))
    if kpts is None:
        assert False,"please set kpts explicitly..";
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    fakecell, hl_blocks = fake_cell_vnl(cell)   ## pyscf.pbc.gto.pseudo.pp_int
    dbgng=1;strict=1;
    if(dbgng>0):
        get_pp_nl01_static.count=get_pp_nl01_static.count+1
        if( get_pp_nl01_static.count==1 or get_pp_nl01_static.count==20):
            ppnl_half_o = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
            fd001=open("ppnl_half_o_01.log","w");
            print("#ppnl_half_o:"+str(datetime.datetime.now()),file=fd001);
            ### print(ppnl_half_o,file=fd001);### fd001.close()
            print( len(ppnl_half_o),file=fd001 );
            for I in range( len(ppnl_half_o) ):
                print(numpy.shape(ppnl_half_o[I]),file=fd001)
                ## (1, 72, 180), (1, 16, 180), (1, 4, 180) ... so this cannot be treated as ndarray
                ## whereas  A[0],A[1],A[2] are ndarrays
            fd001.close()
            ### xxx 20210531  printout("#ppnl_half:"+str(numpy.shape(ppnl_half_o)))  never use shape or ravel
            ### xxx 20210531  ppnl_half_o = numpy.ravel(ppnl_half_o)
            datf01="ppnl_half.dat"
            if( not suppress_prtout() ):
                fd=futils.fopen(datf01,("w" if(get_pp_nl01_static.count==1) else "a")) ## OK
                for col in ppnl_half_o:
                    line=""
                    print("#"+str(numpy.shape(col)),file=fd)
                    for y in numpy.ravel(col):
                       line+=str(y)
                    print(line,file=fd)
                futils.fclose(fd);
            #print("#ppnl_half_ZF:",end=""); print(numpy.shape(ppnl_half_o))  xxx never apply np.shape to ppnl_half
            #for X in ppnl_half_o:
            #    print("#ppnl_half_ZF:",end="");print(numpy.shape(X))
            ppnl_half_ZF = calc_pp_overlaps(cell,[0,0,0],kpts_lst)    ## New subroutine
            comp_datf="ppnl_half_comp_%02d.dat"%(get_pp_nl01_static.count)
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
                    assert False,"Check results:check:"+comp_datf
                if( maxdev > 1.0e-7 or Ndev > 0 ):
                    if(strict>1):
                        assert False,"Check results:check:"+comp_datf
                    else:
                        Logger.Warning("ppnl_half_ZF","ppnl_half_ZF:maxdev=%e:"%(maxdev)+str(vals))
    if(get_pp_nl01_static.count%20==1):
        printout("#calculating ppnl_half with Field:%f"%(math.sqrt(A_over_c[0]*A_over_c[0] + A_over_c[1]*A_over_c[1] + A_over_c[2]*A_over_c[2])))
    
    abs_A_over_c= numpy.sqrt( A_over_c[0]**2 + A_over_c[1]**2 + A_over_c[2]**2 )
    ppnl_half = calc_pp_overlaps(cell,A_over_c,kpts_lst)
    ### DBG if( abs_A_over_c > 1.0e-7 ):
    ### DBG     print_aNmatr("ppnl_half.dat",ppnl_half,Append=True,comment="AoverC="+str(A_over_c))
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
    assert len(ksh_to_Iat)==len(pre),""
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
