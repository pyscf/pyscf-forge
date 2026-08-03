import sys
from scipy import linalg
import numpy as np
import os
import time
import math
import os.path
import datetime
from mpi4py import MPI

from pyscf.pbc import scf as pyscf_pbc_scf
from pyscf.scf.diis import ADIIS,CDIIS,EDIIS
from .diis import DIIS_dmat
from pyscf import lib as pyscf_lib
import pyscf.gto as molgto
from pyscf.gto import Mole
from pyscf.pbc.gto import Cell
from pyscf.gto.mole import nao_nr
from pyscf.pbc import gto, df

from .rttddft_common import rttddft_common
from .print_00 import print_00, close_00, open_00
from .Loglv import printout

from .physicalconstants import physicalconstants,PhysicalConstants
from .Logger import Logger
from .futils import futils
from .serialize import svld_aNbuf,diff_aNbuf

# i1eqb,
#from utils import print_bset,make_tempfile,dNtoa,d1toa,zNtoa,z1toa_polar,write_xyzf,write_xyzstring,read_xyzf,arrayclone,aNmaxdiff,calc_eorbs,hdiag,modify_filename,deviation_from_unitmatrix,z1maxloc,i1prod,parse_dict,dic_to_string,parse_xyzstring
from .utils import print_bset,make_tempfile,dNtoa,d1toa,zNtoa,write_xyzf,write_xyzstring,read_xyzf,arrayclone,\
     dic_to_string,parse_xyzstring,i1prod,prtout_MOocc,calc_eorbs,hdiag,modify_filename,deviation_from_unitmatrix,\
     print_00,open_00,close_00,z1maxloc,i1eqb,z1toa,z1toa_polar,aNmaxdiff,check_wfNorm,popsum,print_TDeorbs,normalize,update_dict,printout_dict 
from .rttddft01 import construct_Aind_args,rttddft_set_Aind_args,rttddft_get_Aind_args,rttddftMOL,rttddftPBC,Logger,check_nao_nr
# from rttddft02 import rttddftMOL_UKS,rttddftMOL_ROKS,rttddftPBC_UKS,rttddftPBC_ROKS 
from .serialize import load_fromfile,parse_file,serialize,construct_tdfield

Dic_wctm={}
# Alternative for read_xyzf
#
# !! in read_xyzf, -outputunit- was implicit (defaulted to ANGS, since xyzf is in ANGS unit) 
# !! whereas in this subroutine, you MUST explicitly specify A or B.
# 
# !! outputunit (A or B) also applies to lattice vectors
# 
def read_mol_or_cell(mol_or_cell,dict=None,outputunit=None): 
    assert (outputunit=='A' or outputunit=='B'), "PLS specify outputunit"
    spdm=3
    natm=mol_or_cell.natm
    BOHRinANGS=PhysicalConstants.BOHRinANGS()
    if(isinstance(mol_or_cell,Cell)):
        # 
        latticevectors_BOHR = np.array( mol_or_cell.lattice_vectors() )
        # print("latticevectors_BOHR",latticevectors_BOHR)
        if( outputunit == 'A' ):
            dict.update({'a':latticevectors_BOHR*BOHRinANGS})
        else:
            dict.update({'a':latticevectors_BOHR})
# We assume that _atom = [ row_1, row_2, ... ,row_nAtm ]
    assert len(mol_or_cell._atom)==natm,"unexpected size of mol_or_cell._atom"
    Sy=[];Rnuc_BOHR=[]
    for ia in range(natm):
        row=mol_or_cell._atom[ia]
        s=row[0];
        assert isinstance(s,str),""
        Sy.append(s)
        xyz=row[1]
        assert len(xyz)==spdm,""
        assert isinstance(xyz[0],float) or isinstance(xyz[0],np.float64),""
        Rnuc_BOHR.append(xyz)
    if(outputunit=='A'):
        Rnuc_ANGS=np.array(Rnuc_BOHR)*BOHRinANGS
        return Rnuc_ANGS,Sy
    else:
        return Rnuc_BOHR,Sy
        

def print_in_MOrep(pbc,nmult_matrix,header,AOmatrix,MOvcs,nmult_MOvcs,Threads=[0]):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if( MPIrank not in Threads ):
        return

    nmult=max( nmult_matrix, nmult_MOvcs )
    for sp in range(nmult):
        xtn=("" if(nmult==0) else ("_upSpin" if(sp==0) else "_dnSpin"))
        Matr=( AOmatrix if(nmult_matrix == 0) else AOmatrix[sp])
        Vcs =( MOvcs    if(nmult_MOvcs  == 0) else MOvcs[sp] )
        print_in_MOrep_1(pbc,header+xtn,Matr,Vcs,Threads=Threads)

def kpoints_diamondlattice(BravaisVectors_au,N_div=None):
    PIx2=6.283185307179586476925286766559
    PI=3.1415926535897932384626433832795
    sqrt2=1.4142135623730950488016887242097
    a=sqrt2*np.sqrt( np.vdot( BravaisVectors_au[0], BravaisVectors_au[0] ) )
    print(a)
    X_=np.array([ PIx2/a, 0.0, 0.0 ])
    W_=np.array([ PIx2/a, PI/a, 0.0 ])
    K_=np.array([ 1.5*PI/a, 1.5*PI/a, 0.0 ])
    L_=np.array([ PI/a, PI/a, PI/a ])
    G_=np.array([ 0.0, 0.0, 0.0 ])
    vertices=[ L_, G_, X_, W_, K_, G_];Nsect=5
##    Ndim=[       20, 20, 20, 20, 20  ];Nsect=5
    if(N_div is None):
        Ndim=[       2, 2, 2, 2, 2  ]
    else:
        Ndim=[    N_div,N_div,N_div,N_div,N_div ]

    labels=['L', 'G', 'X', 'W', 'K', 'G']
    ret=[];trlen=[];s=0.0
    for Isect in range(Nsect):
        A=vertices[Isect];B=vertices[Isect+1]
        N=Ndim[Isect]
        d=np.zeros([3],dtype=np.float64); P=np.zeros([3],dtype=np.float64)
        
        d[0]=(B[0]-A[0])/float(N); d[1]=(B[1]-A[1])/float(N); d[2]=(B[2]-A[2])/float(N);
        P[0]=A[0]; P[1]=A[1]; P[2]=A[2];
        dL=np.sqrt( np.vdot(d,d) )
        for J in range(N):
            ret.append(P.copy());trlen.append(s);
            P=P+d;s=s+dL
    ret.append(P.copy());trlen.append(s);
    return ret,trlen,Ndim,labels  ## ret[NdSUM],trlen[NdSUM],Ndim[Nsect],labels[Nsect+1] Nsect=5,NdSUM=100

def print_in_MOrep_1(pbc,header,AOmatrix,MOvcs,Threads=[0]):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if( MPIrank not in Threads ):
        return
    Ndim=np.shape(AOmatrix)
    nkp=1;nAO=Ndim[0];nMO=Ndim[1]
    if( pbc ):
        nkp=Ndim[0];nAO=Ndim[1];nMO=Ndim[2]
    for kp in range(nkp):
        MO=( MOvcs if(not pbc) else MOvcs[kp])
        DM=( AOmatrix  if(not pbc) else AOmatrix[kp])
        dMO=np.matmul( np.matrix.getH(MO), np.matmul( DM, MO) )
        for I in range(nMO):
            str="";
            for J in range(nMO):
                str+=("%16.6f "%(DM[I][I].real) if(I==J) else "%7.3f %7.3fj "%(DM[I][J].real,DM[I][J].imag))
            print(header+":%02d:%3d"%(kp,I)+str)  ### thread-0 only

#$
#$ check_wfNorms: checks wf norms and (i) fix if deviation exceeds TOL and (ii) print result to logger
#$ @param:AOrep : True if AOrep / False if orthogonal rep (i.e. \sum |wf[:]|**2 = 1.0)
def check_wfNorms(nkpt,nAO,nMO, rttddft,pbc,spinrestriction,wfn,AOrep,sqrdevtol_FIX=1.0e-6,title="",logger=None):
    assert( nAO is not None),"check_wfNorms.."
    ## wfn: R,O:spin-sym   U:spin-pol
    if( spinrestriction == 'U' ):
        check_wfNorm1(nkpt,nAO,nMO, rttddft,pbc,wfn[0],AOrep,sqrdevtol_FIX=sqrdevtol_FIX, title=title+"_upSpin",logger=logger)
        check_wfNorm1(nkpt,nAO,nMO, rttddft,pbc,wfn[1],AOrep,sqrdevtol_FIX=sqrdevtol_FIX, title=title+"_dnSpin",logger=logger)
    else:
        check_wfNorm1(nkpt,nAO,nMO, rttddft,pbc,wfn,AOrep,sqrdevtol_FIX=sqrdevtol_FIX, title=title,logger=logger)

def check_wfNorm1(nkpt,nAO,nMO, rttddft,pbc,wfn,AOrep,sqrdevtol_FIX=1.0e-6,title="",logger=None):
    sqrNorms=np.zeros(nMO)
    kvectors = (None if (not pbc) else np.reshape( rttddft.kpts, (-1,3)))
    maxdev=-1.0; Nfix=0
    for kp in range(nkpt):
        if(AOrep):            
            if( not pbc ):
                tgt=wfn;     S1e=rttddft.get_ovlp()
            else:
                tgt=wfn[kp]; S1e=rttddft.get_ovlp( rttddft.cell, kvectors[kp])
            for I in range(nMO):
                sqrNorms[I]=np.vdot( tgt[:,I], np.matmul( S1e, tgt[:,I] ) ).real
                dev = abs( sqrNorms[I]-1.0 )
                if( dev > maxdev ):
                    maxdev = dev
                if( sqrdevtol_FIX>=0  and  dev > sqrdevtol_FIX ):
                    fac=1.0/np.sqrt( sqrNorms[I] );Nfix+=1
                    for K in range(nAO):
                        tgt[K][I]=tgt[K][I]*fac
        else:
            if( not pbc ):
                tgt=wfn;
            else:
                tgt=wfn[kp];
            for I in range(nMO):
                sqrNorms[I]=np.vdot( tgt[:,I], tgt[:,I] ).real
                dev = abs( sqrNorms[I]-1.0 )
                if( dev > maxdev ):
                    maxdev = dev
                if( sqrdevtol_FIX>=0  and  dev > sqrdevtol_FIX ):
                    fac=1.0/np.sqrt( sqrNorms[I] );Nfix+=1
                    for K in range(nAO):
                        tgt[K][I]=tgt[K][I]*fac
        if(logger is not None):
            logger.Info("#orth_tdMO:%s sqrdevMax:%e Nfix:%d/%d"%(title,maxdev,Nfix,nMO)); ## multithread-adapted
    return maxdev,Nfix

##
## compare_MOs: prints out (LHS^{\dagger} S1e RHS)
##
def compare_MOs(LHS,RHS,S1e_k,title=""):

    NdL=np.shape(LHS);NdR=np.shape(RHS)
    pbc=( len(NdL)==3 )
    nkp=1;nAO=NdL[0];nMO=NdL[1]
    if( pbc ):
        nkp=NdL[0];nAO=NdL[1];nMO=NdL[2]
    for kp in range(nkp):
        lh=LHS;rh=RHS;S1=S1e_k
        if( pbc ):
            lh=LHS[kp];rh=RHS[kp];S1=S1e_k[kp]
        T=np.matmul( np.matrix.getH(lh), np.matmul(S1,rh))
        
        for lmo in range(nMO):
            print_00("#compare_MOs:%s:%02d:%3d:"%(title,kp,lmo) + z1toa_polar(T[lmo]))

def prteorbs01(CanonicalMO,moldyn,rttddft,fpath=None,Append=False,kpoints=None,description="", diamondlattice=False):
    if(fpath is None):
        fpath=(rttddft_common.get_job(True))+( "_Canonicaleorbs0.dat" if(CanonicalMO) else "_prteorbs01.dat")
    fdOUT=open(fpath,('a' if(Append) else 'w'))
    
    for Loop in range(1):
        #if( Loop==1 ):
        #    fpath=(rttddft_common.get_job(True))+"_Canonicaleorbs1.dat"
        #    fdOUT=open(fpath,"w")
        print('## Moldyn.print_canonicaleorbs:'+description+" \t\t "+str(datetime.datetime.now()),file=fdOUT)
        ## diamondlattice=(Loop==1) 
        trlen=None;Ndim=None;labels=None
        ##if(kpoints is None):
        if(diamondlattice):
            kpoints,trlen,Ndim,labels=kpoints_diamondlattice( moldyn.BravaisVectors_au,N_div=3 )
        else:
            kpoints=np.reshape( rttddft.kpts, (-1,3) )
            nKpoints=len(kpoints);trlen=[];s=0.0
            for j in range(nKpoints):
                trlen.append(s);
                if(j<nKpoints-1):
                    s+= np.sqrt( np.vdot( kpoints[j+1]-kpoints[j], kpoints[j+1]-kpoints[j]) )
            ### trlen.append(s)
        if( (labels is not None) and (Ndim is not None) ):
            gnuscript='set xtics ("'+labels[0]+'"  %f'%(0.0)
            Nsect=len(Ndim)
            
            for Isect in range(1,Nsect):
                gnuscript+=', "'+labels[Isect]+'"  %f'%( trlen[ sum(Ndim[0:Isect]) ] )
            gnuscript+=')'
            
        PIx2=6.283185307179586476925286766559
        nKpoints=len(kpoints)
        eOrbs=None;nMO=-1
        if( CanonicalMO ):
            eOrbs,coeffs=rttddft.get_bands(kpoints)
            assert len(eOrbs)==nKpoints,"check eOrbs:"+str(np.shape(eOrbs))
        nMO=0
        for kp in range(nKpoints):
            
            print("#%03d: %12.6f %12.6f %12.6f    %9.3f %9.3f %9.3f"%(
                  kp, kpoints[kp][0],kpoints[kp][1],kpoints[kp][2],  
                  np.vdot( kpoints[kp],moldyn.BravaisVectors_au[0])/PIx2,
                  np.vdot( kpoints[kp],moldyn.BravaisVectors_au[1])/PIx2,
                  np.vdot( kpoints[kp],moldyn.BravaisVectors_au[2])/PIx2 ),file=fdOUT)
            
            nMO=len(eOrbs[kp])
            strbf=" %5d %14.6f    "%(kp,trlen[kp])
            for jMO in range(nMO):
                strbf+=" %16.8f"%(eOrbs[kp][jMO])
            print(strbf,file=fdOUT)
        print("\n\n\n",file=fdOUT)
        fdOUT.close()
    gnu=fpath.replace(".dat","")
    gnuf= gnu+".plt"
    fdGNU=open(gnuf,"w")
    print("HARTREEinEV=27.211386024367243",file=fdGNU)
    print("set term postscript color enhanced",file=fdGNU)
    print("set output \"%s.ps\""%(gnu),file=fdGNU)
    print(gnuscript,file=fdGNU)
    jMO=0
    print("plot \"%s\" using 2:(HARTREEinEV*$%d) title \"\" with linespoints ls %d"%(fpath,3+jMO,jMO+1),end="",file=fdGNU)
    for jMO in range(1,nMO):
        print(",\\\n\"\" using 2:(HARTREEinEV*$%d) title \"\" with linespoints ls %d"%(3+jMO,jMO+1),end="",file=fdGNU)
    fdGNU.close()    

def get_Sinvsqrt(S):
    dbgng=False
    N=len(S)
    dtype=np.array(S).dtype
    ### print("get_Sinvsqrt:"+str(dtype))
    iscomplex=False
    if( dtype == complex or dtype==np.complex128 ):
        iscomplex=True
        eigvals,vecs,info=linalg.lapack.zheev(S)
    else:
        eigvals,vecs,info=linalg.lapack.dsyev(S)
    assert (info==0),"dsyev/zheev failed"
    InvSqrt=np.zeros([N,N],dtype=dtype)
    for I in range(N):
        for J in range(N):
            cdum=( np.complex128(0.0) if(iscomplex) else 0.0)
            for k in range(N):
                cdum+= vecs[I][k]*(1.0/math.sqrt(eigvals[k]))*np.conj( vecs[J][k])
            InvSqrt[I][J]=cdum
    def print_matrix(matr,description,fpath,Append=False,format="%14.6f"):
        fdOU=open(fpath,("a" if(Append) else "w"))
        print("#"+description+" "+str(datetime.datetime.now()),file=fdOU)
        Ndim=np.shape(matr)
        for i in range(Ndim[0]):
            for j in range(Ndim[1]):
                print( format%(matr[i][j].real)+" "+format%(matr[i][j].imag),end="    ",file=fdOU)
            print("\n",end="",file=fdOU)
        fdOU.close()
    if(dbgng):
        dagdevTOL=1.0e-6;ofddevTOL=1.0e-6;devsumTOL=1.0e-6
        dagdevWRN=1.0e-7;ofddevWRN=1.0e-7;devsumWRN=1.0e-7

        dum=np.matmul( InvSqrt, np.matmul( S, InvSqrt) )
        devsum,dagdev,ofddev=deviation_from_unitmatrix( dum )
        ### assert (dagdev<1e-7 and ofddev <1e-7 and devsum<1e-6),"invSsqrt"
        warning=0
        if( dagdev< dagdevTOL and ofddev < ofddevTOL and devsum< devsumTOL):
            if( dagdev>=dagdevWRN or ofddev >=ofddevWRN or devsum>=devsumWRN):
                warning=1
        else:
            warning=-1
        if(warning != 0 ):
            print_matrix(dum,"Invsqrt X S X Invsqrt:dev=%e,%e,%e"%(dagdev,ofddev,devsum),"Invsqrt.log",Append=False,format="%12.4e")
            print_matrix(S,"S Invsqrt:dev=%e,%e,%e"%(dagdev,ofddev,devsum),"SAO.log",Append=False,format="%14.6e")
        if( Moldyn_static_.Ssqrt_dev_max_[0] < dagdev or Moldyn_static_.Ssqrt_dev_max_[1] < ofddev ):
            print_00("#get_Sinvsqrt:dev:DAG%e OFD%e devsum%e"%(dagdev,ofddev,devsum),warning=warning)
            Moldyn_static_.Ssqrt_dev_max_[0]=max( Moldyn_static_.Ssqrt_dev_max_[0],dagdev )
            Moldyn_static_.Ssqrt_dev_max_[1]=max( Moldyn_static_.Ssqrt_dev_max_[1],ofddev )
        assert (dagdev<dagdevTOL and ofddev <ofddevTOL and devsum<devsumTOL),"invSsqrt"
    return InvSqrt;

def calc_mo_energies(md,rttddft,MO_Coeffs=None,tm1_AU=None,h1e_kOR1=None,DM1_kOR1=None,update_rttddft=True, Fock1_kOR1=None):
    if tm1_AU is None: tm1_AU= md.time_AU;
    if h1e_kOR1 is None: h1e_kOR1=md.calc_hcore(rttddft, time_AU=tm1_AU)
    if DM1_kOR1 is None: DM1_kOR1 = md.tdDM
    if MO_Coeffs is None: MO_Coeffs = md.tdMO
    if Fock1_kOR1 is None: Fock1_kOR1= get_FockMat( md, rttddft, tm1_AU=tm1_AU,h1e_kOR1=h1e_kOR1, DM1_kOR1=DM1_kOR1)
    nmult_MO=(2  if(md.spinrestriction=='U') else 1)
    nmult_eorb=nmult_MO
    nmult_Fock=nmult_MO
    nkpt=(1 if(not md.pbc) else md.nkpt)
    nMO =md.nMO
    assert nkpt is not None,""
    Ret=[]
    for sp in range(nmult_MO):
        FockMat=( Fock1_kOR1 if(nmult_Fock==1) else Fock1_kOR1[sp] )
        MOcoefs=( MO_Coeffs if(nmult_MO==1) else MO_Coeffs[sp] )
        buf=[]
        for kp in range(nkpt):
            Fock=( FockMat if(not md.pbc) else FockMat[kp] )
            MO=( MOcoefs if(not md.pbc) else MOcoefs[kp] )
            eorbs=np.zeros(nMO)
            for jmo in range(nMO):
                eorbs[jmo]= np.vdot( MO[:,jmo], np.matmul(Fock, MO[:,jmo]))
            buf.append(eorbs)
        if( not md.pbc):
            buf=buf[0]
        Ret.append(buf)
    if( nmult_MO == 1 ):
        Ret=Ret[0]

    dbgng=False
    if(update_rttddft):
        if( rttddft.mo_energy is not None ):
            assert i1eqb( np.shape(Ret), np.shape(rttddft.mo_energy) ),""+str( [ np.shape(Ret), np.shape(rttddft.mo_energy)])
        rttddft.mo_energy=Ret
    return Ret

def get_FockMat(md,rttddft,tm1_AU=None,h1e_kOR1=None,DM1_kOR1=None):
    if tm1_AU is None: tm1_AU= md.time_AU;
    if h1e_kOR1 is None: h1e_kOR1=md.calc_hcore(rttddft, time_AU=tm1_AU)
    if DM1_kOR1 is None: DM1_kOR1 = md.tdDM
    Fock1_kOR1 = rttddft.get_fock( h1e=h1e_kOR1,dm=DM1_kOR1 ) ## cycle =-1(default) and diis=None 
    return Fock1_kOR1

def get_Fermilvl(moldyn, rttddft, FockMat=None,MO_coefs=None,MO_occ=None, eOrbs=None):
    ## 
    if( MO_occ is None ):
        MO_occ = moldyn.mo_occ
    if( eOrbs is None ):
        eOrbs = get_eOrbs(moldyn,rttddft,FockMat=FockMat,MO_coefs=MO_coefs)
    return rttddft.get_fermi( mo_energy_kpts=eOrbs, mo_occ_kpts=MO_occ )



def get_eOrbs(moldyn,rttddft,FockMat=None,MO_coefs=None):
    pbc=moldyn.pbc
    if( FockMat is None ):
        print("#get_eOrbs:recalculating FockMat" if(moldyn.Fock_last is None) else \
              "#get_eOrbs:use existing FockMat"+str(np.shape(moldyn.Fock_last)))
        FockMat=( moldyn.Fock_last if( moldyn.Fock_last is not None ) else \
                  get_FockMat(moldyn, rttddft) )
    if( MO_coefs is None ):
        MO_coefs = moldyn.tdMO
    nmult_MO=(2 if(moldyn.spinrestriction=='U') else 1)
    nmult_Occ=(2 if(moldyn.spinrestriction!='R') else 1)
    nmult_Fock=(2 if(moldyn.spinrestriction=='U') else 1)
    nmult_eorb=(2 if(moldyn.spinrestriction=='U') else 1)
    eOrbs=[] 
    for sp in range(nmult_eorb):
        eorb1 = []
        FockM =( FockMat if(nmult_Fock==1) else FockMat[sp] )
        MOcof=( MO_coefs if(nmult_MO==1) else MO_coefs[sp] )
        
        nKpoints=(1 if(not pbc) else len(FockM))
        for kp in range(nKpoints):
            Fock=(FockM if(not pbc) else FockM[kp])
            MOs=(MOcof if(not pbc) else MOcof[kp])
            ndim_MOs=np.shape(MOs);nMO=ndim_MOs[1]
            eps=np.zeros([ nMO ] )
            for mo in range(nMO):
                eps[mo]=( np.vdot( MOs[:,mo], np.matmul( Fock, MOs[:,mo] )) ).real
            if( not pbc ):
                eorb1=eps
            else:
                eorb1.append(eps)
        if( nmult_eorb == 1 ):
            eOrbs=eorb1
        else:
            eOrbs.append(eorb1)
    return eOrbs
class Moldyn_static_:
    Bsets_=[]
    Ssqrt_dev_max_=[-1,-1]
    N_calc_gs_=0

class Moldyn:
    ## see serialize.serialize(tgt)
    ## static fields should either (i) start with Capital letter and end with __
    ##                             (ii)start with staticfield_
    Dic1__={"pbc":'b', "exp_to_discard":'f', "Nat":'i', "Rnuc":'D',
           "basis":'s', "pseudo":'s', "a":'D', "nKpoints":'I', "df":'s',
            "td_field":'o', "gauge_LorV":'s', "logger":'o',"cell_dimension":'i',
            "spin":'i', "charge":'i', "mesh":'I', "rcut":'d',
            "_canonicalMOs":'Z|D',"_canonicalEorbs":'D',"_GSfockMat":'D|Z',
            "mo_occ":'D', "tdDM":'Z', "tdMO":'Z', "_orth_tdMO":'b',
            "step":'i', "time_AU":'d', "_tempFockMat":'Z|D', "_t2":'d',
            "_tempOrbs":'Z|D', "_tempEngs":'D', "nAO":'i', "nMO":'i', "nkpt":'i'}
    Serializers__={"logger":Logger.serialize_logger,"_Aind_args":dic_to_string}
    Constructors__={"logger":Logger.construct_logger, "td_field":construct_tdfield,"_Aind_args":construct_Aind_args}
    """ Moldyn object.  1. Stores metadata for generating DFT object
                        2. Stores time-dependent parameters including Rnuc, tdMOs, 
                        3. Stores matrices for speed-up 
                        4. Generates a rtTDDFT object for given Rnuc
    Attributes
    ----------
    Nat : int
    Rnuc : np.array float[Nat][3] 
        Rnuc in the atomic unit 
    Vnuc : np.array float[Nat][3]
        Vnuc (nuclear velocities) in the atomic unit
    Symbols : list of str [Nat]
        Atomic symbols,  H, He, Li, Be, B, C, ... etc

    a : np.array float[3][3]
        Lattice vectors or None
    pseudo : 
    kpts : np.array float[nkpts][3]
        array of kvectors or None

    """
    nkpt_for_non_pbc_=1

    def __init__(self, xc='LDA,VWN', pbc=None,xyzf=None,basis=None,pseudo=None,a=None,nKpoints=None,df=None,exp_to_discard=None,\
                 td_field=None,gauge_LorV=None,logger=None,DFT="RKS",spin=None,charge=None,cell_dimension=None,mesh=None,rcut=None,
                 check_timing=None,cell_precision=None, fixednucleiapprox=None, calc_Aind=None, dt_AU=None, alpha_Aind_over_c=None,
                 Temperature_Kelvin=None, DIIS_params=None, levelshift=None, molecule=None, unitcell=None ):

        assert pbc is not None,"Moldyn#__init__: missing PBC input"
        assert (molecule is None) or (not pbc),"Moldyn#__init__: molecule input for PBC calc"
        assert (unitcell is None) or pbc,      "Moldyn#__init__: unitcell input for non-PBC calc"

        # 2024Mar: molecule or unitcell input (2)
        #  Since molecule/cell may contain basis and pseudo inside, we check them
        if( basis is None ):
            if( molecule is not None ):
                basis=molecule.basis;  
            elif( unitcell is not None ):
                basis=unitcell.basis;
            if( basis is not None ):
                print("Moldyn#__init__: basis set from molecule/unitcell:"+str(basis))
                assert isinstance(basis,str),"we only accept string format of basis"
        if( pseudo is None ):
            if( unitcell is not None ):
                pseudo=unitcell.pseudo
            if( pseudo is not None ):
                print("Moldyn#__init__: pseudo set from molecule/unitcell:"+str(pseudo))
                assert isinstance(pseudo,str),"we only accept string format of pseudo"


        assert ( basis is not None ),"Moldyn"
        assert ( (td_field is None) or \
                 ( (gauge_LorV is not None) and ( (gauge_LorV=='L') or (gauge_LorV=='V'))) ),"Moldyn"

        rttddft_common.Setup()

        ## PhysicalConstants_=get_constants();
        ## BOHRinANGS = PhysicalConstants_['Bohr']
        BOHRinANGS=PhysicalConstants.BOHRinANGS()

        self.check_timing=check_timing

        # Get R,Sy,dict['a'](lattice vectors)  in ANGS ...
        dict=(None if (not pbc) else {"a":None})
        if( xyzf is not None ):
            R_ANGS, self.Symbols = read_xyzf(xyzf,sbuf=None,dict=dict)
        elif( molecule is not None ):
            assert isinstance(molecule,Mole),"Moldyn#__init__ input molecule.02"
            R_ANGS, self.Symbols = read_mol_or_cell(molecule,dict=dict,outputunit='A')
        elif( unitcell is not None ):
            assert isinstance(unitcell,Cell),"Moldyn#__init__ input unitcell.02"
            R_ANGS, self.Symbols = read_mol_or_cell(unitcell,dict=dict,outputunit='A')
        else:
            assert False,"xyzf or molecule or unitcell"

        if(pbc):
            assert dict['a'] is not None,""

        # -------------------------------------------------
        ## self.dft = ( DFT_RKS if( DFT == "RKS") else ( DFT_ROKS if (DFT=="ROKS") else ( DFT_UKS if(DFT=="UKS") else None)))
        self.IZnuc_eff=None  ## 2021.12.25 auxiliary field for nucdyn 
                             
        self.spinrestriction=('R' if( DFT=="RKS") else ( 'O' if (DFT=="ROKS") else ( 'U' if(DFT=="UKS") else None)))
        ### assert False,"DFT:"+DFT
        assert (self.spinrestriction is not None),"wrong DFT:"+DFT
        self.xc=xc
        self.pbc = pbc
        self.exp_to_discard=exp_to_discard
        self.Nat = len( self.Symbols )
        self.Rnuc = np.zeros( [self.Nat,3],dtype=float)
        for I in range(self.Nat):
            for J in range(3):
                self.Rnuc[I][J] = R_ANGS[I][J]/BOHRinANGS
        self.basis = basis
        self.pseudo = pseudo
        self.a = None
        self.nKpoints = None
        self.df = None
        self.td_field = td_field
        self.gauge_LorV = gauge_LorV
        self.logger = logger
        self.cell_dimension = cell_dimension
        self.cell_precision = cell_precision
        self.spin=spin  ## 2020.12.16 for open-shell
        self.charge=charge
        print_00("#Moldyn:spin=",end="");print_00(self.spin)
        self.mesh=mesh
        self.rcut=rcut
        self._canonicalMOs=None;    self._cMOs_Rnuc=None
        self._canonicalEorbs=None
        self._GSfockMat=None

        self._calc_Aind=calc_Aind; self._dt_AU=dt_AU; 
        self._alpha_Aind_over_c=( None if(calc_Aind==0) else\
                                  ( alpha_Aind_over_c if(alpha_Aind_over_c is not None) else 1.0) )
        self._Aind_args=None          ### 2021.07.28: only for saving/loading ...

        if( self._calc_Aind is None ):
            self._calc_Aind=0
        printout("#Moldyn_00:calc_Aind:%d "%(self._calc_Aind),fpath="Aind_over_c.log",Append=False)
        
        self.Temperature_au= None
        if(Temperature_Kelvin is not None):
            HARTREEinEV=27.211386024367243
            KelvinINeV=8.6173324e-5
            temp_eV=Temperature_Kelvin*KelvinINeV
            temp_au=temp_eV/HARTREEinEV
            self.Temperature_au=temp_au
        self.levelshift=levelshift
        self.DIIS_params=DIIS_params
        

        self._DIIS=None

        self.hcore_last=None
        self.vhf_last=None
        self.Fock_last=None
        self.tmAU_hcore_last=None
        self.tmAU_Fock_last=None
        self.tmAU_vhf_last=None
        self._fixednucleiapprox=fixednucleiapprox;    ## 20210610:fixednucleiapprox  None by default 

                           ##                     RKS                              ROKS       UKS 
                           ##                     pbc                 non-pbc      
        self.mo_occ=None   ##                     [nkpt][nMO]      or [nMO]        ...        [2]...

        self.tdDM = None   ## Complex DM          [nkpt][nAO][nAO] or [nAO][nAO]   [2]...     [2]...
        self.tdMO = None   ## Complex MO coefs..  [nkpt][nAO][nMO] or [nAO][nMO]   ...        [2]...
        self._orth_tdMO = None  ## True if tdMO is multiplied by Ssqrt ...

        self.step=0
        self.time_AU = 0.0

        self._tempFockMat=None
        self._t2=None
        self._tempOrbs=None; self_tempEngs=None
        self.nAO=None
        self.nMO=None
        self.nkpt=None
        #---------------------------------------

        if( pbc ):
            self.a = ( np.array(a).copy() if(a is not None) else dict["a"])  ## I leave this as is but we use .BravaisVectors_au for our calculations
            self.BravaisVectors_au = (None if(self.a is None) else self.a/BOHRinANGS)  ## 2021.07.29: when we load Moldyn from .pscf file, we first make an empty Moldyn object
            
        if( nKpoints is not None):
            self.nKpoints = nKpoints.copy() ## [2,2,2] etc.
        if( df is not None ):
            self.df = df                    ## "GDF" or ...

    def gen_rttddft(self, mol_or_cell=None, Rnuc=None,unit=None, nKpoints=None, td_field=None, 
                    gauge_LorV=None, logger=None, set_occ=None, xc=None, check_timing=None, set_timeAU=True, kpts=None):
        """ generates an rtTDDFT object  for given Rnuc (default: self.Rnuc)"""

        retv=self.gen_rttddft_( mol_or_cell=mol_or_cell, Rnuc=Rnuc, unit=unit, nKpoints=nKpoints, td_field=td_field, 
                    gauge_LorV=gauge_LorV, logger=logger, set_occ=set_occ, xc=xc, check_timing=check_timing, 
                    set_timeAU=set_timeAU, kpts=kpts)
        if( kpts is not None ):
            print("#gen_rttddft:",len(kpts),len( np.reshape(retv.kpts, (-1,3))))
            assert len(kpts) == len( np.reshape(retv.kpts, (-1,3))),""

        Rnuc_au = None
        if( Rnuc is None ):
            Rnuc_au = self.Rnuc
            if( mol_or_cell is None ):
                mol_or_cell=retv.mol
                Rnuc_mol_or_cell,Sy=parse_xyzstring( mol_or_cell.atom, unit_BorA='B')
                dev = max( np.ravel(Rnuc_mol_or_cell) - np.ravel(Rnuc_au))
                assert dev < 2e-6,""+str(Rnuc_mol_or_cell)+"/"+str(Rnuc_au)
        else:
            if(unit=='B'):
                Rnuc_au=Rnuc
            else:
                Rnuc_au=np.array(Rnuc)/(physicalconstants.BOHRinANGS)
        print("#gen_rttddft:Rnuc_au:%f "%(self.time_AU)+d1toa(np.ravel(Rnuc_au)) )
        Rnuc_mol,Sy=parse_xyzstring( retv.mol.atom, unit_BorA='B')
        dev1=max( np.ravel( Rnuc_mol ) - np.ravel( Rnuc_au ) )
        if( self.pbc and retv.with_df is not None ):
            df=retv.with_df
            Rnuc_dfcell,Sy=parse_xyzstring( df.cell.atom, unit_BorA='B')

            dev2=max( np.ravel( Rnuc_dfcell ) - np.ravel( Rnuc_au ) )
            print("#gen_rttddft:Rnuc:dev:%e %e"%(dev1,dev2));
            if( dev1 > 1e-6 or dev2 > 1e-6 ):
                Xnuc_mol =np.ravel( Rnuc_mol )
                Xnuc     =np.ravel( Rnuc_au )
                Xnuc_dfcell=np.ravel( Rnuc_dfcell )
                le=len(Xnuc)
                print("# no   retv.mol.atom  Rnuc   retv.df.cell.atom");
                for j in range(le):
                    print(" %3d      %14.6f  %14.6f  %14.6f"%(j,Xnuc_mol[j],Xnuc[j],Xnuc_dfcell[j]))
                

            assert dev1<2e-5,""; assert dev2<2e-5,"" 
        else:
            print("#gen_rttddft:Rnuc:%e"%(dev1));assert dev1<1e-6,"";
        return retv
    def gen_rttddft_(self, mol_or_cell=None, Rnuc=None,unit=None, nKpoints=None, td_field=None, 
                    gauge_LorV=None, logger=None, set_occ=None, xc=None, check_timing=None, 
                    set_timeAU=True, kpts=None, auxbasis=None):

        """ a private function, generates an rtTDDFT object  for given Rnuc (default: self.Rnuc)"""
        
        docheck_timing=False
        if( check_timing is not None ):
            docheck_timing=check_timing
        else:
            if(self.check_timing is not None ):
                docheck_timing=self.check_timing

        if(docheck_timing):
            assert self.pbc,"not yet implemented "

        Rnuc_au=None
        if( Rnuc is not None):
            assert (unit =='A' or unit=='B'),"unit:"+unit
            if( unit=='B'):
                Rnuc_au=Rnuc
            else:
                Rnuc_au=arrayclone(Rnuc);Rnuc_au = Rnuc_au/physicalconstants.BOHRinANGS
        else:
            Rnuc_au=self.Rnuc
        if mol_or_cell is None: mol_or_cell = self.gen_mol_or_cell(Rnuc_au);
        if nKpoints is None: nKpoints = self.nKpoints;
        if td_field is None: td_field = self.td_field;
        if gauge_LorV is None: gauge_LorV = self.gauge_LorV;
        if xc is None: xc=self.xc;
        Logger.write(self.logger,"#gen_rttddft");
        Logger.Check_meshsize(mol_or_cell)
        if( td_field is None):
            assert (gauge_LorV is None),"Gauge should be None"
        else:
            assert ( (gauge_LorV is not None) and ( (gauge_LorV == 'L') or (gauge_LorV == 'V') ) ),"gauge"

        dipolegauge=False
        if( gauge_LorV is not None):
            dipolegauge = (gauge_LorV == 'L')

        ret=None
        if( not self.pbc ):
            if( self.spinrestriction == 'R' ):
                ret=rttddftMOL(mol_or_cell,td_field=td_field,dipolegauge=dipolegauge,xc=xc)
            elif( self.spinrestriction == 'O' ):
                ret=rttddftMOL_ROKS(mol_or_cell,td_field=td_field,dipolegauge=dipolegauge,xc=xc)
            elif( self.spinrestriction == 'U' ):
                ret=rttddftMOL_UKS(mol_or_cell,td_field=td_field,dipolegauge=dipolegauge,xc=xc)
            else:
                assert False,"unknown DFT:"+self.spinrestriction
            ##sprint(type(ret))
        else:
            # print_00("#Moldyn.gen_rttddft:kpts cell.make_kpts(%s):"%(str(nKpoints))+str(mol_or_cell.make_kpts(nKpoints)))
            if(kpts is None ):
                kpts = mol_or_cell.make_kpts(nKpoints)
            else:
                print("#use given kpts:",kpts);

            if( self.spinrestriction == 'R' ):
                ### print_00("nKpoints:",end="");print_00(nKpoints)
                if( docheck_timing ):
                    ret=rttddftPBC_timing(mol_or_cell,kpts=kpts,
                               td_field=td_field,dipolegauge=dipolegauge,xc=xc)
                else:                        
                    ret=rttddftPBC(mol_or_cell,kpts=kpts,
                               td_field=td_field,dipolegauge=dipolegauge,xc=xc)
            elif( self.spinrestriction == 'O' ):
                ret=rttddftPBC_ROKS(mol_or_cell,kpts=kpts,
                               td_field=td_field,dipolegauge=dipolegauge,xc=xc)
            elif( self.spinrestriction == 'U' ):
                ret=rttddftPBC_UKS(mol_or_cell,kpts=kpts,
                               td_field=td_field,dipolegauge=dipolegauge,xc=xc)
        if(self.pbc):
            ret.set_calc_Aind(self._calc_Aind, self._alpha_Aind_over_c); ret._calc_Aind = self._calc_Aind; 
            ret._dt_AU=self._dt_AU; ret.BravaisVectors_au=self.BravaisVectors_au
            rttddft_common.Write_once("Moldyn.calc_Aind", "#gen_rttddft:calc_Aind:%d dt_AU:%s"%(self._calc_Aind,str(ret._dt_AU)),\
                                    filename="Aind")

        if(self._fixednucleiapprox is not None ):             ## 20210610:fixednucleiapprox 
            ret._fixednucleiapprox = self._fixednucleiapprox  
            print_00("#gen_rttddft:fixednucleiapprox:%r"%(ret._fixednucleiapprox))

        ret.set_nOrbs( nkpt=self.nkpt, nAO=self.nAO, nMO=self.nMO)
        dbgng=True;fdLog=sys.stdout
        if(dbgng):
            fdLog=open_00("gen_rttddft.log","a")
            print("job:"+str(rttddft_common.get_job(True))+"ret:"+str(ret))
            print_00("#gen:"+str(type(ret))+" \t"+rttddft_common.get_job(True)+" \t\t"+str(datetime.datetime.now()),file=fdLog)
        if( self.tdMO is not None ):
            ret.mo_coeff=arrayclone( self.tdMO, dtype=np.complex128 )
            print_00("mo_coeff set:"+zNtoa(ret.mo_coeff,Nlimit=5),file=fdLog)
        if( set_occ is not None ):
            if(set_occ):
                ret.set_mo_occ( self.mo_occ,fix_occ=True,clone=False)
                print_00("#mo_occ set:"+dNtoa(self.mo_occ),file=fdLog)
        if(dbgng):
            close_00(fdLog); ##fdLog.close()

        if( logger is not None):
            ret.set_logger(logger)

        if( self.pbc ):
            assert ( self.df is not None ),"df"
        
        if( self.df is not None ):
            if( self.df == "GDF" ):
                gdf=df.GDF( mol_or_cell );
                if( nKpoints is not None):
                    if( kpts is None ):
                        kpts = mol_or_cell.make_kpts(nKpoints)
                    else:
                        kpts=arrayclone(kpts)
                    gdf.kpts= kpts  ## NEVER FORGET THIS...

                
                printout("#GDF:"+" \t\t"+str(rttddft_common.get_job(True)),fpath="df.log",Append=True,dtme=True)
                if( auxbasis is not None):
                    printout("#GDF:gen_rttddft:auxbasis.org:"+str(gdf.auxbasis)+"_%d"%( gdf.get_naoaux() ),
                        fpath="df.log",Append=True,dtme=True,stdout=True)
                    gdf.auxbasis=auxbasis
                    printout("#GDF:gen_rttddft:auxbasis.new:"+str(gdf.auxbasis)+"_%d"%( gdf.get_naoaux() ),
                        fpath="df.log",Append=True,dtme=True,stdout=True)
                else:
                    printout("#GDF:gen_rttddft:auxbasis:"+str(gdf.auxbasis)+"_%d"%( gdf.get_naoaux() ),
                        fpath="df.log",Append=True,dtme=True)

                ret.with_df=gdf
            elif( self.df == "FFTDF" ):
                fftdf = df.FFTDF(mol_or_cell)
                if( nKpoints is not None):
                    if( kpts is None ):
                        kpts = mol_or_cell.make_kpts(nKpoints)
                    else:
                        kpts=arrayclone(kpts)
                    fftdf.kpts= kpts  ## NEVER FORGET THIS...
                ret.with_df=fftdf
                printout("#FFTDF:"+str(fftdf.mesh)+" \t\t"+str(rttddft_common.get_job(True)),fpath="df.log",Append=True,dtme=True)
            elif( self.df == "MDF" ):
                if( kpts is None ):
                    kpts = mol_or_cell.make_kpts(nKpoints)
                mdf = df.MDF(mol_or_cell,kpts=kpts)
                ret.with_df= mdf
                print("mdf.kpts:",mdf.kpts)
                printout("#MDF:"+str(mdf.mesh)+ " auxbasis:"+str(mdf.auxbasis)+"_%d"%( mdf.get_naoaux() )
                         +" \t\t"+str(rttddft_common.get_job(True)),fpath="df.log",Append=True,dtme=True)
                
            else:
                assert False,"unimplemented"
        if( self._calc_Aind != 0 ):
            if( self._Aind_args is not None):
                rttddft_set_Aind_args(ret, self._Aind_args)
            else:
                print("#Aind_args is None.."); ## TODO: XXX XXX in future this SHOULD lead abortion 

        if( set_timeAU ):
            ret._time_AU=self.time_AU
            if( self._calc_Aind != 0 and (ret._Aind_tmAU is not None) ):
                tdiff=abs( ret._time_AU - ret._Aind_tmAU)
                assert tdiff<1.0e-6,"tmAU differs:%f %f %f"%(self.time_AU,ret._time_AU,ret._Aind_tmAU)
                printout("#set_timeAU:%f %f %f"%(self.time_AU,ret._time_AU,ret._Aind_tmAU),fpath="Aind_tmAU.log",Append=True,dtme=True)

        #2021.12.25 addons ----
        if( self.Temperature_au is not None):
            rttddft_common.Write_once("Moldyn.Temperature","smearing:%f au"%(self.Temperature_au),filename="rttddft")
            ret = pyscf_pbc_scf.addons.smearing_(ret, sigma=self.Temperature_au, method='fermi')
        if( self.levelshift is not None ):
            rttddft_common.Write_once("Moldyn.levelshift","levelshift:%f au"%(self.levelshift),filename="rttddft")
            ret.level_shift=self.levelshift
        if( self.DIIS_params is not None ):
            algorithm= self.DIIS_params["algorithm"]
            diis_space=( 6 if("diis_space" not in self.DIIS_params) else self.DIIS_params["diis_space"] )
            diis_space_rollback = ( False if( "diis_space_rollback" not in  self.DIIS_params) else \
                            self.DIIS_params["diis_space_rollback"] )
            rttddft_common.Write_once("Moldyn.DIIS","DIIS:%s %d %r"%(algorithm,diis_space,diis_space_rollback),filename="rttddft")
            if( algorithm == "ADIIS" ):
                ret.diis=ADIIS(ret, ret.diis_file)
            elif( algorithm == "EDIIS" ):
                ret.diis=EDIIS(ret, ret.diis_file)
            elif( algorithm == "CDIIS" ):
                ret.diis=CDIIS(ret, ret.diis_file)  ## this is the default choice
            elif( algorithm == "DIIS_dmat" ):
                ret.diis=DIIS_dmat(6,PBC,scheme="SDFFDS")
            else:
                assert False,"unknown DIIS"+str(params["DIIS"])
            ret.diis.diis_space = diis_space;
            ret.diis.diis_space_rollback = diis_space_rollback
        ### heapcheck("Moldyn.gen_mol_or_cell:AF")
        return ret

    def gen_mol_or_cell( self, Rnuc=None, unit='B', spin=None, charge=None ):
        assert (unit=='B'),"gen_mol_or_cell"
        if Rnuc is None:Rnuc=self.Rnuc;
        if spin is None:spin=self.spin;
        if charge is None:charge=self.charge;
        
        if( not self.pbc ):
            ret = molgto.Mole()
        else:
            ret = gto.Cell()

        if( self.mesh is not None):
            ret.mesh=self.mesh
            ### print_00("#cell.mesh set:",end="");print_00(self.mesh)
        if( self.rcut is not None):
            ret.rcut=self.rcut
            print_00("#cell.rcut set:",end="");print_00(self.rcut)

        if( self.exp_to_discard is not None):
            ret.exp_to_discard = self.exp_to_discard

        if( self.cell_dimension is not None ):
            ret.dimension = self.cell_dimension
        if( charge is not None ):
            ret.charge=charge;
        if( self.cell_precision is not None ):
            ret.precision=self.cell_precision;
        ret.atom = write_xyzstring(Rnuc,self.Symbols,input_unit='B',output_unit='A')
        ret.basis = self.basis

        if( self.pseudo is not None):
            ret.pseudo = self.pseudo

        if( self.pbc ):
            ret.a = self.a  ### this should be an intrinsic property so I leave it .. but let's use .BravaisVectors_au for our calculations
            ret.BravaisVectors_au = self.BravaisVectors_au
            kpts=ret.make_kpts(self.nKpoints)
            rttddft_common.Write_once("Moldyn.kpoints","#Moldyn:kpts cell.make_kpts(%s):"%(str(self.nKpoints))+str(kpts),
                                     filename="Moldyn")
        
        if( spin is not None ):
            ret.spin=spin
        ret.build()

        if( self.nAO is None ):
            self.nAO=check_nao_nr(ret)
            print_00("#Moldyn.nAO:%d"%(self.nAO))
        
        if( self.basis not in Moldyn_static_.Bsets_):
            comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank();MPIsize=comm.Get_size()
            if( MPIsize < 2 or MPIrank==0 ):
                fnme=rttddft_common.get_job(True)+"_Bset.dat"
                print_bset(fnme,ret,Append=(len(Moldyn_static_.Bsets_)>0))
                
            Moldyn_static_.Bsets_.append(self.basis)
        return ret

# i) use existing tdDM : dm=self.tdDM,(update_tdDM=None)
# ii)recalculate tdDM and update md.tdDM : dm=None,(update_tdDM=None)
#                     but no update        dm=None update_tdDM=False
    def calc_energy_tot(self,rttddft,dm=None,h1e=None,vhf=None,time_AU=None,update_tdDM=None,dict=None,
                        filename=None, filename_refr=None):
        
        from .utils import update_dict,printout_dict
        import time
 
        h1eNone=(h1e is None);vhfNone=(vhf is None)
        fncnme='calc_energy_tot';fncdepth=5   ## fncdepth ~ SCF:0 vhf:1 get_j:2
        Wctm000=time.time();Wctm010=Wctm000;dic1_wctm={}

        if time_AU is None:time_AU=self.time_AU
        if update_tdDM is None:update_tdDM=True
        if (dm is None):
            print_00("#calc_energy_tot:dm is None..");
            dm=self.calc_tdDM(rttddft, ovrd=update_tdDM )
        ### if dm is None: dm=self.calc_tdDM(rttddft, ovrd=update_tdDM )
        # ??? 2021.05.14 commented out print("#calc_energy_tot:shape of dm:",end="");print(np.shape(dm))
        if h1e is None:h1e=self.calc_hcore(rttddft,time_AU)
        if( h1eNone ):
            Wctm020=Wctm010;Wctm010=time.time();update_dict(fncnme,dic1_wctm,Dic_wctm,"h1e",Wctm010-Wctm020,depth=fncdepth)
        if vhf is None:vhf=self.calc_veff(rttddft,dm=dm)
        if( vhfNone ):
            Wctm020=Wctm010;Wctm010=time.time();update_dict(fncnme,dic1_wctm,Dic_wctm,"vhf",Wctm010-Wctm020,depth=fncdepth)


        
       # hf.py:  def energy_elec(mf, dm=None, h1e=None, vhf=None):
        # rks.py: def energy_elec(ks, dm=None, h1e=None, vhf=None):
        # krks.py:def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None)
        ec_plus_exc=None
        dbgng=True
        if(dbgng):
            ec=getattr(vhf, 'ecoul', None)
            exc=getattr(vhf, 'exc', None)
            # print("vhf.ecoul:",end="");print(ec)
            # ec1=getattr(vhf[1], 'ecoul', None)
            # print("vhf1.ecoul:",end="");print(ec1)
            if( (ec is not None) and (exc is not None) ):
                ec_plus_exc=ec+exc
                if( dict is not None):
                    dict.update({'ecoulxc':ec_plus_exc})
            nkpt=(1 if(not self.pbc) else self.nkpt)
            eh1=0.0
            nmult=(2 if(self.spinrestriction=='U') else 1);
            ## h1e[nkp][nAO][nAO] / h1e[nAO][nAO]
            for kp in range(nkpt):
                h1e_k= ( h1e if(not self.pbc) else h1e[kp])
                if(self.spinrestriction=='U'):
                    eh1+=np.einsum('ij,ji->', h1e_k, ( dm[0] if(not self.pbc) else dm[0][kp] ))\
                      + np.einsum('ij,ji->', h1e_k, ( dm[1] if(not self.pbc) else dm[1][kp] ))
                else:
                    eh1+=np.einsum('ij,ji->', h1e_k,( dm if(not self.pbc) else dm[kp]))
            if( self.pbc ):
                eh1 = eh1/float(nkpt)
            if( dict is not None):
                dict.update({'ekin':eh1})
            Wctm020=Wctm010;Wctm010=time.time();Wctm_einsum=Wctm010-Wctm020
            update_dict(fncnme,dic1_wctm,Dic_wctm,"einsum_h1e_k",Wctm010-Wctm020,depth=fncdepth)  
            if( self.pbc ):
                test=np.einsum('kij,kji->',h1e,dm,optimize="greed");test/=float(nkpt)
                Wctm020=Wctm010;Wctm010=time.time();Wctm_OptEinsum=Wctm010-Wctm020
                print("calc_eng:OptEinSum:%f elapsed:%f / old:%f elapsed:%f  diff %e"%(test,Wctm_OptEinsum,\
                        eh1,Wctm_einsum,abs(eh1-test)))
                    

        Eel,eCoulXC = rttddft.energy_elec(dm,h1e,vhf)  ## note: 
        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1_wctm,Dic_wctm,"rttddft.energy_elec",Wctm010-Wctm020,depth=fncdepth)  
        if( ec_plus_exc is not None ):
            assert abs(eCoulXC-ec_plus_exc)<1.0e-5,""
        else:
            if( dict is not None):
                dict.update({'ecoulxc':eCoulXC})

        ec=getattr(vhf, 'ecoul', None)
        print_00("vhf.ecoul:",end="");print_00(ec)
        e1=rttddft.scf_summary['e1']
#Moldyn:calc_energy_tot:eCoulXC:-16.630561609700713 / -16.630561609700713 ekin:(25.370127214648235+0j)/25.370127214648235
        print_00("Moldyn:calc_energy_tot:eCoulXC:"+str(ec_plus_exc)+" / "+str(eCoulXC)+" ekin:"+str(eh1)+"/"+str(e1))

        
        Enuc = rttddft.energy_nuc()

        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1_wctm,Dic_wctm,"rttddft.energy_nuc",Wctm010-Wctm020,depth=fncdepth)  
        printout_dict(fncnme,dic1_wctm,Dic_wctm,Wctm010-Wctm000,depth=fncdepth)

        if( filename is not None ):
            comment="Moldyn.calc_energy_tot:%f,%f,%f,%f  \t\t"%( Eel+Enuc, Enuc, eCoulXC, Eel-eCoulXC)\
                     + rttddft_common.get_job(True) + str(self.step)
            svld_aNbuf('S',filename+"_dm.dat",buf=dm,comment=comment)
            svld_aNbuf('S',filename+"_vhf.dat",buf=vhf,comment=comment)
        if( filename_refr is not None ):
            sbuf=[]
            dmREF=svld_aNbuf('L',filename_refr+"_dm.dat",get_comments=sbuf)
            vhfREF=svld_aNbuf('L',filename_refr+"_vhf.dat")
            dev1=aNmaxdiff(dmREF,dm);dev2=aNmaxdiff(vhf,vhfREF)
            print("#CompTo:"+sbuf[0])
            print("#dmDIFF:%e"%(dev1))
            print("#vhfDIFF:%e"%(dev2))
            diff_aNbuf(dm,dmREF)

        return Eel+Enuc,Enuc,eCoulXC,Eel-eCoulXC   ## EelSum,  ncCoul,elCoul,elKE

    def svld_energy(self,SorL,pscfpath,eDict=None,Threads=[0]):
        ## o Checks Energy expectation value at saving/loading Moldyn
        ## o 
        ## 
        comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        fname=pscfpath.replace(".pscf","")
        fname_logf=fname+"_recalc_eng.log"
        if( SorL == 'L'):
            if( not os.path.exists(fname_logf) ):
                rttddft_common.Assert(False,"#!W missing logfile:"+fname_logf,1) ## 2022.03.03 we do not stop here
                return None

        rttddft=self.gen_rttddft( Rnuc=self.Rnuc, unit='B')
        Eel,ncCoul,elCoul,elKE=self.calc_energy_tot(rttddft,filename=fname+"_recalc_eng")
        if( MPIsize<2 or (MPIrank in Threads) ):
            print("#Moldyn.load:Energies:%16.8f %16.8f %16.8f %16.8f"%(Eel,ncCoul,elCoul,elKE))
            if( eDict is not None):
                eDict.update({"Eel":Eel,"ncCoul":ncCoul,"elCoul":elCoul,"elKE":elKE}) 
            if( SorL == 'S' ):
                fd=open(fname_logf,"w")
                print("%6d %14.6f    %16.8f %16.8f %16.8f %16.8f"%(self.step, self.time_AU, Eel,ncCoul,elCoul,elKE),file=fd)
                fd.close()
                return None
            else:
                dev=None
                fd=open(fname_logf,"r");ok=False
                for line in fd:
                    line=line.strip()
                    sA=line.split();nA=len(sA)
                    if(nA>=6):
                        step=int(sA[0]); tm_AU=float(sA[1]); Eel_1=float(sA[2]); ncCoul_1=float(sA[3]);
                        elCoul_1=float(sA[4]);  elKE_1=float(sA[5]);
                        diffs= [ abs(Eel-Eel_1), abs(ncCoul-ncCoul_1), abs(elCoul-elCoul_1), abs(elKE-elKE_1) ]
                        dev=max(diffs)
                        print("#Moldyn.load:energy deviations:"+str(diffs))
                        print("#Moldyn.load:Energies:CALC:%16.8f %16.8f %16.8f %16.8f"%(Eel,ncCoul,elCoul,elKE)) 
                        print("#Moldyn.load:Energies:READ:%16.8f %16.8f %16.8f %16.8f"%(Eel_1,ncCoul_1,elCoul_1,elKE_1)) 

                        ok=True
                fd.close()
                assert ok,"failed reading from:"+fname_logf  ## TODO PLS WEAKEN THIS ASSERTION
                return dev
    def save(self,fpath,delimiter='\n',comment=None,Threads=[0],rttddft=None,caller=None, Barrier=False):
        comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( self._calc_Aind != 0 ):
            dbgng=True ## XXX XXX
            assert rttddft is not None,"PLS set rttddft in save method !!"
            self._Aind_args={}; rttddft_get_Aind_args(rttddft,self._Aind_args)
            if(dbgng):
                fd1=open("Aind_over_c_SAVE.log","w");print(self._Aind_args,file=fd1);fd1.close()
        wt0=time.time(); wt1=wt0; wt_ser=0; wt_sync=0
        STRcaller=("" if(caller is None) else caller),
        print("#moldyn.save:$%02d:%s start \t\t %s"%( MPIrank, STRcaller, str(datetime.datetime.now())),flush=True)
        serialize(self,fpath=fpath,delimiter=delimiter,comment=comment,serializers=Moldyn.Serializers__,
                  Threads=Threads,return_None=True)
        wt2=wt1; wt1=time.time(); wt_ser=wt1-wt2
        print("#moldyn.save:$%02d:%s end elapsed:%f \t\t %s"%( MPIrank, STRcaller, wt_ser,str(datetime.datetime.now())),flush=True)
        if( (wt_ser > 30.0) and (MPIsize<2 or MPIrank==0) ):
            fd1=open("Moldyn_save.log","a")
            print("#Moldyn.save:$%02d:%s elapsed:%f  \t\t "%( MPIrank, STRcaller, wt_ser)+str(rttddft_common.get_job(True))+" \t\t %s"%(str(datetime.datetime.now())),file=fd1)
            fd1.close()

        if( Barrier and MPIsize > 1 ):
            comm.Barrier()
            wt2=wt1; wt1=time.time(); wt_sync=wt1-wt2
        self.svld_energy('S',fpath)
        self._Aind_args=None;
        print("#moldyn.save:$%02d:%s DONE %f(ser:%f,sync:%f)\t\t %s"%( MPIrank, STRcaller, 
               wt_ser+wt_sync, wt_ser,wt_sync, str(datetime.datetime.now()) ),flush=True)
    @staticmethod
    def load(fpath,eDict=None):
        BOHRinANGS = physicalconstants.BOHRinANGS;
        types,values=parse_file(fpath)
        assert "Rnuc" in values,""; assert "Symbols" in values,""; assert "pbc" in values,""
        strRnuc_au=values["Rnuc"]
        if( strRnuc_au.startswith("!") ):
            get_comments=[];fnme1=strRnuc_au[1:]
            Rnuc_au=svld_aNbuf("L",fnme1,get_comments=get_comments)
            Rnuc_au=np.ravel(Rnuc_au);le=len(Rnuc_au);Nat=le//3;assert le==3*Nat,"le=%d"%(le)
            Rnuc_ANGS = np.reshape( [ Rnuc_au[k]*BOHRinANGS for k in range(le) ], [Nat,3] )
            fpath_ref=get_comments[0];
            if(fpath_ref[0]=="#"):
                fpath_ref=(fpath_ref[1:]).strip()
            assert fpath_ref==fpath,"FILE:%s says it belongs to %s %d, not %s %d"%(fnme1,fpath_ref,len(fpath_ref),fpath,len(fpath))
        else:
            strbuf=strRnuc_au.split();le=len(strbuf);Nat=le//3;assert le==3*Nat,"le=%d"%(le)
            Rnuc_au = [ float(strbuf[k]) for k in range(le) ]
            Rnuc_ANGS = np.reshape( [ float(strbuf[k])*BOHRinANGS for k in range(le) ], [Nat,3] )
        Sy= values["Symbols"].split(); assert (len(Sy)==Nat),""
        pbc=eval( values["pbc"] )
        LatticeVectors=None
        if( pbc ):
            assert "a" in values,"";
            LatticeVectors=values["a"]
            if( LatticeVectors.startswith("!") ):
                get_comments=[]
                LatticeVectors= np.ravel( svld_aNbuf("L",LatticeVectors[1:],get_comments=get_comments) )
            else:
                if( isinstance(LatticeVectors,str)):
                    sA=LatticeVectors.split();nA=len(sA)
                    LatticeVectors=[ float(sA[k]) for k in range(nA) ]
        xyzf=make_tempfile(head="tempwks_delete_on_exit",tail=".xyz")
        write_xyzf(xyzf,Rnuc_ANGS,Sy,description="temporarily created for Moldyn.load",LatticeVectors=LatticeVectors)

        basis=values["basis"]
        retv= Moldyn(pbc=pbc,xyzf=xyzf, basis=basis)
        ## print("#Moldyn.load:constructed. load_fromfile...",flush=True)
        load_fromfile( retv, fpath, constructors=Moldyn.Constructors__, logfile="Moldyn_loader.log")
        print("#Moldyn.load:...load END:Gauge:%s:tdF:"%(retv.gauge_LorV)+str(retv.td_field),flush=True)
        dbgng=False
        if(dbgng):
            print("#Moldyn.load:testing REserialization...",flush=True)
            serialize( retv, fpath="Moldyn_load.tmpwks", serializers=Moldyn.Serializers__)
            os.system("diff "+fpath+" Moldyn_load.tmpwks")
            print("#Moldyn.load:check diff...",flush=True)
            ### assert False,""
        if( retv._calc_Aind != 0 ):
            dbgng=True ## XXX XXX
            if(dbgng):
                comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();MPIsize=comm.Get_size()
                if(MPIsize<2 or MPIrank==0):
                    fd1=open("Aind_over_c_LOAD.log","w");print(retv._Aind_args,file=fd1);fd1.close()

        E_dev = retv.svld_energy("L",fpath,eDict=eDict)
        if(E_dev is not None):
            print("#Moldyn.load:deviation:%e"%(E_dev));assert abs(E_dev)<1.0e-6,""
        else:
            rttddft_common.Assert(False,"#Moldyn.load:missing _recalc_eng.log file for pscf file:"+fpath,1)
        return retv

    def check_matrixShape(self,name,A,nkpt=None,nAO=None,nMO=None):
        if nkpt is None: nkpt=self.nkpt
        if nAO is None: nAO=self.nAO
        if nMO is None: nMO=self.nMO
        ### print_00("check_matrixShape:%d %d %d    %d"%(nkpt,nAO,nMO,self.nAO))
        Ndim=np.shape(A)
        dft=self.get_DFT()
        ### print_00("#check_matrixShape:",end="");print_00([nkpt,nAO,nMO,self.nkpt,self.nAO,self.nMO])
        if(name=="MO"):
            if( self.spinrestriction == 'U'):
                assert ( (      self.pbc  and i1eqb( Ndim,[2,nkpt,nAO,nMO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[2,nAO,nMO]) ) ),dft+","+name+":"+str(Ndim)
            else:
                assert ( (      self.pbc  and i1eqb( Ndim,[nkpt,nAO,nMO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[nAO,nMO]) ) ),dft+","+name+":"+str(Ndim)
        elif(name=="eOrbs"):
            # ROKS : e(eigenvalue of ROOTHAAN fock matr) e.mo_ea, e.mo_eb
            # KROKS: e[k](...)                           e[k].mo_ea, e[k].mo_eb
            if( self.spinrestriction == 'U'):
                assert ( (      self.pbc  and i1eqb( Ndim,[2,nkpt,nMO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[2,nMO]) ) ),dft+","+name+":"+str(Ndim)
            else:
                assert ( (      self.pbc  and i1eqb( Ndim,[nkpt,nMO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[nMO]) ) ),dft+","+name+":"+str(Ndim)
        elif(name=="DM"):
            if( self.spinrestriction != 'R'):
                assert ( (      self.pbc  and i1eqb( Ndim,[2,nkpt,nAO,nAO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[2,nAO,nAO]) ) ),dft+","+name+":"+str(Ndim)+"/"+str([2,nAO,nAO])
            else:
                assert ( (      self.pbc  and i1eqb( Ndim,[nkpt,nAO,nAO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[nAO,nAO]) ) ),dft+","+name+\
                        ":"+str(Ndim)+"/"+( str([nkpt,nAO,nAO]) if(self.pbc) else str([nAO,nAO]) )
        elif(name=="Fock"):
            ## ROKS  Fock.focka, .fockb [nAO][nAO]
            ## KROKS  Fock.focka, .fockb [nkpt][nAO][nAO]
            if( self.spinrestriction == 'U'):
                assert ( (      self.pbc  and i1eqb( Ndim,[2,nkpt,nAO,nAO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[2,nAO,nAO]) ) ),dft+","+name+":"+str(Ndim)
            else:
                assert ( (      self.pbc  and i1eqb( Ndim,[nkpt,nAO,nAO]) ) or\
                         ( (not self.pbc) and i1eqb( Ndim,[nAO,nAO]) ) ),dft+","+name+":"+str(Ndim)

    def get_MOvcs(self):
        if( self._tempOrbs is None):
            if( self._tempFockMat is not None ):
                print_00("#get_MOvcs:solving MO from _tempFockMat...");
                self.solve_MO(update=True)
                MOvcs=self._tempOrbs
            else:
                print_00("#get_MOvcs:use canonicalMOs...");
                MOvcs=self._canonicalMOs
        else:
            MOvcs=self._tempOrbs
        return MOvcs

    def print_Fock(self,rttddft,title,FockMat=None,MOvcs=None,Threads=[0]):
        comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
        if( MPIrank not in Threads ):
            return
        if FockMat is None:FockMat=self.calc_fock(rttddft)
        if MOvcs is None: MOvcs=self.get_MOvcs()
        if( MOvcs is None):
            print_00("print_Fock:"+title +"!W no MOvcs..");return
        nmult_Fock=(2 if(self.spinrestriction=='U')else 1)
        nmult_MO  =(2 if(self.spinrestriction=='U')else 1)
        print_in_MOrep(self.pbc,nmult_Fock,"#print_Fock:"+title,FockMat,MOvcs,nmult_MO)
    def print_DMO(self,title,dmat=None,MOvcs=None,Threads=[0]):
        comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
        if( MPIrank not in Threads ):
            return

        if dmat is None: dmat=self.tdDM
        if MOvcs is None: MOvcs=self.get_MOvcs()
        if (MOvcs is None):
            print("print_DMO:"+title +"!W no MOvcs..");return
        nmult_DM=(2 if(self.spinrestriction!='R')else 1)
        nmult_MO=(2 if(self.spinrestriction=='U')else 1)
        print_in_MOrep(self.pbc, nmult_DM,"#print_DMO:"+title,dmat,MOvcs,nmult_MO)

    def print_AO(self,header="",Threads=[0]):
        comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
        if( MPIrank not in Threads ):
            return

        mol_or_cell = self.gen_mol_or_cell();
        print("#print_AO:"+header+":format_basis:",end="");print( format_basis(mol_or_cell.basis) )
        Nsh=len(mol_or_cell._bas)
        for ish in range(Nsh):
            print("#print_AO:"+header+":_bas_%03d:"%(ish),end="");print(mol_or_cell._bas[ish])
            
    def solve_MO(self,rttddft=None,FockMat=None,update=None,comment=None):
        if update is None: update=(FockMat is None); ## default: diagonalize _tempFockMat and put it to .. 
        if FockMat is None: FockMat=self._tempFockMat
        info=0
        if( FockMat is None):
            print_00("#solve_MO:missing FockMat %r"%(self._tempFockMat is not None));
            return None
        Ndim=np.shape(FockMat);
        nAO=( self.nAO if(self.nAO is not None) else rttddft.nAO )
        nMO=( self.nMO if(self.nMO is not None) else rttddft.nMO )
        nkpt=( self.nkpt if(self.nkpt is not None) else rttddft.nkpt )
        self.check_matrixShape("Fock",FockMat,nkpt,nAO,nMO)

        nmult_MO=(2 if(self.spinrestriction=='U') else 1)        
        refMO=(self._tempOrbs if(self._tempOrbs is not None) else self._canonicalMOs)
        EA=None;UA=None
        for sp in range(nmult_MO):
            FM=( FockMat if(nmult_MO==1) else FockMat[sp])
            if(self.pbc):
                E=[];U=[]
                for kp in range(Ndim[0]):
                    E1,U1,info = hdiag(FM[kp])
                    E.append(E1); U.append(U1);
                if( nmult_MO == 1 ):
                    EA=E;UA=U
                else:
                    if(EA is None):
                        EA=[];UA=[]
                    EA.append(E);UA.append(U)
            else:
                E1,U1,info= hdiag(FM)
                if( nmult_MO == 1 ):
                    EA=E1;UA=U1
                else:
                    if(EA is None):
                        EA=[];UA=[]
                    EA.append(E1);UA.append(U1)
        if( update ):
            if( (comment is not None) and (rttddft is not None)):
                if( not pbc ):
                    S1e=rttddft.get_ovlp()
                    for sp in range(nmult_MO):
                        if( nmult_MO==1 ):
                            compare_MOs(UA,refMO,S1e,title=comment)
                        else:
                            compare_MOs(UA[sp],refMO[sp],S1e,title=comment)
                else:
                    for sp in range(nmult_MO):
                        refMO_sp=( refMO if(nmult_MO==1) else refMO[sp] )
                        UA_sp=( UA if(nmult_MO==1) else UA[sp] )
                        kvectors=np.reshape( rttddft.kpts, (-1,3) )
                        for kp in range(Ndim[0]):
                            S1e=rttddft.get_ovlp( rttddft.cell, kvectors[kp])
                            compare_MOs(UA_sp[kp],refMO_sp[kp],S1e,title=comment)
            self._tempOrbs=np.array(UA);self._tempEngs=np.array(EA)
        self.check_matrixShape("MO",UA)
        self.check_matrixShape("eOrbs",EA)
        return EA,UA,info

    def set_Rnuc(self,Rnew,unit):
        fac=1.0
        if( unit=='ANGS' or unit=='A' ):
            fac=1.0/physicalconstants.BOHRinANGS
        for I in range(self.Nat):
            for J in range(3):
                self.Rnuc[I][J] = Rnew[I][J] * fac

    # update_Rnuc : updates (generates) rttddft object with specified Rnuc
    #               (i) partial update (IandJ,coordinate) from md.Rnuc
    #               (ii) full update to Rnew
    #       @param  update_moldyn: whether or not update md.Rnuc
    #               if you only temporarily need rttddft(R+dR), leave it False 
    def update_Rnuc(self, Rnew=None, IandJ=None, coordinate=None, update_moldyn=None, fix_occ=True, 
                    update_canonicalMOs=0,sync=False, R_dbg=None):
        assert (update_moldyn is not None),"update_Rnuc:set update_moldyn explicitly"
        
        if( Rnew is None ):
            assert ( (IandJ is not None) and (coordinate is not None) ),"update_Rnuc.000a" 
            Rnew = np.zeros( [self.Nat,3] )
            for I in range(self.Nat):
                for J in range(3):
                    Rnew[I][J]=self.Rnuc[I][J]
            Rnew[ IandJ[0] ][ IandJ[1] ] = coordinate

            if( R_dbg is not None ):
                diff= max( abs( np.ravel(Rnew) - np.ravel(R_dbg)) )
                print("#diff:%e"%(diff),Rnew,R_dbg);
                assert diff<1.0e-7,""

            if( update_moldyn ):
                self.Rnuc[ IandJ[0] ][ IandJ[1] ] = coordinate
        else:
            assert ( (IandJ is None) and (coordinate is None) ),"update_Rnuc.000b"
            if( update_moldyn ):
                for I in range(self.Nat):
                    for J in range(3):
                        self.Rnuc[I][J]=Rnew[I][J]

            
        logger=self.get_logger(set=True)

        ## here, None means "default" so it perfectly makes sense to pass None to gen_rttddft .. 2021.12.07
        ## ## assert (not self.pbc),"2021.12.06 --- it seems that you are passing nKpoints=None to gen_rttddft.. PLS check the results"

        if( update_canonicalMOs!=0 ):
            temp=self.gen_rttddft(mol_or_cell=None,Rnuc=Rnew,unit='B',nKpoints=None,logger=logger)
            temp._fix_occ=False
            E_gs=temp.calc_gs()

            Fock=None                
            if( update_canonicalMOs & 4 ):
                self._GSfockMat = arrayclone( temp.get_fock() )
                Fock=self._GSfockMat
            if( update_canonicalMOs & 2):
                self._canonicalMOs=arrayclone( temp.mo_coeff )
                self._cMOs_Rnuc = arrayclone( Rnew )
            if( update_canonicalMOs & 1):
                if( Fock is None ):
                    Fock=temp.get_fock()
                if( self.spinrestriction=='U'):
                    self._canonicalEorbs=\
                        np.array( [ calc_eorbs(temp.mo_coeff[0], Fock[0]),calc_eorbs(temp.mo_coeff[1], Fock[1]) ] )
                elif( self.spinrestriction=='R'):
                    self._canonicalEorbs=calc_eorbs(temp.mo_coeff, Fock)
                else:
                    assert False,"ROKS?"

        ret=self.gen_rttddft(mol_or_cell=None,Rnuc=Rnew,unit='B',nKpoints=None,logger=logger) # None in order to apply default
        print("#gen_rttddft:",np.ravel(Rnew))
        if( fix_occ ):
            ret.set_mo_occ(self.mo_occ,fix_occ,clone=True)

        if( sync ):
            comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
            if(MPIsize > 1 ):
                R_loc=arrayclone( self.Rnuc )
                comm.Bcast(self.Rnuc)
                diff=max( np.ravel(R_loc) - np.ravel(self.Rnuc) )
                if( diff > 1e-7 ):
                    print("#Sync:$%02d:non-trivial sync Rnuc:%e"%(diff)+str(R_loc)+">"+str(self.Rnuc))

                rttddft_common.Set_Rnuc( self.Rnuc, step=self.step, time_AU=self.time_AU, Logging=range(0,MPIsize) )
                rttddft_common.Check_cellRnuc( ret.mol, "Moldyn.update_Rnuc")
                if( self.pbc ):
                    if( ret.with_df is not None):
                        if( hasattr( ret.with_df, "cell" ) ):
                            rttddft_common.Check_cellRnuc( ret.with_df.cell, "Moldyn.update_Rnuc:df.cell")

        return ret

    def set_logger(self,logger=None,filename=None,replace=True,rttddft=None):
        if( not replace ):
            if( self.logger is not None):
                print_00("#moldyn already has a logger");
                if( rttddft is not None):
                    return rttddft.set_logger(logger=logger, filename=filename, replace=replace);
                return False
        if( logger is None ):
            assert (filename is not None),""
            self.logger = Logger(filename)
            print_00("#Moldyn.set_logger:creates:"+str(self.logger))
        else:
            self.logger = logger
            print_00("#Moldyn.set_logger:sets:"+str(self.logger))
        if( rttddft is not None):
            retv = rttddft.set_logger(logger=logger, filename=filename, replace=replace);
#        os.system("fopen " + self.logger.filepath);
        os.system("ls -ltrh "+self.logger.filepath);
        return True
    
    def get_logger(self,set=False):
        if( (self.logger is None) and set ):
            self.set_logger(filename="moldyn_dbg.log");
        return self.logger;

    def set_time_AU(self,time,step=None,rttddft=None):
        self.time_AU=time
        if( step is not None ):
            self.step = step
        if( rttddft is not None):
            rttddft._time_AU = time
            rttddft._step = step
    def set_step(self,step,rttddft=None):
        self.step = step
        if( rttddft is not None):
            rttddft._step = step

    def export_DM(self,fpath,dm=None,comment=None):
        if(dm is None):
            dm=self.tdDM
        svld_aNbuf('S',fpath,buf=dm,comment=comment)

    def read_DM(self,fpath,update_self=False,get_comments=None):
        dm=svld_aNbuf('L',fpath,get_comments=get_comments)
        return dm

    # Projects current tdMO onto self._canonicalMOs to get  \langle  canonical MO_{\ell} | tdMO_j \rangle
    # results are (i) printed out on fpath
    #             (ii) partly exported into dict
    def calc_MOprojection( self, rttddft, header,fpath=None,mo_coeff=None,kpts=None,append=True,dict=None):
        fd=sys.stdout
        # assert False,"MOprojection"
        if( self._canonicalMOs is None ):
            assert False,"please set _canonicalMOs .."
        if mo_coeff is None: mo_coeff = self.tdMO
        pbc=self.pbc
        kvectors = (None if (not pbc) else np.reshape( rttddft.kpts, (-1,3)))
        nkpt=(1 if(not pbc) else self.nkpt )
        nAO =self.nAO
        psit_jmoRH=np.zeros( [nAO],dtype=np.complex128)
        excitedstate_populations=None;
        nmult_MO=(2 if(self.spinrestriction=='U') else 1)
        for sp in range(nmult_MO):
            header_spl = "" if(nmult_MO==1) else ("_alpha" if(sp==0) else "_beta")
            ### U : FockMat is two-fold Fa,Fb
            ### O : 
            Fock_sp1=None; FM1=None
            if( self._GSfockMat is not None ):
                Fock_sp1= ( self._GSfockMat if( self.spinrestriction != 'U' ) else self._GSfockMat[sp] )

            for kp in range(nkpt):
                if( not pbc ):
                    S1e = rttddft.get_ovlp()
                else:
                    S1e = rttddft.get_ovlp(rttddft.cell,kvectors[kp])

                if( kpts is not None ):
                    if( not ( kp in kpts ) ):
                        continue
                if( Fock_sp1 is not None ):
                    FM1=( Fock_sp1 if(not pbc) else Fock_sp1[kp])

                if( self.spinrestriction == 'O'):
                    eorbREF =(self._canonicalEorbs if(not pbc) else self._canonicalEorbs[kp])
                    ### if( hasattr( eorbREF, ('mo_ea' if(sp==0) else 'mo_eb'),None) is not None ):
                    ###    eorbREF = ( eorbREF.mo_ea if(sp==0) else eorbREF.mo_eb )
                elif( self.spinrestriction == 'U'):
                    eorbREF =(self._canonicalEorbs[sp] if(not pbc) else self._canonicalEorbs[sp][kp])
                else:
                    eorbREF =(self._canonicalEorbs if(not pbc) else self._canonicalEorbs[kp])

                if( self.spinrestriction == 'U'):
                    cofsRH  = (mo_coeff[sp] if(not pbc) else mo_coeff[sp][kp])
                    cofsLH  = (self._canonicalMOs[sp] if(not pbc) else self._canonicalMOs[sp][kp])
                else:
                    cofsRH  = (mo_coeff if(not pbc) else mo_coeff[kp])
                    cofsLH  = (self._canonicalMOs if(not pbc) else self._canonicalMOs[kp])
                
                assert ( i1eqb(np.shape(cofsRH),[self.nAO, self.nMO]) ),str(np.shape(cofsRH))+"/"+str([self.nAO, self.nMO])
                assert ( i1eqb(np.shape(cofsLH),[self.nAO, self.nMO]) ),str(np.shape(cofsLH))+"/"+str([self.nAO, self.nMO])

                nMOlh=len(cofsLH[0]); nMOrh=len(cofsRH)

                if( excitedstate_populations is None ):
                    excitedstate_populations =( [] if(not pbc) else [ [] for i in range(self.nkpt) ]) if(self.spinrestriction == 'R') else \
                                              ( [ [], [] ] if(not pbc) else [ [ [] for i in range(self.nkpt) ],\
                                                                             [ [] for i in range(self.nkpt) ] ] )
                    eorbs = ( [] if(not pbc) else [ [] for i in range(self.nkpt) ]) if( self.spinrestriction == 'R') else \
                                              ( [ [], [] ] if(not pbc) else [ [ [] for i in range(self.nkpt) ],\
                                                                             [ [] for i in range(self.nkpt) ] ] )
                Xpop1 =( excitedstate_populations if( not pbc) else excitedstate_populations[kp]) if(self.spinrestriction == 'R') else \
                       ( excitedstate_populations[sp] if( not pbc) else excitedstate_populations[sp][kp])
                eOrbs1 = ( eorbs if( not pbc ) else eorbs[kp] ) if(self.spinrestriction == 'R') else \
                        ( eorbs[sp] if( not pbc ) else eorbs[sp][kp] )

                pop=np.zeros([nMOrh,nMOlh])
                
                for jmoRH in range(nMOrh):
                    if( fpath is not None ):
                        if( nkpt>1 ):
                            fd=futils.fopen_00(fpath.replace(".dat","")+"_%d.%d.dat"%(kp,jmoRH),("a" if append else "w") )#OK
                        else:
                            fd=futils.fopen_00(fpath.replace(".dat","")+"_%d.dat"%(jmoRH),("a" if append else "w"));#OK
                    else:
                        fd=sys.stdout
                    ## get single tdMO -----------
                    for k in range(nAO):
                        psit_jmoRH[k]=cofsRH[k][jmoRH]

                    for kLH in range(nMOlh):
                        cdum=np.vdot( cofsLH[:,kLH], np.matmul(S1e,psit_jmoRH) )
                        pop[jmoRH][kLH]=cdum.real**2 + cdum.imag**2

                    eorb=0
                    if( FM1 is not None ):
                        eorb=np.vdot( psit_jmoRH, np.matmul( FM1, psit_jmoRH)).real
                        if( self.spinrestriction == 'O' and hasattr(FM1,'focka')):
                            ea=np.vdot( psit_jmoRH, np.matmul( FM1.focka, psit_jmoRH)).real
                            eb=np.vdot( psit_jmoRH, np.matmul( FM1.fockb, psit_jmoRH)).real
                            eorb = lib.tag_array(eorb, mo_ea=ea, mo_eb=eb)
                    Xpop1.append( 1.0-pop[jmoRH][jmoRH] )
                    eOrbs1.append( eorb )
                    if( self.spinrestriction == 'O' and hasattr(FM1,'focka') ):
                        print_00("%s %f %f %f %f   %f  "%(header+header_spl,eorbREF[jmoRH],eorb,ea,eb,pop[jmoRH][jmoRH])
                                +d1toa(pop[jmoRH]),file=fd)
                    else:
                        print_00("%s %f %f  %f  "%(header+header_spl,eorbREF[jmoRH],eorb,pop[jmoRH][jmoRH])+d1toa(pop[jmoRH]),file=fd)
                    if( fpath is not None ):
                        futils.fclose_00(fd)
        if( dict is not None ):
            if( "excitedstate_populations" in dict ):
                dict[ "excitedstate_populations" ] = excitedstate_populations
            if( "eorbs" in dict ):
                dict[ "eorbs" ] = eorbs
        print_00("calc_MOprojection done");## assert False,"projection"+fpath

    def calc_MOprojection01( self, rttddft, header,FileName,MO_Coeff=None,kpts=None,append=True):
        fd=sys.stdout
        if( self._canonicalMOs is None ):
            assert False,"please set _canonicalMOs .."
        if MO_Coeff is None: MO_Coeff = self.tdMO
        pbc=self.pbc
        kvectors = (None if (not pbc) else np.reshape( rttddft.kpts, (-1,3)))
        nkpt=(1 if(not pbc) else self.nkpt )
        nAO =self.nAO; NcanMO=None; NtdMO=None
        nmult_MO=(2 if(self.spinrestriction=='U') else 1)
        for sp in range(nmult_MO):
            mo_coef1=( MO_Coeff if(nmult_MO==1) else MO_Coeff[sp] )
            canonicalMO1=( self._canonicalMOs if(nmult_MO==1) else self._canonicalMOs[sp])
            S1e=(None if(pbc) else rttddft.get_ovlp())
            filename1 = FileName + ("" if(nmult_MO==1) else "_s%d"%(sp))
            for kp in range(nkpt):
                fnme= filename1 + ("" if(not pbc) else "_k%d"%(kp))
                fdP= open_00(fnme+"_MOphases.dat",("a" if(append) else "w"))
                tdMO =(mo_coef1 if(not pbc) else mo_coef1[kp]);
                canMO=(canonicalMO1 if(not pbc) else canonicalMO1[kp]); #[nAO][nMO]
                
                Ndim_tdMO=np.shape(tdMO); NtdMO=Ndim_tdMO[1]; LdAO=Ndim_tdMO[0];
                Ndim_canMO=np.shape(canMO); NcanMO=Ndim_canMO[1]; assert LdAO==Ndim_canMO[0],""
                Pstring=""
                if(pbc):
                    S1e = rttddft.get_ovlp(rttddft.cell,kvectors[kp])
                PI=3.1415926535897932384626433832795
                for itdMO in range(NtdMO):
                    fdA= open_00(fnme+"_MOampd%03d.dat"%(itdMO),("a" if(append) else "w"))
                    arr =[ np.vdot( canMO[:,jcMO],np.matmul( S1e, tdMO[:,itdMO])) for jcMO in range(NcanMO) ]
                    
                    string=""
                    for jcMO in range(NcanMO):
                        string+=(' '*3) + "%14.6e %12.6f"%( abs(arr[jcMO]), math.atan2( arr[jcMO].imag, arr[jcMO].real )/PI)
                    print_00(header+string,file=fdA);close_00(fdA);### fdA.close()
                    jcMaxLoc =z1maxloc(arr)
                    Pstring+= "%14.6e %12.6f"%( abs(arr[jcMaxLoc]), math.atan2( arr[jcMaxLoc].imag, arr[jcMaxLoc].real )/PI)
                print_00(header+Pstring,file=fdP)
                close_00(fdP);### fdP.close()
        if( not append ):
            gnu="calc_MOprojection01";os.system("gnuf.sh "+gnu);gnuf=gnu+".plt"
            fdg=open_00(gnuf,"a")
            for sp in range(nmult_MO):
                filename1 = FileName + ("" if(nmult_MO==1) else "_s%d"%(sp))
                for kp in range(nkpt):
                    print_00('set title "sp=%d kp=%d'%(sp,kp)+'"',file=fdg) 
                    fnme= filename1 + ("" if(not pbc) else "_k%d"%(kp))
                    fnmP= fnme+"_MOphases.dat";PLOT="plot ";sty=1;
                    for itdMO in range(NtdMO):
                        ENDL=("" if(itdMO+1==NtdMO) else ",\\" )
                        print_00(PLOT+'"'+fnmP+'" using 3:%d with lines ls %d'%(5+2*itdMO,sty)+ENDL,file=fdg);sty+=1;PLOT=""
                for itdMO in range(NtdMO):
                    fnmA=fnme+"_MOampd%03d.dat"%(itdMO)
                    print_00('set title "sp=%d kp=%d MO%03d'%(sp,kp,itdMO)+'"',file=fdg) 
                    for jcMO in range(NcanMO):
                        ENDL=("" if(itdMO+1==NtdMO) else ",\\" )
                        print_00(PLOT+'"'+fnmA+'" using 3:%d with lines ls %d'%(4+2*jcMO,sty)+ENDL,file=fdg);sty+=1;PLOT=""
            close_00(fdg);### fdg.close()

    def transform_tdMO(self,rttddft, ForB, check_norm=True,sqrdevtol=1.0e-6,title="",logger=None):
        # Ssqrt * tdMO 
        if logger is None: logger=self.get_logger(True)
        FWD = (ForB=='F')
        Src= ( self.tdMO if(FWD) else self._orth_MO )
        rttddft.update_Sinv()
        
        nkpt=self.nkpt;nAO=self.nAO;nMO=self.nMO; assert (nAO is not None),""
        nmult_WF=(2 if(self.spinrestriction=='U') else 1)
        for sp in range(nmult_WF):
            srcWF=( Src if(nmult_WF==1) else Src[sp])
            for kp in range(nkpt):
                tgt=srcWF;        Matr=(rttddft._Ssqrt if(FWD) else rttddft._Sinvrt )
                if( self.pbc ):
                    tgt=srcWF[kp];Matr=(rttddft._Ssqrt[kp] if(FWD) else rttddft._Sinvrt[kp] )
                
                tgt = np.matmul( Matr, tgt)  ## Ssqrt[nAO][nAO] * MO_k[nAO][nMO]
        if( check_norm ):
            check_wfNorms( self.nkpt, self.nAO, self.nMO, rttddft,
                           self.pbc, self.spinrestriction, Src, AOrep=(not FWD), sqrdevtol_FIX=1.0e-6, title=title, logger=logger)
        if( FWD ):
            self._orth_MO = Src
            self.tdMO = None
        else:
            self.tdMO = Src
            self._orth_MO = None

    def calc_MOpopul(self,Rnuc,target_MOs=None,deviations=None):
        if( deviations is not None ):
            deviations["N_overpop"]=0
            deviations["Ne"]=0
            deviations["max_overpop"]=0
        if target_MOs is None: target_MOs=self.tdMO
        dev=aNmaxdiff(Rnuc,self.Rnuc);TINY=1.0e-7

        rttddft=self.gen_rttddft( Rnuc=Rnuc, unit='B')
        CanonicalMOsAL=None

        nMO=self.nMO;
        if( (self._canonicalMOs is not None) and (self._cMOs_Rnuc is not None) and nMO>0 ):
            Rdev=max( np.ravel(Rnuc)- np.ravel(self._cMOs_Rnuc) )
            if( Rdev < 1e-7 ):
                CanonicalMOsAL=self._canonicalMOs
        else:
            Egnd=rttddft.calc_gs()
            CanonicalMOsAL=rttddft.mo_coeff
            rttddft.update_nOrbs(); nMO=rttddft.nMO
            if( self.nMO != nMO ):
                assert self.nMO<=0,"%d / %d"%(nMO,self.nMO)
                print("#Moldyn.nMO set:%d"%(nMO));self.nMO=nMO
        assert nMO>0,""

        kvectors=( None if(not self.pbc) else np.reshape(rttddft.kpts, (-1,3)))
        MO_Occ=self.mo_occ
        Ne=sum( np.ravel(MO_Occ) )
        ### rttddft.update_nOrbs()
        
        nkpt=(1 if(not self.pbc) else rttddft.nkpt)
        nmult_MO=(2 if(self.spinrestriction=='U') else 1)
        Ret=[]
        S1e=( rttddft.get_ovlp() if(not self.pbc) else None)
        N_overpop=0;max_overpop=-1
        for sp in range(nmult_MO):
            canonicalMOs = (CanonicalMOsAL if(nmult_MO==1) else CanonicalMOsAL[sp])
            tgtMOs       = (target_MOs if(nmult_MO==1) else target_MOs[sp])
            OCC          = (MO_Occ[sp] if(self.spinrestriction=='U') else MO_Occ)
            ret_s=[]
            for kp in range(nkpt):
                if( self.pbc ):
                    print(nkpt,len(kvectors),kp)
                    S1e=rttddft.get_ovlp(rttddft.cell,kvectors[kp])
                cMOs= (canonicalMOs if(not self.pbc) else canonicalMOs[kp])
                tMOs= (tgtMOs if(not self.pbc) else tgtMOs[kp])
                Occ = (OCC if(not self.pbc) else OCC[kp])
                ## both are nAO * nMO matrices
                pop=np.zeros([nMO,nMO])
                for Iorb in range(nMO):
                    tmo=tMOs[:,Iorb]
                    wgsum=0.0
                    Sxt=np.matmul(S1e,tmo)
                    for jorb in range(nMO):
                        cof=np.vdot(cMOs[:,jorb],Sxt)
                        wg =cof.real**2 + cof.imag**2
                        pop[Iorb][jorb]=wg*Occ[Iorb]; wgsum+=wg
                    if(abs(wgsum-1)>1.0e-6):
                        print_00("Wgsum %f deviates from unity"%(wgsum));
                        assert abs(wgsum-1)<0.1,""
                
                popsum=np.zeros([nMO])
                for jorb in range(nMO):
                    popsum[jorb]=0
                    for Iorb in range(nMO):
                        popsum[jorb]+=pop[Iorb][jorb]
                    if(popsum[jorb]>1.0):
                        N_overpop=N_overpop+1
                        max_overpop=max(max_overpop,(popsum[jorb]-1.0))

                if(not self.pbc and nmult_MO==1 ):
                    if( deviations is not None ):
                        deviations["Ne"]=abs(popsum-Ne); deviations["N_overpop"]=N_overpop
                        deviations["max_overpop"]=max_overpop
                    return pop
                else:
                    if( not self.pbc ):
                        Ret.append(pop)   ### non-pbc && nmult==2
                    else:
                        ret_s.append(pop) ## pbc
            if(self.pbc and nmult_MO==1 ):
                if( deviations is not None ):
                    deviations["Ne"]=abs(popsum-Ne); deviations["N_overpop"]=N_overpop
                    deviations["max_overpop"]=max_overpop
                return ret_s              ### pbc && nmult==1
            else:
                if(self.pbc):
                    Ret.append(ret_s)    ### pbc && nmult==2
        popsum=sum(np.ravel(Ret))
        if( deviations is not None ):
            deviations["Ne"]=abs(popsum-Ne)
            deviations["N_overpop"]=N_overpop
            deviations["max_overpop"]=max_overpop
        print_00("#calc_MOpopul:%f/%f"%(popsum,Ne));assert abs(popsum-Ne)<0.1,""
        return Ret

    def print_MOpopul(self,filename,header="",append=False,Rnuc=None,target_MOs=None,deviations=None):
        if Rnuc is None:Rnuc=self.Rnuc
        if target_MOs is None:target_MOs=self.tdMO
        POP=self.calc_MOpopul(Rnuc=Rnuc,target_MOs=target_MOs,deviations=deviations)
        nkpt=(1 if(not self.pbc) else self.nkpt)
        nMO=self.nMO
        nmult_MO=(2 if(self.spinrestriction=='U') else 1)
        for sp in range(nmult_MO):
            
            Pop=POP if(nmult_MO==1) else POP[sp]
            for kp in range(nkpt):
                fpath=modify_filename( filename,self.pbc, nmult_MO,sp,kp)
                     #modify_filename(      org,pbc,nmult,spin,kpt)
                pop=Pop if (not self.pbc) else Pop[kp]
                arr=[]
                for Iorb in range(nMO):
                    pop_I=0.0
                    for tdo in range(nMO):
                        ### print_00("%d %d /"%(tdo,Iorb)+str(np.shape(pop)))
                        pop_I+=pop[tdo][Iorb]
                    arr.append(pop_I)
                fd=futils.fopen_00(fpath,("a" if(append) else "w"))  ## OK
                if(not append):
                    hlen=len(header.split())
                    print_00("#--header:%d--  %d:MO_pop ..."%(hlen,hlen+1),file=fd)
                print_00(header + d1toa(arr),file=fd)
                futils.fclose_00(fd)

    def normalize_MOcofs(self,rttddft,MO_Coeffs=None,update_self=True,devsum_TOL=1.0e-7,
                         ofddev_TOL=1.0e-7,dagdev_TOL=1.0e-7,verbose=0,dict=None):
        if MO_Coeffs is None: MO_Coeffs=self.tdMO
        nmult_MO=(2 if(self.spinrestriction=='U') else 1)
        pbc = self.pbc
        kvectors = (None if (not pbc) else np.reshape( rttddft.kpts, (-1,3)))
        nkpt =(1 if(not pbc) else self.nkpt)
        S1e = ( rttddft.get_ovlp() if(not pbc) else None)
        Nfix=0
        fixed_MO_Coeffs=(None if(nmult_MO==1 and (not pbc)) else (\
                        [] if(nmult_MO==1 and pbc) else (\
                        [] if(nmult_MO==2 and (not pbc)) else (\
                        [ [], [] ] ))))
        ## 20230925: normalize_MOcofs_log=BufferedWriter01("normalize_MOcofs.log",Append=True)
        devsum_max=-1;dagdev_max=-1;ofddev_max=-1

        for sp in range(nmult_MO):
            MOCOEFS=( MO_Coeffs if(nmult_MO==1) else MO_Coeffs[sp])
            for kp in range(nkpt):
                if( pbc ):
                    S1e=rttddft.get_ovlp( rttddft.cell, kvectors[kp])
                mocoefs=( MOCOEFS if(not pbc) else MOCOEFS[kp])
                CxSxC=np.matmul( np.matrix.getH( mocoefs ), np.matmul(S1e, mocoefs) )
                devsum,dagdev,ofddev=deviation_from_unitmatrix( CxSxC )
                devsum_max=max(devsum_max,devsum)
                ofddev_max=max(ofddev_max,ofddev)
                dagdev_max=max(dagdev_max,dagdev)

                dbgng_MOcofs=True
                
                fixed = mocoefs; isMOfixedHere=False
                if( devsum >= devsum_TOL or ofddev >= ofddev_TOL or dagdev>=dagdev_TOL ):
                    # 20230925: normalize_MOcofs_log.write("#Overlap matrix deviates unity:%e %e %e at R_nuc:"%(\
                    # 20230925:     devsum,dagdev,ofddev)+str(self.Rnuc) )
                    print_00("#Overlap matrix deviates unity:%e %e %e"%(devsum,dagdev,ofddev))
                    invSSqrt= get_Sinvsqrt(CxSxC)
                    fixed = np.matmul( mocoefs,invSSqrt )
                    Nfix+=1; isMOfixedHere=True
                if( not pbc ):
                    if( nmult_MO==1 ):
                        fixed_MO_Coeffs=fixed
                    else:
                        fixed_MO_Coeffs.append( fixed )
                else:
                    if( nmult_MO==1 ):
                        fixed_MO_Coeffs.append( fixed )
                    else:
                        fixed_MO_Coeffs[sp].append(  fixed )

                if( isMOfixedHere and dbgng_MOcofs ):
                    CxSxC_1=np.matmul( np.matrix.getH( fixed ), np.matmul(S1e, fixed) )
                    devsum_1,dagdev_1,ofddev_1=deviation_from_unitmatrix( CxSxC_1 )
                    ## 20230925: normalize_MOcofs_log.write("#Overlap matrix fixed:%e %e %e at R_nuc:"%(\
                    ## 20230925:     devsum_1,dagdev_1,ofddev_1)+str(self.Rnuc) )
                    print_00("#Overlap matrix fixed:%e %e %e at R_nuc:"%(\
                              devsum_1,dagdev_1,ofddev_1)+str(self.Rnuc) )
                    assert devsum_1<devsum_TOL and dagdev_1<dagdev_TOL and ofddev_1<ofddev_TOL,""


        if(dict is not None):
            dict.update({"devsum_max":devsum_max})
            dict.update({"dagdev_max":dagdev_max})
            dict.update({"ofddev_max":ofddev_max})
            dict.update({"Nfix":Nfix})
        if( verbose>1 or (verbose>0 and Nfix>0) ):
            fd=futils.fopen_00("normalize_MOcofs.log","a") ## OK
            print_00("%f  %d: %e %e %e"%( self.time_AU*physicalconstants.aut_in_femtosec,Nfix,\
                devsum_max,dagdev_max,ofddev_max),file=fd)
            futils.fclose_00(fd)
            if( Nfix == 0 ):
                print_00("#normalize_MOcofs:%f  nofix: %e %e %e"%( self.time_AU*physicalconstants.aut_in_femtosec,\
                    devsum_max,dagdev_max,ofddev_max))
        if(update_self):
            self.check_matrixShape( "MO",fixed_MO_Coeffs )
            self.tdMO = np.array( fixed_MO_Coeffs )
        ## normalize_MOcofs_log.flush_ifnecs()
        return np.array( fixed_MO_Coeffs )

    def get_fock(self,rttddft,h1e,dm,time_AU,update_self=True):
        vhf= rttddft.get_veff(rttddft.cell)
        fock=rttddft.get_fock(h1e=h1e,dm=dm,vhf=vhf)
        if(update_self):
            self.vhf_last=vhf
            self.Fock_last=fock
            self.tmAU_Fock_last=time_AU
        return fock

    ## Calculates MOpopulations          PBC                  not
    ##                             UHF : [sp][nkpt][nMO]      [sp][nMO]
    ##                             else:     [nkpt][nMO]          [nMO]
    def calc_MOpopulations(self,rttddft, target_MOs=None, reference_MOs=None, MO_occ=None):

        if(target_MOs is None): target_MOs=self.tdMO;
        if(reference_MOs is None): reference_MOs = self._canonicalMOs
        if(MO_occ is None): MO_occ=self.mo_occ
        nmult_MO=(2 if(self.spinrestriction=='U') else 1)
        nMO=( self.nMO if(self.nMO is not None) else rttddft.nMO )
        nkpt=(1 if(not self.pbc) else self.nkpt)
        kvectors=(None if(not self.pbc) else np.reshape( rttddft.kpts, (-1,3) ))

        dbgng=False;check_MOovlps=True

        N_elec=get_Nele(rttddft)
        eps=1.0e-20
        S1e=(None if(self.pbc) else rttddft.get_ovlp())
        Ret=[]
        for sp in range(nmult_MO):
            targetMOs =    (    target_MOs if(nmult_MO==1) else target_MOs[sp] )
            referenceMOs = ( reference_MOs if(nmult_MO==1) else reference_MOs[sp] )
            MOocc =        (        MO_occ if(nmult_MO==1) else MO_occ[sp] )
            Nele =         (        N_elec if(nmult_MO==1) else N_elec[sp] )
            Ret_sp=[]
            popsum=0.0

            for kp in range(nkpt):
                pop=np.zeros([nMO])
                tgtMOs = ( targetMOs if(not self.pbc) else targetMOs[kp])
                refMOs = ( referenceMOs if(not self.pbc) else referenceMOs[kp])
                Occ    = ( MOocc if(not self.pbc) else MOocc[kp] )
                if( self.pbc ):
                    S1e=rttddft.get_ovlp( rttddft.cell, kvectors[kp])

                if(dbgng):
                    fd1=open("refMO_%02d.dat"%(kp),"a")
                    print("#%f"%(self.time_AU),file=fd1)
                    for nu in range(nMO):
                        string=""
                        for el in range(nMO):
                            string+="%12.6f %12.6f       "%( refMOs[el][nu].real,refMOs[el][nu].imag)
                        print(string,file=fd1);
                    print("\n\n\n",file=fd1)
                    fd1.close()
                        
                ovlp=np.zeros([nMO,nMO],dtype=np.complex128)
                for el in range(nMO):
                    for nu in range(nMO):
                        ovlp[el][nu]=np.vdot( refMOs[:,el], np.matmul( S1e, tgtMOs[:,nu] ) )

                if(check_MOovlps):
                    AUinFS=0.02418884326198665673981200055933
                    fd1=open( rttddft_common.getjob(True)+"_MOovlps.log","a")
                    for nu in range(nMO):
                        string="";
                        for el in range(nMO):
                            cdum=ovlp[el][nu];ampd=abs(cdum);arg=np.arctan2(cdum.imag,cdum.real)
                            string+=" %12.6f %10.4f    "%(ampd,arg)
###                            string+="%12.6f "%(ovlp[el][nu].real)
                        print("#tdMO_%04d: %14.4f "%(nu,self.time_AU*AUinFS)+string,file=fd1)
                    fd1.close()
                
                for el in range(nMO):
                    cum=0.0
                    for nu in range(nMO):
                        if( Occ[nu]<eps ):
                            continue
                        cum+= ovlp[el][nu]*Occ[nu]*np.conj( ovlp[el][nu] )
                    pop[el]=cum.real
                
                popsum+=sum(pop)
                if( not self.pbc ):
                    Ret_sp=pop            ## Ret_sp:[nMO]
                else:
                    Ret_sp.append( pop )  ## Ret_sp:[kp][nMO]
            if( self.pbc ):
                popsum=popsum/float(nkpt)
            
            dev=abs( popsum - Nele)
            if( dev > 1.0e-7 ):
                strwarn="popsum:%16.8f / %16.8f:"%(popsum,Nele)+str(pop)
                rttddft_common.Printout_warning("popsum",strwarn)
                assert dev<1.0e-5,""+strwarn

            if( nmult_MO == 1 ):
                Ret=Ret_sp
            else:
                Ret.append( Ret_sp )
        Avg=None
        if( (nmult_MO == 1) and (not self.pbc) ):
            Avg=Ret
        else: 
            Avg=np.zeros([nMO])
            for sp in range(nmult_MO):
                Pop=(Ret if(nmult_MO==1) else Ret[sp])
                for kp in range(nkpt):
                    pop=(Pop if(not self.pbc) else Pop[kp])
                    for el in range(nMO):
                        Avg[el]+=pop[el]
            if( self.pbc ):
                for el in range(nMO):
                    Avg[el]/=float(nkpt)

        return Ret,Avg

    def save_tempFockMat(self,src,t2,calc_eigvcs=False):
        self.check_matrixShape("Fock",src)
        Ndim=np.shape(src)
        if( self._tempFockMat is None ):
            self._tempFockMat = arrayclone( src )
            self._t2=t2;
        else:
            if( len(Ndim)==2 ):
                for I in range(Ndim[0]):
                    for J in range(Ndim[1]):
                        self._tempFockMat[I][J] = src[I][J]
            elif( len(Ndim)==3 ):
                for I in range(Ndim[0]):
                    for J in range(Ndim[1]):
                        for K in range(Ndim[2]):
                            self._tempFockMat[I][J][K] = src[I][J][K]
            elif( len(Ndim)==4 ):
                for I in range(Ndim[0]):
                    for J in range(Ndim[1]):
                        for K in range(Ndim[2]):
                            for L in range(Ndim[3]):
                                self._tempFockMat[I][J][K][L] = src[I][J][K][L]
            else:
                assert False,""
        if( calc_eigvcs ):
            self.solve_MO(update=True)
        else:
            self._tempOrbs=None;self._tempEorbs=None

    def delete_Fock_last(self,description="",flag=7):
        ### self._dbg_update_FockLast("delete_Fock_last:"+str(description))
        if( flag & 1 !=0 ):
            self.Fock_last=None; self.tmAU_Fock_last=None
        if( flag & 2 !=0 ):
            self.vhf_last=None; self.tmAU_vhf_last=None
        if( flag & 4 !=0 ):
            self.hcore_last=None; self.tmAU_hcore_last=None
    def get_DFT(self):
        ret=("K" if(self.pbc) else "")
        if( self.spinrestriction == 'U'):
            ret+="UKS";  return ret
        elif( self.spinrestriction == 'O'):
            ret+="ROKS"; return ret
        elif( self.spinrestriction == 'R'):
            ret+="RKS";  return ret
        else:
            assert False,"wrong DFT:"+self.spinrestriction


    #> Matrix routines ---
    #> calc_xxx(self, rttddft, MATRICES )
    #>   rttddft.get_xxx(  matrices = (i) INPUT_MATRICES  (ii) time-dpd matrices/wfn self.tdDM etc.)
    def calc_hcore(self,rttddft,time_AU=None, dm1_kOR1=None, tmAU_dmat=None, Dict_hc=None):
        if time_AU is None:time_AU=self.time_AU
        if( dm1_kOR1 is None ):
            dm1_kOR1 = self.tdDM; tmAU_dmat=self.time_AU
        return rttddft.get_hcore(tm_AU = time_AU, dm_kOR1=dm1_kOR1, tmAU_dmat=tmAU_dmat, Dict_hc=Dict_hc)

    def calc_veff(self,rttddft,dm=None,ovrd_tdDM=None):
        if dm is None: dm=self.calc_tdDM(rttddft,ovrd=(True if(ovrd_tdDM is None) else ovrd_tdDM) )
        
        return rttddft.get_veff(dm=dm)

    def calc_fock(self,rttddft,h1e=None,vhf=None,dm=None,time_AU=None,ovrd_tdDM=None):
        if time_AU is None:time_AU=self.time_AU
        if h1e is None: h1e = self.calc_hcore(rttddft,time_AU=time_AU)
        if vhf is None: vhf = self.calc_veff(rttddft,dm=dm,ovrd_tdDM=ovrd_tdDM)
        if dm is None: dm=self.tdDM
        if( self.spinrestriction == 'O'):
            f=rttddft.get_fock( h1e=h1e,vhf=vhf,dm=dm)
        else:
            f = h1e + vhf
        return f


    def calc_tdDM(self,rttddft,ovrd=True):
        if( not self.pbc ):
            # hf.py#make_rdm1 ... this creates complex dm
            ret=rttddft.make_rdm1( mo_coeff=self.tdMO, mo_occ=self.mo_occ )
            ### print_00("calc_tdDM:"+str(np.shape(ret)))
            if( ovrd ):
                Ndim=np.shape(ret)
                if( self.spinrestriction == 'R'):
                    for I in range(Ndim[0]):
                        for J in range(Ndim[1]):
                            self.tdDM[I][J]=ret[I][J]
                else:
                    for I in range(Ndim[0]):
                        for J in range(Ndim[1]):
                            for K in range(Ndim[2]):
                                self.tdDM[I][J][K]=ret[I][J][K]
        else:
            # khf.py
            ret=rttddft.make_rdm1( mo_coeff_kpts=self.tdMO, mo_occ_kpts=self.mo_occ )
            assert (ret is not None),"ret is None"
            if( self.tdDM is None ):
                self.tdDM=np.zeros( np.shape(ret), dtype=np.complex128 )
                print_00("#allocating tdDM:",end="");print_00( np.shape(ret) )
            if( ovrd ):
                Ndim=np.shape(ret);### print_00("#ret:",end="");print_00(Ndim)
                ### print_00("#tdDM:",end="");print_00(np.shape(self.tdDM))
                if( self.spinrestriction == 'R'):
                    for I in range(Ndim[0]):
                        for J in range(Ndim[1]):
                            for K in range(Ndim[2]):
                                self.tdDM[I][J][K]=ret[I][J][K]
                else:
                    for I in range(Ndim[0]):
                        for J in range(Ndim[1]):
                            for K in range(Ndim[2]):
                                for L in range(Ndim[3]):
                                    self.tdDM[I][J][K][L]=ret[I][J][K][L]
        self.check_matrixShape("DM",self.tdDM)
        return self.tdDM

    def calc_gs(self,rttddft,save_canonicalMOs=7,abort_if_failed=None,dm0=None,errbuf=None):
        comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
        calc_Aind_save = self._calc_Aind
        assert (rttddft._calc_Aind == calc_Aind_save),""
        self._calc_Aind=0;
        rttddft._calc_Aind=0;

        if( errbuf is not None ):
            errbuf.clear()

        if( abort_if_failed is None ):
            abort_if_failed = (errbuf is None)  ## abort if errbuf is None

        verbose=True
        if(verbose):
            print_00("#Moldyn:calc_gs:%d"%(rttddft._calc_Aind),flush=True)
        wt0=time.time()
        self.Egnd=rttddft.calc_gs(dm0=dm0)
        wt1=time.time()
        Moldyn_static_.N_calc_gs_+=1
        if( MPIrank==0 and ( (Moldyn_static_.N_calc_gs_ == 1) or (Moldyn_static_.N_calc_gs_ == 5) or \
                             (Moldyn_static_.N_calc_gs_ == 20) or (Moldyn_static_.N_calc_gs_%200==0) ) ):
            rttddft_common.Print_SCFresults(fpath= rttddft_common.get_job(True) +"_SCF.log",Append=False)
        if(verbose):
            print_00("#Moldyn:calc_gs done",flush=True)

        print_eorbs=False
        if( print_eorbs ):
            prteorbs01(True,self,rttddft,description="calc_gs")
            fockmat=get_FockMat(self,rttddft)
            filenames=print_TDeorbs( self.spinrestriction, rttddft.mo_coeff, rttddft, self.pbc, rttddft.mo_coeff,
                   fockmat, rttddft.mo_occ, job=(rttddft_common.get_job(True))+"_GS", Append=False,step=0,tm_au=self.time_AU)
            
        SCFresult=rttddft_common.Get_SCFresult()
        if( SCFresult is not None ):
            if( ('cvgd' in SCFresult) and (SCFresult['cvgd'] is not None) ):
                SCF_cvgd = SCFresult['cvgd']
                if( errbuf is not None ):
                    errbuf.append(-1)

                if( not SCFresult['cvgd'] ):
                    print_00("#!W SCF did not converge",warning=(-1 if(abort_if_failed) else 1));
                    if( abort_if_failed ):
                        assert False,"SCF failed"
                    return None

        rttddft.update_nOrbs()
        self.nkpt=( Moldyn.nkpt_for_non_pbc_ if(not self.pbc) else rttddft.nkpt)
        n_call=rttddft_common.Countup('calc_gs')
        if( self.pbc ):
            fdlog=open_00("calc_gs.log","a")
            mesh=rttddft.cell.mesh
            meshsz=i1prod(mesh)
            strlog=" %4d %5d   %12d %3d %3d %3d   %16.8f  %14.4f"%(\
            n_call, rttddft.nAO, i1prod(mesh),mesh[0],mesh[1],mesh[2], self.Egnd, wt1-wt0)
            strlog+="   "+str(rttddft_common.get_job(True))
            legend="#%4s %5s   %12s %3s %3s %3s   %16s  %14s"%(\
                "call", "nAO", "meshsz","nx","ny","nz", "Egnd", "walltime")
            if(n_call==1):
                print_00(legend,file=fdlog)
            print_00(strlog,file=fdlog);
            close_00(fdlog); ### fdlog.close()
        assert  ( (self.nAO is None) or (self.nAO==rttddft.nAO) ),\
               "wrong nAO:%d/%d"%(self.nAO,rttddft.nAO)
        self.nAO=rttddft.nAO;
        self.nMO=rttddft.nMO;
        print("#calc_gs:MD set:%d %d %d"%(self.nkpt,self.nAO,self.nMO))
        self.tdMO=arrayclone( rttddft.mo_coeff, dtype=np.complex128 )
        self.mo_occ = arrayclone( rttddft.mo_occ )
        prtout_MOocc( rttddft_common.get_job(True)+"_MOocc.dat",rttddft, self.pbc, rttddft._spinrestriction )
        rttddft.set_mo_occ( self.mo_occ, fix_occ=True, clone=False)
        print_00("DFT_%s"%(self.spinrestriction)+"mo_occ:",end="");print_00(np.shape(self.mo_occ))
        print_00("DFT_%s"%(self.spinrestriction)+"mo_cofs:",end="");print_00(np.shape(rttddft.mo_coeff))
        print_00("DFT_%s"%(self.spinrestriction)+"DM:",end="");print_00(np.shape(rttddft.make_rdm1()))
        
        #Ndim=np.shape( rttddft.mo_coeff )
        #if(not self.pbc):
        #    self.nkpt = Moldyn.nkpt_for_non_pbc_
        #    self.nAO = Ndim[0]
        #    self.nMO = Ndim[1]
        #else:
        #    self.nkpt = Ndim[0]
        #    self.nAO = Ndim[1]
        #    self.nMO = Ndim[2]
        dm=rttddft.make_rdm1()
        self.tdDM=arrayclone( dm, dtype=np.complex128 )

        self._canonicalMOs=None;
        self._canonicalEorbs=None
        self._GSfockMat = None
        
        if( self.spinrestriction == 'O'):
            ## note: in ROHF, get_veff returns [2][nAO,nAO] 
            ##       whereas get_fock constructs  (Roothaan Fock mat[nAO,nAO]) plus .focka/.fockb extensions
            FockMat=rttddft.get_fock( h1e=rttddft.get_hcore(tm_AU=self.time_AU) )
        else:
            FockMat=rttddft.get_hcore(tm_AU=self.time_AU) + rttddft.get_veff(rttddft.mol,dm)
        if( save_canonicalMOs & 4 ):
            self._GSfockMat = arrayclone( FockMat )
        if( save_canonicalMOs & 2):
            self._canonicalMOs=arrayclone( rttddft.mo_coeff )
            self._cMOs_Rnuc = arrayclone( self.Rnuc )
        if( save_canonicalMOs & 1):
            if( self.spinrestriction=='U'):
                self._canonicalEorbs = np.array( [ calc_eorbs(rttddft.mo_coeff[0], FockMat[0]),
                        calc_eorbs(rttddft.mo_coeff[1], FockMat[1]) ] )
            elif( self.spinrestriction=='R'):
                self._canonicalEorbs = calc_eorbs(rttddft.mo_coeff, FockMat)
            elif( self.spinrestriction=='O'):
                self._canonicalEorbs = []
                nkp=(1 if(not self.pbc) else self.nkpt)
                print_00( rttddft.mo_coeff )
                for kp in range(nkp):
                    Coefs = ( rttddft.mo_coeff if(not self.pbc) else rttddft.mo_coeff[kp] )
                    fock =  ( FockMat if(not self.pbc) else FockMat[kp] )
                    eorbs_1 = []
                    print_00(np.shape(Coefs));print_00(np.shape(fock))
                    for I in range(self.nMO):
                        eorb=np.vdot( Coefs[:,I], np.matmul( fock, Coefs[:,I])).real
                        if( hasattr( fock, 'focka') ):
                            ea  =np.vdot( Coefs[:,I], np.matmul( fock.focka,Coefs[:,I])).real
                            eb  =np.vdot( Coefs[:,I], np.matmul( fock.fockb,Coefs[:,I])).real
                            eorb = pyscf_lib.tag_array(eorb, mo_ea=ea, mo_eb=eb)
                        eorbs_1.append( eorb )
                    if( not self.pbc ):
                        self._canonicalEorbs = eorbs_1
                    else:
                        self._canonicalEorbs.append( eorbs_1 )
                self._canonicalEorbs = np.array( self._canonicalEorbs )
###            self._canonicalEorbs = arrayclone( rttddft.get_hcore(tm_AU=self.time_AU) + rttddft.get_veff(rttddft.mol,rttddft.make_rdm1()) )

        self._calc_Aind=calc_Aind_save
        rttddft._calc_Aind=calc_Aind_save

        return self.Egnd

""" Note


 #37 methods ...

 svld_energy(self,SorL,pscfpath,eDict=None,Threads=[0]):
 save(self,fpath,delimiter='\n',comment=None,Threads=[0],rttddft=None,caller=None, Barrier=False):
 load(fpath,eDict=None):
 save_tempFockMat(self,src,t2,calc_eigvcs=False):
 delete_Fock_last(self,description="",flag=7):
 get_DFT(self):
 check_matrixShape(self,name,A,nkpt=None,nAO=None,nMO=None):
 get_MOvcs(self):
 print_Fock(self,rttddft,title,FockMat=None,MOvcs=None,Threads=[0]):
 print_DMO(self,title,dmat=None,MOvcs=None,Threads=[0]):
 print_AO(self,header="",Threads=[0]):
 solve_MO(self,rttddft=None,FockMat=None,update=None,comment=None):
 3. gen_mol_or_cell( self, Rnuc=None, unit='B', spin=None, charge=None ):
 1. gen_rttddft(self, mol_or_cell=None, Rnuc=None,unit=None, nKpoints=None, td_field=None, 
 2. gen_rttddft_(self, mol_or_cell=None, Rnuc=None,unit=None, nKpoints=None, td_field=None, 
 set_Rnuc(self,Rnew,unit):
 update_Rnuc(self, Rnew=None, IandJ=None, coordinate=None, update_moldyn=None, fix_occ=True, 
 set_logger(self,logger=None,filename=None,replace=True,rttddft=None):
 get_logger(self,set=False):
 set_time_AU(self,time,step=None,rttddft=None):
 set_step(self,step,rttddft=None):
 calc_hcore(self,rttddft,time_AU=None, dm1_kOR1=None, tmAU_dmat=None, Dict_hc=None):
 calc_veff(self,rttddft,dm=None,ovrd_tdDM=None):
 calc_fock(self,rttddft,h1e=None,vhf=None,dm=None,time_AU=None,ovrd_tdDM=None):
 calc_tdDM(self,rttddft,ovrd=True):
 calc_gs(self,rttddft,save_canonicalMOs=7,abort_if_failed=None,dm0=None,errbuf=None):
 export_DM(self,fpath,dm=None,comment=None):
 read_DM(self,fpath,update_self=False,get_comments=None):
 calc_MOprojection( self, rttddft, header,fpath=None,mo_coeff=None,kpts=None,append=True,dict=None):
 calc_MOprojection01( self, rttddft, header,FileName,MO_Coeff=None,kpts=None,append=True):
 calc_energy_tot(self,rttddft,dm=None,h1e=None,vhf=None,time_AU=None,update_tdDM=None,dict=None,
 transform_tdMO(self,rttddft, ForB, check_norm=True,sqrdevtol=1.0e-6,title="",logger=None):
 calc_MOpopul(self,Rnuc,target_MOs=None,deviations=None):
 print_MOpopul(self,filename,header="",append=False,Rnuc=None,target_MOs=None,deviations=None):
 normalize_MOcofs(self,rttddft,MO_Coeffs=None,update_self=True,devsum_TOL=1.0e-7,
 get_fock(self,rttddft,h1e,dm,time_AU,update_self=True):
 calc_MOpopulations(self,rttddft, target_MOs=None, reference_MOs=None, MO_occ=None):


"""
