import numpy as np
import math
import os
import time
from .futils import futils
from .Logger import Logger
from scipy import linalg
# from physicalconstants import get_constants
from .utils import arrayclone,arrayclonex,print_z3tensor,z1diff,z2diff,d1maxdiff,nmrdff,aNtofile,aNsqrediff,d1diff,calc_eorbs
from .Moldyn import Moldyn
from .Constants import Constants
from .rttddft02 import RTTDDFT_
from mpi4py import MPI
# from pyscf.pbc.grad.krks import Gradients
from .MPIutils01 import MPIutils01
# from pyscf.Loglv import printout
from .calcNacm2 import numrdiff_matrices, convert_dm_AO_to_Orth, convert_dm_Orth_to_AO, get_mu_to_iat, dmtrace,get_derivMatrices_FFTDF
from .heapcheck import heapcheck

import datetime
def aNmaxdiff( lhs,rhs,error_buf=None ):
    if( error_buf is not None):
        error_buf.clear()
    lh=np.ravel(lhs)
    rh=np.ravel(rhs)
    Ld=len(lh);
    if( Ld !=len(rh) ):
        if( error_buf is not None):
            error_buf.append( "dim differs:%d(%s)/%d(%s)"%(Ld,str(np.shape(lhs)),\
                              len(rh),str(np.shape(rhs))) )
            return 1;
    assert (Ld==len(rh)),"dim differs"
    dev=abs(lh[0]-rh[0])
    for I in range(1,Ld):
        dum=abs(lh[I]-rh[I])
        if( dum>dev ):
            dev=dum
    return dev
    
def prtNacm(pbc,spin_restriction,Vdot_Nacm,MOrep=True,refr_MO=None,Vnuc_1D=None,job="",append=True,Istep=-1,tm_au=-1.0,logger=None):
    AUinFS=2.418884326058678e-2
    assert ( spin_restriction=='R' or spin_restriction=='O' or spin_restriction=='U'),"spin_restriction:"+spin_restriction
    ndim_X=np.shape(Vdot_Nacm)  #[ nkpt ] [ nAO ] [ nAO ]
    if( not pbc):
        nkpt=1; nAO=ndim_X[0]
    else:
        nkpt=ndim_X[0]; nAO=ndim_X[1];
    isReal=(np.array(Vdot_Nacm).dtype == float)
    if( not MOrep ):
        absv=np.zeros([nAO,nAO], dtype=np.float64)
        for kp in range(nkpt):
            Mat= Vdot_Nacm if(not pbc) else Vdot_Nacm[kp]
            fnme=( job+"_Nacm.dat" if(not pbc) else job+"_%03d_Nacm.dat"%(kp) )
            fd=futils.fopen(fnme,("a" if append else "w"))
            strVnuc=("" if(Vnuc_1D is None) else "V=%f"%( np.sqrt( np.vdot(Vnuc_1D,Vnuc_1D) ) ) )
            print("#Nacm:%d t=%14.4f a.u. %14.4f fs "%(Istep,tm_au,tm_au*AUinFS) + strVnuc, file=fd)
            for I in range(nAO):
                sbuf=""
                if( isReal ):
                    for J in range(nAO):
                        sbuf+="%9.4f  "%(Mat[I][J])
                else:
                    for J in range(nAO):
                        sbuf+="%7.3f %7.3f     "%(Mat[I][J].real,Mat[I][J].imag)
                print(sbuf,file=fd);
                for J in range(nAO):
                    absv[I][J]=abs(Mat[I][J])
            futils.fclose(fd)
            if( logger is not None ):
                N=len(np.ravel(absv))
                ithTOixj=np.argsort( np.ravel(absv) )
                logger.Info("#Nacm_%d   %6d %14.4f %14.4f   "%( kp, Istep, tm_au, tm_au*AUinFS),end="")
                for ith in range(10):
                    ixj=ithTOixj[ N-ith-1 ];
                    J=ixj%nAO; I=ixj//nAO
                    if( isReal ):
                        logger.Info("%d: %d,%d   %9.4f   "%(ith,I,J,Mat[I][J]),end="")
                    else:
                        logger.Info("%d: %d,%d   %9.3e  (%8.3f+%8.3fj)   "%(ith,I,J,abs(Mat[I][J]),
                                Mat[I][J].real,Mat[I][J].imag),end="")
                logger.Info("")
        return
    
    assert (refr_MO is not None),""
    refMO=( refr_MO if( spin_restriction != 'U' ) else refr_MO[0])
        
    Ndim=np.shape(refMO)
    if( not pbc):
        nkpt=1; nAO=Ndim[0]; nMO=Ndim[1]
    else:
        nkpt=Ndim[0]; nAO=Ndim[1]; nMO=Ndim[2]
    VX_lm=np.zeros( [nMO,nMO], dtype=np.complex128)
    absv=np.zeros([nMO,nMO], dtype=np.float64)
    for kp in range(nkpt):
        Mat= Vdot_Nacm if(not pbc) else Vdot_Nacm[kp]
        MOs= refMO   if(not pbc) else refMO[kp]
        VX_lm= np.matmul( np.matrix.getH(MOs), np.matmul( Mat, MOs ) )
        fnme=( job+"_Nacm.dat" if(not pbc) else job+"_%03d_Nacm.dat"%(kp) )
        fd=futils.fopen(fnme,("a" if append else "w"))
        strVnuc=("" if(Vnuc_1D is None) else "V=%f"%( np.sqrt( np.vdot(Vnuc_1D,Vnuc_1D) ) ) )
        print("#Nacm:%d t=%14.4f a.u. %14.4f fs "%(Istep,tm_au,tm_au*AUinFS) + strVnuc, file=fd)
        for I in range(nMO):
            sbuf=""
            for J in range(nMO):
                sbuf+="%8.3f %8.3fj      "%(VX_lm[I][J].real, VX_lm[I][J].imag)
                absv[I][J]=abs(VX_lm[I][J])
            print(sbuf,file=fd);
        futils.fclose(fd)
        if( not pbc ):
            os.system("ls -ltrh "+fnme);
        if( logger is not None ):
            N=len(np.ravel(absv))
            ithTOixj=np.argsort( np.ravel(absv) )
            logger.Info("#Nacm_%d   %6d %14.4f %14.4f   "%( kp, Istep, tm_au, tm_au*AUinFS),end="")
            for ith in range(10):
                ixj=ithTOixj[ N-ith-1 ];
                J=ixj%nMO; I=ixj//nMO
                logger.Info("%d: %d,%d   %9.3e  (%8.3f+%8.3fj)   "%(ith,I,J,abs(VX_lm[I][J]),
                            VX_lm[I][J].real,VX_lm[I][J].imag),end="")
            logger.Info("")
    if( pbc ):
        os.system("ls -ltrh "+job+"_???_Nacm.dat")

        
##
## \sum [ occ_n eorb_n C^{\mu}_n d S_{\mu\nu}/dR C^{\nu}_n ]
## note:(i)   \nabla E|_{internal} - trCXCe  = \nabla E_{GS}
##      (ii)  trCXCe = trFSXD   at t=0 
def calc_trCXCe(this,MOcofs,MOengs,MOocc,fixedAtoms=None):
    TINY=1.0e-20
    Nfix=(0 if(fixedAtoms is None) else len(fixedAtoms))
    assert ( RTTDDFT_.is_rttddft(this) ),"this"+str(type(this))
    pbc = this._pbc
    kvectors = None if(not pbc) else np.reshape(this.kpts, (-1,3))
    nkpt=(1 if(not pbc) else len(kvectors) )
    nAO =(len(MOcofs) if(not pbc) else len(MOcofs[0]))
    Nat = ( len(this.mol._atom) if(not pbc) else len(this.cell._atom))
    Nat_eff= Nat-Nfix
    Ndir_eff=Nat_eff*3
    ret_buf=[]

    cell_or_mol = (this.mol if(not pbc) else this.cell)
    bfX_al=calc_nacm_AOrep(this,cell_or_mol,pbc,fixedAtoms=fixedAtoms)
    ## 2021.03.21 : iterator -dir- moves within [0,3*Nat_eff)
    for kp in range(nkpt):
        Cofs=( MOcofs if(not pbc) else MOcofs[kp])
        eorb=( MOengs if(not pbc) else MOengs[kp])
        occ =( MOocc  if(not pbc) else MOocc[kp] )
        bfX =( bfX_al if(not pbc) else bfX_al[kp]) 
        nMO = len(occ)
        #\sum_a w_a
        vec=np.zeros( [Ndir_eff] )
        vec[:]=0.0

        dSdR=np.zeros([ Ndir_eff,nAO,nAO],dtype=np.complex128)
        
        for dir in range(Ndir_eff):
            for mu in range(nAO):
                for nu in range(nAO):
                    dSdR[dir][mu][nu] = bfX[dir][mu][nu] +np.conj( bfX[dir][nu][mu] )
        
        for mo in range(nMO):
            if( occ[mo]< TINY ):
                continue
            for dir in range(Ndir_eff):
                vec[dir]+= occ[mo]*eorb[mo] * ( np.vdot(Cofs[:,mo], np.matmul( dSdR[dir],Cofs[:,mo])) ).real
        if( not pbc ):
            return vec   ## non-pbc ends here
        else:
            ret_buf.append(vec)
    assert pbc,"otherwise never come here"
    return ret_buf

# non-pbc :  complex128-ndarray[3*Nat,nAO,nAO]
# pbc     :  complex128-ndarray[nkpt,3*Nat,nAO,nAO]
#
# SCHEME_1 (default) : nmrdiff keeping AO-base density matr
# SCHEME_2 : nmrdiff keeping ORTH_AO-base density matr (i.e. \sqrt{S} \rho_AO \sqrt{S} )
#
# in SCHEME_2, trFSXD has additional tr[ - S^{-1/2} \nabla Ssqrt \rho H - \rho \nabla Ssqrt S^{-1/2} H ]
# 2021.12.01 :: def calc_trFSXD(pbc,spinrestriction,this,FockMat,DensityMat,SCHEME_2=False,fixedAtoms=None):
def calc_trFSXD(pbc,spinrestriction,this,FockMat,DensityMat,SCHEME_2=None,fixedAtoms=None):
    assert SCHEME_2 is not None,""
    Nfixed=( 0 if(fixedAtoms is None) else len(fixedAtoms) )
    # tr[ F(Sinv bfX D + D bfY Sinv) ] ... to be subtracted from dE/dR|_{Dmat-fixed}
    assert ( RTTDDFT_.is_rttddft(this) ),"this"+str(type(this))
    ### calc_nacm_.check_scheme( (2 if(SCHEME_2) else 1) )
    mol_or_cell = ( this.mol if(not pbc) else this.cell )
    Nat=len(mol_or_cell._atom)
    Nat_eff=Nat-Nfixed
    dirALL=[];
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        for J in range(3):
            dirALL.append(3*Iat+J)
    Ndir_eff=len(dirALL)
    ndim_DM=np.shape(DensityMat) ## non PBC:[nAO][nAO]  PBC:[nkpt][nAO][nAO]
    nmult_DM=(1 if(spinrestriction=='R') else 2)
    if( spinrestriction=='R'):
        nAO=(ndim_DM[0] if(not pbc) else ndim_DM[1]) 
    else:
        nAO=(ndim_DM[1] if(not pbc) else ndim_DM[2])
    assert ((this.nAO is None) or nAO==this.nAO),"nAO %d %d"%(nAO,this.nAO)
    ret=np.zeros([3*Nat_eff])
    for I in range( 3*Nat_eff ):
        ret[I]=0.0

    if( spinrestriction=='O'):
        if( not pbc ):
            assert hasattr(FockMat,'focka'),"focka"
        else:
            assert hasattr(FockMat[0],'focka'),"focka"

    this.update_Sinv()
    nmult=(1 if(spinrestriction=='R') else 2)
    if( not pbc ):
        Sinv = this._sINV
        bfX=calc_nacm_AOrep(this,this.mol,pbc,fixedAtoms=fixedAtoms) #([nkpt])[3*Nat][nAO][nAO]
        
        ## dagger w.r. nu,mu
        bfY=np.zeros([3*Nat_eff, nAO,nAO],dtype=np.complex128)
        for kth_dir in range(3*Nat_eff):
            for mu in range(nAO):
                for nu in range(nAO):
                    bfY[kth_dir][mu][nu] =np.conj( bfX[kth_dir][nu][mu] )

        if( SCHEME_2 ):
            nabla_Ssqrt=calc_nabla_Ssqrt( this._Ssqrt, (bfX + bfY))
        
        wks=np.zeros( [nAO, nAO], dtype=np.array(this._Ssqrt).dtype )

        if( SCHEME_2 ):
            #assert False,"-- please confirm you are using this option in a consistent manner --" ## please comment out this assertion after confirmation..
            assert (spinrestriction=='R'),"please check below otherwise"
            for kth_dir in range(Ndir_eff):  ### range(3*Nat):
                wks = np.matmul( this._Ssqrt, nabla_Ssqrt[kth_dir] )
                for mu in range(nAO):
                    for nu in range(nAO):
                        bfX[ kth_dir ][mu][nu]-=wks[mu][nu] ## S^{1/2}\nabla S^{1/2}
                wks = np.matmul( nabla_Ssqrt[ kth_dir ], this._Ssqrt )
                for mu in range(nAO):
                    for nu in range(nAO):
                        bfY[ kth_dir ][mu][nu]-=wks[mu][nu] ## \nabla S^{1/2} S^{1/2}
            ### in fact these two must cancel each other since orthogonalized S matrix should equal unity..
            dum=bfX + bfY
            #print(np.shape(dum))
            #print(np.shape( max( abs(dum))))
            dev=max( abs( np.ravel(dum)) );assert dev<1.0e-6,"bfX_Lowdin + bfY_Lowdin /= 0 :%e"%(dev)
        Fockmatrices=FockMat
        if( spinrestriction=='O' and hasattr(FockMat,'focka') ):
            Fockmatrices=[ FockMat.focka, FockMat.fockb ]
        for dir in range(3*Nat_eff):
            ret[dir]=0.0
            for sp in range(nmult):
                DM=( DensityMat if(nmult==1) else DensityMat[sp] )
                FM=( Fockmatrices if(nmult==1) else Fockmatrices[sp])
                wks=np.matmul( np.matmul(Sinv, bfX[dir]),DM ) +\
                   np.matmul( np.matmul( DM, bfY[dir]),Sinv )
                wks=np.matmul( FM, wks)
            ## ... and Trace...
                csum=np.complex128(0.0)
                for mu in range(nAO):
                    csum+= wks[mu][mu]
                ret[dir]+=csum.real
        return ret;
    else:
        assert (not SCHEME_2),""   ## PLS check carefully and comment out this assertion ..
        ## do the same thing over k points...
        kvectors=np.reshape( this.kpts, (-1,3))
        nkpt=len(kvectors)
        if( nmult == 1 ):
            assert (ndim_DM[0]==nkpt and ndim_DM[1]==ndim_DM[2]),"ndim_DM:"+str(ndim_DM)
        else:
            assert (ndim_DM[0]==2 and ndim_DM[1]==nkpt and ndim_DM[2]==ndim_DM[3]),"ndim_DM:"+str(ndim_DM)
        bfX_kpts = calc_nacm_AOrep(this,this.cell,pbc,fixedAtoms=fixedAtoms)
        for dir in range(3*Nat_eff):
            ret[dir]=0.0

        for kp in range(nkpt):
            S1k  = this.get_ovlp(this.cell,kvectors[kp])
            Sinv = this._sINV[kp]
            bfX  = bfX_kpts[kp]
            
            bfY = np.zeros([3*Nat_eff,nAO,nAO],dtype=np.complex128)
            for dir in range(3*Nat_eff):
                for mu in range(nAO):
                    for nu in range(nAO):
                        bfY[dir][mu][nu] =np.conj( bfX[dir][nu][mu] )

            FockMatrices_kp=FockMat[kp]
            if( spinrestriction=='O' and hasattr(FockMatrices_kp,'focka') ):
                FockMatrices_kp=[ FockMatrices_kp.focka, FockMatrices_kp.fockb ]
            elif( spinrestriction == 'U'):
                FockMatrices_kp=[ FockMat[0][kp], FockMat[1][kp] ]

            for dir in range(3*Nat_eff):
                for sp in range(nmult):
                    DM=(DensityMat[kp] if(nmult==1) else DensityMat[sp][kp])
                    FM=(FockMatrices_kp if(nmult==1) else FockMatrices_kp[sp])
                    wks=np.matmul( np.matmul(Sinv, bfX[dir]),DM ) +\
                       np.matmul( np.matmul( DM, bfY[dir]),Sinv )
                    wks=np.matmul( FM, wks)
                    ## ... and Trace...
                    csum=np.complex128(0.0)
                    for mu in range(nAO):
                        csum+=wks[mu][mu]
                    ret[dir]+=csum.real   ## sum over kpoints
        for dir in range(3*Nat_eff):
            ret[dir]=ret[dir]/float(nkpt)
        return ret;


def calcForce(md,rttddft,SCHEME_2, dm_kpts=None, fock_kpts=None, time_AU=None, Decomposition=None, flag_nmrdiff=None, 
              order_nmdiff=5, displ_nmdiff=0.02, debug=0, dict=None, fixedAtoms=None, force_ref=None, timing=None):
    heapcheck(">> calcForce.START")
    if( timing is None ):
        timing={}
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    timing.update({"prep":None,"nmdiff":None,"trFSXD":None,"analytic":None})
    print("#calcForce::$%02d:Start fixedAtoms:"%(MPIrank),fixedAtoms,flush=True)
    Logger.write(md.logger,"#start calcForce..\n#MD_Rnuc:"+str(md.Rnuc) +"\n#rt_Rnuc:"+ str(rttddft.get_Rnuc('B')))
    cput0=time.time();cput1=cput0
    if time_AU is None: time_AU = md.time_AU
    if dm_kpts is None: dm_kpts = md.calc_tdDM(rttddft,ovrd=True)
    if(fock_kpts is None):
        fock_kpts = md.calc_fock(rttddft,dm=dm_kpts,time_AU=time_AU)
    assert ( (md is not None) and isinstance(md,Moldyn) ),"Moldyn"

# flag_nmrdiff : 0 no / 1 yes, return numeric / 2 yes, return analytic    
    if( flag_nmrdiff is None ): flag_nmrdiff=(0 if( md.df=="FFTDF" ) else 1) 

    ### calc_nacm_.check_scheme( (2 if(SCHEME_2) else 1) )

    pbc=md.pbc
    mol_or_cell = (rttddft.mol if(not pbc) else rttddft.cell)
    cput2=cput1;cput1=time.time();timing["prep"]=cput1-cput2
    dbgng_nmgrads = False; gr1A=None
    if( md.df=="FFTDF" ):
        gr1A_decomposition={};
        if( dbgng_nmgrads ):
            nmgrads = numrdiff_matrices( md, rttddft, dmat=dm_kpts, fixedAtoms=fixedAtoms)
            cput2=cput1;cput1=time.time();timing["nmdiff"]=cput1-cput2
            gr1A,gr1Anmr=calcGrad_analytic_FFTDF(md,rttddft,dmat=dm_kpts, fixedAtoms=fixedAtoms,nmgrads=nmgrads,decomposition=gr1A_decomposition)
            cput2=cput1;cput1=time.time();timing["analytic"]=cput1-cput2
        else:
            print("#calcForce::$%02d:start calcGrad_analytic_FFTDF "%(MPIrank)+str(datetime.datetime.now()),flush=True)
            gr1A=calcGrad_analytic_FFTDF(md,rttddft,dmat=dm_kpts, fixedAtoms=fixedAtoms,nmgrads=None,decomposition=gr1A_decomposition)
            cput2=cput1;cput1=time.time();timing["analytic"]=cput1-cput2
            print("#calcForce::$%02d:end calcGrad_analytic_FFTDF "%(MPIrank)+str(datetime.datetime.now()),flush=True)
        if( fixedAtoms is not None ):
            spdm=3
            Nat=md.Nat; wks=np.zeros([ spdm*Nat ],dtype=np.float64)
            IxJ=0
            for I in range(Nat):
                if( I in fixedAtoms ):
                    continue
                for J in range(spdm):
                    wks[ spdm*I + J ]=gr1A[ IxJ + J ]
                IxJ+=spdm
            assert IxJ==len(gr1A),"%d / %d"%(IxJ,len(gr1A))
            gr1A = wks

        if( flag_nmrdiff == 0 ):
            print("#calcForce:analytic::$%02d:END"%(MPIrank), cput1-cput2, cput1-cput0)
            heapcheck(">> calcForce.END")
            return -gr1A
## XXX XXX  sorry, this does NOT equal to numdiff ...   gr1A=calcGrad_analytic(md,rttddft,dmat=dm_kpts)
## XXX XXX   cput2=cput1;cput1=time.time();timing["analytic"]=cput1-cput2

    print("#calcForce::$%02d:Start calcgrad_nmdiff"%(MPIrank), cput1-cput2, cput1-cput0)
    gr1_ref={}
    gr1=calcgrad_nmdiff(md, rttddft,dmat=dm_kpts,order=order_nmdiff, SCHEME_2=SCHEME_2,
                        displ=displ_nmdiff, fixedAtoms=fixedAtoms,grad_ref=gr1_ref)
    cput2=cput1;cput1=time.time();timing["nmdiff"]=cput1-cput2
    print("#calcForce::$%02d:End calcgrad_nmdiff"%(MPIrank),cput1-cput2, cput1-cput0)

#    gr1A=calcGrad_analytic(md,rttddft,dmat=dm_kpts)
#    cput2=cput1;cput1=time.time();timing["analytic"]=cput1-cput2
#    dev=max( abs(gr1A-gr1) )
#    printout("analyticgrad:dev=%e walltime:%f / %f"%(dev,timing["analytic"],timing["nmdiff"]),\
#             fpath="calcNacm_analyticgrad.log");
    if( Decomposition is not None):
        Decomposition.update({"F1":arrayclonex(gr1,-1)})

    gr2=calc_trFSXD(md.pbc, md.spinrestriction, rttddft,fock_kpts,dm_kpts, fixedAtoms=fixedAtoms, SCHEME_2=SCHEME_2)  ## should we store those matrices ?? 
    if( Decomposition is not None):
        Decomposition.update({"F2":arrayclone(gr2)})
    cput2=cput1;cput1=time.time();timing["trFSXD"]=cput1-cput2
    Logger.write(md.logger,"calcForce:"+str(timing))
    print("#calcForce::$%02d:End trFSXD"%(MPIrank), cput1-cput2, cput1-cput0)

    logfpath="calcForce_walltime_%02d.log"%(MPIrank)
    fdOU=open(logfpath,"a")
    print("#Force:$%02d:%14.4f  "%( MPIrank,cput1-cput0)+("" if(Decomposition is None) else str(Decomposition)) \
          +" \t\t"+str(datetime.datetime.now()), file=fdOU)
    fdOU.close()

    if( debug>0 and md.spinrestriction=='R' and (not md.pbc) and fixedAtoms is None):
        if( (md._canonicalMOs is not None) and (md._canonicalEorbs is not None)  ):  
            gr2b=calc_trCXCe( rttddft, md._canonicalMOs, md._canonicalEorbs, md.mo_occ)
        eorbs=calc_eorbs( md.tdMO, md.calc_fock(rttddft) ) 
        gr2c=calc_trCXCe( rttddft, md.tdMO, eorbs, md.mo_occ )
        if( dict is not None ):
            dict["diff_gr2c"]=d1diff(gr2c,gr2);
            dict["diff_gr2"]=d1diff(gr2b,gr2);dict["Frc_gr2"]=gr1-gr2b
        if( debug>1 ):
            print("grad_EGND:",end="");print(gr1-gr2b)
            aNtofile('S',"grad_EGND.tmpdat",dbuf=gr1-gr2b)
            print("#gr1:",end="");print(gr1)
            print("#gr2:",end="");print(gr2)   #gr2: [ 5.21810413e-05  3.51999785e-01  1.75021151e-14 -2.03708274e-01
            print("#gr2b:",end="");print(gr2b) #gr2b:[ 5.21851825e-05  3.51999740e-01  1.35953869e-15 -2.03708634e-01
        #for I in range(len(gr1)):
        #    gr1[I] -= gr2b[I]
        #for I in range(len(gr1)):
        #    gr1[I] = -gr1[I]
        #return gr1
    cput2=cput1; cput1=time.time();
    print("#calcForce::$%02d:XXX DEBUG"%(MPIrank), cput1-cput2, cput1-cput0)
        

    for I in range(len(gr1)):
        gr1[I] -= gr2[I]

    if( force_ref is not None ):
        for ky in gr1_ref:
            for I in range(len(gr1_ref[ky])):
                gr1_ref[ky][I] -= gr2[I]          ## gr1-gr2
            for I in range(len(gr1_ref[ky])):
                gr1_ref[ky][I] = -gr1_ref[ky][I]  ## grad->Force
            force_ref.update( {ky:gr1_ref[ky]} );
    ## now we obtained an EFFECTIVE grad ...
    ## F = - GRAD
    for I in range(len(gr1)):
        gr1[I] = -gr1[I]

#    print("#calcForce::",gr1)
    print("#calcForce:numdiff::$%02d:END"%(MPIrank), cput1-cput2, cput1-cput0)
    if( gr1A is not None ):
        frc1A = -gr1A
        diff=max( abs( np.ravel(frc1A) - np.ravel(gr1) ) )
        print("#calcForce:comp_numdiff_to_analytic:%e "+str(gr1)+"/"+str(frc1A))
        return (gr1 if(flag_nmrdiff==1) else frc1A)
    return gr1

#0 Li     0.2391259237    -0.0000000000    -0.0000000000
#1 H    -0.2391259237    -0.0000000000    -0.0000000000
#calcForce:: [-2.39178948e-01  3.21501437e-13 -1.33673763e-13  
#2.39170116e-01 8.31081388e-13  2.59211751e-13]
#analyticGrad: [-5.03771029e-01  1.59390934e-11 -4.85982241e-12  5.03762197e-01
# -1.61894379e-11  4.49591261e-12]
#analyticGradB: [-2.53758794e-01  1.13824975e-11 -3.43626669e-12  2.53749962e-01
# -1.22217578e-11  3.05044382e-12]

# returns grad[3*Nat] float ndArray newly allocated
def calcgrad_nmdiff_singlethread(md,rttddft,dmat=None,order=3,displ=0.02,debug=False,SCHEME_2=None,fixedAtoms=None,grad_ref=None):
    ## dbgng_Eel=False ## 
    assert SCHEME_2 is not None,""
    Nfixed=( 0 if(fixedAtoms is None) else len(fixedAtoms) )
    cput00=time.time();cput1=cput00
    if dmat is None: dmat = md.calc_tdDM(rttddft)
    ### calc_nacm_.check_scheme( (2 if(SCHEME_2) else 1) )
    dm_orth=( None if(not SCHEME_2) else convert_dm_AO_to_Orth( rttddft._Ssqrt, dmat, rttddft._pbc, rttddft._spinrestriction ) )
    dm_AOrep=( None if(SCHEME_2) else arrayclone(dmat) )

    assert isinstance(md,Moldyn),"moldyn"
    assert ( RTTDDFT_.is_rttddft(rttddft) ),"rttddft"+str(type(rttddft))
    pbc = md.pbc;  Nat=md.Nat
    hfo=order//2
# _atm: [[ 8 20  1 23  0  0] [ 1 24  1 27  0  0] [ 1 28  1 31  0  0]]  # pointer addresses
# _atom:[('O', [4.7940462054091055, -0.29290754930758456, 0.0]), ('H', [5.808829134300543, 0.29290754930758456, 0.0]), ('H', [3.7794522491301237, 0.29290754930758456, 0.0])]
    Nat_eff=Nat-Nfixed
    grad = np.zeros([3*(Nat-Nfixed)])
    Rnuc_o = np.zeros([Nat,3]);
    Rref=rttddft.get_Rnuc('B')   ## in a.u. 
    ## if( not pbc ):
    ##    Rref=rttddft.mol.atom_coords()
    ## else:
    ##    Rref=rttddft.cell.atom_coords()
    if(md.Rnuc is None):
        print("#setting Rnuc..");md.Rnuc=arrayclone(Rref)
    Rdev = d1maxdiff( np.ravel(Rref), np.ravel(md.Rnuc) )
    assert (Rdev<1.0e-6),"rttddft.R differs from md.Rnuc ..."+str(Rref)+"/"+str(md.Rnuc)
    for Iat in range(Nat):
        for Jdir in range(3):
            Rnuc_o[Iat][Jdir]=Rref[Iat][Jdir]
    R_wrk=np.zeros([Nat,3],dtype=np.float64)
    for Iat in range(Nat):
        for Jdir in range(3):
            R_wrk[Iat][Jdir]=Rnuc_o[Iat][Jdir]
## 
## Here we evaluate Energy derivative due to Rnuc difference but NOT related to density matrix change
##
    if( grad_ref is None ):
        grad_ref=(None if(order<=3) else {})
    else:
        grad_ref.clear();
    if( order == 5 ):
        grad_ref.update( {"5p":np.zeros([Nat_eff,3]),"3p":np.zeros([Nat_eff,3])} )
    elif( order==7 ):
        grad_ref.update( {"7p":np.zeros([Nat_eff,3]),"5p":np.zeros([Nat_eff,3]),"3p":np.zeros([Nat_eff,3])})


##    grad_ref=( None if(order<=3) else (\
##                {"5p":np.zeros([Nat_eff,3]),"3p":np.zeros([Nat_eff,3])} if(order<=5) else \
##                    {"7p":np.zeros([Nat_eff,3]),"5p":np.zeros([Nat_eff,3]),"3p":np.zeros([Nat_eff,3])}) )
    IJ=0; Iat_eff=-1
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        Iat_eff+=1
        for Jdir in range(3):
            Earr=[]
            for step in range(-hfo,hfo+1):
                if( step==0 ):
                    Earr.append(None);continue
                calc=md.update_Rnuc( IandJ=[Iat,Jdir],coordinate=Rnuc_o[Iat][Jdir]+ step*displ, update_moldyn=False)
                R_wrk[Iat][Jdir]=Rnuc_o[Iat][Jdir]+ step*displ
               ## print("#%02d,%02d : %12.6f -> %12.6f"%(Iat,Jdir,Rnuc_o[Iat][Jdir],Rnuc_o[Iat][Jdir]+step*displ))
                E_nuc= calc.energy_nuc(Rnuc_au=R_wrk);
                if( SCHEME_2 ):
                    calc.update_Sinv()
                    dm_AOrep= convert_dm_Orth_to_AO( calc._Sinvrt, dm_orth, calc._pbc, calc._spinrestriction)

                assert abs( calc._time_AU-md.time_AU)<1.0e-7,"time_AU:%f/%f"%(calc._time_AU,md.time_AU)

                h1e_new=calc.get_hcore()             ## this potentially depends on : calc._time_AU
                vhf_new=calc.get_veff(dm=dm_AOrep)   ## rttddft(MOL/PBC) takes "dm" as input
# energy_elec_rks(this, dm=None, h1e=None, vhf=None, verbose=False):
# energy_elec_krks(this, dm_kpts=None, h1e_kpts=None, vhf=None,verbose=False):
                E_el = ( calc.energy_elec( dm=dm_AOrep, h1e=h1e_new, vhf=vhf_new )[0] if(not pbc) else \
                         calc.energy_elec( dm_kpts=dm_AOrep, h1e_kpts=h1e_new, vhf=vhf_new ) [0] )

                ## if(dbgng_Eel):
                ##    trH1xDM=calc_dAOtrace(md,rttddft,dm_AOrep,h1e_new)
                ##    vCoul=vhf_new.ecoul
                ##    vXC=vhf_new.exc
                ##    ref=(trH1xDM + vCoul + vXC);assert abs(E_el-ref)<1e-6,""

                cput2=cput1;cput1=time.time()
                         ## h1e, vhf are left default
                # print("E_nuc:",end="");print(E_nuc)
                # print("E_el:",end="");print(E_el)
                E_tot= E_el + E_nuc
                if( cput1-cput2>100):
                    print("#calcgrad_nmdiff:%d.%d. %d %f %f  elapsed:%10.2f %10.2f"%(Iat,Jdir,step,step*displ,E_tot,cput1-cput2,cput1-cput00))
                if( debug ):
                    assert False,"dbg"
                    calc._fix_occ=False
                    E_gnd=calc.calc_gs()
                    Earr.append( np.array( [E_tot,E_el,E_nuc,E_gnd] ))
                else:
                    Earr.append( np.array( [E_tot,E_el,E_nuc] ))
                R_wrk[Iat][Jdir]=Rnuc_o[Iat][Jdir]
            dict={"7p":None,"5p":None,"3p":None}
            dNp = nmrdff( order,Earr,len(Earr[0]),displ,dict=dict)
            grad[ Iat_eff*3+Jdir ]=dNp[0]
            if( debug ):
                print("Egnd_grad:%d %f   %f %f %f"%(Iat_eff*3+Jdir,dNp[3],dict["7p"][3],dict["5p"][3],dict["3p"][3]) )
            if( order>3 ):
                grad_ref["3p"][Iat_eff][Jdir]=dict["3p"][0];
                grad_ref["5p"][Iat_eff][Jdir]=dict["5p"][0];
            if( order>5 ):
                grad_ref["7p"][Iat_eff][Jdir]=dict["7p"][0];
        if( order > 5 ):
            dev75=aNsqrediff( grad_ref["7p"], grad_ref["5p"])
            dev73=aNsqrediff( grad_ref["7p"], grad_ref["3p"])
            dev53=aNsqrediff( grad_ref["5p"], grad_ref["3p"])

            print("#grad_dev:h=%e 7-5:%e 5-3:%e 7-3:%e"%(\
                displ, math.sqrt(dev75), math.sqrt(dev53), math.sqrt(dev73) ))
        elif( order>3):
            dev53=aNsqrediff( grad_ref["5p"], grad_ref["3p"])
            print("#grad_dev:h=%e 5-3:%e"%(displ, math.sqrt(dev53)))
    assert Iat_eff+1==Nat_eff,"Iat_eff+1:%d / Nat_eff=%d"%(Iat_eff+1,Nat_eff)

    cput1=time.time() 
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank();
    logfpath="calcForce_walltime_%02d.log"%(MPIrank)
    fdOU=open(logfpath,"a")
    print("## calcgrad_nmdiff_singlethread:$%02d:%14.4f  "%(MPIrank,cput1-cput00)
          +" \t\t"+str(datetime.datetime.now()), file=fdOU)
    fdOU.close()

    return grad


"""
mole.py L308:
    ._atom contains  [[atom1, (x, y, z)],...  
    if you cannot find it, you can invoke   .format_atom(self.atom, unit=self.unit)
mole.py L387:
    ._basis {'H': [[0,
        [3.4252509099999999, 0.15432897000000001],
        [0.62391373000000006, 0.53532813999999995],
        [0.16885539999999999, 0.44463454000000002]]], etc.

_bas.append([atom_id, angl, nprim, nctr, kappa, ptr_exp, ptr_coeff, 0])
"""
# mu_to_iat:[0 0 0 0 0 0 0 0 0 1 1 2 2]

def calc_nabla_Ssqrt(Ssqrt,dSdR):
    # dSdR[3*Nat][nAO][nAO]
    # Ssqrt X + X Ssqrt = dSdR
    Nx3=len(dSdR)
    Ndim=np.shape(dSdR)
    ret=[]
    for dir in range(Nx3):
        ret.append( linalg.solve_sylvester( Ssqrt, Ssqrt, dSdR[dir]) )
    dbgng=True
    if(dbgng):
        devs=[]
        for dir in range(Nx3):
            wrk=np.matmul( Ssqrt, ret[dir]) + np.matmul( ret[dir], Ssqrt)
            devs.append( z2diff( wrk, dSdR[dir]) )
        assert( max(devs)<1.0e-5),""
        fd1=open("solve_Sylvester.log","w")
        print("#maxdev:%e"%(max(devs))+str(devs),file=fd1);
        for dir in [0,1,Nx3-1]:
            Nd=np.shape(ret[dir])
            for i in range(Nd[0]):
                for j in range(Nd[1]):
                    print("%12.6f "%(ret[dir][i][j]),end="",file=fd1)
        fd1.close()
        Logger.write_maxv(None,"nabla_Ssqrt_dev",max(devs))
        print("#nabla_Ssqrt:devs:",end="");print(devs)
    return ret


##
## -i Vdot_nacm_eff should be Hermitian
##
## 2020.11.28 revision : you do not need MD for this calculation
def calc_Vdot_nacm_eff(this,Vnuc_1d, Nacm_rep, check_antihermicity=True):
    ## TODO : this could be replaced by some tensor_product routine
    assert (Nacm_rep=='A' or Nacm_rep=='O'),""
    Ndir=len(Vnuc_1d)
    assert ( RTTDDFT_.is_rttddft(this) ),"this"+str(type(this))
    pbc = this._pbc
    mol_or_cell = (this.mol if(not pbc) else this.cell)
    bfX_al = calc_nacm_eff(this,mol_or_cell,Nacm_rep) # nkpt Ndir nAO nAO
    ndim=np.shape(bfX_al)
    nkpt=(1 if(not pbc) else ndim[0])
    nAO=ndim[2]
    if( pbc ):
        assert (nAO==ndim[3]),""
        ret=np.zeros([nkpt,nAO,nAO],dtype=np.complex128)
        for kp in range(nkpt):
            for I in range(nAO):
                for J in range(nAO):
                    dir=0
                    ret[kp][I][J]=bfX_al[kp][dir][I][J]*Vnuc_1d[dir]
                    for dir in range(1,Ndir):
                        ret[kp][I][J]+=bfX_al[kp][dir][I][J]*Vnuc_1d[dir]
                    
    else:               
        assert (nAO==ndim[1]),""
        ret=np.zeros([nAO,nAO],dtype=np.complex128)
        for I in range(nAO):
            for J in range(nAO):
                dir=0
                ret[I][J]=bfX_al[dir][I][J]*Vnuc_1d[dir]
                for dir in range(1,Ndir):
                    ret[I][J]+=bfX_al[dir][I][J]*Vnuc_1d[dir]

    if( check_antihermicity ):
        # Note: S^{-1/2}[ \bfX - S^{1/2}\nabla S^{1/2} ] S^{-1/2} 
        #       + S^{-1/2}[ \bfX^{\dagger} - \nabla S^{1/2} S^{1/2} ] S^{-1/2}
        #       = S^{-1/2}[ \nabla S - \nabla S ] S^{-1/2} = 0
        # 
        dict={"devsum":None,"dagdev":None,"ofddev":None,"dagAt":None,"ofdAt":None}
        if( not pbc ):
            ok=check_Hermicity(-1,ret,dict)
            assert ok,"antiHermicity"
    return ret

#    dict={"devsum":None,"dagdev":None,"ofddev":None,"dagAt":None,"ofdAt":None}
def check_Hermicity(type,A,dict=None,TOL=1.0e-6,ERR=None):
    ##
    ## dev : h[\mu][\nu] - ( - np.conj(h[\nu][\mu]) )
    ##   diag : Re(h[\mu][\mu])
    ##   ofd  : ...
    ##
    Ndim=np.shape(A)
    assert Ndim[0]==Ndim[1],""
    dagdev=-1;dagAt=-1; dag_as=0;
    ofddev=-1;ofdAt=[-1,-1];ofd_as=None;
    devsum=0.0
    if( type == 1 ):
        for I in range(Ndim[0]):
            for J in range(I+1):
                if(I==J):
                    dev=abs(A[I][J].imag); devsum+=dev**2
                    if( dev>dagdev ):
                        dagdev=dev;dagAt=I;dag_as=A[I][J]
                else:
                    dev=abs(A[I][J] - np.conj( A[J][I] )); devsum+=dev**2
                    if( dev>ofddev ):
                        ofddev=dev; ofdAt=[I,J]; ofd_as=[ A[I][J], A[J][I]]
    elif( type == -1):
        for I in range(Ndim[0]):
            for J in range(I+1):
                if(I==J):
                    dev=abs(A[I][J].real); devsum+=dev**2
                    if( dev>dagdev ):
                        dagdev=dev;dagAt=I;dag_as=A[I][J]
                else:
                    dev=abs(A[I][J] + np.conj( A[J][I] )); devsum+=dev**2
                    if( dev>ofddev ):
                        ofddev=dev; ofdAt=[I,J]; ofd_as=[ A[I][J], A[J][I]]
    else:
        assert False,"wrong input"
    if( dict is not None ):
        for ky in dict:
            if( ky=="devsum"):
                dict[ky]=devsum
            if( ky=="dagdev"):
                dict[ky]=dagdev
            if( ky=="ofddev"):
                dict[ky]=ofddev
            if( ky=="dagAt"):
                dict[ky]=dagAt
            if( ky=="ofdAt"):
                dict[ky]=ofdAt
    dev=np.sqrt(devsum)
    retv = True
    if( dev > TOL or dagdev>TOL or ofddev>TOL):
        print("#check_Hermicity:dagdev:%e ([%d][%d]:%f+j%f)"%(dagdev,dagAt,dagAt,dag_as.real,dag_as.imag)+\
              "ofddev:%e [%d][%d] %f+j%f vs %f+j%f"%(ofddev,ofdAt[0],ofdAt[1],ofd_as[0].real,ofd_as[0].imag,\
                                                     ofd_as[1].real,ofd_as[1].imag))
        retv=False
    if( ERR is not None ):
        if( dev > ERR or dagdev>ERR or ofddev>ERR):
            assert False,"check_andihermicity"
    return retv

## 2020.11.28 revision : you do not need MD for this calculation
def calc_nacm_eff(this,mol_or_cell,Nacm_rep,fixedAtoms=None,SCHEME_2=True):
    # this : rttddft object
    dbgng=False; bfX_AOrep=None
    assert (Nacm_rep=='A' or Nacm_rep=='O'),""
    pbc = this._pbc
    kvectors=None
    if( pbc ):
        kvectors=np.reshape( this.kpts, (-1,3))
    ### calc_nacm_.check_scheme( (2 if(SCHEME_2) else 1) )
    nkpt=( 1 if(not pbc) else len(kvectors))
    Nat=len( mol_or_cell._atom)

    dirALL=[];Nat_eff=( Nat if(fixedAtoms is None) else Nat-len(fixedAtoms))
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        for J in range(3):
            dirALL.append(3*Iat+J)
    N_dir=len(dirALL)
    # d/dt ( Ssqrt C ) = S^{-1/2} [ H - i\hbar \dot{\bfR} [ \bfX - \sqrt{S}\nabla[\sqrt{S}] ] ]S^{-1/2} Ssqrt C
    assert ( RTTDDFT_.is_rttddft(this) ),"this"+str(type(this))
    # [N_dir][nAO][nAO]/[nkpt][N_dir][nAO][nAO]
    bfX_al=calc_nacm_AOrep(this,mol_or_cell,pbc,fixedAtoms=fixedAtoms,dtype=np.complex128)

    if(dbgng):
        bfX_AOrep=arrayclone(bfX_al)
        if(not pbc):
            print_z3tensor("calcNacm.log",False,bfX_AOrep,description="Nacm,AOrep")
        else:
            print_z3tensor("calcNacm.log",False,bfX_AOrep[0],description="Nacm,AOrep")
    ## we here subtract  Ssqrt nabla_Ssqrt
    ##                   S^{-1/2} this S^{-1/2} = nabla_Ssqrt S^{-1/2}
    if( Nacm_rep=='O'):
        bfY_al=(None if(not pbc) else [])
        nkp=(1 if(not pbc) else len(kvectors))
        for kp in range(nkp):
            bfX=(bfX_al if(not pbc) else bfX_al[kp])
            ndim=np.shape(bfX)
            bfY=np.zeros(np.shape(bfX),dtype=np.complex128)
            for jth_dir in range(N_dir):
                dir=dirALL[jth_dir]
                for mu in range( ndim[1] ):
                    for nu in range( ndim[2] ):
                        bfY[ jth_dir ][mu][nu] =np.conj( bfX[ jth_dir ][nu][mu] )
            Ssqrt=(this._Ssqrt if(not pbc) else this._Ssqrt[kp])
            nabla_Ssqrt = calc_nabla_Ssqrt(Ssqrt, bfX + bfY)  ## this expects [Ndir][nAO][nAO] tensors
            if(dbgng and kp==0):
                print_z3tensor("calcNacm.log",True,np.array([ np.matmul( Ssqrt[dir], nabla_Ssqrt[dir]) for dir in range(3*Nat_eff) ]),
                                description="S^{-1/2}\nabla S^{1/2}")
            assert isinstance(bfX[dir],np.ndarray),""
            for jth_dir in range(N_dir):
                dir=dirALL[ jth_dir]
                bfX[ jth_dir] -= np.matmul( Ssqrt, nabla_Ssqrt[ jth_dir])
            
            if(dbgng and kp==0):
                print_z3tensor("calcNacm.log",True, bfX, description="bfX_AOrep - S^{-1/2}\nabla S^{1/2}")

## ???     Ndim_X=np.shape(bfX_al) ##[nkpt][Ndir][nAO][nAO]
## ???     N_xyz=( Ndim_X[0] if(not pbc) else Ndim_X[1] )
## ???     assert N_xyz == len(dirALL),"%d / %d"%(N_xyz,dirALL)
## ???     # returns  [ nkpt ] [ Nat*3 ] [ nAO ] [ nAO ]
## ???     this.update_Sinv()
## ???     dSdR_wrk=None
## ???     
## ???     for kp in range(nkpt):
## ???         S1e=(this.get_ovlp() if(not pbc) else this.get_ovlp(this.cell,kvectors[kp]))
## ???         ### Ssqrt,Sinvrt = get_Ssqrt_invsqrt_1(S1e)
## ???         bfX =(bfX_al if(not pbc) else bfX_al[kp]) # [Nx3][nAO][nAO]
## ???         Ndim_bfX=np.shape(bfX)
## ???         if( dSdR_wrk is None ):
## ???             dSdR_wrk=np.zeros( Ndim_bfX,dtype=np.complex128 )
## ???         for I in range(Ndim_bfX[0]):
## ???             for J in range(Ndim_bfX[1]):
## ???                 for K in range(Ndim_bfX[2]):
## ???                     dSdR_wrk[I][J][K] = bfX[I][J][K] + np.conj( bfX[I][K][J] )
## ???         # _Ssqrt[nkpt][nAO][nAO]
## ???         nabla_Ssqrt = calc_nabla_Ssqrt( (this._Ssqrt if(not pbc) else this._Ssqrt[kp]), dSdR_wrk )
## ???         ### print("this._Ssqrt:",end="");print( np.shape(this._Ssqrt) )    #[nkpt][nAO][nAO]
## ???         ### print("nabla_Ssqrt:",end="");print( np.shape(nabla_Ssqrt[0]) ) #[nAO][nAO]
## ???         ith_dir=-1
## ???         for dir_o in dirALL: ### range(Ndim_bfX[0]):
## ???             ith_dir+=1
## ???             SxDS=np.matmul( (this._Ssqrt if(not pbc) else this._Ssqrt[kp]), nabla_Ssqrt[dir_o] )
## ???             ### print("SxDS:",end="");print(np.shape(SxDS))
## ???             if( not pbc ):
## ???                 for J in range(Ndim_bfX[1]):
## ???                     for K in range(Ndim_bfX[2]):
## ???                         bfX_al[ ith_dir ][J][K]-= SxDS[J][K]
## ???             else:
## ???                 for J in range(Ndim_bfX[1]):
## ???                     for K in range(Ndim_bfX[2]):
## ???                         bfX_al[kp][ ith_dir ][J][K]-= SxDS[J][K]
    return bfX_al

class calc_nacm_:
    N_calcgrad_nmdiff=0
    N_call=0
    ##scheme_=None
    scheme_=2;
    @staticmethod
    def check_scheme(scheme):
        if( calc_nacm_.scheme_ is None ):
            calc_nacm_.scheme_=scheme;
            log="scheme:%d"%(scheme);
            fd=futils.fopen("calcNacm.log","w"); print(log,file=fd);print(log);futils.fclose(fd);
        else:
            assert ( calc_nacm_.scheme_ == scheme ),"contradicting scheme";

## returns  [ nkpt ] [ Nat_eff*3 ] [ nAO ] [ nAO ]
## X= <\mu | \Nabla | \nu>
## Y= <(\Nabla \mu) | \nu> = X^{\dagger}
## dS/dR = X + X^{\dagger}
##
## 2020.11.28 revision : you do not need MD for this calculation
## 2021.12.03 renamed for clarity : calc_nacm -> calc_nacm_AOrep
def calc_nacm_AOrep(rttddft,mol_or_cell,pbc, fixedAtoms=None, dtype=None):
    # comp=3  3 component / hermi=2  anti-Hermitian / aosym,out IRRELEVANT for 1e-integ  
    assert (mol_or_cell is not None),"mol_or_cell"
    calc_nacm_.N_call +=1 
    if(dtype is None):
        dtype=(np.complex128 if(pbc) else np.float64)
    
    Nat=len(mol_or_cell._atom)
    elgrad_kptOR1=None
    if( not pbc ):
        elgrad_kptOR1 = rttddft.mol.intor('int1e_ipovlp', comp=3, hermi=2)
        # --- result is np.float64 --
        # assert (isinstance(elgrad_kptOR1[0][0][0],np.complex128)),"float??:"+str(type(elgrad_kptOR1[0][0][0]))
        # dtype1 = type( elgrad_kptOR1[0][0][0] )
        dtype1 = np.array( elgrad_kptOR1 ).dtype  ## this does not distinguish btw float and np.float64
        assert (dtype1==float or dtype1==np.float64),"non-pbc mole-intor dtype1"+str(dtype1)
    else:
        kvectors=np.reshape( rttddft.kpts, (-1,3))
        nkpt=len(kvectors)
        elgrad_kptOR1= rttddft.cell.pbc_intor('int1e_ipovlp',comp=3, hermi=2, kpts=kvectors)
        dtype1 = type( elgrad_kptOR1[0][0][0][0] )
        if( dtype1==complex or dtype1==np.complex128 ):
            dtype=np.complex128
        ## Note form is [ nkpt ] [ 3 ] [ nAO ] [ nAO ]
        ## assert (dtype1==complex or dtype1==np.complex128 ),"pbc cell-intor dtype1"+str(dtype1) in fact the result is REAL for k=0

    ndim_egrad = np.shape( elgrad_kptOR1 )  #[3][nAO][nAO] or [nkpt][3][nAO][nAO]
    bfX_al=None
    Nfixed=(0 if(fixedAtoms is None) else len(fixedAtoms)); Nat_eff=Nat-Nfixed
    if( not pbc ):
        assert (len(ndim_egrad)==3 and ndim_egrad[0]==3 and ndim_egrad[1]==ndim_egrad[2]),""
        nkpt = 1; nAO=ndim_egrad[1]
        bfX_al = np.zeros( [3*Nat_eff, nAO,nAO],dtype=dtype)
    else:
        assert (len(ndim_egrad)==4 and ndim_egrad[1]==3 and ndim_egrad[2]==ndim_egrad[3]),""
        nkpt = ndim_egrad[0]; nAO  = ndim_egrad[2]
        bfX_al = np.zeros( [nkpt,3*Nat_eff, nAO,nAO],dtype=dtype)
    
    Ia_effXjd=0
    mu_to_iat = get_mu_to_iat(rttddft, (rttddft.cell if(pbc) else rttddft.mol) )
    assert len(mu_to_iat)==nAO,"wrong dim:mu_to_iat"
    if( calc_nacm_.N_call == 1 ):
        print("mu_to_iat:",end="");print(mu_to_iat)

    sgn_int1e_ipovlp=Constants.sgn_int1e_ipovlp  ### 

    for kp in range(nkpt):
        elgrad=( elgrad_kptOR1 if(not pbc) else elgrad_kptOR1[kp] )
        Ia_eff=-1
        for Ia in range(Nat):
            if( fixedAtoms is not None ):
                if(Ia in fixedAtoms):
                    continue
            Ia_eff+=1
            for jd in range(3):
                Ia_effXjd=3*Ia_eff + jd
                    # < \mu | del_{Ixj} | \nu > = 0                  if( \nu does not belong to atom_I )
                    #                           = - <\mu|del_j|\nu>  if( \nu belong to atom_I )
                for mu in range(nAO):
                    for nu in range(nAO):
                        val = ( (- sgn_int1e_ipovlp*elgrad[jd][mu][nu]) if(mu_to_iat[nu]==Ia) else 0.0 )
                        if(not pbc):
                            bfX_al[ Ia_effXjd ][mu][nu] = val
                        else:
                            ### print( "bfX_al:%d,%d,%d,%d / %d (%d)"%( kp,Ia_effXjd,mu,nu,nkpt,len(bfX_al) ) )
                            bfX_al[kp][ Ia_effXjd ][mu][nu] = val
    return bfX_al;

def calc_EgsGRAD_nmdiff(md,rttddft,order=5,displ=0.02,fixedAtoms=None):
    grad,grad_ref=calc_EgsGRAD_nmdiff_(md,rttddft,order=order,displ=displ,fixedAtoms=fixedAtoms)
    dev_uplm=max( abs(np.array(grad)-np.array(grad_ref)) )
    return grad;

def calc_EgsGRAD_nmdiff_(md,rttddft,order=5,displ=0.02,fixedAtoms=None):
    hfo=order//2
    ## here we simply numdiff GS energy..
    Rnuc_o = arrayclone( rttddft.get_Rnuc('B') );Nat=len(Rnuc_o)
    wt0=time.time();wt1=wt0
    Ret=[];Ret_ref=(None if(order==3) else [])
    IJ=0; Iat_eff=-1
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        Iat_eff+=1
        for Jdir in range(3):
            Earr=[]
            for step in range(-hfo,hfo+1):
                if( step==0 ):
                    Earr.append(None);continue
                calc=md.update_Rnuc( IandJ=[Iat,Jdir],coordinate=Rnuc_o[Iat][Jdir]+ step*displ, update_moldyn=False)
                calc._fix_occ=False
                E_gs=calc.calc_gs()
                Earr.append(E_gs)
                wt2=wt1;wt1=time.time();
                if(IJ==0):
                    print("#calc_EgsGRAD_nmdiff:single step:%f"%(wt1-wt2))
            Dic={"7p":None,"5p":None,"3p":None}
            dNp = nmrdff( order, Earr, 0, displ,dict=Dic)

            Ret.append(dNp);
            if(order==7):
                Ret_ref.append(Dic["5p"])
            elif(order==5):
                Ret_ref.append(Dic["3p"])
    return Ret,Ret_ref

def calc_EgsGRAD_hfnm(md,rttddft,order=5,displ=0.02,fixedAtoms=None,ForceDecomposition=None, use_trFSXD=True):
    pbc=md.pbc
    GSdm=(None if(not pbc) else [])
    nkpt=(1 if(not pbc) else rttddft.nkpt)
    nAO =rttddft.nAO; nMO=rttddft.nMO
    iscomplexGSDM = ( False if(not pbc) else True )
    dtype_GSDM=(np.float64 if(not iscomplexGSDM) else np.complex128)
    for kp in range(nkpt):
        dm1=np.zeros([nAO,nAO],dtype=dtype_GSDM)
        canonicalMOs=( md._canonicalMOs if(not pbc) else md._canonicalMOs[kp] )
        occ =( rttddft._mo_occ if(not pbc) else rttddft._mo_occ[kp] )
        for i in range(nMO):
            for mu in range(nAO):
                for nu in range(nAO):
                    if( iscomplexGSDM ):
                        dm1[mu][nu] += canonicalMOs[mu][i]* occ[i] * np.conj( canonicalMOs[nu][i] )
                    else:
                        dm1[mu][nu] += canonicalMOs[mu][i]* occ[i] * canonicalMOs[nu][i]
        if( pbc ):
            GSdm.append(dm1)
        else:
            GSdm=dm1;break
    
    hfo=order//2
    ## here we simply numdiff GS energy..
    Rnuc_o = arrayclone( rttddft.get_Rnuc('B') );Nat=len(Rnuc_o)
    wt0=time.time();wt1=wt0
    Gr1=[];Gr1_ref=(None if(order==3) else [])
    IJ=0; Iat_eff=-1
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        Iat_eff+=1
        for Jdir in range(3):
            Earr=[]
            for step in range(-hfo,hfo+1):
                if( step==0 ):
                    Earr.append(None);continue
                calc=md.update_Rnuc( IandJ=[Iat,Jdir],coordinate=Rnuc_o[Iat][Jdir]+ step*displ, update_moldyn=False)
                E_nuc= calc.energy_nuc(); 
                E_el = ( calc.energy_elec( dm=GSdm )[0] if(not pbc) else \
                         calc.energy_elec( dm_kpts=GSdm ) [0] )
                E_tot= E_el + E_nuc
                Earr.append(E_tot)
            Dic={"7p":None,"5p":None,"3p":None}
            dNp = nmrdff( order,Earr, 0, displ, dict=Dic )
            Gr1.append( dNp );
            if(order==7):
                Gr1_ref.append(Dic["5p"])
            elif(order==5):
                Gr1_ref.append(Dic["3p"])
    if(ForceDecomposition is not None):
        ForceDecomposition.update({"F1":arrayclonex(Gr1,-1)})

    ## construct DM ...  GSdm
    if( use_trFSXD ):
        fock_kpts = md.calc_fock(rttddft,dm=GSdm,time_AU=md.time_AU)
        Gr2=calc_trFSXD( md.pbc, md.spinrestriction, rttddft,fock_kpts,GSdm, fixedAtoms=fixedAtoms, SCHEME_2=False)
    else:
        Gr2=calc_trCXCe( rttddft, md._canonicalMOs, md._canonicalEorbs, rttddft._mo_occ )

    if(ForceDecomposition is not None):
        ForceDecomposition.update({"F2":arrayclone(Gr2)})
    le=len(Gr1)
    for I in range(le):
        Gr1[I]=Gr1[I] - Gr2[I]
    return Gr1
#    Gr2=( None if(not pbc) else [] )
#    for kp in range(nkpt):
#        canonicalMOs=( md._canonicalMOs if(not pbc) else md._canonicalMOs[kp] )
#        eorbs=( md._canonicalEorbs if(not pbc) else md._canonicalEorbs[kp] )
#        occ =( rttddft._mo_occ if(not pbc) else rttddft._mo_occ[kp] )
#        Gr2=calc_trCXCe( rttddft, canonicalMOs, eorbs, occ )


def calc_Egnd(md,Rnuc=None,details=False):
    if(Rnuc is None):
        Rnuc=md.Rnuc
    calc=md.update_Rnuc( Rnew=Rnuc, update_moldyn=False)
    calc._fix_occ=False
    E_gs=calc.calc_gs()
    if(details):
        ### vhf=calc.get_veff()
        Eel,eCoulXC = calc.energy_elec() ### vhf=vhf)
        Enuc = calc.energy_nuc()
        ### ec=getattr(vhf, 'ecoul', None)
        return Eel+Enuc,Enuc,eCoulXC,Eel-eCoulXC
    else:
        return E_gs

def calcgrad_nmdiff(md,rttddft,dmat=None,order=3,displ=0.02,debug=False,SCHEME_2=None, 
                    fixedAtoms=None,grad_ref=None):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    dbgng_MPI=True ## XXX XXX
    calc_nacm_.N_calcgrad_nmdiff+=1
    compto_singlethread = (dbgng_MPI and MPIsize>1 and ( calc_nacm_.N_calcgrad_nmdiff==1 or \
                                           calc_nacm_.N_calcgrad_nmdiff==10 or \
                                           calc_nacm_.N_calcgrad_nmdiff==50 or \
                                           calc_nacm_.N_calcgrad_nmdiff%200==0 ))
    retv=None
    timing={};wt00=time.time();wt01=wt00;wt_MPI=-1;wt_SINGLETHREAD=-1
    if( MPIsize > 1 ):
        print("#calcForce:::calcgrad_nmdiff:$%02d:nmdiff_MPI start"%(MPIrank))
        retv=calcgrad_nmdiff_MPI( md,rttddft,dmat=dmat,order=order,displ=displ,debug=debug,
                                SCHEME_2=SCHEME_2, fixedAtoms=fixedAtoms, grad_ref=grad_ref, timing=timing)
        wt02=wt01;wt01=time.time();wt_MPI=wt01-wt02
        print("#calcForce:::calcgrad_nmdiff:$%02d:nmdiff_MPI end"%(MPIrank),wt01-wt02)
    if( MPIsize <=1 or compto_singlethread ):
        retv_MPI=retv
        print("#calcForce:::calcgrad_nmdiff:$%02d:XXX nmdiff_SINGLETHREAD start"%(MPIrank))
        retv=calcgrad_nmdiff_singlethread( md,rttddft,dmat=dmat,order=order,displ=displ,debug=debug,
                                SCHEME_2=SCHEME_2, fixedAtoms=fixedAtoms, grad_ref=grad_ref)
        wt02=wt01;wt01=time.time();wt_SINGLETHREAD=wt01-wt02
        print("#calcForce:::calcgrad_nmdiff:$%02d:XXX nmdiff_SINGLETHREAD end"%(MPIrank),wt01-wt02)
    if( compto_singlethread ):
        dev=max( abs( np.ravel(retv) - np.ravel(retv_MPI)))
        if(MPIrank==0):
            fdDBG=open("calcNacm_MPI_DBG.log","a")
            print("%3d:diff:%e elapsed:%f %f Rnuc:"%(calc_nacm_.N_calcgrad_nmdiff,dev, wt_MPI, wt_SINGLETHREAD)\
                   +str(md.Rnuc),file=fdDBG)
            if( dev > 1e-6 ):
                print("#MPI:"+str( [ "%14.6f"%(retv_MPI[k]) for k in range( len(retv_MPI) ) ] ),file=fdDBG)
                print("#REF:"+str( [ "%14.6f"%(retv[k]) for k in range( len(retv) ) ] ),file=fdDBG)
            fdDBG.close()
    return retv
def calcgrad_nmdiff_MPI(md,rttddft,dmat=None,order=3,displ=0.02,debug=False,SCHEME_2=None, 
                    fixedAtoms=None,grad_ref=None,timing=None):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()

    dbgng_Rnuc=True
    dbgng_MPI=True

    assert SCHEME_2 is not None,""
    dm_orth=( None if(not SCHEME_2) else convert_dm_AO_to_Orth( rttddft._Ssqrt, dmat, rttddft._pbc, rttddft._spinrestriction ) )
    dm_AOrep=( None if(SCHEME_2) else arrayclone(dmat) )

    if(timing is None): timing={}
    timing.update({"thread":MPIrank,"calc":0,"ntask":0,"sync":0,"nmrdiff":0})

    wt00=time.time();wt01=time.time() ## TIMING 
    hfo=order//2
    spdm=3; Nat=md.Nat;
    Nfixed=( 0 if(fixedAtoms is None) else len(fixedAtoms) )
    Nat_eff=Nat-Nfixed
    Iat_al=[]
    for I in range(Nat):
        if( fixedAtoms is not None):
            if(I in fixedAtoms):
                continue
        Iat_al.append(I)
    N_dir = Nat_eff*spdm
    hfo_x_2=hfo*2
    N_task = N_dir * (hfo_x_2)
    myTask = (N_task + MPIsize -1 )//MPIsize
    Ebuf_wrk=np.zeros([N_dir,2*hfo+1]) ## 
    task_offset = min( N_task, MPIrank * myTask)
    task_uplm   = min( N_task, (MPIrank+1)*myTask)

    if(dbgng_Rnuc):
        dum=d1maxdiff( np.ravel(md.Rnuc), np.ravel(rttddft.get_Rnuc('B')));assert dum<1e-6,"" 
    flags_dbg_wrk=None;
    indcs_dbg_wrk=None
    if( dbgng_MPI ):
        flags_dbg_wrk=np.zeros([N_dir,2*hfo+1],dtype=int)
        indcs_dbg_wrk=np.zeros([N_dir,2*hfo+1,4],dtype=int)
        
    MPIutils01.setup()
    MPIutils01.stop_parallel("calcgrad_nmdiff_MPI")

    print("#calcForce:::calcgrad_nmdiff_MPI:$%02d:N_task:%d mine:%d [%d:%d)"%(MPIrank,N_task,myTask,task_offset,task_uplm))

    Rnuc_o= arrayclone( md.Rnuc )
    R_wrk= arrayclone(md.Rnuc)
    for task in range(task_offset,task_uplm):
        step=task % (hfo_x_2); idispl=(step-hfo if(step<hfo) else step+1-hfo);
        dir =task //(hfo_x_2); kdir=dir%spdm; ith_at=dir//spdm;  iat=Iat_al[ith_at]
        calc=md.update_Rnuc( IandJ=[iat,kdir],coordinate=Rnuc_o[iat][kdir] + displ * idispl, update_moldyn=False)
        
        R_wrk[iat][kdir]=Rnuc_o[iat][kdir]+ displ*idispl
        E_nuc= calc.energy_nuc(Rnuc_au=R_wrk);
        if( SCHEME_2 ):
            calc.update_Sinv()
            dm_AOrep= convert_dm_Orth_to_AO( calc._Sinvrt, dm_orth, calc._pbc, calc._spinrestriction)
        h1e_new=calc.get_hcore()             ## this potentially depends on : calc._time_AU
        vhf_new=calc.get_veff(dm=dm_AOrep)   ## rttddft(MOL/PBC) takes "dm" as input
        E_el = ( calc.energy_elec( dm=dm_AOrep, h1e=h1e_new, vhf=vhf_new )[0] if(not md.pbc) else \
                 calc.energy_elec( dm_kpts=dm_AOrep, h1e_kpts=h1e_new, vhf=vhf_new ) [0] )
        E_tot= E_el + E_nuc
        Ebuf_wrk[ dir ][ idispl + hfo ]=E_tot
        R_wrk[iat][kdir]=Rnuc_o[iat][kdir]
        wt02=wt01;wt01=time.time(); timing["calc"]+=(wt01-wt02);timing["ntask"]+=1 ## TIMING 
        if(task==task_offset):
            print("#calcForce:::calcgrad_nmdiff_MPI:$%02d:singlepoint:%f"%(MPIrank,wt01-wt02))
        if( dbgng_MPI ):
            flags_dbg_wrk[ dir ][ idispl + hfo ]+=1
            indcs_dbg_wrk[ dir ][ idispl + hfo ][0]=MPIrank
            indcs_dbg_wrk[ dir ][ idispl + hfo ][1]=iat
            indcs_dbg_wrk[ dir ][ idispl + hfo ][2]=kdir
            indcs_dbg_wrk[ dir ][ idispl + hfo ][3]=idispl

    Ebuf=np.zeros([N_dir,2*hfo+1]) ## 
    comm.Allreduce(Ebuf_wrk, Ebuf)
    wt02=wt01;wt01=time.time(); timing["sync"]=wt01-wt02
    print("#calcForce:::calcgrad_nmdiff_MPI:$%02d:%d tasks total walltime:%f"%(MPIrank,myTask,wt01-wt00))
    if( dbgng_MPI ):
        flags_dbg=np.zeros([N_dir,2*hfo+1],dtype=int)
        indcs_dbg=np.zeros([N_dir,2*hfo+1,4],dtype=int)
        comm.Allreduce(flags_dbg_wrk, flags_dbg)
        comm.Allreduce(indcs_dbg_wrk, indcs_dbg)

    MPIutils01.restart_parallel("calcgrad_nmdiff_MPI")
    
    grad=np.zeros( [N_dir],dtype=np.float64 )
    if( grad_ref is not None ):
        if( order>3 ):
            grad_ref.update({"3p":np.zeros([Nat_eff,3],dtype=np.float64), \
                             "5p":np.zeros([Nat_eff,3],dtype=np.float64)})
        if(order>5):
            grad_ref.update({"7p":np.zeros([Nat_eff,3],dtype=np.float64)})
    IxJ=-1;Iat_eff=-1
    for Iat in range(Nat):
        if( fixedAtoms is not None):
            if(Iat in fixedAtoms):
                continue
        Iat_eff+=1
        for Jdir in range(spdm):
            IxJ+=1
            Dic={"7p":None,"5p":None,"3p":None};
            dNp = nmrdff( order,Ebuf[IxJ][:],dim=0,dx=displ,dict=Dic)
            grad[ IxJ ]=dNp;
            if( dbgng_MPI ):
                for k in range(-hfo,hfo+1):
                    assert (k==0 and flags_dbg[IxJ][k+hfo]==0) or \
                           (k!=0 and flags_dbg[IxJ][k+hfo]==1), "flag@%d:"%(IxJ)+str(flags_dbg[IxJ])
                    if(k!=0):
                        assert indcs_dbg[IxJ][k+hfo][1]==Iat and  \
                               indcs_dbg[IxJ][k+hfo][2]==Jdir and \
                               indcs_dbg[IxJ][k+hfo][3]==k, "indcs@%d,%d:"%(IxJ,k+hfo)+str(indcs_dbg[IxJ][k+hfo])\
                                                                           +"/"+str([Iat,Jdir,k])
            if( grad_ref is not None ):
                if( order>3 ):
                    grad_ref["3p"][Iat_eff][Jdir]=Dic["3p"];
                    grad_ref["5p"][Iat_eff][Jdir]=Dic["5p"];
                if( order>5 ):
                    grad_ref["7p"][Iat_eff][Jdir]=Dic["7p"];
        
    wt02=wt01;wt01=time.time(); timing["nmrdiff"]=wt01-wt02

    logfpath="calcForce_walltime_%02d.log"%(MPIrank)
    fdOU=open(logfpath,"a")
    print("#calcForce:::calcgrad_nmdiff_MPI:$%02d:%14.4f  nTask:%d "%( MPIrank,wt01-wt00, -task_offset+task_uplm)+str(timing)
          +" \t\t"+str(datetime.datetime.now()), file=fdOU)
    fdOU.close()

    print("#calcForce:::calcgrad_nmdiff_MPI:$%02d:END %f"%(MPIrank,wt01-wt00))
    
    return grad

## calcNacm2.py : def calcGrad_analytic(md,rttddft,dmat=None,fixedAtoms=None):
## calcNacm2.py :     from pyscf.pbc.grad.krks import Gradients
## calcNacm2.py :     if dmat is None: dmat = md.calc_tdDM(rttddft)
## calcNacm2.py :     pbc=md.pbc
## calcNacm2.py :     if( not pbc ):
## calcNacm2.py :         assert False,""
## calcNacm2.py :     grad=Gradients( rttddft )
## calcNacm2.py : 
## calcNacm2.py :     mol_or_cell = (rttddft.cell if(pbc) else rttddft.mol)
## calcNacm2.py :     if( pbc ):
## calcNacm2.py :         hcGEN=grad.hcore_generator(mol_or_cell, rttddft.kpts)
## calcNacm2.py :     else:
## calcNacm2.py :         hcGEN=grad.hcore_generator(mol_or_cell)
## calcNacm2.py : 
## calcNacm2.py :     veffgrad=grad.get_veff(dm=dmat)
## calcNacm2.py :     hcgrad =grad.get_hcore()
## calcNacm2.py :     print("veff:",np.shape(veffgrad)) ##(3, 4, 26, 26)
## calcNacm2.py :     print("hc:",np.shape(hcgrad))     ##(4, 3, 26, 26)
## calcNacm2.py :     spdm=3;nkpt=md.nkpt
## calcNacm2.py :     Nat=md.Nat
## calcNacm2.py :     Nat_eff=(Nat if(fixedAtoms is None) else Nat-len(fixedAtoms))
## calcNacm2.py :     Ndir_eff=spdm*Nat_eff
## calcNacm2.py :     ret=np.zeros([Ndir_eff],dtype=np.float64)
## calcNacm2.py :     nAO=md.nAO;assert nAO>0,""
## calcNacm2.py :     vWRK=np.zeros( ([nkpt,nAO,nAO] if(pbc) else [nAO,nAO]),dtype=np.complex128 )
## calcNacm2.py :     hWRK=np.zeros( ([nkpt,nAO,nAO] if(pbc) else [nAO,nAO]),dtype=np.complex128 )
## calcNacm2.py :     mu_to_iat=get_mu_to_iat(rttddft, (rttddft.cell if(pbc) else rttddft.mol) )
## calcNacm2.py :     IxJ=-1;Iat_eff=-1
## calcNacm2.py :     for Iat in range(Nat):
## calcNacm2.py :         if( fixedAtoms is not None ):
## calcNacm2.py :             if( Iat in fixedAtoms ):
## calcNacm2.py :                 continue
## calcNacm2.py :         hcgrads_Iat=hcGEN(Iat)
## calcNacm2.py :         print("hcgrad:",np.shape(hcgrad))
## calcNacm2.py :         Iat_eff+=1
## calcNacm2.py :         for Jdir in range(spdm):
## calcNacm2.py :             IxJ+=1;
## calcNacm2.py :             make_vAO(vWRK,pbc,Iat,Jdir,veffgrad, 0, n_kpoints=nkpt,mu_to_iat=mu_to_iat)
## calcNacm2.py : 
## calcNacm2.py :             make_vAO(hWRK,pbc,Iat,Jdir,hcgrad, 1, n_kpoints=nkpt,mu_to_iat=mu_to_iat)
## calcNacm2.py : 
## calcNacm2.py :             print_matrices(hWRK,hcgrads_Iat[Jdir],"hWRK,hcgrad",fpath="hcgrad.dat",Append=(Iat!=0 or dir!=0))
## calcNacm2.py :             ret[IxJ]=dmtrace( pbc,vWRK+hWRK, dmat)
## calcNacm2.py :     atmlst=[]
## calcNacm2.py :     for Iat in range(Nat):
## calcNacm2.py :         if( fixedAtoms is not None ):
## calcNacm2.py :             if( Iat in fixedAtoms ):
## calcNacm2.py :                 continue
## calcNacm2.py :         atmlst.append(Iat)
## calcNacm2.py :     nucGrad=grad.grad_nuc( mol_or_cell, atmlst)
## calcNacm2.py :     print("nucGrad:",np.shape(nucGrad),"ret:",np.shape(ret))
## calcNacm2.py :     nucGrad=np.reshape(nucGrad,np.shape(ret))
## calcNacm2.py :     ret=ret+nucGrad
## calcNacm2.py :     return ret

def i1eqb(lhs,rhs):
    le=len(lhs);
    if(len(rhs)!=le):
        return False
    for j in range(le):
        if( lhs[j]!=rhs[j] ):
            return False
    return True

def d1toa(buf,maxlen=10):
    buf=np.ravel(buf)
    le=len(buf);le1=min(le,maxlen)
    ret=""
    for j in range(le1):
        ret+="%14.6f "%(buf[j])
    if(le>le1):
        ret+=" ... [%d]:%14.6f"%(le-1,buf[le-1])
    return ret
def d2toa(buf,buf2=None,maxlen=8):
    ndim=np.shape(buf)
    le=len(buf);le1=min(le,maxlen)
    le2=min(ndim[1],maxlen)
    Ret=""
    for I in range(le1):
        ret=""
        for J in range(le2):
            ret+="%14.6f "%(buf[I][J])
        if( ndim[1]>le2 ):
            ret+=" ... [%d]:%14.6f"%( ndim[1]-1, buf[I][ ndim[1]-1 ])

        if( buf2 is not None):
            ret+=" \t\t\t "
            for J in range(le2):
                ret+="%14.6f "%(buf2[I][J])
            if( ndim[1]>le2 ):
                ret+=" ... [%d]:%14.6f"%( ndim[1]-1, buf2[I][ ndim[1]-1 ])

        Ret+=("\n" if(I>0) else "")+ret
    return Ret

def d3toa(lhs,rhs=None):
    NdL=np.shape(lhs);
    if(rhs is not None):
        NdR=np.shape(rhs)
    ret=""
    for kp in range(NdL[0]):
        if(kp>0): ret+="\n\n";
        ret="#%02d\n"%(kp)
        for ip in range(NdL[1]):
            ret="%05d: "
            for jp in range(NdL[2]):
                ret+="%12.6f "%(lhs[kp][ip][jp])
            if(rhs is not None):
                ret+=" \t\t\t "
                for jp in range(NdL[2]):
                    ret+="%12.6f "%(rhs[kp][ip][jp])
            ret+="\n"
    return ret

def print_matrices(lhs,rhs,title,fpath=None,fopen=False,Append=True):
    import sys
    import os
    NdL=np.shape(lhs);NdR=np.shape(rhs)
    fd=sys.stdout;doclose=False
    if( fpath is not None):
        fd=open(fpath,("a" if(Append) else "w"));doclose=True

    errbuf=[]
    dev=aNmaxdiff(lhs,rhs,errbuf)
    print("#print_matrices:%s:SHAPE:%s,%s \t\t\t "%( title, str(NdL),str(NdR)) \
            + str(datetime.datetime.now()),file=fd)
    if( not i1eqb(NdL,NdR) ):
        print("#print_matrices:%s:!W len differs:%s / %s"%(title,str(NdL),str(NdR)),file=fd)
        print("#print_matrices:%s:LHS:"+d1toa(lhs),file=fd )
        print("#print_matrices:%s:RHS:"+d1toa(rhs),file=fd );
        if(doclose):fd.close();
        if(fopen):os.system("fopen "+fpath);
        return
    rank=len(NdL)
    if(rank==1):
        print("#print_matrices:%s:LHS:"+d1toa(lhs),file=fd )
        print("#print_matrices:%s:RHS:"+d1toa(rhs),file=fd);
        if(doclose): fd.close()
        if(fopen):os.system("fopen "+fpath);
        return
    elif(rank==2):
        print("#print_matrices:%s\n"+d2toa(lhs,rhs),file=fd);
        if(doclose): fd.close()
        if(fopen):os.system("fopen "+fpath);
        return
    else:
        print("#print_matrices:%s:LHS:"+d3toa(lhs),file=fd )
        print("#print_matrices:%s:RHS:"+d3toa(rhs),file=fd );
        if(doclose): fd.close()
        if(fopen):os.system("fopen "+fpath);
        return
        

def calcGrad_analytic_FFTDF(md,rttddft,dmat=None,FockMat=None,fixedAtoms=None, decomposition=None, 
                            nmgrads=None, SCHEME_2=False, time_AU=None):
    from .calcNacm2 import get_derivMatrices_FFTDF
    if dmat is None: dmat = md.calc_tdDM(rttddft)
    if time_AU is None: time_AU = md.time_AU
    if(FockMat is None):
        FockMat = md.calc_fock(rttddft,dm=dmat,time_AU=time_AU)
    DerivMatrices= get_derivMatrices_FFTDF(md,rttddft, md.Nat, dmat=dmat,fixedAtoms=fixedAtoms)
    pbc=md.pbc
    spdm=3;nkpt=md.nkpt
    Nat=md.Nat
    Nat_eff=(Nat if(fixedAtoms is None) else Nat-len(fixedAtoms))
    Ndir_eff=spdm*Nat_eff
    ret=np.zeros([Ndir_eff],dtype=np.float64)
    Refr=( None if( nmgrads is None ) else np.zeros([Ndir_eff],dtype=np.float64))

    de_hc=[];de_ve=[]
    nAO=md.nAO;assert nAO>0,""
    IxJ=-1;Iat_eff=-1
    print("hcore:", np.shape( DerivMatrices["hcore"] ),np.shape( DerivMatrices["veff"] ) )
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        for Jdir in range(spdm):
            IxJ+=1
            hWRK=DerivMatrices["hcore"][ IxJ ]
            vWRK=DerivMatrices["veff"][ IxJ ]
            print(np.shape(DerivMatrices["hcore"]),np.shape(DerivMatrices["veff"]), np.shape(hWRK),np.shape(vWRK))
            ret[ IxJ ] = dmtrace( pbc,vWRK+hWRK, dmat)

            if(decomposition is not None ):
                de_hc.append( dmtrace( pbc,hWRK,dmat) )
                de_ve.append( dmtrace( pbc,vWRK,dmat) )

            if( nmgrads is not None ):
                hWRK=nmgrads["hcore"][IxJ]
                vWRK=nmgrads["veff"][IxJ]
                print("dmtrace:",np.shape(hWRK),np.shape(vWRK))
                Refr[ IxJ ] = dmtrace( pbc,vWRK+hWRK, dmat)
    # gr2 [nkpt,3*Nat,nAO,nAO]
    # see L480 : gr2(:=calc_trFSXD) is to be SUBTRACTED from gr1 
    # to keep consistency with 480, we use gr2, but de_s1 :=-gr2
    gr2= calc_trFSXD(md.pbc, md.spinrestriction, rttddft, FockMat, dmat, fixedAtoms=fixedAtoms,
                    SCHEME_2=SCHEME_2)  ## should we store those matrices ?? 
    gr_nc = DerivMatrices["E_nuc"]
    
    if(decomposition is not None):
        decomposition.update({"de_hc":np.ravel(de_hc), "de_ve":np.ravel(de_ve), "de_s1":-np.ravel(gr2), "de_nc":np.ravel(gr_nc)})

    dbgng=True
    if( dbgng ):
        ## returns [nKpt][Nat*3]
        gr2b= calc_trCXCe(rttddft, md._canonicalMOs, md._canonicalEorbs, md.mo_occ,fixedAtoms=fixedAtoms)
        print("#calc_trCXCe:",np.shape(gr2),np.shape(gr2b))
        #print("#trCXCe:diff:",max( abs( np.ravel( gr2 )-np.ravel( gr2b ) ) ))

    print("#calcGrad_analytic_FFTDF:",np.shape(ret),np.shape(gr2),np.shape(gr_nc))
    #
    # see L480 : gr2(:=calc_trFSXD) is to be SUBTRACTED from gr1 
    ret= np.ravel(ret) - np.ravel(gr2) + np.ravel(gr_nc)  ## 2022.01.16: we included gr_nc 
    if(Refr is not None ):
        Refr=Refr - gr2
        nmgrads.update({"Egrad_refr":arrayclone(Refr)})
    if(nmgrads is None ):
        return ret
    else:
        return ret,Refr
