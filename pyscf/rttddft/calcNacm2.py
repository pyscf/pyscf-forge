import numpy as np
import math
import os
import time
import datetime
from .Loglv import printout
from .utils import i1eqb,parse_xyzstring,aNmaxdiff
from .Moldyn import get_eOrbs
from mpi4py import MPI
from .heapcheck import heapcheck
from .rttddft_common import rttddft_common
import numpy as np
from .utils import arrayclone

def fn_save_matrices_std(key,index,dbuf,buf_matrices,atmlst):
    printout=False
    Nat=len(atmlst);spdm=3
    Ndim=np.shape(dbuf)
    print("#fn_save_matrices_std:",key,index,Ndim)
    is_scalar = ( isinstance(dbuf,float) or isinstance(dbuf,np.float64) or\
                  isinstance(dbuf,int) or isinstance(dbuf,np.int64) or \
                  isinstance(dbuf,complex) or isinstance(dbuf,np.complex128) )
    if( not printout ):
        if( index < 0 ):
            buf_matrices.update({key:(dbuf if(is_scalar) else arrayclone(dbuf,label="fn_save_matrices:"+str(key)))})
        else:
            # input h1ao : [ 3 ][ nKpoints ][ nAO ][ nAO ]
            if( key == "hcore"):
                assert Ndim[0]==spdm,""
            if( key not in buf_matrices ):
                buf_matrices.update({key:[ None for iat in range(Nat) ]})
            buf_matrices[key][ index ]=(dbuf if(is_scalar) else arrayclone(dbuf,label="fn_save_matrices:%02d:"%(index)+str(key)))
## 20220508.divide_tasks:                buf_matrices.update({key:[]})
## 20220508.divide_tasks:            if( len(buf_matrices[key])==index ):
## 20220508.divide_tasks:                buf_matrices[key].append( (dbuf if(is_scalar) else arrayclone(dbuf)) )
## 20220508.divide_tasks:            else:
## 20220508.divide_tasks:                assert len(buf_matrices[key])==index,""
## 20220508.divide_tasks:                buf_matrices[key][index] = (dbuf if(is_scalar) else arrayclone(dbuf))
    else:
        assert False,""



##   D
def convert_dm_AO_to_Orth( SsqrtAL, dm_AO, pbc ,spinrestriction ):
    return convert_dmat_( SsqrtAL, dm_AO, pbc ,spinrestriction )

def convert_dm_Orth_to_AO( SinvrtAL, dm_AO, pbc ,spinrestriction ):
    return convert_dmat_( SinvrtAL, dm_AO, pbc, spinrestriction )

def convert_dmat_( Ssqrt_or_invrtAL, dm_AO_or_Orth, pbc, spinrestriction ):
    assert spinrestriction == 'R',"UKS should call this method twice"
    Ret=(None if(not pbc) else [])
    nkpt=(1 if(not pbc) else len( Ssqrt_or_invrtAL))
    for kp in range(nkpt):
        Ssqrt = ( Ssqrt_or_invrtAL if(not pbc) else Ssqrt_or_invrtAL[kp] )
        dAO   = ( dm_AO_or_Orth if(not pbc) else dm_AO_or_Orth[kp] )
        mat =np.matmul( Ssqrt, np.matmul(dAO, Ssqrt) )
        if( not pbc ):
            Ret=mat;break
        else:
            Ret.append(mat)
    return Ret

## v(3 ,Nkpt, nAO-d, nAO)  or  (3,nAO-d,nAO)  xyz_at==0
## h(Nkpt, 3, nAO-d, nAO)  of  (3,nAO-d,nAO)  xyz_at==1
def make_vAO(rttddft, ret,pbc,Iat,Jdir,gradMat,xyz_at, n_kpoints=None,mu_to_iat=None):
    if( mu_to_iat is None ): mu_to_iat = get_mu_to_iat(rttddft, (rttddft.cell if(pbc) else rttddft.mol));
    assert xyz_at==0 or xyz_at==1,""
    Ndim=np.shape(gradMat)
    nkpt=1;nAO=Ndim[1]
    if( pbc ):
        k_at=1-xyz_at;nkpt=Ndim[k_at];nAO=Ndim[2]
    if( n_kpoints is not None):
        assert n_kpoints==nkpt,""
    if(pbc):
        for kp in range(nkpt):
            for mu in range(nAO):
                for nu in range(nAO):
                    ret[kp][mu][nu]=0.0
                    if(mu_to_iat[mu]==Iat):
                        ret[kp][mu][nu]+=( gradMat[ Jdir ][ kp ][ mu ][ nu ] if(xyz_at==0) else \
                                           gradMat[ kp ][ Jdir ][ mu ][ nu ] )
                    if(mu_to_iat[nu]==Iat):
                        ret[kp][mu][nu]+=np.conj( gradMat[ Jdir ][ kp ][ nu ][ mu ] if(xyz_at==0) else \
                                                  gradMat[ kp ][ Jdir ][ nu ][ mu ] )

    else:
        for mu in range(nAO):
            for nu in range(nAO):
                ret[mu][nu]=0.0
                if(mu_to_iat[mu]==Iat):
                    ret[mu][nu]+=gradMat[ Jdir ][ mu ][ nu ]
                if(mu_to_iat[nu]==Iat):
                    ret[mu][nu]+=np.conj( gradMat[ Jdir ][ nu ][ mu ] )
    return ret



def get_elmt(array,index):
    Ndim=np.shape(array);rank=len(Ndim)
    le=len(index);assert le<=rank,""
    if( le == 1 ):
        return array[index[0]]
    elif( le == 2 ):
        return array[ index[0] ][ index[1] ]
    elif( le == 3 ):
        return array[ index[0] ][ index[1] ][ index[2] ]
    elif( le == 4 ):
        return array[ index[0] ][ index[1] ][ index[2] ][ index[3] ]
    elif( le == 5 ):
        return array[ index[0] ][ index[1] ][ index[2] ][ index[3] ][ index[4] ]
    else:
        assert False,""
#
# sqrdiff,diff,at,vals = aNsqrMaxdiffAt
def aNsqrMaxdiffAt(lhs,rhs):
    Lh=lhs;Rh=rhs
    Ndim=np.shape(lhs);rank=len(Ndim)
    if( rank > 5 ):
        Lh=np.ravel(lhs);Rh=np.ravel(rhs)
        Ndim=np.shape(Lh);assert len(Ndim)<=5,""
    Ndim_RHS=np.shape(Rh)
    if( not i1eqb( Ndim, Ndim_RHS ) ):
        print("#!W dimension differs:",Ndim,Ndim_RHS)
        Rh=np.reshape(lhs,Ndim)

    diff=-1;at=None;vals=None
    for I in range(Ndim[0]):
        if(rank==1):
            print("aNsqrMaxDiffat:",Lh[I],Rh[I])
            if( (Lh[I] is None) or (Rh[I] is None)):
                continue
            dum=abs( Lh[I]-Rh[I] )
            if( dum > diff ):
                diff=dum;at=[I];vals=[Lh[I],Rh[I]]
            continue
        for J in range(Ndim[1]):
            if(rank==2):
                dum=abs( Lh[I][J]-Rh[I][J] )
                if( dum > diff ):
                    diff=dum;at=[I,J];vals=[Lh[I][J],Rh[I][J]]
                continue
            for K in range(Ndim[2]):
                if(rank==3):
                    dum=abs( Lh[I][J][K]-Rh[I][J][K] )
                    if( dum > diff ):
                        diff=dum;at=[I,J,K];vals=[Lh[I][J][K], Rh[I][J][K]]
                    continue
                for L in range(Ndim[3]):
                    if(rank==4):
                        dum=abs( Lh[I][J][K][L]-Rh[I][J][K][L] )
                        if( dum > diff ):
                            diff=dum;at=[I,J,K,L];vals=[Lh[I][J][K][L],Rh[I][J][K][L]]
                        continue
                        for M in range(Ndim[4]):
                            if(rank==5):
                                dum=abs( Lh[I][J][K][L][M]-Rh[I][J][K][L][M] )
                                if( dum > diff ):
                                    diff=dum;at=[I,J,K,L,M];vals=[Lh[I][J][K][L][M],Rh[I][J][K][L][M]]
                                continue
    Lh1D=np.ravel(Lh);Rh1D=np.ravel(Rh)
    sqrdiff= np.vdot( Lh1D-Rh1D, Lh1D-Rh1D)

    return sqrdiff,diff,at,vals
##
## 
def numrdiff_matrices( md, rttddft, dmat=None, fixedAtoms=None, order=5, displ=0.02, multithreading=True, Targets=None, Suppl=None,Deviations=None, 
                       SCHEME_2=False ):
    if( dmat is None ): dmat = md.calc_tdDM(rttddft,ovrd=True)
    spdm=3; Nat=md.Nat;
    Nfixed=( 0 if(fixedAtoms is None) else len(fixedAtoms) )
    Nat_eff=Nat-Nfixed
    N_dir = Nat_eff*spdm
    
    if( Targets is None ):
        Targets={"veff":[ None for k in range(N_dir) ], "hcore":[ None for k in range(N_dir) ],
                 "E_el":[ None for k in range(N_dir) ], "E_nuc":[ None for k in range(N_dir) ]}
    if( Suppl is None ):
         Suppl={"veff":[ None for k in range(N_dir) ], "hcore":[ None for k in range(N_dir) ],
                "E_el":[ None for k in range(N_dir) ], "E_nuc":[ None for k in range(N_dir) ]}

    assert SCHEME_2 is not None,""
    dm_orth=( None if(not SCHEME_2) else convert_dm_AO_to_Orth( rttddft._Ssqrt, dmat, rttddft._pbc, rttddft._spinrestriction ) )
    dm_AOrep=( None if(SCHEME_2) else arrayclone(dmat) )

    hfo=order//2
    Iat_al=[]
    for I in range(Nat):
        if( fixedAtoms is not None):
            if(I in fixedAtoms):
                continue
        Iat_al.append(I)
    hfo_x_2=hfo*2
    N_task = N_dir * (hfo_x_2)
    ##                 0         1         2     3
    ##                -3        -2        -1
    Coeffs_3p=[      None,     None,    -0.50, None,    0.50,      None,     None ]
    #                 -3        -2        -1     0
    Coeffs_5p=[      None, 1.0/12.0, -2.0/3.0, None, 2.0/3.0, -1.0/12.0,     None ]
    Coeffs_7p=[ -1.0/60.0,      0.15,    -0.75,None,    0.75,     -0.15, 1.0/60.0 ] 
    Coeffs=    ( Coeffs_3p if(order==3) else ( Coeffs_5p if(order==5) else Coeffs_7p))
    Coeffs_sub=( Coeffs_3p if(order==3) else ( Coeffs_3p if(order==5) else Coeffs_5p))

    MPIsize=1; MPIrank=0
    if( multithreading ):
        comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()

    myTask = (N_task + MPIsize -1 )//MPIsize
    task_offset = min( N_task, MPIrank * myTask)
    task_uplm   = min( N_task, (MPIrank+1)*myTask)

    Rnuc_o= arrayclone( md.Rnuc )
    R_wrk = arrayclone( md.Rnuc )
    
    print("calcNacm2:R_org:",Rnuc_o)
    for task in range(task_offset,task_uplm):
        step=task % (hfo_x_2); idispl=(step-hfo if(step<hfo) else step+1-hfo);
        dir =task //(hfo_x_2); kdir=dir%spdm; ith_at=dir//spdm;  iat=Iat_al[ith_at]
        R_wrk[iat][kdir]=Rnuc_o[iat][kdir]+ displ*idispl
        print("calcNacm2:R_displ[%d][%d]:%f"%(iat,kdir,R_wrk[iat][kdir]),R_wrk)
        calc=md.update_Rnuc( IandJ=[iat,kdir],coordinate=Rnuc_o[iat][kdir] + displ * idispl, update_moldyn=False, 
                             R_dbg=R_wrk)
        
        matr={}
        if( SCHEME_2 ):
            calc.update_Sinv()
            dm_AOrep= convert_dm_Orth_to_AO( calc._Sinvrt, dm_orth, calc._pbc, calc._spinrestriction)
        E_nuc = calc.energy_nuc(Rnuc_au=R_wrk);
        hcore = calc.get_hcore()             ## this potentially depends on : calc._time_AU
        veff  = calc.get_veff(dm=dm_AOrep)   ## rttddft(MOL/PBC) takes "dm" as input
        E_el = ( calc.energy_elec( dm=dm_AOrep, h1e=hcore, vhf=veff )[0] if(not md.pbc) else \
                 calc.energy_elec( dm_kpts=dm_AOrep, h1e_kpts=hcore, vhf=veff ) [0] )

        coeff= Coeffs[ 3 + idispl ];assert coeff is not None
        coeff_SUB= Coeffs_sub[ 3 + idispl ];
        if( "veff" in Targets ):
            if( Targets["veff"][dir] is None ):
                Targets["veff"][dir] = arrayclone( veff * coeff )
            else:
                Targets["veff"][dir] += coeff*veff
        if( "hcore" in Targets ):
            if( Targets["hcore"][dir] is None ):
                Targets["hcore"][dir] = arrayclone( hcore * coeff )
            else:
                Targets["hcore"][dir] += coeff*hcore

        if( "E_el" in Targets ):
            if( Targets["E_el"][dir] is None ):
                Targets["E_el"][dir] = E_el * coeff
            else:
                Targets["E_el"][dir] += coeff*E_el
        if( "E_nuc" in Targets ):
            if( Targets["E_nuc"][dir] is None ):
                Targets["E_nuc"][dir] = E_nuc * coeff
            else:
                Targets["E_nuc"][dir] += coeff*E_nuc

        if( coeff_SUB is not None ):
            if( Suppl["veff"][dir] is None ):
                Suppl["veff"][dir] = arrayclone( veff * coeff_SUB )
            else:
                Suppl["veff"][dir] += coeff_SUB*veff

            if( Suppl["hcore"][dir] is None ):
                Suppl["hcore"][dir] = arrayclone( hcore * coeff_SUB )
            else:
                Suppl["hcore"][dir] += coeff_SUB*hcore

            if( Suppl["E_el"][dir] is None ):
                Suppl["E_el"][dir] = E_el * coeff_SUB
            else:
                Suppl["E_el"][dir] += coeff_SUB*E_el

            if( Suppl["E_nuc"][dir] is None ):
                Suppl["E_nuc"][dir] = E_nuc * coeff_SUB
            else:
                Suppl["E_nuc"][dir] += coeff_SUB*E_nuc
        R_wrk[iat][kdir]=Rnuc_o[iat][kdir]
    if( Deviations is not None ):
        for key in Targets:
            Deviations.update({key:[ -1 for k in range(N_dir) ]})

    for key in Targets:
        for dir in range(N_dir):
            if( multithreading ):
                wks=np.zeros( np.shape(Targets[key][dir]), Targets[key][dir].dtype )
                comm.Allreduce( Targets[key][dir],wks )
                Targets[key][dir] = wks / displ

                comm.Allreduce( Suppl[key][dir],wks )
                Suppl[key][dir] = wks / displ
                
            else:
                Targets[key][dir] = Targets[key][dir] / displ
                Suppl[key][dir] = Suppl[key][dir] / displ

            if( isinstance(Targets[key][dir],float) or isinstance(Targets[key][dir],complex) or\
                isinstance( Targets[key][dir],np.float64) or isinstance(Targets[key][dir],np.complex128) ):
                diff=abs(Targets[key][dir]-Suppl[key][dir]); dev=diff 
                at=[0];vals=[ Targets[key][dir], Suppl[key][dir]]
            else:
                diff,dev,at,vals =aNsqrMaxdiffAt( Targets[key][dir], Suppl[key][dir] )
            
            if( Deviations is not None ):
                Deviations[key][dir]=dev
            print("#nmrdiff:%s_%04d %f %e %s %s"%( key,dir,np.sqrt(diff.real),
                                                   dev,str(vals[0]),str(vals[1])) )
    return Targets

def calcGrad_analytic_FFTDF1(md,rttddft,dmat=None,fixedAtoms=None):
    from pyscf.pbc.grad.krks import Gradients
    if dmat is None: dmat = md.calc_tdDM(rttddft)

    DerivMatrices={}
    print("#>calcGrad_analytic_FFTDF:start calcgrad_analytic_ref")
    dE_ref=calcgrad_analytic_ref(rttddft, DerivMatrices=DerivMatrices)
    print("#>calcGrad_analytic_FFTDF:end calcgrad_analytic_ref:",[ dum for dum in DerivMatrices ])

    pbc=md.pbc
    if( not pbc ):
        assert False,""
    grad=Gradients( rttddft )

    mol_or_cell = (rttddft.cell if(pbc) else rttddft.mol)
    if( pbc ):
        hcGEN=grad.hcore_generator(mol_or_cell, rttddft.kpts)
    else:
        hcGEN=grad.hcore_generator(mol_or_cell)

    veffgrad=grad.get_veff(dm=dmat)
    hcgrad =grad.get_hcore()

    nmrdiff_deviations={}
    print("#>calcGrad_analytic_FFTDF:start numrdiff")
    nmrgrads=numrdiff_matrices( md, rttddft, dmat=dmat, order=5, Deviations=nmrdiff_deviations )
    print("#>calcGrad_analytic_FFTDF:end numrdiff")

    for ky in DerivMatrices:
        if( ky in nmrgrads ):
            lh=DerivMatrices[ky]
            rh=nmrgrads[ky]
            print("#>calcGrad_analytic_FFTDF:ref_vs_nmdiff:"+ky,aNsqrMaxdiffAt(lh,rh))
            if(ky =="veff"):
                printMatrices("veffGrad.dat",lh,rh,description="v_eff:Analytic vs Numeric")
# 
# 
# 
    # assert False,"check Results"

    print("veff:",np.shape(veffgrad)) ##(3, 4, 26, 26)
    print("hc:",np.shape(hcgrad))     ##(4, 3, 26, 26)
    spdm=3;nkpt=md.nkpt
    Nat=md.Nat
    Nat_eff=(Nat if(fixedAtoms is None) else Nat-len(fixedAtoms))
    Ndir_eff=spdm*Nat_eff
    ret=np.zeros([Ndir_eff],dtype=np.float64)
    nAO=md.nAO;assert nAO>0,""
    vWRK=np.zeros( ([nkpt,nAO,nAO] if(pbc) else [nAO,nAO]),dtype=np.complex128 )
    hWRK=np.zeros( ([nkpt,nAO,nAO] if(pbc) else [nAO,nAO]),dtype=np.complex128 )
    mu_to_iat=get_mu_to_iat(rttddft, (rttddft.cell if(pbc) else rttddft.mol) )
    IxJ=-1;Iat_eff=-1
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        hcgrads_Iat=hcGEN(Iat)
        print("hcgrad:",np.shape(hcgrad))
        Iat_eff+=1
        for Jdir in range(spdm):
            IxJ+=1;
            vWRK = make_vAO(rttddft, vWRK,pbc,Iat,Jdir,veffgrad, 0, n_kpoints=nkpt,mu_to_iat=mu_to_iat)

            hWRK = make_vAO(rttddft, hWRK,pbc,Iat,Jdir,hcgrad, 1, n_kpoints=nkpt,mu_to_iat=mu_to_iat)

            ### print_matrices(hWRK,hcgrads_Iat[Jdir],"hWRK,hcgrad",fpath="hcgrad.dat",Append=(Iat!=0 or dir!=0))
            ret[IxJ]=dmtrace( pbc,vWRK+hWRK, dmat)

            vdist,vdev,v_at,v_vals = aNsqrMaxdiffAt( nmrgrads["veff"][ IxJ ], vWRK )
            print("#veffGRAD_%02d: %f %e (%s, %s)"%(IxJ, vdist, vdev, str(v_vals[0]),str(v_vals[1])))

            printMatrices("veffGrad_%02d.dat"%(IxJ),vWRK, nmrgrads["veff"][ IxJ ])

            hdist,hdev,h_at,h_vals = aNsqrMaxdiffAt( nmrgrads["hcore"][ IxJ ],hWRK )
            printMatrices("hcoreGrad_%02d.dat"%(IxJ),hWRK, nmrgrads["hcore"][ IxJ ])

            print("#hcoreGRAD_%02d: %f %e (%s, %s)"%(IxJ, hdist, hdev, str(h_vals[0]),str(h_vals[1])))
            
    atmlst=[]
    for Iat in range(Nat):
        if( fixedAtoms is not None ):
            if( Iat in fixedAtoms ):
                continue
        atmlst.append(Iat)
    nucGrad=grad.grad_nuc( mol_or_cell, atmlst)
    print("nucGrad:",np.shape(nucGrad),"ret:",np.shape(ret))
    eleGrad=arrayclone( ret )
    nucGrad=np.reshape(nucGrad,np.shape(ret))
    ret=ret+nucGrad
    print("#calcgrad_hfnum:",ret)

    
    print("#calcgrad_analytic_ref:",dE_ref)
    ### print("#DerivMatrices:",DerivMatrices)

    sqrdiff,maxdiff,at,vals = aNsqrMaxdiffAt(ret,dE_ref)
    print("#Egrad:", sqrdiff,maxdiff,at,vals )
    print("#E_el:",  aNsqrMaxdiffAt(eleGrad,DerivMatrices["E_el"] ) )
    print("#E_nuc:", aNsqrMaxdiffAt(nucGrad,DerivMatrices["E_nuc"] ) )

    print("#grad_hcore:", aNsqrMaxdiffAt( nmrgrads["hcore"],DerivMatrices["hcore"] ) )
    print("#grad_veff:", aNsqrMaxdiffAt( nmrgrads["veff"],DerivMatrices["veff"] ) )
    return ret


def printMatrices(fpath,lhs,rhs,format=None,description=None,Append=False):
    fd=open(fpath,("a" if(Append) else "w"))

    sqrdist,dev,at,vals = aNsqrMaxdiffAt(lhs,rhs)

    Lh=lhs;Rh=rhs
    Ndim=np.shape(lhs);rank=len(Ndim)
    dtype=( (np.array(lhs)).dtype if(rank<=2) else (np.array(lhs[0])).dtype)
    T=('I' if(dtype==int or dtype==np.int64) else ( 'D' if(dtype==float or dtype==np.float64) else \
                                                    ('Z' if(dtype==complex or dtype==np.complex128) else None ) ) )
    if( format is None ):
        format=("%d" if(T=='I') else ("%14.6f" if(T=='D') else ("%12.6f,%12.6f" if(T=='Z') else None) ) ) 

    print("#%s %s maxdiff:%e @%s %s / %s"%( T, str(np.shape(lhs)),dev,str(at),str(vals[0]),str(vals[1]) ),file=fd)
    if( description is not None ):
        print("### %s"%(description),file=fd)

    if( rank > 5 ):
        Lh=np.ravel(lhs);Rh=np.ravel(rhs)
        Ndim=np.shape(Lh);assert len(Ndim)<=5,""
    Ndim_RHS=np.shape(Rh)
    if( not i1eqb( Ndim, Ndim_RHS ) ):
        Rh=np.reshape(lhs,Ndim)

    def tostring(lv,rv,diff,T,delimiter=" \t ",delimiter2="  \t\t "):
        return format%(lv) + delimiter + format%(rv) + delimiter2 + "%e"%(diff) if(T!='Z') else \
               format%(lv.real,lv.imag) + delimiter + format%(rv.real, rv.imag)+delimiter2+"%e"%(diff) 

    diff=-1;at=None;vals=None
    for I in range(Ndim[0]):
        if(rank==1):
            dum=abs( Lh[I]-Rh[I] )
            print("%d:"%(I) + tostring(Lh[I],Rh[I],dum, T),file=fd )
            continue
        for J in range(Ndim[1]):
            if(rank==2):
                dum=abs( Lh[I][J]-Rh[I][J] )
                print("%d,%d:"%(I,J) + tostring(Lh[I][J],Rh[I][J],dum, T),file=fd )
                continue
            for K in range(Ndim[2]):
                if(rank==3):
                    dum=abs( Lh[I][J][K]-Rh[I][J][K] )
                    print("%d,%d,%d:"%(I,J,K) + tostring(Lh[I][J][K],Rh[I][J][K],dum, T),file=fd )
                    continue
                for L in range(Ndim[3]):
                    if(rank==4):
                        dum=abs( Lh[I][J][K][L]-Rh[I][J][K][L] )
                        print("%d,%d,%d,%d:"%(I,J,K,L) + tostring(Lh[I][J][K][L],Rh[I][J][K][L],dum, T),file=fd )
                        continue
                        for M in range(Ndim[4]):
                            if(rank==5):
                                dum=abs( Lh[I][J][K][L][M]-Rh[I][J][K][L][M] )
                                print("%d,%d,%d,%d,%d:"%(I,J,K,L,M) + tostring(Lh[I][J][K][L][M],Rh[I][J][K][L][M],dum, T),file=fd )
                                continue
    fd.close()
    return sqrdist,dev,at,vals
## einsum("kij,kji",matrix_AOrep,dmAO)
## dmtrace : INPUT : matrix_AOrep[ kp, nAO, nAO ]
##                           dmAO[ kp, nAO, nAO ] 
##           OUTPUT : complex scalar 
def dmtrace(pbc,matrix_AOrep,dmAO,title=None):
    if(title is not None):
        print("#dmtrace:"+title+":",np.shape(matrix_AOrep),np.shape(dmAO))
    nkpt=( 1 if(not pbc) else len(dmAO) )
    ret=np.complex128( 0.0 )
    dbgng=True ## XXX XXX
    wt0=time.time();wt1=wt0
    for kp in range(nkpt):
        dm=( dmAO if(not pbc) else dmAO[kp] )
        mat=( matrix_AOrep if(not pbc) else matrix_AOrep[kp] )
        
        mxd=np.matmul( mat, dm )
        trace=mxd[0][0];Ld=len(mat)
        for j in range(1,Ld):
            trace+=mxd[j][j]
        ret+=trace
    if(pbc):
        ret=ret/float(nkpt)
    wt2=wt1;wt1=time.time();wt_matmul=wt1-wt2

    if(dbgng):
        refr=np.complex128( 0.0 )
        for kp in range(nkpt):
            dm=( dmAO if(not pbc) else dmAO[kp] )
            mat=( matrix_AOrep if(not pbc) else matrix_AOrep[kp] )
            dum=np.einsum("ij,ji",mat,dm)
            refr+=dum
        if(pbc):
            refr=refr/float(nkpt)
        wt2=wt1;wt1=time.time();wt_einsum=wt1-wt2
        ref2=( np.einsum("kij,kji",matrix_AOrep,dmAO) if(pbc) else \
               np.einsum("ij,ji",matrix_AOrep,dmAO) )
        wt2=wt1;wt1=time.time();wt_einsum2=wt1-wt2

        printout("#DBGNG:einsum:%s %s %s elapsed:%f %f %f dev:0,%e,%e"%(\
            str(ret),str(refr),str(ref2),wt_matmul,wt_einsum,wt_einsum2,\
            abs(ret-refr),abs(ref2-ret)),fpath="DMtrace_DBG.log",Append=True)
        

    return ret

def get_mu_to_iat(mf,mol_or_cell):
    assert (mol_or_cell is not None),"mol_or_cell"
    # if cell is None: cell = mf.cell
    cart = mol_or_cell.cart
    Nsh=len(mol_or_cell._bas)
# 631G: O (10s,4p)/[3s,2p]         cart=False  
#        iat el Np Nc ?  ?  ?  ?    
#_bas_0:[ 0  0  6  1  0 40 46  0]  Nb+=Nc*mult
#_bas_1:[ 0  0  3  1  0 52 55  0]  Nb+=Nc*mult
#_bas_2:[ 0  0  1  1  0 58 59  0]  Nb+=Nc*mult
#_bas_3:[ 0  1  3  1  0 60 63  0]  ...
#_bas_4:[ 0  1  1  1  0 66 67  0]
#_bas_5:[ 1  0  3  1  0 32 35  0]
#_bas_6:[ 1  0  1  1  0 38 39  0]
#_bas_7:[ 2  0  3  1  0 32 35  0]
#_bas_8:[ 2  0  1  1  0 38 39  0]
# acct: (11s,6p,3d,2f) -> [5s,4p,3d,2f]
#_bas:iatm, el, NcGTO
#_bas_0:[ 0  0  8  2  0 54 62  0]
#_bas_1:[ 0  0  1  1  0 78 79  0]
#_bas_2:[ 0  0  1  1  0 80 81  0]
#_bas_3:[ 0  0  1  1  0 82 83  0]
#_bas_4:[ 0  1  3  1  0 84 87  0]
#_bas_5:[ 0  1  1  1  0 90 91  0]
#_bas_6:[ 0  1  1  1  0 92 93  0]
#_bas_7:[ 0  1  1  1  0 94 95  0]
#_bas_8:[ 0  2  1  1  0 96 97  0]
#_bas_9:[ 0  2  1  1  0 98 99  0]
#_bas_10:[  0   2   1   1   0 100 101   0]
#_bas_11:[  0   3   1   1   0 102 103   0]
#_bas_12:[  0   3   1   1   0 104 105   0]
#_bas_13:[ 1  0  3  1  0 32 35  0]
#_bas_14:[ 1  0  1  1  0 38 39  0]
#_bas_15:[ 1  0  1  1  0 40 41  0]
#_bas_16:[ 1  0  1  1  0 42 43  0]

# 10s/1s, 1                   (s1)                  (s2)                    (s3)                  (s4)
#      1.533000E+04           5.080000E-04          -1.150000E-04           0.000000E+00           0.000000E+00
#      2.299000E+03           3.929000E-03          -8.950000E-04           0.000000E+00           0.000000E+00
#      5.224000E+02           2.024300E-02          -4.636000E-03           0.000000E+00           0.000000E+00
#      1.473000E+02           7.918100E-02          -1.872400E-02           0.000000E+00           0.000000E+00
#      4.755000E+01           2.306870E-01          -5.846300E-02           0.000000E+00           0.000000E+00
#      1.676000E+01           4.331180E-01          -1.364630E-01           0.000000E+00           0.000000E+00
#      6.207000E+00           3.502600E-01          -1.757400E-01           0.000000E+00           0.000000E+00
#      1.752000E+00           4.272800E-02           1.609340E-01           1.000000E+00           0.000000E+00
#      6.882000E-01          -8.154000E-03           6.034180E-01           0.000000E+00           0.000000E+00
#      2.384000E-01           2.381000E-03           3.787650E-01           0.000000E+00           1.000000E+00
#O    S                       (s5)
#      0.0737600              1.0000000
#O    P                       (p1)                   (p2)                   (p3)
#      3.446000E+01           1.592800E-02           0.000000E+00           0.000000E+00
#      7.749000E+00           9.974000E-02           0.000000E+00           0.000000E+00
#      2.280000E+00           3.104920E-01           0.000000E+00           0.000000E+00
#      7.156000E-01           4.910260E-01           1.000000E+00           0.000000E+00
#      2.140000E-01           3.363370E-01           0.000000E+00           1.000000E+00
#O    P                       (p4)
#      0.0597400              1.0000000
#O    D
#      2.314000E+00           1.000000E+00           0.000000E+00
#      6.450000E-01           0.000000E+00           1.000000E+00
#O    D
#      0.2140000              1.0000000
#O    F
#      1.428000E+00           1.0000000
#O    F
#      0.5000000              1.0000000

    ret=[]
    nAO=0
    for ish in range(Nsh):
        b=mol_or_cell._bas[ish]
        iatm=b[0]
        el = b[1]
        ## 3Hl = (2+l) C 2
        mult=( (el+2)*(el+1)/2 if(cart) else 2*el+1 )
        NcGTO=b[3];nadd=NcGTO*mult
        for k in range(nadd):
            ret.append(iatm) 
        nAO+=nadd
    return np.array(ret)

def calcgrad_analytic_ref(rttddft, dmat=None, DerivMatrices=None):
    from pyscf.pbc.grad.krks import Gradients
    spdm=3
    mygrad = Gradients(rttddft)
    Egrad = mygrad.kernel()

    kpts = rttddft.kpts
    nkpts = len(kpts)

    print("#Egrad_REF:",Egrad, np.shape(Egrad) )
    if( dmat is None ):
        dmat=rttddft.make_rdm1()
    ## copy of rhf.py#kernel 
    mo_energy = mygrad.base.mo_energy
    mo_coeff = mygrad.base.mo_coeff
    mo_occ = mygrad.base.mo_occ
    dme0 = mygrad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    atmlst = mygrad.atmlst
    
    if atmlst is None:
        atmlst = range( rttddft.mol.natm )
        print("atmlst:",atmlst)
    de_el = mygrad.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
    de_nc = mygrad.grad_nuc(atmlst=atmlst)

    if( atmlst is not None ):
        Nat=len(atmlst)
    else:
        Rnuc_au,Sy=parse_xyzstring( rttddft.cell.atom,unit_BorA='B')
        Nat=len(Sy)
    N_dir= spdm * Nat

    if( DerivMatrices is None ):
        DerivMatrices={}
    for ky in ["hcore","veff","E_el","E_nuc","extra",   "de_hc","de_ve","de_nc","de_s1","de_sum"]:
        DerivMatrices.update({ky:[ None for k in range(N_dir) ]})
    DerivMatrices.update({"E_el":de_el}); print("#calcgrad_analytic_ref:DerivMatrices.E_el:",de_el);
    DerivMatrices.update({"E_nuc":de_nc});print("#calcgrad_analytic_ref:DerivMatrices.E_nuc:",de_nc);

    de_sum = np.array(de_el) + np.array(de_nc)
    print("#calcgrad_analytic_ref:Grad_de_sum:",de_sum);  ## #Grad_de_sum: [[ 2.39125924e-01 -1.97550365e-13 -2.43260775e-13]
                                                          ## [-2.39125924e-01 -2.99378416e-14 -9.57948807e-14]]
    print("atmlst",atmlst,mygrad.atmlst)
    ## copy of pbc.grad.khf.py
    aoslices = rttddft.cell.aoslice_by_atom()
    hcore_deriv = mygrad.hcore_generator(rttddft.mol)
    de = np.zeros([Nat,3])

    vhf_grad = mygrad.get_veff(dmat, kpts)
    s1 = mygrad.get_ovlp( rttddft.cell, kpts)
    dme0 = mygrad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    for x, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        
        print("#atmlst_%02d:"%(x),ia,p0,p1)
        h1ao = hcore_deriv(ia)
        de1   = np.einsum('xkij,kji->x', h1ao, dmat).real
        de[x] += np.einsum('xkij,kji->x', h1ao, dmat).real
        le=len(h1ao); assert le==spdm, "h1ao:"+str(np.shape(h1ao))
        print("h1ao:",np.shape(h1ao)) ## 3,nAO,nAO
        for j in range(le):
            DerivMatrices["hcore"][ spdm*x + j ] = arrayclone( h1ao[j] )
            DerivMatrices["de_hc"][ spdm*x + j ] = de1[j]
            ## print("h1ao_j:", np.shape(h1ao[j]), np.shape( DerivMatrices["hcore"][ spdm*x + j ] )) (8, 19, 19) (8, 19, 19)
        # nabla was applied on bra in vhf_grad, *2 for the contributions of nabla|ket>
        de1   = np.einsum('xkij,kji->x', vhf_grad[:,:,p0:p1], dmat[:,:,p0:p1]).real * 2
        de[x] += np.einsum('xkij,kji->x', vhf_grad[:,:,p0:p1], dmat[:,:,p0:p1]).real * 2
        dbg_vhftrace(True, "", h1ao, vhf_grad,p0,p1,dmat,nkpts)
        for j in range(le):
            DerivMatrices["veff"][ spdm*x + j ] = arrayclonex( vhf_grad[j,:,p0:p1], np.shape(vhf_grad[j]), 1,[ p0,p1 ] )
            DerivMatrices["de_ve"][ spdm*x + j ] = de1[j]
            # (3, 8, 19, 19) (3, 8, 5, 19) (3, 8, 19, 19)
            # print("veff_j:", np.shape(vhf_grad), np.shape(vhf_grad[:,:,p0:p1]),np.shape( DerivMatrices["veff"][ spdm*x + j ] )) 
        de1   = -np.einsum('kxij,kji->x', s1[:,:,p0:p1], dme0[:,:,p0:p1]).real * 2
        de[x] -= np.einsum('kxij,kji->x', s1[:,:,p0:p1], dme0[:,:,p0:p1]).real * 2
        for j in range(le):
            DerivMatrices["de_s1"][ spdm*x + j ] = de1[j]
            
        de[x] /= nkpts
        de[x] += mygrad.extra_force(ia, locals())
        extraF = mygrad.extra_force(ia, locals())

        # print("de[x]:",np.shape(de[x]),x,np.shape(de_nc)) ##  de[x]: (3,) 1 (2, 3)
        de[x] += de_nc[x]
        ## this equals grad_de ..
        for j in range(le):
            DerivMatrices["de_nc"][ spdm*x + j ] = de_nc[x][j]
        
        print("#extra_Force:",extraF)  # 0
        if( extraF != 0 ):
            leF=len(extraF)
            if( leF == 1 ):
                for j in range(spdm):
                    DerivMatrices["extra"][ spdm*x + j ] = extraF
            else:
                for j in range(leF):
                    DerivMatrices["extra"][ spdm*x + j ] = extraF[j]

        
    print("#Grad_de:",de); ## OK (egrad + ncgrad)
    print("#DerivMatrices[veff]:",np.shape(DerivMatrices["veff"]))
    for x, ia in enumerate(atmlst):
        for j in range(3):
            DerivMatrices["de_sum"][ spdm*x + j ]=  DerivMatrices["de_nc"][ spdm*x + j ] + DerivMatrices["de_s1"][ spdm*x + j ]\
                                                  + DerivMatrices["de_hc"][ spdm*x + j ] + DerivMatrices["de_ve"][ spdm*x + j ]
#Grad_de: [[ 2.39125924e-01 -1.98401564e-13 -2.43147439e-13]
# [-2.39125924e-01 -2.97821799e-14 -9.55382033e-14]]
# 0 Li     0.2391259237    -0.0000000000    -0.0000000000
# 1 H    -0.2391259237    -0.0000000000    -0.0000000000

    diff=max( abs(np.ravel(de) - np.ravel(de_sum)) )
    ### assert diff < 1e-6,"diff:%e  %s / %s"%(diff, str(de), str(de_sum))
    return de


def dbg_vhftrace(pbc,description,h1ao, vhf_grad,p0,p1,dmat,nKpt):
    Ndim = np.shape(vhf_grad)
    retv = np.einsum('xkij,kji->x', vhf_grad[:,:,p0:p1], dmat[:,:,p0:p1]).real * 2 / nKpt
    refr =[]
    vhf_Matr=[]
    for j in range(3):
        vhf_Matr.append( arrayclonex( vhf_grad[j, :, p0:p1 ]*2, np.shape( vhf_grad[j] ), 1, [p0,p1] ) ) ## we include factor 2 here
        refr.append( dmtrace(pbc, vhf_Matr[j], dmat,title="DBG_vhftrace:"+description) )
    print("dbg_vhftrace:",retv,refr)
#calcForce::decomposition:de_ve: [0.18940722064783058, -1.2209338639400545e-12, 8.102987148622681e-13, -0.18940722064789706, 1.2647494007657436e-12, -7.777776706872856e-13]
# dbg_vhftrace: [ 1.89407221e-01 -1.22357588e-12  8.07765411e-13] [(0.18940722064785204+6.055902239317778e-14j), (-1.2235994094798608e-12-8.900191378437408e-12j), (8.077852155287619e-13-1.1945164388732031e-13j)]
#ref2: [(0.18940722064785204+6.055902239317778e-14j), (-1.2235994094798608e-12-8.900191378437408e-12j), (8.077852155287619e-13-1.1945164388732031e-13j)]
#ref3: [(0.18940722064785204+6.055902239317778e-14j), (-1.2235994094798608e-12-8.900191378437408e-12j), (8.077852155287619e-13-1.1945164388732031e-13j)]

#[ 1.89407221e-01 -1.22093386e-12  8.10298715e-13] 
# [(0.011837951290489439+3.78307677512285e-15j), (-7.630785384279695e-14-5.562614882754251e-13j), (5.064229449443219e-14-7.469133081230207e-15j)]
    ref2=[]
    for j in range(3):
        cum=0.0 + 1j*0.0
        for kp in range(nKpt):
            a = vhf_grad[j,kp,p0:p1]*2   ## nAO',nAO
            b = dmat[kp,:,p0:p1]       ## nAO, nAO'
            axb=np.matmul(a,b)
            le=len(axb)
            tr= 0.0 + 1j*0.0
            for k in range(le):
                tr+= axb[k][k]
            cum +=tr
        cum=cum/float(nKpt);ref2.append(cum)
    print("#ref2:",ref2)
#ref2: [(0.09470361032391551+3.02646142009828e-14j), (-6.104628307423756e-13-4.450091906203401e-12j), (4.051383559554575e-13-5.975306464984165e-14j)]
    ref3=[]
    for j in range(3):
        gr_j = arrayclonex( vhf_grad[j, :, p0:p1 ]*2, np.shape( vhf_grad[j] ), 1, [p0,p1] )
        cum=0.0 + 1j*0.0
        for kp in range(nKpt):
            a = gr_j[kp]   ## nAO',nAO
            b = dmat[kp]   ## nAO, nAO'
            print("axb:",np.shape(a),np.shape(b))
            axb=np.matmul(a,b)
            le=len(axb)
            tr= 0.0 + 1j*0.0
            for k in range(le):
                tr+= axb[k][k]
            cum +=tr
        cum=cum/float(nKpt);ref3.append(cum)
    print("#ref3:",ref3)
    return retv
#ref3: [(0.09470361032391551+3.02646142009828e-14j), (-6.104628307423756e-13-4.450091906203401e-12j), (4.051383559554575e-13-5.975306464984165e-14j)]

def arrayclonex(src,Ndim,jth,Range):
    #
    # jth index [ Range[0]:Range[1] ]
    # ret[:,..., (Range[0]:Range[1]),... ] = src[:,... ]
    #            |_ j_th
    ret=np.zeros(Ndim, src.dtype)
    rank=len(Ndim)

    Ndim_SRC=np.shape(src);rank_SRC=len(Ndim_SRC)
    print("#arrayclonex:_%03d:[%d:%d] in %s"%(jth,Range[0],Range[1],str(Ndim)))
    Ld_jth=Range[1]-Range[0];
    assert rank_SRC == rank, ""
    assert jth<rank,""; 
    assert Range[1]<=Ndim[jth],""
    assert Ndim_SRC[jth]==Ld_jth,""

    if( jth == 0 ):
        ret[Range[0]:Range[1]]=src
    elif( jth == 1 ):
        ret[:,Range[0]:Range[1]]=src
    elif( jth == 2 ):
        ret[:,:,Range[0]:Range[1]]=src
    elif( jth == 3 ):
        ret[:,:,:,Range[0]:Range[1]]=src
    elif( jth == 4 ):
        ret[:,:,:,:,Range[0]:Range[1]]=src
    else:
        assert False,""
    return ret

def concat_buffer(buf,item):
    le0=len(buf)
    buf1=( buf if(isinstance(buf,list)) else buf.tolist() )
    le=len(item)
    for I in range(le):
        buf1.append( item[I] )
    buf=np.array(buf1)
    le1=len(buf)
    assert le1==le0+le,""
    return buf

### def get_derivMatrices_FFTDF(md,rttddft,dmat=None, Nstep=1,fixedAtoms=None):
###    if( Nstep == 1 ):
###        return get_derivMatrices_FFTDF_(md,rttddft,dmat=dmat,atmlst=None)
###    else:
###        ret={}
###        Nat=md.Nat
###        Neach = (Nat + Nstep - 1)//Nstep
###        for Istep in range(Nstep):
###            lwb=Istep*Neach; upb=min(Nat,lwb+Neach)
###            if( upb<=lwb ):
###                continue
###            atmlst=range(lwb,upb)
###            heapcheck(">>>get_derivMatrices_FFTDF:step:%02d/%d"%(Istep,Nstep),text=str(atmlst))
###            buf=get_derivMatrices_FFTDF_(md,rttddft,dmat=None, atmlst=atmlst)
###            for ky in buf:
###                item=buf[ky]
###                if( ky not in ret ):
###                    ret.update({ky:item})
###                else:
###                    sub=ret[ky]
###                    ret.update({ky:concat_buffer(sub,item)})
###        heapcheck(">>>get_derivMatrices_FFTDF:end")
###        return ret

def get_derivMatrices_FFTDF(md,rttddft,Nat,dmat=None, fixedAtoms=None):
# 
# retv :: Dict ["hcore","veff","E_el","E_nuc","extra"][ N_dir ][ scalar OR [nKpt][nAO][nAO] ]
#
    import time
    wt00=time.time();wt1=wt00;timing={}
    atmlst=None ## default,all atoms
    if( fixedAtoms is not None ):
        atmlst=[]
        for I in range(Nat):
            if(I in fixedAtoms):
                continue
            atmlst.append(I)
    from pyscf.pbc.grad.krks import Gradients
    spdm=3
    if( dmat is None ): dmat = md.calc_tdDM(rttddft,ovrd=True)
    # print("#get_derivMatrices_FFTDF:start")
    if( rttddft.mo_coeff is None ):
        rttddft.mo_coeff = md.tdMO
        print("#get_derivMatrices_FFTDF::MO set:md.tdMO")
    else:
        diff=aNmaxdiff( rttddft.mo_coeff, md.tdMO )
        print("#get_derivMatrices_FFTDF::check_diff:|rttddft.mo - moldyn.tdMO|:%e"%(diff))

    if( rttddft.mo_occ is None ):
        rttddft.mo_occ = md.mo_occ
        print("#get_derivMatrices_FFTDF::occ set:md.mo_occ",md.mo_occ)
    else:
        diff=aNmaxdiff( rttddft.mo_occ, md.mo_occ )
        print("#get_derivMatrices_FFTDF::check_diff:|rttddft.mo_occ - moldyn.mo_occ|:%e"%(diff))

    if( rttddft.mo_energy is None ):        
        rttddft.mo_energy = get_eOrbs(md, rttddft)
        print("#get_derivMatrices_FFTDF::mo_energy:md.mo_energy",np.shape(rttddft.mo_energy))
    else:
        print("#get_derivMatrices_FFTDF::mo_energy:",np.shape(rttddft.mo_energy))

    heapcheck(">>>> get_derivMatrices_FFTDF_.start")
    mygrad = Gradients(rttddft)
    wt2=wt1;wt1=time.time();timing.update({"prep":wt1-wt2})
    heapcheck(">>>> get_derivMatrices_FFTDF_.mygrad")
    ## print("#get_derivMatrices_FFTDF::mygrad.base:",mygrad.base,mygrad.base.mo_occ)
    Egrad_decomposition={};Dict_matrices={}
    try:
        if( atmlst is not None ):
            Egrad = mygrad.kernel(atmlst=atmlst,decomposition=Egrad_decomposition,
                                  fn_save_matrices=fn_save_matrices_std, buf_matrices=Dict_matrices)
            heapcheck(">>>> get_derivMatrices_FFTDF_.af grad.kernel:Nat=%d/%d"%(len(atmlst),Nat) )
        else:
            Egrad = mygrad.kernel(decomposition=Egrad_decomposition,
                                  fn_save_matrices=fn_save_matrices_std, buf_matrices=Dict_matrices)
            heapcheck(">>>> get_derivMatrices_FFTDF_.af grad.kernel:Nat=%d"%(Nat) )
    except TypeError as te:
        print("#get_derivMatrices_FFTDF::TypeError",te)
        raise
        Egrad = mygrad.kernel(mo_energy=rttddft.mo_energy, mo_coeff=md.tdMO, mo_occ=md.mo_occ,
                               decomposition=Egrad_decomposition,fn_save_matrices=fn_save_matrices_std, buf_matrices=Dict_matrices)
    wt2=wt1;wt1=time.time();timing.update({"grad_kernel":wt1-wt2})
    print("#get_derivMatrices_FFTDF::Egrad:",Egrad)
    kpts = rttddft.kpts
    nkpts = len(kpts)
#    if( dmat is None ):
#        dmat=rttddft.make_rdm1()

    mo_energy = mygrad.base.mo_energy
    mo_coeff = mygrad.base.mo_coeff
    mo_occ = mygrad.base.mo_occ
    atmlst = mygrad.atmlst
    if atmlst is None:
        atmlst = range( rttddft.mol.natm )
        print("#get_derivMatrices_FFTDF:atmlst:",atmlst)

    print("#get_derivMatrices_FFTDF:Egrad_decomposition:",Egrad_decomposition);
    de_el = Egrad_decomposition["de_el"]
    de_nc = Egrad_decomposition["de_nc"]
    ## de_el = mygrad.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
    ## de_nc = mygrad.grad_nuc(atmlst=atmlst)

    Nat=len(atmlst)
    N_dir= spdm * Nat
    DerivMatrices={}
    for ky in ["hcore","veff","E_el","E_nuc","extra"]:   ## we need  "hcore","veff","E_nuc" 
        DerivMatrices.update({ky:[ None for k in range(N_dir) ]})
    DerivMatrices.update({"E_el":de_el}); print("#calcgrad_analytic_ref:DerivMatrices.E_el:",de_el);
    DerivMatrices.update({"E_nuc":de_nc});print("#calcgrad_analytic_ref:DerivMatrices.E_nuc:",de_nc);
    de_sum = np.array(de_el) + np.array(de_nc)

    aoslices = rttddft.cell.aoslice_by_atom()
    hcore_deriv = None
    if( "hcore" not in Dict_matrices):
        hcore_deriv = mygrad.hcore_generator(rttddft.mol)
    de = np.zeros([Nat,3])

    if("vhf" in Dict_matrices):
        vhf_grad = Dict_matrices["vhf"]
    else:
        vhf_grad = mygrad.get_veff(dmat, kpts)
        wt2=wt1;wt1=time.time();timing.update({"vhf_grad":wt1-wt2})

    s1 = mygrad.get_ovlp( rttddft.cell, kpts)
    wt2=wt1;wt1=time.time();timing.update({"s1":wt1-wt2})
    dme0 = mygrad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    wt2=wt1;wt1=time.time();timing.update({"rdm1e":wt1-wt2})

    de_ve=[]; de_hc=[]; de_s1=[]
    for x, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        if( "hcore" not in Dict_matrices):
            h1ao = hcore_deriv(ia)
            wt2=wt1;wt1=time.time();
            if("h1ao" not in timing):
                timing.update({"h1ao":wt1-wt2})
            else:
                timing["h1ao"]+=wt1-wt2
        else:
            h1ao = Dict_matrices["hcore"][x]
            if( "x_to_ia" in Dict_matrices ):
                idum = Dict_matrices["x_to_ia"][x] 
                assert (ia==idum),"%d/%d"%(ia,idum)+str(Dict_matrices["x_to_ia"])
        de_hc.append( np.einsum('xkij,kji->x', h1ao, dmat).real/float(nkpts) )
        de[x] += np.einsum('xkij,kji->x', h1ao, dmat).real
        le=len(h1ao); assert le==spdm, "h1ao:"+str(np.shape(h1ao))
        ### print("h1ao:",np.shape(h1ao)) ## 3,nAO,nAO
        for j in range(le):
            DerivMatrices["hcore"][ spdm*x + j ] = arrayclone( h1ao[j] )
            ### print("h1ao_j:", np.shape(h1ao[j]), np.shape( DerivMatrices["hcore"][ spdm*x + j ] )) (8, 19, 19) (8, 19, 19)
        
        de[x] += np.einsum('xkij,kji->x', vhf_grad[:,:,p0:p1], dmat[:,:,p0:p1]).real * 2
        de_ve_einsum = np.einsum('xkij,kji->x', vhf_grad[:,:,p0:p1], dmat[:,:,p0:p1]).real * 2 /float(nkpts)
        de_ve_x = [];
        for j in range(le):
            DerivMatrices["veff"][ spdm*x + j ] = arrayclonex( vhf_grad[j,:,p0:p1]*2, np.shape(vhf_grad[j]), 1,[ p0,p1 ] )
            trVeDm =dmtrace(md.pbc, DerivMatrices["veff"][ spdm*x + j ], dmat,title="DBG_vhftrace:")  ## takes account of nkpts
            de_ve_x.append( trVeDm )
            # assert abs(trVeDm - de_ve_einsum[j])<1e-7,"%e / %e"%(trVeDm,de_ve_einsum[j])
            # 2022.01.14 I added factor *2 
            # (3, 8, 19, 19) (3, 8, 5, 19) (3, 8, 19, 19)
            # print("veff_j:", np.shape(vhf_grad), np.shape(vhf_grad[:,:,p0:p1]),np.shape( DerivMatrices["veff"][ spdm*x + j ] ))
        dev = max( np.ravel(de_ve_einsum) -np.ravel(de_ve_x));
        assert dev<1e-6,""
        de_ve.append( de_ve_x )
        de_s1_x =  -np.einsum('kxij,kji->x', s1[:,:,p0:p1], dme0[:,:,p0:p1]).real * 2 /float(nkpts)
        de[x] -= np.einsum('kxij,kji->x', s1[:,:,p0:p1], dme0[:,:,p0:p1]).real * 2
        de_s1.append( de_s1_x )
     
        de[x] /= nkpts
        if( "extra" in Dict_matrices):
            extraF = Dict_matrices["extra"][x]
        else:
            extraF = mygrad.extra_force(ia, locals())
        de[x] += extraF
        de[x] += de_nc[x]
        
        # print("#extra_Force:",extraF)  # 0 (scalar)
        if( extraF != 0 ):
            leF=len(extraF)
            if( leF == 1 ):
                for j in range(spdm):
                    DerivMatrices["extra"][ spdm*x + j ] = extraF
            else:
                for j in range(leF):
                    DerivMatrices["extra"][ spdm*x + j ] = extraF[j]
    wt2=wt1;wt1=time.time();timing.update({"einsum":wt1-wt2})
    diff=max( abs(np.ravel(de) - np.ravel(de_sum)) )
    assertf( diff<1e-6,"de:%s / %s"%(str(de),str(de_sum)),level=1)
    de_hc=np.ravel(de_hc);de_ve=np.ravel(de_ve);de_s1=np.ravel(de_s1);de_nc1D=np.ravel(de_nc)
    le=len(de_hc)
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    n_call = rttddft_common.Countup("get_derivMatrices_FFTDF")
    if( (n_call < 3 or n_call == 20) and MPIrank==0 ):
        print("#get_derivMatrices_FFTDF:decomposition:de_SUM_el:",[ (de_hc[k] + de_ve[k] + de_s1[k])  for k in range(le) ])
        print("#get_derivMatrices_FFTDF:decomposition:de_SUM_tot:",[ (de_hc[k] + de_ve[k] + de_s1[k] + de_nc1D[k])  for k in range(le) ])

    assert diff<1e-6,""
    heapcheck(">>>> get_derivMatrices_FFTDF_.END")
    wt2=wt1;wt1=time.time();timing.update({"misc":wt1-wt2})
    
    if( n_call < 3 or n_call == 20 or (MPIrank==0 and n_call % 200 == 0) ):
        print("#get_derivMatrices:$%02d:n_call%d:elapsed:%f "%(MPIrank,n_call,wt1-wt00)+str(timing))
    return DerivMatrices


#calcForce::decomposition:de_hc: [-0.6048776445325672, 8.792403894534924e-12, -3.1558318916535685e-12, 0.604877644536944, -9.06252851525815e-12, 2.7832008400214535e-12]
#calcForce::decomposition:de_ve: [0.18940722064785215, -1.223575882398597e-12, 8.077654110514213e-13, -0.1894072206479185, 1.2673528294682205e-12, -7.751768152964514e-13]
#calcForce::decomposition:de_s1: [-0.00641148674594829, -7.764954972129708e-12, 2.1050545810436393e-12, 0.006411486745948649, 7.763403923918255e-12, -2.1037317874404715e-12]
#calcForce::decomposition:de_nc: [0.661007834315346, -1.582067810090848e-15, 2.42861286636753e-17, -0.6610078343153459, 1.6930901125533637e-15, 8.673617379884035e-17]
#calcForce::decomposition:de_sum: [0.23912592368468272, -1.9770902780347243e-13, -2.429876134298442e-13, -0.2391259236803718, -3.007867175912176e-14, -9.562102654167053e-14]
#----------------
# -6.04877645e-01 + 0.189407220647904 -6.41148675e-03
#get_derivMatrices_FFTDF:decomposition:de: [[ 2.39125924e-01 -1.97974644e-13 -2.43014473e-13][-2.39125924e-01 -2.98949839e-14 -9.55658494e-14]]
#get_derivMatrices_FFTDF:decomposition:de_hc: [array([-6.04877645e-01,  8.78584213e-12, -3.15464950e-12]), array([ 6.04877645e-01, -9.05599012e-12,  2.78204792e-12])]
#get_derivMatrices_FFTDF:decomposition:de_ve: [[(0.189407220647904+6.060717360941057e-14j), (-1.2194162514183814e-12-8.900234007133403e-12j), (8.043008872629098e-13-1.1939335589588766e-13j)], [(-0.1894072206479706-6.177722945483058e-14j), (1.2631207411529085e-12-1.6996542016366935e-15j), (-7.717209523184741e-13+2.2768259554397516e-15j)]]
#get_derivMatrices_FFTDF:decomposition:de_s1: [array([-6.41148675e-03, -7.76282649e-12,  2.10729056e-12]), array([ 6.41148675e-03,  7.76127545e-12, -2.10597145e-12])]
#get_derivMatrices_FFTDF:decomposition:de_SUM: [(-0.4218819106306605+6.060717360941057e-14j), (-1.9640061461054911e-13-8.900234007133403e-12j), (-2.4305804815260727e-13-1.1939335589588766e-13j), (0.4218819106349726-6.177722945483058e-14j), (-3.159393465172485e-14-1.6996542016366935e-15j), (-9.564448447206493e-14+2.2768259554397516e-15j)]
#get_derivMatrices_FFTDF:decomposition:de: [[ 2.39125924e-01 -1.97740343e-13 -2.42813838e-13][-2.39125924e-01 -2.96294247e-14 -9.57975145e-14]]
#get_derivMatrices_FFTDF:decomposition:de_hc: [array([-6.04877645e-01,  8.77734037e-12, -3.16133005e-12]), array([ 6.04877645e-01, -9.04744418e-12,  2.78871370e-12])]
#get_derivMatrices_FFTDF:decomposition:de_ve: [[(0.18940722064782067+6.063017054759368e-14j), (-1.2187964451932518e-12-8.90022237725848e-12j), (8.073021354744291e-13-1.1951131488111554e-13j)], [(-0.18940722064788743-6.176721904863186e-14j), (1.2629824019592143e-12-1.698066185035282e-15j), (-7.74733394092281e-13+2.2804314402416723e-15j)]]
#get_derivMatrices_FFTDF:decomposition:de_s1: [array([-6.41148675e-03, -7.75469846e-12,  2.11118352e-12]), array([ 6.41148675e-03,  7.75314676e-12, -2.10986267e-12])]
#get_derivMatrices_FFTDF:decomposition:de_SUM: [(-0.4218819106306605+6.063017054759368e-14j), (-1.9615453588342445e-13-8.90022237725848e-12j), (-2.4284439387733383e-13-1.1951131488111554e-13j), (0.421881910634975-6.176721904863186e-14j), (-3.131502240630195e-14-1.698066185035282e-15j), (-9.588236984042695e-14+2.2804314402416723e-15j)]

#    print("#get_derivMatrices_FFTDF:decomposition:de:",de)
#    print("#get_derivMatrices_FFTDF:decomposition:de_hc:",de_hc) # [ 4.83902116e+00 -7.25173118e-11  2.23130409e-11]
#    print("#get_derivMatrices_FFTDF:decomposition:de_ve:",de_ve) 
# [ [(0.1894072206477209+6.063126496833584e-14j), (-1.2281381848999787e-12-8.900200137431778e-12j), (8.080897407252345e-13-1.1946683087308191e-13j)], 
#   [(-0.18940722064778714-6.17692666198519e-14j), (1.2719304961047628e-12-1.7058857014271801e-15j), (-7.754284352342526e-13+2.2895488658148303e-15j)]]
# get_derivMatrices_FFTDF:decomposition:de_s1: [array([-6.41148675e-03, -7.76284290e-12,  2.11073496e-12]), array([ 6.41148675e-03,  7.76129120e-12, -2.10940827e-12])]
#    print("#get_derivMatrices_FFTDF:decomposition:de_s1:",de_s1)

def assertf(bool,msg,loglv=0,level=-1):
    if( loglv > 0 or (not bool) ):
        fd=open("assertf.log","a")
        strbuf=("#assertf:log:" if(bool) else "#python assertion failed:")+str(msg)+" \t "+str(datetime.datetime.now())\
               +( " \t\t "+str(rttddft_common.get_job(True)) if(rttddft_common.get_job(True) is not None) else "")
        print(strbuf,file=fd);
        fd.close()
    if(bool):
        return 0
    if(level >=0 ):
        print(strbuf);return 1
    assert False,strbuf
