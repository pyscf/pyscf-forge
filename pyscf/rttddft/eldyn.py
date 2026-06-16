import numpy as np
import os
import random
import time
import sys
import math
import datetime
from mpi4py import MPI

from .rttddft_common import rttddft_common
## from pyscf.pyscf_common import pyscf_common
from .Loglv import printout
from .mpiutils import mpi_Bcast

from .futils import futils
from .calcNacm import calc_Vdot_nacm_eff,prtNacm
from .physicalconstants import physicalconstants, PhysicalConstants
from .laserfield import kickfield
from .rttddft01 import calc_FieldEng_cell,update_Aind_over_c,get_dipole,get_current,get_Nele,trace_dmat
from .diis01 import DIIS01,dot_AOmatrices
# from utils import update_dict,printout_dict,ixj_to_IandJ,readwritezN,i1eqb, print_Htensor,dump_zNarray,read_zNdaf,i1toa,a1maxdiff,d1toa,zNtoa,arrayclone,to_complexarray,aNmaxdiff,print_TDeorbs,a2sqrdiff,calc_Sroot
from .utils import update_dict,printout_dict,readwritezN,i1toa,d1toa,arrayclone,to_complexarray,\
aNmaxdiff,print_TDeorbs,i1eqb,a1maxdiff,ixj_to_IandJ,a2sqrdiff,calc_Sroot,normalize,calc_ldos,tostring
from .Moldyn import get_FockMat
from .Logger import Logger
## from serialize import diff_objects,serialize
from .bandstructure import calc_ldos
## plot_dispersion_kLine

class eldyn_:
    Dipoles_ini_=None
    Dipoles_latest_=None
    N_debuglog=0

class eldyn_default_:
    DIIS_nstep_thr=6
    DIIS_ervec_thr=1.0e-4
    dm2diff_TOL=5.0e-7
    maxit=200
    DIIS_bufsz=10

# here we also use the md object since we need complex WF
# @param tm_fs_offset  None >> start afresh
#                      tm_fs >> restart from tm_fs ... md.time_AU must be consistent with tm_fs ...
def calc_OAS(md, rttddft, Evector, gauge_LorV, dmfilepath, 
        tm_fs_end=24.0, tm_fs_step=0.008, tm_fs_offset=None,
        tmStart_AU = 0.0, tmKick_AU =None, Nstep_Kick=20, logger=None, logfile=None, propagator="CN",
        Nstep_dump=None, tsecond_dump=None, WCt00=None, dumpfilename=None, Nstep_calcEng=None, 
        Nstep_calcDOS=None, check_timing=False, params=None, job=None ):
## GPAW recommends Nstep_Kick = kick_ampd / 1.0e-4
## Here default values are kick_ampd = 1.0e-3 so Nstep_Kick=20 is large enough

    calc_MOprojection=False

    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    check_sync=1
    if(job is not None):
        rttddft_common.set_job(job)
    else:
        rttddft_common.set_job("_".join( ["OAS",  gauge_LorV+'G', "dt%3.1f"%(tm_fs_step*1e+3)+"as"]))

    
    verbose=0
    if( tsecond_dump is not None ):
        assert WCt00 is not None,""
    restart=(tm_fs_offset is not None)
    if(tm_fs_offset is not None):
        assert (tm_fs_offset > tm_fs_step),""
    wct_00=time.time()

    assert (propagator=="CN" or propagator=="Magnus"),"wrong propagator:"+propagator
    use_Magnus=(propagator=="Magnus")
    fnme1="calc_OAS.log";tm00=time.time()
    printout("#calc_OAS:"+rttddft_common.get_job(True)
        +":"+propagator+":%7.3fas tend:%8.3ffs"%(tm_fs_step*1000,tm_fs_end),fpath=fnme1,Append=True,Threads=[0]);
    if( logger is None ):
        assert (logfile is not None),"logfile"
        logger=Logger(logfile)
        md.logger=logger
    aut_in_femtosec=PhysicalConstants.aut_in_femtosec(); 
    pbc = md.pbc
    assert (gauge_LorV=='L'  or gauge_LorV=='V'),"wrong gauge:"+gauge_LorV
    if(tmKick_AU is None):
        tmKick_AU = 1.0    # By definition this must be unity
    
    if( MPIsize>1 ):
        sqrediff,maxabsdiff,At,vals = mpi_aNdiff(md.tdMO,"calc_OAS.initial_tdMO",sync=True)
        printout("#tdMO: %6d %04d %16.8f %16.6e %10.4f+j%10.4f/%10.4f+j%10.4f at "%(0,0,\
                np.sqrt(sqrediff),maxabsdiff,vals[0].real,vals[0].imag,vals[1].real,vals[1].imag)\
                +str(At), fnme_format="calc_OAS_sync%02d.log",Append=False)
    
    F=None;nAO=None;nkpt=None
    rttddft.update_Sinv()

    # if MPIsize>1, we synchronize Sinv ( you may not need such operation )
    #
    if( (MPIsize>1) and (check_sync>1) ):
        sqrediff,maxabsdiff,At,vals = mpi_aNdiff( rttddft._Sinvrt,"calc_OAS.initial_Sinvrt",sync=True)
        printout("### Ssqrt: %6d %04d %16.8f %16.6e %10.4f+j%10.4f/%10.4f+j%10.4f at "%(0,0,\
                np.sqrt(sqrediff),maxabsdiff,vals[0].real,vals[0].imag,vals[1].real,vals[1].imag)\
                +str(At), fnme_format="calc_OAS_sync%02d.log",Append=True)
        
    rttddft_common.Dict_setv("rtTDDFT_Istep",0)
    if( gauge_LorV == 'L' and (not restart) ):
        printout("#calc_OAS: applying delta-like pulse..");
        logfile="calc_OAS.log";
        printout("#calc_OAS:DIPOLE GAUGE:%f %f %f"%(Evector[0],Evector[1],Evector[2]),fpath=logfile,Append=True,Threads=[0]);
        rttddft.dipolegauge=True; rttddft.velocitygauge=False
        # apply  H= - (-e r : E) 
        if( not pbc ):
            dipole_matr = rttddft.mol.intor('int1e_r',comp=3, hermi=1) # 3,nAO,nAO

            if( F is None ):
                Ndim_dp = np.shape(dipole_matr)
                nAO=Ndim_dp[1]; assert (Ndim_dp[0]==3 and Ndim_dp[2]==nAO),""
                F=np.zeros([nAO,nAO],np.complex128)
            for I in range(nAO):
                for J in range(nAO):
                    F[I][J]= Evector[0] * dipole_matr[0][I][J]
                    for dir in range(1,3):
                        F[I][J]+= Evector[dir] * dipole_matr[dir][I][J]
            
            S1e_1=rttddft.get_ovlp()
            if( calc_MOprojection ):
                md.calc_MOprojection(rttddft,"",fpath="calcOAS_checkMO.dat",append=False)

            if(use_Magnus):
                tdMO_AF = Magnus_AOrep(rttddft._Ssqrt,rttddft._Sinvrt,nAO,F,F, tmKick_AU, md.tdMO)
            else:
                tdMO_AF = CNlnr_AOrep (rttddft._Ssqrt,rttddft._Sinvrt,F,F, tmKick_AU, Nstep_Kick, md.tdMO,
                                       check_vecnorms=True, check_projection="CheckMO_CNlnrAOrep.dat", rttddft=rttddft )
            for I in range(len(tdMO_AF)):
                for J in range(len(tdMO_AF[0])):
                    md.tdMO[I][J]=tdMO_AF[I][J]
            dict={"excitedstate_populations":None}
            if( calc_MOprojection ):
                md.calc_MOprojection(rttddft,"",fpath="calcOAS_checkMO.dat",dict=dict)
                logger.Info("#excitedstate_populations:"+d1toa(dict["excitedstate_populations"]))
        else:
            #
            # Usually LG does not make sense in the periodic direction.
            # But it does make sense if it is perpendicular to the periodic direction 
            # (e.g. 2D system in the x,y plane + electric field in z direction ) 
            #
            kvectors = np.reshape( rttddft.kpts, (-1,3))
            dipole_matr = rttddft.cell.pbc_intor('int1e_r', comp=3, hermi=1, kpts=kvectors)
            if( F is None ):
                Ndim_dp = np.shape(dipole_matr)  # nkp,3,nAO,nAO
                nAO =Ndim_dp[2]; nkp=Ndim_dp[0];
                assert (Ndim_dp[1]==3 and Ndim_dp[2]==Ndim_dp[3]),"--"
                F= np.zeros([nAO,nAO], dtype=np.complex128)
            
            for kp in range(nkp):
                S1e_k =rttddft.get_ovlp(rttddft.cell,kvectors[kp])
                for I in range(nAO):
                    for J in range(nAO):
                        F[I][J]= Evector[0] * dipole_matr[kp][0][I][J]
                        for dir in range(1,3):
                            F[I][J]+= Evector[dir] * dipole_matr[kp][dir][I][J]

                if(use_Magnus):
                    tdMO_AF = Magnus_AOrep(rttddft._Ssqrt[kp],rttddft._Sinvrt[kp],nAO,F,F, tmKick_AU, md.tdMO[kp])
                else:
                    tdMO_AF = CNlnr_AOrep (rttddft._Ssqrt[kp],rttddft._Sinvrt[kp],F,F,tmKick_AU,Nstep_Kick,md.tdMO[kp],kp=kp,
                                       check_vecnorms=True, rttddft=rttddft)
                
                for I in range(len(tdMO_AF)):
                    for J in range(len(tdMO_AF[0])):
                        md.tdMO[kp][I][J]=tdMO_AF[I][J]

        if( MPIsize>1 ):
            sqrediff,maxabsdiff,At,vals = mpi_aNdiff(md.tdMO,"calc_OAS.0th_tdMO",sync=True)
            printout("calc_OAS.initial_tdMO: %16.8f %16.6e %10.4f+j%10.4f/%10.4f+j%10.4f at "%(\
                    np.sqrt(sqrediff),maxabsdiff,vals[0].real,vals[0].imag,vals[1].real,vals[1].imag)\
                    +str(At), fnme_format="calc_OAS_sync%02d.log",Append=False)
    elif( gauge_LorV == 'L' and restart ):
        rttddft.dipolegauge=True; rttddft.velocitygauge=False
        logfile="calc_OAS.log";
        printout("#calc_OAS:restart:tm_fs_offset=%f:DIPOLE GAUGE:%f %f %f"%(tm_fs_offset,Evector[0],Evector[1],Evector[2]),
                fpath=logfile,Append=True,Threads=[0]);
    elif( gauge_LorV == 'V' ):
        logfile="calc_OAS.log";
        if( restart ):
            printout("#calc_OAS:restart:tm_fs_offset=%f:VELOCITY GAUGE:%f %f %f"%(tm_fs_offset,Evector[0],Evector[1],Evector[2]),
                    fpath=logfile,Append=True,Threads=[0]);
        else:
            printout("#calc_OAS:VELOCITY GAUGE:%f %f %f"%(Evector[0],Evector[1],Evector[2]),
                    fpath=logfile,Append=True,Threads=[0]);

        rttddft.dipolegauge=False; rttddft.velocitygauge=True
        field=None

        fd01=open(rttddft_common.get_job(True)+"_tdfield.log","a");print("tdField:kickfield",file=fd01);fd01.close()
        field=kickfield(Evector,tmStart_AU=tmStart_AU)

        logger.Info("#calc_OAS:velocity gauge... kickField:"+str(Evector)+" tmStart:%f"%(tmStart_AU));
        md.td_field=field
        md.gauge_LorV='V'
        rttddft._td_field=field
    
    if( check_timing ):
        rttddft.print_log("calc_OAS:zero_th-kick-step:",timing=True);

    logger.Info("#calc_OAS: starting normal time propagation....:dt=%f T=%f"%(tm_fs_step,tm_fs_end));

    calc_OAS_propagate( md, rttddft, dmfilepath, Evector, tmKick_AU, 
                        tm_fs_end=tm_fs_end,tm_fs_step=tm_fs_step, tm_fs_offset=tm_fs_offset, 
                        propagator=propagator, Nstep_dump=Nstep_dump, tsecond_dump=tsecond_dump, 
                        WCt00=WCt00, dumpfilename=dumpfilename, restart=restart, Nstep_calcEng=Nstep_calcEng,
                        Nstep_calcDOS=Nstep_calcDOS, check_timing=check_timing, params=params)
    
    logfile="calc_OAS.log";tm01=time.time();
    printout("## calc_OAS:"+ rttddft_common.get_job(True)+":End:%f sec"%(tm01-tm00),
           fpath=logfile,Append=True,Threads=[0]);


def calc_OAS_propagate( md, rttddft, dmfilepath, Evector, tmKick_AU, 
                        tm_fs_end=24.0,tm_fs_step=0.008, tm_fs_offset=None, logger=None, propagator="CN",
                        Nstep_dump=None, tsecond_dump=None, WCt00=None, dumpfilename=None, Nstep_no_save=None, 
                        tsecond_no_save=60*60, verbose=1, restart=False, Nstep_calcEng=None, Nstep_calcDOS=None,
                        check_timing=False, params=None, nmax_warn=-1, nmax_seq_warn=3):
    
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    WCt01=time.time();WCtCUR=WCt01
    if( tsecond_dump is not None ):
        assert WCt00 is not None,""
    if( (Nstep_dump is not None) or (tsecond_dump is not None) ):
        assert dumpfilename is not None 
    # by default this starts from this.time_AU and propagates up to tm_fs_end
    # 
    if logger is None:logger=md.get_logger(set=True);
    aut_in_femtosec=PhysicalConstants.aut_in_femtosec()
    assert (propagator=="CN" or propagator=="Magnus"),"wrong propagator:"+propagator
    use_Magnus=(propagator=="Magnus")
    job=rttddft_common.get_job(True)
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank()
    fno_incf=-1
    checkField=(rttddft._td_field is not None) ## True

    cLIGHT_AU=PhysicalConstants.CLIGHTinAU()
    Fallbackschemes=[ [{'type':'best','reltthr':10}] ] ## 'best' fallback should be the last one
    N_fallbackschemes=len( Fallbackschemes )
    
    n_warn=0;n_seq_warn=0

    restart=False
    tm_AU_step=tm_fs_step/aut_in_femtosec
    if( tm_fs_offset is not None ):
        tm_AU_offset = tm_fs_offset/aut_in_femtosec
        ### rttddft.time_AU = tm_AU_offset; 
        assert abs(tm_AU_offset-md.time_AU)<1.0e-4,"tm_AU_offset:%f / MD:%f "%(tm_AU_offset,md.time_AU)
        md.set_time_AU(tm_AU_offset,step=0,rttddft=rttddft)
        restart=True
    else:
        assert ( abs(md.time_AU)<1.0e-6),""
    calc_current=True
    Nstep = int( round( (tm_fs_end - md.time_AU*aut_in_femtosec)/tm_fs_step ) )
    logger.Info("#calc_OAS_propagate:calculating %d steps dt=%f"%(Nstep,tm_fs_step))
    
    logfpath="calc_OAS_propagate.log"
    if( not restart and (MPIrank==0) ):
        fd=futils.fopen(dmfilepath,"w")
        print("#Gauge=%s Kick_strength:%f %f %f tmKick=%f (au) %f (fs)  Nstep=%d"%( 
            ("VG" if(rttddft.velocitygauge) else "LG"),Evector[0],Evector[1],Evector[2],tmKick_AU,tmKick_AU*aut_in_femtosec,Nstep), file=fd)
        futils.fclose(fd)
        os.system("ls -ltrh "+dmfilepath)
    elif( MPIrank==0 ):
        printout("#restart:%f au %f fs Nstep=%d"%(md.time_AU,md.time_AU*aut_in_femtosec,Nstep),fpath=logfpath,Append=True,Threads=[0])
    Dic={"devsum_max":-1, "dagdev_max":-1,"ofddev_max":-1,"Nfix":0};nlog=0
    
    Istep_saved_last=None;dumpfilenumber=1
    dbgng=False;dbgng_dipole=True;dbgng_iter=True
    N_elec=get_Nele( rttddft )
    En_tot_t0=None
    tdFockMat=None
    if(Nstep_calcEng is not None):
        fpath= rttddft_common.get_job(True) + "_tdEng.dat";Edict={'ecoul':None,'ekin':None}
        printout("#%5s %14s %14s    %16s   %16s %16s %16s   %18s %16s %14s   %14s %14s %14s"%(
               "step","time_AU","time_fs","Eel", "ncCoul","elCoul","elKE","enField","En_tot","En_tot_dev",
               "DAGdev","OFDdev","dev_Nele"),fpath=fpath,Threads=[0],Append=False)
        Eel,ncCoul,elCoul,elKE=md.calc_energy_tot(rttddft,dict=Edict)
        DAGdev,OFDdev,Nfix=check_orthnorm(md,rttddft,md.tdMO)
        trDM=trace_dmat(rttddft, md.tdDM, rttddft._spinrestriction, rttddft._pbc)
        
        En_tot=Eel;enField=0
        if( rttddft._calc_Aind != 0 ):
            enField,enFieldTOT=calc_FieldEng_cell(rttddft)
            En_tot=Eel+enField
            print("#En_tot:%e %e %e"%(Eel,enField,En_tot))
        En_tot_t0=En_tot
        printout("%6d %14.6f %14.6f    %16.8f   %16.8f %16.8f %16.8f   %18.6e %16.8f %14.4e   %14.4e %14.4e %14.4e"%(
                0,md.time_AU,md.time_AU*aut_in_femtosec,Eel, ncCoul,elCoul,elKE, enField,En_tot,abs(En_tot-En_tot_t0),
                DAGdev,OFDdev,abs(trDM-N_elec)),fpath=fpath,Append=True,Threads=[0])
        
        tdFockMat=get_FockMat(md, rttddft)
        filenames=print_TDeorbs( md.spinrestriction, md.tdMO, rttddft, md.pbc, md._canonicalMOs,
                   tdFockMat, md.mo_occ, job=rttddft_common.get_job(True), Append=(restart),step=0,tm_au=md.time_AU)
        printout("#calc_OAS_propagate:TDeorbs:"+str(filenames))
    ldosf_nblock=0
    if( Nstep_calcDOS is not None):
        if( tdFockMat is None ):
            tdFockMat=get_FockMat(md, rttddft)
        ldosf_nblock = calc_ldos( rttddft, rttddft_common.get_job(True), md.tdMO, tdFockMat,\
            emin_eV=None, emax_eV=None, de_eV=0.050, Nstep=None, 
            iappend=ldosf_nblock, header="#%d:t=%14.4f"%(ldosf_nblock, md.time_AU*aut_in_femtosec), 
            trailer="\n\n\n", spinrestriction=md.spinrestriction,
            widths_eV=[0.2123, 0.1], gnuplot=True, N_elec=N_elec )
    wt_001=time.time();wt_1=wt_001
    skip_MOprojections=1
    check_tdmatr=False
    Nstep_syncMO=1
    n_buf=0;buf_results=[];logfile=rttddft_common.get_job(True)+"_calcOASpropagate.log";nstep_prtout=20;threads_printlog=[0]
    fd1=open(logfile,'w');fd1.close()

    def print_log(text,Threads=threads_printlog):
        if( MPIrank not in Threads ):
            return
        fd1=open(logfile,'a');print(text,file=fd1);fd1.close()
    def printout_buf(Threads=threads_printlog):
        if( MPIrank not in Threads ):
            return
        fd1=open(logfile,'a');
        for item in buf_results:
            print(str(item),file=fd1);
        fd1.close()

    # printout("#calc_OAS_propagate:df:"+str( type(rttddft.with_df) ),fpath="df.log",Append=True)
    for Istep in range(Nstep):
        if( checkField and (Istep<100 or Istep%200==0) ):
            fd11=open("eldyn_tdfield.log","a");
            A_div_c = rttddft._td_field.get_vectorfield(md.time_AU)/cLIGHT_AU
            print("%16.8f   %16.8f %16.8f %16.8f"%(md.time_AU,A_div_c[0],A_div_c[1],A_div_c[2]), file=fd11);fd11.close()

        if( MPIsize>1 and (Nstep_syncMO>0 and Istep>0 and Istep%Nstep_syncMO==0) ):
            fdDBG=open("eldynDBG_%02d.log"%(MPIrank),"a");print("tdMO:",str(np.shape(md.tdMO)),"step:",Istep,file=fdDBG);fdDBG.close()
            sqrediff,maxabsdiff,At,vals = mpi_aNdiff(md.tdMO,"tdMO_calc_OAS_propagate%04d"%(Istep),sync=True)
            printout("#tdMO: %6d %04d %16.8f %16.6e %10.4f+j%10.4f/%10.4f+j%10.4f at "%(Istep,0,\
                    np.sqrt(sqrediff),maxabsdiff,vals[0].real,vals[0].imag,vals[1].real,vals[1].imag)\
                    +str(At), fnme_format="calc_OAS_sync%02d.log",Append=False)
        if(check_tdmatr):
            name1="tdMO_"+str(rttddft_common.get_job(True))
            fpth1,maxdiff1,indxAT1,sqrdiffsum1,strdiff1 = readwritezN(
                'W',name1,md.tdMO,description="calc_OAS_tdMO %05d %14.4f (fs)"%(Istep,md.time_AU*aut_in_femtosec),
                logfile=name1+"_compTO.log",iverbose=1)
            name2="tdDM_"+str(rttddft_common.get_job(True))
            fpth2,maxdiff2,indxAT2,sqrdiffsum2,strdiff2 = readwritezN(
                'W',name2,md.tdDM,description="calc_OAS_tdDM %05d %14.4f (fs)"%(Istep,md.time_AU*aut_in_femtosec),
                logfile=name2+"_compTO.log",iverbose=1)


        rttddft_common.Dict_setv("rtTDDFT_Istep",Istep)

        avoid_saving=False
        dm_kOR1 = md.calc_tdDM(rttddft) ## this also updates md.tdDM ...
        mol_dip=[]
        current=None
        if(calc_current):
            current=get_current(rttddft,densitymatrix=dm_kOR1)
        dict_dipoles = get_dipole( rttddft, dm_kOR1,"B",filepath=dmfilepath,headline=(not restart and (Istep==0)),
                                   header="%14.6f "%(md.time_AU), current=current, molecular_dipole=mol_dip,caller="eldyn#calc_OAS_propagate")
        if( eldyn_.Dipoles_ini_ is None ):
            eldyn_.Dipoles_ini_=dict_dipoles
        eldyn_.Dipoles_latest_=dict_dipoles

        DIIS_nstep_thr=( eldyn_default_.DIIS_nstep_thr if( dict_getv(params,'DIIS_nstep_thr') is None) else int(params['DIIS_nstep_thr']) )
        dm2diff_TOL=( eldyn_default_.dm2diff_TOL if( dict_getv(params,'dm2diff_TOL') is None) else float(params['dm2diff_TOL']) )
        maxit=eldyn_default_.maxit

        n_fallback=0;poor_cvg=False;nw=0;str_warning=None;cvgd=False;n_iter=-1
        for ith_fallback in range(N_fallbackschemes):
            fallbackscheme=Fallbackschemes[ith_fallback]
            Dict_warnings={};
            n_iter=eldyn_singlestep(md,rttddft, tm2_AU= md.time_AU + tm_AU_step,propagator=propagator, maxit=maxit, \
                                DIIS_nstep_thr=DIIS_nstep_thr, dm2diff_TOL=dm2diff_TOL,Dict_warnings=Dict_warnings,
                                List_fallbackschemes=fallbackscheme)
            
            nw=len(Dict_warnings)
            str_warning=(None if(nw==0) else "#warning%02d: calc_OAS_propagate.%05d.%02d:"%(n_warn+1,Istep,ith_fallback)+str(Dict_warnings))
            if(n_iter>=0):
                cvgd=True; break
            else:
                if(n_buf>0):
                    printout_buf();buf_results.clear();n_buf=0
                print_log("#%6d  %12.6f  %14.4f  %d  %d  "%(Istep,md.time_AU*aut_in_femtosec, md.time_AU, ith_fallback,n_iter)+str_warning)

        buf_results.append(" %6d  %12.6f  %14.4f  %d  %d"%(Istep,md.time_AU*aut_in_femtosec, md.time_AU, ith_fallback, n_iter));n_buf+=1
        if( n_buf >= nstep_prtout ):
            printout_buf();buf_results.clear();n_buf=0
        if( not cvgd ):
            assert False,""
        if(nw>0):
            printout(str_warning,fnme_format='calc_OAS_propagate_%02d_warnings.txt',Append=True,dtme=True,warning=-1);
            if(n_buf>0):
                printout_buf();buf_results.clear();n_buf=0
            print_log(str_warning);
            n_warn+=1;n_seq_warn+=1
            strerr=None
            if( n_seq_warn > nmax_seq_warn ):
                strerr="calc_OAS_propagate.%05d.%02d  n_seq_warn=%d > %d"%(Istep,ith_fallback,n_seq_warn,nmax_seq_warn)
            if( (nmax_warn>=0) and (n_warn > nmax_warn) ):
                strerr="calc_OAS_propagate.%05d.%02d  n_warn=%d > %d"%(Istep,ith_fallback,n_warn,nmax_warn)
            if(strerr is not None):
                printout(strerr,fnme_format='calc_OAS_propagate_%02d_warnings.txt',Append=True,dtme=True,warning=-1);
                assert False,""+strerr
        else:
            if( n_seq_warn > 0 ):
                printout("### step %05d converges:by %d iterations at %d th fallback ###"%(
                       Istep,n_iter,ith_fallback),fnme_format='calc_OAS_propagate_%02d_warnings.txt',Append=True,dtme=True)
            n_seq_warn=0
        wt_2=wt_1;wt_1=time.time();
        if(dbgng_iter): printout("#calc_OAS:%05d  iter:%6d  elapsed:%14.4f %14.4f"%(Istep,n_iter,wt_1-wt_2,wt_1-wt_001))
        ### print("## calc_OAS_propagate:AF-Step%d"%(Istep),flush=True)
        dict={"devsum_max":None, "dagdev_max":None,"ofddev_max":None,"Nfix":None}
        md.normalize_MOcofs(rttddft,MO_Coeffs=md.tdMO, dict=dict)
        Dic["devsum_max"]=max( Dic["devsum_max"],dict["devsum_max"] )
        Dic["dagdev_max"]=max( Dic["dagdev_max"],dict["dagdev_max"] )
        Dic["ofddev_max"]=max( Dic["ofddev_max"],dict["ofddev_max"] )
        Dic["Nfix"]+=dict["Nfix"]
###        nfix,maxdev = check_nrmz(md,rttddft,md.tdMO,norm_dev_tol=1.0e-6)
                
        prtout=( (dict["Nfix"]>0 and nlog<10) or (Istep%200==0) )
        if(prtout):
            printout("step:%d Nfix=%d devsum_max=%e dagdev_max=%e ofddev_max=%e / cum: Nfix=%d devsum_max=%e dagdev_max=%e ofddev_max=%e"%(
                    Istep,dict["Nfix"],dict["devsum_max"],dict["dagdev_max"],dict["ofddev_max"],
                    Dic["Nfix"],Dic["devsum_max"],Dic["dagdev_max"],Dic["ofddev_max"]),
                    fpath=logfpath,Append=True,Threads=[0],flush=(Istep%100==0)); nlog=nlog+1                
        
        md.set_time_AU( md.time_AU+tm_AU_step, md.step+1,rttddft=rttddft)

        if(Nstep_calcEng is not None):
            if((Istep+1)%Nstep_calcEng==0):
                fdDAF=open( rttddft_common.get_job(True) + "_tdEng.dat","a");dict={'ecoul':None,'ekin':None}
                Eel,ncCoul,elCoul,elKE = md.calc_energy_tot(rttddft,dict=dict)
                DAGdev,OFDdev,Nfix=check_orthnorm(md,rttddft,md.tdMO)

                En_tot=Eel;enField=0
                if( rttddft._calc_Aind != 0 ):
                    enField,enFieldSUM=calc_FieldEng_cell(rttddft)
                    En_tot=Eel+enField

                trDM=trace_dmat(rttddft, md.tdDM, rttddft._spinrestriction, rttddft._pbc)
                for fd1 in [fdDAF,sys.stdout]:
                    print("%6d %14.6f %14.6f    %16.8f   %16.8f %16.8f %16.8f   %18.6e %16.8f %14.4e   %14.4e %14.4e %14.4e"%(
                        Istep+1,md.time_AU,md.time_AU*aut_in_femtosec,Eel, ncCoul,elCoul,elKE, enField,En_tot,abs(En_tot-En_tot_t0),
                        DAGdev,OFDdev,abs(trDM-N_elec)),file=fd1)
                tdFockMat=get_FockMat(md, rttddft)
                print_TDeorbs( md.spinrestriction, md.tdMO, rttddft, md.pbc, md._canonicalMOs,
                          tdFockMat, md.mo_occ, job=rttddft_common.get_job(True), Append=True,step=Istep+1,tm_au=md.time_AU)
                fdDAF.close()
                
        if( Nstep_calcDOS is not None):
            if( (Istep+1)%Nstep_calcDOS == 0 ):
                ldosf_nblock = calc_ldos( rttddft, rttddft_common.get_job(True), md.tdMO, tdFockMat, emin_eV=0.0, emax_eV=40.0, de_eV=0.050, Nstep=None, 
                    moldyn=md, iappend=ldosf_nblock, header="#%d:t=%14.4f"%(ldosf_nblock, md.time_AU*aut_in_femtosec), 
                    trailer="\n\n\n", spinrestriction=md.spinrestriction,
                    widths_eV=[0.2123, 0.1], gnuplot=False )
        #if( check_timing ):
        #    if( (Istep+1)==1 or (Istep+1)==10 or (Istep+1)==20 or (Istep+1)%50==0 ):
        #        if(isinstance(rttddft,rttddftPBC_timing)):
        #            rttddft.print_log( "#calc_OAS:step=%05d"%(Istep+1), timing=True)
        #            rttddft.print_timerecord( header="### calc_OAS:step=%05d"%(Istep+1) )
        if( (Istep+1)==1 or (Istep+1)==10 or (Istep+1)==50 ):
            WCtOLD=WCtCUR;WCtCUR=time.time();perstep_tsec=(WCtCUR-WCt01)/float(Istep+1)
            if(Nstep_no_save is None):
                Nstep_no_save=max(1, int(round(tsecond_no_save/perstep_tsec)))
                if(Nstep_dump is not None):
                    Nstep_no_save = min( Nstep_no_save, Nstep_dump )
            if(verbose>0):
                printout("#calc_OAS_propagate:elapsed %f / %d steps avg %f total %f"%( WCtCUR-WCt01,(Istep+1),perstep_tsec, 
                WCtCUR-WCt01 if(WCt00 is None) else WCtCUR-WCt00) )
        dbgng=False
        if( (Nstep_no_save is not None) and (Istep_saved_last is not None) ):
            avoid_saving=(Istep < (Istep_saved_last + Nstep_no_save) )
        
##        if( MPIsize<2 or MPIrank==0 ):
        save_here=False
        Force_save=False
        for fno in range(fno_incf+1,5):
            dum="eldyn_save_%02d.inc"%(fno_incf);
            if(os.path.exists(dum)):
                Force_save=True;fno_incf=fno;print("saving eldyn:"+dum)   
        if( (not avoid_saving) and ( (tsecond_dump is not None) or Force_save )):
            WCtCUR=time.time();
            if( ( (WCtCUR-WCt00) + perstep_tsec*0.5 > tsecond_dump ) or Force_save ):
                save_here=True
        if( (not save_here) and (not avoid_saving) and (Nstep_dump is not None) ):
            if( (Istep+1)%Nstep_dump == 0 ):
                save_here=True
        if( save_here ):
            dumpfilenumber=1-dumpfilenumber;
            md.save(fpath=dumpfilename+"%02d"%(dumpfilenumber)+".pscf",delimiter='\n',
              comment="step:%d time=%f(au) of "%(Istep+1,md.time_AU) + rttddft_common.get_job(True), rttddft=rttddft,
              caller="eldyn.L1152,calc_OAS_propagate, step%06d"%(Istep+1), Barrier=True);
            Istep_saved_last=Istep;avoid_saving=True
            if(dbgng):
                test_reload(md,dumpfilename+"%02d"%(dumpfilenumber)+".pscf")
                assert False,""

def eldyn_singlestep(md,rttddft,tm2_AU, mo_coeff_IN=None, 
        h1e_kOR1=None, DM1_kOR1=None, tm1_AU=None, 
        DM2_kOR1=None, maxit=eldyn_default_.maxit, Nstep_CN=20,
        DIIS_bufsz=eldyn_default_.DIIS_bufsz, DIIS_ervec_thr=eldyn_default_.DIIS_ervec_thr,
        DIIS_nstep_thr=eldyn_default_.DIIS_nstep_thr,
        dm2_mixing_factor=0.50,DIISscheme={"updated":'D',"errvecs":'D'},
        DIIS_checkbuf_vnorm_thr=1.0e-5, DIIS_checkbuf_vnorm_relt_thr=1.0e-2,DIIS_checkbuf_nSleep=6,
        dm2diff_TOL=eldyn_default_.dm2diff_TOL, F2diff_TOL=5.0e-7, difflog = None, append_difflog=False, 
        update_attribute=True, Dict_warnings=None, List_fallbackschemes=None,
        logger=None, logger2=None,Vnuc_au_1D=None, Nacm_rep=None, fallback=0, force_update=False,
        step="", params_prtNacm=None, verbose=False, details=None, append_iterationlog=False, 
        Niter_clear_buffer=None,Niter_check_buffer=None,diffthr_clear_buffer=None,
        propagator="CN", synchronize=True, abort_if_failed=True, reuse_DIIS=False ):

    #> MPI
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
                   # check_sync:   3:        Fock1
    check_sync=1   # check_sync:   2:        h1e,DM1,DM2,h2e(:)
                   # check_sync:   1:normal  md.hcore_last, md.Fock_last
                   # check_sync:   0:minimal 
    syncthr=2;niter_syncthr=4;


    #> PhysicalConstants : we only need aut_in_femtosec.
    aut_in_femtosec=PhysicalConstants.aut_in_femtosec()

    #> fncdepth, walltime etc. for logging
    depth=1;fncnme="eldyn_singlestep";fncdepth=1   ## fncdepth ~ SCF:0 vhf:1 get_j:2
    wt_010=rttddft_common.Get_time();wct_01=time.time()
    Wctm000=wct_01;Wctm010=Wctm000;N_call=rttddft_common.Countup("eldyn_singlestep");dic1_timing={}
    Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)
    if( Dic_timing is None ):
        rttddft_common.Dict_setv('timing_'+fncnme, {});
        Dic_timing=rttddft_common.Dict_getv('timing_'+fncnme, default=None)


    #> propagator, either CN or Magnus
    assert (propagator=="CN" or propagator=="Magnus"),"wrong propagator:"+propagator
    use_Magnus=(propagator=="Magnus")
    pbc = md.pbc


    #> Fallback schemes
    diffthr_keep_best=-1;diff_best=None;MO_best=None  ## by default, (dm2diff < diffthr_keep_best) never occurs
    if(List_fallbackschemes is not None):
        for fbs in List_fallbackschemes:
            if( fbs['type']=='best'):
                assert (Dict_warnings is not None),""
                absthr_default=None; reltthr_default=10
                absthr=dict_getv(fbs,'absthr',None)
                if( absthr is not None ):
                    diffthr_keep_best=absthr
                else:
                    reltthr=dict_getv(fbs,'reltthr',reltthr_default)
                    diffthr_keep_best=dm2diff_TOL*reltthr

    #> DIIS param : Niter_clear_buffer = How frequently you clear DIIS buffer
    #>                  Note, usually you reach convergence in less than 10 iterations.
    if(Niter_clear_buffer is None):
        Niter_clear_buffer=70; diffthr_clear_buffer=10*dm2diff_TOL;Nskip_clear_buffer=20;Navoid_clrbuf=70;
        if( fallback> 0 ):
            Niter_clear_buffer=400;diffthr_clear_buffer=20*dm2diff_TOL;  ## Niter_check_buffer=200;

    #assert ( RTTDDFT_.is_rttddft(rttddft) ),"rttddft"
    assert ( rttddft._fix_occ ),"set fix_occ first"
    assert ( tm2_AU is not None),""
    if mo_coeff_IN is None: mo_coeff_IN = md.tdMO
    if tm1_AU is None: tm1_AU= md.time_AU;
    if DM1_kOR1 is None: DM1_kOR1 = md.calc_tdDM(rttddft )
    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"setup2_dm1",Wctm010-Wctm020,depth=fncdepth)  

    #> A. Calculate h1e (hcore)
    # 
    if( h1e_kOR1 is None ):
        if( md.hcore_last is None ):
            h1e_kOR1=md.calc_hcore(rttddft, time_AU=tm1_AU, dm1_kOR1=DM1_kOR1, tmAU_dmat=tm1_AU)
            Wctm020=Wctm010;Wctm010=time.time()
            update_dict(fncnme,dic1_timing,Dic_timing,"setup1_h1eA",Wctm010-Wctm020,depth=fncdepth)  
        else:
            h1e_kOR1=arrayclone(md.hcore_last); ## aNcpy(md.hcore_last,h1e_kOR1)
            
            assert abs(md.tmAU_hcore_last-tm1_AU)<1.0e-6,"%f / %f :%e"%(md.tmAU_hcore_last,\
                tm1_AU, abs(md.tmAU_hcore_last-tm1_AU))
            Wctm020=Wctm010;Wctm010=time.time()
            update_dict(fncnme,dic1_timing,Dic_timing,"setup1_h1eB",Wctm010-Wctm020,depth=fncdepth)  

    #> B. Create DM2_kOR1 
    #>     note that DM1_kOR1: density matr at t=t_1,            fixed
    #>               DM2_kOR1: density matr at t=t_1+\Delta t    updated at each iteration
    if DM2_kOR1 is None: DM2_kOR1 = arrayclone(DM1_kOR1)
    if logger is None: logger=md.get_logger(True)
    SIMPLE_DOT_PRODUCT_=0; AO_DOT_PRODUCT_=1; ORTHNORM_AOMATRICES_=2

    if( Dict_warnings is not None ):
        Dict_warnings.clear()
    nmod_check_nrmz=1
    
    description="#eldyn_singlestep:tm1=%f (%f fs) dt=%f (%f fs) Nstep_CN=%d eps_CN=%f "%(\
        tm1_AU, tm1_AU*aut_in_femtosec, (tm2_AU-tm1_AU), (tm2_AU-tm1_AU)*aut_in_femtosec, Nstep_CN,\
        (tm2_AU-tm1_AU)/float(Nstep_CN))
    I_step=int(round(tm1_AU/(tm2_AU-tm1_AU)))

    loglvl=1; Nmod_iterationlogf=20;
    iteration_logfpath="eldyn_singlestep_%03d.log"%(I_step%Nmod_iterationlogf);
    threads_iterationlog=[0]
    
    devs=[ [], [] ];devs_bfsz=50;diis_status=[];iter_strbuf=[]; Niter_printed=0

    printout("## step:%03d: %14.4f au %14.4f fs "%(I_step,tm1_AU,tm1_AU*aut_in_femtosec),
            fpath=iteration_logfpath,Append=False,Threads=threads_iterationlog)
    
    def prtout_iteration(iter_unitoffset,d2buf,ibuf,sbuf,bfsz,cvgd,wct_INI,niter_printed):
        strresult=None
        if(cvgd is not None):
            wct_END=time.time();
            if(cvgd):
                strresult=" converged         N_iter:%5d    walltime: %14.4f"%(iter_unitoffset,(wct_END-wct_INI))
            else:
                strresult=" DID NOT converge  N_iter:%5d    walltime: %14.4f"%(iter_unitoffset,(wct_END-wct_INI))
       
        if(niter_printed==0):
            printout("#%5s  %14s  %14s  %4s"%("iter","dm2Diff","FckDiff","DIIS"),
                    fpath=iteration_logfpath,Append=True,Threads=threads_iterationlog)
        for it_0offset in range(niter_printed,iter_unitoffset):  ## @20th(019) 0,20 > @50th(049) 20,50 
            imod=it_0offset%bfsz
            printout(" %5d  %14.4e  %14.4e  %4d  "%(it_0offset,d2buf[0][imod],d2buf[1][imod],ibuf[imod])+("" if(sbuf is None) else sbuf[imod]),
                    fpath=iteration_logfpath,Append=True,Threads=threads_iterationlog)
        if( len(d2buf[0])>=bfsz ):
            d2buf[0].clear();d2buf[1].clear();ibuf.clear();
        niter_printed=iter_unitoffset
        return niter_printed
    def append_log(text,stdout=True):
        printout(text,fpath=iteration_logfpath,Append=True,Threads=threads_iterationlog,stdout=stdout);

    params={"tm1_AU":tm1_AU,"tm2_AU":tm2_AU,"Nstep_CN":Nstep_CN};
    converger={"DIIS_bufsz":DIIS_bufsz,"DIIS_ervec_thr":DIIS_ervec_thr,"DIIS_nstep_thr":DIIS_nstep_thr,"dm2_mixing_factor":dm2_mixing_factor};
    description+="\n##CONVERGER:"+str(converger);         params.update(converger);
    thresholds={"maxit":maxit,"dm2diff_TOL":dm2diff_TOL};
    description+="\n##THRESHOLDS:"+str(thresholds);       params.update(thresholds);
    misc={"update_attribute":update_attribute};
    description+="\n##misc:"+str(misc);                   params.update(misc)
    logger.timing("singlestep_%d"%(md.step),start=True)

    Ndim_DM2_kOR1 = np.shape( DM2_kOR1 )
    rank_DM2_kOR1 = len( Ndim_DM2_kOR1 )
    nAO=md.nAO;nMO=md.nMO
    nkpt=(1 if (not md.pbc) else md.nkpt) 
    kvectors=( None if(not md.pbc) else np.reshape( rttddft.kpts, (-1,3)) )
    Ndim_Fock = [2,md.nkpt,md.nAO,md.nAO] if( md.pbc and md.spinrestriction=='U') else (\
                [2,md.nAO,md.nAO] if(md.spinrestriction=='U') else (\
                [md.nkpt,md.nAO,md.nAO] if(md.pbc) else [md.nAO, md.nAO]))  # assume mol:[nAO][nAO] / pbc:[nkpt][nAO][nAO]...
    Ndim_hcore =([md.nkpt,md.nAO,md.nAO] if(md.pbc) else [md.nAO, md.nAO])
    dtype_hcore=np.float64
    rank_Fock=len(Ndim_Fock)
    Ndim_MOcoeff=np.shape( mo_coeff_IN );
    rank_MOcoeff=len(Ndim_MOcoeff)

    #> set S^{-1}, S^{-1/2}, S^{1/2} 
    rttddft.update_Sinv()
    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"setup3_Sinv",Wctm010-Wctm020,depth=fncdepth)  
    
    #> spin multiplicity
    nmult_Fock=(2 if (md.spinrestriction=='U') else 1)
    nmult_DM  =(1 if (md.spinrestriction=='R') else 2)
    nmult_MO = nmult_Fock

    AOmatr_innerproduct=AO_DOT_PRODUCT_ ### SIMPLE_DOT_PRODUCT_
    
    diis=None; ## nmin_DIIS=0
    if(DIIS_bufsz>0):
        if( reuse_DIIS ):
            if(md._DIIS is None ):
                md._DIIS=DIIS01(Ndim_DM2_kOR1,DIIS_bufsz,dtype=np.complex128,ervec_thr=DIIS_ervec_thr,nstep_thr=DIIS_nstep_thr,mixing_factor=dm2_mixing_factor)
            diis=md._DIIS
        else:
            diis=DIIS01(Ndim_DM2_kOR1,DIIS_bufsz,dtype=np.complex128,ervec_thr=DIIS_ervec_thr,nstep_thr=DIIS_nstep_thr,mixing_factor=dm2_mixing_factor)
    SAO=None
    if( AOmatr_innerproduct == AO_DOT_PRODUCT_ ):
        kvecs=(None if(not rttddft._pbc) else np.reshape(rttddft.kpts, (-1,3)))
        nkpts=(   1 if(not rttddft._pbc) else len(kvecs) )
        SAO=( rttddft.get_ovlp() if(not rttddft._pbc) else \
              [ rttddft.get_ovlp(rttddft.cell,kvecs[k]) for k in range(nkpts) ])
        if( diis is not None ):
            diis.update_AOparams({'AOrep':True,'SAO':SAO,'spinrestriction':rttddft._spinrestriction,'DorF':'D'})
    orthnorm_ervecs=( AOmatr_innerproduct == ORTHNORM_AOMATRICES_ )
    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"setup4_DIIS",Wctm010-Wctm020,depth=fncdepth)  
    
    #> Choice of DIIS vectors -- in principle you can use either Density_matrix or Fock_matrix for error vec and updated vecs
    #> 
    #> updated vecs : {u_j}
    #> error vecs   : {e_j}
    #> 
    #>  next vector = \sum_j  u_j c_j    
    #>                                                           
    #>  0   (e_1|e_1)  (e_1|e_2) ...  (e_1|e_N)  1    c_1
    #>  0                                             c_2
    #>  0 =                                         x c_3
    #>  ..                                            ...
    #>  1   1          1         ...  1          0    -\lambda
    diis_updated = 0; diis_errvecs = 0
    FOCK_=2; DMAT_=1; ## constant bit flags 
    diis_ervec_mixing=None; diis_oldvec_mixing=None
    if( (diis is not None) and (DIISscheme is not None) ):
        diis_updated=( FOCK_ if( DIISscheme["updated"]=='F') else ( DMAT_ if( DIISscheme["updated"]=='D') else -1))
        diis_errvecs=( FOCK_ if( DIISscheme["errvecs"]=='F') else ( DMAT_ if( DIISscheme["errvecs"]=='D') else -1))
        if( diis_updated == diis_errvecs and (AOmatr_innerproduct != ORTHNORM_AOMATRICES_) ):
            diis_ervec_mixing=dm2_mixing_factor; diis_oldvec_mixing=None
        else:
            diis_ervec_mixing=None; diis_oldvec_mixing=dm2_mixing_factor
        assert (diis_updated>0 and diis_errvecs>0),""
        description+="\n##DIISscheme:"+str( DIISscheme )
    V_eff=None

    logger.only_once("eldyn_singlestep",description);
    idbgng=0
    ## MATRICES ---
    ### md.Fock_last=None  20210616 commented out
    if( md.Fock_last is not None ):
        if( abs(md.tmAU_Fock_last-tm1_AU)>1.0e-6 ):
            printout("Fock1A:tm1_AU %14.4f / tmAU_Fock_last %14.4f"%(tm1_AU,md.tmAU_Fock_last),
                fpath="eldyn_FockLast.log",Append=True,Threads=[0]);
            md.Fock_last=None

    ## Fock1:  allocation=ALWAYS  calc=IFF md.Fock is None
    Fock1_kOR1_eff = np.zeros( Ndim_Fock, dtype=np.complex128 )
    if( md.Fock_last is None ):
        
        Fock1_kOR1_eff = rttddft.get_fock( h1e=h1e_kOR1,dm=DM1_kOR1,vhf=md.vhf_last ) ## cycle =-1(default) and diis=None 
        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1_timing,Dic_timing,"setup5_Fock1A",Wctm010-Wctm020,depth=fncdepth)  
    else:
        Fock1_kOR1_eff = aNcpy( md.Fock_last, Fock1_kOR1_eff )
        assert abs(md.tmAU_Fock_last-tm1_AU)<1.0e-6,""
        Wctm020=Wctm010;Wctm010=time.time()
        update_dict(fncnme,dic1_timing,Dic_timing,"setup5_Fock1B",Wctm010-Wctm020,depth=fncdepth)  

    #> calculate  hcore at t=tm2_AU and put it in  md.hcore_last
    #>
    if( rttddft._calc_Aind >= 0 ):
        md.hcore_last=md.calc_hcore(rttddft, time_AU=tm2_AU, dm1_kOR1=DM1_kOR1, tmAU_dmat=tm1_AU, Dict_hc=None)
    else:
        md.hcore_last=md.calc_hcore(rttddft, time_AU=tm2_AU, dm1_kOR1=DM2_kOR1, tmAU_dmat=tm2_AU, Dict_hc=None)

    md.tmAU_hcore_last=tm2_AU; printout("#tmAU_hcore_last:%f"%(md.tmAU_hcore_last),fpath="moldyn_upd.log",Append=True,dtme=True)
    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"setup6_hcore2",Wctm010-Wctm020,depth=fncdepth)  

    #> calculate  vhf at t=tm2_AU and put it in  md.vhf_last
    #>           Fock at t=tm2_AU and put it in  md.Fock_last

    mol_or_cell=( rttddft.mol if(not rttddft._pbc) else rttddft.cell) ## 2021.07.12  cell->mol_or_cell
    md.vhf_last = rttddft.get_veff( mol_or_cell, dm=DM2_kOR1 )        ## 2021.07.12  cell->mol_or_cell
    md.Fock_last= rttddft.get_fock( h1e=md.hcore_last,dm=DM2_kOR1,vhf=md.vhf_last)

    md.tmAU_Fock_last=tm2_AU; md.tmAU_vhf_last=tm2_AU
    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"setup7_Fock2",Wctm010-Wctm020,depth=fncdepth)  

    if( MPIsize>1 ):
        string=""
        if( check_sync > 0 ):
            sqrediff0,maxabsdiff0,At0,vals0 = mpi_aNdiff(DM2_kOR1,"DM2_eldyn%04d_PREP"%(I_step),sync=True)
            
        if( check_sync > 1 ):
            sqrediff1,maxabsdiff1,At1,vals1 = mpi_aNdiff(h1e_kOR1,"h1e_eldyn%04d_PREP"%(I_step),sync=True)
            sqrediff2,maxabsdiff2,At2,vals2 = mpi_aNdiff(DM1_kOR1,"DM1_eldyn%04d_PREP"%(I_step),sync=True)
            sqrediff3,maxabsdiff3,At3,vals3 = mpi_aNdiff(md.hcore_last,"h2e_eldyn%04d_PREP"%(I_step),sync=True)
            if( I_step < 3  or I_step==20 ):
                string+= "#DM2: %6d %4d "%(I_step,0) + mpidiffs_tostring_(sqrediff0,maxabsdiff0,At0,vals0) \
                    +"#h1e: %6d %4d "%(I_step,0) + mpidiffs_tostring_(sqrediff1,maxabsdiff1,At1,vals1) \
                    +"#DM1: %6d %4d "%(I_step,0) + mpidiffs_tostring_(sqrediff2,maxabsdiff2,At2,vals2)
        if( check_sync > 2):
            sqrediff5,maxabsdiff5,At5,vals5 = mpi_aNdiff(Fock1_kOR1_eff,"Fock1_eldyn%04d_PREP"%(I_step),sync=True)
            sqrediff6,maxabsdiff6,At6,vals6 = mpi_aNdiff(md.Fock_last,"Fock2_eldyn%04d_PREP"%(I_step),sync=True)
            if( I_step < 3  or I_step==20 ):
                string+= "#Fock1: %6d %4d "%(I_step,0) + mpidiffs_tostring_(sqrediff5,maxabsdiff5,At5,vals5) \
                        +"#Fock2: %6d %4d "%(I_step,0) + mpidiffs_tostring_(sqrediff6,maxabsdiff6,At6,vals6)
        if( (I_step < 3  or I_step==20) and len(string)> 0 ):
            printout(string, fnme_format="calc_OAS_sync%02d.log",Append=True)

    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"setup8_sync",Wctm010-Wctm020,depth=fncdepth)  

    MOcoeff_kOR1 = arrayclone(mo_coeff_IN) ### .copy()

    #> Fock, DM2_kOR1  at the previous iteration step is stored in Fock2_old, DM2_kOR1_old
    Fock2_old  = np.zeros( Ndim_Fock, dtype=np.complex128)
    DM2_kOR1_old = arrayclone( DM2_kOR1 ) ### .copy()
    MOcoeff_kOR1org = arrayclone(mo_coeff_IN) ### .copy()

    ## (i) Fock1_kOR1_eff contains possible Nacm
    ## (ii) md.Fock_last does NOT contain Nacm.. it is to be treated separately
    
    if( Vnuc_au_1D is not None ):
        ## We simply assume Vnuc does not change over a short period...
        Vdot_nacm_eff = calc_Vdot_nacm_eff(rttddft,Vnuc_au_1D, Nacm_rep=Nacm_rep, check_antihermicity=True)
        if( params_prtNacm is not None):
            Istep=params_prtNacm["Istep"];job=params_prtNacm["job"];logger=params_prtNacm["logger"]
            prtNacm( pbc,md.spinrestriction, Vdot_nacm_eff, MOrep=True, refr_MO=md.tdMO, Vnuc_1D=Vnuc_au_1D, job=job, Istep=Istep, tm_au=rttddft._time_AU,\
                     logger=md.get_logger(True) )
        Fock1_kOR1_eff = to_complexarray( Fock1_kOR1_eff )
        Ndim_Fock1=np.shape(Fock1_kOR1_eff)
        if( not pbc ):    ## h1e[nAO][nAO] -= i\hbar Vdot_nacm_eff[nAO][nAO].. 
            for i in range(Ndim_Fock1[0]):
                for j in range(Ndim_Fock1[1]):
                    Fock1_kOR1_eff[i][j]-= 1j*Vdot_nacm_eff[i][j]
        else:
            for i in range(Ndim_Fock1[0]):
                for j in range(Ndim_Fock1[1]):
                    for k in range(Ndim_Fock1[2]):
                        Fock1_kOR1_eff[i][j][k]-= 1j*Vdot_nacm_eff[i][j][k]

    cvgd=False; iter=0; dm2diff=-1.0
    writer=None
    iter_startdiis=-1;iter_clrbuf=-1
    if( diis is not None):
        diis.writer=writer    
    dict_DIIS_coefs=None
    diis_started=False
    diff_min=None;dmOpt=None
    DM2_indices=[]
    if( details is not None and ("dmOpt" in details) ):
        if( details["dmOpt"] is None ):
            details["dmOpt"]=np.zeros( Ndim_DM2_kOR1, dtype=np.complex128)
        diff_min=details["diff_min"];dmOpt=details["dmOpt"]

    if(True):
        Dic1={"devsum_max":None, "dagdev_max":None,"ofddev_max":None,"Nfix":None}
        md.normalize_MOcofs(rttddft,MO_Coeffs=md.tdMO, dict=Dic1)
        assertf(Dic1['Nfix']==0,"normalize_MOcofs retv:"+str(Dic1),1)
        iter_strbuf.append(str(Dic1))
    
    dm2diff_min=9.9e+20; dm2diff_at=-1

    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"setup9_misc",Wctm010-Wctm020,depth=fncdepth)  

    Wctm_ITER00=time.time()
    while( not cvgd and iter<maxit):
        if( iter>=1 and iter<=10 ):
            Wctm020=Wctm010;Wctm010=time.time()
            update_dict(fncnme,dic1_timing,Dic_timing,"iter%02d"%(iter),Wctm010-Wctm020,depth=fncdepth)  

        ## all quantities are derived from
        ## (i)  md.hcore_last(h2e), h1e_kOR1, DM1_kOR1, Fock1_kOR1_eff  ---- already been synchronized
        ## (ii) DM2_kOR1    note F2 or md.Fock_last is calculated from h2e and DM2 ...

        wt_020=rttddft_common.Get_time()
        if(iter==1 or iter==10 or iter%50==0 ):
            logger.timing("iteration_%d"%(iter))

        iter+=1 ## iter=1,2,3...,maxit
        
        #> 1. MOcoeff_kOR1 <-- MOcoeff_kOR1org
        assert rank_MOcoeff<=4,""

        for I in range(Ndim_MOcoeff[0]):
            for J in range(Ndim_MOcoeff[1]):
                if(rank_MOcoeff==2):
                     MOcoeff_kOR1[I][J]=MOcoeff_kOR1org[I][J]
                else:
                    for K in range(Ndim_MOcoeff[2]):
                        if( rank_MOcoeff==3):
                            MOcoeff_kOR1[I][J][K]=MOcoeff_kOR1org[I][J][K]
                        else:
                            for L in range(Ndim_MOcoeff[3]):
                                MOcoeff_kOR1[I][J][K][L]=MOcoeff_kOR1org[I][J][K][L]

        #> 2. Fock2_old <-- Fock2_kOR1
        assert rank_Fock<=4,""
        for I in range(Ndim_Fock[0]):
            for J in range(Ndim_Fock[1]):
                if(rank_Fock==2):
                     Fock2_old[I][J]=md.Fock_last[I][J]
                else:
                    for K in range(Ndim_Fock[2]):
                        if( rank_Fock==3):
                            Fock2_old[I][J][K]=md.Fock_last[I][J][K]
                        else:
                            for L in range(Ndim_Fock[3]):
                                Fock2_old[I][J][K][L]=md.Fock_last[I][J][K][L]

        #> 3. construct Fock2
        if( iter>1 and (rttddft._calc_Aind < 0) ):              ## < calc_Aind >
            md.hcore_last=md.calc_hcore(rttddft, time_AU=tm2_AU, dm1_kOR1=DM2_kOR1, tmAU_dmat=tm2_AU) 

        if( (V_eff is None) or ((diis_updated & FOCK_) == 0) ):
            
            mol_or_cell=( rttddft.mol if(not rttddft._pbc) else rttddft.cell) ## 2021.07.12  cell->mol_or_cell
            md.vhf_last = rttddft.get_veff( mol_or_cell, dm=DM2_kOR1 )        ## 2021.07.12  cell->mol_or_cell
            md.Fock_last = rttddft.get_fock( h1e=md.hcore_last, dm=DM2_kOR1,vhf=md.vhf_last )
            
            
        sqrediff_IN1=-1;maxabsdiff_IN1=-1;At_IN1=None;vals_IN1=None
        #> 4. propagate MOcoeff ...
        for sp in range(nmult_MO):
            mo_coeff = ( MOcoeff_kOR1 if(nmult_MO==1) else MOcoeff_kOR1[sp])
            f2_kOR1  = ( md.Fock_last if(nmult_MO==1) else md.Fock_last[sp]) 
            f1_kOR1  = ( Fock1_kOR1_eff if(nmult_MO==1) else Fock1_kOR1_eff[sp])     ## 2021.12.01 : it now contains possible Nacm term

            # Note : f2_kOR1 originally points to (a part of) md.Fock_last            [ i.e. possible modifications affect md.Focklast ]
            #        but if( Vnuc_au_1D is not None ) it is cloned into a local copy  [ i.e. modifications do not affect md.Focklast ] 
            if( Vnuc_au_1D is not None ):
                f2_kOR1 = to_complexarray(f2_kOR1,do_clone=True); Nd_f2=np.shape(f2_kOR1)
                assert ( (pbc and len(Nd_f2)==3) or ((not pbc) and len(Nd_f2)==2) ),""
                if( pbc ):
                    for i in range(Nd_f2[0]):
                        for j in range(Nd_f2[1]):
                            for k in range(Nd_f2[2]):
                                f2_kOR1[i][j][k] -= 1j*Vdot_nacm_eff[i][j][k]
                else:
                    for i in range(Nd_f2[0]):
                        for j in range(Nd_f2[1]):
                            f2_kOR1[i][j] -= 1j*Vdot_nacm_eff[i][j]

            #print("eldyn_singlestep:",pbc)
            if( not pbc ):
                ## it seems that the LHS is newly generated here...
                if(use_Magnus):
                    mo_coeff = Magnus_AOrep(rttddft._Ssqrt,rttddft._Sinvrt,nAO,f2_kOR1,f1_kOR1,tm2_AU-tm1_AU,mo_coeff)
                else:
                    mo_coeff = CNlnr_AOrep (rttddft._Ssqrt,rttddft._Sinvrt,f2_kOR1,f1_kOR1,tm2_AU-tm1_AU,Nstep_CN,mo_coeff,
                                        check_vecnorms=(md.step%200==1),rttddft=rttddft,moldyn=md)
                if(nmult_MO==1):
                    MOcoeff_kOR1=mo_coeff
                else:
                    MOcoeff_kOR1[sp]=mo_coeff
            else:
                if(use_Magnus):
                    for k in range(nkpt):
                        assert False,"#2021.08.03 XXX XXX I think this should be : mo_coeff[k]=Magnus_AOrep(...)  PLS fix and TEST it..."
                        mo_coeff = Magnus_AOrep(rttddft._Ssqrt[k],rttddft._Sinvrt[k],nAO,f2_kOR1[k],f1_kOR1[k],tm2_AU-tm1_AU,mo_coeff[k])
                else:
                    #print("eldyn_singlestep:",nkpt,np.shape(rttddft._Ssqrt[0]))
                    for k in range(nkpt):
                        mo_coeff[k] = CNlnr_AOrep (rttddft._Ssqrt[k],rttddft._Sinvrt[k],f2_kOR1[k],f1_kOR1[k],tm2_AU-tm1_AU,Nstep_CN,mo_coeff[k],
                                           kp=k, check_vecnorms=(md.step%200==1),rttddft=rttddft,moldyn=md)
                if(nmult_MO==1):
                    MOcoeff_kOR1[k]=mo_coeff[k]
                else:
                    MOcoeff_kOR1[sp][k]=mo_coeff[k]

        strchecknrmz=""
        if( nmod_check_nrmz>0 ):
            if( iter%nmod_check_nrmz==0 ):
                Dic1={"devsum_max":None, "dagdev_max":None,"ofddev_max":None,"Nfix":None}
                md.normalize_MOcofs(rttddft,MO_Coeffs=MOcoeff_kOR1, dict=Dic1)
                assertf(Dic1['Nfix']==0,"normalize_MOcofs %04d.%03d retv:"%(I_step,iter)+str(Dic1),1)
                strchecknrmz= str(Dic1)
        iter_strbuf.append(strchecknrmz)
    
        #> 5. DM2_kOR1_old <- DM2_kOR1
        for I in range(Ndim_DM2_kOR1[0]):
            for J in range(Ndim_DM2_kOR1[1]):
                if(rank_DM2_kOR1==2):
                     DM2_kOR1_old[I][J]=DM2_kOR1[I][J]
                else:
                    for K in range(Ndim_DM2_kOR1[2]):
                        if( rank_DM2_kOR1==3):
                            DM2_kOR1_old[I][J][K]=DM2_kOR1[I][J][K]
                        else:
                            for L in range(Ndim_DM2_kOR1[3]):
                                DM2_kOR1_old[I][J][K][L]=DM2_kOR1[I][J][K][L]

        
        #> 6. new DM2 from MOcoeff_kOR1
        DM2_kOR1 = rttddft.make_rdm1( MOcoeff_kOR1, md.mo_occ )  ## 2020.11.28 make DM out of tdMO

        fckdiff=( -1.0 if(Fock2_old is None) else aNmaxdiff(Fock2_old,md.Fock_last) )
        
        vcdiff1D=np.ravel( DM2_kOR1_old-DM2_kOR1 );
        if( AOmatr_innerproduct == AO_DOT_PRODUCT_ ):
            assert SAO is not None,""
            dm2diff=dot_AOmatrices( vcdiff1D, vcdiff1D, SAO, rttddft._spinrestriction,'D')
        else:
            dm2diff=np.vdot( vcdiff1D, vcdiff1D )
        assert abs(dm2diff.imag)<1.0e-7 and dm2diff.real>=0.0,"dm2diff:"+str(dm2diff)
        dm2diff=np.sqrt( abs(dm2diff.real) )

        if( MPIsize>1 and (niter_syncthr > 0) and (iter)%niter_syncthr==0 ):
            ## NOTE: here we do synchronize threads  -----
            sqrediff_MOcofs,maxabsdiff_MOcofs,At_MOcofs,vals_MOcofs = mpi_aNdiff(MOcoeff_kOR1,"MOcoeff_kOR1_eldyn%03d.%03d"%(I_step,iter),sync=True)
            sqrediff_DM2,maxabsdiff_DM2,At_DM2,vals_DM2 = mpi_aNdiff(DM2_kOR1,"DM2_kOR1_eldyn%03d.%03d"%(I_step,iter),sync=True)
            string=" %6d  %4d       %14.6e       %14.6e       %10.4f %10.4f   %10.4f %10.4f       %10.4f %10.4f   %10.4f %10.4f"%(\
                    I_step,iter,maxabsdiff_MOcofs,maxabsdiff_DM2, 
                    vals_MOcofs[0].real,vals_MOcofs[0].imag,  vals_MOcofs[1].real,vals_MOcofs[1].imag,
                    vals_DM2[0].real,vals_DM2[0].imag,  vals_DM2[1].real,vals_DM2[1].imag)
            legend="#%6s  %4s       %14s       %14s       %21s   %21s       %21s   %21s"%(\
                    "Istep","iter","maxabsdiff_MO","maxabsdiff_DM","MO_loc","MO_ref","DM2_loc","DM2_ref")
            fd1=open("eldyn_syncthreads_%02d.log"%(MPIrank),'a');
            print(string,file=fd1);
            fd1.close()

        if(loglvl>=3):
            print_dm2(DM2_kOR1_old,DM2_kOR1,DM2_indices,iter,filepath=iteration_logfpath)

        devs[0].append(dm2diff);devs[1].append(fckdiff);diis_status.append(diis_started)

        if( writer is not None ):
            writer.append("%d %e %e %d"%(iter,dm2diff,fckdiff,(1 if(diis_started) else 0)))
        if( iter==20 ):
            Niter_printed = prtout_iteration(iter,devs,diis_status,iter_strbuf,devs_bfsz,None,wct_01,Niter_printed)
        elif( iter>=50 and iter%50==0 ):
            Niter_printed = prtout_iteration(iter,devs,diis_status,iter_strbuf,devs_bfsz,None,wct_01,Niter_printed)

        ## ----------------------------------------------------------
        if( (diff_min is None) or diff_min> dm2diff):
            diff_min=dm2diff; 
            if( details is not None and ("dmOpt" in details) ):
                #7. details["dmOpt"] <- DM2_kOR1
                for I in range(Ndim_DM2_kOR1[0]):
                    for J in range(Ndim_DM2_kOR1[1]):
                        if(rank_DM2_kOR1==2):
                             details["dmOpt"][I][J]=DM2_kOR1[I][J]
                        else:
                            for K in range(Ndim_DM2_kOR1[2]):
                                if( rank_DM2_kOR1==3):
                                    details["dmOpt"][I][J][K]=DM2_kOR1[I][J][K]
                                else:
                                    for L in range(Ndim_DM2_kOR1[3]):
                                        details["dmOpt"][I][J][K][L]=DM2_kOR1[I][J][K][L]
        ## -----------------------------------------------------------
        if( MPIsize > 1 ):
            diffs_local=[dm2diff,fckdiff]
            diffs_00=sync_dbuf([dm2diff,fckdiff],"sync_eldyn_diffs_%03d.%03d"%(I_step,iter),root=0)
            dm2diff=diffs_00[0];fckdiff=diffs_00[1]
            icvgd=(1 if(dm2diff<dm2diff_TOL) else 0);
            ibuf=np.zeros([3],dtype=np.int64);ibuf[0]=icvgd
            mpi_Bcast("eldyn_cvgd",ibuf)
            if( icvgd != ibuf[0] ):
                printout("## eldyn.icvgd differs:%d  local(%02d):%d (dm2diff_sync:%e %d dm2diff_org:%e %d)/ common:%d"%(\
                    iter, MPIrank,icvgd, dm2diff,(1 if(dm2diff<dm2diff_TOL) else 0),
                    diffs_local[0],(1 if(diffs_local[0]<dm2diff_TOL) else 0), ibuf[0]),Append=True,warning=1,
                    fnme_format="eldyn_iteration_warning%02d.log");
                icvgd=ibuf[0]
        else:
            icvgd=(1 if(dm2diff<dm2diff_TOL) else 0);


        if( icvgd != 0 ):
            logger.info("#iteration converges");cvgd=True;break

        if( dm2diff < dm2diff_min ):
            dm2diff_min=dm2diff; dm2diff_at=iter

        if( dm2diff < diffthr_keep_best ):
            if( MO_best is None ):
                MO_best=arrayclone(MOcoeff_kOR1);diff_best=dm2diff
            elif( dm2diff < diff_best ):
                diff_best=dm2diff
                ndim_MO=np.shape(MOcoeff_kOR1);le_ndim_MO=len(ndim_MO)
                for I in range(ndim_MO[0]):
                    for J in range(ndim_MO[1]):
                        if( le_ndim_MO==2 ):
                            MO_best[I][J]=MOcoeff_kOR1[I][J]
                        else:
                            for K in range(ndim_MO[2]):
                                if( le_ndim_MO==3):
                                    MO_best[I][J][K]=MOcoeff_kOR1[I][J][K]
                                else:
                                    for L in range(ndim_MO[3]):
                                        if( le_ndim_MO==4):
                                            MO_best[I][J][K][L]=MOcoeff_kOR1[I][J][K][L]
                                        else:
                                            for M in range(ndim_MO[4]):
                                                MO_best[I][J][K][L][M]=MOcoeff_kOR1[I][J][K][L][M]


        if( fallback < 0 ):
            if( abs(fallback)==iter ):   ## -1 : single iteration etc. 
                cvgd=True; break;

        if( (diis_updated & FOCK_) != 0 and (diis_errvecs & FOCK_) !=0 ):
            # 8. calculate Fock
            
            mol_or_cell=( rttddft.mol if(not rttddft._pbc) else rttddft.cell) ## 2021.07.12  cell->mol_or_cell
            md.vhf_last = rttddft.get_veff( mol_or_cell, dm=DM2_kOR1 )        ## 2021.07.12  cell->mol_or_cell
            md.Fock_last = rttddft.get_fock( h1e=md.hcore_last,dm=DM2_kOR1,vhf=md.vhf_last )
            ### md._dbg_update_FockLast("eldyn_singlestep.1128",dm=DM2_kOR1,Rnuc=md.Rnuc)
            

        if( diis is not None):
            
            Nremov=None
            if( Navoid_clrbuf <=0 or (iter_clrbuf<0 or iter-iter_clrbuf>Navoid_clrbuf)):
                iter_eff=(iter if(iter_clrbuf<0) else iter-iter_clrbuf)
                if( iter_eff>=Niter_clear_buffer and (iter_eff-Niter_clear_buffer)%Nskip_clear_buffer==0 and dm2diff> diffthr_clear_buffer):
                    range_m=3; n_stepback=10; relt_thr=0.10; Nkeep=0
                    recent_m_sqrerrors=diis.get_recent_sqrerrors(range_m,0);         recent_errmin=np.sqrt( min(recent_m_sqrerrors) )
                    old_m_sqrerrors   =diis.get_recent_sqrerrors(range_m,n_stepback);   old_errmin=np.sqrt( min(old_m_sqrerrors) )
                    
                    if( recent_errmin / old_errmin >= relt_thr ):
                        Nremov=diis.check_buffer_02(Nkeep,order='E',nSleep=6)
                        append_log("## diis_clear  iter:%6d  Nremov=%d  recent_errmin:%e / old_errmin:%e"%(iter,Nremov,recent_errmin,old_errmin))
                    else:
                        append_log("## skipping diis.clear  iter:%6d  recent_errmin:%e / old_errmin:%e"%(iter,recent_errmin,old_errmin))
                if( Nremov is not None):
                    iter_clrbuf=iter
                    log="#eldyn_singlestep:DIIS removes %d vectors.."%(Nremov)
                    if( writer is not None ):
                        writer.append(log)
                        writer.append("#DIIS_vecs:"+i1toa(diis.ibuf,N=min(diis.nbuf, diis.bfsz)))
            

        if( diis is not None ):
        #
        # dm-update:  vec=dm2, err=delta FockMat / delta dm2;  F2[ updated_dm2 ] -> CNlnr_and_dm2 [ F2 ] 
        # Fock-update:vec=H2,  err=delta dm2 / delta FockMat;                       CNlnr_and_dm2 [ updated_F2 ] -> F2[ dm2' ]
        #
            
            dict_DIIS_coefs={"coef":None,"ibuf":None}
            started=diis.started
            if( started and iter_startdiis<0 ):
                iter_startdiis=iter
            elif( (not started) and iter_startdiis>=0 ):
                iter_startdiis=-1

            ervec_AOrep = (DM2_kOR1 - DM2_kOR1_old) if( (diis_errvecs & DMAT_)!=0 ) else (md.Fock_last - Fock2_old )
            ervec = ( calc_DIIServec(rttddft, ervec_AOrep, ('D' if( (diis_errvecs & DMAT_)!=0 ) else 'F') ) if(orthnorm_ervecs) else \
                      ervec_AOrep )
            
            if( (diis_updated & FOCK_)!=0 ):
                md.Fock_last= diis.update(md.Fock_last, ervec=ervec,dict=dict_DIIS_coefs,Iter=iter,  ervec_mixing=diis_ervec_mixing,oldvec_mixing=diis_oldvec_mixing)
                ### md.dbg_update_FockLast("eldyn_singlestep.1199",dm=[],Rnuc=md.Rnuc)
                md.vhf_last=None; md.tmAU_vhf_last=None   ## 20210616: we discard vhf since it is no longer consistent to FockMat.. 

            elif( (diis_updated & DMAT_)!=0 ):
                DM2_kOR1=diis.update(DM2_kOR1, ervec=ervec,dict=dict_DIIS_coefs,Iter=iter, 
                            ervec_mixing=diis_ervec_mixing,oldvec_mixing=diis_oldvec_mixing) ### , ervec_norm=ervec_norm)
                dic1={}
                DM2_kOR1=normalize_DM(rttddft,DM2_kOR1,dev_TOL=1.0e-7,Istep=I_step,Iter=iter,Dict=dic1)
                if( dic1['Nfix']>0 ):
                    printout(str(dic1),fpath=iteration_logfpath,Append=True,Threads=[0],warning=1)
            
            diis_started = diis.started
            

        if( (diis_updated & FOCK_) != 0 and (diis_errvecs & DMAT_) !=0 ):
            # 8. calculate Fock iff DIIS is off
            if( not diis.started ):
                
                mol_or_cell=( rttddft.mol if(not rttddft._pbc) else rttddft.cell) ## 2021.07.12  cell->mol_or_cell
                md.vhf_last = rttddft.get_veff( mol_or_cell, dm=DM2_kOR1 )        ## 2021.07.12  cell->mol_or_cell
                md.Fock_last = rttddft.get_fock( h1e=md.hcore_last,dm=DM2_kOR1,vhf=md.vhf_last )
                ### md.dbg_update_FockLast("eldyn_singlestep.1225",dm=DM2_kOR1,Rnuc=md.Rnuc)
                
        wt_030=rttddft_common.Get_time()
        rttddft_common.Update_timing('single_SCFiter',wt_030-wt_020)
    
    Niter_printed = prtout_iteration(iter,devs,diis_status,iter_strbuf,devs_bfsz,cvgd,wct_01,Niter_printed)

    Wctm020=Wctm010;Wctm010=time.time()
    update_dict(fncnme,dic1_timing,Dic_timing,"iterations",Wctm010-Wctm_ITER00,depth=fncdepth)  

    if( (not cvgd) and (MO_best is not None)):
        ndim_MO=np.shape(MOcoeff_kOR1);le_ndim_MO=len(ndim_MO)
        for I in range(ndim_MO[0]):
            for J in range(ndim_MO[1]):
                if( le_ndim_MO==2 ):
                    MOcoeff_kOR1[I][J]=MO_best[I][J]
                else:
                    for K in range(ndim_MO[2]):
                        if( le_ndim_MO==3):
                            MOcoeff_kOR1[I][J][K]=MO_best[I][J][K]
                        else:
                            for L in range(ndim_MO[3]):
                                if( le_ndim_MO==4):
                                    MOcoeff_kOR1[I][J][K][L]=MO_best[I][J][K][L]
                                else:
                                    for M in range(ndim_MO[4]):
                                        MOcoeff_kOR1[I][J][K][L][M]=MO_best[I][J][K][L][M]
        DM2_kOR1 = rttddft.make_rdm1( MOcoeff_kOR1, md.mo_occ )
        cvgd=True; Dict_warnings.update({'dm2diff':diff_best})

    if( abort_if_failed ):
        assert cvgd," iteration did not converge:best=%e[%d] ... note: set abort_if_failed=False if you have a valid fallback scheme..."%(dm2diff_min, dm2diff_at)

    if( fallback < 0 ):
        Logger.write(writer,"#End iteration at:%d dm2diff=%e"%(iter,dm2diff))
    if( details is not None and ("diff_min" in details) ):
        details["diff_min"]=diff_min

    rttddft_common.Dict_setv("rtTDDFT_Iter",0)
    if( not cvgd ):
        if( details is not None and ("dmOpt" in details) ):
            
            # 9. DM2_kOR1 <- details["dmOpt"]
            for I in range( Ndim_DM2_kOR1[0] ):
                for J in range( Ndim_DM2_kOR1[1] ):
                    if(rank_DM2_kOR1==2):
                        DM2_kOR1[I][J] = details["dmOpt"][I][J]
                    else:
                        for K in range( Ndim_DM2_kOR1[2] ):
                            if(rank_DM2_kOR1==3):
                                DM2_kOR1[I][J][K] = details["dmOpt"][I][J][K]
                            else:
                                for L in range(Ndim_DM2_kOR1[3]):
                                    DM2_kOR1[I][J][K][L] = details["dmOpt"][I][J][K][L]
            # 10.update Fock (DM2_kOR1)
            mol_or_cell=( rttddft.mol if(not rttddft._pbc) else rttddft.cell) ## 2021.07.12  cell->mol_or_cell
            md.vhf_last = rttddft.get_veff( mol_or_cell, dm=DM2_kOR1 )        ## 2021.07.12  cell->mol_or_cell
            md.Fock_last = rttddft.get_fock( h1e=md.hcore_last, dm=DM2_kOR1, vhf=md.vhf_last )

            # 11.propagate MO_coeff
            for sp in range(nmult_MO):
                mo_coeff = ( MOcoeff_kOR1 if(nmult_MO==1) else MOcoeff_kOR1[sp])
                f2_kOR1  = ( md.Fock_last if(nmult_MO==1) else md.Fock_last[sp]) 
                f1_kOR1  = ( Fock1_kOR1_eff if(nmult_MO==1) else Fock1_kOR1_eff[sp]) 

                if( not pbc ):
                    mo_coeff = CNlnr_AOrep (rttddft._Ssqrt,rttddft._Sinvrt,f2_kOR1,f1_kOR1,tm2_AU-tm1_AU,Nstep_CN,mo_coeff,
                                             check_vecnorms=(md.step%200==1),rttddft=rttddft,moldyn=md)
                    if( nmult_MO == 1 ):
                        MOcoeff_kOR1 = mo_coeff
                    else:
                        MOcoeff_kOR1[sp] = mo_coeff
                else:
                    for k in range(nkpt):
                        mo_coeff[k] = CNlnr_AOrep (rttddft._Ssqrt[k],rttddft._Sinvrt[k],f2_kOR1[k],f1_kOR1[k],tm2_AU-tm1_AU,Nstep_CN,mo_coeff[k],
                                               kp=k, check_vecnorms=(md.step%200==1),rttddft=rttddft,moldyn=md)
                        if( nmult_MO == 1 ):
                            MOcoeff_kOR1[k] = mo_coeff[k]
                        else:
                            MOcoeff_kOR1[sp][k] = mo_coeff[k]

            DM2_NEW = rttddft.make_rdm1( MOcoeff_kOR1, md.mo_occ )
            dm2diff=dot_AOmatrices( DM2_NEW-DM2_kOR1, DM2_NEW-DM2_kOR1, SAO, rttddft._spinrestriction,'D')
            dm2diff=np.sqrt( abs(dm2diff.real) )

            Logger.write( writer,"#fallback:apply best result:diff_min %e >> final diff %e"%(details["diff_min"],dm2diff))
            

    if( rttddft._calc_Aind != 0 ):
        update_Aind_over_c(rttddft)
        Aind_overC=rttddft._Aind_Gearvc_C[0];tm_fs=rttddft._Aind_tmAU*physicalconstants.aut_in_femtosec
    
    result = "Iteration "+("converged" if(cvgd) else "did not converge") + " after %d iterations."%(iter)
    if( writer is not None ):
        if( len(devs[0])>0 ):
            for j in range(len(devs[0])):
                writer.append("%d %e %e %d"%(j,devs[0][j],devs[1][j],(1 if(diis_status[j]) else 0)))
        writer.append("#"+result)
        renameto=None
        if( iter >=100 ):
            if( not append_iterationlog ):
                renameto="eldyn_iteration_%s.log"%(step)
        writer.close(renameto)

    if( not cvgd ):
        if( details is not None ):
            details["dm2diff"]=dm2diff;
    
    logger.timing(result,end=True)
    logger.Info(result)

    if(synchronize):
        sync_threads(md,rttddft,MOcoeff_kOR1,DM2_kOR1,I_step) ## sync

    ## md.print_DMO( "#moldyn_main:AF:%s %r %r"%(step,update_attribute,cvgd), dmat=dm2_kOR1 )
    if(update_attribute):
        if( cvgd or (force_update and (details is not None and ("dmOpt" in details))) ):
            #if(dbgng_tdMO):
            #    diff3=aNmaxdiff(MOcoeff_kOR1,md.tdMO,comment="finalMO-md.tdMO:%02d"%(I_step),lower_thr=1.0e-6,iverbose=2,
            #        title="finalMO-md.tdMO:%02d"%(I_step),logfile="finalMO-md.tdMO_diff.log")

            for sp in range(nmult_MO):
                mo_coeff=(MOcoeff_kOR1 if(nmult_MO==1) else MOcoeff_kOR1[sp])
                tdMO=(md.tdMO if(nmult_MO==1) else md.tdMO[sp])
                nd_mo=np.shape(mo_coeff)
                for I in range( nd_mo[0] ):
                    for J in range( nd_mo[1] ):
                        if( not pbc ):
                            tdMO[I][J] = mo_coeff[I][J]
                        else:
                            for K in range( nd_mo[2] ):
                                tdMO[I][J][K] = mo_coeff[I][J][K]
                dm2_kOR1=(DM2_kOR1 if(nmult_MO==1) else DM2_kOR1[sp])
                tddm    =(md.tdDM if(nmult_MO==1) else md.tdDM[sp])
                nd_dm=np.shape(dm2_kOR1)
                for I in range( nd_dm[0] ):
                    for J in range( nd_dm[1] ):
                        if( not pbc ):
                            tddm[I][J] =dm2_kOR1[I][J]
                        else:
                            for K in range( nd_dm[2] ):
                                tddm[I][J][K] = dm2_kOR1[I][J][K]
            md.save_tempFockMat(md.Fock_last,tm2_AU)
        else:
            logger.warning("#singlestep:iteration did not converge");

    
    wt_100=rttddft_common.Get_time()
    rttddft_common.Print_timing("eldyn_singlestep",["setup","Sinv","hcore","fock","CNlnr","rdm2","diis",
                                                  "diis_misc","setup2","setup3","setup4","misc1","Fallback1","af-iteration"],
                               walltime=wt_100-wt_010,depth=depth,N_iter=iter)
    def fnprtoutdict2(arg):
        return (arg<=10 or arg%10==0) 
    Wctm010=time.time()
    printout_dict(fncnme,dic1_timing,Dic_timing,Wctm010-Wctm000,depth=fncdepth,fnprtout=fnprtoutdict2)
    return (iter if cvgd else min([-1,-iter]))

# CNlnr_AOrep : wrapper of CNlnr. Canonicalizes input
#               S^{-0.5} H S^{-0.5} ( S^{0.5} C ) = i\hbar \partial_t (S^{0.5} C)
# @input : c_in[nAO][Nv]
# @input : check_projection : str output filepath
# @return: ret[nAO][Nv]
def CNlnr_AOrep(Ssqrt,Sinvrt,H2,H1,dt,Nstep,c_in,Nv=None,kp=-1, eorbs=None,check_vecnorms=False,check_projection=None,rttddft=None,moldyn=None):
    # here c_in is [*][nAO] dimensional input and we want to propagate Nv vectors 
    # if( "propagator" in rttddft_common.params ):
    #    assert rttddft_common.params["propagator"]=="CN",""

    if(Nv is None):
        Nv=len(c_in[0])
    nAO=len(c_in)

#    if(kp==0):
#        jc=0
#        col_j=[ row[jc] for row in c_in ]  # c_in[:,jc]
#        print("H0xC0:",end="");print( np.matmul(H1,col_j))
#        print("S0xC0*e:",end="");print( np.matmul( np.matmul(Ssqrt,Ssqrt),col_j)*eorbs[jc])
        
    # Sinv(:,:) H Sinv(:,:)   Sinv is Hermitian..
    H1modif=np.matmul( np.matmul( Sinvrt,H1), Sinvrt)
    # 
    H2modif=np.matmul( np.matmul( Sinvrt,H2), Sinvrt)
    #print("CNlnr_AOrep:",np.shape(H1),np.shape(H2),np.shape(Sinvrt),np.shape(H1modif),np.shape(H2modif))    
    ret=np.zeros([nAO,Nv],dtype=np.complex128)
    # print("check_projection"+check_projection);assert False,""
    def fnCheckMO(step,tme,coefs):
        if( (moldyn is None) or (rttddft is None) ):
            return 
        ndim=np.shape(coefs)
        WKS=np.zeros([ ndim[0],ndim[1] ],dtype=np.complex128)
        WKS=np.matmul(Sinvrt, coefs) ## Canonical MOs 
        moldyn.calc_MOprojection(rttddft, "%d %f "%(step,tme),
             fpath=("CheckMO_CNlnrAOrep.dat" if (check_projection is None) else check_projection),mo_coeff=WKS)
        ### calc_MOprojection( self, rttddft, header,fpath=None,mo_coeff=None,kpts=None,append=True,dict=None)
    SxC_in=np.matmul( Ssqrt, c_in)  # this allocates a new matrix..
    if( check_projection is not None ):
        assert (rttddft is not None),"check_projection"
        ret=np.matmul( Sinvrt, CNlnr(H2modif,H1modif,dt,Nstep,SxC_in,check_vecnorms=check_vecnorms,fnc=fnCheckMO) )
    else:
        ret=np.matmul( Sinvrt, CNlnr(H2modif,H1modif,dt,Nstep,SxC_in,check_vecnorms=check_vecnorms) )
    return ret


# CNlnr : linearly interpolated Crank-Nicolson: 
# @input:  H2[nAO][nAO], H1[nAO][nAO], dt:time in AU
#          c_in[nAO][Nv] 
# @return: c[nAO][Nv] (newly allocated)
def CNlnr(H2,H1,dt,Nstep,c_in, norm_dev_tol=1.0e-6, check_vecnorms=False, fnc=None):
    # H1*(1-u) + H2*u   u=j+0.5/Nstep  j=0,1,...,Nstep-1
    # H1 + j*D   with D=(H2-H1)/N
    ## print("#CNlnr:%f"%(dt))
    assert (c_in.dtype == np.complex128),"wrong type of input for c_in:"+str(c_in.dtype)
    nAO=len(c_in);Nv=len(c_in[0])
    fac=1.0/float(Nstep)
    DH=np.zeros([nAO,nAO],dtype=np.complex128); H=np.zeros([nAO,nAO],dtype=np.complex128)
    c=np.zeros([nAO,Nv],dtype=np.complex128)
    DH[:][:]=(H2[:][:]-H1[:][:])*fac
    H[:][:]=H1[:][:] + 0.50*DH[:][:]
    c[:][:]=c_in[:][:]
    eps= dt/float(Nstep)
    for j in range(Nstep):
        c=CNstep1(H,eps,c)
        if( fnc is not None ):
            fnc(j,eps*(j+1),c)
        H[:][:] = H[:][:] + DH[:][:]
    # (c^{\dagger} c)_{jj} is the norm of jth vector
    wks=np.zeros([Nv,Nv],dtype=np.complex128)
    #wks=np.matmul( np.matrix(c_in).getH(), c_in)
    wks=np.matmul(  np.transpose(np.conj(c_in)), c_in)
    sqnorm0=[]
    for j in range(Nv):
        sqnorm0.append( wks[j][j].real )
    wks=np.matmul(  np.transpose(np.conj(c)), c)
    #wks=np.matmul( np.matrix(c).getH(), c)
    sqnorm1=[]
    for j in range(Nv):
        sqnorm1.append( wks[j][j].real )
    devmx=-1; at=-1
    nWarn=0
    for j in range(Nv):
        ratio=math.sqrt( sqnorm1[j]/sqnorm0[j] )
        dev1=abs(ratio-1.0)
        if( devmx< dev1 ):
            devmx= dev1; at=j
        if( abs( 1.0- ratio)> norm_dev_tol):
            nWarn+=1
            fac=math.sqrt( sqnorm0[j]/sqnorm1[j] )
            for k in range(nAO):
                c[k][j]=c[k][j]*fac
    if( check_vecnorms  or nWarn>0 ):
        
        printout("norm:N=%d:dt=%e:maxdev:%e %f->%f"%(Nstep,eps,devmx,math.sqrt(sqnorm0[at]),math.sqrt(sqnorm1[at])))
        
    return c


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
def mpi_aNdiff(buf,key,compto=0,sync=False,diff_THR=-1.0, details=False):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()    
    if( MPIsize < 2 ):
        return
    assert compto>=0,"" ## (-1) may mean comp to MPIrank-1 (mod size) th thread but..
    assert compto<MPIsize,"" ## (-1) may mean comp to MPIrank-1 (mod size) th thread but..
    wks=arrayclone(buf)
    mpi_Bcast("diff_"+key,wks,root=compto)
    sqrediff,maxabsdiff,At,vals=aNdiff(buf,wks)
    
    if( details and  (MPIrank != compto) ):
        fnme="diff_"+key+"_%02d-%02d.dat"%(MPIrank,compto)
        print_aNbufx2(buf,wks,fpath=None,fnme_format=None,fpath_threadlocal=fnme,Threads=[MPIrank],
                      Nd=5, half=False,description="#lhs:thread%02d / rhs:thread%02d"%(MPIrank,compto))

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


def aNdiff(A,B):
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

def normalize_DM(rttddft,dmat,dev_TOL=1.0e-7,Istep=-1,Iter=-1,Dict=None):
    dbgng=True
    Nfix=0;string="";dev=0
    trDM=trace_dmat(rttddft, dmat, rttddft._spinrestriction, rttddft._pbc)
    N_ele=get_Nele(rttddft)
    if( rttddft._spinrestriction == 'R' ):
        ## N_ele and trDM are both scalar ...
        dev=abs( N_ele-trDM );strfac=" fac:"
        if( dev > dev_TOL ):
            fac=N_ele/trDM;strfac="%14.4e "%(fac)
            dmat=np.array(dmat)*fac;Nfix=Nfix+1
        string="#normalize_DM:"+("@%04d.%03d:"%(Istep,Iter) if(Istep>0 and Iter>0) else "")\
               +"dev:%14.4e "%(dev)+strfac
    else:
        devs=[-1,-1];
        for sp in range(2):
            devs[sp]=abs( N_ele[sp]-trDM[sp] );strfac=" fac:"
            if( devs[sp] > dev_TOL ):
                fac=N_ele[sp]/trDM[sp];strfac="%14.4e "%(fac)
                dmat[sp] = np.array(dmat[sp])*fac; Nfix=Nfix+1
            if( sp== 0 ):
                string="#normalize_DM:"+("@%04d.%03d:"%(Istep,Iter) if(Istep>0 and Iter>0) else "")
            string+=" dev[%s]:%14.4e "%(sp,devs[sp])+strfac
        dev=max(devs)
    if(Dict is not None):
        Dict.update({'Nfix':Nfix,'log':string,'dev':dev})

    if(Nfix>0 and dbgng):
        trDM=trace_dmat(rttddft, dmat, rttddft._spinrestriction, rttddft._pbc)
        dev=( abs(N_ele - trDM) if(rttddft._spinrestriction == 'R') else max( [ abs(N_ele[0] - trDM[0]), abs(N_ele[1] - trDM[1]) ] ))
        assert dev<dev_TOL,"dev=%e for "%(dev) + str(trDM)+"/"+str(N_ele)
    return dmat

def assertf(bool,text,level):
    rttddft_common.Assert(bool,text,level)

def dict_getv(dict,key,default=None):
    if(dict is None):
        return default
    if(key in dict):
        return dict[key]
    else:
        return default;

def calc_DIIServec(rttddft, errorvec_AOrep, DorF, dm=None ):
    from rttddft01 import get_Nele
    printout("#calc_DIIServec:input Ndim:"+str(np.shape(errorvec_AOrep))+str(errorvec_AOrep.dtype))
    nmult=( (1 if(rttddft._spinrestriction=='R') else 2) if(DorF=='D') else \
           ((2 if(rttddft._spinrestriction=='U') else 1) if(DorF=='F') else None) )
    assert nmult is not None,"wrong INPUT:"+rttddft._spinrestriction+"/"+DorF
    pbc=rttddft._pbc;nkpt=1;
    if(pbc):
        kvectors=np.reshape( rttddft.kpts, (-1,3))
        nkpt=len(kvectors)
    Ret=[]
    for sp in range(nmult):
        ervec=( errorvec_AOrep if(nmult==1) else errorvec_AOrep[sp] )
        Ret_s=[]
        for kp in range(nkpt):
            vec=( ervec if(not pbc) else ervec[kp])
            Ssqrt=( rttddft._Ssqrt if(not pbc) else rttddft._Ssqrt[kp] )
            if( kp==0 and sp==0 ):
                assert i1eqb(np.shape(vec),  [rttddft.nAO,rttddft.nAO]),""+str( np.shape(vec) ) +"/%d"%(rttddft.nAO)
                assert i1eqb(np.shape(Ssqrt),[rttddft.nAO,rttddft.nAO]),""+str( np.shape(Ssqrt))+"/%d"%(rttddft.nAO)
            ret=np.matmul( Ssqrt, np.matmul( vec, Ssqrt))
            Ret_s.append(ret)
        if(not pbc):
            Ret_s = Ret_s[0]
        Ret.append(Ret_s)
    if(nmult==1):
        Ret=Ret[0]
    retv=np.array(Ret)
    printout("#returning:"+str(type(retv))+str(np.shape(retv))+str(retv.dtype))

    if(dm is not None):
        N_ele=get_Nele(rttddft);dev=-1
        for sp in range(nmult):
            dm_s=( dm if(nmult==1) else dm[sp] )
            Ne_s=( N_ele if(nmult==1) else N_ele[sp])
            DMtrace=0.0
            for kp in range(nkpt):
                vec=( dm_s if(not pbc) else dm_s[kp])
                Ssqrt=( rttddft._Ssqrt if(not pbc) else rttddft._Ssqrt[kp] )
                mat= np.matmul( Ssqrt, np.matmul( vec, Ssqrt))
                for I in range(rttddft.nAO):
                    DMtrace+=mat[I][I]
            if(pbc):
                DMtrace=DMtrace/float(nkpt)
            diff=abs(DMtrace-Ne_s); dev=max(diff,dev)
            assert (diff<1.0e-6),"DMtrace:%f / %f"%(DMtrace,Ne_s)
        printout("#check_dmtrace:maxdev:%e from Ne:"%(dev)+str(N_ele))
    return retv

def aNcpy(src,dst):
    Ndim_s=np.shape(src)
    Ndim_d=np.shape(dst)
    if( not i1eqb(Ndim_s,Ndim_d) ):
        printout("#aNcpy:!W Ndim_s:"+str(Ndim_s)+"/Ndim_d:"+str(Ndim_d))
        Ldim=i1prod(Ndim_s)
        src_1D=np_ravel(src)
        ret_1D=np.zeros([Ldim],dtype=(src_1D.dtype))
        for j in range(Ldim):
            ret_1D[j]=src_1D[j]
        return np.reshape(ret_1D,Ndim_s)
    else:
        le_Ndim=len(Ndim_s)
        if(le_Ndim==1):
            for I in range(Ndim_s[0]):
                dst[I]=src[I]
        elif(le_Ndim==2):
            for I in range(Ndim_s[0]):
                for J in range(Ndim_s[1]):
                    dst[I][J]=src[I][J]
        elif(le_Ndim==3):
            for I in range(Ndim_s[0]):
                for J in range(Ndim_s[1]):
                    for K in range(Ndim_s[2]):
                        dst[I][J][K]=src[I][J][K]
        elif(le_Ndim==4):
            for I in range(Ndim_s[0]):
                for J in range(Ndim_s[1]):
                    for K in range(Ndim_s[2]):
                        for L in range(Ndim_s[3]):
                            dst[I][J][K][L]=src[I][J][K][L]
        elif(le_Ndim==5):
            for I in range(Ndim_s[0]):
                for J in range(Ndim_s[1]):
                    for K in range(Ndim_s[2]):
                        for L in range(Ndim_s[3]):
                            for M in range(Ndim_s[4]):
                                dst[I][J][K][L][M]=src[I][J][K][L][M]
        else:
            assert False,"Ndim_s:"+str(Ndim_s)
        return dst

# CNstep1 : performs a single step Crank-Nicolson:  (1+ 1j hfeps H)ret = (1 - 1j hfeps H)v
# 
#   @input:   H[nAO][nAO]
#             v[nAO][Nv]
#             eps           time step in AU
#   @return:  z[nAO][Nv]
def CNstep1(H,eps,v):
    
    hfeps=eps*0.5
    nAO=len(v); Nv=len(v[0])
    
    A=np.identity(nAO,dtype=np.complex128)
    A[:][:]=A[:][:] + 1j*H[:][:]*hfeps 
    invA=np.linalg.inv(A)
    z=np.zeros([nAO,Nv],dtype=np.complex128)
    z[:][:]=np.matmul(invA, v[:][:] - 1j*hfeps* np.matmul(H,v))

    return z;

def check_orthnorm(md,rttddft,tdMO,dag_tol=1.0e-6,ofd_tol=1.0e-6):
    pbc = md.pbc
    kvectors=None
    if( pbc ):
        kvectors=np.reshape( rttddft.kpts, (-1,3))
    nmult_MO=( 2 if(md.spinrestriction=='U') else 1)
    nkpt=(1 if (not md.pbc) else md.nkpt)
    DAGdev=-1.0;DAG_ref=None
    OFDdev=-1.0;OFD_ref=None
    S1e=(None if(pbc) else rttddft.get_ovlp())
    Nfix=0
    for spin in range(nmult_MO):
        MO=( tdMO if(nmult_MO==1) else tdMO[spin] )
        for kp in range(nkpt):
            dagdev=-1.0;dag_ref=None
            ofddev=-1.0;ofd_ref=None
            MO1=( MO if(not pbc) else MO[kp] )
            if(pbc):
                S1e= rttddft.get_ovlp(rttddft.cell,kvectors[kp])
            ndim=np.shape(MO1);
            nAO=ndim[0];nMO=ndim[1]
            Sij = np.zeros([nMO,nMO],dtype=np.complex128)
            for Imo in range(nMO):
                SxI=np.matmul(S1e,MO1[:,Imo])
                Sij[Imo][Imo]=np.vdot( MO1[:,Imo],SxI )
                norm=np.sqrt( Sij[Imo][Imo].real )
                dum=abs(norm-1.0)
                if( dagdev<dum ):
                    dagdev=dum; dag_ref=[Imo,Imo]
                for Jmo in range(Imo):
                    cdum = np.vdot( MO1[:,Jmo],SxI )
                    Sij[Jmo][Imo]=cdum
                    Sij[Imo][Jmo]=np.conj( cdum )
                    dum=np.sqrt( cdum.real**2 + cdum.imag**2 )
                    if( ofddev < dum ):
                        ofddev=dum; ofd_ref=[Imo,Jmo]
            if( DAGdev< dagdev ):
                DAGdev=dagdev;DAGref= [spin,kp,dag_ref[0],dag_ref[1]]
            if( OFDdev < ofddev ):
                OFDdev=ofddev; OFDref=[spin,kp,ofd_ref[0],ofd_ref[1]]

            if(ofddev>=ofd_tol or dagdev>=dag_tol):
                printout("#check_orthnorm:!W large dev:DAG:%e [%d][%d] %f+j%f / OFD:%e [%d][%d] %f+j%f"%(
                    dagdev,dag_ref[0],dag_ref[1], Sij[dag_ref[0]][dag_ref[1]].real, Sij[dag_ref[0]][dag_ref[1]].imag,
                    ofddev,ofd_ref[0],ofd_ref[1], Sij[ofd_ref[0]][ofd_ref[1]].real, Sij[ofd_ref[0]][ofd_ref[1]].imag))

                Sij_invsqrt = calc_Sroot(Sij,[-1])
                MO_new = np.matmul(MO, Sij_invsqrt)
                Dic={'DAGdev':None,'OFDdev':None}
                dev_new=a2sqrdiff( np.matmul( np.matrix.getH( MO ), np.matmul( S1e, MO ) ), np.eye(nMO), DAG_OFDdev=Dic)
                printout("#check_orthnorm:dev=%e,%e -> %e,%e"%(dagdev,ofddev,Dic['DAGdev'],Dic['OFDdev']))
                assert dev_new<1.0e-7,""
                Nfix+=1
    return DAGdev,OFDdev,Nfix

def check_nrmz(md,rttddft,tdMO,norm_dev_tol=1.0e-6):
    pbc = md.pbc
    nmult_MO=( 2 if(md.spinrestriction=='U') else 1)
    nkpt=(1 if (not md.pbc) else md.nkpt)
    maxdev=-1.0;nfix=0;value_ref=None
    for spin in range(nmult_MO):
        MO=( tdMO if(nmult_MO==1) else tdMO[spin] )
        for kp in range(nkpt):
            MO1=( MO if(not pbc) else MO[kp] )
            ### dtyp =( np.array( MO1 ).dtype )
            ### icmplx=(dtyp == np.complex128 or dtyp==complex )
            ### print(dtyp);print(icmplx)
            S1e=( rttddft.get_ovlp() if(not pbc) else rttddft.get_ovlp(rttddft.cell,kvectors[kp]) )
            ndim=np.shape(MO1);
            nAO=ndim[0];nMO=ndim[1]
            assert ( len(ndim)==2 ),""
            for jmo in range(nMO):
                norm= np.vdot( MO1[:,jmo], np.matmul( S1e, MO1[:,jmo] ) )
                norm=norm.real
                ### print("norm:",end="");print(norm)
                dev=abs(norm-1)
                if( dev > maxdev ):
                    maxdev=dev;value_ref=norm
                if( dev >= norm_dev_tol ):
                    fac=1.0/math.sqrt(norm);nfix+=1
                    for kao in range(nAO):
                        MO1[kao][jmo]=MO1[kao][jmo]*fac
    return nfix,maxdev

def test_reload(moldyn1,dumpfpath):
    os.system("ls -ltrh "+dumpfpath)
    md2=Moldyn.load(dumpfpath)
    Ndiff,Nsame,Nerr= diff_objects(moldyn1,md2,verbose=1,diff_TOL=1.0e-7)
    md2.save("moldyn_dbg.pscf",delimiter='\n',comment=" time=%f(au) of "%(md2.time_AU) + rttddft_common.get_job(True));
    os.system("ls -ltrh "+dumpfpath)
    os.system("ls -ltrh moldyn_dbg.pscf")

def print_dm2(cur,prev,Indices,Istep,text="",Istep_upd=[0,4,9,19],N_uplm=[5,10,15,20],TINY=5.0e-6,
            filepath=None):
    Ndim=np.shape(cur);Ld=i1prod(Ndim)
    lhs=np.ravel( np.array(cur) )
    rhs=np.ravel( np.array(prev))
    le=len(lhs);assert (le==Ld),"ravel:"+str(Ndim)+" >> "+str(np.shape(lhs))
    maxdev=a1maxdiff(lhs,rhs)
    sort_diff=( len(Indices)==0 or (Istep in Istep_upd) )
    
    if( sort_diff ):
        Iupd=0
        for k in range(1,len(Istep_upd)):
            if(Istep==Istep_upd[k]):
                Iupd=k
        arr=[ abs( lhs[k]-rhs[k] ) for k in range(le) ]
        ithTOk=np.argsort(arr)
        Nadd=0
        for j in range(le):
            ith=le-j-1
            ks=ithTOk[ith];val=arr[ks]
            if( val < TINY ):
                break
            if( ks in Indices ):
                continue
            Indices.append(ks);Nadd+=1;
            if(len(Indices)>N_uplm[Iupd]):
                break
    string="%06d:%14.4e:"%(Istep,maxdev);delimiter='\t '
    idcs=Indices
    if( len(Indices)==0 ):
        Nprt=min(10,le);idcs=[ k for k in range(Nprt) ]
    for I in idcs:
        string+="%03d:%12.6f+j%12.6f  %12.6f+j%12.6f "%(
                I,lhs[I].real,lhs[I].imag, rhs[I].real, rhs[I].imag)+delimiter
    printout("#prtDM2:"+text+" "+string,fpath=filepath,Append=True,Threads=[0],stdout=True);




def sync_dbuf(data,key,root=0,check_diff=None):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size() 
    Ld=len(data);
    
    bfsz=30;assert(Ld<=bfsz),"";
    buf=np.zeros([bfsz],dtype=np.float64)
    for j in range(Ld):
        buf[j]=data[j]

    cpy=None
    if(check_diff is not None):
        cpy=arrayclone(data)
    mpi_Bcast( "sync_dbuf."+key,buf,root=root, barrier=True, iverbose=1, Append=False)
    if(check_diff is not None):
        if(cpy is not None ):
            fnme="checksync_"+key+"_%02d-%02d.dat"%(MPIrank,root);
            if( "fnme_format" in check_diff ):
                if(check_diff["fnme_format"] is not None):
                    fnme=check_diff["fnme_format"]%(MPIrank,root)
            fd1=open(fnme,'a');string=""
            if( "description" in check_diff ):
                string=str(check_diff["description"])+":"
            str1="";str2="";str3="";
            for k in range(Ld):
                str1+="%12.6f "%(cpy[k])
                str2+="%12.6f "%(buf[k])
                str3+="%10.3e "%(abs(buf[k]-cpy[k]))
            print(string+" \tLHS:"+str1+ " \tRHS:"+str2+ " \tDIFF:"+str3,file=fd1)
            fd1.close()
    return np.array( [ buf[k] for k in range(Ld) ] )


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

def sync_threads(md,tddft,mo_coeff,dm2_kOR1,Istep=0,flag=15,check_diff=None,subkey="",flag_details=0):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()    
    if( MPIsize < 2 ):
        if(Istep==0):
            print("#sync_tdWF:no sync"); ### iff MPIsize<2
        return

    if((flag & 1)!=0):
        do_sync1(mo_coeff,barrier=True,key="eldyn.step%03d"%(Istep)+subkey+".mo_coeff",root=0,
                 iverbose=(2 if(Istep<2 or Istep==10 or Istep==50) else 1),check_diff=check_diff,details=( (flag_details%1)!=0 ) )
    if((flag & 2)!=0):
        do_sync1(dm2_kOR1,barrier=True,key="eldyn.step%03d"%(Istep)+subkey+".dm2_kOR1",root=0,
                iverbose=(2 if(Istep<2) else 0),check_diff=check_diff,details=( (flag_details%2)!=0 ) )
    if((flag & 4)!=0):
        do_sync1(md.hcore_last,barrier=True,key="eldyn.step%03d"%(Istep)+subkey+".hc_last",root=0,
                iverbose=(2 if(Istep<2) else 0))
    if((flag & 8)!=0):
        do_sync1(md.Fock_last,barrier=True,key="eldyn.step%03d"%(Istep)+subkey+".Fock_last",root=0,
                iverbose=(2 if(Istep<2) else 0))
    key="eldyn.step%03d.sync_end"%(Istep)
    comm_Barrier(key)

def print_aNbuf(buf,fpath=None,fnme_format=None,fpath_threadlocal=None,Threads=None,Append=False,Nd=8, half=False):
    Ndim=np.shape(buf);leNdim=len(Ndim)
    dt=(np.array(buf)).dtype
    iscmplx=( dt == complex or dt == np.complex128 )
    path,fd,Threads=path_fd_Threads_(fpath=fpath,fnme_format=fnme_format,fpath_threadlocal=fpath_threadlocal,
                                       Threads=Threads,Append=Append)

    def print_a2buf(bfr,Nd,fd_out,is_complx,half=False):
        ndim=np.shape(bfr)
        nx=min(ndim[0],Nd);ny_0=min(ndim[1],Nd)
        for ix in range(nx):
            string=""
            if(is_complx):
                ny=( min(ix+1,ny_0) if(half) else ny_0)
                for jy in range(ny):
                    string+="%10.4f %10.4f      "%( bfr[ix][jy].real, bfr[ix][jy].imag )
            else:
                ny=( min(ix+1,ny_0) if(half) else ny_0)
                for jy in range(ny):
                    string+="%12.6f      "%( bfr[ix][jy] )
            print(string,file=fd_out)

    if( leNdim==2 ):
        print_a2buf( buf,Nd,fd, iscmplx)
    elif( leNdim==3 ):
        n1=min(3,Nd,Ndim[0])
        for I in range(n1):
            print(('\n\n\n' if(I>0) else '')+"#%03d "%(I)+str(Ndim),file=fd)
            print_a2buf( buf[I],Nd,fd, iscmplx,half=half)
    elif( leNdim==4 ):
        n1=min(2,Nd,Ndim[0]);n2=min(2,Nd,Ndim[1]);IxJ=-1
        for I in range(n1):
            for J in range(n2):
                IxJ+=1;print( ('\n\n\n' if(IxJ>0) else '')+"#%03d %d,%d"%(IxJ,I,J)+str(Ndim),file=fd)
                print_a2buf( buf[I][J],Nd,fd, iscmplx,half=half)
    if(path is not None):
        fd.close()
def print_aNbufx2(lhs,rhs,fpath=None,fnme_format=None,fpath_threadlocal=None,Threads=None,Append=False, Nd=5, half=False,description=""):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()    
    Ndim=np.shape(lhs);leNdim=len(Ndim)
    sqrediff,maxabsdiff,At,vals=aNdiff(lhs,rhs)
    dt=(np.array(lhs)).dtype
    iscmplx=( dt == complex or dt == np.complex128 )
    path,fd,Threads=path_fd_Threads_(fpath=fpath,fnme_format=fnme_format,fpath_threadlocal=fpath_threadlocal,
                                       Threads=Threads,Append=Append)
    if( MPIrank not in Threads ):
        return

    print("#"+description+" dist:%e maxdiff:%e %f+j%f/%f+j%f at "%(
            np.sqrt(sqrediff),maxabsdiff,vals[0].real,vals[0].imag,
            vals[1].real,vals[1].imag)+str(At),file=fd)

    def print_a2bufx2(lbfr,rbfr,Nd,fd_out,is_complx,half=False,diff=False,description=""):
        ndim=np.shape(lbfr)
        nx=min(ndim[0],Nd);ny_0=min(ndim[1],Nd)
        if(diff):
            sqrediff1,maxabsdiff1,At1,vals1=aNdiff(lbfr,rbfr)
            print("##"+description+" dist:%e maxdiff:%e %f+j%f/%f+j%f at "%(
                    np.sqrt(sqrediff1),maxabsdiff1,vals1[0].real,vals1[0].imag,
                    vals1[1].real,vals1[1].imag)+str(At),file=fd_out)
        for ix in range(nx):
            string=""
            if(is_complx):
                ny=( min(ix+1,ny_0) if(half) else ny_0)
                for jy in range(ny):
                    string+="%10.4f %10.4f      "%( lbfr[ix][jy].real, lbfr[ix][jy].imag )
                string+=' '*((ny_0-ny)*27)
                for jy in range(ny):
                    string+="%10.4f %10.4f      "%( rbfr[ix][jy].real, rbfr[ix][jy].imag )
                string+=' '*((ny_0-ny)*27)
                for jy in range(ny):
                    string+="%9.3e    "%( abs(lbfr[ix][jy]-rbfr[ix][jy]) )
            else:
                ny=( min(ix+1,ny_0) if(half) else ny_0)
                for jy in range(ny):
                    string+="%12.6f      "%( lbfr[ix][jy] )
                string+=' '*((ny_0-ny)*18)
                for jy in range(ny):
                    string+="%12.6f      "%( rbfr[ix][jy] )
                string+=' '*((ny_0-ny)*18)
                for jy in range(ny):
                    string+="%9.3e    "%( abs(lbfr[ix][jy]-rbfr[ix][jy]) )
            print(string,file=fd_out)

    if( leNdim==2 ):
        print_a2bufx2( lhs,rhs,Nd,fd, iscmplx, half=half, diff=False,description="")
    elif( leNdim==3 ):
        n1=min(3,Nd,Ndim[0])
        for I in range(n1):
            string=('\n\n\n' if(I>0) else '')+"#%03d "%(I)+str(Ndim)
            print_a2bufx2( lhs[I],rhs[I],Nd,fd, iscmplx,half=half, diff=True,description=string )
    elif( leNdim==4 ):
        n1=min(2,Nd,Ndim[0]);n2=min(2,Nd,Ndim[1]);IxJ=-1
        for I in range(n1):
            for J in range(n2):
                IxJ+=1;description= ('\n\n\n' if(IxJ>0) else '')+"#%03d %d,%d"%(IxJ,I,J)+str(Ndim)
                print_a2bufx2( lhs[I][J],rhs[I][J], Nd,fd, iscmplx,half=half, diff=True,description=description)
    if( path is not None ):
        fd.close()
def i1prod(Iarr):
    le=len(Iarr)
    if(le==0):
        return 1
    else:
        ret=Iarr[0]
        for k in range(1,le):
            ret=ret*Iarr[k]
        return ret

def do_sync1(buffer,barrier=True,key="",root=0,iverbose=3, check_diff=None, details=False):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()    

    if( isinstance(buffer,np.ndarray) ):
        bcast_ndarray(buffer,barrier=barrier,key=key,root=root,iverbose=iverbose, check_diff=check_diff,details=details)
    else:
        le=len(buffer)
        if( isinstance( buffer[0],np.ndarray) ):
            for I in range(le):
                bcast_ndarray( buffer[I],barrier=( barrier and I==0 ),key=key+"[%02d]"%(I),
                               root=root,iverbose=(iverbose if(I==0) else iverbose-1), check_diff=check_diff,details=details )
        else:
            for I in range(le):
                buffer[I]=np.array(buffer[I])
                bcast_ndarray( buffer[I],barrier=( barrier and I==0 ),key=key+"[%02d]"%(I),
                               root=root,iverbose=(iverbose if(I==0) else iverbose-1), check_diff=check_diff,details=details )
def path_fd_Threads_(fpath=None,fnme_format=None,fpath_threadlocal=None,Threads=None,Append=False):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    fd=None;path=None;
    if(fpath is None):
        path_threadlocal=None;
        if(fpath_threadlocal is not None):
            path_threadlocal=fpath_threadlocal;
        elif( fnme_format is not None ):
            path_threadlocal=fnme_format%(MPIrank)

        if( path_threadlocal is not None ):
            if(Threads is None):
                Threads=[ j for j in range(MPIsize) ]
            fd=open(path_threadlocal,('a' if(Append) else 'w'))
            return path_threadlocal,fd,Threads
        else:
            if(Threads is None):
                Threads=[0]
            fd=sys.stdout
            return None,fd,Threads
    else:
        if(Threads is None):
            Threads=[0]
        fd=open(fpath,('a' if(Append) else 'w'))
        return fpath,fd,Threads
def np_ravel(src):
    return np.ravel(np.array(src))
def bcast_ndarray(buf,barrier=True,key="",root=0,iverbose=3,check_diff=None,details=False):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size() 

    if( not MPIutils01.Multithreading() ):
        MPIutils01.DebugLog("!W:MPI is off:eldyn.bcast_ndarray")
        assert False,""

    string = "#eldyn.comm_Bcast:%02d:"%(MPIrank)+str(key)+"....."+\
             "  \t\t"+str(np.shape(buf))+"  "+str( (np.array(buf)).dtype )
    fnme_format='mpi_Bcast_%02d.log'
    Append = True; flush=True ## XXX XXX
    printout(string,fnme_format=fnme_format,Type='mpi',Append=Append,flush=flush, dtme=True, stdout=(iverbose>1) )

    cpy=None
    if( (check_diff is not None) and (MPIrank != root) ):
        cpy=arrayclone(buf)
   
    if( barrier ):
        comm_Barrier(key,iverbose=iverbose)


    ## strlog="#do_sync:%s:%02d:shape:"+str(np.shape(buf))
    ## if( iverbose > 1 ):
    ##     printout(strlog+" start...  \t\t "+str(datetime.datetime.now()),flush=(iverbose>2))
    comm.Bcast(buf,root=root)

    if( (check_diff is not None) and (MPIrank != root) ):
        if( cpy is not None ):
            sqrediff,maxabsdiff,At,vals=aNdiff(buf,cpy)
            fnme="checksync_"+key+"_%02d-%02d.dat"%(MPIrank,root);
            if( "fnme_format" in check_diff ):
                if(check_diff["fnme_format"] is not None):
                    fnme=check_diff["fnme_format"]%(MPIrank,root)
            
            fd1=open(fnme,'a');header=""
            if( "description" in check_diff ):
                header=check_diff["description"]+":"
            print( header+"%16.8f  %16.6e (%s/%s) at "%(sqrediff,maxabsdiff,str(vals[0]),str(vals[1]))+str(At),file=fd1)
            fd1.close();
            if(details):
                fnme2="checksync_"+key+"_%02d-%02d_details.dat"%(MPIrank,root)
                print_aNbufx2(cpy,buf,fpath=fnme2, Threads=[MPIrank],Append=False, Nd=5, half=False,description="")
            cpy=None

    string="#eldyn.comm_Bcast:%02d:done"%(MPIrank)+str(key)+" done "
    printout(string,fnme_format=fnme_format,Type='mpi',Append=Append,flush=flush, dtme=True, stdout=(iverbose>1) )
    
    ## if( iverbose > 1 ):
    ##    printout(strlog+" done  \t\t "+str(datetime.datetime.now()),flush=(iverbose>2))
def comm_Barrier(key="",iverbose=3):
    mpi_Barrier(("eldyn_comm_Barrier" if(key=="") else key))

def Magnus_AOrep(Ssqrt,Sinvrt,Ld,H2,H1,dt,C_in):
    ## Magnus propagator :  Heff = 0.5( H(t1) + H(t2) )+ (np.sqrt(3)/12)dt 
    # if( "propagator" in rttddft_common.params ):
    #     assert rttddft_common.params["propagator"]=="Magnus",""

    ret= np.matmul( Sinvrt, Magnus_step(Ld, np.matmul( Sinvrt, np.matmul(H2, Sinvrt) ),
                                              np.matmul( Sinvrt, np.matmul(H1, Sinvrt) ),dt, np.matmul(Ssqrt,C_in) ) )
    
    return ret
def Magnus_step(Ld,H2,H1,dt,C_in):
    sqrt3_over6=0.28867513459481288225457439025098
    sqrt3_over12=0.14433756729740644112728719512549
    DH=H2-H1
    Ndim=np.shape(C_in);nAO=Ndim[0];nMO=Ndim[1]
    fac1=0.5 - sqrt3_over6; H_t1=H1 + DH*fac1
    fac2=0.5 + sqrt3_over6; H_t2=H1 + DH*fac2
    Heff=np.zeros( [ Ld, Ld], dtype=np.complex128 )
    fac=0.0
    Heff= 0.50*( H_t1 + H_t2 ) -1j*dt*sqrt3_over12*( np.matmul(H_t2,H_t1) - np.matmul(H_t1,H_t2) ) 

    ### print("Magnus_commutator:",end="");print( dt*sqrt3_over12*( np.matmul(H_t2,H_t1) - np.matmul(H_t1,H_t2) ) )
    ### print("Magnus_FULL",end="");print( 0.50*( H_t1 + H_t2 ) -1j*dt*sqrt3_over12*( np.matmul(H_t2,H_t1) - np.matmul(H_t1,H_t2) ))
    ### print("Magnus_Heff",end="");print(Heff)
    eng,vecs=np.linalg.eig(Heff)
    ### print(np.array(eng).dtype);assert False,""
    expfacs=np.zeros([Ld],np.complex128)
    for j in range(Ld):
        arg=eng[j].real*dt; expfacs[j]=math.cos(arg) - 1j*math.sin(arg)
    U=np.zeros( [Ld,Ld], np.complex128 )
    for ia in range(Ld):
        for jm in range(Ld):
            cum=0.0
            for k in range(Ld):
                cum+= vecs[ia][k] * expfacs[k]* np.conj( vecs[jm][k] )
            U[ia][jm]=cum
    ### print("Magnus_propagation:%f+j%f ..."%(C_in[0][0].real,C_in[0][0].imag));
    ret=np.matmul( U, C_in )
    ### print("Magnus_propagation:%f+j%f ..."%(ret[0][0].real,ret[0][0].imag));
    return ret
