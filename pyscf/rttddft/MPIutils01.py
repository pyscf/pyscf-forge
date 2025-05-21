from mpi4py import MPI
from .pyscf_common import pyscf_common
import datetime
import time
import math
import numpy as np

##
## since 2021.Nov  
##  controls parallelization 
##  2021.12.25 renamed from MPIutils -> MPIutils01
##
class MPIutils01:
    initialized_=False
    size_=None;rank_=None
    multithreading_=False
    stop_parallel_=None
    logfpath_=None

    @staticmethod
    def logger(content,dtme=True,Threads=[0]):  ## set Threads=None to force print out 
        if( Threads is not None ):
            if( MPIutils01.rank_ not in Threads ):
                return
        else:
            content="#$%02d:"%(MPIutils01.rank_)
        if( MPIutils01.logfpath_ is None ):
            MPIutils01.logfpath_= ("" if(pyscf_common.job is None) else pyscf_common.job)+"_MPIutils01.log"
            fd=open(MPIutils01.logfpath_,"w");
        else:
            fd=open(MPIutils01.logfpath_,"a");
        if( dtme ):
            content=content + " \t\t "+str(datetime.datetime.now())
        print(content,file=fd);fd.close()

        
    @staticmethod
    def Multithreading():
        return MPIutils01.multithreading_
    @staticmethod
    def DebugLog(content,warning=0):
        if( warning == 0 ):
            MPIutils01.logger(content)
        else:
            MPIutils01.logger(content, Threads=None)

    @staticmethod
    def DebugTrace(key,content,warning=0):
        AUinFS=2.418884326058678e-2
        if( (pyscf_common.step_ is not None) and (pyscf_common.time_AU_ is not None) ):
            content+=" step:%6d %12.4f %12.4f"%(pyscf_common.step_, pyscf_common.time_AU_, \
                                                pyscf_common.time_AU_*AUinFS)
        if( warning == 0 ):
            MPIutils01.logger("#DebugTrace:"+key+":"+content,dtme=True)
        else:
            MPIutils01.logger(content, Threads=None)
    @staticmethod
    def setup(multithreading=True):
        if( MPIutils01.initialized_ ):
            return
        comm = MPI.COMM_WORLD
        MPIutils01.size_ = comm.Get_size()
        MPIutils01.rank_ = comm.Get_rank()
        MPIutils01.multithreading_ = ( multithreading and (MPIutils01.size_>1) )
        print("#MPIutils01.$%02d:setup:%d %d %r"%( MPIutils01.rank_,MPIutils01.rank_,MPIutils01.size_,MPIutils01.multithreading_))
        return MPIutils01.rank_, MPIutils01.size_
    @staticmethod
    def stop_parallel(reason="",barrier=True):
        if( not MPIutils01.multithreading_ ):
            return
        if( barrier ):
            comm = MPI.COMM_WORLD
            comm.Barrier()

        MPIutils01.stop_parallel_=reason;
        MPIutils01.multithreading_=False
        MPIutils01.logger("stop_parallel"+str(reason),Threads=None)
    @staticmethod
    def restart_parallel(reason="",barrier=True):
        if( MPIutils01.multithreading_ ):
            return
        if( barrier ):
            comm = MPI.COMM_WORLD
            comm.Barrier()
        reason_stop=MPIutils01.stop_parallel_
        MPIutils01.stop_parallel_=None;
        MPIutils01.multithreading_=True
        MPIutils01.logger("restart_parallel"+str(reason_stop)+"->",str(reason),Threads=None)

            
