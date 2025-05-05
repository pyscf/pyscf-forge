import numpy
import datetime
import sys
import math
from mpi4py import MPI
from .Loglv import Loglv,printout

class mpi_utils_:
    Counter={}
    @staticmethod
    def Countup(key,inc=True):   ## by default 1,2,3,... 
        if( key not in mpi_utils_.Counter ):
            if( not inc ):
                return 0
            else:
                mpi_utils_.Counter.update({key:0})
        if(inc):
            mpi_utils_.Counter[key]+=1
        return mpi_utils_.Counter[key]

# def mpi_print :: see Loglv.py#printout
# 
def mpi_Barrier(key,logfile=None,stdout=False,Append=True): ## False XXX XXX
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank()
    MPIsize=comm.Get_size()
    if( MPIsize<2 ):
        return
    stdout = (stdout and (MPIrank==0) )
    string="#mpi_Barrier:"+str(key)+":%02d ..."%(MPIrank)+str(datetime.datetime.now());
    fnme_format='mpi_Bcast_%02d.log';fpath=None
    printout(string,fnme_format=fnme_format,fpath=fpath,Type='mpi',dtme=True, stdout=stdout )
    # if( logfile is not None ):
    #    fdOU=open(logfile+"_%02d"%(MPIrank),('a' if(Append) else 'w'))
    #     print(string,file=fdOU);fdOU.close()   ### mth-adapted
    # dbgng_Barrier=True
    # logfiles=[ 'mpi_Bcast_%02d.log'%(MPIrank) ] ### , 'mpi_Bcast.log']
    # for f in logfiles:
    #    fdOU=open(f,('a' if(Append) else 'w'))
    #    print(string,file=fdOU);fdOU.close()   
    # if( stdout and MPIrank==0 ):
    #     print(string,flush=True)

    comm.Barrier()
    
    string="#mpi_Barrier:"+str(key)+":%02d leaving "%(MPIrank)+str(datetime.datetime.now());
    printout(string,fnme_format=fnme_format,fpath=fpath,Type='mpi',dtme=True, stdout=stdout )
    # if( logfile is not None ):
    #     fdOU=open(logfile+"_%02d"%(MPIrank),'w')
    #     print(string,file=fdOU);fdOU.close()
    # for f in logfiles:
    #     fdOU=open(f,('a' if(Append) else 'w'))
    #     print(string,file=fdOU);fdOU.close()
    # if( stdout and MPIrank==0 ):
    #     print(string,flush=True)

def mpi_Bcast(key,buf,root=None, barrier=True, iverbose=1, Append=False):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank()
    MPIsize=comm.Get_size()
    N_call= mpi_utils_.Countup("Bcast."+key)
    string = "#comm_Bcast:%02d:"%(MPIrank)+str(key)+"....."+\
             "  \t\t"+str(numpy.shape(buf))+"  "+str( (numpy.array(buf)).dtype )
    fnme_format='mpi_Bcast_%02d.log'
    printout(string,fnme_format=fnme_format,Type='mpi',Append=Append, dtme=True, stdout=(iverbose>1) )
    if( barrier ):
        comm.Barrier()

    if( root is not None ):
        comm.Bcast(buf,root=root)
    else:
        comm.Bcast(buf)
    string="#comm_Bcast:%02d:done"%(MPIrank)+str(key)+" done "
    printout(string,fnme_format=fnme_format,Type='mpi',Append=Append, dtme=True, stdout=(iverbose>1) )
    
def mpi_Bcast_old(key,buf,root=None, barrier=True, iverbose=1, Append=False):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank()
    MPIsize=comm.Get_size()
    N_call= mpi_utils_.Countup(key)
    # print("#%02d:mpiutils.mpi_Bcast:%s:Ncall=%d"%(MPIrank,str(key),N_call))
    # if( N_call<3 or N_call==10 or N_call==100):
    #    iverbose+=1
    # prtout = ( iverbose > 1 )

    loglv=Loglv.loglv('mpi')
    redundancy=1
    fdA=[];fdOU=None;### fdOU2=None
    if( redundancy <= loglv ):
        fdOU=open('mpi_Bcast_%02d.log'%(MPIrank), ('a' if(Append) else 'w') );fdA.append(fdOU);
        ### fdOU2=open('mpi_Bcast.log', ('a' if(Append) else 'w') );fdA.append(fdOU2);
        if( MPIrank==0 ):
            fdA.append(sys.stdout)
    for fd in fdA:
        if( (root is None) or (root != MPIrank) ):
            print("#comm_Bcast:%02d:"%(MPIrank)+str(key)+"....."+ str(datetime.datetime.now()),file=fd)
        else:
            print("#comm_Bcast:root:%02d:"%(MPIrank)+str(key)+"....."+\
                  "  \t\t"+str(numpy.shape(buf))+"  "+str( (numpy.array(buf)).dtype )
                   + str(datetime.datetime.now()),file=fd)
    if(fdOU is not None):
        fdOU.close()
    ### if(fdOU2 is not None):
    ###     fdOU2.close()

    if( barrier ):
        comm.Barrier()

    if( root is not None ):
        comm.Bcast(buf,root=root)
    else:
        comm.Bcast(buf)

    fdA=[];fdOU=None;### fdOU2=None
    if( redundancy <= loglv ):
        fdOU=open('mpi_Bcast_%02d.log'%(MPIrank),('a' if(Append) else 'w') );fdA.append(fdOU);
        ### fdOU2=open('mpi_Bcast.log', ('a' if(Append) else 'w') );fdA.append(fdOU2);
        if( MPIrank==0 ):
            fdA.append(sys.stdout)
    for fd in fdA:
        if( (root is None) or (root != MPIrank) ):
            print("#comm_Bcast:%02d:done"%(MPIrank)+str(key)+" done "+ str(datetime.datetime.now()) ,file=fd)
        else:
            print("#comm_Bcast:root:%02d:done"%(MPIrank)+str(key)+" done"+
                  "  \t\t"+str(numpy.shape(buf))+"  "+str( (numpy.array(buf)).dtype )
                   + str(datetime.datetime.now()),file=fd )
    if( fdOU is not None ):
        fdOU.close()
    ### if(fdOU2 is not None):
    ###    fdOU2.close()
def mpi_dbgtrace(text,buf=None,redundancy=1,iverbose=1):
    printout(text, dtme=True,stdout=True,
             Type='mpi',redundancy=redundancy-(iverbose-1), warning=(1 if(redundancy<0) else 0) )

def mpi_dbgtrace_old(text,buf=None,redundancy=1,iverbose=1):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPIsize = comm.Get_size()
    MPIrank = comm.Get_rank()

    loglv=Loglv.loglv('mpi')
    if( redundancy >= loglv+(iverbose-1) ):
        return

    string="#mpi_dbgtrace:%03d:"%(MPIrank)+text
    if( buf is not None):
        string+=" buf:"+str(type(buf))+" "+str(buf.dtype)+" "+str(numpy.shape(buf))
        bf=numpy.ravel(buf);le=len(bf);str1=""
        nprt=min(le,10)
        if( bf.dtype==float or bf.dtype==numpy.float64 ):
            for j in range(nprt):
                str1+="%14.6f "%(bf[j])
            if(le>nprt):
                str1+=" ... [%d-1]:"%(le)+"%14.6f"%(bf[le-1])
        else:
            for j in range(nprt):
                str1+=str(bf[j])+" "
            if(le>nprt):
                str1+=" ... [%d-1]:"%(le)+str(bf[le-1])
        string+=str1
    
    fdO=open("mpi_dbgtrace_%02d.log"%(MPIrank),"a");
    for fd in [fdO,sys.stdout]:
        print(string,file=fd,flush=True);
    fdO.close()
