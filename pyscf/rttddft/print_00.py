import sys
from mpi4py import MPI
import datetime

## see also utils.py
def print_00(text,flush=False,file=sys.stdout,end='\n',dtme=False,warning=0,Threads=[0]):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if( isinstance(Threads,str) ):
        if( Threads.strip()=='all' ):
            Threads= range(MPIsize)
    if((MPIrank not in Threads) and (warning >=0) ):
        return
    if(dtme):
        text+=' \t\t'+str(datetime.datetime.now())
    if( file is None ):
        file=sys.stdout
    print(text,flush=flush,file=file,end=end)

def open_00(fpath,mode,default=None):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        if(default is None):
            return None
        elif(isinstance(default,str) and default == "dummy"):
            return open("dummy.log",mode)
        else:
            return default;
    return open(fpath,mode)

def close_00(fd):
    if( fd is None ):
        return
    elif( fd == sys.stdout ):
        return
    else:
        fd.close();return

