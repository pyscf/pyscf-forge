from .pyscf_common import pyscf_common
from mpi4py import MPI
import datetime
import sys

# SUPER           THIS     SUBORDINATES
#                          --- never import -- 
# pyscf_common <- Loglv <- mpiutils
#                       <- update_dict
# 
class Loglv:
    loglv_default_={'mpi':1,'dbg':1,'default':1}
    loglv_keyset=['mpi','dbg','default']
    _Logfile_format={'format':None,'append':[]}  ## _Logfile_format%(MPIrank)
    _Logfile={'path':None,'append':None}
    _Print_datetime=False
    warnings_=[]
    loglv_=None  ## {'mpi':INT,'dbg':INT}
    @staticmethod
    def Set_Logfile(fpath,Append=False):
        if( Loglv._Logfile['path'] is not None ):
            if( fpath != Loglv._Logfile['path'] ):
                pyscf_common.Warning("Loglv._Logfile: %s is NOT overridden by %s"%(Loglv._Logfile['path'],fpath))
            return
        Loglv._Logfile['path']=fpath;Loglv._Logfile['append']=Append
    @staticmethod
    def Set_Logfileformat(format,Append=False):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank()
        MPIsize=comm.Get_size()
        if( Loglv._Logfile_format['format'] is not None ):
            if( fpath != Loglv._Logfile_format['format'] ):
                pyscf_common.Warning("Loglv._Logfile: %s is NOT overridden by %s"%(Loglv._Logfile_format['format'],format))
            return
            Loglv._Logfile_format['format']=format;Loglv._Logfile_format['append']=[]
            for I in range(MPIsize):
                Loglv._Logfile_format['append'].append(Append)
    @staticmethod
    def Logfile(buf=None,update=True):
        if( Loglv._Logfile['path'] is None ):
            return None
        if(buf is not None):
            buf.clear();buf.append(Loglv._Logfile['append'])
        if(update):
            Loglv._Logfile['append']=True
        return Loglv._Logfile['path']

    @staticmethod
    def Logfile_format(buf=None,update=True):
        if( Loglv._Logfile_format['format'] is None ):
            return None
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank()
        if(buf is not None):
            buf.clear();buf.append(Loglv._Logfile_format['append'][MPIrank])
        if(update):
            Loglv._Logfile_format['append'][MPIrank]=True
        return Loglv._Logfile['format']
    @staticmethod
    def loglv(key="dbg"):
        
        if( Loglv.loglv_ is None ):
            if( pyscf_common.params is not None ):
                if( 'loglv' in pyscf_common.params ):
                    dum=pyscf_common.params['loglv']
                    if( dum is not None ):
                        if( isinstance(dum,int) ):
                            Loglv.loglv_={'mpi':dum,'dbg':dum,'default':dum}
                        elif( isinstance(dum,dict) ):
                            Loglv.loglv_=Loglv.loglv_default_.copy();
                            for ky in dum:
                                Loglv.loglv_.update({ky:int(dum[ky])})
                        else:
                            assert False,":"+str(type(dum))
                else:
                    Loglv.loglv_=Loglv.loglv_default_.copy();
            else:
                Loglv.loglv_=Loglv.loglv_default_.copy();
        if(key in Loglv.loglv_):
            return Loglv.loglv_[key]
        else:
            dum="unknownkey:"+key
            if ( dum not in Loglv.warnings_ ):
                Loglv.warnings_.append(dum)
                print("#!W Loglv.loglv_:unknown key:"+str(key));
          
            return Loglv.loglv_['default'];

## printout : please do not use -file- option
def printout(text,end='\n',flush=False,
             Threads=None,fpath=None,fnme_format=None,
             Append=False,dtme=None,stdout=None,
             Type='default',redundancy=None, warning=None):
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank()
    MPIsize=comm.Get_size()

    loglv=Loglv.loglv(Type)
    Threads_default_stdout=[0]
    Threads_default_multifiles=[ j for j in range(MPIsize) ]
    Threads_default_singlefile=[0]

    buf=[]
    if( fnme_format is None ):
        fnme_format=Loglv.Logfile_format(buf); 
        if(fnme_format is not None):
            Append=buf[0]

    if( Threads is not None ):
        if( isinstance(Threads,str) ):
            if( Threads=='all' ):
                Threads=[ j for j in range(MPIsize) ]
    path=None; 
    if( fpath is not None):  ## (1) given path (assuming it as multithread-adapted filename) 
        threads=( Threads if(Threads is not None) else Threads_default_multifiles )
        path=fpath;   
    else:
        if( fnme_format is not None): 
            threads=( Threads if(Threads is not None) else Threads_default_multifiles )
            path=( fnme_format%(MPIrank) if(MPIrank in threads) else None ) ## (2) newly given format or (3) _Logfile_format
        else:
            threads=( Threads if(Threads is not None) else Threads_default_stdout )
            path = ( None if(MPIrank not in threads) else Loglv.Logfile( buf ) ) ## (4) _Logfile
            if( path is not None ):
                Append=buf[0]

    if(dtme is None):dtme=Loglv._Print_datetime
    if(dtme):
        text+=' \t\t'+str(datetime.datetime.now())

    if( redundancy is None ):
        redundancy=(0 if(path is not None) else 1)

    if( redundancy >= loglv and (warning is None)):
        return

##    threads=( Threads if( (Threads is not None) and (fpath is None) and (fnme_format is None) )\
##              else Threads_default_stdout )
    if( MPIrank not in threads ):
        return

    if( path is not None ):
        fd=open(path,('a' if(Append) else 'w'))
        print(text,file=fd);fd.close()
    if( warning is not None):
        severity=( warning if(isinstance(warning,int)) else 1 )
        pyscf_common.Print_warning(text,severity=severity);

    if( stdout is None ):
        stdout = (True if(path is None) else False) 
    if( not stdout ):
        return

    print(text,flush=flush,end=end)
    
def dbgtrace(text,flush=None,redundancy=1,key=None,dtm=False):
    loglv=Loglv.loglv('dbg')
    if( redundancy >= loglv ):
        return
    if( flush is None):
        flush=( redundancy<(loglv-1) )

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPIsize = comm.Get_size()
    MPIrank = comm.Get_rank()
    string="#dbgtrace:" + ("" if(key is None) else str(key)+":")
    
    if( MPIsize > 1 ):
        string+="%02d:"%(MPIrank)
    string+=text

    if( dtm ):
        string+=" \t\t"+str(datetime.datetime.now())
    print(string,flush=flush)

