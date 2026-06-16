import os
import sys
from mpi4py import MPI

class futils:
    _Buffer=[]
    @staticmethod
    def fopen_00(file,mode='r',buffering=-1,encoding=None,errors=None,newline=None,closefd=True,opener=None,default=None):
        comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
        if(MPIrank!=0):
            if(default is None):
                return None
            elif(isinstance(default,str) and default == "dummy"):
                return open("dummy.log",mode)
            else:
                return default;
        return futils.fopen(file,mode=mode,buffering=buffering,encoding=encoding,errors=errors,newline=newline,closefd=closefd,opener=opener)
    @staticmethod
    def fclose_00(fd):
        if( fd is None ):
            return
        elif( fd == sys.stdout ):
            return
        else:
            fd.close();return

    @staticmethod
    def fopen(file,mode='r',buffering=-1,encoding=None,errors=None,newline=None,closefd=True,opener=None):
        fd=open(file,mode=mode,buffering=buffering,encoding=encoding,errors=errors,newline=newline,closefd=closefd,opener=opener);
        futils._Buffer.append(file);
        futils.prtoutifnecs()
        return fd
    @staticmethod
    def prtoutifnecs():
        if( os.path.exists("futils.in") ):
            print("#prtoutifncs:%d"%( len(futils._Buffer) )+str(futils._Buffer))

    @staticmethod
    def fclose(fd):
        n=len(futils._Buffer)
        ok=False
        for j in range(n):
            if( fd.name==futils._Buffer[j] ):
                futils._Buffer.pop(j);ok=True;break
        if( not ok ):
            print("!E missing file:"+fd.name+" in list:"+str(futils._Buffer));
        n=len(futils._Buffer)
        ## if( n>2 ):
        ##    print("check Buffer:len=%d "%(n)+ str(futils._Buffer))
        fd.close()
        futils.prtoutifnecs()
