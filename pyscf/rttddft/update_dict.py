import numpy as np
import math
import os
import os.path
import sys
import time
import datetime
from .Loglv import Loglv,printout
from mpi4py import MPI
'''
from .update_dict import printout_dict,update_dict
from .pyscf_common import pyscf_common

fncnme='get_j_kpts';depth=2   ## depth ~ SCF:0 vhf:1 get_j:2
wct000=time.time();wct010=wct000;dic1={}
Dic=pyscf_common.Dict_getv('timing_'+fncnme, default=None)
if( Dic is None ):
    pyscf_common.Dict_setv('timing_'+fncnme, {});
    Dic=pyscf_common.Dict_getv('timing_'+fncnme, default=None)

wct020=wct010;wct010=time.time()
update_dict(fncnme,dic1,Dic,"setup",wct010-wct020,depth=depth)
...
printout_dict(fncnme,dic1,Dic,wct010-wct000,depth=depth)


'''
def update_dict_(dic1,Dic,key,val,append=False,iverbose=1,depth=0):
    
    if(not append):
        dic1.update({key:{'sum':val,'sqrsum':val**2,'count':1}})
    else:
        if( key not in dic1 ):
            dic1.update({key:{'sum':val,'sqrsum':val**2,'count':1}})
        else:
            dic1[key]['sum']+=val
            dic1[key]['sqrsum']+=val**2
            dic1[key]['sum']+=val

    if( key not in Dic ):
        Dic.update({key:{"sum":0.0,"sqrsum":0.0,"count":0}})

    Dic[key]['sum']+=val
    Dic[key]['sqrsum']+=val**2
    Dic[key]['count']+=1
    return Dic[key]['count']

def avg_dict_(dic,key):
    if( key not in dic ):
        return 0.0,0.0,0
    ### print("#avg_dict_:"+key+str(dic[key]))
    if( dic[key]['count']==0 ):
        return 0.0,0.0,0
    else:
        av=dic[key]['sum']/dic[key]['count']
        stdv=np.sqrt( abs( dic[key]['sqrsum']/dic[key]['count'] - av**2 ))
    return av,stdv,dic[key]['count']

def suppress_prtout():
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();
    return ( MPIrank!=0 )

def update_dict(fnc,dic1,Dic,key,val,append=False,iverbose=1,redundancy=None,depth=0):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD; rank = comm.Get_rank()
    Key=fnc+'.'+key
    update_dict_(dic1,Dic,Key,val,append=append,iverbose=iverbose,depth=depth)
    loglv=Loglv.loglv('timing')
    if( redundancy is None ): redundancy=1;
    prtout=(redundancy<(loglv+iverbose-1))
    if( Dic[Key]['count'] == 1 ):
        prtout=(redundancy<(loglv+iverbose))
    prtout=( prtout and ( not suppress_prtout() ) )

    if( prtout ):
        string='  '*depth + "#%03d:%s.%s:%14.4f"%(rank,fnc,key,val)
        fdF=open('walltime_%03d.log'%(rank),'a')
        av,stdv,count=avg_dict_(Dic,Key)
        string+="        \t\t\t  %14.4f %12.2e %3d"%(av,stdv,count)
        for fd in [ fdF, sys.stdout ]:
            print(string,file=fd)
        fdF.close()
    return Dic[Key]['count']

def printout_dict(fnc,dic1,Dic,val,depth=0,force=False,redundancy=1):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD; rank = comm.Get_rank()
    n=update_dict_(dic1,Dic,fnc,val,iverbose=0,depth=depth)

    loglv=Loglv.loglv('timing')
    prtout=(redundancy < loglv )
    if( n==1 or n==2 or n==10):
        prtout=(redundancy < loglv+1 )
    
    prtout=( prtout and ( not suppress_prtout() ) )
    if(force):
        prtout=True
    # fd1=open("prtout_dict.log","a")
    # print("%s %d %d %d %r %f"%(fnc,n,redundancy,loglv,prtout,val),file=fd1);
    # fd1.close()
    if( not prtout ):
        return n

    nth=Dic[fnc]['count']
    string='  '*depth + "#%03d:%s:%4d %14.4f"%(rank,fnc,nth,val)
    cum=0.0
    str1=""
    for Ky in dic1:
        ky=Ky.replace(fnc+".","")
        if(ky==fnc):
            continue
        str1+="%s:%12.4f "%(ky,dic1[Ky]['sum'])
    string+='{'+str1+'}'
    fdF=open('walltime_%03d.dat'%(rank),'a')
    for fd in [ fdF, sys.stdout ]:
        print(string,file=fd)
    fdF.close()
    return n
