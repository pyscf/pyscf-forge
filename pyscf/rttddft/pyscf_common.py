import numpy as np
import sys
import os
import math
import os.path
import time
import datetime
from mpi4py import MPI

class Stack:
    def __init__(self,bfsz):
        self.buf=[]
        self.bfsz=bfsz
        self.o=0
        self.len=0
    def contains(self,item):
        return (item in self.buf)
    def append(self,item):
        if(self.len>=self.bfsz):
            self.buf[ self.o ]=item;
            self.o=(self.o+1)%self.bfsz
        else:
            self.buf.append(item); self.len+=1
        return self
    def toarray(self):
        return [ self.buf[(self.o+k)%self.bfsz] for k in range(self.len) ] 
    def clear(self):
        self.buf.clear();self.o=0;self.len=0
    def last(ith=0):   ## here we define  ith=0: last item   ith=1:last-1 ... 
        ##   o+0
        ## [ 0, 1 ... le-1 ]
        ##   le-1          0th
        if( ith >= self.len ):
            print("#Stack.last %d / %d"%(ith,self.len))
            return None
        else:
            indx= ( self.o+(self.len-1-ith) )%self.bfsz
            return self.buf[indx]
class Filewriter:
    def __init__(self,fpath, bfsz=10):
        self.fpath=fpath
        self._append=False
        self.strbuf=[];self.nbuf=0
        self.bfsz=bfsz
    def prtout(self):
        ### print("Logwriter:%s nbuf=%d printing out..."%(self.fpath,self.nbuf))
        fd=open(self.fpath,('a' if(self._append) else 'w'))
        retv=0
        for item in self.strbuf:
            print(item,file=fd);retv+=1
        fd.close(); self._append=True
        return retv
        
    def append(self,text,timing=False):
        self.strbuf.append(text); self.nbuf+=1
        ### print("Logwriter:%s nbuf=%d"%(self.fpath,self.nbuf))
        if( self.nbuf >=self.bfsz ):
            self.prtout()
            self.nbuf=0;self.strbuf.clear()
class pyscf_common:
    SCF_buf=[]; SCF_bfsz=5; n_SCF=0; 
    SCF_warnings={}  # key:{'count':0,'i_SCF':[ 5 most recent occurrence ] }
    dic={}
    params=None
    job=None
    _Time_000=None;wt_010={}
    Key=None;KeyHistory=None;Logwriter=None;Timing={}
    _iVerbose=2
    _Counter={}
    _Dict={}
    _Keys_Write_once=[]
    _Warning_logfile=None
    _Filenames=[]
    
    time_AU_=None
    step_=None
    # 2021.12.31 ---------------------
    Rnuc_=None
    Dict_Print_warning_={}

    Dict_Clipboard={} 
    @staticmethod
    def Clipboard(key,value=None,default=None,clear=None,set_value=False):
        if(value is not None or set_value ):
            pyscf_common.Dict_Clipboard.update({key:value})
        else:
            if(key in pyscf_common.Dict_Clipboard ):
                if( clear is not None):
                    if(clear):
                        return pyscf_common.Dict_Clipboard.pop(key)
                return pyscf_common.Dict_Clipboard[key]
            else:
                print("!W key:%s is not in pyscf_common.Dict_Clipboard"%( str(key) ))
                return default

    @staticmethod
    def Printout_warning(key,content,level=-1, nth_call=[1], dtme=True, Threads=[0], fpath=None, Append=None):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()

        isempty=( len(pyscf_common.Dict_Print_warning_)==0 )
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( key not in pyscf_common.Dict_Print_warning_ ):
            pyscf_common.Dict_Print_warning_.update({key:0});
        pyscf_common.Dict_Print_warning_[key]+=1
        if( dtme ):
            content+=" \t\t "+str(datetime.datetime.now())
        nth=pyscf_common.Dict_Print_warning_[key]

        logfpath0=pyscf_common.job + "_Print_warning.log"
        Fpaths=[ logfpath0 ];Flags=[ ("w" if(isempty) else "a") ]
        if(fpath is not None):
            if( Append is None ):
                Append=True
            Fpaths.append(fpath);Flags.append("a" if(Append) else "w") 

        if( MPIsize < 2 or MPIrank in Threads ):
            Nf=len(Fpaths)
            if( nth in nth_call ):
                for I in range(Nf):
                    fdOU=open( Fpaths[I], Flags[I] )
                    print("#%s:"%(key)+content,file=fdOU)
                    fdOU.close()
                print("#%s:"%(key)+content)
         
                if( isempty and abs(level)>1 ):
                    os.system("fopen "+logfpath0)
    @staticmethod
    def Check_cellRnuc(cell, caller=""):
        Rnuc_au,Sy=parse_xyzstring(cell.atom,unit_BorA='B')
        return pyscf_common.Check_Rnuc(Rnuc_au, caller)
    @staticmethod
    def Check_Rnuc(testee, caller=""):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( pyscf_common.Rnuc_ is None ):
            return None
        dev = max( np.ravel( pyscf_common.Rnuc_ ) - np.ravel( testee ) )
        dbgng=True  ## XXX XXX
        prtout=( dbgng or dev>1.0e-6 )
        if( prtout ):

            path="Set_Rnuc_%02d.log"%(MPIrank)
            fdOU=open(path,"a")
            if( dev < 1e-6 ):
                print("###Check:%s:$%02d: %e"%(caller,MPIrank,dev),file=fdOU)
            else:

                Nat=len( testee )
                diffs=[ np.sqrt( (testee[I][0]-pyscf_common.Rnuc_[I][0])**2 +
                                 (testee[I][1]-pyscf_common.Rnuc_[I][1])**2 + 
                                 (testee[I][2]-pyscf_common.Rnuc_[I][2])**2 ) for I in range(Nat) ]
                strbuf=""
                for I in range(Nat):
                    strbuf+=" %12.6f %12.6f %12.6f    /   %12.6f %12.6f %12.6f   %e \n"%(
                        testee[I][0],testee[I][1],testee[I][2], 
                        pyscf_common.Rnuc_[I][0],pyscf_common.Rnuc_[I][1],pyscf_common.Rnuc_[I][2], diffs[I])
                
                strbuf+=" \t\t "+str(datetime.datetime.now())
                print("#Check:%s:$%02d:!W: %s"%(caller,MPIrank,strbuf),file=fdOU)
            fdOU.close()
        return dev
    @staticmethod
    def Set_Rnuc(Rnuc,step=None,time_AU=None,Logging=[0],sync=False ):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        AUinFS=2.418884326058678e-2

        strRnuc=""
        Nat=len(Rnuc)
        for I in range(Nat):
            strRnuc+=("" if(I==0) else ";")+"%12.6f,%12.6f,%12.6f"%(Rnuc[I][0],Rnuc[I][1],Rnuc[I][2])
        logfpath = "Set_Rnuc_%02d.log"%(MPIrank)
        if( sync ):
            print("#Set_Rnuc:Bcast..")
            comm.Bcast(Rnuc,root=0)
            fd1=open(logfpath,"a");
            strlog="#Bcast:"
            strlog+=strRnuc
            if( step is not None ):    strlog+=" step=%d"%(step);
            if( time_AU is not None ): strlog+=" time=%14.5f,%12.4f"%(time_AU,time_AU*AUinFS)
            strlog+=" \t\t "+str( datetime.datetime.now() )
            print(strlog,file=fd1)
            fd1.close()

        diffs=None
        if( pyscf_common.Rnuc_ is not None ):
            Nat=len( pyscf_common.Rnuc_ )
            diffs=[ np.sqrt( (Rnuc[I][0]-pyscf_common.Rnuc_[I][0])**2 +
                             (Rnuc[I][1]-pyscf_common.Rnuc_[I][1])**2 + 
                             (Rnuc[I][2]-pyscf_common.Rnuc_[I][2])**2 ) for I in range(Nat) ]
            dev=max( diffs )

        if( Logging is not None ):
            if( MPIrank in Logging ):
                path="Set_Rnuc_%02d.log"%(MPIrank)
                fdOU=open(path,"a")
                strbuf=""
                strbuf +=strRnuc
                if( step is not None ):    strbuf+=" step=%d"%(step);
                if( time_AU is not None ): strbuf+=" time=%14.5f,%12.4f"%(time_AU,time_AU*AUinFS)
                strbuf+=" \t\t "+str( datetime.datetime.now() )
                print(strbuf,file=fdOU);fdOU.close()
                print("#Set_Rnuc:"+strbuf, diffs)


        Ndim=np.shape(Rnuc)
        if( step is not None ):
            pyscf_common.step_=step
        if( time_AU is not None ):
            pyscf_common.time_AU_=time_AU
        print("#pyscf_common.Rnuc:$%02d:set:"%(MPIrank),Rnuc,step,time_AU)
        rank=len(Ndim);assert rank==1 or rank==2,""
        pyscf_common.Rnuc_=np.zeros(Ndim,dtype=np.float64)
        if(rank==2):
            for I in range(Ndim[0]):
                for J in range(Ndim[1]):
                    pyscf_common.Rnuc_[I][J]=Rnuc[I][J]
        else:
            for I in range(Ndim[0]):
                pyscf_common.Rnuc_[I]=Rnuc[I]

    @staticmethod
    def Get_Rnuc(clone=False):
        if(not clone):
            return pyscf_common.Rnuc_

        Ndim=np.shape(pyscf_common.Rnuc_)
        rank=len(Ndim);assert rank==1 or rank==2,""
        ret=np.zeros(Ndim,dtype=np.float64)
        if(rank==2):
            for I in range(Ndim[0]):
                for J in range(Ndim[1]):
                    ret[I][J]=pyscf_common.Rnuc_[I][J]
        else:
            for I in range(Ndim[0]):
                ret[I]=pyscf_common.Rnuc_[I]
        return ret

    @staticmethod
    def Assert(bool,text,severity):
        if(bool):
            return 0
        pyscf_common.Print_warning(text,severity=severity)
        if( severity<0 ):
            assert False,""+text
    @staticmethod
    def Print_warning(text,severity=1):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( severity>=0 and MPIrank>0 ):
            return
        Append=True
        if( pyscf_common._Warning_logfile is None ):
            pyscf_common._Warning_logfile=("pyscf_" if(pyscf_common.job is None) else pyscf_common.job)\
                                          +"_warning.log"
            Append=False
        fd=open( pyscf_common._Warning_logfile,('a' if(Append) else 'w'))
        print(text+" \t\t"+str(datetime.datetime.now()),file=fd)
        fd.close()

    @staticmethod
    def Write_once(key,text,fpath=None,filename=None,Append=False,stdout=True):
        if( key in pyscf_common._Keys_Write_once ):
            return
        else:
            pyscf_common._Keys_Write_once.append(key)
        fdA=[];fdOU=None
        if(stdout or (fpath is None)):
            fdA.append(sys.stdout)

        if(fpath is not None):
            fdOU=open(fpath,('a' if(Append) else 'w'))
            fdA.append(fdOU)
        else:
            if(filename is not None):
                append_file=True
                if( filename not in pyscf_common._Filenames ):
                    pyscf_common._Filenames.append( filename );append_file=False
                fpath=pyscf_common.job+"_"+filename+".log"
                ## print("Write_once:%s %r"%(fpath,append_file))
                fdOU=open(fpath,('a' if(append_file) else 'w'))
                fdA.append(fdOU)
        for fd in fdA:
            print(key+":"+text,file=fd)
        if(fdOU is not None):
            fdOU.close()

    @staticmethod
    def Dict_setv(key,val):
        ### print("#pyscf_common:Dict_setv:%s:%s"%(str(key),str(val)))
        pyscf_common._Dict.update({key:val})
    @staticmethod
    def Dict_popv(key,default=None):
        print("#pyscf_common:Dict_popv:%s:%s..."%(
                str(key),("none" if(key not in pyscf_common._Dict) else str(pyscf_common._Dict[key])) ))
        if( key in pyscf_common._Dict ):
            return pyscf_common._Dict.pop(key)
        else:
            return default
    @staticmethod
    def Dict_getv(key,default=None):
        ### print("#pyscf_common:Dict_getv:%s:%s..."%(
        ###        str(key),("none" if(key not in pyscf_common._Dict) else str(pyscf_common._Dict[key])) ))
        if( key in pyscf_common._Dict ):
            return pyscf_common._Dict[key]
        else:
            return default

    ### Nat=None;nAO=None;nMO=None;nKpoints=None
    @staticmethod
    def Set_iVerbose(level):
        pyscf_common._iVerbose=level

    @staticmethod
    def Setup(bfsz=5):
        if( pyscf_common._Time_000 is None ):
            pyscf_common._Time_000=time.time()
        if( pyscf_common.KeyHistory is None ):
            pyscf_common.KeyHistory=Stack(bfsz)
        if( pyscf_common.Logwriter is None ):
            pyscf_common.Logwriter=Filewriter(\
                ("pyscf_common" if (pyscf_common.job is None) else pyscf_common.job)\
                + "_logger.log" )
    @staticmethod
    def Clock000(set_if_none=True):
        if( pyscf_common._Time_000 is not None ):
            return pyscf_common._Time_000
        else:
            if( not set_if_none ):
                return None
            else:
                pyscf_common._Time_000=time.time()
                return pyscf_common._Time_000

    @staticmethod
    def set_params(dict,job=None):
        def get1(dic,key,default=None):
            if(key in dic):
                return dic[key]
            else:
                return default;
        if( job is None):
            job= get1(dict, "name","pySCF_job") + "_" + get1(dict,"xc","") + get1(dict, "basis","") +  get1(dict,"branch","")
        pyscf_common.job=job
        pyscf_common.params=dict
        pyscf_common.Setup()

    @staticmethod
    def Get_time():
        return time.time()-pyscf_common._Time_000
##  def somefunc(...)
##    depth=1   # 1 : SCF,eldyn_singlestep etc. 
##    wt_010= pyscf_common.Start_timer("",depth=depth)
##    -- step1 --
##    pyscf_common.Switch_timer("","",depth=depth)
##    -- step2 --
##    pyscf_common.Stop_timer("",depth=depth)
##    pyscf_common.Start_timer("",depth=depth)
##    wt_100=pyscf_common.Get_time()
##    pyscf_common.Print_timing("", ["",'','',''],
##                              walltime=wt_100-wt_010,depth=depth)
    @staticmethod
    def get_MPIrank(format=None):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if( format is not None ):
            return format%(rank)
        return rank

    @staticmethod
    def get_MPIsize(format=None):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        if( format is not None ):
            return format%(size)
        return size


    @staticmethod
    def Start_timer(key,flush=False,depth=1):
        if( pyscf_common._Time_000 is None ):
            pyscf_common.Setup()
        pyscf_common.wt_010.update({key: time.time() - pyscf_common._Time_000})
        oldky=pyscf_common.Key
        if( oldky is not None ):
            pyscf_common.KeyHistory.append(oldky);
        pyscf_common.Key=key
        
        ### print("#Start_timer:"+str(oldky)+" >> current:"+str(pyscf_common.Key))
        if( flush or pyscf_common._iVerbose>1):
            header='#'*(2*depth)
            pyscf_common.Logwriter.append( header + "start:%24s  %14.4f "%(key, pyscf_common.wt_010[key])
                                          +" \t\t\t "+("" if(pyscf_common.job is None) else pyscf_common.job) 
                                          + " "+str(datetime.datetime.now())) 
        return pyscf_common.wt_010[key]
    @staticmethod
    def Switch_timer(key1,key2,flush=False,depth=1):
        ### print("#Switch_timer:BF:"+str(key1)+" /current:"+str(pyscf_common.Key))
        ret= pyscf_common.Stop_timer(key1,key2=key2,flush=flush,depth=depth)
        ### print("#Switch_timer:AF:"+str(key2)+" /current:"+str(pyscf_common.Key))
        return ret
    @staticmethod
    def Stop_timer(key,key2=None,flush=False,depth=1,subkey=""):
        ### print("#stop_timer:"+key+" current:"+str(pyscf_common.Key))
        wctNow=time.time() - pyscf_common._Time_000
#        if( key == pyscf_common.Key ):
        if( key in pyscf_common.wt_010 ):
            if( key in [ 'get_veff', 'get_j'] ):
                key1=key+"_1st"
                if( key1 not in pyscf_common.Timing):
                    pyscf_common.Update_timing( key1, wctNow-pyscf_common.wt_010[key], flush=flush)
                else:
                    key2=key+"_2nd";tCUR=wctNow-pyscf_common.wt_010[key]
                    key_to_upd=key2
                    if( key2 in pyscf_common.Timing):
                        av   = pyscf_common.Timing[key2]['sum']/pyscf_common.Timing[key2]['count']
                        stdv = np.sqrt( abs( pyscf_common.Timing[key2]['sqrsum']/pyscf_common.Timing[key2]['count'] - av**2 ) )
                        av_1   = pyscf_common.Timing[key1]['sum']/pyscf_common.Timing[key1]['count']
                        stdv_1 = np.sqrt( abs( pyscf_common.Timing[key1]['sqrsum']/pyscf_common.Timing[key1]['count'] - av_1**2 ) )
                        if( abs(av - tCUR)> max(1.0, stdv)*3 ):
                            if( abs(av_1-tCUR)<abs(av-tCUR) ):
                                key_to_upd=key1
                            print("#get_Veff:%s:time %f / get_Veff_2nd av=%f(%f)  av1=%f(%f)  ... %s"%(subkey,tCUR,av,stdv,av_1,stdv_1,key_to_upd))
                    pyscf_common.Update_timing( key_to_upd, wctNow-pyscf_common.wt_010[key], flush=flush)

            else:
                pyscf_common.Update_timing( key, wctNow-pyscf_common.wt_010[key], flush=flush)

            if( flush or pyscf_common._iVerbose>1):
                header='#'*(2*depth)
                pyscf_common.Logwriter.append( header + "END:%24s  %14.4f  %14.4f"%(key, wctNow-pyscf_common.wt_010[key], wctNow)
                                              +" \t\t\t "+("" if(pyscf_common.job is None) else pyscf_common.job) 
                                              + " "+str(datetime.datetime.now())) 

            pyscf_common.Key=None; pyscf_common.wt_010.pop(key,None)
        else:
            pyscf_common.Logwriter.append("## key:%s did not end  tmNow:%f"%(key,wctNow))
            ### pyscf_common.Key=None; pyscf_common.wt_010=None
            if( flush or pyscf_common._iVerbose>1):
                header='#'*(2*depth)
                pyscf_common.Logwriter.append( header + "END:%24s  %14s  %14.4f"%(key, "--", wctNow)
                                              +" \t\t\t "+("" if(pyscf_common.job is None) else pyscf_common.job) 
                                              + " "+str(datetime.datetime.now())) 
        if( key2 is None ):
            return wctNow
        else:
            ### print("#Switching to:"+str(key2))
            return pyscf_common.Start_timer(key2)
    @staticmethod
    def Update_timing(key,walltime,flush=False,doNotPrtOut=False):
        if( key not in pyscf_common.Timing ):
            pyscf_common.Timing.update({key:{'count':0,'sum':0.0,'sqrsum':0.0}})
        pyscf_common.Timing[key]['count']+=1
        pyscf_common.Timing[key]['sum']+=walltime
        pyscf_common.Timing[key]['sqrsum']+=walltime**2
        ncount=pyscf_common.Timing[key]['count']
        ### print("#pyscf_common.Update_timing:%s:%d"%(key,ncount))
        if( not doNotPrtOut ):
            if( ncount == 1 or ncount==10 or  ncount==20 or ncount == 100 or  ncount== 200 or ncount==1000 ):
                avg= pyscf_common.Timing[key]['sum'] / pyscf_common.Timing[key]['count']
                stdv=np.sqrt( abs( pyscf_common.Timing[key]['sqrsum'] / pyscf_common.Timing[key]['count'] - avg**2 ) )
                pyscf_common.Logwriter.append("#AvgTime:%24s  %14.4f  %14.4e  %d   %14.4f  avg/stdv/count/sum"%(key,avg,stdv,
                             pyscf_common.Timing[key]['count'], pyscf_common.Timing[key]['sum']))
        return pyscf_common.Timing[key]['count']
    @staticmethod
    def Print_timing(name,keys,walltime=None,flush=False,depth=1,N_iter=None,subkey=""):
        ### print("#pyscf_common.print_timing..");
        if( N_iter is not None ):
            ky1=name+"_N_iter"
            pyscf_common.Update_timing(ky1,N_iter,flush=False,doNotPrtOut=False)
            if(ky1 not in keys):
                keys.append(ky1)
        if( name in [ 'get_veff', 'get_j', 'Get_Veff'] ):
            name1=name+"_1st"
            if( name1 not in pyscf_common.Timing):
                pyscf_common._Print_timing( name1, keys, walltime=walltime, flush=flush, depth=depth)
            else:
                name2=name+"_2nd"
                name_upd=name2
                if( name2 in pyscf_common.Timing ):
                    av   = pyscf_common.Timing[name2]['sum']/pyscf_common.Timing[name2]['count']
                    stdv = np.sqrt( abs( pyscf_common.Timing[name2]['sqrsum']/pyscf_common.Timing[name2]['count'] - av**2 ) )
                    av_1   = pyscf_common.Timing[name1]['sum']/pyscf_common.Timing[name1]['count']
                    stdv_1 = np.sqrt( abs( pyscf_common.Timing[name1]['sqrsum']/pyscf_common.Timing[name1]['count'] - av_1**2 ) )
                    if( abs(av - walltime)> max(1.0, stdv)*3 ):
                        ### ??? XXX XXX if( abs(av_1-walltime)<abs(av-walltime) ):
                        ### ??? XXX XXX    name_upd=key1   << note:key1 is undefined 
                        print("#get_Veff:%s:time %f / get_Veff_2nd av=%f(%f)  av1=%f(%f)  ... %s"%(subkey,walltime,av,stdv,av_1,stdv_1,name_upd))
                pyscf_common._Print_timing( name_upd, keys, walltime=walltime, flush=flush, depth=depth)
        else:
            pyscf_common._Print_timing( name, keys, walltime=walltime, flush=flush, depth=depth)
    @staticmethod
    def _Print_timing(name, keys, walltime=None, flush=False, depth=1):
        fpath=("pyscf_common" if(pyscf_common.job is None) else pyscf_common.job)+"_"+name+"_walltime.log"
        wctNow=time.time() - pyscf_common._Time_000
        ### print("#pyscf_common._Print_timing..%f"%(wctNow) +fpath);
        
        ncount=None;keyAL=None
        if(walltime is not None):
            ncount=pyscf_common.Update_timing( name,walltime,doNotPrtOut=True)
            keyAL=[name]
            for ky in keys:
                keyAL.append(ky)
            ### print("#pyscf_common._Print_timing.1: %s %d"%(name,ncount))
        else:
            ncount=99999
            for ky in keys:
                if( ky in pyscf_common.Timing):
                    ncount=min(ncount,pyscf_common.Timing[key]['count'])
            keyAL=[]
            for ky in keys:
                keyAL.append(ky)
            ### print("#pyscf_common._Print_timing.2: %s %d"%(name,ncount))

        if( flush or pyscf_common._iVerbose>1):
            header='#'*(2*depth)
            strtm=("--" if(walltime is None) else "%14.4f"%(walltime))
            pyscf_common.Logwriter.append( header + "END:%24s  %14s  %14.4f"%(name, strtm, wctNow)
                                          +" \t\t\t "+("" if(pyscf_common.job is None) else pyscf_common.job) 
                                          + " "+str(datetime.datetime.now())) 

        if( ncount == 1 or ncount==10 or  ncount==20 or ncount == 100 or  ncount== 200 or ncount==1000 ):
            string="";legend="";ncol=0
            for key in keyAL:
                if( key not in pyscf_common.Timing ):
                    keyset=[ ky for ky in pyscf_common.Timing ]
                    print("#!W %s not in Timing:"%(key)+str(keyset));continue
                avg= pyscf_common.Timing[key]['sum'] / pyscf_common.Timing[key]['count'] 
                stdv=np.sqrt( abs( pyscf_common.Timing[key]['sqrsum'] / pyscf_common.Timing[key]['count'] - avg**2 ) )
                string+="   %18.4f %18.4e %8d %18.4f     "%(avg,stdv,
                         pyscf_common.Timing[key]['count'], pyscf_common.Timing[key]['sum'])
                legend+="   %4d:%24s_%24s_av,stdv,cnt,sum   "%(ncol+1,key,name)
                ncol+=4
            fdOu=open( fpath,('w' if(ncount==1) else 'a'))
            if( ncount==1 ):
                print(legend,file=fdOu);
            print(string,file=fdOu);
            fdOu.close()
        return ncount
    @staticmethod
    def Log(text,fnme=None,stdout=True,flush=False):
        if(fnme is not None):
            fd1=open(fnme,"a");print(text,file=fd1);fd1.close()
        if(stdout):
            print(text);
        
    @staticmethod
    def Countup(key,inc=True):   ## by default 1,2,3,... 
        if( key not in pyscf_common._Counter ):
            if( not inc ):
                return 0
            else:
                pyscf_common._Counter.update({key:0})
        if(inc):
            pyscf_common._Counter[key]+=1
        return pyscf_common._Counter[key]

    @staticmethod
    def count(key,inc=False):
        
        if(key in pyscf_common.dic):
            if(inc):
                pyscf_common.dic[key]=pyscf_common.dic[key]+1
            return pyscf_common.dic[key]
        else:
            ret=0
            if(inc):
                ret+=1
            pyscf_common.dic.update( {key:ret} )
            return ret;
    @staticmethod
    def Append_SCFwarning(item,i_SCF=None):
        if( i_SCF is None ):
            i_SCF = pyscf_common.n_SCF
        if( item not in pyscf_common.SCF_warnings ):
            pyscf_common.SCF_warnings.update({item:{'count':0,'i_SCF':Stack(5)}})
        pyscf_common.SCF_warnings[item]['count']+=1
        if( not pyscf_common.SCF_warnings[item]['i_SCF'].contains(i_SCF) ):
            pyscf_common.SCF_warnings[item]['i_SCF'].append(i_SCF)

    @staticmethod
    def Setup_SCFbuf():
        if( len(pyscf_common.SCF_buf)==0 ):
            for I in range(pyscf_common.SCF_bfsz):
                pyscf_common.SCF_buf.append({'iter':0, 'time':0, 'cvgd':None})
            pyscf_common.SCF_buf.append({'iter':{'sum':0.0,'sqrsum':0.0}, 'time':{'sum':0.0,'sqrsum':0.0} })
    @staticmethod
    def Get_SCFresult(nth=0):
        if( nth >= pyscf_common.SCF_bfsz):
            return None
        i_SCF= pyscf_common.n_SCF-1 - nth
        if( i_SCF < 0 ):
            return None
        iMOD=i_SCF%pyscf_common.SCF_bfsz
        return pyscf_common.SCF_buf[iMOD]

    @staticmethod
    def Set_SCFresult(cvgd, iter, Egnd, time, warnings=None,dbuf=None):
        import sys
        import datetime
        if( len(pyscf_common.SCF_buf)==0 ):
            pyscf_common.Setup_SCFbuf()
        pyscf_common.n_SCF+=1
        i_SCF= pyscf_common.n_SCF-1
        iMOD=i_SCF%pyscf_common.SCF_bfsz
        pyscf_common.SCF_buf[iMOD]['cvgd']=cvgd
        pyscf_common.SCF_buf[iMOD]['iter']=iter
        pyscf_common.SCF_buf[iMOD]['time']=time
        ### pyscf_common.SCF_buf[iMOD]['warnings'].clear()
        pyscf_common.SCF_buf[pyscf_common.SCF_bfsz]['iter']['sum']+=iter
        pyscf_common.SCF_buf[pyscf_common.SCF_bfsz]['iter']['sqrsum']+=iter**2

        pyscf_common.SCF_buf[pyscf_common.SCF_bfsz]['time']['sum']+=time
        pyscf_common.SCF_buf[pyscf_common.SCF_bfsz]['time']['sqrsum']+=time**2
        if( not cvgd ):
            fdOU=open('SCF_failure_%05d.log'%(i_SCF),'a')
            string="#SCF:%05d failed after %d iterations"%(i_SCF,iter)+" \t\t\t "+str(datetime.datetime.now())
            if( warnings is not None ):
                for item in warnings:
                    string+='\n##!W: '+str(item)
            for fd in [fdOU,sys.stdout]:
                print(string,file=fd)
            if( dbuf is not None ):
                le=len(dbuf);
                print("#%4s %16s %14s %14s"%('iter','Etot','|g|','ddm'),file=fdOU)
                for I in range(le):
                    print(" %4d %16.8f %14.6e %14.6e"%(I,dbuf[I][0],dbuf[I][1],dbuf[I][2]),file=fdOU)
            fdOU.close()

        if( warnings is not None ):
            for item in warnings:
                pyscf_common.Append_SCFwarning(item,i_SCF=i_SCF)
        if( pyscf_common.n_SCF == 1 or pyscf_common.n_SCF==5 or pyscf_common.n_SCF==10 or pyscf_common.n_SCF == 20 or 
            pyscf_common.n_SCF %100==0 ):
            fpath=("" if(pyscf_common.job is None) else pyscf_common.job)+"_SCF.log"
            pyscf_common.Print_SCFresults( fpath,Append=( pyscf_common.n_SCF>1 ))
    @staticmethod
    def Print_SCFresults(fpath=None,Append=False):
        fdA=[ sys.stdout ]
        fdout=(None if(fpath is None) else open(fpath,('a' if(Append) else 'w')))
        if(fdout is not None):
            fdA.append(fdout)
        Dic=pyscf_common.SCF_buf[pyscf_common.SCF_bfsz]
        strWarning=""
        for key in pyscf_common.SCF_warnings:
            wdict= pyscf_common.SCF_warnings[key]
            strWarning +=key+":%3d / %4d"%( pyscf_common.SCF_warnings[key]['count'],pyscf_common.n_SCF ) \
                            +str( pyscf_common.SCF_warnings[key]['i_SCF'].toarray() )
        if( len(strWarning.strip())>0 ):
            strWarning="  Warning:"+strWarning
        for fd in fdA:
            is_stdout=(fd==sys.stdout)
            av_iter=Dic['iter']['sum']/float(pyscf_common.n_SCF)
            stdv_iter=np.sqrt( abs( Dic['iter']['sqrsum']/float(pyscf_common.n_SCF) -av_iter**2 ) )

            av_time=Dic['time']['sum']/float(pyscf_common.n_SCF)
            stdv_time=np.sqrt( abs( Dic['time']['sqrsum']/float(pyscf_common.n_SCF) -av_iter**2 ) )
            if( (not Append) and (fdout is not None) and fd==fdout ):
                print("#%6s    %6s  %14s  %14s    %14s %14s %14s %14s "%("n_SCF",
                    "sum_iter","avg_iter","stdv_iter",
                    "sum_time","avg_time","stdv_time","per-iter-time"),file=fd)
            
            #                        ITER           TIME
            #                  n_SCF  sum  av  stdv          sum   av stdv  PER-ITER-time
            print("%s %6d    %6d  %14.6f  %14.4e    %14.6f %14.6f %14.4e %14.6f "%(
                    ("#SCFresults:" if(is_stdout) else ""), pyscf_common.n_SCF,
                    Dic['iter']['sum'], av_iter, stdv_iter,
                    Dic['time']['sum'], av_time, stdv_time, Dic['time']['sum']/float(Dic['iter']['sum']))
                  +strWarning,file=fd)
        if(fdout is not None):
            fdout.close()


def parse_xyzstring(str,unit_BorA):
    assert (unit_BorA=='B' or unit_BorA=='A'),""
    fac=1.0
    if( unit_BorA=='B' ):
        BOHRinANGS=0.5291772105638411
        fac=1.0/BOHRinANGS  ## converts from ANGS to BOHR
    arr=str.split('\n')
    nl=0;Nat=-1
    R=[];Sy=[];
    for line in arr:
        ## print("#parse_xyzstring:"+line)
        line=line.strip()
        Cols=line.split(';')
        for col in Cols:
            col=col.strip();
            if(col==""):
                continue
            arr=col.split()
            if(len(arr)==0):
                continue
            Sy.append(arr[0].strip())
            R.append([ float(arr[1])*fac,float(arr[2])*fac,float(arr[3])*fac])
    ## print("#parse_xyzstring:returns:",R)
    return R,Sy
def write_xyzstring(R,Sy,input_unit,output_unit='A'):
    if( output_unit==input_unit ):
        str="";
        Nat=len(Sy)
        for Iat in range(Nat):
            str+="%s %f %f %f\n"%(Sy[Iat],R[Iat][0],R[Iat][1],R[Iat][2]);
        return str;
            
    assert output_unit=='A',""
    assert input_unit=='B',""
    fac=1.0
    if( input_unit=='B' and output_unit=='A'):
        BOHRinANGS=0.5291772105638411
        fac=BOHRinANGS  ## 0.5291... 
    str="";
    Nat=len(Sy)
    for Iat in range(Nat):
        str+="%s %f %f %f\n"%(Sy[Iat],R[Iat][0]*fac,R[Iat][1]*fac,R[Iat][2]*fac);
    return str;


def sync_Rnuc(cell,who_am_I=""):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    Rnuc_au,Sy=parse_xyzstring(cell.atom,unit_BorA='B')
    Rnuc_refr=pyscf_common.Get_Rnuc();step=pyscf_common.step_;time_AU=pyscf_common.time_AU_
    if( step is None ):step=-1
    if( time_AU is None ):time_AU=0.0
    if( Rnuc_refr is not None ):
        Rdiff = max( np.ravel(Rnuc_au)- np.ravel( Rnuc_refr ) )
        print("#dbgng_20211229:aft.py.625:Rdiff:@%s %e %d,%f:"%(who_am_I,Rdiff,step,time_AU)+str(Rnuc_au)+"/"+str( Rnuc_refr ))
        assert Rdiff<1.0e-6,""
    if(MPIsize > 1 ):
        Rnuc_1D=np.ravel(Rnuc_au);le=len(Rnuc_1D)
        Rnuc_1D_sync=np.zeros([le])
        for j in range(le):
            Rnuc_1D_sync[j]=Rnuc_1D[j] 
        comm.Bcast(Rnuc_1D_sync,root=0)
        dev=max( abs(Rnuc_1D_sync-Rnuc_1D) )
        step=pyscf_common.step_
        step=( -1 if(step is None) else step )
        fd=open("sync_Rnuc_%02d.log"%(MPIrank),"a")
        print("#sync_Rnuc:%06d:%02d: %e"%(step,MPIrank,dev)+str(datetime.datetime.now()),file=fd)
        if( dev >= 1e-6 ):
            print("#sync_Rnuc:",[ "%14.6f"%(Rnuc_1D_sync[k]) for k in range(le) ],
                                [ "%14.6f"%(Rnuc_1D[k]) for k in range(le) ]      ,file=fd)
            Nat=le//3
            Rnuc_sync = np.reshape( Rnuc_1D_sync, [Nat,3] )
            cell.atom=write_xyzstring( Rnuc_sync, Sy,input_unit='B',output_unit='A')
        fd.close()
        ## assert dev<1.0e-6,"sync:%e"%(dev)
