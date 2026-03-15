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

class rttddft_common:
    # Note we have removed: mf_SCF_|Dict_Clipboard|params

    SCF_buf=[]; SCF_bfsz=5; n_SCF=0; 
    SCF_warnings={}  # key:{'count':0,'i_SCF':[ 5 most recent occurrence ] }

    _Keys_Write_once=[];_Filenames_Write_once=[]

    job=None
    Rnuc_=None
    time_AU_=None
    step_=None
    
    _Counter={}
    _Dict={}

    Dict_Print_warning_={}

    _Time_000=None;
    wt_010={}; Key=None; KeyHistory=None;Timing={}

    _iVerbose=1;Logwriter=None;Logfpath_=None;_Warning_logfile=None

    Dict_Clipboard={} 
    _Params={}
    @staticmethod
    def Params_update(dic):
        if( rttddft_common._Params is None ):
            rttddft_common._Params={}
        rttddft_common._Params.update(dic)

    @staticmethod
    def Params_get(key,default=None):
        if( rttddft_common._Params is None ):
            return default
        if( key not in rttddft_common._Params ):
            return default
        return rttddft_common._Params[key];

    @staticmethod
    def set_params(dict,job=None):
        def get1(dic,key,default=None):
            if(key in dic):
                return dic[key]
            else:
                return default;
        if( job is None):
            job= get1(dict, "name","pySCF_job") + "_" + get1(dict,"xc","") + get1(dict, "basis","") +  get1(dict,"branch","")
        rttddft_common.job=job
        rttddft_common._Params=dict
        rttddft_common.Setup()

    @staticmethod
    def Clipboard(key,value=None,default=None,clear=None,set_value=False):
        if(value is not None or set_value ):
            rttddft_common.Dict_Clipboard.update({key:value})
            ### print("#Clipboard:SET:%s:"%(key),value)
        else:
            if(key in rttddft_common.Dict_Clipboard ):
                if( clear is not None):
                    if(clear):
                        item = rttddft_common.Dict_Clipboard.pop(key)
                        print("#Clipboard:CLEAR:%s:"%(key),item)
                        return item
                return rttddft_common.Dict_Clipboard[key]
            else:
                # keyset=[ ky for ky in rttddft_common.Dict_Clipboard ]
                # print("!W key:%s is not in rttddft_common.Dict_Clipboard"%( str(key) ),keyset)
                return default

    @staticmethod
    def Setup(bfsz=5):
        if( rttddft_common._Time_000 is None ):
            rttddft_common._Time_000=time.time()
        if( rttddft_common.KeyHistory is None ):
            rttddft_common.KeyHistory=Stack(bfsz)
        if( rttddft_common.Logwriter is None ):
            rttddft_common.Logwriter=Filewriter(\
                ("rttddft_common" if (rttddft_common.job is None) else rttddft_common.job)\
                + "_logger.log" )
        
    @staticmethod
    def Get_time():
        if( rttddft_common._Time_000 is None ):
            rttddft_common.Setup()
        
        return time.time()-rttddft_common._Time_000

    @staticmethod
    def Dict_setv(key,val):
        rttddft_common._Dict.update({key:val})

    @staticmethod
    def Dict_getv(key,default=None):
        if( key in rttddft_common._Dict ):
            return rttddft_common._Dict[key]
        else:
            return default

    @staticmethod
    def set_job(job,override=False):
        if( rttddft_common.job is not None ):
            if( not override ):
                return
        rttddft_common.job=job

    @staticmethod
    def get_job(safe=False):
        if( safe ):
            if( rttddft_common.job is None ):
                return ""
        return rttddft_common.job

    @staticmethod
    def DebugWrite(key="",content="",array=None,buf=None,Threads=[0],open=True,dtme=True):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( Threads is not None ):
            if( MPIsize>1 and (MPIrank not in Threads) ):
                return 0
        fd=None;isNew=False
        if( rttddft_common.Logfpath_ is None ):
            today=datetime.date.today()
            fnme="pyscfdbg_%04d%02d%02d.log"%(today.year,today.month,today.day)
            if( os.path.exists(fnme) ):
                fd=open(fnme,"a");print("\n\n\n### %s %s --------"%(str(rttddft_common.job,datetime.datetime.now())),file=fd)
            else:
                fd=open(fnme,"w");print("### %s %s --------"%(str(rttddft_common.job,datetime.datetime.now())),file=fd)
            rttddft_common.Logfpath_=fnme;
            isNew=True
        else:
            fd=open( rttddft_common.Logfpath_,"a");
        if( array is not None ):
            content+="  "+str(array)
        if( dtme ):
            content+="   \t\t"+str(datetime.datetime.now())
        print("#%s:%s"%(key,content),file=fd);
        if( buf is not None ):
            print("#%s:buf:%s"%(key,str(buf)),file=fd)
        fd.close()
        if( isNew ):
            os.system("ls -ltrh "+rttddft_common.Logfpath_)
    @staticmethod
    def Printout_warning(key,content,level=-1, nth_call=[1], dtme=True, Threads=[0], fpath=None, Append=None):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()

        isempty=( len(rttddft_common.Dict_Print_warning_)==0 )
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( key not in rttddft_common.Dict_Print_warning_ ):
            rttddft_common.Dict_Print_warning_.update({key:0});
        rttddft_common.Dict_Print_warning_[key]+=1
        if( dtme ):
            content+=" \t\t "+str(datetime.datetime.now())
        nth=rttddft_common.Dict_Print_warning_[key]

        logfpath0=rttddft_common.get_job(True) + "_Print_warning.log"
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
                    os.system("ls -ltrh "+logfpath0)
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
        if( rttddft_common.Rnuc_ is not None ):
            Nat=len( rttddft_common.Rnuc_ )
            diffs=[ np.sqrt( (Rnuc[I][0]-rttddft_common.Rnuc_[I][0])**2 +
                             (Rnuc[I][1]-rttddft_common.Rnuc_[I][1])**2 + 
                             (Rnuc[I][2]-rttddft_common.Rnuc_[I][2])**2 ) for I in range(Nat) ]
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
            rttddft_common.step_=step
        if( time_AU is not None ):
            rttddft_common.time_AU_=time_AU
        print("#rttddft_common.Rnuc:$%02d:set:"%(MPIrank),Rnuc,step,time_AU)
        rank=len(Ndim);assert rank==1 or rank==2,""
        rttddft_common.Rnuc_=np.zeros(Ndim,dtype=np.float64)
        if(rank==2):
            for I in range(Ndim[0]):
                for J in range(Ndim[1]):
                    rttddft_common.Rnuc_[I][J]=Rnuc[I][J]
        else:
            for I in range(Ndim[0]):
                rttddft_common.Rnuc_[I]=Rnuc[I]

    @staticmethod
    def Get_Rnuc(clone=False):
        if(not clone):
            return rttddft_common.Rnuc_

        Ndim=np.shape(rttddft_common.Rnuc_)
        rank=len(Ndim);assert rank==1 or rank==2,""
        ret=np.zeros(Ndim,dtype=np.float64)
        if(rank==2):
            for I in range(Ndim[0]):
                for J in range(Ndim[1]):
                    ret[I][J]=rttddft_common.Rnuc_[I][J]
        else:
            for I in range(Ndim[0]):
                ret[I]=rttddft_common.Rnuc_[I]
        return ret

    @staticmethod
    def Assert(bool,text,severity):
        if(bool):
            return 0
        rttddft_common.Print_warning(text,severity=severity)
        if( severity<0 ):
            assert False,""+text
    @staticmethod
    def Print_warning(text,severity=1):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( severity>=0 and MPIrank>0 ):
            return
        Append=True
        if( rttddft_common._Warning_logfile is None ):
            rttddft_common._Warning_logfile=("pyscf_" if(rttddft_common.job is None) else rttddft_common.job)\
                                          +"_warning.log"
            Append=False
        fd=open( rttddft_common._Warning_logfile,('a' if(Append) else 'w'))
        print(text+" \t\t"+str(datetime.datetime.now()),file=fd)
        fd.close()

    @staticmethod
    def Write_once(key,text,fpath=None,filename=None,Append=False,stdout=True):
        if( key in rttddft_common._Keys_Write_once ):
            return
        else:
            rttddft_common._Keys_Write_once.append(key)
        fdA=[];fdOU=None
        if(stdout or (fpath is None)):
            fdA.append(sys.stdout)

        if(fpath is not None):
            fdOU=open(fpath,('a' if(Append) else 'w'))
            fdA.append(fdOU)
        else:
            if(filename is not None):
                append_file=True
                if( filename not in rttddft_common._Filenames_Write_once ):
                    rttddft_common._Filenames_Write_once.append( filename );append_file=False
                fpath=rttddft_common.get_job(True)+"_"+filename+".log"
                ## print("Write_once:%s %r"%(fpath,append_file))
                fdOU=open(fpath,('a' if(append_file) else 'w'))
                fdA.append(fdOU)
        for fd in fdA:
            print(key+":"+text,file=fd)
        if(fdOU is not None):
            fdOU.close()

    @staticmethod
    def Append_SCFwarning(item,i_SCF=None):
        if( i_SCF is None ):
            i_SCF = rttddft_common.n_SCF
        if( item not in rttddft_common.SCF_warnings ):
            rttddft_common.SCF_warnings.update({item:{'count':0,'i_SCF':Stack(5)}})
        rttddft_common.SCF_warnings[item]['count']+=1
        if( not rttddft_common.SCF_warnings[item]['i_SCF'].contains(i_SCF) ):
            rttddft_common.SCF_warnings[item]['i_SCF'].append(i_SCF)

    @staticmethod
    def Setup_SCFbuf():
        if( len(rttddft_common.SCF_buf)==0 ):
            for I in range(rttddft_common.SCF_bfsz):
                rttddft_common.SCF_buf.append({'iter':0, 'time':0, 'cvgd':None})
            rttddft_common.SCF_buf.append({'iter':{'sum':0.0,'sqrsum':0.0}, 'time':{'sum':0.0,'sqrsum':0.0} })
    @staticmethod
    def Get_SCFresult(nth=0):
        if( nth >= rttddft_common.SCF_bfsz):
            return None
        i_SCF= rttddft_common.n_SCF-1 - nth
        if( i_SCF < 0 ):
            return None
        iMOD=i_SCF%rttddft_common.SCF_bfsz
        return rttddft_common.SCF_buf[iMOD]

    @staticmethod
    def Set_SCFresult(cvgd, iter, Egnd, time, warnings=None,dbuf=None):
        import sys
        import datetime
        if( len(rttddft_common.SCF_buf)==0 ):
            rttddft_common.Setup_SCFbuf()
        rttddft_common.n_SCF+=1
        i_SCF= rttddft_common.n_SCF-1
        iMOD=i_SCF%rttddft_common.SCF_bfsz
        rttddft_common.SCF_buf[iMOD]['cvgd']=cvgd
        rttddft_common.SCF_buf[iMOD]['iter']=iter
        rttddft_common.SCF_buf[iMOD]['time']=time
        ### rttddft_common.SCF_buf[iMOD]['warnings'].clear()
        rttddft_common.SCF_buf[rttddft_common.SCF_bfsz]['iter']['sum']+=iter
        rttddft_common.SCF_buf[rttddft_common.SCF_bfsz]['iter']['sqrsum']+=iter**2

        rttddft_common.SCF_buf[rttddft_common.SCF_bfsz]['time']['sum']+=time
        rttddft_common.SCF_buf[rttddft_common.SCF_bfsz]['time']['sqrsum']+=time**2
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
                rttddft_common.Append_SCFwarning(item,i_SCF=i_SCF)
        if( rttddft_common.n_SCF == 1 or rttddft_common.n_SCF==5 or rttddft_common.n_SCF==10 or rttddft_common.n_SCF == 20 or 
            rttddft_common.n_SCF %100==0 ):
            fpath=("" if(rttddft_common.job is None) else rttddft_common.job)+"_SCF.log"
            rttddft_common.Print_SCFresults( fpath,Append=( rttddft_common.n_SCF>1 ))

    @staticmethod
    def Print_SCFresults(fpath=None,Append=False):
        fdA=[ sys.stdout ]
        fdout=(None if(fpath is None) else open(fpath,('a' if(Append) else 'w')))
        if(fdout is not None):
            fdA.append(fdout)
        if( len(rttddft_common.SCF_buf) <= rttddft_common.SCF_bfsz ):
            print("#Print_SCFresults:len:%d / bfsz:%d"%( len(rttddft_common.SCF_buf), rttddft_common.SCF_bfsz ))
            return

        Dic=rttddft_common.SCF_buf[rttddft_common.SCF_bfsz]
        strWarning=""
        for key in rttddft_common.SCF_warnings:
            wdict= rttddft_common.SCF_warnings[key]
            strWarning +=key+":%3d / %4d"%( rttddft_common.SCF_warnings[key]['count'],rttddft_common.n_SCF ) \
                            +str( rttddft_common.SCF_warnings[key]['i_SCF'].toarray() )
        if( len(strWarning.strip())>0 ):
            strWarning="  Warning:"+strWarning
        for fd in fdA:
            is_stdout=(fd==sys.stdout)
            av_iter=Dic['iter']['sum']/float(rttddft_common.n_SCF)
            stdv_iter=np.sqrt( abs( Dic['iter']['sqrsum']/float(rttddft_common.n_SCF) -av_iter**2 ) )

            av_time=Dic['time']['sum']/float(rttddft_common.n_SCF)
            stdv_time=np.sqrt( abs( Dic['time']['sqrsum']/float(rttddft_common.n_SCF) -av_iter**2 ) )
            if( (not Append) and (fdout is not None) and fd==fdout ):
                print("#%6s    %6s  %14s  %14s    %14s %14s %14s %14s "%("n_SCF",
                    "sum_iter","avg_iter","stdv_iter",
                    "sum_time","avg_time","stdv_time","per-iter-time"),file=fd)
            
            #                        ITER           TIME
            #                  n_SCF  sum  av  stdv          sum   av stdv  PER-ITER-time
            print("%s %6d    %6d  %14.6f  %14.4e    %14.6f %14.6f %14.4e %14.6f "%(
                    ("#SCFresults:" if(is_stdout) else ""), rttddft_common.n_SCF,
                    Dic['iter']['sum'], av_iter, stdv_iter,
                    Dic['time']['sum'], av_time, stdv_time, Dic['time']['sum']/float(Dic['iter']['sum']))
                  +strWarning,file=fd)
        if(fdout is not None):
            fdout.close()

    @staticmethod
    def Countup(key,inc=True):   ## by default 1,2,3,... 
        if( key not in rttddft_common._Counter ):
            if( not inc ):
                return 0
            else:
                rttddft_common._Counter.update({key:0})
        if(inc):
            rttddft_common._Counter[key]+=1
        return rttddft_common._Counter[key]
    @staticmethod
    def count(key,inc=False):
        return rttddft_common.Countup(key,inc=inc)

    @staticmethod
    def Update_timing(key,walltime,flush=False,doNotPrtOut=False):
        if( key not in rttddft_common.Timing ):
            rttddft_common.Timing.update({key:{'count':0,'sum':0.0,'sqrsum':0.0}})
        rttddft_common.Timing[key]['count']+=1
        rttddft_common.Timing[key]['sum']+=walltime
        rttddft_common.Timing[key]['sqrsum']+=walltime**2
        ncount=rttddft_common.Timing[key]['count']
        ### print("#rttddft_common.Update_timing:%s:%d"%(key,ncount))
        if( not doNotPrtOut ):
            if( ncount == 1 or ncount==10 or  ncount==20 or ncount == 100 or  ncount== 200 or ncount==1000 ):
                avg= rttddft_common.Timing[key]['sum'] / rttddft_common.Timing[key]['count']
                stdv=np.sqrt( abs( rttddft_common.Timing[key]['sqrsum'] / rttddft_common.Timing[key]['count'] - avg**2 ) )
                print("#AvgTime:%24s  %14.4f  %14.4e  %d   %14.4f  avg/stdv/count/sum"%(key,avg,stdv,
                             rttddft_common.Timing[key]['count'], rttddft_common.Timing[key]['sum']))
        return rttddft_common.Timing[key]['count']
    @staticmethod
    def Print_timing(name,keys,walltime=None,flush=False,depth=1,N_iter=None,subkey=""):
        ### print("#rttddft_common.print_timing..");
        if( N_iter is not None ):
            ky1=name+"_N_iter"
            rttddft_common.Update_timing(ky1,N_iter,flush=False,doNotPrtOut=False)
            if(ky1 not in keys):
                keys.append(ky1)
        if( name in [ 'get_veff', 'get_j', 'Get_Veff'] ):
            name1=name+"_1st"
            if( name1 not in rttddft_common.Timing):
                rttddft_common._Print_timing( name1, keys, walltime=walltime, flush=flush, depth=depth)
            else:
                name2=name+"_2nd"
                name_upd=name2
                if( name2 in rttddft_common.Timing ):
                    av   = rttddft_common.Timing[name2]['sum']/rttddft_common.Timing[name2]['count']
                    stdv = np.sqrt( abs( rttddft_common.Timing[name2]['sqrsum']/rttddft_common.Timing[name2]['count'] - av**2 ) )
                    av_1   = rttddft_common.Timing[name1]['sum']/rttddft_common.Timing[name1]['count']
                    stdv_1 = np.sqrt( abs( rttddft_common.Timing[name1]['sqrsum']/rttddft_common.Timing[name1]['count'] - av_1**2 ) )
                    if( abs(av - walltime)> max(1.0, stdv)*3 ):
                        print("#get_Veff:%s:time %f / get_Veff_2nd av=%f(%f)  av1=%f(%f)  ... %s"%(subkey,walltime,av,stdv,av_1,stdv_1,name_upd))
                rttddft_common._Print_timing( name_upd, keys, walltime=walltime, flush=flush, depth=depth)
        else:
            rttddft_common._Print_timing( name, keys, walltime=walltime, flush=flush, depth=depth)
    @staticmethod
    def _Print_timing(name, keys, walltime=None, flush=False, depth=1):
        fpath=("rttddft_common" if(rttddft_common.job is None) else rttddft_common.job)+"_"+name+"_walltime.log"
        wctNow=time.time() - rttddft_common._Time_000
        ### print("#rttddft_common._Print_timing..%f"%(wctNow) +fpath);
        
        ncount=None;keyAL=None
        if(walltime is not None):
            ncount=rttddft_common.Update_timing( name,walltime,doNotPrtOut=True)
            keyAL=[name]
            for ky in keys:
                keyAL.append(ky)
            ### print("#rttddft_common._Print_timing.1: %s %d"%(name,ncount))
        else:
            ncount=99999
            for ky in keys:
                if( ky in rttddft_common.Timing):
                    ncount=min(ncount,rttddft_common.Timing[key]['count'])
            keyAL=[]
            for ky in keys:
                keyAL.append(ky)
            ### print("#rttddft_common._Print_timing.2: %s %d"%(name,ncount))

        if( flush ):
            header='#'*(2*depth)
            strtm=("--" if(walltime is None) else "%14.4f"%(walltime))
            print( header + "END:%24s  %14s  %14.4f"%(name, strtm, wctNow)
                                          +" \t\t\t "+("" if(rttddft_common.job is None) else rttddft_common.job) 
                                          + " "+str(datetime.datetime.now())) 

        if( ncount == 1 or ncount==10 or  ncount==20 or ncount == 100 or  ncount== 200 or ncount==1000 ):
            string="";legend="";ncol=0
            for key in keyAL:
                if( key not in rttddft_common.Timing ):
                    keyset=[ ky for ky in rttddft_common.Timing ]
                    print("#!W %s not in Timing:"%(key)+str(keyset));continue
                avg= rttddft_common.Timing[key]['sum'] / rttddft_common.Timing[key]['count'] 
                stdv=np.sqrt( abs( rttddft_common.Timing[key]['sqrsum'] / rttddft_common.Timing[key]['count'] - avg**2 ) )
                string+="   %18.4f %18.4e %8d %18.4f     "%(avg,stdv,
                         rttddft_common.Timing[key]['count'], rttddft_common.Timing[key]['sum'])
                legend+="   %4d:%24s_%24s_av,stdv,cnt,sum   "%(ncol+1,key,name)
                ncol+=4
            fdOu=open( fpath,('w' if(ncount==1) else 'a'))
            if( ncount==1 ):
                print(legend,file=fdOu);
            print(string,file=fdOu);
            fdOu.close()
        return ncount

    @staticmethod
    def Assert(bool,text,severity):
        if(bool):
            return 0
        rttddft_common.Print_warning(text,severity=severity)
        if( severity<0 ):
            assert False,""+text
    @staticmethod
    def Print_warning(text,severity=1):
        comm=MPI.COMM_WORLD
        MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
        if( severity>=0 and MPIrank>0 ):
            return
        Append=True
        if( rttddft_common._Warning_logfile is None ):
            rttddft_common._Warning_logfile=("pyscf_" if( rttddft_common.job is None) else rttddft_common.job)\
                                          +"_warning.log"
            Append=False
        fd=open( rttddft_common._Warning_logfile,('a' if(Append) else 'w'))
        print(text+" \t\t"+str(datetime.datetime.now()),file=fd)
        fd.close()




    @staticmethod
    def Start_timer(key,flush=False,depth=1):
        if( rttddft_common._Time_000 is None ):
            rttddft_common.Setup()
        rttddft_common.wt_010.update({key: time.time() - rttddft_common._Time_000})
        oldky=rttddft_common.Key
        if( oldky is not None ):
            rttddft_common.KeyHistory.append(oldky);
        rttddft_common.Key=key
        
        ### print("#Start_timer:"+str(oldky)+" >> current:"+str(rttddft_common.Key))
        if( flush or rttddft_common._iVerbose>1):
            header='#'*(2*depth)
            rttddft_common.Logwriter.append( header + "start:%24s  %14.4f "%(key, rttddft_common.wt_010[key])
                                          +" \t\t\t "+("" if(rttddft_common.job is None) else rttddft_common.job) 
                                          + " "+str(datetime.datetime.now())) 
        return rttddft_common.wt_010[key]
    @staticmethod
    def Switch_timer(key1,key2,flush=False,depth=1):
        ret= rttddft_common.Stop_timer(key1,key2=key2,flush=flush,depth=depth)
        return ret
    @staticmethod
    def Stop_timer(key,key2=None,flush=False,depth=1,subkey=""):
        if( rttddft_common._Time_000 is None):
            rttddft_common.Setup();
        wctNow=time.time() - rttddft_common._Time_000
#        if( key == rttddft_common.Key ):
        if( key in rttddft_common.wt_010 ):
            if( key in [ 'get_veff', 'get_j'] ):
                key1=key+"_1st"
                if( key1 not in rttddft_common.Timing):
                    rttddft_common.Update_timing( key1, wctNow-rttddft_common.wt_010[key], flush=flush)
                else:
                    key2=key+"_2nd";tCUR=wctNow-rttddft_common.wt_010[key]
                    key_to_upd=key2
                    if( key2 in rttddft_common.Timing):
                        av   = rttddft_common.Timing[key2]['sum']/rttddft_common.Timing[key2]['count']
                        stdv = np.sqrt( abs( rttddft_common.Timing[key2]['sqrsum']/rttddft_common.Timing[key2]['count'] - av**2 ) )
                        av_1   = rttddft_common.Timing[key1]['sum']/rttddft_common.Timing[key1]['count']
                        stdv_1 = np.sqrt( abs( rttddft_common.Timing[key1]['sqrsum']/rttddft_common.Timing[key1]['count'] - av_1**2 ) )
                        if( abs(av - tCUR)> max(1.0, stdv)*3 ):
                            if( abs(av_1-tCUR)<abs(av-tCUR) ):
                                key_to_upd=key1
                            print("#get_Veff:%s:time %f / get_Veff_2nd av=%f(%f)  av1=%f(%f)  ... %s"%(subkey,tCUR,av,stdv,av_1,stdv_1,key_to_upd))
                    rttddft_common.Update_timing( key_to_upd, wctNow-rttddft_common.wt_010[key], flush=flush)

            else:
                rttddft_common.Update_timing( key, wctNow-rttddft_common.wt_010[key], flush=flush)

            if( flush or rttddft_common._iVerbose>1):
                header='#'*(2*depth)
                rttddft_common.Logwriter.append( header + "END:%24s  %14.4f  %14.4f"%(key, wctNow-rttddft_common.wt_010[key], wctNow)
                                              +" \t\t\t "+("" if(rttddft_common.job is None) else rttddft_common.job) 
                                              + " "+str(datetime.datetime.now())) 

            rttddft_common.Key=None; rttddft_common.wt_010.pop(key,None)
        else:
            rttddft_common.Logwriter.append("## key:%s did not end  tmNow:%f"%(key,wctNow))
            ### rttddft_common.Key=None; rttddft_common.wt_010=None
            if( flush or rttddft_common._iVerbose>1):
                header='#'*(2*depth)
                rttddft_common.Logwriter.append( header + "END:%24s  %14s  %14.4f"%(key, "--", wctNow)
                                              +" \t\t\t "+("" if(rttddft_common.job is None) else rttddft_common.job) 
                                              + " "+str(datetime.datetime.now())) 
        if( key2 is None ):
            return wctNow
        else:
            ### print("#Switching to:"+str(key2))
            return rttddft_common.Start_timer(key2)
