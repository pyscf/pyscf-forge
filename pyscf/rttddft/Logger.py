import numpy as np
import os
import time
import sys
import datetime
from pyscf.pbc.gto import Cell
from .rttddft_common import rttddft_common
from .futils import futils
from .serialize import serialize, read_strtype, diff_objects, load_fromlist
from mpi4py import MPI

def suppress_prtout():
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();
    return ( MPIrank!=0 )

def fill_dict(tgt,value=None):
    for key in tgt:
        tgt[key]=value
def clone_dict(org):
    ret={};
    for key in org:
        ret.update({key:org[key]})
    return ret

class Logger:
    _Logfile=None
    _TimingLogfile=None
    _WarningLogfile=None
    _Append=False
    _Nwarning=0
    _Warnings={}
    _Params={}
    _Counter={}
    _Table={}
    @staticmethod
    def Getlogfpath(extension="_Logger.log"):
        job=rttddft_common.get_job(False)
        if( job is None ):
            fpath="Logger"+extension
        else:
            fpath=job + extension
        return fpath;
    @staticmethod
    def Setv(key,value):
        if( key in Logger._Table ):
            oldv=Logger._Table[key];
            if( oldv != value ):
                Logger.Warning("value of "+str(key)+" changes:"+str(oldv)+" >> "+str(value))
                Logger._Table[key]=value; return 2;
            else:
                return 0;
        else:
            Logger._Table.update({key:value})
            return 1
    @staticmethod
    def Getv(key,default=None):
        if( key in Logger._Table ):
            return Logger._Table[key]
        else:
            return default
    @staticmethod
    def Check_MOs(mf,MO):
        if( Logger.Getv("nAO") is not None ):
            return 0
        ndim_MO=np.shape(MO)
        pbc= isinstance( mf.mol,Cell );Logger.Setv("pbc",pbc)
        # UHF [2,nkpt,nAO,nMO] / [2,nAO,nMO] 
        # RHF   [nkpt,nAO,nMO] / [nAO,nMO]
        nkpt=None;nAO=None;nMO=None;
        if(pbc):
            if(len(ndim_MO)==4):
                if(ndim_MO[0]==2):
                    nkpt=ndim_MO[1];nAO=ndim_MO[2];nMO=ndim_MO[3]
                else:
                    Logger.Warning("ndim_MO:"+str(ndim_MO))
            elif( len(ndim_MO)==3 ):
                nkpt=ndim_MO[0];nAO=ndim_MO[1];nMO=ndim_MO[2]
            if(nAO is not None):
                Logger.Setv("nkpt",nkpt);
                Logger.Setv("nAO",nAO);Logger.Setv("nMO",nMO);
                return 1
            else:
                return -1
        else:
            if(len(ndim_MO)==3):
                if(ndim_MO[0]==2):
                    nAO=ndim_MO[1];nMO=ndim_MO[2]
                else:
                    Logger.Warning("ndim_MO:"+str(ndim_MO))
            elif( len(ndim_MO)==2 ):
                nAO=ndim_MO[0];nMO=ndim_MO[1]
            if(nAO is not None):
                Logger.Setv("nAO",nAO);Logger.Setv("nMO",nMO);
                return 1
            else:
                return -1
            
                
    @staticmethod
    def Countup(key):
        if( key in Logger._Counter ):
            Logger._Counter[key]+=1
        else:
            Logger._Counter.update({key:1})
        return Logger._Counter[key]
    @staticmethod
    def Check_meshsize(mol_or_cell,refresh=False):
        if( "meshsz" in Logger._Params ):
            if( not refresh ):
                return Logger._Params["meshsz"]
        if( mol_or_cell is None ):
            return -1
        mesh=getattr( mol_or_cell,"mesh",None )
        if( mesh is not None):
            meshLd=max(mesh)
            le=len(mesh);
            meshsz=1;
            for j in range(le):
                meshsz=meshsz*mesh[j]
            Logger._Params.update({"meshsz":meshsz})
            fdOut=open("meshsize.log","a");
            for fd1 in [fdOut,sys.stdout]:
                print("#mesh:"+str(mesh)+" %d "%(meshsz)+rttddft_common.get_job(True)+" "+str(datetime.datetime.now()))
            fdOut.close()
            return Logger._Params["meshsz"]
        else:
            return -1
    @staticmethod
    def Warning(title,msg):
        text="#!W:"+msg+" \tjob:"+rttddft_common.get_job(True)+"  \t"+str( datetime.datetime.now() )
        wORa='a'
        if( Logger._WarningLogfile is None ):
            Logger._WarningLogfile=Logger.Getlogfpath(extension="_Warning.log");wORa='w'
        fd=open(Logger._WarningLogfile, wORa);
        print(text,file=fd);fd.close()
        Logger.log(text);

        if( title in Logger._Warnings ):
            if(not (msg in Logger._Warnings[title]) ):
                n=len(Logger._Warnings[title])
                if( n<10 ):
                    Logger._Warnings[title].append(msg);n+=1
                    if(n==10):
                        fd=open("Logger.warning",("w" if(Logger._Nwarning==0) else "a"));Logger._Nwarning+=1
                        for fd1 in [fd,sys.stdout]:
                            print(text+str(Logger._Warnings[title]),file=fd1);
                        fd.close()
            print(text,flush=True);return
        else:
            Logger._Warnings.update({title:[msg]})
            fd=open("Logger.warning",("w" if(Logger._Nwarning==0) else "a"));Logger._Nwarning+=1
            for fd1 in [fd,sys.stdout]:
                print(text+str(Logger._Warnings[title]),file=fd1);
            fd.close()
            
    verbose=False
    @staticmethod
    def serialize_logger(this,delimiter=';'):
        ret="type:"+str(type(this))
        print("serialize_logger:"+str(this))
        sdum=serialize(this,delimiter=delimiter,verbose=0)
        if(sdum is None):
            print("serialize "+str(this)+" returns None");assert False,""
        ret+=delimiter+ serialize(this,delimiter=delimiter,verbose=0)
        dbgng=False
        if(dbgng):
            fpath1="test_serialize_logger01.pscf"
            fpath2="test_serialize_logger02.pscf"
            serialize(this,fpath=fpath1,additional_fields={"type":str(type(this))} )
            fd1=open(fpath1,"r");
            string=None
            for line in fd1:
                line=line.strip();
                if(len(line)==0):
                    continue
                if(line[0]=='#'):
                    continue
                if(string is None):
                    string=line
                else:
                    string+=delimiter+line
            fd1.close()
            logger2=Logger.construct_logger(string,delimiter=delimiter)
            serialize(logger2,fpath=fpath2,additional_fields={"type":str(type(this))})
            os.system("ls -ltrh "+fpath1);
            os.system("ls -ltrh "+fpath2);
            dict_logs={}
            Nd,Ns,Ne=diff_objects(this,logger2,dict_logs=dict_logs)
            print([Nd,Ns,Ne]);
            print(dict_logs)
            assert False,""
        return ret
    @staticmethod
    def construct_logger(string, delimiter=';'):
        print("construct_logger from:"+string)
        print(str(string.split(delimiter)))
        if( string=="None" ):
            return None
        retv=None
        sbuf=string.split(delimiter)
        col0=sbuf.pop(0)
        sarr=col0.split(':');
        key=sarr[0].strip()
        assert (key=="type"),"" 
        print(col0);
        print(sarr);
        strtype=sarr[1].strip();
        ty=read_strtype(strtype)
        if( ty=="Logger.Logger" or ty=="Logger" ):
            retv=Logger(filepath="dummy.log")
        assert (retv is not None),""+ty
        load_fromlist(retv,sbuf)
        retv.time_00=time.time()
        return retv
    _Array=[];
    _Dict={};
###    _timing={"get_hcore":None,"get_veff":None,"get_ovlp":None,"energy_tot":None,"SCF":None,"SCF_iter":None}
    _timing={"get_hcore":None,"get_veff":None,"get_ovlp":None,"energy_tot":None,"SCF_calculation":None,
             "nSCFcycle":None,"avg_SCF_cycle":None,"single_SCF_cycle":None}
    _timing_buffer={}
    @staticmethod
    def Clear_timing(uniquetitle=None,printout=False):
        if(uniquetitle is not None):
            Logger._timing_buffer.update({uniquetitle:clone_dict(Logger._timing)})
        if(printout):
            Logger.log( "#timing_summary:"+("" if(uniquetitle is None) else uniquetitle+":")+str(Logger._timing))
        fill_dict(Logger._timing, None)
    Time_00_=None
    def __init__(self,filepath=None,append=False):
        if( Logger.Time_00_ is None):
            Logger.Time_00_ = time.time()
        self.filepath=filepath;
        self._append=append
        self.time_00=time.time()
        self.time=[0,0,0] # time[0]:timing("start") time[1]=time[2];time[1]=time.time()
        self.function=""
        self._array=[]
        print("logger:initlz"+filepath);
    @staticmethod
    def log(text,timing=True,end='\n',filepath=None,stdout=False,flush=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        if( Logger.Time_00_ is None):
            Logger.Time_00_ = time.time()
        if(timing):
            tcur=time.time();
            text+="  time:%12.3f"%(tcur-Logger.Time_00_)
        if( Logger._Logfile is None ):
            Logger._Logfile = Logger.Getlogfpath();
        fd1=futils.fopen(Logger._Logfile,("a" if Logger._Append else "w"));Logger._Append=True
        ### print("#Logger:printing on "+Logger._Logfile);
        fd2=None
        fdlist=[ fd1 ]
        if( stdout ): fdlist.append( sys.stdout )
        if( filepath is not None ):
            fd2=futils.fopen( filepath,("a" if Append else "w")); fdlist.append(fd2)
        for fd in fdlist:
            print(text,end=end,file=fd,flush=flush);
        futils.fclose(fd1);
        if(fd2 is not None):
            futils.fclose(fd2)
    @staticmethod
    def write(logger,text,level=None,end="\n",timing=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        if( Logger.Time_00_ is None):
            Logger.Time_00_ = time.time()

        if( logger is None ):
            if( timing ):
                tcur=time.time()
                text+="  time:%12.3f"%(tcur-Logger.Time_00_)
            Logger.log(text)
            print(text,end=end);return
        if(level is None):
            logger.info(text,end=end,timing=timing)
        if( level == 'i'):
            logger.info(text,end=end,timing=timing)
        elif( level == 'I'):
            logger.info(text,end=end,timing=timing)
        elif( level == 'w'):
            logger.warning(text,end=end,timing=timing)
    @staticmethod
    def write_once(logger,title,content,fnme=None,append=False):
        if( suppress_prtout() ):
            return
        if( title in Logger._Array ):
            return 0
        if(fnme is not None):
            fd=futils.fopen(fnme,("a" if(append) else "w"))
            print(content,file=fd);futils.fclose(fd);
        fd=open("Logger_write_once.log","a");
        print("title:"+str(title)+" content:"+str(content)+" job:"+str(rttddft_common.get_job(True))+" date:"+str(datetime.datetime.now()))
        if( rttddft_common.get_job(False) is not None ):
            print(title+":"+content+"\t\t"+rttddft_common.get_job(True)+" "+str(datetime.datetime.now()),file=fd);
        else:
            print(title+":"+content+"\t\t"+" "+str(datetime.datetime.now()),file=fd);

        fd.close()
        Logger._Array.append(title)
        Logger.write(logger,content); return 1
    @staticmethod
    def write_maxv(logger,title,value):
        if( suppress_prtout() ):
            return
        if( not (title in Logger._Dict) ):
            Logger._Dict.update({title:value});
            Logger.write(logger,"#Logger:maxv:set:"+title+":%e"%(value));
            return 1;
        old=Logger._Dict[title]
        if( value > old ):
            Logger._Dict[title]=value;
            Logger.write(logger,"#Logger:maxv:upd:"+title+":%e <- %e"%(value,old));
    @staticmethod
    def write_minv(logger,title,value):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        if( not (title in Logger._Dict) ):
            Logger._Dict.update({title:value});
            Logger.write(logger,"#Logger:minv:set:"+title+":%e"%(value));
            return 1;
        old=Logger._Dict[title]
        if( value < old ):
            Logger._Dict[title]=value;
            Logger.write(logger,"#Logger:minv:upd:"+title+":%e <- %e"%(value,old));

    def only_once(self,title,content):
        if( suppress_prtout() ):
            return 0 ## XXX XXX 
        if( title in self._array ):
            return 0
        self._array.append(title)
        self.Info(content)

    #timing(FUNCTIONNAME,start=True);
    #timing("end of XXX")
    #timing(None,end=True)
    def timing(self, title,start=False,end=False, stdout=False, flush=False):
        if(start):
            self.function=title
            self.time[0]=time.time()
            self.time[1]=self.time[0];self.time[2]=self.time[0]; 
            return self.time[0]
        else:
            self.time[2]=self.time[1];self.time[1]=time.time();
            self.print_timing( \
                (("#end of function:"+self.function+("" if title is None else title)) if end else title) \
                +" elapsed: step=%f  function=%f total=%f"%(self.time[1]-self.time[2],self.time[1]-self.time[0],self.time[1]-self.time_00),stdout=stdout,flush=flush )

    def _printout(self,msg,end="\n",timing=False,fpath=None,Append=False,stdout=False,flush=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        if( timing ):
            tcur=time.time()
            msg+=" \t time:%12.3f"%(tcur-self.time_00)
        fdA=[]
        if( fpath is not None ):
            fd=open(fpath,("a" if(Append) else "w"));fdA.append(fd)
        if( self.filepath is not None ):
            fd=open(self.filepath,("a" if(self._append) else "w"));self._append=True;fdA.append(fd)
        for fd in fdA:
            print(msg,file=fd,end=end,flush=flush);fd.close();
        
        Logger.log(msg,stdout=stdout,flush=flush);
        
    def print_timing(self,msg,end="\n",stdout=False,flush=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        fd=None
        if( Logger._TimingLogfile is None ):
            Logger._TimingLogfile = Logger.Getlogfpath(extension='_timing.log')
            fd=open( Logger._TimingLogfile,'w')
        else:
            fd=open( Logger._TimingLogfile,'a')
        print(msg,file=fd,flush=flush);
        fd.close()
        Logger.log(msg,end=end,stdout=stdout,flush=flush)

    def Info(self,msg,end="\n",timing=False,stdout=False,flush=False):       ## ~ only once or twice per function call 
        if( suppress_prtout() ):
            return  ## XXX XXX 
        self._printout(msg,end=end,timing=timing,stdout=stdout,flush=flush)

    def info(self,msg,end="\n",timing=False,stdout=False,flush=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        self._printout(msg,end=end,timing=timing,stdout=stdout,flush=flush)

    def warning(self,msg,end="\n",timing=False,stdout=True,flush=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        self._printout(msg,end=end,timing=timing,stdout=stdout,flush=flush)

class filewriter(Logger):
    def __init__(self,filepath,bfsz=40,append=False):
        self.time_00=time.time()
        self.time=[0,0,0] # time[0]:timing("start") time[1]=time[2];time[1]=time.time()
        self.filepath=filepath;self._append=append;self.block=-1
        self.buffer=[];self.bfsz=bfsz;self.len=0
    def newblock(self,title):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        if( self.len>0 ):
            self.flush()
        self.block+=1;
        self.append("\n\n\n"+"#%d:"%(self.block) + title)

    def flush(self):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        ### print("writing on "+self.filepath);
        ### assert(self._append),"not in append mode"
        fd=futils.fopen(self.filepath,("a" if(self._append) else "w"))
        for item in self.buffer:
            print(item,file=fd)
        futils.fclose(fd);self._append=True;self.len=0;self.buffer=[];
    def append(self,item,Flush=False,end="\n",timing=False,stdout=False,flush=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        if(self.block<0):
            self.block=0
        if( timing ):
            tcur=time.time()
            item+="  time:%12.3f"%(tcur-self.time_00)

        self.buffer.append(item);self.len+=1
        if( self.len>=self.bfsz or Flush or flush):
            self.flush()
        if(stdout):
            print(item,flush=flush)
    def close(self, rename=None):
        if( len(self.buffer)>0 ):
            self.flush()
        if( rename is not None):
            os.system("mv "+self.filepath+" "+rename)
            self.filepath=rename

    def Info(self,msg,end="\n",timing=False,stdout=False,flush=False):       ## ~ only once or twice per function call 
        if( suppress_prtout() ):
            return  ## XXX XXX 
        self.append(msg,end=end,timing=timing,stdout=stdout,flush=flush)

    def info(self,msg,end="\n",timing=False,stdout=False,flush=False):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        self.append(msg,end=end,timing=timing,stdout=stdout,flush=flush)

    def warning(self,msg,end="\n",timing=False,stdout=True,flush=True):
        if( suppress_prtout() ):
            return  ## XXX XXX 
        self.append(msg,end=end,timing=timing,stdout=stdout,flush=flush)
