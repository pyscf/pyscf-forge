import datetime
import numpy as np
import sys
import os
import time
from .utils import i1toa,d1toa,z1toa,iNtoa,dNtoa,zNtoa,sNtoa,aNmaxdiff,i1eqb
from .Stdlogger import Stdlogger
from .laserfield import kickfield, CWField01, gaussianpulse01, Trapezoidalpulse
from mpi4py import MPI
from .Loglv import printout
from .rttddft_common import rttddft_common

def serialize_CDIIS(this,delimiter=';'):
    return serialize(this,delimiter=delimiter)
def construct_CDIIS(string,delimiter=';'):
    if( string=="None" ):
        return None
    retv=None
    sbuf=string.split(delimiter)
    col0=sbuf.pop(0); 
    sA=col0.split(':'); key=sA[0].strip();strtype=read_strtype( sA[1].strip() );
    assert (key=="type"),""
    assert (strtype=="CDIIS"),"strtype:"+strtype
    types,values=parse_slist(sbuf)
    retv=CDIIS()
    return load_fromlist(retv,sbuf)
    
def diff_dict(lh,rh,verbose=0):
    Ndiff=0; Nsame=0; Nerr=0; strErr=""; strDiff=""
    for ky in lh:
        if(not(ky in rh)):
            Nerr+=1; strErr+="#%d:%s lh:%s rh:missing"%(Nerr,str(ky),str(lh))+";"
        else:
            nd,ns,ne=diff_objects( lh[ky], rh[ky] )
            if(ne>0):
                Nerr+=1; strErr+="#%d:%s lh:%s rh:%s  diff:%d,%d,%d"%(Nerr,str(ky),str(lh),str(rh),nd,ns,ne)+";"
            elif(nd>0):
                Ndiff+=1;strDiff+="#%d:%s lh:%s rh:%s  diff:%d,%d,%d"%(Ndiff,str(ky),str(lh),str(rh),nd,ns,ne)+";"
            elif(ns>0):
                Nsame+=1;
            else:
                strErr+="###:%s lh:%s rh:%s  diff:%d,%d,%d"%(str(ky),str(lh),str(rh),nd,ns,ne)+";"
    strLog="";
    if(len(strDiff)>0):
        if(len(strLog)>0):
            strLog="ERR:%d:"%(Nerr)+strErr+";\t DIFF:%d:"%(Ndiff)+strDiff
        else:
            strLog="DIFF:%d:"%(Ndiff)+strDiff
    else:
        if(len(strErr)>0):
            strLog="ERR:%d:"%(Nerr)+strErr
    return Ndiff,Nsame,Nerr,strLog
def load_fromfile( proto, fpath, constructors=None, excluding=None, logfile=None):
    sbuf=[];n=0
    fdIN=open(fpath,"r")
    for line in fdIN:
        line=line.strip();le=len(line)
        if( le==0 ):
            continue
        sbuf.append(line);
        ## print("#load_fromfile:%03d:"%(n)+line);n+=1
    fdIN.close()
    return load_fromlist( proto, sbuf, constructors=constructors, excluding=excluding,logfile=logfile )

def load_fromstring( proto, string, delimiter=';', constructors=None):
    if( string=="None" ):
        return None
    
    sbuf=( string.split() if( delimiter is None ) else string.split(delimiter) );
    return load_fromlist( proto, sbuf,constructors=constructors)

def load_fromlist( proto, sBuf, constructors=None, excluding=None, logfile=None, append=False ):
    ### print("#load_fromlist:%s:"%(str(type(proto)))+str(sBuf))
    loglevel=1; logfd=sys.stdout
    if(logfile is not None):
        logfd=open(logfile,('a' if(append) else 'w') );loglevel=2
    Nbuf=len(sBuf)
    seqno=1
    for ibuf in range(Nbuf):
        line=sBuf[ibuf].strip();le=len(line)
        if(le<1):
            continue
        if( line.startswith('#') ):
            Stdlogger.printout(1,"#skipping:"+line);continue

        if(2<loglevel):
            print("#load_fromlist:### %d:"%(ibuf)+str(line),file=logfd)

        Ndim=None;val=None

        sarr=line.split(':')
        key=sarr[0].strip();

        if( excluding is not None ):
            if(key in excluding):
                if(1<loglevel):
                    print(" ... SKIP",file=logfd)
                Stdlogger.printout(1,"#skipping:excluding:"+line); continue

        strval=None;
        if(len(sarr)==2):
            assert False,"20220218:never come here since you should never refer to sarr[2]..."
            strval=sarr[2].strip() ## TODO FIXME... 
        elif(len(sarr)>2):
            colonAT=[];nC=0;le=len(line)
            for j in range(le):
                if(line[j]==':'):
                    colonAT.append(j);nC+=1;
                    if(nC==2):
                        break
            assert nC==2,""
            strval=line[colonAT[1]+1:]
            assert strval.startswith(sarr[2]),""

        Ty=sarr[1];val=None
        if( strval.startswith("!") ):
            typ=Ty[0];
            fpath1=strval[1:];get_comments=[]
            if(typ=='I' or typ=='D' or typ=='Z'):
                val=svld_aNbuf('L',fpath1,get_comments=get_comments);
                nd=np.shape(val);
                if( len(Ty)>1 ):
                    sdum=Ty[1:].split(',');n=len(sdum)
                    Ndim=[ int(sdum[kk]) for kk in range(n) ]
                    if(not i1eqb(nd,Ndim) ):
                        val=np.reshape(val,Ndim)
                setattr(proto,key,val)
                if(1<loglevel):
                    print("#load_fromlist:%d:"%(seqno) +str(key)+":"+str(typ)+":"+str(np.shape(val)),file=logfd);seqno+=1
                continue
            else:
                assert False,"wrong type:"+typ


        #print("#load_fromlist:val:"+strval);
        #print("#load_fromlist:Ty:"+sarr[1]);

        
        ### print("#load_fromlist:key:"+key+":"+Ty+":"+strval,flush=True);
        if( constructors is not None):
            if( key in constructors ):
                fnc=constructors[key]
                print("#load_fromlist:key:"+key+":"+Ty+":apply constructor:"+str(fnc))
                val=fnc(strval)
                setattr(proto,key,val);continue
        if( val is None ):
            if( Ty=='dict' ):
                if( (strval is None) or (len(strval)==0) or (strval == '{}') ):
                    val={}
                else:
                    strle=len( strval.strip() )
                    assert strval[0]=='{' and strval[strle-1]=='}',""
                    contents=strval[1:strle-1]
                    cols=contents.split(',');n1=len(cols);ncols=n1//2;assert n1==2*ncols,""
                    val={};
                    for jc in range(ncols):
                        ky1=cols[2*jc];v1=cols[2*jc+1]
                        if( v1.isnumeric() ):
                            ndot=0;nExpo=0;lv=len(v1);
                            for kk in range(lv):
                                if(v1[kk]=='.'):
                                    ndot+=1
                                elif( v1[kk]=='E' or v1[kk]=='e'):
                                    nExpo+=1
                            if(ndot>0 or nExpo>0):
                                v1=float(v1)
                            else:
                                v1=int(v1)
                        val.update({ky1:v1})
                setattr(proto,key,val)
                if(1<loglevel):
                    print("#load_fromlist:%d:"%(seqno) +str(key)+":"+str(typ)+":"+str(val),file=logfd);seqno+=1
            else:
                typ=Ty[0];
                
                if( len(Ty)>1 ):
                    sdum=Ty[1:].split(',');n=len(sdum)
                    Ndim=[ int(sdum[kk]) for kk in range(n) ]

                if(typ=='i'):
                    val=int( strval )
                elif(typ=='d'):
                    val=float( strval )
                elif(typ=='z'):
                    sdum=strval.split();
                    val=float(sdum[0]) + 1j*float(sdum[1])
                elif(typ=='s'):
                    val=strval;
                elif(typ=='b'):
                    val=eval(strval);
                elif(typ=='I'):
                    sA=strval.split();nA=len(sA)
                    val=np.reshape( [ int(sA[k]) for k in range(nA) ], Ndim)
                elif(typ=='D'):
                    sA=strval.split();nA=len(sA)
                    val=np.reshape( [ float(sA[k]) for k in range(nA) ], Ndim)
                elif(typ=='Z'):
                    sA=strval.split();nA1=len(sA);nA=nA1//2; assert (nA1%2==0),""
                    val=np.reshape( [ float(sA[2*k]) + 1j*float(sA[2*k+1]) for k in range(nA) ], Ndim)
                elif(typ=='S'):
                    sA=strval.split();
                    val=np.reshape(sA, Ndim)
                elif(typ=='o'):
                    if( strval == "None" ):
                        val=None
                    else:
                        assert constructors is not None,"don't know how to create:"+key+":"+typ
                        assert key in constructors,"no valid constructor for "+key
                        Stdlogger.printout(1,"#creating:"+key+" from:"+strval)
                        val=constructors[key](strval)
                else:
                    assert False,"wrong typ:"+typ

                setattr(proto,key,val)
                if(1<loglevel):
                    print("#load_fromlist:%d:"%(seqno) +str(key)+":"+str(typ)+":"+str(val),file=logfd);seqno+=1
    if(logfile is not None):
        logfd.close();os.system("ls -ltrh "+logfile)
    return proto

def parse_file(fpath):
    types={};values={}
    fd=open(fpath,"r")
    slist=[]
    for line in fd:
        line=line.strip();le=len(line);
        if(le<1):
            continue
        if(line.startswith('#')):
            Stdlogger.printout(1,"#check_serialization:reading off:"+line);continue
        slist.append(line)
    return parse_slist(slist)
def parse_slist(slist):
    types={};values={}
    ith=-1
    for item in slist:
        ith+=1
        arr=item.strip().split(':');assert (len(arr)>1),"slist#%d:%s"%(ith,item)
        key=arr[0].strip()
        typ=arr[1].strip(); types.update({key:typ})
        if( len(arr)>2 ):
            val=arr[2].strip(); values.update({key:val});
        else:
            values.update({key:None});
    return types,values
def is_method_or_function(tgt,verbose=False):
    if(tgt is None):
        if( verbose ):
            print("!W is_method:None input..");
        return -1
    strtype=str( type(tgt) )
    
    if( "method" in strtype ):
        if( strtype.endswith("\'method\'>") ):
            return 2
        if( strtype.endswith("\'builtin_function_or_method\'>")):
            return 2
        print("check:"+strtype);
        print("tgt:"+str(tgt));
        if( strtype.endswith("\'method_descriptor\'>") ):
            return 2
        
        assert False,""
        return 1

    if( "function" in strtype ):
        if( strtype.endswith("\'function\'>") ):
            return 2
        print("Check:"+strtype);assert False,""
        return 1
    return 0

def check_serialization(fpath=None,string=None,delimiter=';',Dict=None,check_type=True):
    assert (fpath is not None) or (strins is not None),""
    assert (fpath is None) or (strins is None),""
    assert (Dict is not None),""
    types,values=( parse_file(fpath) if (fpath is not None) else 
                                           parse_slist(string.split(delimiter)))
    
    if( check_type ):
        assert "type" in types,""
        strtype=types.pop("type") ## str(type)
        dum=values.pop("type") ## None

    Ndiff=0;Nsame=0;Nerr=0;strLOG=""
    for key in Dict:
        if( key in types ):
            lhs=Dict[key].split('|')
            rhs=types[key];
            if(rhs in lhs):
                Nsame+=1
            else:
                Ndiff+=1; strLOG+=key+":"+ str(rhs)+ "/"+str(lhs)+";"
        else:
            Nerr+=1; strLOG+=key+":not in string;"
    for key in types:
        if( key in Dict ):
            continue
        else:
            Nerr+=1; strLOG+=key+":not in given types;"
    return Ndiff,Nsame,Nerr,strLOG            

def read_strtype(string):
    string=string.strip();
    if(string.startswith("<class \'") and string.endswith("\'>") ):
        string=string.replace("<class \'","").replace("\'>","");
        return string
    else:
        return None

def serialize_listedfields(obj,keylist,delimiter="\n"):
    ret="";n=0
#    primitive_types=[str,int,float,complex,np.complex128
    str_or_int=[str,int]
    floats=[float,np.float64]
    complexes=[complex,np.complex128]
    for key in keylist:
        val=getattr(obj,key,None)
        if( val is None ):
            ret+=("" if(n==0) else delimiter) +key+":o:None";n+=1
        elif( isinstance(val,bool) ):
            ret+=("" if(n==0) else delimiter) +key+":b:%r"%(val);n+=1
        elif( isinstance(val,str) ):
            ret+=("" if(n==0) else delimiter) +key+":s:%s"%(val);n+=1
        elif( isinstance(val,int) ):
            ret+=("" if(n==0) else delimiter) +key+":i:%d"%(val);n+=1
        elif( isinstance(val,float) or isinstance(val,np.float64) ):
            ret+=("" if(n==0) else delimiter) +key+":d:%24.12e"%(val);n+=1
        elif( isinstance(val,float) or isinstance(val,np.complex128) ):
            ret+=("" if(n==0) else delimiter) +key+":z:%24.12e %24.12e"%(val.real,val.imag);n+=1
        elif( isinstance(val,dict) ):
            if( len(val) == 0 ):
                typ='dict';strval="{}"
            else:
                typ='dict';strval=None
                for ky1 in val:
                    if(strval is None):
                        strval="{"+ str(ky1).strip() + ","+ str( val[ky1] ).strip()
                    else:
                        strval+=","+ str(ky1).strip() + ","+ str( val[ky1] ).strip()
                strval+="}"
            ret+=("" if(n==0) else delimiter) +key+":dict:%s"%(strval);n+=1
        elif( isinstance(val,list) or isinstance(val,np.ndarray) ):
            Ndim=np.shape(val);v1D=np.ravel(val)
            le=len(v1D)
            if(le==0):
                typ="I"+i1toa(Ndim,delimiter=',')
            if(isinstance(v1D[0],int)):
                typ="I"+i1toa(Ndim,delimiter=',')
                strval=iNtoa(v1D,delimiter=" ")
            elif(isinstance(v1D[0],str)):
                typ="S"+i1toa(Ndim,delimiter=',') 
                strval=sNtoa(v1D,delimiter=" ")
            elif( isinstance(v1D[0],float) or isinstance(v1D[0],np.float64) ):
                typ="D"+i1toa(Ndim,delimiter=',') 
                strval=d1toa(v1D,format="%24.12e",delimiter=" ")
            elif( isinstance(v1D[0],complex) or isinstance(v1D[0],np.complex128)):
                typ="Z"+i1toa(Ndim,delimiter=',') 
                strval=z1toa(v1D,format="%24.12e %24.12e",delimiter="   ")
            else:
                print("OBJ:"+str(type(obj))+" "+str(obj))
                print("KEY:"+str(key)+" "+str(type(val))+" "+str(val))
                print("TYP:"+str(type(v1D[0]))+" "+str(v1D[0]))
                typ="o";strval=""
                for j in range(le):
                    strval+=("" if(j==0) else " ") +str(v1D[j])
            ret+=("" if(n==0) else delimiter) +key+":%s:%s"%(typ,strval);n+=1
    return ret
# usually additional_fields[key] looks like  i:3776    D3,3:1.192 1.333 1.467 ...
def serialize(obj,fpath=None,delimiter='\n',comment=None,serializers=None,depth=1,
              additional_fields=None, fields_excluded=None,verbose=0, strict=True, Threads=[0], return_None=False, Dict=None ):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    Ld_uplm=100 ## TODO 
    keys_written_on_file=[]
    wt0=time.time()
    if( return_None ):
        if(fpath is not None):
            printout("#serialize:return_None is True but no fpath input...",warning=1)
            return_None=False

    if( (MPIsize > 1) and (MPIrank not in Threads) ):
        return None
    assert obj is not None,""
    header="#%d"%(depth);
    for J in range(1,depth):
        header+="  "
    print("##serialize:main:depth=%d:"%(depth) +str(obj)+":"+str(type(obj)),flush=True)
    primitive_types=[ str,int,float,np.float64,complex,np.complex128]
    collection_types=[ list,dict,np.ndarray ]
    for T in primitive_types:
        if( isinstance(obj,T) ):
            print("#!W primitive type in serialize method:"+str(obj));return str(obj)
    for T in collection_types:
        if( isinstance(obj,T) ):
            print("#!W collection type in serialize method:"+str(obj));return str(obj)
   

    Stdlogger.printout(1,"#Serialize:"+header+str(type(obj))+":"+str(obj) ,verbose=verbose, wt0=wt0,dtme=True)
    keylist=dir(obj)
    nkey=len(keylist)
    Stdlogger.printout(1,"keylist:%d:"%(nkey) +str(keylist), wt0=wt0,dtme=True)
    fd=( None if(fpath is None) else open(fpath,"w"))
    if(fd is not None):
        print("#"+str(datetime.datetime.now()),file=fd) ## FPRINT
        if( comment is not None):
            print("#"+comment,file=fd)                  ## FPRINT
    ret=None
    if( additional_fields is not None):
        for key in additional_fields:
            sdum=str(key)+":"+str( additional_fields[key])
            if( not return_None):
                if( ret is None ):
                    ret=sdum;   ### print("#RET:"+ret)
                else:
                    ret += delimiter+sdum;   ### print("#ret:"+ret)
            if( fd is not None):
                print(sdum, file=fd)                    ## FPRINT
    Nerr=0;
    for Iky in range(nkey):
        if(verbose>2):
            print("#%d:%s"%(Iky,str(ret)),flush=True)
        key=keylist[Iky].strip();le=len(key);### print("#Top of the loop:%d:"%(Iky)+key)

        if( fields_excluded is not None):
            if(key in fields_excluded):
                continue
        ###if(Iky==0):
        ###    Stdlogger.printout(1,"#serialize:"+header+":skipping:0th:"+ key,verbose=verbose); continue
        if(key[0]=='_' and key[le-1]=='_'):
            Stdlogger.printout(1,"#serialize:"+header+":SKipping:_xxx_"+ key,verbose=verbose, wt0=wt0,dtme=True); continue
        if(le>1):
            if( key[le-1]=='_' and key[le-2]=='_'): 
                Stdlogger.printout(1,"#serialize:"+header+":SKipping:Xxx__"+ key,verbose=verbose, wt0=wt0,dtme=True); continue
        if(key.startswith("staticfield_")):
            Stdlogger.printout(1,"#serialize:"+header+":SKipping:staticfield"+ key,verbose=verbose, wt0=wt0,dtme=True); continue
                    
        val=getattr(obj,key);strval=None
        if( val is None ):
            ### print("#!W "+key+" val is None");
            if( not return_None):
###                ret+=delimiter+key+":o:None";
                if( ret is None):
                    ret= key+":o:None";
                else:
                    ret+=delimiter+key+":o:None";
            continue
        if( serializers is not None ):
            if( key in serializers ):
                print("#serializing:key=%s"%(key),flush=True)
                typ='o'
                strval=serializers[key](val)
        if( (strval is None) and is_method_or_function(val)>0 ):
            Stdlogger.printout(1,"       skip method.. KEY=%s VAL=%s"%( str(key),str(val)),verbose=verbose)
            continue;
        if( strval is None ):
            Stdlogger.printout(1,"#serialize:"+header+" key:"+key,verbose=verbose);types=[str,bool,int,float,complex]
            if( val is None ):
                typ='o';strval="None"
            elif( isinstance(val,str) ):
                Stdlogger.printout(2,key+":str",verbose=verbose, wt0=wt0,dtme=True)
                typ='s';strval=str(val)
            elif( isinstance(val,bool) ):   ## << check bool first:  isinstance( BOOLEAN, int ) returns -True- .. 
                Stdlogger.printout(2,key+":bool",verbose=verbose, wt0=wt0,dtme=True)
                typ='b';strval=str(val)
            elif( isinstance(val,int) ):
                Stdlogger.printout(2,key+":int",verbose=verbose, wt0=wt0,dtme=True)
                typ='i';strval="%d"%(val)
            elif( isinstance(val,float) or isinstance(val,np.float64) ):
                Stdlogger.printout(2,key+":float",verbose=verbose, wt0=wt0,dtme=True)
                typ='d';strval="%24.12e"%(val)
            elif( isinstance(val,complex) or isinstance(val,np.complex128) ):
                Stdlogger.printout(2,key+":complex",verbose=verbose, wt0=wt0,dtme=True)
                typ='z';strval="%24.12e %24.12e"%(val.real, val.imag)
            elif( isinstance(val,list) or isinstance(val,np.ndarray) ):
                Stdlogger.printout(2,key+":list_or_ndarray:"+str( np.shape(val) ),verbose=verbose)
                Ndim=np.shape(val)
                a1D =np.ravel( np.array(val) );dty=a1D.dtype
                Ld_tot=len(a1D)
                if( (Ld_tot > Ld_uplm) and (fpath is not None) ):
                    fnme=( fpath if(not fpath.endswith(".pscf")) else fpath.replace(".pscf",""))
                    fpath2=fnme+"_"+str(key)+".xpscf"
                    if( type(a1D[0])==str or type(a1D[0])==np.str_ ):
                        typ='S'+iNtoa(Ndim,delimiter=',');strval=sNtoa(val,delimiter="  ")
                    else:
                        assert key not in keys_written_on_file,"key:"+key+" is not unique.."+str(keys_written_on_file)
                        keys_written_on_file.append(key);
                        if( dty==int ):
                            typ='I'+iNtoa(Ndim,delimiter=',');strval="!"+fpath2;svld_aNbuf("S",fpath2,buf=val,comment=fpath)
                        elif( dty==float or dty==np.float64 ):
                            typ='D'+iNtoa(Ndim,delimiter=',');strval="!"+fpath2;svld_aNbuf("S",fpath2,buf=val,comment=fpath)
                        elif( dty==complex or dty==np.complex128 ):
                            typ='Z'+iNtoa(Ndim,delimiter=',');strval="!"+fpath2;svld_aNbuf("S",fpath2,buf=val,comment=fpath)
                        else:
                            strval=str(val);
                            print("#serialize:"+"dty:"+str(dty)+" KEY:"+str(key)+" type:"+str(type(val)))
                            print("#serialize:"+str(val))
                            Stdlogger.printout(1,"#Serialize:"+header+str(type(obj))+":"+str(obj),verbose=verbose)
                            Stdlogger.printout(1,"key:"+str(key)+" type:"+str(type(val))+" dtype:"+str(np.array(val).dtype)+" val:"+str(val),verbose=verbose)
                            assert False,""
                elif( len(a1D) == 0 ):
                    if( isinstance(val,list) ):
                        typ='I0';strval=""
                    else:
                        typ=('I0' if(val.dtype==int) else ('D0' if(val.dtype==float or val.dtype==np.float64) else 
                            ('Z0' if(val.dtype==complex or val.dtype==np.complex128) else 'S0')));strval=""
                else:
                    if( type(a1D[0])==str or type(a1D[0])==np.str_ ):
                        typ='S'+iNtoa(Ndim,delimiter=',');strval=sNtoa(val,delimiter="  ")
                    else:
                        if( dty==int ):
                            typ='I'+iNtoa(Ndim,delimiter=',');strval=iNtoa(val,delimiter="  ")
                        elif( dty==float or dty==np.float64 ):
                            typ='D'+iNtoa(Ndim,delimiter=',');strval=dNtoa(val,format="%24.12e",delimiter="  ")
                        elif( dty==complex or dty==np.complex128 ):
                            typ='Z'+iNtoa(Ndim,delimiter=',');strval=zNtoa(val,format="%24.12e %24.12e", delimiter="     ")
                        ### elif( dty==str ):
                        ###    typ='S'+iNtoa(Ndim,delimiter=',');strval=sNtoa(val,delimiter="  ")
                        else:
                            strval=str(val);
                            print("#serialize:"+"dty:"+str(dty)+" KEY:"+str(key)+" type:"+str(type(val)))
                            print("#serialize:"+str(val))
                            Stdlogger.printout(1,"#Serialize:"+header+str(type(obj))+":"+str(obj),verbose=verbose)
                            Stdlogger.printout(1,"key:"+str(key)+" type:"+str(type(val))+" dtype:"+str(np.array(val).dtype)+" val:"+str(val),verbose=verbose)
                            ### assert False,"dty:"+str(dty)+" KEY:"+str(key)+" type:"+str(type(val))
            elif( isinstance(val,dict) ):
                if( len(val) == 0 ):
                    typ='dict';strval="{}"
                else:
                    typ='dict';strval=None
                    for ky in val:
                        if(strval is None):
                            strval="{"+ str(ky).strip() + ","+ str( val[ky] ).strip()
                        else:
                            strval+=","+ str(ky).strip() + ","+ str( val[ky] ).strip()
                    strval+="}"
            else:
                assert key is not None,"key"
                assert val is not None,"val"
                Stdlogger.printout(1,"Key:"+key+":o:"+str(val)+ "...serializing object..",verbose=verbose)
                if(verbose>1):
                    print("###serialize:main:depth=%d:invoking serialize:Key=%s val=%s type=%s"%(depth,str(key),str(val),str(type(val))),flush=True)
                typ='o';strval=serialize(val, delimiter=';', depth=depth+1);
                if( strval is None ):
                    strval=""; print("!W serialize:"+key+":"+str(val)+" returns None")
                    ##if(not strict):
                    ##    Nerr=Nerr+1
                    ##else:
                    ##    assert strval is not None,"serialize:"+str(val)+" Key:"+str(key)
                strval=strval.strip()
                if( len(strval)==0 ):
                    strval="type:"+str(type(val))
                else:
                    strval="type:"+str(type(val))+";"+strval
        
        sdum=key+":"+typ+":"+strval
        if( fd is not None):
            print(sdum, file=fd)
        if( not return_None):
            if( ret is None):
                ret= sdum;   ### print("#Ret:"+ret)
            else:
                ret += delimiter + sdum;   ### print("#ret:"+ret)
    print("#serialize:those written on external file:"+str(keys_written_on_file))
    assert Nerr==0,"Nerr:%d"%(Nerr)
    return ret

## Ndiff,Nsame,Nerr
def diff_objects(lhs,rhs,verbose=0,diff_TOL=1.0e-7,dict_logs=None,keylist=None):   ## returns Ndiff,Nsame,Nerr
    if( isinstance(lhs,bool) or isinstance(lhs,int) or isinstance(lhs,str) ):
        if( lhs==rhs ):
            return 0,1,0;
        else:
            return 1,0,0;
    if( isinstance(lhs,float) or isinstance(lhs,np.float64) 
        or isinstance(lhs,np.complex128) or isinstance(lhs,complex) ):
        if( abs(lhs-rhs)<diff_TOL ):
            return 0,1,0
        else:
            scale=0.50*( abs(lhs)+abs(rhs) )
            if( scale > 1 and abs(lhs-rhs)/scale < diff_TOL ):
                Stdlogger.printout(1, "#OK: %s / %s diff:%f/%f= %e < TOL"%(str(lhs),str(rhs),abs(lhs-rhs),scale,abs(lhs-rhs)/scale))
                return 0,1,0
            Stdlogger.printout(1, "#DIFF: %s / %s diff:%f/%f= %e > TOL"%(str(lhs),str(rhs),abs(lhs-rhs),scale,abs(lhs-rhs)/scale))
            return 1,0,0;
    if(keylist is None):
        keylist=set_intersection( dir(lhs), dir(rhs),verbose=1)
    keylistR=dir(rhs)
    nkey=len(keylist)
    if( dict_logs is None ):
        dict_logs={}
    Nsame=0;Ndiff=0;Nerr=0;
    dict_logs.update({"diff":"","err":"","warning":""})
    nWarn=0
    for Iky in range(nkey):
        
        key=keylist[Iky].strip();le=len(key)
        ## if(Iky==0):
        ##    if(verbose>0):
        ##        print("#compare_objects:skipping_0th:"+ key);
        ##    continue
        if(key[0]=='_' and key[le-1]=='_'):
            Stdlogger.printout(2,"#compare_objects:skipping:"+ key,verbose=verbose);
            continue
        lh=getattr(lhs,key,None);rh=getattr(rhs,key,None)
        if( (lh is None) or (rh is None) ):
            if( (lh is None) and (rh is None) ):
                Nsame+=1;continue
            else:
                Nerr+=1;continue
        
        ismL=is_method_or_function(lh);ismR=is_method_or_function(rh)
        if( ismL or ismR ):
            if( ismL and ismR ):
                continue
            else:
                Nerr+=1;continue
        if( key in keylistR ):
            if( key == "bit_length" ):
                print(str(type(lhs))+" has field:"+key+":"+str(getattr(lhs,key))) ## ERROR
                print(str(type(rhs))+" has field:"+key+":"+str(getattr(rhs,key))) ## ERROR
                assert False,""
            if( isinstance(lh,dict) ):
                if( not isinstance(rh,dict)):
                    Nerr+=1;dict_logs["err"]+="#%d:%s lh:%s rh:%s"%(Nerr,key,str(type(lh)),str(type(rh)))
                else:
                    nd1,ns1,ne1,strlog1=diff_dict(lh,rh,verbose=0)
                    if( ne1>0 ):
                        Nerr+=1; dict_logs["err"]+="#%d:%s lh:%s rh:%s diff:%d,%d,%s"%(Nerr,key,str(lh),str(rh),nd1,ns1,ne1,strlog1)
                    elif(nd1>0):
                        Ndiff+=1; dict_logs["diff"]+="#%d:%s lh:%s rh:%s diff:%d,%d,%s"%(Nerr,key,str(lh),str(rh),nd1,ns1,ne1,strlog1)
                    elif(ns1>0):
                        Nsame+=1
                    else:
                        dict_logs["err"]+="###:%s lh:%s rh:%s diff fails:%d,%d,%d,%s"%(key,str(lh),str(rh),nd1,ns1,ne1,strlog1)
                continue
            elif( isinstance(lh,str)):
                if( isinstance(rh,str)):
                    if(lh == rh ):
                        Nsame+=1
                    else:
                        Ndiff+=1; dict_logs["diff"]+="#%d:%s:  %s / %s"%(Ndiff,key,str(lh),str(rh))
                else:
                    Nerr+=1; dict_logs["err"]+="#%d:%s:  %s / %s"%(Nerr,key,str(lh),str(rh))
            elif( isinstance(lh,float)   or isinstance(lh, np.float64) or 
                  isinstance(lh,complex) or isinstance(lh, np.complex128) ):
                if( abs(lh-rh)< diff_TOL ):
                    Nsame+=1
                else:
                    scale=0.50*( abs(lh) + abs(rh) )
                    if( scale>1 and abs(lh-rh)/scale < diff_TOL ):
                        Nsame+=1
                    else:
                        Ndiff+=1; dict_logs["diff"]+="#%d:%s:  %s / %s"%(Ndiff,key,str(lh),str(rh))
            elif( isinstance(lh,list) or isinstance(lh,np.ndarray) ):
                lh=np.ravel(lh);rh=np.ravel(rh)
                nlh=len(lh);nrh=len(rh);
                if(nlh==nrh):
                    jdiff=None
                    for j in range(nlh):
                        if( isinstance(lh[j],str) or isinstance(lh[j],int) ):
                            if( lh[j]==rh[j] ):
                                continue
                            else:
                                jdiff=j;break
                        elif( isinstance(lh[j],float) or isinstance(lh[j],np.float64) or 
                              isinstance(lh[j],complex) or isinstance(lh[j],np.complex128) ):
                            df=abs( lh[j]-rh[j] );
                            if( df< diff_TOL ):
                                continue;
                            else:
                                scale=0.50*( abs(lh[j]) + abs(rh[j]) )
                                if( scale > 1.0 and abs(lh[j]-rh[j])/scale < diff_TOL):
                                    continue
                                else:
                                    jdiff=j;break
                        else:
                            if(nWarn<5):
                                strWarn="#!W diff_objects:lh:%s %s rh:%s %s"%( str(type(lh)),str(lh),str(type(rh)),str(rh) )
                                Stdlogger.printout(0,strWarn);
                                dict_logs["warning"]+=strWarn; nWarn+=1
                                
                            nd2,ns2,ne2=diff_objects(lh,rh,verbose=0)
                            if(ne2>0 or nd2>0):
                                jdiff=j;break
                    if(jdiff is None):
                        Nsame+=1
                    else:
                        Ndiff+=1; dict_logs["diff"]+="#%d:%s[%d]:  %s / %s"%( Ndiff,key,jdiff,str(lh[jdiff]),str(rh[jdiff]) )
                else:
                    Nerr+=1; dict_logs["err"]+="#%d:%s:  len %d / %d"%(Nerr,key,nlh,nrh)
            else:
                # --- comp btw. objects --- 
                ### print("#check:%s:"%(key),end="");print(type(lh),end="    ");print(type(rh))
                ### print("#check:%s:"%(key),end="");print(lh,end="    ");print(rh)
                if( lh is None or rh is None):
                    if( (rh is None) and (lh is None)):
                        Nsame+=1
                    else:
                        Ndiff+=1; dict_logs["diff"]+="#%d:%s:  %s / %s"%( Ndiff,key,str(lh),str(rh) )
                else:
                    nD,nS,nE=diff_objects(lh,rh,verbose=verbose,diff_TOL=diff_TOL) 
                    if(nE>0):
                        Nerr+=1; err+="#%d:%s:  %s / %s  Ndiff:%d Nsame:%d Nerr:%d"%(Nerr,key,str(lh),str(rh),nD,nS,nE);
                    else:
                        if(nD>0):
                            Ndiff+=1; dict_logs["diff"]+="#%d:%s:  %s / %s  %d %d %d"%( Ndiff,key,str(lh),str(rh),nD,nS,nE ) 
                        else:
                            Stdlogger.printout(1,"#diff:%s:"%(str(type(lhs))) + key+":"+str(lh)+"/"+str(rh)+" returns:%d,%d,%d"%(nD,nS,nE))
                            Nsame+=1; assert nS>0,""
    return Ndiff,Nsame,Nerr

## BELOW WE DEFINE CONSTRUCTORS ...
##
## constructors have to create objects from string input.
## usually you need to create default object and --load-- properties on it..
##
def construct_tdfield(string, delimiter=';'):
    if( string=="None" ):
        return None
    retv=None
    sbuf=string.split(delimiter)
    col0=sbuf.pop(0)
    sarr=col0.split(':');
    key=sarr[0].strip()
    assert (key=="type"),""        
    strtype=sarr[1].strip();
    dum=read_strtype(strtype)
    if( dum is not None):
        strtype=dum
    if( "CWField01" in strtype ):
        e0=0.0; omega=1.0; ts=0.0  # 2021.06.07 dummy_omega = 1.0
        retv=CWField01(e0, omega, ts); ### CWfield(e0, omega, ts)
    elif( "gaussianpulse01" in strtype ):
        freq_eV=1.0e-5; peakIntensity_WpeCM2=0.0
        retv=gaussianpulse01(freq_eV, peakIntensity_WpeCM2,fwhm_as=1.0)
    elif( "kickfield" in strtype ):
        ElectricfieldVector=[0.0, 0.0, 0.0]; tmKick_AU=0.0;
        retv=kickfield( ElectricfieldVector,tmKick_AU=tmKick_AU) 
    elif( "Trapezoidalpulse" in strtype ):
        freq_eV=1.0e-5; peakIntensity_WpeCM2=0.0
        retv=Trapezoidalpulse( freq_eV, peakIntensity_WpeCM2,tAsc_fs=1.0,tFlat_fs=1.0,tDsc_fs=1.0,tc_fs=1.5)
    else:
        assert False,"strtype:"+strtype
    
    retv = load_fromlist(retv, sbuf)
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    if( MPIsize<2 or MPIrank==0 ):
        logf="serialize_py_construct_tdfield.log"
        fd1=open(logf,"a");
        print("#construct_tdfield:"+str(rttddft_common.get_job(True))+" \t\t"+str( datetime.datetime.now() ),file=fd1)
        print(retv.tostring(),file=fd1)
        retv.printout_tdfields( 0.0, 24.0, 0.008, logf,Append=True,Threads=[0])
    ### print("Field:"+retv.tostring()); 
    ### assert False,""
    return retv;
def serialize_tdfield(this,delimiter=';'):
    if( this is None ):
        return "None"
    ret="type:"+str(type(this))+delimiter+serialize(this,delimiter=delimiter)
    return ret
    #if( isinstance(this,CWField01) ):
    #    ret += delimiter+"e0:%24.12e"%(this.e0) +delimiter+"omega:%24.12e"%(this.omega) 
    #         + delimiter+"ts:%24.12e"%(this.ts)
    #elif( isinstance(this, gaussianpulse01) ):
    #    
    #    ret += delimiter+"omega:%24.12e"%(this.omega) +delimiter+"

def i1prod(ibuf):
    buf=np.ravel(ibuf)
    le=len(buf)
    if(le<1):
        return 0;
    ret=buf[0];
    for j in range(1,le):
        ret*=buf[j]
    return ret;

def a1_to_a(buf,delimiter=","):
    if( isinstance(buf,np.ndarray) ):
        dtype=buf.dtype;
    elif( isinstance(buf,list) and len(buf)>0 ):
        dtype=np.array( buf[0] ).dtype
    else:
        dtype=np.complex128
    
    if(dtype==int or dtype==np.int64):
        return i1_to_a(buf,delimiter=delimiter)
    elif(dtype==float or dtype==np.float64):
        return d1_to_a(buf,delimiter=delimiter)
    elif(dtype==complex or dtype==np.complex128):
        return z1_to_a(buf,delimiter=delimiter)
    else:
        assert False,""+str(dtype)
def i1_to_a(ibuf,format="%d",delimiter=","):
    buf=np.ravel(ibuf)
    le=len(buf)
    sbuf=[ format%(buf[j]) for j in range(le) ]
    return delimiter.join(sbuf)
#    ret="";dlmt=""
#    for j in range(le):
#        ret+=dlmt+format%(buf[j]);dlmt=delimiter
#    return ret
def d1_to_a(dbuf,format="%24.8e",delimiter=","):
    buf=np.ravel(dbuf)
    le=len(buf)
    sbuf=[ format%(buf[j]) for j in range(le) ]
    return delimiter.join(sbuf)
#    ret="";dlmt=""
#    for j in range(le):
#        ret+=dlmt+format%(buf[j]);dlmt=delimiter
#    return ret

def z1_to_a(zbuf,format="%24.8e,%24.8e",delimiter=",  "):
    buf=np.ravel(zbuf)
    le=len(buf)
    sbuf=[ format%(buf[j].real, buf[j].imag) for j in range(le) ]
    return delimiter.join(sbuf)
#    ret="";dlmt=""
#    for j in range(le):
#        ret+=dlmt+format%( buf[j].real, buf[j].imag );dlmt=delimiter
#    return ret

def prtI1(fd,ibuf,format="%d",delimiter=",",ncol=6):
    buf=np.ravel(ibuf)
    le=len(buf)
    ret="";dlmt=""
    nrow=(le+ncol-1)//ncol;
    ixj=0
    for i in range(nrow):
        ret="";dlmt=""
        nc=min(ncol,(le-ixj))
        sbuf=[ format%( buf[ixj+k] ) for k in range(nc)]
        print( delimiter.join(sbuf),file=fd )
        ixj+=nc
        ## for j in range(nc):
        ##    ret+=dlmt+format%(buf[ixj]); ixj+=1; dlmt=delimiter
        ## print(ret,file=fd);
    assert ixj==le,""
    
def prtD1(fd,dbuf,format="%24.8e",delimiter=",",ncol=6):
    buf=np.ravel(dbuf)
    le=len(buf)
    ret="";dlmt=""
    nrow=(le+ncol-1)//ncol;
    ixj=0
    for i in range(nrow):
        ret="";dlmt=""
        nc=min(ncol,(le-ixj))
        sbuf=[ format%( buf[ixj+k] ) for k in range(nc)]
        print( delimiter.join(sbuf),file=fd )
        ixj+=nc
    assert ixj==le,""
##        for j in range(nc):
##            ret+=dlmt+format%(buf[ixj]); ixj+=1; dlmt=delimiter
##        print(ret,file=fd);

def prtZ1(fd,zbuf,format="%24.8e,%24.8e",delimiter=",  ",ncol=4):
    # print(np.shape(zbuf))
    buf=np.ravel(zbuf)
    le=len(buf)
    # print(np.shape(buf),le)
    ret="";dlmt=""
    nrow=(le+ncol-1)//ncol;
    ixj=0
    for i in range(nrow):
        ret="";dlmt=""
        nc=min(ncol,(le-ixj))
        sbuf=[ format%( buf[ixj+k].real, buf[ixj+k].imag ) for k in range(nc) ]
        print( delimiter.join(sbuf),file=fd )
        ixj+=nc
    assert ixj==le,""
##        for j in range(nc):
##            ret+=dlmt+format%( buf[ixj].real, buf[ixj].imag ); ixj+=1; dlmt=delimiter
##        print(ret,file=fd);

def svld_aNbuf(SorL,path,buf=None,comment=None,get_comments=None):
    STRdtype=None;dtype=None
    if(SorL=='S'):
        if( isinstance(buf,np.ndarray) ):
            dtype=buf.dtype;
        elif( isinstance(buf,list) and len(buf)>0 ):
            dtype=np.array( buf[0] ).dtype
        else:
            dtype=np.complex128

        STRdtype=( "I" if(dtype==int or dtype==np.int64) else \
                    ( "D" if(dtype==float or dtype==np.float64) else \
                        ( "Z" if(dtype==complex or dtype==np.complex128) else None ))) 
        assert STRdtype is not None,"dtype:"+str(dtype)

        Ndim=np.shape(buf)
        fd=open(path,'w')
        if( comment is not None ):
            print("#%s"%(comment),file=fd);
        print("%s:%s"%(STRdtype,i1_to_a(Ndim)),file=fd)
        if(dtype==int or dtype==np.int64):
            prtI1(fd,buf)
        elif(dtype==float or dtype==np.float64):
            prtD1(fd,buf)
        elif(dtype==complex or dtype==np.complex128):
            prtZ1(fd,buf)
        fd.close()
    else:
        fd=open(path,'r')
        STRdtype=None;ret=None
        Ngot=0;
        for line in fd:
            if( line[0]=='#'):
                if( get_comments is not None ):
                    get_comments.append(line)
                print("reading off:"+line);continue
            if( STRdtype is None ):
                sA=line.split(":")
                STRdtype=sA[0];
                dtype=(np.int64 if(STRdtype=='I') else \
                       (np.float64 if(STRdtype=='D') else \
                        (np.complex128 if(STRdtype=='Z') else None))); assert dtype is not None,""+STRdtype

                sub=sA[1].split(',');n1=len(sub)
                Ndim=[ int(sub[j]) for j in range(n1) ]
                Ldim=i1prod(Ndim)
                if( STRdtype == 'Z' ):
                    ret=np.zeros([2*Ldim],dtype=np.float64)
                else:
                    ret=np.zeros([Ldim],dtype=dtype)
                continue
            sA=line.split(',');nA=len(sA)
            if(STRdtype=='I'):
                for j in range(nA):
                    ret[Ngot+j]=int(sA[j])
                Ngot+=nA;
            elif(STRdtype=='D'):
                for j in range(nA):
                    ret[Ngot+j]=np.float64(sA[j])
                Ngot+=nA;
            elif(STRdtype=='Z'):
                for j in range(nA):
                    ret[Ngot+j]=np.float64(sA[j])
                Ngot+=nA;
        fd.close()
        if( STRdtype=='Z' ):
            assert Ngot==2*Ldim,"read:%d/2*Ldim:%d"%(Ngot,2*Ldim)
            wks=np.zeros([Ldim],dtype=dtype)
            for j in range(Ldim):
                wks[j]= ret[2*j] + 1j*ret[2*j+1]
            ret=wks
        else:
            assert Ngot==Ldim,"read:%d/Ldim:%d"%(Ngot,Ldim)
        
        return np.reshape(ret,Ndim)

def i1eqb(lhs,rhs):
    if(len(lhs)!=len(rhs)):
        return False
    le=len(lhs)
    for j in range(le):
        if(lhs[j]!=rhs[j]):
            return False
    return True
def diff_aNbuf(lhs,rhs,diff_thr=1.0e-6,label=None,fpath=None):
    Ndim=np.shape(lhs)
    Ndim_R=np.shape(rhs)
    lh=lhs;rh=rhs
    if( not i1eqb(Ndim,Ndim_R) ):
        rh=np.reshape(rhs,Ndim)
    rank=len(Ndim)
    if(rank>4):
        Ndim_2=[-1,Ndim[rank-3],Ndim[rank-2],Ndim[rank-1]]
        lh=np.reshape(lhs,Ndim_2)
        rh=np.reshape(rhs,Ndim_2)
        Ndim=Ndim_2
    fd=sys.stdout
    if(fpath is not None):
        fd=open(fpath,"w")

    diff=aNmaxdiff(lhs,rhs)
    if( label is not None ):
        print("#diff_aNbuf:%s:diff=%e"%(label,diff), file=fd);
    if( diff < diff_thr ):
        return diff
    if(rank<2):
        print("#: %e\n<%s\n>%s"%(I, a1_to_a(lhs),a1_to_a(rhs)), file=fd)
        return diff
    for I in range(Ndim[0]):
        if(rank<3):
            dum=aNmaxdiff(lhs[I],rhs[I])
            if(dum>diff_thr):
                print("#%d: %e\n<%s\n>%s"%(I,dum, a1_to_a(lhs[I]),a1_to_a(rhs[I])), file=fd)
            continue
        for J in range(Ndim[1]):
            if(rank<4):
                dum=aNmaxdiff(lhs[I][J],rhs[I][J])
                if(dum>diff_thr):
                    print("#%d,%d: %e\n<%s\n>%s"%(I,J,dum, a1_to_a(lhs[I][J]),a1_to_a(rhs[I][J])), file=fd)
                continue

            for K in range(Ndim[2]):
                dum=aNmaxdiff(lhs[I][J][K],rhs[I][J][K])
                if(dum>diff_thr):
                    print("#%d,%d,%d: %e\n<%s\n>%s"%(I,J,K,dum, a1_to_a(lhs[I][J][K]),a1_to_a(rhs[I][J][K])), file=fd)
    if(fpath is not None):
        fd.close()
        os.system("ls -ltrh "+fpath)
    return diff
