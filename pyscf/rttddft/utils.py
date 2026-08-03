import numpy as np
import math
import os
import os.path
import sys
import time
from .futils import futils
import scipy
import scipy.linalg
import datetime
from mpi4py import MPI
from .physicalconstants import PhysicalConstants

class utils_static_:
    counter_={}
    keys_write_once_=[]
    @staticmethod
    def Countup(key,inc=True):
        if( key not in utils_static_.counter_ ):
            utils_static_.counter_.update({key:0})
        if(inc):
            utils_static_.counter_[key]+=1
        return utils_static_.counter_[key]

def write_once(key,text,fnme_format=None,fpath=None,Threads=None,Append=False,stdout=False,dtme=True):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank(); MPIsize=comm.Get_size()
    if( Threads is None ):
        Threads = ( [0] if(fpath is not None) else \
                    ( range(MPIsize) if( fnme_format is not None ) else [0] ))
    elif( isinstance(Threads,str) ):
        if( Threads.strip()=='all' ):
            Threads= range(MPIsize)
    if( MPIrank not in Threads ):
        return
    if( key in utils_static_.keys_write_once_ ):
        return
    utils_static_.keys_write_once_.append(key)
    fdA=[];fdOU=None;path=None
    if( fpath is None ):
        if( fnme_format is None):
            fdA.append(sys.stdout)
        else:
            path=fnme_format%(MPIrank)
    else:
        path=fpath
    if(path is not None):
        fdOU=open(fpath,('a' if(Append) else 'w'));fdA.append(fdOU)
        if( stdout ):
            fdA.append(sys.stdout)
    if(dtme):
        text+=' \t\t'+str(datetime.datetime.now())
    
    for fd in fdA:
        print("#write_once:"+str(key)+":"+text,file=fd);
    if( fdOU is not None ):
        fdOU.close()

def i1prod(A):
    le=len(A)
    ret=1
    for j in range(le):
        ret*=A[j]
    return ret
def prtout_MOocc(fnme, mf,pbc,spinrestriction,Threads=[0]):
##           RKS        ROKS                       UKS           
##  mo_occ   [nMO]      [2,nMO]                      [2,nMO]
##            KRKS
##  mo_occ    [nKPT,nMO]      [2,nKPT,nMO]           [2,nKPT,nMO]

    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank();MPIsize=comm.Get_size()
    if( MPIsize>0 and (MPIrank not in Threads) ):
        return
#    print("prtout_MOocc:"+str(type(mf.mo_occ))) #<class 'numpy.ndarray'>
#    print("prtout_MOocc:"+str(len(mf.mo_occ)))
#    print("prtout_MOocc:"+str(np.shape(mf.mo_occ[0])))
    

    MO_occ=np.array( mf.mo_occ );
    Ndim=np.shape(MO_occ);leNdim=len(Ndim)
    if(pbc):
        assert( (spinrestriction=='R' and leNdim==2) or 
                (spinrestriction!='R' and leNdim==3) ),"";
    else:
        assert( (spinrestriction=='R' and leNdim==1) or 
                (spinrestriction!='R' and leNdim==2) ),"";
    
    nmult_OCC=(1 if(spinrestriction=='R') else 2)
    for sp in range(nmult_OCC):
        Occ=( MO_occ if(nmult_OCC==1) else MO_occ[sp])
        fd=open( (fnme if(nmult_OCC==1) else fnme.replace(".dat","")+"_upSpin.dat"),"w")
        nKpt=(1 if(not pbc) else (Ndim[0] if(spinrestriction=='R') else Ndim[1]) );
        nMO =( (Ndim[0] if(nmult_OCC==1) else Ndim[1]) if(not pbc) else \
               (Ndim[1] if(spinrestriction=='R') else Ndim[2] ) )
        for kp in range(nKpt):
            if(pbc):
                print("#%03d"%(kp),file=fd)
            occ=( Occ if(not pbc) else Occ[kp] );
            string=""
            for jmo in range(nMO):
                string+="%16.8f "%(occ[jmo])
            print(string,file=fd);
        fd.close()

def print_00(text,flush=False,file=sys.stdout,end='\n',dtme=False,warning=0,Threads=[0]):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank(); MPIsize=comm.Get_size()
    if( isinstance(Threads,str) ):
        if( Threads.strip()=='all' ):
            Threads= range(MPIsize)
    if( (MPIrank not in Threads) and (warning >=0) ):
        return
    if(dtme):
        text+=' \t\t'+str(datetime.datetime.now())
    if( file is None ):
        file=sys.stdout
    print(text,flush=flush,file=file,end=end)

def dbglogger(msg,refresh=False,stdout=False):
    print_00(msg);

def read_inpf(dict_ret,fnme,silent=False,check_all_keys=True,abort_if_fail=True):
    ngot=0
    if ( os.path.exists(fnme)):
        inpf=futils.fopen( fnme,"r");## OK
        if( not silent):
            dbglogger("##read_inpf:%s"%(fnme),stdout=True);
        for line in inpf:
            line=line.strip()
            if( line=="" ):
                continue
            if( line.startswith('#') ):
                print_00("#read_inpf:reading off:"+line); continue
            a=line.split(':')
            if( len(a)>=2 ):
                ky_in_dict = False
                for ky in dict_ret:
                    if(a[0] == ky):
                        ky_in_dict=True
                        a[1]=a[1].rstrip();
                        dict_ret[ky]=a[1];ngot=ngot+1
                        if( not silent ):
                            dbglogger("##read_inpf:set_param:%s %s"%(ky,a[1]))
                        break
                if( check_all_keys ):
                    assert ky_in_dict,"key:"+a[0]+" is not recognized"
            if( len(a)!=2 ):
                if( not silent ):
                    dbglogger("##read_inpf:possible parse error:len=%d for"%(len(a))+line);
        futils.fclose(inpf)
        return ngot
    else:
        if(abort_if_fail):
            assert False,"missing file:"+fnme
        if( not silent ):
            dbglogger("##read_inpf:no such file:"+fnme)
        return 0
def print_params(job,params):
    ##if( world.rank != 0 ):
    ##    return
    fd=futils.fopen(job+"_params.log","w");## OK
    for ky in params:
        if( params[ky] is None ):
            print_00(ky+": none",file=fd);
        else:
            print_00(ky+":"+str(params[ky]),file=fd)
    futils.fclose(fd)

def i1eqb(a,b,verbose=False,title=""):
    ndimA=np.shape(a);ndimB=np.shape(b)
    if( not i1eqb1(ndimA,ndimB) ):
        if(verbose):
            print_00("i1eqb:"+title+":dimension differs:A:",end="");print_00(ndimA,end="");print_00(" B:",end="");print_00(ndimB);
        return False
    return i1eqb1(a,b,verbose,title)
def i1eqb1(a,b,verbose=False,title=""):
    A=np.ravel(a);B=np.ravel(b)
    lA=len(A);lB=len(B)
    if(lA!=lB):
        return False
    for I in range(lA):
        if(A[I] != B[I]):
            if(verbose):
                print_00("#i1eqb1:"+title+":[%d] %d / %d"%(I,A[I],B[I]))
            return False
    return True

def toString(a,Nmax=None,format=None,polar=False):
    A=np.ravel(a)
    Ld=len(A);
    Nw=(Ld if(Nmax is None) else min([Ld,Nmax]))
    dtype=np.array(A).dtype
    ret=""
    if( (dtype == complex) or (dtype == np.complex128) ):
        if( not polar ):
            if format is None: format="%14.6f %14.6fj   ";
            for I in range(Nw):
                ret+=format%(A[I].real,A[I].imag)
        else:
            if format is None: format="%14.4e %6.3fpi   ";
            for I in range(Nw):
                ret+=format%( abs(A[I]), np.atan2(A[I].imag, A[I].real));
    else:
        if format is None: format="%14.6f   ";
        for I in range(Nw):
            ret+=format%(A[I])
    return ret;

def a1maxdiff(A,B):
    I=0;dev=abs(A[I]-B[I]);at=I
    La=len(A);Lb=len(B)
    assert (La==Lb),"a1maxdiff"
    for I in range(1,La):
        dum=abs(A[I]-B[I]);
        if(dum>dev):
            dev=dum;at=I
    return dev

def d1maxdiff(A,B):
    return a1maxdiff(A,B)

def d1diff(A,B,length=-1):
    return math.sqrt( d1sqrediff(A,B,length=length))

def d1dist(A,B,length=-1):
    return math.sqrt( d1sqrediff(A,B,length=length))
def d2dist(A,B,length=-1):
    return d1dist(np.ravel(A), np.ravel(B))

def d1sqrediff(A,B,length=-1):
    La=len(A);Lb=len(B)
    L=length
    if( L < 0 ):
        L=min(La,Lb)
    else:
        L=min( length, min(La,Lb))
    ret=0.0
    for i in range(L):
        ret = ret + (A[i]-B[i])*(A[i]-B[i]);
    return ret

def z1diff(A,B,title=None,err=None):
    N=len(A)
    dev=abs(A[0]-B[0]);at=0
    for I in range(1,N):
        dum=abs(A[I]-B[I]);
        if(dev<dum):
            dev=dum;at=I
    if( err is not None):
        assert (dev<err),"#z1diff:"+(title if (title is not None) else "")+\
                ":%e @%d:%10.4f+j%10.4f / %10.4f+j%10.4f"%(dev,at,A[at].real,A[at].imag,B[at].real,B[at].imag)
    else:
        if(title is not None):
            print_00("#z1diff:"+title+":%e @%d:%10.4f+j%10.4f / %10.4f+j%10.4f"%(dev,at,A[at].real,A[at].imag,
                B[at].real,B[at].imag));
    
    return dev

def z2diff(A,B,title=None,tol=None,err=None):
    N=len(A);M=len(A[0])
    dev=-1;at=[-1,-1]
    for I in range(N):
        for J in range(M):
            dum=abs(A[I][J]-B[I][J]);
            if(dev<dum):
                dev=dum;at=[I,J]
    if( (title is not None) or (tol is not None and dev>tol) ):
        print_00("#z2diff:%s:dev=%e @[%d,%d] %10.4f+j%10.4f / %10.4f+j%10.4f"%(
                title if(title is not None) else "", dev, at[0],at[1],
                A[at[0]][at[1]].real, A[at[0]][at[1]].imag,
                B[at[0]][at[1]].real, B[at[0]][at[1]].imag))
    if(err is not None):
        assert (dev<err),"z2diff:dev=%e"%(dev)
    return dev

def aNsqrediff(lhs,rhs):
    dtype=np.array(lhs).dtype
    lh=np.ravel(lhs)
    rh=np.ravel(rhs)
    dum = np.vdot( lh-rh, lh-rh )
    if( (dtype is complex) or (dtype is np.complex128) ):
        return dum.real
    else:
        return dum

def print_bset(fnme,cell,Append=False,Threads=[0]):
    _basis=cell._basis

    if(_basis is None ):
        return -1
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank();MPIsize=comm.Get_size()
    if( MPIsize > 1 and (MPIrank not in Threads) ):
        return
    spdfgh_=[ 's','p','d','f','g','h','i','j']
    fd=open(fnme,('a' if(Append) else 'w'))
    for elmt in _basis:
        name=None
        if( isinstance(cell.basis,str) ):
            name=cell.basis
        elif( isinstance(cell.basis,dict) ):
            if( elmt in cell.basis ):
                name=cell.basis[elmt]
        if( name is None ):
            name=str( cell.basis )
        
        bs_A=_basis[elmt]
        nblock=len(bs_A)
        for iblc in range(nblock):
            Buf=bs_A[iblc]
            Buflen=len(Buf); NpGTO=Buflen-1
            ell=Buf[0]
            string=" %s %3s %s "%(name,elmt,spdfgh_[ell])
            for J in range(1,Buflen):
                alph_cofs = Buf[J]
                line=" %18.8e           "%(alph_cofs[0])
                ncols=len(alph_cofs)
                for j in range(1,ncols):
                    line+="%16.6e   "%(alph_cofs[j])
                string+='\n'+line
            print(string,file=fd)
    ### print(_basis)
    ### assert False,""
    fd.close()

def make_tempfile(head=None,tail=""):
    retv=None;iter=0
    if( head is None ):
        head="temp"
    while(retv is None):
        iter+=1
        t_ms=int( round( time.time()*1000 ) )
        pid = os.getpid()
        fnme= head + "_%08d"%(int(round(float(t_ms)*float(pid)))) + tail
        if(os.path.exists(fnme)):
            if(iter>20):
                assert False,""
            print_00("!W tempfile:"+fnme+" exists..",warning=-1);continue
        else:
           retv=fnme; break
    return retv

def dNtoa(buf,format="%14.6f",delimiter=" "):
    return d1toa(np.ravel(buf),format=format,delimiter=delimiter)

def d1toa(arr,format="%14.6f",delimiter=" "):
    if( isinstance(arr,float) or isinstance(arr,np.float64) or isinstance(arr,int) ):
        return format%(arr)
    Dic={};wt0=time.time();wt1=wt0 # XXX XXX
    A=np.ravel( np.array(arr) )
    wt2=wt1;wt1=time.time(); Dic.update({"ravel":wt1-wt2})
    nA=len(A)
    
    sbuf=[ format%(A[j]) for j in range(nA) ]
    wt2=wt1;wt1=time.time(); Dic.update({"strjoin":wt1-wt2,"tot":wt1-wt0})
    ## print("d1toa_join,%6d:"%(nA)+str(Dic))
    return delimiter.join(sbuf)

def d2toa(a,format="%14.6f ",delimiter="\t\t\t "):
    ret="";
    for I in range(len(a)):
        for J in range(len(a[0])):
            ret=ret + format%(a[I][J])
        ret=ret+delimiter
    return ret

def zNtoa(A,format="%10.4f %10.4f",delimiter=" ",Nlimit=None):
    a=np.ravel(A)
    return z1toa(a,format=format,delimiter=delimiter,Nlimit=Nlimit)

def z1toa(arr,format="%10.4f %10.4f",delimiter=" ",Nlimit=None):
    if( isinstance(arr,complex) or isinstance(arr,np.complex128) ):
        return format%(arr.real,arr.imag)
    Dic={};wt0=time.time();wt1=wt0 # XXX XXX
    A=np.ravel( np.array(arr) )
    wt2=wt1;wt1=time.time(); Dic.update({"ravel":wt1-wt2})
    nA=len(A)
    
    sbuf=[ format%(A[j].real,A[j].imag) for j in range(nA) ]
    wt2=wt1;wt1=time.time(); Dic.update({"strjoin":wt1-wt2,"tot":wt1-wt0})
    print("z1toa_join,%6d:"%(nA)+str(Dic))
    return delimiter.join(sbuf)

def z1toa_polar(A,format="%8.3e %4.2fpi  "):
    N=len(A);ret=""
    pi=3.1415926535897932384626433832795
    for I in range(N):
        r=abs(A[I]);th_pi=math.atan2( A[I].imag, A[I].real)/pi
        ret+=format%(r,th_pi)
    return ret

def z2toa(a,format="%8.3f+%8.3fj   ",delimiter="\t\t\t "):
    if( a.dtype == float ):
        return d2toa(a);
    ret="";
    for I in range(len(a)):
        for J in range(len(a[0])):
            ret=ret + format%( a[I][J].real, a[I][J].imag )
        ret=ret+delimiter
    return ret

def write_xyzf(fnme,R,S,description="",Eng=None,ResidualForce=None,LatticeVectors=None,a=None):
    Nat=len(S)
    fd=futils.fopen(fnme,"w");## OK
    ### print("#write_xyzf:",len(R),len(S),fnme)
    print_00("%d"%(Nat),file=fd)
    secondline="#"+description
    if( Eng is not None ):
        secondline+="E=%16.8f "%(Eng)
    if( ResidualForce is not None ):
        secondline+="ResidualForce=%14.4e "%(ResidualForce)
    if( a is not None ):
        assert LatticeVectors is None,""
        LatticeVectors=a
    if( LatticeVectors is not None ):
        vec=np.ravel(LatticeVectors); le=len(vec); assert le==9,""
        sdum="%14.6f"%(vec[0]);sdum=sdum.strip();
        for I in range(1,le):
            s="%14.6f"%(vec[I])
            sdum=sdum+','+s.strip();sdum=sdum.strip()
        secondline+=" a:"+sdum
    print_00(secondline,file=fd);

    for I in range(Nat):
        print_00("%s %16.8f %16.8f %16.8f"%(S[I],R[I][0],R[I][1],R[I][2]),file=fd)
        ### print("%s %16.8f %16.8f %16.8f"%(S[I],R[I][0],R[I][1],R[I][2]))
    futils.fclose(fd)

def check_equivalence(bool,title,values=None):
    from .rttddft_common import rttddft_common
    jth=rttddft_common.Countup("check_equivalence:"+title.strip())
    if( jth==1 or (not bool) ):
        fd=open("check_equivalence.log","a")
        nth=rttddft_common.Countup("check_equivalence")
        if(nth==1):
            print("\n##%s %s"%(rttddft_common.get_job(True),str(datetime.datetime.now())),file=fd)
        print("%d  %s:%s  %s\n"%(nth,title.strip(),str(bool),( str(values) if(values is not None) else "")),file=fd)
        fd.close()

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
        fac=PhysicalConstants.BOHRinANGS();
    str="";
    Nat=len(Sy)
    for Iat in range(Nat):
        str+="%s %f %f %f\n"%(Sy[Iat],R[Iat][0]*fac,R[Iat][1]*fac,R[Iat][2]*fac);
    return str;

def get_LatticeVectors(line,ierr=-1,filename=""):
    nGot=-1
    keys=["a:","BravaisVectors:","LatticeVectors:"]
    dbuf=None
    print_00("#get_LatticeVectors:"+line)
    for ky in keys:
        iat=line.find(ky);
        if( iat>=0 ):
            lky=len(ky);
            dbuf=parse_doublesx( line[iat+lky:], Nwant=9);
            print_00("get_LatticeVectors:parsing:"+line[iat+lky:]+str(dbuf))
            nGot=max(nGot,len(dbuf))
            if(len(dbuf)==9):
                return np.reshape(dbuf,[3,3])
    if( nGot>=0 ):
        if( nGot!=9 ):
            warningmsg="!E:"+filename+" wrong length of LatticeVectors:"+str(nGot)+str(dbuf)
            if(ierr<0):
                assert False,warningmsg
            else:
                print_00(warningmsg);
        else:
            dbuf=np.reshape(dbuf,[3,3])
        return dbuf;
    else:
        return None
    return None

def read_xyzf(fnme,sbuf=None,dict=None,output_unit=None):
    BOHRinANGS=None
    if( output_unit is not None ):
        assert (output_unit=='Bohr' or output_unit=='B'),"unknown output_unit:"+output_unit
        BOHRinANGS=PhysicalConstants.BOHRinANGS()
    fd=futils.fopen(fnme,"r");## OK
    nl=0;Nat=-1
    R=[];Sy=[];
    for line in fd:
        nl+=1
        line=line.strip()
        if(nl==1):
            Nat=int(line);continue
        elif(nl==2):
            if(sbuf is not None):
                sbuf.append(line)
            dbuf=get_LatticeVectors(line,filename=fnme)
            if(dbuf is not None):
                if(dict is not None):
                    dict.update({'a':dbuf})
                    print_00("LatticeVectors:"+str(dbuf) )
                else:
                    print_00("!W reading off LatticeVectors:"+str(dbuf) )
            else:
                print_00("reading off:"+line)
            continue
        else:
            line=line.strip();le=len(line)
            if(le<1):
                continue
            arr=line.split()
            Sy.append(arr[0].strip())
            if( output_unit is not None ):
                R.append([ float(arr[1])/BOHRinANGS,float(arr[2])/BOHRinANGS,float(arr[3])/BOHRinANGS])
            else:
                R.append([ float(arr[1]),float(arr[2]),float(arr[3])])
            
    futils.fclose(fd)
    return R,Sy

def dump_zNarray(zNa,fpath,Ncol=6,comment=None):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return

    fd=open(fpath,"w")
    a=np.ravel(zNa)
    if(comment is not None):
        print("#%s"%(comment),file=fd)
    le=len(a)
    Nrow=le//Ncol
    I=0
    for Irow in range(Nrow):
        line=""
        for J in range(Ncol):
            line+="%24.10e %24.10e    "%(a[I+J].real,a[I+J].imag)
        print(line,file=fd)
        I+=6
    while(I<le):
        print("%24.10e %24.10e    "%(a[I].real,a[I].imag),file=fd);I+=1
    fd.close()

def distinct_set(org):
    N=len(org);
    ret=[]
    for I in range(N):
        sdum=org[I].strip();
        if(sdum in ret):
            continue
        ret.append(sdum)
    return ret

def get_strPolarization(vec,tiny=1.0e-8,strxyz=['x', 'y','z'],residual_thr=1.0e-5,default=None):
    
    v=np.ravel(vec)
    le=len(v)
    norm=0.0;
    for k in range(le):
        norm+= v[k]**2
    norm=np.sqrt(norm)
    if( norm < tiny ):
        return default
    dir=[ v[k]/norm for k in range(le) ]
    dir_abs=[ abs(dir[k]) for k in range(le) ]
    ith_to_j = np.argsort(dir_abs);
    if( ith_to_j[le-1] < len(strxyz) ):
        retv=strxyz[ ith_to_j[le-1] ]
        residual=0.0;
        for ith in range(le-1):
            residual += v[ ith_to_j[ith] ]**2
        residual=np.sqrt(residual)
        if( residual < residual_thr ):
            return retv
        else:
            print_00("#get_strPolarization:large residual norm:%f .. dir="%(residual)+str(dir))
            return default
    else:
        print_00("check ith_to_j:"+str(ith_to_j) + " for:"+str(vec));
        return default

def arrayclone(org,dtype=None):
    ndim=np.shape(org)
    if dtype is None: dtype=np.array(org).dtype
    ret=np.zeros( ndim, dtype=dtype)
    ord=len(ndim)
    if( ord==4 ):
        for I in range(ndim[0]):
            for J in range(ndim[1]):
                for K in range(ndim[2]):
                    for L in range(ndim[3]):
                        ret[I][J][K][L]=org[I][J][K][L]
    elif( ord==3 ):
        for I in range(ndim[0]):
            for J in range(ndim[1]):
                for K in range(ndim[2]):
                    ret[I][J][K]=org[I][J][K]
    elif( ord==2 ):
        for I in range(ndim[0]):
            for J in range(ndim[1]):
                ret[I][J]=org[I][J]
    elif( ord==1 ):
        for I in range(ndim[0]):
            ret[I]=org[I]
    else:
        Ld=ndim[0];
        for k in range(1,ord):
            Ld=Ld*ndim[k]
        o_1D=np.ravel(org);
        r_1D=np.zeros([Ld])
        for k in range(Ld):
            r_1D[k]=o_1D[k]
        ret=np.reshape(r_1D,ndim)
    return ret

def arrayclonex(org,fac,dtype=None):
    ndim=np.shape(org)
    if dtype is None: dtype=np.array(org).dtype
    ret=np.zeros( ndim, dtype=dtype)
    ord=len(ndim)
    if( ord==4 ):
        for I in range(ndim[0]):
            for J in range(ndim[1]):
                for K in range(ndim[2]):
                    for L in range(ndim[3]):
                        ret[I][J][K][L]=org[I][J][K][L]*fac
    elif( ord==3 ):
        for I in range(ndim[0]):
            for J in range(ndim[1]):
                for K in range(ndim[2]):
                    ret[I][J][K]=org[I][J][K]*fac
    elif( ord==2 ):
        for I in range(ndim[0]):
            for J in range(ndim[1]):
                ret[I][J]=org[I][J]*fac
    elif( ord==1 ):
        for I in range(ndim[0]):
            ret[I]=org[I]*fac
    else:
        Ld=ndim[0];
        for k in range(1,ord):
            Ld=Ld*ndim[k]
        o_1D=np.ravel(org);
        r_1D=np.zeros([Ld])
        for k in range(Ld):
            r_1D[k]=o_1D[k]*fac
        ret=np.reshape(r_1D,ndim)
    return ret

def d1x2toa(A,B,format="%14.6f %14.6f   "):
    ret="";
    Na=len(A);Nb=len(B)
    N=min(Na,Nb)
    for j in range(N):
        ret+=format%(A[j],B[j])
    return ret

def i1toa(A,N=None,format="%d", delimiter=" "):
    if(isinstance(A,int) or isinstance(A,np.int64)):
        return format%(A)
    if N is None:N=len(A)
    sbuf=[ format%(A[j]) for j in range(N) ]
    return delimiter.join(sbuf)

def iNtoa(A,format="%d",delimiter=" "):
    a=np.ravel(A)
    return i1toa(a,format=format, delimiter=delimiter)

def sNtoa(arr,format="%s",delimiter=" "):
    return s1toa( np.ravel( np.array(arr) ),format=format, delimiter=delimiter) 

def s1toa(arr,format="%s",delimiter=" "):
    A=np.ravel( np.array(arr) )
    return delimiter.join(arr)

def parse_ints(str,delimiter=','):
    iarr = [];
    if( delimiter is None ):
        sarr=str.split()
    else:
        sarr = str.split(delimiter)
    for x in sarr:
        iarr.append( int(x) )
    return iarr

def parse_doublesx(line,Nwant=None):
    le=len(line)
    ret=[]
    I=0;nGot=0
    while(I<le):
        while(I<le):
            if( ('0'<= line[I] and line[I]<='9') or (line[I]=='+') or (line[I]=='-') or line[I]=='.' or line[I]=='E' or line[I]=='e'):
                break
            I+=1 ## NEVER FORGET
        if(I>=le):
            break
        J=I;
        while(J+1<le):
            if( ('0'<= line[J+1] and line[J+1]<='9') or (line[J+1]=='+') or (line[J+1]=='-') or line[J+1]=='.' or line[J+1]=='E' or line[J+1]=='e'):
                J+=1
            else:
                break
        dum=float(line[I:(J+1)]);ret.append(dum);nGot+=1
        ### print_00("parsing:"+line[I:(J+1)]+" ..."+str(ret));
        if( Nwant is not None):
            if(nGot>=Nwant):
                return ret;
        I=J+1 ## NEVER FORGET
    return ret;
def parse_intsx(line,Nwant=None):
    le=len(line)
    ret=[]
    I=0;nGot=0
    while(I<le):
        while(I<le):
            if( ('0'<= line[I] and line[I]<='9') or (line[I]=='+') or (line[I]=='-') ):
                break
            I+=1 ## NEVER FORGET
        if(I>=le):
            break
        J=I;
        while(J+1<le):
            if( ('0'<= line[J+1] and line[J+1]<='9') ):
                J+=1
            else:
                break
        dum=int(line[I:(J+1)]);ret.append(dum);nGot+=1
        ### print_00("parsing:"+line[I:(J+1)]+" ..."+str(ret));
        if( Nwant is not None):
            if(nGot>=Nwant):
                return ret;
        I=J+1 ## NEVER FORGET
    return ret;


def parse_doubles(str,delimiter=','):
    darr = [];
    sarr=str.split()
    for x in sarr:
        x=x.strip()
        if( len(x)==0 ):
            continue
        if( delimiter is not None ):
            subarr=x.split(delimiter)
            for y in subarr:
                y=y.strip()
                if( len(y)==0 ):
                    continue
                darr.append(float(y))
        else:
            darr.append(float(x))

    return darr


def aNmaxdiff(A,B,comment=None,lower_thr=None,verbose=False,iverbose=None,title=None,logfile=None,logf_append=True):
    a=np.ravel( np.array(A) )
    b=np.ravel( np.array(B) )
    if(iverbose is None):
        iverbose=( 1 if(not verbose) else 2 )
    fdA=[sys.stdout]
    fdOU=( None if(logfile is None) else open( logfile,('a' if(logf_append) else 'w') ) );fdA.append(fdOU)
    if(iverbose>2):
        for fd in fdA:
            lea=len(a);leb=len(b)
            print_00( "#aNmaxdiff:%sLH:%d:"%( ("" if(comment is None) else comment+':'),lea) +str( np.array(A).dtype )+str(np.shape( np.array(A) )),file=fd)
            print_00( "#aNmaxdiff:%sRH:%d:"%( ("" if(comment is None) else comment+':'),leb) +str( np.array(B).dtype )+str(np.shape( np.array(B) )),file=fd)

    ret=abs(a[0]-b[0]);at=0;vals=[a[0],b[0]];
    N=len(a)
    for I in range(1,N):
        dum=abs(a[I]-b[I])
        if( dum>ret ):
            ret=dum; at=I; vals=[a[I],b[I]]
    prtout=(iverbose>1)
    if(lower_thr is not None):
        if(ret> 0.1*lower_thr):
            prtout=(iverbose>0)
    if(prtout):
        for fd in fdA:
            print_00("#aNmaxdiff:%e "%(ret)+("" if(comment is None) else comment)\
                +"@"+ixj_to_IandJ(at,np.shape(A),toString=True)+" values:"+str(vals),file=fd)
    if(fdOU is not None):
        fdOU.close()
    return ret

def print_z3tensor(fpath,Append,T,description=""):
    Ndim=np.shape(T);rank=len(Ndim)
    if( rank!=3 and rank!=4 ):
        print("#print_z3tensor:!W wrong dim:"+str(Ndim));return
    nkp=(1 if(rank==3) else Ndim[0])
    strdtm=str(datetime.datetime.now())
    fd1=open(fpath,("a" if(Append) else "w"))
    print("###%s \t\t"%(description)+strdtm,file=fd1)
    for kp in range(nkp):
        M=(T if(rank==3) else T[kp])
        nd=np.shape(M)
        for i in range(nd[0]):
            if(rank==4):
                print("#%d,%d"%(kp,i),file=fd1)
            else:
                print("#%04d"%(i),file=fd1)
            for j in range(nd[1]):
                for k in range(nd[2]):
                    print("%11.5f %11.5f    "%(M[i][j][k].real, M[i][j][k].imag),end="",file=fd1)
                print("\n",end="",file=fd1)
            print("\n\n",end="",file=fd1)
    fd1.close()

# nmrdff: 2n+1 point numerical differentiation  n=1,2,3
# @input: order : 3/5/7
# @input: arr   : [order][ dim ] 
#                 eg. [ f(x-2h), f(x-h), NONE, f(x+h), f(x+2h)]
#                   each can be an array[dim]  (if dim>0)
#                               of scalar      (if dim==0)
# @input: dim   : linear dimension of the input arrays (see above)
# @
def nmrdff( order,arr,dim,dx,dict=None):
    if(dim==0):
        dtype=np.array( [arr[0]] ).dtype
    else:
        dtype=np.array( arr[0]).dtype
    ##
    ## note: arr might contain None
    ## e.g.) [ F(x-h), None, F(x+h) ]
    ## so for dim>0 np.array( arr ).dtype does not work
    ##
    assert( dtype==float or dtype==complex),"dtype="+str(dtype)
    assert( order==3 or order==5 or order==7),"order"
    hf=order//2  # 1,2,3
    ## print_00("hf:%d"%(hf));print_00(dtype)
    dxINV=1.0/dx

    df3p=0.0
    if(dim>0):
        df3p=np.zeros([dim],dtype=dtype)
    df3p=-arr[ -1 + hf]*0.50
    df3p+=arr[  1 + hf]*0.50
    df3p= df3p*dxINV
    if( dict is not None ):
        dict.update({"3p":df3p})
##        if( "3p" in dict ):
##            dict["3p"]=df3p

    if( order > 3 ):
        df5p=0.0
        if(dim>0):
            df5p=np.zeros([dim],dtype=dtype)
            df5p= arr[   -2 + hf].copy()
        else:
            df5p= arr[   -2 + hf]
        df5p-=arr[   -1 + hf]*8.0
        df5p+=arr[    1 + hf]*8.0
        df5p-=arr[    2 + hf]
        df5p=df5p*(dxINV/12.0)
        if( dict is not None ):
            dict.update({"5p":df5p})
##            if( "5p" in dict ):
##                dict["5p"]=df5p
 
    inv3= 1.0/3.0
    if( order > 5 ):
        df7p=0.0
        if(dim>0):
            df7p=np.zeros([dim],dtype=dtype)
        df7p=-arr[   -3 + hf]*inv3;
        df7p+=arr[   -2 + hf]*3.0 ;
        df7p-=arr[   -1 + hf]*15.0;
        df7p+=arr[    1 + hf]*15.0;
        df7p-=arr[    2 + hf]*3.0 ;
        df7p+=arr[    3 + hf]*inv3;
        df7p=df7p*(dxINV/20.0);
        
        if( dict is not None ):
            dict.update({"7p":df7p})
##            if( "7p" in dict ):
##                dict["7p"]=df7p
    if( order==7 ):
        return df7p
    elif( order==5):
        return df5p
    else:
        return df3p

def aNtofile(ope,fpath,dbuf=None):
    dtype=np.array(dbuf).dtype
    src1D=np.ravel(dbuf);Ld=len(src1D)
    if( ope == 'S' ):
        print_00("#aNtofile:writing on"+fpath)
        fd=futils.fopen(fpath,"w"); ## OK
        if(dtype==complex):
            print_00("C %d"%(Ld),file=fd)
            for I in range(Ld):
                print_00("%18.10f %18.10f     "%(src1D[I].real,src1D[I].imag),end=("\n" if((I+1)%10==0) else ""),file=fd)
        else:
            print_00("F %d"%(Ld),file=fd)
            for I in range(Ld):
                print_00("%18.10f   "%(src1D[I]),end=("\n" if((I+1)%20==0) else ""),file=fd)
        futils.fclose(fd);return None
    elif( ope=='L' or ope=='C'):
        print_00("#aNtofile:reading from"+fpath)
        fd=futils.fopen(fpath,"r");## OK
        nl=0;N=0;buf=None
        for line in fd:
            nl+=1
            if(nl==1):
                a=line.split();dtype=float
                if(a[0].strip()=="C"):
                    dtype=complex
                elif(a[0].strip()=="F"):
                    dtype=float
                Ld=int(a[1]);
                buf=np.zeros([Ld],dtype=dtype);### print_00("#dim=%d %s /%s file:"%(len(buf),a[1],line)+fpath)
                continue
            else:
                a=line.split();n=len(a);
                if( dtype == complex ):
                    for i in range(n/2):
                        buf[N]=np.complex128( float(a[2*i]) + 1j*float(a[2*i+1])); N+=1
                else:
                    for i in range(n):
                        ## print_00("buf%d/%d a%d/%d"%(N,len(buf),i,len(a)))
                        buf[N]=float(a[i]); N+=1
        futils.fclose(fd)
        assert (N==Ld),"read=%d/%d"%(N,Ld)
        if( dbuf is not None ):
            dev=abs( src1D[0] - buf[0] );at=0
            for I in range(1,Ld):
                dum=abs( src1D[I]-buf[I] )
                if(dum>dev):
                    dev=dum;at=I
            print_00("#maxdiff=%e @[%d]"%(dev,at),end="");print_00(buf[at],end=" / ");print_00(src1D[at])

        return buf
    else:
        assert False,"unknown operation:"+ope

def calc_eorbs(Cofs,FockMatr):
    NdimC=np.shape(Cofs)
    iscomplex=False
    if(  (np.array(Cofs).dtype == complex) or (np.array(FockMatr).dtype == complex) ):
        iscomplex=True
    ret=None
    if( len(NdimC)==3 ):
        print_00("#calc_eorbs:FockMatr:",end="");print_00(np.shape(FockMatr)); #[nKp][nAO][nAO]
        print_00("#calc_eorbs:Cofs:",end="");print_00(np.shape(Cofs));#[nKp][nAO][nMO]
        nKP=NdimC[0]; nAO=NdimC[1]; nMO=NdimC[2]; ret=np.zeros([nKP,nMO])
        for KP in range(nKP):
            cofK=Cofs[KP];## print_00(np.shape(cofK))
            if( iscomplex ):
                for I in range(nMO):
                    ret[KP][I]=np.vdot( cofK[:,I], np.matmul( FockMatr[KP], cofK[:,I])).real
            else:
                for I in range(nMO):
                    ret[KP][I]=np.vdot( cofK[:,I], np.matmul( FockMatr[KP], cofK[:,I]))

    else:
        nKP=1; nAO=NdimC[0]; nMO=NdimC[1]; ret=np.zeros([nMO])
        if( iscomplex ):
            for I in range(nMO):
                ret[I]=np.vdot( Cofs[:,I], np.matmul( FockMatr, Cofs[:,I])).real
        else:
            for I in range(nMO):
                ret[I]=np.vdot( Cofs[:,I], np.matmul( FockMatr, Cofs[:,I]))
    return ret;

def hdiag(matr,check_soln=False,**kwargs):
    ##
    ## diagonalizes Hermitian matrix via most efficient method ...
    ##
    Ld=len(matr)
    if(Ld>=1024):
        E,U = np.linalg.eig(matr)
        info=0 ## otherwise throws exception
    else:
        E,U,info = scipy.linalg.lapack.zheev(matr)
    if( check_soln ):
        title="hdiag:"
        if( "title" in kwargs ):
            title=kwargs["title"]
        devmx= check_eigvecs(title,matr,U,E);
        if( "tol" in kwargs ):
            assert (devmx<tol),"hdiag_"+title+" devmx:%e"%(devmx)

    return E,U,info

def dic_to_string(tgt,delimiter=',',bra='{',ket='}'):
    ret="";nth=0
    for ky in tgt:
        val=tgt[ky];strval=None
        if( isinstance(val,list) or isinstance(val,np.ndarray)):
            strval=dN_to_str(val)
        else:
            strval=str(val)
        ret+=(delimiter if(nth>0) else '')+ ky+delimiter+strval; nth=nth+1
    return bra+ret+ket;


def parse_xyzstring(str,unit_BorA):
    assert (unit_BorA=='B' or unit_BorA=='A'),""
    fac=1.0
    if( unit_BorA=='B' ):
        fac=1.0/PhysicalConstants.BOHRinANGS();

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
        fac=PhysicalConstants.BOHRinANGS();
    str="";
    Nat=len(Sy)
    for Iat in range(Nat):
        str+="%s %f %f %f\n"%(Sy[Iat],R[Iat][0]*fac,R[Iat][1]*fac,R[Iat][2]*fac);
    return str;

def update_dict(fnc,dic1,Dic,key,val,append=False,iverbose=1,depth=0):
    from mpi4py import MPI

    Key=fnc+'.'+key
    update_dict_(dic1,Dic,Key,val,append=append,iverbose=iverbose,depth=depth)
    prtout=(iverbose>1)
    if( Dic[Key]['count'] == 1 ):
        prtout=(iverbose>0)

    if( prtout):
        comm=MPI.COMM_WORLD
        rank=comm.Get_rank()
        
        string='  '*depth + "#%03d:%s.%s:%14.4f"%(rank,fnc,key,val)
        fdF=open('walltime_%03d.log'%(rank),'a')
        av,stdv,count=avg_dict_(Dic,Key)
        string+="        \t\t\t  %14.4f %12.2e %3d"%(av,stdv,count)
        fdA=[ fdF ]
        if( rank == 0 ):
            fdA.append(sys.stdout)
        for fd in fdA:
            print(string,file=fd)
        fdF.close()


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
def avg_dict_(dic,key):
    if( key not in dic ):
        return 0.0,0.0,0
    ### print_00("#avg_dict_:"+key+str(dic[key]))
    if( dic[key]['count']==0 ):
        return 0.0,0.0,0
    else:
        av=dic[key]['sum']/dic[key]['count']
        stdv=np.sqrt( abs( dic[key]['sqrsum']/dic[key]['count'] - av**2 ))
    return av,stdv,dic[key]['count']

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

def printout_dict(fnc,dic1,Dic,val,depth=0,force=False,fnprtout=None):
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    rank=comm.Get_rank()

    update_dict_(dic1,Dic,fnc,val,iverbose=0,depth=depth)
    nth=Dic[fnc]['count']
    if( fnprtout is not None ):
        prtout=(force or fnprtout(nth) )
    else:
        prtout=(force or (nth<4 or nth==10 or nth==100 or (nth>100 and nth%200==0)))
    if( not prtout ):
        return 0
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
    fdA=[ fdF ]
    if( rank == 0 ):
        fdA.append(sys.stdout)
    for fd in fdA:
        print(string,file=fd)
    fdF.close()
    return nth

def readwritezN(RorW,name,zbuf,Ith=None,Ndim_zbuf=None,format="%s_%04d.dat",compTO=None,description="",logfile=None,iverbose=1):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if( MPIrank > 0 ):
        write_once("utils.readwritezN:mpi",
                "utils.readwritezN invoked by thread %02d"%(MPIrank))
    
    if(Ith is None):
        if( RorW=='R'):
            Ith=readwritezN_static_.Countup(name,inc=False)
        elif( RorW=='W'):
            Ith=readwritezN_static_.Countup(name,inc=True)
    fpath=format%(name,Ith)

    if( iverbose>(0 if(RorW=='W') else 1) ):
        print_00("#readwritezN:"+RorW+" Ith:%d FILE:"%(Ith)+fpath)

    if( compTO is None ):
        compTO=Ith-1

    retv=None
    if( RorW=='W' ):
        maxdiff=None;indexAT=None;sqrdiffsum=None;strdiff=None
        readwritezN_writeone_(fpath,zbuf,header='#'+description, iverbose=iverbose)
        if( compTO >=0 ):
            maxdiff,indexAT,sqrdiffsum,strdiff = readwritezN_compto(zbuf,Ith, name,compTO,format=format,logfile=logfile,iverbose=iverbose)
        return fpath,maxdiff,indexAT,sqrdiffsum,strdiff
    elif(RorW=='R'):
        sbuf=[]
        ret=readwritezN_readone_(fpath,sbuf,Ndim=Ndim_zbuf, iverbose=iverbose)
        if( compTO >=0 ):
            readwritezN_compto(ret,Ith, name,compTO,format=format,logfile=logfile,iverbose=iverbose)
        return ret,sbuf
    else:
        assert False,""+RorW
        return None


def to_complexarray(src, do_clone=False):
    if( (np.array(src).dtype == complex) or np.array(src).dtype == np.complex128 ):
        if( do_clone ):
            return arrayclone(src)
        return src
    else:
#converting to complex from:float64
###        print_00("#converting to complex from:",end="");print_00(np.array(src).dtype)
        Ndim=np.shape( np.array(src) ) 
        srcv=np.ravel( np.array(src) )
        Ld=len(srcv)
        ret=np.zeros( [Ld], dtype=np.complex128 )
        for I in range(Ld):
            ret[I]=srcv[I]
        return np.reshape( ret, Ndim)

def print_TDeorbs(spinrestriction,TdMO,rttddft,pbc,RefMO,
                  FockMat,MO_Occ,job="temp",Append=True,step=-1,tm_au=-1,get_eorbs=None,get_pop=None,
                  md=None,popsum_dev=None,title=""):
    assert spinrestriction!='O',"please double check dimensions: nmult_MO=1 nmult_Occ=2 and nmult_Fock=1 ?"
    nmult_MO=(2 if(spinrestriction=='U') else 1)
    nmult_Fock=nmult_MO
    filenamelist=[]
    FermiLv_au=(0.0 if(not pbc) else rttddft.get_fermi())
    for sp in range(nmult_MO):
        xtn="" if(nmult_MO==1) else ("_upSpin" if(sp==0)else "_dnSpin")
        mo= TdMO if(nmult_MO==1) else TdMO[sp];          #[kp][nAO][nMO]
        rfmo=RefMO if(nmult_MO==1) else RefMO[sp];       #[kp][nAO][nMO]
        fock=FockMat if(nmult_Fock==1) else FockMat[sp]; #[kp][nAO][nAO]
        occ= MO_Occ if(nmult_MO==1) else MO_Occ[sp];     #[kp][nMO]
        #print_00("TdMO:",end="");print_00(np.shape(mo))
        #print_00("rfmo:",end="");print_00(np.shape(rfmo))
        #print_00("fock:",end="");print_00(np.shape(fock))
        #print_00("occ:",end="");print_00(np.shape(occ))
        get_eorbs1=None;get_pop1=None
        if( get_eorbs is not None):
            if( nmult_MO == 1 ):
                get_eorbs1=get_eorbs;
            else:
                get_eorbs1=[];
        if( get_pop is not None):
            if( nmult_MO == 1 ):
                get_pop1=get_pop;
            else:
                get_pop1=[];
        fermiLv1_au=( 0.0 if(not pbc) else ( FermiLv_au if(spinrestriction!='U') else FermiLv_au[sp] ) )
        flist1 = print_tdeorbs1(mo,rttddft,pbc,rfmo,fock,occ,job=job+xtn,Append=Append,step=step,tm_au=tm_au,\
                                get_eorbs=get_eorbs1,get_pop=get_pop1,md=md,popsum_dev=popsum_dev,title=title, 
                                FermiLv_au=fermiLv1_au)
        if( get_eorbs is not None):
            if( nmult_MO != 1 ):
                get_eorbs.append( get_eorbs1 );
        if( get_pop is not None):
            if( nmult_MO != 1 ):
                get_pop.append( get_pop1 );

        for fnme in flist1:
            filenamelist.append(fnme)
    return filenamelist;

## jobname( name=params["name"], branch=params["branch"], basis=params["basis"], xc=params["xc"],
##          timestep_as=params["timestep_as"], strField=None )             
    
def jobname(**kwargs):
    return format_jobname(kwargs);

def format_jobname(params,pbc=None):
    name=getv_safe(params,"name",default="")

    name=getv_safe(params,"name",default="")
    branch_or_ver=( ("" if("ver" not in params) else params["ver"]) if("branch" not in params) else params["branch"])
    basis=getv_safe(params,"basis",default="")
    if( "exp_to_discard" in params):
        if( params["exp_to_discard"] is not None):
            basis +="e%4.2f"%( float(params["exp_to_discard"]) )
    xc   =getv_safe(params,"xc",default="").replace(",","-")
    strKpoints=""
    if( pbc is None or (not pbc) ):
        for kw in params:
            for ref in ["nKpoints","nkpoints"]:
                if(kw==ref):
                    nKpoints=params[kw];
                    if( nKpoints is not None ):
                        if(isinstance(nKpoints,str)):
                            sA=nKpoints.split(',');NsA=len(sA)
                            nKpoints=[ int(sA[j]) for j in range(NsA) ]
                        assert isinstance(nKpoints[0],int),""; assert len(nKpoints)==3,""
                        strKpoints="_Kp%d%d%d"%(nKpoints[0],nKpoints[1],nKpoints[2])
    DFT=""
    if( "spinrestriction" in params ):
        if( params["spinrestriction"] is not None):
            DFT=("" if( params["spinrestriction"] =='R' ) else ( "UKS" if (params["spinrestriction"] =='U') 
                 else ("ROKS" if(params["spinrestriction"] =='O') else None) ) ) 
            assert DFT is not None,"wrong spinrestriction:"+ params["spinrestriction"]
    str_dt=("" if("timestep_as" not in params) else "_dt"+("%5.2f"%(params["timestep_as"])).strip() )
    strField=("" if("strField" not in params) else ("" if( params["strField"] is None ) else "_"+params["strField"] ) )
    return name + branch_or_ver + "_" + DFT + basis + xc + strKpoints + str_dt + strField

def isScalarNumber(arg):
    return (isinstance(arg,int) or isinstance(arg,float) or isinstance(arg,complex) or isinstance(arg,np.int64)\
            or isinstance(arg,np.float64) or isinstance(arg,np.complex128) ) 

def Schmidt_orth(tgt,TINY=1.0e-8):
    Ndim=np.shape(tgt)
    Nv=Ndim[1]
    Ret=[];n=0
    for I in range(Nv):
        vdum=tgt[:,I].copy();
        for j in range(n):
            cof=np.vdot( Ret[j],vdum )
            vdum = vdum - cof*Ret[j]
        le=np.sqrt( np.vdot(vdum, vdum))
        if( le < TINY ):
            print("linearly dpd:%03d:%e"%(I,le));continue
        else:
            Ret.append( vdum/le );n+=1
    return Ret

def check_matrixrank(tgt,TINY=1.0e-8):
    orth=Schmidt_orth(tgt,TINY=TINY)
    le=len(orth)
    return le



def atomicsymbol_to_atomicnumber(S):
    atomicsymbols_=["H",                                                                                 "He",\
                    "Li","Be",                                                   "B", "C", "N", "O", "F","Ne",\
                    "Na","Mg",                                                  "Al","Si", "P", "S","Cl","Ar",\
                     "K","Ca","Sc","Ti", "V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",\
	                "Rb","Sr", "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pb","Ag","Cd","In","Sn","Sb","Te", "I","Xe"]

    for I in range( len(atomicsymbols_) ):
        if( atomicsymbols_[I]==S ):
            return I+1
    assert False,"wrong symbol:"+S

def atomicnumber_to_atomicsymbol(Z):
    atomicsymbols_=["H",                                                                                 "He",\
                    "Li","Be",                                                   "B", "C", "N", "O", "F","Ne",\
                    "Na","Mg",                                                  "Al","Si", "P", "S","Cl","Ar",\
                     "K","Ca","Sc","Ti", "V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",\
	                "Rb","Sr", "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pb","Ag","Cd","In","Sn","Sb","Te", "I","Xe"]
    le=len(atomicsymbols_)
    assert ((Z-1)<le),"Z=%d"%(Z)
    return atomicsymbols_[Z-1];


def write_filex(fpath,**kwargs):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if( MPIrank > 0 ):
        write_once("utils.write_file:mpi",
                "utils.write_file invoked by thread %02d"%(MPIrank))
    fd=futils.fopen(fpath,"w"); ## OK
    for ky in kwargs:
        x=kwargs[ky]
        if( isinstance(x,np.ndarray) ):
            if(len(x)==0):
                print(ky+":I",end="\t",file=fd);
            elif(x.dtype==float):
                print(ky+":D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==np.float64):
                print(ky+":D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==complex):
                print(ky+":Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==np.complex128):
                print(ky+":Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==int):
                print(ky+":I",end="\t",file=fd);print(a1toa(x),file=fd)
        elif( isinstance(x,list)):
            if(len(x)==0):
                print(ky+":I",end="\t",file=fd);
            elif(isinstance(x[0],float) ):
                print(ky+":D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],np.float64)):
                print(ky+":D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],complex)):
                print(ky+":Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],np.complex128)):
                print(ky+":Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],int)):
                print(ky+":I",end="\t",file=fd);print(a1toa(x),file=fd)
        elif(isinstance(x,float) ):
            print(ky+":d",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,np.float64)):
            print(ky+":d",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,complex)):
            print(ky+":z",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,np.complex128)):
            print(ky+":z",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,int)):
            print(ky+":i",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,str)):
            print(ky+":s",end="\t",file=fd);print(str(x),file=fd)
        else:
            print(ky+":unknown type:"+str(type(x)),end="\t",file=fd);print(str(x),file=fd)
    futils.fclose(fd)

def readwrite_xbuf(RorW,fnme,data=None):
    if(RorW=='R' or RorW=='r'):
        ret=[]
        fd=futils.fopen(fnme,"r");## OK
        dtype=None
        for line in fd:
            if(dtype is None):
                line=line.strip()
                if(line=="d"):
                    dtype=np.float64
                elif(line=='z'):
                    dtype=np.complex128
                elif(line=='i'):
                    dtype=int
                else:
                    assert False,""+line
                continue
            if( dtype == int ):
                arr=parse_ints(line,delimiter=None)
                n=len(arr)
                for i in range(n):
                    ret.append(arr[i])
            else:
                arr=parse_doubles(line,delimiter=None)
                if( dtype == np.complex128 ):
                    n=len(arr)//2
                    for i in range(n):
                        ret.append( np.complex128( arr[2*i]+1j*arr[2*i+1] ))
                else:
                    n=len(arr)
                    for i in range(n):
                        ret.append(arr[i])
        futils.fclose(fd)
        return np.array(ret)
    else:
        assert (data is not None),"fnme:"+fnme
        data_1d=np.ravel(data)
        fd=futils.fopen(fnme,"w") ## OK
        dtype=np.array(data_1d).dtype
        if( dtype == int ):
            print("i",file=fd);
        elif(dtype==float or dtype==np.float64):
            print("d",file=fd);
        elif(dtype==complex or dtype==np.complex128):
            print("z",file=fd);
        else:
            assert False,""+dtype
        ncol=8
        Ld=len(data_1d)
        sbuf="";
        for i in range(Ld):
            if( dtype==int):
                sbuf+="%d "%(data_1d[i])
            elif( dtype==float or dtype==np.float64):
                sbuf+="%18.10e "%(data_1d[i])
            elif( dtype==complex or dtype==np.complex128):
                sbuf+="%18.10e %18.10e "%(data_1d[i].real,data_1d[i].imag)

            if( (i+1)%ncol==0 ):
                print(sbuf,file=fd);sbuf=""
        if(sbuf!=""):
            print(sbuf,file=fd);sbuf=""
        futils.fclose(fd)

def print_a2maxdiff(A,B,fpath,Append=False,description=""):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return aNmaxdiff(A,B)

    fd=open(fpath,("a" if(Append) else "w"))
    Ndim=np.shape(A);assert i1eqb(Ndim,np.shape(B)),"";
    assert len(Ndim)==2,""

    diff=aNmaxdiff(A,B)
    if(Append):
        print("\n\n",file=fd);
    print("##%s maxdiff:%14.6e"%(description,diff),file=fd);
    for I in range(Ndim[0]):
        for J in range(Ndim[1]):
            print("%04d %04d  %14.6e      %12.6f %12.6f      %12.6f %12.6f"%(\
                    I,J,abs(A[I][J]-B[I][J]),A[I][J].real,A[I][J].imag,B[I][J].real,B[I][J].imag),file=fd)
    fd.close()
    return diff

def print_Hmatrices(title,A,B,Nref=5,fpath=None,Append=False):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return

    Ndim=np.shape(A)
    assert i1eqb(Ndim,np.shape(B)),"print_Hmatrices:"+str(Ndim)+"/"+str( np.shape(B) )
    if( len(Ndim)==2 ):
        print_Hmatrices_1(title,A,B,Nref,fpath)
        return
    else:
        if(len(Ndim)==3):
            for I in range(Ndim[0]):
                print_Hmatrices_1(title+"_K%03d"%(I),A[I],B[I],Nref,fpath,Append=(Append if(I==0) else True))
        else:
            assert False,""+str(Ndim)


def write_file(fpath,*args):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if( MPIrank > 0 ):
        write_once("utils.write_file:mpi",
                "utils.write_file invoked by thread %02d"%(MPIrank))
    fd=futils.fopen(fpath,"w"); ## OK
    for x in args:
        if( isinstance(x,np.ndarray) ):
            if(len(x)==0):
                print("I",end="\t",file=fd);
            elif(x.dtype==float):
                print("D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==np.float64):
                print("D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==complex):
                print("Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==np.complex128):
                print("Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(x.dtype==int):
                print("I",end="\t",file=fd);print(a1toa(x),file=fd)
        elif( isinstance(x,list)):
            if(len(x)==0):
                print("I",end="\t",file=fd);
            elif(isinstance(x[0],float) ):
                print("D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],np.float64)):
                print("D",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],complex)):
                print("Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],np.complex128)):
                print("Z",end="\t",file=fd);print(a1toa(x),file=fd)
            elif(isinstance(x[0],int)):
                print("I",end="\t",file=fd);print(a1toa(x),file=fd)
        elif(isinstance(x,float) ):
            print("d",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,np.float64)):
            print("d",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,complex)):
            print("z",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,np.complex128)):
            print("z",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,int)):
            print("i",end="\t",file=fd);print(str(x),file=fd)
        elif(isinstance(x,str)):
            print("s",end="\t",file=fd);print(str(x),file=fd)
        else:
            print("unknown type:"+str(type(x)),end="\t",file=fd);print(str(x),file=fd)
    futils.fclose(fd)

def list_to_a1(src):
    if( isinstance(src,list) or isinstance(src,np.ndarray) ):
        ret=[]
        for x in src:
            if( isinstance(x,list) or isinstance(x,np.ndarray)):
                arr=list_to_a1(x)
                for y in arr:
                    ret.append(y)
            else:
                ### print_00("x is not list or ndarray",end="");print_00(x)
                ret.append(x)
        return ret;
    else:
        print_00("!W Neither list or ndarray",end="");print_00(src)
        return src

def gamma_hfint(arg):
    if(arg%2==0): ## Gamma(n)=(n-1)! 
        p=arg/2;
        return int_factorial(p-1);
    else:
        return hfodd_factorial(arg)

def modify_filename(org,pbc,nmult,spin,kpt):
    if(not pbc and (nmult==1) ):
        return org
    head=None;tail=None
    for xtn in [".dat",".log",".out"]:
        if(org.endswith(xtn)):
            head=org.replace(xtn,"");tail=xtn; break
    if(head is None):
        head=org;tail=""
    if( not pbc ):
        return head+("_upSpin" if(spin==0) else "_dnSpin")+tail
    if( nmult==1 ):
        return head+ "_k%03d"%(kpt) + tail
    return head+("_upSpin" if(spin==0) else "_dnSpin")\
               + "_k%03d"%(kpt) + tail
def deviation_from_unitmatrix(A):
    N=len(A)
    devsum=0.0; dagdev=-1.0; ofddev=-1.0
    for I in range(N):
        for J in range(N):
            if(I==J):
                dev=abs( A[I][J]-1.0); devsum+=dev**2; dagdev=max(dagdev,dev)
            else:
                dev=abs( A[I][J] );    devsum+=dev**2; ofddev=max(ofddev, dev)
    return devsum,dagdev,ofddev

def comp_zbufs(Lhs,Rhs,fpath,Append=False,description=""):
    fd=open(fpath,("a" if(Append) else "w"))
    print("## "+description,file=fd)
    lhs=np.ravel(Lhs);rhs=np.ravel(Rhs)
    leL=len(lhs);leR=len(rhs)
    le=min(leL,leR)
    sL="< ";sR="> ";sD="# ";n=0
    for I in range(le):
        sL+="%15.9f %15.9f    "%(lhs[I].real,lhs[I].imag)
        sR+="%15.9f %15.9f    "%(rhs[I].real,rhs[I].imag)
        sD+=(' '*15)+" %15.6e    "%(abs(rhs[I]-lhs[I]));  n+=1
        if(n%6==0):
            print(sL,file=fd);print(sR,file=fd);print(sD,file=fd);
            sL="";sR="";sD="";n=0;
    if(n>0):
        print(sL,file=fd);print(sR,file=fd);print(sD,file=fd);
    fd.close()

def print_aNmatr(fpath,A,Append=False,comment=None):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return

    Ndim=np.shape(A)
    fd=open(fpath,('a' if(Append) else 'w'))
    if(comment is not None):
        print("###"+comment,file=fd)     ### you come here iff MPIrank==0 
    if(len(Ndim)==3):
        for I in range(Ndim[0]):
            print("#%04d:"%(I),file=fd)
            for J in range(Ndim[1]):
                string=""
                for K in range(Ndim[2]):
                    string+="%11.6f %11.6f      "%(A[I][J][K].real,A[I][J][K].imag)
                print(string,file=fd);
            print("\n\n",file=fd)
        fd.close()
    elif( len(Ndim)==2 ):
        for J in range(Ndim[0]):
            string=""
            for K in range(Ndim[1]):
                string+="%11.6f %11.6f      "%(A[J][K].real,A[J][K].imag)
            print(string,file=fd);
        print("\n\n",file=fd)
    elif( len(Ndim)==4 ):
        IxJ=0
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                print("#%04d:(%03d,%03d)"%(IxJ,I,J),file=fd)
                string=""
                for K in range(Ndim[2]):
                    for L in range(Ndim[3]):
                        string+="%11.6f %11.6f      "%(A[I][J][K][L].real,A[I][J][K][L].imag)
                    print(string,file=fd);
                print("\n\n",file=fd)
    else:
        assert False,""+str(Ndim)
    fd.close()

def prtaNx2(title,A,B):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return

    print("#prtaNx2:"+title+" Ndim:"+str(np.shape(A))+" "+str(np.shape(B)) )
    a=np.ravel(A);
    if( not ( isinstance(a[0],np.float64) or isinstance(a[0],float) or 
              isinstance(a[0],np.complex128) or isinstance(a[0],complex) ) ):
        buf=[]
        for x in a:
            for y in np.ravel(x):
                buf.append(y); 
                if( len(buf)%20==1 ):
                    print("append:",end="");print(y)
        print("a:",end="");print(np.shape(buf))
        a=buf
    b=np.ravel(B)
    print(np.shape(a));print("a:",end="");print(a)
    print(np.shape(b));print("b:",end="");print(b)
    la=len(a)
    lb=len(b)
    N=(lb if(lb>la) else la)
    line=""
    for i in range(N):
        line="#prtaNx2:"+title+":%d:"%(i)
        if(i<la):
            line+=str(a[i])+" "
        else:
            line+="\t";
        if(i<lb):
            line+=str(b[i])+" "
        else:
            line+="\t";
        if(i<la and i<lb):
            print(abs(a[i]-b[i]))
            line+=" diff:%e"%(abs(a[i]-b[i]))
        print(line)
def read_zbuf(fpath,verbose=False):
    fd=futils.fopen(fpath,"r")## OK
    block=0
    Ndim=None;Ld=None
    ret=[];zbuf=None
    for line in fd:
        line=line.strip()
        if(line.startswith('#')):
            if(zbuf is not None):
                le=len(zbuf)
                print_00("#block:%d read=%d / %d "%(block,le,Ld))
                
                if(le==Ld):
                    ret.append(np.reshape(zbuf,Ndim))
                else:
                    ##assert (le==Ld),("Wrong number of data:read=%d / %d Ndim:"%(le,Ld) + str(Ndim))
                    ret.append(zbuf); print_00("#block:%d WRONG DIM:%d / %d "%(block,le,Ld))
            zbuf=[]
            line=line.replace("#","");
            sarr=line.split();
            Ndim=[];Ld=1;block+=1
            for x in sarr:
                Ndim.append( int(x) );Ld*=int(x)
            print_00("#read_zbuf:",end="");print_00(Ndim)
            if(verbose):
                print_00("#reading Ndim:",end="");print_00(Ndim)
        else:
            le1=len(zbuf);
            sarr=line.split();n=len(sarr);n=n//2
            for j in range(n):
                zbuf.append( float( sarr[2*j] ) + 1j*float( sarr[2*j+1] ) )
            if(verbose):
                print_00("#reading in:%d %d %s... %s %f+j%f ... "%( n,len(zbuf),sarr[0],sarr[2*n-1],zbuf[le1].real, zbuf[le1].imag))
    futils.fclose(fd)

    if(zbuf is not None):
        le=len(zbuf)
        print_00("#block:%d read=%d / %d "%(block,le,Ld))
        
        assert (le==Ld),"Wrong number of data:read=%d / %d Ndim:"%(le,Ld) + str(Ndim)
        ret.append(np.reshape(zbuf,Ndim))
    
    return ret

def getv_safe(Dict,key,default=None):
    if( isinstance(key,list) ):
        for ky in key:
            if( ky in Dict ):
                return Dict[ky]
    else:
        if( key in Dict ):
            return Dict[key]
    return default

def print_Hmatrices_1(title,A,B,Nref=5,fpath=None,Append=False):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return

    TINY=1.0e-8
    fdOU=(None if(fpath is None) else open(fpath,("a" if(Append) else "w")) )
    fd=( fdOU if(fdOU is not None) else sys.stdout)

    maxdiff=aNmaxdiff(A,B)

    max_Im=-1;where=[]
    Ndim=np.shape(A)
    assert len(Ndim)==2,"Ndim="+str(Ndim)+" len(Ndim)/=2"
    Nrow=Ndim[0];Ncol=Ndim[1]
    for I in range(Nrow):
        for J in range(Ncol):
            if(max_Im<abs(A[I][J].imag)):
                max_Im=abs(A[I][J].imag);where=[0,I,J]
            if(max_Im<abs(B[I][J].imag)):
                max_Im=abs(B[I][J].imag);where=[1,I,J]
    if( Append ):
        print("\n\n",file=fd)
    print("#print_Hmatrices:%s:maxdiff=%e max_Im=%e [%d][%d] %f+j%f / %f+j%f"%(
        title, maxdiff, max_Im, where[1], where[2],
        A[ where[1] ][ where[2] ].real, A[ where[1] ][ where[2] ].imag, 
        B[ where[1] ][ where[2] ].real, B[ where[1] ][ where[2] ].imag),file=fd)
    isReal=( max_Im<TINY )
    Nc=min(Ncol,Nref)
    Nr=min(Nrow,Nref);N=min(Nc,Nr)
    if(isReal):
        for I in range(N):
            string ="%10.5f"%(A[I][0]);
            for J in range(1,I+1):
                string+=" %10.5f"%(A[I][J])
            string+=((N-I-1)*(10+1))*' ';
            string+=' \t '

            string+="%10.5f"%(B[I][0]);
            for J in range(1,I+1):
                string+=" %10.5f"%(B[I][J])
            string+=((N-I-1)*(10+1))*' ';
            string+=' \t\t '

            string+="%7.2e"%( abs(A[I][0]-B[I][0]) );
            for J in range(1,I+1):
                string+=" %7.2e"%( abs(A[I][J]-B[I][J]) )
            if(fdOU is None ):
                print('#print_Hmatrices:'+title+':'+string)
            else:
                print(string,file=fdOU)
            
    else:
        for I in range(N):
            string ="%10.5f %10.5f"%(A[I][0].real,A[I][0].imag);
            for J in range(1,I+1):
                string+=" %10.5f %10.5f"%(A[I][J].real,A[I][J].imag)
            string+=((N-I-1)*((10+1)*2))*' ';
            string+=' \t '

            string+="%10.5f %10.5f"%(B[I][0].real,B[I][0].imag);
            for J in range(1,I+1):
                string+=" %10.5f %10.5f"%(B[I][J].real,B[I][J].imag)
            string+=((N-I-1)*((10+1)*2))*' ';

            string+="%8.2e"%( abs(A[I][0]-B[I][0]) );
            for J in range(1,I+1):
                string+=" %8.2e"%( abs(A[I][J]-B[I][J]) )
            if(fdOU is None ):
                print('#print_Hmatrices:'+title+':'+string)
            else:
                print(string,file=fdOU)
    if(fdOU is not None):
        fdOU.close()

def a1toa(arr):
    ret=""
    for item in arr:
        ret=ret+str(item)+" "
    return ret

def ixj_to_IandJ(ixj,Ndim,toString=False):
    rank=len(Ndim)
    ret=np.zeros([rank],dtype=int)
    J=rank;n=ixj
    while(J>0):
        J-=1;ret[J]=n%Ndim[J];n=n//Ndim[J];
    if( toString ):
        return i1toa(ret)
    else:
        return ret

def IxJ_to_IandJ(IxJ,Ndim):
    le=len(Ndim);
    ret=[ -1 for k in range(le) ]
    cur=IxJ
    for i in range(le):
        J=le-i-1
        ret[J]=cur%Ndim[J]
        cur=(cur-ret[J])//Ndim[J]
    return ret

def popsum(pop, spinrestriction, pbc, nKpt ):
    ndim=np.shape(pop);le_ndim=len(ndim);Ld=len(pop)
    ret=0.0;
    if(pbc):
        for I in range(Ld):
            ret += sum( np.ravel(pop[I]) )
        ret=ret/float(nKpt)
    else:
        if( le_ndim == 1 ):
            ret=sum(pop)
        else:
            for I in range(Ld):
                ret += sum( np.ravel(pop[I]) )
    return ret

## 
## print_tdeorbs1: receives spin-resolved matrix
##  tdMO[kp][nAO][nMO], refmo[kp][nAO][nMO], 
##
def print_tdeorbs1(tdMO,rttddft,pbc,refMO,FockMat,mo_occ,job="temp",Append=True,step=-1,tm_au=-1, 
                get_eorbs=None,get_pop=None,md=None,popsum_dev=None,title="", FermiLv_au=0.0):
    ## AUinFS=0.02418884326198665673981200055933
    # mo_occ : UHF,ROHF  [sp][nMO]  / [sp][nKpt][nMO]
    #          RHF       [nMO]      / [nKpt][nMO]
    Ndim_MO=np.shape(tdMO)
    
    TINY=1.0e-20
    AUinFS=2.418884326058678e-2
    Ndim=np.shape(tdMO)
    if( not pbc ):
        nkpt=1;nAO=Ndim[0];nMO=Ndim[1]
    else:
        nkpt=Ndim[0];nAO=Ndim[1];nMO=Ndim[2]

    kvectors = np.reshape( rttddft.kpts, (-1,3) )
    ### print("ndim_occ:",end="");print(np.shape(mo_occ))
    ### print("mo_occ:",end="");print(mo_occ)
    
    ndim_occ=np.shape(mo_occ);
    N_ele=0.0
    if( len(ndim_occ)==1 ):
        N_ele=sum(mo_occ)
    else:
        for I in range(len(mo_occ)):
            N_ele += sum( np.ravel(mo_occ[I]) )
            
    kvectors = (None if(not pbc) else np.reshape( rttddft.kpts, (-1,3)))
    
    eorbs=np.zeros([nMO])
    pop  =np.zeros([nMO]);avg_pop=( np.zeros([nMO]) if(pbc and nkpt>1) else None)
    filenames=[ job+('' if(not pbc) else '_*')+"_eorbs.dat",
           job+('' if(not pbc) else '_*')+"_tdpop.dat"] 
    popsum=0.0
    for kp in range(nkpt):
        wfn=( tdMO if(not pbc) else tdMO[kp])
        ref_wfn=( refMO if(not pbc) else refMO[kp] )
        if( not pbc ):
            S1e= rttddft.get_ovlp();                          F = FockMat;  Occ=mo_occ
        else:
            S1e= rttddft.get_ovlp(rttddft.cell,kvectors[kp]); F=FockMat[kp];Occ=mo_occ[kp]
        # (i) : C F C
        
        for al in range(nMO):
            eps = np.vdot( wfn[:,al], np.matmul( F, wfn[:,al] ) ).real
            eorbs[al]=eps

        if( get_eorbs is not None ):
            if( not pbc ):
                for al in range(nMO):
                    get_eorbs.append(eorbs[al])
            else:
                get_eorbs.append(eorbs.copy())
            
        eorbf = ( job if(not pbc) else job+"_kp%03d"%(kp) )+"_eorbs.dat"
        fd=futils.fopen( eorbf, ("a" if Append else "w"))
        if( not Append ):
            print_00("#%6s  %14s %14s   %16s    %14s"%("step","tm_au","tm_fs","FermiLv_au","eorbs"),file=fd)
            print_00("### %03d  %12.6f %12.6f %12.6f"%(kp, kvectors[kp][0], kvectors[kp][1], kvectors[kp][2]),file=fd)
        print_00(" %6d  %14.4f %14.4f   %16.8f  "%(step,tm_au,tm_au*AUinFS,FermiLv_au) + d1toa(eorbs,format="%14.6f "),file=fd)
        futils.fclose(fd)
        # (ii) : \sum_a | < C0_n | S | C_a > |^{2} * w_a

        for n in range(nMO):
            pop[n]=0.0
            for al in range(nMO):
                if( Occ[al]<TINY ):
                    continue
                cof = np.vdot( ref_wfn[:,n], np.matmul( S1e, wfn[:,al] ))
                wg = cof.real**2 + cof.imag**2
                pop[n] += wg* Occ[al]

        if( get_pop is not None ):
            if( not pbc ):
                for al in range(nMO):
                    get_pop.append(pop[al])
            else:
                get_pop.append(pop.copy())

        if( avg_pop is not None ):
            for n in range(nMO):
                avg_pop[n]+= pop[n]
        popsum+= sum(pop)

        popf = ( job if(not pbc) else job+"_kp%03d"%(kp) )+"_tdpop.dat"
        fd=futils.fopen( popf, ("a" if Append else "w"))
        if( not Append ):
            print_00("#%5s %14s %14s  %14s"%("step","tm_au","tm_fs","eorbs"),file=fd)
        print_00("%6d %14.4f %14.4f "%(step,tm_au,tm_au*AUinFS) + d1toa(pop,format="%24.14f "),file=fd)
        futils.fclose(fd)
        ### os.system("fopen "+popf);
    if( avg_pop is not None ):
        for n in range(nMO):
            avg_pop[n]/= float(nkpt)
        popf = job + "_tdpop.dat"
        fd=futils.fopen( popf, ("a" if Append else "w"))
        if( not Append ):
            print_00("#%5s %14s %14s  %14s"%("step","tm_au","tm_fs","eorbs"),file=fd)
        print_00("%6d %14.4f %14.4f "%(step,tm_au,tm_au*AUinFS) + d1toa(avg_pop,format="%24.14f "),file=fd)
        futils.fclose(fd)
        ### os.system("fopen "+popf);
    if( pbc ):
        popsum=popsum/float(nkpt)

    if( abs(popsum-N_ele)>1.0e-5 ):
        print_00("#print_tdeorbs1:popsum:%14.6f / N_ele:%14.6f "%(popsum,N_ele)+title,warning=1) 
        trdm    = calc_trDM1( tdMO,mo_occ,pbc,rttddft)
        trdmREF = calc_trDM1(refMO,mo_occ,pbc,rttddft)
        print_00("#print_tdeorbs1:trDM:%14.6f %14.6f"%(trdm,trdmREF),warning=1)
        if( md is not None ):
            Dic1={}
            md.normalize_MOcofs(rttddft,MO_Coeffs=refMO,dict=Dic1,update_self=False)
            print_00("popsum_dev:refMO_dev:"+str(Dic1))
            Dic1={}
            md.normalize_MOcofs(rttddft,MO_Coeffs=tdMO,dict=Dic1)
            print_00("popsum_dev:%e orth:"%(abs(popsum-N_ele))+str(Dic1))
    if(popsum_dev is not None):
        popsum_dev.clear()
        popsum_dev.append( abs(popsum-N_ele) )
#    assert abs(popsum-N_ele)<1.0e-3,"popsum deviates"
    xassertf( abs(popsum-N_ele)<1.0e-3,"print_tdeorbs1:popsum %f/%f dev %e"%(popsum,N_ele,abs(popsum-N_ele)),1)
    return filenames


def check_eigvecs(title,A,U,E):
    Ndim=np.shape(U)
    nMO=Ndim[1];devmx=0.0
    for I in range(nMO):
        V=np.matmul( A,U[:,I])
        C=U[:,I]*E[I]
        dev=z1diff(C,V)
        devmx=max([dev,devmx])
    print_00("check_soln:%s %e"%(title,devmx))
    return devmx;

def int_factorial(arg):
        #0 1 2 3 4    5   6    7     8      9       10
    ref=[1,1,2,6,24,120,720,5040,40320,362880, 3628800]
    if(arg<=10):
        return ref[arg];
    else:
        n=10;ret=float(ref[n])
        for I in range(n+1,arg+1):
            ret*=I
        return ret

def hfodd_factorial(odd):
    assert (odd%2==1),"hfodd_factorial:%d"%(odd)
    sqrt_pi=1.7724538509055160272981674833411;
    p=(odd-1)//2;## 1,3,5,... => 0,1,2,... and  retv[p]= Gamma((2p+1)/2) = ((2p-1)/2)*Gamma[p-1]
    #      0(1/2)   1(3/2)       2             3              4(9/2)
    retv=[ sqrt_pi, 0.5*sqrt_pi, 0.75*sqrt_pi, 1.875*sqrt_pi, 6.5625*sqrt_pi]
    n_last=4
    if(p<=n_last):
        return retv[p]
    ret=retv[n_last]
    for i in range(n_last+1,p+1):  ## eg. Gamma(9/2)*(5-0.5) ...
        ret*=(i-0.5)
    return ret

def gamma_hfint(arg):
    if(arg%2==0): ## Gamma(n)=(n-1)! 
        p=arg/2;
        return int_factorial(p-1);
    else:
        return hfodd_factorial(arg)

def gammln(xx):
    cof=[ 76.18009172947146e+00, -86.50532032941677e+00,
          24.01409824083091e+00, -1.231739572450155e+00,
           0.1208650973866179e-02, -0.5395239384953e-05];
    x=xx
    y=x; ### y=x=xx;
    tmp=x+5.5; tmp=tmp-(x+0.5)*math.log(tmp);
    ser=1.000000000190015;
    for j in range(6):
        y+=1; ser+=cof[j]/float(y)  ### for(j=0;j<6;j++)ser+=cof[j]/++y;
    return -tmp+math.log(2.5066282746310005*ser/x);

class readwritezN_static_:
    counter={}

    @staticmethod
    def Countup(key,inc=True,offset=0):   ## by default 1,2,3,... 
        if( key not in readwritezN_static_.counter ):
            readwritezN_static_.counter.update({key:0})
            return 0;
        if(inc):
            readwritezN_static_.counter[key]+=1
        return readwritezN_static_.counter[key]
# 
# @return 'R': (zbuf[nblock][ --ndarray-- ] or zbuf[ --Ndim_zbuf-- ]), list_of_commentlines[nblock]
#         'W': fpath,maxdiff,indexAT,sqrdiffsum,strdiff
# 
def readwritezN(RorW,name,zbuf,Ith=None,Ndim_zbuf=None,format="%s_%04d.dat",compTO=None,description="",logfile=None,iverbose=1):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if( MPIrank > 0 ):
        write_once("utils.readwritezN:mpi",
                "utils.readwritezN invoked by thread %02d"%(MPIrank))
    
    if(Ith is None):
        if( RorW=='R'):
            Ith=readwritezN_static_.Countup(name,inc=False)
        elif( RorW=='W'):
            Ith=readwritezN_static_.Countup(name,inc=True)
    fpath=format%(name,Ith)

    if( iverbose>(0 if(RorW=='W') else 1) ):
        print_00("#readwritezN:"+RorW+" Ith:%d FILE:"%(Ith)+fpath)

    if( compTO is None ):
        compTO=Ith-1

    retv=None
    if( RorW=='W' ):
        maxdiff=None;indexAT=None;sqrdiffsum=None;strdiff=None
        readwritezN_writeone_(fpath,zbuf,header='#'+description, iverbose=iverbose)
        if( compTO >=0 ):
            maxdiff,indexAT,sqrdiffsum,strdiff = readwritezN_compto(zbuf,Ith, name,compTO,format=format,logfile=logfile,iverbose=iverbose)
        return fpath,maxdiff,indexAT,sqrdiffsum,strdiff
    elif(RorW=='R'):
        sbuf=[]
        ret=readwritezN_readone_(fpath,sbuf,Ndim=Ndim_zbuf, iverbose=iverbose)
        if( compTO >=0 ):
            readwritezN_compto(ret,Ith, name,compTO,format=format,logfile=logfile,iverbose=iverbose)
        return ret,sbuf
    else:
        assert False,""+RorW
        return None
    
# maxdiff,indexAT,sqrdiffsum,strdiff
def readwritezN_compto(zbuf,Ith,name,Jth,format,logfile=None,iverbose=1,diffTOL=1.0e-6):
    comm=MPI.COMM_WORLD
    MPIrank=comm.Get_rank()

    Ndim=np.shape(zbuf)
    fpth2=format%(name,Jth);sbuf=[]
    error=False
    maxdiff=None; indexAT=None; sqrdiffsum=None; strret=None
    if( not os.path.exists(fpth2) ):
        strret="!W: %d th of %s : %s does not exist"%(Jth,name,fpth2)
        error=True
    else:
        zbf2=readwritezN_readone_(fpth2,sbuf,Ndim=Ndim)
        maxdiff,indexAT,vals,sqrdiffsum = maxdiff_zbufs_(zbuf,zbf2)
        fdA=[sys.stdout];fdOU=None
        strret="#%s %04d/%04d maxdiff:%e @%s %f,%f / %f,%f dist:%e"%(\
                name,Ith,Jth,maxdiff,str(indexAT),vals[0].real,vals[0].imag,
                vals[1].real, vals[1].imag, np.sqrt(sqrdiffsum))
    if( MPIrank == 0 ):
        if( error or iverbose>2 or \
            (iverbose>1 and (diffTOL>0 and maxdiff>0.1*diffTOL)) or\
            (iverbose>0 and (diffTOL>0 and maxdiff>diffTOL)) ):
            if(logfile is not None):
                fdOU=open(logfile,"a");fdA.append(fdOU)
            
            for fd in fdA:
                print(strret,file=fd)
            if(fdOU is not None):
                fdOU.close()
    return maxdiff,indexAT,sqrdiffsum,strret

def readwritezN_writeblock_(fd,cbuf=None,format="%16.8f %16.8f",delimiter='        \t'):
    
    assert cbuf is not None,""
    Ndim=np.shape(cbuf);leNdim=len(Ndim)
    if(leNdim==2):
        for I in range(Ndim[0]):
            str=format%(cbuf[I][0].real, cbuf[I][0].imag)
            for J in range(1,Ndim[1]):
                str+=delimiter+format%(cbuf[I][J].real, cbuf[I][J].imag)
            print(str,file=fd)
    elif(leNdim==1):
        str=format%(cbuf[0].real, cbuf[0].imag)
        for I in range(1,Ndim[0]):
            str+=delimiter+format%(cbuf[I].real, cbuf[I].imag)
        print(str,file=fd)
    else:
        assert False,""
def readwritezN_writeone_(fpath,zbuf,Append=False,header=None,iverbose=1):
    fd=open(fpath,('a' if(Append) else 'w'))
    Ndim=np.shape(zbuf);leNdim=len(Ndim)
    if(header is not None):
        print_00(header,file=fd);
    if(leNdim<=2):
        readwritezN_writeblock_(fd,cbuf=zbuf)
    elif(leNdim==3):
        for I in range(Ndim[0]):
            print_00( ('\n\n\n' if(I>0) else '')+'###%05d:%04d'%(I,I),file=fd)
            readwritezN_writeblock_(fd,cbuf=zbuf[I])
    elif(leNdim==4):
        IxJ=-1
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                IxJ+=1
                print( ('\n\n\n' if(IxJ>0) else '')+'###%05d:%04d,%04d'%(IxJ,I,J),file=fd)
                readwritezN_writeblock_(fd,cbuf=zbuf[I][J])
    elif(leNdim==5):
        IxJ=-1
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                for K in range(Ndim[2]):
                    IxJ+=1
                    print( ('\n\n\n' if(IxJ>0) else '')+'###%05d:%04d,%04d,%04d'%(IxJ,I,J,K),file=fd)
                    readwritezN_writeblock_(fd,cbuf=zbuf[I][J][K])
    else:
        assert False,"Ndim:"+str(Ndim)

def readwritezN_readone_(fpath,sbuf,Ndim=None,iverbose=1):
    ## [ ndArray, ... ]
    sbuf.clear()
    fd=open(fpath,'r')
    Ret=[];nblock=0
    cbuf=None;nblank=0;commentlines="";ncl=0
    for line0 in fd:
        line=line0.strip()
        le=len(line)
        if(le<1):
            nblank+=1;continue
        if(line.startswith('#')):
            commentlines+=("" if(ncl==0) else "\n")+line;continue
        else:
            if(nblank>1):
                if(cbuf is not None):
                    if(iverbose>1):
                        print_00( "#read:%05d th block(%s)  in file:%s"%(len(Ret),commentlines,fpath) )
                    Ret.append(np.array(cbuf));sbuf.append(commentlines);
                    cbuf=None;nblock+=1
                nblank=0;commentlines="";ncl=0
            carr=parse_complexes(line);
            assert carr is not None,""
            if( cbuf is None ):
                cbuf= [];
            cbuf.append(carr)
    fd.close()
    if(cbuf is not None):
        Ret.append(np.array(cbuf));sbuf.append(commentlines);
    if(Ndim is not None):
        Ret=np.reshape(Ret,Ndim)
    return Ret;

def dN_to_str(arr,format="%20.12f",delimiter=' ',bra='[',ket=']'):
    Ndim=np.shape(arr);leNdim=len(Ndim)
    ret=""
    if(leNdim==1):
        le=Ndim[0];
        for J in range(le):
            ret+=("" if(J==0) else delimiter)+format%(arr[J])
    elif(leNdim==2):
        for I in range(Ndim[0]):
            row=""
            for J in range(Ndim[1]):
                row+=("" if(J==0) else delimiter)+format%(arr[I][J])
            ret+=bra+row+ket
    return bra+ret+ket


def xassertf(bool,msg,lvl):
    if(bool):
        return 0
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank(); MPIsize=comm.Get_size()
    fd=open("assertion_%02d_warning.log"%(MPIrank),"a")
    msg+='  \t\t'+str(datetime.datetime.now())
    print(msg,file=fd);
    print("#!W py assertion failed:"+msg);
    if(lvl<0):
        assert False,""

def calc_trDM1(MO_coefs,MO_occ,pbc,rttddft):
    ## MOcofs[nKpt][nAO][nMO] mo_occ[nKpt][nMO]
    wct00=time.time()
    Ndim_MOcofs=np.shape(MO_coefs)
    Ndim_MOocc =np.shape(MO_occ)
    if( pbc ):
        assert len(Ndim_MOcofs)==3 and len(Ndim_MOocc)==2,""
        nKpt=Ndim_MOcofs[0];nAO=Ndim_MOcofs[1];nMO=Ndim_MOcofs[2]
    else:
        assert len(Ndim_MOcofs)==2 and len(Ndim_MOocc)==1,""
        nKpt=1;nAO=Ndim_MOcofs[0];nMO=Ndim_MOcofs[1]

    S1e = (None if(pbc) else rttddft.get_ovlp())
    kvectors = (None if (not pbc) else np.reshape( rttddft.kpts, (-1,3)))

    ret=0.0
    for kp in range(nKpt):
        coefs=( MO_coefs if(not pbc) else MO_coefs[kp] )
        occ  =( MO_occ   if(not pbc) else MO_occ[kp] )
        if( pbc ):
            S1e=rttddft.get_ovlp( rttddft.cell, kvectors[kp])
        dm1=np.matmul( coefs, np.matmul( np.diag(occ), np.matrix.getH(coefs)))
#        dm1=np.zeros([nAO,nAO],dtype=np.complex128)
#        for el in range(nMO):
#            for mu in range(nAO):
#                for nu in range(nAO):
#                    dm1[mu][nu] += coefs[mu,el] * occ[el] * np.conj( coefs[nu,el] )
        DxS=np.matmul(dm1,S1e)
        trDM1=matrixTrace(DxS)
        ret=ret+trDM1.real
    if(pbc):
        ret=ret/float(nKpt)
    wct99=time.time();print("trDM1:elapsed %f"%(wct99-wct00))
    return ret

def normalize(A):
    le=len(A)
    cum=0.0
    dtype=np.array(A).dtype
    if( dtype == np.float64 or dtype==float ):
        for I in range(le):
            cum+=A[I]**2
    else:
        for I in range(le):
            cum+=A[I].real**2 + A[I].imag**2
    norm=np.sqrt(cum);fac=1.0/norm
    return np.array( [ A[I]*fac for I in range(le) ] )

def maxdiff_zbufs_(Lhs,Rhs,Ndim=None):
    if(Ndim is None):
        Ndim=np.shape(Lhs);
    lh=np.ravel(Lhs);rh=np.ravel(Rhs);
    le=len(lh);lerh=len(rh);
    if( lerh != le ):
        print_00("!E:length differs:%d/%d"%(le,lerh));
        return None,None,None,None
    maxdiff=-1;Iat=-1;values=None;sqrdiffsum=0.0
    for I in range(le):
        dum=abs(lh[I]-rh[I]);sqrdiffsum+=dum*dum
        if(maxdiff<dum):
            maxdiff=dum;Iat=I;values=[lh[I],rh[I]]
    
    IandJ_at=IxJ_to_IandJ(Iat,Ndim)
    return maxdiff,IandJ_at,values,sqrdiffsum

def parse_complexes(string,strict=True):
    
    sbuf=string.split()
    le=len(sbuf)
    ## assert le%2==0,"odd len:"+str(le)
    nCol=le//2
    ret=[]
    for I in range(nCol):
        ret.append( float(sbuf[2*I]) + 1j*float(sbuf[2*I+1]) )
    if(le%2!=0):
        if(strict):
            print_00("#!W:parse_complexes:odd num of columns:%d"%(le));
            return None
    return ret;

def matrixTrace(tgt):
    Ndim=np.shape(tgt);Ld=Ndim[0]
    assert len(Ndim)==2 and Ndim[0]==Ndim[1],""+str(Ndim)
    if(Ld<1):
        return 0.0
    ret=tgt[0][0];
    for I in range(1,Ld):
        ret+=tgt[I][I]
    return ret
def parse_dict(src,delimiter=[None,'='],trim_blank=True,float_fields=[],int_fields=[],complex_fields=[],abort=False):
    
    arr=(src.split() if(delimiter[0] is None) else src.split(delimiter[0]));
    ret={}
    for item in arr:
        if(trim_blank):
            item=item.strip()
        a=item.split(delimiter[1]);
        if( len(a)!=2 ):
            assert (not abort),item+":len=%d"%( len(a) )
        key=( a[0].strip() if(trim_blank) else a[0])
        val=( a[1].strip() if(trim_blank) else a[1])
        if( key in float_fields ):
            val=np.float64(val)
        if( key in int_fields ):
            val=int(val)
        if( key in complex_fields ):
            val=np.complex128(val)
        ret.update({key:val})
    return ret
def check_Hermicity_xKaa(H_or_A,matrix,comp,nKpt_or_None,tol=1.0e-7,Dic=None,title=""):
    retv=0;nerror=0
    assert H_or_A=='H' or H_or_A=='A',""
    devmax=-1;at=None;devsum=0.0;ndev=0;vals=None
    nKpt=(1 if(nKpt_or_None is None) else nKpt_or_None)
    for dir in range(comp):
        for kp in range(nKpt):
            mat=(matrix[dir] if(nKpt_or_None is None) else matrix[dir][kp])
            ndim=np.shape(mat)
            if( ndim[0]!=ndim[1] ):
                nerror+=1;continue
            if( H_or_A == 'H'):
                for I in range(ndim[0]):
                    for J in range(I+1):
                        dum=abs( np.conj(mat[J][I])-mat[I][J] )  ## for I==J, this is abs(imaginary) ..
                        devsum+=dum**2
                        if( dum > devmax ):
                            devmax=dum; at=[dir,kp,I,J]; vals=[ mat[I][J], mat[J][I] ]
                        if( dum > tol ):
                            ndev+=1
            else:
                for I in range(ndim[0]):
                    for J in range(I+1):
                        dum=abs( np.conj(mat[J][I])+mat[I][J] )  ## for I==J, this is abs(real) ..
                        devsum+=dum**2
                        if( dum > devmax ):
                            devmax=dum; at=[dir,kp,I,J]; vals=[ mat[I][J], mat[J][I] ]
                        if( dum > tol ):
                            ndev+=1
    if(Dic is not None):
        Dic.update({"ndev":ndev,"devmax":devmax,"at":at,"devsum":devsum,"values":vals})
    if(ndev>0):
        print("#check_Hermicity:failed:%s:Ndev:%d"%(title,ndev)\
            +str({"ndev":ndev,"devmax":devmax,"at":at,"devsum":devsum,"values":vals}))
    return ndev;

def calc_3Dvolume(Vectors):
    vecs=np.reshape( Vectors, [3,3] )
    axb= AxB( vecs[0], vecs[1] )
    retv= np.vdot( vecs[2], axb )
    return retv

def toComplexArray(src):
    if( np.array(src).dtype == complex ):
        print_00("#toComplexArray:is complex");
        return src
    if( np.array(src).dtype == np.complex128 ):
        print_00("#toComplexArray:is complex128");
        return src
    Ndim=np.shape(src)
    ret=np.zeros( Ndim, dtype=np.complex128 )
    rank=len(Ndim)
    if( rank==1 ):
        for I in range(Ndim[0]):
            ret[I]=src[I]
    elif(rank==2):
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                ret[I][J]=src[I][J]
    elif(rank==3):
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                for K in range(Ndim[2]):
                    ret[I][J][K] = src[I][J][K]
    elif(rank==4):
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                for K in range(Ndim[2]):
                    for L in range(Ndim[3]):
                        ret[I][J][K][L] = src[I][J][K][L]
    else:
        assert False,"unimplemented"
    return ret

def print_z2array_(fpath,A,Append=False,header=None):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return

    fd=open(fpath,('a' if(Append) else 'w'));
    if( header is not None ):
        print(header,file=fd);
    Ndim=np.shape(A)
    assert len(Ndim)<=2,""
    if( len(Ndim)==2 ):
        for I in range(Ndim[0]):
            string=""
            for J in range(Ndim[1]):
                string+="%12.6f %12.6f        "%(A[I][J].real,A[I][J].imag)
            print(string,file=fd)
    elif( len(Ndim)==1 ):
        string=""
        for J in range(Ndim[0]):
            string+="%12.6f %12.6f        "%(A[J].real,A[J].imag)
        print(string,file=fd)
    fd.close()
        
def print_z2array(fpath,A,comment="",Append=False):
    comm = MPI.COMM_WORLD; MPIrank = comm.Get_rank()
    if(MPIrank!=0):
        return

    Ndim=np.shape(A);header=None
    leNdim=len(Ndim)
    if( leNdim<=2 ):
        if( comment is not None ):
            header='#'+str(comment)+" Ndim:"+str(Ndim)
        print_z2array_(fpath,A,Append=Append,header=header)
    elif( leNdim==3 ):
        for I in range(Ndim[0]):
            header=( '#'+comment+'\n##%03d '%(I)+comment+" Ndim:"+str(Ndim) if(I==0) else '\n\n\n##%03d '%(I))
            print_z2array_(fpath,A[I],Append=(True if(I>0) else Append),header=header)
    elif( leNdim==4 ):
        IxJ=-1
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                IxJ+=1
                header=('#'+comment+" Ndim:"+str(Ndim)+'\n##%05d:%02d,%02d '%(IxJ,I,J) if(IxJ==0) else 
                        '\n\n\n##%05d:%02d,%02d '%(IxJ,I,J))
                print_z2array_(fpath,A[I][J],Append=(True if(IxJ>0) else Append),header=header)
    elif( leNdim==5 ):
        IxJxK=-1
        for I in range(Ndim[0]):
            for J in range(Ndim[1]):
                for K in range(Ndim[2]):
                    IxJxK+=1
                    header=('#'+comment+" Ndim:"+str(Ndim)+'\n##%05d:%02d,%02d,%02d '%(IxJxK,I,J,K) if(IxJxK==0) else 
                            '\n\n\n##%05d:%02d,%02d,%02d '%(IxJxK,I,J,K))
                    print_z2array_(fpath,A[I][J][K],Append=(True if(IxJxK>0) else Append),header=header)
    else:
        assert False,""
def dist3D(a,b):
    return np.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 )
def AxB(A,B):
    return np.array([ A[1]*B[2] - A[2]*B[1],
             A[2]*B[0] - A[0]*B[2],
             A[0]*B[1] - A[1]*B[0] ])

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

def z1maxloc(arr):
    N=len(arr)
    mxv=-1.0;at=None
    for k in range(N):
        a=abs(arr[k])
        if( mxv < a ):
            mxv=a;at=k
    return at

def check_wfNorm(rttddft,pbc,wfn,AOrep,sqrdevtol_FIX=1.0e-6,title=""):
    Ndim=np.shape(wfn)
    if( len(Ndim)==2 ):
        nkpt=1;nAO=Ndim[0];nMO=Ndim[1]; 
    elif( len(Ndim)==3 ):
        nkpt=Ndim[0];nAO=Ndim[1];nMO=Ndim[2]
    else:
        assert False,""

    sqrNorms=np.zeros(nMO)
    kvectors = (None if (not pbc) else np.reshape( rttddft.kpts, (-1,3)))
    maxdev=-1.0; Nfix=0
    for kp in range(nkpt):
        if(AOrep):            
            if( not pbc ):
                tgt=wfn;     S1e=rttddft.get_ovlp()
            else:
                tgt=wfn[kp]; S1e=rttddft.get_ovlp( rttddft.cell, kvectors[kp])
            for I in range(nMO):
                sqrNorms[I]=np.vdot( tgt[:,I], np.matmul( S1e, tgt[:,I] ) ).real
                dev = abs( sqrNorms[I]-1.0 )
                if( dev > maxdev ):
                    maxdev = dev
                if( sqrdevtol_FIX>=0  and  dev > sqrdevtol_FIX ):
                    fac=1.0/np.sqrt( sqrNorms[I] );Nfix+=1
                    for K in range(nAO):
                        tgt[K][I]=tgt[K][I]*fac
        else:
            if( not pbc ):
                tgt=wfn;
            else:
                tgt=wfn[kp];
            for I in range(nMO):
                sqrNorms[I]=np.vdot( tgt[:,I], tgt[:,I] ).real
                dev = abs( sqrNorms[I]-1.0 )
                if( dev > maxdev ):
                    maxdev = dev
                if( sqrdevtol_FIX>=0  and  dev > sqrdevtol_FIX ):
                    fac=1.0/np.sqrt( sqrNorms[I] );Nfix+=1
                    for K in range(nAO):
                        tgt[K][I]=tgt[K][I]*fac

##        logger.Info("#orth_tdMO:%s sqrdevMax:%e Nfix:%d/%d"%(title,maxdev,Nfix,nMO));

def a2sqrdiff(lhs,rhs,DAG_OFDdev=None):
    Ndim=np.shape(lhs)
    Ndim2=np.shape(rhs)
    assert i1eqb(Ndim,Ndim2),""
    DAGdev=-1;OFDdev=-1;DAGref=None;OFDref=None
    cum=0.0
    for I in range(Ndim[0]):
        for J in range(Ndim[1]):
            cdum=lhs[I][J]-rhs[I][J]
            sqrDiff= cdum.real**2 + cdum.imag**2
            diff= np.sqrt(sqrDiff)
            if(I==J):
                if(DAGdev<diff):
                    DAGdev=diff;DAGref=[I,I]
            else:
                if(OFDdev<diff):
                    OFDdev=diff;OFDref=[I,J]
            cum+= sqrDiff
    if( DAG_OFDdev is not None):
        DAG_OFDdev.update({'DAGdev':DAGdev,'OFDdev':OFDdev})

    return sqrDiff

def calc_Sroot(S,powX2):
    Nret=len(powX2)
    dtype=np.array(S).dtype
    iscomplex=False
    if( dtype == complex ):
        iscomplex=True
        eigvals,vecs,info=scipy.linalg.lapack.zheev(S)
    else:
        eigvals,vecs,info=scipy.linalg.lapack.dsyev(S)
    assert (info==0),"dsyev/zheev failed"
    N=len(S)
    Retv=[]

    for Iret in range(Nret):
        pX2=powX2[Iret]
        X=np.zeros([Ld,Ld],dtype=(np.complex128 if(iscomplex) else np.float64))
        for I in range(N):
            for J in range(N):
                cdum=( np.complex128(0.0) if(iscomplex) else 0.0)
                if( pX2 == 1 ):
                    for k in range(N):
                        cdum+= vecs[I][k]*math.sqrt(eigvals[k])*np.conj( vecs[J][k])
                elif( pX2 == -1 ):
                    for k in range(N):
                        cdum+= vecs[I][k]*(1.0/math.sqrt(eigvals[k]))*np.conj( vecs[J][k])
                elif( pX2 == -2 ):
                    for k in range(N):
                        cdum+= vecs[I][k]*(1.0/eigvals[k])*np.conj(vecs[J][k])
                X[I][J]=cdum
        if(Nret==1):
            return X
        Retv.append(X)
    return Retv

def calc_ldos( filename, iappend, 
               pbc, nkpt,
               ebuf,wbuf,ibuf,ncols=3,au_to_eV=False,
               emin_eV=None, emax_eV=None, de_eV=None, Nstep=None, margin_eV=None, FermiLv_au=None,
               fileheader=None,widths_eV=[0.2123, 2.7211386024367243, 0.1, 0.01]):
    
    HARTREEinEV=27.211386024367243
    ithTOjs= np.argsort(ebuf)
    e_eV_sorted=np.zeros([bfsz],dtype=np.float64); 
    w_sorted=np.zeros([bfsz],dtype=np.float64);
    i_sorted=np.zeros([bfsz,ncols],dtype=int)
    e_fac=(1.0 if(not au_to_eV) else HARTREEinEV)
    for ith in range(bfsz):
        js=ithTOjs[ith];e_eV_sorted[ith]=ebuf[js]*e_fac;
        w_sorted[ith]=wbuf[js];i_sorted[ith]=ibuf[js]
    if(pbc):
        assert nkpt is not None,""
    if(pbc and nkpt>1):
        w_fac=1.0/float(nkpt)
        for ith in range(bfsz):
            w_sorted[ith]=w_sorted[ith]*w_fac
    eorb_min_eV=e_eV_sorted[0]
    eorb_max_eV=e_eV_sorted[bfsz-1]

    daf_eorbs=filename+"_eorbs.dat"
    fd=open(daf_eorbs,("a" if(iappend!=0) else "w"))
    if( fileheader is not None):
        print(fileheader,file=fd);
    print("#%19s  %20s  %5s %5s %5s"%("eorb_eV","weight","sp","kp","mo"),file=fd) 
    for ith in range(bfsz):
        print("%20.10f  %20.10f  %5d %5d %5d"%(e_eV_sorted[ith],w_sorted[ith], 
                     i_sorted[ith][0],i_sorted[ith][1],i_sorted[ith][2]),file=fd)
    if(trailer is not None):
        print(trailer,file=fd)
    fd.close()

    if( (de_eV is None) and (Nstep is None ) ):
        min_w=min(widths_eV)
        de_eV=( 0.25*min_w if(min_w<0.1) else 0.1*min_w )
    max_w=max(widths_eV)
    if( margin_eV is None ):
        margin_eV=max( 2*max_w, (8.0 if de_eV is None else de_eV*20) )
        printout("#calc_ldos:margin_eV:%f   %f %f %f"%(margin_eV,2*max_w,8.0,de_eV*20))
    if( emin_eV is None ):
        emin_eV= eorb_min_eV - margin_eV
    if( emax_eV is None ):
        emax_eV= eorb_max_eV + margin_eV

    if( de_eV is not None):
        Nstep=int(math.ceil((emax_eV-emin_eV)/de_eV))
    elif( Nstep is not None):
        de_eV=(emax_eV-emin_eV)/float(Nstep)
    printout("#calc_ldos:Nstep:%d de_eV:%f"%(Nstep,de_eV))

    sqrt2pi=2.506628274631000502415765284811
    daf_ldos=filename+"_ldos.dat"
    fd=open( daf_ldos,("a" if(iappend!=0) else "w"))
    if( header is not None):
        print(header,file=fd);
    e_eV=emin_eV
    wgt_ref=(1.0 if(spinrestriction=='U') else 2.0)/(1.0 if(not pbc) else float(nkpt))
    wgsum = sum(w_sorted);
    print("#calc_ldos:wgsum:%f"%(wgsum))
    if( N_elec is not None):
        assert abs(N_elec-wgsum)<1.0e-3,""
    ## exp(-0.5*((e_eV-ej_eV)/wid)**2)
    Nwid=len(widths_eV)
    nrmz_factors=[ 1.0/(sqrt2pi*widths_eV[kwid]) for kwid in range(Nwid) ]
    fn=[ 0.0 for kwid in range(Nwid) ]
    fn_occ=[ 0.0 for kwid in range(Nwid) ]
    cum=[ 0.0 for kwid in range(Nwid) ]
    cum_occ=[ 0.0 for kwid in range(Nwid) ]

    I0=-1
    for I in range(Nstep):
        e_eV=emin_eV + I*de_eV
        for kwid in range(Nwid):
            fn[kwid]=0.0; fn_occ[kwid]=0.0
            for js in range(bfsz):
                arg=-0.5*( ((e_eV-e_eV_sorted[js])/widths_eV[kwid])**2 )
                if(arg<ln_tiny):
                    continue
                gsf=nrmz_factors[kwid]*math.exp(arg);
###                if( js==0 and (I0<0 or I%10==0) ):
###                    print("#E=%f s=0(e=%f) contributes:%f*%f .."%(
###                        e_eV,e_sorted[js], nrmz_factors[kwid]*math.exp(arg),w_sorted[js]))
                fn[kwid]+= gsf*wgt_ref;
                fn_occ[kwid]+= gsf*w_sorted[js]
            cum[kwid]+=fn[kwid]
            cum_occ[kwid]+=fn_occ[kwid]
        if(I==0):
            string="#%14s   "%("e_eV")
            for kwid in range(Nwid):
                str="wid=%9.4f eV: fn,fn_occ "%(widths_eV[kwid])
                string+=" %40s"%(str)
            print(string,file=fd)

        string="%15.6f   "%(e_eV)
        for kwid in range(Nwid):
            string+="       %16.8f  %16.8f"%(fn[kwid],fn_occ[kwid])
        print(string,file=fd)
    string="";
    for kwid in range(Nwid):
        cum[kwid]*=de_eV;cum_occ[kwid]*=de_eV
        string+="       %9.4f %9.4f    "%(cum[kwid],cum_occ[kwid])
        print("#sum:"+string,file=fd)
    if(trailer is not None):
        print(trailer,file=fd)
    fd.close()
    
    FermiLv_eV=None
    if( FermiLv_au is not None ):
        FermiLv_eV=FermiLv_au*HARTREEinEV
    if( gnuplot ):
        fdgnu=open(filename+"_ldos.plt","w");
        print("set term postscript color\nset output \"%s.ps\"\nload \"C:/cygwin64/usr/local/bin/stdcolor.plt\""%(filename+"_ldos"),file=fdgnu)
# 1 : 2,3   4,5 ...
        for kwid in range(Nwid):
            print("set title \"width=%9.4f eV: occsum=%7.3f  Fermi level=%12.6f eV\""%(
                   widths_eV[kwid],cum_occ[kwid],FermiLv_eV),file=fdgnu)
            print('plot "'+daf_ldos+"\" using 1:%d with lines ls 1,\\"%( 2*kwid+2 ),file=fdgnu)
            print("\"\" using 1:%d with lines ls 106"%( 2*kwid+3 ),file=fdgnu)
        fdgnu.close()
    return iappend+1

def tostring(val,format=None,delimiter=","):
    if( isinstance(val,int) or isinstance(val,np.int64) ):
        frmt=("%d" if(format is None) else format)
        return frmt%(val)
    if( isinstance(val,float) or isinstance(val,np.float64) ):
        frmt=("%14.6f" if(format is None) else format)
        return frmt%(val)
    if( isinstance(val,complex) or isinstance(val,np.complex128) ):
        frmt=("%12.5f+j*%12.5f" if(format is None) else format)
        return frmt%(val.real,val.imag)
    if( isinstance(val,str) ):
        return val
    if( isinstance(val,list) or isinstance(val,np.ndarray) ):
        le=len(val)
        sA=[ tostring(val[i],format) for i in range(le) ]
        return delimiter.join(sA)

def split_fpath(org,delimiter='.',fpath_delimiters=['/'],get_dir=False,default=""):
    le=len(org)
    dir=None;core=None;leg=None
    jslash=None
    for j in range(le):
        if( org[le-j-1] in fpath_delimiters ):
            jslash=le-j-1;break

    jdot=None
    for j in range(le):
        if( org[le-j-1]==delimiter ):
            jdot=le-j-1;break
    if(jdot is None):
        core=org;leg=default
    else:
        if( (jslash is not None) and (jdot<jslash) ):
            core=org;leg=default
        else:
            core=org[:jdot]
            leg=org[jdot+1:]
    if(not get_dir):
        return core,leg
    if(jslash is None):
        return default,core,leg
    else:
        dir=org[:jslash]
        if( jdot is None):
            core=org[jslash+1:];leg=default;
        else:
            core=org[jslash+1:jdot];leg=org[jdot+1:]
        return dir,core,leg
