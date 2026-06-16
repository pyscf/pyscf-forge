import numpy as np
import math
import os
import os.path
import sys
import time
import scipy
import scipy.linalg
import datetime
from .Moldyn import get_Fermilvl
from pyscf.pbc.gto import Cell
from .physicalconstants import PhysicalConstants
from .rttddft_common import rttddft_common
from .utils import d1toa,jobname,i1eqb,parse_ints,read_xyzf,write_xyzstring,isScalarNumber,check_matrixrank,\
                  parse_xyzstring,atomicsymbol_to_atomicnumber,atomicnumber_to_atomicsymbol,check_equivalence
from .serialize import serialize
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import df as pbc_df
from .Loglv import printout
from mpi4py import MPI
from .rttddft01 import get_Nele

#
# labels : common symbols of high-symmetrt points of Brillouin zones
def get_named_Kpoints(lattice,names, BravaisVectors):
    # see https://en.wikipedia.org/wiki/Brillouin_zone
    bravaisvectors = np.reshape( BravaisVectors, [3,3] )
    PIx2=6.283185307179586476925286766559
    PI=3.1415926535897932384626433832795
    sqrt2=1.4142135623730950488016887242097
    sqrt3=1.7320508075688772935274463415059
    if( lattice == "NaCl" or lattice == "diamond" ):
        ## both lattice have the same translation vector (1/sqrt2, 1/sqrt2, 0 ) etc.
        a=np.sqrt( np.vdot( bravaisvectors[0], bravaisvectors[0] ) )
        ret=[]
        N=len(names)
        for I in range(N):
            name=names[I].strip()
            if( name == "X" ):
                ret.append( np.array([1.0/sqrt2, 0.0, 0.0 ])*(PIx2/a) )
            elif( name == "W"):
                ret.append( np.array([1.0/sqrt2, 0.5/sqrt2, 0.0 ])*(PIx2/a) )
            elif( name == "K"):
                ret.append( np.array([ 0.75/sqrt2, 0.75/sqrt2, 0.0 ])*(PIx2/a) )
            elif( name == "L"):
                ret.append( np.array([ 0.5/sqrt2, 0.5/sqrt2, 0.5/sqrt2 ])*(PIx2/a) )
            elif( name == "U"):
                ret.append( np.array([ 1.0/sqrt2, 0.25/sqrt2, 0.25/sqrt2 ])*(PIx2/a) )
            elif( name == "gamma" or (name.lower)=="gamma" ):
                ret.append( np.array([ 0.0, 0.0, 0.0]) )
            else:
                assert False,""+name
        return ret
    elif( lattice == "hexagonal" or lattice == "hcp" ):
        ret=[]
        N=len(names)
        for I in range(N):
            name=names[I].strip()
            a=np.sqrt( np.vdot( bravaisvectors[0], bravaisvectors[0] ) )
            c=np.sqrt( np.vdot( bravaisvectors[2], bravaisvectors[2] ) )
            if( name == "K" ):
                ret.append( np.array( [2*PIx2/(3.0*a), 0.0, 0.0] ) )
            elif( name == "M" ):
                ret.append( np.array( [PI/a,-PI/(sqrt3*a), 0.0] ) )
###                ret.append( np.array( [0.0, PIx2/(sqrt3*a), 0.0] ) )
            elif( name == "gamma" ):
                ret.append( np.array( [0.0, 0.0, 0.0] ) )

            elif( name == "H" ):
                ret.append( np.array( [2*PIx2/(3.0*a), 0.0, PI/c] ) )
            elif( name == "L" ):
                ret.append( np.array( [PI/a, -PI/(sqrt3*a), PI/c] ) )
            elif( name == "A" ):
                ret.append( np.array( [0.0, 0.0, PI/c] ) )

            else:
                assert False,""
        return ret
    else:
        assert False,""+lattice
#
# labels : common symbols of high-symmetrt points of Brillouin zones
def gen_kpoints(cell,lattice,labels,nKpoints,BravaisVectors):
    PIx2=6.283185307179586476925286766559
    sqrt2=1.4142135623730950488016887242097
    sqrt3=1.7320508075688772935274463415059
    bravaisvectors = np.reshape( BravaisVectors, [3,3] )
    kpts = cell.make_kpts(nKpoints)
    nkpts = len(kpts)
    kbuf=[ kpts[kp] for kp in range(nkpts) ]
    if( lattice == "NaCl"  or lattice == "diamond" ):
        a=np.sqrt( np.vdot( bravaisvectors[0], bravaisvectors[0]))
        nkpt=len(kbuf)
        for kp in range(nkpt):
            kvc = np.array( kbuf[kp] )
            scaled = kvc / ( PIx2/a )
            print("#kvecs:%03d: %12.6f,%12.6f,%12.6f   %12.6f,%12.6f,%12.6f"%(kp,\
                  kvc[0],kvc[1],kvc[2], scaled[0],scaled[1],scaled[2]))
        namedKpoints = get_named_Kpoints(lattice,labels, BravaisVectors)
        N=len(namedKpoints)
        for I in range(N):
            v= np.array( namedKpoints[I] )
            dist=None; at=-1
            for j in range(nkpt):
                rhs = np.array(kbuf[j])
                dum = np.sqrt( np.vdot( rhs - v, rhs - v) )
                dist = ( dum if(dist is None) else min(dist,dum))
                if( dist < 1e-6 ):
                    at=j;break
            if( at < 0 ):
                print("#%03d.%s::"%(I,labels[I])+ str(["%16.8f"%(dum) for dum in v]))
                kbuf.append(v);nkpt+=1;continue
            else:
                print("### %03d.%s:at %03d"%(I,labels[I],at))
    kvecs=np.array(kbuf)
    print("#gen_kpoints:%d"%(len(kvecs)),type(kvecs),np.shape(kvecs))
    return kvecs
def find_n_closests(tgtvc,n,Vecs,BravaisVectors=None,Ld=None, TINY=1.0e-8,pbc=None):
    tgt=np.array(tgtvc)
    PIx2=6.283185307179586476925286766559
    dbgng=True
    if( pbc is None ):
        pbc = (True if(BravaisVectors is not None) else False)
    bvectors=None
    if( pbc and (BravaisVectors is not None) ):
        bvectors = np.transpose( np.linalg.inv( np.reshape(BravaisVectors,[3,3]) ) )
    vecs=( Vecs if(Ld is None) else np.reshape(Vecs,(-1,Ld)) )
    Nvec=len(vecs)  ## c order vecs[Nvec][Ld]
    assert n<=Nvec,""
    dist=None;at=None;min_at=None
    shiftvecs=[ np.zeros([3],dtype=int) for J in range(Nvec) ]
    dbuf=[]
    for J in range(Nvec):
        rhs = np.array(vecs[J])
        if( not pbc ):
            dum = np.sqrt( np.vdot( rhs - tgt, rhs - tgt) )
        else:
            vdiff = np.zeros([3]);
            for k in range(3):
                rk=np.vdot( rhs, BravaisVectors[k] )
                lk=np.vdot( tgt, BravaisVectors[k] )
                idum = int( round( (rk-lk)/PIx2 ) )
                dk=rk-lk
                if( idum != 0 ):
                    print("shifting by Gvector..")
                    dk-=idum*PIx2;shiftvecs[J][k]= -idum
                vdiff=vdiff + dk*bvectors[k]
            dum=np.sqrt( np.vdot(vdiff,vdiff) ) ## E001
            if( dbgng ):
                print("dum,rhs:",dum,rhs,tgt,shiftvecs[J],bvectors[0],bvectors[1],bvectors[2])
                vref = rhs - tgt + shiftvecs[J][0]*PIx2*bvectors[0] + shiftvecs[J][1]*PIx2*bvectors[1] \
                                 + shiftvecs[J][2]*PIx2*bvectors[2]
                print("vref:",vref)
                vdiffSQR = np.vdot(vref-vdiff, vref-vdiff); print("vdiffSQR:",vdiffSQR)
                test = np.sqrt( ( np.vdot(vref-vdiff, vref-vdiff) ).real )
                assert test<1e-6,"diff=%e Ishift:"%(test)+str(shiftvecs[J])
        dbuf.append(dum)
    ithTOjs=np.argsort(dbuf)
    return np.array( ithTOjs[0:n] ), [ dbuf[ ithTOjs[ith] ] for ith in range(n) ],\
          [ shiftvecs[ ithTOjs[ith] ] for ith in range(n) ]

def expand_vec(Tgt,Vecs,ierr=1):
    tgt=np.array(Tgt);vecs=np.array(Vecs)
    # we exclude trivial cases
    mat = np.array(Vecs[:3]).transpose()
    try:
        invm = np.linalg.inv(mat)
    except np.linalg.LinAlgError as e:
        print("inversion failed:",mat)
        rank=check_matrixrank(mat)
        print("rank:",rank)
        print("expand_vec:inversion fails:",e)
        if( ierr >0 ):
            return None
        else:
            assert False,""
    cofs = np.matmul( invm, tgt )
    dbgng=True
    if( dbgng ):
        test= cofs[0]*vecs[0] + cofs[1]*vecs[1] + cofs[2]*vecs[2]
        diff = np.sqrt( ( np.vdot(test - tgt, test - tgt) ).real )
        assert diff<1e-6,""
    return cofs
###    AP = tgt-vecs[0]
###
###    AB = vecs[1]-vecs[0];eAB=AB/(np.sqrt( np.vdot(AB,AB)))
###    AC = vecs[2]-vecs[0];eAC=AC/(np.sqrt( np.vdot(AC,AC)))
###
###    g_AC = AC - AB * np.vdot( AC,AB )/np.vdot( AB,AB )
###    g_AB = 
###    work = arrayclone(AP)
###    pAC = np.vdot(work,eAC)
###    work -= pAC*eAC
    
def interpolate_eorb(kvec,Vecs,eOrbs,BravaisVectors,TINY=1.0e-7,Description=None,Indices=None):
    Nvec=len(Vecs)
    n=min(7,Nvec);
    indcs, distances, shifts = find_n_closests(kvec,n,Vecs,BravaisVectors=BravaisVectors,Ld=None, TINY=1.0e-8,pbc=True)
    if( distances[0] < TINY ):
        if(Description is not None):
            Description.append("kp=%03d"%(indcs[0]))
#            Description.append("K:%9.4f,%9.4f,%9.4f kp=%03d"%(kvec[0],kvec[1],kvec[2],indcs[0]))
        if(Indices is not None):
            Indices.clear();Indices.append(indcs[0])
        return eOrbs[ indcs[0] ][:]
    else:
        if( n >=3 ):
            bset = select_bset(3, [ Vecs[ indcs[j] ] for j in range(n) ],TINY=1.0e-7,err=None)
            cofs = expand_vec(kvec,bset,-1 )
            if(Description is not None):
                Description.append("%9.4f x(%03d:%12.6f) + %9.4f x(%03d:%12.6f) + %9.4f x(%03d:%12.6f)"%(
                    cofs[0],indcs[0],eOrbs[ indcs[0] ][0],  cofs[1],indcs[1],eOrbs[ indcs[1] ][0],
                    cofs[2],indcs[2],eOrbs[ indcs[2] ][0]))
            if(Indices is not None):
                Indices.clear(); Indices.append(indcs[0]); Indices.append(indcs[1]); Indices.append(indcs[2])
            return cofs[0]*eOrbs[ indcs[0] ] + cofs[1]*eOrbs[ indcs[1] ] + cofs[2]*eOrbs[ indcs[2] ]
        else:
            assert False,""

def select_bset(rank,Src,TINY=1.0e-7,err=None):
    Orth=[];Ret=[];n=0;Isrc=0
    Nsrc=len(Src)
    while(n<rank and Isrc<Nsrc):
        vdum=Src[Isrc].copy();le=np.sqrt( np.vdot(vdum, vdum))
        for j in range(n):
            cof=np.vdot( Orth[j],vdum )
            ### print("%s - %f*%s = %s"%(str(vdum),cof,str(Ret[j]),str(vdum - cof*Ret[j])))
            vdum = vdum - cof*Orth[j]
            
            le=np.sqrt( np.vdot(vdum, vdum))
            if( le < TINY ):
                break
        if(le<TINY):
            print("#select_bset:skip dpd");
            Isrc+=1; continue
        else:
            print("#select_bset:adopt %02d:%e"%(Isrc,le),Src[Isrc]);
            Orth.append( vdum/le )
            Ret.append( Src[Isrc].copy() );n+=1
            Isrc+=1; continue
    if( n< rank ):
        assert False,""
    return Ret
def find_closest(tgt,Vecs,Ld=None, TINY=1.0e-8):
    vecs=( Vecs if(Ld is None) else np.reshape(Vecs,(-1,Ld)) )
    Nvec=len(vecs)  ## c order vecs[Nvec][Ld]
    dist=None;at=None;min_at=None
    for j in range(Nvec):
        rhs = np.array(vecs[j])
        dum = np.sqrt( (np.vdot( rhs - tgt, rhs - tgt)).real )
        if( dist is None  or (dist > dum) ):
            dist = dum; min_at=j
            if( dist < TINY ):
                at=j;break
    return min_at,dist,at
def plot_bandstructure(fpath,Append,lattice,eOrbs,kvecs,labels, BravaisVectors, Occ=None,gnuplot=False, Ndiv=40):
    # eOrbs[nKpt][nMO]
    Ndim=np.shape(eOrbs);
    assert len(Ndim)==2,""
    nKpoints=Ndim[0]; nMO=Ndim[1]
    # labels =["W","L","gamma","X","W","K"]
    PIx2=6.283185307179586476925286766559
    bravaisvectors = np.reshape( BravaisVectors, [3,3] )
    a=np.sqrt( np.vdot( bravaisvectors[0], bravaisvectors[0]))
    N=len(labels)
    namedKpoints = get_named_Kpoints(lattice,labels, BravaisVectors)
    fd=open(fpath,('a' if(Append) else 'w'))
    nkpts=len(kvecs)
    eLast=None;kvcLast=None
    seqno=0;seqlen=0.0
    b_0= (PIx2/a) / float(Ndiv)
    print("#%5s %12s %12s   %s    %12s %12s %12s    %12s %12s %12s"%(\
           "seqno","seqlen","seqlen/(PIx2/a)", "eOrbs:nMO=%d"%(nMO), 
           "k_x","k_y","k_z", "k_x/(PIx2/a)","k_y/(PIx2/a)","k_z/(PIx2/a)"),file=fd)
    
    for I in range(N):
        kvI= np.array( namedKpoints[I] )
        min_at,dist,at = find_closest( kvI,kvecs, TINY=1.0e-6)
        sbuf=""
        for m in range(nMO):
            sbuf+="%16.8f "%(eOrbs[ min_at ][m])
        print("### %03d %5s %5d %10.2e %s    %12.6f %12.6f %12.6f   %12.6f %12.6f %12.6f"%(\
               I,labels[I],min_at,dist,sbuf, kvI[0],kvI[1],kvI[2],
               kvI[0]/(PIx2/a),kvI[1]/(PIx2/a),kvI[2]/(PIx2/a)), file=fd)

    HOMO=None;SOMO=None;LUMO=None
    if( Occ is not None ):
        kp=0;Ndim=np.shape(Occ)
        if( nKpoints == Ndim[0] and nMO==Ndim[1] ):
            for j in range(nMO):
                idum = int( round(Occ[kp][j]))
                if( idum == 1 and (SOMO is None) ):
                    SOMO=j
                elif( idum == 2 ):
                    HOMO=j
                else:
                    if(LUMO is None):
                        LUMO=j

    I_to_s=[ -1 for I in range(N) ]
    for I in range(N):
        kvI= np.array( namedKpoints[I] )
        min_at,dist,at = find_closest( kvI,kvecs, TINY=1.0e-6)
        j=( at if(at is not None) else min_at )
        if( at is None ):
            print("#!W cannot find:%s:dist=%f"%(labels[I],dist))
        epsI=eOrbs[ min_at ][:]
        if( eLast is not None ):
            kdiff= np.array(kvI) - np.array(kvcLast)
            le=np.sqrt( np.vdot( kvI-kvcLast,kvI-kvcLast))
            Nsect= int(math.ceil(le/b_0))
            b=le/float(Nsect)
            kvc=np.array( kvcLast.copy() )
            r=0.0
            for k in range(Nsect):
                seqno+=1;seqlen+=b;kvc+= kdiff/float(Nsect);r+=1.0/float(Nsect)
                eps = (1.0 - r)*eLast + r*epsI
                strbuf=[];ibuf=[]
                ### eps = interpolate_eorb(kvc, kvecs,eOrbs,BravaisVectors,Description=strbuf,Indices=ibuf)
                if( k==Nsect-1 ):
                    assert len(ibuf)==1,""
                string=""
                for m in range(nMO):
                    string+="%16.8f "%(eps[m])
                print(" %5d %12.6f %12.6f %12.6f   %s    %12.6f %12.6f %12.6f    %12.6f %12.6f %12.6f %s"%(\
                    seqno,I+r,seqlen,seqlen/(PIx2/a), string, kvc[0],kvc[1],kvc[2],
                    kvc[0]/(PIx2/a),kvc[1]/(PIx2/a),kvc[2]/(PIx2/a),(labels[I] if(k==Nsect-1) else strbuf[0]) ),file=fd)
        else:
            sbuf=""
            for m in range(nMO):
                sbuf+="%16.8f "%(epsI[m])
            print(" %5d %12.6f %12.6f %12.6f   %s    %12.6f %12.6f %12.6f    %12.6f %12.6f %12.6f %s"%(\
                seqno,0.0,seqlen,seqlen/(PIx2/a), sbuf, kvI[0],kvI[1],kvI[2],
                kvI[0]/(PIx2/a),kvI[1]/(PIx2/a),kvI[2]/(PIx2/a),labels[0] ),file=fd)
        eLast=np.array(epsI).copy(); kvcLast=np.array( kvI.copy() )
        I_to_s[I]=seqlen
    fd.close()
    if( gnuplot ):
        gnu=fpath.replace(".dat","");os.system("gnuf.sh "+gnu)
        gnuf=gnu+".plt";
        fd=open(gnuf,"a")
        sbuf="set xtics (";dlmt=""
        for I in range(N):
            sbuf+=dlmt+"\"%s\" %f"%(labels[I],I_to_s[I]);dlmt=","
        sbuf+=")"
        print(sbuf,file=fd)
        nPlot=(1 if(HOMO is None) else 2)
        for p in range(nPlot):
            PLOT="plot ";tgtf=fpath;lbl=""
            m0=0;m1=nMO
            if(p==1):
                m0=max(0,HOMO-2);m1=min(nMO,(HOMO+3))
            for m in range(m0,m1):
                lbl=""
                if( p == 1 ):
                    lbl=( "HOMO-%d"%(HOMO-m) if(m<HOMO) else \
                          ("HOMO" if(m==HOMO) else \
                            ("LUMO" if(m==HOMO+1) else \
                              "LUMO-%d"%(m-HOMO-1) )))
                print(PLOT+"\"%s\" index 0:0 using 3:%d title \"%s\" with lines ls %d"%(tgtf,5+m,lbl,1+m),end="",file=fd)
                PLOT=",\\\n";tgtf=""
        fd.close()
        os.system("plot.sh "+gnu)
def xopen(path,mode,Threads=[0]):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    if( MPIsize>1 and (MPIrank not in Threads) ):
        return None
    fd=open(path,mode)
    return fd
def xprint(text, sep=' ', end='\n', file=None, flush=False):
    if( file is None ):
        return
    print(text, sep=sep, end=end, file=file,flush=flush)

def getv(dic,key,default=None):
    if( key in dic ):
        return dic[key]
    return default

def calc_bandstructure(params,job=None,strLatticetype=None):

    HARTREEinEV=PhysicalConstants.HARTREEinEV()

    xyzf=params["xyzf"]
    spinrestriction=params["spinrestriction"]
    xc=params["xc"]
    if( job is None ): job=jobname( name=xyzf.replace(".xyz",""), basis=params["basis"], xc=params["xc"],
                                    nKpoints=params["nKpoints"], spinrestriction=params["spinrestriction"] ) + "_pbcEgnd"
    rttddft_common.set_params(params,job)
    dic={'a':None}
###    strxyz= read_xyzf_to_string(xyzf, dict=dic)
    Rnuc_ANGS,Sy=read_xyzf(xyzf,dict=dic)
    strxyz= write_xyzstring(Rnuc_ANGS, Sy, input_unit='A',output_unit='A')
### somehow this always fails ... 
#  File "/usr/src/opt_pyscf/pyscf/pyscf/gto/mole.py", line 349, in format_atom
#    fmt_atoms = from_zmatrix('\n'.join(fmt_atoms))
#  File "/usr/src/opt_pyscf/pyscf/pyscf/gto/mole.py", line 3595, in from_zmatrix
#    ang   = float(vals[3])/180*numpy.pi
#  IndexError: tuple index out of range
###    cell = pbc_gto.M( a = dic['a'],atom = strxyz, basis = params["basis"])
    cell=pbc_gto.Cell(); cell.a=dic['a'];cell.atom=strxyz; cell.basis=params["basis"]
    
    if( getv(params,"mesh") is not None):
        mesh=getv(params,"mesh")
        if(isinstance(mesh,str)):
            mesh=parse_ints(mesh,',')
        cell.mesh= mesh
    if( getv(params,"exp_to_discard") is not None):
        cell.exp_to_discard= getv(params,"exp_to_discard")
    cell.build()
    nKpoints=parse_ints(params["nKpoints"],',');xprint("nKpoints:"+str(nKpoints))
    xprint("LatticeVectors:"+str(dic['a']))
    kpts=cell.make_kpts( nKpoints )
    ksDFT = pbc_dft.KRKS(cell,kpts,xc=xc) ##.mix_density_fit()
    if( getv(params,"df") != "GDF"):
        assert False,""
    else:
        gdf=pbc_df.GDF(cell); gdf.kpts= kpts;ksDFT.with_df=gdf;
    
    cput1 =time.time();
    Egnd_au = ksDFT.kernel()
    Egnd_eV = Egnd_au * HARTREEinEV
    cput2 =time.time(); tstep=cput2-cput1;
    xfd=xopen(job+".log","w")
    xprint("#Egnd:%16.8f  %16.8f   \t\t %14.4f"%(Egnd_au,Egnd_eV,tstep),file=xfd) 
    if(xfd is not None):
        xfd.close()
    if( strLatticetype is not None):
        plot_dispersion(strLatticetype,ksDFT,job+"_dispersion.dat")
    emin_eV=float( getv(params,"emin_eV","0.0") )
    emax_eV=float( getv(params,"emax_eV","30.0") )
    de_eV=float( getv(params,"de_eV","0.025") )
    calc_ldos(ksDFT, job, emin_eV, emax_eV, de_eV=None, Nstep=None, 
              iappend=0, OrbitalEngs=None, Occupation=None, spinrestriction='R',TINY=1.0e-10,
              widths_eV=[2.7211386024367243, 0.2123, 0.1], gnuplot=True )

def get_orthogonalvecs(vecs,Iref=0, vcnorm_thr=1.0e-6):  ## vecs[Nvec][Ld] 
    Nv=len(vecs);Ld=len(vecs[0]);dtype=np.array(vecs[0]).dtype
    assert dtype==complex or dtype==np.complex128 or dtype==float or dtype==np.float64,""
    ret=[];Nret=0
    vcnorms=[]; Indices=[]
    WKS=np.zeros([Ld],dtype=dtype)
    for Iv in range(Nv):
        if(Iv==Iref):
            continue
        for k in range(Ld):
            WKS[k]= vecs[Iv][k]-vecs[Iref][k]

        for jr in range(Nret):
            ovlp=np.vdot( ret[jr], WKS )
            cof =ovlp/vcnorms[jr]
            if( abs(cof)>1.0e-10 ):
                for k in range(Ld):
                    WKS[k]-=cof*ret[jr][k]
        dum= np.sqrt( ( np.vdot(WKS,WKS) ).real )
        if( dum > vcnorm_thr ):
            ret.append( np.array( [ vecs[Iv][k]-vecs[Iref][k] for k in range(Ld) ] ) )
            vcnorms.append( np.sqrt( (np.vdot(vecs[Iv], vecs[Iv])).real ) )
            Indices.append( Iv )
    return ret,Indices
def plot_dispersion(strLatticetype,mf,fpath,iappend=0, header=None, trailer=None, 
                    OrbitalEngs=None,Occupation=None,spinrestriction='R', **kwargs):
    if (OrbitalEngs is None): OrbitalEngs=mf.mo_energy;
    PI=3.1415926535897932384626433832795
    PIx2=6.283185307179586476925286766559
    sqrt3=1.7320508075688772935274463415059

    BOHRinANGS=PhysicalConstants.BOHRinANGS();

    xfd=xopen(fpath,("a" if(iappend!=0) else "w"))
    if(header is not None):
        xprint(header,file=xfd)
    nmult_MO=(2 if(spinrestriction=='U') else 1)
    if( strLatticetype[0:6]=='planer' ):
        kpts=mf.kpts
        assert len(kpts)>2,""

        xfdDAF=xopen(fpath,"w");
        legend="#%4s    %12s %12s %12s   "%("J","kvec","kvec","kvec")+"   %14s"%("eorb")
        xprint(legend,file=xfdDAF)
        for sp in range(nmult_MO):
            if(nmult_MO>1):
                xprint( ("\n\n\n" if(sp>0) else "") + "#%d:spin=%d"%(sp,sp),file=xfdDAF);
            eOrbs=( OrbitalEngs if(nmult_MO==1) else OrbitalEngs[sp])
            nMO=len(eOrbs)
            ebuf=np.zeros([nMO],dtype=np.float64)
            N=20
            for I in range(3):
                for J in range(N):
                    r=J/float(N)
                    kvec=[ (1.0-r)*kpts[I][jj] + r*kpts[(I+1)%3][jj] for jj in range(3) ]
                    string=" %4d    %12.4f %12.4f %12.4f   "%(J,kvec[0],kvec[1],kvec[2])
                    for mo in range(nMO):
                        ebuf.append( (1.0-r)*eOrbs[I][mo] + r*eOrbs[(I+1)%3][mo])
                    for mo in range(nMO):
                        string+="  %14.6f"%(ebuf[mo])
                    xprint(string,file=xfdDAF);
        if(xfdDAF is not None):
            xfdDAF.close()
        ### os.system("ls -ltrh "+fpath)
    if( strLatticetype[0:3]=='hex' ):
    
        latticeVecs=mf.cell.a; 
        latticeVecNorms=[ np.sqrt( latticeVecs[J][0]**2 + latticeVecs[J][1]**2 + latticeVecs[J][2]**2) for J in range(2) ]
        ### this should be rCC*sqrt(3) .. 
        rCC_ANGS =latticeVecNorms[0]/sqrt3
        xprint("#LatticeVecNorm:%14.6f rCC:%14.6f"%(latticeVecNorms[0],rCC_ANGS))
        rCC_ANGS=1.42
        rCC=rCC_ANGS/BOHRinANGS
        kpREF=0
        ### rCC=float(strLatticetype[3:])
        v_Gamma=[ 0.0, 0.0, 0.0]
        v_K=[ PIx2/(3*rCC), PIx2/(3.0*sqrt3*rCC), 0.0]
        v_M=[ PIx2/(3*rCC), 0.0, 0.0 ]
        vcs_01=[ v_Gamma,v_K,v_M,v_Gamma ]
        kpts=mf.kpts
        xprint("#Kpoints:"+str(kpts))
        xprint("#Gam,K,M points:"+str([v_Gamma,v_K,v_M]))
        Nkpts=len(kpts)
        
        ## assuming [0] to be the zero-vecto
        bset, kp_bset=get_orthogonalvecs(kpts);Nb=len(bset)
        norms=[ np.sqrt( (np.vdot(bset[jb],bset[jb])).real ) for jb in range(Nb) ]
        cofs=np.zeros([Nb],dtype=np.float64)
        xfdDAF=xopen(fpath,"w");
        legend="#%4s    %12s %12s %12s   "%("J","kvec","kvec","kvec")+"   %14s"%("eorb")
        xprint(legend,file=xfdDAF)
        for sp in range(nmult_MO):
            if(nmult_MO>1):
                xprint( ("\n\n\n" if(sp>0) else "") + "#%d:spin=%d"%(sp,sp),file=xfdDAF);
            eOrbs=( OrbitalEngs if(nmult_MO==1) else OrbitalEngs[sp])
            nMO=len(eOrbs)
            ebuf=np.zeros([nMO],dtype=np.float64)
            N=20
            Ld=len(vcs_01[0])
            for I in range(3):
                for J in range(N):
                    r=float(J)/float(N)
                    v=[ vcs_01[I][k]*(1.0-r) + vcs_01[I+1][k]*r  for k in range(Ld) ]
                    for jb in range(Nb):
                        cofs[jb] = np.vdot(bset[jb],v)/norms[jb]
                    for k in range(Ld):
                        v[k] -= cofs[jb]*bset[jb][k]
                    for mo in range(nMO):
                        ebuf[mo]=eOrbs[kpREF][mo]
                        for jb in range(Nb):
                            ebuf[mo]+= cofs[jb]*( eOrbs[ kp_bset[jb] ][mo] - eOrbs[kpREF][mo] )
                    string=" %4d    %12.4f %12.4f %12.4f   "%(J,
                        vcs_01[I][0]*(1.0-r) + vcs_01[I+1][0]*r,
                        vcs_01[I][1]*(1.0-r) + vcs_01[I+1][1]*r,
                        vcs_01[I][2]*(1.0-r) + vcs_01[I+1][2]*r)
                    for mo in range(nMO):
                        string+="  %14.6f"%(ebuf[mo])
                    xprint(string,file=xfdDAF);
        if(xfdDAF is not None):
            xfdDAF.close()
        os.system("ls -ltrh "+fpath)
    if( strLatticetype=='X' or strLatticetype=='Y' or strLatticetype=='Z'):
        # 1D dispersion
        dir=( 0 if(strLatticetype=='X') else ( 1 if(strLatticetype=='Y') else (2 if(strLatticetype=='Z') else None)))
        kvectors=np.reshape( mf.kpts, (-1,3))
        nkpt=len(kvectors)
        kzA=[]
        for kp in range(nkpt):
            kzA.append(kvectors[kp][dir])
        ith_to_kp=np.argsort(kzA)
        kzmin=kzA[ith_to_kp[0]];kzmax=kzA[ith_to_kp[nkpt-1]]
        kz0=( kzmin if("kz0" not in kwargs) else kwargs["kz0"] )
        kz1=( kzmax if("kz1" not in kwargs) else kwargs["kz1"] )
        Ndiv=( 100 if("Ndiv" not in kwargs) else kwargs["Ndiv"] );
        
        dkz=(kz1-kz0)/float(Ndiv)
        dk_tiny =0.02*dkz
        for J in range(Ndiv+1):
            kz=kz0 + dkz*J
            string="15.6f  \t\t"%(kz)
            ithLWR=None;ithUPR=None;ithAT=None
            for ith in range(nkpt):
                if( abs(kz-kzA[ ith_to_kp[ith] ])<dk_tiny ):
                    ithAT=ith;break
                elif(kz < kzA[ ith_to_kp[ith] ]):
                    ithUPR=ith;break
                ithLWR=ith;continue

            for sp in range(nmult_MO):
                eOrbs=( OrbitalEngs if(nmult_MO==1) else OrbitalEngs[sp])
                nMO=len(eOrbs)
                engs=None
                if(ithAT is not None):
                    kp=ith_to_kp[ithAT]
                    engs =eOrbs[kp]

                elif( ithLWR is not None and ithUPR is not None):
                    kzLWR=kvectors[ithLWR][dir];kpLWR=ith_to_kp[ithLWR]
                    kzUPR=kvectors[ithUPR][dir];kpUPR=ith_to_kp[ithUPR]
                    facLWR=(kzUPR-kz)/(kzUPR-kzLWR)
                    facUPR=1.0-facLWR
                    engs =[ facLWR*eOrbs[kpLWR][mo] for mo in range(nMO) ]
                else:
                    assert False,""
                for mo in range(nMO):
                    string+="  %14.6f"%(engs[mo])
            xprint(string,file=xfd)
    if(trailer is not None):
        xprint(trailer,file=xfd)
    if(xfd is not None):
        xfd.close()
    return iappend+1
def sample_var(cum,sqrcum,count):
    if(count==0):
        return 0.0
    av=cum/float(count);var=sqrcum/count - av*av
    if( count==1 ):
        return var
    var*=float(count)/float(count-1)
    return var

def check_walltime(key,wtm,dic1,dic2,ope_PorR="R",bfsz=0,fpath=None,Append=True,description=""):
    if(ope_PorR=="R"):
        if(key not in dic1):
            dic1.update({key:wtm})
        else:
            dic1[key]+=wtm

        if( key not in dic2 ):
            dic2.update({key:{"count":0,"sum":0.0,"sqrsum":0.0,"buf":[]}} if(bfsz>0) else \
                        {key:{"count":0,"sum":0.0,"sqrsum":0.0}})
        dic2[key]["count"]+=1
        dic2[key]["sum"]+=wtm
        dic2[key]["sqrsum"]+=wtm**2
        if( bfsz > 0 ):
            while( len(dic2[key]["buf"])>=bfsz ):
                dic2[key]["buf"].pop(0)
            dic2[key]["buf"].append(wtm)
    
    elif(ope_PorR=="P"):
        ret1="";ret2=""
        delimiter="    "
        ret1=delimiter.join( [ "%s:%11.3f"%(key,dic1[key]) for key in dic1 ] )
        
        ## count=min( [ dic2[key]["count"] for key in dic2 ] )
        ## ret2=delimiter.join( [ "%s:%11.3f \pm %11.2e n=%d"%( key, dic2[key]["sum"]/float(dic2[key]["count"]),
        ##                        np.sqrt(abs(sample_var(dic2[key]["sum"],dic2[key]["sqrsum"],dic2[key]["count"]))),dic2[key]["count"]) for key in dic2 ])
        fdOU=(None if(fpath is None) else open(fpath,("a" if(Append) else "w")))
        fd=(sys.stdout if(fdOU is None) else fdOU)
        print("%s %s"%(description,ret1),file=fd);
        ## print("## avg over %d:%s"%(count,ret2),file=fd)
        if(fdOU is not None):
            fdOU.close()
## note: FWHM=0.5eV corresponds to width_eV=0.2123
##       0.5*( (FwHM/2)**2 / wid**2 ) = ln(2)
## FILENAME _ldos.dat, _eorbs.dat etc.
def calc_ldos(mf, filename, MO_coefs, FockMat, emin_eV=None, emax_eV=None, de_eV=None, Nstep=None, 
              moldyn=None,
              iappend=0, header=None, trailer=None, OrbitalEngs=None, Occupation=None, spinrestriction='R',TINY=1.0e-10,
              widths_eV=[0.2123, 2.7211386024367243, 0.1, 0.01], gnuplot=True, N_elec=None, margin_eV=None, Threads=[0] ):
    if (Occupation is None): Occupation=mf.mo_occ;
    if (moldyn is not None): Occupation=moldyn.mo_occ
    nmult_MO=(2 if(spinrestriction=='U') else 1)
    nmult_Occ=(2 if(spinrestriction!='R') else 1)
    nmult_Fock=(2 if(spinrestriction=='U') else 1)
    nmult_eorb=(2 if(spinrestriction=='U') else 1)
    pbc=isinstance( mf.mol, Cell)

    fnm1=filename.strip();le1=len(fnm1)
    if(fnm1.endswith(".dat")):fnm1=fnm1[:le1-4]
    le1=len(fnm1)
    for ky in ["_ldos","_lDOS"]:
        if(fnm1.endswith(ky)):
            fnm1=fnm1[:le1-5]
    fnm_pDOS=fnm1+"_pDOS.dat";
    MO_engs=OrbitalEngs
    if(MO_engs is None):
        MO_engs=calc_eorbs(mf, MO_coefs, FockMat, spinrestriction)
    calc_pDOS(mf,fnm_pDOS,MO_engs, MO_coefs, Occupation )
    
    #           UKS                    ROKS                RKS
    # MOcofs    [2]([nKpt])[nAO][nMO]  ([nKpt])[nAO][nMO]  ([nKpt])[nAO][nMO]
    # Occ       [2]([nKpt])[nMO]       [2]([nKpt])[nMO]    ([nKpt])[nMO]
    # Fock      [2]([nKpt])[nAO][nAO]  ([nKpt])[nAO][nAO]  ([nKpt])[nAO][nAO]
    # eOrbs     [2]([nKpt])[nMO]       ([nKpt])[nMO]       ([nKpt])[nMO]
    if( nmult_MO > 1 or nmult_Occ > 1 ):
        FermiLv_au=( 0.0 if(not pbc) else ( mf.get_fermi() if(moldyn is None ) else \
                     get_Fermilvl( moldyn,mf,FockMat=FockMat,MO_coefs=MO_coefs,MO_occ=Occupation, eOrbs=OrbitalEngs)) )
# 2022.01.09 : rttddft generated by Moldyn.update_Rnuc does NOT have mf.mo_coefs NOR mf.mo_occ and causes trouble if you use mf.get_fermi()
#        FermiLv_au=(0.0 if( not pbc ) else mf.get_fermi() )
        assert nmult_MO==2,""
        strUpDn=[ "_upSpin","_dnSpin" ]
        for sp in range(nmult_MO):
            fermilv_au =( FermiLv_au if( isScalarNumber(FermiLv_au) ) else FermiLv_au[sp] )

            fockmat    =( None if(FockMat is None) else ( FockMat  if(nmult_Fock==1) else FockMat[sp]  ) )
            mo_coefs   =( None if(MO_coefs is None) else ( MO_coefs if(nmult_MO==1)   else MO_coefs[sp] ) )
            occupation =( None if(Occupation is None) else ( Occupation if(nmult_Occ==1) else Occupation[sp] ) )
            orbitalengs=( None if(OrbitalEngs is None) else ( OrbitalEngs if(nmult_eorb==1) else OrbitalEngs[sp] ) )
            calc_ldos1( 1, 1, 1, 1, mf, filename+strUpDn[sp], mo_coefs, fockmat, emin_eV=emin_eV, emax_eV=emax_eV, de_eV=de_eV, Nstep=Nstep, 
              iappend=iappend, header=header, trailer=trailer, OrbitalEngs=orbitalengs, Occupation=occupation, TINY=TINY,
              widths_eV=widths_eV, gnuplot=gnuplot, N_elec=N_elec, margin_eV=margin_eV, Threads=Threads, FermiLv_au=fermilv_au )
        fermilv_au =( FermiLv_au if( isScalarNumber(FermiLv_au) ) else FermiLv_au[0] )
        return calc_ldos1( nmult_MO, nmult_Occ, nmult_Fock, nmult_eorb, mf, filename, MO_coefs, FockMat, emin_eV=emin_eV, emax_eV=emax_eV, de_eV=de_eV, Nstep=Nstep, 
          iappend=iappend, header=header, trailer=trailer, OrbitalEngs=OrbitalEngs, Occupation=Occupation, TINY=TINY,
          widths_eV=widths_eV, gnuplot=gnuplot, N_elec=N_elec, margin_eV=margin_eV, Threads=Threads, FermiLv_au=fermilv_au )
        
    else:
        print("#calc_ldos:check mf.mo_energy:",(mf.mo_energy is not None))
        if(  mf.mo_energy is None ):
            eorbs=[]
            nKpoints=(1 if(not pbc) else len(FockMat))
            for kp in range(nKpoints):
                Fock=(FockMat if(not pbc) else FockMat[kp])
                MOs=(MO_coefs if(not pbc) else MO_coefs[kp])
                ndim_MOs=np.shape(MOs);nMO=ndim_MOs[1]
                eps=np.zeros([ nMO ],dtype=np.complex128 )
                for mo in range(nMO):
                    eps[mo]=np.vdot( MOs[:,mo], np.matmul( Fock, MOs[:,mo] ))
                if( not pbc ):
                    eorbs=eps
                else:
                    eorbs.append(eps)
            mf.mo_energy=eorbs
        print("#calc_ldos:check mf.mo_energy:",(mf.mo_energy is not None))
        FermiLv_au=(0.0 if( not pbc ) else ( mf.get_fermi() if(moldyn is None) else \
                                             get_Fermilvl( moldyn,mf,FockMat=FockMat,MO_coefs=MO_coefs,MO_occ=Occupation, eOrbs=OrbitalEngs) ))
        return calc_ldos1( nmult_MO, nmult_Occ, nmult_Fock, nmult_eorb, mf, filename, MO_coefs, FockMat, emin_eV=emin_eV, emax_eV=emax_eV, de_eV=de_eV, Nstep=Nstep, 
          iappend=iappend, header=header, trailer=trailer, OrbitalEngs=OrbitalEngs, Occupation=Occupation, TINY=TINY,
          widths_eV=widths_eV, gnuplot=gnuplot, N_elec=N_elec, margin_eV=margin_eV, Threads=Threads, FermiLv_au=FermiLv_au )
    # print("check FermiLv..") # (-0.21302746781232837, -0.21302746781232834)  for UKS, fermilv is an array

def calc_eorbs(mf, MO_coefs, FockMat, spinrestriction):
    pbc=isinstance( mf.mol, Cell)
    nmult_MO=(2 if(spinrestriction=='U') else 1)
    
    ret=[]
    for sp in range(nmult_MO):
        MOcofs=( MO_coefs if(nmult_MO==1) else MO_coefs[sp])
        fockmat=( FockMat if(nmult_MO==1) else FockMat[sp])
        ebuf_sp=[]
        nkpt=( 1 if(not pbc) else len(MOcofs) )
        for kp in range(nkpt):
            MOs=( MOcofs if(not pbc) else MOcofs[kp])
            fck=( fockmat if(not pbc) else fockmat[kp])
            nMO=len(MOs)
            engs=[ np.vdot( MOs[:,el], np.matmul( fck, MOs[:,el] ) ).real for el in range(nMO) ]
            if(pbc):
                ebuf_sp.append(engs)
            else:
                ebuf_sp=engs
        if( nmult_MO == 1 ):
            ret=ebuf_sp
        else:
            ret.append(ebuf_sp)
    return np.array(ret)

def calc_ldos1(nmult_MO, nmult_Occ, nmult_Fock, nmult_eorb, mf, filename, MO_coefs, FockMat, emin_eV=None, emax_eV=None, de_eV=None, Nstep=None, 
              iappend=0, header=None, trailer=None, OrbitalEngs=None, Occupation=None, TINY=1.0e-10,
              widths_eV=[0.2123, 2.7211386024367243, 0.1, 0.01], gnuplot=True, N_elec=None, margin_eV=None, Threads=[0], FermiLv_au=0.0 ):

    HARTREEinEV=PhysicalConstants.HARTREEinEV();
    
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    if( MPIsize>1 and (MPIrank not in Threads) ):
        return iappend

    ln_tiny=math.log(TINY)
    pbc=isinstance( mf.mol, Cell)
    nkpt=None
    ebuf=None;wbuf=None;nbuf=0;ibuf=None;bfsz=0;nbuf_s=[0,0]
    ebuf_s=None;wbuf_s=None;ibuf_s=None;nbuf_s=[0,0]
    for sp in range(nmult_MO):
        MOcofs=(MO_coefs if(nmult_MO==1) else MO_coefs[sp])
        fockmat=(FockMat if(nmult_MO==1) else FockMat[sp])
        ### eorbs=(OrbitalEngs if(nmult_eorbs==1) else OrbitalEngs[sp])
        orbocc=(Occupation if(nmult_Occ==1) else Occupation[sp])
        Ndim_MO=np.shape(MOcofs)  # [nAO,nMO] or [nKpt,nAO,nMO]
        Ndim_occ=np.shape(orbocc); ### assert i1eqb(Ndim_eorbs,Ndim_occ),""
        nkpt=(1 if(not pbc) else Ndim_MO[0])
        nMO =(Ndim_MO[1] if(not pbc) else Ndim_MO[2])
        if(ebuf is None):
            bfsz=nkpt*nMO*nmult_MO
            ebuf=np.zeros([bfsz],dtype=np.float64); wbuf=np.zeros([bfsz],dtype=np.float64);
            ibuf=np.zeros([bfsz,3],dtype=int)
            if( nmult_MO > 1 ):
                ebuf_s=np.zeros([nmult_MO,nkpt*nMO ],dtype=np.float64); 
                wbuf_s=np.zeros([nmult_MO,nkpt*nMO],dtype=np.float64);
                ibuf_s=np.zeros([nmult_MO,nkpt*nMO,3],dtype=int)
                
        for kp in range(nkpt):
            MOs=(MOcofs if(not pbc) else MOcofs[kp])
            fck=(fockmat if(not pbc) else fockmat[kp])
            
            occ=(orbocc if(not pbc) else orbocc[kp])
            for el in range(nMO):
                en= np.vdot( MOs[:,el], np.matmul( fck, MOs[:,el] ) ).real
                ebuf[nbuf]=en; wbuf[nbuf]=occ[el]; ibuf[nbuf]=[sp,kp,el]; nbuf+=1
                if( nmult_MO > 1 ):
                    ebuf_s[ sp ][ nbuf_s[sp] ]= en; 
                    wbuf_s[ sp ][ nbuf_s[sp] ]= occ[el]; 
                    ibuf_s[ sp ][ nbuf_s[sp] ]= [sp,kp,el]; nbuf_s[sp]+=1
    assert nbuf==bfsz,""
    ithTOjs= np.argsort(ebuf)
    e_eV_sorted=np.zeros([bfsz],dtype=np.float64); 
    w_sorted=np.zeros([bfsz],dtype=np.float64);
    i_sorted=np.zeros([bfsz,3],dtype=int)
    for ith in range(bfsz):
        js=ithTOjs[ith];e_eV_sorted[ith]=ebuf[js]*HARTREEinEV;
        w_sorted[ith]=wbuf[js];i_sorted[ith]=ibuf[js]
    if(pbc and nkpt>1):
        fac=1.0/float(nkpt)
        for ith in range(bfsz):
            w_sorted[ith]=w_sorted[ith]*fac
    eorb_min_eV=e_eV_sorted[0]
    eorb_max_eV=e_eV_sorted[bfsz-1]


    daf_eorbs=filename+"_eorbs.dat"
    xfd=xopen(daf_eorbs,("a" if( iappend!=0 ) else "w"))
    if( header is not None):
        xprint(header,file=xfd);
    xprint("#%19s  %20s  %5s %5s %5s"%("eorb_eV","weight","sp","kp","mo"),file=xfd) 
    for ith in range(bfsz):
        xprint("%20.10f  %20.10f  %5d %5d %5d"%(e_eV_sorted[ith],w_sorted[ith], 
                     i_sorted[ith][0],i_sorted[ith][1],i_sorted[ith][2]),file=xfd)
    if(trailer is not None):
        xprint(trailer,file=xfd)
    if( xfd is not None):
        xfd.close()

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
    xfd=xopen( daf_ldos,("a" if(iappend!=0) else "w"))
    if( header is not None):
        xprint(header,file=xfd);
    e_eV=emin_eV
    wgt_ref=(1.0 if(nmult_Occ==2) else 2.0)/(1.0 if(not pbc) else float(nkpt))
    wgsum = sum(w_sorted);
    printout("#calc_ldos:wgsum:%f"%(wgsum))
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
            xprint(string,file=xfd)

        string="%15.6f   "%(e_eV)
        for kwid in range(Nwid):
            string+="       %16.8f  %16.8f"%(fn[kwid],fn_occ[kwid])
        xprint(string,file=xfd)
    string="";
    for kwid in range(Nwid):
        cum[kwid]*=de_eV;cum_occ[kwid]*=de_eV
        string+="       %9.4f %9.4f    "%(cum[kwid],cum_occ[kwid])
        xprint("#sum:"+string,file=xfd)
    if(trailer is not None):
        xprint(trailer,file=xfd)
    if(xfd is not None):
        xfd.close()
    
    FermiLv_eV=FermiLv_au*HARTREEinEV
    if( gnuplot ):
        xfdgnu=xopen(filename+"_ldos.plt","w");
        xprint("set term postscript color\nset output \"%s.ps\"\nload \"C:/cygwin64/usr/local/bin/stdcolor.plt\""%(filename+"_ldos"),
                file=xfdgnu)
# 1 : 2,3   4,5 ...
        xprint("EF=%f"%(FermiLv_eV),file=xfdgnu)
        xprint("set xlabel \"E (eV)\"",file=xfdgnu)
        for kwid in range(Nwid):
            xprint("set title \"width=%9.4f eV\""%(widths_eV[kwid]),file=xfdgnu);
            xprint("##occsum=%7.3f"%(cum_occ[kwid]),file=xfdgnu)
            xprint('plot "'+daf_ldos+"\" using ($1-EF):%d title \"\" with lines ls 1,\\"%( 2*kwid+2 ),file=xfdgnu)
            xprint("\"\" using ($1-EF):%d title \"\" with lines ls 108"%( 2*kwid+3 ),file=xfdgnu)
            if( kwid + 1 == Nwid ):
                xprint("set xrange [-15:15]",file=xfdgnu)
                xprint("replot",file=xfdgnu)
        if(xfdgnu is not None):
            xfdgnu.close()
    return iappend+1


def read_eorbs(path):
    fdIN=open(path,"r")
    kp=None;kvec=None;eOrbs=None;kVectors=[]
    for line in fdIN:
        line=line.strip();le=len(line);blank=0
        if(le<1):
            blank+=1;continue
        if( line[0]=='#' ):
            if( line[1]=='#'):
                sA=line.split()
                for col in [ sA[1],sA[2] ]:
                    arr=col.split(":");
                    if(arr[0]=="nkpt"):
                        nkpt=int(arr[1])
                    elif(arr[0]=="nMO"):
                        nMO=int(arr[1])
                    else:
                        assert False,""
                if( (nkpt is not None) and (nMO is not None) ):
                    eOrbs=np.zeros([ nkpt,nMO ],dtype=np.float64)
                print("#reading off:"+line)
                continue
            sA=line.split()
            sdum=sA[0];
            kp=int(sdum[1:])
            kvec=[ float(sA[k]) for k in range(1,4) ]
            kVectors.append(kvec)
            assert len(kVectors)==(kp+1),""
            continue
        else:
            sA=line.split()
            jmo=int(sA[0].strip())
            eps=float(sA[1].strip())
            eOrbs[kp][jmo]=eps
    return eOrbs,kVectors
def print_eOrbs(md,rttddft,Orbital_Energies,fnme=None,description="",step=None):
    if( step is None ):
        step=md.step
    nmult_eOrbs=(2 if(md.spinrestriction=="U") else 1)
    kvcs=(None if(not md.pbc) else np.reshape( rttddft.kpts, (-1,3)))
    for sp in range(nmult_eOrbs): 
        if(fnme is None):
            fnme=("" if(rttddft_common.job is None) else rttddft_common.job)+"_step%06d"%(step)
        path=fnme +("" if(nmult_eOrbs==1) else "_sp%d"%(sp))+"_print_eOrbs.dat"
        fdOUT=open(path,"w")
        eOrbs = (Orbital_Energies if(nmult_eOrbs==1) else Orbital_Energies[sp])
        Ndim=np.shape(eOrbs);
        nkpt=Ndim[0];nMO=Ndim[1]
        print("### nkpt:%d nMO:%d %s %s %s"%(nkpt,nMO,description,str(rttddft_common.get_job(True)),str(datetime.datetime.now())),
               file=fdOUT)
        for kp in range(nkpt):
            print("#%03d %14.6f %14.6f %14.6f"%(kp, kvcs[kp][0],kvcs[kp][1],kvcs[kp][2]),file=fdOUT)
            for j in range(nMO):
                print(" %5d %16.8f"%(j,eOrbs[kp][j]),file=fdOUT)
            print("\n\n",file=fdOUT)
        fdOUT.close()

def dist_from_edge(X,A,B):
    t,h=calc_perpendicular_leg(X,A,B)
    return h

def calc_perpendicular_leg(X,A,B):
    x=np.array(X);a=np.array(A);b=np.array(B)
    v=b-a; vSQR=np.vdot(v,v)
    hSQR = np.vdot(x-b,x-b) - ( np.vdot(x-b, v)**2 )/vSQR
    if( hSQR < 0 ):
        assert abs(hSQR)<1e-7,"hSQR:%e %e-%e"%(hSQR,np.vdot(x-b,x-b),np.vdot(x-b,v)**2/vSQR)
        hSQR=abs(hSQR)
    h=np.sqrt(hSQR)
    t=-np.vdot( x-b,v)/vSQR
    return t,h
        
def vecnorm(vec):
    v=np.array(vec)
    return np.sqrt( np.vdot(v,v) )

def modify_cell(src,mf=None,mesh=None,kpts=None,nKpoints=None):
    from pyscf.pbc import gto
    if(mesh is None):
        mesh=src.mesh
    else:
        print("#extending mesh:",src.mesh,">>",mesh)
    ret = gto.Cell()
    ret.mesh= mesh
    ret.rcut=src.rcut
    ret.exp_to_discard = src.exp_to_discard
    ret.dimension = src.dimension
    ret.charge=src.charge
    ret.spin=src.spin
    ret.precision=src.precision
    ret.atom = src.atom.copy()
    ret.basis = src.basis
    ret.pseudo = src.pseudo
    ret.a = src.a.copy()  ### this should be an intrinsic property so I leave it .. but let's use .BravaisVectors_au for our calculations
    # ret.BravaisVectors_au = src.BravaisVectors_au
    if( kpts is None ):
        if( nKpoints is not None ):
            kpts=ret.make_kpts(nKpoints)
        elif( mf is not None ):
            kpts=mf.kpts
    ret.build()
    dbgng=True
    if( dbgng ):
        fd1=open("modify_cell_SRC.log","w");print(serialize(src),file=fd1);fd1.close()
        fd2=open("modify_cell_DST.log","w");
        print("#",mesh,kpts,nKpoints,file=fd2)
        print(serialize(ret),file=fd2);fd2.close()
        os.system("fopen modify_cell_SRC.log");os.system("fopen modify_cell_DST.log");
    return ret
# 
# Here you can apply BravaisVectors of A SINGLE CELL (<=> your supercell) to plot larger BZ
# 
def plot_dispersion_kLine(fpath,mf,dmat,BravaisVectors,lattice=None,NamedKpoints=None,Append=False):
    bravaisvectors=BravaisVectors;description=None
    dum=rttddft_common.Params_get("dimerization")
    if( dum is not None ):
        sA=dum.split(",");nA=len(sA);nlm=[ int(sA[k]) for k in range(nA) ]
        bravaisvectors=np.array([ np.array( BravaisVectors[0] )/nlm[0],\
                              np.array( BravaisVectors[1] )/nlm[1],\
                              np.array( BravaisVectors[1] )/nlm[2] ])
        description=" dimerization:"+str(nlm)
    if( NamedKpoints is None ):
        assert lattice is not None,""
        if( lattice in [ "NaCl", "diamond" ] ):
            labels=["W","L","gamma","X","W","K"]
            NamedKpoints=get_named_Kpoints(lattice,labels, bravaisvectors)
        elif( lattice in ["hexagonal","hcp"] ):
            labels=["K","M","gamma","K"]
            if( mf.cell.dimension == 3 ):
                labels=["gamma","M","K","gamma","A","L","K","A"]
            NamedKpoints=get_named_Kpoints(lattice,labels, bravaisvectors)
        else:
            assert False,""
    plot_dispersion_kLine_(fpath,mf,dmat,NamedKpoints,labels,Append=Append,description=description)
##
## You might rebuils cell with larger meshsize..
##
def plot_dispersion_kLine_(fpath,mf,dmat,NamedKpoints,labels,dk_refr=None,cell=None,mesh=None,Append=False, 
                           description=None):
    
    if( cell is None ):
        if(mesh is not None):
            cell=modify_cell( mf.cell, mf=mf, mesh=mesh)
        else:
            cell=mf.cell

    kpts_band=[]
    Np=len(NamedKpoints)
    # eg. 4 NamedKpoints 0,1,2,3 ----
    # 1  0.0, 0.1, 0.2 ... 0.9
    # 2  1.0, ...          1.9
    # 3                    2.9  3.0
    Nkpts_array=[]
    Dbgng=True

    Nkpts_UPL=None
    Nkpts_UPL=rttddft_common.Params_get("Nkpts_UPL")
    if(Nkpts_UPL is not None ):
        Nkpts_UPL=int(Nkpts_UPL)
        
    for I in range(1,Np):
        Kv_0= np.array(NamedKpoints[I-1]); Kv_1=np.array(NamedKpoints[I])
        Delta_K = Kv_1- Kv_0
        len_Delta_K= vecnorm(Delta_K)
        if( dk_refr is None ):
            dk_refr=len_Delta_K/20.0
        Ndiv = int( np.ceil( len_Delta_K/dk_refr ) ); Ndiv=max(1,Ndiv)
        Nkpts_array.append(Ndiv)
        dK = Delta_K*(1.0/float(Ndiv))
        Kv=Kv_0
        for J in range(Ndiv):
            kpts_band.append( np.array( [ Kv[0],Kv[1],Kv[2] ] ) )
            Kv += dK
        dum=vecnorm( Kv - Kv_1);assert dum<1e-6,""
        if( I+1 == Np ):
            kpts_band.append( np.array( [ Kv[0],Kv[1],Kv[2] ] ) )
    Nkpts_Kline=len(kpts_band)
    if(Dbgng):
        kvecs=np.reshape( mf.kpts, (-1,3))
        nkvcs=len(kvecs)
        for kp in range(nkvcs):
            kpts_band.append( kvecs[kp] )

    Nkpts_band=len(kpts_band)
    kpts_DM = np.reshape( mf.kpts, (-1,3))
    Nkpts_DM = len(kpts_DM)

    # 2022.02.15 ... we divide kpts_band into subarrays
    # hereafter we replace  kpts_band_I 
    
    nBlock=(1 if(Nkpts_UPL is None) else \
              (1 if(Nkpts_band<=Nkpts_UPL) else int( math.ceil( Nkpts_band/float(Nkpts_UPL)) )) )
    kpt_offset=0
    eig_kpts = []
    cofs=[]
    wt0=time.time();wt1=wt0; wt_hc=[];wt_veff=[];wt_eig=[]
    for ib in range(nBlock):
        kpts_band_I= kpts_band;
        if( nBlock > 1 ):
            kpts_band_I = [ kpts_band[k] for k in range(ib*Nkpts_UPL, min(Nkpts_band, (ib+1)*Nkpts_UPL)) ]
        Nkpts_band_I = len(kpts_band_I)
        if( Nkpts_band_I < 1 ):
            continue  ##... never come here ...
                     
        Hc=mf.get_hcore(cell_or_mol=mf.cell,kpts=kpts_band_I); wt2=wt1;wt1=time.time();wt_hc.append( wt1-wt2 );
        assert len(Hc)==Nkpts_band_I,""
        Veff=mf.get_veff(cell=mf.cell,kpts_band=kpts_band_I); wt2=wt1;wt1=time.time();wt_veff.append( wt1-wt2 );
        assert len(Veff)==Nkpts_band_I,""

        nMO=None
        Hmlt = Hc + Veff
        # S1e=[]
        for k in range(Nkpts_band_I):
            ## @see khf.py#eig
            S_k=mf.get_ovlp( cell, kpts_band_I[k]);# S1e.append(S_k)
            e, c = mf._eigh(Hmlt[k], S_k)
            if( nMO is None ):
                nMO=len(e)
            else:
                nMO=min( len(e),nMO)
            eig_kpts.append(e); 
            cofs.append(c)
        wt2=wt1;wt1=time.time();wt_eig.append(wt1-wt2);
    wt1=time.time();wt_sum=wt1-wt0;
    fd1=open("plot_dispersion_kLine_walltime.log","a");
    print("Nkpt:%5d nBlock:%d walltime:%f (hc:%f,veff:%f,eig:%f)"%(Nkpts_band,nBlock,wt_sum,sum(wt_hc),sum(wt_veff),sum(wt_eig))\
          +("" if(nBlock==1) else str(wt_hc)+str(wt_veff)+str(wt_eig))+"\t\t"+str(rttddft_common.get_job(True)),file=fd1)
    fd1.close();os.system("fopen plot_dispersion_kLine_walltime.log")
    if(Dbgng):
        fnmeDBG="plot_dispersion_kLine.log"; fdDBG=open(fnmeDBG,"w")
        for I in range(Nkpts_DM):
            S_kI=mf.get_ovlp( cell, kpts_band[ Nkpts_Kline + I ]);
            engs_kI=np.array( eig_kpts[ Nkpts_Kline + I ] )
            cofs_kI=np.array( cofs[ Nkpts_Kline + I ] )
            eREF_kI=np.array( mf.mo_energy[I] )
            cREF_kI=np.array( mf.mo_coeff[I] )
            ediff=max( abs( engs_kI - eREF_kI ) )
            cdiff=max( abs( np.ravel(cofs_kI) - np.ravel(cREF_kI) ))
            covlp=np.vdot( cREF_kI, np.matmul( S_kI, cofs_kI))
            covl_II= np.vdot( cofs_kI,np.matmul( S_kI, cofs_kI))
            covl_REF=np.vdot( cREF_kI, np.matmul( S_kI,cREF_kI))
            print("#calc_kLine:kp:%03d: ediff:%16.6e, cdiff:%16.6e |covlp|:%f (%f+j%f) / II (%f+j%f), refI,refI (%f+j%f)"%(\
                  I,ediff,cdiff,abs(covlp),covlp.real,covlp.imag, covl_II.real,covl_II.imag, covl_REF.real,covl_REF.imag),file=fdDBG)

            if( nBlock==1 and (ediff > 1e-6 or cdiff > 1e-6) ):
                ebuf=[]
                nMO=len(engs_kI)
                for ell in range(nMO):
                    eps=np.vdot( cREF_kI[ell],np.matmul(Hmlt[ Nkpts_Kline + I ],cREF_kI[ell] ) )
                    ebuf.append(eps)
                ediff1=max( abs( np.array(eps) - engs_kI))
                ediff2=max( abs( np.array(eps) - eREF_kI))
                print("#calc_kLine:kp:%03d: diff(<i|H|i>,new_eig):%16.6e, diff(<i|H|i>,canonical eps):%16.6e"%(I,ediff1,ediff2),file=fdDBG)
        fdDBG.close()
        os.system("fopen "+fnmeDBG);
    fdDAF=open(fpath,("a" if(Append) else "w"))
    gnu=fpath.replace(".dat","");os.system("gnuf.sh "+gnu); gnuf=gnu+".plt"
    fdGNU=open(gnuf,"a");
    if( description is not None ):
        print("set title \"dispersion along kline, %s\"\n"%(description),file=fdGNU)
    else:
        print("set title \"dispersion along kline\"\n",file=fdGNU)

    kl=0;s=0.0;I_to_s=[ 0.0 ]
    for I in range(1,Np):
        Kv_0= np.array(NamedKpoints[I-1]); Kv_1=np.array(NamedKpoints[I])
        Delta_K = Kv_1- Kv_0
        len_Delta_K= vecnorm(Delta_K)
        Ndiv=Nkpts_array[I-1]
        dk_len=len_Delta_K/float(Ndiv);dK=Delta_K*(1.0/float(Ndiv))
        Kv=Kv_0
        for J in range(Ndiv):
            ## this starts from 0
            print(" %5d  %14.6f  %12.6f %12.6f %12.6f      %s"%(kl+J,s,Kv[0],Kv[1],Kv[2],
                   d1toa(eig_kpts[kl+J])),file=fdDAF)
            Kv+=dK;s+=dk_len;
        if( I+1 == Np ):
            print(" %5d  %14.6f  %12.6f %12.6f %12.6f      %s"%(kl+Ndiv,s,Kv[0],Kv[1],Kv[2],
                   d1toa(eig_kpts[kl+Ndiv])),file=fdDAF)
        I_to_s.append(s);kl+=Ndiv
    sbuf="set xtics (";dlmt=""
    for I in range(Np):
        sbuf+=dlmt+"\"%s\" %f"%(labels[I],I_to_s[I]);dlmt=","
    sbuf+=")"
    print(sbuf,file=fdGNU)
    mu_au=mf.get_fermi( mo_energy_kpts=mf.mo_energy, mo_occ_kpts=mf.mo_occ)
    print("EF_au=%f"%(mu_au),file=fdGNU)
    print("set ylabel \"{/Symbol e}_k-{/Symbol m} (eV)\"",file=fdGNU)
    N_ele=int(round(get_Nele(mf)))
    HOMO=int( (N_ele+1)//2 )
    MOstt=HOMO-2;MOupl=HOMO+4
    MOstt=max(0,HOMO-2);MOupl=min(nMO,HOMO+4)
    print("MOstt:",MOstt,MOupl,HOMO,range(MOstt,MOupl))
    MOlabels=[ ("HOMO-%d"%(HOMO-mo) if(mo<HOMO) else \
             ( "HOMO" if(mo==HOMO) else ("LUMO" if(mo==HOMO+1) else "LUMO+%d"%(mo-HOMO-1)) ) ) for mo in range(MOstt,MOupl) ]
    for loop in range(2):
        PLOT="plot ";FNME=fpath;STY=1
        for mo in range(MOstt,MOupl):
            print(PLOT+"\""+FNME+"\""+" using 2:(HARTREEinEV*($"+str(6+mo)+"-EF_au)) title \""\
                  + ("" if(loop==1) else MOlabels[mo-MOstt]) \
                  + "\" with lines ls "+str(STY),end="",file=fdGNU)
            PLOT=",\\\n";FNME="";STY+=1
        print("\n",file=fdGNU)
        MOstt=0;MOupl=nMO;MOlabels=None
    fdGNU.close();fdDAF.close()
    os.system("plot.sh "+gnu)

def gen_bset_info(cell,IZnuc):
    ## generalization of format_basis({'Ti':'sto-3g','O':'sto-3g'}) ..
    from pyscf.pbc.gto.cell import format_basis
    Nat=len(IZnuc)
    if( isinstance(cell.basis,str) ):
        dict={}
        for iat in range(Nat):
            Z=IZnuc[iat]
            Sy=atomicnumber_to_atomicsymbol(Z)
            dict.update({Sy:cell.basis})
        ret=format_basis(dict)
        return ret
    elif( isinstance(cell.basis,dict) ):
        ret=format_basis(cell.basis)
        return ret
    else:
        assert False,"check cell.basis and its type.."+str(type(cell.basis))+" "+str(cell.basis)        
        return None

def calc_pDOS(mf,dafpath,MO_engs,MO_coefs,MO_occ,de_eV=0.01, widths_eV=None, LowdinPop=True ):
    from pyscf.pbc.gto import Cell
    dbgng_einsum=True # XXX XXX
    TINY=1.0e-12
    SQRT2ln2=1.177410022515474691
    if(widths_eV is None):
        widths_eV=[ 0.50/(2*SQRT2ln2), 0.05/(2*SQRT2ln2) ]
        # 0.21233, 0.021233
    HARTREEinEV=PhysicalConstants.HARTREEinEV()

    dic_wt={};dic_wtAV={}
    wt0=time.time();wt1=wt0
    mol_or_cell=mf.mol
    pbc=(None if(not hasattr(mf,"_pbc")) else mf._pbc)
    if( pbc is None ):
        pbc=isinstance(mol_or_cell,Cell);
    if( pbc ):
        assert isinstance( mol_or_cell,Cell),""
    Rnuc_au,Sy=parse_xyzstring( mol_or_cell.atom, unit_BorA='B')
    EF_au=mf.get_fermi( mo_energy_kpts=MO_engs, mo_occ_kpts=MO_occ)
    EF_eV=EF_au*HARTREEinEV
    Nat=len(Sy)
    IZnuc=[];
    for i in range(Nat):
        IZnuc.append( atomicsymbol_to_atomicnumber( Sy[i] ) )
    bset_info=gen_bset_info(mol_or_cell,IZnuc)
    nAO = mol_or_cell.nao_nr()
    aTOlmax={};aTOlmin={}
    muTOell=[ -1 for jao in range(nAO) ]
    muTOatm=[ "" for jao in range(nAO) ]
    if( mol_or_cell.cart ):
        print("!W cartesian bset..");
    mu_off=0;o=0
    for Iat in range(Nat):
        atI=Sy[Iat]
        info=bset_info[atI]
        Nblc=len(info)
        lmin=9999;lmax=-1
        for Iblc in range(Nblc):
            ell=info[Iblc][0];nmult=( (ell+2)*(ell+1)//2 if(mol_or_cell.cart) else 2*ell+1)
            lmin=min(lmin,ell);lmax=max(lmax,ell)
            ## info[Iblc][0]    :: ell
            ## info[Iblc][1]    :: 7.2610457926, -0.2798628497, 0.0501840556, 0.0 
            ## info[Iblc][2]    :: alph[1]
            ## ...
            ## info[Iblc][nPGTO]:: alph[nPGTO-1]
            Ld=len(info[Iblc])
            alphcofs=np.array([ info[Iblc][k] for k in range(1,Ld) ]);nd=np.shape(alphcofs)
            print("#Alphcofs:%d.%d ::%s %s %s"%(Iat,Iblc, str(nd), str(type(alphcofs)), str(alphcofs)))

            nPGTO=nd[0]; nCGTO=nd[1]-1
            print("#alphcofs:%d.%d  alpha:%s"%(Iat,Iblc, str(alphcofs[:,0])))
            for cgt in range(nCGTO):
                print("#alphcofs:%d.%d.%d  coefs:%s"%(Iat,Iblc, cgt, str(alphcofs[:,cgt+1])))
            for cgt in range(nCGTO):
                for k in range(nmult):
                    muTOell[ mu_off+k ]=ell
                    muTOatm[ mu_off+k ]=atI
                mu_off+=nmult
                print("%d.%d: l=%d nmult=%d cum:%d"%(Iat,Iblc,ell,nmult,mu_off))
        if( atI not in aTOlmax ):
            aTOlmax.update({atI:lmax})
            aTOlmin.update({atI:lmin})
        else:
            assert lmax==aTOlmax[atI],""
            assert lmin==aTOlmin[atI],""
    
    assert mu_off==nAO,"mu_cum:%d / %d"%(mu_off,nAO)
    
    spdf_=[ "s","p","d","f","g","h","i","j" ]
    distinctAtoms=[];N_rows=0
    rowTOlabel=[]
    P=[]
    for A in aTOlmax:
        distinctAtoms.append(A)
        lmax=aTOlmax[A]
        lmin=aTOlmin[A]
        for ell in range(lmin,lmax+1):
            rowTOlabel.append(A+"_"+spdf_[ell])
            iA=np.zeros([nAO],dtype=int)
            for mu in range(nAO):
                if( muTOatm[mu]==A and muTOell[mu]==ell ):
                    iA[mu]=1
            P.append(iA)
        n_ell=lmax-lmin+1
        N_rows+=n_ell
    nDa=len(distinctAtoms)
    for mu in range(nAO):
        nONE=0
        for row in range(N_rows):
            if(P[row][mu]==1):
                nONE+=1
        assert nONE==1,""
    rowTOlabel.append("sum");row_sum=len(rowTOlabel)-1
    P.append( np.array([ 1 for mu in range(nAO)]) ); N_rows+=1
    P=np.array(P)

    Ndim=np.shape(MO_engs)
    assert len(Ndim)==2," PLS extend this method to include molecules, UKS etc... Ndim:"+str(Ndim)
    nKpt=Ndim[0];nMO=Ndim[1]
    
    moe1D=np.ravel(MO_engs)
    moemin=min( moe1D );moemax=max(moe1D)
    wideVmin=min( widths_eV); wideVmax=max( widths_eV);
    Nwid=len(widths_eV)
    sqrt2PI = 2.506628274631000502415765284811
    emin_eV=moemin*HARTREEinEV -wideVmax*10
    emax_eV=moemax*HARTREEinEV +wideVmax*10
    NstepTOT=int(round( (emax_eV-emin_eV)/de_eV ))
    print("#calc_pDOS:[%f:%f)/%d "%(emin_eV,emax_eV,de_eV))
    lnTINY=np.log(TINY);sqrt2xlnTINY=np.sqrt(-2*lnTINY)
    jwidTOediffUPL= [ sqrt2xlnTINY * widths_eV[k] for k in range(Nwid) ]
    jwidTOfac     = [ 1.0/(sqrt2PI*widths_eV[k]) for k in range(Nwid) ]
    hfINVwidths   = [ 0.50/(widths_eV[k]**2) for k in range(Nwid) ]

    fdDAF=open(dafpath,"w")
    gnu=dafpath.replace(".dat","");gnuf=gnu+".plt"
    fdGNU=open(gnuf,"w");
    print("set term postscript color\nset output \"%s.ps\"\nload \"C:/cygwin64/usr/local/bin/stdcolor.plt\""%(gnu),file=fdGNU)
    print("EFeV=%18.8f"%(EF_eV),file=fdGNU)
    ncol=0
    legend="#%14s "%("%d:E_eV"%(ncol+1));ncol+=1
    gnucmd="";
    for jw in range(Nwid):
        gnucmd+="\nset title \"pDOS width=%f eV\"\n"%(widths_eV[jw]);PLOT="plot ";DAF=dafpath;STY=2
        for row in range(N_rows):
            legend+="   ".join([ "%16s %16s"%( "%d:occ-%s"%(ncol+1,rowTOlabel[row]), "%d:%s"%(ncol+2,rowTOlabel[row]) ) ])+"      "
            gnucmd+=PLOT+"\""+DAF+"\" using ($1-EFeV):%d title \"%s\" with lines ls %d"%(ncol+2,rowTOlabel[row],STY)
            ncol+=2;PLOT=",\\\n";DAF="";STY+=1
    print(legend,file=fdDAF);print("##"+legend,file=fdGNU)
    print(gnucmd,file=fdGNU)
    kvectors=np.reshape( mf.kpts, (-1,3))
    assert len(kvectors)==nKpt,"check nKpt:%d"%(nKpt)+str(np.shape(kvectors))

    SxC=None;sSQRTxC=None
    if(LowdinPop):
        sSQRTxC=np.zeros([nKpt,nAO,nMO],dtype=np.complex128)
        mf.update_Sinv()
        sSQRT=mf._Ssqrt
        for kp in range(nKpt):
            sSQRTxC[kp]=np.matmul( sSQRT[kp], MO_coefs[kp])
    else:
        SxC=np.zeros([nKpt,nAO,nMO],dtype=np.complex128)
        for kp in range(nKpt):
            S1e=mf.get_ovlp( mol_or_cell,kvectors[kp] )
            SxC[kp]=np.matmul(S1e,MO_coefs[kp])
            ## SxC.append( np.matmul(S1e,MO_coefs[kp]) )
    
    occSums=np.zeros([Nwid,N_rows],dtype=np.complex128)
    totSums=np.zeros([Nwid,N_rows],dtype=np.complex128)
    
    #
    # We here make expo[Nwid][nMO][NstepTOT]
    # we set upl : 3 x 100 x 4000 for example...
    LdTOT=Nwid*nMO*NstepTOT
    LdUPL=1200000;nBlock=1;Nstep_ref=NstepTOT
    if(LdTOT > LdUPL):
        nBlock=int( math.ceil( LdTOT/float(LdUPL) ) )
        Nstep_ref=int( math.ceil( NstepTOT/nBlock ) )

    both = True ## TODO 
    AOpop_Lowdin  =(None if((not LowdinPop) and (not both) ) else np.zeros([nKpt,nAO,nMO], dtype=np.float64))
    AOpop_Mulliken=(None if(LowdinPop and (not both)) else np.zeros([nKpt,nAO,nMO], dtype=np.complex128))
    if(LowdinPop or both):
        # |sSQRT[mu][:]C[:][io]|**2
        mf.update_Sinv()
        sSQRT=mf._Ssqrt
        for kp in range(nKpt):
            sSQRTxC=np.matmul(sSQRT[kp],MO_coefs[kp]) ## [nAO,nAO] x [nAO,nMO]
            for ao in range(nAO):
                for mo in range(nMO):
                    AOpop_Lowdin[kp][ao][mo]= sSQRTxC[ao][mo].real**2 + sSQRTxC[ao][mo].imag**2
    if((not LowdinPop) or both):
        # S[mu][:] C[:][o] C[mu][o]^{\ast}
        for kp in range(nKpt):
            S1e=mf.get_ovlp( mol_or_cell,kvectors[kp] )
            SxC=np.matmul(S1e,MO_coefs[kp])
            for ao in range(nAO):
                for mo in range(nMO):
                    AOpop_Mulliken[kp][ao][mo]= SxC[ao][mo]* np.conj(MO_coefs[kp][ao][mo])

    rowTOpop=None;AOpop=None
    AOpop=( AOpop_Lowdin if(LowdinPop) else AOpop_Mulliken )

    if( AOpop_Lowdin is not None ):
        a1d=np.ravel(AOpop_Lowdin)
        Nnega=np.count_nonzero( a1d<0.0 )
        if(Nnega>0):
            minv=min(a1d)
            rttddft_common.Printout_warning("LowdinPop","Nnega:%d minv:%f"%(Nnega,minv))
    if( AOpop_Mulliken is not None ):
        a1d=np.ravel(AOpop_Mulliken)
        Nnega=np.count_nonzero( a1d<0.0 )
        if(Nnega>0):
            minv=min(a1d)
            rttddft_common.Printout_warning("MullikenPop","Nnega:%d minv:%f"%(Nnega,minv))
    check_populations=True

    if(check_populations):
        N_ele = get_Nele(mf)
        fdOU=open("populations.dat","w");delimiter="    "
        rowTOlpop =(None if((not LowdinPop) and (not both) ) else np.zeros([N_rows], dtype=np.float64))
        rowTOmpop =(None if(LowdinPop and (not both)) else np.zeros([N_rows], dtype=np.complex128))

        if(LowdinPop or both):
            for kp in range(nKpt):
                rowTOlpop += np.einsum('rm,mi,i -> r',P,AOpop_Lowdin[kp],MO_occ[kp])
            rowTOlpop *= (1.0/float(nKpt))
            print("LowdinPop:", delimiter.join( [ "%s:%f"%(rowTOlabel[r],rowTOlpop[r]) for r in range(N_rows) ]), file=fdOU)
            if( row_sum > 0 ):
                testee = rowTOlpop[row_sum];dev=abs(testee-N_ele)
                print("#LowdinPop:%f / %f %e"%(testee,N_ele,dev))
                if( dev > 1e-6 ):
                    rttddft_common.Assert(dev<1e-6,"#LowdinPop:%f / %f %e"%(testee,N_ele,dev),1)
        if((not LowdinPop) or both):
            for kp in range(nKpt):
                rowTOmpop += np.einsum('rm,mi,i -> r',P,AOpop_Mulliken[kp],MO_occ[kp])
            rowTOmpop *= (1.0/float(nKpt))
            print("MullikenPop:", delimiter.join( [ "%s:%f"%(rowTOlabel[r],rowTOmpop[r]) for r in range(N_rows) ]), file=fdOU)

        rowTOpop=(rowTOlpop if(LowdinPop) else rowTOmpop)

    step0=0;step1=Nstep_ref
    wt2=wt1;wt1=time.time()
    for iBlc in range(nBlock):
        assert step0<NstepTOT,""

        e0=emin_eV + step0*de_eV
        e1=emin_eV + step1*de_eV
        Nstep=step1-step0
        args=np.zeros([Nwid,nMO,Nstep],dtype=np.float64)

        pDOSocc=np.zeros([Nwid,N_rows,Nstep],dtype=np.complex128)
        pDOStot=np.zeros([Nwid,N_rows,Nstep],dtype=np.complex128)

        for kp in range(nKpt):
            eorbs_eV=np.array( MO_engs[kp] )*HARTREEinEV
            for jw in range(Nwid):
                for io in range(nMO):
                    args[jw][io]=- ( np.linspace( e0-eorbs_eV[io],e1-eorbs_eV[io],num=Nstep,endpoint=False )**2 )*hfINVwidths[jw]
            wt2=wt1;wt1=time.time()
            expo=np.exp(args)
            for jw in range(Nwid):
                expo[jw]*=jwidTOfac[jw]

            # P[R][mu] SxC[mu][i] occ[i] C^{\ast}[mu][i] expon[JW][i][STEP] 
            # P[R][mu] AOpop[mu][i] occ[i] expon[JW][i][STEP] --> P[JW][R][STEP]
            pDOSocc += np.einsum('rm,mi,i,jip->jrp', P,AOpop[kp],MO_occ[kp],expo)
            pDOStot += np.einsum('rm,mi,jip->jrp',   P,AOpop[kp],expo)
            #ValueError: operands could not be broadcast together with shapes (6,2,18503) (2,6,18503) (6,2,18503)
        pDOSocc*=(1.0/float(nKpt))
        pDOStot*=(1.0/float(nKpt))

        for jw in range(Nwid):
            for ir in range(N_rows):
                occSums[jw][ir] += sum( pDOSocc[jw][ir] )*de_eV
                totSums[jw][ir] += sum( pDOStot[jw][ir] )*de_eV

        e=e0;delimiter="  ";
        sA=np.ravel( [ ["%16.6e %16.6e"%(pDOSocc[jw][ir][0].real,pDOStot[jw][ir][0].real) for ir in range(N_rows)]\
                       for jw in range(Nwid) ] ) 
        line=delimiter.join(sA)
        print(" %14.6f %s"%(e,line),file=fdDAF)
        for s in range(1,Nstep):
            jxi=0
            for jw in range(Nwid):
                for ir in range(N_rows):
                    sA[ jxi+ir ]="%16.6e %16.6e"%(pDOSocc[jw][ir][s].real,pDOStot[jw][ir][s].real)
                jxi+=N_rows
            line=delimiter.join(sA)
            e+=de_eV
            print(" %14.6f %s"%(e,line),file=fdDAF)

        step0+=Nstep;step1=min( step1+Nstep,NstepTOT ) #<< PLS NEVER REMOVE THIS ---
    assert step0>=NstepTOT,""

#        impartMax1=-1;impartMax2=-1
#        for jw in range(Nwid):
#            for row in range(N_rows):
#                string+="   ".join([ "%16.6e %16.6e"%(pDOSocc[row][jw].real,pDOStot[row][jw].real) ])+"      "
#                impartMax1 =max( impartMax1,abs(pDOSocc[row][jw].imag) )
#                impartMax2 =max( impartMax2,abs(pDOStot[row][jw].imag) )
#        print(string,file=fdDAF)
#        print("Imaginarypart:%e,%e"%(impartMax1,impartMax2))
#        rttddft_common.Assert( impartMax1<1e-5 and impartMax2<1e-5, "Imaginarypart:%e,%e"%(impartMax1,impartMax2), 1)
    fdDAF.close();fdGNU.close()
    os.system("fopen "+gnuf)
    check_walltime("",-1,dic_wt,dic_wtAV,ope_PorR="P",fpath="calc_pDOS_walltime.log",
          description="WT for nKpt:%d,nAO:%d,Nstep:%d,Nrow:%d ... %s"%( nKpt, nAO, Nstep, N_rows, str(rttddft_common.get_job(True))) )
    
    fnmeC=dafpath.replace(".dat","")+"_checksum.dat"
    fd1=open(fnmeC,"a");
    for jw in range(Nwid):
        delimiter="    "
        sA=[ "%s:%14.6f"%(rowTOlabel[row],occSums[jw][row]) for row in range(N_rows) ]
        string=delimiter.join( sA );refr=""
        if( rowTOpop is not None):
            refr=" pop:" + delimiter.join( [ "%s:%f"%(rowTOlabel[r],rowTOpop[r]) for r in range(N_rows) ])
        print("#wid:%f  OccSum:%s %s"%( widths_eV[jw],string,refr ),file=fd1)
        sA=[ "%s:%14.6f"%(rowTOlabel[row],totSums[jw][row]) for row in range(N_rows) ]
        string=delimiter.join( sA )
        print("###w:%f  TotSum:%s"%( widths_eV[jw],string ),file=fd1)
    fd1.close()
    os.system("fopen "+fnmeC)
def einsum_pDOS1_(P, SxC_kp, MO_coefs_kp, MO_occ_kp, gsf, N_row,Nwid,nAO,nMO):
    ret1=np.zeros([N_row,Nwid],dtype=np.complex128);
    ret2=np.zeros([N_row,Nwid],dtype=np.complex128);
    for row in range(N_row):
        for jw in range(Nwid):
            cum1=np.complex128(0.0);cum2=np.complex128(0.0)
            for mu in range(nAO):
                for iMO in range(nMO):
                    dum= P[row][mu]*SxC_kp[mu][iMO]*np.conj(MO_coefs_kp[mu][iMO])*gsf[iMO][jw]
                    cum1+= dum*MO_occ_kp[iMO]
                    cum2+= dum
            ret1[row][jw]=cum1
            ret2[row][jw]=cum2
    return ret1,ret2

