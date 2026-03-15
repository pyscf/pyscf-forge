import os
import time
import numpy
from pyscf.pbc.gto import pseudo, estimate_ke_cutoff, error_for_ke_cutoff
from .utils import aNmaxdiff
from .gto_ps_pp_int01 import with_df_get_pp,get_pp_nl01
from pyscf.pbc import df
from pyscf.pbc import tools
from pyscf import lib
from pyscf.pbc.df import ft_ao
from pyscf.pbc.lib.kpts_helper import gamma_point
from .Loglv import printout
def dbg_fftdf_get_pp(scf,kpts):
    AoverC_al=[ numpy.array([0.0,0.0,0.0]),
                numpy.array([-1.603e-3, 1.868e-3, 0.645e-2]),
                numpy.array([-0.1192, 0.0645, 0.08848]),
                numpy.array([18.68,  14.67, -15.82]) ]
    ver=".3" ## calc_vppnl_by_k 
    scale_FFTmesh=0
    scale_FFTcutoff=0
    extension=""
    nA=len( AoverC_al )
    cell=scf.cell
    gdf=df.GDF(cell)
    gdf.kpts= kpts
    scf.with_df=gdf
    ppGDFal=[];ppFFTDFal=[];ppFFTDFalB=[]
    for Ia in range(nA):
        AoverC=AoverC_al[Ia]
        pp=with_df_get_pp(scf.with_df,AoverC,kpts=kpts,rttddft=scf)
        ppGDFal.append(pp)

    fdLOG=open("comp_ppMat%s.log"%(ver),"a")
    fftdf = df.FFTDF(cell)
    if(scale_FFTmesh>0):
        print("#original mesh:"+str(fftdf.mesh),file=fdLOG)
        for k in range(3):
            fftdf.mesh[k]=fftdf.mesh[k]*scale_FFTmesh
        print("#revised mesh:"+str(fftdf.mesh),file=fdLOG)
        extension="meshszX%d"%(scale_FFTmesh)
    if(scale_FFTcutoff>0):
        print("#original cutoff:%s mesh:"%(str(cell.ke_cutoff))+str(fftdf.mesh),file=fdLOG)
        if( cell.ke_cutoff is not None ):
            cell.ke_cutoff=cell.ke_cutoff*scale_FFTcutoff
            cell.build()
        else:
            cell.build()
            for k in range(3):
                cell.mesh[k]=cell.mesh[k]*scale_FFTcutoff
            print("expanding.mesh:",cell.mesh,file=fdLOG)
        print("cell.mesh:",cell.mesh,file=fdLOG)
        fftdf = df.FFTDF(cell)
        print("fftdf.mesh:",fftdf.mesh,file=fdLOG)
        print("#revised cutoff:%s mesh:"%(str(cell.ke_cutoff))+str(fftdf.mesh),file=fdLOG)
        extension="cutoffX%d"%( scale_FFTcutoff )
    fftdf.kpts= kpts
    scf.with_df=fftdf
    ### scf.kernel()
    for Ia in range(nA):
        AoverC=AoverC_al[Ia]
        pp=with_fftdf_get_pp(scf.with_df,AoverC,kpts=kpts, calc_vppnl_by_k=False)
        ppFFTDFal.append(pp)
    for Ia in range(nA):
        AoverC=AoverC_al[Ia]
        pp=with_fftdf_get_pp(scf.with_df,AoverC,kpts=kpts, calc_vppnl_by_k=True)
        ppFFTDFalB.append(pp)
    
    for Ia in range(nA):
        AoverC=AoverC_al[Ia]
        diff=aNmaxdiff(ppGDFal[Ia],ppFFTDFal[Ia])
        print(" %3d  %14.4e"%(Ia,diff),file=fdLOG)
        print_Hmatrices_1("#%03d:"%(Ia)+str(AoverC), ppGDFal[Ia],ppFFTDFal[Ia], Nref=5,
                fpath="comp_ppMat%s%03d%s.log"%(ver,Ia,extension), Append=False,TINY=1.0e-8)
    for Ia in range(nA):
        AoverC=AoverC_al[Ia]
        diff=aNmaxdiff(ppFFTDFal[Ia],ppFFTDFalB[Ia])
        print("calc_vppnl_by_k: %3d  %14.4e"%(Ia,diff),file=fdLOG)
        print_Hmatrices_1("#%03d:"%(Ia)+str(AoverC), ppGDFal[Ia],ppFFTDFal[Ia], Nref=5,
                fpath="comp_ppMat%s%03d%s.log"%(ver,Ia,extension), Append=True,TINY=1.0e-8)

    fdLOG.close()
    os.system("ls -ltrh comp_ppMat%s*.log"%(ver))

def with_fftdf_get_pp(mydf, AoverC, kpts=None, calc_vppnl_by_k=False):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    assert isinstance(AoverC,numpy.ndarray),"AoverC" 
    from pyscf import gto
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mesh = mydf.mesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)
    ngrids = len(vpplocG)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, mesh).real
    vpp = [0] * len(kpts_lst)
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
        ao_ks = ao_ks_etc[0]
        for k, ao in enumerate(ao_ks):
            vpp[k] += lib.dot(ao.T.conj()*vpplocR[p0:p1], ao)
        ao = ao_ks = None

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = numpy.empty((48,ngrids), dtype=numpy.complex128)
    def vppnl_by_k(kpt,AoverC):
#.0        Gk = Gv + kpt + AoverC
#.0        G_rad = lib.norm(Gk, axis=1)
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        Gk = Gv + kpt + AoverC
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5   ## .1:with kpt+AoverC >> wrong !!
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = numpy.ndarray((nl,l*2+1,ngrids), dtype=numpy.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = buf[:p1]
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = numpy.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./cell.vol)

    ABS_AoverC_TINY=1.0e-10
    wt00=time.time();wt01=wt00;wt02=wt01
    
    wt_A=0
    vppnl_al=None
    if(not calc_vppnl_by_k):
        vppnl_al=get_pp_nl01(cell, AoverC, kpts_lst)
        wt02=wt01;wt01=time.time(); wt_A=wt01-wt00

    abs_A_over_c = numpy.sqrt(  AoverC[0]**2 + AoverC[1]**2 + AoverC[2]**2 )
    for k, kpt in enumerate(kpts_lst):
        if( calc_vppnl_by_k ):
            wt01=time.time();
            vppnl = vppnl_by_k(kpt,AoverC)
            wt02=wt01;wt01=time.time();wt_A+=(wt01-wt02)
        else:
            vppnl = vppnl_al[k]
        if gamma_point(kpt):
            if( abs_A_over_c < ABS_AoverC_TINY ):
                vpp[k] = vpp[k].real + vppnl.real
            else:
                vpp[k] = vpp[k].real + vppnl
        else:
            vpp[k] += vppnl

    printout("ppnl%s:%f"%( ("_by_k" if(calc_vppnl_by_k) else "_lib"), wt_A),fpath="ppnl_walltime.log",\
             Append=True)

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return numpy.asarray(vpp)

def print_Hmatrices_1(title,A,B,Nref=5,fpath=None,Append=False,TINY=1.0e-8):

    fdOU=(None if(fpath is None) else open(fpath,("a" if(Append) else "w")) )
    fd=( fdOU if(fdOU is not None) else sys.stdout)
    if( Append ):
        print("\n\n",file=fd)
    Ndim=numpy.shape(A)
    rank=len(Ndim)
    if(rank==2):
        print_Hmatrices_1_(fd,title,A,B,Nref=Nref,TINY=TINY)
    elif(rank==3):
        for kp in range(Ndim[0]):
            print_Hmatrices_1_(fd,"kp=%02d "%(kp)+title,A[kp],B[kp],Nref=Nref,TINY=TINY)
            print("\n\n",file=fd)
    if(fdOU is not None):
        fdOU.close()
    
def print_Hmatrices_1_(fd,title,A,B,Nref=5,TINY=1.0e-8):


    maxdiff=aNmaxdiff(A,B)

    max_Im=-1;where=None
    Ndim=numpy.shape(A)
    assert len(Ndim)==2,"Ndim="+str(Ndim)+" len(Ndim)/=2"
    Nrow=Ndim[0];Ncol=Ndim[1]
    for I in range(Nrow):
        for J in range(Ncol):
            if(max_Im<abs(A[I][J].imag)):
                max_Im=abs(A[I][J].imag);where=[0,I,J]
            if(max_Im<abs(B[I][J].imag)):
                max_Im=abs(B[I][J].imag);where=[1,I,J]
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
            ##if(fdOU is None ):
            ##    print('#print_Hmatrices:'+title+':'+string)
            ##else:
            print(string,file=fd)
            
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
            ## if(fdOU is None ):
            ##    print('#print_Hmatrices:'+title+':'+string)
            ## else:
            print(string,file=fd)
