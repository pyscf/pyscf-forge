import numpy as np
import os
import math
#shell_info_A[0]  [0, [18.121074109, -0.1275232222, -0.0248047945, 0.0, 0.0], 
#                     [7.37750664, -0.0812376594, -0.0171596817, 0.0, 0.0], 
#                     [2.8817585986, 0.3277442268, 0.0724090847, 0.0, 0.0], 
#                     [1.0990235357, 0.5946658147, 0.1739886836, 0.0, 0.0], 
#                     [0.400738772, 0.2436047977, 0.1404493622, 0.0, 0.0], 
#                     [0.091685942, -0.0017574619, -0.3860533721, 1.0, 0.0], 
#                     [0.0309750158, 0.0016745741, -0.736061992, 0.0, 1.0]] , 
#            [1]    [1, [18.121074109, 0.0729874345, 0.0, 0.0], 
#                      [7.37750664, 0.199671759, 0.0, 0.0], 
#                      [2.8817585986, 0.3527504683, 0.0, 0.0], 
#                      [1.0990235357, 0.3936231447, 0.0, 0.0], 
#                      [0.400738772, 0.2171401481, 0.0, 0.0], [0.091685942, 0.0205070521, 1.0, 0.0], [0.0309750158, -0.0045593404, 0.0, 1.0]], [2, [0.0973, 1.0]]]
# org[A] = shell_info[ 0:nSh[A] ]
#             shell_info[jsh] = [ ell_jsh, rows[0:nPGTO[elmt,jsh]]
#                                          rows[kPGTO]=[ alph, cof_0, cof_1, ... ]
def print_bset(org,exp_to_discard=None,comment=""):
    TINY=1.0e-30
    if( exp_to_discard is None ):
        exp_to_discard=-1.0
    ret=""
    for A in org:
        shell_info_A=org[A];
        nSh_A = len(shell_info_A)
        ret+=A+":nSh_A=%d\n"%(nSh_A)
        shell_info_A_nw=[]; nSh_A_removed=0
        for jSh_A in range(nSh_A):
            ell=int( shell_info_A[jSh_A][0] )
            alph_cofs = shell_info_A[jSh_A][1:]
            ### o.k. print("alph_cofs:"+str(np.shape(alph_cofs))+":"+str(alph_cofs))
            
            nPGTO=len(alph_cofs)
            row_0=alph_cofs[0];
            ncols=len(row_0)-1;
            alph=[ alph_cofs[kpgto][0] for kpgto in range(nPGTO) ]
            ncols_nw=0;jcol_remov=[]
            for jcol in range(ncols):
                cofs=[ alph_cofs[kpgto][1+jcol] for kpgto in range(nPGTO) ]
                cofs_nw=[];sqrsum=0.0
                for kpgto in range(nPGTO):
                    if( alph[kpgto] >= exp_to_discard ):
                        cofs_nw.append(cofs[kpgto]);sqrsum+=cofs[kpgto]**2
                sqrtnorm=np.sqrt(sqrsum);
                if(sqrtnorm>TINY):
                    ncols_nw+=1
                else:
                    jcol_remov.append(jcol)

            strbuf=["   %2d %11.4e %2s     "%( kpgto,alph_cofs[kpgto][0],\
                                                ("" if(alph_cofs[kpgto][0]>=exp_to_discard) else "***") ) for kpgto in range(nPGTO)]

            for kpgto in range(nPGTO):
                for jcol in range(ncols):
                    strbuf[kpgto]+="%12.6f%1s      "%(alph_cofs[kpgto][1+jcol],("x" if(jcol in jcol_remov) else ""))
            for line in strbuf:
                ret+="\n"+line
            ret+="\n";
    print("#print_bset:"+comment)
    print("#print_bset:"+ret)

def filter_by_exponent(org,exp_to_discard):
    TINY=1.0e-30
    ret={}
    for A in org:
        shell_info_A=org[A];
        ### print("#shell_info_%s:"%(A)+str(shell_info_A))
        ### print("#shell_info_%s:"%(A)+str(np.shape(shell_info_A)))
        nSh_A = len(shell_info_A)
        shell_info_A_nw=[]; nSh_A_removed=0
        for jSh_A in range(nSh_A):
            ell=int( shell_info_A[jSh_A][0] )
            alph_cofs = shell_info_A[jSh_A][1:]
            ### print("alph_cofs:"+str(np.shape(alph_cofs))+":"+str(alph_cofs))
            nPGTO=len(alph_cofs)
            row_0=alph_cofs[0];
            ncols=len(row_0)-1;
            alph_cofs_nw=[]; ncols_nw=0
            # i. find cols to discard ...
            alph=[ alph_cofs[kpgto][0] for kpgto in range(nPGTO) ]
            #if( min(alph)>=exp_to_discard ):
            #    ncols_nw=ncols;
            #    alph_cofs_nw=[]
            #    for kpgto in range(nPGTO):
            #        alph_cofs_nw.append([ alph_cofs[kpgto][0],
            #                              [ alph_cofs[kpgto][1+jcol] for jcol in range(ncols) ]])
            #else:
            jcol_remov=[]
            for jcol in range(ncols):
                cofs=[ alph_cofs[kpgto][1+jcol] for kpgto in range(nPGTO) ]
                cofs_nw=[];sqrsum=0.0
                for kpgto in range(nPGTO):
                    if( alph[kpgto] >= exp_to_discard ):
                        cofs_nw.append(cofs[kpgto]);sqrsum+=cofs[kpgto]**2
                sqrtnorm=np.sqrt(sqrsum);
                if(sqrtnorm>TINY):
                    ncols_nw+=1
                else:
                    jcol_remov.append(jcol)
            if(ncols_nw<1):
                print("#filter_by_exponent:dropping block %s.%d:"%(A,jSh_A) +str(alph))
                assert max(alph)<exp_to_discard,""
                nSh_A_removed+=1
                continue
            
            alph_cofs_nw=[]
            for kpgto in range(nPGTO):
                if( alph[kpgto]< exp_to_discard ):
                    continue
                acfs=[ alph_cofs[kpgto][0] ]
                for jcol in range(ncols):
                    if( jcol in jcol_remov ):
                        continue
                    else:
                        acfs.append( alph_cofs[kpgto][1+jcol])
                alph_cofs_nw.append( acfs )
                ### print("#alph_cofs_nw.append:%d:"%(len(alph_cofs_nw))+str([ alph_cofs[kpgto][0], cofs ]))
            buf=[]
            buf.append(ell);
            for dum in alph_cofs_nw:
                buf.append(dum)
            shell_info_A_nw.append(buf)
            if( exp_to_discard < 1.0e-20 ):
                print("#shell_info_A_nw.append:"+str(buf)) ### +str([ell, alph_cofs_nw ]))
                print("#shell_info_A[%d]:"%(jSh_A)+str(shell_info_A[jSh_A]))
                assert a1eqb(shell_info_A[jSh_A],buf);    ###[ell, alph_cofs_nw ])
        nSh_A_nw=len(shell_info_A_nw)
        assert (nSh_A_nw == nSh_A-nSh_A_removed), "%d/%d(=%d-%d)"%(nSh_A_nw, nSh_A-nSh_A_removed, nSh_A, nSh_A_removed)
        ret.update({A:shell_info_A_nw})
    if( exp_to_discard < 1.0e-20 ):
        print("#org:"+str(org))
        print("#ret:"+str(ret))
    return ret

def diceqb(A,B):
    for x in A:
        if( x in B ):
            if( not a1eqb( A[x],B[x] ) ):
                return False
        else:
            return False
    return True

def i1eqb(A,B):
    leA=len(A)
    if( leA==len(B) ):
        
        for j in range(leA):
            if(A[j]!=B[j]):
                return False
        return True
    else:
        return False
def a1eqb(A,B,TOL=1.0e-7):
    if( i1eqb(np.shape(A),np.shape(B)) ):
        a=np.ravel(A);b=np.ravel(B)
        lea=len(a);
        if(lea==len(b)):
            for j in range(lea):
                lhs=a[j];rhs=b[j]
                if( isinstance(lhs,float) or isinstance(lhs,np.float64) ):
                    if(abs(lhs-rhs)>=TOL):
                        print("#a1eqb:%f/%f %e"%(lhs,rhs,abs(lhs-rhs)));
                        return False
                elif( isinstance(lhs,int) ):
                    if( lhs!=rhs ):
                        return False
                elif( isinstance(lhs,list) ):
                    if( not a1eqb(lhs,rhs,TOL=TOL) ):
                        return False
            return True
        else:
            print("#shape differs:%d/%d"%(lea,len(b)));return False
    else:
        print("#Shape differs:"+str(np.shape(A))+" / "+str(np.shape(B)));return False
