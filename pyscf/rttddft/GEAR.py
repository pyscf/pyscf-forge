import numpy as np
import os
#
#  Zc,Zp = dGEA2ndOne(dh,Zc,Zp,Accel,iFirst)
#  Zc[5][Ld] corrector,  F(t), F'(t), ... 
#  Zp[5][Ld] predictor,  F(t+h),F'(t+h),...

# You DO NEED to set Accel(t=0) when you initialize this array ... it is reflected on Zp vector as well
# rGEAR2ndOne updates Zc[2] and propagates Zp[2] ..
def initz_rGEARvec(dh,Xini,Vini,Accel,Ld):

    ZcOUT=np.zeros([5,Ld],dtype=np.float64)
    ZpOUT=np.zeros([5,Ld],dtype=np.float64)
    assert isinstance(Xini,np.ndarray),""
    assert isinstance(Vini,np.ndarray),""
    assert isinstance(Accel,np.ndarray),""
    
    ZpOUT[0]=Xini
    ZpOUT[1]=Vini*dh; ## d1Scle(Zp1,Ld,dh,1) 
    ZpOUT[2]=Accel*(0.5*(dh**2))  ## d1Scle(Accel,Ld,dh,2)
    ZpOUT[3]=0.0;      ## Zp3(:)=0.0d+0;  Zp4(:)=0.0d+0;
    ZpOUT[4]=0.0;      
    ZcOUT[0]=ZpOUT[0];    ## Zc0(:)=Zp0(:);
    ZcOUT[1]=ZpOUT[1];    ## Zc1(:)=Zp1(:);
    ZcOUT[2]=ZpOUT[2];    ## Zc2(:)=Zp2(:);
    ZcOUT[3]=0.0;      ## Zc3(:)=0.0d+0;
    ZcOUT[4]=0.0;      ## Zc4(:)=0.0d+0;
    
    return ZcOUT,ZpOUT

##
## converted from my own f90 source dGEA2ndOne  
##
def rGEAR2ndOne(dh,Zc,Zp,Accel,Ld):
    assert isinstance(Accel,np.ndarray),""
    Zc[2]=Accel*(0.5*(dh**2))
    return rGEARstp(Zc[2], 2,Zc,Zp,Ld)

def rGEARstp(Zcj,jIndx,Zc,Zp,iDim):
    if(jIndx == 2):
        c0=(1.9e+01)/(1.2e+02)
        c1=0.75e+00
        c2=1.0e+00
        c3=0.50e+00
        c4=(1.0e+00)/(1.2e+01)
    else:
        c0=(2.51e+00)/(7.2e+00)
        c1=1.0e+00
        c2=(1.1e+00)/(1.2e+00)
        c3=(1.0e+00)/(3.0e+00)
        c4=(1.0e+00)/(2.4e+01)
    if(jIndx == 2 ):
        dv=Zcj - Zp[2]
    else:
        dv=Zcj - Zp[1]
    
    Zc[0]=Zp[0]+c0*dv
    Zc[1]=Zp[1]+c1*dv
    Zc[2]=Zp[2]+c2*dv
    Zc[3]=Zp[3]+c3*dv
    Zc[4]=Zp[4]+c4*dv

    Zp[0]= Zc[0]+ Zc[1]+ Zc[2]+ Zc[3]+ Zc[4]
    Zp[1]= Zc[1]+ 2.0*Zc[2]+ 3.0*Zc[3]+ 4.0*Zc[4]
    Zp[2]= Zc[2]+ 3.0*Zc[3]+ 6.0*Zc[4]
    Zp[3]= Zc[3]+ 4.0*Zc[4]
    Zp[4]= Zc[4]

    return Zc,Zp

def polynomial(o,x,cofs):
    if(o==0):
        return polynomial_0(x,cofs)
    else:
        N=len(cofs);
        if(o==1):
            if(N<2):    # N==1 constant -> 0 
                return 0.0
            cofs1=[ k*cofs[k] for k in range(1,N) ]
            return polynomial_0(x,cofs1)
        elif(o==2):
            if( N<3):
                return 0.0
            cofs1=[ k*(k-1)*cofs[k] for k in range(2,N) ]
            return polynomial_0(x,cofs1)
        else:
            assert False,""

def polynomial_0(x,cofs):
    Nupl=len(cofs)
    retv=cofs[Nupl-1];
    for J in range(1,Nupl):
        retv=retv*x + cofs[Nupl-J-1]
    return retv

def fn1(o,x,xc,alphcofs):
    expf=np.exp( -alphcofs[0]*(x-xc)**2 );
    if(o==0):
        ### print("Alphcofs:",alphcofs[1:])
        return expf*polynomial(0,x,alphcofs[1:])
    elif(o==1):
        return -2*alphcofs[0]*(x-xc)*expf*polynomial(0,x,alphcofs[1:])\
               +expf*polynomial(1,x,alphcofs[1:]) 
    else:
        return (-2*alphcofs[0] + 4*alphcofs[0]*alphcofs[0]*( (x-xc)**2 ) )*expf*polynomial(0,x,alphcofs[1:]) \
            +2*(-2*alphcofs[0]*(x-xc))*expf*polynomial(1,x,alphcofs[1:]) \
            + expf*polynomial(2,x,alphcofs[1:])




## T=2.0
## x0=0.0
## xc=1.603
## Ld=3
## alphCofs=[ [1.192, -1.868, 1.192, 1.467, 0.645],
##            [3.142, 1.543, 0.894, 0.794, 0.375],
##            [0.645, 3.776,-8.848, 0.314, 0.272] ]
## N_iter=5
## ## Implicit Scheme
## ## initz GEAR vector
## ## t_0 :  a(0)        :   Xp(h)   Xc(0) 
## ## t_1 :  a(h)        :   Xp(2h)  Xc(h) 
## 
## R0=np.array( [ fn1(0,x0,xc,alphCofs[k]) for k in range(Ld) ] )
## V0=np.array( [ fn1(1,x0,xc,alphCofs[k]) for k in range(Ld) ] )
## 
## Implicit=True
## for strh in ["0.1", "0.05", "0.01"]:
##     h=float(strh)
##     N=int(round(T/h))
##     Accel=np.zeros([Ld],dtype=np.float64)
##     Zc,Zp=initz_rGEARvec( h,R0,V0,Accel,Ld)
##     Zc_nxt=np.zeros([5,Ld],dtype=np.float64);Zp_nxt=np.zeros([5,Ld],dtype=np.float64)
##     fnme="testInteg.1%s_"%("IMPLICIT" if(Implicit) else "EXPLICIT")+strh;fd=open(fnme+".dat","w");
##     Rref=np.zeros([Ld],dtype=np.float64)
##     Vref=np.zeros([Ld],dtype=np.float64)
##     x=x0
##     for step in range(N+1):
##         for iter in range(N_iter):
##             for k in range(Ld):
##                 Accel[k]=fn1(2,x,xc,alphCofs[k])
##             #OK: for ii in range(5):
##             #OK:    for jj in range(Ld):
##             #OK:        Zc_nxt[ii][jj]=Zc[ii][jj];Zp_nxt[ii][jj]=Zp[ii][jj]
##             #OK: Zc_nxt,Zp_nxt=rGEAR2ndOne(h,Zc_nxt,Zp_nxt,Accel,Ld)
##             Zc_nxt,Zp_nxt=rGEAR2ndOne(h,Zc,Zp,Accel,Ld)
##         ## update Zc,Zp using the latest one..
##         for ii in range(5):
##             for jj in range(Ld):
##                 Zc[ii][jj]=Zc_nxt[ii][jj];Zp[ii][jj]=Zp_nxt[ii][jj]
## 
##         if(Implicit):
##             R=Zc[0]; V=Zc[1]/h
##             for k in range(Ld):
##                 Rref[k]=fn1(0,x,xc,alphCofs[k]); Vref[k]=fn1(1,x,xc,alphCofs[k])
##             xDev=np.vdot( R-Rref, R-Rref)
##             vDev=np.vdot( V-Vref, V-Vref)
##             print("%12.6f   %14.6f  %14.6f  %14.6f     %14.6f  %14.6f  %14.6f   %14.6f  %14.6f  %14.6f     %14.6f  %14.6f  %14.6f  %14.6e %14.6e"%(\
##                 x,  R[0],R[1],R[2],  Rref[0],Rref[1],Rref[2],  V[0],V[1],V[2],  Vref[0],Vref[1],Vref[2], np.sqrt(xDev), np.sqrt(vDev)),file=fd)
##         else:
##             R=Zp[0]; V=Zp[1]/h
##             for k in range(Ld):
##                 Rref[k]=fn1(0,x+h,xc,alphCofs[k]); Vref[k]=fn1(1,x+h,xc,alphCofs[k])
##             xDev=np.vdot( R-Rref, R-Rref)
##             vDev=np.vdot( V-Vref, V-Vref)
##             print("%12.6f   %14.6f  %14.6f  %14.6f     %14.6f  %14.6f  %14.6f   %14.6f  %14.6f  %14.6f     %14.6f  %14.6f  %14.6f  %14.6e %14.6e"%(\
##                 x+h,  R[0],R[1],R[2],  Rref[0],Rref[1],Rref[2],  V[0],V[1],V[2],  Vref[0],Vref[1],Vref[2], np.sqrt(xDev), np.sqrt(vDev)),file=fd)
## 
##         x=x+h
##         
## 
## ## Explicit Scheme ...
## ###for strh in ["0.05", "0.10", "0.01"]:
## ###    h=float(strh)
## ###    N=int(round(T/h))
## ###    Zc=np.zeros([5,Ld],dtype=np.float64)
## ###    Zp=np.zeros([5,Ld],dtype=np.float64)
## ###    R=np.zeros([Ld],dtype=np.float64)
## ###    V=np.zeros([Ld],dtype=np.float64)
## ###    Rref=np.zeros([Ld],dtype=np.float64)
## ###    Vref=np.zeros([Ld],dtype=np.float64)
## ###    Accel=np.zeros([Ld],dtype=np.float64)
## ###
## ###    x=x0;
## ###    ## set IC on Zp array...
## ###    for k in range(Ld):
## ###        Zp[0][k]=fn1(0,x,xc,alphCofs[k])
## ###        Zp[1][k]=fn1(1,x,xc,alphCofs[k])
## ###
## ###    fnme="testInteg_"+strh;fd=open(fnme+".dat","w");
## ###    iFirst=True
## ###    for J in range(N):
## ###        for k in range(Ld):
## ###            Accel[k]=fn1(2,x,xc,alphCofs[k])
## ###        Zc,Zp=rGEAR2ndOne(h,Zc,Zp,Accel,Ld,iFirst);iFirst=False
## ###        R=Zp[0]; V=Zp[1]/h
## ###        for k in range(Ld):
## ###            Rref[k]=fn1(0,x,xc,alphCofs[k]); Vref[k]=fn1(1,x,xc,alphCofs[k])
## ###        xDev=np.vdot( R-Rref, R-Rref)
## ###        vDev=np.vdot( V-Vref, V-Vref)
## ###        print("%12.6f   %14.6f  %14.6f  %14.6f     %14.6f  %14.6f  %14.6f   %14.6f  %14.6f  %14.6f     %14.6f  %14.6f  %14.6f  %14.6e %14.6e"%(\
## ###            x+h,  R[0],R[1],R[2],  Rref[0],Rref[1],Rref[2],  V[0],V[1],V[2],  Vref[0],Vref[1],Vref[2], np.sqrt(xDev), np.sqrt(vDev)),file=fd)
## ###        x=x+h
## ###    fd.close()
##     os.system("ls -ltrh "+fnme+".dat")
##     os.system("gnuf.sh "+fnme)
##     fd=open(fnme+".plt","a")
##     print("plot \"%s.dat\" using 1:2 with lines ls 1,\\"%(fnme),file=fd);
##     print("\"\" using 1:5 with lines ls 101,\\",file=fd);
##     print("\"\" using 1:3 with lines ls 2,\\",file=fd);
##     print("\"\" using 1:6 with lines ls 102,\\",file=fd);
##     print("\"\" using 1:4 with lines ls 3,\\",file=fd);
##     print("\"\" using 1:7 with lines ls 103",file=fd);
## 
##     print("plot \"%s.dat\" using 1:8 with lines ls 1,\\"%(fnme),file=fd);
##     print("\"\" using 1:11 with lines ls 101,\\",file=fd);
##     print("\"\" using 1:9 with lines ls 2,\\",file=fd);
##     print("\"\" using 1:12 with lines ls 102,\\",file=fd);
##     print("\"\" using 1:10 with lines ls 3,\\",file=fd);
##     print("\"\" using 1:13 with lines ls 103",file=fd);
##     fd.close()
## ### INTEGER,intent(in) :: iDim,jIndx
## ### Double Precision,intent(inout) :: zc0(iDim),zc1(iDim),zc2(iDim),zc3(iDim),zc4(iDim),zcj(iDim),&
## ###           zp0(iDim),zp1(iDim),zp2(iDim),zp3(iDim),zp4(iDim),dv(iDim)
## ### DOUBLE PRECISION :: c0,c1,c2,c3,c4
## ###  IF(jIndx .EQ. 2) THEN
## ###    dv(:) = zcj(:)-zp2(:)
## ###  ELSE
## ###    dv(:) = zcj(:)-zp1(:)
## ###  END IF
## ###   zc0(:)=zp0(:)+c0*dv(:)
## ###   zc1(:)=zp1(:)+c1*dv(:)
## ###   zc2(:)=zp2(:)+c2*dv(:)
## ###   zc3(:)=zp3(:)+c3*dv(:)
## ###   zc4(:)=zp4(:)+c4*dv(:)
## ###
## ###   zp0(:)=zc0(:)+zc1(:)+zc2(:)+zc3(:)+zc4(:)
## ###   zp1(:)=zc1(:)+(2.0d+0)*zc2(:)+(3.0d+0)*zc3(:)+(4.0d+0)*zc4(:)
## ###   zp2(:)=zc2(:)+(3.0d+0)*zc3(:)+(6.0d+0)*zc4(:)
## ###   zp3(:)=zc3(:)+(4.0d+0)*zc4(:)
## ###   zp4(:)=zc4(:)
