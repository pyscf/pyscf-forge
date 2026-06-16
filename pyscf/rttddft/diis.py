#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

from functools import reduce
import numpy
import scipy.linalg
import scipy.optimize
from pyscf import lib
from pyscf.lib import logger

DEBUG = False

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    def __init__(self, mf=None, filename=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.space = 8

    def update(self, s, d, f, *args, **kwargs):
        errvec = get_err_vec(s, d, f)
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        xnew = lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

SCFDIIS = SCF_DIIS = DIIS = CDIIS

def get_err_vec(s, d, f):
    '''error vector = SDF - FDS'''
    if isinstance(f, numpy.ndarray) and f.ndim == 2:
        sdf = reduce(numpy.dot, (s,d,f))
        errvec = sdf.T.conj() - sdf

    elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
        errvec = []
        for i in range(f.shape[0]):
            sdf = reduce(numpy.dot, (s[i], d[i], f[i]))
            errvec.append((sdf.T.conj() - sdf))
        errvec = numpy.vstack(errvec)

    elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
        nao = s.shape[-1]
        s = lib.asarray((s,s)).reshape(-1,nao,nao)
        return get_err_vec(s, d.reshape(s.shape), f.reshape(s.shape))
    else:
        raise RuntimeError('Unknown SCF DIIS type')
    return errvec


class EDIIS(lib.diis.DIIS):
    '''SCF-EDIIS
    Ref: JCP 116, 8255 (2002); DOI:10.1063/1.1470195
    '''
    def update(self, s, d, f, mf, h1e, vhf):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = numpy.zeros(self.space)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = mf.energy_elec(d, h1e, vhf)[0]
        self._head += 1

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        etot, c = ediis_minimize(es, ds, fs)
        logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        return fock

def ediis_minimize(es, ds, fs):
    nx = es.size
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    diag = df.diagonal()
    df = diag[:,None] + diag - df - df.T

    def costf(x):
        c = x**2 / (x**2).sum()
        return numpy.einsum('i,i', c, es) - numpy.einsum('i,ij,j', c, df, c)

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = es - 2*numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()


class ADIIS(lib.diis.DIIS):
    '''
    Ref: JCP 132, 054109 (2010); DOI:10.1063/1.3304922
    '''
    def update(self, s, d, f, mf, h1e, vhf):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = numpy.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = numpy.zeros(shape, dtype=f.dtype)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        fun, c = adiis_minimize(ds, fs, self._head)
        if self.verbose >= logger.DEBUG1:
            etot = mf.energy_elec(d, h1e, vhf)[0] + fun
            logger.debug1(self, 'E %s  diis-c %s ', etot, c)
        fock = numpy.einsum('i,i...pq->...pq', c, fs)
        self._head += 1
        return fock

def adiis_minimize(ds, fs, idnewest):
    nx = ds.shape[0]
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = numpy.einsum('inpq,jnqp->ij', ds, fs).real
    d_fn = df[:,idnewest]
    dn_f = df[idnewest]
    dn_fn = df[idnewest,idnewest]
    dd_fn = d_fn - dn_fn
    df = df - d_fn[:,None] - dn_f + dn_fn

    def costf(x):
        c = x**2 / (x**2).sum()
        return (numpy.einsum('i,i', c, dd_fn) * 2 +
                numpy.einsum('i,ij,j', c, df, c))

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = 2*dd_fn
        fc+= numpy.einsum('j,kj->k', c, df)
        fc+= numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = numpy.random.random(nx)
        dfx0 = numpy.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
        print((dfx0 - grad(x0)) / dfx0)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()

class DIIS_dmat:
    def __init__(self,size,pbc,scheme="SDFFDS",istep_start_DIIS=6):
        self.size=size;
        self.nbuf=0
        self.scheme=scheme
        self.vecbuf=[ None for j in range(self.nbuf) ]
        self.erbuf= [ None for j in range(self.nbuf) ]
        self.pbc=pbc
        self.minimum_buflen = 4
        self.status=0
        self.step=-1
        self.thr_start_DIIS = 1.0e-3
        self.nstep_start_DIIS = 6
        self.Ndim_dmat=None
        self.Ld_dmat=None
        self.dtype=None
        self.mixing_ratio=0.50   ## 1 means full update
        self.S1e=None;self.nKpoints=None
        self.mat=None
        self.istep_start_DIIS= istep_start_DIIS
        self.rollback=None;self.space=size ## we formally define these parameters
## DM[nKpoints][nAO][nAO]
    def inner_product(self,lhs1D,rhs1D):
        if( self.S1e is None ):
            return numpy.vdot( lhs1D,rhs1D)
        else:
            Lhs=numpy.reshape(lhs1D,self.Ndim_dmat)
            Rhs=numpy.reshape(rhs1D,self.Ndim_dmat)
            nKpoints=(1 if(not self.pbc) else self.Ndim_dmat[0])
            cum=0.0
            for kp in range(nKpoints):
                lh=( Lhs if(not self.pbc) else Lhs[kp] )
                rh=( Rhs if(not self.pbc) else Rhs[kp] )
                s1=( self.S1e if(not self.pbc) else self.S1e[kp] )
                dum = numpy.matmul( lh, numpy.matmul( s1, numpy.matmul( rh, s1)))
                ld=len(dum)
                for j in range(ld):
                    cum+=dum[j][j]
            if(self.pbc):
                cum=cum/nKpoints
            return cum

    def get_damped_dmat(self, ibuf, Dmat1D):
        if( ibuf == 0 ):
            print("#DIIS_dmat:damped_dmat:%d 1*New"%(self.step))
            return numpy.reshape( Dmat1D, self.Ndim_dmat )
        else:
            last=(ibuf-1)%self.size
            print("#DIIS_dmat:damped_dmat:%d %f*New + (1-r)*Old[%d] "%(self.step,self.mixing_ratio,last ))
            return numpy.reshape( self.mixing_ratio * Dmat1D + ( 1 - self.mixing_ratio )* self.vecbuf[last], \
                                  self.Ndim_dmat )
    def clear_buffers(self):
        print("#DIIS_dmat.clear..:");
        self.Ndim_dmat=None;self.Ld_dmat=None;self.dtype=None
        self.vecbuf.clear();self.erbuf.clear();self.mat=None;self.nbuf=0
        self.status=0;self.step=-1

    def update(self, istep, Dmat, ervec=None, S1e=None,Fock=None, ervec_norm=None):
        print("#DIIS_dmat.update:istep,nbuf:",istep,self.nbuf)
        self.step=istep
        if( self.Ndim_dmat is None ):
            self.Ndim_dmat = numpy.shape(Dmat)
            rank=len(self.Ndim_dmat)
            Ld=self.Ndim_dmat[0]
            for j in range(1,rank):
                Ld=Ld*self.Ndim_dmat[j]
            self.Ld_dmat=Ld

            dtype=( Dmat.dtype if(isinstance(Dmat,numpy.ndarray)) else \
                    numpy.array( Dmat[0] ).dtype )
            self.dtype=dtype
            print("#DIIS_dmat.update:Ndim,Ld,dtype:",self.Ndim_dmat,Ld,self.dtype )
        dtype=( Dmat.dtype if(isinstance(Dmat,numpy.ndarray)) else \
                numpy.array( Dmat[0] ).dtype )
        if( self.dtype != dtype ):
            print("#DIIS.dtype:",self.dtype,dtype)
            self.dtype=dtype

        Dm1D=numpy.ravel(Dmat)

        if( ervec is None ):
            ervec= self.calc_errorvec( Dmat,S1e=S1e,Fock=Fock)
        if( self.status == 0 ):
            ## DIIS is not yet started
            if( ervec is not None ):
                ermax= max(ervec)
                skip_diis =(ermax > self.thr_start_DIIS and istep < self.istep_start_DIIS)
                print("#DIIS_dmat.update:ermax:%6d %14.6e %r"%(istep,ermax,skip_diis))
                if( skip_diis ):
                    return self.get_damped_dmat( self.nbuf, Dm1D)

        loc= self.nbuf%self.size
        print("#DIIS_dmat.update:nbuf/loc/bfsz:",self.nbuf,loc,self.size)
        if( len(self.vecbuf)<=loc ):
            self.vecbuf.append( arrayclone( Dm1D ) )
        else:
            if( self.vecbuf[loc] is None ):
                self.vecbuf[loc] = arrayclone( Dm1D )
            else:
                le=len(Dm1D)
                for j in range(le):
                    self.vecbuf[loc][j]=Dm1D[j]
        
        if( ervec is None ):
            self.status=1
            ## only at the 1st item
            assert self.nbuf==0,""
            return Dmat
        else:
            if( len(self.erbuf)<=loc):
                self.erbuf.append( arrayclone( ervec ) )
            else:
                if( self.erbuf[loc] is None ):
                    self.erbuf[loc] = arrayclone( ervec )
                else:
                    le=len( ervec )
                    for j in range(le):
                        self.erbuf[loc][j]=ervec[j]
        self.status=2
        ibuf=self.nbuf
        self.nbuf+=1
        if( self.mat is None ):
            self.mat = numpy.zeros([self.size,self.size],dtype=self.dtype)
        if( ibuf < self.size ):
            for j in range(ibuf+1):
                self.mat[ibuf][j]= self.inner_product( self.erbuf[loc], self.erbuf[j] )
        else:
            for j in range(loc+1):
                self.mat[ loc][j]= self.inner_product( self.erbuf[loc], self.erbuf[j] )
            for j in range(loc+1,self.size):
                self.mat[ j][loc]= self.inner_product( self.erbuf[j], self.erbuf[loc] )
        
        if( self.nbuf < self.minimum_buflen ):
            return self.get_damped_dmat(ibuf, Dm1D)
        Nv=min(self.size, self.nbuf)
        Ld=Nv+1
        wks=numpy.zeros( [Ld,Ld],dtype= self.dtype)
        for i in range(Nv):
            for j in range(i+1):
                wks[i][j]=self.mat[i][j]
                if(i!=j):
                    wks[j][i]=numpy.conj( self.mat[i][j] )
        for j in range(Ld-1):
            wks[Ld-1][j]=1.0
            wks[j][Ld-1]=1.0
        wks[Ld-1][Ld-1]=0.0

        w=numpy.zeros([Ld],dtype=self.dtype )
        for j in range(Ld-1):
            w[j]=0.0
        w[Ld-1]=1.0
        coef,res,rank,sv=numpy.linalg.lstsq(wks,w,rcond=None) ## this is equivalent to svdsolv

        retv_1d=numpy.zeros([self.Ld_dmat],dtype=self.dtype)
        retv_1d[:]=coef[0]*self.vecbuf[0][:]
        for J in range(1,Nv):
            retv_1d[:]+=coef[J]*self.vecbuf[J][:]
        return numpy.reshape( retv_1d, self.Ndim_dmat )
# eg. nbuf==2 -----------
#       0   1   2   3     
#   0   
#   1                  
#   2   *   *   *
# eg. nbuf==2 -----------
#       0   1   2     
#   0   *   
#   1   *               
#   2   *
    def calc_errorvec(self,Dmat,S1e=None,Fock=None):
        if( self.scheme == "SDFFDS" ):
            return self.calc_SDFFDS( Dmat, S1e, Fock)
        else:
            if( self.vecbuf[0] is None ):
                return None
##
## nbuf==0 but we have 0-th item
##
            if( self.nbuf == 0 ):
                last=0
            else:
                last=(self.nbuf-1)%self.size
            Dm1D=numpy.ravel(Dmat)
            print("dbgng:DM1D:",numpy.shape(Dm1D))
            return Dm1D-self.vecbuf[last]
 

    def calc_SDFFDS(self, Dmat,S1e,Fock):
        Ndim=numpy.shape(Dmat);rank=len(Ndim)
        #       RHF              UHF
        #  MOL  [nAO][nAO]       [2][nAO][nAO]
        #  PBC  [nkp][nAO][nAO] 
        assert (self.pbc and rank==3) or ( (not self.pbc) and rank==2),""
        nKpt=(1 if(not self.pbc) else Ndim[0])
        nAO =(Ndim[0] if(not self.pbc) else Ndim[1])
        ret =(None if(not self.pbc) else [] )
        for kp in range(nKpt):
            s1 =( S1e if(not self.pbc) else S1e[kp])
            fo =( Fock if(not self.pbc) else Fock[kp])
            dm =( Dmat if(not self.pbc) else Dmat[kp])
            sdffds= numpy.matmul( s1, numpy.matmul( dm, fo)) \
                    - numpy.matmul( fo, numpy.matmul( dm, s1))
            if( not self.pbc ):
                ret=numpy.ravel( sdffds );break
            else:
                ret.append( numpy.ravel( sdffds ));break
        return (ret if( not self.pbc ) else numpy.ravel(ret))

def arrayclone(src):
    dtype=None
    if( isinstance(src,numpy.ndarray) ):
        dtype=src.dtype
    else:
        dtype=( numpy.array(src[0]) ).dtype
           
    ndim=numpy.shape(src);rank=len(ndim)
    ret=numpy.zeros( ndim, dtype )
    for i in range(ndim[0]):
        if(rank==1):
            ret[i]=src[i];continue
        for j in range(ndim[1]):
            if(rank==2):
                ret[i][j]=src[i][j];continue
            for k in range(ndim[2]):
                if(rank==3):
                    ret[i][j][k]=src[i][j][k];continue
                for l in range(ndim[3]):
                    if(rank==4):
                        ret[i][j][k][l]=src[i][j][k][l];continue
                    for m in range(ndim[4]):
                        ret[i][j][k][l][m]=src[i][j][k][l][m];continue
    return ret
