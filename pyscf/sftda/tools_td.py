#/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

import numpy
from functools import reduce
from pyscf import lib
from pyscf.scf.uhf import spin_square as spin_square_scf

def spin_square(mf,xy,extype=0,tdtype='TDDFT'):
    r'''calculator of <S^2> of excited states using tddft/tda.
        Ref. J. Chem. Phys. 2011, 134, 134101.

    Args:
        mf :
            UKS object
        xy : tuple
            transition vactor of i-th state

    Kwargs:
        extype : int
            excitation types: 0,1,2
            excitation types: spin-flip-up, spin-flip-down, spin-conserved
        tdtype : str
            'TDDFT' or 'TDA' for different objects

    Returns:
        ssI :
            The expectation value of S^2.
    '''
    mo = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = numpy.where(mo_occ[0]>0)[0]
    occidxb = numpy.where(mo_occ[1]>0)[0]
    viridxa = numpy.where(mo_occ[0]==0)[0]
    viridxb = numpy.where(mo_occ[1]==0)[0]
    mooa = mo[0][:,occidxa]
    moob = mo[1][:,occidxb]
    mova = mo[0][:,viridxa]
    movb = mo[1][:,viridxb]

    ovlp = mf.get_ovlp()
    # get the <S^2>_0, 2S_0+1 for the ground state
    ss0,dsp1 = spin_square_scf((mooa,moob),ovlp)
    s = (dsp1-1) *.5

    x,y = xy
    # sxx_xx : spin transfer matrix
    sab_oo = reduce(numpy.dot, (mooa.conj().T, ovlp, moob))
    sba_oo = sab_oo.conj().T
    sab_vo = reduce(numpy.dot, (mova.conj().T, ovlp, moob))
    sba_vo = reduce(numpy.dot, (movb.conj().T, ovlp, mooa))
    sab_vv = reduce(numpy.dot, (mova.conj().T, ovlp, movb))
    sba_vv = sab_vv.conj().T

    if extype==0 or extype==1:
        x_ab,x_ba = x
        y_ab,y_ba = y

        if extype==0:
            x_ab = x_ab.transpose(1,0)
            if tdtype=='TDDFT':
                y_ba = y_ba.transpose(1,0)

            P_ab = lib.einsum('ai,aj,jk,ki',x_ab.conj(),x_ab,sab_oo.T.conj(),sab_oo)\
                  -lib.einsum('ai,bi,kb,ak',x_ab.conj(),x_ab,sab_vo.T.conj(),sab_vo)\
                  +lib.einsum('ai,bj,jb,ai',x_ab.conj(),x_ab,sab_vo.T.conj(),sab_vo)

            if tdtype=='TDDFT':
                P_ab+= lib.einsum('ai,aj,ik,kj',y_ba.conj(),y_ba,sab_oo,sab_oo.T.conj())\
                      -lib.einsum('ai,bi,ka,bk',y_ba.conj(),y_ba,sba_vo.T.conj(),sba_vo)\
                      +lib.einsum('ai,bj,ia,bj',y_ba.conj(),y_ba,sba_vo.T.conj(),sba_vo)

                P_ab-= lib.einsum('ai,bj,ai,bj',x_ab.conj(),y_ba,sab_vo,sba_vo) *2.0

            ds2 = P_ab + 2*s+1

        elif extype==1:
            x_ba = x_ba.transpose(1,0)
            if tdtype=='TDDFT':
                y_ab = y_ab.transpose(1,0)

            P_ab = lib.einsum('ai,aj,jk,ki',x_ba.conj(),x_ba,sba_oo.T.conj(),sba_oo)\
                  -lib.einsum('ai,bi,kb,ak',x_ba.conj(),x_ba,sba_vo.T.conj(),sba_vo)\
                  +lib.einsum('ai,bj,jb,ai',x_ba.conj(),x_ba,sba_vo.T.conj(),sba_vo)

            if tdtype=='TDDFT':
                P_ab+= lib.einsum('ai,aj,ik,kj',y_ab.conj(),y_ab,sba_oo,sba_oo.T.conj())\
                      -lib.einsum('ai,bi,ka,bk',y_ab.conj(),y_ab,sab_vo.T.conj(),sab_vo)\
                      +lib.einsum('ai,bj,ia,bj',y_ab.conj(),y_ab,sab_vo.T.conj(),sab_vo)

                P_ab-= lib.einsum('ai,bj,ai,bj',x_ba.conj(),y_ab,sba_vo,sab_vo) *2.0

            ds2 = P_ab - 2*s+1

    elif extype==2:
        x_aa,x_bb = x
        y_aa,y_bb = y
        x_aa = x_aa.transpose(1,0)
        x_bb = x_bb.transpose(1,0)
        if tdtype=='TDDFT':
            y_aa = y_aa.transpose(1,0)
            y_bb = y_bb.transpose(1,0)

        P_ab = lib.einsum('ai,aj,ki,jk',x_aa.conj(),x_aa,sba_oo,sab_oo)\
              -lib.einsum('ai,bi,ak,kb',x_aa.conj(),x_aa,sab_vo,sab_vo.conj().T)
        P_ab+= lib.einsum('ai,aj,ki,jk',x_bb.conj(),x_bb,sab_oo,sba_oo)\
              -lib.einsum('ai,bi,ak,kb',x_bb.conj(),x_bb,sba_vo,sba_vo.conj().T)
        P_ab-= lib.einsum('ai,bj,ji,ab',x_aa.conj(),x_bb,sba_oo,sab_vv) *2.0

        if tdtype=='TDDFT':
            P_ab+= lib.einsum('ai,aj,kj,ik',y_aa.conj(),y_aa,sba_oo,sab_oo)\
                  -lib.einsum('ai,bi,bk,ka',y_aa.conj(),y_aa,sab_vo,sab_vo.conj().T)
            P_ab+= lib.einsum('ai,aj,kj,ik',y_bb.conj(),y_bb,sab_oo,sba_oo)\
                  -lib.einsum('ai,bi,bk,ka',y_bb.conj(),y_bb,sba_vo,sba_vo.conj().T)
            P_ab-= lib.einsum('ai,bj,ba,ij',y_aa.conj(),y_bb,sba_vv,sab_oo) *2.0

            P_ab+= lib.einsum('ai,bj,aj,bi',x_aa.conj(),y_bb,sab_vo,sba_vo) *2.0
            P_ab+= lib.einsum('ai,bj,aj,bi',x_bb.conj(),y_aa,sba_vo,sab_vo) *2.0

        ds2 = P_ab

    ssI= ss0 + ds2
    return ssI
