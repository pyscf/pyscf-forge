# The code in this file actually has nothing to do with CSFs
# Ideally, this file would be a part of pyscf/fci/spin_op.py
# However, I am sticking it here to avoid messy conflicts

from functools import reduce
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

def norm_sdown (smult, nelec):
    s2 = smult-1
    m2 = nelec[0] - nelec[1]
    return np.sqrt ((s2-m2+2)*(s2+m2)/4)

def norm_sup (smult, nelec):
    s2 = smult-1
    m2 = nelec[0] - nelec[1]
    return np.sqrt ((s2-m2)*(s2+m2+2)/4)

def contract_sladder(fcivec, norb, nelec, op=-1):
    ''' Contract spin ladder operator S+ or S- with fcivec.
        Changes neleca - nelecb without altering <S2>
        Obtained by modifying pyscf.fci.spin_op.contract_ss
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    assert (op in (-1,1)), 'op = -1 or 1'
    if ((op==-1 and (neleca==0 or nelecb==norb)) or
        (op==1 and (neleca==norb or nelecb==0))): return np.zeros ((0,0))
    # ^ Annihilate vacuum state ^

    def gen_map(fstr_index, nelec, des=True):
        a_index = fstr_index(range(norb), nelec)
        amap = np.zeros((a_index.shape[0],norb,2), dtype=np.int32)
        if des:
            for k, tab in enumerate(a_index):
                amap[k,tab[:,1]] = tab[:,2:]
        else:
            for k, tab in enumerate(a_index):
                amap[k,tab[:,0]] = tab[:,2:]
        return amap

    if op==-1:
        aindex = gen_map(cistring.gen_des_str_index, neleca)
        bindex = gen_map(cistring.gen_cre_str_index, nelecb, False)
    else:
        aindex = gen_map(cistring.gen_cre_str_index, neleca, False)
        bindex = gen_map(cistring.gen_des_str_index, nelecb)

    ci1 = np.zeros((cistring.num_strings(norb,neleca+op),
                   cistring.num_strings(norb,nelecb-op)))
    nspin_comm = neleca-1 if op==-1 else neleca
    spin_comm_fac = (-1) ** nspin_comm
    for i in range(norb):
        signa = aindex[:,i,1]
        signb = bindex[:,i,1]
        maska = np.where(signa!=0)[0]
        maskb = np.where(signb!=0)[0]
        addra = aindex[maska,i,0]
        addrb = bindex[maskb,i,0]
        citmp = lib.take_2d(fcivec, maska, maskb)
        citmp *= signa[maska].reshape(-1,1)
        citmp *= signb[maskb]
        citmp *= spin_comm_fac
        #: ci1[addra.reshape(-1,1),addrb] += citmp
        lib.takebak_2d(ci1, citmp, addra, addrb)
    return ci1

def contract_sdown (ci, norb, nelec):
    return contract_sladder (ci, norb, nelec, op=-1)

def contract_sup (ci, norb, nelec):
    return contract_sladder (ci, norb, nelec, op=1)

def as_list (ci0):
    if isinstance (ci0, np.ndarray) and ci0.ndim == 2:
        ci1 = [ci0,]
    else:
        ci1 = list (ci0)
    return ci1

def like_ci0 (ci1, ci0):
    if isinstance (ci0, np.ndarray) and ci0.ndim == 2:
        ci1 = ci1[0]
    else:
        ci1 = np.stack (ci1, axis=0)
    return ci1

def mdown (ci0, norb, nelec, smult):
    '''For a high-spin CI vector, rotate the spin axis away from the Z axis.

    Args:
        ci0 : ndarray or list of ndarrays
            Contains one or multiple CI vectors in the Hilbert space of
            (sum (nelec) + smult - 1) // 2 spin-up electrons and
            (sum (nelec) - smult + 1) // 2 spin-down electrons (i.e., m=s)
            with spin multiplicity given by smult
        norb : integer
            Number of orbitals
        nelec : (integer, integer)
            Number of spin-up and spin-down electrons in ci1
        smult : integer
            Spin multiplicity of ci0 and ci1

    Returns:
        ci1 : ndarray
            Contains one or multiple CI vectors in the Hilbert space of nelec[0] spin-up electrons
            and nelec[1] spin-down electrons occupying norb orbitals. If ci0 is spin-pure, then ci1
            is the same state with a different Z axis (i.e., a different m microstate).
    '''
    ci1 = as_list (ci0)
    neleca, nelecb = _unpack_nelec (nelec)
    for i in range (len (ci1)):
        for j in range (((smult-1)-(neleca-nelecb))//2,0,-1):
            ci1[i] = contract_sdown (ci1[i], norb, (neleca+j,nelecb-j))
            ci1[i] /= norm_sdown (smult, (neleca+j,nelecb-j))
    return like_ci0 (ci1, ci0)

def mup (ci0, norb, nelec, smult):
    '''For a high-spin CI vector, rotate the spin axis towards the Z axis.

    Args:
        ci0 : ndarray or list of ndarrays
            Contains one or multiple CI vectors in the Hilbert space of nelec[0] spin-up electrons
            and nelec[1] spin-down electrons occupying norb orbitals.
        norb : integer
            Number of orbitals
        nelec : (integer, integer)
            Number of spin-up and spin-down electrons in ci0
        smult : integer
            Spin multiplicity of ci0 and ci1

    Returns:
        ci1 : ndarray
            Contains one or multiple CI vectors in the Hilbert space of
            (sum (nelec) + smult - 1) // 2 spin-up electrons and
            (sum (nelec) - smult + 1) // 2 spin-down electrons (i.e., m=s).
            If ci0 is spin-pure, then ci1 is the same state with a different Z axis (i.e., a
            different m microstate).
    '''
    ci1 = as_list (ci0)
    neleca, nelecb = _unpack_nelec (nelec)
    for i in range (len (ci1)):
        for j in range (((smult-1)-(neleca-nelecb))//2):
            ci1[i] = contract_sup (ci1[i], norb, (neleca+j,nelecb-j))
            ci1[i] /= norm_sup (smult, (neleca+j,nelecb-j))
    return like_ci0 (ci1, ci0)

if __name__ == '__main__':
    import sys
    import time
    from pyscf.fci.direct_spin1 import contract_2e
    from pyscf.fci.spin_op import spin_square0
    t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
    nelec, norb = (int (argv) for argv in sys.argv[1:])
    nelec = (min (norb, nelec), nelec - min(norb, nelec))
    smult = nelec[0]-nelec[1]+1
    print ("Testing the spin ladder operators for a {}e, {}o s={} space".format (sum (nelec), norb, (smult-1)/2))
    cishape = [cistring.num_strings (norb, ne) for ne in nelec]
    np.random.seed(1)
    ci = np.random.rand (*cishape)
    eri = np.random.rand (norb,norb,norb,norb)
    ci /= linalg.norm (ci)
    print (" neleca nelecb ndeta ndetb {:>5s} {:>13s} {:>5s} {:>5s}".format ("cc","chc","ss","2s+1"))
    def print_line (c, ne):
        try:
            ss, smult = spin_square0 (c, norb, ne)
            hc = contract_2e (eri, c, norb, ne)
            chc = c.conj ().ravel ().dot (hc.ravel ())
        except AssertionError:
            assert (any ([n<0 for n in ne]))
            ss, smult, chc = (0.0, 1.0, 0.0)
        cc = linalg.norm (c)
        ndeta, ndetb = c.shape
        print (" {:>6d} {:>6d} {:>5d} {:>5d} {:5.3f} {:13.9f} {:5.3f} {:5.3f}".format (
            ne[0], ne[1], ndeta, ndetb, cc, chc, ss, smult))
    print_line (ci, nelec)
    for ndown in range (smult-1):
        ci = contract_sdown (ci, norb, nelec)
        nelec = (nelec[0]-1, nelec[1]+1)
        print_line (ci, nelec)
    for nup in range (smult):
        ci = contract_sup (ci, norb, nelec)
        nelec = (nelec[0]+1, nelec[1]-1)
        print_line (ci, nelec)
    print ("Time elapsed {} clock ; {} wall".format (lib.logger.process_clock () - t0,
                                                     lib.logger.perf_counter () - w0))

