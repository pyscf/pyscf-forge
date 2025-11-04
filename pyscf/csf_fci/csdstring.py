from pyscf.fci import cistring
from pyscf.lib import logger, load_library
from scipy import special
import numpy as np
import ctypes
import time
import os
libcsf = load_library ('libcsf')

# "CSD" means configuration-spin-determinant because I don't know what "split-GUGA" means
# The ordering, from slowest-changing index to fastest-changing index, is by 1) number of
# electron pairs, 2) configuration of paired electrons in all orbitals, 3) configuration of unpaired electrons
# in the norb - npair remaining orbitals, 4) configuration of up-spins in the unpaired electron string. Each of 2, 3,
# and 4 are internally ordered in the same way that PySCF orders CI addresses based on CI strings

def check_csd_mask_size (norb, neleca, nelecb):
    ''' Calculate the size of the mask index array to reorder a CI vector of (neleca, nelecb) electrons in norb orbitals
    and assert that it's not too big for 32-bit integers '''
    ndeta = special.binom (norb, neleca)
    ndetb = special.binom (norb, nelecb)
    mask_size = ndeta * ndetb
    mask_size_lim_gd = 2**32 / 1e9

    assert (mask_size / 1e9 <= mask_size_lim_gd), \
        '{:.2f} billion determinants; more than {:.2f} billion not supported'.format (mask_size / 1e9, mask_size_lim_gd)
    return mask_size

def make_csd_mask (norb, neleca, nelecb):
    ''' Get a mask index to reorder a (flattened) CI vector matrix in terms of
        (double_configuration, single_configuration, spin_configuration)

    mask[idx_csd] = idx_dd '''

    #t_start = logger.perf_counter ()
    ndeta = int (special.comb (norb, neleca))
    ndetb = int (special.comb (norb, nelecb))
    mask = np.empty (ndeta*ndetb, dtype=np.uint32)
    min_npair, npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    pair_size = npair_dconf_size * npair_sconf_size * npair_spins_size
    for npair in range (min_npair, min (neleca, nelecb)+1):
        ipair = npair - min_npair
        nspin = neleca + nelecb - 2*npair
        mask[npair_offset[ipair]:][:pair_size[ipair]] = get_nspin_dets (norb, neleca, nelecb, nspin).flat
    #print ("Time to make csf mask index array: {:.2f} seconds, memory: {:.2f} MB".format (
    #        logger.perf_counter () - t_start, mask.nbytes / 1e6, ndeta*ndetb*4 / 1e6))
    return mask

def make_econf_det_mask (norb, neleca, nelecb, csd_mask):
    ''' Get a mask index to identify the electron configuration (i.e., in csd order) of a given determinant pair address
    (in determinant-pair order) '''
    ndeta = int (special.comb (norb, neleca))
    ndetb = int (special.comb (norb, nelecb))
    mask = np.empty (ndeta*ndetb, dtype=np.uint32)
    min_npair, npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    npair_conf_size = npair_dconf_size * npair_sconf_size
    npair_det_size = npair_conf_size * npair_spins_size
    iconf = 0
    for npair in range (min_npair, min (neleca, nelecb)+1):
        ipair = npair - min_npair
        irange = np.arange (iconf, iconf+npair_conf_size[ipair], dtype=np.uint32)
        iconf += npair_conf_size[ipair]
        mask[npair_offset[ipair]:][:npair_det_size[ipair]] = np.repeat (irange, npair_spins_size[ipair])
    return mask[np.argsort (csd_mask)]

def get_nspin_dets (norb, neleca, nelecb, nspin):
    ''' Grab all determinant pair addresses corresponding to nspin unpaired electrons, sorted by spin configuration
        and separated into electron configuration blocks for easy spin-state transformations

    Args:
    norb, neleca, neleb, norb are integers

    Returns: 2d array of integers, rows (dim0) are electron configurations, columns (dim1) are spin configurations
        Address for the raveled version of the matrix CI vector, i.e., ideta*ndetb + idetb
    '''

    #t_start = logger.perf_counter ()
    assert ((neleca + nelecb - nspin) % 2 == 0)
    npair = (neleca + nelecb - nspin) // 2
    min_npair, npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    offset = npair_offset[npair-min_npair]
    conf_size = npair_dconf_size[npair-min_npair] * npair_sconf_size[npair-min_npair]
    spin_size = npair_spins_size[npair-min_npair]
    #t_ref = logger.perf_counter ()
    ddaddrs = csdaddrs2ddaddrs (norb, neleca, nelecb, list (range (offset, offset+(conf_size*spin_size))))
    #t_sub = logger.perf_counter () - t_ref
    ddaddrs = ddaddrs[0,:] * int (round (special.binom (norb, nelecb))) + ddaddrs[1,:]
    ddaddrs = ddaddrs.reshape (conf_size, spin_size)
    #t_tot = logger.perf_counter () - t_start
    return ddaddrs

def ddaddrs2csdaddrs (norb, neleca, nelecb, ddaddrs):
    ''' Convert double-determinant (DD) CI vector element addresses, [deta, detb], into
        configuration-spin-determinant (CSD) vector element addresses, [npair, dconf, sconf, spinstate],
        to facilitate a later transformation into CSFs. In the CSD format,
            max (0, neleca + nelecb - norb) <= npair <= nelecb is the number of electron pairs

            dconf is the address for a particular configuration of npair pairs'
            use cistring for npair 'electrons' in norb orbitals

            sconf is the address for a particular configuration of the neleca + nelecb - 2*npair (= nunp)
            unpaired electrons given dconf; use cistring for nunp electrons in norb - npair orbitals

            spinstate is the state of (nunp + neleca - nelecb)/2 alpha and (nunp - neleca + nelecb)/2 beta spins;
            use cistring for (nunp + neleca - nelecb)/2 'electrons' in nunp 'orbitals'

        Args:
        norb, neleca, nelecb are integers
        ddaddrs is an array that specifies double determinant CI vector element address
            If 1d, interpreted as deta_addr*ndetb + detb_addr
            If 2d, interpreted as row/column i is interpreted as the ith index pair [deta, detb], if there are 2
                columns/rows

        Returns:
        csdaddrs, list of integers for the CI vector in the form of a 1d array

    '''

    ddaddrs = format_ddaddrs (norb, neleca, nelecb, ddaddrs)
    ddstrs = np.asarray ([cistring.addrs2str (norb, neleca, ddaddrs[0]), cistring.addrs2str (norb, nelecb, ddaddrs[1])],
                         dtype=np.int64)
    csdstrs = ddstrs2csdstrs (norb, neleca, nelecb, ddstrs)
    csdaddrs = csdstrs2csdaddrs (norb, neleca, nelecb, csdstrs)
    return csdaddrs

def csdaddrs2ddaddrs (norb, neleca, nelecb, csdaddrs):
    ''' Inverse operation of ddaddrs2csdaddrs
        ddaddrs is returned in the format of a contiguous 2d ndarray with shape (naddrs,2), where naddrs is the length
        of csdaddrs
    '''
    #t_start = logger.perf_counter ()
    csdaddrs = np.asarray (csdaddrs)
    if not csdaddrs.flags['C_CONTIGUOUS']:
        csdaddrs = np.ravel (csdaddrs, order='C')
    #t0 = logger.perf_counter ()
    csdstrs = csdaddrs2csdstrs (norb, neleca, nelecb, csdaddrs)
    #t1 = logger.perf_counter ()
    ddstrs = csdstrs2ddstrs (norb, neleca, nelecb, csdstrs)
    #t2 = logger.perf_counter ()
    ddaddrs = np.ascontiguousarray (
        [cistring.strs2addr (norb, neleca, ddstrs[0]), cistring.strs2addr (norb, nelecb, ddstrs[1])], dtype=np.int32)
    #t3 = logger.perf_counter ()
    #t_tot = logger.perf_counter () - t_start
    return ddaddrs

def csdstrs2csdaddrs (norb, neleca, nelecb, csdstrs):
    assert (len (csdstrs[0]) == len (csdstrs[1]))
    assert (len (csdstrs[0]) == len (csdstrs[2]))
    assert (len (csdstrs[0]) == len (csdstrs[3]))
    min_npair, npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    csdaddrs = np.empty (len (csdstrs[0]), dtype=np.int32)
    for npair, offset, dconf_size, sconf_size, spins_size in zip (range (min_npair, min (neleca, nelecb)+1),
            npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size):
        nspins = neleca + nelecb - 2*npair
        nup = (nspins + neleca - nelecb) // 2
        assert ((nspins + neleca - nelecb) % 2 == 0)

        idx = (csdstrs[0] == npair)
        dconf_addr = cistring.strs2addr (norb, npair, csdstrs[1,idx])
        sconf_addr = cistring.strs2addr (norb - npair, nspins, csdstrs[2,idx])
        spins_addr = cistring.strs2addr (nspins, nup, csdstrs[3,idx])

        csdaddrs[idx] = np.asarray ([offset + (dconf * sconf_size * spins_size)
                                            + (sconf * spins_size)
                                            + spins for dconf, sconf, spins in zip (
                                            dconf_addr, sconf_addr, spins_addr)],
                                            dtype=np.int32)
    return csdaddrs


def csdaddrs2csdstrs (norb, neleca, nelecb, csdaddrs):
    ''' This is extremely slow because of the amount of repetition in dconf_addr, sconf_addr, and spins_addr! '''
    #t_start = logger.perf_counter ()
    min_npair, npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    csdstrs = np.empty ((4, len (csdaddrs)), dtype=np.int64)
    for npair, offset, dconf_size, sconf_size, spins_size in zip (range (min_npair, min (neleca, nelecb)+1),
            npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size):
        nspins = neleca + nelecb - 2*npair
        nup = (nspins + neleca - nelecb) // 2
        assert ((nspins + neleca - nelecb) % 2 == 0)
        next_offset = offset + (dconf_size * sconf_size * spins_size)

        idx = (csdaddrs >= offset) & (csdaddrs < next_offset)
        if len (idx) == 1:
            if not idx[0]:
                continue
        dconf_addr = (csdaddrs[idx] - offset) // (sconf_size * spins_size)
        dconf_rem  = (csdaddrs[idx] - offset)  % (sconf_size * spins_size)
        sconf_addr = dconf_rem // spins_size
        spins_addr = dconf_rem  % spins_size
        csdstrs[0,idx] = npair

        #t_ref = logger.perf_counter ()
        dconf_addr_uniq, dconf_addr_uniq2full = np.unique (dconf_addr, return_inverse=True)
        try:
            csdstrs[1,idx] = cistring.addrs2str (norb, npair, dconf_addr_uniq)[dconf_addr_uniq2full]
        except TypeError:
            csdstrs[1,idx] = cistring.addr2str (norb, npair, dconf_addr_uniq)

        sconf_addr_uniq, sconf_addr_uniq2full = np.unique (sconf_addr, return_inverse=True)
        try:
            csdstrs[2,idx] = cistring.addrs2str (norb - npair, nspins, sconf_addr_uniq)[sconf_addr_uniq2full]
        except TypeError:
            csdstrs[2,idx] = cistring.addr2str (norb - npair, nspins, sconf_addr_uniq)

        spins_addr_uniq, spins_addr_uniq2full = np.unique (spins_addr, return_inverse=True)
        try:
            csdstrs[3,idx] = cistring.addrs2str (nspins, nup, spins_addr_uniq)[spins_addr_uniq2full]
        except TypeError:
            csdstrs[3,idx] = cistring.addr2str (nspins, nup, spins_addr_uniq)
        #print ("{:.2f} seconds in cistring".format (logger.perf_counter () - t_ref))

        '''
        t_ref = logger.perf_counter ()
        try:
            csdstrs[1,idx] = cistring.addrs2str (norb, npair, dconf_addr)
            csdstrs[2,idx] = cistring.addrs2str (norb - npair, nspins, sconf_addr)
            csdstrs[3,idx] = cistring.addrs2str (nspins, nup, spins_addr)
        except TypeError:
            csdstrs[1,idx] = cistring.addr2str (norb, npair, dconf_addr)
            csdstrs[2,idx] = cistring.addr2str (norb - npair, nspins, sconf_addr)
            csdstrs[3,idx] = cistring.addr2str (nspins, nup, spins_addr)
        print ("{:.2f} seconds in cistring".format (logger.perf_counter () - t_ref))
        '''

    #print ("{:.2f} seconds spent in csdaddrs2csdstrs".format (logger.perf_counter () - t_start))
    return csdstrs

def get_csdaddrs_shape (norb, neleca, nelecb):
    ''' For a system of neleca + nelecb electrons with MS = (neleca - nelecb) occupying norb orbitals,
        get shape information about the irregular csdaddrs-type CI vector array (number of pairs, pair config,
        unpair config, spin state)

        Args:
        norb, neleca, nelecb are integers

        Returns:
        min_npair, integer, the lowest possible number of electron pairs
        npair_offset, 1d ndarray of integers
            npair_offset[i] points to the first determinant of a csdaddrs-sorted CI vector with i+min_npair electron
            pairs
        npair_dconf_size, 1d ndarray of integers
            npair_dconf_size[i] = number of pair configurations with i+min_npair electron pairs
        npair_sconf_size, 1d ndarray of integers
            npair_sconf_size[i] = number of unpaired electron configurations for a system of neleca+nelecb electrons
            with npair paired
        npair_spins_size, 1d ndarray of integers
            npair_spins_size[i] = number of states of neleca+nelecb - 2*npair spins with MS = (neleca - nelecb) / 2
    '''
    #assert (neleca >= nelecb)
    nless = min (neleca, nelecb)
    min_npair = max (0, neleca + nelecb - norb)
    nspins = [neleca + nelecb - 2*npair for npair in range (min_npair, nless+1)]
    nfreeorbs = [norb - npair for npair in range (min_npair, nless+1)]
    nas = [(nspin + neleca - nelecb) // 2 for nspin in nspins]
    for nspin in nspins:
        assert ((nspin + neleca - nelecb) % 2 == 0)

    npair_dconf_size = np.asarray ([int (round (special.binom (norb, npair)))
                                    for npair in range (min_npair, nless+1)], dtype=np.int32)
    npair_sconf_size = np.asarray ([int (round (special.binom (nfreeorb, nspin)))
                                    for nfreeorb, nspin in zip (nfreeorbs, nspins)], dtype=np.int32)
    npair_spins_size = np.asarray ([int (round (special.binom (nspin, na)))
                                    for nspin, na in zip (nspins, nas)], dtype=np.int32)

    npair_sizes = np.asarray ([0] + [i * j * k for i,j,k in zip (npair_dconf_size, npair_sconf_size, npair_spins_size)],
                              dtype=np.int32)
    npair_offset = np.asarray ([np.sum (npair_sizes[:i+1]) for i in range (len (npair_sizes))], dtype=np.int32)
    assert (npair_offset[-1] == int (round (special.binom (norb, neleca) * special.binom (norb, nelecb)))), npair_offset

    return min_npair, npair_offset[:-1], npair_dconf_size, npair_sconf_size, npair_spins_size

def ddstrs2csdstrs (norb, neleca, nelecb, ddstrs):
    ''' Transform from DD ci strings to CSD ci strings '''

    nstr = len (ddstrs[0])
    csdstrs = np.empty ((4, nstr), dtype=np.int64, order='C')
    if ddstrs.shape[1] == 2:
        ddstrs = ddstrs.T
    if not ddstrs.flags['C_CONTIGUOUS']:
        ddstrs = np.ascontiguousarray (ddstrs)
    libcsf.FCICSFddstrs2csdstrs (csdstrs.ctypes.data_as (ctypes.c_void_p),
                                ddstrs.ctypes.data_as (ctypes.c_void_p),
                                ctypes.c_size_t (nstr),
                                ctypes.c_int (norb),
                                ctypes.c_int (neleca), ctypes.c_int (nelecb))
    return csdstrs


def csdstrs2ddstrs (norb, neleca, nelecb, csdstrs):
    ''' Transform from CSD ci strings to DD ci strings '''

    nstr = len (csdstrs[0])
    ddstrs = np.empty ((2, nstr), dtype=np.int64, order='C')
    if not csdstrs.flags['C_CONTIGUOUS']:
        csdstrs = np.ravel (csdstrs)
    libcsf.FCICSFcsdstrs2ddstrs (ddstrs.ctypes.data_as (ctypes.c_void_p),
                                csdstrs.ctypes.data_as (ctypes.c_void_p),
                                ctypes.c_size_t (nstr),
                                ctypes.c_int (norb),
                                ctypes.c_int (neleca), ctypes.c_int (nelecb))
    return ddstrs

def pretty_csdaddrs (norb, neleca, nelecb, csdaddrs):
    ''' Printable string for csd strings based on their addresses '''
    csdstrs = csdaddrs2csdstrs (norb, neleca, nelecb, csdaddrs)
    output = []
    for i in range (len (csdstrs[0])):
        k = 0
        l = 0
        out = ''
        dconf_str = bin (csdstrs[1,i])[2:]
        npair = dconf_str.count ('1')
        while len (dconf_str) < norb:
            dconf_str = '0' + dconf_str
        sconf_str = bin (csdstrs[2,i])[2:]
        nspin = sconf_str.count ('1')
        while len (sconf_str) < norb - npair:
            sconf_str = '0' + sconf_str
        spins_str = bin (csdstrs[3,i])[2:]
        while len (spins_str) < nspin:
            spins_str = '0' + spins_str
        for j in range (len (dconf_str)):
            if dconf_str[j] == '1':
                out = out + '2'
            else:
                if sconf_str[k] == '1':
                    if spins_str[l] == '1':
                        out = out + 'a'
                    else:
                        out = out + 'b'
                    l = l + 1
                else:
                    out = out + '0'
                k = k + 1
        output.append (out)
    return (output)

def format_ddaddrs (norb, neleca, nelecb, ddaddrs):
    ''' Represent as a 2darray with shape (2,*), given ddaddrs passed as 2darray with shape (*,2) or 1darray with
        shape (*) '''
    ddaddrs = np.asarray (ddaddrs, dtype=np.int32)
    ndeta = int (round (special.binom (norb, neleca)))
    ndetb = int (round (special.binom (norb, nelecb)))
    assert (len (ddaddrs.shape) < 3), ddaddrs.shape
    if len (ddaddrs.shape) == 2:
        assert (2 in ddaddrs.shape), ddaddrs.shape
        if ddaddrs.shape[0] == 2 and ddaddrs.flags['C_CONTIGUOUS']:
            new_ddaddrs = ddaddrs
        else:
            ravelorder = tuple(('C', 'F'))[ddaddrs.shape.index (2)]
            new_ddaddrs = np.ravel (ddaddrs, order=ravelorder).reshape (2, -1)
    else:
        assert (np.all (ddaddrs < ndeta*ndetb))
        new_ddaddrs = np.empty ((2,len (ddaddrs)), dtype=np.int32)
        new_ddaddrs[0,:] = ddaddrs // ndetb
        new_ddaddrs[1,:] = ddaddrs  % ndetb
    assert (new_ddaddrs.shape[0] == 2)
    assert (np.all (new_ddaddrs[0,:] < ndeta)), new_ddaddrs
    assert (np.all (new_ddaddrs[1,:] < ndetb)), new_ddaddrs
    assert (new_ddaddrs.flags['C_CONTIGUOUS'])
    return new_ddaddrs

def unpack_confaddrs (norb, neleca, nelecb, addrs):
    ''' Unpack an address for a configuration into npair, domo addrs, and somo addrs '''
    min_npair, npair_offset, npair_dconf_size, npair_sconf_size, npair_spins_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    npair = np.zeros (len (addrs))
    domo_addrs = np.zeros (len (addrs))
    somo_addrs = np.zeros (len (addrs))
    for ix, addr in enumerate (addrs):
        npair[ix] = np.where (npair_offset <= addr)[0][-1]
        domo_addrs[ix] = (addr - npair_offset[npair[ix]]) // npair_dconf_size
        somo_addrs[ix] = (addr - npair_offset[npair[ix]]) % npair_dconf_size
        npair[ix] += min_npair
    return npair, domo_addrs, somo_addrs







