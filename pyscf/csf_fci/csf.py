import numpy as np
import scipy
import ctypes
import time
from pyscf import lib, ao2mo, __config__
from pyscf.fci import direct_spin1, cistring, direct_uhf
from pyscf.fci.direct_spin1 import _unpack, _unpack_nelec, _get_init_guess, kernel_ms1
from pyscf.lib.numpy_helper import tag_array
from pyscf.csf_fci.csdstring import get_csdaddrs_shape
from pyscf.csf_fci.csfstring import count_all_csfs, get_spin_evecs
from pyscf.csf_fci.csfstring import get_csfvec_shape
from pyscf.csf_fci.csfstring import CSFTransformer
'''
    MRH 03/24/2019
    IMPORTANT: this solver will interpret a two-component one-body Hamiltonian as [h1e_charge, h1e_spin] where
    h1e_charge = h^p_q (a'_p,up a_q,up + a'_p,down a_q,down)
    h1e_spin   = h^p_q (a'_p,up a_q,up - a'_p,down a_q,down)
    This is to preserve interoperability with the members of direct_spin1_symm, since there is no direct_uhf_symm in
    pyscf yet. Only with an explicitly CSF-based solver can such potentials be included in a calculation that retains
    S^2 symmetry. Multicomponent two-body integrals are currently not available (so this feature is only for use with,
    e.g., ROHF-CASSCF with with some SOMOs outside of the active space or LASSCF with multiple nonsinglet fragments,
    not UHF-CASSCF).
'''

libfci = lib.load_library('libfci')
libcsf = lib.load_library('libcsf')

def unpack_h1e_ab (h1e):
    h = np.asarray (h1e)
    if h.ndim == 3 and h.shape[0] == 2:
        return h1e[0], h1e[1]
    return h1e, h1e

def unpack_1RDM_ab (dm1):
    dm1 = np.asarray (dm1)
    if dm1.ndim == 3 and dm1.shape[0] == 2:
        return dm1[0], dm1[1]
    return dm1/2, dm1/2

def unpack_h1e_cs (h1e):
    h1e_a, h1e_b = unpack_h1e_ab (h1e)
    h1e_c = (h1e_a + h1e_b) / 2.0
    h1e_s = (h1e_a - h1e_b) / 2.0
    return h1e_c, h1e_s

def unpack_1RDM_cs (dm):
    dma, dmb = unpack_1RDM_ab (dm)
    return dma + dmb, dma - dmb


def get_init_guess(norb, nelec, nroots, hdiag_csf, transformer):
    ''' The existing _get_init_guess function will work in the csf basis if I pass it with na, nb = ncsf, 1.
    This might change in future PySCF versions though.

    ...For point-group symmetry, I pass the direct_spin1.py version of _get_init_guess with na, nb = ncsf_sym, 1 and
    hdiag_csf including only csfs of the right point-group symmetry.
    This should clean up the symmetry-breaking "noise" in direct_spin1_symm.py! '''
    neleca, nelecb = _unpack_nelec (nelec)
    ncsf_sym = transformer.ncsf
    assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs of symmetry {}".format (
        nroots, ncsf_sym, transformer.wfnsym)
    hdiag_csf = transformer.pack_csf (hdiag_csf)
    ci = _get_init_guess (ncsf_sym, 1, nroots, hdiag_csf, nelec)
    return transformer.vec_csf2det (ci)

def make_hdiag_det (fci, h1e, eri, norb, nelec):
    ''' Wrap to the uhf version in order to use two-component h1e '''
    return direct_uhf.make_hdiag (unpack_h1e_ab (h1e), [eri, eri, eri], norb, nelec)

def make_hdiag_csf (h1e, eri, norb, nelec, transformer, hdiag_det=None, max_memory=None):
    m0 = lib.current_memory ()[0]
    smult = transformer.smult
    if hdiag_det is None:
        hdiag_det = make_hdiag_det (None, h1e, eri, norb, nelec)
    eri = ao2mo.restore(1, eri, norb)
    tlib = wlib = 0
    neleca, nelecb = _unpack_nelec (nelec)
    min_npair, npair_csd_offset, npair_dconf_size, npair_sconf_size, npair_sdet_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    _, npair_csf_offset, _, _, npair_csf_size = get_csfvec_shape (norb, neleca, nelecb, smult)
    npair_econf_size = npair_dconf_size * npair_sconf_size
    max_npair = min (neleca, nelecb)
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    #ndeta_all = cistring.num_strings(norb, neleca)
    ndetb_all = cistring.num_strings(norb, nelecb)
    hdiag_csf = np.ascontiguousarray (np.zeros (ncsf_all, dtype=np.float64))
    hdiag_csf_check = np.ones (ncsf_all, dtype=np.bool_)
    for npair in range (min_npair, max_npair+1):
        ipair = npair - min_npair
        nconf = int (npair_econf_size[ipair])
        ndet = int (npair_sdet_size[ipair])
        ncsf = int (npair_csf_size[ipair])
        if ncsf == 0:
            continue
        csd_offset = npair_csd_offset[ipair]
        det_addr = transformer.csd_mask[csd_offset:][:nconf*ndet]
        # mem safety
        deltam = lib.current_memory ()[0] - m0
        mem_remaining = max_memory - deltam
        safety_factor = 1.2
        nfloats = float(nconf)*ndet*ndet + float(det_addr.size)
        mem_floats = nfloats * np.dtype (float).itemsize / 1e6
        mem_ints = det_addr.dtype.itemsize * det_addr.size * 3 / 1e6
        mem = safety_factor * (mem_floats + mem_ints)
        memstr = ("hdiag_csf of {} orbitals, ({},{}) electrons and smult={} with {} "
                  "doubly-occupied orbitals ({} configurations and {} determinants) requires {} "
                  "MB > {} MB remaining of {} MB max").format (
            norb, neleca, nelecb, smult, npair, nconf, ndet, mem, mem_remaining, max_memory)
        if mem > mem_remaining:
            raise MemoryError (memstr)
        # end mem safety
        nspin = neleca + nelecb - 2*npair
        csf_offset = npair_csf_offset[ipair]
        hdiag_conf = np.ascontiguousarray (np.zeros ((nconf, ndet, ndet), dtype=np.float64))
        if ndet == 1:
            # Closed-shell singlets
            assert (ncsf == 1)
            hdiag_csf[csf_offset:][:nconf] = hdiag_det[det_addr.flat]
            hdiag_csf_check[csf_offset:][:nconf] = False
            continue
        det_addra, det_addrb = divmod (det_addr, ndetb_all)
        det_stra = np.ascontiguousarray (cistring.addrs2str (norb, neleca, det_addra).reshape (nconf, ndet, order='C'))
        det_strb = np.ascontiguousarray (cistring.addrs2str (norb, nelecb, det_addrb).reshape (nconf, ndet, order='C'))
        det_addr = det_addr.reshape (nconf, ndet, order='C')
        hdiag_conf = np.ascontiguousarray (np.zeros ((nconf, ndet, ndet), dtype=np.float64))
        hdiag_conf_det = np.ascontiguousarray (hdiag_det[det_addr], dtype=np.float64)
        t1 = lib.logger.process_clock ()
        w1 = lib.logger.perf_counter ()
        libcsf.FCICSFhdiag (hdiag_conf.ctypes.data_as (ctypes.c_void_p),
                            hdiag_conf_det.ctypes.data_as (ctypes.c_void_p),
                            eri.ctypes.data_as (ctypes.c_void_p),
                            det_stra.ctypes.data_as (ctypes.c_void_p),
                            det_strb.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_uint (norb), ctypes.c_size_t (nconf), ctypes.c_size_t (ndet))
        tlib += lib.logger.process_clock () - t1
        wlib += lib.logger.perf_counter () - w1
        umat = get_spin_evecs (nspin, neleca, nelecb, smult)
        hdiag_conf = np.tensordot (hdiag_conf, umat, axes=1)
        hdiag_conf *= umat[np.newaxis,:,:]
        hdiag_csf[csf_offset:][:nconf*ncsf] = hdiag_conf.sum (1).ravel (order='C')
        hdiag_csf_check[csf_offset:][:nconf*ncsf] = False
    assert (np.count_nonzero (hdiag_csf_check) == 0), np.count_nonzero (hdiag_csf_check)
    #print ("Time in hdiag_csf library: {}, {}".format (tlib, wlib))
    return hdiag_csf


def make_hdiag_csf_slower (h1e, eri, norb, nelec, transformer, hdiag_det=None, max_memory=None):
    ''' This is tricky because I need the diagonal blocks for each configuration in order to get
    the correct csf hdiag values, not just the diagonal elements for each determinant. '''
    smult = transformer.smult
    #t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
    #tstr = tlib = tloop = wstr = wlib = wloop = 0
    if hdiag_det is None:
        hdiag_det = make_hdiag_det (None, h1e, eri, norb, nelec)
    eri = ao2mo.restore(1, eri, norb)
    neleca, nelecb = _unpack_nelec (nelec)
    min_npair, npair_csd_offset, npair_dconf_size, npair_sconf_size, npair_sdet_size = get_csdaddrs_shape (
        norb, neleca, nelecb)
    _, npair_csf_offset, _, _, npair_csf_size = get_csfvec_shape (norb, neleca, nelecb, smult)
    npair_econf_size = npair_dconf_size * npair_sconf_size
    max_npair = min (neleca, nelecb)
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    #ndeta_all = cistring.num_strings(norb, neleca)
    ndetb_all = cistring.num_strings(norb, nelecb)
    hdiag_csf = np.ascontiguousarray (np.zeros (ncsf_all, dtype=np.float64))
    hdiag_csf_check = np.ones (ncsf_all, dtype=np.bool_)
    for npair in range (min_npair, max_npair+1):
        ipair = npair - min_npair
        nconf = npair_econf_size[ipair]
        ndet = npair_sdet_size[ipair]
        ncsf = npair_csf_size[ipair]
        if ncsf == 0:
            continue
        nspin = neleca + nelecb - 2*npair
        csd_offset = npair_csd_offset[ipair]
        csf_offset = npair_csf_offset[ipair]
        hdiag_conf = np.ascontiguousarray (np.zeros ((nconf, ndet, ndet), dtype=np.float64))
        det_addr = transformer.csd_mask[csd_offset:][:nconf*ndet]
        if ndet == 1:
            # Closed-shell singlets
            assert (ncsf == 1)
            hdiag_csf[csf_offset:][:nconf] = hdiag_det[det_addr.flat]
            hdiag_csf_check[csf_offset:][:nconf] = False
            continue
        umat = get_spin_evecs (nspin, neleca, nelecb, smult)
        det_addra, det_addrb = divmod (det_addr, ndetb_all)
        #t1, w1 = lib.logger.process_clock (), lib.logger.perf_counter ()
        det_stra = cistring.addrs2str (norb, neleca, det_addra).reshape (nconf, ndet, order='C')
        det_strb = cistring.addrs2str (norb, nelecb, det_addrb).reshape (nconf, ndet, order='C')
        #tstr += lib.logger.process_clock () - t1
        #wstr += lib.logger.perf_counter () - w1
        det_addr = det_addr.reshape (nconf, ndet, order='C')
        diag_idx = np.diag_indices (ndet)
        # It looks like the library call below is, itself, usually responsible for about 50% of the
        # clock and wall time that this function consumes.
        #t1, w1 = lib.logger.process_clock (), lib.logger.perf_counter ()
        for iconf in range (nconf):
            addr = det_addr[iconf]
            assert (len (addr) == ndet)
            stra = det_stra[iconf]
            strb = det_strb[iconf]
            #t2, w2 = lib.logger.process_clock (), lib.logger.perf_counter ()
            libfci.FCIpspace_h0tril(hdiag_conf[iconf].ctypes.data_as(ctypes.c_void_p),
                h1e.ctypes.data_as(ctypes.c_void_p),
                eri.ctypes.data_as(ctypes.c_void_p),
                stra.ctypes.data_as(ctypes.c_void_p),
                strb.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(norb), ctypes.c_int(ndet))
            #tlib += lib.logger.process_clock () - t2
            #wlib += lib.logger.perf_counter () - w2
            #hdiag_conf[iconf][diag_idx] = hdiag_det[addr]
            #hdiag_conf[iconf] = lib.hermi_triu(hdiag_conf[iconf])
        for iconf in range (nconf): hdiag_conf[iconf] = lib.hermi_triu (hdiag_conf[iconf])
        for iconf in range (nconf): hdiag_conf[iconf][diag_idx] = hdiag_det[det_addr[iconf]]
        #tloop += lib.logger.process_clock () - t1
        #wloop += lib.logger.perf_counter () - w1

        hdiag_conf = np.tensordot (hdiag_conf, umat, axes=1)
        hdiag_conf = (hdiag_conf * umat[np.newaxis,:,:]).sum (1)
        hdiag_csf[csf_offset:][:nconf*ncsf] = hdiag_conf.ravel (order='C')
        hdiag_csf_check[csf_offset:][:nconf*ncsf] = False
    assert (np.count_nonzero (hdiag_csf_check) == 0), np.count_nonzero (hdiag_csf_check)
    #print ("Total time in hdiag_csf: {}, {}".format (lib.logger.process_clock ()-t0, lib.logger.perf_counter ()-w0))
    #print ("    Loop: {}, {}".format (tloop, wloop))
    #print ("    Library: {}, {}".format (tlib, wlib))
    #print ("    Cistring: {}, {}".format (tstr, wstr))
    return hdiag_csf

# Exploring g2e nan bug; remove later?
def _debug_g2e (fci, g2e, eri, norb):
    g2e_ninf = np.count_nonzero (np.isinf (g2e))
    g2e_nnan = np.count_nonzero (np.isnan (g2e))
    if (g2e_ninf == 0) and (g2e_nnan == 0): return
    lib.logger.note (fci, 'ERROR: g2e has {} infs and {} nans (norb = {}; shape = {})'.format (
        g2e_ninf, g2e_nnan, norb, g2e.shape))
    lib.logger.note (fci, 'type (eri) = {}'.format (type (eri)))
    lib.logger.note (fci, 'eri.shape = {}'.format (eri.shape))
    lib.logger.note (fci, 'eri.dtype = {}'.format (eri.dtype))
    eri_ninf = np.count_nonzero (np.isinf (eri))
    eri_nnan = np.count_nonzero (np.isnan (eri))
    lib.logger.note (fci, 'eri has {} infs and {} nans'.format (eri_ninf, eri_nnan))
    raise ValueError ('g2e has {} infs and {} nans (norb = {}; shape = {})'.format (
        g2e_ninf, g2e_nnan, norb, g2e.shape))
    return

def pspace (fci, h1e, eri, norb, nelec, transformer, hdiag_det=None, hdiag_csf=None, npsp=200, max_memory=None):
    ''' Note that getting pspace for npsp CSFs is substantially more costly than getting it for npsp determinants,
    until I write code than can evaluate Hamiltonian matrix elements of CSFs directly. On the other hand
    a pspace of determinants contains many redundant degrees of freedom for the same reason. Therefore I have
    reduced the default pspace size by a factor of 2.'''
    m0 = lib.current_memory ()[0]
    if norb > 63:
        raise NotImplementedError('norb > 63')
    if max_memory is None: max_memory=fci.max_memory

    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    neleca, nelecb = _unpack_nelec(nelec)
    h1e = np.ascontiguousarray(h1e)
    eri = ao2mo.restore(1, eri, norb)
    nb = cistring.num_strings(norb, nelecb)
    if hdiag_det is None:
        hdiag_det = fci.make_hdiag(h1e, eri, norb, nelec)
    if hdiag_csf is None:
        hdiag_csf = fci.make_hdiag_csf(h1e, eri, norb, nelec, hdiag_det=hdiag_det, max_memory=max_memory)
    csf_addr = np.arange (hdiag_csf.size, dtype=np.int32)
    if transformer.wfnsym is None:
        ncsf_sym = hdiag_csf.size
    else:
        idx_sym = transformer.confsym[transformer.econf_csf_mask] == transformer.wfnsym
        ncsf_sym = np.count_nonzero (idx_sym)
        csf_addr = csf_addr[idx_sym]
    if ncsf_sym > npsp:
        try:
            csf_addr = csf_addr[np.argpartition(hdiag_csf[csf_addr], npsp-1)[:npsp]]
        except AttributeError:
            csf_addr = csf_addr[np.argsort(hdiag_csf[csf_addr])[:npsp]]

    # To build
    econf_addr = np.unique (transformer.econf_csf_mask[csf_addr])
    det_addr = np.concatenate ([np.nonzero (transformer.econf_det_mask == conf)[0]
        for conf in econf_addr])
    lib.logger.debug (fci, ("csf.pspace: Lowest-energy %s CSFs correspond to %s configurations"
        " which are spanned by %s determinants"), npsp, econf_addr.size, det_addr.size)

    addra, addrb = divmod(det_addr, nb)
    stra = cistring.addrs2str(norb, neleca, addra)
    strb = cistring.addrs2str(norb, nelecb, addrb)
    npsp_det = len(det_addr)
    safety_factor = 1.2
    nfloats_h0 = (npsp_det+npsp)**2.0
    mem_h0 = safety_factor * nfloats_h0 * np.dtype (float).itemsize / 1e6
    deltam = lib.current_memory ()[0] - m0
    mem_remaining = max_memory - deltam
    memstr = ("pspace_size of {} CSFs -> {} determinants requires {} MB, cf {} MB "
              "remaining memory").format (npsp, npsp_det, mem_h0, mem_remaining)
    if mem_h0 > mem_remaining:
        raise MemoryError (memstr)
    lib.logger.debug (fci, memstr)
    h0 = np.zeros((npsp_det,npsp_det))
    h1e_ab = unpack_h1e_ab (h1e)
    h1e_a = np.ascontiguousarray(h1e_ab[0])
    h1e_b = np.ascontiguousarray(h1e_ab[1])
    g2e = ao2mo.restore(1, eri, norb)
    g2e_ab = g2e_bb = g2e_aa = g2e
    _debug_g2e (fci, g2e, eri, norb) # Exploring g2e nan bug; remove later?
    t0 = lib.logger.timer_debug1 (fci, "csf.pspace: index manipulation", *t0)
    libfci.FCIpspace_h0tril_uhf(h0.ctypes.data_as(ctypes.c_void_p),
                                h1e_a.ctypes.data_as(ctypes.c_void_p),
                                h1e_b.ctypes.data_as(ctypes.c_void_p),
                                g2e_aa.ctypes.data_as(ctypes.c_void_p),
                                g2e_ab.ctypes.data_as(ctypes.c_void_p),
                                g2e_bb.ctypes.data_as(ctypes.c_void_p),
                                stra.ctypes.data_as(ctypes.c_void_p),
                                strb.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(npsp_det))
    t0 = lib.logger.timer_debug1 (fci, "csf.pspace: pspace Hamiltonian in determinant basis", *t0)

    for i in range(npsp_det):
        h0[i,i] = hdiag_det[det_addr[i]]
    h0 = lib.hermi_triu(h0)

    try:
        if fci.verbose > lib.logger.DEBUG1: evals_before = scipy.linalg.eigh (h0)[0]
    except ValueError as e:
        lib.logger.debug1 (fci, ("ERROR: h0 has {} infs, {} nans; h1e_a has {} infs, {} nans; "
            "h1e_b has {} infs, {} nans; g2e has {} infs, {} nans, norb = {}, npsp_det = {}").format (
            np.count_nonzero (np.isinf (h0)), np.count_nonzero (np.isnan (h0)),
            np.count_nonzero (np.isinf (h1e_a)), np.count_nonzero (np.isnan (h1e_a)),
            np.count_nonzero (np.isinf (h1e_b)), np.count_nonzero (np.isnan (h1e_b)),
            np.count_nonzero (np.isinf (g2e)), np.count_nonzero (np.isnan (g2e)),
            norb, npsp_det))
        evals_before = np.zeros (npsp_det)
        raise (e) from None

    h0, csf_addr = transformer.mat_det2csf_confspace (h0, econf_addr)
    t0 = lib.logger.timer_debug1 (fci, "csf.pspace: transform pspace Hamiltonian into CSF basis", *t0)

    if fci.verbose > lib.logger.DEBUG1:
        lib.logger.debug1 (fci, "csf.pspace: eigenvalues of h0 before transformation %s", evals_before)
        evals_after = scipy.linalg.eigh (h0)[0]
        lib.logger.debug1 (fci, "csf.pspace: eigenvalues of h0 after transformation %s", evals_after)
        idx = [np.argmin (np.abs (evals_before - ev)) for ev in evals_after]
        resid = evals_after - evals_before[idx]
        lib.logger.debug1 (fci, "csf.pspace: best h0 eigenvalue matching differences after transformation: %s", resid)
        lib.logger.debug1 (fci, "csf.pspace: if the transformation of h0 worked the following number will be zero: %s",
                           np.max (np.abs(resid)))

    # We got extra CSFs from building the configurations most of the time.
    if csf_addr.size > npsp:
        try:
            csf_addr_2 = np.argpartition(np.diag (h0), npsp-1)[:npsp]
        except AttributeError:
            csf_addr_2 = np.argsort(np.diag (h0))[:npsp]
        csf_addr = csf_addr[csf_addr_2]
        h0 = h0[np.ix_(csf_addr_2,csf_addr_2)]
    npsp_csf = csf_addr.size
    lib.logger.debug1 (fci, "csf_solver.pspace: asked for %s-CSF pspace; found %s CSFs", npsp, npsp_csf)

    t0 = lib.logger.timer_debug1 (fci, "csf.pspace wrapup", *t0)
    return csf_addr, h0

def kernel(fci, h1e, eri, norb, nelec, smult=None, idx_sym=None, ci0=None,
           tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, pspace_size=None, max_memory=None,
           orbsym=None, wfnsym=None, ecore=0, transformer=None, **kwargs):
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
        kwargs.pop ('verbose')
    else: verbose = lib.logger.Logger (stdout=fci.stdout, verbose=fci.verbose)
    if ((isinstance (verbose, lib.logger.Logger) and verbose.verbose >= lib.logger.WARN)
        or (isinstance (verbose, int) and verbose >= lib.logger.WARN)):
        fci.check_sanity()
    if nroots is None: nroots = fci.nroots
    if pspace_size is None: pspace_size = fci.pspace_size
    if davidson_only is None: davidson_only = fci.davidson_only
    if transformer is None: transformer = fci.transformer
    if max_memory is None: max_memory = fci.max_memory
    nelec = _unpack_nelec(nelec, fci.spin)
    neleca, nelecb = nelec
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    hdiag_det = fci.make_hdiag (h1e, eri, norb, nelec)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: hdiag_det", *t0)
    hdiag_csf = fci.make_hdiag_csf (h1e, eri, norb, nelec, hdiag_det=hdiag_det, max_memory=max_memory)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: hdiag_csf", *t0)
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    if idx_sym is None:
        ncsf_sym = ncsf_all
    else:
        ncsf_sym = np.count_nonzero (idx_sym)
    nroots = min(ncsf_sym, nroots)
    if nroots is not None:
        assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsf_sym)
    link_indexa, link_indexb = _unpack(norb, nelec, None)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    addr, h0 = fci.pspace(h1e, eri, norb, nelec, idx_sym=idx_sym, hdiag_det=hdiag_det, hdiag_csf=hdiag_csf,
                          npsp=max(pspace_size,nroots))
    lib.logger.debug1 (fci, 'csf.kernel: error of hdiag_csf: %s', np.amax (np.abs (hdiag_csf[addr]-np.diag (h0))))
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make pspace", *t0)
    if pspace_size > 0:
        pw, pv = fci.eig (h0)
    else:
        pw = pv = None

    # MRH 05/01/2023: apparently the code has never been here until now, because the indexing down
    # there was obviously messed up...
    if pspace_size >= ncsf_sym and not davidson_only:
        if ncsf_sym == 1:
            civec = transformer.vec_csf2det (pv[:,0].reshape (1,1))
            return pw[0]+ecore, civec
        elif nroots > 1:
            civec = np.empty((nroots,ncsf_sym))
            civec[:,:] = pv[:,:nroots].T
            civec = transformer.vec_csf2det (civec)
            return pw[:nroots]+ecore, [c.reshape(na,nb) for c in civec]
        elif abs(pw[0]-pw[1]) > 1e-12:
            civec = np.empty((ncsf_sym))
            civec[:] = pv[:,0]
            civec = transformer.vec_csf2det (civec)
            return pw[0]+ecore, civec.reshape(na,nb)

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    if idx_sym is None:
        precond = fci.make_precond(hdiag_csf, pw, pv, addr)
    else:
        addr_bool = np.zeros (ncsf_all, dtype=np.bool_)
        addr_bool[addr] = True
        precond = fci.make_precond(hdiag_csf[idx_sym], pw, pv, addr_bool[idx_sym])
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make preconditioner", *t0)
    '''
    fci.eci, fci.ci = \
            kernel_ms1(fci, h1e, eri, norb, nelec, ci0, None,
                       tol, lindep, max_cycle, max_space, nroots,
                       davidson_only, pspace_size, ecore=ecore, **kwargs)
    '''
    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: h2e", *t0)
    def hop(x):
        x_det = transformer.vec_csf2det (x)
        hx = fci.contract_2e(h2e, x_det, norb, nelec, (link_indexa,link_indexb))
        return transformer.vec_det2csf (hx, normalize=False).ravel ()

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make hop", *t0)
    if ci0 is None:
        if hasattr(fci, 'get_init_guess'):
            def ci0 ():
                return transformer.vec_det2csf (fci.get_init_guess(norb, nelec, nroots, hdiag_csf))


        else:
            def ci0():  # lazy initialization to reduce memory footprint
                x0 = []
                for i in range(nroots):
                    x = np.zeros(ncsf_sym)
                    x[addr[i]] = 1
                    x0.append(x)
                return x0
    else:
        if isinstance(ci0, np.ndarray) and ci0.size == na*nb:
            ci0 = [transformer.vec_det2csf (ci0.ravel ())]
        else:
            nrow = len (ci0)
            ci0 = np.asarray (ci0).reshape (nrow, -1, order='C')
            ci0 = np.ascontiguousarray (ci0)
            if nrow==1: ci0 = ci0[0]
            ci0 = transformer.vec_det2csf (ci0)
            ci0 = [c for c in ci0.reshape (nrow, -1)]
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: ci0 handling", *t0)

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    #with lib.with_omp_threads(fci.threads):
    #    e, c = lib.davidson(hop, ci0, precond, tol=fci.conv_tol, lindep=fci.lindep)
    e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=verbose, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: running fci.eig", *t0)
    c = transformer.vec_csf2det (c, order='C')
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: transforming final ci vector", *t0)
    if nroots > 1:
        return e+ecore, [ci.reshape(na,nb) for ci in c]
    else:
        return e+ecore, c.reshape(na,nb)

class CSFFCISolver: # parent class
    _keys = {'smult', 'transformer'}
    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)
    make_hdiag = make_hdiag_det

    def __init__(self, mol=None, smult=None):
        self.smult = smult
        self.transformer = None
        super().__init__(mol)

    def make_hdiag_csf (self, h1e, eri, norb, nelec, hdiag_det=None, smult=None, max_memory=None):
        self.norb = norb
        self.nelec = nelec
        if smult is not None:
            self.smult = smult
        self.check_transformer_cache ()
        if max_memory is None: max_memory = self.max_memory
        return make_hdiag_csf (h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det, max_memory=max_memory)

    def absorb_h1e (self, h1e, eri, norb, nelec, fac=1):
        h1e_c, h1e_s = unpack_h1e_cs (h1e)
        h2eff = super().absorb_h1e (h1e_c, eri, norb, nelec, fac)
        if h1e_s is not None:
            h2eff = tag_array (h2eff, h1e_s=h1e_s)
        return h2eff

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        hc = super().contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)
        if hasattr (eri, 'h1e_s'):
            hc += direct_uhf.contract_1e ([eri.h1e_s, -eri.h1e_s], fcivec, norb, nelec, link_index)
        return hc

    def pspace (self, h1e, eri, norb, nelec, hdiag_det=None, hdiag_csf=None, npsp=200, **kwargs):
        self.norb = norb
        self.nelec = nelec
        if 'smult' in kwargs:
            self.smult = kwargs['smult']
            kwargs.pop ('smult')
        self.check_transformer_cache ()
        max_memory = kwargs.get ('max_memory', self.max_memory)
        return pspace (self, h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det,
            hdiag_csf=hdiag_csf, npsp=npsp, max_memory=max_memory)

    def log_transformer_cache (self, tverbose=0, **kwargs):
        if len (kwargs):
            self.__dict__.update (kwargs)
            self.check_transformer_cache ()
        if self.transformer is None:
            return
        log = lib.logger.new_logger (self, self.verbose)
        noprint = lambda *args, **kwargs: None
        printer = (noprint, log.error, log.warn, log.note, log.info, log.debug, log.debug1,
                   log.debug2, log.debug3, log.debug4, print)[tverbose]
        self.transformer.print_config (printer)

    def print_transformer_cache (self, **kwargs):
        return self.log_transformer_cache (10, **kwargs)

class FCISolver (CSFFCISolver, direct_spin1.FCISolver):
    r''' get_init_guess uses csfstring.py and csdstring.py to construct a spin-symmetry-adapted initial guess, and the
    Davidson algorithm is carried out in the CSF basis. However, the ci attribute is put in the determinant basis at the
    end of it all, and "ci0" is also assumed to be in the determinant basis.'''

    def get_init_guess(self, norb, nelec, nroots, hdiag_csf, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.check_transformer_cache ()
        return get_init_guess (norb, nelec, nroots, hdiag_csf, self.transformer)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        self.norb = norb
        self.nelec = nelec
        if 'smult' in kwargs:
            self.smult = kwargs['smult']
            kwargs.pop ('smult')
        self.check_transformer_cache ()
        self.log_transformer_cache (lib.logger.DEBUG)
        e, c = kernel (self, h1e, eri, norb, nelec, smult=self.smult,
            idx_sym=None, ci0=ci0, transformer=self.transformer, **kwargs)
        self.eci, self.ci = e, c
        return e, c

    def check_transformer_cache (self):
        assert (isinstance (self.smult, (int, np.number)))
        neleca, nelecb = _unpack_nelec (self.nelec)
        if self.transformer is None:
            self.transformer = CSFTransformer (self.norb, neleca, nelecb, self.smult,
                                               max_memory=self.max_memory)
        else:
            self.transformer._update_spin_cache (self.norb, neleca, nelecb, self.smult)
