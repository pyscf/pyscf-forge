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
from pyscf.csf_fci.csf import unpack_h1e_ab, _debug_g2e

libfci = lib.load_library('libfci')
libcsf = lib.load_library('libcsf')

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

    npsp_det = len(det_addr)
    npsp_csf = len(csf_addr)

    lib.logger.debug (fci, ("csf.pspace: Lowest-energy %s CSFs correspond to %s configurations"
        " which are spanned by %s determinants"), npsp_csf, econf_addr.size, npsp_det)

    addra, addrb = divmod(det_addr, nb)
    stra = cistring.addrs2str(norb, neleca, addra)
    strb = cistring.addrs2str(norb, nelecb, addrb)
    safety_factor = 1.2
    nfloats_h0 = (npsp_det+npsp_csf)**2.0
    mem_h0 = safety_factor * nfloats_h0 * np.dtype (float).itemsize / 1e6
    deltam = lib.current_memory ()[0] - m0
    mem_remaining = max_memory - deltam
    memstr = ("pspace_size of {} CSFs -> {} determinants requires {} MB, cf {} MB "
              "remaining memory").format (npsp_csf, npsp_det, mem_h0, mem_remaining)
    if mem_h0 > mem_remaining:
        raise MemoryError (memstr)
    lib.logger.debug (fci, memstr)
    h0 = np.ascontiguousarray(np.zeros((npsp_det,npsp_det), dtype=np.float64))
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
    lib.logger.debug1 (fci, "csf_solver.pspace: asked for %s-CSF pspace; found %s CSFs",
                       csf_addr.size, npsp_csf)
    if csf_addr.size > npsp_csf:
        try:
            csf_addr_2 = np.argpartition(np.diag (h0), npsp_csf-1)[:npsp_csf]
        except AttributeError:
            csf_addr_2 = np.argsort(np.diag (h0))[:npsp_csf]
        csf_addr = csf_addr[csf_addr_2]
        h0 = h0[np.ix_(csf_addr_2,csf_addr_2)]

    t0 = lib.logger.timer_debug1 (fci, "csf.pspace wrapup", *t0)
    return csf_addr, h0

