import numpy as np
import sys, os, time
import ctypes
from pyscf.csf_fci import csdstring
from pyscf.fci import cistring
from pyscf.fci.spin_op import spin_square0
from pyscf import lib
from pyscf.lib import numpy_helper, param
from scipy import special, linalg
from functools import reduce
from pyscf.fci.direct_spin1_symm import _gen_strs_irrep
libcsf = lib.load_library ('libcsf')

class ImpossibleCIvecError (RuntimeError):
    def __init__(self, message, ndet=None, ncsf=None, norb=None, neleca=None, nelecb=None):
        self.message = message
        self.ndet = ndet
        self.ncsf = ncsf
        self.norb = norb
        self.neleca = neleca
        self.nelecb = nelecb

    def __str__(self):
        return self.message

class ImpossibleSpinError (ImpossibleCIvecError):
    def __init__(self, message, ndet=None, ncsf=None, norb=None, neleca=None, nelecb=None, smult=None):
        ImpossibleCIvecError.__init__(self, message, ndet=ndet, ncsf=ncsf, norb=norb,
                                      neleca=neleca, nelecb=nelecb)
        self.smult = smult

class CSFTransformer (lib.StreamObject):
    def __init__(self, norb, neleca, nelecb, smult, orbsym=None, wfnsym=None,
                 max_memory=param.MAX_MEMORY):
        self._norb = self._neleca = self._nelecb = self._smult = self._orbsym = None
        self.wfnsym = wfnsym
        self._update_spin_cache (norb, neleca, nelecb, smult)
        self.orbsym = orbsym
        self.max_memory = max_memory

    def project_civec (self, detarr, order='C', normalize=True, return_norm=False):
        raise NotImplementedError

    def vec_det2csf (self, civec, order='C', normalize=True, return_norm=False):
        vec_on_cols = (order.upper () == 'F')
        civec, norm = transform_civec_det2csf (civec, self._norb, self._neleca,
            self._nelecb, self._smult, csd_mask=self.csd_mask, do_normalize=normalize,
            vec_on_cols=vec_on_cols, max_memory=self.max_memory)
        civec = self.pack_csf (civec, order=order)
        if return_norm: return civec, norm
        return civec

    def vec_csf2det (self, civec, order='C', normalize=True, return_norm=False):
        vec_on_cols = (order.upper () == 'F')
        civec, norm = transform_civec_csf2det (self.unpack_csf (civec), self._norb, self._neleca,
            self._nelecb, self._smult, csd_mask=self.csd_mask, do_normalize=normalize,
            vec_on_cols=vec_on_cols, max_memory=self.max_memory)
        if return_norm: return civec, norm
        return civec

    def mat_det2csf (self, mat):
        raise NotImplementedError

    def mat_csf2det (self, mat):
        raise NotImplementedError

    def mat_det2csf_confspace (self, mat, confs):
        mat, csf_addr = transform_opmat_det2csf_pspace (mat, confs, self._norb, self._neleca,
            self._nelecb, self._smult, self.csd_mask, self.econf_det_mask, self.econf_csf_mask,
            max_memory=self.max_memory)
        return mat, csf_addr

    def pack_csf (self, csfvec, order='C'):
        if self.wfnsym is None or self._orbsym is None:
            return csfvec
        vec_on_cols = (order.upper () == 'F')
        idx_sym = (self.confsym[self.econf_csf_mask] == self.wfnsym)
        return pack_sym_ci (csfvec, idx_sym, vec_on_cols=vec_on_cols)

    def unpack_csf (self, csfvec, order='C'):
        if self.wfnsym is None or self._orbsym is None:
            return csfvec
        vec_on_cols = (order.upper () == 'F')
        idx_sym = (self.confsym[self.econf_csf_mask] == self.wfnsym)
        return unpack_sym_ci (csfvec, idx_sym, vec_on_cols=vec_on_cols)

    def pack_det (self, detvec, order='C'):
        if self.wfnsym is None or self._orbsym is None:
            return detvec
        vec_on_cols = (order.upper () == 'F')
        idx_sym = (self.confsym[self.econf_det_mask] == self.wfnsym)
        detvec = np.asarray (detvec)
        if detvec.shape[-2:] == (self.ndeta, self.ndetb):
            detvec = np.squeeze (detvec.reshape (-1, self.ndet))
        return pack_sym_ci (detvec, idx_sym, vec_on_cols=vec_on_cols)

    def unpack_det (self, detvec, order='C'):
        if self.wfnsym is None or self._orbsym is None:
            return detvec
        vec_on_cols = (order.upper () == 'F')
        idx_sym = (self.confsym[self.econf_det_mask] == self.wfnsym)
        return unpack_sym_ci (detvec, idx_sym, vec_on_cols=vec_on_cols)

    def _update_spin_cache (self, norb, neleca, nelecb, smult):
        if any ([self._norb != norb, self._neleca != neleca, self._nelecb != nelecb, self._smult != smult]):
            self.csd_mask = csdstring.make_csd_mask (norb, neleca, nelecb)
            self.econf_det_mask = csdstring.make_econf_det_mask (norb, neleca, nelecb, self.csd_mask)
            self.econf_csf_mask = make_econf_csf_mask (norb, neleca, nelecb, smult)
            self._norb = norb
            self._neleca = neleca
            self._nelecb = nelecb
            self._smult = smult
            if self._orbsym is not None:
                self.confsym = make_confsym (self.norb, self.neleca, self.nelecb, self.econf_det_mask, self._orbsym)

    def _update_symm_cache (self, orbsym):
        if (orbsym is not None) and (self._orbsym is None or np.any (orbsym != self._orbsym)):
            self.confsym = make_confsym (self.norb, self.neleca, self.nelecb, self.econf_det_mask, orbsym)
        self._orbsym = orbsym

    def printable_largest_csf (self, csfvec, npr, order='C', isdet=False, normalize=True):
        if isdet:
            csfvec = self.vec_det2csf (csfvec, order=order, normalize=normalize)
        csfvec = self.unpack_csf (csfvec, order=order) # Don't let symmetry scramble CSF indexing
        csfvec = np.asarray (csfvec)
        if csfvec.ndim == 1:
            nvec = 1
            ncsf = csfvec.size
            csfvec_list = [csfvec]
        elif order.upper () == 'C':
            nvec, ncsf = csfvec.shape
            csfvec_list = [csfvec[i,:] for i in range (nvec)]
        elif order.upper () == 'F':
            ncsf, nvec = csfvec.shape
            csfvec_list = [csfvec[:,i] for i in range (nvec)]
        else:
            raise RuntimeError ('order must be "C" or "F"')
        npr = min (ncsf, npr)
        idx_sort = [np.argsort (-np.abs (ivec))[:npr] for ivec in csfvec_list]
        csfvec_list = [ivec[idx] for ivec, idx in zip (csfvec_list, idx_sort)]
        printable = [printable_csfstring (self._norb, self._neleca, self._nelecb, self._smult, idx) for idx in idx_sort]
        csfvec_list = np.stack (csfvec_list, axis=(order.upper () == 'F'))
        printable = np.stack (printable, axis=(order.upper () == 'F'))
        return printable, csfvec_list

    def printable_csfstring (self, idx):
        isarr = hasattr (idx, '__len__')
        idx = np.atleast_1d (idx)
        if (self.wfnsym is not None) and (self._orbsym is not None):
            idx_sym = (self.confsym[self.econf_csf_mask] == self.wfnsym)
            idx = np.argwhere (idx_sym)[idx]
        ret = printable_csfstring (self._norb, self._neleca, self._nelecb, self._smult, idx)
        if not isarr: ret = ret[0]
        return ret

    # Setting the below properties triggers cache updating
    @property
    def norb (self):
        return self._norb
    @norb.setter
    def norb (self, x):
        self._update_spin_cache (x, self._neleca, self._nelecb, self._smult)
        return self._norb

    @property
    def neleca (self):
        return self._neleca
    @neleca.setter
    def neleca (self, x):
        self._update_spin_cache (self._norb, x, self._nelecb, self._smult)
        return self._neleca

    @property
    def nelecb (self):
        return self._nelecb
    @nelecb.setter
    def nelecb (self, x):
        self._update_spin_cache (self._norb, self._neleca, x, self._smult)
        return self._nelecb

    @property
    def smult (self):
        return self._smult
    @smult.setter
    def smult (self, x):
        self._update_spin_cache (self._norb, self._neleca, self._nelecb, x)
        return self._smult

    @property
    def orbsym (self):
        return self._orbsym
    @orbsym.setter
    def orbsym (self, x):
        self._update_symm_cache (x)
        return self._orbsym

    @property
    def ndeta (self):
        return int (special.comb (self._norb, self._neleca, exact=True))

    @property
    def ndetb (self):
        return int (special.comb (self._norb, self._nelecb, exact=True))

    @property
    def ndet (self):
        return self.ndeta * self.ndetb

    @property
    def ncsf (self):
        if self.wfnsym is None or self._orbsym is None: return self.econf_csf_mask.size
        return (np.count_nonzero (self.confsym[self.econf_csf_mask] == self.wfnsym))

    def print_config (self, printer=print):
        printer ('***** CSFTransformer configuration *****')
        printer ('norb = {}'.format (self.norb))
        printer ('neleca, nelecb = {}, {}'.format (self.neleca, self.nelecb))
        printer ('smult = {}'.format (self.smult))
        printer ('orbsym = {}'.format (self.orbsym))
        printer ('wfnsym = {}'.format (self.wfnsym))
        printer ('ndeta, ndetb = {}, {}'.format (self.ndeta, self.ndetb))
        printer ('ncsf = {}'.format (self.ncsf))

def unpack_sym_ci (ci, idx, vec_on_cols=False):
    if idx is None: return ci
    tot_len = idx.size
    sym_len = np.count_nonzero (idx)
    if isinstance (ci, list) or isinstance (ci, tuple):
        assert (ci[0].size == sym_len), '{} {}'.format (ci[0].size, sym_len)
        dummy = np.zeros ((len (ci), tot_len), dtype=ci[0].dtype)
        dummy[:,idx] = np.asarray (ci)[:,:]
        if isinstance (ci, list):
            ci = list (dummy)
        else:
            ci = tuple (dummy)
        return ci
    elif ci.ndim == 2:
        if vec_on_cols:
            ci = ci.T
        assert (ci.shape[1] == sym_len), '{} {}'.format (ci.shape, sym_len)
        dummy = np.zeros ((ci.shape[0], tot_len), dtype=ci.dtype)
        dummy[:,idx] = ci
        if vec_on_cols:
            dummy = dummy.T
        return dummy
    else:
        assert (ci.ndim == 1), ci.ndim
        dummy = np.zeros (tot_len, dtype=ci.dtype)
        dummy[idx] = ci
        return dummy

def pack_sym_ci (ci, idx, vec_on_cols=False):
    if idx is None: return ci
    tot_len = idx.size
    if isinstance (ci, list) or isinstance (ci, tuple):
        assert (ci[0].size == tot_len), '{} {}'.format (ci[0].size, tot_len)
        dummy = np.asarray (ci)[:,idx]
        if isinstance (ci, list):
            ci = list (dummy)
        else:
            ci = tuple (dummy)
        return ci
    elif ci.ndim == 2:
        if vec_on_cols:
            ci = ci.T
        assert (ci.shape[1] == tot_len), '{} {}'.format (ci.shape, tot_len)
        dummy = ci[:,idx]
        if vec_on_cols:
            dummy = dummy.T
        return dummy
    else:
        assert (ci.ndim == 1)
        try:
            ci=ci[idx]
        except Exception as e:
            print (ci.shape, idx.shape)
            raise (e)
        return ci


def make_confsym (norb, neleca, nelecb, econf_det_mask, orbsym):
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    airreps = birreps = _gen_strs_irrep(strsa, orbsym)
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        birreps = _gen_strs_irrep(strsb, orbsym)
    nconf, addr = np.unique (econf_det_mask, return_index=True)
    nconf = nconf.size
    # Note: econf_det_mask[addr] = np.arange (nconf)
    # All determinants of the same configuration have the same point group
    conf_addra = addr // len (birreps)
    conf_addrb = addr % len (birreps)
    confsym = airreps[conf_addra] ^ birreps[conf_addrb]
    return confsym

def check_spinstate_norm (detarr, norb, neleca, nelecb, smult, csd_mask=None):
    ''' Calculate the norm of the given CI vector projected onto spin-state smult (= 2S+1) '''
    return transform_civec_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=csd_mask)[1]


def project_civec_csf (detarr, norb, neleca, nelecb, smult, csd_mask=None,
                       max_memory=param.MAX_MEMORY):
    ''' Project the total spin = s [= (smult-1) / 2] component of a CI vector using CSFs

    Args
    detarr: 2d ndarray of shape (ndeta,ndetb)
        ndeta = norb choose neleca
        ndetb = norb choose nelecb
    norb, neleca, nelecb, smult: ints

    Returns
    detarr: ndarray of unchanged shape
        Normalized CI vector in terms of determinants
    detnorm: float
    '''

    #detarr = transform_civec_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=csd_mask)
    #detarr = transform_civec_csf2det (detarr, norb, neleca, nelecb, smult, csd_mask=csd_mask)
    #return detarr
    ndeta = int (special.comb (norb, neleca, exact=True))
    ndetb = int (special.comb (norb, nelecb, exact=True))
    ndet = ndeta*ndetb
    assert (detarr.shape == tuple((ndeta,ndetb)) or detarr.shape == tuple((ndet,))), '{} {}'.format (detarr.shape, ndet)
    detarr = np.ravel (detarr, order='C')

    detarr = _transform_detcsf_vec_or_mat (detarr, norb, neleca, nelecb, smult, reverse=False, op_matrix=False,
                                           csd_mask=csd_mask, project=True, max_memory=max_memory)
    try:
        detnorm = linalg.norm (detarr)
    except Exception as ex:
        assert (detarr.shape == tuple((1,))), "{} {}".format (detarr.shape, ex)
        detnorm = detarr[0]
    '''
    if np.isclose (detnorm, 0):
        raise RuntimeWarning (('CI vector projected into CSF space (norb, na, nb, s = {}, {}, {}, {})'
            ' has zero norm; did you project onto a different spin before?').format (norb, neleca, nelecb, (smult-1)/2))
    '''
    return detarr / detnorm, detnorm

def transform_civec_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=None, vec_on_cols=False,
                             do_normalize=True, max_memory=param.MAX_MEMORY):
    ''' Express CI vector in terms of CSFs for spin s

    Args
    detarr: ndarray of size = ndet, or list, tuple, or kD-ndarray (k>1) of size ndet * nvec
        ndet = (norb choose neleca) * (norb choose nelecb)
    norb, neleca, nelecb, smult: ints

    Kwargs
    csd_mask: ndarray of shape (ndet,), ints
        Index mask array for reordering determinant pairs in csd order
    vec_on_cols: bool
        If true, multiple CI vectors are considered to be on the last dimension, not the first
        (i.e., an eigenvector matrix) (requires 2d ndarray for detarr)
    do_normalize: bool
        If false, do NOT normalize the vector (i.e., if it is a matrix-vector product

    Returns
    csfarr: same data type as detarr
        Normalized CI vector in terms of CSFs, with zero-norm vectors dropped
    csfnorm: ndarray of (maximum) length nvec, floats
    '''

    ndeta = int (special.comb (norb, neleca, exact=True))
    ndetb = int (special.comb (norb, nelecb, exact=True))
    ndet = ndeta*ndetb
    is_list = isinstance (detarr, list)
    is_tuple = isinstance (detarr, tuple)
    if is_list or is_tuple:
        is_flat = False
        nvec = len (detarr)
        detarr = np.ascontiguousarray (detarr)
    else:
        if (detarr.size % ndet != 0):
            raise ImpossibleCIvecError (('Impossible CI vector size {0} for system with {1} '
                                         'determinants').format (detarr.size, ndet), ndet=ndet,
                                        norb=norb, neleca=neleca, nelecb=nelecb)
        nvec = detarr.size // ndet
        is_flat = len (detarr.shape) == 1
    if vec_on_cols:
        detarr = detarr.reshape (ndet, nvec)
        detarr = np.ascontiguousarray (detarr.T)
    else:
        detarr = np.ascontiguousarray (detarr.reshape (nvec, ndet))


    # Driver needs an ndarray of explicit shape (*, ndet)
    csfarr = _transform_detcsf_vec_or_mat (detarr, norb, neleca, nelecb, smult, reverse=False, op_matrix=False,
                                           csd_mask=csd_mask, project=False, max_memory=max_memory)
    if csfarr.size == 0:
        assert (False)
        return np.zeros (0, dtype=detarr.dtype), 0.0

    # Manipulate csfarr back into the original shape
    csfnorm = linalg.norm (csfarr, axis=1)
    idx_norm = ~np.isclose (csfnorm, 0)
    if do_normalize:
        csfarr[idx_norm,:] /= csfnorm[idx_norm,np.newaxis]
    if vec_on_cols:
        csfarr = csfarr.T
    csfarr = np.ascontiguousarray (csfarr)
    if is_flat:
        csfarr = csfarr.ravel ()
    elif is_list:
        csfarr = list (csfarr)
    elif is_tuple:
        csfarr = tuple (csfarr)

    if csfnorm.size == 1:
        csfnorm = csfnorm[0]
    elif csfnorm.size == 0:
        csfnorm = 0.0
    return csfarr, csfnorm

def transform_civec_csf2det (csfarr, norb, neleca, nelecb, smult, csd_mask=None, vec_on_cols=False,
                             do_normalize=True, max_memory=param.MAX_MEMORY):
    ''' Transform CI vector in terms of CSFs back into determinants

    Args
    csfarr: ndarray of size = ncsf, or list, tuple, or kD-ndarray (k>1) of size ncsf * nvec
    norb, neleca, nelecb, smult: ints

    Kwargs
    csd_mask: ndarray of shape (ndet,), ints
        Index mask array for reordering determinant pairs in csd order
    vec_on_cols: bool
        If true, multiple CI vectors are considered to be on the last dimension, not the first
        (i.e., an eigenvector matrix) (requires 2d ndarray for detarr)
    do_normalize: bool
        If false, do NOT normalize the vector (i.e., if it is a matrix-vector product


    Returns
    detarr: same data type as csfarr. Last dimension is of length ndeta*ndetb
        Normalized CI vector in terms of CSFs, with zero-norm vectors dropped
    detnorm: ndarray of (maximum) length nvec, floats
    '''
    if np.asarray (csfarr).size == 0:
        return np.zeros (0, dtype=csfarr.dtype), 0.0

    ndeta = int (special.comb (norb, neleca, exact=True))
    ndetb = int (special.comb (norb, nelecb, exact=True))
    ndet = ndeta*ndetb
    ncsf = count_all_csfs (norb, neleca, nelecb, smult)
    is_list = isinstance (csfarr, list)
    is_tuple = isinstance (csfarr, tuple)
    if is_list or is_tuple:
        is_flat = False
        nvec = len (csfarr)
        csfarr = np.ascontiguousarray (csfarr)
    else:
        if (csfarr.size % ncsf != 0):
            raise ImpossibleCIvecError (('Impossible CI vector size {0} for system with {1} '
                                         'CSFs').format (csfarr.size, ncsf), ncsf=ncsf, norb=norb,
                                        neleca=neleca, nelecb=nelecb, smult=smult)
        nvec = csfarr.size // ncsf
        is_flat = len (csfarr.shape) == 1
    if vec_on_cols:
        csfarr = csfarr.reshape (ncsf, nvec)
        csfarr = np.ascontiguousarray (csfarr.T)
    else:
        csfarr = np.ascontiguousarray (csfarr.reshape (nvec, ncsf))

    detarr = _transform_detcsf_vec_or_mat (csfarr, norb, neleca, nelecb, smult, reverse=True, op_matrix=False,
                                           csd_mask=csd_mask, project=False, max_memory=max_memory)

    # Manipulate detarr back into the original shape
    detnorm = linalg.norm (detarr, axis=1)
    if do_normalize:
        detarr = detarr / detnorm[:,np.newaxis]
    detarr = detarr.reshape (nvec, ndet)
    if vec_on_cols:
        detarr = detarr.T
    detarr = np.ascontiguousarray (detarr)
    if is_flat:
        detarr = detarr.ravel ()
    elif is_list:
        detarr = list (detarr)
    elif is_tuple:
        detarr = tuple (detarr)

    if detnorm.size == 1:
        detnorm = detnorm[0]
    elif detnorm.size == 0:
        detnorm = 0.0
    return detarr, detnorm

def transform_opmat_det2csf (detarr, norb, neleca, nelecb, smult, csd_mask=None,
                             max_memory=param.MAX_MEMORY):
    ''' Express operator matrix in terms of CSFs for spin s

    Args
    detarr: ndarray of shape (ndet, ndet) or (ndet**2,)
        ndet = (norb choose neleca) * (norb choose nelecb)
    norb, neleca, nelecb, smult: ints

    Returns
    csfarr: contiguous ndarray of shape (ncsf, ncsf) where ncsf < ndet
        Operator matrix in terms of csfs
    '''

    ndeta = int (special.comb (norb, neleca, exact=True))
    ndetb = int (special.comb (norb, nelecb, exact=True))
    ndet = ndeta*ndetb
    assert (detarr.shape == tuple((ndet,ndet)) or detarr.shape == tuple((ndet**2,))), "{} {} (({},{}),{})".format (
        detarr.shape, ndet, neleca, nelecb, norb)
    csfarr = _transform_detcsf_vec_or_mat (detarr, norb, neleca, nelecb, smult, reverse=False, op_matrix=True,
                                           csd_mask=csd_mask, project=False, max_memory=max_memory)
    return csfarr

def _transform_detcsf_vec_or_mat (arr, norb, neleca, nelecb, smult, reverse=False, op_matrix=False, csd_mask=None,
                                  project=False, max_memory=param.MAX_MEMORY):
    ''' Wrapper to manipulate array into correct shape and transform both dimensions if an operator matrix

    Args
    arr: ndarray of shape (nrow, ncol)
        ncol = num determinant pairs (or num csfs if reverse = True
        nrow = integer divisor of arr.size by ncol
    norb, neleca, nelecb, smult : ints
        num orbitals, num alpha electrons, num beta electrons, and 2S+1

    Kwargs
    reverse: bool
        if true, transform back into determinants from csfs
    op_matrix: bool
        if true, arr is transformed along both dimensions (must be square matrix)
    csd_mask: ndarray of shape ndeta*ndetb (int)
        index array for reordering determinant pairs in csd format
    project: bool
        if true, arr is projected, rather than transformed

    Returns
    arr: ndarray of shape (nrow, ncol)
    '''

    ndeta = int (special.comb (norb, neleca, exact=True))
    ndetb = int (special.comb (norb, nelecb, exact=True))
    ndet_all = ndeta*ndetb
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)

    ncol = ncsf_all if reverse else ndet_all
    nrow = arr.size // ncol
    if op_matrix:
        assert (nrow == ncol), "operator matrix must be square"
    assert (arr.shape == tuple((nrow, ncol))), "array shape should be {0}; is {1}".format ((nrow, ncol), arr.shape)

    arr = _transform_det2csf (arr, norb, neleca, nelecb, smult, reverse=reverse, csd_mask=csd_mask,
                              project=project, max_memory=max_memory)
    if op_matrix:
        arr = arr.T
        arr = _transform_det2csf (arr, norb, neleca, nelecb, smult, reverse=reverse,
                                  csd_mask=csd_mask, project=project, max_memory=max_memory)
        arr = numpy_helper.transpose (arr, inplace=True)

    if arr.size == 0:
        nrow = 1
        ncol = 0
    else:
        ncol = ndet_all if (reverse or project) else ncsf_all
        nrow = arr.size // ncol
    if op_matrix:
        assert (nrow == ncol), "operator matrix must be square"
    assert (arr.shape == tuple((nrow, ncol))), "array shape should be {0}; is {1}".format ((nrow, ncol), arr.shape)

    return arr


def _transform_det2csf (inparr, norb, neleca, nelecb, smult, reverse=False, csd_mask=None,
                        project=False, max_memory=param.MAX_MEMORY):
    ''' Must take an array of shape (*, ndet) or (*, ncsf) '''
    #t_start = lib.logger.perf_counter ()
    time_umat = 0
    time_mult = 0
    time_getdet = 0
    size_umat = 0

    min_npair, npair_csd_offset, npair_dconf_size, npair_sconf_size, npair_sdet_size = csdstring.get_csdaddrs_shape (
        norb, neleca, nelecb)
    _, npair_csf_offset, _, _, npair_csf_size = get_csfvec_shape (norb, neleca, nelecb, smult)
    nrow = inparr.shape[0]
    ndeta_all = int (special.comb (norb, neleca, exact=True))
    ndetb_all = int (special.comb (norb, nelecb, exact=True))
    ndet_all = ndeta_all * ndetb_all
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)

    ncol_out = ndet_all if (reverse or project) else ncsf_all
    #ncol_in = ndet_all if ((not reverse) or project) else ncsf_all
    if not project:
        outarr = np.ascontiguousarray (np.zeros ((nrow, ncol_out), dtype=np.float64))
        csf_addrs = np.zeros (ncsf_all, dtype=np.bool_)
    # Initialization is necessary because not all determinants have a csf for all spin states

    #max_npair = min (nelecb, (neleca + nelecb - int (round (2*s))) // 2)
    max_npair = min (neleca, nelecb)
    for npair in range (min_npair, max_npair+1):
        ipair = npair - min_npair
        ncsf = npair_csf_size[ipair]
        nspin = neleca + nelecb - 2*npair
        nconf = npair_dconf_size[ipair] * npair_sconf_size[ipair]
        ndet = npair_sdet_size[ipair]
        csf_offset = npair_csf_offset[ipair]
        csd_offset = npair_csd_offset[ipair]
        if (ncsf == 0) and not project:
            continue
        if not project:
            csf_addrs[:] = False
            csf_addrs_ipair = csf_addrs[csf_offset:][:nconf*ncsf].reshape (nconf, ncsf) # Note: this is a view

        t_ref = lib.logger.perf_counter ()
        if csd_mask is None:
            det_addrs = csdstring.get_nspin_dets (norb, neleca, nelecb, nspin)
        else:
            det_addrs = csd_mask[csd_offset:][:nconf*ndet].reshape (nconf, ndet, order='C')
        assert (det_addrs.shape[0] == nconf)
        assert (det_addrs.shape[1] == ndet)
        time_getdet += lib.logger.perf_counter () - t_ref

        if (ncsf == 0):
            inparr[:,det_addrs] = 0
            continue

        # sign convention of PySCF is A' B' |vac>
        # sign convention of get_spin_evecs is A'(unpaired) B'(unpaired) C'(pairs) |vac>
        ncomm = npair * (npair-1) // 2 # A'(paired) B'(paired) -> C'(pairs)
        ncomm += npair * max (0, nelecb-npair) # A'B' -> AB'(unpaired) AB'(paired)
        sgn = (-1) ** (ncomm % 2)

        t_ref = lib.logger.perf_counter ()
        umat = sgn * get_spin_evecs (nspin, neleca, nelecb, smult, max_memory=max_memory)
        umat = np.asarray_chkfinite (umat)
        size_umat = max (size_umat, umat.nbytes)
        ncsf_blk = ncsf # later on I can use this variable to implement a generator form of get_spin_evecs to save
        #                 memory when there are too many csfs
        assert (umat.shape[0] == ndet)
        assert (umat.shape[1] == ncsf_blk)
        if project:
            Pmat = np.dot (umat, umat.T)
        time_umat += lib.logger.perf_counter () - t_ref

        if not project:
            csf_addrs_ipair[:,:ncsf_blk] = True # Note: edits csf_addrs

        # The elements of csf_addrs and det_addrs are addresses for the flattened vectors and matrices (inparr.flat and
        # outarr.flat)
        # Passing them unflattened as indices of the flattened arrays should result in a 3-dimensional array if I
        # understand numpy's indexing rules correctly
        # For the lvalues, I think it's necessary to flatten csf_addrs and det_addrs to avoid an exception
        # Hopefully this is parallel under the hood, and hopefully the OpenMP reduction epsilon doesn't ruin the spin
        # eigenvectors
        t_ref = lib.logger.perf_counter ()
        if project:
            inparr[:,det_addrs] = np.tensordot (inparr[:,det_addrs], Pmat, axes=1)
        elif not reverse:
            outarr[:,csf_addrs] = np.tensordot (inparr[:,det_addrs], umat, axes=1).reshape (nrow, ncsf_blk*nconf)
        else:
            outarr[:,det_addrs] = np.tensordot (inparr[:,csf_addrs].reshape (nrow, nconf, ncsf_blk), umat,
                                                axes=((2,),(1,)))
        time_mult += lib.logger.perf_counter () - t_ref

    if project:
        outarr = inparr
    else:
        outarr = outarr.reshape (nrow, ncol_out)
    '''
    d = ['determinants','csfs']
    print (('Transforming {} into {} summary: {:.2f} seconds to get determinants,'
            ' {:.2f} seconds to build umat, {:.2f} seconds matrix-vector multiplication,
            ' {:.2f} MB largest umat').format (d[reverse], d[~reverse], time_getdet, time_umat,
            ' {:.2f} MB largest umat').format (d[reverse], d[~reverse], time_getdet, time_umat,
            time_mult, size_umat / 1e6))
    print ('Total time spend in _transform_det2csf: {:.2f} seconds'.format (lib.logger.perf_counter () - t_start))
    '''
    return outarr

def transform_opmat_det2csf_pspace (op, econfs, norb, neleca, nelecb, smult, csd_mask,
                                    econf_det_mask, econf_csf_mask, max_memory=param.MAX_MEMORY):
    ''' Transform an operator matrix from the determinant basis to the csf basis, in a subspace of determinants spanning
        the electron configurations addressed by econfs

    Args:
        op: square ndarray
            operator matrix in the determinant basis
            the basis must be arranged in the order (econfs[0], det[0]), (econfs [0], det[1]), ...
            (econfs[1], det[0]), ...
            where det[n] is the nth configuration of spins compatible with a given spinless
            electron configuration (i.e., a configuration with no singly-occupied orbitals has only 1 determinant, etc.)
            It MUST include ALL determinants within each electron configuration given by econfs
        econfs: ndarray of ints
            addresses for electron configurations in the canonical order defined by csdstring.py
            (NOT determinants, refers to strings like 2 2 1 2 0 0)
        norb, neleca, nelecb, smult: ints
            basic parameters: numbers of orbitals and electrons and 2s+1
        csd_mask: ndarray of ints
            csd_mask[idx_csd] = idx_dd
        econf_det_mask: ndarray of ints
            econf_det_mask[idx_dd] = idx_econf
        econf_csf_mask: ndarray of ints
            econf_det_mask[idx_csf] = idx_econf

    Returns:
        op: ndarray
            In csf basis
        csf_addrs: ndarray of ints
            CI vector element addresses in CSF basis
    '''

    # I basically need to invert econf_det_mask and econf_csf_mask. I can't use argsort for this because their elements
    # aren't unique
    # csf_addrs needs to be sorted because I don't want to make a second reduced_csd_mask index for the csf-basis
    # version (see below);
    # just return the damn thing in the canonical order!
    det_addrs = np.concatenate ([np.nonzero (econf_det_mask == conf)[0] for conf in econfs])
    csf_addrs = np.sort (np.concatenate ([np.nonzero (econf_csf_mask == conf)[0] for conf in econfs]))
    ndet_all = det_addrs.size
    ncsf_all = csf_addrs.size
    # econfs could have been provided in any order and defines the indexing of "op" (via det_addrs as generated above).
    #   I need to generate a mask array to address the elements of "op"
    #   in the "canonical" ordering (ipair, doubly-occupied string, singly-occupied string, and spin configuration)
    #   that I developed in csd string but spanning only those configurations that are included in econfs.
    #   The next line should accomplish this. np.argsort (mask) inverts a given mask index array (assuming its elements
    #   are unique).
    #   So for instance, np.argsort (csd_mask)[det_addrs] gives you csd addresses.
    #   Then np.argsort (np.argsort (csd_mask)[det_addrs])[csd_addrs] inverts it twice, and gives you determinant
    #   addresses, but if det_addrs doesn't span the whole space, then csd_addrs can't either. In other words, the csd
    #   indices are compressed and correspond to the elements of op.
    reduced_csd_mask = np.argsort (np.argsort (csd_mask)[det_addrs])
    assert (op.shape == (ndet_all, ndet_all)), "operator matrix shape problem ({} for det_addrs of size {})".format (
        op.shape, det_addrs.size)
    min_npair, npair_csd_offset, npair_dconf_size, npair_sconf_size, npair_sdet_size = csdstring.get_csdaddrs_shape (
        norb, neleca, nelecb)
    _, npair_csf_offset, _, _, npair_csf_size = get_csfvec_shape (norb, neleca, nelecb, smult)
    npair_econf_size = npair_dconf_size * npair_sconf_size
    max_npair = min (neleca, nelecb)
    assert (np.count_nonzero (reduced_csd_mask[1:] - reduced_csd_mask[:-1] - 1)==0)
    def ax_b (mat):
        nrow = mat.shape[0]
        assert (mat.shape[1] == ndet_all)
        outmat = np.zeros ((nrow, ncsf_all), dtype=mat.dtype)
        det_offset = 0
        csf_offset = 0
        full_conf_offset = 0
        full_det_offset = 0
        for npair in range (min_npair, max_npair+1):
            ipair = npair - min_npair
            nconf_full = npair_econf_size[ipair]
            nconf = np.count_nonzero (np.isin (range (full_conf_offset, full_conf_offset+nconf_full), econfs))
            full_conf_offset += nconf_full
            ncsf = npair_csf_size[ipair]
            ndet = npair_sdet_size[ipair]
            full_det_offset += nconf_full * ndet
            if nconf == 0 or ncsf == 0 or ndet == 0:
                continue

            ci = csf_offset
            cj = ci + nconf*ncsf

            di = reduced_csd_mask[det_offset]
            dj = di + nconf*ndet
            mat_ij = mat[:,di:dj].reshape (nrow, nconf, ndet)

            # sign convention of PySCF is A' B' |vac>
            # sign convention of get_spin_evecs is A'(unpaired) B'(unpaired) C'(pairs) |vac>
            ncomm = npair * (npair-1) // 2 # A'(paired) B'(paired) -> C'(pairs)
            ncomm += npair * max (0, nelecb-npair) # A'B' -> AB'(unpaired) AB'(paired)
            sgn = (-1) ** (ncomm % 2)

            nspin = neleca + nelecb - 2*npair
            umat = sgn * get_spin_evecs (nspin, neleca, nelecb, smult, max_memory=max_memory)
            umat = np.asarray_chkfinite (umat)

            outmat[:,ci:cj] = np.tensordot (mat_ij, umat, axes=1).reshape (nrow, ncsf*nconf, order='C')

            det_offset += nconf*ndet
            csf_offset += nconf*ncsf

        assert (det_offset <= ndet_all), "{} {}".format (det_offset, ndet_all) # Can be less because npair < min_npair
        #                                                                        corresponds to some dets that are
        #                                                                        skipped
        assert (csf_offset == ncsf_all), "{} {}".format (csf_offset, ncsf_all)
        return outmat

    op = ax_b (op).conj ().T
    op = ax_b (op).conj ().T
    return op, csf_addrs



def make_econf_csf_mask (norb, neleca, nelecb, smult):
    ''' Make a mask index matching csfs to electron configurations '''

    min_npair, npair_offset, npair_dconf_size, npair_sconf_size, npair_csf_size = get_csfvec_shape (
        norb, neleca, nelecb, smult)
    ncsf = count_all_csfs (norb, neleca, nelecb, smult)
    mask = np.empty (ncsf, dtype=np.uint32)
    npair_conf_size = npair_dconf_size * npair_sconf_size
    npair_size = npair_conf_size * npair_csf_size
    iconf = 0
    for npair in range (min_npair, min (neleca, nelecb)+1):
        ipair = npair - min_npair
        irange = np.arange (iconf, iconf+npair_conf_size[ipair], dtype=np.uint32)
        iconf += npair_conf_size[ipair]
        mask[npair_offset[ipair]:][:npair_size[ipair]] = np.repeat (irange, npair_csf_size[ipair])
    return mask

def csf_gentable (nspin, smult):
    ''' Example of a genealogical coupling table for 8 spins and s = 1 (triplet), counting from the final state
        back to the null state:

           28 28 19 10  4  1  .
            |  9  9  6  3  1  .
            |  |  3  3  2  1  .
            |  |  |  1  1  t  .
                        .  .  .
                           .  .
                              .

        Top left (0,0) is the null state (nspin = 0, s = 0).
        Position (3,5) (1/2 nspin - s, 1/2 nspin + s) is the target state (nspin=8, s=1).
        Numbers count paths from that position to the target state, moving only down or to the right; gen(0,0)=gen(0,1)
        is the total number of CSFs with this spin state for any electron configuration with 8 unpaired electrons.
        Moving left corresponds to bit=1; moving down corresponds to bit=0.
        Vertical lines are not defined (s<0) but stored as zero [so array has shape=(1/2 nspin - s + 1 ,
        1/2 nspin + s + 1)]. Dots are not stored but are defined as zero.
        Rotate 45 degrees counter clockwise and nspins is on the horizontal from left to right, and spin is on the
        vertical from bottom to top. To compute the address from a string, sum the numbers which appear above and to the
        right of every coordinate reached from above. For example, the string 01101101 turns down in the second, fifth,
        and eighth places (counting from right to left) so its address is 19 (0,2 [0+2=2]) + 3 (1,4 [1+4=5]) + 0
        (2,6[2+6=8]) = 22. To compute the string from the address, find the largest number on each row that's less than
        the address, subtract it, set every unassignd bit up to that column to 1, set the bit in that column to 0, go
        down a row, reset the column index to zero, and repeat. For example, address 15 is 10 (0,3) + 3 (1,4) + 2 (2,4),
        so the string is 11001011 again indexing the bits from right to left.
    '''

    assert ((smult - 1) % 2 == nspin % 2), "{} {}".format (smult, nspin)
    if smult > nspin+1:
        return np.zeros ((1,1), dtype=np.int32)

    n0 = (nspin - (smult - 1)) // 2
    n1 = (nspin + (smult - 1)) // 2

    gentable = np.zeros ((n0+1,n1+1), dtype=np.int32)
    gentable[n0,n0:] = 1
    for i0 in range (n0, 0, -1):
        row = gentable[i0,i0:]
        row = [sum (row[i1:]) for i1 in range (len (row))]
        row = [row[0]] + list (row)
        gentable[i0-1,i0-1:] = row
    return gentable

def count_csfs (nspin, smult):
    return csf_gentable (nspin, smult)[0,0]

def count_all_csfs (norb, neleca, nelecb, smult):
    a,b,c = get_csfvec_shape (norb, neleca, nelecb, smult)[-3:]
    return np.sum (a*b*c)

def get_csfvec_shape (norb, neleca, nelecb, smult):
    ''' For a system of neleca + nelecb electrons with MS = (neleca - nelecb) occupying norb orbitals,
        get shape information about the irregular CI vector array in terms of csfs (number of pairs, pair config,
        unpair config, coupling string)

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
        npair_csf_size, 1d ndarray of integers
            npair_csf_size[i] = number of coupling vectors leading to spin = s for neleca+nelecb - 2*npair spins
    '''
    s = (smult - 1) / 2
    ms = (neleca - nelecb) / 2
    #assert (neleca >= nelecb)
    nless = min (neleca, nelecb)
    if neleca>norb or nelecb>norb:
        raise ImpossibleCIvecError ("({}e+{}e,{}o)".format (neleca, nelecb, norb),
                                    norb=norb, neleca=neleca, nelecb=nelecb)
    if (abs (neleca-nelecb) > smult - 1):
        raise ImpossibleSpinError ("Impossible quantum numbers: s = {}; ms = {}".format (s, ms),
                                   norb=norb, neleca=neleca, nelecb=nelecb, smult=smult)
    min_npair = max (0, neleca + nelecb - norb)
    nspins = [neleca + nelecb - 2*npair for npair in range (min_npair, nless+1)]
    if (smult-1)>nspins[0]:
        raise ImpossibleSpinError ('Maximum possible 2s+1 = {} for ({}e+{}e,{}o)'.format (
            nspins[0]+1, neleca, nelecb, norb), norb=norb, neleca=neleca, nelecb=nelecb,
            smult=smult)
    nfreeorbs = [norb - npair for npair in range (min_npair, nless+1)]
    for nspin in nspins:
        assert ((nspin + neleca - nelecb) % 2 == 0)

    npair_dconf_size = np.asarray ([int (special.comb (norb, npair, exact=True))
                                    for npair in range (min_npair, nless+1)], dtype=np.int32)
    npair_sconf_size = np.asarray ([int (special.comb (nfreeorb, nspin, exact=True))
                                    for nfreeorb, nspin in zip (nfreeorbs, nspins)], dtype=np.int32)
    npair_csf_size = np.asarray ([count_csfs (nspin, smult) for nspin in nspins]).astype (np.int32)

    npair_sizes = np.asarray ([0] + [i * j * k for i,j,k in zip (npair_dconf_size, npair_sconf_size, npair_csf_size)],
                              dtype=np.int32)
    npair_offset = np.asarray ([np.sum (npair_sizes[:i+1]) for i in range (len (npair_sizes))], dtype=np.int32)
    ndeta, ndetb = (int (special.comb (norb, n, exact=True)) for n in (neleca, nelecb))
    assert (npair_offset[-1] <= ndeta*ndetb), "{} determinants and {} csfs".format (ndeta*ndetb, npair_offset[-1])

    return min_npair, npair_offset[:-1], npair_dconf_size, npair_sconf_size, npair_csf_size

def get_spin_evecs (nspin, neleca, nelecb, smult, max_memory=param.MAX_MEMORY):
    ms = (neleca - nelecb) / 2
    s = (smult - 1) / 2
    #assert (neleca >= nelecb)
    assert (abs (neleca - nelecb) <= smult - 1)
    assert (abs (neleca - nelecb) <= nspin)
    assert (abs (neleca - nelecb) % 2 == (smult-1) % 2)
    assert (abs (neleca - nelecb) % 2 == nspin % 2)

    na = (nspin + neleca - nelecb) // 2
    ndet = int (special.comb (nspin, na, exact=True))
    ncsf = count_csfs (nspin, smult)

    #t_start = lib.logger.perf_counter ()
    spinstrs = cistring.addrs2str (nspin, na, list (range (ndet)))
    assert (len (spinstrs) == ndet), "should have {} spin strings; have {} (nspin={}, ms={}".format (
        ndet, len (spinstrs), nspin, ms)

    #t_start = lib.logger.perf_counter ()
    scstrs = addrs2str (nspin, smult, list (range (ncsf)))
    assert (len (scstrs) == ncsf), "should have {} coupling strings; have {} (nspin={}, s={})".format (
        ncsf, len (scstrs), nspin, s)

    mem_current = lib.current_memory ()[0]
    mem_rem = max_memory - mem_current
    mem_reqd = ndet * ncsf * np.dtype (np.float64).itemsize / 1e6
    if mem_reqd > mem_rem:
        memstr = ('CSF unitary matrix for {} unpaired of {} total electrons w/ s={:.1f} is too big'
                  " ({} MB req'd of {} MB remaining; {} MB total available)").format (
            nspin, neleca+nelecb, s, mem_reqd, mem_rem, max_memory)
        raise MemoryError (memstr)
    umat = np.ones ((ndet, ncsf), dtype=np.float64)
    twoS = smult-1
    twoMS = neleca - nelecb

    #t_start = lib.logger.perf_counter ()
    libcsf.FCICSFmakecsf (umat.ctypes.data_as (ctypes.c_void_p),
                        spinstrs.ctypes.data_as (ctypes.c_void_p),
                        scstrs.ctypes.data_as (ctypes.c_void_p),
                        ctypes.c_int (nspin),
                        ctypes.c_size_t (ndet),
                        ctypes.c_size_t (ncsf),
                        ctypes.c_int (twoS),
                        ctypes.c_int (twoMS))

    return umat

def _test_spin_evecs (nspin, neleca, nelecb, smult, S2mat=None):
    s = (smult - 1) / 2
    ms = (neleca - nelecb) / 2
    assert (ms <= s)
    assert (smult-1 <= nspin)
    assert (nspin >= neleca + nelecb)

    na = (nspin + neleca - nelecb) // 2
    ndet = int (special.comb (nspin, na, exact=True))
    ncsf = count_csfs (nspin, smult)

    spinstrs = cistring.addrs2str (nspin, na, list (range (ndet)))

    if S2mat is None:
        S2mat = np.zeros ((ndet, ndet), dtype=np.float64)
        twoMS = int (round (2 * ms))

        t_start = lib.logger.perf_counter ()
        libcsf.FCICSFmakeS2mat (S2mat.ctypes.data_as (ctypes.c_void_p),
                             spinstrs.ctypes.data_as (ctypes.c_void_p),
                             ctypes.c_size_t (ndet),
                             ctypes.c_int (nspin),
                             ctypes.c_int (twoMS))
        print ("TIME: {} seconds to make S2mat for {} spins with s={}, ms={}".format (
            lib.logger.perf_counter() - t_start, nspin, (smult-1)/2, ms))
        print ("MEMORY: {} MB for {}-spin S2 matrix with s={}, ms={}".format (S2mat.nbytes / 1e6,
            nspin, (smult-1)/2, ms))

    umat = get_spin_evecs (nspin, neleca, nelecb, smult)
    print ("MEMORY: {} MB for {}-spin csfs with s={}, ms={}".format (umat.nbytes / 1e6,
        nspin, (smult-1)/2, ms))
    assert (umat.shape == tuple((ndet, ncsf))), "umat shape should be ({},{}); is {}".format (ndet, ncsf, umat.shape)

    s = (smult-1)/2
    t_start = lib.logger.perf_counter ()
    isorth = np.allclose (np.dot (umat.T, umat), np.eye (umat.shape[1]))
    ortherr = linalg.norm (np.dot (umat.T, umat) - np.eye (umat.shape[1]))
    S2mat_csf = reduce (np.dot, (umat.T, S2mat, umat))
    S2mat_csf_comp = s * (s+1) * np.eye (umat.shape[1])
    S2mat_csf_err = linalg.norm (S2mat_csf - S2mat_csf_comp)
    diagsS2 = np.allclose (S2mat_csf, S2mat_csf_comp)
    passed = isorth and diagsS2
    print ("TIME: {} seconds to analyze umat for {} spins with s={}, ms={}".format (
        lib.logger.perf_counter() - t_start, nspin, s, ms))


    print (('For a system of {} spins with total spin {} and spin projection {}'
            ', {} CSFs found from {} determinants by Clebsch-Gordan algorithm').format (
            nspin, s, ms, umat.shape[1], len (spinstrs)))
    print ('Did the Clebsch-Gordan algorithm give orthonormal eigenvectors? {}'.format (
        ('NO (err = {})'.format (ortherr), 'Yes')[isorth]))
    print ('Did the Clebsch-Gordan algorithm diagonalize the S2 matrix with the correct eigenvalues? {}'.format (
        ('NO (err = {})'.format (S2mat_csf_err), 'Yes')[diagsS2]))
    print ('nspin = {}, S = {}, MS = {}: {}'.format (nspin, s, ms, ('FAILED','Passed')[passed]))
    sys.stdout.flush ()

    return umat, S2mat

def get_scstrs (nspin, smult):
    ''' This is not a great way to do this, but I seriously can't think of any straightforward way to put the coupling
    strings in order... '''
    if (smult >= nspin):
        return np.ones ((0), dtype=np.int64)
    elif (nspin == 0):
        return np.zeros ((1), dtype=np.int64)
    assert (int (round (smult + nspin)) % 2 == 1), "npsin = {}; 2S+1 = {}".format (nspin, smult)
    nup = (nspin + smult - 1) // 2
    ndet = int (special.comb (nspin, nup))
    scstrs = cistring.addrs2str (nspin, nup, list (range (ndet)))
    mask = np.ones (len (scstrs), dtype=np.bool_)

    libcsf.FCICSFgetscstrs (scstrs.ctypes.data_as (ctypes.c_void_p),
                            mask.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_size_t (len (scstrs)),
                            ctypes.c_int (nspin))

    return np.ascontiguousarray (scstrs[mask], dtype=np.int64)


# This is only for a given set of spins, not the full csf string
def addrs2str (nspin, smult, addrs):
    addrs = np.ascontiguousarray (addrs, dtype=np.int32)
    nstr = len (addrs)
    strs = np.ascontiguousarray (np.zeros (nstr, dtype=np.int64))
    gentable = np.ravel (csf_gentable (nspin, smult))
    twoS = smult - 1
    libcsf.FCICSFaddrs2str (strs.ctypes.data_as (ctypes.c_void_p),
                            addrs.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_size_t (nstr),
                            gentable.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (nspin),
                            ctypes.c_int (twoS))

    return strs

# This is only for a given set of spins, not the full csf string
def strs2addr (nspin, smult, strs):
    strs = np.ascontiguousarray (strs, dtype=np.int64)
    nstr = len (strs)
    addrs = np.zeros (nstr, dtype=np.int32)
    gentable = np.ravel (csf_gentable (nspin, smult))
    twoS = smult - 1
    libcsf.FCICSFstrs2addr (addrs.ctypes.data_as (ctypes.c_void_p),
                            strs.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_size_t (nstr),
                            gentable.ctypes.data_as (ctypes.c_void_p),
                            ctypes.c_int (nspin),
                            ctypes.c_int (twoS))
    return addrs


def check_all_umat_size (norb, neleca, nelecb):
    ''' Calcluate the number of elements of the unitary matrix between all possible determinants
    and all possible CSFs for (neleca, nelecb) electrons in norb orbitals '''
    min_smult = neleca - nelecb + 1
    min_npair = max (0, neleca + nelecb - norb)
    max_smult = neleca + nelecb - 2*min_npair + 1
    return sum ([check_tot_umat_size (norb, neleca, nelecb, smult) for smult in range (min_smult, max_smult, 2)])

def check_tot_umat_size (norb, neleca, nelecb, smult):
    ''' Calculate the number of elements of the unitary matrix between all possible determinants
    and CSFs of a given spin state for (neleca, nelecb) electrons in norb orbitals '''
    min_npair = max (0, neleca + nelecb - norb)
    return sum ([check_umat_size (norb, neleca, nelecb, npair, smult)
                 for npair in range (min_npair, min (neleca, nelecb)+1)])

def check_umat_size (norb, neleca, nelecb, npair, smult):
    ''' Calculate the number of elements of the unitary matrix between determinants with npair electron pairs
    and CSFs of a given spin state for (neleca, nelecb) electrons in norb orbitals '''
    nspin = neleca + nelecb - 2*npair
    na = (nspin + neleca - nelecb) // 2
    ndet = special.binom (nspin, na)
    ncsf = count_csfs (nspin, smult)
    return ndet * ncsf




if __name__ == '__main__':
    for nspin in range (21):
        for s in np.arange ((nspin%2)/2, (nspin/2)+0.1, 1):
            smult = int (round (2*s+1))
            ncsf = count_csfs (nspin, smult)
            print ("Supposedly {} csfs of {} spins with overall smult = {}".format (ncsf, nspin, smult))
            rand_addrs = np.random.randint (0, high=ncsf, size=min (ncsf, 5), dtype=np.int32)
            rand_strs = addrs2str (nspin, smult, rand_addrs)
            rand_addrs_2 = strs2addr (nspin, smult, rand_strs)
            assert (np.all (rand_addrs == rand_addrs_2))

    for nspin in range (15):
        for ms in np.arange (-(nspin/2), (nspin/2)+0.1, 1):
            evals = []
            evecs = []
            S2mat = None
            for s in np.arange (abs (ms), (nspin/2)+0.1, 1):
                smult = int (round (2*s+1))
                neleca = int (round (nspin/2 + ms))
                nelecb = int (round (nspin/2 - ms))
                umat, S2mat = _test_spin_evecs (nspin, neleca, nelecb, smult, S2mat=S2mat)
                evecs.append (umat)
                evals.append (s*(s+1)*np.ones (umat.shape[1]))
            print ("COLLECTIVE RESULTS:")
            t_start = lib.logger.perf_counter ()
            evals = np.concatenate (evals)
            evecs = np.concatenate (evecs, axis=-1)
            print ('Is the final CSF vector matrix square with correct dimension? {} vs {}'.format (
                evecs.shape, S2mat.shape))
            issquare = np.all (evecs.shape == S2mat.shape)
            if not issquare:
                print ("{} spins, {} projected spin overall: FAILED".format (nspin, ms))
                continue
            ovlp = np.dot (evecs.T, evecs)
            S2op = reduce (np.dot, (evecs.T, S2mat, evecs))
            ovlperr = linalg.norm (ovlp - np.eye (evecs.shape[1]))
            diagerr = linalg.norm (S2op - np.diag (evals))
            isorthnorm = ovlperr < 1e-8
            diagsS2 = diagerr < 1e-8
            print ("TIME: {} seconds to analyze umat for {} spins with ms={} and all s".format (
                lib.logger.perf_counter() - t_start, nspin, ms))
            print ("Is the final CSF vector matrix unitary? {}".format (
                ("NO (err = {})".format (ovlperr), "Yes")[isorthnorm]))
            print (('Does the final CSF vector matrix correctly diagonalize S2?'
                    ' {}').format (('NO (err = {})'.format (diagerr), 'Yes')[diagsS2]))
            print ("{} spins, {} projected spin overall: {}".format (
                nspin, ms, ("FAILED", "Passed")[isorthnorm and diagsS2]))
            sys.stdout.flush ()


def unpack_csfaddrs (norb, neleca, nelecb, smult, addrs):
    min_npair, npair_csf_offset, npair_dconf_size, npair_sconf_size, npair_spincpl_size = get_csfvec_shape (
        norb, neleca, nelecb, smult)
    npair = np.empty (len (addrs), dtype=np.int32)
    domo_addrs = np.empty (len (addrs), dtype=np.int64)
    somo_addrs = np.empty (len (addrs), dtype=np.int64)
    spincpl_addrs = np.empty (len (addrs), dtype=np.int64)
    for ix, iaddr in enumerate (addrs):
        npair[ix] = np.where (npair_csf_offset <= iaddr)[0][-1]
        sconf_size = npair_sconf_size[npair[ix]]
        spincpl_size = npair_spincpl_size[npair[ix]]
        iad = np.squeeze (iaddr - npair_csf_offset[npair[ix]])
        npair[ix] += min_npair
        domo_addrs[ix] = iad // (sconf_size * spincpl_size)
        iad = iad % (sconf_size * spincpl_size)
        somo_addrs[ix] = iad // spincpl_size
        spincpl_addrs[ix] = iad % spincpl_size
    return npair, domo_addrs, somo_addrs, spincpl_addrs

def csfaddrs2str (norb, neleca, nelecb, smult, addrs):
    npair, domo_addrs, somo_addrs, spincpl_addrs = unpack_csfaddrs (norb, neleca, nelecb, smult, addrs)
    domo_str = np.zeros (len (addrs), dtype=np.int64)
    somo_str = np.zeros (len (addrs), dtype=np.int64)
    spincpl_str = np.zeros (len (addrs), dtype=np.int64)
    for uniq_npair in np.unique (npair):
        idx_uniq = (npair == uniq_npair)
        notdomo = norb - uniq_npair
        nspin = neleca + nelecb - 2*uniq_npair
        domo_str[idx_uniq] = cistring.addrs2str (norb, uniq_npair, domo_addrs[idx_uniq]) if uniq_npair > 0 else -1
        somo_str[idx_uniq] = cistring.addrs2str (notdomo, nspin, somo_addrs[idx_uniq])
        spincpl_str[idx_uniq] = addrs2str (nspin, smult, spincpl_addrs[idx_uniq])
    return domo_str, somo_str, spincpl_str

def printable_csfstring (norb, neleca, nelecb, smult, addrs):
    domo_str, somo_str, spincpl_str = csfaddrs2str (norb, neleca, nelecb, smult, addrs)
    return _printable_csfstring (norb, neleca, nelecb, smult, domo_str, somo_str, spincpl_str)

def _printable_csfstring (norb, neleca, nelecb, smult, domo_str, somo_str, spincpl_str):
    strs = []
    nullpair = np.repeat (['0'], norb)
    for d, s, p in zip (domo_str, somo_str, spincpl_str):
        mystr = nullpair if d < 0 else np.asarray (list (bin (d)[2:].zfill (norb)))
        idx_pair = (mystr == '1')
        mystr[idx_pair] = '2'
        count_notpair = np.count_nonzero (~idx_pair)
        if count_notpair:
            sstr = np.asarray (list (bin (s)[2:].zfill (count_notpair)))
            mystr[~idx_pair] = sstr
            idx_somo = mystr == '1'
            nsomo = np.count_nonzero (idx_somo)
            if nsomo:
                cplstr = np.asarray (list (bin (p)[2:].zfill (nsomo)))
                idx_up = np.where (idx_somo)[0][cplstr=='1']
                idx_down = np.where (idx_somo)[0][cplstr=='0']
                mystr[idx_up] = 'u'
                mystr[idx_down] = 'd'
        strs.append (''.join ([m for m in mystr]))
    return strs


