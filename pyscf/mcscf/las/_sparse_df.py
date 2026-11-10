# Copied from mrh/my_pyscf/df/sparse_df.py for the LASSCF migration.
# TODO (migration step 5): refactor or replace.
from pyscf import lib
import numpy as np
from scipy import linalg
from pyscf.lib import load_library
import ctypes, time
libsint = load_library ('liblas')

class sparsedf_array (np.ndarray):
    def __new__(cls, inp, nmo=None):
        assert (inp.flags['C_CONTIGUOUS'] or inp.flags['F_CONTIGUOUS'])
        self = np.asarray (inp).view (cls)
        self.naux = self.shape[0]
        self.nmo = nmo
        if self.nmo is None:
            if self.ndim == 2:
                nmo1 = 1
                off = 0
                for self.nmo1 in range (1, self.shape[1]):
                    off += nmo1
                    if off == self.shape[1]: break
                    nmo1 += 1
                self.nmo = (nmo1, nmo1)
            else:
                self.nmo = self.shape[1:]
        return self

    def __array_finalize__(self, obj):
        if obj is None: return
        self.naux = getattr(obj, 'naux', None)
        self.nmo = getattr(obj, 'nmo', None)
        self.iao_nent = getattr(obj, 'iao_nent', None)
        self.iao_entlist = getattr(obj, 'iao_entlist', None)
        self.iao_sort = getattr(obj, 'iao_sort', None)
        self.nent_max = getattr(obj, 'nent_max', None)
        self.nentpair = getattr(obj, 'nentpair', None)
        self.entpair = getattr(obj, 'entpair', None)

    def pack_mo (self):
        if self.ndim == 2: return self
        elif self.ndim == 3: return sparsedf_array (lib.pack_tril (self), nmo=self.nmo)
        else: raise (RuntimeError, 'wrong number of dimensions')

    def unpack_mo (self):
        if self.ndim == 3: return self
        elif self.ndim == 2: return (sparsedf_array (lib.unpack_tril (self), nmo=self.nmo))
        else: raise (RuntimeError, 'wrong number of dimensions')

    def transpose_mo (self):
        assert (self.ndim == 3), "Can't transpose packed lower-triangular matrix (by definition this is pointless)"
        return sparsedf_array (lib.numpy_helper.transpose (self, axes=(0,2,1), inplace=(self.nmo[0] == self.nmo[1])), nmo=(self.nmo[1], self.nmo[0]))
        
    def naux_fast (self): # Since naux is always the first index, this corresponds to making the array F-contiguous
        return sparsedf_array (np.asfortranarray (self), nmo=self.nmo)

    def naux_slow (self): # Since naux is always the first index, this corresponds to making the array C-contiguous
        return sparsedf_array (np.ascontiguousnarray (self), nmo=self.nmo)

    def get_sparsity_ (self, thresh=1e-8):
        metric = linalg.norm (self, axis=0)
        if metric.ndim == 1: metric = lib.unpack_tril (metric)
        metric = metric > thresh
        self.iao_nent = np.count_nonzero (metric, axis=1).astype (np.int32)
        self.nent_max = np.amax (self.iao_nent)
        self.iao_entlist = -np.ones ((self.nmo[0], self.nent_max), dtype=np.int32)
        for irow, nent in enumerate (self.iao_nent):
            self.iao_entlist[irow,:nent] = np.where (metric[irow])[0]
        self.iao_sort = np.argsort (self.iao_nent).astype (np.int32)
        metric = lib.pack_tril (metric)
        self.nentpair = np.count_nonzero (metric)
        self.entpair = np.where (metric)[0]

    def contract1 (self, cmat):
        ''' Contract 1 AO index with a dense matrix using sparse arithmetic on dense memory storage

        Args:
            cmat : np.ndarray of shape (nao, nmo)

        Returns:
            vPuv : np.ndarray of shape (nao, nmo, naux) stored in row-major order
        '''
        if self.ndim == 3: return self.pack_mo ()
        if not self.flags['C_CONTIGUOUS']: self = self.naux_slow ()
        cmat = np.ascontiguousarray (cmat)
        nao = self.nmo[0]
        nmo = cmat.shape[1]
        if self.nent_max is None: self.get_sparsity_ ()
        vPuv = np.zeros ((nao, nmo, self.naux), dtype=self.dtype).view (sparsedf_array)
        wrk = np.zeros ((lib.num_threads (), self.nent_max, (self.naux+nmo)), dtype = self.dtype).view (sparsedf_array)
        libsint.SINT_SDCDERI_DDMAT (self.ctypes.data_as (ctypes.c_void_p),
            cmat.ctypes.data_as (ctypes.c_void_p),
            vPuv.ctypes.data_as (ctypes.c_void_p),
            wrk.ctypes.data_as (ctypes.c_void_p),
            self.iao_sort.ctypes.data_as (ctypes.c_void_p),
            self.iao_nent.ctypes.data_as (ctypes.c_void_p),
            self.iao_entlist.ctypes.data_as (ctypes.c_void_p),
            ctypes.c_int (nao), ctypes.c_int (self.naux),
            ctypes.c_int (nmo), ctypes.c_int (self.nent_max))
        wrk = None
        return vPuv

    def contract2 (self, vPuv):
        ''' Contract the auxbasis and one AO basis indices with multiplicand vPuv, as when computing the exchange matrix of Hartree--Fock.
        
        Args:
            vPuv : np.ndarray of shape (nao, nao, naux) stored in row-major order. The FIRST index is contracted
                with self ; the output of contract1 must be TRANSPOSED ON THE FIRST 2 INDICES in order to generate
                the vk matrix

        Returns:
            vk : np.ndarray of shape (nao, nao)
        '''
        if self.ndim == 3: return self.pack_mo ()
        if not self.flags['F_CONTIGUOUS']: self = self.naux_fast ()
        if self.nent_max is None: self.get_sparsity_ ()
        nao = self.nmo[0]
        vk = np.zeros ((nao, nao), dtype=self.dtype)
        wrk = np.zeros ((lib.num_threads (), self.nent_max, self.naux), dtype=self.dtype)
        libsint.SINT_SDCDERI_VK (self.ctypes.data_as (ctypes.c_void_p),
            vPuv.ctypes.data_as (ctypes.c_void_p),
            vk.ctypes.data_as (ctypes.c_void_p),
            wrk.ctypes.data_as (ctypes.c_void_p),
            self.iao_sort.ctypes.data_as (ctypes.c_void_p),
            self.iao_nent.ctypes.data_as (ctypes.c_void_p),
            self.iao_entlist.ctypes.data_as (ctypes.c_void_p),
            ctypes.c_int (nao), ctypes.c_int (self.naux),
            ctypes.c_int (self.nent_max))
        wrk = None
        vk = lib.hermi_sum (vk, inplace=True)
        vk[np.diag_indices (nao)] /= 2
        return vk

    def vk_svd (self, mo_coeff, mo_occ, thresh=1e-8):
        t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        if self.ndim == 3: return self.pack_mo ()
        if not self.flags['C_CONTIGUOUS']: self = self.naux_slow ()
        nao = self.nmo[0]
        idx = np.abs (mo_occ) > 1e-8
        nmo = np.count_nonzero (idx)
        global_K = min (self.nent_max, nmo)
        mo = mo_coeff[:,idx] * np.sqrt (mo_occ[idx])[None,:]
        cderi_lvec = np.zeros ((nao, self.naux, global_K), dtype=self.dtype)
        mo_lvec = np.zeros ((nao, self.nent_max, nmo), dtype=self.dtype)
        imo_nent = np.zeros_like (self.iao_nent)
        lwrk = (4*global_K*global_K) + (8*global_K) + self.nent_max * (global_K + self.naux + nmo)
        lwrk = max (lwrk, self.nent_max * (self.nent_max + nmo))
        lwrk *= lib.num_threads ()
        wrk = np.zeros (lwrk, dtype=self.dtype)
        libsint.SINT_SDCDERI_MO_LVEC (self.ctypes.data_as (ctypes.c_void_p),
            mo.ctypes.data_as (ctypes.c_void_p),
            cderi_lvec.ctypes.data_as (ctypes.c_void_p),
            mo_lvec.ctypes.data_as (ctypes.c_void_p),
            wrk.ctypes.data_as (ctypes.c_void_p),
            ctypes.c_double (thresh),
            self.iao_sort.ctypes.data_as (ctypes.c_void_p),
            self.iao_nent.ctypes.data_as (ctypes.c_void_p),
            self.iao_entlist.ctypes.data_as (ctypes.c_void_p),
            imo_nent.ctypes.data_as (ctypes.c_void_p),
            ctypes.c_int (nao), ctypes.c_int (self.naux),
            ctypes.c_int (nmo), ctypes.c_int (self.nent_max))
        print ("First C function time: {} clock ; {} wall".format (lib.logger.process_clock () - t0, lib.logger.perf_counter () - w0))
        t0, w0 = lib.logger.process_clock (), lib.logger.perf_counter ()
        wrk[:self.nent_max*(self.nent_max+nmo)] = 0.0
        vk = np.zeros ((nao, nao), dtype=mo_coeff.dtype)
        libsint.SINT_SDCDERI_DDMAT_MOSVD (cderi_lvec.ctypes.data_as (ctypes.c_void_p),
            mo_lvec.ctypes.data_as (ctypes.c_void_p),
            vk.ctypes.data_as (ctypes.c_void_p),
            wrk.ctypes.data_as (ctypes.c_void_p),
            imo_nent.ctypes.data_as (ctypes.c_void_p),
            ctypes.c_int (nao), ctypes.c_int (nmo),
            ctypes.c_int (self.naux), ctypes.c_int (self.nent_max),
            ctypes.c_int (global_K))
        print ("Second C function time: {} clock ; {} wall".format (lib.logger.process_clock () - t0, lib.logger.perf_counter () - w0))
        return vk


