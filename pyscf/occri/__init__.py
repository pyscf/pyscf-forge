import pyscf
import numpy
import time
import ctypes
from pyscf import lib
from pyscf.occri import occri_k

"""
This file is part of a proprietary software owned by Sandeep Sharma (sanshar@gmail.com).
Unless the parties otherwise agree in writing, users are subject to the following terms.

(0) This notice supersedes all of the header license statements.
(1) Users are not allowed to show the source code to others, discuss its contents, 
    or place it in a location that is accessible by others.
(2) Users can freely use resulting graphics for non-commercial purposes. 
    Credits shall be given to the future software of Sandeep Sharma as appropriate.
(3) Sandeep Sharma reserves the right to revoke the access to the code any time, 
    in which case the users must discard their copies immediately.
    """

# Load the shared library
liboccri = lib.load_library('liboccri')

# Bind functions
ndpointer = numpy.ctypeslib.ndpointer

occRI_vR = liboccri.occRI_vR
occRI_vR.restype = None
occRI_vR.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]


class OCCRI(pyscf.pbc.df.fft.FFTDF):

    def __init__(self, mydf, kmesh = [1,1,1], **kwargs,):
        
        self.method = mydf.__module__.rsplit('.', 1)[-1]
        self.df_obj = mydf
        
        assert self.method in ['hf', 'uhf', 'khf', 'kuhf', 'rks', 'uks', 'krks', 'kuks']
        
        self.StartTime = time.time()
        cell = mydf.cell
        super().__init__(cell=cell)  # Need this for pyscf's eval_ao function
        self.exxdiv = "ewald"
        self.cell = cell
        self.kmesh = kmesh
        
        self.Nk = numpy.prod(self.kmesh)
        self.kpts = self.cell.make_kpts(
            self.kmesh, space_group_symmetry=False, time_reversal_symmetry=False, wrap_around=True
        )

        self.joblib_njobs = lib.numpy_helper._np_helper.get_omp_threads()

        self.get_j = pyscf.pbc.df.fft_jk.get_j_kpts

        if str(self.method[0]) == 'k':
            self.get_jk = self.get_jk_kpts
            self.get_k = occri_k.get_k_occRI_kpts
        else:
            self.get_k = occri_k.occRI_get_k_opt

    def get_jk(
        self,
        dm=None,
        hermi=1,
        kpt=None,
        kpts_band=None,
        with_j=None,
        with_k=None,
        omega=None,
        exxdiv=None,
        **kwargs,
    ):
        dm_shape = dm.shape
        dm = dm.reshape(-1, dm_shape[-2], dm_shape[-1])
        
        if with_j:
            vj = self.get_j(self, dm)

        if with_k:
            vk = self.get_k(self, dm, exxdiv)

        if with_j:
            vj = numpy.asarray(vj, dtype=dm.dtype).reshape(dm_shape)
        else:
            vj = None

        if with_k:
            vk = numpy.asarray(vk, dtype=dm.dtype).reshape(dm_shape)
        else:
            vk = None

        return vj, vk

    def __del__(self):
        return

    def copy(self):
        """Returns a shallow copy"""
        return self.view(self.__class__)

    def get_keyword_arguments(self):
        # Retrieve all attributes, excluding those in cell
        return {key: value for key, value in self.__dict__.items() if key != "cell"}
