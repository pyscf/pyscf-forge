from pyscf import lib
import numpy as np
import ctypes


libpw = lib.load_library("libpwscf")

def add_rotated_realspace_func_(fin, fout, mesh, rot, wt):
    assert fin.dtype == np.float64
    assert fout.dtype == np.float64
    assert fin.flags.c_contiguous
    assert fout.flags.c_contiguous
    shape = np.asarray(mesh, dtype=np.int32, order="C")
    assert fout.size == np.prod(shape)
    assert fin.size == np.prod(shape)
    rot = np.asarray(rot, dtype=np.int32, order="C")
    assert rot.shape == (3, 3)
    libpw.add_rotated_realspace_func(
        fin.ctypes, fout.ctypes, shape.ctypes, rot.ctypes, ctypes.c_double(wt)
    )


