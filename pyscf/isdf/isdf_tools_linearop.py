#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

############ sys module ############

import copy
import numpy as np
import numpy
import scipy
import ctypes, sys
from pyscf import lib
libisdf = lib.load_library('libisdf')

def square_inPlace(a):
    
    assert(a.dtype == numpy.double)
    fn = getattr(libisdf, "NPdsquare_inPlace", None)
    assert(fn is not None)

    fn(a.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_size_t(a.size))

    return a

def d_i_ij_ij(a, b, out=None):
    assert(a.dtype == b.dtype)
    assert(a.shape[0] == b.shape[0])
    assert(a.ndim == 1)
    assert(b.ndim == 2)

    if a.dtype != numpy.double:
        raise NotImplementedError
    else:
        fn = getattr(libisdf, "NPd_i_ij_ij", None)
        assert(fn is not None)

    if out is None:
        out = numpy.empty_like(a)

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_size_t(b.shape[0]),
       ctypes.c_size_t(b.shape[1]))

    return out

def d_ij_j_ij(a, b, out=None):
    assert(a.dtype == b.dtype)
    assert(a.shape[1] == b.shape[0])
    assert(a.ndim == 2)
    assert(b.ndim == 1)

    if a.dtype != numpy.double:
        raise NotImplementedError
    else:
        fn = getattr(libisdf, "NPd_ij_j_ij", None)
        assert(fn is not None)

    if out is None:
        out = numpy.empty_like(a)

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_size_t(a.shape[0]),
       ctypes.c_size_t(a.shape[1]))

    return out

def cwise_mul(a, b, out=None):
    assert(a.size == b.size)
    assert(a.dtype == b.dtype)

    if a.dtype != numpy.double:
        raise NotImplementedError
    else:
        fn = getattr(libisdf, "NPdcwisemul", None)
        assert(fn is not None)

    if out is None:
        out = numpy.empty_like(a)

    fn(out.ctypes.data_as(ctypes.c_void_p),
       a.ctypes.data_as(ctypes.c_void_p),
       b.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_size_t(a.size))

    return out