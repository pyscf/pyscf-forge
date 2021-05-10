import h5py
from pyscf import lib

import numpy as np
import tempfile
from time import time, process_time
from functools import wraps, partial


# To avoid too slow single-threaded einsum if pyscf-tblis is not available
lib.numpy_helper._numpy_einsum = partial(np.einsum, optimize=True)
# Doubly hybrid functionals xc code in detail
XC_DH_MAP = {   # [xc_s, xc_n, cc, c_os, c_ss]
    "mp2": ("HF", None, 1, 1, 1),
    "xyg3": ("B3LYPg", "0.8033*HF - 0.0140*LDA + 0.2107*B88, 0.6789*LYP", 0.3211, 1, 1),
    "xygjos": ("B3LYPg", "0.7731*HF + 0.2269*LDA, 0.2309*VWN3 + 0.2754*LYP", 0.4364, 1, 0),
    "xdhpbe0": ("PBE0", "0.8335*HF + 0.1665*PBE, 0.5292*PBE", 0.5428, 1, 0),
    "b2plyp": ("0.53*HF + 0.47*B88, 0.73*LYP", None, 0.27, 1, 1),
    "mpw2plyp": ("0.55*HF + 0.45*mPW91, 0.75*LYP", None, 0.25, 1, 1),
    "pbe0dh": ("0.5*HF + 0.5*PBE, 0.875*PBE", None, 0.125, 1, 1),
    "pbeqidh": ("0.693361*HF + 0.306639*PBE, 0.666667*PBE", None, 0.333333, 1, 1),
    "pbe02": ("0.793701*HF + 0.206299*PBE, 0.5*PBE", None, 0.5, 1, 1),
}


class TicToc:

    def __init__(self):
        self.t = time()
        self.p = process_time()

    def tic(self):
        self.t = time()
        self.p = process_time()

    def toc(self, msg=""):
        t = time() - self.t
        p = process_time() - self.p
        print("Wall: {:12.4f}, CPU: {:12.4f}, Ratio: {:6.1f}, msg: {:}".format(t, p, p / t * 100, msg))
        self.tic()


class HybridDict(dict):
    """
    HybridDict: Inherited dictionary class

    A dictionary specialized to store data both in memory and in disk.
    """
    def __init__(self, chkfile_name=None, dir=None, **kwargs):
        super(HybridDict, self).__init__(**kwargs)
        # initialize input variables
        if dir is None:
            dir = lib.param.TMPDIR
        if chkfile_name is None:
            self._chkfile = tempfile.NamedTemporaryFile(dir=dir)
            chkfile_name = self._chkfile.name
        # create or open exist chkfile
        self.chkfile_name = chkfile_name
        self.chkfile = h5py.File(self.chkfile_name, "r+")

    def create(self, name, data=None, incore=True, shape=None, dtype=None, **kwargs):
        # if name in self:
        #     try:  # don't create a new space if tensor already exists
        #         if self[name].shape == shape:
        #             self[name][:] = 0
        #     except:
        #         pass
        #     self.delete(name)
        dtype = dtype if dtype is not None else np.float64
        if not incore:
            self.chkfile.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)
            self.setdefault(name, self.chkfile[name])
        elif data is not None:
            self.setdefault(name, data)
        elif data is None and shape is not None:
            self.setdefault(name, np.zeros(shape=shape, dtype=dtype))
        else:
            raise ValueError("Could not handle create!")
        return self.get(name)

    def delete(self, key):
        val = self.pop(key)
        if isinstance(val, h5py.Dataset):
            del self.chkfile[key]

    def load(self, key):
        return np.asarray(self.get(key))


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0, p0 = time(), process_time()
        result = f(*args, **kwargs)
        t1, p1 = time(), process_time()
        with open("tmp_timing.log", "a") as log:
            log.write(" {0:50s}, Wall: {1:10.3f} s, CPU: {2:10.3f} s, ratio {3:7.1f}%\n"
                  .format(f.__qualname__, t1-t0, p1-p0, (p1-p0)/(t1-t0) * 100))
        return result
    return wrapper


def parse_xc_dh(xc_dh: str):
    xc_dh = xc_dh.replace("-", "").replace("_", "").lower()
    return XC_DH_MAP[xc_dh]


def gen_batch(minval, maxval, nbatch):
    return [slice(i, (i + nbatch) if i + nbatch < maxval else maxval) for i in range(minval, maxval, nbatch)]


def gen_shl_batch(mol, nbatch):
    res = []
    p, q = 0, 0
    sp, sq = 0, 0
    for sv, v in enumerate(mol.ao_loc_nr()):
        if v - p <= nbatch:
            q = v
            sq = sv
        else:
            res.append((slice(sp, sq), slice(p, q)))
            p, q = q, v
            sp, sq = sq, sv
    res.append((slice(sp, sq), slice(p, q)))
    return res


def calc_batch_size(unit_flop, mem_avail, pre_flop=0):
    # mem_avail: in MB
    max_memory = 0.8 * mem_avail - pre_flop * 8 / 1024 ** 2
    batch_size = int(max(max_memory // (unit_flop * 8 / 1024 ** 2), 1))
    return batch_size
