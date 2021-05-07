from __future__ import annotations

from dh.dhutil import parse_xc_dh, gen_batch, gen_shl_batch, calc_batch_size, HybridDict
from pyscf import lib, gto, df, dft
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
import numpy as np
import h5py

einsum = lib.einsum


def kernel(mf_dh: RDFDH):
    mf_dh.mf.run()
    pass


def get_cholesky_eri_mo(mol: gto.Mole, aux: gto.Mole, C: np.ndarray, Y_mo: np.ndarray or h5py.Dataset):
    nmo, naux = C.shape[0], aux.nao
    L_inv = np.linalg.inv(np.linalg.cholesky(aux.intor("int2c2e")))
    int3c2e_generator = int3c_wrapper(mol, aux, "int3c2e", "s1")
    max_memory = mol.max_memory - lib.current_memory()[0]
    nbatch_shl = calc_batch_size(2 * nmo**2, max_memory, L_inv.size)
    for saux_shl, saux in gen_shl_batch(aux, nbatch_shl):
        int3c2e = int3c2e_generator((0, mol.nbas, 0, mol.nbas, saux_shl.start, saux_shl.stop))
        nbatch_mo = calc_batch_size(2 * nmo*naux, max_memory, L_inv.size + int3c2e.size)
        for sp in gen_batch(0, nmo, nbatch_mo):
            Y_mo[sp] += einsum("uvQ, PQ, up, vq -> pqP", int3c2e, L_inv[:, saux], C[:, sp], C)
    return Y_mo


class RDFDH(lib.StreamObject):

    def __init__(self,
                 mol: gto.Mole,
                 xc: str or tuple = "XYG3",
                 auxbasis_jk: str or dict = "weigend",
                 auxbasis_ri: str or dict or None = None,
                 grids: dft.Grids = None,
                 max_memory: float = None):
        # Parse xc code
        # It's tricky to say that, self.xc refers to SCF xc, and self.xc_dh refers to double hybrid xc
        self.xc_dh = xc
        xc_list = parse_xc_dh(xc) if isinstance(xc, str) else xc
        self.xc, self.xc_n, self.cc, self.c_os, self.c_ss = xc_list
        # parse scf method
        mf = dft.RKS(mol, xc=self.xc).density_fit(auxbasis=auxbasis_jk)
        mf.grids = grids if grids else mf.grids
        self.mf = mf
        # parse auxiliary basis
        self.df_jk = mf.with_df.build()
        self.aux_jk = self.df_jk.auxmol
        self.aux_ri = df.make_auxmol(mol, auxbasis=auxbasis_ri) if auxbasis_ri else self.aux_jk
        # parse non-consistent method
        self.mf_n = None
        if self.xc_n or self.xc_n == self.xc:
            self.mf_n = dft.RKS(mol, xc=self.xc_n).density_fit(auxbasis=auxbasis_jk)
            self.mf_n.grids = self.mf.grids
        # parse maximum memory
        self.max_memory = max_memory if max_memory else mol.max_memory
        # other preparation
        self.tensors = HybridDict()
        self.mol = mol

