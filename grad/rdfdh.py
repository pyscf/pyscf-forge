from __future__ import annotations

import dh.rdfdh
from dh.rdfdh import get_cderi_mo
from dh.dhutil import calc_batch_size, gen_batch
from pyscf import gto, lib
import numpy as np

einsum = lib.einsum


def get_H_1_ao(mol: gto.Mole):
    natm, nao = mol.natm, mol.nao
    int1e_ipkin = mol.intor("int1e_ipkin")
    int1e_ipnuc = mol.intor("int1e_ipnuc")
    Z_A = mol.atom_charges()
    H_1_ao = np.zeros((natm, 3, nao, nao))
    for A, (_, _, A0, A1) in enumerate(mol.aoslice_by_atom()):
        sA = slice(A0, A1)
        H_1_ao[A, :, sA, :] -= int1e_ipkin[:, sA, :]
        H_1_ao[A, :, sA, :] -= int1e_ipnuc[:, sA, :]
        with mol.with_rinv_as_nucleus(A):
            H_1_ao[A] -= Z_A[A] * mol.intor("int1e_iprinv")
    H_1_ao += H_1_ao.swapaxes(-1, -2)
    H_1_ao.shape = (natm * 3, nao, nao)
    return H_1_ao


class Gradients(dh.rdfdh.RDFDH):

    def __init__(self,
                 mol: gto.Mole,
                 *args, **kwargs):
        super(Gradients, self).__init__(mol, *args, **kwargs)
        # tune flags
