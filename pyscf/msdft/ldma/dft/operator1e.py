"""Differentiable contractions with one-electron AO integrals."""

import torch
import pyscf.pbc.gto
from torch.autograd import Function


class OneElectronOperatorAO(Function):
    @staticmethod
    def forward(ctx, density_matrices_ao, mol, intor):
        if isinstance(mol, pyscf.pbc.gto.cell.Cell):
            integrals_ao = mol.pbc_intor(intor)
        else:
            integrals_ao = mol.intor_symmetric(intor)
        integrals = torch.from_numpy(integrals_ao).to(
            dtype=density_matrices_ao.dtype, device=density_matrices_ao.device
        )
        ctx.save_for_backward(integrals)
        return torch.einsum("ab,abij->ij", integrals, density_matrices_ao)

    @staticmethod
    def backward(ctx, grad_output):
        integrals, = ctx.saved_tensors
        return torch.einsum("ab,ij->abij", integrals, grad_output), None, None


__all__ = ["OneElectronOperatorAO"]
