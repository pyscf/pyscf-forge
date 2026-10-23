"""Differentiable Hartree-like matrix functional in the AO basis."""

import numpy
import torch
from pyscf.scf import hf
import pyscf.pbc.gto
import pyscf.pbc.scf
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class _HartreeFunctionalAO(Function):
    @staticmethod
    def forward(ctx, density_matrices_ao, mol):
        nbasis, _, nstate, _ = density_matrices_ao.size()
        matrices = [density_matrices_ao[:, :, i, j].detach().cpu().numpy()
                    for i in range(nstate) for j in range(i, nstate)]
        if isinstance(mol, pyscf.pbc.gto.cell.Cell):
            potentials = pyscf.pbc.scf.hf.get_j(
                mol, matrices, kpt=numpy.array([0.0, 0.0, 0.0]))
        else:
            potentials, _ = hf.get_jk(mol, matrices, with_j=True, with_k=False)
        potential_tensor = numpy.zeros((nbasis, nbasis, nstate, nstate))
        offset = 0
        for i in range(nstate):
            for j in range(i, nstate):
                potential_tensor[:, :, i, j] = potentials[offset]
                potential_tensor[:, :, j, i] = potentials[offset]
                offset += 1
        potential_tensor = torch.from_numpy(potential_tensor).to(density_matrices_ao)
        ctx.save_for_backward(potential_tensor)
        return 0.5 * torch.einsum("abik,abkj->ij", density_matrices_ao, potential_tensor)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        potentials, = ctx.saved_tensors
        gradient = 0.5 * (
            torch.einsum("mj,abjn->abmn", grad_output, potentials)
            + torch.einsum("abmj,jn->abmn", potentials, grad_output)
        )
        return gradient, None


class HartreeFunctionalAO(torch.nn.Module):
    def __init__(self, mol):
        super().__init__()
        self.mol = mol

    def forward(self, density_matrices_ao):
        return _HartreeFunctionalAO.apply(density_matrices_ao, self.mol)


__all__ = ["HartreeFunctionalAO"]
