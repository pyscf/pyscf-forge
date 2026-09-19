"""Exact-exchange matrix functional in an atomic-orbital basis."""

import numpy
import pyscf.pbc.gto
import pyscf.scf
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class _ExactExchangeFunctionalAO(Function):
    @staticmethod
    def forward(ctx, density_matrices_ao, mol):
        nbasis, _, nstate, _ = density_matrices_ao.size()
        matrices = [
            density_matrices_ao[:, :, i, j].detach().cpu().numpy()
            for i in range(nstate)
            for j in range(nstate)
        ]
        potentials = pyscf.scf.jk.get_jk(
            mol, matrices, len(matrices) * ["ijkl,kj->il"])
        exchange_potentials = numpy.zeros((nbasis, nbasis, nstate, nstate))
        offset = 0
        for i in range(nstate):
            for j in range(nstate):
                exchange_potentials[:, :, i, j] = potentials[offset]
                offset += 1
        exchange_potentials = torch.from_numpy(exchange_potentials).to(density_matrices_ao)
        ctx.save_for_backward(exchange_potentials)
        return -0.5 * torch.einsum(
            "rsik,rskj->ij", density_matrices_ao, exchange_potentials)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        exchange_potentials, = ctx.saved_tensors
        gradient = -0.5 * (
            torch.einsum("mj,rsnj->rsmn", grad_output, exchange_potentials)
            + torch.einsum("rsjm,jn->rsmn", exchange_potentials, grad_output)
        )
        return gradient, None


class ExactExchangeFunctionalAO(torch.nn.Module):
    def __init__(self, mol):
        super().__init__()
        if isinstance(mol, pyscf.pbc.gto.cell.Cell):
            raise NotImplementedError("Exact exchange not implemented for periodic cells")
        self.mol = mol

    def forward(self, density_matrices_ao):
        return _ExactExchangeFunctionalAO.apply(density_matrices_ao, self.mol)


__all__ = ["ExactExchangeFunctionalAO"]
