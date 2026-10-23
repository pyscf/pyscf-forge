import torch

from .operator1e import OneElectronOperatorAO


class NuclearFunctionalAO(torch.nn.Module):
    def __init__(self, mol):
        super().__init__()
        self.mol = mol

    def forward(self, density_matrices_ao):
        result = OneElectronOperatorAO.apply(density_matrices_ao, self.mol, "int1e_nuc")
        if self.mol.has_ecp():
            result = result + OneElectronOperatorAO.apply(density_matrices_ao, self.mol, "ECPscalar")
        return result
