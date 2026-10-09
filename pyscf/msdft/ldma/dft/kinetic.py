import torch

from .operator1e import OneElectronOperatorAO


class KineticFunctionalAO(torch.nn.Module):
    def __init__(self, mol):
        super().__init__()
        self.mol = mol

    def forward(self, density_matrices_ao):
        return OneElectronOperatorAO.apply(density_matrices_ao, self.mol, "int1e_kin")
