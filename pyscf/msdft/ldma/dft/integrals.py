"""Cached one-electron integral contractions for target-state densities."""

import torch
import pyscf.pbc.gto


class OneElectronIntegralCache(torch.nn.Module):
    def __init__(self, mol, intor=None):
        super().__init__()
        self.mol = mol
        if isinstance(mol, pyscf.pbc.gto.cell.Cell):
            if intor is None:
                values = mol.pbc_intor("int1e_kin") + mol.pbc_intor("int1e_nuc")
            else:
                values = mol.pbc_intor(intor)
        elif intor is None:
            values = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
            if mol.has_ecp():
                values = values + mol.intor_symmetric("ECPscalar")
        else:
            values = mol.intor_symmetric(intor)
        self.register_buffer("integrals_ao", torch.from_numpy(values).double())

    def integrals_mo(self, mo_coeff):
        values = self.integrals_ao.to(dtype=mo_coeff.dtype, device=mo_coeff.device)
        return torch.einsum("ap,ab,bq->pq", mo_coeff, values, mo_coeff)

    def matrix_elements_from_gamma(self, gamma_mo, mo_coeff):
        gamma_total = torch.einsum("ss...->...", gamma_mo)
        return torch.einsum("pq,pqij->ij", self.integrals_mo(mo_coeff), gamma_total)


__all__ = ["OneElectronIntegralCache"]
