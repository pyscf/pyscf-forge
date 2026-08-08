import numpy
import pytest
import torch
from pyscf import gto
from pyscf.pbc import gto as pbcgto

from pyscf.msdft.ldma.dft.exact_exchange import ExactExchangeFunctionalAO
from pyscf.msdft.ldma.dft.hartree import HartreeFunctionalAO
from pyscf.msdft.ldma.dft.integrals import OneElectronIntegralCache
from pyscf.msdft.ldma.dft.operator1e import OneElectronOperatorAO


def molecule():
    return gto.M(atom="H 0 0 -0.35; H 0 0 0.35", basis="sto-3g",
                 spin=0, verbose=0)


def test_one_electron_operator_value_and_gradcheck():
    mol = molecule()
    nao = mol.nao_nr()
    density = torch.randn(nao, nao, 2, 2, dtype=torch.double, requires_grad=True)
    value = OneElectronOperatorAO.apply(density, mol, "int1e_kin")
    reference = torch.einsum(
        "ab,abij->ij", torch.from_numpy(mol.intor_symmetric("int1e_kin")), density)
    torch.testing.assert_close(value, reference)
    assert torch.autograd.gradcheck(
        lambda dm: OneElectronOperatorAO.apply(dm, mol, "int1e_kin"), (density,))


def test_hartree_and_exact_exchange_gradients_are_finite():
    mol = molecule()
    nao = mol.nao_nr()
    factor = torch.randn(nao, nao, 2, 2, dtype=torch.double)
    density = 0.5 * (factor + factor.permute(1, 0, 3, 2))
    density.requires_grad_(True)
    energy = HartreeFunctionalAO(mol)(density).sum() + ExactExchangeFunctionalAO(mol)(density).sum()
    energy.backward()
    assert torch.isfinite(density.grad).all()


@pytest.mark.parametrize("functional_class", [HartreeFunctionalAO,
                                               ExactExchangeFunctionalAO])
def test_two_electron_functional_gradcheck(functional_class):
    mol = molecule()
    nao = mol.nao_nr()
    density = torch.randn(nao, nao, 1, 1, dtype=torch.double,
                          requires_grad=True)
    assert torch.autograd.gradcheck(functional_class(mol), (density,),
                                    eps=1.0e-6, atol=1.0e-5, rtol=1.0e-4)


def periodic_cell():
    cell = pbcgto.Cell()
    cell.atom = "H 0 0 0; H 0 0 1.4"
    cell.a = numpy.eye(3) * 4.0
    cell.basis = "sto-3g"
    cell.spin = 0
    cell.mesh = [7, 7, 7]
    cell.verbose = 0
    cell.build()
    return cell


def test_periodic_one_electron_and_integral_cache_branches():
    cell = periodic_cell()
    nao = cell.nao_nr()
    density = torch.eye(nao, dtype=torch.double)[:, :, None, None]
    kinetic = OneElectronOperatorAO.apply(density, cell, "int1e_kin")
    reference = torch.trace(torch.from_numpy(cell.pbc_intor("int1e_kin"))).reshape(1, 1)
    torch.testing.assert_close(kinetic, reference)
    cache = OneElectronIntegralCache(cell, "int1e_kin")
    torch.testing.assert_close(cache.integrals_ao,
                               torch.from_numpy(cell.pbc_intor("int1e_kin")))
    assert torch.isfinite(HartreeFunctionalAO(cell)(density)).all()
    with pytest.raises(NotImplementedError, match="periodic"):
        ExactExchangeFunctionalAO(cell)
