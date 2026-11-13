class _DFLASCI: # Tag
    pass

# The constructors import their modules when called, because those modules import _DFLASCI from
# this package.

_ARGS_DOC = '''

    Args:
        mf_or_mol : SCF object or Mole
            Mean-field reference. A Mole is wrapped in an RHF object that is not run.
        ncas_sub : sequence of int
            Number of active orbitals in each fragment
        nelecas_sub : sequence of int or of (int, int)
            Number of active electrons in each fragment, total or (alpha, beta)

    Kwargs:
        Passed to the class, e.g. spin_sub (spin multiplicity of each fragment), wfnsym_sub,
        ncore, frozen

    Returns:
        The class with point-group symmetry if mf_or_mol.mol.symmetry is set, and with density
        fitting if the SCF object has with_df.
    '''

def LASCI (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    from pyscf.mcscf.las import lasci
    return lasci.LASCI (mf_or_mol, ncas_sub, nelecas_sub, **kwargs)
LASCI.__doc__ = ('''Localized active space CI: optimize the CI vectors of each fragment for
    fixed orbitals.''' + _ARGS_DOC)

def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    from pyscf.mcscf.las import lasscf
    return lasscf.LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs)
LASSCF.__doc__ = ('''Localized active space self-consistent field: optimize the CI vectors of
    each fragment and the orbitals.''' + _ARGS_DOC)
