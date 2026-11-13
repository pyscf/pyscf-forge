from pyscf import ao2mo, lib
from pyscf.mcscf.addons import StateAverageMCSCFSolver
import numpy as np
from pyscf.mcpdft.otfnal import transfnal, get_transfnal
from pyscf.mcpdft.mcpdft import _get_e_decomp
from pyscf.mcpdft.mcpdft import _PDFT, _mcscf_env

class _LASPDFT(_PDFT):
    'MC-PDFT energy for a LASSCF wavefunction'

    def get_h2eff(self, mo_coeff=None):
        'Compute the active space two-particle Hamiltonian.'
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:, ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:, ncore:nocc]

        if getattr(self._scf, '_eri', None) is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        elif getattr (self, 'with_df', False):
            eri = self.with_df.ao2mo(mo_coeff)

        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri

    def multi_state(self, method='Lin'):
        raise NotImplementedError("MS-PDFT not available for the LAS wave functions.")

def _get_mcpdft_child_class(mc, _pdft_class=_LASPDFT):
    from pyscf.mcscf.las.lasscf import LASSCFNoSymm
    mc_doc = (mc.__class__.__doc__ or 'No docstring for MC-SCF parent method')
    _is_lasscf = isinstance (mc, LASSCFNoSymm)

    class PDFT(_pdft_class, mc.__class__):
        __doc__ = mc_doc + '\n\n' + _pdft_class.__doc__
        _mc_class = mc.__class__

        def get_h2eff(self, mo_coeff=None):
            if self._in_mcscf_env:
                return mc.__class__.get_h2eff(self, mo_coeff=mo_coeff)
            else:
                return _pdft_class.get_h2eff(self, mo_coeff=mo_coeff)

        # Have to pass this due to dump_chk, which won't work for LAS.
        def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                                 grids_level=None, grids_attr=None, dump_chk=False, **kwargs):
            e_tot, e_ot, e_states = _pdft_class.compute_pdft_energy_(self, mo_coeff=mo_coeff,
                    ci=ci, ot=ot, otxc=otxc, grids_level=grids_level, grids_attr=grids_attr,
                    dump_chk=False, **kwargs)
            # LAS isn't a StateAverageMCSCFSolver, so the parent treats nroots>1 like multi-root
            # CASCI: per-state energies in e_tot, e_states untouched (still the values from before
            # the MC-SCF step). Store them in e_states as well. (Unlike SA-MC-PDFT, e_tot remains
            # the per-state list rather than the weighted average.) Not for LASSI-PDFT (mrh), where
            # e_states holds the LAS rootspace energies.
            if self.nroots > 1 and not getattr (self, 'DoLASSI', False):
                self.e_states = list (e_states)
            return e_tot, e_ot, e_states

        def analyze(self):
            raise NotImplementedError('Analyze function is not yet implemented for LAS-PDFT')

        def get_energy_decomposition(self, **kwargs):
            raise NotImplementedError('EDA is not yet implemented for LAS-PDFT')

        make_one_casdm1s = mc.__class__.state_make_casdm1s
        make_one_casdm2 = mc.__class__.state_make_casdm2

        # TODO: compatibility with MC-PDFT checkpoint dumping
        dump_chk = mc.__class__.dump_chk

        # TODO: in pyscf/mcpdft/mcpdft.py::optimize_mcscf_, generalize the number
        # of return arguments. Then the redefinition below will be unnecessary.
        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
            '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
            Has the same calling signature as the parent kernel method. '''

            with _mcscf_env(self):
                if _is_lasscf:
                    self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                        self._mc_class.kernel(self, mo_coeff, ci0=ci0, **kwargs)[:-2]
                else: # LASCI.kernel returns (converged, e_tot, e_states, e_cas, e_lexc, ci)
                    self.e_mcscf, self.e_cas, self.ci = \
                        self._mc_class.kernel(self, mo_coeff, ci0=ci0, **kwargs)[1::2]
                self.fcisolver.nroots = self.nroots

    return PDFT

def _instantiate(PDFT, mc, ot, **kwargs):
    pdft = PDFT(mc._scf, mc.ncas_sub, mc.nelecas_sub, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(mc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft

def get_mcpdft_child_class(mc, ot, DoLASSI=False, states=None,
                           _pdft_class=_LASPDFT, **kwargs):
    if DoLASSI:
        raise NotImplementedError ('LASSI-PDFT is implemented in mrh '
                                   '(mrh.my_pyscf.mcpdft.lassipdft)')

    PDFT = _get_mcpdft_child_class(mc, _pdft_class=_pdft_class)
    if mc.ci is not None:
        mc.fcisolver.nroots = mc.nroots
    else:
        mc.fcisolver.nroots = 1
    return _instantiate(PDFT, mc, ot, **kwargs)

def _laspdft (las_class, mc_or_mf_or_mol, ot, ncas_sub=None, nelecas_sub=None, ncore=None,
              spin_sub=None, frozen=None, _get_mcpdft_child_class=None, **kwargs):
    '''Shared body of LASSCFPDFT and LASCIPDFT. _get_mcpdft_child_class defaults to
    get_mcpdft_child_class; mrh passes its own, which adds LASSI-PDFT.'''
    import copy
    from pyscf import gto
    from pyscf.lib import logger
    from pyscf.mcscf.las.lasci import LASCINoSymm
    if _get_mcpdft_child_class is None: _get_mcpdft_child_class = get_mcpdft_child_class
    if isinstance (mc_or_mf_or_mol, LASCINoSymm):
        mc0 = mc1 = mc_or_mf_or_mol
        if frozen is not None: mc1.frozen = frozen
    else:
        mc0 = None
        if isinstance (mc_or_mf_or_mol, gto.Mole) and mc_or_mf_or_mol.symmetry:
            logger.warn (mc_or_mf_or_mol,
                         'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')
        las_kwargs = {'ncore': ncore, 'spin_sub': spin_sub}
        if frozen is not None: las_kwargs['frozen'] = frozen
        mc1 = las_class (mc_or_mf_or_mol, ncas_sub, nelecas_sub, **las_kwargs)
    mc2 = _get_mcpdft_child_class (mc1, ot, **kwargs)
    if mc0 is not None:
        mc2.mo_coeff = mc0.mo_coeff.copy ()
        mc2.ci = copy.deepcopy (mc0.ci)
        mc2.converged = mc0.converged
    return mc2

_CONSTRUCTOR_DOC = '''

    Args:
        mc_or_mf_or_mol : LAS object, SCF object or Mole
            An existing LAS object (whose orbitals and CI vectors are copied), or the mean-field
            reference for a new one
        ot : str or instance of :class:`OTFnal`
            On-top functional, e.g. 'tPBE'

    Kwargs:
        ncas_sub, nelecas_sub : sequences
            Fragment active spaces, as for pyscf.mcscf.las.LASSCF; taken from mc_or_mf_or_mol if
            it is a LAS object
        ncore, spin_sub, frozen :
            Passed to the LAS constructor (frozen is set on an existing LAS object)
        Others are passed to the LAS-PDFT class.
    '''

def LASSCFPDFT (mc_or_mf_or_mol, ot, ncas_sub=None, nelecas_sub=None, **kwargs):
    from pyscf.mcscf import las
    if ncas_sub is None: ncas_sub = getattr (mc_or_mf_or_mol, 'ncas_sub', None)
    if nelecas_sub is None: nelecas_sub = getattr (mc_or_mf_or_mol, 'nelecas_sub', None)
    return _laspdft (las.LASSCF, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, **kwargs)
LASSCFPDFT.__doc__ = 'LAS-PDFT on a LASSCF wave function.' + _CONSTRUCTOR_DOC

def LASCIPDFT (mc_or_mf_or_mol, ot, ncas_sub=None, nelecas_sub=None, **kwargs):
    from pyscf.mcscf import las
    if ncas_sub is None: ncas_sub = getattr (mc_or_mf_or_mol, 'ncas_sub', None)
    if nelecas_sub is None: nelecas_sub = getattr (mc_or_mf_or_mol, 'nelecas_sub', None)
    return _laspdft (las.LASCI, mc_or_mf_or_mol, ot, ncas_sub, nelecas_sub, **kwargs)
LASCIPDFT.__doc__ = 'LAS-PDFT on a LASCI wave function.' + _CONSTRUCTOR_DOC

LASSCF = LASSCFPDFT
LASCI = LASCIPDFT

# Importing this module also adds the constructors to the pyscf.mcpdft namespace, next to
# mcpdft.CASSCF and mcpdft.CASCI, so that after "from pyscf.mcpdft import laspdft" one can write
# mcpdft.LASSCF (...). Names that pyscf.mcpdft already has are left alone.
def _add_to_mcpdft_namespace ():
    from pyscf import mcpdft
    for name in ('LASSCFPDFT', 'LASCIPDFT', 'LASSCF', 'LASCI'):
        if not hasattr (mcpdft, name):
            setattr (mcpdft, name, globals ()[name])
_add_to_mcpdft_namespace ()
