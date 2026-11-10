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
        raise NotImplementedError(f"MS-PDFT not avaialble for the LAS wave functions.")

def _get_mcpdft_child_class(mc, _pdft_class=_LASPDFT):
    mc_doc = (mc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

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
            return _pdft_class.compute_pdft_energy_(self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
                    grids_level=grids_level, grids_attr=grids_attr, dump_chk=False, **kwargs)

        def analyze(self):
            raise NotImplementedError('Analyze function is not yet implemented for LAS-PDFT')

        def get_energy_decomposition(self, **kwargs):
            raise NotImplementedError('EDA is not yet implemented for LAS-PDFT')

        make_one_casdm1s = mc.__class__.state_make_casdm1s
        make_one_casdm2 = mc.__class__.state_make_casdm2

        # TODO: compatibility with MC-PDFT checkpoint dumping
        dump_chk = mc.__class__.dump_chk

        # TODO: in pyscf-forge/pyscf/mcpdft/mcpdft.py::optimize_mcscf_, generalize the number
        # of return arguments. Then the redefinition below will be unnecessary. 
        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
            '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
            Has the same calling signature as the parent kernel method. '''

            with _mcscf_env(self):
                self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                    self._mc_class.kernel(self, mo_coeff, ci0=ci0, **kwargs)[:-2]
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
