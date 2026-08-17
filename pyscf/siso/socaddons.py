#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

'''
Helper Functions for SOC
'''

import numpy as np
from pyscf import lib, mcscf, mrpt, symm
from pyscf.csf_fci import csf_solver
from pyscf.siso import amfi as amfIntegrals

logger = lib.logger

def _validate_modelspace(modelspace, mol=None, ncas=None, nelecas=None):
    """
    Validate and normalize a SISO model-space specification.
    """
    errmsg = ("modelspace must be a non-empty list of "
              "(nroots, spin_multiplicity[, wfnsym]) entries")
    if not isinstance(modelspace, (list, tuple)) or not modelspace:
        raise TypeError(errmsg)

    states = []
    for index, state in enumerate(modelspace):
        if not isinstance(state, (list, tuple)) or len(state) not in (2, 3):
            msg = f"modelspace entry {index} must have number of roots \
                and respective spin multiplicities"
            raise TypeError(msg)

        nroots, spinmult = state[:2]
        wfnsym = state[2] if len(state) == 3 else None
        if isinstance(nroots, bool) or not isinstance(nroots, (int, np.integer)):
            msg = f"nroots in modelspace entry {index} must be an integer"
            raise TypeError(msg)
        if nroots <= 0:
            msg = f"nroots in modelspace entry {index} must be positive"
            raise ValueError(msg)
        if isinstance(spinmult, bool) or not isinstance(
                spinmult, (int, np.integer)):
            msg = f"spin multiplicity in modelspace entry {index} must be an integer"
            raise TypeError(msg)
        if spinmult <= 0:
            msg = f"spin multiplicity in modelspace entry {index} must be positive"
            raise ValueError(msg)
        if (wfnsym is not None and
                (isinstance(wfnsym, bool) or
                 not isinstance(wfnsym, (str, int, np.integer)))):
            msg = f"wfnsym in modelspace entry {index} must be a string or integer"
            raise TypeError(msg)

        if wfnsym is not None and mol is not None:
            if not mol.symmetry:
                msg = ("wfnsym was specified, but molecular symmetry is not enabled")
                raise ValueError(msg)
            try:
                if isinstance(wfnsym, str):
                    symm.irrep_name2id(mol.groupname, wfnsym)
                else:
                    symm.irrep_id2name(mol.groupname, int(wfnsym))
            except (KeyError, ValueError) as err:
                msg = (
                    f"wfnsym {wfnsym!r} in modelspace entry {index} is not "
                    f"valid for point group {mol.groupname}")
                raise ValueError(msg) from err

        if isinstance(wfnsym, (int, np.integer)):
            wfnsym = int(wfnsym)
        states.append((int(nroots), int(spinmult), wfnsym))

    if len({spinmult % 2 for _, spinmult, _ in states}) != 1:
        msg = ("modelspace cannot mix odd- and even-electron spin multiplicities")
        raise ValueError(msg)

    for spinmult in {state[1] for state in states}:
        spin_states = [state for state in states if state[1] == spinmult]
        symmetries = [state[2] for state in spin_states]
        if any(wfnsym is None for wfnsym in symmetries) and any(
                wfnsym is not None for wfnsym in symmetries):
            msg = ("modelspace cannot mix symmetry-qualified and unqualified "
                f"entries for spin multiplicity {spinmult}")
            raise ValueError(msg)
        specified_symmetries = [wfnsym for wfnsym in symmetries
                                if wfnsym is not None]
        if len(set(specified_symmetries)) != len(specified_symmetries):
            msg = (
                "duplicate wfnsym entries were supplied for spin multiplicity "
                f"{spinmult}; combine their roots into one entry")
            raise ValueError(msg)

    if nelecas is not None:
        nelec = int(np.sum(nelecas))
        for _, spinmult, _ in states:
            if (spinmult - 1) % 2 != nelec % 2:
                raise ValueError(
                    f"spin multiplicity {spinmult} is incompatible with "
                    f"{nelec} active electrons")
        if ncas is not None:
            max_twos = min(nelec, 2 * int(ncas) - nelec)
            for _, spinmult, _ in states:
                if spinmult - 1 > max_twos:
                    raise ValueError(
                        f"spin multiplicity {spinmult} is not possible for "
                        f"{nelec} electrons in {ncas} active orbitals")

    # State-average solvers and SISO both require states to be grouped by spin.
    return sorted(states, key=lambda state: state[1])


def _validate_state_weights(weights, nstates):
    '''
    Validate and normalize a SISO state weight specification.
    '''
    if weights is None:
        return np.ones(nstates) / nstates
    try:
        weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as err:
        raise TypeError("weights must be a one-dimensional numeric sequence") from err
    if weights.ndim != 1 or weights.size != nstates:
        raise ValueError(f"weights must contain one value for each of the {nstates} states")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("weights must sum to one")
    return weights


def _aggregate_modelspace(modelspace):
    """Combine symmetry sectors that have the same spin multiplicity."""
    aggregated = []
    for nroots, spinmult, _ in modelspace:
        if aggregated and aggregated[-1][1] == spinmult:
            aggregated[-1] = (aggregated[-1][0] + nroots, spinmult)
        else:
            aggregated.append((nroots, spinmult))
    return aggregated


def compute_nevpt2_energies(mc, modelspace):
    """
    Compute model-space NEVPT2 energies.

    A separate CASCI calculation is performed for every spin and symmetry
    sector in modelspace using the optimized orbitals in mc.mo_coeff.
    The resulting strongly-contracted NEVPT2 total energies are ordered in
    the same way as the state-average solvers. This function does not modify
    ``mc.e_states``.

    Args:
        mc: converged CAS object
        modelspace: Model-space entries ``(nroots, spinmult[, wfnsym])``

    Returns:
        e_states: list of float
            NEVPT2 total energies for the model-space states.
    """
    mol = mc._scf.mol
    states = _validate_modelspace(modelspace, mol=mol, ncas=mc.ncas,
                                  nelecas=mc.nelecas)

    if getattr(mc, 'mo_coeff', None) is None:
        raise ValueError("mc.mo_coeff is required to compute NEVPT2 energies")

    nelec = int(np.sum(mc.nelecas))
    energies = []
    for nroots, spinmult, wfnsym in states:
        twos = spinmult - 1
        nelecas = ((nelec + twos) // 2, (nelec - twos) // 2)
        casci = mcscf.CASCI(mc._scf, mc.ncas, nelecas, ncore=mc.ncore)
        casci.fcisolver = csf_solver(mol, smult=spinmult)
        casci.fcisolver.spin = twos
        if wfnsym is not None:
            casci.fcisolver.wfnsym = wfnsym
        casci.fcisolver.nroots = nroots

        e_casci = np.asarray(casci.kernel(mc.mo_coeff)[0]).reshape(-1)
        if e_casci.size != nroots:
            raise RuntimeError(
                f"CASCI returned {e_casci.size} energies for a model-space "
                f"sector requesting {nroots} roots")
        for root, energy in enumerate(e_casci):
            energies.append(energy + mrpt.NEVPT(casci, root=root).kernel())

    return energies


def socintegrals(mol, somf=True, amf=True, mmf=False, soc1e=True, soc2e=True, ham='DKH', dm=None):
    '''
    Wrapper for the SOC integral generation
    1e and 2e integrals are generated using the amfi module.
    In case of mmfi, the parent wavefunction density matrix is used.
    args:
        mol:
            molecule object
        somf: bool
            spin-orbit mean field integrals.
        amf: bool
            atomic mean field integrals.
            In this case, amf dm is generated.
        mmf: bool
            molecular mean field integrals.
        soc1e: bool
            include 1e SOC integrals.
        soc2e: bool
            include 2e SOC integrals.
        ham: str
            SOC Hamiltonian (BP or DKH)
        dm: np.array (nao, nao), optional
            density matrix of parent wavefunction.
    returns:
        hso:
            SOC integrals of dimension (3, nao, nao)
    '''

    # Sanity checks
    flags = {'somf': somf, 'amf': amf, 'mmf': mmf,
             'soc1e': soc1e, 'soc2e': soc2e}
    if any(not isinstance(value, (bool, np.bool_)) for value in flags.values()):
        raise TypeError("somf, amf, mmf, soc1e, and soc2e must be boolean")
    if not isinstance(ham, str) or ham.upper() not in ('BP', 'DKH'):
        raise ValueError("ham must be 'BP' or 'DKH'")
    ham = ham.upper()
    if not somf:
        raise NotImplementedError("Explicit 2e SOC integrals are not implemented yet")

    if mol.has_ecp():
        raise NotImplementedError("ECP is not supported yet.")

    if not soc1e and not soc2e:
        raise ValueError("At least one of soc1e and soc2e must be enabled")
    if amf == mmf:
        raise ValueError("Exactly one of amf and mmf must be enabled")
    if mmf and dm is None:
        raise ValueError("dm is required when mmf=True")

    if amf:
        dm0 = amfIntegrals.compute_amfi_dm(mol)
    elif mmf:
        dm0 = dm

    log = logger.Logger(mol.stdout, mol.verbose)
    cpu0 = logger.process_clock(), logger.perf_counter()
    hso1e, hso2e = amfIntegrals.compute_soc_integrals(mol, dm0, ham=ham)
    log.timer("SOC integrals generation took: ", *cpu0)

    if soc1e and soc2e:
        hso = hso1e+hso2e
    elif soc1e:
        hso = hso1e
    elif soc2e:
        hso = hso2e
    return np.array([x.T for x in hso])

def state_average_solver(mc, states, weights=None, ms=None):
    '''
    Wrapper for the generating the SACASSCF solver.
    args:
        mc: pyscf.mcscf.CASSCF object
            CASSCF object to be used for SACASSCF.
        states: list of tuples
            Each tuple contains (nroots, spinmult, wfnsym).
            nroots: int, number of roots for the state.
            spinmult: int, spin multiplicity of the state.
            wfnsym: int or None, symmetry of the wavefunction.
        weights: np.array or None
            Weights for each state. If None, equal weights are assigned.
        ms: str or None
            Method for mixing states. If 'lin', linear mixing is used.
            Otherwise, state average mixing is used.
    returns:
        mc: state-averaged/mix CAS object
    '''
    mol = mc._scf.mol

    def _construct_solver(mol, smult, wfnsym, nroots):
        solver = csf_solver(mol, smult=smult)
        solver.wfnsym = wfnsym
        solver.nroots = nroots
        solver.spin = smult - 1
        return solver

    if ms not in (None, 'lin'):
        raise ValueError("ms must be None or 'lin'")

    states = _validate_modelspace(states, mol=mol, ncas=mc.ncas,
                                  nelecas=mc.nelecas)

    solvers = [_construct_solver(mol, smult, wfnsym, nroots)
                          for (nroots, smult, wfnsym) in states]

    statetot = sum(state[0] for state in states)
    weights = _validate_state_weights(weights, statetot)

    if ms == 'lin':
        return mc.multi_state_mix(solvers, weights, "lin")
    else:
        return mcscf.state_average_mix_(mc, solvers, weights)


# Backward-compatible name retained for existing callers.
sacasscf_solver = state_average_solver

if __name__ == "__main__":
    from pyscf import scf, gto
    xyz ='''O  0.00000000   0.08111156   0.00000000
            H  0.78620605   0.66349738   0.00000000
            H -0.78620605   0.66349738   0.00000000'''

    mol = gto.M(atom=xyz, basis='cc-pvtz-dk', verbose=5)
    mf = scf.RHF(mol).sfx2c1e().run()
    dm = mf.make_rdm1()

    # AMFI Integrals
    hso = socintegrals(mol, ham='DKH')

    # MMFI Integrals
    # hso_mmfi = socintegrals(mol, amf=False, mmf=True, ham='DKH', dm=dm)
