#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
#
# Author: Yi Deng <yideng@uchicago.edu>
#

"""Generalized active space self-consistent field.

GASSCF reuses ``newton_casscf`` joint orbital/CI Newton driver,
with GASCI kernels providing the active-space CI, RDM and spin operations.
GAS definitions use ``gas_orbs``, ``gas_restr`` and ``gas_restr_type`` and
follow the same normalization rules as :mod:`pyscf.mcscf.gasci`.

Supported features include state averaging, GAS-safe canonicalization and
pseudo natural orbitals, density fitting, energy scanners and the GAS spin
penalty. Computational orbitals and CI vectors must be real-valued.
Unrestricted/spinor references, spatial symmetry, X2C, solvent models,
analytic gradients/NACs, GPU conversion and the legacy CASSCF drivers are
not supported.
"""

from collections import OrderedDict
import hashlib
import json
import sys
from types import FunctionType

import numpy

from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf.df.df_jk import _DFHF
from pyscf.lib import logger
from pyscf.fci import addons as fci_addons
from pyscf.mcscf import addons
from pyscf.mcscf import addons_gas
from pyscf.mcscf import df as mcdf
from pyscf.mcscf import fci_gas
from pyscf.mcscf import gasci
from pyscf.mcscf import mc1step
from pyscf.mcscf import newton_casscf

__all__ = ["GASSCF", "DFGASSCF"]


def _unsupported(feature):
    raise NotImplementedError(feature + " is not implemented for GASSCF")


def _check_symmetry(obj, mo_coeff=None, mol=None):
    """Reject spatial symmetry before orbital metadata can be discarded."""
    if mol is not None and mol.symmetry:
        _unsupported("molecular point-group symmetry")
    for source in (obj, getattr(obj, "_scf", None),
                   getattr(obj, "fcisolver", None)):
        if source is None:
            continue
        if getattr(getattr(source, "mol", None), "symmetry", False):
            _unsupported("molecular point-group symmetry")
        for name in ("extrasym", "orbsym", "wfnsym"):
            if getattr(source, name, None) is not None:
                _unsupported(name + " symmetry constraints")
        if getattr(getattr(source, "mo_coeff", None), "orbsym", None) is not None:
            _unsupported("orbital symmetry labels (orbsym)")
    if getattr(mo_coeff, "orbsym", None) is not None:
        _unsupported("orbital symmetry labels (orbsym)")


def _check_scf_reference(mf):
    """Reject unsupported references before conversion or integral work."""
    _check_symmetry(mf)
    if getattr(mf, "with_x2c", None) is not None:
        _unsupported("X2C")
    if getattr(mf, "with_solvent", None) is not None:
        _unsupported("solvent models")
    if isinstance(mf, (scf.ghf.GHF, scf.dhf.DHF)):
        _unsupported("generalized or relativistic spinor SCF references")


def _nuc_grad_method(self, state=None):
    """Reject nuclear gradients, including native state-average dispatch."""
    _unsupported("analytic nuclear gradient evaluation")


def _is_native_casscf_warning(message):
    return "SO-CASSCF" in message and "experimental feature" in message


class _GASSCFLogFilter:
    """Write-through stream filter for native Newton/CASSCF messages.

    GASSCF deliberately reuses ``newton_casscf`` driver.
    The numerical driver still contains CASSCF/CASCI text labels and one
    CASSCF-specific experimental-feature warning.  This filter changes only
    user-visible text while leaving the driver and all numerical data untouched.
    """

    _REPLACEMENTS = (
        ("Start SO-CASSCF (newton CASSCF)",
         "Start SO-GASSCF (newton GASSCF)"),
        ("newton CASSCF", "newton GASSCF"),
        ("Second order CASSCF", "Second order GASSCF"),
        ("SO-CASSCF", "SO-GASSCF"),
        ("CASSCF", "GASSCF"),
        ("CASCI", "GASCI"),
        ("CAS (", "GAS ("),
        ("CAS space", "GAS space"),
        ("CAS-space", "GAS-space"),
        ("E(CI)", "E(GASCI)"),
    )

    def __init__(self, stream):
        self._stream = sys.stdout if stream is None else stream
        self._pending = ""

    def _rewrite_line(self, line):
        if _is_native_casscf_warning(line):
            return ""
        for old, new in self._REPLACEMENTS:
            line = line.replace(old, new)
        return line

    def write(self, text):
        original_length = len(text)
        text = self._pending + text
        if not text:
            return original_length
        if text.endswith("\n"):
            self._pending = ""
            lines = text.splitlines(True)
        else:
            lines = text.splitlines(True)
            if lines and not lines[-1].endswith("\n"):
                self._pending = lines.pop()
            else:
                self._pending = ""
        rewritten = "".join(self._rewrite_line(line) for line in lines)
        if rewritten:
            self._stream.write(rewritten)
        return original_length

    def flush(self):
        if self._pending:
            rewritten = self._rewrite_line(self._pending)
            self._pending = ""
            if rewritten:
                self._stream.write(rewritten)
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _GASSCFLogger(logger.Logger):
    """Logger for native Newton without global stream mutation."""

    def warn(self, msg, *args):
        rendered = msg % args if args else msg
        if _is_native_casscf_warning(rendered):
            return
        return super().warn(msg, *args)


def _gasscf_newton_kernel(casscf, *args, **kwargs):
    """Run native Newton with calculation-local GAS logging."""

    verbose = kwargs.get("verbose", logger.NOTE)
    if not isinstance(verbose, logger.Logger):
        verbose = _GASSCFLogger(casscf.stdout, verbose)
    kwargs["verbose"] = verbose
    # Native Newton consumes objectives throughout its complete trajectory.
    # Convert only the returned tuple before mc1step assigns/logs public slots.
    native_kernel = newton_casscf.kernel
    if hasattr(casscf.fcisolver, 'ss_penalty'):
        # ``kernel -> update_orb_ci -> gen_g_hop`` resolves module globals,
        # not ``casscf.gen_g_hop``. Rebind the original function in a private
        # namespace and replace only the response entry point, leaving the
        # imported CASSCF module untouched.

        namespace = dict(vars(newton_casscf))
        namespace['gen_g_hop'] = gen_g_hop
        for name in ('update_orb_ci', 'kernel'):
            original = getattr(newton_casscf, name)
            rebound = FunctionType(original.__code__, namespace, original.__name__,
                                   original.__defaults__, original.__closure__)
            rebound.__kwdefaults__ = original.__kwdefaults__
            namespace[name] = rebound
        native_kernel = namespace['kernel']
    result = native_kernel(casscf, *args, **kwargs)
    physical, gas_physical = gasci._publish_energy_results(
        casscf, result[1], result[2])
    converged = result[0]
    if casscf.ncas == result[4].shape[1] and not casscf.internal_rotation:
        # Native Newton returns True unconditionally in its full-active CAS
        # shortcut. The GASCI bridge has the actual CI convergence status.
        converged = casscf.converged
    return (converged, physical, gas_physical) + result[3:]


def gen_g_hop(mc, mo, ci0, eris, verbose=None):
    """Add spin-penalty contributions to the Newton CI derivatives.

    The penalty has no orbital dependence, so ``contract_2e`` remains
    physical. For normalized ``c``, with ``p = <c|P|c>`` and
    ``r = Pc - pc``, the CI gradient and Hessian corrections are
    ``2 w r`` and
    ``2 w [(P-p)v - r(c.v) - c(r.v)]``, respectively.
    """

    _check_symmetry(mc, mo)
    gradient, update, hop, hdiag = newton_casscf.gen_g_hop(
        mc, mo, ci0, eris, verbose)
    solver = mc.fcisolver
    parameters = fci_gas._spin_penalty_parameters(
        solver, mc.ncas, mc._effective_nelecas())
    if parameters is None or parameters[0] == 0:
        return gradient, update, hop, hdiag

    # The spin plan owns only Python/NumPy data and remains valid after the
    # solver drops its cache. No borrowed C pointers escape in these closures.
    if solver.cache_plans:
        plan = solver._get_spin_plan(mc.ncas, mc.nelecas)
    else:
        plan = solver.make_spin_plan(mc.ncas, mc.nelecas)

    def penalty(vector):
        return fci_gas._spin_penalty_action(plan.contract, vector, parameters)

    roots = [ci0] if solver.nroots == 1 else ci0
    roots = [numpy.asarray(c).ravel() for c in roots]
    weights = tuple(getattr(mc, 'weights', (1.,)))
    ngorb = gradient.size - sum(c.size for c in roots)
    diagonal = fci_gas._spin_penalty_diagonal(
        plan.diagonal_vector(), parameters)
    gradient = gradient.copy()
    hdiag = hdiag.copy()
    terms = []
    start = ngorb
    for c, weight in zip(roots, weights):
        block = slice(start, start + c.size)
        start += c.size
        if weight == 0:
            continue
        pc = penalty(c)
        energy = c.dot(pc)
        residual = pc - energy * c
        gradient[block] += 2 * weight * residual
        hdiag[block] += 2 * weight * (diagonal - energy - 2 * residual * c)
        terms.append((block, c, weight, energy, residual))

    def penalized_hop(vector):
        result = hop(vector)
        for block, c, weight, energy, residual in terms:
            v = vector[block]
            result[block] += 2 * weight * (
                penalty(v) - energy * v - residual * c.dot(v)
                - c * residual.dot(v))
        return result

    def penalized_update(u, ci):
        result = update(u, ci)
        current = [ci] if solver.nroots == 1 else ci
        start = ngorb
        for c, weight in zip(current, weights):
            c = numpy.asarray(c).ravel()
            block = slice(start, start + c.size)
            start += c.size
            if weight == 0:
                continue
            # Native g_update normalizes each current CI vector as well.
            c = c / numpy.linalg.norm(c)
            pc = penalty(c)
            result[block] += 2 * weight * (pc - c.dot(pc) * c)
        return result

    return gradient, penalized_update, penalized_hop, hdiag


class _GASFCISolver(fci_gas.FCISolver):
    """GASCI solver shell with GASSCF-owned helper plans.

    Ordinary GASCI remains implemented by :mod:`fci_gas`.  This subclass owns
    only reusable helper plans needed by the joint GASSCF orbital optimizer.
    Adapted or copied solvers always start with empty caches, so C workspaces
    are not shared by adaptation or copy. An explicit ``plan=`` argument is
    separately validated and borrowed for that call, without taking ownership.
    """

    _keys = set(fci_gas.FCISolver._keys) | {"cache_plans", "ss_penalty", "ss_value"}
    _MAX_CONTRACT_PLANS = 3

    # PySCF native Newton probes these optional FCI hooks with ``getattr``.
    # GASCI deliberately does not provide CAS linkstr arrays, and arbitrary
    # active-space CI transformations do not preserve a restricted GAS space.
    # Hide the inherited GASCI diagnostics here so native CASSCF falls back to
    # link_index=None and does not call CAS-only helper APIs.
    gen_linkstr = None
    transform_ci_for_orbital_rotation = None

    def __init__(self, mol=None, gas_orbs=None, gas_restr=None,
                 gas_restr_type=addons_gas.GAS_RESTR_SPIN_SUPERGROUP,
                 lib=None, *, cache_plans=True):
        super().__init__(
            mol, gas_orbs=gas_orbs, gas_restr=gas_restr,
            gas_restr_type=gas_restr_type, lib=lib)
        self.cache_plans = bool(cache_plans)
        self._init_plan_cache()

    def _init_plan_cache(self):
        """Detach Newton-owned helper plans without closing borrowed objects.

        This method is used immediately after shallow adaptation or copy.  It
        must not close anything because the copied attributes may still belong
        to the source solver.  Live resources are released by :meth:`close`.
        """

        self._topology_key = None
        self._contract_space = None
        self._contract_plans = OrderedDict()
        self._rdm_plan = None
        self._spin_plan = None

    def _space_key(self, norb, nelec):
        """Return a normalized topology key for GAS helper-plan reuse."""

        gas_orbs, nelec, blocks = self._space_spec(norb, nelec)
        return (
            tuple(int(value) for value in gas_orbs),
            tuple(int(value) for value in nelec),
            tuple(tuple(int(item) for item in row) for row in blocks),
            id(self.lib),
        )

    def _ensure_topology(self, norb, nelec):
        """Drop cached plans when the normalized GAS topology changes."""

        key = self._space_key(norb, nelec)
        if getattr(self, "_topology_key", None) != key:
            self.close()
            self._topology_key = key
        return key

    @staticmethod
    def _contract_eri_key(eri):
        array = fci_gas._as_c_double(eri)
        digest = hashlib.sha256(array.view(numpy.uint8)).digest()
        return array, (tuple(int(value) for value in array.shape), digest)

    def _get_contract_plan(self, eri, norb, nelec):
        """Return a Newton-owned Hamiltonian contraction plan for one ERI."""

        eri, key = self._contract_eri_key(eri)
        self._ensure_topology(norb, nelec)
        if self._contract_space is None:
            self._contract_space = self.make_space(
                norb, nelec, compress_links=True)
        plan = self._contract_plans.pop(key, None)
        if plan is None:
            if len(self._contract_plans) >= self._MAX_CONTRACT_PLANS:
                _, evicted = self._contract_plans.popitem(last=False)
                evicted.close()
            plan = fci_gas.GasContractPlan(self._contract_space, eri.copy())
            plan.eri.flags.writeable = False
            plan.gos.flags.writeable = False
        self._contract_plans[key] = plan
        return plan

    def _get_rdm_plan(self, norb, nelec):
        """Return a Newton-owned raw-link GAS RDM plan."""

        self._ensure_topology(norb, nelec)
        if self._rdm_plan is None:
            self._rdm_plan = self.make_rdm_plan(norb, nelec)
        return self._rdm_plan

    def _rdm_plan_context(self, norb, nelec, plan=None):
        """Supply an owned cache entry or let GASCI manage a temporary plan."""

        if plan is None and self.cache_plans:
            plan = self._get_rdm_plan(norb, nelec)
        return super()._rdm_plan_context(norb, nelec, plan)

    def _get_spin_plan(self, norb, nelec):
        """Return a Newton-owned independent ``S^2`` contraction plan."""

        self._ensure_topology(norb, nelec)
        if self._spin_plan is None:
            self._spin_plan = self.make_spin_plan(norb, nelec)
        return self._spin_plan

    def spin_square(self, ci, norb, nelec, *args, **kwargs):
        """Return ``(<S^2>, 2S+1)`` without re-entering SA RDM wrappers.

        State-average dispatch can route ``spin_square`` back through the
        decorated RDM methods, causing a single GAS CI vector to be treated
        as multiple roots. Call the GASCI base RDM implementation directly;
        the plan-context hook supplies the required RDM plan.
        """

        ci = self._as_state_specific_ci(ci)
        nelec = fci_addons._unpack_nelec(nelec, self.spin)
        rdm1s, rdm2s = super().make_rdm12s(ci, norb, nelec)
        return fci_gas.spin_square_from_rdm12s(rdm1s, rdm2s, nelec)


    def close(self):
        """Release Newton-owned helper plans; repeated calls are safe."""

        contract_plans = getattr(self, "_contract_plans", None)
        if contract_plans is not None:
            for plan in list(contract_plans.values()):
                plan.close()
            contract_plans.clear()
        base = getattr(self, "base", None)
        if base is not None and base is not self and hasattr(base, "close"):
            base.close()
        contract_space = getattr(self, "_contract_space", None)
        if contract_space is not None:
            contract_space.close()
            self._contract_space = None
        rdm_plan = getattr(self, "_rdm_plan", None)
        if rdm_plan is not None:
            rdm_plan.close()
            self._rdm_plan = None
        self._spin_plan = None
        self._topology_key = None

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    *args, **kwargs):
        """Contract an absorbed Hamiltonian, reusing a Newton-owned plan."""

        plan = kwargs.pop("plan", None)
        if plan is not None:
            return super().contract_2e(
                eri, fcivec, norb, nelec, link_index,
                *args, plan=plan, **kwargs)
        compress_links = bool(kwargs.pop("compress_links", True))
        if args or kwargs or not self.cache_plans or not compress_links:
            return super().contract_2e(
                eri, fcivec, norb, nelec, link_index,
                *args, compress_links=compress_links, **kwargs)
        plan = self._get_contract_plan(eri, norb, nelec)
        return super().contract_2e(
            eri, fcivec, norb, nelec, link_index, plan=plan)

    @staticmethod
    def _as_state_specific_ci(ci):
        """Unwrap native Newton's singleton CI-list convention.

        Native Newton may represent a state-specific CI vector as ``[ci]``.
        GASCI methods expect a single flattened GAS CI vector, so this bridge
        accepts only the singleton form. Multiroot handling remains at the
        outer GASSCF level.
        """

        if isinstance(ci, (list, tuple)):
            if len(ci) != 1:
                _unsupported("multiroot CI density dispatch")
            return ci[0]
        return ci

    def make_rdm1s(self, ci, norb, nelec, link_index=None, *, plan=None):
        ci = self._as_state_specific_ci(ci)
        return super().make_rdm1s(
            ci, norb, nelec, link_index, plan=plan)

    def make_rdm1(self, ci, norb, nelec, link_index=None, *, plan=None):
        ci = self._as_state_specific_ci(ci)
        return super().make_rdm1(
            ci, norb, nelec, link_index, plan=plan)

    def make_rdm12s(self, ci, norb, nelec, link_index=None,
                    reorder=True, *, plan=None):
        ci = self._as_state_specific_ci(ci)
        return super().make_rdm12s(
            ci, norb, nelec, link_index, reorder, plan=plan)

    def make_rdm12(self, ci, norb, nelec, link_index=None,
                   reorder=True, *, plan=None):
        ci = self._as_state_specific_ci(ci)
        return super().make_rdm12(
            ci, norb, nelec, link_index, reorder, plan=plan)

    make_rdm2 = fci_gas.FCISolver.make_rdm2

    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None,
                    *, plan=None):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        return super().trans_rdm1s(
            cibra, ciket, norb, nelec, link_index, plan=plan)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None,
                   *, plan=None):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        return super().trans_rdm1(
            cibra, ciket, norb, nelec, link_index, plan=plan)

    def trans_rdm12s(self, cibra, ciket, norb, nelec, link_index=None,
                     reorder=True, *, plan=None):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        return super().trans_rdm12s(
            cibra, ciket, norb, nelec, link_index, reorder, plan=plan)

    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True, *, plan=None):
        cibra = self._as_state_specific_ci(cibra)
        ciket = self._as_state_specific_ci(ciket)
        return super().trans_rdm12(
            cibra, ciket, norb, nelec, link_index, reorder, plan=plan)

    def contract_ss(self, fcivec, norb, nelec):
        """Contract ``S^2`` with a GAS CI vector, reusing a spin plan."""

        fcivec = self._as_state_specific_ci(fcivec)
        if not self.cache_plans:
            return super().contract_ss(fcivec, norb, nelec)
        return self._get_spin_plan(norb, nelec).contract(fcivec)

    def copy(self):
        result = super().copy()
        result._init_plan_cache()
        return result


def _validated_user_gas_orbs(gas_orbs):
    """Validate GAS orbital counts with the GASCI integer contract."""

    return addons_gas._integer_vector(gas_orbs, "gas_orbs")


def _new_gas_solver(mf, gas_orbs, gas_restr, gas_restr_type, cache_plans):
    if gas_orbs is None:
        raise ValueError(
            "gas_orbs is required without an explicit GASCI solver")
    gas_orbs = _validated_user_gas_orbs(gas_orbs)
    if gas_restr_type is None:
        gas_restr_type = addons_gas.GAS_RESTR_SPIN_SUPERGROUP
    if cache_plans is None:
        cache_plans = True
    return _GASFCISolver(
        getattr(mf, "mol", None), gas_orbs=gas_orbs,
        gas_restr=gas_restr, gas_restr_type=gas_restr_type,
        cache_plans=cache_plans)

def _decorated_solver_classes():
    """Return native wrapper classes that should decorate GASSCF, not solver."""

    return tuple(cls for cls in (
        getattr(addons, "StateAverageFCISolver", None),
        getattr(addons, "StateAverageMixFCISolver", None),
        getattr(addons, "StateSpecificFCISolver", None),
        getattr(fci_addons, "SpinPenaltyFCISolver", None),
    ) if cls is not None)


def _adapt_solver(fcisolver, cache_plans):
    """Return a Newton-GASSCF-owned shell for an explicit GASCI solver.

    Scientific GASCI settings are copied from the input solver.  Newton-owned
    helper plans are intentionally not borrowed.  Predecorated solvers are
    rejected. Configure ordinary state averaging and spin penalty through the
    GASSCF object methods; state-specific excited-state wrappers are unsupported.
    """

    if isinstance(fcisolver, _decorated_solver_classes()):
        _unsupported("predecorated solver input; decorate the GASSCF object")
    if isinstance(fcisolver, _GASFCISolver):
        solver = fcisolver.copy()
    elif type(fcisolver) is fci_gas.FCISolver:
        solver = lib.view(fcisolver, _GASFCISolver)
        solver._init_plan_cache()
    else:
        _unsupported("external/non-GASCI solver adaptation")

    if solver.gas_orbs is None:
        raise ValueError("explicit GASCI solver must define gas_orbs")
    solver.gas_orbs = _validated_user_gas_orbs(solver.gas_orbs)
    if solver.gas_restr_type is None:
        solver.gas_restr_type = addons_gas.GAS_RESTR_SPIN_SUPERGROUP
    if cache_plans is None:
        solver.cache_plans = bool(getattr(fcisolver, "cache_plans", True))
    else:
        solver.cache_plans = bool(cache_plans)
    return solver


def _digest(value):
    """Return a stable digest for small JSON-like scanner invariants."""

    payload = json.dumps(
        value, sort_keys=True, default=lambda item: numpy.asarray(item).tolist(),
        allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _system_signature(mol):
    """Return molecular invariants that an energy scanner must preserve."""

    return (
        tuple(mol.atom_symbol(i) for i in range(mol.natm)),
        int(mol.charge), int(mol.spin), bool(mol.cart),
        _digest(mol._basis), _digest(mol._ecp), int(mol.nao_nr()),
    )


def _problem_signature(mc):
    """Return GAS/Newton invariants that must remain fixed during scans."""

    gas_orbs, nelec, blocks = mc.fcisolver._space_spec(mc.ncas, mc.nelecas)
    limits = addons_gas.check_kernel_limits(gas_orbs, nelec, blocks)
    return {
        "ncore": int(mc.ncore),
        "ncas": int(mc.ncas),
        "nelec": tuple(int(value) for value in nelec),
        "gas_orbs": tuple(int(value) for value in gas_orbs),
        "spin_supergroups": _digest(numpy.asarray(blocks).tolist()),
        "ndet": int(limits["ndet_estimate"]),
        "nroots": int(getattr(mc.fcisolver, "nroots", 1)),
        "weights": tuple(float(value) for value in getattr(mc, "weights", (1.0,))),
        "spin_penalty": (
            None if not hasattr(mc.fcisolver, "ss_penalty") else
            (float(mc.fcisolver.ss_penalty),
             None if getattr(mc.fcisolver, "ss_value", None) is None else
             float(mc.fcisolver.ss_value))),
        "frozen": _digest(mc.frozen),
    }


def _copy_ci(ci, signature):
    """Copy and validate a scanner CI guess against the GAS model."""

    if ci is None:
        return None
    nroots = int(signature["nroots"])
    ndet = int(signature["ndet"])
    if nroots == 1:
        values = list(ci) if isinstance(ci, (list, tuple)) else [ci]
    elif isinstance(ci, (list, tuple)):
        values = list(ci)
    else:
        array = numpy.asarray(ci)
        if array.shape == (nroots, ndet):
            values = list(array)
        elif array.shape == (ndet, nroots):
            values = list(array.T)
        else:
            raise ValueError("scanner CI must contain one vector per GAS root")
    if len(values) != nroots:
        raise ValueError("scanner CI root count does not match the GAS model")

    roots = []
    for value in values:
        array = numpy.asarray(value)
        if numpy.iscomplexobj(array):
            raise NotImplementedError("complex scanner CI coefficients")
        array = numpy.asarray(array, dtype=float).reshape(-1)
        norm = numpy.linalg.norm(array)
        if array.size != ndet or not numpy.all(numpy.isfinite(array)) or norm < 1e-14:
            raise ValueError(
                "scanner CI has invalid GAS length, norm or coefficients")
        roots.append(numpy.array(array, copy=True))
    return roots[0] if nroots == 1 else roots


def _orbital_groups(mc):
    """Return projection priority groups that preserve GAS subspace ordering."""

    groups = []
    start = int(mc.ncore)
    for size in mc.fcisolver._space_spec(mc.ncas, mc.nelecas)[0]:
        stop = start + int(size)
        groups.append(list(range(start, stop)))
        start = stop
    if mc.ncore:
        groups.append(list(range(int(mc.ncore))))
    return groups


def _copy_df(with_df):
    """Copy DF settings with fresh geometry caches and output-file ownership."""

    result = with_df.copy()
    # DF.reset clears integral/JK caches, but retains the output target. A
    # copied object must not overwrite the source's named or temporary file.
    result._cderi_to_save = None
    return result.reset(with_df.mol)


def _as_scanner(mc):
    """Return an energy-only scanner with fixed GAS/Newton objective metadata."""

    mc.validate_capabilities()
    if isinstance(mc, lib.SinglePointScanner):
        return mc
    source = mc.copy()
    source.mo_coeff = None if mc.mo_coeff is None else numpy.array(
        mc.mo_coeff, copy=True)
    source.ci = _copy_ci(source._resolve_ci_guess(), _problem_signature(source))
    return lib.set_class(
        _GASSCFScanner(source), (_GASSCFScanner, source.__class__),
        source.__class__.__name__ + "Scanner")


class _GASSCFScanner(lib.SinglePointScanner):
    """Energy scanner for a fixed GASSCF objective and atom/basis model."""

    _keys = {"scan_info", "_scan_problem", "_scan_system"}

    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)
        self._scf = mc._scf.as_scanner()
        self._scan_problem = _problem_signature(mc)
        self._scan_system = _system_signature(mc.mol)
        self.scan_info = None

    def reset(self, mol=None):
        """Invalidate scan results while retaining native orbital/CI guesses."""
        super().reset(mol)
        self.scan_info = None
        self.converged = False
        return self

    def __call__(self, mol_or_geom, mo_coeff=None, ci0=None):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        _check_symmetry(self, mo_coeff, mol)
        if _system_signature(mol) != self._scan_system:
            raise ValueError(
                "energy scanner requires the same atoms/order, charge, spin, "
                "basis/ECP and AO size")

        self.validate_capabilities()
        signature = _problem_signature(self)
        if signature != self._scan_problem:
            raise ValueError(
                "GAS definition, roots, weights or orbital constraints "
                "changed; create a new scanner")

        old_mol = self.mol
        previous_mo = None if self.mo_coeff is None else numpy.array(
            self.mo_coeff, copy=True)
        guess_ci = _copy_ci(self._resolve_ci_guess(ci0), signature)

        self.reset(mol)
        self._scf(mol)
        self.mol = mol
        self.validate_capabilities()

        if mo_coeff is not None:
            guess_mo = numpy.asarray(mo_coeff)
            mo_source = "explicit MO in current AO basis"
        elif previous_mo is None:
            guess_mo = numpy.asarray(self._scf.mo_coeff)
            mo_source = "SCF MO guess"
        else:
            guess_mo = addons.project_init_guess(
                self, previous_mo, prev_mol=old_mol,
                priority=_orbital_groups(self), use_hf_core=False)
            mo_source = "projected previous GASSCF MO by GAS blocks"

        # Share GASCI's validation and configurable warning/error thresholds.
        error = self._check_mo_orthonormality(guess_mo)

        self.scan_info = {
            "MO_source": mo_source,
            "CI_source": "explicit" if ci0 is not None else (
                "previous GAS CI guess" if guess_ci is not None else
                "native initial guess"),
            "initial_MO_metric_error": error,
            "problem": signature,
            "projection_groups": _orbital_groups(self),
            "native_converged": None,
            "scope": "Energy only; root order follows macro GASCI.",
        }
        energy = self.kernel(numpy.array(guess_mo, copy=True), guess_ci)[0]
        self.scan_info.update(
            native_converged=bool(self.converged),
            energy=float(energy),
            e_states=numpy.atleast_1d(getattr(self, "e_states", energy)).tolist(),
        )
        return energy


class _StateAverageGASSCF(addons.StateAverageMCSCF):
    """State-average marker with GAS-specific undo/cache cleanup."""

    @property
    def e_states(self):
        return self._physical_e_states()

    @property
    def e_average(self):
        if self._gas_energy_results is None:
            return None
        return float(numpy.dot(self.weights, self.e_states))

    _finalize = gasci.GASCI._finalize

    def undo_state_average(self):
        """Return a single-state copy with independent GAS/SCF/DF resources."""

        source = self.copy()
        result = addons.StateAverageMCSCF.undo_state_average(source)
        gasci._clear_energy_results(result)
        result.fcisolver.nroots = 1
        return result


class _DFGASSCF(mcdf._DFCASSCF):
    """Density-fitting mixin with GAS-specific user-visible labels."""

    __name_mixin__ = "DF"

    # The DF mixin precedes GASSCF in the MRO and supplies its own CASSCF
    # gradient constructors. Guard both the public and native SA hook names.
    nuc_grad_method = _nuc_grad_method
    Gradients = nuc_grad_method
    _state_average_nuc_grad_method = nuc_grad_method

    def reset(self, mol=None):
        # Native DF.reset changes its auxiliary molecule before calling super.
        _check_symmetry(self, mol=mol)
        return super().reset(mol)

    def dump_flags(self, verbose=None):
        super(mcdf._DFCAS, self).dump_flags(verbose)
        logger.info(
            self,
            "DFGASSCF: density fitting for JK matrix and 2e integral "
            "transformation")
        return self

    def undo_df(self):
        result = lib.view(self.copy(),
                          lib.drop_class(self.__class__, _DFGASSCF))
        try:
            del result.with_df
        except AttributeError:
            pass
        return result


class GASSCF(newton_casscf.CASSCF):
    """Joint GASSCF orbital optimizer for a determinant GASCI active space.

    Args:
        mf : SCF object or Mole
            Mean-field reference, or a molecule from which RHF is constructed.
            UHF/UKS inputs use their native to_rhf() conversion. Active DF-SCF
            references automatically select DF-GASSCF.
        ncas : int
            Total number of active orbitals, equal to ``sum(gas_orbs)``.
        nelecas : int or pair of ints
            Number of active electrons, optionally resolved as alpha/beta.
        gas_orbs : sequence of ints, optional
            Ordered numbers of active orbitals in the GAS subspaces.  The total
            must equal ``ncas``.  If omitted, one GAS contains all active
            orbitals, as in GASCI.
        gas_restr : object, optional
            GAS restriction in the syntax selected by ``gas_restr_type``.
        gas_restr_type : str, optional
            ``spin-supergroup``, ``supergroup``, ``cumulative-occ`` or ``ras``.
            If omitted, the GASCI default ``spin-supergroup`` is used.
        ncore : int, optional
            Number of inactive doubly occupied orbitals.
        frozen : int or sequence of ints, optional
            Frozen orbital specification forwarded to the native Newton class.
        fcisolver : fci_gas.FCISolver, optional
            GASCI solver whose settings are copied into an owned adapter with
            fresh helper-plan caches. It must define gas_orbs; do not also
            supply gas_orbs, gas_restr or gas_restr_type to the constructor.
            Predecorated SA/spin-penalty wrappers and external solvers are
            unsupported; use GASSCF's state_average() and fix_spin_() methods.
        cache_plans : bool, optional
            Whether GASSCF reuses owned GAS contraction/RDM/spin plans.
            None preserves an explicit solver's setting, or defaults to True.

    Notes:
        The public call is ``GASSCF(mf, ncas, nelecas, ...)``, following GASCI.
        For compatibility, calls supplying ``nelecas`` and ``gas_orbs`` (or
        an explicit GASCI solver) by keyword may omit ``ncas``; it is then
        inferred from the GAS orbital counts.

        The optimizer reuses joint orbital/CI Newton driver.
        GAS-specific CI, RDM, spin and orbital-rotation operations are supplied
        by the determinant GASCI adapter.

    Energies:
        kernel() and gasci() return (e_tot, e_gas, ci, mo_coeff, mo_energy).
        Public e_tot and e_gas (also e_cas) exclude spin penalties. On SA
        objects these are weighted energies; e_states contains physical root
        energies in solver order. After a penalized solve, spin_energy_report()
        provides physical, penalty and objective energies. Newton optimizes the
        penalized objective; its internal casci() bridge returns that objective.

    Natural-orbital analysis:
        ``get_gas_natorb(state=i)`` returns a full MO matrix and active-space
        occupations for root i. ``get_gas_average_natorb()`` uses the SA
        density. ``get_gas_pseudo_natorb()`` diagonalizes each GAS block and
        returns occupations as one array per subspace. These methods leave
        the computational orbitals and CI vectors unchanged.
        On SA objects, pseudo-natural orbitals default to the weighted density;
        state=i selects a root. True natural orbitals may mix GAS subspaces
        and are for analysis only. Complex analysis orbitals cannot be passed
        to the real-valued GAS solver.

        To rotate computational orbitals and CI together, explicitly call
        ``mc.canonicalize_(gas_pseudo_natorb=True)``. The variant without the
        trailing underscore returns the transformed results without writeback.

        Real analysis orbitals can be exported with Molden writer::

            from pyscf.tools import molden
            mo_no, active_occ = mc.get_gas_natorb(state=0)
            occupations = numpy.zeros(mo_no.shape[1])
            occupations[:mc.ncore] = 2
            occupations[mc.ncore:mc.ncore + mc.ncas] = active_occ
            # Zero energy placeholders; occupations are not orbital energies.
            molden.from_mo(mc.mol, "gas_natorb.molden", mo_no,
                           occ=occupations, ene=numpy.zeros(mo_no.shape[1]))

        For pseudo-natural orbitals, concatenate the occupation arrays in
        GAS order before filling the active slice. These block occupations
        alone do not represent off-diagonal density between GAS subspaces.
    """

    _keys = set(newton_casscf.CASSCF._keys) | {
        "gas_orbs", "gas_restr", "gas_restr_type", "cache_plans",
        "e_spin_penalty", "_gas_energy_results", "_gas_ci_signature",
        "spin_penalty_method"}

    def __init__(self, mf, ncas=None, nelecas=None, gas_orbs=None,
                 gas_restr=None, gas_restr_type=None, *, ncore=None, frozen=None,
                 fcisolver=None, cache_plans=None):
        self._initialize(mf, ncas, nelecas, gas_orbs, gas_restr, gas_restr_type,
                         ncore=ncore, frozen=frozen, fcisolver=fcisolver,
                         cache_plans=cache_plans)
        if isinstance(self._scf, _DFHF) and self._scf.with_df:
            fitted = self.density_fit()
            self.__dict__.update(fitted.__dict__)
            self.__class__ = fitted.__class__

    def _initialize(self, mf, ncas=None, nelecas=None, gas_orbs=None,
                    gas_restr=None, gas_restr_type=None, *, ncore=None,
                    frozen=None, fcisolver=None, cache_plans=None):
        """Initialize the common unwrapped object before selecting DF once."""
        if isinstance(mf, gto.MoleBase):
            mf = mf.RHF()
        _check_scf_reference(mf)
        if isinstance(mf, scf.uhf.UHF):
            mf = mf.to_rhf()
        _check_scf_reference(mf)
        if nelecas is None:
            raise TypeError("GASSCF requires nelecas")
        if ncas is not None:
            if (isinstance(ncas, (bool, numpy.bool_)) or
                    not isinstance(ncas, (int, numpy.integer))):
                raise TypeError("ncas must be an integer")
            ncas = int(ncas)
            if ncas <= 0:
                raise ValueError("ncas must be positive")

        if fcisolver is None:
            if gas_orbs is None and ncas is not None:
                gas_orbs = (ncas,)
            solver = _new_gas_solver(
                mf, gas_orbs, gas_restr, gas_restr_type, cache_plans)
        else:
            if (gas_orbs is not None or gas_restr is not None or
                    gas_restr_type is not None):
                raise ValueError(
                    "gas_orbs, gas_restr and gas_restr_type are supplied "
                    "by the explicit fcisolver")
            solver = _adapt_solver(fcisolver, cache_plans)

        gas_ncas = sum(solver.gas_orbs)
        if ncas is None:
            ncas = gas_ncas
        elif ncas != gas_ncas:
            raise ValueError(
                "ncas (%d) must equal sum(gas_orbs) (%d)" % (ncas, gas_ncas))

        super().__init__(mf, ncas, nelecas, ncore=ncore, frozen=frozen)
        self.fcisolver = solver
        self.fcisolver.mol = self.mol
        _check_symmetry(self)
        self.e_spin_penalty = None
        self._gas_energy_results = None
        self._gas_ci_signature = None
        self.spin_penalty_method = None


    def _push_gasscf_log_labels(self):
        """Install a calculation-local filter for native Newton labels."""

        stdout = getattr(self, "stdout", None)
        restore_stdout = not isinstance(stdout, _GASSCFLogFilter)
        if restore_stdout:
            self.stdout = _GASSCFLogFilter(stdout)
        return stdout, restore_stdout

    def dump_flags(self, verbose=None):
        """Print GASSCF flags using GAS terminology."""

        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        if self.mo_coeff is None:
            log.info("GAS (%de+%de, %do), ncore = %d",
                     self.nelecas[0], self.nelecas[1], ncas, ncore)
        else:
            nvir = self.mo_coeff.shape[1] - ncore - ncas
            log.info("GAS (%de+%de, %do), ncore = %d, nvir = %d",
                     self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        log.info("gas_orbs = %s", self.gas_orbs)
        log.info("gas_restr_type = %s", self.gas_restr_type)
        log.info("gas_restr = %s", self.gas_restr)
        log.info("cache GAS helper plans = %s",
                 getattr(self.fcisolver, "cache_plans", None))
        if self.frozen is not None:
            log.info("frozen orbitals %s", str(self.frozen))
        if hasattr(self.fcisolver, "ss_penalty"):
            target = getattr(self.fcisolver, "ss_value", None)
            log.info("spin penalty shift = %g", self.fcisolver.ss_penalty)
            log.info("target S^2 = %s", "minimum" if target is None else target)
        log.info("max_cycle_macro = %d", self.max_cycle_macro)
        log.info("max_cycle_micro = %d", self.max_cycle_micro)
        log.info("conv_tol = %g", self.conv_tol)
        log.info("conv_tol_grad = %s", self.conv_tol_grad)
        log.info("orbital rotation max_stepsize = %g", self.max_stepsize)
        log.info("augmented hessian ah_max_cycle = %d", self.ah_max_cycle)
        log.info("augmented hessian ah_conv_tol = %g", self.ah_conv_tol)
        log.info("augmented hessian ah_linear dependence = %g", self.ah_lindep)
        log.info("augmented hessian ah_level shift = %g", self.ah_level_shift)
        log.info("augmented hessian ah_start_tol = %g", self.ah_start_tol)
        log.info("augmented hessian ah_start_cycle = %d", self.ah_start_cycle)
        log.info("augmented hessian ah_grad_trust_region = %g",
                 self.ah_grad_trust_region)
        log.info("kf_trust_region = %g", self.kf_trust_region)
        log.info("kf_interval = %d", self.kf_interval)
        log.info("natorb = %s", self.natorb)
        log.info("canonicalization = %s", self.canonicalization)
        log.info("chkfile = %s", self.chkfile)
        log.info("max_memory %d MB (current use %d MB)",
                 self.max_memory, lib.current_memory()[0])
        log.info("internal_rotation = %s", self.internal_rotation)
        solver_stdout = getattr(self.fcisolver, "stdout", None)
        had_solver_stdout = hasattr(self.fcisolver, "stdout")
        try:
            # ``direct_spin1.FCISolver`` stores its own stdout at construction
            # time.  Route the GASCI solver flags to the current GASSCF stream
            # so redirected/captured output remains self-contained.
            self.fcisolver.stdout = self.stdout
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass
        finally:
            if had_solver_stdout:
                self.fcisolver.stdout = solver_stdout
            else:
                try:
                    del self.fcisolver.stdout
                except AttributeError:
                    pass
        if self.mo_coeff is None:
            log.warn("Orbital for GASSCF is not specified.  You probably need "
                     "call SCF.kernel() to initialize orbitals.")
        return self


    def validate_capabilities(self):
        """Validate the supported GASSCF feature set.

        This guard makes unsupported combinations fail before entering the native
        CASSCF driver and sets ``internal_rotation`` whenever
        active-active inter-GAS rotations are part of the orbital variables.
        SA/DF objects must carry the GAS-specific outer adapters as well as
        the matching native solver wrappers. Use the GASSCF object methods
        or DFGASSCF factory to construct them.
        """

        _check_symmetry(self)
        _check_scf_reference(self._scf)
        if getattr(self, "with_solvent", None) is not None:
            _unsupported("solvent models")
        if isinstance(self, mcdf._DFHessianCASSCF):
            _unsupported("density-fitted approximate Hessian")
        if isinstance(self, mcdf._DFCAS) and not isinstance(self, _DFGASSCF):
            _unsupported(
                "unadapted PySCF DF wrapper; use mc.density_fit() on the "
                "original GASSCF object")
        if (isinstance(self, addons.StateAverageMCSCF) and
                not isinstance(self, _StateAverageGASSCF)):
            _unsupported(
                "unadapted PySCF state-average wrapper; use "
                "mc.state_average(weights) on the original GASSCF object")
        if not isinstance(self.fcisolver, _GASFCISolver):
            _unsupported("non-adapted GASCI solver")
        if isinstance(self.fcisolver, addons.StateAverageMixFCISolver):
            _unsupported("state-average-mix GASCI solver")
        if isinstance(self.fcisolver, addons.StateSpecificFCISolver):
            _unsupported("state-specific GASCI solver wrapper")

        if isinstance(self.fcisolver, fci_addons.SpinPenaltyFCISolver):
            _unsupported(
                "PySCF SpinPenaltyFCISolver wrapper; use GASSCF.fix_spin_")

        if getattr(self, "natorb", False):
            _unsupported("GAS natural-orbital rotation")

        is_sa_mc = isinstance(self, addons.StateAverageMCSCF)
        is_sa_solver = isinstance(self.fcisolver, addons.StateAverageFCISolver)
        if is_sa_mc != is_sa_solver:
            raise ValueError(
                "state_average requires matching MCSCF and GASCI solver wrappers")
        if is_sa_mc:
            weights = self._validate_weights(self.weights)
            if int(getattr(self.fcisolver, "nroots", 1)) != len(weights):
                raise ValueError("nroots/weights mismatch")
        elif int(getattr(self.fcisolver, "nroots", 1)) != 1:
            raise ValueError(
                "nroots>1 requires ordinary state_average support")

        gas_orbs, gas_restr = self._normalized_restriction()
        if sum(gas_orbs) != self.ncas:
            raise ValueError("ncas must equal sum(gas_orbs)")
        nelecas = self._effective_nelecas()
        addons_gas.check_kernel_limits(gas_orbs, nelecas, gas_restr)
        if hasattr(self.fcisolver, "ss_penalty"):
            if not addons_gas.is_spin_complete(gas_orbs, nelecas, gas_restr):
                raise ValueError(
                    "fix_spin_ requires a spin-complete GAS restriction")
            fci_gas._spin_penalty_parameters(
                self.fcisolver, self.ncas, nelecas)
        self.internal_rotation = len(gas_orbs) > 1
        self.fcisolver.mol = self.mol
        return self

    def close(self):
        """Release GASSCF-owned GAS helper plans; repeated calls are safe."""

        self.fcisolver.close()
        return self

    def reset(self, mol=None):
        """Reset molecular data and drop GAS helper-plan caches."""

        _check_symmetry(self, mol=mol)
        self.close()
        gasci._clear_energy_results(self)
        result = super().reset(mol)
        self.fcisolver.mol = self.mol
        return result

    def update_from_chk(self, chkfile=None):
        """Load native checkpoint fields and invalidate derived energy reports.

        Physical e_tot/e_gas remain available from the file. Root energies,
        e_average and spin-penalty diagnostics require a new calculation,
        since the native checkpoint does not store the full GAS energy report.
        Its CI vector has no GAS-space signature: reuse it explicitly via ci0
        only when the caller knows the determinant basis is compatible.
        """

        result = super().update_from_chk(chkfile)
        self._gas_ci_signature = None
        energies = self.e_tot, self.e_cas
        gasci._clear_energy_results(self)
        self.e_tot, self.e_cas = energies
        return result

    update = update_from_chk

    def copy(self):
        """Copy GAS/SCF state with independent solver and DF cache ownership.

        DF settings are retained, but integrals are rebuilt on demand and DF
        output files are not inherited. SCF and MCSCF share the new DF object
        only when they shared the source DF object.
        MO and CI arrays retain shallow-copy semantics. as_scanner()
        additionally copies these arrays for independent scanning.
        """

        result = super().copy()
        result.fcisolver = self.fcisolver.copy()
        result.fcisolver.mol = result.mol
        result._scf = self._scf.copy()
        scf_df = getattr(self._scf, "with_df", None)
        if scf_df is not None:
            result._scf.with_df = _copy_df(scf_df)
        mc_df = getattr(self, "with_df", None)
        if mc_df is not None:
            result.with_df = (result._scf.with_df if mc_df is scf_df
                              else _copy_df(mc_df))
        return result

    def newton(self):
        """Return this GASSCF object after validation."""

        return self.validate_capabilities()

    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        """Return the independent orbital-rotation mask for GAS orbital optimization.

        The native CASSCF mask contains core-active, core-external and
        active-external rotations.  GASSCF adds active-active rotations between
        different GAS subspaces because such rotations change the constrained
        GAS wave function.  Rotations within one GAS subspace remain redundant
        orbital gauge degrees of freedom and are excluded.
        """

        _check_symmetry(self)
        nmo = int(nmo)
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        gas_orbs = _validated_user_gas_orbs(self.gas_orbs)
        if sum(gas_orbs) != ncas:
            raise ValueError("sum(gas_orbs) must equal ncas")

        mask = numpy.zeros((nmo, nmo), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True

        first_active = ncore
        offset = ncore
        for norb in gas_orbs:
            start = offset
            stop = start + norb
            mask[start:stop, first_active:start] = True
            offset = stop

        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:, :frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:, frozen] = False
        return mask

    def rotate_mo(self, mo, u, log=None):
        _check_symmetry(self, mo)
        return super().rotate_mo(mo, u, log)

    _effective_nelecas = gasci.GASCI._effective_nelecas

    def _normalized_restriction(self, return_info=False):
        """Return the normalized GAS definition used by GASCI kernels."""

        gas_orbs = _validated_user_gas_orbs(self.gas_orbs)
        return addons_gas.normalize_gas_spec(
            gas_orbs, self._effective_nelecas(),
            self.gas_restr, self.gas_restr_type,
            return_info=return_info)

    def gas_space_info(self):
        """Return normalized GAS metadata and compact C-space information."""

        gas_orbs, blocks, info = self._normalized_restriction(return_info=True)
        return gasci._gas_space_info(self, gas_orbs, blocks, info)

    # Public density/property conventions are shared with GASCI: state=None
    # selects the weighted density on SA objects, while state=i selects root i.
    # Keep the Newton solver adapter and its singleton-CI/plan dispatch intact.
    spin_energy_report = gasci.spin_energy_report
    _physical_e_states = gasci._physical_e_states
    _clear_energy_results = gasci._clear_energy_results

    _has_state_weights = gasci.GASCI._has_state_weights
    _state_weights = gasci.GASCI._state_weights
    _base_fcisolver_method = gasci.GASCI._base_fcisolver_method
    _ci_for_rdm = gasci.GASCI._ci_for_rdm
    _spin_square_for_ci = gasci.GASCI._spin_square_for_ci

    def _select_ci(self, ci=None, state=0):
        ci = self.ci if ci is None else ci
        if ci is None:
            raise ValueError("CI vector is not available")
        return gasci.GASCI._select_ci(self, ci, state)

    def _spin_square_for_roots(self, roots, ncas, nelecas):
        # Bypass the SA averaging wrapper for each selected root while keeping
        # the Newton-owned RDM plan reuse in _GASFCISolver.spin_square.
        method = self._base_fcisolver_method("spin_square")
        return [method(ci, ncas, nelecas) for ci in roots]

    make_gasdm1 = gasci.GASCI.make_gasdm1
    make_gasdm1s = gasci.GASCI.make_gasdm1s
    make_gasdm12 = gasci.GASCI.make_gasdm12
    make_gasdm12s = gasci.GASCI.make_gasdm12s
    make_gasdm2 = gasci.GASCI.make_gasdm2
    trans_gasdm1 = gasci.GASCI.trans_gasdm1
    trans_gasdm1s = gasci.GASCI.trans_gasdm1s
    trans_gasdm12 = gasci.GASCI.trans_gasdm12
    trans_gasdm12s = gasci.GASCI.trans_gasdm12s
    trans_gasdm2 = gasci.GASCI.trans_gasdm2
    make_rdm1 = gasci.GASCI.make_rdm1
    make_rdm1s = gasci.GASCI.make_rdm1s
    spin_square = gasci.GASCI.spin_square

    # Reuse GASCI analysis without installing analysis orbitals on this object.
    _natural_eigensystem = staticmethod(gasci.GASCI._natural_eigensystem)
    sort_mo = gasci.GASCI.sort_mo
    get_gas_natorb = gasci.GASCI.get_gas_natorb
    get_gas_average_natorb = gasci.GASCI.get_gas_average_natorb
    get_gas_pseudo_natorb = gasci.GASCI.get_gas_pseudo_natorb
    get_gas_pseudo_natorb_occupations = gasci.GASCI.get_gas_pseudo_natorb_occupations
    _gas_analysis_label = "GASSCF"
    analyze = gasci.GASCI.analyze
    to_gpu = gasci.GASCI.to_gpu

    def _check_mo_orthonormality(self, mo_coeff=None, verbose=None):
        _check_symmetry(self, mo_coeff)
        return gasci.GASCI._check_mo_orthonormality(self, mo_coeff, verbose)

    def _rotate_gas_orbitals(self, mo_coeff, rotation):
        _check_symmetry(self, mo_coeff)
        return gasci.GASCI._rotate_gas_orbitals(self, mo_coeff, rotation)

    def _gasdm1s_to_ao(self, gasdm1s, mo_coeff, ncas, ncore):
        _check_symmetry(self, mo_coeff)
        return gasci.GASCI._gasdm1s_to_ao(self, gasdm1s, mo_coeff, ncas, ncore)

    def get_fock(self, mo_coeff=None, ci=None, eris=None, gasdm1=None,
                 verbose=None, *, casdm1=None):
        """Build the AO generalized Fock matrix with GASCI's density API.

        ``gasdm1`` can select a root-specific density on an SA object.
        ``casdm1`` is a compatibility alias for native PySCF callers; supply
        only one of the two. Positional arguments retain the native order.
        """

        _check_symmetry(self, mo_coeff)
        if casdm1 is not None:
            if gasdm1 is not None:
                raise ValueError("supply only one of gasdm1 and casdm1")
            gasdm1 = casdm1
        return gasci.GASCI.get_fock(
            self, mo_coeff, ci, eris, gasdm1, verbose)

    get_h1gas = gasci.GASCI.get_h1gas

    get_h2gas = gasci.GASCI.get_h2gas

    def get_grad(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        """Return the packed orbital gradient in mc1step convention.

        Twice this vector is the orbital block of the joint Newton gradient.
        As in native CASSCF, omitted densities trigger a fixed-orbital CI solve;
        supplied densities are used directly, including on state-average objects.
        Explicit or stored orbitals must pass the shared GASCI MO validation,
        even when densities and ERIs are supplied. Validation precedes AO2MO.
        """

        self.validate_capabilities()
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        self._check_mo_orthonormality(mo_coeff)
        if eris is None:
            eris = self.ao2mo(mo_coeff)
        if casdm1_casdm2 is None:
            _, _, ci = self.casci(mo_coeff, eris=eris)
            casdm1_casdm2 = self.fcisolver.make_rdm12(ci, self.ncas, self.nelecas)
        dm1, dm2 = casdm1_casdm2
        # The inherited get_grad dispatches through self.gen_g_hop, whose
        # joint orbital/CI signature differs from this orbital-only interface.
        return mc1step.gen_g_hop(self, mo_coeff, 1, dm1, dm2, eris)[0]

    _gas_problem_signature = gasci.GASCI._gas_problem_signature
    _ci_matches_signature = gasci.GASCI._ci_matches_signature
    _clear_ci_guess = gasci.GASCI._clear_ci_guess

    def _resolve_ci_guess(self, ci0=None):
        """Use an explicit CI guess or a stored guess with a matching GAS basis."""

        if ci0 is None:
            if self._ci_matches_signature(self.ci, self._gas_problem_signature()):
                ci0 = self.ci
            else:
                # Also clear self.ci: native CASSCF falls back to this slot
                # when the explicitly supplied ci0 is None.
                self._clear_ci_guess()
        return ci0

    def _prepare_fixed_orbital_gasci(
            self, mo_coeff=None, ci0=None, *, validate=False, verbose=None):
        """Resolve MO/CI guesses, validating at public calculation entries.

        Internal Newton calls use the already validated problem and updated
        orbitals; they must not repeat the AO metric check at every macro step.
        """

        _check_symmetry(self, mo_coeff)
        if validate:
            self.validate_capabilities()
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
            if mo_coeff is None and self._scf.mol.nelectron > 0:
                self._scf.run()
                mo_coeff = self._scf.mo_coeff
        if validate:
            self._check_mo_orthonormality(mo_coeff, verbose)
        self.mo_coeff = mo_coeff
        if validate:
            self.check_sanity()
        ci0 = self._resolve_ci_guess(ci0)
        self.fcisolver.mol = self.mol
        return mo_coeff, ci0

    def _run_fixed_orbital_gasci(
            self, mo_coeff=None, ci0=None, verbose=None, eris=None):
        """Run fixed-orbital GASCI and update result slots."""

        mo_coeff, ci0 = self._prepare_fixed_orbital_gasci(mo_coeff, ci0)
        signature = self._gas_problem_signature()
        gasci._clear_energy_results(self)
        gasci_obj = (
            self if eris is None else
            mc1step._fake_h_for_fast_casci(self, mo_coeff, eris))
        objective, gas_objective, self.ci = gasci.kernel(
            gasci_obj, mo_coeff, ci0=ci0, verbose=verbose)
        self._gas_ci_signature = signature
        if getattr(self.fcisolver, "converged", None) is not None:
            self.converged = bool(numpy.all(self.fcisolver.converged))
        else:
            self.converged = True
        gasci._publish_energy_results(self, objective, gas_objective)
        # This return belongs to the internal Newton bridge, not the public API.
        return objective, gas_objective, self.ci

    def gasci(self, mo_coeff=None, ci0=None, verbose=None):
        """Run the fixed-orbital GASCI problem associated with this object.

        This convenience method solves the GASCI problem associated with the
        current orbitals without performing orbital optimization. An implicit
        CI guess is reused only for the same normalized GAS space/root count.
        Pass ci0 explicitly to supply a guess without a recorded GAS signature.
        """

        mo_coeff, ci0 = self._prepare_fixed_orbital_gasci(
            mo_coeff, ci0, validate=True, verbose=verbose)
        e_tot, e_gas, ci = self._run_fixed_orbital_gasci(
            mo_coeff, ci0, verbose)
        return self.e_tot, self.e_gas, ci, self.mo_coeff, self.mo_energy

    def casci(self, mo_coeff=None, ci0=None, eris=None,
              verbose=None, envs=None):
        """Internal Newton bridge: return objective energies, not public energies.

        Use :meth:`gasci` for a public fixed-orbital calculation.
        """

        log = logger.new_logger(self, verbose)
        if hasattr(self.fcisolver, "ss_penalty"):
            log.info("Spin penalty active: internal GASCI/macro E and dE "
                     "refer to the optimization objective")
        e_tot, e_gas, ci = self._run_fixed_orbital_gasci(
            mo_coeff, ci0, verbose, eris=eris)

        if numpy.ndim(e_gas) != 0:
            raise RuntimeError(
                "Multiple roots are detected in fcisolver.  GASSCF does not "
                "know which state to optimize.\n"
                "Use state_average(weights) for a multiroot GASSCF objective.")

        # Mirror pyscf.mcscf.newton_casscf.CASSCF.casci logging exactly,
        # changing only CAS/CASSCF terminology to GAS/GASSCF.
        if envs is not None and log.verbose >= logger.INFO:
            log.debug("GAS space CI energy = %.15g", e_gas)

            ss = None
            if getattr(self.fcisolver, "spin_square", None):
                try:
                    ss = self.fcisolver.spin_square(
                        ci, self.ncas, self.nelecas)
                except NotImplementedError:
                    ss = None

            if "imacro" in envs:
                stat = envs["stat"]
                message = ("macro %d (%d JK  %d micro), "
                           "GASSCF E = %.15g  dE = %.4g  |grad|=%5.3g")
                values = (envs["imacro"], stat.tot_hop + stat.tot_kf,
                          stat.imic, e_tot, e_tot - envs["elast"],
                          envs["norm_gall"])
            else:
                message, values = "GASCI E = %.15g", (e_tot,)
                if ss is not None:
                    message += "  dE = %.8g"
                    values += (e_tot - envs.get("elast", 0),)
            if ss is not None:
                message += "  S^2 = %.7f"
                values += (ss[0],)
            log.info(message, *values)

        return e_tot, e_gas, ci

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, sort=False,
                     gas_natorb=False, gasdm1=None, verbose=None,
                     cas_natorb=None, *, gas_pseudo_natorb=False, casdm1=None,
                     **kwargs):
        """Return ``(mo_coeff, ci, mo_energy)`` without writing to this object.

        By default only core/external orbitals are canonicalized. Opt in with
        ``gas_pseudo_natorb=True`` to diagonalize the density separately within
        each GAS subspace and transform every CI root into the new basis,
        including zero-weight roots. SA uses the weighted density unless
        ``gasdm1`` is supplied in the input active-orbital basis. ``casdm1``
        is a compatibility alias; supply only one of these density arguments.

        Active occupations are ordered from largest to smallest within each
        unfrozen block of a GAS subspace. Frozen orbitals are preserved;
        spatial-symmetry settings and labels are not supported. ``sort`` orders
        unfrozen core/external energies. ``mo_energy`` contains
        Fock diagonal elements, not pseudo-natural occupations.

        These are pseudo-natural orbitals: density between GAS subspaces need
        not vanish. Canonicalization across GAS subspaces is unsupported, as are
        ``gas_natorb=True`` and ``cas_natorb=True``. Use ``canonicalize_`` to
        write the returned orbitals, CI and orbital energies to this object.
        """

        _check_symmetry(self, mo_coeff)
        if gas_natorb or cas_natorb:
            _unsupported("GAS natural-orbital rotation")
        if casdm1 is not None:
            if gasdm1 is not None:
                raise ValueError("supply only one of gasdm1 and casdm1")
            gasdm1 = casdm1
        if not gas_pseudo_natorb:
            return gasci.GASCI.canonicalize(
                self, mo_coeff, ci, eris, sort=sort, gasdm1=gasdm1,
                verbose=verbose, **kwargs)

        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        ci = self.ci if ci is None else ci
        if mo_coeff is None or ci is None:
            raise ValueError("pseudo-natural canonicalization requires orbitals and CI")
        if gasdm1 is None:
            gasdm1 = self.make_gasdm1(
                ci=ci, state=None if self._has_state_weights() else 0)
        gasdm1 = numpy.asarray(gasdm1)
        if (gasdm1.shape != (self.ncas, self.ncas)
                or numpy.iscomplexobj(gasdm1)
                or not numpy.all(numpy.isfinite(gasdm1))):
            raise ValueError("gasdm1 must be a finite real (ncas, ncas) matrix")

        nmo = mo_coeff.shape[1]
        frozen = numpy.zeros(nmo, dtype=bool)
        if isinstance(self.frozen, (int, numpy.integer)):
            frozen[:self.frozen] = True
        elif self.frozen is not None:
            frozen[self.frozen] = True
        rotation = numpy.eye(self.ncas)
        offset = 0
        for size in self.gas_orbs:
            indices = numpy.arange(offset, offset + size)
            indices = indices[~frozen[self.ncore + indices]]
            if indices.size:
                block = numpy.ix_(indices, indices)
                _, vectors = self._natural_eigensystem(gasdm1[block], sort=True)
                rotation[block] = vectors
            offset += size

        # Transform CI explicitly, without exposing a general rotation hook to
        # Newton. In particular, never solve GASCI again in the rotated basis.
        def transform(vector):
            return self.fcisolver.transform_ci_within_gas(
                vector, self.ncas, self.nelecas, rotation)

        if isinstance(ci, (list, tuple)):
            ci_new = type(ci)(transform(vector) for vector in ci)
        elif numpy.ndim(ci) == 2 and self.fcisolver.nroots > 1:
            ci_new = numpy.asarray([transform(vector) for vector in ci])
        else:
            ci_new = transform(ci)
        # Both calls use the original basis, so a supplied ERIS remains valid.
        fock = self.get_fock(mo_coeff, ci, eris, gasdm1, verbose)
        mo_new, _, mo_energy = gasci.GASCI.canonicalize(
            self, mo_coeff, ci, eris, sort=sort, gasdm1=gasdm1,
            verbose=verbose, **kwargs)
        active = slice(self.ncore, self.ncore + self.ncas)
        mo_new[:, active] = mo_coeff[:, active] @ rotation
        mo_energy[active] = numpy.einsum(
            'pi,pi->i', mo_new[:, active], fock @ mo_new[:, active])
        return mo_new, ci_new, mo_energy

    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      gas_natorb=False, gasdm1=None, verbose=None,
                      cas_natorb=None, *, gas_pseudo_natorb=False, casdm1=None,
                      **kwargs):
        """Write the canonicalized MO, CI and orbital energies to this object.

        Returns the same three values as :meth:`canonicalize`; the method
        without the trailing underscore leaves these result attributes intact.
        """

        mo_coeff, ci, mo_energy = self.canonicalize(
            mo_coeff, ci, eris, sort=sort, gas_natorb=gas_natorb,
            gasdm1=gasdm1, verbose=verbose, cas_natorb=cas_natorb,
            gas_pseudo_natorb=gas_pseudo_natorb, casdm1=casdm1, **kwargs)
        self.mo_coeff = mo_coeff
        self.ci = ci
        self.mo_energy = mo_energy
        return mo_coeff, ci, mo_energy

    def cas_natorb(self, *args, **kwargs):
        _unsupported("CAS/GAS natural-orbital rotation")

    cas_natorb_ = cas_natorb

    @staticmethod
    def _validate_weights(weights):
        """Return at least two validated state-average weights."""

        weights = addons_gas._validate_state_weights(weights)
        if weights.size < 2:
            raise ValueError("state_average requires at least two weights")
        return tuple(float(value) for value in weights)

    def state_average(self, weights=(.5, .5), wfnsym=None):
        """Return an ordinary state-average GASSCF object.

        Zero-weight roots are still solved and stored, but contribute neither
        to the weighted energy nor to its gradient and Hessian. Their presence
        may affect multiroot solver work and the numerical optimization path.
        Supply at least two finite nonnegative weights summing to one. This
        method returns a copy; state_average_() updates the object in place.
        ``wfnsym`` and SA-mix are not supported.
        """

        if wfnsym is not None:
            _unsupported("wfnsym")
        self.validate_capabilities()
        weights = self._validate_weights(weights)
        source = self.undo_state_average() if isinstance(
            self, addons.StateAverageMCSCF) else self.copy()
        source.validate_capabilities()
        result = addons.state_average(source, weights, wfnsym=None)
        result.__class__ = lib.replace_class(
            result.__class__, addons.StateAverageMCSCF, _StateAverageGASSCF)
        gasci._clear_energy_results(result)
        return result.validate_capabilities()

    def state_average_(self, weights=(.5, .5), wfnsym=None):
        result = self.state_average(weights, wfnsym)
        self.close()
        self.__class__ = result.__class__
        self.__dict__ = result.__dict__
        return self

    def state_average_mix(self, *args, **kwargs):
        _unsupported("state-average-mix GASSCF")

    state_average_mix_ = state_average_mix

    def fix_spin_(self, shift=.2, ss=None):
        """Enable the GASCI-native spin penalty in place and return ``self``.

        Both ``fix_spin`` and ``fix_spin_`` modify this object. The target
        ``ss`` is the ``S(S+1)`` value. The GASCI spin-penalty Hamiltonian
        operates directly in the GAS representation and requires a
        spin-complete GAS restriction.
        """

        self.validate_capabilities()
        self.close()
        gasci._set_spin_penalty(self, shift, ss)
        return self.validate_capabilities()

    fix_spin = fix_spin_

    def undo_fix_spin_(self):
        """Disable GASCI-native spin penalty in place."""

        self.close()
        gasci.GASCI.undo_fix_spin_(self)
        return self.validate_capabilities()

    def undo_fix_spin(self):
        """Return a copied GASSCF object without spin penalty."""

        return self.copy().undo_fix_spin_()

    def density_fit(self, auxbasis=None, with_df=None):
        """Return a DF-GASSCF object using CASSCF DF machinery.

        Density fitting changes only integral and J/K construction; GAS
        restrictions, CI vectors, RDMs and spin-penalty handling remain
        managed by the GASSCF/GASCI layer.
        """

        self.validate_capabilities()
        result = mcdf.density_fit(self, auxbasis=auxbasis, with_df=with_df)
        # Native DF construction copies __dict__ without calling our copy().
        # Do not detach or close caches on an idempotent return of self.
        if result is not self and result.fcisolver is self.fcisolver:
            result.fcisolver = self.fcisolver.copy()
            result.fcisolver.mol = result.mol
        if isinstance(result, _DFGASSCF):
            return result
        if isinstance(result, mcdf._DFCASSCF):
            result.__class__ = lib.replace_class(
                result.__class__, mcdf._DFCASSCF, _DFGASSCF)
        return result

    def approx_hessian(self, auxbasis=None, with_df=None):
        """Reject the unvalidated native DF-only Hessian approximation."""

        _unsupported("density-fitted approximate Hessian")

    def sfx2c1e(self, *args, **kwargs):
        """Reject X2C conversion, including the inherited alias names."""
        _unsupported("X2C")

    x2c = x2c1e = sfx2c1e

    def ddCOSMO(self, *args, **kwargs):
        """Reject solvent wrappers, including the inherited alias names."""
        _unsupported("solvent models")

    DDCOSMO = ddPCM = DDPCM = PCM = ddCOSMO

    def as_scanner(self):
        """Return an energy-only scanner for a fixed GASSCF objective.

        Stored CI guesses are copied and reused only when their recorded GAS
        basis matches the current model. Checkpoint CI has no such metadata;
        pass it explicitly as ``scanner(mol, ci0=ci)`` when its basis is known
        to be compatible. Creating a scanner does not change the source CI.
        Resetting a scanner invalidates its scan report and convergence flag;
        stored orbitals and compatible CI remain available as initial guesses.
        Keep atoms/order, charge, spin, basis/ECP, core and GAS definitions,
        root count, weights, spin penalty and frozen orbitals fixed. Explicit
        mo_coeff is interpreted in the current geometry's AO basis; otherwise
        previous orbitals are projected with GAS-block priorities. Root order
        follows each GASCI solve; the scanner does not track state identity.
        """

        return _as_scanner(self)

    def state_specific_(self, *args, **kwargs):
        _unsupported("state-specific GASSCF")

    state_specific = state_specific_

    # StateAverageMCSCF's Gradients/NACs aliases dispatch through these hooks.
    # Defining only the public methods would leave those inherited paths open.
    nuc_grad_method = _nuc_grad_method
    Gradients = nuc_grad_method
    _state_average_nuc_grad_method = nuc_grad_method

    def nac_method(self, *args, **kwargs):
        _unsupported("nonadiabatic coupling evaluation")

    NACs = nac_method
    _state_average_nac_method = nac_method

    gen_g_hop = gen_g_hop

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        """Run joint orbital/CI optimization with native newton_casscf.

        The native Newton macro/micro control flow is reused directly.
        GAS-specific behavior enters through the orbital mask, fixed-orbital
        GASCI bridge, and GASCI solver dispatch methods above. Public energies
        exclude spin penalties; Newton continues to optimize their objective.
        Implicit CI guesses follow the same GAS-space checks as :meth:`gasci`.
        """

        mo_coeff, ci0 = self._prepare_fixed_orbital_gasci(
            mo_coeff, ci0, validate=True)
        gasci._clear_energy_results(self)
        if (self.ncas == mo_coeff.shape[1] and
                not self.internal_rotation and not self.canonicalization):
            e_tot, e_gas, ci = self._run_fixed_orbital_gasci(mo_coeff, ci0)
            self.mo_energy = None
            return self.e_tot, self.e_gas, ci, self.mo_coeff, self.mo_energy
        stdout, restore_stdout = self._push_gasscf_log_labels()
        try:
            result = mc1step.CASSCF.kernel(
                self, mo_coeff, ci0, callback, _gasscf_newton_kernel)
            return result
        finally:
            self.close()
            if restore_stdout:
                self.stdout.flush()
                self.stdout = stdout

    def dump_chk(self, envs_or_file):
        """Use checkpoint format with physical public energies.

        Newton's local dictionary contains objective energies. Substitute the
        public energies in a copy; leave its convergence variables untouched.
        Filename-based calls already use the physical object attributes.
        """
        if isinstance(envs_or_file, dict):
            envs_or_file = dict(envs_or_file)
            envs_or_file["e_tot"] = self.e_tot
            envs_or_file["e_cas"] = self.e_cas
        return super().dump_chk(envs_or_file)

    def mc1step(self, *args, **kwargs):
        """Reject mc1step entry points; use :meth:`kernel` for joint Newton."""
        raise NotImplementedError(
            "GASSCF does not support legacy mc1step entry points; "
            "use kernel() for joint Newton orbital optimization")

    # The joint Newton driver does not use these inherited mc1step helpers.
    solve_approx_ci = mc1step
    update_casdm = mc1step
    rotate_orb_cc = mc1step

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        _unsupported("two-step GASSCF kernel")

    @property
    def gas_orbs(self):
        """Ordered numbers of active orbitals in each GAS subspace."""

        return self.fcisolver.gas_orbs

    @gas_orbs.setter
    def gas_orbs(self, value):
        self.fcisolver.gas_orbs = (
            None if value is None else _validated_user_gas_orbs(value))

    @property
    def gas_restr(self):
        """GAS restriction in the syntax selected by ``gas_restr_type``."""

        return self.fcisolver.gas_restr

    @gas_restr.setter
    def gas_restr(self, value):
        self.fcisolver.gas_restr = value

    @property
    def gas_restr_type(self):
        """Restriction syntax forwarded to GASCI normalization."""

        return self.fcisolver.gas_restr_type

    @gas_restr_type.setter
    def gas_restr_type(self, value):
        self.fcisolver.gas_restr_type = value

    @property
    def cache_plans(self):
        """Whether GASSCF may reuse owned GAS helper plans."""

        return self.fcisolver.cache_plans

    @cache_plans.setter
    def cache_plans(self, value):
        self.fcisolver.cache_plans = bool(value)

    @property
    def ngas(self):
        """Number of user-visible GAS subspaces."""

        return len(self.gas_orbs)

    @property
    def e_gas(self):
        """Alias for the inherited active-space energy slot."""

        return self.e_cas


def DFGASSCF(mf, ncas=None, nelecas=None, gas_orbs=None, gas_restr=None,
             gas_restr_type=None, *, ncore=None, frozen=None, fcisolver=None,
             cache_plans=None, auxbasis=None, with_df=None):
    """Create a density-fitted :class:`GASSCF` object.

    This mirrors ``DFCASCI/DFCASSCF`` construction style while keeping
    the user-facing GAS interface identical to :class:`GASSCF`.
    Mole inputs first create a DF-RHF reference. Explicit DF options are
    applied once, without an intervening automatic DF construction.
    """

    if isinstance(mf, gto.MoleBase):
        mf = mf.RHF().density_fit()
    mc = GASSCF.__new__(GASSCF)
    mc._initialize(
        mf, ncas, nelecas, gas_orbs=gas_orbs, gas_restr=gas_restr,
        gas_restr_type=gas_restr_type, ncore=ncore,
        frozen=frozen, fcisolver=fcisolver,
        cache_plans=cache_plans)
    return mc.density_fit(auxbasis=auxbasis, with_df=with_df)
