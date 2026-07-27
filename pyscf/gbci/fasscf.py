#!/usr/bin/env python
#
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
# Authors: Jiseong Park <fark4308@snu.ac.kr>
#          Minseok Oh <msjeff2001@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>


"""Frozen Active-Space SCF tools built on top of PySCF.

This module keeps selected active orbitals fixed while reoptimizing the
complementary orbital space with SCF-like iterations. It is based on
docs/ref_fasscf.py, with a class-centered API and a small helper surface.
"""

from contextlib import contextmanager
from types import MethodType

import numpy as np

from pyscf import __config__, lib
from pyscf.fci import cistring
from pyscf.lib import logger
from pyscf.scf import rohf
from pyscf.soscf import ciah, newton_ah
from pyscf.gbci.gbci import str2occ


TIGHT_GRAD_CONV_TOL = getattr(
    __config__, "scf_hf_kernel_tight_grad_conv_tol", True
)
MISSING = object()
FOCK_ERROR = (
    "FASSCF inactive-block diagonalization requires a 2D effective Fock matrix. "
    "Use RHF/ROHF-like PySCF objects or provide an effective 2D Fock."
)


class FASSCFResult:
    """Container returned by FASSCF kernels."""

    def __init__(
        self,
        converged,
        e_tot,
        mo_energy,
        mo_coeff,
        mo_occ=None,
        dm=None,
        cycles=0,
        mode="fasscf",
    ):
        self.converged = converged
        self.e_tot = e_tot
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.dm = dm
        self.cycles = cycles
        self.mode = mode

    def as_tuple(self):
        """Return a tuple close to the original reference-code API."""
        if self.mode in (
            "group_average",
            "group_average_soscf",
            "group_average_newton_soscf",
        ):
            return self.converged, self.e_tot, self.mo_energy, self.mo_coeff
        return self.converged, self.e_tot, self.mo_energy, self.mo_coeff, self.mo_occ

class FASSCF(rohf.ROHF):
    """ROHF-derived optimizer with frozen active orbitals."""

    def __init__(
        self,
        gbci,
        target=None,
        ncas=None,
        ncore=None,
        nelecas=None,
        mo_coeff=None,
        mo_energy=None,
        conv_tol=1e-10,
        conv_tol_grad=None,
        max_cycle=100,
        dump_chk=True,
        conv_check=True,
        callback=None,
        scf_options=None,
        restore_on_failure=True,
    ):
        mf = gbci._scf

        if hasattr(mf, "mol"):
            rohf.ROHF.__init__(self, mf.mol)
            for key, value in mf.__dict__.items():
                if key not in ("_chkfile", "chkfile"):
                    setattr(self, key, value)
        else:
            rohf.ROHF.__init__(self, mf)

        ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
        ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
        mo_coeff = getattr(gbci, "mo_coeff", None) if mo_coeff is None else mo_coeff
        mo_energy = getattr(gbci, "mo_energy", None) if mo_energy is None else mo_energy

        active_orbitals = range(ncore, ncore+ ncas)
        core_orbitals = range(ncore)

        self.gbci = gbci
        self.target = target
        self.ncas = ncas
        self.ncore = ncore
        self.nelecas = nelecas
        self.active_orbitals = None if active_orbitals is None else list(active_orbitals)
        self.core_orbitals = None if core_orbitals is None else list(core_orbitals)
        if mo_coeff is not None:
            self.mo_coeff = mo_coeff
        if mo_energy is not None:
            self.mo_energy = mo_energy
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.max_cycle = max_cycle
        self.fasscf_dump_chk = dump_chk
        self.conv_check = conv_check
        self.callback = callback
        self.scf_options = {"diis": False}
        self.scf_options.update(scf_options or {})
        self.restore_on_failure = restore_on_failure

        self.converged = False
        self.e_tot = None
        self.mo_occ = None
        self.dm = None
        self.cycles = 0
        self.last_result = None

    def set_scf_options(self, **options):
        """Update default SCF controls used by subsequent kernels."""
        self.scf_options.update(options)
        return self

    def active_occ_from_target(self, target=None):
        """Return the active occupation selected by target."""
        if target is None:
            target = self.target
        if target is None:
            raise ValueError("target is required when active_occ is not provided.")

        target_arr = np.asarray(target)
        if target_arr.ndim != 0:
            return np.asarray(target_arr, dtype=float)

        po_list = self.gbci.possible_occ()
        target_index = int(target_arr)
        if target_index < 0 or target_index >= len(po_list):
            raise IndexError(
                f"target index {target_index} is outside possible_occ size {len(po_list)}."
            )
        return np.asarray(po_list[target_index], dtype=float)

    def kernel(
        self,
        target=None,
        mo_coeff=None,
        mo_energy=None,
        active_orbitals=None,
        core_orbitals=None,
        active_occ=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        conv_check=None,
        scf_options=None,
        **scf_kwargs,
    ):
        """Run active-frozen FASSCF."""
        if "init_dm" in scf_kwargs:
            raise RuntimeError('Keyword argument "init_dm" is replaced by "dm0".')

        mf = self
        mol = mf.mol
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
            logger.info(mf, "Set gradient conv threshold to %g", conv_tol_grad)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        conv_check = self.conv_check if conv_check is None else conv_check
        callback = self.callback if callback is None else callback

        mo_coeff, mo_energy, active_idx, core_idx, active_occ, mo_occ = (
            self.prepare_orbitals(
                mo_coeff, mo_energy, active_orbitals, core_orbitals, active_occ,
                target,
            )
        )
        initial_mo_coeff = np.array(mo_coeff, copy=True)
        options = self.merged_options(
            scf_options,
            scf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )

        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(mf, options):
            dm = mf.make_rdm1(mo_coeff, mo_occ) if dm0 is None else dm0
            h1e = mf.get_hcore(mol)
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "init E= %.15g", e_tot)

            s1e = mf.get_ovlp(mol)
            self.log_overlap_condition(mf, s1e, conv_tol)

            scf_conv = False
            cycles = 0
            if max_cycle <= 0:
                fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)
                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                return self.finish(
                    FASSCFResult(False, e_tot, mo_energy, mo_coeff, mo_occ, dm, 0)
                )

            mf_diis = self.make_diis(mf, h1e, s1e, vhf, dm)
            self.call_hook(mf, "pre_kernel", locals())
            cput1 = logger.timer(mf, "initialize scf", *cput0)
            fock_last = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = np.asarray(
                    mf.get_fock(
                        h1e, s1e, vhf, dm,
                        cycle=cycle, diis=mf_diis, fock_last=fock_last,
                    )
                )
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                dm = mf.make_rdm1(mo_coeff, mo_occ)
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)

                fock_last = fock
                fock_plain = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock_plain.ndim != 2:
                    raise ValueError(FOCK_ERROR)
                norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock_plain))
                if not TIGHT_GRAD_CONV_TOL:
                    norm_gorb /= np.sqrt(norm_gorb.size)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))
                logger.info(
                    mf,
                    "cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )

                if callable(getattr(mf, "check_convergence", None)):
                    scf_conv = mf.check_convergence(locals())
                elif abs(e_tot - last_hf_e) < conv_tol and norm_ddm < np.sqrt(conv_tol):
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())

                cput1 = logger.timer(mf, f"cycle= {cycles}", *cput1)
                if scf_conv:
                    break

            if scf_conv and conv_check:
                dm_last = dm
                last_hf_e = e_tot
                fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                dm = mf.make_rdm1(mo_coeff, mo_occ)
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)
                fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)
                norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
                if not TIGHT_GRAD_CONV_TOL:
                    norm_gorb /= np.sqrt(norm_gorb.size)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))

                loose_e_tol = conv_tol * 10
                loose_g_tol = conv_tol_grad * 3
                if callable(getattr(mf, "check_convergence", None)):
                    scf_conv = mf.check_convergence(locals())
                else:
                    scf_conv = abs(e_tot - last_hf_e) < loose_e_tol or norm_gorb < loose_g_tol
                logger.info(
                    mf,
                    "Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g",
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )
                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())

            if getattr(mf, "disp", None) is not None:
                e_disp = mf.get_dispersion()
                mf.scf_summary["dispersion"] = e_disp
                e_tot += e_disp

            logger.timer(mf, "scf_cycle", *cput0)
            self.call_hook(mf, "post_kernel", locals())

        if not scf_conv and self.restore_on_failure:
            mo_coeff = initial_mo_coeff

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="fasscf",
            )
        )

    run = kernel

    def soscf_kernel(
        self,
        target=None,
        mo_coeff=None,
        active_orbitals=None,
        core_orbitals=None,
        active_occ=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dm0=None,
        freeze_active=True,
        canonicalization=False,
        dump_chk=None,
        callback=None,
        scf_options=None,
        **soscf_kwargs,
    ):
        """Run inactive-only SOSCF with the active orbitals frozen."""
        mf = self
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        mo_coeff, _, active_idx, _, _, mo_occ = self.prepare_orbitals(
            mo_coeff, None, active_orbitals, core_orbitals, active_occ, target
        )
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        soscf = mf.newton()
        options = self.merged_options(
            scf_options,
            soscf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        mol = mf.mol
        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(soscf, options):
            h1e = mf.get_hcore(mol)
            s1e = mf.get_ovlp(mol)
            dm = mf.make_rdm1(mo_coeff, mo_occ) if dm0 is None else dm0
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "SOSCF init E= %.15g", e_tot)

            scf_conv = False
            cycles = 0
            mo_energy = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                g_full, h_op_full, h_diag_full = soscf.gen_g_hop(
                    mo_coeff, mo_occ, fock
                )

                active_mask = np.zeros(mo_occ.size, dtype=bool)
                active_mask[active_idx] = True
                occidxa = mo_occ > 0
                occidxb = mo_occ == 2
                viridxa = ~occidxa
                viridxb = ~occidxb
                uniq_var_a = viridxa[:, None] & occidxa[None, :]
                uniq_var_b = viridxb[:, None] & occidxb[None, :]
                uniq_ab = uniq_var_a | uniq_var_b
                if freeze_active:
                    inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
                else:
                    inactive_rotation = np.ones((mo_occ.size, mo_occ.size), dtype=bool)
                inactive_var = inactive_rotation[uniq_ab]

                g_full = np.asarray(g_full)
                if inactive_var.size != g_full.size:
                    raise RuntimeError(
                        "Unable to map ROHF SOSCF variables to inactive-only space."
                    )
                g_orb = g_full[inactive_var]
                h_diag = np.asarray(h_diag_full)[inactive_var]
                norm_gorb = np.linalg.norm(g_orb)

                if g_orb.size == 0:
                    scf_conv = True
                    break

                def h_op(x):
                    x_full = np.zeros_like(g_full)
                    x_full[inactive_var] = x
                    return np.asarray(h_op_full(x_full))[inactive_var]

                def g_op():
                    return g_orb

                def precond(x, e):
                    hdiagd = h_diag - (e - soscf.ah_level_shift)
                    hdiagd[abs(hdiagd) < 1e-8] = 1e-8
                    return x / hdiagd

                dx = np.zeros_like(g_orb)
                ah_conv_tol = min(norm_gorb**2, soscf.ah_conv_tol)
                ah_start_tol = min(norm_gorb * 5, soscf.ah_start_tol)
                for ah_end, ihop, w, dxi, hdxi, residual, seig in ciah.davidson_cc(
                    h_op,
                    g_op,
                    precond,
                    g_orb,
                    tol=ah_conv_tol,
                    max_cycle=soscf.ah_max_cycle,
                    lindep=soscf.ah_lindep,
                    verbose=logger.new_logger(soscf, soscf.verbose),
                ):
                    if (
                        ah_end
                        or ihop == soscf.ah_max_cycle
                        or (
                            np.linalg.norm(residual) < ah_start_tol
                            and ihop >= soscf.ah_start_cycle
                        )
                        or seig < soscf.ah_lindep
                    ):
                        dx = np.array(dxi, copy=True)
                        dxmax = np.max(abs(dx)) if dx.size else 0
                        if dxmax > soscf.max_stepsize:
                            dx *= soscf.max_stepsize / dxmax
                        break

                dx_full = np.zeros_like(g_full)
                dx_full[inactive_var] = dx
                u = soscf.update_rotate_matrix(dx_full, mo_occ, mo_coeff=mo_coeff)
                mo_coeff = soscf.rotate_mo(mo_coeff, u)
                if freeze_active:
                    mo_coeff[:, active_idx] = active_mo_coeff

                dm = mf.make_rdm1(mo_coeff, mo_occ)
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                fock_mo = mo_coeff.conj().T @ np.asarray(fock) @ mo_coeff
                mo_energy = np.diag(fock_mo)
                if np.allclose(mo_energy.imag, 0):
                    mo_energy = mo_energy.real
                e_tot = mf.energy_tot(dm, h1e, vhf)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))

                logger.info(
                    mf,
                    "SOSCF cycle= %d E= %.15g  delta_E= %4.3g  |g_inact|= %4.3g  |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )

                if abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())
                if scf_conv:
                    break

            logger.timer(mf, "inactive-only SOSCF", *cput0)

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="soscf",
            )
        )

    soscf = soscf_kernel

    def newton_soscf_kernel(
        self,
        target=None,
        mo_coeff=None,
        active_orbitals=None,
        core_orbitals=None,
        active_occ=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dm0=None,
        freeze_active=True,
        canonicalization=False,
        dump_chk=None,
        callback=None,
        scf_options=None,
        **newton_kwargs,
    ):
        """Run PySCF Newton SOSCF with active-involving rotations removed."""
        mf = self
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        mo_coeff, _, active_idx, _, _, mo_occ = self.prepare_orbitals(
            mo_coeff, None, active_orbitals, core_orbitals, active_occ, target
        )
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        soscf = mf.newton()
        original_gen_g_hop = soscf.gen_g_hop
        original_get_grad = soscf.get_grad
        original_update_rotate_matrix = soscf.update_rotate_matrix
        original_rotate_mo = soscf.rotate_mo

        def gen_g_hop_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None, *args, **kwargs):
            g_full, h_op_full, h_diag_full = original_gen_g_hop(
                mo_coeff1, mo_occ1, fock_ao, *args, **kwargs
            )
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]

            g_full = np.asarray(g_full)
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map Newton SOSCF variables to inactive-only space."
                )
            g_orb = g_full[inactive_var]
            h_diag = np.asarray(h_diag_full)[inactive_var]

            def h_op(x):
                x_full = np.zeros_like(g_full)
                x_full[inactive_var] = x
                return np.asarray(h_op_full(x_full))[inactive_var]

            return g_orb, h_op, h_diag

        def get_grad_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None):
            g_full = np.asarray(original_get_grad(mo_coeff1, mo_occ1, fock_ao))
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map Newton SOSCF gradient to inactive-only space."
                )
            return g_full[inactive_var]

        def update_rotate_matrix_inactive(
            newton_mf, dx, mo_occ1, u0=1, mo_coeff=None
        ):
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if np.asarray(dx).size == int(np.count_nonzero(inactive_var)):
                dx_full = np.zeros(inactive_var.size, dtype=np.asarray(dx).dtype)
                dx_full[inactive_var] = dx
            else:
                dx_full = dx
            return original_update_rotate_matrix(dx_full, mo_occ1, u0, mo_coeff)

        def rotate_mo_freeze(newton_mf, mo_coeff1, u, log=None):
            mo = original_rotate_mo(mo_coeff1, u, log)
            if freeze_active:
                mo = np.array(mo, copy=True)
                mo[:, active_idx] = active_mo_coeff
            return mo

        soscf.gen_g_hop = MethodType(gen_g_hop_inactive, soscf)
        soscf.get_grad = MethodType(get_grad_inactive, soscf)
        soscf.update_rotate_matrix = MethodType(update_rotate_matrix_inactive, soscf)
        soscf.rotate_mo = MethodType(rotate_mo_freeze, soscf)

        options = self.merged_options(
            scf_options,
            newton_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        cycle_info = {"cycles": 0}

        def newton_callback(envs):
            if "imacro" in envs:
                cycle_info["cycles"] = max(cycle_info["cycles"], envs["imacro"] + 1)
            if callable(callback):
                callback(envs)

        with self.temporary_options(soscf, options):
            scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = newton_ah.kernel(
                soscf,
                mo_coeff,
                mo_occ,
                dm=dm0,
                conv_tol=soscf.conv_tol,
                conv_tol_grad=soscf.conv_tol_grad,
                max_cycle=soscf.max_cycle,
                dump_chk=dump_chk,
                callback=newton_callback,
                verbose=soscf.verbose,
            )

        dm = mf.make_rdm1(mo_coeff, mo_occ)
        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycle_info["cycles"],
                mode="newton_soscf",
            )
        )

    newton_soscf = newton_soscf_kernel

    def mixed_routine(
        self,
        target=None,
        damp=0.5,
        level_shift=0.5,
        soscf=True,
        max_cycle1=50,
        max_cycle2=100,
        dump_chk=None,
    ):
        options = {"damp": damp, "level_shift": level_shift}
        result = self.kernel(
            target=target,
            max_cycle=max_cycle1,
            dump_chk=dump_chk,
            scf_options=options,
        )

        if soscf and not self.converged:
            return self.newton_soscf_kernel(
                target=target,
                max_cycle=max_cycle2,
                dump_chk=dump_chk,
            )
        return result


    def prepare_orbitals(
        self,
        mo_coeff,
        mo_energy,
        active_orbitals,
        core_orbitals,
        active_occ,
        target=None,
    ):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            mo_coeff = getattr(self, "mo_coeff", None)
        if mo_coeff is None:
            raise ValueError("mo_coeff is required.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("FASSCF expects a 2D MO coefficient array.")

        nmo = mo_coeff.shape[1]
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_energy is None:
            mo_energy = getattr(self, "mo_energy", None)
        if mo_energy is None:
            mo_energy = np.zeros(nmo)
        else:
            mo_energy = np.asarray(mo_energy)

        active_orbitals = (
            self.active_orbitals if active_orbitals is None else active_orbitals
        )
        core_orbitals = self.core_orbitals if core_orbitals is None else core_orbitals
        active_idx = self.validate_indices(active_orbitals, nmo, "active_orbitals")
        core_idx = self.validate_indices(core_orbitals, nmo, "core_orbitals")

        overlap = np.intersect1d(active_idx, core_idx)
        if overlap.size:
            raise ValueError(f"active_orbitals and core_orbitals overlap: {overlap.tolist()}")

        if active_occ is None:
            active_occ = self.active_occ_from_target(target)
        if active_occ is None:
            raise ValueError("active_occ is required.")
        active_occ = np.asarray(active_occ, dtype=float)
        if active_occ.size != active_idx.size:
            raise ValueError(
                f"active_occ has length {active_occ.size}, expected {active_idx.size}."
            )

        mo_occ = np.zeros(nmo)
        mo_occ[core_idx] = 2.0
        mo_occ[active_idx] = active_occ
        return mo_coeff, mo_energy, active_idx, core_idx, active_occ, mo_occ

    def validate_indices(self, indices, nmo, name):
        if indices is None:
            raise ValueError(f"{name} is required.")
        idx = np.asarray(indices, dtype=int).reshape(-1)
        if idx.size != np.unique(idx).size:
            raise ValueError(f"{name} contains duplicate indices.")
        if idx.size and (idx.min() < 0 or idx.max() >= nmo):
            raise ValueError(f"{name} contains indices outside 0..{nmo - 1}.")
        return np.sort(idx)

    def merged_options(self, scf_options, direct_options, **runtime_options):
        options = dict(self.scf_options)
        options.update(scf_options or {})
        options.update(direct_options)
        for key, value in runtime_options.items():
            if value is not None:
                options[key] = value
        return options

    @contextmanager
    def temporary_options(self, obj, options):
        old_values = {}
        for key, value in options.items():
            old_values[key] = getattr(obj, key, MISSING)
            setattr(obj, key, value)
        try:
            yield
        finally:
            for key, old_value in old_values.items():
                if old_value is MISSING:
                    try:
                        delattr(obj, key)
                    except AttributeError:
                        pass
                else:
                    setattr(obj, key, old_value)

    def make_diis(self, mf, h1e, s1e, vhf, dm):
        if isinstance(getattr(mf, "diis", None), lib.diis.DIIS):
            return mf.diis
        if not getattr(mf, "diis", False):
            return None

        diis_class = getattr(mf, "DIIS", None)
        if diis_class is None:
            return None
        mf_diis = diis_class(mf, getattr(mf, "diis_file", None))
        if hasattr(mf_diis, "space"):
            mf_diis.space = getattr(mf, "diis_space", mf_diis.space)
        if hasattr(mf_diis, "rollback"):
            mf_diis.rollback = getattr(mf, "diis_space_rollback", mf_diis.rollback)

        try:
            fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
            if fock.ndim != 2:
                raise ValueError(FOCK_ERROR)
            _, mf_diis.Corth = mf.eig(fock, s1e)
        except Exception as exc:
            logger.debug(mf, "Unable to initialize DIIS Corth: %s", exc)
        return mf_diis

    def log_overlap_condition(self, mf, s1e, conv_tol):
        cond = lib.cond(s1e)
        logger.debug(mf, "cond(S) = %s", cond)
        if np.max(cond) * 1e-17 > conv_tol:
            logger.warn(
                mf,
                "Singularity detected in overlap matrix (condition number = %4.3g). "
                "SCF may be inaccurate and hard to converge.",
                np.max(cond),
            )

    def call_hook(self, obj, name, arg):
        method = getattr(obj, name, None)
        if callable(method):
            return method(arg)
        return None

    def finish(self, result):
        self.converged = result.converged
        self.e_tot = result.e_tot
        self.mo_energy = result.mo_energy
        self.mo_coeff = result.mo_coeff
        self.mo_occ = result.mo_occ
        self.dm = result.dm
        self.cycles = result.cycles
        self.last_result = result
        return result



class GroupAverageFASSCF(FASSCF):
    """FASSCF subclass for group-averaged active-space densities."""

    def kernel(
        self,
        target_group,
        group_info_list,
        gbci=None,
        po_list=None,
        group=None,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        conv_check=None,
        scf_options=None,
        **scf_kwargs,
    ):
        """Run group-average FASSCF using this object as the SCF object."""
        if "init_dm" in scf_kwargs:
            raise RuntimeError('Keyword argument "init_dm" is replaced by "dm0".')

        mf = self
        if not isinstance(mf, rohf.ROHF):
            raise TypeError("FASSCF group-average mode requires a PySCF ROHF-like object.")

        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        conv_check = self.conv_check if conv_check is None else conv_check
        callback = self.callback if callback is None else callback

        gbci = self.gbci if gbci is None else gbci

        if mo_coeff is None and gbci is not None:
            mo_coeff = getattr(gbci, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("mo_coeff is required for group-average FASSCF.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("group-average FASSCF expects a 2D MO coefficient array.")

        if gbci is not None:
            ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
            nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
            ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = self.ncore if ncore is None else ncore
        if ncas is None or nelecas is None or ncore is None:
            raise ValueError("group-average FASSCF requires ncas, nelecas, and ncore.")

        ncas = int(ncas)
        ncore = int(ncore)
        nmo = mo_coeff.shape[1]
        active_idx = np.arange(ncore, ncore + ncas, dtype=int)
        self.validate_indices(active_idx, nmo, "active_orbitals")
        initial_mo_coeff = np.array(mo_coeff, copy=True)

        strings_a = cistring.make_strings(range(ncas), int(nelecas[0]))
        strings_b = cistring.make_strings(range(ncas), int(nelecas[1]))
        nb = len(strings_b)
        if group_info_list is None:
            raise ValueError("group_info_list is required for group-average FASSCF.")

        flat = np.asarray(group_info_list, dtype=object).reshape(-1)
        if flat.size != len(strings_a) * nb:
            raise ValueError(
                "group_info_list size does not match the alpha/beta determinant space."
            )
        target_conf = np.asarray(
            [i for i, label in enumerate(flat) if label == target_group], dtype=int
        )
        if target_conf.size == 0:
            raise ValueError(f"No active-space configurations found for {target_group!r}.")

        stra_idx = target_conf // nb
        strb_idx = target_conf % nb
        occs_a = np.asarray([str2occ(s, ncas) for s in strings_a])
        occs_b = np.asarray([str2occ(s, ncas) for s in strings_b])
        avg_occa = occs_a[stra_idx].mean(axis=0)
        avg_occb = occs_b[strb_idx].mean(axis=0)
        as_mo_coeff = mo_coeff[:, active_idx]
        as_dm_a = (as_mo_coeff * avg_occa).dot(as_mo_coeff.conj().T)
        as_dm_b = (as_mo_coeff * avg_occb).dot(as_mo_coeff.conj().T)
        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        mo_occ[active_idx] = avg_occa + avg_occb

        options = self.merged_options(
            scf_options,
            scf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )

        mol = mf.mol
        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(mf, options):
            if dm0 is None:
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
            else:
                dm = dm0
            h1e = mf.get_hcore(mol)
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "init E= %.15g", e_tot)

            scf_conv = False
            cycles = 0
            s1e = mf.get_ovlp(mol)
            self.log_overlap_condition(mf, s1e, conv_tol)
            mf_diis = self.make_diis(mf, h1e, s1e, vhf, dm)
            cput1 = logger.timer(mf, "initialize scf", *cput0)
            fock_last = None
            mo_energy = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = np.asarray(
                    mf.get_fock(
                        h1e, s1e, vhf, dm,
                        cycle=cycle, diis=mf_diis, fock_last=fock_last,
                    )
                )
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)
                fock_last = fock

                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))
                logger.info(
                    mf,
                    "cycle= %d E= %.15g  delta_E= %4.3g |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_ddm,
                )
                if callable(getattr(mf, "check_convergence", None)):
                    scf_conv = mf.check_convergence(locals())
                elif abs(e_tot - last_hf_e) < conv_tol and norm_ddm < np.sqrt(conv_tol):
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())
                cput1 = logger.timer(mf, f"cycle= {cycles}", *cput1)
                if scf_conv:
                    break

            if scf_conv and conv_check:
                dm_last = dm
                last_hf_e = e_tot
                fock = np.asarray(mf.get_fock(h1e, s1e, vhf, dm))
                if fock.ndim != 2:
                    raise ValueError(FOCK_ERROR)

                nmo = mo_coeff.shape[1]
                inactive_idx = np.setdiff1d(np.arange(nmo), active_idx, assume_unique=True)
                if inactive_idx.size:
                    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
                    reduced_fock = fock_mo[np.ix_(inactive_idx, inactive_idx)]
                    _, inactive_rotation = mf.eig(reduced_fock, np.eye(inactive_idx.size))
                    new_mo_coeff = np.array(mo_coeff, copy=True)
                    new_mo_coeff[:, inactive_idx] = mo_coeff[:, inactive_idx].dot(
                        inactive_rotation
                    )
                    mo_coeff = new_mo_coeff

                fock_diag = np.diag(mo_coeff.conj().T @ fock @ mo_coeff)
                mo_energy = fock_diag.real if np.allclose(fock_diag.imag, 0) else fock_diag
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                e_tot = mf.energy_tot(dm, h1e, vhf)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))
                if abs(e_tot - last_hf_e) < conv_tol or norm_ddm < conv_tol_grad:
                    scf_conv = True
                logger.info(
                    mf,
                    "Extra cycle  E= %.15g  delta_E= %4.3g |ddm|= %4.3g",
                    e_tot,
                    e_tot - last_hf_e,
                    norm_ddm,
                )
                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())

            logger.timer(mf, "scf_cycle", *cput0)

        if not scf_conv and self.restore_on_failure:
            mo_coeff = initial_mo_coeff

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="group_average",
            )
        )

    group_average = kernel

    def mixed_routine(
        self,
        target_group,
        group_info_list,
        damp=0.5,
        level_shift=0.5,
        soscf=True,
        max_cycle1=100,
        max_cycle2=300,
        dump_chk=None,
    ):
        options = {"damp": damp, "level_shift": level_shift}
        result = self.kernel(
            target_group=target_group,
            group_info_list=group_info_list,
            max_cycle=max_cycle1,
            dump_chk=dump_chk,
            scf_options=options,
        )

        if soscf and not self.converged:
            return self.newton_soscf_kernel(
                target_group=target_group,
                group_info_list=group_info_list,
                max_cycle=max_cycle2,
                dump_chk=dump_chk,
            )
        return result

    def soscf_kernel(
        self,
        target_group,
        group_info_list,
        gbci=None,
        po_list=None,
        group=None,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        freeze_active=True,
        canonicalization=False,
        scf_options=None,
        **soscf_kwargs,
    ):
        """Run inactive-only SOSCF with group-averaged active density."""
        if "init_dm" in soscf_kwargs:
            raise RuntimeError('Keyword argument "init_dm" is replaced by "dm0".')

        mf = self
        if not isinstance(mf, rohf.ROHF):
            raise TypeError("FASSCF group-average SOSCF requires a PySCF ROHF-like object.")

        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        if conv_tol_grad is None:
            conv_tol_grad = np.sqrt(conv_tol)
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        gbci = self.gbci if gbci is None else gbci

        if mo_coeff is None and gbci is not None:
            mo_coeff = getattr(gbci, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("mo_coeff is required for group-average SOSCF.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("group-average SOSCF expects a 2D MO coefficient array.")

        if gbci is not None:
            ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
            nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
            ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = self.ncore if ncore is None else ncore
        if ncas is None or nelecas is None or ncore is None:
            raise ValueError("group-average SOSCF requires ncas, nelecas, and ncore.")

        ncas = int(ncas)
        ncore = int(ncore)
        nmo = mo_coeff.shape[1]
        active_idx = np.arange(ncore, ncore + ncas, dtype=int)
        self.validate_indices(active_idx, nmo, "active_orbitals")
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        strings_a = cistring.make_strings(range(ncas), int(nelecas[0]))
        strings_b = cistring.make_strings(range(ncas), int(nelecas[1]))
        nb = len(strings_b)
        if group_info_list is None:
            raise ValueError("group_info_list is required for group-average SOSCF.")

        flat = np.asarray(group_info_list, dtype=object).reshape(-1)
        if flat.size != len(strings_a) * nb:
            raise ValueError(
                "group_info_list size does not match the alpha/beta determinant space."
            )
        target_conf = np.asarray(
            [i for i, label in enumerate(flat) if label == target_group], dtype=int
        )
        if target_conf.size == 0:
            raise ValueError(f"No active-space configurations found for {target_group!r}.")

        stra_idx = target_conf // nb
        strb_idx = target_conf % nb
        occs_a = np.asarray([str2occ(s, ncas) for s in strings_a])
        occs_b = np.asarray([str2occ(s, ncas) for s in strings_b])
        avg_occa = occs_a[stra_idx].mean(axis=0)
        avg_occb = occs_b[strb_idx].mean(axis=0)
        as_dm_a = (active_mo_coeff * avg_occa).dot(active_mo_coeff.conj().T)
        as_dm_b = (active_mo_coeff * avg_occb).dot(active_mo_coeff.conj().T)

        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        mo_occ[active_idx] = avg_occa + avg_occb

        soscf = mf.newton()
        options = self.merged_options(
            scf_options,
            soscf_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        mol = mf.mol
        cput0 = (logger.process_clock(), logger.perf_counter())
        with self.temporary_options(soscf, options):
            h1e = mf.get_hcore(mol)
            s1e = mf.get_ovlp(mol)
            if dm0 is None:
                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
            else:
                dm = dm0
            vhf = mf.get_veff(mol, dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            logger.info(mf, "group-average SOSCF init E= %.15g", e_tot)

            scf_conv = False
            cycles = 0
            mo_energy = None

            for cycle in range(max_cycle):
                cycles = cycle + 1
                dm_last = dm
                last_hf_e = e_tot

                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                g_full, h_op_full, h_diag_full = soscf.gen_g_hop(
                    mo_coeff, mo_occ, fock
                )

                active_mask = np.zeros(nmo, dtype=bool)
                active_mask[active_idx] = True
                occidxa = mo_occ > 0
                occidxb = mo_occ == 2
                viridxa = ~occidxa
                viridxb = ~occidxb
                uniq_var_a = viridxa[:, None] & occidxa[None, :]
                uniq_var_b = viridxb[:, None] & occidxb[None, :]
                uniq_ab = uniq_var_a | uniq_var_b
                if freeze_active:
                    inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
                else:
                    inactive_rotation = np.ones((nmo, nmo), dtype=bool)
                inactive_var = inactive_rotation[uniq_ab]

                g_full = np.asarray(g_full)
                if inactive_var.size != g_full.size:
                    raise RuntimeError(
                        "Unable to map group-average SOSCF variables to inactive-only space."
                    )
                g_orb = g_full[inactive_var]
                h_diag = np.asarray(h_diag_full)[inactive_var]
                norm_gorb = np.linalg.norm(g_orb)

                if g_orb.size == 0:
                    scf_conv = True
                    break

                def h_op(x):
                    x_full = np.zeros_like(g_full)
                    x_full[inactive_var] = x
                    return np.asarray(h_op_full(x_full))[inactive_var]

                def g_op():
                    return g_orb

                def precond(x, e):
                    hdiagd = h_diag - (e - soscf.ah_level_shift)
                    hdiagd[abs(hdiagd) < 1e-8] = 1e-8
                    return x / hdiagd

                dx = np.zeros_like(g_orb)
                ah_conv_tol = min(norm_gorb**2, soscf.ah_conv_tol)
                ah_start_tol = min(norm_gorb * 5, soscf.ah_start_tol)
                for ah_end, ihop, w, dxi, hdxi, residual, seig in ciah.davidson_cc(
                    h_op,
                    g_op,
                    precond,
                    g_orb,
                    tol=ah_conv_tol,
                    max_cycle=soscf.ah_max_cycle,
                    lindep=soscf.ah_lindep,
                    verbose=logger.new_logger(soscf, soscf.verbose),
                ):
                    if (
                        ah_end
                        or ihop == soscf.ah_max_cycle
                        or (
                            np.linalg.norm(residual) < ah_start_tol
                            and ihop >= soscf.ah_start_cycle
                        )
                        or seig < soscf.ah_lindep
                    ):
                        dx = np.array(dxi, copy=True)
                        dxmax = np.max(abs(dx)) if dx.size else 0
                        if dxmax > soscf.max_stepsize:
                            dx *= soscf.max_stepsize / dxmax
                        break

                dx_full = np.zeros_like(g_full)
                dx_full[inactive_var] = dx
                u = soscf.update_rotate_matrix(dx_full, mo_occ, mo_coeff=mo_coeff)
                mo_coeff = soscf.rotate_mo(mo_coeff, u)
                if freeze_active:
                    mo_coeff[:, active_idx] = active_mo_coeff

                core_mo_coeff = mo_coeff[:, :ncore]
                dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
                dm = np.asarray(
                    (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
                )
                vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
                fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
                fock_mo = mo_coeff.conj().T @ np.asarray(fock) @ mo_coeff
                mo_energy = np.diag(fock_mo)
                if np.allclose(mo_energy.imag, 0):
                    mo_energy = mo_energy.real
                e_tot = mf.energy_tot(dm, h1e, vhf)
                norm_ddm = np.linalg.norm(np.asarray(dm) - np.asarray(dm_last))

                logger.info(
                    mf,
                    "group-average SOSCF cycle= %d E= %.15g  delta_E= %4.3g  |g_inact|= %4.3g  |ddm|= %4.3g",
                    cycles,
                    e_tot,
                    e_tot - last_hf_e,
                    norm_gorb,
                    norm_ddm,
                )

                if abs(e_tot - last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
                    scf_conv = True

                if dump_chk:
                    self.call_hook(mf, "dump_chk", locals())
                if callable(callback):
                    callback(locals())
                if scf_conv:
                    break

            logger.timer(mf, "group-average inactive-only SOSCF", *cput0)

        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycles,
                mode="group_average_soscf",
            )
        )

    def newton_soscf_kernel(
        self,
        target_group,
        group_info_list,
        gbci=None,
        po_list=None,
        group=None,
        mo_coeff=None,
        ncas=None,
        nelecas=None,
        ncore=None,
        conv_tol=None,
        conv_tol_grad=None,
        max_cycle=None,
        dump_chk=None,
        dm0=None,
        callback=None,
        freeze_active=True,
        canonicalization=False,
        scf_options=None,
        **newton_kwargs,
    ):
        """Run PySCF Newton SOSCF with group-averaged active density."""
        mf = self
        conv_tol = self.conv_tol if conv_tol is None else conv_tol
        conv_tol_grad = self.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        dump_chk = self.fasscf_dump_chk if dump_chk is None else dump_chk
        callback = self.callback if callback is None else callback

        gbci = self.gbci if gbci is None else gbci
        if mo_coeff is None and gbci is not None:
            mo_coeff = getattr(gbci, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("mo_coeff is required for group-average Newton SOSCF.")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("group-average Newton SOSCF expects a 2D MO coefficient array.")

        if gbci is not None:
            ncas = getattr(gbci, "ncas", None) if ncas is None else ncas
            nelecas = getattr(gbci, "nelecas", None) if nelecas is None else nelecas
            ncore = getattr(gbci, "ncore", None) if ncore is None else ncore
        ncas = self.ncas if ncas is None else ncas
        nelecas = self.nelecas if nelecas is None else nelecas
        ncore = self.ncore if ncore is None else ncore
        if ncas is None or nelecas is None or ncore is None:
            raise ValueError("group-average Newton SOSCF requires ncas, nelecas, and ncore.")

        ncas = int(ncas)
        ncore = int(ncore)
        nmo = mo_coeff.shape[1]
        active_idx = np.arange(ncore, ncore + ncas, dtype=int)
        self.validate_indices(active_idx, nmo, "active_orbitals")
        active_mo_coeff = np.array(mo_coeff[:, active_idx], copy=True)

        strings_a = cistring.make_strings(range(ncas), int(nelecas[0]))
        strings_b = cistring.make_strings(range(ncas), int(nelecas[1]))
        nb = len(strings_b)
        if group_info_list is None:
            raise ValueError("group_info_list is required for group-average Newton SOSCF.")
        flat = np.asarray(group_info_list, dtype=object).reshape(-1)
        if flat.size != len(strings_a) * nb:
            raise ValueError(
                "group_info_list size does not match the alpha/beta determinant space."
            )
        target_conf = np.asarray(
            [i for i, label in enumerate(flat) if label == target_group], dtype=int
        )
        if target_conf.size == 0:
            raise ValueError(f"No active-space configurations found for {target_group!r}.")

        stra_idx = target_conf // nb
        strb_idx = target_conf % nb
        occs_a = np.asarray([str2occ(s, ncas) for s in strings_a])
        occs_b = np.asarray([str2occ(s, ncas) for s in strings_b])
        avg_occa = occs_a[stra_idx].mean(axis=0)
        avg_occb = occs_b[strb_idx].mean(axis=0)
        as_dm_a = (active_mo_coeff * avg_occa).dot(active_mo_coeff.conj().T)
        as_dm_b = (active_mo_coeff * avg_occb).dot(active_mo_coeff.conj().T)

        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        mo_occ[active_idx] = avg_occa + avg_occb

        soscf = mf.newton()
        original_gen_g_hop = soscf.gen_g_hop
        original_get_grad = soscf.get_grad
        original_update_rotate_matrix = soscf.update_rotate_matrix
        original_rotate_mo = soscf.rotate_mo

        def make_group_average_dm(newton_mf, mo_coeff1, mo_occ1=None):
            core_mo_coeff = mo_coeff1[:, :ncore]
            dm_core_per_spin = core_mo_coeff.dot(core_mo_coeff.conj().T)
            return np.asarray(
                (dm_core_per_spin + as_dm_a, dm_core_per_spin + as_dm_b)
            )

        def gen_g_hop_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None, *args, **kwargs):
            g_full, h_op_full, h_diag_full = original_gen_g_hop(
                mo_coeff1, mo_occ1, fock_ao, *args, **kwargs
            )
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]

            g_full = np.asarray(g_full)
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map group-average Newton SOSCF variables to inactive-only space."
                )
            g_orb = g_full[inactive_var]
            h_diag = np.asarray(h_diag_full)[inactive_var]

            def h_op(x):
                x_full = np.zeros_like(g_full)
                x_full[inactive_var] = x
                return np.asarray(h_op_full(x_full))[inactive_var]

            return g_orb, h_op, h_diag

        def get_grad_inactive(newton_mf, mo_coeff1, mo_occ1, fock_ao=None):
            g_full = np.asarray(original_get_grad(mo_coeff1, mo_occ1, fock_ao))
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if inactive_var.size != g_full.size:
                raise RuntimeError(
                    "Unable to map group-average Newton SOSCF gradient to inactive-only space."
                )
            return g_full[inactive_var]

        def update_rotate_matrix_inactive(
            newton_mf, dx, mo_occ1, u0=1, mo_coeff=None
        ):
            active_mask = np.zeros(mo_occ1.size, dtype=bool)
            active_mask[active_idx] = True
            occidxa = mo_occ1 > 0
            occidxb = mo_occ1 == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa[:, None] & occidxa[None, :]
            uniq_var_b = viridxb[:, None] & occidxb[None, :]
            uniq_ab = uniq_var_a | uniq_var_b
            if freeze_active:
                inactive_rotation = (~active_mask[:, None]) & (~active_mask[None, :])
            else:
                inactive_rotation = np.ones((mo_occ1.size, mo_occ1.size), dtype=bool)
            inactive_var = inactive_rotation[uniq_ab]
            if np.asarray(dx).size == int(np.count_nonzero(inactive_var)):
                dx_full = np.zeros(inactive_var.size, dtype=np.asarray(dx).dtype)
                dx_full[inactive_var] = dx
            else:
                dx_full = dx
            return original_update_rotate_matrix(dx_full, mo_occ1, u0, mo_coeff)

        def rotate_mo_freeze(newton_mf, mo_coeff1, u, log=None):
            mo = original_rotate_mo(mo_coeff1, u, log)
            if freeze_active:
                mo = np.array(mo, copy=True)
                mo[:, active_idx] = active_mo_coeff
            return mo

        soscf.make_rdm1 = MethodType(make_group_average_dm, soscf)
        soscf.gen_g_hop = MethodType(gen_g_hop_inactive, soscf)
        soscf.get_grad = MethodType(get_grad_inactive, soscf)
        soscf.update_rotate_matrix = MethodType(update_rotate_matrix_inactive, soscf)
        soscf.rotate_mo = MethodType(rotate_mo_freeze, soscf)

        options = self.merged_options(
            scf_options,
            newton_kwargs,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
        )
        options["canonicalization"] = canonicalization

        cycle_info = {"cycles": 0}

        def newton_callback(envs):
            if "imacro" in envs:
                cycle_info["cycles"] = max(cycle_info["cycles"], envs["imacro"] + 1)
            if callable(callback):
                callback(envs)

        with self.temporary_options(soscf, options):
            scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = newton_ah.kernel(
                soscf,
                mo_coeff,
                mo_occ,
                dm=dm0,
                conv_tol=soscf.conv_tol,
                conv_tol_grad=soscf.conv_tol_grad,
                max_cycle=soscf.max_cycle,
                dump_chk=dump_chk,
                callback=newton_callback,
                verbose=soscf.verbose,
            )

        dm = make_group_average_dm(soscf, mo_coeff, mo_occ)
        return self.finish(
            FASSCFResult(
                bool(scf_conv),
                float(e_tot),
                mo_energy,
                mo_coeff,
                mo_occ,
                dm,
                cycle_info["cycles"],
                mode="group_average_newton_soscf",
            )
        )

    group_average_newton_soscf_kernel = newton_soscf_kernel
    group_average_newton_soscf = newton_soscf_kernel
    group_average_soscf_kernel = soscf_kernel
    group_average_soscf = soscf_kernel


def fasscf_kernel(gbci, target, **kwargs):
    """Function-style compatibility wrapper for ordinary FASSCF."""
    driver = FASSCF(gbci)
    return driver.kernel(target=target, **kwargs).as_tuple()


def state_average_fasscf_kernel(fasscf, target_group, group_info_list, **kwargs):
    """Function-style compatibility wrapper for group-average FASSCF."""
    if not isinstance(fasscf, GroupAverageFASSCF):
        raise TypeError(
            "state_average_fasscf_kernel expects a GroupAverageFASSCF object first."
        )
    return fasscf.group_average_kernel(
        target_group=target_group,
        group_info_list=group_info_list,
        **kwargs,
    ).as_tuple()


def state_average_soscf_kernel(fasscf, target_group, group_info_list, **kwargs):
    """Function-style compatibility wrapper for group-average SOSCF."""
    if not isinstance(fasscf, GroupAverageFASSCF):
        raise TypeError(
            "state_average_soscf_kernel expects a GroupAverageFASSCF object first."
        )
    return fasscf.group_average_soscf_kernel(
        target_group=target_group,
        group_info_list=group_info_list,
        **kwargs,
    ).as_tuple()
