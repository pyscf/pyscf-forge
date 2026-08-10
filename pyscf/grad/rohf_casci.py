#!/usr/bin/env python
#
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
# Author: Jiseong Park <fark4308@snu.ac.kr>
# Edited by: Seunghoon Lee <seunghoonlee@snu.ac.kr>

from functools import reduce
from dataclasses import dataclass
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.lib import logger
from pyscf.grad import casci as casci_grad
from pyscf.grad.mp2 import _shell_prange
from scipy.sparse.linalg import LinearOperator, gmres

DEFAULT_CPHF_TOL = 1e-12
DEFAULT_CPHF_MAX_CYCLE = 100

@dataclass
class _ROHFResponseData:
    spaces: tuple
    pairs: list
    wa: numpy.ndarray
    wb: numpy.ndarray
    same_pairs: list
    occa: numpy.ndarray
    occb: numpy.ndarray
    dma: numpy.ndarray
    dmb: numpy.ndarray
    focka: numpy.ndarray
    fockb: numpy.ndarray
    vresp: object
    spin_resolved_response: bool


def _build_rohf_response_data(
    mf,
    mol,
    mo_coeff,
    h1,
    ncore,
    nocc,
    include_active_active=False,
):
    """Build the common-orbital response space and reference densities.

    CASCI is invariant to rotations within the active space, whereas GBCI's
    grouped-bath construction need not be.  ``include_active_active`` adds
    the canonical response of active-active pairs within the same ROHF
    occupation space for the latter case.
    """
    _, nmo = mo_coeff.shape
    source_occ = numpy.asarray(mf.mo_occ)
    coreidx = numpy.where(source_occ == 2)[0]
    openidx = numpy.where(source_occ == 1)[0]
    viridx = numpy.where(source_occ == 0)[0]
    spaces = (coreidx, openidx, viridx)

    if mol.spin >= 0:
        occa = (source_occ > 0).astype(float)
        occb = (source_occ == 2).astype(float)
    else:
        occa = (source_occ == 2).astype(float)
        occb = (source_occ > 0).astype(float)

    pairs = [(e, i) for e in viridx for i in coreidx]
    pairs.extend((e, x) for e in viridx for x in openidx)
    pairs.extend((x, i) for x in openidx for i in coreidx)
    wa = numpy.empty(len(pairs))
    wb = numpy.empty(len(pairs))
    for idx, (p, q) in enumerate(pairs):
        da = occa[q] - occa[p]
        db = occb[q] - occb[p]
        denominator = da + db
        if abs(denominator) < 1e-14:
            raise RuntimeError("ROHF response pair has zero occupation weight")
        wa[idx] = da / denominator
        wb[idx] = db / denominator

    active_mask = numpy.zeros(nmo, dtype=bool)
    active_mask[ncore:nocc] = True
    same_pairs = []
    for space in spaces:
        for p_pos, p in enumerate(space):
            for q in space[:p_pos]:
                crosses_active_boundary = active_mask[p] != active_mask[q]
                both_active = active_mask[p] and active_mask[q]
                if crosses_active_boundary or (
                    include_active_active and both_active
                ):
                    same_pairs.append((p, q))

    dma = (mo_coeff * occa[None, :]) @ mo_coeff.T
    dmb = (mo_coeff * occb[None, :]) @ mo_coeff.T
    focka, fockb = numpy.einsum(
        "pi,spq,qj->sij",
        mo_coeff.conj(),
        h1 + mf.get_veff(mol, (dma, dmb)),
        mo_coeff,
    )
    return _ROHFResponseData(
        spaces=spaces,
        pairs=pairs,
        wa=wa,
        wb=wb,
        same_pairs=same_pairs,
        occa=occa,
        occb=occb,
        dma=dma,
        dmb=dmb,
        focka=focka,
        fockb=fockb,
        vresp=mf.gen_response(hermi=1),
        spin_resolved_response=isinstance(mf, scf.rohf.ROHF),
    )

def _solve_rohf_adjoint(mf, mo_coeff, orbital_gradient, data):
    """Solve the matrix-free adjoint equation in the ROHF response space."""
    nmo = mo_coeff.shape[1]

    def response_transpose(vector):
        vector = numpy.asarray(vector)
        if vector.shape != (len(data.pairs),):
            raise ValueError(
                f"ROHF adjoint vector has shape {vector.shape}, "
                f"expected {(len(data.pairs),)}"
            )
        seeda = numpy.zeros((nmo, nmo), dtype=vector.dtype)
        seedb = numpy.zeros((nmo, nmo), dtype=vector.dtype)
        for value, weight_a, weight_b, (p, q) in zip(
            vector,
            data.wa,
            data.wb,
            data.pairs,
        ):
            seeda[p, q] += value * weight_a
            seedb[p, q] += value * weight_b
        seeda = 0.5 * (seeda + seeda.T)
        seedb = 0.5 * (seedb + seedb.T)

        response = data.focka @ seeda.T + data.focka.T @ seeda
        response += data.fockb @ seedb.T + data.fockb.T @ seedb
        if data.spin_resolved_response:
            seeda_ao = mo_coeff @ seeda @ mo_coeff.T
            seedb_ao = mo_coeff @ seedb @ mo_coeff.T
            va_ao, vb_ao = data.vresp(numpy.asarray((seeda_ao, seedb_ao)))
            va_mo = mo_coeff.T @ va_ao @ mo_coeff
            vb_mo = mo_coeff.T @ vb_ao @ mo_coeff
            response += 2.0 * va_mo * data.occa[None, :]
            response += 2.0 * vb_mo * data.occb[None, :]
        else:
            seed_ao = mo_coeff @ (seeda + seedb) @ mo_coeff.T
            v_mo = mo_coeff.T @ data.vresp(seed_ao) @ mo_coeff
            response += 2.0 * v_mo * (data.occa + data.occb)[None, :]
        return numpy.asarray(
            [response[p, q] - response[q, p] for p, q in data.pairs]
        )

    g_ref = numpy.asarray(
        [orbital_gradient[p, q] for p, q in data.pairs]
    )
    same_seed = numpy.zeros((nmo, nmo))
    same_weighted_pairs = []
    roothaan_energy = numpy.asarray(mf.mo_energy)
    for p, q in data.same_pairs:
        denominator = roothaan_energy[p] - roothaan_energy[q]
        g_pq = orbital_gradient[p, q]
        if abs(denominator) < 1e-10:
            if abs(g_pq) > 1e-12:
                raise RuntimeError("Degenerate active-boundary response is ambiguous")
            continue
        weight = -g_pq / denominator
        same_seed[p, q] += weight
        same_weighted_pairs.append((p, q, weight))

    same_seed = 0.5 * (same_seed + same_seed.T)
    favg = 0.5 * (data.focka + data.fockb)
    same_response = favg @ same_seed.T + favg.T @ same_seed
    if data.spin_resolved_response:
        half_seed_ao = mo_coeff @ (0.5 * same_seed) @ mo_coeff.T
        va_ao, vb_ao = data.vresp(
            numpy.asarray((half_seed_ao, half_seed_ao))
        )
        va_mo = mo_coeff.T @ va_ao @ mo_coeff
        vb_mo = mo_coeff.T @ vb_ao @ mo_coeff
        same_response += 2.0 * va_mo * data.occa[None, :]
        same_response += 2.0 * vb_mo * data.occb[None, :]
    else:
        seed_ao = mo_coeff @ same_seed @ mo_coeff.T
        v_mo = mo_coeff.T @ data.vresp(seed_ao) @ mo_coeff
        same_response += 2.0 * v_mo * (data.occa + data.occb)[None, :]
    same_correction = numpy.asarray(
        [same_response[p, q] - same_response[q, p] for p, q in data.pairs]
    )

    rhs = g_ref + same_correction
    if rhs.size:
        diagonal = numpy.asarray(
            [roothaan_energy[p] - roothaan_energy[q] for p, q in data.pairs]
        )
        safe_diagonal = diagonal.copy()
        small = numpy.abs(safe_diagonal) < 1e-8
        safe_diagonal[small] = numpy.where(
            safe_diagonal[small] < 0.0,
            -1e-8,
            1e-8,
        )
        operator = LinearOperator(
            (rhs.size, rhs.size),
            matvec=response_transpose,
            dtype=rhs.dtype,
        )
        preconditioner = LinearOperator(
            (rhs.size, rhs.size),
            matvec=lambda vector: vector / safe_diagonal,
            dtype=rhs.dtype,
        )
        zvec, info = gmres(
            operator,
            rhs,
            M=preconditioner,
            rtol=DEFAULT_CPHF_TOL,
            atol=0.0,
            restart=min(rhs.size, 40),
            maxiter=DEFAULT_CPHF_MAX_CYCLE,
        )
        residual = float(numpy.linalg.norm(response_transpose(zvec) - rhs))
        residual_limit = max(
            DEFAULT_CPHF_TOL * max(1.0, float(numpy.linalg.norm(rhs))) * 10.0,
            1e-10,
        )
        if residual > residual_limit:
            raise RuntimeError(
                "matrix-free ROHF adjoint did not converge: "
                f"gmres info={info}, residual={residual:.3e}, "
                f"limit={residual_limit:.3e}"
            )
    else:
        zvec = numpy.zeros_like(rhs)
        residual = 0.0

    return zvec, g_ref, same_weighted_pairs,


def _build_rohf_response_ao(
    mo_coeff,
    Imat,
    zvec,
    g_ref,
    same_weighted_pairs,
    data,
):
    """Build AO response densities and the energy-weighted overlap matrix."""
    nmo = mo_coeff.shape[1]
    fock_seed_a = numpy.zeros((nmo, nmo))
    fock_seed_b = numpy.zeros((nmo, nmo))
    for value, weight_a, weight_b, (p, q) in zip(
        zvec,
        data.wa,
        data.wb,
        data.pairs,
    ):
        fock_seed_a[p, q] -= value * weight_a
        fock_seed_b[p, q] -= value * weight_b
    for p, q, weight in same_weighted_pairs:
        fock_seed_a[p, q] += 0.5 * weight
        fock_seed_b[p, q] += 0.5 * weight
    fock_seed_a = 0.5 * (fock_seed_a + fock_seed_a.T)
    fock_seed_b = 0.5 * (fock_seed_b + fock_seed_b.T)
    fock_seed_a_ao = mo_coeff @ fock_seed_a @ mo_coeff.T
    fock_seed_b_ao = mo_coeff @ fock_seed_b @ mo_coeff.T
    zvec_ao = fock_seed_a_ao + fock_seed_b_ao

    # Fold every S1-dependent direct, adjoint, and same-space term into
    # one energy-weighted MO matrix.
    overlap_response = data.focka @ fock_seed_a.T + data.focka.T @ fock_seed_a
    overlap_response += data.fockb @ fock_seed_b.T + data.fockb.T @ fock_seed_b
    if data.spin_resolved_response:
        va_ao, vb_ao = data.vresp(
            numpy.asarray((fock_seed_a_ao, fock_seed_b_ao))
        )
        va_mo = mo_coeff.T @ va_ao @ mo_coeff
        vb_mo = mo_coeff.T @ vb_ao @ mo_coeff
        overlap_response += 2.0 * va_mo * data.occa[None, :]
        overlap_response += 2.0 * vb_mo * data.occb[None, :]
    else:
        v_ao = data.vresp(fock_seed_a_ao + fock_seed_b_ao)
        v_mo = mo_coeff.T @ v_ao @ mo_coeff
        overlap_response += 2.0 * v_mo * (data.occa + data.occb)[None, :]

    overlap_mo = Imat.copy()
    for value, (p, q) in zip(g_ref, data.pairs):
        overlap_mo[p, q] -= 0.5 * value
    for space in data.spaces:
        overlap_mo[numpy.ix_(space, space)] += 0.5 * overlap_response[
            numpy.ix_(space, space)
        ]
    for p, q in data.pairs:
        overlap_mo[p, q] += overlap_response[q, p]
    overlap_ao = mo_coeff @ overlap_mo @ mo_coeff.T

    return zvec_ao, overlap_ao, fock_seed_a_ao, fock_seed_b_ao

def _grad_elec(mc_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc._scf.mo_coeff
    if ci is None: ci = mc.ci
    mf = mc._scf

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mc_grad, verbose)
    mol = mc_grad.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    # neleca, nelecb = mol.nelec
    # assert (neleca == nelecb)
    # orbo = mo_coeff[:,:neleca]
    # orbv = mo_coeff[:,neleca:]

    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(numpy.dot, (mo_cas, casdm1, mo_cas.T))
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_coeff, mo_cas), compact=False)
    aapa = aapa.reshape(ncas,ncas,nmo,ncas)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    # Imat = h1_{pi} gamma1_{iq} + h2_{pijk} gamma_{iqkj}
    Imat = numpy.zeros((nmo,nmo))
    Imat[:,:nocc] = reduce(numpy.dot, (mo_coeff.T, h1 + vhf_c + vhf_a, mo_occ)) * 2
    Imat[:,ncore:nocc] = reduce(numpy.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
    Imat[:,ncore:nocc] += lib.einsum('uviw,vuwt->it', aapa, casdm2)
    orbital_gradient = 2.0 * (Imat - Imat.T)
    aapa = vj = vk = vhf_c = vhf_a = None

    response_data = _build_rohf_response_data(mf, mol,mo_coeff,h1,ncore,nocc)
    zvec, g_ref, same_weighted_pairs= _solve_rohf_adjoint(mf,mo_coeff,orbital_gradient,response_data)
    (zvec_ao, overlap_ao, fock_seed_a_ao, fock_seed_b_ao) = _build_rohf_response_ao(
        mo_coeff, Imat, zvec, g_ref, same_weighted_pairs, response_data)

    dma = response_data.dma
    dmb = response_data.dmb

    casci_dm1 = dm_core + dm_cas
    hcore_deriv = mc_grad.hcore_generator(mol)
    s1 = mc_grad.get_ovlp(mol)
    wtot = fock_seed_a_ao + fock_seed_b_ao
    dmtot = dma + dmb

    diag_idx = numpy.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    max_memory = mc_grad.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / ((aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, casci_dm1)
        de[k] += numpy.einsum('xij,ij->x', h1ao, zvec_ao)

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de[k] -= numpy.einsum('xijw,ijw->x', eri1, dm2_ao) * 2

            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf,-1))
                eri1tmp = eri1tmp.reshape(p1-p0,nf,nao,nao)
                de[k,i] -= numpy.einsum('ijkl,lk,ij', eri1tmp, dm_core, casci_dm1[p0:p1,q0:q1]) * 2
                de[k,i] += numpy.einsum('ijkl,jk,il', eri1tmp, dm_core[q0:q1], casci_dm1[p0:p1])
                de[k,i] -= numpy.einsum('ijkl,lk,ij', eri1tmp, dm_cas, dm_core[p0:p1,q0:q1]) * 2
                de[k,i] += numpy.einsum('ijkl,jk,il', eri1tmp, dm_cas[q0:q1], dm_core[p0:p1])

                de[k,i] -= numpy.einsum("ijkl,ij,kl", eri1tmp, wtot[p0:p1, q0:q1], dmtot, optimize=True) * 2
                de[k,i] -= numpy.einsum("ijkl,kl,ij", eri1tmp, wtot, dmtot[p0:p1, q0:q1], optimize=True) * 2
                for response_density, reference_density in (
                    (fock_seed_a_ao, dma),
                    (fock_seed_b_ao, dmb),
                ):
                    de[k,i] += numpy.einsum("ijkl,il,jk", eri1tmp, response_density[p0:p1],
                                            reference_density[q0:q1], optimize=True) * 2
                    de[k,i] += numpy.einsum("ijkl,kj,li", eri1tmp, response_density[:, q0:q1],
                                            reference_density[:, p0:p1], optimize=True) * 2

            eri1 = eri1tmp = None

        de[k] -= numpy.einsum("xij,ij->x", s1[:, p0:p1], overlap_ao[p0:p1], optimize=True)
        de[k] -= numpy.einsum("xij,ji->x", s1[:, p0:p1], overlap_ao[:, p0:p1], optimize=True)

    log.timer('CASCI nuclear gradients', *time0)
    return de

class Gradients(casci_grad.Gradients):
    def __init__(self, mc):
        casci_grad.Gradients.__init__(self, mc)

    def grad_elec(
        self,
        mo_coeff=None,
        ci=None,
        atmlst=None,
        verbose=None,
    ):
        if isinstance(self.base._scf, scf.rohf.ROHF):
            return _grad_elec(
                self,
                mo_coeff=mo_coeff,
                ci=ci,
                atmlst=atmlst,
                verbose=verbose,
            )

        return super().grad_elec(
            mo_coeff=mo_coeff,
            ci=ci,
            atmlst=atmlst,
            verbose=verbose,
        )
