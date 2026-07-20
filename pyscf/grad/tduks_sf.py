#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
# Ref:
# J. Chem. Theory Comput. 2025, 21, 6, 3010
#

from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.dft import numint
from pyscf.dft import numint2c
from pyscf.grad import rks as rks_grad
from pyscf.grad import tdrhf as tdrhf_grad
from pyscf.grad import tdrks as tdrks_grad
from pyscf.grad import tduks as tduks_grad
from pyscf.sftda.numint2c_sftd import mcfun_eval_xc_adapter_sf


def grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
    """
    Electronic part of spin-flip TDA/TDDFT nuclear gradients.

    Args:
        td_grad : grad.tduks_sf.Gradients object.

        x_y : a two-element list of numpy arrays
            TDDFT X and Y amplitudes. If Y is set to 0, this function computes
            TDA energy gradients.
    """
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]
    nao = mo_coeff[0].shape[0]
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    if td_grad.base.extype == 0:
        y, x = x_y
        if not isinstance(x, np.ndarray):
            x = np.zeros((nocca, nvirb))
    elif td_grad.base.extype == 1:
        x, y = x_y
        if not isinstance(y, np.ndarray):
            y = np.zeros((noccb, nvira))

    dvva = lib.einsum('ia,ib->ab', y, y)
    dvvb = lib.einsum('ia,ib->ab', x, x)
    dooa = -lib.einsum('ia,ja->ij', x, x)
    doob = -lib.einsum('ia,ja->ij', y, y)

    dmzooa = reduce(np.dot, (orboa, dooa, orboa.T))
    dmzooa += reduce(np.dot, (orbva, dvva, orbva.T))
    dmzoob = reduce(np.dot, (orbob, doob, orbob.T))
    dmzoob += reduce(np.dot, (orbvb, dvvb, orbvb.T))

    dmx = reduce(np.dot, (orbvb, x.T, orboa.T))  # ba
    dmy = reduce(np.dot, (orbob, y, orbva.T))  # ba
    dmt = dmx + dmy

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

    f1vo, f1oo, vxc1, k1ao = _contract_xc_kernel(td_grad, mf.xc, dmt, (dmzooa, dmzoob), True, True, max_memory)

    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        vj0, vk0 = mf.get_jk(mol, (dmzooa, dmzoob), hermi=1)  # (2, nao, nao)
        vk1 = mf.get_k(mol, dmt, hermi=0) * hyb  # (nao, nao)
        vk0 = vk0 * hyb
        if omega != 0:
            vk0 += mf.get_k(mol, (dmzooa, dmzoob), hermi=1, omega=omega) * (alpha - hyb)
            vk1 += mf.get_k(mol, dmt, hermi=0, omega=omega) * (alpha - hyb)

        veff0doo = vj0[0] + vj0[1] - vk0 + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo = reduce(np.dot, (mo_coeff[1].T, f1vo[0] - vk1, mo_coeff[0]))
        wvoa += lib.einsum('ac,ka->ck', veff0mo[noccb:, nocca:], x)
        wvoa -= lib.einsum('jk,jc->ck', veff0mo[:noccb, :nocca], y)
        wvob += lib.einsum('ac,ka->ck', veff0mo.T[nocca:, noccb:], y)
        wvob -= lib.einsum('jk,jc->ck', veff0mo.T[:nocca, :noccb], x)

    else:
        vj0 = mf.get_j(mol, (dmzooa, dmzoob), hermi=1)  # (2, nao, nao)
        veff0doo = vj0[0] + vj0[1] + f1oo[:, 0] + k1ao[:, 0]
        wvoa = reduce(np.dot, (orbva.T, veff0doo[0], orboa))
        wvob = reduce(np.dot, (orbvb.T, veff0doo[1], orbob))

        veff0mo = reduce(np.dot, (mo_coeff[1].T, f1vo[0], mo_coeff[0]))
        wvoa += lib.einsum('ac,ka->ck', veff0mo[noccb:, nocca:], x)
        wvoa -= lib.einsum('jk,jc->ck', veff0mo[:noccb, :nocca], y)
        wvob += lib.einsum('ac,ka->ck', veff0mo.T[nocca:, noccb:], y)
        wvob -= lib.einsum('jk,jc->ck', veff0mo.T[:nocca, :noccb], x)

    vresp = mf.gen_response(hermi=1)

    def fvind(x):
        xa = x[0, : nvira * nocca].reshape(nvira, nocca)
        xb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        dma = reduce(np.dot, (orbva, xa, orboa.T))
        dmb = reduce(np.dot, (orbvb, xb, orbob.T))
        dm1 = np.stack((dma + dma.T, dmb + dmb.T))
        v1 = vresp(dm1)
        v1a = reduce(np.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(np.dot, (orbvb.T, v1[1], orbob))
        return np.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(
        fvind, mo_energy, mo_occ, (wvoa, wvob), max_cycle=td_grad.cphf_max_cycle, tol=td_grad.cphf_conv_tol
    )[0]
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = np.empty((2, nao, nao))
    z1ao[0] = reduce(np.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(np.dot, (orbvb, z1b, orbob.T))
    veff = vresp((z1ao + z1ao.transpose(0, 2, 1)))

    im0a = np.zeros((nmoa, nmoa))
    im0b = np.zeros((nmob, nmob))
    im0a[:nocca, :nocca] = reduce(np.dot, (orboa.T, veff0doo[0] + veff[0], orboa))
    im0b[:noccb, :noccb] = reduce(np.dot, (orbob.T, veff0doo[1] + veff[1], orbob))
    im0a[:nocca, :nocca] += lib.einsum('al,ka->lk', veff0mo[noccb:, :nocca], x)
    im0b[:noccb, :noccb] += lib.einsum('al,ka->lk', veff0mo.T[nocca:, :noccb], y)
    im0a[nocca:, nocca:] = lib.einsum('jd,jc->dc', veff0mo[:noccb, nocca:], y)
    im0b[noccb:, noccb:] = lib.einsum('jd,jc->dc', veff0mo.T[:nocca, noccb:], x)
    im0a[:nocca, nocca:] = lib.einsum('jk,jc->kc', veff0mo[:noccb, :nocca], y) * 2
    im0b[:noccb, noccb:] = lib.einsum('jk,jc->kc', veff0mo.T[:nocca, :noccb], x) * 2

    zeta_a = (mo_energy[0][:, None] + mo_energy[0]) * 0.5
    zeta_b = (mo_energy[1][:, None] + mo_energy[1]) * 0.5
    zeta_a[nocca:, :nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:, :noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca, nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb, noccb:] = mo_energy[1][noccb:]
    dm1a = np.zeros((nmoa, nmoa))
    dm1b = np.zeros((nmob, nmob))
    dm1a[:nocca, :nocca] = dooa
    dm1b[:noccb, :noccb] = doob
    dm1a[nocca:, nocca:] = dvva
    dm1b[noccb:, noccb:] = dvvb
    dm1a[nocca:, :nocca] = z1a * 2
    dm1b[noccb:, :noccb] = z1b * 2
    dm1a[:nocca, :nocca] += np.eye(nocca)  # for ground state
    dm1b[:noccb, :noccb] += np.eye(noccb)
    im0a = reduce(np.dot, (mo_coeff[0], im0a + zeta_a * dm1a, mo_coeff[0].T))
    im0b = reduce(np.dot, (mo_coeff[1], im0b + zeta_b * dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    mf_grad = td_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dmz1dooa = 4 * z1ao[0] + 2 * dmzooa
    dmz1doob = 4 * z1ao[1] + 2 * dmzoob
    oo0a = reduce(np.dot, (orboa, orboa.T))
    oo0b = reduce(np.dot, (orbob, orbob.T))
    as_dm1 = oo0a + oo0b + (dmz1dooa + dmz1doob) * 0.5

    if with_k:
        dm = (oo0a, dmz1dooa + dmz1dooa.T, oo0b, dmz1doob + dmz1doob.T)
        vj, vk = td_grad.get_jk(mol, dm, hermi=1)
        vj = vj.reshape(2, 2, 3, nao, nao)
        vk = vk.reshape(2, 2, 3, nao, nao) * hyb
        vk1 = -td_grad.get_k(mol, (dmt, dmt.T)) * hyb
        if omega != 0:
            vk += td_grad.get_k(mol, dm, omega=omega).reshape(2, 2, 3, nao, nao) * (alpha - hyb)
            vk1 += -td_grad.get_k(mol, (dmt, dmt.T), omega=omega) * (alpha - hyb)
        veff1 = vj[0] + vj[1] - vk
    else:
        dm = (oo0a, dmz1dooa + dmz1dooa.T, oo0b, dmz1doob + dmz1doob.T)
        vj = td_grad.get_j(mol, dm, hermi=1).reshape(2, 2, 3, nao, nao)
        veff1 = vj[0] + vj[1]
        veff1 = np.stack((veff1, veff1))

    fxcz1 = tduks_grad._contract_xc_kernel(td_grad, mf.xc, 2 * z1ao, None, False, False, max_memory)[0]
    veff1[:, 0] += vxc1[:, 1:]
    veff1[:, 1] += (f1oo[:, 1:] + fxcz1[:, 1:] + k1ao[:, 1:]) * 4
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = np.zeros((len(atmlst), 3))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]

        # Ground state gradients
        h1ao = hcore_deriv(ia)
        de[k] = lib.einsum('xpq,pq->x', h1ao, as_dm1)
        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], oo0a[p0:p1]) * 2
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], oo0b[p0:p1]) * 2

        # Excitation energy gradients
        de[k] -= lib.einsum('xpq,pq->x', s1[:, p0:p1], im0[p0:p1])
        de[k] -= lib.einsum('xqp,pq->x', s1[:, p0:p1], im0[:, p0:p1])

        de[k] += lib.einsum('xpq,pq->x', veff1a[0, :, p0:p1], dmz1dooa[p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,pq->x', veff1b[0, :, p0:p1], dmz1doob[p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,qp->x', veff1a[0, :, p0:p1], dmz1dooa[:, p0:p1]) * 0.5
        de[k] += lib.einsum('xpq,qp->x', veff1b[0, :, p0:p1], dmz1doob[:, p0:p1]) * 0.5
        de[k] += lib.einsum('xij,ij->x', veff1a[1, :, p0:p1], oo0a[p0:p1]) * 0.5
        de[k] += lib.einsum('xij,ij->x', veff1b[1, :, p0:p1], oo0b[p0:p1]) * 0.5

        if td_grad.base.collinear_samples > 0:
            de[k] += lib.einsum('xpq,pq->x', f1vo[1:, p0:p1], dmt[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', f1vo[1:, p0:p1], dmt.T[p0:p1]) * 2

        if with_k:
            de[k] += lib.einsum('xpq,pq->x', vk1[0, :, p0:p1], dmt[p0:p1]) * 2
            de[k] += lib.einsum('xpq,pq->x', vk1[1, :, p0:p1], dmt.T[p0:p1]) * 2

        de[k] += td_grad.extra_force(ia, locals())

    log.timer('TDUKS nuclear gradients', *time0)
    return de


def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True, with_kxc=True, max_memory=2000):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dmvo = (dmvo + dmvo.T) * 0.5

    f1vo = np.zeros((4, nao, nao))
    deriv = 2
    if dmoo is not None:
        f1oo = np.zeros((2, 4, nao, nao))
    else:
        f1oo = None
    if with_vxc:
        v1ao = np.zeros((2, 4, nao, nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao = np.zeros((2, 4, nao, nao))
        deriv = 3
    else:
        k1ao = None

    if td_grad.base.collinear_samples > 0:
        # create a mc object to use mcfun.
        nimc = numint2c.NumInt2C()
        nimc.collinear = 'mcol'
        nimc.collinear_samples = td_grad.base.collinear_samples
        eval_xc_eff = mcfun_eval_xc_adapter_sf(nimc, xc_code)

    if xctype == 'HF':
        return f1vo, f1oo, v1ao, k1ao
    elif xctype == 'LDA':
        fmat_, ao_deriv = tdrks_grad._lda_eval_mat_, 1
    elif xctype == 'GGA':
        fmat_, ao_deriv = tdrks_grad._gga_eval_mat_, 2
    elif xctype == 'MGGA':
        fmat_, ao_deriv = tdrks_grad._mgga_eval_mat_, 2
        logger.warn(td_grad, 'TDUKS-MGGA Gradients may be inaccurate due to grids response')
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        if xctype == 'LDA':
            ao0 = ao[0]
        else:
            ao0 = ao
        rho = (
            ni.eval_rho2(mol, ao0, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl=False),
            ni.eval_rho2(mol, ao0, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl=False),
        )
        if td_grad.base.collinear_samples > 0:
            rho_z = np.array([rho[0] + rho[1], rho[0] - rho[1]])
            fxc_sf, kxc_sf = eval_xc_eff(xc_code, rho_z, deriv, xctype=xctype)[2:4]
            kxc_sf = np.stack((kxc_sf[:, :, 0] + kxc_sf[:, :, 1], kxc_sf[:, :, 0] - kxc_sf[:, :, 1]), axis=2)
            rho1 = ni.eval_rho(mol, ao0, dmvo, mask, xctype, hermi=1, with_lapl=False)
            if xctype == 'LDA':
                rho1 = rho1[np.newaxis]
            wv = lib.einsum('yg,xyg,g->xg', rho1, 2 * fxc_sf, weight)  # 2 for f_xx + f_yy
            fmat_(mol, f1vo, ao, wv, mask, shls_slice, ao_loc)

            if with_kxc:
                wv = lib.einsum('xg,yg,xyczg,g->czg', rho1, rho1, 2 * kxc_sf, weight)
                fmat_(mol, k1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
                fmat_(mol, k1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

        if dmoo is not None or with_vxc:
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv=2, spin=1)[1:]

        if dmoo is not None:
            rho2 = np.asarray(
                (
                    ni.eval_rho(mol, ao0, dmoo[0], mask, xctype, hermi=1, with_lapl=False),
                    ni.eval_rho(mol, ao0, dmoo[1], mask, xctype, hermi=1, with_lapl=False),
                )
            )
            if xctype == 'LDA':
                rho2 = rho2[:, np.newaxis]
            wv = lib.einsum('axg,axbyg,g->byg', rho2, fxc, weight)
            fmat_(mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)

        if with_vxc:
            wv = vxc * weight
            fmat_(mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    f1vo[1:] *= -1
    if f1oo is not None:
        f1oo[:, 1:] *= -1
    if v1ao is not None:
        v1ao[:, 1:] *= -1
    if k1ao is not None:
        k1ao[:, 1:] *= -1
    return f1vo, f1oo, v1ao, k1ao


class Gradients(tdrhf_grad.Gradients):
    cphf_max_cycle = tdrhf_grad.Gradients.cphf_max_cycle + 20

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None):
        return grad_elec(self, xy, atmlst, self.max_memory, self.verbose)


Grad = Gradients

from pyscf import sftda

sftda.uks_sf.TDA_SF.Gradients = sftda.uks_sf.TDDFT_SF.Gradients = lib.class_as_method(Gradients)
