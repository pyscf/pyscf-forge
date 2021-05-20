from __future__ import annotations

from pyscf.dft.numint import _dot_ao_dm, _contract_rho

from dh import RDFDH
from dh.dhutil import calc_batch_size, gen_batch, gen_shl_batch, timing
from pyscf import gto, lib, df
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
import numpy as np

einsum = lib.einsum


def kernel(mf_dh: Gradients):
    # unrestricted method requires dump t_ijab_αβ to disk; controling αβ and SS dumping is too hard for me
    dump_t_ijab = True if mf_dh.unrestricted else mf_dh.with_t_ijab

    mf_dh.build()
    if mf_dh.mo_coeff is NotImplemented:
        mf_dh.run_scf()
    mf_dh.prepare_H_1()
    mf_dh.prepare_S_1()
    mf_dh.prepare_integral()
    mf_dh.prepare_xc_kernel()
    mf_dh.prepare_pt2(dump_t_ijab=dump_t_ijab)
    mf_dh.prepare_lagrangian(gen_W=True)
    mf_dh.prepare_D_r()
    mf_dh.prepare_gradient_jk()
    mf_dh.prepare_gradient_gga()
    mf_dh.prepare_gradient_pt2()
    mf_dh.prepare_gradient_enfunc()
    mf_dh.grad_tot = mf_dh.de = mf_dh.grad_jk + mf_dh.grad_gga + mf_dh.grad_pt2 + mf_dh.grad_enfunc
    return mf_dh.grad_tot


def contract_multiple_rho(ao1, ao2):
    if len(ao1.shape) == 2:
        return _contract_rho(ao1, ao2)
    assert len(ao1.shape) == 3
    res = np.empty(ao1.shape[:2])
    for i in range(ao1.shape[0]):
        res[i] = _contract_rho(ao1[i], ao2)
    return res


@timing
def get_rho_derivs(ao, dm, mol, mask):
    X, Y, Z, XX, XY, XZ, YY, YZ, ZZ = range(1, 10)
    ngrid, natm = ao.shape[1], mol.natm
    nao = dm.shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    # contract for dm
    # aod = einsum("uv, rgv -> rgu", dm, ao[:4])  -- 3 lines
    aod = np.empty((4, ngrid, nao))
    for i in range(4):
        aod[i] = _dot_ao_dm(mol, ao[i], dm, mask, shls_slice, ao_loc)
    # rho_01
    # rho_01 = einsum("rgu, gu -> rg", ao[:4], aod[0])
    rho_01 = contract_multiple_rho(ao[:4], aod[0])
    rho_01[1:] *= 2
    rho_0, rho_1 = rho_01[0], rho_01[1:]
    # rho_2
    rho_2 = np.empty((3, 3, ngrid))
    # rho_2T = 2 * einsum("Tgu, gu -> Tg", ao[4:10], aod[0])
    rho_2T = contract_multiple_rho(ao[4:10], aod[0])
    for i, j, ij in zip(
            (X , X , X , Y , Y , Z ),
            ( X,  Y,  Z,  Y,  Z,  Z),
            (XX, XY, XZ, YY, YZ, ZZ)):
        # rho_2[i-1, j-1] = rho_2T[ij-4] + 2 * einsum("gu, gu -> g", ao[i], aod[j])
        rho_2[i-1, j-1] = rho_2T[ij-4] + 2 * _contract_rho(ao[i], aod[j])
        if i != j:
            rho_2[j-1, i-1] = rho_2[i-1, j-1]

    # atomic derivatives
    @timing
    def rho_atom_deriv(A):
        _, _, A0, A1 = mol.aoslice_by_atom()[A]
        sA = slice(A0, A1)
        # rho_A1 = - 2 * einsum("rgu, gu -> rg", ao[1:4, :, sA], aod[0, :, sA])
        rho_A1 = - 2 * contract_multiple_rho(ao[1:4, :, sA], aod[0, :, sA])
        rho_A2 = np.empty((3, 3, ngrid))
        # rho_A2T = - 2 * einsum("Tgu, gu -> Tg", ao[4:10, :, sA], aod[0, :, sA])
        rho_A2T = - 2 * contract_multiple_rho(ao[4:10, :, sA], aod[0, :, sA])
        for i, j, ij in zip(
            (X , X , X ,  Y, Y , Y ,  Z,  Z, Z ),
            ( X,  Y,  Z, X ,  Y,  Z, X , Y ,  Z),
            (XX, XY, XZ, XY, YY, YZ, XZ, YZ, ZZ)):
            # rho_A2[i-1, j-1] = rho_A2T[ij-4] - 2 * einsum("gu, gu -> g", ao[i, :, sA], aod[j, :, sA])
            rho_A2[i-1, j-1] = rho_A2T[ij-4] - 2 * _contract_rho(ao[i, :, sA], aod[j, :, sA])
        return rho_A1, rho_A2

    return rho_0, rho_1, rho_2, rho_atom_deriv


@timing
def get_H_1_ao(mol: gto.Mole):
    natm, nao = mol.natm, mol.nao
    int1e_ipkin = mol.intor("int1e_ipkin")
    int1e_ipnuc = mol.intor("int1e_ipnuc")
    Z_A = mol.atom_charges()
    H_1_ao = np.zeros((natm, 3, nao, nao))
    for A, (_, _, A0, A1) in enumerate(mol.aoslice_by_atom()):
        sA = slice(A0, A1)
        H_1_ao[A, :, sA, :] -= int1e_ipkin[:, sA, :]
        H_1_ao[A, :, sA, :] -= int1e_ipnuc[:, sA, :]
        with mol.with_rinv_as_nucleus(A):
            H_1_ao[A] -= Z_A[A] * mol.intor("int1e_iprinv")
    H_1_ao += H_1_ao.swapaxes(-1, -2)
    H_1_ao.shape = (natm * 3, nao, nao)
    return H_1_ao


@timing
def get_S_1_ao(mol: gto.Mole):
    natm, nao = mol.natm, mol.nao
    int1e_ipovlp = mol.intor("int1e_ipovlp")
    S_1_ao = np.zeros((natm, 3, nao, nao))
    for A, (_, _, A0, A1) in enumerate(mol.aoslice_by_atom()):
        sA = slice(A0, A1)
        S_1_ao[A, :, sA, :] = - int1e_ipovlp[:, sA, :]
    S_1_ao += S_1_ao.swapaxes(-1, -2)
    S_1_ao.shape = (natm * 3, nao, nao)
    return S_1_ao


def generator_L_1(aux):
    # derivative of cholesky lower triangular 2c2e integral
    # this involves direct inverse of 2c2e integral, so their should be no auxiliary basis dependency
    # L here does not refer to PT2 lagrangian
    L = np.linalg.cholesky(aux.intor("int2c2e"))
    L_inv = np.linalg.inv(L)
    l = np.zeros_like(L)
    for i in range(l.shape[0]):
        l[i, :i] = 1
        l[i, i] = 1 / 2
    int2c2e_1 = aux.intor("int2c2e_ip1")

    def lambda_L_1(A):
        _, _, A0a, A1a = aux.aoslice_by_atom()[A]
        m = L_inv[:, A0a:A1a] @ int2c2e_1[:, A0a:A1a] @ L_inv.T
        m += m.swapaxes(-1, -2)
        L_1 = - L @ (l * m)
        return L_1
    return L_inv, lambda_L_1


@timing
def get_gradient_jk(dfobj: df.DF, C, D, D_r, Y_mo, cx, cx_n, max_memory=2000):
    mol, aux = dfobj.mol, dfobj.auxmol
    natm, nao, nmo, nocc = mol.natm, mol.nao, C.shape[-1], mol.nelec[0]
    naux = Y_mo.shape[0]
    # this algorithm asserts naux = aux.nao, i.e. no linear dependency in auxiliary basis
    assert naux == aux.nao
    so = slice(0, nocc)

    D_r_symm = (D_r + D_r.T) / 2
    D_r_ao = C @ D_r_symm @ C.T
    D_mo = np.zeros((nmo, nmo))
    for i in range(nocc):
        D_mo[i, i] = 2

    Y_dot_D, Y_dot_D_r = np.zeros(naux), np.zeros(naux)
    for i in range(nocc):
        Y_dot_D += 2 * Y_mo[:, i, i]
    nbatch = calc_batch_size(nmo**2, max_memory)
    for saux in gen_batch(0, naux, nbatch):
        Y_dot_D_r[saux] = einsum("Ppq, pq -> P", Y_mo[saux], D_r_symm)

    Y_ip = np.asarray(Y_mo[:, so])

    L_inv, L_1_gen = generator_L_1(aux)
    int3c2e_ip1_gen = int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")
    int3c2e_ip2_gen = int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")
    C0, C1 = C[:, so], cx * C @ D_r_symm + 0.5 * cx_n * C @ D_mo

    grad_contrib = np.zeros((natm, 3))
    for A in range(natm):
        shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
        shA0a, shA1a, _, _ = aux.aoslice_by_atom()[A]

        Y_1_dot_D = np.zeros((3, naux))
        Y_1_dot_D_r = np.zeros((3, naux))
        Y_1_mo_D_r = np.zeros((3, naux, nocc, nmo))

        nbatch = calc_batch_size(3*(nao+nocc)*naux, max_memory, Y_1_mo_D_r.size + Y_ip.size)
        for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
            su = slice(U0, U1)
            int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux.nbas))
            Y_1_mo_D_r -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip1, L_inv, C0[su], C1)
            Y_1_mo_D_r -= einsum("tuvQ, PQ, up, vi -> tPip", int3c2e_ip1, L_inv, C1[su], C0)
            Y_1_dot_D -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D[su])
            Y_1_dot_D_r -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D_r_ao[su])

        nbatch = calc_batch_size(3*nao*(nao+nocc), max_memory, Y_1_mo_D_r.size + Y_ip.size)
        for shP0, shP1, P0, P1 in gen_shl_batch(aux, nbatch, shA0a, shA1a):
            sp = slice(P0, P1)
            int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
            Y_1_mo_D_r -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip2, L_inv[:, sp], C0, C1)
            Y_1_dot_D -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D)
            Y_1_dot_D_r -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D_r_ao)

        L_1 = L_1_gen(A)
        L_1_dot_inv = einsum("tRQ, PR -> tPQ", L_1, L_inv)
        Y_1_mo_D_r -= einsum("Qiq, qp, tPQ -> tPip", Y_ip, cx * D_r_symm + 0.5 * cx_n * D_mo, L_1_dot_inv)
        Y_1_dot_D -= einsum("Q, tPQ -> tP", Y_dot_D, L_1_dot_inv)
        Y_1_dot_D_r -= einsum("Q, tPQ -> tP", Y_dot_D_r, L_1_dot_inv)

        grad_contrib[A] = (
            + einsum("P, tP -> t", Y_dot_D, Y_1_dot_D_r)
            + einsum("P, tP -> t", Y_dot_D_r, Y_1_dot_D)
            + einsum("P, tP -> t", Y_dot_D, Y_1_dot_D)
            - 2 * einsum("Pip, tPip -> t", Y_ip, Y_1_mo_D_r))

    return grad_contrib


@timing
def get_gradient_gga(C, D_r, xc_setting, xc_kernel, vxc_n=None, max_memory=2000):
    # reference HF
    if xc_kernel[1] is None:
        return get_gradient_gga_hfref(xc_setting, vxc_n, max_memory)

    ni, mol, grids, xc, D = xc_setting
    rho, vxc, fxc = xc_kernel
    natm, nao = mol.natm, mol.nao

    D_r_symm = (D_r + D_r.T) / 2
    D_r_ao = C @ D_r_symm @ C.T

    grad_contrib = np.zeros((natm, 3))
    ig = 0
    for ao, mask, weight, _ in ni.block_loop(mol, grids, nao, deriv=2, max_memory=max_memory):
        rho_0, rho_1, rho_2, get_rho_A = get_rho_derivs(ao, D, mol, mask)
        rho_X_0, rho_X_1, rho_X_2, get_rho_X_A = get_rho_derivs(ao, D_r_ao, mol, mask)
        gamma_XD = 2 * einsum("rg, rg -> g", rho_X_1, rho_1)
        sg = slice(ig, ig + weight.size)
        fr, fg = vxc[0][sg] * weight, vxc[1][sg] * weight
        frr, frg, fgg = fxc[0][sg] * weight, fxc[1][sg] * weight, fxc[2][sg] * weight
        if vxc_n is None:
            fr_n, fg_n = fr, fg
        else:
            fr_n, fg_n = vxc_n[0][sg] * weight, vxc_n[1][sg] * weight

        for A in range(natm):
            rho_A1, rho_A2 = get_rho_A(A)
            rho_X_A1, rho_X_A2 = get_rho_X_A(A)
            gamma_A1 = 2 * einsum("rg, trg -> tg", rho_1, rho_A2)
            grad_contrib[A] += (
                    + einsum("g, tg, g -> t", frr, rho_A1, rho_X_0)
                    + einsum("g, tg, g -> t", frg, rho_A1, gamma_XD)
                    + einsum("g, tg, g -> t", frg, gamma_A1, rho_X_0)
                    + einsum("g, tg, g -> t", fgg, gamma_A1, gamma_XD)
                    + einsum("g, tg -> t", fr, rho_X_A1)
                    + 2 * einsum("g, trg, rg -> t", fg, rho_A2, rho_X_1)
                    + 2 * einsum("g, rg, trg -> t", fg, rho_1, rho_X_A2)
                    + einsum("g, tg -> t", fr_n, rho_A1)
                    + einsum("g, tg -> t", fg_n, gamma_A1))
        ig += weight.size
    return grad_contrib


@timing
def get_gradient_gga_hfref(xc_setting, vxc_n, max_memory=2000):
    ni, mol, grids, xc, D = xc_setting
    natm, nao = mol.natm, mol.nao

    grad_contrib = np.zeros((natm, 3))
    ig = 0
    for ao, mask, weight, _ in ni.block_loop(mol, grids, nao, deriv=2, max_memory=max_memory):
        rho_0, rho_1, rho_2, get_rho_A = get_rho_derivs(ao, D, mol, mask)
        sg = slice(ig, ig + weight.size)
        fr_n, fg_n = vxc_n[0][sg] * weight, vxc_n[1][sg] * weight

        for A in range(natm):
            rho_A1, rho_A2 = get_rho_A(A)
            gamma_A1 = 2 * einsum("rg, trg -> tg", rho_1, rho_A2)
            grad_contrib[A] += (
                    + einsum("g, tg -> t", fr_n, rho_A1)
                    + einsum("g, tg -> t", fg_n, gamma_A1))
        ig += weight.size
    return grad_contrib


class Gradients(RDFDH):

    def __init__(self, mol: gto.Mole, skip_construct=False, *args, **kwargs):
        if not skip_construct:
            super(Gradients, self).__init__(mol, *args, **kwargs)
        # results
        self.grad_jk = NotImplemented
        self.grad_gga = NotImplemented
        self.grad_pt2 = NotImplemented
        self.grad_enfunc = NotImplemented
        self.grad_tot = NotImplemented
        self.de = NotImplemented

    def prepare_H_1(self):
        H_1_ao = get_H_1_ao(self.mol)
        H_1_mo = self.C.T @ H_1_ao @ self.C
        self.tensors.create("H_1_ao", H_1_ao)
        self.tensors.create("H_1_mo", H_1_mo)

    def prepare_S_1(self):
        S_1_ao = get_S_1_ao(self.mol)
        S_1_mo = self.C.T @ S_1_ao @ self.C
        self.tensors.create("S_1_ao", S_1_ao)
        self.tensors.create("S_1_mo", S_1_mo)

    def prepare_gradient_jk(self):
        D_r = self.tensors.load("D_r")
        Y_mo = self.tensors["Y_mo_jk"]
        # a special treatment
        cx_n = self.cx_n if self.xc_n else self.cx
        self.grad_jk = get_gradient_jk(self.df_jk, self.C, self.D, D_r, Y_mo, self.cx, cx_n, self.get_memory())

    def prepare_gradient_gga(self):
        # assert prepare_xc_kernel has been called
        tensors = self.tensors
        D_r = tensors.load("D_r")
        xc_setting = self.mf_s._numint, self.mol, self.grids, self.xc, self.D
        if "rho" not in tensors:
            self.grad_gga = 0
            return
        rho = tensors["rho"]
        if self.ni._xc_type(self.xc) == "GGA":
            vxc, fxc = tensors["vxc" + self.xc], tensors["fxc" + self.xc]
        else:
            vxc, fxc = None, None
        xc_kernel = rho, vxc, fxc
        vxc_n = None
        if self.xc_n:
            vxc_n = self.tensors.get("vxc" + self.xc_n, None)
            if vxc_n is None and self.ni._xc_type(self.xc_n) == "HF":
                vxc_n = np.zeros((2, rho.size))
        self.grad_gga = get_gradient_gga(self.C, D_r, xc_setting, xc_kernel, vxc_n, self.get_memory())

    @timing
    def prepare_gradient_pt2(self):
        tensors = self.tensors
        C, D, e = self.C, self.D, self.e
        mol, aux_ri = self.mol, self.aux_ri
        natm, nao, nmo, nocc, nvir, naux = mol.natm, self.nao, self.nmo, self.nocc, self.nvir, self.df_ri.get_naoaux()
        # this algorithm asserts naux = aux.nao, i.e. no linear dependency in auxiliary basis
        assert naux == aux_ri.nao
        so, sv, sa = self.so, self.sv, self.sa

        D_r = tensors.load("D_r")
        H_1_mo = tensors.load("H_1_mo")
        grad_corr = einsum("pq, Apq -> A", D_r, H_1_mo)

        W_I = tensors.load("W_I")
        W_II = - einsum("pq, q -> pq", D_r, e)
        W_III = np.zeros((nmo, nmo))
        W_III[so, so] = - 0.5 * self.Ax0_Core(so, so, sa, sa)(D_r)
        W = W_I + W_II + W_III
        W_ao = C @ W @ C.T
        S_1_ao = tensors.load("S_1_ao")
        grad_corr += einsum("uv, Auv -> A", W_ao, S_1_ao)

        # generate L_1_ri
        L_inv, L_1_gen = generator_L_1(aux_ri)

        # generate Y_1_ia_ri
        int3c2e_ip1_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip1", "s1")
        int3c2e_ip2_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip2", "s1")
        Y_ia_ri = np.asarray(tensors["Y_mo_ri"][:, so, sv])

        def lambda_Y_1_ia_ri(A):
            L_1_ri = L_1_gen(A)
            Y_1_ia_ri = np.zeros((3, naux, nocc, nvir))
            shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
            shA0a, shA1a, _, _ = aux_ri.aoslice_by_atom()[A]

            nbatch = calc_batch_size(3*(nao+nocc)*naux, self.get_memory(), Y_1_ia_ri.size)
            for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
                su = slice(U0, U1)
                int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux_ri.nbas))
                Y_1_ia_ri -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip1, L_inv, C[su, so], C[:, sv])
                Y_1_ia_ri -= einsum("tuvQ, PQ, ua, vi -> tPia", int3c2e_ip1, L_inv, C[su, sv], C[:, so])

            nbatch = calc_batch_size(3*nao*(nao+nocc), self.get_memory(), Y_1_ia_ri.size)
            for shP0, shP1, P0, P1 in gen_shl_batch(aux_ri, nbatch, shA0a, shA1a):
                sp = slice(P0, P1)
                int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
                Y_1_ia_ri -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip2, L_inv[:, sp], C[:, so], C[:, sv])

            Y_1_ia_ri -= einsum("Qia, tRQ, PR -> tPia", Y_ia_ri, L_1_ri, L_inv)
            return Y_1_ia_ri

        # final contribution from G_ia_ri
        # 4 * einsum("iaP, AiaP -> A", G_ia_ri, gradh.Y_mo_1_ri[:, so, sv]))
        G_ia_ri = tensors.load("G_ia_ri")
        for A in range(natm):
            grad_corr[3*A:3*A+3] += 4 * einsum("Pia, tPia -> t", G_ia_ri, lambda_Y_1_ia_ri(A))
        grad_corr.shape = (natm, 3)

        self.grad_pt2 = grad_corr

    @timing
    def prepare_gradient_enfunc(self):
        tensors = self.tensors
        natm = self.mol.natm
        Co, eo, D = self.Co, self.eo, self.D
        so = self.so

        grad_contrib = self.mf_s.Gradients().grad_nuc()
        grad_contrib.shape = (natm * 3,)

        H_1_ao = tensors.load("H_1_ao")
        S_1_mo = tensors.load("S_1_mo")

        grad_contrib += einsum("Auv, uv -> A", H_1_ao, D)
        if self.xc_n is None:
            grad_contrib -= 2 * np.einsum("Ai, i -> A", S_1_mo[:, so, so].diagonal(0, -1, -2), eo)
        else:
            nc_F_0_ij = einsum("ui, uv, vj -> ij", Co, self.mf_n.get_fock(dm=D), Co)
            grad_contrib -= 2 * einsum("Aij, ij -> A", S_1_mo[:, so, so], nc_F_0_ij)
        grad_contrib.shape = (natm, 3)

        self.grad_enfunc = grad_contrib

    kernel = kernel

