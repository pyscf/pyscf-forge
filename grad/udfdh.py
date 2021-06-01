from __future__ import annotations
# dh import
try:
    from dh.udfdh import UDFDH
    from dh.dhutil import calc_batch_size, gen_batch, gen_shl_batch, tot_size, timing
    from dh.grad.rdfdh import get_H_1_ao, get_S_1_ao, generator_L_1
    from dh.grad.rdfdh import Gradients as RGradients
except ImportError:
    from pyscf.dh.udfdh import UDFDH
    from pyscf.dh.dhutil import calc_batch_size, gen_batch, gen_shl_batch, tot_size, timing
    from pyscf.dh.grad.rdfdh import get_H_1_ao, get_S_1_ao, generator_L_1
    from pyscf.dh.grad.rdfdh import Gradients as RGradients
# pyscf import
from pyscf import gto, lib, df
from pyscf.df.grad.rhf import _int3c_wrapper as int3c_wrapper
from pyscf.dftd3 import itrf
# other import
import numpy as np
import itertools
import ctypes

einsum = lib.einsum
α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


@timing
def get_gradient_jk(dfobj: df.DF, C, D, D_r, Y_mo, cx, cx_n, max_memory=2000):
    mol, aux = dfobj.mol, dfobj.auxmol
    natm, nao, nmo, nocc = mol.natm, mol.nao, C.shape[-1], mol.nelec
    mocc = max(nocc)
    naux = Y_mo[0].shape[0]
    # this algorithm asserts naux = aux.nao, i.e. no linear dependency in auxiliary basis
    assert naux == aux.nao
    so = slice(0, nocc[α]), slice(0, nocc[β])

    D_r_symm = (D_r + D_r.swapaxes(-1, -2)) / 2
    D_r_ao = einsum("sup, spq, svq -> suv", C, D_r_symm, C)
    D_mo = np.zeros((2, nmo, nmo))
    for σ in (α, β):
        for i in range(nocc[σ]):
            D_mo[σ, i, i] = 1

    Y_dot_D, Y_dot_D_r = np.zeros((2, naux)), np.zeros((2, naux))
    nbatch = calc_batch_size(nmo**2, max_memory)
    for σ in (α, β):
        for i in range(nocc[σ]):
            Y_dot_D[σ] += Y_mo[σ][:, i, i]
        for saux in gen_batch(0, naux, nbatch):
            Y_dot_D_r[σ][saux] = einsum("Ppq, pq -> P", Y_mo[σ][saux], D_r_symm[σ])

    Y_ip = [np.asarray(Y_mo[σ][:, so[σ]]) for σ in (α, β)]
    L_inv, L_1_gen = generator_L_1(aux)
    int3c2e_ip1_gen = int3c_wrapper(mol, aux, "int3c2e_ip1", "s1")
    int3c2e_ip2_gen = int3c_wrapper(mol, aux, "int3c2e_ip2", "s1")
    C0 = [C[σ][:, so[σ]] for σ in (α, β)]
    D1 = [cx * D_r_symm[σ] + 0.5 * cx_n * D_mo[σ] for σ in (α, β)]
    C1 = [C[σ] @ D1[σ] for σ in (α, β)]

    grad_contrib = np.zeros((natm, 3))
    for A in range(natm):
        shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
        shA0a, shA1a, _, _ = aux.aoslice_by_atom()[A]

        Y_1_mo_D_r = [np.zeros((3, naux, nocc[σ], nmo)) for σ in (α, β)]
        Y_1_dot_D, Y_1_dot_D_r = np.zeros((2, 3, naux)), np.zeros((2, 3, naux))

        pre_flop = tot_size(Y_1_mo_D_r, Y_ip, Y_1_dot_D, Y_1_dot_D_r)
        nbatch = calc_batch_size(3*(nao+mocc)*naux, max_memory, pre_flop)
        for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
            su = slice(U0, U1)
            int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux.nbas))
            for σ in (α, β):
                Y_1_mo_D_r[σ] -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip1, L_inv, C0[σ][su], C1[σ])
                Y_1_mo_D_r[σ] -= einsum("tuvQ, PQ, up, vi -> tPip", int3c2e_ip1, L_inv, C1[σ][su], C0[σ])
                Y_1_dot_D[σ] -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D[σ][su])
                Y_1_dot_D_r[σ] -= 2 * einsum("tuvQ, PQ, uv -> tP", int3c2e_ip1, L_inv, D_r_ao[σ][su])

        nbatch = calc_batch_size(3*nao*(nao+mocc), max_memory, pre_flop)
        for shP0, shP1, P0, P1 in gen_shl_batch(aux, nbatch, shA0a, shA1a):
            sp = slice(P0, P1)
            int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
            for σ in (α, β):
                Y_1_mo_D_r[σ] -= einsum("tuvQ, PQ, ui, vp -> tPip", int3c2e_ip2, L_inv[:, sp], C0[σ], C1[σ])
                Y_1_dot_D[σ] -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D[σ])
                Y_1_dot_D_r[σ] -= einsum("tuvQ, PQ, uv -> tP", int3c2e_ip2, L_inv[:, sp], D_r_ao[σ])

        L_1 = L_1_gen(A)
        L_1_dot_inv = einsum("tRQ, PR -> tPQ", L_1, L_inv)
        for σ in (α, β):
            Y_1_mo_D_r[σ] -= einsum("Qiq, qp, tPQ -> tPip", Y_ip[σ], D1[σ], L_1_dot_inv)
            Y_1_dot_D[σ] -= einsum("Q, tPQ -> tP", Y_dot_D[σ], L_1_dot_inv)
            Y_1_dot_D_r[σ] -= einsum("Q, tPQ -> tP", Y_dot_D_r[σ], L_1_dot_inv)
            # RI-K contribution
            grad_contrib[A] += - 2 * einsum("Pip, tPip -> t", Y_ip[σ], Y_1_mo_D_r[σ])

        # RI-J contribution
        for σ, ς in itertools.product((α, β), (α, β)):
            grad_contrib[A] += (
                + einsum("P, tP -> t", Y_dot_D[σ], Y_1_dot_D_r[ς])
                + einsum("P, tP -> t", Y_dot_D_r[σ], Y_1_dot_D[ς])
                + einsum("P, tP -> t", Y_dot_D[σ], Y_1_dot_D[ς]))
    return grad_contrib


class Gradients(UDFDH, RGradients):

    def __init__(self, mol: gto.Mole, *args, skip_construct=False, **kwargs):
        if not skip_construct:
            super(Gradients, self).__init__(mol, *args, **kwargs)
        # results
        self.grad_jk = NotImplemented
        self.grad_gga = NotImplemented
        self.grad_pt2 = NotImplemented
        self.grad_enfunc = NotImplemented
        self.grad_tot = NotImplemented
        self.de = NotImplemented

    @timing
    def prepare_H_1(self):
        H_1_ao = get_H_1_ao(self.mol)
        H_1_mo = np.array([einsum("up, Auv, vq -> Apq", self.C[σ], H_1_ao, self.C[σ]) for σ in (α, β)])
        self.tensors.create("H_1_ao", H_1_ao)
        self.tensors.create("H_1_mo", H_1_mo)

    @timing
    def prepare_S_1(self):
        S_1_ao = get_S_1_ao(self.mol)
        S_1_mo = np.array([einsum("up, Auv, vq -> Apq", self.C[σ], S_1_ao, self.C[σ]) for σ in (α, β)])
        self.tensors.create("S_1_ao", S_1_ao)
        self.tensors.create("S_1_mo", S_1_mo)

    def prepare_gradient_jk(self):
        D_r = self.tensors.load("D_r")
        Y_mo = [self.tensors["Y_mo_jk" + str(σ)] for σ in (α, β)]
        # a special treatment
        cx_n = self.cx_n if self.xc_n else self.cx
        self.grad_jk = get_gradient_jk(self.df_jk, self.C, self.D, D_r, Y_mo, self.cx, cx_n, self.get_memory())

    @timing
    def prepare_gradient_gga(self):
        tensors = self.tensors
        if "rho" not in tensors:
            self.grad_gga = 0
            return self
        # --- LAZY CODE ---
        from pyscf import grad, hessian
        ni, mol, grids = self.ni, self.mol, self.grids
        natm = mol.natm
        C, D = self.C, self.D
        grad_contrib = np.zeros((natm, 3))

        xc = self.xc_n if self.xc_n else self.xc
        if self.ni._xc_type(xc) == "GGA":  # energy functional contribution
            veff_1_gga = grad.uks.get_vxc(ni, mol, grids, xc, D)[1]
            for A, (_, _, A0, A1) in enumerate(mol.aoslice_by_atom()):
                grad_contrib[A] += 2 * einsum("stuv, suv -> t", veff_1_gga[:, :, A0:A1], D[:, A0:A1])

        if self.ni._xc_type(self.xc) == "GGA":  # reference functional skeleton fock derivative contribution
            D_r = tensors.load("D_r")
            D_r_symm = (D_r + D_r.swapaxes(-1, -2)) / 2
            D_r_ao = einsum("sup, spq, svq -> suv", C, D_r_symm, C)

            F_1_ao_dfa = np.array(hessian.uks._get_vxc_deriv1(self.mf_s.Hessian(), C, self.mo_occ, 2000))
            grad_contrib += einsum("suv, sAtuv -> At", D_r_ao, F_1_ao_dfa)

        self.grad_gga = grad_contrib
        return self

    @timing
    def prepare_gradient_pt2(self):
        tensors = self.tensors
        C, D, e = self.C, self.D, self.e
        mol, aux_ri = self.mol, self.aux_ri
        natm, nao, nmo, nocc, nvir, naux = mol.natm, self.nao, self.nmo, self.nocc, self.nvir, self.df_ri.get_naoaux()
        mocc, mvir = max(nocc), max(nvir)
        # this algorithm asserts naux = aux.nao, i.e. no linear dependency in auxiliary basis
        assert naux == aux_ri.nao
        so, sv, sa = self.so, self.sv, self.sa

        D_r = tensors.load("D_r")
        H_1_mo = tensors.load("H_1_mo")
        grad_corr = einsum("spq, sApq -> A", D_r, H_1_mo)
        if not self.eval_pt2:
            grad_corr.shape = (natm, 3)
            self.grad_pt2 = grad_corr
            return

        W_I = tensors.load("W_I")
        W_II = - einsum("spq, sq -> spq", D_r, e)
        W_III_tmp = self.Ax0_Core(so, so, sa, sa)(D_r)
        W = W_I + W_II
        for σ in (α, β):
            W[σ][so[σ], so[σ]] += - 0.5 * W_III_tmp[σ]
        W_ao = einsum("sup, spq, svq -> suv", C, W, C)
        S_1_ao = tensors.load("S_1_ao")
        grad_corr += np.einsum("suv, Auv -> A", W_ao, S_1_ao)
        grad_corr.shape = (natm, 3)

        L_inv, L_1_gen = generator_L_1(aux_ri)
        int3c2e_ip1_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip1", "s1")
        int3c2e_ip2_gen = int3c_wrapper(mol, aux_ri, "int3c2e_ip2", "s1")
        Y_ia_ri = [np.asarray(tensors["Y_mo_ri" + str(σ)][:, so[σ], sv[σ]]) for σ in (α, β)]
        G_ia_ri = [tensors.load("G_ia_ri" + str(σ)) for σ in (α, β)]

        for A in range(natm):
            L_1_ri = L_1_gen(A)
            Y_1_ia_ri = [np.zeros((3, naux, nocc[σ], nvir[σ])) for σ in (α, β)]
            shA0, shA1, _, _ = mol.aoslice_by_atom()[A]
            shA0a, shA1a, _, _ = aux_ri.aoslice_by_atom()[A]

            nbatch = calc_batch_size(3*(nao+mocc)*naux, self.get_memory(), tot_size(Y_1_ia_ri))
            for shU0, shU1, U0, U1 in gen_shl_batch(mol, nbatch, shA0, shA1):
                su = slice(U0, U1)
                int3c2e_ip1 = int3c2e_ip1_gen((shU0, shU1, 0, mol.nbas, 0, aux_ri.nbas))
                for σ in (α, β):
                    Y_1_ia_ri[σ] -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip1, L_inv, C[σ][su, so[σ]], C[σ][:, sv[σ]])
                    Y_1_ia_ri[σ] -= einsum("tuvQ, PQ, ua, vi -> tPia", int3c2e_ip1, L_inv, C[σ][su, sv[σ]], C[σ][:, so[σ]])

            nbatch = calc_batch_size(3*nao*(nao+mocc), self.get_memory(), tot_size(Y_1_ia_ri))
            for shP0, shP1, P0, P1 in gen_shl_batch(aux_ri, nbatch, shA0a, shA1a):
                sp = slice(P0, P1)
                int3c2e_ip2 = int3c2e_ip2_gen((0, mol.nbas, 0, mol.nbas, shP0, shP1))
                for σ in (α, β):
                    Y_1_ia_ri[σ] -= einsum("tuvQ, PQ, ui, va -> tPia", int3c2e_ip2, L_inv[:, sp], C[σ][:, so[σ]], C[σ][:, sv[σ]])

            for σ in (α, β):
                Y_1_ia_ri[σ] -= einsum("Qia, tRQ, PR -> tPia", Y_ia_ri[σ], L_1_ri, L_inv)
                grad_corr[A] += einsum("Pia, tPia -> t", G_ia_ri[σ], Y_1_ia_ri[σ])
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

        grad_contrib += np.einsum("Auv, suv -> A", H_1_ao, D, optimize=True)  # TODO check PySCF lib.einsum why fails
        if self.xc_n is None:
            for σ in (α, β):
                grad_contrib -= np.einsum("Ai, i -> A", S_1_mo[σ][:, so[σ], so[σ]].diagonal(0, -1, -2), eo[σ])
        else:
            # TODO see whether get_fock could use mo_coeff to accelearate RI-K
            F_0_ao_n = self.mf_n.get_fock(dm=D)
            nc_F_0_ij = [(Co[σ].T @ F_0_ao_n[σ] @ Co[σ]) for σ in (α, β)]
            for σ in (α, β):
                grad_contrib -= einsum("Aij, ij -> A", S_1_mo[σ][:, so[σ], so[σ]], nc_F_0_ij[σ])
        grad_contrib.shape = (natm, 3)

        # handle dftd3 situation
        mol = self.mol
        if "D3" in self.xc_add:
            drv = itrf.libdftd3.wrapper_params
            params = np.asarray(self.xc_add["D3"][0], order="F")
            version = self.xc_add["D3"][1]
            coords = np.asarray(mol.atom_coords(), order="F")
            itype = np.asarray(mol.atom_charges(), order="F")
            edisp = np.zeros(1)
            grad = np.zeros((mol.natm, 3))
            drv(
                ctypes.c_int(mol.natm),  # natoms
                coords.ctypes.data_as(ctypes.c_void_p),  # coords
                itype.ctypes.data_as(ctypes.c_void_p),  # itype
                params.ctypes.data_as(ctypes.c_void_p),  # params
                ctypes.c_int(version),  # version
                edisp.ctypes.data_as(ctypes.c_void_p),  # edisp
                grad.ctypes.data_as(ctypes.c_void_p))  # grads)
            grad_contrib += grad

        self.grad_enfunc = grad_contrib

    def base_method(self) -> UDFDH:
        self.__class__ = UDFDH
        return self

