import numpy, scipy
from pyscf import df, dft, pbc, scf
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.lib import logger, current_memory
from .rpprpa_davidson import RppRPADavidson, pprpa_get_trial_vector, pprpa_expand_space
from .rpprpa_direct import pprpa_print_a_pair, inner_product, pprpa_orthonormalize_eigenvector



def kernel(pprpa):
    # initialize trial vector and product matrix
    data_type = pprpa._scf.mo_coeff.dtype
    tri_vec = numpy.zeros(shape=[pprpa.max_vec, pprpa.full_dim], dtype=data_type)
    tri_vec_sig = numpy.zeros(shape=[pprpa.max_vec], dtype=data_type)
    if pprpa.channel == "pp":
        ntri = min(pprpa.nroot * 4, pprpa.vv_dim)
    else:
        ntri = min(pprpa.nroot * 4, pprpa.oo_dim)
    tri_vec[:ntri], tri_vec_sig[:ntri] = pprpa_get_trial_vector(pprpa, ntri)
    mv_prod = numpy.zeros_like(tri_vec)

    iter = 0
    nprod = 0  # number of contracted vectors
    while iter < pprpa.max_iter:
        logger.info(
            pprpa, "\nppRPA Davidson %d-th iteration, ntri= %d , nprod= %d .",
            iter + 1, ntri, nprod)
        mv_prod[nprod:ntri] = pprpa_contraction(pprpa, tri_vec[nprod:ntri])
        nprod = ntri

        # get ppRPA matrix and metric matrix in subspace
        m_tilde = numpy.dot(tri_vec[:ntri].conj(), mv_prod[:ntri].T)
        w_tilde = numpy.zeros_like(m_tilde)
        for i in range(ntri):
            if inner_product(tri_vec[i].conj(), tri_vec[i], pprpa.oo_dim).real > 0:
                w_tilde[i, i] = 1
            else:
                w_tilde[i, i] = -1

        # diagonalize subspace matrix
        if data_type == numpy.double:
            alphar, _, beta, _, v_tri, _, _ = scipy.linalg.lapack.dggev(
                m_tilde, w_tilde, compute_vl=0)
        elif data_type == numpy.complex128:
            alphar, beta, _, v_tri, _, _ = scipy.linalg.lapack.zggev(
                m_tilde, w_tilde, compute_vl=0)
        e_tri = (alphar / beta).real
        v_tri = v_tri.T  # Fortran matrix to Python order

        if pprpa.channel == "pp":
            # sort eigenvalues and eigenvectors by ascending order
            idx = e_tri.argsort()
            e_tri = e_tri[idx]
            v_tri = v_tri[idx, :]

            # re-order all states by signs, first hh then pp
            sig = numpy.zeros(shape=[ntri], dtype=int)
            for i in range(ntri):
                if numpy.sum(v_tri[i].conj() * tri_vec_sig[:ntri] * v_tri[i]).real > 0:
                    sig[i] = 1
                else:
                    sig[i] = -1

            hh_index = numpy.where(sig < 0)[0]
            pp_index = numpy.where(sig > 0)[0]
            e_tri_hh = e_tri[hh_index]
            e_tri_pp = e_tri[pp_index]
            e_tri[:len(hh_index)] = e_tri_hh
            e_tri[len(hh_index):] = e_tri_pp
            v_tri_hh = v_tri[hh_index]
            v_tri_pp = v_tri[pp_index]
            v_tri[:len(hh_index)] = v_tri_hh
            v_tri[len(hh_index):] = v_tri_pp

            # get only two-electron addition energy
            first_state=len(hh_index)
            pprpa.exci = e_tri[first_state:first_state+pprpa.nroot]
        else:
            # sort eigenvalues and eigenvectors by descending order
            idx = e_tri.argsort()[::-1]
            e_tri = e_tri[idx]
            v_tri = v_tri[idx, :]

            # re-order all states by signs, first pp then hh
            sig = numpy.zeros(shape=[ntri], dtype=int)
            for i in range(ntri):
                if numpy.sum(v_tri[i].conj() * tri_vec_sig[:ntri] * v_tri[i]).real > 0:
                    sig[i] = 1
                else:
                    sig[i] = -1

            hh_index = numpy.where(sig < 0)[0]
            pp_index = numpy.where(sig > 0)[0]
            e_tri_hh = e_tri[hh_index]
            e_tri_pp = e_tri[pp_index]
            e_tri[:len(pp_index)] = e_tri_pp
            e_tri[len(pp_index):] = e_tri_hh
            v_tri_hh = v_tri[hh_index]
            v_tri_pp = v_tri[pp_index]
            v_tri[:len(pp_index)] = v_tri_pp
            v_tri[len(pp_index):] = v_tri_hh

            # get only two-electron removal energy
            first_state=len(pp_index)
            pprpa.exci = e_tri[first_state:first_state+pprpa.nroot]

        ntri_old = ntri
        conv, ntri = pprpa_expand_space(
            pprpa=pprpa, first_state=first_state, tri_vec=tri_vec,
            tri_vec_sig=tri_vec_sig, mv_prod=mv_prod, v_tri=v_tri)
        logger.info(pprpa, "add %d new trial vectors.", ntri - ntri_old)

        iter += 1
        if conv is True:
            break

    assert conv is True, "ppRPA Davidson is not converged!"
    logger.info(
        pprpa, "\nppRPA Davidson converged in %d iterations, final subspace size = %d",
        iter, nprod)

    pprpa_orthonormalize_eigenvector(pprpa.multi, pprpa.nocc_act, pprpa.exci, pprpa.xy)

    return


def pprpa_contraction(pprpa, tri_vec):
    """GppRPA contraction.

    Args:
        pprpa (GppRPA_Davidson): GppRPA_Davidson object.
        tri_vec (double/complex ndarray): trial vector.

    Returns:
        mv_prod (double/complex ndarray): product between ppRPA matrix and trial vectors.
    """
    # for convenience, use XX as XX_act in this function
    nocc, nvir, nmo = pprpa.nocc_act, pprpa.nvir_act, pprpa.nmo_act
    mo_energy = pprpa.mo_energy_act
    Lpi = pprpa.Lpi
    Lpa = pprpa.Lpa
    naux = Lpi.shape[0]

    ntri = tri_vec.shape[0]
    mv_prod = numpy.zeros(shape=[ntri, pprpa.full_dim], dtype=tri_vec.dtype)

    tri_row_o, tri_col_o = numpy.tril_indices(nocc, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir, -1)

    for ivec in range(ntri):
        z_oo = numpy.zeros(shape=[nocc, nocc], dtype=tri_vec.dtype)
        z_vv = numpy.zeros(shape=[nvir, nvir], dtype=tri_vec.dtype)
        z_oo[tri_row_o, tri_col_o] = tri_vec[ivec][:pprpa.oo_dim]
        z_oo[numpy.diag_indices(nocc)] *= 1.0 / numpy.sqrt(2)
        z_oo = numpy.ascontiguousarray(z_oo.T)
        z_vv[tri_row_v, tri_col_v] = tri_vec[ivec][pprpa.oo_dim:]
        z_vv[numpy.diag_indices(nvir)] *= 1.0 / numpy.sqrt(2)
        z_vv = numpy.ascontiguousarray(z_vv.T)

        # Lpqz_{L,pr} = \sum_s Lpq_{L,ps} z_{rs}
        Lpq_z = numpy.zeros(shape=[naux * nmo, nmo], dtype=tri_vec.dtype)
        Lpq_z[:, :nocc] = Lpi.reshape(naux * nmo, nocc) @ z_oo
        Lpq_z[:, nocc:] = Lpa.reshape(naux * nmo, nvir) @ z_vv

        # transpose and reshape for faster multiplication
        Lpq_z = Lpq_z.reshape(naux, nmo, nmo).transpose(1, 0, 2).conj()
        Lpq_z = Lpq_z.reshape(nmo, naux * nmo)

        # MV_{pq} = \sum_{Lr} Lpq_{L,pr} Lpqz_{L,qr} - Lpq_{L,qr} Lpqz_{L,pr}
        # -MV_{qp}* = - \sum_{Lr} Lpq_{L,rp} Lpqz_{L,qr}^* + Lpq_{L,rq} Lpqz_{L,pr}^*
        mv_prod_full = numpy.zeros(shape=[nmo, nmo], dtype=tri_vec.dtype)
        mv_prod_full[:nocc, :nocc] = Lpq_z[:nocc] @ Lpi.reshape(naux * nmo, nocc)
        mv_prod_full[nocc:, nocc:] = Lpq_z[nocc:] @ Lpa.reshape(naux * nmo, nvir)
        mv_prod_full -= mv_prod_full.T
        mv_prod_full[numpy.diag_indices(nmo)] *= 1.0 / numpy.sqrt(2)
        mv_prod[ivec][: pprpa.oo_dim] =\
            -mv_prod_full[:nocc, :nocc][tri_row_o, tri_col_o].conj()
        mv_prod[ivec][pprpa.oo_dim:] = \
            -mv_prod_full[nocc:, nocc:][tri_row_v, tri_col_v].conj()

    orb_sum_oo = (mo_energy[None, : nocc] + mo_energy[: nocc, None]) - 2.0 * pprpa.mu
    orb_sum_oo = orb_sum_oo[tri_row_o, tri_col_o]
    orb_sum_vv = (mo_energy[None, nocc:] + mo_energy[nocc:, None]) - 2.0 * pprpa.mu
    orb_sum_vv = orb_sum_vv[tri_row_v, tri_col_v]
    for ivec in range(ntri):
        oz_oo = -orb_sum_oo * tri_vec[ivec][:pprpa.oo_dim]
        mv_prod[ivec][:pprpa.oo_dim] += oz_oo
        oz_vv = orb_sum_vv * tri_vec[ivec][pprpa.oo_dim:]
        mv_prod[ivec][pprpa.oo_dim:] += oz_vv

    return mv_prod


def ao2mo(pprpa, full_mo=False):
    """Get three-center density-fitting matrix in MO active space.

    Args:
        pprpa: GppRPA object.

    Returns:
        Lpq (complex/double ndarray): three-center DF matrix in MO active space.
    """
    mf = pprpa._scf
    mo_coeff = mf.mo_coeff
    nocc = pprpa.nocc

    nao = pprpa._scf.mol.nao_nr()
    mo = numpy.asarray(mo_coeff, order='F')
    if full_mo:
        nocc_act, nvir_act, nmo_act = nocc, mo.shape[1] - nocc, mo.shape[1]
        ijslice = (0, mo.shape[1], 0, mo.shape[1])
    else:
        nocc_act, nvir_act, nmo_act = pprpa.nocc_act, pprpa.nvir_act, pprpa.nmo_act
        ijslice = (nocc-nocc_act, nocc+nvir_act, nocc-nocc_act, nocc+nvir_act)

    if isinstance(mf, (scf.ghf.GHF, dft.gks.GKS)):
        # molecule
        if getattr(mf, 'with_df', None):
            pprpa.with_df = mf.with_df
        else:
            pprpa.with_df = df.DF(mf.mol)
            if pprpa.auxbasis is not None:
                pprpa.with_df.auxbasis = pprpa.auxbasis
            else:
                pprpa.with_df.auxbasis = df.make_auxbasis(
                    mf.mol, mp2fit=True)
        pprpa._keys.update(['with_df'])

        naux = pprpa.with_df.get_naoaux()
        mem_incore = (2*nao**2*naux) * 8/1e6
        mem_incore += (nmo_act**2*naux) * 16/1e6
        mem_now = current_memory()[0]

        mo_a = mf.mo_coeff[:nao, :]
        mo_b = mf.mo_coeff[nao:, :]

        Lpq = None
        if (mem_incore + mem_now < pprpa.max_memory) or pprpa.mol.incore_anyway:
            if mo_a.dtype == numpy.double: # real coefficients, no SOC
                Lpq = nr_e2(pprpa.with_df._cderi, mo_a, ijslice, aosym='s2')
                Lpq += nr_e2(pprpa.with_df._cderi, mo_b, ijslice, aosym='s2')
            else:
                Lpq = numpy.zeros((naux, nmo_act * nmo_act), dtype=numpy.complex128)
                ijslice = (ijslice[0], ijslice[1], mo_a.shape[1] + ijslice[2], mo_a.shape[1] + ijslice[3])
                mo_tmp = numpy.asarray(numpy.hstack((mo_a.real, mo_a.real)), order='F')
                Lpq += nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')
                mo_tmp = numpy.asarray(numpy.hstack((mo_a.imag, mo_a.imag)), order='F')
                Lpq += nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')
                mo_tmp = numpy.asarray(numpy.hstack((mo_b.real, mo_b.real)), order='F')
                Lpq += nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')
                mo_tmp = numpy.asarray(numpy.hstack((mo_b.imag, mo_b.imag)), order='F')
                Lpq += nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')
                mo_tmp = numpy.asarray(numpy.hstack((mo_a.real, mo_a.imag)), order='F')
                Lpq += 1j * nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')
                mo_tmp = numpy.asarray(numpy.hstack((mo_a.imag, -mo_a.real)), order='F')
                Lpq += 1j * nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')
                mo_tmp = numpy.asarray(numpy.hstack((mo_b.real, mo_b.imag)), order='F')
                Lpq += 1j * nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')
                mo_tmp = numpy.asarray(numpy.hstack((mo_b.imag, -mo_b.real)), order='F')
                Lpq += 1j * nr_e2(pprpa.with_df._cderi, mo_tmp, ijslice, aosym='s2')

            return Lpq.reshape(naux, nmo_act, nmo_act)
        else:
            logger.warn(pprpa, 'Memory may not be enough!')
            raise NotImplementedError
    elif isinstance(mf, (pbc.scf.ghf.GHF, pbc.dft.gks.GKS)):
        raise NotImplementedError("GppRPA for periodic systems is not implemented yet.")
    else:
        raise NotImplementedError("GppRPA only supports GHF/GKS reference now.")


def complex_matrix_norm(matrix):
    """Calculate the Complex norm of each matrix element.

    Args:
        matrix (complex ndarray): input matrix.

    Returns:
        out (double ndarray): norm square of each element.
    """
    out = numpy.zeros(matrix.shape, dtype=numpy.double)
    out = numpy.power(numpy.abs(matrix.real), 2) + numpy.power(numpy.abs(matrix.imag), 2)
    return out


# analysis functions
def pprpa_davidson_print_eigenvector(pprpa, exci0, exci, xy):
    """Print dominant components of an eigenvector.

    Args:
        pprpa (RppRPADavidson): ppRPA object.
        exci0 (double): lowest eigenvalue.
        exci (double array): ppRPA eigenvalue.
        xy (double ndarray): ppRPA eigenvector.
    """
    nocc = pprpa.nocc
    nocc_act, nvir_act = pprpa.nocc_act, pprpa.nvir_act
    oo_dim = int((nocc_act - 1) * nocc_act / 2)
    logger.info(pprpa, "\n     print GppRPA excitations: triplet\n")

    tri_row_o, tri_col_o = numpy.tril_indices(nocc_act, -1)
    tri_row_v, tri_col_v = numpy.tril_indices(nvir_act, -1)

    nroot = len(exci)
    au2ev = 27.211386
    if pprpa.channel == "pp":
        for iroot in range(nroot):
            logger.info(pprpa, "#%-d excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV",
                  iroot + 1, (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev)
            if nocc_act > 0:
                full = numpy.zeros(shape=[nocc_act, nocc_act], dtype=xy.dtype)
                full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
                full = complex_matrix_norm(full)
                pairs = numpy.argwhere(full > pprpa.print_thresh)
                for i, j in pairs:
                    pprpa_print_a_pair(
                        pprpa, is_pp=False, p=i, q=j, percentage=full[i, j])

            full = numpy.zeros(shape=[nvir_act, nvir_act], dtype=xy.dtype)
            full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
            full = complex_matrix_norm(full)
            pairs = numpy.argwhere(full > pprpa.print_thresh)
            for a, b in pairs:
                pprpa_print_a_pair(
                    pprpa, is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            logger.info(pprpa, "")
    else:
        for iroot in range(nroot):
            logger.info(pprpa, "#%-d de-excitation:  exci= %-12.4f  eV   2e=  %-12.4f  eV",
                        iroot + 1, (exci[iroot] - exci0) * au2ev, exci[iroot] * au2ev)
            full = numpy.zeros(shape=[nocc_act, nocc_act], dtype=xy.dtype)
            full[tri_row_o, tri_col_o] = xy[iroot][:oo_dim]
            full = complex_matrix_norm(full)
            pairs = numpy.argwhere(full > pprpa.print_thresh)
            for i, j in pairs:
                pprpa_print_a_pair(
                    pprpa, is_pp=False, p=i, q=j, percentage=full[i, j])

            if nvir_act > 0:
                full = numpy.zeros(shape=[nvir_act, nvir_act], dtype=xy.dtype)
                full[tri_row_v, tri_col_v] = xy[iroot][oo_dim:]
                full = complex_matrix_norm(full)
                pairs = numpy.argwhere(full > pprpa.print_thresh)
                for a, b in pairs:
                    pprpa_print_a_pair(
                        pprpa, is_pp=True, p=a+nocc, q=b+nocc, percentage=full[a, b])
            logger.info(pprpa, "")

    return


def analyze_pprpa_davidson(pprpa):
    """Analyze ppRPA (Davidson algorithm) excited states.

    Args:
        pprpa (RppRPADavidson): GppRPA object.
    """
    logger.info(pprpa, "\nanalyze GppRPA results.")

    assert pprpa.exci is not None
    if pprpa.channel == "pp":
        exci0 = pprpa.exci[0]
    else:
        exci0 = pprpa.exci[0]
    pprpa_davidson_print_eigenvector(
        pprpa, exci0=exci0, exci=pprpa.exci, xy=pprpa.xy)
    return


class GppRPADavidson(RppRPADavidson):
    def __init__(
            self, mf, nocc_act=None, nvir_act=None, channel="pp", nroot=5,
            max_vec=200, max_iter=100, residue_thresh=1.0e-7, print_thresh=0.1,
            auxbasis=None):
        super().__init__(
            mf, nocc_act, nvir_act, channel, nroot, max_vec, max_iter,
            residue_thresh, print_thresh, auxbasis)
        # Most of the GPPRPA formulations are just the complex-valued
        # extention of those in triplet RPPRA, since they both use
        # the antisymmetric two-electron integrals.
        self.multi = "t"

    analyze = analyze_pprpa_davidson
    ao2mo = ao2mo

    def check_memory(self):
        """Check required memory.
        """
        # intermediate in contraction; mv_prod, tri_vec, xy
        mem = (3 * self.max_vec * self.full_dim) * 8 / 1.0e6
        if self._scf.mo_coeff.dtype == numpy.complex128:
            mem *= 2
        if mem < 1000:
            logger.info(self, "GppRPA needs at least %d MB memory.", mem)
        else:
            logger.info(self, "GppRPA needs at least %.1f GB memory.", mem/1.0e3)
        return

    def kernel(self):
        """Run GppRPA davidson diagonalization.
        """
        self.check_parameter()
        self.check_memory()

        cput0 = (logger.process_clock(), logger.perf_counter())
        if self.Lpi is None or self.Lpa is None:
            Lpq = self.ao2mo()
            self.Lpi = numpy.ascontiguousarray(Lpq[:, :, :self.nocc_act])
            self.Lpa = numpy.ascontiguousarray(Lpq[:, :, self.nocc_act:])
            logger.timer(self, "GppRPA integral transformation:", *cput0)
        self.dump_flags()
        kernel(pprpa=self)
        logger.timer(self, "GppRPA Davidson:", *cput0)

        self.exci_t = self.exci.copy()
        self.xy_t = self.xy.copy()
        return
