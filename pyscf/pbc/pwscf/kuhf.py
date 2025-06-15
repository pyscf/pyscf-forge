""" Spin-unrestricted Hartree-Fock in the Plane Wave Basis
"""


import h5py
import copy
import numpy as np

from pyscf.pbc import gto, scf
from pyscf.pbc.pwscf import khf, pw_helper
from pyscf.pbc.pwscf.pw_helper import get_kcomp, set_kcomp
from pyscf.lib import logger
from pyscf import __config__


def get_spin_component(C_ks, s):
    return get_kcomp(C_ks, s, load=False)


def get_nband(mf, nbandv, nbandv_extra):
    cell = mf.cell
    if isinstance(nbandv, int): nbandv = [nbandv] * 2
    if isinstance(nbandv_extra, int): nbandv_extra = [nbandv_extra] * 2
    nbando = cell.nelec
    nbandv_tot = [nbandv[s] + nbandv_extra[s] for s in [0,1]]
    nband = [nbando[s] + nbandv[s] for s in [0,1]]
    nband_tot = [nbando[s] + nbandv_tot[s] for s in [0,1]]

    return nbando, nbandv_tot, nband, nband_tot


def dump_moe(mf, moe_ks, mocc_ks, nband=None, trigger_level=logger.DEBUG):
    if nband is None: nband = [None,None]
    if isinstance(nband, int): nband = [nband,nband]
    for s in [0,1]:
        khf.dump_moe(mf, moe_ks[s], mocc_ks[s],
                     nband=nband[s], trigger_level=trigger_level)


def get_mo_energy(mf, C_ks, mocc_ks, mesh=None, Gv=None, exxdiv=None,
                  vj_R=None, ret_mocc=True, full_ham=False):
    cell = mf.cell
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    moe_ks = [None] * 2
    for s in [0,1]:
        C_ks_s = get_spin_component(C_ks, s)
        moe_ks[s] = khf.get_mo_energy(mf, C_ks_s, mocc_ks[s],
                                      mesh=mesh, Gv=Gv, exxdiv="none",
                                      vj_R=vj_R, comp=s,
                                      ret_mocc=False, full_ham=full_ham)

    if full_ham:
        return moe_ks

    # determine mo occ and apply ewald shift if requested
    mocc_ks = mf.get_mo_occ(moe_ks)
    if exxdiv is None: exxdiv = mf.exxdiv
    if exxdiv == "ewald":
        nkpts = len(mf.kpts)
        for s in [0,1]:
            for k in range(nkpts):
                # TODO why does it not work to apply ewald in khf.get_mo_energy
                moe_ks[s][k][mocc_ks[s][k] > khf.THR_OCC] -= mf.madelung

    if ret_mocc:
        return moe_ks, mocc_ks
    else:
        return moe_ks


def get_mo_occ(cell, moe_ks=None, C_ks=None):
    mocc_ks = [None] * 2
    for s in [0,1]:
        nocc = cell.nelec[s]
        if moe_ks is not None:
            mocc_ks[s] = khf.get_mo_occ(cell, moe_ks[s], nocc=nocc)
        elif C_ks is not None:
            C_ks_s = get_spin_component(C_ks, s)
            mocc_ks[s] = khf.get_mo_occ(cell, C_ks=C_ks_s, nocc=nocc)
        else:
            raise RuntimeError

    return mocc_ks


def get_init_guess(cell0, kpts, basis=None, pseudo=None, nvir=0,
                   key="hcore", out=None):
    """
    Args:
        nvir (int):
            Number of virtual bands to be evaluated. Default is zero.
        out (h5py group):
            If provided, the orbitals are written to it.
    """

    log = logger.Logger(cell0.stdout, cell0.verbose)

    nkpts = len(kpts)
    if out is None:
        out = [[None]*nkpts, [None]*nkpts]
    else:
        for s in [0,1]:
            if "%d"%s in out: del out["%d"%s]
            out.create_group("%d"%s)

    if basis is None: basis = cell0.basis
    if pseudo is None: pseudo = cell0.pseudo
    cell = cell0.copy()
    cell.basis = basis
    if len(cell._ecp) > 0:  # use GTH to avoid the slow init time of ECP
        gth_pseudo = {}
        for iatm in range(cell0.natm):
            atm = cell0.atom_symbol(iatm)
            if atm in gth_pseudo:
                continue
            q = cell0.atom_charge(iatm)
            if q == 0:  # Ghost atom
                continue
            else:
                gth_pseudo[atm] = "gth-pade-q%d"%q
        log.debug("Using the GTH-PP for init guess: %s", gth_pseudo)
        cell.pseudo = gth_pseudo
        cell.ecp = cell._ecp = cell._ecpbas = None
    else:
        cell.pseudo = pseudo
    cell.ke_cutoff = cell0.ke_cutoff
    cell.verbose = 0
    cell.build()

    log.info("generating init guess using %s basis", cell.basis)

    if len(kpts) < 30:
        pmf = scf.KUHF(cell, kpts)
    else:
        pmf = scf.KUHF(cell, kpts).density_fit()

    if key.lower() == "cycle1":
        pmf.max_cycle = 0
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    elif key.lower() in ["hcore", "h1e"]:
        h1e = pmf.get_hcore()
        h1e = [h1e, h1e]
        s1e = pmf.get_ovlp()
        mo_energy, mo_coeff = pmf.eig(h1e, s1e)
        mo_occ = pmf.get_occ(mo_energy, mo_coeff)
    elif key.lower() == "scf":
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    else:
        raise NotImplementedError("Init guess %s not implemented" % key)

    log.debug1("converting init MOs from GTO basis to PW basis")

    # TODO: support specifying nvir for each kpt (useful for e.g., metals)
    if isinstance(nvir, int): nvir = [nvir,nvir]

    mocc_ks_spin = [None] * 2
    for s in [0,1]:
        nocc = cell.nelec[s]
        nmo_ks = [len(mo_occ[s][k]) for k in range(nkpts)]
        ntot = nocc + nvir[s]
        ntot_ks = [min(ntot,nmo_ks[k]) for k in range(nkpts)]

        C_ks = get_spin_component(out, s)
        pw_helper.get_C_ks_G(cell, kpts, mo_coeff[s], ntot_ks, out=C_ks,
                             verbose=cell0.verbose)
        mocc_ks = [mo_occ[s][k][:ntot_ks[k]] for k in range(nkpts)]

        C_ks = khf.orth_mo(cell0, C_ks, mocc_ks)

        C_ks, mocc_ks = khf.add_random_mo(cell0, [ntot]*nkpts, C_ks, mocc_ks)

        mocc_ks_spin[s] = mocc_ks

    return out, mocc_ks_spin


def init_guess_by_chkfile(cell, chkfile_name, nvir, project=None, out=None):
    if isinstance(nvir, int): nvir = [nvir] * 2

    from pyscf.pbc.scf import chkfile
    scf_dict = chkfile.load_scf(chkfile_name)[1]
    mocc_ks = scf_dict["mo_occ"]
    nkpts = len(mocc_ks[0])
    if out is None: out = [[None] * nkpts for s in [0,1]]
    if isinstance(out, h5py.Group):
        for s in [0,1]:
            key = "%d"%s
            if key in out: del out[key]
            out.create_group(key)
    C_ks = out
    for s in [0,1]:
        ntot_ks = [None] * nkpts
        C_ks_s = get_spin_component(C_ks, s)
        with h5py.File(chkfile_name, "r") as f:
            C0_ks_s = f["mo_coeff/%d"%s]
            for k in range(nkpts):
                set_kcomp(get_kcomp(C0_ks_s, k), C_ks_s, k)
        for k in range(nkpts):
            nocc = np.sum(mocc_ks[s][k]>khf.THR_OCC)
            ntot_ks[k] = max(nocc+nvir[s], len(mocc_ks[s][k]))

        C_ks_s, mocc_ks[s] = khf.init_guess_from_C0(cell, C_ks_s, ntot_ks,
                                                    out=C_ks_s,
                                                    mocc_ks=mocc_ks[s])

    return C_ks, mocc_ks


def update_pp(mf, C_ks):
    tick = np.asarray([logger.process_clock(), logger.perf_counter()])
    if "t-ppnl" not in mf.scf_summary:
        mf.scf_summary["t-ppnl"] = np.zeros(2)

    mf.with_pp.update_vppnloc_support_vec(C_ks, ncomp=2)

    tock = np.asarray([logger.process_clock(), logger.perf_counter()])
    mf.scf_summary["t-ppnl"] += tock - tick


def update_k(mf, C_ks, mocc_ks):
    tick = np.asarray([logger.process_clock(), logger.perf_counter()])
    if "t-ace" not in mf.scf_summary:
        mf.scf_summary["t-ace"] = np.zeros(2)

    for s in [0,1]:
        C_ks_s = get_kcomp(C_ks, s, load=False)
        mf.with_jk.update_k_support_vec(C_ks_s, mocc_ks[s], mf.kpts, comp=s)

    tock = np.asarray([logger.process_clock(), logger.perf_counter()])
    mf.scf_summary["t-ace"] += tock - tick


def eig_subspace(mf, C_ks, mocc_ks, mesh=None, Gv=None, vj_R=None, exxdiv=None,
                 comp=None):
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
    moe_ks = [None] * 2
    for s in [0,1]:
        C_ks_s = get_spin_component(C_ks, s)
        mocc_ks_s = mocc_ks[s]
        C_ks_s, moe_ks[s], mocc_ks[s] = khf.eig_subspace(mf, C_ks_s, mocc_ks_s,
                                                         mesh=mesh, Gv=Gv,
                                                         vj_R=vj_R,
                                                         exxdiv="none", comp=s)
        if isinstance(C_ks, list): C_ks[s] = C_ks_s

    # determine mo occ and apply ewald shift if requested
    mocc_ks = mf.get_mo_occ(moe_ks)
    if exxdiv is None: exxdiv = mf.exxdiv
    if exxdiv == "ewald":
        nkpts = len(mf.kpts)
        for s in [0,1]:
            for k in range(nkpts):
                # TODO double-counting?
                moe_ks[s][k][mocc_ks[s][k] > khf.THR_OCC] -= mf.madelung

    return C_ks, moe_ks, mocc_ks


def energy_elec(mf, C_ks, mocc_ks, mesh=None, Gv=None, moe_ks=None,
                vj_R=None, exxdiv=None):
    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    kpts = mf.kpts
    nkpts = len(kpts)

    e_ks = np.zeros(nkpts)
    if moe_ks is None:
        if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
        e_comp = 0  # np.zeros(5)
        for s in [0,1]:
            C_ks_s = get_spin_component(C_ks, s)
            for k in range(nkpts):
                kpt = kpts[k]
                occ = np.where(mocc_ks[s][k] > khf.THR_OCC)[0]
                Co_k = get_kcomp(C_ks_s, k, occ=occ)
                e_comp_k = mf.apply_Fock_kpt(Co_k, kpt, mocc_ks, mesh, Gv,
                                             vj_R, exxdiv, comp=s,
                                             ret_E=True)[1]
                e_comp_k *= 0.5
                e_ks[k] += np.sum(e_comp_k)
                e_comp += e_comp_k
        e_comp /= nkpts

        if exxdiv == "ewald":
            e_comp[mf.scf_summary["e_comp_name_lst"].index("ex")] += \
                                                        mf.etot_shift_ewald

        for comp,e in zip(mf.scf_summary["e_comp_name_lst"],e_comp):
            mf.scf_summary[comp] = e
    else:
        for s in [0,1]:
            C_ks_s = get_spin_component(C_ks, s)
            moe_ks_s = moe_ks[s]
            for k in range(nkpts):
                kpt = kpts[k]
                occ = np.where(mocc_ks[s][k] > khf.THR_OCC)[0]
                Co_k = get_kcomp(C_ks_s, k, occ=occ)
                e1_comp = mf.apply_hcore_kpt(Co_k, kpt, mesh, Gv, mf.with_pp,
                                             comp=s, ret_E=True)[1]
                e1_comp *= 0.5
                e_ks[k] += np.sum(e1_comp) * 0.5 + np.sum(moe_ks_s[k][occ]) * 0.5
    e_scf = np.sum(e_ks) / nkpts

    if moe_ks is None and exxdiv == "ewald":
        # Note: ewald correction is not needed if e_tot is computed from moe_ks
        # since the correction is already in the mo energy
        e_scf += mf.etot_shift_ewald

    return e_scf


def converge_band(mf, C_ks, mocc_ks, kpts, Cout_ks=None,
                  mesh=None, Gv=None,
                  vj_R=None,
                  conv_tol_davidson=1e-6,
                  max_cycle_davidson=100,
                  verbose_davidson=0):

    nkpts = len(kpts)

    conv_ks = [None] * 2
    moeout_ks = [None] * 2
    fc_ks = [None] * 2
    if isinstance(C_ks, list):
        if Cout_ks is None: Cout_ks = [None] * 2
    else:
        Cout_ks = C_ks
    for s in [0,1]:
        C_ks_s = get_spin_component(C_ks, s)
        conv_ks[s], moeout_ks[s], Cout_ks_s, fc_ks[s] = khf.converge_band(
                            mf, C_ks_s, mocc_ks[s], kpts, mesh=mesh, Gv=Gv,
                            vj_R=vj_R, comp=s,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)

        if isinstance(C_ks, list): Cout_ks[s] = Cout_ks_s

    fc_ks = [fc_ks[0][k]+fc_ks[1][k] for k in range(nkpts)]

    return conv_ks, moeout_ks, Cout_ks, fc_ks


class PWKUHF(khf.PWKRHF):

    def __init__(self, cell, kpts=np.zeros((1,3)), ekincut=None,
                 exxdiv=getattr(__config__, 'pbc_scf_PWKUHF_exxdiv', 'ewald')):

        khf.PWKRHF.__init__(self, cell, kpts=kpts, exxdiv=exxdiv)

        self.nvir = [0,0]
        self.nvir_extra = [1,1]

    def get_init_guess_key(self, cell=None, kpts=None, basis=None, pseudo=None,
                           nvir=None, key="hcore", out=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if nvir is None: nvir = self.nvir

        if key in ["h1e","hcore","cycle1","scf"]:
            C_ks, mocc_ks = get_init_guess(cell, kpts,
                                           basis=basis, pseudo=pseudo,
                                           nvir=nvir, key=key, out=out)
        else:
            logger.warn(self, "Unknown init guess %s", key)
            raise RuntimeError

        return C_ks, mocc_ks

    def get_init_guess_C0(self, C0_ks, nvir=None, out=None):
        if nvir is None: nvir = self.nvir
        if isinstance(nvir, int): nvir = [nvir,nvir]
        nocc = self.cell.nelec
        nkpts = len(self.kpts)
        if out is None:
            out = [[None]*nkpts, [None]*nkpts]
        elif isinstance(out, h5py.Group):
            for s in [0,1]:
                if "%d"%s in out: del out["%d"%s]
                out.create_group("%d"%s)
        C_ks = out
        mocc_ks = [None] * 2
        for s in [0,1]:
            ntot_ks = [nocc[s]+nvir[s]] * len(self.kpts)
            C_ks_s = get_spin_component(C_ks, s)
            C0_ks_s = get_spin_component(C0_ks, s)
            n0_ks = [get_kcomp(C0_ks_s, k, load=False).shape[0]
                     for k in range(nkpts)]
            mocc_ks[s] = [np.asarray([1 if i < nocc[s] else 0
                          for i in range(n0_ks[k])]) for k in range(nkpts)]
            C_ks_s, mocc_ks[s] = khf.init_guess_from_C0(self.cell, C0_ks_s,
                                                        ntot_ks, out=C_ks_s,
                                                        mocc_ks=mocc_ks[s])

        return C_ks, mocc_ks

    def init_guess_by_chkfile(self, chk=None, nvir=None, project=None,
                              out=None):
        if chk is None: chk = self.chkfile
        if nvir is None: nvir = self.nvir
        return init_guess_by_chkfile(self.cell, chk, nvir, project=project,
                                     out=out)
    def from_chk(self, chk=None, project=None, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def get_mo_occ(mf, moe_ks=None, C_ks=None):
        return get_mo_occ(mf.cell, moe_ks, C_ks)

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None):
        return self.with_jk.get_vj_R(C_ks, mocc_ks, mesh=mesh, Gv=Gv, ncomp=2)

    get_nband = get_nband
    dump_moe = dump_moe
    update_pp = update_pp
    update_k = update_k
    eig_subspace = eig_subspace
    get_mo_energy = get_mo_energy
    energy_elec = energy_elec
    converge_band = converge_band


if __name__ == "__main__":
    cell = gto.Cell(
        atom = "C 0 0 0",
        a = np.eye(3) * 4,
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
        spin=2,
    )
    cell.mesh = [25, 25, 25]
    cell.build()
    cell.verbose = 6

    nk = 1
    kmesh = (nk,)*3
    kpts = cell.make_kpts(kmesh)

    umf = PWKUHF(cell, kpts)
    umf.nvir = [0,2]
    umf.nvir_extra = 4
    umf.kernel()

    umf.dump_scf_summary()

    assert(abs(umf.e_tot - -5.39994570429868) < 1e-5)
