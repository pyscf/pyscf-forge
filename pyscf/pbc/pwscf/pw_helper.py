""" Helper functions for PW SCF
"""


import copy
import h5py
import tempfile
import numpy as np
import scipy.linalg

from pyscf.pbc import tools, df
from pyscf.pbc.dft import rks
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib
from pyscf.lib import logger


""" Helper functions
"""
def get_kcomp(C_ks, k, load=True, occ=None, copy=False):
    if C_ks is None: return None
    if isinstance(C_ks, (list,np.ndarray)):
        if occ is None:
            if copy:
                return C_ks[k].copy()
            else:
                return C_ks[k]
        else:
            if copy:
                return C_ks[k][occ].copy()
            else:
                return C_ks[k][occ]
    else:
        key = "%d"%k
        if load:
            if occ is None:
                return C_ks[key][()]
            else:
                if isinstance(occ, np.ndarray):
                    occ = occ.tolist()
                return C_ks[key][occ]
        else:
            return C_ks[key]
def safe_write(h5grp, key, val, occ=None):
    if key in h5grp:
        if occ is None:
            if h5grp[key].shape == val.shape:
                h5grp[key][()] = val
            else:
                del h5grp[key]
                h5grp[key] = val
        else:
            h5grp[key][occ] = val
    else:
        h5grp[key] = val
def set_kcomp(C_k, C_ks, k, occ=None, copy=False):
    if isinstance(C_ks, (list,np.ndarray)):
        if occ is None:
            if copy:
                C_ks[k] = C_k.copy()
            else:
                C_ks[k] = C_k
        else:
            if copy:
                C_ks[k][occ] = C_k.copy()
            else:
                C_ks[k][occ] = C_k
    else:
        key = "%d"%k
        safe_write(C_ks, key, C_k, occ)
def acc_kcomp(C_k, C_ks, k, occ=None):
    if isinstance(C_ks, (list,np.ndarray)):
        if occ is None:
            C_ks[k] += C_k
        else:
            C_ks[k][occ] += C_k
    else:
        key = "%d"%k
        if occ is None:
            C_ks[key][()] += C_k
        else:
            if isinstance(occ, np.ndarray):
                occ = occ.tolist()
            C_ks[key][occ] += C_k
def scale_kcomp(C_ks, k, scale):
    if isinstance(C_ks, (list,np.ndarray)):
        C_ks[k] *= scale
    else:
        key = "%d"%k
        C_ks[key][()] *= scale


def timing_call(func, args, tdict, tname):
    tick = np.asarray([logger.process_clock(), logger.perf_counter()])

    res = func(*args)

    tock = np.asarray([logger.process_clock(), logger.perf_counter()])
    if tname not in tdict:
        tdict[tname] = np.zeros(2)
    tdict[tname] += tock - tick

    return res


def orth(cell, C, thr_nonorth=1e-6, thr_lindep=1e-12, follow=True):
    n = C.shape[0]
    norm = lib.einsum("ig,ig->i", C.conj(), C).real**0.5
    C *= 1./norm.reshape(-1,1)
    S = lib.dot(C.conj(), C.T)
    nonorth_err = np.max(np.abs(S - np.eye(S.shape[0])))
    if nonorth_err < thr_nonorth:
        return C

    e, u = scipy.linalg.eigh(S)
    idx_keep = np.where(e > thr_lindep)[0]
    nkeep = idx_keep.size
    if n == nkeep:  # symm orth
        lib.logger.debug2(cell, "Cond nubmer = %.3e", e.max()/e.min())
        if follow:
            # reorder to maximally overlap original orbs
            idx = []
            for i in range(n):
                order = np.argsort(np.abs(u[i]))[::-1]
                for j in order:
                    if j not in idx:
                        break
                idx.append(j)
            U = lib.dot(u[:,idx]*e[idx]**-0.5, u[:,idx].conj()).T
        else:
            U = lib.dot(u*e**-0.5, u.conj()).T
    else:   # cano orth
        lib.logger.debug2(cell, "Cond nubmer = %.3e  Drop %d orbitals",
                          e.max()/e.min(), n-nkeep)
        U = (u[:,idx_keep]*e[idx_keep]**-0.5).T
    C = lib.dot(U, C)

    return C


def get_nocc_ks_from_mocc(mocc_ks):
    return np.asarray([np.sum(np.asarray(mocc) > 0) for mocc in mocc_ks])


def get_C_ks_G(cell, kpts, mo_coeff_ks, n_ks, out=None, verbose=0):
    """ Return Cik(G) for input MO coeff. The normalization convention is such that Cik(G).conj()@Cjk(G) = delta_ij.
    """
    log = logger.new_logger(cell, verbose)

    nkpts = len(kpts)
    if out is None: out = [None] * nkpts

    dtype = np.complex128
    dsize = 16

    mydf = df.FFTDF(cell)
    mesh = mydf.mesh
    ni = mydf._numint

    coords = mydf.grids.coords
    ngrids = coords.shape[0]
    weight = mydf.grids.weights[0]
    fac = (weight/ngrids)**0.5

    frac = 0.5  # to be safe
    cur_memory = lib.current_memory()[0]
    max_memory = (cell.max_memory - cur_memory) * frac
    log.debug1("max_memory= %s MB (currently used %s MB)", cell.max_memory, cur_memory)
    # FFT needs 2 temp copies of MOs
    extra_memory = 2*ngrids*np.max(n_ks)*dsize / 1.e6
    # add 1 for ao_ks
    perk_memory = ngrids*(np.max(n_ks)+1)*dsize / 1.e6
    kblksize = min(int(np.floor((max_memory-extra_memory) / perk_memory)),
                   nkpts)
    if kblksize <= 0:
        log.warn("Available memory %s MB cannot perform conversion for orbitals"
                 " of a single k-point. Calculations may crash and `cell.memory"
                 " = %s` is recommended.",
                 max_memory, (perk_memory + extra_memory) / frac + cur_memory)

    log.debug1("max memory= %s MB, extra memory= %s MB, perk memory= %s MB,"
               " kblksize= %s", max_memory, extra_memory, perk_memory, kblksize)

    for k0,k1 in lib.prange(0, nkpts, kblksize):
        nk = k1 - k0
        C_ks_R = [np.zeros([ngrids,n_ks[k]], dtype=dtype)
                   for k in range(k0,k1)]
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts[k0:k1]):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for krel, ao in enumerate(ao_ks):
                k = krel + k0
                kpt = kpts[k].reshape(-1,1)
                C_k = mo_coeff_ks[k][:,:n_ks[k]]
                C_ks_R[krel][p0:p1] = lib.dot(ao, C_k)
                if not gamma_point(kpt):
                    phase = np.exp(-1j * lib.dot(coords[p0:p1], kpt))
                    C_ks_R[krel][p0:p1] *= phase
                    phase = None
            ao = ao_ks = None

        for krel in range(nk):
            C_k_R = tools.fft(C_ks_R[krel].T * fac, mesh)
            set_kcomp(C_k_R, out, krel+k0)

    return out


""" Contracted PW
"""
def get_mesh_map(cell, ke_cutoff, ke_cutoff2, mesh=None, mesh2=None):
    """ Input ke_cutoff > ke_cutoff2, hence define a dense grid "mesh" and
    a sparse grid "mesh2" where mesh2 is rigorously a subset of mesh. This
    function returns the indices of grid points in mesh2 in mesh.
    """

    latvec = cell.lattice_vectors()
    if mesh is None:
        mesh = tools.cutoff_to_mesh(latvec, ke_cutoff)
    else:
        mesh = np.asarray(mesh)
    if mesh2 is None:
        mesh2 = tools.cutoff_to_mesh(latvec, ke_cutoff2)
    else:
        mesh2 = np.asarray(mesh2)
    assert(np.all(mesh>=mesh2))
    rs = [np.fft.fftfreq(mesh[i], 1./mesh[i]) for i in range(3)]
    rs2 = [np.fft.fftfreq(mesh2[i], 1./mesh2[i]) for i in range(3)]
    idxr = [np.where(abs(rs[i][:,None]-rs2[i])<1e-3)[0] for i in range(3)]
    nr = [len(rs[i]) for i in range(3)]
    mesh_map = np.ravel(((idxr[0]*nr[1]*nr[2])[:,None] +
                          idxr[1]*nr[2])[:,:,None] + idxr[2])

    return mesh_map


def remove_pGTO_from_cGTO_(bdict, amax=None, amin=None, verbose=0):
    """ Removing from input GTO basis all primitive GTOs whose exponents are >amax or <amin.
    """
    from pyscf import gto as mol_gto
    from pyscf.pbc import gto as pbc_gto
    def prune(blist):
        if amin is None and amax is None:
            return blist

        amax_ = 1e10 if amax is None else amax
        amin_ = -1 if amin is None else amin

        blist_new = []
        for lbs in blist:
            l = lbs[0]
            bs = []
            for b in lbs[1:]:
                e = b[0]
                if amin_ < e < amax_:
                    bs.append(b)
            if len(bs) > 0:
                blist_new.append([l] + bs)

        return blist_new

    ang_map = ["S", "P", "D", "F", "G", "H", "I", "J"]

    log = lib.logger.Logger(verbose=verbose)
    log.debug1("Generating basis...")
    bdict_new = {}
    for atm,basis in bdict.items():
        if isinstance(basis, str):
            if "gth" in basis.lower():
                cell = pbc_gto.M(atom="%s 0 0 0"%atm, basis=basis, spin=1)
                blist = cell._basis[atm]
            else:
                blist = mol_gto.basis.load(basis, atm)
        else:
            blist = basis
        bdict_new[atm] = prune(blist)

        for lbs in bdict_new[atm]:
            l = lbs[0]
            bs = lbs[1:]
            log.debug1("%s %s", atm, ang_map[l])
            for b in bs:
                log.debug1("%.10f " * len(b), *b)
    log.debug1("")

    return bdict_new


def cpw_from_cell(cell_cpw, kpts, out=None):
    nao = cell_cpw.nao_nr()
    nkpts = len(kpts)
    Cao_ks = [np.eye(nao)+0j for k in range(nkpts)]
    nao_ks = np.ones(nkpts,dtype=int) * nao
    if out is None: out = [None] * nkpts
    out = get_C_ks_G(cell_cpw, kpts, Cao_ks, nao_ks, out=out)
    return out


def gto2cpw(cell, basis, kpts, amin=None, amax=None, ke_or_mesh=None, out=None):
    """ Get the contracted PWs for input GTO basis
    Args:
        basis:
            Some examples:
                basis = "ccecp-cc-pVDZ" (applies to all atoms in "cell")
                basis = {"C": "ccecp-cc-pVDZ", "N": "gth-tzv2p"}
                basis = {"C": [[0,[12,0.7],[5,0.3],[1,0.5]]], "N": "gth-szv"}
        amin/amax:
            If provided, all primitive GTOs from the basis that have exponents >amax or <amin will be removed.
        ke_or_mesh:
            If list/tuple/numpy array, interpreted as mesh
            otherwise, interpreted as ke_cutoff.
            Default is None which uses the same ke_cutoff/mesh from input "cell".
        out:
            None --> return a list of numpy arrays (incore mode)
            hdf5 group --> saved to the hdf5 group (outcore mode)
    """
# formating basis
    atmsymbs = cell._basis.keys()
    if isinstance(basis, str):
        basisdict = {atmsymb: basis for atmsymb in atmsymbs}
    elif isinstance(basis, dict):
        assert(basis.keys() == atmsymbs)
        basisdict = basis
    else:
        raise TypeError("Input basis must be either a str or dict.")
# pruning pGTOs that have unwanted exponents
    basisdict = remove_pGTO_from_cGTO_(basisdict, amax=amax, amin=amin)
# make a new cell with the modified GTO basis
    cell_cpw = cell.copy()
    cell_cpw.basis = basisdict
    if ke_or_mesh is not None:
        if isinstance(ke_or_mesh, (list,tuple,np.ndarray)):
            cell_cpw.mesh = ke_or_mesh
        else:
            cell_cpw.ke_cutoff = ke_or_mesh
    cell_cpw.verbose = 0
    cell_cpw.build()
# GTOs --> CPWs
    out = cpw_from_cell(cell_cpw, kpts, out=out)

    return out


def gtomf2pwmf(mf, chkfile=None):
    """
    Args:
        chkfile (str):
            A hdf5 file to store chk variables (mo_energy, mo_occ, etc.).
            If not provided, a temporary file is generated.
    """
    from pyscf.pbc import scf
    assert(isinstance(mf, (scf.khf.KRHF,scf.kuhf.KUHF,scf.uhf.UHF)))

    from pyscf.pbc import pwscf
    cell = mf.cell
    kpts = getattr(mf, "kpts", np.zeros((1,3)))
    nkpts = len(kpts)
# transform GTO MO coeff to PW MO coeff
    Cgto_ks = mf.mo_coeff
    if isinstance(mf, scf.khf.KRHF):
        pwmf = pwscf.KRHF(cell, kpts)
        nmo_ks = [Cgto_ks[k].shape[1] for k in range(nkpts)]
        pwmf.mo_coeff = C_ks = get_C_ks_G(cell, kpts, Cgto_ks, nmo_ks)
        pwmf.mo_energy = moe_ks = mf.mo_energy
        pwmf.mo_occ = mocc_ks = mf.mo_occ
        pwmf.e_tot = mf.e_tot
    elif isinstance(mf, scf.kuhf.KUHF):
        pwmf = pwscf.KUHF(cell, kpts)
        C_ks = [None] * 2
        for s in [0,1]:
            nmo_ks = [Cgto_ks[s][k].shape[1] for k in range(nkpts)]
            C_ks[s] = get_C_ks_G(cell, kpts, Cgto_ks[s], nmo_ks)
        pwmf.mo_coeff = C_ks
        pwmf.mo_energy = moe_ks = mf.mo_energy
        pwmf.mo_occ = mocc_ks = mf.mo_occ
        pwmf.e_tot = mf.e_tot
    elif isinstance(mf, scf.uhf.UHF):
        pwmf = pwscf.KUHF(cell, kpts)
        C_ks = [None] * 2
        for s in [0,1]:
            nmo_ks = [Cgto_ks[s].shape[1]]
            C_ks[s] = get_C_ks_G(cell, kpts, [Cgto_ks[s]], nmo_ks)
        pwmf.mo_coeff = C_ks
        pwmf.mo_energy = moe_ks = [[mf.mo_energy[s]] for s in [0,1]]
        pwmf.mo_occ = mocc_ks = [[mf.mo_occ[s]] for s in [0,1]]
        pwmf.e_tot = mf.e_tot
    else:
        raise TypeError
# update chkfile
    if chkfile is None:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        chkfile = swapfile.name
        swapfile = None
    pwmf.chkfile = chkfile
    from pyscf.pbc.pwscf.chkfile import dump_scf
    dump_scf(mf.cell, pwmf.chkfile, mf.e_tot, moe_ks, mocc_ks, C_ks)
    pwmf.converged = True

    return pwmf


""" kinetic energy
"""
def apply_kin_kpt(C_k, kpt, mesh, Gv):
    no = C_k.shape[0]
    kG = kpt + Gv if np.sum(np.abs(kpt)) > 1.E-9 else Gv
    kG2 = np.einsum("gj,gj->g", kG, kG) * 0.5
    Cbar_k = C_k * kG2

    return Cbar_k


""" Charge mixing methods
"""
class _Mixing:
    def __init__(self, mf):
        self.cycle = 0
        if isinstance(mf, rks.KohnShamDFT):
            self._ks = True
        else:
            self._ks = False

    def _extract_kwargs(self, f):
        if self._ks:
            return {
                "exc": f.exc,
                "vxcdot": f.vxcdot,
                "vxc_R": f.vxc_R,
                "vtau_R": f.vtau_R,
            }
        else:
            return {}

    def _tag(self, f, kwargs):
        if self._ks:
            return lib.tag_array(f, **kwargs)
        else:
            return f

    def _next_step(self, mf, f, ferr):
        raise NotImplementedError

    def next_step(self, mf, f, flast):
        ferr = f - flast
        kwargs = self._extract_kwargs(f)
        return self._tag(self._next_step(mf, f, ferr), kwargs)


class SimpleMixing(_Mixing):
    def __init__(self, mf, beta=0.3):
        super().__init__(mf)
        self.beta = beta

    def _next_step(self, mf, f, ferr):
        self.cycle += 1

        return f - ferr * self.beta

    def next_step(self, mf, f, flast):
        ferr = f - flast
        kwargs = self._extract_kwargs(f)
        kwargslast = self._extract_kwargs(flast)
        for kw in ["vxc_R", "vtau_R"]:
            if kw in kwargs and kwargs[kw] is not None:
                kwargs[kw] = self._next_step(
                    mf, kwargs[kw].ravel(), (kwargs[kw] - kwargslast[kw]).ravel()
                ).reshape(kwargs[kw].shape)
        return self._tag(self._next_step(mf, f, ferr), kwargs)

from pyscf.lib.diis import DIIS
class AndersonMixing(_Mixing):
    def __init__(self, mf, ndiis=10, diis_start=1):
        super().__init__(mf)
        self.diis = DIIS()
        self.diis.space = ndiis
        self.diis.min_space = diis_start

    def _next_step(self, mf, f, ferr):
        self.cycle += 1

        return self.diis.update(f, ferr)
