import pyscf
import numpy
import scipy
import time
import functools
from pyscf import lib
from pyscf.pbc import tools
from pyscf.occri import occri_k

"""
This file is part of a proprietary software owned by Sandeep Sharma (sanshar@gmail.com).
Unless the parties otherwise agree in writing, users are subject to the following terms.

(0) This notice supersedes all of the header license statements.
(1) Users are not allowed to show the source code to others, discuss its contents, 
    or place it in a location that is accessible by others.
(2) Users can freely use resulting graphics for non-commercial purposes. 
    Credits shall be given to the future software of Sandeep Sharma as appropriate.
(3) Sandeep Sharma reserves the right to revoke the access to the code any time, 
    in which case the users must discard their copies immediately.
    """

def make_natural_orbitals(dms, S):
    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros((dms.shape[0], dms.shape[1]) )
    for i, dm in enumerate(dms):
        # Diagonalize the DM in AO
        A = lib.reduce(numpy.dot, (S, dm, S))
        w, v = scipy.linalg.eigh(A, b=S)

        # Flip since they're in increasing order
        mo_occ[i] = numpy.flip(w)
        mo_coeff[i] = numpy.flip(v, axis=1)
    return mo_coeff, mo_occ

class OCCRI(pyscf.pbc.df.fft.FFTDF):

    def __init__(self, mydf, kmesh = [1,1,1], **kwargs,):
        
        self.method = mydf.__module__.rsplit('.', 1)[-1]
        
        assert self.method in ['hf', 'uhf', 'khf', 'kuhf', 'rks', 'uks', 'krks', 'kuks']
        
        self.StartTime = time.time()
        cell = mydf.cell
        super().__init__(cell=cell)  # Need this for pyscf's eval_ao function
        self.exxdiv = "ewald"
        self.cell = cell
        self.kmesh = kmesh
        
        self.Nk = numpy.prod(self.kmesh)
        self.kpts = self.cell.make_kpts(
            self.kmesh, space_group_symmetry=False, time_reversal_symmetry=False, wrap_around=True
        )

        self.joblib_njobs = lib.numpy_helper._np_helper.get_omp_threads()

        self.get_j = mydf.get_j

        if str(self.method[0]) == 'k':
            self.get_jk = self.get_jk_kpts
            self.get_k = occri_k.get_k_occRI_kpts
        else:
            self.get_k = occri_k.get_k_occRI

    def get_jk(
        self,
        dm=None,
        hermi=1,
        kpt=None,
        kpts_band=None,
        with_j=None,
        with_k=None,
        omega=None,
        exxdiv=None,
        **kwargs,
    ):

        dm_shape = dm.shape
        if getattr(dm, "mo_coeff", None) is not None:
            mo_coeff = numpy.asarray(dm.mo_coeff)
            mo_occ = numpy.asarray(dm.mo_occ)
        else:
            mo_coeff, mo_occ = make_natural_orbitals(dm.reshape(-1, dm_shape[-2], dm_shape[-1]), self.S)

        dm = dm.reshape(-1, dm_shape[-2], dm_shape[-1])
        nsets = dm.shape[0]
        
        if mo_coeff.ndim != dm.ndim:
            mo_coeff = mo_coeff.reshape(dm.shape)
        if mo_occ.ndim != (dm.ndim -1) :
            mo_occ = mo_occ.reshape(-1, dm_shape.shape[-1])
        
        if with_j:
            vj = self.get_j(self, dm)

        vk = []
        if with_k:
            tol = 1.0e-6
            is_occ = mo_occ > tol
            mo_coeff = [coeff[:, is_occ[i]].T for i, coeff in enumerate(mo_coeff)]
            mo_coeff = [lib.tag_array(coeff, mo_occ=mo_occ[i][is_occ[i]]) for i, coeff in enumerate(mo_coeff)]
            for ii in range(nsets):
                Kuv = self.get_k(self, mo_coeff[ii])
                if exxdiv is not None:
                    Kuv += self.madelung * functools.reduce(numpy.matmul, (self.S, dm[ii], self.S))
                vk.append(Kuv)

        if with_j:
            vj = numpy.asarray(vj, dtype=dm.dtype).reshape(dm_shape)
        else:
            vj = None

        if with_k:
            vk = numpy.asarray(vk, dtype=dm.dtype).reshape(dm_shape)
        else:
            vk = None

        return vj, vk

    def get_jk_kpts(
        self,
        dm=None,
        hermi=1,
        kpts=None,
        kpts_band=None,
        with_j=None,
        with_k=None,
        omega=None,
        exxdiv=None,
        **kwargs,
    ):  
        skip_k = False
        if self.coulomb_only:
            skip_k = True
            with_k = False

        nK = self.Nk
        if with_k:
            if getattr(dm, "mo_coeff", None) is not None:
                mo_coeff = numpy.asarray(dm.mo_coeff)
                mo_occ = numpy.asarray(dm.mo_occ)
                if mo_coeff.ndim == 3:
                    mo_coeff = mo_coeff.reshape(-1, nK, mo_coeff.shape[-2], mo_coeff.shape[-1])
                if mo_occ.ndim == 2:
                    mo_occ = mo_occ.reshape(-1, nK, mo_occ.shape[-1])
            else:
                raise ValueError("Must send MO Coefficients with DM!!!")

        nao = dm.shape[-1]
        dm = dm.reshape(-1, nK, nao, nao)
        nsets = dm.shape[0]

        # Build Grids
        build_k = getattr(self, "full_grids_k", None) is None and with_k
        build_j = getattr(self, "full_grids_j", None) is None and with_j and not self.get_j_from_pyscf
        if build_j or build_k:
            self.build(build_j, build_k, incore=self.incore)
        # if self.get_j_from_pyscf:
        #     log = logger.Logger(self.stdout, self.verbose)
        #     self.tasks = pyscf.pbc.dft.multigrid.multigrid.multi_grids_tasks(
        #         self.cell, self.cell.mesh, log
        #     )
        #     log.debug("Multigrid ntasks %s", len(self.tasks))

        is_contracted_basis = self.cell.nao != self.cell_unc.nao

        vj = [None] * nsets
        if with_j:
            t0 = time.time()
            if self.get_j_from_pyscf:
                vj = pyscf.pbc.df.fft_jk.get_j_kpts(self, dm, kpts=self.kpts)
            else:
                if is_contracted_basis:
                    unc_dm = [[self.c @ dm[i][k] @ self.c.T for k in range(nK)] for i in range(nsets)]
                    Juv = self.get_j(self, unc_dm[0])  ## <--- address for UHF
                    vj = numpy.asarray([self.c.T @ jii @ self.c for jii in Juv]).reshape(dm.shape)
                else:
                    vj = self.get_j(self, dm[0]).reshape(dm.shape)  ## <--- address for UHF

            t_jk = time.time() - t0
            self.Times_["Coulomb"] += t_jk

        vk = [None] * nsets
        if with_k:
            tol = 1.0e-6
            is_occ = mo_occ > tol
            # dm2 = numpy.asarray(
            #     [
            #         coeff_k[:, occ_k > 0] * occ_k[occ_k > 0] @ coeff_k[:, occ_k > 0].conj().T
            #         for coeff_k, occ_k in zip(mo_coeff[0], mo_occ[0])
            #     ]
            # )
            # print("nocc:", numpy.sum(is_occ, axis=2), flush=True)
            mo_coeff = [[mo_coeff[i][k][:, is_occ[i][k]] for k in range(nK)] for i in range(nsets)]
            if is_contracted_basis:
                mo_coeff = [[self.c @ mo_coeff[i][k] for k in range(nK)] for i in range(nsets)]
            mo_coeff = [[mo_coeff[i][k].T for k in range(nK)] for i in range(nsets)]
            mo_coeff = [
                [lib.tag_array(mo_coeff[i][k], mo_occ=mo_occ[i][k][is_occ[i][k]]) for k in range(nK)]
                for i in range(nsets)
            ]

            t0 = time.time()
            for ii in range(nsets):
                Kuv = self.get_k(self, mo_coeff[ii])
                if is_contracted_basis:
                    Kuv = [self.c.T @ Kuv[k] @ self.c for k in range(nK)]
                if exxdiv is not None:
                    for k in range(nK):
                        Kuv[k] += self.madelung * functools.reduce(numpy.matmul, (self.S[k], dm[ii][k], self.S[k]))
                vk[ii] = numpy.asarray(Kuv)
            t_jk = time.time() - t0
            self.Times_["Exchange"] += t_jk

        if with_k and nsets == 1:
            vk = vk[0]

        if with_j and nsets == 1 and not self.get_j_from_pyscf:
            vj = vj[0]

        if skip_k:
            vk = vj * 0.0

        self.scf_iter += 1
        return vj, vk

    def build(self, with_j=True, with_k=True, incore=False):
        t0 = time.time()

        to_uncontracted_basis(self)
        self.atoms = make_atoms(self.cell_unc, self.rcut_epsilon)
        self.madelung = tools.pbc.madelung(self.cell, self.kpts)
        if self.use_kpt_symm:
            self.S = self.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts)
            self.S_unc = self.cell_unc.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts)
        else:
            self.S = self.cell.pbc_intor("int1e_ovlp", hermi=1).astype(numpy.float64)
            self.S_unc = self.cell_unc.pbc_intor("int1e_ovlp", hermi=1).astype(numpy.float64)

        if self.get_j_from_pyscf:
            with_j = False
        build_grids(self, with_j, with_k)
        if self.use_kpt_symm:
            if with_k:
                for kgrid in self.full_grids_k:
                    register_fft_factory_kpts(kgrid.mesh, self.fftw_njobs)
            if with_j:
                for jgrid in self.full_grids_j:
                    register_fft_factory_kpts(jgrid.mesh, self.fftw_njobs)
        else:
            for kgrid in self.full_grids_k:
                register_fft_factory(kgrid.mesh, self.fftw_njobs)
        if with_j:
            if with_k:
                for kgrid in self.full_grids_k:
                    register_fft_factory(kgrid.mesh, self.fftw_njobs)
            if with_j:
                for jgrid in self.full_grids_j:
                    register_fft_factory(jgrid.mesh, self.fftw_njobs)
        self.Times_["Grids"] += time.time() - t0

        if incore and with_j:
            eval_all_ao(self, self.atom_grids_j, self.full_grids_j, return_aos=False)

        if with_k:
            # Get W and intialize ISDF exchange
            t0 = time.time()
            do_isdf_and_build_w(self)
            self.Times_["ISDF"] += time.time() - t0

    def __del__(self):
        return

    def copy(self):
        """Returns a shallow copy"""
        return self.view(self.__class__)

    def get_keyword_arguments(self):
        # Retrieve all attributes, excluding any non-keyword arguments
        return {key: value for key, value in self.__dict__.items() if key != "cell"}

    def print_times(self):
        print()
        print("Wall time: ", time.time() - self.StartTime, flush=True)
        print("  >Initialize      :{0:18.2f}".format(self.Times_["Init"]), flush=True)
        print("  >Grids           :{0:18.2f}".format(self.Times_["Grids"]), flush=True)
        print("  >AOs             :{0:18.2f}".format(self.Times_["AOs"]), flush=True)
        print("  >ISDF-pts        :{0:18.2f}".format(self.Times_["ISDF-pts"]), flush=True)
        print("  >ISDF-vec        :{0:18.2f}".format(self.Times_["ISDF-vec"]), flush=True)
        print("  >ISDF-fft        :{0:18.2f}".format(self.Times_["ISDF-fft"]), flush=True)
        print("  >Coulomb         :{0:18.2f}".format(self.Times_["Coulomb"]), flush=True)
        print("  >Exchange        :{0:18.2f}".format(self.Times_["Exchange"]), flush=True)
        print("  >Diffuse-Diffuse :{0:18.2f}".format(self.Times_["Diffuse-Diffuse"]), flush=True)
        print("  >Sharp-Diffuse   :{0:18.2f}".format(self.Times_["Sharp-Diffuse"]), flush=True)
        print("  >Sharp-Sharp     :{0:18.2f}".format(self.Times_["Sharp-Sharp"]), flush=True)
        print("  >Diagonalize     :{0:18.2f}".format(self.Times_["Diagonalize"]), flush=True)
        print()

