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

import numpy as np

from pyscf.lib import logger
from pyscf.pbc import df
from pyscf.pbc import scf as pbcscf
from pyscf.pbc.dft import numint as pbcnumint
from pyscf.pbc.dft import rks
from pyscf.pbc.scf import khf
from pyscf.pbc.tools.pbc import get_monkhorst_pack_size


def exchange_energy_stagger(kmf, stagger_type=2, df_type=None, kshift_rel=0.5, verbose=logger.NOTE,
                            with_vk=False, nks=None):
    """ Computes the exchange energy using the staggered mesh method based on
        J. Chem. Theory Comput. 2024, 20, 18, 7958-7968.

    Args:
        kmf (khf.KSCF): KSCF object (KRKS or KRHF supported). For Non-SCF and
            Split-SCF, the SCF should be converged.
        type (int, optional): Regular (0), Split-SCF (1), or Non-SCF (2). See
            examples/pbc/22-kpoints_khf_stagger.py for usage. Defaults to
            Non-SCF.
        df_type (pbc.df.FFTDF or pbc.df.GDF, optional): Density fitting class.
            Defaults to df.FFTDF if cell.dimension <= 2, df.GDF otherwise.
        kshift_rel (float, optional): kpt mesh shift in relative coordinates of
            grid size. kshift_rel = 1.0 is equivalent to kshift_rel = 0.0.
            Defaults to 0.5.
        verbose (int, optional): Level of verbosity for debugging. Defaults to
            logger.NOTE.
        with_vk (bool, optional): Set True to return exchange matrix. Defaults
            to False.
        nks (array-like of size 3, optional): Number of kpoints in each
            direction of MP Mesh; used only for type="regular." Defaults to
            None.

    Returns:
        results (dict): Dictionary containing the following keys:
            E_stagger_M (float): Staggered mesh exchange energy with
                Madelung-like correction.
            E_stagger (float): Staggered mesh exchange energy.
            vk (np.array, optional): Exchange matrix if with_vk=True.
    """

    kmf.cell = kmf.cell
    log = logger.Logger(kmf.cell.stdout, verbose)
    Nk = np.prod(nks)
    Nk_combined = 2 * Nk

    if df_type is None:
        if kmf.cell.dimension <=2:
            df_type = df.GDF
        else:
            df_type = df.FFTDF

    Kmat = None

    kshift_abs = kmf.cell.get_abs_kpts([kshift_rel/n for n in nks])
    if kmf.cell.dimension <=2:
        kshift_abs[2] = 0
    elif kmf.cell.dimension == 1:
        kshift_abs[1] = 0

    # Prepare the kpoint meshes
    if stagger_type == 0:
        kpts_unshifted = kmf.kpts[:Nk_combined//2,:]
        kpts_shifted = kmf.kpts[Nk_combined//2:,:]
    elif stagger_type == 1 or stagger_type == 2:
        kpts_unshifted = kmf.kpts
        kpts_shifted = kmf.kpts + kshift_abs

    # Build or find the density matrices on shifted and unshifted_meshes
    if stagger_type == 0: # Regular
        dm_combined = kmf.make_rdm1()
        dm_unshifted = dm_combined[:Nk_combined//2,:,:]
        dm_shifted = dm_combined[Nk_combined//2:,:,:]

    elif stagger_type == 1: # Split-SCF
        dm_unshifted = kmf.make_rdm1()

        # Run SCF on shifted mesh
        kmf_shifted = pbcscf.KHF(kmf.cell, kpts_shifted)
        kmf_shifted.with_df = df_type(kmf.cell, kpts_shifted).build()  # For 2d,1d, df_type cannot be FFTDF
        kmf_shifted.kernel()
        dm_shifted = kmf_shifted.make_rdm1()

    elif stagger_type == 2: # Non-SCF
        dm_unshifted = kmf.make_rdm1()

        # Construct the Fock Matrix
        h1e = khf.get_hcore(kmf, cell=kmf.cell, kpts=kpts_shifted)
        Veff = kmf.get_veff(kmf.cell, dm_unshifted, kpts=kpts_unshifted, kpts_band=kpts_shifted)
        F_shift = h1e + Veff
        s1e = khf.get_ovlp(kmf, cell=kmf.cell, kpts=kpts_shifted)

        # Diagonalize to get densities at shifted mesh
        mo_energy_shift, mo_coeff_shift = kmf.eig(F_shift, s1e)
        mo_occ_shift = kmf.get_occ(mo_energy_kpts=mo_energy_shift, mo_coeff_kpts=mo_coeff_shift)
        dm_shifted = kmf.make_rdm1(mo_coeff_kpts=mo_coeff_shift, mo_occ_kpts=mo_occ_shift)

    log.debug1("Original kmesh: ", kpts_unshifted)
    log.debug1("Shifted kmesh:", kpts_shifted)

    # Compute the staggered mesh exact exchange energy
    _, Kmat = kmf.get_jk(cell=kmf.cell, dm_kpts=dm_unshifted,
                         kpts=kpts_unshifted, kpts_band=kpts_shifted, with_j=False)
    E_stagger = -1./Nk * np.einsum('kij,kji', dm_shifted, Kmat) * 0.5
    E_stagger /= 2.

    # Madelung-like correction if exxdiv=="ewald"
    if kmf.exxdiv == "ewald":
        madelung = KHF_stagger.converge_modified_madelung(kmf, kshift_abs,nks)
    else:
        log.warn("No madelung-like correction used")
        madelung = 0

    nocc = kmf.cell.tot_electrons() // 2
    E_stagger_M = E_stagger + nocc * madelung

    # Finalize Results
    results_dict = {
        "E_stagger_M":np.real(E_stagger_M),
        "E_stagger":np.real(E_stagger),
    }
    if with_vk:
        results_dict["vk"] = Kmat

    return results_dict


type_to_id = {
    'regular': 0,
    'standard': 0,
    'split-scf': 1,
    'splitscf': 1,
    'split_scf': 1,
    'non-scf': 2,
    'nonscf': 2,
    'non_scf': 2,
}

id_to_type = {
    0: 'regular',
    1: 'split-scf',
    2: 'non-scf',
}


def get_stagger_type_id(stagger_type):
    id = type_to_id.get(stagger_type.lower(), None)
    if id is None:
        raise ValueError("Invalid stagger_type: %s, must be one of %s" % (stagger_type, list(type_to_id.keys())))
    return id


class KHF_stagger(khf.KSCF):
    def __init__(self, mf, stagger_type='non-scf',kshift_rel=0.5, with_vk=False, **kwargs):
        self.mf = mf
        self.cell = mf.cell
        self.stdout = mf.cell.stdout
        self.verbose = mf.cell.verbose
        self.rsjk = mf.rsjk
        # self.max_memory = self.cell.max_memory
        self.with_df = mf.with_df
        self.stagger_type= get_stagger_type_id(stagger_type)
        self.kshift_rel = kshift_rel
        # self.kpts = mf.kpts
        self.df_type = mf.with_df.__class__
        # self.mo_coeff = mf.mo_coeff_kpts
        if self.stagger_type != 0:
            self.dm_kpts = mf.make_rdm1()
        else:
            self.dm_kpts = None
        self.nks = get_monkhorst_pack_size(self.cell, self.kpts)
        self.Nk = np.prod(self.nks)
        self.with_vk = with_vk
        self.dimension = mf.cell.dimension
        self.is_rks = isinstance(mf, rks.KohnShamDFT)
        if self.is_rks:
            self.xc = mf.xc
        else:
            self.xc = None
        self.log = logger.Logger(self.stdout, self.verbose)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)

        log.info('Staggered mesh method for exact exchange, type = %s', id_to_type[self.stagger_type])
        log.info('Exchange Energy (Stagger, %s) = %.15g', id_to_type[self.stagger_type], self.ek)
        log.info('Total Energy (Stagger, %s) = %.15g', id_to_type[self.stagger_type], self.e_tot)
        return self

    def compute_energy_components(self,hcore=True,nuc=True,j=True,k=False,xc=False):
        Nk = self.Nk
        dm_kpts = self.dm_kpts
        if hcore:
            h1e = self.mf.get_hcore()
            self.ehcore = 1. / Nk * np.einsum('kij,kji->', h1e, dm_kpts).real
        if nuc:
            self.enuc = self.mf.energy_nuc()
        if j:
            Jo, _ = self.mf.get_jk(cell=self.mf.cell, dm_kpts=dm_kpts, kpts=self.mf.kpts,
                                   kpts_band=self.mf.kpts, with_k=False)

            ej = 1. / Nk * np.einsum('kij,kji', Jo, dm_kpts) * 0.5
            self.ej = ej.real
        if k:
            results = exchange_energy_stagger(self.cell, self.kpts, stagger_type=self.stagger_type,
                                              df_type=self.df_type, dm_kpts=self.dm_kpts,
                                              mo_coeff_kpts=self.mo_coeff_kpts, kshift_rel=self.kshift_rel,
                                              with_vk=self.with_vk)
            self.ek = results["E_stagger_M"]
        if xc:
            # Note: XC here has no exact exchange, i.e. exc_PBE0 = PBEc + 0.75 * PBEx + 0.25 * HFx, with HFx = 0.0
            ni = pbcnumint.KNumInt()
            _, exc, _  = pbcnumint.nr_rks(ni,self.cell, self.mf.grids, self.xc, dm_kpts, kpts=self.kpts)
            self.exc = exc

    def run_scf_stagger(self, conv_tol=1e-10, conv_tol_grad=None, dm0=None, callback=None, conv_check=True, **kwargs):
        """Runs the SCF calculation with the combined unshifted and shifted kpt grids.
        Intended for stagger_type='regular'.
        """

        self.log.note("Running SCF for KHF Staggered Mesh (regular)")
        kshift_abs = self.cell.get_abs_kpts([self.kshift_rel / n for n in self.nks])
        if self.cell.dimension <=2:
            kshift_abs[2] = 0
            if self.cell.dimension == 1:
                kshift_abs[1] = 0

        Nk = np.prod(self.nks) * 2 # For unshifted and shifted mesh
        self.Nk = Nk

        kmesh_shifted = self.kpts + kshift_abs
        # Combine the unshifted and shifted meshes
        kmesh_combined = np.concatenate((self.kpts,kmesh_shifted),axis=0)

        # Build new KMF object with combined kpoint mesh
        kmf_combined = pbcscf.KHF(self.cell, kmesh_combined)
        kmf_combined.conv_tol = conv_tol
        kmf_combined.conv_tol_grad = conv_tol_grad
        kmf_combined.callback = callback
        kmf_combined.conv_check = conv_check
        kmf_combined.with_df = self.df_type(self.cell, kmesh_combined).build() # For 2d,1d, df_type cannot be FFTDF

        kmf_combined.kernel(dm0=dm0,**kwargs)
        dm_combined = kmf_combined.make_rdm1()

        # Set Attributes
        self.dm_kpts = dm_combined
        self.kpts = kmesh_combined
        self.mf = kmf_combined

    @staticmethod
    def build_probe_cell(mf, nks=None):
        import copy
        # Make unit cell with single probe charge at the origin.
        if nks is None:
            nks = get_monkhorst_pack_size(mf.cell, mf.kpts)
        ecell = copy.copy(mf.cell)
        ecell._atm = np.array([[1, mf.cell._env.size, 0, 0, 0, 0]])
        ecell._env = np.append(mf.cell._env, [0., 0., 0.])
        ecell.unit = 'B'

        # Expand the unit cell to the supercell based on the Monkhorst-Pack grid size
        ecell.a = np.einsum('xi,x->xi', mf.cell.lattice_vectors(), nks)
        ecell.mesh = np.asarray(mf.cell.mesh) * nks

        return ecell

    @staticmethod
    def modified_madelung(cell_input, kshift_abs, ew_eta=None, ew_cut=None):
        if ew_eta is None or ew_cut is None:
            ew_eta, ew_cut = cell_input.get_ewald_params(cell_input.precision, cell_input.mesh)
        chargs = cell_input.atom_charges()
        log_precision = np.log(cell_input.precision / (chargs.sum() * 16 * np.pi ** 2))
        ke_cutoff = -2 * ew_eta ** 2 * log_precision
        # Get FFT mesh from cutoff value
        mesh = cell_input.cutoff_to_mesh(ke_cutoff)
        # Get grid
        Gv, Gvbase, weights = cell_input.get_Gv_weights(mesh=mesh)
        # Get q+G points
        G_combined = Gv + kshift_abs

        # Calculate |q+G|^2 values of the shifted points
        qG2 = np.einsum('gi,gi->g', G_combined, G_combined)
        qG2[qG2 == 0] = 1e200
        component = 4 * np.pi / qG2 * np.exp(-qG2 / (4 * ew_eta ** 2))
        sum_ovrG_term = weights*np.einsum('i->',component).real
        self_term = 2*ew_eta/np.sqrt(np.pi)
        return sum_ovrG_term - self_term

    @staticmethod
    def converge_modified_madelung(kmf, kshift_abs, nks=None, thresh=1e-8, verbose=logger.NOTE):
        # Converge Modified Madelung Constant by iteratively increasing eta
        log = logger.Logger(kmf.cell.stdout, verbose)
        log.debug1("Iteratively converging Modified Madelung Constant with thresh=" + str(thresh))
        count_iter = 1
        ecell = KHF_stagger.build_probe_cell(kmf,nks=nks)
        ew_eta, ew_cut = ecell.get_ewald_params(kmf.cell.precision, kmf.cell.mesh)
        prev = 0
        converged_madelung = 0
        while True and kmf.cell.dimension !=1:
            madelung = KHF_stagger.modified_madelung(cell_input=ecell, kshift_abs=kshift_abs, ew_eta=ew_eta,
                                                     ew_cut=ew_cut)
            log.debug1("Iteration number " + str(count_iter))
            log.debug1("Madelung:" + str(madelung))
            log.debug1("Eta:" + str(ew_eta))
            if count_iter > 1 and abs(madelung - prev) < thresh:
                converged_madelung = madelung
                break
            if count_iter > 30:
                log.error("Madelung constant not converged")
                raise ValueError("Madelung constant not converged")
            ew_eta *= 2
            count_iter += 1
            prev = madelung
        return converged_madelung

    def kernel(self, dm0=None, conv_tol=1e-10, conv_tol_grad=None, callback=None, conv_check=True, **kwargs):
        # Rerun SCF if using Regular version, otherwise check if SCF is converged
        if self.stagger_type == 0:
            self.run_scf_stagger(conv_tol=conv_tol, conv_tol_grad=conv_tol_grad, dm0=dm0, callback=callback,
                           conv_check=conv_check, **kwargs)
        else:
            assert self.mf.converged, "Converged KSCF required for Non-SCF and Split-SCF"

        # Compute Staggered Mesh Exchange Energy
        results = exchange_energy_stagger(self.mf,self.stagger_type,df_type=self.df_type,kshift_rel=self.kshift_rel,
                                          nks=self.nks)
        self.ek = results["E_stagger_M"]

        # Setup for total energy computation
        if isinstance(self.mf,rks.KohnShamDFT):
            xc = True
            ni = pbcnumint.KNumInt()
            _, _, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=self.cell.spin)
        elif isinstance(self.mf,khf.KRHF):
            xc = False
            self.exc = 0.0
            hyb = 1.0
        else:
            logger.error("KHF Stagger: Invalid SCF type", self.mf.__class__)
            raise ValueError("Invalid SCF type")

        self.compute_energy_components(hcore=True,nuc=True,j=True,k=False,xc=xc)
        self.e_tot = hyb * self.ek + self.ehcore + self.enuc + self.ej + self.exc
        self.dump_flags()
        return self.e_tot


if __name__ == '__main__':
    from pyscf.pbc import gto
    '''
    Hydrogen dimer
    '''
    cell = gto.Cell()
    cell.pseudo = 'gth-pbe'
    cell.basis = 'gth-szv'
    cell.ke_cutoff = 100
    cell.atom = '''
        H 3.00   3.00   2.10
        H 3.00   3.00   3.90
        '''
    cell.a = '''
        6.0   0.0   0.0
        0.0   6.0   0.0
        0.0   0.0   6.0
        '''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    # HF calcuation to base Non-SCF and Split-SCF staggered mesh calculations on.
    nks = [2, 2, 2]
    kpts = cell.make_kpts(nks, with_gamma_point=True)
    kmf = pbcscf.KRHF(cell, kpts, exxdiv='ewald')
    kmf.with_df = df.GDF(cell, kpts).build()
    ehf = kmf.kernel()

    '''
    KHF Stagger, Non-SCF version
    Compute densities at shifted mesh non-SCF using F_unshifted. Additional cost
    is ~ 1 extra K-build.
    '''
    kmf_stagger = KHF_stagger(kmf,"non-scf")
    kmf_stagger.kernel()
    etot = kmf_stagger.e_tot
    ek_stagger = kmf_stagger.ek

    print('Non-SCF Stagger')
    print('Total energy: ', etot)
    print('Exchange energy: ', ek_stagger)
