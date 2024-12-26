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

from pyscf.pbc.scf import khf
from pyscf.pbc import df
import numpy as np
from pyscf.lib import logger
from pyscf.pbc.tools.pbc import get_monkhorst_pack_size



def madelung_modified(cell, kpts, shifted, ew_eta=None, anisotropic=False):
    # Here, the only difference from overleaf is that eta here is defined as 4eta^2 = eta_paper
    from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
    from pyscf.pbc.gto.cell import get_Gv_weights

    printstr = "Modified Madelung correction"
    if anisotropic:
        printstr += " with anisotropy"
        assert not isinstance(ew_eta, int)
        raise NotImplementedError("Anisotropic Madelung correction not correctly implemented yet")

    print(printstr)
    # Make ew_eta into array to allow for anisotropy if len==3
    ew_eta = np.array(ew_eta)

    Nk = get_monkhorst_pack_size(cell, kpts)
    import copy
    cell_input = copy.copy(cell)
    cell_input._atm = np.array([[1, cell._env.size, 0, 0, 0, 0]])
    cell_input._env = np.append(cell._env, [0., 0., 0.])
    cell_input.unit = 'B' # ecell.verbose = 0
    cell_input.a = a = np.einsum('xi,x->xi', cell.lattice_vectors(), Nk)

    if ew_eta is None:
        ew_eta, _ = cell_input.get_ewald_params(cell_input.precision, cell_input.mesh)
    chargs = cell_input.atom_charges()
    log_precision = np.log(cell_input.precision / (chargs.sum() * 16 * np.pi ** 2))
    ke_cutoff = -2 * np.mean(ew_eta) ** 2 * log_precision
    # Get FFT mesh from cutoff value
    mesh = cell_input.cutoff_to_mesh(ke_cutoff)
    
    # Get grid
    Gv, Gvbase, weights = cell_input.get_Gv_weights(mesh=mesh)
    # Get q+G points
    G_combined = Gv + shifted
    absG2 = np.einsum('gi,gi->g', G_combined, G_combined)

    if cell_input.dimension ==3:
        # Calculate |q+G|^2 values of the shifted points
        qG2 = np.einsum('gi,gi->g', G_combined, G_combined)
        if anisotropic:
            denom = -1 / (4 * ew_eta ** 2)
            exponent = np.einsum('gi,gi,i->g', G_combined, G_combined, denom)
            exponent[exponent == 0] = -1e200
            component = 4 * np.pi / qG2 * np.exp(exponent)
        else:
            qG2[qG2 == 0] = 1e200
            component = 4 * np.pi / qG2 * np.exp(-qG2 / (4 * ew_eta ** 2))

        # First term
        sum_term = weights*np.einsum('i->',component).real
        # Second Term
        if anisotropic:
            from scipy.integrate import tplquad, nquad
            from cubature import cubature
            denom = -1 / (4 * ew_eta ** 2)

            # i denotes coordinate, g denotes vector number 
            def integrand(x, y, z):
                qG = np.array([x, y, z])
                denom = -1 / (4 * ew_eta ** 2)
                exponent = np.einsum('i,i,i->', qG, qG, denom)
                qG2 = np.einsum('i,i->', qG, qG)
                out = 4 * np.pi / qG2 * np.exp(exponent)
                
                # Handle special case when x, y, and z are very small
                if np.isscalar(out):
                    if (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12):
                        out = 0
                else:
                    mask = (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12)
                    out[mask] = 0  # gaussian case
                return out
            
            def integrand_vectorized(x,y,z):
                qG = np.array([x, y, z])
                denom = -1 / (4 * ew_eta ** 2)
                exponent = np.einsum('ig,ig,i->g', qG, qG, denom)
                qG2 = np.einsum('ig,ig->g', qG, qG)
                out = 4 * np.pi / qG2 * np.exp(exponent)

                # Handle special case when x, y, and z are very small
                if np.isscalar(out):
                    if (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12):
                        out = 0
                else:
                    mask = (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12) & (np.abs(z) < 1e-12)
                    out[mask] = 0  # gaussian case

                return out

            x_min, x_max = -10,10
            y_min, y_max = -10,10
            z_min, z_max = -10,10
            global_tol = 1e-5
            integral_cart_imag = cubature(lambda xall: integrand_vectorized(xall[:,0], xall[:,1], xall[:,2]), 3, 1,
                                          [x_min, y_min, z_min], [x_max, y_max, z_max],relerr=global_tol,
                                          abserr=global_tol, vectorized=False)[0][0]
            # integral_cart_imag = cubature(lambda xall: integrand_vectorized(xall[:,0], xall[:,1], xall[:,2]), 3, 1,
            #                     [x_min, y_min, z_min], [x_max, y_max, z_max],relerr=global_tol,
            #                     abserr=global_tol, vectorized=False)[0][0]
            # subterm =(2*np.pi)**(-3) * tplquad(integrand, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)[0]
            # subterm =(2*np.pi)**(-3) * nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]],points=[[0,0,0]])[0]
            # nquad(integrand, [[-10, 10], [-10, 10], [-10, 10]],opts={'points':[0,0,0]})
        else:
            sub_term = 2*np.mean(ew_eta)/np.sqrt(np.pi)
        ewovrl = 0.0
        ewself_2 = 0.0
        print("Ewald components = %.15g, %.15g, %.15g,%.15g" % (ewovrl/2, sub_term/2,ewself_2/2, sum_term/2))
        return sub_term - sum_term

    elif cell_input.dimension == 2:  # Truncated Coulomb
        from scipy.special import erfc, erf
        # The following 2D ewald summation is taken from:
        # R. Sundararaman and T. Arias PRB 87, 2013
        def fn(eta, Gnorm, z):
            Gnorm_z = Gnorm * z
            large_idx = Gnorm_z > 20.0
            ret = np.zeros_like(Gnorm_z)
            x = Gnorm / 2. / eta + eta * z
            with np.errstate(over='ignore'):
                erfcx = erfc(x)
                ret[~large_idx] = np.exp(Gnorm_z[~large_idx]) * erfcx[~large_idx]
                ret[large_idx] = np.exp((Gnorm * z - x ** 2)[large_idx]) * erfcx[large_idx]
            return ret

        def gn(eta, Gnorm, z):
            return np.pi / Gnorm * (fn(eta, Gnorm, z) + fn(eta, Gnorm, -z))

        def gn0(eta, z):
            return -2 * np.pi * (z * erf(eta * z) + np.exp(-(eta * z) ** 2) / eta / np.sqrt(np.pi))

        b = cell_input.reciprocal_vectors()
        inv_area = np.linalg.norm(np.cross(b[0], b[1])) / (2 * np.pi) ** 2
        # Perform the reciprocal space summation over  all reciprocal vectors
        # within the x,y plane.
        planarG2_idx = np.logical_and(Gv[:, 2] == 0, absG2 > 0.0)

        G_combined = G_combined[planarG2_idx]
        absG2 = absG2[planarG2_idx]
        absG = absG2 ** (0.5)
        # Performing the G != 0 summation.
        coords = np.array([[0,0,0]])
        rij = coords[:, None, :] - coords[None, :, :] # should be just the zero vector for correction.
        Gdotr = np.einsum('ijx,gx->ijg', rij, G_combined)
        ewg = np.einsum('i,j,ijg,ijg->', chargs, chargs, np.cos(Gdotr),
                        gn(ew_eta, absG, rij[:, :, 2:3]))
        # Performing the G == 0 summation.
        # ewg += np.einsum('i,j,ij->', chargs, chargs, gn0(ew_eta, rij[:, :, 2]))

        ewg *= inv_area # * 0.5

        ewg_analytical = 2 * ew_eta / np.sqrt(np.pi)
        
        return ewg - ewg_analytical


def kernel(kmf, type="Non-SCF", df_type=None, kshift_rel=0.5, verbose=logger.NOTE, with_vk=False):
    
    from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
    from pyscf.pbc import gto,scf
    #To Do: Additional control arguments such as custom shift, scf control (cycles ..etc), ...
    #Cell formatting used in the built in Madelung code
    icell = kmf.cell

    log = logger.Logger(icell.stdout, verbose)
    
    ikpts = kmf.kpts
    dm_kpts = kmf.make_rdm1()
    
    def set_cell(mf):
        import copy
        Nk = get_monkhorst_pack_size(mf.cell, mf.kpts)
        ecell = copy.copy(mf.cell)
        ecell._atm = np.array([[1, mf.cell._env.size, 0, 0, 0, 0]])
        ecell._env = np.append(mf.cell._env, [0., 0., 0.])
        ecell.unit = 'B'
        # ecell.verbose = 0
        ecell.a = np.einsum('xi,x->xi', mf.cell.lattice_vectors(), Nk)
        ecell.mesh = np.asarray(mf.cell.mesh) * Nk
        return ecell

    # Function for Madelung constant calculation following formula in Stephen's paper
    def staggered_Madelung(cell_input, shifted, ew_eta=None, ew_cut=None, dm_kpts=None):
        # Here, the only difference from overleaf is that eta here is defined as 4eta^2 = eta_paper
        from pyscf.pbc.gto.cell import get_Gv_weights
        nk = get_monkhorst_pack_size(icell, ikpts)
        if ew_eta is None or ew_cut is None:
            ew_eta, ew_cut = cell_input.get_ewald_params(cell_input.precision, cell_input.mesh)
        chargs = cell_input.atom_charges()
        log_precision = np.log(cell_input.precision / (chargs.sum() * 16 * np.pi ** 2))
        ke_cutoff = -2 * ew_eta ** 2 * log_precision
        # Get FFT mesh from cutoff value
        mesh = cell_input.cutoff_to_mesh(ke_cutoff)
        # if cell_input.dimension <= 2:
        #     mesh[2] = 1
        # if cell_input.dimension == 1:
        #     mesh[1] = 1
        # Get grid
        Gv, Gvbase, weights = cell_input.get_Gv_weights(mesh = mesh)
        #Get q+G points
        G_combined = Gv + shifted
        absG2 = np.einsum('gi,gi->g', G_combined, G_combined)


        if cell_input.dimension ==3:
            # Calculate |q+G|^2 values of the shifted points
            qG2 = np.einsum('gi,gi->g', G_combined, G_combined)
            # Note: Stephen - remove those points where q+G = 0
            qG2[qG2 == 0] = 1e200
            # Now putting the ingredients together
            component = 4 * np.pi / qG2 * np.exp(-qG2 / (4 * ew_eta ** 2))
            # First term
            sum_term = weights*np.einsum('i->',component).real
            # Second Term
            sub_term = 2*ew_eta/np.sqrt(np.pi)
            return sum_term - sub_term

        elif cell_input.dimension == 2:  # Truncated Coulomb
            from scipy.special import erfc, erf
            # The following 2D ewald summation is taken from:
            # R. Sundararaman and T. Arias PRB 87, 2013
            def fn(eta, Gnorm, z):
                Gnorm_z = Gnorm * z
                large_idx = Gnorm_z > 20.0
                ret = np.zeros_like(Gnorm_z)
                x = Gnorm / 2. / eta + eta * z
                with np.errstate(over='ignore'):
                    erfcx = erfc(x)
                    ret[~large_idx] = np.exp(Gnorm_z[~large_idx]) * erfcx[~large_idx]
                    ret[large_idx] = np.exp((Gnorm * z - x ** 2)[large_idx]) * erfcx[large_idx]
                return ret

            def gn(eta, Gnorm, z):
                return np.pi / Gnorm * (fn(eta, Gnorm, z) + fn(eta, Gnorm, -z))

            def gn0(eta, z):
                return -2 * np.pi * (z * erf(eta * z) + np.exp(-(eta * z) ** 2) / eta / np.sqrt(np.pi))

            b = cell_input.reciprocal_vectors()
            inv_area = np.linalg.norm(np.cross(b[0], b[1])) / (2 * np.pi) ** 2
            # Perform the reciprocal space summation over  all reciprocal vectors
            # within the x,y plane.
            planarG2_idx = np.logical_and(Gv[:, 2] == 0, absG2 > 0.0)

            G_combined = G_combined[planarG2_idx]
            absG2 = absG2[planarG2_idx]
            absG = absG2 ** (0.5)
            # Performing the G != 0 summation.
            coords = np.array([[0,0,0]])
            rij = coords[:, None, :] - coords[None, :, :] # should be just the zero vector for correction.
            Gdotr = np.einsum('ijx,gx->ijg', rij, G_combined)
            ewg = np.einsum('i,j,ijg,ijg->', chargs, chargs, np.cos(Gdotr),
                            gn(ew_eta, absG, rij[:, :, 2:3]))
            # Performing the G == 0 summation.
            # ewg += np.einsum('i,j,ij->', chargs, chargs, gn0(ew_eta, rij[:, :, 2]))

            ewg *= inv_area # * 0.5

            ewg_analytical = 2 * ew_eta / np.sqrt(np.pi)
            return ewg - ewg_analytical


    if df_type is None:
        if icell.dimension <=2:
            df_type = df.GDF
        else:
            df_type = df.FFTDF
    if with_vk:
        Kmat = None
    if type == 0: # Regular
        # Extract kpts and append the shifted mesh
        nks = get_monkhorst_pack_size(icell, ikpts)
        shift = icell.get_abs_kpts([kshift_rel / n for n in nks])
        
        if icell.dimension <=2:
            shift[2] =  0
            if icell.dimension == 1:
                shift[1] = 0
                
        Nk = np.prod(nks) * 2
        log.debug("Shift is: " + str(shift))
        kmesh_shifted = ikpts + shift
        combined = np.concatenate((ikpts,kmesh_shifted),axis=0)
        log.debug(combined)

        mf2 = scf.KHF(icell, combined)
        mf2.with_df = df_type(icell, combined).build() #For 2d,1d, df_type cannot be FFTDF

        print(mf2.kernel())
        d_m = mf2.make_rdm1()
        #Get dm at kpoints in unshifted mesh
        dm2 = d_m[:Nk//2,:,:]
        #Get dm at kpoints in shifted mesh
        dm_shift = d_m[Nk//2:,:,:]
        #K matrix on shifted mesh with potential defined by dm on unshifted mesh
        _, Kmat = mf2.get_jk(cell=mf2.cell, dm_kpts= dm2, kpts=ikpts, kpts_band = kmesh_shifted)
        E_stagger = -1. / Nk * np.einsum('kij,kji', dm_shift, Kmat)
        E_stagger /= 2

        #Madelung constant computation
        count_iter = 1
        mf2.kpts = ikpts
        ecell = set_cell(mf2)
        ew_eta, ew_cut = ecell.get_ewald_params(mf2.cell.precision, mf2.cell.mesh)
        prev = 0
        conv_Madelung = 0
        while True and icell.dimension !=1:
            Madelung = staggered_Madelung(cell_input=ecell, shifted=shift, ew_eta=ew_eta, ew_cut=ew_cut)
            log.debug1("Iteration number " + str(count_iter))
            log.debug1("Madelung:" + str(Madelung))
            log.debug1("Eta:" + str(ew_eta))
            if count_iter > 1 and abs(Madelung - prev) < 1e-8:
                conv_Madelung = Madelung
                break
            if count_iter > 30:
                log.debug1("Error. Madelung constant not converged")
                break
            ew_eta *= 2
            count_iter += 1
            prev = Madelung

        nocc = mf2.cell.tot_electrons() // 2
        E_stagger_M = E_stagger + nocc * conv_Madelung
        results_dict = {
            "E_stagger_M":np.real(E_stagger_M),
            "E_stagger":np.real(E_stagger),
        }
        if with_vk:
            results_dict["vk"] = Kmat
            
    elif type == 1: # Split-SCF
        mfs = scf.KHF(icell, ikpts)
        print(mfs.kernel())
        nks = get_monkhorst_pack_size(mfs.cell, mfs.kpts)
        shift = mfs.cell.get_abs_kpts([kshift_rel / n for n in nks])
        kmesh_shifted = ikpts + shift
        #Calculation on shifted mesh
        mf2 = scf.KHF(icell, kmesh_shifted)
        mf2.with_df = df_type(icell, ikpts).build()  # For 2d,1d, df_type cannot be FFTDF
        print(mf2.kernel())
        dm_2 = mf2.make_rdm1()
        #Get K matrix on shifted kpts, dm from unshifted mesh
        _, Kmat = mf2.get_jk(cell = mf2.cell, dm_kpts = kmf.make_rdm1(), kpts = ikpts, kpts_band = kmesh_shifted)
        Nk = np.prod(nks)
        E_stagger = -1. / Nk * np.einsum('kij,kji', dm_2, Kmat) * 0.5
        E_stagger/=2

        #Madelung calculation
        count_iter = 1
        cell = set_cell(mf2)
        ew_eta, ew_cut = cell.get_ewald_params(mf2.cell.precision, mf2.cell.mesh)
        prev = 0
        conv_Madelung = 0
        while True and icell.dimension !=1:
            Madelung = staggered_Madelung(cell_input=cell, shifted=shift, ew_eta=ew_eta, ew_cut=ew_cut)
            # Madelung = madelung_modified(cell=ecell, shifted=shift, ew_eta=ew_eta, ew_cut=ew_cut)

            log.debug1("Iteration number " + str(count_iter))
            log.debug1("Madelung:" + str(Madelung))
            log.debug1("Eta:" + str(ew_eta))
            if count_iter > 1 and abs(Madelung - prev) < 1e-8:
                conv_Madelung = Madelung
                break
            if count_iter > 30:
                log.debug1("Error. Madelung constant not converged")
                break
            ew_eta *= 2
            count_iter += 1
            prev = Madelung

        nocc = mf2.cell.tot_electrons() // 2
        E_stagger_M = E_stagger + nocc * conv_Madelung
        results_dict = {
            "E_stagger_M":np.real(E_stagger_M),
            "E_stagger":np.real(E_stagger),
        }
        if with_vk:
            results_dict["vk"] = Kmat
    elif type == 2: # Non-SCF
        mf2 = kmf
        nocc = mf2.cell.tot_electrons()//2



        # Run SCF; should be one cycle if converged
        mf2.kernel()
        dm_un = mf2.make_rdm1()
        

        #  Defining size and making shifted mesh
        nks = get_monkhorst_pack_size(mf2.cell, mf2.kpts)
        shift = mf2.cell.get_abs_kpts([kshift_rel/n for n in nks])
        if icell.dimension <=2:
            shift[2] = 0
        elif icell.dimension == 1:
            shift[1] = 0
        kmesh_shifted = mf2.kpts + shift
        print(mf2.kpts)
        print(kmesh_shifted)

        print("\n")
        # if dm_kpts is None:
        #     log.debug("Converged Density Matrix")
        # else:
        #     log.debug("Input density matrix")

        for i in range(0,dm_un.shape[0]):
            log.debug1("kpt: " + str(mf2.kpts[i]) + "\n")
            mat = dm_un[i,:,:]
            for j in mat:
                log.debug1(' '.join(str(np.real(el)) for el in j))

        # Construct the Fock Matrix
        h1e = khf.get_hcore(mf2, cell=mf2.cell, kpts=kmesh_shifted)
        _, Kmat = mf2.get_jk(cell=mf2.cell, dm_kpts=dm_un, kpts=mf2.kpts, kpts_band=kmesh_shifted,
                    exxdiv='ewald',with_j=False)
        # Veff = Jmat - Kmat/2
        Veff = mf2.get_veff(cell=mf2.cell, dm_kpts=dm_un, kpts=mf2.kpts, kpts_band=kmesh_shifted)
        F_shift = h1e + Veff
        s1e = khf.get_ovlp(mf2, cell=mf2.cell, kpts=kmesh_shifted)
        mo_energy_shift, mo_coeff_shift = mf2.eig(F_shift, s1e)
        mo_occ_shift = mf2.get_occ(mo_energy_kpts=mo_energy_shift, mo_coeff_kpts=mo_coeff_shift)
        dm_shift = mf2.make_rdm1(mo_coeff_kpts=mo_coeff_shift, mo_occ_kpts=mo_occ_shift)

        # Computing the Staggered mesh energy
        Nk = np.prod(nks)
        E_stagger = -1./Nk * np.einsum('kij,kji', dm_shift, Kmat) * 0.5
        E_stagger/=2

        count_iter = 1
        ecell = set_cell(mf2)
        ew_eta, ew_cut = ecell.get_ewald_params(mf2.cell.precision, mf2.cell.mesh)
        prev = 0
        conv_Madelung = 0
        while True and icell.dimension !=1:
            Madelung = staggered_Madelung(cell_input=ecell, shifted=shift, ew_eta=ew_eta, ew_cut=ew_cut)
            print("Iteration number " + str(count_iter))
            print("Madelung:" + str(Madelung))
            print("Eta:" + str(ew_eta))
            if count_iter>1 and abs(Madelung-prev)<1e-8:
                conv_Madelung = Madelung
                break
            if count_iter>30:
                print("Error. Madelung constant not converged")
                break
            ew_eta*=2
            count_iter+=1
            prev = Madelung

        nocc = mf2.cell.tot_electrons()//2
        E_stagger_M = E_stagger + nocc*conv_Madelung
        print("Non SCF")

        
        results_dict = {
            "E_stagger_M":np.real(E_stagger_M),
            "E_stagger":np.real(E_stagger),
        }
        if with_vk:
            results_dict["vk"] = Kmat
            
    else:
        raise ValueError("Invalid stagger type", type)
    
    return results_dict



stagger_type_id = {
    'regular':  0,
    'standard': 0,
    'split-scf': 1,
    'splitscf': 1,
    'split_scf': 1,
    'non-scf': 2,
    'nonscf': 2,
    'non_scf': 2,
}

def get_stagger_type_id(stagger_type):
    return stagger_type_id.get(stagger_type.lower())

class KHF_stagger(khf.KSCF):
    def __init__(self, mf, stagger_type='regular',kshift_rel=0.5, with_vk=False, **kwargs):
        self.mf = mf
        self.cell = mf.cell
        self.stdout = mf.cell.stdout
        self.verbose = mf.cell.verbose
        self.rsjk = False
        self.max_memory = self.cell.max_memory
        self.with_df = mf.with_df
        self.stagger_type= get_stagger_type_id(stagger_type)
        self.kshift_rel = kshift_rel
        self.kpts = mf.kpts
        self.df_type = mf.with_df.__class__
        self.mo_coeff = mf.mo_coeff_kpts
        self.dm_kpts = mf.make_rdm1()
        self.nks = get_monkhorst_pack_size(self.cell, self.kpts)
        self.Nk = np.prod(self.nks)
        self.with_vk = with_vk
        
        
        
        
    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)

        log.info('Staggerd mesh method for exact exchange, type = %s', self.stagger_type)

        
        log.info('ek (%s) = %.15g', self.stagger_type, self.ek)
        log.info('etot (%s) = %.15g', self.stagger_type, self.e_tot)
        return self
    
    def compute_energy_components(self,hcore=True,nuc=True,j=True,k=False):
        Nk = self.Nk
        dm_kpts = self.dm_kpts

        if hcore:
            h1e = self.mf.get_hcore()
            self.ehcore = 1. / Nk * np.einsum('kij,kji->', h1e, dm_kpts).real
        if nuc:
            self.enuc = self.mf.energy_nuc()
        if j:
            Jo, _ = self.mf.get_jk(cell=self.mf.cell, dm_kpts=dm_kpts, kpts=self.mf.kpts, kpts_band=self.mf.kpts, with_k=False)

            ej = 1. / Nk * np.einsum('kij,kji', Jo, dm_kpts) * 0.5
            self.ej = ej.real

        if k:
            results = kernel(self.cell, self.kpts, type=self.stagger_type, df_type=self.df_type, dm_kpts=self.dm_kpts, mo_coeff_kpts=self.mo_coeff_kpts, kshift_rel=self.kshift_rel,with_vk=self.with_vk)
            self.ek = results["E_stagger_M"] 

    def kernel(self):
        results = kernel(self.mf,self.stagger_type,df_type=self.df_type,kshift_rel=self.kshift_rel)
        self.ek = results["E_stagger_M"] 
        
        self.compute_energy_components(hcore=True,nuc=True,j=True,k=False)
        self.e_tot = self.ek + self.ehcore + self.enuc + self.ej
        return self.e_tot
        
        
