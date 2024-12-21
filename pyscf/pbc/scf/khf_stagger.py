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
import numpy as np



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


def kernel(icell, ikpts, version="Non-SCF", df_type=None, dm_kpts=None, mo_coeff_kpts=None, 
                kshift_rel=0.5, fourinterp=False, ss_params={}, modified_madelung_params={}):
    from pyscf.pbc.tools.pbc import get_monkhorst_pack_size
    from pyscf.pbc import gto,scf
    #To Do: Additional control arguments such as custom shift, scf control (cycles ..etc), ...
    #Cell formatting used in the built in Madelung code
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

    if fourinterp:
        assert version == "Non-SCF", "Fourier interpolation only available for Non-SCF version"

    if version == "Regular":
        nks = get_monkhorst_pack_size(icell, ikpts)
        shift = icell.get_abs_kpts([kshift_rel / n for n in nks])
        if icell.dimension <=2:
            shift[2] =  0
        elif icell.dimension == 1:
            shift[1] = 0
        Nk = np.prod(nks) * 2
        print("Shift is: " + str(shift))
        kmesh_shifted = ikpts + shift
        combined = np.concatenate((ikpts,kmesh_shifted),axis=0)
        print(combined)

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
            print("Iteration number " + str(count_iter))
            print("Madelung:" + str(Madelung))
            print("Eta:" + str(ew_eta))
            if count_iter > 1 and abs(Madelung - prev) < 1e-8:
                conv_Madelung = Madelung
                break
            if count_iter > 30:
                print("Error. Madelung constant not converged")
                break
            ew_eta *= 2
            count_iter += 1
            prev = Madelung

        nocc = mf2.cell.tot_electrons() // 2
        E_stagger_M = E_stagger + nocc * conv_Madelung
        print("One Shot")

    elif version == "Split-SCF":
        #Regular scf calculation
        mfs = scf.KHF(icell, ikpts)
        print(mfs.kernel())
        nks = get_monkhorst_pack_size(mfs.cell, mfs.kpts)
        shift = mfs.cell.get_abs_kpts([kshift_rel / n for n in nks])
        kmesh_shifted = mfs.kpts + shift
        #Calculation on shifted mesh
        mf2 = scf.KHF(icell, kmesh_shifted)
        mf2.with_df = df_type(icell, ikpts).build()  # For 2d,1d, df_type cannot be FFTDF
        print(mf2.kernel())
        dm_2 = mf2.make_rdm1()
        #Get K matrix on shifted kpts, dm from unshifted mesh
        _, Kmat = mf2.get_jk(cell = mf2.cell, dm_kpts = mfs.make_rdm1(), kpts = mfs.kpts, kpts_band = mf2.kpts)
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

            print("Iteration number " + str(count_iter))
            print("Madelung:" + str(Madelung))
            print("Eta:" + str(ew_eta))
            if count_iter > 1 and abs(Madelung - prev) < 1e-8:
                conv_Madelung = Madelung
                break
            if count_iter > 30:
                print("Error. Madelung constant not converged")
                break
            ew_eta *= 2
            count_iter += 1
            prev = Madelung

        nocc = mf2.cell.tot_electrons() // 2
        E_stagger_M = E_stagger + nocc * conv_Madelung

        print("Two Shot")
    else: # Non-SCF
        mf2 = scf.KHF(icell,ikpts, exxdiv='ewald')
        mf2.with_df = df_type(icell, ikpts).build()
        nocc = mf2.cell.tot_electrons()//2

        if dm_kpts is None:
            print(mf2.kernel())
            # Get converged density matrix
            dm_un = mf2.make_rdm1()
        else:
            dm_un = dm_kpts

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
        if dm_kpts is None:
            print("Converged Density Matrix")
        else:
            print("Input density matrix")

        for i in range(0,dm_un.shape[0]):
            print("kpt: " + str(mf2.kpts[i]) + "\n")
            mat = dm_un[i,:,:]
            for j in mat:
                print(' '.join(str(np.real(el)) for el in j))

        # Construct the Fock Matrix
        h1e = get_hcore(mf2, cell=mf2.cell, kpts=kmesh_shifted)
        Jmat, Kmat = mf2.get_jk(cell=mf2.cell, dm_kpts=dm_un, kpts=mf2.kpts, kpts_band=kmesh_shifted,
                    exxdiv='ewald')
        # Veff = Jmat - Kmat/2
        Veff = mf2.get_veff(cell=mf2.cell, dm_kpts=dm_un, kpts=mf2.kpts, kpts_band=kmesh_shifted)
        F_shift = h1e + Veff
        s1e = get_ovlp(mf2, cell=mf2.cell, kpts=kmesh_shifted)
        mo_energy_shift, mo_coeff_shift = mf2.eig(F_shift, s1e)
        mo_occ_shift = mf2.get_occ(mo_energy_kpts=mo_energy_shift, mo_coeff_kpts=mo_coeff_shift)
        dm_shift = mf2.make_rdm1(mo_coeff_kpts=mo_coeff_shift, mo_occ_kpts=mo_occ_shift)

        # Computing the Staggered mesh energy
        Nk = np.prod(nks)
        E_stagger = -1./Nk * np.einsum('kij,kji', dm_shift, Kmat) * 0.5
        E_stagger/=2

        if modified_madelung_params:
            gauss_params = modified_madelung_params.get('gauss_params')
            num_gaussians = modified_madelung_params.get('num_gaussians')
            
            num_gauss_params = int(np.rint(len(gauss_params)/num_gaussians))
            assert np.isclose(np.sum(gauss_params[0::num_gauss_params]), nocc)

            print('Computing Integral terms for Modified Madelung correction')
            chi = 0
            shifted = shift
            for i in range(num_gaussians):
                if num_gauss_params == 4:
                    c_i, sigma_x, sigma_y, sigma_z = gauss_params[i*num_gauss_params:(i+1)*num_gauss_params]
                elif num_gauss_params == 2:
                    c_i, sigma = gauss_params[i*num_gauss_params:(i+1)*num_gauss_params]
                    sigma_x = sigma_y = sigma_z = sigma

                # Detect anisotropy
                anisotropic = False
                if np.abs(sigma_x - sigma_y) < 1e-8 and np.abs(sigma_y - sigma_z) < 1e-8:
                    ew_eta_i = 1./np.sqrt(2.) * np.mean([sigma_x, sigma_y, sigma_z])# TODO: Implement anisotropy
                else:
                    ew_eta_i = 1./np.sqrt(2.) * np.array([sigma_x, sigma_y, sigma_z])
                    anisotropic = True
                # ew_eta = 20
                # ew_eta = 0.219935106676302
                chi_i = madelung_modified(icell, ikpts, shifted, ew_eta=ew_eta_i,anisotropic=anisotropic)
                chi = chi + c_i * chi_i
                print("Term ", i)
                if anisotropic:
                    print(f" Input  sigma x = {sigma_x:.12f}, sigma y = {sigma_y:.12f}, sigma z = {sigma_z:.12f}")
                else:
                    print(f" Input mean sigma: {np.mean([sigma_x, sigma_y, sigma_z]):.12f}")

                print(f" Input ew_eta:     {ew_eta_i:.12f}")
                print(f" Coefficient:      {c_i:.12f}")
                print(f" Chi:              {chi_i:.12f}")
                print(f" Contribution:     {c_i * chi_i:.12f}")

            nocc = mf2.cell.tot_electrons()//2
            E_stagger_M = E_stagger + chi

        else:
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

        if ss_params:
            import pyscf.pbc.scf.ss_localizers as ss_localizers

            # Load default params
            N_local = ss_params.get('nlocal', 3)
            localizer = ss_params.get('localizer')
            H_use_unscaled = ss_params.get('H_use_unscaled', False)
            full_domain = ss_params.get('full_domain', True)
            cart_sphr_split = ss_params.get('cart_sphr_split', True)
            vhR_symm = ss_params.get('vhR_symm', False)
            subtract_nocc = ss_params.get('subtract_nocc', True)
            nufft_gl = ss_params.get('nufft_gl', True)
            n_fft = ss_params.get('n_fft', 350)
            r1_prefactor = ss_params.get('r1_prefactor', 1.0)
            SqG_filenames = ss_params.get('SqG_filenames', None)

            # Extract uKpts from each set of kpts
            # Unshifted
            if mo_coeff_kpts is None:
                raise RuntimeError("mo_coeff_kpts must be provided for fourier interpolation")
            # _, E_madelung1, uKpts1, _, kGrid1 = make_ss_inputs(mf2,mf2.kpts,dm_un, mo_coeff_kpts)
            shiftFac = [0.5]*3
            # _, _, uKpts2, qGrid, kGrid2 = make_ss_inputs(mf2,kmesh_shifted,dm_shift, mo_coeff_shift,   shiftFac=shiftFac)
            E_standard, E_madelung, uKpts1, uKpts2, kGrid1,kGrid2, qGrid = make_ss_inputs_stagger(
                mf2,kpts_i=ikpts,kpts_j=kmesh_shifted,dm_i=dm_un,dm_j=dm_shift, mo_coeff_i=mo_coeff_kpts,
                mo_coeff_j=mo_coeff_shift,shiftFac=shiftFac)

            # Set some parameters
            nkpts = np.prod(nks)
            nocc = mf2.cell.tot_electrons() // 2
            nbands = nocc
            NsCell = mf2.cell.mesh
            nG = np.prod(NsCell)

            Lvec_real = mf2.cell.lattice_vectors()
            L_delta = Lvec_real / NsCell[:, None]
            dvol = np.abs(np.linalg.det(L_delta))

            # Evaluate wavefunction on all real space grid points
            # Establishing real space grid (Generalized for arbitary volume defined by 3 vectors)
            xv, yv, zv = np.meshgrid(np.arange(NsCell[0]), np.arange(NsCell[1]), np.arange(NsCell[2]), indexing='ij')
            mesh_idx = np.hstack([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)])
            rptGrid3D = mesh_idx @ L_delta
            #   Step 1.4: compute the pair product
            Lvec_recip = icell.reciprocal_vectors()
            Gx = np.fft.fftfreq(NsCell[0], d=1 / NsCell[0])
            Gy = np.fft.fftfreq(NsCell[1], d=1 / NsCell[1])
            Gz = np.fft.fftfreq(NsCell[2], d=1 / NsCell[2])
            Gxx, Gyy, Gzz = np.meshgrid(Gx, Gy, Gz, indexing='ij')
            GptGrid3D = np.hstack((Gxx.reshape(-1, 1), Gyy.reshape(-1, 1), Gzz.reshape(-1, 1))) @ Lvec_recip

            # aoval = kmf.cell.pbc_eval_gto("GTO
            # SqG = np.zeros((nkpts, nG), dtype=np.float64)
            cell = mf2.cell
            SqG = build_SqG_k1k2(nkpts, nG, nbands, kGrid1, kGrid2,qGrid, mf2, uKpts1, uKpts2,rptGrid3D, dvol, NsCell, GptGrid3D,nks, debug_options={})

            if subtract_nocc:
                SqG = SqG - nocc  # remove the zero order approximate nocc
                assert (np.abs(np.min(SqG)) -nocc< 1e-4)
            else:
                assert (np.abs(np.min(SqG)) < 1e-4)

            #   Exchange energy can be formulated as
            #   Ex = prefactor_ex * bz_dvol * sum_{q} (\sum_G S(q+G) * 4*pi/|q+G|^2)
            prefactor_ex = -1 / (8 * np.pi ** 3)
            bz_dvol = np.abs(np.linalg.det(Lvec_recip)) / nkpts

            #   Step 3.1: define the local domain as multiple of BZ
            Lvec_recip = icell.reciprocal_vectors()

            LsCell_bz_local = N_local * Lvec_recip
            LsCell_bz_local_norms = np.linalg.norm(LsCell_bz_local, axis=1)

            #   localizer for the local domain
            # r1 = np.min(LsCell_bz_local_norms) / 2
            # r1 = r1_prefactor * r1
            r1, closest_plane_vectors = closest_fbz_distance(Lvec_recip,N_local)

            
            #   reciprocal lattice within the local domain
            if N_local % 2 == 1:
                Grid_1D = np.concatenate((np.arange(0, (N_local - 1) // 2 + 1), np.arange(-(N_local - 1) // 2, 0)))
            else:
                # At low Nlocal/Nk, this matters, because we want the direction where G is incremented to be opposite of 
                # the default direction of a boundary-value q.
                Grid_1D = np.concatenate((np.arange(0, N_local // 2 + 1), np.arange(-N_local // 2 +1, 0)))

            Gxx_local, Gyy_local, Gzz_local = np.meshgrid(Grid_1D, Grid_1D, Grid_1D, indexing='ij')
            GptGrid3D_local = np.hstack(
                (Gxx_local.reshape(-1, 1), Gyy_local.reshape(-1, 1), Gzz_local.reshape(-1, 1))) @ Lvec_recip

            #   location/index of GptGrid3D_local within 'GptGrid3D'
            idx_GptGrid3D_local = []
            for Gl in GptGrid3D_local:
                idx_tmp = np.where(np.linalg.norm(Gl[None, :] - GptGrid3D, axis=1) < 1e-8)[0]
                if len(idx_tmp) != 1:
                    raise TypeError("Cannot locate local G vector in the reciprocal lattice.")
                else:
                    idx_GptGrid3D_local.append(idx_tmp[0])
            idx_GptGrid3D_local = np.array(idx_GptGrid3D_local)

            #   focus on S(q + G) with q in qGrid and G in GptGrid3D_local
            SqG_local = SqG[:, idx_GptGrid3D_local]

            #   Step 3.2: compute the Fourier transform of 1/|q|^2
            nqG_local = N_local * nks  # lattice size along each dimension in the real-space (equal to q + G size)
            Lvec_real_local = Lvec_real / N_local  # dual real cell of local domain LsCell_bz_local

            Rx = np.fft.fftfreq(nqG_local[0], d=1 / nqG_local[0])
            Ry = np.fft.fftfreq(nqG_local[1], d=1 / nqG_local[1])
            Rz = np.fft.fftfreq(nqG_local[2], d=1 / nqG_local[2])
            Rxx, Ryy, Rzz = np.meshgrid(Rx, Ry, Rz, indexing='ij')
            RptGrid3D_local = np.hstack((Rxx.reshape(-1, 1), Ryy.reshape(-1, 1), Rzz.reshape(-1, 1))) @ Lvec_real_local

            if H_use_unscaled:
                r1_unscaled = N_local/2. # in the basis of reciprocal vectors now.
                H = lambda q: localizer(q,r1_prefactor * r1_unscaled)
            else:
                H = lambda q: localizer(q,r1_prefactor * r1)
                
            if full_domain:
                from scipy.optimize import root_scalar

                # Define r1_h and bounds
                r1_h = r1
                # xbounds = LsCell_bzlocal[0] * np.array([-1/2, 1/2])
                # ybounds = LsCell_bzlocal[1] * np.array([-1/2, 1/2])
                # zbounds = LsCell_bzlocal[2] * np.array([-1/2, 1/2])

                # Find the closest boundary
                # min_dir = 0  # Python uses 0-based indexing

                if cart_sphr_split:
                    h_tol = 5e-7 # the Gaussian reaches this value at the closest boundary
                    unit_vec = np.array([r1, 0, 0]) # arbitrary direction

                    # Define the zetafunc
                    # from ss_localizers import localizer_gauss
                    # import pyscf.pbc.scf.ss_localizers as ss_localizers

                    zetafunc = lambda zeta: ss_localizers.localizer_gauss(unit_vec, r1, zeta)- h_tol
                    
                    # Solve for zeta_tol using root finding (equivalent of fzero in MATLAB)
                    result = root_scalar(zetafunc, bracket=[0.1, 10])  # Adjust bracket range if needed
                    if result.converged:
                        zeta_tol = result.root
                        rmult = 1 / zeta_tol
                    else:
                        raise ValueError("Root finding for zetafunc did not converge")
                    # rmult = 0.2
                    # Call the Fourier integration function
                    CoulR = fourier_integration_3d(Lvec_recip, Lvec_real, nks, N_local, r1_h, vhR_symm, True, rmult, RptGrid3D_local, nufft_gl, n_fft)

                else:
                    raise NotImplementedError("Must use cart-sph split")
                    # CoulR = fourier_integration_3d(N, xbounds, ybounds, zbounds, r1_h, True, False, np.nan, RptGrid_Fourier)

            else:   
                raise NotImplementedError("Must use full domain")

                CoulR = 4 * np.pi / normR * sici(normR * r1)[0]
                CoulR[normR < 1e-8] = 4 * np.pi * r1
                
            #   Step 4: Compute the correction

            ss_correction = 0
            #   Integral with Fourier Approximation
            for iq, qpt in enumerate(qGrid):
                qG = qpt[None, :] + GptGrid3D_local
                exp_mat = np.exp(1j * (qG @ RptGrid3D_local.T))
                tmp = (exp_mat @ CoulR) / np.abs(np.linalg.det(LsCell_bz_local))
                if H_use_unscaled:
                    qG_unscaled = qG @ np.linalg.inv(Lvec_recip)
                    tmp = SqG_local[iq, :].T * H(qG_unscaled) * tmp
                else:
                    tmp = SqG_local[iq, :].T * H(qG) * tmp

                ss_correction += np.real(np.sum(tmp))

            int_terms = bz_dvol * prefactor_ex*4*np.pi*ss_correction

            #   Quadrature with Coulomb kernel
            for iq, qpt in enumerate(qGrid):
                qG = qpt[None, :] + GptGrid3D_local
                if H_use_unscaled:
                    qG_unscaled = qG @ np.linalg.inv(Lvec_recip)
                    tmp = SqG_local[iq, :].T * H(qG_unscaled) / np.sum(qG ** 2, axis=1)
                else:
                    tmp = SqG_local[iq, :].T * H(qG) / np.sum(qG ** 2, axis=1)

                tmp[np.isinf(tmp) | np.isnan(tmp)] = 0
                ss_correction -= np.sum(tmp)

            quad_terms = bz_dvol*prefactor_ex*4*np.pi*(ss_correction) - int_terms
            ss_correction = bz_dvol* 4 * np.pi * ss_correction  # Coulomb kernel = 4 pi / |q|^2

            #   Step 5: apply the correction
            if subtract_nocc:
                # e_ex_ss = np.real(E_madelung + prefactor_ex * ss_correction)
                E_stagger_ss = np.real(E_stagger_M + prefactor_ex * ss_correction)

            else:
                # e_ex_ss = np.real(E_standard + prefactor_ex * ss_correction)
                E_stagger_ss = np.real(E_stagger + prefactor_ex * ss_correction)

        results_dict = {
            "E_stagger_M":np.real(E_stagger_M),
            "E_stagger":np.real(E_stagger),
            # "E_madelung":np.real(E_madelung),
            "E_stagger_ss":0,
            "int_term":0,
            "quad_term":0,
        }

        if ss_params:
            results_dict["E_stagger_ss"] = E_stagger_ss
            results_dict["E_madelung"] = E_madelung
            results_dict["ss_correction"] = prefactor_ex*ss_correction
            results_dict["int_term"] = int_terms
            results_dict["quad_term"] = quad_terms
            
        return results_dict




class KHF_stagger(khf.KSCF):
    def __init__(self, mf):
        pass
    def dump_flags(self):
        pass
    def kernel(self):
        pass
    