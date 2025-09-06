from pyscf.pbc.gto.cell import Cell
from pyscf.pbc.pwscf.ncpp_cell import NCPPCell
from pyscf.data.elements import ELEMENTS, ELEMENTS_PROTON, \
        _rm_digit, charge, _symbol, _std_symbol, _atom_symbol, is_ghost_atom, \
        _std_symbol_without_ghost
from pyscf.pbc.pwscf.upf import get_nc_data_from_upf
import numpy as np
import os



if __name__ == "__main__":
    from pyscf.pbc import gto
    from pyscf.pbc.pwscf.krks import PWKRKS

    kwargs = dict(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
        #symmorphic=True,
        #space_group_symmetry=True,
        verbose=6,
    )

    cell = gto.Cell(**kwargs)
    cell.build()

    kwargs.pop("pseudo")
    nccell = NCPPCell(**kwargs)
    nccell.build(sg15_path="../../gpaw_data/sg15_oncv_upf_2020-02-06/")

    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(
        kmesh,
        #time_reversal_symmetry=True,
        #space_group_symmetry=True,
    )

    # from pyscf.pbc.pwscf import kpt_symm

    ens1 = []
    ens2 = []
    ecuts = [18.38235294, 22.05882353, 25.73529412, 29.41176471, 33.08823529,
             36.76470588, 44.11764706, 55.14705882, 73.52941176, 91.91176471]
    for ecut in ecuts:
        print("\n")
        print("ECUT", ecut)
        mf = PWKRKS(cell, kpts, xc="PBE", ecut_wf=ecut)
        # mf = kpt_symm.KsymAdaptedPWKRKS(cell, kpts, xc="PBE", ecut_wf=ecut)
        mf.damp_type = "simple"
        mf.damp_factor = 0.7
        mf.nvir = 4 # converge first 4 virtual bands
        mf.kernel()
        mf.dump_scf_summary()
        ens1.append(mf.e_tot)

        mf2 = PWKRKS(nccell, kpts, xc="PBE", ecut_wf=ecut)
        # mf = kpt_symm.KsymAdaptedPWKRKS(cell, kpts, xc="PBE", ecut_wf=ecut)
        mf2.damp_type = "simple"
        mf2.damp_factor = 0.7
        mf2.nvir = 4 # converge first 4 virtual bands
        mf2.init_pp()
        mf2.init_jk()
        # mf2.energy_tot(C_ks=mf.mo_coeff, mocc_ks=mf.mo_occ)
        mf2.kernel()
        ens2.append(mf2.e_tot)
        mf2.dump_scf_summary()
    print()
    for ens in [ens1, ens2]:
        print(ens)
        print(ecuts[:-1])
        print(27.2 * (np.array(ens[:-1]) - ens[-1]))
        print()
