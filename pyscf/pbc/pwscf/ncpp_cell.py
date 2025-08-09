from pyscf.pbc.gto.cell import Cell
from pyscf.gto.mole import MoleBase
from pyscf.data.elements import ELEMENTS, ELEMENTS_PROTON, \
        _rm_digit, charge, _symbol, _std_symbol, _atom_symbol, is_ghost_atom, \
        _std_symbol_without_ghost
from pyscf.pbc.pwscf.upf import get_nc_data_from_upf
import os


class NCPPCell(Cell):
    def build(self, **kwargs):
        if "pseudo" in kwargs or "ecp" in kwargs:
            raise ValueError("pseudo and ecp not supported")
        if "sg15_path" not in kwargs:
            raise ValueError("sg15_path must be supplied")
        sg15_path = kwargs.pop("sg15_path")
        super().build(**kwargs)

        uniq_atoms = {a[0] for a in self._atom}
        # Unless explicitly input, PP should not be assigned to ghost atoms
        # TODO test ghosts?
        atoms_wo_ghost = [a for a in uniq_atoms if not is_ghost_atom(a)]
        _pseudo = {a: "SG15" for a in atoms_wo_ghost}
        fmt_pseudo = {}
        for atom, atom_pp in _pseudo.items():
            symb = _symbol(atom)
            assert isinstance(symb, str)
            stdsymb = _std_symbol_without_ghost(symb)
            fname = os.path.join(
                sg15_path, f"{stdsymb}_ONCV_PBE-1.2.upf"
            )
            fmt_pseudo[symb] = get_nc_data_from_upf(fname)
        self._pseudo = _pseudo = fmt_pseudo
        self.pseudo = "SG15"

        for ia, atom in enumerate(self._atom):
            symb = atom[0]
            if (symb in _pseudo and
                # skip ghost atoms
                self._atm[ia, 0] != 0):
                self._atm[ia, 0] = _pseudo[symb]["z"]
        self._built = True


if __name__ == "__main__":
    import numpy as np
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
        mf = PWKRKS(cell, kpts, xc="PBE", ekincut=ecut)
        # mf = kpt_symm.KsymAdaptedPWKRKS(cell, kpts, xc="PBE", ekincut=ecut)
        mf.damp_type = "simple"
        mf.damp_factor = 0.7
        mf.nvir = 4 # converge first 4 virtual bands
        mf.kernel()
        mf.dump_scf_summary()
        ens1.append(mf.e_tot)

        mf2 = PWKRKS(nccell, kpts, xc="PBE", ekincut=ecut)
        # mf = kpt_symm.KsymAdaptedPWKRKS(cell, kpts, xc="PBE", ekincut=ecut)
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

    assert(abs(mf.e_tot - -10.673452914596) < 1.e-5)
