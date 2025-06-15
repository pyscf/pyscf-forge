from pyscf import __config__
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.pwscf import khf, kuhf, krks
import numpy as np


class PWKUKS(krks.PWKohnShamDFT, kuhf.PWKUHF):

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kuhf.PWKUHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        krks.PWKohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kuhf.PWKUHF.dump_flags(self)
        krks.PWKohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        out = self._transfer_attrs_(kuhf.PWKUHF(self.cell, self.kpts))
        # TODO might need to setup up ACE here if xc is not hybrid
        return out

    def get_mo_energy(self, C_ks, mocc_ks, mesh=None, Gv=None, exxdiv=None,
                      vj_R=None, ret_mocc=True, full_ham=False):
        if vj_R is None: vj_R = self.get_vj_R(C_ks, mocc_ks)
        res = kuhf.PWKUHF.get_mo_energy(self, C_ks, mocc_ks, mesh=mesh, Gv=Gv,
                                        exxdiv=exxdiv, vj_R=vj_R,
                                        ret_mocc=ret_mocc, full_ham=full_ham)
        if ret_mocc:
            moe_ks = res[0]
        else:
            moe_ks = res
        moe_ks[0][0] = lib.tag_array(moe_ks[0][0], xcdiff=vj_R.exc-vj_R.vxcdot)
        return res

    def energy_elec(self, C_ks, mocc_ks, mesh=None, Gv=None, moe_ks=None,
                    vj_R=None, exxdiv=None):
        e_scf = kuhf.PWKUHF.energy_elec(self, C_ks, mocc_ks, moe_ks=moe_ks,
                                        mesh=mesh, Gv=Gv, vj_R=vj_R,
                                        exxdiv=exxdiv)
        # When energy is computed from the orbitals, we need to account for
        # the different between \int vxc rho and \int exc rho.
        if moe_ks is not None:
            e_scf += moe_ks[0][0].xcdiff
        return e_scf

    def update_k(self, C_ks, mocc_ks):
        ni = self._numint
        if ni.libxc.is_hybrid_xc(self.xc):
            super().update_k(C_ks, mocc_ks)
        elif "t-ace" not in self.scf_summary:
            self.scf_summary["t-ace"] = np.zeros(2)


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

    umf = PWKUKS(cell, kpts, xc="PBE0")
    umf.damp_type = "simple"
    umf.damp_factor = 0.7
    umf.nvir = [0,2]
    umf.nvir_extra = 4
    umf.kernel()

    umf.dump_scf_summary()

    assert(abs(umf.e_tot - -5.39994570429868) < 1e-5)

