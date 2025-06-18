from pyscf.pbc.scf.addons import _SmearingKSCF
from pyscf.pbc.pwscf import khf
from pyscf.lib import logger
from pyscf import lib
from pyscf import __config__
import numpy as np


SMEARING_METHOD = getattr(__config__, 'pbc_scf_addons_smearing_method', 'fermi')


def smearing(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    '''Fermi-Dirac or Gaussian smearing'''
    if not isinstance(mf, khf.PWKRHF):
        raise ValueError("For PW mode only")

    if isinstance(mf, _SmearingPWKSCF):
        mf.sigma = sigma
        mf.smearing_method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    return lib.set_class(_SmearingPWKSCF(mf, sigma, method, mu0, fix_spin),
                         (_SmearingPWKSCF, mf.__class__))


def smearing_(mf, *args, **kwargs):
    mf1 = smearing(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf


def _occ_from_C(C_ks):
    nkpts = len(C_ks)
    if nocc == 0:
        mocc_ks = [np.zeros(get_kcomp(C_ks,k,load=False).shape[0])
                   for k in range(nkpts)]
    else:
        mocc_ks = [None] * nkpts
        for k in range(nkpts):
            C_k = get_kcomp(C_ks, k, load=False)
            mocc_ks[k] = np.asarray([min(2, max(0, nocc - i))
                                     for i in range(C_k.shape[0])])
    return mocc_ks


class _SmearingPWKSCF(_SmearingKSCF):

    def get_mo_occ(self, moe_ks=None, C_ks=None, nocc=None):
        cell = self.cell
        if nocc is None:
            nocc = cell.nelectron / 2.0
        else:
            assert nocc == cell.nelectron / 2.0
        if moe_ks is not None:
            mocc_ks = self.get_occ(mo_energy_kpts=np.array(moe_ks))
            print("OCC CALC", self.istype("KUHF"), self.istype("PWKUHF"), np.sum(mocc_ks), np.max(mocc_ks), np.asarray(mocc_ks).size)
            if self.istype("KUHF") or self.istype("PWKUHF"):
                mocc_ks = [[2 * occ for occ in mocc_ks[0]], [2 * occ for occ in mocc_ks[1]]]
        elif C_ks is not None:
            if self.istype("KUHF") or self.istype("PWKUHF"):
                mocc_ks = [_occ_from_C(C_ks[0]), _occ_from_C(C_ks[1])]
            else:
                mocc_ks = _occ_from_C(C_ks)
        else:
            raise RuntimeError

        return mocc_ks

    def energy_tot(self, C_ks, mocc_ks, moe_ks=None, mesh=None, Gv=None,
                   vj_R=None, exxdiv=None):
        e_tot = khf.PWKRHF.energy_tot(self, C_ks, mocc_ks, moe_ks=moe_ks,
                                      mesh=mesh, Gv=Gv, vj_R=vj_R,
                                      exxdiv=exxdiv)
        if self.sigma and self.smearing_method and self.entropy is not None:
            self.e_free = e_tot - self.sigma * self.entropy
            self.e_zero = e_tot - self.sigma * self.entropy * .5
            logger.info(self, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                        e_tot, self.e_free, self.e_zero)
        return e_tot

