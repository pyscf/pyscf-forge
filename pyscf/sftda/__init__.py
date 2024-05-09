from pyscf import scf
from pyscf.lib import logger

try:
    from pyscf.dft import KohnShamDFT
    from pyscf.sftda import uks_sf
    from pyscf.sftda.uks_sf import TDUKS_SF
except (ImportError, IOError):
    pass

def TDA_SF(mf):
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        if isinstance(mf, KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.TDA_SF()

def TDDFT_SF(mf):
    print('Warning!!! SF-TDDFT ruining in the slow divergence, ' + \
          'you can choose get_ab_sf() to construct the full matrix ' + \
          'to obtain the excited energies.')
    return mf.TDDFT_SF()
