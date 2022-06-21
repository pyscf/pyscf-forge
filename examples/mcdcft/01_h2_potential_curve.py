import pyscf
from pyscf import scf, gto, mcscf, lib
import numpy as np
#from pyscf.fci import csf_solver
from pyscf.mcdcft import mcdcft

'''
This input file performs a potential energy scan of H2 using cBLYP (density
coherence functional without reparameterization).
'''

def run(r, chkfile=None, mo_coeff=None):
    r /= 2
    mol = gto.M(atom=f'H  0 0 {r}; H 0 0 -{r}', basis='cc-pvtz', 
          symmetry=False, verbose=3, unit='angstrom')
    mf = scf.RHF(mol)
    mf.kernel()
    mc = mcdcft.CASSCF(mf, 'BLYP', 2, 2, ot_name='cBLYP', 
                       grids_level=6)
    #mc.fcisolver = csf_solver(mol, smult=1)
    mc.fix_spin_(ss=0)
    if mo_coeff is not None:
        mo_coeff = mcscf.addons.project_init_guess(mc, mo_coeff, priority=[[0,1]])
    if chkfile is not None:
        mc.chkfile = chkfile
    mc.kernel(mo_coeff=mo_coeff)
    return mc


def scan():
    mo_coeff = None
    method = 'MC-DCFT'

    rrange = np.arange(0.5, 6.1, 0.1)

    for r in rrange:
        chkfile = f'{method}_{r:02.2f}.chk'
        mc = run(r, chkfile=chkfile, mo_coeff=mo_coeff)
        if mc and mc.converged:
            mo_coeff = mc.mo_coeff
            e_tot = mc.e_tot
            mc.dump_mcdcft_chk(chkfile)
        else:
            print(f"Warning: bond distance {r:02.2f} not converged")


if __name__ == '__main__':
    lib.num_threads(2)
    scan()

