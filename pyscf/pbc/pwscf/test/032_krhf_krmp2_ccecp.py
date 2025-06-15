""" Check PW-KRHF, PW-KRMP2 and read init guess from chkfile
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, pwscf
from pyscf import lib


if __name__ == "__main__":
    kmesh = [2,1,1]
    ke_cutoff = 30
    pseudo = "ccecp"
    atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
    a = np.asarray(
        [[0.       , 1.78339987, 1.78339987],
        [1.78339987, 0.        , 1.78339987],
        [1.78339987, 1.78339987, 0.        ]])

# cell
    cell = gto.Cell(
        atom=atom,
        a=a,
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff
    )
    cell.build()
    cell.verbose = 6

# kpts
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

# tempfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    chkfile = swapfile.name
    swapfile = None

# krhf
    pwmf = pwscf.KRHF(cell, kpts)
    pwmf.nvir = 10 # request 10 virtual states
    pwmf.chkfile = chkfile
    pwmf.kernel(save_ccecp_kb=True)

    e_tot0 = -10.6261884956522
    assert(abs(pwmf.e_tot - e_tot0) < 1.e-6)

# krhf init from chkfile
    pwmf.init_guess = "chkfile"
    pwmf.kernel()

    assert(abs(pwmf.e_tot - e_tot0) < 1.e-6)

# input C0
    pwmf.kernel(C0=pwmf.mo_coeff)

    assert(abs(pwmf.e_tot - e_tot0) < 1.e-6)

# krmp2
    pwmp = pwscf.KMP2(pwmf)
    pwmp.kernel()

    assert(abs(pwmp.e_corr - -0.136781915070538) < 1.e-4)
