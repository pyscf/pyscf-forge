# Requires a separate JAX installation.

import os

os.environ["OMP_NUM_THREADS"] = (
    "1"  # For reproducibility. LNO makes slightly different orbitals every time.
)
import sys

import numpy as np
from pyscf.data.elements import chemcore

from pyscf import gto, mp, scf
from pyscf.afqmc.lnoafqmc import LNOAFQMC, prep_local_orbitals
from pyscf.lib import logger

log = logger.Logger(sys.stdout, 6)


# water dimer
atom = """
O   -1.485163346097   -0.114724564047    0.000000000000
H   -1.868415346097    0.762298435953    0.000000000000
H   -0.533833346097    0.040507435953    0.000000000000
O    1.416468653903    0.111264435953    0.000000000000
H    1.746241653903   -0.373945564047   -0.758561000000
H    1.746241653903   -0.373945564047    0.758561000000
"""

basis = "631g"

mol = gto.M(atom=atom, basis=basis)
mol.verbose = 3
frozen = chemcore(mol)

mf = scf.RHF(mol).density_fit()
mf.kernel()  # type: ignore

# canonical
mmp = mp.MP2(mf, frozen=frozen)
mmp.kernel()
efull_mp2 = mmp.e_corr

lo_coeff, frag_lolist = prep_local_orbitals(mf, frozen=frozen)
# frag_lolist = [[0]]  # One can run a particular fragment like this
# LNO-AFQMC calculation: here we can scan over a list of thresholds
mcc = LNOAFQMC(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
mcc.seed = 1234
mcc.n_blocks = 100
mcc.n_walkers = 200

gamma = 10  # thresh_occ / thresh_vir
threshs = np.asarray([1e-5])
elno_afqmc_uncorr = np.zeros_like(threshs)
stocherr_afqmc = np.zeros_like(threshs)
elno_mp2 = np.zeros_like(threshs)
elno_afqmc = np.zeros_like(threshs)
for i, thresh in enumerate(threshs):
    mcc.lno_thresh = [thresh * gamma, thresh]
    mcc.kernel()
    stocherr_afqmc[i] = mcc.afqmc_error_ecorr
    elno_afqmc_uncorr[i] = mcc.e_corr_afqmc
    elno_mp2[i] = mcc.e_corr_pt2
    elno_afqmc[i] = mcc.e_corr_afqmc - elno_mp2[i] + efull_mp2

log.info("")
for i, thresh in enumerate(threshs):
    e0 = elno_afqmc_uncorr[i]
    err = stocherr_afqmc[i]
    e1 = elno_afqmc[i]
    log.info(
        "thresh = %.3e  E_corr(LNO-AFQMC) = %.15g +/- %.15g  E_corr(LNO-AFQMC+∆PT2) = %.15g +/- %.15g",
        thresh,
        e0,
        err,
        e1,
        err,
    )
