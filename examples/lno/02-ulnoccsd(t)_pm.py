import numpy as np
from pyscf import gto, scf, lno, lo, mp, cc
from pyscf.lib import logger
from pyscf.lno import tools
    
# S22-2: water dimer
atom = '''
    O   -1.485163346097   -0.114724564047    0.000000000000
    H   -1.868415346097    0.762298435953    0.000000000000
    H   -0.533833346097    0.040507435953    0.000000000000
    O    1.416468653903    0.111264435953    0.000000000000
    H    1.746241653903   -0.373945564047   -0.758561000000
    H    1.746241653903   -0.373945564047    0.758561000000
'''
basis = 'cc-pvdz'

mol = gto.M(atom=atom, basis=basis, spin=0, verbose=4, max_memory=16000)
mf = scf.UHF(mol).density_fit()
mf.kernel()
frozen = 2

# canonical
mmp = mp.UMP2(mf, frozen=frozen)
mmp.kernel()

mcc = cc.UCCSD(mf, frozen=frozen)
eris = mcc.ao2mo()
mcc.kernel(eris=eris)
eccsd_t = mcc.ccsd_t(eris=eris)

# PM
orbocca = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
orbloca = lo.PipekMezey(mol, orbocca).kernel()
orboccb = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
orblocb = lo.PipekMezey(mol, orboccb).kernel()
orbloc = [orbloca, orblocb]

# LNO
lno_type = ['1h'] * 2
lno_thresh = [1e-4,1e-4]
oa = [[[i],[]] for i in range(orbloca.shape[1])]
ob = [[[],[i]] for i in range(orblocb.shape[1])]
frag_lolist = oa + ob

mlno = lno.ULNOCCSD_T(mf, orbloc, frag_lolist, lno_type=lno_type, lno_thresh=lno_thresh, frozen=frozen)
mlno.lo_proj_thresh_active = None
mlno.verbose_imp = 4
mlno.kernel()
ecc = mlno.e_corr_ccsd
ecc_t = mlno.e_corr_ccsd_t
ecc_pt2corrected = mlno.e_corr_ccsd_pt2corrected(mmp.e_corr)
ecc_t_pt2corrected = mlno.e_corr_ccsd_t_pt2corrected(mmp.e_corr)

log = logger.new_logger(mol)
log.info('lno_thresh = %s\n'
         '    E_corr(CCSD)     = %.10f  rel = %6.2f%%  '
         'diff = % .10f',
         lno_thresh, ecc, ecc/mcc.e_corr*100, ecc-mcc.e_corr)
log.info('    E_corr(CCSD_T)   = %.10f  rel = %6.2f%%  '
         'diff = % .10f',
         ecc_t, ecc_t/(mcc.e_corr+eccsd_t)*100,
         ecc_t-(mcc.e_corr+eccsd_t))
log.info('    E_corr(CCSD+PT2) = %.10f  rel = %6.2f%%  '
         'diff = % .10f',
         ecc_pt2corrected, ecc_pt2corrected/mcc.e_corr*100,
         ecc_pt2corrected - mcc.e_corr)
log.info('    E_corr(CCSD_T+PT2)   = %.10f  rel = %6.2f%%  '
         'diff = % .10f',
         ecc_t_pt2corrected, ecc_t_pt2corrected/(mcc.e_corr+eccsd_t)*100,
         ecc_t_pt2corrected-(mcc.e_corr+eccsd_t))