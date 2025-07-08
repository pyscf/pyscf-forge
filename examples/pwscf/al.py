from pyscf.pbc import gto, scf, mp
from pyscf.gto.basis import parse_nwchem, parse_cp2k_pp
from pyscf.pbc.gto.cell import fromfile
from pyscf.pbc.pwscf import khf, kuhf, krks, kuks
from pyscf.pbc.pwscf.smearing import smearing_
from pyscf.pbc.pwscf import kpt_symm

def get_basis(atoms):
    basname = 'dz'
    fbas = '../2022_data_for_paper_cc_al_li/basis_sets/%s.dat' % basname
    basis = {atm : parse_nwchem.load(fbas, atm) for atm in atoms}
    return basis


cell = gto.Cell()
a, atom = fromfile('Al.poscar', None)
cell.a = a
cell.set_geom_(atom, unit='Angstrom', inplace=True)
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.verbose = 4  # 6
cell.space_group_symmetry = True
cell.symmorphic = True
cell.max_memory = 200000
cell.precision = 1e-8
cell.exp_to_discard = 0.1
cell.mesh = [24, 24, 24]
cell.build()

kpts = cell.make_kpts(
    [3, 3, 3],
    # scaled_center=[0.6223, 0.2953, 0.0000],
)

spinpol = False
sym = True
if False:
    if spinpol:
        kmf = kuhf.PWKUHF(cell, kpts)
    else:
        kmf = khf.PWKRHF(cell, kpts)
    kmf = smearing_(kmf, sigma=0.01, method='gauss')
    kmf.xc = "PBE"
    kmf.nvir = 3
    kmf.conv_tol = 1e-7
    kmf.conv_tol_grad = 2e-3
    kmf.kernel()
    kmf.dump_scf_summary()
else:
    if sym:
        ks = cell.make_kpts([3,3,3], time_reversal_symmetry=True, space_group_symmetry=True)
        if spinpol:
            kmf = kpt_symm.KsymAdaptedPWKUKS(cell, ks)
        else:
            kmf = kpt_symm.KsymAdaptedPWKRKS(cell, ks)
    else:
        if spinpol:
            kmf = kuks.PWKUKS(cell, kpts)
        else:
            kmf = krks.PWKRKS(cell, kpts)
    kmf = smearing_(kmf, sigma=0.01, method='gauss')
    kmf.xc = "PBE"
    kmf.nvir = 3
    kmf.conv_tol = 1e-7
    kmf.conv_tol_grad = 2e-3
    kmf.kernel()
    kmf.dump_scf_summary()

