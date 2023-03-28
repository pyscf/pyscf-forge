#!/usr/bin/env python
# Copyright 2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
WFX File format

Ref.
http://aim.tkgristmill.com/wfxformat.html
'''

__all__ = ['from_mo', 'from_scf', 'from_mcscf', 'from_chkfile', 'load']

import sys
import contextlib
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf.scf.uhf import UHF
from pyscf.pbc.gto import Cell
from pyscf.pbc.scf.khf import KSCF
from pyscf.tools.wfn_format import TYPE_MAP


def make_tagger(fout):
    @contextlib.contextmanager
    def tagger(session):
        fout.write('<%s>\n' % session)
        yield
        fout.write('</%s>\n' % session)
    return tagger

def write_mo(fout, mol, mo_coeff, mo_energy=None, mo_occ=None):
    if not isinstance(mo_coeff, np.ndarray) or mo_coeff.ndim == 3:
        raise NotImplementedError('WFX for UHF orbitals')

    mol, ctr = mol.decontract_basis()
    mo_coeff = np.dot(scipy.linalg.block_diag(*ctr), mo_coeff)
    if not mol.cart:
        mo_coeff = mol.cart2sph_coeff().dot(mo_coeff)

    nmo = mo_coeff.shape[1]
    centers = []
    types = []
    exps = []
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        es = mol.bas_exp(ib)
        for t in TYPE_MAP[l]:
            types.append(t)
        ncart = (l + 1) * (l + 2) // 2
        exps.extend([es[0]]*ncart)
        centers.extend([ia+1]*ncart)
    centers = np.array(centers)
    types = np.array(types)
    exps = np.array(exps)
    nprim, nmo = mo_coeff.shape
    neleca, nelecb = mol.nelec

    tag = make_tagger(fout)

    with tag('Title'):
        fout.write(' From PySCF\n')

    with tag('Keywords'):
        fout.write(' GTO\n')
    with tag('Number of Nuclei'):
        fout.write(' %s\n' % mol.natm)
    with tag('Number of Perturbations'):
        fout.write(' 0\n')
    with tag('Net Charge'):
        fout.write(' %g\n' % mol.charge)
    with tag('Number of Electrons'):
        fout.write(' %d\n' % mol.nelectron)
    with tag('Number of Alpha Electrons'):
        fout.write(' %d\n' % neleca)
    with tag('Number of Beta Electrons'):
        fout.write(' %d\n' % nelecb)
    with tag('Electronic Spin Multiplicity'):
        fout.write(' %d\n' % mol.multiplicity)

    if mol.has_ecp():
        with tag('Number of Core Electrons'):
            ncore = sum(mol.atom_nelec_core(i) for i in range(mol.natm))
            fout.write(' %d\n' % ncore)

    with tag('Nuclear Names'):
        for i, symb in enumerate(mol.elements):
            fout.write(' %s%d\n' % (symb, i+1))
    with tag('Atomic Numbers'):
        for symb in mol.elements:
            fout.write(' %d\n' % gto.mole.charge(symb))
    with tag('Nuclear Charges'):
        for c in mol.atom_charges():
            fout.write(' %19.12e\n' % c)
    with tag('Nuclear Cartesian Coordinates'):  # in Bohrs
        for c in mol.atom_coords():
            fout.write(' %19.12e %19.12e %19.12e\n' % tuple(c))

    if isinstance(mol, Cell):
        cell = mol
        with tag('Number of Translation Vectors'):
            fout.write(' %d\n' % cell.dimension)
        with tag('Translation Vectors'):
            a = cell.lattice_vectors()
            for r in a[:cell.dimension]:
                fout.write(' %15.8e %15.8e %15.8e\n' % (r[0], r[1], r[2]))

    with tag('Number of Primitives'):
        fout.write(' %d\n' % nprim)
    with tag('Primitive Centers'):
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%19d'%x for x in centers[i0:i1]))
    with tag('Primitive Types'):
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%19d'%x for x in types[i0:i1]))
    with tag('Primitive Exponents'):
        for i0, i1 in lib.prange(0, nprim, 5):
            fout.write(' %s\n' % ' '.join('%19.12e'%x for x in exps[i0:i1]))

    #TODO: Core density
    #with tag('Additional Electron Density Function (EDF)'):
    #    with tag('Number of EDF Primitives'):
    #        pass
    #    with tag('EDF Primitive Centers'):
    #        pass
    #    with tag('EDF Primitive Types'):
    #        pass
    #    with tag('EDF Primitive Exponents'):
    #        pass
    #    with tag('EDF Primitive Coefficients'):
    #        pass


    with tag('Number of Occupied Orbitals'):
        fout.write(' %d\n' % nmo)
    if mo_occ is not None:
        with tag('Molecular Orbital Occupation Numbers'):
            for c in mo_occ:
                fout.write(' %19.12e\n' % c)
    if mo_energy is not None:
        with tag('Molecular Orbital Energies'):
            for e in mo_energy:
                fout.write(' %19.12e\n' % e)
    with tag('Molecular Orbital Spin Types'):
        for k in range(nmo):
            fout.write(' Alpha and Beta\n')

    with tag('Molecular Orbital Primitive Coefficients'):
        for k in range(nmo):
            with tag('MO Number'):
                fout.write(' %d\n' % (k+1))

            for i0, i1 in lib.prange(0, nprim, 5):
                fout.write(' %s\n' % ' '.join('%19.12e'%x for x in mo_coeff[i0:i1,k]))

def from_mo(mol, filename=None, mo_coeff=None, ene=None, occ=None):
    '''Dump orbitals in WFX format'''
    if filename is None:
        write_mo(sys.stdout, mol, mo_coeff, mo_energy=ene, mo_occ=occ)
    else:
        with open(filename, 'w') as fout:
            write_mo(fout, mol, mo_coeff, mo_energy=ene, mo_occ=occ)

def from_scf(mf, filename=None):
    '''Dump an SCF object in WFX format'''
    if isinstance(mf, UHF):
        raise NotImplementedError('UHF/UKS object')

    if filename is None:
        fout = sys.stdout
    else:
        fout = open(filename, 'w')

    tag = make_tagger(fout)

    if isinstance(mf, KSCF):
        nkpts = len(mf.kpts)
        weight = 1. / nkpts
        mo_energy_kpts = np.hstack(mf.mo_energy)
        mo_occ_kpts = np.hstack(mf.mo_occ)
        k_index = np.hstack([np.repeat(i+1, c.size)
                             for i, c in enumerate(mf.mo_occ)])
        idx = mo_energy_kpts.argsort()

        write_mo(fout, mf.cell, np.hstack(mf.mo_coeff),
                 mo_energy_kpts[idx], mo_occ_kpts[idx])

        with tag('Orbital Type'):
            fout.write('canonical natural\n')
        with tag('Type of Orbital Primitive Coefficients'):
            fout.write(' complex\n')

        with tag('Number of Kpoints'):
            fout.write(' %d\n' % nkpts)
        with tag('Kpoint Weights'):
            for i in range(nkpts):
                fout.write(' %g\n' % weight)
        with tag('Kpoint Index for Orbitals'):
            for i in idx:
                fout.write(' %d\n' % k_index[i])
        with tag('Kpoint Fractional Coordinates'):
            dimension = mf.cell.dimension
            kpts = mf.cell.get_scaled_kpts(mf.kpts)
            for k in kpts[:,:dimension]:
                fout.write((' %g'*dimension) % tuple(k) + '\n')
    else:
        write_mo(fout, mf.mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ)
        with tag('Orbital Type'):
            fout.write(' canonical natural\n')
        with tag('Type of Orbital Primitive Coefficients'):
            fout.write(' real\n')
        # TODO:
        with tag('Virial Ratio (-V/T)'):
            fout.write(' 0.000000000000e+00\n')

    with tag('Energy = T + Vne + Vee + Vnn'):
        fout.write( '%19.12e\n' % mf.e_tot)

    if filename is not None:
        fout.close()

def from_mcscf(mc, filename=None, cas_natorb=False):
    '''Dump an MCSCF object in WFX format'''
    raise NotImplementedError

def from_chkfile(filename=None, chkfile=None):
    '''Read HF/DFT results from chkfile and dump them in WFX format'''
    mol, mfdic = scf.chkfile.load_scf(chkfile)
    mf = mol.RHF()
    mf.__dict__.update(mfdic)
    from_scf(mf, filename)

def load(wfx_file, verbose=0):
    '''Read SCF results from a WFX file then construct mol and SCF objects
    '''
    raise NotImplementedError

if __name__ == '__main__':
    from pyscf import mcscf, symm
    mol = gto.M(atom='N 0 0 0; N 0 0 2.88972599',
                unit='B',
                basis='ccpvdz',
                symmetry=True)
    mf = mol.RHF(mol).run()
    from_scf(mf)
