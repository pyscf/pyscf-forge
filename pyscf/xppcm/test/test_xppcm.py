#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import gto, scf, cc, mcscf
from pyscf.solvent import pcm
from pyscf import xppcm
from pyscf.xppcm import (Rahm, get_z0, get_z, get_eps, get_cavity_volume,
                         get_cavity_volume_atom, calculate_analytical_pressure,
                         get_e_rep, EH_BOHR3_TO_GPA)
from pyscf.xppcm.analytical_cavity import union_of_spheres_volume
from pyscf.xppcm.pcm_derivatives import pol_derivatives

# Cyclohexane as the compressing medium
Z0 = get_z0(36, 0.7781, 84.1595)
EPS_0 = 2.0165
F_0 = 1.30


def setUpModule():
    global mol_be, mol_h2, volume_0
    mol_be = gto.M(atom='Be 0 0 0', basis='cc-pvdz', verbose=0)
    mol_h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
    volume_0 = get_cavity_volume_atom(mol_be, Rahm * F_0)


def tearDownModule():
    global mol_be, mol_h2, volume_0
    del mol_be, mol_h2, volume_0


def be_xp(f, with_pcm=False, conv_tol=1e-11):
    '''RHF with XP (and optionally IEF-PCM) for Be at cavity scaling f.'''
    radii = Rahm * f
    vc = get_cavity_volume_atom(mol_be, radii)
    z = get_z(vc, volume_0, Z0)
    mf = scf.RHF(mol_be).XP(z=z, radii_table=radii)
    if with_pcm:
        mf = mf.PCM()
        mf.with_solvent.method = 'IEF-PCM'
        mf.with_solvent.eps = get_eps(vc, volume_0, EPS_0)
        mf.with_solvent.radii_table = radii
    mf.conv_tol = conv_tol
    mf.kernel()
    return mf, vc


def two_sphere_union_volume(r1, r2, d):
    '''Closed-form volume of the union of two spheres at distance d.'''
    v1 = 4.0 / 3.0 * np.pi * r1**3
    v2 = 4.0 / 3.0 * np.pi * r2**3
    if d >= r1 + r2:
        return v1 + v2
    if d <= abs(r1 - r2):
        return max(v1, v2)
    lens = (np.pi * (r1 + r2 - d)**2
            * (d**2 + 2*d*r2 - 3*r2**2 + 2*d*r1 + 6*r1*r2 - 3*r1**2) / (12*d))
    return v1 + v2 - lens


class KnownValues(unittest.TestCase):
    def test_union_of_spheres_volume(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.]])
        for r1, r2, d in [(1.5, 1.0, 1.2),   # partially overlapping
                          (1.5, 1.0, 3.0),   # disjoint
                          (1.5, 1.0, 0.2),   # nested
                          (2.0, 2.0, 2.0)]:  # equal radii
            coords[1, 0] = d
            v = union_of_spheres_volume(coords, np.array([r1, r2]))
            self.assertAlmostEqual(v, two_sphere_union_volume(r1, r2, d), 9)

    def test_cavity_volume_single_atom(self):
        radii = Rahm * 0.9
        self.assertAlmostEqual(get_cavity_volume(mol_be, radii),
                               get_cavity_volume_atom(mol_be, radii), 9)

    def test_z_and_eps_scaling(self):
        self.assertAlmostEqual(get_z(volume_0, volume_0, Z0), Z0, 12)
        self.assertAlmostEqual(get_eps(volume_0, volume_0, EPS_0), EPS_0, 12)
        # s = (V/V0)^(1/3) = 1/2:  z = z0 s^-(3+eta),  eps - 1 = (eps0 - 1) s^-3
        self.assertAlmostEqual(get_z(volume_0 / 8, volume_0, Z0, eta=6), Z0 * 2**9, 10)
        self.assertAlmostEqual(get_eps(volume_0 / 8, volume_0, EPS_0), 1 + (EPS_0 - 1) * 8, 10)

    def test_xp_scf_energy(self):
        mf_vac = scf.RHF(mol_be).run(conv_tol=1e-11)
        mf, _ = be_xp(0.9)
        self.assertGreater(get_e_rep(mf), 0)
        self.assertGreater(mf.e_tot, mf_vac.e_tot)
        self.assertEqual(mf.__class__.__name__, 'XPRHF')
        # undo_xp() gives back a plain RHF object
        mf_undone = mf.undo_xp()
        self.assertEqual(mf_undone.__class__.__name__, 'RHF')
        self.assertAlmostEqual(mf_undone.run(conv_tol=1e-11).e_tot, mf_vac.e_tot, 9)
        # attaching XP twice is an error
        with self.assertRaises(ValueError):
            mf.XP(z=0.1, radii_table=Rahm)

    def test_analytical_vs_numerical_pressure_xp(self):
        f, h = 0.94, 0.005
        mf, vc = be_xp(f)
        p_ana = calculate_analytical_pressure(mf, vc)
        (mfp, vp), (mfm, vm) = be_xp(f + h), be_xp(f - h)
        p_num = -(mfp.e_tot - mfm.e_tot) / (vp - vm) * EH_BOHR3_TO_GPA
        self.assertGreater(p_ana, 10.0)
        self.assertAlmostEqual(p_ana, p_num, delta=0.1)

    def test_analytical_vs_numerical_pressure_xppcm(self):
        # z and eps both follow the cavity volume, so the finite difference
        # probes the repulsion, surface, polarisation and dielectric terms.
        f, h = 0.94, 0.005
        mf, vc = be_xp(f, with_pcm=True)
        terms = calculate_analytical_pressure(mf, vc, return_terms=True)
        (mfp, vp), (mfm, vm) = be_xp(f + h, with_pcm=True), be_xp(f - h, with_pcm=True)
        p_num = -(mfp.e_tot - mfm.e_tot) / (vp - vm) * EH_BOHR3_TO_GPA
        self.assertAlmostEqual(terms['total'], p_num, delta=0.1)
        self.assertAlmostEqual(
            terms['total'],
            terms['repulsion'] + terms['surface'] + terms['polarisation'] + terms['dielectric'], 9)
        self.assertNotEqual(terms['polarisation'], 0.0)

    def test_pol_derivatives_vs_finite_difference(self):
        mf, _ = be_xp(0.94, with_pcm=True)
        pcmobj = mf.with_solvent
        dm = mf.make_rdm1()
        de_dlnf, de_deps = pol_derivatives(pcmobj, dm)

        def e_pol(scale, eps):
            p = pcm.PCM(mol_be)
            p.method = pcmobj.method
            p.eps = eps
            p.radii_table = pcmobj.radii_table * scale
            p.lebedev_order = pcmobj.lebedev_order
            p.build()
            return p._get_vind(dm)[0]

        h = 1e-4
        fd_dlnf = (e_pol(np.exp(h), pcmobj.eps) - e_pol(np.exp(-h), pcmobj.eps)) / (2 * h)
        fd_deps = (e_pol(1.0, pcmobj.eps + h) - e_pol(1.0, pcmobj.eps - h)) / (2 * h)
        self.assertAlmostEqual(de_dlnf, fd_dlnf, 8)
        self.assertAlmostEqual(de_deps, fd_deps, 9)

    def test_post_hf_two_electron(self):
        # For two electrons CCSD is exact, so its density, E_rep and pressure
        # must coincide with those of FCI (CASCI with every orbital active).
        radii = Rahm * 0.9
        vc = get_cavity_volume(mol_h2, radii)
        mf = scf.RHF(mol_h2).XP(z=0.3, radii_table=radii).run(conv_tol=1e-11)
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-10
        mycc.conv_tol_normt = 1e-8
        mycc.kernel()
        myfci = mcscf.CASCI(mf, mol_h2.nao, mol_h2.nelec).run()
        self.assertAlmostEqual(mycc.e_tot, myfci.e_tot, 7)
        self.assertAlmostEqual(get_e_rep(mycc), get_e_rep(myfci), 7)
        self.assertAlmostEqual(calculate_analytical_pressure(mycc, vc),
                               calculate_analytical_pressure(myfci, vc), 5)
        # correlation lowers the exterior density, hence E_rep
        self.assertLess(get_e_rep(myfci), get_e_rep(mf))

    def test_xpcav_energy_terms(self):
        radii = Rahm * 0.9
        z, p_gpa = 0.3, 5.0
        mf_xp = scf.RHF(mol_h2).XP(z=z, radii_table=radii).run(conv_tol=1e-11)
        mf_cav = scf.RHF(mol_h2).XPCav(z=z, radii_table=radii, p_gpa=p_gpa).run(conv_tol=1e-11)
        e_cav = p_gpa / EH_BOHR3_TO_GPA * get_cavity_volume(mol_h2, radii)
        self.assertAlmostEqual(mf_cav.get_e_cav(), e_cav, 10)
        self.assertAlmostEqual(mf_cav.e_tot, mf_xp.e_tot + e_cav, 9)
        self.assertAlmostEqual(mf_cav.energy_nuc(), mol_h2.energy_nuc() + e_cav, 10)
        with self.assertRaises(ValueError):
            scf.RHF(mol_h2).XPCav(z=z, radii_table=radii)   # p_gpa is required
        with self.assertRaises(NotImplementedError):
            mf_xp.nuc_grad_method()                          # XP has no gradient

    def test_xpcav_gradient_vs_finite_difference(self):
        radii = Rahm * 0.9
        z, p_gpa, grid_level = 0.3, 5.0, 7

        def run(coords):
            m = gto.M(atom=[['H', c] for c in coords], basis='sto-3g', unit='Bohr', verbose=0)
            mf = scf.RHF(m).XPCav(z=z, radii_table=radii, p_gpa=p_gpa, grid_level=grid_level)
            mf.conv_tol = 1e-12
            mf.kernel()
            return mf

        coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
        mf = run(coords)
        g = mf.nuc_grad_method()
        g.verbose = 0
        de = g.kernel()
        self.assertEqual(de.shape, (2, 3))
        # translational invariance
        self.assertAlmostEqual(abs(de.sum(axis=0)).max(), 0.0, 6)
        # central finite difference; the energy carries grid noise (see the
        # XPCav docstring), hence the loose tolerance
        h = 2e-3
        cp, cm = coords.copy(), coords.copy()
        cp[1, 2] += h
        cm[1, 2] -= h
        fd = (run(cp).e_tot - run(cm).e_tot) / (2 * h)
        self.assertAlmostEqual(de[1, 2], fd, delta=3e-4)

    def test_pressure_requires_xp(self):
        mf = scf.RHF(mol_be).run()
        with self.assertRaises(ValueError):
            calculate_analytical_pressure(mf, 1.0)


if __name__ == "__main__":
    print("Full Tests for xppcm")
    unittest.main()
