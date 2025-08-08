#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

""" UPF file parser. Currently just for parsing SG15 pseudos.
"""

import xml.etree.ElementTree as ET
import numpy as np
from math import factorial as fac


def _parse_array_upf(entry, dtype=float):
    return np.fromstring(entry.text, dtype=dtype, sep=' ')


def get_nc_data_from_upf(fname):
    tree = ET.parse(fname)
    root = tree.getroot()
    pp_local = root.find('PP_LOCAL')
    pp_local = _parse_array_upf(pp_local)
    mesh_dat = root.find('PP_MESH')
    pp_r = _parse_array_upf(mesh_dat.find('PP_R'))
    pp_dr = _parse_array_upf(mesh_dat.find('PP_RAB'))
    pp_nl = root.find('PP_NONLOCAL')
    dij = None
    projectors = []
    for child in pp_nl:
        if child.tag == "PP_DIJ":
            dij = _parse_array_upf(child)
        else:
            proj_index = int(child.attrib["index"]) - 1
            l = int(child.attrib["angular_momentum"])
            cutoff_index = int(child.attrib["cutoff_radius_index"])
            projector = _parse_array_upf(child)
            projectors.append({
                "n": proj_index,
                "l": l,
                "cut": cutoff_index,
                "rproj": projector,
                "kproj": fft_upf(pp_r, projector, l, mul_by_r=False)[1]
            })
    assert dij is not None
    dij = dij.reshape(len(projectors), len(projectors))
    _deriv = make_radial_derivative_calculator(pp_r, 2, 2)[0]
    d1 = _deriv(pp_local * pp_r)
    charge = d1 / pp_r
    charge[0] = charge[1]
    pp_k, chargek = fft_upf(pp_r, charge, 0)
    chargek[:] /= 4 * np.pi
    locpotk = chargek * 4 * np.pi / pp_k**2
    if False:
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        plt.plot(pp_k, chargek)
        plt.show()
    assert (np.diag(np.diag(dij)) == dij).all(), "dij must be diagonal"
    return {
        "z": int(round(float(root.find("PP_HEADER").attrib["z_valence"]))),
        "projectors": projectors,
        "dij": 0.5 * dij,  # convert to Ha
        "local_part": {
            "real": charge,
            "recip": chargek,
            "finite_g0": 2 * (chargek[1] - chargek[0]) / (pp_k[1] - pp_k[0]),
            "locpotk": locpotk,
        },
        "grids": {
            "r": pp_r,
            "dr": pp_dr,
            "k": pp_k,
        }
    }


def _get_deriv_weights(r_g, D, i, istart, deriv_order):
    y = np.zeros(D)
    diffs = np.empty((D, D))
    y[deriv_order] = 1
    rc = r_g[i]
    for j in range(D):
        r = r_g[istart + j]
        for k in range(D):
            diffs[k, j] = (r - rc) ** k
    return np.linalg.solve(diffs, y)


def fsbt(l, f_g, r_g, G_k, mul_by_r):
    """
    This is the Fast spherical Bessel transform implemented in GPAW.

    Returns::

          oo
         / 2
         |r dr j (Gr) f(r),
         /      l
          0

    using l+1 fft's."""

    N = (len(G_k) - 1) * 2
    f_k = 0.0
    if mul_by_r:
        F_g = f_g * r_g
    else:
        F_g = f_g
    for n in range(l + 1):
        f_k += (r_g[1] * (1j)**(l + 1 - n) *
                fac(l + n) / fac(l - n) / fac(n) / 2**n *
                np.fft.rfft(F_g, N)).real * G_k**(l - n)
        F_g[1:] /= r_g[1:]

    f_k[1:] /= G_k[1:]**(l + 1)
    if l == 0:
        f_k[0] = np.dot(r_g, f_g * r_g) * r_g[1]
    return f_k


def fft_upf(r, f, l, mul_by_r=True):
    N = r.size
    G = np.linspace(0, np.pi / r[1], N // 2 + 1)
    fk = 4 * np.pi * fsbt(l, f, r, G, mul_by_r=mul_by_r)
    return G, fk


def make_radial_derivative_calculator(r_g, deriv_order=1, stencil_order=2):
    """
    This utility function takes an arbitrary radial grid and returns
    a function that calculates numerical derivatives on that grid.
    Based on the function in CiderPress of the same name. This
    function might be less precise than more sophisticated
    techniques of the same order, but it has the benefit that it
    can be used on arbitrary radial grids, without knowledge of
    the particular grid being used. A second function is also
    returned that can evaluated the derivative of the radial
    derivative with respect to a change in function value.

    Args:
        r_g (np.ndarray): grid on which to compute derivative
        deriv_order (int): order of the derivative
        stencil_order (int): 2*stencil_order+1 nearby points are
            use to compute the derivative.
    """
    N = r_g.size
    assert N > stencil_order, "Grid too small"
    assert stencil_order > 0, "Order must be > 0"
    D = 2 * stencil_order + 1
    SO = stencil_order
    weight_list = np.empty((D, N))
    for i in range(SO):
        weight_list[:, i] = _get_deriv_weights(r_g, D, i, 0, deriv_order)
    for i in range(SO, N - SO):
        weight_list[:, i] = _get_deriv_weights(r_g, D, i, i - SO, deriv_order)
    for i in range(N - SO, N):
        weight_list[:, i] = _get_deriv_weights(r_g, D, i, N - D, deriv_order)
    end = N - D + 1

    def _eval_radial_deriv(func_xg):
        deriv_xg = np.empty_like(func_xg)
        deriv_xg[..., :SO] = np.einsum(
            "...g,gd->...d", func_xg[..., :D], weight_list[:, :SO]
        )
        deriv_xg[..., -SO:] = np.einsum(
            "...g,gd->...d", func_xg[..., -D:], weight_list[:, -SO:]
        )
        deriv_xg[..., SO:-SO] = weight_list[0, SO:-SO] * func_xg[..., :end]
        for d in range(1, D):
            deriv_xg[..., SO:-SO] += (
                weight_list[d, SO:-SO] * func_xg[..., d : end + d]
            )
        return deriv_xg

    def _eval_radial_deriv_bwd(vderiv_xg):
        vfunc_xg = np.zeros_like(vderiv_xg)
        vfunc_xg[..., :end] = (
            weight_list[0, SO:-SO] * vderiv_xg[..., SO:-SO]
        )
        for d in range(1, D):
            vfunc_xg[..., d : end + d] += (
                weight_list[d, SO:-SO] * vderiv_xg[..., SO:-SO]
            )
        vfunc_xg[..., :D] += np.einsum(
            "...d,gd->...g", vderiv_xg[..., :SO], weight_list[:, :SO]
        )
        vfunc_xg[..., -D:] += np.einsum(
            "...d,gd->...g", vderiv_xg[..., -SO:], weight_list[:, -SO:]
        )
        return vfunc_xg

    return _eval_radial_deriv, _eval_radial_deriv_bwd



if __name__ == "__main__":
    fname = '../../gpaw_data/sg15_oncv_upf_2020-02-06/C_ONCV_PBE-1.2.upf'
    get_nc_data_from_upf(fname)
