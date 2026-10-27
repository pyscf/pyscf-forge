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

'''
XP-PCM: eXtreme-Pressure Polarizable Continuum Model

Pauli repulsion (XP) and cavitation (XPCav) decorators for SCF objects,
analytical XP-PCM pressures for SCF and post-HF methods, and the cavity
volume and PCM derivative helpers they rely on.
'''

from .xppcm import (
    XP, XPCav,
    get_e_rep, get_z0, get_z, get_eps,
    calculate_fitted_pressure, calculate_numerical_pressure, calculate_analytical_pressure,
    get_cavity_volume, get_cavity_volume_atom,
    Rahm, Vdw, Mod_Bondi,
)
