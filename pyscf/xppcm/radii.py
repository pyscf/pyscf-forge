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
Atomic radii used by the XP-PCM module that are not available in
pyscf.data.radii. Same conventions as that module: tables are indexed by
nuclear charge (index 0 is the ghost atom), given in Bohr, and elements
without a tabulated value carry the placeholder ``unknown``.
'''

import numpy
from pyscf.lib.parameters import BOHR

unknown = 1.999999   # placeholder, as in pyscf.data.radii

# Rahm et al., Chem. Eur. J. 2016, 22, 14625-14632
RAHM = 1/BOHR * numpy.array((
    unknown,  # Ghost atom
    1.54,       #  1 H
    1.34,       #  2 He
    2.20,       #  3 Li
    2.19,       #  4 Be
    2.05,       #  5 B
    1.90,       #  6 C
    1.79,       #  7 N
    1.71,       #  8 O
    1.63,       #  9 F
    1.56,       # 10 Ne
    2.25,       # 11 Na
    2.40,       # 12 Mg
    2.39,       # 13 Al
    2.32,       # 14 Si
    2.23,       # 15 P
    2.14,       # 16 S
    2.06,       # 17 Cl
    1.97,       # 18 Ar
    2.34,       # 19 K
    2.70,       # 20 Ca
    2.63,       # 21 Sc
    2.57,       # 22 Ti
    2.52,       # 23 V
    2.33,       # 24 Cr
    2.42,       # 25 Mn
    2.26,       # 26 Fe
    2.22,       # 27 Co
    2.19,       # 28 Ni
    2.17,       # 29 Cu
    2.22,       # 30 Zn
    2.33,       # 31 Ga
    2.34,       # 32 Ge
    2.31,       # 33 As
    2.24,       # 34 Se
    2.19,       # 35 Br
    2.12,       # 36 Kr
    2.40,       # 37 Rb
    2.79,       # 38 Sr
    2.74,       # 39 Y
    2.68,       # 40 Zr
    2.51,       # 41 Nb
    2.44,       # 42 Mo
    2.41,       # 43 Tc
    2.37,       # 44 Ru
    2.33,       # 45 Rh
    2.15,       # 46 Pd
    2.25,       # 47 Ag
    2.38,       # 48 Cd
    2.46,       # 49 In
    2.48,       # 50 Sn
    2.46,       # 51 Sb
    2.42,       # 52 Te
    2.38,       # 53 I
    2.32,       # 54 Xe
    2.49,       # 55 Cs
    2.93,       # 56 Ba
    2.84,       # 57 La
    2.82,       # 58 Ce
    2.86,       # 59 Pr
    2.84,       # 60 Nd
    2.83,       # 61 Pm
    2.80,       # 62 Sm
    2.80,       # 63 Eu
    2.77,       # 64 Gd
    2.76,       # 65 Tb
    2.75,       # 66 Dy
    2.73,       # 67 Ho
    2.72,       # 68 Er
    2.71,       # 69 Tm
    2.77,       # 70 Yb
    2.70,       # 71 Lu
    2.64,       # 72 Hf
    2.58,       # 73 Ta
    2.53,       # 74 W
    2.49,       # 75 Re
    2.44,       # 76 Os
    2.33,       # 77 Ir
    2.30,       # 78 Pt
    2.26,       # 79 Au
    2.29,       # 80 Hg
    2.42,       # 81 Tl
    2.49,       # 82 Pb
    2.50,       # 83 Bi
    2.50,       # 84 Po
    2.47,       # 85 At
    2.43,       # 86 Rn
    2.58,       # 87 Fr
    2.92,       # 88 Ra
    2.93,       # 89 Ac
    2.89,       # 90 Th
    2.85,       # 91 Pa
    2.83,       # 92 U
    2.80,       # 93 Np
    2.78,       # 94 Pu
    2.76,       # 95 Am
    2.76,       # 96 Cm
    unknown,    # 97 Bk
    unknown,    # 98 Cf
    unknown,    # 99 Es
    unknown,    #100 Fm
    unknown,    #101 Md
    unknown,    #102 No
    unknown,    #103 Lr
))
