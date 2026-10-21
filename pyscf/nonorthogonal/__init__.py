#!/usr/bin/env python
#
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
Nonorthogonal determinant methods: extended nonorthogonal Wick theorem
matrix elements between arbitrary excitations of nonorthogonal references,
and nonorthogonal configuration interaction (NOCI) over them.
'''

from pyscf.nonorthogonal.wicks import Determinant, WickEngine, get_hamiltonian
from pyscf.nonorthogonal.noci import NOCI, NOCISD, NOCISDT, excitations
