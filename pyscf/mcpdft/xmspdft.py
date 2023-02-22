#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

def safock_energies(mc, mo_coeff=None, ci=None, h2eff=None, eris=None):
    '''The "objective" function we are optimizing when solving for the 
    SA-Fock eigenstates is that the SA-Fock energy (total or average) is
    minimized with the constraint to the final states being orthonormal.
    '''
    return None, 0, None, 

def solve_safock(mc, mo_coeff=None, ci=None):
    pass
