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
# Authors:
#          Zhenyu Zhu <ajz34@outlook.com>
#          Shirong Wang <srwang20@fudan.edu.cn>
#

from . import dhutil as dhutil
from . import rdfdh, udfdh
from .dh import to_dh as to_dh
from pyscf import gto


def DFDH(mf_or_mol, *args, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        if mf_or_mol.spin != 0:
            return udfdh.UDFDH(mf_or_mol, *args, **kwargs)
        return rdfdh.RDFDH(mf_or_mol, *args, **kwargs)
    else:
        if mf_or_mol.istype('RHF'):
            return rdfdh.RDFDH(mf_or_mol, *args, **kwargs)
        return udfdh.UDFDH(mf_or_mol, *args, **kwargs)
