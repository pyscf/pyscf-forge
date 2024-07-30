#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefath <mhennefarth@uchicago.edu>

from pyscf.lib.chkfile import load_mol, load

def load_pdft(chkfile):
    return load_mol(chkfile), load(chkfile, "pdft")

def dump_pdft(mc, chkfile=None, key="pdft", ):
    """Save MC-PDFT calculation results in chkfile"""
    if chkfile is None: chkfile = mc.chkfile
