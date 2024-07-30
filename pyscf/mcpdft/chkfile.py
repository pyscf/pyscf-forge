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

import h5py
from pyscf.lib.chkfile import load_mol, load
from pyscf.lib import H5FileWrap

def load_pdft(chkfile):
    return load_mol(chkfile), load(chkfile, "pdft")

def dump_mcpdft(mc, chkfile=None, key="pdft", e_tot=None, e_ot=None, e_states=None):
    """Save MC-PDFT calculation results in chkfile"""
    if chkfile is None: chkfile = mc.chkfile
    if e_tot is None: e_tot = mc.e_tot
    if e_ot is None: e_ot = mc.e_ot
    if e_states is None: e_states = mc.e_states

    if h5py.is_hdf5(chkfile):
        mode = 'a'
    else:
        mode = 'w'

    with H5FileWrap(chkfile, mode) as fh5:
        if mode == "a" and key in fh5:
            del (fh5[key])
    
        def store(subkey, val):
            if val is not None:
                fh5[key+"/"+subkey] = val

        store("e_tot", e_tot)
        store("e_ot", e_ot)
        store("e_states", e_states)
