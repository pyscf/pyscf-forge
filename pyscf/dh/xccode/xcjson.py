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

import os
import json
from pyscf import dft
import re


# All 5-th functional detailed dictionary
FUNCTIONALS_DICT = dict()  # type: dict[str, dict]
dir_functionals = os.path.join(os.path.dirname(os.path.abspath(__file__)), "functionals")
for file_name in os.listdir(dir_functionals):
    with open(os.path.join(dir_functionals, file_name), "r") as f:
        FUNCTIONALS_DICT.update(json.load(f))
FUNCTIONALS_DICT = {key.upper(): val for key, val in FUNCTIONALS_DICT.items()}

# Handle alias for 5-th functionals
FUNCTIONALS_DICT_ADD = dict()
for key in FUNCTIONALS_DICT:
    for alias in FUNCTIONALS_DICT[key].get("alias", []):
        FUNCTIONALS_DICT_ADD[alias] = FUNCTIONALS_DICT[key]
        FUNCTIONALS_DICT_ADD[alias]["see_also"] = key
FUNCTIONALS_DICT.update(FUNCTIONALS_DICT_ADD)

# handle underscores for 5-th functionals
FUNCTIONALS_DICT_ADD = dict()
for key in FUNCTIONALS_DICT:
    sub_key = re.sub("[-_/]", "", key)
    if sub_key != key:
        FUNCTIONALS_DICT_ADD[sub_key] = FUNCTIONALS_DICT[key]
        FUNCTIONALS_DICT_ADD[sub_key]["see_also"] = key
FUNCTIONALS_DICT.update(FUNCTIONALS_DICT_ADD)

# handle D3 -> D3BJ suffix (old naming convention)
FUNCTIONALS_DICT_ADD = dict()
for key in FUNCTIONALS_DICT:
    if key.endswith("D3BJ"):
        key_d3 = key[:-4] + "D3"
        if key_d3 not in FUNCTIONALS_DICT:
            FUNCTIONALS_DICT_ADD[key_d3] = FUNCTIONALS_DICT[key]
FUNCTIONALS_DICT.update(FUNCTIONALS_DICT_ADD)

# advanced correlation contributors
dir_functionals = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correlations")
with open(os.path.join(dir_functionals, "definition_adv_corr.json"), "r") as f:
    ADV_CORR_DICT = json.load(f)
with open(os.path.join(dir_functionals, "alias.json"), "r") as f:
    ADV_CORR_ALIAS = json.load(f)

# Dashed names
_NAME_WITH_DASH = {key.replace("_", "-"): key for key in FUNCTIONALS_DICT if "_" in key}
_NAME_WITH_DASH.update({
    key.replace("_", "-"): key
    for key in list(ADV_CORR_DICT.keys()) + list(ADV_CORR_ALIAS.keys())
    if "_" in key})
_NAME_WITH_DASH.update(dft.libxc._NAME_WITH_DASH)
# avoid cases that have multiple hyphens; replace string with larger string first
_NAME_WITH_DASH = sorted(_NAME_WITH_DASH.items(), key=lambda v: -len(v[1]))
