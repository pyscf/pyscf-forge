#!/usr/bin/env bash

set -e

sudo apt-get -qq install \
    gcc \
    libblas-dev \
    cmake \
    curl

python -m pip install --upgrade pip
pip install "scipy<1.16"
pip install pytest

# Forge must not ship an __init__.py that already exists in PySCF core: a regular
# install would overwrite core's file, and uninstalling forge would delete it (#219).
# Check against core before forge is installed on top of it.
pip install pyscf
core=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('pyscf').origin))")
clash=$(cd pyscf && find . -name __init__.py -not -path ./__init__.py | while read -r f; do
    if [ -e "$core/$f" ]; then echo "pyscf/${f#./}"; fi
done)
if [ -n "$clash" ]; then
    echo "These files would overwrite PySCF core's (see README, \"Path Conflicts\"):"
    echo "$clash"
    exit 1
fi

pip install .
pip install -U jax

pip install trexio
pip install mcfun
pip install --no-deps pyscf-dispersion==1.5.0

# TODO: check if pyscf code is changed using dist-info file
#pip uninstall -y pyscf-forge
