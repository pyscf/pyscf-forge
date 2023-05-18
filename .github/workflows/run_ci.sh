#!/usr/bin/env bash

set -e

sudo apt-get -qq install \
    gcc \
    libblas-dev \
    cmake \
    curl

python -m pip install --upgrade pip
pip install pytest
pip install .

# TODO: check if pyscf code is changed using dist-info file
#pip uninstall -y pyscf-forge
