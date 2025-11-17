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
pip install .

pip install trexio

# TODO: check if pyscf code is changed using dist-info file
#pip uninstall -y pyscf-forge
