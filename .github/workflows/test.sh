#!/usr/bin/env bash
  
set -e

cd ./pyscf
pytest --import-mode=importlib -k 'not _slow'
