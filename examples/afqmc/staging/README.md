# Staged AFQMC Workflow

These scripts split AFQMC into two phases:

1. run the trial generation step and write staged inputs to disk
2. load the staged file later and run AFQMC

This is useful when the PySCF setup is CPU-heavy while the AFQMC calculation
is intended for a different machine like a GPU.
