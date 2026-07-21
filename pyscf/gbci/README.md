# GBCI and GBPDFT

This directory contains the GBCI/GBCI implementation and the GBPDFT
extension.

GBPDFT is now included inside the `pyscf.gbci` package. It is no longer used
as a separate `pyscf.gbpdft` package.

## Dependencies

GBPDFT uses the compiled GBCI helper library for fast on-top pair-density
contractions. Build the PySCF-forge C extensions before running GBPDFT.

## Modules

Main GBCI/GBCI modules:

- `pyscf.gbci.gbci`
- `pyscf.gbci.direct_gbci`
- `pyscf.gbci.rdm`

GBPDFT modules:

- `pyscf.gbci.gbpdft`
- `pyscf.gbci.otpd`

## Import Changes

Use GBPDFT through `pyscf.gbci`:

```python
from pyscf.gbci import gbpdft
from pyscf.gbci import otpd
```

Do not use the old separate-package imports:

```python
from pyscf import gbpdft
from pyscf.gbpdft import otpd
```

## GBCI Usage

```python
from pyscf import gto, scf, gbci

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()

mc = gbci.gbci(mf, ncas=2, nelecas=(1, 1))
mc.fcisolver.nroots = 1

e_gbci, e_cas, ci = mc.kernel()

print("GBCI energy:", e_gbci)
```

## GBPDFT Usage

Import the GBPDFT module from `pyscf.gbci`:

```python
from pyscf.gbci import gbpdft
```

### Build GBPDFT directly from an SCF object

```python
from pyscf import gto, scf
from pyscf.gbci import gbpdft

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()

pdft = gbpdft.GBCI(
    mf,
    "tPBE",
    ncas=2,
    nelecas=(1, 1),
)

pdft.fcisolver.nroots = 1

e_tot, e_ot, e_gbci, e_cas, ci = pdft.kernel()

print("GBPDFT total energy:", e_tot)
print("on-top energy:", e_ot)
print("GBCI energy:", e_gbci)
```

### Build GBPDFT from an existing GBCI/GBCI object

```python
from pyscf import gto, scf, gbci
from pyscf.gbci import gbpdft

mol = gto.M(
    atom="Li 0 0 0; H 0 0 1.6",
    basis="sto-3g",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()

mc = gbci.gbci(mf, ncas=2, nelecas=(1, 1))
mc.fcisolver.nroots = 1

pdft = gbpdft.gbci(mc, "tPBE")

e_tot, e_ot, e_gbci, e_cas, ci = pdft.kernel()

print("GBPDFT total energy:", e_tot)
```

## Examples

Example input files are available in:

```text
pyscf/gbci/examples/
```

For example:

```bash
python pyscf/gbci/examples/NH3_F2_GBCI.py
python pyscf/gbci/examples/NH3_F2_GBPDFT.py
python pyscf/gbci/examples/NH3_F2_grouping_cases.py --case atom
```
