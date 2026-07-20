#!/usr/bin/env python

import builtins

from pyscf import gto


print(gto.Mole.__name__)

real_import = builtins.__import__


def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split(".", 1)[0]
    if top in {"jax", "jaxlib"}:
        raise ModuleNotFoundError(f"No module named '{top}'")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = fake_import

from pyscf import lno

print(lno.LNO.__name__)

try:
    from pyscf.afqmc import lnoafqmc  # noqa: F401
except ImportError as err:
    print(type(err).__name__)
    print(err)
else:
    raise SystemExit("unexpected success")
