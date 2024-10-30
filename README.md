# pyscf-forge
pyscf-forge is a staging ground for code that may be suitable for pyscf-core. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for further guidelines.

## Install
pyscf-forge can be installed using the command
```
pip install pyscf-forge
```
This command adds features in pyscf-forge to the PySCF package. They can be used
as if they were natively developed within PySCF. For example,
```
from pyscf import gto, mcpdft
mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)
mf = mol.RHF().run ()
# Calling the MC-PDFT method provided by the pyscf-forge modules
mc = mcpdft.CASCI(mf, 'tPBE', 6, 8).run()
```

To access the newest features of pyscf-forge, you can install them from git
repository by running the command
```
pip install git+https://github.com/pyscf/pyscf-forge
```

## Configuring the Development Environment
If you are developing new features or modifying the code in `pyscf-forge`, an editable installation is recommended.
By configuring the package in editable mode, you can modify existing modules and add new features to `pyscf-forge`.
After cloning the library to your local repository, there are two ways to enable the editable installation:

### Method 1. Using pip for editable installation
Install the package with the following pip command:
```
pip install --no-deps -e /path/to/pyscf-forge
```
This command creates a `.pth` file in `~/.local/lib/python3.*/site-packages/`
or other Python runtime paths. It is recommended to use this method with Python
virtual environment.

### Method 2. Setting an environment variable
Define the `PYSCF_EXT_PATH` environment variable to point to your local `pyscf-forge` directory:
```
export PYSCF_EXT_PATH=/path/to/pyscf-forge
```
The PySCF package can read the `PYSCF_EXT_PATH` environment and load modules
from this path at runtime. For more details of `PYSCF_EXT_PATH` environment
and the extensions management, refer to the PySCF installation manual
https://pyscf.org/install.html#extension-modules

## Adding New Features: An Example
Suppose you need to create a module in `pyscf-forge` that provides a plane-wave basis for crystalline computation with periodic boundary conditions (PBC).
You can follow these steps to add the module:

1. Install `pyscf-forge` in editable installation mode.

2. Create a folder named `pyscf-forge/pyscf/pw`.
Thanks to the editable installation mode, this folder can be readily imported as a regular `pyscf` module.
```
>>> from pyscf import pw
>>> print(pw)
<module 'pyscf.pw' (namespace)>

>>> print(pw.__path__)
_NamespacePath(['/home/ubuntu/pyscf-forge/pyscf/pw'])
```

3. Add Python code files to the `pyscf-forge/pyscf/pw` directory.
This process is similar to developing new methods in the main `pyscf` repository.
For example, you can add the following Python files into the `pyscf-forge/pyscf/pw` folder:
```
pyscf-forge
├── ...
└── pyscf
    ├── ...
    └── pw
        ├── __init__.py
        ├── dft
        │   ├── __init__.py
        │   ├── krks.py
        │   └── kuks.py
        └── scf
            ├── __init__.py
            ├── krhf.py
            └── kuhf.py
```

### Path Conflicts
There may exist scenarios that the directory you plan to create already exists within `pyscf`.
For example, if you want to add a new method, like `pp_rpa.py`, to the `pyscf/tdscf` folder,
this could conflict with the existing `pyscf.tdscf` module in the pyscf core repository.
Adding features to existing modules requires more complex configuration.

To import the `pp_rpa` module from the `pyscf-forge` repository, you will need to make certain modifications in the `__init__.py` file of the `pyscf` core module
(in this demonstration, this file is located at `/home/ubuntu/.local/lib/python3.10/site-packages/pyscf/tdscf/__init__.py`).
Add the following line of code to modify the `__path__` attribute of the `pyscf.tdscf` module:
```
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

This command extends the search path of the `tdscf` module, resulting in the `__path__` attribute being set to:
```
['/home/ubuntu/.local/lib/python3.10/site-packages/pyscf/tdscf',
 '/home/ubuntu/pyscf-forge/pyscf/tdscf']
```
This configuration allows Python to locate and load the new `pp_rpa.py` module from the extended directory in `pyscf-forge`.

Note that the `pyscf` core `tdscf` folder already contains an `__init__.py` file.
To avoid overwriting the existing `__init__.py` file in `pyscf` during the installation of `pyscf-forge`,
you should not add an `__init__.py` file in the `pyscf-forge/pyscf/tdscf` directory.

The structure of the core packages and the components of `pyscf-forge` can be organized as follows:
```
pyscf
├── ...
└── pyscf
    ├── ...
    └── tdscf
        ├── __init__.py  // modify the __path__ attribute in pyscf core module
        ├── rhf.py
        ├── rks.py
        └── ...

pyscf-forge
├── ...
└── pyscf
    ├── ...
    └── tdscf  // no __init__.py file in pyscf-forge
        └── pp_rpa.py
```

When installing the `pyscf-forge` wheels using `pip install` in the normal
installation mode, the `pp_rpa.py` file will be added to the `pyscf/tdscf`
folder, integrating seamlessly as part of the regular `pyscf` module.
After this standard installation, there is no need to adjust the `__path__`
attribute, as all features and modules are located within the same directory.
