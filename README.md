# pyscf-forge
pyscf-forge is a staging ground for code that may be suitable for pyscf-core. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for further guidelines.

## Install
pyscf-forge can be installed using the command
```
pip install pyscf-forge
```
This command adds features in pyscf-forge to the PySCF package. They can be used
as if they were natively developed within PySCF.

To access the newest features of pyscf-forge, you can install them from git
repository by running the command
```
pip install git+https://github.com/pyscf/pyscf-forge
```

If you are developing new features and would like to modify the code in
pyscf-forge, editable installation is better suited. After cloning the library
to your local repository, there are two possible ways to enable the editable
installation:

1. You can install the package using the pip command
```
pip install --no-deps -e /path/to/pyscf-forge
```
This method will adds a `.pth` file in `~/.local/lib/python3.*/site-packages/`
or other Python runtime paths. It is recommended to use this method with Python
virtual environment.

2. Setting an environment variable
```
export PYSCF_EXT_PATH=/path/to/pyscf-forge
```
The PySCF package can read the `PYSCF_EXT_PATH` environment and load the modules
within this package at runtime. For more details of `PYSCF_EXT_PATH` environment
and the extensions management, please see the PySCF installation manual
https://pyscf.org/install.html#extension-modules
