# Contributing Guidelines

PySCF-forge repository is a place where new methodologies, pilot features, and
testing functionalities can be developed and tested before being integrated into
the PySCF core branch (https://github.com/pyscf/pyscf). The PySCF core branch is
supposed to serve methods that have been published and widely used.

When you install both the PySCF and PySCF-forge packages, you can access the
features of PySCF-forge using `from pyscf import ...` statement, just like you
would access the regular features in the PySCF core branch.

You might have noticed several PySCF extensions hosted in the PySCF organization
page on GitHub (https://github.com/pyscf). These extensions are created to
decouple compilation and maintenance efforts from the PySCF core branch. They
are expected to follow the same standards of documentation, testing,
cross-Python-version compatibility, and maintenance requirements as the core
branch. You are welcome to release new features as PySCF extensions. If you
choose to develop features as PySCF extensions, please contact any PySCF
maintainers at https://github.com/orgs/pyscf/people .

## Principles

* Tests, documentation and compatibility with multiple Python versions are not
  mandatory as in the PySCF core branch. However, it is still recommended to
  include well-designed tests, examples, documentations to help users to
  understand the new feature.

* The standard for merging pull requests.
  A code review will still be required for pull requests (PR) for PySCF-forge.
  Please update the code appropriately based on reviewers' suggestions.
  The PR will not be merged until all comments are addressed. Additionally,
  there is a quick static code check for the code that should all pass.

* Avoiding filename and module conflicts.
  PySCF-forge manages sub-packages through the mechanism "named packages".
  When installing PySCF-forge, the sub-packages will be installed in the same directory as PySCF.
  This ensures that these features can be accessed within the same namespace as PySCF.
  If any packages or files have the same name as those in PySCF, they will overwrite the existing ones.
  For example, if you create a `pyscf-forge/pyscf/__init__.py` file, it will
  replace the existing `__init__.py` file in PySCF and may lead to PySCF runtime
  errors. It is important to avoid creating files or directories that already
  exist in the PySCF core branch.

* Dependencies.
  There is no restriction on adding dependencies. However, dependencies can
  sometimes leads to conflicts in certain packages. Therefore, please add
  dependencies cautiously and only include necessary libraries. If dependencies
  cause installation issues, your feature might be removed (see the next rule).

* Module compatibility and removal policy.
  If a module causes installation or compatibility issues with existing modules or
  those in the PySCF core branch, a post will be created on GitHub issue board to
  address the problem. Features may be removed without notifying to the contributor.

* Compiling C/C++ extensions.
  You can utilize CMake or `setuptools.Extension` in setup.py within the
  PySCF-forge to compile simple extensions. For complex compiling configurations,
  it is advisable to release the feature as a PySCF extension. C/C++ libraries
  should be compiled and placed under the directory `pyscf-forge/pyscf/lib`.
  This is the location where .so files will be packaged in wheel.

## How to transfer a feature to PySCF core branch?

After a feature has been added to PySCF-forge for over 6 months, developers can
open an issue to request transferring the feature to the PySCF core branch.
The proposal template can be accessed at (TBD). The proposal will be reviewed
during the PySCF board meetings, held approximately every 3 months.

---

Thank you for considering contributing your works to PySCF community!
