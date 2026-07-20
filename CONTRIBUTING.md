# Contributing Guidelines

The PySCF-forge repository is a place to stage new methodologies that may one day be incorporated
into the PySCF core branch (https://github.com/pyscf/pyscf). By placing your code into PySCF-forge, 
existing PySCF maintainers will have a chance to comment on the code and work with you to assess the feasibility and
suitability for the core. Placing your code in PySCF-forge means that pull requests 
have to be approved by an existing PySCF maintainer, so it is not the place for code that is in
the earliest stages of development. Code in PySCF-forge does not necessarily make it into the core branch: this requires
explicit approval from the board (see below). Note that the PySCF core branch is
intended only for methods that have been published and are widely used (or have the clear potential for wide use).

When you install both the PySCF and PySCF-forge packages, you can access the
features of PySCF-forge using `from pyscf import ...` statement, just like you
would access the regular features in the PySCF core branch.

You might have noticed several PySCF extensions hosted in the PySCF organization
page on GitHub (https://github.com/pyscf). These extensions have been created to
decouple compilation and maintenance efforts from the PySCF core branch. 
They are expected to follow the same standards of documentation, testing,
cross-Python-version compatibility, and maintenance requirements as the core
branch, but the PySCF maintainers do not actively ensure these standards are met. 
Most new features which are built using PySCF, but which do not affect the core functionality, are best released as extensions,
and you always welcome to do so.  If you
choose to develop a feature as a PySCF extension, please contact any of the PySCF
maintainers at https://github.com/orgs/pyscf/people.

## Principles

* Tests, documentation and compatibility with multiple Python versions are not
  mandatory as in the PySCF core branch. However, it is still recommended to
  include well-designed tests, examples, documentations to help users to
  understand the new feature.

* The standard for merging pull requests.
  A code review will still be required for pull requests (PR) for PySCF-forge.
  Please update the code appropriately based on reviewers' suggestions.
  The PR will not be merged until all comments are addressed. Additionally,
  there is a quick static code check, which the code should pass.

* Avoiding filename and module conflicts.
  PySCF-forge manages sub-packages through the mechanism of "namespace packages".
  When installing PySCF-forge, the sub-packages will be installed in the same
  directory as PySCF. This ensures that these features can be accessed within
  the same namespace as PySCF. If any packages or files have the same name as
  those in PySCF, they will overwrite the existing ones. For example, if you
  create a `pyscf-forge/pyscf/__init__.py` file, it will replace the existing
  `__init__.py` file in PySCF and may lead to PySCF runtime errors. It is
  important to avoid creating files or directories that already exist in the
  PySCF core branch.

* Dependencies.
  There is no restriction on adding dependencies. However, dependencies can
  sometimes leads to conflicts in certain packages. Therefore, please add
  dependencies cautiously and only include necessary libraries. If dependencies
  cause installation issues, your feature might be removed (see the next rule).

* Module compatibility and removal policy.
  If a module causes installation or compatibility issues with existing modules or
  those in the PySCF core branch, a post will be created on GitHub issue board to
  address the problem. Features may be removed without notifying the contributor.

* Compiling C/C++ extensions.
  You can utilize CMake or `setuptools.Extension` in setup.py within the
  PySCF-forge to compile simple extensions. For complex compiling configurations,
  it is advisable to release the feature as a PySCF extension. C/C++ libraries
  should be compiled and placed under the directory `pyscf-forge/pyscf/lib`.
  This is the location where .so files will be packaged in wheel.

## How to transfer a feature to the PySCF core branch?

After a feature has been added to PySCF-forge for over 6 months, developers can
open an issue to request to transfer the feature to the PySCF core branch.
A Feature Transfer Proposal template is available when creating a new GitHub
issue. The proposal will be reviewed
during the PySCF board meetings, held approximately every 3 months. If it is decided that
the feature is incompatible with the requirements of the core branch, the board may recommend
additional modifications, or that the feature be removed from PySCF-forge.

---

Thank you for considering contributing your work to the PySCF community!
