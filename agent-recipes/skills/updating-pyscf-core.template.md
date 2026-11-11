# PySCF version upgrades

PySCF-Forge unittests do not run in CI in the PySCF core module, so version
upgrades typically break PySCF-Forge. Modifications to PySCF-Forge must be made
whenever this happens *via* a PR cycle.

## Procedure

1. Run all PySCF-Forge unittests, including ones marked `*_slow` that are
   excluded from CI, using the latest available PySCF pip version or `master`
   branch commit. Sort failures into categories, using reruns with the second
   most recent PySCF core version available on pip as a benchmark:
    - accidental: non-reproducible numerical noise.
    - already broken: a test was already broken, even when run with the previous
      version of PySCF core.
    - API change: PySCF core changed its API, leading to altered behavior or a
      crash.
    - Behavioral change: With the same API or with the API fixed, PySCF core's
      behavior changed in a way that PySCF-Forge depends on.
2. Fix API changes and rerun tests.
3. Propose fixes for behavioral changes for user review.
    - If the tolerance of a unittest is essentially tighter than the tolerance
      of an iterative algorithm (such as SCF) that produced the tested result,
      then it is reasonable to simply update the reference value or loosen the
      test tolerance. Otherwise, try to avoid this "solution".
4. Consult the user and determine the course of action for "already broken"
   tests.
5. With user review and disclosure of AI assistance, compose and post an issue
   thread on PySCF-Forge github listing the API and behavioral changes that led
   to unittest failures in PySCF-Forge.
6. With user review and disclosure of AI assistance, compose and post a pull
   request fixing the issues listed.

