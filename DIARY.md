## Current status ##

This `csf_mem` branch represents my attempt to address GitHub issue #156. The goal is to prevent construction in memory of any array of size proportional to $e^{2norb}$ at the three hotspots discussed in that issue thread.

In the `pspace` and `make_hdiag_csf` hotspots, the master branch currently builds Hamiltonian matrices in determinant subspaces and then applies the transformations implemented in csfstring.py (ultimately, `get_spin_evecs`) to put them into the CSF basis. The problem is that we run out of memory for this well before we run out of memory to store the CI vector. Although one could just separate the arrays into blocks dealt with sequentially to defeat this problem, I elected to attempt to implement the vector-coupling approach of Drake and Schlesinger (PRA volume 15 page 1990 year 1977, DOI:10.1103/PhysRevA.15.1990), which virtually all CSF work in the literature ultimately cites to. (This paper is available on my laptop in ~/Papers/PhysRevA.15.1990.pdf. An LLM can read most of it but one or two graphical equations might be hard to see. It is a scan of a scientific paper from the 1970s with nonstandard diagrammatic equations everywhere.) One of my objectives in going this route was to build personal competency and intuition in how to do the quantum mechanics of angular momentum: just blocking over the arrays may have been easier, but it wouldn't have built skills that transfer to the different contexts of LASSI, spin-orbit coupling, etc. I believe this part of the project is now complete.

The final step is to refactor the determinant <-> CSF basis unitary transformations in lines 676-708 and 814-817 of `csfstring.py` to avoid constructing the entire dense `umat` matrix (return value of `get_spin_evecs`) at any one time. The two possibilities are

1. Block over rows or columns based on available memory, or
2. Do the whole transformation one spin at a time, entirely on the fly, without ever allocating a `umat` array.

The `umat` array is just a matrix product state using Clebsch-Gordan coefficients:

$$U_{\vec{m}\vec{s}} = \prod_{i=1}^n \langle\bar{s}_{i-1},\bar{m}_{i-1},\frac{1}{2},m_i|\bar{s}_i,\bar{m_i}\rangle$$

where $s_i, m_i \in \pm \frac{1}{2}$, $\bar{s}_i = \sum_j^i s_i, \bar{m}_i = \sum_j^i m_i$, all $\bar{s}_i \ge 0$, $s_0 = +\frac{1}{2}$, $\bar{s}_n = S =$ `(smult - 1) // 2` = the target spin multiplicity and $\bar{m}_n = M =$ `(neleca - nelecb) // 2` the spin polarization quantum number.

## Historical notes ##

By commit d4f9f180d19ed2f91890fc95534791bab4bcc769 I realized that the CSF sign convention implemented in pyscf-forge didn't respect the rule that adding or removing paired electrons should leave the sign unchanged. I had the option of either compensating for this inconsistency in `csf_op.c`, or modifying the sign handling in `csfstring.py`; I chose the latter, because adding or removing a pair of electrons *should* leave the sign unchanged. This fix was completed by commit fcbd77fe81eed8f1014ce5d09d96c637d7dfd8ea.

## Stumbling blocks ##

If this is in a worktree, remember to use the `pyscf_here` command from bashrc when testing!

