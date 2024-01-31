from pyscf import gto, scf,  lib, mcpdft

# NAC signs are really, really hard to nail down.
# There are arbitrary signs associated with
# 1. The MO coefficients
# 2. The CI vectors
# 3. Almost any kind of post-processing (natural-orbital analysis, etc.)
# 4. Developer convention on whether the bra index or ket index is 1st
# It MIGHT help comparison to OpenMolcas if you load a rasscf.h5 file
# I TRIED to choose the same convention for #4 as OpenMolcas.
mol = gto.M (atom='Li 0 0 0;H 1.5 0 0', basis='sto-3g',
             output='LiH_cms2ftlda22_sto3g.log', verbose=lib.logger.INFO)

mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=3)
# Quasi-ultrafine is ALMOST the same thing as
#   ```
#   grid input
#   nr=100
#   lmax=41
#   rquad=ta
#   nopr
#   noro
#   end of grid input
#   ```
# in SEWARD
mc.fix_spin_(ss=0, shift=1)
mc = mc.multi_state ([0.5,0.5], 'cms').run (conv_tol=1e-10)

mc_nacs = mc.nac_method()

# 1. <1|d0/dR>
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=1 2
#    ```
nac = mc_nacs.kernel (state=(0,1))
print ("\nNAC <1|d0/dR>:\n",nac)
print ("Notice that according to the NACs printed above, rigidly moving the")
print ("molecule along the bond axis changes the electronic wave function, which")
print ("is obviously unphysical. This broken translational symmetry is due to the")
print ("antisymmetric orbital-overlap derivative in the Hellmann-Feynman part of")
print ("the 'model state contribution'. Omitting the antisymmetric orbital-overlap")
print ("derivative corresponds to the use of the 'electron-translation factors' of")
print ("Fatehi and Subotnik and is requested by passing 'use_etfs=True'.")

# 2. <1|d0/dR> w/ ETFs (i.e., w/out model-state Hellmann-Feynman contribution)
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=1 2
#    NOCSF
#    ```
nac = mc_nacs.kernel (state=(0,1), use_etfs=True)
print ("\nNAC <1|d0/dR> w/ ETFs:\n", nac)
print ("These NACs are much more well-behaved: moving the molecule rigidly around")
print ("in space doesn't induce any change to the electronic wave function.")

# 3. <0|d1/dR>
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=2 1
#    ```
nac = mc_nacs.kernel (state=(1,0))
print ("\nThe NACs are antisymmetric with respect to state transposition.")
print ("NAC <0|d1/dR>:\n", nac)

# 4. <0|d1/dR> w/ ETFs
#    Equivalent OpenMolcas input:
#    ```
#    &ALASKA
#    NAC=2 1
#    NOCSF
#    ```
nac = mc_nacs.kernel (state=(1,0), use_etfs=True)
print ("NAC <0|d1/dR> w/ ETFs:\n", nac)


# 5. <1|d0/dR>*(E1-E0) = <0|d1/dR>*(E0-E1)
#    I'm not aware of any OpenMolcas equivalent for this, but all the information
#    should obviously be in the output file, as long as you aren't right at a CI.
nac_01 = mc_nacs.kernel (state=(0,1), mult_ediff=True)
nac_10 = mc_nacs.kernel (state=(1,0), mult_ediff=True)
print ("\nNACs diverge at conical intersections (CI). The important question")
print ("is how quickly it diverges. You can get at this by calculating NACs")
print ("multiplied by the energy difference using the keyword 'mult_ediff=True'.")
print ("This yields a quantity which is symmetric wrt state interchange and is")
print ("finite at a CI.")
print ("NAC <1|d0/dR>*(E1-E0):\n", nac_01)
print ("NAC <0|d1/dR>*(E0-E1):\n", nac_10)

# 6. <1|d0/dR>*(E1-E0) w/ ETFs
#    For comparison with 7 below.
nac_01 = mc_nacs.kernel (state=(0,1), mult_ediff=True, use_etfs=True)
nac_10 = mc_nacs.kernel (state=(1,0), mult_ediff=True, use_etfs=True)
print ("\nUnlike the SA-CASSCF case, using both 'use_etfs=True' and")
print ("'mult_ediff=True' DOES NOT reproduce the first derivative of the")
print ("off-diagonal element of the potential matrix. This is because the")
print ("model-state contribution generates a symmetric contribution to the")
print ("response equations and changes the values of the Lagrange multipliers,")
print ("even if the Hellmann-Feynman part of the model-state contribution is")
print ("omitted. You can get the gradients of the potential couplings by")
print ("passing a tuple to the gradient method instance instead.")
print ("<1|d0/dR>*(E1-E0) w/ ETFs:\n",nac_01)
print ("<0|d1/dR>*(E0-E1) w/ ETFs:\n",nac_10)

# 7. <1|dH/dR|0>
#    THIS is the quantity one uses to optimize MECIs
mc_grad = mc.nuc_grad_method ()
v01 = mc_grad.kernel (state=(0,1))
v10 = mc_grad.kernel (state=(1,0))
print ("<1|dH/dR|0>:\n", v01)
print ("<0|dH/dR|1>:\n", v10)

