import time
import numpy as np
import os
from pyscf.rttddft.eldyn import calc_OAS
from pyscf.rttddft.Moldyn import Moldyn,calc_mo_energies
from pyscf import gto
from pyscf.pbc import gto as pbcgto
from pyscf.rttddft.physicalconstants import PhysicalConstants
from pyscf.rttddft.utils import read_xyzf
from pyscf.rttddft.rttddft_common import rttddft_common
## Molecular calculation ##
TASK="OAS_noxyzf"

## YOU CAN MODIFY HERE TO TEST either h-BN or CH4 ....
target_system="h-BN" # "CH4" # ; 
# possible restart 
restart=None
#restart=1
params_restart={
 "moldyn_pscf":"OAS_noxyzf_h-BN_gth-dzvp_PBE,PBE_k3.3.1_gth-pbe_FFTDF_VG_Ez1.0e-03_dt8as01.pscf",
 "tm_fs_offset":0.16,
 "dmfilepath":"OAS_noxyzf_h-BN_gth-dzvp_PBE,PBE_k3.3.1_gth-pbe_FFTDF_VG_Ez1.0e-03_dt8as_dm.dat"}


df=None; pseudo=None; basis=None; mol=None; cell=None;

#1. Target system / DFT
if( target_system == "CH4"):
    PBC=False
    ### xyzf="CH4_acct_bfgs_optg.xyz"
    basis="aug-ccpvTz"
    gauge_LorV="L"
    Evector=(0.0, 0.0, 1.0e-3)
    DFT='RKS';xc="LDA,VWN";
    mol = gto.M(atom = 'C      -0.00002631       0.00024432       0.00007638;'\
    +'H       1.09719302      -0.00099542      -0.00067323;'\
    +'H      -0.36460184       1.03512412       0.00149901;'\
    +'H      -0.36573517      -0.51759524       0.89561804;'\
    +'H      -0.36696120      -0.51556008      -0.89613718', basis="aug-ccpvTz")
elif( target_system == "h-BN"):
    PBC=True
    ### xyzf="hBN_monolayer_a250.4pm_1x1x1.xyz"
    basis="gth-dzvp"
    gauge_LorV='V'
    ## Evector=(0.0, 1.0e-3, 0.0)
    Evector=(0.0, 0.0, 1.0e-3)
    DFT='RKS';xc="PBE,PBE";df="FFTDF";pseudo="gth-pbe"
    ## nKpoints=[9,9,1];cell_dimension=2
    nKpoints=[3,3,1];cell_dimension=2
    cell = pbcgto.Cell()
    cell.build(
        atom = '''B       0.00000000       0.00000000       0.00000000; N       1.44568507       0.00000000       0.00000000''',
        a=np.array([ [2.168528, -1.252000, 0.000000], [2.168528, 1.252000, 0.000000],  [0.000000, 0.000000, 20.00] ]),
        basis = basis,
        pseudo = pseudo)
    ### dict_latticevectors={"a":None};R,Sy=read_xyzf(xyzf,dict=dict_latticevectors)
else:
    assert False,"Please set params manually"

#print(type(Evector))
#Evector2=[0,0,1e-3]
#print(type(Evector2))
#print(Evector2.copy())
#print(Evector.copy())  # ERROR
#if( isinstance(Evector,tuple) ):
#    print("tuple->list")
#    print(list(Evector).copy()) # OKAY
# Evector3=np.array(Evector2) 
# print(Evector3.copy())  # OKAY
# assert False,""

#3. rt-propagation
str_dt_as="8"
tm_fs_step=float(str_dt_as)*1e-3
tm_fs_end=24.0
propagator="CN"

#4. externalfield

#5. dump file output
Nstep_dump=10 #100
tsecond_dump=3600  # 3600*47
dumpfilename=None

#6. additional output control
Nstep_calcEng=None #1 # None  # or some integer value
Nstep_calcDOS=None #1 # None
check_timing=None



#6. misc 

if( basis is None ):
    if( mol is not None ):
        basis=mol.basis
    elif( cell is not None):
        basis=cell.basis
assert basis is not None,""

strfield='E'
Eabs=np.sqrt( np.vdot(Evector,Evector) )
STRxyz=('x','y','z')
for k in range(3):
    if( abs(Evector[k])/Eabs > 1e-3 ):
        strfield=strfield+STRxyz[k]
strfield=strfield+("%6.1e"%(Eabs)).strip()

if( not PBC ):
    job="_".join( [TASK, target_system, basis, xc, gauge_LorV+'G', strfield, "dt"+str_dt_as+"as"])
else:
    strKpoints='.'.join( [ '%d'%(nkj) for nkj in nKpoints ] )
    job="_".join( [TASK, target_system, basis, xc, "k"+strKpoints, pseudo, df, gauge_LorV+'G', strfield, "dt"+str_dt_as+"as"])

if(restart is not None):
    job=job+"_%02d"%(restart)
    
rttddft_common.Setup()
rttddft_common.set_job(job)

## print(job);assert False,""+job

dmfilepath = job+"_dm.dat"
dumpfilename = job

dt_AU=tm_fs_step/PhysicalConstants.aut_in_femtosec()

tm_fs_offset=( None  if( restart is None) else params_restart["tm_fs_offset"])


WCt00=time.time()  

## some parameter check ...
if( PBC ):
    assert nKpoints is not None,"nKpoints"



set_occ=( None if(restart is None ) else True )
if(restart is None):
    #1. Construct Moldyn object
    if( not PBC ):
        moldyn=Moldyn( pbc=False, basis=basis, molecule=mol,                #xyzf=xyzf,
                       DFT=DFT,   xc=xc,     gauge_LorV=None,
                       fixednucleiapprox=True, dt_AU=dt_AU)
    else:
        moldyn=Moldyn( pbc=True, basis=basis, pseudo=pseudo, unitcell=cell, #xyzf=xyzf, 
                       gauge_LorV=None,df=df, DFT=DFT, xc=xc,
                       nKpoints=nKpoints, cell_dimension=cell_dimension,
                       dt_AU=dt_AU)
    
    #2. Get rttddft
    rttddft=moldyn.gen_rttddft(set_occ=set_occ)
    #3. calc GS
    moldyn.calc_gs(rttddft)
    ## R,Sy=read_xyzf(xyzf);distinctAtms=distinct_set(Sy);
    if( PBC ):
        if( gauge_LorV == 'V' ):
            rttddft.set_constantfield(True)

else:
    # required parameters for restart calc.
    assert ( params_restart["moldyn_pscf"] is not None),""
    assert os.path.exists(params_restart["moldyn_pscf"]),""
    assert ( params_restart["tm_fs_offset"] is not None),""
    assert ( params_restart["dmfilepath"] is not None),""

    moldyn=Moldyn.load(params_restart["moldyn_pscf"])
    rttddft = moldyn.gen_rttddft(set_occ=set_occ)

    tm_fs_offset=float( params_restart["tm_fs_offset"] )
    tm_au_offset=tm_fs_offset/PhysicalConstants.aut_in_femtosec()
    dmfilepath=params_restart["dmfilepath"]

    if( Nstep_calcEng is not None ):
        if( rttddft.mo_occ is None ):
            rttddft.mo_occ = rttddft._mo_occ
        calc_mo_energies(moldyn,rttddft,update_rttddft=True)

calc_OAS(moldyn, rttddft, Evector, gauge_LorV, dmfilepath, 
         tm_fs_end=tm_fs_end, tm_fs_step=tm_fs_step, tm_fs_offset=tm_fs_offset,
         logfile="OAS_"+job+".log",
         propagator=propagator, Nstep_dump=Nstep_dump, tsecond_dump=tsecond_dump, WCt00=WCt00,
         dumpfilename=dumpfilename, Nstep_calcEng=Nstep_calcEng,
         Nstep_calcDOS=Nstep_calcDOS)

