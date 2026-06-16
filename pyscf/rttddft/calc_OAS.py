import sys
import numpy as np
import os
import os.path
import time
from scipy import linalg

from pyscf.pbc.gto import intor_cross
from pyscf.pbc.dft import KRKS
import pyscf.gto as molgto
from pyscf.gto import Mole
from pyscf.pbc.gto import Cell
from pyscf.gto import Mole
from pyscf.pbc import gto, scf, dft, df
from pyscf.pbc.scf.khf import get_occ as khf_get_occ
from pyscf.pbc.scf.khf import make_rdm1 as khf_make_rdm1

from .pyscf_common import pyscf_common

from .MPIutils01 import MPIutils01

from .physicalconstants import physicalconstants
from .utils import read_inpf,read_xyzf,dump_zNarray,distinct_set,get_strPolarization,parse_doubles,parse_ints
from .laserfield import *

from .diis01 import DIIS01
#from rttddft01 import load_Aind_over_c,rttddftPBC, rttddftMOL,serialize_rttddft,get_dipole
from .rttddft01 import load_Aind_over_c,rttddftPBC, rttddftMOL,get_dipole
from .eldyn import calc_OAS
from .Moldyn import Moldyn,calc_mo_energies,get_FockMat
from .serialize import serialize,serialize_CDIIS
from .bandstructure import calc_bandstructure,plot_dispersion_kLine,calc_ldos
from .GEAR import initz_rGEARvec
from .with_fftdf_get_pp import dbg_fftdf_get_pp

#PhysicalConstants_=get_constants();
aut_in_femtosec= physicalconstants.aut_in_femtosec; #PhysicalConstants_['_aut']*1.0e+15 ## i.e. .00241888 femtosec  never use ['AUT'] 
HARTREEinEV = physicalconstants.HARTREEinEV;        #PhysicalConstants_['Hartree']
KelvinINeV=8.6173324e-5


params_inp=None

argv=sys.argv
narg=len(argv)
iarg=0        # start from argv[1] (not 0)
help=False
while(iarg<narg-1):
    iarg+=1;
    arg=argv[iarg].strip();
    if( arg.startswith("-") ):
        if( arg.startswith("--help")):
            help=True;continue
        assert False,"unknown option:"+arg
    else:
        if( params_inp is None ):
            params_inp=arg;continue
if( params_inp is None ):
    params_inp = "params.inp"
    if( not os.path.exists(params_inp) ):
        help=True;

params={"xyzf":None,"name":None,"timestep_as":None,"PBC":None,"nstep_CN":20,
        "tm_fs_offset":None,"df":None,"exp_to_discard":None,"xc":"LDA,VWN",
        "basis":None,"pseudo":None,"dimension":None,"mesh":None,"charge":None,
        "nKpoints":None,"spin":None,"rcut":None,"propagator":None,"spinrestriction":None,
        "Evector":None, "gauge":None, "tm_fs_end":24.0, "dmfilepath":None,"check_timing":None,
        "bandstructure":None,"Nstep_calcDOS":None,"check_timing":None,"cell_precision":None,
        'DIIS_nstep_thr':None, 'dm2diff_TOL':None,"calc_pbc_overlaps_output":None,
        "kickfield_hfWidth_fs":None,"load_Aind_over_c":None,"nr_rks_par_LDA":None,
        "calc_Aind":None,"fixednucleiapprox":"True","alpha_Aind_over_c":None,
        "partialupdate_hcore":None,"dimerization":None,"Nkpts_UPL":None,
        "auxbasis":None, "calc_gsDOS":None,
        "smearing":None,"Temperature":None,
        ### "iflag_use_ZFpsp":None,
        "Aind_over_c_Nintpl":20,"Sinvsqrt_devtol":None,"Nstep_calcEng":None,
        "tm_hour_to_dump":47.2, "Nstep_dump":None, "moldyn_pscf":None,"restart":None,"levelshift":None,
        "tmKick_AU":None, "Nstep_Kick":1, "Nstep_calcEng":None, "observables":None, "testrun":None }
default_propagator="CN";
if( help ):
    print("#Please set following parameters on "+params_inp)
    print("-----------------------------------")
    for ky in params:
        if( params[ky] is None ):
            print(ky+":")
        else:
            print("#"+ky+":"+str(params[ky]))
    assert False,"..."
WCt00=time.time()  ## WallClockTime
read_inpf(params,params_inp)
required_fields=["xyzf","name","timestep_as","basis","PBC","gauge"]
for ky in required_fields:
    assert (params[ky] is not None),"#param "+ky+" is missing in paramf";
    
strAind=("" if( params["calc_Aind"] is None ) else \
          ("" if(int(params["calc_Aind"])==0) else "Aind"+params["calc_Aind"].strip()+\
            "_alph"+("1.0" if(params["alpha_Aind_over_c"] is None) else params["alpha_Aind_over_c"])) )

if( params["Nstep_calcDOS"] is not None):
    params["Nstep_calcDOS"]=int(params["Nstep_calcDOS"])
if( params["check_timing"] is not None):
    params["check_timing"]=eval(params["check_timing"])
if( params["fixednucleiapprox"] is not None ):
    params["fixednucleiapprox"]=eval( params["fixednucleiapprox"] )
if( params["alpha_Aind_over_c"] is not None ):
    params["alpha_Aind_over_c"]=float( params["alpha_Aind_over_c"] )

Evector=parse_doubles( params["Evector"] )
gauge_DorV=params["gauge"];strField=""
params["gauge"]=params["gauge"].strip()
if( params["gauge"] in [ "D", "dipole", "L", "length", "lg"]):
    gauge_DorV = 'D';
    if(not("lg" in params["name"])):
        strField="_lg"+get_strPolarization(Evector,strxyz=['X','Y','Z'],default="")
elif( params["gauge"] in [ "V", "velocity", "vg"]):
    gauge_DorV = 'V';
    if(not("vg" in params["name"])):
        strField="_vg"+get_strPolarization(Evector,strxyz=['X','Y','Z'],default="")
else:
    assert False,"illegal gauge input"


PBC=eval( params["PBC"] )
name=params["name"]
strKpoints=""
if( PBC ):
    if( params["nKpoints"] is not None ):
        nKpoints = parse_ints( params["nKpoints"] )
        strKpoints="_K%dx%dx%d"%(nKpoints[0],nKpoints[1],nKpoints[2])

job= name + "_" + strAind+"_"+strKpoints + "_" + params["basis"] + ( str(params["xc"]).replace(",","-") ) \
    + ("" if( params["df"] is None or params["df"] == "GDF") else  params["df"]) \
    + "_dt"+params["timestep_as"]+ strField

if( "propagator" in params):
    if( params["propagator"] is not None):
        job=name + "_" + strAind + "_" + params["basis"] + "_"+params["propagator"]+"dt"+params["timestep_as"]
    else:
        params.pop("propagator")

Temperature_Kelvin=None;
if( params["smearing"] is not None or params["Temperature"] is not None ):
    Temperature_Kelvin=float( params["Temperature"] if(params["Temperature"] is not None) else \
                              params["smearing"]  )
    job+=( "_%03dK"%(int(round(Temperature_Kelvin))) if(Temperature_Kelvin>1) else \
           ("%eK"%(Temperature_Kelvin)).strip())

restart=None
if( params["restart"] is not None ):
    restart=int( params["restart"] )
if( restart is not None ):
    job=job+"_%02d"%(restart)

pyscf_common.set_params(params,job)
MPIutils01.setup()

cell_dimension=None
if( params["dimension"] is not None):
    params["dimension"]=int(params["dimension"])
    cell_dimension=params["dimension"]
if( params["spin"] is not None):
    params["spin"]=int(params["spin"])
if( params["charge"] is not None):
    params["charge"]=float(params["charge"])
if( params["cell_precision"] is not None):
    params["cell_precision"]=float(params["cell_precision"])
if( params["exp_to_discard"] is not None):
    params["exp_to_discard"] = float( params["exp_to_discard"] )
if( params["mesh"] is not None):
    params["mesh"]=parse_ints( params["mesh"] )
if( params["rcut"] is not None):
    params["rcut"]=float( params["rcut"] )
if( params["Nstep_calcEng"] is not None):
    params["Nstep_calcEng"]=int( params["Nstep_calcEng"] )
if( params["calc_Aind"] is not None):
    params["calc_Aind"]=int( params["calc_Aind"] )
dt_as=float(params["timestep_as"])
dt_AU=(dt_as*1.0e-3)/physicalconstants.aut_in_femtosec


spinrestriction=params["spinrestriction"];DFT=None
if( params["spinrestriction"] is None ):
    assert ( (params["spin"] is None) or (params["spin"]==0) ),"please set spinrestriction"
    spinrestriction='R';DFT="RKS"
else:
    DFT=('RKS' if(spinrestriction=='R') else ('UKS' if(spinrestriction=='U') else ('ROKS' if(spinrestriction=='O') else None)))
    
assert DFT is not None,"wrong spinrestriction:"+spinrestriction


moldyn=None; tm_fs_offset=None; dmfilepath=None
if( params["restart"] is not None):
    assert ( params["moldyn_pscf"] is not None),""
    assert ( params["tm_fs_offset"] is not None),""
    assert ( params["dmfilepath"] is not None),""

    assert os.path.exists(params["moldyn_pscf"]),""
    moldyn=Moldyn.load(params["moldyn_pscf"])
    print("#tdMO:af_Moldyn.load:%s .."%( str(moldyn.tdMO[0][0]) ))
    # dump_zNarray(moldyn.tdMO, fpath=params["moldyn_pscf"]+"_load_tdMO.dmp")
    tm_fs_offset=float( params["tm_fs_offset"] )
    tm_au_offset=tm_fs_offset/aut_in_femtosec
    assert (abs(moldyn.time_AU-tm_au_offset)<1.0e-5),"%16.8f / %16.8f"%(moldyn.time_AU,tm_au_offset)
    dmfilepath=params["dmfilepath"]
    
else:
    assert ( params["moldyn_pscf"] is None),""
    assert params["tm_fs_offset"] is None,""
    
    ##
    ## moldyn.gauge_LorV is set None at this point
    ## in velocity gauge calculation, it will later be set 'V' inside -calc_OAS- subroutine...
    ##
    if( not PBC ):
        moldyn=Moldyn( pbc=PBC, xyzf=params["xyzf"], basis=params["basis"],pseudo=params["pseudo"],
                       exp_to_discard=params["exp_to_discard"],spin=params["spin"], charge=params["charge"],mesh=params["mesh"],
                       rcut=params["rcut"],gauge_LorV=None,df=params["df"],DFT=DFT,
                       fixednucleiapprox=params["fixednucleiapprox"],
                       Temperature_Kelvin=Temperature_Kelvin,
                       xc=params["xc"], calc_Aind=None, dt_AU=dt_AU)
    else:
        assert (params["nKpoints"] is not None),"nKpoints"
        nKpoints = parse_ints( params["nKpoints"] )
        dict={"a":None};R,Sy=read_xyzf(params["xyzf"],dict=dict)
        moldyn=Moldyn( pbc=PBC, xyzf=params["xyzf"], basis=params["basis"], pseudo=params["pseudo"],
                       exp_to_discard=params["exp_to_discard"],spin=params["spin"], charge=params["charge"],mesh=params["mesh"],
                       rcut=params["rcut"],gauge_LorV=None,df=params["df"], DFT=DFT,
                       a=dict["a"],nKpoints=nKpoints, cell_dimension=cell_dimension,
                       check_timing=params["check_timing"],cell_precision=params["cell_precision"],
                       fixednucleiapprox=params["fixednucleiapprox"],
                       Temperature_Kelvin=Temperature_Kelvin,
                       xc=params["xc"], calc_Aind=params["calc_Aind"], dt_AU=dt_AU,alpha_Aind_over_c=params["alpha_Aind_over_c"] )
    dmfilepath=params["dmfilepath"] # most likely to be None 
    
set_occ=( None if( params["restart"] is None ) else True )
rttddft = moldyn.gen_rttddft(set_occ=set_occ)

if( PBC ):
    if( gauge_DorV == 'V' ):
        rttddft.set_constantfield(True)

if( params["restart"] is not None ):
    ## 2021.08.25  --- it looks that rttddft thus constructed causes error if we invoke get_fermi for example
    ##                 since mo_energy_kpts field is None ( it works perfectly for calc_OAS but print_TDeorbs fails )
    ## here we copy from restart step in eldyn_main.py ...
    if( params["Nstep_calcEng"] is not None ):
        if( rttddft.mo_occ is None ):
            rttddft.mo_occ = rttddft._mo_occ
        calc_mo_energies(moldyn,rttddft,update_rttddft=True)
        

if( params["levelshift"] is not None ):
    rttddft.level_shift=float(params["levelshift"])

if( params["load_Aind_over_c"] is not None ):
    print("#load int Aind_over_c :"+str(params["load_Aind_over_c"]),flush=True)
    time_AU=moldyn.time_AU
    Aind_over_c=load_Aind_over_c( rttddft, time_AU, rttddft._dt_AU, params["load_Aind_over_c"] )
    rttddft._Aind_tmAU = time_AU
    assert rttddft._dt_AU>1.0e-8 and rttddft._Aind_tmAU>1.0e-6,""
    rttddft._Aind_Gearvc_C, rttddft._Aind_Gearvc_P = initz_rGEARvec( rttddft._dt_AU, np.array(Aind_over_c), np.zeros([3]) ,np.zeros([3]) ,3)
    print("#load_Aind_over_c done",flush=True)

observables=[]
if( params["observables"] is not None ):
    observables=( params["observables"].strip() ).split(',');
    No=len(observables);
    for Io in range(No):
        observables[Io]=observables[Io].strip()

if( params["testrun"] is not None ):
    assert False,"testrun"
    
if( params["restart"] is None ):
    moldyn.calc_gs(rttddft)
    
    if( "bandstructure" in params ):
        bs=params[ "bandstructure" ]
        if( bs is not None ):
            ## for dimer dispersion, set "dimerization":2,2,2 in params.inp ...
            lattice=bs.strip()
            dmat= rttddft.make_rdm1()
            plot_dispersion_kLine(str(pyscf_common.job)+"_gs_dispersion.dat",rttddft,dmat,moldyn.BravaisVectors_au,
                                  lattice=lattice)
    if( "calc_gsDOS" in params ):
        if( params["calc_gsDOS"] is not None ):
            if( eval(params["calc_gsDOS"]) ):
                FockMat=get_FockMat(moldyn,rttddft)
                calc_ldos(rttddft, job+"_gsDOS.dat", moldyn._canonicalMOs, FockMat, de_eV=0.01,
                        moldyn=moldyn, widths_eV=[0.2123, 0.02123] )

    R,Sy=read_xyzf(params["xyzf"]);distinctAtms=distinct_set(Sy);
    moldip=[]
    dict_dipoles= get_dipole( rttddft, moldyn.tdDM,'B',filepath=None,header=None,molecular_dipole=moldip,caller="calc_OAS.L247")
    
    strdipoles="";legend_dipoles=""
    if( moldyn.spinrestriction !='U' ):
        strdipoles="%16.8f %16.8f %16.8f     %16.8f %16.8f %16.8f    %18.8f %18.8f %18.8f"%(
            dict_dipoles['dipole'][0], dict_dipoles['dipole'][1], dict_dipoles['dipole'][2],
            dict_dipoles['dipole_velocity'][0], dict_dipoles['dipole_velocity'][1], dict_dipoles['dipole_velocity'][2],
            moldip[0][0], moldip[0][1], moldip[0][2])
        legend_dipoles="%16s %16s %16s     %16s %16s %16s    %18s %18s %18s"%(
            "dipole_x","dipole_y","dipole_z","dipvel_x","dipvel_y","dipvel_z","moldip_x","moldip_y","moldip_z");
    Nat=len(Sy)

if( params["restart"] is not None):
    dump_zNarray(rttddft.mo_coeff, fpath=params["moldyn_pscf"]+"_load_tdMO_rttddft.dmp")
    print("#tdMO:af_Moldyn.load:%s .."%( str(moldyn.tdMO[0][0]) ))

check_timing=(False if(params["check_timing"] is None) else params["check_timing"])

if( dmfilepath is None ):
    dmfilepath = job+"_dm.dat"
tm_fs_end=float( params["tm_fs_end"] )
timestep_as = float( params["timestep_as"])
tm_fs_step = (1.0e-3)*timestep_as
tmKick_AU= None if(params["tmKick_AU"] is None) else float(params["tmKick_AU"])
Nstep_Kick = int( params["Nstep_Kick"])  

# Propagator :
#   The default propagator is the Crank-Nicolson propagator ("CN") 
#   We have optionally implemented the 2nd order Magnus propagator but it did not improve performance
# 
propagator=default_propagator
if( "propagator" in params ):
    if( params["propagator"] is not None):
        propagator=params["propagator"]

# Dump file options:
#   You can specify the point you output the dump file (file needed for restarting).
#   (i)  Every N steps
#   (ii) N hour after starting calculation
#
Nstep_dump=(None if(params["Nstep_dump"] is None) else int(params["Nstep_dump"]))
tmsecond_dump=None;
if( params["tm_hour_to_dump"] is not None ):
    hr=float( params["tm_hour_to_dump"] )
    tsecond_dump=hr*60*60

calc_OAS(moldyn, rttddft, Evector, gauge_DorV, dmfilepath, 
             tm_fs_end=tm_fs_end, tm_fs_step=tm_fs_step, tm_fs_offset=tm_fs_offset,
             tmKick_AU=tmKick_AU, Nstep_Kick=Nstep_Kick,logfile="OAS_"+job+".log",
             propagator=propagator, Nstep_dump=Nstep_dump, tsecond_dump=tsecond_dump, WCt00=WCt00,
             dumpfilename="OAS_"+job, Nstep_calcEng=params["Nstep_calcEng"],
             Nstep_calcDOS=params["Nstep_calcDOS"],check_timing=check_timing, params=params )
