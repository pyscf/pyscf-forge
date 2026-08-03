import numpy as np
import datetime
import sys
import os
import time
from pyscf.dft.uks import UKS
from pyscf.dft.roks import ROKS
from pyscf.pbc.dft.kroks import KROKS
from pyscf.pbc.dft.kuks import KUKS
from .Logger import Logger
from .rttddft01 import rttddftMOL,rttddftPBC,get_HOMO,get_SOMO,get_LUMO,calc_gs,set_mo_occ,update_Sinv,energy_tot,get_Rnuc,get_hcore,get_logger,set_nOrbs,set_logger,calc_phases,print_eorbocc,get_populations,update_nOrbs,update_timing,tocomplex
from pyscf import __config__
from pyscf.pbc.dft import gen_grid
from .Dbglogger import Dbglogger
from pyscf.pbc.scf.kuhf import get_occ as kuhf_get_occ

# 2020.12.12 : check SPIN multiplicity: mo_occ|mo_coeff|dm
# calc_gs : OK
# set_mo_occ : OK
# get_HOMO ... 
# 
#
 
def energy_elec_super(this, dm=None, h1e=None, vhf=None, verbose=False):
    ### if h1e is None: h1e=this.get_hcore()
    if( dm is None ):
        dm = this.make_rdm1()
    if( not this._pbc):
        ### print("energy_elec_super:dm:",end="");print(np.shape(dm))
        ### print("energy_elec_super:h1e",end="");print(np.shape(h1e))
        retv = this.get_super().energy_elec(dm=dm,h1e=h1e,vhf=vhf)
    else:
        retv = this.get_super().energy_elec(dm_kpts=dm,h1e_kpts=h1e,vhf=vhf);## energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None)

    ##dbgng=True
    ##if(dbgng):
    ##    if vhf is None: vhf = get_veff(dm=dm)
    ##    ec=getattr(vhf, 'ecoul', None)
    ##    if(ec is not None):
    ##        print("Ecoulomb:%f %f"%(ec.real,ec.imag))
        
    if(verbose):
        Logger.write(this._logger,"#rttddftMOL:energy_elec_super:%f"%(retv) )
    return retv

## get_occ : this formally applies to non-pbc as well as PBC
## def get_occ(self, mo_energy_kpts=None, mo_coeff_kpts=None)
def get_occ(this, mo_energy=None, mo_coeff=None):
    if( this._fix_occ ):
        assert (this._mo_occ is not None),""
        return this._mo_occ
    else:
        return this.get_super().get_occ(mo_energy,mo_coeff)

## UKS... get_ovlp|get_hcore|get_veff|get_fock|get_occ|energy_elec|energy_tot|get_rdm1
##   class UKS(rks.KohnShamDFT, uhf.UHF):
##      __init__(self, mol, xc='LDA,VWN'):            #@uks.py:
##      get_ovlp
##      get_hcore
##      get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1) #@uks.py
##      get_fock
##      get_occ
##      get_rdm1
##      energy_elec(ks, dm=None, h1e=None, vhf=None): #@uks.py:
##      energy_tot
##      kernel
##  class KUKS(rks.KohnShamDFT, kuhf.KUHF)
##      energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None): #@kuks.py
##      get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, kpts=None, kpts_band=None):
class rttddftMOL_UKS(UKS):
    def __init__(self, mol, xc='LDA,VWN',td_field=None,dipolegauge=False,logger=None):
        self._pbc=False
        self._spinrestriction='U'
        self._td_field=td_field
        self._time_AU=0.0
        self._fix_occ=False
        self._logger=logger
        self._step=None
        self._mo_occ=None
        if( dipolegauge ):
            assert (td_field is not None),"field is missing"
            self.dipolegauge=(td_field is not None); 
            self.velocitygauge=False
        else:
            self.velocitygauge=(td_field is not None); 
            self.dipolegauge=False;
        self._sINV=None
        self._Sinvrt=None
        self._Ssqrt=None
        self.nkpt=None; self.nAO=None; self.nMO=None;
        self._fixednucleiapprox=None  ## 20210610:fixednucleiapprox
        self._calc_Aind=False
        UKS.__init__(self,mol,xc)  ## __init__(self, mol, xc='LDA,VWN')
    get_HOMO = get_HOMO
    get_SOMO = get_SOMO
    get_LUMO = get_LUMO

    update_nOrbs = update_nOrbs        
    set_nOrbs = set_nOrbs

    calc_gs = calc_gs
    set_mo_occ = set_mo_occ
    def get_super(self):
        return super()
    update_Sinv = update_Sinv

    energy_tot = energy_tot

    energy_elec = energy_elec_super

    get_Rnuc = get_Rnuc

    get_hcore = get_hcore

    get_logger = get_logger

    set_logger = set_logger

    get_populations = get_populations

    calc_phases = calc_phases

    print_eorbocc = print_eorbocc

    get_occ = get_occ

#    def get_occ(self, mo_energy=None, mo_coeff=None):
#        if( self._fix_occ ):
#            assert (self._mo_occ is not None),""
#            return self._mo_occ
#        else:
#            return super().get_occ(mo_energy,mo_coeff)

class rttddftMOL_ROKS(ROKS):
    def __init__(self, mol, xc='LDA,VWN',td_field=None,dipolegauge=False,logger=None):
        self._pbc=False
        self.nkpt=None; self.nAO=None; self.nMO=None;
        self._spinrestriction='O'
        self._td_field=td_field
        self._time_AU=0.0
        self._fix_occ=False
        self._logger=logger
        self._step=None
        self._mo_occ=None
        if( dipolegauge ):
            assert (td_field is not None),"field is missing"
            self.dipolegauge=(td_field is not None); 
            self.velocitygauge=False
        else:
            self.velocitygauge=(td_field is not None); 
            self.dipolegauge=False;
        self._sINV=None
        self._Sinvrt=None
        self._Ssqrt=None
        self._calc_Aind=False
        ROKS.__init__(self,mol,xc)
    get_HOMO = get_HOMO
    get_SOMO = get_SOMO
    get_LUMO = get_LUMO

    update_nOrbs = update_nOrbs        
    set_nOrbs = set_nOrbs

    calc_gs = calc_gs
    set_mo_occ = set_mo_occ
    def get_super(self):
        return super()
    update_Sinv = update_Sinv

    energy_tot = energy_tot

    energy_elec = energy_elec_super

    get_Rnuc = get_Rnuc

    get_hcore = get_hcore

    get_logger = get_logger

    set_logger = set_logger

    get_populations = get_populations

    calc_phases = calc_phases

    print_eorbocc = print_eorbocc

    get_occ = get_occ

class rttddftPBC_UKS(KUKS):
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',td_field=None, dipolegauge=False, logger=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), calc_Aind=False):
        self._pbc=True
        self._spinrestriction='U'
        self._td_field=td_field
        self._time_AU=0.0
        self._step=None
        Logger.write_once( None, "rttddft_xc:"+xc, 
                content="rttddftPBC:"+xc+" \t\t "+str(datetime.datetime.now()),
                fnme="rttddft_xc.log", append=True )
        self._fix_occ=False
        self._mo_occ=None
        self._Egnd = None

        self._logger=logger
        if( dipolegauge ):
            assert (td_field is not None),"field is missing"
            self.dipolegauge=(td_field is not None); 
            self.velocitygauge=False
        else:
            self.velocitygauge=(td_field is not None); 
            self.dipolegauge=False;
        self.timing=None
        self._sINV=None
        self._Sinvrt=None
        self._Ssqrt=None
        self.nkpt=None; self.nAO=None; self.nMO=None;

        self._fixednucleiapprox=None  ## 20210612:fixednucleiapprox
        self.vloc1_=None              ## 20210612:fixednucleiapprox
        self.vloc2_=None              ## 20210612:fixednucleiapprox
        self.int1e_kin_=None          ## 20210612:fixednucleiapprox
        self._calc_Aind=calc_Aind

        KUKS.__init__(self, cell, kpts=kpts, xc=xc,exxdiv=exxdiv)


##    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
##                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):

    get_HOMO = get_HOMO
    get_SOMO = get_SOMO
    get_LUMO = get_LUMO

    update_nOrbs = update_nOrbs        
    set_nOrbs = set_nOrbs

    calc_gs = calc_gs
    set_mo_occ = set_mo_occ
    def get_super(self):
        return super()
    update_Sinv = update_Sinv

    energy_tot = energy_tot

    energy_elec = energy_elec_super

    get_Rnuc = get_Rnuc

    get_hcore = get_hcore

    get_logger = get_logger

    set_logger = set_logger

    get_populations = get_populations

    calc_phases = calc_phases

    print_eorbocc = print_eorbocc

    def get_occ(self, mo_energy_kpts=None, mo_coeff_kpts=None):
        Dbglogger.write("#rttddftPBC_UHF:get_occ:")
        if( self._fix_occ ):
            assert (self._mo_occ is not None),""
            return self._mo_occ
        else:
#                      get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None) in file kuhf.py
            return kuhf_get_occ(self,mo_energy_kpts,mo_coeff_kpts)
    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, kpts=None, kpts_band=None):
        cput0=time.time();
        retv= kuks_get_veff(self,cell,dm=dm,dm_last=dm_last,vhf_last=vhf_last,hermi=hermi,kpts=kpts,kpts_band=kpts_band)
        cput1=time.time();
        update_timing(self,"get_veff",cput1-cput0)
        return retv

def kuks_get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    from pyscf.pbc.dft import multigrid
    from pyscf.lib import logger as pyscf_logger
    from pyscf import lib as pyscf_lib
    '''
    COPY of kuks.get_veff  we need complex conversion of vx
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (time.clock(), time.time())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_uks(ks.with_df, ks.xc, dm, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        pyscf_logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = pyscf_logger.timer(ks, 'vxc', *t0)
        return vxc

    # ndim = 4 : dm.shape = ([alpha,beta], nkpts, nao, nao)
    ground_state = (dm.ndim == 4 and dm.shape[0] == 2 and kpts_band is None)

    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm, ks.grids, kpts)
        t0 = pyscf_logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, dm, 0,
                                        kpts, kpts_band)
        pyscf_logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = pyscf_logger.timer(ks, 'vxc', *t0)

    weight = 1./len(kpts)

    if not hybrid:
        vj = ks.get_j(cell, dm[0]+dm[1], hermi, kpts, kpts_band)
        ## XXX XXX 
        if( (np.array(vxc).dtype == np.float64 or np.array(vxc).dtype == float) and 
            (np.array(vj).dtype == np.complex128 or np.array(vj).dtype == complex) ):
            ### print("#kuks_get_veff:complex conversion ...",flush=True)
            Ndim_vxc=np.shape(vxc)  # [2][nKpt][nAO][nAO]
            ### print("#vxc:"+str(Ndim_vxc));#vxc:(2, 1, 16, 16)
            if( len(Ndim_vxc) < 4 ):
                Ndim_vxc=[ np.shape(vxc[sp]) for sp in range(2) ]
                print(Ndim_vxc);assert False,""
            vxc=tocomplex(vxc)

        vxc += vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vj = vj[0] + vj[1]
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc += vj - vk

        if ground_state:
            exc -= (np.einsum('Kij,Kji', dm[0], vk[0]) +
                    np.einsum('Kij,Kji', dm[1], vk[1])).real * .5 * weight

    if ground_state:
        ecoul = np.einsum('Kij,Kji', dm[0]+dm[1], vj).real * .5 * weight
    else:
        ecoul = None

    vxc = pyscf_lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

class rttddftPBC_ROKS(KROKS):
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',td_field=None, dipolegauge=False, logger=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'), calc_Aind=False):
        self._pbc=True
        self._spinrestriction='O'
        self._td_field=td_field
        self._time_AU=0.0
        self._fix_occ=False
        self._logger=logger
        self._step=None
        self._mo_occ=None
        if( dipolegauge ):
            assert (td_field is not None),"field is missing"
            self.dipolegauge=(td_field is not None); 
            self.velocitygauge=False
        else:
            self.velocitygauge=(td_field is not None); 
            self.dipolegauge=False;
        self._sINV=None
        self._Sinvrt=None
        self._Ssqrt=None
        self.nkpt=None; self.nAO=None; self.nMO=None;
        self._calc_Aind=calc_Aind
        KROKS.__init__(self,cell, kpts=kpts, xc=xc,exxdiv=exxdiv)
#    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
#                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):

    get_HOMO = get_HOMO
    get_SOMO = get_SOMO
    get_LUMO = get_LUMO

    update_nOrbs = update_nOrbs        
    set_nOrbs = set_nOrbs

    calc_gs = calc_gs
    set_mo_occ = set_mo_occ
    def get_super(self):
        return super()
    update_Sinv = update_Sinv

    energy_tot = energy_tot

    energy_elec = energy_elec_super

    get_Rnuc = get_Rnuc

    get_hcore = get_hcore

    get_logger = get_logger

    set_logger = set_logger

    get_populations = get_populations

    calc_phases = calc_phases

    print_eorbocc = print_eorbocc

class RTTDDFT_:
    @staticmethod
    def is_rttddft(item):
        return ( isinstance(item,rttddftMOL) or isinstance(item,rttddftPBC) or \
        isinstance(item,rttddftMOL_ROKS) or isinstance(item,rttddftPBC_ROKS) or \
        isinstance(item,rttddftMOL_UKS) or isinstance(item,rttddftPBC_UKS) )
