from .physicalconstants import physicalconstants
import numpy as np
import math
from mpi4py import MPI
# PhysicalConstants_=get_constants();

# PhysicalConstants_['_c'] >> physicalconstants.CLIGHTinMPS
# PhysicalConstants_["_aut"] >> ((physicalconstants.aut_in_femtosec)*(1e-15))
# PhysicalConstants_['Bohr'] >> physicalconstants.BOHRinANGS
# PhysicalConstants_['Hartree'] >>   physicalconstants.HARTREEinEV
def vcdiff(A,B):
    D=np.array(A)-np.array(B)
    return np.sqrt( np.vdot(D, D) )
def printout_tdfields(laser, t0_fs,t1_fs,dt_fs,fpath,Append=False,Threads=[0], h=None):
    comm=MPI.COMM_WORLD; MPIrank=comm.Get_rank(); MPIsize=comm.Get_size()
    ## PhysicalConstants_=get_constants();
    aut_in_fs= physicalconstants.aut_in_femtosec; ## PhysicalConstants_['_aut']*1.0e+15
    t0_au=t0_fs/aut_in_fs; t1_au=t1_fs/aut_in_fs; dt_au=dt_fs/aut_in_fs
    cLIGHT_AU = physicalconstants.CLIGHTinAU; ## PhysicalConstants_['_c'] * PhysicalConstants_["_aut"] / ( PhysicalConstants_['Bohr']*(1.0e-10) )
    
    print_Evecs=( hasattr(laser,"get_electricfield"))
    print_Avecs=( hasattr(laser,"get_vectorfield"))
    N=int(round( (t1_au-t0_au)/dt_au ))
    t=t0_au-dt_au
    fd1=open(fpath,('a' if(Append) else 'w'))

    string  = "#%14s %14s"%("t_au","t_fs");
    if( print_Evecs ):
        string += "      %14s %14s %14s"%("Ex","Ey","Ez")
    if( print_Avecs ):
        string += "      %14s %14s %14s"%("Ax/c","Ay/c","Az/c")
    print(string,file=fd1)

    
    for I in range(N):
        t+=dt_au
        string=" %14.4f %14.4f"%(t,t*aut_in_fs)
        E=None
        if( print_Evecs ):
            E=laser.get_electricfield(t)
            string+="      %14.6f %14.6f %14.6f"%(E[0],E[1],E[2])
        if( print_Avecs ):
            A_over_c=laser.get_vectorfield(t)/cLIGHT_AU
            string+="      %14.6f %14.6f %14.6f"%(A_over_c[0],A_over_c[1],A_over_c[2])
        if( print_Evecs and print_Avecs and (h is not None) ):
            b2 = laser.get_vectorfield(t-2*h)/cLIGHT_AU
            b1 = laser.get_vectorfield(t-h)/cLIGHT_AU
            f1 = laser.get_vectorfield(t+h)/cLIGHT_AU
            f2 = laser.get_vectorfield(t+2*h)/cLIGHT_AU
            d3p = (f1-b1)/(2*h)
            d5p = ( 8*(f1-b1) - (f2-b2) )/(12*h)
            nmrerr = vcdiff( d5p, d3p)
            diff = vcdiff( -d5p, E)
            string+="      %14.6f %14.6f %14.6f   %14.6f %14.6f %14.6f   %12.4e (nmerr~%10.2e) %r"%(\
                    -d5p[0],-d5p[1],-d5p[2], -d3p[0],-d3p[1],-d3p[2], diff, nmrerr, diff<min(1.0e-6,10*nmrerr))
        print(string,file=fd1)
    fd1.close()

def calc_lambda1eV_nm():
    return physicalconstants.LAMBDA1EVinNM;
    # PhysicalConstants_=get_constants();
    # return (PhysicalConstants_['_hplanck']*PhysicalConstants_['_c']/PhysicalConstants_['_e'])*(1e+9);

def wavelength_to_omega(wavelength_nm,eV_au_or_INVattosec):
    # PhysicalConstants_=get_constants();
    c_nmPERattosec = physicalconstants.CLIGHTinMPS*(1e-9) # PhysicalConstants_['_c'] * (1e-9)                  # (1e+9)*(1e-18)
    lambda1eV_nm=calc_lambda1eV_nm();
    assert (abs(1239.842-lambda1eV_nm)<1.0),"please check this @laserpulses.py"
    if( eV_au_or_INVattosec == 'eV'):
        return lambda1eV_nm/wavelength_nm
    elif( eV_au_or_INVattosec == 'INVattosec'):
        return 2* math.pi * c_nmPERattosec / wavelength_nm
    elif( eV_au_or_INVattosec == 'au'):
        return (lambda1eV_nm/wavelength_nm)/physicalconstants.HARTREEinEV

def omega_to_wavelength(omega_au):
    # PhysicalConstants_=get_constants();
    c_nmPERattosec = physicalconstants.CLIGHTinMPS*(1e-9) # PhysicalConstants_['_c'] * (1e-9)                  # (1e+9)*(1e-18)
    lambda1eV_nm=calc_lambda1eV_nm();
    omega_eV= omega_au*physicalconstants.HARTREEinEV
    return lambda1eV_nm/omega_eV

def FieldStrength_to_intensity(Fpeak_au):
    # PhysicalConstants_=get_constants();
    me_kg=physicalconstants.MEinKG;    bohr_m=physicalconstants.BOHRinANGS*(1.0e-10); bohr_ANGS=physicalconstants.BOHRinANGS; 
    Hartree_eV=physicalconstants.HARTREEinEV;
    aut_sec=physicalconstants.aut_in_femtosec*(1.0e-15); c_mps=physicalconstants.CLIGHTinMPS;
    e_C=physicalconstants.ECHARGEinCOULOMB

    Fpeak_VpeANGS = Fpeak_au / (bohr_ANGS / Hartree_eV)
    peakIntensity_SI = 0.50*( (Fpeak_VpeANGS*(1e+10))**2 )*( physicalconstants.EPS0inSI*physicalconstants.CLIGHTinMPS )
    peakIntensity_WpeCM2 = peakIntensity_SI*(1.0e-4)
    return peakIntensity_WpeCM2

def intensity_to_FieldStrength(peakIntensity_WpeCM2,au_or_VpeANGS):
    # PhysicalConstants_=get_constants();

    me_kg=physicalconstants.MEinKG;    bohr_m=physicalconstants.BOHRinANGS*(1.0e-10); bohr_ANGS=physicalconstants.BOHRinANGS; 
    Hartree_eV=physicalconstants.HARTREEinEV;
    aut_sec=physicalconstants.aut_in_femtosec*(1.0e-15); c_mps=physicalconstants.CLIGHTinMPS;
    e_C=physicalconstants.ECHARGEinCOULOMB


    peakIntensity_SI = peakIntensity_WpeCM2*(1.0e+4)  ## 0.5* eps * (E**2) * c
    Fpeak_VpeANGS = math.sqrt( 2*peakIntensity_SI/(physicalconstants.EPS0inSI*physicalconstants.CLIGHTinMPS))*(1e-10)

    Fpeak_au = Fpeak_VpeANGS * bohr_ANGS / Hartree_eV

    I0=0.50*( me_kg*bohr_m/(aut_sec**2)/e_C )**2 * c_mps * physicalconstants.EPS0inSI
    I0_WpCM2=I0*(1.0e-4)
    Fpeak_au2 = math.sqrt( peakIntensity_WpeCM2 / I0_WpCM2 )
    assert (abs(Fpeak_au-Fpeak_au2)<1.0e-5), "conflicting values"

    if( au_or_VpeANGS == 'VpeANGS' ):
        return Fpeak_VpeANGS
    elif( au_or_VpeANGS == 'au' ):
        return Fpeak_au
    else:
        assert False,"unknown directive:"+au_or_VpeANGS

class laserfield:
    def __init__(self):
        assert False,""
    def get_vectorfield(self,tm_AU):
        assert False,""
    def get_electricfield(self,tm_AU):
        assert False,""

    def printout(self,file,t0,t1,dt=None, Nstep=None):
        ## PhysicalConstants_=get_constants();
        bohr_ANGS=physicalconstants.BOHRinANGS;
        Hartree_eV=physicalconstants.HARTREEinEV;freq_eV=self.omega*Hartree_eV
        aut_in_fs= physicalconstants.aut_in_femtosec; ## PhysicalConstants_['_aut']*1.0e+15
        Fpeak_au = ( self.e0 if(isinstance(self,CWField01)) else  self.Fpeak)
        Fpeak_VpeANGS= Fpeak_au * Hartree_eV/bohr_ANGS
        peakIntensity_SI = ( (Fpeak_VpeANGS*(1e+10))**2 )*0.5*(PhysicalConstants_['_eps0']*physicalconstants.CLIGHTinMPS)
        peakIntensity_WpeCM2 = peakIntensity_SI*(1.0e-4)
        fd=open(file,"w")
        if( isinstance(self,gaussianpulse01) ):
            print("freq:%12.4f eV => w=%14.6f; I:%16.3f => F_0=%14.6f;  sigma=%14.6f; tc=%f %f(fs)"%(
              freq_eV, self.omega, peakIntensity_WpeCM2, self.Fpeak,  self.sigma, self.tc, self.tc*aut_in_fs), file=fd)
        elif( isinstance(self,CWField01) ):
            print("freq:%12.4f eV => w=%14.6f; I:%16.3f => F_0=%14.6f;  ts=%f %f(fs)"%(
              freq_eV, self.omega, peakIntensity_WpeCM2, Fpeak_au, self.ts, self.ts*aut_in_fs  ), file=fd)
            
        print("#%6s %14s %16s   %16s %16s %16s   %16s %16s %16s"%("step","time_au","Esqr", "A_x","A_y","A_z","E_x","E_y","E_z"),file=fd)

        if( dt is None ):
            dtref = 0.1 * 2*(np.pi)/self.omega
            if( Nstep is None ):
                dt = dtref; Nstep=int( math.ceil( (t1-t0)/dt ) )
            else:
                dt = (t1-t0)/float(Nstep)
        else:
            if( Nstep is None ):
                Nstep=int( math.ceil( (t1-t0)/dt ) )
        t=t0-dt
        for j in range(Nstep):
            t+=dt
            A=self.get_vectorfield(t)
            E=self.get_electricfield(t)
            Esqr=np.dot(E,E)
            print(" %6d %14.4f %16.8f   %16.8f %16.8f %16.8f   %16.8f %16.8f %16.8f"%(j,t,Esqr, A[0],A[1],A[2],E[0],E[1],E[2]),file=fd)
        fd.close()

class kickfield:
    def __init__(self,ElectricfieldVector,tmKick_AU=None,tmStart_AU=0.0):
        # PhysicalConstants_=get_constants();
        self.tmKick_AU = tmKick_AU
        self.tmStart_AU= tmStart_AU
        print(ElectricfieldVector)
        if( isinstance(ElectricfieldVector,tuple) ):
            self.ElectricfieldVector = list(ElectricfieldVector).copy()
        else:
            self.ElectricfieldVector = ElectricfieldVector.copy()
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ## physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )
        self.Avec1 = np.array( [ -cLIGHT_AU*ElectricfieldVector[0], -cLIGHT_AU*ElectricfieldVector[1], -cLIGHT_AU*ElectricfieldVector[2] ])

    def tostring(self,delimiter='\n'):
        ## PhysicalConstants_=get_constants();
        aut_in_attosec= physicalconstants.aut_in_femtosec*(1.0e+3); ##PhysicalConstants_['_aut']*1.0e+18
        retv="#laserfield.kickfield:E:"+ str(self.ElectricfieldVector)
        return retv
    printout_tdfields=printout_tdfields

    def printout(self,file,t0,t1,dt=None, Nstep=None):
        print("kickfield")

    def get_vectorfield(self,tm_AU):
        if(tm_AU<self.tmStart_AU):
            return np.array( [0.0, 0.0, 0.0] )
###        elif(tm_AU<=(self.tmStart_AU + self.tmKick_AU)):
###            tfac=(tm_AU-self.tmStart_AU)
###            return np.array( [ self.Avec1[0]*tfac, self.Avec1[1]*tfac, self.Avec1[2]*tfac ] )
        else:
            return np.array( [ self.Avec1[0], self.Avec1[1], self.Avec1[2] ] )
# This inherits 
# see also       C:/cygwin64/usr/src/opt_pyscf/pyscf/pyscf/pbc/scf/khf.py
# You must respect the signatures defined by the superclass
#    get_hcore(mf, cell=None, kpts=None)                 >> get_hcore(self,cell=None,kpts=None,tm_AU=None)
#    get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None)>> get_occ( ...)
#    get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
#             diis_start_cycle=None, level_shift_factor=None, damp_factor=None)
#    eig(self, h_kpts, s_kpts) @khf.py
#      foreach k: self._eigh(h_kpts[k], s_kpts[k]) and stacks the results

class kickfield01:
    def __init__(self,ElectricfieldVector,hfWidth_AU,tmKick_AU=None,tmStart_AU=0.0):
        # PhysicalConstants_=get_constants();
        self.tmKick_AU = tmKick_AU
        self.tmStart_AU= tmStart_AU
        self.hfWidth_AU= hfWidth_AU
        
        print(ElectricfieldVector)
        self.ElectricfieldVector = ElectricfieldVector.copy()
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ##  physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )
        self.Avec1 = np.array( [ -cLIGHT_AU*ElectricfieldVector[0], -cLIGHT_AU*ElectricfieldVector[1], -cLIGHT_AU*ElectricfieldVector[2] ])

    def tostring(self,delimiter='\n'):
        # PhysicalConstants_=get_constants();
        aut_in_attosec= physicalconstants.aut_in_femtosec*(1.0e+3); ##= PhysicalConstants_['_aut']*1.0e+18
        retv="#laserfield.kickfield:E:"+ str(self.ElectricfieldVector)
        return retv
    printout_tdfields=printout_tdfields

    def printout(self,file,t0,t1,dt=None, Nstep=None):
        print("kickfield")

    def get_vectorfield(self,tm_AU):
        if(tm_AU< self.tmStart_AU - self.hfWidth_AU ):
            return np.array([0.0, 0.0, 0.0])
        elif( tm_AU < self.tmStart_AU + self.hfWidth_AU ):
            fac=0.50*(1 + (tm_AU-self.tmStart_AU)/self.hfWidth_AU)
            return np.array( [ fac*self.Avec1[0], fac*self.Avec1[1], fac*self.Avec1[2] ] )
        else:
            return np.array( [ self.Avec1[0], self.Avec1[1], self.Avec1[2] ] )
class CWField01(laserfield):
    """
    Continuously oscillating laser field which switches on linearly.

    Parameters:
      e0   electric field strength  (in atomic units)
      w    field frequency (in atomic units)
      ts   switch on time  (in atomic units)
    """
    def __init__(self, e0, omega, ts, polarization=[0.0,0.0,1.0], toffset=None):
        if(toffset is None):
            toffset=ts
        # PhysicalConstants_=get_constants();
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ## = physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )
        assert ( abs( cLIGHT_AU - 137.035999)<2.0e-6),"check cLIGHT_AU:%f"%(cLIGHT_AU)
        self.e0 = e0
        self.omega  = omega
        self.a0 = - e0*cLIGHT_AU/self.omega
        self.ts = ts
        self.toffset = toffset  ### 20210607 we set this explicitly..
        assert abs(ts-toffset)<1e-8,"PLS make sure if this is what you really want." 
        ### print(ts);print(toffset);print([ts==0,ts==0.0,toffset==0,toffset==0.0])
        # assert ts==0.0 and toffset==0.0,"consider Trapezoidal pulse"
        #self.time = -9999.99
        self.polarization=np.array(polarization)

    printout_tdfields=printout_tdfields

    def tostring(self,delimiter='\n'):
        # PhysicalConstants_=get_constants();
        aut_in_attosec= physicalconstants.aut_in_femtosec*(1.0e+3); ##= PhysicalConstants_['_aut']*1.0e+18
        retv="#laserfield.CWField01:"
        retv+= delimiter+"#omega,Fpeak,Apeak:"+str( [self.omega,self.e0,self.a0] )
        retv+= delimiter+"#freq_eV:%f wavelength:%f"%(self.omega*physicalconstants.HARTREEinEV,omega_to_wavelength(self.omega))
        retv+= delimiter+"#ts:%f au/ %f fs"%( self.ts, self.ts*aut_in_attosec*(1.0e-3))
        retv+= delimiter+"#polarization:"+str( self.polarization )
        return retv;

    def get_vectorfield(self,tm_AU):
        if(tm_AU < self.toffset):
            ampd=0.0
        elif( tm_AU < self.ts):
            ampd = self.a0 * ((tm_AU-self.toffset)/ self.ts) * np.sin(self.omega * tm_AU)
        else:
            ampd = self.a0 * np.sin(self.omega * (tm_AU-self.toffset) )
        print("VectorField:",ampd,tm_AU>self.ts,self.a0 );## assert False,""+str(ampd)
        return np.array([ ampd*self.polarization[0], ampd*self.polarization[1], ampd*self.polarization[2] ])
    def get_electricfield(self,tm_AU):
        if(tm_AU < self.ts):
            ampd = self.e0 * (tm_AU / self.ts) * np.cos(self.omega * tm_AU)\
                    + self.e0 * (1.0/ (self.omega * self.ts) )* np.sin( self.omega * tm_AU)
        else:
            ampd = self.e0 * np.cos(self.omega * tm_AU)
        return np.array([ ampd*self.polarization[0], ampd*self.polarization[1], ampd*self.polarization[2] ])

class Trapezoidalpulse(laserfield):
    def __init__(self, freq_eV, peakIntensity_WpeCM2, 
            polarization=[0,0,1], tAsc_fs=None, tFlat_fs=None, tDsc_fs=None, tc_fs=None, tOffset_fs=0.0,
            fwhm_as=-1.0, phase=0.0, sincos=None):
        # PhysicalConstants_=get_constants();
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ## = physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )
        aut_in_fs= physicalconstants.aut_in_femtosec ## i.e. 24.1888 attosec

        self.omega= freq_eV/physicalconstants.HARTREEinEV
        self.Fpeak= intensity_to_FieldStrength(peakIntensity_WpeCM2,"au")
        self.Apeak= (cLIGHT_AU/self.omega)* self.Fpeak  ## (\omega/c )A = E
        norm = math.sqrt( polarization[0]**2 + polarization[1]**2 + polarization[2]**2 )
        assert norm>0.01 and norm<=100, "wrong polarization vector"
        if(abs(norm-1)>1.0e-8):
            self.polarization=np.array(polarization)*(1.0/norm)
        else:
            self.polarization=np.array(polarization)
        if( tc_fs is None ):
            tc_fs = tAsc_fs + tFlat_fs*0.50

        self.tc   = tc_fs / aut_in_fs
        self.tAsc = tAsc_fs / aut_in_fs
        self.tFlat= tFlat_fs / aut_in_fs 
        self.tDsc = tDsc_fs / aut_in_fs

        self.A1 = None
        self.A2 = None
        self.A3 = None

        self.t0 = tOffset_fs / aut_in_fs
        self.t1 = self.t0 + self.tAsc
        self.t2 = self.t1 + self.tFlat
        self.t3 = self.t2 + self.tDsc

        self.phi0=phase
        self.nstep=0; self.time=-999.9;
        if sincos is not None:
            if( sincos == 'sin' ):
                self.phi0=self.phi0-math.pi*0.50

    def get_electricfield(self,tm_AU):
        if( (tm_AU > self.t3) or (tm_AU < self.t0) ):
            return np.array([ 0.0, 0.0, 0.0 ])
        phase=(tm_AU-self.tc)*self.omega + self.phi0
        if( tm_AU < self.t1 ):
            ampd = ( self.Fpeak * (tm_AU - self.t0)/self.tAsc )*math.cos(phase)
            return ampd*self.polarization
        elif( tm_AU < self.t2 ):
            ampd = self.Fpeak * math.cos(phase)
            return ampd*self.polarization
        else:
            ampd = ( self.Fpeak * ( 1.0 - (tm_AU-self.t2)/self.tAsc ) )*math.cos(phase)
            return ampd*self.polarization
    
    def get_vectorfield(self,tm_AU):
        if( tm_AU < self.t0 ):
            return np.array([ 0.0, 0.0, 0.0 ])
        elif( tm_AU < self.t1 ):
            return self.get_vectorfield_(1,tm_AU)
        elif( tm_AU < self.t2 ):
            if( self.A1 is None ):
                self.A1 = self.get_vectorfield_(1,self.t1)
            return self.get_vectorfield_(2,tm_AU)
        elif( tm_AU < self.t3 ):
            if( self.A2 is None ):
                self.A2 = self.get_vectorfield_(2,self.t2)
            return self.get_vectorfield_(3,tm_AU)
        elif( tm_AU > self.t3 ):
            if( self.A3 is None ):
                self.A3 = self.get_vectorfield_(3,self.t3)
            return self.A3

    def get_vectorfield_(self,step,tm_AU):
        phase=(tm_AU-self.tc)*self.omega + self.phi0
        if( step == 1 ):
            phase_0=(self.t0-self.tc)*self.omega + self.phi0
            ampd = (-self.Apeak) * ( ((tm_AU-self.t0)/self.tAsc)*math.sin(phase) \
                                    + (1.0/(self.omega * self.tAsc))*( math.cos(phase)-math.cos(phase_0) ) )
            return ampd * self.polarization
        elif( step == 2 ):
            phase_1=(self.t1-self.tc)*self.omega + self.phi0
            ampd = (-self.Apeak) * ( math.sin(phase) - math.sin(phase_1) )
            return self.A1 + ampd*self.polarization
        elif( step == 3 ):
            phase_2=(self.t2-self.tc)*self.omega + self.phi0
            ampd = (-self.Apeak) * ( ( 1.0 - (tm_AU-self.t2)/self.tDsc )*math.sin(phase) - math.sin(phase_2) \
                                     - (1.0/(self.omega * self.tDsc))*( math.cos(phase)-math.cos(phase_2) ) )
            return self.A2 + ampd*self.polarization
        else:
            assert False,""
            return None
    def printout_tdfields(laser, t0_fs,t1_fs,dt_fs,fpath,Append=False,Threads=[0]):
        
        printout_tdfields(laser, t0_fs,t1_fs,dt_fs,fpath,Append=False,Threads=[0],h=0.02)

    def tostring(self,delimiter='\n'):
        #PhysicalConstants_=get_constants();
        aut_in_fs= physicalconstants.aut_in_femtosec; ## PhysicalConstants_['_aut']*1.0e+15
        retv="#laserfield.Trapezoidalpulse:"
        retv+= delimiter+"#omega,Fpeak,Apeak:"+str( [self.omega,self.Fpeak,self.Apeak] )
        retv+= delimiter+"#freq_eV,wavelength,Intensity:%f,%f,%7.2e"%(self.omega*physicalconstants.HARTREEinEV, 
                                                                      omega_to_wavelength(self.omega), FieldStrength_to_intensity(self.Fpeak))
        retv+= delimiter+"#tAsc-tFlat-tDsc:%5.1f-%5.1f-%5.1f tc:%5.1f"%(\
               self.tAsc*aut_in_fs, self.tFlat*aut_in_fs, self.tDsc*aut_in_fs, self.tc*aut_in_fs)
        retv+= delimiter+"#polarization:"+str( self.polarization )
        return retv;
        
# laser=gaussianpulse01
class gaussianpulse01(laserfield):
    """ Gaussian Pulse
    """
    def __init__(self, freq_eV, peakIntensity_WpeCM2, 
            polarization=[0,0,1], tc_as=0.0, sigma_as=-1.0,
            fwhm_as=-1.0, phase=0.0, sincos=None):
        assert sigma_as>0 or fwhm_as>0, "either sigma or fwhm should be >0"
        peakIntensity_SI=peakIntensity_WpeCM2*(1.0e+4)  ## 0.5* eps * (E**2) * c 
        """ 
        This function is invoked as   add_linear_field( strength(time) ...)
        take AU input and return AU output
        """
        #PhysicalConstants_=get_constants();
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ## = physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )

        self.omega= freq_eV/physicalconstants.HARTREEinEV
        self.Fpeak= intensity_to_FieldStrength(peakIntensity_WpeCM2,"au")
        self.Apeak= (cLIGHT_AU/self.omega)* self.Fpeak  ## (\omega/c )A = E

        aut_in_attosec= physicalconstants.aut_in_femtosec*(1.0e+3); ##= PhysicalConstants_['_aut']*1.0e+18 ## i.e. 24.1888 attosec
        if(sigma_as<0):
            self.sigma= (0.5*fwhm_as) * math.sqrt( 1.00/math.sqrt(math.log(2.0)) ) / aut_in_attosec
                                                   ## 2020.09.19 11.34 JST: intensity-FWHM
                                                   ## -0.5(0.5fwhm/sigma)**2 = -0.5( ln(2) ) 
        norm = math.sqrt( polarization[0]**2 + polarization[1]**2 + polarization[2]**2 )
        assert norm>0.01 and norm<=100, "wrong polarization vector"
        if(abs(norm-1)>1.0e-8):
            self.polarization=np.array(polarization)*(1.0/norm)
        else:
            self.polarization=np.array(polarization)

# A = - Apeak * polarization[:] * sin(\omega*(t-tc)) * exp(...) 
# E =   Fpeak * polarization[:] * cos(\omega*(t-tc)) * exp(...) + ...terms arising from deriv. of exp.. 
#       Fpeak:= Apeak * \omega / c

        self.tc=tc_as / aut_in_attosec
        self.phi0=phase
        self.nstep=0; self.time=-999.9;
        if sincos is not None:
            if( sincos == 'sin' ):
                self.phi0=self.phi0-math.pi*0.50
    printout_tdfields=printout_tdfields

    def tostring(self,delimiter='\n'):
        #PhysicalConstants_=get_constants();
        aut_in_attosec= physicalconstants.aut_in_femtosec*(1.0e+3); ##= PhysicalConstants_['_aut']*1.0e+18
        retv="#laserfield.gaussianpulse01:"
        retv+= delimiter+"#omega,Fpeak,Apeak:"+str( [self.omega,self.Fpeak,self.Apeak] )
        retv+= delimiter+"#freq_eV,wavelength,Intensity:%f,%f,%7.2e"%(self.omega*physicalconstants.HARTREEinEV, 
                                                                      omega_to_wavelength(self.omega), FieldStrength_to_intensity(self.Fpeak))
        retv+= delimiter+"#tc:%f au %f fs"%( self.tc, self.tc*aut_in_attosec*(1.0e-3))
        retv+= delimiter+"#polarization:"+str( self.polarization )
        return retv;
    def get_vectorfield(self,tm_AU):
        arg=(tm_AU-self.tc)/self.sigma
        phase=(tm_AU-self.tc)*self.omega + self.phi0  ## cosine wave means max ampd at t=tc. 
        ampd= - self.Apeak*math.sin(phase)*math.exp(0.0-0.5*arg*arg)
        return np.array([ ampd*self.polarization[0], ampd*self.polarization[1], ampd*self.polarization[2] ])
    def get_electricfield(self,tm_AU):
        #PhysicalConstants_=get_constants();
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ## = physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )
        arg=(tm_AU-self.tc)/self.sigma
        phase=(tm_AU-self.tc)*self.omega + self.phi0
        ampd = self.Apeak * ( (self.omega/cLIGHT_AU)*math.cos(phase) \
                              - ( (tm_AU-self.tc)/(cLIGHT_AU* (self.sigma**2)) )*math.sin(phase)) * math.exp(-0.5*arg*arg) 
        return np.array([ ampd*self.polarization[0], ampd*self.polarization[1], ampd*self.polarization[2] ])

class sinsquarepulse(laserfield):
    def __init__(self, freq_eV, peakIntensity_WpeCM2, width_as, 
            polarization=[0,0,1], tc_as=0.0, phase=0.0, sincos=None):
        assert width_as>0, "width_as should be >0"
        """ 
        This function is invoked as   add_linear_field( strength(time) ...)
        take AU input and return AU output
        """
        #PhysicalConstants_=get_constants();
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ## = physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )
        aut_in_attosec= physicalconstants.aut_in_femtosec*(1.0e+3); ##= PhysicalConstants_['_aut']*1.0e+18 ## i.e. 24.1888 attosec

        PI=3.1415926535897932384626433832795
        self.omega= freq_eV/physicalconstants.HARTREEinEV
        self.Fpeak= intensity_to_FieldStrength(peakIntensity_WpeCM2,"au")
        self.Apeak= (cLIGHT_AU/self.omega)* self.Fpeak  ## (\omega/c )A = E
        wid_AU = width_as/aut_in_attosec
        self.kappa= PI/wid_AU
        self.width_AU=wid_AU
        norm = math.sqrt( polarization[0]**2 + polarization[1]**2 + polarization[2]**2 )
        assert norm>0.01 and norm<=100, "wrong polarization vector"
        if(abs(norm-1)>1.0e-8):
            self.polarization=np.array(polarization)*(1.0/norm)
        else:
            self.polarization=np.array(polarization)

# A = - Apeak * polarization[:] * sin(\omega*(t-tc)) * exp(...) 
# E =   Fpeak * polarization[:] * cos(\omega*(t-tc)) * exp(...) + ...terms arising from deriv. of exp.. 
#       Fpeak:= Apeak * \omega / c

        self.tc=tc_as / aut_in_attosec

        self.toff=self.tc-self.width_AU*0.50

        self.phi0=phase
        self.nstep=0; self.time=-999.9;
        if sincos is not None:
            if( sincos == 'sin' ):
                self.phi0=self.phi0-math.pi*0.50
    printout_tdfields=printout_tdfields
    def get_vectorfield(self,tm_AU):
        PI=3.1415926535897932384626433832795
        arg= self.kappa*(tm_AU-self.toff)
        if( arg < 0 or arg>PI ):
            return np.zeros([3],dtype=np.float64); 
        phase=(tm_AU-self.tc)*self.omega + self.phi0  ## cosine wave means max ampd at t=tc. 
        ampd= - self.Apeak*math.sin(phase)*( math.sin(arg)**2 )
        return np.array([ ampd*self.polarization[0], ampd*self.polarization[1], ampd*self.polarization[2] ])
    def get_electricfield(self,tm_AU):
        #PhysicalConstants_=get_constants();
        cLIGHT_AU = physicalconstants.CLIGHTinAU; ## = physicalconstants.CLIGHTinMPS * ((physicalconstants.aut_in_femtosec)*(1e-15)) / ( PhysicalConstants_['Bohr']*(1.0e-10) )
        arg= self.kappa*(tm_AU-self.toff)   ### ampd arg
        PI=3.1415926535897932384626433832795
        if( arg < 0 or arg>PI ):
            return np.zeros([3],dtype=np.float64); 
        phase=(tm_AU-self.tc)*self.omega + self.phi0  ## cosine wave means max ampd at t=tc.
        si=math.sin(arg);co=math.cos(arg);
        ampd= self.Apeak *( (self.omega/cLIGHT_AU)*math.cos(phase)*(si**2)  \
                            + (2.0*self.kappa/cLIGHT_AU) * si*co * math.sin(phase) )
        return np.array([ ampd*self.polarization[0], ampd*self.polarization[1], ampd*self.polarization[2] ])
    def tostring(self,delimiter='\n'):
        #PhysicalConstants_=get_constants();
        aut_in_attosec= physicalconstants.aut_in_femtosec*(1.0e+3); ##= PhysicalConstants_['_aut']*1.0e+18
        retv="#laserfield.sinsquarepulse:"
        retv+= delimiter+"#omega,Fpeak,Apeak:"+str( [self.omega,self.Fpeak,self.Apeak] )
        retv+= delimiter+"#freq_eV,wavelength,Intensity:%f,%f,%7.2e"%(self.omega*physicalconstants.HARTREEinEV, 
                                            omega_to_wavelength(self.omega), FieldStrength_to_intensity(self.Fpeak))
        retv+= delimiter+"#tc:%f au %f fs"%( self.tc, self.tc*aut_in_attosec*(1.0e-3))
        retv+= delimiter+"#polarization:"+str( self.polarization )
        return retv;
