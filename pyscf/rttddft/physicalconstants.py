class PhysicalConstants:
    def __setattr__(self,name,value):
        if( name in self.__dict__ ):
            assert False,"do not change const:"+name
        self.__dict__[name]=value
    def __init__(self):
        print("__init__ PhysicalConstants",flush=True)
        self.CLIGHTinMPS=299792458.0; self.AUTinFEMTOSEC=2.418884326058678e-2;
        self.BOHRinANGS=0.5291772105638411;self.HARTREEinEV=27.211386024367243
        self.PLANCKinJOULSEC=6.62607004e-34;
        self.ECHARGEinCOULOMB=1.6021766208e-19;
        self.EMASSinKG=9.10938356e-31
        self.EPS0inSI=8.85418781762039e-12
        self.aut_in_femtosec=self.AUTinFEMTOSEC
        self.CLIGHTinAU=self.CLIGHTinMPS * (self.AUTinFEMTOSEC*1e-15) \
                        / ( self.BOHRinANGS*(1.0e-10) )
        self.LAMBDA1EVinNM=( self.PLANCKinJOULSEC * self.CLIGHTinMPS/self.ECHARGEinCOULOMB )*(1e+9);


# o to add more constants, you set them in __init__ method
# o you can access to immutable constants as shown below...
# print(PhysicalConstants.Getv().BOHRinANGS)
# print(PhysicalConstants.Getv("AUTinFEMTOSEC"))
# print(PhysicalConstants.Getv("AUTinFEMTOSEC","BOHRinANGS"))
# print(PhysicalConstants.CLIGHTinAU())
#
    Instance_=None
    @staticmethod
    def Getv(*args):
        if( PhysicalConstants.Instance_ is None ):
            PhysicalConstants.Instance_=PhysicalConstants()
        le=len(args)
        if(le==0):
            return PhysicalConstants.Instance_;
        ret=[]
        for name in args:
            dum=getattr( PhysicalConstants.Instance_, name)
            if(le==1):
                return dum
            ret.append(dum)
        return ret
    @staticmethod
    def CLIGHTinAU():
        if( PhysicalConstants.Instance_ is None ):
            PhysicalConstants.Instance_=PhysicalConstants()
        cLIGHT_AU = PhysicalConstants.Instance_.CLIGHTinMPS \
                    * (PhysicalConstants.Instance_.AUTinFEMTOSEC*1e-15) \
                      / ( PhysicalConstants.Instance_.BOHRinANGS*(1.0e-10))
        return cLIGHT_AU

    @staticmethod
    def BOHRinANGS():
        if( PhysicalConstants.Instance_ is None ):
            PhysicalConstants.Instance_=PhysicalConstants()
        val=PhysicalConstants.Instance_.BOHRinANGS
        return val

    @staticmethod
    def aut_in_femtosec():
        if( PhysicalConstants.Instance_ is None ):
            PhysicalConstants.Instance_=PhysicalConstants()
        val=PhysicalConstants.Instance_.AUTinFEMTOSEC
        return val

    @staticmethod
    def HARTREEinEV():
        if( PhysicalConstants.Instance_ is None ):
            PhysicalConstants.Instance_=PhysicalConstants()
        val=PhysicalConstants.Instance_.HARTREEinEV
        return val


physicalconstants=PhysicalConstants.Getv()


## def get_constants():
##     print("\n\n#!W get_constants:this subroutine is to be removed in the near future\n\n")
##     dic={"_c":299792458.0, "_mu0":1.2566370614359173e-06, "_Grav":6.67408e-11, "_hplanck":6.62607004e-34,\
##     "_e":1.6021766208e-19, "_me":9.10938356e-31, "_mp":1.672621898e-27, "_Nav":6.022140857e+23,\
##     "_k":1.38064852e-23, "_amu":1.66053904e-27, "_eps0":8.85418781762039e-12, "_hbar":1.0545718001391127e-34,\
##     "Ang":1.0, "Angstrom":1.0, "nm":10.0, "Bohr":0.5291772105638411,\
##     "eV":1.0, "Hartree":27.211386024367243, "kJ":6.241509125883258e+21, "kcal":2.611447418269555e+22,\
##     "mol":6.022140857e+23, "Rydberg":13.605693012183622, "Ry":13.605693012183622, "Ha":27.211386024367243,\
##     "second":98226947884640.62, "fs":0.09822694788464063, "kB":8.617330337217213e-05, "Pascal":6.241509125883258e-12,\
##     "GPa":0.006241509125883258, "Debye":0.20819433442462576, "alpha":0.007297352566206496, "invcm":0.0001239841973964072,\
##     "_aut":2.418884326058678e-17, "_auv":2187691.262715653, "_auf":8.238723368557715e-08, "_aup":29421015271080.86,\
##     "AUT":0.0023759962463473982, "m":10000000000.0, "kg":6.0221408585491615e+26, "s":98226947884640.62,\
##     "A":63541.719052630964, "J":6.241509125883258e+18, "C":6.241509125883258e+18};
##     return dic

if(__name__ == "__main__"):
    dic={"_c":299792458.0, "_mu0":1.2566370614359173e-06, "_Grav":6.67408e-11, "_hplanck":6.62607004e-34,\
    "_e":1.6021766208e-19, "_me":9.10938356e-31, "_mp":1.672621898e-27, "_Nav":6.022140857e+23,\
    "_k":1.38064852e-23, "_amu":1.66053904e-27, "_eps0":8.85418781762039e-12, "_hbar":1.0545718001391127e-34,\
    "Ang":1.0, "Angstrom":1.0, "nm":10.0, "Bohr":0.5291772105638411,\
    "eV":1.0, "Hartree":27.211386024367243, "kJ":6.241509125883258e+21, "kcal":2.611447418269555e+22,\
    "mol":6.022140857e+23, "Rydberg":13.605693012183622, "Ry":13.605693012183622, "Ha":27.211386024367243,\
    "second":98226947884640.62, "fs":0.09822694788464063, "kB":8.617330337217213e-05, "Pascal":6.241509125883258e-12,\
    "GPa":0.006241509125883258, "Debye":0.20819433442462576, "alpha":0.007297352566206496, "invcm":0.0001239841973964072,\
    "_aut":2.418884326058678e-17, "_auv":2187691.262715653, "_auf":8.238723368557715e-08, "_aup":29421015271080.86,\
    "AUT":0.0023759962463473982, "m":10000000000.0, "kg":6.0221408585491615e+26, "s":98226947884640.62,\
    "A":63541.719052630964, "J":6.241509125883258e+18, "C":6.241509125883258e+18};
    
    aut_in_femtosec= dic['_aut']*1.0e+15
    BOHRinANGS=dic['Bohr']
    HARTREEinEV=dic['Hartree']

    dum=PhysicalConstants.HARTREEinEV(); diff=abs(dum-HARTREEinEV); assert diff<1e-7,""; print(dum,diff)
    dum=PhysicalConstants.BOHRinANGS();  diff=abs(dum-BOHRinANGS); assert diff<1e-7,""; print(dum,diff)
    dum=PhysicalConstants.aut_in_femtosec(); diff=abs(dum-aut_in_femtosec); assert diff<1e-7,""; print(dum,diff)

    

    dum=physicalconstants.HARTREEinEV; diff=abs(dum-HARTREEinEV); assert diff<1e-7,""; print(dum,diff)
    dum=physicalconstants.BOHRinANGS;  diff=abs(dum-BOHRinANGS); assert diff<1e-7,""; print(dum,diff)
    dum=physicalconstants.aut_in_femtosec; diff=abs(dum-aut_in_femtosec); assert diff<1e-7,""; print(dum,diff)
