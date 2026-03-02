#ifndef _INCLUDE_cConstants_H
#include "macro_util.h"
#include "FortInt.h"
#define _INCLUDE_cConstants_H

#define __SQRT2 1.4142135623730950488016887242097
#define AMUinAU 1822.8884855409495559369254593646
#define Planck_eVfs 4.1356673336325151050603328445925
#define HARTREEinEV 27.21138386
#define PI         3.141592653589793238
#define __fourPI   12.566370614359172952
#define EXP_ONE    2.718281828459045235
#define BOHRinANGS 0.52917720859
#define AUinFS     0.024188843262
// hbar/HARTREE = 4.1356673336325151050603328445925/dPI/HARTREEinEV
#define byte char
#define Dlnk double
#define Ilnk int
static char __AtomicSymbols[118][3]={
	"H", "He",
	"Li","Be",                                                   "B", "C", "N", "O", "F","Ne",
	"Na","Mg",                                                  "Al","Si", "P", "S","Cl","Ar",
	 "K","Ca","Sc","Ti", "V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
	"Rb","Sr", "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pb","Ag","Cd","In","Sn","Sb","Te", "I","Xe",
	"Cs","Ba",  //56
	"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",  "Ho","Er","Tm","Yb","Lu",
	"Hf","Ta","W","Re","Os",  "Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
	"","","","","","","","","","",  "","","","","","","","","","",  "",""};

static const double __AmassInAMU[118]={
  1.00790,  4.00260,  6.94000, 9.01218,
 10.81000, 12.01100, 14.00670, 15.99940,
 18.99840, 20.17900, 22.98977, 24.30500,
 26.98154, 28.08550, 30.97376, 32.06000,
 35.45300, 39.94800, 39.09830, 40.08000,
 44.95590, 47.90000, 50.94150, 51.99600,
 54.93800, 55.84700, 58.93320, 58.71000,
 63.54600, 65.38000, 69.73500, 72.59000,
 74.92160, 78.96000, 79.90400, 83.80000,
 85.46780, 87.62000, 88.90590, 91.22000,
 92.90640, 95.94000, 98.90620, 101.0700,
102.9055, 106.4000, 107.8680, 112.4100,
114.8200, 118.6900, 121.7500, 127.6000,
126.9045, 131.3000,                               // this is Xe(Z=54)
132.9054, 137.3300,
-1,-1,-1,-1,-1,  -1,-1,-1,-1,-1,   -1,-1,-1,-1,-1,        //  57-71
178.4900,
180.9479,
183.8500, 186.2070, 190.2000, 192.2200,
195.0900,
196.9665, 200.5900, 204.3700, 207.2000,
208.9804,                                              // this is Bi(Z=83)
-1,-1,-1,-1,-1,  -1,-1,-1,-1,-1,   -1,-1,-1,-1,-1,
-1,-1,-1,-1,-1,  -1,-1,-1,-1,-1,   -1,-1,-1,-1,-1,
-1,-1,-1,-1,-1};

#define __AtSymbolToZnuc(X,nz)  { for(nz=0;nz<118;nz++){ if(strcmp(X,__AtomicSymbols[nz])==0)break;} if(nz>=118)nz=-1;else nz++;}
#define __ZnucToAmass(nz)    ( __AmassInAMU[((nz)-1)] )
#define __AtSymbolToAmass(X,val)  { int iznuctmp;for(iznuctmp=0;iznuctmp<118;iznuctmp++){ if(strcmp(X,__AtomicSymbols[iznuctmp])==0)break;}\
if(iznuctmp>=118){ __assertf(0,("unknown alabel:%s\n",X),-1);} val=__AmassInAMU[(iznuctmp)]; }

#endif
