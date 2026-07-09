# Plans of DH functionals to be tested

## Gaussian (16 Rev B.01)

Already or to be implemented

- [x] B2PLYP
- [x] mPW2PLYP
- [x] B2PLYP-D3
- [x] DSD-PBEP86 (with D3BJ)
- [x] PBE0-DH
- [x] PBE-QIDH

Will not be implemented

- B2PLYPD (D2 dispersion)
- mPW2PLYPD (D2 dispersion)

## Q-Chem (6.0)

- [x] DSD-PBEPBE-D3
- [x] wB97X-2(LP)
- [x] wB97X-2(TQZ)
- [x] XYG3
- [x] XYGJ-OS
- [x] B2PLYP
- [x] B2GPPLYP
- [x] DSD-PBEP86-D3
- [x] LS1DH-PBE
- [x] PBE-QIDH
- [x] PBE0-2
- [x] PBE0-DH
- [ ] wB97M(2) *VV10*
- [x] PTPSS-D3
- [x] DSD-PBEB95-D3
- [x] PWPB95-D3

## MRCC (2022-03-18)

- [x] B2PLYP
- [x] B2GPPLYP
- [ ] DSD-PBEhB95-D3
- [x] XYG3
- [ ] SCAN0-2 *numerical discrepency*
- [x] dRPA75 *not tested*
- [ ] SCS-dRPA75
- [x] RS-PBE-PBE
- [x] RS-PBE-P86
- [x] RS-B88-LYP
- [x] RS-PW91-PW91

Will not be implemented

- [ ] DSD-PBEP86-D3 (source of coefficients may be different)

## ORCA (5.0.2)

Note that most functionals in ORCA is not (and may not be) tested.
It can be hard work to ensure results of PySCF and ORCA agree within 5e-6 Hartree for water molecule.

Some XC (especially PBE-involving) is not the same among ORCA, LibXC and other software.
A possible way to make sure problem originates from functional implementation:

    ORCA keyword -> ORCA 

- [x] B2PLYP
- [x] mPW2PLYP
- [x] B2GP-PLYP
- [ ] B2KP-PLYP
- [ ] B2TP-PLYP
- [x] PWPB95 *LibXC*
- [x] PBE-QIDH
- [x] PBE0-DH
- [x] DSD-BLYP-D3
- [x] DSD-PBEP86-D3
- [x] DSD-PBEP95-D3
- [x] wB2PLYP
- [x] wB2GP-PLYP
- [x] RSX-QIDH
- [x] RSX-0DH
- [x] wB88PP86
- [x] wPBEPP86
- [x] wB97X-2
- [ ] SCS/SOS-B2PLYP21
- [ ] SCS/SOS-PBE-QIDH
- [ ] SCS/SOS-B2GP-PLYP21
- [ ] SCS/SOS-wB2PLYP
- [ ] SCS/SOS-wB2GP-PLYP
- [ ] SCS/SOS-RSX-QIDH
- [ ] SCS/SOS-wB88PP86
- [ ] SCS/SOS-wPBEPP86

## ADF 2023.1

Not ADF user currently.

- [x] B2PLYP
- [ ] B2PIPLYP
- [ ] B2TPLYP
- [x] B2GPPLYP
- [ ] B2KPLYP
- [ ] B2NCPLYP
- [x] mPW2PLYP
- [ ] mPW2KPLYP
- [ ] mPW2NCPLYP
- [ ] DH-BLYP
- [x] PBE0-DH
- [x] PBE-QIDH
- [x] LS1-DH
- [x] PBE0-2
- [ ] LS1-TPSS
- [ ] DS1-TPSS
- [ ] various DSD/DOD functionals
- [ ] SD-SCAN69

## Other Known Software

- Psi4 (4.0b5): all functionals are included in Gaussian and Q-Chem.
- TurboMole (7.7.1): B2PLYP realized (as functional that has keyword).
- Molpro (2022.3): B2PLYP realized (as functional that has keyword).
- NWChem (6.5): B2PLYP realized (no specific keyword).
- 
