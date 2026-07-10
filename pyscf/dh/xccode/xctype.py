"""
Type of exchange-correlation

Notes
-----
Parameters of exchange-correlation are also defined in this module.

Program functionality of parameter handling lies in ``XCInfo``, however, import definition exists here.

Except for special cases, parameter table should be defined as follows. For example of RSMP2,

- parameter of this type is ([explanation, addability])
    - ["range-separate omega", False],
    - ["oppo-spin coefficient", True],
    - ["same-spin coefficient", True],
- parameter list in program is something like (0.7, 1.3, 0.6);
- input parameter (0.7), then set parameter list (0.7, 1, 1),
  i.e., addable parameter filled by one if addable;
- canonical representation of ``0.6*RSMP2(0.7, 1.3, 0.6)`` should be ``RSMP2(0.7, 0.78, 0.36)``,
  i.e., if there is addable parameters, then coefficient factor may be multiplied into these parameters.
"""

import enum
from enum import Flag


class XCType(Flag):
    """
    Exchange-correlation types.
    """
    # region exact exchange
    HF = enum.auto()
    "Hartree-Fock"
    RSH = enum.auto()
    "Range-separate"
    EXX = HF | RSH
    "Exact exchange that requires ERI evaluation"
    # endregion

    # region low rung category
    LDA = enum.auto()
    "Local density approximation"
    GGA = enum.auto()
    "Generalized gradient approximation"
    MGGA = enum.auto()
    "meta generalized gradient approximation"
    RUNG_LOW = EXX | LDA | GGA | MGGA
    "Low-rung (1st - 4th) approximation"
    # endregion

    # region high rung
    MP2 = enum.auto()
    "MP2 correlation contribution"
    RSMP2 = enum.auto()
    "Range-separate MP2 correlation contribution"
    IEPA = enum.auto()
    "IEPA-like correlation contribution"

    RS_RING_CCD = enum.auto()
    "Range-separate ring-CCD correlation contribution"
    RPA = RS_RING_CCD
    "RPA-like correlation contribution"
    RUNG_HIGH = MP2 | RSMP2 | IEPA | RPA
    "High-rung (5th) approximation"
    # endregion

    # region nuc_corr
    DFTD3 = enum.auto()
    "DFT-D3 vDW dispersion correction"
    DFTD4 = enum.auto()
    "DFT-D4 vDW dispersion correction"
    NUC_CORR = DFTD3 | DFTD4
    "Correction schemes that only depends on neucleu coordinates"

    # region vDW
    VV10 = enum.auto()
    "Vydrov and van Voorhis 2010"
    VDW = VV10
    "Van der Waals contribution"
    # endregion

    # region type of exch, corr or hyb
    EXCH = enum.auto()
    "Exchange contribution"
    CORR = enum.auto()
    "Correlation contribution"
    HYB = enum.auto()
    "Hybrid DFA that includes multiple contributions of correlation and EXX contribution"
    # endregion

    # region special configurations
    SSR = enum.auto()
    "Scaled short-range"
    PYSCF_PARSABLE = enum.auto()
    "Functional contribution that is able to be parsed by pyscf.dft.numint"
    WITH_EXT_PARAM = enum.auto()
    "Functional with external parameters for libxc"
    # endregion

    UNKNOWN = 0

    def check_sanity(self):
        """ Check contradictory flags or must-exist flags. """

        # check contradictory
        def contradictory(*args):
            exist = [self & arg != self.UNKNOWN for arg in args]
            assert sum(exist) <= 1

        # decompose flag
        def decompose(flag):
            bits = bin(flag.value)[2:][::-1]
            return [self.__class__(2**n) for n, v in enumerate(bits) if v == "1"]

        contradictory(*decompose(self.EXX))
        contradictory(*decompose(self.RUNG_LOW))
        contradictory(*decompose(self.RUNG_HIGH))
        contradictory(*decompose(self.NUC_CORR))
        contradictory(*decompose(self.VDW))
        contradictory(self.CORR, self.EXCH, self.HYB)
        contradictory(self.RUNG_HIGH, self.RUNG_LOW, self.NUC_CORR, self.VDW)

        # check special rules
        # SSR
        if self.SSR in self:
            assert self.EXCH in self or self.CORR in self
            assert self & self.RUNG_LOW

        # additional check for parameters list
        contradictory(*self.def_parameters().keys())

        # must exist flags
        assert (self.EXCH | self.CORR | self.HYB) & self
        assert (self.RUNG_HIGH | self.RUNG_LOW | self.NUC_CORR | self.VDW) & self

    @classmethod
    def def_parameters(cls) -> dict:
        """ Definition dictionary of addability of parameter list for XCInfo.

        If some xc type is not listed, then this type should not have a parameter.
        """
        dct = {
            cls.RSH: [
                ["range-separate omega", False],
            ],
            cls.MP2: [
                ["oppo-spin coefficient", True],
                ["same-spin coefficient", True],
            ],
            cls.RSMP2: [
                ["range-separate omega", False],
                ["oppo-spin coefficient", True],
                ["same-spin coefficient", True],
            ],
            cls.IEPA: [
                ["oppo-spin coefficient", True],
                ["same-spin coefficient", True],
            ],
            cls.RS_RING_CCD: [
                ["range-separate omega", False],
                ["oppo-spin coefficient", True],
                ["same-spin coefficient", True],
            ],
            cls.VV10: [
                ["parameter that controls damping of R6", False],
                ["parameter for C6 coefficient", False],
            ],
            cls.SSR: [
                ["functional to be scaled by range-separate coefficient", False],
                ["range-separate omega", False],
            ],
            cls.DFTD3: [
                ["Damping type", False],
            ],
            cls.DFTD4: [],
        }
        return dct

    def addable_parameters(self) -> list:
        """ List of addability of parameters """
        for key in self.def_parameters():
            if key in self:
                return [lst[1] for lst in self.def_parameters()[key]]
        return []
