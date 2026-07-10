"""
Information of one exchange-correlation term

Notes
-----
For regex of xc token, one may try: https://regex101.com/r/xnD5jM/1

Groups of regex search

0. match all token for an entire xc code for one term
1. match if comma in string; if exists, then split exchange and correlation parts (pyscf convention);
2. sign of factor
3. factor (absolute value) with asterisk
4. name of xc term (dash not allowed)
5. list of parameters with parentheses (words allowed, splited by comma or semicolon)
"""
import copy
from dataclasses import dataclass
import re
from typing import List
from .xctype import XCType
from .xcjson import ADV_CORR_ALIAS, ADV_CORR_DICT
from pyscf import dft


REGEX_XC = r"((,?)([+-]*)([0-9.]+\*)?([\w@]+)(\([\w+\-\*.,;=]+\))?)"


@dataclass
class XCInfo:
    """
    Exchange-correlation information.

    xc info refers to one component of exchange or correlation contribution.
    """

    fac: float
    """ Factor of xc contribution. """
    name: str
    """ Name of xc contribution. """
    parameters: list
    """ Parameters list of xc contribution. """
    type: XCType
    """ Type of xc contribution. """
    additional: dict
    """ (experimental) Additional parameters.

    Parameters that is not sutiable to be passed as string token, or very advanced parameters.
    """

    def __init__(self, fac, name, parameters, typ, additional=None):
        self.fac = fac
        self.name = name
        self.parameters = parameters
        self.type = typ
        if additional is None:
            self.additional = dict()
        else:
            assert isinstance(additional, dict)
            self.additional = additional
        self.round()

    def round(self, ndigits=10) -> "XCInfo":
        """ Round floats for XC factor and parameters. """
        def adv_round(f, n):
            if f == round(f):
                return round(f)
            return round(f, n)

        self.fac = adv_round(self.fac, ndigits)
        self.parameters = [
            adv_round(f, ndigits) if isinstance(f, (float, int)) else str(f)
            for f in self.parameters]
        self.additional = {
            k: adv_round(f, ndigits) if isinstance(f, (float, int)) else str(f)
            for k, f in self.additional.items()}
        return self

    @property
    def token(self) -> str:
        """ Standardlized name of XC contribution.

        Note that additional parameter is not considered.
        """
        self.round()
        token = ""
        # 1. factor
        if self.fac < 0:
            token += "- "
        if abs(self.fac) != 1:
            token += str(abs(self.fac)) + "*"
        # 2. name
        token += self.name
        # 3. parameter and additional
        param_tokens = [str(f) for f in self.parameters]
        param_tokens += [f"{key}={val}" for key, val in self.additional.items()]
        if len(param_tokens) != 0:
            token += "(" + ", ".join(param_tokens) + ")"
        # finalize: upper_case
        token = token.upper()
        return token

    @classmethod
    def parse_xc_info(cls, inp: str, guess_type: XCType = XCType.UNKNOWN) -> "XCInfo" or List["XCInfo"]:
        """ Parse xc info from regex groups.

        See Also
        --------
        parse_token
        """
        # parse input by regex
        inp = inp.strip().replace(" ", "").upper()
        match = re.findall(REGEX_XC, inp)
        if len(match) != 1:
            raise ValueError("Read from info failed in that prehaps multiple info is required.")
        if inp != match[0][0]:
            raise ValueError(
                "XC token {:} is not successfully parsed.\n"
                "Regex match of this token becomes {:}".format(inp, match[0][0]))
        inp = match[0]
        _, comma, sgn, fac, name, parameters_str = inp

        # additional check for guess type of correlation
        if comma == ",":
            guess_type = XCType.CORR
        if guess_type == XCType.UNKNOWN:
            guess_type = XCType.HYB
        assert guess_type in [XCType.HYB, XCType.EXCH, XCType.CORR]

        # parse fac
        sgn = (-1)**sgn.count("-")
        if len(fac) > 0:
            fac = sgn * float(fac[:-1])
        else:
            fac = sgn

        # parse parameters
        def try_convert_float(s):
            try:
                return float(s)
            except ValueError:
                return s

        parameters = []
        additional = {}
        if len(parameters_str) > 0:
            for item in re.split(r"[,;]", parameters_str[1:-1]):
                if "=" not in item:
                    parameters.append(try_convert_float(item))
                else:
                    assert item.count("=") == 1
                    key, val = item.split("=")
                    additional[key] = try_convert_float(val)

        # build basic information
        xc_info = XCInfo(fac, name, parameters, XCType.UNKNOWN, additional=additional)
        # for advanced correlations, try to substitute alias first
        if xc_info.name in ADV_CORR_ALIAS:
            xc_info.name = ADV_CORR_ALIAS[xc_info.name]
            return cls.parse_xc_info(xc_info.token, guess_type)

        # parse xc type
        xc_type = cls.parse_xc_type(xc_info, guess_type)
        xc_info.type = xc_type

        # handle special cases
        # SR_HF and RSH
        handled_rsh = cls.handle_rsh(xc_info)
        if handled_rsh is not None:
            return handled_rsh
        # SR_MP2
        handled_sr_mp2 = cls.handle_sr_mp2(xc_info)
        if handled_sr_mp2 is not None:
            return handled_sr_mp2

        # fill default parameters
        xc_info = cls.parse_default_parameters(xc_info)

        # finalize
        xc_info.check_sanity()
        xc_info = xc_info.try_move_fac()
        xc_info.round()
        return xc_info

    @classmethod
    def parse_xc_type(cls, xc_info: "XCInfo", guess_type: XCType) -> XCType:
        """ Try to parse xc type from name. """
        assert guess_type in [XCType.HYB, XCType.CORR, XCType.EXCH]
        name = xc_info.name
        guess_name = name
        xc_type = XCType.UNKNOWN
        ni = dft.numint.NumInt()

        # detect simple cases of HF and RSH (RSH may have parameters, which may complicates)
        if name == "HF":
            return XCType.HF | XCType.EXCH | XCType.PYSCF_PARSABLE
        if name in ["LR_HF", "SR_HF", "RSH"]:
            return XCType.RSH | XCType.EXCH | XCType.PYSCF_PARSABLE

        # detect usual cases
        if guess_type & XCType.CORR:
            guess_name = "," + name
        try:
            # try if pyscf parse_xc success (must be low_rung)
            dft_type = ni._xc_type(guess_name)
            if len(xc_info.parameters) != 0:
                raise ValueError(f"Currently do not accept parameters (list of floats) for {dft_type}. "
                                 f"Please try to use additional (dictionary of floats) to specify ext_params of libxc.")
            if len(xc_info.additional) == 0:
                xc_type |= XCType.PYSCF_PARSABLE
            else:
                # if additional parameters included, then these should be handled by ext_param in libxc
                xc_type |= XCType.WITH_EXT_PARAM
            # except special cases
            if name == "VV10" and len(xc_info.parameters) > 0:
                raise ValueError("We use VV10 as vDW correction with parameters, instead of XC_GGA_XC_VV10.")
            # parse dft type
            if guess_type != XCType.HYB:
                xc_type |= guess_type
            if dft_type == "NLC":
                raise ValueError("NLC code (with __VV10 by PySCF) is not accepted currently!")
            assert dft_type != "HF"
            DFT_TYPE_MAP = {
                "LDA": XCType.LDA,
                "GGA": XCType.GGA,
                "MGGA": XCType.MGGA,
            }
            xc_type |= DFT_TYPE_MAP[dft_type]
            # parse hf type
            rsh_and_hyb_coeff = list(ni.rsh_and_hybrid_coeff(guess_name))
            if rsh_and_hyb_coeff[1:3] != [0, 0]:
                xc_type |= XCType.HYB
            # handle type
            # if not HYB | EXCH | CORR; then assign as HYB
            if not ((XCType.HYB | XCType.EXCH | XCType.CORR) & xc_type):
                xc_type |= XCType.HYB
            # if HYB, then CORR or EXCH will not exist
            if XCType.HYB in xc_type:
                xc_type &= ~(XCType.CORR | XCType.EXCH)
        except KeyError:
            # Key that is not parsible by pyscf must lies in high-rung or vdw contributions
            # SSR (scaled short-range)
            if name == "SSR":
                if guess_type == XCType.HYB:
                    raise ValueError(
                        "Function by SSR should be explicitly defined as exchange or correlation functional, "
                        "i.e., split exch-corr by comma.")
                xc_type |= guess_type | XCType.SSR
                # further parse token in parameter
                inner_info = cls.parse_xc_info(xc_info.parameters[0], guess_type=guess_type)
                assert inner_info.type & XCType.RUNG_LOW and not inner_info.type & XCType.EXX
                xc_type |= inner_info.type & XCType.RUNG_LOW
                return xc_type
            # advanced correlation and vdw
            if not (guess_type & XCType.CORR):
                raise KeyError(
                    "Advanced component {:} is not low-rung exchange contribution.\n"
                    "Please consider add a comma to separate exch and corr.".format(name))
            type_map = {
                "MP2": XCType.MP2,
                "RSMP2": XCType.RSMP2,
                "IEPA": XCType.IEPA,
                "RS_RING_CCD": XCType.RS_RING_CCD,
                "VDW": XCType.VDW,
                "DFTD3": XCType.DFTD3,
                "DFTD4": XCType.DFTD4,
            }
            if name in ADV_CORR_DICT:
                xc_type |= XCType.CORR | type_map[ADV_CORR_DICT[name]["type"]]
            else:
                raise KeyError("Unknown advanced C component {:}.".format(name))
        return xc_type

    @classmethod
    def parse_default_parameters(cls, xc_info: "XCInfo") -> "XCInfo":
        """ Fill addable parameters (or setting default parameters for future possible API usage). """
        # not listed in definition of parameters
        lst_addable = xc_info.type.addable_parameters()
        # listed in definition of parameters
        if len(xc_info.parameters) == len(lst_addable):
            # additional check that if addable parameters is number
            for n, addable in enumerate(lst_addable):
                if addable and not convertable_to_float(xc_info.parameters[n]):
                    raise ValueError("Some parameters that should be addable is not float number!")
            return xc_info
        elif len(xc_info.parameters) == len(lst_addable) - sum(lst_addable):
            # fill in addable parameters
            n1 = 0
            parameter_new = []
            for n, addable in enumerate(lst_addable):
                if not addable:
                    parameter_new.append(xc_info.parameters[n1])
                    n1 += 1
                else:
                    parameter_new.append(1)
            xc_info.parameters = parameter_new
            return xc_info
        else:
            raise ValueError("Number of parameter number is probably not correct for term {:}!".format(xc_info.token))

    @classmethod
    def handle_rsh(cls, info: "XCInfo") -> None or List["XCInfo"]:
        """ Parse RSH parameters """
        info.round()
        if info.name == "SR_HF" or (info.name == "LR_HF" and info.parameters[0] < 0):
            assert len(info.parameters) == 1
            token_hf = "{:}*HF".format(str(info.fac))
            token_lr_hf = "{:}*LR_HF({:})".format(str(- info.fac), str(info.parameters[0]))
            return [
                cls.parse_xc_info(token_hf),
                cls.parse_xc_info(token_lr_hf)]
        if info.name == "RSH":
            assert len(info.parameters) == 3
            omega, alpha, beta = info.parameters
            token_hf = "{:}*HF".format(str(info.fac * (alpha + beta)))
            token_lr_hf = "{:}*LR_HF({:})".format(str(- info.fac * beta), str(omega))
            return [
                cls.parse_xc_info(token_hf),
                cls.parse_xc_info(token_lr_hf)]
        return None

    @classmethod
    def handle_sr_mp2(cls, info: "XCInfo") -> None or "XCInfo":
        """ Change SR_MP2 to RS_MP2.

        By PySCF's notation, short-range is realized by setting omega to be smaller than zero.
        """
        info.round()
        if info.name == "SR_MP2":
            info = info.copy()
            info.name = "RS_MP2"
            info.parameters[0] *= -1
            return cls.parse_xc_info(info.token, XCType.CORR)
        return None

    def copy(self) -> "XCInfo":
        return copy.deepcopy(self)

    def check_sanity(self):
        """ Check sanity for parameter and flags. """
        self.type.check_sanity()
        self.round()
        lst_addable = self.type.addable_parameters()
        assert len(self.parameters) == len(lst_addable)
        # additional check that if addable parameters is number
        for n, addable in enumerate(lst_addable):
            if addable and not convertable_to_float(self.parameters[n]):
                raise ValueError("Some parameters that should be addable is not float number!")

    def try_move_fac(self):
        """ Try to move outer factor into addable parameters. """
        self.check_sanity()
        lst_addable = self.type.addable_parameters()
        if sum(lst_addable) > 0:
            fac = self.fac
            for n, addable in enumerate(lst_addable):
                if addable:
                    self.parameters[n] *= fac
            self.fac = 1
        return self

    def mergable(self, other: "XCInfo") -> bool:
        """ Check whether two terms are mergable. """
        self.check_sanity()
        other.check_sanity()
        if (
                self.name != other.name
                or self.type != other.type
                or self.additional != other.additional
        ):
            return False

        lst_addable = self.type.addable_parameters()
        assert len(self.parameters) == len(other.parameters) == len(lst_addable)
        for n, addable in enumerate(lst_addable):
            if not addable and self.parameters[n] != other.parameters[n]:
                return False
        return True

    def merge(self, other: "XCInfo", inplace=False) -> "XCInfo":
        """ Merge two exchange-correlation terms by adding factors or addable parameters. """
        assert self.mergable(other)
        this = self
        other = other.copy()
        if not inplace:
            this = self.copy()
        this.try_move_fac()
        other.try_move_fac()

        lst_addable = this.type.addable_parameters()
        assert len(this.parameters) == len(other.parameters) == len(lst_addable)
        if sum(lst_addable) == 0:
            this.fac += other.fac
        else:
            for n, addable in enumerate(lst_addable):
                if addable:
                    this.parameters[n] += other.parameters[n]
        return this

    def is_zero(self, cutoff=1e-10):
        """ Check if this term is zero (by checking factor or addable parameters). """
        self.check_sanity()
        if abs(self.fac) < cutoff:
            return True
        lst_addable = self.type.addable_parameters()
        if sum(lst_addable) == 0:
            return False
        small = [abs(self.parameters[n]) < cutoff for n, addable in enumerate(lst_addable) if addable]
        return all(small)


def convertable_to_float(v):
    try:
        float(v)
        return True
    except ValueError:
        return False
