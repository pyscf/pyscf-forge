"""
Exchange-correlation code parsing utility for doubly hybrid
"""
import itertools
import re
import copy
import warnings
from typing import List
from .xctype import XCType
from .xcinfo import XCInfo, REGEX_XC
from .xcjson import _NAME_WITH_DASH, FUNCTIONALS_DICT


class XCList:
    """
    Stores exchange and correlation information and processing xc tokens.
    """
    _xc_list: List[XCInfo]
    """ List of detailed exchange and correlation information. """

    @property
    def xc_list(self):
        for info in self._xc_list:
            info.round()
        return self._xc_list

    @xc_list.setter
    def xc_list(self, xc_list):
        for info in xc_list:
            info.round()
        self._xc_list = xc_list

    def __init__(self, token=None, code_scf=None, **kwargs):
        self.xc_list = []
        if token:
            if not isinstance(code_scf, bool):
                raise ValueError("Must pass boolean option of `code_scf`.")
            self.xc_list = self.build_from_token(token, code_scf, **kwargs)

    @classmethod
    def build_from_token(
            cls, token: str, code_scf: bool,
            trim=True,
    ):
        xclist = cls()
        xclist.xc_list = cls.parse_token(token, code_scf)
        if trim:
            xclist.trim()
        return xclist

    @classmethod
    def build_from_list(cls, xc_list: list, trim=True):
        xclist = cls()
        xclist.xc_list = xc_list
        if trim:
            xclist.trim()
        return xclist

    # region XCList parsing

    @classmethod
    def parse_token(cls, token: str, code_scf: bool):
        """ Parse xc token to list of XCInfo without any modification or merging
        (except name with dash).

        Notes
        -----
        For regex of xc token, one may try: https://regex101.com/r/YeQU5m/1

        Groups of regex search

        0. match all token for an entire xc code for one term
        1. match if comma in string; if exists, then split exchange and correlation parts (pyscf convention);
        2. sign of factor
        3. factor (absolute value) with asterisk
        4. name of xc term
        5. list of parameters with parentheses
        """
        token = token.upper().replace(" ", "")
        for key, val in _NAME_WITH_DASH:
            token = token.replace(key, val)
        if len(token) == 0:
            return []
        # handle case where only exchange terms
        all_exch = False
        if token[-1] == ",":
            all_exch = True
            token = token[:-1]
        # match
        match = re.findall(REGEX_XC, token)
        # sanity check: matched patterns should be exactly equilvant to original token
        if token != "".join([group[0] for group in match]):
            raise ValueError(
                "XC token {:} XCTypes not successfully parsed.\n"
                "Regex match of this token becomes {:}".format(token, "".join([group[0] for group in match])))
        # check if splitting exchange and correlation is required
        # sanity check first: only zero or one comma
        comma_first = [group[0][0] == "," for group in match]
        split_xc = sum(comma_first)
        if split_xc > 1 or (split_xc == 1 and all_exch):
            raise ValueError("Only zero or one comma is required to split exchange and correlation parts!")
        index_split = -1
        if split_xc:
            index_split = comma_first.index(True)
        if all_exch:
            split_xc = 1
            index_split = len(comma_first)
        xc_list = []
        for n, re_group in enumerate(match):
            if not split_xc:
                guess_type = XCType.HYB
            elif n < index_split:
                guess_type = XCType.EXCH
            else:
                guess_type = XCType.CORR
            # see if is known doubly hybrid functional
            token, _, sgn, fac, name, parameters = re_group
            sgn = (-1)**sgn.count("-")
            fac = sgn if len(fac) == 0 else sgn * float(fac[:-1])

            FUNCTIONALS_DICT_UPPER = {key.upper(): val for (key, val) in FUNCTIONALS_DICT.items()}
            if name in FUNCTIONALS_DICT_UPPER and guess_type == XCType.HYB:
                if len(parameters) != 0:
                    raise ValueError(
                        "XC name {:} have parameters as hybrid functional, which is not acceptable.".format(name))
                entry = FUNCTIONALS_DICT_UPPER[name]
                if code_scf:
                    xc_list_add = cls.parse_token(entry.get("code_scf", entry["code"]), code_scf)
                else:
                    xc_list_add = cls.parse_token(entry["code"], code_scf)
                for xc_info in xc_list_add:
                    xc_info.fac *= fac
                xc_list += xc_list_add
                continue
            # otherwise, parse contribution information
            xc_infos = XCInfo.parse_xc_info(token, guess_type)
            # xc_info may be list[XCInfo], so we handle it by list
            if not isinstance(xc_infos, list):
                xc_infos = [xc_infos]
            for xc_info in xc_infos:
                if not (code_scf and not (xc_info.type & XCType.RUNG_LOW)):
                    xc_list.append(xc_info)
        return xc_list

    def extract_by_xctype(self, xc_type: XCType or callable) -> "XCList":
        """ Extract xc components by type of xc.

        Parameters
        ----------
        xc_type
            If ``xc_type`` is ``XCType`` instance, then extract xc info by this type.
            Otherwise, ``xc_type`` is a rule to define which type may be acceptable.
        """
        if isinstance(xc_type, XCType):
            xc_list = [info for info in self.xc_list if xc_type & info.type]
        else:
            xc_list = [info for info in self.xc_list if xc_type(info.type)]
        ret = XCList()
        ret.xc_list = copy.deepcopy(xc_list)
        return ret.trim()

    # endregion

    # region merging

    def trim(self):
        """ Merge same items and remove terms that contribution coefficient (factor) is zero. """
        xc_list_trimed = []  # type: list[XCInfo]
        for info1 in self.xc_list:
            skip_info = False
            # see if mergable
            for info2 in xc_list_trimed:
                if info2.mergable(info1):
                    info2.merge(info1, inplace=True)
                    skip_info = True
                    break
            if not skip_info:
                xc_list_trimed.append(info1)
        # check values finally
        remove_index = [n for n, info in enumerate(xc_list_trimed) if info.is_zero()]
        for n in remove_index[::-1]:
            xc_list_trimed.pop(n)
        self.xc_list = xc_list_trimed
        return self.sort()

    # endregion

    # region XCList basic utilities

    def copy(self):
        return copy.deepcopy(self)

    @property
    def token(self) -> str:
        """ Return a token that represents xc functional in a somehow standard way. """
        self.sort()
        if len(self) == 0:
            return ""
        xc_list_x = self.extract_by_xctype(lambda t: XCType.CORR not in t)
        xc_list_c = self.extract_by_xctype(XCType.CORR)
        token = " + ".join([info.token for info in xc_list_x]).replace("+ -", "-")
        if len(xc_list_c) > 0:
            token += ", " + " + ".join([info.token for info in xc_list_c]).replace("+ -", "-")
        if len(self.extract_by_xctype(XCType.HYB)) == 0 and len(xc_list_c) == 0:
            token += ","
        token = token.strip()
        return token

    def sort(self):
        """ Sort list of xc in unique way. """
        xc_list = self.copy()

        # way of sort
        def token_for_sort(info: XCInfo):
            t = info.name
            t += str(info.parameters)
            t += str(info.type)
            t += str(info.fac)
            return t

        def exclude(l, t):
            idx = l.index(t)
            l.pop(idx)

        def extracting(l, t):
            if isinstance(t, XCType):
                return [i for i in l if t & i.type]
            else:
                return [i for i in l if t(i.type)]

        # 1. split general and correlation contribution
        xc_list_x = extracting(xc_list, lambda t: XCType.CORR not in t)
        xc_list_c = extracting(xc_list, XCType.CORR)
        # 2. extract HF and RSH first, then pure and hybrid
        xc_lists_x = []
        for xctype in [XCType.HF, XCType.RSH, XCType.EXCH, XCType.HYB, lambda _: True]:
            inner_list = extracting(xc_list_x, xctype)
            for info in inner_list:
                exclude(xc_list_x, info)
            xc_lists_x.append(inner_list)
        # 3. extract pure, MP2, IEPA, RPA, VDW
        xc_lists_c = []
        for xctype in [
                XCType.RUNG_LOW, XCType.MP2, XCType.RSMP2, XCType.IEPA, XCType.RPA,
                XCType.NUC_CORR, XCType.VDW, lambda _: True]:
            inner_list = extracting(xc_list_c, xctype)
            for info in inner_list:
                exclude(xc_list_c, info)
            xc_lists_c.append(inner_list)
        # 4. sort each category
        for lst in xc_lists_x + xc_lists_c:
            lst.sort(key=token_for_sort)
        xc_lists_sorted_x = list(itertools.chain(*xc_lists_x))
        xc_lists_sorted_c = list(itertools.chain(*xc_lists_c))
        new_list = XCList()
        new_list.xc_list = xc_lists_sorted_x + xc_lists_sorted_c
        # sanity check
        if new_list != self:
            warnings.warn("Sorted xc_list is not the same to the original xc list. Double check may required.")
        self.xc_list = new_list.xc_list
        return self

    def remove(self, info: XCInfo or "XCList", inplace=True):
        """ Remove one term form list. """
        lst = self if inplace else self.copy()
        if isinstance(info, XCInfo):
            lst.xc_list.remove(info)
        else:  # isinstance(info, XCList)
            for i in info:
                lst.xc_list.remove(i)
        return lst.trim()

    def __iter__(self):
        for info in self.xc_list:
            yield info

    def __len__(self):
        return len(self.xc_list)

    def __getitem__(self, item):
        return self.xc_list[item]

    def __add__(self, other: "XCList"):
        new_obj = self.copy()
        new_obj += other
        return new_obj

    def __iadd__(self, other: "XCList"):
        self.xc_list += other.xc_list

    def __mul__(self, other: float):
        new_obj = self.copy()
        new_obj *= other
        return new_obj

    def __imul__(self, other: float):
        for xc_info in self.xc_list:
            xc_info.fac *= other
        return self

    def __eq__(self, other: "XCList"):
        tokens_self = [str(info) for info in self.xc_list]
        tokens_other = [str(info) for info in other.xc_list]
        tokens_self.sort()
        tokens_other.sort()
        return "".join(tokens_self) == "".join(tokens_other)

    # endregion


class XCDH:
    """ XC object for doubly hybrid functional.

    One may initialize XCDH object by one string (such as "XYG3"),
    or initialize by two strings indicating SCF and energy parts
    (such as ["PBE", "0.5*HF + 0.5*PBE, 0.75*PBE + 0.25*MP2"]).
    """
    xc_eng: XCList
    """ XC list for energy evaluation. """
    xc_scf: XCList
    """ XC list for SCF evaluation. """

    def __init__(self, token=None, **kwargs):
        if token:
            self.build_from_token(token, **kwargs)

    def build_from_token(self, token: str or tuple or list, **kwargs):
        if isinstance(token, (tuple, list)):
            if len(token) != 2:
                raise ValueError("If token passed in by tuple, then it must be two parts (scf, eng).")
            self.xc_scf = XCList.build_from_token(token[0], True, **kwargs)
            self.xc_eng = XCList.build_from_token(token[1], False, **kwargs)
        else:
            self.xc_scf = XCList.build_from_token(token, True, **kwargs)
            self.xc_eng = XCList.build_from_token(token, False, **kwargs)


if __name__ == '__main__':
    lt = XCList.build_from_token("XYG3", False)
    print(lt.xc_list)
    print(lt.token)
