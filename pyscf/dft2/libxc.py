#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          Susi Lehtola <susi.lehtola@gmail.com>

'''
XC functional, the interface to libxc
(http://www.tddft.org/programs/octopus/wiki/index.php/Libxc)
'''

import sys
import warnings
import math
import numpy
from functools import lru_cache
from pyscf import lib
from pyscf.dft.xc.utils import remove_dup, format_xc_code
from pyscf.dft import xc_deriv
from pyscf.dft.libxc import XC_CODES, XC, PROBLEMATIC_XC, XC_KEYS, XC_ALIAS, _NAME_WITH_DASH
from pyscf import __config__
from .libxc_cffi import ffi as _ffi, lib as _lib

def libxc_version():
    '''Returns the version of libxc'''
    return _ffi.string(_lib.xc_version_string()).decode("UTF-8")

def libxc_reference():
    '''Returns the reference to libxc'''
    return _ffi.string(_lib.xc_reference()).decode("UTF-8")

def libxc_reference_doi():
    '''Returns the reference to libxc'''
    return _ffi.string(_lib.xc_reference_doi()).decode("UTF-8")

__version__ = libxc_version()
__reference__ = libxc_reference()
__reference_doi__ = libxc_reference_doi()

_XC_FUNC_TYPE_PTR_TYPE = _ffi.typeof("xc_func_type*")
_XC_FUNC_TYPE_ARRAY_TYPE = _ffi.typeof("xc_func_type[]")
_DOUBLE_PTR_TYPE = _ffi.typeof("double*")
_DOUBLE_ARRAY_TYPE = _ffi.typeof("double[]")
_DOUBLE_ARRAY_2_TYPE = _ffi.typeof("double[2]")
_DOUBLE_ARRAY_3_TYPE = _ffi.typeof("double[3]")

def xc_func_init(xc_code, nspin=_lib.XC_UNPOLARIZED):
    func = _ffi.new(_XC_FUNC_TYPE_PTR_TYPE)
    if _lib.xc_func_init(func, xc_code, nspin) != 0:
        raise ValueError(f"Functional '{xc_code}' not found")
    #  xc_func_end() will automatically be called when `func` is destructed
    return _ffi.gc(func, _lib.xc_func_end)

def _to_xc_objs(xc_code, spin=0):
    '''Parse an xc_code into xc_objs
       xc_objs is a list containing all the cffi objects representing the
       `xc_func_type` struct of each functional components'''
    if isinstance(xc_code, list):
        return xc_code
    hyb, fn_facs = parse_xc(xc_code)
    nspin = _lib.XC_POLARIZED if spin else _lib.XC_UNPOLARIZED
    return [xc_func_init(xid, nspin) for xid, fac in fn_facs]

def _to_xc_info(xc_code, spin=0):
    '''Parse an xc_code into xc_info
       xc_info is a tuple containing four components:
       1. xc_objs: a list containing all the cffi objects representing the
          `xc_func_type` struct of each functional components
       2. xc_arr: a cffi array of `xc_func_type` struct
          of each functional components
       3. hyb: hybrid coefficients as produced by `parse_xc()`
       4. fn_facs: functional factors as produced by `parse_xc()`
    '''
    if isinstance(xc_code, tuple):
        return xc_code
    hyb, fn_facs = parse_xc(xc_code)
    nspin = _lib.XC_POLARIZED if spin else _lib.XC_UNPOLARIZED
    n = len(fn_facs)
    xc_arr = _ffi.new(_XC_FUNC_TYPE_ARRAY_TYPE, n)
    xc_objs = [_ffi.addressof(xc_arr, i) for i in range(n)]
    for (xid, fac), func in zip(fn_facs, xc_objs):
        if _lib.xc_func_init(func, xid, nspin) != 0:
            raise ValueError(f"Functional '{xid}' not found")

    def destructor(xc_arr):
        for func in xc_objs:
            _lib.xc_func_end(func)

    xc_arr = _ffi.gc(xc_arr, destructor)
    return xc_objs, xc_arr, hyb, fn_facs

XC_CODE_WARNING = 'Use of xc_code may cause performance issues. Use XCFunctional instead.'

def xc_reference(xc_code):
    '''Returns the references to a functional as a list of str'''
    warnings.warn(XC_CODE_WARNING)
    return _xc_reference(_to_xc_objs(xc_code))

def _xc_reference(xc_objs):
    refs = []
    for xc in xc_objs:
        for i in range(_lib.XC_MAX_REFERENCES):
            c_ref = xc.info.refs[i]
            if c_ref:
                refs.append(_ffi.string(c_ref.ref).decode("UTF-8"))
    return refs

def _build_xc_family_table():
    def add(d, key, value):
        key = getattr(_lib, key, None)
        if key is None:
            return
        d[key] = value
    table = {}
    add(table, "XC_FAMILY_LDA",      (0, False))
    add(table, "XC_FAMILY_GGA",      (1, False))
    add(table, "XC_FAMILY_MGGA",     (2, False))
    add(table, "XC_FAMILY_HYB_LDA",  (0, True ))
    add(table, "XC_FAMILY_HYB_GGA",  (1, True ))
    add(table, "XC_FAMILY_HYB_MGGA", (2, True ))
    return table

_XC_TYPE_TABLE = _build_xc_family_table()
_XC_FAMILIES = ["LDA", "GGA", "MGGA", "UNKNOWN"]

def xc_type(xc_code):
    '''Returns the type of a functional as a str
    '''
    warnings.warn(XC_CODE_WARNING)
    if xc_code is None:
        return None
    if isinstance(xc_code, str):
        if '__VV10' in xc_code:
            raise RuntimeError('Deprecated notation for NLC functional.')
    xc_objs = _to_xc_objs(xc_code)
    return _xc_type(xc_objs)

def _xc_type(xc_objs):
    if not xc_objs:
        return 'HF'
    types = (_XC_TYPE_TABLE.get(xc.info.family, (-1, None))[0] for xc in xc_objs)
    return _XC_FAMILIES[max(types)]

def is_lda(xc_code):
    '''Returns True if a functional is a LDA
    '''
    warnings.warn(XC_CODE_WARNING)
    return xc_type(xc_code) == 'LDA'

def _is_lda(xc_objs):
    return _xc_type(xc_objs) == 'LDA'

def is_hybrid_xc(xc_code):
    '''Returns True if a functional is a hybrid functional
    '''
    warnings.warn(XC_CODE_WARNING)
    if xc_code is None:
        return False
    elif isinstance(xc_code, str):
        if 'HF' in xc_code:
            return True
        xc_info = _to_xc_info(xc_code)
        return _is_hybrid_xc(xc_info)
    elif numpy.issubdtype(type(xc_code), numpy.integer):
        xc_info = _to_xc_info(xc_code)
        return _is_hybrid_xc(xc_info)
    else:
        return any((is_hybrid_xc(x) for x in xc_code))

def _is_hybrid_xc(xc_info):
    if _hybrid_coeff(xc_info) != 0:
        return True
    if _rsh_coeff(xc_info) != (0, 0, 0):
        return True
    return False

def is_meta_gga(xc_code):
    '''Returns True if a functional is a meta GGA
    '''
    warnings.warn(XC_CODE_WARNING)
    return xc_type(xc_code) == 'MGGA'

def _is_meta_gga(xc_objs):
    return _xc_type(xc_objs) == 'MGGA'

def is_gga(xc_code):
    '''Returns True if a functional is a GGA
    '''
    warnings.warn(XC_CODE_WARNING)
    return xc_type(xc_code) == 'GGA'

def _is_gga(xc_objs):
    return _xc_type(xc_objs) == 'GGA'

@lru_cache(100)
def is_nlc(xc_code):
    # identify nlc by xc_code itself if enable_nlc is None
    warnings.warn(XC_CODE_WARNING)
    if isinstance(xc_code, str) or \
            numpy.issubdtype(type(xc_code), numpy.integer):
        xc_objs = _to_xc_objs(xc_code)
        return _is_nlc(xc_objs)
    else:
        return any((is_nlc(x) for x in xc_code))

def _is_nlc(xc_objs):
    return any(xc.info.flags & _lib.XC_FLAGS_VV10 for xc in xc_objs)

def needs_laplacian(xc_code):
    warnings.warn(XC_CODE_WARNING)
    xc_objs = _to_xc_objs(xc_code)
    return _needs_laplacian(xc_objs)

def _needs_laplacian(xc_objs):
    return any(xc.info.flags & _lib.XC_FLAGS_NEEDS_LAPLACIAN for xc in xc_objs)

_DERIV_FLAGS_TABLE = [
    _lib.XC_FLAGS_HAVE_EXC, # order 0
    _lib.XC_FLAGS_HAVE_VXC, # order 1
    _lib.XC_FLAGS_HAVE_FXC, # order 2
    _lib.XC_FLAGS_HAVE_KXC, # order 3
    _lib.XC_FLAGS_HAVE_LXC, # order 4
]

@lru_cache(100)
def max_deriv_order(xc_code):
    warnings.warn(XC_CODE_WARNING)
    xc_objs = _to_xc_objs(xc_code)
    return _max_deriv_order(xc_objs)

def _max_deriv_order(xc_objs):
    if xc_objs:
        # find the minimum order of all functionals
        order = 4
        for xc in xc_objs:
            flags = xc.info.flags
            for i in range(order, -1, -1):
                if flags & _DERIV_FLAGS_TABLE[i]:
                    order = i
                    break
            else:
                return -1
        return order
    else:
        return 3

def test_deriv_order(xc_code, deriv, raise_error=False):
    support = deriv <= max_deriv_order(xc_code)
    if not support and raise_error:
        from pyscf.dft import xcfun
        msg = ('libxc library does not support derivative order %d for  %s' %
               (deriv, xc_code))
        try:
            if xcfun.test_deriv_order(xc_code, deriv, raise_error=False):
                msg += ('''
    This functional derivative is supported in the xcfun library.
    The following code can be used to change the libxc library to xcfun library:

        from pyscf.dft import xcfun
        mf._numint.libxc = xcfun
''')
            raise NotImplementedError(msg)
        except KeyError as e:
            sys.stderr.write('\n'+msg+'\n')
            sys.stderr.write('%s not found in xcfun library\n\n' % xc_code)
            raise e
    return support

@lru_cache(100)
def hybrid_coeff(xc_code, spin=0):
    '''Support recursively defining hybrid functional
    '''
    warnings.warn(XC_CODE_WARNING)
    xc_info = _to_xc_info(xc_code)
    return _hybrid_coeff(xc_info)

def _hybrid_coeff(xc_info, spin=0):
    xc_objs, xc_arr, hyb, fn_facs = xc_info
    hybs = (fac * _lib.xc_hyb_exx_coef(xc) for xc, (xid, fac) in zip(xc_objs, fn_facs)
            if _XC_TYPE_TABLE.get(xc.info.family, (None, False))[1])
    return hyb[0] + sum(hybs)

@lru_cache(100)
def nlc_coeff(xc_code):
    '''Get NLC coefficients
    '''
    warnings.warn(XC_CODE_WARNING)
    xc_info = _to_xc_info(xc_code)
    return _nlc_coeff(xc_info)

def _nlc_coeff(xc_info):
    xc_objs, xc_arr, hyb, fn_facs = xc_info
    nlc_pars = []
    nlc_tmp = _ffi.new(_DOUBLE_ARRAY_2_TYPE)
    for xc, (xid, fac) in zip(xc_objs, fn_facs):
        if xc.info.flags & _lib.XC_FLAGS_VV10:
            _lib.xc_nlc_coef(xc, nlc_tmp, nlc_tmp + 1)
            nlc_pars.append((tuple(nlc_tmp), fac))
    return tuple(nlc_pars)

@lru_cache(100)
def rsh_coeff(xc_code):
    '''Range-separated parameter and HF exchange components: omega, alpha, beta

    Exc_RSH = c_LR * LR_HFX + c_SR * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
            = alpha * HFX   + beta * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
            = alpha * LR_HFX + hyb * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec

    SR_HFX = < pi | (1-erf(-omega r_{12}))/r_{12} | iq >
    LR_HFX = < pi | erf(-omega r_{12})/r_{12} | iq >
    alpha = c_LR
    beta = c_SR - c_LR = hyb - alpha
    '''
    warnings.warn(XC_CODE_WARNING)
    if xc_code is None:
        return 0, 0, 0

    check_omega = True
    if isinstance(xc_code, str) and ',' in xc_code:
        # Parse only X part for the RSH coefficients.  This is to handle
        # exceptions for C functionals such as M11.
        xc_code = format_xc_code(xc_code)
        xc_code = xc_code.split(',')[0] + ','
        if 'SR_HF' in xc_code or 'LR_HF' in xc_code or 'RSH(' in xc_code:
            check_omega = False
    xc_info = _to_xc_info(xc_code)
    return _rsh_coeff(xc_info, check_omega)

def _rsh_coeff(xc_info, check_omega=True):
    xc_objs, xc_arr, hyb, fn_facs = xc_info

    hyb, alpha, omega = hyb
    beta = hyb - alpha
    rsh_pars = [omega, alpha, beta]
    rsh_tmp = _ffi.new(_DOUBLE_ARRAY_3_TYPE)
    for xc, (xid, fac) in zip(xc_objs, fn_facs):
        _lib.xc_hyb_cam_coef(xc, rsh_tmp, rsh_tmp + 1, rsh_tmp + 2)
        if rsh_pars[0] == 0:
            rsh_pars[0] = rsh_tmp[0]
        elif check_omega:
            # Check functional is actually a CAM functional
            if rsh_tmp[0] != 0 and not (xc.info.flags & _lib.XC_FLAGS_HYB_CAM):
                raise KeyError('Libxc functional %i employs a range separation '
                               'kernel that is not supported in PySCF' % xid)
            # Check omega
            if (rsh_tmp[0] != 0 and rsh_pars[0] != rsh_tmp[0]):
                raise ValueError('Different values of omega found for RSH functionals')
        rsh_pars[1] += rsh_tmp[1] * fac
        rsh_pars[2] += rsh_tmp[2] * fac
    return tuple(rsh_pars)

def parse_xc_name(xc_name='LDA,VWN'):
    '''Convert the XC functional name to libxc library internal ID.
    '''
    fn_facs = parse_xc(xc_name)[1]
    return fn_facs[0][0], fn_facs[1][0]

@lru_cache(100)
def parse_xc(description):
    r'''Rules to input functional description:

    * The given functional description must be a one-line string.
    * The functional description is case-insensitive.
    * The functional description string has two parts, separated by ",".  The
      first part describes the exchange functional, the second is the correlation
      functional.

      - If "," was not in string, the entire string is considered as a
        compound XC functional (including both X and C functionals, such as b3lyp).
      - To input only X functional (without C functional), leave the second
        part blank. E.g. description='slater,' means pure LDA functional.
      - To neglect X functional (just apply C functional), leave the first
        part blank. E.g. description=',vwn' means pure VWN functional.
      - If compound XC functional is specified, no matter whether it is in the
        X part (the string in front of comma) or the C part (the string behind
        comma), both X and C functionals of the compound XC functional will be
        used.

    * The functional name can be placed in arbitrary order.  Two name needs to
      be separated by operators "+" or "-".  Blank spaces are ignored.
      NOTE the parser only reads operators "+" "-" "*".  / is not in support.
    * A functional name can have at most one factor.  If the factor is not
      given, it is set to 1.  Compound functional can be scaled as a unit. For
      example '0.5*b3lyp' is equivalent to
      'HF*0.1 + .04*LDA + .36*B88, .405*LYP + .095*VWN'
    * String "HF" stands for exact exchange (HF K matrix).  Putting "HF" in
      correlation functional part is the same to putting "HF" in exchange
      part.
    * String "RSH" means range-separated operator. Its format is
      RSH(omega, alpha, beta).  Another way to input RSH is to use keywords
      SR_HF and LR_HF: "SR_HF(0.1) * alpha_plus_beta" and "LR_HF(0.1) *
      alpha" where the number in parenthesis is the value of omega.
    * Be careful with the libxc convention on GGA functional, in which the LDA
      contribution has been included.

    Args:
        description : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" was appeared in the string, it stands for the exact exchange.

    Returns:
        decoded XC description, with the data structure
        (hybrid, alpha, omega), ((libxc-Id, fac), (libxc-Id, fac), ...)
    '''  # noqa: E501

    hyb = [0, 0, 0]  # hybrid, alpha, omega (== SR_HF, LR_HF, omega)
    if description is None:
        return tuple(hyb), ()
    elif numpy.issubdtype(type(description), numpy.integer):
        return tuple(hyb), ((description, 1.),)
    elif not isinstance(description, str): #isinstance(description, (tuple,list)):
        return parse_xc('%s,%s' % tuple(description))

    description = description.upper()
    if '-D3' in description or '-D4' in description:
        from pyscf.scf.dispersion import parse_dft
        description, _, _ = parse_dft(description)
        description = description.upper()

    if (description in ('B3P86', 'B3LYP', 'X3LYP') and
        not getattr(parse_xc, 'b3lyp5_warned', False) and
        not hasattr(__config__, 'B3LYP_WITH_VWN5')):
        parse_xc.b3lyp5_warned = True
        warnings.warn('Since PySCF-2.3, B3LYP (and B3P86) are changed to the VWN-RPA variant, '
                      'corresponding to the original definition by Stephens et al. (issue 1480) '
                      'and the same as the B3LYP functional in Gaussian. '
                      'To restore the VWN5 definition, you can put the setting '
                      '"B3LYP_WITH_VWN5 = True" in pyscf_conf.py')

    def assign_omega(omega, hyb_or_sr, lr=0):
        if hyb[2] == omega or omega == 0:
            hyb[0] += hyb_or_sr
            hyb[1] += lr
        elif hyb[2] == 0:
            hyb[0] += hyb_or_sr
            hyb[1] += lr
            hyb[2] = omega
        else:
            raise ValueError('Different values of omega found for RSH functionals')
    fn_facs = []

    def parse_token(token, ftype, search_xc_alias=False):
        if token:
            if token[0] == '-':
                sign = -1
                token = token[1:]
            else:
                sign = 1
            if '*' in token:
                fac, key = token.split('*')
                if fac[0].isalpha():
                    fac, key = key, fac
                fac = sign * float(fac)
            else:
                fac, key = sign, token

            if key[:3] == 'RSH':
                # RSH(alpha; beta; omega): Range-separated-hybrid functional
                # See also utils.format_xc_code
                alpha, beta, omega = [float(x) for x in key[4:-1].split(';')]
                assign_omega(omega, fac*(alpha+beta), fac*alpha)
            elif key == 'HF':
                hyb[0] += fac
                hyb[1] += fac  # also add to LR_HF
            elif 'SR_HF' in key:
                if '(' in key:
                    omega = float(key.split('(')[1].split(')')[0])
                    assign_omega(omega, fac, 0)
                else:  # Assuming this omega the same to the existing omega
                    hyb[0] += fac
            elif 'LR_HF' in key:
                if '(' in key:
                    omega = float(key.split('(')[1].split(')')[0])
                    assign_omega(omega, 0, fac)
                else:
                    hyb[1] += fac  # == alpha
            elif key.isdigit():
                fn_facs.append((int(key), fac))
            else:
                if search_xc_alias and key in XC_ALIAS:
                    x_id = XC_ALIAS[key]
                elif key in XC_CODES:
                    x_id = XC_CODES[key]
                else:
                    possible_xc_for = fpossible_dic[ftype]
                    possible_xc = XC_KEYS.intersection(possible_xc_for(key))
                    if possible_xc:
                        if len(possible_xc) > 1:
                            sys.stderr.write('Possible xc_code %s matches %s. '
                                             % (list(possible_xc), key))
                            for x_id in possible_xc:  # Prefer X functional
                                if '_X_' in x_id:
                                    break
                            else:
                                x_id = possible_xc.pop()
                            sys.stderr.write('XC parser takes %s\n' % x_id)
                            sys.stderr.write('You can add prefix to %s for a '
                                             'specific functional (e.g. X_%s, '
                                             'HYB_MGGA_X_%s)\n'
                                             % (key, key, key))
                        else:
                            x_id = possible_xc.pop()
                        x_id = XC_CODES[x_id]
                    else:
                        # Some libxc functionals may not be listed in the
                        # XC_CODES table. Query libxc directly
                        x_id = _lib.xc_functional_get_number(key.encode())
                        if x_id == -1:
                            raise KeyError(f"LibXCFunctional: name '{key}' not found.")
                if isinstance(x_id, str):
                    hyb1, fn_facs1 = parse_xc(x_id)
                    # Recursively scale the composed functional, to support e.g. '0.5*b3lyp'
                    if hyb1[0] != 0 or hyb1[1] != 0:
                        assign_omega(hyb1[2], hyb1[0]*fac, hyb1[1]*fac)
                    fn_facs.extend([(xid, c*fac) for xid, c in fn_facs1])
                elif x_id is None:
                    raise NotImplementedError('%s functional %s' % (ftype, key))
                else:
                    fn_facs.append((x_id, fac))
    def possible_x_for(key):
        return {key,
                    'LDA_X_'+key, 'GGA_X_'+key, 'MGGA_X_'+key,
                    'HYB_GGA_X_'+key, 'HYB_MGGA_X_'+key}
    def possible_xc_for(key):
        return {key, 'LDA_XC_'+key, 'GGA_XC_'+key, 'MGGA_XC_'+key,
                    'HYB_LDA_XC_'+key, 'HYB_GGA_XC_'+key, 'HYB_MGGA_XC_'+key}
    def possible_k_for(key):
        return {key,
                    'LDA_K_'+key, 'GGA_K_'+key,}
    def possible_x_k_for(key):
        return possible_x_for(key).union(possible_k_for(key))
    def possible_c_for(key):
        return {key,
                    'LDA_C_'+key, 'GGA_C_'+key, 'MGGA_C_'+key}
    fpossible_dic = {'X': possible_x_for,
                     'C': possible_c_for,
                     'compound XC': possible_xc_for,
                     'K': possible_k_for,
                     'X or K': possible_x_k_for}

    description = format_xc_code(description)

    if '-' in description:  # To handle e.g. M06-L
        for key in _NAME_WITH_DASH:
            if key in description:
                description = description.replace(key, _NAME_WITH_DASH[key])

    if ',' in description:
        x_code, c_code = description.split(',')
        for token in x_code.replace('-', '+-').replace(';+', ';').split('+'):
            parse_token(token, 'X or K')
        for token in c_code.replace('-', '+-').replace(';+', ';').split('+'):
            parse_token(token, 'C')
    else:
        for token in description.replace('-', '+-').replace(';+', ';').split('+'):
            # dftd3 cannot be used in a custom xc description
            assert '-d3' not in token
            parse_token(token, 'compound XC', search_xc_alias=True)
    if hyb[2] == 0: # No omega is assigned. LR_HF is 0 for normal Coulomb operator
        hyb[1] = 0
    return tuple(hyb), tuple(remove_dup(fn_facs))

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    r'''Interface to call libxc library to evaluate XC functional, potential
    and functional derivatives.

    * The given functional xc_code must be a one-line string.
    * The functional xc_code is case-insensitive.
    * The functional xc_code string has two parts, separated by ",".  The
      first part describes the exchange functional, the second part sets the
      correlation functional.

      - If "," not appeared in string, the entire string is treated as the
        name of a compound functional (containing both the exchange and
        the correlation functional) which was declared in the functional
        aliases list. The full list of functional aliases can be obtained by
        calling the function pyscf.dft.xcfun.XC_ALIAS.keys() .

        If the string was not found in the aliased functional list, it is
        treated as X functional.

      - To input only X functional (without C functional), leave the second
        part blank. E.g. description='slater,' means a functional with LDA
        contribution only.

      - To neglect the contribution of X functional (just apply C functional),
        leave blank in the first part, e.g. description=',vwn' means a
        functional with VWN only.

      - If compound XC functional is specified, no matter whether it is in the
        X part (the string in front of comma) or the C part (the string behind
        comma), both X and C functionals of the compound XC functional will be
        used.

    * The functional name can be placed in arbitrary order.  Two names need to
      be separated by operators "+" or "-".  Blank spaces are ignored.
      NOTE the parser only reads operators "+" "-" "*".  / is not supported.

    * A functional name can have at most one factor.  If the factor is not
      given, it is set to 1.  Compound functional can be scaled as a unit. For
      example '0.5*b3lyp' is equivalent to
      'HF*0.1 + .04*LDA + .36*B88, .405*LYP + .095*VWN'

    * String "HF" stands for exact exchange (HF K matrix).  "HF" can be put in
      the correlation functional part (after comma). Putting "HF" in the
      correlation part is the same to putting "HF" in the exchange part.

    * String "RSH" means range-separated operator. Its format is
      RSH(omega, alpha, beta).  Another way to input RSH is to use keywords
      SR_HF and LR_HF: "SR_HF(0.1) * alpha_plus_beta" and "LR_HF(0.1) *
      alpha" where the number in parenthesis is the value of omega.

    * Be careful with the libxc convention of GGA functional, in which the LDA
      contribution is included.

    Args:
        xc_code : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" (exact exchange) is appeared in the string, the HF part will
            be skipped.  If an empty string "" is given, the returns exc, vxc,...
            will be vectors of zeros.
        rho : ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))

    Kwargs:
        spin : int
            spin polarized if spin > 0
        relativity : int
            No effects.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        ex, vxc, fxc, kxc

        where

        * vxc = (vrho, vsigma, vlapl, vtau) for restricted case

        * vxc for unrestricted case
          | vrho[:,2]   = (u, d)
          | vsigma[:,3] = (uu, ud, dd)
          | vlapl[:,2]  = (u, d)
          | vtau[:,2]   = (u, d)

        * fxc for restricted case:
          (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)

        * fxc for unrestricted case:
          | v2rho2[:,3]     = (u_u, u_d, d_d)
          | v2rhosigma[:,6] = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
          | v2sigma2[:,6]   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
          | v2lapl2[:,3]
          | v2tau2[:,3]     = (u_u, u_d, d_d)
          | v2rholapl[:,4]
          | v2rhotau[:,4]   = (u_u, u_d, d_u, d_d)
          | v2lapltau[:,4]
          | v2sigmalapl[:,6]
          | v2sigmatau[:,6] = (uu_u, uu_d, ud_u, ud_d, dd_u, dd_d)

        * kxc for restricted case:
          (v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
           v3rho2lapl, v3rho2tau,
           v3rhosigmalapl, v3rhosigmatau,
           v3rholapl2, v3rholapltau, v3rhotau2,
           v3sigma2lapl, v3sigma2tau,
           v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
           v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)

        * kxc for unrestricted case:
          | v3rho3[:,4]         = (u_u_u, u_u_d, u_d_d, d_d_d)
          | v3rho2sigma[:,9]    = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
          | v3rhosigma2[:,12]   = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
          | v3sigma3[:,10]      = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)
          | v3rho2lapl[:,6]
          | v3rho2tau[:,6]      = (u_u_u, u_u_d, u_d_u, u_d_d, d_d_u, d_d_d)
          | v3rhosigmalapl[:,12]
          | v3rhosigmatau[:,12] = (u_uu_u, u_uu_d, u_ud_u, u_ud_d, u_dd_u, u_dd_d,
                                   d_uu_u, d_uu_d, d_ud_u, d_ud_d, d_dd_u, d_dd_d)
          | v3rholapl2[:,6]
          | v3rholapltau[:,8]
          | v3rhotau2[:,6]      = (u_u_u, u_u_d, u_d_d, d_u_u, d_u_d, d_d_d)
          | v3sigma2lapl[:,12]
          | v3sigma2tau[:,12]   = (uu_uu_u, uu_uu_d, uu_ud_u, uu_ud_d, uu_dd_u, uu_dd_d,
                                   ud_ud_u, ud_ud_d, ud_dd_u, ud_dd_d, dd_dd_u, dd_dd_d)
          | v3sigmalapl2[:,9]
          | v3sigmalapltau[:,12]
          | v3sigmatau2[:,9]    = (uu_u_u, uu_u_d, uu_d_d, ud_u_u, ud_u_d, ud_d_d, dd_u_u, dd_u_d, dd_d_d)
          | v3lapl3[:,4]
          | v3lapl2tau[:,6]
          | v3lapltau2[:,6]
          | v3tau3[:,4]         = (u_u_u, u_u_d, u_d_d, d_d_d)

        see also libxc_itrf.c
    '''  # noqa: E501
    warnings.warn(XC_CODE_WARNING)
    xc_info = _to_xc_info(xc_code, spin)
    return _eval_xc_(xc_info, rho, spin, relativity, deriv, omega, verbose)

def _eval_xc_(xc_info, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    outbuf = _eval_xc(xc_info, rho, spin, deriv, omega)
    exc = outbuf[0]
    vxc = fxc = kxc = None
    xctype = _xc_type(xc_info[0])
    if xctype == 'LDA' and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1]]
        if deriv > 1:
            fxc = [outbuf[2]]
        if deriv > 2:
            kxc = [outbuf[3]]
    elif xctype == 'GGA' and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1], outbuf[2]]
        if deriv > 1:
            fxc = [outbuf[3], outbuf[4], outbuf[5]]
        if deriv > 2:
            kxc = [outbuf[6], outbuf[7], outbuf[8], outbuf[9]]
    elif xctype == 'LDA' and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T]
        if deriv > 1:
            fxc = [outbuf[3:6].T]
        if deriv > 2:
            kxc = [outbuf[6:10].T]
    elif xctype == 'GGA' and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T, outbuf[3:6].T]
        if deriv > 1:
            fxc = [outbuf[6:9].T, outbuf[9:15].T, outbuf[15:21].T]
        if deriv > 2:
            kxc = [outbuf[21:25].T, outbuf[25:34].T, outbuf[34:46].T, outbuf[46:56].T]
    elif xctype == 'MGGA' and spin == 0:
        if deriv > 0:
            vxc = [outbuf[1], outbuf[2], None, outbuf[3]]
        if deriv > 1:
            fxc = [
                # v2rho2, v2rhosigma, v2sigma2,
                outbuf[4], outbuf[5], outbuf[6],
                # v2lapl2, v2tau2,
                None, outbuf[9],
                # v2rholapl, v2rhotau,
                None, outbuf[7],
                # v2lapltau, v2sigmalapl, v2sigmatau,
                None, None, outbuf[8]]
        if deriv > 2:
            # v3lapltau2 might not be strictly 0
            # outbuf[18] = 0
            kxc = [
                # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                outbuf[10], outbuf[11], outbuf[12], outbuf[13],
                # v3rho2lapl, v3rho2tau,
                None, outbuf[14],
                # v3rhosigmalapl, v3rhosigmatau,
                None, outbuf[15],
                # v3rholapl2, v3rholapltau, v3rhotau2,
                None, None, outbuf[16],
                # v3sigma2lapl, v3sigma2tau,
                None, outbuf[17],
                # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
                None, None, outbuf[18],
                # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
                None, None, None, outbuf[19]]
    elif xctype == 'MGGA' and spin == 1:
        if deriv > 0:
            vxc = [outbuf[1:3].T, outbuf[3:6].T, None, outbuf[6:8].T]
        if deriv > 1:
            # v2lapltau might not be strictly 0
            # outbuf[39:43] = 0
            fxc = [
                # v2rho2, v2rhosigma, v2sigma2,
                outbuf[8:11].T, outbuf[11:17].T, outbuf[17:23].T,
                # v2lapl2, v2tau2,
                None, outbuf[33:36].T,
                # v2rholapl, v2rhotau,
                None, outbuf[23:27].T,
                # v2lapltau, v2sigmalapl, v2sigmatau,
                None, None, outbuf[27:33].T]
        if deriv > 2:
            # v3lapltau2 might not be strictly 0
            # outbuf[204:216] = 0
            kxc = [
                # v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                outbuf[36:40].T, outbuf[40:49].T, outbuf[49:61].T, outbuf[61:71].T,
                # v3rho2lapl, v3rho2tau,
                None, outbuf[71:77].T,
                # v3rhosigmalapl, v3rhosigmatau,
                None, outbuf[77:89].T,
                # v3rholapl2, v3rholapltau, v3rhotau2,
                None, None, outbuf[89:95].T,
                # v3sigma2lapl, v3sigma2tau,
                None, outbuf[95:107].T,
                # v3sigmalapl2, v3sigmalapltau, v3sigmatau2,
                None, None, outbuf[107:116].T,
                # v3lapl3, v3lapl2tau, v3lapltau2, v3tau3)
                None, None, None, outbuf[116:120].T]
    return exc, vxc, fxc, kxc

_GGA_SORT = {
    (1, 2): numpy.array([
        6, 7, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ]),
    (1, 3): numpy.array([
        21, 22, 25, 26, 27, 23, 28, 29, 30, 34, 35, 36, 37, 38, 39, 24, 31, 32,
        33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    ]),
    (1, 4): numpy.array([
        56, 57, 61, 62, 63, 58, 64, 65, 66, 73, 74, 75, 76, 77, 78, 59, 67, 68,
        69, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 60,
        70, 71, 72, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
        122, 123, 124, 125,
    ])
}
_MGGA_SORT = {
    (0, 2): numpy.array([
        0, # v2rho2
        1, # v2rhosigma
        3, # v2rhotau
        2, # v2sigma2
        4, # v2sigmatau
        5, # v2tau2
    ]) + 4,
    (0, 3): numpy.array([
        0, # v3rho3
        1, # v3rho2sigma
        4, # v3rho2tau
        2, # v3rhosigma2
        5, # v3rhosigmatau
        6, # v3rhotau2
        3, # v3sigma3
        7, # v3sigma2tau
        8, # v3sigmatau2
        9, # v3tau3
    ]) + 10,
    (0, 4): numpy.array([
        0,  # v4rho4
        1,  # v4rho3sigma
        5,  # v4rho3tau
        2,  # v4rho2sigma2
        6,  # v4rho2sigmatau
        7,  # v4rho2tau2
        3,  # v4rhosigma3
        8,  # v4rhosigma2tau
        9,  # v4rhosigmatau2
        10, # v4rhotau3
        4,  # v4sigma4
        11, # v4sigma3tau
        12, # v4sigma2tau2
        13, # v4sigmatau3
        14, # v4tau4
    ]) + 20,
    (1, 2): numpy.array([
        8, 9, 11, 12, 13, 23, 24, 10, 14, 15, 16, 25, 26, 17, 18, 19, 27, 28,
        20, 21, 29, 30, 22, 31, 32, 33, 34, 35,
    ]),
    (1, 3): numpy.array([
        36, 37, 40, 41, 42, 71, 72, 38, 43, 44, 45, 73, 74, 49, 50, 51, 77, 78,
        52, 53, 79, 80, 54, 81, 82, 89, 90, 91, 39, 46, 47, 48, 75, 76, 55, 56,
        57, 83, 84, 58, 59, 85, 86, 60, 87, 88, 92, 93, 94, 61, 62, 63, 95, 96,
        64, 65, 97, 98, 66, 99, 100, 107, 108, 109, 67, 68, 101, 102, 69, 103,
        104, 110, 111, 112, 70, 105, 106, 113, 114, 115, 116, 117, 118, 119,
    ]),
    (1, 4): numpy.array([
        120, 121, 125, 126, 127, 190, 191, 122, 128, 129, 130, 192, 193, 137,
        138, 139, 198, 199, 140, 141, 200, 201, 142, 202, 203, 216, 217, 218,
        123, 131, 132, 133, 194, 195, 143, 144, 145, 204, 205, 146, 147, 206,
        207, 148, 208, 209, 219, 220, 221, 155, 156, 157, 225, 226, 158, 159,
        227, 228, 160, 229, 230, 249, 250, 251, 161, 162, 231, 232, 163, 233,
        234, 252, 253, 254, 164, 235, 236, 255, 256, 257, 267, 268, 269, 270,
        124, 134, 135, 136, 196, 197, 149, 150, 151, 210, 211, 152, 153, 212,
        213, 154, 214, 215, 222, 223, 224, 165, 166, 167, 237, 238, 168, 169,
        239, 240, 170, 241, 242, 258, 259, 260, 171, 172, 243, 244, 173, 245,
        246, 261, 262, 263, 174, 247, 248, 264, 265, 266, 271, 272, 273, 274,
        175, 176, 177, 275, 276, 178, 179, 277, 278, 180, 279, 280, 295, 296,
        297, 181, 182, 281, 282, 183, 283, 284, 298, 299, 300, 184, 285, 286,
        301, 302, 303, 313, 314, 315, 316, 185, 186, 287, 288, 187, 289, 290,
        304, 305, 306, 188, 291, 292, 307, 308, 309, 317, 318, 319, 320, 189,
        293, 294, 310, 311, 312, 321, 322, 323, 324, 325, 326, 327, 328, 329,
    ])
}

def eval_xc1(xc_code, rho, spin=0, deriv=1, omega=None):
    '''Similar to eval_xc.
    Returns an array with the order of derivatives following xcfun convention.
    '''
    warnings.warn(XC_CODE_WARNING)
    xc_info = _to_xc_info(xc_code, spin)
    return _eval_xc1(xc_info, rho, spin, deriv, omega)

def _eval_xc1(xc_info, rho, spin=0, deriv=1, omega=None):
    out = _eval_xc(xc_info, rho, spin, deriv=deriv, omega=omega)
    xctype = _xc_type(xc_info[0])
    if deriv <= 1:
        return out
    elif xctype == 'LDA' or xctype == 'HF':
        return out
    elif xctype == 'GGA':
        if spin == 0:
            return out
        else:
            idx = [numpy.arange(6)] # up to deriv=1
            for i in range(2, deriv+1):
                idx.append(_GGA_SORT[(spin, i)])
    else: # MGGA
        if spin == 0:
            idx = [numpy.arange(4)] # up to deriv=1
        else:
            idx = [numpy.arange(8)] # up to deriv=1
        for i in range(2, deriv+1):
            idx.append(_MGGA_SORT[(spin, i)])
    return out[numpy.hstack(idx)]

def _eval_xc(xc_info, rho, spin=0, deriv=1, omega=None):
    xc_objs, xc_arr, hyb, fn_facs = xc_info
    assert deriv <= _max_deriv_order(xc_objs)
    xctype = _xc_type(xc_objs)
    assert xctype in ('HF', 'LDA', 'GGA', 'MGGA')

    rho = numpy.asarray(rho, order='C', dtype=numpy.double)
    if xctype == 'MGGA' and rho.shape[-2] == 6:
        rho = numpy.asarray(rho[...,[0,1,2,3,5],:], order='C')

    if omega is not None:
        hyb = hyb[:2] + (float(omega),)

    fn_ids = [x[0] for x in fn_facs]
    facs   = [x[1] for x in fn_facs]
    if hyb[2] != 0:
        # Current implementation does not support different omegas for
        # different RSH functionals if there are multiple RSHs
        omega = [hyb[2]] * len(facs)
    else:
        omega = [0] * len(facs)
    fn_ids_set = set(fn_ids)
    if fn_ids_set.intersection(PROBLEMATIC_XC):
        problem_xc = [PROBLEMATIC_XC[k]
                      for k in fn_ids_set.intersection(PROBLEMATIC_XC)]
        warnings.warn('Libxc functionals %s may have discrepancy to xcfun '
                      'library.\n' % problem_xc)

    if _needs_laplacian(xc_objs):
        raise NotImplementedError('laplacian in meta-GGA method')

    nvar, xlen = xc_deriv._XC_NVAR[xctype, spin]
    ngrids = rho.shape[-1]
    rho = rho.reshape(spin+1,nvar,ngrids)
    outlen = lib.comb(xlen+deriv, deriv)
    out = numpy.zeros((outlen,ngrids))
    n = len(fn_ids)
    if n > 0:
        density_threshold = 0
        _lib.LIBXC_eval_xc(n,
                           xc_arr,
                           facs,
                           omega,
                           spin, deriv,
                           nvar, ngrids,
                           outlen,
                           _ffi.cast(_DOUBLE_PTR_TYPE, rho.ctypes.data),
                           _ffi.cast(_DOUBLE_PTR_TYPE, out.ctypes.data),
                           density_threshold)
    return out

def eval_xc_eff(xc_code, rho, deriv=1, omega=None):
    r'''Returns the derivative tensor against the density parameters

    [density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a]

    or spin-polarized density parameters

    [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
     [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].

    It differs from the eval_xc method in the derivatives of non-local part.
    The eval_xc method returns the XC functional derivatives to sigma
    (|\nabla \rho|^2)

    Args:
        rho: 2-dimensional or 3-dimensional array
            Total density or (spin-up, spin-down) densities (and their
            derivatives if GGA or MGGA functionals) on grids

    Kwargs:
        deriv: int
            derivative orders
        omega: float
            define the exponent in the attenuated Coulomb for RSH functional
    '''
    xc_info = _to_xc_info(xc_code, 0)
    return _eval_xc_eff(xc_info, rho, deriv, omega)

def _eval_xc_eff(xc_info, rho, deriv=1, omega=None):
    xctype = _xc_type(xc_info[0])
    rho = numpy.asarray(rho, order='C', dtype=numpy.double)
    if xctype == 'MGGA' and rho.shape[-2] == 6:
        rho = numpy.asarray(rho[...,[0,1,2,3,5],:], order='C')

    spin_polarized = rho.ndim >= 2 and rho.shape[0] == 2
    if spin_polarized:
        spin = 1
    else:
        spin = 0
    out = eval_xc1(xc_info, rho, spin, deriv, omega)
    return xc_deriv.transform_xc(rho, out, xctype, spin, deriv)

def define_xc_(ni, description, xctype='LDA', hyb=0, rsh=(0,0,0), spin=0):
    '''Define XC functional.  See also :func:`eval_xc` for the rules of input description.

    Args:
        ni : an instance of :class:`NumInt`

        description : str
            A string to describe the linear combination of different XC functionals.
            The X and C functional are separated by comma like '.8*LDA+.2*B86,VWN'.
            If "HF" was appeared in the string, it stands for the exact exchange.

    Kwargs:
        xctype : str
            'LDA' or 'GGA' or 'MGGA'
        hyb : float
            hybrid functional coefficient
        rsh : a list of three floats
            coefficients (omega, alpha, beta) for range-separated hybrid functional.
            omega is the exponent factor in attenuated Coulomb operator e^{-omega r_{12}}/r_{12}
            alpha is the coefficient for long-range part, hybrid coefficient
            can be obtained by alpha + beta

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> mf = dft.RKS(mol)
    >>> define_xc_(mf._numint, '.2*HF + .08*LDA + .72*B88, .81*LYP + .19*VWN')
    >>> mf.kernel()
    -76.3783361189611
    >>> define_xc_(mf._numint, 'LDA*.08 + .72*B88 + .2*HF, .81*LYP + .19*VWN')
    >>> mf.kernel()
    -76.3783361189611
    >>> def eval_xc(xc_code, rho, *args, **kwargs):
    ...     exc = 0.01 * rho**2
    ...     vrho = 0.01 * 3 * rho**2
    ...     vxc = (vrho, None, None, None)
    ...     fxc = None  # 2nd order functional derivative
    ...     kxc = None  # 3rd order functional derivative
    ...     return exc, vxc, fxc, kxc
    >>> define_xc_(mf._numint, eval_xc, xctype='LDA')
    >>> mf.kernel()
    48.8525211046668
    '''
    if isinstance(description, str):
        func = XCFunctional(description, spin)
        ni._func = func
        ni.eval_xc = func.eval_xc_
        ni.hybrid_coeff = func.hybrid_coeff_
        ni.rsh_coeff = func.rsh_coeff_
        ni._xc_type = func.xc_type_

    elif callable(description):
        ni.eval_xc = description
        ni.hybrid_coeff = lambda *args, **kwargs: hyb
        ni.rsh_coeff = lambda *args, **kwargs: rsh
        ni._xc_type = lambda *args: xctype
    else:
        raise ValueError('Unknown description %s' % description)
    return ni

def define_xc(ni, description, xctype='LDA', hyb=0, rsh=(0,0,0)):
    return define_xc_(ni.copy(), description, xctype, hyb, rsh)
define_xc.__doc__ = define_xc_.__doc__

class XCFunctional:
    '''
    A high-level API for constructing LibXC exchange-correlation functionals

    Attributes for XCFunctional:
        xc_code : str
            A string to describe the XC functional. See `parse_xc` for details.
        spin : int
            spin polarized if spin > 0
        xc_objs : list of cffi objects representing the `xc_func_type` struct
            of each functional components
        xc_info : tuple with the following four components:
            1. xc_objs: a list containing all the cffi objects representing the
               `xc_func_type` struct of each functional components
            2. xc_arr: a cffi array of `xc_func_type` struct
               of each functional components
            3. hyb: hybrid coefficients as produced by `parse_xc()`
            4. fn_facs: functional factors as produced by `parse_xc()`
        hyb : hybrid coefficients as produced by `parse_xc()`
        fn_facs : functional factors as produced by `parse_xc()`
        obj_by_id : dict
            LibXC functional IDs are used as the key, and the
            corresponding cffi object of the `xc_func_type` structs
            are used as the value.
            This dict is useful for users who wish to perform low-level
            operations on the underlying LibXC data structure.

    '''
    def __init__(self, xc_code, spin=0):
        self.xc_code = xc_code
        self.spin = spin
        self.xc_info = _to_xc_info(xc_code, spin)
        self.xc_objs, self.xc_arr, self.hyb, self.fn_facs = self.xc_info
        self.obj_by_id = {xid: func for (xid, fac), func in zip(self.fn_facs, self.xc_objs)}

    def eval_xc(self, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        return _eval_xc_(self.xc_info, rho, spin, relativity, deriv, omega, verbose)

    def eval_xc_(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        '''eval_xc with a dummy `xc_code` parameter for backward compatibility'''
        return _eval_xc_(self.xc_info, rho, spin, relativity, deriv, omega, verbose)

    def rsh_coeff(self):
        return _rsh_coeff(self.xc_info)

    def rsh_coeff_(self, *args, **kwargs):
        return _rsh_coeff(self.xc_info)

    def hybrid_coeff(self):
        return _hybrid_coeff(self.xc_info, self.spin)

    def hybrid_coeff_(self, *args, **kwargs):
        return _hybrid_coeff(self.xc_info, self.spin)

    def xc_type(self):
        return _xc_type(self.xc_objs)

    def xc_type_(self, *args, **kwargs):
        return _xc_type(self.xc_objs)

    # TODO: Implement all other APIs

    def set_ext_params(self, fn_id, parameter):
        '''Set external parameters of the LibXC functional component
        using the `xc_func_set_ext_params` API of LibXC.

    Args:
        fn_id : int
            The LibXC functional ID
        parameter : ndarray or list of float
            The parameters to be set
        '''
        func = self.obj_by_id[fn_id]
        parameter = numpy.asarray(parameter, dtype=numpy.double)
        n = _lib.xc_func_info_get_n_ext_params(func.info)
        assert n == parameter.size, \
                f"""Unexpected size of external parameters for functional {fn_id}.
Expected {n} but {parameter.size} provided."""
        parameter = _ffi.cast(_DOUBLE_PTR_TYPE, parameter.ctypes.data)
        _lib.xc_func_set_ext_params(func, parameter)

