#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
Methods to: perform bitwise operations on dataframes
"""
from functools import reduce
import operator
import pandas as pd
import numba

#@numba.njit
def selectbiton(array_cand_type, mask):
    return [((cand_type & mask) == mask) for cand_type in array_cand_type]

#@numba.njit
def selectbitoff(array_cand_type, mask):
    return [((cand_type & mask) == 0) for cand_type in array_cand_type]

def tag_bit_df(dfin, namebitmap, activatedbit):
    bitson = activatedbit[0]
    bitsoff = activatedbit[1]
    array_cand_type = dfin.loc[:, namebitmap].values.astype("int")
    res_on = pd.Series([True]*len(array_cand_type))
    res_off = pd.Series([True]*len(array_cand_type))
    res = pd.Series()

    if bitson:
        mask = reduce(operator.or_, ((1 << bit) for bit in bitson), 0)
        bitmapon = selectbiton(array_cand_type, mask)
        res_on = pd.Series(bitmapon)
    if bitsoff:
        mask = reduce(operator.or_, ((1 << bit) for bit in bitsoff), 0)
        bitmapoff = selectbitoff(array_cand_type, mask)
        res_off = pd.Series(bitmapoff)
    res = res_on & res_off
    return res

def filter_bit_df(dfin, namebitmap, activatedbit):
    res = tag_bit_df(dfin, namebitmap, activatedbit)
    df_sel = dfin[res.values]
    return df_sel
