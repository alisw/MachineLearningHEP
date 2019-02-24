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
import numba
import pandas as pd

@numba.njit
def selectbiton(array_cand_type, bits):
    is_selected = []
    for cand_type  in array_cand_type:
        iscandsel = 1
        for bit in bits:
            if not (cand_type >> bit) & 0x1:
                iscandsel = 0
                break
        if iscandsel == 1:
            is_selected.append(True)
        else:
            is_selected.append(False)
    return is_selected

def selectbitoff(array_cand_type, bits):
    is_selected = []
    for cand_type  in array_cand_type:
        iscandsel = 1
        for bit in bits:
            if (cand_type >> bit) & 0x1:
                iscandsel = 0
                break
        if iscandsel == 1:
            is_selected.append(True)
        else:
            is_selected.append(False)
    return is_selected


def filter_bit_df(dfin, namebitmap, activatedbit):
    bitson = activatedbit[0]
    bitsoff = activatedbit[1]
    array_cand_type = dfin.loc[:, namebitmap].values
    res_on = pd.Series([True]*len(array_cand_type))
    res_off = pd.Series([True]*len(array_cand_type))
    res = pd.Series()

    if bitson:
        bitmapon = selectbiton(array_cand_type, bitson)
        res_on = pd.Series(bitmapon)
    if bitsoff:
        bitmapoff = selectbitoff(array_cand_type, bitsoff)
        res_off = pd.Series(bitmapoff)
    res = res_on & res_off
    df_sel = dfin[res.values]
    return df_sel
