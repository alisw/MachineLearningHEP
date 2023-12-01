#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
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
import numpy as np

def tag_bit_df(dfin, namebitmap, activatedbit):
    mask_on = reduce(operator.or_, ((1 << bit) for bit in activatedbit[0]), 0)
    mask_off = reduce(operator.or_, ((1 << bit) for bit in activatedbit[1]), 0)
    ar = dfin[namebitmap].values.astype['int']
    return np.logical_and(np.bitwise_and(ar, mask_on) == mask_on,
                          np.bitwise_and(ar, mask_off) == 0)

def filter_bit_df(dfin, namebitmap, activatedbit):
    return dfin[tag_bit_df(dfin, namebitmap, activatedbit)]
