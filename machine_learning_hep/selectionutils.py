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
utilities for fiducial acceptance and pid selections
"""

import numba
import numpy as np

@numba.njit
def selectfidacc(array_pt, array_y):
    array_is_sel = []
    for icand, pt in enumerate(array_pt):
        if pt > 5:
            if np.absolute(array_y[icand]) < 0.8:
                array_is_sel.append(True)
            else:
                array_is_sel.append(False)
        else:
            yfid = -0.2/15 * pt**2 + 1.9/15 * pt + 0.5
            if np.absolute(array_y[icand]) < yfid:
                array_is_sel.append(True)
            else:
                array_is_sel.append(False)
    return array_is_sel

# pylint: disable=too-many-arguments
@numba.njit
def selectpid_dstokkpi(array_nsigma_tpc_pi_0, array_nsigma_tpc_k_0, \
    array_nsigma_tof_pi_0, array_nsigma_tof_k_0, \
        array_nsigma_tpc_k_1, array_nsigma_tof_k_1, \
            array_nsigma_tpc_pi_2, array_nsigma_tpc_k_2, \
                array_nsigma_tof_pi_2, array_nsigma_tof_k_2, nsigmacut):

    array_is_pid_sel = []

    for icand in range(array_nsigma_tpc_pi_0):
        is_track_0_sel = array_nsigma_tpc_pi_0[icand] < nsigmacut \
            or array_nsigma_tof_pi_0[icand] < nsigmacut \
                or array_nsigma_tpc_k_0[icand] < nsigmacut \
                    or array_nsigma_tof_k_0[icand] < nsigmacut
        #second track must be a kaon
        is_track_1_sel = array_nsigma_tpc_k_1[icand] < nsigmacut \
            or array_nsigma_tof_k_1[icand] < nsigmacut
        is_track_2_sel = array_nsigma_tpc_pi_2[icand] < nsigmacut \
            or array_nsigma_tof_pi_2[icand] < nsigmacut \
                or array_nsigma_tpc_k_2[icand] < nsigmacut \
                    or array_nsigma_tof_k_2[icand] < nsigmacut
        if is_track_0_sel and is_track_1_sel and is_track_2_sel:
            array_is_pid_sel.append(True)
        else:
            array_is_pid_sel.append(False)
    return array_is_pid_sel

@numba.njit
def selectpid_dzerotokpi(array_nsigma_tpc_pi_0, array_nsigma_tpc_k_0, \
    array_nsigma_tof_pi_0, array_nsigma_tof_k_0, \
        array_nsigma_tpc_pi_1, array_nsigma_tpc_k_1, \
            array_nsigma_tof_pi_1, array_nsigma_tof_k_1, nsigmacut):

    array_is_pid_sel = []

    for icand in range(array_nsigma_tpc_pi_0):
        is_track_0_sel = array_nsigma_tpc_pi_0[icand] < nsigmacut \
            or array_nsigma_tof_pi_0[icand] < nsigmacut \
                or array_nsigma_tpc_k_0[icand] < nsigmacut \
                    or array_nsigma_tof_k_0[icand] < nsigmacut
        is_track_1_sel = array_nsigma_tpc_pi_1[icand] < nsigmacut \
            or array_nsigma_tof_pi_1[icand] < nsigmacut \
                or array_nsigma_tpc_k_1[icand] < nsigmacut \
                    or array_nsigma_tof_k_1[icand] < nsigmacut
        if is_track_0_sel and is_track_1_sel:
            array_is_pid_sel.append(True)
        else:
            array_is_pid_sel.append(False)
    return array_is_pid_sel

@numba.njit
def selectpid_lctov0bachelor(array_nsigma_tpc, array_nsigma_tof, nsigmacut):
    #nsigma for desired species (i.e. p in case of pK0s or pi in case of piL)
    array_is_pid_sel = []

    for icand in range(array_nsigma_tpc):
        is_track_sel = array_nsigma_tpc[icand] < nsigmacut or \
            array_nsigma_tof[icand] < nsigmacut
        if is_track_sel:
            array_is_pid_sel.append(True)
        else:
            array_is_pid_sel.append(False)
    return array_is_pid_sel
