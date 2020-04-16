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
Script containing validation histograms on the event granularity
"""

# pylint: disable=too-many-lines
# pylint: disable=import-error, no-name-in-module
from machine_learning_hep.bitwise import filter_bit_df
from utilities_plot import buildarray, buildbinning
from machine_learning_hep.validation.validation import ValidationCollection


def fill_validation_multiplicity(dfevt, dfevtevtsel, df_reco):
    """
    Create histograms for the validation on the event level as a function of the multiplicity
    """
    _ = len(df_reco)

    # Binning definition
    binning_ntrklt = buildbinning(200, -0.5, 199.5)
    binning_v0m = buildbinning(1500, -0.5, 1499.5)
    binning_zvtx = buildbinning(100, -15.0, 15)
    binning_v0m_perc = [0]
    while binning_v0m_perc[-1] + 0.1 < 10:
        binning_v0m_perc.append(binning_v0m_perc[-1] + 0.1)
    while binning_v0m_perc[-1] + 1 < 100:
        binning_v0m_perc.append(binning_v0m_perc[-1] + 1)
    binning_v0m_perc = buildarray(binning_v0m_perc)

    # Make and fill histograms
    val = ValidationCollection(dfevt[dfevt.is_ev_rej_INT7 == 0])
    # val = ValidationCollection(dfevt[dfevtevtsel])
    # val = ValidationCollection(dfevt[dfevt])
    for i in "v0m v0m_eq v0m_corr v0m_eq_corr".split():
        val.make_and_fill(binning_ntrklt, "n_tracklets", binning_v0m, i)
        val.make_and_fill(binning_v0m, i, binning_v0m_perc, "perc_v0m")

    for i in "n_tracklets n_tracklets_corr n_tracklets_corr_shm".split():
        val.make_and_fill(binning_ntrklt, i, binning_v0m_perc, "perc_v0m")

    val.reset(dfevtevtsel, "")
    val.make_and_fill(binning_ntrklt, "n_tracklets", binning_ntrklt, "n_tracklets_corr")
    val.make_and_fill(binning_zvtx, "z_vtx_reco", binning_ntrklt, "n_tracklets_corr")
    val.make_and_fill(binning_zvtx, "z_vtx_reco", binning_ntrklt, "n_tracklets")

    val.make_and_fill(binning_ntrklt, "n_tracklets_corr")
    val.make_and_fill(binning_ntrklt, "n_tracklets_corr_shm")

    val.reset(filter_bit_df(dfevt, "is_ev_rej", [[4], []]), "pileup")
    val.make_and_fill(binning_ntrklt, "n_tracklets_corr")
    # val.reset(dfevtevtsel.query("is_ev_sel_shm == 1"), "spd")
    # val.make_and_fill(binning_ntrklt, "n_tracklets_corr")

    val.reset(df_reco, "")
    val.make_and_fill(
        binning_ntrklt, "n_tracklets_corr_sub", binning_ntrklt, "n_tracklets_corr"
    )
    return val
