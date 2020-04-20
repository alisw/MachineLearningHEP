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

from machine_learning_hep.bitwise import filter_bit_df
from machine_learning_hep.validation.validation import ValidationCollection
from machine_learning_hep.utilities_plot import buildbinning


def fill_validation_multiplicity(dfevt, dfevtevtsel, df_reco):
    """
    Create histograms for the validation on the event level as a function of the multiplicity
    """
    _ = len(df_reco)
    # Binning definition
    binning_ntrklt = buildbinning(200, -0.5, 199.5)
    binning_ntrklt_diff = buildbinning(10, -0.5, 9.5)
    binning_v0m = buildbinning(1500, -0.5, 1499.5)
    binning_zvtx = buildbinning(100, -15.0, 15)
    binning_v0m_perc = buildbinning(100, 0, 1)
    binning_v0m_perc += buildbinning(89, 1.1, 10)
    binning_v0m_perc += buildbinning(89, 11, 100)

    # Make and fill histograms
    val = ValidationCollection(dfevt[dfevt.is_ev_rej_INT7 == 0])
    # val = ValidationCollection(dfevt[dfevtevtsel])
    # val = ValidationCollection(dfevt[dfevt])
    for i in ["v0m", "v0m_eq", "v0m_corr", "v0m_eq_corr"]:
        val.make_and_fill(binning_ntrklt, "n_tracklets", binning_v0m, i)
        val.make_and_fill(binning_v0m, i, binning_v0m_perc, "perc_v0m")

    for i in ["n_tracklets", "n_tracklets_corr", "n_tracklets_corr_shm"]:
        val.make_and_fill(binning_ntrklt, i, binning_v0m_perc, "perc_v0m")
        val.make_and_fill(binning_v0m_perc, "perc_v0m", binning_ntrklt, i)

    val.reset_input(dfevtevtsel, "")
    val.make_and_fill(binning_ntrklt, "n_tracklets",
                      binning_ntrklt, "n_tracklets_corr")
    val.make_and_fill(binning_zvtx, "z_vtx_reco",
                      binning_ntrklt, "n_tracklets_corr")
    val.make_and_fill(binning_zvtx, "z_vtx_reco",
                      binning_ntrklt, "n_tracklets")

    val.make_and_fill(binning_ntrklt, "n_tracklets_corr")
    val.make_and_fill(binning_ntrklt, "n_tracklets_corr_shm")

    val.reset_input(filter_bit_df(dfevt, "is_ev_rej", [[4], []]), "pileup")
    val.make_and_fill(binning_ntrklt, "n_tracklets_corr")
    # val.reset_input(dfevtevtsel.query("is_ev_sel_shm == 1"), "spd")
    # val.make_and_fill(binning_ntrklt, "n_tracklets_corr")

    df_reco["n_tracklets_corr-n_tracklets_corr_sub"] = (
        df_reco["n_tracklets_corr"] - df_reco["n_tracklets_corr_sub"]
    )
    for i in [[df_reco, ""],
              [df_reco[df_reco.is_ev_rej_INT7 == 0], "MB"],
              [df_reco.query("is_ev_sel_shm == 1"), "HMSPD"],
              ]:
        val.reset_input(*i)
        val.make_and_fill(
            binning_ntrklt,
            "n_tracklets_corr",
            binning_ntrklt_diff,
            "n_tracklets_corr-n_tracklets_corr_sub",
        )
        val.make_and_fill(
            binning_ntrklt, "n_tracklets_corr_sub", binning_ntrklt, "n_tracklets_corr"
        )
        val.make_and_fill(
            binning_ntrklt, "n_tracklets_corr", binning_ntrklt, "n_tracklets_corr_sub"
        )

    return val
