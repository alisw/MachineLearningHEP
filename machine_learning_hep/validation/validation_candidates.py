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
Script containing validation histograms on the candidate granularity
"""

from machine_learning_hep.utilities_plot import buildbinning, buildarray
from machine_learning_hep.validation.validation import ValidationCollection


def fill_validation_candidates(df_reco, tag=""):
    """
    Create histograms for the validation on the event level as a function of the multiplicity
    """
    _ = len(df_reco)

    # Binning definition
    binning_nsigma = buildbinning(1, -1000, -998)
    binning_nsigma += buildbinning(2000, -100, 100)
    binning_pt = buildbinning(400, 0, 100)
    binning_pt = buildarray([1, 2, 4, 6, 8, 12, 24])
    binning_eta = buildbinning(100, -1, 1)
    binning_phi = buildbinning(100, 0, 7)
    binning_inv_mass = buildbinning(100, 2, 2.5)
    binning_v0m_perc = buildbinning(100, 0, 1)
    binning_v0m_perc += buildbinning(89, 1.1, 10)
    binning_v0m_perc += buildbinning(89, 11, 100)
    binning_ntrklt = buildbinning(200, -0.5, 199.5)

    # Make and fill histograms
    val = ValidationCollection(df_reco, tag=tag)

    # PID information
    for i in ["TPC", "TOF"]:
        for j in ["Pi", "K", "Pr"]:
            for k in ["0", "1"]:
                yaxis = [binning_nsigma, f"nsig{i}_{j}_{k}"]
                val.make_and_fill(binning_pt, "p_prong0", *yaxis)
                val.make_and_fill(binning_pt, "p_prong1", *yaxis)
                val.make_and_fill(binning_pt, "pt_prong0", *yaxis)
                val.make_and_fill(binning_pt, "pt_prong1", *yaxis)
                val.make_and_fill(binning_pt, "pt_cand", *yaxis)
                val.make_and_fill(binning_eta, "eta_cand", *yaxis)
                val.make_and_fill(binning_phi, "phi_cand", *yaxis)

    # Invariant mass
    val.make_and_fill(binning_inv_mass, "inv_mass",
                      binning_v0m_perc, "perc_v0m")
    val.make_and_fill(binning_inv_mass, "inv_mass",
                      binning_ntrklt, "n_tracklets_corr")
    for i, j in enumerate(binning_pt[0:-1]):
        # Defining pT interval
        lower_pt = j
        upper_pt = binning_pt[i+1]
        pt_interval = "_pt_cand_{:.1f}-{:.1f}".format(lower_pt, upper_pt)
        # Cutting the DF in the pT interval
        df_ptcut = df_reco[df_reco.pt_cand > lower_pt]
        df_ptcut = df_ptcut[df_ptcut.pt_cand < upper_pt]
        # Resetting validation collection to use the pT cut DF
        val.reset_input(df_ptcut, tag=tag + pt_interval)
        # Filling histograms with inv mass and multiplicity
        val.make_and_fill(binning_inv_mass, "inv_mass",
                          binning_v0m_perc, "perc_v0m")
        val.make_and_fill(binning_inv_mass, "inv_mass",
                          binning_ntrklt, "n_tracklets_corr")

    return val
