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

from machine_learning_hep.utilities_plot import buildbinning
from machine_learning_hep.validation.validation import ValidationCollection


def fill_validation_candidates(df_reco, tag=""):
    """
    Create histograms for the validation on the event level as a function of the multiplicity
    """
    _ = len(df_reco)

    # Binning definition
    binning_nsigma = buildbinning(1, -1000, -998)
    binning_nsigma += buildbinning(2000, -100, 100)
    binning_pt = buildbinning(100, 0, 100)
    binning_eta = buildbinning(100, -1, 1)
    binning_phi = buildbinning(100, 0, 7)

    # Make and fill histograms
    val = ValidationCollection(df_reco, tag=tag)
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

    return val
