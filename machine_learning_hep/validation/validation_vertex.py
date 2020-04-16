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
Script containing validation histograms on the event granularity for the vertex monitoring
"""

# pylint: disable=too-many-lines
# pylint: disable=import-error, no-name-in-module
# from machine_learning_hep.bitwise import filter_bit_df
from utilities_plot import buildbinning
from machine_learning_hep.validation.validation import ValidationCollection


def fill_validation_vertex(dfevt, dfevtevtsel, df_reco):
    """
    Create histograms for the validation on the event level as a function of the multiplicity
    """
    _ = len(df_reco)
    __ = len(dfevtevtsel)

    # Binning definition
    # binning_xyvtx = buildbinning(100, -1.0, 1)
    binning_zvtx = buildbinning(100, -15.0, 15)

    # Make and fill histograms
    val = ValidationCollection(dfevt[dfevt.is_ev_rej_INT7 == 0])
    # val.make_and_fill(binning_xyvtx, "x_vtx_reco")
    # val.make_and_fill(binning_xyvtx, "y_vtx_reco")
    val.make_and_fill(binning_zvtx, "z_vtx_reco")

    return val
