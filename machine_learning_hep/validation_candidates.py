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
Script containing all helper functions related to plotting with ROOT

Script also contains the "class Errors", used for systematic uncertainties (to
replace AliHFSystErr from AliPhysics).
"""
# pylint: disable=too-many-lines
# pylint: disable=import-error, no-name-in-module
from machine_learning_hep.bitwise import filter_bit_df
from utilities_plot import buildarray, buildbinning, makefill1dhist, makefill2dhist


def fill_validation_candidates(dfevt, dfevtevtsel, df_reco):
    """
    Create histograms for the validation on the event level as a function of the multiplicity
    """
    _ = len(df_reco)

    # Binning definition
    binning_nsigma = buildbinning(2000, -100, 100)
    binning_pt = buildbinning(100, 0, 100)

    # Make and fill histograms
    hlist = []
    df_src = None

    def make_and_fill(binx, namex, biny=None, namey=None, tag=""):
        """
        Makes histogram and fills them based on their axis titles
        """
        if namey:
            h_name = f"h_{namex}_vs_{namey}{tag}"
            h_tit = f" ; {namex} ; {namey}"
            hlist.append(makefill2dhist(df_src, h_name, binx, biny, namex, namey))
            hlist[-1].SetTitle(h_tit)
        else:
            h_name = f"h_{namex}{tag}"
            h_tit = f" ; {namex} ; Entries"
            hlist.append(makefill1dhist(df_src, h_name, h_tit, binx, namex))

    df_src = df_reco

    make_and_fill(binning_pt, "pt_cand", binning_nsigma, "nsigTOF_Pi_0")
    make_and_fill(binning_pt, "pt_cand", binning_nsigma, "nsigTOF_Pi_1")

    return hlist
