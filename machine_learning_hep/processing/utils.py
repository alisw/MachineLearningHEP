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
utilities for processing
"""

import sys


def make_mass_histo_suffix(bin1_name, bin1_low, bin1_up,
                           bin2_name=None, bin2_low=None, bin2_up=None,
                           ml_cut=None):
    """Make common suffix for mass histograms

    Args:
        bin1_name: str
        bin1_low: float
        bin1_up: float
        bin2_name: str (optional)
        bin2_low: float (optional)
        bin2_up: float (optional)
        ml_cut: float (optional)
    Returns:
        str
            suffix to be used for mass histograms
    """
    suffix = f"{bin1_name}{bin1_low}_{bin1_up}"
    if ml_cut is not None:
        suffix = f"{suffix}_{ml_cut:.2f}"
    if bin2_name is not None and bin2_low is not None and bin2_up is not None:
        suffix = f"{suffix}{bin2_name}_{bin2_low:.2f}_{bin2_up:.2f}"
    return suffix


class ProcesserHelper: # pylint: disable=too-few-public-methods
    """Helper class for common database operations

    Helper class useful when processing and analysing to derive
    names or generic strings (like save suffix) from the database

    """
    def __init__(self, database, case, ana_type, period=None):
        """Init

        Args:
            database: dict
                configuration database
                (everything below top node aka case)
            case: str
                the particle case
            ana_type: str
                name of analysis node sub-section to be used
            period: int (optional)
                data taking period number
                (None indicates all periods merged)
        """
        self.database = database
        self.case = case
        self. ana_type = ana_type
        self.period = period
        self.analysis = database["analysis"][ana_type]


    def make_mass_histo_suffix(self, ibin1, ibin2=None):
        """Derive mass histogram suffix from analysis bins

        Given one (or two) differential bin number(s), derive mass histogram
        suffix using public make_mass_histo_suffix

        Args:
            ibin1: int
                index of first differential bin
            ibin2: int (optional)
                index of second differential bin
        Returns:
        str
            suffix to be used for mass histograms
        """
        bin1_match_id = self.analysis.get("binning_matching", None)
        ml_cut = None
        if bin1_match_id:
            ml_cut = self.database["mlapplication"]["probcutoptimal"][bin1_match_id[ibin1]]
        bin1_name = self.database["var_binning"]
        bin1_low = self.analysis["sel_an_binmin"][ibin1]
        bin1_up = self.analysis["sel_an_binmax"][ibin1]
        bin2_name = None
        bin2_low = None
        bin2_up = None
        if ibin2 is not None:
            bin2_name = self.analysis.get("var_binning2", None)
            bin2_low = self.analysis.get("sel_binmin2", None)
            bin2_up = self.analysis.get("sel_binmax2", None)
            if not bin2_name or not bin2_low or not bin2_up:
                print(f"Try to derive suffix for ibin2 {ibin2} but either \"var_binning2\", " \
                      f"\"sel_binmin2\" or \"sel_binmax2\" is missing")
                sys.exit(1)
            bin2_low = bin2_low[ibin2]
            bin2_up = bin2_up[ibin2]

        return make_mass_histo_suffix(bin1_name, bin1_low, bin1_up,
                                      bin2_name, bin2_low, bin2_up,
                                      ml_cut)
