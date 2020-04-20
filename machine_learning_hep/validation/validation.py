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
Script base function for validation histograms
"""

from machine_learning_hep.utilities_plot import makefill1dhist, makefill2dhist
from machine_learning_hep.logger import get_logger


class ValidationCollection:
    """
    Class for managing histograms of validation
    """

    def __init__(self, dataframe=None, tag="", verbose=False):
        self.source_dataframe = dataframe
        self.collection_tag = tag
        self.histograms = []
        self.verbose = verbose

    def reset_input(self, dataframe, tag):
        self.source_dataframe = dataframe
        self.collection_tag = tag

    def make_and_fill(self, binx, namex, biny=None, namey=None):
        """
        Makes histogram and fills them based on their axis titles
        """
        h = None
        if namey:
            # Check that column exists
            if namex not in self.source_dataframe:
                get_logger().warning(
                    "Columns %s for X axis does not exist in dataframe, skipping histogram", namex)
                return
            if namey not in self.source_dataframe:
                get_logger().warning(
                    "Columns %s for Y axis does not exist in dataframe, skipping histogram", namey)
                return
            h_name = f"hVal_{namex}_vs_{namey}{self.collection_tag}"
            h_tit = f" ; {namex} ; {namey}"
            h = makefill2dhist(self.source_dataframe, h_name,
                               binx, biny, namex, namey)
            h.SetTitle(h_tit)
        else:
            # Check that column exists
            if namex not in self.source_dataframe:
                get_logger().warning(
                    "Columns %s for X axis does not exist in dataframe, skipping histogram", namex)
                return
            h_name = f"hVal_{namex}{self.collection_tag}"
            h_tit = f" ; {namex} ; Entries"
            h = makefill1dhist(self.source_dataframe,
                               h_name, h_tit, binx, namex)
        if self.verbose:
            get_logger().info("Filling histogram %s", h.GetName())
        self.histograms.append(h)

    def write(self):
        for i in self.histograms:
            if self.verbose:
                get_logger().info("Writing histogram %s", i.GetName())
            i.Write()
