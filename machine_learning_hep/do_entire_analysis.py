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
main script for doing data processing, machine learning and analysis
"""

# replacements for root_numpy functionality
# TODO: move to separate file
def fill_hist(hist, array, weights):
    assert array.ndim() == 1 and weights.ndim() == 1, 'fill_hist handles 1d histos only'
    hist.FillN(array.len(), array, weights)

def hist2array(hist):
    assert hist.GetDimension() == 1
    return [hist.GetBinContent(x) for x in range(hist.GetNbinsX())]

def array2hist(array, hist):
    assert array.ndim() == 1 and hist.GetDimension() == array.ndim()
    for i, x in enumerate(array):
        hist.SetBinContent(i + 1, x)

from machine_learning_hep.steer_analysis import main

main()
