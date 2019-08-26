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
main script for doing systematics
"""
#import os
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
#from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, gROOT
#from ROOT import TLegend
#from ROOT import TStyle
#from ROOT import TLatex
#from machine_learning_hep.globalfitter import fitter

# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements
class Systematics:
    species = "systematics"
    def __init__(self, datap, case, typean):

        self.case = case
        self.typean = typean
        self.p_prob_range = datap["systematics"]["probvariation"]["prob_range"]

    @staticmethod
    def loadstyle():
        gROOT.SetStyle("Plain")
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(0)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)
        gStyle.SetOptTitle(0)

    def probvariation(self):
        self.loadstyle()
        print(self.p_prob_range)
