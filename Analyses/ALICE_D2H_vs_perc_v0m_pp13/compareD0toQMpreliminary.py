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
main script for doing final stage analysis
"""
import os
from math import sqrt
from shutil import copyfile
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend, TLatex
from ROOT import gROOT, kRed, kGreen, kBlack, kBlue, kOrange, kViolet, kAzure
from ROOT import TStyle, gPad
from machine_learning_hep.utilities_plot import plot_histograms, load_root_style

def compareD0toQMpreliminary():
    filembqm = TFile.Open("../ALICE_D2H_vs_mult_pp13/data/PreliminaryQM19/finalcrossD0ppMBvspt_ntrklmult0.root", "READ")
    fileour = TFile.Open("/data/DerivedResults/D0kAnywithJets/vAN-20200304_ROOT6-1/pp_data/resultsMBvspt_perc_v0m/finalcrossD0ppMBvspt_perc_v0mmult0.root", "READ")
    hsigmacorrqm = filembqm.Get("histoSigmaCorr")
    hsigmacorrour = fileour.Get("histoSigmaCorr")
    hsigmacorrour.Divide(hsigmacorrqm)

    c = TCanvas("c", "c", 500, 500)
    c.cd()
    hsigmacorrour.Draw()
    c.SaveAs("canvas.pdf")

compareD0toQMpreliminary()
