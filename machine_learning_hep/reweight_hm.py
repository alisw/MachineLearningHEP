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
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import gROOT, TFile, TCanvas, TF1, TH1F
from machine_learning_hep.utilities import plot_histograms

filespd = TFile.Open("/data/DerivedVal/mcvalspdhm_18/AnalysisResultsROOTEvtVal.root", "read")
filemb = TFile.Open("/data/DerivedVal/dataval_18/AnalysisResultsROOTEvtVal.root", "read")


histospd = filespd.Get("hbitINT7vsn_tracklets_corr")
histomb = filemb.Get("hbitINT7vsn_tracklets_corr")
hratio = histomb.Clone("hratio")
hratio.Divide(histospd)
hweight = TH1F("Weights0", "Weights0", 200, -0.5, 199.5)
c = TCanvas("c", "c", 1000, 1000)
c.cd()
hratio.GetXaxis().SetRangeUser(30,100)
func = TF1("func", "[0]+x*[1]+x*x*[2]+x*x*x*[3]", 35, 100)
hratio.Fit(func, "L", "", 35, 100)
hratio.Draw()
for ibin in range(hweight.GetNbinsX()):
    bincenter = hweight.GetBinCenter(ibin + 1)
    hweight.SetBinContent(ibin+1, func.Eval(bincenter))
    hweight.SetBinError(ibin+1, 0.)

c.SaveAs("canvasDs.pdf")
f = TFile("reweighting/prodDs_spdhm/mcweights.root", "recreate")
f.cd()
hratio.Write()
hweight.Write()
f.Close()
