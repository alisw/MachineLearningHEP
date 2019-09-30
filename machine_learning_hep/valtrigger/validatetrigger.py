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
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend
from ROOT import gROOT
from ROOT import TStyle, gPad

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
def validatetrigger():

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)

    ccross = TCanvas('cCross', 'The Fit Canvas', 100, 600)
    ccross = TCanvas('cCross', 'The Fit Canvas')
    ccross.SetCanvasSize(1500, 1500)
    ccross.SetWindowSize(500, 500)
    ccross.SetLogx()

    legyield = TLegend(.3, .65, .7, .85)
    legyield.SetBorderSize(0)
    legyield.SetFillColor(0)
    legyield.SetFillStyle(0)
    legyield.SetTextFont(42)
    legyield.SetTextSize(0.035)

    files = ["resultsSPDvspt/finalcrossD0pptestSPDvsptmult3Weight.root",
             "resultsSPDvspt/finalcrossD0pptestSPDvsptmult3NoWeight.root",
             "resultsMBvspt_ntrkl/finalcrossD0pptestMBvspt_ntrklmult3.root"]

    colors = [1, 2, 4]
    legends = ["triggered weight ntracklets 60-100",
               "triggered no weight ntracklets 60-100",
               "MB ntracklets 60-100"]
    hempty = TH1F("hempty", "hempty", 100, 0, 30)
    hempty.GetYaxis().SetTitleOffset(1.2)
    hempty.GetYaxis().SetTitleFont(42)
    hempty.GetXaxis().SetTitleFont(42)
    hempty.GetYaxis().SetLabelFont(42)
    hempty.GetXaxis().SetLabelFont(42)
    hempty.GetXaxis().SetRangeUser(1,30)
    hempty.GetYaxis().SetTitle("Corrected yield (AU)")
    hempty.GetXaxis().SetTitle("p_{T} (GeV)")
    hempty.SetMinimum(1e5)
    hempty.SetMaximum(1e9)
    hempty.Draw()

    legyield = TLegend(.3, .65, .7, .85)
    legyield.SetBorderSize(0)
    legyield.SetFillColor(0)
    legyield.SetFillStyle(0)
    legyield.SetTextFont(42)
    legyield.SetTextSize(0.035)
    histolist = []
    hempty.Draw()
    gPad.SetLogy()
    myfileWeight = TFile.Open(files[0], "read")
    myfileNoWeight = TFile.Open(files[1], "read")
    myfileMB = TFile.Open(files[2], "read")

    hWeight = myfileWeight.Get("histoSigmaCorr")
    hNoWeight = myfileNoWeight.Get("histoSigmaCorr")
    hMB = myfileMB.Get("histoSigmaCorr")

    hWeight.SetLineColor(1)
    hNoWeight.SetLineColor(2)
    hMB.SetLineColor(4)

    hWeight.Draw("same")
    hNoWeight.Draw("same")
    hMB.Draw("same")

    legyield.AddEntry(hWeight, legends[0], "LEP")
    legyield.AddEntry(hNoWeight, legends[1], "LEP")
    legyield.AddEntry(hMB, legends[2], "LEP")
    legyield.Draw()
    ccross.SaveAs("ccrossSPD.eps")

validatetrigger()
