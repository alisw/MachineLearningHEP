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
from ROOT import TStyle

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
def ratio(imult):

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)

    ccross = TCanvas('cCross', 'The Fit Canvas', 100, 600)
    fileoutcrossd0pp = TFile.Open("finalcrossD0pp.root")
    fileoutcrossdspp = TFile.Open("finalcrossDspp.root")
    fileoutcrossLcpkpipp = TFile.Open("finalcrossLcpKpipp.root")
    fileoutcrossLcpk0s = TFile.Open("finalcrossLcpK0spp.root")

    with open("data/database_ml_parameters_D0pp.yml", 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)

    nbins = len(data_param["D0pp"]["analysis"]["sel_binmax2"])
    print("nbins", nbins)
    ccross = TCanvas('cCross', 'The Fit Canvas')
    ccross.SetCanvasSize(1500, 1500)
    ccross.SetWindowSize(500, 500)
    ccross.SetLogx()
    colorparticle = [[600, 632, 880], [600, 632, 880]]
    markerstyle = [[21, 21, 21], [22, 22, 22]]
    legendtxt = [["Ds < 20 tracklets", "LcK0s < 20 tracklets", "LcpKpi < 20 tracklets"], \
                   ["Ds > 20 tracklets", "LcK0s > 20 tracklets", "LcpKpi > 20 tracklets"]]

    leg = TLegend(.5, .65, .7, .85)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    hcrossD0pp = fileoutcrossd0pp.Get("hcross%d" % (imult))
    hcrossDspp = fileoutcrossdspp.Get("hcross%d" % (imult))
    hcrossLcpK0spp = fileoutcrossLcpk0s.Get("hcross%d" % (imult))
    hcrossLcpKpipp = fileoutcrossLcpkpipp.Get("hcross%d" % (imult))
    hcrossDspp.Divide(hcrossD0pp)
    hcrossLcpK0spp.Divide(hcrossD0pp)
    hcrossLcpKpipp.Divide(hcrossD0pp)
    hcrossDspp.SetMarkerStyle(markerstyle[imult-1][0])
    hcrossLcpK0spp.SetMarkerStyle(markerstyle[imult-1][1])
    hcrossLcpKpipp.SetMarkerStyle(markerstyle[imult-1][2])
    hcrossDspp.SetMarkerColor(colorparticle[imult-1][0])
    hcrossLcpK0spp.SetMarkerColor(colorparticle[imult-1][1])
    hcrossLcpKpipp.SetMarkerColor(colorparticle[imult-1][2])
    hcrossDspp.SetLineColor(colorparticle[imult-1][0])
    hcrossLcpK0spp.SetLineColor(colorparticle[imult-1][1])
    hcrossLcpKpipp.SetLineColor(colorparticle[imult-1][2])
    hcrossDspp.SetMarkerSize(2.5)
    hcrossLcpK0spp.SetMarkerSize(2.5)
    hcrossLcpKpipp.SetMarkerSize(2.5)
    hcrossDspp.GetXaxis().SetTitle("p_{T} (GeV)")
    hcrossDspp.GetYaxis().SetTitle("Particle ratio")
    hcrossDspp.GetYaxis().SetRangeUser(0., 1.)
    hcrossDspp.Draw()
    hcrossLcpKpipp.Draw("same")
    hcrossLcpK0spp.Draw("same")
    leg.AddEntry(hcrossDspp, legendtxt[imult-1][0], "LEP")
    leg.AddEntry(hcrossLcpKpipp, legendtxt[imult-1][1], "LEP")
    leg.AddEntry(hcrossLcpK0spp, legendtxt[imult-1][2], "LEP")

    leg.Draw()
    ccross.SaveAs("ComparisonRatios%d.pdf" % imult)
ratio(1)
ratio(2)
