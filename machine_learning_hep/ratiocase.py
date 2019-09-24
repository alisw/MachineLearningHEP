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
from ROOT import gROOT, kRed, kGreen, kBlack, kBlue
from ROOT import TStyle

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
def ratiocase(case_num, case_den, arraytype, isv0m=False):

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)

    with open("data/database_ml_parameters_%s.yml" % case_num, 'r') as param_config_num:
        data_param_num = yaml.load(param_config_num, Loader=yaml.FullLoader)

    with open("data/database_ml_parameters_%s.yml" % case_den, 'r') as param_config_den:
        data_param_den = yaml.load(param_config_den, Loader=yaml.FullLoader)

    folder_num_allperiods = \
        data_param_num[case_num]["analysis"][arraytype[0]]["data"]["resultsallp"]
    folder_den_allperiods = \
        data_param_den[case_den]["analysis"][arraytype[0]]["data"]["resultsallp"]
    if isv0m is False:
        folder_num_triggered = \
            data_param_num[case_num]["analysis"][arraytype[1]]["data"]["results"][3]
        folder_den_triggered = \
            data_param_den[case_den]["analysis"][arraytype[1]]["data"]["results"][3]
        partMB = [0, 1, 2]
        partHM = [3]
    else:
        folder_num_triggered = \
            data_param_num[case_num]["analysis"][arraytype[1]]["data"]["resultsallp"]
        folder_den_triggered = \
            data_param_den[case_den]["analysis"][arraytype[1]]["data"]["resultsallp"]
        partMB = [0, 1, 2]
        partHM = [3]

    binsmin_num = data_param_num[case_num]["analysis"][arraytype[0]]["sel_binmin2"]
    binsmax_num = data_param_num[case_num]["analysis"][arraytype[0]]["sel_binmax2"]
    name_num = data_param_num[case_num]["analysis"][arraytype[0]]["latexnamemeson"]
    name_den = data_param_den[case_den]["analysis"][arraytype[0]]["latexnamemeson"]
    latexbin2var = data_param_num[case_num]["analysis"][arraytype[0]]["latexbin2var"]
    plotbin = data_param_num[case_num]["analysis"][arraytype[0]]["plotbin"]

    br_num = data_param_num[case_num]["ml"]["opt"]["BR"]
    br_den = data_param_den[case_den]["ml"]["opt"]["BR"]

    ccross = TCanvas('cRatioCross', 'The Fit Canvas', 100, 600)
    ccross = TCanvas('cRatioCross', 'The Fit Canvas')
    ccross.SetCanvasSize(1500, 1500)
    ccross.SetWindowSize(500, 500)

    legyield = TLegend(.3, .65, .7, .85)
    legyield.SetBorderSize(0)
    legyield.SetFillColor(0)
    legyield.SetFillStyle(0)
    legyield.SetTextFont(42)
    legyield.SetTextSize(0.035)

    file_num_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                     (folder_num_allperiods, case_num, arraytype[0]))
    file_den_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                     (folder_den_allperiods, case_den, arraytype[0]))
    if isv0m is False:
        file_num_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                        (folder_num_triggered, case_num, arraytype[1]))
        file_den_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                        (folder_den_triggered, case_den, arraytype[1]))
    else:
        file_num_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                        (folder_num_triggered, case_num, arraytype[1]))
        file_den_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                        (folder_den_triggered, case_den, arraytype[1]))
        print("%s/finalcross%s%smulttot.root" % (folder_den_triggered, case_den, arraytype[1]))
        print(file_den_triggered)

    colors = [kBlack, kRed, kGreen+2, kBlue]
    for imult in partMB:
        print(imult)
        hratio = file_num_allperiods.Get("histoSigmaCorr%d" % (imult))
        hratio.Scale(1./br_num)
        hcross_den = file_den_allperiods.Get("histoSigmaCorr%d" % (imult))
        hcross_den.Scale(1./br_den)
        hratio.Divide(hcross_den)
        hratio.SetLineColor(colors[imult])
        hratio.SetMarkerColor(colors[imult])
        hratio.GetXaxis().SetTitle("#it{p}_{T} (GeV)")
        hratio.GetYaxis().SetTitle("%s / %s" % (name_num, name_den))
        hratio.GetYaxis().SetRangeUser(0., 1.)
        if plotbin[imult] == 1:
            hratio.Draw("same")
            legyieldstring = "%.1f < %s < %.1f (from MB)" % \
                    (binsmin_num[imult], latexbin2var, binsmax_num[imult])
            legyield.AddEntry(hratio, legyieldstring, "LEP")

    for imult in partHM:
        print(imult)
        hratioHM = file_num_triggered.Get("histoSigmaCorr%d" % (imult))
        hratioHM.Scale(1./br_num)
        hcrossHM_den = file_den_triggered.Get("histoSigmaCorr%d" % (imult))
        hcrossHM_den.Scale(1./br_den)
        hratioHM.Divide(hcrossHM_den)
        hratioHM.SetLineColor(colors[imult])
        hratioHM.SetMarkerColor(colors[imult])
        if plotbin[imult] == 1:
            hratioHM.Draw("same")
            if isv0m is False:
                legyieldstring = "%.1f < %s < %.1f (from HMSPD)" % \
                        (binsmin_num[imult], latexbin2var, binsmax_num[imult])
            else:
                legyieldstring = "%.1f < %s < %.1f (from HMV0m)" % \
                        (binsmin_num[imult], latexbin2var, binsmax_num[imult])
            legyield.AddEntry(hratioHM, legyieldstring, "LEP")
    legyield.Draw()

    ccross.SaveAs("ComparisonRatios_%s%s_%scombined%s.eps" % \
                  (case_num, case_den, arraytype[0], arraytype[1]))

ratiocase("LcpK0spp", "D0pp", ["MBvspt_ntrkl", "SPDvspt"])
ratiocase("LcpK0spp", "D0pp", ["MBvspt_v0m", "V0mvspt"], True)
