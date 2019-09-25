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
from ROOT import gStyle, TLegend, TLatex
from ROOT import gROOT, kRed, kGreen, kBlack, kBlue
from ROOT import TStyle, gPad

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
def plot_hfptspectrum_comb(case, arraytype, isv0m=False):

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)
    gStyle.SetTitleOffset(1.15, "y")
    gStyle.SetTitleFont(42, "xy")
    gStyle.SetLabelFont(42, "xy")
    gStyle.SetTitleSize(0.042, "xy")
    gStyle.SetLabelSize(0.035, "xy")
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)

    folder_MB_allperiods = data_param[case]["analysis"][arraytype[0]]["data"]["resultsallp"]
    if isv0m is False:
        folder_triggered = data_param[case]["analysis"][arraytype[1]]["data"]["results"][3]
    else:
        folder_triggered = data_param[case]["analysis"][arraytype[1]]["data"]["resultsallp"]

    binsmin = data_param[case]["analysis"][arraytype[0]]["sel_binmin2"]
    binsmax = data_param[case]["analysis"][arraytype[0]]["sel_binmax2"]
    name = data_param[case]["analysis"][arraytype[0]]["latexnamemeson"]
    latexbin2var = data_param[case]["analysis"][arraytype[0]]["latexbin2var"]
    plotbinMB = data_param[case]["analysis"][arraytype[0]]["plotbin"]
    plotbinHM = data_param[case]["analysis"][arraytype[1]]["plotbin"]
    br = data_param[case]["ml"]["opt"]["BR"]

    fileres_MB_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                 (folder_MB_allperiods, case, arraytype[0]))
    fileres_MB = [TFile.Open("%s/finalcross%s%smult%d.root" % (folder_MB_allperiods, \
                        case, arraytype[0], i)) for i in [0, 1, 2]]

    fileres_trig_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                    (folder_triggered, case, arraytype[1]))
    fileres_trig = [TFile.Open("%s/finalcross%s%smult%d.root" % (folder_MB_allperiods, \
                          case, arraytype[0], i)) for i in [0, 1, 2]]

    #Corrected yield plot
    ccross = TCanvas('cCross', 'The Fit Canvas')
    ccross.SetCanvasSize(1500, 1500)
    ccross.SetWindowSize(500, 500)
    ccross.cd(1).DrawFrame(0, 1.e-9, 30, 10, ";#it{p}_{T} (GeV/#it{c});Corrected yield %s" % name)
    #ccross.SetLogx()

    legyield = TLegend(.25, .65, .65, .85)
    legyield.SetBorderSize(0)
    legyield.SetFillColor(0)
    legyield.SetFillStyle(0)
    legyield.SetTextFont(42)
    legyield.SetTextSize(0.035)

    colors = [kBlack, kRed, kGreen+2, kBlue]
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        gPad.SetLogy()
        hyield = fileres_MB_allperiods.Get("histoSigmaCorr%d" % (imult))
        hyield.Scale(1./br)
        hyield.SetLineColor(colors[imult])
        hyield.SetMarkerColor(colors[imult])
        hyield.SetMarkerStyle(21)
        hyield.Draw("same")
        legyieldstring = "%.1f < %s < %.1f (MB)" % \
                    (binsmin[imult], latexbin2var, binsmax[imult])
        legyield.AddEntry(hyield, legyieldstring, "LEP")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        gPad.SetLogy()
        hyieldHM = fileres_trig_allperiods.Get("histoSigmaCorr%d" % (imult))
        hyieldHM.Scale(1./br)
        hyieldHM.SetLineColor(colors[imult])
        hyieldHM.SetMarkerColor(colors[imult])
        hyieldHM.SetMarkerStyle(21)
        hyieldHM.Draw("same")
        legyieldstring = "%.1f < %s < %.1f (HM)" % \
              (binsmin[imult], latexbin2var, binsmax[imult])
        legyield.AddEntry(hyieldHM, legyieldstring, "LEP")
    legyield.Draw()

    ccross.SaveAs("ComparisonCorrYields_%s_%scombined%s.eps" % \
                  (case, arraytype[0], arraytype[1]))

    #Efficiency plot
    cEff = TCanvas('cEff', '', 800, 400)
    cEff.Divide(2)
    cEff.cd(1).DrawFrame(0, 1.e-4, 25, 1., ';#it{p}_{T} (GeV/#it{c});Prompt (Acc #times eff)')
    cEff.cd(1).SetLogy()

    legeff = TLegend(.3, .15, .7, .35)
    legeff.SetBorderSize(0)
    legeff.SetFillColor(0)
    legeff.SetFillStyle(0)
    legeff.SetTextFont(42)
    legeff.SetTextSize(0.035)

    lstyle = [1, 2, 3, 4]
    idx = 0
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        hEffpr = fileres_MB[idx].Get("hDirectEffpt")
        hEffpr.SetLineColor(colors[imult])
        hEffpr.SetLineStyle(lstyle[imult])
        hEffpr.SetMarkerColor(colors[imult])
        hEffpr.SetMarkerStyle(21)
        hEffpr.SetMarkerSize(0.8)
        hEffpr.Draw("same")
        legeffstring = "%.1f < %s < %.1f (MB)" % \
                         (binsmin[imult], latexbin2var, binsmax[imult])
        legeff.AddEntry(hEffpr, legeffstring, "LEP")
        idx = idx + 1

    idx = 0
    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        hEffprHM = fileres_trig[idx].Get("hDirectEffpt")
        hEffprHM.SetLineColor(colors[imult])
        hEffprHM.SetLineStyle(lstyle[imult])
        hEffprHM.SetMarkerColor(colors[imult])
        hEffprHM.SetMarkerStyle(21)
        hEffprHM.SetMarkerSize(0.8)
        hEffprHM.Draw("same")
        legeffstring = "%.1f < %s < %.1f (HM)" % \
                    (binsmin[imult], latexbin2var, binsmax[imult])
        legeff.AddEntry(hEffprHM, legeffstring, "LEP")
        idx = idx + 1
    legeff.Draw()

    cEff.cd(2).DrawFrame(0, 1.e-4, 25, 1., ';#it{p}_{T} (GeV/#it{c});Feed-down (Acc #times eff)')
    cEff.cd(2).SetLogy()

    idx = 0
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        hEfffd = fileres_MB[idx].Get("hFeedDownEffpt")
        hEfffd.SetLineColor(colors[imult])
        hEfffd.SetLineStyle(lstyle[imult])
        hEfffd.SetMarkerColor(colors[imult])
        hEfffd.SetMarkerStyle(21)
        hEfffd.SetMarkerSize(0.8)
        hEfffd.Draw("same")
        idx = idx + 1

    idx = 0
    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        gPad.SetLogy()
        hEfffdHM = fileres_trig[idx].Get("hFeedDownEffpt")
        hEfffdHM.SetLineColor(colors[imult])
        hEfffdHM.SetLineStyle(lstyle[imult])
        hEfffdHM.SetMarkerColor(colors[imult])
        hEfffdHM.SetMarkerStyle(21)
        hEfffdHM.Draw("same")
        idx = idx + 1

    cEff.SaveAs("ComparisonEfficiencies_%s_%scombined%s.eps" % \
                  (case, arraytype[0], arraytype[1]))

    #fprompt
    cfPrompt = TCanvas('cfPrompt', '', 800, 800)
    cfPrompt.Divide(2, 2)

    pt = TLatex()
    pt.SetTextSize(0.04)

    idx = 0
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        cfPrompt.cd(imult+1).DrawFrame(0, 0, 25, 1.05, ';#it{p}_{T} (GeV/#it{c});#it{f}_{prompt}')
        grfPrompt = fileres_MB[idx].Get("gFcConservative")
        grfPrompt.SetTitle(';#it{p}_{T} (GeV/#it{c});#it{f}_{prompt}')
        grfPrompt.SetLineColor(colors[imult])
        grfPrompt.SetMarkerColor(colors[imult])
        grfPrompt.SetMarkerStyle(21)
        grfPrompt.SetMarkerSize(0.5)
        grfPrompt.Draw("ap")
        pt.DrawTextNDC(0.15, 0.15, "%.1f < %s < %.1f (MB)" % \
                     (binsmin[imult], latexbin2var, binsmax[imult]))
        idx = idx + 1

    idx = 0
    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        cfPrompt.cd(imult+1).DrawFrame(0, 0, 25, 1.05, ';#it{p}_{T} (GeV/#it{c});#it{f}_{prompt}')
        grfPromptHM = fileres_trig[idx].Get("gFcConservative")
        grfPromptHM.SetLineColor(colors[imult])
        grfPromptHM.SetMarkerColor(colors[imult])
        grfPromptHM.SetMarkerStyle(21)
        grfPromptHM.SetMarkerSize(0.5)
        grfPromptHM.Draw("ap")
        pt.DrawTextNDC(0.15, 0.15, "%.1f < %s < %.1f (MB)" % \
                     (binsmin[imult], latexbin2var, binsmax[imult]))
        idx = idx + 1

    cfPrompt.SaveAs("ComparisonfPrompt_%s_%scombined%s.eps" % \
                  (case, arraytype[0], arraytype[1]))

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
def plot_hfptspectrum_ratios_comb(case_num, case_den, arraytype, isv0m=False):

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)
    gStyle.SetTitleOffset(1.15, "y")
    gStyle.SetTitleFont(42, "xy")
    gStyle.SetLabelFont(42, "xy")
    gStyle.SetTitleSize(0.042, "xy")
    gStyle.SetLabelSize(0.035, "xy")
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

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
    else:
        folder_num_triggered = \
            data_param_num[case_num]["analysis"][arraytype[1]]["data"]["resultsallp"]
        folder_den_triggered = \
            data_param_den[case_den]["analysis"][arraytype[1]]["data"]["resultsallp"]

    binsmin_num = data_param_num[case_num]["analysis"][arraytype[0]]["sel_binmin2"]
    binsmax_num = data_param_num[case_num]["analysis"][arraytype[0]]["sel_binmax2"]
    name_num = data_param_num[case_num]["analysis"][arraytype[0]]["latexnamemeson"]
    name_den = data_param_den[case_den]["analysis"][arraytype[0]]["latexnamemeson"]
    latexbin2var = data_param_num[case_num]["analysis"][arraytype[0]]["latexbin2var"]
    plotbinMB = data_param_num[case_num]["analysis"][arraytype[0]]["plotbin"]
    plotbinHM = data_param_num[case_num]["analysis"][arraytype[1]]["plotbin"]
    br_num = data_param_num[case_num]["ml"]["opt"]["BR"]
    br_den = data_param_den[case_den]["ml"]["opt"]["BR"]

    file_num_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                     (folder_num_allperiods, case_num, arraytype[0]))
    file_den_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                     (folder_den_allperiods, case_den, arraytype[0]))
    file_num_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                      (folder_num_triggered, case_num, arraytype[1]))
    file_den_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                      (folder_den_triggered, case_den, arraytype[1]))

    ccross = TCanvas('cRatioCross', 'The Fit Canvas')
    ccross.SetCanvasSize(1500, 1500)
    ccross.SetWindowSize(500, 500)
    ccross.cd(1).DrawFrame(0.9, 0, 30, 1, ";#it{p}_{T} (GeV/#it{c});%s / %s" % (name_num, name_den))
    ccross.cd(1).SetLogx()

    legyield = TLegend(.4, .68, .8, .88)
    legyield.SetBorderSize(0)
    legyield.SetFillColor(0)
    legyield.SetFillStyle(0)
    legyield.SetTextFont(42)
    legyield.SetTextSize(0.025)

    colors = [kBlack, kRed, kGreen+2, kBlue]
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        hratio = file_num_allperiods.Get("histoSigmaCorr%d" % (imult))
        hratio.Scale(1./br_num)
        hcross_den = file_den_allperiods.Get("histoSigmaCorr%d" % (imult))
        hcross_den.Scale(1./br_den)
        hratio.Divide(hcross_den)
        hratio.SetLineColor(colors[imult])
        hratio.SetMarkerColor(colors[imult])
        hratio.SetMarkerStyle(21)
        hratio.Draw("same")
        legyieldstring = "%.1f < %s < %.1f (MB)" % \
                    (binsmin_num[imult], latexbin2var, binsmax_num[imult])
        legyield.AddEntry(hratio, legyieldstring, "LEP")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        hratioHM = file_num_triggered.Get("histoSigmaCorr%d" % (imult))
        hratioHM.Scale(1./br_num)
        hcrossHM_den = file_den_triggered.Get("histoSigmaCorr%d" % (imult))
        hcrossHM_den.Scale(1./br_den)
        hratioHM.Divide(hcrossHM_den)
        hratioHM.SetLineColor(colors[imult])
        hratioHM.SetMarkerColor(colors[imult])
        hratioHM.Draw("same")
        legyieldstring = "%.1f < %s < %.1f (HM)" % \
                (binsmin_num[imult], latexbin2var, binsmax_num[imult])
        legyield.AddEntry(hratioHM, legyieldstring, "LEP")
    legyield.Draw()

    ccross.SaveAs("ComparisonRatios_%s%s_%scombined%s.eps" % \
                  (case_num, case_den, arraytype[0], arraytype[1]))


plot_hfptspectrum_comb("LcpK0spp", ["MBvspt_ntrkl", "SPDvspt"])
plot_hfptspectrum_comb("LcpK0spp", ["MBvspt_v0m", "V0mvspt"], True)
plot_hfptspectrum_comb("LcpK0spp", ["MBvspt_perc", "V0mvspt_perc_v0m"], True)
plot_hfptspectrum_comb("D0pp", ["MBvspt_ntrkl", "SPDvspt"])
plot_hfptspectrum_comb("D0pp", ["MBvspt_v0m", "V0mvspt"], True)
plot_hfptspectrum_comb("D0pp", ["MBvspt_perc", "V0mvspt_perc_v0m"], True)

plot_hfptspectrum_ratios_comb("LcpK0spp", "D0pp", ["MBvspt_ntrkl", "SPDvspt"])
plot_hfptspectrum_ratios_comb("LcpK0spp", "D0pp", ["MBvspt_v0m", "V0mvspt"], True)
plot_hfptspectrum_ratios_comb("LcpK0spp", "D0pp", ["MBvspt_perc", "V0mvspt_perc_v0m"], True)
