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
from machine_learning_hep.utilities_plot import plot_histograms

FILES_NOT_FOUND = []
# One single particle ratio
# pylint: disable=too-many-branches, too-many-arguments
def plot_hfptspectrum_ml_over_std(case_ml, ana_type_ml, period_number, filepath_std, case_std,
                                  scale_std=None, map_std_bins=None, mult_bin=None,
                                  ml_histo_names=None, std_histo_names=None, suffix=""):

    with open("data/database_ml_parameters_%s.yml" % case_ml, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    if period_number < 0:
        filepath_ml = data_param[case_ml]["analysis"][ana_type_ml]["data"]["resultsallp"]
    else:
        filepath_ml = data_param[case_ml]["analysis"][ana_type_ml]["data"]["results"][period_number]

    # Get pt spectrum files
    if mult_bin is None:
        mult_bin = 0
    path_ml = f"{filepath_ml}/finalcross{case_ml}{ana_type_ml}mult{mult_bin}.root"
    if not os.path.exists(path_ml):
        FILES_NOT_FOUND.append(path_ml)
        return

    name = data_param[case_ml]["analysis"][ana_type_ml]["latexnamemeson"]

    file_ml = TFile.Open(path_ml, "READ")
    file_std = TFile.Open(filepath_std, "READ")

    # Collect histo names to quickly loop later
    histo_names = ["hDirectMCpt", "hFeedDownMCpt", "hDirectMCptMax", "hDirectMCptMin",
                   "hFeedDownMCptMax", "hFeedDownMCptMin", "hDirectEffpt", "hFeedDownEffpt",
                   "hRECpt", "histoYieldCorr", "histoYieldCorrMax", "histoYieldCorrMin",
                   "histoSigmaCorr", "histoSigmaCorrMax", "histoSigmaCorrMin"]
    if ml_histo_names is None:
        ml_histo_names = histo_names
    if std_histo_names is None:
        std_histo_names = histo_names


    for hn_ml, hn_std in zip(ml_histo_names, std_histo_names):
        histo_ml = file_ml.Get(hn_ml)
        histo_std_tmp = file_std.Get(hn_std)
        histo_std = None

        if not histo_ml:
            print(f"Could not find histogram {hn_ml}, continue...")
            continue
        if not histo_std_tmp:
            print(f"Could not find histogram {hn_std}, continue...")
            continue

        if "MC" not in hn_ml and map_std_bins is not None:
            histo_std = histo_ml.Clone("std_rebin")
            histo_std.Reset("ICESM")

            contents = [0] * histo_ml.GetNbinsX()
            errors = [0] * histo_ml.GetNbinsX()

            for ml_bin, std_bins in map_std_bins:
                for b in std_bins:
                    contents[ml_bin-1] += histo_std_tmp.GetBinWidth(b) * \
                            histo_std_tmp.GetBinContent(b) / histo_ml.GetBinWidth(ml_bin)
                    errors[ml_bin-1] += histo_std_tmp.GetBinError(b) * histo_std_tmp.GetBinError(b)

            for b in range(histo_std.GetNbinsX()):
                histo_std.SetBinContent(b+1, contents[b])
                histo_std.SetBinError(b+1, sqrt(errors[b]))

        else:
            histo_std = histo_std_tmp.Clone("std_cloned")

        if scale_std is not None:
            histo_std.Scale(scale_std)

        folder_plots = data_param[case_ml]["analysis"]["dir_general_plots"]
        if not os.path.exists(folder_plots):
            print("creating folder ", folder_plots)
            os.makedirs(folder_plots)

        h_ratio = histo_ml.Clone(f"{histo_ml.GetName()}_ratio")
        h_ratio.Divide(histo_std)

        save_path = f"{folder_plots}/{hn_ml}_ml_std_{case_ml}_over_{case_std}_{suffix}.eps"

        plot_histograms([h_ratio], False, False, None, histo_ml.GetTitle(),
                        "#it{p}_{T} (GeV/#it{c}", f"{name} / {case_std}", "",
                        save_path)

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
def plot_hfptspectrum_comb(case, arraytype):

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

    folder_plots = data_param[case]["analysis"]["dir_general_plots"]
    if not os.path.exists(folder_plots):
        print("creating folder ", folder_plots)
        os.makedirs(folder_plots)

    folder_MB_allperiods = data_param[case]["analysis"][arraytype[0]]["data"]["resultsallp"]
    folder_triggered = data_param[case]["analysis"][arraytype[1]]["data"]["resultsallp"]

    binsmin = data_param[case]["analysis"][arraytype[0]]["sel_binmin2"]
    binsmax = data_param[case]["analysis"][arraytype[0]]["sel_binmax2"]
    name = data_param[case]["analysis"][arraytype[0]]["latexnamemeson"]
    latexbin2var = data_param[case]["analysis"][arraytype[0]]["latexbin2var"]
    plotbinMB = data_param[case]["analysis"][arraytype[0]]["plotbin"]
    plotbinHM = data_param[case]["analysis"][arraytype[1]]["plotbin"]
    br = data_param[case]["ml"]["opt"]["BR"]
    sigmav0 = data_param[case]["analysis"]["sigmav0"]

    fileres_MB_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                 (folder_MB_allperiods, case, arraytype[0]))
    fileres_MB = [TFile.Open("%s/finalcross%s%smult%d.root" % (folder_MB_allperiods, \
                        case, arraytype[0], i)) for i in range(len(plotbinMB))]

    fileres_trig_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                    (folder_triggered, case, arraytype[1]))
    fileres_trig = [TFile.Open("%s/finalcross%s%smult%d.root" % (folder_triggered, \
                          case, arraytype[1], i)) for i in range(len(plotbinMB))]

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

    colors = [kBlack, kRed, kGreen+2, kBlue, kViolet-1, kOrange+2, kAzure+1, kOrange-7]
    tryunmerged = True
    if fileres_MB_allperiods and fileres_trig_allperiods:

        for imult, iplot in enumerate(plotbinMB):
            if not iplot:
                continue
            gPad.SetLogy()
            hyield = fileres_MB_allperiods.Get("histoSigmaCorr%d" % (imult))
            hyield.Scale(1./(br * sigmav0 * 1e12))
            hyield.SetLineColor(colors[imult % len(colors)])
            hyield.SetMarkerColor(colors[imult % len(colors)])
            hyield.SetMarkerStyle(21)
            hyield.SetMarkerSize(0.8)
            hyield.Draw("same")
            legyieldstring = "%.1f #leq %s < %.1f (MB)" % \
                        (binsmin[imult], latexbin2var, binsmax[imult])
            legyield.AddEntry(hyield, legyieldstring, "LEP")

        for imult, iplot in enumerate(plotbinHM):
            if not iplot:
                continue
            gPad.SetLogy()
            hyieldHM = fileres_trig_allperiods.Get("histoSigmaCorr%d" % (imult))
            hyieldHM.Scale(1./(br * sigmav0 * 1e12))
            hyieldHM.SetLineColor(colors[imult % len(colors)])
            hyieldHM.SetMarkerColor(colors[imult % len(colors)])
            hyieldHM.SetMarkerStyle(21)
            hyieldHM.SetMarkerSize(0.8)
            hyieldHM.Draw("same")
            legyieldstring = "%.1f #leq %s < %.1f (HM)" % \
                  (binsmin[imult], latexbin2var, binsmax[imult])
            legyield.AddEntry(hyieldHM, legyieldstring, "LEP")
        legyield.Draw()

        ccross.SaveAs("%s/PtSpec_ComparisonCorrYields_%s_%scombined%s.eps" % \
                  (folder_plots, case, arraytype[0], arraytype[1]))
        tryunmerged = False
    else:
        print("---Warning: Issue with merged, trying with unmerged files for %s (%s, %s)---" % \
                 (case, arraytype[0], arraytype[1]))

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        if not fileres_MB[imult]:
            print("---Warning: Issue with MB file. Eff, FD, CY plot skipped for %s (%s, %s)---" % \
                   (case, arraytype[0], arraytype[1]))
            return
    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        if not fileres_trig[imult]:
            print("---Warning: Issue with HM file. Eff, FD, CY plot skipped for %s (%s, %s)---" % \
                   (case, arraytype[0], arraytype[1]))
            return

    if tryunmerged is True:
        for imult, iplot in enumerate(plotbinMB):
            if not iplot:
                continue
            gPad.SetLogy()
            hyield = fileres_MB[imult].Get("histoSigmaCorr%d" % (imult))
            hyield.Scale(1./(br * sigmav0 * 1e12))
            hyield.SetLineColor(colors[imult % len(colors)])
            hyield.SetMarkerColor(colors[imult % len(colors)])
            hyield.SetMarkerStyle(21)
            hyield.SetMarkerSize(0.8)
            hyield.Draw("same")
            legyieldstring = "%.1f #leq %s < %.1f (MB)" % \
                        (binsmin[imult], latexbin2var, binsmax[imult])
            legyield.AddEntry(hyield, legyieldstring, "LEP")

        for imult, iplot in enumerate(plotbinHM):
            if not iplot:
                continue
            gPad.SetLogy()
            hyieldHM = fileres_trig[imult].Get("histoSigmaCorr%d" % (imult))
            hyieldHM.Scale(1./(br * sigmav0 * 1e12))
            hyieldHM.SetLineColor(colors[imult % len(colors)])
            hyieldHM.SetMarkerColor(colors[imult % len(colors)])
            hyieldHM.SetMarkerStyle(21)
            hyieldHM.SetMarkerSize(0.8)
            hyieldHM.Draw("same")
            legyieldstring = "%.1f #leq %s < %.1f (HM)" % \
                  (binsmin[imult], latexbin2var, binsmax[imult])
            legyield.AddEntry(hyieldHM, legyieldstring, "LEP")
        legyield.Draw()

        ccross.SaveAs("%s/PtSpec_ComparisonCorrYields_%s_%scombined%s.eps" % \
                  (folder_plots, case, arraytype[0], arraytype[1]))

    #Efficiency plot
    cEff = TCanvas('cEff', '', 800, 400)
    cEff.Divide(2)
    cEff.cd(1).DrawFrame(0, 1.e-4, 25, 1., \
                         ";#it{p}_{T} (GeV/#it{c});Prompt %s (Acc #times eff)" % name)
    cEff.cd(1).SetLogy()

    legeff = TLegend(.3, .15, .7, .35)
    legeff.SetBorderSize(0)
    legeff.SetFillColor(0)
    legeff.SetFillStyle(0)
    legeff.SetTextFont(42)
    legeff.SetTextSize(0.035)

    lstyle = [1, 2, 3, 4, 5]
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        hEffpr = fileres_MB[imult].Get("hDirectEffpt")
        hEffpr.SetLineColor(colors[imult % len(colors)])
        hEffpr.SetLineStyle(lstyle[imult % len(lstyle)])
        hEffpr.SetMarkerColor(colors[imult % len(colors)])
        hEffpr.SetMarkerStyle(21)
        hEffpr.SetMarkerSize(0.8)
        hEffpr.Draw("same")
        legeffstring = "%.1f #leq %s < %.1f (MB)" % \
                         (binsmin[imult], latexbin2var, binsmax[imult])
        legeff.AddEntry(hEffpr, legeffstring, "LEP")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        hEffprHM = fileres_trig[imult].Get("hDirectEffpt")
        hEffprHM.SetLineColor(colors[imult % len(colors)])
        hEffprHM.SetLineStyle(lstyle[imult % len(lstyle)])
        hEffprHM.SetMarkerColor(colors[imult % len(colors)])
        hEffprHM.SetMarkerStyle(21)
        hEffprHM.SetMarkerSize(0.8)
        hEffprHM.Draw("same")
        legeffstring = "%.1f #leq %s < %.1f (HM)" % \
                    (binsmin[imult], latexbin2var, binsmax[imult])
        legeff.AddEntry(hEffprHM, legeffstring, "LEP")
    legeff.Draw()

    cEff.cd(2).DrawFrame(0, 1.e-4, 25, 1., \
                         ";#it{p}_{T} (GeV/#it{c});Feed-down %s (Acc #times eff)" % name)
    cEff.cd(2).SetLogy()

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        hEfffd = fileres_MB[imult].Get("hFeedDownEffpt")
        hEfffd.SetLineColor(colors[imult % len(colors)])
        hEfffd.SetLineStyle(lstyle[imult % len(lstyle)])
        hEfffd.SetMarkerColor(colors[imult % len(colors)])
        hEfffd.SetMarkerStyle(21)
        hEfffd.SetMarkerSize(0.8)
        hEfffd.Draw("same")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        hEfffdHM = fileres_trig[imult].Get("hFeedDownEffpt")
        hEfffdHM.SetLineColor(colors[imult % len(colors)])
        hEfffdHM.SetLineStyle(lstyle[imult % len(lstyle)])
        hEfffdHM.SetMarkerColor(colors[imult % len(colors)])
        hEfffdHM.SetMarkerStyle(21)
        hEfffdHM.Draw("same")

    cEff.SaveAs("%s/PtSpec_ComparisonEfficiencies_%s_%scombined%s.eps" % \
                  (folder_plots, case, arraytype[0], arraytype[1]))

    #Efficiency ratio plot
    cEffRatio = TCanvas('cEffRatio', '', 800, 400)
    cEffRatio.Divide(2)
    cEffRatio.cd(1).DrawFrame(0, 0.5, 25, 1.5, \
                         ";#it{p}_{T} (GeV/#it{c});Prompt %s (Acc #times eff) Ratio" % name)

    hEffprden = TH1F()
    if plotbinMB[0] == 1:
        hEffprden = fileres_MB[0].Get("hDirectEffpt")
    if plotbinHM[0] == 1:
        hEffprden = fileres_trig[0].Get("hDirectEffpt")

    lstyle = [1, 2, 3, 4]
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        if imult == 0:
            hEffpr = hEffprden.Clone()
        else:
            hEffpr = fileres_MB[imult].Get("hDirectEffpt")
        hEffpr.SetLineColor(colors[imult % len(colors)])
        hEffpr.SetLineStyle(lstyle[imult % len(lstyle)])
        hEffpr.SetMarkerColor(colors[imult % len(colors)])
        hEffpr.SetMarkerStyle(21)
        hEffpr.SetMarkerSize(0.8)
        hEffpr.Divide(hEffprden)
        hEffpr.Draw("same")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        if imult == 0:
            hEffprHM = hEffprden.Clone()
        else:
            hEffprHM = fileres_trig[imult].Get("hDirectEffpt")
        hEffprHM.SetLineColor(colors[imult % len(colors)])
        hEffprHM.SetLineStyle(lstyle[imult % len(lstyle)])
        hEffprHM.SetMarkerColor(colors[imult % len(colors)])
        hEffprHM.SetMarkerStyle(21)
        hEffprHM.SetMarkerSize(0.8)
        hEffprHM.Divide(hEffprden)
        hEffprHM.Draw("same")
    legeff.Draw()

    cEffRatio.cd(2).DrawFrame(0, 0.5, 25, 1.5, \
                         ";#it{p}_{T} (GeV/#it{c});Feed-down %s (Acc #times eff) Ratio" % name)

    hEfffdden = TH1F()
    if plotbinMB[0] == 1:
        hEfffdden = fileres_MB[0].Get("hFeedDownEffpt")
    if plotbinHM[0] == 1:
        hEfffdden = fileres_trig[0].Get("hFeedDownEffpt")

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        if imult == 0:
            hEfffd = hEfffdden.Clone()
        else:
            hEfffd = fileres_MB[imult].Get("hFeedDownEffpt")
        hEfffd.SetLineColor(colors[imult % len(colors)])
        hEfffd.SetLineStyle(lstyle[imult % len(lstyle)])
        hEfffd.SetMarkerColor(colors[imult % len(colors)])
        hEfffd.SetMarkerStyle(21)
        hEfffd.SetMarkerSize(0.8)
        hEfffd.Divide(hEfffdden)
        hEfffd.Draw("same")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        if imult == 0:
            hEfffdHM = hEfffdden.Clone()
        else:
            hEfffdHM = fileres_trig[imult].Get("hFeedDownEffpt")
        hEfffdHM.SetLineColor(colors[imult % len(colors)])
        hEfffdHM.SetLineStyle(lstyle[imult % len(lstyle)])
        hEfffdHM.SetMarkerColor(colors[imult % len(colors)])
        hEfffdHM.SetMarkerStyle(21)
        hEfffdHM.Divide(hEfffdden)
        hEfffdHM.Draw("same")

    cEffRatio.SaveAs("%s/PtSpec_ComparisonEfficienciesRatio_%s_%scombined%s.eps" % \
                  (folder_plots, case, arraytype[0], arraytype[1]))

    #fprompt
    cfPrompt = TCanvas('cfPrompt', '', 1200, 800)
    cfPrompt.Divide(3, 2)

    pt = TLatex()
    pt.SetTextSize(0.04)

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        cfPrompt.cd(imult+1).DrawFrame(0, 0, 25, 1.05, \
                                       ";#it{p}_{T} (GeV/#it{c});#it{f}_{prompt} %s" % name)
        grfPrompt = fileres_MB[imult].Get("gFcConservative")
        grfPrompt.SetTitle(";#it{p}_{T} (GeV/#it{c});#it{f}_{prompt} %s" % name)
        grfPrompt.SetLineColor(colors[imult % len(colors)])
        grfPrompt.SetMarkerColor(colors[imult % len(colors)])
        grfPrompt.SetMarkerStyle(21)
        grfPrompt.SetMarkerSize(0.5)
        grfPrompt.Draw("ap")
        pt.DrawLatexNDC(0.15, 0.15, "%.1f #leq %s < %.1f (MB)" % \
                     (binsmin[imult], latexbin2var, binsmax[imult]))

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        cfPrompt.cd(imult+1).DrawFrame(0, 0, 25, 1.05, \
                                       ";#it{p}_{T} (GeV/#it{c});#it{f}_{prompt} %s" % name)
        grfPromptHM = fileres_trig[imult].Get("gFcConservative")
        grfPromptHM.SetTitle(";#it{p}_{T} (GeV/#it{c});#it{f}_{prompt} %s" % name)
        grfPromptHM.SetLineColor(colors[imult % len(colors)])
        grfPromptHM.SetMarkerColor(colors[imult % len(colors)])
        grfPromptHM.SetMarkerStyle(21)
        grfPromptHM.SetMarkerSize(0.5)
        grfPromptHM.Draw("ap")
        pt.DrawLatexNDC(0.15, 0.15, "%.1f #leq %s < %.1f (HM)" % \
                     (binsmin[imult], latexbin2var, binsmax[imult]))

    cfPrompt.SaveAs("%s/PtSpec_ComparisonfPrompt_%s_%scombined%s.eps" % \
                  (folder_plots, case, arraytype[0], arraytype[1]))

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
def plot_hfptspectrum_ratios_comb(case_num, case_den, arraytype):

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

    folder_plots_num = data_param_num[case_num]["analysis"]["dir_general_plots"]
    folder_plots_den = data_param_den[case_den]["analysis"]["dir_general_plots"]
    if not os.path.exists(folder_plots_num):
        print("creating folder ", folder_plots_num)
        os.makedirs(folder_plots_num)
    if not os.path.exists(folder_plots_den):
        print("creating folder ", folder_plots_den)
        os.makedirs(folder_plots_den)

    folder_num_allperiods = \
        data_param_num[case_num]["analysis"][arraytype[0]]["data"]["resultsallp"]
    folder_den_allperiods = \
        data_param_den[case_den]["analysis"][arraytype[0]]["data"]["resultsallp"]
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
    sigmav0_num = data_param_num[case_num]["analysis"]["sigmav0"]
    sigmav0_den = data_param_den[case_den]["analysis"]["sigmav0"]

    file_num_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                     (folder_num_allperiods, case_num, arraytype[0]))
    file_den_allperiods = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                     (folder_den_allperiods, case_den, arraytype[0]))
    file_num_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                      (folder_num_triggered, case_num, arraytype[1]))
    file_den_triggered = TFile.Open("%s/finalcross%s%smulttot.root" % \
                                      (folder_den_triggered, case_den, arraytype[1]))

    if not file_num_allperiods or not file_num_triggered:
        print("---Warning: Issue with %s merged files. Meson ratio plot skipped (%s, %s)---" % \
                 (case_num, arraytype[0], arraytype[1]))
        return
    if not file_den_allperiods or not file_den_triggered:
        print("---Warning: Issue with %s merged files. Meson ratio plot skipped (%s, %s)---" % \
                 (case_den, arraytype[0], arraytype[1]))
        return

    rootfilename = "%s/ComparisonRatios_%s%s_%scombined%s.root" % \
                     (folder_plots_num, case_num, case_den, arraytype[0], arraytype[1])
    fileoutput = TFile.Open(rootfilename, "recreate")

    ccross = TCanvas('cRatioCross', 'The Fit Canvas')
    ccross.SetCanvasSize(1500, 1500)
    ccross.SetWindowSize(500, 500)
    maxplot = 1.0
    if case_num == "Dspp":
        maxplot = 0.5
    ccross.cd(1).DrawFrame(0.9, 0, 30, maxplot, ";#it{p}_{T} (GeV/#it{c});%s / %s" % \
                           (name_num, name_den))
    ccross.cd(1).SetLogx()

    legyield = TLegend(.4, .68, .8, .88)
    legyield.SetBorderSize(0)
    legyield.SetFillColor(0)
    legyield.SetFillStyle(0)
    legyield.SetTextFont(42)
    legyield.SetTextSize(0.025)

    colors = [kBlack, kRed, kGreen+2, kBlue, kViolet-1, kOrange+2, kAzure+1, kOrange-7]
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        hratio = file_num_allperiods.Get("histoSigmaCorr%d" % (imult))
        hratio.Scale(1./(br_num * sigmav0_num * 1e12))
        hcross_den = file_den_allperiods.Get("histoSigmaCorr%d" % (imult))
        hcross_den.Scale(1./(br_den * sigmav0_den * 1e12))
        hratio.Divide(hcross_den)
        hratio.SetLineColor(colors[imult % len(colors)])
        hratio.SetMarkerColor(colors[imult % len(colors)])
        hratio.SetMarkerStyle(21)
        hratio.SetTitle(";#it{p}_{T} (GeV/#it{c});%s / %s" % (name_num, name_den))
        hratio.Draw("same")
        legyieldstring = "%.1f #leq %s < %.1f (MB)" % \
                    (binsmin_num[imult], latexbin2var, binsmax_num[imult])
        legyield.AddEntry(hratio, legyieldstring, "LEP")
        fileoutput.cd()
        hratio.Write("hratio_fromMB_%.1f_%s_%.1f" % \
                          (binsmin_num[imult], latexbin2var, binsmax_num[imult]))

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        hratioHM = file_num_triggered.Get("histoSigmaCorr%d" % (imult))
        hratioHM.Scale(1./(br_num * sigmav0_num * 1e12))
        hcrossHM_den = file_den_triggered.Get("histoSigmaCorr%d" % (imult))
        hcrossHM_den.Scale(1./(br_den * sigmav0_den * 1e12))
        hratioHM.Divide(hcrossHM_den)
        hratioHM.SetLineColor(colors[imult % len(colors)])
        hratioHM.SetMarkerColor(colors[imult % len(colors)])
        hratioHM.SetTitle(";#it{p}_{T} (GeV/#it{c});%s / %s" % (name_num, name_den))
        hratioHM.Draw("same")
        legyieldstring = "%.1f #leq %s < %.1f (HM)" % \
                (binsmin_num[imult], latexbin2var, binsmax_num[imult])
        legyield.AddEntry(hratioHM, legyieldstring, "LEP")
        fileoutput.cd()
        hratioHM.Write("hratio_fromHM_%.1f_%s_%.1f" % \
                          (binsmin_num[imult], latexbin2var, binsmax_num[imult]))
    legyield.Draw()

    ccross.SaveAs("%s/PtSpec_ComparisonRatios_%s%s_%scombined%s_logx.eps" % \
                  (folder_plots_num, case_num, case_den, arraytype[0], arraytype[1]))
    ccross.SaveAs("%s/PtSpec_ComparisonRatios_%s%s_%scombined%s_logx.eps" % \
                  (folder_plots_den, case_num, case_den, arraytype[0], arraytype[1]))

    ccross.cd(1).SetLogx(0)
    ccross.SaveAs("%s/PtSpec_ComparisonRatios_%s%s_%scombined%s.eps" % \
                  (folder_plots_num, case_num, case_den, arraytype[0], arraytype[1]))
    ccross.SaveAs("%s/PtSpec_ComparisonRatios_%s%s_%scombined%s.eps" % \
                  (folder_plots_den, case_num, case_den, arraytype[0], arraytype[1]))

    fileoutput.cd()
    ccross.Write()
    fileoutput.Close()

    rootfilenameden = "%s/ComparisonRatios_%s%s_%scombined%s.root" % \
                        (folder_plots_den, case_num, case_den, arraytype[0], arraytype[1])
    copyfile(rootfilename, rootfilenameden)
    print("---Output stored in:", rootfilename, "and", rootfilenameden, "---")

gROOT.SetBatch(True)

#EXAMPLE HOW TO USE plot_hfptspectrum_comb
#  ---> Combines and plots the output of HFPtSpectrum in nice way
#plot_hfptspectrum_comb("Dspp", ["MBvspt_ntrkl", "SPDvspt"])

#EXAMPLE HOW TO USE plot_hfptspectrum_ratios_comb
#  ---> Combines and plots particle-ratio both with MLHEP
#plot_hfptspectrum_ratios_comb("Dspp", "D0pp", ["MBvspt_ntrkl", "SPDvspt"])

#EXAMPLES HOW TO USE plot_hfptspectrum_ml_over_std
#  ---> Plots particle-ratio with MLHEP and inputfile from STD analyses
#plot_hfptspectrum_ml_over_std("Dspp", "MBvspt_ntrkl", -1,
#                              "data/std_results/HFPtSpectrum_D0_merged_20191010.root",
#                              "D0", 2.27 / 3.89, None, 0, ["histoSigmaCorr"], ["histoSigmaCorr"])
#plot_hfptspectrum_ml_over_std("Dspp", "MBvspt_ntrkl", 0,
#                              "data/std_results/HFPtSpectrum_D0_2016_prel_5tev_20191015.root",
#                              "D0", 2.27 / 3.89,
#                              [(1, [1]), (2, [2, 3]), (3, [4, 5]), (4, [6]), (5, [7]), (6, [8])],
#                              0, ["histoSigmaCorr"], ["histoSigmaCorr"],
#                              "_prelim_5tev")
