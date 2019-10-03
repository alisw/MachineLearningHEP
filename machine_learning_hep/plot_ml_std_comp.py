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
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend, TPad, TLine
from ROOT import gROOT, kRed, kGreen

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches, too-many-locals
def plot_hfspectrum_years_ratios(histo_ml, histo_std, title, x_label, y_label, save_path):

    max_y = max(histo_ml.GetMaximum(), histo_std.GetMaximum())
    min_y = max(histo_ml.GetMinimum(), histo_std.GetMinimum())
    if not min_y > 0.:
        min_y = 10.e-9

    max_x = max(histo_ml.GetXaxis().GetXmax(), histo_std.GetXaxis().GetXmax())
    min_x = max(histo_ml.GetXaxis().GetXmin(), histo_std.GetXaxis().GetXmin())

    #Corrected yield plot
    legend = TLegend(.45, .65, .85, .85)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.02)

    canvas = TCanvas('cCross', 'The Fit Canvas', 800, 800)
    canvas.SetLogy()
    pad_up = TPad("pad_up", "", 0., 0.4, 1., 1.)
    pad_up.SetBottomMargin(0.)
    pad_up.Draw()
    #ccross.SetLogx()
    pad_up.SetLogy()
    pad_up.cd()
    pad_up.DrawFrame(min_x, 0.5 * min_y, max_x, 2 * max_y, f"{title};;{y_label}")
    # ML histogram
    legend.AddEntry(histo_ml, "ML")
    histo_ml.SetLineColor(kGreen + 2)
    histo_ml.SetMarkerStyle(2)
    histo_ml.SetMarkerColor(kGreen + 2)
    histo_ml.SetStats(0)
    histo_ml.Draw("same")

    histo_ratio = histo_ml.Clone("ratio")
    histo_ratio.Divide(histo_std)
    # STD histogram
    legend.AddEntry(histo_std, "STD")
    histo_std.SetLineColor(kRed + 2)
    histo_std.SetLineStyle(7)
    histo_std.SetMarkerStyle(5)
    histo_std.SetMarkerColor(kRed + 2)
    histo_std.SetStats(0)
    histo_std.Draw("same")
    legend.Draw()

    canvas.cd()
    pad_double = TPad("pad_double", "", 0., 0.05, 1., 0.4)
    pad_double.SetTopMargin(0.)
    pad_double.SetBottomMargin(0.3)
    pad_double.Draw()
    pad_double.cd()
    frame_double = pad_double.DrawFrame(min_x, 0.5 * histo_ratio.GetMinimum(),
                                        max_x, 2 * histo_ratio.GetMaximum(),
                                        f"; {x_label} ; ML / STD")
    frame_double.SetTitleFont(42, "xy")
    frame_double.SetTitleSize(0.04, "xy")
    frame_double.SetLabelSize(0.04, "xy")
    histo_ratio.Draw("same")

    line_unity = TLine(frame_double.GetXaxis().GetXmin(), 1.,
                       frame_double.GetXaxis().GetXmax(), 1.)
    line_unity.SetLineColor(histo_std.GetLineColor())
    line_unity.SetLineStyle(histo_std.GetLineStyle())
    line_unity.Draw()

    canvas.SaveAs(save_path)
    canvas.Close()


def compare_ml_std(case_ml, ana_type_ml, filepath_std, map_std_bins=None):

    with open("data/database_ml_parameters_%s.yml" % case_ml, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    filepath_ml = data_param[case_ml]["analysis"][ana_type_ml]["data"]["resultsallp"]

    # Scale by branching ratio
    #br = data_param[case_ml]["ml"]["opt"]["BR"]
    #sigmav0 = data_param[case_ml]["analysis"]["sigmav0"]

    # Get pt spectrum files
    file_ml = TFile.Open(f"{filepath_ml}/finalcross{case_ml}{ana_type_ml}mult0.root", "READ")
    file_std = TFile.Open(filepath_std, "READ")

    # Collect histo names to quickly loop later
    histo_names = ["hDirectMCpt", "hFeedDownMCpt", "hDirectMCptMax", "hDirectMCptMin",
                   "hFeedDownMCptMax", "hFeedDownMCptMin", "hDirectEffpt", "hFeedDownEffpt",
                   "hRECpt", "histoYieldCorr", "histoYieldCorrMax", "histoYieldCorrMin",
                   "histoSigmaCorr", "histoSigmaCorrMax", "histoSigmaCorrMin"]

    for hn in histo_names:
        histo_ml = file_ml.Get(hn)
        histo_std_tmp = file_std.Get(hn)
        histo_std = None

        if not histo_ml or not histo_std:
            print(f"Could not find histogram {hn}, continue...")
            continue

        if "MC" not in hn and map_std_bins is not None:
            histo_std = histo_ml.Clone("std_rebin")
            histo_std.Reset("ICESM")
            histo_std_keep_bins.Reset("ICESM")
            histo_std_rebin_bins.Reset("ICESM")

            contents = [0 * histo_ml.GetNbins()]
            errors = [0 * histo_ml.GetNbins()]

            for std_bin, ml_bin in map_std_bins:
                for m  

            treat_std_as_ml_bins = []
            for rebins in merge_std_bins:
                ml_bin = rebins[0]
                content = 0.
                error = 0.
                for ibin in rebins:
                    content += histo_std_tmp.GetBinContent(ibin)
                    error += histo_std_tmp.GetBinError(ibin) * histo_std_tmp.GetBinError(ibin)
                histo_std_rebin_bins.SetBinContent(ml_bin, content)
                histo_std_rebinn_bins.SetBinError(ml_bin, sqrt(error))

            map_ml_std_bins = []
            counter = 1
            for ibin in histo_std_tmp.GetNbinsX():

                


        else:
            histo_std = histo_std_tmp.Clone("std_cloned")



        folder_plots = os.path.join(filepath_ml, "ml_std_comparison")
        if not os.path.exists(folder_plots):
            print("creating folder ", folder_plots)
            os.makedirs(folder_plots)

        save_path = f"{folder_plots}/{hn}_ml_std.eps"

        plot_hfspectrum_years_ratios(histo_ml, histo_std, histo_ml.GetTitle(),
                                     "#it{p}_{T} (GeV/#it{c}",
                                     histo_ml.GetYaxis().GetTitle(), save_path)

#####################################

gROOT.SetBatch(True)

#gROOT.SetStyle("Plain")
gStyle.SetOptStat(0)
gStyle.SetOptStat(0000)
gStyle.SetPalette(0)
gStyle.SetCanvasColor(0)
gStyle.SetFrameFillColor(0)
#gStyle.SetOptTitle(0)
gStyle.SetTitleOffset(1.15, "y")
gStyle.SetTitleFont(42, "xy")
gStyle.SetLabelFont(42, "xy")
gStyle.SetTitleSize(0.02, "xy")
gStyle.SetLabelSize(0.02, "xy")
gStyle.SetPadTickX(1)
gStyle.SetPadTickY(1)


compare_ml_std("D0pp", "MBvspt_ntrkl", "data/std_results/HPT_D020161718.root", [[2,3], [4,5]])
compare_ml_std("LcpKpipp", "MBvspt_ntrkl", "data/std_results/HP_Lc_newCut.root", [[2,3], [4,5]])
