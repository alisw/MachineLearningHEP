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
from ROOT import gStyle, TLegend
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
    canvas.cd(1).DrawFrame(min_x, 0.5 * min_y, max_x, 2 * max_y, f"{title};{x_label};{y_label}")
    # ML histogram
    legend.AddEntry(histo_ml, "ML")
    histo_ml.SetLineColor(kGreen + 2)
    histo_ml.SetMarkerStyle(2)
    histo_ml.SetMarkerColor(kGreen + 2)
    histo_ml.SetStats(0)
    histo_ml.Draw("same")
    # STD histogram
    legend.AddEntry(histo_std, "STD")
    histo_std.SetLineColor(kRed + 2)
    histo_std.SetLineStyle(7)
    histo_std.SetMarkerStyle(5)
    histo_std.SetMarkerColor(kRed + 2)
    histo_std.SetStats(0)
    histo_std.Draw("same")
    legend.Draw()
    canvas.SaveAs(save_path)
    canvas.Close()

def compare_ml_std(case_ml, ana_type_ml, filepath_std):

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
        histo_std = file_std.Get(hn)
        if not histo_ml or not histo_std:
            print(f"Could not find histogram {hn}, continue...")
            continue

        folder_plots = os.path.join(filepath_ml, "ml_std_comparison")
        if not os.path.exists(folder_plots):
            print("creating folder ", folder_plots)
            os.makedirs(folder_plots)

        save_path = f"{folder_plots}/{hn}_ml_std.eps"

        plot_hfspectrum_years_ratios(histo_ml, histo_std, histo_ml.GetTitle(),
                                     histo_ml.GetXaxis().GetTitle(),
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


compare_ml_std("D0pp", "MBvspt_ntrkl", "data/std_results/HPT_D020161718.root")
compare_ml_std("LcpKpipp", "MBvspt_ntrkl", "data/std_results/HP_Lc_newCut.root")
