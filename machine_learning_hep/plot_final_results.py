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
from array import array
from math import sqrt
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, gStyle, gROOT, TH1F, TGraphAsymmErrors
from ROOT import kBlue, kAzure, kOrange, kGreen, kBlack, kRed, kWhite
from machine_learning_hep.utilities import plot_histograms

FILES_NOT_FOUND = []


def results(histos_central, systematics_rel_all, title, legend_titles, x_label, y_label,
            save_path, **kwargs):

    if len(histos_central) != len(legend_titles):
        print(f"Number of legend titles {len(legend_titles)} differs from number of " \
              f"histograms {len(histos_central)}")
        return

    colors = kwargs.get("colors", [kRed - i for i in range(len(histos_central))])
    colors = colors * 2
    markerstyles = [1] * len(histos_central) + [20] * len(histos_central)
    draw_options = ["E2"] * len(histos_central) + [""] * len(histos_central)
    legend_titles = [None] * len(histos_central) + legend_titles

    histos_syst = []
    for h, systematics_rel in zip(histos_central, systematics_rel_all):
        n_bins_central = h.GetNbinsX()
        bin_centers = array("d", [h.GetXaxis().GetBinCenter(b+1) for b in range(n_bins_central)])
        bin_contents = array("d", [h.GetBinContent(b+1) for b in range(n_bins_central)])
        # First find minimum bin width and make it a bit smaller even for the overlayed systematics
        syst_width = min([h.GetBinWidth(b+1) for b in range(h.GetNbinsX())])
        syst_width = 0.5 * syst_width
        syst_width = 0.5 * syst_width

        x_syst_low = array("d", [syst_width for _ in range(n_bins_central)])
        x_syst_up = array("d", [syst_width for _ in range(n_bins_central)])
        print(bin_contents)
        print(x_syst_low)
        print(x_syst_up)

        # Low and upper errors
        systematics_rel_squ = [[0, 0] for _ in range(h.GetNbinsX())]
        for systs in systematics_rel:
            for i, syst in enumerate(systs):
                systematics_rel_squ[i][0] += (syst[0] * syst[0])
                systematics_rel_squ[i][1] += (syst[1] * syst[1])
        y_syst_low = [1 - sqrt(syst_rel_squ[0]) for syst_rel_squ in systematics_rel_squ]
        y_syst_up = [1 + sqrt(syst_rel_squ[1]) for syst_rel_squ in systematics_rel_squ]

        y_syst_low = array("d", [abs(y_syst_low[b] - 1) * h.GetBinContent(b+1) \
                for b in range(n_bins_central)])
        y_syst_up = array("d", [abs(y_syst_up[b] -1) * h.GetBinContent(b+1) \
                for b in range(n_bins_central)])
        print(y_syst_low)
        print(y_syst_up)
        gr_err = TGraphAsymmErrors(n_bins_central, bin_centers, bin_contents, x_syst_low,
                                   x_syst_up, y_syst_low, y_syst_up)

        histos_syst.append(gr_err)

    plot_histograms([*histos_syst, *histos_central], True, False, legend_titles, title,
                    x_label, y_label, "", save_path, linesytles=[1], markerstyles=markerstyles,
                    colors=colors, linewidths=[1], draw_options=draw_options,
                    fillstyles=[0])

def get_param(case):
    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    return data_param

def extract_histo(case, ana_type, mult_bin, period_number, histo_name):
    data_param = get_param(case)
    filepath = data_param[case]["analysis"][ana_type]["data"]["resultsallp"]
    if period_number < 0:
        period_number = -1
        filepath = data_param[case]["analysis"][ana_type]["data"]["results"][period_number]

    path = f"{filepath}/finalcross{case}{ana_type}mult{mult_bin}.root"
    in_file = TFile.Open(path, "READ")
    histo = in_file.Get(histo_name)
    histo.SetDirectory(0)
    return histo

def make_standard_save_path(case, prefix):
    data_param = get_param(case)
    folder_plots = data_param[case]["analysis"]["dir_general_plots"]
    folder_plots = f"{folder_plots}/final"
    if not os.path.exists(folder_plots):
        print("creating folder ", folder_plots)
        os.makedirs(folder_plots)
    return f"{folder_plots}/{prefix}.eps"

#############################################################
gROOT.SetBatch(True)

CASE = "Dspp"
ANA_MB = "MBvspt_ntrkl"
ANA_HM = "SPDvspt"
YEAR_NUMBER = -1 # -1 refers to all years merged

LEGEND_TITLES = ["0 < n_{trkl} < #infty (MB)", "1 < n_{trkl} < 9 (MB)", "10 < n_{trkl} < 29 (MB)",
                 "30 < n_{trkl} < 59 (MB)", "60 < n_{trkl} < 99 (HM)"]

COLORS = [kBlue, kGreen + 2, kRed - 2, kAzure + 3, kOrange + 7]

# Get the ML histogram of the particle case and analysis type
# Everything available in the HFPtSpetrum can be requested
# From MB
HISTOS = []
for mb in range(4):
    histo_ = extract_histo(CASE, ANA_MB, mb, YEAR_NUMBER, "histoSigmaCorr")
    histo_.SetName(f"{histo_.GetName()}_{mb}")
    HISTOS.append(histo_)

# Append the HM histogram
HISTO = extract_histo(CASE, ANA_HM, 4, YEAR_NUMBER, "histoSigmaCorr")
HISTO.SetName(f"{HISTO.GetName()}_4")
HISTOS.append(HISTO)


# Relative systematic uncertainties
REL_SYST = [[[(0.05, 0.1), (0.01, 0.04), (0.03, 0.05), (0.02, 0.12), (0.01, 0.04)],
             [(0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08)]],
            [[(0.05, 0.1), (0.01, 0.04), (0.03, 0.05), (0.02, 0.12), (0.01, 0.04)],
             [(0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08)]],
            [[(0.05, 0.1), (0.01, 0.04), (0.03, 0.05), (0.02, 0.12), (0.01, 0.04)],
             [(0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08)]],
            [[(0.05, 0.1), (0.01, 0.04), (0.03, 0.05), (0.02, 0.12), (0.01, 0.04)],
             [(0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08)]],
            [[(0.05, 0.1), (0.01, 0.04), (0.03, 0.05), (0.02, 0.12), (0.01, 0.04)],
             [(0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08), (0.08, 0.08)]]]

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, "histoSigmaCorr_all_years_MB_HM")


results(HISTOS, REL_SYST, f"Corrected cross section", LEGEND_TITLES,
        "#it{p}_{T} (GeV/#it{c})", "#sigma (D_{s}) #times BR(D_{s} #rightarrow KK#pi)",
        SAVE_PATH, colors=COLORS)
