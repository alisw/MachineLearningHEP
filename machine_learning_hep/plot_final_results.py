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
from ROOT import TFile, gStyle, gROOT, TH1F, TGraphAsymmErrors, TH1
from ROOT import kBlue, kAzure, kOrange, kGreen, kBlack, kRed, kWhite
from machine_learning_hep.utilities import plot_histograms

FILES_NOT_FOUND = []


def results(histos_central, systematics_rel_all, title, legend_titles, x_label, y_label,
            save_path, **kwargs):

    if len(histos_central) != len(legend_titles):
        print(f"Number of legend titles {len(legend_titles)} differs from number of " \
              f"histograms {len(histos_central)}")
        return

    add_to_syst = kwargs.get("add_to_syst", [None] * len(histos_central))
    colors = kwargs.get("colors", [kRed - i for i in range(len(histos_central))])
    if len(histos_central) != len(colors):
        print(f"Number of colors {len(colors)} differs from number of " \
              f"histograms {len(histos_central)}")
        return
    colors = colors * 2
    markerstyles = [1] * len(histos_central) + [20] * len(histos_central)
    draw_options = ["E2"] * len(histos_central) + [""] * len(histos_central)
    legend_titles = [None] * len(histos_central) + legend_titles

    if systematics_rel_all is None:
        systematics_rel_all = [None] * len(histos_central)

    errs_syst = []
    for h, systematics_rel, syst_add in zip(histos_central, systematics_rel_all, add_to_syst):
        n_bins_central = h.GetNbinsX()
        bin_centers = array("d", [h.GetXaxis().GetBinCenter(b+1) for b in range(n_bins_central)])
        bin_contents = array("d", [h.GetBinContent(b+1) for b in range(n_bins_central)])
        # First find minimum bin width and make it a bit smaller even for the overlayed systematics
        syst_width = min([h.GetBinWidth(b+1) for b in range(h.GetNbinsX())])
        syst_width = 0.25 * syst_width

        # Low and upper errors
        systematics_rel_squ = [[0, 0] for _ in range(h.GetNbinsX())]
        if systematics_rel is not None:
            for systs in systematics_rel:
                for i, syst in enumerate(systs):
                    systematics_rel_squ[i][0] += (syst[0] * syst[0])
                    systematics_rel_squ[i][1] += (syst[1] * syst[1])

        # To finally combine all the y errors, that can be done more efficiently for sure...

        # Get square root from all
        y_syst_low = [sqrt(syst_rel_squ[0]) for syst_rel_squ in systematics_rel_squ]
        y_syst_up = [sqrt(syst_rel_squ[1]) for syst_rel_squ in systematics_rel_squ]

        # Make absolute from relative
        y_syst_low = [y_syst_low[b] * h.GetBinContent(b+1) for b in range(n_bins_central)]
        y_syst_up = [y_syst_up[b] * h.GetBinContent(b+1) for b in range(n_bins_central)]

        # If an additional TGraphAsymmErrors object is given, add to what was given
        # in user syst list
        if syst_add is not None:
            y_syst_low = [sqrt(y_syst_low[b] * y_syst_low[b] + \
                    syst_add.GetErrorYlow(b+1) * syst_add.GetErrorYlow(b+1)) \
                    for b in range(n_bins_central)]
            y_syst_up = [sqrt(y_syst_up[b] * y_syst_up[b] + \
                    syst_add.GetErrorYhigh(b+1) * syst_add.GetErrorYhigh(b+1)) \
                    for b in range(n_bins_central)]

        # Now make an array out of the list of errors
        y_syst_low = array("d", y_syst_low)
        y_syst_up = array("d", y_syst_up)

        # Mandatory to make the boxes in X fine
        x_syst_low = array("d", [syst_width for _ in range(n_bins_central)])
        x_syst_up = array("d", [syst_width for _ in range(n_bins_central)])

        # Make the final error object to be used
        gr_err = TGraphAsymmErrors(n_bins_central, bin_centers, bin_contents, x_syst_low,
                                   x_syst_up, y_syst_low, y_syst_up)

        errs_syst.append(gr_err)

    plot_histograms([*errs_syst, *histos_central], True, False, legend_titles, title,
                    x_label, y_label, "", save_path, linesytles=[1], markerstyles=markerstyles,
                    colors=colors, linewidths=[1], draw_options=draw_options,
                    fillstyles=[0])

def get_param(case):
    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    return data_param

def extract_histo_or_error(case, ana_type, mult_bin, period_number, histo_name):
    data_param = get_param(case)
    filepath = data_param[case]["analysis"][ana_type]["data"]["resultsallp"]
    if period_number < 0:
        period_number = -1
        filepath = data_param[case]["analysis"][ana_type]["data"]["results"][period_number]

    path = f"{filepath}/finalcross{case}{ana_type}mult{mult_bin}.root"
    in_file = TFile.Open(path, "READ")
    histo = in_file.Get(histo_name)
    if isinstance(histo, TH1):
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
                 "30 < n_{trkl} < 59 (MB)"]

COLORS = [kBlue, kGreen + 2, kRed - 2, kAzure + 3]

# Get the ML histogram of the particle case and analysis type
# Everything available in the HFPtSpetrum can be requested
# From MB
HISTOS = []
ERRS = []
for mb in range(4):
    histo_ = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "histoSigmaCorr")
    histo_.SetName(f"{histo_.GetName()}_{mb}")
    HISTOS.append(histo_)
    errs = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "gSigmaCorr")
    errs.SetName(f"{errs.GetName()}_{mb}")
    ERRS.append(errs)

# Relative systematic uncertainties
# This table is taken from https://indico.cern.ch/event/855807/contributions/3601563/
#                           attachments/1928380/3193129/HF_171019_Ds_MLHEP.pdf
# slides 32 and 33
#
# How the following list works
# -> The elements of the most outer list correspond to the list of histograms to be plotted, hence
#    for 4 mult. bins
# -> The next level contains different sources of uncertainties
# -> The inner most list contains tuples, one tuple per pT bin. The first entry is the lower,
#    the second one the upper error
REL_SYST = [[[(0.02, 0.02), (0.02, 0.02), (0.02, 0.02), (0.02, 0.02), (0.03, 0.03)],
             [(0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06)],
             [(0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)],
             [(0.05, 0.05), (0.05, 0.04), (0.05, 0.04), (0.06, 0.05), (0.05, 0.04)],
             [(0.02, 0.02), (0.005, 0.005), (0.05, 0.005), (0.005, 0.005), (0., 0.)],
             [(0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035)],
             [(0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03)]],
            [[(0.04, 0.04), (0.04, 0.04), (0.03, 0.03), (0.06, 0.06), (0.22, 0.22)],
             [(0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06)],
             [(0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)],
             [(0.04, 0.12), (0.05, 0.12), (0.05, 0.12), (0.05, 0.12), (0.06, 0.12)],
             [(0.005, 0.005), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
             [(0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035)],
             [(0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03)]],
            [[(0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.06, 0.06)],
             [(0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06)],
             [(0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)],
             [(0.15, 0.1), (0.15, 0.1), (0.15, 0.1), (0.15, 0.1), (0.15, 0.1)],
             [(0.005, 0.005), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
             [(0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035)],
             [(0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03)]],
            [[(0.04, 0.04), (0.03, 0.03), (0.04, 0.04), (0.05, 0.05), (0.06, 0.06)],
             [(0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06)],
             [(0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)],
             [(0.12, 0.04), (0.12, 0.05), (0.12, 0.04), (0.12, 0.04), (0.12, 0.05)],
             [(0.005, 0.005), (0.005, 0.005), (0., 0.), (0., 0.), (0., 0.)],
             [(0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035)],
             [(0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.03, 0.03)]]]

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, "histoSigmaCorr_all_years_MB")


# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS, REL_SYST, f"Corrected cross section", LEGEND_TITLES,
        "#it{p}_{T} (GeV/#it{c})", "#sigma (D_{s}) #times BR(D_{s} #rightarrow KK#pi)",
        SAVE_PATH, colors=COLORS, add_to_syst=ERRS)



#############################################################################
##################### NOW ADD HM AND DO ANOTHER PLOT  #######################
#############################################################################

LEGEND_TITLES = ["0 < n_{trkl} < #infty (MB)", "1 < n_{trkl} < 9 (MB)", "10 < n_{trkl} < 29 (MB)",
                 "30 < n_{trkl} < 59 (MB)", "60 < n_{trkl} < 99 (HM)"]

COLORS = [kBlue, kGreen + 2, kRed - 2, kAzure + 3, kOrange + 7]


# Append the HM histogram
HISTO = extract_histo_or_error(CASE, ANA_HM, 4, YEAR_NUMBER, "histoSigmaCorr")
HISTO.SetName(f"{HISTO.GetName()}_4")
HISTOS.append(HISTO)

# Relative systematic uncertainties
# Added to the list already obtained for MB from above. Additional comments are given there.
REL_SYST.append([[(0.1, 0.1), (0.04, 0.04), (0.02, 0.02), (0.05, 0.05), (0.07, 0.07)],
                 [(0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06), (0.06, 0.06)],
                 [(0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)],
                 [(0.12, 0.04), (0.12, 0.05), (0.12, 0.04), (0.12, 0.04), (0.12, 0.05)],
                 [(0.1, 0.1), (0.1, 0.1), (0.1, 0.1), (0.1, 0.1), (0.1, 0.1)],
                 [(0.01, 0.01), (0.01, 0.01), (0., 0.), (0., 0.), (0., 0.)],
                 [(0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035), (0.035, 0.035)],
                 [(0.03, 0.1), (0.03, 0.1), (0.03, 0.1), (0.03, 0.1), (0.03, 0.1)]])

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, "histoSigmaCorr_all_years_MB_HM")

results(HISTOS, REL_SYST, f"Corrected cross section", LEGEND_TITLES,
        "#it{p}_{T} (GeV/#it{c})", "#sigma (D_{s}) #times BR(D_{s} #rightarrow KK#pi)",
        SAVE_PATH, colors=COLORS)
