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
from ROOT import TFile, gStyle, gROOT, TH1F, TGraphAsymmErrors, TH1
from ROOT import kBlue, kAzure, kOrange, kGreen, kBlack, kRed, kWhite
from ROOT import Double
from machine_learning_hep.utilities_plot import plot_histograms, save_histograms, Errors
from machine_learning_hep.utilities_plot import calc_systematic_multovermb
from machine_learning_hep.utilities_plot import divide_all_by_first_multovermb
from machine_learning_hep.utilities_plot import divide_by_eachother, divide_by_eachother_barlow
from machine_learning_hep.utilities_plot import calc_systematic_mesonratio
from machine_learning_hep.utilities_plot import calc_systematic_mesondoubleratio

def results(histos_central, systematics, title, legend_titles, x_label, y_label,
            save_path, ratio, **kwargs):

    if systematics and (len(histos_central) != len(systematics)):
        print(f"Number of systematics {len(systematics)} differs from number of " \
              f"histograms {len(histos_central)}")
        return

    if len(histos_central) != len(legend_titles):
        print(f"Number of legend titles {len(legend_titles)} differs from number of " \
              f"histograms {len(histos_central)}")
        return

    colors = kwargs.get("colors", [kRed - i for i in range(len(histos_central))])
    if len(histos_central) != len(colors):
        print(f"Number of colors {len(colors)} differs from number of " \
              f"histograms {len(histos_central)}")
        return

    y_range = kwargs.get("y_range", [1e-8, 1])

    markerstyles = [1] * len(histos_central) + [20] * len(histos_central) \
            if systematics else [20] * len(histos_central)
    draw_options = ["E2"] * len(histos_central) + [""] * len(histos_central) \
            if systematics else [""] * len(histos_central)
    if systematics:
        colors = colors * 2
        legend_titles = [None] * len(histos_central) + legend_titles
        histos_central = systematics + histos_central

    if ratio is False:
        plot_histograms(histos_central, kwargs.get("log_y", True), [False, False, y_range],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])
    elif ratio == 1:
        plot_histograms(histos_central, kwargs.get("log_y", True), [True, False, y_range],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])
    elif ratio == 2:
        plot_histograms(histos_central, kwargs.get("log_y", False), [False, True, [0, 3.1]],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])
    elif ratio == 3:
        plot_histograms(histos_central, kwargs.get("log_y", False), [False, True, [0, 3.1]],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])
    else:
        plot_histograms(histos_central, kwargs.get("log_y", False), [False, True, [0, 1.0]],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])

def get_param(case):
    # First check if that is already a valid path
    if not os.path.exists(case):
        # if not, make a guess for the filepath
        case = f"data/database_ml_parameters_{case}.yml"

    data_param = None
    with open(case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    case = list(data_param.keys())[0]
    data_param = data_param[case]
    return case, data_param

def extract_histo_or_error(case, ana_type, mult_bin, period_number, histo_name, filepath=None):
    case, data_param = get_param(case)
    if filepath is None:
        filepath = data_param["analysis"][ana_type]["data"]["resultsallp"]
    if period_number >= 0:
        filepath = data_param["analysis"][ana_type]["data"]["results"][period_number]

    path = f"{filepath}/finalcross{case}{ana_type}mult{mult_bin}.root"
    in_file = TFile.Open(path, "READ")
    histo = in_file.Get(histo_name)
    if not histo or not isinstance(histo, TH1):
        print(f"Cannot read histogram {histo_name} from path {path}")
        return None
    histo.SetDirectory(0)
    return histo

def make_standard_save_path(case, prefix):
    _, data_param = get_param(case)
    folder_plots = data_param["analysis"]["dir_general_plots"]
    folder_plots = f"{folder_plots}/final"
    if not os.path.exists(folder_plots):
        print("creating folder ", folder_plots)
        os.makedirs(folder_plots)
    return f"{folder_plots}/{prefix}.eps"

#############################################################
gROOT.SetBatch(True)

ANA_MB = "MBvspt_perc_v0m"

LEGEND_TITLES = ["#kern[0.5]{0.0} #leq V0M_{percentile} #leq 100.0 (MB)",
                 "30.0 #kern[0.3]{#leq} V0M_{percentile} #leq 100.0 (MB)",
                 "#kern[0.5]{0.1} #kern[0.3]{#leq} V0M_{percentile} #leq 30.0 (MB)"]

COLORS = [kBlue, kGreen + 2, kRed - 2]

# Histograms to be studied in HFPtSpectrum files
HFPT_SPECTRUM_NAMES = ["histoSigmaCorr", "hDirectEffpt", "hFeedDownEffpt", "hRECpt",
                       "histoYieldCorr"]
HFPT_SPECTRUM_RANGES = [(10, 10e8), (10e-8, 1), (10e-8, 1), (10, 10e8), (10, 10e8)]

# Loop over histograms
for hn, yr in zip(HFPT_SPECTRUM_NAMES, HFPT_SPECTRUM_RANGES):

    HISTOS_D0 = []
    HISTOS_LC = []

    # pp X-Sec and branching ratios
    SIGMAV0 = 57.8e9
    BRLC = 0.0109
    BRD0 = 0.0389


    DB_PATH_D0 = "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/data/" \
            "data_prod_20200304/database_ml_parameters_D0pp_zg_0304.yml"
    DB_PATH_LC = "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/data/" \
            "data_prod_20200304/database_ml_parameters_LcpK0spp_20200301.yml"

    YEAR_NUMBER = -1 # -1 refers to all years merged

    # Run over multiplicity bins (hard coded for now), 0th bin is V0M percentile
    # [0, 100] -> integrated
    for mb in range(3):
        histo_lc = extract_histo_or_error(DB_PATH_LC, ANA_MB, mb, YEAR_NUMBER, hn)
        histo_lc.SetName(f"{histo_lc.GetName()}_{mb}")
        histo_d0 = extract_histo_or_error(DB_PATH_D0, ANA_MB, mb, YEAR_NUMBER, hn)
        histo_d0.SetName(f"{histo_d0.GetName()}_{mb}")

        histo_lc.Scale(1./SIGMAV0)
        histo_lc.Scale(1./BRLC)
        histo_d0.Scale(1./SIGMAV0)
        histo_d0.Scale(1./BRD0)
        HISTOS_LC.append(histo_lc)
        HISTOS_D0.append(histo_d0)


    ##################
    # Lc to D0 ratio #
    ##################
    if hn == "histoSigmaCorr":

        # X-Sec for D0 and Lc for all mult bins

        SAVE_PATH = f"plot_D0pp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_D0, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                SAVE_PATH, False, colors=COLORS)
        SAVE_PATH = f"plot_LcpK0spp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_LC, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                SAVE_PATH, False, colors=COLORS)


        # Ratio Lc / D0 for all mult bins

        # For now strip the first bin of Lc by hand. Introduce funtion in the future
        # This is necessary as Lc spectrum goes down to 1 - 2 whereas D0 starts at 2 - 4
        HISTOS_LC_STRIP = []

        for mb in range(3):
            histo_lc_new = HISTOS_D0[mb].Clone(f"{HISTOS_LC[mb].GetName()}_lc_over_d0")
            histo_lc_new.Reset("ICEMS")
            histo_lc_old = HISTOS_LC[mb]
            for i in range(1, histo_lc_new.GetNbinsX() + 1):
                histo_lc_new.SetBinContent(i, histo_lc_old.GetBinContent(i+1))
                histo_lc_new.SetBinError(i, histo_lc_old.GetBinError(i+1))
            HISTOS_LC_STRIP.append(histo_lc_new)


        HISTOS_LC_OVER_D0 = divide_by_eachother(HISTOS_LC_STRIP, HISTOS_D0, [1, 1])
        SAVE_PATH = f"plot_LcpK0spp_over_D0pp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_LC_OVER_D0, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                SAVE_PATH, False, colors=COLORS, y_range=[0, 1], log_y=False)


    ##################
    # COMPARE TO STD #
    ##################

    cross_section_std_top_dir = "../ALICE_D2H_vs_mult_pp13/data/PreliminaryQM19"
    cross_section_std_D0 = f"{cross_section_std_top_dir}/finalcrossD0ppMBvspt_ntrklmult0.root"

    file_std = TFile.Open(cross_section_std_D0, "READ")
    # This is the TRUE STD HFPtSpectrum file
    # The first pT bin is 1 - 2 and it has to be stripped in the following
    histo_std_d0 = file_std.Get(hn)

    # Clone from the MLHEP binned D0 histogram (starting with pT bin 2 - 4)
    histo_std_d0_new = HISTOS_D0[0].Clone(f"{HISTOS_D0[mb].GetName()}_over_std_d0")
    histo_std_d0_new.Reset("ICEMS")
    for i in range(1, histo_std_d0_new.GetNbinsX() + 1):
        histo_std_d0_new.SetBinContent(i, histo_std_d0.GetBinContent(i+1))
        histo_std_d0_new.SetBinError(i, histo_std_d0.GetBinError(i+1))
    # We also clone another time because we have to scale back the pp X-Sec and branching ratio for
    # an apple-to-apple comparison (we did the inverse scaling above before...)
    histo_ml_d0_new = HISTOS_D0[0].Clone(f"{HISTOS_D0[mb].GetName()}_ml_over_std_d0")
    histo_ml_d0_new.Scale(SIGMAV0 * BRD0)

    # Now, histo_std_d0_new is the histogram from the STD analysis starting with pT bin 2-4
    # histo_ml_d0_new is the corresponding histogram from the MLHEP analysis package in STD mode

    SAVE_PATH = f"plot_D0pp_ML__year_{YEAR_NUMBER}__over_D0pp_STD_{hn}.eps"

    results([histo_std_d0_new, histo_ml_d0_new], None, "", ["STD", "ML"], "#it{p}_{T} (GeV/#it{c})",
            "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
            SAVE_PATH, 1, colors=[kBlue, kRed + 2], y_range=yr)
