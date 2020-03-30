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
import sys
import os
# pylint: disable=import-error, no-name-in-module, unused-import
from array import array
import yaml
from ROOT import TFile, gStyle, gROOT, TH1F, TGraphAsymmErrors, TH1
from ROOT import kBlue, kAzure, kOrange, kGreen, kBlack, kRed, kWhite, kViolet, kYellow
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


def get_total_eff(case, ana_type, mult_bin, period_number):
    case, data_param = get_param(case)
    if period_number >= 0:
        filepath = data_param["analysis"][ana_type]["mc"]["results"][period_number]
    else:
        filepath = data_param["analysis"][ana_type]["mc"]["resultsallp"]

    path = f"{filepath}/efficiencies{case}{ana_type}.root"
    in_file = TFile.Open(path, "READ")
    histo_name = f"eff_mult{mult_bin}"
    histo = in_file.Get(histo_name)
    if not histo or not isinstance(histo, TH1):
        print(f"Cannot read histogram {histo_name} from path {path}")
        return None
    histo.SetDirectory(0)
    return histo


def get_raw_yields(case, ana_type, mult_bin, period_number):
    case, data_param = get_param(case)
    if period_number >= 0:
        filepath = data_param["analysis"][ana_type]["data"]["results"][period_number]
    else:
        filepath = data_param["analysis"][ana_type]["data"]["resultsallp"]

    path = f"{filepath}/Yields_nprongs_{ana_type}.root"
    in_file = TFile.Open(path, "READ")
    histo_name = f"hyields{mult_bin}"
    histo = in_file.Get(histo_name)
    if not histo or not isinstance(histo, TH1):
        print(f"Cannot read histogram {histo_name} from path {path}")
        return None
    histo.SetDirectory(0)
    return histo

def get_signifs(case, ana_type, mult_bin, period_number):
    case, data_param = get_param(case)
    if period_number >= 0:
        filepath = data_param["analysis"][ana_type]["data"]["results"][period_number]
    else:
        filepath = data_param["analysis"][ana_type]["data"]["resultsallp"]

    path = f"{filepath}/Significances_nprongs_{ana_type}.root"
    in_file = TFile.Open(path, "READ")
    histo_name = f"hsignifs{mult_bin}"
    histo = in_file.Get(histo_name)
    if not histo or not isinstance(histo, TH1):
        print(f"Cannot read histogram {histo_name} from path {path}")
        return None
    histo.SetDirectory(0)
    return histo

def get_backgrounds(case, ana_type, mult_bin, period_number):
    case, data_param = get_param(case)
    if period_number >= 0:
        filepath = data_param["analysis"][ana_type]["data"]["results"][period_number]
    else:
        filepath = data_param["analysis"][ana_type]["data"]["resultsallp"]

    path = f"{filepath}/Backgrounds_nprongs_{ana_type}.root"
    in_file = TFile.Open(path, "READ")
    histo_name = f"hbkgs{mult_bin}"
    histo = in_file.Get(histo_name)
    if not histo or not isinstance(histo, TH1):
        print(f"Cannot read histogram {histo_name} from path {path}")
        return None
    histo.SetDirectory(0)
    return histo

def get_sigmas(case, ana_type, mult_bin, period_number, data_mc="data"):
    case, data_param = get_param(case)
    if period_number >= 0:
        filepath = data_param["analysis"][ana_type]["data"]["results"][period_number]
    else:
        filepath = data_param["analysis"][ana_type]["data"]["resultsallp"]

    if mult_bin >= 0 and data_mc == "mc":
        print("ERROR: No MC histogram of sigmas available for specific mult bin in MC")
        sys.exit(1)

    path = f"{filepath}/Sigmas_nprongs_{ana_type}.root"
    histo_name = f"hsigmas{mult_bin}"
    if mult_bin < 0:
        path = f"{filepath}/Sigmas_mult_int_nprongs_{ana_type}.root"
        histo_name = f"hsigmas_init_{data_mc}"
    in_file = TFile.Open(path, "READ")
    histo = in_file.Get(histo_name)

    if not histo or not isinstance(histo, TH1):
        print(f"Cannot read histogram {histo_name} from path {path}")
        return None
    histo.SetDirectory(0)
    return histo

def get_means(case, ana_type, mult_bin, period_number, data_mc="data"):
    case, data_param = get_param(case)
    if period_number >= 0:
        filepath = data_param["analysis"][ana_type]["data"]["results"][period_number]
    else:
        filepath = data_param["analysis"][ana_type]["data"]["resultsallp"]

    if mult_bin >= 0 and data_mc == "mc":
        print("ERROR: No MC histogram of sigmas available for specific mult bin in MC")
        sys.exit(1)

    path = f"{filepath}/Means_nprongs_{ana_type}.root"
    histo_name = f"hmeanss{mult_bin}"
    if mult_bin < 0:
        path = f"{filepath}/Means_mult_int_nprongs_{ana_type}.root"
        histo_name = f"hmeanss_init_{data_mc}"
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

def strip_last_bin(histo):
    axis = histo.GetXaxis()
    edges_new = array("d", [axis.GetBinLowEdge(i) for i in range(1, histo.GetNbinsX() + 1)])
    histo_new = type(histo)(histo.GetName(), histo.GetTitle(), histo.GetNbinsX() - 1, edges_new)
    for i in range(1, histo_new.GetNbinsX() + 1):
        histo_new.SetBinContent(i, histo.GetBinContent(i))
        histo_new.SetBinError(i, histo.GetBinError(i))
    histo_new.SetDirectory(0)
    return histo_new





#############################################################
gROOT.SetBatch(True)
# pp X-Sec and branching ratios
SIGMAV0 = 57.8e9
BRLC = 0.0109
BRD0 = 0.0389


ANA_MB = "MBvspt_perc_v0m"

LEGEND_TITLES = ["#kern[0.5]{0.0} #leq V0M_{percentile} #leq 100.0 (MB)",
                 "30.0 #kern[0.3]{#leq} V0M_{percentile} #leq 100.0 (MB)",
                 "#kern[0.5]{0.1} #kern[0.3]{#leq} V0M_{percentile} #leq 30.0 (MB)"]

COLORS = [kBlue, kGreen + 2, kRed - 2]

DIR = "nominal"
DB_PATH_D0 = "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/" \
        "MBvspt_perc_v0m/database_ml_parameters_D0pp_zg_0304.yml"
DB_PATH_LC_NOMINAL = "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/" \
        "MBvspt_perc_v0m/database_ml_parameters_LcpK0spp_20200301_nominal.yml"
DB_PATH_LC_SIGMA_DATA = "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/" \
    "MBvspt_perc_v0m/database_ml_parameters_LcpK0spp_20200301_sigma_data.yml"
DB_PATH_LC_SIGMA_FREE = "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/" \
        "MBvspt_perc_v0m/database_ml_parameters_LcpK0spp_20200301_sigma_free.yml"

DB_PATHS_LC = [DB_PATH_LC_NOMINAL, DB_PATH_LC_SIGMA_DATA, DB_PATH_LC_SIGMA_FREE]

if DIR and not os.path.exists(DIR):
    os.makedirs(DIR)

####################################
#                                  #
# LcpK0spp raw yield summary plots #
#                                  #
####################################

# Make raw yield plots Lc
HISTOS_ = [strip_last_bin(get_raw_yields(DB_PATH_LC_NOMINAL, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/yields_LcpK0spp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "raw yield d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (3#sigma)",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 15000], log_y=False)


# Make sigma plots Lc of final fits
HISTOS_ = [strip_last_bin(get_sigmas(DB_PATH_LC_NOMINAL, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/sigmas_LcpK0spp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#sigma_{fit} [GeV/#it{c}^{2}]",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 0.05], log_y=False)

# Make init sigma plots Lc
HISTOS_ = [strip_last_bin(get_sigmas(DB_PATH_LC_NOMINAL, ANA_MB, -1, -1, data_mc)) \
        for data_mc in ("data", "mc")]
SAVE_PATH = f"{DIR}/sigmas_init_LcpK0spp_{ANA_MB}.eps"
results(HISTOS_, None, "", ["data", "MC"], "#it{p}_{T} (GeV/#it{c})",
        "#sigma_{fit} [GeV/#it{c}^{2}]",
        SAVE_PATH, False, colors=COLORS[:2], y_range=[0, 0.05], log_y=False)


# Make means plots Lc of final fits
HISTOS_ = [strip_last_bin(get_means(DB_PATH_LC_NOMINAL, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/means_LcpK0spp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#mu_{fit} [GeV/#it{c}^{2}]",
        SAVE_PATH, False, colors=COLORS, y_range=[2.275, 2.305], log_y=False)

# Make significace plots Lc of final fits
HISTOS_ = [strip_last_bin(get_signifs(DB_PATH_LC_NOMINAL, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/signifs_LcpK0spp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "Significance (3#sigma)",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 20], log_y=False)

# Make backgrounds plots Lc of final fits
HISTOS_ = [strip_last_bin(get_backgrounds(DB_PATH_LC_NOMINAL, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/backgrounds_LcpK0spp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "background d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (3#sigma)",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 20000], log_y=False)

# Make sigmas, means, significancies, raw yield, backgrounds for different fit scenarios in Lc
if DIR == "nominal":
    LEG_SIGMA_COMP = ["#sigma fixed to MC", "#sigma fixed to data", "#sigma free"]
    for mb in range(3):
        HISTOS_ = [strip_last_bin(get_raw_yields(path, ANA_MB, mb, -1)) for path in DB_PATHS_LC]
        SAVE_PATH = f"{DIR}/yields_sigma_opt_LcpK0spp_{ANA_MB}_mult_{mb}.eps"
        plot_histograms(HISTOS_, False, [False, False, [0, 15000]],
                        LEG_SIGMA_COMP, LEGEND_TITLES[mb], "#it{p}_{T} (GeV/#it{c})",
                        "raw yield d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (3#sigma)",
                        "", SAVE_PATH, linestyles=[1, 7, 10],
                        markerstyles=[22, 23, 8], colors=COLORS, linewidths=[1],
                        fillstyles=[0])
        HISTOS_ = [strip_last_bin(get_backgrounds(path, ANA_MB, mb, -1)) for path in DB_PATHS_LC]
        SAVE_PATH = f"{DIR}/backgrounds_sigma_opt_LcpK0spp_{ANA_MB}_mult_{mb}.eps"
        plot_histograms(HISTOS_, False, [False, False, [0, 100000]],
                        LEG_SIGMA_COMP, LEGEND_TITLES[mb], "#it{p}_{T} (GeV/#it{c})",
                        "background d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (3#sigma)",
                        "", SAVE_PATH, linestyles=[1, 7, 10],
                        markerstyles=[22, 23, 8], colors=COLORS, linewidths=[1],
                        fillstyles=[0])
        HISTOS_ = [strip_last_bin(get_signifs(path, ANA_MB, mb, -1)) for path in DB_PATHS_LC]
        SAVE_PATH = f"{DIR}/signifs_sigma_opt_LcpK0spp_{ANA_MB}_mult_{mb}.eps"
        plot_histograms(HISTOS_, False, [False, False, [0, 20]],
                        LEG_SIGMA_COMP, LEGEND_TITLES[mb], "#it{p}_{T} (GeV/#it{c})",
                        "Significance (3#sigma)",
                        "", SAVE_PATH, linestyles=[1, 7, 10],
                        markerstyles=[22, 23, 8], colors=COLORS, linewidths=[1],
                        fillstyles=[0])
        HISTOS_ = [strip_last_bin(get_sigmas(path, ANA_MB, mb, -1)) for path in DB_PATHS_LC]
        SAVE_PATH = f"{DIR}/sigmas_sigma_opt_LcpK0spp_{ANA_MB}_mult_{mb}.eps"
        plot_histograms(HISTOS_, False, [False, False, [0, 0.05]],
                        LEG_SIGMA_COMP, LEGEND_TITLES[mb], "#it{p}_{T} (GeV/#it{c})",
                        "#sigma_{fit} [GeV/#it{c}^{2}]",
                        "", SAVE_PATH, linestyles=[1, 7, 10],
                        markerstyles=[22, 23, 8], colors=COLORS, linewidths=[1],
                        fillstyles=[0])
        HISTOS_ = [strip_last_bin(get_means(path, ANA_MB, mb, -1)) for path in DB_PATHS_LC]
        SAVE_PATH = f"{DIR}/means_sigma_opt_LcpK0spp_{ANA_MB}_mult_{mb}.eps"
        plot_histograms(HISTOS_, False, [False, False, [2.275, 2.305]],
                        LEG_SIGMA_COMP, LEGEND_TITLES[mb], "#it{p}_{T} (GeV/#it{c})",
                        "#mu_{fit} [GeV/#it{c}^{2}]",
                        "", SAVE_PATH, linestyles=[1, 7, 10],
                        markerstyles=[22, 23, 8], colors=COLORS, linewidths=[1],
                        fillstyles=[0])


# Compare sigmas and means pear year to all years merged in Lc




# SIGMAS
YEARS_DATA_MC = [(-1, "mc"), (-1, "data"), (0, "mc"), (0, "data"), (1, "mc"), (1, "data"),
                 (2, "mc"), (2, "data")]
LABELS_DATA_MC = ["years merged, MC (ref.)", "years merged, data", "2016, MC", "2016, data",
                  "2017, MC", "2017, data", "2018, MC", "2018, data"]
HISTOS_YEARS_SIGMAS = [strip_last_bin(get_sigmas(DB_PATHS_LC[0], ANA_MB, -1, year, mc_data)) \
        for year, mc_data in YEARS_DATA_MC]
SAVE_PATH = f"{DIR}/sigmas_init_years_comp_LcpK0spp_{ANA_MB}.eps"
plot_histograms(HISTOS_YEARS_SIGMAS, False, [True, False, [0.005, 0.02]],
                LABELS_DATA_MC, "", "#it{p}_{T} (GeV/#it{c})", "#sigma_{fit} [GeV/#it{c}^{2}]",
                "", SAVE_PATH, linestyles=[1, 7, 10],
                markerstyles=[22, 23, 8], colors=[kBlue, kOrange+2, kGreen+2, kRed-2, kAzure+1,
                                                  kViolet+2, kYellow-3], linewidths=[1],
                fillstyles=[0])

# MEANS
YEAR_LEGEND = ["years merged", "2016", "2017", "2018"]
for mb in range(3):
    HISTOS_YEARS_MEANS = [strip_last_bin(get_means(DB_PATHS_LC[0], ANA_MB, mb, year)) \
            for year in [-1, 0, 1, 2]]
    SAVE_PATH = f"{DIR}/means_init_years_comp_LcpK0spp_{ANA_MB}_mult_{mb}.eps"
    plot_histograms(HISTOS_YEARS_MEANS, False, [True, False, [2.275, 2.305]],
                    YEAR_LEGEND, LEGEND_TITLES[mb], "#it{p}_{T} (GeV/#it{c})",
                    "#mu_{fit} [GeV/#it{c}^{2}]",
                    "", SAVE_PATH, linestyles=[1, 7, 10],
                    markerstyles=[22, 23, 8], colors=[kBlue, kOrange+2, kGreen+2, kRed-2,
                                                      kAzure+1, kYellow-3],
                    linewidths=[1], fillstyles=[0])


for mb in range(3):
    HISTOS_YEARS_CORR_YIELDS = \
            [strip_last_bin( \
            extract_histo_or_error(DB_PATHS_LC[0], ANA_MB, mb, year, "histoSigmaCorr")) \
            for year in [-1, 0, 1, 2]]
    for h in HISTOS_YEARS_CORR_YIELDS:
        h.Scale(1./SIGMAV0)
        h.Scale(1./BRLC)
    SAVE_PATH = f"{DIR}/histoSigmaCorr_init_years_comp_LcpK0spp_{ANA_MB}_mult_{mb}.eps"
    plot_histograms(HISTOS_YEARS_CORR_YIELDS, True, [True, False, [1e-8, 1]],
                    YEAR_LEGEND, LEGEND_TITLES[mb], "#it{p}_{T} (GeV/#it{c})",
                    "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                    "year / alle years", SAVE_PATH, linestyles=[1, 7, 10],
                    markerstyles=[22, 23, 8], colors=COLORS,
                    linewidths=[1], fillstyles=[0])


####################################
#                                  #
# D0pp raw yield summary plots     #
#                                  #
####################################



# Make raw yield plots D0
HISTOS_ = [strip_last_bin(get_raw_yields(DB_PATH_D0, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/yields_D0pp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "raw yield d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (3#sigma)",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 30000], log_y=False)


# Make sigma plots Lc of final fits
HISTOS_ = [strip_last_bin(get_sigmas(DB_PATH_D0, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/sigmas_D0pp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#sigma_{fit} [GeV/#it{c}^{2}]",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 0.025], log_y=False)

# Make init sigma plots Lc
HISTOS_ = [strip_last_bin(get_sigmas(DB_PATH_D0, ANA_MB, -1, -1, data_mc)) \
        for data_mc in ("data", "mc")]
SAVE_PATH = f"{DIR}/sigmas_init_D0pp_{ANA_MB}.eps"
results(HISTOS_, None, "", ["data", "MC"], "#it{p}_{T} (GeV/#it{c})",
        "#sigma_{fit} [GeV/#it{c}^{2}]",
        SAVE_PATH, False, colors=COLORS[:2], y_range=[0, 0.025], log_y=False)


# Make sigma plots Lc of final fits
HISTOS_ = [strip_last_bin(get_means(DB_PATH_D0, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/means_D0pp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#mu_{fit} [GeV/#it{c}^{2}]",
        SAVE_PATH, False, colors=COLORS, y_range=[2.275, 2.305], log_y=False)

# Make significace plots Lc of final fits
HISTOS_ = [strip_last_bin(get_signifs(DB_PATH_D0, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/signifs_D0pp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "Significance (3#sigma)",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 100], log_y=False)

# Make backgrounds plots Lc of final fits
HISTOS_ = [strip_last_bin(get_backgrounds(DB_PATH_D0, ANA_MB, mb, -1)) for mb in range(3)]
SAVE_PATH = f"{DIR}/backgrounds_D0pp_{ANA_MB}.eps"
results(HISTOS_, None, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "background d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (3#sigma)",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 20000], log_y=False)

################
# Efficiencies #
################

HISTOS_EFF_LC = [strip_last_bin(get_total_eff(DB_PATH_LC_NOMINAL, ANA_MB, mb, -1)) \
        for mb in range(3)]
EFFS_ERR_LC = [ \
        Errors.make_root_asymm(\
        h, Errors(h.GetNbinsX()).get_total_for_spectra_plot(), const_x_err=0.3) \
        for h in HISTOS_EFF_LC]
for ierr, e in enumerate(EFFS_ERR_LC):
    e.SetName(f"gr_TotSyst_{ierr}")
SAVE_PATH = f"{DIR}/EffAcc_LcpK0spp_{ANA_MB}.eps"
results(HISTOS_EFF_LC, EFFS_ERR_LC, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "Eff #times Acc (#Lambda_{c}^{+})",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 1.], log_y=False)

HISTOS_EFF_D0 = [strip_last_bin(get_total_eff(DB_PATH_D0, ANA_MB, mb, -1)) for mb in range(3)]
EFFS_ERR_D0 = [ \
        Errors.make_root_asymm( \
        h, Errors(h.GetNbinsX()).get_total_for_spectra_plot(), const_x_err=0.3) \
        for h in HISTOS_EFF_D0]
for ierr, e in enumerate(EFFS_ERR_D0):
    e.SetName(f"gr_TotSyst_{ierr}")
SAVE_PATH = f"{DIR}/EffAcc_D0pp_{ANA_MB}.eps"
results(HISTOS_EFF_D0, EFFS_ERR_D0, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "Eff #times Acc (D^{0})",
        SAVE_PATH, False, colors=COLORS, y_range=[0, 0.6], log_y=False)



# Histograms to be studied in HFPtSpectrum files
HFPT_SPECTRUM_NAMES = ["histoSigmaCorr", "hDirectEffpt", "hFeedDownEffpt", "hRECpt",
                       "histoYieldCorr"]
HFPT_SPECTRUM_RANGES = [(10, 10e8), (10e-8, 1), (10e-8, 1), (10, 10e8), (10, 10e8)]


ERR_FILES_LC = ["errors/LcpK0spp/histoSigmaCorr_0.yaml",
                "errors/LcpK0spp/histoSigmaCorr_1.yaml",
                "errors/LcpK0spp/histoSigmaCorr_2.yaml"]

ERR_FILES_D0 = ["errors/D0pp/histoSigmaCorr_0.yaml",
                "errors/D0pp/histoSigmaCorr_1.yaml",
                "errors/D0pp/histoSigmaCorr_2.yaml"]
# Loop over histograms
for hn, yr in zip(HFPT_SPECTRUM_NAMES, HFPT_SPECTRUM_RANGES):

    HISTOS_D0 = []
    HISTOS_LC = []
    ERRS_LC = []
    ERRS_D0 = []
    ERRS = []



    YEAR_NUMBER = -1 # -1 refers to all years merged

    # Run over multiplicity bins (hard coded for now), 0th bin is V0M percentile
    # [0, 100] -> integrated
    for mb in range(3):
        histo_lc = strip_last_bin(extract_histo_or_error(DB_PATH_LC_NOMINAL, ANA_MB, mb,
                                                         YEAR_NUMBER, hn))
        histo_lc.SetName(f"{histo_lc.GetName()}_{mb}")
        histo_d0 = strip_last_bin(extract_histo_or_error(DB_PATH_D0, ANA_MB, mb, YEAR_NUMBER, hn))
        histo_d0.SetName(f"{histo_d0.GetName()}_{mb}")

        HISTOS_LC.append(histo_lc)
        HISTOS_D0.append(histo_d0)

        if hn == "histoSigmaCorr":
            histo_lc.Scale(1./SIGMAV0)
            histo_lc.Scale(1./BRLC)
            histo_d0.Scale(1./SIGMAV0)
            histo_d0.Scale(1./BRD0)
            errs = Errors(histo_lc.GetNbinsX())
            errs.read(ERR_FILES_LC[mb])
            ERRS.append(errs)
            ERRS_LC.append(Errors.make_root_asymm(histo_lc, errs.get_total_for_spectra_plot(), \
                                              const_x_err=0.3))
            ERRS_LC[-1].SetName(f"gr_TotSyst_{mb}")

            errs = Errors(histo_d0.GetNbinsX())
            ERRS.append(errs)
            ERRS_D0.append(Errors.make_root_asymm(histo_d0, errs.get_total_for_spectra_plot(), \
                                              const_x_err=0.3))
            ERRS_D0[-1].SetName(f"gr_TotSyst_{mb}")


        if hn == "hDirectEffpt":
            histo_lc.GetYaxis().SetTitle("Acc #times Eff")
            histo_d0.GetYaxis().SetTitle("Acc #times Eff")
            errs = Errors(histo_lc.GetNbinsX())
            ERRS.append(errs)
            ERRS_LC.append(Errors.make_root_asymm(histo_lc, errs.get_total_for_spectra_plot(), \
                                              const_x_err=0.3))
            ERRS_LC[-1].SetName(f"gr_TotSyst_{mb}")

            errs = Errors(histo_d0.GetNbinsX())
            ERRS.append(errs)
            ERRS_D0.append(Errors.make_root_asymm(histo_d0, errs.get_total_for_spectra_plot(), \
                                              const_x_err=0.3))
            ERRS_D0[-1].SetName(f"gr_TotSyst_{mb}")



    ################
    # Efficiencies #
    ################
    if hn == "hDirectEffpt":
        # X-Sec for D0 and Lc for all mult bins
        SAVE_PATH = f"{DIR}/plot_D0pp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_D0, ERRS_D0, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                SAVE_PATH, False, colors=COLORS)
        # X-Sec for D0 and Lc for all mult bins
        SAVE_PATH = f"{DIR}/plot_LcpK0spp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_LC, ERRS_LC, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                SAVE_PATH, False, colors=COLORS)



    ##################
    # Lc to D0 ratio #
    ##################
    if hn == "histoSigmaCorr":


        # D0 vs D0 QM19
        QM_FILE_D0 = \
                TFile.Open("QM19/D0CorrectedYieldPerEvent_MBvspt_ntrkl_1999_19_1029_3059.root")
        histo_d0_qm19 = strip_last_bin(QM_FILE_D0.Get("histoSigmaCorr_0"))
        SAVE_PATH = f"{DIR}/plot_D0pp_vs_D0pp_QM19_{hn}.eps"
        results([histo_d0_qm19, HISTOS_D0[0]], None, "", ["QM19", "This analysis"],
                "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (D^{0}, mult. int.)",
                SAVE_PATH, 1, colors=[kBlue, kGreen + 2])
        QM_FILE_LC = \
                TFile.Open("QM19/LcpKpiCorrectedYieldPerEvent_MBvspt_ntrkl_1999_19_1029_3059.root")
        histo_lc_qm19 = strip_last_bin(QM_FILE_LC.Get("histoSigmaCorr_0"))
        SAVE_PATH = f"{DIR}/plot_LcpK0spp_vs_LcpKpipp_QM19_{hn}.eps"
        results([histo_lc_qm19, HISTOS_LC[0]], None, "", ["QM19", "This analysis"],
                "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c}) (#Lambda_{c}^{+}, mult. int.)",
                SAVE_PATH, 1, colors=[kBlue, kGreen + 2])


        # X-Sec for D0 and Lc for all mult bins
        SAVE_PATH = f"{DIR}/plot_D0pp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_D0, ERRS_D0, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                SAVE_PATH, False, colors=COLORS)
        SAVE_PATH = f"{DIR}/plot_LcpK0spp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_LC, ERRS_LC, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
                SAVE_PATH, False, colors=COLORS)


        # Ratio Lc / D0 for all mult bins


        ERRS_LC_OVER_D0 = []
        HISTOS_LC_OVER_D0 = divide_by_eachother(HISTOS_LC, HISTOS_D0, [1, 1])
        for ierr, (e, h) in enumerate(zip(ERR_FILES_D0, HISTOS_LC_OVER_D0)):
            errs = Errors(h.GetNbinsX())
            errs.read(e)
            ERRS.append(errs)
            ERRS_LC_OVER_D0.append(Errors.make_root_asymm(h, errs.get_total_for_spectra_plot(), \
                                              const_x_err=0.3))
            ERRS_LC_OVER_D0[-1].SetName(f"gr_TotSyst_{ierr}")
        SAVE_PATH = f"{DIR}/plot_LcpK0spp_over_D0pp_year_{YEAR_NUMBER}_{hn}.eps"
        results(HISTOS_LC_OVER_D0, ERRS_LC_OVER_D0, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
                "#sigma_{#Lambda_{c}^{+}} / #sigma_{D^{0}}",
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
    for ibin in range(1, histo_std_d0_new.GetNbinsX() + 1):
        histo_std_d0_new.SetBinContent(ibin, histo_std_d0.GetBinContent(ibin+1))
        histo_std_d0_new.SetBinError(ibin, histo_std_d0.GetBinError(ibin+1))
    # We also clone another time because we have to scale back the pp X-Sec and branching ratio for
    # an apple-to-apple comparison (we did the inverse scaling above before...)
    histo_ml_d0_new = HISTOS_D0[0].Clone(f"{HISTOS_D0[mb].GetName()}_ml_over_std_d0")
    histo_ml_d0_new.Scale(SIGMAV0 * BRD0)

    # Now, histo_std_d0_new is the histogram from the STD analysis starting with pT bin 2-4
    # histo_ml_d0_new is the corresponding histogram from the MLHEP analysis package in STD mode

    SAVE_PATH = f"{DIR}/plot_D0pp_ML__year_{YEAR_NUMBER}__over_D0pp_STD_{hn}.eps"

    results([histo_std_d0_new, histo_ml_d0_new], None, "", ["STD", "ML"], "#it{p}_{T} (GeV/#it{c})",
            "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
            SAVE_PATH, 1, colors=[kBlue, kRed + 2], y_range=yr)
