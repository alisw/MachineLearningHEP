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
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, gStyle, gROOT
from machine_learning_hep.utilities import plot_histograms

FILES_NOT_FOUND = []

# pylint: disable=too-many-branches
def compare_ml_std(case_ml, ana_type_ml, period_number, filepath_std, scale_std=None,
                   map_std_bins=None, mult_bin=None, ml_histo_names=None, std_histo_names=None,
                   suffix=""):

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


        save_path = \
                f"{folder_plots}/{hn_ml}_ml_std_mult_{mult_bin}_period_{period_number}{suffix}.eps"

        plot_histograms([histo_std, histo_ml], True, True, ["STD", "ML"], histo_ml.GetTitle(),
                        "#it{p}_{T} (GeV/#it{c}", histo_ml.GetYaxis().GetTitle(), "ML / STD",
                        save_path)

# pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-arguments
def compare_ml_std_ratio(case_ml_1, case_ml_2, ana_type_ml, period_number, filepath_std_1,
                         filepath_std_2, scale_std_1=None, scale_std_2=None, map_std_bins=None,
                         mult_bin=None, ml_histo_names=None, std_histo_names_1=None,
                         std_histo_names_2=None, suffix=""):

    with open("data/database_ml_parameters_%s.yml" % case_ml_1, 'r') as param_config:
        data_param_1 = yaml.load(param_config, Loader=yaml.FullLoader)
    with open("data/database_ml_parameters_%s.yml" % case_ml_2, 'r') as param_config:
        data_param_2 = yaml.load(param_config, Loader=yaml.FullLoader)
    if period_number < 0:
        filepath_ml_1 = data_param_1[case_ml_1]["analysis"][ana_type_ml]["data"]["resultsallp"]
        filepath_ml_2 = data_param_2[case_ml_2]["analysis"][ana_type_ml]["data"]["resultsallp"]
    else:
        filepath_ml_1 = \
                data_param_1[case_ml_1]["analysis"][ana_type_ml]["data"]["results"][period_number]
        filepath_ml_2 = \
                data_param_2[case_ml_2]["analysis"][ana_type_ml]["data"]["results"][period_number]

    name_1 = data_param_1[case_ml_1]["analysis"][ana_type_ml]["latexnamemeson"]
    name_2 = data_param_2[case_ml_2]["analysis"][ana_type_ml]["latexnamemeson"]
    # Get pt spectrum files
    if mult_bin is None:
        mult_bin = 0
    path_ml_1 = f"{filepath_ml_1}/finalcross{case_ml_1}{ana_type_ml}mult{mult_bin}.root"
    path_ml_2 = f"{filepath_ml_2}/finalcross{case_ml_2}{ana_type_ml}mult{mult_bin}.root"
    if not os.path.exists(path_ml_1):
        FILES_NOT_FOUND.append(path_ml_1)
        return
    if not os.path.exists(path_ml_2):
        FILES_NOT_FOUND.append(path_ml_2)
        return

    file_ml_1 = TFile.Open(path_ml_1, "READ")
    file_ml_2 = TFile.Open(path_ml_2, "READ")
    file_std_1 = TFile.Open(filepath_std_1, "READ")
    file_std_2 = TFile.Open(filepath_std_2, "READ")

    # Collect histo names to quickly loop later
    histo_names = ["hDirectMCpt", "hFeedDownMCpt", "hDirectMCptMax", "hDirectMCptMin",
                   "hFeedDownMCptMax", "hFeedDownMCptMin", "hDirectEffpt", "hFeedDownEffpt",
                   "hRECpt", "histoYieldCorr", "histoYieldCorrMax", "histoYieldCorrMin",
                   "histoSigmaCorr", "histoSigmaCorrMax", "histoSigmaCorrMin"]

    if ml_histo_names is None:
        ml_histo_names = histo_names
    if std_histo_names_1 is None:
        std_histo_names_1 = histo_names
    if std_histo_names_2 is None:
        std_histo_names_2 = histo_names

    for hn_ml, hn_std_1, hn_std_2 in zip(ml_histo_names, std_histo_names_1, std_histo_names_2):
        histo_ml_1 = file_ml_1.Get(hn_ml)
        histo_ml_2 = file_ml_2.Get(hn_ml)
        histo_std_tmp_1 = file_std_1.Get(hn_std_1)
        histo_std_tmp_2 = file_std_2.Get(hn_std_2)
        histo_std_1 = None
        histo_std_2 = None

        if not histo_ml_1:
            print(f"Could not find histogram {hn_ml}, continue...")
            continue
        if not histo_ml_2:
            print(f"Could not find histogram {hn_ml}, continue...")
            continue
        if not histo_std_tmp_1:
            print(f"Could not find histogram {hn_std_1}, continue...")
            continue
        if not histo_std_tmp_2:
            print(f"Could not find histogram {hn_std_2}, continue...")
            continue

        if "MC" not in hn_ml and map_std_bins is not None:
            histo_std_1 = histo_ml_1.Clone("std_rebin_1")
            histo_std_1.Reset("ICESM")
            histo_std_2 = histo_ml_2.Clone("std_rebin_2")
            histo_std_2.Reset("ICESM")

            contents = [0] * histo_ml_1.GetNbinsX()
            errors = [0] * histo_ml_1.GetNbinsX()

            for ml_bin, std_bins in map_std_bins:
                for b in std_bins:
                    contents[ml_bin-1] += histo_std_tmp_1.GetBinContent(b) / len(std_bins)
                    errors[ml_bin-1] += \
                            histo_std_tmp_1.GetBinError(b) * histo_std_tmp_1.GetBinError(b)

            for b in range(histo_std_1.GetNbinsX()):
                histo_std_1.SetBinContent(b+1, contents[b])
                histo_std_1.SetBinError(b+1, sqrt(errors[b]))

            contents = [0] * histo_ml_2.GetNbinsX()
            errors = [0] * histo_ml_2.GetNbinsX()

            for ml_bin, std_bins in map_std_bins:
                for b in std_bins:
                    contents[ml_bin-1] += histo_std_tmp_2.GetBinContent(b) / len(std_bins)
                    errors[ml_bin-1] += \
                            histo_std_tmp_2.GetBinError(b) * histo_std_tmp_2.GetBinError(b)

            for b in range(histo_std_2.GetNbinsX()):
                histo_std_2.SetBinContent(b+1, contents[b])
                histo_std_2.SetBinError(b+1, sqrt(errors[b]))

        else:
            histo_std_1 = histo_std_tmp_1.Clone("std_cloned_1")
            histo_std_2 = histo_std_tmp_2.Clone("std_cloned_2")

        if scale_std_1 is not None:
            histo_std_1.Scale(scale_std_1)
        if scale_std_2 is not None:
            histo_std_2.Scale(scale_std_2)

        histo_ratio_ml = histo_ml_1.Clone("{histo_ml_1.GetName()}_ratio")
        histo_ratio_ml.Divide(histo_ml_2)
        histo_ratio_std = histo_std_1.Clone("{histo_std_1.GetName()}_ratio")
        histo_ratio_std.Divide(histo_std_2)

        folder_plots = data_param_1[case_ml_1]["analysis"]["dir_general_plots"]
        if not os.path.exists(folder_plots):
            print("creating folder ", folder_plots)
            os.makedirs(folder_plots)

        save_path = f"{folder_plots}/ratio_{case_ml_1}_{case_ml_2}_{hn_ml}_ml_std_mult_" \
                    f"{mult_bin}_period_{period_number}{suffix}.eps"

        plot_histograms([histo_ratio_std, histo_ratio_ml], True, True, ["STD", "ML"], "Ratio",
                        "#it{p}_{T} (GeV/#it{c}", f"{name_1} / {name_2}", "ML / STD",
                        save_path)

        folder_plots = data_param_2[case_ml_2]["analysis"]["dir_general_plots"]
        if not os.path.exists(folder_plots):
            print("creating folder ", folder_plots)
            os.makedirs(folder_plots)

        save_path = f"{folder_plots}/ratio_{case_ml_1}_{case_ml_2}_{hn_ml}_ml_std_mult_" \
                    f"{mult_bin}_period_{period_number}{suffix}.eps"

        plot_histograms([histo_ratio_std, histo_ratio_ml], True, True, ["STD", "ML"], "Ratio",
                        "#it{p}_{T} (GeV/#it{c}", f"{name_1} / {name_2}", "ML / STD",
                        save_path)


#####################################

gROOT.SetBatch(True)

SIGMA0 = 57.8e-9

######
# D0 #
######
compare_ml_std("D0pp", "MBvspt_ntrkl", -1, "data/std_results/HFPtSpectrum_D0_merged_20191010.root",
               None, [(1, [1]), (2, [2]), (3, [3]), (4, [4]), (5, [5]), (6, [6])], 0,
               ["histoSigmaCorr"], ["histoSigmaCorr"])
#compare_ml_std("D0pp", "MBvspt_ntrkl", -1, "data/std_results/D0_19_corryield.root", 1. / SIGMA0,
#               None, 1, ["histoYieldCorr"], ["corrYield_19"])
#compare_ml_std("D0pp", "MBvspt_ntrkl", -1, "data/std_results/D0_1029_corryield.root", 1. / SIGMA0,
#               None, 2, ["histoYieldCorr"], ["corrYield_1029"])
#compare_ml_std("D0pp", "MBvspt_ntrkl", -1, "data/std_results/D0_3059_corryield.root", 1. / SIGMA0,
#               None, 3, ["histoYieldCorr"], ["corrYield_3059"])

# PRELIM COMPARISON
# 2016 vs. 2016 int. mult.
compare_ml_std("D0pp", "MBvspt_ntrkl", 0,
               "data/std_results/HFPtSpectrum_D0_2016_prel_20191010.root", None,
               [(1, [1]), (2, [2, 3]), (3, [4, 5]), (4, [6, 7]), (5, [8, 9]), (6, [10, 11])],
               0, ["histoSigmaCorr"], ["histoSigmaCorr"], "_prelim")
# mreged vs. 2016 int. mult.
compare_ml_std("D0pp", "MBvspt_ntrkl", -1,
               "data/std_results/HFPtSpectrum_D0_2016_prel_20191010.root", None,
               [(1, [1]), (2, [2, 3]), (3, [4, 5]), (4, [6, 7]), (5, [8, 9]), (6, [10, 11])],
               0, ["histoSigmaCorr"], ["histoSigmaCorr"], "_prelim")
######
# Lc #
######
# Correct for branching ratios BR(Lc->pKpi) / BR(Lc->pK0s)
compare_ml_std("LcpK0spp", "MBvspt_ntrkl", -1,
               "data/std_results/HFPtSpectrum_LcpKpi_merged_20191010.root", 1./5.75, None, 0,
               ["histoSigmaCorr"], ["histoSigmaCorr"])
#compare_ml_std("LcpK0spp", "MBvspt_ntrkl", -1, "data/std_results/Lc_19_corryield.root",
#               1./5.75 / SIGMA0, None, 1, ["histoYieldCorr"], ["corrYield_19"])
#compare_ml_std("LcpK0spp", "MBvspt_ntrkl", -1, "data/std_results/Lc_1029_corryield.root",
#               1./5.75 / SIGMA0, None, 2, ["histoYieldCorr"], ["corrYield_1029"])
#compare_ml_std("LcpK0spp", "MBvspt_ntrkl", -1, "data/std_results/Lc_3059_corryield.root",
#                1./5.75 / SIGMA0, None, 3, ["histoYieldCorr"], ["corrYield_3059"])

# RATIOS
###########
# Lc / D0 #
###########
compare_ml_std_ratio("LcpK0spp", "D0pp", "MBvspt_ntrkl", -1,
                     "data/std_results/HFPtSpectrum_LcpKpi_merged_20191010.root",
                     "data/std_results/HFPtSpectrum_D0_merged_20191010.root", 1./5.75, None, None,
                     0, ["histoSigmaCorr"], ["histoSigmaCorr"], ["histoSigmaCorr"])
#compare_ml_std_ratio("LcpK0spp", "D0pp", "MBvspt_ntrkl", -1,
#                     "data/std_results/Lc_19_corryield.root",
#                     "data/std_results/D0_19_corryield.root", 1./5.75 / SIGMA0, 1. / SIGMA0,
#                     None, 1, ["histoYieldCorr"], ["corrYield_19"], ["corrYield_19"])
#compare_ml_std_ratio("LcpK0spp", "D0pp", "MBvspt_ntrkl", -1,
#                     "data/std_results/Lc_1029_corryield.root",
#                     "data/std_results/D0_1029_corryield.root", 1./5.75 / SIGMA0, 1. / SIGMA0,
#                     None, 2, ["histoYieldCorr"], ["corrYield_1029"], ["corrYield_1029"])
#compare_ml_std_ratio("LcpK0spp", "D0pp", "MBvspt_ntrkl", -1,
#                     "data/std_results/Lc_3059_corryield.root",
#                     "data/std_results/D0_3059_corryield.root", 1./5.75 / SIGMA0, 1. / SIGMA0,
#                     None, 3, ["histoYieldCorr"], ["corrYield_3059"], ["corrYield_3059"])
# PRELIM RESULTS
compare_ml_std_ratio("LcpK0spp", "D0pp", "MBvspt_ntrkl", -1,
                     "data/std_results/HFPtSpectrum_LcpKpi_2016_prel_20191010.root",
                     "data/std_results/HFPtSpectrum_D0_2016_prel_20191010.root", 1./5.75,
                     None, [(1, [1]), (2, [2, 3]), (3, [4, 5]), (4, [6]), (5, [7]), (6, [8])], 0,
                     ["histoSigmaCorr"], ["histoSigmaCorr"], ["histoSigmaCorr"], "_prelim")

###########
# Ds / D0 #
###########

compare_ml_std_ratio("Dspp", "D0pp", "MBvspt_ntrkl", -1,
                     "data/std_results/HFPtSpectrum_Ds_merged_20191010.root",
                     "data/std_results/HFPtSpectrum_D0_merged_20191010.root", None, None, None,
                     0, ["histoSigmaCorr"], ["hCrossSectionStatisticError"], ["histoSigmaCorr"])
if FILES_NOT_FOUND:
    print("FILES NOT FOUND:")
    for f in FILES_NOT_FOUND:
        print(f)
