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

def compare_ml_std(case_ml, ana_type_ml, period_number, filepath_std, scale_std=None,
                   map_std_bins=None):

    with open("data/database_ml_parameters_%s.yml" % case_ml, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    if period_number < 0:
        filepath_ml = data_param[case_ml]["analysis"][ana_type_ml]["data"]["resultsallp"]
    else:
        filepath_ml = data_param[case_ml]["analysis"][ana_type_ml]["data"]["results"][period_number]

    # Get pt spectrum files
    path_ml = f"{filepath_ml}/finalcross{case_ml}{ana_type_ml}mult0.root"
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

    for hn in histo_names:
        histo_ml = file_ml.Get(hn)
        histo_std_tmp = file_std.Get(hn)
        histo_std = None

        if not histo_ml or not histo_std_tmp:
            print(f"Could not find histogram {hn}, continue...")
            continue

        if "MC" not in hn and map_std_bins is not None:
            histo_std = histo_ml.Clone("std_rebin")
            histo_std.Reset("ICESM")

            contents = [0] * histo_ml.GetNbinsX()
            errors = [0] * histo_ml.GetNbinsX()

            for ml_bin, std_bins in map_std_bins:
                for b in std_bins:
                    contents[ml_bin-1] += histo_std_tmp.GetBinContent(b) / len(std_bins)
                    errors[ml_bin-1] += histo_std_tmp.GetBinError(b) * histo_std_tmp.GetBinError(b)

            for b in range(histo_std.GetNbinsX()):
                histo_std.SetBinContent(b+1, contents[b])
                histo_std.SetBinError(b+1, sqrt(errors[b]))

        else:
            histo_std = histo_std_tmp.Clone("std_cloned")
            if scale_std is not None:
                histo_std.Scale(scale_std)

        folder_plots = os.path.join(filepath_ml, "ml_std_comparison")
        if not os.path.exists(folder_plots):
            print("creating folder ", folder_plots)
            os.makedirs(folder_plots)

        save_path = f"{folder_plots}/{hn}_ml_std.eps"

        plot_histograms([histo_std, histo_ml], True, True, ["STD", "ML"], histo_ml.GetTitle(),
                        "#it{p}_{T} (GeV/#it{c}", histo_ml.GetYaxis().GetTitle(), "ML / STD",
                        save_path)

#####################################

gROOT.SetBatch(True)

compare_ml_std("D0pp", "MBvspt_ntrkl", -1, "data/std_results/HFPtSpectrum_D0_20191003.root")
compare_ml_std("D0pp", "MBvspt_ntrkl", -1, "data/std_results/HFPtSpectrum_D0_20191003.root")
# Correct for branching ratios BR(Lc->pKpi) / BR(Lc->pK0s)
compare_ml_std("LcpK0spp", "MBvspt_ntrkl", -1, "data/std_results/HFPtSpectrum_LcpKpi_20191003.root",
               1./5.75)
compare_ml_std("D0pp", "MBvspt_ntrkl", 0, "data/std_results/HFPtSpectrum_D0_2016_20191003.root",
               None, [(1, [1]), (2, [2, 3]), (3, [4, 5]), (4, [6, 7]), (5, [8, 9]), (6, [10, 11])])

if FILES_NOT_FOUND:
    print("FILES NOT FOUND:")
    for f in FILES_NOT_FOUND:
        print(f)
