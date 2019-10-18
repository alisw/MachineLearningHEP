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
from machine_learning_hep.utilities import plot_histograms, Errors


def results(histos_central, systematics, title, legend_titles, x_label, y_label,
            save_path, **kwargs):

    if len(histos_central) != len(systematics):
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

    colors = colors * 2
    markerstyles = [1] * len(histos_central) + [20] * len(histos_central)
    draw_options = ["E2"] * len(histos_central) + [""] * len(histos_central)
    legend_titles = [None] * len(histos_central) + legend_titles

    plot_histograms([*systematics, *histos_central], True, False, legend_titles, title,
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
ERROR_FILES = ["data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_0.yaml",
               "data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_1.yaml",
               "data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_2.yaml",
               "data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_3.yaml",
               "data/errors/Dspp/SPDvspt/errors_histoSigmaCorr_4.yaml"]

for mb in range(4):
    histo_ = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "histoSigmaCorr")
    histo_.SetName(f"{histo_.GetName()}_{mb}")
    HISTOS.append(histo_)

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILES[mb])
    ERRS.append(Errors.make_root_asymm(histo_, errs.get_total(), const_x_err=0.3))

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_all_years_{ANA_MB}_MB")


# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS, ERRS, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#sigma (D_{s}) #times BR(D_{s} #rightarrow KK#pi)", SAVE_PATH, colors=COLORS)


#############################################################################
##################### NOW ADD HM AND DO ANOTHER PLOT  #######################
#############################################################################

LEGEND_TITLES.append("60 < n_{trkl} < 99 (HM)")

COLORS.append(kOrange + 7)


# Append the HM histogram
HISTO_HM = extract_histo_or_error(CASE, ANA_HM, 4, YEAR_NUMBER, "histoSigmaCorr")
HISTO_HM.SetName(f"{HISTO_HM.GetName()}_4")
HISTOS.append(HISTO_HM)
ERRS_HM = Errors(HISTO_HM.GetNbinsX())
ERRS_HM.read(ERROR_FILES[4])
ERRS.append(Errors.make_root_asymm(HISTO_HM, ERRS_HM.get_total(), const_x_err=0.3))

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_all_years_MB_{ANA_MB}_HM_{ANA_HM}")

results(HISTOS, ERRS, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#sigma (D_{s}) #times BR(D_{s} #rightarrow KK#pi)",
        SAVE_PATH, colors=COLORS)
