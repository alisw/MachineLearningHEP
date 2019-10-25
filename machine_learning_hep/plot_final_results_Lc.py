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
from machine_learning_hep.utilities import plot_histograms, Errors
from machine_learning_hep.utilities import calc_systematic_multovermb
from machine_learning_hep.utilities import divide_all_by_first_multovermb
from machine_learning_hep.utilities import divide_by_eachother, divide_by_eachother_barlow
from machine_learning_hep.utilities import calc_systematic_mesonratio
from machine_learning_hep.utilities import calc_systematic_mesondoubleratio

def results(histos_central, systematics, title, legend_titles, x_label, y_label,
            save_path, ratio, **kwargs):

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

    if ratio is False:
        plot_histograms([*systematics, *histos_central], True, [False, False, [1e-8, 1]], legend_titles, title,
                        x_label, y_label, "", save_path, linesytles=[1], markerstyles=markerstyles,
                        colors=colors, linewidths=[1], draw_options=draw_options,
                        fillstyles=[0])
    elif ratio is True:
        plot_histograms([*systematics, *histos_central], True, [False, True, [0.01, 100]],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])
    elif ratio == 2:
        plot_histograms([*systematics, *histos_central], False, [False, True, [0, 3.1]],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])
    else:
        plot_histograms([*systematics, *histos_central], False, [False, True, [0, 1.0]],
                        legend_titles, title, x_label, y_label, "", save_path, linesytles=[1],
                        markerstyles=markerstyles, colors=colors, linewidths=[1],
                        draw_options=draw_options, fillstyles=[0])

def get_param(case):
    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    return data_param

def extract_histo_or_error(case, ana_type, mult_bin, period_number, histo_name, filepath=None):
    data_param = get_param(case)
    if filepath is None:
        filepath = data_param[case]["analysis"][ana_type]["data"]["resultsallp"]
    if period_number >= 0:
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

CASE = "LcpKpipp"
ANA_MB = "MBvspt_ntrkl"
ANA_HM = "SPDvspt"
YEAR_NUMBER = -1 # -1 refers to all years merged

LEGEND_TITLES = ["#kern[1]{0} #kern[-0.05]{#leq} #it{N}_{tracklets} < #infty (MB)",
                 "#kern[1.6]{1} #kern[0.3]{#leq} #it{N}_{tracklets} < 9 (MB)",
                 "10 #leq #it{N}_{tracklets} < 29 (MB)", "30 #leq #it{N}_{tracklets} < 59 (MB)"]
LEGEND_TITLES3 = ["#kern[1.6]{1} #kern[0.3]{#leq} #it{N}_{tracklets} < 9 (MB)",
                 "30 #leq #it{N}_{tracklets} < 59 (MB)"]
LEGEND_TITLESHM = ["#kern[1]{0} #kern[-0.05]{#leq} #it{N}_{tracklets} < #infty (MB)",
                   "#kern[1.6]{1} #kern[0.3]{#leq} #it{N}_{tracklets} < 9 (MB)",
                   "10 #leq #it{N}_{tracklets} < 29 (MB)", "30 #leq #it{N}_{tracklets} < 59 (MB)",
                   "60 #leq #it{N}_{tracklets} < 99 (HM)"]
LEGEND_TITLES2 = ["#kern[1.6]{1} #kern[0.3]{#leq} #it{N}_{tracklets} < 9 (MB)",
                  "10 #leq #it{N}_{tracklets} < 29 (MB)", "30 #leq #it{N}_{tracklets} < 59 (MB)"]
LEGEND_TITLES4 = ["[10 #leq #it{N}_{tracklets} < 29] / [1 #leq #it{N}_{tracklets} < 10]",
                  "[30 #leq #it{N}_{tracklets} < 59] / [1 #leq #it{N}_{tracklets} < 10]"]
LEGEND_TITLES5 = ["[#kern[1.6]{1} #kern[0.3]{#leq} #it{N}_{tracklets} < 9] / [#kern[1]{0} #kern[-0.05]{#leq} #it{N}_{tracklets} < #infty]",
                 "[30 #leq #it{N}_{tracklets} < 59] / [#kern[1]{0} #kern[-0.05]{#leq} #it{N}_{tracklets} < #infty]"]

COLORS = [kBlue, kGreen + 2, kRed - 2, kAzure + 3]
COLORS3 = [kGreen + 2, kAzure + 3]
COLORSHM = [kBlue, kGreen + 2, kRed - 2, kAzure + 3, kOrange + 7]
COLORS2 = [kGreen + 2, kRed - 2, kAzure + 3]
COLORS4 = [kRed - 2, kAzure + 3]

# Get the ML histogram of the particle case and analysis type
# Everything available in the HFPtSpetrum can be requested
# From MB
HISTOS = []
ERRS = []
ERRS_GR = []
ERROR_FILES = ["data/errors/LcpKpipp_full/MBvspt_ntrkl/errors_histoSigmaCorr_0.yaml",
               "data/errors/LcpKpipp_full/MBvspt_ntrkl/errors_histoSigmaCorr_1.yaml",
               "data/errors/LcpKpipp_full/MBvspt_ntrkl/errors_histoSigmaCorr_2.yaml",
               "data/errors/LcpKpipp_full/MBvspt_ntrkl/errors_histoSigmaCorr_3.yaml",
               "data/errors/LcpKpipp_full/SPDvspt/errors_histoSigmaCorr_4.yaml"]
ERROR_FILESD0 = ["data/errors/D0pp_full/MBvspt_ntrkl/errors_histoSigmaCorr_0.yaml",
                 "data/errors/D0pp_full/MBvspt_ntrkl/errors_histoSigmaCorr_1.yaml",
                 "data/errors/D0pp_full/MBvspt_ntrkl/errors_histoSigmaCorr_2.yaml",
                 "data/errors/D0pp_full/MBvspt_ntrkl/errors_histoSigmaCorr_3.yaml",
                 "data/errors/D0pp_full/SPDvspt/errors_histoSigmaCorr_4.yaml"]

PATHD0 = "data/std_results/23Oct/"
PATHLC = "data/std_results/23Oct/"

SIGMAV0 = 57.8e9
BRLC = 0.0623

for mb in range(4):
    histo_ = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "histoSigmaCorr", PATHLC)
    histo_.SetName(f"{histo_.GetName()}_{mb}")
    histo_.Scale(1./SIGMAV0)
    histo_.Scale(1./BRLC)
    if mb == 0:
        histo_.Scale(1./0.92)
        histo_.Scale(1./0.94)
    HISTOS.append(histo_)

    DICTNB = {}
    GRFD = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "gFcCorrConservative", PATHLC)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTNB["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILES[mb], DICTNB)
    ERRS.append(errs)
    ERRS_GR.append(Errors.make_root_asymm(histo_, errs.get_total(), const_x_err=0.3))
    ERRS_GR[mb].SetName("%s%d" % (ERRS_GR[mb].GetName(), mb))

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_all_years_{ANA_MB}_MB")


# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS, ERRS_GR, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
        SAVE_PATH, False, colors=COLORS)

#############################################################################
##################### Plot spectra mult / spectra MB ########################
#############################################################################

#Divide by MB
HISTOS_DIV = divide_all_by_first_multovermb(HISTOS)
#Remove MB one
HISTOS_DIVMB = HISTOS_DIV[1:]
ERRS_GR_DIV = []
for mb, _ in enumerate(HISTOS_DIVMB):
    tot_mult_over_MB = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], HISTOS[0].GetNbinsX())
    ERRS_GR_DIV.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB, const_x_err=0.3))
    ERRS_GR_DIV[mb].SetName("%s%d" % (ERRS_GR_DIV[mb].GetName(), mb+1))

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_MultOverMB_all_years_{ANA_MB}_MB")

# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS_DIVMB, ERRS_GR_DIV, "", LEGEND_TITLES2, "#it{p}_{T} (GeV/#it{c})",
        "Ratio to d#it{N}/(d#it{p}_{T})|_{|y|<0.5}  mult. int.",
        SAVE_PATH, True, colors=COLORS2)



###########################################################################
##################### Plot Lc / D0 (mult and MB)  #########################
###########################################################################

HISTOS_LC = []
HISTOS_D0 = []
ERRS_LC = []
ERRS_D0 = []
ERRS_GR_DIVD0 = []

for mb in range(4):

    #if mb == 0 or mb == 2:
    #    continue
    histo_ = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "histoSigmaCorr", PATHLC)
    histo_.SetName(f"{histo_.GetName()}_Lc{mb}")
    HISTOS_LC.append(histo_)

    DICTNB = {}
    GRFD = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "gFcCorrConservative", PATHLC)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTNB["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILES[mb], DICTNB)
    ERRS_LC.append(errs)

PATHD0 = "data/std_results/23Oct/"
for mb in range(4):

    #if mb == 0 or mb == 2:
    #    continue
    histo_ = extract_histo_or_error("D0pp", "MBvspt_ntrkl", mb, YEAR_NUMBER, \
                                    "histoSigmaCorr", PATHD0)
    histo_.SetName(f"{histo_.GetName()}_D0{mb}")
    HISTOS_D0.append(histo_)

    DICTNB = {}
    GRFD = extract_histo_or_error("D0pp", "MBvspt_ntrkl", mb, YEAR_NUMBER, \
                                  "gFcConservative", PATHD0)

    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTNB["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILESD0[mb], DICTNB)
    ERRS_D0.append(errs)

HISTOS_LCOVERD0 = divide_by_eachother(HISTOS_LC, HISTOS_D0, [6.23, 3.89])

for mb, _ in enumerate(HISTOS_LCOVERD0):
    tot_Lc_over_D0 = calc_systematic_mesonratio(ERRS_LC[mb], ERRS_D0[mb], \
                                                HISTOS_LCOVERD0[mb].GetNbinsX())
    ERRS_GR_DIVD0.append(Errors.make_root_asymm(HISTOS_LCOVERD0[mb], \
                                                tot_Lc_over_D0, const_x_err=0.3))
    ERRS_GR_DIVD0[mb].SetName("%s%d" % (ERRS_GR_DIVD0[mb].GetName(), mb+1))

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_LcOverD0_all_years_{ANA_MB}_MB")

# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
#results(HISTOS_LCOVERD0, ERRS_GR_DIVD0, "", LEGEND_TITLES3, "#it{p}_{T} (GeV/#it{c})",
#        "#Lambda_{c}^{+} / D^{0}", SAVE_PATH, None, colors=COLORS3)
results(HISTOS_LCOVERD0, ERRS_GR_DIVD0, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#Lambda_{c}^{+} / D^{0}", SAVE_PATH, None, colors=COLORS)



###########################################################################
##################### Plot Lc / D0 double ratio   #########################
###########################################################################

HISTO_DR_M = []
ERRS_GR_DR_M = []
HISTO_DR_M.append(divide_by_eachother([HISTOS_LCOVERD0[2]], [HISTOS_LCOVERD0[1]])[0])
HISTO_DR_M.append(divide_by_eachother([HISTOS_LCOVERD0[3]], [HISTOS_LCOVERD0[1]])[0])

for mb, _ in enumerate(HISTO_DR_M):
    num = 2
    den = 1
    if mb == 1:
         numb = 3
    tot_Lc_over_D0_DR = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                         HISTO_DR_M[mb].GetNbinsX())
    ERRS_GR_DR_M.append(Errors.make_root_asymm(HISTO_DR_M[mb], \
                                                tot_Lc_over_D0_DR, const_x_err=0.3))
    ERRS_GR_DR_M[mb].SetName("%s%d" % (ERRS_GR_DR_M[mb].GetName(), mb+1))

SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_LcOverD0_DoubleRatioM_all_years_{ANA_MB}_MB")

results(HISTO_DR_M, ERRS_GR_DR_M, "", LEGEND_TITLES4, "#it{p}_{T} (GeV/#it{c})",
        "[#Lambda_{c}^{+} / D^{0}]_{i} / [#Lambda_{c}^{+} / D^{0}]_{j}", SAVE_PATH, 2, colors=COLORS4)

#Stat Unc procedure to be changed
#HISTO_DR_MB = []
#HISTO_DR_MB.append(divide_by_eachother(HISTOS_LCOVERD0[1], HISTOS_LCOVERD0[0]))
#HISTO_DR_MB.append(divide_by_eachother(HISTOS_LCOVERD0[3], HISTOS_LCOVERD0[0]))

HISTO_DR_MB = []
ERRS_GR_DR_MB = []
HISTO_DR_MB.append(divide_by_eachother_barlow([HISTOS_LCOVERD0[1]], [HISTOS_LCOVERD0[0]])[0])
HISTO_DR_MB.append(divide_by_eachother_barlow([HISTOS_LCOVERD0[3]], [HISTOS_LCOVERD0[0]])[0])

for mb, _ in enumerate(HISTO_DR_MB):
    num = 1
    den = 0
    if mb == 1:
         numb = 3
    tot_Lc_over_D0_DR = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                         HISTO_DR_MB[mb].GetNbinsX())
    ERRS_GR_DR_MB.append(Errors.make_root_asymm(HISTO_DR_MB[mb], \
                                                tot_Lc_over_D0_DR, const_x_err=0.3))
    ERRS_GR_DR_MB[mb].SetName("%s%d" % (ERRS_GR_DR_MB[mb].GetName(), mb+1))

SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_LcOverD0_DoubleRatioMB_all_years_{ANA_MB}_MB")

results(HISTO_DR_MB, ERRS_GR_DR_MB, "", LEGEND_TITLES5, "#it{p}_{T} (GeV/#it{c})",
        "[#Lambda_{c}^{+} / D^{0}]_{i} / [#Lambda_{c}^{+} / D^{0}]_{j}", SAVE_PATH, 2, colors=COLORS3)
