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
from machine_learning_hep.utilities import plot_histograms, save_histograms, Errors
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
ERRS_GR_TOT = []
ERRS_GR_FD = []
ERRS_GR_WOFD = []
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
    print("NOTE: Scaling with 1./57.8e9 (corr Y/ev) and 1./0.0623 (BR Lc)")
    histo_.Scale(1./SIGMAV0)
    histo_.Scale(1./BRLC)
    if mb == 0:
        print("NOTE: Scaling MB with 1./0.92 (kINT7 trigger eff.) and 1./0.94 (nEvents for 1-999)")
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
    ERRS_GR_TOT.append(Errors.make_root_asymm(histo_, errs.get_total_for_spectra_plot(), const_x_err=0.3))
    ERRS_GR_FD.append(Errors.make_root_asymm(histo_, errs.get_total_for_spectra_plot(True), const_x_err=0.3))
    ERRS_GR_WOFD.append(Errors.make_root_asymm(histo_, errs.get_total_for_spectra_plot(False), const_x_err=0.3))
    ERRS_GR_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"CorrectedYieldPerEvent_{ANA_MB}_1999_19_1029_3059")
save_histograms([*HISTOS, *ERRS_GR_TOT, *ERRS_GR_FD, *ERRS_GR_WOFD], SAVE_PATH)
SAVE_PATH = make_standard_save_path(CASE, f"CorrectedYieldPerEvent_{ANA_MB}_1999_19_1029_3059_MLHEPCanvas")

# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS, ERRS_GR_TOT, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
        SAVE_PATH, False, colors=COLORS)

#############################################################################
##################### Plot spectra mult / spectra MB ########################
#############################################################################

#Divide by MB
HISTOS_DIV = divide_all_by_first_multovermb(HISTOS)
#Remove MB one
HISTOS_DIVMB = HISTOS_DIV[1:]
ERRS_GR_DIV_TOT = []
ERRS_GR_DIV_WOFD = []
ERRS_GR_DIV_FD = []
for mb, _ in enumerate(HISTOS_DIVMB):
    tot_mult_over_MB = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], HISTOS[0].GetNbinsX())
    tot_mult_over_MB_FD = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], HISTOS[0].GetNbinsX(), True)
    tot_mult_over_MB_WOFD = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], HISTOS[0].GetNbinsX(), False)

    ERRS_GR_DIV_TOT.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB, const_x_err=0.3))
    ERRS_GR_DIV_WOFD.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB_WOFD, const_x_err=0.3))
    ERRS_GR_DIV_FD.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB_FD, const_x_err=0.3))
    ERRS_GR_DIV_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_DIV_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_DIV_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"MultOverMB_{ANA_MB}_19_1029_3059")
save_histograms([*HISTOS_DIVMB, *ERRS_GR_DIV_TOT, *ERRS_GR_DIV_FD, *ERRS_GR_DIV_WOFD], SAVE_PATH)
SAVE_PATH = make_standard_save_path(CASE, f"MultOverMB_{ANA_MB}_19_1029_3059_MLHEPCanvas")

# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS_DIVMB, ERRS_GR_DIV_TOT, "", LEGEND_TITLES2, "#it{p}_{T} (GeV/#it{c})",
        "Ratio to d#it{N}/(d#it{p}_{T})|_{|y|<0.5}  mult. int.",
        SAVE_PATH, True, colors=COLORS2)



###########################################################################
##################### Plot Lc / D0 (mult and MB)  #########################
###########################################################################

HISTOS_LC = []
HISTOS_D0 = []
ERRS_LC = []
ERRS_D0 = []
ERRS_GR_DIVD0_TOT = []
ERRS_GR_DIVD0_FD = []
ERRS_GR_DIVD0_WOFD = []

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
    tot_Lc_over_D0_FD = calc_systematic_mesonratio(ERRS_LC[mb], ERRS_D0[mb], \
                                                   HISTOS_LCOVERD0[mb].GetNbinsX(), True)
    tot_Lc_over_D0_WOFD = calc_systematic_mesonratio(ERRS_LC[mb], ERRS_D0[mb], \
                                                     HISTOS_LCOVERD0[mb].GetNbinsX(), False)
    ERRS_GR_DIVD0_TOT.append(Errors.make_root_asymm(HISTOS_LCOVERD0[mb], \
                                                    tot_Lc_over_D0, const_x_err=0.3))
    ERRS_GR_DIVD0_FD.append(Errors.make_root_asymm(HISTOS_LCOVERD0[mb], \
                                                   tot_Lc_over_D0_FD, const_x_err=0.3))
    ERRS_GR_DIVD0_WOFD.append(Errors.make_root_asymm(HISTOS_LCOVERD0[mb], \
                                                     tot_Lc_over_D0_WOFD, const_x_err=0.3))
    ERRS_GR_DIVD0_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_DIVD0_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_DIVD0_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"LcOverD0_{ANA_MB}_1999_19_1029_3059")
save_histograms([*HISTOS_LCOVERD0, *ERRS_GR_DIVD0_TOT, *ERRS_GR_DIVD0_FD, *ERRS_GR_DIVD0_WOFD], SAVE_PATH)
SAVE_PATH = make_standard_save_path(CASE, f"LcOverD0_{ANA_MB}_1999_19_1029_3059_MLHEPCanvas")

# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
#results(HISTOS_LCOVERD0, ERRS_GR_DIVD0, "", LEGEND_TITLES3, "#it{p}_{T} (GeV/#it{c})",
#        "#Lambda_{c}^{+} / D^{0}", SAVE_PATH, None, colors=COLORS3)
results(HISTOS_LCOVERD0, ERRS_GR_DIVD0_TOT, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "#Lambda_{c}^{+} / D^{0}", SAVE_PATH, None, colors=COLORS)



###########################################################################
##################### Plot Lc / D0 double ratio   #########################
###########################################################################

HISTO_DR_M = []
ERRS_GR_DR_M_TOT = []
ERRS_GR_DR_M_FD = []
ERRS_GR_DR_M_WOFD = []
HISTO_DR_M.append(divide_by_eachother([HISTOS_LCOVERD0[2]], [HISTOS_LCOVERD0[1]])[0])
HISTO_DR_M.append(divide_by_eachother([HISTOS_LCOVERD0[3]], [HISTOS_LCOVERD0[1]])[0])

for mb, _ in enumerate(HISTO_DR_M):
    num = 2
    den = 1
    if mb == 1:
         numb = 3
    tot_Lc_over_D0_DR = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                         HISTO_DR_M[mb].GetNbinsX())
    tot_Lc_over_D0_DR_FD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                            HISTO_DR_M[mb].GetNbinsX(), True)
    tot_Lc_over_D0_DR_WOFD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                              HISTO_DR_M[mb].GetNbinsX(), False)
    ERRS_GR_DR_M_TOT.append(Errors.make_root_asymm(HISTO_DR_M[mb], \
                                                   tot_Lc_over_D0_DR, const_x_err=0.3))
    ERRS_GR_DR_M_FD.append(Errors.make_root_asymm(HISTO_DR_M[mb], \
                                                  tot_Lc_over_D0_DR_FD, const_x_err=0.3))
    ERRS_GR_DR_M_WOFD.append(Errors.make_root_asymm(HISTO_DR_M[mb], \
                                                    tot_Lc_over_D0_DR_WOFD, const_x_err=0.3))
    ERRS_GR_DR_M_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_DR_M_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_DR_M_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"LcOverD0_DoubleRatioWith19_{ANA_MB}_1029_3059")
save_histograms([*HISTO_DR_M, *ERRS_GR_DR_M_TOT, *ERRS_GR_DR_M_FD, *ERRS_GR_DR_M_WOFD], SAVE_PATH)
SAVE_PATH = make_standard_save_path(CASE, f"LcOverD0_DoubleRatioWith19_{ANA_MB}_1029_3059_MLHEPCanvas")

results(HISTO_DR_M, ERRS_GR_DR_M_TOT, "", LEGEND_TITLES4, "#it{p}_{T} (GeV/#it{c})",
        "[#Lambda_{c}^{+} / D^{0}]_{i} / [#Lambda_{c}^{+} / D^{0}]_{j}", SAVE_PATH, 2, colors=COLORS4)

HISTO_DR_MB = []
ERRS_GR_DR_MB_TOT = []
ERRS_GR_DR_MB_FD = []
ERRS_GR_DR_MB_WOFD = []
HISTO_DR_MB.append(divide_by_eachother_barlow([HISTOS_LCOVERD0[1]], [HISTOS_LCOVERD0[0]])[0])
HISTO_DR_MB.append(divide_by_eachother_barlow([HISTOS_LCOVERD0[3]], [HISTOS_LCOVERD0[0]])[0])

for mb, _ in enumerate(HISTO_DR_MB):
    num = 1
    den = 0
    if mb == 1:
         numb = 3
    tot_Lc_over_D0_DR = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                         HISTO_DR_MB[mb].GetNbinsX())
    tot_Lc_over_D0_DR_FD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                            HISTO_DR_MB[mb].GetNbinsX(), True)
    tot_Lc_over_D0_DR_WOFD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], ERRS_D0[den],
                                                              HISTO_DR_MB[mb].GetNbinsX(), False)
    ERRS_GR_DR_MB_TOT.append(Errors.make_root_asymm(HISTO_DR_MB[mb], \
                                                    tot_Lc_over_D0_DR, const_x_err=0.3))
    ERRS_GR_DR_MB_FD.append(Errors.make_root_asymm(HISTO_DR_MB[mb], \
                                                   tot_Lc_over_D0_DR_FD, const_x_err=0.3))
    ERRS_GR_DR_MB_WOFD.append(Errors.make_root_asymm(HISTO_DR_MB[mb], \
                                                     tot_Lc_over_D0_DR_WOFD, const_x_err=0.3))
    ERRS_GR_DR_MB_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_DR_MB_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_DR_MB_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(CASE, f"LcOverD0_DoubleRatioWithMB_{ANA_MB}_19_3059")
save_histograms([*HISTO_DR_MB, *ERRS_GR_DR_MB_TOT, *ERRS_GR_DR_MB_FD, *ERRS_GR_DR_MB_WOFD], SAVE_PATH)
SAVE_PATH = make_standard_save_path(CASE, f"LcOverD0_DoubleRatioWithMB_{ANA_MB}_19_3059_MLHEPCanvas")

results(HISTO_DR_MB, ERRS_GR_DR_MB_TOT, "", LEGEND_TITLES5, "#it{p}_{T} (GeV/#it{c})",
        "[#Lambda_{c}^{+} / D^{0}]_{i} / [#Lambda_{c}^{+} / D^{0}]_{j}", SAVE_PATH, 2, colors=COLORS3)
