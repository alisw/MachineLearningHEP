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
from machine_learning_hep.utilities import divide_by_eachother
from machine_learning_hep.utilities import calc_systematic_mesonratio

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
    else:
        plot_histograms([*systematics, *histos_central], False, [False, True, [0, 0.6]],
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

CASE = "Dspp"
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

COLORS3 = [kGreen + 2, kAzure + 3]
COLORS = [kBlue, kGreen + 2, kRed - 2, kAzure + 3]
COLORSHM = [kBlue, kGreen + 2, kRed - 2, kAzure + 3, kOrange + 7]
COLORS2 = [kGreen + 2, kRed - 2, kAzure + 3]

# Get the ML histogram of the particle case and analysis type
# Everything available in the HFPtSpetrum can be requested
# From MB
HISTOS = []
ERRS = []
ERRS_GR = []
ERROR_FILES = ["data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_0.yaml",
               "data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_1.yaml",
               "data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_2.yaml",
               "data/errors/Dspp/MBvspt_ntrkl/errors_histoSigmaCorr_3.yaml",
               "data/errors/Dspp/SPDvspt/errors_histoSigmaCorr_4.yaml"]
ERROR_FILESD0 = ["data/errors/D0pp/MBvspt_ntrkl/errors_histoSigmaCorr_0.yaml",
                 "data/errors/D0pp/MBvspt_ntrkl/errors_histoSigmaCorr_1.yaml",
                 "data/errors/D0pp/MBvspt_ntrkl/errors_histoSigmaCorr_2.yaml",
                 "data/errors/D0pp/MBvspt_ntrkl/errors_histoSigmaCorr_3.yaml",
                 "data/errors/D0pp/SPDvspt/errors_histoSigmaCorr_4.yaml"]

SIGMAV0 = 57.8e9
BRDS = 0.0227

for mb in range(4):
    histo_ = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "histoSigmaCorr")
    histo_.SetName(f"{histo_.GetName()}_{mb}")
    histo_.Scale(1./SIGMAV0)
    histo_.Scale(1./BRDS)
    if mb == 0:
        histo_.Scale(1./0.92)
        histo_.Scale(1./0.94)
    HISTOS.append(histo_)

    DICTNB = {}
    GRFD = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "gFcConservative")
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

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_all_years_{ANA_MB}_MB")


# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS, ERRS_GR, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "d#it{N}/(d#it{p}_{T})|_{|y|<0.5} (GeV^{-1} #it{c})",
        SAVE_PATH, False, colors=COLORS)


#############################################################################
##################### NOW ADD HM AND DO ANOTHER PLOT  #######################
#############################################################################

# Append the HM histogram
HISTO_HM = extract_histo_or_error(CASE, ANA_HM, 4, YEAR_NUMBER, "histoSigmaCorr")
HISTO_HM.SetName(f"{HISTO_HM.GetName()}_4")
HISTOS.append(HISTO_HM)

DICTNB = {}
GRFD = extract_histo_or_error(CASE, ANA_MB, 4, YEAR_NUMBER, "gFcConservative")
ERRORNB = []
EYHIGH = GRFD.GetEYhigh()
EYLOW = GRFD.GetEYlow()
YVAL = GRFD.GetY()
for i in range(HISTO_HM.GetNbinsX()):
    ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
DICTNB["feeddown_NB"] = ERRORNB

ERRS_HM = Errors(HISTO_HM.GetNbinsX())
ERRS_HM.read(ERROR_FILES[4], DICTNB)
ERRS_GR.append(Errors.make_root_asymm(HISTO_HM, ERRS_HM.get_total_for_spectra_plot(), \
               const_x_err=0.3))
ERRS_GR[4].SetName("%s%d" % (ERRS_GR[4].GetName(), 4))

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_all_years_MB_{ANA_MB}_HM_{ANA_HM}")

results(HISTOS, ERRS_GR, "", LEGEND_TITLESHM, "#it{p}_{T} (GeV/#it{c})",
        "d^{2}#sigma/(d#it{p}_{T}d#it{y}) #times BR(D_{s}^{+} #rightarrow #phi#pi #rightarrow KK#pi) (#mub GeV^{-1} #it{c})",
        SAVE_PATH, False, colors=COLORSHM)


#############################################################################
##################### Plot spectra mult / spectra MB ########################
#############################################################################

#Divide by MB
HISTOS_DIV = divide_all_by_first_multovermb(HISTOS)
#Remove HM one
HISTOS_DIVMB = HISTOS_DIV[:-1]
#Remove MB one
HISTOS_DIVMB = HISTOS_DIVMB[1:]
ERRS_GR_DIV = []
for mb, _ in enumerate(HISTOS_DIVMB):
    tot_mult_over_MB = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], HISTOS[0].GetNbinsX())
    ERRS_GR_DIV.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB, const_x_err=0.3))
    ERRS_GR_DIV[mb].SetName("%s%d" % (ERRS_GR_DIV[mb].GetName(), mb+1))

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_MultOverMB_all_years_{ANA_MB}_MB")

# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS_DIVMB, ERRS_GR_DIV, "", LEGEND_TITLES2, "#it{p}_{T} (GeV/#it{c})",
        "Ratio to d#it{N}/(d#it{p}_{T})|_{|y|<0.5} mult. int.",
        SAVE_PATH, True, colors=COLORS2)



###########################################################################
##################### Plot Ds / D0 (mult and MB)  #########################
###########################################################################

HISTOS_DS = []
HISTOS_D0 = []
ERRS_DS = []
ERRS_D0 = []
ERRS_GR_DIVD0 = []

for mb in range(4):

    #if mb == 0 or mb == 2:
    #    continue
    histo_ = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "histoSigmaCorr")
    histo_.SetName(f"{histo_.GetName()}_Ds{mb}")
    HISTOS_DS.append(histo_)

    DICTNB = {}
    GRFD = extract_histo_or_error(CASE, ANA_MB, mb, YEAR_NUMBER, "gFcConservative")
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTNB["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILES[mb], DICTNB)
    ERRS_DS.append(errs)

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
    #-1 because D0 has also bin [1-2]
    for i in range(histo_.GetNbinsX()-1):
        ERRORNB.append([0, 0, EYLOW[i+2], EYHIGH[i+2], YVAL[i+2]])
    DICTNB["feeddown_NB"] = ERRORNB

    #-1 because D0 has also bin [1-2]
    errs = Errors(histo_.GetNbinsX()-1)
    errs.read(ERROR_FILESD0[mb], DICTNB)
    ERRS_D0.append(errs)

HISTOS_DSOVERD0 = divide_by_eachother(HISTOS_DS, HISTOS_D0, [2.27, 3.89], [2,4,6,8,12,24])

for mb, _ in enumerate(HISTOS_DSOVERD0):
    tot_Ds_over_D0 = calc_systematic_mesonratio(ERRS_DS[mb], ERRS_D0[mb], \
                                                HISTOS_DSOVERD0[mb].GetNbinsX())
    ERRS_GR_DIVD0.append(Errors.make_root_asymm(HISTOS_DSOVERD0[mb], \
                                                tot_Ds_over_D0, const_x_err=0.3))
    ERRS_GR_DIVD0[mb].SetName("%s%d" % (ERRS_GR_DIVD0[mb].GetName(), mb+1))

# Save globally in Ds directory
SAVE_PATH = make_standard_save_path(CASE, f"histoSigmaCorr_DsOverD0_all_years_{ANA_MB}_MB")

# As done here one can add an additional TGraphAsymmErrors per histogram. Those values will
# be added to the list the user has defined here.
# The list of error objects can contain None and in the end have the same length as number
# of histograms
results(HISTOS_DSOVERD0, ERRS_GR_DIVD0, "", LEGEND_TITLES, "#it{p}_{T} (GeV/#it{c})",
        "D_{s}^{+} / D^{0}", SAVE_PATH, None, colors=COLORS)
#results(HISTOS_DSOVERD0, ERRS_GR_DIVD0, "", LEGEND_TITLES3, "#it{p}_{T} (GeV/#it{c})",
#        "D_{s}^{+} / D^{0}", SAVE_PATH, None, colors=COLORS3)
