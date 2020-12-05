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

NB: Duplicate of macro in AN Note repository. Will not work here!
    Just as an example how functions can be used
"""
import os
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, gStyle, gROOT, TH1F, TGraphAsymmErrors, TH1
from ROOT import kBlue, kAzure, kOrange, kGreen, kBlack, kRed, kWhite
from ROOT import Double
from support.utilities_plot import Errors
from support.utilities_plot import calc_systematic_multovermb
from support.utilities_plot import divide_all_by_first_multovermb
from support.utilities_plot import divide_by_eachother, divide_by_eachother_barlow
from support.utilities_plot import calc_systematic_mesonratio
from support.utilities_plot import calc_systematic_mesondoubleratio
from support.utilities_plot import plot_histograms, save_histograms

def extract_histo(case, ana_type, mult_bin, histo_name, filepath, filename=None):

    path = f"{filepath}/finalcross{case}{ana_type}mult{mult_bin}.root"
    if filename is not None:
        path = f"{filepath}/{filename}"
    in_file = TFile.Open(path, "READ")
    histo = in_file.Get(histo_name)
    if isinstance(histo, TH1):
        histo.SetDirectory(0)
    return histo

def make_standard_save_path(prefix, filepath):

    folder_plots = f"{filepath}"
    if not os.path.exists(folder_plots):
        print("creating folder ", folder_plots)
        os.makedirs(folder_plots)
    return f"{folder_plots}/{prefix}.eps"


#############################################################################
############################# Input arguments  ##############################
#############################################################################

CASE = "LcpKpipp"
CASEAV = "LcAverage"
CASED0 = "D0pp"
ANA_MB = "MBvspt_ntrkl"
ANA_HM = "SPDvspt"
NMULTBINS = 5
SIGMAV0 = 57.8e9
BRLC = 0.0623 #pKpi BR, the average one is already scaled
BRD0 = 0.0389

PATH_IN = "inputfiles_QM19/"
PATH_IN_HM = "inputfiles_HP20/"
PATH_OUT = "final_rootfiles/"
FILEMB_LC_ALLHISTOS = "finalcrossLcpKpippMBvspt_ntrklmult0_WithoutNbx2_WithAllHistos.root"

ERROR_FILESLC = ["syst_QM19/LcpKpipp/errors_histoSigmaCorr_0.yaml",
                 "syst_QM19/LcpKpipp/errors_histoSigmaCorr_1.yaml",
                 "syst_QM19/LcpKpipp/errors_histoSigmaCorr_2.yaml",
                 "syst_QM19/LcpKpipp/errors_histoSigmaCorr_3.yaml",
                 "syst_HP20/syst_forLcaverage/errors_avg_histoSigmaCorr_4.yaml"]
ERROR_FILESD0 = ["syst_QM19/D0pp/MBvspt_ntrkl_Lcbinning/errors_histoSigmaCorr_0.yaml",
                 "syst_QM19/D0pp/MBvspt_ntrkl_Lcbinning/errors_histoSigmaCorr_1.yaml",
                 "syst_QM19/D0pp/MBvspt_ntrkl_Lcbinning/errors_histoSigmaCorr_2.yaml",
                 "syst_QM19/D0pp/MBvspt_ntrkl_Lcbinning/errors_histoSigmaCorr_3.yaml",
                 "syst_HP20/D0pp/errors_histoSigmaCorr_4.yaml"]

#Needed because different binnings
PATH_BINNEDASLC19 = "inputfiles_QM19/Lc_BinnedAsLc19/"
PATH_BINNEDASLC3059 = "inputfiles_QM19/Lc_BinnedAsLc3059/"
PATHD0_BINNEDASLC = "inputfiles_QM19/D0_BinnedAsLc/"
ERROR_BINNEDASLC19 = "syst_QM19/LcpKpipp/errors_histoSigmaCorr_0_for19.yaml"
ERROR_BINNEDASLC3059 = "syst_QM19/LcpKpipp/errors_histoSigmaCorr_0_for3059.yaml"

HISTOS = []
ERRS = []
ERRSFOR19 = []
ERRSFOR3059 = []
ERRS_GR_TOT = []
ERRS_GR_FD = []
ERRS_GR_WOFD = []
gROOT.SetBatch(True)


#############################################################################
############################### Plot spectra ################################
#############################################################################

for mb in range(NMULTBINS):
    print("\nAnalysing multiplicity interval", mb)

    PATH_MIX = PATH_IN
    if mb == 4:
        PATH_MIX = PATH_IN_HM

    histo_ = extract_histo(CASE, ANA_MB, mb, "histoSigmaCorr", PATH_MIX)
    if mb == 4:
        histo_ = extract_histo(CASEAV, ANA_MB, mb, "histoSigmaCorr_average", PATH_MIX)
    histo_.SetName(f"histoSigmaCorr_{mb}")

    if PATH_MIX == "inputfiles_QM19/":
        print("  NOTE: Scaling with 1. /", SIGMAV0, "(sigmaV0)")
        histo_.Scale(1./SIGMAV0)
    else:
        print("  No scaling for sigmaV0 (", SIGMAV0, "). Should be applied already in file")

    if mb != 4:
        print("  NOTE: Scaling with 1. /", BRLC, "(BR Lc)")
        histo_.Scale(1./BRLC)
    else:
        print("  No scaling for BR (", BRLC, "). Should be applied already in file")

    HISTOS.append(histo_)

    DICTEXTRA = {}
    if mb != 4:
        if mb == 0:
            HISTOEFF = extract_histo(CASE, ANA_MB, mb, "hDirectEffpt", PATH_MIX, \
                                     FILEMB_LC_ALLHISTOS)
        else:
            HISTOEFF = extract_histo(CASE, ANA_MB, mb, "hDirectEffpt", PATH_MIX)
        ERROREFF = []
        for i in range(histo_.GetNbinsX()):
            RELERREFF = HISTOEFF.GetBinError(i+1) / HISTOEFF.GetBinContent(i+1)
            ERROREFF.append([0, 0, RELERREFF, RELERREFF])
        DICTEXTRA["statunceff"] = ERROREFF

    GRFD = extract_histo(CASE, ANA_MB, mb, "gFcCorrConservative", PATH_MIX)
    if mb == 4:
        GRFD = extract_histo(CASEAV, ANA_MB, mb, "gFcCorrConservative_average", PATH_MIX)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTEXTRA["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILESLC[mb], DICTEXTRA)
    ERRS.append(errs)

    ERRS_GR_TOT.append(Errors.make_root_asymm(histo_, errs.get_total_for_spectra_plot(), \
                                              const_x_err=0.3))
    ERRS_GR_FD.append(Errors.make_root_asymm(histo_, errs.get_total_for_spectra_plot(True), \
                                             const_x_err=0.3))
    ERRS_GR_WOFD.append(Errors.make_root_asymm(histo_, errs.get_total_for_spectra_plot(False), \
                                               const_x_err=0.3))
    ERRS_GR_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save histograms + systematics in result directory
FILENAME = f"LcpKpiAvgCorrectedYieldPerEvent_{ANA_MB}_1999_19_1029_3059_6099"
SAVE_PATH = make_standard_save_path(FILENAME, PATH_OUT)
save_histograms([*HISTOS, *ERRS_GR_TOT, *ERRS_GR_FD, *ERRS_GR_WOFD], SAVE_PATH)


#############################################################################
##################### Plot spectra mult / spectra MB ########################
#############################################################################

#Divide multiplicity spectra by MB
#  Cannot use divide_all_by_first_multovermb due to different pT binning.
#  Coded in ... way

HISTOS19 = []
HISTOS1029 = []
HISTOS3059 = []
HISTOS6099 = []

HISTO019 = extract_histo(CASE, ANA_MB, 0, "histoSigmaCorr", PATH_BINNEDASLC19)
HISTO019.SetName(f"histoSigmaCorr_{0}")
HISTOS19.append(HISTO019)

HISTO119 = extract_histo(CASE, ANA_MB, 1, "histoSigmaCorr", PATH_IN)
HISTO119.SetName(f"histoSigmaCorr_{1}")
HISTOS19.append(HISTO119)

HISTO01029 = extract_histo(CASE, ANA_MB, 0, "histoSigmaCorr", PATH_IN)
HISTO01029.SetName(f"histoSigmaCorr_{0}")
HISTOS1029.append(HISTO01029)

HISTO21029 = extract_histo(CASE, ANA_MB, 2, "histoSigmaCorr", PATH_IN)
HISTO21029.SetName(f"histoSigmaCorr_{2}")
HISTOS1029.append(HISTO21029)

HISTO03059 = extract_histo(CASE, ANA_MB, 0, "histoSigmaCorr", PATH_BINNEDASLC3059)
HISTO03059.SetName(f"histoSigmaCorr_{0}")
HISTOS3059.append(HISTO03059)

HISTO33059 = extract_histo(CASE, ANA_MB, 3, "histoSigmaCorr", PATH_IN)
HISTO33059.SetName(f"histoSigmaCorr_{3}")
HISTOS3059.append(HISTO33059)

HISTO06099 = extract_histo(CASE, ANA_MB, 0, "histoSigmaCorr", PATH_IN)
HISTO06099.SetName(f"histoSigmaCorr_{0}")
print("  NOTE: Scaling with 1. /", SIGMAV0, " and 1. /", BRLC, " (sigmaV0 and BR LC)")
HISTO06099.Scale(1./SIGMAV0)
HISTO06099.Scale(1./BRLC)
HISTOS6099.append(HISTO06099)

HISTO46099 = extract_histo(CASEAV, ANA_MB, 4, "histoSigmaCorr_average", PATH_IN_HM)
HISTO46099.SetName(f"histoSigmaCorr_{4}")
HISTOS6099.append(HISTO46099)

HISTOS_DIVMB = []
HISTOS_DIVMB.append(divide_all_by_first_multovermb(HISTOS19)[1])
HISTOS_DIVMB.append(divide_all_by_first_multovermb(HISTOS1029)[1])
HISTOS_DIVMB.append(divide_all_by_first_multovermb(HISTOS3059)[1])
HISTOS_DIVMB.append(divide_all_by_first_multovermb(HISTOS6099)[1])


ERRS_GR_DIV_TOT = []
ERRS_GR_DIV_WOFD = []
ERRS_GR_DIV_FD = []


DICTEXTRA = {}
HISTOEFF = extract_histo(CASE, ANA_MB, 0, "hDirectEffpt", PATH_IN, FILEMB_LC_ALLHISTOS)
ERROREFF = []
for i in range(HISTOS[0].GetNbinsX()):
    if i >= 5:
        continue
    RELERREFF = HISTOEFF.GetBinError(i+1) / HISTOEFF.GetBinContent(i+1)
    ERROREFF.append([0, 0, RELERREFF, RELERREFF])
DICTEXTRA["statunceff"] = ERROREFF

GRFD = extract_histo(CASE, ANA_MB, 0, "gFcCorrConservative", PATH_IN)
ERRORNB = []
EYHIGH = GRFD.GetEYhigh()
EYLOW = GRFD.GetEYlow()
YVAL = GRFD.GetY()
for i in range(HISTOS[0].GetNbinsX()):
    if i >= 5:
        continue
    ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
DICTEXTRA["feeddown_NB"] = ERRORNB

ERRFOR19 = Errors(HISTOS_DIVMB[0].GetNbinsX())
ERRFOR19.read(ERROR_BINNEDASLC19, DICTEXTRA)
ERRSFOR19.append(ERRFOR19)

TOT_MULT_OVER_MB = calc_systematic_multovermb(ERRS[1], ERRSFOR19[0], \
                                              HISTOS_DIVMB[0].GetNbinsX(), True)
TOT_MULT_OVER_MB_FD = calc_systematic_multovermb(ERRS[1], ERRSFOR19[0], \
                                                 HISTOS_DIVMB[0].GetNbinsX(), True, True)
TOT_MULT_OVER_MB_WOFD = calc_systematic_multovermb(ERRS[1], ERRSFOR19[0], \
                                                   HISTOS_DIVMB[0].GetNbinsX(), True, False)

ERRS_GR_DIV_TOT.append(Errors.make_root_asymm(HISTOS_DIVMB[0], TOT_MULT_OVER_MB, \
                                              const_x_err=0.3))
ERRS_GR_DIV_WOFD.append(Errors.make_root_asymm(HISTOS_DIVMB[0], TOT_MULT_OVER_MB_WOFD, \
                                               const_x_err=0.3))
ERRS_GR_DIV_FD.append(Errors.make_root_asymm(HISTOS_DIVMB[0], TOT_MULT_OVER_MB_FD, \
                                             const_x_err=0.3))
ERRS_GR_DIV_TOT[0].SetName("gr_TotSyst_%d" % 0)
ERRS_GR_DIV_FD[0].SetName("gr_FDSyst_%d" % 0)
ERRS_GR_DIV_WOFD[0].SetName("gr_TotSyst_woFD_%d" % 0)


TOT_MULT_OVER_MB = calc_systematic_multovermb(ERRS[2], ERRS[0], \
                                              HISTOS[2].GetNbinsX(), True)
TOT_MULT_OVER_MB_FD = calc_systematic_multovermb(ERRS[2], ERRS[0], \
                                                 HISTOS[2].GetNbinsX(), True, True)
TOT_MULT_OVER_MB_WOFD = calc_systematic_multovermb(ERRS[2], ERRS[0], \
                                                   HISTOS[2].GetNbinsX(), True, False)

ERRS_GR_DIV_TOT.append(Errors.make_root_asymm(HISTOS_DIVMB[1], TOT_MULT_OVER_MB, \
                                              const_x_err=0.3))
ERRS_GR_DIV_WOFD.append(Errors.make_root_asymm(HISTOS_DIVMB[1], TOT_MULT_OVER_MB_WOFD, \
                                               const_x_err=0.3))
ERRS_GR_DIV_FD.append(Errors.make_root_asymm(HISTOS_DIVMB[1], TOT_MULT_OVER_MB_FD, \
                                             const_x_err=0.3))
ERRS_GR_DIV_TOT[1].SetName("gr_TotSyst_%d" % 1)
ERRS_GR_DIV_FD[1].SetName("gr_FDSyst_%d" % 1)
ERRS_GR_DIV_WOFD[1].SetName("gr_TotSyst_woFD_%d" % 1)


DICTEXTRA = {}
HISTOEFF = extract_histo(CASE, ANA_MB, 0, "hDirectEffpt", PATH_IN, FILEMB_LC_ALLHISTOS)
ERROREFF = []
for i in range(HISTOS[0].GetNbinsX()):
    if i == 0:
        continue
    RELERREFF = HISTOEFF.GetBinError(i+1) / HISTOEFF.GetBinContent(i+1)
    ERROREFF.append([0, 0, RELERREFF, RELERREFF])
DICTEXTRA["statunceff"] = ERROREFF

GRFD = extract_histo(CASE, ANA_MB, 0, "gFcCorrConservative", PATH_IN)
ERRORNB = []
EYHIGH = GRFD.GetEYhigh()
EYLOW = GRFD.GetEYlow()
YVAL = GRFD.GetY()
for i in range(HISTOS[0].GetNbinsX()):
    if i == 0:
        continue
    ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
DICTEXTRA["feeddown_NB"] = ERRORNB

ERRFOR3059 = Errors(HISTOS_DIVMB[0].GetNbinsX())
ERRFOR3059.read(ERROR_BINNEDASLC3059, DICTEXTRA)
ERRSFOR3059.append(ERRFOR3059)

TOT_MULT_OVER_MB = calc_systematic_multovermb(ERRS[3], ERRSFOR3059[0], \
                                              HISTOS_DIVMB[2].GetNbinsX(), True)
TOT_MULT_OVER_MB_FD = calc_systematic_multovermb(ERRS[3], ERRSFOR3059[0], \
                                                 HISTOS_DIVMB[2].GetNbinsX(), True, True)
TOT_MULT_OVER_MB_WOFD = calc_systematic_multovermb(ERRS[3], ERRSFOR3059[0], \
                                                   HISTOS_DIVMB[2].GetNbinsX(), True, False)

ERRS_GR_DIV_TOT.append(Errors.make_root_asymm(HISTOS_DIVMB[2], TOT_MULT_OVER_MB, \
                                              const_x_err=0.3))
ERRS_GR_DIV_WOFD.append(Errors.make_root_asymm(HISTOS_DIVMB[2], TOT_MULT_OVER_MB_WOFD, \
                                               const_x_err=0.3))
ERRS_GR_DIV_FD.append(Errors.make_root_asymm(HISTOS_DIVMB[2], TOT_MULT_OVER_MB_FD, \
                                             const_x_err=0.3))
ERRS_GR_DIV_TOT[2].SetName("gr_TotSyst_%d" % 2)
ERRS_GR_DIV_FD[2].SetName("gr_FDSyst_%d" % 2)
ERRS_GR_DIV_WOFD[2].SetName("gr_TotSyst_woFD_%d" % 2)


TOT_MULT_OVER_MB = calc_systematic_multovermb(ERRS[4], ERRS[0], \
                                              HISTOS_DIVMB[3].GetNbinsX(), False)
TOT_MULT_OVER_MB_FD = calc_systematic_multovermb(ERRS[4], ERRS[0], \
                                                 HISTOS_DIVMB[3].GetNbinsX(), False, True)
TOT_MULT_OVER_MB_WOFD = calc_systematic_multovermb(ERRS[4], ERRS[0], \
                                                   HISTOS_DIVMB[3].GetNbinsX(), False, False)

ERRS_GR_DIV_TOT.append(Errors.make_root_asymm(HISTOS_DIVMB[3], TOT_MULT_OVER_MB, \
                                              const_x_err=0.3))
ERRS_GR_DIV_WOFD.append(Errors.make_root_asymm(HISTOS_DIVMB[3], TOT_MULT_OVER_MB_WOFD, \
                                               const_x_err=0.3))
ERRS_GR_DIV_FD.append(Errors.make_root_asymm(HISTOS_DIVMB[3], TOT_MULT_OVER_MB_FD, \
                                             const_x_err=0.3))
ERRS_GR_DIV_TOT[3].SetName("gr_TotSyst_%d" % 3)
ERRS_GR_DIV_FD[3].SetName("gr_FDSyst_%d" % 3)
ERRS_GR_DIV_WOFD[3].SetName("gr_TotSyst_woFD_%d" % 3)

# Save globally in Lc directory
SAVE_PATH = make_standard_save_path(f"LcpKpiAvgMultOverMB_{ANA_MB}_19_1029_3059_6099", PATH_OUT)
save_histograms([*HISTOS_DIVMB, *ERRS_GR_DIV_TOT, *ERRS_GR_DIV_FD, *ERRS_GR_DIV_WOFD], SAVE_PATH)


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

#Get Lc histograms
for mb in range(NMULTBINS):
    PATH_MIX = PATH_IN
    if mb == 4:
        PATH_MIX = PATH_IN_HM

    histo_ = extract_histo(CASE, ANA_MB, mb, "histoSigmaCorr", PATH_MIX)
    if mb == 4:
        histo_ = extract_histo(CASEAV, ANA_MB, mb, "histoSigmaCorr_average", PATH_MIX)
    histo_.SetName(f"histoSigmaCorr_Lc{mb}")

    if PATH_MIX == "inputfiles_QM19/":
        print("  NOTE: Scaling with 1. /", SIGMAV0, "(sigmaV0) for Lc")
        histo_.Scale(1./SIGMAV0)
    else:
        print("  No scaling for sigmaV0 (", SIGMAV0, "). Should be applied already in file")

    if mb != 4:
        print("  NOTE: Scaling with 1. /", BRLC, "(BR Lc)")
        histo_.Scale(1./BRLC)
    else:
        print("  No scaling for BR (", BRLC, "). Should be applied already in file")
    HISTOS_LC.append(histo_)

    DICTEXTRA = {}
    if mb != 4:
        if mb == 0:
            HISTOEFF = extract_histo(CASE, ANA_MB, mb, "hDirectEffpt", PATH_MIX, \
                                     FILEMB_LC_ALLHISTOS)
        else:
            HISTOEFF = extract_histo(CASE, ANA_MB, mb, "hDirectEffpt", PATH_MIX)
        ERROREFF = []
        for i in range(histo_.GetNbinsX()):
            RELERREFF = HISTOEFF.GetBinError(i+1) / HISTOEFF.GetBinContent(i+1)
            ERROREFF.append([0, 0, RELERREFF, RELERREFF])
        DICTEXTRA["statunceff"] = ERROREFF

    GRFD = extract_histo(CASE, ANA_MB, mb, "gFcCorrConservative", PATH_MIX)
    if mb == 4:
        GRFD = extract_histo(CASEAV, ANA_MB, mb, "gFcCorrConservative_average", PATH_MIX)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTEXTRA["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILESLC[mb], DICTEXTRA)
    ERRS_LC.append(errs)

#Get D0 histograms
for mb in range(NMULTBINS):
    PATH_MIX = PATHD0_BINNEDASLC
    if mb == 4:
        PATH_MIX = PATH_IN_HM

    histo_ = extract_histo(CASED0, ANA_MB, mb, "histoSigmaCorr", PATH_MIX)
    histo_.SetName(f"histoSigmaCorr_D0{mb}")
    if PATH_MIX == "inputfiles_QM19/D0_BinnedAsLc/":
        print("  NOTE: Scaling with 1. /", SIGMAV0, "for D0")
        histo_.Scale(1./SIGMAV0)
    else:
        print("  No scaling for sigmaV0 (", SIGMAV0, "). Should be applied already in file")
    print("  NOTE: Scaling with 1. /", BRD0, "(BR D0)")
    histo_.Scale(1./BRD0)
    HISTOS_D0.append(histo_)

    DICTEXTRA = {}
    HISTOEFF = extract_histo(CASED0, ANA_MB, mb, "hDirectEffpt", PATH_MIX)
    ERROREFF = []
    for i in range(histo_.GetNbinsX()):
        RELERREFF = HISTOEFF.GetBinError(i+1) / HISTOEFF.GetBinContent(i+1)
        ERROREFF.append([0, 0, RELERREFF, RELERREFF])
    DICTEXTRA["statunceff"] = ERROREFF

    GRFD = extract_histo(CASED0, ANA_MB, mb, "gFcConservative", PATH_MIX)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTEXTRA["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILESD0[mb], DICTEXTRA)
    ERRS_D0.append(errs)

HISTOS_LCOVERD0 = divide_by_eachother(HISTOS_LC, HISTOS_D0, [1, 1])

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
SAVE_PATH = make_standard_save_path(f"LcpKpiAvgOverD0_{ANA_MB}_1999_19_1029_3059_6099", PATH_OUT)
save_histograms([*HISTOS_LCOVERD0, *ERRS_GR_DIVD0_TOT, *ERRS_GR_DIVD0_FD, \
                 *ERRS_GR_DIVD0_WOFD], SAVE_PATH)


###########################################################################
##################### Plot Lc / D0 double ratio   #########################
###########################################################################

HISTO_DR_M = []
ERRS_GR_DR_M_TOT = []
ERRS_GR_DR_M_FD = []
ERRS_GR_DR_M_WOFD = []
HISTO_DR_M.append(divide_by_eachother([HISTOS_LCOVERD0[2]], [HISTOS_LCOVERD0[1]], \
                                      None, [1, 2, 4, 6, 8, 12])[0])
HISTO_DR_M.append(divide_by_eachother([HISTOS_LCOVERD0[3]], [HISTOS_LCOVERD0[1]], \
                                      None, [2, 4, 6, 8, 12])[0])
HISTO_DR_M.append(divide_by_eachother([HISTOS_LCOVERD0[4]], [HISTOS_LCOVERD0[1]], \
                                      None, [1, 2, 4, 6, 8, 12])[0])

for mb, _ in enumerate(HISTO_DR_M):
    SAMEMC = False #No statistics in common between [1-9] and other multiplicity bins
    dropbins = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    num = 2
    den = 1
    if mb == 1:
        dropbins = [[0, 1, 2, 3], [1, 2, 3, 4]]
        num = 3
    if mb == 2:
        num = 4
    tot_Lc_over_D0_DR = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], \
                                                         ERRS_LC[den], ERRS_D0[den], \
                                                         HISTO_DR_M[mb].GetNbinsX(), \
                                                         SAMEMC, dropbins)
    tot_Lc_over_D0_DR_FD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], \
                                                            ERRS_LC[den], ERRS_D0[den], \
                                                            HISTO_DR_M[mb].GetNbinsX(), \
                                                            SAMEMC, dropbins, True)
    tot_Lc_over_D0_DR_WOFD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], \
                                                              ERRS_LC[den], ERRS_D0[den], \
                                                              HISTO_DR_M[mb].GetNbinsX(), \
                                                              SAMEMC, dropbins, False)
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
FILENAME = f"LcpKpiAvgOverD0_DoubleRatioWith19_{ANA_MB}_1029_3059_6099"
SAVE_PATH = make_standard_save_path(FILENAME, PATH_OUT)
save_histograms([*HISTO_DR_M, *ERRS_GR_DR_M_TOT, *ERRS_GR_DR_M_FD, *ERRS_GR_DR_M_WOFD], SAVE_PATH)


HISTO_DR_MB = []
ERRS_GR_DR_MB_TOT = []
ERRS_GR_DR_MB_FD = []
ERRS_GR_DR_MB_WOFD = []
HISTO_DR_MB.append(divide_by_eachother_barlow([HISTOS_LCOVERD0[1]], [HISTOS_LCOVERD0[0]], \
                                               None, [1, 2, 4, 6, 8, 12])[0])
HISTO_DR_MB.append(divide_by_eachother_barlow([HISTOS_LCOVERD0[3]], [HISTOS_LCOVERD0[0]], \
                                               None, [2, 4, 6, 8, 12, 24])[0])
HISTO_DR_MB.append(divide_by_eachother_barlow([HISTOS_LCOVERD0[4]], [HISTOS_LCOVERD0[0]], \
                                               None, [1, 2, 4, 6, 8, 12, 24])[0])

for mb, _ in enumerate(HISTO_DR_MB):
    SAMEMC = True
    dropbins = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    num = 1
    den = 0
    if mb == 1:
        dropbins = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]
        num = 3
    if mb == 2: #HM bin, usually 4, but in this loop 2
        SAMEMC = False
        dropbins = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
        num = 4
    tot_Lc_over_D0_DR = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], ERRS_LC[den], \
                                                         ERRS_D0[den], \
                                                         HISTO_DR_MB[mb].GetNbinsX(), \
                                                         SAMEMC, dropbins)
    tot_Lc_over_D0_DR_FD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], \
                                                            ERRS_LC[den], ERRS_D0[den], \
                                                            HISTO_DR_MB[mb].GetNbinsX(), \
                                                            SAMEMC, dropbins, True)
    tot_Lc_over_D0_DR_WOFD = calc_systematic_mesondoubleratio(ERRS_LC[num], ERRS_D0[num], \
                                                              ERRS_LC[den], ERRS_D0[den], \
                                                              HISTO_DR_MB[mb].GetNbinsX(), \
                                                              SAMEMC, dropbins, False)
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
FILENAME = f"LcpKpiAvgOverD0_DoubleRatioWithMB_{ANA_MB}_19_3059_6099"
SAVE_PATH = make_standard_save_path(FILENAME, PATH_OUT)
save_histograms([*HISTO_DR_MB, *ERRS_GR_DR_MB_TOT, *ERRS_GR_DR_MB_FD, \
                 *ERRS_GR_DR_MB_WOFD], SAVE_PATH)
