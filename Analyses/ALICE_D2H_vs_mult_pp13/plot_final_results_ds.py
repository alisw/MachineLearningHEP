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
from machine_learning_hep.utilities_plot import Errors
from machine_learning_hep.utilities_plot import calc_systematic_multovermb
from machine_learning_hep.utilities_plot import divide_all_by_first_multovermb
from machine_learning_hep.utilities_plot import divide_by_eachother
from machine_learning_hep.utilities_plot import calc_systematic_mesonratio
from machine_learning_hep.utilities_plot import plot_histograms, save_histograms

def extract_histo(case, ana_type, mult_bin, histo_name, filepath):

    path = f"{filepath}/finalcross{case}{ana_type}mult{mult_bin}.root"
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

CASE = "Dspp"
CASED0 = "D0pp"
ANA_MB = "MBvspt_ntrkl"
ANA_HM = "SPDvspt"
NMULTBINS = 5
SIGMAV0 = 57.8e9
BRDS = 0.0227
BRD0 = 0.0389

PATH_IN = "inputfiles_QM19/"
PATH_IN_HM = "inputfiles_HP20/"
PATH_OUT = "final_rootfiles/"
ERROR_FILESDS = ["syst_QM19/Dspp/errors_histoSigmaCorr_0.yaml",
                 "syst_QM19/Dspp/errors_histoSigmaCorr_1.yaml",
                 "syst_QM19/Dspp/errors_histoSigmaCorr_2.yaml",
                 "syst_QM19/Dspp/errors_histoSigmaCorr_3.yaml",
                 "syst_HP20/Dspp/errors_histoSigmaCorr_4.yaml"]
ERROR_FILESD0 = ["syst_QM19/D0pp/MBvspt_ntrkl_Dsbinning/errors_histoSigmaCorr_0.yaml",
                 "syst_QM19/D0pp/MBvspt_ntrkl_Dsbinning/errors_histoSigmaCorr_1.yaml",
                 "syst_QM19/D0pp/MBvspt_ntrkl_Dsbinning/errors_histoSigmaCorr_2.yaml",
                 "syst_QM19/D0pp/MBvspt_ntrkl_Dsbinning/errors_histoSigmaCorr_3.yaml",
                 "syst_HP20/D0pp/Dsbinning_errors_histoSigmaCorr_4.yaml"]

HISTOS = []
ERRS = []
ERRS_GR_TOT = []
ERRS_GR_FD = []
ERRS_GR_WOFD = []
gROOT.SetBatch(True)


#############################################################################
############################### Plot spectra ################################
#############################################################################

for mb in range(NMULTBINS):
    PATH_MIX = PATH_IN
    if mb == 4:
        PATH_MIX = PATH_IN_HM

    histo_ = extract_histo(CASE, ANA_MB, mb, "histoSigmaCorr", PATH_MIX)
    histo_.SetName(f"histoSigmaCorr_{mb}")
    print("\nAnalysing multiplicity interval", mb)
    if PATH_MIX == "inputfiles_QM19/":
        print("  NOTE: Scaling with 1. /", SIGMAV0, "(corr Y/ev)")
        histo_.Scale(1./SIGMAV0)
    else:
        print("  No scaling for sigmaV0 (", SIGMAV0, "). Should be applied already in file")
    print("  NOTE: Scaling with 1. /", BRDS, "(BR Ds)")
    histo_.Scale(1./BRDS)

    HISTOS.append(histo_)

    DICTEXTRA = {}
    HISTOEFF = extract_histo(CASE, ANA_MB, mb, "hDirectEffpt", PATH_MIX)
    ERROREFF = []
    for i in range(histo_.GetNbinsX()):
        RELERREFF = HISTOEFF.GetBinError(i+1) / HISTOEFF.GetBinContent(i+1)
        ERROREFF.append([0, 0, RELERREFF, RELERREFF])
    DICTEXTRA["statunceff"] = ERROREFF

    GRFD = extract_histo(CASE, ANA_MB, mb, "gFcConservative", PATH_MIX)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTEXTRA["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILESDS[mb], DICTEXTRA)
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
if NMULTBINS == 5:
    namefile = f"DsCorrectedYieldPerEvent_{ANA_MB}_1999_19_1029_3059_6099"
    SAVE_PATH = make_standard_save_path(namefile, PATH_OUT)
else:
    namefile = f"DsCorrectedYieldPerEvent_{ANA_MB}_1999_19_1029_3059"
    SAVE_PATH = make_standard_save_path(namefile, PATH_OUT)
save_histograms([*HISTOS, *ERRS_GR_TOT, *ERRS_GR_FD, *ERRS_GR_WOFD], SAVE_PATH)


#############################################################################
##################### Plot spectra mult / spectra MB ########################
#############################################################################

#Divide multiplicity spectra by MB
HISTOS_DIV = divide_all_by_first_multovermb(HISTOS)
#Remove MB one
HISTOS_DIVMB = HISTOS_DIV[1:]
ERRS_GR_DIV_TOT = []
ERRS_GR_DIV_WOFD = []
ERRS_GR_DIV_FD = []
for mb, _ in enumerate(HISTOS_DIVMB):
    SAMEMC = True
    if mb == 3: #HM bin, usually 4, but in this loop 3
        SAMEMC = False
    tot_mult_over_MB = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], \
                                                  HISTOS[0].GetNbinsX(), SAMEMC)
    tot_mult_over_MB_FD = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], \
                                                     HISTOS[0].GetNbinsX(), SAMEMC, True)
    tot_mult_over_MB_WOFD = calc_systematic_multovermb(ERRS[mb+1], ERRS[0], \
                                                       HISTOS[0].GetNbinsX(), SAMEMC, False)

    ERRS_GR_DIV_TOT.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB, \
                                                  const_x_err=0.3))
    ERRS_GR_DIV_WOFD.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB_WOFD, \
                                                   const_x_err=0.3))
    ERRS_GR_DIV_FD.append(Errors.make_root_asymm(HISTOS_DIVMB[mb], tot_mult_over_MB_FD, \
                                                 const_x_err=0.3))
    ERRS_GR_DIV_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_DIV_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_DIV_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save histograms + systematics in result directory
if NMULTBINS == 5:
    SAVE_PATH = make_standard_save_path(f"DsMultOverMB_{ANA_MB}_19_1029_3059_6099", PATH_OUT)
else:
    SAVE_PATH = make_standard_save_path(f"DsMultOverMB_{ANA_MB}_19_1029_3059", PATH_OUT)
save_histograms([*HISTOS_DIVMB, *ERRS_GR_DIV_TOT, *ERRS_GR_DIV_FD, *ERRS_GR_DIV_WOFD], SAVE_PATH)


###########################################################################
##################### Plot Ds / D0 (mult and MB)  #########################
###########################################################################

HISTOS_DS = []
HISTOS_D0 = []
ERRS_DS = []
ERRS_D0 = []
ERRS_GR_DIVD0_TOT = []
ERRS_GR_DIVD0_FD = []
ERRS_GR_DIVD0_WOFD = []

#Get Ds histograms
for mb in range(NMULTBINS):
    PATH_MIX = PATH_IN
    if mb == 4:
        PATH_MIX = PATH_IN_HM

    histo_ = extract_histo(CASE, ANA_MB, mb, "histoSigmaCorr", PATH_MIX)
    histo_.SetName(f"histoSigmaCorr_Ds{mb}")
    if PATH_MIX == "inputfiles_QM19/":
        print("  NOTE: Scaling with 1. /", SIGMAV0, "for Ds")
        histo_.Scale(1./SIGMAV0)
    else:
        print("  No scaling for sigmaV0 (", SIGMAV0, "). Should be applied already in file")
    HISTOS_DS.append(histo_)

    DICTEXTRA = {}
    HISTOEFF = extract_histo(CASE, ANA_MB, mb, "hDirectEffpt", PATH_MIX)
    ERROREFF = []
    for i in range(histo_.GetNbinsX()):
        RELERREFF = HISTOEFF.GetBinError(i+1) / HISTOEFF.GetBinContent(i+1)
        ERROREFF.append([0, 0, RELERREFF, RELERREFF])
    DICTEXTRA["statunceff"] = ERROREFF

    GRFD = extract_histo(CASE, ANA_MB, mb, "gFcConservative", PATH_MIX)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    for i in range(histo_.GetNbinsX()):
        ERRORNB.append([0, 0, EYLOW[i+1], EYHIGH[i+1], YVAL[i+1]])
    DICTEXTRA["feeddown_NB"] = ERRORNB

    errs = Errors(histo_.GetNbinsX())
    errs.read(ERROR_FILESDS[mb], DICTEXTRA)
    ERRS_DS.append(errs)

#Get D0 histograms
for mb in range(NMULTBINS):
    PATH_MIX = PATH_IN
    if mb == 4:
        PATH_MIX = PATH_IN_HM

    histo_ = extract_histo(CASED0, ANA_MB, mb, "histoSigmaCorr", PATH_MIX)
    histo_.SetName(f"histoSigmaCorr_D0{mb}")
    if PATH_MIX == "inputfiles_QM19/":
        print("  NOTE: Scaling with 1. /", SIGMAV0, "for D0")
        histo_.Scale(1./SIGMAV0)
    else:
        print("  No scaling for sigmaV0 (", SIGMAV0, "). Should be applied already in file")
    HISTOS_D0.append(histo_)

    DICTEXTRA = {}
    HISTOEFF = extract_histo(CASED0, ANA_MB, mb, "hDirectEffpt", PATH_MIX)
    ERROREFF = []
    #-1 because D0 has also bin [1-2]
    for i in range(histo_.GetNbinsX()-1):
        RELERREFF = HISTOEFF.GetBinError(i+2) / HISTOEFF.GetBinContent(i+2)
        ERROREFF.append([0, 0, RELERREFF, RELERREFF])
    DICTEXTRA["statunceff"] = ERROREFF

    GRFD = extract_histo(CASED0, ANA_MB, mb, "gFcConservative", PATH_MIX)
    ERRORNB = []
    EYHIGH = GRFD.GetEYhigh()
    EYLOW = GRFD.GetEYlow()
    YVAL = GRFD.GetY()
    #-1 because D0 has also bin [1-2]
    for i in range(histo_.GetNbinsX()-1):
        ERRORNB.append([0, 0, EYLOW[i+2], EYHIGH[i+2], YVAL[i+2]])
    DICTEXTRA["feeddown_NB"] = ERRORNB

    #-1 because D0 has also bin [1-2]
    errs = Errors(histo_.GetNbinsX()-1)
    errs.read(ERROR_FILESD0[mb], DICTEXTRA)
    ERRS_D0.append(errs)

HISTOS_DSOVERD0 = divide_by_eachother(HISTOS_DS, HISTOS_D0, [BRDS, BRD0], [2, 4, 6, 8, 12, 24])

for mb, _ in enumerate(HISTOS_DSOVERD0):
    tot_Ds_over_D0 = calc_systematic_mesonratio(ERRS_DS[mb], ERRS_D0[mb], \
                                                HISTOS_DSOVERD0[mb].GetNbinsX())
    tot_Ds_over_D0_FD = calc_systematic_mesonratio(ERRS_DS[mb], ERRS_D0[mb], \
                                                   HISTOS_DSOVERD0[mb].GetNbinsX(), True)
    tot_Ds_over_D0_WOFD = calc_systematic_mesonratio(ERRS_DS[mb], ERRS_D0[mb], \
                                                     HISTOS_DSOVERD0[mb].GetNbinsX(), False)
    ERRS_GR_DIVD0_TOT.append(Errors.make_root_asymm(HISTOS_DSOVERD0[mb], \
                                                    tot_Ds_over_D0, const_x_err=0.3))
    ERRS_GR_DIVD0_FD.append(Errors.make_root_asymm(HISTOS_DSOVERD0[mb], \
                                                   tot_Ds_over_D0_FD, const_x_err=0.3))
    ERRS_GR_DIVD0_WOFD.append(Errors.make_root_asymm(HISTOS_DSOVERD0[mb], \
                                                     tot_Ds_over_D0_WOFD, const_x_err=0.3))
    ERRS_GR_DIVD0_TOT[mb].SetName("gr_TotSyst_%d" % mb)
    ERRS_GR_DIVD0_FD[mb].SetName("gr_FDSyst_%d" % mb)
    ERRS_GR_DIVD0_WOFD[mb].SetName("gr_TotSyst_woFD_%d" % mb)

# Save histograms + systematics in result directory
if NMULTBINS == 5:
    SAVE_PATH = make_standard_save_path(f"DsOverD0_{ANA_MB}_1999_19_1029_3059_6099", PATH_OUT)
else:
    SAVE_PATH = make_standard_save_path(f"DsOverD0_{ANA_MB}_1999_19_1029_3059", PATH_OUT)
save_histograms([*HISTOS_DSOVERD0, *ERRS_GR_DIVD0_TOT, *ERRS_GR_DIVD0_FD, \
                 *ERRS_GR_DIVD0_WOFD], SAVE_PATH)
