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
from ROOT import Double, TCanvas, gPad
from machine_learning_hep.utilities_plot import Errors
from machine_learning_hep.utilities_plot import average_pkpi_pk0s

def extract_histo(histo_name, path):

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

SIGMAV0 = 57.8e9
BRPKPI = 0.0623
BRPK0S = 0.0109
PTBINS = 6

FILE_PKPI = "inputfiles_HP20/finalcrossLcpKpippMBvspt_ntrklmult4.root"
FILE_PK0S = "inputfiles_HP20/finalcrossLcpK0sppMBvspt_ntrklmult4.root"
FILE_PK0S2 = "inputfiles_HP20/FeeddownSyst_NbNbx2_LcpK0sppMBvspt_ntrkl.root"
FILE_OUT = "inputfiles_HP20/finalcrossLcAverageMBvspt_ntrklmult4.root"

ERROR_PKPI = "syst_HP20/syst_forLcaverage/LcpKpipp/errors_histoSigmaCorr_4.yaml"
ERROR_PK0S = "syst_HP20/syst_forLcaverage/LcpK0spp/errors_histoSigmaCorr_4.yaml"
ERROR_OUT = "syst_HP20/syst_forLcaverage/errors_avg_histoSigmaCorr_4.yaml"

gROOT.SetBatch(True)


#############################################################################
################################# Average ###################################
#############################################################################

HISTO_PKPI = extract_histo("histoSigmaCorr_rebin", FILE_PKPI)
HISTO_PKPI.SetName(f"histoSigmaCorr_pKpi")
HISTO_PKPI.Scale(1./BRPKPI)
HISTO_PK0S = extract_histo("histoSigmaCorr", FILE_PK0S)
HISTO_PK0S.SetName(f"histoSigmaCorr_pK0s")
HISTO_PK0S.Scale(1./BRPK0S)
HISTO_PK0S.Scale(1./SIGMAV0)

GRFD_PKPI = extract_histo("gFcCorrConservative", FILE_PKPI)
GRFD_PK0S = extract_histo("gNbCorrConservative", FILE_PK0S2)

DICTEXTRA_PKPI = {}
HISTOEFF_PKPI = extract_histo("hDirectEffpt", FILE_PKPI)
ERROREFF_PKPI = []
for i in range(HISTOEFF_PKPI.GetNbinsX()):
    print(HISTOEFF_PKPI.GetBinCenter(i+1))
    RELERREFF = HISTOEFF_PKPI.GetBinError(i+1) / HISTOEFF_PKPI.GetBinContent(i+1)
    if i == 0:
        ERROREFF_PKPI.append([0, 0, 99, 99])
    else:
        ERROREFF_PKPI.append([0, 0, RELERREFF, RELERREFF])
DICTEXTRA_PKPI["statunceff"] = ERROREFF_PKPI
ERRSPKPI = Errors(PTBINS)
ERRSPKPI.read(ERROR_PKPI, DICTEXTRA_PKPI)

DICTEXTRA_PK0S = {}
HISTOEFF_PK0S = extract_histo("hDirectEffpt", FILE_PK0S)
ERROREFF_PK0S = []
for i in range(HISTOEFF_PK0S.GetNbinsX()):
    RELERREFF = HISTOEFF_PK0S.GetBinError(i+1) / HISTOEFF_PK0S.GetBinContent(i+1)
    ERROREFF_PK0S.append([0, 0, RELERREFF, RELERREFF])
DICTEXTRA_PK0S["statunceff"] = ERROREFF_PK0S
ERRSPK0S = Errors(PTBINS)
ERRSPK0S.read(ERROR_PK0S, DICTEXTRA_PK0S)

MATCHPKPI = [-99, 1, 2, 3, 4, 5]
MATCHPK0S = [1, 2, 3, 4, 5, 6]
MATCHPKPIGR = [-99, 2, 3, 4, 5, 6] #Empty bin 0-1 still in fprompt tgraph
MATCHPK0SGR = [1, 2, 3, 4, 5, 6]

AVGCORRYIELD, AVGSTATUNC, AVGFPROMPT, \
  AVGFPROMPTLOW, AVGFPROMPTHIGH, AVGERROR = average_pkpi_pk0s(HISTO_PKPI, HISTO_PK0S,
                                                              GRFD_PKPI, GRFD_PK0S,
                                                              ERRSPKPI, ERRSPK0S,
                                                              MATCHPKPI, MATCHPK0S,
                                                              MATCHPKPIGR, MATCHPK0SGR)

HISTAVG = HISTO_PK0S.Clone("histoSigmaCorr_average")
GRFDAVG = TGraphAsymmErrors(PTBINS)
for ipt in range(PTBINS):
    HISTAVG.SetBinContent(ipt+1, AVGCORRYIELD[ipt])
    HISTAVG.SetBinError(ipt+1, AVGSTATUNC[ipt])
    GRFDAVG.SetPoint(ipt+1, GRFD_PK0S.GetX()[ipt+1], AVGFPROMPT[ipt])
    GRFDAVG.SetPointError(ipt+1, GRFD_PK0S.GetEXlow()[ipt+1], GRFD_PK0S.GetEXhigh()[ipt+1], \
                          AVGFPROMPTLOW[ipt], AVGFPROMPTHIGH[ipt])

AVGERROR.print()
print("\n\n Store above in", ERROR_OUT)

print("\n\n\nStoring ROOT objects in", FILE_OUT)
FOUT = TFile(FILE_OUT, "RECREATE")
FOUT.cd()
HISTAVG.Write("histoSigmaCorr_average")
HISTO_PKPI.Write("histoSigmaCorr_pKpi")
HISTO_PK0S.Write("histoSigmaCorr_pK0s")

GRFDAVG.Write("gFcCorrConservative_average")
GRFD_PKPI.Write("gFcCorrConservative_pKpi")
GRFD_PK0S.Write("gNbCorrConservative_pK0s")
FOUT.Close()
