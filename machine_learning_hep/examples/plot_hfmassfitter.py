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
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend, TLatex
from ROOT import Double
from ROOT import gROOT, kRed, kGreen, kBlack, kBlue, kOrange, kViolet, kAzure
from ROOT import TStyle, gPad
from machine_learning_hep.utilities import make_file_path
from machine_learning_hep.utilities_plot import load_root_style

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
def plot_hfmassfitter(case, arraytype):

    load_root_style()

    with open("../data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)

    folder_plots = data_param[case]["analysis"]["dir_general_plots"]
    if not os.path.exists(folder_plots):
        print("creating folder ", folder_plots)
        os.makedirs(folder_plots)

    v_var_binning = data_param[case]["var_binning"]
    lpt_finbinminMB = data_param[case]["analysis"][arraytype[0]]["sel_an_binmin"]
    lpt_finbinmaxMB = data_param[case]["analysis"][arraytype[0]]["sel_an_binmax"]
    bin_matchingMB = data_param[case]["analysis"][arraytype[0]]["binning_matching"]
    lpt_finbinminHM = data_param[case]["analysis"][arraytype[1]]["sel_an_binmin"]
    lpt_finbinmaxHM = data_param[case]["analysis"][arraytype[1]]["sel_an_binmax"]
    bin_matchingHM = data_param[case]["analysis"][arraytype[1]]["binning_matching"]
    lpt_probcutfin = data_param[case]["mlapplication"]["probcutoptimal"]
    ptranges = lpt_finbinminMB.copy()
    ptranges.append(lpt_finbinmaxMB[-1])
    p_nptbins = len(lpt_finbinminMB)

    lvar2_binminMB = data_param[case]["analysis"][arraytype[0]]["sel_binmin2"]
    lvar2_binmaxMB = data_param[case]["analysis"][arraytype[0]]["sel_binmax2"]
    v_var2_binningMB = data_param[case]["analysis"][arraytype[0]]["var_binning2"]
    lvar2_binminHM = data_param[case]["analysis"][arraytype[1]]["sel_binmin2"]
    lvar2_binmaxHM = data_param[case]["analysis"][arraytype[1]]["sel_binmax2"]
    v_var2_binningHM = data_param[case]["analysis"][arraytype[1]]["var_binning2"]
    p_nbin2 = len(lvar2_binminMB)

    name = data_param[case]["analysis"][arraytype[0]]["latexnamehadron"]
    latexbin2var = data_param[case]["analysis"][arraytype[0]]["latexbin2var"]
    plotbinMB = data_param[case]["analysis"][arraytype[0]]["plotbin"]
    plotbinHM = data_param[case]["analysis"][arraytype[1]]["plotbin"]

    d_resultsdataMB = data_param[case]["analysis"][arraytype[0]]["data"]["resultsallp"]
    d_resultsdataHM = data_param[case]["analysis"][arraytype[1]]["data"]["resultsallp"]
    yields_filename = "yields"


    signfhistos = [TH1F("hsignf%d" % (imult), "", \
                                    p_nptbins, array("d", ptranges)) \
                                    for imult in range(p_nbin2)]
    meanhistos = [TH1F("hmean%d" % (imult), "", \
                                    p_nptbins, array("d", ptranges)) \
                                    for imult in range(p_nbin2)]
    sigmahistos = [TH1F("hsigma%d" % (imult), "", \
                                    p_nptbins, array("d", ptranges)) \
                                    for imult in range(p_nbin2)]
    sighistos = [TH1F("hsig%d" % (imult), "", \
                                p_nptbins, array("d", ptranges)) \
                                for imult in range(p_nbin2)]
    backhistos = [TH1F("hback%d" % (imult), "", \
                                  p_nptbins, array("d", ptranges)) \
                                  for imult in range(p_nbin2)]

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        func_filename = make_file_path(d_resultsdataMB, yields_filename, "root",
                                       None, [case, arraytype[0]])
        func_file = TFile.Open(func_filename, "READ")

        for ipt in range(p_nptbins):
            bin_id = bin_matchingMB[ipt]
            suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (v_var_binning, lpt_finbinminMB[ipt],
                          lpt_finbinmaxMB[ipt], lpt_probcutfin[bin_id],
                          v_var2_binningMB, lvar2_binminMB[imult],
                          lvar2_binmaxMB[imult])
            load_dir = func_file.GetDirectory(suffix)
            mass_fitter = load_dir.Get("fitter")
            sign = 0
            esign = 0
            rootsign = Double(sign)
            rootesign = Double(esign)
            mass_fitter.Significance(3, rootsign, rootesign)
            signfhistos[imult].SetBinContent(ipt + 1, rootsign)
            signfhistos[imult].SetBinError(ipt + 1, rootesign)
            mean = mass_fitter.GetMean()
            emean = mass_fitter.GetMeanUncertainty()
            meanhistos[imult].SetBinContent(ipt + 1, mean)
            meanhistos[imult].SetBinError(ipt + 1, emean)
            sigma = mass_fitter.GetSigma()
            esigma = mass_fitter.GetSigmaUncertainty()
            sigmahistos[imult].SetBinContent(ipt + 1, sigma)
            sigmahistos[imult].SetBinError(ipt + 1, esigma)
            sig = mass_fitter.GetRawYield()
            esig = mass_fitter.GetRawYieldError()
            sighistos[imult].SetBinContent(ipt + 1, sig)
            sighistos[imult].SetBinError(ipt + 1, esig)
            back = 0
            eback = 0
            rootback = Double(back)
            rooteback = Double(eback)
            mass_fitter.Background(3, rootback, rooteback)
            backhistos[imult].SetBinContent(ipt + 1, rootback)
            backhistos[imult].SetBinError(ipt + 1, rooteback)

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        func_filename = make_file_path(d_resultsdataHM, yields_filename, "root",
                                       None, [case, arraytype[1]])
        func_file = TFile.Open(func_filename, "READ")

        for ipt in range(p_nptbins):
            bin_id = bin_matchingHM[ipt]
            suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (v_var_binning, lpt_finbinminHM[ipt],
                          lpt_finbinmaxHM[ipt], lpt_probcutfin[bin_id],
                          v_var2_binningHM, lvar2_binminHM[imult],
                          lvar2_binmaxHM[imult])
            load_dir = func_file.GetDirectory(suffix)
            mass_fitter = load_dir.Get("fitter")
            sign = 0
            esign = 0
            rootsign = Double(sign)
            rootesign = Double(esign)
            mass_fitter.Significance(3, rootsign, rootesign)
            signfhistos[imult].SetBinContent(ipt + 1, rootsign)
            signfhistos[imult].SetBinError(ipt + 1, rootesign)
            mean = mass_fitter.GetMean()
            emean = mass_fitter.GetMeanUncertainty()
            meanhistos[imult].SetBinContent(ipt + 1, mean)
            meanhistos[imult].SetBinError(ipt + 1, emean)
            sigma = mass_fitter.GetSigma()
            esigma = mass_fitter.GetSigmaUncertainty()
            sigmahistos[imult].SetBinContent(ipt + 1, sigma)
            sigmahistos[imult].SetBinError(ipt + 1, esigma)
            sig = mass_fitter.GetRawYield()
            esig = mass_fitter.GetRawYieldError()
            sighistos[imult].SetBinContent(ipt + 1, sig)
            sighistos[imult].SetBinError(ipt + 1, esig)
            back = 0
            eback = 0
            rootback = Double(back)
            rooteback = Double(eback)
            mass_fitter.Background(3, rootback, rooteback)
            backhistos[imult].SetBinContent(ipt + 1, rootback)
            backhistos[imult].SetBinError(ipt + 1, rooteback)

    #Significance fit plot
    csign = TCanvas('cSign', 'The Fit Canvas')
    csign.SetCanvasSize(1500, 1500)
    csign.SetWindowSize(500, 500)
    maxplot = 25
    if case == "D0pp":
        maxplot = 120
    if case == "Dspp":
        maxplot = 40
    csign.cd(1).DrawFrame(0, 0, 30, maxplot, ";#it{p}_{T} (GeV/#it{c});Significance %s" % name)

    leg = TLegend(.25, .65, .65, .85)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    colors = [kBlack, kRed, kGreen+2, kBlue, kViolet-1, kOrange+2, kAzure+1, kOrange-7]
    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        signfhistos[imult].SetLineColor(colors[imult % len(colors)])
        signfhistos[imult].SetMarkerColor(colors[imult % len(colors)])
        signfhistos[imult].SetMarkerStyle(21)
        signfhistos[imult].Draw("same")
        legyieldstring = "%.1f #leq %s < %.1f (MB)" % \
                    (lvar2_binminMB[imult], latexbin2var, lvar2_binmaxMB[imult])
        leg.AddEntry(signfhistos[imult], legyieldstring, "LEP")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        signfhistos[imult].SetLineColor(colors[imult % len(colors)])
        signfhistos[imult].SetMarkerColor(colors[imult % len(colors)])
        signfhistos[imult].SetMarkerStyle(21)
        signfhistos[imult].Draw("same")
        legyieldstring = "%.1f #leq %s < %.1f (HM)" % \
                    (lvar2_binminHM[imult], latexbin2var, lvar2_binmaxHM[imult])
        leg.AddEntry(signfhistos[imult], legyieldstring, "LEP")
    leg.Draw()
    csign.SaveAs("%s/MassFit_Signf_%s_%scombined%s.eps" % \
                 (folder_plots, case, arraytype[0], arraytype[1]))


    #Mean fit plot
    cmean = TCanvas('cMean', 'The Fit Canvas')
    cmean.SetCanvasSize(1500, 1500)
    cmean.SetWindowSize(500, 500)
    minplot = 2.27
    maxplot = 2.31
    if case == "D0pp":
        minplot = 1.85
        maxplot = 1.89
    if case == "Dspp":
        minplot = 1.95
        maxplot = 1.99
    cmean.cd(1).DrawFrame(0, minplot, 30, maxplot, ";#it{p}_{T} (GeV/#it{c});Mean %s" % name)

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        meanhistos[imult].SetLineColor(colors[imult % len(colors)])
        meanhistos[imult].SetMarkerColor(colors[imult % len(colors)])
        meanhistos[imult].SetMarkerStyle(21)
        meanhistos[imult].Draw("same")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        meanhistos[imult].SetLineColor(colors[imult % len(colors)])
        meanhistos[imult].SetMarkerColor(colors[imult % len(colors)])
        meanhistos[imult].SetMarkerStyle(21)
        meanhistos[imult].Draw("same")
    leg.Draw()
    cmean.SaveAs("%s/MassFit_Mean_%s_%scombined%s.eps" % \
                 (folder_plots, case, arraytype[0], arraytype[1]))


    #Sigma fit plot (to add MC!)
    csigm = TCanvas('cSigma', 'The Fit Canvas')
    csigm.SetCanvasSize(1500, 1500)
    csigm.SetWindowSize(500, 500)
    maxplot = 0.03
    if case == "D0pp":
        maxplot = 0.04
    csigm.cd(1).DrawFrame(0, 0, 30, maxplot, ";#it{p}_{T} (GeV/#it{c});Width %s" % name)

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        sigmahistos[imult].SetLineColor(colors[imult % len(colors)])
        sigmahistos[imult].SetMarkerColor(colors[imult % len(colors)])
        sigmahistos[imult].SetMarkerStyle(21)
        sigmahistos[imult].Draw("same")

    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        sigmahistos[imult].SetLineColor(colors[imult % len(colors)])
        sigmahistos[imult].SetMarkerColor(colors[imult % len(colors)])
        sigmahistos[imult].SetMarkerStyle(21)
        sigmahistos[imult].Draw("same")
    leg.Draw()
    csigm.SaveAs("%s/MassFit_Sigma_%s_%scombined%s.eps" % \
                 (folder_plots, case, arraytype[0], arraytype[1]))


    #Signal fit plot
    csig = TCanvas('cSig', 'The Fit Canvas')
    csig.SetCanvasSize(1500, 1500)
    csig.SetWindowSize(500, 500)
    csig.cd(1)

    #First draw HM for scale
    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        sighistos[imult].SetTitle(";#it{p}_{T} (GeV/#it{c});Signal %s" % name)
        sighistos[imult].SetLineColor(colors[imult % len(colors)])
        sighistos[imult].SetMarkerColor(colors[imult % len(colors)])
        sighistos[imult].SetMarkerStyle(21)
        sighistos[imult].Draw("same")

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        sighistos[imult].SetTitle(";#it{p}_{T} (GeV/#it{c});Signal %s" % name)
        sighistos[imult].SetLineColor(colors[imult % len(colors)])
        sighistos[imult].SetMarkerColor(colors[imult % len(colors)])
        sighistos[imult].SetMarkerStyle(21)
        sighistos[imult].Draw("same")
    leg.Draw()
    csig.SaveAs("%s/MassFit_Signal_%s_%scombined%s.eps" % \
                (folder_plots, case, arraytype[0], arraytype[1]))


    #Background fit plot
    cback = TCanvas('cBack', 'The Fit Canvas')
    cback.SetCanvasSize(1500, 1500)
    cback.SetWindowSize(500, 500)
    cback.cd(1)

    #First draw HM for scale
    for imult, iplot in enumerate(plotbinHM):
        if not iplot:
            continue
        backhistos[imult].SetTitle(";#it{p}_{T} (GeV/#it{c});Background %s" % name)
        backhistos[imult].SetLineColor(colors[imult % len(colors)])
        backhistos[imult].SetMarkerColor(colors[imult % len(colors)])
        backhistos[imult].SetMarkerStyle(21)
        backhistos[imult].Draw("same")

    for imult, iplot in enumerate(plotbinMB):
        if not iplot:
            continue
        backhistos[imult].SetTitle(";#it{p}_{T} (GeV/#it{c});Background %s" % name)
        backhistos[imult].SetLineColor(colors[imult % len(colors)])
        backhistos[imult].SetMarkerColor(colors[imult % len(colors)])
        backhistos[imult].SetMarkerStyle(21)
        backhistos[imult].Draw("same")
    leg.Draw()
    cback.SaveAs("%s/MassFit_Background_%s_%scombined%s.eps" % \
                 (folder_plots, case, arraytype[0], arraytype[1]))

#####################################

gROOT.SetBatch(True)

#EXAMPLE HOW TO USE plot_hfmassfitter
#  ---> Combines and plots the output of AliHFInvMassFitter in nice way
#plot_hfmassfitter("Dspp", ["MBvspt_ntrkl", "SPDvspt"])
