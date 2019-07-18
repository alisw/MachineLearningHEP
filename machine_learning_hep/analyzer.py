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
from ROOT import TFile, TH1F, TCanvas
from machine_learning_hep.globalfitter import fitter

# pylint: disable=too-few-public-methods, too-many-instance-attributes
class Analyzer:
    species = "analyzer"
    def __init__(self, datap, case):


        #namefiles pkl
        self.case = case
        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"]["sel_an_binmax"]
        self.bin_matching = datap["analysis"]["binning_matching"]
        self.p_nptbins = len(self.lpt_finbinmin)
        self.lpt_probcutfin = datap["analysis"]["probcutoptimal"]

        self.lvar2_binmin = datap["analysis"]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"]["var_binning2"]
        self.p_nbin2 = len(self.lvar2_binmin)

        self.d_resultsallpmc = datap["analysis"]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"]["data"]["resultsallp"]

        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, self.n_filemass)
        self.n_filecross = datap["files_names"]["crossfilename"]
        self.p_mass_fit_lim = datap["analysis"]['mass_fit_lim']

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)

        self.p_bin_width = datap["analysis"]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        #parameter fitter
        self.p_sgnfunc = datap["analysis"]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"]["bkgfunc"]
        self.p_masspeak = datap["analysis"]["masspeak"]
        self.p_massmin = datap["analysis"]["massmin"]
        self.p_massmax = datap["analysis"]["massmax"]
        self.p_rebin = datap["analysis"]["rebin"]
        self.p_includesecpeak = datap["analysis"]["includesecpeak"]
        self.p_masssecpeak = datap["analysis"]["masssecpeak"]
        self.p_fixedmean = datap["analysis"]["FixedMean"]
        self.p_fixingaussigma = datap["analysis"]["SetFixGaussianSigma"]
        self.p_fixingausmean = datap["analysis"]["SetInitialGaussianMean"]
        self.p_dolike = datap["analysis"]["dolikelihood"]
        self.p_sigmaarray = datap["analysis"]["sigmaarray"]
        self.p_fixedsigma = datap["analysis"]["FixedSigma"]
        self.p_casefit = datap["analysis"]["fitcase"]
        self.p_dofullevtmerge = datap["dofullevtmerge"]

        self.ptranges = self.lpt_finbinmin.copy()
        self.ptranges.append(self.lpt_finbinmax[-1])
        self.lmult_yieldshisto = [TH1F("hyields%d" % (imult), "", \
            self.p_nptbins, array("d", self.ptranges)) for imult in range(self.p_nbin2)]

        self.p_nevents = datap["analysis"]["nevents"]
        self.p_sigmamb = datap["ml"]["opt"]["sigma_MB"]
        self.p_br = datap["ml"]["opt"]["BR"]

    def fitter(self):
        lfile = TFile.Open(self.n_filemass)
        fileout = TFile.Open("yields%s.root" % self.case, "recreate")
        for imult in range(self.p_nbin2):
            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%d_%d" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                h_invmass = lfile.Get("hmass" + suffix)
                rawYield, rawYieldErr = \
                    fitter(h_invmass, self.p_casefit, self.p_sgnfunc[ipt], self.p_bkgfunc[ipt], \
                    self.p_masspeak, self.p_rebin[ipt], self.p_dolike, self.p_fixingausmean, \
                    self.p_fixingaussigma, self.p_sigmaarray[ipt], self.p_massmin[ipt], \
                    self.p_massmax[ipt], self.p_fixedmean, self.p_fixedsigma, \
                    self.d_resultsallpdata, suffix)
                rawYield = rawYield/(self.lpt_finbinmax[ipt] - self.lpt_finbinmin[ipt])
                rawYieldErr = rawYieldErr/(self.lpt_finbinmax[ipt] - self.lpt_finbinmin[ipt])
                self.lmult_yieldshisto[imult].SetBinContent(ipt + 1, rawYield)
                self.lmult_yieldshisto[imult].SetBinError(ipt + 1, rawYieldErr)
            fileout.cd()
            self.lmult_yieldshisto[imult].Write()

        cYields = TCanvas('cYields', 'The Fit Canvas')
        cYields.SetLogy()
        lfile = TFile.Open("yields%s.root" % self.case)
        for imult in range(self.p_nbin2):
            self.lmult_yieldshisto[imult].SetMinimum(1)
            self.lmult_yieldshisto[imult].SetMaximum(1e14)
            self.lmult_yieldshisto[imult].SetLineColor(imult+1)
            self.lmult_yieldshisto[imult].Draw("same")
        cYields.SaveAs("Yields%s.pdf" % self.case)

    def efficiency(self):
        lfileeff = TFile.Open(self.n_fileff)
        fileouteff = TFile.Open("efficiencies%s.root" % self.case, "recreate")
        cEff = TCanvas('cEff', 'The Fit Canvas')
        for imult in range(self.p_nbin2):
            stringbin2 = "_%s_%d_%d" % (self.v_var2_binning,
                                        self.lvar2_binmin[imult],
                                        self.lvar2_binmax[imult])
            h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
            h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)

            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
            h_sel_pr.SetLineColor(imult+1)
            h_sel_pr.Draw("same")
            fileouteff.cd()
            h_sel_pr.SetName("eff_mult%d" % imult)
            h_sel_pr.Write()
        cEff.SaveAs("Eff%s.pdf" % self.case)

    def plotter(self):

        fileouteff = TFile.Open("efficiencies%s.root" % self.case)
        fileoutyield = TFile.Open("yields%s.root" % self.case)
        fileoutcross = TFile.Open("finalcross%s.root" % self.case, "recreate")
        cCross = TCanvas('cCross', 'The Fit Canvas')
        cCross.SetLogy()
        for imult in range(self.p_nbin2):
            heff = fileouteff.Get("eff_mult%d" % (imult))
            hcross = fileoutyield.Get("hyields%d" % (imult))
            hcross.Divide(heff)
            hcross.SetLineColor(imult+1)
            norm = 2 * self.p_br * self.p_nevents / (self.p_sigmamb * 1e12)
            hcross.Scale(1./norm)
            fileoutcross.cd()
            hcross.SetName("hcross%d" % imult)
            hcross.GetYaxis().SetRangeUser(1e4, 1e14)
            hcross.Draw("same")
            hcross.Write()
        cCross.SaveAs("Cross%s.pdf" % self.case)
