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
# pylint: disable=too-many-lines
import os
# pylint: disable=unused-wildcard-import, wildcard-import
#from array import *
import numpy as np
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TF1, TH1D
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils, AliHFInvMassMultiTrialFit
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed, kOrange
from ROOT import TLatex
from ROOT import gInterpreter, gPad
# HF specific imports
from machine_learning_hep.io import dump_yaml_from_dict
from machine_learning_hep.utilities import folding, get_bins, make_latex_table, parallelizer
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.analysis.analyzer import Analyzer

# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
class AnalyzerDhadrons(Analyzer):
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        self.p_sgnfunc = datap["analysis"][self.typean]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"][self.typean]["bkgfunc"]
        self.p_masspeak = datap["analysis"][self.typean]["masspeak"]
        self.p_massmin = datap["analysis"][self.typean]["massmin"]
        self.p_massmax = datap["analysis"][self.typean]["massmax"]
        self.p_rebin = datap["analysis"][self.typean]["rebin"]
        self.p_fix_mean = datap["analysis"][self.typean]["fix_mean"]
        self.p_fix_sigma = datap["analysis"][self.typean]["fix_sigma"]

        self.p_masspeaksec = None
        self.p_fix_sigmasec = None
        self.p_sigmaarraysec = None
        if self.p_sgnfunc[0] == 1:
            self.p_masspeaksec = datap["analysis"][self.typean]["masspeaksec"]
            self.p_fix_sigmasec = datap["analysis"][self.typean]["fix_sigmasec"]
            self.p_sigmaarraysec = datap["analysis"][self.typean]["sigmaarraysec"]

        self.fitter = None
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)


    # pylint: disable=import-outside-toplevel
    def fit(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        hyields = TH1F("hyields", "hyields", self.p_nptbins, self.analysis_bin_lims)
        for ipt in range(self.p_nptfinbins):
            print(self.p_sgnfunc[ipt])
            suffix = "%s%d_%d" % (self.v_var_binning, self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            print(self.n_filemass)
            myfilemc = TFile(self.n_filemass_mc, "read")
            histomassmc= myfilemc.Get("hmass_sig" + suffix)
            histomassmc_reb = AliVertexingHFUtils.RebinHisto(histomassmc, \
                                        self.p_rebin[ipt], -1)
            histomassmc_reb = TH1F()
            fittermc = AliHFInvMassFitter(histomassmc_reb, self.p_massmin[ipt], self.p_massmax[ipt],
                                        self.p_bkgfunc[ipt], 1)
            fittermc.SetInitialGaussianMean(self.p_masspeak)
            out=fittermc.MassFitter(0)
            print("I have made MC fit for sigma initialization")
            myfile = TFile(self.n_filemass, "read")
            histomass= myfile.Get("hmass" + suffix)
            histomass_reb = AliVertexingHFUtils.RebinHisto(histomass, \
                                        self.p_rebin[ipt], -1)
            histomass_reb = TH1F()
            fitter = AliHFInvMassFitter(histomass_reb, self.p_massmin[ipt], self.p_massmax[ipt],
                                        self.p_bkgfunc[ipt], self.p_sgnfunc[ipt])
            fitter.SetInitialGaussianSigma(fittermc.GetSigma())
            if self.p_fix_sigma[ipt] is True:
                fitter.SetFixGaussianSigma(fittermc.GetSigma())
            if self.p_sgnfunc[ipt] == 1:
                if self.p_fix_sigmasec[ipt] is True:
                    fitter.SetFixSecondGaussianSigma(self.p_sigmaarraysec[ipt])

            out=fitter.MassFitter(0)
            ry=fitter.GetRawYield()
            ery=fitter.GetRawYieldError()
            hyields.SetBinContent(ipt+1, ry)
            hyields.SetBinError(ipt+1, ery)

        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        print(fileout_name)
        fileout = TFile(fileout_name, "RECREATE")
        hyields.Write()
        fileout.Close()
        gROOT.SetBatch(tmp_is_root_batch)

    def efficiency(self):
        self.loadstyle()
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        lfileeff = TFile.Open(self.n_fileff)
        fileouteff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc, \
                                 self.case, self.typean), "recreate")
        cEff = TCanvas('cEff', 'The Fit Canvas')
        cEff.SetCanvasSize(1900, 1500)
        cEff.SetWindowSize(500, 500)

        legeff = TLegend(.5, .65, .7, .85)
        legeff.SetBorderSize(0)
        legeff.SetFillColor(0)
        legeff.SetFillStyle(0)
        legeff.SetTextFont(42)
        legeff.SetTextSize(0.035)

        h_gen_pr = lfileeff.Get("h_gen_pr")
        h_sel_pr = lfileeff.Get("h_sel_pr")
        h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
        h_sel_pr.SetMinimum(0.)
        h_sel_pr.SetMaximum(1.5)
        fileouteff.cd()
        h_sel_pr.SetName("eff")
        h_sel_pr.Write()
        h_sel_pr.Draw("same")
        legeff.AddEntry(h_sel_pr, "prompt efficiency", "LEP")
        h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
        h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s %s (1/GeV)" \
                % (self.p_latexnhadron, self.typean))

        h_gen_fd = lfileeff.Get("h_gen_fd")
        h_sel_fd = lfileeff.Get("h_sel_fd")
        h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
        fileouteff.cd()
        h_sel_fd.SetMinimum(0.)
        h_sel_fd.SetMaximum(1.5)
        h_sel_fd.SetName("eff_fd")
        h_sel_fd.Write()
        legeff.AddEntry(h_sel_pr, "feeddown efficiency", "LEP")
        h_sel_pr.Draw("same")
        legeff.Draw()
        cEff.SaveAs("%s/Eff%s%s.eps" % (self.d_resultsallpmc,
                                            self.case, self.typean))
        print("Efficiency finished")
        fileouteff.Close()

    # pylint: disable=import-outside-toplevel
    def makenormyields(self):
        gROOT.SetBatch(True)
        self.loadstyle()
        print("making yields")
        fileouteff = "%s/efficiencies%s%s.root" % \
                      (self.d_resultsallpmc, self.case, self.typean)
        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        gROOT.LoadMacro("HFPtSpectrum.C")
        from ROOT import HFPtSpectrum, HFPtSpectrum2
        namehistoeffprompt = "eff"
        namehistoefffeed = "eff_fd"
        nameyield = "hyields"
        fileoutcross = "%s/finalcross%s%s.root" % \
                   (self.d_resultsallpdata, self.case, self.typean)
        norm = -1
        lfile = TFile.Open(self.n_filemass)
        hNorm = lfile.Get("hEvForNorm")
        normfromhisto = hNorm.GetBinContent(1)

        HFPtSpectrum(self.p_indexhpt, self.p_inputfonllpred, \
        fileouteff, namehistoeffprompt, namehistoefffeed, yield_filename, nameyield, \
        fileoutcross, norm, self.p_sigmav0 * 1e12, self.p_fd_method, self.p_cctype)

        cCross = TCanvas('cCross', 'The Fit Canvas')
        cCross.SetCanvasSize(1900, 1500)
        cCross.SetWindowSize(500, 500)
        cCross.SetLogy()

        legcross = TLegend(.5, .65, .7, .85)
        legcross.SetBorderSize(0)
        legcross.SetFillColor(0)
        legcross.SetFillStyle(0)
        legcross.SetTextFont(42)
        legcross.SetTextSize(0.035)

        myfile = TFile.Open(fileoutcross, "read")
        hcross = myfile.Get("histoSigmaCorr")
        hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnhadron)
        hcross.GetYaxis().SetTitle("d#sigma/d#it{p}_{T} (%s) %s" %
                                   (self.p_latexnhadron, self.typean))
        legcross.AddEntry(hcross, "cross section", "LEP")
        cCross.SaveAs("%s/Cross%s%sVs%s.eps" % (self.d_resultsallpdata,
                                                self.case, self.typean, self.v_var_binning))
