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
#import pickle
#import numpy as np
#import pandas as pd
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TF1, TF2, TH1D
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils, AliHFInvMassMultiTrialFit
#from ROOT import RooRealVar, RooArgList, RooArgSet, RooLinkedList
#from ROOT import RooGenericPdf, RooDataSet, RooAbsData, RooFit
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed, kOrange
from ROOT import TLatex
from ROOT import gInterpreter, gPad
# HF specific imports
from machine_learning_hep.fitting.helpers import MLFitter
from machine_learning_hep.logger import get_logger
from machine_learning_hep.io import dump_yaml_from_dict
from machine_learning_hep.utilities import folding, get_bins, make_latex_table
from machine_learning_hep.utilities import parallelizer, openfile
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.analysis.analyzer import Analyzer

#from ROOT import RooUnfoldResponse
#from ROOT import RooUnfold
#from ROOT import RooUnfoldBayes
# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
# pylint: disable=invalid-name, too-many-arguments
class AnalyzerDhadrons_hfcorr(Analyzer):
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
        self.p_mass_fit_lim = datap["analysis"][self.typean]["mass_fit_lim"]
        self.p_bin_width = datap["analysis"][self.typean]["bin_width"]
        self.p_num_bins_2d = datap["analysis"][self.typean]["num_bins_2d"]
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                     self.p_bin_width))

        self.p_masspeaksec = None
        self.p_fix_sigmasec = None
        self.p_sigmaarraysec = None
        if self.p_sgnfunc[0] == 1:
            self.p_masspeaksec = datap["analysis"][self.typean]["masspeaksec"]
            self.p_fix_sigmasec = datap["analysis"][self.typean]["fix_sigmasec"]
            self.p_sigmaarraysec = datap["analysis"][self.typean]["sigmaarraysec"]

        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]

        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

        self.fitter = None
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)

        self.resultsdata = datap["analysis"][typean]["data"]["results"]
        self.resultsmc = datap["analysis"][typean]["mc"]["results"]
        self.nperiods = datap["multi"]["mc"]["nperiods"]

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc, n_filemass_name)
        self.n_filecross = datap["files_names"]["crossfilename"]

        # Output directories and filenames
        self.yields_filename = "yields"
        self.fits_dirname = "fits"
        self.yields_syst_filename = "yields_syst"
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

    @staticmethod
    def loadstyle():
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(1)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)

    @staticmethod
    def make_pre_suffix(args):
        """
        Construct a common file suffix from args
        """
        try:
            _ = iter(args)
        except TypeError:
            args = [args]
        else:
            if isinstance(args, str):
                args = [args]
        args = [str(a) for a in args]
        return "_".join(args)

    @staticmethod
    def make_file_path(directory, filename, extension, prefix=None, suffix=None):
        if prefix is not None:
            filename = Analyzer.make_pre_suffix(prefix) + "_" + filename
        if suffix is not None:
            filename = filename + "_" + Analyzer.make_pre_suffix(suffix)
        extension = extension.replace(".", "")
        return os.path.join(directory, filename + "." + extension)


    def min_par_fit(self, p1_0, p1_1, p1_2, p2_0, p2_1, p2_2, p3_0, p3_1, p3_2, p4_0,
                    p4_1, p4_2):
        gauss = "(%f*exp(-0.5*((x - %f)/%f)**2))" % (p1_0, p1_1, p1_2)
        pol = "(%f + %f * x + %f * x**2)" % (p2_0, p2_1, p2_2)
        gauss_not = "(%f*exp(-0.5*((y - %f)/%f)**2))" % (p3_0, p3_1, p3_2)
        pol_not = "(%f + %f * y + %f * y**2)" % (p4_0, p4_1, p4_2)
        fit_func = ("+[0] *" + gauss + "*" + gauss_not +
                    "+[1] *" + gauss + "*" + pol_not +
                    "+[2]*" + pol + "*" + gauss_not +
                    "+[3]*" + pol + "*" + pol_not)
        total_fit = TF2("total_fit", fit_func, self.p_mass_fit_lim[0],
                        self.p_mass_fit_lim[1], self.p_mass_fit_lim[0],
                        self.p_mass_fit_lim[1])
        return total_fit

#    @staticmethod
#    def roo_pdf(p1_0, p1_1, p1_2, p2_0, p2_1, p2_2, p3_0, p3_1, p3_2, p4_0,
#                p4_1, p4_2):
#        gauss = "(%f*exp(-0.5*((@0 - %f)/%f)**2))" % (p1_0, p1_1, p1_2)
#        pol = "(%f + %f * @0 + %f * @0**2)" %  (p2_0, p2_1, p2_2)
#        gauss_not = "(%f*exp(-0.5*((@1 - %f)/%f)**2))" %  (p3_0, p3_1, p3_2)
#        pol_not = "(%f + %f * @1 + %f * @1**2)" % (p4_0, p4_1, p4_2)
#        fit_func = ("@2*" + gauss + "*" + gauss_not +
#                    "+@3*" + gauss + "*" + pol_not +
#                    "+@4*" + pol + "*" + gauss_not +
#                    "+@5*" + pol + "*" + pol_not)
#        return fit_func


    def fit_tmp(self):
        rebin_const = int(self.p_num_bins/self.p_num_bins_2d)
        while self.p_num_bins % rebin_const != 0:
            rebin_const += -1
        sig_fun = TF1("signal fit function", "gaus", self.p_mass_fit_lim[0],
                      self.p_mass_fit_lim[1])
        bkg_fun = TF1("background fit function", "pol2", self.p_mass_fit_lim[0],
                      self.p_mass_fit_lim[1])
        myfilemc = TFile(self.n_filemass_mc, "read")
        print(self.n_filemass_mc)
        histomassmc_1_sig = myfilemc.Get("inv mass sig 1")
        histomassmc_1_bkg = myfilemc.Get("inv mass bkg 1")
        histomassmc_2_sig = myfilemc.Get("inv mass sig 2")
        histomassmc_2_bkg = myfilemc.Get("inv mass bkg 2")

        histomassmc_1_sig = histomassmc_1_sig.Rebin(rebin_const, "histomassmc_1_sig")
        histomassmc_1_sig.Scale(1/histomassmc_1_sig.Integral())
        histomassmc_1_sig.Fit(sig_fun)
        par_sig_1 = sig_fun.GetParameters()
        histomassmc_1_bkg = histomassmc_1_bkg.Rebin(rebin_const, "histomassmc_1_bkg")
        histomassmc_1_bkg.Scale(1/histomassmc_1_bkg.Integral())
        histomassmc_1_bkg.Fit(bkg_fun)
        par_bkg_1 = bkg_fun.GetParameters()
        histomassmc_2_sig = histomassmc_2_sig.Rebin(rebin_const, "histomassmc_2_sig")
        histomassmc_2_sig.Scale(1/histomassmc_2_sig.Integral())
        histomassmc_2_sig.Fit(sig_fun)
        par_sig_2 = sig_fun.GetParameters()
        histomassmc_2_bkg = histomassmc_2_bkg.Rebin(rebin_const, "histomassmc_2_bkg")
        histomassmc_2_bkg.Scale(1/histomassmc_2_bkg.Integral())
        histomassmc_2_bkg.Fit(bkg_fun)
        par_bkg_2 = bkg_fun.GetParameters()

        histomass_sig_sig = myfilemc.Get("hmass DDbar signal signal")
        histomass_sig_sig = histomass_sig_sig.RebinX(rebin_const,
                                                     "histomass_sig_sig")
        histomass_sig_sig = histomass_sig_sig.RebinY(rebin_const,
                                                     "histomass_sig_sig")
        num_sigsig = histomass_sig_sig.Integral()
        histomass_sig_bkg = myfilemc.Get("hmass DDbar signal background")
        histomass_sig_bkg = histomass_sig_bkg.RebinX(rebin_const,
                                                     "histomass_sig_bkg")
        histomass_sig_bkg = histomass_sig_bkg.RebinY(rebin_const,
                                                     "histomass_sig_bkg")
        num_sigbkg = histomass_sig_bkg.Integral()
        histomass_bkg_sig = myfilemc.Get("hmass DDbar background signal")
        histomass_bkg_sig = histomass_bkg_sig.RebinX(rebin_const,
                                                     "histomass_bkg_sig")
        histomass_bkg_sig = histomass_bkg_sig.RebinY(rebin_const,
                                                     "histomass_bkg_sig")
        num_bkgsig = histomass_bkg_sig.Integral()
        histomass_bkg_bkg = myfilemc.Get("hmass DDbar background background")
        histomass_bkg_bkg = histomass_bkg_bkg.RebinX(rebin_const,
                                                     "histomass_bkg_bkg")
        histomass_bkg_bkg = histomass_bkg_bkg.RebinY(rebin_const,
                                                     "histomass_bkg_bkg")
        num_bkgbkg = histomass_bkg_bkg.Integral()
        set_params = [par_sig_1[0], par_sig_1[1], par_sig_1[2],
                      par_bkg_1[0], par_bkg_1[1], par_bkg_1[2],
                      par_sig_2[0], par_sig_2[1], par_sig_2[2],
                      par_bkg_2[0], par_bkg_2[1], par_bkg_2[2]]
        fit_params = [num_sigsig, num_sigbkg, num_bkgsig, num_bkgbkg]
        return set_params, fit_params

    # pylint: disable=import-outside-toplevel
    def fit(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        rebin_const = int(self.p_num_bins/self.p_num_bins_2d)
        while self.p_num_bins % rebin_const != 0:
            rebin_const += -1

        print(self.d_resultsallpmc, self.d_resultsallpdata)
        myfilemc = TFile(self.n_filemass_mc, "read")
        histomass_full = myfilemc.Get("hmass DDbar Full data cand 1 Full data cand 2")
        histomass_full = histomass_full.RebinX(rebin_const,
                                               "histomass_full")
        histomass_full = histomass_full.RebinY(rebin_const,
                                               "histomass_full")
        print(histomass_full.GetNbinsX())
        print(histomass_full.GetNbinsY())
        set_params, fit_params = self.fit_tmp()
        print(set_params, fit_params)
        fit_fun_mc = self.min_par_fit(set_params[0], set_params[1],
                                      set_params[2], set_params[3],
                                      set_params[4], set_params[5],
                                      set_params[6], set_params[7],
                                      set_params[8], set_params[9],
                                      set_params[10], set_params[11])
        fit_fun_mc.SetParameters(fit_params[0], fit_params[1],
                                 fit_params[2], fit_params[3])
        fit_fun_data = fit_fun_mc
        #fit_fun_data.SetParLimits(0, 5e4, 1e6)
        #fit_fun_data.SetParLimits(1, 0., 1e6)
        #fit_fun_data.SetParLimits(2, 0., 1e6)
        #fit_fun_data.SetParLimits(3, 0., 5e4)
        #fitter.Config().ParSettings(2).SetLowerLimit(0);
        histomass_full.Fit(fit_fun_mc, "I")
        fileout_name = self.make_file_path(self.d_resultsallpmc, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        print(fileout_name)
        fileout = TFile(fileout_name, "RECREATE")
        histomass_full.SetOption("surf1")
        histomass_full.Write()
        fit_fun_mc.Write()
        fileout.Close()

        myfilemc = TFile(self.n_filemass, "read")
        histomass_full_data = myfilemc.Get("hmass DDbar Full data cand 1 Full data cand 2")
        histomass_full_data = histomass_full_data.RebinX(rebin_const,
                                                         "histomass_full_data")
        histomass_full_data = histomass_full_data.RebinY(rebin_const,
                                                         "histomass_full_data")
        histomass_full_data.Fit(fit_fun_data, "L")
        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")
        histomass_full_data.SetOption("surf1")
        histomass_full_data.Write()
        fit_fun_data.Write()
        fileout.Close()
        gROOT.SetBatch(tmp_is_root_batch)

#    def efficiency(self):
#        self.loadstyle()
#        tmp_is_root_batch = gROOT.IsBatch()
#        gROOT.SetBatch(True)
#
#        lfileeff = TFile.Open(self.n_fileff)
#        fileouteff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc, \
#                                 self.case, self.typean), "recreate")
#        cEff = TCanvas('cEff', 'The Fit Canvas')
#        cEff.SetCanvasSize(1900, 1500)
#        cEff.SetWindowSize(500, 500)
#
#        legeff = TLegend(.5, .65, .7, .85)
#        legeff.SetBorderSize(0)
#        legeff.SetFillColor(0)
#        legeff.SetFillStyle(0)
#        legeff.SetTextFont(42)
#        legeff.SetTextSize(0.035)
#
#        h_gen_pr = lfileeff.Get("h_gen_pr")
#        h_sel_pr = lfileeff.Get("h_sel_pr")
#        h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
#        h_sel_pr.SetMinimum(0.)
#        h_sel_pr.SetMaximum(1.5)
#        fileouteff.cd()
#        h_sel_pr.SetName("eff")
#        h_sel_pr.Write()
#        h_sel_pr.Draw("same")
#        legeff.AddEntry(h_sel_pr, "prompt efficiency", "LEP")
#        h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
#        h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s %s (1/GeV)" \
#                % (self.p_latexnmeson, self.typean))
#
#        h_gen_fd = lfileeff.Get("h_gen_fd")
#        h_sel_fd = lfileeff.Get("h_sel_fd")
#        h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
#        fileouteff.cd()
#        h_sel_fd.SetMinimum(0.)
#        h_sel_fd.SetMaximum(1.5)
#        h_sel_fd.SetName("eff_fd")
#        h_sel_fd.Write()
#        legeff.AddEntry(h_sel_pr, "feeddown efficiency", "LEP")
#        h_sel_pr.Draw("same")
#        legeff.Draw()
#        cEff.SaveAs("%s/Eff%s%s.eps" % (self.d_resultsallpmc,
#                                            self.case, self.typean))
#        print("Efficiency finished")
#        fileouteff.Close()
#
#    # pylint: disable=import-outside-toplevel
#    def makenormyields(self):
#        gROOT.SetBatch(True)
#        self.loadstyle()
#        print("making yields")
#        fileouteff = "%s/efficiencies%s%s.root" % \
#                      (self.d_resultsallpmc, self.case, self.typean)
#        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
#                                             None, [self.case, self.typean])
#        gROOT.LoadMacro("HFPtSpectrum.C")
#        from ROOT import HFPtSpectrum, HFPtSpectrum2
#        namehistoeffprompt = "eff"
#        namehistoefffeed = "eff_fd"
#        nameyield = "hyields"
#        fileoutcross = "%s/finalcross%s%s.root" % \
#                   (self.d_resultsallpdata, self.case, self.typean)
#        norm = -1
#        lfile = TFile.Open(self.n_filemass)
#        hNorm = lfile.Get("hEvForNorm")
#        normfromhisto = hNorm.GetBinContent(1)
#
#        HFPtSpectrum(self.p_indexhpt, self.p_inputfonllpred, \
#        fileouteff, namehistoeffprompt, namehistoefffeed, yield_filename, nameyield, \
#        fileoutcross, norm, self.p_sigmav0 * 1e12, self.p_fd_method, self.p_cctype)
#
#        cCross = TCanvas('cCross', 'The Fit Canvas')
#        cCross.SetCanvasSize(1900, 1500)
#        cCross.SetWindowSize(500, 500)
#        cCross.SetLogy()
#
#        legcross = TLegend(.5, .65, .7, .85)
#        legcross.SetBorderSize(0)
#        legcross.SetFillColor(0)
#        legcross.SetFillStyle(0)
#        legcross.SetTextFont(42)
#        legcross.SetTextSize(0.035)
#
#        myfile = TFile.Open(fileoutcross, "read")
#        hcross = myfile.Get("histoSigmaCorr")
#        hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnmeson)
#        hcross.GetYaxis().SetTitle("d#sigma/d#it{p}_{T} (%s) %s" %
#                                   (self.p_latexnmeson, self.typean))
#        legcross.AddEntry(hcross, "cross section", "LEP")
#        cCross.SaveAs("%s/Cross%s%sVs%s.eps" % (self.d_resultsallpdata,
#                                                self.case, self.typean, self.v_var_binning))
