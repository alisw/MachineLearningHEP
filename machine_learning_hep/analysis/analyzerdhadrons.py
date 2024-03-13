#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
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
#from array import array
#import itertools
# pylint: disable=import-error, no-name-in-module, unused-import, consider-using-f-string
# from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TF1, TH1D
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed, kOrange
from ROOT import TLatex
from ROOT import gInterpreter, gPad
# HF specific imports
from machine_learning_hep.fitting.helpers import MLFitter
from machine_learning_hep.logger import get_logger
from machine_learning_hep.io import dump_yaml_from_dict
from machine_learning_hep.utilities import folding, get_bins, make_latex_table, parallelizer
from machine_learning_hep.root import save_root_object
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.hf_pt_spectrum import hf_pt_spectrum
# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme


class AnalyzerDhadrons(Analyzer):  # pylint: disable=invalid-name
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)
        self.logger = get_logger()
        # namefiles pkl
        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.p_nptbins = len(self.lpt_finbinmin)
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]
        self.triggerbit = datap["analysis"][self.typean].get("triggerbit", "")

        dp = datap["analysis"][self.typean]
        self.d_prefix_mc = dp["mc"].get("prefix_dir_res")
        self.d_prefix_data = dp["data"].get("prefix_dir_res")
        self.d_resultsallpmc = self.d_prefix_mc + dp["mc"]["results"][period] \
            if period is not None \
            else self.d_prefix_mc + dp["mc"]["resultsallp"]
        self.d_resultsallpdata =  + dp["data"]["results"][period] \
            if period is not None \
            else self.d_prefix_data + dp["data"]["resultsallp"]

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(
            self.d_resultsallpmc, n_filemass_name)
        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']

        # Output directories and filenames
        self.yields_filename = "yields"
        self.fits_dirname = os.path.join(
            self.d_resultsallpdata, f"fits_{case}_{typean}")
        self.yields_syst_filename = "yields_syst"
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)
        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) /
                                    self.p_bin_width))
        # parameter fitter
        self.sig_fmap = {"kGaus": 0, "k2Gaus": 1, "kGausSigmaRatioPar": 2}
        self.bkg_fmap = {"kExpo": 0, "kLin": 1,
                         "Pol2": 2, "kNoBk": 3, "kPow": 4, "kPowEx": 5}
        # For initial fit in integrated mult bin
        self.init_fits_from = datap["analysis"][self.typean]["init_fits_from"]
        self.p_sgnfunc = datap["analysis"][self.typean]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"][self.typean]["bkgfunc"]
        self.p_masspeak = datap["analysis"][self.typean]["masspeak"]
        self.p_massmin = datap["analysis"][self.typean]["massmin"]
        self.p_massmax = datap["analysis"][self.typean]["massmax"]
        self.rebins = datap["analysis"][self.typean]["rebin"]

        self.p_includesecpeaks = datap["analysis"][self.typean].get(
            "includesecpeak", None)
        self.p_masssecpeak = datap["analysis"][self.typean].get(
            "masssecpeak", None)
        self.p_fix_masssecpeaks = datap["analysis"][self.typean].get(
            "fix_masssecpeak", None)
        self.p_widthsecpeak = datap["analysis"][self.typean].get(
            "widthsecpeak", None)
        self.p_fix_widthsecpeak = datap["analysis"][self.typean].get(
            "fix_widthsecpeak", None)
        if self.p_includesecpeaks is None:
            self.p_includesecpeaks = [False for ipt in range(self.p_nptbins)]
            self.p_masssecpeak = None
            self.p_fix_masssecpeaks = [False for ipt in range(self.p_nptbins)]
            self.p_widthsecpeak = None
            self.p_fix_widthsecpeak = None

        self.p_fixedmean = datap["analysis"][self.typean]["FixedMean"]
        self.p_use_user_gauss_sigma = datap["analysis"][self.typean]["SetInitialGaussianSigma"]
        self.p_max_perc_sigma_diff = datap["analysis"][self.typean]["MaxPercSigmaDeviation"]
        self.p_exclude_nsigma_sideband = datap["analysis"][self.typean]["exclude_nsigma_sideband"]
        self.p_nsigma_signal = datap["analysis"][self.typean]["nsigma_signal"]
        self.p_fixingaussigma = datap["analysis"][self.typean]["SetFixGaussianSigma"]
        self.p_use_user_gauss_mean = datap["analysis"][self.typean]["SetInitialGaussianMean"]
        self.p_dolike = datap["analysis"][self.typean]["dolikelihood"]
        self.p_sigmaarray = datap["analysis"][self.typean]["sigmaarray"]
        self.p_fixedsigma = datap["analysis"][self.typean]["FixedSigma"]
        self.p_casefit = datap["analysis"][self.typean]["fitcase"]
        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]
        self.p_dofullevtmerge = datap["dofullevtmerge"]
        self.p_dodoublecross = datap["analysis"][self.typean]["dodoublecross"]
        self.ptranges = self.lpt_finbinmin.copy()
        self.ptranges.append(self.lpt_finbinmax[-1])
        self.p_dobkgfromsideband = datap["analysis"][self.typean].get(
            "dobkgfromsideband", None)
        if self.p_dobkgfromsideband is None:
            self.p_dobkgfromsideband = False
        # More specific fit options
        self.include_reflection = datap["analysis"][self.typean].get(
            "include_reflection", False)

        self.p_nevents = datap["analysis"][self.typean]["nevents"]
        self.p_sigmamb = datap["analysis"]["sigmamb"]
        self.p_br = datap["ml"]["opt"]["BR"]

        # Systematics
        self.mt_syst_dict = datap["analysis"][self.typean].get(
            "systematics", None)
        self.d_mt_results_path = os.path.join(
            self.d_resultsallpdata, "multi_trial")

        self.p_anahpt = datap["analysis"]["anahptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_inputfonllpred = datap["analysis"]["inputfonllpred"]
        self.p_triggereff = datap["analysis"][self.typean].get("triggereff", [1])
        self.p_triggereffunc = datap["analysis"][self.typean].get(
            "triggereffunc", [0])

        self.root_objects = []

        # Fitting
        self.fitter = None
        self.p_performval = datap["analysis"].get(
            "event_cand_validation", None)

    # pylint: disable=import-outside-toplevel
    def fit(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        self.fitter = MLFitter(self.case, self.datap, self.typean,
                               self.n_filemass, self.n_filemass_mc)
        self.fitter.perform_pre_fits()
        self.fitter.perform_central_fits()
        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")
        self.fitter.draw_fits(self.d_resultsallpdata, fileout)
        fileout.Close()

        if self.p_dobkgfromsideband:
            self.fitter.bkg_fromsidebands(self.d_resultsallpdata, self.n_filemass,
                                          self.p_mass_fit_lim, self.p_bkgfunc, self.p_masspeak)

        self.fitter.save_fits(self.fits_dirname)
        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    def yield_syst(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        if not self.fitter:
            self.fitter = MLFitter(self.case, self.datap, self.typean,
                                   self.n_filemass, self.n_filemass_mc)
            if not self.fitter.load_fits(self.fits_dirname):
                self.logger.error(
                    "Cannot load fits from dir %s", self.fits_dirname)
                return

        # Additional directory needed where the intermediate results of the multi trial are
        # written to
        dir_yield_syst = os.path.join(self.d_resultsallpdata, "multi_trial")
        self.fitter.perform_syst(dir_yield_syst)
        # Directory of intermediate results and plot output directory are the same here
        self.fitter.draw_syst(dir_yield_syst, dir_yield_syst)

        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    def efficiency(self):
        self.loadstyle()
        print(self.n_fileff)
        lfileeff = TFile.Open(self.n_fileff)
        lfileeff.ls()
        fileouteff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc,
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
        h_sel_pr.Draw("same")
        fileouteff.cd()
        h_sel_pr.SetName("eff")
        h_sel_pr.Write()
        h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
        h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s %s (1/GeV)"
                                     % (self.p_latexnhadron, self.typean))
        h_sel_pr.SetMinimum(0.001)
        h_sel_pr.SetMaximum(1.0)
        gPad.SetLogy()
        cEff.SaveAs("%s/Eff%s%s.eps" % (self.d_resultsallpmc,
                                        self.case, self.typean))

        cEffFD = TCanvas('cEffFD', 'The Fit Canvas')
        cEffFD.SetCanvasSize(1900, 1500)
        cEffFD.SetWindowSize(500, 500)

        legeffFD = TLegend(.5, .65, .7, .85)
        legeffFD.SetBorderSize(0)
        legeffFD.SetFillColor(0)
        legeffFD.SetFillStyle(0)
        legeffFD.SetTextFont(42)
        legeffFD.SetTextSize(0.035)

        h_gen_fd = lfileeff.Get("h_gen_fd")
        h_sel_fd = lfileeff.Get("h_sel_fd")
        h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
        h_sel_fd.Draw("same")
        fileouteff.cd()
        h_sel_fd.SetName("eff_fd")
        h_sel_fd.Write()
        h_sel_fd.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
        h_sel_fd.GetYaxis().SetTitle("Acc x efficiency feed-down %s %s (1/GeV)"
                                     % (self.p_latexnhadron, self.typean))
        h_sel_fd.SetMinimum(0.001)
        h_sel_fd.SetMaximum(1.)
        gPad.SetLogy()
        legeffFD.Draw()
        cEffFD.SaveAs("%s/EffFD%s%s.eps" % (self.d_resultsallpmc,
                                            self.case, self.typean))


    @staticmethod
    def calculate_norm(self, hevents, hselevents): #TO BE FIXED WITH EV SEL
        if not hevents:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hevents")
        if not hselevents:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hselevents")

        n_events = hevents.Integral()
        n_selevents = hselevents.Integral()

        return n_events, n_selevents

    def makenormyields(self):  # pylint: disable=import-outside-toplevel, too-many-branches
        gROOT.SetBatch(True)
        self.loadstyle()

        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        if not os.path.exists(yield_filename):
            self.logger.fatal(
                "Yield file %s could not be found", yield_filename)

        fileouteff = f"{self.d_resultsallpmc}/efficiencies{self.case}{self.typean}.root"
        if not os.path.exists(fileouteff):
            self.logger.fatal(
                "Efficiency file %s could not be found", fileouteff)

        fileoutcross = "%s/finalcross%s%s.root" % \
            (self.d_resultsallpdata, self.case, self.typean)

        namehistoeffprompt = "eff"
        namehistoefffeed = "eff_fd"
        nameyield = "hyields0"

        histonorm = TH1F("histonorm", "histonorm", 1, 0, 1)

        filemass = TFile.Open(self.n_filemass)
        hevents = filemass.Get("all_events")
        hselevents = filemass.Get("sel_events")
        norm, selnorm = self.calculate_norm(self, hevents, hselevents)
        histonorm.SetBinContent(1, selnorm)
        self.logger.warning("Number of events %d", norm)
        self.logger.warning("Number of events after event selection %d", selnorm)

        if self.p_dobkgfromsideband:
            fileoutbkg = TFile.Open("%s/Background_fromsidebands_%s_%s.root" % \
                                    (self.d_resultsallpdata, self.case, self.typean))
            hbkg = fileoutbkg.Get("hbkg_fromsidebands")
            hbkg.Scale(1./selnorm)
            fileoutbkgscaled = TFile.Open("%s/NormBackground_fromsidebands_%s_%s.root" % \
                                          (self.d_resultsallpdata, self.case,
                                           self.typean), "RECREATE")
            fileoutbkgscaled.cd()
            hbkg.Write()
            fileoutbkgscaled.Close()

        hf_pt_spectrum(self.p_anahpt,
                           self.p_br,
                           self.p_inputfonllpred,
                           self.p_fd_method,
                           fileouteff,
                           namehistoeffprompt,
                           namehistoefffeed,
                           yield_filename,
                           nameyield,
                           selnorm,
                           self.p_sigmamb,
                           fileoutcross)

        fileoutcrosstot = TFile.Open("%s/finalcross%s%stot.root" %
                                     (self.d_resultsallpdata, self.case, self.typean), "recreate")

        f_fileoutcross = TFile.Open(fileoutcross)
        if f_fileoutcross:
            hcross = f_fileoutcross.Get("hptspectrum")
            fileoutcrosstot.cd()
            hcross.Write()
        histonorm.Write()
        fileoutcrosstot.Close()
