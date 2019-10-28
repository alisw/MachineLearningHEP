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
from array import *
from subprocess import Popen
import numpy as np
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TF1, TH1D
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed, kOrange
from ROOT import TLatex
from ROOT import gInterpreter, gPad
# HF specific imports
from machine_learning_hep.globalfitter import Fitter
from  machine_learning_hep.logger import get_logger
from  machine_learning_hep.io import dump_yaml_from_dict
from machine_learning_hep.utilities import folding, get_bins, plot_histograms, make_latex_table, \
                                           parallelizer
#from ROOT import RooUnfoldResponse
#from ROOT import RooUnfold
#from ROOT import RooUnfoldBayes
# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
class Analyzer:
    species = "analyzer"
    def __init__(self, datap, case, typean,
                 resultsdata, resultsmc, valdata, valmc):

        self.logger = get_logger()
        #namefiles pkl
        self.case = case
        self.typean = typean
        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.p_nptbins = len(self.lpt_finbinmin)
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]

        self.lvar2_binmin = datap["analysis"][self.typean]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"][self.typean]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]
        self.v_var2_binning_gen = datap["analysis"][self.typean]["var_binning2_gen"]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.p_nbin2 = len(self.lvar2_binmin)

        self.d_resultsallpmc = resultsmc
        self.d_resultsallpdata = resultsdata
        self.p_corrmb_typean = datap["analysis"][self.typean]["corresp_mb_typean"]
        if self.p_corrmb_typean is not None:
            self.results_mb = datap["analysis"][self.p_corrmb_typean]["data"]["resultsallp"]

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc, n_filemass_name)
        self.n_filecross = datap["files_names"]["crossfilename"]
        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']

        # Output directories and filenames
        self.yields_filename = "yields"
        self.yields_syst_filename = "yields_syst"
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)
        self.n_evtvalroot = datap["files_names"]["namefile_evtvalroot"]

        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        #parameter fitter
        self.sig_fmap = {"kGaus": 0, "k2Gaus": 1, "kGausSigmaRatioPar": 2}
        self.bkg_fmap = {"kExpo": 0, "kLin": 1, "Pol2": 2, "kNoBk": 3, "kPow": 4, "kPowEx": 5}
        # For initial fit in integrated mult bin
        self.init_fits_from = datap["analysis"][self.typean]["init_fits_from"]
        self.p_sgnfunc = datap["analysis"][self.typean]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"][self.typean]["bkgfunc"]
        self.p_masspeak = datap["analysis"][self.typean]["masspeak"]
        self.p_massmin = datap["analysis"][self.typean]["massmin"]
        self.p_massmax = datap["analysis"][self.typean]["massmax"]
        # Enable rebinning per pT and multiplicity
        # Note that this is not a deepcopy in case it's already a list of lists
        self.rebins = datap["analysis"][self.typean]["rebin"].copy()
        if not isinstance(self.rebins[0], list):
            self.rebins = [self.rebins for _ in range(self.p_nbin2)]

        self.p_includesecpeaks = datap["analysis"][self.typean].get("includesecpeak", None)
        if self.p_includesecpeaks is None:
            self.p_includesecpeaks = [False for ipt in range(self.p_nptbins)]
        # Now we have a list, either the one given by the user or the default one just filled above
        self.p_includesecpeaks = self.p_includesecpeaks.copy()
        if not isinstance(self.p_includesecpeaks[0], list):
            self.p_inculdesecpeaks = [self.p_includesecpeaks for _ in range(self.p_nbin2)]

        self.p_masssecpeak = datap["analysis"][self.typean]["masssecpeak"] \
                if self.p_includesecpeaks else None

        self.p_fix_masssecpeaks = datap["analysis"][self.typean].get("fix_masssecpeak", None)
        if self.p_fix_masssecpeaks is None:
            self.p_fix_masssecpeaks = [False for ipt in range(self.p_nptbins)]
        # Now we have a list, either the one given by the user or the default one just filled above
        self.p_fix_masssecpeaks = self.p_fix_masssecpeaks.copy()
        if not isinstance(self.p_fix_masssecpeaks[0], list):
            self.p_fix_masssecpeaks = [self.p_fix_masssecpeaks for _ in range(self.p_nbin2)]

        self.p_widthsecpeak = datap["analysis"][self.typean]["widthsecpeak"] \
                if self.p_includesecpeaks else None
        self.p_fix_widthsecpeak = datap["analysis"][self.typean]["fix_widthsecpeak"] \
                if self.p_includesecpeaks else None
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
        self.p_latexnmeson = datap["analysis"][self.typean]["latexnamemeson"]
        self.p_latexbin2var = datap["analysis"][self.typean]["latexbin2var"]
        self.p_dofullevtmerge = datap["dofullevtmerge"]
        self.p_dodoublecross = datap["analysis"][self.typean]["dodoublecross"]
        self.ptranges = self.lpt_finbinmin.copy()
        self.ptranges.append(self.lpt_finbinmax[-1])
        self.var2ranges = self.lvar2_binmin.copy()
        self.var2ranges.append(self.lvar2_binmax[-1])
        # More specific fit options
        self.include_reflection = datap["analysis"][self.typean].get("include_reflection", False)
        print(self.var2ranges)

        self.p_nevents = datap["analysis"][self.typean]["nevents"]
        self.p_bineff = datap["analysis"][self.typean]["usesinglebineff"]
        self.p_fprompt_from_mb = datap["analysis"][self.typean]["fprompt_from_mb"]
        self.p_sigmamb = datap["ml"]["opt"]["sigma_MB"]
        self.p_br = datap["ml"]["opt"]["BR"]

        self.d_valevtdata = valdata
        self.d_valevtmc = valmc

        self.f_evtvaldata = os.path.join(self.d_valevtdata, self.n_evtvalroot)
        self.f_evtvalmc = os.path.join(self.d_valevtmc, self.n_evtvalroot)

        self.f_evtnorm = os.path.join(self.d_resultsallpdata, "correctionsweights.root")

        # Systematics
        self.mt_syst_dict = datap["analysis"][self.typean].get("systematics", None)
        self.d_mt_results_path = os.path.join(self.d_resultsallpdata, "multi_trial")

        self.p_indexhpt = datap["analysis"]["indexhptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_sigmav0 = datap["analysis"]["sigmav0"]
        self.p_triggereff = datap["analysis"][self.typean].get("triggereff", [1] * 10)
        self.p_triggereffunc = datap["analysis"][self.typean].get("triggereffunc", [0] * 10)

        self.apply_weights = datap["analysis"][self.typean]["triggersel"]["weighttrig"]
        self.root_objects = []

        self.get_crossmb_from_path = datap["analysis"][self.typean].get("get_crossmb_from_path", \
                                                                        None)
        self.path_for_crossmb = datap["analysis"][self.typean].get("path_for_crossmb", None)

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

    def test_aliphysics(self):
        test_macro = "/tmp/aliphysics_test.C"
        with open(test_macro, "w") as m:
            m.write("void aliphysics_test()\n{\n")
            m.write("TH1F* h = new TH1F(\"name\", \"\", 2, 1., 2.);\n \
                    auto fitter = new AliHFInvMassFitter(h, 1., 2., 1, 1);\n \
                    if(fitter) { std::cerr << \" Success \"; \n \
                    delete fitter; }\n \
                    else { std::cerr << \"Fail\"; }\n \
                    std::cerr << std::endl; }")
        proc = Popen(["root", "-l", "-b", "-q", test_macro])
        success = proc.wait()
        if success != 0:
            self.logger.fatal("You are not in the AliPhysics env")


    # pylint: disable=too-many-branches, too-many-locals, too-many-nested-blocks
    def fitter(self):
        # Test if we are in AliPhysics env
        #self.test_aliphysics()
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        from ROOT import AliHFInvMassFitter, AliVertexingHFUtils
        # Enable ROOT batch mode and reset in the end

        self.loadstyle()

        # Immediately fail if something weird was chosen for fit init
        if self.init_fits_from[0] not in  ["mc", "data"]:
            self.logger.fatal("Fit can only be initialized from \"data\" or \"mc\"")

        lfile = TFile.Open(self.n_filemass, "READ")
        lfile_mc = TFile.Open(self.n_filemass_mc, "READ")

        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")
        # Summarize in mult histograms in pT bins
        yieldshistos = [TH1F("hyields%d" % (imult), "", \
                self.p_nptbins, array("d", self.ptranges)) for imult in range(self.p_nbin2)]
        means_histos = [TH1F("hmeanss%d" % (imult), "", \
                self.p_nptbins, array("d", self.ptranges)) for imult in range(self.p_nbin2)]
        sigmas_histos = [TH1F("hsigmas%d" % (imult), "", \
                self.p_nptbins, array("d", self.ptranges)) for imult in range(self.p_nbin2)]
        have_summary_pt_bins = []
        means_init_mc_histos = TH1F("hmeans_init_mc", "",
                                    self.p_nptbins, array("d", self.ptranges))
        sigmas_init_mc_histos = TH1F("hsigmas_init_mc", "",
                                     self.p_nptbins, array("d", self.ptranges))
        means_init_data_histos = TH1F("hmeans_init_data", "",
                                      self.p_nptbins, array("d", self.ptranges))
        sigmas_init_data_histos = TH1F("hsigmas_init_data", "",
                                       self.p_nptbins, array("d", self.ptranges))

        if self.p_nptbins < 9:
            nx = 4
            ny = 2
            canvy = 533
        elif self.p_nptbins < 13:
            nx = 4
            ny = 3
            canvy = 800
        else:
            nx = 5
            ny = 4
            canvy = 1200
        canvas_init_mc = TCanvas("canvas_init_mc", "MC", 1000, canvy)
        canvas_init_data = TCanvas("canvas_init_data", "Data", 1000, canvy)
        canvas_data = [TCanvas("canvas_data%d" % (imult), "Data", 1000, canvy) \
                       for imult in range(self.p_nbin2)]
        canvas_init_mc.Divide(nx, ny)
        canvas_init_data.Divide(nx, ny)
        for imult in range(self.p_nbin2):
            canvas_data[imult].Divide(nx, ny)

        # Fit mult integrated MC and data in integrated multiplicity bin for all pT bins
        # Hence, extract respective bin of second variable
        bin_mult_int = self.p_bineff if self.p_bineff is not None else 0
        mult_int_min = self.lvar2_binmin[bin_mult_int]
        mult_int_max = self.lvar2_binmax[bin_mult_int]

        fit_status = {}
        mean_min = 9999.
        mean_max = 0.
        sigma_min = 9999.
        sigma_max = 0.
        minperc = 1 - self.p_max_perc_sigma_diff
        maxperc = 1 + self.p_max_perc_sigma_diff
        mass_fitter = []
        ifit = -1

        # Start fitting...
        for imult in range(self.p_nbin2):
            if imult not in fit_status:
                fit_status[imult] = {}
            mass_fitter_mc_init = []
            mass_fitter_data_init = []
            # Collect additional objects to be drawn outside of pT loop
            add_draw_objects = []
            for ipt in range(self.p_nptbins):
                if ipt not in fit_status[imult]:
                    fit_status[imult][ipt] = {}
                bin_id = self.bin_matching[ipt]

                # Initialize mean and sigma with user seeds. This is also the fallback if initial
                # MC and data fits fail
                mean_for_data = self.p_masspeak
                sigma_for_data = self.p_sigmaarray[ipt]
                means_sigmas_init = [(3, mean_for_data, sigma_for_data)]

                # Store mean, sigma asked to be used by user
                mean_case_user = None
                sigma_case_user = None

                ########################
                # START initialize fit #
                ########################
                # Get integrated histograms
                fit_status[imult][ipt]["init_MC"] = {}
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, mult_int_min, mult_int_max)

                suffix_write = "%s%d_%d_%s_%.2f_%.2f" % \
                               (self.v_var_binning, self.lpt_finbinmin[ipt],
                                self.lpt_finbinmax[ipt],
                                self.v_var2_binning, mult_int_min, mult_int_max)
                h_invmass_mc_init = lfile_mc.Get("hmass_sig" + suffix)

                h_mc_init_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass_mc_init,
                                                                  self.rebins[bin_mult_int][ipt],
                                                                  -1)
                h_mc_init_rebin = TH1F()
                h_mc_init_rebin_.Copy(h_mc_init_rebin)
                h_mc_init_rebin.SetTitle("%.1f < #it{p}_{T} < %.1f (prob > %.2f)" \
                                         % (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt], \
                                            self.lpt_probcutfin[bin_id]))
                h_mc_init_rebin.GetXaxis().SetTitle("#it{M}_{inv} (GeV/#it{c}^{2})")
                h_mc_init_rebin.GetYaxis().SetTitle("Entries/(%.0f MeV/#it{c}^{2})" \
                                                    % (h_mc_init_rebin.GetBinWidth(1) * 1000))
                h_mc_init_rebin.GetYaxis().SetTitleOffset(1.1)

                #mass_fitter_mc_init.append(AliHFInvMassFitter(h_mc_init_rebin, self.p_massmin[ipt],
                #                                              self.p_massmax[ipt],
                #                                              self.bkg_fmap[self.p_bkgfunc[ipt]],
                #                                              self.sig_fmap[self.p_sgnfunc[ipt]]))

                # Force to go on with final fit although there is no estimated signal
                #mass_fitter_mc_init[ipt].SetCheckSignalCountsAfterFirstFit(False)
                #if self.p_dolike:
                #    mass_fitter_mc_init[ipt].SetUseLikelihoodFit()
                #mass_fitter_mc_init[ipt].SetInitialGaussianMean(mean_for_data)
                #mass_fitter_mc_init[ipt].SetInitialGaussianSigma(sigma_for_data)
                #mass_fitter_mc_init[ipt].SetNSigma4SideBands(self.p_exclude_nsigma_sideband)
                #success = mass_fitter_mc_init[ipt].MassFitter(False)
                success = False
                user_init_success = False
                for r in [2., 3., 4., 5., 6., 7.]:
                    guess_mean = h_mc_init_rebin.GetMean()
                    guess_sigma = h_mc_init_rebin.GetRMS()
                    guess_low_range = guess_mean - r * guess_sigma
                    guess_up_range = guess_mean + r * guess_sigma
                    guess_int = h_mc_init_rebin.Integral(h_mc_init_rebin.FindBin(guess_low_range),
                                                         h_mc_init_rebin.FindBin(guess_up_range),
                                                         "width")
                    gaus_func = TF1("mc_init_gaus", "gaus", guess_low_range, guess_up_range)
                    gaus_func.SetParameter(0, guess_int)
                    gaus_func.SetParameter(1, guess_mean)
                    gaus_func.SetParameter(2, guess_sigma)
                    h_mc_init_rebin.Fit(gaus_func, "BEL0+", "", guess_low_range, guess_up_range)
                    int_tmp = gaus_func.GetParameter(0)
                    mean_tmp = gaus_func.GetParameter(1)
                    mean_err_tmp = gaus_func.GetParError(1)
                    sigma_tmp = gaus_func.GetParameter(2)
                    sigma_err_tmp = gaus_func.GetParError(2)
                    ndf = gaus_func.GetNDF()
                    chi2 = gaus_func.GetChisquare()
                    chi2ndf = chi2 / ndf if ndf > 0 else 0
                    print("########################")
                    print(f"Fit n RMS {r}")
                    print(f"Guessed sigma: {guess_sigma}")
                    print(f"fit range: {guess_low_range} - {guess_up_range}")
                    print(f"fitted sigma {sigma_tmp}")
                    print(f"chi2 / NDF {chi2ndf}")

                    if int_tmp * sigma_tmp > 0 and \
                            guess_mean - guess_sigma < mean_tmp < guess_mean + guess_sigma and \
                            sigma_tmp < 1.1 * guess_sigma and chi2ndf > 0:
                        success = True
                        mass_fitter_mc_init.append(gaus_func)
                        break


                fit_status[imult][ipt]["init_MC"]["sigma"] = -1.
                fit_status[imult][ipt]["init_MC"]["success"] = success
                if not success:
                    self.logger.error("Could not do initial fit on MC")

                else:
                    mean_for_data = mass_fitter_mc_init[-1].GetParameter(1)
                    sigma_for_data = mass_fitter_mc_init[-1].GetParameter(2)
                    mean_err_tmp = mass_fitter_mc_init[-1].GetParError(1)
                    sigma_err_tmp = mass_fitter_mc_init[-1].GetParError(2)
                    means_sigmas_init.insert(0, (2, mean_for_data, sigma_for_data))
                    fit_status[imult][ipt]["init_MC"]["sigma"] = sigma_for_data
                    if ipt not in have_summary_pt_bins:
                        means_init_mc_histos.SetBinContent(ipt + 1, mean_for_data)
                        means_init_mc_histos.SetBinError(ipt + 1, mean_err_tmp)
                        sigmas_init_mc_histos.SetBinContent(ipt + 1, sigma_for_data)
                        sigmas_init_mc_histos.SetBinError(ipt + 1, sigma_err_tmp)

                    if self.init_fits_from[ipt] == "mc":
                        user_init_success = True
                        mean_case_user = mean_for_data
                        sigma_case_user = sigma_for_data
                canvas = TCanvas("fit_canvas_mc_init", suffix_write, 700, 700)
                canvas.cd()
                h_mc_init_rebin.Draw()
                mass_fitter_mc_init[-1].SetLineColor(kBlue)
                mass_fitter_mc_init[-1].Draw("same")

                #mass_fitter_mc_init[ipt].DrawHere(canvas, self.p_nsigma_signal)
                canvas.SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                  "fittedplot_integrated_mc", "eps",
                                                  None, suffix_write))
                canvas.Close()
                canvas_init_mc.cd(ipt+1)
                h_mc_init_rebin.Draw()
                mass_fitter_mc_init[-1].Draw("same")
                mass_fitter_mc_init.append(h_mc_init_rebin)
                #mass_fitter_mc_init[ipt].DrawHere(gPad, self.p_nsigma_signal)

                # Now, try also for data
                fit_status[imult][ipt]["init_data"] = {}
                histname = "hmass"
                if self.apply_weights is True:
                    histname = "h_invmass_weight"
                    self.logger.info("*********** I AM USING WEIGHTED HISTOGRAMS")
                # Weighted histograms onnly for data at the moment
                h_invmass_init = lfile.Get(histname + suffix)
                h_data_init_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass_init,
                                                                    self.rebins[bin_mult_int][ipt],
                                                                    -1)
                h_data_init_rebin = TH1F()
                h_data_init_rebin_.Copy(h_data_init_rebin)
                h_data_init_rebin.SetTitle("%.1f < #it{p}_{T} < %.1f (prob > %.2f)" \
                                           % (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt], \
                                              self.lpt_probcutfin[bin_id]))
                h_data_init_rebin.GetXaxis().SetTitle("#it{M}_{inv} (GeV/#it{c}^{2})")
                h_data_init_rebin.GetYaxis().SetTitle("Entries/(%.0f MeV/#it{c}^{2})" \
                                                      % (h_data_init_rebin.GetBinWidth(1) * 1000))
                h_data_init_rebin.GetYaxis().SetTitleOffset(1.1)

                mass_fitter_data_init.append(AliHFInvMassFitter(h_data_init_rebin,
                                                                self.p_massmin[ipt],
                                                                self.p_massmax[ipt],
                                                                self.bkg_fmap[self.p_bkgfunc[ipt]],
                                                                self.sig_fmap[self.p_sgnfunc[ipt]]))

                # Force to go on with final fit although there is no estimated signal
                mass_fitter_data_init[ipt].SetCheckSignalCountsAfterFirstFit(False)
                if self.p_dolike:
                    mass_fitter_data_init[ipt].SetUseLikelihoodFit()
                mass_fitter_data_init[ipt].SetInitialGaussianMean(mean_for_data)
                mass_fitter_data_init[ipt].SetInitialGaussianSigma(sigma_for_data)
                mass_fitter_data_init[ipt].SetNSigma4SideBands(self.p_exclude_nsigma_sideband)
                # Second peak?
                if self.p_includesecpeaks[bin_mult_int][ipt]:
                    secpeakwidth = self.p_widthsecpeak * sigma_for_data
                    mass_fitter_data_init[ipt].IncludeSecondGausPeak(self.p_masssecpeak, \
                                               self.p_fix_masssecpeaks[bin_mult_int][ipt], \
                                               secpeakwidth, \
                                               self.p_fix_widthsecpeak)
                success = mass_fitter_data_init[ipt].MassFitter(False)
                fit_status[imult][ipt]["init_data"]["success"] = False
                fit_status[imult][ipt]["init_data"]["sigma"] = -1.
                if success == 1:
                    if ipt not in have_summary_pt_bins:
                        means_init_data_histos.SetBinContent(ipt + 1, \
                                mass_fitter_data_init[ipt].GetMean())
                        means_init_data_histos.SetBinError(ipt + 1, \
                                mass_fitter_data_init[ipt].GetMeanUncertainty())
                        sigmas_init_data_histos.SetBinContent(ipt + 1, \
                                mass_fitter_data_init[ipt].GetSigma())
                        sigmas_init_data_histos.SetBinError(ipt + 1, \
                                mass_fitter_data_init[ipt].GetSigmaUncertainty())

                    sigmafit = mass_fitter_data_init[ipt].GetSigma()
                    if minperc * sigma_for_data < sigmafit < maxperc * sigma_for_data:
                        means_sigmas_init.insert(0, (1, mass_fitter_data_init[ipt].GetMean(),
                                                     mass_fitter_data_init[ipt].GetSigma()))
                        fit_status[imult][ipt]["init_data"]["success"] = True
                        fit_status[imult][ipt]["init_data"]["sigma"] = \
                                mass_fitter_data_init[ipt].GetSigma()
                        if self.init_fits_from[ipt] == "data":
                            mean_case_user = mass_fitter_data_init[ipt].GetMean()
                            sigma_case_user = mass_fitter_data_init[ipt].GetSigma()
                            user_init_success = True

                canvas = TCanvas("fit_canvas_data_init", suffix_write, 700, 700)
                mass_fitter_data_init[ipt].DrawHere(canvas, self.p_nsigma_signal)
                canvas.SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                  "fittedplot_integrated", "eps",
                                                  None, suffix_write))
                canvas.Close()
                canvas_init_data.cd(ipt+1)
                mass_fitter_data_init[ipt].DrawHere(gPad, self.p_nsigma_signal)

                # Initialize mean and sigma with user seeds.
                if self.p_use_user_gauss_sigma[ipt] is True:
                    mean_for_data = self.p_masspeak
                    sigma_for_data = self.p_sigmaarray[ipt]
                    means_sigmas_init.insert(0, (3, mean_for_data, sigma_for_data))

                # Remember that we have filled this pT bin
                have_summary_pt_bins.append(ipt)
                ######################
                # END initialize fit #
                ######################

                # Collect all possible fit cases
                fit_cases = []
                for fix in [False, True]:
                    for ms in means_sigmas_init:
                        fit_cases.append((ms[0], ms[1], ms[2], fix))
                if user_init_success and self.p_use_user_gauss_sigma[ipt] is False:
                    fit_cases.insert(0, (0, mean_case_user, sigma_case_user,
                                         self.p_fixingaussigma[ipt]))
                elif self.p_use_user_gauss_sigma[ipt] is True:
                    fit_cases.insert(0, (0, self.p_masspeak, self.p_sigmaarray[ipt], \
                                         self.p_fixingaussigma[ipt]))
                else:
                    self.logger.error("Cannot initialise fit with what is requested by the " \
                                      "user... Try fallback options")

                # Now comes the actual fit
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                suffix_write = "%s%d_%d_%s_%.2f_%.2f" % \
                               (self.v_var_binning, self.lpt_finbinmin[ipt],
                                self.lpt_finbinmax[ipt],
                                self.v_var2_binning, self.lvar2_binmin[imult],
                                self.lvar2_binmax[imult])
                histname = "hmass"
                if self.apply_weights is True:
                    histname = "h_invmass_weight"
                    self.logger.info("*********** I AM USING WEIGHTED HISTOGRAMS")
                # Weighted histograms onnly for data at the moment
                h_invmass = lfile.Get(histname + suffix)
                h_invmass_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass,
                                                                  self.rebins[imult][ipt], -1)
                h_invmass_rebin = TH1F()
                h_invmass_rebin_.Copy(h_invmass_rebin)
                h_invmass_rebin.SetTitle("%.1f < #it{p}_{T} < %.1f (prob > %.2f)" \
                                         % (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt], \
                                            self.lpt_probcutfin[bin_id]))
                h_invmass_rebin.GetXaxis().SetTitle("#it{M}_{inv} (GeV/#it{c}^{2})")
                h_invmass_rebin.GetYaxis().SetTitle("Entries/(%.0f MeV/#it{c}^{2})" \
                                                    % (h_invmass_rebin.GetBinWidth(1) * 1000))
                h_invmass_rebin.GetYaxis().SetTitleOffset(1.1)

                h_invmass_mc = lfile_mc.Get("hmass" + suffix)
                h_invmass_mc_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass_mc,
                                                                     self.rebins[imult][ipt], -1)
                h_invmass_mc_rebin = TH1F()
                h_invmass_mc_rebin_.Copy(h_invmass_mc_rebin)
                success = 0

                fit_status[imult][ipt]["data"] = {}

                # First try not fixing sigma for all cases (mean always floating)
                user_case_success = True
                for case, mean, sigma, fix in fit_cases:

                    ifit = ifit + 1
                    mass_fitter.append(AliHFInvMassFitter(h_invmass_rebin, self.p_massmin[ipt],
                                                          self.p_massmax[ipt],
                                                          self.bkg_fmap[self.p_bkgfunc[ipt]],
                                                          self.sig_fmap[self.p_sgnfunc[ipt]]))

                    # Force to go on with final fit although there is no estimated signal
                    mass_fitter[ifit].SetCheckSignalCountsAfterFirstFit(False)
                    if self.p_dolike:
                        mass_fitter[ifit].SetUseLikelihoodFit()
                    # At this point *_for_data is either
                    # -> the seed value extracted from integrated data pre-fit if successful
                    # -> the seed value extracted from integrated MC pre-fit if successful
                    #    and data pre-fit failed
                    # -> the seed value set by the user in the database if both
                    #    data and MC pre-fit fail
                    mass_fitter[ifit].SetInitialGaussianMean(mean)
                    mass_fitter[ifit].SetInitialGaussianSigma(sigma)
                    #if self.p_fixedmean:
                    #    mass_fitter[ifit].SetFixGaussianMean(mean_for_data)
                    if fix:
                        mass_fitter[ifit].SetFixGaussianSigma(sigma)

                    mass_fitter[ifit].SetNSigma4SideBands(self.p_exclude_nsigma_sideband)

                    if self.include_reflection:
                        h_invmass_refl = AliVertexingHFUtils.AdaptTemplateRangeAndBinning(
                            lfile_mc.Get("hmass_refl" + suffix), h_invmass_rebin,
                            self.p_massmin[ipt], self.p_massmax[ipt])

                        if h_invmass_refl.Integral() > 0.:
                            mass_fitter[ifit].SetTemplateReflections(h_invmass_refl, "1gaus",
                                                                     self.p_massmin[ipt],
                                                                     self.p_massmax[ipt])
                            r_over_s = h_invmass_mc_rebin.Integral(
                                h_invmass_mc_rebin.FindBin(self.p_massmin[ipt]),
                                h_invmass_mc_rebin.FindBin(self.p_massmax[ipt]))
                            if r_over_s > 0.:
                                r_over_s = h_invmass_refl.Integral(
                                    h_invmass_refl.FindBin(self.p_massmin[ipt]),
                                    h_invmass_refl.FindBin(self.p_massmax[ipt])) / r_over_s
                                mass_fitter[ifit].SetFixReflOverS(r_over_s)
                        else:
                            self.logger.warning("Reflection requested but template empty")
                    if self.p_includesecpeaks[imult][ipt]:
                        secpeakwidth = self.p_widthsecpeak * sigma
                        mass_fitter[ifit].IncludeSecondGausPeak(self.p_masssecpeak,
                                                                self.p_fix_masssecpeaks[imult][ipt],
                                                                secpeakwidth,
                                                                self.p_fix_widthsecpeak)
                    fit_status[imult][ipt]["data"]["fix"] = fix
                    fit_status[imult][ipt]["data"]["case"] = case
                    success = mass_fitter[ifit].MassFitter(False)
                    if success == 1:
                        sigma_final = mass_fitter[ifit].GetSigma()
                        if minperc * sigma < sigma_final < maxperc * sigma:
                            break
                        self.logger.warning("Free fit succesful, but bad sigma. Skipped!")
                    if case == 0:
                        user_case_success = False

                fit_status[imult][ipt]["data"]["success"] = success

                pinfos = TPaveText(0.12, 0.7, 0.47, 0.89, "NDC")
                pinfos.SetBorderSize(0)
                pinfos.SetFillStyle(0)
                pinfos.SetTextAlign(11)
                pinfos.SetTextSize(0.03)
                add_draw_objects.append(pinfos)
                if not user_init_success:
                    text = pinfos.AddText(f"USER INIT CASE FAILED ({self.init_fits_from[ipt]}, " \
                            f"sigma fixed {self.p_fixingaussigma[ipt]})")
                    text.SetTextColor(kRed)

                if not user_case_success:
                    text = pinfos.AddText(f"FIT WITH USER INIT CASE FAILED " \
                            f"({self.init_fits_from[ipt]}, " \
                            f"sigma fixed {self.p_fixingaussigma[ipt]})")
                    text.SetTextColor(kRed)
                if not success:
                    text = pinfos.AddText("FIT FAILED")
                    text.SetTextColor(kRed)

                canvas = TCanvas("fit_canvas", suffix, 700, 700)
                mass_fitter[ifit].DrawHere(canvas, self.p_nsigma_signal)
                canvas.cd()
                pinfos.Draw()
                if self.apply_weights is False:
                    canvas.SaveAs(self.make_file_path(self.d_resultsallpdata, "fittedplot", "eps",
                                                      None, suffix_write))
                else:
                    canvas.SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                      "fittedplotweights",
                                                      "eps", None, suffix_write))
                canvas.Close()
                canvas_data[imult].cd(ipt+1)
                mass_fitter[ifit].DrawHere(gPad, self.p_nsigma_signal)
                pinfos.Draw()

                if success == 1:
                    # In case of success == 2, no signal was found, in case of 0, fit failed
                    rawYield = mass_fitter[ifit].GetRawYield()
                    rawYieldErr = mass_fitter[ifit].GetRawYieldError()
                    yieldshistos[imult].SetBinContent(ipt + 1, rawYield)
                    yieldshistos[imult].SetBinError(ipt + 1, rawYieldErr)

                    mean_fit = mass_fitter[ifit].GetMean()
                    mean_min = min(mean_fit, mean_min)
                    mean_max = max(mean_fit, mean_max)

                    means_histos[imult].SetBinContent(ipt + 1, mean_fit)
                    means_histos[imult].SetBinError(ipt + 1, mass_fitter[ifit].GetMeanUncertainty())

                    sigma_fit = mass_fitter[ifit].GetSigma()
                    sigma_min = min(sigma_fit, sigma_min)
                    sigma_max = max(sigma_fit, sigma_max)
                    fit_status[imult][ipt]["data"]["sigma"] = sigma_fit

                    sigmas_histos[imult].SetBinContent(ipt + 1, sigma_fit)
                    sigmas_histos[imult].SetBinError(ipt + 1, \
                                                     mass_fitter[ifit].GetSigmaUncertainty())

                    # Residual plot
                    c_res = TCanvas('cRes', 'The Fit Canvas', 800, 800)
                    c_res.cd()
                    h_pulls = h_invmass_rebin.Clone(f"{h_invmass_rebin.GetName()}_pull")
                    h_residual_trend = \
                            h_invmass_rebin.Clone(f"{h_invmass_rebin.GetName()}_residual_trend")
                    h_pulls_trend = \
                            h_invmass_rebin.Clone(f"{h_invmass_rebin.GetName()}_pulls_trend")
                    _ = mass_fitter[ifit].GetOverBackgroundResidualsAndPulls( \
                            h_pulls, h_residual_trend, h_pulls_trend, self.p_massmin[ipt],
                            self.p_massmax[ipt])

                    h_residual_trend.Draw()
                    c_res.SaveAs(self.make_file_path(self.d_resultsallpdata, "residual", "eps",
                                                     None, suffix_write))
                    c_res.Close()

                else:
                    self.logger.error("Fit failed for suffix %s", suffix_write)

                # Write fitters to file
                fit_root_dir = fileout.mkdir(suffix)
                fit_root_dir.WriteObject(mass_fitter_mc_init[ipt], "gaus_mc_init")
                fit_root_dir.WriteObject(mass_fitter_data_init[ipt], "fitter_data_init")
                fit_root_dir.WriteObject(mass_fitter[ifit], "fitter")

            #####################
            # Back in mult loop #
            #####################

            suffix2 = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin[imult], \
                                        self.lvar2_binmax[imult])
            if imult == 0:
                canvas_init_mc.SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                          "canvas_InitMC",
                                                          "eps", None, suffix2))
                canvas_init_mc.Close()
                canvas_init_data.SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                            "canvas_InitData",
                                                            "eps", None, suffix2))
                canvas_init_data.Close()
            canvas_data[imult].SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                          "canvas_FinalData",
                                                          "eps", None, suffix2))
            #canvas_data[imult].Close()
            fileout.cd()
            yieldshistos[imult].Write()
            means_histos[imult].Write()
            sigmas_histos[imult].Write()

            del mass_fitter_mc_init[:]
            del mass_fitter_data_init[:]

        del mass_fitter[:]

        # Write the fit status dict
        dump_yaml_from_dict(fit_status, self.make_file_path(self.d_resultsallpdata, "fit_status",
                                                            "yaml"))

        # Plot some summary historgrams
        leg_strings = [f"{self.lvar2_binmin[imult]} #leq {self.p_latexbin2var} < " \
                       f"{self.lvar2_binmax[imult]}" for imult in range(self.p_nbin2)]
        save_name = self.make_file_path(self.d_resultsallpdata, "Yields", "eps", None,
                                        [self.case, self.typean])
        # Yields summary plot
        plot_histograms(yieldshistos, True, True, leg_strings, "uncorrected yields",
                        "#it{p}_{T} (GeV/#it{c})",
                        f"Uncorrected yields {self.p_latexnmeson} {self.typean}", "mult. / int.",
                        save_name)
        save_name = self.make_file_path(self.d_resultsallpdata, "Means", "eps", None,
                                        [self.case, self.typean])
        # Means summary plot
        plot_histograms(means_histos, False, True, leg_strings, "Means",
                        "#it{p}_{T} (GeV/#it{c})",
                        "#mu_{fit} " + f"{self.p_latexnmeson} {self.typean}", "mult. / int.",
                        save_name)
        save_name = self.make_file_path(self.d_resultsallpdata, "Sigmas", "eps", None,
                                        [self.case, self.typean])
        #Sigmas summary plot
        plot_histograms(sigmas_histos, False, True, leg_strings, "Sigmas",
                        "#it{p}_{T} (GeV/#it{c})",
                        "#sigma_{fit} " + f"{self.p_latexnmeson} {self.typean}", "mult. / int.",
                        save_name)

        # Plot the initialized means and sigma for MC and data
        save_name = self.make_file_path(self.d_resultsallpdata, "Means_mult_int", "eps", None,
                                        [self.case, self.typean])
        plot_histograms([means_init_mc_histos, means_init_data_histos], False, False,
                        ["MC", "data"], "Means of int. mult.", "#it{p}_{T} (GeV/#it{c})",
                        "#mu_{fit} " + f"{self.p_latexnmeson} {self.typean}", "", save_name)

        save_name = self.make_file_path(self.d_resultsallpdata, "Sigmas_mult_int", "eps", None,
                                        [self.case, self.typean])
        plot_histograms([sigmas_init_mc_histos, sigmas_init_data_histos], False, False,
                        ["MC", "data"], "Sigmas of int. mult.", "#it{p}_{T} (GeV/#it{c})",
                        "#sigma_{fit} " + f"{self.p_latexnmeson} {self.typean}", "", save_name)

        fileout.Close()
        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-branches
    def yield_syst(self):
        if self.mt_syst_dict is None:
            self.logger.warning("Could not find parameters for doing systemtics. Skip...")
            return
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        # Load configurable multi trial variations
        rebin = self.mt_syst_dict.get("rebin", None)
        fit_ranges_low = self.mt_syst_dict.get("massmin", None)
        fit_ranges_up = self.mt_syst_dict.get("massmax", None)
        bincount_sigma = self.mt_syst_dict.get("bincount_sigma", None)
        bkg_funcs = self.mt_syst_dict.get("bkg_funcs", None)
        # Sigma of MT can either be initialized from MC ("mc"), data ("data")
        # or central ("central" ==> default)
        initialize_sigma_from = self.mt_syst_dict.get("init_sigma_from", "central")
        if not isinstance(initialize_sigma_from, list):
            initialize_sigma_from = [initialize_sigma_from] * self.p_nptbins
        if not isinstance(initialize_sigma_from[0], list):
            initialize_sigma_from = [initialize_sigma_from] * self.p_nbin2

        if not bkg_funcs:
            self.logger.error("You need to choose at least one background function for " \
                              "the multi trial")
            return

        from ROOT import AliHFInvMassMultiTrialFit, AliVertexingHFUtils

        if not os.path.exists(self.d_mt_results_path):
            os.makedirs(self.d_mt_results_path)

        lfile = TFile.Open(self.n_filemass, "READ")
        lfile_mc = TFile.Open(self.n_filemass_mc, "READ")

        file_fits_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        file_fits = TFile(file_fits_name, "READ")

        # Collect arguments for parallelized multi trial run
        for imult in range(self.p_nbin2):
            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]

                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])

                suffix_write = "%s%d_%d_%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])


                # That we need to obtain the yield of the central fit and the initial mean
                # set to the multi trial
                mass_fitter_nominal = file_fits.GetDirectory(suffix).Get("fitter")
                histname = "hmass"
                if self.apply_weights is True:
                    histname = "h_invmass_weight"
                    self.logger.info("*********** I AM USING WEIGHTED HISTOGRAMS")
                # Weighted histograms onnly for data at the moment
                h_invmass_ = lfile.Get(histname + suffix)

                h_invmass = TH1D()
                h_invmass_.Copy(h_invmass)

                # Next we need the sigma to be used for the multi trial
                sigma_init = mass_fitter_nominal.GetSigma()

                if initialize_sigma_from[imult][ipt] == "user":
                    sigma_init = self.p_sigmaarray[ipt]
                elif initialize_sigma_from[imult][ipt] == "mc":
                    fit = file_fits.GetDirectory(suffix).Get("gaus_mc_init")
                    sigma_init = fit.GetParameter(2)
                elif initialize_sigma_from[imult][ipt] == "data":
                    fit = file_fits.GetDirectory(suffix).Get("fitter_data_init")
                    sigma_init = fit.GetSigma()

                # Initialize the multi trial
                multi_trial = AliHFInvMassMultiTrialFit()

                multi_trial.SetSuffixForHistoNames("")
                multi_trial.SetDrawIndividualFits(False)
                # This is always the mean of the central fit
                multi_trial.SetMass(mass_fitter_nominal.GetMean())
                multi_trial.SetSigmaGaussMC(sigma_init)

                # First, disable all
                multi_trial.SetUseExpoBackground(False)
                multi_trial.SetUseLinBackground(False)
                multi_trial.SetUsePol2Background(False)
                multi_trial.SetUsePol3Background(False)
                multi_trial.SetUsePol4Background(False)
                multi_trial.SetUsePol5Background(False)
                multi_trial.SetUsePowerLawBackground(False)
                multi_trial.SetUsePowerLawTimesExpoBackground(False)


                for bkg in bkg_funcs:
                    if bkg == "kExpo":
                        multi_trial.SetUseExpoBackground(True)
                        continue
                    if bkg == "kLin":
                        multi_trial.SetUseLinBackground(True)
                        continue
                    if bkg == "Pol2":
                        multi_trial.SetUsePol2Background(True)
                        continue
                    if bkg == "Pol3":
                        multi_trial.SetUsePol3Background(True)
                        continue
                    if bkg == "Pol4":
                        multi_trial.SetUsePol4Background(True)
                        continue
                    if bkg == "Pol5":
                        multi_trial.SetUsePol5Background(True)
                        continue

                    self.logger.fatal("Unknown background %s for multi trial", bkg)

                if rebin:
                    rebin_steps = [self.rebins[imult][ipt] + rel_rb \
                            if self.rebins[imult][ipt] + rel_rb > 0 else 1 for rel_rb in rebin]
                    # To only have unique values and we don't care about the order we can just do
                    rebin_steps = list(set(rebin_steps))
                    rebin_steps = array("i", rebin_steps)
                    multi_trial.ConfigureRebinSteps(len(rebin_steps), rebin_steps)
                if fit_ranges_low:
                    low_lim_steps = array("d", fit_ranges_low)
                    multi_trial.ConfigureLowLimFitSteps(len(fit_ranges_low),
                                                        low_lim_steps)
                if fit_ranges_up:
                    up_lim_steps = array("d", fit_ranges_up)
                    multi_trial.ConfigureUpLimFitSteps(len(fit_ranges_up), up_lim_steps)

                if bincount_sigma:
                    multi_trial.ConfigurenSigmaBinCSteps(len(bincount_sigma),
                                                         array("d", bincount_sigma))

                # Prepare for reflections if requested
                h_invmass_mc = None
                h_invmass_mc_refl = None
                if self.include_reflection:
                    h_invmass_mc_ = lfile_mc.Get("hmass" + suffix)
                    h_invmass_mc_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass_mc_,
                                                                         self.rebins[imult][ipt],
                                                                         -1)
                    h_invmass_mc = TH1F()
                    h_invmass_mc_rebin_.Copy(h_invmass_mc)
                    h_invmass_mc_refl_ = lfile_mc.Get("hmass_refl" + suffix)
                    h_invmass_mc_refl = AliVertexingHFUtils.AdaptTemplateRangeAndBinning(
                        h_invmass_mc_refl_, h_invmass,
                        self.p_massmin[ipt], self.p_massmax[ipt])
                    if h_invmass_mc_refl.Integral() > 0.:
                        multi_trial.SetTemplatesForReflections(h_invmass_mc_refl, h_invmass_mc_)
                        r_over_s = h_invmass_mc.Integral(
                            h_invmass_mc.FindBin(self.p_massmin[ipt]),
                            h_invmass_mc.FindBin(self.p_massmax[ipt]))
                        if r_over_s > 0.:
                            r_over_s = h_invmass_mc_refl.Integral(
                                h_invmass_mc_refl.FindBin(self.p_massmin[ipt]),
                                h_invmass_mc_refl.FindBin(self.p_massmax[ipt])) / r_over_s
                            multi_trial.SetFixRefoS(r_over_s)
                    else:
                        self.logger.warning("Reflection requested but template empty")

                if self.p_includesecpeaks[imult][ipt]:
                    # To init the second peak we need to know what the user has chosen to be the
                    # initialisation for the central fit since from that the sec. peak width was
                    # derived
                    sigma_sec = None
                    if self.p_use_user_gauss_sigma[ipt] is True:
                        sigma_sec = self.p_sigmaarray[ipt] * self.p_widthsecpeak
                    elif self.init_fits_from[ipt] == "data":
                        fit = file_fits.GetDirectory(suffix).Get("fitter_data_init")
                        sigma_sec = fit.GetSigma() * self.p_widthsecpeak
                    else:
                        fit = file_fits.GetDirectory(suffix).Get("gaus_mc_init")
                        sigma_sec = fit.GetParameter(2) * self.p_widthsecpeak
                    #p_widthsecpeak to be fixed
                    multi_trial.IncludeSecondGausPeak(self.p_masssecpeak,
                                                      self.p_fix_masssecpeaks[imult][ipt],
                                                      sigma_sec,
                                                      self.p_fix_widthsecpeak)

                # Just make sure it's kept until the workflow is done
                self.root_objects.append(multi_trial)
                mt_filename = self.make_file_path(self.d_mt_results_path, "multi_trial",
                                                  "root", None, suffix_write)

                if multi_trial.DoMultiTrials(h_invmass):
                    multi_trial.SaveToRoot(mt_filename)

        self.plot_multi_trial()

        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    def plot_multi_trial(self):

        if not os.path.exists(self.d_mt_results_path):
            self.logger.error("Could not find multi trial results directory %s. Skip...",
                              self.d_mt_results_path)
            return

        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        gROOT.LoadMacro("PlotMultiTrial.C")
        from ROOT import PlotMultiTrial

        func_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                            None, [self.case, self.typean])
        func_file = TFile.Open(func_filename, "READ")

        # Some derived values from multi trial
        mt_derived_filename = self.make_file_path(self.d_mt_results_path, "multi_trial_summary",
                                                  "root", None, [self.case, self.typean])
        mt_derived_file = TFile.Open(mt_derived_filename, "RECREATE")

        max_chisquare_ndf = self.mt_syst_dict.get("max_chisquare_ndf", 2.)
        # As this plotting function is called right after the actual multi trial,
        # the background function list could also just be assumed to be there...
        bkg_funcs = self.mt_syst_dict.get("bkg_funcs", None)
        consider_free_sigma = self.mt_syst_dict.get("consider_free_sigma",
                                                    [False for i in range(self.p_nptbins)])

        for imult in range(self.p_nbin2):
            h_nominal = TH1F("h_nominal_sum", "", self.p_nptbins, array("d", self.ptranges))
            h_fit_all = TH1F("h_yield_sum", "", self.p_nptbins, array("d", self.ptranges))
            h_fit_all = TH1F("h_yield_sum", "", self.p_nptbins, array("d", self.ptranges))
            h_bincount_all0 = TH1F("h_bincount_sum0", "", self.p_nptbins, array("d", self.ptranges))
            h_bincount_all1 = TH1F("h_bincount_sum1", "", self.p_nptbins, array("d", self.ptranges))

            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]

                suffix_write = "%s%d_%d_%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])

                mt_filename = self.make_file_path(self.d_mt_results_path, "multi_trial",
                                                  "root", None, suffix_write)
                if not os.path.exists(mt_filename):
                    self.logger.warning("Multi trial file %s does not exist. MT might have " \
                                        "failed before. Skipping...", mt_filename)
                    continue
                # Get the nominal fit
                load_dir = func_file.GetDirectory(suffix)
                mass_fitter = load_dir.Get("fitter")

                rawYield = mass_fitter.GetRawYield()
                rawYieldError = mass_fitter.GetRawYieldError()
                mean_fit = mass_fitter.GetMean()
                sigma_fit = mass_fitter.GetSigma()
                chisquare_fit = mass_fitter.GetChiSquare()

                self.logger.info("Process file %s", mt_filename)
                title = f"{self.lpt_finbinmin[ipt]} GeV/c < {self.v_var_binning} < " \
                        f"{self.lpt_finbinmax[ipt]} GeV/c, {self.lvar2_binmin[imult]} < " \
                        f"{self.v_var2_binning} < {self.lvar2_binmax[imult]}"
                derived_dir = mt_derived_file.mkdir(suffix)

                used_bkgs = array("b", ["kExpo" in bkg_funcs,
                                        "kLin" in bkg_funcs,
                                        "Pol2" in bkg_funcs,
                                        "Pol3" in bkg_funcs,
                                        "Pol4" in bkg_funcs,
                                        "Pol5" in bkg_funcs])

                n_bins_bincount = len(self.mt_syst_dict.get("bincount_sigma", []))
                PlotMultiTrial(mt_filename, rawYield, mean_fit, sigma_fit, chisquare_fit,
                               max_chisquare_ndf, used_bkgs, n_bins_bincount,
                               consider_free_sigma[ipt], self.d_mt_results_path, suffix,
                               title, derived_dir)

                h_mt_fit = derived_dir.Get("h_mt_fit")
                h_mt_bc0 = derived_dir.Get("h_mt_bc0")
                h_mt_bc1 = derived_dir.Get("h_mt_bc1")

                h_nominal.SetBinContent(ipt + 1, rawYield)
                h_nominal.SetBinError(ipt + 1, rawYieldError)
                h_fit_all.SetBinContent(ipt + 1, h_mt_fit.GetMean())
                h_fit_all.SetBinError(ipt + 1, h_mt_fit.GetRMS())
                h_bincount_all0.SetBinContent(ipt + 1, h_mt_bc0.GetMean())
                h_bincount_all0.SetBinError(ipt + 1, h_mt_bc0.GetRMS())
                h_bincount_all1.SetBinContent(ipt + 1, h_mt_bc1.GetMean())
                h_bincount_all1.SetBinError(ipt + 1, h_mt_bc1.GetRMS())

            filename_mt_summary = self.make_file_path(self.d_mt_results_path, "multi_trial_summary",
                                                      "eps", None, [imult])

            plot_histograms([h_nominal, h_fit_all, h_bincount_all0, h_bincount_all1], False, True,
                            ["central fit", "mean MT fit", "mean MT BC (bkg. fit)",
                             "mean MT BC (bkg. refit)"],
                            f"{self.lvar2_binmin[imult]} < {self.v_var2_binning} " \
                            f"< {self.lvar2_binmax[imult]}", "#it{p}_{T} (GeV/c)", "yield",
                            "MT / central", filename_mt_summary, colors=[kBlack, kBlue, kGreen+2,
                                                                         kOrange+5])
            column_names = ["central fit", "mean MT fit", "mean MT BC (bkg. fit)",
                            "mean MT BC (bkg. refit)",
                            "rel.unc. central", "rel. (central -  MT fit)",
                            "rel. (central - MT BC (bkg. fit))",
                            "rel. (central -  MT BC (bkg. refit))"]
            row_names = [f"{self.lpt_finbinmin[ipt]} GeV/c < {self.v_var_binning} < " \
                         f"{self.lpt_finbinmax[ipt]} GeV/c" for ipt in range(self.p_nptbins)]
            rows = []
            for b in range(h_nominal.GetNbinsX()):
                yield_nominal = h_nominal.GetBinContent(b + 1)
                yield_nominal_err = h_nominal.GetBinError(b + 1)
                yield_mt_fit = h_fit_all.GetBinContent(b + 1)
                yield_mt_bc0 = h_bincount_all0.GetBinContent(b + 1)
                yield_mt_bc1 = h_bincount_all1.GetBinContent(b + 1)
                rel_centr_fit = yield_nominal_err / yield_nominal if yield_nominal > 0. \
                        else 0.
                rel_mt_fit = (yield_nominal - yield_mt_fit) / yield_nominal if yield_nominal > 0. \
                        else 0.
                rel_mt_bc0 = (yield_nominal - yield_mt_bc0) / yield_nominal if yield_nominal > 0. \
                        else 0.
                rel_mt_bc1 = (yield_nominal - yield_mt_bc1) / yield_nominal if yield_nominal > 0. \
                        else 0.
                rows.append([f"{yield_nominal:.2f} ({yield_nominal_err:.2f})",
                             f"{yield_mt_fit:.2f} ({h_fit_all.GetBinError(b + 1):.2f})",
                             f"{yield_mt_bc0:.2f} ({h_bincount_all0.GetBinError(b + 1):.2f})",
                             f"{yield_mt_bc1:.2f} ({h_bincount_all1.GetBinError(b + 1):.2f})",
                             f"{rel_centr_fit:.3f}",
                             f"{rel_mt_fit:.3f}",
                             f"{rel_mt_bc0:.3f}",
                             f"{rel_mt_bc1:.3f}"])

            caption = f"{self.lvar2_binmin[imult]} < {self.v_var2_binning} < " \
                      f"{self.lvar2_binmax[imult]}"
            filename_mt_summary = self.make_file_path(self.d_mt_results_path, "multi_trial_summary",
                                                      "tex", None, [imult])
            make_latex_table(column_names, row_names, rows, caption, filename_mt_summary)

        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    def efficiency(self):
        self.loadstyle()

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

        for imult in range(self.p_nbin2):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning_gen, \
                                            self.lvar2_binmin[imult], \
                                            self.lvar2_binmax[imult])
            h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
            h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
            h_sel_pr.SetLineColor(imult+1)
            h_sel_pr.Draw("same")
            fileouteff.cd()
            h_sel_pr.SetName("eff_mult%d" % imult)
            h_sel_pr.Write()
            legeffstring = "%.1f #leq %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legeff.AddEntry(h_sel_pr, legeffstring, "LEP")
            h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s %s (1/GeV)" \
                    % (self.p_latexnmeson, self.typean))
            h_sel_pr.SetMinimum(0.)
            h_sel_pr.SetMaximum(1.5)
        legeff.Draw()
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

        for imult in range(self.p_nbin2):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning_gen, \
                                            self.lvar2_binmin[imult], \
                                            self.lvar2_binmax[imult])
            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
            h_sel_fd.SetLineColor(imult+1)
            h_sel_fd.Draw("same")
            fileouteff.cd()
            h_sel_fd.SetName("eff_fd_mult%d" % imult)
            h_sel_fd.Write()
            legeffFDstring = "%.1f #leq %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legeffFD.AddEntry(h_sel_fd, legeffFDstring, "LEP")
            h_sel_fd.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            h_sel_fd.GetYaxis().SetTitle("Acc x efficiency feed-down %s %s (1/GeV)" \
                    % (self.p_latexnmeson, self.typean))
            h_sel_fd.SetMinimum(0.)
            h_sel_fd.SetMaximum(1.5)
        legeffFD.Draw()
        cEffFD.SaveAs("%s/EffFD%s%s.eps" % (self.d_resultsallpmc,
                                            self.case, self.typean))
    def feeddown(self):
        # TODO: Propagate uncertainties.
        self.loadstyle()
        file_resp = TFile.Open(self.n_fileff)
        file_eff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc, \
                              self.case, self.typean))
        file_out = TFile.Open("%s/feeddown%s%s.root" % \
                              (self.d_resultsallpmc, self.case, self.typean), "recreate")

        # Get feed-down detector response.
        his_resp_fd = file_resp.Get("his_resp_jet_fd")
        arr_resp_fd = hist2array(his_resp_fd).T
        bins_final = np.array([his_resp_fd.GetYaxis().GetBinLowEdge(i) for i in \
            range(1, his_resp_fd.GetYaxis().GetNbins() + 2)])
        # TODO: Normalise so that projection on the pt_gen = 1.
        can_resp_fd = TCanvas("can_resp_fd", "Feed-down detector response", 800, 800)
        his_resp_fd.Draw("colz")
        can_resp_fd.SetLogz()
        can_resp_fd.SetLeftMargin(0.15)
        can_resp_fd.SetRightMargin(0.15)
        can_resp_fd.SaveAs("%s/ResponseFD%s%s.eps" % (self.d_resultsallpmc, \
                            self.case, self.typean))

        # Get the number of generated jets for feed-down normalisation.
        his_njets = file_resp.Get("his_njets_gen")
        n_jets_gen = his_njets.GetBinContent(1)

        # Get simulated pt_cand vs. pt_jet of non-prompt jets.
        his_sim_fd = file_resp.Get("his_ptc_ptjet_fd")
        his_sim_fd.Scale(1./n_jets_gen) # Normalise by the total number of selected jets.
        arr_sim_fd = hist2array(his_sim_fd).T
        can_sim_fd = TCanvas("can_sim_fd", \
                        "Simulated pt cand vs. pt jet of non-prompt jets", 800, 800)
        his_sim_fd.Draw("colz")
        can_sim_fd.SetLogz()
        can_sim_fd.SetLeftMargin(0.15)
        can_sim_fd.SetRightMargin(0.15)
        can_sim_fd.SaveAs("%s/GenFD%s%s.eps" % (self.d_resultsallpmc, \
                            self.case, self.typean))

        for imult in range(self.p_nbin2):
            # Get efficiencies.
            his_eff_pr = file_eff.Get("eff_mult%d" % imult)
            his_eff_fd = file_eff.Get("eff_fd_mult%d" % imult)
            his_eff_pr = his_eff_pr.Clone("his_eff_pr1_%d" % imult)
            his_eff_fd = his_eff_fd.Clone("his_eff_fd1_%d" % imult)
            his_eff_pr.SetLineColor(2)
            his_eff_fd.SetLineColor(3)
            his_eff_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            his_eff_pr.GetYaxis().SetTitle("reconstruction efficiency %s %s" \
                    % (self.p_latexnmeson, self.typean))
            his_eff_pr.GetYaxis().SetRangeUser(0, 0.6)
            leg_eff = TLegend(.5, .15, .7, .35)
            leg_eff.SetBorderSize(0)
            leg_eff.SetFillColor(0)
            leg_eff.SetFillStyle(0)
            leg_eff.SetTextFont(42)
            leg_eff.SetTextSize(0.035)
            can_eff = TCanvas("can_eff%d" % imult, "Efficiency%d" % imult, 800, 600)
            his_eff_pr.Draw("same")
            his_eff_fd.Draw("same")
            legeffstring = "%.1f #leq %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            leg_eff.SetHeader(legeffstring)
            leg_eff.AddEntry(his_eff_pr, "prompt", "LEP")
            leg_eff.AddEntry(his_eff_fd, "non-prompt", "LEP")
            leg_eff.Draw()
            can_eff.SaveAs("%s/Efficiency%s%s%s.eps" % (self.d_resultsallpmc, \
                            self.case, self.typean, imult))
            # Get the ratio of efficiencies.
            his_eff_fd.Divide(his_eff_pr)
            arr_eff_ratio = hist2array(his_eff_fd)
            # Get the feed-down yield = response * simulated non-prompts * ratio of efficiencies.
            arr_sim_fd_eff_smeared = arr_resp_fd.dot(arr_sim_fd.dot(arr_eff_ratio))
            his_fd = TH1F("fd_mult%d" % imult, \
                          "Feed-down_mult%d;#it{p}_{T}^{jet ch.} (GeV/#it{c});"
                          "d#it{#sigma}^{jet ch.}/d#it{p}_{T} (arb. units)" % imult, \
                          len(bins_final) - 1, bins_final)
            array2hist(arr_sim_fd_eff_smeared, his_fd)
            his_fd.GetYaxis().SetTitleOffset(1.3)
            his_fd.GetYaxis().SetTitleFont(42)
            his_fd.GetYaxis().SetLabelFont(42)
            his_fd.GetXaxis().SetTitleFont(42)
            his_fd.GetXaxis().SetLabelFont(42)
            file_out.cd()
            his_fd.Write()
            can_fd = TCanvas("can_fd%d" % imult, "Feeddown spectrum", 800, 600)
            his_fd.Draw("same")
            can_fd.SetLogy()
            can_fd.SetLeftMargin(0.12)
            can_fd.SaveAs("%s/Feeddown%s%s%s.eps" % (self.d_resultsallpmc, \
                            self.case, self.typean, imult))

        # Get simulated pt_cand vs. pt_jet vs. z of non-prompt jets.
        his_sim_z_fd = file_resp.Get("his_ptc_ptjet_z_fd")
        his_sim_z_fd.Scale(1./n_jets_gen) # Normalise by the total number of selected jets.
        his_sim_z_fd_2d_gen = his_sim_z_fd.Project3D("yz")
        bins_z = get_bins(his_sim_z_fd_2d_gen.GetXaxis())
        bins_ptjet = get_bins(his_sim_z_fd_2d_gen.GetYaxis())
        his_sim_z_fd_2d_eff = TH2F("fd_z_ptjet_eff", "fd_z_ptjet_eff", \
            len(bins_z) - 1, bins_z, len(bins_ptjet) - 1, bins_ptjet) # final feed-down histogram
        his_sim_z_fd_2d_eff.Sumw2()
        # x axis = z, y axis = pt_jet
        # Scale with the ratio of efficiencies.
        # loop over pt_jet bins
        for i_ptjet in range(self.p_nbin2):
            bin_ptjet = i_ptjet + 1
            # Get efficiencies.
            his_eff_pr = file_eff.Get("eff_mult%d" % i_ptjet)
            his_eff_fd = file_eff.Get("eff_fd_mult%d" % i_ptjet)
            his_eff_pr = his_eff_pr.Clone("his_eff_pr2_%d" % i_ptjet)
            his_eff_fd = his_eff_fd.Clone("his_eff_fd2_%d" % i_ptjet)
            his_eff_fd.Divide(his_eff_pr)
            his_eff_ratio = his_eff_fd
            # loop over z bins
            for i_z in range(his_sim_z_fd_2d_eff.GetNbinsX()):
                bin_z = i_z + 1
                his_sim_z_fd.GetYaxis().SetRange(bin_ptjet, bin_ptjet)
                his_sim_z_fd.GetZaxis().SetRange(bin_z, bin_z)
                his_sim_ptc_fd = his_sim_z_fd.Project3D("x") # pt_cand
                his_sim_ptc_fd.Multiply(his_eff_ratio)
                his_sim_z_fd_2d_eff.SetBinContent(bin_z, bin_ptjet, his_sim_ptc_fd.Integral())
                his_sim_z_fd_2d_eff.SetBinError(bin_z, bin_ptjet, 0)

        # Smear (fold) the simulated distribution with the response matrix.
        resp_z = file_resp.Get("resp_z")
        his_sim_z_fd_2d_folded = resp_z.ApplyToTruth(his_sim_z_fd_2d_eff)
        # Alternative way of folding without RooUnfold
        #his_sim_z_fd_2d_folded = his_sim_z_fd_2d_eff.Clone("his_sim_z_fd_2d_folded")
        #his_sim_z_fd_2d_folded.Reset()
        #his_sim_z_fd_2d_folded = folding(his_sim_z_fd_2d_eff, resp_z, his_sim_z_fd_2d_folded)
        resp_z_proj = resp_z.Hresponse()
        can_resp_z = TCanvas("can_resp_z", "can_resp_z", 800, 600)
        resp_z_proj.Draw("colz")
        can_resp_z.SetLogz()
        can_resp_z.SaveAs("%s/Feeddown-z-response_%s%s.eps" \
                % (self.d_resultsallpmc, self.case, self.typean))

        for i_ptjet in range(self.p_nbin2):
            bin_ptjet = i_ptjet + 1
            can_ff_fd_fold = TCanvas("can_fd_z_all%d" % i_ptjet, "Feeddown FF all", 800, 600)
            his_ff_fd_gen = his_sim_z_fd_2d_gen.ProjectionX("ff_gen%d" \
                % i_ptjet, bin_ptjet, bin_ptjet)
            his_ff_fd_eff = his_sim_z_fd_2d_eff.ProjectionX("ff_eff%d" \
                % i_ptjet, bin_ptjet, bin_ptjet)
            his_ff_fd_fold = his_sim_z_fd_2d_folded.ProjectionX("ff_fold%d" \
                % i_ptjet, bin_ptjet, bin_ptjet)
            his_ff_fd_gen.GetYaxis().SetTitle("#frac{1}{#it{N}_{jet}} #frac{d#it{N}}{d#it{z}}")
            his_ff_fd_gen.GetYaxis().SetTitleOffset(1.6)
            his_ff_fd_gen.GetYaxis().SetTitleFont(42)
            his_ff_fd_gen.GetYaxis().SetLabelFont(42)
            his_ff_fd_gen.GetYaxis().SetRangeUser(0.0, 1.1 * max(his_ff_fd_gen.GetMaximum(), \
                his_ff_fd_eff.GetMaximum(), his_ff_fd_fold.GetMaximum()))
            his_ff_fd_gen.SetLineColor(1)
            his_ff_fd_eff.SetLineColor(2)
            his_ff_fd_fold.SetLineColor(3)
            leg_ff_fd = TLegend(.15, .65, .35, .85)
            leg_ff_fd.SetBorderSize(0)
            leg_ff_fd.SetFillColor(0)
            leg_ff_fd.SetFillStyle(0)
            leg_ff_fd.SetTextFont(42)
            leg_ff_fd.SetTextSize(0.035)
            his_ff_fd_gen.SetTitle("")
            his_ff_fd_eff.SetTitle("")
            his_ff_fd_fold.SetTitle("")
            his_ff_fd_gen.Draw("")
            his_ff_fd_eff.Draw("same")
            his_ff_fd_fold.Draw("same")
            leg_ff_fd.AddEntry(his_ff_fd_gen, "generated", "LEP")
            leg_ff_fd.AddEntry(his_ff_fd_eff, "eff.-scaled", "LEP")
            leg_ff_fd.AddEntry(his_ff_fd_fold, "folded", "LEP")
            leg_ff_fd.Draw()
            can_ff_fd_fold.SetLeftMargin(0.15)
            can_ff_fd_fold.SaveAs("%s/Feeddown-z-all_%s%s%s.eps" \
                % (self.d_resultsallpmc, self.case, self.typean, i_ptjet))

        file_out.cd()
        his_sim_z_fd_2d_gen.Write("fd_z_ptjet_gen")
        his_sim_z_fd_2d_eff.Write("fd_z_ptjet_eff")
        his_sim_z_fd_2d_folded.Write("fd_z_ptjet_fold")
        resp_z_proj.Write("resp_z_proj")

        file_resp.Close()
        file_eff.Close()
        file_out.Close()

    # pylint: disable=too-many-locals
    def side_band_sub(self):
        self.loadstyle()
        lfile = TFile.Open(self.n_filemass)
        func_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                            None, [self.case, self.typean])
        func_file = TFile.Open(func_filename, "READ")
        eff_file = TFile.Open("%s/efficiencies%s%s.root" % \
                              (self.d_resultsallpmc, self.case, self.typean))
        fileouts = TFile.Open("%s/side_band_sub%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "recreate")
        for imult in range(self.p_nbin2):
            heff = eff_file.Get("eff_mult%d" % imult)
            hz = None
            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                hzvsmass = lfile.Get("hzvsmass" + suffix)
                load_dir = func_file.GetDirectory(suffix)
                mass_fitter = load_dir.Get("fitter")
                mean = mass_fitter.GetMean()
                sigma = mass_fitter.GetSigma()
                binmasslow2sig = hzvsmass.GetXaxis().FindBin(mean - 2*sigma)
                masslow2sig = mean - 2*sigma
                binmasshigh2sig = hzvsmass.GetXaxis().FindBin(mean + 2*sigma)
                masshigh2sig = mean + 2*sigma
                binmasslow4sig = hzvsmass.GetXaxis().FindBin(mean - 4*sigma)
                masslow4sig = mean - 4*sigma
                binmasshigh4sig = hzvsmass.GetXaxis().FindBin(mean + 4*sigma)
                masshigh4sig = mean + 4*sigma
                binmasslow9sig = hzvsmass.GetXaxis().FindBin(mean - 9*sigma)
                masslow9sig = mean - 9*sigma
                binmasshigh9sig = hzvsmass.GetXaxis().FindBin(mean + 9*sigma)
                masshigh9sig = mean + 9*sigma

                hzsig = hzvsmass.ProjectionY("hzsig" + suffix, \
                             binmasslow2sig, binmasshigh2sig, "e")
                hzsig.Rebin(100)
                hzbkgleft = hzvsmass.ProjectionY("hzbkgleft" + suffix, \
                             binmasslow9sig, binmasslow4sig, "e")
                hzbkgleft.Rebin(100)
                hzbkgright = hzvsmass.ProjectionY("hzbkgright" + suffix, \
                             binmasshigh4sig, binmasshigh9sig, "e")
                hzbkgright.Rebin(100)
                hzbkg = hzbkgleft.Clone("hzbkg" + suffix)
                hzbkg.Add(hzbkgright)
                hzbkg_scaled = hzbkg.Clone("hzbkg_scaled" + suffix)
                bkg_fit = mass_fitter.GetBackgroundRecalcFunc()
                area_scale_denominator = bkg_fit.Integral(masslow9sig, masslow4sig) + \
                bkg_fit.Integral(masshigh4sig, masshigh9sig)
                area_scale = bkg_fit.Integral(masslow2sig, masshigh2sig)/area_scale_denominator
                hzsub = hzsig.Clone("hzsub" + suffix)
                hzsub.Add(hzbkg, -1*area_scale)
                hzsub_noteffscaled = hzsub.Clone("hzsub_noteffscaled" + suffix)
                hzbkg_scaled.Scale(area_scale)
                eff = heff.GetBinContent(ipt+1)
                hzsub.Scale(1.0/(eff*0.9545))
                if ipt == 0:
                    hz = hzsub.Clone("hz")
                else:
                    hz.Add(hzsub)
                fileouts.cd()
                hzsig.Write()
                hzbkgleft.Write()
                hzbkgright.Write()
                hzbkg.Write()
                hzsub.Write()
                hz.Write()
                cside = TCanvas('cside' + suffix, 'The Fit Canvas')
                cside.SetCanvasSize(1900, 1500)
                cside.SetWindowSize(500, 500)
                hzvsmass.Draw("colz")

                cside.SaveAs("%s/zvsInvMass%s%s_%s.eps" % (self.d_resultsallpdata,
                                                           self.case, self.typean, suffix))

                csubsig = TCanvas('csubsig' + suffix, 'The Side-Band Sub Signal Canvas')
                csubsig.SetCanvasSize(1900, 1500)
                csubsig.SetWindowSize(500, 500)
                hzsig.Draw()

                csubsig.SaveAs("%s/side_band_sub_signal%s%s_%s.eps" % \
                               (self.d_resultsallpdata, self.case, self.typean, suffix))

                csubbkg = TCanvas('csubbkg' + suffix, 'The Side-Band Sub Background Canvas')
                csubbkg.SetCanvasSize(1900, 1500)
                csubbkg.SetWindowSize(500, 500)
                hzbkg.Draw()

                csubbkg.SaveAs("%s/side_band_sub_background%s%s_%s.eps" % \
                               (self.d_resultsallpdata, self.case, self.typean, suffix))

                csubz = TCanvas('csubz' + suffix, 'The Side-Band Sub Canvas')
                csubz.SetCanvasSize(1900, 1500)
                csubz.SetWindowSize(500, 500)
                hzsub.Draw()

                csubz.SaveAs("%s/side_band_sub%s%s_%s.eps" % \
                             (self.d_resultsallpdata, self.case, self.typean, suffix))



                legsigbkgsubz = TLegend(.2, .65, .35, .85)
                legsigbkgsubz.SetBorderSize(0)
                legsigbkgsubz.SetFillColor(0)
                legsigbkgsubz.SetFillStyle(0)
                legsigbkgsubz.SetTextFont(42)
                legsigbkgsubz.SetTextSize(0.035)
                csigbkgsubz = TCanvas('csigbkgsubz' + suffix, 'The Side-Band Canvas')
                csigbkgsubz.SetCanvasSize(1900, 1500)
                csigbkgsubz.SetWindowSize(500, 500)
                legsigbkgsubz.AddEntry(hzsig, "signal", "LEP")
                hzsig.GetYaxis().SetRangeUser(0.0, max(hzsig.GetBinContent(hzsig.GetMaximumBin()), \
                    hzbkg_scaled.GetBinContent(hzbkg_scaled.GetMaximumBin()), \
                    hzsub_noteffscaled.GetBinContent(hzsub_noteffscaled.GetMaximumBin()))*1.2)
                hzsig.SetLineColor(2)
                hzsig.Draw()
                legsigbkgsubz.AddEntry(hzbkg_scaled, "side-band", "LEP")
                hzbkg_scaled.SetLineColor(3)
                hzbkg_scaled.Draw("same")
                legsigbkgsubz.AddEntry(hzsub_noteffscaled, "subtracted", "LEP")
                hzsub_noteffscaled.SetLineColor(4)
                hzsub_noteffscaled.Draw("same")
                legsigbkgsubz.Draw()

                csigbkgsubz.SaveAs("%s/side_band_%s%s_%s.eps" % \
                             (self.d_resultsallpdata, self.case, self.typean, suffix))
            cz = TCanvas('cz' + suffix, 'The Efficiency Corrected Signal Yield Canvas')
            cz.SetCanvasSize(1900, 1500)
            cz.SetWindowSize(500, 500)
            hz.Draw()

            cz.SaveAs("%s/efficiencycorrected_fullsub%s%s_%s_%.2f_%.2f.eps" % \
                      (self.d_resultsallpdata, self.case, self.typean, self.v_var2_binning, \
                       self.lvar2_binmin[imult], self.lvar2_binmax[imult]))
        fileouts.Close()

    def plotter(self):
        gROOT.SetBatch(True)
        self.loadstyle()

        fileouteff = TFile.Open("%s/efficiencies%s%s.root" % \
                                (self.d_resultsallpmc, self.case, self.typean))
        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        fileoutyield = TFile.Open(yield_filename, "READ")
        fileoutcross = TFile.Open("%s/finalcross%s%s.root" % \
                                  (self.d_resultsallpdata, self.case, self.typean), "recreate")

        cCrossvsvar1 = TCanvas('cCrossvsvar1', 'The Fit Canvas')
        cCrossvsvar1.SetCanvasSize(1900, 1500)
        cCrossvsvar1.SetWindowSize(500, 500)
        cCrossvsvar1.SetLogy()

        legvsvar1 = TLegend(.5, .65, .7, .85)
        legvsvar1.SetBorderSize(0)
        legvsvar1.SetFillColor(0)
        legvsvar1.SetFillStyle(0)
        legvsvar1.SetTextFont(42)
        legvsvar1.SetTextSize(0.035)

        listvalues = []
        listvalueserr = []

        for imult in range(self.p_nbin2):
            listvalpt = []
            bineff = -1
            if self.p_bineff is None:
                bineff = imult
                print("Using efficiency for each var2 bin")
            else:
                bineff = self.p_bineff
                print("Using efficiency always from bin=", bineff)
            heff = fileouteff.Get("eff_mult%d" % (bineff))
            hcross = fileoutyield.Get("hyields%d" % (imult))
            hcross.Divide(heff)
            hcross.SetLineColor(imult+1)
            norm = 2 * self.p_br * self.p_nevents / (self.p_sigmamb * 1e12)
            hcross.Scale(1./norm)
            fileoutcross.cd()
            hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnmeson)
            hcross.GetYaxis().SetTitle("d#sigma/d#it{p}_{T} (%s) %s" %
                                       (self.p_latexnmeson, self.typean))
            hcross.SetName("hcross%d" % imult)
            hcross.GetYaxis().SetRangeUser(1e1, 1e10)
            legvsvar1endstring = "%.1f < %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legvsvar1.AddEntry(hcross, legvsvar1endstring, "LEP")
            hcross.Draw("same")
            hcross.Write()
            listvalpt = [hcross.GetBinContent(ipt+1) for ipt in range(self.p_nptbins)]
            listvalues.append(listvalpt)
            listvalerrpt = [hcross.GetBinError(ipt+1) for ipt in range(self.p_nptbins)]
            listvalueserr.append(listvalerrpt)
        legvsvar1.Draw()
        cCrossvsvar1.SaveAs("%s/Cross%s%sVs%s.eps" % (self.d_resultsallpdata,
                                                      self.case, self.typean, self.v_var_binning))

        cCrossvsvar2 = TCanvas('cCrossvsvar2', 'The Fit Canvas')
        cCrossvsvar2.SetCanvasSize(1900, 1500)
        cCrossvsvar2.SetWindowSize(500, 500)
        cCrossvsvar2.SetLogy()

        legvsvar2 = TLegend(.5, .65, .7, .85)
        legvsvar2.SetBorderSize(0)
        legvsvar2.SetFillColor(0)
        legvsvar2.SetFillStyle(0)
        legvsvar2.SetTextFont(42)
        legvsvar2.SetTextSize(0.035)
        hcrossvsvar2 = [TH1F("hcrossvsvar2" + "pt%d" % ipt, "", \
                        self.p_nbin2, array("d", self.var2ranges)) \
                        for ipt in range(self.p_nptbins)]

        for ipt in range(self.p_nptbins):
            print("pt", ipt)
            for imult in range(self.p_nbin2):
                hcrossvsvar2[ipt].SetLineColor(ipt+1)
                hcrossvsvar2[ipt].GetXaxis().SetTitle("%s" % self.p_latexbin2var)
                hcrossvsvar2[ipt].GetYaxis().SetTitle(self.p_latexnmeson)
                binmulrange = self.var2ranges[imult+1]-self.var2ranges[imult]
                if self.p_dodoublecross is True:
                    hcrossvsvar2[ipt].SetBinContent(imult+1, listvalues[imult][ipt]/binmulrange)
                    hcrossvsvar2[ipt].SetBinError(imult+1, listvalueserr[imult][ipt]/binmulrange)
                else:
                    hcrossvsvar2[ipt].SetBinContent(imult+1, listvalues[imult][ipt])
                    hcrossvsvar2[ipt].SetBinError(imult+1, listvalueserr[imult][ipt])

                hcrossvsvar2[ipt].GetYaxis().SetRangeUser(1e4, 1e10)
            legvsvar2endstring = "%.1f < %s < %.1f GeV/#it{c}" % \
                   (self.lpt_finbinmin[ipt], "#it{p}_{T}", self.lpt_finbinmax[ipt])
            hcrossvsvar2[ipt].Draw("same")
            legvsvar2.AddEntry(hcrossvsvar2[ipt], legvsvar2endstring, "LEP")
        legvsvar2.Draw()
        cCrossvsvar2.SaveAs("%s/Cross%s%sVs%s.eps" % (self.d_resultsallpdata,
                                                      self.case, self.typean, self.v_var2_binning))


    @staticmethod
    def calculate_norm(mode, filename, trigger, var, multmin, multmax, doweight):
        fileout = TFile.Open(filename, "read")
        labeltrigger = "hbit%svs%s" % (trigger, var)
        norm = -1

        if not fileout:
            return -1
        if mode == 0:
            namehistomulti = None
            if doweight is True:
                namehistomulti = "hmultweighted%svs%s" % (trigger, var)
            else:
                namehistomulti = "hmult%svs%s" % (trigger, var)
            hmult = fileout.Get(namehistomulti)
            if not hmult:
                print("MISSING NORMALIZATION MULTIPLICITY")

            binminv = hmult.GetXaxis().FindBin(multmin)
            binmaxv = hmult.GetXaxis().FindBin(multmax)
            norm = hmult.Integral(binminv, binmaxv)

        if mode == 1:
            namehsel = None
            namehnovtx = None
            namehvtxout = None
            if doweight is False:
                print("Counting n.events not corrected for the trigger")
                namehsel = 'sel_' + labeltrigger
                namehnovtx = 'novtx_' + labeltrigger
                namehvtxout = 'vtxout_' + labeltrigger
            else:
                print("Counting n.events corrected for the trigger")
                namehsel = 'sel_' + labeltrigger + "weighted"
                namehnovtx = 'novtx_' + labeltrigger + "weighted"
                namehvtxout = 'vtxout_' + labeltrigger + "weighted"
            print(namehsel)
            print(namehnovtx)
            print(namehvtxout)
            hsel = fileout.Get(namehsel)
            hnovt = fileout.Get(namehnovtx)
            hvtxout = fileout.Get(namehvtxout)

            binminv = hsel.GetXaxis().FindBin(multmin)
            binmaxv = hsel.GetXaxis().FindBin(multmax)

            if not hsel:
                print("Missing hsel")
            if not hnovt:
                print("Missing hnovt")
            if not hvtxout:
                print("Missing hvtxout")
            n_sel = hsel.Integral(binminv, binmaxv)
            n_novtx = hnovt.Integral(binminv, binmaxv)
            n_vtxout = hvtxout.Integral(binminv, binmaxv)
            if n_sel + n_vtxout > 0:
                norm = (n_sel + n_novtx) - n_novtx * n_vtxout / (n_sel + n_vtxout)
        return norm

    def makenormyields(self):
        gROOT.SetBatch(True)
        self.loadstyle()
        #self.test_aliphysics()
        #filedataval = TFile.Open(self.f_evtnorm)
        fileouteff = "%s/efficiencies%s%s.root" % \
                      (self.d_resultsallpmc, self.case, self.typean)
        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        gROOT.LoadMacro("HFPtSpectrum.C")
        gROOT.LoadMacro("HFPtSpectrum2.C")
        from ROOT import HFPtSpectrum, HFPtSpectrum2
        for imult in range(self.p_nbin2):
            bineff = -1
            if self.p_bineff is None:
                bineff = imult
                print("Using efficiency for each var2 bin")
            else:
                bineff = self.p_bineff
                print("Using efficiency always from bin=", bineff)
            namehistoeffprompt = "eff_mult%d" % bineff
            namehistoefffeed = "eff_fd_mult%d" % bineff
            nameyield = "hyields%d" % imult
            fileoutcrossmult = "%s/finalcross%s%smult%d.root" % \
                (self.d_resultsallpdata, self.case, self.typean, imult)
            #hNorm = lfile.Get("hEvForNorm_mult%d" % imult)
            #norm = hNorm.GetBinContent(1)
            norm = -1
            lfile = TFile.Open(self.n_filemass)
            hNorm = lfile.Get("hEvForNorm_mult%d" % imult)
            normfromhisto = hNorm.GetBinContent(1)
            norm = self.calculate_norm(1, self.f_evtnorm, self.triggerbit, \
                          self.v_var2_binning_gen, self.lvar2_binmin[imult], \
                          self.lvar2_binmax[imult], self.apply_weights)
            normold = self.calculate_norm(0, self.f_evtnorm, self.triggerbit, \
                          self.v_var2_binning_gen, self.lvar2_binmin[imult], \
                          self.lvar2_binmax[imult], self.apply_weights)
            print("--------- NORMALIZATION -----------")
            print(self.triggerbit, self.v_var2_binning,
                  self.lvar2_binmin[imult], self.lvar2_binmax[imult])
            print("N. events selected=", normold, "N. events counter =", norm)
            if abs(norm - normfromhisto)/norm > 0.05 and self.apply_weights is False:
                print("inconistent number of events", normfromhisto, norm)

            filecrossmb = None
            if self.p_fprompt_from_mb is True and self.p_fd_method == 2:
                if self.p_corrmb_typean is not None:
                    pathtoreplace = os.path.basename(os.path.normpath(self.d_resultsallpdata))
                    pathreplaceby = os.path.basename(os.path.normpath(self.results_mb))
                    resultpathmb = self.d_resultsallpdata.replace(pathtoreplace, pathreplaceby)
                    filecrossmb = "%s/finalcross%s%smult0.root" % (resultpathmb, self.case, \
                                                                   self.p_corrmb_typean)
                    if self.get_crossmb_from_path is not None:
                        filecrossmb = self.path_for_crossmb
                    self.logger.info("Looking for %s", filecrossmb)
                    if os.path.exists(filecrossmb):
                        self.logger.info("Calculating spectra using fPrompt from MB. "\
                                         "Assuming MB is bin 0: %s", filecrossmb)
                    else:
                        self.logger.fatal("First run MB if you want to use MB fPrompt!")

            if self.p_fprompt_from_mb is None or self.p_fd_method != 2 or \
              (imult == 0 and self.p_corrmb_typean is None):
                HFPtSpectrum(self.p_indexhpt, \
                 "inputsCross/D0DplusDstarPredictions_13TeV_y05_all_300416_BDShapeCorrected.root", \
                 fileouteff, namehistoeffprompt, namehistoefffeed, yield_filename, nameyield, \
                 fileoutcrossmult, norm, self.p_sigmav0 * 1e12, self.p_fd_method, self.p_cctype)
            else:
                if filecrossmb is None:
                    filecrossmb = "%s/finalcross%s%smult0.root" % \
                                   (self.d_resultsallpdata, self.case, self.typean)
                    self.logger.info("Calculating spectra using fPrompt from MB. "\
                                     "Assuming MB is bin 0: %s", filecrossmb)
                HFPtSpectrum2(filecrossmb, self.p_triggereff[imult], self.p_triggereffunc[imult], \
                              fileouteff, namehistoeffprompt, namehistoefffeed, \
                              yield_filename, nameyield, fileoutcrossmult, norm, \
                              self.p_sigmav0 * 1e12)

        fileoutcrosstot = TFile.Open("%s/finalcross%s%smulttot.root" % \
            (self.d_resultsallpdata, self.case, self.typean), "recreate")

        for imult in range(self.p_nbin2):
            fileoutcrossmult = "%s/finalcross%s%smult%d.root" % \
                (self.d_resultsallpdata, self.case, self.typean, imult)
            f_fileoutcrossmult = TFile.Open(fileoutcrossmult)
            if not f_fileoutcrossmult:
                continue
            hcross = f_fileoutcrossmult.Get("histoSigmaCorr")
            hcross.SetName("histoSigmaCorr%d" % imult)
            fileoutcrosstot.cd()
            hcross.Write()
        fileoutcrosstot.Close()


    def plotternormyields(self):
        gROOT.SetBatch(True)
        cCrossvsvar1 = TCanvas('cCrossvsvar1', 'The Fit Canvas')
        cCrossvsvar1.SetCanvasSize(1900, 1500)
        cCrossvsvar1.SetWindowSize(500, 500)
        cCrossvsvar1.SetLogy()
        cCrossvsvar1.cd()
        legvsvar1 = TLegend(.5, .65, .7, .85)
        legvsvar1.SetBorderSize(0)
        legvsvar1.SetFillColor(0)
        legvsvar1.SetFillStyle(0)
        legvsvar1.SetTextFont(42)
        legvsvar1.SetTextSize(0.035)
        fileoutcrosstot = TFile.Open("%s/finalcross%s%smulttot.root" % \
            (self.d_resultsallpdata, self.case, self.typean))

        for imult in range(self.p_nbin2):
            hcross = fileoutcrosstot.Get("histoSigmaCorr%d" % imult)
            hcross.Scale(1./(self.p_sigmav0 * 1e12))
            hcross.SetLineColor(imult+1)
            hcross.SetMarkerColor(imult+1)
            hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnmeson)
            hcross.GetYaxis().SetTitleOffset(1.3)
            hcross.GetYaxis().SetTitle("Corrected yield/events (%s) %s" %
                                       (self.p_latexnmeson, self.typean))
            hcross.GetYaxis().SetRangeUser(1e-10, 1)
            legvsvar1endstring = "%.1f #leq %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legvsvar1.AddEntry(hcross, legvsvar1endstring, "LEP")
            hcross.Draw("same")
        legvsvar1.Draw()
        cCrossvsvar1.SaveAs("%s/CorrectedYieldsNorm%s%sVs%s.eps" % (self.d_resultsallpdata,
                                                                    self.case, self.typean,
                                                                    self.v_var_binning))
    def studyevents(self):
        gROOT.SetBatch(True)
        self.loadstyle()
        filedata = TFile.Open(self.f_evtvaldata)
        triggerlist = ["HighMultV0", "HighMultSPD", "INT7"]
        varlist = ["v0m_corr", "n_tracklets_corr", "perc_v0m"]
        fileout_name = "%s/correctionsweights.root" % self.d_valevtdata
        fileout = TFile.Open(fileout_name, "recreate")
        fileout.cd()
        for ivar, var in enumerate(varlist):
            labelMB = "hbitINT7vs%s" % (var)
            hden = filedata.Get(labelMB)
            hden.SetName("hmultINT7vs%s" % (var))
            hden.Write()
            for trigger in triggerlist:
                labeltriggerANDMB = "hbit%sANDINT7vs%s" % (trigger, var)
                labeltrigger = "hbit%svs%s" % (trigger, var)
                heff = filedata.Get(labeltriggerANDMB)
                if not heff or not hden:
                    continue
                heff.Divide(heff, hden, 1.0, 1.0, "B")
                hratio = filedata.Get(labeltrigger)
                hmult = hratio.Clone("hmult%svs%s" % (trigger, var))
                hmultweighted = hratio.Clone("hmultweighted%svs%s" % (trigger, var))
                if not hratio:
                    continue
                hratio.Divide(hratio, hden, 1.0, 1.0, "B")

                ctrigger = TCanvas('ctrigger%s' % trigger, 'The Fit Canvas')
                ctrigger.SetCanvasSize(3500, 2000)
                ctrigger.Divide(3, 2)

                ctrigger.cd(1)
                heff.SetMaximum(2.)
                heff.GetXaxis().SetTitle("offline %s" % var)
                heff.SetMinimum(0.)
                heff.GetYaxis().SetTitle("trigger efficiency from MB events")
                heff.SetLineColor(1)
                heff.Draw()
                heff.Write()

                ctrigger.cd(2)
                hratio.GetXaxis().SetTitle("offline %s" % var)
                hratio.GetYaxis().SetTitle("ratio triggered/MB")
                hratio.GetYaxis().SetTitleOffset(1.3)
                hratio.Write()
                hratio.SetLineColor(1)
                hratio.Draw()
                func = TF1("func_%s_%s" % (trigger, var), \
                           "([0]/(1+TMath::Exp(-[1]*(x-[2]))))", 0, 1000)
                if ivar == 0:
                    func.SetParameters(300, .1, 570)
                    func.SetParLimits(1, 0., 10.)
                    func.SetParLimits(2, 0., 1000.)
                    func.SetRange(550., 1100.)
                    func.SetLineWidth(1)
                    hratio.Fit(func, "L", "", 550, 1100)
                    func.Draw("same")
                    func.SetLineColor(ivar+1)
                if ivar == 1:
                    func.SetParameters(100, .1, 50)
                    func.SetParLimits(1, 0., 10.)
                    func.SetParLimits(2, 0., 200.)
                    func.SetRange(45., 105)
                    func.SetLineWidth(1)
                    hratio.Fit(func, "L", "", 45, 105)
                    func.SetLineColor(ivar+1)
                if ivar == 2:
                    func.SetParameters(315, -30., .2)
                    func.SetParLimits(1, -100., 0.)
                    func.SetParLimits(2, 0., .5)
                    func.SetRange(0., .15)
                    func.SetLineWidth(1)
                    hratio.Fit(func, "w", "", 0, .15)
                    func.SetLineColor(ivar+1)
                func.Write()
                funcnorm = func.Clone("funcnorm_%s_%s" % (trigger, var))
                funcnorm.FixParameter(0, funcnorm.GetParameter(0)/funcnorm.GetMaximum())
                funcnorm.Write()
                ctrigger.cd(3)
                maxhistx = 0
                if ivar == 0:
                    minhistx = 300
                    maxhistx = 1000
                    fulleffmin = 700
                    fulleffmax = 800
                elif ivar == 1:
                    minhistx = 40
                    maxhistx = 150
                    fulleffmin = 80
                    fulleffmax = 90
                else:
                    minhistx = .0
                    maxhistx = .5
                    fulleffmin = 0.
                    fulleffmax = 0.03
                hempty = TH1F("hempty_%d" % ivar, "hempty", 100, 0, maxhistx)
                hempty.GetYaxis().SetTitleOffset(1.2)
                hempty.GetYaxis().SetTitleFont(42)
                hempty.GetXaxis().SetTitleFont(42)
                hempty.GetYaxis().SetLabelFont(42)
                hempty.GetXaxis().SetLabelFont(42)
                hempty.GetXaxis().SetTitle("offline %s" % var)
                hempty.GetYaxis().SetTitle("trigger efficiency from effective")
                hempty.Draw()
                funcnorm.SetLineColor(1)
                funcnorm.Draw("same")

                ctrigger.cd(4)
                gPad.SetLogy()
                leg1 = TLegend(.2, .75, .4, .85)
                leg1.SetBorderSize(0)
                leg1.SetFillColor(0)
                leg1.SetFillStyle(0)
                leg1.SetTextFont(42)
                leg1.SetTextSize(0.035)
                hmult.GetXaxis().SetTitle("offline %s" % var)
                hmult.GetYaxis().SetTitle("entries")
                hmult.SetLineColor(1)
                hden.SetLineColor(2)
                hmultweighted.SetLineColor(3)
                hmult.Draw()
                hmult.SetMaximum(1e10)
                hden.Draw("same")
                for ibin in range(hmult.GetNbinsX()):
                    myweight = funcnorm.Eval(hmult.GetBinCenter(ibin + 1))
                    hmultweighted.SetBinContent(ibin + 1, hmult.GetBinContent(ibin+1) / myweight)
                hmult.Write()
                hmultweighted.Write()
                hmultweighted.Draw("same")
                leg1.AddEntry(hden, "MB distribution", "LEP")
                leg1.AddEntry(hmult, "triggered uncorr", "LEP")
                leg1.AddEntry(hmultweighted, "triggered corr.", "LEP")
                leg1.Draw()
                print("event before", hmult.GetEntries(), "after",
                      hmultweighted.Integral())
                ctrigger.cd(5)
                leg2 = TLegend(.2, .75, .4, .85)
                leg2.SetBorderSize(0)
                leg2.SetFillColor(0)
                leg2.SetFillStyle(0)
                leg2.SetTextFont(42)
                leg2.SetTextSize(0.035)
                linear = TF1("lin_%s_%s" % (trigger, var), \
                           "[0]", fulleffmin, fulleffmax)
                hratioMBcorr = hmultweighted.Clone("hratioMBcorr")
                hratioMBcorr.Divide(hden)
                hratioMBuncorr = hmult.Clone("hratioMBuncorr")
                hratioMBuncorr.Divide(hden)
                hratioMBuncorr.Fit(linear, "w", "", fulleffmin, fulleffmax)
                hratioMBuncorr.Scale(1./linear.GetParameter(0))
                hratioMBcorr.Scale(1./linear.GetParameter(0))
                hratioMBcorr.SetLineColor(3)
                hratioMBuncorr.SetLineColor(2)
                hratioMBcorr.GetXaxis().SetTitle("offline %s" % var)
                hratioMBcorr.GetYaxis().SetTitle("entries")
                hratioMBcorr.GetXaxis().SetRangeUser(minhistx, maxhistx)
                hratioMBcorr.GetYaxis().SetRangeUser(0.8, 1.2)
                hratioMBcorr.Draw()
                hratioMBuncorr.Draw("same")
                leg2.AddEntry(hratioMBcorr, "triggered/MB", "LEP")
                leg2.AddEntry(hratioMBuncorr, "triggered/MB corr.", "LEP")
                leg2.Draw()
                ctrigger.cd(6)
                ptext = TPaveText(.05, .1, .95, .8)
                ptext.AddText("%s" % (trigger))
                ptext.AddText("MB events=%f M" % (float(hden.Integral())/1.e6))
                ptext.AddText("%s events=%f M" % (trigger, float(hmult.Integral())/1.e6))
                ptext.Draw()

                hsel = filedata.Get('sel_' + labeltrigger)
                hnovtx = filedata.Get('novtx_' + labeltrigger)
                hvtxout = filedata.Get('vtxout_' + labeltrigger)
                hselweighted = hsel.Clone('sel_' + labeltrigger + "weighted")
                hnovtxweighted = hnovtx.Clone('novtx_' + labeltrigger + "weighted")
                hvtxoutweighted = hvtxout.Clone('vtxout_' + labeltrigger + "weighted")

                for ibin in range(hmult.GetNbinsX()):
                    myweight = funcnorm.Eval(hsel.GetBinCenter(ibin + 1))
                    hselweighted.SetBinContent(ibin + 1, \
                        hsel.GetBinContent(ibin+1) / myweight)
                    hnovtxweighted.SetBinContent(ibin + 1, \
                        hnovtx.GetBinContent(ibin+1) / myweight)
                    hvtxoutweighted.SetBinContent(ibin + 1, \
                        hvtxout.GetBinContent(ibin+1) / myweight)
                hsel.Write()
                hnovtx.Write()
                hvtxout.Write()
                hselweighted.Write()
                hnovtxweighted.Write()
                hvtxoutweighted.Write()

                ctrigger.SaveAs(self.make_file_path(self.d_valevtdata, \
                        "ctrigger_%s_%s" % (trigger, var), "eps", \
                        None, None))

        cscatter = TCanvas("cscatter", 'The Fit Canvas')
        cscatter.SetCanvasSize(2100, 800)
        cscatter.Divide(3, 1)
        hv0mvsperc = filedata.Get("hv0mvsperc")
        hntrklsperc = filedata.Get("hntrklsperc")
        hntrklsv0m = filedata.Get("hntrklsv0m")
        if hv0mvsperc:
            cscatter.cd(1)
            gPad.SetLogx()
            hv0mvsperc.GetXaxis().SetTitle("percentile (max value = 100)")
            hv0mvsperc.GetYaxis().SetTitle("V0M corrected for z")
            hv0mvsperc.Draw("colz")
        if hntrklsperc:
            cscatter.cd(2)
            gPad.SetLogx()
            gPad.SetLogz()
            hntrklsperc.GetYaxis().SetRangeUser(0., 200.)
            hntrklsperc.GetXaxis().SetTitle("percentile (max value = 100)")
            hntrklsperc.GetYaxis().SetTitle("SPD ntracklets for z")
            hntrklsperc.Draw("colz")
        if hntrklsv0m:
            cscatter.cd(3)
            hntrklsv0m.GetYaxis().SetRangeUser(0., 200.)
            gPad.SetLogx()
            gPad.SetLogz()
            hntrklsv0m.GetXaxis().SetTitle("V0M corrected for z")
            hntrklsv0m.GetYaxis().SetTitle("SPD ntracklets for z")
            hntrklsv0m.Draw("colz")
        cscatter.SaveAs(self.make_file_path(self.d_valevtdata, "cscatter", "eps", \
                                            None, None))
