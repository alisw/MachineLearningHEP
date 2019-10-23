############################################################################
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
from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes
from machine_learning_hep.utilities import folding, setup_histogram, setup_pad, setup_legend, setup_tgraph, draw_latex, tg_sys
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


        self.lvar2_binmin_reco = datap["analysis"][self.typean].get("sel_binmin2_reco", None)
        self.lvar2_binmax_reco = datap["analysis"][self.typean].get("sel_binmax2_reco", None)
        self.lvar2_binmin_gen = datap["analysis"][self.typean].get("sel_binmin2_gen", None)
        self.lvar2_binmax_gen = datap["analysis"][self.typean].get("sel_binmax2_gen", None)
        self.lvarshape_binmin_reco = datap["analysis"][self.typean].get("sel_binminshape_reco", None)
        self.lvarshape_binmax_reco = datap["analysis"][self.typean].get("sel_binmaxshape_reco", None)
        self.lvarshape_binmin_gen = datap["analysis"][self.typean].get("sel_binminshape_gen", None)
        self.lvarshape_binmax_gen = datap["analysis"][self.typean].get("sel_binmaxshape_gen", None)

        self.niter_unfolding = datap["analysis"][self.typean].get("niterunfolding", None)
        self.choice_iter_unfolding = datap["analysis"][self.typean].get("niterunfoldingchosen", None)
        self.niterunfoldingregup = datap["analysis"][self.typean].get("niterunfoldingregup", None)
        self.niterunfoldingregdown = datap["analysis"][self.typean].get("niterunfoldingregdown", None)

        self.signal_sigma = datap["analysis"][self.typean].get("signal_sigma", None)
        self.sideband_sigma_1_left = datap["analysis"][self.typean].get("sideband_sigma_1_left", None)
        self.sideband_sigma_1_right = datap["analysis"][self.typean].get("sideband_sigma_1_right", None)
        self.sideband_sigma_2_left = datap["analysis"][self.typean].get("sideband_sigma_2_left", None)
        self.sideband_sigma_2_right = datap["analysis"][self.typean].get("sideband_sigma_2_right", None)
        self.sigma_scale = datap["analysis"][self.typean].get("sigma_scale", None)
        self.sidebandleftonly = datap["analysis"][self.typean].get("sidebandleftonly", None)

        self.powheg_path_prompt = datap["analysis"][self.typean].get("powheg_path_prompt", None)
        self.powheg_path_nonprompt = datap["analysis"][self.typean].get("powheg_path_nonprompt", None)

        self.powheg_prompt_variations = datap["analysis"][self.typean].get("powheg_prompt_variations", None)
        self.powheg_prompt_variations_path = datap["analysis"][self.typean].get("powheg_prompt_variations_path", None)

        self.pythia8_prompt_variations = datap["analysis"][self.typean].get("pythia8_prompt_variations", None)
        self.pythia8_prompt_variations_path = datap["analysis"][self.typean].get("pythia8_prompt_variations_path", None)

        self.systematic_catagories = datap["analysis"][self.typean].get("systematic_catagories", None)
        self.systematic_variations = datap["analysis"][self.typean].get("systematic_variations", None)
        self.systematic_correlation = datap["analysis"][self.typean].get("systematic_correlation", None)
        self.systematic_rms  = datap["analysis"][self.typean].get("systematic_rms", None)
        self.systematic_symmetrise  = datap["analysis"][self.typean].get("systematic_symmetrise", None)

        self.branching_ratio = datap["analysis"][self.typean].get("branching_ratio", None)
        self.xsection_inel = datap["analysis"][self.typean].get("xsection_inel", None)

        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.p_nbin2 = len(self.lvar2_binmin)
        self.p_nbin2_reco = len(self.lvar2_binmin_reco)
        self.p_nbin2_gen = len(self.lvar2_binmin_gen)
        self.p_nbinshape_reco = len(self.lvarshape_binmin_reco)
        self.p_nbinshape_gen = len(self.lvarshape_binmin_gen)

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
        self.var2ranges_reco = self.lvar2_binmin_reco.copy()
        self.var2ranges_reco.append(self.lvar2_binmax_reco[-1])
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        self.varshaperanges_reco = self.lvarshape_binmin_reco.copy()
        self.varshaperanges_reco.append(self.lvarshape_binmax_reco[-1])
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])
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
        feeddown_input_file = TFile.Open(self.n_fileff)
        file_eff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc, \
                              self.case, self.typean))
        fileouts = TFile.Open("%s/feeddown%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "recreate")

        response_matrix = feeddown_input_file.Get("response_matrix_nonprompt")
        #input_data = feeddown_input_file.Get("hzvsjetptvscandpt_gen_nonprompt")
        powheg_input_file = TFile.Open(self.powheg_path_nonprompt)
        input_data = powheg_input_file.Get("fh3_feeddown")
        output_template = feeddown_input_file.Get("hzvsjetpt_reco")

        hzvsjetpt_gen_nocuts = feeddown_input_file.Get("hzvsjetpt_gen_nocuts_nonprompt")
        hzvsjetpt_gen_eff = feeddown_input_file.Get("hzvsjetpt_gen_cuts_nonprompt")
        hzvsjetpt_gen_eff.Divide(hzvsjetpt_gen_nocuts)

        hzvsjetpt_reco_nocuts = feeddown_input_file.Get("hzvsjetpt_reco_nocuts_nonprompt")
        hzvsjetpt_reco_eff = feeddown_input_file.Get("hzvsjetpt_reco_cuts_nonprompt")
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)
        
        sideband_input_data_file = TFile.Open("%s/sideband_sub%s%s.root" % \
                                               (self.d_resultsallpdata, self.case, self.typean))
        sideband_input_data = sideband_input_data_file.Get("hzvsjetpt")

        hz_genvsreco_list=[]
        hjetpt_genvsreco_list=[]
        
        hjetpt_fracdiff_list=[]
        hz_fracdiff_list=[]
        heff_pr_list=[]
        heff_fd_list=[]
        input_data_zvsjetpt_list=[]
        input_data_scaled = TH2F()

        
        cgen_eff = TCanvas('cgen_eff_nonprompt ', 'gen efficiency applied to feedown')
        pgen_eff = TPad('pgen_eff_nonprompt ', 'gen efficiency applied to feedown',0.0,0.0,1.0,1.0)
        setup_pad(pgen_eff)
        cgen_eff.SetCanvasSize(1900, 1500)
        cgen_eff.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt_gen_eff)
        hzvsjetpt_gen_eff.SetXTitle("z^{gen}")
        hzvsjetpt_gen_eff.SetYTitle("p_{T,jet}^{gen}")
        hzvsjetpt_gen_eff.Draw("text")
        cgen_eff.SaveAs("%s/cgen_kineeff_nonprompt.eps" % (self.d_resultsallpdata))
        
        creco_eff = TCanvas('creco_eff_nonprompt ', 'reco efficiency applied to feedown')
        preco_eff = TPad('preco_eff_nonprompt ', 'reco efficiency applied to feedown',0.0,0.0,1.0,1.0)
        setup_pad(preco_eff)
        creco_eff.SetCanvasSize(1900, 1500)
        creco_eff.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt_reco_eff)
        hzvsjetpt_reco_eff.SetXTitle("z^{reco}")
        hzvsjetpt_reco_eff.SetYTitle("p_{T,jet}^{reco}")
        hzvsjetpt_reco_eff.Draw("text")
        creco_eff.SaveAs("%s/creco_kineeff_nonprompt.eps" % (self.d_resultsallpdata))

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            hz_genvsreco_list.append(feeddown_input_file.Get("hz_genvsreco_nonprompt"+suffix))

            cz_genvsreco = TCanvas('cz_genvsreco_nonprompt'+suffix, 'response matrix 2D projection')
            pz_genvsreco = TPad('pz_genvsreco_nonprompt ', 'response matrix 2D projection',0.0,0.001,1.0,1.0)
            setup_pad(pz_genvsreco)
            cz_genvsreco.SetLogz()
            pz_genvsreco.SetLogz()
            cz_genvsreco.SetCanvasSize(1900, 1500)
            cz_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hz_genvsreco_list[ibin2])
            hz_genvsreco_list[ibin2].SetXTitle("z^{gen}")
            hz_genvsreco_list[ibin2].SetYTitle("z^{reco}")
            hz_genvsreco_list[ibin2].Draw("colz")
            latex = TLatex(0.2,0.85,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cz_genvsreco.SaveAs("%s/cz_genvsreco_nonprompt_%s.eps" % (self.d_resultsallpdata,suffix))

        for ibinshape in range(self.p_nbinshape_reco):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco_list.append(feeddown_input_file.Get("hjetpt_genvsreco_nonprompt"+suffix))

            cjetpt_genvsreco = TCanvas('cjetpt_genvsreco_nonprompt'+suffix, 'response matrix 2D projection')
            pjetpt_genvsreco = TPad('pjetpt_genvsreco_nonprompt'+suffix, 'response matrix 2D projection',0.0,0.001,1.0,1.0)
            setup_pad(pjetpt_genvsreco)
            cjetpt_genvsreco.SetLogz()
            pjetpt_genvsreco.SetLogz()
            cjetpt_genvsreco.SetCanvasSize(1900, 1500)
            cjetpt_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hjetpt_genvsreco_list[ibinshape])
            hjetpt_genvsreco_list[ibinshape].SetXTitle("p_{T,jet}^{gen}")
            hjetpt_genvsreco_list[ibinshape].SetYTitle("p_{T,jet}^{reco}")
            hjetpt_genvsreco_list[ibinshape].Draw("colz")
            latex = TLatex(0.2,0.8,"%.2f < z < %.2f" % (self.lvarshape_binmin_reco[ibinshape],self.lvarshape_binmax_reco[ibinshape]))
            draw_latex(latex)
            cjetpt_genvsreco.SaveAs("%s/cjetpt_genvsreco_nonprompt_%s.eps" % (self.d_resultsallpdata,suffix))

        hz_genvsreco_full=feeddown_input_file.Get("hz_genvsreco_full_nonprompt")
        hjetpt_genvsreco_full=feeddown_input_file.Get("hjetpt_genvsreco_full_nonprompt")

        cz_genvsreco = TCanvas('cz_genvsreco_full_nonprompt', 'response matrix 2D projection')
        pz_genvsreco = TPad('pz_genvsreco_full_nonprompt'+suffix, 'response matrix 2D projection',0.0,0.001,1.0,1.0)
        setup_pad(pz_genvsreco)
        cz_genvsreco.SetLogz()
        pz_genvsreco.SetLogz()
        cz_genvsreco.SetCanvasSize(1900, 1500)
        cz_genvsreco.SetWindowSize(500, 500)
        setup_histogram(hz_genvsreco_full)
        hz_genvsreco_full.SetXTitle("z^{gen}")
        hz_genvsreco_full.SetYTitle("z^{reco}")
        hz_genvsreco_full.Draw("colz")
        cz_genvsreco.SaveAs("%s/cz_genvsreco_full_nonprompt.eps" % (self.d_resultsallpdata))

        cjetpt_genvsreco = TCanvas('cjetpt_genvsreco_full_nonprompt', 'response matrix 2D projection')
        pjetpt_genvsreco = TPad('pjetpt_genvsreco_full_nonprompt'+suffix, 'response matrix 2D projection',0.0,0.001,1.0,1.0)
        setup_pad(pjetpt_genvsreco)
        cjetpt_genvsreco.SetLogz()
        pjetpt_genvsreco.SetLogz()
        cjetpt_genvsreco.SetCanvasSize(1900, 1500)
        cjetpt_genvsreco.SetWindowSize(500, 500)
        setup_histogram(hjetpt_genvsreco_full)
        hjetpt_genvsreco_full.SetXTitle("p_{T,jet}^{gen}")
        hjetpt_genvsreco_full.SetYTitle("p_{T,jet}^{reco}")
        hjetpt_genvsreco_full.Draw("colz")
        cjetpt_genvsreco.SaveAs("%s/cjetpt_genvsreco_full_nonprompt.eps" % (self.d_resultsallpdata))
        
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff_list.append(feeddown_input_file.Get("hjetpt_fracdiff_nonprompt"+suffix))
            heff_pr_list.append(file_eff.Get("eff_mult%d" % ibin2))
            heff_fd_list.append(file_eff.Get("eff_fd_mult%d" % ibin2))
            
            ceff = TCanvas('ceff '+suffix, 'prompt and non-prompt efficiencies'+suffix)
            peff = TPad('peff'+suffix, 'prompt and non-prompt efficiencies',0.0,0.001,1.0,1.0)
            setup_pad(peff)
            ceff.SetCanvasSize(1900, 1500)
            ceff.SetWindowSize(500, 500)
            leg_eff = TLegend(.65, .5, .8, .7, "")
            setup_legend(leg_eff)
            setup_histogram(heff_pr_list[ibin2],2)
            leg_eff.AddEntry(heff_pr_list[ibin2],"prompt","LEP")
            heff_pr_list[ibin2].SetXTitle("p_{T,Lc}")
            heff_pr_list[ibin2].SetYTitle("Efficiency")
            heff_pr_list[ibin2].Draw()
            setup_histogram(heff_fd_list[ibin2],4)
            leg_eff.AddEntry(heff_fd_list[ibin2],"non-prompt","LEP")
            heff_fd_list[ibin2].Draw("same")
            leg_eff.Draw("same")
            latex = TLatex(0.6,0.3,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            ceff.SaveAs("%s/ceff_prompt_nonprompt_%s.eps" % (self.d_resultsallpdata, suffix))

        cjetpt_fracdiff = TCanvas('cjetpt_fracdiff ', 'non-prompt jetpt response fractional differences')
        pjetpt_fracdiff = TPad('pjetpt_fracdiff', 'non-prompt jetpt response fractional differences',0.0,0.001,1.0,1.0)
        setup_pad(pjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        pjetpt_fracdiff.SetLogy()
        cjetpt_fracdiff.SetCanvasSize(1900, 1500)
        cjetpt_fracdiff.SetWindowSize(500, 500)
        leg_jetpt_fracdiff = TLegend(.65, .5, .8, .8, "p_{T,jet}^{gen}")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            setup_histogram(hjetpt_fracdiff_list[ibin2],ibin2+1)
            leg_jetpt_fracdiff.AddEntry(hjetpt_fracdiff_list[ibin2],"%d-%d GeV/c" %(self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2]),"LEP")
            if ibin2 ==0 :
                hjetpt_fracdiff_list[ibin2].SetXTitle("(p_{T,jet}^{reco}-p_{T,jet}^{gen})/p_{T,jet}^{gen}")
                hjetpt_fracdiff_list[ibin2].GetYaxis().SetRangeUser(0.001,hjetpt_fracdiff_list[ibin2].GetMaximum()*3)
            hjetpt_fracdiff_list[ibin2].Draw("same")
        leg_jetpt_fracdiff.Draw("same")
        cjetpt_fracdiff.SaveAs("%s/cjetpt_fracdiff_nonprompt.eps" % (self.d_resultsallpdata))
            
        for ibinshape in range(self.p_nbinshape_gen):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_list.append(feeddown_input_file.Get("hz_fracdiff_nonprompt"+suffix))

        cz_fracdiff = TCanvas('cz_fracdiff ', 'non-prompt z response fractional differences')
        pz_fracdiff = TPad('pz_fracdiff', 'non-prompt z response fractional differences',0.0,0.001,1.0,1.0)
        setup_pad(pz_fracdiff)
        cz_fracdiff.SetLogy()
        pz_fracdiff.SetLogy()
        cz_fracdiff.SetCanvasSize(1900, 1500)
        cz_fracdiff.SetWindowSize(500, 500)
        leg_z_fracdiff = TLegend(.2, .5, .4, .85, "z")
        setup_legend(leg_z_fracdiff)
        for ibinshape in range(self.p_nbinshape_gen):
            setup_histogram(hz_fracdiff_list[ibinshape],ibinshape+1)
            leg_z_fracdiff.AddEntry(hz_fracdiff_list[ibinshape],"%.4f-%.4f" %(self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape]),"LEP")
            if ibin2==0:
                hz_fracdiff_list[ibin2].SetXTitle("(z}^{reco}-z^{gen})/z^{gen}")
                hz_fracdiff_list[ibin2].GetYaxis().SetRangeUser(0.001,hz_fracdiff_list[ibin2].GetMaximum()*3)
            hz_fracdiff_list[ibinshape].Draw("same")
        leg_z_fracdiff.Draw("same")
        cz_fracdiff.SaveAs("%s/cz_fracdiff_nonprompt.eps" % (self.d_resultsallpdata))
        
            
        for ipt in range(len(self.lpt_finbinmin)):
            bin_id = self.bin_matching[ipt]
            suffix = "%s%d_%d_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id])
            input_data.GetZaxis().SetRange(ipt+1,ipt+1)
            input_data_zvsjetpt_list.append(input_data.Project3D("input_data_zvsjetpt"+suffix+"_yxe"))
            for ibin2 in range(self.p_nbin2_gen):
                for ibinshape in range(self.p_nbinshape_gen):
                    if(heff_pr_list[ibin2].GetBinContent(ipt+1)==0 or heff_fd_list[ibin2].GetBinContent(ipt+1)==0):
                        input_data_zvsjetpt_list[ipt].SetBinContent(ibinshape+1,ibin2+1,0.0)
                    else:
                        input_data_zvsjetpt_list[ipt].SetBinContent(ibinshape+1,ibin2+1,input_data_zvsjetpt_list[ipt].GetBinContent(ibinshape+1,ibin2+1)*(heff_fd_list[ibin2].GetBinContent(ipt+1)/heff_pr_list[ibin2].GetBinContent(ipt+1)))
            if ipt==0:
                input_data_scaled = input_data_zvsjetpt_list[ipt].Clone("input_data_scaled")
            else:
                input_data_scaled.Add(input_data_zvsjetpt_list[ipt])
        input_data_scaled.Multiply(hzvsjetpt_gen_eff)
        input_data_scaled.Scale(self.p_nevents*self.branching_ratio/self.xsection_inel)
        folded = folding(input_data_scaled, response_matrix, output_template)
        folded.Sumw2()
        folded.Divide(hzvsjetpt_reco_eff)

        folded_z_list=[]
        input_data_scaled_z_list=[]
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                             (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])

            folded_z_list.append(folded.ProjectionX("folded_z_nonprompt_"+suffix,ibin2+1,ibin2+1,"e"))
            input_data_scaled_z_list.append(input_data_scaled.ProjectionX("Powheg_scaled_nonprompt_"+suffix,input_data_scaled.GetYaxis().FindBin(self.lvar2_binmin_gen[ibin2]),input_data_scaled.GetYaxis().FindBin(self.lvar2_binmin_gen[ibin2]),"e"))        
            c_fd_fold = TCanvas('c_fd_fold '+suffix, 'Powheg and folded'+suffix)
            p_fd_fold = TPad('p_fd_fold'+suffix, 'Powheg and folded'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(p_fd_fold)
            c_fd_fold.SetCanvasSize(1900, 1500)
            c_fd_fold.SetWindowSize(500, 500)
            leg_fd_fold = TLegend(.2, .75, .4, .85, "")
            setup_legend(leg_fd_fold)
            setup_histogram(input_data_scaled_z_list[ibin2],2)
            leg_fd_fold.AddEntry(input_data_scaled_z_list[ibin2],"Powheg eff corrected","LEP")
            input_data_scaled_z_list[ibin2].GetYaxis().SetRangeUser(0.0,input_data_scaled_z_list[ibin2].GetMaximum()*1.5)
            input_data_scaled_z_list[ibin2].SetXTitle("z")
            input_data_scaled_z_list[ibin2].Draw()
            setup_histogram(folded_z_list[ibin2],4)
            leg_fd_fold.AddEntry(folded_z_list[ibin2],"folded","LEP")
            folded_z_list[ibin2].Draw("same")
            leg_fd_fold.Draw("same")
            latex = TLatex(0.4,0.25,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_fd_fold.SaveAs("%s/cfolded_Powheg_%s.eps" % (self.d_resultsallpdata, suffix))
        fileouts.cd()
        sideband_input_data_subtracted = sideband_input_data.Clone("sideband_input_data_subtracted")
        sideband_input_data_subtracted.Add(folded,-1)
        sideband_input_data_subtracted.Write()
        
        sideband_input_data_z=[]
        sideband_input_data_subtracted_z=[]
        
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            sideband_input_data_z.append(sideband_input_data.ProjectionX("sideband_input_data_z"+suffix,ibin2+1,ibin2+1,"e"))
            sideband_input_data_subtracted_z.append(sideband_input_data_subtracted.ProjectionX("sideband_input_data_subtracted_z"+suffix,ibin2+1,ibin2+1,"e"))
            cfeeddown = TCanvas('cfeeddown'+suffix, 'cfeeddown'+suffix)
            pfeeddown = TPad('pfeeddown'+suffix, 'cfeeddown'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pfeeddown)
            if ibin2 is not 2:
                cfeeddown.SetLogy()
                pfeeddown.SetLogy()
            cfeeddown.SetCanvasSize(1900, 1500)
            cfeeddown.SetWindowSize(500, 500)
            legmin =.2
            legmax =.4
            if ibin2 == 2:
                legmin =.7
                legmax =.85
            leg_feeddown = TLegend(.2, legmin, .4, legmax, "")
            setup_legend(leg_feeddown)
            setup_histogram(sideband_input_data_z[ibin2],2)
            leg_feeddown.AddEntry(sideband_input_data_z[ibin2],"prompt+non-prompt","LEP")
            sideband_input_data_z[ibin2].GetYaxis().SetRangeUser(0.1,sideband_input_data_z[ibin2].GetMaximum()*3)
            sideband_input_data_z[ibin2].SetXTitle("z")
            sideband_input_data_z[ibin2].Draw()
            setup_histogram(sideband_input_data_subtracted_z[ibin2],3)
            leg_feeddown.AddEntry(sideband_input_data_subtracted_z[ibin2],"subtracted (prompt)","LEP")
            sideband_input_data_subtracted_z[ibin2].Draw("same")
            setup_histogram(folded_z_list[ibin2],4)
            leg_feeddown.AddEntry(folded_z_list[ibin2],"non-prompt powheg","LEP")
            folded_z_list[ibin2].Draw("same")
            leg_feeddown.Draw("same")
            if ibin2 is not 2:
                latex = TLatex(0.6,0.3,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            else:
                latex = TLatex(0.6,0.75,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cfeeddown.SaveAs("%s/cfeeddown_subtraction_%s.eps" % (self.d_resultsallpdata, suffix))
            
            feeddown_fraction = folded_z_list[ibin2].Clone("feeddown_fraction"+suffix)
            feeddown_fraction_denominator = folded_z_list[ibin2].Clone("feeddown_denominator"+suffix)
            feeddown_fraction_denominator.Add(sideband_input_data_z[ibin2])
            feeddown_fraction.Divide(feeddown_fraction_denominator)
            
            cfeeddown_fraction = TCanvas('cfeeddown_fraction'+suffix, 'cfeeddown_fraction'+suffix)
            pfeeddown_fraction = TPad('pfeeddown_fraction'+suffix, 'cfeeddown_fraction'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pfeeddown_fraction)
            if ibin2 is not 2:
                cfeeddown_fraction.SetLogy()
                pfeeddown_fraction.SetLogy()
            cfeeddown_fraction.SetCanvasSize(1900, 1500)
            cfeeddown_fraction.SetWindowSize(500, 500)
            setup_histogram(feeddown_fraction,4)
            feeddown_fraction.SetXTitle("z")
            feeddown_fraction.SetYTitle("b-feeddown fraction")
            feeddown_fraction.Draw()
            latex = TLatex(0.6,0.75,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cfeeddown_fraction.SaveAs("%s/cfeeddown_fraction_%s.eps" % (self.d_resultsallpdata, suffix))
            
            

        cfeeddown_output = TCanvas('cfeeddown_output', 'cfeeddown_output')
        pfeeddown_output = TPad('pfeeddown_output', 'pfeeddown_output',0.0,0.001,1.0,1.0)
        setup_pad(pfeeddown_output)
        cfeeddown_output.SetCanvasSize(1900, 1500)
        cfeeddown_output.SetWindowSize(500, 500)
        setup_histogram(sideband_input_data_subtracted)
        sideband_input_data_subtracted.Draw("text")
        cfeeddown_output.SaveAs("%s/cfeeddown_output.eps" % (self.d_resultsallpdata))
        print("end of folding")
    # pylint: disable=too-many-locals
    def side_band_sub(self):
        self.loadstyle()
        lfile = TFile.Open(self.n_filemass)
        func_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                            None, [self.case, self.typean])
        func_file = TFile.Open(func_filename, "READ")
        print(func_filename)
        eff_file = TFile.Open("%s/efficiencies%s%s.root" % \
                              (self.d_resultsallpmc, self.case, self.typean))
        fileouts = TFile.Open("%s/sideband_sub%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "RECREATE")
        fileouts.cd()

        zbin_reco=[]
        nzbin_reco=self.p_nbinshape_reco
        zbin_reco =self.varshaperanges_reco
        zbinarray_reco=array('d',zbin_reco)

        jetptbin_reco =[]
        njetptbin_reco=self.p_nbin2_reco
        jetptbin_reco = self.var2ranges_reco
        jetptbinarray_reco=array('d',jetptbin_reco)

        hzvsjetpt = TH2F("hzvsjetpt","",nzbin_reco, zbinarray_reco, njetptbin_reco, jetptbinarray_reco)
        hzvsjetpt.Sumw2()
        for imult in range(self.p_nbin2_reco):
            heff = eff_file.Get("eff_mult%d" % imult)
            hz = None
            first_fit = 0
            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[imult], self.lvar2_binmax_reco[imult])
                hzvsmass = lfile.Get("hzvsmass" + suffix)
                load_dir = func_file.GetDirectory(suffix)
                mass_fitter = load_dir.Get("fitter")
                mean = mass_fitter.GetMean()
                sigma = mass_fitter.GetSigma()
                binmasslow2sig = hzvsmass.GetXaxis().FindBin(mean - self.signal_sigma*sigma)
                masslow2sig = mean - self.signal_sigma*sigma
                binmasshigh2sig = hzvsmass.GetXaxis().FindBin(mean + self.signal_sigma*sigma)
                masshigh2sig = mean + self.signal_sigma*sigma
                binmasslow4sig = hzvsmass.GetXaxis().FindBin(mean - self.sideband_sigma_1_left*sigma)
                masslow4sig = mean - self.sideband_sigma_1_left*sigma
                binmasshigh4sig = hzvsmass.GetXaxis().FindBin(mean + self.sideband_sigma_1_right*sigma)
                masshigh4sig = mean + self.sideband_sigma_1_right*sigma
                binmasslow9sig = hzvsmass.GetXaxis().FindBin(mean - self.sideband_sigma_2_left*sigma)
                masslow9sig = mean - self.sideband_sigma_2_left*sigma
                binmasshigh9sig = hzvsmass.GetXaxis().FindBin(mean + self.sideband_sigma_2_right*sigma)
                masshigh9sig = mean + self.sideband_sigma_2_right*sigma

                hzsig = hzvsmass.ProjectionY("hzsig" + suffix, \
                             binmasslow2sig, binmasshigh2sig, "e")
                hzbkgleft = hzvsmass.ProjectionY("hzbkgleft" + suffix, \
                             binmasslow9sig, binmasslow4sig, "e")
                hzbkgright = hzvsmass.ProjectionY("hzbkgright" + suffix, \
                             binmasshigh4sig, binmasshigh9sig, "e")
                hzbkg = hzbkgleft.Clone("hzbkg" + suffix)
                if self.sidebandleftonly is False :
                    hzbkg.Add(hzbkgright)
                hzbkg_scaled = hzbkg.Clone("hzbkg_scaled" + suffix)
                bkg_fit = mass_fitter.GetBackgroundRecalcFunc()

                area_scale_denominator = -1
                print("---------", bkg_fit)
                if not bkg_fit:
                    continue
                area_scale_denominator = bkg_fit.Integral(masslow9sig, masslow4sig) + \
                bkg_fit.Integral(masshigh4sig, masshigh9sig)
                area_scale = bkg_fit.Integral(masslow2sig, masshigh2sig)/area_scale_denominator
                hzsub = hzsig.Clone("hzsub" + suffix)
                hzsub.Add(hzbkg, -1*area_scale)
                hzsub_noteffscaled = hzsub.Clone("hzsub_noteffscaled" + suffix)
                hzbkg_scaled.Scale(area_scale)
                eff = heff.GetBinContent(ipt+1)
                if eff > 0.0 :
                    hzsub.Scale(1.0/(eff*self.sigma_scale))
                if first_fit == 0:
                    hz = hzsub.Clone("hz")
                    first_fit=1
                else:
                    hz.Add(hzsub)
                hzsig.Write("hzsig" + suffix)
                hzbkgleft.Write("hzbkgleft" + suffix)
                hzbkgright.Write("hzbkgright" + suffix)
                hzbkg.Write("hzbkg" + suffix)
                hzsub.Write("hzsub" + suffix)

                csubz = TCanvas('csubz' + suffix, 'The Side-Band Sub Canvas'+suffix)
                psubz = TPad('psubz', 'psubz',0.0,0.001,1.0,1.0)
                setup_pad(psubz)
                csubz.SetCanvasSize(1900, 1500)
                csubz.SetWindowSize(500, 500)
                setup_histogram(hzsub,4)
                hzsub.GetYaxis().SetRangeUser(hzsub.GetMinimum(),hzsub.GetMaximum()*1.2)
                hzsub.SetXTitle("z")
                hzsub.Draw()
                latex = TLatex(0.6,0.85,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_reco[imult],self.lvar2_binmax_reco[imult]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.8,"%.2f < p_{T,Lc} < %.2f GeV/c" % (self.lpt_finbinmin[ipt],self.lpt_finbinmax[ipt]))
                draw_latex(latex2)
                csubz.SaveAs("%s/side_band_subtracted%s%s_%s.eps" % \
                             (self.d_resultsallpdata, self.case, self.typean, suffix))



                csigbkgsubz = TCanvas('csigbkgsubz' + suffix, 'The Side-Band Canvas'+suffix)
                psigbkgsubz = TPad('psigbkgsubz'+suffix, 'psigbkgsubz'+suffix,0.0,0.001,1.0,1.0)
                setup_pad(psigbkgsubz)
                csigbkgsubz.SetCanvasSize(1900, 1500)
                csigbkgsubz.SetWindowSize(500, 500)
                legsigbkgsubz = TLegend(.2, .65, .35, .85)
                setup_legend(legsigbkgsubz)
                setup_histogram(hzsig,2)
                legsigbkgsubz.AddEntry(hzsig, "signal", "LEP")
                hzsig.GetYaxis().SetRangeUser(0.0, max(hzsig.GetBinContent(hzsig.GetMaximumBin()), \
                    hzbkg_scaled.GetBinContent(hzbkg_scaled.GetMaximumBin()), \
                    hzsub_noteffscaled.GetBinContent(hzsub_noteffscaled.GetMaximumBin()))*1.4)
                hzsig.SetXTitle("z")
                hzsig.Draw()
                setup_histogram(hzbkg_scaled,3)
                legsigbkgsubz.AddEntry(hzbkg_scaled, "side-band", "LEP")
                hzbkg_scaled.Draw("same")
                setup_histogram(hzsub_noteffscaled,4)
                legsigbkgsubz.AddEntry(hzsub_noteffscaled, "subtracted", "LEP")
                hzsub_noteffscaled.Draw("same")
                legsigbkgsubz.Draw("same")
                latex = TLatex(0.6,0.85,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_reco[imult],self.lvar2_binmax_reco[imult]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.8,"%.2f < p_{T,Lc} < %.2f GeV/c" % (self.lpt_finbinmin[ipt],self.lpt_finbinmax[ipt]))
                draw_latex(latex2)
                csigbkgsubz.SaveAs("%s/side_band_%s%s_%s.eps" % \
                             (self.d_resultsallpdata, self.case, self.typean, suffix))
                             
            cz = TCanvas('cz' + suffix, 'The Efficiency Corrected Signal Yield Canvas'+suffix)
            pz = TPad('pz'+suffix, 'The Efficiency Corrected Signal Yield Canvas'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pz)
            cz.SetCanvasSize(1900, 1500)
            cz.SetWindowSize(500, 500)
            setup_histogram(hz,4)
            hz.SetXTitle("z")
            hz.Draw()
            latex = TLatex(0.6,0.85,"%.2f < p_{T,jet} < %.2f GeV/c" % (self.lvar2_binmin_reco[imult],self.lvar2_binmax_reco[imult]))
            draw_latex(latex)
            cz.SaveAs("%s/efficiencycorrected_fullsub%s%s_%s_%.2f_%.2f.eps" % \
                      (self.d_resultsallpdata, self.case, self.typean, self.v_var2_binning, \
                       self.lvar2_binmin[imult], self.lvar2_binmax[imult]))
                       
            for zbins in range(nzbin_reco):
                hzvsjetpt.SetBinContent(zbins+1,imult+1,hz.GetBinContent(zbins+1))                                                                                                                                                                                                                                                                                     
                hzvsjetpt.SetBinError(zbins+1,imult+1,hz.GetBinError(zbins+1)) 
            #    if hz.GetBinContent(zbins+1) >= 0.0 :
             #       hzvsjetpt.SetBinContent(zbins+1,imult+1,hz.GetBinContent(zbins+1))
              #      hzvsjetpt.SetBinError(zbins+1,imult+1,hz.GetBinError(zbins+1))
               # else:
                #    hzvsjetpt.SetBinContent(zbins+1,imult+1,0.0)
                 #   hzvsjetpt.SetBinError(zbins+1,imult+1,0.0)
            hz.Scale(1.0/hz.Integral(1,-1))
            hz.Write("hz" + suffix)


        hzvsjetpt.Write("hzvsjetpt")
        czvsjetpt = TCanvas('czvsjetpt', '2D input to unfolding')
        pzvsjetpt = TPad('pzvsjetpt', '2D input to unfolding',0.0,0.001,1.0,1.0)
        setup_pad(pzvsjetpt)
        czvsjetpt.SetCanvasSize(1900, 1500)
        czvsjetpt.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt)
        hzvsjetpt.SetXTitle("z")
        hzvsjetpt.SetYTitle("p_{T,jet}")
        hzvsjetpt.Draw("text")
        czvsjetpt.SaveAs("%s/czvsjetpt.eps" % self.d_resultsallpdata)
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
                HFPtSpectrum2(filecrossmb, fileouteff, namehistoeffprompt, namehistoefffeed, \
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
    def unfolding(self):
        print("unfolding starts")
        lfile = TFile.Open(self.n_filemass,"update")
        fileouts = TFile.Open("%s/unfolding_results%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "recreate")

        unfolding_input_data_file = TFile.Open("%s/feeddown%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean))
        unfolding_input_file = TFile.Open(self.n_fileff)
        response_matrix = unfolding_input_file.Get("response_matrix")
        hzvsjetpt_reco_nocuts = unfolding_input_file.Get("hzvsjetpt_reco_nocuts")
        hzvsjetpt_reco_eff = unfolding_input_file.Get("hzvsjetpt_reco_cuts")
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)
        input_data = unfolding_input_data_file.Get("sideband_input_data_subtracted")
        input_data.Multiply(hzvsjetpt_reco_eff)
        input_data_z=[]
        input_mc_gen = unfolding_input_file.Get("hzvsjetpt_gen_unmatched")
        input_mc_gen_z=[]
        mc_reco_matched = unfolding_input_file.Get("hzvsjetpt_reco")
        mc_reco_matched_z=[]
        mc_gen_matched = unfolding_input_file.Get("hzvsjetpt_gen")
        mc_gen_matched_z=[]
        mc_reco_gen_matched_z_ratio=[]
        hjetpt_fracdiff_list=[]
        hz_fracdiff_list=[]
        kinematic_eff=[]
        hz_gen_nocuts=[]

        hz_genvsreco_list=[]
        hjetpt_genvsreco_list=[]

        input_data_jetpt=input_data.ProjectionY("input_data_jetpt",1,self.p_nbinshape_reco,"e")

        input_powheg_file = TFile.Open(self.powheg_path_prompt)
        input_powheg = input_powheg_file.Get("fh2_powheg_prompt")
        input_powheg_xsection = input_powheg_file.Get("fh2_powheg_prompt_xsection")
        input_powheg_file_sys = []
        input_powheg_sys=[]
        input_powheg_xsection_sys=[]
        for i_powheg in range(len(self.powheg_prompt_variations)):
            input_powheg_file_sys.append(TFile.Open("%s%s.root" % (self.powheg_prompt_variations_path,self.powheg_prompt_variations[i_powheg])))
            input_powheg_sys.append(input_powheg_file_sys[i_powheg].Get("fh2_powheg_prompt"))
            input_powheg_xsection_sys.append(input_powheg_file_sys[i_powheg].Get("fh2_powheg_prompt_xsection"))
        input_powheg_z=[]
        input_powheg_xsection_z=[]
        input_powheg_sys_z=[]
        input_powheg_xsection_sys_z=[]
        tg_powheg=[]
        tg_powheg_xsection=[]


        
                
        
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            input_data_z.append(input_data.ProjectionX("input_data_z"+suffix,ibin2+1,ibin2+1,"e"))
            mc_reco_matched_z.append(mc_reco_matched.ProjectionX("mc_reco_matched_z"+suffix,ibin2+1,ibin2+1,"e"))
            mc_reco_matched_z[ibin2].Scale(1.0/mc_reco_matched_z[ibin2].Integral(1,-1))
            mc_gen_matched_z.append(mc_gen_matched.ProjectionX("mc_det_matched_z"+suffix,mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]),mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]),"e"))
            mc_gen_matched_z[ibin2].Scale(1.0/mc_gen_matched_z[ibin2].Integral(1,-1))
            mc_reco_gen_matched_z_ratio.append(mc_reco_matched_z[ibin2].Clone("input_mc_reco_gen_matched_z_ratio"+suffix))
            mc_reco_gen_matched_z_ratio[ibin2].Divide(mc_gen_matched_z[ibin2])

            c_mc_reco_gen_matched_z_ratio = TCanvas('c_mc_reco_gen_matched_z_ratio '+suffix, 'Reco/Gen Ratio')
            p_mc_reco_gen_matched_z_ratio = TPad('p_mc_reco_gen_matched_z_ratio'+suffix, 'c_mc_reco_gen_matched_z_ratio'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(p_mc_reco_gen_matched_z_ratio)
            c_mc_reco_gen_matched_z_ratio.SetCanvasSize(1900, 1500)
            c_mc_reco_gen_matched_z_ratio.SetWindowSize(500, 500)
            setup_histogram(mc_reco_gen_matched_z_ratio[ibin2])
            mc_reco_gen_matched_z_ratio[ibin2].SetXTitle("z")
            mc_reco_gen_matched_z_ratio[ibin2].SetYTitle("reconstructed/generated")
            mc_reco_gen_matched_z_ratio[ibin2].Draw("same")
            latex = TLatex(0.2,0.85,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_mc_reco_gen_matched_z_ratio.SaveAs("%s/mc_reco_gen_matched_z_ratio_%s.eps" % (self.d_resultsallpdata, suffix))

            c_mc_reco_gen_matched_z = TCanvas('c_mc_reco_gen_matched_z '+suffix, 'Reco vs Gen')
            p_mc_reco_gen_matched_z = TPad('p_mc_reco_gen_matched_z'+suffix, 'Reco vs Gen'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(p_mc_reco_gen_matched_z)
            c_mc_reco_gen_matched_z.SetCanvasSize(1900, 1500)
            c_mc_reco_gen_matched_z.SetWindowSize(500, 500)
            leg_mc_reco_gen_matched_z = TLegend(.75, .7, .9, .85, "")
            setup_legend(leg_mc_reco_gen_matched_z)
            setup_histogram(mc_reco_matched_z[ibin2],2)
            leg_mc_reco_gen_matched_z.AddEntry(mc_reco_matched_z[ibin2],"reco","LEP")
            mc_reco_matched_z[ibin2].SetXTitle("z")
            mc_reco_matched_z[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.01)
            mc_reco_matched_z[ibin2].GetYaxis().SetRangeUser(0.0,mc_reco_matched_z[ibin2].GetMaximum()*1.5)
            mc_reco_matched_z[ibin2].Draw()
            setup_histogram(mc_gen_matched_z[ibin2],4)
            leg_mc_reco_gen_matched_z.AddEntry(mc_gen_matched_z[ibin2],"gen","LEP")
            mc_gen_matched_z[ibin2].Draw("same")
            leg_mc_reco_gen_matched_z.Draw("same")
            latex = TLatex(0.2,0.85,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_mc_reco_gen_matched_z.SaveAs("%s/mc_reco_gen_matched_z_%s.eps" % (self.d_resultsallpdata, suffix))

            hz_genvsreco_list.append(unfolding_input_file.Get("hz_genvsreco"+suffix))

            cz_genvsreco = TCanvas('cz_genvsreco_'+suffix, 'response matrix 2D projection')
            pz_genvsreco = TPad('pz_genvsreco'+suffix, 'response matrix 2D projection'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pz_genvsreco)
            cz_genvsreco.SetLogz()
            pz_genvsreco.SetLogz()
            cz_genvsreco.SetCanvasSize(1900, 1500)
            cz_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hz_genvsreco_list[ibin2])
            hz_genvsreco_list[ibin2].SetXTitle("z^{gen}")
            hz_genvsreco_list[ibin2].SetYTitle("z^{reco}")
            hz_genvsreco_list[ibin2].Draw("colz")
            latex = TLatex(0.2,0.85,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cz_genvsreco.SaveAs("%s/cz_genvsreco_%s.eps" % (self.d_resultsallpdata,suffix))

        for ibinshape in range(self.p_nbinshape_reco):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco_list.append(unfolding_input_file.Get("hjetpt_genvsreco"+suffix))

            cjetpt_genvsreco = TCanvas('cjetpt_genvsreco'+suffix, 'response matrix 2D projection'+suffix)
            pjetpt_genvsreco = TPad('pjetpt_genvsreco'+suffix, 'response matrix 2D projection'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pjetpt_genvsreco)
            cjetpt_genvsreco.SetLogz()
            pjetpt_genvsreco.SetLogz()
            cjetpt_genvsreco.SetCanvasSize(1900, 1500)
            cjetpt_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hjetpt_genvsreco_list[ibinshape])
            hjetpt_genvsreco_list[ibinshape].SetXTitle("z^{gen}")
            hjetpt_genvsreco_list[ibinshape].SetYTitle("z^{reco}")
            hjetpt_genvsreco_list[ibinshape].Draw("colz")
            latex = TLatex(0.2,0.85,'%.2f < z < %.2f' % (self.lvarshape_binmin_reco[ibinshape],self.lvarshape_binmax_reco[ibinshape]))
            draw_latex(latex)
            cjetpt_genvsreco.SaveAs("%s/cjetpt_genvsreco_%s.eps" % (self.d_resultsallpdata,suffix))

        hz_genvsreco_full=unfolding_input_file.Get("hz_genvsreco_full")
        hjetpt_genvsreco_full=unfolding_input_file.Get("hjetpt_genvsreco_full")

        cz_genvsreco_full = TCanvas('cz_genvsreco_full', 'response matrix 2D projection')
        pz_genvsreco_full = TPad('pz_genvsreco_full', 'response matrix 2D projection',0.0,0.001,1.0,1.0)
        setup_pad(pz_genvsreco_full)
        cz_genvsreco_full.SetLogz()
        pz_genvsreco_full.SetLogz()
        cz_genvsreco_full.SetCanvasSize(1900, 1500)
        cz_genvsreco_full.SetWindowSize(500, 500)
        setup_histogram(hz_genvsreco_full)
        hz_genvsreco_full.SetXTitle("z^{gen}")
        hz_genvsreco_full.SetYTitle("z^{reco}")
        hz_genvsreco_full.Draw("colz")
        cz_genvsreco_full.SaveAs("%s/cz_genvsreco_full.eps" % (self.d_resultsallpdata))

        cjetpt_genvsreco_full = TCanvas('cjetpt_genvsreco_full', 'response matrix 2D projection')
        pjetpt_genvsreco_full = TPad('pjetpt_genvsreco_full', 'response matrix 2D projection',0.0,0.001,1.0,1.0)
        setup_pad(pjetpt_genvsreco_full)
        cjetpt_genvsreco_full.SetLogz()
        pjetpt_genvsreco_full.SetLogz()
        cjetpt_genvsreco_full.SetCanvasSize(1900, 1500)
        cjetpt_genvsreco_full.SetWindowSize(500, 500)
        setup_histogram(hjetpt_genvsreco_full)
        hjetpt_genvsreco_full.SetXTitle("p_{T,jet}^{gen}")
        hjetpt_genvsreco_full.SetYTitle("p_{T,jet}^{reco}")
        hjetpt_genvsreco_full.Draw("colz")
        cjetpt_genvsreco_full.SaveAs("%s/cjetpt_genvsreco_full.eps" % (self.d_resultsallpdata))
            
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff_list.append(unfolding_input_file.Get("hjetpt_fracdiff_prompt"+suffix))
            kinematic_eff.append(unfolding_input_file.Get("hz_gen_cuts"+suffix))
            hz_gen_nocuts.append(unfolding_input_file.Get("hz_gen_nocuts"+suffix))
            kinematic_eff[ibin2].Divide(hz_gen_nocuts[ibin2])
            ckinematic_eff = TCanvas("ckinematic_eff " + suffix, "Kinematic Eff" + suffix)
            pkinematic_eff = TPad('pkinematic_eff' + suffix, "Kinematic Eff" + suffix,0.0,0.001,1.0,1.0)
            setup_pad(pkinematic_eff)
            ckinematic_eff.SetCanvasSize(1900, 1500)
            ckinematic_eff.SetWindowSize(500, 500)
            setup_histogram(kinematic_eff[ibin2],4)
            kinematic_eff[ibin2].SetXTitle("z")
            kinematic_eff[ibin2].SetYTitle("kinematic eff")
            kinematic_eff[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01,self.lvarshape_binmax_reco[-1]-0.001)
            kinematic_eff[ibin2].Draw()
            latex = TLatex(0.6,0.25,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            ckinematic_eff.SaveAs("%s/cgen_kineeff_%s.eps" % (self.d_resultsallpdata, suffix))
            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_mc_gen_z[ibin2].Scale(1.0/input_mc_gen_z[ibin2].Integral(input_mc_gen_z[ibin2].FindBin(self.lvarshape_binmin_reco[0]),input_mc_gen_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])))
            input_powheg_z.append(input_powheg.ProjectionX("input_powheg_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_z[ibin2].Scale(1.0/input_powheg_z[ibin2].Integral(input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[0]),input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])))
            input_powheg_xsection_z.append(input_powheg_xsection.ProjectionX("input_powheg_xsection_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_sys_z_iter=[]
            input_powheg_xsection_sys_z_iter=[]
            for i_powheg in range(len(self.powheg_prompt_variations)):
                input_powheg_sys_z_iter.append(input_powheg_sys[i_powheg].ProjectionX("input_powheg_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix,ibin2+1,ibin2+1,"e"))
                input_powheg_sys_z_iter[i_powheg].Scale(1.0/input_powheg_sys_z_iter[i_powheg].Integral(input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[0]),input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[-1])))
                input_powheg_xsection_sys_z_iter.append(input_powheg_xsection_sys[i_powheg].ProjectionX("input_powheg_xsection_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_sys_z.append(input_powheg_sys_z_iter)
            input_powheg_xsection_sys_z.append(input_powheg_xsection_sys_z_iter)
            tg_powheg.append(tg_sys(input_powheg_z[ibin2], input_powheg_sys_z[ibin2]))
            tg_powheg_xsection.append(tg_sys(input_powheg_xsection_z[ibin2], input_powheg_xsection_sys_z[ibin2]))
            
        
        kinematic_eff_jetpt = unfolding_input_file.Get("hjetpt_gen_cuts")
        hjetpt_gen_nocuts=unfolding_input_file.Get("hjetpt_gen_nocuts")
        kinematic_eff_jetpt.Divide(hjetpt_gen_nocuts)
        ckinematic_eff_jetpt = TCanvas("ckinematic_eff_jetpt", "Kinematic Eff_jetpt")
        pkinematic_eff_jetpt = TPad('pkinematic_eff_jetpt', "Kinematic Eff_jetpt",0.0,0.001,1.0,1.0)
        setup_pad(pkinematic_eff_jetpt)
        ckinematic_eff_jetpt.SetCanvasSize(1900, 1500)
        ckinematic_eff_jetpt.SetWindowSize(500, 500)
        setup_histogram(kinematic_eff_jetpt)
        kinematic_eff_jetpt.SetXTitle("p_{T,jet}")
        kinematic_eff_jetpt.SetYTitle("kinematic eff")
        kinematic_eff_jetpt.GetXaxis().SetRangeUser(self.lvar2_binmin_reco[0]+0.01,self.lvar2_binmax_reco[-1]-0.001)
        kinematic_eff_jetpt.Draw()
        latex = TLatex(0.6,0.25,'%.2f < z < %.2f' % (self.lvarshape_binmin_reco[0],self.lvarshape_binmax_reco[-1]))
        draw_latex(latex)
        ckinematic_eff_jetpt.SaveAs("%s/cgen_kineeff_jetpt.eps" % (self.d_resultsallpdata))
        
        for ibinshape in range(self.p_nbinshape_gen):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_list.append(unfolding_input_file.Get("hz_fracdiff_prompt"+suffix))

        cjetpt_fracdiff = TCanvas('cjetpt_fracdiff ', 'prompt jetpt response fractional differences')
        pjetpt_fracdiff = TPad('pjetpt_fracdiff', "prompt jetpt response fractional differences",0.0,0.001,1.0,1.0)
        setup_pad(pjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        pjetpt_fracdiff.SetLogy()
        cjetpt_fracdiff.SetCanvasSize(1900, 1500)
        cjetpt_fracdiff.SetWindowSize(500, 500)
        leg_jetpt_fracdiff = TLegend(.65, .6, .8, .8, "p_{T,jet} GeV/c")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            setup_histogram(hjetpt_fracdiff_list[ibin2],ibin2+1)
            leg_jetpt_fracdiff.AddEntry(hjetpt_fracdiff_list[ibin2],"%.2f-%.2f" %(self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2]),"LEP")
            if ibin2==0:
                hjetpt_fracdiff_list[ibin2].GetYaxis().SetRangeUser(0.001,hjetpt_fracdiff_list[ibin2].GetMaximum()*2)
                hjetpt_fracdiff_list[ibin2].SetXTitle("(p_{T,jet}^{reco}-p_{T,jet}^{gen})/p_{T,jet}^{gen}")
            hjetpt_fracdiff_list[ibin2].Draw("same")
        leg_jetpt_fracdiff.Draw("same")
        cjetpt_fracdiff.SaveAs("%s/cjetpt_fracdiff_prompt.eps" % (self.d_resultsallpdata))

        creco_eff = TCanvas('creco_eff ', 'reco efficiency applied to input data')
        preco_eff = TPad('preco_eff', "reco efficiency applied to input data",0.0,0.001,1.0,1.0)
        setup_pad(preco_eff)
        creco_eff.SetCanvasSize(1900, 1500)
        creco_eff.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt_reco_eff)
        hzvsjetpt_reco_eff.Draw("text")
        creco_eff.SaveAs("%s/creco_kineeff.eps" % (self.d_resultsallpdata))


        cz_fracdiff = TCanvas('cz_fracdiff ', 'prompt z response fractional differences')
        pz_fracdiff = TPad('pz_fracdiff', "prompt z response fractional differences",0.0,0.001,1.0,1.0)
        setup_pad(pz_fracdiff)
        cz_fracdiff.SetLogy()
        pz_fracdiff.SetLogy()
        cz_fracdiff.SetCanvasSize(1900, 1500)
        cz_fracdiff.SetWindowSize(500, 500)
        leg_z_fracdiff = TLegend(.2, .5, .5, .9, "z")
        setup_legend(leg_z_fracdiff)
        for ibinshape in range(self.p_nbinshape_gen):
            setup_histogram(hz_fracdiff_list[ibinshape],ibinshape+1)
            leg_z_fracdiff.AddEntry(hz_fracdiff_list[ibinshape],"%.2f-%.2f" %(self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape]),"LEP")
            if ibinshape==0:
                hz_fracdiff_list[ibinshape].GetYaxis().SetRangeUser(0.001,hz_fracdiff_list[ibinshape].GetMaximum()*2)
                hz_fracdiff_list[ibinshape].SetXTitle("(z^{reco}-z^{gen})/z^{gen}")
            hz_fracdiff_list[ibinshape].Draw("same")
        leg_z_fracdiff.Draw("same")
        cz_fracdiff.SaveAs("%s/cz_fracdiff_prompt.eps" % (self.d_resultsallpdata))
        
        fileouts.cd()
        h_dummy = TH1F("hdummy","",1,0,1.0)
        unfolded_z_scaled_list=[]
        unfolded_z_xsection_list=[]
        unfolded_jetpt_scaled_list=[]
        refolding_test_list=[]
        refolding_test_jetpt_list=[]
        for i in range(self.niter_unfolding) :
            unfolded_z_scaled_list_iter=[]
            unfolded_z_xsection_list_iter=[]
            refolding_test_list_iter=[]
            unfolding_object = RooUnfoldBayes(response_matrix, input_data, i+1)
            unfolded_zvsjetpt = unfolding_object.Hreco(2)

            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                unfolded_z = unfolded_zvsjetpt.ProjectionX("unfolded_z_"+suffix,ibin2+1,ibin2+1,"e")
                unfolded_z.Sumw2()
                unfolded_z_scaled = unfolded_z.Clone("unfolded_z_scaled_%d_%s" % (i+1,suffix)) 
                unfolded_z_scaled.Divide(kinematic_eff[ibin2])
                unfolded_z_xsection = unfolded_z_scaled.Clone("unfolded_z_xsection_%d_%s" % (i+1,suffix))
                unfolded_z_xsection.Scale((self.xsection_inel)/(self.p_nevents*self.branching_ratio))
                unfolded_z_scaled.Scale(1.0/unfolded_z_scaled.Integral(unfolded_z_scaled.FindBin(self.lvarshape_binmin_reco[0]),unfolded_z_scaled.FindBin(self.lvarshape_binmin_reco[-1])))
                unfolded_z_scaled.Write("unfolded_z_%d_%s" % (i+1,suffix))
                unfolded_z_xsection.Write("unfolded_z_xsection_%d_%s" % (i+1,suffix))
                unfolded_z_scaled_list_iter.append(unfolded_z_scaled)
                unfolded_z_xsection_list_iter.append(unfolded_z_xsection)
                cunfolded_z = TCanvas('cunfolded_z'+suffix, '1D output of unfolding'+suffix)
                punfolded_z = TPad('punfolded_z'+suffix, "1D output of unfolding"+suffix,0.0,0.001,1.0,1.0)
                setup_pad(punfolded_z)
                cunfolded_z.SetCanvasSize(1900, 1500)
                cunfolded_z.SetWindowSize(500, 500)
                setup_histogram(unfolded_z_scaled,4)
                unfolded_z_scaled.GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01,self.lvarshape_binmax_reco[-1]-0.001)
                unfolded_z_scaled.SetXTitle("z")
                unfolded_z_scaled.SetYTitle("1/N_{jets} dN/dz")
                unfolded_z_scaled.Draw()
                latex = TLatex(0.6,0.25,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
                draw_latex(latex2)
                cunfolded_z.SaveAs("%s/cunfolded_z_%d_%s.eps" % (self.d_resultsallpdata, i+1, suffix))


            unfolded_jetpt = unfolded_zvsjetpt.ProjectionY("unfolded_jetpt",1,self.p_nbinshape_gen,"e")
            unfolded_jetpt.Sumw2()
            unfolded_jetpt_scaled = unfolded_jetpt.Clone("unfolded_jetpt_scaled_%d" % (i+1))
            unfolded_jetpt_scaled.Divide(kinematic_eff_jetpt)
            unfolded_jetpt_scaled.Scale(1.0/unfolded_jetpt_scaled.Integral(unfolded_jetpt_scaled.FindBin(self.lvar2_binmin_reco[0]),unfolded_jetpt_scaled.FindBin(self.lvar2_binmin_reco[-1])))
            unfolded_jetpt_scaled.Write("unfolded_jetpt_%d" % (i+1))
            unfolded_jetpt_scaled_list.append(unfolded_jetpt_scaled)
            cunfolded_jetpt = TCanvas('cunfolded_jetpt', '1D output of unfolding')
            punfolded_jetpt = TPad('punfolded_jetpt', "1D output of unfolding",0.0,0.001,1.0,1.0)
            setup_pad(punfolded_jetpt)
            cunfolded_jetpt.SetCanvasSize(1900, 1500)
            cunfolded_jetpt.SetWindowSize(500, 500)
            setup_histogram(unfolded_jetpt_scaled,4)
            unfolded_jetpt_scaled.GetXaxis().SetRangeUser(self.lvar2_binmin_reco[0]+0.01,self.lvar2_binmax_reco[-1]-0.001)
            unfolded_jetpt_scaled.SetXTitle("p_{T,jet} GeV/c")
            unfolded_jetpt_scaled.SetYTitle("1/N_{jets} dN/dp_{T,jet}")
            unfolded_jetpt_scaled.Draw()
            latex = TLatex(0.6,0.85,'%.2f < z < %.2f' % (self.lvarshape_binmin_reco[0],self.lvarshape_binmax_reco[-1]))
            draw_latex(latex)
            latex2 = TLatex(0.6,0.8,'iteration %d' % (i+1))
            draw_latex(latex2)
            cunfolded_jetpt.SaveAs("%s/cunfolded_jetpt_%d.eps" % (self.d_resultsallpdata, i+1))

            unfolded_z_scaled_list.append(unfolded_z_scaled_list_iter)
            unfolded_z_xsection_list.append(unfolded_z_xsection_list_iter)
            refolded = folding(unfolded_zvsjetpt, response_matrix, input_data)
            refolded.Sumw2()

            for ibin2 in range(self.p_nbin2_reco):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                refolded_z=refolded.ProjectionX("refolded_z",ibin2+1,ibin2+1,"e")
                refolding_test = input_data_z[ibin2].Clone("refolding_test_%d_%s" % (i+1,suffix))
                refolding_test.Divide(refolded_z)
                refolding_test_list_iter.append(refolding_test)
                cfolded_z = TCanvas('cfolded_z '+suffix, '1D output of folding'+suffix)
                pfolded_z = TPad('pfolded_z'+suffix, "1D output of ufolding"+suffix,0.0,0.001,1.0,1.0)
                setup_pad(pfolded_z)
                cfolded_z.SetCanvasSize(1900, 1500)
                cfolded_z.SetWindowSize(500, 500)
                setup_histogram(refolding_test,4)
                refolding_test.GetYaxis().SetRangeUser(0.5,1.5)
                refolding_test.SetXTitle("z")
                refolding_test.SetYTitle("refolding test")
                refolding_test.Draw()
                latex = TLatex(0.6,0.25,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
                draw_latex(latex2)
                cfolded_z.SaveAs("%s/cfolded_z_%d_%s.eps" % (self.d_resultsallpdata, i+1, suffix))

            refolded_jetpt=refolded.ProjectionY("refolded_jetpt",1,self.p_nbinshape_gen,"e")
            refolding_test_jetpt = input_data_jetpt.Clone("refolding_test_%d" % (i+1))
            refolding_test_jetpt.Divide(refolded_jetpt)
            refolding_test_jetpt_list.append(refolding_test_jetpt)
            cfolded_jetpt = TCanvas('cfolded_jetpt ' '1D output of folding')
            pfolded_jetpt = TPad('pfolded_jetpt', "1D output of folding",0.0,0.001,1.0,1.0)
            setup_pad(pfolded_jetpt)
            cfolded_jetpt.SetCanvasSize(1900, 1500)
            cfolded_jetpt.SetWindowSize(500, 500)
            setup_histogram(refolding_test_jetpt,4)
            refolding_test_jetpt.GetYaxis().SetRangeUser(0.5,1.5)
            refolding_test_jetpt.SetXTitle("p_{T,jet} GeV/c")
            refolding_test_jetpt.SetYTitle("refolding test")
            refolding_test_jetpt.Draw()
            latex = TLatex(0.2,0.25,'%.2f < z < %.2f GeV/c' % (self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1]))
            draw_latex(latex)
            latex2 = TLatex(0.2,0.2,'iteration %d' % (i+1))
            draw_latex(latex2)
            cfolded_jetpt.SaveAs("%s/cfolded_jetpt_%d.eps" % (self.d_resultsallpdata, i+1))
            
            refolding_test_list.append(refolding_test_list_iter)
            
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cconvergence_z = TCanvas('cconvergence_z ' + suffix, '1D output of convergence')
            pconvergence_z = TPad('pconvergence_z', "1D output of convergence",0.0,0.001,1.0,1.0)
            setup_pad(pconvergence_z)
            cconvergence_z.SetCanvasSize(1900, 1500)
            cconvergence_z.SetWindowSize(500, 500)
            leg_z = TLegend(.7, .45, .85, .85, "iterations")
            setup_legend(leg_z)
            for i in range(self.niter_unfolding) :
                setup_histogram(unfolded_z_scaled_list[i][ibin2],i+1)
                leg_z.AddEntry(unfolded_z_scaled_list[i][ibin2],("%d" % (i+1)),"LEP")
                if i==0 :
                    unfolded_z_scaled_list[i][ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01,self.lvarshape_binmax_reco[-1]-0.001)
                    unfolded_z_scaled_list[i][ibin2].GetYaxis().SetRangeUser(0,unfolded_z_scaled_list[i][ibin2].GetMaximum()*2.0)
                    unfolded_z_scaled_list[i][ibin2].SetXTitle("z")
                    unfolded_z_scaled_list[i][ibin2].SetYTitle("1/N_{jets} dN/dz")
                unfolded_z_scaled_list[i][ibin2].Draw("same")
                leg_z.Draw("same")
                latex = TLatex(0.6,0.2,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                cconvergence_z.SaveAs("%s/convergence_z_%s.eps" % (self.d_resultsallpdata,suffix))
                
            cinput_mc_gen_z = TCanvas('cinput_mc_gen_z '+suffix, '1D gen pythia z')
            pinput_mc_gen_z = TPad('pinput_mc_gen_z'+suffix, "1D gen pythia z"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pinput_mc_gen_z)
            cinput_mc_gen_z.SetCanvasSize(1900, 1500)
            cinput_mc_gen_z.SetWindowSize(500, 500)
            leg_input_mc_gen_z = TLegend(.2, .73, .45, .88, "")
            setup_legend(leg_input_mc_gen_z)
            setup_histogram(input_mc_gen_z[ibin2],4)
            leg_input_mc_gen_z.AddEntry(input_mc_gen_z[ibin2], "PYTHIA", "LEP")
            input_mc_gen_z[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01,self.lvarshape_binmax_reco[-1]-0.001)
            input_mc_gen_z[ibin2].GetYaxis().SetRangeUser(0.0,input_mc_gen_z[ibin2].GetMaximum()*2)
            input_mc_gen_z[ibin2].SetXTitle("z")
            input_mc_gen_z[ibin2].SetYTitle("1/N_{jets} dN/dz")
            input_mc_gen_z[ibin2].Draw()
            setup_histogram(unfolded_z_scaled_list[self.choice_iter_unfolding][ibin2],2)
            leg_input_mc_gen_z.AddEntry(unfolded_z_scaled_list[self.choice_iter_unfolding][ibin2], "unfolded ALICE data", "LEP")
            unfolded_z_scaled_list[self.choice_iter_unfolding][ibin2].Draw("same")
            setup_histogram(input_powheg_z[ibin2],3)
            leg_input_mc_gen_z.AddEntry(input_powheg_z[ibin2], "POWHEG + PYTHIA", "LEP")
            input_powheg_z[ibin2].Draw("same")
            setup_tgraph(tg_powheg[ibin2],30,0.3)
            tg_powheg[ibin2].Draw("5")
            leg_input_mc_gen_z.Draw("same")
            latex = TLatex(0.6,0.2,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            cinput_mc_gen_z.SaveAs("%s/cinput_mc_gen_z_%s.eps" % (self.d_resultsallpdata,suffix))
            cinput_mc_gen_z.SaveAs("%s/cinput_mc_gen_z_%s.pdf" % (self.d_resultsallpdata,suffix))


            cinput_mc_gen_z_xsection = TCanvas('cinput_mc_gen_z_xsection '+suffix, '1D gen pythia z xsection')
            pinput_mc_gen_z_xsection = TPad('pinput_mc_gen_z_xsection'+suffix, "1D gen pythia z xsection"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pinput_mc_gen_z_xsection)
            cinput_mc_gen_z_xsection.SetCanvasSize(1900, 1500)
            cinput_mc_gen_z_xsection.SetWindowSize(500, 500)
            leg_input_mc_gen_z_xsection = TLegend(.2, .73, .45, .88, "")
            setup_legend(leg_input_mc_gen_z_xsection)
            setup_histogram(unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2],4)
            leg_input_mc_gen_z_xsection.AddEntry(unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2], "unfolded ALICE data", "LEP")
            unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01,self.lvarshape_binmax_reco[-1]-0.001)
            unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2].GetYaxis().SetRangeUser(0.0,unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2].GetMaximum()*2)
            unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2].SetXTitle("z")
            unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2].SetYTitle("xsection dN/dz")
            unfolded_z_xsection_list[self.choice_iter_unfolding][ibin2].Draw()
            setup_histogram(input_powheg_xsection_z[ibin2],3)
            leg_input_mc_gen_z.AddEntry(input_powheg_xsection_z[ibin2], "POWHEG + PYTHIA", "LEP")
            input_powheg_xsection_z[ibin2].Draw("same")
            setup_tgraph(tg_powheg_xsection[ibin2],30,0.3)
            tg_powheg_xsection[ibin2].Draw("5")
            latex = TLatex(0.6,0.2,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_input_mc_gen_z_xsection.Draw("same")
            cinput_mc_gen_z_xsection.SaveAs("%s/cinput_mc_gen_z_xsection_%s.eps" % (self.d_resultsallpdata,suffix))
            cinput_mc_gen_z_xsection.SaveAs("%s/cinput_mc_gen_z_xsection_%s.pdf" % (self.d_resultsallpdata,suffix))
        
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            cconvergence_refolding_z = TCanvas('cconvergence_refolding_z '+suffix, '1D output of refolding convergence'+suffix)
            pconvergence_refolding_z = TPad('pconvergence_refolding_z'+suffix, "1D output of refolding convergence"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pconvergence_refolding_z)
            cconvergence_refolding_z.SetCanvasSize(1900, 1500)
            cconvergence_refolding_z.SetWindowSize(500, 500)
            leg_refolding_z = TLegend(.7, .5, .85, .9, "iterations")
            setup_legend(leg_refolding_z)
            for i in range(self.niter_unfolding) :
                setup_histogram(refolding_test_list[i][ibin2],i+1)
                leg_refolding_z.AddEntry(refolding_test_list[i][ibin2],("%d" % (i+1)),"LEP")
                refolding_test_list[i][ibin2].Draw("same")
                if i==0 :
                    refolding_test_list[i][ibin2].SetXTitle("z")
                    refolding_test_list[i][ibin2].SetYTitle("refolding test")
                    refolding_test_list[i][ibin2].GetYaxis().SetRangeUser(0.5,2.0)
            leg_refolding_z.Draw("same")
            latex = TLatex(0.6,0.2,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cconvergence_refolding_z.SaveAs("%s/convergence_refolding_z_%s.eps" % (self.d_resultsallpdata, suffix))

                
            input_data_z_scaled=input_data_z[ibin2].Clone("input_data_z_scaled_%s" % suffix)
            input_data_z_scaled.Scale(1.0/input_data_z_scaled.Integral(1,-1))

            cunfolded_not_z = TCanvas('cunfolded_not_z '+suffix, 'Unfolded vs not Unfolded'+suffix)
            punfolded_not_z = TPad('punfolded_not_z'+suffix, "Unfolded vs not Unfolded"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(punfolded_not_z)
            cunfolded_not_z.SetCanvasSize(1900, 1500)
            cunfolded_not_z.SetWindowSize(500, 500)
            leg_cunfolded_not_z = TLegend(.15, .75, .35, .9, "")
            setup_legend(leg_cunfolded_not_z)
            setup_histogram(unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1],2)
            leg_cunfolded_not_z.AddEntry(unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1], "unfolded ALICE data", "LEP")
            unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01,self.lvarshape_binmax_reco[-1]-0.001)
            unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetYaxis().SetRangeUser(0.0,unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetMaximum()*1.5)
            unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].Draw()
            setup_histogram(input_data_z_scaled,4)
            leg_cunfolded_not_z.AddEntry(input_data_z_scaled, "Side-Band sub, Eff Corrected", "LEP")
            input_data_z_scaled.Draw("same")
            leg_cunfolded_not_z.Draw("same")
            latex = TLatex(0.7,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cunfolded_not_z.SaveAs("%s/cunfolded_not_z_%s.eps" % (self.d_resultsallpdata, suffix))

            zbin_reco =self.varshaperanges_reco
            zbinarray_reco=array('d',zbin_reco)
            h_unfolded_not_stat_error=TH1F("h_unfolded_not_stat_error"+suffix,"h_unfolded_not_stat_error"+suffix,self.p_nbinshape_reco,zbinarray_reco)
            for ibinshape in range(self.p_nbinshape_reco):
                error_on_unfolded = unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetBinError(input_mc_gen.GetXaxis().FindBin(self.lvarshape_binmin_reco[ibinshape]))
                content_on_unfolded = unfolded_z_scaled_list[self.choice_iter_unfolding][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetBinContent(input_mc_gen.GetXaxis().FindBin(self.lvarshape_binmin_reco[ibinshape]))
                error_on_input_data = input_data_z_scaled.GetBinError(ibinshape+1)
                content_on_input_data = input_data_z_scaled.GetBinContent(ibinshape+1)
                if error_on_input_data is not 0 :
                    h_unfolded_not_stat_error.SetBinContent(ibinshape+1,(error_on_unfolded/content_on_unfolded)/(error_on_input_data/content_on_input_data))
                else :
                    h_unfolded_not_stat_error.SetBinContent(ibinshape+1,0.0)
            cunfolded_not_stat_error = TCanvas('cunfolded_not_stat_error '+suffix, 'Ratio of stat error after to before unfolding'+suffix)
            punfolded_not_stat_error = TPad('punfolded_not_stat_error'+suffix, "Ratio of stat error after to before unfolding"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(punfolded_not_stat_error)
            cunfolded_not_stat_error.SetCanvasSize(1900, 1500)
            cunfolded_not_stat_error.SetWindowSize(500, 500)
            setup_histogram(h_unfolded_not_stat_error,4)
            h_unfolded_not_stat_error.SetXTitle("z")
            h_unfolded_not_stat_error.SetYTitle("relative stat err")
            h_unfolded_not_stat_error.GetYaxis().SetRangeUser(0.0,h_unfolded_not_stat_error.GetMaximum()*1.6)
            h_unfolded_not_stat_error.Draw()
            latex = TLatex(0.2,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_reco[ibin2],self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cunfolded_not_stat_error.SaveAs("%s/unfolded_not_stat_error_%s.eps" % (self.d_resultsallpdata,suffix))

            
        cconvergence_jetpt = TCanvas('cconvergence_jetpt ', '1D output of convergence')
        pconvergence_jetpt = TPad('pconvergence_jetpt', "1D output of convergence",0.0,0.001,1.0,1.0)
        setup_pad(pconvergence_jetpt)
        cconvergence_jetpt.SetCanvasSize(1900, 1500)
        cconvergence_jetpt.SetWindowSize(500, 500)
        leg_jetpt = TLegend(.7, .5, .85, .9, "iterations")
        setup_legend(leg_jetpt)
        for i in range(self.niter_unfolding) :
            setup_histogram(unfolded_jetpt_scaled_list[i],i+1)
            leg_jetpt.AddEntry(unfolded_jetpt_scaled_list[i],("%d" % (i+1)),"LEP")
            if i==0 :
                unfolded_jetpt_scaled_list[i].GetXaxis().SetRangeUser(self.lvar2_binmin_reco[0]+0.01,self.lvar2_binmax_reco[-1]-0.001)
                unfolded_jetpt_scaled_list[i].GetYaxis().SetRangeUser(0,unfolded_jetpt_scaled_list[i].GetMaximum()*2.0)
                unfolded_jetpt_scaled_list[i].SetXTitle("p_{T,jet} GeV/c")
                unfolded_jetpt_scaled_list[i].SetYTitle("1/n_{jets} dN/dp_{T,jet}")
            unfolded_jetpt_scaled_list[i].Draw("same")
        leg_jetpt.Draw("same")
        latex = TLatex(0.2,0.8,'%.2f < z < %.2f' % (self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1]))
        draw_latex(latex)
        cconvergence_jetpt.SaveAs("%s/convergence_jetpt.eps" % (self.d_resultsallpdata))

        cconvergence_refolding_jetpt = TCanvas('cconvergence_refolding_jetpt ', '1D output of refolding convergence')
        pconvergence_refolding_jetpt = TPad('pconvergence_refolding_jetpt', "1D output of refolding convergence",0.0,0.001,1.0,1.0)
        setup_pad(pconvergence_refolding_jetpt)
        cconvergence_refolding_jetpt.SetCanvasSize(1900, 1500)
        cconvergence_refolding_jetpt.SetWindowSize(500, 500)
        leg_refolding_jetpt = TLegend(.7, .5, .85, .9, "iterations")
        setup_legend(leg_refolding_jetpt)
        for i in range(self.niter_unfolding) :
            setup_histogram(refolding_test_jetpt_list[i],i+1)
            leg_refolding_jetpt.AddEntry(refolding_test_jetpt_list[i],("%d" % (i+1)),"LEP")
            refolding_test_jetpt_list[i].Draw("same")
            refolding_test_jetpt_list[i].SetXTitle("p_{T,jet} GeV/c")
            refolding_test_jetpt_list[i].SetYTitle("1/n_{jets} dN/dp_{T,jet}")
            refolding_test_jetpt_list[i].GetYaxis().SetRangeUser(0.5,2.0)
        leg_refolding_jetpt.Draw("same")
        latex = TLatex(0.2,0.8,'%.2f < z < %.2f' % (self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1]))
        draw_latex(latex)
        cconvergence_refolding_jetpt.SaveAs("%s/convergence_refolding_jetpt.eps" % (self.d_resultsallpdata))
        
    def unfolding_closure(self):
        lfile = TFile.Open(self.n_filemass,"update")
        fileouts = TFile.Open("%s/unfolding_closure_results%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "recreate")
        unfolding_input_file = TFile.Open(self.n_fileff)
        response_matrix = unfolding_input_file.Get("response_matrix_closure")
        hzvsjetpt_reco_nocuts = unfolding_input_file.Get("hzvsjetpt_reco_nocuts_closure")
        hzvsjetpt_reco_eff = unfolding_input_file.Get("hzvsjetpt_reco_cuts_closure")
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)
        input_mc_det = unfolding_input_file.Get("input_closure_reco")
        input_mc_det.Multiply(hzvsjetpt_reco_eff)                        
        input_mc_gen = unfolding_input_file.Get("input_closure_gen")
        kinematic_eff=[]
        hz_gen_nocuts=[]
        input_mc_det_z=[]
        input_mc_gen_z=[]

        hjetpt_fracdiff_list=[]
        hz_fracdiff_list=[]

        kinematic_eff_jetpt = unfolding_input_file.Get("hjetpt_gen_cuts_closure")
        hjetpt_gen_nocuts=unfolding_input_file.Get("hjetpt_gen_nocuts_closure")
        kinematic_eff_jetpt.Divide(hjetpt_gen_nocuts)
        input_mc_gen_jetpt=input_mc_gen.ProjectionY("input_mc_gen_jetpt",1,self.p_nbinshape_gen,"e")
        input_mc_gen_jetpt.Scale(1.0/input_mc_gen_jetpt.Integral(1,-1))

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            input_mc_det_z.append(input_mc_det.ProjectionX("input_mc_det_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_mc_det_z[ibin2].Scale(1.0/input_mc_det_z[ibin2].Integral(1,-1))

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_mc_gen_z[ibin2].Scale(1.0/input_mc_gen_z[ibin2].Integral(1,-1))            
            kinematic_eff.append(unfolding_input_file.Get("hz_gen_cuts_closure"+suffix))
            hz_gen_nocuts.append(unfolding_input_file.Get("hz_gen_nocuts_closure"+suffix))
            kinematic_eff[ibin2].Divide(hz_gen_nocuts[ibin2])

        
        unfolded_z_closure_list=[]
        unfolded_jetpt_closure_list=[]

        for i in range(self.niter_unfolding) :
            unfolded_z_closure_list_iter=[]
            unfolding_object = RooUnfoldBayes(response_matrix, input_mc_det, i+1)
            unfolded_zvsjetpt = unfolding_object.Hreco(2)
            unfolded_zvsjetpt.Sumw2()
            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                unfolded_z = unfolded_zvsjetpt.ProjectionX("unfolded_z_%d_%s" % (i+1,suffix),ibin2+1,ibin2+1,"e")
                unfolded_z.Divide(kinematic_eff[ibin2])
                unfolded_z.Scale(1.0/unfolded_z.Integral(1,-1))
                unfolded_z.Divide(input_mc_gen_z[ibin2])
                fileouts.cd()
                unfolded_z.Write("closure_test_%d_%s" % (i+1,suffix))
                unfolded_z_closure_list_iter.append(unfolded_z)
            
                cclosure_z = TCanvas('cclosure_z '+suffix, '1D output of closure'+suffix)
                pclosure_z = TPad('pclosure_z'+suffix, "1D output of closure"+suffix,0.0,0.001,1.0,1.0)
                setup_pad(pclosure_z)
                cclosure_z.SetCanvasSize(1900, 1500)
                cclosure_z.SetWindowSize(500, 500)
                setup_histogram(unfolded_z,4)
                unfolded_z.GetYaxis().SetRangeUser(0.5,1.5)
                unfolded_z.GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01,self.lvarshape_binmax_reco[-1]-0.001)
                unfolded_z.SetXTitle("z")
                unfolded_z.SetYTitle("closure test")
                unfolded_z.Draw()
                latex = TLatex(0.6,0.25,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
                draw_latex(latex2)
                cclosure_z.SaveAs("%s/cclosure_z_%d_%s.eps" % (self.d_resultsallpdata, i+1,suffix))

            unfolded_jetpt = unfolded_zvsjetpt.ProjectionY("unfolded_jetpt_%d" % (i+1),1,self.p_nbinshape_gen,"e")
            unfolded_jetpt.Divide(kinematic_eff_jetpt)
            unfolded_jetpt.Scale(1.0/unfolded_jetpt.Integral(1,-1))
            unfolded_jetpt.Divide(input_mc_gen_jetpt)
            fileouts.cd()
            unfolded_jetpt.Write("closure_test_jetpt_%d" % (i+1))
            unfolded_jetpt_closure_list.append(unfolded_jetpt)

            cclosure_jetpt = TCanvas('cclosure_jetpt ', '1D output of closure')
            pclosure_jetpt = TPad('pclosure_jetpt', "1D output of closure",0.0,0.001,1.0,1.0)
            setup_pad(pclosure_jetpt)
            cclosure_jetpt.SetCanvasSize(1900, 1500)
            cclosure_jetpt.SetWindowSize(500, 500)
            setup_histogram(unfolded_jetpt,4)
            unfolded_jetpt.GetYaxis().SetRangeUser(0.5,1.5)
            unfolded_jetpt.GetXaxis().SetRangeUser(0.21,0.99)
            unfolded_jetpt.SetXTitle("p_{T,jet} GeV/c")
            unfolded_jetpt.SetYTitle("closure test")
            unfolded_jetpt.Draw()
            latex = TLatex(0.6,0.25,'%.2f < z < %.2f' % (self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1]))
            draw_latex(latex)
            latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
            draw_latex(latex2)
            cclosure_jetpt.SaveAs("%s/cclosure_jetpt_%d.eps" % (self.d_resultsallpdata, i+1))
                
            unfolded_z_closure_list.append(unfolded_z_closure_list_iter)

        input_data_z=[]
        unfolding_input_data_file = TFile.Open("%s/sideband_sub%s%s.root" % \
                                               (self.d_resultsallpdata, self.case, self.typean))
        input_data = unfolding_input_data_file.Get("hzvsjetpt")
        
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            input_data_z.append(input_data.ProjectionX("input_data_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_data_z[ibin2].Scale(1.0/input_data_z[ibin2].Integral(1,-1))
            

        
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cconvergence_closure_z = TCanvas('cconvergence_closure_z '+suffix, '1D output of closure convergence'+suffix)
            pconvergence_closure_z = TPad('pconvergence_closure_z'+suffix, "1D output of closure convergence"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pconvergence_closure_z)
            cconvergence_closure_z.SetCanvasSize(1900, 1500)
            cconvergence_closure_z.SetWindowSize(500, 500)
            leg_closure = TLegend(.7, .5, .85, .9, "iterations")
            setup_legend(leg_closure)
            for i in range(self.niter_unfolding) :
                setup_histogram(unfolded_z_closure_list[i][ibin2],i+1)
                leg_closure.AddEntry(unfolded_z_closure_list[i][ibin2],("%d" % (i+1)),"LEP")
                if i == 0:
                    unfolded_z_closure_list[i][ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01,self.lvarshape_binmax_gen[-1]-0.001)
                    unfolded_z_closure_list[i][ibin2].GetYaxis().SetRangeUser(0.6,2.0)
                    unfolded_z_closure_list[i][ibin2].SetXTitle("z")
                    unfolded_z_closure_list[i][ibin2].SetYTitle("closure test")
                unfolded_z_closure_list[i][ibin2].Draw("same")
            leg_closure.Draw("same")
            latex = TLatex(0.2,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            cconvergence_closure_z.SaveAs("%s/convergence_closure_z_%s.eps" % (self.d_resultsallpdata, suffix))

        cconvergence_closure_jetpt = TCanvas('cconvergence_closure_jetpt ', '1D output of closure convergence')
        pconvergence_closure_jetpt = TPad('pconvergence_closure_jetpt', "1D output of closure convergence",0.0,0.001,1.0,1.0)
        setup_pad(pconvergence_closure_jetpt)
        cconvergence_closure_jetpt.SetCanvasSize(1900, 1500)
        cconvergence_closure_jetpt.SetWindowSize(500, 500)
        leg_closure_jetpt = TLegend(.7, .5, .85, .9, "iterations")
        setup_legend(leg_closure_jetpt)
        for i in range(self.niter_unfolding) :
            setup_histogram(unfolded_jetpt_closure_list[i],i+1)
            leg_closure_jetpt.AddEntry(unfolded_jetpt_closure_list[i],("%d" % (i+1)),"LEP")
            if i == 0:
                unfolded_jetpt_closure_list[i].GetXaxis().SetRangeUser(self.lvar2_binmin_gen[0]+0.01,self.lvar2_binmax_gen[-1]-0.001)
                unfolded_jetpt_closure_list[i].GetYaxis().SetRangeUser(0.6,2.0)
                unfolded_jetpt_closure_list[i].SetXTitle("p_{T,jet} GeV")
                unfolded_jetpt_closure_list[i].SetYTitle("closure test")
            unfolded_jetpt_closure_list[i].Draw("same")
        leg_closure_jetpt.Draw("same")
        latex = TLatex(0.6,0.25,'%.2f < z < %.2f' % (self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1]))
        draw_latex(latex)
        cconvergence_closure_jetpt.SaveAs("%s/convergence_closure_jetpt.eps" % (self.d_resultsallpdata))


    def jetsystematics(self):
        
        input_file_default=TFile.Open("%s/unfolding_results%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "update")

        input_hisotgrams_default=[]
        for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                input_hisotgrams_default.append(input_file_default.Get("unfolded_z_%d_%s" % (self.choice_iter_unfolding,suffix)))
        
        input_files_sys=[]
        for sys_cat in range(len(self.systematic_catagories)):
            if self.systematic_catagories[sys_cat]=="regularisation":
                continue
            input_files_sysvar=[]
            for sys_var in range(self.systematic_variations[sys_cat]):
                input_files_sysvar.append(TFile.Open("/data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1/systematics/%s/sys_%d/pp_data/resultsMBjetvspt/unfolding_resultsLcpK0sppMBjetvspt.root" % (self.systematic_catagories[sys_cat],sys_var+1),"update"))
            input_files_sys.append(input_files_sysvar) 

        input_histograms_sys=[]
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_histograms_syscat=[]
            for sys_cat in range(len(self.systematic_catagories)):
                input_histograms_syscatvar=[]
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if self.systematic_catagories[sys_cat]== "regularisation" :
                        if sys_var==0:
                            input_histograms_syscatvar.append(input_file_default.Get("unfolded_z_%d_%s" % (self.niterunfoldingregdown,suffix)))
                        else:
                            input_histograms_syscatvar.append(input_file_default.Get("unfolded_z_%d_%s" % (self.niterunfoldingregup,suffix)))
                    else:
                        input_histograms_syscatvar.append(input_files_sys[sys_cat][sys_var].Get("unfolded_z_%d_%s" % (self.choice_iter_unfolding,suffix)))
                input_histograms_syscat.append(input_histograms_syscatvar)
            input_histograms_sys.append(input_histograms_syscat)

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            nsys=0
            csysvar = TCanvas('csysvar '+suffix, 'systematic variations'+suffix)
            psysvar = TPad('psysvar'+suffix, "systematic variations"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(psysvar)
            csysvar.SetCanvasSize(1900, 1500)
            csysvar.SetWindowSize(500, 500)
            leg_sysvar = TLegend(.7, .5, .85, .9, "systematics")
            setup_legend(leg_sysvar)
            leg_sysvar.AddEntry(input_hisotgrams_default[ibin2],"default","LEP")
            setup_histogram(input_hisotgrams_default[ibin2],1)
            input_hisotgrams_default[ibin2].GetYaxis().SetRangeUser(0.0,input_hisotgrams_default[ibin2].GetMaximum()*1.5)
            input_hisotgrams_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01,self.lvarshape_binmax_gen[-1]-0.001)
            input_hisotgrams_default[ibin2].SetXTitle("z")
            input_hisotgrams_default[ibin2].SetYTitle("1/n_{jets} dN/dz")
            input_hisotgrams_default[ibin2].Draw()
            for sys_cat in range(len(self.systematic_catagories)):
                for sys_var in range(self.systematic_variations[sys_cat]):
                    nsys=nsys+1
                    leg_sysvar.AddEntry(input_histograms_sys[ibin2][sys_cat][sys_var],("%s_%d" % (self.systematic_catagories[sys_cat],sys_var+1)),"LEP")
                    setup_histogram(input_histograms_sys[ibin2][sys_cat][sys_var],nsys+1)
                    input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
            latex = TLatex(0.2,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_sysvar.Draw("same")
            csysvar.SaveAs("%s/ysvar_%s.eps" % (self.d_resultsallpdata, suffix))


            for sys_cat in range(len(self.systematic_catagories)):
                suffix2="_%s" % (self.systematic_catagories[sys_cat])
                nsys=0
                csysvar_each = TCanvas('csysvar '+suffix2+suffix, 'systematic variations'+suffix2+suffix)
                psysvar_each = TPad('psysvar'+suffix2+suffix, "systematic variations"+suffix2+suffix,0.0,0.001,1.0,1.0)
                setup_pad(psysvar_each)
                csysvar_each.SetCanvasSize(1900, 1500)
                csysvar_each.SetWindowSize(500, 500)
                leg_sysvar_each = TLegend(.7, .45, .85, .85, self.systematic_catagories[sys_cat])
                setup_legend(leg_sysvar_each)
                leg_sysvar_each.AddEntry(input_hisotgrams_default[ibin2],"default","LEP")
                setup_histogram(input_hisotgrams_default[ibin2],1)
                if sys_cat == 0 :
                    input_hisotgrams_default[ibin2].GetYaxis().SetRangeUser(0.0,input_hisotgrams_default[ibin2].GetMaximum()*2.5)
                input_hisotgrams_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01,self.lvarshape_binmax_gen[-1]-0.001)
                input_hisotgrams_default[ibin2].SetXTitle("z")
                input_hisotgrams_default[ibin2].SetYTitle("1/n_{jets} dN/dz")
                input_hisotgrams_default[ibin2].Draw()
                for sys_var in range(self.systematic_variations[sys_cat]):
                    nsys=nsys+1
                    leg_sysvar_each.AddEntry(input_histograms_sys[ibin2][sys_cat][sys_var],("%d" % (sys_var+1)),"LEP")
                    setup_histogram(input_histograms_sys[ibin2][sys_cat][sys_var],nsys+1)
                    input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
                latex = TLatex(0.2,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                leg_sysvar_each.Draw("same")
                csysvar_each.SaveAs("%s/ysvar%s_%s.eps" % (self.d_resultsallpdata, suffix2, suffix))


            

        sys_up=[]
        sys_down=[]
        sys_up_full=[]
        sys_down_full=[]
        for ibin2 in range(self.p_nbin2_gen):
            sys_up_jetpt=[]
            sys_down_jetpt=[]
            sys_up_z_full=[]
            sys_down_z_full=[]
            for ibinshape in range(self.p_nbinshape_gen):
                sys_up_z=[]
                sys_down_z=[]
                error_full_up=0
                error_full_down=0
                for sys_cat in range(len(self.systematic_catagories)):
                    error_var_up=0
                    error_var_down=0
                    count_sys_up=0
                    count_sys_down=0
                    for sys_var in range(self.systematic_variations[sys_cat]):
                        error = input_histograms_sys[ibin2][sys_cat][sys_var].GetBinContent(ibinshape+1)-input_hisotgrams_default[ibin2].GetBinContent(ibinshape+1)
                        if error >= 0 :
                            if self.systematic_rms[sys_cat] is True:
                                error_var_up+=error*error
                                count_sys_up=count_sys_up+1
                            else:
                                if error > error_var_up :
                                    error_var_up=error
                        else:
                            if self.systematic_rms[sys_cat] is True:
                                error_var_down+=error*error
                                count_sys_down=count_sys_down+1
                            else:
                                if abs(error) > error_var_down :
                                    error_var_down = abs(error)
                    if self.systematic_rms[sys_cat]  is True:
                        if count_sys_up is not 0:
                            error_var_up = error_var_up/count_sys_up
                        else :
                            error_var_up=0.0
                        error_var_up=sqrt(error_var_up)
                        if count_sys_down is not 0:
                            error_var_down = error_var_down/count_sys_down
                        else :
                            error_var_down=0.0
                        error_var_down=sqrt(error_var_down)
                    if self.systematic_symmetrise[sys_cat] is True :
                        if error_var_up > error_var_down:
                            error_var_down = error_var_up
                        else :
                            error_var_up = error_var_down
                    error_full_up+=error_var_up*error_var_up
                    error_full_down+=error_var_down*error_var_down
                    sys_up_z.append(error_var_up)
                    sys_down_z.append(error_var_down)
                error_full_up=sqrt(error_full_up)
                sys_up_z_full.append(error_full_up)
                error_full_down=sqrt(error_full_down)
                sys_down_z_full.append(error_full_down)
                sys_up_jetpt.append(sys_up_z)
                sys_down_jetpt.append(sys_down_z)
            sys_up_full.append(sys_up_z_full)
            sys_down_full.append(sys_down_z_full)
            sys_up.append(sys_up_jetpt)
            sys_down.append(sys_down_jetpt)


        tgsys=[]
        tgsys_cat=[]
        for ibin2 in range(self.p_nbin2_gen):
            shapebins_centres=[]
            shapebins_contents=[]
            shapebins_widths_up=[]
            shapebins_widths_down=[]
            shapebins_error_up=[]
            shapebins_error_down=[]
            tgsys_cat_z=[]
            for ibinshape in range(self.p_nbinshape_gen):
                shapebins_centres.append(input_hisotgrams_default[ibin2].GetBinCenter(ibinshape+1))
                shapebins_contents.append(input_hisotgrams_default[ibin2].GetBinContent(ibinshape+1))
                shapebins_widths_up.append(input_hisotgrams_default[ibin2].GetBinWidth(ibinshape+1)*0.5)
                shapebins_widths_down.append(input_hisotgrams_default[ibin2].GetBinWidth(ibinshape+1)*0.5)
                shapebins_error_up.append(sys_up_full[ibin2][ibinshape])
                shapebins_error_down.append(sys_down_full[ibin2][ibinshape])
            shapebins_centres_array = array('d',shapebins_centres)
            shapebins_contents_array = array('d',shapebins_contents)
            shapebins_widths_up_array = array('d',shapebins_widths_up)
            shapebins_widths_down_array = array('d',shapebins_widths_down)
            shapebins_error_up_array = array('d',shapebins_error_up)
            shapebins_error_down_array = array('d',shapebins_error_down)
            for sys_cat in range(len(self.systematic_catagories)):
                shapebins_contents_cat=[]
                shapebins_error_up_cat=[]
                shapebins_error_down_cat=[]
                for ibinshape in range(self.p_nbinshape_gen):
                    shapebins_contents_cat.append(1.0)
                    shapebins_error_up_cat.append(sys_up[ibin2][ibinshape][sys_cat]/input_hisotgrams_default[ibin2].GetBinContent(ibinshape+1))
                    shapebins_error_down_cat.append(sys_down[ibin2][ibinshape][sys_cat]/input_hisotgrams_default[ibin2].GetBinContent(ibinshape+1))
                shapebins_contents_cat_array = array('d',shapebins_contents_cat)
                shapebins_error_up_cat_array = array('d',shapebins_error_up_cat)
                shapebins_error_down_cat_array = array('d',shapebins_error_down_cat)
                tgsys_cat_z.append(TGraphAsymmErrors(self.p_nbinshape_gen,shapebins_centres_array,shapebins_contents_cat_array,shapebins_widths_down_array,shapebins_widths_up_array,shapebins_error_down_cat_array,shapebins_error_up_cat_array))
            tgsys_cat.append(tgsys_cat_z)
                
            tgsys.append(TGraphAsymmErrors(self.p_nbinshape_gen,shapebins_centres_array,shapebins_contents_array,shapebins_widths_down_array,shapebins_widths_up_array,shapebins_error_down_array,shapebins_error_up_array))

        h_default_stat_err=[]
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            h_default_stat_err.append(input_hisotgrams_default[ibin2].Clone("h_default_stat_err"+suffix))
            for i in range(h_default_stat_err[ibin2].GetNbinsX()):
                h_default_stat_err[ibin2].SetBinContent(i+1,1.0)
                h_default_stat_err[ibin2].SetBinError(i+1,input_hisotgrams_default[ibin2].GetBinError(i+1)/input_hisotgrams_default[ibin2].GetBinContent(i+1))



        input_pythia8_file = []
        input_pythia8 = []
        input_pythia8_xsection = []
        input_pythia8_z=[]
        input_pythia8_xsection_z=[]
        for i_pythia8 in range(len(self.pythia8_prompt_variations)):
            input_pythia8_file.append(TFile.Open("%s%s.root" % (self.pythia8_prompt_variations_path,self.pythia8_prompt_variations[i_pythia8])))
            input_pythia8.append(input_pythia8_file[i_pythia8].Get("fh2_pythia8_prompt"))
            input_pythia8_xsection.append(input_pythia8_file[i_pythia8].Get("fh2_pythia8_prompt_xsection"))
            input_pythia8_z_jetpt=[]
            input_pythia8_xsection_z_jetpt=[]
            for ibin2 in range(self.p_nbin2_gen) :
                suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                input_pythia8_z_jetpt.append(input_pythia8[i_pythia8].ProjectionX("input_pythia8"+self.pythia8_prompt_variations[i_pythia8]+suffix,ibin2+1,ibin2+1,"e"))
                input_pythia8_z_jetpt[ibin2].Scale(1.0/input_pythia8_z_jetpt[ibin2].Integral(1,-1))
                input_pythia8_xsection_z_jetpt.append(input_pythia8_xsection[i_pythia8].ProjectionX("input_pythia8_xsection"+self.pythia8_prompt_variations[i_pythia8]+suffix,ibin2+1,ibin2+1,"e"))
            input_pythia8_z.append(input_pythia8_z_jetpt)
            input_pythia8_xsection_z.append(input_pythia8_xsection_z_jetpt)

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cfinalwsys = TCanvas('cfinalwsys '+suffix, 'final result with systematic errors'+suffix)
            pfinalwsys = TPad('pfinalwsys'+suffix, "final result with systematic errors"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pfinalwsys)
            cfinalwsys.SetCanvasSize(1900, 1500)
            cfinalwsys.SetWindowSize(500, 500)
            leg_finalwsys = TLegend(.65, .6, .85, .8, "")
            setup_legend(leg_finalwsys)
            leg_finalwsys.AddEntry(input_hisotgrams_default[ibin2],"ALICE Data","LEP")
            setup_histogram(input_hisotgrams_default[ibin2],4)
            input_hisotgrams_default[ibin2].GetYaxis().SetRangeUser(0.0,input_hisotgrams_default[ibin2].GetMaximum()/2.5)
            input_hisotgrams_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01,self.lvarshape_binmax_gen[-1]-0.001)
            input_hisotgrams_default[ibin2].SetXTitle("z")
            input_hisotgrams_default[ibin2].SetYTitle("1/n_{jets} dN/dz")
            input_hisotgrams_default[ibin2].Draw()
            leg_finalwsys.AddEntry(tgsys[ibin2],"systematic error","LEP")
            setup_tgraph(tgsys[ibin2],17,0.3)
            tgsys[ibin2].Draw("5")
            latex = TLatex(0.2,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_finalwsys.Draw("same")
            cfinalwsys.SaveAs("%s/finalwsys_%s.pdf" % (self.d_resultsallpdata, suffix))


            cfinalwsys_wmodels = TCanvas('cfinalwsys_wmodels '+suffix, 'final result with systematic errors with models'+suffix)
            pfinalwsys_wmodels = TPad('pfinalwsys_wmodels'+suffix, "final result with systematic errors with models"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pfinalwsys_wmodels)
            cfinalwsys_wmodels.SetCanvasSize(1900, 1500)
            cfinalwsys_wmodels.SetWindowSize(500, 500)
            leg_finalwsys_wmodels = TLegend(.6, .6, .8, .8, "")
            setup_legend(leg_finalwsys_wmodels)
            leg_finalwsys_wmodels.AddEntry(input_hisotgrams_default[ibin2],"ALICE Data","LEP")
            setup_histogram(input_hisotgrams_default[ibin2],1)
            input_hisotgrams_default[ibin2].GetYaxis().SetRangeUser(0.0,input_hisotgrams_default[ibin2].GetMaximum())
            input_hisotgrams_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01,self.lvarshape_binmax_gen[-1]-0.001)
            input_hisotgrams_default[ibin2].SetXTitle("z")
            input_hisotgrams_default[ibin2].SetYTitle("1/n_{jets} dN/dz")
            input_hisotgrams_default[ibin2].Draw()
            leg_finalwsys_wmodels.AddEntry(tgsys[ibin2],"systematic error","LEP")
            setup_tgraph(tgsys[ibin2],17,0.3)
            tgsys[ibin2].Draw("5")
            for i_pythia8 in range(len(self.pythia8_prompt_variations)):
                setup_histogram(input_pythia8_z[i_pythia8][ibin2],i_pythia8+2)
                leg_finalwsys_wmodels.AddEntry(input_pythia8_z[i_pythia8][ibin2],self.pythia8_prompt_variations[i_pythia8],"LEP")
                input_pythia8_z[i_pythia8][ibin2].Draw("same")         
            latex = TLatex(0.2,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_finalwsys_wmodels.Draw("same")
            cfinalwsys_wmodels.SaveAs("%s/finalwsys_wmodels_%s.pdf" % (self.d_resultsallpdata, suffix))

            crelativesys = TCanvas('crelativesys '+suffix, 'relative systematic errors'+suffix)
            prelativesys = TPad('prelativesys'+suffix, "relative systematic errors"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(prelativesys)
            crelativesys.SetCanvasSize(1900, 1500)
            crelativesys.SetWindowSize(500, 500)
            leg_relativesys = TLegend(.7, .5, .85, .9, "")
            setup_legend(leg_relativesys)
            for sys_cat in range(len(self.systematic_catagories)):
                setup_tgraph(tgsys_cat[ibin2][sys_cat],sys_cat+1,0.3)
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetRangeUser(0.0,2.8)
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01,self.lvarshape_binmax_gen[-1]-0.001)
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetTitle("z")
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetTitle("relative systematic error")
                leg_relativesys.AddEntry(tgsys_cat[ibin2][sys_cat],self.systematic_catagories[sys_cat],"LEP")
                if sys_cat == 0:
                    tgsys_cat[ibin2][sys_cat].Draw("A2")
                else :
                    tgsys_cat[ibin2][sys_cat].Draw("2")
            setup_histogram(h_default_stat_err[ibin2],1)
            h_default_stat_err[ibin2].Draw("same")
            latex = TLatex(0.2,0.8,'%.2f < p_{T,jet} < %.2f GeV/c' % (self.lvar2_binmin_gen[ibin2],self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_relativesys.Draw("same")
            crelativesys.SaveAs("%s/relativesys_%s.pdf" % (self.d_resultsallpdata, suffix))
