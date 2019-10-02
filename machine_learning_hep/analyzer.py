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
from math import sqrt
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
from subprocess import Popen
import numpy as np
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TCanvas, TPad, TF1
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed
from ROOT import TLatex
from ROOT import gInterpreter, gPad
# HF specific imports
from machine_learning_hep.globalfitter import Fitter
from  machine_learning_hep.logger import get_logger
from  machine_learning_hep.io import dump_yaml_from_dict
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
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.p_nbin2 = len(self.lvar2_binmin)

        self.d_resultsallpmc = resultsmc
        self.d_resultsallpdata = resultsdata

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
        self.p_rebin = datap["analysis"][self.typean]["rebin"]
        self.p_includesecpeak = datap["analysis"][self.typean]["includesecpeak"]
        self.p_masssecpeak = datap["analysis"][self.typean]["masssecpeak"] \
                if self.p_includesecpeak else None
        self.p_fix_masssecpeak = datap["analysis"][self.typean]["fix_masssecpeak"] \
                if self.p_includesecpeak else None
        self.p_widthsecpeak = datap["analysis"][self.typean]["widthsecpeak"] \
                if self.p_includesecpeak else None
        self.p_fix_widthsecpeak = datap["analysis"][self.typean]["fix_widthsecpeak"] \
                if self.p_includesecpeak else None
        if self.p_includesecpeak is None:
            self.p_includesecpeak = [False for ipt in range(self.p_nptbins)]
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
        self.p_sigmamb = datap["ml"]["opt"]["sigma_MB"]
        self.p_br = datap["ml"]["opt"]["BR"]

        self.d_valevtdata = valdata
        self.d_valevtmc = valmc

        self.f_evtvaldata = os.path.join(self.d_valevtdata, self.n_evtvalroot)
        self.f_evtvalmc = os.path.join(self.d_valevtmc, self.n_evtvalroot)

        self.f_evtnorm = os.path.join(self.d_resultsallpdata, "correctionsweights.root")

        # Systematics
        syst_dict = datap["analysis"][self.typean].get("systematics", None)
        self.p_max_chisquare_ndf_syst = syst_dict["max_chisquare_ndf"] \
                if syst_dict is not None else None
        self.p_rebin_syst = syst_dict["rebin"] if syst_dict is not None else None
        self.p_fit_ranges_low_syst = syst_dict["massmin"] if syst_dict is not None else None
        self.p_fit_ranges_up_syst = syst_dict["massmax"] if syst_dict is not None else None
        self.p_bincount_sigma_syst = syst_dict["bincount_sigma"] if syst_dict is not None else None

        self.p_indexhpt = datap["analysis"]["indexhptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_sigmav0 = datap["analysis"]["sigmav0"]
        self.apply_weights = datap["analysis"][self.typean]["triggersel"]["weighttrig"]

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
        if self.init_fits_from not in  ["mc", "data"]:
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
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, mult_int_min, mult_int_max)

                suffix_write = "%s%d_%d_%s_%.2f_%.2f" % \
                               (self.v_var_binning, self.lpt_finbinmin[ipt],
                                self.lpt_finbinmax[ipt],
                                self.v_var2_binning, mult_int_min, mult_int_max)
                h_invmass_mc_init = lfile_mc.Get("hmass" + suffix)

                h_mc_init_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass_mc_init,
                                                                  self.p_rebin[ipt], -1)
                h_mc_init_rebin = TH1F()
                h_mc_init_rebin_.Copy(h_mc_init_rebin)
                h_mc_init_rebin.SetTitle("%.1f < #it{p}_{T} < %.1f (prob > %.2f)" \
                                         % (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt], \
                                            self.lpt_probcutfin[bin_id]))
                h_mc_init_rebin.GetXaxis().SetTitle("#it{M}_{inv} (GeV/#it{c}^{2})")
                h_mc_init_rebin.GetYaxis().SetTitle("Entries/(%.0f MeV/#it{c}^{2})" \
                                                    % (h_mc_init_rebin.GetBinWidth(1) * 1000))
                h_mc_init_rebin.GetYaxis().SetTitleOffset(1.1)

                mass_fitter_mc_init.append(AliHFInvMassFitter(h_mc_init_rebin, self.p_massmin[ipt],
                                                              self.p_massmax[ipt],
                                                              self.bkg_fmap[self.p_bkgfunc[ipt]],
                                                              self.sig_fmap[self.p_sgnfunc[ipt]]))

                # Force to go on with final fit although there is no estimated signal
                mass_fitter_mc_init[ipt].SetCheckSignalCountsAfterFirstFit(False)
                if self.p_dolike:
                    mass_fitter_mc_init[ipt].SetUseLikelihoodFit()
                mass_fitter_mc_init[ipt].SetInitialGaussianMean(mean_for_data)
                mass_fitter_mc_init[ipt].SetInitialGaussianSigma(sigma_for_data)
                mass_fitter_mc_init[ipt].SetNSigma4SideBands(self.p_exclude_nsigma_sideband)
                success = mass_fitter_mc_init[ipt].MassFitter(False)
                fit_status[imult][ipt]["init_MC"] = False
                if success == 1:
                    mean_for_data = mass_fitter_mc_init[ipt].GetMean()
                    sigma_for_data = mass_fitter_mc_init[ipt].GetSigma()
                    means_sigmas_init.insert(0, (2, mean_for_data, sigma_for_data))
                    if ipt not in have_summary_pt_bins:
                        means_init_mc_histos.SetBinContent(ipt + 1,
                                                           mass_fitter_mc_init[ipt].GetMean())
                        means_init_mc_histos.SetBinError(ipt + 1, \
                                mass_fitter_mc_init[ipt].GetMeanUncertainty())
                        sigmas_init_mc_histos.SetBinContent(ipt + 1,
                                                            mass_fitter_mc_init[ipt].GetSigma())
                        sigmas_init_mc_histos.SetBinError(ipt + 1, \
                                mass_fitter_mc_init[ipt].GetSigmaUncertainty())

                    fit_status[imult][ipt]["init_MC"] = True
                    if self.init_fits_from == "mc":
                        mean_case_user = mean_for_data
                        sigma_case_user = sigma_for_data
                else:
                    self.logger.error("Could not do initial fit on MC")

                canvas = TCanvas("fit_canvas_mc_init", suffix_write, 700, 700)
                mass_fitter_mc_init[ipt].DrawHere(canvas, self.p_nsigma_signal)
                canvas.SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                  "fittedplot_integrated_mc", "eps",
                                                  None, suffix_write))
                canvas.Close()
                canvas_init_mc.cd(ipt+1)
                mass_fitter_mc_init[ipt].DrawHere(gPad, self.p_nsigma_signal)

                # Now, try also for data
                histname = "hmass"
                if self.apply_weights is True:
                    histname = "h_invmass_weight"
                    self.logger.info("*********** I AM USING WEIGHTED HISTOGRAMS")
                # Weighted histograms onnly for data at the moment
                h_invmass_init = lfile.Get(histname + suffix)
                h_data_init_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass_init,
                                                                    self.p_rebin[ipt], -1)
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
                if self.p_includesecpeak[ipt]:
                    mass_fitter_data_init[ipt].IncludeSecondGausPeak(self.p_masssecpeak,
                                                                     self.p_fix_masssecpeak,
                                                                     self.p_widthsecpeak,
                                                                     self.p_fix_widthsecpeak)
                success = mass_fitter_data_init[ipt].MassFitter(False)
                fit_status[imult][ipt]["init_data"] = False
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
                        fit_status[imult][ipt]["init_data"] = True
                        if self.init_fits_from == "data":
                            mean_case_user = mass_fitter_data_init[ipt].GetMean()
                            sigma_case_user = mass_fitter_data_init[ipt].GetSigma()

                canvas = TCanvas("fit_canvas_data_init", suffix_write, 700, 700)
                mass_fitter_data_init[ipt].DrawHere(canvas, self.p_nsigma_signal)
                canvas.SaveAs(self.make_file_path(self.d_resultsallpdata,
                                                  "fittedplot_integrated", "eps",
                                                  None, suffix_write))
                canvas.Close()
                canvas_init_data.cd(ipt+1)
                mass_fitter_data_init[ipt].DrawHere(gPad, self.p_nsigma_signal)

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
                if mean_case_user is not None:
                    fit_cases.insert(0, (0, mean_case_user, sigma_case_user, self.p_fixingaussigma))
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
                h_invmass_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass, self.p_rebin[ipt], -1)
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
                                                                     self.p_rebin[ipt], -1)
                h_invmass_mc_rebin = TH1F()
                h_invmass_mc_rebin_.Copy(h_invmass_mc_rebin)
                success = 0

                fit_status[imult][ipt]["data"] = {}

                # First try not fixing sigma for all cases (mean always floating)
                for case, mean, sigma, fix in fit_cases:

                    ifit = ifit + 1
                    mass_fitter.append(AliHFInvMassFitter(h_invmass_rebin, self.p_massmin[ipt],
                                                          self.p_massmax[ipt],
                                                          self.bkg_fmap[self.p_bkgfunc[ipt]],
                                                          self.sig_fmap[self.p_sgnfunc[ipt]]))

                    # Force to go on with final fit although there is no estimated signal
                    mass_fitter[ipt].SetCheckSignalCountsAfterFirstFit(False)
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

                        #h_invmass_refl = AliVertexingHFUtils.RebinHisto(
                        #    lfile_mc.Get("hmass_refl" + suffix), self.p_rebin[ipt], -1)
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
                    if self.p_includesecpeak[ipt]:
                        mass_fitter[ifit].IncludeSecondGausPeak(self.p_masssecpeak,
                                                                self.p_fix_masssecpeak,
                                                                self.p_widthsecpeak,
                                                                self.p_fix_widthsecpeak)
                    fit_status[imult][ipt]["data"]["fix"] = fix
                    fit_status[imult][ipt]["data"]["case"] = case
                    success = mass_fitter[ifit].MassFitter(False)
                    if success == 1:
                        sigma_final = mass_fitter[ifit].GetSigma()
                        if minperc * sigma < sigma_final < maxperc * sigma:
                            break
                        self.logger.warning("Free fit succesful, but bad sigma. Skipped!")

                fit_status[imult][ipt]["data"]["success"] = success


                canvas = TCanvas("fit_canvas", suffix, 700, 700)
                mass_fitter[ifit].DrawHere(canvas, self.p_nsigma_signal)
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

                    sigmas_histos[imult].SetBinContent(ipt + 1, sigma_fit)
                    sigmas_histos[imult].SetBinError(ipt + 1, \
                                                     mass_fitter[ifit].GetSigmaUncertainty())

                    # Residual plot
                    h_pulls = h_invmass_rebin.Clone(f"{h_invmass_rebin.GetName()}_pull")
                    h_residual_trend = \
                            h_invmass_rebin.Clone(f"{h_invmass_rebin.GetName()}_residual_trend")
                    h_pulls_trend = \
                            h_invmass_rebin.Clone(f"{h_invmass_rebin.GetName()}_pulls_trend")
                    h_sig_sub = mass_fitter[ifit].GetOverBackgroundResidualsAndPulls( \
                            h_pulls, h_residual_trend, h_pulls_trend, self.p_massmin[ipt],
                            self.p_massmax[ipt])

                    c_res = TCanvas('cRes', 'The Fit Canvas', 800, 800)
                    c_res.cd()
                    h_sig_sub.Draw()
                    c_res.SaveAs(self.make_file_path(self.d_resultsallpdata, "residual", "eps",
                                                     None, suffix_write))
                    c_res.Close()

                else:
                    self.logger.error("Fit failed for suffix %s", suffix_write)

                # Write fitters to file
                fit_root_dir = fileout.mkdir(suffix)
                fit_root_dir.WriteObject(mass_fitter_mc_init[ipt], "fitter_mc_init")
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


            del mass_fitter_mc_init[:]
            del mass_fitter_data_init[:]

        del mass_fitter[:]

        # Write the fit status dict
        dump_yaml_from_dict(fit_status, self.make_file_path(self.d_resultsallpdata, "fit_status",
                                                            "yaml"))
        # Yields summary plot
        cYields = TCanvas('cYields', 'The Fit Canvas')
        cYields.SetCanvasSize(1900, 1500)
        cYields.SetWindowSize(500, 500)
        cYields.SetLogy()

        legyield = TLegend(.5, .65, .7, .85)
        legyield.SetBorderSize(0)
        legyield.SetFillColor(0)
        legyield.SetFillStyle(0)
        legyield.SetTextFont(42)
        legyield.SetTextSize(0.035)

        # Means summary plot
        cMeans = TCanvas('cMeans', 'Mean summary')
        cMeans.SetCanvasSize(1900, 1500)
        cMeans.SetWindowSize(500, 500)

        leg_means = TLegend(.5, .65, .7, .85)
        leg_means.SetBorderSize(0)
        leg_means.SetFillColor(0)
        leg_means.SetFillStyle(0)
        leg_means.SetTextFont(42)
        leg_means.SetTextSize(0.035)

        # Means summary plot
        cSigmas = TCanvas('cSigma', 'Sigma summary')
        cSigmas.SetCanvasSize(1900, 1500)
        cSigmas.SetWindowSize(500, 500)

        leg_sigmas = TLegend(.5, .65, .7, .85)
        leg_sigmas.SetBorderSize(0)
        leg_sigmas.SetFillColor(0)
        leg_sigmas.SetFillStyle(0)
        leg_sigmas.SetTextFont(42)
        leg_sigmas.SetTextSize(0.035)

        for imult in range(self.p_nbin2):
            legstring = "%.1f #leq %s < %.1f" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            # Draw yields
            cYields.cd()
            yieldshistos[imult].SetMinimum(1)
            yieldshistos[imult].SetMaximum(1e6)
            yieldshistos[imult].SetLineColor(imult+1)
            yieldshistos[imult].Draw("same")
            legyield.AddEntry(yieldshistos[imult], legstring, "LEP")
            yieldshistos[imult].GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            yieldshistos[imult].GetYaxis().SetTitle("Uncorrected yields %s %s (1/GeV)" \
                    % (self.p_latexnmeson, self.typean))

            cMeans.cd()
            means_histos[imult].SetMinimum(0.999 * mean_min)
            means_histos[imult].SetMaximum(1.001 * mean_max)
            means_histos[imult].SetLineColor(imult+1)
            means_histos[imult].Draw("same")
            leg_means.AddEntry(means_histos[imult], legstring, "LEP")
            means_histos[imult].GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            means_histos[imult].GetYaxis().SetTitle("#mu_{fit} %s %s" \
                    % (self.p_latexnmeson, self.typean))

            cSigmas.cd()
            sigmas_histos[imult].SetMinimum(0.99 * sigma_min)
            sigmas_histos[imult].SetMaximum(1.01 * sigma_max)
            sigmas_histos[imult].SetLineColor(imult+1)
            sigmas_histos[imult].Draw("same")
            leg_sigmas.AddEntry(sigmas_histos[imult], legstring, "LEP")
            sigmas_histos[imult].GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            sigmas_histos[imult].GetYaxis().SetTitle("#sigma_{fit} %s %s" \
                    % (self.p_latexnmeson, self.typean))

        cYields.cd()
        legyield.Draw()
        save_name = self.make_file_path(self.d_resultsallpdata, "Yields", "eps", None,
                                        [self.case, self.typean])
        cYields.SaveAs(save_name)
        cYields.Close()

        cMeans.cd()
        leg_means.Draw()
        save_name = self.make_file_path(self.d_resultsallpdata, "Means", "eps", None,
                                        [self.case, self.typean])
        cMeans.SaveAs(save_name)
        cMeans.Close()

        cSigmas.cd()
        leg_sigmas.Draw()
        save_name = self.make_file_path(self.d_resultsallpdata, "Sigmas", "eps", None,
                                        [self.case, self.typean])
        cSigmas.SaveAs(save_name)
        cSigmas.Close()

        # Plot the initialized means and sigma for MC and data
        # Yields summary plot
        def plot_fast(canvas, histo, legend, label_x, label_y, save_path):
            canvas.cd()
            canvas.SetCanvasSize(1900, 1500)
            canvas.SetWindowSize(500, 500)
            legend.SetBorderSize(0)
            legend.SetFillColor(0)
            legend.SetFillStyle(0)
            legend.SetTextFont(42)
            legend.SetTextSize(0.035)
            histo.GetXaxis().SetTitle(label_x)
            histo.GetYaxis().SetTitle(label_y)
            histo.Draw()
            legend_string = f"{mult_int_min} < {self.v_var2_binning} < {mult_int_max}"
            legend.AddEntry(histo, legend_string)
            legend.Draw()
            canvas.SaveAs(save_path)

        c_init = TCanvas('c_init', 'The Fit Canvas')
        legend = TLegend(.5, .65, .7, .85)
        save_name = self.make_file_path(self.d_resultsallpdata, "Means_mc_init", "eps", None,
                                        [self.case, self.typean])
        plot_fast(c_init, means_init_mc_histos, legend, "#it{p}_{T} (GeV/c)", "#mu_{init_mc}",
                  save_name)
        c_init.Close()

        c_init = TCanvas('c_init', 'The Fit Canvas')
        legend = TLegend(.5, .65, .7, .85)
        save_name = self.make_file_path(self.d_resultsallpdata, "Sigmas_mc_init", "eps", None,
                                        [self.case, self.typean])
        plot_fast(c_init, sigmas_init_mc_histos, legend, "#it{p}_{T} (GeV/c)", "#sigma_{init_mc}",
                  save_name)
        c_init.Close()

        c_init = TCanvas('c_init', 'The Fit Canvas')
        legend = TLegend(.5, .65, .7, .85)
        save_name = self.make_file_path(self.d_resultsallpdata, "Means_data_init", "eps", None,
                                        [self.case, self.typean])
        plot_fast(c_init, means_init_data_histos, legend, "#it{p}_{T} (GeV/c)", "#mu_{init_data}",
                  save_name)
        c_init.Close()

        c_init = TCanvas('c_init', 'The Fit Canvas')
        legend = TLegend(.5, .65, .7, .85)
        save_name = self.make_file_path(self.d_resultsallpdata, "Sigmas_data_init", "eps", None,
                                        [self.case, self.typean])
        plot_fast(c_init, sigmas_init_data_histos, legend, "#it{p}_{T} (GeV/c)",
                  "#sigma_{init_data}", save_name)
        c_init.Close()

        fileout.Close()
        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-branches
    def yield_syst(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        # First check if systematics can be computed by checking if parameters are set
        if self.p_rebin_syst is None:
            self.logger.error("Parameters for systematics calculation not set. Skip...")
            return


        # We need both the mass histograms and the nominal fits. First check, whether they exist
        func_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename,
                                            "root", None, [self.case, self.typean])
        if not os.path.exists(func_filename) or not os.path.exists(self.n_filemass):
            self.logger.fatal("Cannot find ROOT files with nominal fits and raw " \
                              "mass histograms at %s and %s, respectively", func_filename,
                              self.n_filemass)

        # Open files with nominal fits and raw mass histograms
        lfile = TFile.Open(self.n_filemass)
        func_file = TFile.Open(func_filename, "READ")

        # Variations written to dedicated file
        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_syst_filename,
                                           "root", None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")

        # One fitter to extract the respective nominal fit and one used for the variation
        mass_fitter_nominal = Fitter()
        mass_fitter_syst = Fitter()

        # Keep all additional objects in a plot until it has been saved. Otherwise,
        # they will be deleted by Python as soon as something goes out of scope
        tmp_plot_objects = []

        color_mt_fit = kBlue
        color_mt_bincount = kGreen + 2
        color_nominal = kBlack

        # Used here internally for plotting
        def draw_histos(pad, draw_legend, nominals, hori_vert,
                        histos, plot_options, colors, save_path=None):
            pad.cd()
            if draw_legend:
                legend = TLegend(0.12, 0.7, 0.48, 0.88)
                # pylint: disable=cell-var-from-loop
                tmp_plot_objects.append(legend)
                legend.SetLineWidth(0)
                legend.SetTextSize(0.02)

            lines = []
            x_min = histos[0].GetXaxis().GetXmin()
            x_max = histos[0].GetXaxis().GetXmax()
            y_max = histos[0].GetMaximum()
            for i, h in enumerate(histos):
                x_min = min(h.GetXaxis().GetXmin(), x_min)
                x_max = max(h.GetXaxis().GetXmax(), x_max)
                y_max = max(h.GetMaximum(), y_max)
                h.SetFillStyle(0)
                h.SetStats(0)
                h.SetLineColor(colors[i])
                h.SetFillColor(colors[i])
                h.SetMarkerColor(colors[i])
                h.SetLineWidth(1)
            plot_options = " ".join(["same", plot_options])
            for h, nom in zip(histos, nominals):
                if draw_legend:
                    legend.AddEntry(h, h.GetName())
                h.GetXaxis().SetRangeUser(x_min, x_max)
                h.GetYaxis().SetRangeUser(0., 1.5 * y_max)
                h.Draw(plot_options)
                if hori_vert is not None:
                    if hori_vert == "v":
                        # vertical lines
                        lines.append(TLine(nom, 0., nom, 1.2 * y_max))
                    else:
                        # horizontal lines
                        lines.append(TLine(x_min, nom, x_max, nom))
                    lines[-1].SetLineColor(h.GetLineColor())
                    lines[-1].SetLineWidth(1)
                    lines[-1].Draw("same")
            if draw_legend:
                legend.Draw("same")
            # pylint: disable=cell-var-from-loop
            tmp_plot_objects.append(lines)
            pad.Update()
            if save_path is not None:
                pad.SaveAs(save_path)

        for imult in range(self.p_nbin2):
            # Prepare lists summarising multi trial results in bins of pT
            # Assumin pT bins don't overlap
            array_pt = []
            for low, up in zip(self.lpt_finbinmin, self.lpt_finbinmax):
                if low not in array_pt:
                    array_pt.append(low)
                if up not in array_pt:
                    array_pt.append(up)
            array_pt = array("d", array_pt)
            histo_mt_fit_pt = TH1F("histo_mt_fit_pt", "", len(array_pt) - 1, array_pt)
            histo_mt_bincount_pt = TH1F("histo_mt_bincount_pt", "", len(array_pt) - 1, array_pt)
            histo_nominal_pt = TH1F("histo_nominal_pt", "", len(array_pt) - 1, array_pt)

            fileout.cd()

            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                h_invmass = lfile.Get("hmass" + suffix)

                # Get the nominal fit values to compare to
                mass_fitter_nominal.load(func_file.GetDirectory(suffix), True)
                if not mass_fitter_nominal.fit_success:
                    continue
                yield_nominal = mass_fitter_nominal.yield_sig
                yield_err_nominal = mass_fitter_nominal.yield_sig_err
                bincount_nominal, bincount_err_nominal = \
                        mass_fitter_nominal.bincount(self.p_nsigma_signal)
                mean_nominal = mass_fitter_nominal.mean_fit
                sigma_nominal = mass_fitter_nominal.sigma_fit
                chisquare_ndf_nominal = mass_fitter_nominal.tot_fit_func.GetNDF()
                chisquare_ndf_nominal = mass_fitter_nominal.tot_fit_func.GetChisquare() / \
                        chisquare_ndf_nominal if chisquare_ndf_nominal > 0. else 0.

                # Collect variation values
                yields_syst = []
                yields_syst_err = []
                bincounts_syst = []
                bincounts_syst_err = []
                means_syst = []
                sigmas_syst = []
                chisquares_syst = []

                # Crazy nested loop
                # For now only go for fixed sigma and free mean as this is what
                # we do for the nominal
                for fix_mean in [False]:
                    for fix_sigma in [True]:
                        for rebin in self.p_rebin_syst:
                            for fr_up in self.p_fit_ranges_up_syst:
                                for fr_low in self.p_fit_ranges_low_syst:
                                    mass_fitter_syst.initialize(h_invmass, self.p_sgnfunc[ipt],
                                                                self.p_bkgfunc[ipt], rebin,
                                                                mass_fitter_nominal.mean_fit,
                                                                mass_fitter_nominal.sigma_fit,
                                                                fix_mean, fix_sigma,
                                                                self.p_exclude_nsigma_sideband,
                                                                self.p_nsigma_signal, fr_low,
                                                                fr_up)

                                    mass_fitter_syst.do_likelihood()
                                    success = mass_fitter_syst.fit()
                                    chisquare_ndf_syst = mass_fitter_syst.tot_fit_func.GetNDF()
                                    chisquare_ndf_syst = \
                                            mass_fitter_syst.tot_fit_func.GetChisquare() / \
                                            chisquare_ndf_syst if chisquare_ndf_syst > 0. else 0.
                                    # Only if the fit was successful and in case the chisquare does
                                    # exceed the nominal too much we extract the values from this
                                    # variation
                                    if success and \
                                            0. < chisquare_ndf_syst < \
                                            self.p_max_chisquare_ndf_syst:
                                        rawYield = mass_fitter_syst.yield_sig
                                        rawYieldErr = mass_fitter_syst.yield_sig_err
                                        yields_syst.append(rawYield)
                                        yields_syst_err.append(rawYieldErr)
                                        means_syst.append(mass_fitter_syst.mean_fit)
                                        sigmas_syst.append(mass_fitter_syst.sigma_fit)
                                        chisquares_syst.append(chisquare_ndf_syst)
                                        for sigma in self.p_bincount_sigma_syst:
                                            rawBC, rawBC_err = mass_fitter_syst.bincount(sigma)
                                            if rawBC is not None:
                                                bincounts_syst.append(rawBC)
                                                bincounts_syst_err.append(rawBC_err)

                fileout.cd()
                # Each pT and secondary binning gets its own directory in the output ROOT file
                root_dir = fileout.mkdir(suffix)
                root_dir.cd()
                # Let's use the same binning for fitted and bincount values
                min_y = min(min(yields_syst), min(bincounts_syst)) if yields_syst else 0
                max_y = max(max(yields_syst), max(bincounts_syst)) if yields_syst else 1
                histo_yields = TH1F("yields_syst", "", 25, 0.9 * min_y + 1, 1.1 * max_y + 1)
                histo_bincounts = TH1F("bincounts_syst", "", 25, 0.9 * min_y + 1, 1.1 * max_y + 1)

                # Let's use the same binning for fitted and bincount values
                min_y = min(min(yields_syst_err), min(bincounts_syst_err)) if yields_syst else 0
                max_y = max(max(yields_syst_err), max(bincounts_syst_err)) if yields_syst else 1
                histo_yields_err = TH1F("yields_syst_err", "", 30, 0.9 * min_y + 1,
                                        1.1 * max_y + 1)
                histo_bincounts_err = TH1F("bincounts_syst_err", "", 30, 0.9 * min_y + 1,
                                           1.1 * max_y + 1)

                # Means, sigmas, chi squares
                histo_means = TH1F("means_syst", "", len(means_syst), 0.5, len(means_syst) + 0.5)
                histo_means.SetMarkerStyle(2)
                histo_sigmas = TH1F("sigmas_syst", "", len(sigmas_syst), 0.5,
                                    len(sigmas_syst) + 0.5)
                histo_sigmas.SetMarkerStyle(2)
                histo_chisquares = TH1F("chisquares_syst", "", len(chisquares_syst), 0.5,
                                        len(chisquares_syst) + 0.5)
                histo_chisquares.SetMarkerStyle(2)
                # Fill the histograms if there is at least one good fit from the variation
                if yields_syst:
                    i_bin = 1
                    for y, y_err, bc, bc_err, m, s, cs in zip(yields_syst,
                                                              yields_syst_err,
                                                              bincounts_syst,
                                                              bincounts_syst_err,
                                                              means_syst,
                                                              sigmas_syst,
                                                              chisquares_syst):
                        histo_yields.Fill(y)
                        histo_yields_err.Fill(y_err)
                        histo_means.SetBinContent(i_bin, m)
                        histo_sigmas.SetBinContent(i_bin, s)
                        histo_chisquares.SetBinContent(i_bin, cs)
                        i_bin += 1
                    for bc, bc_err in zip(bincounts_syst, bincounts_syst_err):
                        histo_bincounts.Fill(bc)
                        histo_bincounts_err.Fill(bc_err)
                else:
                    self.logger.error("No systematics could be derived for %s", suffix)

                # First, write the histgrams for potential re-usage
                histo_yields.Write()
                histo_yields_err.Write()
                histo_bincounts.Write()
                histo_bincounts_err.Write()
                histo_means.Write()
                histo_sigmas.Write()
                histo_chisquares.Write()

                # Draw into canvas
                canvas = TCanvas("syst_canvas", "", 1400, 800)
                canvas.Divide(3, 2)
                pad = canvas.cd(5)
                filename = self.make_file_path(self.d_resultsallpdata, self.yields_syst_filename,
                                               "eps", None, suffix)
                histo_yields.GetXaxis().SetTitle("yield")
                histo_yields.GetYaxis().SetTitle("# entries")
                draw_histos(pad, True, [yield_nominal, bincount_nominal], "v",
                            [histo_yields, histo_bincounts], "hist",
                            [color_mt_fit, color_mt_bincount], filename)
                pad = canvas.cd(4)
                filename = self.make_file_path(self.d_resultsallpdata, self.yields_syst_filename,
                                               "eps", None, ["err", suffix])
                histo_yields_err.GetXaxis().SetTitle("yield err")
                histo_yields_err.GetYaxis().SetTitle("# entries")
                draw_histos(pad, True,
                            [yield_err_nominal, bincount_err_nominal], "v",
                            [histo_yields_err, histo_bincounts_err], "hist",
                            [color_mt_fit, color_mt_bincount], filename)
                pad = canvas.cd(1)
                filename = self.make_file_path(self.d_resultsallpdata, "means_syst", "eps",
                                               None, suffix)
                histo_means.GetXaxis().SetTitle("trial")
                histo_means.GetYaxis().SetTitle("#mu")
                draw_histos(pad, False, [mean_nominal], None,
                            [histo_means], "p", [color_mt_fit], filename)
                pad = canvas.cd(2)
                filename = self.make_file_path(self.d_resultsallpdata, "sigmas_syst", "eps",
                                               None, suffix)
                histo_sigmas.GetXaxis().SetTitle("trial")
                histo_sigmas.GetYaxis().SetTitle("#sigma")
                draw_histos(pad, False, [sigma_nominal], None,
                            [histo_sigmas], "p", [color_mt_fit], filename)
                pad = canvas.cd(3)
                filename = self.make_file_path(self.d_resultsallpdata, "chisquares_syst", "eps",
                                               None, suffix)
                histo_chisquares.GetXaxis().SetTitle("trial")
                histo_chisquares.GetYaxis().SetTitle("#chi^{2}/NDF")
                draw_histos(pad, False, [chisquare_ndf_nominal], None,
                            [histo_chisquares], "p", [color_mt_fit], filename)


                def create_text(pos_x, pos_y, text, color=kBlack):
                    root_text = TText(pos_x, pos_y, text)
                    root_text.SetTextSize(0.03)
                    root_text.SetTextColor(color)
                    root_text.SetNDC()
                    return root_text
                # Add some numbers
                pad = canvas.cd(6)

                root_texts = []
                root_texts.append(create_text(0.05, 0.93, "Fit yields"))

                mean_fit = histo_yields.GetMean()
                rms_fit = histo_yields.GetRMS()
                unc_mean = rms_fit / mean_fit * 100 if mean_fit > 0. else 0.
                min_val = histo_yields.GetBinLowEdge(histo_yields.FindFirstBinAbove())
                last_bin = histo_yields.FindFirstBinAbove()
                max_val = histo_yields.GetBinLowEdge(last_bin) + histo_yields.GetBinWidth(last_bin)
                diff_min_max = (max_val - min_val) / sqrt(12)
                unc_min_max = diff_min_max / mean_fit * 100 if mean_fit > 0. else 0.

                root_texts.append(create_text(0.05, 0.88, f"nominal = {yield_nominal:.0f}"))

                root_texts.append(create_text(0.05, 0.83,
                                              f"MEAN = " \
                                              f"{mean_fit:.0f}",
                                              color_mt_fit))

                root_texts.append(create_text(0.05, 0.78,
                                              f"RMS = " \
                                              f"{rms_fit:.0f} ({unc_mean:.2f}%)", color_mt_fit))

                root_texts.append(create_text(0.05, 0.73,
                                              f"MIN = {min_val:.0f}" \
                                              f"    " \
                                              f"MAX = {max_val:.0f}", color_mt_fit))

                root_texts.append(create_text(0.05, 0.68,
                                              f"(MAX - MIN) / sqrt(12) = " \
                                              f"{diff_min_max:.0f} ({unc_min_max:.2f}%)",
                                              color_mt_fit))

                mean_bc = histo_bincounts.GetMean()
                rms_bc = histo_bincounts.GetRMS()
                unc_mean = rms_bc / mean_bc * 100 if mean_bc > 0. else 0.
                min_val = histo_bincounts.GetBinLowEdge(histo_bincounts.FindFirstBinAbove())
                last_bin = histo_bincounts.FindFirstBinAbove()
                max_val = histo_bincounts.GetBinLowEdge(last_bin) + \
                        histo_bincounts.GetBinWidth(last_bin)
                diff_min_max = (max_val - min_val) / sqrt(12)
                unc_min_max = diff_min_max / mean_bc * 100 if mean_bc > 0. else 0.

                root_texts.append(create_text(0.05, 0.58, "Bin count yields"))

                root_texts.append(create_text(0.05, 0.53,
                                              f"nominal = {bincount_nominal:.0f}"))

                root_texts.append(create_text(0.05, 0.48,
                                              f"MEAN = " \
                                              f"{mean_bc:.0f}", color_mt_bincount))

                root_texts.append(create_text(0.05, 0.43,
                                              f"RMS = " \
                                              f"{rms_bc:.0f}", color_mt_bincount))

                root_texts.append(create_text(0.05, 0.38,
                                              f"MIN = {min_val:.0f}" \
                                              f"    " \
                                              f"MAX = {max_val:.0f}", color_mt_bincount))

                root_texts.append(create_text(0.05, 0.33,
                                              f"(MAX - MIN) / sqrt(12) = " \
                                              f"{diff_min_max:.0f} ({unc_min_max:.2f}%)",
                                              color_mt_bincount))

                root_texts.append(create_text(0.05, 0.23, "Deviations"))

                diff = yield_nominal - mean_fit
                diff_ratio = diff / yield_nominal * 100 if yield_nominal > 0. else 0.
                root_texts.append(create_text(0.05, 0.18,
                                              f"yield fit (nominal) - yield fit " \
                                              f"(multi) = {diff:.0f} " \
                                              f"({diff_ratio:.2f}%)", kRed + 2))

                diff = yield_nominal - mean_bc
                diff_ratio = diff / yield_nominal * 100 if yield_nominal > 0. else 0.
                root_texts.append(create_text(0.05, 0.13,
                                              f"yield fit (nominal) - yield " \
                                              f"bincount (multi) = " \
                                              f"{diff:.0f} " \
                                              f"({diff_ratio:.2f}%)", kRed + 2))

                for t in root_texts:
                    t.Draw()

                filename = self.make_file_path(self.d_resultsallpdata, "all_syst", "eps",
                                               None, suffix)
                canvas.SaveAs(filename)
                canvas.Close()

                # Put in final histogram
                histo_mt_fit_pt.SetBinContent(ipt + 1, mean_fit)
                histo_mt_fit_pt.SetBinError(ipt + 1, rms_fit)
                histo_mt_bincount_pt.SetBinContent(ipt + 1, mean_bc)
                histo_mt_bincount_pt.SetBinError(ipt + 1, rms_bc)
                histo_nominal_pt.SetBinContent(ipt + 1, yield_nominal)
                histo_nominal_pt.SetBinError(ipt + 1, yield_err_nominal)

            # Draw into canvas
            histo_mt_fit_pt.SetMarkerStyle(20)
            histo_mt_fit_pt.SetFillStyle(0)
            histo_mt_fit_pt.GetYaxis().SetTitleSize(20)
            histo_mt_fit_pt.GetYaxis().SetTitleFont(43)
            histo_mt_fit_pt.GetYaxis().SetTitleOffset(1.55)
            histo_mt_fit_pt.GetYaxis().SetTitle("yield")
            histo_mt_fit_pt.GetYaxis().SetLabelSize(15)
            histo_mt_fit_pt.GetYaxis().SetLabelFont(43)
            histo_mt_bincount_pt.SetMarkerStyle(20)
            histo_mt_bincount_pt.SetFillStyle(0)
            histo_nominal_pt.SetMarkerStyle(20)
            histo_nominal_pt.SetFillStyle(0)

            histo_ratio_mt_fit = histo_mt_fit_pt.Clone(histo_mt_fit_pt.GetName() + "_ratio")
            histo_ratio_mt_fit.Divide(histo_nominal_pt)
            histo_ratio_mt_fit.GetYaxis().SetTitleSize(20)
            histo_ratio_mt_fit.GetYaxis().SetTitleFont(43)
            histo_ratio_mt_fit.GetYaxis().SetTitleOffset(1.55)
            histo_ratio_mt_fit.GetYaxis().SetLabelSize(15)
            histo_ratio_mt_fit.GetYaxis().SetLabelFont(43)
            histo_ratio_mt_fit.GetYaxis().SetTitle("multi / nominal")
            histo_ratio_mt_fit.GetXaxis().SetTitleSize(20)
            histo_ratio_mt_fit.GetXaxis().SetLabelSize(15)
            histo_ratio_mt_fit.GetXaxis().SetLabelFont(43)
            histo_ratio_mt_fit.GetXaxis().SetTitleFont(43)
            histo_ratio_mt_fit.GetXaxis().SetTitleOffset(4.)
            histo_ratio_mt_fit.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")

            histo_ratio_mt_bincount = \
                    histo_mt_bincount_pt.Clone(histo_mt_bincount_pt.GetName() + "_ratio")
            histo_ratio_mt_bincount.Divide(histo_nominal_pt)

            canvas_mt = TCanvas("some_canvas", "", 800, 800)

            canvas_mt.cd()
            pad_up = TPad("pad_up", "", 0., 0.3, 1., 1.)
            pad_up.SetBottomMargin(0.)
            pad_up.Draw()

            draw_histos(pad_up, True, [0, 0, 0], None,
                        [histo_mt_fit_pt, histo_mt_bincount_pt, histo_nominal_pt], "e2p",
                        [color_mt_fit, color_mt_bincount, color_nominal])

            text_box = TPaveText(0.5, 0.7, 1., 0.89, "NDC")
            text_box.SetBorderSize(0)
            text_box.SetFillStyle(0)
            text_box.SetTextAlign(11)
            text_box.SetTextSize(20)
            text_box.SetTextFont(43)
            text_box.AddText(f"{self.p_latexnmeson} | analysis type: {self.typean}")
            text_box.AddText(f"{self.lvar2_binmin[imult]} #leq {self.v_var2_binning} < " \
                             f"{self.lvar2_binmax[imult]}")
            pad_up.cd()
            text_box.Draw()
            canvas_mt.cd()
            pad_ratio = TPad("pad_ratio", "", 0., 0.05, 1., 0.3)
            pad_ratio.SetTopMargin(0.)
            pad_ratio.SetBottomMargin(0.3)
            pad_ratio.Draw()
            draw_histos(pad_ratio, False, [0, 0], None,
                        [histo_ratio_mt_fit, histo_ratio_mt_bincount], "p",
                        [color_mt_fit, color_mt_bincount])
            line_unity = TLine(histo_ratio_mt_bincount.GetXaxis().GetXmin(), 1.,
                               histo_ratio_mt_bincount.GetXaxis().GetXmax(), 1.)
            line_unity.Draw()

            pad_ratio.cd()
            # Reset the range of the ratio plot
            y_max_ratio = 1.3
            y_min_ratio = 0.7
            histo_ratio_mt_fit.GetYaxis().SetRangeUser(y_min_ratio, y_max_ratio)
            histo_ratio_mt_bincount.GetYaxis().SetRangeUser(y_min_ratio, y_max_ratio)

            def replace_with_arrows(histo, center_value, min_value, max_value):
                arrows = []
                for i in range(1, histo.GetNbinsX() + 1):
                    content = histo.GetBinContent(i)
                    if content < min_value:
                        histo.SetBinContent(i, 0.)
                        histo.SetBinError(i, 0.)
                        bin_center = histo.GetBinCenter(i)
                        arrows.append(TArrow(bin_center, center_value, bin_center, min_value, 0.04))
                    elif content > max_value:
                        histo.SetBinContent(i, 0.)
                        histo.SetBinError(i, 0.)
                        bin_center = histo.GetBinCenter(i)
                        arrows.append(TArrow(bin_center, center_value, bin_center, max_value, 0.04))
                return arrows

            arrows_fit = replace_with_arrows(histo_ratio_mt_fit, 1., y_min_ratio, y_max_ratio)
            for a in arrows_fit:
                a.SetLineColor(color_mt_fit)
                a.SetLineStyle(2)
                a.Draw()

            arrows_bc = replace_with_arrows(histo_ratio_mt_bincount, 1., y_min_ratio, y_max_ratio)
            for a in arrows_bc:
                a.SetLineColor(color_mt_bincount)
                a.SetLineStyle(7)
                a.Draw()

            filename = self.make_file_path(self.d_resultsallpdata, "multi_trial_summary", "eps",
                                           None, [f"{self.lvar2_binmin[imult]:.2f}",
                                                  f"{self.lvar2_binmax[imult]:.2f}"])


            canvas_mt.SaveAs(filename)
            canvas_mt.Close()

        fileout.Write()
        fileout.Close()

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
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
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
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
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
        his_sim_z_fd_2d = his_sim_z_fd.Project3D("yz") # final feed-down histogram
        # x axis = z, y axis = pt_jet
        his_sim_z_fd_2d.Reset()
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
            for i_z in range(his_sim_z_fd_2d.GetNbinsX()):
                bin_z = i_z + 1
                his_sim_z_fd.GetYaxis().SetRange(bin_ptjet, bin_ptjet)
                his_sim_z_fd.GetZaxis().SetRange(bin_z, bin_z)
                his_sim_ptc_fd = his_sim_z_fd.Project3D("x") # pt_cand
                his_sim_ptc_fd.Multiply(his_eff_ratio)
                his_sim_z_fd_2d.SetBinContent(bin_z, bin_ptjet, his_sim_ptc_fd.Integral())
            can_ff_fd = TCanvas("can_fd_z%d" % i_ptjet, "Feeddown FF", 800, 600)
            his_ff_fd = his_sim_z_fd_2d.ProjectionX("ff%d" % i_ptjet, bin_ptjet, bin_ptjet, "e")
            his_ff_fd.GetYaxis().SetTitle("#frac{1}{#it{N}_{jet}} #frac{d#it{N}}{d#it{z}}")
            his_ff_fd.GetYaxis().SetTitleOffset(1.6)
            his_ff_fd.GetYaxis().SetTitleFont(42)
            his_ff_fd.GetYaxis().SetLabelFont(42)
            his_ff_fd.Draw()
            can_ff_fd.SetLeftMargin(0.15)
            can_ff_fd.SaveAs("%s/Feeddown-z-effscaled_%s%s%s.eps" % (self.d_resultsallpmc, \
                            self.case, self.typean, i_ptjet))

        # TODO: Building the response matrix and smearing.
        file_out.cd()
        his_sim_z_fd_2d.Write("fd_z_ptjet")

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
    def calculate_norm(filename, trigger, var, multmin, multmax, doweight):
        fileout = TFile.Open(filename, "read")
        if not fileout:
            return -1
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
        return norm

    def makenormyields(self):
        gROOT.SetBatch(True)
        print("SSSSSSSSSSSSSSS")
        self.loadstyle()
        #self.test_aliphysics()
        #filedataval = TFile.Open(self.f_evtnorm)

        fileouteff = "%s/efficiencies%s%s.root" % \
                      (self.d_resultsallpmc, self.case, self.typean)
        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        gROOT.LoadMacro("HFPtSpectrum.C")
        from ROOT import HFPtSpectrum
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
            norm = -1
            norm = self.calculate_norm(self.f_evtnorm, self.triggerbit, \
                         self.v_var2_binning, self.lvar2_binmin[imult], \
                         self.lvar2_binmax[imult], self.apply_weights)
            print(self.apply_weights, self.lvar2_binmin[imult], self.lvar2_binmax[imult], norm)
#
#            hSelMult = filedataval.Get('sel_' + labelhisto)
#            hNoVtxMult = filedataval.Get('novtx_' + labelhisto)
#            hVtxOutMult = filedataval.Get('vtxout_' + labelhisto)
#
#            # normalisation based on multiplicity histograms
#            binminv = hSelMult.GetXaxis().FindBin(self.lvar2_binmin[imult])
#            binmaxv = hSelMult.GetXaxis().FindBin(self.lvar2_binmax[imult])
#
#            n_sel = hSelMult.Integral(binminv, binmaxv)
#            n_novtx = hNoVtxMult.Integral(binminv, binmaxv)
#            n_vtxout = hVtxOutMult.Integral(binminv, binmaxv)
#            norm = (n_sel + n_novtx) - n_novtx * n_vtxout / (n_sel + n_vtxout)
#
#            print('new normalization: ', norm, norm_old)
#
            # Now use the function we have just compiled above
            HFPtSpectrum(self.p_indexhpt, \
                "inputsCross/D0DplusDstarPredictions_13TeV_y05_all_300416_BDShapeCorrected.root", \
                fileouteff, namehistoeffprompt, namehistoefffeed, yield_filename, nameyield, \
                fileoutcrossmult, norm, self.p_sigmav0 * 1e12, self.p_fd_method, self.p_cctype)
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
        triggerlist = ["HighMultV0", "HighMultSPD", "HighMultV0"]
        varlist = ["v0m_corr", "n_tracklets_corr", "perc_v0m"]
        fileout_name = "%s/correctionsweights.root" % self.d_valevtdata
        fileout = TFile.Open(fileout_name, "recreate")
        fileout.cd()
        for i, trigger in enumerate(triggerlist):
            labeltriggerANDMB = "hbit%sANDINT7vs%s" % (triggerlist[i], varlist[i])
            labelMB = "hbitINT7vs%s" % varlist[i]
            labeltrigger = "hbit%svs%s" % (triggerlist[i], varlist[i])
            hden = filedata.Get(labelMB)
            hden.SetName("hmultINT7vs%s" % (varlist[i]))
            hden.Write()
            heff = filedata.Get(labeltriggerANDMB)
            if not heff or not hden:
                continue
            heff.Divide(heff, hden, 1.0, 1.0, "B")
            hratio = filedata.Get(labeltrigger)
            hmult = hratio.Clone("hmult%svs%s" % (triggerlist[i], varlist[i]))
            hmultweighted = hratio.Clone("hmultweighted%svs%s" % (triggerlist[i], varlist[i]))
            if not hratio:
                continue
            hratio.Divide(hratio, hden, 1.0, 1.0, "B")

            ctrigger = TCanvas('ctrigger%s' % trigger, 'The Fit Canvas')
            ctrigger.SetCanvasSize(3500, 2000)
            ctrigger.Divide(3, 2)



            ctrigger.cd(1)
            heff.SetMaximum(2.)
            heff.GetXaxis().SetTitle("offline %s" % varlist[i])
            heff.SetMinimum(0.)
            heff.GetYaxis().SetTitle("trigger efficiency from MB events")
            heff.SetLineColor(1)
            heff.Draw()
            heff.Write()

            ctrigger.cd(2)
            hratio.GetXaxis().SetTitle("offline %s" % varlist[i])
            hratio.GetYaxis().SetTitle("ratio triggered/MB")
            hratio.GetYaxis().SetTitleOffset(1.3)
            hratio.Write()
            hratio.SetLineColor(1)
            hratio.Draw()
            func = TF1("func_%s_%s" % (triggerlist[i], varlist[i]), \
                       "([0]/(1+TMath::Exp(-[1]*(x-[2]))))", 0, 1000)
            if i == 0:
                func.SetParameters(300, .1, 570)
                func.SetParLimits(1, 0., 10.)
                func.SetParLimits(2, 0., 1000.)
                func.SetRange(550., 1100.)
                func.SetLineWidth(1)
                hratio.Fit(func, "L", "", 550, 1100)
                func.Draw("same")
                func.SetLineColor(i+1)
            if i == 1:
                func.SetParameters(100, .1, 50)
                func.SetParLimits(1, 0., 10.)
                func.SetParLimits(2, 0., 200.)
                func.SetRange(45., 105)
                func.SetLineWidth(1)
                hratio.Fit(func, "L", "", 45, 105)
                func.SetLineColor(i+1)
            if i == 2:
                func.SetParameters(315, -30., .2)
                func.SetParLimits(1, -100., 0.)
                func.SetParLimits(2, 0., .5)
                func.SetRange(0., .15)
                func.SetLineWidth(1)
                hratio.Fit(func, "w", "", 0, .15)
                func.SetLineColor(i+1)
            func.Write()
            funcnorm = func.Clone("funcnorm_%s_%s" % (triggerlist[i], varlist[i]))
            funcnorm.FixParameter(0, funcnorm.GetParameter(0)/funcnorm.GetMaximum())
            funcnorm.Write()
            ctrigger.cd(3)
            maxhistx = 0
            if i == 0:
                minhistx = 300
                maxhistx = 1000
                fulleffmin = 700
                fulleffmax = 800
            elif i == 1:
                minhistx = 40
                maxhistx = 150
                fulleffmin = 80
                fulleffmax = 90
            else:
                minhistx = .0
                maxhistx = .5
                fulleffmin = 0.
                fulleffmax = 0.03
            hempty = TH1F("hempty_%d" % i, "hempty", 100, 0, maxhistx)
            hempty.GetYaxis().SetTitleOffset(1.2)
            hempty.GetYaxis().SetTitleFont(42)
            hempty.GetXaxis().SetTitleFont(42)
            hempty.GetYaxis().SetLabelFont(42)
            hempty.GetXaxis().SetLabelFont(42)
            hempty.GetXaxis().SetTitle("offline %s" % varlist[i])
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
            hmult.GetXaxis().SetTitle("offline %s" % varlist[i])
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
            linear = TF1("lin_%s_%s" % (triggerlist[i], varlist[i]), \
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
            hratioMBcorr.GetXaxis().SetTitle("offline %s" % varlist[i])
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
            ptext.Draw()
            ctrigger.SaveAs(self.make_file_path(self.d_valevtdata, \
                    "ctrigger_%s_%s" % (trigger, varlist[i]), "eps", \
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
