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
#from array import array
#import itertools
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
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
# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
class AnalyzerDhadrons(Analyzer): # pylint: disable=invalid-name
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)
        self.logger = get_logger()
        #namefiles pkl
        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.p_nptbins = len(self.lpt_finbinmin)
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]
        self.triggerbit = datap["analysis"][self.typean].get("triggerbit", "")

        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc, n_filemass_name)
        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']

        # Output directories and filenames
        self.yields_filename = "yields"
        self.fits_dirname = os.path.join(self.d_resultsallpdata, f"fits_{case}_{typean}")
        self.yields_syst_filename = "yields_syst"
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)
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
        self.rebins = datap["analysis"][self.typean]["rebin"]

        self.p_includesecpeaks = datap["analysis"][self.typean].get("includesecpeak", None)
        self.p_masssecpeak = datap["analysis"][self.typean].get("masssecpeak", None)
        self.p_fix_masssecpeaks = datap["analysis"][self.typean].get("fix_masssecpeak", None)
        self.p_widthsecpeak = datap["analysis"][self.typean].get("widthsecpeak", None)
        self.p_fix_widthsecpeak = datap["analysis"][self.typean].get("fix_widthsecpeak", None)
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
        # More specific fit options
        self.include_reflection = datap["analysis"][self.typean].get("include_reflection", False)

        self.p_nevents = datap["analysis"][self.typean]["nevents"]
        self.p_sigmamb = datap["ml"]["opt"]["sigma_MB"]
        self.p_br = datap["ml"]["opt"]["BR"]

        # Systematics
        self.mt_syst_dict = datap["analysis"][self.typean].get("systematics", None)
        self.d_mt_results_path = os.path.join(self.d_resultsallpdata, "multi_trial")

        self.p_indexhpt = datap["analysis"]["indexhptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_sigmav0 = datap["analysis"]["sigmav0"]
        self.p_inputfonllpred = datap["analysis"]["inputfonllpred"]
        self.p_triggereff = datap["analysis"][self.typean].get("triggereff", [1])
        self.p_triggereffunc = datap["analysis"][self.typean].get("triggereffunc", [0])

        self.root_objects = []

        # Fitting
        self.fitter = None
        self.p_performval = datap["analysis"].get("event_cand_validation", None)

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
                self.logger.error("Cannot load fits from dir %s", self.fits_dirname)
                return

        # Additional directory needed where the intermediate results of the multi trial are
        # written to
        dir_yield_syst = os.path.join(self.d_resultsallpdata, "multi_trial")
        self.fitter.perform_syst(dir_yield_syst)
        # Directory of intermediate results and plot output directory are the same here
        self.fitter.draw_syst(dir_yield_syst, dir_yield_syst)

        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    #def efficiency(self):
    #To be added from dhadron_mult

    #def plotter(self):
    #To be added from dhadron_mult

    @staticmethod
    def calculate_norm(hsel, hnovt, hvtxout):
        if not hsel:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hsel")
        if not hnovt:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hnovt")
        if not hvtxout:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hvtxout")

        n_sel = hsel.Integral()
        n_novtx = hnovt.Integral()
        n_vtxout = hvtxout.Integral()
        norm = -1
        if n_sel + n_vtxout > 0:
            norm = (n_sel + n_novtx) - n_novtx * n_vtxout / (n_sel + n_vtxout)
        return norm

    def makenormyields(self): # pylint: disable=import-outside-toplevel, too-many-branches
        filemass = TFile.Open(self.n_filemass)
        labeltrigger = "hbit%s" % (self.triggerbit)
        hsel = filemass.Get("sel_%s" % labeltrigger)
        hnovtx = filemass.Get("novtx_%s" % labeltrigger)
        hvtxout = filemass.Get("vtxout_%s" % labeltrigger)
        norm = self.calculate_norm(hsel, hnovtx, hvtxout)
        self.logger.warning("Number of events %d", norm)

    #def plotternormyields(self):
    #To be added from dhadron_mult

    #def plottervalidation(self):
    #To be added from dhadron_mult
