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


from os.path import join
import os
from glob import glob
from array import array

#pylint: disable=too-many-lines
from ROOT import TFile, TH1F, TCanvas, gStyle, Double  #pylint: disable=import-error, no-name-in-module

from machine_learning_hep.logger import get_logger
from machine_learning_hep.utilities import make_file_path
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.fitting.utils import save_fit, load_fit
from machine_learning_hep.fitting.fitters import FitAliHF, FitROOTGauss, FitSystAliHF

class MLFitParsFactory: # pylint: disable=too-many-instance-attributes, too-many-statements
    """
    Managing MLHEP specific fit parameters and is used to collect and retrieve all information
    required to initialise a (systematic) fit
    """

    SIG_FUNC_MAP = {"kGaus": 0, "k2Gaus": 1, "kGausSigmaRatioPar": 2}
    BKG_FUNC_MAP = {"kExpo": 0, "kLin": 1, "Pol2": 2, "kNoBk": 3, "kPow": 4, "kPowEx": 5}

    def __init__(self, database: dict, ana_type: str, file_data_name: str, file_mc_name: str): # pylint: disable=too-many-branches
        """
        Initialize MLFitParsFactory
        Args:
            database: dictionary of the entire analysis database
            ana_type: specifying the analysis within the database to be done
            file_data_name: file path where to find data histograms to fit
            file_mc_name: file path where to find MC histograms to fit
        """

        self.logger = get_logger()

        ana_config = database["analysis"][ana_type]

        self.mltype = database["ml"]["mltype"]

        # File config
        self.file_data_name = file_data_name
        self.file_mc_name = file_mc_name

        # Binning
        self.bin1_name = database["var_binning"]
        self.bins1_edges_low = ana_config["sel_an_binmin"]
        self.bins1_edges_up = ana_config["sel_an_binmax"]
        self.n_bins1 = len(self.bins1_edges_low)
        self.bin2_name = ana_config["var_binning2"]
        self.bin2_gen_name = ana_config["var_binning2_gen"]
        self.bins2_edges_low = ana_config["sel_binmin2"]
        self.bins2_edges_up = ana_config["sel_binmax2"]
        self.n_bins2 = len(self.bins2_edges_low)

        bineff = ana_config["usesinglebineff"]
        self.bins2_int_bin = bineff if bineff is not None else 0

        self.prob_cut_fin = database["analysis"][ana_type].get("probcuts", None)
        # Make it backwards-compatible
        if not self.prob_cut_fin:
            bin_matching = database["analysis"][ana_type]["binning_matching"]
            prob_cut_fin_tmp = database["mlapplication"]["probcutoptimal"]
            self.prob_cut_fin = []
            for i in range(self.n_bins1):
                bin_id = bin_matching[i]
                self.prob_cut_fin.append(prob_cut_fin_tmp[bin_id])

        # Fit method flags
        self.init_fits_from = ana_config["init_fits_from"]
        self.pre_fit_class_mc = ana_config.get("pre_fits_mc", ["kGaus"] * len(self.init_fits_from))
        self.sig_func_name = ana_config["sgnfunc"]
        self.bkg_func_name = ana_config["bkgfunc"]
        self.fit_range_low = ana_config["massmin"]
        self.fit_range_up = ana_config["massmax"]
        self.likelihood = ana_config["dolikelihood"]
        self.rebin = ana_config["rebin"]
        try:
            iter(self.rebin[0])
        except TypeError:
            self.rebin = [self.rebin for _ in range(self.n_bins2)]

        # Initial fit parameters
        self.mean = ana_config["masspeak"]
        try:
            iter(self.mean)
        except TypeError:
            self.mean = [self.mean] * self.n_bins1
        try:
            iter(self.mean[0])
        except TypeError:
            self.mean = [self.mean] * self.n_bins2

        self.fix_mean = ana_config["FixedMean"]
        self.use_user_mean = ana_config["SetInitialGaussianMean"]
        self.sigma = ana_config["sigmaarray"]
        self.fix_sigma = ana_config["SetFixGaussianSigma"]
        self.use_user_sigma = ana_config["SetInitialGaussianSigma"]
        self.use_user_mean = ana_config["SetInitialGaussianMean"]
        try:
            iter(self.use_user_mean)
        except TypeError:
            self.use_user_mean = [self.use_user_mean] * self.n_bins1
        self.max_rel_sigma_diff = ana_config["MaxPercSigmaDeviation"]
        self.n_sigma_sideband = ana_config["exclude_nsigma_sideband"]
        self.n_sigma_signal = ana_config["nsigma_signal"]
        self.rel_sigma_bound = ana_config["MaxPercSigmaDeviation"]

        # Second peak flags
        self.include_sec_peak = ana_config.get("includesecpeak", [False] * self.n_bins1)
        try:
            iter(self.include_sec_peak[0])
        except TypeError:
            self.include_sec_peak = [self.include_sec_peak for _ in range(self.n_bins2)]

        self.sec_mean = ana_config["masssecpeak"] if self.include_sec_peak else None
        self.fix_sec_mean = ana_config.get("fix_masssecpeak", [False] * self.n_bins1)
        try:
            iter(self.fix_sec_mean[0])
        except TypeError:
            self.fix_sec_mean = [self.fix_sec_mean for _ in range(self.n_bins2)]
        self.sec_sigma = ana_config["widthsecpeak"] if self.include_sec_peak else None
        self.fix_sec_sigma = ana_config["fix_widthsecpeak"] if self.include_sec_peak else None

        # Reflections flag
        self.include_reflections = ana_config.get("include_reflection", False)

        # Is this a trigger weighted histogram?
        self.apply_weights = ana_config["triggersel"].get("usetriggcorrfunc", None) is not None

        # Systematics
        self.syst_pars = ana_config.get("systematics", {})
        self.syst_init_sigma_from = None
        self.syst_consider_free_sigma = None
        self.syst_rel_var_sigma_up = None
        self.syst_rel_var_sigma_down = None
        if self.syst_pars:
            self.syst_init_sigma_from = self.syst_pars.get("init_sigma_from", "central")
            if not isinstance(self.syst_init_sigma_from, list):
                self.syst_init_sigma_from = [self.syst_init_sigma_from] * self.n_bins1
            if not isinstance(self.syst_init_sigma_from[0], list):
                self.syst_init_sigma_from = [self.syst_init_sigma_from] * self.n_bins2

            self.syst_consider_free_sigma = self.syst_pars.get("consider_free_sigma", False)
            try:
                iter(self.syst_consider_free_sigma)
            except TypeError:
                self.syst_consider_free_sigma = [self.syst_consider_free_sigma] * self.n_bins1

            self.syst_rel_var_sigma_up = self.syst_pars.get("rel_var_sigma_up", None)
            try:
                iter(self.syst_rel_var_sigma_up)
            except TypeError:
                self.syst_rel_var_sigma_up = [self.syst_rel_var_sigma_up] * self.n_bins1

            self.syst_rel_var_sigma_down = self.syst_pars.get("rel_var_sigma_down", None)
            try:
                iter(self.syst_rel_var_sigma_down)
            except TypeError:
                self.syst_rel_var_sigma_down = [self.syst_rel_var_sigma_down] * self.n_bins1


    def make_ali_hf_fit_pars(self, ibin1, ibin2):
        """
        Making fit paramaters for AliHF mass fitter
        Args:
            ibin1: Number of bin of first binning variable
            ibin2: Number of bin of second binning variable
        Returns:
            dictionary of fit parameters
        """

        fit_pars = {"sig_func_name": MLFitParsFactory.SIG_FUNC_MAP[self.sig_func_name[ibin1]],
                    "bkg_func_name": MLFitParsFactory.BKG_FUNC_MAP[self.bkg_func_name[ibin1]],
                    "likelihood": self.likelihood,
                    "rebin": self.rebin[ibin2][ibin1],
                    "fit_range_low": self.fit_range_low[ibin1],
                    "fit_range_up": self.fit_range_up[ibin1],
                    "n_sigma_sideband": self.n_sigma_sideband,
                    "rel_sigma_bound": self.rel_sigma_bound,
                    "mean": self.mean[ibin2][ibin1],
                    "sigma": self.sigma[ibin1],
                    "fix_mean": self.fix_mean,
                    "fix_sigma": self.fix_sigma[ibin1]}

        fit_pars["include_sec_peak"] = self.include_sec_peak[ibin2][ibin1]
        if self.include_sec_peak[ibin2][ibin1]:
            fit_pars["sec_mean"] = self.sec_mean
            fit_pars["fix_sec_mean"] = self.fix_sec_mean[ibin2][ibin1]
            fit_pars["sec_sigma"] = self.sec_sigma
            fit_pars["fix_sec_sigma"] = self.fix_sec_sigma
            fit_pars["use_sec_peak_rel_sigma"] = True

        if self.include_reflections:
            fit_pars["include_reflections"] = True
            fit_pars["fix_reflections_s_over_b"] = True
        else:
            fit_pars["include_reflections"] = False

        return fit_pars


    def make_ali_hf_syst_pars(self, ibin1, ibin2):
        """
        Making fit systematic paramaters for AliHF mass fitter
        Args:
            ibin1: Number of bin of first binning variable
            ibin2: Number of bin of second binning variable
        Returns:
            dictionary of systematic fit parameters
        """

        fit_pars = {"mean": None,
                    "sigma": None,
                    "rebin": self.rebin[ibin2][ibin1],
                    "fit_range_low": self.fit_range_low[ibin1],
                    "fit_range_up": self.fit_range_up[ibin1],
                    "likelihood": self.likelihood,
                    "n_sigma_sideband": self.n_sigma_sideband,
                    "mean_ref": None,
                    "sigma_ref": None,
                    "yield_ref": None,
                    "chi2_ref": None,
                    "signif_ref": None,
                    "fit_range_low_syst": self.syst_pars.get("massmin", None),
                    "fit_range_up_syst": self.syst_pars.get("massmax", None),
                    "bin_count_sigma_syst": self.syst_pars.get("bincount_sigma", None),
                    "bkg_func_names_syst": self.syst_pars.get("bkg_funcs", None),
                    "rebin_syst": self.syst_pars.get("rebin", None),
                    # Check DB
                    "consider_free_sigma_syst": self.syst_consider_free_sigma[ibin1],
                    "rel_var_sigma_up_syst": self.syst_rel_var_sigma_up[ibin1],
                    "rel_var_sigma_down_syst": self.syst_rel_var_sigma_down[ibin1],
                    "signif_min_syst": self.syst_pars.get("min_signif", 3.),
                    "chi2_max_syst": self.syst_pars.get("max_chisquare_ndf", 2.)}

        fit_pars["include_sec_peak"] = self.include_sec_peak[ibin2][ibin1]
        if self.include_sec_peak[ibin2][ibin1]:
            fit_pars["sec_mean"] = self.sec_mean
            fit_pars["fix_sec_mean"] = self.fix_sec_mean[ibin2][ibin1]
            fit_pars["sec_sigma"] = self.sec_sigma
            fit_pars["fix_sec_sigma"] = self.fix_sec_sigma
            fit_pars["use_sec_peak_rel_sigma"] = True

        if self.include_reflections:
            fit_pars["include_reflections"] = True
            fit_pars["fix_reflections_s_over_b"] = True
        else:
            fit_pars["include_reflections"] = False

        return fit_pars


    def make_suffix(self, ibin1, ibin2):
        """
        Build name suffix to find histograms in ROOT file
        Args:
            ibin1: Number of bin of first binning variable
            ibin2: Number of bin of second binning variable
        Returns:
            Suffix string
        """
        if self.mltype == "MultiClassification":
            return "%s%d_%d_%.2f%.2f%s_%.2f_%.2f" % \
                   (self.bin1_name, self.bins1_edges_low[ibin1],
                    self.bins1_edges_up[ibin1], self.prob_cut_fin[ibin1][0],
                    self.prob_cut_fin[ibin1][1], self.bin2_name,
                    self.bins2_edges_low[ibin2], self.bins2_edges_up[ibin2])
        return "%s%d_%d_%.2f%s_%.2f_%.2f" % \
               (self.bin1_name, self.bins1_edges_low[ibin1],
                self.bins1_edges_up[ibin1], self.prob_cut_fin[ibin1],
                self.bin2_name, self.bins2_edges_low[ibin2],
                self.bins2_edges_up[ibin2])


    def get_histograms(self, ibin1, ibin2, get_data=True, get_mc=False, get_reflections=False):
        """
        Get histograms according to specified bins
        Args:
            ibin1: Number of bin of first binning variable
            ibin2: Number of bin of second binning variable
            get_data: get the data histogram
            get_mc: get the MC histogram
            get_reflections: get the MC reflections histogram
        Returns:
            histograms as requested. None for each that was not requested
        """
        suffix = self.make_suffix(ibin1, ibin2)

        histo_data = None
        if get_data:
            file_data = TFile.Open(self.file_data_name, "READ")
            histo_name = "h_invmass_weight" if self.apply_weights else "hmass"
            histo_data = file_data.Get(histo_name + suffix)
            histo_data.SetDirectory(0)

        if not (get_mc or get_reflections):
            return histo_data, None, None

        file_mc = TFile.Open(self.file_mc_name, "READ")
        histo_mc = None
        histo_reflections = None
        if get_mc:
            histo_mc = file_mc.Get("hmass_sig" + suffix)
            histo_mc.SetDirectory(0)
        if get_reflections:
            histo_reflections = file_mc.Get("hmass_refl" + suffix)
            histo_reflections.SetDirectory(0)

        return histo_data, histo_mc, histo_reflections


    def get_fit_pars(self, ibin1, ibin2):
        """
        Collect histograms, fit paramaters and the information whether this fit should be
        initialised from another one.
        Args:
            ibin1: Number of bin of first binning variable
            ibin2: Number of bin of second binning variable
        Returns:
            dictionary with all required information to initialise a fit
        """

        fit_pars = self.make_ali_hf_fit_pars(ibin1, ibin2)
        histo_data, histo_mc, histo_reflections = self.get_histograms(ibin1, ibin2, \
                get_data=True, get_mc=True, \
                get_reflections=fit_pars["include_reflections"])

        lock_override_init = ["sigma"] if self.use_user_sigma[ibin1] else []
        if self.use_user_mean[ibin1]:
            lock_override_init.append("mean")

        return {"histograms": {"data": histo_data,
                               "mc": histo_mc,
                               "reflections": histo_reflections},
                "init_from": self.init_fits_from[ibin1],
                "lock_override_init": lock_override_init,
                "init_pars": fit_pars,
                "pre_fit_mc": {"type_gauss": self.pre_fit_class_mc[ibin1]}}


    def get_syst_pars(self, ibin1, ibin2):
        """
        Collect histograms, fit paramaters and the information whether this systematic fit should
        be initialised from another one.
        Args:
            ibin1: Number of bin of first binning variable
            ibin2: Number of bin of second binning variable
        Returns:
            dictionary with all required information to initialise a systematic fit
        """

        if not self.syst_pars:
            self.logger.warning("There are no systematics parameters defined. Skip...")
            return None

        fit_pars = self.make_ali_hf_syst_pars(ibin1, ibin2)
        histo_data, histo_mc, histo_reflections = self.get_histograms(ibin1, ibin2, \
                get_data=True, get_mc=fit_pars["include_reflections"], \
                get_reflections=fit_pars["include_reflections"])

        return {"histograms": {"data": histo_data,
                               "mc": histo_mc,
                               "reflections": histo_reflections},
                "init_from": self.syst_init_sigma_from[ibin2][ibin1],
                "init_pars": fit_pars}


    def yield_fit_pars(self):
        """
        Yield bin numbers and corresponding fit parameters one-by-one
        """
        for ibin2 in range(self.n_bins2):
            for ibin1 in range(self.n_bins1):
                yield ibin1, ibin2, self.get_fit_pars(ibin1, ibin2)


    def yield_syst_pars(self):
        """
        Yield bin numbers and corresponding systematic fit parameters one-by-one
        """
        for ibin2 in range(self.n_bins2):
            for ibin1 in range(self.n_bins1):
                yield ibin1, ibin2, self.get_syst_pars(ibin1, ibin2)


class MLFitter: # pylint: disable=too-many-instance-attributes
    """
    Wrapper around all available fits insatntiated and used in an MLHEP analysis run.
    """


    def __init__(self, case: str, database: dict, ana_type: str,
                 data_out_dir: str, mc_out_dir: str):
        """
        Initialize MLFitter
        Args:
            database: dictionary of the entire analysis database
            ana_type: specifying the analysis within the database to be done
            file_data_name: file path where to find data histograms to fit
            file_mc_name: file path where to find MC histograms to fit
        """

        self.logger = get_logger()

        self.case = case
        self.ana_type = ana_type
        self.ana_config = database["analysis"][ana_type]

        self.pars_factory = MLFitParsFactory(database, ana_type, data_out_dir, mc_out_dir)

        self.pre_fits_mc = None
        self.pre_fits_data = None
        self.central_fits = None
        self.init_central_fits_from = None
        self.lock_override_init = None
        self.syst_fits = None
        self.init_syst_fits_from = None

        # Flags
        self.is_initialized_fits = False
        self.done_pre_fits = False
        self.done_central_fits = False
        self.is_initialized_syst = False
        self.done_syst = False


    def initialize_fits(self):
        """
        Initialize all fits required in an MLHEP analysis run. Using MLFitParsFactory to retrieve
        fit parameters.
        """

        if self.is_initialized_fits:
            self.logger.warning("Fits already initialized. Skip...")
            return
        self.pre_fits_mc = {}
        self.pre_fits_data = {}
        self.central_fits = {}
        self.init_central_fits_from = {}
        self.lock_override_init = {}

        pre_fits_bins1 = []
        for ibin1, ibin2, pars in self.pars_factory.yield_fit_pars():
            self.central_fits[(ibin1, ibin2)] = FitAliHF( \
                    pars["init_pars"], \
                    histo=pars["histograms"]["data"], \
                    histo_mc=pars["histograms"]["mc"], \
                    histo_reflections=pars["histograms"]["reflections"])
            self.init_central_fits_from[(ibin1, ibin2)] = pars["init_from"]
            self.lock_override_init[(ibin1, ibin2)] = pars["lock_override_init"]
            if ibin1 in pre_fits_bins1:
                continue

            pre_fits_bins1.append(ibin1)

            self.pre_fits_mc[ibin1] = FitROOTGauss(pars["init_pars"],
                                                   histo=pars["histograms"]["mc"],
                                                   **pars["pre_fit_mc"])
            self.pre_fits_data[ibin1] = FitAliHF( \
                    pars["init_pars"], \
                    histo=pars["histograms"]["data"], \
                    histo_mc=pars["histograms"]["mc"], \
                    histo_reflections=pars["histograms"]["reflections"])
        self.is_initialized_fits = True


    def perform_pre_fits(self):
        """
        Perform all pre-fits whose fitted parameters might be used to initialize central fits.
        """

        if self.done_pre_fits:
            self.logger.warning("Pre-fits already done. Skip...")
            return

        if not self.is_initialized_fits:
            self.initialize_fits()

        self.logger.info("Perform pre-fits on MC")
        for fit in self.pre_fits_mc.values():
            fit.override_init_pars(fix_mean=False, fix_sigma=False, likelihood=False)
            fit.fit()
        self.logger.info("Perform pre-fits on data")
        for fit in self.pre_fits_data.values():
            fit.override_init_pars(fix_mean=False, fix_sigma=False)
            fit.fit()
        self.done_pre_fits = True


    def perform_central_fits(self):
        """
        Perform all central fits and initialize from pre-fits if requested.
        """

        if self.done_central_fits:
            self.logger.warning("Central fits already done. Skip...")
            return

        if not self.done_pre_fits:
            self.perform_pre_fits()

        for (ibin1, ibin2), fit in self.central_fits.items():
            pre_fit = None
            if self.init_central_fits_from[(ibin1, ibin2)] == "mc":
                pre_fit = self.pre_fits_mc[ibin1]
            else:
                pre_fit = self.pre_fits_data[ibin1]
            if not pre_fit.success and self.lock_override_init[(ibin1, ibin2)] \
                    and "sigma" not in self.lock_override_init[(ibin1, ibin2)]:
                self.logger.warning("Requested pre-fit on %s not successful but requested for " \
                                    "central fit in bins (%i, %i). Skip...",
                                    self.init_central_fits_from[(ibin1, ibin2)], ibin1, ibin2)
                continue

            override_init_pars = pre_fit.get_fit_pars() if pre_fit and pre_fit.success else {}
            if self.lock_override_init[(ibin1, ibin2)]:
                for name in self.lock_override_init[(ibin1, ibin2)]:
                    _ = override_init_pars.pop(name, None)

            self.logger.info("Perform central fit in bin (%i, %i)", ibin1, ibin2)
            fit.override_init_pars(**override_init_pars)
            fit.fit()

        self.done_central_fits = True


    def get_central_fit(self, ibin1, ibin2):
        """
        Retrieve a central fit based on specified bin numbers
        initialised from another one.
        Args:
            ibin1: Number of bin of first binning variable
            ibin2: Number of bin of second binning variable
        Returns:
            Fit, if valid bin numbers given, None otherwise
        """

        return self.central_fits.get((ibin1, ibin2), None)


    def print_fits(self):
        """
        Print pre-fits and central  fits
            - bin numbers
            - fit name (e.g. its class name)
            - fit parameters
        """

        self.logger.info("Print all fits")
        print("Pre-fits for data")
        for ibin1, fit in self.pre_fits_data.items():
            print(f"Bin1: {ibin1}")
            print(fit)
        print("\n ####################################\n")
        print("Pre-fits for MC")
        for ibin1, fit in self.pre_fits_mc.items():
            print(f"Bin1: {ibin1}")
            print(fit)
        print("\n ####################################\n")
        print("Central fits")
        for (ibin1, ibin2), fit in self.central_fits.items():
            print(f"Bin1, bin2: ({ibin1}, {ibin2})")
            print(fit)
        self.logger.info("Print all fits done")


    def initialize_syst(self):
        """
        Initialize all systematic fits required in an MLHEP analysis run. Using MLFitParsFactory
        to retrieve systematic fit parameters.
        """

        if self.is_initialized_syst:
            self.logger.warning("Syst already initialized. Skip...")
            return
        if not self.done_central_fits:
            self.logger.warning("Fits have not been done yet. Skip...")
            return

        self.syst_fits = {}
        self.init_syst_fits_from = {}

        for ibin1, ibin2, pars in self.pars_factory.yield_syst_pars():
            if not pars:
                self.syst_fits[(ibin1, ibin2)] = None
                continue
            self.syst_fits[(ibin1, ibin2)] = FitSystAliHF( \
                    pars["init_pars"], \
                    histo=pars["histograms"]["data"], \
                    histo_mc=pars["histograms"]["mc"], \
                    histo_reflections=pars["histograms"]["reflections"])
            self.init_syst_fits_from[(ibin1, ibin2)] = pars["init_from"]

        self.is_initialized_syst = True


    def perform_syst(self, results_dir):
        """
        Perform all systematic fits and initialize from central-fits if requested.
        """

        if self.done_syst:
            self.logger.warning("Syst already fitted. Skip...")
            return

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if not self.is_initialized_syst:
            self.initialize_syst()

        for (ibin1, ibin2), fit in self.syst_fits.items():
            if not fit:
                self.logger.warning("No systematic fit for bins (%i, %i). Skip...",
                                    ibin1, ibin2)
                continue

            if not self.central_fits[(ibin1, ibin2)].success:
                self.logger.warning("Central fit not successful for bins (%i, %i). Skip...",
                                    ibin1, ibin2)
                continue

            # Prepare to overwrite some ini parameters
            pre_fit = None
            central_fit = self.central_fits[(ibin1, ibin2)]
            init_from = self.init_syst_fits_from[(ibin1, ibin2)]
            if init_from == "pre_fit" and self.init_central_fits_from[(ibin1, ibin2)] == "mc":
                pre_fit = self.pre_fits_mc[ibin1]
            elif init_from == "pre_fit" and self.init_central_fits_from[(ibin1, ibin2)] == "data":
                pre_fit = self.pre_fits_data[ibin1]
            else:
                pre_fit = central_fit

            # Get reference parameters
            signif = Double()
            signif_err = Double()
            central_fit.kernel.Significance(self.pars_factory.n_sigma_signal, signif, signif_err)
            central_fit_pars = central_fit.get_fit_pars()
            overwrite_init = {"yield_ref": central_fit.kernel.GetRawYield(),
                              "mean_ref": central_fit_pars["mean"],
                              "sigma_ref": central_fit_pars["sigma"],
                              "chi2_ref": central_fit.kernel.GetReducedChiSquare(),
                              "signif_ref": signif}
            # Get mean and sigma for fit init
            pre_fit_pars = pre_fit.get_fit_pars()
            overwrite_init["mean"] = pre_fit_pars["mean"]
            overwrite_init["sigma"] = pre_fit_pars["sigma"]

            fit.override_init_pars(**overwrite_init)

            # Set the path for intermediate results which are produced by the multi trial fitter
            fit.results_path = os.path.join(results_dir,
                                            f"multi_trial_bin1_{ibin1}_bin2_{ibin2}.root")
            fit.fit()

        self.done_syst = True


    def get_bins2(self):
        bins2 = []
        for (_, ibin2) in self.central_fits:
            if ibin2 in bins2:
                continue
            bins2.append(ibin2)
        return bins2


    def draw_fits(self, save_dir, root_dir=None): # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        """
        Draw all fits one-by-one
        Args:
            save_dir: directory where to save plots
            root_dir: TDirectory where to save summary plots
        """
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(1)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)

        bins2 = self.get_bins2()
        bins1_ranges = self.pars_factory.bins1_edges_low.copy()
        bins1_ranges.append(self.pars_factory.bins1_edges_up[-1])
        n_bins1 = len(bins1_ranges) - 1

        def fill_wrapper(histo, ibin, central, err=None):
            histo.SetBinContent(ibin, central)
            if err is not None:
                histo.SetBinError(ibin, err)

        # Summarize in mult histograms in pT bins
        yieldshistos = {ibin2: TH1F("hyields%d" % (ibin2), "", \
                n_bins1, array("d", bins1_ranges)) for ibin2 in bins2}
        means_histos = {ibin2:TH1F("hmeans%d" % (ibin2), "", \
                n_bins1, array("d", bins1_ranges)) for ibin2 in bins2}
        sigmas_histos = {ibin2: TH1F("hsigmas%d" % (ibin2), "", \
                n_bins1, array("d", bins1_ranges)) for ibin2 in bins2}
        signifs_histos = {ibin2: TH1F("hsignifs%d" % (ibin2), "", \
                n_bins1, array("d", bins1_ranges)) for ibin2 in bins2}
        refls_histos = {ibin2: TH1F("hrefl%d" % (ibin2), "", \
                n_bins1, array("d", bins1_ranges)) for ibin2 in bins2}
        have_summary_pt_bins = []
        means_init_mc_histos = TH1F("hmeans_init_mc", "", n_bins1, array("d", bins1_ranges))
        sigmas_init_mc_histos = TH1F("hsigmas_init_mc", "", n_bins1, array("d", bins1_ranges))
        means_init_data_histos = TH1F("hmeans_init_data", "", n_bins1, array("d", bins1_ranges))
        sigmas_init_data_histos = TH1F("hsigmas_init_data", "", n_bins1, array("d", bins1_ranges))

        nx = 4
        ny = 2
        canvy = 533
        if n_bins1 > 12:
            nx = 5
            ny = 4
            canvy = 1200
        elif n_bins1 > 8:
            nx = 4
            ny = 3
            canvy = 800

        canvas_init_mc = TCanvas("canvas_init_mc", "MC", 1000, canvy)
        canvas_init_data = TCanvas("canvas_init_data", "Data", 1000, canvy)
        canvas_data = {ibin2: TCanvas("canvas_data%d" % (ibin2), "Data", 1000, canvy) \
                       for ibin2 in bins2}
        canvas_init_mc.Divide(nx, ny)
        canvas_init_data.Divide(nx, ny)

        for c in canvas_data.values():
            c.Divide(nx, ny)

        # Need to cache some object for which the canvas is only written after the loop...
        for (ibin1, ibin2), fit in self.central_fits.items():

            # Some variables set for drawing
            if self.pars_factory.mltype == "MultiClassification":
                title = f"{self.pars_factory.bins1_edges_low[ibin1]:.1f} < #it{{p}}_{{T}} < " \
                        f"{self.pars_factory.bins1_edges_up[ibin1]:.1f}" \
                        f"(prob0 <= {self.pars_factory.prob_cut_fin[ibin1][0]:.2f} &" \
                        f"prob1 >= {self.pars_factory.prob_cut_fin[ibin1][1]:.2f})"
            else:
                title = f"{self.pars_factory.bins1_edges_low[ibin1]:.1f} < #it{{p}}_{{T}} < " \
                        f"{self.pars_factory.bins1_edges_up[ibin1]:.1f}" \
                        f"(prob > {self.pars_factory.prob_cut_fin[ibin1]:.2f})"

            x_axis_label = "#it{M}_{inv} (GeV/#it{c}^{2})"
            n_sigma_signal = self.pars_factory.n_sigma_signal

            suffix_write = self.pars_factory.make_suffix(ibin1, ibin2)

            kernel = fit.kernel
            histo = fit.histo

            # Central fits
            y_axis_label = \
                    f"Entries/({histo.GetBinWidth(1) * 1000:.0f} MeV/#it{{c}}^{{2}})"
            canvas = TCanvas("fit_canvas", suffix_write, 700, 700)
            fit.draw(canvas, sigma_signal=n_sigma_signal, x_axis_label=x_axis_label,
                     y_axis_label=y_axis_label, title=title)
            if self.pars_factory.apply_weights is False:
                canvas.SaveAs(make_file_path(save_dir, "fittedplot", "eps", None,
                                             suffix_write))
            else:
                canvas.SaveAs(make_file_path(save_dir, "fittedplotweights", "eps", None,
                                             suffix_write))
            canvas.Close()
            fit.draw(canvas_data[ibin2].cd(ibin1+1), sigma_signal=n_sigma_signal,
                     x_axis_label=x_axis_label, y_axis_label=y_axis_label, title=title)

            if fit.success:
                fill_wrapper(yieldshistos[ibin2], ibin1 + 1,
                             kernel.GetRawYield(), kernel.GetRawYieldError())
                fill_wrapper(means_histos[ibin2], ibin1 + 1,
                             kernel.GetMean(), kernel.GetMeanUncertainty())
                fill_wrapper(sigmas_histos[ibin2], ibin1 + 1,
                             kernel.GetSigma(), kernel.GetSigmaUncertainty())
                fill_wrapper(refls_histos[ibin2], ibin1 + 1,
                             kernel.GetReflOverSig(), kernel.GetReflOverSigUncertainty())

                signif = Double()
                signif_err = Double()
                kernel.Significance(n_sigma_signal, signif, signif_err)
                fill_wrapper(signifs_histos[ibin2], ibin1 + 1, signif, signif_err)

                # Residual plot
                c_res = TCanvas('cRes', 'The Fit Canvas', 800, 800)
                c_res.cd()
                h_pulls = histo.Clone(f"{histo.GetName()}_pull")
                h_residual_trend = histo.Clone(f"{histo.GetName()}_residual_trend")
                h_pulls_trend = histo.Clone(f"{histo.GetName()}_pulls_trend")
                if self.pars_factory.include_reflections:
                    _ = kernel.GetOverBackgroundPlusReflResidualsAndPulls( \
                            h_pulls, h_residual_trend, h_pulls_trend, \
                            self.pars_factory.fit_range_low[ibin1], \
                            self.pars_factory.fit_range_up[ibin1])
                else:
                    _ = kernel.GetOverBackgroundResidualsAndPulls( \
                            h_pulls, h_residual_trend, h_pulls_trend, \
                            self.pars_factory.fit_range_low[ibin1], \
                            self.pars_factory.fit_range_up[ibin1])
                h_residual_trend.Draw()
                c_res.SaveAs(make_file_path(save_dir, "residual", "eps", None, suffix_write))
                c_res.Close()


            # Summary plots to be done only once per pT bin
            if ibin1 in have_summary_pt_bins:
                continue

            have_summary_pt_bins.append(ibin1)

            # Pre-fit MC
            suffix_write = self.pars_factory.make_suffix(ibin1, self.pars_factory.bins2_int_bin)

            pre_fit_mc = self.pre_fits_mc[ibin1]
            kernel = pre_fit_mc.kernel
            histo = pre_fit_mc.histo
            y_axis_label = \
                    f"Entries/({histo.GetBinWidth(1) * 1000:.0f} MeV/#it{{c}}^{{2}})"
            canvas = TCanvas("fit_canvas_mc_init", suffix_write, 700, 700)
            pre_fit_mc.draw(canvas, x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                            title=title)

            canvas.SaveAs(make_file_path(save_dir, "fittedplot_integrated_mc", "eps", None,
                                         suffix_write))
            canvas.Close()
            pre_fit_mc.draw(canvas_init_mc.cd(ibin1+1), x_axis_label=x_axis_label,
                            y_axis_label=y_axis_label, title=title)


            if pre_fit_mc.success:
                # Only fill these summary plots in case of success
                means_init_mc_histos.SetBinContent(ibin1 + 1, kernel.GetParameter(1))
                means_init_mc_histos.SetBinError(ibin1 + 1, kernel.GetParError(1))
                sigmas_init_mc_histos.SetBinContent(ibin1 + 1, kernel.GetParameter(2))
                sigmas_init_mc_histos.SetBinError(ibin1 + 1, kernel.GetParError(2))


            pre_fit_data = self.pre_fits_data[ibin1]
            kernel = pre_fit_data.kernel
            histo = pre_fit_data.histo


            # Pre-fit data
            y_axis_label = \
                    f"Entries/({histo.GetBinWidth(1) * 1000:.0f} MeV/#it{{c}}^{{2}})"
            canvas = TCanvas("fit_canvas_data_init", suffix_write, 700, 700)
            pre_fit_data.draw(canvas, sigma_signal=n_sigma_signal, x_axis_label=x_axis_label,
                              y_axis_label=y_axis_label, title=title)
            canvas.SaveAs(make_file_path(save_dir, "fittedplot_integrated", "eps", None,
                                         suffix_write))
            canvas.Close()
            pre_fit_data.draw(canvas_init_data.cd(ibin1+1), sigma_signal=n_sigma_signal,
                              x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                              title=title)

            if pre_fit_data.success:
                # Only fill these summary plots in case of success
                means_init_data_histos.SetBinContent(ibin1 + 1, kernel.GetMean())
                means_init_data_histos.SetBinError(ibin1 + 1, kernel.GetMeanUncertainty())
                sigmas_init_data_histos.SetBinContent(ibin1 + 1, kernel.GetSigma())
                sigmas_init_data_histos.SetBinError(ibin1 + 1, kernel.GetSigmaUncertainty())


        canvas_init_mc.SaveAs(make_file_path(save_dir, "canvas_InitMC", "eps"))
        canvas_init_mc.Close()
        canvas_init_data.SaveAs(make_file_path(save_dir, "canvas_InitData", "eps"))
        canvas_init_data.Close()
        for ibin2 in bins2:
            suffix2 = f"ibin2_{ibin2}"
            canvas_data[ibin2].SaveAs(make_file_path(save_dir, "canvas_FinalData", "eps", None,
                                                     suffix2))
            if root_dir:
                root_dir.cd()
                yieldshistos[ibin2].Write()
                means_histos[ibin2].Write()
                sigmas_histos[ibin2].Write()
                signifs_histos[ibin2].Write()
                refls_histos[ibin2].Write()
            #canvas_data[ibin2].Close()


        latex_bin2_var = self.ana_config["latexbin2var"]
        latex_hadron_name = self.ana_config["latexnamehadron"]
        # Plot some summary historgrams
        leg_strings = [f"{self.pars_factory.bins2_edges_low[ibin2]} #leq {latex_bin2_var} < " \
                       f"{self.pars_factory.bins2_edges_up[ibin2]}" for ibin2 in bins2]
        save_name = make_file_path(save_dir, "Yields", "eps", None, [self.case, self.ana_type])
        # Yields summary plot
        plot_histograms([yieldshistos[ibin2] for ibin2 in bins2], True, True, leg_strings,
                        "uncorrected yields", "#it{p}_{T} (GeV/#it{c})",
                        f"Uncorrected yields {latex_hadron_name} {self.ana_type}", "mult. / int.",
                        save_name)
        save_name = make_file_path(save_dir, "Means", "eps", None, [self.case, self.ana_type])
        # Means summary plot
        plot_histograms([means_histos[ibin2] for ibin2 in bins2], False, True, leg_strings, "Means",
                        "#it{p}_{T} (GeV/#it{c})",
                        "#mu_{fit} " + f"{latex_hadron_name} {self.ana_type}", "mult. / int.",
                        save_name)
        save_name = make_file_path(save_dir, "Sigmas", "eps", None, [self.case, self.ana_type])
        #Sigmas summary plot
        plot_histograms([sigmas_histos[ibin2] for ibin2 in bins2], False, True, leg_strings,
                        "Sigmas", "#it{p}_{T} (GeV/#it{c})",
                        "#sigma_{fit} " + f"{latex_hadron_name} {self.ana_type}", "mult. / int.",
                        save_name)

        # Plot the initialized means and sigma for MC and data
        save_name = make_file_path(save_dir, "Means_mult_int", "eps", None,
                                   [self.case, self.ana_type])
        plot_histograms([means_init_mc_histos, means_init_data_histos], False, False,
                        ["MC", "data"], "Means of int. mult.", "#it{p}_{T} (GeV/#it{c})",
                        "#mu_{fit} " + f"{latex_hadron_name} {self.ana_type}", "", save_name)

        save_name = make_file_path(save_dir, "Sigmas_mult_int", "eps", None,
                                   [self.case, self.ana_type])
        plot_histograms([sigmas_init_mc_histos, sigmas_init_data_histos], False, False,
                        ["MC", "data"], "Sigmas of int. mult.", "#it{p}_{T} (GeV/#it{c})",
                        "#sigma_{fit} " + f"{latex_hadron_name} {self.ana_type}", "", save_name)


    def draw_syst(self, save_dir, results_dir, root_dir=None): # pylint: disable=too-many-branches, too-many-statements, too-many-locals
        """Draw all fits one-by-one

        Args:
            save_dir: directory where to save plots
            results_dir: where to find intermediate results of the multi trial
            root_dir: TDirectory where to save summary plots
        """

        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(1)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)

        bins1_ranges = self.pars_factory.bins1_edges_low.copy()
        bins1_ranges.append(self.pars_factory.bins1_edges_up[-1])

        for (ibin1, ibin2), fit in self.syst_fits.items():
            if not fit:
                self.logger.warning("No systematic fit for bins (%i, %i). Skip...",
                                    ibin1, ibin2)
                continue

            # Some variables set for drawing
            if self.pars_factory.mltype == "MultiClassification":
                title = f"{self.pars_factory.bins1_edges_low[ibin1]:.1f} < #it{{p}}_{{T}} < " \
                        f"{self.pars_factory.bins1_edges_up[ibin1]:.1f}" \
                        f"(prob0 <= {self.pars_factory.prob_cut_fin[ibin1][0]:.2f} &" \
                        f"prob1 >= {self.pars_factory.prob_cut_fin[ibin1][1]:.2f})"
            else:
                title = f"{self.pars_factory.bins1_edges_low[ibin1]:.1f} < #it{{p}}_{{T}} < " \
                        f"{self.pars_factory.bins1_edges_up[ibin1]:.1f}" \
                        f"(prob > {self.pars_factory.prob_cut_fin[ibin1]:.2f})"

            suffix_write = self.pars_factory.make_suffix(ibin1, ibin2)

            fit.results_path = os.path.join(results_dir,
                                            f"multi_trial_bin1_{ibin1}_bin2_{ibin2}.root")

            # Central fits
            canvas = TCanvas("fit_canvas", suffix_write, 1400, 800)
            fit.draw(canvas, title=title)

            if self.pars_factory.apply_weights is False:
                canvas.SaveAs(make_file_path(save_dir, "multi_trial", "eps", None,
                                             suffix_write))
            else:
                canvas.SaveAs(make_file_path(save_dir, "multi_trial_weights", "eps", None,
                                             suffix_write))

            if root_dir:
                root_dir.cd()
                canvas.Write(f"multi_trial_{suffix_write}")

            canvas.Close()


    @staticmethod
    def save_all_(fits, save_dir):
        """
        Write dictionary of fits to disk together with its parameters
        Args:
            fits: key->fit dictionary of fist
            save_dir: directory where to write
        """

        for i, (key, fit) in enumerate(fits.items()):
            save_dir_fit = join(save_dir, f"fit_{i}")
            annotations = {"key": key}
            save_fit(fit, save_dir_fit, annotations)


    def save_fits(self, top_save_dir):
        """
        Write all fits there are
        Args:
            top_save_dir: parent directory where sub-directory for pre-fits and central fits
                          are created within
        """

        self.save_all_(self.pre_fits_mc, join(top_save_dir, "pre_fits_mc"))
        self.save_all_(self.pre_fits_data, join(top_save_dir, "pre_fits_data"))
        self.save_all_(self.central_fits, join(top_save_dir, "central_fits"))


    @staticmethod
    def load_all_(fits, save_dir):
        """
        Read back a given class of fits (central or pre) from disk
        Args:
            save_dir: directory where they have been written to previously
        Returns:
            Success status: True if fits could be read back, False otherwise
        """

        fit_dirs = [d for d in glob(join(save_dir, "*")) if "fit_" in d]
        if not fit_dirs:
            return False
        for d in fit_dirs:
            fit, annotations = load_fit(d)
            fit.has_attempt = True
            key = annotations["key"]
            try:
                key = tuple(key)
            except TypeError:
                pass
            fits[key] = fit
        return True


    def load_fits(self, top_save_dir):
        """
        Read back all fits written to disk
        Args:
            top_save_dir: directory where they have been written to previously
        Returns:
            Success status: True if fits could be read back, False otherwise
        """

        if self.pre_fits_mc or self.pre_fits_data or self.central_fits:
            self.logger.warning("Overriding fits")
        self.pre_fits_mc = {}
        self.pre_fits_data = {}
        self.central_fits = {}
        success = self.load_all_(self.pre_fits_mc, join(top_save_dir, "pre_fits_mc")) and \
                  self.load_all_(self.pre_fits_data, join(top_save_dir, "pre_fits_data")) and \
                  self.load_all_(self.central_fits, join(top_save_dir, "central_fits"))
        # Flags
        self.is_initialized_fits = True
        self.done_pre_fits = True
        self.done_central_fits = False
        return success
