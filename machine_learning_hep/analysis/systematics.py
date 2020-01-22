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
Main script for doing the systematic studies. Standalone, so some parts similar to analyzer.py

At the moment includes: Cut variation and MC pT shape
The raw yield systematic is done within analyzer.py
"""
# pylint: disable=unused-wildcard-import, wildcard-import
# pylint: disable=no-name-in-module
# pylint: disable=import-error
from os.path import join, exists
from os import makedirs
import math
from array import *
import pickle
from root_numpy import fill_hist, evaluate
from ROOT import gROOT, gPad
from ROOT import TFile, TH1F, TCanvas, TLegend
from ROOT import kRed, kGreen, kBlack, kBlue, kOrange, kViolet, kAzure, kYellow
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.utilities import seldf_singlevar, openfile, make_file_path
from machine_learning_hep.utilities_plot import load_root_style_simple, load_root_style
from machine_learning_hep.utilities import mergerootfiles, get_timestamp_string
from machine_learning_hep.analysis.analyzer import Analyzer, AnalyzerAfterBurner

class SystematicsAfterBurner(AnalyzerAfterBurner):
    # pylint: disable=useless-super-delegation
    def __init__(self, database, case, typean):
        super().__init__(database, case, typean)

    def ml_cutvar_mass(self):

        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/cutvar_mass/" \
                     f"{get_timestamp_string()}/"
        # pylint: disable=not-an-iterable
        files_mass_cutvar = [syst.n_filemass_cutvar for syst in self.analyzers]

        filename_mass = self.datap["files_names"]["histofilename"].replace(".root", "_cutvar.root")
        merged_file = join(self.datap["analysis"][self.typean]["data"]["resultsallp"], "cutvar")
        merged_file = join(merged_file, filename_mass)

        mergerootfiles(files_mass_cutvar, merged_file, tmp_merged)


    def ml_cutvar_eff(self):

        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/cutvar_eff/" \
                     f"{get_timestamp_string()}/"
        # pylint: disable=not-an-iterable
        files_eff_cutvar = [syst.n_fileeff_cutvar for syst in self.analyzers]

        filename_eff = self.datap["files_names"]["efffilename"].replace(".root", "_cutvar.root")
        merged_file = join(self.datap["analysis"][self.typean]["data"]["resultsallp"], "cutvar")
        merged_file = join(merged_file, filename_eff)

        mergerootfiles(files_eff_cutvar, merged_file, tmp_merged)

    def mcptshape(self):

        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/mcptshape_eff/" \
                     f"{get_timestamp_string()}/"
        # pylint: disable=not-an-iterable
        files_eff_cutvar = [syst.n_fileeff_ptshape for syst in self.analyzers]

        filename_eff = self.datap["files_names"]["efffilename"].replace(".root", "_ptshape.root")
        merged_file = join(self.datap["analysis"][self.typean]["data"]["resultsallp"],
                           filename_eff)

        mergerootfiles(files_eff_cutvar, merged_file, tmp_merged)


# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes, too-many-statements, too-many-arguments
# pylint: disable=too-many-branches, too-many-nested-blocks
class Systematics(Analyzer):
    species = "systematics"
    def __init__(self, datap, case, typean, period, run_param):
        super().__init__(datap, case, typean, period)

        self.run_param = run_param
        self.p_period = datap["multi"]["data"]["period"][period] if period is not None \
                else "merged"
        self.d_pkl_decmerged_mc = datap["mlapplication"]["mc"]["pkl_skimmed_decmerged"][period] \
                if period is not None else ""
        self.d_pkl_decmerged_data = \
                datap["mlapplication"]["data"]["pkl_skimmed_decmerged"][period] \
                if period is not None else ""
        self.d_results = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]
        self.d_val = datap["validation"]["data"]["dir"][period] \
                if period is not None else datap["validation"]["data"]["dirmerged"]
        self.runlistrigger = datap["validation"]["runlisttrigger"][self.p_period] \
                if period is not None else None


        self.v_var_binning = datap["var_binning"]
        #Binning used when skimming/training
        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])
        #Analysis binning
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        #Second analysis binning
        self.lvar2_binmin = datap["analysis"][self.typean]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"][self.typean]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]
        self.v_var2_binning_gen = datap["analysis"][self.typean]["var_binning2_gen"]

        #ML model variables
        self.p_modelname = datap["mlapplication"]["modelname"]
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]
        self.lpt_probcutpre_mc = datap["mlapplication"]["probcutpresel"]["mc"]
        self.lpt_probcutpre_data = datap["mlapplication"]["probcutpresel"]["data"]

        #Extra pre-selections
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_presel_gen_eff = datap["analysis"][self.typean]["presel_gen_eff"]
        self.s_trigger_mc = datap["analysis"][self.typean]["triggersel"]["mc"]
        self.s_trigger_data = datap["analysis"][self.typean]["triggersel"]["data"]
        self.apply_weights = datap["analysis"][self.typean]["triggersel"]["weighttrig"]

        #Build names for input pickle files (data, mc_reco, mc_gen)
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.lpt_recodec_mc = [self.n_reco.replace(".pkl", "%d_%d_%.2f.pkl" % \
                              (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                               self.lpt_probcutpre_mc[i])) for i in range(self.p_nptbins)]
        self.lpt_recodec_data = [self.n_reco.replace(".pkl", "%d_%d_%.2f.pkl" % \
                                (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                                 self.lpt_probcutpre_data[i])) for i in range(self.p_nptbins)]
        self.lpt_recodecmerged_mc = [join(self.d_pkl_decmerged_mc, self.lpt_recodec_mc[ipt])
                                     for ipt in range(self.p_nptbins)]
        self.lpt_recodecmerged_data = [join(self.d_pkl_decmerged_data, \
                                       self.lpt_recodec_data[ipt]) for ipt in range(self.p_nptbins)]
        self.lpt_gensk = [self.n_gen.replace(".pkl", "_%s%d_%d.pkl" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_gendecmerged = [join(self.d_pkl_decmerged_mc, self.lpt_gensk[ipt]) \
                                 for ipt in range(self.p_nptbins)]

        #Build names for intermediate output ROOT files
        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_filemass_cutvar = self.n_filemass.replace(".root", "_cutvar.root")
        self.n_fileeff_cutvar = self.n_fileeff.replace(".root", "_cutvar.root")
        self.n_fileeff_ptshape = self.n_fileeff.replace(".root", "_ptshape.root")
        #Build directories for intermediate output ROOT files
        self.d_results_cv = join(self.d_results, "cutvar")
        if not exists(self.d_results_cv):
            makedirs(self.d_results_cv)
        self.n_filemass_cutvar = join(self.d_results_cv, self.n_filemass_cutvar)
        self.n_fileeff_cutvar = join(self.d_results_cv, self.n_fileeff_cutvar)
        self.n_fileeff_ptshape = join(self.d_results, self.n_fileeff_ptshape)
        #Final file names for analyzer.py
        self.yields_filename_std = "yields"
        self.efficiency_filename_std = "efficiencies"
        self.cross_filename_std = "finalcross"
        #Final file names used for systematics
        self.yields_filename = "yields_cutvar"
        self.efficiency_filename = "efficiencies_cutvar"
        self.efficiency_filename_pt = "efficiencies_mcptshape"
        self.cross_filename = "finalcross_cutvar"
        self.ptspectra_filename = "ptspectra_for_weights"

        #Variables for cross section/corrected yield calculation (NB: not all corr are applied)
        self.f_evtnorm = join(self.d_results, "correctionsweights.root")
        self.p_indexhpt = datap["analysis"]["indexhptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_sigmav0 = datap["analysis"]["sigmav0"]
        self.p_bineff = datap["analysis"][self.typean]["usesinglebineff"]
        self.p_fprompt_from_mb = datap["analysis"][self.typean]["fprompt_from_mb"]
        self.p_triggereff = datap["analysis"][self.typean].get("triggereff", [1] * 10)
        self.p_triggereffunc = datap["analysis"][self.typean].get("triggereffunc", [0] * 10)
        self.p_inputfonllpred = datap["analysis"]["inputfonllpred"]

        #Variables for the systematic variations
        self.p_cutvar_minrange = datap["systematics"]["probvariation"]["cutvarminrange"]
        self.p_cutvar_maxrange = datap["systematics"]["probvariation"]["cutvarmaxrange"]
        self.p_ncutvar = datap["systematics"]["probvariation"]["ncutvar"]
        self.p_maxperccutvar = datap["systematics"]["probvariation"]["maxperccutvar"]
        self.p_fixedmean = datap["systematics"]["probvariation"]["fixedmean"]
        self.p_fixedsigma = datap["systematics"]["probvariation"]["fixedsigma"]
        self.p_weights = datap["systematics"]["mcptshape"]["weights"]
        self.p_weights_min_pt = datap["systematics"]["mcptshape"]["weights_min_pt"]
        self.p_weights_max_pt = datap["systematics"]["mcptshape"]["weights_max_pt"]
        self.p_weights_bins = datap["systematics"]["mcptshape"]["weights_bins"]

        #For fitting
        #For mass histos
        self.p_mass_fit_lim = datap["analysis"][self.typean]["mass_fit_lim"]
        self.p_bin_width = datap["analysis"][self.typean]["bin_width"]
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        #For rebinning mass and yield histos
        self.rebins = datap["analysis"][self.typean]["rebin"].copy()
        if not isinstance(self.rebins[0], list):
            self.rebins = [self.rebins for _ in range(len(self.lvar2_binmin))]
        self.ptranges = self.lpt_finbinmin.copy()
        self.ptranges.append(self.lpt_finbinmax[-1])
        #For AliHFInvMassFitter
        self.p_sgnfunc = datap["analysis"][self.typean]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"][self.typean]["bkgfunc"]
        self.sig_fmap = {"kGaus": 0, "k2Gaus": 1, "kGausSigmaRatioPar": 2}
        self.bkg_fmap = {"kExpo": 0, "kLin": 1, "Pol2": 2, "kNoBk": 3, "kPow": 4, "kPowEx": 5}
        self.p_massmin = datap["analysis"][self.typean]["massmin"]
        self.p_massmax = datap["analysis"][self.typean]["massmax"]
        #Extra AliHFInvMassFitter settings
        self.p_dolike = datap["analysis"][self.typean]["dolikelihood"]
        self.p_masspeak = datap["analysis"][self.typean]["masspeak"]
        self.p_sigmaarray = datap["analysis"][self.typean]["sigmaarray"]
        self.p_exclude_nsigma_sideband = datap["analysis"][self.typean]["exclude_nsigma_sideband"]
        self.p_nsigma_signal = datap["analysis"][self.typean]["nsigma_signal"]
        #Options for reflections (e.g. D0)
        self.p_include_reflection = datap["analysis"][self.typean].get("include_reflection", False)
        #Options for second peak of Ds
        self.p_includesecpeaks = datap["analysis"][self.typean].get("includesecpeak", None)
        self.p_widthsecpeak = datap["analysis"][self.typean].get("widthsecpeak", None)
        self.p_masssecpeak = datap["analysis"][self.typean].get("masssecpeak", None)
        self.p_fix_masssecpeaks = datap["analysis"][self.typean].get("fix_masssecpeak", None)
        self.p_fix_widthsecpeak = datap["analysis"][self.typean].get("fix_widthsecpeak", None)
        #Some safety for p_fix_masssecpeaks that is checked with [i][j]
        if self.p_includesecpeaks is not None:
            if self.p_fix_masssecpeaks is None:
                self.p_fix_masssecpeaks = [False for ipt in range(self.p_nptbins)]
            self.p_fix_masssecpeaks = self.p_fix_masssecpeaks.copy()
            if not isinstance(self.p_fix_masssecpeaks[0], list):
                self.p_fix_masssecpeaks = [self.p_fix_masssecpeaks \
                                           for _ in range(len(self.lvar2_binmin))]
        #Some safety for p_includesecpeaks that is checked with [i][j]
        if self.p_includesecpeaks is None:
            self.p_includesecpeaks = [False for ipt in range(self.p_nptbins)]
        self.p_includesecpeaks = self.p_includesecpeaks.copy()
        if not isinstance(self.p_includesecpeaks[0], list):
            self.p_includesecpeaks = [self.p_includesecpeaks for _ in range(len(self.lvar2_binmin))]

        self.min_cv_cut = None
        self.max_cv_cut = None

        # Flag whether some internals methods have been executed
        self.done_mass = False
        self.done_eff = False
        self.done_fit = False

    def get_after_burner(self):
        return SystematicsAfterBurner(self.datap, self.case, self.typean)


    def define_cutvariation_limits(self):
        """
        Cut Variation: Defines the probability cuts based on a max percentage variation (set in DB)
        Produces N (set in DB) tighter and N looser probability cuts
        """

        # Check if that has been run already
        if self.min_cv_cut:
            return

        # Also all periods merged need to define ML cut limits
        if self.period is None:
            self.min_cv_cut = [0.] * self.p_nptfinbins
            self.max_cv_cut = [1.] * self.p_nptfinbins
            return

        self.logger.info("Defining systematic cut variations for period: %s", \
                         self.p_period)

        self.min_cv_cut = []
        self.max_cv_cut = []
        cent_cv_cut = []
        ncutvar_temp = self.p_ncutvar * 2
        for ipt in range(self.p_nptfinbins):

            bin_id = self.bin_matching[ipt]
            df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged_mc[bin_id], "rb"))
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_trigger_mc is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger_mc)

            df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)

            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning_gen, \
                                self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning_gen, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning_gen, \
                                         self.lvar2_binmin[0], \
                                         self.lvar2_binmax[0])
            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning_gen, \
                                        self.lvar2_binmin[0], \
                                        self.lvar2_binmax[0])

            df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
            df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]

            selml_cent = "y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[bin_id])
            df_reco_sel_pr = df_reco_presel_pr.query(selml_cent)
            len_gen_pr = len(df_gen_sel_pr)
            eff_cent = len(df_reco_sel_pr)/len_gen_pr

            stepsmin = \
              (self.lpt_probcutfin[bin_id] - self.p_cutvar_minrange[bin_id]) / ncutvar_temp
            self.min_cv_cut.append(self.lpt_probcutfin[bin_id])
            cent_cv_cut.append(self.lpt_probcutfin[bin_id])
            df_reco_cvmin_pr = df_reco_presel_pr
            for icv in range(ncutvar_temp):
                self.min_cv_cut[ipt] = self.p_cutvar_minrange[bin_id] + icv * stepsmin
                selml_min = "y_test_prob%s>%s" % (self.p_modelname, self.min_cv_cut[ipt])
                df_reco_cvmin_pr = df_reco_cvmin_pr.query(selml_min)
                eff_min = len(df_reco_cvmin_pr)/len_gen_pr
                if eff_cent == 0:
                    break
                if eff_min / eff_cent < 1 + self.p_maxperccutvar:
                    break

            eff_min = len(df_reco_cvmin_pr)/len_gen_pr

            stepsmax = \
              (self.p_cutvar_maxrange[bin_id] - self.lpt_probcutfin[bin_id]) / ncutvar_temp
            self.max_cv_cut.append(self.lpt_probcutfin[bin_id])
            df_reco_cvmax_pr = df_reco_sel_pr
            for icv in range(ncutvar_temp):
                self.max_cv_cut[ipt] = self.lpt_probcutfin[bin_id] + icv * stepsmax
                selml_max = "y_test_prob%s>%s" % (self.p_modelname, self.max_cv_cut[ipt])
                df_reco_cvmax_pr = df_reco_cvmax_pr.query(selml_max)
                eff_max = len(df_reco_cvmax_pr)/len_gen_pr
                if eff_cent == 0:
                    break
                if eff_max / eff_cent < 1 - self.p_maxperccutvar:
                    break

            eff_max = len(df_reco_cvmax_pr)/len_gen_pr

        print("Limits for cut variation defined, based on eff %-var of: ", self.p_maxperccutvar)
        print("--Cut variation minimum: ", self.min_cv_cut)
        print("--Central probability cut: ", cent_cv_cut)
        print("--Cut variation maximum: ", self.max_cv_cut)


    def ml_cutvar_mass(self):
        """
        Cut Variation: Create ROOT file with mass histograms
        Histogram for each variation, for each pT bin, for each 2nd binning bin

        Similar as process_histomass_single(self, index) in processor.py
        """

        # Do this only for a single period
        if self.period is None:
            return

        # Define limits first
        self.define_cutvariation_limits()

        myfile = TFile.Open(self.n_filemass_cutvar, "recreate")

        print("Using run selection for mass histo", self.runlistrigger[self.triggerbit], \
              "for period", self.p_period)
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.lpt_recodecmerged_data[bin_id], "rb"))

            stepsmin = (self.lpt_probcutfin[bin_id] - self.min_cv_cut[ipt]) / self.p_ncutvar
            stepsmax = (self.max_cv_cut[ipt] - self.lpt_probcutfin[bin_id]) / self.p_ncutvar
            ntrials = 2 * self.p_ncutvar + 1
            icvmax = 1

            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger_data is not None:
                df = df.query(self.s_trigger_data)
            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df = selectdfrunlist(df, self.run_param[self.runlistrigger[self.triggerbit]], \
                                 "run_number")

            arr_selml_cv = []
            for icv in range(ntrials):
                if icv < self.p_ncutvar:
                    selml_cvval = self.min_cv_cut[ipt] + icv * stepsmin
                elif icv == self.p_ncutvar:
                    selml_cvval = self.lpt_probcutfin[bin_id]
                else:
                    selml_cvval = self.lpt_probcutfin[bin_id] + icvmax * stepsmax
                    icvmax = icvmax + 1
                selml_cv = "y_test_prob%s>%s" % (self.p_modelname, selml_cvval)

                arr_selml_cv.append(selml_cvval)
                df = df.query(selml_cv)

                for ibin2 in range(len(self.lvar2_binmin)):
                    suffix = "%s%d_%d_%d_%s%.2f_%.2f" % \
                             (self.v_var_binning, self.lpt_finbinmin[ipt],
                              self.lpt_finbinmax[ipt], icv,
                              self.v_var2_binning, self.lvar2_binmin[ibin2],
                              self.lvar2_binmax[ibin2])
                    h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                     self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    h_invmass_weight = TH1F("h_invmass_weight" + suffix, "", self.p_num_bins,
                                            self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])

                    df_bin = seldf_singlevar(df, self.v_var2_binning,
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])

                    fill_hist(h_invmass, df_bin.inv_mass)

                    if "INT7" not in self.triggerbit:
                        fileweight_name = "%s/correctionsweights.root" % self.d_val
                        fileweight = TFile.Open(fileweight_name, "read")
                        namefunction = "funcnorm_%s_%s" % (self.triggerbit, self.v_var2_binning)
                        funcweighttrig = fileweight.Get(namefunction)
                        if funcweighttrig:
                            weights = evaluate(funcweighttrig, df_bin[self.v_var2_binning])
                            weightsinv = [1./weight for weight in weights]
                            fill_hist(h_invmass_weight, df_bin.inv_mass, weights=weightsinv)
                    myfile.cd()
                    h_invmass.Write()
                    h_invmass_weight.Write()

            print(" Selection variations for [", self.lpt_finbinmin[ipt], "-", \
                  self.lpt_finbinmax[ipt], "]:  \n   ", arr_selml_cv)

        self.done_mass = True


    def ml_cutvar_eff(self):
        """
        Cut Variation: Create ROOT file with efficiencies
        Histogram for each variation, for each 2nd binning bin

        Similar as process_efficiency_single(self, index) in processor.py
        """

        # Do this only for a single period
        if self.period is None:
            return

        # Define limits first
        self.define_cutvariation_limits()

        myfile = TFile.Open(self.n_fileeff_cutvar, "recreate")

        h_gen_pr = []
        h_sel_pr = []
        h_gen_fd = []
        h_sel_fd = []

        print("Using run selection for eff histo", self.runlistrigger[self.triggerbit], \
              "for period", self.p_period)
        idx = 0
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.lpt_recodecmerged_mc[bin_id], "rb"))

            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger_mc is not None:
                df = df.query(self.s_trigger_mc)
            df = selectdfrunlist(df, \
                                 self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df = seldf_singlevar(df, self.v_var_binning, self.lpt_finbinmin[ipt], \
                                 self.lpt_finbinmax[ipt])

            df_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
            df_gen = df_gen.query(self.s_presel_gen_eff)
            df_gen = selectdfrunlist(df_gen, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_gen = seldf_singlevar(df_gen, self.v_var_binning, self.lpt_finbinmin[ipt], \
                                     self.lpt_finbinmax[ipt])

            stepsmin = (self.lpt_probcutfin[bin_id] - self.min_cv_cut[ipt]) / self.p_ncutvar
            stepsmax = (self.max_cv_cut[ipt] - self.lpt_probcutfin[bin_id]) / self.p_ncutvar
            ntrials = 2 * self.p_ncutvar + 1
            icvmax = 1

            arr_selml_cv = []
            idx = 0
            for icv in range(ntrials):
                if icv < self.p_ncutvar:
                    selml_cvval = self.min_cv_cut[ipt] + icv * stepsmin
                elif icv == self.p_ncutvar:
                    selml_cvval = self.lpt_probcutfin[bin_id]
                else:
                    selml_cvval = self.lpt_probcutfin[bin_id] + icvmax * stepsmax
                    icvmax = icvmax + 1
                selml_cv = "y_test_prob%s>%s" % (self.p_modelname, selml_cvval)

                arr_selml_cv.append(selml_cvval)
                df = df.query(selml_cv)

                for ibin2 in range(len(self.lvar2_binmin)):
                    stringbin2 = "_%d_%s_%.2f_%.2f" % (icv, \
                                                self.v_var2_binning_gen, \
                                                self.lvar2_binmin[ibin2], \
                                                self.lvar2_binmax[ibin2])

                    if ipt == 0:
                        n_bins = len(self.lpt_finbinmin)
                        analysis_bin_lims_temp = self.lpt_finbinmin.copy()
                        analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
                        analysis_bin_lims = array('f', analysis_bin_lims_temp)
                        h_gen_pr.append(TH1F("h_gen_pr" + stringbin2, \
                                             "Prompt Generated in acceptance |y|<0.5", \
                                             n_bins, analysis_bin_lims))
                        h_sel_pr.append(TH1F("h_sel_pr" + stringbin2, \
                                             "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                                             n_bins, analysis_bin_lims))
                        h_gen_fd.append(TH1F("h_gen_fd" + stringbin2, \
                                             "FD Generated in acceptance |y|<0.5", \
                                             n_bins, analysis_bin_lims))
                        h_sel_fd.append(TH1F("h_sel_fd" + stringbin2, \
                                             "FD Reco and sel in acc |#eta|<0.8 and sel", \
                                             n_bins, analysis_bin_lims))

                    df_bin = seldf_singlevar(df, self.v_var2_binning_gen, \
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                    df_gen_bin = seldf_singlevar(df_gen, self.v_var2_binning_gen, \
                                                 self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])

                    df_sel_pr = df_bin[df_bin.ismcprompt == 1]
                    df_gen_pr = df_gen_bin[df_gen_bin.ismcprompt == 1]
                    df_sel_fd = df_bin[df_bin.ismcfd == 1]
                    df_gen_fd = df_gen_bin[df_gen_bin.ismcfd == 1]

                    h_gen_pr[idx].SetBinContent(ipt + 1, len(df_gen_pr))
                    h_gen_pr[idx].SetBinError(ipt + 1, math.sqrt(len(df_gen_pr)))
                    h_sel_pr[idx].SetBinContent(ipt + 1, len(df_sel_pr))
                    h_sel_pr[idx].SetBinError(ipt + 1, math.sqrt(len(df_sel_pr)))

                    h_gen_fd[idx].SetBinContent(ipt + 1, len(df_gen_fd))
                    h_gen_fd[idx].SetBinError(ipt + 1, math.sqrt(len(df_gen_fd)))
                    h_sel_fd[idx].SetBinContent(ipt + 1, len(df_sel_fd))
                    h_sel_fd[idx].SetBinError(ipt + 1, math.sqrt(len(df_sel_fd)))
                    idx = idx + 1

            print(" Selection variations for [", self.lpt_finbinmin[ipt], "-", \
                  self.lpt_finbinmax[ipt], "]:  \n   ", arr_selml_cv)

        myfile.cd()
        for i in range(idx):
            h_gen_pr[i].Write()
            h_sel_pr[i].Write()
            h_gen_fd[i].Write()
            h_sel_fd[i].Write()

        self.done_eff = True


    # pylint: disable=import-outside-toplevel
    def ml_cutvar_fit(self):
        """
        Cut Variation: Fit invariant mass histograms with AliHFInvMassFitter
        If requested, sigma+mean can be fixed to central fit

        Similar as fitter(self) in analyzer.py
        """

        # Define limits first
        self.define_cutvariation_limits()

        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        from ROOT import AliHFInvMassFitter, AliVertexingHFUtils
        # Enable ROOT batch mode and reset in the end

        load_root_style_simple()

        lfile = TFile.Open(self.n_filemass_cutvar, "READ")

        ntrials = 2 * self.p_ncutvar + 1
        icvmax = 1

        mass_fitter = []
        ifit = 0

        for icv in range(ntrials):

            fileout_name = make_file_path(self.d_results_cv, self.yields_filename, "root", \
                                          None, [self.typean, str(icv)])
            fileout = TFile(fileout_name, "RECREATE")

            yieldshistos = [TH1F("hyields%d" % (imult), "", self.p_nptfinbins, \
                            array("d", self.ptranges)) for imult in range(len(self.lvar2_binmin))]

            if self.p_nptfinbins < 9:
                nx = 4
                ny = 2
                canvy = 533
            elif self.p_nptfinbins < 13:
                nx = 4
                ny = 3
                canvy = 800
            else:
                nx = 5
                ny = 4
                canvy = 1200

            canvas_data = [TCanvas("canvas_cutvar%d_%d" % (icv, imult), "Data", 1000, canvy) \
                           for imult in range(len(self.lvar2_binmin))]

            for imult in range(len(self.lvar2_binmin)):
                canvas_data[imult].Divide(nx, ny)

            for imult in range(len(self.lvar2_binmin)):

                mean_for_data, sigma_for_data = self.load_central_meansigma(imult)

                for ipt in range(self.p_nptfinbins):
                    bin_id = self.bin_matching[ipt]

                    suffix = "%s%d_%d_%d_%s%.2f_%.2f" % \
                             (self.v_var_binning, self.lpt_finbinmin[ipt],
                              self.lpt_finbinmax[ipt], icv,
                              self.v_var2_binning, self.lvar2_binmin[imult],
                              self.lvar2_binmax[imult])

                    stepsmin = (self.lpt_probcutfin[bin_id] - self.min_cv_cut[ipt]) / self.p_ncutvar
                    stepsmax = (self.max_cv_cut[ipt] - self.lpt_probcutfin[bin_id]) / self.p_ncutvar
                    ntrials = 2 * self.p_ncutvar + 1

                    selml_cvval = 0
                    if icv < self.p_ncutvar:
                        selml_cvval = self.min_cv_cut[ipt] + icv * stepsmin
                    elif icv == self.p_ncutvar:
                        selml_cvval = self.lpt_probcutfin[bin_id]
                    else:
                        selml_cvval = self.lpt_probcutfin[bin_id] + icvmax * stepsmax

                    histname = "hmass"
                    if self.apply_weights is True:
                        histname = "h_invmass_weight"
                        self.logger.info("*********** I AM USING WEIGHTED HISTOGRAMS")

                    h_invmass = lfile.Get(histname + suffix)
                    h_invmass_rebin_ = AliVertexingHFUtils.RebinHisto(h_invmass, \
                                                                      self.rebins[imult][ipt], -1)
                    h_invmass_rebin = TH1F()
                    h_invmass_rebin_.Copy(h_invmass_rebin)
                    h_invmass_rebin.SetTitle("%.1f < #it{p}_{T} < %.1f (prob > %.4f)" \
                                             % (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt], \
                                                selml_cvval))
                    h_invmass_rebin.GetXaxis().SetTitle("#it{M}_{inv} (GeV/#it{c}^{2})")
                    h_invmass_rebin.GetYaxis().SetTitle("Entries/(%.0f MeV/#it{c}^{2})" \
                                                        % (h_invmass_rebin.GetBinWidth(1) * 1000))
                    h_invmass_rebin.GetYaxis().SetTitleOffset(1.1)

                    mass_fitter.append(AliHFInvMassFitter(h_invmass_rebin, self.p_massmin[ipt],
                                                          self.p_massmax[ipt],
                                                          self.bkg_fmap[self.p_bkgfunc[ipt]],
                                                          self.sig_fmap[self.p_sgnfunc[ipt]]))

                    if self.p_dolike:
                        mass_fitter[ifit].SetUseLikelihoodFit()

                    mass_fitter[ifit].SetInitialGaussianMean(self.p_masspeak)
                    mass_fitter[ifit].SetInitialGaussianSigma(self.p_sigmaarray[ipt])
                    if self.p_fixedmean:
                        mass_fitter[ifit].SetFixGaussianMean(mean_for_data[ipt])
                    if self.p_fixedsigma:
                        mass_fitter[ifit].SetFixGaussianSigma(sigma_for_data[ipt])

                    mass_fitter[ifit].SetNSigma4SideBands(self.p_exclude_nsigma_sideband)
                    mass_fitter[ifit].SetCheckSignalCountsAfterFirstFit(False)

                    #Reflections to be included
                    if self.p_include_reflection:
                        self.logger.info("Reflections not yet included in cut variation fitter!")

                    if self.p_includesecpeaks[imult][ipt]:
                        secpeakwidth = self.p_widthsecpeak * sigma_for_data[ipt]
                        mass_fitter[ifit].IncludeSecondGausPeak(self.p_masssecpeak, \
                                          self.p_fix_masssecpeaks[imult][ipt], \
                                          secpeakwidth, \
                                          self.p_fix_widthsecpeak)
                    success = mass_fitter[ifit].MassFitter(False)

                    canvas_data[imult].cd(ipt+1)
                    if success != 1:
                        mass_fitter[ifit].GetHistoClone().Draw()
                        self.logger.error("Fit failed for suffix %s", suffix)
                        ifit = ifit + 1
                        continue
                    mass_fitter[ifit].DrawHere(gPad, self.p_nsigma_signal)

                    # Write fitters to file
                    fit_root_dir = fileout.mkdir(suffix)
                    fit_root_dir.WriteObject(mass_fitter[ifit], "fittercutvar")

                    # In case of success == 2, no signal was found, in case of 0, fit failed
                    rawYield = mass_fitter[ifit].GetRawYield()
                    rawYieldErr = mass_fitter[ifit].GetRawYieldError()
                    yieldshistos[imult].SetBinContent(ipt + 1, rawYield)
                    yieldshistos[imult].SetBinError(ipt + 1, rawYieldErr)
                    ifit = ifit + 1

                suffix2 = "cutvar%d_%s%.2f_%.2f" % \
                           (icv, self.v_var2_binning, self.lvar2_binmin[imult], \
                           self.lvar2_binmax[imult])

                canvas_data[imult].SaveAs(make_file_path(self.d_results_cv, "canvas_FinalData", \
                                                         "eps", None, suffix2))
                fileout.cd()
                yieldshistos[imult].Write()

            fileout.Close()
            if icv > self.p_ncutvar:
                icvmax = icvmax + 1

        del mass_fitter[:]
        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

        # Conclude fitting with efficiencies
        self.ml_cutvar_efficiency_after_fit()

        self.done_fit = True

    def ml_cutvar_efficiency_after_fit(self):
        """
        Cut Variation: Extract prompt and feeddown efficiencies

        Similar as efficiency(self) in analyzer.py
        """
        load_root_style_simple()

        lfileeff = TFile.Open(self.n_fileeff_cutvar, "READ")

        ntrials = 2 * self.p_ncutvar + 1
        for icv in range(ntrials):
            fileout_name = make_file_path(self.d_results_cv, self.efficiency_filename, "root", \
                                          None, [self.typean, str(icv)])
            fileout = TFile(fileout_name, "RECREATE")

            for imult in range(len(self.lvar2_binmin)):

                stringbin2 = "_%d_%s_%.2f_%.2f" % (icv, self.v_var2_binning_gen, \
                                                self.lvar2_binmin[imult], \
                                                self.lvar2_binmax[imult])

                h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
                h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
                h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")

                h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
                h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)
                h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")

                fileout.cd()
                h_sel_pr.SetName("eff_mult%d" % imult)
                h_sel_fd.SetName("eff_fd_mult%d" % imult)
                h_sel_pr.Write()
                h_sel_fd.Write()

            fileout.Close()

    # pylint: disable=import-outside-toplevel
    def cutvariation_makenormyields(self):
        """
        Cut Variation: Calculate cross section/corrected yield. NB: Not the full
        normalisation/correction are applied, so results may differ from central

        Similar as makenormyields(self) in analyzer.py
        """
        gROOT.SetBatch(True)
        load_root_style_simple()
        gROOT.LoadMacro("HFPtSpectrum.C")
        from ROOT import HFPtSpectrum, HFPtSpectrum2

        ntrials = 2 * self.p_ncutvar + 1
        for icv in range(ntrials):

            fileouteff = make_file_path(self.d_results_cv, self.efficiency_filename, \
                                        "root", None, [self.typean, str(icv)])
            yield_filename = make_file_path(self.d_results_cv, self.yields_filename, \
                                            "root", None, [self.typean, str(icv)])

            filecrossmb = ""
            for imult in range(len(self.lvar2_binmin)):
                bineff = -1
                if self.p_bineff is None:
                    bineff = imult
                    self.logger.info("Using efficiency for each var2 bin")
                else:
                    bineff = self.p_bineff
                    self.logger.info("Using efficiency always from bin = %f", bineff)

                namehistoeffprompt = "eff_mult%d" % bineff
                namehistoefffeed = "eff_fd_mult%d" % bineff
                nameyield = "hyields%d" % imult
                fileoutcrossmult = make_file_path(self.d_results_cv, self.cross_filename, \
                                                  "root", None, [self.typean, "cutvar", str(icv), \
                                                                 "mult", str(imult)])
                norm = -1
                norm = self.calculate_norm(self.f_evtnorm, self.triggerbit, \
                             self.v_var2_binning_gen, self.lvar2_binmin[imult], \
                             self.lvar2_binmax[imult], self.apply_weights)
                self.logger.info("Not full normalisation is applied. " \
                                 "Result may differ from central.")
                #Keep it simple, don't apply full normalisation

                #Keep it simple, don't correct HM with MB fprompt, but with HM mult-int
                if self.p_fprompt_from_mb is None or imult == 0 or self.p_fd_method != 2:
                    HFPtSpectrum(self.p_indexhpt, self.p_inputfonllpred, \
                     fileouteff, namehistoeffprompt, namehistoefffeed, yield_filename, nameyield, \
                     fileoutcrossmult, norm, self.p_sigmav0 * 1e12, self.p_fd_method, self.p_cctype)
                    filecrossmb = fileoutcrossmult
                else:
                    self.logger.info("Calculating spectra using fPrompt from mult-int.\n  "\
                                         "Assuming mult-int is bin 0:   \n%s", filecrossmb)
                    self.logger.info("HM mult classes take fprompt from HM mult-integrated.\n  " \
                                     "Result may differ from central where MB mult-int is taken.")
                    HFPtSpectrum2(filecrossmb, self.p_triggereff[imult], \
                                  self.p_triggereffunc[imult], fileouteff, \
                                  namehistoeffprompt, namehistoefffeed, \
                                  yield_filename, nameyield, fileoutcrossmult, norm, \
                                  self.p_sigmav0 * 1e12)

            fileoutcrosstot = TFile.Open(make_file_path(self.d_results_cv, self.cross_filename, \
                                                        "root", None, [self.typean, "cutvar", \
                                                        str(icv), "multtot"]), "recreate")

            for imult in range(len(self.lvar2_binmin)):
                fileoutcrossmult = make_file_path(self.d_results_cv, self.cross_filename, \
                                                  "root", None, [self.typean, "cutvar", str(icv), \
                                                                 "mult", str(imult)])
                f_fileoutcrossmult = TFile.Open(fileoutcrossmult)
                if not f_fileoutcrossmult:
                    continue
                hcross = f_fileoutcrossmult.Get("histoSigmaCorr")
                hcross.SetName("histoSigmaCorr%d" % imult)
                fileoutcrosstot.cd()
                hcross.Write()
                f_fileoutcrossmult.Close()
            fileoutcrosstot.Close()


    def ml_cutvar_makeplots(self, plotname):
        """
        Cut Variation: Make final plots.
        For the moment, value should be assigned by analyser
        """

        local_min_cv_cut, local_max_cv_cut = (self.min_cv_cut, self.max_cv_cut) \
                if self.done_mass or self.done_eff or self.done_fit else (None, None)

        load_root_style()

        leg = TLegend(.15, .65, .85, .85)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.024)
        leg.SetNColumns(4)
        colours = [kBlack, kRed, kGreen+2, kBlue, kOrange+2, kViolet-1, \
                   kAzure+1, kOrange-7, kViolet+2, kYellow-3]

        ntrials = 2 * self.p_ncutvar + 1
        for imult in range(len(self.lvar2_binmin)):

            canv = TCanvas("%s%d" % (plotname, imult), '', 400, 400)

            diffratio = 2 * self.p_maxperccutvar
            if plotname == "histoSigmaCorr":
                diffratio = self.p_maxperccutvar + 0.15
            ptmax = self.lpt_finbinmax[-1] + 1
            canv.cd(1).DrawFrame(0, 1 - diffratio, ptmax, 1 + diffratio, \
                                 "%s %.2f < %s < %.2f;#it{p}_{T} (GeV/#it{c});Ratio %s" % \
                                 (self.typean, self.lvar2_binmin[imult], self.v_var2_binning, \
                                  self.lvar2_binmax[imult], plotname))

            fileoutcrossmultref = make_file_path(self.d_results_cv, self.cross_filename, \
                                                 "root", None, [self.typean, "cutvar", \
                                                                str(self.p_ncutvar), \
                                                                "mult", str(imult)])

            f_fileoutcrossmultref = TFile.Open(fileoutcrossmultref)
            href = f_fileoutcrossmultref.Get(plotname)
            imk = 0
            icol = 0
            legname = "looser"
            hcutvar = []

            markers = [20, 21, 22, 23]
            for icv in range(ntrials):
                if icv == self.p_ncutvar:
                    markers = [markers[i] + 4 for i in range(len(markers))]
                    imk = 0
                    legname = "tighter"
                    continue
                if icol == len(colours) - 1:
                    imk = imk + 1
                fileoutcrossmult = make_file_path(self.d_results_cv, self.cross_filename, \
                                                  "root", None, [self.typean, "cutvar", str(icv), \
                                                                 "mult", str(imult)])
                f_fileoutcrossmult = TFile.Open(fileoutcrossmult)
                hcutvar.append(f_fileoutcrossmult.Get(plotname))
                hcutvar[icol].SetDirectory(0)
                hcutvar[icol].SetLineColor(colours[icol % len(colours)])
                hcutvar[icol].SetMarkerColor(colours[icol % len(colours)])
                hcutvar[icol].SetMarkerStyle(markers[imk])
                hcutvar[icol].SetMarkerSize(0.8)
                hcutvar[icol].Divide(href)
                hcutvar[icol].Draw("same")
                if imult == 0:
                    leg.AddEntry(hcutvar[icol], "Set %d (%s)" % (icv, legname), "LEP")
                icol = icol + 1
                f_fileoutcrossmult.Close()
            leg.Draw()
            canv.SaveAs("%s/Cutvar_%s_mult%d.eps" % (self.d_results_cv, plotname, imult))
            f_fileoutcrossmultref.Close()

        if plotname == "histoSigmaCorr":
            if self.p_nptfinbins < 9:
                nx = 4
                ny = 2
                canvy = 533
            elif self.p_nptfinbins < 13:
                nx = 4
                ny = 3
                canvy = 800
            else:
                nx = 5
                ny = 4
                canvy = 1200

            canv = [TCanvas("canvas_corryieldvspt%d_%d" % (icv, imult), "Data", \
                             1000, canvy) for imult in range(len(self.lvar2_binmin))]
            arrhistos = [None for ipt in range(self.p_nptfinbins)]
            for imult in range(len(self.lvar2_binmin)):
                for ipt in range(self.p_nptfinbins):
                    arrhistos[ipt] = TH1F("hcorryieldvscut%d%d" % (imult, ipt), \
                                          "%d < #it{p}_{T} < %d;cut set;Corr. Yield" % \
                                          (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt]),
                                          ntrials, -0.5, ntrials - 0.5)
                    arrhistos[ipt].SetDirectory(0)

                for icv in range(ntrials):
                    fileoutcrossmult = make_file_path(self.d_results_cv, \
                                                      self.cross_filename, "root", None, \
                                                      [self.typean, "cutvar", str(icv), \
                                                       "mult", str(imult)])
                    f_fileoutcrossmult = TFile.Open(fileoutcrossmult)
                    hcutvar2 = f_fileoutcrossmult.Get(plotname)
                    for ipt in range(self.p_nptfinbins):
                        arrhistos[ipt].SetBinContent(icv + 1, hcutvar2.GetBinContent(ipt + 1))
                        arrhistos[ipt].SetBinError(icv + 1, hcutvar2.GetBinError(ipt + 1))
                    f_fileoutcrossmult.Close()

                canv[imult].Divide(nx, ny)
                for ipt in range(self.p_nptfinbins):
                    canv[imult].cd(ipt + 1)
                    arrhistos[ipt].SetLineColor(colours[ipt])
                    arrhistos[ipt].SetMarkerColor(colours[ipt])
                    arrhistos[ipt].Draw("ep")
                canv[imult].SaveAs("%s/Cutvar_CorrYieldvsSet_mult%d.eps" % (self.d_results_cv, \
                                                                            imult))

            if local_min_cv_cut is not None and local_max_cv_cut is not None:

                probcuts = [None for ipt in range(self.p_nptfinbins)]
                probarr = [None for icv in range(2 + 2 * ntrials)]
                for ipt in range(self.p_nptfinbins):
                    bin_id = self.bin_matching[ipt]
                    stepsmin = (self.lpt_probcutfin[bin_id] - local_min_cv_cut[ipt]) / \
                            self.p_ncutvar
                    stepsmax = (local_max_cv_cut[ipt] - self.lpt_probcutfin[bin_id]) / \
                            self.p_ncutvar

                    probarr[0] = 0
                    icvmax = 1
                    for icv in range(ntrials):
                        if icv < self.p_ncutvar:
                            probarr[2 * icv + 1] = local_min_cv_cut[ipt] + (icv - 0.1) * stepsmin
                            probarr[2 * icv + 2] = local_min_cv_cut[ipt] + (icv + 0.1) * stepsmin
                        elif icv == self.p_ncutvar:
                            probarr[2 * icv + 1] = self.lpt_probcutfin[bin_id] - 0.1 * stepsmax
                            probarr[2 * icv + 2] = self.lpt_probcutfin[bin_id] + 0.1 * stepsmax
                        else:
                            probarr[2 * icv + 1] = self.lpt_probcutfin[bin_id] + \
                                                   (icvmax - 0.1) * stepsmax
                            probarr[2 * icv + 2] = self.lpt_probcutfin[bin_id] + \
                                                   (icvmax + 0.1) * stepsmax
                            icvmax = icvmax + 1
                    probarr[-1] = 1
                    probcuts[ipt] = probarr[:]

                canv2 = [TCanvas("canvas_corryieldvsprob%d_%d" % (icv, imult), "Data", \
                                 1000, canvy) for imult in range(len(self.lvar2_binmin))]
                arrhistos2 = [None for ipt in range(self.p_nptfinbins)]
                for imult in range(len(self.lvar2_binmin)):
                    for ipt in range(self.p_nptfinbins):
                        arrhistos2[ipt] = TH1F("hcorryieldvsprob%d%d" % (imult, ipt), \
                                              "%d < #it{p}_{T} < %d;Probability;Corr. Yield" % \
                                              (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt]), \
                                               len(probcuts[ipt]) - 1, array('f', probcuts[ipt]))
                        arrhistos2[ipt].SetDirectory(0)

                    icvmax = 1
                    for icv in range(ntrials):
                        fileoutcrossmult = make_file_path(self.d_results_cv, \
                                                          self.cross_filename, "root", None, \
                                                          [self.typean, "cutvar", str(icv), \
                                                           "mult", str(imult)])
                        f_fileoutcrossmult = TFile.Open(fileoutcrossmult)
                        hcutvar2 = f_fileoutcrossmult.Get(plotname)

                        for ipt in range(self.p_nptfinbins):
                            bin_id = self.bin_matching[ipt]
                            stepsmin = (self.lpt_probcutfin[bin_id] - local_min_cv_cut[ipt]) / \
                                       self.p_ncutvar
                            stepsmax = (local_max_cv_cut[ipt] - self.lpt_probcutfin[bin_id]) / \
                                       self.p_ncutvar
                            selml_cvval = 0
                            if icv < self.p_ncutvar:
                                selml_cvval = local_min_cv_cut[ipt] + icv * stepsmin
                            elif icv == self.p_ncutvar:
                                selml_cvval = self.lpt_probcutfin[bin_id]
                            else:
                                selml_cvval = self.lpt_probcutfin[bin_id] + icvmax * stepsmax
                            ibin = arrhistos2[ipt].FindBin(selml_cvval)
                            arrhistos2[ipt].SetBinContent(ibin, hcutvar2.GetBinContent(ipt + 1))
                            arrhistos2[ipt].SetBinError(ibin, hcutvar2.GetBinError(ipt + 1))
                        if icv > self.p_ncutvar:
                            icvmax = icvmax + 1
                        f_fileoutcrossmult.Close()

                    canv2[imult].Divide(nx, ny)
                    for ipt in range(self.p_nptfinbins):
                        canv2[imult].cd(ipt + 1)
                        arrhistos2[ipt].SetLineColor(colours[ipt])
                        arrhistos2[ipt].SetLineWidth(1)
                        arrhistos2[ipt].SetMarkerColor(colours[ipt])
                        arrhistos2[ipt].Draw("ep")
                    canv2[imult].SaveAs("%s/Cutvar_CorrYieldvsProb_mult%d.eps" % \
                                        (self.d_results_cv, imult))


    def ml_cutvar_cross(self):

        # Make normalized yields first
        self.cutvariation_makenormyields()
        # Then plot for each of these
        for name in ["histoSigmaCorr", "hDirectEffpt", "hFeedDownEffpt", "hRECpt"]:
            self.ml_cutvar_makeplots(name)


    def load_central_meansigma(self, imult):
        """
        Cut Variation: Get parameters (mean and sigma) from central fit
        """
        func_filename_std = make_file_path(self.d_results, self.yields_filename_std, \
                                           "root", None, [self.case, self.typean])

        massfile_std = TFile.Open(func_filename_std, "READ")
        means_histo = massfile_std.Get("hmeanss%d" % (imult))
        sigmas_histo = massfile_std.Get("hsigmas%d" % (imult))

        mean_for_data = []
        sigma_for_data = []
        for ipt in range(self.p_nptfinbins):

            mean_for_data.append(means_histo.GetBinContent(ipt + 1))
            sigma_for_data.append(sigmas_histo.GetBinContent(ipt + 1))

        massfile_std.Close()
        return mean_for_data, sigma_for_data


    def mcptshape_get_generated(self):
        """
        MC pT-shape: Get generated pT spectra from MC to define weights
        """
        fileout_name = make_file_path(self.d_results, self.ptspectra_filename, \
                                      "root", None, [self.typean, self.case])
        myfile = TFile(fileout_name, "RECREATE")

        for ibin2 in range(len(self.lvar2_binmin)):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning_gen, \
                                        self.lvar2_binmin[ibin2], \
                                        self.lvar2_binmax[ibin2])

            h_gen_pr = TH1F("h_gen_pr" + stringbin2, "Prompt Generated in acceptance |y|<0.5", \
                            400, 0, 40)
            h_gen_fd = TH1F("h_gen_fd" + stringbin2, "FD Generated in acceptance |y|<0.5", \
                            400, 0, 40)

            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]

                df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
                df_mc_gen = df_mc_gen.query("abs(y_cand) < 0.5")
                df_mc_gen = selectdfrunlist(df_mc_gen, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")

                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning_gen, \
                                            self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])

                df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
                df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]

                fill_hist(h_gen_pr, df_gen_sel_pr.pt_cand)
                fill_hist(h_gen_fd, df_gen_sel_fd.pt_cand)
            myfile.cd()
            h_gen_pr.Write()
            h_gen_fd.Write()
        myfile.Close()


    def mcptshape_build_efficiencies(self):
        """
        MC pT-shape: Create ROOT file with unweighted and weighted efficiencies
        Histogram for (un)weighted, for each 2nd binning bin

        Similar as process_efficiency_single(self, index) in processor.py
        """
        myfile = TFile.Open(self.n_fileeff_ptshape, "recreate")

        print("Using run selection for eff histo", self.runlistrigger[self.triggerbit], \
              "for period", self.p_period)
        for ibin2 in range(len(self.lvar2_binmin)):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning_gen, \
                                        self.lvar2_binmin[ibin2], \
                                        self.lvar2_binmax[ibin2])

            n_bins = len(self.lpt_finbinmin)
            analysis_bin_lims_temp = self.lpt_finbinmin.copy()
            analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
            analysis_bin_lims = array('f', analysis_bin_lims_temp)

            h_gen_pr = TH1F("h_gen_pr" + stringbin2, \
                            "Prompt Generated in acceptance |y|<0.5", \
                            n_bins, analysis_bin_lims)
            h_presel_pr = TH1F("h_presel_pr" + stringbin2, \
                               "Prompt Reco in acc |#eta|<0.8 and sel", \
                               n_bins, analysis_bin_lims)
            h_sel_pr = TH1F("h_sel_pr" + stringbin2, \
                            "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                            n_bins, analysis_bin_lims)
            h_gen_fd = TH1F("h_gen_fd" + stringbin2, \
                            "FD Generated in acceptance |y|<0.5", \
                            n_bins, analysis_bin_lims)
            h_presel_fd = TH1F("h_presel_fd" + stringbin2, \
                               "FD Reco in acc |#eta|<0.8 and sel", \
                               n_bins, analysis_bin_lims)
            h_sel_fd = TH1F("h_sel_fd" + stringbin2, \
                            "FD Reco and sel in acc |#eta|<0.8 and sel", \
                            n_bins, analysis_bin_lims)

            bincounter = 0
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                selml = "y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[bin_id])

                df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged_mc[bin_id], "rb"))
                if self.s_evtsel is not None:
                    df_mc_reco = df_mc_reco.query(self.s_evtsel)
                if self.s_trigger_mc is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger_mc)
                df_mc_reco = selectdfrunlist(df_mc_reco, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")

                df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
                df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
                df_mc_gen = selectdfrunlist(df_mc_gen, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")

                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning_gen, \
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning_gen, \
                                            self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])

                df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
                df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
                df_reco_sel_pr = None
                df_reco_sel_pr = df_reco_presel_pr.query(selml)

                df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
                df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
                df_reco_sel_fd = None
                df_reco_sel_fd = df_reco_presel_fd.query(selml)

                val = len(df_gen_sel_pr)
                err = math.sqrt(val)
                h_gen_pr.SetBinContent(bincounter + 1, val)
                h_gen_pr.SetBinError(bincounter + 1, err)
                val = len(df_reco_presel_pr)
                err = math.sqrt(val)
                h_presel_pr.SetBinContent(bincounter + 1, val)
                h_presel_pr.SetBinError(bincounter + 1, err)
                val = len(df_reco_sel_pr)
                err = math.sqrt(val)
                h_sel_pr.SetBinContent(bincounter + 1, val)
                h_sel_pr.SetBinError(bincounter + 1, err)

                val = len(df_gen_sel_fd)
                err = math.sqrt(val)
                h_gen_fd.SetBinContent(bincounter + 1, val)
                h_gen_fd.SetBinError(bincounter + 1, err)
                val = len(df_reco_presel_fd)
                err = math.sqrt(val)
                h_presel_fd.SetBinContent(bincounter + 1, val)
                h_presel_fd.SetBinError(bincounter + 1, err)
                val = len(df_reco_sel_fd)
                err = math.sqrt(val)
                h_sel_fd.SetBinContent(bincounter + 1, val)
                h_sel_fd.SetBinError(bincounter + 1, err)

                bincounter = bincounter + 1

            hw_gen_pr = TH1F("h_gen_pr_weight" + stringbin2, "Prompt Generated in acc |y|<0.5", \
                             n_bins, analysis_bin_lims)
            hw_presel_pr = TH1F("h_presel_pr_weight" + stringbin2, \
                                "Prompt Reco in acc |#eta|<0.8 and sel", \
                                 n_bins, analysis_bin_lims)
            hw_sel_pr = TH1F("h_sel_pr_weight" + stringbin2, \
                             "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                             n_bins, analysis_bin_lims)
            hw_gen_fd = TH1F("h_gen_fd_weight" + stringbin2, "FD Generated in acc |y|<0.5", \
                             n_bins, analysis_bin_lims)
            hw_presel_fd = TH1F("h_presel_fd_weight" + stringbin2, \
                                "FD Reco in acc |#eta|<0.8 and sel", \
                                n_bins, analysis_bin_lims)
            hw_sel_fd = TH1F("h_sel_fd_weight" + stringbin2, \
                             "FD Reco and sel in acc |#eta|<0.8 and sel", \
                             n_bins, analysis_bin_lims)

            bincounter = 0
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                selml = "y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[bin_id])

                df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged_mc[bin_id], "rb"))
                if self.s_evtsel is not None:
                    df_mc_reco = df_mc_reco.query(self.s_evtsel)
                if self.s_trigger_mc is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger_mc)
                df_mc_reco = selectdfrunlist(df_mc_reco, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")

                df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
                df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
                df_mc_gen = selectdfrunlist(df_mc_gen, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")

                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning_gen, \
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning_gen, \
                                            self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])

                df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
                df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
                df_reco_sel_pr = None
                df_reco_sel_pr = df_reco_presel_pr.query(selml)

                df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
                df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
                df_reco_sel_fd = None
                df_reco_sel_fd = df_reco_presel_fd.query(selml)

                array_pt_gencand_gen = df_gen_sel_pr.pt_cand.values
                array_pt_recocand_reco_presel = df_reco_presel_pr.pt_cand.values
                array_pt_recocand_reco_sel = df_reco_sel_pr.pt_cand.values

                val, err = self.get_reweighted_count(array_pt_gencand_gen)
                hw_gen_pr.SetBinContent(bincounter + 1, val)
                hw_gen_pr.SetBinError(bincounter + 1, err)
                val, err = self.get_reweighted_count(array_pt_recocand_reco_presel)
                hw_presel_pr.SetBinContent(bincounter + 1, val)
                hw_presel_pr.SetBinError(bincounter + 1, err)
                val, err = self.get_reweighted_count(array_pt_recocand_reco_sel)
                hw_sel_pr.SetBinContent(bincounter + 1, val)
                hw_sel_pr.SetBinError(bincounter + 1, err)

                array_pt_gencand_genfd = df_gen_sel_fd.pt_cand.values
                array_pt_recocand_reco_preselfd = df_reco_presel_fd.pt_cand.values
                array_pt_recocand_reco_selfd = df_reco_sel_fd.pt_cand.values

                val, err = self.get_reweighted_count(array_pt_gencand_genfd)
                hw_gen_fd.SetBinContent(bincounter + 1, val)
                hw_gen_fd.SetBinError(bincounter + 1, err)
                val, err = self.get_reweighted_count(array_pt_recocand_reco_preselfd)
                hw_presel_fd.SetBinContent(bincounter + 1, val)
                hw_presel_fd.SetBinError(bincounter + 1, err)
                val, err = self.get_reweighted_count(array_pt_recocand_reco_selfd)
                hw_sel_fd.SetBinContent(bincounter + 1, val)
                hw_sel_fd.SetBinError(bincounter + 1, err)

                bincounter = bincounter + 1

            myfile.cd()
            h_gen_pr.Write()
            h_presel_pr.Write()
            h_sel_pr.Write()
            h_gen_fd.Write()
            h_presel_fd.Write()
            h_sel_fd.Write()
            hw_gen_pr.Write()
            hw_presel_pr.Write()
            hw_sel_pr.Write()
            hw_gen_fd.Write()
            hw_presel_fd.Write()
            hw_sel_fd.Write()
        myfile.Close()

    def mcptshape_efficiency(self):
        """
        MC pT-shape: Extract prompt and feeddown efficiencies
        Systematic = difference wrt 1 for ratio unweighted / weighted
        """
        load_root_style_simple()

        lfileeff = TFile.Open(self.n_fileeff_ptshape, "READ")

        fileout_name = make_file_path(self.d_results, self.efficiency_filename_pt, \
                                      "root", None, [self.typean, self.case])
        fileout = TFile(fileout_name, "RECREATE")

        for imult in range(len(self.lvar2_binmin)):

            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning_gen, \
                                           self.lvar2_binmin[imult], \
                                           self.lvar2_binmax[imult])

            h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
            h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")

            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")

            hw_gen_pr = lfileeff.Get("h_gen_pr_weight" + stringbin2)
            hw_sel_pr = lfileeff.Get("h_sel_pr_weight" + stringbin2)
            hw_sel_pr.Divide(hw_sel_pr, hw_gen_pr, 1.0, 1.0, "B")

            hw_gen_fd = lfileeff.Get("h_gen_fd_weight" + stringbin2)
            hw_sel_fd = lfileeff.Get("h_sel_fd_weight" + stringbin2)
            hw_sel_fd.Divide(hw_sel_fd, hw_gen_fd, 1.0, 1.0, "B")

            fileout.cd()
            h_sel_pr.SetName("eff_mult%d" % imult)
            h_sel_fd.SetName("eff_fd_mult%d" % imult)
            hw_sel_pr.SetName("eff_weight_mult%d" % imult)
            hw_sel_fd.SetName("eff_weight_fd_mult%d" % imult)
            h_sel_pr.Write()
            h_sel_fd.Write()
            hw_sel_pr.Write()
            hw_sel_fd.Write()
        fileout.Close()

    def mcptshape_makeplots(self):
        """
        MC pT shape: Make final plots.
        For the moment, value should be assigned by analyser
        """
        load_root_style()

        leg = TLegend(.15, .65, .85, .85)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.024)
        colours = [kBlack, kRed]
        markers = [20, 21]

        fileout_name = make_file_path(self.d_results, self.efficiency_filename_pt, \
                                      "root", None, [self.typean, self.case])

        f_fileout = TFile.Open(fileout_name)

        hweights = []
        hnoweights = []
        hfdweights = []
        hfdnoweights = []
        for imult in range(len(self.lvar2_binmin)):

            canv = TCanvas("systmcptshape_%d" % imult, '', 400, 400)
            plotname = "No weights / Weights"
            ptmax = self.lpt_finbinmax[-1] + 1
            canv.cd(1).DrawFrame(0, 0.85, ptmax, 1.15, \
                                 "%s %.2f < %s < %.2f;#it{p}_{T} (GeV/#it{c});Ratio %s" % \
                                 (self.typean, self.lvar2_binmin[imult], self.v_var2_binning, \
                                  self.lvar2_binmax[imult], plotname))

            hweights.append(f_fileout.Get("eff_weight_mult%d" % imult))
            hweights[imult].SetDirectory(0)
            hnoweights.append(f_fileout.Get("eff_mult%d" % imult))
            hnoweights[imult].SetDirectory(0)
            hfdweights.append(f_fileout.Get("eff_weight_fd_mult%d" % imult))
            hfdweights[imult].SetDirectory(0)
            hfdnoweights.append(f_fileout.Get("eff_fd_mult%d" % imult))
            hfdnoweights[imult].SetDirectory(0)

            hnoweights[imult].Divide(hnoweights[imult], hweights[imult], 1.0, 1.0, "B")
            hfdnoweights[imult].Divide(hfdnoweights[imult], hfdweights[imult], 1.0, 1.0, "B")

            hnoweights[imult].SetLineColor(colours[0])
            hnoweights[imult].SetMarkerColor(colours[0])
            hnoweights[imult].SetMarkerStyle(markers[0])
            hnoweights[imult].SetMarkerSize(0.8)
            hnoweights[imult].Draw("same")

            hfdnoweights[imult].SetLineColor(colours[1])
            hfdnoweights[imult].SetMarkerColor(colours[1])
            hfdnoweights[imult].SetMarkerStyle(markers[1])
            hfdnoweights[imult].SetMarkerSize(0.8)
            hfdnoweights[imult].Draw("same")

            if imult == 0:
                leg.AddEntry(hnoweights[imult], "Prompt", "LEP")
                leg.AddEntry(hfdnoweights[imult], "Feed-down", "LEP")

            leg.Draw()
            canv.SaveAs("%s/MCpTshape_Syst_mult%d.eps" % (self.d_results, imult))
        f_fileout.Close()


    def mcptshape(self):

        # Do only per period
        if self.period is not None:
            self.mcptshape_get_generated()
            self.mcptshape_build_efficiencies()

        # Do for all
        self.mcptshape_efficiency()
        self.mcptshape_makeplots()


    def get_reweighted_count(self, arraypt):
        """
        MC pT-shape: Reweight array of pTs from dataframe based on pT weights
        """
        weights = arraypt.copy()
        binwidth = (self.p_weights_max_pt - self.p_weights_min_pt)/self.p_weights_bins
        for j in range(weights.shape[0]):
            pt = arraypt[j]
            if pt - self.p_weights_min_pt < 0:
                self.logger.warning("pT_gen < minimum pT of weights!")
            ptbin_weights = int((pt - self.p_weights_min_pt)/binwidth)
            #improvement: make linear extrapolation with bins next to it
            weights[j] = self.p_weights[ptbin_weights]
        val = sum(weights)
        err = math.sqrt(val)
        return val, err

    def calculate_norm(self, filename, trigger, var, multmin, multmax, doweight):
        """
        General: Calculates number of events used to normalise
        NB: Uncorrected for simplicity as systematic variations, see
        calculate_norm in analyzer.py for full function
        """
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
            self.logger.fatal("MISSING NORMALIZATION MULTIPLICITY")
        binminv = hmult.GetXaxis().FindBin(multmin)
        binmaxv = hmult.GetXaxis().FindBin(multmax)
        norm = hmult.Integral(binminv, binmaxv)
        fileout.Close()
        return norm
