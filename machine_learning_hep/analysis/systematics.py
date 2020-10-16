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
# pylint: disable=no-name-in-module
# pylint: disable=import-error
import sys
from time import sleep
from os.path import join, exists
from os import makedirs
from operator import itemgetter
from copy import deepcopy, copy
from random import shuffle

from ROOT import TFile, TCanvas, TLegend
from ROOT import kRed, kGreen, kBlack, kBlue, kOrange, kViolet, kAzure, kYellow
from ROOT import TGraphErrors

from machine_learning_hep.utilities_plot import load_root_style
from machine_learning_hep.fitting.helpers import MLFitter
from machine_learning_hep.multiprocesser import MultiProcesser
from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict
from machine_learning_hep.logger import get_logger


class SystematicsMLWP: # pylint: disable=too-few-public-methods, too-many-instance-attributes
    species = "systematicsmlwp"

    def __init__(self, datap, case, typean,
                 analyzers, multiprocesser_mc, multiprocesser_data,
                 multi_class_opt=None):

        self.logger = get_logger()
        self.datap = datap
        self.case = case
        self.typean = typean

        self.nominal_analyzer_merged = analyzers[-1]
        # This is used to read some members from
        self.nominal_processer_mc = multiprocesser_mc.process_listsample[0]
        self.multiprocesser_mc = multiprocesser_mc
        self.multiprocesser_data = multiprocesser_data

        #Variables for the systematic variations
        self.p_cutvar_minrange = datap["systematics"]["probvariation"]["cutvarminrange"]
        self.p_cutvar_maxrange = datap["systematics"]["probvariation"]["cutvarmaxrange"]
        self.p_ncutvar = datap["systematics"]["probvariation"]["ncutvar"]
        self.p_maxperccutvar = datap["systematics"]["probvariation"]["maxperccutvar"]
        self.p_fixedmean = datap["systematics"]["probvariation"]["fixedmean"]
        self.p_fixedsigma = datap["systematics"]["probvariation"]["fixedsigma"]
        # Require a minimum significance or a maximum chi2 for individual fits
        self.min_signif_fit = datap["systematics"]["probvariation"].get("min_signif_fit", -1.)
        self.max_red_chi2_fit = datap["systematics"]["probvariation"].get("max_red_chi2_fit", -1.)

        self.syst_out_dir = "ML_WP_syst"
        self.processers_mc_syst = None
        self.processers_data_syst = None
        self.analyzers_syst = None

        self.n_trials = 2 * self.p_ncutvar

        self.successful_write = None

        # Central WPs as well as lower and upper boundaries according to
        # efficiency threshold
        self.cent_cv_cut = []
        self.cent_cv_cut_orig = []
        self.min_cv_cut = []
        self.max_cv_cut = []
        # Derived working points
        self.ml_wps = []

        self.nominal_means = []
        self.nominal_sigmas = []

        #For multiclassification. Combined variations not yet implemented
        self.mcopt = multi_class_opt
        if self.mcopt is not None:
            if self.mcopt > len(self.p_cutvar_minrange[0]) - 1:
                self.logger.fatal("Multitrial option for systematic not valid")
            self.p_cutvar_minrange = list(map(itemgetter(self.mcopt), self.p_cutvar_minrange))
            self.p_cutvar_maxrange = list(map(itemgetter(self.mcopt), self.p_cutvar_maxrange))
            self.syst_out_dir = f"ML_WP_syst_MultiClass{self.mcopt}"


    def __read_nominal_fit_values(self):

        if self.nominal_means:
            return

        # An analyzer has it only if the fitting procedure was done
        # so if the nominal analyzer was run in the same process,
        # we can just obtain everything from it already
        fitter = self.nominal_analyzer_merged.fitter

        if fitter is None:

            fitter = MLFitter(self.nominal_analyzer_merged.case,
                              self.nominal_analyzer_merged.datap,
                              self.nominal_analyzer_merged.typean,
                              self.nominal_analyzer_merged.n_filemass,
                              self.nominal_analyzer_merged.n_filemass_mc)
            fitter.load_fits(self.nominal_analyzer_merged.fits_dirname)

        ana_n_first_binning = self.nominal_analyzer_merged.p_nptbins
        ana_n_second_binning = self.nominal_analyzer_merged.p_nbin2

        self.nominal_means = [[None] * ana_n_first_binning \
                for _ in range(ana_n_second_binning)]
        self.nominal_sigmas = [[None] * ana_n_first_binning \
                for _ in range(ana_n_second_binning)]

        for ibin1 in range(ana_n_first_binning):
            for ibin2 in range(ana_n_second_binning):
                fit = fitter.get_central_fit(ibin1, ibin2)
                self.nominal_means[ibin2][ibin1] = fit.kernel.GetMean()
                self.nominal_sigmas[ibin2][ibin1] = fit.kernel.GetSigma()


    def __define_cutvariation_limits(self): #pylint: disable=too-many-statements
        """obtain ML WP limits (lower/upper) keeping required efficiency variation

        This runs a MultiProcesser and an Analyzer both derived from the nominal
        Processer and Analyzer class.

        Both are run as long as the boundaries are found.

        Boundaries are searched for in the analysis with all periods merged and are
        defined for the MB integrated multiplicity bin.

        """

        # Only do this once for all
        if self.cent_cv_cut:
            return

        # use multiprocesser here, prepare database
        datap = deepcopy(self.datap)

        results_dirs_periods = [join(d, "tmp_ml_wp_limits") \
                for d in datap["analysis"][self.typean]["mc"]["results"]]
        results_dir_all = join(datap["analysis"][self.typean]["mc"]["resultsallp"],
                               "tmp_ml_wp_limits")

        datap["analysis"][self.typean]["mc"]["results"] = results_dirs_periods
        datap["analysis"][self.typean]["mc"]["resultsallp"] = results_dir_all

        for rdp in results_dirs_periods:
            if exists(rdp):
                continue
            makedirs(rdp)
        if not exists(results_dir_all):
            makedirs(results_dir_all)

        # MultiProcesser to cover all at once
        multi_processer_effs = MultiProcesser(self.case, self.nominal_processer_mc.__class__, datap,
                                              self.typean, self.multiprocesser_mc.run_param,
                                              "mc")

        # construct analyzer for all periods merged and use it for finding ML WP boundaries
        analyzer_effs = self.nominal_analyzer_merged.__class__(datap, self.case, self.typean, None)

        n_pt_bins = self.nominal_processer_mc.p_nptfinbins

        self.cent_cv_cut = self.nominal_processer_mc.lpt_probcutfin
        self.cent_cv_cut_orig = self.nominal_processer_mc.lpt_probcutfin
        if self.mcopt is not None:
            self.cent_cv_cut = list(map(itemgetter(self.mcopt), self.cent_cv_cut))

        ana_n_first_binning = analyzer_effs.p_nptbins

        bin_matching = self.nominal_analyzer_merged.bin_matching

        nominal_effs = [None] * ana_n_first_binning
        for ibin1 in range(ana_n_first_binning):
            nominal_effs[ibin1], _ = self.nominal_analyzer_merged.get_efficiency(ibin1, 0)

        self.min_cv_cut = [None] * ana_n_first_binning
        self.max_cv_cut = [None] * ana_n_first_binning

        ncutvar_temp = self.p_ncutvar * 2

        stepsmin = []
        stepsmax = []

        modelname = self.nominal_processer_mc.p_modelname
        multiclasslabels = self.nominal_processer_mc.multiclass_labels

        def found_all_boundaries(boundaries):
            """helper to check whether all boundaries have been fixed
            """
            if None in boundaries:
                return False
            return True


        def compute_new_boundaries(wps, boundaries):
            """helper to compute boundaries if not yet fixed
            """
            if found_all_boundaries(boundaries):
                return
            wps_strings = ["y_test_prob%s>%s" % (modelname, wps[ipt]) \
                    for ipt in range(n_pt_bins)]
            probvar0 = 'y_test_prob' + modelname + multiclasslabels[0]
            probvar1 = 'y_test_prob' + modelname + multiclasslabels[1]
            if self.mcopt == 0:
                wps_strings = ["%s<=%s and %s>=%s" % (probvar0, wps[ipt], probvar1, \
                               self.cent_cv_cut_orig[ipt][1]) for ipt in range(n_pt_bins)]
                wps_multi = [[wps[ipt], self.cent_cv_cut_orig[ipt][1]] for ipt in range(n_pt_bins)]
            elif self.mcopt == 1:
                wps_strings = ["%s<=%s and %s>=%s" % (probvar0, self.cent_cv_cut_orig[ipt][0], \
                               probvar1, wps[ipt]) for ipt in range(n_pt_bins)]
                wps_multi = [[self.cent_cv_cut_orig[ipt][0], wps[ipt]] for ipt in range(n_pt_bins)]

            # update processers and analyzer ML WPs
            for proc in multi_processer_effs.process_listsample:
                proc.l_selml = wps_strings
            analyzer_effs.lpt_probcutfin = wps
            if self.mcopt is not None:
                analyzer_effs.lpt_probcutfin = wps_multi

            # Run both
            multi_processer_effs.multi_efficiency()
            analyzer_effs.efficiency()

            # Read and compare efficiencies to nominal ones. Add if not yet found
            for ibin1 in range(ana_n_first_binning):
                eff_new, _ = analyzer_effs.get_efficiency(ibin1, 0)
                if abs(eff_new - nominal_effs[ibin1]) / nominal_effs[ibin1] < self.p_maxperccutvar \
                        and boundaries[ibin1] is None:
                    boundaries[ibin1] = wps[bin_matching[ibin1]]


        # Define stepping up and down from nominal WPs
        for ipt in range(n_pt_bins):

            stepsmin.append( \
              (self.cent_cv_cut[ipt] - self.p_cutvar_minrange[ipt]) / ncutvar_temp)

            stepsmax.append( \
              (self.p_cutvar_maxrange[ipt] - self.cent_cv_cut[ipt]) / ncutvar_temp)

        # Attempt to find WP variations up and down
        for icv in range(ncutvar_temp):
            if found_all_boundaries(self.min_cv_cut) \
                    and found_all_boundaries(self.max_cv_cut):
                break

            wps = [self.p_cutvar_minrange[ipt] + icv * stepsmin[ipt] for ipt in range(n_pt_bins)]
            compute_new_boundaries(wps, self.min_cv_cut)
            wps = [self.p_cutvar_maxrange[ipt] - icv * stepsmax[ipt] for ipt in range(n_pt_bins)]
            compute_new_boundaries(wps, self.max_cv_cut)

        print("Limits for cut variation defined, based on eff %-var of: ", self.p_maxperccutvar)
        print("--Cut variation boundaries minimum: ", self.min_cv_cut)
        print("--Central probability cut: ", self.cent_cv_cut)
        print("--Cut variation boundaries maximum: ", self.max_cv_cut)



    def __make_working_points(self):
        self.ml_wps = [[] for _ in range(self.n_trials)]

        n_pt_bins = self.nominal_processer_mc.p_nptfinbins
        for ipt in range(n_pt_bins):

            stepsmin = (self.cent_cv_cut[ipt] - self.min_cv_cut[ipt]) / self.p_ncutvar
            stepsmax = (self.max_cv_cut[ipt] - self.cent_cv_cut[ipt]) / self.p_ncutvar

            for icv in range(self.p_ncutvar):
                lower_cut = self.min_cv_cut[ipt] + icv * stepsmin
                upper_cut = self.cent_cv_cut[ipt] + (icv + 1) * stepsmax

                if self.mcopt == 0:
                    self.ml_wps[icv].append([lower_cut, self.cent_cv_cut_orig[ipt][1]])
                    self.ml_wps[self.p_ncutvar + icv].append([upper_cut, \
                                                              self.cent_cv_cut_orig[ipt][1]])
                elif self.mcopt == 1:
                    self.ml_wps[icv].append([self.cent_cv_cut_orig[ipt][0], lower_cut])
                    self.ml_wps[self.p_ncutvar + icv].append([self.cent_cv_cut_orig[ipt][0], \
                                                              upper_cut])
                else:
                    self.ml_wps[icv].append(lower_cut)
                    self.ml_wps[self.p_ncutvar + icv].append(upper_cut)

    def __prepare_trial(self, i_trial):


        datap = deepcopy(self.datap)
        datap["analysis"][self.typean]["mc"]["results"] = \
                [join(d, self.syst_out_dir, f"trial_{i_trial}") \
                for d in datap["analysis"][self.typean]["mc"]["results"]]
        datap["analysis"][self.typean]["mc"]["resultsallp"] = \
                join(datap["analysis"][self.typean]["mc"]["resultsallp"], \
                self.syst_out_dir, f"trial_{i_trial}")

        datap["analysis"][self.typean]["data"]["results"] = \
                [join(d, self.syst_out_dir, f"trial_{i_trial}") \
                for d in datap["analysis"][self.typean]["data"]["results"]]
        datap["analysis"][self.typean]["data"]["resultsallp"] = \
                join(datap["analysis"][self.typean]["data"]["resultsallp"], \
                self.syst_out_dir, f"trial_{i_trial}")

        for new_dir in \
                datap["analysis"][self.typean]["mc"]["results"] + \
                [datap["analysis"][self.typean]["mc"]["resultsallp"]] + \
                datap["analysis"][self.typean]["data"]["results"] + \
                [datap["analysis"][self.typean]["data"]["resultsallp"]]:
            if not exists(new_dir):
                makedirs(new_dir)

        datap["analysis"][self.typean]["probcuts"] = self.ml_wps[i_trial]

        # For now take PDG mean
        # However, we could have means and sigmas per pT AND mult, but for that the DB logic
        # has to be changed
        datap["analysis"][self.typean]["FixedMean"] = True
        datap["analysis"][self.typean]["masspeak"] = self.nominal_means
        datap["analysis"][self.typean]["sigmaarray"] = self.nominal_sigmas[0]
        datap["analysis"][self.typean]["SetFixGaussianSigma"] = \
                [True] * len(self.nominal_sigmas[0])
        datap["analysis"][self.typean]["SetInitialGaussianSigma"] = \
                [True] * len(self.nominal_sigmas[0])
        datap["analysis"][self.typean]["SetInitialGaussianMean"] = \
                [True] * len(self.nominal_sigmas[0])

        # Processers
        self.processers_mc_syst[i_trial] = MultiProcesser(self.case,
                                                          self.nominal_processer_mc.__class__,
                                                          datap, self.typean,
                                                          self.multiprocesser_mc.run_param, "mc")
        self.processers_data_syst[i_trial] = MultiProcesser(self.case,
                                                            self.nominal_processer_mc.__class__,
                                                            datap, self.typean,
                                                            self.multiprocesser_mc.run_param,
                                                            "data")

        self.analyzers_syst[i_trial] = self.nominal_analyzer_merged.__class__(datap, self.case,
                                                                              self.typean, None)


    def __ml_cutvar_mass(self, i_trial):
        """
        Cut Variation: Create ROOT file with mass histograms
        Histogram for each variation, for each pT bin, for each 2nd binning bin

        Similar as process_histomass_single(self, index) in processor.py
        """

        self.processers_mc_syst[i_trial].multi_histomass()
        self.processers_data_syst[i_trial].multi_histomass()


    def __ml_cutvar_eff(self, i_trial):
        """
        Cut Variation: Create ROOT file with efficiencies
        Histogram for each variation, for each 2nd binning bin

        Similar as process_efficiency_single(self, index) in processor.py
        """

        self.processers_mc_syst[i_trial].multi_efficiency()


    def __ml_cutvar_ana(self, i_trial):
        """
        Cut Variation: Fit invariant mass histograms with AliHFInvMassFitter
        If requested, sigma+mean can be fixed to central fit

        Similar as fitter(self) in analyzer.py
        """

        self.analyzers_syst[i_trial].fit()
        self.analyzers_syst[i_trial].efficiency()
        self.analyzers_syst[i_trial].makenormyields()
        self.analyzers_syst[i_trial].plotternormyields()

    @staticmethod
    def __style_histograms(histos, style_numbers=None):
        colours = [kRed, kGreen+2, kBlue, kOrange+2, kViolet-1, kAzure+1, kOrange-7,
                   kViolet+2, kYellow-3]
        linestyles = [1, 7, 19]
        markers_closed = [43, 47, 20, 22, 23]
        markers_open = [42, 46, 24, 26, 32]

        if not style_numbers:
            style_numbers = [0] * len(histos)

        for i, (h, s) in enumerate(zip(histos, style_numbers)):
            markers = markers_closed if s == 0 else markers_open
            h.SetLineColor(colours[i % len(colours)])
            h.SetLineStyle(linestyles[i % len(linestyles)])
            h.SetMarkerStyle(markers[i % len(markers)])
            h.SetMarkerColor(colours[i % len(colours)])


    @staticmethod
    def __get_histogram(filepath, name):
        file_in = TFile.Open(filepath, "READ")
        histo = file_in.Get(name)
        histo.SetDirectory(0)
        return histo


    @staticmethod
    def __adjust_min_max(histos):
        h_min = min([h.GetMinimum() for h in histos])
        h_max = max([h.GetMaximum() for h in histos])

        delta = h_max - h_min

        h_min = h_min - 0.1 * delta
        h_max = h_max + 0.75 * delta

        for h in histos:
            h.GetYaxis().SetRangeUser(h_min, h_max)
            h.GetYaxis().SetMaxDigits(3)


    def __make_single_plot(self, name, ibin2, successful):

        # Nominal histogram
        successful_tmp = copy(successful)
        successful_tmp.sort()

        filename = f"finalcross{self.case}{self.typean}mult{ibin2}.root"
        filepath = join(self.nominal_analyzer_merged.d_resultsallpdata, filename)
        nominal_histo = self.__get_histogram(filepath, name)
        nominal_histo.SetLineColor(kBlack)
        nominal_histo.SetMarkerStyle(1)
        nominal_histo.SetLineWidth(3)

        histos = []
        legend_strings = []
        style_numbers = []
        for succ in successful_tmp:
            filename = f"finalcross{self.case}{self.typean}mult{ibin2}.root"
            filepath = join(self.analyzers_syst[succ].d_resultsallpdata, filename)
            leg_string = "higher" if succ >= self.p_ncutvar else "lower"
            style_number = 0 if succ >= self.p_ncutvar else 1
            legend_strings.append(leg_string)
            style_numbers.append(style_number)
            histos.append(self.__get_histogram(filepath, name))

        for h in histos:
            h.Divide(nominal_histo)
        nominal_histo.Divide(nominal_histo)

        self.__style_histograms(histos, style_numbers)

        legend = TLegend(0.15, 0.7, 0.85, 0.89)
        legend.SetNColumns(4)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.02)

        histos.insert(0, nominal_histo)
        legend_strings.insert(0, "nominal")
        style_numbers.insert(0, 0)
        for h, l in zip(histos, legend_strings):
            h.SetTitle("")
            legend.AddEntry(h, l)
            h.GetXaxis().SetTitle("#it{p}_{T} [GeV/#it{c}]")
            h.GetYaxis().SetTitle("WP variation / nominal")
        self.__adjust_min_max(histos, )


        canvas = TCanvas("c", "", 800, 800)
        canvas.cd()

        for h in histos:
            h.Draw("same")
        legend.Draw("same")

        save_path = join(self.nominal_analyzer_merged.d_resultsallpdata, self.syst_out_dir,
                         f"ml_wp_syst_{name}_ibin2_{ibin2}.eps")
        canvas.SaveAs(save_path)
        canvas.Close()

    def __make_summary_plot(self, name, ibin2, successful):

        # Nominal histogram
        successful_tmp = copy(successful)
        successful_tmp.sort()

        filename = f"finalcross{self.case}{self.typean}mult{ibin2}.root"
        filepath = join(self.nominal_analyzer_merged.d_resultsallpdata, filename)
        nominal_histo = self.__get_histogram(filepath, name)

        histos = []
        ml_trials = []
        for succ in successful_tmp:
            filename = f"finalcross{self.case}{self.typean}mult{ibin2}.root"
            filepath = join(self.analyzers_syst[succ].d_resultsallpdata, filename)
            histos.append(self.__get_histogram(filepath, name))
            ml_trials.append(list(map(itemgetter(self.mcopt), self.ml_wps[succ])))

        nptbins = self.nominal_processer_mc.p_nptfinbins
        gr = [TGraphErrors(0) for _ in range(nptbins)]
        for ipt in range(nptbins):
            gr[ipt].SetTitle("pT bin %d" % ipt)
            gr[ipt].SetPoint(0, self.cent_cv_cut[ipt], nominal_histo.GetBinContent(ipt+1))
            gr[ipt].SetPointError(0, 0.0001, nominal_histo.GetBinError(ipt+1))
            for iml, succ in enumerate(successful_tmp):
                gr[ipt].SetPoint(iml + 1, ml_trials[succ][ipt],
                                 histos[succ].GetBinContent(ipt+1))
                gr[ipt].SetPointError(iml + 1, 0.0001, histos[succ].GetBinError(ipt+1))

        canvas = TCanvas("cvsml%d" % ibin2, "", 1200, 800)
        if len(gr) <= 6:
            canvas.Divide(3, 2)
        elif len(gr) <= 12:
            canvas.Divide(4, 3)
        else:
            canvas.Divide(5, 4)
        for i, graph in enumerate(gr):
            canvas.cd(i+1)
            graph.Draw("a*")

        save_path = join(self.nominal_analyzer_merged.d_resultsallpdata, self.syst_out_dir,
                         f"ml_wp_syst_{name}_vs_MLcut_ibin2_{ibin2}.eps")
        canvas.SaveAs(save_path)
        canvas.Close()

    def __plot(self, successful):
        """summary plots
        """

        load_root_style()

        for name in ["histoSigmaCorr", "hDirectEffpt", "hFeedDownEffpt", "hRECpt"]:
            for ibin2 in range(self.nominal_analyzer_merged.p_nbin2):
                self.__make_single_plot(name, ibin2, successful)
        for ibin2 in range(self.nominal_analyzer_merged.p_nbin2):
            self.__make_summary_plot("histoSigmaCorr", ibin2, successful)

    def __write_working_points(self):
        write_yaml = {"central": self.cent_cv_cut,
                      "lower_limits": self.min_cv_cut,
                      "upper_limits": self.max_cv_cut,
                      "working_points": self.ml_wps}
        save_path = join(self.nominal_analyzer_merged.d_resultsallpdata, self.syst_out_dir,
                         "working_points.yaml")
        dump_yaml_from_dict(write_yaml, save_path)


    def __load_working_points(self):
        save_path = join(self.nominal_analyzer_merged.d_resultsallpdata, self.syst_out_dir,
                         "working_points.yaml")
        if not exists(save_path):
            print(f"Cannot load working points. File {save_path} doesn't exist")
            sys.exit(1)
        read_yaml = parse_yaml(save_path)

        self.cent_cv_cut = read_yaml["central"]
        self.min_cv_cut = read_yaml["lower_limits"]
        self.max_cv_cut = read_yaml["upper_limits"]
        self.ml_wps = read_yaml["working_points"]


    def __add_trial_to_save(self, i_trial):
        if self.successful_write is None:
            self.successful_write = []
        self.successful_write.append(i_trial)


    def __write_successful_trials(self):
        if not self.successful_write:
            return
        write_yaml = {"successful_trials": self.successful_write}
        save_path = join(self.nominal_analyzer_merged.d_resultsallpdata, self.syst_out_dir,
                         "successful_trials.yaml")
        dump_yaml_from_dict(write_yaml, save_path)


    def __read_successful_trials(self):
        save_path = join(self.nominal_analyzer_merged.d_resultsallpdata, self.syst_out_dir,
                         "successful_trials.yaml")
        if not exists(save_path):
            print(f"Cannot load working points. File {save_path} doesn't (yet) exist.")
            print("Do full syst in 10s...")
            sleep(10)
            return []
        return parse_yaml(save_path)["successful_trials"]


    def ml_systematics(self, do_only_analysis=False, resume=False):
        """central method to call for ML WP systematics
        """

        # Make sure the summary directory exists aleady
        save_path = join(self.nominal_analyzer_merged.d_resultsallpdata, self.syst_out_dir)
        if not exists(save_path):
            makedirs(save_path)

        successful = []
        self.processers_mc_syst = [None] * self.n_trials
        self.processers_data_syst = [None] * self.n_trials
        self.analyzers_syst = [None] * self.n_trials

        # This step has to be regardless
        steps = [self.__prepare_trial]

        if do_only_analysis and resume:
            print("EITHER do only the anaysis step OR resume")
            sys.exit(1)

        if do_only_analysis or resume:
            # Only analysis part, so attempt to read the working points
            # which were dumped to YAML before.
            self.__load_working_points()
            shuffled = self.__read_successful_trials()

        if not resume and not do_only_analysis:
            # Otherwise we go through the entire heavy chain
            self.__define_cutvariation_limits()
            self.__make_working_points()
            # Write working points so we can read them in later
            self.__write_working_points()
            # Shuffle --> Doing some larger and smaller variations in case of keyboard interrupt
            shuffled = list(range(self.n_trials))

        if resume or not do_only_analysis:
            steps.append(self.__ml_cutvar_mass)
            steps.append(self.__ml_cutvar_eff)
            # Only when doing the heavy processer part we consider writing a successful
            # trial because the analysis can be done quickly
            steps.append(self.__add_trial_to_save)

        # Obtain nominal means and sigmas
        self.__read_nominal_fit_values()

        if resume:
            for s in shuffled:
                successful.append(s)
                self.__prepare_trial(s)
                self.__add_trial_to_save(s)
            shuffled = [i for i in range(self.n_trials) if i not in shuffled]

        shuffle(shuffled)

        # This is always done at the end
        steps.append(self.__ml_cutvar_ana)

        try:
            for i in shuffled:
                for step in steps:
                    step(i)
                successful.append(i)
        except KeyboardInterrupt:
            pass

        self.__write_successful_trials()
        self.__plot(successful)
