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
import numpy as np
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend, TLine, TText
from ROOT import gROOT
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed
from ROOT import TLatex
from machine_learning_hep.globalfitter import Fitter
from  machine_learning_hep.logger import get_logger
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
        self.p_sgnfunc = datap["analysis"][self.typean]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"][self.typean]["bkgfunc"]
        self.p_masspeak = datap["analysis"][self.typean]["masspeak"]
        self.p_massmin = datap["analysis"][self.typean]["massmin"]
        self.p_massmax = datap["analysis"][self.typean]["massmax"]
        self.p_rebin = datap["analysis"][self.typean]["rebin"]
        self.p_includesecpeak = datap["analysis"][self.typean]["includesecpeak"]
        self.p_masssecpeak = datap["analysis"][self.typean]["masssecpeak"]
        self.p_fixedmean = datap["analysis"][self.typean]["FixedMean"]
        self.p_use_user_gauss_sigma = datap["analysis"][self.typean]["SetInitialGaussianSigma"]
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
        print(self.var2ranges)
        self.lmult_yieldshisto = [TH1F("hyields%d" % (imult), "", \
            self.p_nptbins, array("d", self.ptranges)) for imult in range(self.p_nbin2)]

        self.p_nevents = datap["analysis"][self.typean]["nevents"]
        self.p_bineff = datap["analysis"][self.typean]["usesinglebineff"]
        self.p_sigmamb = datap["ml"]["opt"]["sigma_MB"]
        self.p_br = datap["ml"]["opt"]["BR"]

        self.d_valevtdata = valdata
        self.d_valevtmc = valmc

        self.f_evtvaldata = os.path.join(self.d_valevtdata, self.n_evtvalroot)
        self.f_evtvalmc = os.path.join(self.d_valevtmc, self.n_evtvalroot)

        # Systematics
        syst_dict = datap["analysis"][self.typean].get("systematics", None)
        self.p_max_chisquare_ndf_syst = syst_dict["max_chisquare_ndf"] \
                if syst_dict is not None else None
        self.p_rebin_syst = syst_dict["rebin"] if syst_dict is not None else None
        self.p_fit_ranges_low_syst = syst_dict["massmin"] if syst_dict is not None else None
        self.p_fit_ranges_up_syst = syst_dict["massmax"] if syst_dict is not None else None
        self.p_bincount_sigma_syst = syst_dict["bincount_sigma"] if syst_dict is not None else None


    @staticmethod
    def loadstyle():
        gROOT.SetStyle("Plain")
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(1)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)
        gStyle.SetOptTitle(0)


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


    def fitter(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        self.loadstyle()
        mass_fitter = Fitter()

        lfile = TFile.Open(self.n_filemass)
        lfile_mc = TFile.Open(self.n_filemass_mc, "READ")

        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")

        for imult in range(self.p_nbin2):
            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                h_invmass = lfile.Get("hmass" + suffix)
                h_invmass_mc = lfile_mc.Get("hmass" + suffix)

                # First do it for MC only
                mass_fitter.initialize(h_invmass_mc, self.p_sgnfunc[ipt], self.p_bkgfunc[ipt],
                                       self.p_rebin[ipt], self.p_masspeak, self.p_sigmaarray[ipt],
                                       False, False, self.p_exclude_nsigma_sideband,
                                       self.p_nsigma_signal, self.p_massmin[ipt],
                                       self.p_massmax[ipt])
                mass_fitter.do_likelihood()
                mass_fitter.fit()
                mass_fitter.draw_fit(self.make_file_path(self.d_resultsallpdata, "fittedplot_mc",
                                                         "eps", None, suffix))

                # And now with data
                mass_fitter.initialize(h_invmass, self.p_sgnfunc[ipt], self.p_bkgfunc[ipt],
                                       self.p_rebin[ipt], mass_fitter.mean_fit,
                                       mass_fitter.sigma_fit, self.p_fixedmean,
                                       self.p_fixingaussigma, self.p_exclude_nsigma_sideband,
                                       self.p_nsigma_signal, self.p_massmin[ipt],
                                       self.p_massmax[ipt])
                mass_fitter.fit()
                mass_fitter.draw_fit(self.make_file_path(self.d_resultsallpdata, "fittedplot",
                                                         "eps", None, suffix))

                fileout.cd()
                save_dir = fileout.mkdir(suffix)
                mass_fitter.save(save_dir)

                rawYield = mass_fitter.yield_sig / \
                        (self.lpt_finbinmax[ipt] - self.lpt_finbinmin[ipt])
                rawYieldErr = mass_fitter.yield_sig_err / \
                        (self.lpt_finbinmax[ipt] - self.lpt_finbinmin[ipt])
                self.lmult_yieldshisto[imult].SetBinContent(ipt + 1, rawYield)
                self.lmult_yieldshisto[imult].SetBinError(ipt + 1, rawYieldErr)
            fileout.cd()
            self.lmult_yieldshisto[imult].Write()
        fileout.Close()
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

        lfile = TFile.Open(fileout_name)
        for imult in range(self.p_nbin2):
            self.lmult_yieldshisto[imult].SetMinimum(1)
            self.lmult_yieldshisto[imult].SetMaximum(1e6)
            self.lmult_yieldshisto[imult].SetLineColor(imult+1)
            self.lmult_yieldshisto[imult].Draw("same")
            legyieldstring = "%.1f < %s < %.1f GeV/c" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legyield.AddEntry(self.lmult_yieldshisto[imult], legyieldstring, "LEP")
            self.lmult_yieldshisto[imult].GetXaxis().SetTitle("p_{T} (GeV)")
            self.lmult_yieldshisto[imult].GetYaxis().SetTitle("Uncorrected yields %s %s (1/GeV)" \
                    % (self.p_latexnmeson, self.typean))

        legyield.Draw()
        yields_save_name = self.make_file_path(self.d_resultsallpdata, "Yields", "eps", None,
                                               [self.case, self.typean])
        cYields.SaveAs(yields_save_name)
        lfile.Close()

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

        for imult in range(self.p_nbin2):
            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                h_invmass = lfile.Get("hmass" + suffix)

                # Get the nominal fit values to compare to
                mass_fitter_nominal.load(func_file.GetDirectory(suffix), True)
                yield_nominal = mass_fitter_nominal.yield_sig
                yield_err_nominal = mass_fitter_nominal.yield_sig_err
                bincount_nominal, bincount_err_nominal = \
                        mass_fitter_nominal.bincount(self.p_nsigma_signal)
                bincount_nominal = bincount_nominal
                bincount_err_nominal = bincount_err_nominal
                mean_nominal = mass_fitter_nominal.mean_fit
                sigma_nominal = mass_fitter_nominal.sigma_fit
                chisquare_ndf_nominal = mass_fitter_nominal.tot_fit_func.GetChisquare() / \
                        mass_fitter_nominal.tot_fit_func.GetNDF()

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
                                    chisquare_ndf_syst = \
                                    mass_fitter_syst.tot_fit_func.GetChisquare() / \
                                            mass_fitter_syst.tot_fit_func.GetNDF()
                                    # Only if the fit was successful and in case the chisquare does
                                    # exceed the nominal too much we extract the values from this
                                    # variation
                                    if success and \
                                            chisquare_ndf_syst < self.p_max_chisquare_ndf_syst:
                                        rawYield = mass_fitter_syst.yield_sig #/ \
                                        rawYieldErr = mass_fitter_syst.yield_sig_err #/ \
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

                # Keep all additional objects in a plot until it has been saved. Otherwise,
                # they will be deleted by Python as soon as something goes out of scope
                tmp_plot_objects = []

                # Used here internally for plotting
                def draw_histos(pad, x_axis_label, y_axis_label, draw_legend, nominals, hori_vert,
                                histos, plot_options, save_path):
                    colors = [kBlue, kGreen + 2]
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
                        h.SetFillStyle(3004)
                        h.SetStats(0)
                        h.SetLineColor(colors[i%len(colors)])
                        h.SetFillColor(colors[i%len(colors)])
                        h.SetMarkerColor(colors[i%len(colors)])
                        h.SetLineWidth(1)
                        h.GetXaxis().SetTitle(x_axis_label)
                        h.GetYaxis().SetTitle(y_axis_label)
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
                    pad.SaveAs(save_path)

                # Draw into canvas
                canvas = TCanvas("syst_canvas", "", 1400, 800)
                canvas.Divide(3, 2)
                pad = canvas.cd(5)
                filename = self.make_file_path(self.d_resultsallpdata, self.yields_syst_filename,
                                               "eps", None, suffix)
                draw_histos(pad, "yield", "# entries", True, [yield_nominal, bincount_nominal], "v",
                            [histo_yields, histo_bincounts], "hist", filename)
                pad = canvas.cd(4)
                filename = self.make_file_path(self.d_resultsallpdata, self.yields_syst_filename,
                                               "eps", None, ["err", suffix])
                draw_histos(pad, "yield_err", "# entries", True,
                            [yield_err_nominal, bincount_err_nominal], "v",
                            [histo_yields_err, histo_bincounts_err], "hist", filename)
                pad = canvas.cd(1)
                filename = self.make_file_path(self.d_resultsallpdata, "means_syst", "eps",
                                               None, suffix)
                draw_histos(pad, "trial", "#mu", False, [mean_nominal], None,
                            [histo_means], "p", filename)
                pad = canvas.cd(2)
                filename = self.make_file_path(self.d_resultsallpdata, "sigmas_syst", "eps",
                                               None, suffix)
                draw_histos(pad, "trial", "#sigma", False, [sigma_nominal], None,
                            [histo_sigmas], "p", filename)
                pad = canvas.cd(3)
                filename = self.make_file_path(self.d_resultsallpdata, "chisquares_syst", "eps",
                                               None, suffix)
                draw_histos(pad, "trial", "#chi^{2}/NDF", False, [chisquare_ndf_nominal], None,
                            [histo_chisquares], "p", filename)


                def create_text(pos_x, pos_y, text, color=kBlack):
                    root_text = TText(pos_x, pos_y, text)
                    root_text.SetTextSize(0.03)
                    root_text.SetTextColor(color)
                    root_text.SetNDC()
                    return root_text
                # Add some numbers
                pad = canvas.cd(6)

                root_texts = []
                fit_color = histo_yields.GetLineColor()
                bc_color = histo_bincounts.GetLineColor()
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
                                              fit_color))

                root_texts.append(create_text(0.05, 0.78,
                                              f"RMS = " \
                                              f"{rms_fit:.0f} ({unc_mean:.2f}%)", fit_color))

                root_texts.append(create_text(0.05, 0.73,
                                              f"MIN = {min_val:.0f}" \
                                              f"    " \
                                              f"MAX = {max_val:.0f}", fit_color))

                root_texts.append(create_text(0.05, 0.68,
                                              f"(MAX - MIN) / sqrt(12) = " \
                                              f"{diff_min_max:.0f} ({unc_min_max:.2f}%)",
                                              fit_color))

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
                                              f"{mean_bc:.0f}", bc_color))

                root_texts.append(create_text(0.05, 0.43,
                                              f"RMS = " \
                                              f"{rms_bc:.0f}", bc_color))

                root_texts.append(create_text(0.05, 0.38,
                                              f"MIN = {min_val:.0f}" \
                                              f"    " \
                                              f"MAX = {max_val:.0f}", bc_color))

                root_texts.append(create_text(0.05, 0.33,
                                              f"(MAX - MIN) / sqrt(12) = " \
                                              f"{diff_min_max:.0f} ({unc_min_max:.2f}%)",
                                              bc_color))

                root_texts.append(create_text(0.05, 0.23, "Deviations"))

                diff = yield_nominal - mean_fit
                diff_ratio = diff / yield_nominal * 100
                root_texts.append(create_text(0.05, 0.18,
                                              f"yield fit (nominal) - yield fit " \
                                              f"(multi) = {diff:.0f} " \
                                              f"({diff_ratio:.2f}%)", kRed + 2))

                diff = yield_nominal - mean_bc
                diff_ratio = diff / yield_nominal * 100
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
            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)

            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
            h_sel_pr.SetLineColor(imult+1)
            h_sel_pr.Draw("same")
            fileouteff.cd()
            h_sel_pr.SetName("eff_mult%d" % imult)
            h_sel_pr.Write()
            h_sel_fd.SetName("eff_fd_mult%d" % imult)
            h_sel_fd.Write()
            legeffstring = "%.1f < %s < %.1f GeV/c" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legeff.AddEntry(h_sel_pr, legeffstring, "LEP")
            h_sel_pr.GetXaxis().SetTitle("p_{T} (GeV)")
            h_sel_pr.GetYaxis().SetTitle("Uncorrected yields %s %s (1/GeV)" \
                    % (self.p_latexnmeson, self.typean))
            h_sel_pr.SetMinimum(0.)
            h_sel_pr.SetMaximum(1.5)
        legeff.Draw()
        cEff.SaveAs("%s/Eff%s%s.eps" % (self.d_resultsallpmc,
                                        self.case, self.typean))

    def feeddown(self):
        # TODO: Propagate uncertainties.
        self.loadstyle()
        file_resp = TFile.Open(self.n_fileff)
        file_eff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc, \
                              self.case, self.typean))
        file_out = TFile.Open("%s/feeddown%s%s.root" % \
                              (self.d_resultsallpmc, self.case, self.typean), "recreate")

        # Get feed-down detector response
        his_resp_fd = file_resp.Get("his_resp_jet_fd")
        arr_resp_fd = hist2array(his_resp_fd).T
        bins_final = np.array([his_resp_fd.GetYaxis().GetBinLowEdge(i) for i in \
            range(1, his_resp_fd.GetYaxis().GetNbins() + 2)])
        # TODO: Normalize so that projection on the pt_gen = 1.
        can_resp_fd = TCanvas("can_resp_fd", "Feed-down detector response", 800, 800)
        his_resp_fd.Draw("colz")
        can_resp_fd.SetLogz()
        can_resp_fd.SetLeftMargin(0.15)
        can_resp_fd.SetRightMargin(0.15)
        can_resp_fd.SaveAs("%s/ResponseFD%s%s.eps" % (self.d_resultsallpmc, \
                            self.case, self.typean))

        # Get simulated pt_cand vs. pt_jet of non-prompt jets.
        his_sim_fd = file_resp.Get("his_ptc_ptjet_fd")
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
            legeffstring = "%.1f < %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            leg_eff.SetHeader(legeffstring)
            leg_eff.AddEntry(his_eff_pr, "prompt", "LEP")
            leg_eff.AddEntry(his_eff_fd, "non-prompt", "LEP")
            leg_eff.Draw()
            can_eff.SaveAs("%s/Efficiency%s%s%s.eps" % (self.d_resultsallpmc, \
                            self.case, self.typean, imult))
            arr_eff_pr = hist2array(his_eff_pr)
            arr_eff_fd = hist2array(his_eff_fd)
            # Get the ratio of efficiencies.
            arr_eff_ratio = arr_eff_fd / arr_eff_pr
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
                #suffix = self.make_pre_suffix([self.v_var_binning,
                #                               f"{self.lpt_finbinmin[ipt]:.2f}",
                #                               f"{self.lpt_finbinmax[ipt]:.2f}",
                #                               f"{self.lpt_probcutfin[bin_id]:.2f}",
                #                               self.v_var2_binning,
                #                               f"{self.lvar2_binmin[imult]:.2f}",
                #                               f"{self.lvar2_binmax[imult]:.2f}"])
                hzvsmass = lfile.Get("hzvsmass" + suffix)
                load_dir = func_file.GetDirectory(suffix)
                mass_fitter = Fitter()
                mass_fitter.load(load_dir)
                sig_fit = mass_fitter.sig_fit_func #func_file.Get("sigfit" + suffix)
                mean = sig_fit.GetParameter(1)
                sigma = sig_fit.GetParameter(2)
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
                bkg_fit = func_file.Get("bkgrefit" + suffix)
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
            hcross.GetXaxis().SetTitle("p_{T} %s (GeV)" % self.p_latexnmeson)
            hcross.GetYaxis().SetTitle("d#sigma/dp_{T} (%s) %s" %
                                       (self.p_latexnmeson, self.typean))
            hcross.SetName("hcross%d" % imult)
            hcross.GetYaxis().SetRangeUser(1e1, 1e10)
            legvsvar1endstring = "%.1f < %s < %.1f GeV/c" % \
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
            legvsvar2endstring = "%.1f < %s < %.1f GeV/c" % \
                    (self.lpt_finbinmin[ipt], "p_{T}", self.lpt_finbinmax[ipt])
            hcrossvsvar2[ipt].Draw("same")
            legvsvar2.AddEntry(hcrossvsvar2[ipt], legvsvar2endstring, "LEP")
        legvsvar2.Draw()
        cCrossvsvar2.SaveAs("%s/Cross%s%sVs%s.eps" % (self.d_resultsallpdata,
                                                      self.case, self.typean, self.v_var2_binning))

    def studyevents(self):
        self.loadstyle()
        filedata = TFile.Open(self.f_evtvaldata)
        triggerlist = ["HighMultV0", "HighMultSPD"]
        varlist = ["v0m", "n_tracklets"]

        ctrigger = TCanvas('ctrigger', 'The Fit Canvas')
        ctrigger.SetCanvasSize(2100, 1000)
        ctrigger.Divide(2, 1)
        for i, _ in enumerate(triggerlist):
            ctrigger.cd(i+1)
            leg = TLegend(.5, .65, .7, .85)
            leg.SetBorderSize(0)
            leg.SetFillColor(0)
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.035)

            labeltriggerANDMB = "hclass%sANDINT7vs%s" % (triggerlist[i], varlist[i])
            labelMB = "hclassINT7vs%s" % varlist[i]
            heff = filedata.Get(labeltriggerANDMB)
            hden = filedata.Get(labelMB)
            heff.SetName(heff.GetName() + "_new")
            heff.SetLineColor(i+1)
            heff.Divide(heff, hden, 1.0, 1.0, "B")
            heff.SetMaximum(2.)
            heff.GetXaxis().SetTitle("offline %s" % varlist[i])
            heff.SetMinimum(0.)
            heff.GetYaxis().SetTitle("trigger efficiency")
            heff.Draw("epsame")
            leg.AddEntry(heff, triggerlist[i], "LEP")
            leg.Draw()
            print("INDEX", i)
        ctrigger.SaveAs(self.make_file_path(self.d_valevtdata, "ctrigger", "eps", \
                                        None, None))

        ccorrection = TCanvas('ccorrection', 'The Fit Canvas')
        ccorrection.SetCanvasSize(2100, 1000)
        ccorrection.Divide(2, 1)
        for i, _ in enumerate(triggerlist):
            ccorrection.cd(i+1)
            leg = TLegend(.5, .65, .7, .85)
            leg.SetBorderSize(0)
            leg.SetFillColor(0)
            leg.SetFillStyle(0)
            leg.SetTextFont(42)
            leg.SetTextSize(0.035)

            labeltrigger = "hclass%svs%s" % (triggerlist[i], varlist[i])
            labelMB = "hclassINT7vs%s" % varlist[i]
            hratio = filedata.Get(labeltrigger)
            hden = filedata.Get(labelMB)
            hratio.SetLineColor(i+1)
            hratio.Divide(hratio, hden, 1.0, 1.0, "B")
            hratio.GetXaxis().SetTitle("offline %s" % varlist[i])
            hratio.SetMinimum(0.)
            hratio.GetYaxis().SetTitle("ratio %s/MB" % triggerlist[i])
            hratio.Draw("epsame")
            leg.AddEntry(hratio, triggerlist[i], "LEP")
            leg.Draw()
        ccorrection.SaveAs(self.make_file_path(self.d_valevtdata, "ccorrection", "eps", \
                                        None, None))
