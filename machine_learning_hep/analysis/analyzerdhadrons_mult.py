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
from array import array
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
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.analysis.analyzer import Analyzer
# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
class AnalyzerDhadrons_mult(Analyzer): # pylint: disable=invalid-name
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        #namefiles pkl
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

        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

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
        self.fits_dirname = "fits"
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

        self.d_valevtdata = datap["validation"]["data"]["dir"][period] \
                if period is not None else datap["validation"]["data"]["dirmerged"]
        self.d_valevtmc = datap["validation"]["mc"]["dir"][period] \
                if period is not None else datap["validation"]["mc"]["dirmerged"]

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
        self.p_inputfonllpred = datap["analysis"]["inputfonllpred"]
        self.p_triggereff = datap["analysis"][self.typean].get("triggereff", [1] * 10)
        self.p_triggereffunc = datap["analysis"][self.typean].get("triggereffunc", [0] * 10)

        self.apply_weights = datap["analysis"][self.typean]["triggersel"]["weighttrig"]
        self.root_objects = []

        self.get_crossmb_from_path = datap["analysis"][self.typean].get("get_crossmb_from_path", \
                                                                        None)
        self.path_for_crossmb = datap["analysis"][self.typean].get("path_for_crossmb", None)

        # Fitting
        self.fitter = None


    # pylint: disable=import-outside-toplevel
    def fit(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        self.fitter = MLFitter(self.datap, self.typean, self.n_filemass, self.n_filemass_mc)
        self.fitter.perform_pre_fits()
        self.fitter.perform_central_fits()
        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")
        self.fitter.draw_fits(self.d_resultsallpdata, fileout)
        fileout.Close()
        fileout_name = os.path.join(self.d_resultsallpdata,
                                    f"{self.fits_dirname}_{self.case}_{self.typean}")
        self.fitter.save_fits(fileout_name)
        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)


    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-branches
    # pylint: disable=import-outside-toplevel
    def yield_syst(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        if not self.fitter:
            fileout_name = os.path.join(self.d_resultsallpdata,
                                        f"{self.fits_dirname}_{self.case}_{self.typean}")
            self.fitter = MLFitter(self.datap, self.typean, self.n_filemass, self.n_filemass_mc)
            if not self.fitter.load_fits(fileout_name):
                self.logger.error("Cannot load fits from dir %s", fileout_name)
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

    # pylint: disable=import-outside-toplevel
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
                HFPtSpectrum(self.p_indexhpt, self.p_inputfonllpred, \
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
