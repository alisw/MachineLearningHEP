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
from machine_learning_hep.root import save_root_object
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.analysis.analyzer import Analyzer
# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
class AnalyzerDhadrons_mult(Analyzer): # pylint: disable=invalid-name
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)
        self.logger = get_logger()
        self.logger.warning("TEST")
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
        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        self.p_nbx2 = datap["analysis"][self.typean].get("isNbx2", "")
        if "isNbx2" not in datap["analysis"][self.typean]:
            self.p_nbx2 = False
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
        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]
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
        self.p_performval = datap["analysis"].get("event_cand_validation", None)

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
            legeffstring = "%.1f #leq %s < %.1f" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legeff.AddEntry(h_sel_pr, legeffstring, "LEP")
            h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s %s (1/GeV)" \
                    % (self.p_latexnhadron, self.typean))
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
            legeffFDstring = "%.1f #leq %s < %.1f" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legeffFD.AddEntry(h_sel_fd, legeffFDstring, "LEP")
            h_sel_fd.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            h_sel_fd.GetYaxis().SetTitle("Acc x efficiency feed-down %s %s (1/GeV)" \
                    % (self.p_latexnhadron, self.typean))
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
            hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnhadron)
            hcross.GetYaxis().SetTitle("d#sigma/d#it{p}_{T} (%s) %s" %
                                       (self.p_latexnhadron, self.typean))
            hcross.SetName("hcross%d" % imult)
            hcross.GetYaxis().SetRangeUser(1e1, 1e10)
            legvsvar1endstring = "%.1f < %s < %.1f" % \
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
                hcrossvsvar2[ipt].GetYaxis().SetTitle(self.p_latexnhadron)
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
    def calculate_norm(hsel, hnovt, hvtxout, multmin, multmax):
        if not hsel:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hsel")
        if not hnovt:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hnovt")
        if not hvtxout:
            # pylint: disable=undefined-variable
            self.logger.error("Missing hvtxout")

        binminv = hsel.GetXaxis().FindBin(multmin)
        binmaxv = hsel.GetXaxis().FindBin(multmax)

        n_sel = hsel.Integral(binminv, binmaxv)
        n_novtx = hnovt.Integral(binminv, binmaxv)
        n_vtxout = hvtxout.Integral(binminv, binmaxv)
        norm = -1
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
        from ROOT import HFPtSpectrum, HFPtSpectrum2, HFPtSpectrumRescaled
        histonorm = TH1F("histonorm", "histonorm", self.p_nbin2, 0, self.p_nbin2)
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
            if ((self.p_nbx2) and imult == 0):
                fileoutcrossmultRescaled = "%s/finalcross%s%smult%d_Rescaled.root" % \
                     (self.d_resultsallpdata, self.case, self.typean, imult)
            norm = -1
            filemass = TFile.Open(self.n_filemass)
            labeltrigger = "hbit%svs%s" % (self.triggerbit, self.v_var2_binning_gen)
            if self.apply_weights is True:
                labeltrigger = labeltrigger + "_weight"
            hsel = filemass.Get("sel_%s" % labeltrigger)
            hnovtx = filemass.Get("novtx_%s" % labeltrigger)
            hvtxout = filemass.Get("vtxout_%s" % labeltrigger)
            norm = self.calculate_norm(hsel, hnovtx, hvtxout,
                                       self.lvar2_binmin[imult],
                                       self.lvar2_binmax[imult])
            histonorm.SetBinContent(imult + 1, norm)
            # pylint: disable=logging-not-lazy
            self.logger.warning("Number of events %d for mult bin %d" % (norm, imult))
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
                if self.p_nbx2:
                    HFPtSpectrumRescaled(self.p_indexhpt, self.p_inputfonllpred, \
                    fileouteff, namehistoeffprompt, namehistoefffeed, yield_filename, nameyield, \
                    fileoutcrossmultRescaled, norm, self.p_sigmav0 * 1e12, \
                    self.p_fd_method, self.p_cctype)
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
        histonorm.Write()
        fileoutcrosstot.Close()

        if self.p_nbx2:
            gROOT.LoadMacro("CombineFeedDownMCSubtractionMethodsUncertainties.C")
            from ROOT import CombineFeedDownMCSubtractionMethodsUncertainties
            fileoutcrossmult0 = "%s/finalcross%s%smult0.root" % \
                (self.d_resultsallpdata, self.case, self.typean)
            fileoutFeeddownTot = "%s/FeeddownSyst_NbNbx2_%s%s.root" % \
                (self.d_resultsallpdata, self.case, self.typean)
            CombineFeedDownMCSubtractionMethodsUncertainties(fileoutcrossmultRescaled, \
                fileoutcrossmult0, fileoutFeeddownTot, self.p_inputfonllpred, 6)


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
            hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnhadron)
            hcross.GetYaxis().SetTitleOffset(1.3)
            hcross.GetYaxis().SetTitle("Corrected yield/events (%s) %s" %
                                       (self.p_latexnhadron, self.typean))
            hcross.GetYaxis().SetRangeUser(1e-10, 1)
            legvsvar1endstring = "%.1f #leq %s < %.1f" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legvsvar1.AddEntry(hcross, legvsvar1endstring, "LEP")
            hcross.Draw("same")
        legvsvar1.Draw()
        cCrossvsvar1.SaveAs("%s/CorrectedYieldsNorm%s%sVs%s.eps" % (self.d_resultsallpdata,
                                                                    self.case, self.typean,
                                                                    self.v_var_binning))
    def plottervalidation(self):
        if self.p_performval is False:
            self.logger.fatal(
                "The validation step was set to false. You dont \
                                have produced the histograms you need for the \
                                validation stage. Please rerun the histomass \
                                step"
            )
        self.logger.info("I AM RUNNING THE PLOTTER VALIDATION STEP")
        # You can find all the input files in the self.n_filemass. At the
        # moment we dont do tests for the MC file that would be in any case
        # self.n_filemass_mc. This function will be run on only the single
        # merged LHC16,LHC17, LHC18 file or also on the separate years
        # depending on how you set the option doperperiod in the
        # default_complete.yml database.
        def do_validation_plots(input_file_name,
                                output_path,
                                ismc=False,
                                pileup_fraction=True,
                                tpc_tof_me=True):
            gROOT.SetBatch(True)

            input_file = TFile(input_file_name, "READ")
            if not input_file or not input_file.IsOpen():
                self.logger.fatal("Did not find file %s", input_file.GetName())

            def get_histo(namex, namey=None, tag=""):
                """
                Gets a histogram from a file
                """
                h_name = f"hVal_{namex}"
                if namey:
                    h_name += f"_vs_{namey}"
                h_name += tag
                h = input_file.Get(h_name)
                if not h:
                    input_file.ls()
                    self.logger.fatal(
                        "Did not find %s in file %s", h_name, input_file.GetName()
                    )
                return h

            def do_plot(histo):
                """
                Plots the histogram in a new canvas, if it is a TH2, it also plots the profile.
                The canvas has the same name as the histogram and it is saved to the output_path
                """
                canvas = TCanvas(histo.GetName(), histo.GetName())
                profile = None
                histo.Draw("COLZ")
                if "TH2" in histo.ClassName():
                    if "nsig" in histo.GetYaxis().GetTitle():
                        histo.GetYaxis().SetRangeUser(-100, 100)
                    profile = histo.ProfileX(histo.GetName() + "_profile")
                    profile.SetLineWidth(2)
                    profile.SetLineColor(2)
                    profile.Draw("same")
                gPad.SetLogz()
                gPad.Update()
                save_root_object(canvas, path=output_path)

            # Plot all validation histogram
            for i in range(0, input_file.GetListOfKeys().GetEntries()):
                key_name = input_file.GetListOfKeys().At(i).GetName()
                if not key_name.startswith("hVal_"):
                    continue
                do_plot(input_file.Get(key_name))

            # Fraction of pileup events
            if pileup_fraction:
                hnum = get_histo("n_tracklets_corr", tag="pileup")
                hnum.SetName(hnum.GetName() + "_eventfraction")
                hden = get_histo("n_tracklets_corr")
                hnum.Divide(hnum, hden)
                hnum.GetYaxis().SetTitle("Fraction of events")
                do_plot(hnum)

            def plot_validation_candidate(tag):
                # Compute TPC-TOF matching efficiency
                if tpc_tof_me:
                    for i in "Pi K".split():
                        for j in "0 1".split():
                            for k in "p pt".split():
                                hname = [f"{k}_prong{j}",
                                         f"nsigTOF_{i}_{j}", tag]
                                hnum = get_histo(*hname)
                                hnum = hnum.ProjectionX(
                                    hnum.GetName() + "_num", 2, -1)
                                hden = get_histo(*hname)
                                hden = hden.ProjectionX(
                                    hden.GetName() + "_den")
                                hnum.Divide(hnum, hden, 1, 1, "B")
                                hnum.SetName(
                                    hnum.GetName().replace(
                                        "_num", "_TPC-TOF_MatchingEfficiency"
                                    )
                                )
                                hnum.GetYaxis().SetTitle("TPC-TOF_MatchingEfficiency")
                                do_plot(hnum)

            plot_validation_candidate(tag="")
            # Part dedicated to MC Checks
            if not ismc:
                input_file.Close()
                return

            plot_validation_candidate(tag="MC")
            input_file.Close()

        do_validation_plots(self.n_filemass, self.d_resultsallpdata)
        do_validation_plots(self.n_filemass_mc,
                            self.d_resultsallpmc, ismc=True)
