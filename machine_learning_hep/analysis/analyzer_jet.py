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
import numpy as np # pylint: disable=unused-import
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TF1, TH1D, TLatex, TGraphAsymmErrors
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed, kOrange
from ROOT import gInterpreter, gPad
from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes
# HF specific imports
from machine_learning_hep.fitting.helpers import MLFitter
from machine_learning_hep.logger import get_logger
from machine_learning_hep.io import dump_yaml_from_dict
from machine_learning_hep.utilities import folding, get_bins, make_latex_table, parallelizer
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.utilities import setup_histogram, setup_pad
from machine_learning_hep.utilities import setup_legend, setup_tgraph, draw_latex, tg_sys

# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
class AnalyzerJet(Analyzer):
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]

        self.p_sgnfunc = datap["analysis"][self.typean]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"][self.typean]["bkgfunc"]
        self.p_masspeak = datap["analysis"][self.typean]["masspeak"]
        self.p_massmin = datap["analysis"][self.typean]["massmin"]
        self.p_massmax = datap["analysis"][self.typean]["massmax"]
        self.p_rebin = datap["analysis"][self.typean]["rebin"]
        self.p_fix_mean = datap["analysis"][self.typean]["fix_mean"]
        self.p_fix_sigma = datap["analysis"][self.typean]["fix_sigma"]

        self.p_sigmaarray = datap["analysis"][self.typean]["sigmaarray"]
        self.p_masspeaksec = None
        self.p_fix_sigmasec = None
        self.p_sigmaarraysec = None
        if self.p_sgnfunc[0] == 1:
            self.p_masspeaksec = datap["analysis"][self.typean]["masspeaksec"]
            self.p_fix_sigmasec = datap["analysis"][self.typean]["fix_sigmasec"]
            self.p_sigmaarraysec = datap["analysis"][self.typean]["sigmaarraysec"]

        self.fitter = None

        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]
        self.lvar2_binmin_reco = datap["analysis"][self.typean].get("sel_binmin2_reco", None)
        self.lvar2_binmax_reco = datap["analysis"][self.typean].get("sel_binmax2_reco", None)

        self.lvar2_binmin_gen = datap["analysis"][self.typean].get("sel_binmin2_gen", None)
        self.lvar2_binmax_gen = datap["analysis"][self.typean].get("sel_binmax2_gen", None)

        self.lvarshape_binmin_reco = \
            datap["analysis"][self.typean].get("sel_binminshape_reco", None)
        self.lvarshape_binmax_reco = \
            datap["analysis"][self.typean].get("sel_binmaxshape_reco", None)

        self.lvarshape_binmin_gen = \
            datap["analysis"][self.typean].get("sel_binminshape_gen", None)
        self.lvarshape_binmax_gen = \
            datap["analysis"][self.typean].get("sel_binmaxshape_gen", None)

        self.niter_unfolding = \
            datap["analysis"][self.typean].get("niterunfolding", None)
        self.choice_iter_unfolding = \
            datap["analysis"][self.typean].get("niterunfoldingchosen", None)
        self.niterunfoldingregup = \
            datap["analysis"][self.typean].get("niterunfoldingregup", None)
        self.niterunfoldingregdown = \
            datap["analysis"][self.typean].get("niterunfoldingregdown", None)

        self.signal_sigma = \
            datap["analysis"][self.typean].get("signal_sigma", None)
        self.sideband_sigma_1_left = \
            datap["analysis"][self.typean].get("sideband_sigma_1_left", None)
        self.sideband_sigma_1_right = \
            datap["analysis"][self.typean].get("sideband_sigma_1_right", None)
        self.sideband_sigma_2_left = \
            datap["analysis"][self.typean].get("sideband_sigma_2_left", None)
        self.sideband_sigma_2_right = \
            datap["analysis"][self.typean].get("sideband_sigma_2_right", None)
        self.sigma_scale = \
            datap["analysis"][self.typean].get("sigma_scale", None)
        self.sidebandleftonly = \
            datap["analysis"][self.typean].get("sidebandleftonly", None)

        self.powheg_path_prompt = \
            datap["analysis"][self.typean].get("powheg_path_prompt", None)
        self.powheg_path_nonprompt = \
            datap["analysis"][self.typean].get("powheg_path_nonprompt", None)

        self.powheg_prompt_variations = \
            datap["analysis"][self.typean].get("powheg_prompt_variations", None)
        self.powheg_prompt_variations_path = \
            datap["analysis"][self.typean].get("powheg_prompt_variations_path", None)

        self.powheg_nonprompt_variations = \
            datap["analysis"][self.typean].get("powheg_nonprompt_variations", None)
        self.powheg_nonprompt_variations_path = \
            datap["analysis"][self.typean].get("powheg_nonprompt_variations_path", None)

        self.pythia8_prompt_variations = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations", None)
        self.pythia8_prompt_variations_path = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations_path", None)
        self.pythia8_prompt_variations_legend = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations_legend", None)

        self.systematic_catagories = \
            datap["analysis"][self.typean].get("systematic_catagories", None)
        self.systematic_variations = \
            datap["analysis"][self.typean].get("systematic_variations", None)
        self.systematic_correlation = \
            datap["analysis"][self.typean].get("systematic_correlation", None)
        self.systematic_rms = \
            datap["analysis"][self.typean].get("systematic_rms", None)
        self.systematic_symmetrise = \
            datap["analysis"][self.typean].get("systematic_symmetrise", None)
        self.systematic_rms_both_sides = \
            datap["analysis"][self.typean].get("systematic_rms_both_sides", None)

        self.branching_ratio = \
            datap["analysis"][self.typean].get("branching_ratio", None)
        self.xsection_inel = \
            datap["analysis"][self.typean].get("xsection_inel", None)


        self.p_nbin2_reco = len(self.lvar2_binmin_reco)
        self.p_nbin2_gen = len(self.lvar2_binmin_gen)
        self.p_nbinshape_reco = len(self.lvarshape_binmin_reco)
        self.p_nbinshape_gen = len(self.lvarshape_binmin_gen)


        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc, n_filemass_name)

        # Output directories and filenames
        self.yields_filename = "yields"
        self.fits_dirname = "fits"
        self.yields_syst_filename = "yields_syst"
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)


        self.p_latexnmeson = datap["analysis"][self.typean]["latexnamemeson"]
        self.p_latexbin2var = datap["analysis"][self.typean]["latexbin2var"]
        self.var2ranges_reco = self.lvar2_binmin_reco.copy()
        self.var2ranges_reco.append(self.lvar2_binmax_reco[-1])
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        self.varshaperanges_reco = self.lvarshape_binmin_reco.copy()
        self.varshaperanges_reco.append(self.lvarshape_binmax_reco[-1])
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])
        self.p_nevents = datap["analysis"][self.typean]["nevents"]

        # Fitting
        self.fitter = None

    # pylint: disable=import-outside-toplevel
    def fit(self):
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")
        myfilemc = TFile(self.n_filemass_mc, "read")
        myfile = TFile(self.n_filemass, "read")
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            for ibin2 in range(self.p_nbin2_reco):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                          self.lvar2_binmax_reco[ibin2])
                histomassmc = myfilemc.Get("hmass_sig" + suffix)
                histomassmc_reb = AliVertexingHFUtils.RebinHisto(histomassmc, \
                                            self.p_rebin[ipt], -1)
                histomassmc_reb_f = TH1F()
                histomassmc_reb.Copy(histomassmc_reb_f)
                fittermc = AliHFInvMassFitter(histomassmc_reb_f, \
                    self.p_massmin[ipt], self.p_massmax[ipt], self.p_bkgfunc[ipt], 0)
                fittermc.SetInitialGaussianMean(self.p_masspeak)
                out = fittermc.MassFitter(1)
                print("I have made MC fit for sigma initialization, status: %d" % out)
                histomass = myfile.Get("hmass" + suffix)
                histomass_reb = AliVertexingHFUtils.RebinHisto(histomass, \
                                            self.p_rebin[ipt], -1)
                histomass_reb_f = TH1F()
                histomass_reb.Copy(histomass_reb_f)
                fitter = AliHFInvMassFitter(histomass_reb_f, self.p_massmin[ipt], \
                    self.p_massmax[ipt], self.p_bkgfunc[ipt], self.p_sgnfunc[ipt])
                fitter.SetInitialGaussianSigma(fittermc.GetSigma())
                fitter.SetInitialGaussianMean(fittermc.GetMean())
                if self.p_fix_sigma[ipt] is True:
                    fitter.SetFixGaussianSigma(fittermc.GetSigma())
                if self.p_sgnfunc[ipt] == 1:
                    if self.p_fix_sigmasec[ipt] is True:
                        fitter.SetFixSecondGaussianSigma(self.p_sigmaarraysec[ipt])
                out = fitter.MassFitter(1)
                fit_dir = fileout.mkdir(suffix)
                fit_dir.WriteObject(fitter, "fitter%d" % (ipt))
                c_fitted_result = TCanvas('c_fitted_result ' + suffix, 'Fitted Result')
                p_fitted_result = TPad('p_fitted_result' + suffix,
                                       'p_fitted_result' + suffix, 0.0, 0.001, 1.0, 1.0)
                bkg_func = fitter.GetBackgroundRecalcFunc()
                sgn_func = fitter.GetMassFunc()
                setup_pad(p_fitted_result)
                c_fitted_result.SetCanvasSize(1900, 1500)
                c_fitted_result.SetWindowSize(500, 500)
                setup_histogram(histomass_reb)
                histomass_reb.SetXTitle("mass")
                histomass_reb.SetYTitle("counts")
                histomass_reb.Draw("same")
                if out == 1:
                    bkg_func.SetLineColor(kGreen)
                    sgn_func.SetLineColor(kRed)
                    sgn_func.Draw("same")
                    bkg_func.Draw("same")
                latex = TLatex(0.2, 0.85, '%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}' \
                    % (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.2, 0.8, '%.2f < #it{p}_{T, Lc} < %.2f GeV/#it{c}' % \
                    (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt]))
                draw_latex(latex2)
                c_fitted_result.SaveAs("%s/fitted_result_%s.eps" % \
                    (self.d_resultsallpdata, suffix))
        myfilemc.Close()
        myfile.Close()
        fileout.Close()
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

        for imult in range(self.p_nbin2_reco):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                            self.lvar2_binmin_reco[imult], \
                                            self.lvar2_binmax_reco[imult])
            h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
            h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
            h_sel_pr.SetLineColor(imult+1)
            h_sel_pr.Draw("same")
            fileouteff.cd()
            h_sel_pr.SetName("eff_mult%d" % imult)
            h_sel_pr.Write()
            legeffstring = "%.1f #leq %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[imult], self.p_latexbin2var,
                     self.lvar2_binmax_reco[imult])
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

        for imult in range(self.p_nbin2_reco):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                            self.lvar2_binmin_gen[imult], \
                                            self.lvar2_binmax_gen[imult])
            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
            h_sel_fd.SetLineColor(imult+1)
            h_sel_fd.Draw("same")
            fileouteff.cd()
            h_sel_fd.SetName("eff_fd_mult%d" % imult)
            h_sel_fd.Write()
            legeffFDstring = "%.1f #leq %s < %.1f GeV/#it{c}" % \
                    (self.lvar2_binmin_gen[imult], self.p_latexbin2var,
                     self.lvar2_binmax_gen[imult])
            legeffFD.AddEntry(h_sel_fd, legeffFDstring, "LEP")
            h_sel_fd.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            h_sel_fd.GetYaxis().SetTitle("Acc x efficiency feed-down %s %s (1/GeV)" \
                    % (self.p_latexnmeson, self.typean))
            h_sel_fd.SetMinimum(0.)
            h_sel_fd.SetMaximum(1.5)
        legeffFD.Draw()
        cEffFD.SaveAs("%s/EffFD%s%s.eps" % (self.d_resultsallpmc,
                                            self.case, self.typean))
