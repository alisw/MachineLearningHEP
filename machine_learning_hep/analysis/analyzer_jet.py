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

    # pylint: disable=too-many-locals
    def side_band_sub(self):
        #This function perform side band subtraction of the histograms.
        #The input files for this function are coming from:
        #    - root file containing the histograms of mass vs z called here
        #     "hzvsmass". There is one for each bin of HF pt and jet pt.
        #    - fit function performed in the fit function above fit() called in
        #    this function "func_file"
        #    - several histograms coming from the efficiency ROOT file

        self.loadstyle()
        lfile = TFile.Open(self.n_filemass)

        func_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                            None, [self.case, self.typean])
        func_file = TFile.Open(func_filename, "READ")
        eff_file = TFile.Open("%s/efficiencies%s%s.root" % \
                              (self.d_resultsallpmc, self.case, self.typean))
        fileouts = TFile.Open("%s/sideband_sub%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "RECREATE")
        fileouts.cd()

        # These are the reconstructed level bins for jet pt and z values

        zbin_reco = []
        nzbin_reco = self.p_nbinshape_reco
        zbin_reco = self.varshaperanges_reco
        zbinarray_reco = array('d', zbin_reco)

        jetptbin_reco = []
        njetptbin_reco = self.p_nbin2_reco
        jetptbin_reco = self.var2ranges_reco
        jetptbinarray_reco = array('d', jetptbin_reco)

        # hzvsjetpt is going to be the side-band subtracted histogram of z vs
        # jet that is going to be filled after subtraction

        hzvsjetpt = TH2F("hzvsjetpt", "", nzbin_reco, zbinarray_reco,
                         njetptbin_reco, jetptbinarray_reco)
        hzvsjetpt.Sumw2()

        # This is a loop over jet pt and over HF candidate pT

        for imult in range(self.p_nbin2_reco):
            heff = eff_file.Get("eff_mult%d" % imult)
            hz = None
            first_fit = 0
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[imult],
                          self.lvar2_binmax_reco[imult])

                # In this part of the code we extract for each bin of jet pt
                # and HF pT the fit function of the data fit to extract mean and
                # sigma. IF THERE IS NO GOOD FIT THE GIVEN BIN IS DISCARDED AND
                # WILL NOT ENTER THE FINAL RESULT

                load_dir = func_file.GetDirectory(suffix)
                mass_fitter = load_dir.Get("fitter%d" % (ipt))
                mean = mass_fitter.GetMean()
                sigma = mass_fitter.GetSigma()
                bkg_fit = mass_fitter.GetBackgroundRecalcFunc()

                # Here I define the boundaries for the side-band subtractions
                # based on the results of the fit. We get usually 4-9 sigma from
                # the mean in both sides to extract the side band distributions

                hzvsmass = lfile.Get("hzvsmass" + suffix)
                binmasslow2sig = \
                    hzvsmass.GetXaxis().FindBin(mean - self.signal_sigma*sigma)
                masslow2sig = mean - self.signal_sigma*sigma
                binmasshigh2sig = \
                    hzvsmass.GetXaxis().FindBin(mean + self.signal_sigma*sigma)
                masshigh2sig = mean + self.signal_sigma*sigma
                binmasslow4sig = \
                    hzvsmass.GetXaxis().FindBin(mean - self.sideband_sigma_1_left*sigma)
                masslow4sig = \
                    mean - self.sideband_sigma_1_left*sigma
                binmasshigh4sig = \
                    hzvsmass.GetXaxis().FindBin(mean + self.sideband_sigma_1_right*sigma)
                masshigh4sig = \
                    mean + self.sideband_sigma_1_right*sigma
                binmasslow9sig = \
                    hzvsmass.GetXaxis().FindBin(mean - self.sideband_sigma_2_left*sigma)
                masslow9sig = \
                    mean - self.sideband_sigma_2_left*sigma
                binmasshigh9sig = \
                    hzvsmass.GetXaxis().FindBin(mean + self.sideband_sigma_2_right*sigma)
                masshigh9sig = \
                    mean + self.sideband_sigma_2_right*sigma

                # here we project over the z-axis the 2d distributions in the
                # three regions = signal region, left and right side-band

                hzsig = hzvsmass.ProjectionY("hzsig" + suffix, \
                             binmasslow2sig, binmasshigh2sig, "e")
                hzbkgleft = hzvsmass.ProjectionY("hzbkgleft" + suffix, \
                             binmasslow9sig, binmasslow4sig, "e")
                hzbkgright = hzvsmass.ProjectionY("hzbkgright" + suffix, \
                             binmasshigh4sig, binmasshigh9sig, "e")

                # the background histogram is made by adding the left and
                # right side band in general. self.sidebandleftonly = True is
                # just made for systematic studies

                # Below a list of histrograms are defined:
                #    - hzsig is as discussed before the distribution of z in
                #      the signal region not background subtracted
                #    - hzsub is the z-distribution after background subtraction
                #      using sidebands, efficiency corrected.
                #    - hzsub_noteffscaled is the z-distribution after background
                #      subtraction not corrected for efficiency
                #    - hzbkg_scaled is the bkg distribution scaled for the
                #      factor used to perform the background subtraction

                hzbkg = hzbkgleft.Clone("hzbkg" + suffix)
                if self.sidebandleftonly is False:
                    hzbkg.Add(hzbkgright)
                hzbkg_scaled = hzbkg.Clone("hzbkg_scaled" + suffix)

                area_scale_denominator = -1
                if not bkg_fit: # if there is no background fit it continues
                    continue
                area_scale_denominator = bkg_fit.Integral(masslow9sig, masslow4sig) + \
                bkg_fit.Integral(masshigh4sig, masshigh9sig)
                if area_scale_denominator == 0:
                    continue
                area_scale = \
                    bkg_fit.Integral(masslow2sig, masshigh2sig)/area_scale_denominator # 0.4
                hzsub = hzsig.Clone("hzsub" + suffix)
                hzsub.Add(hzbkg, -1*area_scale)
                hzsub_noteffscaled = hzsub.Clone("hzsub_noteffscaled" + suffix)
                hzbkg_scaled.Scale(area_scale)
                eff = heff.GetBinContent(ipt+1)
                if eff > 0.0:
                    hzsub.Scale(1.0/(eff*self.sigma_scale))
                if first_fit == 0:
                    hz = hzsub.Clone("hz")
                    first_fit = 1
                else:
                    hz.Add(hzsub)
                fileouts.cd()
                hzsig.Write("hzsig" + suffix)
                hzbkgleft.Write("hzbkgleft" + suffix)
                hzbkgright.Write("hzbkgright" + suffix)
                hzbkg.Write("hzbkg" + suffix)
                hzsub.Write("hzsub" + suffix)

                # This canvas will contain the distributions of the side band
                # subtracted z-distributions in bin of the reco jet pt
                # variable, corrected for HF candidate efficiency

                csubz = TCanvas('csubz' + suffix, 'The Side-Band Sub Canvas' + suffix)
                psubz = TPad('psubz', 'psubz', 0.0, 0.001, 1.0, 1.0)
                setup_pad(psubz)
                csubz.SetCanvasSize(1900, 1500)
                csubz.SetWindowSize(500, 500)
                setup_histogram(hzsub, 4)
                hzsub.GetYaxis().SetRangeUser(hzsub.GetMinimum(), hzsub.GetMaximum()*1.2)
                hzsub.SetXTitle("#it{z}_{#parallel}^{ch}")
                hzsub.Draw()
                latex = TLatex(0.6, 0.85, "%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}" % \
                               (self.lvar2_binmin_reco[imult], self.lvar2_binmax_reco[imult]))
                draw_latex(latex)
                latex2 = TLatex(0.6, 0.8,
                                "%.2f < #it{p}_{T, #Lambda_{c}^{#plus}} < %.2f GeV/#it{c}" \
                                % (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt]))
                draw_latex(latex2)
                csubz.SaveAs("%s/step1_side_band_subtracted_effcorrected_%s%s_%s.eps" % \
                             (self.d_resultsallpdata, self.case, self.typean, suffix))

                # csigbkgsubz
                # This canvas contains the hzsig distributions of z in the signal
                # region (signal+bkg), the hzbkg_scaled distribution of
                # background rescaled, hzsub_noteffscaled the signal subtracted
                # distribution without efficiency corrections.

                csigbkgsubz = TCanvas('csigbkgsubz' + suffix, 'The Side-Band Canvas' + suffix)
                psigbkgsubz = TPad('psigbkgsubz' + suffix, 'psigbkgsubz' + suffix,
                                   0.0, 0.001, 1.0, 1.0)
                setup_pad(psigbkgsubz)
                csigbkgsubz.SetCanvasSize(1900, 1500)
                csigbkgsubz.SetWindowSize(500, 500)
                legsigbkgsubz = TLegend(.18, .70, .35, .85)
                setup_legend(legsigbkgsubz)
                setup_histogram(hzsig, 2)
                legsigbkgsubz.AddEntry(hzsig, "signal region", "LEP")
                hz_min = min(hzsig.GetMinimum(0.1), hzbkg_scaled.GetMinimum(0.1),
                             hzsub_noteffscaled.GetMinimum(0.1))
                hz_max = max(hzsig.GetMaximum(), hzbkg_scaled.GetMaximum(),
                             hzsub_noteffscaled.GetMaximum())
                hz_ratio = hz_max / hz_min
                hz_margin_max = 0.5
                hz_margin_min = 0.1
                hzsig.GetYaxis().SetRangeUser(hz_min / (1. if hz_ratio == 0 \
                    else pow(hz_ratio, hz_margin_min)), hz_max * pow(hz_ratio, hz_margin_max))
                hzsig.GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0] + 0.01, \
                                              self.lvarshape_binmax_reco[-1] - 0.001)
                hzsig.SetXTitle("#it{z}_{#parallel}^{ch}")
                hzsig.SetYTitle("Yield")
                hzsig.SetTitle("")
                hzsig.GetYaxis().SetTitleOffset(1.4)
                hzsig.GetYaxis().SetMaxDigits(3)
                hzsig.Draw()
                setup_histogram(hzbkg_scaled, 3, 24)
                legsigbkgsubz.AddEntry(hzbkg_scaled, "side-band region", "LEP")
                hzbkg_scaled.Draw("same")
                setup_histogram(hzsub_noteffscaled, 4, 28)
                legsigbkgsubz.AddEntry(hzsub_noteffscaled, "subtracted", "LEP")
                hzsub_noteffscaled.Draw("same")
                legsigbkgsubz.Draw("same")
                latex = TLatex(0.42, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
                draw_latex(latex)
                latex1 = TLatex(0.42, 0.8, "charged jets, anti-#it{k}_{T}, \
                                #it{R} = 0.4, #left|#it{#eta}_{jet}#right| < 0.5")
                draw_latex(latex1)
                latex2 = TLatex(0.42, 0.75, "%.0f < #it{p}_{T, jet}^{ch} < %.0f GeV/#it{c}" \
                                % (self.lvar2_binmin_reco[imult], self.lvar2_binmax_reco[imult]))
                draw_latex(latex2)
                latex3 = TLatex(0.42, 0.7, "with #Lambda_{c}^{#plus} (& cc), %.0f < \
                                #it{p}_{T, #Lambda_{c}^{#plus}} < %.0f GeV/#it{c}"
                                % (self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt]))
                draw_latex(latex3)
                if hz_ratio != 0:
                    psigbkgsubz.SetLogy()
                csigbkgsubz.SaveAs("%s/step1_side_band_sigbkg%s%s_%s.eps" % \
                             (self.d_resultsallpdata, self.case, self.typean, suffix))

            suffix = "_%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_reco[imult],
                          self.lvar2_binmax_reco[imult])
            if first_fit == 0:
                print("No successful fits for: %s" % suffix)
                continue


            # We are now outside of the loop of HF candidate pt. We are going now
            # to plot the "hz" histogram, which contains the Add of all the
            # bkg-subtracted efficiency corrected distributions of all the HF
            # candidate pt bins put together. Each "hz" distribution made for each
            # jet pt is normalized by its own area. We also fill a 2D histogram
            # called "hzvsjetpt" that contains all the z distributions of all jet pt.

            cz = TCanvas('cz' + suffix,
                         'The Efficiency Corrected Signal Yield Canvas' + suffix)
            pz = TPad('pz' + suffix, 'The Efficiency Corrected Signal Yield Canvas' + suffix,
                      0.0, 0.001, 1.0, 1.0)
            setup_pad(pz)
            cz.SetCanvasSize(1900, 1500)
            cz.SetWindowSize(500, 500)
            setup_histogram(hz, 4)
            hz.SetXTitle("#it{z}_{#parallel}^{ch}")
            hz.Draw()
            latex = TLatex(0.6, 0.85, "%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}" %
                           (self.lvar2_binmin_reco[imult], self.lvar2_binmax_reco[imult]))
            draw_latex(latex)
            cz.SaveAs("%s/step1_effcorr_bkgsub_HFptintegrated_sub%s%s_%s_%.2f_%.2f.eps" % \
                      (self.d_resultsallpdata, self.case, self.typean, self.v_var2_binning, \
                       self.lvar2_binmin_reco[imult], self.lvar2_binmax_reco[imult]))

            for zbins in range(nzbin_reco):
                hzvsjetpt.SetBinContent(zbins+1, imult+1, hz.GetBinContent(zbins+1))
                hzvsjetpt.SetBinError(zbins+1, imult+1, hz.GetBinError(zbins+1))
            hz.Scale(1.0/hz.Integral(1, -1))
            fileouts.cd()
            hz.Write("hz" + suffix)

        fileouts.cd()
        hzvsjetpt.Write("hzvsjetpt")
        czvsjetpt = TCanvas('czvsjetpt', '2D input to unfolding (not normalized)')
        pzvsjetpt = TPad('pzvsjetpt', '2D input to unfolding', 0.0, 0.001, 1.0, 1.0)
        setup_pad(pzvsjetpt)
        czvsjetpt.SetCanvasSize(1900, 1500)
        czvsjetpt.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt)
        hzvsjetpt.SetXTitle("#it{z}_{#parallel}^{ch}")
        hzvsjetpt.SetYTitle("#it{p}_{T, jet}")
        hzvsjetpt.Draw("text")
        czvsjetpt.SaveAs("%s/step1_czvsjetpt_inputunfolding.eps" % self.d_resultsallpdata)
        fileouts.Close()

    # pylint: disable=too-many-branches
    def feeddown(self):

        #In this function we compute the feeddown fraction to be subtracted to
        #extract the prompt z distributions of HF tagged jets.

        #The ingredients are the efficiency file that contains prompt and
        #non-prompt efficiency for HF meson reconstruction as a function of pT
        #in bins of jet pt (file_eff) and the output file of the jet processer that
        #contains all the response matrix and jet efficiencies (feeddown_input_file).


        self.loadstyle()
        feeddown_input_file = TFile.Open(self.n_fileff)
        file_eff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc, \
                              self.case, self.typean))
        fileouts = TFile.Open("%s/feeddown%s%s.root" % \
                              (self.d_resultsallpdata, self.case, self.typean), "recreate")

        #The response matrix for non prompt HF meson response_matrix is taken
        #from the feeddown_input_file file and it is calculated as the 4D
        #scatter plots of reco and gen z and jet pt for D-jet candidates in the
        #range of min-max for both reco and gen variables.


        response_matrix = feeddown_input_file.Get("response_matrix_nonprompt")

        # fh3_feeddown is 3d histogram from powheg+pythia prediction that
        # contains z vs jet_pt vs HF pt.

        powheg_input_file = TFile.Open(self.powheg_path_nonprompt)
        input_data = powheg_input_file.Get("fh3_feeddown")

        # output_template is the reco jet pt vs z for candidates in the reco
        # min-max region
        output_template = feeddown_input_file.Get("hzvsjetpt_reco")

        # hzvsjetpt_gen_nocuts_nonprompt is the 2d plot of gen z vs gen jet pt
        # for events in the gen min-max range
        hzvsjetpt_gen_nocuts = feeddown_input_file.Get("hzvsjetpt_gen_nocuts_nonprompt")
        # hzvsjetpt_gen_cuts_nonprompt is the 2d plot of gen z vs gen jet pt
        # for events in the gen and reco min-max range
        hzvsjetpt_gen_eff = feeddown_input_file.Get("hzvsjetpt_gen_cuts_nonprompt")
        hzvsjetpt_gen_eff.Divide(hzvsjetpt_gen_nocuts)

        # hzvsjetpt_reco_nocuts_nonprompt is the 2d plot of reco z vs reco jet pt
        # for events in the reco min-max range
        hzvsjetpt_reco_nocuts = feeddown_input_file.Get("hzvsjetpt_reco_nocuts_nonprompt")
        # hzvsjetpt_reco_cuts_nonprompt is the 2d plot of reco z vs reco jet pt
        # for events in the reco and gen min-max range
        hzvsjetpt_reco_eff = feeddown_input_file.Get("hzvsjetpt_reco_cuts_nonprompt")
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)

        sideband_input_data_file = TFile.Open("%s/sideband_sub%s%s.root" % \
                                               (self.d_resultsallpdata, self.case, self.typean))
        sideband_input_data = sideband_input_data_file.Get("hzvsjetpt")

        hz_genvsreco_list = []
        hjetpt_genvsreco_list = []

        hjetpt_fracdiff_list = []
        hz_fracdiff_list = []
        heff_pr_list = []
        heff_fd_list = []
        input_data_zvsjetpt_list = []
        input_data_scaled = TH2F()

        # cgen_eff is the efficiency that a candidate generated in the gen
        # limits has reco values in the reco limits

        cgen_eff = TCanvas('cgen_eff_nonprompt ', 'gen efficiency applied to feedown')
        pgen_eff = TPad('pgen_eff_nonprompt ', 'gen efficiency applied to feedown',
                        0.0, 0.0, 1.0, 1.0)
        setup_pad(pgen_eff)
        cgen_eff.SetCanvasSize(1900, 1500)
        cgen_eff.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt_gen_eff)
        hzvsjetpt_gen_eff.SetXTitle("z^{gen}")
        hzvsjetpt_gen_eff.SetYTitle("#it{p}_{T, jet}^{gen}")
        hzvsjetpt_gen_eff.Draw("text")
        cgen_eff.SaveAs("%s/cgen_kineeff_nonprompt.eps" % (self.d_resultsallpdata))

        # creco_eff is the kinematic efficiency that a candidate in reco limits
        # limits has gen values in the gen limits

        creco_eff = TCanvas('creco_eff_nonprompt ', 'reco efficiency applied to feedown')
        preco_eff = TPad('preco_eff_nonprompt ', 'reco efficiency applied to feedown',
                         0.0, 0.0, 1.0, 1.0)
        setup_pad(preco_eff)
        creco_eff.SetCanvasSize(1900, 1500)
        creco_eff.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt_reco_eff)
        hzvsjetpt_reco_eff.SetXTitle("z^{reco}")
        hzvsjetpt_reco_eff.SetYTitle("#it{p}_{T, jet}^{reco}")
        hzvsjetpt_reco_eff.Draw("text")
        creco_eff.SaveAs("%s/creco_kineeff_nonprompt.eps" % (self.d_resultsallpdata))

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                      self.lvar2_binmax_reco[ibin2])
            hz_genvsreco_list.append(feeddown_input_file.Get("hz_genvsreco_nonprompt" + suffix))

            cz_genvsreco = TCanvas('cz_genvsreco_nonprompt' + suffix,
                                   'response matrix 2D projection')
            pz_genvsreco = TPad('pz_genvsreco_nonprompt ',
                                'response matrix 2D projection', 0.0, 0.001, 1.0, 1.0)
            setup_pad(pz_genvsreco)
            cz_genvsreco.SetLogz()
            pz_genvsreco.SetLogz()
            cz_genvsreco.SetCanvasSize(1900, 1500)
            cz_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hz_genvsreco_list[ibin2])
            hz_genvsreco_list[ibin2].SetXTitle("z^{gen}")
            hz_genvsreco_list[ibin2].SetYTitle("z^{reco}")
            hz_genvsreco_list[ibin2].Draw("colz")
            latex = TLatex(0.2, 0.85, '%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}' \
                           % (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cz_genvsreco.SaveAs("%s/cz_genvsreco_nonprompt_%s.eps" % \
                                (self.d_resultsallpdata, suffix))

        for ibinshape in range(self.p_nbinshape_reco):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_reco[ibinshape],
                      self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco_list.append( \
                feeddown_input_file.Get("hjetpt_genvsreco_nonprompt" + suffix))

            cjetpt_genvsreco = TCanvas('cjetpt_genvsreco_nonprompt' + suffix,
                                       'response matrix 2D projection')
            pjetpt_genvsreco = TPad('pjetpt_genvsreco_nonprompt' + suffix,
                                    'response matrix 2D projection', 0.0, 0.001, 1.0, 1.0)
            setup_pad(pjetpt_genvsreco)
            cjetpt_genvsreco.SetLogz()
            pjetpt_genvsreco.SetLogz()
            cjetpt_genvsreco.SetCanvasSize(1900, 1500)
            cjetpt_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hjetpt_genvsreco_list[ibinshape])
            hjetpt_genvsreco_list[ibinshape].SetXTitle("#it{p}_{T, jet}^{gen}")
            hjetpt_genvsreco_list[ibinshape].SetYTitle("#it{p}_{T, jet}^{reco}")
            hjetpt_genvsreco_list[ibinshape].Draw("colz")
            latex = TLatex(0.2, 0.8, "%.2f < #it{z}_{#parallel}^{ch} < %.2f" % \
                (self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape]))
            draw_latex(latex)
            cjetpt_genvsreco.SaveAs("%s/cjetpt_genvsreco_nonprompt_%s.eps" % \
                                    (self.d_resultsallpdata, suffix))

        hz_genvsreco_full = feeddown_input_file.Get("hz_genvsreco_full_nonprompt")
        hjetpt_genvsreco_full = feeddown_input_file.Get("hjetpt_genvsreco_full_nonprompt")

        cz_genvsreco = TCanvas('cz_genvsreco_full_nonprompt', 'response matrix 2D projection')
        pz_genvsreco = TPad('pz_genvsreco_full_nonprompt' + suffix,
                            'response matrix 2D projection', 0.0, 0.001, 1.0, 1.0)
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

        cjetpt_genvsreco = TCanvas('cjetpt_genvsreco_full_nonprompt',
                                   'response matrix 2D projection')
        pjetpt_genvsreco = TPad('pjetpt_genvsreco_full_nonprompt' + suffix,
                                'response matrix 2D projection', 0.0, 0.001, 1.0, 1.0)
        setup_pad(pjetpt_genvsreco)
        cjetpt_genvsreco.SetLogz()
        pjetpt_genvsreco.SetLogz()
        cjetpt_genvsreco.SetCanvasSize(1900, 1500)
        cjetpt_genvsreco.SetWindowSize(500, 500)
        setup_histogram(hjetpt_genvsreco_full)
        hjetpt_genvsreco_full.SetXTitle("#it{p}_{T, jet}^{gen}")
        hjetpt_genvsreco_full.SetYTitle("#it{p}_{T, jet}^{reco}")
        hjetpt_genvsreco_full.Draw("colz")
        cjetpt_genvsreco.SaveAs("%s/cjetpt_genvsreco_full_nonprompt.eps" % \
                                (self.d_resultsallpdata))

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2],
                      self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff_list.append( \
                feeddown_input_file.Get("hjetpt_fracdiff_nonprompt" + suffix))
            heff_pr_list.append(file_eff.Get("eff_mult%d" % ibin2))
            heff_fd_list.append(file_eff.Get("eff_fd_mult%d" % ibin2))

            ceff = TCanvas('ceff ' + suffix, 'prompt and non-prompt efficiencies' + suffix)
            peff = TPad('peff' + suffix, 'prompt and non-prompt efficiencies',
                        0.0, 0.001, 1.0, 1.0)
            setup_pad(peff)
            ceff.SetCanvasSize(1900, 1500)
            ceff.SetWindowSize(500, 500)
            leg_eff = TLegend(.65, .55, .8, .7, "")
            setup_legend(leg_eff)
            setup_histogram(heff_pr_list[ibin2], 2)
            leg_eff.AddEntry(heff_pr_list[ibin2], "prompt", "LEP")
            heff_pr_list[ibin2].GetYaxis().SetRangeUser( \
                0.5*min(heff_pr_list[ibin2].GetMinimum(), heff_fd_list[ibin2].GetMinimum()), \
                1.1*max(heff_pr_list[ibin2].GetMaximum(), heff_fd_list[ibin2].GetMaximum()))
            heff_pr_list[ibin2].SetXTitle("#it{p}_{T, #Lambda_{c}^{#plus}} (GeV/#it{c})")
            heff_pr_list[ibin2].SetYTitle("Efficiency #times Acceptance ")
            heff_pr_list[ibin2].SetTitleOffset(1.2, "Y")
            heff_pr_list[ibin2].SetTitle("")
            heff_pr_list[ibin2].Draw()
            setup_histogram(heff_fd_list[ibin2], 4, 24)
            leg_eff.AddEntry(heff_fd_list[ibin2], "non-prompt", "LEP")
            heff_fd_list[ibin2].Draw("same")
            leg_eff.Draw("same")
            latex = TLatex(0.52, 0.45, "ALICE Preliminary")
            draw_latex(latex)
            latex2 = TLatex(0.52, 0.4, "PYTHIA 6, pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex2)
            latex3 = TLatex(0.52, 0.35, "#Lambda_{c}^{#plus} \
                            #rightarrow p K_{S}^{0} (and charge conj.)")
            draw_latex(latex3)
            latex4 = TLatex(0.52, 0.3, "in charged jets, anti-#it{k}_{T}, #it{R} = 0.4")
            draw_latex(latex4)
            latex5 = TLatex(0.52, 0.25, "%.0f < #it{p}_{T, jet}^{ch} < %.0f GeV/#it{c}" \
                            % (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex5)
            latex6 = TLatex(0.52, 0.2, "#left|#it{#eta}_{jet}#right| < 0.5")
            draw_latex(latex6)
            ceff.SaveAs("%s/ceff_prompt_nonprompt_%s.eps" % (self.d_resultsallpdata, suffix))

        cjetpt_fracdiff = TCanvas('cjetpt_fracdiff ',
                                  'non-prompt jetpt response fractional differences')
        pjetpt_fracdiff = TPad('pjetpt_fracdiff', \
            'non-prompt jetpt response fractional differences', 0.0, 0.001, 1.0, 1.0)
        setup_pad(pjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        pjetpt_fracdiff.SetLogy()
        cjetpt_fracdiff.SetCanvasSize(1900, 1500)
        cjetpt_fracdiff.SetWindowSize(500, 500)
        leg_jetpt_fracdiff = TLegend(.65, .5, .8, .8, "#it{p}_{T, jet}^{gen}")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            setup_histogram(hjetpt_fracdiff_list[ibin2], ibin2+1)
            leg_jetpt_fracdiff.AddEntry(hjetpt_fracdiff_list[ibin2], \
                "%d-%d GeV/#it{c}" %(self.lvar2_binmin_gen[ibin2], \
                self.lvar2_binmax_gen[ibin2]), "LEP")
            if ibin2 == 0:
                hjetpt_fracdiff_list[ibin2].SetXTitle(\
                    "(#it{p}_{T, jet}^{reco} #minus #it{p}_{T, jet}^{gen})/#it{p}_{T, jet}^{gen}")
                hjetpt_fracdiff_list[ibin2].GetYaxis().SetRangeUser(0.001, \
                    hjetpt_fracdiff_list[ibin2].GetMaximum()*3)
            hjetpt_fracdiff_list[ibin2].Draw("same")
        leg_jetpt_fracdiff.Draw("same")
        cjetpt_fracdiff.SaveAs("%s/cjetpt_fracdiff_nonprompt.eps" % (self.d_resultsallpdata))

        for ibinshape in range(self.p_nbinshape_gen):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_list.append(feeddown_input_file.Get("hz_fracdiff_nonprompt" + suffix))

        cz_fracdiff = TCanvas('cz_fracdiff ', 'non-prompt z response fractional differences')
        pz_fracdiff = TPad('pz_fracdiff', 'non-prompt z response fractional differences',
                           0.0, 0.001, 1.0, 1.0)
        setup_pad(pz_fracdiff)
        cz_fracdiff.SetLogy()
        pz_fracdiff.SetLogy()
        cz_fracdiff.SetCanvasSize(1900, 1500)
        cz_fracdiff.SetWindowSize(500, 500)
        leg_z_fracdiff = TLegend(.2, .5, .4, .85, "z")
        setup_legend(leg_z_fracdiff)
        for ibinshape in range(self.p_nbinshape_gen):
            setup_histogram(hz_fracdiff_list[ibinshape], ibinshape+1)
            leg_z_fracdiff.AddEntry(hz_fracdiff_list[ibinshape], \
                "%.4f-%.4f" %(self.lvarshape_binmin_gen[ibinshape], \
                self.lvarshape_binmax_gen[ibinshape]), "LEP")
            if ibin2 == 0:
                hz_fracdiff_list[ibin2].SetXTitle("(z}^{reco}-z^{gen})/z^{gen}")
                hz_fracdiff_list[ibin2].GetYaxis().SetRangeUser(0.001, \
                    hz_fracdiff_list[ibin2].GetMaximum()*3)
            hz_fracdiff_list[ibinshape].Draw("same")
        leg_z_fracdiff.Draw("same")
        cz_fracdiff.SaveAs("%s/cz_fracdiff_nonprompt.eps" % (self.d_resultsallpdata))


        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            suffix = "%s%d_%d_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id])
            input_data.GetZaxis().SetRange(ipt+1, ipt+1)
            input_data_zvsjetpt_list.append( \
                input_data.Project3D("input_data_zvsjetpt" + suffix + "_yxe"))
            for ibin2 in range(self.p_nbin2_gen):
                for ibinshape in range(self.p_nbinshape_gen):
                    if(heff_pr_list[ibin2].GetBinContent(ipt+1) == 0 or \
                       heff_fd_list[ibin2].GetBinContent(ipt+1) == 0):
                        input_data_zvsjetpt_list[ipt].SetBinContent(ibinshape+1, ibin2 + 1, 0.0)
                    else:
                        input_data_zvsjetpt_list[ipt].SetBinContent(ibinshape+1, ibin2+1, \
                            input_data_zvsjetpt_list[ipt].GetBinContent(ibinshape+1, ibin2+1)* \
                            (heff_fd_list[ibin2].GetBinContent(ipt + 1)/ \
                             heff_pr_list[ibin2].GetBinContent(ipt+1)))
            if ipt == 0:
                input_data_scaled = input_data_zvsjetpt_list[ipt].Clone("input_data_scaled")
            else:
                input_data_scaled.Add(input_data_zvsjetpt_list[ipt])
        input_data_scaled.Multiply(hzvsjetpt_gen_eff)
        input_data_scaled.Scale(self.p_nevents*self.branching_ratio/self.xsection_inel)
        folded = folding(input_data_scaled, response_matrix, output_template)
        folded.Sumw2()
        folded.Divide(hzvsjetpt_reco_eff)

        folded_z_list = []
        input_data_scaled_z_list = []
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])

            folded_z_list.append(folded.ProjectionX("folded_z_nonprompt_" + suffix,
                                                    ibin2+1, ibin2+1, "e"))
            input_data_scaled_z_list.append( \
                input_data_scaled.ProjectionX("Powheg_scaled_nonprompt_" + suffix, \
                    input_data_scaled.GetYaxis().FindBin(self.lvar2_binmin_gen[ibin2]), \
                    input_data_scaled.GetYaxis().FindBin(self.lvar2_binmin_gen[ibin2]), "e"))
            c_fd_fold = TCanvas('c_fd_fold ' + suffix, 'Powheg and folded' + suffix)
            p_fd_fold = TPad('p_fd_fold' + suffix,
                             'Powheg and folded' + suffix, 0.0, 0.001, 1.0, 1.0)
            setup_pad(p_fd_fold)
            c_fd_fold.SetCanvasSize(1900, 1500)
            c_fd_fold.SetWindowSize(500, 500)
            leg_fd_fold = TLegend(.2, .75, .4, .85, "")
            setup_legend(leg_fd_fold)
            setup_histogram(input_data_scaled_z_list[ibin2], 2)
            leg_fd_fold.AddEntry(input_data_scaled_z_list[ibin2], "Powheg eff corrected", "LEP")
            input_data_scaled_z_list[ibin2].GetYaxis().SetRangeUser(0.0, \
                    input_data_scaled_z_list[ibin2].GetMaximum()*1.5)
            input_data_scaled_z_list[ibin2].SetXTitle("#it{z}_{#parallel}^{ch}")
            input_data_scaled_z_list[ibin2].Draw()
            setup_histogram(folded_z_list[ibin2], 4)
            leg_fd_fold.AddEntry(folded_z_list[ibin2], "folded", "LEP")
            folded_z_list[ibin2].Draw("same")
            leg_fd_fold.Draw("same")
            latex = TLatex(0.4, 0.25, "%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_fd_fold.SaveAs("%s/cfolded_Powheg_%s.eps" % (self.d_resultsallpdata, suffix))
        fileouts.cd()
        sideband_input_data_subtracted = \
            sideband_input_data.Clone("sideband_input_data_subtracted")
        sideband_input_data_subtracted.Add(folded, -1)
        for ibin2 in range(self.p_nbin2_reco):
            for ibinshape in range(self.p_nbinshape_reco):
                if sideband_input_data_subtracted.GetBinContent( \
                sideband_input_data_subtracted.FindBin(self.lvarshape_binmin_reco[ibinshape], \
                self.lvar2_binmin_reco[ibin2])) < 0.0:
                    sideband_input_data_subtracted.SetBinContent( \
                        sideband_input_data_subtracted.FindBin( \
                            self.lvarshape_binmin_reco[ibinshape], \
                            self.lvar2_binmin_reco[ibin2]), 0.0)
        sideband_input_data_subtracted.Write()

        sideband_input_data_z = []
        sideband_input_data_subtracted_z = []

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                      self.lvar2_binmax_reco[ibin2])
            sideband_input_data_z.append( \
                sideband_input_data.ProjectionX("sideband_input_data_z" + suffix,
                                                ibin2+1, ibin2+1, "e"))
            sideband_input_data_subtracted_z.append( \
                sideband_input_data_subtracted.ProjectionX( \
                    "sideband_input_data_subtracted_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            cfeeddown = TCanvas('cfeeddown' + suffix, 'cfeeddown' + suffix)
            pfeeddown = TPad('pfeeddown' + suffix, 'cfeeddown' + suffix, 0.0, 0.001, 1.0, 1.0)
            setup_pad(pfeeddown)
            if ibin2 != 2:
                cfeeddown.SetLogy()
                pfeeddown.SetLogy()
            cfeeddown.SetCanvasSize(1900, 1500)
            cfeeddown.SetWindowSize(500, 500)
            legmin = .2
            legmax = .4
            if ibin2 == 2:
                legmin = .7
                legmax = .85
            leg_feeddown = TLegend(.2, legmin, .4, legmax, "")
            setup_legend(leg_feeddown)
            setup_histogram(sideband_input_data_z[ibin2], 2)
            leg_feeddown.AddEntry(sideband_input_data_z[ibin2], "prompt+non-prompt", "LEP")
            sideband_input_data_z[ibin2].GetYaxis().SetRangeUser(0.1, \
                    sideband_input_data_z[ibin2].GetMaximum()*3)
            sideband_input_data_z[ibin2].SetXTitle("#it{z}_{#parallel}^{ch}")
            sideband_input_data_z[ibin2].SetYTitle("Yeild")
            sideband_input_data_z[ibin2].Draw()
            setup_histogram(sideband_input_data_subtracted_z[ibin2], 3)
            leg_feeddown.AddEntry(sideband_input_data_subtracted_z[ibin2],
                                  "subtracted (prompt)", "LEP")
            sideband_input_data_subtracted_z[ibin2].Draw("same")
            setup_histogram(folded_z_list[ibin2], 4)
            leg_feeddown.AddEntry(folded_z_list[ibin2], "non-prompt powheg", "LEP")
            folded_z_list[ibin2].Draw("same")
            leg_feeddown.Draw("same")
            if ibin2 != 2:
                latex = TLatex(0.6, 0.3, "%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}" \
                    % (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
                latex = TLatex(0.6, 0.3, "%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
            else:
                latex = TLatex(0.6, 0.75, "%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cfeeddown.SaveAs("%s/cfeeddown_subtraction_%s.eps" % \
                             (self.d_resultsallpdata, suffix))

            feeddown_fraction = folded_z_list[ibin2].Clone("feeddown_fraction" + suffix)
            feeddown_fraction_denominator = \
                sideband_input_data_z[ibin2].Clone("feeddown_denominator" + suffix)
            feeddown_fraction.Divide(feeddown_fraction_denominator)
            feeddown_fraction.Write()

            cfeeddown_fraction = TCanvas('cfeeddown_fraction' + suffix,
                                         'cfeeddown_fraction' + suffix)
            pfeeddown_fraction = TPad('pfeeddown_fraction' + suffix,
                                      'cfeeddown_fraction' + suffix, 0.0, 0.001, 1.0, 1.0)
            setup_pad(pfeeddown_fraction)
            if ibin2 != 2:
                cfeeddown_fraction.SetLogy()
                pfeeddown_fraction.SetLogy()
            cfeeddown_fraction.SetCanvasSize(1900, 1500)
            cfeeddown_fraction.SetWindowSize(500, 500)
            setup_histogram(feeddown_fraction, 4)
            feeddown_fraction.SetXTitle("#it{z}_{#parallel}^{ch}")
            feeddown_fraction.SetYTitle("b-feeddown fraction")
            feeddown_fraction.Draw()
            latex = TLatex(0.6, 0.75, "%.2f < #it{p}_{T, jet} < %.2f GeV/#it{c}" % \
                           (self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]))
            latex = TLatex(0.6, 0.7, "powheg based estimation")
            draw_latex(latex)
            cfeeddown_fraction.SaveAs("%s/cfeeddown_fraction_%s.eps" % \
                                      (self.d_resultsallpdata, suffix))

        cfeeddown_output = TCanvas('cfeeddown_output', 'cfeeddown_output')
        pfeeddown_output = TPad('pfeeddown_output', 'pfeeddown_output',
                                0.0, 0.001, 1.0, 1.0)
        setup_pad(pfeeddown_output)
        cfeeddown_output.SetCanvasSize(1900, 1500)
        cfeeddown_output.SetWindowSize(500, 500)
        setup_histogram(sideband_input_data_subtracted)
        sideband_input_data_subtracted.Draw("text")
        cfeeddown_output.SaveAs("%s/cfeeddown_output.eps" % (self.d_resultsallpdata))
        print("end of folding")
