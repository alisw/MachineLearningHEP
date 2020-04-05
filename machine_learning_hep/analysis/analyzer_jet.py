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
# pylint: disable=too-many-lines, bad-whitespace, line-too-long
import os
from math import sqrt
from array import array
import numpy as np
import yaml
# pylint: disable=import-error, no-name-in-module
import uproot
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TLatex, TGraphAsymmErrors
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils
from ROOT import TLegend
from ROOT import gROOT
from ROOT import kGreen, kRed # kBlue, kBlack, kOrange
from ROOT import RooUnfoldBayes
# HF specific imports
from machine_learning_hep.utilities import folding, equal_binning_lists, make_message_notfound, checkmakedir
from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.utilities import setup_histogram, setup_pad
from machine_learning_hep.utilities import setup_legend, setup_tgraph, draw_latex, tg_sys
from machine_learning_hep.do_systematics import healthy_structure, format_varname, format_varlabel
from machine_learning_hep.utilities_plot import makefill2dhist, makefill3dhist
from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.utilities import seldf_singlevar

# pylint: disable=too-many-instance-attributes, too-many-statements
class AnalyzerJet(Analyzer):
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        # machine learning
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]

        # normalisation
        self.p_nevents = datap["analysis"][self.typean]["nevents"]
        self.branching_ratio = \
            datap["analysis"][self.typean].get("branching_ratio", None)
        self.xsection_inel = \
            datap["analysis"][self.typean].get("xsection_inel", None)

        # plotting
        # LaTeX string
        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]
        self.p_latexndecay = datap["analysis"][self.typean]["latexnamedecay"]
        self.p_latexbin2var = datap["analysis"][self.typean]["latexbin2var"]
        self.v_varshape_latex = datap["analysis"][self.typean]["var_shape_latex"]

        # first variable (hadron pt)
        self.v_var_binning = datap["var_binning"] # name
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin) # number of bins
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.var1ranges = self.lpt_finbinmin.copy()
        self.var1ranges.append(self.lpt_finbinmax[-1])
        self.var1binarray = array('d', self.var1ranges) # array of bin edges to use in histogram constructors

        # second variable (jet pt)
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"] # name
        self.lvar2_binmin_reco = datap["analysis"][self.typean].get("sel_binmin2_reco", None)
        self.lvar2_binmax_reco = datap["analysis"][self.typean].get("sel_binmax2_reco", None)
        self.p_nbin2_reco = len(self.lvar2_binmin_reco) # number of reco bins
        self.lvar2_binmin_gen = datap["analysis"][self.typean].get("sel_binmin2_gen", None)
        self.lvar2_binmax_gen = datap["analysis"][self.typean].get("sel_binmax2_gen", None)
        self.p_nbin2_gen = len(self.lvar2_binmin_gen) # number of gen bins
        self.var2ranges_reco = self.lvar2_binmin_reco.copy()
        self.var2ranges_reco.append(self.lvar2_binmax_reco[-1])
        self.var2binarray_reco = array('d', self.var2ranges_reco) # array of bin edges to use in histogram constructors
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        self.var2binarray_gen = array('d', self.var2ranges_gen) # array of bin edges to use in histogram constructors

        # observable (z, shape,...)
        self.v_varshape_binning = datap["analysis"][self.typean]["var_binningshape"] # name (reco)
        self.v_varshape_binning_gen = datap["analysis"][self.typean]["var_binningshape_gen"] # name (gen)
        self.lvarshape_binmin_reco = \
            datap["analysis"][self.typean].get("sel_binminshape_reco", None)
        self.lvarshape_binmax_reco = \
            datap["analysis"][self.typean].get("sel_binmaxshape_reco", None)
        self.p_nbinshape_reco = len(self.lvarshape_binmin_reco) # number of reco bins
        self.lvarshape_binmin_gen = \
            datap["analysis"][self.typean].get("sel_binminshape_gen", None)
        self.lvarshape_binmax_gen = \
            datap["analysis"][self.typean].get("sel_binmaxshape_gen", None)
        self.p_nbinshape_gen = len(self.lvarshape_binmin_gen) # number of gen bins
        self.varshaperanges_reco = self.lvarshape_binmin_reco.copy()
        self.varshaperanges_reco.append(self.lvarshape_binmax_reco[-1])
        self.varshapebinarray_reco = array('d', self.varshaperanges_reco) # array of bin edges to use in histogram constructors
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])
        self.varshapebinarray_gen = array('d', self.varshaperanges_gen) # array of bin edges to use in histogram constructors

        # fitting
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

        # side-band subtraction
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

        # feed-down
        self.powheg_path_nonprompt = \
            datap["analysis"][self.typean].get("powheg_path_nonprompt", None)
        # systematics variations

        # models to compare with
        # POWHEG + PYTHIA 6
        self.powheg_path_prompt = \
            datap["analysis"][self.typean].get("powheg_path_prompt", None)
        self.powheg_prompt_variations = \
            datap["analysis"][self.typean].get("powheg_prompt_variations", None)
        self.powheg_prompt_variations_path = \
            datap["analysis"][self.typean].get("powheg_prompt_variations_path", None)
        # PYTHIA 8
        self.pythia8_prompt_variations_path = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations_path", None)
        self.pythia8_prompt_variations = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations", None)
        self.pythia8_prompt_variations_legend = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations_legend", None)

        # unfolding
        self.niter_unfolding = \
            datap["analysis"][self.typean].get("niterunfolding", None)
        self.choice_iter_unfolding = \
            datap["analysis"][self.typean].get("niterunfoldingchosen", None)

        # systematics
        # import parameters of variations from the variation database
        path_sys_db = datap["analysis"][self.typean].get("systematics_db", None)
        if not path_sys_db:
            self.logger.fatal(make_message_notfound("the variation database"))
        with open(path_sys_db, 'r') as file_sys:
            db_sys = yaml.safe_load(file_sys)
        if not healthy_structure(db_sys):
            self.logger.fatal("Bad structure of the variation database.")
        db_sys = db_sys["categories"]
        self.systematic_catnames = [catname for catname, val in db_sys.items() if val["activate"]]
        self.n_sys_cat = len(self.systematic_catnames)
        self.systematic_catlabels = [""] * self.n_sys_cat
        self.systematic_varnames = [None] * self.n_sys_cat
        self.systematic_varlabels = [None] * self.n_sys_cat
        self.systematic_variations = [0] * self.n_sys_cat
        self.systematic_correlation = [None] * self.n_sys_cat
        self.systematic_rms = [False] * self.n_sys_cat
        self.systematic_symmetrise = [False] * self.n_sys_cat
        self.systematic_rms_both_sides = [False] * self.n_sys_cat
        self.powheg_nonprompt_varnames = []
        for c, catname in enumerate(self.systematic_catnames):
            self.systematic_catlabels[c] = db_sys[catname]["label"]
            self.systematic_varnames[c] = []
            self.systematic_varlabels[c] = []
            for varname, val in db_sys[catname]["variations"].items():
                n_var = len(val["activate"])
                for a, act in enumerate(val["activate"]):
                    if act:
                        varname_i = format_varname(varname, a, n_var)
                        varlabel_i = format_varlabel(val["label"], a, n_var)
                        self.systematic_varnames[c].append(varname_i)
                        self.systematic_varlabels[c].append(varlabel_i)
                        if catname == "powheg":
                            self.powheg_nonprompt_varnames.append(varname_i)
            self.systematic_variations[c] = len(self.systematic_varnames[c])
            self.systematic_correlation[c] = db_sys[catname]["correlation"]
            self.systematic_rms[c] = db_sys[catname]["rms"]
            self.systematic_symmetrise[c] = db_sys[catname]["symmetrise"]
            self.systematic_rms_both_sides[c] = db_sys[catname]["rms_both_sides"]

        # output directories
        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]
        # create otput directories in case they do not exist
        for dir_out in [self.d_resultsallpmc, self.d_resultsallpdata]:
            if not os.path.exists(dir_out):
                checkmakedir(dir_out)

        # input directories (processor output)
        self.d_resultsallpmc_proc = self.d_resultsallpmc
        self.d_resultsallpdata_proc = self.d_resultsallpdata
        # use a different processor output
        if "data_proc" in datap["analysis"][typean]:
            self.d_resultsallpdata_proc = datap["analysis"][typean]["data_proc"]["results"][period] \
                    if period is not None else datap["analysis"][typean]["data_proc"]["resultsallp"]
        if "mc_proc" in datap["analysis"][typean]:
            self.d_resultsallpmc_proc = datap["analysis"][typean]["mc_proc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc_proc"]["resultsallp"]

        # input files
        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata_proc, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc_proc, n_filemass_name)
        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc_proc, self.n_fileff)

        # output filenames
        self.yields_filename = "yields"
        self.fits_dirname = "fits"
        self.yields_syst_filename = "yields_syst"
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

        # Save the database in the results directory.
        path_db_out = os.path.join(self.d_resultsallpdata, "database_%s_%s.yml" % (self.case, self.typean))
        with open(path_db_out, 'w') as file_db_out:
            yaml.safe_dump(datap, file_db_out, default_flow_style=False)

    def fit(self):
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile.Open(fileout_name, "recreate")
        if not fileout:
            self.logger.fatal(make_message_notfound(fileout_name))
        myfilemc = TFile.Open(self.n_filemass_mc)
        if not myfilemc:
            self.logger.fatal(make_message_notfound(self.n_filemass_mc))
        myfile = TFile.Open(self.n_filemass)
        if not myfile:
            self.logger.fatal(make_message_notfound(self.n_filemass))
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            for ibin2 in range(self.p_nbin2_reco):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                          self.lvar2_binmax_reco[ibin2])
                histomassmc = myfilemc.Get("hmass_sig" + suffix)
                if not histomassmc:
                    self.logger.fatal(make_message_notfound("hmass_sig" + suffix, self.n_filemass_mc))
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
                if not histomass:
                    self.logger.fatal(make_message_notfound("hmass" + suffix, self.n_filemass))
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
                histomass_reb.SetTitle("")
                histomass_reb.SetXTitle("mass (GeV/#it{c}^{2})")
                histomass_reb.SetYTitle("counts")
                histomass_reb.Draw("same")
                if out == 1:
                    bkg_func.SetLineColor(kGreen)
                    sgn_func.SetLineColor(kRed)
                    sgn_func.Draw("same")
                    bkg_func.Draw("same")
                latex = TLatex(0.2, 0.85, '%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.2, 0.8, '%.2f < #it{p}_{T, %s} < %.2f GeV/#it{c}' % \
                    (self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt],self.lvar2_binmax_reco[ibin2])))
                draw_latex(latex2)
                c_fitted_result.SaveAs("%s/step0_fitted_result_%s.eps" % \
                    (self.d_resultsallpdata, suffix))
        myfilemc.Close()
        myfile.Close()
        fileout.Close()
        gROOT.SetBatch(tmp_is_root_batch)

    def efficiency(self):
        self.loadstyle()

        lfileeff = TFile.Open(self.n_fileff)
        if not lfileeff:
            self.logger.fatal(make_message_notfound(self.n_fileff))
        path = "%s/efficiencies%s%s.root" % (self.d_resultsallpmc, self.case, self.typean)
        fileouteff = TFile.Open(path, "recreate")
        if not fileouteff:
            self.logger.fatal(make_message_notfound(path))

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
            legeff.AddEntry(h_sel_pr, legeffstring, "LE")
            h_sel_pr.SetTitle("")
            h_sel_pr.GetXaxis().SetTitle("#it{p}_{T}^{%s} (GeV/#it{c})" % self.p_latexnhadron)
            h_sel_pr.GetYaxis().SetTitle("prompt %s-jet efficiency" % self.p_latexnhadron)
            h_sel_pr.SetMinimum(0.)
            h_sel_pr.SetMaximum(1.5)
        legeff.Draw()
        cEff.SaveAs("%s/Eff%s%s.eps" % (self.d_resultsallpdata,
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
            legeffFD.AddEntry(h_sel_fd, legeffFDstring, "LE")
            h_sel_fd.SetTitle("")
            h_sel_fd.GetXaxis().SetTitle("#it{p}_{T}^{%s} (GeV/#it{c})" % self.p_latexnhadron)
            h_sel_fd.GetYaxis().SetTitle("non-prompt %s-jet efficiency" % self.p_latexnhadron)
            h_sel_fd.SetMinimum(0.)
            h_sel_fd.SetMaximum(1.5)
        legeffFD.Draw()
        cEffFD.SaveAs("%s/EffFD%s%s.eps" % (self.d_resultsallpdata, \
                                            self.case, self.typean))

    # pylint: disable=too-many-locals, too-many-branches
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
        if not lfile:
            self.logger.fatal(make_message_notfound(self.n_filemass))
        func_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                            None, [self.case, self.typean])
        func_file = TFile.Open(func_filename)
        if not func_file:
            self.logger.fatal(make_message_notfound(func_filename))
        path = "%s/efficiencies%s%s.root" % (self.d_resultsallpmc, self.case, self.typean)
        eff_file = TFile.Open(path)
        if not eff_file:
            self.logger.fatal(make_message_notfound(path))
        path = "%s/sideband_sub%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        fileouts = TFile.Open(path, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(path))
        fileouts.cd()

        # hzvsjetpt is going to be the side-band subtracted histogram of z vs
        # jet that is going to be filled after subtraction

        hzvsjetpt = TH2F("hzvsjetpt", "", self.p_nbinshape_reco, self.varshapebinarray_reco,
                         self.p_nbin2_reco, self.var2binarray_reco)
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

                # Below a list of histograms are defined:
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
                    bkg_fit.Integral(masslow2sig, masshigh2sig)/area_scale_denominator
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
                hzsub_noteffscaled.Write()

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
                hzsub.SetTitle("Signal yield, bg subtracted, efficiency corrected")
                hzsub.SetXTitle(self.v_varshape_latex)
                hzsub.SetYTitle("yield")
                hzsub.GetYaxis().SetTitleOffset(1.5)
                hzsub.Draw()
                latex = TLatex(0.6, 0.85, "%.2f < %s < %.2f GeV/#it{c}" % (self.lvar2_binmin_reco[imult], self.p_latexbin2var, self.lvar2_binmax_reco[imult]))
                draw_latex(latex)
                latex2 = TLatex(0.6, 0.8,
                                "%.2f < #it{p}_{T, %s} < %.2f GeV/#it{c}" \
                                % (self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt],self.lvar2_binmax_reco[imult])))
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
                legsigbkgsubz.AddEntry(hzsig, "signal region", "P")
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
                hzsig.SetTitle("")
                hzsig.SetXTitle(self.v_varshape_latex)
                hzsig.SetYTitle("Yield")
                hzsig.GetYaxis().SetTitleOffset(1.4)
                hzsig.GetYaxis().SetMaxDigits(3)
                hzsig.Draw()
                setup_histogram(hzbkg_scaled, 3, 24)
                legsigbkgsubz.AddEntry(hzbkg_scaled, "side-band region", "P")
                hzbkg_scaled.Draw("same")
                setup_histogram(hzsub_noteffscaled, 4, 28)
                legsigbkgsubz.AddEntry(hzsub_noteffscaled, "subtracted", "P")
                hzsub_noteffscaled.Draw("same")
                legsigbkgsubz.Draw("same")
                #PREL latex = TLatex(0.42, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
                latex = TLatex(0.42, 0.85, "pp, #sqrt{#it{s}} = 13 TeV")
                draw_latex(latex)
                latex1 = TLatex(0.42, 0.8, ("charged jets, anti-#it{k}_{T}, "
                                            "#it{R} = 0.4, #left|#it{#eta}_{jet}#right| < 0.5"))
                draw_latex(latex1)
                latex2 = TLatex(0.42, 0.75, "%.0f < %s < %.0f GeV/#it{c}" \
                                % (self.lvar2_binmin_reco[imult], self.p_latexbin2var, self.lvar2_binmax_reco[imult]))
                draw_latex(latex2)
                latex3 = TLatex(0.42, 0.7, ("with %s (& cc), %.0f < "
                                            "#it{p}_{T, %s} < %.0f GeV/#it{c}")
                                % (self.p_latexnhadron, self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt],self.lvar2_binmax_reco[imult])))
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
            hz.SetTitle("Signal yield, bg subtracted, efficiency corrected")
            hz.SetXTitle(self.v_varshape_latex)
            hz.SetYTitle("yield")
            hz.GetYaxis().SetTitleOffset(1.5)
            hz.Draw()
            latex = TLatex(0.6, 0.85, "%.2f < %s < %.2f GeV/#it{c}" %
                           (self.lvar2_binmin_reco[imult], self.p_latexbin2var, self.lvar2_binmax_reco[imult]))
            draw_latex(latex)
            cz.SaveAs("%s/step1_effcorr_bkgsub_HFptintegrated_sub%s%s_%s_%.2f_%.2f.eps" % \
                      (self.d_resultsallpdata, self.case, self.typean, self.v_var2_binning, \
                       self.lvar2_binmin_reco[imult], self.lvar2_binmax_reco[imult]))

            for zbins in range(self.p_nbinshape_reco):
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
        hzvsjetpt.SetTitle("")
        hzvsjetpt.SetXTitle(self.v_varshape_latex)
        hzvsjetpt.SetYTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
        hzvsjetpt.Draw("text")
        czvsjetpt.SaveAs("%s/step1_czvsjetpt_inputunfolding.eps" % self.d_resultsallpdata)
        fileouts.Close()

    def feeddown(self):

        #In this function we compute the feeddown fraction to be subtracted to
        #extract the prompt z distributions of HF tagged jets.

        #The ingredients are the efficiency file that contains prompt and
        #non-prompt efficiency for HF hadron reconstruction as a function of pT
        #in bins of jet pt (file_eff) and the output file of the jet processer that
        #contains all the response matrix and jet efficiencies (feeddown_input_file).


        self.loadstyle()
        feeddown_input_file = TFile.Open(self.n_fileff)
        if not feeddown_input_file:
            self.logger.fatal(make_message_notfound(self.n_fileff))
        path = "%s/efficiencies%s%s.root" % (self.d_resultsallpmc, self.case, self.typean)
        file_eff = TFile.Open(path)
        if not file_eff:
            self.logger.fatal(make_message_notfound(path))
        path = "%s/feeddown%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        fileouts = TFile.Open(path, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(path))

        response_matrix = feeddown_input_file.Get("response_matrix_nonprompt")

        # input_data is 3d histogram from powheg+pythia prediction that
        # contains z vs jet_pt vs HF pt.
        input_data = self.get_simulated_yields(self.powheg_path_nonprompt, 3, False)
        if not input_data:
            self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_nonprompt))

        # Ensure correct binning: x - shape, y - jet pt, z - pt hadron
        if not equal_binning_lists(input_data, list_x = self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(input_data, list_y = self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")
        if not equal_binning_lists(input_data, list_z = self.var1ranges):
            self.logger.fatal("Error: Incorrect binning in z.")

        fileouts.cd()
        input_data.Write()

        # output_template is the reco jet pt vs z for candidates in the reco
        # min-max region
        output_template = feeddown_input_file.Get("hzvsjetpt_reco")
        # Ensure correct binning: x - shape, y - jet pt
        if not equal_binning_lists(output_template, list_x = self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(output_template, list_y = self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")

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

        path = "%s/sideband_sub%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        sideband_input_data_file = TFile.Open(path)
        if not sideband_input_data_file:
            self.logger.fatal(make_message_notfound(path))
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
        hzvsjetpt_gen_eff.SetTitle("")
        hzvsjetpt_gen_eff.SetXTitle("%s^{gen}" % self.v_varshape_latex)
        hzvsjetpt_gen_eff.SetYTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hzvsjetpt_gen_eff.Draw("text")
        cgen_eff.SaveAs("%s/step2_cgen_kineeff_nonprompt.eps" % (self.d_resultsallpdata))

        # creco_eff is the kinematic efficiency that a candidate in reco limits
        # limits has gen values in the gen limits

        creco_eff = TCanvas('creco_eff_nonprompt ', 'reco efficiency applied to feedown')
        preco_eff = TPad('preco_eff_nonprompt ', 'reco efficiency applied to feedown',
                         0.0, 0.0, 1.0, 1.0)
        setup_pad(preco_eff)
        creco_eff.SetCanvasSize(1900, 1500)
        creco_eff.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt_reco_eff)
        hzvsjetpt_reco_eff.SetTitle("")
        hzvsjetpt_reco_eff.SetXTitle("%s^{rec}" % self.v_varshape_latex)
        hzvsjetpt_reco_eff.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hzvsjetpt_reco_eff.Draw("text")
        creco_eff.SaveAs("%s/step2_creco_kineeff_nonprompt.eps" % (self.d_resultsallpdata))

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
            hz_genvsreco_list[ibin2].SetTitle("")
            hz_genvsreco_list[ibin2].SetXTitle("%s^{gen}" % self.v_varshape_latex)
            hz_genvsreco_list[ibin2].SetYTitle("%s^{rec}" % self.v_varshape_latex)
            hz_genvsreco_list[ibin2].Draw("colz")
            latex = TLatex(0.2, 0.85, '%.2f < %s < %.2f GeV/#it{c}' \
                           % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cz_genvsreco.SaveAs("%s/step2_c_response_z_genvsreco_nonprompt_%s.eps" % \
                                (self.d_resultsallpdata, suffix))

        for ibinshape in range(self.p_nbinshape_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
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
            hjetpt_genvsreco_list[ibinshape].SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
            hjetpt_genvsreco_list[ibinshape].SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
            hjetpt_genvsreco_list[ibinshape].Draw("colz")
            latex = TLatex(0.2, 0.8, "%.2f < %s < %.2f" % \
                (self.lvarshape_binmin_reco[ibinshape], self.v_varshape_latex, self.lvarshape_binmax_reco[ibinshape]))
            draw_latex(latex)
            cjetpt_genvsreco.SaveAs("%s/step_2_c_response_jetpt_genvsreco_nonprompt_%s.eps" % \
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
        hz_genvsreco_full.SetTitle("")
        hz_genvsreco_full.SetXTitle("%s^{gen}" % self.v_varshape_latex)
        hz_genvsreco_full.SetYTitle("%s^{rec}" % self.v_varshape_latex)
        hz_genvsreco_full.Draw("colz")
        cz_genvsreco.SaveAs("%s/step2_c_response_z_genvsreco_full_nonprompt.eps" % \
                            (self.d_resultsallpdata))

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
        hjetpt_genvsreco_full.SetTitle("")
        hjetpt_genvsreco_full.SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hjetpt_genvsreco_full.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hjetpt_genvsreco_full.Draw("colz")
        cjetpt_genvsreco.SaveAs("%s/step_2_c_response_jetpt_genvsreco_full_nonprompt.eps" % \
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
            leg_eff.AddEntry(heff_pr_list[ibin2], "prompt", "P")
            heff_pr_list[ibin2].GetYaxis().SetRangeUser( \
                0.5*min(heff_pr_list[ibin2].GetMinimum(), heff_fd_list[ibin2].GetMinimum()), \
                1.1*max(heff_pr_list[ibin2].GetMaximum(), heff_fd_list[ibin2].GetMaximum()))
            heff_pr_list[ibin2].SetTitle("")
            heff_pr_list[ibin2].SetXTitle("#it{p}_{T, %s} (GeV/#it{c})" % self.p_latexnhadron)
            heff_pr_list[ibin2].SetYTitle("Efficiency #times Acceptance")
            heff_pr_list[ibin2].SetTitleOffset(1.2, "Y")
            heff_pr_list[ibin2].SetTitle("")
            heff_pr_list[ibin2].Draw()
            setup_histogram(heff_fd_list[ibin2], 4, 24)
            leg_eff.AddEntry(heff_fd_list[ibin2], "non-prompt", "P")
            heff_fd_list[ibin2].Draw("same")
            leg_eff.Draw("same")
            #PREL latex = TLatex(0.52, 0.45, "ALICE Preliminary")
            #PREL draw_latex(latex)
            latex2 = TLatex(0.52, 0.4, "PYTHIA 6, pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex2)
            latex3 = TLatex(0.52, 0.35, ("%s #rightarrow %s (and charge conj.)" % (self.p_latexnhadron, self.p_latexndecay)))
            draw_latex(latex3)
            latex4 = TLatex(0.52, 0.3, "in charged jets, anti-#it{k}_{T}, #it{R} = 0.4")
            draw_latex(latex4)
            latex5 = TLatex(0.52, 0.25, "%.0f < %s < %.0f GeV/#it{c}" \
                            % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex5)
            latex6 = TLatex(0.52, 0.2, "#left|#it{#eta}_{jet}#right| < 0.5")
            draw_latex(latex6)
            ceff.SaveAs("%s/step2_ceff_prompt_nonprompt_%s.eps" % (self.d_resultsallpdata, suffix))

        cjetpt_fracdiff = TCanvas('cjetpt_fracdiff ',
                                  'non-prompt jetpt response fractional differences')
        pjetpt_fracdiff = TPad('pjetpt_fracdiff', \
            'non-prompt jetpt response fractional differences', 0.0, 0.001, 1.0, 1.0)
        setup_pad(pjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        pjetpt_fracdiff.SetLogy()
        cjetpt_fracdiff.SetCanvasSize(1900, 1500)
        cjetpt_fracdiff.SetWindowSize(500, 500)
        leg_jetpt_fracdiff = TLegend(.65, .5, .8, .8, "#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            setup_histogram(hjetpt_fracdiff_list[ibin2], ibin2+1)
            leg_jetpt_fracdiff.AddEntry(hjetpt_fracdiff_list[ibin2], \
                "%g-%g" % (self.lvar2_binmin_gen[ibin2], \
                self.lvar2_binmax_gen[ibin2]), "P")
            if ibin2 == 0:
                hjetpt_fracdiff_list[ibin2].SetTitle("")
                hjetpt_fracdiff_list[ibin2].SetXTitle(\
                    "(#it{p}_{T, jet}^{rec} #minus #it{p}_{T, jet}^{gen})/#it{p}_{T, jet}^{gen}")
                hjetpt_fracdiff_list[ibin2].GetYaxis().SetRangeUser(0.001, \
                    hjetpt_fracdiff_list[ibin2].GetMaximum()*3)
            hjetpt_fracdiff_list[ibin2].Draw("same")
        leg_jetpt_fracdiff.Draw("same")
        cjetpt_fracdiff.SaveAs("%s/cjetpt_fracdiff_nonprompt.eps" % (self.d_resultsallpdata))

        for ibinshape in range(self.p_nbinshape_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_list.append(feeddown_input_file.Get("hz_fracdiff_nonprompt" + suffix))

        cz_fracdiff = TCanvas('cz_fracdiff ', 'non-prompt z response fractional differences')
        pz_fracdiff = TPad('pz_fracdiff', 'non-prompt z response fractional differences',
                           0.0, 0.001, 1.0, 1.0)
        setup_pad(pz_fracdiff)
        cz_fracdiff.SetLogy()
        pz_fracdiff.SetLogy()
        cz_fracdiff.SetCanvasSize(1900, 1500)
        cz_fracdiff.SetWindowSize(500, 500)
        leg_z_fracdiff = TLegend(.2, .5, .4, .85, self.v_varshape_latex)
        setup_legend(leg_z_fracdiff)
        for ibinshape in range(self.p_nbinshape_gen):
            setup_histogram(hz_fracdiff_list[ibinshape], ibinshape+1)
            leg_z_fracdiff.AddEntry(hz_fracdiff_list[ibinshape], \
                "%.2f-%.2f" % (self.lvarshape_binmin_gen[ibinshape], \
                self.lvarshape_binmax_gen[ibinshape]), "P")
            if ibinshape == 0:
                hz_fracdiff_list[ibinshape].SetTitle("")
                hz_fracdiff_list[ibinshape].SetXTitle("(%s^{rec} #minus %s^{gen})/%s^{gen}" % (self.v_varshape_latex, self.v_varshape_latex, self.v_varshape_latex))
                hz_fracdiff_list[ibinshape].GetYaxis().SetRangeUser(0.001, \
                    hz_fracdiff_list[ibinshape].GetMaximum()*3)
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
            leg_fd_fold.AddEntry(input_data_scaled_z_list[ibin2], "POWHEG, eff. scaled", "P")
            input_data_scaled_z_list[ibin2].GetYaxis().SetRangeUser(0.0, \
                    input_data_scaled_z_list[ibin2].GetMaximum()*1.5)
            input_data_scaled_z_list[ibin2].SetTitle("")
            input_data_scaled_z_list[ibin2].SetXTitle(self.v_varshape_latex)
            input_data_scaled_z_list[ibin2].SetYTitle("Yield")
            input_data_scaled_z_list[ibin2].Draw()
            setup_histogram(folded_z_list[ibin2], 4)
            leg_fd_fold.AddEntry(folded_z_list[ibin2], "POWHEG, eff. scaled, folded", "P")
            folded_z_list[ibin2].Draw("same")
            leg_fd_fold.Draw("same")
            latex = TLatex(0.4, 0.25, "%.2f < %s < %.2f GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
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
            leg_feeddown.AddEntry(sideband_input_data_z[ibin2], "prompt+non-prompt", "P")
            sideband_input_data_z[ibin2].GetYaxis().SetRangeUser(0.1, \
                    sideband_input_data_z[ibin2].GetMaximum()*3)
            sideband_input_data_z[ibin2].SetTitle("")
            sideband_input_data_z[ibin2].SetXTitle(self.v_varshape_latex)
            sideband_input_data_z[ibin2].SetYTitle("Yield")
            sideband_input_data_z[ibin2].Draw()
            setup_histogram(sideband_input_data_subtracted_z[ibin2], 3)
            leg_feeddown.AddEntry(sideband_input_data_subtracted_z[ibin2],
                                  "subtracted (prompt)", "P")
            sideband_input_data_subtracted_z[ibin2].Draw("same")
            setup_histogram(folded_z_list[ibin2], 4)
            leg_feeddown.AddEntry(folded_z_list[ibin2], "non-prompt powheg", "P")
            folded_z_list[ibin2].Draw("same")
            leg_feeddown.Draw("same")
            if ibin2 != 2:
                latex = TLatex(0.6, 0.3, "%.2f < %s < %.2f GeV/#it{c}" \
                    % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
                latex = TLatex(0.6, 0.3, "%.2f < %s < %.2f GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            else:
                latex = TLatex(0.6, 0.75, "%.2f < %s < %.2f GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
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
            feeddown_fraction.SetTitle("")
            feeddown_fraction.SetXTitle(self.v_varshape_latex)
            feeddown_fraction.SetYTitle("feed-down fraction")
            feeddown_fraction.Draw()
            latex = TLatex(0.6, 0.75, "%.2f < %s < %.2f GeV/#it{c}" % \
                           (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
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
        sideband_input_data_subtracted.SetTitle("")
        sideband_input_data_subtracted.SetXTitle(self.v_varshape_latex)
        sideband_input_data_subtracted.SetYTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
        sideband_input_data_subtracted.Draw("text")
        cfeeddown_output.SaveAs("%s/cfeeddown_output.eps" % (self.d_resultsallpdata))
        print("end of folding")

    def unfolding(self):
        print("unfolding starts")
        path = "%s/unfolding_results%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        fileouts = TFile.Open(path, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(path))

        path = "%s/feeddown%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        unfolding_input_data_file = TFile.Open(path)
        if not unfolding_input_data_file:
            self.logger.fatal(make_message_notfound(path))
        input_data = unfolding_input_data_file.Get("sideband_input_data_subtracted")
        if not input_data:
            self.logger.fatal(make_message_notfound("sideband_input_data_subtracted", path))

        unfolding_input_file = TFile.Open(self.n_fileff)
        if not unfolding_input_file:
            self.logger.fatal(make_message_notfound(self.n_fileff))
        response_matrix = unfolding_input_file.Get("response_matrix")
        if not response_matrix:
            self.logger.fatal(make_message_notfound("response_matrix", self.n_fileff))
        hzvsjetpt_reco_nocuts = unfolding_input_file.Get("hzvsjetpt_reco_nocuts")
        if not hzvsjetpt_reco_nocuts:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco_nocuts", self.n_fileff))
        hzvsjetpt_reco_eff = unfolding_input_file.Get("hzvsjetpt_reco_cuts")
        if not hzvsjetpt_reco_eff:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco_cuts", self.n_fileff))

        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)
        input_data.Multiply(hzvsjetpt_reco_eff)

        input_mc_gen = unfolding_input_file.Get("hzvsjetpt_gen_unmatched")
        if not input_mc_gen:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen_unmatched", self.n_fileff))
        mc_reco_matched = unfolding_input_file.Get("hzvsjetpt_reco")
        if not mc_reco_matched:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco", self.n_fileff))
        mc_gen_matched = unfolding_input_file.Get("hzvsjetpt_gen")
        if not mc_gen_matched:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen", self.n_fileff))

        input_data_z=[]
        input_mc_gen_z=[]
        mc_reco_matched_z=[]
        mc_gen_matched_z=[]
        mc_reco_gen_matched_z_ratio=[]
        hjetpt_fracdiff_list=[]
        hz_fracdiff_list=[]
        kinematic_eff=[]
        hz_gen_nocuts=[]

        hz_genvsreco_list=[]
        hjetpt_genvsreco_list=[]

        input_data_jetpt=input_data.ProjectionY("input_data_jetpt",1, self.p_nbinshape_reco,"e")

        input_powheg = self.get_simulated_yields(self.powheg_path_prompt, 2, True)
        if not input_powheg:
            self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_prompt))
        input_powheg_xsection = input_powheg.Clone(input_powheg.GetName() + "_xsec")

        # Ensure correct binning: x - shape, y - jet pt
        if not equal_binning_lists(input_powheg, list_x = self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(input_powheg, list_y = self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")
        # Ensure correct binning: x - shape, y - jet pt
        if not equal_binning_lists(input_powheg_xsection, list_x = self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(input_powheg_xsection, list_y = self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")

        input_powheg_sys=[]
        input_powheg_xsection_sys=[]
        for i_powheg in range(len(self.powheg_prompt_variations)):
            path = "%s%s.root" % (self.powheg_prompt_variations_path, self.powheg_prompt_variations[i_powheg])
            input_powheg_sys_i = self.get_simulated_yields(path, 2, True)
            if not input_powheg_sys_i:
                self.logger.fatal(make_message_notfound("simulated yields", path))
            input_powheg_sys_i.SetName("fh2_prompt_%s_%d" % (self.v_varshape_binning, i_powheg))
            input_powheg_sys.append(input_powheg_sys_i)
            input_powheg_xsection_sys_i = input_powheg_sys_i.Clone(input_powheg_sys_i.GetName() + "_xsec")
            input_powheg_xsection_sys.append(input_powheg_xsection_sys_i)
        input_powheg_z=[]
        input_powheg_xsection_z=[]
        input_powheg_sys_z=[]
        input_powheg_xsection_sys_z=[]
        tg_powheg=[]
        tg_powheg_xsection=[]





        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            input_data_z.append(input_data.ProjectionX("input_data_z"+suffix,ibin2+1,ibin2+1,"e"))
            mc_reco_matched_z.append(mc_reco_matched.ProjectionX("mc_reco_matched_z"+suffix,ibin2+1,ibin2+1,"e"))
            mc_reco_matched_z[ibin2].Scale(1.0/mc_reco_matched_z[ibin2].Integral(1,-1))
            mc_gen_matched_z.append(mc_gen_matched.ProjectionX("mc_det_matched_z"+suffix,mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]),mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]),"e"))
            mc_gen_matched_z[ibin2].Scale(1.0/mc_gen_matched_z[ibin2].Integral(1,-1))
            mc_reco_gen_matched_z_ratio.append(mc_reco_matched_z[ibin2].Clone("input_mc_reco_gen_matched_z_ratio"+suffix))
            mc_reco_gen_matched_z_ratio[ibin2].Divide(mc_gen_matched_z[ibin2])

            c_mc_reco_gen_matched_z_ratio = TCanvas('c_mc_reco_gen_matched_z_ratio '+suffix, 'Reco/Gen Ratio')
            p_mc_reco_gen_matched_z_ratio = TPad('p_mc_reco_gen_matched_z_ratio'+suffix, 'c_mc_reco_gen_matched_z_ratio'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(p_mc_reco_gen_matched_z_ratio)
            c_mc_reco_gen_matched_z_ratio.SetCanvasSize(1900, 1500)
            c_mc_reco_gen_matched_z_ratio.SetWindowSize(500, 500)
            setup_histogram(mc_reco_gen_matched_z_ratio[ibin2])
            mc_reco_gen_matched_z_ratio[ibin2].SetTitle("")
            mc_reco_gen_matched_z_ratio[ibin2].SetXTitle(self.v_varshape_latex)
            mc_reco_gen_matched_z_ratio[ibin2].SetYTitle("reconstructed/generated")
            mc_reco_gen_matched_z_ratio[ibin2].Draw("same")
            latex = TLatex(0.2,0.85,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_mc_reco_gen_matched_z_ratio.SaveAs("%s/mc_reco_gen_matched_z_ratio_%s.eps" % (self.d_resultsallpdata, suffix))

            c_mc_reco_gen_matched_z = TCanvas('c_mc_reco_gen_matched_z '+suffix, 'Reco vs Gen')
            p_mc_reco_gen_matched_z = TPad('p_mc_reco_gen_matched_z'+suffix, 'Reco vs Gen'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(p_mc_reco_gen_matched_z)
            c_mc_reco_gen_matched_z.SetCanvasSize(1900, 1500)
            c_mc_reco_gen_matched_z.SetWindowSize(500, 500)
            leg_mc_reco_gen_matched_z = TLegend(.75, .7, .9, .85, "")
            setup_legend(leg_mc_reco_gen_matched_z)
            setup_histogram(mc_reco_matched_z[ibin2],2)
            leg_mc_reco_gen_matched_z.AddEntry(mc_reco_matched_z[ibin2],"reco","P")
            mc_reco_matched_z[ibin2].SetTitle("")
            mc_reco_matched_z[ibin2].SetXTitle(self.v_varshape_latex)
            mc_reco_matched_z[ibin2].SetYTitle("normalised yield")
            mc_reco_matched_z[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
            mc_reco_matched_z[ibin2].GetYaxis().SetRangeUser(0.0,mc_reco_matched_z[ibin2].GetMaximum()*1.5)
            mc_reco_matched_z[ibin2].Draw()
            setup_histogram(mc_gen_matched_z[ibin2],4)
            leg_mc_reco_gen_matched_z.AddEntry(mc_gen_matched_z[ibin2],"gen","P")
            mc_gen_matched_z[ibin2].Draw("same")
            leg_mc_reco_gen_matched_z.Draw("same")
            latex = TLatex(0.2,0.85,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_mc_reco_gen_matched_z.SaveAs("%s/mc_reco_gen_matched_z_%s.eps" % (self.d_resultsallpdata, suffix))

            hz_genvsreco_list.append(unfolding_input_file.Get("hz_genvsreco"+suffix))

            cz_genvsreco = TCanvas('cz_genvsreco_'+suffix, 'response matrix 2D projection')
            pz_genvsreco = TPad('pz_genvsreco'+suffix, 'response matrix 2D projection'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pz_genvsreco)
            cz_genvsreco.SetLogz()
            pz_genvsreco.SetLogz()
            cz_genvsreco.SetCanvasSize(1900, 1500)
            cz_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hz_genvsreco_list[ibin2])
            hz_genvsreco_list[ibin2].SetTitle("")
            hz_genvsreco_list[ibin2].SetXTitle("%s^{gen}" % self.v_varshape_latex)
            hz_genvsreco_list[ibin2].SetYTitle("%s^{rec}" % self.v_varshape_latex)
            hz_genvsreco_list[ibin2].Draw("colz")
            latex = TLatex(0.2,0.85,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cz_genvsreco.SaveAs("%s/cz_genvsreco_%s.eps" % (self.d_resultsallpdata, suffix))

        for ibinshape in range(self.p_nbinshape_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco_list.append(unfolding_input_file.Get("hjetpt_genvsreco"+suffix))

            cjetpt_genvsreco = TCanvas('cjetpt_genvsreco'+suffix, 'response matrix 2D projection'+suffix)
            pjetpt_genvsreco = TPad('pjetpt_genvsreco'+suffix, 'response matrix 2D projection'+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pjetpt_genvsreco)
            cjetpt_genvsreco.SetLogz()
            pjetpt_genvsreco.SetLogz()
            cjetpt_genvsreco.SetCanvasSize(1900, 1500)
            cjetpt_genvsreco.SetWindowSize(500, 500)
            setup_histogram(hjetpt_genvsreco_list[ibinshape])
            hjetpt_genvsreco_list[ibinshape].SetTitle("")
            hjetpt_genvsreco_list[ibinshape].SetXTitle("%s^{gen}" % self.v_varshape_latex)
            hjetpt_genvsreco_list[ibinshape].SetYTitle("%s^{rec}" % self.v_varshape_latex)
            hjetpt_genvsreco_list[ibinshape].Draw("colz")
            latex = TLatex(0.2,0.85,'%.2f < %s < %.2f' % (self.lvarshape_binmin_reco[ibinshape], self.v_varshape_latex, self.lvarshape_binmax_reco[ibinshape]))
            draw_latex(latex)
            cjetpt_genvsreco.SaveAs("%s/cjetpt_genvsreco_%s.eps" % (self.d_resultsallpdata, suffix))

        hz_genvsreco_full=unfolding_input_file.Get("hz_genvsreco_full")
        hjetpt_genvsreco_full=unfolding_input_file.Get("hjetpt_genvsreco_full")

        cz_genvsreco_full = TCanvas('cz_genvsreco_full', 'response matrix 2D projection')
        pz_genvsreco_full = TPad('pz_genvsreco_full', 'response matrix 2D projection',0.0,0.001,1.0,1.0)
        setup_pad(pz_genvsreco_full)
        cz_genvsreco_full.SetLogz()
        pz_genvsreco_full.SetLogz()
        cz_genvsreco_full.SetCanvasSize(1900, 1500)
        cz_genvsreco_full.SetWindowSize(500, 500)
        setup_histogram(hz_genvsreco_full)
        hz_genvsreco_full.SetTitle("")
        hz_genvsreco_full.SetXTitle("%s^{gen}" % self.v_varshape_latex)
        hz_genvsreco_full.SetYTitle("%s^{rec}" % self.v_varshape_latex)
        hz_genvsreco_full.Draw("colz")
        cz_genvsreco_full.SaveAs("%s/cz_genvsreco_full.eps" % (self.d_resultsallpdata))

        cjetpt_genvsreco_full = TCanvas('cjetpt_genvsreco_full', 'response matrix 2D projection')
        pjetpt_genvsreco_full = TPad('pjetpt_genvsreco_full', 'response matrix 2D projection',0.0,0.001,1.0,1.0)
        setup_pad(pjetpt_genvsreco_full)
        cjetpt_genvsreco_full.SetLogz()
        pjetpt_genvsreco_full.SetLogz()
        cjetpt_genvsreco_full.SetCanvasSize(1900, 1500)
        cjetpt_genvsreco_full.SetWindowSize(500, 500)
        setup_histogram(hjetpt_genvsreco_full)
        hjetpt_genvsreco_full.SetTitle("")
        hjetpt_genvsreco_full.SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hjetpt_genvsreco_full.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hjetpt_genvsreco_full.Draw("colz")
        cjetpt_genvsreco_full.SaveAs("%s/cjetpt_genvsreco_full.eps" % (self.d_resultsallpdata))

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff_list.append(unfolding_input_file.Get("hjetpt_fracdiff_prompt"+suffix))
            kinematic_eff.append(unfolding_input_file.Get("hz_gen_cuts"+suffix))
            hz_gen_nocuts.append(unfolding_input_file.Get("hz_gen_nocuts"+suffix))
            kinematic_eff[ibin2].Divide(hz_gen_nocuts[ibin2])
            ckinematic_eff = TCanvas("ckinematic_eff " + suffix, "Kinematic Eff" + suffix)
            pkinematic_eff = TPad('pkinematic_eff' + suffix, "Kinematic Eff" + suffix,0.0,0.001,1.0,1.0)
            setup_pad(pkinematic_eff)
            ckinematic_eff.SetCanvasSize(1900, 1500)
            ckinematic_eff.SetWindowSize(500, 500)
            setup_histogram(kinematic_eff[ibin2],4)
            kinematic_eff[ibin2].SetTitle("")
            kinematic_eff[ibin2].SetXTitle(self.v_varshape_latex)
            kinematic_eff[ibin2].SetYTitle("kinematic efficiency")
            kinematic_eff[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
            kinematic_eff[ibin2].Draw()
            latex = TLatex(0.6,0.25,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            ckinematic_eff.SaveAs("%s/cgen_kineeff_%s.eps" % (self.d_resultsallpdata, suffix))

            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_mc_gen_z[ibin2].Scale(1.0/input_mc_gen_z[ibin2].Integral(input_mc_gen_z[ibin2].FindBin(self.lvarshape_binmin_reco[0]),input_mc_gen_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])),"width")
            input_powheg_z.append(input_powheg.ProjectionX("input_powheg_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_z[ibin2].Scale(1.0/input_powheg_z[ibin2].Integral(input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[0]),input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])),"width")
            input_powheg_xsection_z.append(input_powheg_xsection.ProjectionX("input_powheg_xsection_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_xsection_z[ibin2].Scale(1.0,"width")
            input_powheg_sys_z_iter=[]
            input_powheg_xsection_sys_z_iter=[]
            for i_powheg in range(len(self.powheg_prompt_variations)):
                input_powheg_sys_z_iter.append(input_powheg_sys[i_powheg].ProjectionX("input_powheg_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix,ibin2+1,ibin2+1,"e"))
                input_powheg_sys_z_iter[i_powheg].Scale(1.0/input_powheg_sys_z_iter[i_powheg].Integral(input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[0]),input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[-1])),"width")
                input_powheg_xsection_sys_z_iter.append(input_powheg_xsection_sys[i_powheg].ProjectionX("input_powheg_xsection_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix,ibin2+1,ibin2+1,"e"))
                input_powheg_xsection_sys_z_iter[i_powheg].Scale(1.0,"width")
            input_powheg_sys_z.append(input_powheg_sys_z_iter)
            input_powheg_xsection_sys_z.append(input_powheg_xsection_sys_z_iter)
            tg_powheg.append(tg_sys(input_powheg_z[ibin2], input_powheg_sys_z[ibin2]))
            tg_powheg_xsection.append(tg_sys(input_powheg_xsection_z[ibin2], input_powheg_xsection_sys_z[ibin2]))


        kinematic_eff_jetpt = unfolding_input_file.Get("hjetpt_gen_cuts")
        hjetpt_gen_nocuts=unfolding_input_file.Get("hjetpt_gen_nocuts")
        kinematic_eff_jetpt.Divide(hjetpt_gen_nocuts)
        ckinematic_eff_jetpt = TCanvas("ckinematic_eff_jetpt", "Kinematic Eff_jetpt")
        pkinematic_eff_jetpt = TPad('pkinematic_eff_jetpt', "Kinematic Eff_jetpt",0.0,0.001,1.0,1.0)
        setup_pad(pkinematic_eff_jetpt)
        ckinematic_eff_jetpt.SetCanvasSize(1900, 1500)
        ckinematic_eff_jetpt.SetWindowSize(500, 500)
        setup_histogram(kinematic_eff_jetpt)
        kinematic_eff_jetpt.SetTitle("")
        kinematic_eff_jetpt.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
        kinematic_eff_jetpt.SetYTitle("kinematic efficiency")
        kinematic_eff_jetpt.GetXaxis().SetRangeUser(self.lvar2_binmin_reco[0]+0.01, self.lvar2_binmax_reco[-1]-0.001)
        kinematic_eff_jetpt.Draw()
        latex = TLatex(0.6,0.25,'%.2f < %s < %.2f' % (self.lvarshape_binmin_reco[0], self.v_varshape_latex, self.lvarshape_binmax_reco[-1]))
        draw_latex(latex)
        ckinematic_eff_jetpt.SaveAs("%s/cgen_kineeff_jetpt.eps" % (self.d_resultsallpdata))

        for ibinshape in range(self.p_nbinshape_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_list.append(unfolding_input_file.Get("hz_fracdiff_prompt"+suffix))

        cjetpt_fracdiff = TCanvas('cjetpt_fracdiff', 'prompt jetpt response fractional differences')
        pjetpt_fracdiff = TPad('pjetpt_fracdiff', "prompt jetpt response fractional differences",0.0,0.001,1.0,1.0)
        setup_pad(pjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        pjetpt_fracdiff.SetLogy()
        cjetpt_fracdiff.SetCanvasSize(1900, 1500)
        cjetpt_fracdiff.SetWindowSize(500, 500)
        leg_jetpt_fracdiff = TLegend(.65, .6, .8, .8, "#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            setup_histogram(hjetpt_fracdiff_list[ibin2],ibin2+1)
            leg_jetpt_fracdiff.AddEntry(hjetpt_fracdiff_list[ibin2],"%g-%g" % (self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2]),"P")
            if ibin2 == 0:
                hjetpt_fracdiff_list[ibin2].GetYaxis().SetRangeUser(0.001,hjetpt_fracdiff_list[ibin2].GetMaximum()*2)
                hjetpt_fracdiff_list[ibin2].SetTitle("")
                hjetpt_fracdiff_list[ibin2].SetXTitle("(#it{p}_{T, jet}^{rec} #minus #it{p}_{T, jet}^{gen})/#it{p}_{T, jet}^{gen}")
            hjetpt_fracdiff_list[ibin2].Draw("same")
        leg_jetpt_fracdiff.Draw("same")
        cjetpt_fracdiff.SaveAs("%s/cjetpt_fracdiff_prompt.eps" % (self.d_resultsallpdata))

        creco_eff = TCanvas('creco_eff ', 'reco efficiency applied to input data')
        preco_eff = TPad('preco_eff', "reco efficiency applied to input data",0.0,0.001,1.0,1.0)
        setup_pad(preco_eff)
        creco_eff.SetCanvasSize(1900, 1500)
        creco_eff.SetWindowSize(500, 500)
        setup_histogram(hzvsjetpt_reco_eff)
        hzvsjetpt_reco_eff.SetTitle("")
        hzvsjetpt_reco_eff.SetXTitle(self.v_varshape_latex)
        hzvsjetpt_reco_eff.SetYTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
        hzvsjetpt_reco_eff.Draw("text")
        creco_eff.SaveAs("%s/creco_kineeff.eps" % (self.d_resultsallpdata))


        cz_fracdiff = TCanvas('cz_fracdiff', 'prompt z response fractional differences')
        pz_fracdiff = TPad('pz_fracdiff', "prompt z response fractional differences",0.0,0.001,1.0,1.0)
        setup_pad(pz_fracdiff)
        cz_fracdiff.SetLogy()
        pz_fracdiff.SetLogy()
        cz_fracdiff.SetCanvasSize(1900, 1500)
        cz_fracdiff.SetWindowSize(500, 500)
        leg_z_fracdiff = TLegend(.2, .5, .4, .85, self.v_varshape_latex)
        setup_legend(leg_z_fracdiff)
        for ibinshape in range(self.p_nbinshape_gen):
            setup_histogram(hz_fracdiff_list[ibinshape],ibinshape+1)
            leg_z_fracdiff.AddEntry(hz_fracdiff_list[ibinshape],"%.2f-%.2f" % (self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape]),"P")
            if ibinshape == 0:
                hz_fracdiff_list[ibinshape].GetYaxis().SetRangeUser(0.001,hz_fracdiff_list[ibinshape].GetMaximum()*2)
                hz_fracdiff_list[ibinshape].SetTitle("")
                hz_fracdiff_list[ibinshape].SetXTitle("(%s^{rec} #minus %s^{gen})/%s^{gen}" % (self.v_varshape_latex, self.v_varshape_latex, self.v_varshape_latex))
            hz_fracdiff_list[ibinshape].Draw("same")
        leg_z_fracdiff.Draw("same")
        cz_fracdiff.SaveAs("%s/cz_fracdiff_prompt.eps" % (self.d_resultsallpdata))

        fileouts.cd()
        unfolded_z_scaled_list=[]
        unfolded_z_xsection_list=[]
        unfolded_jetpt_scaled_list=[]
        refolding_test_list=[]
        refolding_test_jetpt_list=[]
        i_iter_choice = self.choice_iter_unfolding - 1 # list index of the chosen iteration
        for i in range(self.niter_unfolding):
            unfolded_z_scaled_list_iter=[]
            unfolded_z_xsection_list_iter=[]
            refolding_test_list_iter=[]
            unfolding_object = RooUnfoldBayes(response_matrix, input_data, i+1)
            unfolded_zvsjetpt = unfolding_object.Hreco(2)

            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                unfolded_z = unfolded_zvsjetpt.ProjectionX("unfolded_z_"+suffix,ibin2+1,ibin2+1,"e")
                unfolded_z.Sumw2()
                unfolded_z_scaled = unfolded_z.Clone("unfolded_z_scaled_%d_%s" % (i+1, suffix))
                unfolded_z_scaled.Divide(kinematic_eff[ibin2])
                unfolded_z_xsection = unfolded_z_scaled.Clone("unfolded_z_xsection_%d_%s" % (i+1, suffix))
                unfolded_z_xsection.Scale((self.xsection_inel)/(self.p_nevents*self.branching_ratio),"width")
                unfolded_z_scaled.Scale(1.0/unfolded_z_scaled.Integral(unfolded_z_scaled.FindBin(self.lvarshape_binmin_reco[0]),unfolded_z_scaled.FindBin(self.lvarshape_binmin_reco[-1])),"width")
                unfolded_z_scaled.Write("unfolded_z_%d_%s" % (i+1, suffix))
                unfolded_z_xsection.Write("unfolded_z_xsection_%d_%s" % (i+1, suffix))
                unfolded_z_scaled_list_iter.append(unfolded_z_scaled)
                unfolded_z_xsection_list_iter.append(unfolded_z_xsection)
                cunfolded_z = TCanvas('cunfolded_z'+suffix, '1D output of unfolding'+suffix)
                punfolded_z = TPad('punfolded_z'+suffix, "1D output of unfolding"+suffix,0.0,0.001,1.0,1.0)
                setup_pad(punfolded_z)
                cunfolded_z.SetCanvasSize(1900, 1500)
                cunfolded_z.SetWindowSize(500, 500)
                setup_histogram(unfolded_z_scaled,4)
                unfolded_z_scaled.GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
                unfolded_z_scaled.SetTitle("")
                unfolded_z_scaled.SetXTitle(self.v_varshape_latex)
                unfolded_z_scaled.SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
                unfolded_z_scaled.Draw()
                latex = TLatex(0.6,0.25,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
                draw_latex(latex2)
                cunfolded_z.SaveAs("%s/cunfolded_z_%d_%s.eps" % (self.d_resultsallpdata, i+1, suffix))
                # Save the selected iteration under a special name.
                if i == i_iter_choice:
                    unfolded_z_scaled.Write("unfolded_z_sel_%s" % suffix)
                    unfolded_z_xsection.Write("unfolded_z_xsection_sel_%s" % suffix)
                    cunfolded_z.SaveAs("%s/cunfolded_z_sel_%s.eps" % (self.d_resultsallpdata, suffix))


            unfolded_jetpt = unfolded_zvsjetpt.ProjectionY("unfolded_jetpt",1, self.p_nbinshape_gen,"e")
            unfolded_jetpt.Sumw2()
            unfolded_jetpt_scaled = unfolded_jetpt.Clone("unfolded_jetpt_scaled_%d" % (i+1))
            unfolded_jetpt_scaled.Divide(kinematic_eff_jetpt)
            unfolded_jetpt_scaled.Scale(1.0/unfolded_jetpt_scaled.Integral(unfolded_jetpt_scaled.FindBin(self.lvar2_binmin_reco[0]),unfolded_jetpt_scaled.FindBin(self.lvar2_binmin_reco[-1])),"width")
            unfolded_jetpt_scaled.Write("unfolded_jetpt_%d" % (i+1))
            unfolded_jetpt_scaled_list.append(unfolded_jetpt_scaled)
            cunfolded_jetpt = TCanvas('cunfolded_jetpt', '1D output of unfolding')
            punfolded_jetpt = TPad('punfolded_jetpt', "1D output of unfolding",0.0,0.001,1.0,1.0)
            setup_pad(punfolded_jetpt)
            cunfolded_jetpt.SetCanvasSize(1900, 1500)
            cunfolded_jetpt.SetWindowSize(500, 500)
            setup_histogram(unfolded_jetpt_scaled,4)
            unfolded_jetpt_scaled.GetXaxis().SetRangeUser(self.lvar2_binmin_reco[0]+0.01, self.lvar2_binmax_reco[-1]-0.001)
            unfolded_jetpt_scaled.SetTitle("")
            unfolded_jetpt_scaled.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
            unfolded_jetpt_scaled.SetYTitle("1/#it{N}_{jets} d#it{N}/d%s (#it{c}/GeV)" % self.p_latexbin2var)
            unfolded_jetpt_scaled.Draw()
            latex = TLatex(0.6,0.85,'%.2f < %s < %.2f' % (self.lvarshape_binmin_reco[0], self.v_varshape_latex, self.lvarshape_binmax_reco[-1]))
            draw_latex(latex)
            latex2 = TLatex(0.6,0.8,'iteration %d' % (i+1))
            draw_latex(latex2)
            cunfolded_jetpt.SaveAs("%s/cunfolded_jetpt_%d.eps" % (self.d_resultsallpdata, i+1))

            unfolded_z_scaled_list.append(unfolded_z_scaled_list_iter)
            unfolded_z_xsection_list.append(unfolded_z_xsection_list_iter)
            refolded = folding(unfolded_zvsjetpt, response_matrix, input_data)
            refolded.Sumw2()

            for ibin2 in range(self.p_nbin2_reco):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                refolded_z=refolded.ProjectionX("refolded_z",ibin2+1,ibin2+1,"e")
                refolding_test = input_data_z[ibin2].Clone("refolding_test_%d_%s" % (i+1, suffix))
                refolding_test.Divide(refolded_z)
                refolding_test_list_iter.append(refolding_test)
                cfolded_z = TCanvas('cfolded_z '+suffix, '1D output of folding'+suffix)
                pfolded_z = TPad('pfolded_z'+suffix, "1D output of ufolding"+suffix,0.0,0.001,1.0,1.0)
                setup_pad(pfolded_z)
                cfolded_z.SetCanvasSize(1900, 1500)
                cfolded_z.SetWindowSize(500, 500)
                setup_histogram(refolding_test,4)
                refolding_test.GetYaxis().SetRangeUser(0.5,1.5)
                refolding_test.SetTitle("")
                refolding_test.SetXTitle(self.v_varshape_latex)
                refolding_test.SetYTitle("refolding test")
                refolding_test.Draw()
                latex = TLatex(0.6,0.25,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
                draw_latex(latex2)
                cfolded_z.SaveAs("%s/cfolded_z_%d_%s.eps" % (self.d_resultsallpdata, i+1, suffix))

            refolded_jetpt=refolded.ProjectionY("refolded_jetpt",1, self.p_nbinshape_gen,"e")
            refolding_test_jetpt = input_data_jetpt.Clone("refolding_test_%d" % (i+1))
            refolding_test_jetpt.Divide(refolded_jetpt)
            refolding_test_jetpt_list.append(refolding_test_jetpt)
            cfolded_jetpt = TCanvas('cfolded_jetpt ' '1D output of folding')
            pfolded_jetpt = TPad('pfolded_jetpt', "1D output of folding",0.0,0.001,1.0,1.0)
            setup_pad(pfolded_jetpt)
            cfolded_jetpt.SetCanvasSize(1900, 1500)
            cfolded_jetpt.SetWindowSize(500, 500)
            setup_histogram(refolding_test_jetpt,4)
            refolding_test_jetpt.GetYaxis().SetRangeUser(0.5,1.5)
            refolding_test_jetpt.SetTitle("")
            refolding_test_jetpt.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
            refolding_test_jetpt.SetYTitle("refolding test")
            refolding_test_jetpt.Draw()
            latex = TLatex(0.2,0.25,'%.2f < %s < %.2f' % (self.lvarshape_binmin_gen[0], self.v_varshape_latex, self.lvarshape_binmax_gen[-1]))
            draw_latex(latex)
            latex2 = TLatex(0.2,0.2,'iteration %d' % (i+1))
            draw_latex(latex2)
            cfolded_jetpt.SaveAs("%s/cfolded_jetpt_%d.eps" % (self.d_resultsallpdata, i+1))

            refolding_test_list.append(refolding_test_list_iter)

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cconvergence_z = TCanvas('cconvergence_z ' + suffix, '1D output of convergence')
            pconvergence_z = TPad('pconvergence_z', "1D output of convergence",0.0,0.001,1.0,1.0)
            setup_pad(pconvergence_z)
            cconvergence_z.SetCanvasSize(1900, 1500)
            cconvergence_z.SetWindowSize(500, 500)
            leg_z = TLegend(.7, .45, .85, .85, "iterations")
            setup_legend(leg_z)
            for i in range(self.niter_unfolding):
                setup_histogram(unfolded_z_scaled_list[i][ibin2],i+1)
                leg_z.AddEntry(unfolded_z_scaled_list[i][ibin2],("%d" % (i+1)),"P")
                if i == 0:
                    unfolded_z_scaled_list[i][ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
                    unfolded_z_scaled_list[i][ibin2].GetYaxis().SetRangeUser(0,unfolded_z_scaled_list[i][ibin2].GetMaximum()*2.0)
                    unfolded_z_scaled_list[i][ibin2].SetTitle("")
                    unfolded_z_scaled_list[i][ibin2].SetXTitle(self.v_varshape_latex)
                    unfolded_z_scaled_list[i][ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
                unfolded_z_scaled_list[i][ibin2].Draw("same")
                leg_z.Draw("same")
                latex = TLatex(0.6,0.2,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                cconvergence_z.SaveAs("%s/convergence_z_%s.eps" % (self.d_resultsallpdata, suffix))

            cinput_mc_gen_z = TCanvas('cinput_mc_gen_z '+suffix, '1D gen pythia z')
            pinput_mc_gen_z = TPad('pinput_mc_gen_z'+suffix, "1D gen pythia z"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pinput_mc_gen_z)
            cinput_mc_gen_z.SetCanvasSize(1900, 1500)
            cinput_mc_gen_z.SetWindowSize(500, 500)
            leg_input_mc_gen_z = TLegend(.2, .73, .45, .88, "")
            setup_legend(leg_input_mc_gen_z)
            setup_histogram(input_mc_gen_z[ibin2],4)
            leg_input_mc_gen_z.AddEntry(input_mc_gen_z[ibin2], "PYTHIA 6", "P")
            input_mc_gen_z[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
            input_mc_gen_z[ibin2].GetYaxis().SetRangeUser(0.0,input_mc_gen_z[ibin2].GetMaximum()*2)
            input_mc_gen_z[ibin2].SetTitle("")
            input_mc_gen_z[ibin2].SetXTitle(self.v_varshape_latex)
            input_mc_gen_z[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_mc_gen_z[ibin2].Draw()
            setup_histogram(unfolded_z_scaled_list[i_iter_choice][ibin2],2)
            leg_input_mc_gen_z.AddEntry(unfolded_z_scaled_list[i_iter_choice][ibin2], "unfolded ALICE data", "P")
            unfolded_z_scaled_list[i_iter_choice][ibin2].Draw("same")
            setup_histogram(input_powheg_z[ibin2],3)
            leg_input_mc_gen_z.AddEntry(input_powheg_z[ibin2], "POWHEG + PYTHIA 6", "P")
            input_powheg_z[ibin2].Draw("same")
            setup_tgraph(tg_powheg[ibin2],30,0.3)
            tg_powheg[ibin2].Draw("5")
            leg_input_mc_gen_z.Draw("same")
            latex = TLatex(0.6,0.2,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            cinput_mc_gen_z.SaveAs("%s/cinput_mc_gen_z_%s.eps" % (self.d_resultsallpdata, suffix))
            cinput_mc_gen_z.SaveAs("%s/cinput_mc_gen_z_%s.pdf" % (self.d_resultsallpdata, suffix))


            cinput_mc_gen_z_xsection = TCanvas('cinput_mc_gen_z_xsection '+suffix, '1D gen pythia z xsection')
            pinput_mc_gen_z_xsection = TPad('pinput_mc_gen_z_xsection'+suffix, "1D gen pythia z xsection"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pinput_mc_gen_z_xsection)
            cinput_mc_gen_z_xsection.SetCanvasSize(1900, 1500)
            cinput_mc_gen_z_xsection.SetWindowSize(500, 500)
            leg_input_mc_gen_z_xsection = TLegend(.2, .73, .45, .88, "")
            setup_legend(leg_input_mc_gen_z_xsection)
            setup_histogram(unfolded_z_xsection_list[i_iter_choice][ibin2],4)
            leg_input_mc_gen_z_xsection.AddEntry(unfolded_z_xsection_list[i_iter_choice][ibin2], "unfolded ALICE data", "P")
            unfolded_z_xsection_list[i_iter_choice][ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
            unfolded_z_xsection_list[i_iter_choice][ibin2].GetYaxis().SetRangeUser(0.0,unfolded_z_xsection_list[i_iter_choice][ibin2].GetMaximum()*2)
            unfolded_z_xsection_list[i_iter_choice][ibin2].SetTitle("")
            unfolded_z_xsection_list[i_iter_choice][ibin2].SetXTitle(self.v_varshape_latex)
            unfolded_z_xsection_list[i_iter_choice][ibin2].SetYTitle("d#it{#sigma}/d%s (mb)" % self.v_varshape_latex)
            unfolded_z_xsection_list[i_iter_choice][ibin2].Draw()
            setup_histogram(input_powheg_xsection_z[ibin2],3)
            leg_input_mc_gen_z_xsection.AddEntry(input_powheg_xsection_z[ibin2], "POWHEG + PYTHIA 6", "P")
            input_powheg_xsection_z[ibin2].Draw("same")
            setup_tgraph(tg_powheg_xsection[ibin2],30,0.3)
            tg_powheg_xsection[ibin2].Draw("5")
            latex = TLatex(0.6,0.2,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_input_mc_gen_z_xsection.Draw("same")
            cinput_mc_gen_z_xsection.SaveAs("%s/cinput_mc_gen_z_xsection_%s.eps" % (self.d_resultsallpdata, suffix))
            cinput_mc_gen_z_xsection.SaveAs("%s/cinput_mc_gen_z_xsection_%s.pdf" % (self.d_resultsallpdata, suffix))

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            cconvergence_refolding_z = TCanvas('cconvergence_refolding_z '+suffix, '1D output of refolding convergence'+suffix)
            pconvergence_refolding_z = TPad('pconvergence_refolding_z'+suffix, "1D output of refolding convergence"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pconvergence_refolding_z)
            cconvergence_refolding_z.SetCanvasSize(1900, 1500)
            cconvergence_refolding_z.SetWindowSize(500, 500)
            leg_refolding_z = TLegend(.7, .5, .85, .9, "iterations")
            setup_legend(leg_refolding_z)
            for i in range(self.niter_unfolding):
                setup_histogram(refolding_test_list[i][ibin2],i+1)
                leg_refolding_z.AddEntry(refolding_test_list[i][ibin2],("%d" % (i+1)),"P")
                refolding_test_list[i][ibin2].Draw("same")
                if i == 0:
                    refolding_test_list[i][ibin2].SetTitle("")
                    refolding_test_list[i][ibin2].SetXTitle(self.v_varshape_latex)
                    refolding_test_list[i][ibin2].SetYTitle("refolding test")
                    refolding_test_list[i][ibin2].GetYaxis().SetRangeUser(0.5,2.0)
            leg_refolding_z.Draw("same")
            latex = TLatex(0.6,0.2,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cconvergence_refolding_z.SaveAs("%s/convergence_refolding_z_%s.eps" % (self.d_resultsallpdata, suffix))


            input_data_z_scaled=input_data_z[ibin2].Clone("input_data_z_scaled_%s" % suffix)
            input_data_z_scaled.Scale(1.0/input_data_z_scaled.Integral(1,-1),"width")
            cunfolded_not_z = TCanvas('cunfolded_not_z '+suffix, 'Unfolded vs not Unfolded'+suffix)
            punfolded_not_z = TPad('punfolded_not_z'+suffix, "Unfolded vs not Unfolded"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(punfolded_not_z)
            cunfolded_not_z.SetCanvasSize(1900, 1500)
            cunfolded_not_z.SetWindowSize(500, 500)
            leg_cunfolded_not_z = TLegend(.15, .75, .35, .9, "")
            setup_legend(leg_cunfolded_not_z)
            setup_histogram(unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1],2)
            leg_cunfolded_not_z.AddEntry(unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1], "unfolded", "P")
            unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
            unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetYaxis().SetRangeUser(0.0,unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetMaximum()*1.5)
            unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].SetTitle("")
            unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].Draw()
            setup_histogram(input_data_z_scaled,4)
            leg_cunfolded_not_z.AddEntry(input_data_z_scaled, "side-band subtracted, eff. corrected", "P")
            input_data_z_scaled.Draw("same")
            leg_cunfolded_not_z.Draw("same")
            latex = TLatex(0.7,0.8,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cunfolded_not_z.SaveAs("%s/cunfolded_not_z_%s.eps" % (self.d_resultsallpdata, suffix))

            h_unfolded_not_stat_error = TH1F("h_unfolded_not_stat_error"+suffix, "h_unfolded_not_stat_error"+suffix, self.p_nbinshape_reco, self.varshapebinarray_reco)
            for ibinshape in range(self.p_nbinshape_reco):
                error_on_unfolded = unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetBinError(input_mc_gen.GetXaxis().FindBin(self.lvarshape_binmin_reco[ibinshape]))
                content_on_unfolded = unfolded_z_scaled_list[i_iter_choice][input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2])-1].GetBinContent(input_mc_gen.GetXaxis().FindBin(self.lvarshape_binmin_reco[ibinshape]))
                error_on_input_data = input_data_z_scaled.GetBinError(ibinshape+1)
                content_on_input_data = input_data_z_scaled.GetBinContent(ibinshape+1)
                if error_on_input_data != 0 and content_on_unfolded != 0:
                    h_unfolded_not_stat_error.SetBinContent(ibinshape+1,(error_on_unfolded*content_on_input_data)/(content_on_unfolded*error_on_input_data))
                else:
                    h_unfolded_not_stat_error.SetBinContent(ibinshape+1,0.0)
            cunfolded_not_stat_error = TCanvas('cunfolded_not_stat_error '+suffix, 'Ratio of stat error after to before unfolding'+suffix)
            punfolded_not_stat_error = TPad('punfolded_not_stat_error'+suffix, "Ratio of stat error after to before unfolding"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(punfolded_not_stat_error)
            cunfolded_not_stat_error.SetCanvasSize(1900, 1500)
            cunfolded_not_stat_error.SetWindowSize(500, 500)
            setup_histogram(h_unfolded_not_stat_error,4)
            h_unfolded_not_stat_error.SetTitle("Ratio of rel. stat. unc. after to before unfolding")
            h_unfolded_not_stat_error.SetXTitle(self.v_varshape_latex)
            h_unfolded_not_stat_error.SetYTitle("ratio")
            h_unfolded_not_stat_error.GetYaxis().SetRangeUser(0.0,h_unfolded_not_stat_error.GetMaximum()*1.6)
            h_unfolded_not_stat_error.Draw()
            latex = TLatex(0.2,0.8,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cunfolded_not_stat_error.SaveAs("%s/unfolded_not_stat_error_%s.eps" % (self.d_resultsallpdata, suffix))


        cconvergence_jetpt = TCanvas('cconvergence_jetpt ', '1D output of convergence')
        pconvergence_jetpt = TPad('pconvergence_jetpt', "1D output of convergence",0.0,0.001,1.0,1.0)
        setup_pad(pconvergence_jetpt)
        cconvergence_jetpt.SetCanvasSize(1900, 1500)
        cconvergence_jetpt.SetWindowSize(500, 500)
        leg_jetpt = TLegend(.7, .5, .85, .9, "iterations")
        setup_legend(leg_jetpt)
        for i in range(self.niter_unfolding):
            setup_histogram(unfolded_jetpt_scaled_list[i],i+1)
            leg_jetpt.AddEntry(unfolded_jetpt_scaled_list[i],("%d" % (i+1)),"P")
            if i == 0:
                unfolded_jetpt_scaled_list[i].GetXaxis().SetRangeUser(self.lvar2_binmin_reco[0]+0.01, self.lvar2_binmax_reco[-1]-0.001)
                unfolded_jetpt_scaled_list[i].GetYaxis().SetRangeUser(0,unfolded_jetpt_scaled_list[i].GetMaximum()*2.0)
                unfolded_jetpt_scaled_list[i].SetTitle("")
                unfolded_jetpt_scaled_list[i].SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
                unfolded_jetpt_scaled_list[i].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s (#it{c}/GeV)" % self.p_latexbin2var)
            unfolded_jetpt_scaled_list[i].Draw("same")
        leg_jetpt.Draw("same")
        latex = TLatex(0.2,0.8,'%.2f < %s < %.2f' % (self.lvarshape_binmin_gen[0], self.v_varshape_latex, self.lvarshape_binmax_gen[-1]))
        draw_latex(latex)
        cconvergence_jetpt.SaveAs("%s/convergence_jetpt.eps" % (self.d_resultsallpdata))

        cconvergence_refolding_jetpt = TCanvas('cconvergence_refolding_jetpt ', '1D output of refolding convergence')
        pconvergence_refolding_jetpt = TPad('pconvergence_refolding_jetpt', "1D output of refolding convergence",0.0,0.001,1.0,1.0)
        setup_pad(pconvergence_refolding_jetpt)
        cconvergence_refolding_jetpt.SetCanvasSize(1900, 1500)
        cconvergence_refolding_jetpt.SetWindowSize(500, 500)
        leg_refolding_jetpt = TLegend(.7, .5, .85, .9, "iterations")
        setup_legend(leg_refolding_jetpt)
        for i in range(self.niter_unfolding):
            setup_histogram(refolding_test_jetpt_list[i],i+1)
            leg_refolding_jetpt.AddEntry(refolding_test_jetpt_list[i],("%d" % (i+1)),"P")
            refolding_test_jetpt_list[i].Draw("same")
            refolding_test_jetpt_list[i].SetTitle("")
            refolding_test_jetpt_list[i].SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
            refolding_test_jetpt_list[i].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s (#it{c}/GeV)" % self.p_latexbin2var)
            refolding_test_jetpt_list[i].GetYaxis().SetRangeUser(0.5,2.0)
        leg_refolding_jetpt.Draw("same")
        latex = TLatex(0.2,0.8,'%.2f < %s < %.2f' % (self.lvarshape_binmin_gen[0], self.v_varshape_latex, self.lvarshape_binmax_gen[-1]))
        draw_latex(latex)
        cconvergence_refolding_jetpt.SaveAs("%s/convergence_refolding_jetpt.eps" % (self.d_resultsallpdata))

    def unfolding_closure(self):
        path = "%s/unfolding_closure_results%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        fileouts = TFile.Open(path, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(path))
        unfolding_input_file = TFile.Open(self.n_fileff)
        if not unfolding_input_file:
            self.logger.fatal(make_message_notfound(self.n_fileff))
        response_matrix = unfolding_input_file.Get("response_matrix_closure")
        hzvsjetpt_reco_nocuts = unfolding_input_file.Get("hzvsjetpt_reco_nocuts_closure")
        hzvsjetpt_reco_eff = unfolding_input_file.Get("hzvsjetpt_reco_cuts_closure")
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)
        input_mc_det = unfolding_input_file.Get("input_closure_reco")
        input_mc_det.Multiply(hzvsjetpt_reco_eff)
        input_mc_gen = unfolding_input_file.Get("input_closure_gen")
        kinematic_eff=[]
        hz_gen_nocuts=[]
        input_mc_det_z=[]
        input_mc_gen_z=[]

        kinematic_eff_jetpt = unfolding_input_file.Get("hjetpt_gen_cuts_closure")
        hjetpt_gen_nocuts=unfolding_input_file.Get("hjetpt_gen_nocuts_closure")
        kinematic_eff_jetpt.Divide(hjetpt_gen_nocuts)
        input_mc_gen_jetpt=input_mc_gen.ProjectionY("input_mc_gen_jetpt",1, self.p_nbinshape_gen,"e")
        input_mc_gen_jetpt.Scale(1.0/input_mc_gen_jetpt.Integral(1,-1))

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            input_mc_det_z.append(input_mc_det.ProjectionX("input_mc_det_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_mc_det_z[ibin2].Scale(1.0/input_mc_det_z[ibin2].Integral(1,-1))

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_mc_gen_z[ibin2].Scale(1.0/input_mc_gen_z[ibin2].Integral(1,-1))
            kinematic_eff.append(unfolding_input_file.Get("hz_gen_cuts_closure"+suffix))
            hz_gen_nocuts.append(unfolding_input_file.Get("hz_gen_nocuts_closure"+suffix))
            kinematic_eff[ibin2].Divide(hz_gen_nocuts[ibin2])


        unfolded_z_closure_list=[]
        unfolded_jetpt_closure_list=[]

        for i in range(self.niter_unfolding):
            unfolded_z_closure_list_iter=[]
            unfolding_object = RooUnfoldBayes(response_matrix, input_mc_det, i+1)
            unfolded_zvsjetpt = unfolding_object.Hreco(2)
            unfolded_zvsjetpt.Sumw2()
            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                unfolded_z = unfolded_zvsjetpt.ProjectionX("unfolded_z_%d_%s" % (i+1, suffix),ibin2+1,ibin2+1,"e")
                unfolded_z.Divide(kinematic_eff[ibin2])
                unfolded_z.Scale(1.0/unfolded_z.Integral(1,-1))
                unfolded_z.Divide(input_mc_gen_z[ibin2])
                fileouts.cd()
                unfolded_z.Write("closure_test_%d_%s" % (i+1, suffix))
                unfolded_z_closure_list_iter.append(unfolded_z)

                cclosure_z = TCanvas('cclosure_z '+suffix, '1D output of closure'+suffix)
                pclosure_z = TPad('pclosure_z'+suffix, "1D output of closure"+suffix,0.0,0.001,1.0,1.0)
                setup_pad(pclosure_z)
                cclosure_z.SetCanvasSize(1900, 1500)
                cclosure_z.SetWindowSize(500, 500)
                setup_histogram(unfolded_z,4)
                unfolded_z.GetYaxis().SetRangeUser(0.5,1.5)
                unfolded_z.GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
                unfolded_z.SetTitle("")
                unfolded_z.SetXTitle(self.v_varshape_latex)
                unfolded_z.SetYTitle("closure test")
                unfolded_z.Draw()
                latex = TLatex(0.6,0.25,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
                draw_latex(latex2)
                cclosure_z.SaveAs("%s/cclosure_z_%d_%s.eps" % (self.d_resultsallpdata, i+1, suffix))

            unfolded_jetpt = unfolded_zvsjetpt.ProjectionY("unfolded_jetpt_%d" % (i+1),1, self.p_nbinshape_gen,"e")
            unfolded_jetpt.Divide(kinematic_eff_jetpt)
            unfolded_jetpt.Scale(1.0/unfolded_jetpt.Integral(1,-1))
            unfolded_jetpt.Divide(input_mc_gen_jetpt)
            fileouts.cd()
            unfolded_jetpt.Write("closure_test_jetpt_%d" % (i+1))
            unfolded_jetpt_closure_list.append(unfolded_jetpt)

            cclosure_jetpt = TCanvas('cclosure_jetpt ', '1D output of closure')
            pclosure_jetpt = TPad('pclosure_jetpt', "1D output of closure",0.0,0.001,1.0,1.0)
            setup_pad(pclosure_jetpt)
            cclosure_jetpt.SetCanvasSize(1900, 1500)
            cclosure_jetpt.SetWindowSize(500, 500)
            setup_histogram(unfolded_jetpt,4)
            unfolded_jetpt.GetYaxis().SetRangeUser(0.5,1.5)
            unfolded_jetpt.GetXaxis().SetRangeUser(0.21,0.99)
            unfolded_jetpt.SetTitle("")
            unfolded_jetpt.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
            unfolded_jetpt.SetYTitle("closure test")
            unfolded_jetpt.Draw()
            latex = TLatex(0.6,0.25,'%.2f < %s < %.2f' % (self.lvarshape_binmin_gen[0], self.v_varshape_latex, self.lvarshape_binmax_gen[-1]))
            draw_latex(latex)
            latex2 = TLatex(0.6,0.2,'iteration %d' % (i+1))
            draw_latex(latex2)
            cclosure_jetpt.SaveAs("%s/cclosure_jetpt_%d.eps" % (self.d_resultsallpdata, i+1))

            unfolded_z_closure_list.append(unfolded_z_closure_list_iter)

        path = "%s/sideband_sub%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        unfolding_input_data_file = TFile.Open(path)
        if not unfolding_input_data_file:
            self.logger.fatal(make_message_notfound(path))
        input_data = unfolding_input_data_file.Get("hzvsjetpt")

        input_data_z=[]
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            input_data_z.append(input_data.ProjectionX("input_data_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_data_z[ibin2].Scale(1.0/input_data_z[ibin2].Integral(1,-1))

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cconvergence_closure_z = TCanvas('cconvergence_closure_z '+suffix, '1D output of closure convergence'+suffix)
            pconvergence_closure_z = TPad('pconvergence_closure_z'+suffix, "1D output of closure convergence"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pconvergence_closure_z)
            cconvergence_closure_z.SetCanvasSize(1900, 1500)
            cconvergence_closure_z.SetWindowSize(500, 500)
            leg_closure = TLegend(.7, .5, .85, .9, "iterations")
            setup_legend(leg_closure)
            for i in range(self.niter_unfolding):
                setup_histogram(unfolded_z_closure_list[i][ibin2],i+1)
                leg_closure.AddEntry(unfolded_z_closure_list[i][ibin2],("%d" % (i+1)),"P")
                if i == 0:
                    unfolded_z_closure_list[i][ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01, self.lvarshape_binmax_gen[-1]-0.001)
                    unfolded_z_closure_list[i][ibin2].GetYaxis().SetRangeUser(0.6,2.0)
                    unfolded_z_closure_list[i][ibin2].SetTitle("")
                    unfolded_z_closure_list[i][ibin2].SetXTitle(self.v_varshape_latex)
                    unfolded_z_closure_list[i][ibin2].SetYTitle("closure test")
                unfolded_z_closure_list[i][ibin2].Draw("same")
            leg_closure.Draw("same")
            latex = TLatex(0.2,0.8,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            cconvergence_closure_z.SaveAs("%s/convergence_closure_z_%s.eps" % (self.d_resultsallpdata, suffix))

        cconvergence_closure_jetpt = TCanvas('cconvergence_closure_jetpt ', '1D output of closure convergence')
        pconvergence_closure_jetpt = TPad('pconvergence_closure_jetpt', "1D output of closure convergence",0.0,0.001,1.0,1.0)
        setup_pad(pconvergence_closure_jetpt)
        cconvergence_closure_jetpt.SetCanvasSize(1900, 1500)
        cconvergence_closure_jetpt.SetWindowSize(500, 500)
        leg_closure_jetpt = TLegend(.7, .5, .85, .9, "iterations")
        setup_legend(leg_closure_jetpt)
        for i in range(self.niter_unfolding):
            setup_histogram(unfolded_jetpt_closure_list[i],i+1)
            leg_closure_jetpt.AddEntry(unfolded_jetpt_closure_list[i],("%d" % (i+1)),"P")
            if i == 0:
                unfolded_jetpt_closure_list[i].GetXaxis().SetRangeUser(self.lvar2_binmin_gen[0]+0.01, self.lvar2_binmax_gen[-1]-0.001)
                unfolded_jetpt_closure_list[i].GetYaxis().SetRangeUser(0.6,2.0)
                unfolded_jetpt_closure_list[i].SetTitle("")
                unfolded_jetpt_closure_list[i].SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
                unfolded_jetpt_closure_list[i].SetYTitle("closure test")
            unfolded_jetpt_closure_list[i].Draw("same")
        leg_closure_jetpt.Draw("same")
        latex = TLatex(0.6,0.25,'%.2f < %s < %.2f' % (self.lvarshape_binmin_gen[0], self.v_varshape_latex, self.lvarshape_binmax_gen[-1]))
        draw_latex(latex)
        cconvergence_closure_jetpt.SaveAs("%s/convergence_closure_jetpt.eps" % (self.d_resultsallpdata))

    # pylint: disable=too-many-nested-blocks, fixme
    def jetsystematics(self):
        string_default = "default/default"
        if string_default not in self.d_resultsallpdata:
            self.logger.fatal("Not a default database! Cannot run systematics.")

        debug = True
        if debug:
            print("Categories: ", self.systematic_catnames)
            print("Category labels: ", self.systematic_catlabels)
            print("Numbers of variations: ", self.systematic_variations)
            print("Variations: ", self.systematic_varnames)
            print("Variation labels: ", self.systematic_varlabels)
            print("Correlation: ", self.systematic_correlation)
            print("RMS: ", self.systematic_rms)
            print("Symmetrisation: ", self.systematic_symmetrise)
            print("RMS both sides: ", self.systematic_rms_both_sides)
            print("Feed-down variations: ", self.powheg_nonprompt_varnames)
        #return

        path_def = "%s/unfolding_results%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        input_file_default = TFile.Open(path_def, "update")
        if not input_file_default:
            self.logger.fatal(make_message_notfound(path_def))

        input_powheg = self.get_simulated_yields(self.powheg_path_prompt, 2, True)
        if not input_powheg:
            self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_prompt))
        input_powheg_xsection = input_powheg.Clone(input_powheg.GetName() + "_xsec")

        input_powheg_sys=[]
        input_powheg_xsection_sys=[]
        for i_powheg in range(len(self.powheg_prompt_variations)):
            path = "%s%s.root" % (self.powheg_prompt_variations_path, self.powheg_prompt_variations[i_powheg])
            input_powheg_sys_i = self.get_simulated_yields(path, 2, True)
            if not input_powheg_sys_i:
                self.logger.fatal(make_message_notfound("simulated yields", path))
            input_powheg_sys_i.SetName("fh2_prompt_%s_%d" % (self.v_varshape_binning, i_powheg))
            input_powheg_sys.append(input_powheg_sys_i)
            input_powheg_xsection_sys_i = input_powheg_sys_i.Clone(input_powheg_sys_i.GetName() + "_xsec")
            input_powheg_xsection_sys.append(input_powheg_xsection_sys_i)
        input_powheg_z=[]
        input_powheg_xsection_z=[]
        input_powheg_sys_z=[]
        input_powheg_xsection_sys_z=[]
        tg_powheg=[]
        tg_powheg_xsection=[]


        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_powheg_z.append(input_powheg.ProjectionX("input_powheg_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_z[ibin2].Scale(1.0/input_powheg_z[ibin2].Integral(input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[0]),input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])),"width")
            input_powheg_xsection_z.append(input_powheg_xsection.ProjectionX("input_powheg_xsection_z"+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_xsection_z[ibin2].Scale(1.0,"width")
            input_powheg_sys_z_iter=[]
            input_powheg_xsection_sys_z_iter=[]
            for i_powheg in range(len(self.powheg_prompt_variations)):
                input_powheg_sys_z_iter.append(input_powheg_sys[i_powheg].ProjectionX("input_powheg_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix,ibin2+1,ibin2+1,"e"))
                input_powheg_sys_z_iter[i_powheg].Scale(1.0/input_powheg_sys_z_iter[i_powheg].Integral(input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[0]),input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[-1])),"width")
                input_powheg_xsection_sys_z_iter.append(input_powheg_xsection_sys[i_powheg].ProjectionX("input_powheg_xsection_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix,ibin2+1,ibin2+1,"e"))
            input_powheg_sys_z.append(input_powheg_sys_z_iter)
            input_powheg_xsection_sys_z.append(input_powheg_xsection_sys_z_iter)
            tg_powheg.append(tg_sys(input_powheg_z[ibin2], input_powheg_sys_z[ibin2]))
            tg_powheg_xsection.append(tg_sys(input_powheg_xsection_z[ibin2], input_powheg_xsection_sys_z[ibin2]))







        input_histograms_default=[]
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            name_his = "unfolded_z_sel_%s" % suffix
            input_histograms_default.append(input_file_default.Get(name_his))
            if not input_histograms_default[ibin2]:
                self.logger.fatal(make_message_notfound(name_his, path_def))

        input_files_sys=[]
        for sys_cat in range(self.n_sys_cat):
            input_files_sysvar=[]
            for sys_var, varname in enumerate(self.systematic_varnames[sys_cat]):
                path = path_def.replace(string_default, self.systematic_catnames[sys_cat] + "/" + varname)
                input_files_sysvar.append(TFile.Open(path,"update"))
                if not input_files_sysvar[sys_var]:
                    self.logger.fatal(make_message_notfound(path))
            input_files_sys.append(input_files_sysvar)

        input_histograms_sys=[]
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            name_his = "unfolded_z_sel_%s" % suffix
            input_histograms_syscat=[]
            for sys_cat in range(self.n_sys_cat):
                input_histograms_syscatvar=[]
                for sys_var in range(self.systematic_variations[sys_cat]):
                    input_histograms_syscatvar.append(input_files_sys[sys_cat][sys_var].Get(name_his))
                    path_file = path_def.replace(string_default, self.systematic_catnames[sys_cat] + "/" + self.systematic_varnames[sys_cat][sys_var])
                    if not input_histograms_syscatvar[sys_var]:
                        self.logger.fatal(make_message_notfound(name_his, path_file))
                    if debug:
                        print("Variation: %s, %s: got histogram %s from file %s" % (self.systematic_catnames[sys_cat], self.systematic_varnames[sys_cat][sys_var], name_his, path_file))
                    #input_histograms_syscatvar[sys_var].Scale(1.0,"width") #remove these later and put normalisation directly in systematics
                input_histograms_syscat.append(input_histograms_syscatvar)
            input_histograms_sys.append(input_histograms_syscat)

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            nsys = 0
            csysvar = TCanvas('csysvar '+suffix, 'systematic variations'+suffix)
            psysvar = TPad('psysvar'+suffix, "systematic variations"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(psysvar)
            csysvar.SetCanvasSize(1900, 1500)
            csysvar.SetWindowSize(500, 500)
            leg_sysvar = TLegend(.7, .5, .85, .9, "systematics")
            setup_legend(leg_sysvar)
            leg_sysvar.AddEntry(input_histograms_default[ibin2],"default","P")
            setup_histogram(input_histograms_default[ibin2],1)
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(0.0,input_histograms_default[ibin2].GetMaximum()*1.5)
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01, self.lvarshape_binmax_gen[-1]-0.001)
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw()
            for sys_cat in range(self.n_sys_cat):
                for sys_var in range(self.systematic_variations[sys_cat]):
                    nsys=nsys+1
                    leg_sysvar.AddEntry(input_histograms_sys[ibin2][sys_cat][sys_var],("%s, %s" % (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var])), "P")
                    setup_histogram(input_histograms_sys[ibin2][sys_cat][sys_var],nsys+1)
                    input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
            latex = TLatex(0.2,0.8,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_sysvar.Draw("same")
            csysvar.SaveAs("%s/ysvar_%s.eps" % (self.d_resultsallpdata, suffix))


            for sys_cat in range(self.n_sys_cat):
                suffix2="_%s" % (self.systematic_catnames[sys_cat])
                nsys = 0
                csysvar_each = TCanvas('csysvar '+suffix2+suffix, 'systematic variations'+suffix2+suffix)
                psysvar_each = TPad('psysvar'+suffix2+suffix, "systematic variations"+suffix2+suffix,0.0,0.001,1.0,1.0)
                setup_pad(psysvar_each)
                csysvar_each.SetCanvasSize(1900, 1500)
                csysvar_each.SetWindowSize(500, 500)
                leg_sysvar_each = TLegend(.7, .45, .85, .85, self.systematic_catlabels[sys_cat])
                setup_legend(leg_sysvar_each)
                leg_sysvar_each.AddEntry(input_histograms_default[ibin2],"default","P")
                setup_histogram(input_histograms_default[ibin2],1)
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        if sys_cat == 0:
                            input_histograms_default[ibin2].GetYaxis().SetRangeUser(0.0,input_histograms_default[ibin2].GetMaximum()*2.5)
                        input_histograms_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01, self.lvarshape_binmax_gen[-1]-0.001)
                        input_histograms_default[ibin2].SetTitle("")
                        input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
                        input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
                        input_histograms_default[ibin2].Draw()
                    nsys=nsys+1
                    leg_sysvar_each.AddEntry(input_histograms_sys[ibin2][sys_cat][sys_var], self.systematic_varlabels[sys_cat][sys_var], "P")
                    setup_histogram(input_histograms_sys[ibin2][sys_cat][sys_var], nsys+1)
                    input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
                latex = TLatex(0.2,0.8,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                leg_sysvar_each.Draw("same")
                csysvar_each.SaveAs("%s/ysvar%s_%s.eps" % (self.d_resultsallpdata, suffix2, suffix))




        sys_up=[]
        sys_down=[]
        sys_up_full=[]
        sys_down_full=[]
        for ibin2 in range(self.p_nbin2_gen):
            sys_up_jetpt=[]
            sys_down_jetpt=[]
            sys_up_z_full=[]
            sys_down_z_full=[]
            for ibinshape in range(self.p_nbinshape_gen):
                sys_up_z=[]
                sys_down_z=[]
                error_full_up = 0
                error_full_down = 0
                for sys_cat in range(self.n_sys_cat):
                    error_var_up = 0
                    error_var_down = 0
                    count_sys_up = 0
                    count_sys_down = 0
                    for sys_var in range(self.systematic_variations[sys_cat]):
                        error = input_histograms_sys[ibin2][sys_cat][sys_var].GetBinContent(ibinshape+1)-input_histograms_default[ibin2].GetBinContent(ibinshape+1)
                        if error >= 0:
                            if self.systematic_rms[sys_cat] is True:
                                error_var_up+=error*error
                                count_sys_up=count_sys_up+1
                            else:
                                if error > error_var_up:
                                    error_var_up=error
                        else:
                            if self.systematic_rms[sys_cat] is True:
                                if self.systematic_rms_both_sides[sys_cat] is True:
                                    error_var_up+=error*error
                                    count_sys_up=count_sys_up+1
                                else:
                                    error_var_down+=error*error
                                    count_sys_down=count_sys_down+1
                            else:
                                if abs(error) > error_var_down:
                                    error_var_down = abs(error)
                    if self.systematic_rms[sys_cat]  is True:
                        if count_sys_up != 0:
                            error_var_up = error_var_up/count_sys_up
                        else:
                            error_var_up = 0.0
                        error_var_up=sqrt(error_var_up)
                        if count_sys_down != 0:
                            error_var_down = error_var_down/count_sys_down
                        else:
                            error_var_down = 0.0
                        if self.systematic_rms_both_sides[sys_cat] is True:
                            error_var_down=error_var_up
                        else:
                            error_var_down=sqrt(error_var_down)
                    if self.systematic_symmetrise[sys_cat] is True:
                        if error_var_up > error_var_down:
                            error_var_down = error_var_up
                        else:
                            error_var_up = error_var_down
                    error_full_up+=error_var_up*error_var_up
                    error_full_down+=error_var_down*error_var_down
                    sys_up_z.append(error_var_up)
                    sys_down_z.append(error_var_down)
                error_full_up=sqrt(error_full_up)
                sys_up_z_full.append(error_full_up)
                error_full_down=sqrt(error_full_down)
                sys_down_z_full.append(error_full_down)
                sys_up_jetpt.append(sys_up_z)
                sys_down_jetpt.append(sys_down_z)
            sys_up_full.append(sys_up_z_full)
            sys_down_full.append(sys_down_z_full)
            sys_up.append(sys_up_jetpt)
            sys_down.append(sys_down_jetpt)


        tgsys=[]
        tgsys_cat=[]
        for ibin2 in range(self.p_nbin2_gen):
            shapebins_centres=[]
            shapebins_contents=[]
            shapebins_widths_up=[]
            shapebins_widths_down=[]
            shapebins_error_up=[]
            shapebins_error_down=[]
            tgsys_cat_z=[]
            for ibinshape in range(self.p_nbinshape_gen):
                shapebins_centres.append(input_histograms_default[ibin2].GetBinCenter(ibinshape+1))
                shapebins_contents.append(input_histograms_default[ibin2].GetBinContent(ibinshape+1))
                shapebins_widths_up.append(input_histograms_default[ibin2].GetBinWidth(ibinshape+1)*0.5)
                shapebins_widths_down.append(input_histograms_default[ibin2].GetBinWidth(ibinshape+1)*0.5)
                shapebins_error_up.append(sys_up_full[ibin2][ibinshape])
                shapebins_error_down.append(sys_down_full[ibin2][ibinshape])
            shapebins_centres_array = array('d',shapebins_centres)
            shapebins_contents_array = array('d',shapebins_contents)
            shapebins_widths_up_array = array('d',shapebins_widths_up)
            shapebins_widths_down_array = array('d',shapebins_widths_down)
            shapebins_error_up_array = array('d',shapebins_error_up)
            shapebins_error_down_array = array('d',shapebins_error_down)
            for sys_cat in range(self.n_sys_cat):
                shapebins_contents_cat=[]
                shapebins_error_up_cat=[]
                shapebins_error_down_cat=[]
                for ibinshape in range(self.p_nbinshape_gen):
                    shapebins_contents_cat.append(1.0)
                    shapebins_error_up_cat.append(sys_up[ibin2][ibinshape][sys_cat]/input_histograms_default[ibin2].GetBinContent(ibinshape+1))
                    shapebins_error_down_cat.append(sys_down[ibin2][ibinshape][sys_cat]/input_histograms_default[ibin2].GetBinContent(ibinshape+1))
                shapebins_contents_cat_array = array('d',shapebins_contents_cat)
                shapebins_error_up_cat_array = array('d',shapebins_error_up_cat)
                shapebins_error_down_cat_array = array('d',shapebins_error_down_cat)
                tgsys_cat_z.append(TGraphAsymmErrors(self.p_nbinshape_gen,shapebins_centres_array,shapebins_contents_cat_array,shapebins_widths_down_array,shapebins_widths_up_array,shapebins_error_down_cat_array,shapebins_error_up_cat_array))
            tgsys_cat.append(tgsys_cat_z)

            tgsys.append(TGraphAsymmErrors(self.p_nbinshape_gen,shapebins_centres_array,shapebins_contents_array,shapebins_widths_down_array,shapebins_widths_up_array,shapebins_error_down_array,shapebins_error_up_array))

        h_default_stat_err=[]
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            h_default_stat_err.append(input_histograms_default[ibin2].Clone("h_default_stat_err"+suffix))
            for i in range(h_default_stat_err[ibin2].GetNbinsX()):
                h_default_stat_err[ibin2].SetBinContent(i+1,1.0)
                h_default_stat_err[ibin2].SetBinError(i+1,input_histograms_default[ibin2].GetBinError(i+1)/input_histograms_default[ibin2].GetBinContent(i+1))



        input_pythia8 = []
        input_pythia8_xsection = []
        input_pythia8_z=[]
        input_pythia8_xsection_z=[]
        for i_pythia8 in range(len(self.pythia8_prompt_variations)):
            path = "%s%s.root" % (self.pythia8_prompt_variations_path, self.pythia8_prompt_variations[i_pythia8])
            input_pythia8_i = self.get_simulated_yields(path, 2, True)
            if not input_pythia8_i:
                self.logger.fatal(make_message_notfound("simulated yields", path))
            input_pythia8_i.SetName("fh2_pythia_prompt_%s_%d" % (self.v_varshape_binning, i_pythia8))
            input_pythia8.append(input_pythia8_i)
            input_pythia8_xsection_i = input_pythia8_i.Clone(input_pythia8_i.GetName() + "_xsec")
            input_pythia8_xsection.append(input_pythia8_xsection_i)

            # Ensure correct binning: x - shape, y - jet pt
            if not equal_binning_lists(input_pythia8[i_pythia8], list_x = self.varshaperanges_gen):
                self.logger.fatal("Error: Incorrect binning in x.")
            if not equal_binning_lists(input_pythia8[i_pythia8], list_y = self.var2ranges_gen):
                self.logger.fatal("Error: Incorrect binning in y.")
            if not equal_binning_lists(input_pythia8_xsection[i_pythia8], list_x = self.varshaperanges_gen):
                self.logger.fatal("Error: Incorrect binning in x.")
            if not equal_binning_lists(input_pythia8_xsection[i_pythia8], list_y = self.var2ranges_gen):
                self.logger.fatal("Error: Incorrect binning in y.")

            input_pythia8_z_jetpt=[]
            input_pythia8_xsection_z_jetpt=[]
            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                input_pythia8_z_jetpt.append(input_pythia8[i_pythia8].ProjectionX("input_pythia8"+self.pythia8_prompt_variations[i_pythia8]+suffix,ibin2+1,ibin2+1,"e"))
                input_pythia8_z_jetpt[ibin2].Scale(1.0/input_pythia8_z_jetpt[ibin2].Integral(1,-1),"width")
                input_pythia8_xsection_z_jetpt.append(input_pythia8_xsection[i_pythia8].ProjectionX("input_pythia8_xsection"+self.pythia8_prompt_variations[i_pythia8]+suffix,ibin2+1,ibin2+1,"e"))
            input_pythia8_z.append(input_pythia8_z_jetpt)
            input_pythia8_xsection_z.append(input_pythia8_xsection_z_jetpt)

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cfinalwsys = TCanvas('cfinalwsys '+suffix, 'final result with systematic errors'+suffix)
            pfinalwsys = TPad('pfinalwsys'+suffix, "final result with systematic errors"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pfinalwsys)
            cfinalwsys.SetCanvasSize(1900, 1500)
            cfinalwsys.SetWindowSize(500, 500)
            leg_finalwsys = TLegend(.65, .6, .85, .7, "")
            setup_legend(leg_finalwsys)
            leg_finalwsys.AddEntry(input_histograms_default[ibin2],"data","P")
            setup_histogram(input_histograms_default[ibin2],4)
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(0.0,input_histograms_default[ibin2].GetMaximum()*1.2/2.5)
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01, self.lvarshape_binmax_gen[-1]-0.001)
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            #input_histograms_default[ibin2].SetTitleOffset(1.2,"Y")
            input_histograms_default[ibin2].Draw("")
            setup_tgraph(tgsys[ibin2],17,0.3)
            tgsys[ibin2].Draw("5")
            leg_finalwsys.AddEntry(tgsys[ibin2],"syst. unc.","F")
            input_histograms_default[ibin2].Draw("AXISSAME")
            #PREL latex = TLatex(0.18,0.85,"ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.18,0.85,"pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.18,0.8,"%s (& cc) in charged jets, anti-#it{k}_{T}, #it{R} = 0.4, #left|#it{#eta}_{jet}#right| < 0.5" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.18,0.75,"%.0f < %s < %.0f GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            latex3 = TLatex(0.18,0.7,"%.0f < #it{p}_{T, %s} < %.0f GeV/#it{c}" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1],self.lvar2_binmax_reco[ibin2])))
            draw_latex(latex3)
            leg_finalwsys.Draw("same")
            cfinalwsys.SaveAs("%s/finalwsys_%s.pdf" % (self.d_resultsallpdata, suffix))


            cfinalwsys_wmodels = TCanvas('cfinalwsys_wmodels '+suffix, 'final result with systematic errors with models'+suffix)
            pfinalwsys_wmodels = TPad('pfinalwsys_wmodels'+suffix, "final result with systematic errors with models"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pfinalwsys_wmodels)
            cfinalwsys_wmodels.SetCanvasSize(1900, 1500)
            cfinalwsys_wmodels.SetWindowSize(500, 500)
            leg_finalwsys_wmodels = TLegend(.55, .55, .65, .75, "")
            setup_legend(leg_finalwsys_wmodels)
            leg_finalwsys_wmodels.AddEntry(input_histograms_default[ibin2],"data","P")
            setup_histogram(input_histograms_default[ibin2],4)
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(0.0,input_histograms_default[ibin2].GetMaximum())
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01, self.lvarshape_binmax_gen[-1]-0.001)
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw()
            setup_tgraph(tgsys[ibin2],17,0.3)
            tgsys[ibin2].Draw("5")
            leg_finalwsys_wmodels.AddEntry(tgsys[ibin2],"syst. unc.","F")
            setup_histogram(input_powheg_z[ibin2],418)
            input_powheg_z[ibin2].SetMarkerStyle(24)
            leg_finalwsys_wmodels.AddEntry(input_powheg_z[ibin2], "POWHEG #plus PYTHIA 6", "P")
            input_powheg_z[ibin2].Draw("same")
            setup_tgraph(tg_powheg[ibin2],418,0.3)
            tg_powheg[ibin2].Draw("5")
            markers_pythia = [27, 28]
            for i_pythia8 in range(len(self.pythia8_prompt_variations)):
                setup_histogram(input_pythia8_z[i_pythia8][ibin2],i_pythia8+1,markers_pythia[i_pythia8],2.)
                leg_finalwsys_wmodels.AddEntry(input_pythia8_z[i_pythia8][ibin2], self.pythia8_prompt_variations_legend[i_pythia8],"P")
                input_pythia8_z[i_pythia8][ibin2].Draw("same")
            input_histograms_default[ibin2].Draw("AXISSAME")
            #PREL latex = TLatex(0.18,0.85,"ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.18,0.85,"pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.18,0.8,"%s (& cc) in charged jets, anti-#it{k}_{T}, #it{R} = 0.4, #left|#it{#eta}_{jet}#right| < 0.5" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.18,0.75,"%.0f < %s < %.0f GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            #latex3 = TLatex(0.18,0.7,"%.1f < %s #leq %.1f" % (self.lvarshape_binmin_reco[0], self.v_varshape_latex, self.lvarshape_binmax_reco[-1]))
            latex3 = TLatex(0.18,0.7,"%.0f < #it{p}_{T, %s} < %.0f GeV/#it{c}" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1],self.lvar2_binmax_reco[ibin2])))
            draw_latex(latex3)
            #latex4 = TLatex(0.18,0.65,"pp, #sqrt{#it{s}} = 13 TeV")
            #draw_latex(latex4)
            leg_finalwsys_wmodels.Draw("same")
            cfinalwsys_wmodels.SaveAs("%s/finalwsys_wmodels_%s.pdf" % (self.d_resultsallpdata, suffix))

            crelativesys = TCanvas('crelativesys '+suffix, 'relative systematic errors'+suffix)
            prelativesys = TPad('prelativesys'+suffix, "relative systematic errors"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(prelativesys)
            crelativesys.SetCanvasSize(1900, 1500)
            crelativesys.SetWindowSize(500, 500)
            leg_relativesys = TLegend(.7, .5, .85, .9, "")
            setup_legend(leg_relativesys)
            for sys_cat in range(self.n_sys_cat):
                setup_tgraph(tgsys_cat[ibin2][sys_cat],sys_cat+1,0.3)
                tgsys_cat[ibin2][sys_cat].SetFillStyle(0)
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetRangeUser(0.0,2.8)
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetRangeUser(self.lvarshape_binmin_gen[0]+0.01, self.lvarshape_binmax_gen[-1]-0.001)
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetTitle(self.v_varshape_latex)
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetTitle("relative systematic error")
                leg_relativesys.AddEntry(tgsys_cat[ibin2][sys_cat], self.systematic_catlabels[sys_cat],"LEP")
                if sys_cat == 0:
                    tgsys_cat[ibin2][sys_cat].Draw("A2")
                else:
                    tgsys_cat[ibin2][sys_cat].Draw("2")
            setup_histogram(h_default_stat_err[ibin2],1)
            h_default_stat_err[ibin2].Draw("same")
            latex = TLatex(0.2,0.8,'%.2f < %s < %.2f GeV/#it{c}' % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_relativesys.Draw("same")
            crelativesys.SaveAs("%s/relativesys_%s.pdf" % (self.d_resultsallpdata, suffix))

        path_fd = "%s/feeddown%s%s.root" % (self.d_resultsallpdata, self.case, self.typean)
        file_feeddown = TFile.Open(path_fd)
        if not file_feeddown:
            self.logger.fatal(make_message_notfound(path_fd))
        file_feeddown_variations=[]
        for i_powheg, varname in enumerate(self.powheg_nonprompt_varnames):
            path = path_fd.replace(string_default, "powheg/" + varname)
            file_feeddown_variations.append(TFile.Open(path, "update"))
            if not file_feeddown_variations[i_powheg]:
                self.logger.fatal(make_message_notfound(path))
        h_feeddown_fraction=[]
        h_feeddown_fraction_variations=[]
        tg_feeddown_fraction=[]
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
              (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            h_feeddown_fraction_variations_niter=[]
            h_feeddown_fraction.append(file_feeddown.Get("feeddown_fraction"+suffix))
            for i_powheg in range(len(self.powheg_nonprompt_varnames)):
                h_feeddown_fraction_variations_niter.append(file_feeddown_variations[i_powheg].Get("feeddown_fraction"+suffix))

            h_feeddown_fraction_variations.append(h_feeddown_fraction_variations_niter)
            tg_feeddown_fraction.append(tg_sys(h_feeddown_fraction[ibin2], h_feeddown_fraction_variations[ibin2]))

            cfeeddown_fraction = TCanvas('cfeeddown_fraction '+suffix, 'feeddown fraction'+suffix)
            pfeeddown_fraction = TPad('pfeeddown_fraction'+suffix, "feeddown fraction"+suffix,0.0,0.001,1.0,1.0)
            setup_pad(pfeeddown_fraction)
            cfeeddown_fraction.SetCanvasSize(1900, 1500)
            cfeeddown_fraction.SetWindowSize(500, 500)
            setup_histogram(h_feeddown_fraction[ibin2],4)
            h_feeddown_fraction[ibin2].GetYaxis().SetRangeUser(0.0,0.15)
            h_feeddown_fraction[ibin2].GetXaxis().SetRangeUser(self.lvarshape_binmin_reco[0]+0.01, self.lvarshape_binmax_reco[-1]-0.001)
            h_feeddown_fraction[ibin2].GetXaxis().SetTitle(self.v_varshape_latex)
            h_feeddown_fraction[ibin2].GetYaxis().SetTitle("feed-down fraction")
            h_feeddown_fraction[ibin2].GetYaxis().SetTitleOffset(1.4)
            h_feeddown_fraction[ibin2].SetTitle("")
            h_feeddown_fraction[ibin2].Draw("same")
            setup_tgraph(tg_feeddown_fraction[ibin2],4,0.3)
            tg_feeddown_fraction[ibin2].Draw("5")
            #PREL latex = TLatex(0.18,0.85,"ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.18,0.85,"pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.18,0.8,"%s (& cc) in charged jets, anti-#it{k}_{T}, #it{R} = 0.4, #left|#it{#eta}_{jet}#right| < 0.5" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.18,0.75,"%.0f < %s < %.0f GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            #latex3 = TLatex(0.18,0.7,"%.1f < %s #leq %.1f" % (self.lvarshape_binmin_reco[0], self.v_varshape_latex, self.lvarshape_binmax_reco[-1]))
            latex3 = TLatex(0.18,0.7,"%.0f < #it{p}_{T, %s} < %.0f GeV/#it{c}" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1],self.lvar2_binmax_reco[ibin2])))
            draw_latex(latex3)
            #latex4 = TLatex(0.18,0.65,"pp, #sqrt{#it{s}} = 13 TeV")
            #draw_latex(latex4)
            latex5 = TLatex(0.18,0.6,"stat. unc. from data")
            draw_latex(latex5)
            latex6 = TLatex(0.18,0.55,"syst. unc. from POWHEG #plus PYTHIA 6")
            draw_latex(latex6)
            #latex7 = TLatex(0.65,0.75,"POWHEG based")
            #draw_latex(latex7)
            cfeeddown_fraction.SaveAs("%s/feeddown_fraction_werros_%s.pdf" % (self.d_resultsallpdata, suffix))

    def get_simulated_yields(self, file_path:str, dim:int, prompt: bool):
        '''Create a histogram from a simulation tree.
        file_path - input file path
        dim - dimension of the output histogram: 2, 3
        prompt - prompt or non-prompt: True, False'''

        print("Starting the histogram extraction from an MC tree\nInput file: %s" % file_path)

        if dim not in (2, 3):
            self.logger.fatal("Error: %d is not a supported dimension.", dim)

        # Get the normalisation factor.
        file_sim = TFile.Open(file_path)
        if not file_sim:
            self.logger.fatal(make_message_notfound(file_path))
        pr_xsec = file_sim.Get("fHistXsection")
        if not pr_xsec:
            self.logger.fatal(make_message_notfound("fHistXsection", file_path))
        scale_factor = pr_xsec.GetBinContent(1)/pr_xsec.GetEntries()
        file_sim.Close()

        # Load the tree.
        if "D0" in self.case:
            tree_name = "tree_D0"
            print("Loading the D0 tree")
        elif "Lc" in self.case:
            tree_name = "tree_Lc"
            print("Loading the Lc tree")
        else:
            self.logger.fatal(make_message_notfound("the particle name", self.case))
        tree_sim = uproot.open(file_path)[tree_name]
        if not tree_sim:
            self.logger.fatal(make_message_notfound(tree_name, file_path))

        print("Converting")
        # Convert it into a dataframe.
        list_branches = ["pt_cand", "eta_cand", "phi_cand", "y_cand", "pdg_parton", "pt_jet", \
            "eta_jet", "phi_jet", "delta_r_jet", "z", "n_const", "zg_jet", "rg_jet", "nsd_jet", \
            "Pt_mother_jet", "k0_jet", "k1_jet", "k2_jet", "kT_jet"]
        try:
            df_sim = tree_sim.pandas.df(branches=list_branches)
        except Exception: # pylint: disable=broad-except
            self.logger.fatal(make_message_notfound("variables", tree_name))

        print("Entries in the tree:", len(df_sim))
        print("Filtering %sprompt hadrons" % ("" if prompt else "non-"))
        # Apply the same cuts as in gen MC.
        # cut on jet pt
        df_sim = seldf_singlevar(df_sim, self.v_var2_binning, self.lvar2_binmin_gen[0], self.lvar2_binmax_gen[-1])
        # cut on hadron pt
        df_sim = seldf_singlevar(df_sim, self.v_var_binning, self.lpt_finbinmin[0], self.lpt_finbinmax[-1])
        # acceptance cut
        #sel_jet_gen = "abs(y_cand) < 0.5 and abs(eta_jet) < 0.5"
        sel_jet_gen = "abs(eta_jet) <= 0.5"
        df_sim = df_sim.query(sel_jet_gen)
        # pt-dependent rapidity cut
        sel_cand_array = selectfidacc(df_sim["pt_cand"].values, df_sim["y_cand"].values)
        df_sim = df_sim[np.array(sel_cand_array, dtype=bool)]
        # prompt vs. non-prompt selection
        pdg_parton_good = 4 if prompt else 5
        df_sim = df_sim[df_sim["pdg_parton"] == pdg_parton_good]

        # Reject single-track jets.
        #sel_jet_nconst = "n_const > 1"
        #df_sim = df_sim.query(sel_jet_nconst)

        print("Entries after filtering:", len(df_sim))
        # Create, fill and scale the histogram.
        print("Filling a %dD histogram" % dim)
        if dim == 2:
            # Binning: x - shape, y - jet pt
            his2 = makefill2dhist(df_sim, "h2_yield_sim", \
                self.varshapebinarray_gen, self.var2binarray_gen, \
                self.v_varshape_binning, self.v_var2_binning)
            print("Scaling with:", scale_factor)
            his2.Scale(scale_factor)
            print("Entries in the histogram:", his2.GetEntries())
            print("Returning")
            return his2
        if dim == 3:
            # Binning: x - shape, y - jet pt, z - pt hadron
            his3 = makefill3dhist(df_sim, "h3_yield_sim", \
                self.varshapebinarray_gen, self.var2binarray_gen, self.var1binarray, \
                self.v_varshape_binning, self.v_var2_binning, self.v_var_binning)
            print("Scaling with:", scale_factor)
            his3.Scale(scale_factor)
            print("Entries in the histogram:", his3.GetEntries())
            print("Returning")
            return his3
        return None
