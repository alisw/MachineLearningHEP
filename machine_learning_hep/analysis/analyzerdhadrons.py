#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
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
from pathlib import Path
from array import array
import numpy as np
# pylint: disable=unused-wildcard-import, wildcard-import
# pylint: disable=import-error, no-name-in-module, unused-import, consider-using-f-string
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TF1, TH1
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import gInterpreter, gPad
from ROOT import kBlue, kCyan
from machine_learning_hep.fitting.roofitter import RooFitter, calc_signif
from machine_learning_hep.fitting.roofitter import create_text_info, add_text_info_fit, add_text_info_perf
# HF specific imports
from machine_learning_hep.fitting.helpers import MLFitter
from machine_learning_hep.logger import get_logger
from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.hf_pt_spectrum import hf_pt_spectrum
from machine_learning_hep.utils.hist import (get_dim, project_hist)
# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
# pylint: disable=consider-using-enumerate fixme


class AnalyzerDhadrons(Analyzer):  # pylint: disable=invalid-name
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)
        self.logger = get_logger()
        # namefiles pkl
        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.p_nptbins = len(self.lpt_finbinmin)
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]
        self.triggerbit = datap["analysis"][self.typean].get("triggerbit", "")

        dp = datap["analysis"][self.typean]
        self.d_prefix_mc = dp["mc"].get("prefix_dir_res")
        self.d_prefix_data = dp["data"].get("prefix_dir_res")
        self.d_resultsallpmc = self.d_prefix_mc + dp["mc"]["results"][period] \
            if period is not None \
            else self.d_prefix_mc + dp["mc"]["resultsallp"]
        self.d_resultsallpdata =  + dp["data"]["results"][period] \
            if period is not None \
            else self.d_prefix_data + dp["data"]["resultsallp"]

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(
            self.d_resultsallpmc, n_filemass_name)
        self.mltype = datap["ml"]["mltype"]
        # Output directories and filenames
        self.yields_filename = "yields"
        self.fits_dirname = os.path.join(
            self.d_resultsallpdata, f"fits_{case}_{typean}")
        self.yields_syst_filename = "yields_syst"
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)
        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        self.p_rebin = datap["analysis"][self.typean]['n_rebin']
        self.p_pdfnames = datap["analysis"][self.typean]['pdf_names']
        self.p_param_names = datap["analysis"][self.typean]['param_names']

        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]
        self.p_dobkgfromsideband = datap["analysis"][self.typean].get(
            "dobkgfromsideband", None)
        if self.p_dobkgfromsideband is None:
            self.p_dobkgfromsideband = False
        # More specific fit options
        self.include_reflection = datap["analysis"][self.typean].get(
            "include_reflection", False)

        self.p_sigmamb = datap["analysis"]["sigmamb"]
        self.p_br = datap["ml"]["opt"]["BR"]

        self.bins_candpt = np.asarray(self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:], 'd')
        self.nbins = len(self.bins_candpt) - 1
        self.fit_levels = self.cfg('fit_levels', ['mc', 'data'])
        self.fit_sigma = {}
        self.fit_mean = {}
        self.fit_func_bkg = {}
        self.fit_range = {}

        self.path_fig = Path(f'fig/{self.case}/{self.typean}')
        for folder in ['qa', 'fit', 'roofit', 'sideband', 'signalextr', 'fd', 'uf']:
            (self.path_fig / folder).mkdir(parents=True, exist_ok=True)

        self.rfigfile = TFile(str(self.path_fig / 'output.root'), 'recreate')

        self.fitter = RooFitter()
        self.roo_ws = {}
        self.roows = {}

        # Systematics
        self.mt_syst_dict = datap["analysis"][self.typean].get(
            "systematics", None)
        self.d_mt_results_path = os.path.join(
            self.d_resultsallpdata, "multi_trial")

        self.p_anahpt = datap["analysis"]["anahptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_crosssec_prompt = datap["analysis"]["crosssec_prompt"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_inputfonllpred = datap["analysis"]["inputfonllpred"]
        self.p_triggereff = datap["analysis"][self.typean].get("triggereff", [1])
        self.p_triggereffunc = datap["analysis"][self.typean].get(
            "triggereffunc", [0])

        self.root_objects = []

        # Fitting
        self.p_performval = datap["analysis"].get(
            "event_cand_validation", None)


    #region helpers
    def _save_canvas(self, canvas, filename):
        # folder = self.d_resultsallpmc if mcordata == 'mc' else self.d_resultsallpdata
        canvas.SaveAs(f'fig/{self.case}/{self.typean}/{filename}')


    def _save_hist(self, hist, filename, option = ''):
        if not hist:
            self.logger.error('no histogram for <%s>', filename)
            # TODO: remove file if it exists?
            return
        c = TCanvas()
        if isinstance(hist, TH1) and get_dim(hist) == 2 and 'texte' not in option:
            option += 'texte'
        hist.Draw(option)
        self._save_canvas(c, filename)
        rfilename = filename.split('/')[-1]
        rfilename = rfilename.removesuffix('.png')
        self.rfigfile.WriteObject(hist, rfilename)

    #region fitting
    def _roofit_mass(self, level, hist, ipt, pdfnames, param_names, fitcfg, roows = None, filename = None):
        if fitcfg is None:
            return None, None
        res, ws, frame, residual_frame = self.fitter.fit_mass_new(hist, pdfnames, fitcfg, level, roows, True)
        frame.SetTitle(f'inv. mass for p_{{T}} {self.bins_candpt[ipt]} - {self.bins_candpt[ipt+1]} GeV/c')
        c = TCanvas()

        textInfoRight = create_text_info(0.62, 0.68, 1.0, 0.89)
        add_text_info_fit(textInfoRight, frame, ws, param_names)

        textInfoLeft = create_text_info(0.12, 0.68, 0.6, 0.89)
        if level == "data":
            mean_sgn = ws.var(self.p_param_names["gauss_mean"])
            sigma_sgn = ws.var(self.p_param_names["gauss_sigma"])
            (sig, sig_err, bkg, bkg_err,
            signif, signif_err, s_over_b, s_over_b_err
            ) = calc_signif(ws, res, pdfnames, param_names, mean_sgn, sigma_sgn)

            add_text_info_perf(textInfoLeft, sig, sig_err, bkg, bkg_err, s_over_b, s_over_b_err, signif, signif_err)

        frame.Draw()
        textInfoRight.Draw()
        textInfoLeft.Draw()

        if res.status() == 0:
            self._save_canvas(c, filename)
        else:
            self.logger.warning('Invalid fit result for %s', hist.GetName())
            # func_tot.Print('v')
            filename = filename.replace('.png', '_invalid.png')
            self._save_canvas(c, filename)

        if level == "data":
            residual_frame.SetTitle(f'inv. mass for p_{{T}} {self.bins_candpt[ipt]} - {self.bins_candpt[ipt+1]} GeV/c')
            cres = TCanvas()
            residual_frame.Draw()
            filename = filename.replace('.png', '_residual.png')
            self._save_canvas(cres, filename)

        return res, ws


    def _fit_mass(self, hist, filename = None):
        if hist.GetEntries() == 0:
            raise UserWarning('Cannot fit histogram with no entries')
        fit_range = self.cfg('mass_fit.range')
        func_sig = TF1('funcSig', self.cfg('mass_fit.func_sig'), *fit_range)
        func_bkg = TF1('funcBkg', self.cfg('mass_fit.func_bkg'), *fit_range)
        par_offset = func_sig.GetNpar()
        func_tot = TF1('funcTot', f"{self.cfg('mass_fit.func_sig')} + {self.cfg('mass_fit.func_bkg')}({par_offset})")
        func_tot.SetParameter(0, hist.GetMaximum()/3.) # TODO: better seeding?
        for par, value in self.cfg('mass_fit.par_start', {}).items():
            self.logger.debug('Setting par %i to %g', par, value)
            func_tot.SetParameter(par, value)
        for par, value in self.cfg('mass_fit.par_constrain', {}).items():
            self.logger.debug('Constraining par %i to (%g, %g)', par, value[0], value[1])
            func_tot.SetParLimits(par, value[0], value[1])
        for par, value in self.cfg('mass_fit.par_fix', {}).items():
            self.logger.debug('Fixing par %i to %g', par, value)
            func_tot.FixParameter(par, value)
        fit_res = hist.Fit(func_tot, "SQL", "", fit_range[0], fit_range[1])
        if fit_res and fit_res.Get() and fit_res.IsValid():
            # TODO: generalize
            par = func_tot.GetParameters()
            idx = 0
            for i in range(func_sig.GetNpar()):
                func_sig.SetParameter(i, par[idx])
                idx += 1
            for i in range(func_bkg.GetNpar()):
                func_bkg.SetParameter(i, par[idx])
                idx += 1
            if filename:
                c = TCanvas()
                hist.Draw()
                func_sig.SetLineColor(kBlue)
                func_sig.Draw('lsame')
                func_bkg.SetLineColor(kCyan)
                func_bkg.Draw('lsame')
                self._save_canvas(c, filename)
        else:
            self.logger.warning('Invalid fit result for %s', hist.GetName())
            # func_tot.Print('v')
            filename = filename.replace('.png', '_invalid.png')
            self._save_hist(hist, filename)
            # TODO: how to deal with this

        return (fit_res, func_sig, func_bkg)


    # pylint: disable=too-many-branches,too-many-statements
    def fit(self):
        self.logger.info("Fitting inclusive mass distributions")
        gStyle.SetOptFit(1111)
        for level in self.fit_levels:
            self.fit_mean[level] = [None] * self.nbins
            self.fit_sigma[level] = [None] * self.nbins
            self.fit_func_bkg[level] = [None] * self.nbins
            self.fit_range[level] = [None] * self.nbins
            self.roo_ws[level] = [None] * self.nbins
            rfilename = self.n_filemass_mc if "mc" in level else self.n_filemass
            fitcfg = None

            fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
            fileout = TFile(fileout_name, "RECREATE")

            yieldshistos = TH1F("hyields0", "", \
                                len(self.lpt_finbinmin), array("d", self.bins_candpt))
            meanhistos = TH1F("hmean0", "", \
                                len(self.lpt_finbinmin), array("d", self.bins_candpt))
            sigmahistos = TH1F("hsigmas0", "", \
                                len(self.lpt_finbinmin), array("d", self.bins_candpt))
            signifhistos = TH1F("hsignifs0", "", \
                                len(self.lpt_finbinmin), array("d", self.bins_candpt))
            soverbhistos = TH1F("hSoverB0", "", \
                                len(self.lpt_finbinmin), array("d", self.bins_candpt))

            with TFile(rfilename) as rfile:
                for ipt in range(len(self.lpt_finbinmin)):
                    self.logger.debug('fitting %s - %i', level, ipt)
                    roows = self.roows.get(ipt)
                    if self.mltype == "MultiClassification":
                        suffix = "%s%d_%d_%.2f%.2f%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt][0],
                          self.lpt_probcutfin[ipt][1], self.lpt_probcutfin[ipt][2])
                    else:
                        suffix = "%s%d_%d_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt])
                    h_invmass = rfile.Get('hmass' + suffix)
                    # Rebin
                    h_invmass.Rebin(self.p_rebin[ipt])
                    if h_invmass.GetEntries() < 100: # TODO: reconsider criterion
                        self.logger.error('Not enough entries to fit for %s bin %d', level, ipt)
                        continue
                    ptrange = (self.bins_candpt[ipt], self.bins_candpt[ipt+1])

                    if self.cfg('mass_fit'):
                        fit_res, _, func_bkg = self._fit_mass(
                            h_invmass,
                            f'fit/h_mass_fitted_pthf-{ptrange[0]}-{ptrange[1]}_{level}.png')
                        if fit_res and fit_res.Get() and fit_res.IsValid():
                            self.fit_mean[level][ipt] = fit_res.Parameter(1)
                            self.fit_sigma[level][ipt] = fit_res.Parameter(2)
                            self.fit_func_bkg[level][ipt] = func_bkg
                        else:
                            self.logger.error('Fit failed for %s bin %d', level, ipt)

                    if self.cfg('mass_roofit'):
                        for entry in self.cfg('mass_roofit', []):
                            if lvl := entry.get('level'):
                                if lvl != level:
                                    continue
                            if ptspec := entry.get('ptrange'):
                                if ptspec[0] > ptrange[0] or ptspec[1] < ptrange[1]:
                                    continue
                            fitcfg = entry
                            break
                        self.logger.debug("Using fit config for %i: %s", ipt, fitcfg)
                        if datasel := fitcfg.get('datasel'):
                            h = rfile.Get(f'h_mass-pthf_{datasel}')
                            h_invmass = project_hist(h, [0], {1: (ipt+1, ipt+1)}) # TODO: under-/overflow for jets

                        for fixpar in fitcfg.get('fix_params', []):
                            if roows.var(fixpar):
                                roows.var(fixpar).setConstant(True)
                        if h_invmass.GetEntries() == 0:
                            continue
                        roo_res, roo_ws = self._roofit_mass(
                            level, h_invmass, ipt, self.p_pdfnames, self.p_param_names, fitcfg, roows,
                            f'roofit/h_mass_fitted_pthf-{ptrange[0]}-{ptrange[1]}_{level}.png')
                        self.roo_ws[level][ipt] = roo_ws
                        self.roows[ipt] = roo_ws
                        if roo_res.status() == 0:
                            if level in ('data', 'mc_sig'):
                                self.fit_mean[level][ipt] = roo_ws.var(self.p_param_names["gauss_mean"]).getValV()
                                self.fit_sigma[level][ipt] = roo_ws.var(self.p_param_names["gauss_sigma"]).getValV()
                            var_m = fitcfg.get('var', 'm')
                            pdf_bkg = roo_ws.pdf(self.p_pdfnames["pdf_bkg"])
                            if pdf_bkg:
                                self.fit_func_bkg[level][ipt] = pdf_bkg.asTF(roo_ws.var(var_m))
                            self.fit_range[level][ipt] = (roo_ws.var(var_m).getMin('fit'), \
                                                          roo_ws.var(var_m).getMax('fit'))
                        else:
                            self.logger.error('RooFit failed for %s bin %d', level, ipt)

                        if level == "data":
                            mean_sgn = roo_ws.var(self.p_param_names["gauss_mean"])
                            sigma_sgn = roo_ws.var(self.p_param_names["gauss_sigma"])
                            (sig, sig_err, _, _,
                                signif, signif_err, s_over_b, s_over_b_err
                            ) = calc_signif(roo_ws, roo_res, self.p_pdfnames, self.p_param_names, mean_sgn, sigma_sgn)

                            yieldshistos.SetBinContent(ipt+1, sig)
                            yieldshistos.SetBinError(ipt+1, sig_err)
                            meanhistos.SetBinContent(ipt+1, mean_sgn.getVal())
                            meanhistos.SetBinError(ipt+1, mean_sgn.getError())
                            sigmahistos.SetBinContent(ipt+1, sigma_sgn.getVal())
                            sigmahistos.SetBinError(ipt+1, sigma_sgn.getError())
                            signifhistos.SetBinContent(ipt+1, signif)
                            signifhistos.SetBinError(ipt+1, signif_err)
                            soverbhistos.SetBinContent(ipt+1, s_over_b)
                            soverbhistos.SetBinError(ipt+1, s_over_b_err)
                fileout.cd()
                yieldshistos.Write()
                meanhistos.Write()
                sigmahistos.Write()
                signifhistos.Write()
                soverbhistos.Write()
            fileout.Close()

    def yield_syst(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        if not self.fitter:
            self.fitter = MLFitter(self.case, self.datap, self.typean,
                                   self.n_filemass, self.n_filemass_mc)
            if not self.fitter.load_fits(self.fits_dirname):
                self.logger.error(
                    "Cannot load fits from dir %s", self.fits_dirname)
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
        print(self.n_fileff)
        lfileeff = TFile.Open(self.n_fileff)
        lfileeff.ls()
        fileouteff = TFile.Open("%s/%s%s%s.root" % (self.d_resultsallpmc, self.efficiency_filename,
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

        h_gen_pr = lfileeff.Get("h_gen_pr")
        h_sel_pr = lfileeff.Get("h_sel_pr")
        h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
        h_sel_pr.Draw("same")
        fileouteff.cd()
        h_sel_pr.SetName("eff")
        h_sel_pr.Write()
        h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
        h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s %s (1/GeV)"
                                     % (self.p_latexnhadron, self.typean))
        h_sel_pr.SetMinimum(0.001)
        h_sel_pr.SetMaximum(1.0)
        gPad.SetLogy()
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

        h_gen_fd = lfileeff.Get("h_gen_fd")
        h_sel_fd = lfileeff.Get("h_sel_fd")
        h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
        h_sel_fd.Draw("same")
        fileouteff.cd()
        h_sel_fd.SetName("eff_fd")
        h_sel_fd.Write()
        h_sel_fd.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
        h_sel_fd.GetYaxis().SetTitle("Acc x efficiency feed-down %s %s (1/GeV)"
                                     % (self.p_latexnhadron, self.typean))
        h_sel_fd.SetMinimum(0.001)
        h_sel_fd.SetMaximum(1.)
        gPad.SetLogy()
        legeffFD.Draw()
        cEffFD.SaveAs("%s/EffFD%s%s.eps" % (self.d_resultsallpmc,
                                            self.case, self.typean))


    @staticmethod
    def calculate_norm(logger, hevents, hselevents): #TO BE FIXED WITH EV SEL
        if not hevents:
            # pylint: disable=undefined-variable
            logger.error("Missing hevents")
        if not hselevents:
            # pylint: disable=undefined-variable
            logger.error("Missing hselevents")

        n_events = hevents.Integral()
        n_selevents = hselevents.Integral()

        return n_events, n_selevents

    def makenormyields(self):  # pylint: disable=import-outside-toplevel, too-many-branches
        gROOT.SetBatch(True)
        self.loadstyle()

        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        yield_filename = "/data8/majak/MLHEP/input-fd-10092024/yields-1224_split_bkg_0.60_0.60_fd_0.00-fixed-sigma.root"
        if not os.path.exists(yield_filename):
            self.logger.fatal(
                "Yield file %s could not be found", yield_filename)

        fileouteff = f"{self.d_resultsallpmc}/{self.efficiency_filename}{self.case}{self.typean}.root"
        if not os.path.exists(fileouteff):
            self.logger.fatal(
                "Efficiency file %s could not be found", fileouteff)

        fileoutcross = "%s/finalcross%s%s.root" % \
            (self.d_resultsallpdata, self.case, self.typean)

        namehistoeffprompt = "eff"
        namehistoefffeed = "eff_fd"
        nameyield = "hRawYields"

        histonorm = TH1F("histonorm", "histonorm", 1, 0, 1)

        filemass = TFile.Open(self.n_filemass)
        hevents = filemass.Get("all_events")
        hselevents = filemass.Get("sel_events")
        norm, selnorm = self.calculate_norm(self.logger, hevents, hselevents)
        histonorm.SetBinContent(1, selnorm)
        self.logger.warning("Number of events %d", norm)
        self.logger.warning("Number of events after event selection %d", selnorm)

        if self.p_dobkgfromsideband:
            fileoutbkg = TFile.Open("%s/Background_fromsidebands_%s_%s.root" % \
                                    (self.d_resultsallpdata, self.case, self.typean))
            hbkg = fileoutbkg.Get("hbkg_fromsidebands")
            hbkg.Scale(1./selnorm)
            fileoutbkgscaled = TFile.Open("%s/NormBackground_fromsidebands_%s_%s.root" % \
                                          (self.d_resultsallpdata, self.case,
                                           self.typean), "RECREATE")
            fileoutbkgscaled.cd()
            hbkg.Write()
            fileoutbkgscaled.Close()

        output_prompt = []
        hf_pt_spectrum(self.p_anahpt,
                           self.p_br,
                           self.p_inputfonllpred,
                           self.p_fd_method,
                           None,
                           fileouteff,
                           namehistoeffprompt,
                           namehistoefffeed,
                           yield_filename,
                           nameyield,
                           selnorm,
                           self.p_sigmamb,
                           output_prompt,
                           fileoutcross,
                           self.p_crosssec_prompt)

        fileoutcrosstot = TFile.Open("%s/finalcross%s%stot.root" %
                                     (self.d_resultsallpdata, self.case, self.typean), "recreate")

        f_fileoutcross = TFile.Open(fileoutcross)
        if f_fileoutcross:
            hcross = f_fileoutcross.Get("hptspectrum")
            hcrossbr = f_fileoutcross.Get("hptspectrum_wo_br")
            fileoutcrosstot.cd()
            hcross.Write()
            hcrossbr.Write()
        histonorm.Write()
        fileoutcrosstot.Close()
