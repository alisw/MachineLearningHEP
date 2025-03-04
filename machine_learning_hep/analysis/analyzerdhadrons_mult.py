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
# pylint: disable=unused-wildcard-import, wildcard-import
import os
from array import array
from pathlib import Path

import numpy as np

# pylint: disable=import-error, no-name-in-module, unused-import, consider-using-f-string
from ROOT import (
    TF1,
    TH1,
    TH1D,
    TH1F,
    TH2F,
    TArrow,
    TCanvas,
    TDirectory,
    TFile,
    TLegend,
    TLine,
    TPad,
    TPaveLabel,
    TPaveText,
    TStyle,
    TText,
    gInterpreter,
    gPad,
    gROOT,
    gStyle,
    kBlue,
    kCyan,
)

from machine_learning_hep.analysis.analyzer import Analyzer

# HF specific imports
from machine_learning_hep.fitting.roofitter import (
    RooFitter,
    add_text_info_fit,
    add_text_info_perf,
    calc_signif,
    create_text_info,
)
from machine_learning_hep.hf_pt_spectrum import hf_pt_spectrum
from machine_learning_hep.logger import get_logger
from machine_learning_hep.root import save_root_object
from machine_learning_hep.utils.hist import get_dim, project_hist


# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
# pylint: disable=consider-using-enumerate, fixme
class AnalyzerDhadrons_mult(Analyzer):  # pylint: disable=invalid-name
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)
        self.logger = get_logger()
        self.logger.warning("TEST")
        # namefiles pkl
        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.p_nptbins = len(self.lpt_finbinmin)
        self.lpt_probcutfin_tmp = datap["mlapplication"]["probcutoptimal"]

        self.signal_loss = datap["analysis"][self.typean].get("signal_loss", "")
        self.lvar2_binmin = datap["analysis"][self.typean]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"][self.typean]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]
        self.v_var2_binning_gen = datap["analysis"][self.typean]["var_binning2_gen"]
        self.p_nbin2 = len(self.lvar2_binmin)

        dp = datap["analysis"][typean]
        self.d_prefix_mc = dp["mc"].get("prefix_dir_res")
        self.d_prefix_data = dp["data"].get("prefix_dir_res")
        self.d_resultsallpmc = self.d_prefix_mc + (
            dp["mc"]["results"][period] if period is not None else dp["mc"]["resultsallp"]
        )
        self.d_resultsallpdata = self.d_prefix_data + (
            dp["data"]["results"][period] if period is not None else dp["data"]["resultsallp"]
        )

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc, n_filemass_name)
        self.mltype = datap["ml"]["mltype"]
        self.n_filecross = datap["files_names"]["crossfilename"]
        self.p_mass_fit_lim = datap["analysis"][self.typean]["mass_fit_lim"]

        # Output directories and filenames
        self.yields_filename = "yields"
        self.fits_dirname = os.path.join(self.d_resultsallpdata, f"fits_{case}_{typean}")
        self.efficiency_filename = "efficiencies"
        self.sideband_subtracted_filename = "sideband_subtracted"

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)
        self.p_bin_width = datap["analysis"][self.typean]["bin_width"]

        self.p_rebin = datap["analysis"][self.typean]["n_rebin"]
        self.p_pdfnames = datap["analysis"][self.typean]["pdf_names"]
        self.p_param_names = datap["analysis"][self.typean]["param_names"]

        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]
        self.p_latexbin2var = datap["analysis"][self.typean]["latexbin2var"]
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

        # Roofit
        self.bins_candpt = np.asarray(self.cfg("sel_an_binmin", []) + self.cfg("sel_an_binmax", [])[-1:], "d")
        self.nbins = len(self.bins_candpt) - 1
        self.fit_levels = self.cfg("fit_levels", ["mc", "data"])
        self.fit_sigma = {}
        self.fit_mean = {}
        self.fit_func_bkg = {}
        self.fit_range = {}

        self.path_fig = Path(f"fig/{self.case}/{self.typean}")
        for folder in ["qa", "fit", "roofit", "sideband", "signalextr", "fd", "uf"]:
            (self.path_fig / folder).mkdir(parents=True, exist_ok=True)

        self.rfigfile = TFile(str(self.path_fig / "output.root"), "recreate")

        self.fitter = RooFitter()
        self.roo_ws = {}
        self.roows = {}

        self.p_anahpt = datap["analysis"]["anahptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_sigmamb = datap["analysis"]["sigmamb"]
        self.p_inputfonllpred = datap["analysis"]["inputfonllpred"]
        self.root_objects = []

        self.get_crossmb_from_path = datap["analysis"][self.typean].get("get_crossmb_from_path", None)
        self.path_for_crossmb = datap["analysis"][self.typean].get("path_for_crossmb", None)

        # Take efficiencies from another analysis.
        self.path_file_eff = datap["analysis"][self.typean].get("path_eff", None)
        self.mult_bin_eff = datap["analysis"][self.typean].get("mult_bin_eff", None)

        if (self.path_file_eff and not self.mult_bin_eff) or (not self.path_file_eff and self.mult_bin_eff):
            # That is incoherent
            self.logger.fatal('Either both or none of the lists "path_eff" and "mult_bin_eff"must be specified')

        if not self.path_file_eff:
            self.path_file_eff = [None] * self.p_nbin2
            self.mult_bin_eff = [None] * self.p_nbin2

        if len(self.path_file_eff) != self.p_nbin2 or len(self.mult_bin_eff) != self.p_nbin2:
            self.logger.fatal(
                "Efficiencies are requested to be taken from another analysis. "
                'Make sure lists "path_eff" and "mult_bin_eff" have the same '
                "length as the number of those bins (%i).",
                self.p_nbin2,
            )

        self.p_performval = datap["analysis"].get("event_cand_validation", None)

    # pylint: disable=import-outside-toplevel
    # region helpers
    def _save_canvas(self, canvas, filename):
        # folder = self.d_resultsallpmc if mcordata == 'mc' else self.d_resultsallpdata
        canvas.SaveAs(f"fig/{self.case}/{self.typean}/{filename}")

    def _save_hist(self, hist, filename, option=""):
        if not hist:
            self.logger.error("no histogram for <%s>", filename)
            # TODO: remove file if it exists?
            return
        c = TCanvas()
        if isinstance(hist, TH1) and get_dim(hist) == 2 and "texte" not in option:
            option += "texte"
        hist.Draw(option)
        self._save_canvas(c, filename)
        rfilename = filename.split("/")[-1]
        rfilename = rfilename.removesuffix(".png")
        self.rfigfile.WriteObject(hist, rfilename)

    # region fitting
    def _roofit_mass(self, level, hist, ipt, pdfnames, param_names, fitcfg, roows=None, filename=None):
        if fitcfg is None:
            return None, None
        res, ws, frame, residual_frame = self.fitter.fit_mass_new(hist, pdfnames, fitcfg, level, roows, True)
        frame.SetTitle(f"inv. mass for p_{{T}} {self.bins_candpt[ipt]} - {self.bins_candpt[ipt + 1]} GeV/c")
        c = TCanvas()

        textInfoRight = create_text_info(0.62, 0.68, 1.0, 0.89)
        add_text_info_fit(textInfoRight, frame, ws, param_names)

        textInfoLeft = create_text_info(0.12, 0.68, 0.6, 0.89)
        if level == "data":
            mean_sgn = ws.var(self.p_param_names["gauss_mean"])
            sigma_sgn = ws.var(self.p_param_names["gauss_sigma"])
            (sig, sig_err, bkg, bkg_err, signif, signif_err, s_over_b, s_over_b_err) = calc_signif(
                ws, res, pdfnames, param_names, mean_sgn, sigma_sgn
            )

            add_text_info_perf(textInfoLeft, sig, sig_err, bkg, bkg_err, s_over_b, s_over_b_err, signif, signif_err)

        frame.Draw()
        textInfoRight.Draw()
        textInfoLeft.Draw()

        if res.status() == 0:
            self._save_canvas(c, filename)
        else:
            self.logger.warning("Invalid fit result for %s", hist.GetName())
            # func_tot.Print('v')
            filename = filename.replace(".png", "_invalid.png")
            self._save_canvas(c, filename)

        if level == "data":
            residual_frame.SetTitle(
                f"inv. mass for p_{{T}} {self.bins_candpt[ipt]} - {self.bins_candpt[ipt + 1]} GeV/c"
            )
            cres = TCanvas()
            residual_frame.Draw()
            filename = filename.replace(".png", "_residual.png")
            self._save_canvas(cres, filename)

        return res, ws

    def _fit_mass(self, hist, filename=None):
        if hist.GetEntries() == 0:
            raise UserWarning("Cannot fit histogram with no entries")
        fit_range = self.cfg("mass_fit.range")
        func_sig = TF1("funcSig", self.cfg("mass_fit.func_sig"), *fit_range)
        func_bkg = TF1("funcBkg", self.cfg("mass_fit.func_bkg"), *fit_range)
        par_offset = func_sig.GetNpar()
        func_tot = TF1("funcTot", f"{self.cfg('mass_fit.func_sig')} + {self.cfg('mass_fit.func_bkg')}({par_offset})")
        func_tot.SetParameter(0, hist.GetMaximum() / 3.0)  # TODO: better seeding?
        for par, value in self.cfg("mass_fit.par_start", {}).items():
            self.logger.debug("Setting par %i to %g", par, value)
            func_tot.SetParameter(par, value)
        for par, value in self.cfg("mass_fit.par_constrain", {}).items():
            self.logger.debug("Constraining par %i to (%g, %g)", par, value[0], value[1])
            func_tot.SetParLimits(par, value[0], value[1])
        for par, value in self.cfg("mass_fit.par_fix", {}).items():
            self.logger.debug("Fixing par %i to %g", par, value)
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
                func_sig.Draw("lsame")
                func_bkg.SetLineColor(kCyan)
                func_bkg.Draw("lsame")
                self._save_canvas(c, filename)
        else:
            self.logger.warning("Invalid fit result for %s", hist.GetName())
            # func_tot.Print('v')
            filename = filename.replace(".png", "_invalid.png")
            self._save_hist(hist, filename)
            # TODO: how to deal with this

        return (fit_res, func_sig, func_bkg)

    # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks
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
            fileout_name = self.make_file_path(
                self.d_resultsallpdata, self.yields_filename, "root", None, [self.case, self.typean]
            )
            fileout = TFile(fileout_name, "RECREATE")
            with TFile(rfilename) as rfile:
                for ibin2 in range(len(self.lvar2_binmin)):
                    yieldshistos = TH1F(
                        "hyields%d" % (ibin2), "", len(self.lpt_finbinmin), array("d", self.bins_candpt)
                    )
                    meanhistos = TH1F("hmean%d" % (ibin2), "", len(self.lpt_finbinmin), array("d", self.bins_candpt))
                    sigmahistos = TH1F("hsigmas%d" % (ibin2), "", len(self.lpt_finbinmin), array("d", self.bins_candpt))
                    signifhistos = TH1F(
                        "hsignifs%d" % (ibin2), "", len(self.lpt_finbinmin), array("d", self.bins_candpt)
                    )
                    soverbhistos = TH1F(
                        "hSoverB%d" % (ibin2), "", len(self.lpt_finbinmin), array("d", self.bins_candpt)
                    )

                    lpt_probcutfin = [None] * self.nbins
                    for ipt in range(len(self.lpt_finbinmin)):
                        lpt_probcutfin[ipt] = self.lpt_probcutfin_tmp[self.bin_matching[ipt]]
                        self.logger.debug("fitting %s - %i - %i", level, ipt, ibin2)
                        roows = self.roows.get(ipt)
                        if self.mltype == "MultiClassification":
                            suffix = "%s%d_%d_%.2f%.2f%s_%.2f_%.2f" % (
                                self.v_var_binning,
                                self.lpt_finbinmin[ipt],
                                self.lpt_finbinmax[ipt],
                                lpt_probcutfin[ipt][0],
                                lpt_probcutfin[ipt][1],
                                self.v_var2_binning,
                                self.lvar2_binmin[ibin2],
                                self.lvar2_binmax[ibin2],
                            )
                        else:
                            suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % (
                                self.v_var_binning,
                                self.lpt_finbinmin[ipt],
                                self.lpt_finbinmax[ipt],
                                lpt_probcutfin[ipt],
                                self.v_var2_binning,
                                self.lvar2_binmin[ibin2],
                                self.lvar2_binmax[ibin2],
                            )
                        h_invmass = rfile.Get("hmass" + suffix)
                        # Rebin
                        h_invmass.Rebin(self.p_rebin[ipt])
                        if h_invmass.GetEntries() < 100:  # TODO: reconsider criterion
                            self.logger.error(
                                "Not enough entries to fit for %s, pt bin %d, mult bin %d", level, ipt, ibin2
                            )
                            continue
                        ptrange = (self.bins_candpt[ipt], self.bins_candpt[ipt + 1])
                        multrange = (self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])

                        if self.cfg("mass_fit"):
                            fit_res, _, func_bkg = self._fit_mass(
                                h_invmass,
                                f"fit/h_mass_fitted_pthf-{ptrange[0]}-{ptrange[1]}"
                                f"_{self.v_var2_binning}-{multrange[0]}-{multrange[1]}_{level}.png",
                            )
                            if fit_res and fit_res.Get() and fit_res.IsValid():
                                self.fit_mean[level][ipt] = fit_res.Parameter(1)
                                self.fit_sigma[level][ipt] = fit_res.Parameter(2)
                                self.fit_func_bkg[level][ipt] = func_bkg
                            else:
                                self.logger.error("Fit failed for %s bin %d", level, ipt)

                        if self.cfg("mass_roofit"):
                            for entry in self.cfg("mass_roofit", []):
                                if lvl := entry.get("level"):
                                    if lvl != level:
                                        continue
                                if ptspec := entry.get("ptrange"):
                                    if ptspec[0] > ptrange[0] or ptspec[1] < ptrange[1]:
                                        continue
                                fitcfg = entry
                                break
                            self.logger.debug("Using fit config for %i: %s", ipt, fitcfg)
                            if datasel := fitcfg.get("datasel"):
                                h = rfile.Get(f"h_mass-pthf_{datasel}")
                                h_invmass = project_hist(h, [0], {1: (ipt + 1, ipt + 1)})

                            for fixpar in fitcfg.get("fix_params", []):
                                if roows.var(fixpar):
                                    roows.var(fixpar).setConstant(True)
                            if h_invmass.GetEntries() == 0:
                                continue

                            directory_path = Path(f"{self.path_fig}/roofit/mult_{multrange[0]}-{multrange[1]}")
                            # Create the directory if it doesn't exist
                            directory_path.mkdir(parents=True, exist_ok=True)

                            roo_res, roo_ws = self._roofit_mass(
                                level,
                                h_invmass,
                                ipt,
                                self.p_pdfnames,
                                self.p_param_names,
                                fitcfg,
                                roows,
                                f"roofit/mult_{multrange[0]}-{multrange[1]}/"
                                f"h_mass_fitted_pthf-{ptrange[0]}-{ptrange[1]}"
                                f"_{self.v_var2_binning}-{multrange[0]}-{multrange[1]}_{level}.png",
                            )
                            # if level == 'mc':
                            #     roo_ws.Print()
                            self.roo_ws[level][ipt] = roo_ws
                            self.roows[ipt] = roo_ws
                            if roo_res.status() == 0:
                                if level in ("data", "mc_sig"):
                                    self.fit_mean[level][ipt] = roo_ws.var(self.p_param_names["gauss_mean"]).getValV()
                                    self.fit_sigma[level][ipt] = roo_ws.var(self.p_param_names["gauss_sigma"]).getValV()
                                var_m = fitcfg.get("var", "m")
                                pdf_bkg = roo_ws.pdf(self.p_pdfnames["pdf_bkg"])
                                if pdf_bkg:
                                    self.fit_func_bkg[level][ipt] = pdf_bkg.asTF(roo_ws.var(var_m))
                                self.fit_range[level][ipt] = (
                                    roo_ws.var(var_m).getMin("fit"),
                                    roo_ws.var(var_m).getMax("fit"),
                                )
                            else:
                                self.logger.error("RooFit failed for %s bin %d", level, ipt)

                            if level == "data":
                                mean_sgn = roo_ws.var(self.p_param_names["gauss_mean"])
                                sigma_sgn = roo_ws.var(self.p_param_names["gauss_sigma"])
                                (sig, sig_err, _, _, signif, signif_err, s_over_b, s_over_b_err) = calc_signif(
                                    roo_ws, roo_res, self.p_pdfnames, self.p_param_names, mean_sgn, sigma_sgn
                                )

                                yieldshistos.SetBinContent(ipt + 1, sig)
                                yieldshistos.SetBinError(ipt + 1, sig_err)
                                meanhistos.SetBinContent(ipt + 1, mean_sgn.getVal())
                                meanhistos.SetBinError(ipt + 1, mean_sgn.getError())
                                sigmahistos.SetBinContent(ipt + 1, sigma_sgn.getVal())
                                sigmahistos.SetBinError(ipt + 1, sigma_sgn.getError())
                                signifhistos.SetBinContent(ipt + 1, signif)
                                signifhistos.SetBinError(ipt + 1, signif_err)
                                soverbhistos.SetBinContent(ipt + 1, s_over_b)
                                soverbhistos.SetBinError(ipt + 1, s_over_b_err)
                    fileout.cd()
                    yieldshistos.Write()
                    meanhistos.Write()
                    sigmahistos.Write()
                    signifhistos.Write()
                    soverbhistos.Write()
                fileout.Close()

    def get_efficiency(self, ibin1, ibin2):
        fileouteff = TFile.Open(f"{self.d_resultsallpmc}/efficiencies{self.case}{self.typean}.root", "read")
        h = fileouteff.Get(f"eff_mult{ibin2}")
        return h.GetBinContent(ibin1 + 1), h.GetBinError(ibin1 + 1)

    def efficiency(self):
        self.loadstyle()

        lfileeff = TFile.Open(self.n_fileff)
        fileouteff = TFile.Open(f"{self.d_resultsallpmc}/efficiencies{self.case}{self.typean}.root", "recreate")
        cEff = TCanvas("cEff", "The Fit Canvas")
        cEff.SetCanvasSize(1900, 1500)
        cEff.SetWindowSize(500, 500)
        cEff.SetLogy()

        legeff = TLegend(0.5, 0.20, 0.7, 0.45)
        legeff.SetBorderSize(0)
        legeff.SetFillColor(0)
        legeff.SetFillStyle(0)
        legeff.SetTextFont(42)
        legeff.SetTextSize(0.035)

        if self.signal_loss:
            cSl = TCanvas("cSl", "The Fit Canvas")
            cSl.SetCanvasSize(1900, 1500)
            cSl.SetWindowSize(500, 500)
            legsl = TLegend(0.5, 0.20, 0.7, 0.45)
            legsl.SetBorderSize(0)
            legsl.SetFillColor(0)
            legsl.SetFillStyle(0)
            legsl.SetTextFont(42)
            legsl.SetTextSize(0.035)

        for imult in range(self.p_nbin2):
            stringbin2 = "_{}_{:.2f}_{:.2f}".format(
                self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult]
            )
            legeffstring = "{:.1f} #leq {} < {:.1f}".format(
                self.lvar2_binmin[imult],
                self.p_latexbin2var,
                self.lvar2_binmax[imult],
            )

            if self.signal_loss:
                h_gen_pr_sl = lfileeff.Get("h_signal_loss_gen_pr" + stringbin2)
                h_sel_pr_sl = lfileeff.Get("h_signal_loss_rec_pr" + stringbin2)
                h_sel_pr_sl.Divide(h_sel_pr_sl, h_gen_pr_sl, 1.0, 1.0, "B")
                h_sel_pr_sl.SetLineColor(imult + 1)
                h_sel_pr_sl.SetMarkerColor(imult + 1)
                h_sel_pr_sl.SetMarkerStyle(21)
                cSl.cd()
                h_sel_pr_sl.Draw("same")
                fileouteff.cd()
                h_sel_pr_sl.SetName("signal_loss_pr_mult%d" % imult)
                h_sel_pr_sl.Write()

                legsl.AddEntry(h_sel_pr_sl, legeffstring, "LEP")
                h_sel_pr_sl.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
                h_sel_pr_sl.GetYaxis().SetTitle("Signal loss (prompt) %s" % (self.p_latexnhadron))
                h_sel_pr_sl.SetMinimum(0.7)
                h_sel_pr_sl.SetMaximum(1.0)

            h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
            h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")

            if self.signal_loss:
                h_sel_pr.Multiply(h_sel_pr_sl)

            h_sel_pr.SetLineColor(imult + 1)
            h_sel_pr.SetMarkerColor(imult + 1)
            h_sel_pr.SetMarkerStyle(21)
            cEff.cd()
            h_sel_pr.Draw("same")
            fileouteff.cd()
            h_sel_pr.SetName("eff_mult%d" % imult)
            h_sel_pr.Write()
            legeff.AddEntry(h_sel_pr, legeffstring, "LEP")
            h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s" % (self.p_latexnhadron))
            h_sel_pr.SetMinimum(0.0004)
            h_sel_pr.SetMaximum(0.4)

        if self.signal_loss:
            cSl.cd()
            legsl.Draw()
            cSl.SaveAs(f"{self.d_resultsallpmc}/SignalLoss{self.case}{self.typean}.eps")

            cSlFD = TCanvas("cSlFD", "The Fit Canvas")
            cSlFD.SetCanvasSize(1900, 1500)
            cSlFD.SetWindowSize(500, 500)
            legslFD = TLegend(0.5, 0.20, 0.7, 0.45)
            legslFD.SetBorderSize(0)
            legslFD.SetFillColor(0)
            legslFD.SetFillStyle(0)
            legslFD.SetTextFont(42)
            legslFD.SetTextSize(0.035)

        cEff.cd()
        legeff.Draw()
        cEff.SaveAs(f"{self.d_resultsallpmc}/Eff{self.case}{self.typean}.eps")

        cEffFD = TCanvas("cEffFD", "The Fit Canvas")
        cEffFD.SetCanvasSize(1900, 1500)
        cEffFD.SetWindowSize(500, 500)
        cEffFD.SetLogy()
        legeffFD = TLegend(0.5, 0.20, 0.7, 0.45)
        legeffFD.SetBorderSize(0)
        legeffFD.SetFillColor(0)
        legeffFD.SetFillStyle(0)
        legeffFD.SetTextFont(42)
        legeffFD.SetTextSize(0.035)

        for imult in range(self.p_nbin2):
            stringbin2 = "_{}_{:.2f}_{:.2f}".format(
                self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult]
            )
            legeffFDstring = "{:.1f} #leq {} < {:.1f}".format(
                self.lvar2_binmin[imult],
                self.p_latexbin2var,
                self.lvar2_binmax[imult],
            )

            if self.signal_loss:
                h_gen_fd_sl = lfileeff.Get("h_signal_loss_gen_fd" + stringbin2)
                h_sel_fd_sl = lfileeff.Get("h_signal_loss_rec_fd" + stringbin2)
                h_sel_fd_sl.Divide(h_sel_fd_sl, h_gen_fd_sl, 1.0, 1.0, "B")
                h_sel_fd_sl.SetLineColor(imult + 1)
                h_sel_fd_sl.SetMarkerColor(imult + 1)
                h_sel_fd_sl.SetMarkerStyle(21)
                cSlFD.cd()
                h_sel_fd_sl.Draw("same")
                fileouteff.cd()
                h_sel_fd_sl.SetName("signal_loss_fd_mult%d" % imult)
                h_sel_fd_sl.Write()

                legslFD.AddEntry(h_sel_fd_sl, legeffstring, "LEP")
                h_sel_fd_sl.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
                h_sel_fd_sl.GetYaxis().SetTitle("Signal loss (feeddown) %s" % (self.p_latexnhadron))
                h_sel_fd_sl.SetMinimum(0.7)
                h_sel_fd_sl.SetMaximum(1.0)

            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")

            if self.signal_loss:
                h_sel_fd.Multiply(h_sel_fd_sl)

            h_sel_fd.SetLineColor(imult + 1)
            h_sel_fd.SetMarkerColor(imult + 1)
            h_sel_fd.SetMarkerStyle(21)
            cEffFD.cd()
            h_sel_fd.Draw("same")
            fileouteff.cd()
            h_sel_fd.SetName("eff_fd_mult%d" % imult)
            h_sel_fd.Write()
            legeffFD.AddEntry(h_sel_fd, legeffFDstring, "LEP")
            h_sel_fd.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
            h_sel_fd.GetYaxis().SetTitle("Acc x efficiency feed-down %s" % (self.p_latexnhadron))
            h_sel_fd.SetMinimum(0.0004)
            h_sel_fd.SetMaximum(0.4)

        cEffFD.cd()
        legeffFD.Draw()
        cEffFD.SaveAs(f"{self.d_resultsallpmc}/EffFD{self.case}{self.typean}.eps")
        if self.signal_loss:
            cSlFD.cd()
            legslFD.Draw()
            cSlFD.SaveAs(f"{self.d_resultsallpmc}/SignalLossFD{self.case}{self.typean}.eps")

    def plotter(self):
        gROOT.SetBatch(True)
        self.loadstyle()

        fileouteff = TFile.Open(f"{self.d_resultsallpmc}/efficiencies{self.case}{self.typean}.root")
        yield_filename = self.make_file_path(
            self.d_resultsallpdata, self.yields_filename, "root", None, [self.case, self.typean]
        )
        fileoutyield = TFile.Open(yield_filename, "READ")
        fileoutcross = TFile.Open(f"{self.d_resultsallpdata}/finalcross{self.case}{self.typean}.root", "recreate")

        cCrossvsvar1 = TCanvas("cCrossvsvar1", "The Fit Canvas")
        cCrossvsvar1.SetCanvasSize(1900, 1500)
        cCrossvsvar1.SetWindowSize(500, 500)
        cCrossvsvar1.SetLogy()

        legvsvar1 = TLegend(0.5, 0.65, 0.7, 0.85)
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
            hcross.SetLineColor(imult + 1)
            norm = 2 * self.p_br * self.p_nevents / (self.p_sigmamb * 1e12)
            hcross.Scale(1.0 / norm)
            fileoutcross.cd()
            hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnhadron)
            hcross.GetYaxis().SetTitle(f"d#sigma/d#it{{p}}_{{T}} ({self.p_latexnhadron}) {self.typean}")
            hcross.SetName("hcross%d" % imult)
            hcross.GetYaxis().SetRangeUser(1e1, 1e10)
            legvsvar1endstring = "{:.1f} < {} < {:.1f}".format(
                self.lvar2_binmin[imult],
                self.p_latexbin2var,
                self.lvar2_binmax[imult],
            )
            legvsvar1.AddEntry(hcross, legvsvar1endstring, "LEP")
            hcross.Draw("same")
            hcross.Write()
            listvalpt = [hcross.GetBinContent(ipt + 1) for ipt in range(self.p_nptbins)]
            listvalues.append(listvalpt)
            listvalerrpt = [hcross.GetBinError(ipt + 1) for ipt in range(self.p_nptbins)]
            listvalueserr.append(listvalerrpt)
        legvsvar1.Draw()
        cCrossvsvar1.SaveAs(f"{self.d_resultsallpdata}/Cross{self.case}{self.typean}Vs{self.v_var_binning}.eps")

        cCrossvsvar2 = TCanvas("cCrossvsvar2", "The Fit Canvas")
        cCrossvsvar2.SetCanvasSize(1900, 1500)
        cCrossvsvar2.SetWindowSize(500, 500)
        cCrossvsvar2.SetLogy()

        legvsvar2 = TLegend(0.5, 0.65, 0.7, 0.85)
        legvsvar2.SetBorderSize(0)
        legvsvar2.SetFillColor(0)
        legvsvar2.SetFillStyle(0)
        legvsvar2.SetTextFont(42)
        legvsvar2.SetTextSize(0.035)
        hcrossvsvar2 = [
            TH1F("hcrossvsvar2" + "pt%d" % ipt, "", self.p_nbin2, array("d", self.var2ranges))
            for ipt in range(self.p_nptbins)
        ]

        for ipt in range(self.p_nptbins):
            print("pt", ipt)
            for imult in range(self.p_nbin2):
                hcrossvsvar2[ipt].SetLineColor(ipt + 1)
                hcrossvsvar2[ipt].GetXaxis().SetTitle("%s" % self.p_latexbin2var)
                hcrossvsvar2[ipt].GetYaxis().SetTitle(self.p_latexnhadron)
                hcrossvsvar2[ipt].SetBinContent(imult + 1, listvalues[imult][ipt])
                hcrossvsvar2[ipt].SetBinError(imult + 1, listvalueserr[imult][ipt])

                hcrossvsvar2[ipt].GetYaxis().SetRangeUser(1e4, 1e10)
            legvsvar2endstring = "{:.1f} < {} < {:.1f} GeV/#it{{c}}".format(
                self.lpt_finbinmin[ipt],
                "#it{p}_{T}",
                self.lpt_finbinmax[ipt],
            )
            hcrossvsvar2[ipt].Draw("same")
            legvsvar2.AddEntry(hcrossvsvar2[ipt], legvsvar2endstring, "LEP")
        legvsvar2.Draw()
        cCrossvsvar2.SaveAs(f"{self.d_resultsallpdata}/Cross{self.case}{self.typean}Vs{self.v_var2_binning}.eps")

    @staticmethod
    def calculate_norm(logger, hevents, hselevents):  # TO BE FIXED WITH EV SEL
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
        yield_filename = self.make_file_path(
            self.d_resultsallpdata, self.yields_filename, "root", None, [self.case, self.typean]
        )

        for imult in range(self.p_nbin2):
            # Choose where efficiencies to take from. Either this mult. bin, another mult. bin
            # in this analysis or another mult. bin from another analysis specified explicitly
            # by the user.
            fileouteff = (
                f"{self.d_resultsallpmc}/efficiencies{self.case}{self.typean}.root"
                if not self.path_file_eff[imult]
                else self.path_file_eff[imult]
            )
            if not os.path.exists(fileouteff):
                self.logger.fatal("Efficiency file %s could not be found", fileouteff)
            bineff = -1
            if self.mult_bin_eff[imult] is not None:
                bineff = self.mult_bin_eff[imult]
                print(f"Use efficiency from bin {bineff} from file {fileouteff}")
            elif self.p_bineff is None:
                bineff = imult
                print("Using efficiency for each var2 bin")
            else:
                bineff = self.p_bineff
                print(f"Using efficiency always from bin={bineff}")

            filemass = TFile.Open(self.n_filemass)
            histonorm = filemass.Get("histonorm")

            namehistoeffprompt = f"eff_mult{bineff}"
            namehistoefffeed = f"eff_fd_mult{bineff}"
            nameyield = "hyields%d" % imult
            fileoutcrossmult = "%s/finalcross%s%smult%d.root" % (self.d_resultsallpdata, self.case, self.typean, imult)

            # Bin1 is all events. Bin2 is all sel events. Mult bins start from Bin3.
            norm = histonorm.GetBinContent(imult + 3)
            # pylint: disable=logging-not-lazy
            self.logger.warning("Number of events %d for mult bin %d" % (norm, imult))

            if self.p_fprompt_from_mb:
                if imult == 0:
                    fileoutcrossmb = "{}/finalcross{}{}mult0.root".format(
                        self.d_resultsallpdata, self.case, self.typean
                    )
                    output_prompt = []
                    if self.p_nevents is not None:
                        norm = self.p_nevents
                    self.logger.warning("Corrected Number of events %d for mult bin %d" % (norm, imult))
                    hf_pt_spectrum(
                        self.p_anahpt,
                        self.p_br,
                        self.p_inputfonllpred,
                        self.p_fd_method,
                        None,
                        fileouteff,
                        namehistoeffprompt,
                        namehistoefffeed,
                        yield_filename,
                        nameyield,
                        norm,
                        self.p_sigmamb,
                        output_prompt,
                        fileoutcrossmb,
                    )
                else:
                    # filecrossmb = TFile.Open("%s/finalcross%s%smult0.root" %  \
                    # (self.d_resultsallpdata, self.case, self.typean), "recreate")
                    self.logger.info("Calculating spectra using fPrompt from MB. Assuming MB is bin 0")
                    self.p_fd_method = "ext"
                    hf_pt_spectrum(
                        self.p_anahpt,
                        self.p_br,
                        self.p_inputfonllpred,
                        self.p_fd_method,
                        output_prompt,
                        fileouteff,
                        namehistoeffprompt,
                        namehistoefffeed,
                        yield_filename,
                        nameyield,
                        norm,
                        self.p_sigmamb,
                        output_prompt,
                        fileoutcrossmult,
                    )
            else:
                hf_pt_spectrum(
                    self.p_anahpt,
                    self.p_br,
                    self.p_inputfonllpred,
                    self.p_fd_method,
                    None,
                    fileouteff,
                    namehistoeffprompt,
                    namehistoefffeed,
                    yield_filename,
                    nameyield,
                    norm,
                    self.p_sigmamb,
                    output_prompt,
                    fileoutcrossmult,
                )

        fileoutcrosstot = TFile.Open(
            f"{self.d_resultsallpdata}/finalcross{self.case}{self.typean}multtot.root", "recreate"
        )

        for imult in range(self.p_nbin2):
            fileoutcrossmult = "%s/finalcross%s%smult%d.root" % (self.d_resultsallpdata, self.case, self.typean, imult)
            f_fileoutcrossmult = TFile.Open(fileoutcrossmult)
            if not f_fileoutcrossmult:
                continue
            hcross = f_fileoutcrossmult.Get("hptspectrum")
            hcross.SetName("hptspectrum%d" % imult)
            fileoutcrosstot.cd()
            hcross.Write()
        fileoutcrosstot.Close()

    def plotternormyields(self):
        gROOT.SetBatch(True)
        cCrossvsvar1 = TCanvas("cCrossvsvar1", "The Fit Canvas")
        cCrossvsvar1.SetCanvasSize(1900, 1500)
        cCrossvsvar1.SetWindowSize(500, 500)
        cCrossvsvar1.SetLogy()
        cCrossvsvar1.cd()
        legvsvar1 = TLegend(0.5, 0.65, 0.7, 0.85)
        legvsvar1.SetBorderSize(0)
        legvsvar1.SetFillColor(0)
        legvsvar1.SetFillStyle(0)
        legvsvar1.SetTextFont(42)
        legvsvar1.SetTextSize(0.035)
        fileoutcrosstot = TFile.Open(f"{self.d_resultsallpdata}/finalcross{self.case}{self.typean}multtot.root")

        for imult in range(self.p_nbin2):
            hcross = fileoutcrosstot.Get("histoSigmaCorr%d" % imult)
            hcross.Scale(1.0 / (self.p_sigmamb * 1e12))
            hcross.SetLineColor(imult + 1)
            hcross.SetMarkerColor(imult + 1)
            hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnhadron)
            hcross.GetYaxis().SetTitleOffset(1.3)
            hcross.GetYaxis().SetTitle(f"Corrected yield/events ({self.p_latexnhadron}) {self.typean}")
            hcross.GetYaxis().SetRangeUser(1e-10, 1)
            legvsvar1endstring = "{:.1f} #leq {} < {:.1f}".format(
                self.lvar2_binmin[imult],
                self.p_latexbin2var,
                self.lvar2_binmax[imult],
            )
            legvsvar1.AddEntry(hcross, legvsvar1endstring, "LEP")
            hcross.Draw("same")
        legvsvar1.Draw()
        cCrossvsvar1.SaveAs(
            "{}/CorrectedYieldsNorm{}{}Vs{}.eps".format(
                self.d_resultsallpdata, self.case, self.typean, self.v_var_binning
            )
        )
        fileoutcrosstot.Close()
