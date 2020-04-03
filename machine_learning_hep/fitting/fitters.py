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
Definition of FitterBase, Fitter and SystFitter classes
Definition of FitBase, FitAliHF, FitROOT classes
"""

# pylint: disable=too-many-lines

from copy import deepcopy
from array import array
from math import sqrt

# pylint: disable=import-error, no-name-in-module, unused-import
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils, AliHFInvMassMultiTrialFit
from ROOT import TFile, TH1F, TH1D, TF1, TPaveText, TLine, TLegend, Double, TLatex
from ROOT import kBlue, kRed, kGreen, kMagenta, kOrange, kPink, kCyan, kYellow, kBlack

from machine_learning_hep.logger import get_logger

# pylint: disable=too-many-instance-attributes
class FitBase:
    """
    Common base class for FitAliHF and FitROOT.
    """

    # pylint: disable=unused-argument
    def __init__(self, init_pars, **kwargs):
        self.logger = get_logger()
        # If nit/fitting attempt was made
        self.has_attempt = False
        # Whether fitting was successful in the end
        self.success = False
        # Initialisation parameters
        self.user_init_pars = deepcopy(init_pars)
        self.init_pars = None
        # Default init parameters (to be modified for deriving classes)
        self.default_init_pars = {"mean": None,
                                  "fix_mean": False,
                                  "sigma": None,
                                  "fix_sigma": False,
                                  "rebin": None,
                                  "fit_range_low": None,
                                  "fit_range_up": None,
                                  "likelihood": True,
                                  "n_sigma_sideband": None,
                                  "sig_func_name": None,
                                  "bkg_func_name": None}
        # Fitted parameters (to be modified for deriving classes)
        self.fit_pars = {}
        # The fit kernel
        self.kernel = None


    def make_default_init_pars(self):
        """
        Small wrapper for constructing default inititalisation parameters
        Returns:
            Dictionary of default initialisation parameters
        """

        return deepcopy(self.default_init_pars)


    def get_fit_pars(self):
        """
        Small wrapper providing deep copy of fit parameters
        Returns:
            Dictionary of fitted parameters
        """

        return deepcopy(self.fit_pars)

    def override_init_pars(self, **init_pars):
        for par, val in init_pars.items():
            if par in self.user_init_pars:
                self.user_init_pars[par] = val


    def init_fit(self):
        """
        Common initialisation steps
        Returns:
            Success status
        """

        self.logger.info("Init fit")

        # Potentially found a fit to initialise from
        self.init_pars = self.make_default_init_pars()

        # Collect key which haven't changed
        #pars_not_changed = []
        for k in list(self.init_pars.keys()):
            if k in self.user_init_pars:
                self.init_pars[k] = self.user_init_pars.pop(k)
        #        continue
        #    pars_not_changed.append(k)

        #self.logger.debug("Following default parameters are used")
        #for p in pars_not_changed:
            #print(p)

        return True


    def init_kernel(self):
        """
        Initialize the fit kernel. To be overwritten by the deriving class
        """

        self.logger.debug("Init kernel")
        return True


    def fit_kernel(self):
        """
        Fit the fit kernel. To be overwritten by the deriving class
        """

        self.logger.debug("Fit kernel")
        return True


    def set_fit_pars(self):
        """
        Set final fitted parameters. To be overwritten by the deriving class
        """


    def fit(self):
        """
        Initialize and fit. This is common and not to be overwritten by a deriving class
        """

        if self.has_attempt:
            self.logger.info("Was already fitted. Skip...")
            return
        if self.init_fit():
            if self.init_kernel():
                self.success = self.fit_kernel()
                if self.success:
                    self.set_fit_pars()
        self.has_attempt = True


    def draw(self, root_pad, title=None, x_axis_label=None, y_axis_label=None, **draw_args):
        """
        Draw this fit. This is common and not to be overwritten by a deriving class. Arguments
        are forwarded to draw_kernel after common sanity checks
        Args:
            root_pad: a TVirtualPad to draw the fit in
            title: title in the root_pad
            x_axis_label: ...
            y_axis_label: ...
            draw_args: dictionary for further arguments used for drawing.
        """

        # Keep like this to be able to insert common draw procedure here
        if not self.has_attempt:
            self.logger.info("Fit not done yet, nothing to draw. Skip...")
            return
        if not self.success:
            pinfos = self.add_pave_helper_(0.12, 0.7, 0.47, 0.89, "NDC")
            self.add_text_helper_(pinfos, "FIT FAILED", kRed + 2)
            if "add_root_objects" in draw_args:
                draw_args["add_root_objects"].append(pinfos)
            else:
                draw_args["add_root_objects"] = [pinfos]

        self.draw_kernel(root_pad, title=title, x_axis_label=x_axis_label,
                         y_axis_label=y_axis_label, **draw_args)


    # pylint: disable=unused-argument, dangerous-default-value
    def draw_kernel(self, root_pad, root_objects=[], **draw_args):
        """
        Draw method specific to the used kernel. To be overwritten by the derivin class
        Args:
            root_pad: a TVirtualPad to draw the fit in
            root_objects: list to collect further internally created ROOT objects such that they
                          would not be deleted before the fit has been saved
            draw_args: dictionary for further arguments used for drawing.
        """

        self.logger.debug("Draw kernel")


    @staticmethod
    def add_text_helper_(pave, line, color=None):
        """
        Helper to put a text line into a TPave object
        Args:
            pave: ROOT TPave object
            line: string to be added
            color (optional): Color of the text
        """

        text = pave.AddText(line)
        text.SetTextAlign(11)
        text.SetTextSize(0.024)
        text.SetTextFont(42)
        if color:
            text.SetTextColor(color)

    @staticmethod
    def add_pave_helper_(x_min, y_min, x_max, y_max, opt="NDC"):
        """
        Helper to create a TPave object
        Args:
            x_min, ...: Relative coordinates within the ROOT TVirtualPad
            opt: further options passed to the constructor of TPave
        Returns:
            A TPave object
        """

        pave = TPaveText(x_min, y_min, x_max, y_max, opt)
        pave.SetBorderSize(0)
        pave.SetFillStyle(0)
        pave.SetMargin(0.)
        return pave

# pylint: enable=too-many-instance-attributes


class FitROOT(FitBase):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_objects = None


    def set_root_objects(self, root_objects):
        self.root_objects = root_objects
        self.update_root_objects()

    def update_root_objects(self):
        pass


    def __str__(self):
        string = f"--------------------------------\n" \
                 f"Class: {self.__class__.__name__}\n" \
                 f"Kernel: {self.kernel.__class__.__name__}, {self.kernel}\n" \
                 f"Init parameters:\n"
        string += str(self.init_pars)
        string += "\nROOT objects\n"
        for name, obj in self.root_objects.items():
            string += f"\tName: {name}, object: {obj}\n"
        string += "--------------------------------"
        return string


class FitAliHF(FitROOT):
    """
    Class with AliHFMassFitter as core fitting utility
    """
    def __init__(self, *args, histo=None, histo_mc=None, histo_reflections=None, **base_args):
        super().__init__(*args, **base_args)
        self.histo = histo
        self.histo_mc = histo_mc
        self.histo_reflections = histo_reflections
        # AliHF fitter

        self.default_init_pars = {"mean": None,
                                  "fix_mean": False,
                                  "sigma": None,
                                  "fix_sigma": False,
                                  "include_sec_peak": False,
                                  "sec_mean": None,
                                  "fix_sec_mean": False,
                                  "sec_sigma": None,
                                  "fix_sec_sigma": False,
                                  "use_sec_peak_rel_sigma": True,
                                  "include_reflections": False,
                                  "fix_reflections_s_over_b": True,
                                  "rebin": None,
                                  "fit_range_low": None,
                                  "fit_range_up": None,
                                  "likelihood": True,
                                  "n_sigma_sideband": None,
                                  "rel_sigma_bound": None,
                                  "sig_func_name": None,
                                  "bkg_func_name": None}
        # Fitted parameters (to be modified for deriving classes)
        # Only those corresponding to init parameters are here. Specific parameters/values
        # provided by the kernel have to be extracted from that directly.
        self.fit_pars = {"mean": None,
                         "sigma": None}
        self.update_root_objects()


    def update_root_objects(self):
        if self.root_objects is None:
            self.root_objects = {}
        self.histo = self.root_objects.get("histo", self.histo)
        self.histo_mc = self.root_objects.get("histo_mc", self.histo_mc)
        self.histo_reflections = self.root_objects.get("histo_reflections", self.histo_reflections)

        self.root_objects["histo"] = self.histo
        self.root_objects["histo_mc"] = self.histo_mc
        self.root_objects["histo_reflections"] = self.histo_reflections


    def init_kernel(self):

        self.update_root_objects()

        if self.init_pars["rebin"]:
            histo_rebin_ = AliVertexingHFUtils.RebinHisto(self.histo, self.init_pars["rebin"], -1)
            self.histo = TH1F()
            histo_rebin_.Copy(self.histo)
            self.histo.SetName(f"{self.histo.GetName()}_fit_histo")
        else:
            self.histo = self.histo.Clone(f"{self.histo.GetName()}_fit_histo")


        self.kernel = AliHFInvMassFitter(self.histo,
                                         self.init_pars["fit_range_low"],
                                         self.init_pars["fit_range_up"],
                                         self.init_pars["bkg_func_name"],
                                         self.init_pars["sig_func_name"])
        self.kernel.SetCheckSignalCountsAfterFirstFit(False)
        if self.init_pars["likelihood"]:
            self.kernel.SetUseLikelihoodFit()
        self.kernel.SetInitialGaussianMean(self.init_pars["mean"])
        self.kernel.SetInitialGaussianSigma(self.init_pars["sigma"])
        self.kernel.SetNSigma4SideBands(self.init_pars["n_sigma_sideband"])
        if self.init_pars["fix_sigma"]:
            self.kernel.SetFixGaussianSigma(self.init_pars["sigma"])

        if self.init_pars["include_reflections"]:

            self.histo_reflections = AliVertexingHFUtils.AdaptTemplateRangeAndBinning( \
                    self.histo_reflections, self.histo, self.init_pars["fit_range_low"],
                    self.init_pars["fit_range_up"])

            if not self.init_pars["fix_reflections_s_over_b"]:
                self.kernel.SetTemplateReflections(self.histo_reflections, "1gaus",
                                                   self.init_pars["fit_range_low"],
                                                   self.init_pars["fit_range_up"])
            else:
                if self.init_pars["rebin"]:
                    histo_mc_rebin_ = AliVertexingHFUtils.RebinHisto(self.histo_mc,
                                                                     self.init_pars["rebin"], -1)

                    self.histo_mc = TH1F()
                    histo_mc_rebin_.Copy(self.histo_mc)
                    self.histo_mc.SetName(f"{self.histo_mc.GetName()}_fit_histo")
                else:
                    self.histo_mc = self.histo_mc.Clone(f"{self.histo_mc.GetName()}_fit_histo")

                r_over_s = self.histo_mc.Integral(
                    self.histo_mc.FindBin(self.init_pars["fit_range_low"]),
                    self.histo_mc.FindBin(self.init_pars["fit_range_up"]))
                if r_over_s > 0.:
                    r_over_s = self.histo_reflections.Integral(
                        self.histo_reflections.FindBin(self.init_pars["fit_range_low"]),
                        self.histo_reflections.FindBin(self.init_pars["fit_range_up"])) / r_over_s
                    self.kernel.SetTemplateReflections(self.histo_reflections, "1gaus",
                                                       self.init_pars["fit_range_low"],
                                                       self.init_pars["fit_range_up"])
                    self.kernel.SetFixReflOverS(r_over_s)

        if self.init_pars["include_sec_peak"]:
            sec_sigma = self.init_pars["sigma"] * self.init_pars["sec_sigma"] \
                    if self.init_pars["use_sec_peak_rel_sigma"] \
                    else self.init_pars["sec_sigma"]
            self.kernel.IncludeSecondGausPeak(self.init_pars["sec_mean"],
                                              self.init_pars["fix_sec_mean"],
                                              sec_sigma,
                                              self.init_pars["fix_sec_sigma"])

        return True


    def fit_kernel(self):
        success = self.kernel.MassFitter(False)
        if success:
            if self.kernel.GetRawYield() < 0.:
                return False
            if self.init_pars["rel_sigma_bound"]:
                fit_sigma = self.kernel.GetSigma()
                min_sigma = (1 - self.init_pars["rel_sigma_bound"]) * self.init_pars["sigma"]
                max_sigma = (1 + self.init_pars["rel_sigma_bound"]) * self.init_pars["sigma"]
                return min_sigma < fit_sigma < max_sigma
        return success


    def set_fit_pars(self):
        self.fit_pars["mean"] = self.kernel.GetMean()
        self.fit_pars["sigma"] = self.kernel.GetSigma()


    #pylint: disable=too-many-locals, too-many-statements, dangerous-default-value
    def draw_kernel(self, root_pad, root_objects=[], **draw_args):

        n_sigma_signal = draw_args.pop("sigma_signal", 3)
        mean_dim = draw_args.pop("mean_dim", "GeV/#it{c}^{2}")
        mean_scale = draw_args.pop("mean_scale", 1.)
        sigma_dim = draw_args.pop("sigma_dim", "MeV/#it{c}^{2}")
        sigma_scale = draw_args.pop("sigma_scale", 1000.)

        add_root_objects = draw_args.pop("add_root_objects", None)

        # Now comes some styling
        color_sig = kBlue - 3
        color_bkg_refit = kRed + 2
        color_refl = kGreen + 2
        color_sec_peak = kMagenta + 3
        self.histo.SetMarkerStyle(20)

        draw_objects = [self.histo]
        draw_options = ["PE"] + [""] * 3
        sig_func = self.kernel.GetMassFunc()
        if sig_func:
            # Might be nullptr
            sig_func.SetLineColor(color_sig)
            draw_objects.append(sig_func)

        bkg_func = self.kernel.GetBackgroundFullRangeFunc()
        if bkg_func:
            # Might be nullptr
            draw_objects.append(bkg_func)

        bkg_refit_func = self.kernel.GetBackgroundRecalcFunc()
        if bkg_refit_func:
            # Might be nullptr
            draw_objects.append(bkg_refit_func)
            bkg_refit_func.SetLineColor(color_bkg_refit)
        refl_func = self.kernel.GetReflFunc() if self.init_pars["include_reflections"] else None
        if refl_func:
            # Could either be None or a nullptr
            draw_objects.append(refl_func)
            draw_options.append("")
        sec_peak_func = self.kernel.GetSecondPeakFunc() \
                if self.init_pars["include_sec_peak"] else None
        if sec_peak_func:
            # Could either be None or a nullptr
            draw_objects.append(sec_peak_func)
            draw_options.append("")


        y_plot_max = self.histo.GetMaximum()
        y_plot_min = self.histo.GetMinimum()
        for i in range(1, self.histo.GetNbinsX() + 1):
            y_max_tmp = self.histo.GetBinContent(i) + self.histo.GetBinError(i)
            y_min_tmp = self.histo.GetBinContent(i) - self.histo.GetBinError(i)
            y_plot_max = max(y_plot_max, y_max_tmp)
            y_plot_min = min(y_plot_min, y_min_tmp)

        for do in draw_objects:
            y_plot_max = max(y_plot_max, do.GetMaximum())
            y_plot_min = min(y_plot_min, do.GetMinimum())
        # Leave some space for putting info
        y_rel_plot_range = 0.6
        y_rel_header_range = 0.3
        y_rel_footer_range = 0.1

        y_full_range = (y_plot_max - y_plot_min) / y_rel_plot_range
        y_min = y_plot_min - y_rel_footer_range * y_full_range
        y_max = y_plot_max + y_rel_header_range * y_full_range

        root_pad.SetLeftMargin(0.12)
        frame = root_pad.cd().DrawFrame(self.init_pars["fit_range_low"], y_min,
                                        self.init_pars["fit_range_up"], y_max,
                                        f"{draw_args.pop('title', '')} ; " \
                                        f"{draw_args.pop('x_axis_label', '')} ; " \
                                        f"{draw_args.pop('y_axis_label', '')}")

        frame.GetYaxis().SetTitleOffset(1.7)
        frame.GetYaxis().SetMaxDigits(4)


        sig = self.kernel.GetRawYield()
        sig_err = self.kernel.GetRawYieldError()
        bkg = Double()
        bkg_err = Double()
        self.kernel.Background(n_sigma_signal, bkg, bkg_err)
        signif = Double()
        signif_err = Double()
        self.kernel.Significance(n_sigma_signal, signif, signif_err)
        sig_o_bkg = sig / bkg if bkg > 0. else -1.

        root_objects.append(self.add_pave_helper_(0.15, 0.7, 0.48, 0.89, "NDC"))
        self.add_text_helper_(root_objects[-1], f"S = {sig:.0f} #pm {sig_err:.0f}")
        self.add_text_helper_(root_objects[-1],
                              f"B({n_sigma_signal}#sigma) = {bkg:.0f} " \
                              f"#pm {bkg_err:.0f}")
        self.add_text_helper_(root_objects[-1], f"S/B({n_sigma_signal}#sigma) = {sig_o_bkg:.4f}")
        self.add_text_helper_(root_objects[-1],
                              f"Signif({n_sigma_signal}#sigma) = " \
                              f"{signif:.1f} #pm {signif_err:.1f}")
        root_objects[-1].Draw()

        root_objects.append(self.add_pave_helper_(0.55, 0.75, 0.89, 0.89, "NDC"))
        self.add_text_helper_(root_objects[-1],
                              f"#chi/ndf = {self.kernel.GetReducedChiSquare():.4f}", color_sig)
        self.add_text_helper_(root_objects[-1],
                              f"#mu = {self.kernel.GetMean()*mean_scale:.4f} " \
                              f"#pm " \
                              f"{self.kernel.GetMeanUncertainty()*mean_scale:.4f} " \
                              f"{mean_dim}", color_sig)
        self.add_text_helper_(root_objects[-1],
                              f"#sigma = " \
                              f"{self.kernel.GetSigma()*sigma_scale:.4f} " \
                              f"#pm " \
                              f"{self.kernel.GetSigmaUncertainty()*sigma_scale:.4f} " \
                              f"{sigma_dim}", color_sig)
        root_objects[-1].Draw()

        x_min_add = 0.45
        y_min_tmp = 0.11
        y_delta = 0.05
        if sec_peak_func:
            sec_peak_func.SetLineColor(color_sec_peak)
            sec_mean = sec_peak_func.GetParameter(1)
            sec_sigma = sec_peak_func.GetParameter(2)
            root_objects.append(self.add_pave_helper_(x_min_add, y_min_tmp, 0.89,
                                                      y_min_tmp + y_delta, "NDC"))
            self.add_text_helper_(root_objects[-1], f"#mu_{{sec}} = {sec_mean*mean_scale:.4f} " \
                                                    f"{mean_dim}, #sigma_{{sec}} = " \
                                                    f"{sec_sigma*sigma_scale:.4f} " \
                                                    f"{sigma_dim}", color_sec_peak)
            root_objects[-1].Draw()
            y_min_tmp += y_delta
        if refl_func:
            refl_func.SetLineColor(color_refl)
            refl = self.kernel.GetReflOverSig()
            refl_err = self.kernel.GetReflOverSigUncertainty()
            root_objects.append(self.add_pave_helper_(x_min_add, y_min_tmp, 0.89,
                                                      y_min_tmp + y_delta, "NDC"))
            self.add_text_helper_(root_objects[-1], f"Refl/S = {refl:.4f} #pm {refl_err:.4f}",
                                  color_refl)
            root_objects[-1].Draw()
            y_min_tmp += y_delta

        for dob, dop in zip(draw_objects, draw_options):
            dob.Draw(f"same {dop}")

        if add_root_objects:
            for aro in add_root_objects:
                root_objects.append(aro)
                aro.Draw("same")


class FitROOTGauss(FitROOT):
    """
    Class with specific ROOT TF1 as core fitting utility
    """
    def __init__(self, *args, histo=None, **base_args):
        super().__init__(*args, **base_args)
        self.histo = histo

        self.default_init_pars = {"mean": None,
                                  "sigma": None,
                                  "rebin": None,
                                  "use_user_fit_range": False,
                                  "fit_range_low": None,
                                  "fit_range_up": None,
                                  "n_rms_fix": None,
                                  "n_rms_start": 3,
                                  "n_rms_stop": 8,
                                  "likelihood": False}
        # Fitted parameters (to be modified for deriving classes)
        # Only those corresponding to init parameters are here. Specific parameters/values
        # provided by the kernel have to be extracted from that directly.
        self.fit_pars = {"mean": None,
                         "sigma": None}

        self.update_root_objects()

    def update_root_objects(self):
        if self.root_objects is None:
            self.root_objects = {}
        self.histo = self.root_objects.get("histo", self.histo)
        self.root_objects["histo"] = self.histo


    def init_kernel(self):

        self.update_root_objects()

        if self.init_pars["rebin"]:
            histo_rebin_ = AliVertexingHFUtils.RebinHisto(self.histo, self.init_pars["rebin"], -1)
            self.histo = TH1F()
            histo_rebin_.Copy(self.histo)

        # If only a specific number of RMS should be considered for the fit range
        if self.init_pars["n_rms_fix"]:
            self.init_pars["n_rms_start"] = self.init_pars["n_rms_fix"]
            self.init_pars["n_rms_stop"] = self.init_pars["n_rms_fix"]

        if self.init_pars["n_rms_start"] > self.init_pars["n_rms_stop"]:
            self.logger.fatal("Stop fit range of MC fit is < start, start: %i, stop: %i",
                              self.init_pars["n_rms_start"], self.init_pars["n_rms_stop"])
        return True


    def fit_kernel_(self, mean_init, sigma_init, int_init, fit_range_low, fit_range_up):
        fit_func = TF1("fit_func", "gaus", fit_range_low, fit_range_up)
        fit_func.SetParameter(0, int_init)
        fit_func.SetParameter(1, mean_init)
        fit_func.SetParameter(2, sigma_init)
        fit_string = "BE0+"
        if self.init_pars["likelihood"]:
            fit_string += "L"
        self.histo.Fit(fit_func, fit_string, "", fit_range_low, fit_range_up)

        int_fit = fit_func.GetParameter(0)
        mean_fit = fit_func.GetParameter(1)
        sigma_fit = fit_func.GetParameter(2)
        chi2ndf = fit_func.GetNDF()
        chi2ndf = fit_func.GetChisquare() / chi2ndf if chi2ndf > 0. else 0.

        success = True
        if int_fit * sigma_fit < 0. \
                or mean_init - sigma_init > mean_fit or mean_fit > mean_init + sigma_init \
                or 1.1 * sigma_init < sigma_fit or chi2ndf <= 0.:
            success = False

        return fit_func, success


    def fit_kernel(self):
        guess_mean = self.histo.GetMean()
        guess_sigma = self.histo.GetRMS()

        if self.init_pars["use_user_fit_range"]:
            guess_int = self.histo.Integral(self.histo.FindBin(self.init_pars["fit_range_low"]),
                                            self.histo.FindBin(self.init_pars["fit_range_up"]),
                                            "width")
            self.kernel, success = self.fit_kernel_(guess_mean, guess_sigma, guess_int,
                                                    self.init_pars["fit_range_low"],
                                                    self.init_pars["fit_range_up"])
            return success

        for r in range(self.init_pars["n_rms_start"], self.init_pars["n_rms_stop"] + 1):
            guess_fit_range_low = guess_mean - r * guess_sigma
            guess_fit_range_up = guess_mean + r * guess_sigma
            guess_int = self.histo.Integral(self.histo.FindBin(guess_fit_range_low),
                                            self.histo.FindBin(guess_fit_range_up),
                                            "width")
            self.kernel, success = self.fit_kernel_(guess_mean, guess_sigma, guess_int,
                                                    guess_fit_range_low, guess_fit_range_up)
            if success:
                return success

        return False


    def set_fit_pars(self):
        self.fit_pars["mean"] = self.kernel.GetParameter(1)
        self.fit_pars["sigma"] = self.kernel.GetParameter(2)


    #pylint: disable=dangerous-default-value
    def draw_kernel(self, root_pad, root_objects=[], **draw_args):

        title = draw_args.pop("title", "")
        x_axis_label = draw_args.pop("x_axis_label", "")
        y_axis_label = draw_args.pop("y_axis_label", "")
        mean_dim = draw_args.pop("mean_dim", "GeV/#it{c}^{2}")
        mean_scale = draw_args.pop("mean_scale", 1.)
        sigma_dim = draw_args.pop("sigma_dim", "MeV/#it{c}^{2}")
        sigma_scale = draw_args.pop("sigma_scale", 1000.)

        add_root_objects = draw_args.pop("add_root_objects", None)

        if draw_args:
            self.logger.warning("There are unknown draw arguments")

        root_pad.cd()

        draw_objects = [self.histo, self.kernel]

        x_min = self.init_pars["fit_range_low"]
        x_max = self.init_pars["fit_range_up"]
        y_min = 0.
        y_max = self.histo.GetMaximum() * 1.8

        # Now comes the styling
        color_sig = kBlue - 3
        self.histo.SetMarkerStyle(20)
        self.kernel.SetLineColor(color_sig)

        root_pad.SetLeftMargin(0.12)
        frame = root_pad.cd().DrawFrame(x_min, y_min, x_max, y_max,
                                        f"{title} ; {x_axis_label} ; {y_axis_label}")

        frame.GetYaxis().SetTitleOffset(1.7)
        frame.GetYaxis().SetMaxDigits(4)

        if draw_args:
            self.logger.warning("There are unknown draw arguments")

        self.histo.SetTitle(title)
        self.histo.GetXaxis().SetTitle(x_axis_label)
        self.histo.GetYaxis().SetTitle(y_axis_label)

        red_chisqu = self.kernel.GetNDF()
        red_chisqu = self.kernel.GetChisquare() / red_chisqu if red_chisqu > 0. else 0.
        mean = self.kernel.GetParameter(1) * mean_scale
        mean_err = self.kernel.GetParError(1) * mean_scale
        sigma = self.kernel.GetParameter(2) * sigma_scale
        sigma_err = self.kernel.GetParError(2) * sigma_scale

        root_objects.append(self.add_pave_helper_(0.55, 0.75, 0.89, 0.89, "NDC"))
        self.add_text_helper_(root_objects[-1],
                              f"#chi/ndf = {red_chisqu:.4f}", color_sig)
        self.add_text_helper_(root_objects[-1],
                              f"#mu = {mean:.4f} #pm {mean_err:.4f} {mean_dim}", color_sig)
        self.add_text_helper_(root_objects[-1],
                              f"#sigma = {sigma:.4f} #pm {sigma_err:.4f} {sigma_dim}", color_sig)
        root_objects[-1].Draw()

        for dob in draw_objects:
            dob.Draw("same")

        if add_root_objects:
            for aro in add_root_objects:
                root_objects.append(aro)
                aro.Draw("same")

# pylint: disable=too-many-instance-attributes
class FitSystAliHF(FitROOT):
    """
    Class with AliHFMassFitter as core fitting utility
    """

    def __init__(self, *args, histo=None, histo_mc=None, histo_reflections=None, **base_args):
        super().__init__(*args, **base_args)
        self.histo = histo
        self.histo_mc = histo_mc
        self.histo_reflections = histo_reflections

        self.default_init_pars = {"mean": None,
                                  "sigma": None,
                                  "include_sec_peak": False,
                                  "sec_mean": None,
                                  "fix_sec_mean": False,
                                  "sec_sigma": None,
                                  "fix_sec_sigma": False,
                                  "use_sec_peak_rel_sigma": True,
                                  "include_reflections": False,
                                  "fix_reflections_s_over_b": True,
                                  "mean_ref": None,
                                  "sigma_ref": None,
                                  "yield_ref": None,
                                  "chi2_ref": None,
                                  "rebin": None,
                                  "fit_range_low": None,
                                  "fit_range_up": None,
                                  "likelihood": True,
                                  "n_sigma_sideband": None,
                                  "fit_range_low_syst": None,
                                  "fit_range_up_syst": None,
                                  "bin_count_sigma_syst": None,
                                  "bkg_func_names_syst": None,
                                  "rebin_syst": None,
                                  "consider_free_sigma_syst": None,
                                  "signif_min_syst": None,
                                  "chi2_max_syst": None}
        # Fitted parameters (to be modified for deriving classes)
        # Only those corresponding to init parameters are here. Specific parameters/values
        # provided by the kernel have to be extracted from that directly.
        self.fit_pars = None
        self.results_path = base_args.get("results_path", None)
        self.update_root_objects()


    def update_root_objects(self):
        if self.root_objects is None:
            self.root_objects = {}
        self.histo = self.root_objects.get("histo", self.histo)
        self.histo_mc = self.root_objects.get("histo_mc", self.histo_mc)
        self.histo_reflections = self.root_objects.get("histo_reflections", self.histo_reflections)

        self.root_objects["histo"] = self.histo
        self.root_objects["histo_mc"] = self.histo_mc
        self.root_objects["histo_reflections"] = self.histo_reflections


    def init_kernel(self):

        self.update_root_objects()

        self.histo = self.histo.Clone(f"{self.histo.GetName()}_fit_histo")
        self.kernel = AliHFInvMassMultiTrialFit()

        self.kernel.SetSuffixForHistoNames("")
        self.kernel.SetDrawIndividualFits(False)
        # This is always the mean of the central fit
        self.kernel.SetMass(self.init_pars["mean"])
        self.kernel.SetSigmaGaussMC(self.init_pars["sigma"])

        # First, disable all
        self.kernel.SetUseExpoBackground("kExpo" in self.init_pars["bkg_func_names_syst"])
        self.kernel.SetUseLinBackground("kLin" in self.init_pars["bkg_func_names_syst"])
        self.kernel.SetUsePol2Background("Pol2" in self.init_pars["bkg_func_names_syst"])
        self.kernel.SetUsePol3Background("Pol3" in self.init_pars["bkg_func_names_syst"])
        self.kernel.SetUsePol4Background("Pol4" in self.init_pars["bkg_func_names_syst"])
        self.kernel.SetUsePol5Background("Pol5" in self.init_pars["bkg_func_names_syst"])
        # NOTE Not used at the momemnt
        self.kernel.SetUsePowerLawBackground(False)
        self.kernel.SetUsePowerLawTimesExpoBackground(False)

        if self.init_pars["rebin_syst"]:
            rebin_steps = [self.init_pars["rebin"] + rel_rb \
                    if self.init_pars["rebin"] + rel_rb > 0 \
                    else 1 for rel_rb in self.init_pars["rebin_syst"]]
            # To only have unique values and we don't care about the order we can just do
            rebin_steps = array("i", list(set(rebin_steps)))
            self.kernel.ConfigureRebinSteps(len(rebin_steps), rebin_steps)
        if self.init_pars["fit_range_low_syst"]:
            low_lim_steps = array("d", self.init_pars["fit_range_low_syst"])
            self.kernel.ConfigureLowLimFitSteps(len(self.init_pars["fit_range_low_syst"]),
                                                low_lim_steps)
        if self.init_pars["fit_range_up_syst"]:
            up_lim_steps = array("d", self.init_pars["fit_range_up_syst"])
            self.kernel.ConfigureUpLimFitSteps(len(self.init_pars["fit_range_up_syst"]),
                                               up_lim_steps)

        if self.init_pars["bin_count_sigma_syst"]:
            self.kernel.ConfigurenSigmaBinCSteps(len(self.init_pars["bin_count_sigma_syst"]),
                                                 array("d", self.init_pars["bin_count_sigma_syst"]))

        if self.init_pars["include_reflections"]:
            histo_mc_ = AliVertexingHFUtils.RebinHisto(self.histo_mc,
                                                       self.init_pars["rebin"],
                                                       -1)
            self.histo_mc = TH1F()
            histo_mc_.Copy(self.histo_mc)
            self.histo_reflections = AliVertexingHFUtils.AdaptTemplateRangeAndBinning(
                self.histo_reflections, self.histo,
                self.init_pars["fit_range_low"], self.init_pars["fit_range_up"])
            if self.histo_reflections.Integral() > 0.:
                self.kernel.SetTemplatesForReflections(self.histo_reflections, self.histo_mc)
                r_over_s = self.histo_mc.Integral(
                    self.histo_mc.FindBin(self.init_pars["fit_range_low"]),
                    self.histo_mc.FindBin(self.init_pars["fit_range_up"]))
                if r_over_s > 0.:
                    r_over_s = self.histo_reflections.Integral(
                        self.histo_reflections.FindBin(self.init_pars["fit_range_low"]),
                        self.histo_reflections.FindBin(self.init_pars["fit_range_up"])) \
                                / r_over_s
                    self.kernel.SetFixRefoS(r_over_s)
            else:
                self.logger.warning("Reflection requested but template empty")

        if self.init_pars["include_sec_peak"]:
            #p_widthsecpeak to be fixed
            sec_sigma = self.init_pars["sigma"] * self.init_pars["sec_sigma"] \
                    if self.init_pars["use_sec_peak_rel_sigma"] \
                    else self.init_pars["sec_sigma"]
            self.kernel.IncludeSecondGausPeak(self.init_pars["sec_mean"],
                                              self.init_pars["fix_sec_mean"],
                                              sec_sigma,
                                              self.init_pars["fix_sec_sigma"])
        return True


    def fit_kernel(self):

        histo_double = TH1D()
        self.histo.Copy(histo_double)
        success = self.kernel.DoMultiTrials(histo_double)
        if success and self.results_path:
            self.kernel.SaveToRoot(self.results_path)
        return success


    def set_fit_pars(self):
        pass
        #self.fit_pars["mean"] = self.kernel.GetMean()
        #self.fit_pars["sigma"] = self.kernel.GetSigma()


    #pylint: disable=dangerous-default-value, too-many-branches, too-many-statements, too-many-locals
    def draw_kernel(self, root_pad, root_objects=[], **draw_args):

        if not self.results_path:
            self.logger.warning("Don't have a result file so cannot draw. Skip...")
            return

        title = draw_args.pop("title", "")

        # Which background functions are used?
        used_bkgs = array("b", ["kExpo" in self.init_pars["bkg_func_names_syst"],
                                "kLin" in self.init_pars["bkg_func_names_syst"],
                                "Pol2" in self.init_pars["bkg_func_names_syst"],
                                "Pol3" in self.init_pars["bkg_func_names_syst"],
                                "Pol4" in self.init_pars["bkg_func_names_syst"],
                                "Pol5" in self.init_pars["bkg_func_names_syst"]])

        # Number of bin count variations
        n_bins_bincount = len(self.init_pars["bin_count_sigma_syst"]) \
                if self.init_pars["bin_count_sigma_syst"] else 0

        # The following is just crazy

        # Cache reference values
        mean_ref = self.init_pars["mean_ref"]
        sigma_ref = self.init_pars["sigma_ref"]
        yield_ref = self.init_pars["yield_ref"]

        bkg_treat = ""
        input_file = TFile.Open(self.results_path, "READ")

        # Prepare variables
        bkg_colors = [kPink - 6, kCyan + 3, kGreen - 1, kYellow - 2, kRed - 6, kBlue - 6]
        n_back_func_cases = 6
        n_config_cases = 6
        color_bc0 = kGreen + 2
        color_bc1 = kOrange + 5
        min_bc_range = 1
        max_bc_range = n_bins_bincount
        n_bc_ranges = n_bins_bincount
        conf_case = ["FixedSigFreeMean",
                     "FixedSigUp",
                     "FixedSigDw",
                     "FreeSigFreeMean",
                     "FreeSigFixedMean",
                     "FixedSigFixedMean"]

        # Names of background functions used internally
        bkg_func = ["Expo", "Lin", "Pol2", "Pol3", "Pol4", "Pol5"]

        tot_cases = n_config_cases * n_back_func_cases
        # Mask to flag what's en/disabled
        # 0 => not used; 1 => used for fit; 2 => used also for bin count
        mask = [0] * tot_cases #0,0,0,0,0,0,   // fixed sigma, free mean (Expo, Lin, Pol2,Pol3,Pol4)
                               #0,0,0,0,0,0,   // fixed sigma upper
                               #0,0,0,0,0,0,   // fixed sigma lower
                               #0,0,0,0,0,0,   // free sigma, free mean
                               #0,0,0,0,0,0,   // free sigma, fixed mean
                               #0,0,0,0,0,0,   // fixed mean, fixed sigma

        # Enable only the background cases we ran the multi trial with
        plot_case = 2 if max_bc_range >= min_bc_range else 1
        for i in range(6):
            if used_bkgs[i] > 0:
                mask[i] = plot_case
                mask[30+i] = plot_case
                if self.init_pars["consider_free_sigma_syst"]:
                    mask[18+i] = plot_case
                    mask[24+i] = plot_case

        # Extract histograms from file
        histo6 = [None] * tot_cases
        kjh = 0
        for i_conf in range(n_config_cases):
            for i_type in range(n_back_func_cases):
                histo_name = f"hRawYieldTrial{bkg_func[i_type]}{conf_case[i_conf]}{bkg_treat}"
                histo6[kjh] = input_file.Get(histo_name)
                if not histo6[kjh]:
                    self.logger.warning("Histo %s not found", histo_name)
                    mask[kjh] = 0
                kjh += 1


        # Prepare variables for counting
        tot_trials = 0
        successful_trials = 0
        tot_trials_bc0 = 0
        tot_trials_bc1 = 0
        tot_histos = 0
        first = [0] * tot_cases
        last = [0] * tot_cases
        first_bc0 = [0] * tot_cases
        last_bc0 = [0] * tot_cases
        first_bc1 = [0] * tot_cases
        last_bc1 = [0] * tot_cases
        #tlabels = [None] * (tot_cases+1)


        for nc in range(tot_cases):
            if not mask[nc]:
                continue

            first[nc] = tot_trials
            tot_trials += histo6[nc].GetNbinsX()
            last[nc] = tot_trials
            tot_histos += 1

            # This we might include later
            #ttt = histo6[nc].GetName()
            #ttt = ttt.replace("hRawYieldTrial", "")
            #if "FixedMean" in ttt:
            #    ttt = "Fix #mu"
            #elif "FixedSp20" in ttt:
            #    ttt = "#sigma+"
            #elif "fixedSm20" in ttt:
            #    ttt = "#sigma-"
            #elif "FreeS" in ttt:
            #    ttt = "Free #sigma"
            #ttt = ttt.replace("FixedS", "")
            #if bkg_treat and bkg_treat in ttt:
            #    ttt = ttt.replace(bkg_treat, "")

            #tlabels[nc] = TLatex(first[nc] + 0.02 * tot_trials, 10, ttt)
            #tlabels[nc].SetTextColor(kMagenta+2)
            #tlabels[nc].SetTextColor(kMagenta+2)

            # Extract bin count cases
            if mask[nc] == 2:
                hbcname = histo6[nc].GetName()
                # Take bin count from background function of total fit
                hbcname = hbcname.replace("Trial", "TrialBinC0")
                hbc2dt = input_file.Get(hbcname)
                first_bc0[nc] = tot_trials_bc0
                tot_trials_bc0 += hbc2dt.GetNbinsX()
                last_bc0[nc] = tot_trials_bc0

                hbcname = hbcname.replace("TrialBinC0", "TrialBinC1")
                hbc2dt = input_file.Get(hbcname)
                first_bc1[nc] = tot_trials_bc1
                tot_trials_bc1 += hbc2dt.GetNbinsX()
                last_bc1[nc] = tot_trials_bc1

        # We need one histogram per background function, do it brute force with a dictionary
        h_raw_yield_all_bkgs = {}
        h_mean_all_bkgs = {}
        h_sigma_all_bkgs = {}
        h_chi2_all_bkgs = {}

        for nc in range(tot_cases):
            if not mask[nc]:
                continue
            hmeanname = histo6[nc].GetName()
            for i_bkg in range(n_back_func_cases):
                if bkg_func[i_bkg] in h_raw_yield_all_bkgs:
                    continue

                if bkg_func[i_bkg] in hmeanname:
                    h_raw_yield_all_bkgs[bkg_func[i_bkg]] = \
                            TH1F(f"hRawYieldAll_{bkg_func[i_bkg]}",
                                 " ; Trial # ; raw yield", tot_trials, 0., tot_trials)
                    h_raw_yield_all_bkgs[bkg_func[i_bkg]].SetLineColor(bkg_colors[i_bkg])
                    h_raw_yield_all_bkgs[bkg_func[i_bkg]].SetMarkerColor(bkg_colors[i_bkg])
                    h_raw_yield_all_bkgs[bkg_func[i_bkg]].SetStats(0)

                    h_mean_all_bkgs[bkg_func[i_bkg]] = \
                            TH1F(f"hMeanAll_{bkg_func[i_bkg]}",
                                 " ; Trial # ; Gaussian mean", tot_trials, 0., tot_trials)
                    h_mean_all_bkgs[bkg_func[i_bkg]].SetLineColor(bkg_colors[i_bkg])
                    h_mean_all_bkgs[bkg_func[i_bkg]].SetMarkerColor(bkg_colors[i_bkg])
                    h_mean_all_bkgs[bkg_func[i_bkg]].SetMinimum(0.8 * mean_ref)
                    h_mean_all_bkgs[bkg_func[i_bkg]].SetMaximum(1.2 * mean_ref)
                    h_mean_all_bkgs[bkg_func[i_bkg]].SetStats(0)

                    h_sigma_all_bkgs[bkg_func[i_bkg]] = \
                            TH1F(f"hSigmaAll_{bkg_func[i_bkg]}",
                                 " ; Trial # ; Gaussian Sigma", tot_trials, 0., tot_trials)
                    h_sigma_all_bkgs[bkg_func[i_bkg]].SetLineColor(bkg_colors[i_bkg])
                    h_sigma_all_bkgs[bkg_func[i_bkg]].SetMarkerColor(bkg_colors[i_bkg])
                    h_sigma_all_bkgs[bkg_func[i_bkg]].SetMinimum(0.)
                    h_sigma_all_bkgs[bkg_func[i_bkg]].SetMaximum(1.1 * sigma_ref)
                    h_sigma_all_bkgs[bkg_func[i_bkg]].SetStats(0)

                    h_chi2_all_bkgs[bkg_func[i_bkg]] = \
                            TH1F(f"hChi2All_{bkg_func[i_bkg]}",
                                 " ; Trial # ; Chi2", tot_trials, 0., tot_trials)
                    h_chi2_all_bkgs[bkg_func[i_bkg]].SetLineColor(bkg_colors[i_bkg])
                    h_chi2_all_bkgs[bkg_func[i_bkg]].SetMarkerColor(bkg_colors[i_bkg])
                    h_chi2_all_bkgs[bkg_func[i_bkg]].SetMarkerStyle(7)
                    h_chi2_all_bkgs[bkg_func[i_bkg]].SetStats(0)


        # Create histograms for fit and bin count yield to be plotted in the end
        h_raw_yield_all_bc0 = TH1F(f"hRawYieldAllBC0", " ; Trial # ; raw yield BC0",
                                   tot_trials_bc0 * n_bc_ranges, 0.,
                                   tot_trials_bc0 * n_bc_ranges)

        h_raw_yield_all_bc1 = TH1F(f"hRawYieldAllBC1", " ; Trial # ; raw yield BC1",
                                   tot_trials_bc1 * n_bc_ranges, 0.,
                                   tot_trials_bc1 * n_bc_ranges)



        lower_edge_yield_histos = yield_ref - 1.5 * yield_ref
        lower_edge_yield_histos = max(0., lower_edge_yield_histos)

        upper_edge_yield_histos = yield_ref + 1.5 * yield_ref

        h_raw_yield_dist_all = TH1F("hRawYieldDistAll", "  ; raw yield", 200,
                                    lower_edge_yield_histos, upper_edge_yield_histos)
        h_raw_yield_dist_all.SetFillStyle(3003)
        h_raw_yield_dist_all.SetFillColor(kBlue + 1)

        h_raw_yield_dist_all_bc0 = TH1F("hRawYieldDistAllBC0", "  ; raw yield", 200,
                                        lower_edge_yield_histos, upper_edge_yield_histos)
        h_raw_yield_dist_all_bc1 = TH1F("hRawYieldDistAllBC1", "  ; raw yield", 200,
                                        lower_edge_yield_histos, upper_edge_yield_histos)
        h_raw_yield_dist_all_bc0.SetFillStyle(3004)
        h_raw_yield_dist_all_bc1.SetFillStyle(3004)
        # NOTE Note used at the moment
        #TH1F* hStatErrDistAll=new TH1F("hStatErrDistAll","  ; Stat Unc on Yield",300,0,10000);
        #TH1F* hRelStatErrDistAll=new TH1F("hRelStatErrDistAll",
        #                                  "  ; Rel Stat Unc on Yield",100,0.,1.);
        #######################################################################
        min_yield = 999999.
        max_yield = 0.
        sumy = [0.] * 4
        sumwei = [0.] * 4
        sumerr = [0.] * 4
        counts = 0.
        wei = [None] * 4
        max_filled = -1

        ##################
        # Extract yields #
        ##################
        for nc in range(tot_cases):
            if not mask[nc]:
                continue
            histo_name = histo6[nc].GetName()
            hmeanname = histo_name.replace("RawYield", "Mean")
            hmeant6 = input_file.Get(hmeanname)

            hsigmaname = histo_name.replace("RawYield", "Sigma")
            hsigmat6 = input_file.Get(hsigmaname)

            hchi2name = histo_name.replace("RawYield", "Chi2")
            hchi2t6 = input_file.Get(hchi2name)

            hsignifname = histo_name.replace("RawYield", "Signif")
            hsignift6 = input_file.Get(hsignifname)

            hbcname = histo_name.replace("Trial", "TrialBinC0")
            hbc2dt060 = input_file.Get(hbcname)

            hbcname = hbcname.replace("TrialBinC0", "TrialBinC1")
            hbc2dt060_bc1 = input_file.Get(hbcname)

            for ib in range(1, histo6[nc].GetNbinsX() + 1):
                ry = histo6[nc].GetBinContent(ib)
                ery = histo6[nc].GetBinError(ib)

                pos = hmeant6.GetBinContent(ib)
                epos = hmeant6.GetBinError(ib)

                sig = hsigmat6.GetBinContent(ib)
                esig = hsigmat6.GetBinError(ib)

                chi2 = hchi2t6.GetBinContent(ib)
                signif = hsignift6.GetBinContent(ib)

                # Fill
                if ry < 0.001 or (0.5 * ry) < ery or ery < (0.01 * ry) \
                        or chi2 > self.init_pars["chi2_max_syst"] \
                        or signif < self.init_pars["signif_min_syst"]:
                    continue
                successful_trials += 1
                # Get the right histograms to fill
                bkg_func_name = hmeant6.GetName()
                for bkg_func_test in bkg_func:
                    if bkg_func_test in bkg_func_name:
                        bkg_func_name = bkg_func_test
                        break

                h_raw_yield_dist_all.Fill(ry)
                h_raw_yield_all_bkgs[bkg_func_name].SetBinContent(first[nc] + ib, ry)
                h_raw_yield_all_bkgs[bkg_func_name].SetBinError(first[nc] + ib, ery)
                # NOTE Not used at the moment
                #hStatErrDistAll->Fill(ery);
                #hRelStatErrDistAll->Fill(ery/ry);
                min_yield = min(ry, min_yield)
                max_yield = max(ry, max_yield)

                wei[0] = 1.
                wei[1] = 1. / (ery * ery)
                wei[2] = 1. / (ery * ery / (ry * ry))
                wei[3] = 1. / (ery * ery / ry)
                for kw in range(4):
                    sumy[kw] += wei[kw] * ry
                    sumerr[kw] += wei[kw] * wei[kw] * ery * ery
                    sumwei[kw] += wei[kw]

                counts += 1.
                h_sigma_all_bkgs[bkg_func_name].SetBinContent(first[nc] + ib, sig)
                h_sigma_all_bkgs[bkg_func_name].SetBinError(first[nc] + ib, esig)
                h_mean_all_bkgs[bkg_func_name].SetBinContent(first[nc] + ib, pos)
                h_mean_all_bkgs[bkg_func_name].SetBinError(first[nc] + ib, epos)
                h_chi2_all_bkgs[bkg_func_name].SetBinContent(first[nc] + ib, chi2)
                h_chi2_all_bkgs[bkg_func_name].SetBinError(first[nc] + ib, 0.000001)

                if mask[nc] == 2:
                    for iy in range(min_bc_range, max_bc_range + 1):
                        bc = hbc2dt060.GetBinContent(ib, iy)
                        ebc = hbc2dt060.GetBinError(ib, iy)
                        bc_1 = hbc2dt060_bc1.GetBinContent(ib, iy)
                        ebc_1 = hbc2dt060_bc1.GetBinError(ib, iy)
                        #if(bc>0.001 && ebc<0.5*bc && bc<5.*ry){
                        if bc < 0.001:
                            continue
                        the_bin = iy + (first_bc0[nc] + ib - 1) * n_bc_ranges
                        h_raw_yield_all_bc0.SetBinContent(the_bin - 2, bc)
                        h_raw_yield_all_bc0.SetBinError(the_bin - 2, ebc)
                        h_raw_yield_dist_all_bc0.Fill(bc)
                        if h_raw_yield_all_bc0.GetBinCenter(the_bin - 2) > max_filled:
                            max_filled = h_raw_yield_all_bc0.GetBinCenter(the_bin-2)
                        the_bin = iy + (first_bc1[nc] + ib - 1) * n_bc_ranges
                        h_raw_yield_all_bc1.SetBinContent(the_bin - 2, bc_1)
                        h_raw_yield_all_bc1.SetBinError(the_bin - 2, ebc_1)
                        h_raw_yield_dist_all_bc1.Fill(bc_1)
                        if h_raw_yield_all_bc1.GetBinCenter(the_bin - 2) > max_filled:
                            max_filled = h_raw_yield_all_bc1.GetBinCenter(the_bin - 2)


        weiav = [0.] * 4
        eweiav = [0.] * 4
        for kw in range(4):
            if sumwei[kw] > 0.:
                weiav[kw] = sumy[kw] / sumwei[kw]
                eweiav[kw] = sqrt(sumerr[kw]) / sumwei[kw]

        ##################
        # Style and plot #
        ##################
        h_raw_yield_all_bc0.SetStats(0)
        h_raw_yield_all_bc0.SetMarkerColor(color_bc0)
        h_raw_yield_all_bc0.SetLineColor(color_bc0)
        h_raw_yield_dist_all_bc0.SetLineColor(color_bc0)
        h_raw_yield_dist_all_bc0.SetFillColor(color_bc0)
        h_raw_yield_dist_all_bc0.SetLineWidth(1)
        h_raw_yield_dist_all_bc0.SetLineStyle(1)
        if h_raw_yield_dist_all_bc0.GetEntries() > 0:
            h_raw_yield_dist_all_bc0.Scale(\
                    h_raw_yield_dist_all.GetEntries() / h_raw_yield_dist_all_bc0.GetEntries())

        h_raw_yield_all_bc1.SetStats(0)
        h_raw_yield_all_bc1.SetMarkerColor(color_bc1)
        h_raw_yield_all_bc1.SetLineColor(color_bc1)
        h_raw_yield_dist_all_bc1.SetLineColor(color_bc1)
        h_raw_yield_dist_all_bc1.SetFillColor(color_bc1)
        h_raw_yield_dist_all_bc1.SetLineWidth(1)
        h_raw_yield_dist_all_bc1.SetLineStyle(1)
        if h_raw_yield_dist_all_bc1.GetEntries() > 0:
            h_raw_yield_dist_all_bc1.Scale(\
                    h_raw_yield_dist_all.GetEntries() / h_raw_yield_dist_all_bc1.GetEntries())

        h_raw_yield_dist_all.SetStats(0)
        h_raw_yield_dist_all.SetLineWidth(1)

        l = TLine(yield_ref, 0., yield_ref, h_raw_yield_dist_all.GetMaximum())
        l.SetLineColor(kRed)
        l.SetLineWidth(2)

        ll = TLine(0., yield_ref, tot_trials, yield_ref)
        ll.SetLineColor(kRed)
        ll.SetLineWidth(2)

        root_pad.Divide(3, 2)
        sigma_pad = root_pad.cd(1)
        sigma_pad.SetLeftMargin(0.13)
        sigma_pad.SetRightMargin(0.06)
        for histo in  h_sigma_all_bkgs.values():
            histo.GetYaxis().SetTitleOffset(1.7)
            histo.Draw("same")
            root_objects.append(histo)
            histo.SetDirectory(0)

        bkg_func_legend = TLegend(0.2, 0.2, 0.5, 0.5)
        bkg_func_legend.SetTextSize(0.04)
        bkg_func_legend.SetBorderSize(0)
        bkg_func_legend.SetFillStyle(0)
        root_objects.append(bkg_func_legend)

        mean_pad = root_pad.cd(2)
        mean_pad.SetLeftMargin(0.13)
        mean_pad.SetRightMargin(0.06)
        for name, histo in  h_mean_all_bkgs.items():
            histo.GetYaxis().SetTitleOffset(1.7)
            histo.Draw("same")
            root_objects.append(histo)
            histo.SetDirectory(0)
            bkg_func_legend.AddEntry(histo, name)

        bkg_func_legend.Draw("same")
        chi2_pad = root_pad.cd(3)
        chi2_pad.SetLeftMargin(0.13)
        chi2_pad.SetRightMargin(0.06)

        for histo in h_chi2_all_bkgs.values():
            histo.GetYaxis().SetTitleOffset(1.7)
            histo.Draw("same")
            root_objects.append(histo)
            histo.SetDirectory(0)

        yield_pad = root_pad.cd(4)
        yield_pad.Divide(1, 2)
        yield_pad_sub = yield_pad.cd(1)

        yield_pad_sub.SetLeftMargin(0.13)
        yield_pad_sub.SetRightMargin(0.06)
        new_max = 0.

        for histo in h_raw_yield_all_bkgs.values():
            tmp_max = 1.25 * (histo.GetMaximum() + histo.GetBinError(1))
            new_max = max(new_max, tmp_max)
            histo.GetYaxis().SetTitleOffset(1.7)


        for histo in h_raw_yield_all_bkgs.values():
            if max_filled > 0:
                histo.GetXaxis().SetRangeUser(0., max_filled)
            histo.SetMaximum(new_max)
            histo.Draw("same")
            root_objects.append(histo)
            histo.SetDirectory(0)

        ll.Draw("same")
        yield_pad_sub = yield_pad.cd(2)
        yield_pad_sub.SetLeftMargin(0.13)
        yield_pad_sub.SetRightMargin(0.06)
        root_objects.append(ll)

        h_raw_yield_all_bc0.GetYaxis().SetTitleOffset(1.7)
        h_raw_yield_all_bc0.Draw("same")
        root_objects.append(h_raw_yield_all_bc0)
        h_raw_yield_all_bc0.SetDirectory(0)

        h_raw_yield_all_bc1.GetYaxis().SetTitleOffset(1.7)
        h_raw_yield_all_bc1.Draw("same")
        root_objects.append(h_raw_yield_all_bc1)
        h_raw_yield_all_bc1.SetDirectory(0)

        ll_bc = TLine(0., yield_ref, tot_trials * n_bc_ranges, yield_ref)
        ll_bc.SetLineColor(kRed)
        ll_bc.SetLineWidth(2)
        ll_bc.Draw("same")
        root_objects.append(ll_bc)


        yield_pad = root_pad.cd(5)
        yield_pad.SetLeftMargin(0.14)
        yield_pad.SetRightMargin(0.06)

        h_raw_yield_dist_all.SetTitle(title)
        h_raw_yield_dist_all.Draw("same")
        root_objects.append(h_raw_yield_dist_all)
        h_raw_yield_dist_all.SetDirectory(0)
        h_raw_yield_dist_all.GetXaxis().SetRangeUser(min_yield * 0.8, max_yield * 1.2)

        h_raw_yield_dist_all_bc0.Draw("sameshist")
        root_objects.append(h_raw_yield_dist_all_bc0)
        h_raw_yield_dist_all_bc0.SetDirectory(0)
        h_raw_yield_dist_all_bc1.Draw("sameshist")
        root_objects.append(h_raw_yield_dist_all_bc1)
        h_raw_yield_dist_all_bc1.SetDirectory(0)
        l.Draw("same")
        root_objects.append(l)
        yield_pad.Update()

        # This might be taken care of later
        #st = h_raw_yield_dist_all.GetListOfFunctions().FindObject("stats")
        #st.SetY1NDC(0.71)
        #st.SetY2NDC(0.9)
        #stb0 = h_raw_yield_dist_all_bc0.GetListOfFunctions().FindObject("stats")
        #stb0.SetY1NDC(0.51)
        #stb0.SetY2NDC(0.7)
        #stb0.SetTextColor(h_raw_yield_dist_all_bc0.GetLineColor())
        perc = array("d", [0.15, 0.5, 0.85]) # quantiles for +-1 sigma
        lim70 = array("d", [0.] * 3)
        h_raw_yield_dist_all.GetQuantiles(3, lim70, perc)


        #######################
        # Numbers and summary #
        #######################
        sum_pad = root_pad.cd(6)
        sum_pad.SetLeftMargin(0.14)
        sum_pad.SetRightMargin(0.06)
        aver = h_raw_yield_dist_all.GetMean()
        rel_succ_trials = successful_trials / tot_trials if tot_trials > 0 else 0.
        trel_succ_trials = TLatex(0.15, 0.93, f"succ. trials= " \
                f"{successful_trials} / {tot_trials} " \
                f"({rel_succ_trials * 100.:.2f}%)")
        trel_succ_trials.SetTextSize(0.04)
        trel_succ_trials.SetNDC()
        trel_succ_trials.Draw("same")
        root_objects.append(trel_succ_trials)

        tmean = TLatex(0.15, 0.87, f"mean={aver:.3f}")
        tmean.SetTextSize(0.04)
        tmean.SetNDC()
        tmean.Draw("same")
        root_objects.append(tmean)

        tmedian = TLatex(0.15, 0.81, f"median={lim70[1]:.3f}")
        tmedian.SetTextSize(0.04)
        tmedian.SetNDC()
        tmedian.Draw("same")
        root_objects.append(tmedian)

        aver_bc0 = h_raw_yield_dist_all_bc0.GetMean()
        tmean_bc0 = TLatex(0.15, 0.75, f"mean(BinCount0)={aver_bc0:.3f}")
        tmean_bc0.SetTextSize(0.04)
        tmean_bc0.SetNDC()
        tmean_bc0.SetTextColor(h_raw_yield_dist_all_bc0.GetLineColor())
        tmean_bc0.Draw("same")
        root_objects.append(tmean_bc0)

        aver_bc1 = h_raw_yield_dist_all_bc1.GetMean()
        tmean_bc1 = TLatex(0.15, 0.69, f"mean(BinCount1)={aver_bc1:.3f}")
        tmean_bc1.SetTextSize(0.04)
        tmean_bc1.SetNDC()
        tmean_bc1.SetTextColor(h_raw_yield_dist_all_bc1.GetLineColor())
        tmean_bc1.Draw("same")
        root_objects.append(tmean_bc1)

        val = h_raw_yield_dist_all.GetRMS()
        val_rel = val / aver if aver != 0 else 0
        thrms = TLatex(0.15, 0.60, f"rms={val:.3f} ({val_rel:.2f}%)")
        thrms.SetTextSize(0.04)
        thrms.SetNDC()
        thrms.Draw("same")
        root_objects.append(thrms)

        val = h_raw_yield_dist_all_bc0.GetRMS()
        val_rel = val / aver_bc0 * 100. if aver_bc0 != 0 else 0
        thrms_bc0 = TLatex(0.15, 0.54, f"rms(BinCount0)={val:.3f} ({val_rel:.2f}%)")
        thrms_bc0.SetTextSize(0.04)
        thrms_bc0.SetNDC()
        thrms_bc0.SetTextColor(h_raw_yield_dist_all_bc0.GetLineColor())
        thrms_bc0.Draw("same")
        root_objects.append(thrms_bc0)

        val = h_raw_yield_dist_all_bc1.GetRMS()
        val_rel = val / aver_bc1 * 100. if aver_bc1 != 0 else 0
        thrms_bc1 = TLatex(0.15, 0.48, f"rms(BinCount1)={val:.3f} ({val_rel:.2f}%)")
        thrms_bc1.SetTextSize(0.04)
        thrms_bc1.SetNDC()
        thrms_bc1.SetTextColor(h_raw_yield_dist_all_bc1.GetLineColor())
        thrms_bc1.Draw("same")
        root_objects.append(thrms_bc1)

        tmin = TLatex(0.15, 0.39, f"min={min_yield:.2f}      max={max_yield:.2f}")
        tmin.SetTextSize(0.04)
        tmin.SetNDC()
        tmin.Draw("same")
        root_objects.append(tmin)

        val = (max_yield - min_yield) / sqrt(12)
        val_rel = val / aver * 100. if aver != 0 else 0
        trms = TLatex(0.15, 0.33, f"(max-min)/sqrt(12)={val:.3f} ({val_rel:.2f}%)")
        trms.SetTextSize(0.04)
        trms.SetNDC()
        trms.Draw("same")
        root_objects.append(trms)

        mean_ref_label = TLatex(0.15, 0.27, f"mean(ref)={yield_ref:.2f}")
        mean_ref_label.SetTextSize(0.04)
        mean_ref_label.SetNDC()
        mean_ref_label.SetTextColor(kRed)
        mean_ref_label.Draw("same")
        root_objects.append(mean_ref_label)

        val_rel = 100 * (yield_ref - aver) / yield_ref if yield_ref != 0 else 0
        mean_ref_diff = TLatex(0.15, 0.21,
                               f"mean(ref)-mean(fit)={yield_ref - aver:.3f}  " \
                               f"({val_rel:.2f}%)")
        mean_ref_diff.SetTextSize(0.04)
        mean_ref_diff.SetNDC()
        mean_ref_diff.SetTextColor(kBlack)
        mean_ref_diff.Draw("same")
        root_objects.append(mean_ref_diff)

        val_rel = 100 * (yield_ref - aver_bc0) / yield_ref if yield_ref != 0 else 0
        mean_ref_diff_bc0 = TLatex(0.15, 0.15,
                                   f"mean(ref)-mean(BC0)={yield_ref - aver_bc0:.3f}  " \
                                   f"({val_rel:.2f}%)")
        mean_ref_diff_bc0.SetTextSize(0.04)
        mean_ref_diff_bc0.SetNDC()
        mean_ref_diff_bc0.SetTextColor(h_raw_yield_dist_all_bc0.GetLineColor())
        mean_ref_diff_bc0.Draw("same")
        root_objects.append(mean_ref_diff_bc0)

        val_rel = 100 * (yield_ref - aver_bc1) / yield_ref if yield_ref != 0 else 0
        mean_ref_diff_bc1 = TLatex(0.15, 0.09,
                                   f"mean(ref)-mean(BC1)={yield_ref - aver_bc1:.3f}  " \
                                   f"({val_rel:.2f})%")
        mean_ref_diff_bc1.SetTextSize(0.04)
        mean_ref_diff_bc1.SetNDC()
        mean_ref_diff_bc1.SetTextColor(h_raw_yield_dist_all_bc1.GetLineColor())
        mean_ref_diff_bc1.Draw("same")
        root_objects.append(mean_ref_diff_bc1)

        if draw_args:
            self.logger.warning("There are unknown draw arguments")
# pylint: enable=too-many-instance-attributes
