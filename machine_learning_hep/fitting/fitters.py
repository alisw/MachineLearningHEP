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

from copy import deepcopy
from array import array

# pylint: disable=import-error, no-name-in-module, unused-import
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils, AliHFInvMassMultiTrialFit
from ROOT import TH1F, TF1, TPaveText, Double
from ROOT import kBlue, kRed, kGreen, kMagenta

from machine_learning_hep.logger import get_logger

# pylint: disable=too-many-instance-attributes
class FitBase:
    """
    Common base class for FitAliHF and FitROOT.
    """

    def __init__(self, init_pars):
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

        pass


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
                                         self.init_pars["sig_func_name"],
                                         self.init_pars["bkg_func_name"])
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

        draw_objects = [self.histo]
        draw_options = ["PE"] + [""] * 3
        sig_func = self.kernel.GetMassFunc()
        draw_objects.append(sig_func)
        bkg_func = self.kernel.GetBackgroundFullRangeFunc()
        draw_objects.append(bkg_func)
        bkg_refit_func = self.kernel.GetBackgroundRecalcFunc()
        draw_objects.append(bkg_refit_func)
        refl_func = self.kernel.GetReflFunc() if self.init_pars["include_reflections"] else None
        if refl_func:
            draw_objects.append(refl_func)
            draw_options.append("")
        sec_peak_func = self.kernel.GetSecondPeakFunc() \
                if self.init_pars["include_sec_peak"] else None
        if sec_peak_func:
            draw_objects.append(sec_peak_func)
            draw_options.append("")

        y_max = self.histo.GetMaximum()
        for do in draw_objects:
            y_max = max(y_max, do.GetMaximum())
        # Leave some space for putting info
        y_max *= 1.8

        # Now comes some styling
        color_sig = kBlue - 3
        color_bkg_refit = kRed + 2
        color_refl = kGreen + 2
        color_sec_peak = kMagenta + 3
        self.histo.SetMarkerStyle(20)
        bkg_refit_func.SetLineColor(color_bkg_refit)
        sig_func.SetLineColor(color_sig)

        root_pad.SetLeftMargin(0.12)
        frame = root_pad.cd().DrawFrame(self.init_pars["fit_range_low"], 0.,
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
                                  "likelihood": True}
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

        for r in range(2, 8):
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
        super().__init__(**base_args)
        self.histo = histo
        self.histo_mc = histo_mc
        self.histo_reflections = histo_reflections

        self.default_init_pars = {"mean": None,
                                  "sigma": None,
                                  "mean_central": None,
                                  "sigma_central": None,
                                  "include_sec_peak": False,
                                  "sec_mean": None,
                                  "fix_sec_mean": False,
                                  "sec_sigma": None,
                                  "fix_sec_sigma": False,
                                  "use_sec_peak_rel_sigma": True,
                                  "include_reflections": False,
                                  "fix_reflections_s_over_b": True,
                                  "rebin": None,
                                  "rebin_central": None,
                                  "fit_range_low": None,
                                  "fit_range_up": None,
                                  "fit_range_low_central": None,
                                  "fit_range_up_central": None,
                                  "bin_count_sigma": None,
                                  "bkg_func_names": None,
                                  "likelihood": True,
                                  "n_sigma_sideband": None}
        # Fitted parameters (to be modified for deriving classes)
        # Only those corresponding to init parameters are here. Specific parameters/values
        # provided by the kernel have to be extracted from that directly.
        self.fit_pars = None
        self.results_path = None
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
        self.kernel.SetUseExpoBackground("kExpo" in self.init_pars["bkg_func_names"])
        self.kernel.SetUseLinBackground("kLin" in self.init_pars["bkg_func_names"])
        self.kernel.SetUsePol2Background("Pol2" in self.init_pars["bkg_func_names"])
        self.kernel.SetUsePol3Background("Pol3" in self.init_pars["bkg_func_names"])
        self.kernel.SetUsePol4Background("Pol4" in self.init_pars["bkg_func_names"])
        self.kernel.SetUsePol5Background("Pol5" in self.init_pars["bkg_func_names"])
        # NOTE Not used at the momemnt
        self.kernel.SetUsePowerLawBackground(False)
        self.kernel.SetUsePowerLawTimesExpoBackground(False)

        if self.init_pars["rebin"]:
            rebin_steps = [self.init_pars["rebin_central"] + rel_rb \
                    if self.init_pars["rebin_central"] + rel_rb > 0 \
                    else 1 for rel_rb in self.init_pars["rebin"]]
            # To only have unique values and we don't care about the order we can just do
            rebin_steps = array("i", list(set(rebin_steps)))
            self.kernel.ConfigureRebinSteps(len(rebin_steps), rebin_steps)
        if self.init_pars["fit_range_low"]:
            low_lim_steps = array("d", self.init_pars["fit_range_low"])
            self.kernel.ConfigureLowLimFitSteps(len(self.init_pars["fit_range_low"]),
                                                low_lim_steps)
        if self.init_pars["fit_range_up"]:
            up_lim_steps = array("d", self.init_pars["fit_range_up"])
            self.kernel.ConfigureUpLimFitSteps(len(self.init_pars["fit_range_up"]), up_lim_steps)

        if self.init_pars["bincount_sigma"]:
            self.kernel.ConfigurenSigmaBinCSteps(len(self.init_pars["bincount_sigma"]),
                                                 array("d", self.init_pars["bincount_sigma"]))

        if self.init_pars["include_reflections"]:
            histo_mc_ = AliVertexingHFUtils.RebinHisto(self.histo_mc,
                                                       self.init_pars["rebin_central"],
                                                       -1)
            self.histo_mc = TH1F()
            histo_mc_.Copy(self.histo_mc)
            self.histo_reflections = AliVertexingHFUtils.AdaptTemplateRangeAndBinning(
                self.histo_reflections, self.histo,
                self.init_pars["fit_range_low_central"], self.init_pars["fit_range_up_central"])
            if self.histo_reflections.Integral() > 0.:
                self.kernel.SetTemplatesForReflections(self.histo_reflections, self.histo_mc)
                r_over_s = self.histo_mc.Integral(
                    self.histo_mc.FindBin(self.init_pars["fit_range_low_central"]),
                    self.histo_mc.FindBin(self.init_pars["fit_range_up_central"]))
                if r_over_s > 0.:
                    r_over_s = self.histo_reflections.Integral(
                        self.histo_reflections.FindBin(self.init_pars["fit_range_low_central"]),
                        self.histo_reflections.FindBin(self.init_pars["fit_range_up_central"])) \
                                / r_over_s
                    self.kernel.SetFixRefoS(r_over_s)
            else:
                self.logger.warning("Reflection requested but template empty")

        if self.init_pars["include_sec_peaks"]:
            #p_widthsecpeak to be fixed
            sec_sigma = self.init_pars["sigma"] * self.init_pars["sec_sigma"] \
                    if self.init_pars["use_sec_peak_rel_sigma"] \
                    else self.init_pars["sec_sigma"]
            self.kernel.IncludeSecondGausPeak(self.init_pars["sec_mean"],
                                              self.init_pars["fix_sec_mean"],
                                              sec_sigma,
                                              self.init_pars["fix_sec_sigma"])


    def fit_kernel(self):
        success = self.kernel.DoMultiTrials(self.histo)
        if success and self.results_path:
            self.kernel.SaveToRoot(self.results_path)
        return success


    def set_fit_pars(self):
        self.fit_pars["mean"] = self.kernel.GetMean()
        self.fit_pars["sigma"] = self.kernel.GetSigma()


    #pylint: disable=dangerous-default-value
    def draw_kernel(self, root_pad, root_objects=[], **draw_args):

        title = draw_args.pop("title", "")
        x_axis_label = draw_args.pop("x_axis_label", "")
        y_axis_label = draw_args.pop("y_axis_label", "")
        sigma_signal = draw_args.pop("sigma_signal", 3)

        add_root_objects = draw_args.pop("add_root_objects", None)


        if draw_args:
            self.logger.warning("There are unknown draw arguments")

        self.histo.SetTitle(title)
        self.histo.GetXaxis().SetTitle(x_axis_label)
        self.histo.GetYaxis().SetTitle(y_axis_label)

        self.histo.GetYaxis().SetTitleiOffset(1.1)

        root_pad.cd()
        self.kernel.DrawHere(root_pad, sigma_signal)

        if add_root_objects:
            for aro in add_root_objects:
                aro.Draw("same")
# pylint: enable=too-many-instance-attributes
