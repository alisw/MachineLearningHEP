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
Methods to: fit inv. mass
"""

from math import sqrt, pi, exp
# pylint: disable=import-error,no-name-in-module
from ROOT import TF1, gStyle, TCanvas, TPaveText, Double, TVirtualFitter, \
                 kGreen, kRed, kBlue, TGraph, gROOT
from  machine_learning_hep.logger import get_logger


gROOT.ProcessLine("struct FitValues { Double_t mean; Double_t sigma; Double_t mean_fit; \
                                      Double_t sigma_fit; Bool_t fix_mean; Bool_t fix_sigma; \
                                      Double_t nsigma_sig; Double_t nsigma_sideband; \
                                      Double_t fit_range_low; Double_t fit_range_up; \
                                      Bool_t success;};")

# pylint: disable=wrong-import-position, ungrouped-imports
from ROOT import FitValues

def fixpar(massmin, massmax, masspeak, range_signal):
    par_fix1 = Double(massmax-massmin)
    par_fix2 = Double(massmax+massmin)
    par_fix3 = Double(massmax*massmax*massmax-massmin*massmin*massmin)
    par_fix4 = Double(masspeak)
    par_fix5 = Double(range_signal)
    return par_fix1, par_fix2, par_fix3, par_fix4, par_fix5

def gaus_fit_func(xval, par):
    return par[0] / sqrt(2. * pi) / par[2] * \
           exp(-(xval[0] - par[1]) * (xval[0] - par[1]) / 2. / par[2] / par[2])

def signal_func(func_name, sgnfunc, fit_range_low, fit_range_up):
    if sgnfunc != "kGaus":
        get_logger().fatal("Unknown signal fit function %s", sgnfunc)
    func = TF1(func_name, gaus_fit_func, fit_range_low, fit_range_up, 3)
    func.SetParNames("Int", "Mean", "Sigma")
    return func

def pol1_func_sidebands(xval, par):
    if par[6] > 0 and abs(xval[0] - par[4]) < par[5]:
        TF1.RejectPoint()
        return 0.
    return par[0] / par[2] + par[1] * (xval[0] - 0.5 * par[3])

def pol2_func_sidebands(xval, par):
    if par[8] > 0 and abs(xval[0] - par[6]) < par[7]:
        TF1.RejectPoint()
        return 0.
    return par[0] / par[3] + par[1] * (xval[0] - 0.5 * par[4]) + par[2] * \
           (xval[0] * xval[0] - 1/3. * par[5] / par[3])

def bkg_fit_func(func_name, func_type, massmin, massmax, integralhisto, masspeak, range_signal,
                 reject_signal_region=True):
    # Immediately exit if function is unknown
    if func_type not in ["Pol1", "Pol2"]:
        get_logger().fatal("Unkown background fit function %s", func_type)

    par_fix1, par_fix2, par_fix3, par_fix4, par_fix5 = \
        fixpar(massmin, massmax, masspeak, range_signal)

    # In the following return asap
    if func_type == "Pol1":
        back_fit = TF1(func_name, pol1_func_sidebands, massmin, massmax, 7)
        back_fit.SetParNames("BkgInt", "Slope", "", "", "", "")
        back_fit.SetParameters(integralhisto, -100.)
        back_fit.FixParameter(2, par_fix1)
        back_fit.FixParameter(3, par_fix2)
        back_fit.FixParameter(4, par_fix4)
        back_fit.FixParameter(5, par_fix5)
        back_fit.FixParameter(6, 1 if reject_signal_region else -1)
        return back_fit

    back_fit = TF1(func_name, pol2_func_sidebands, massmin, massmax, 9)
    back_fit.SetParNames("BkgInt", "Coeff1", "Coeff2", "AlwaysFixedPar1", "AlwaysFixedPar2",
                         "AlwaysFixedPar3", "HelperParMassPeak", "HelperParSigRange",
                         "HelperParRejectSigRange")
    back_fit.SetParameters(integralhisto, -10., 5.)
    back_fit.FixParameter(3, par_fix1)
    back_fit.FixParameter(4, par_fix2)
    back_fit.FixParameter(5, par_fix3)
    # Set helper parameters such that bkg fit does not account for that in signal range
    back_fit.FixParameter(6, par_fix4)
    back_fit.FixParameter(7, par_fix5)
    back_fit.FixParameter(8, 1 if reject_signal_region else -1)
    return back_fit

def tot_func(bkgfunc, massmax, massmin):
    # Immediately exit if function is unknown
    if bkgfunc not in ["Pol1", "Pol2"]:
        get_logger().fatal("Unkown background fit dunction %s", bkgfunc)

    # in the following return asap
    if bkgfunc == "Pol1":
        return "[0]/(%f)+[1]*(x-0.5*(%f))                                    \
                +[2]/(sqrt(2.*pi))/[4]*(exp(-(x-[3])*(x-[3])/2./[4]/[4]))" % \
                ((massmax-massmin), (massmax+massmin))

    return "[0]/(%f)+[1]*(x-0.5*(%f))+[2]*(x*x-1/3.*(%f)/(%f))           \
            +[3]/(sqrt(2.*pi))/[5]*(exp(-(x-[4])*(x-[4])/2./[5]/[5]))" % \
           ((massmax - massmin), (massmax + massmin),
            (massmax * massmax * massmax - massmin * massmin * massmin),
            (massmax-massmin))


# pylint: disable=too-many-instance-attributes
class Fitter:
    species = "fitter"
    def __init__(self):

        self.logger = get_logger()
        # These are filled after the fit has been done
        self.yield_sig = None
        self.yield_sig_err = None
        self.yield_bkg = None
        self.yield_bkg_err = None
        self.mean_fit = None
        self.mean_err_fit = None
        self.sigma_fit = None
        self.sigma_err_fit = None
        self.significance = None
        self.errsignificance = None

        # Some initial values
        self.mean = None
        self.sigma = None
        self.fix_mean = None
        self.fix_sigma = None

        # The following are derived after the initialization
        self.tot_fit_func = None
        self.sig_fit_func = None
        self.bkg_sideband_fit_func = None
        self.bkg_fit_func = None
        self.bkg_tot_fit_func = None

        # Some further options
        self.fit_options = ""
        self.nsigma_sideband = None
        self.nsigma_sig = None
        self.fit_range_low = None
        self.fit_range_up = None

        # The original histogram to be fitted
        self.histo_to_fit = None
        # The histogram after background subtraction after the fit has been performed
        #self.histo_sideband_sub = None

        # Flag whether it has been fitted
        self.fitted = False
        self.fit_success = False

    # pylint: disable=too-many-arguments
    def initialize(self, histo, sig_func_name, bkg_func_name, rebin, mean, sigma, fix_mean,
                   fix_sigma, nsigma_sideband, nsigma_sig, fit_range_low, fit_range_up):

        self.histo_to_fit = histo.Clone(histo.GetName() + "_for_fit")
        self.histo_to_fit.Rebin(rebin)
        self.mean = mean
        self.sigma = sigma
        self.fix_mean = fix_mean
        self.fix_sigma = fix_sigma
        self.nsigma_sideband = nsigma_sideband
        self.nsigma_sig = nsigma_sig
        # Make the fit range safe
        self.fit_range_low = max(fit_range_low, self.histo_to_fit.GetBinLowEdge(2))
        self.fit_range_up = min(fit_range_up,
                                self.histo_to_fit.GetBinLowEdge(self.histo_to_fit.GetNbinsX()))

        bkg_int_initial = Double(histo.Integral(self.histo_to_fit.FindBin(fit_range_low),
                                                self.histo_to_fit.FindBin(fit_range_up),
                                                "width"))
        self.sig_fit_func = signal_func("sig_fit", sig_func_name, fit_range_low, fit_range_up)
        self.bkg_sideband_fit_func = bkg_fit_func("bkg_fit_sidebands", bkg_func_name, fit_range_low,
                                                  fit_range_up, bkg_int_initial, mean,
                                                  nsigma_sideband * sigma)
        self.bkg_fit_func = bkg_fit_func("bkg_fit", bkg_func_name, fit_range_low, fit_range_up,
                                         bkg_int_initial, mean, nsigma_sideband * sigma, False)
        self.bkg_tot_fit_func = bkg_fit_func("bkg_fit_from_tot_fit", bkg_func_name, fit_range_low,
                                             fit_range_up, bkg_int_initial, mean,
                                             nsigma_sideband * sigma, False)
        self.tot_fit_func = TF1("tot_fit", tot_func(bkg_func_name, fit_range_up, fit_range_low),
                                fit_range_low, fit_range_up)
        self.fitted = False
        self.fit_success = False

    def do_likelihood(self):
        self.fit_options = "L,E"


    def update_check_signal_fit(self):
        error_list = []
        if self.yield_sig < 0. < self.sigma_fit or self.sigma_fit < 0. < self.yield_sig:
            error_list.append(f"Both integral pre-factor and sigma have to have the same sign. " \
                              f"However, pre-factor is {self.yield_sig} and sigma is " \
                              f"{self.sigma_fit}.")
        if self.mean_fit < 0.:
            error_list.append(f"Mean is negative: {self.mean_fit}")

        if abs(self.sigma_fit) > 100 * self.sigma:
            error_list.append(f"Fitted sigma is larger than 100 times initial sigma " \
                              f"{self.sigma:.4f} vs. {self.sigma_fit:.4f}")
        if error_list:
            return "\n".join(error_list)

        # Seems sane, set both sigma and int_sig to positive values
        self.yield_sig = abs(self.yield_sig)
        self.sig_fit_func.SetParameter(0, abs(self.sig_fit_func.GetParameter(0)))
        self.sigma_fit = abs(self.sigma_fit)
        self.sig_fit_func.SetParameter(2, abs(self.sig_fit_func.GetParameter(2)))

        return ""

    def derive_yields(self):
        self.logger.info("calculate signal, backgroud, S/B, significance")
        self.mean_fit = self.sig_fit_func.GetParameter(1)
        self.mean_err_fit = self.sig_fit_func.GetParError(1)
        # Could be negative together with the integral pre-factor
        self.sigma_fit = self.sig_fit_func.GetParameter(2)
        self.sigma_err_fit = self.sig_fit_func.GetParError(2)

        minMass_fit = self.mean_fit - self.nsigma_sig * self.sigma_fit
        maxMass_fit = self.mean_fit + self.nsigma_sig * self.sigma_fit
        leftBand = self.histo_to_fit.FindBin(self.mean_fit - self.nsigma_sideband * self.sigma_fit)
        rightBand = self.histo_to_fit.FindBin(self.mean_fit + self.nsigma_sideband * self.sigma_fit)
        intB = self.histo_to_fit.Integral(1, leftBand) + \
               self.histo_to_fit.Integral(rightBand, self.histo_to_fit.GetNbinsX())
        sum2 = 0.
        for i_left in range(1, leftBand + 1):
            sum2 += self.histo_to_fit.GetBinError(i_left) * self.histo_to_fit.GetBinError(i_left)
        for i_right in range(rightBand, (self.histo_to_fit.GetNbinsX()) + 1):
            sum2 += self.histo_to_fit.GetBinError(i_right) * self.histo_to_fit.GetBinError(i_right)
        intBerr = sqrt(sum2)
        self.yield_bkg = self.bkg_tot_fit_func.Integral(minMass_fit, maxMass_fit) / \
                         Double(self.histo_to_fit.GetBinWidth(1))
        #if background <= 0:
        #    return -1, -1
        self.yield_bkg_err = 0
        if intB > 0:
            # Rescale the error
            self.yield_bkg_err = intBerr / intB * self.yield_bkg

        self.logger.info("Background: %s, error background: %s", self.yield_bkg, self.yield_bkg_err)
        self.yield_sig = self.sig_fit_func.GetParameter(0) / \
                         Double(self.histo_to_fit.GetBinWidth(1))
        self.yield_sig_err = self.sig_fit_func.GetParError(0) / \
                             Double(self.histo_to_fit.GetBinWidth(1))




        self.logger.info("Raw yield: %f, raw yield error: %f", self.yield_sig, self.yield_sig_err)
        errSigSq = self.yield_sig_err * self.yield_sig_err
        errBkgSq = self.yield_bkg_err * self.yield_bkg_err
        sigPlusBkg = self.yield_bkg + self.yield_sig
        self.significance = 0
        self.errsignificance = 0
        if sigPlusBkg > 0 and self.yield_sig > 0:
            self.significance = self.yield_sig / (sqrt(sigPlusBkg))
            self.errsignificance = self.significance * (sqrt((errSigSq + errBkgSq) / \
                              (4. * sigPlusBkg * sigPlusBkg) +                  \
                              (self.yield_bkg / sigPlusBkg) * errSigSq /            \
                              self.yield_sig / self.yield_sig))

        self.logger.info("Significance: %f, error significance: %f", self.significance,
                         self.errsignificance)

    def bincount(self, nsigma, use_integral=True):

        if not self.fitted:
            self.logger.error("Cannot compute bincount. Fit required first!")
            return None, None

        # Now yield from bin count
        bincount = 0.
        bincount_err = 0.
        leftBand = self.histo_to_fit.FindBin(self.mean_fit - nsigma * self.sigma_fit)
        rightBand = self.histo_to_fit.FindBin(self.mean_fit + nsigma * self.sigma_fit)
        for b in range(leftBand, rightBand + 1, 1):
            bkg_count = 0
            if use_integral:
                bkg_count = self.bkg_fit_func.Integral(self.histo_to_fit.GetBinLowEdge(b),
                                                       self.histo_to_fit.GetBinLowEdge(b) + \
                                                       self.histo_to_fit.GetBinWidth(b)) / \
                                                       self.histo_to_fit.GetBinWidth(b)
            else:
                bkg_count = self.bkg_fit_func.Eval(self.histo_to_fit.GetBinCenter(b))

            bincount += self.histo_to_fit.GetBinContent(b) - bkg_count
            bincount_err += self.histo_to_fit.GetBinError(b) * self.histo_to_fit.GetBinError(b)

        return bincount, sqrt(bincount_err)

    def save(self, root_dir):
        if not self.fitted:
            self.logger.error("Not fitted yet, nothing to save")
            return
        root_dir.cd()

        self.sig_fit_func.Write()
        self.bkg_sideband_fit_func.Write()
        self.bkg_fit_func.Write()
        self.bkg_tot_fit_func.Write()
        self.tot_fit_func.Write()
        self.histo_to_fit.Write("histo_to_fit")
        fit_values = FitValues()
        fit_values.mean = self.mean
        fit_values.sigma = self.sigma
        fit_values.mean_fit = self.mean_fit
        fit_values.sigma_fit = self.sigma_fit
        fit_values.fix_mean = self.fix_mean
        fit_values.fix_sigma = self.fix_sigma
        fit_values.fit_range_low = self.fit_range_low
        fit_values.fit_range_up = self.fit_range_up
        fit_values.nsigma_sig = self.nsigma_sig
        fit_values.nsigma_sideband = self.nsigma_sideband
        fit_values.success = self.fit_success

        root_dir.WriteObject(fit_values, "fit_values")

    def load(self, root_dir, force=False):
        if self.fitted and not force:
            self.logger.warning("Was fitted before and will be overwritten with what is found " \
                                "in ROOT dir%s", root_dir.GetName())

        self.sig_fit_func = root_dir.Get("sig_fit")
        self.bkg_sideband_fit_func = root_dir.Get("bkg_fit_sidebands")
        self.bkg_tot_fit_func = root_dir.Get("bkg_fit_from_tot_fit")
        self.bkg_fit_func = root_dir.Get("bkg_fit")
        self.tot_fit_func = root_dir.Get("tot_fit")
        self.histo_to_fit = root_dir.Get("histo_to_fit")

        fit_values = FitValues()
        fit_values = root_dir.Get("fit_values")
        self.mean = fit_values.mean
        self.sigma = fit_values.sigma
        self.mean_fit = fit_values.mean_fit
        self.sigma_fit = fit_values.sigma_fit
        self.fix_mean = fit_values.fix_mean
        self.fix_sigma = fit_values.fix_sigma
        self.fit_range_low = fit_values.fit_range_low
        self.fit_range_up = fit_values.fit_range_up
        self.nsigma_sideband = fit_values.nsigma_sideband
        self.nsigma_sig = fit_values.nsigma_sig

        self.derive_yields()
        # Check the signal fit
        error = self.update_check_signal_fit()

        self.fitted = True
        self.fit_success = (error == "")

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches,
    # pylint: disable=too-many-statements
    def fit(self):
        """
        Some comments:
            -> The object 'back_fit' is fitted using only the side-bands
            -> The object 'tot_fit' is fitted to the entire extracted inv-mass distribution
            -> The object 'back_refit' is not used for fitting but only stores the background fit
               parameters from fitted 'tot_fit'
            -> The object 'sig_fit' is not used for fitting but only stores the signal fit
               parameters from fitted 'tot_fit'
        """

        self.logger.info("Do fitting")

        TVirtualFitter.SetDefaultFitter("Minuit")

        # Prepare what can be prepared
        npar_sig = self.sig_fit_func.GetNpar()
        # Need to do it that way since background functions have a few dummy parameters
        npar_bkg = self.tot_fit_func.GetNpar() - npar_sig

        self.logger.info("Initial parameters for signal fit are")
        print(f"mean = {self.mean}\nsigma = {self.sigma}")

        self.logger.debug("fit background (just side bands)")
        self.histo_to_fit.Fit(self.bkg_sideband_fit_func, ("R,%s,+,0" % (self.fit_options)))

        # Prepare a function to store the signal parameters which will finally be extracted
        # from the total fit. So this is just a helper for now
        minForSig = self.mean - self.nsigma_sideband * self.sigma
        maxForSig = self.mean + self.nsigma_sideband * self.sigma
        binForMinSig = self.histo_to_fit.FindBin(minForSig)
        binForMaxSig = self.histo_to_fit.FindBin(maxForSig)
        sum_tot = 0.
        sumback = 0.
        for ibin in range(binForMinSig, binForMaxSig + 1):
            sum_tot += self.histo_to_fit.GetBinContent(ibin)
            sumback += self.bkg_sideband_fit_func.Eval(self.histo_to_fit.GetBinCenter(ibin))
        integsig = Double((sum_tot - sumback) * (self.histo_to_fit.GetBinWidth(1)))

        self.sig_fit_func.SetParameter(0, integsig)
        self.sig_fit_func.SetParameter(1, self.mean)
        self.sig_fit_func.SetParameter(2, self.sigma)

        self.logger.info("fit all (signal + background)")
        self.tot_fit_func.SetLineColor(4)
        parmin = Double()
        parmax = Double()
        for ipar in range(0, npar_sig):
            self.tot_fit_func.SetParameter(ipar + npar_bkg, self.sig_fit_func.GetParameter(ipar))
            self.sig_fit_func.GetParLimits(ipar, parmin, parmax)
            self.tot_fit_func.SetParLimits(ipar + npar_bkg, parmin, parmax)
        for ipar in range(0, npar_bkg):
            self.tot_fit_func.SetParameter(ipar, self.bkg_sideband_fit_func.GetParameter(ipar))
            self.bkg_fit_func.SetParameter(ipar, self.bkg_sideband_fit_func.GetParameter(ipar))
        if self.fix_mean:
            # Mass peak would be fixed to what user sets
            self.tot_fit_func.FixParameter(npar_bkg + 1, self.mean)
        if self.fix_sigma is True:
            # Sigma would be fixed to what the fit to MC gives
            self.tot_fit_func.FixParameter(npar_bkg + 2,
                                           self.tot_fit_func.GetParameter(npar_bkg + 2))
        self.histo_to_fit.Fit(self.tot_fit_func, ("R,%s,+,0" % (self.fit_options)))

        for ipar in range(0, npar_bkg):
            self.bkg_tot_fit_func.SetParameter(ipar, self.tot_fit_func.GetParameter(ipar))
            self.bkg_tot_fit_func.SetParError(ipar, self.tot_fit_func.GetParameter(ipar))
        for ipar in range(npar_bkg, (npar_bkg + npar_sig)):
            self.sig_fit_func.SetParameter(ipar - npar_bkg, self.tot_fit_func.GetParameter(ipar))
            self.sig_fit_func.SetParError(ipar - npar_bkg, self.tot_fit_func.GetParError(ipar))

        self.derive_yields()

        # Check the signal fit
        error = self.update_check_signal_fit()
        if error != "":
            self.logger.error("Signal fit probably bad for following reasons:\n%s", error)

        self.fitted = True
        self.fit_success = (error == "")
        return self.fit_success

    def draw_fit(self, save_name, flag_plot_message=None, shade_regions=False):
        #Draw
        self.histo_to_fit.GetXaxis().SetTitle("Invariant Mass L_{c}^{+}(GeV/c^{2})")
        self.histo_to_fit.SetStats(0)

        c1 = TCanvas('c1', 'The Fit Canvas', 700, 700)
        c1.cd()
        gStyle.SetOptStat(0)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)
        c1.cd()
        self.histo_to_fit.GetXaxis().SetRangeUser(self.fit_range_low, self.fit_range_up)
        # Adjust y-range for good readability
        histo_min = self.histo_to_fit.GetMinimum() * 0.9
        histo_max = self.histo_to_fit.GetMaximum() + (self.histo_to_fit.GetMaximum() - histo_min)
        self.histo_to_fit.GetYaxis().SetRangeUser(histo_min, histo_max)
        self.histo_to_fit.SetMarkerStyle(20)
        self.histo_to_fit.SetMarkerSize(1)
        #histo.SetMinimum(0.)
        self.histo_to_fit.Draw("PE")
        self.bkg_tot_fit_func.Draw("same")
        self.tot_fit_func.Draw("same")
        c1.Update()

        # Shading sideband area
        sideband_fill_left = None
        sideband_fill_right = None
        sig_fill = None
        bkg_fill = None
        if shade_regions:
            sideband_fill_left = self.bkg_tot_fit_func.Clone("bkg_fit_fill_left")
            sideband_fill_left.SetRange(self.mean_fit - 9 * self.sigma_fit,
                                        self.mean_fit - self.nsigma_sideband * self.sigma_fit)
            sideband_fill_left.SetLineWidth(0)
            sideband_fill_left.SetFillColor(self.bkg_tot_fit_func.GetLineColor())
            sideband_fill_left.SetFillStyle(3001)
            sideband_fill_left.Draw("same fc")

            sideband_fill_right = self.bkg_tot_fit_func.Clone("bkg_fit_fill_right")
            sideband_fill_right.SetRange(self.mean_fit + self.nsigma_sideband * self.sigma_fit,
                                         self.mean_fit + 9 * self.sigma_fit)
            sideband_fill_right.SetLineWidth(0)
            sideband_fill_right.SetFillColor(self.bkg_tot_fit_func.GetLineColor())
            sideband_fill_right.SetFillStyle(3001)
            sideband_fill_right.Draw("same fc")

            # Shading bakground in signal region
            bkg_fill = self.bkg_tot_fit_func.Clone("bkg_fit_under_sig_fill")
            bkg_fill.SetRange(self.mean_fit - self.nsigma_sig * self.sigma_fit,
                              self.mean_fit + self.nsigma_sig * self.sigma_fit)
            bkg_fill.SetLineWidth(0)
            bkg_fill.SetFillColor(kRed + 2)
            bkg_fill.SetFillStyle(3001)
            bkg_fill.Draw("same fc")

            # Shading signal above background
            n_points = 100
            dx = (2 * self.nsigma_sig * self.sigma_fit) / n_points
            sig_fill = TGraph(2 * n_points)
            sig_fill.SetFillColor(kGreen + 2)
            sig_fill.SetFillStyle(3001)
            range_low = self.mean_fit - self.nsigma_sig * self.sigma_fit
            range_up = self.mean_fit + self.nsigma_sig * self.sigma_fit
            for ip in range(n_points):
                sig_fill.SetPoint(ip, range_low + ip * dx,
                                  self.tot_fit_func.Eval(range_low + ip * dx))
                sig_fill.SetPoint(n_points + ip, range_up - ip * dx,
                                  self.bkg_tot_fit_func.Eval(range_up - ip * dx))
            sig_fill.Draw("f")

        #write info.
        pinfos = TPaveText(0.12, 0.7, 0.47, 0.89, "NDC")
        pinfos.SetBorderSize(0)
        pinfos.SetFillStyle(0)
        pinfos.SetTextAlign(11)
        pinfos.SetTextSize(0.03)
        pinfom = TPaveText(0.5, 0.7, 1., .89, "NDC")
        pinfom.SetTextAlign(11)
        pinfom.SetBorderSize(0)
        pinfom.SetFillStyle(0)
        pinfom.SetTextColor(kBlue)
        pinfom.SetTextSize(0.03)
        chisquare_ndf = self.tot_fit_func.GetNDF()
        chisquare_ndf = self.tot_fit_func.GetChisquare() / chisquare_ndf if chisquare_ndf > 0. \
                else 0.
        pinfom.AddText("#chi^{2}/NDF = %f" % (chisquare_ndf))
        pinfom.AddText("%s = %.3f #pm %.3f" % (self.sig_fit_func.GetParName(1),\
            self.sig_fit_func.GetParameter(1), self.sig_fit_func.GetParError(1)))
        pinfom.AddText("%s = %.3f #pm %.3f" % (self.sig_fit_func.GetParName(2),\
            self.sig_fit_func.GetParameter(2), self.sig_fit_func.GetParError(2)))
        pinfom.Draw()
        flag_info = None
        if flag_plot_message is not None:
            flag_info = TPaveText(0.5, 0.5, 1., 0.68, "NDC")
            flag_info.SetBorderSize(0)
            flag_info.SetFillStyle(0)
            flag_info.SetTextAlign(11)
            flag_info.SetTextSize(0.03)
            for t in flag_plot_message:
                text = flag_info.AddText(t)
                text.SetTextColor(kRed + 2)
            flag_info.Draw()

        sig_text = pinfos.AddText("S = %.0f #pm %.0f " % (self.yield_sig, self.yield_sig_err))
        sig_text.SetTextColor(kGreen + 2)
        bkg_text = pinfos.AddText("B (%.0f#sigma) = %.0f #pm %.0f" % \
            (self.nsigma_sig, self.yield_bkg, self.yield_bkg_err))
        bkg_text.SetTextColor(kRed + 2)
        sig_over_back = self.yield_sig / self.yield_bkg if self.yield_bkg > 0. else 0.
        pinfos.AddText("S/B (%.0f#sigma) = %.4f " % (self.nsigma_sig, sig_over_back))
        pinfos.AddText("Signif (%.0f#sigma) = %.1f #pm %.1f " %\
            (self.nsigma_sig, self.significance, self.errsignificance))
        pinfos.Draw()

        c1.Update()
        c1.SaveAs(save_name)
        c1.Close()
