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

import math
# pylint: disable=import-error,no-name-in-module
from ROOT import TF1, gStyle, TCanvas, TPaveText, Double, TVirtualFitter, \
                 kBlue
from  machine_learning_hep.logger import get_logger

def fixpar(massmin, massmax, masspeak, range_signal):
    par_fix1 = Double(massmax-massmin)
    par_fix2 = Double(massmax+massmin)
    par_fix3 = Double(massmax*massmax*massmax-massmin*massmin*massmin)
    par_fix4 = Double(masspeak)
    par_fix5 = Double(range_signal)
    return par_fix1, par_fix2, par_fix3, par_fix4, par_fix5

def gaus_fit_func(xval, par):
    return par[0] / math.sqrt(2. * math.pi) / par[2] * \
           math.exp(-(xval[0] - par[1]) * (xval[0] - par[1]) / 2. / par[2] / par[2])

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
    back_fit.SetParameters(integralhisto, -10., 5)
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
        return "[0]/(%s)+[1]*(x-0.5*(%s))                                    \
                +[2]/(sqrt(2.*pi))/[4]*(exp(-(x-[3])*(x-[3])/2./[4]/[4]))" % \
                ((massmax-massmin), (massmax+massmin))

    return "[0]/(%s)+[1]*(x-0.5*(%s))+[2]*(x*x-1/3.*(%s)/(%s))           \
            +[3]/(sqrt(2.*pi))/[5]*(exp(-(x-[4])*(x-[4])/2./[5]/[5]))" % \
           ((massmax - massmin), (massmax + massmin),
            (massmax * massmax * massmax - massmin * massmin * massmin),
            (massmax-massmin))


# pylint: disable=too-many-instance-attributes
class Fitter:
    def __init__(self):

        self.logger = get_logger()
        # These are filled after the fit has been done
        self.yield_sig = None
        self.yield_sig_err = None
        self.yield_bkg = None
        self.yield_bkg_err = None
        self.sig_bincount = None
        self.sig_bincount_err = None
        self.bkg_bincount = None
        self.bkg_bincount_err = None
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

    # pylint: disable=too-many-arguments
    def initialize(self, histo, sig_func_name, bkg_func_name, rebin, mean, sigma, fix_mean,
                   fix_sigma, nsigma_sideband, nsigma_sig, fit_range_low, fit_range_up):

        self.mean = mean
        self.sigma = sigma
        self.fix_mean = fix_mean
        self.fix_sigma = fix_sigma
        self.nsigma_sideband = nsigma_sideband
        self.nsigma_sig = nsigma_sig
        self.fit_range_low = fit_range_low
        self.fit_range_up = fit_range_up
        self.histo_to_fit = histo.Clone(histo.GetName() + "_for_fit")
        self.histo_to_fit.Rebin(rebin)

        self.nsigma_sideband = nsigma_sideband

        bkg_int_initial = Double(histo.Integral(histo.FindBin(fit_range_low),
                                                histo.FindBin(fit_range_up), "width"))
        self.sig_fit_func = signal_func("sig_fit", sig_func_name, fit_range_low, fit_range_up)
        self.bkg_sideband_fit_func = bkg_fit_func("bkg_fit_sidebands", bkg_func_name, fit_range_low,
                                                  fit_range_up, bkg_int_initial, mean,
                                                  nsigma_sideband * sigma)
        self.bkg_tot_fit_func = bkg_fit_func("bkg_fit_from_tot_fit", bkg_func_name,
                                             fit_range_low, fit_range_up, bkg_int_initial, mean,
                                             nsigma_sideband * sigma, False)
        self.tot_fit_func = TF1("tot_fit", tot_func(bkg_func_name, fit_range_up, fit_range_low),
                                fit_range_low, fit_range_up)

    def do_likelihood(self):
        self.fit_options = "L,E"

    def update_check_signal_fit(self):
        error_list = []
        int_sig = self.sig_fit_func.GetParameter(0)
        mean = self.sig_fit_func.GetParameter(1)
        sigma = self.sig_fit_func.GetParameter(2)
        if int_sig < 0. < sigma or sigma < 0. < int_sig:
            error_list.append(f"Both integral pre-factor and sigma have to have the same sign. " \
                              f"However, pre-factor is {int_sig} and sigma is {sigma}.")
        if mean < 0.:
            error_list.append(f"Mean is negative: {mean}")

        if error_list:
            return "\n".join(error_list)

        # Seems sane, set both sigma and int_sig to positive values
        self.sig_fit_func.SetParameter(0, abs(int_sig))
        self.sig_fit_func.SetParameter(2, abs(sigma))

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
        intBerr = math.sqrt(sum2)
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
        if sigPlusBkg > 0:
            self.significance = self.yield_sig / (math.sqrt(sigPlusBkg))
            self.errsignificance = self.significance * (math.sqrt((errSigSq + errBkgSq) / \
                              (4. * sigPlusBkg * sigPlusBkg) +                  \
                              (self.yield_bkg / sigPlusBkg) * errSigSq /            \
                              self.yield_sig / self.yield_sig))

        self.logger.info("Significance: %f, error significance: %f", self.significance,
                         self.errsignificance)

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
        if self.fix_mean:
            # Mass peak would be fixed to what user sets
            self.tot_fit_func.FixParameter(npar_bkg + 1, self.mean)
        if self.fix_sigma is True:
            # Sigma would be fixed to what the fit to MC gives
            self.tot_fit_func.FixParameter(npar_bkg + 2,
                                           self.tot_fit_func.GetParameter(npar_bkg + 2))
        #tot_fit.SetParNames(*par_names_bkg, *par_names_sig)
        self.histo_to_fit.Fit(self.tot_fit_func, ("R,%s,+,0" % (self.fit_options)))

        for ipar in range(0, npar_bkg):
            self.bkg_tot_fit_func.SetParameter(ipar, self.tot_fit_func.GetParameter(ipar))
            self.bkg_tot_fit_func.SetParError(ipar, self.tot_fit_func.GetParameter(ipar))
        for ipar in range(npar_bkg, (npar_bkg + npar_sig)):
            self.sig_fit_func.SetParameter(ipar - npar_bkg, self.tot_fit_func.GetParameter(ipar))
            self.sig_fit_func.SetParError(ipar - npar_bkg, self.tot_fit_func.GetParError(ipar))

        # Check the signal fit
        error = self.update_check_signal_fit()
        if error != "":
            self.logger.error("Signal fit probably bad for following reasons:\n%s", error)

        self.derive_yields()

    def draw_fit(self, save_name):
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
        pinfom.AddText("#chi^{2}/NDF = %f" % (self.tot_fit_func.GetChisquare() / \
                self.tot_fit_func.GetNDF()))
        pinfom.AddText("%s = %.3f #pm %.3f" % (self.sig_fit_func.GetParName(1),\
            self.sig_fit_func.GetParameter(1), self.sig_fit_func.GetParError(1)))
        pinfom.AddText("%s = %.3f #pm %.3f" % (self.sig_fit_func.GetParName(2),\
            self.sig_fit_func.GetParameter(2), self.sig_fit_func.GetParError(2)))
        pinfom.Draw()
        pinfos.AddText("S = %.0f #pm %.0f " % (self.yield_sig, self.yield_sig_err))
        pinfos.AddText("B (%.0f#sigma) = %.0f #pm %.0f" % \
            (self.nsigma_sig, self.yield_bkg, self.yield_bkg_err))
        sig_over_back = self.yield_sig / self.yield_bkg if self.yield_bkg > 0. else 0.
        pinfos.AddText("S/B (%.0f#sigma) = %.4f " % (self.nsigma_sig, sig_over_back))
        pinfos.AddText("Signif (%.0f#sigma) = %.1f #pm %.1f " %\
            (self.nsigma_sig, self.significance, self.errsignificance))
        pinfos.Draw()

        c1.Update()
        c1.SaveAs(save_name)
        c1.Close()
