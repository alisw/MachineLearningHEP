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
from ROOT import TF1, gStyle, TCanvas, TPaveText, Double, TVirtualFitter, kBlue, kGray # pylint: disable=import-error,no-name-in-module
from  machine_learning_hep.logger import get_logger

def fixpar(massmin, massmax, masspeak, range_signal):
    par_fix1 = Double(massmax-massmin)
    par_fix2 = Double(massmax+massmin)
    par_fix3 = Double(massmax*massmax*massmax-massmin*massmin*massmin)
    par_fix4 = Double(masspeak)
    par_fix5 = Double(range_signal)
    return par_fix1, par_fix2, par_fix3, par_fix4, par_fix5

def update_check_signal_fit(sig_fit):
    error_list = []
    int_sig = sig_fit.GetParameter(0)
    mean = sig_fit.GetParameter(1)
    sigma = sig_fit.GetParameter(2)
    if int_sig < 0. < sigma or sigma < 0. < int_sig:
        error_list.append(f"Both integral pre-factor and sigma have to have the same sign. " \
                          f"However, pre-factor is {int_sig} and sigma is {sigma}.")
    if mean < 0.:
        error_list.append(f"Mean is negative: {mean}")

    if error_list:
        return "\n".join(error_list)

    # Seems sane, set both sigma and int_sig to positive values
    sig_fit.SetParameter(0, abs(int_sig))
    sig_fit.SetParameter(2, abs(sigma))

    return ""

def signal_func(sgnfunc):
    if sgnfunc != "kGaus":
        get_logger().fatal("Unknown signal fit function %s", sgnfunc)
    return "[0]/(sqrt(2.*pi))/[2]*(exp(-(x-[1])*(x-[1])/2./[2]/[2]))"

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
        get_logger().fatal("Unkown background fit dunction %s", func_type)

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
    back_fit.SetParNames("BkgInt", "Coef1", "Coef2", "AlwaysFixedPar1", "AlwaysFixedPar2",
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

# pylint: disable=too-many-arguments, too-many-locals, too-many-branches,
# pylint: disable=too-many-statements
def fitter(histo, case, sgnfunc, bkgfunc, masspeak, rebin, dolikelihood, setinitialgaussianmean,
           setfixgaussiansigma, sigma_sig, massmin, massmax, fixedmean, outputfolder, suffix,
           draw_side_band_fit=False):
    """
    Some comments:
        -> The object 'back_fit' is fitted using only the side-bands
        -> The object 'tot_fit' is fitted to the entire extracted inv-mass distribution
        -> The object 'back_refit' is not used for fitting but only stores the background fit
           parameters from fitted 'tot_fit'
        -> The object 'sig_fit' is not used for fitting but only stores the signal fit
           parameters from fitted 'tot_fit'
    """
    logger = get_logger()

    logger.info("Do fitting")
    logger.info("Argument \"setinitialgaussianmean\" set to %u, however not used at the moment",
                setinitialgaussianmean)

    if "Lb" not in case and "Lc" not in case and "D0" not in case and "Ds" not in case:
        logger.warning("Can only do the fit for Lc, D0 or Ds, however found case %s", case)
        return -1, -1

    histo.GetXaxis().SetTitle("Invariant Mass L_{c}^{+}(GeV/c^{2})")
    histo.Rebin(rebin)
    histo.SetStats(0)
    TVirtualFitter.SetDefaultFitter("Minuit")

    # Prepare what can be prepared
    npar_sig = 3
    npar_bkg = 3 if bkgfunc == "Pol2" else 2
    par_names_sig = ["IntSig", "Mean", "Sigma"]
    par_names_bkg = ["IntBkg", "BkgCoeff1", "BkgCoeff2"] if bkgfunc == "Pol2" \
                                                         else ["IntBkg", "BkgCoeff1"]

    fitOption = "L,E" if dolikelihood else ""

    logger.info("Initial parameters for signal fit are")
    print(f"mean = {masspeak}\nsigma = {sigma_sig}")

    logger.debug("fit background (just side bands)")
    nSigma4SideBands = 4.
    range_signal = nSigma4SideBands * sigma_sig
    integralhisto = Double(histo.Integral(histo.FindBin(massmin), histo.FindBin(massmax), "width"))
    back_fit = bkg_fit_func("bkg_fit_sidebands", bkgfunc, massmin, massmax, integralhisto,
                            masspeak, range_signal)
    histo.Fit("bkg_fit_sidebands", ("R,%s,+,0" % (fitOption)))

    #logger.debug("refit background (all range)")
    # Prepare this function to store the background fit parameters obtained from the total fit
    back_refit = bkg_fit_func("bkg_fit_from_tot_fit", bkgfunc, massmin, massmax, integralhisto,
                              masspeak, range_signal, False)

    # Prepare a function to store the signal parameters which will finally be extracted
    # from the total fit. So this is just a helper for now
    minForSig = masspeak - 4. * sigma_sig
    maxForSig = masspeak + 4. * sigma_sig
    binForMinSig = histo.FindBin(minForSig)
    binForMaxSig = histo.FindBin(maxForSig)
    sum_tot = 0.
    sumback = 0.
    for ibin in range(binForMinSig, binForMaxSig + 1):
        sum_tot += histo.GetBinContent(ibin)
        sumback += back_fit.Eval(histo.GetBinCenter(ibin))
    integsig = Double((sum_tot - sumback) * (histo.GetBinWidth(1)))
    sig_fit = TF1("sig_fit", signal_func(sgnfunc), massmin, massmax)
    sig_fit.SetParameter(0, integsig)
    sig_fit.SetParameter(1, masspeak)
    sig_fit.SetParameter(2, sigma_sig)
    sig_fit.SetParNames(*par_names_sig)

    logger.debug("Prepare parameters for total fit")

    logger.info("fit all (signal + background)")
    tot_fit = TF1("tot_fit", tot_func(bkgfunc, massmax, massmin), massmin, massmax)
    tot_fit.SetLineColor(4)
    parmin = Double()
    parmax = Double()
    for ipar in range(0, npar_sig):
        tot_fit.SetParameter(ipar + npar_bkg, sig_fit.GetParameter(ipar))
        sig_fit.GetParLimits(ipar, parmin, parmax)
        tot_fit.SetParLimits(ipar + npar_bkg, parmin, parmax)
    for ipar in range(0, npar_bkg):
        tot_fit.SetParameter(ipar, back_fit.GetParameter(ipar))
    if fixedmean:
        tot_fit.FixParameter(npar_bkg + 1, masspeak)
    if setfixgaussiansigma is True:
        tot_fit.FixParameter(npar_bkg + 2, sigma_sig)
    tot_fit.SetParNames(*par_names_bkg, *par_names_sig)
    histo.Fit("tot_fit", ("R,%s,+,0" % (fitOption)))

    logger.info("calculate signal, backgroud, S/B, significance")
    for ipar in range(0, npar_bkg):
        back_refit.SetParameter(ipar, tot_fit.GetParameter(ipar))
        back_refit.SetParError(ipar, tot_fit.GetParameter(ipar))
    for ipar in range(npar_bkg, (npar_bkg + npar_sig)):
        sig_fit.SetParameter(ipar - npar_bkg, tot_fit.GetParameter(ipar))
        sig_fit.SetParError(ipar - npar_bkg, tot_fit.GetParError(ipar))

    # Check the signal fit
    error = update_check_signal_fit(sig_fit)
    if error != "":
        logger.error("Signal fit probably bad for following reasons:\n%s", error)

    nsigma = 3.0
    fMass = sig_fit.GetParameter(1)
    # Could be negative together with the integral pre-factor
    fSigmaSgn = sig_fit.GetParameter(2)
    minMass_fit = fMass - nsigma * fSigmaSgn
    maxMass_fit = fMass + nsigma * fSigmaSgn
    leftBand = histo.FindBin(fMass - nSigma4SideBands * fSigmaSgn)
    rightBand = histo.FindBin(fMass + nSigma4SideBands * fSigmaSgn)
    intB = histo.Integral(1, leftBand) + histo.Integral(rightBand, histo.GetNbinsX())
    sum2 = 0.
    for i_left in range(1, leftBand + 1):
        sum2 += histo.GetBinError(i_left) * histo.GetBinError(i_left)
    for i_right in range(rightBand, (histo.GetNbinsX()) + 1):
        sum2 += histo.GetBinError(i_right) * histo.GetBinError(i_right)
    intBerr = math.sqrt(sum2)
    background = back_refit.Integral(minMass_fit, maxMass_fit) / Double(histo.GetBinWidth(1))
    #if background <= 0:
    #    return -1, -1
    errbackground = 0
    if intB > 0:
        # Rescale the error
        errbackground = intBerr / intB * background

    logger.info("Background: {background}, error background: %s", errbackground)
    rawYield = sig_fit.GetParameter(0) / Double(histo.GetBinWidth(1))
    rawYieldErr = sig_fit.GetParError(0) / Double(histo.GetBinWidth(1))

    logger.info("Raw yield: %f, raw yield error: %f", rawYield, rawYieldErr)
    sigOverback = rawYield / background
    errSigSq = rawYieldErr * rawYieldErr
    errBkgSq = errbackground * errbackground
    sigPlusBkg = background + rawYield
    significance = 0
    errsignificance = 0
    if sigPlusBkg > 0:
        significance = rawYield/(math.sqrt(sigPlusBkg))
        errsignificance = significance * (math.sqrt((errSigSq + errBkgSq) / \
                          (4. * sigPlusBkg * sigPlusBkg) +                  \
                          (background / sigPlusBkg) * errSigSq /            \
                          rawYield / rawYield))

    logger.info("Significance: %f, error significance: %f", significance, errsignificance)
    #Draw
    c1 = TCanvas('c1', 'The Fit Canvas', 700, 700)

    gStyle.SetOptStat(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    c1.cd()
    histo.GetXaxis().SetRangeUser(massmin, massmax)
    # Adjust y-range for good readability
    histo_min = histo.GetMinimum() * 0.9
    histo_max = histo.GetMaximum() + (histo.GetMaximum() - histo_min)
    histo.GetYaxis().SetRangeUser(histo_min, histo_max)
    histo.SetMarkerStyle(20)
    histo.SetMarkerSize(1)
    #histo.SetMinimum(0.)
    histo.Draw("PE")
    back_refit.Draw("same")
    tot_fit.Draw("same")
    if draw_side_band_fit:
        back_fit.SetLineColor(kGray + 1)
        back_fit.SetLineStyle(2)
        back_fit.Draw("same")
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
    pinfom.AddText("%s = %.3f #pm %.3f" % (sig_fit.GetParName(1),\
        sig_fit.GetParameter(1), sig_fit.GetParError(1)))
    pinfom.AddText("%s = %.3f #pm %.3f" % (sig_fit.GetParName(2),\
        sig_fit.GetParameter(2), sig_fit.GetParError(2)))
    pinfom.Draw()
    pinfos.AddText("S = %.0f #pm %.0f " % (rawYield, rawYieldErr))
    pinfos.AddText("B (%.0f#sigma) = %.0f #pm %.0f" % \
        (nsigma, background, errbackground))
    pinfos.AddText("S/B (%.0f#sigma) = %.4f " % (nsigma, sigOverback))
    pinfos.AddText("Signif (%.0f#sigma) = %.1f #pm %.1f " %\
        (nsigma, significance, errsignificance))
    pinfos.Draw()
    c1.Update()
    c1.SaveAs("%s/fittedplot%s.eps" % (outputfolder, suffix))
    return rawYield, rawYieldErr, sig_fit, back_fit
