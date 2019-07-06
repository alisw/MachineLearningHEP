
"""
Methods to: fit inv. mass
Authors: xiuxiu.jiang@cern.ch
"""

import math
from array import array
from ROOT import TF1, TFile
from ROOT import gStyle
from ROOT import kBlue, kGray
from ROOT import TCanvas, TPaveText, Double
from ROOT import TVirtualFitter

def gaus_func(sgnfunc):
    if sgnfunc == "kGaus":
        sigval = "[0]/(sqrt(2.*pi))/[2]*(exp(-(x-[1])*(x-[1])/2./[2]/[2]))"
    return sigval

def pol1_func(x, par):
    plo1_func = par[0]/par[2]+par[1]*(x[0]-0.5*par[3])
    return plo1_func

def pol2_func(x, par):
    plo2_func = par[0]/par[3]+par[1]*(x[0]-0.5*par[4])+par[2]*\
        (x[0]*x[0]-1/3.*par[5]/par[3])
    return plo2_func

def pol1_func_sidebands(x, par):
    if abs(x[0]-par[4]) < par[5] is True:
        TF1.RejectPoint()
    plo1_func = par[0]/par[2]+par[1]*(x[0]-0.5*par[3])
    return plo1_func

def pol2_func_sidebands(x, par):
    if abs(x[0]-par[6]) < par[7] is True:
        TF1.RejectPoint()
    plo2_func = par[0]/par[3]+par[1]*(x[0]-0.5*par[4])+par[2]*\
        (x[0]*x[0]-1/3.*par[5]/par[3])
    return plo2_func

def fixpar(massmin, massmax, masspeak, range_signal):
    par_fix1 = Double(massmax-massmin)
    par_fix2 = Double(massmax+massmin)
    par_fix3 = Double(massmax*massmax*massmax-massmin*massmin*massmin)
    par_fix4 = Double(masspeak)
    par_fix5 = Double(range_signal)
    return par_fix1, par_fix2, par_fix3, par_fix4, par_fix5

def func_fit(fitsidebands, bkgfunc, massmin, massmax, integralHisto, masspeak, range_signal):
    par_fix1, par_fix2, par_fix3, par_fix4, par_fix5 =\
        fixpar(massmin, massmax, masspeak, range_signal)
    if fitsidebands:
        if bkgfunc == "Pol1":
            back_fit_new = TF1("back_fit_new", pol1_func_sidebands, massmin, massmax, 6)
            back_fit_new.SetParNames("BkgInt", "Slope", "", "", "", "")
            back_fit_new.SetParameters(integralHisto, -100.)
            back_fit_new.FixParameter(2, par_fix1)
            back_fit_new.FixParameter(3, par_fix2)
            back_fit_new.FixParameter(4, par_fix4)
            back_fit_new.FixParameter(5, par_fix5)
        if bkgfunc == "Pol2":
            back_fit_new = TF1("back_fit_new", pol2_func_sidebands, massmin, massmax, 8)
            back_fit_new.SetParNames("BkgInt", "Coef1", "Coef2", "", "", "", "", "")
            back_fit_new.SetParameters(integralHisto, -10., 5)
            back_fit_new.FixParameter(3, par_fix1)
            back_fit_new.FixParameter(4, par_fix2)
            back_fit_new.FixParameter(5, par_fix3)
            back_fit_new.FixParameter(6, par_fix4)
            back_fit_new.FixParameter(7, par_fix5)
        return back_fit_new
    else:
        if bkgfunc == "Pol1":
            back_refit_new = TF1("back_refit", pol1_func, massmin, massmax, 4)
            back_refit_new.SetParNames("BkgInt", "Slope", "", "")
            back_refit_new.SetParameters(integralHisto, -100.)
            back_refit_new.FixParameter(2, par_fix1)
            back_refit_new.FixParameter(3, par_fix2)
        if bkgfunc == "Pol2":
            back_refit_new = TF1("back_refit", pol2_func, massmin, massmax, 6)
            back_refit_new.SetParNames("BkgInt", "Coef1", "Coef2", "", "", "")
            back_refit_new.SetParameters(integralHisto, -10., 5)
            back_refit_new.FixParameter(3, par_fix1)
            back_refit_new.FixParameter(4, par_fix2)
            back_refit_new.FixParameter(5, par_fix3)
        return back_refit_new

def tot_func(bkg, massmax, massmin):
    print("tot_function = plo2 function + gaus function")
    if bkg == "Pol1":
        tot_func = "[0]/(%s)+[1]*(x-0.5*(%s))\
            + [3]/(sqrt(2.*pi))/[5]*(exp(-(x-[4])*(x-[4])/2./[5]/[5]))"%\
            ((massmax-massmin), (massmax+massmin))
    if bkg == "Pol2":
        tot_func = "[0]/(%s)+[1]*(x-0.5*(%s))+[2]*(x*x-1/3.*(%s)/(%s))\
            + [3]/(sqrt(2.*pi))/[5]*(exp(-(x-[4])*(x-[4])/2./[5]/[5]))"%\
            ((massmax-massmin), (massmax+massmin), (massmax*massmax*massmax\
            -massmin*massmin*massmin), (massmax-massmin))
    return tot_func


def fitter(histo, case, sgnfunc, bkgfunc, masspeak, rebin, dolikelihood,\
    setinitialgaussianmean, setfixgaussiansigma, sigma, massmin, massmax,\
    fixedmean, fixedsigma, outputfolder, varbin, minvar, maxvar):
    print(bkgfunc)
    if "Lc" in case:
        print("add my fitter")
        histo.GetXaxis().SetTitle("Invariant Mass L_{c}^{+}(GeV/c^{2})")
        histo.Rebin(rebin)
        histo.SetStats(0)
        TVirtualFitter.SetDefaultFitter("Minuit")
        print("begin fit")
        if dolikelihood is True:
            fitOption = "L,E"
        if setinitialgaussianmean is True:
            mean = masspeak
        if setfixgaussiansigma is True:
            sigmaSgn = sigma
            fixedsigma == 1

        print("fit background (just side bands)")
        fitsidebands = True
        nSigma4SideBands = 4.
        range_signal = nSigma4SideBands*sigmaSgn
        integralHisto = Double(histo.Integral(histo.FindBin(massmin), histo.FindBin(massmax), "width"))
        back_fit = func_fit(fitsidebands, bkgfunc, massmin, massmax,\
            integralHisto, masspeak, range_signal)
        back_fit.SetLineColor(kBlue+3)
        histo.Fit("back_fit", ("R,%s,+,0" % (fitOption)))
        back_fit.SetLineColor(kGray+1)
        back_fit.SetLineStyle(2)

        print("refit (all range)")
        fitsidebands = False
        back_refit = func_fit(fitsidebands, bkgfunc, massmin, massmax,\
            integralHisto, masspeak, range_signal)
        back_refit.SetLineColor(kBlue+3)
        histo.Fit("back_refit", ("R,%s,+,0" % (fitOption)))
        back_refit.SetLineColor(2)

        print(" fit signal")
        minForSig = mean-4.*sigmaSgn
        maxForSig = mean+4.*sigmaSgn
        binForMinSig = histo.FindBin(minForSig)
        binForMaxSig = histo.FindBin(maxForSig)
        sum_tot = 0.
        sumback = 0.
        for ibin in range(binForMinSig, binForMaxSig+1):
            sum_tot += histo.GetBinContent(ibin)
            sumback += back_fit.Eval(histo.GetBinCenter(ibin))
        integsig = Double((sum_tot-sumback)*(histo.GetBinWidth(1)))
        gaus_fit = TF1("gaus_fit", gaus_func(sgnfunc), massmin, massmax)
        gaus_fit.SetLineColor(5)
        gaus_fit.SetParameter(0, integsig)
        gaus_fit.SetParameter(1, mean)
        if fixedmean == 1:
            gaus_fit.FixParameter(1, mean)
        gaus_fit.SetParameter(2, sigmaSgn)
        if fixedsigma == 1:
            gaus_fit.FixParameter(2, sigmaSgn)
        gaus_fit.SetParNames("SgnInt", "Mean", "Sigma")

        print("get and set fit Parameters")
        par_back1 = back_fit.GetParameters()
        par_gaus2 = gaus_fit.GetParameters()
        if bkgfunc == "Pol1":
            par = array('d', 5*[0.])
            for ipar_back in range(0, 2):
                par[ipar_back] = par_back1[ipar_back]
            for ipar_gaus in range(2, 5):
                par[ipar_gaus] = par_gaus2[ipar_gaus-2]
        if bkgfunc == "Pol2":
            par = array('d', 6*[0.])
            for ipar_back in range(0, 3):
                par[ipar_back] = par_back1[ipar_back]
            for ipar_gaus in range(3, 6):
                par[ipar_gaus] = par_gaus2[ipar_gaus-3]


        print("fit all (signal + background)")
        tot_fit = TF1("tot_fit", tot_func(bkgfunc, massmax, massmin), massmin, massmax)
        tot_fit.SetLineColor(4)
        tot_fit.SetParameters(par)
        if bkgfunc == "Pol1":
            nParsBkg = 2
            for ipar_tot in range(0, 3):
                parmin = Double()
                parmax = Double()
                gaus_fit.GetParLimits(ipar_tot, parmin, parmax)
                tot_fit.SetParLimits(ipar_tot+nParsBkg, parmin, parmax)
            if fixedsigma == 1:
                tot_fit.FixParameter(4, sigmaSgn)
            if fixedmean == 1:
                tot_fit.FixParameter(3, mean)
            tot_fit.SetParNames("BkgInt", "Coef1", "SgnInt", "Mean", "Sigma")
        if bkgfunc == "Pol2":
            nParsBkg = 3
            for ipar_tot in range(0, 3):
                parmin = Double()
                parmax = Double()
                gaus_fit.GetParLimits(ipar_tot, parmin, parmax)
                tot_fit.SetParLimits(ipar_tot+nParsBkg, parmin, parmax)
            if fixedsigma == 1:
                tot_fit.FixParameter(5, sigmaSgn)
            if fixedmean == 1:
                tot_fit.FixParameter(4, mean)
            tot_fit.SetParNames("BkgInt", "Coef1", "Coef2", "SgnInt", "Mean", "Sigma")
        histo.Fit("tot_fit", ("R,%s,+,0" % (fitOption)))

        print("calculate signal, backgroud, S/B, significance")
        if bkgfunc == "Pol1":
            for ipar in range(0, 2):
                back_refit.SetParameter(ipar, tot_fit.GetParameter(ipar))
            for iparS in range(2, 5):
                gaus_fit.SetParameter(iparS-2, tot_fit.GetParameter(iparS))
        if bkgfunc == "Pol2":
            for ipar in range(0, 3):
                back_refit.SetParameter(ipar, tot_fit.GetParameter(ipar))
            for iparS in range(3, 6):
                gaus_fit.SetParameter(iparS-3, tot_fit.GetParameter(iparS))
        nsigma = 3.0
        fMass = gaus_fit.GetParameter(1)
        fSigmaSgn = gaus_fit.GetParameter(2)
        minMass_fit = fMass - nsigma*fSigmaSgn
        maxMass_fit = fMass + nsigma*fSigmaSgn
        intB = back_refit.GetParameter(0)
        intBerr = back_refit.GetParError(0)
        leftBand = histo.FindBin(fMass-nSigma4SideBands*fSigmaSgn)
        rightBand = histo.FindBin(fMass+nSigma4SideBands*fSigmaSgn)
        intB = histo.Integral(1, leftBand)+histo.Integral(rightBand, histo.GetNbinsX())
        sum2 = 0.
        for i_left in range(1, leftBand+1):
            sum2 += histo.GetBinError(i_left)*histo.GetBinError(i_left)
        for i_right in range(rightBand, (histo.GetNbinsX())+1):
            sum2 += histo.GetBinError(i_right)*histo.GetBinError(i_right)
        intBerr = math.sqrt(sum2)
        background = back_refit.Integral(minMass_fit, maxMass_fit)/Double(histo.GetBinWidth(1))
        errbackground = intBerr/intB*background
        print(background, errbackground)
        rawYield = tot_fit.GetParameter(nParsBkg)/Double(histo.GetBinWidth(1))
        rawYieldErr = tot_fit.GetParError(nParsBkg)/Double(histo.GetBinWidth(1))
        print(rawYield, rawYieldErr)
        sigOverback = rawYield/background
        errSigSq = rawYieldErr*rawYieldErr
        errBkgSq = errbackground*errbackground
        sigPlusBkg = background+rawYield
        significance = rawYield/(math.sqrt(sigPlusBkg))
        errsignificance = significance*(math.sqrt((errSigSq+errBkgSq)/(4.*sigPlusBkg*sigPlusBkg)\
            +(background/sigPlusBkg)*errSigSq/rawYield/rawYield))
        print(significance, errsignificance)
        #Draw
        c1 = TCanvas('c1', 'The Fit Canvas')
        gStyle.SetOptStat(0)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)
        c1.cd()
        histo.GetXaxis().SetRangeUser(massmin, massmax)
        histo.SetMarkerStyle(20)
        histo.SetMinimum(0.)
        histo.Draw("PE")
        back_refit.Draw("same")
        tot_fit.Draw("same")
        #write info.
        pinfos = TPaveText(0.12, 0.65, 0.47, 0.89, "NDC")
        pinfom = TPaveText(0.6, 0.7, 1., .87, "NDC")
        pinfos.SetBorderSize(0)
        pinfos.SetFillStyle(0)
        pinfom.SetBorderSize(0)
        pinfom.SetFillStyle(0)
        pinfom.SetTextColor(kBlue)
        if bkgfunc == "Pol1":
            pinfom.AddText("%s = %.3f #pm %.3f" % (tot_fit.GetParName(3),\
                tot_fit.GetParameter(3), tot_fit.GetParError(3)))
            pinfom.AddText("%s = %.3f #pm %.3f" % (tot_fit.GetParName(4),\
                tot_fit.GetParameter(4), tot_fit.GetParError(4)))
        if bkgfunc == "Pol2":
            pinfom.AddText("%s = %.3f #pm %.3f" % (tot_fit.GetParName(4),\
                tot_fit.GetParameter(4), tot_fit.GetParError(4)))
            pinfom.AddText("%s = %.3f #pm %.3f" % (tot_fit.GetParName(5),\
                tot_fit.GetParameter(5), tot_fit.GetParError(5)))
        pinfom.Draw()
        pinfos.AddText("S = %.0f #pm %.0f " % (rawYield, rawYieldErr))
        pinfos.AddText("B (%.0f#sigma) = %.0f #pm %.0f" % \
            (nsigma, background, errbackground))
        pinfos.AddText("S/B (%.0f#sigma) = %.4f " % (nsigma, sigOverback))
        pinfos.AddText("Signif (%.0f#sigma) = %.1f #pm %.1f " %\
            (nsigma, significance, errsignificance))
        pinfos.Draw()
        c1.Update()
        c1.SaveAs("%s/fittedplot%s_%d_%d.pdf" % (outputfolder, varbin, minvar, maxvar))
