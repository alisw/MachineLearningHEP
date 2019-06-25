
"""
Methods to: fit inv. mass
"""
import os
import sys
import math
from array import array
import yaml
from ROOT import TH1F, TF1, TFile
from ROOT import gROOT, gStyle
from ROOT import kBlue, kGray
from ROOT import TCanvas, TPaveText, Double

def gaus_func():
    sigval = "[0]/(sqrt(2.*pi))/[2]*(exp(-(x-[1])*(x-[1])/2./[2]/[2]))"
    return sigval

def plo2_func(bkg, massmax, massmin):
    plo2_func = ""
    if bkg == "Pol1":
        plo2_func = "[0]/(%s)+[1]*(x-0.5*(%s))"%\
            ((massmax-massmin), (massmax+massmin))
    if bkg == "Pol2":
        plo2_func = "[0]/(%s)+[1]*(x-0.5*(%s))+[2]*(x*x-1/3.*(%s)/(%s))"%\
            ((massmax-massmin), (massmax+massmin), (massmax*massmax*massmax-\
            massmin*massmin*massmin), (massmax-massmin))
    return plo2_func

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
    fixedmean, fixedsigma, outputfolder):
    print(bkgfunc)
    if "Lc" in case:
        print("add my fitter")
        histo.GetXaxis().SetTitle("Invariant Mass L_{c}^{+}(GeV/c^{2})")
        histo.Rebin(rebin)
        histo.SetStats(0)
        print("begin fit")
        if dolikelihood is True:
            fitOption = "L,E"
        if setinitialgaussianmean is True:
            mean = masspeak
        if setfixgaussiansigma is True:
            sigmaSgn = sigma
            fixedsigma == 1

        print("fit background (just side bands)")
        nSigma4SideBands = 4.
        integralHisto = histo.Integral(histo.FindBin(massmin), histo.FindBin(massmax), "width")
        back_fit = TF1("back_fit", plo2_func(bkgfunc, massmax, massmin), massmin, massmax)
        back_fit.SetParNames("BkgInt", "Coef1", "Coef2")
        back_fit.SetParameters(integralHisto, -10., 5)
        back_fit.SetLineColor(kBlue+3)
        minbin = histo.FindBin(massmin)
        maxbin = histo.FindBin(massmax)
        for xbin in range(minbin, maxbin+1):
            xvalue = histo.GetBinCenter(xbin)
            if ((abs(xvalue-masspeak) < (nSigma4SideBands*sigmaSgn))\
                & (xvalue >= massmin) & (xvalue <= massmax)) is True:
                back_fit.RejectPoint()
        histo.Fit("back_fit", ("R,%s,+,0" % (fitOption)))
        back_fit.SetLineColor(kGray+1)
        back_fit.SetLineStyle(2)

        print("refit (all range)")
        back_refit = TF1("back_refit", plo2_func(bkgfunc, massmax, massmin), massmin, massmax)
        back_refit.SetParNames("BkgInt", "Coef1", "Coef2")
        back_refit.SetParameters(integralHisto, -10., 5)
        back_fit.SetLineColor(kBlue+3)
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
        gaus_fit = TF1("gaus_fit", gaus_func(), massmin, massmax)
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
        par = array('d', 6*[0.])
        for ipar_back in range(0, 3):
            par[ipar_back] = par_back1[ipar_back]
        for ipar_gaus in range(3, 6):
            par[ipar_gaus] = par_gaus2[ipar_gaus-3]

        print("fit all (signal + background)")
        tot_fit = TF1("tot_fit", tot_func(bkgfunc, massmax, massmin), massmin, massmax)
        tot_fit.SetLineColor(4)
        tot_fit.SetParameters(par)
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
        #write fit file
        fout = TFile.Open("%s/afterfit.root" % (outputfolder), "recreate")
        fout.cd()
        c1.Write()
        c1.SaveAs("%s/afterfit.pdf" % (outputfolder))
