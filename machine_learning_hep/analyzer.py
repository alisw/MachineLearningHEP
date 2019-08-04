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
import os
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend
from ROOT import gROOT
from ROOT import TStyle
from machine_learning_hep.globalfitter import fitter

# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements
class Analyzer:
    species = "analyzer"
    def __init__(self, datap, case):


        #namefiles pkl
        self.case = case
        self.v_var_binning = datap["var_binning"]
        self.lpt_finbinmin = datap["analysis"]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"]["sel_an_binmax"]
        self.bin_matching = datap["analysis"]["binning_matching"]
        self.p_nptbins = len(self.lpt_finbinmin)
        self.lpt_probcutfin = datap["analysis"]["probcutoptimal"]

        self.lvar2_binmin = datap["analysis"]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"]["var_binning2"]
        self.p_nbin2 = len(self.lvar2_binmin)

        self.d_resultsallpmc = datap["analysis"]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"]["data"]["resultsallp"]

        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, self.n_filemass)
        self.n_filecross = datap["files_names"]["crossfilename"]
        self.p_mass_fit_lim = datap["analysis"]['mass_fit_lim']

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)
        self.n_evtvalroot = datap["files_names"]["namefile_evtvalroot"]

        self.p_bin_width = datap["analysis"]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        #parameter fitter
        self.p_sgnfunc = datap["analysis"]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"]["bkgfunc"]
        self.p_masspeak = datap["analysis"]["masspeak"]
        self.p_massmin = datap["analysis"]["massmin"]
        self.p_massmax = datap["analysis"]["massmax"]
        self.p_rebin = datap["analysis"]["rebin"]
        self.p_includesecpeak = datap["analysis"]["includesecpeak"]
        self.p_masssecpeak = datap["analysis"]["masssecpeak"]
        self.p_fixedmean = datap["analysis"]["FixedMean"]
        self.p_fixingaussigma = datap["analysis"]["SetFixGaussianSigma"]
        self.p_fixingausmean = datap["analysis"]["SetInitialGaussianMean"]
        self.p_dolike = datap["analysis"]["dolikelihood"]
        self.p_sigmaarray = datap["analysis"]["sigmaarray"]
        self.p_fixedsigma = datap["analysis"]["FixedSigma"]
        self.p_casefit = datap["analysis"]["fitcase"]
        self.p_latexnmeson = datap["analysis"]["latexnamemeson"]
        self.p_latexbin2var = datap["analysis"]["latexbin2var"]
        self.p_dofullevtmerge = datap["dofullevtmerge"]
        self.p_dodoublecross = datap["analysis"]["dodoublecross"]
        self.ptranges = self.lpt_finbinmin.copy()
        self.ptranges.append(self.lpt_finbinmax[-1])
        self.var2ranges = self.lvar2_binmin.copy()
        self.var2ranges.append(self.lvar2_binmax[-1])
        print(self.var2ranges)
        self.lmult_yieldshisto = [TH1F("hyields%d" % (imult), "", \
            self.p_nptbins, array("d", self.ptranges)) for imult in range(self.p_nbin2)]

        self.p_nevents = datap["analysis"]["nevents"]
        self.p_sigmamb = datap["ml"]["opt"]["sigma_MB"]
        self.p_br = datap["ml"]["opt"]["BR"]

        self.d_valevtdata = datap["validation"]["data"]["dirmerged"]
        self.d_valevtmc = datap["validation"]["mc"]["dirmerged"]
        self.f_evtvaldata = os.path.join(self.d_valevtdata, self.n_evtvalroot)
        self.f_evtvalmc = os.path.join(self.d_valevtmc, self.n_evtvalroot)

    def fitter(self):
        lfile = TFile.Open(self.n_filemass)
        fileout = TFile.Open("yields%s.root" % self.case, "recreate")
        for imult in range(self.p_nbin2):
            for ipt in range(self.p_nptbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[imult], self.lvar2_binmax[imult])
                h_invmass = lfile.Get("hmass" + suffix)
                rawYield, rawYieldErr = \
                    fitter(h_invmass, self.p_casefit, self.p_sgnfunc[ipt], self.p_bkgfunc[ipt], \
                    self.p_masspeak, self.p_rebin[ipt], self.p_dolike, self.p_fixingausmean, \
                    self.p_fixingaussigma, self.p_sigmaarray[ipt], self.p_massmin[ipt], \
                    self.p_massmax[ipt], self.p_fixedmean, self.p_fixedsigma, \
                    self.d_resultsallpdata, suffix)
                rawYield = rawYield/(self.lpt_finbinmax[ipt] - self.lpt_finbinmin[ipt])
                rawYieldErr = rawYieldErr/(self.lpt_finbinmax[ipt] - self.lpt_finbinmin[ipt])
                self.lmult_yieldshisto[imult].SetBinContent(ipt + 1, rawYield)
                self.lmult_yieldshisto[imult].SetBinError(ipt + 1, rawYieldErr)
            fileout.cd()
            self.lmult_yieldshisto[imult].Write()

        cYields = TCanvas('cYields', 'The Fit Canvas')
        cYields.SetLogy()
        lfile = TFile.Open("yields%s.root" % self.case)
        for imult in range(self.p_nbin2):
            self.lmult_yieldshisto[imult].SetMinimum(1)
            self.lmult_yieldshisto[imult].SetMaximum(1e14)
            self.lmult_yieldshisto[imult].SetLineColor(imult+1)
            self.lmult_yieldshisto[imult].Draw("same")
        cYields.SaveAs("Yields%s.eps" % self.case)

    def efficiency(self):
        lfileeff = TFile.Open(self.n_fileff)
        fileouteff = TFile.Open("efficiencies%s.root" % self.case, "recreate")
        cEff = TCanvas('cEff', 'The Fit Canvas')
        for imult in range(self.p_nbin2):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                            self.lvar2_binmin[imult], \
                                            self.lvar2_binmax[imult])
            h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
            h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)

            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
            h_sel_pr.SetLineColor(imult+1)
            h_sel_pr.Draw("same")
            fileouteff.cd()
            h_sel_pr.SetName("eff_mult%d" % imult)
            h_sel_pr.Write()
        cEff.SaveAs("Eff%s.eps" % self.case)

    def plotter(self):

        gROOT.SetStyle("Plain")
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(0)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)
        gStyle.SetOptTitle(0)

        fileouteff = TFile.Open("efficiencies%s.root" % self.case)
        fileoutyield = TFile.Open("yields%s.root" % self.case)
        fileoutcross = TFile.Open("finalcross%s.root" % self.case, "recreate")

        cCrossvsvar1 = TCanvas('cCrossvsvar1', 'The Fit Canvas')
        cCrossvsvar1.SetCanvasSize(1900, 1500)
        cCrossvsvar1.SetWindowSize(500, 500)
        cCrossvsvar1.SetLogy()

        legvsvar1 = TLegend(.5, .65, .7, .85)
        legvsvar1.SetBorderSize(0)
        legvsvar1.SetFillColor(0)
        legvsvar1.SetFillStyle(0)
        legvsvar1.SetTextFont(42)
        legvsvar1.SetTextSize(0.035)

        listvalues = []
        listvalueserr = []

        for imult in range(self.p_nbin2):
            listvalpt = []
            heff = fileouteff.Get("eff_mult%d" % (imult))
            hcross = fileoutyield.Get("hyields%d" % (imult))
            hcross.Divide(heff)
            hcross.SetLineColor(imult+1)
            norm = 2 * self.p_br * self.p_nevents / (self.p_sigmamb * 1e12)
            hcross.Scale(1./norm)
            fileoutcross.cd()
            hcross.GetXaxis().SetTitle("p_{T} %s (GeV)" % self.p_latexnmeson)
            hcross.GetYaxis().SetTitle("d#sigma/dp_{T} (%s)" % self.p_latexnmeson)
            hcross.SetName("hcross%d" % imult)
            hcross.GetYaxis().SetRangeUser(1e4, 1e10)
            legvsvar1endstring = "%.1f < %s < %.1f GeV/c" % \
                    (self.lvar2_binmin[imult], self.p_latexbin2var, self.lvar2_binmax[imult])
            legvsvar1.AddEntry(hcross, legvsvar1endstring, "LEP")
            hcross.Draw("same")
            hcross.Write()
            listvalpt = [hcross.GetBinContent(ipt+1) for ipt in range(self.p_nptbins)]
            listvalues.append(listvalpt)
            listvalerrpt = [hcross.GetBinError(ipt+1) for ipt in range(self.p_nptbins)]
            listvalueserr.append(listvalerrpt)
        legvsvar1.Draw()
        cCrossvsvar1.SaveAs("Cross%sVs%s.eps" % (self.case, self.v_var_binning))

        cCrossvsvar2 = TCanvas('cCrossvsvar2', 'The Fit Canvas')
        cCrossvsvar2.SetCanvasSize(1900, 1500)
        cCrossvsvar2.SetWindowSize(500, 500)
        cCrossvsvar2.SetLogy()

        legvsvar2 = TLegend(.5, .65, .7, .85)
        legvsvar2.SetBorderSize(0)
        legvsvar2.SetFillColor(0)
        legvsvar2.SetFillStyle(0)
        legvsvar2.SetTextFont(42)
        legvsvar2.SetTextSize(0.035)
        hcrossvsvar2 = [TH1F("hcrossvsvar2" + "pt%d" % ipt, "", \
                        self.p_nbin2, array("d", self.var2ranges)) \
                        for ipt in range(self.p_nptbins)]

        for ipt in range(self.p_nptbins):
            print("pt", ipt)
            for imult in range(self.p_nbin2):
                hcrossvsvar2[ipt].SetLineColor(ipt+1)
                hcrossvsvar2[ipt].GetXaxis().SetTitle("%s" % self.p_latexbin2var)
                hcrossvsvar2[ipt].GetYaxis().SetTitle(self.p_latexnmeson)
                binmulrange = self.var2ranges[imult+1]-self.var2ranges[imult]
                if self.p_dodoublecross is True:
                    hcrossvsvar2[ipt].SetBinContent(imult+1, listvalues[imult][ipt]/binmulrange)
                    hcrossvsvar2[ipt].SetBinError(imult+1, listvalueserr[imult][ipt]/binmulrange)
                else:
                    hcrossvsvar2[ipt].SetBinContent(imult+1, listvalues[imult][ipt])
                    hcrossvsvar2[ipt].SetBinError(imult+1, listvalueserr[imult][ipt])

                hcrossvsvar2[ipt].GetYaxis().SetRangeUser(1e4, 1e10)
            legvsvar2endstring = "%.1f < %s < %.1f GeV/c" % \
                    (self.lpt_finbinmin[ipt], "p_{T}", self.lpt_finbinmax[ipt])
            hcrossvsvar2[ipt].Draw("same")
            legvsvar2.AddEntry(hcrossvsvar2[ipt], legvsvar2endstring, "LEP")
        legvsvar2.Draw()
        cCrossvsvar2.SaveAs("Cross%sVs%s.eps" % (self.case, self.v_var2_binning))

    def studyevents(self):
        gROOT.SetStyle("Plain")
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(0)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)
        gStyle.SetOptTitle(0)

        filedata = TFile.Open(self.f_evtvaldata)
        filemc = TFile.Open(self.f_evtvalmc)
        v0mn_trackletsdata = filedata.Get("v0mn_tracklets")
        v0mn_trackletsmc = filemc.Get("v0mn_tracklets")

        cscatter = TCanvas('cscatter', 'The Fit Canvas')
        cscatter.SetCanvasSize(1900, 1000)
        cscatter.Divide(2, 1)
        cscatter.cd(1)
        v0mn_trackletsdata.GetXaxis().SetTitle("offline V0 (data)")
        v0mn_trackletsdata.GetYaxis().SetTitle("offline SPD (data)")
        v0mn_trackletsdata.Draw("colz")
        cscatter.cd(2)
        v0mn_trackletsmc.GetXaxis().SetTitle("offline V0 (mc)")
        v0mn_trackletsmc.GetYaxis().SetTitle("offline SPD (mc)")
        v0mn_trackletsmc.Draw("colz")
        cscatter.SaveAs("cscatter.pdf")

        labelsv0 = ["kINT7_vsv0m", "HighMultSPD_vsv0m", "HighMultV0_vsv0m"]
        labelsspd = ["kINT7_vsntracklets", "HighMultSPD_vsntracklets", "HighMultV0_vsntracklets"]
        cutonspd = [20, 30, 40, 50, 60]

        ctrigger = TCanvas('ctrigger', 'The Fit Canvas')
        ctrigger.SetCanvasSize(2100, 1000)
        ctrigger.Divide(2, 1)
        ctrigger.cd(1)
        leg = TLegend(.5, .65, .7, .85)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)
        for i, lab in enumerate(labelsv0):
            heff = filedata.Get("hnum%s" % lab)
            hden = filedata.Get("hden%s" % lab)
            heff.SetLineColor(i+1)
            heff.Divide(heff, hden, 1.0, 1.0, "B")
            heff.SetMaximum(2.)
            heff.GetXaxis().SetTitle("offline V0M")
            heff.GetYaxis().SetTitle("trigger efficiency")
            heff.Draw("epsame")
            leg.AddEntry(heff, labelsv0[i], "LEP")
        leg.Draw()

        ctrigger.cd(2)
        lega = TLegend(.5, .65, .7, .85)
        lega.SetBorderSize(0)
        lega.SetFillColor(0)
        lega.SetFillStyle(0)
        lega.SetTextFont(42)
        lega.SetTextSize(0.035)
        for i, lab in enumerate(labelsspd):
            heff = filedata.Get("hnum%s" % lab)
            hden = filedata.Get("hden%s" % lab)
            heff.SetLineColor(i+1)
            heff.Divide(heff, hden, 1.0, 1.0, "B")
            heff.GetXaxis().SetTitle("offline SPD mul")
            heff.GetYaxis().SetTitle("trigger efficiency")
            heff.SetMaximum(2.)
            heff.Draw("epsame")
            lega.AddEntry(heff, labelsspd[i], "LEP")
        lega.Draw()
        ctrigger.SaveAs("ctrigger.pdf")

        ccutstudy = TCanvas('ccutstudy', 'The Fit Canvas')
        ccutstudy.SetCanvasSize(2200, 1000)
        ccutstudy.Divide(2, 1)
        ccutstudy.cd(1)
        legc = TLegend(.5, .65, .7, .85)
        legc.SetBorderSize(0)
        legc.SetFillColor(0)
        legc.SetFillStyle(0)
        legc.SetTextFont(42)
        legc.SetTextSize(0.035)
        for i, lab in enumerate(cutonspd):
            hdenv0mdata = filedata.Get("hdenv0m")
            heffdata = filedata.Get("hnumv0mspd%d" % lab)
            heffdata.SetLineColor(i+1)
            heffdata.Divide(heffdata, hdenv0mdata, 1.0, 1.0, "B")
            heffdata.SetMaximum(2.)
            heffdata.GetXaxis().SetTitle("offline V0M (data)")
            heffdata.GetYaxis().SetTitle("pseudo efficiency")
            heffdata.Draw("epsame")
            legc.AddEntry(heffdata, "SPD mult >=%d" % lab, "LEP")
            legc.Draw()
        ccutstudy.cd(2)
        for i, lab in enumerate(cutonspd):
            hdenv0mmc = filemc.Get("hdenv0m")
            heffmc = filemc.Get("hnumv0mspd%d" % lab)
            heffmc.SetLineColor(i+1)
            heffmc.Divide(heffmc, hdenv0mmc, 1.0, 1.0, "B")
            heffmc.SetMaximum(2.)
            heffmc.GetXaxis().SetTitle("offline V0M (mc)")
            heffmc.GetYaxis().SetTitle("pseudo efficiency")
            heffmc.Draw("epsame")
            legc.Draw()
        ccutstudy.SaveAs("ccutstudy.pdf")
