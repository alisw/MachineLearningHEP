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

from ROOT import TFile, TH1F, TCanvas, TF1 # pylint: disable=import-error,no-name-in-module, unused-import
from ROOT import TLine, gROOT, gStyle, TLegend # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.utilities_plot import rebin_histogram, buildbinning, buildhisto

# Configuration variables
FIT_RANGE = [40., 100.]  # Fit range
HISTO_RANGE = [0., 150.]  # Histogram plotting range
REBIN = True  # Rebin histograms
REBIN_BINNING = buildbinning(100, -.5, 99.5)
REBIN_BINNING += buildbinning(25, 100.5, 199.5)
SHOW_FUNC_RATIO = True  # Shows the ratio of the histogram to the fit function

gROOT.SetStyle("Plain")
gStyle.SetOptStat(0)
gStyle.SetOptStat(0000)
gStyle.SetPalette(0)
gStyle.SetCanvasColor(0)
gStyle.SetFrameFillColor(0)
gStyle.SetOptTitle(0)

# pylint: disable=line-too-long, invalid-name
filedatatrg = TFile.Open("/data/DerivedResults/D0kAnywithJets/vAN-20200304_ROOT6-1/pp_2018_data/376_20200304-2028/resultsSPDvspt_ntrkl_trigger/masshisto.root")
filedatamb = TFile.Open("/data/DerivedResults/D0kAnywithJets/vAN-20200304_ROOT6-1/pp_2018_data/376_20200304-2028/resultsMBvspt_ntrkl_trigger/masshisto.root")
hden = filedatamb.Get("hn_tracklets_corr")
hnum = filedatatrg.Get("hn_tracklets_corr")
if REBIN:
    hden_rebin = buildhisto(hden.GetName() + "_den_rebin",
                            hden.GetTitle(), REBIN_BINNING)
    hden = rebin_histogram(hden, hden_rebin)
    hnum_rebin = buildhisto(hnum.GetName() + "_num_rebin",
                            hnum.GetTitle(), REBIN_BINNING)
    hnum = rebin_histogram(hnum, hnum_rebin)
hratio = hnum.Clone("hratio")
hdend = filedatamb.Get("hn_tracklets_corr_withd")
hnumd = filedatatrg.Get("hn_tracklets_corr_withd")
if REBIN:
    hdend_rebin = buildhisto(hdend.GetName() + "_dend_rebin",
                             hdend.GetTitle(), REBIN_BINNING)
    hdend = rebin_histogram(hdend, hdend_rebin)
    hnumd_rebin = buildhisto(hnumd.GetName() + "_numd_rebin",
                             hnumd.GetTitle(), REBIN_BINNING)
    hnumd = rebin_histogram(hnumd, hnumd_rebin)
hratiod = hnumd.Clone("hratiod")
hratio.Divide(hden)
hratiod.Divide(hdend)

ctrigger = TCanvas('ctrigger', 'The Fit Canvas')
ctrigger.SetCanvasSize(2500, 2000)
ctrigger.Divide(3, 2)
ctrigger.cd(1)
leg = TLegend(.5, .65, .7, .85)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)

hden.GetYaxis().SetTitle("Entries")
hden.GetXaxis().SetTitle("n_tracklets_corr")
hden.GetXaxis().SetRangeUser(*HISTO_RANGE)
hden.Draw("pe")
hden.SetLineColor(2)
hnum.Draw("pesame")
leg.AddEntry(hden, "MB", "LEP")
leg.AddEntry(hnum, "SPD", "LEP")
leg.Draw()
#
ctrigger.cd(2)
hratio.GetYaxis().SetTitle("SPD/MB (no D required)")
hratio.GetXaxis().SetTitle("n_tracklets_corr")
hratio.GetXaxis().SetRangeUser(*HISTO_RANGE)
hratio.Draw("pe")
func = TF1("func", "([0]/(1+TMath::Exp(-[1]*(x-[2]))))", *HISTO_RANGE)
func.SetParameters(300, .1, 570)
func.SetParLimits(1, 0., 10.)
func.SetParLimits(2, 0., 1000.)
func.SetRange(*FIT_RANGE)
func.SetLineWidth(1)
hratio.Fit(func, "L", "", *FIT_RANGE)
func.Draw("same")
# Ratio to fit function
if SHOW_FUNC_RATIO:
    ctrigger.cd(3)
    hfunratio = hratio.DrawCopy()
    hfunratio.GetListOfFunctions().Clear()
    hfunratio.GetYaxis().SetTitle(hfunratio.GetYaxis().GetTitle()
                                  + " ratio to fit function")
    for i in range(1, hfunratio.GetNbinsX()+1):
        x = hfunratio.GetXaxis().GetBinCenter(i)
        y = [hfunratio.GetBinContent(i), hfunratio.GetBinError(i)]
        ratio = y[0]/func.Eval(x)
        ratio_error = y[1]/func.Eval(x)
        hfunratio.SetBinContent(i, ratio)
        hfunratio.SetBinError(i, ratio_error)
#
ctrigger.cd(4)
hnumd.GetYaxis().SetTitle("Entries")
hnumd.GetXaxis().SetTitle("n_tracklets_corr")
hnumd.GetXaxis().SetRangeUser(*HISTO_RANGE)
hdend.SetLineColor(2)
hnumd.Draw("pe")
hdend.Draw("pesame")
leg.Draw()
#
ctrigger.cd(5)
hratiod.GetYaxis().SetTitle("SPD/MB (D required)")
hratiod.GetXaxis().SetTitle("n_tracklets_corr")
hratiod.GetXaxis().SetRangeUser(*HISTO_RANGE)
hratiod.Draw("pe")
funcd = TF1("func", "([0]/(1+TMath::Exp(-[1]*(x-[2]))))", *HISTO_RANGE)
funcd.SetParameters(300, .1, 570)
funcd.SetParLimits(1, 0., 10.)
funcd.SetParLimits(2, 0., 1000.)
funcd.SetRange(*FIT_RANGE)
funcd.SetLineWidth(1)
hratiod.Fit(funcd, "L", "", *FIT_RANGE)
func.SetLineColor(1)
func.Draw("same")
funcd.SetLineColor(4)
funcd.Draw("same")
ctrigger.cd(6)
hempty = TH1F("hempty", "hempty", 200, 0., 100.)
hempty.Draw()
funcnorm = func.Clone("funcSPDvspt_ntrkl_norm")
funcnorm.FixParameter(0, funcnorm.GetParameter(0)/funcnorm.GetMaximum())
funcnormd = funcd.Clone("funcdSPDvspt_ntrkl_norm")
funcnormd.FixParameter(0, funcnormd.GetParameter(0)/funcnormd.GetMaximum())
hempty.GetXaxis().SetTitle("n_tracklets_corr")
hempty.GetYaxis().SetTitle("Efficiency")
funcnorm.Draw("same")
funcnormd.Draw("same")
line = TLine(60, 0, 60, 1)
line.SetLineStyle(2)
line.Draw("same")
ctrigger.SaveAs("SPDtrigger.pdf")
# pylint: disable=line-too-long
foutput = TFile.Open("../Analyses/ALICE_D2H_vs_mult_pp13/reweighting/data_2018/triggerSPDvspt_ntrkl.root", "recreate")
foutput.cd()
hratio.SetName("hratioSPDvspt_ntrkl")
hratio.Write()
func.SetName("funcSPDvspt_ntrkl")
func.Write()
funcnorm.Write()
hratiod.SetName("hratiodSPDvspt_ntrkl")
hratiod.Write()
funcd.SetName("funcdSPDvspt_ntrkl")
funcd.Write()
funcnormd.Write()
print("Press enter to continue")
input()
foutput.Close()
