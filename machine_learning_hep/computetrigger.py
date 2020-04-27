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

from ROOT import TFile, TH1F, TCanvas, TF1, gPad  # pylint: disable=import-error,no-name-in-module, unused-import
from ROOT import TLine, gROOT, gStyle, TLegend  # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.utilities_plot import load_root_style, rebin_histogram, buildbinning, buildhisto
import argparse
from sys import argv


def main(input_trg="/data/DerivedResults/D0kAnywithJets/vAN-20200304_ROOT6-1/pp_2018_data/376_20200304-2028/resultsSPDvspt_ntrkl_trigger/masshisto.root",
         input_mb="/data/DerivedResults/D0kAnywithJets/vAN-20200304_ROOT6-1/pp_2018_data/376_20200304-2028/resultsMBvspt_ntrkl_trigger/masshisto.root",
         #  output_path="../Analyses/ALICE_D2H_vs_mult_pp13/reweighting/data_2018/",
         output_path="/tmp/",
         min_draw_range=0,
         max_draw_range=150,
         min_fit_range=40.,
         max_fit_range=100.,
         rebin_histo=True,
         show_func_ratio=True):

    draw_range = [min_draw_range,
                  max_draw_range]
    fit_range = [min_fit_range,
                 max_fit_range]

    re_binning = buildbinning(100, -.5, 99.5)
    re_binning += buildbinning(25, 100.5, 199.5)

    load_root_style()  # Loading the default style

    # Get the input data and compute efficiency
    filedatatrg = TFile.Open(input_trg, "READ")
    filedatamb = TFile.Open(input_mb, "READ")
    hden = filedatamb.Get("hn_tracklets_corr")
    hnum = filedatatrg.Get("hn_tracklets_corr")
    if rebin_histo:
        hden_rebin = buildhisto(hden.GetName() + "_den_rebin",
                                hden.GetTitle(), re_binning)
        hden = rebin_histogram(hden, hden_rebin)
        hnum_rebin = buildhisto(hnum.GetName() + "_num_rebin",
                                hnum.GetTitle(), re_binning)
        hnum = rebin_histogram(hnum, hnum_rebin)
    hratio = hnum.Clone("hratio")
    hdend = filedatamb.Get("hn_tracklets_corr_withd")
    hnumd = filedatatrg.Get("hn_tracklets_corr_withd")
    if rebin_histo:
        hdend_rebin = buildhisto(hdend.GetName() + "_dend_rebin",
                                 hdend.GetTitle(), re_binning)
        hdend = rebin_histogram(hdend, hdend_rebin)
        hnumd_rebin = buildhisto(hnumd.GetName() + "_numd_rebin",
                                 hnumd.GetTitle(), re_binning)
        hnumd = rebin_histogram(hnumd, hnumd_rebin)
    hratiod = hnumd.Clone("hratiod")
    hratio.Divide(hden)
    hratiod.Divide(hdend)

    # Prepare the canvas
    ctrigger = TCanvas('ctrigger', 'The Fit Canvas')
    ctrigger.SetCanvasSize(2500, 2000)
    ctrigger.Divide(3, 2)
    leg = TLegend(.5, .65, .7, .85)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)

    # Draw source without D
    ctrigger.cd(1)
    hden.GetYaxis().SetTitle("Entries")
    hden.GetXaxis().SetTitle("n_tracklets_corr")
    hden.GetXaxis().SetRangeUser(*draw_range)
    hden.Draw("pe")
    hden.SetLineColor(2)
    hnum.Draw("pesame")
    leg.AddEntry(hden, "MB", "LEP")
    leg.AddEntry(hnum, "SPD", "LEP")
    leg.Draw()
    # Draw efficiency and fit it
    ctrigger.cd(2)
    hratio.GetYaxis().SetTitle("SPD/MB (no D required)")
    hratio.GetXaxis().SetTitle("n_tracklets_corr")
    hratio.GetXaxis().SetRangeUser(*draw_range)
    hratio.Draw("pe")
    func = TF1("func", "([0]/(1+TMath::Exp(-[1]*(x-[2]))))", *draw_range)
    func.SetParameters(300, .1, 570)
    func.SetParLimits(1, 0., 10.)
    func.SetParLimits(2, 0., 1000.)
    func.SetRange(*fit_range)
    func.SetLineWidth(1)
    hratio.Fit(func, "L", "", *fit_range)
    func.Draw("same")
    # Ratio to fit function
    if show_func_ratio:
        ctrigger.cd(3)
        hfunratio = hratio.DrawCopy()
        hfunratio.GetListOfFunctions().Clear()
        yaxis = hfunratio.GetYaxis()
        yaxis.SetTitle(yaxis.GetTitle()
                       + " ratio to fit function")
        for i in range(1, hfunratio.GetNbinsX()+1):
            x = hfunratio.GetXaxis().GetBinCenter(i)
            y = [hfunratio.GetBinContent(i),
                 hfunratio.GetBinError(i)]
            ratio = y[0]/func.Eval(x)
            ratio_error = y[1]/func.Eval(x)
            hfunratio.SetBinContent(i, ratio)
            hfunratio.SetBinError(i, ratio_error)
    # Draw source with D
    ctrigger.cd(4)
    hnumd.GetYaxis().SetTitle("Entries")
    hnumd.GetXaxis().SetTitle("n_tracklets_corr")
    hnumd.GetXaxis().SetRangeUser(*draw_range)
    hdend.SetLineColor(2)
    hnumd.Draw("pe")
    hdend.Draw("pesame")
    leg.Draw()
    # Draw efficiency and fit it
    ctrigger.cd(5)
    hratiod.GetYaxis().SetTitle("SPD/MB (D required)")
    hratiod.GetXaxis().SetTitle("n_tracklets_corr")
    hratiod.GetXaxis().SetRangeUser(*draw_range)
    hratiod.Draw("pe")
    funcd = TF1("func", "([0]/(1+TMath::Exp(-[1]*(x-[2]))))", *draw_range)
    funcd.SetParameters(300, .1, 570)
    funcd.SetParLimits(1, 0., 10.)
    funcd.SetParLimits(2, 0., 1000.)
    funcd.SetRange(*fit_range)
    funcd.SetLineWidth(1)
    hratiod.Fit(funcd, "L", "", *fit_range)
    func.SetLineColor(1)
    func.Draw("same")
    funcd.SetLineColor(4)
    funcd.Draw("same")
    # Draw both fitting functions
    ctrigger.cd(6)
    hframe = gPad.DrawFrame(min_draw_range, 0,
                            max_draw_range, 1,
                            ";n_tracklets_corr;Efficiency")
    funcnorm = func.Clone("funcSPDvspt_ntrkl_norm")
    funcnorm.FixParameter(0, funcnorm.GetParameter(0)/funcnorm.GetMaximum())
    funcnormd = funcd.Clone("funcdSPDvspt_ntrkl_norm")
    funcnormd.FixParameter(0, funcnormd.GetParameter(0)/funcnormd.GetMaximum())
    funcnorm.Draw("same")
    funcnormd.Draw("same")
    line = TLine(60, 0, 60, 1)
    line.SetLineStyle(2)
    line.Draw("same")
    ctrigger.SaveAs(output_path + "/SPDtrigger.pdf")
    # pylint: disable=line-too-long
    foutput = TFile.Open(output_path + "/triggerSPDvspt_ntrkl.root",
                         "recreate")
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
    if "-b" not in argv:
        print("Press enter to continue")
        input()
    foutput.Close()


if __name__ == "__main__":
    # Configuration variables
    PARSER = argparse.ArgumentParser(description="Compute the trigger")
    PARSER.add_argument("--input_trg",
                        help="input file for triggered data")
    PARSER.add_argument("--input_mb",
                        help="input file for MB data")
    PARSER.add_argument("--output_path",
                        help="output path for pdf and root files",
                        default="/tmp/")
    PARSER.add_argument("--min_draw_range",
                        help="Minimum histogram plotting range",
                        default=0.,
                        type=float)
    PARSER.add_argument("--max_draw_range",
                        help="Maximum histogram plotting range",
                        default=150.,
                        type=float)
    PARSER.add_argument("--min_fit_range",
                        help="Minimum fit range",
                        default=40.,
                        type=float)
    PARSER.add_argument("--max_fit_range",
                        help="Maximum fit range",
                        default=100.,
                        type=float)
    PARSER.add_argument("--rebin_histo",
                        help="Rebin the histogram",
                        default=True,
                        type=bool)
    PARSER.add_argument("--show_func_ratio",
                        help="Shows the ratio between the function and the fitted histogram",
                        default=True,
                        type=bool)

    PARSER.print_help()
    ARGS = PARSER.parse_args()
    print(ARGS)
    main(input_trg=ARGS.input_trg,
         input_mb=ARGS.input_mb,
         output_path=ARGS.output_path,
         min_draw_range=ARGS.min_draw_range,
         max_draw_range=ARGS.max_draw_range,
         min_fit_range=ARGS.min_fit_range,
         max_fit_range=ARGS.max_fit_range,
         rebin_histo=ARGS.rebin_histo,
         show_func_ratio=ARGS.show_func_ratio)
