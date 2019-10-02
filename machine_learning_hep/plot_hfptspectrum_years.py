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
# pylint: disable=unused-wildcard-import, wildcard-import
from array import *
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend, TLatex
from ROOT import gROOT, kRed, kGreen, kBlack, kBlue
from ROOT import TStyle, gPad, TPad, TLine

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches, too-many-locals
def plot_hfspectrum_years_ratios(case_1, case_2, ana_type):

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)
    gStyle.SetTitleOffset(1.15, "y")
    gStyle.SetTitleFont(42, "xy")
    gStyle.SetLabelFont(42, "xy")
    gStyle.SetTitleSize(0.02, "xy")
    gStyle.SetLabelSize(0.02, "xy")
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    with open("data/database_ml_parameters_%s.yml" % case_1, 'r') as param_config:
        data_param_1 = yaml.load(param_config, Loader=yaml.FullLoader)

    with open("data/database_ml_parameters_%s.yml" % case_2, 'r') as param_config:
        data_param_2 = yaml.load(param_config, Loader=yaml.FullLoader)

    use_period = data_param_1[case_1]["analysis"][ana_type]["useperiod"]
    latexbin2var = data_param_1[case_1]["analysis"][ana_type]["latexbin2var"]
    result_paths_1 = [data_param_1[case_1]["analysis"][ana_type]["data"]["results"][i] \
            for i in range(len(use_period)) if use_period[i]]
    result_paths_1.insert(0, data_param_1[case_1]["analysis"][ana_type]["data"]["resultsallp"])

    result_paths_2 = [data_param_2[case_2]["analysis"][ana_type]["data"]["results"][i] \
            for i in range(len(use_period)) if use_period[i]]
    result_paths_2.insert(0, data_param_2[case_2]["analysis"][ana_type]["data"]["resultsallp"])

    # Assume same in all particle cases
    periods = [data_param_1[case_1]["multi"]["data"]["period"][i] \
            for i in range(len(use_period)) if use_period[i]]
    periods.insert(0, "merged")

    binsmin = data_param_1[case_1]["analysis"][ana_type]["sel_binmin2"]
    binsmax = data_param_1[case_1]["analysis"][ana_type]["sel_binmax2"]

    name_1 = data_param_1[case_1]["analysis"][ana_type]["latexnamemeson"]
    name_2 = data_param_2[case_2]["analysis"][ana_type]["latexnamemeson"]

    br_1 = data_param_1[case_1]["ml"]["opt"]["BR"]
    br_2 = data_param_2[case_2]["ml"]["opt"]["BR"]
    sigmav0_1 = data_param_1[case_1]["analysis"]["sigmav0"]
    sigmav0_2 = data_param_2[case_2]["analysis"]["sigmav0"]

    files_tot_cross_1 = [TFile.Open("%s/finalcross%s%smulttot.root" % \
                                    (folder, case_1, ana_type)) for folder in result_paths_1]
    files_tot_cross_2 = [TFile.Open("%s/finalcross%s%smulttot.root" % \
                                    (folder, case_2, ana_type)) for folder in result_paths_2]
    files_mult_1 = []
    files_mult_2 = []
    for i in [0, 1, 2, 3]:
        files_mult_1.append([TFile.Open("%s/finalcross%s%smult%d.root" \
                % (folder, case_1, ana_type, i)) for folder in result_paths_1])
        files_mult_2.append([TFile.Open("%s/finalcross%s%smult%d.root" \
                % (folder, case_2, ana_type, i)) for folder in result_paths_2])

    linestyles = [1, 1, 1, 1]
    markerstyles = [2, 4, 5, 32]
    colors = [kBlack, kRed, kGreen+2, kBlue]

    for imult in [0, 1, 2, 3]:

        legyield = TLegend(.25, .65, .65, .85)
        legyield.SetTextFont(42)
        legyield.SetTextSize(0.03)
        legyield.SetBorderSize(0)
        legyield.SetFillColor(0)
        legyield.SetFillStyle(0)
        legyield.SetTextFont(42)
        counter = 0
        histos = []
        min_y = 9999.
        max_y = 0.
        for period, root_file_1, root_file_2 in zip(periods, files_tot_cross_1, files_tot_cross_2):
            hyield_1 = root_file_1.Get("histoSigmaCorr%d" % (imult))
            hyield_2 = root_file_2.Get("histoSigmaCorr%d" % (imult))
            hyield_1.Scale(1./(br_1 * sigmav0_1 * 1e12))
            hyield_2.Scale(1./(br_2 * sigmav0_2 * 1e12))
            hyield_ratio = hyield_1.Clone(f"{case_1}_{case_2}_ratio_{period}_{imult}")
            hyield_ratio.Divide(hyield_2)
            histos.append(hyield_ratio)

            max_y = max(hyield_ratio.GetMaximum(), max_y)
            min_y = min(hyield_ratio.GetMinimum(), min_y)

            hyield_ratio.SetLineColor(colors[counter % len(colors)])
            hyield_ratio.SetLineStyle(linestyles[counter % len(linestyles)])
            hyield_ratio.SetMarkerStyle(markerstyles[counter % len(markerstyles)])
            hyield_ratio.SetMarkerColor(colors[counter % len(colors)])
            legyieldstring = "%.1f #leq %s < %.1f (%s), %s" % \
                        (binsmin[imult], latexbin2var, binsmax[imult], ana_type, period)
            legyield.AddEntry(hyield_ratio, legyieldstring, "LEP")
            counter += 1

        # Now, do year over merged (where merged is always the last in the lists)
        double_ratios = []
        min_y_r = 9999.
        max_y_r = 0.
        for p in range(1, len(periods)):
            h_double_r = histos[p].Clone(f"{histos[p].GetName()}_ratio")
            h_double_r.Divide(h_double_r, histos[0], 1, 1, "B")
            #h_double_r.Divide(histos[0])
            double_ratios.append(h_double_r)

            max_y_r = max(h_double_r.GetMaximum(), max_y_r)
            min_y_r = min(h_double_r.GetMinimum(), min_y_r)

            h_double_r.SetLineColor(colors[p % len(colors)])
            h_double_r.SetLineStyle(linestyles[p % len(linestyles)])
            h_double_r.SetMarkerStyle(markerstyles[p % len(markerstyles)])
            h_double_r.SetMarkerColor(colors[p % len(colors)])
            h_double_r.SetLineWidth(1)

        #Corrected yield plot
        ccross = TCanvas('cCross', 'The Fit Canvas')
        ccross.SetCanvasSize(1500, 1500)
        ccross.SetWindowSize(500, 500)
        ccross.cd()
        pad_up = TPad("pad_up", "", 0., 0.4, 1., 1.)
        pad_up.SetBottomMargin(0.)
        pad_up.Draw()
        #ccross.SetLogx()
        pad_up.cd()
        pad_up.DrawFrame(0, 0.5 * min_y, 30, 2 * max_y,
                         ";; yield ratio %s / %s" % (name_1, name_2))
        for h in histos:
            h.Draw("same")
        legyield.Draw()

        ccross.cd()
        pad_double = TPad("pad_double", "", 0., 0.05, 1., 0.4)
        pad_double.SetTopMargin(0.)
        pad_double.SetBottomMargin(0.3)
        pad_double.Draw()
        pad_double.cd()
        frame_double = pad_double.DrawFrame(0, 0.5 * min_y_r, 30, 2 * max_y_r,
                                            ";#it{p}_{T} (GeV/#it{c}); year/merged")
        frame_double.SetTitleFont(42, "xy")
        frame_double.SetTitleSize(0.04, "xy")
        frame_double.SetLabelSize(0.04, "xy")
        for h in double_ratios:
            h.Draw("same")

        line_unity = TLine(frame_double.GetXaxis().GetXmin(), 1.,
                           frame_double.GetXaxis().GetXmax(), 1.)
        line_unity.SetLineColor(histos[0].GetLineColor())
        line_unity.SetLineStyle(histos[0].GetLineStyle())
        line_unity.Draw()

        ccross.SaveAs("ComparisonCorrYields_%s_%s_%s_combined%s_%d.eps" % \
                  (case_1, case_2, ana_type, "_".join(periods), imult))
        ccross.Close()

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches, too-many-locals
def plot_hfspectrum_years(case, ana_type):

    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetOptStat(0000)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetOptTitle(0)
    gStyle.SetTitleOffset(1.15, "y")
    gStyle.SetTitleFont(42, "xy")
    gStyle.SetLabelFont(42, "xy")
    gStyle.SetTitleSize(0.02, "xy")
    gStyle.SetLabelSize(0.02, "xy")
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)

    use_period = data_param[case]["analysis"][ana_type]["useperiod"]
    result_paths = [data_param[case]["analysis"][ana_type]["data"]["results"][i] \
            for i in range(len(use_period)) if use_period[i]]
    periods = [data_param[case]["multi"]["data"]["period"][i] \
            for i in range(len(use_period)) if use_period[i]]

    result_paths.insert(0, data_param[case]["analysis"][ana_type]["data"]["resultsallp"])
    periods.insert(0, "merged")

    binsmin = data_param[case]["analysis"][ana_type]["sel_binmin2"]
    binsmax = data_param[case]["analysis"][ana_type]["sel_binmax2"]
    name = data_param[case]["analysis"][ana_type]["latexnamemeson"]
    latexbin2var = data_param[case]["analysis"][ana_type]["latexbin2var"]
    br = data_param[case]["ml"]["opt"]["BR"]
    sigmav0 = data_param[case]["analysis"]["sigmav0"]

    files_tot_cross = [TFile.Open("%s/finalcross%s%smulttot.root" \
            % (folder, case, ana_type)) for folder in result_paths]
    files_mult = []
    for i in [0, 1, 2, 3]:
        files_mult.append([TFile.Open("%s/finalcross%s%smult%d.root" \
                % (folder, case, ana_type, i)) for folder in result_paths])

    linestyles = [1, 1, 1, 1]
    markerstyles = [2, 4, 5, 32]
    colors = [kBlack, kRed, kGreen+2, kBlue]

    print("################")
    print(f"case {case} in analysis {ana_type}")

    for imult in [0, 1, 2, 3]:
        #Corrected yield plot

        legyield = TLegend(.25, .65, .65, .85)
        legyield.SetBorderSize(0)
        legyield.SetFillColor(0)
        legyield.SetFillStyle(0)
        legyield.SetTextFont(42)
        legyield.SetTextSize(0.02)

        #Corrected eff plot
        ceff = TCanvas('cEff', 'The Fit Canvas')
        ceff.SetCanvasSize(1500, 1500)
        ceff.SetWindowSize(500, 500)

        leg_eff = TLegend(.25, .65, .65, .85)
        leg_eff.SetBorderSize(0)
        leg_eff.SetFillColor(0)
        leg_eff.SetFillStyle(0)
        leg_eff.SetTextFont(42)
        leg_eff.SetTextSize(0.02)

        counter = 0
        histos_cross = []
        histos_eff = []
        max_y = 0.
        min_y = 9999.
        max_y_eff = 0.
        min_y_eff = 9999.
        for period, root_file in zip(periods, files_tot_cross):

            h_eff = files_mult[imult][counter].Get("hDirectEffpt")
            h_eff.SetLineStyle(linestyles[counter % len(linestyles)])
            h_eff.SetLineColor(colors[counter % len(colors)])
            h_eff.SetMarkerStyle(markerstyles[counter % len(markerstyles)])
            h_eff.SetMarkerColor(colors[counter % len(colors)])
            h_eff.GetXaxis().SetTitleSize(0.02)
            h_eff.GetXaxis().SetTitleSize(0.02)
            h_eff.GetYaxis().SetTitleSize(0.02)
            histos_eff.append(h_eff)
            max_y_eff = max(h_eff.GetMaximum(), max_y_eff)
            if h_eff.GetMinimum(0.) > 0.:
                print(h_eff.GetMinimum(0.))
                min_y_eff = min(h_eff.GetMinimum(0.), min_y_eff)
            else:
                print(f"Smaller than/equal 0.: {hyield.GetMinimum()} in period {period} for " \
                      f"multiplicity {imult} and case {case}")
            comment_eff = ""
            if h_eff.Integral() <= 0. or h_eff.GetEntries() == 0:
                print(f"Empty period {period}, {case}, {ana_type}, mult {imult}")
                comment_eff = "(empty)"
            legyieldstring = "%.1f #leq %s < %.1f (%s), %s %s" \
                    % (binsmin[imult], latexbin2var, binsmax[imult], ana_type, period, comment_eff)
            leg_eff.AddEntry(h_eff, legyieldstring, "LEP")

            print(f"Mult {imult}, period {period}")
            print(f"In file {root_file}")

            hyield = root_file.Get("histoSigmaCorr%d" % (imult))
            hyield.Scale(1./(br * sigmav0 * 1e12))
            hyield.SetLineStyle(linestyles[counter % len(linestyles)])
            hyield.SetLineColor(colors[counter % len(colors)])
            hyield.SetMarkerStyle(markerstyles[counter % len(markerstyles)])
            hyield.SetMarkerColor(colors[counter % len(colors)])
            hyield.GetXaxis().SetTitleSize(0.02)
            hyield.GetXaxis().SetTitleSize(0.02)
            hyield.GetYaxis().SetTitleSize(0.02)
            histos_cross.append(hyield)
            max_y = max(hyield.GetMaximum(), max_y)
            if hyield.GetMinimum(0.) > 0.:
                print(hyield.GetMinimum(0.))
                min_y = min(hyield.GetMinimum(0.), min_y)
            else:
                print(f"Smaller than/equal 0.: {hyield.GetMinimum()} in period {period} for " \
                      f"multiplicity {imult} and case {case}")
            comment = ""
            if hyield.Integral() <= 0. or hyield.GetEntries() == 0:
                print(f"Empty period {period}, {case}, {ana_type}, mult {imult}")
                comment = "(empty)"
            legyieldstring = "%.1f #leq %s < %.1f (%s), %s %s" % \
                        (binsmin[imult], latexbin2var, binsmax[imult], ana_type, period, comment)
            legyield.AddEntry(hyield, legyieldstring, "LEP")
            counter += 1

        # Efficiencies
        ceff.cd(1).DrawFrame(0, min_y_eff / 1000., 30, 1000. * max_y_eff,
                             ";#it{p}_{T} (GeV/#it{c}); Efficiencies %s" % name)
        ceff.cd()
        gPad.SetLogy()
        gPad.SetLogy()
        for h in histos_eff:
            h.Draw("same")
        leg_eff.Draw()

        ceff.SaveAs("ComparisonEffs_%s_%s_combined%s_%d.eps" \
                % (case, ana_type, "_".join(periods), imult))
        ceff.Close()

        # Prepare ratio plot, year/merged (merged is always first in the histo list)
        double_ratios = []
        min_y_r = 9999.
        max_y_r = 0.
        for p in range(1, len(periods)):
            h_double_r = histos_cross[p].Clone(f"{histos_cross[p].GetName()}_ratio")
            h_double_r.Divide(h_double_r, histos_cross[0], 1, 1, "B")
            #h_double_r.Divide(histos[0])
            double_ratios.append(h_double_r)

            max_y_r = max(h_double_r.GetMaximum(), max_y_r)
            min_y_r = min(h_double_r.GetMinimum(), min_y_r)

            h_double_r.SetLineColor(colors[p % len(colors)])
            h_double_r.SetLineStyle(linestyles[p % len(linestyles)])
            h_double_r.SetMarkerStyle(markerstyles[p % len(markerstyles)])
            h_double_r.SetMarkerColor(colors[p % len(colors)])
            h_double_r.SetLineWidth(1)

        #Corrected yield plot
        ccross = TCanvas('cCross', 'The Fit Canvas')
        ccross.SetCanvasSize(1500, 1500)
        ccross.SetWindowSize(500, 500)
        ccross.cd()
        pad_up = TPad("pad_up", "", 0., 0.4, 1., 1.)
        pad_up.SetBottomMargin(0.)
        pad_up.Draw()
        #ccross.SetLogx()
        pad_up.cd()
        pad_up.SetLogy()
        pad_up.DrawFrame(0, min_y / 100., 30, 100. * max_y, ";;Corrected yield %s" % name)
        for h in histos_cross:
            h.Draw("same")
        legyield.Draw()

        ccross.cd()
        pad_double = TPad("pad_double", "", 0., 0.05, 1., 0.4)
        pad_double.SetTopMargin(0.)
        pad_double.SetBottomMargin(0.3)
        pad_double.Draw()
        pad_double.cd()
        frame_double = pad_double.DrawFrame(0, 0.5 * min_y_r, 30, 2 * max_y_r,
                                            ";#it{p}_{T} (GeV/#it{c}); year/merged")
        frame_double.SetTitleFont(42, "xy")
        frame_double.SetTitleSize(0.04, "xy")
        frame_double.SetLabelSize(0.04, "xy")
        for h in double_ratios:
            h.Draw("same")

        line_unity = TLine(frame_double.GetXaxis().GetXmin(), 1.,
                           frame_double.GetXaxis().GetXmax(), 1.)
        line_unity.SetLineColor(histos_cross[0].GetLineColor())
        line_unity.SetLineStyle(histos_cross[0].GetLineStyle())
        line_unity.Draw()

        ccross.SaveAs("ComparisonCorrYields_%s_%s_combined%s_%d.eps" % \
                  (case, ana_type, "_".join(periods), imult))
        ccross.Close()

#####################################

gROOT.SetBatch(True)

plot_hfspectrum_years("LcpK0spp", "MBvspt_ntrkl")
plot_hfspectrum_years("LcpK0spp", "MBvspt_v0m")
plot_hfspectrum_years("LcpK0spp", "MBvspt_perc")
plot_hfspectrum_years("LcpK0spp", "V0mvspt")
plot_hfspectrum_years("LcpK0spp", "V0mvspt_perc_v0m")
plot_hfspectrum_years("LcpK0spp", "SPDvspt")

plot_hfspectrum_years("D0pp", "MBvspt_ntrkl")
plot_hfspectrum_years("D0pp", "MBvspt_v0m")
plot_hfspectrum_years("D0pp", "MBvspt_perc")
plot_hfspectrum_years("D0pp", "V0mvspt")
plot_hfspectrum_years("D0pp", "V0mvspt_perc_v0m")
plot_hfspectrum_years("D0pp", "SPDvspt")

plot_hfspectrum_years("Dspp", "MBvspt_ntrkl")
plot_hfspectrum_years("Dspp", "MBvspt_v0m")
plot_hfspectrum_years("Dspp", "MBvspt_perc")
plot_hfspectrum_years("Dspp", "V0mvspt")
plot_hfspectrum_years("Dspp", "V0mvspt_perc_v0m")
plot_hfspectrum_years("Dspp", "SPDvspt")

plot_hfspectrum_years("LcpKpipp", "MBvspt_ntrkl")
plot_hfspectrum_years("LcpKpipp", "MBvspt_v0m")
plot_hfspectrum_years("LcpKpipp", "MBvspt_perc")
plot_hfspectrum_years("LcpKpipp", "V0mvspt")
plot_hfspectrum_years("LcpKpipp", "V0mvspt_perc_v0m")
plot_hfspectrum_years("LcpKpipp", "SPDvspt")

print("RATIOS over D0")
plot_hfspectrum_years_ratios("Dspp", "D0pp", "MBvspt_ntrkl")
plot_hfspectrum_years_ratios("Dspp", "D0pp", "MBvspt_v0m")
plot_hfspectrum_years_ratios("Dspp", "D0pp", "MBvspt_perc")
plot_hfspectrum_years_ratios("Dspp", "D0pp", "V0mvspt")
plot_hfspectrum_years_ratios("Dspp", "D0pp", "V0mvspt_perc_v0m")
plot_hfspectrum_years_ratios("Dspp", "D0pp", "SPDvspt")

plot_hfspectrum_years_ratios("LcpK0spp", "D0pp", "MBvspt_ntrkl")
plot_hfspectrum_years_ratios("LcpK0spp", "D0pp", "MBvspt_v0m")
plot_hfspectrum_years_ratios("LcpK0spp", "D0pp", "MBvspt_perc")
plot_hfspectrum_years_ratios("LcpK0spp", "D0pp", "V0mvspt")
plot_hfspectrum_years_ratios("LcpK0spp", "D0pp", "V0mvspt_perc_v0m")
plot_hfspectrum_years_ratios("LcpK0spp", "D0pp", "SPDvspt")

plot_hfspectrum_years_ratios("LcpKpipp", "D0pp", "MBvspt_ntrkl")
plot_hfspectrum_years_ratios("LcpKpipp", "D0pp", "MBvspt_v0m")
plot_hfspectrum_years_ratios("LcpKpipp", "D0pp", "MBvspt_perc")
plot_hfspectrum_years_ratios("LcpKpipp", "D0pp", "V0mvspt")
plot_hfspectrum_years_ratios("LcpKpipp", "D0pp", "V0mvspt_perc_v0m")
plot_hfspectrum_years_ratios("LcpKpipp", "D0pp", "SPDvspt")
