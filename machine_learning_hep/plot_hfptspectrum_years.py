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
import yaml
from ROOT import TFile, TH1F, TCanvas
from ROOT import gStyle, TLegend, TLatex
from ROOT import gROOT, kRed, kGreen, kBlack, kBlue
from ROOT import TStyle, gPad, TPad, TLine

files_not_found = []

def find_axes_limits(histos, use_log_y=False):

    max_y = histos[0].GetMaximum()
    min_y = histos[0].GetMinimum()
    if not min_y > 0. and use_log_y:
        min_y = 10.e-9

    max_x = histos[0].GetXaxis().GetXmax()
    min_x = histos[0].GetXaxis().GetXmin()

    for h in histos:
        min_x = min(min_x, h.GetXaxis().GetXmin())
        max_x = max(max_x, h.GetXaxis().GetXmax())
        min_y = min(min_y, h.GetMinimum(0.)) if use_log_y else min(min_y, h.GetMinimum())
        max_y = max(max_y, h.GetMaximum())

    return min_x, max_x, min_y, max_y

def style_histograms(histos):

    linestyles = [1, 1, 1, 1]
    markerstyles = [2, 4, 5, 32]
    colors = [kBlack, kRed, kGreen+2, kBlue]

    for i, h in enumerate(histos):
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineStyle(linestyles[i % len(linestyles)])
        h.SetMarkerStyle(markerstyles[i % len(markerstyles)])
        h.SetMarkerColor(colors[i % len(colors)])
        h.GetXaxis().SetTitleSize(0.02)
        h.GetXaxis().SetTitleSize(0.02)
        h.GetYaxis().SetTitleSize(0.02)

def divide_all_by_first(histos):

    histos_ratio = []
    for h in histos:
        histos_ratio.append(h.Clone(f"{h.GetName()}_ratio"))
        histos_ratio[-1].Divide(histos[0])
    return histos_ratio

def put_in_pad(pad, use_log_y, histos, title, x_label, y_label):
    min_x, max_x, min_y, max_y = find_axes_limits(histos, use_log_y)
    pad.SetLogy(use_log_y)
    pad.cd()
    scale_frame_y = (0.1, 10.) if use_log_y else (0.7, 1.2)
    pad.DrawFrame(min_x, min_y * scale_frame_y[0], max_x, max_y * scale_frame_y[1],
                  f"{title};{x_label};{y_label}")
    for h in histos:
        h.Draw("same")

# The first histogram in the list is assumed to be the others need to be divided by
def make_ratio_plot(histos, use_log_y, legend_titles, title, x_label, y_label_up, y_label_ratio,
                    save_path):

    style_histograms(histos)
    histos_ratio = divide_all_by_first(histos)

    canvas = TCanvas('canvas', 'The Fit Canvas', 800, 800)
    pad_up = TPad("pad_up", "", 0., 0.4, 1., 1.)
    pad_up.SetBottomMargin(0.)
    pad_up.Draw()

    #Corrected yield plot
    legend = TLegend(.45, .65, .85, .85)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.02)

    put_in_pad(pad_up, use_log_y, histos, title, "", y_label_up)
    for h, l in zip(histos, legend_titles):
        legend.AddEntry(h, l)

    pad_up.cd()
    legend.Draw()

    canvas.cd()
    pad_ratio = TPad("pad_ratio", "", 0., 0.05, 1., 0.4)
    pad_ratio.SetTopMargin(0.)
    pad_ratio.SetBottomMargin(0.3)
    pad_ratio.Draw()

    put_in_pad(pad_ratio, False, histos_ratio, "", x_label, y_label_ratio)

    canvas.SaveAs(save_path)
    canvas.Close()

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches, too-many-locals
def plot_hfspectrum_years_ratios(case_1, case_2, ana_type, mult_bins=[0, 1, 2, 3]):

    with open("data/database_ml_parameters_%s.yml" % case_1, 'r') as param_config:
        data_param_1 = yaml.load(param_config, Loader=yaml.FullLoader)

    with open("data/database_ml_parameters_%s.yml" % case_2, 'r') as param_config:
        data_param_2 = yaml.load(param_config, Loader=yaml.FullLoader)

    folder_plots_1 = data_param_1[case_1]["analysis"]["dir_general_plots"]
    folder_plots_2 = data_param_2[case_2]["analysis"]["dir_general_plots"]
    folder_plots_1 = folder_plots_1 + "/comp_years"
    folder_plots_2 = folder_plots_2 + "/comp_years"
    if not os.path.exists(folder_plots_1):
        print("creating folder ", folder_plots_1)
        os.makedirs(folder_plots_1)
    if not os.path.exists(folder_plots_2):
        print("creating folder ", folder_plots_2)
        os.makedirs(folder_plots_2)

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

    #br_1 = data_param_1[case_1]["ml"]["opt"]["BR"]
    #br_2 = data_param_2[case_2]["ml"]["opt"]["BR"]
    #sigmav0_1 = data_param_1[case_1]["analysis"]["sigmav0"]
    #sigmav0_2 = data_param_2[case_2]["analysis"]["sigmav0"]

    files_mult_1 = []
    files_mult_2 = []
    periods_string = "_".join(periods)
    for imult in mult_bins:
        files_years_1 = []
        files_years_2 = []
        for folder_1, folder_2 in zip(result_paths_1, results_path_2):
            path_1 = f"{folder_1}/finalcross{case_1}{ana_type}mult{imult}.root"
            path_2 = f"{folder_2}/finalcross{case_2}{ana_type}mult{imult}.root"
            if not os.path.exists(path_1) or not os.path.exists(path_2):
                files_not_found.append(f"{path_1} or {path_2}")
                continue
            files_years_1.append(TFile.Open(path_1, "READ"))
            files_years_2.append(TFile.Open(path_2, "READ"))

        files_mult_1.append(files_years_1)
        files_mult_2.append(files_years_2)

        histos = []
        legend_titles = []
        for period, root_file_1, root_file_2 in zip(periods, files_years_1, files_years_2):
            hyield_1 = root_file_1.Get("histoSigmaCorr")
            hyield_2 = root_file_2.Get("histoSigmaCorr")
            #hyield_1.Scale(1./(br_1 * sigmav0_1 * 1e12))
            #hyield_2.Scale(1./(br_2 * sigmav0_2 * 1e12))
            hyield_ratio = hyield_1.Clone(f"{case_1}_{case_2}_ratio_{period}_{imult}")
            hyield_ratio.Divide(hyield_2)
            histos.append(hyield_ratio)

            l_string = f"{binsmin[imult]:.1f} #leq {latexbin2var} < {binsmax[imult]:.1f} "\
                       f"({ana_type}), {period}"
            legend_titles.append(l_string)

        sub_folder = os.path.join(folder_plots, "ratios", ana_type)
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        save_path = f"{sub_folder}/{histos[0].GetName()}_combined_{periods_string}_{imult}.eps"

        make_ratio_plot(histos, True, legend_titles, histos[0].GetTitle(),
                        "#it{p}_{T} (GeV/#it{c})", histos[0].GetYaxis().GetTitle(),
                        "year / merged", save_path)

# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches, too-many-locals
def plot_hfspectrum_years(case, ana_type, mult_bins=[0, 1, 2, 3]):

    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)

    folder_plots = data_param[case]["analysis"]["dir_general_plots"]
    folder_plots = folder_plots + "/comp_years"
    if not os.path.exists(folder_plots):
        print("creating folder ", folder_plots)
        os.makedirs(folder_plots)

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

    for imult in mult_bins:
        files_years = []
        for folder in result_paths:
            path = f"{folder}/finalcross{case}{ana_type}mult{imult}.root"
            if not os.path.exists(path):
                files_not_found.append(path)
                continue
            files_years.append(TFile.Open(path, "READ"))
        files_mult.append(files_years)

    print("################")
    print(f"case {case} in analysis {ana_type}")

    histo_names = ["hDirectMCpt", "hFeedDownMCpt", "hDirectMCptMax", "hDirectMCptMin",
                   "hFeedDownMCptMax", "hFeedDownMCptMin", "hDirectEffpt", "hFeedDownEffpt",
                   "hRECpt", "histoYieldCorr", "histoYieldCorrMax", "histoYieldCorrMin",
                   "histoSigmaCorr", "histoSigmaCorrMax", "histoSigmaCorrMin"]

    periods_string = "_".join(periods)

    for hn in histo_names:

        for imult in mult_bins:

            histos = []
            legend_titles = []
            for period, root_file in zip(periods, files_mult[imult]):

                print(f"Mult {imult}, period {period}")
                print(f"In file {root_file}")

                if not root_file:
                    print(f"COULD NOT OPEN ROOT FILE FOR {imult}, {period}")
                    continue
                histos.append(root_file.Get(hn))
                comment = ""
                if histos[-1].Integral() <= 0. or histos[-1].GetEntries() == 0:
                    print(f"Empty period {period}, {case}, {ana_type}, mult {imult}")
                    comment = "(empty)"
                l_string = f"{binsmin[imult]:.1f} #leq {latexbin2var} < {binsmax[imult]:.1f} "\
                           f"({ana_type}), {period} {comment}"
                legend_titles.append(l_string)

            sub_folder = os.path.join(folder_plots, ana_type)
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            save_path = f"{sub_folder}/{hn}_combined_{periods_string}_{imult}.eps"
            
            make_ratio_plot(histos, True, legend_titles, histos[0].GetTitle(),
                            "#it{p}_{T} (GeV/#it{c})", histos[0].GetYaxis().GetTitle(),
                            "year / merged", save_path)

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

print("FILES NOT FOUND:")
for f in files_not_found:
    print(f)
