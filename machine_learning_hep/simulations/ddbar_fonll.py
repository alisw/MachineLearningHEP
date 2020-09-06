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
preliminary studies for cross section estimation
"""
from array import array
import pandas as pd
from ROOT import TCanvas, TH1F, gROOT, TLatex, gPad  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.utilities import setup_histogram, draw_latex
from machine_learning_hep.utilities_plot import load_root_style

p_fonllband = 'max'
ptmin = 0
ptmax = 30
delta_pt = ptmax - ptmin
p_fragf = 0.6086
f_fonll = "fo_pp_d0meson_5TeV_y0p5.csv"
p_br = 3.95e-2
p_ncoll = 401
p_sigmamb = 57.8e-3
acc = 1
p_fprompt = 1

gROOT.SetBatch(True)

pt_range = [0, 2, 4, 6, 8, 12, 16, 20, 30]
eff_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]
effAA_range = [0.001, 0.003, 0.01, 0.02, 0.06, 0.1, 0.1, 0.1, 0.1]
raa_range = [0.8, 0.7, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3]
bins = array('f', pt_range)

hfonllc = TH1F("hfonllc", "", len(pt_range) - 1, bins)
hfonllDtoKpi = TH1F("hfonllDtoKpi", "", len(pt_range) - 1, bins)
hyieldc = TH1F("hyieldc", "", len(pt_range) - 1, bins)
hyieldDtoKpi = TH1F("hyieldDtoKpi", "", len(pt_range) - 1, bins)
hyieldDtoKpirsel = TH1F("hyieldDtoKpirsel", "", len(pt_range) - 1, bins)
hyieldcAA = TH1F("hyieldcAA", "", len(pt_range) - 1, bins)
hyieldDtoKpiAA = TH1F("hyieldDtoKpiAA", "", len(pt_range) - 1, bins)
hyieldDtoKpirselAA = TH1F("hyieldDtoKpirselAA", "", len(pt_range) - 1, bins)

hyieldDtoKpipairrsel = TH1F("hyieldDtoKpipairrsel", "", len(pt_range) - 1, bins)
hyieldDtoKpipairrselAA = TH1F("hyieldDtoKpirselpairAA", "", len(pt_range) - 1, bins)

df_fonll = pd.read_csv(f_fonll)

for i, ptmin in enumerate(pt_range):
    if i == len(pt_range) - 1:
        break
    ptmax = pt_range[i+1]
    binwidth = pt_range[i+1] - pt_range[i]
    df_fonll_in_pt = df_fonll.query('(pt >= @ptmin) and (pt < @ptmax)')[p_fonllband]
    crossc = df_fonll_in_pt.sum() * 1e-12 /binwidth
    cross = df_fonll_in_pt.sum() * p_br * p_fragf * 1e-12 /binwidth
    hfonllc.SetBinContent(i+1, crossc)
    hfonllDtoKpi.SetBinContent(i+1, cross)
    hyieldDtoKpirsel.SetBinContent(i+1, cross * binwidth * eff_range[i] / p_sigmamb)
    hyieldDtoKpi.SetBinContent(i+1, cross * binwidth / p_sigmamb)
    hyieldc.SetBinContent(i+1, crossc * binwidth / p_sigmamb)
    hyieldDtoKpiAA.SetBinContent(i+1, cross * binwidth * p_ncoll / p_sigmamb)
    hyieldDtoKpirselAA.SetBinContent(i+1, cross * binwidth * p_ncoll * effAA_range[i]/ p_sigmamb)
    hyieldcAA.SetBinContent(i+1, crossc * binwidth * p_ncoll/ p_sigmamb)

    hyieldDtoKpipairrsel.SetBinContent(i+1, cross * binwidth * eff_range[i] *
                                       eff_range[i] * p_br * p_fragf/ p_sigmamb)
    hyieldDtoKpipairrselAA.SetBinContent(i+1, cross * binwidth * effAA_range[i]
                                         * effAA_range[i] * raa_range[i] *
                                         raa_range[i] * p_br * p_fragf * p_ncoll/ p_sigmamb)
    print("min,max", ptmin, ptmax, cross)

load_root_style()

histo_list = [hfonllc, hyieldc, hyieldcAA, hfonllDtoKpi,
              hyieldDtoKpi, hyieldDtoKpirsel, hyieldDtoKpiAA,
              hyieldDtoKpirselAA, hyieldDtoKpipairrsel, hyieldDtoKpipairrselAA]
min_list = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
max_list = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]
xaxis_list = ["p_{T} (GeV)", "p_{T} (GeV)", "p_{T} (GeV)", \
              "p_{T} (GeV)", "p_{T} (GeV)", "p_{T} (GeV)",
              "p_{T} (GeV)", "p_{T} (GeV)", "p_{T} (GeV)", "p_{T} (GeV)"]
yaxis_list = ["d#sigma/dp_{T} (b/GeV)", "Counts", "Counts", \
              "d#sigma/dp_{T} (b/GeV)", "Counts", "Counts",
              "Counts", "Counts", "Counts", "Counts"]
text_list = ["c-quark production cross section",
             "Average number of c quarks per event pp",
             "Average number of c quarks per event PbPb",
             "D^{0} #rightarrow K#pi (BR included) in pp",
             "Average number of D^{0} per event pp",
             "Average number of D^{0} per event pp recosel",
             "Average number of D^{0} per event PbPb",
             "Average number of D^{0} per event PbPb recosel",
             "Average number of D^{0}-D^{0}bar pair per event pp recosel",
             "Average number of D^{0}-D^{0}bar pair per event AA recosel"]
list_latex = []
c = TCanvas("canvas", "canvas", 3000, 2000)
c.Divide(4, 3)
for i, _ in enumerate(xaxis_list):
    c.cd(i + 1)
    gPad.SetLogy()
    setup_histogram(histo_list[i])
    histo_list[i].SetMinimum(min_list[i])
    histo_list[i].SetMaximum(max_list[i])
    histo_list[i].SetXTitle(xaxis_list[i])
    histo_list[i].SetYTitle(yaxis_list[i])
    latex = TLatex(0.2, 0.83, text_list[i])
    list_latex.append(latex)
    histo_list[i].Draw()
    draw_latex(list_latex[i], 1, 0.04)
c.SaveAs("generated_level.pdf")

#### estimated predictions ####
hyieldDtoKpipairrsel2B = hyieldDtoKpipairrsel.Clone("hyieldDtoKpipairrsel2B")
hyieldDtoKpipairrsel200B = hyieldDtoKpipairrsel.Clone("hyieldDtoKpipairrsel200B")
hyieldDtoKpipairrselAA100M = hyieldDtoKpipairrselAA.Clone("hyieldDtoKpipairrselAA100M")
hyieldDtoKpipairrselAA50B = hyieldDtoKpipairrselAA.Clone("hyieldDtoKpipairrselAA50B")
hyieldDtoKpipairrselAA2500B = hyieldDtoKpipairrselAA.Clone("hyieldDtoKpipairrselAA2500B")

histo_list_est = [hyieldDtoKpipairrsel2B, hyieldDtoKpipairrsel200B,
                  hyieldDtoKpipairrselAA100M, hyieldDtoKpipairrselAA50B,
                  hyieldDtoKpipairrselAA2500B]
min_list_est = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
max_list_est = [1e10, 1e10, 1e10, 1e10, 1e10]
xaxis_list_est = ["p_{T} (GeV)", "p_{T} (GeV)", "p_{T} (GeV)", \
                  "p_{T} (GeV)", "p_{T} (GeV)"]
yaxis_list_est = ["Counts", "Counts", "Counts", "Counts", "Counts"]
text_list_est = ["D^{0}-D^{0}bar pairs pp recosel 2B",
                 "D^{0}-D^{0}bar pairs pp recosel 200B",
                 "D^{0}-D^{0}bar pairs pp recosel 100M",
                 "D^{0}-D^{0}bar pairs pp recosel 50B",
                 "D^{0}-D^{0}bar pairs pp recosel 2500B"]
nevents_list_ext = [2e9, 200*2e9, 100*1e6, 50*1e9, 2500*1e9]

for ihisto, _ in enumerate(histo_list_est):
    histo_list_est[ihisto].Scale(nevents_list_ext[ihisto])

list_est_latex = []
c_est = TCanvas("canvas", "canvas", 3000, 2000)
c_est.Divide(3, 2)
for i, _ in enumerate(xaxis_list_est):
    c_est.cd(i + 1)
    gPad.SetLogy()
    setup_histogram(histo_list_est[i])
    histo_list_est[i].SetMinimum(min_list_est[i])
    histo_list_est[i].SetMaximum(max_list_est[i])
    histo_list_est[i].SetXTitle(xaxis_list_est[i])
    histo_list_est[i].SetYTitle(yaxis_list_est[i])
    latex = TLatex(0.2, 0.83, text_list_est[i])
    list_est_latex.append(latex)
    histo_list_est[i].Draw("HIST")
    draw_latex(list_est_latex[i], 1, 0.04)
c_est.SaveAs("generated_level_est.pdf")
