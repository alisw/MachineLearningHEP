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
import os
import time
from array import array
from math import sqrt
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ROOT import TFile, TCanvas, TH1F, TF1, gROOT, TLatex, gPad  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.utilities import setup_histogram, setup_canvas, draw_latex
from machine_learning_hep.utilities_plot import load_root_style

p_fonllband='max'
ptmin=0
ptmax=30
delta_pt = ptmax - ptmin
p_fragf = 0.6086
f_fonll = "fo_pp_d0meson_5TeV_y0p5.csv"
p_br = 3.95e-2
p_ncoll = 401
p_sigmamb = 57.8e-3
acc = 1
p_fprompt = 1

pt_range = [0,2,4,6,8,12,16,20,30]
eff_range = [0.1,0.2,0.3,0.4,0.5,0.5,0.5,0.5,0.5]
effAA_range = [0.1,0.2,0.3,0.4,0.5,0.5,0.5,0.5,0.5]
bins = array( 'f', pt_range)

hfonllc = TH1F("hfonllc", "", len(pt_range) - 1, bins)
hfonllDtoKpi = TH1F("hfonllDtoKpi", "", len(pt_range) - 1, bins)
hyieldc = TH1F("hyieldc", "", len(pt_range) - 1, bins)
hyieldDtoKpi = TH1F("hyieldDtoKpi", "", len(pt_range) - 1, bins)
hyieldDtoKpirsel = TH1F("hyieldDtoKpi", "", len(pt_range) - 1, bins)
hyieldcAA = TH1F("hyieldcAA", "", len(pt_range) - 1, bins)
hyieldDtoKpiAA = TH1F("hyieldDtoKpiAA", "", len(pt_range) - 1, bins)
hyieldDtoKpirselAA = TH1F("hyieldDtoKpirselAA", "", len(pt_range) - 1, bins)
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
    print("min,max", ptmin, ptmax, cross)

load_root_style()

histo_list = [hfonllc, hyieldc, hyieldcAA, hfonllDtoKpi,
              hyieldDtoKpi, hyieldDtoKpirsel, hyieldDtoKpiAA, hyieldDtoKpirselAA]
min_list = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
max_list = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]
xaxis_list = ["p_{T} (GeV)", "p_{T} (GeV)", "p_{T} (GeV)", \
              "p_{T} (GeV)", "p_{T} (GeV)", "p_{T} (GeV)",
              "p_{T} (GeV)", "p_{T} (GeV)"]
yaxis_list = ["d#sigma/dp_{T} (b/GeV)", "Counts", "Counts", \
              "d#sigma/dp_{T} (b/GeV)", "Counts", "Counts",
              "Counts", "Counts"]
text_list = ["c-quark production cross section",
             "Average number of c quarks per event pp",
             "Average number of c quarks per event PbPb",
             "D^{0} #rightarrow K#pi (BR included) in pp",
             "Average number of D^{0} per event pp",
             "Average number of D^{0} per event pp recosel",
             "Average number of D^{0} per event PbPb",
             "Average number of D^{0} per event PbPb recosel"]
list_latex = []
c = TCanvas("canvas", "canvas", 2000, 1200);
c.Divide(3, 3)
for i in range(len(xaxis_list)):
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
c.SaveAs("generated_level.png")
