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
bins = array( 'f', pt_range)

hfonllc = TH1F("hfonllc", "", len(pt_range) - 1, bins)
hfonllDtoKpi = TH1F("hfonllDtoKpi", "", len(pt_range) - 1, bins)
hyieldc = TH1F("hyieldc", "", len(pt_range) - 1, bins)
hyield = TH1F("hyield", "", len(pt_range) - 1, bins)
hyieldcAA = TH1F("hyieldcAA", "", len(pt_range) - 1, bins)
hyieldAA = TH1F("hyieldAA", "", len(pt_range) - 1, bins)
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
    hyield.SetBinContent(i+1, 2 * cross * binwidth / p_sigmamb)
    hyieldc.SetBinContent(i+1, crossc * binwidth / p_sigmamb)
    hyieldAA.SetBinContent(i+1, 2 * cross * binwidth * p_ncoll / p_sigmamb)
    hyieldcAA.SetBinContent(i+1, crossc * binwidth * p_ncoll/ p_sigmamb)
    print("min,max", ptmin, ptmax, cross)

load_root_style()
c = TCanvas("canvas", "canvas", 2000, 1200);
c.Divide(3, 2)
#setup_canvas(c)
c.cd(1)
gPad.SetLogy()
c.SetLogy()
setup_histogram(hfonllc)
hfonllDtoKpi.SetMinimum(1e-8);
hfonllc.SetMaximum(1e3);
hfonllc.Draw()
hfonllc.SetXTitle("p_{T} (GeV)")
hfonllc.SetYTitle("d#sigma/dp_{T} (b/GeV)")
latexfonllc = TLatex(0.2, 0.83, "c-quark production cross section")
draw_latex(latexfonllc, 1, 0.04)
c.cd(2)
gPad.SetLogy()
setup_histogram(hyieldc)
hyieldc.SetMinimum(1e-8);
hyieldc.SetMaximum(1e3);
hyieldc.Draw()
hyieldc.SetXTitle("p_{T} (GeV)")
latexyieldc = TLatex(0.2, 0.83, "Average number of charm quark per event")
draw_latex(latexyieldc, 1, 0.04)
hyieldc.SetYTitle("Counts")
c.cd(3)
gPad.SetLogy()
setup_histogram(hyieldcAA)
hyieldcAA.SetMinimum(1e-8);
hyieldcAA.SetMaximum(1e3);
hyieldcAA.Draw()
hyieldcAA.SetXTitle("p_{T} (GeV)")
latexyieldcAA = TLatex(0.2, 0.83, "Average number of charm quark per event in AA")
draw_latex(latexyieldcAA, 1, 0.04)
hyieldcAA.SetYTitle("Counts")
c.cd(4)
gPad.SetLogy()
c.SetLogy()
setup_histogram(hfonllDtoKpi)
hfonllDtoKpi.SetMinimum(1e-8);
hfonllDtoKpi.SetMaximum(1e3);
hfonllDtoKpi.Draw()
hfonllDtoKpi.SetXTitle("p_{T} (GeV)")
hfonllDtoKpi.SetYTitle("d#sigma/dp_{T} (b/GeV)")
latexfonll = TLatex(0.2, 0.83, "FONLL upper band D^{0} #rightarrow K#pi (BR included)")
draw_latex(latexfonll, 1, 0.04)
c.cd(5)
gPad.SetLogy()
setup_histogram(hyield)
hyield.SetMinimum(1e-8);
hyield.SetMaximum(1e3);
hyield.Draw()
hyield.SetXTitle("p_{T} (GeV)")
latexyield = TLatex(0.2, 0.83, "Average number of D^{0} and antiparticles per event")
draw_latex(latexyield, 1, 0.04)
hyield.SetYTitle("Counts")
c.cd(6)
gPad.SetLogy()
setup_histogram(hyieldAA)
hyieldAA.SetMinimum(1e-8);
hyieldAA.SetMaximum(1e3);
hyieldAA.Draw()
hyieldAA.SetXTitle("p_{T} (GeV)")
latexyieldAA = TLatex(0.2, 0.83, "Average number of D^{0} and antiparticles per event in AA")
draw_latex(latexyieldAA, 1, 0.04)
hyieldAA.SetYTitle("Counts")
c.SaveAs("generated_level.png")
