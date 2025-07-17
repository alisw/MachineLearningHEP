#!/usr/bin/env python3

# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.

# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".

# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

"""
file: plot_invmass_fit_dzero_dplus_lambdac.py
brief: script to produce invariant mass fit plot for article, the CFG file should be the one used for inv mass fit
usage: python3 plot_invmass_fit_dzero_dplus_lambdac.py CFG
author: Alexandre Bigot <alexandre.bigot@cern.ch>, Strasbourg University
"""

import argparse

import yaml
import ROOT
from ROOT import (TF1, TCanvas, TDatabasePDG, TFile, TLatex, TLegend, TMath,
                  gROOT, kAzure, kBlack, kBlue, kGreen, kFullCircle, kRed, TPad)

from style_formatter import set_global_style, set_object_style

# enumerator
D0, DPLUS, LAMBDAC_TO_PKPI, LAMBDAC_TO_PK0S = 0, 1, 2, 3

# colours
RED = kRed + 1
BLUE = kBlue + 1
AZURE = kAzure + 4
Green = kGreen + 1

# conversion
GEV2MEV = 1000

# canvas dimensions
WIDTH = 520
HEIGHT = 500

# text size
SIZE_TEXT_LAT_ALICE = 28
SIZE_TEXT_LAT_LABEL_FOR_COLL_SYSTEM = 24
SIZE_TEXT_LAT_LABEL = 20
SIZE_TEXT_LEGEND = 19


def get_name_infile(particle, suffix):
    """
    Helper method to get the name of the input file according to the particle

    Parameters
    ----------
    - particle (int): particle ID

    Returns
    ----------
    - name_infile (string): name of the input file
    """

    name_infile = ""
    name_infile_promptEnhanced = ""
    name_infile_FDEnhanced = ""
    if particle == D0:
        name_infile_promptEnhanced = "../RawYieldResult/CentralValue/RawYieldsData_D0_pPb5TeV_FD_pos00.root"
        name_infile_FDEnhanced = "../RawYieldResult/CentralValue/RawYieldsData_D0_pPb5TeV_FD_pos13.root"
    elif particle == DPLUS:
        name_infile = "../Results/Dplus/rawYield_Dplus_nonprompt_enhanced.root"
    elif particle == LAMBDAC_TO_PKPI:
        name_infile_promptEnhanced = f"/data8/majak/invmass-plots/massesmasshisto{suffix}.root"
        name_infile_FDEnhanced = "fits_non_prompt.root"
    elif particle == LAMBDAC_TO_PK0S:
        name_infile = ""

    return name_infile_promptEnhanced, name_infile_FDEnhanced


def get_title_xaxis(particle):
    """
    Helper method to get the title of x axis according to the particle

    Parameters
    ----------
    - particle (int): particle ID

    Returns
    ----------
    - title_xaxis (string): title of x axis
    """

    title_xaxis = ""
    if particle == D0:
        title_xaxis = "#it{M}(K#pi) (GeV/#it{c}^{2})"
    elif particle == DPLUS:
        title_xaxis = "#it{M}(#piK#pi) (GeV/#it{c}^{2})"
    elif particle == LAMBDAC_TO_PKPI:
        title_xaxis = "#it{M}(pK#pi) (GeV/#it{c}^{2})"
    elif particle == LAMBDAC_TO_PK0S:
        title_xaxis = "#it{M}(pK^{0}_{S}) (GeV/#it{c}^{2})"

    return title_xaxis


def get_h_value_err(h, i_bin, convert_to_mev=False):
    """
    Helper method to get bin content and error of an histogram

    Parameters
    ----------
    - h (TH1): histogram
    - i_bin (int): bin number
    - convert_to_mev (int): apply conversion from GeV to MeV

    Returns
    ----------
    - value (float): bin content of h
    - error (float): bin error of h
    """

    value = h.GetBinContent(i_bin)
    error = h.GetBinError(i_bin)

    print(f"i_bin: {i_bin} value {value} error {error} first bin value: {h.GetBinContent(1)}")

    if convert_to_mev:
        value *= GEV2MEV
        error *= GEV2MEV

    return value, error


def draw_info(lat_label, particle):
    """
    Helper method to draw particle-dependent information on canvas

    Parameters
    ----------
    - lat_label (TLatex): TLatex instance
    - particle (int): particle ID
    """

    info = ""
    fnonprompt = ""
    if particle == D0:
        info = "D^{0} #rightarrow K^{#font[122]{-}}#pi^{+} and charge conj."
        # fnonprompt = "#it{f}_{ non-prompt}^{ raw} = 0.750 #pm  0.016 (stat.) #pm 0.008 (syst.)"
        #fnonprompt = "#it{f}_{ non-prompt}^{ raw} = 0.531"
    elif particle == DPLUS:
        info = "D^{+} #rightarrow #pi^{+}K^{#font[122]{-}}#pi^{+} and charge conj."
        fnonprompt = "#it{f}_{ non-prompt}^{ raw} = 0.787 #pm  0.022 (stat.) #pm 0.016 (syst.)"
    elif particle == LAMBDAC_TO_PKPI:
        info = "#Lambda_{c}^{+}  #rightarrow pK^{#font[122]{-}}#pi^{+} and charge conj."
        fnonprompt = "#it{f}_{ non-prompt}^{ raw} = 0.630 #pm  0.056 (stat.) #pm 0.050 (syst.)"
    elif particle == LAMBDAC_TO_PK0S:
        info = "#Lambda_{c}^{+}  #rightarrow pK^{0}_{S} and charge conj."
        fnonprompt = "#it{f}_{ non-prompt}^{ raw} = 0.549 #pm  0.138 (stat.) #pm 0.055 (syst.)"

    lat_label.DrawLatex(0.19, 0.85, info)
    #lat_label.DrawLatex(0.19, 0.16, fnonprompt)


def save_canvas(canvas, particle, pt_mins, pt_maxs, i_pt, mult):
    """
    Helper method to save canvas according to particle

    Parameters
    ----------
    - canvas (TCanvas): a canvas
    - particle (int): particle ID
    """

    out_dir = "/data8/majak/invmass-plots/"
    name = ""
    if particle == D0:
        name = "Dzero"
    elif particle == DPLUS:
        name = "Dplus"
    elif particle == LAMBDAC_TO_PKPI:
        name = "LambdacToPKPi"
    elif particle == LAMBDAC_TO_PK0S:
        name = "LambdacToPKzeroShort"

    mult = f"{mult[i_pt]}_" if mult[i_pt] else ""
    for ext in ["pdf", "png", "eps"]:
        canvas.SaveAs(f"{out_dir}InvMassFit{name}_{mult}Pt_{pt_mins[i_pt]:.0f}_{pt_maxs[i_pt]:.0f}.{ext}")


# pylint: disable=too-many-locals,too-many-statements
def main(particle, i_pt, cfg, batch):
    """
    Main method for a single bin (for article plots)

    Parameters
    ----------
    - particle (int): particle ID
    - i_pt (int): pT bin number
    """

    set_global_style(padtopmargin=0.07, padleftmargin=0.14, padbottommargin=0.125, titleoffsety=1.3, titleoffsetx=1., maxdigits=3)

    # import configurables
    pt_mins = cfg["pp13.6TeVFD"]["PtMin"]
    pt_maxs = cfg["pp13.6TeVFD"]["PtMax"]
    mass_mins = cfg["pp13.6TeVFD"]["MassMin"]
    mass_maxs = cfg["pp13.6TeVFD"]["MassMax"]
    rebin = cfg["pp13.6TeVFD"]["Rebin"]
    mult = cfg["pp13.6TeVFD"]["Mult"]
    mult_latex = cfg["pp13.6TeVFD"]["MultLatex"]

    print(f"Plotting for {pt_mins[i_pt]}-{pt_maxs[i_pt]}")

    name_infile_promptEnhanced, name_infile_FDEnhanced = get_name_infile(particle, f"{pt_mins[i_pt]:.0f}{pt_maxs[i_pt]:.0f}")

    file_promptEnhanced = TFile.Open(name_infile_promptEnhanced)
    #file_FDEnhanced = TFile.Open(name_infile_FDEnhanced)

    hmean_promptEnhanced = file_promptEnhanced.Get("hist_means_lc")
    hsigma_promptEnhanced = file_promptEnhanced.Get("hist_sigmas_lc")

    #hmean_FDEnhanced = file_FDEnhanced.Get("hRawYieldsMean")
    #hsigma_FDEnhanced = file_FDEnhanced.Get("hRawYieldsSigma")

    hsignal_promptEnhanced = file_promptEnhanced.Get("hist_rawyields_lc")
    #hsignal_FDEnhanced = file_FDEnhanced.Get("hRawYields")

    mult_suffix = f"_{mult[i_pt]}" if mult[i_pt] else ""
    name_hmass = f"hmass{pt_mins[i_pt]:.0f}{pt_maxs[i_pt]:.0f}{mult_suffix}"
    print(f"file {name_infile_promptEnhanced} hist {name_hmass}")
    hmass_promptEnhanced = file_promptEnhanced.Get(name_hmass)
    #hmass_FDEnhanced = file_FDEnhanced.Get(name_hmass)
    hmass_promptEnhanced.Rebin(rebin[i_pt])
    #hmass_FDEnhanced.Rebin(rebin[i_pt])

    title_xaxis = get_title_xaxis(particle)
    width_bin = hmass_promptEnhanced.GetBinWidth(i_pt+1)
    bin_max = hmass_promptEnhanced.GetMaximumBin()
    bin_min = hmass_promptEnhanced.GetMinimumBin()
    
    ymax_promptEnhanced = 1.2*(hmass_promptEnhanced.GetMaximum() + hmass_promptEnhanced.GetBinError(bin_max))
    ymin_promptEnhanced = 0.8*(hmass_promptEnhanced.GetMinimum() - hmass_promptEnhanced.GetBinError(bin_min))
    #ymin_FDEnhanced, ymax_FDEnhanced = 0., 1.2*(hmass_FDEnhanced.GetMaximum() + hmass_FDEnhanced.GetBinError(bin_max))

    title = f"{pt_mins[i_pt]:.0f} < #it{{p}}_{{T}} < {pt_maxs[i_pt]:.0f} GeV/#it{{c}};{title_xaxis};" \
        f"Counts per {width_bin*GEV2MEV:.0f} MeV/#it{{c}}^{{2}}"

    #fit_tot_promptEnhanced = file_promptEnhanced.Get(f"totalTF_{pt_mins[i_pt]:.0f}_{pt_maxs[i_pt]:.0f}")
    fit_tot_promptEnhanced = file_promptEnhanced.Get(f"total_func_lc_pt{pt_mins[i_pt]:.0f}_{pt_maxs[i_pt]:.0f}")
    #fit_bkg_promptEnhanced = file_promptEnhanced.Get(f"bkgTF_{pt_mins[i_pt]:.0f}_{pt_maxs[i_pt]:.0f}")
    fit_bkg_promptEnhanced = file_promptEnhanced.Get(f"bkg_0_lc_pt{pt_mins[i_pt]:.0f}_{pt_maxs[i_pt]:.0f}")
    #fit_refl_promptEnhanced = file_promptEnhanced.Get(f"freflect;13")



    #fit_tot_FDEnhanced = file_FDEnhanced.Get(f"totalTF_{pt_mins[i_pt]:.0f}.0_{pt_maxs[i_pt]:.0f}.0")
    #fit_bkg_FDEnhanced = file_FDEnhanced.Get(f"bkgTF_{pt_mins[i_pt]:.0f}.0_{pt_maxs[i_pt]:.0f}.0")
    #fit_refl_FDEnhanced = file_FDEnhanced.Get(f"freflect;13")

    print("Calculating mean")
    mean_promptEnhanced, err_mean_promptEnhanced = get_h_value_err(hmean_promptEnhanced, i_pt + 1, True)
    #mean_FDEnhanced, err_mean_FDEnhanced = get_h_value_err(hmean_FDEnhanced, 13, True)
    print("Calculating sigma")
    sigma_promptEnhanced, _ = get_h_value_err(hsigma_promptEnhanced, i_pt + 1, True)
    #sigma_FDEnhanced, _ = get_h_value_err(hsigma_FDEnhanced, 13, True)
    print("Calculating yield")
    signal_promptEnhanced, err_signal_promptEnhanced = get_h_value_err(hsignal_promptEnhanced, i_pt + 1)
    #signal_FDEnhanced, err_signal_FDEnhanced = get_h_value_err(hsignal_FDEnhanced, 13)

    lat_alice = TLatex()
    lat_alice.SetNDC()
    lat_alice.SetTextSize(SIZE_TEXT_LAT_ALICE)
    lat_alice.SetTextFont(43)
    lat_alice.SetTextColor(kBlack)

    lat_label = TLatex()
    lat_label.SetNDC()
    lat_label.SetTextFont(43)
    lat_label.SetTextColor(kBlack)

    # lat_label = TLatex()
    # lat_label.SetNDC()
    # lat_label.SetTextFont(43)
    # lat_label.SetTextColor(kBlack)

    # str_mu = f"#it{{#mu}} = ({mean:.0f} #pm {err_mean:.0f}) MeV/#it{{c}}^{{2}}"
    # str_sigma = f"#it{{#sigma}} = {sigma:.0f} MeV/#it{{c}}^{{2}}"
    str_sig_promptEnhanced = f'#it{{S}} = {signal_promptEnhanced:.0f} #pm {err_signal_promptEnhanced:.0f}'
    #str_sig_FDEnhanced = f'#it{{S}} = {signal_FDEnhanced:.0f} #pm {err_signal_FDEnhanced:.0f}'

    if particle == D0:
        legend = TLegend(0.6, 0.54, 0.87, 0.75)
    else:
        legend = TLegend(0.62, 0.58, 0.85, 0.72)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(43)
    legend.SetTextSize(SIZE_TEXT_LEGEND)
    legend.AddEntry(fit_tot_promptEnhanced, 'Total fit function', 'l')
    legend.AddEntry(fit_bkg_promptEnhanced, '#splitline{Combinatorial}{background}', 'l')
    if particle == D0:
        legend.AddEntry(fit_refl_promptEnhanced, 'K#minus#pi reflected', 'l')

    c = TCanvas("c", "", WIDTH, HEIGHT)
    # Create the first pad
    pad1 = TPad("promptEnhanced", "Prompt Enhanced", 0., 0., 1., 1.)
    if not pad1:
        raise RuntimeError("Failed to create pad1")
    pad1.Draw()
    pad1.cd()  # Switch to pad1
    frame_promptEnhanced = pad1.DrawFrame(mass_mins[i_pt], ymin_promptEnhanced, mass_maxs[i_pt], ymax_promptEnhanced, title)
    frame_promptEnhanced.GetYaxis().SetDecimals()

    #c.cd()
    # Create the second pad
    #pad2 = TPad("NonPromptEnhanced", "Non-prompt enhanced", 0.5, 0., 1., 1.)
    #if not pad2:
    #    raise RuntimeError("Failed to create pad2")
    #pad2.Draw()
    #pad2.cd()  # Switch to pad2
    #frame_FDEnhanced = pad2.DrawFrame(mass_mins[i_pt], ymin_FDEnhanced, mass_maxs[i_pt], ymax_FDEnhanced, title)
    #frame_FDEnhanced.GetYaxis().SetDecimals()



    #c.cd()
    #pad1.cd()
    set_object_style(hmass_promptEnhanced, linewidth=3, linecolor=kBlack, markersize=0.5)
    set_object_style(fit_tot_promptEnhanced, linewidth=3, linecolor=kBlue)
    set_object_style(fit_bkg_promptEnhanced, linewidth=3, linecolor=kRed, linestyle=2)
    #set_object_style(fit_refl_promptEnhanced, linewidth=3, linecolor=kGreen+2, linestyle=9)
    
    hmass_promptEnhanced.Draw("sameE")
    fit_bkg_promptEnhanced.Draw("same")
    fit_tot_promptEnhanced.Draw("same")
    #fit_refl_promptEnhanced.Draw("same")

    lat_alice.DrawLatex(0.19, 0.85, 'ALICE Preliminary')
    lat_label.SetTextSize(SIZE_TEXT_LAT_LABEL_FOR_COLL_SYSTEM)
    lat_label.DrawLatex(0.19, 0.79, 'pp,#kern[-0.08]{ #sqrt{#it{s}} = 13.6 TeV,}#kern[-0.08]{ #it{L}_{int} = 5 pb^{#minus1}}')
    lat_label.SetTextSize(SIZE_TEXT_LAT_LABEL)
    #draw_info(lat_label, particle)
    lat_label.DrawLatex(0.19, 0.73, f'{pt_mins[i_pt]:.0f} < #it{{p}}_{{T}} < {pt_maxs[i_pt]:.0f} GeV/#it{{c}}')
    #lat_label.DrawLatex(0.19, 0.3, 'Prompt enhanced')
    #lat_label.DrawLatex(0.7, 0.85, '|#it{y}| < 0.5')
    #fnonprompt_promptEnhanced = "#it{f}_{ non-prompt}^{ raw} = 0.246 #pm 0.007 (stat.)" # (4, 5) GeV
    #fnonprompt_promptEnhanced = "#it{f}_{ non-prompt}^{ raw} = 0.30 #pm 0.02 (stat.)" # (0, 1) GeV
    #lat_label.DrawLatex(0.19, 0.18, fnonprompt_promptEnhanced)

    # lat_label.DrawLatex(0.19, 0.64, str_mu)
    # lat_label.DrawLatex(0.19, 0.58, str_sigma)
    #lat_label.DrawLatex(0.19, 0.24, str_sig_promptEnhanced)
    if mult_latex[i_pt]:
        lat_label.DrawLatex(0.19, 0.24, mult_latex[i_pt])
    lat_label.DrawLatex(0.19, 0.18, "#Lambda_{c}^{#plus} #rightarrow pK^{#minus}#pi^{#plus} and charge conj.")
    #lat_label.DrawLatex(0.19, 0.16, "#it{L}_{int} = 5 pb^{-1}")

    legend.Draw()

    #c.cd()
    #pad2.cd()
    #set_object_style(hmass_FDEnhanced, linewidth=3, linecolor=kBlack)
    #set_object_style(fit_tot_FDEnhanced, linewidth=3, linecolor=kBlue)
    #set_object_style(fit_bkg_FDEnhanced, linewidth=3, linecolor=kRed, linestyle=2)
    #set_object_style(fit_refl_FDEnhanced, linewidth=3, linecolor=kGreen+2, linestyle=9)
    #hmass_FDEnhanced.Draw("same")
    #fit_bkg_FDEnhanced.Draw("same")
    #fit_tot_FDEnhanced.Draw("same")
    #fit_refl_FDEnhanced.Draw("same")

    #lat_alice.DrawLatex(0.19, 0.85, 'ALICE Preliminary')
    #lat_label.SetTextSize(SIZE_TEXT_LAT_LABEL_FOR_COLL_SYSTEM)
    #lat_label.DrawLatex(0.19, 0.79, 'pp, #sqrt{#it{s}} = 13.6 TeV')
    #lat_label.SetTextSize(SIZE_TEXT_LAT_LABEL)
    #draw_info(lat_label, particle)
    #lat_label.DrawLatex(0.19, 0.3, 'Non-prompt enhanced')
    #lat_label.DrawLatex(0.7, 0.85, '|#it{y}| < 0.5')
    #fnonprompt_FDEnhanced = "#it{f}_{ non-prompt}^{ raw} = 0.690 #pm 0.008 (stat.)" # (4, 5) GeV
    #fnonprompt_FDEnhanced = "#it{f}_{ non-prompt}^{ raw} = 0.70 #pm 0.02 (stat.)" # (0, 1) GeV
    #lat_label.DrawLatex(0.19, 0.18, fnonprompt_FDEnhanced)
    
    # lat_label.DrawLatex(0.19, 0.64, str_mu)
    # lat_label.DrawLatex(0.19, 0.58, str_sigma)
    #lat_label.DrawLatex(0.19, 0.24, str_sig_FDEnhanced)

    #legend.Draw()

    #c.Update()
    c.cd()

    save_canvas(c, particle, pt_mins, pt_maxs, i_pt, mult)

    if not batch:
        input("Press enter to exit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text", default="config.yml", help="config file name for ml")
    parser.add_argument("--batch", help="suppress video output", action="store_true")
    args = parser.parse_args()

    print("Loading analysis configuration: ...", end="\r")
    with open(args.config, "r", encoding="utf-8") as yml_cfg:
        configuration = yaml.load(yml_cfg, yaml.FullLoader)
    print("Loading analysis configuration: Done!")

    for i_pt in range(len(configuration["pp13.6TeVFD"]["PtMin"])):
        main(particle=LAMBDAC_TO_PKPI, i_pt=i_pt, cfg=configuration, batch=args.batch)
    # main(particle=DPLUS, i_pt=3, cfg=configuration, batch=args.batch)
