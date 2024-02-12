#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
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
file: hf_pt_spectrum.py
brief: script for computation of pT-differential yields (cross sections)
usage: python3 HfPtSpectrum.py CONFIG
author: Fabrizio Grosa <fabrizio.grosa@cern.ch>, CERN
"""

import sys
import numpy as np  # pylint: disable=import-error

from hf_analysis_utils import ( # pylint: disable=import-error
    compute_crosssection,
    compute_fraction_fc,
    compute_fraction_nb,
    get_hist_binlimits,
)
from ROOT import (  # pylint: disable=import-error,no-name-in-module
    TH1,
    TH1F,
    TCanvas,
    TFile,
    TGraphAsymmErrors,
    TStyle,
    gPad,
    gROOT,
    kAzure,
    kFullCircle,
)

def hf_pt_spectrum(channel, # pylint: disable=too-many-locals, too-many-arguments, too-many-statements, too-many-branches
                   b_ratio,
                   inputfonllpred,
                   frac_method,
                   eff_filename,
                   effprompt_histoname,
                   effnonprompt_histoname,
                   yield_filename,
                   yield_histoname,
                   norm,
                   sigmamb,
                   output_file):

    # final plots style settings
    style_hist = TStyle('style_hist','Histo graphics style')
    style_hist.SetOptStat("n")
    style_hist.SetMarkerColor(kAzure + 4)
    style_hist.SetMarkerStyle(kFullCircle)
    style_hist.SetMarkerSize(1)
    style_hist.SetHistLineColor(kAzure + 4)
    style_hist.SetHistLineWidth(2)
    style_hist.SetLabelSize(0.030)
    style_hist.SetLabelOffset(0.010)
    style_hist.SetTitleXOffset(1.3)
    style_hist.SetTitleYOffset(1.3)
    style_hist.SetDrawOption("AP")
    gROOT.SetStyle("style_hist")
    gROOT.ForceStyle()

    if channel not in [
        "D0toKpi",
        "DplustoKpipi",
        "DstoKpipi",
        "DstartoD0pi",
        "LctopKpi",
        "LctopK0S",
    ]:
        print(f"\033[91mERROR: channel {channel} not supported. Exit\033[0m")
        sys.exit(2)

    if frac_method not in ["Nb", "fc"]:
        print(
            f"\033[91mERROR: method to subtract nonprompt"
            f" {frac_method} not supported. Exit\033[0m"
        )
        sys.exit(5)

    fonll_hist_name = {
        "D0toKpi": "hD0Kpi",
        "DplustoKpipi": "hDpluskpipi",
        "DstoKKpi": "hDsPhipitoKkpi",
        "DstartoD0pi": "hDstarD0pi",
        "LctopKpi": "hLcpkpi",
        "LctopK0S": "hLcK0sp",
    }

    histos = {}

    histos["FONLL"] = {"prompt": {}, "nonprompt": {}}
    infile_fonll = TFile.Open(inputfonllpred)
    for pred in ("central", "min", "max"):
        histos["FONLL"]["nonprompt"][pred] = infile_fonll.Get(
            f"{fonll_hist_name[channel]}fromBpred_{pred}_corr"
        )
        histos["FONLL"]["nonprompt"][pred].SetDirectory(0)
        if frac_method == "fc":
            histos["FONLL"]["prompt"][pred] = infile_fonll.Get(
                f"{fonll_hist_name[channel]}pred_{pred}"
            )
            histos["FONLL"]["prompt"][pred].SetDirectory(0)

    infile_fonll.Close()

    infile_rawy = TFile.Open(yield_filename)
    histos["rawyields"] = infile_rawy.Get(yield_histoname)
    if not histos["rawyields"]:
        print(
            f"\033[91mERROR: raw-yield histo {yield_histoname}"
            f" not found in {yield_filename}. Exit\033[0m"
        )
        sys.exit(6)
    histos["rawyields"].SetDirectory(0)

    infile_rawy.Close()

    infile_eff = TFile.Open(eff_filename)
    histos["acceffp"] = infile_eff.Get(effprompt_histoname)
    if not histos["acceffp"]:
        print(
            f"\033[91mERROR: prompt (acc x eff) histo {effprompt_histoname}"
            f" not found in {eff_filename}. Exit\033[0m"
        )
        sys.exit(8)
    histos["acceffp"].SetDirectory(0)
    histos["acceffnp"] = infile_eff.Get(effnonprompt_histoname)
    if not histos["acceffnp"]:
        print(
            f"\033[91mERROR: nonprompt (acc x eff) histo {effprompt_histoname}"
            f"not found in {eff_filename}. Exit\033[0m"
        )
        sys.exit(9)
    histos["acceffnp"].SetDirectory(0)
    infile_eff.Close()

    # consistency check of bins
    ptlims = {}
    for histo in ["rawyields", "acceffp", "acceffnp"]:
        ptlims[histo] = get_hist_binlimits(histos[histo])
        if (
            histo != "rawyields"
            and not np.equal(ptlims[histo], ptlims["rawyields"]).all()
        ):
            print("\033[91mERROR: histo binning not consistent. Exit\033[0m")
            sys.exit(10)

    # compute cross section
    axistit_cross = "d#sigma/d#it{p}_{T} (pb GeV^{-1} #it{c})"
    axistit_cross_times_br = "d#sigma/d#it{p}_{T} #times BR (pb GeV^{-1} #it{c})"
    axistit_pt = "#it{p}_{T} (GeV/#it{c})"
    axistit_fprompt = "#if{f}_{prompt}"
    gfraction = TGraphAsymmErrors()
    gfraction.SetNameTitle("gfraction", f";{axistit_pt};{axistit_fprompt}")

    hptspectrum = TH1F(
        "hptspectrum",
        f";{axistit_pt};{axistit_cross}",
        len(ptlims["rawyields"]) - 1,
        ptlims["rawyields"],
    )
    hptspectrum_wo_br = TH1F(
        "hptspectrum_wo_br",
        f";{axistit_pt};{axistit_cross_times_br}",
        len(ptlims["rawyields"]) - 1,
        ptlims["rawyields"],
    )
    hnorm = TH1F(
        "hnorm",
        "hnorm",
        1,
        0,
        1
    )

    for i_pt, (ptmin, ptmax) in enumerate(
        zip(ptlims["rawyields"][:-1], ptlims["rawyields"][1:])
    ):
        pt_cent = (ptmax + ptmin) / 2
        pt_delta = ptmax - ptmin
        rawy = histos["rawyields"].GetBinContent(i_pt + 1)
        rawy_unc = histos["rawyields"].GetBinError(i_pt + 1)
        eff_times_acc_prompt = histos["acceffp"].GetBinContent(i_pt + 1)
        eff_times_acc_nonprompt = histos["acceffnp"].GetBinContent(i_pt + 1)
        ptmin_fonll = (
            histos["FONLL"]["nonprompt"]["central"].GetXaxis().FindBin(ptmin * 1.0001)
        )
        ptmax_fonll = (
            histos["FONLL"]["nonprompt"]["central"].GetXaxis().FindBin(ptmax * 0.9999)
        )
        crosssec_nonprompt_fonll = [
            histos["FONLL"]["nonprompt"][pred].Integral(
                ptmin_fonll, ptmax_fonll, "width"
            )
            / (ptmax - ptmin)
            for pred in histos["FONLL"]["nonprompt"]
        ]

        # compute prompt fraction
        if frac_method == "Nb":
            frac = compute_fraction_nb(  # BR already included in FONLL prediction
                rawy,
                eff_times_acc_prompt,
                eff_times_acc_nonprompt,
                crosssec_nonprompt_fonll,
                pt_delta,
                1.0,
                1.0,
                norm,
                sigmamb,
            )
        elif frac_method == "fc":
            crosssec_prompt_fonll = [
                histos["FONLL"]["prompt"][pred].Integral(
                    ptmin_fonll, ptmax_fonll, "width"
                )
                / (ptmax - ptmin)
                for pred in histos["FONLL"]["prompt"]
            ]
            frac, _ = compute_fraction_fc(
                eff_times_acc_prompt,
                eff_times_acc_nonprompt,
                crosssec_prompt_fonll,
                crosssec_nonprompt_fonll,
            )

        # compute cross section times BR
        crosssec, crosssec_unc = compute_crosssection(
            rawy,
            rawy_unc,
            frac[0],
            eff_times_acc_prompt,
            ptmax - ptmin,
            1.0,
            sigmamb,
            norm,
            1.0,
            frac_method,
        )

        hptspectrum.SetBinContent(i_pt + 1, crosssec / b_ratio)
        hptspectrum.SetBinError(i_pt + 1, crosssec_unc / b_ratio)
        hptspectrum_wo_br.SetBinContent(i_pt + 1, crosssec)
        hptspectrum_wo_br.SetBinError(i_pt + 1, crosssec_unc)
        hnorm.SetBinContent(1, norm)
        gfraction.SetPoint(i_pt, pt_cent, frac[0])
        #gfraction.SetPointError(
        #    i_pt, pt_delta / 2, pt_delta / 2, frac[0] - frac[1], frac[2] - frac[0]
        #)

    c = TCanvas("c", "c", 600, 800)
    c.Divide (1, 2)
    c.cd(1)
    gPad.SetLogy(True)
    hptspectrum.Draw()
    c.cd(2)
    gPad.SetLogy(True)
    hptspectrum_wo_br.Draw()
    output_pdf = output_file.replace("root", "pdf")
    c.Print(output_pdf)

    output_file = TFile.Open(output_file, "recreate")

    hptspectrum.Write()
    hptspectrum_wo_br.Write()
    hnorm.Write()
    #gfraction.Write()

    for _, value in histos.items():
        if isinstance(value, TH1):
            value.Write()
        #else:
        #    for flav in histos[hist]:
        #        for pred in histos[hist][flav]:
        #            histos[hist][flav][pred].Write()
    output_file.Close()
