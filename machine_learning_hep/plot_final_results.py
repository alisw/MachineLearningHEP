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
from array import array
from math import sqrt
# pylint: disable=import-error, no-name-in-module, unused-import
import yaml
from ROOT import TFile, gStyle, gROOT, kBlack, kRed, kWhite, TH1F
from machine_learning_hep.utilities import plot_histograms

FILES_NOT_FOUND = []

def make_syst_bin_edges(histo_central):
    # First find minimum bin width and make it a bit smaller even for the overlayed systematics
    syst_width = min([histo_central.GetBinWidth(b+1) for b in range(histo_central.GetNbinsX())])
    syst_width = 0.5 * syst_width

    bin_edges = []
    # Since we are multiplying the number of bins by 3 keep track which of these bins corresponds
    # to the central binning
    match_nominal_bins = []
    # Start at 2 and add 3 all the time in the loop below
    initial_bin_match = 2
    axis = histo_central.GetXaxis()
    # This basically now splits each bin in 3
    for b in range(axis.GetNbins()):
        match_nominal_bins.append(initial_bin_match)
        bin_edges.append(axis.GetBinLowEdge(b+1))
        bin_edges.append(axis.GetBinCenter(b+1) - syst_width / 2.)
        bin_edges.append(axis.GetBinCenter(b+1) + syst_width / 2.)
        initial_bin_match += 3
    # Last bin edge has to be added by hand
    bin_edges.append(axis.GetBinUpEdge(axis.GetNbins()))

    return match_nominal_bins, bin_edges


def results(histo_central, systematics_rel, title, legend_title, x_label, y_label,
            save_path):

    match_nominal_bins, bin_edges = make_syst_bin_edges(histo_central)
    bin_edges = array("d", bin_edges)

    histo_syst = TH1F("syst", "", len(bin_edges) - 1, bin_edges)
    systematics_rel_squ = [0] * histo_central.GetNbinsX()
    for systs in systematics_rel:
        for i, syst in enumerate(systs):
            systematics_rel_squ[i] += (syst * syst)
    for bc, bs, syst_squ in zip(range(histo_central.GetNbinsX()), match_nominal_bins,
                                systematics_rel_squ):
        syst = sqrt(syst_squ) * histo_central.GetBinContent(bc+1)
        histo_syst.SetBinContent(bs, histo_central.GetBinContent(bc+1))
        histo_syst.SetBinError(bs, syst)

    plot_histograms([histo_central, histo_syst], True, False, [legend_title, "syst"], title,
                    x_label, y_label, "", save_path, linesytles=[1], markerstyles=[20, 1],
                    colors=[kBlack, kRed], linewidths=[1, 1], draw_options=["", "E2"],
                    fillstyles=[0, 0], fillcolors=[kWhite, kRed])

def get_param(case):
    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    return data_param

def extract_histo(case, ana_type, mult_bin, period_number, histo_name):
    data_param = get_param(case)
    filepath = data_param[case]["analysis"][ana_type]["data"]["resultsallp"]
    if period_number < 0:
        period_number = -1
        filepath = data_param[case]["analysis"][ana_type]["data"]["results"][period_number]

    path = f"{filepath}/finalcross{case}{ana_type}mult{mult_bin}.root"
    in_file = TFile.Open(path, "READ")
    histo = in_file.Get(histo_name)
    histo.SetDirectory(0)
    return histo

def make_standard_save_path(case, ana_type, mult_bin, period_number, prefix):
    data_param = get_param(case)
    folder_plots = data_param[case]["analysis"]["dir_general_plots"]
    folder_plots = f"{folder_plots}/final/{ana_type}"
    if not os.path.exists(folder_plots):
        print("creating folder ", folder_plots)
        os.makedirs(folder_plots)
    return f"{folder_plots}/{prefix}_mult_{mult_bin}_period_{period_number}.eps"

#############################################################
gROOT.SetBatch(True)

CASE = "Dspp"
ANA = "MBvspt_ntrkl"
MULT_BIN = 0
YEAR_NUMBER = -1 # -1 refers to all years merged

# Get the ML histogram of the particle case and analysis type
# Everything available in the HFPtSpetrum can be requested
HISTO = extract_histo(CASE, ANA, MULT_BIN, YEAR_NUMBER, "histoSigmaCorr")
SAVE_PATH = make_standard_save_path(CASE, ANA, MULT_BIN, YEAR_NUMBER, "histoSigmaCorr")

REL_SYST = [[0.05, 0.05, 0.05, 0.05, 0.05], [0.08, 0.08, 0.08, 0.08, 0.08]]

results(HISTO, REL_SYST, f"Corrected cross section", "pp @ #sqrt{s} = 13 TeV",
        "#it{p}_{T} (GeV/#it{c})", "#sigma (D_{s}) #times BR(D_{s} #rightarrow KK#pi)", SAVE_PATH)
