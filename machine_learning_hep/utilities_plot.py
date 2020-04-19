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
Script containing all helper functions related to plotting with ROOT

Script also contains the "class Errors", used for systematic uncertainties (to
replace AliHFSystErr from AliPhysics).
"""
# pylint: disable=too-many-lines
from array import array
import math
import numpy as np
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
# pylint: disable=import-error, no-name-in-module
from ROOT import TH1F, TH2F, TFile, TH1, TH3F, TGraphAsymmErrors
from ROOT import TPad, TCanvas, TLegend, kBlack, kGreen, kRed, kBlue, kWhite
from ROOT import gStyle, gROOT
from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict
from machine_learning_hep.logger import get_logger

def buildarray(listnumber):
    """
    Build an array out of a list, useful for histogram binning
    """
    arraynumber = array('d', listnumber)
    return arraynumber

def buildbinning(nbinsx, xlow, xup):
    """
    Build a list for binning out of bin limits and number of bins
    """
    listnumber = [xlow + (xup - xlow) / nbinsx * i for i in range(nbinsx + 1)]
    return buildarray(listnumber)

def buildhisto(h_name, h_tit, arrayx, arrayy=None, arrayz=None):
    """
    Create a histogram of size 1D, 2D, 3D, depending on the number of arguments given
    """
    histo = None
    def binning(binning_array):
        return len(binning_array) - 1, binning_array
    if arrayz:
        histo = TH3F(h_name, h_tit, *binning(arrayx), *binning(arrayy), *binning(arrayz))
    elif arrayy:
        histo = TH2F(h_name, h_tit, *binning(arrayx), *binning(arrayy))
    else:
        histo = TH1F(h_name, h_tit, *binning(arrayx))
    histo.Sumw2()
    return histo

def makefill1dhist(df_, h_name, h_tit, arrayx, nvar1):
    """
    Create a TH1F histogram and fill it with one variables from a dataframe.
    """
    histo = buildhisto(h_name, h_tit, arrayx)
    fill_hist(histo, df_[nvar1])
    return histo

def build2dhisto(titlehist, arrayx, arrayy):
    """
    Create a TH2 histogram from two axis arrays.
    """
    return buildhisto(titlehist, titlehist, arrayx, arrayy)

def makefill2dhist(df_, titlehist, arrayx, arrayy, nvar1, nvar2):
    """
    Create a TH2F histogram and fill it with two variables from a dataframe.
    """
    histo = build2dhisto(titlehist, arrayx, arrayy)
    df_rd = df_[[nvar1, nvar2]]
    arr2 = df_rd.to_numpy()
    fill_hist(histo, arr2)
    return histo

def makefill3dhist(df_, titlehist, arrayx, arrayy, arrayz, nvar1, nvar2, nvar3):
    """
    Create a TH3F histogram and fill it with three variables from a dataframe.
    """

    histo = buildhisto(titlehist, titlehist, arrayx, arrayy, arrayz)
    #df_rd = df_[[nvar1, nvar2, nvar3]]
    #arr3 = df_rd.to_numpy()
    #fill_hist(histo, arr3) # this does not work, gives an empty histogram
    for row in df_.itertuples():
        histo.Fill(getattr(row, nvar1), getattr(row, nvar2), getattr(row, nvar3))
    return histo

def fill2dhist(df_, histo, nvar1, nvar2):
    """
    Fill a TH2 histogram with two variables from a dataframe.
    """
    df_rd = df_[[nvar1, nvar2]]
    arr2 = df_rd.values
    fill_hist(histo, arr2)
    return histo


def rebin_histogram(src_histo, new_histo):
    """
    Rebins the content of the histogram src_histo into new_histo.
    If average is set to True, the bin width is considered when rebinning.
    """
    if "TH1" not in src_histo.ClassName() and "TH1" not in new_histo.ClassName():
        get_logger().fatal("So far, can only work with TH1")
    x_axis_new = new_histo.GetXaxis()
    x_axis_src = new_histo.GetXaxis()
    for i in range(1, x_axis_new.GetNbins() + 1):
        x_new = [x_axis_new.GetBinLowEdge(i),
                 x_axis_new.GetBinUpEdge(i),
                 x_axis_new.GetBinWidth(i),
                 x_axis_new.GetBinCenter(i)]
        width_src = []
        y_src = []
        ye_src = []
        for j in range(1, x_axis_src.GetNbins() + 1):
            x_src = [x_axis_src.GetBinLowEdge(j),
                     x_axis_src.GetBinUpEdge(j),
                     x_axis_src.GetBinWidth(j)]
            if x_src[1] <= x_new[0]:
                continue
            if x_src[0] >= x_new[1]:
                continue
            if x_src[0] < x_new[0]:
                get_logger().fatal("For bin %i, bin %i low edge is too low! [%f, %f] vs [%f, %f]", i, j, x_new[0], x_new[1], x_src[0], x_src[1])
            if x_src[1] > x_new[1]:
                get_logger().fatal("For bin %i, bin %i up edge is too high! [%f, %f] vs [%f, %f]", i, j, x_new[0], x_new[1], x_src[0], x_src[1])
            y_src.append(src_histo.GetBinContent(j))
            ye_src.append(src_histo.GetBinError(j))
            width_src.append(x_src[-1])
        if abs(sum(width_src) - x_new[2]) > 0.00001:
            get_logger().fatal("Width does not match")
        new_histo.SetBinContent(i, sum(y_src))
        new_histo.SetBinError(i, np.sqrt(sum(j**2 for j in ye_src)))
    return new_histo

def load_root_style_simple():
    """
    Set basic ROOT style for histograms
    """
    gStyle.SetOptStat(0)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)

def load_root_style():
    """
    Set more advanced ROOT style for histograms
    """
    gROOT.SetStyle("Plain")
    gStyle.SetOptStat(0)
    gStyle.SetPalette(0)
    gStyle.SetCanvasColor(0)
    gStyle.SetFrameFillColor(0)
    gStyle.SetTitleOffset(1.15, "y")
    gStyle.SetTitleFont(42, "xy")
    gStyle.SetLabelFont(42, "xy")
    gStyle.SetTitleSize(0.042, "xy")
    gStyle.SetLabelSize(0.035, "xy")
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

def scatterplotroot(dfevt, nvar1, nvar2, nbins1, min1, max1, nbins2, min2, max2):
    """
    Make TH2F scatterplot between two variables from dataframe
    """
    hmult1_mult2 = TH2F(nvar1 + nvar2, nvar1 + nvar2, nbins1, min1, max1, nbins2, min2, max2)
    dfevt_rd = dfevt[[nvar1, nvar2]]
    arr2 = dfevt_rd.values
    fill_hist(hmult1_mult2, arr2)
    return hmult1_mult2

def find_axes_limits(histos, use_log_y=False):
    """
    Finds common axes limits for list of histograms provided
    """
    # That might be considered to be a hack since it now only has a chance to work
    # reasonably well if there is at least one histogram.
    max_y = min([h.GetMinimum() for h in histos if isinstance(h, TH1)])
    min_y = min([h.GetMaximum() for h in histos if isinstance(h, TH1)])
    if not min_y > 0. and use_log_y:
        min_y = 10.e-9

    max_x = max([h.GetXaxis().GetXmax() for h in histos])
    min_x = max([h.GetXaxis().GetXmin() for h in histos])

    for h in histos:
        if not isinstance(h, TH1):
            # That might be considered to be a hack since it now only has a chance to work
            # reasonably well if there is at least one histogram.
            continue
        min_x = min(min_x, h.GetXaxis().GetXmin())
        max_x = max(max_x, h.GetXaxis().GetXmax())
        min_y_tmp = h.GetBinContent(h.GetMinimumBin())
        if min_y_tmp > 0. and use_log_y or not use_log_y:
            min_y = min(min_y, h.GetBinContent(h.GetMinimumBin()))
        max_y = max(max_y, h.GetBinContent(h.GetMaximumBin()))

    return min_x, max_x, min_y, max_y

def style_histograms(histos, linestyles=None, markerstyles=None, colors=None, linewidths=None,
                     fillstyles=None, fillcolors=None):
    """
    Loops over given line- and markerstyles as well as colors applying them to the given list
    of histograms. The list of histograms might be larger than the styles provided. In that case
    the styles start again
    """
    if linestyles is None:
        linestyles = [1, 1, 1, 1]
    if markerstyles is None:
        markerstyles = [2, 4, 5, 32]
    if colors is None:
        colors = [kBlack, kRed, kGreen + 2, kBlue]
    if linewidths is None:
        linewidths = [1]
    if fillstyles is None:
        fillstyles = [0]
    if fillcolors is None:
        fillcolors = [kWhite]

    for i, h in enumerate(histos):
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineStyle(linestyles[i % len(linestyles)])
        h.SetMarkerStyle(markerstyles[i % len(markerstyles)])
        h.SetMarkerColor(colors[i % len(colors)])
        h.SetLineWidth(linewidths[i % len(linewidths)])
        h.SetFillStyle(fillstyles[i % len(fillstyles)])
        h.SetFillColor(fillcolors[i % len(fillcolors)])
        h.GetXaxis().SetTitleSize(0.02)
        h.GetXaxis().SetTitleSize(0.02)
        h.GetYaxis().SetTitleSize(0.02)

def divide_all_by_first(histos):
    """
    Divides all histograms in the list by the first one in the list and returns the
    divided histograms in the same order
    """

    histos_ratio = []
    for h in histos:
        histos_ratio.append(h.Clone(f"{h.GetName()}_ratio"))
        histos_ratio[-1].Divide(histos[0])

    return histos_ratio

def divide_by_eachother(histos1, histos2, scale=None, rebin2=None):
    """
    Divides all histos1 by histos2 and returns the
    divided histograms in the same order
    """

    if len(histos1) != len(histos2):
        get_logger().fatal("Number of histograms mismatch, %i vs. %i", \
                            len(histos1), len(histos2))

    histos_ratio = []
    for i, _ in enumerate(histos1):

        if rebin2 is not None:
            rebin = array('d', rebin2)
            histos1[i] = histos1[i].Rebin(len(rebin2)-1, f"{histos1[i].GetName()}_rebin", rebin)
            histos2[i] = histos2[i].Rebin(len(rebin2)-1, f"{histos2[i].GetName()}_rebin", rebin)

        if scale is not None:
            histos1[i].Scale(1./scale[0])
            histos2[i].Scale(1./scale[1])

        histos_ratio.append(histos1[i].Clone(f"{histos1[i].GetName()}_ratio"))
        histos_ratio[-1].Divide(histos2[i])

    return histos_ratio

def divide_by_eachother_barlow(histos1, histos2, scale=None, rebin2=None):
    """
    Divides all histos1 by histos2 using Barlow for stat. unc. and returns the
    divided histograms in the same order
    """

    if len(histos1) != len(histos2):
        get_logger().fatal("Number of histograms mismatch, %i vs. %i", \
                            len(histos1), len(histos2))

    histos_ratio = []
    for i, _ in enumerate(histos1):

        if rebin2 is not None:
            rebin = array('d', rebin2)
            histos1[i] = histos1[i].Rebin(len(rebin2)-1, f"{histos1[i].GetName()}_rebin", rebin)
            histos2[i] = histos2[i].Rebin(len(rebin2)-1, f"{histos2[i].GetName()}_rebin", rebin)

        if scale is not None:
            histos1[i].Scale(1./scale[0])
            histos2[i].Scale(1./scale[1])

        stat1 = []
        stat2 = []
        for j in range(histos1[i].GetNbinsX()):
            stat1.append(histos1[i].GetBinError(j+1) / histos1[i].GetBinContent(j+1))
            stat2.append(histos2[i].GetBinError(j+1) / histos2[i].GetBinContent(j+1))

        histos_ratio.append(histos1[i].Clone(f"{histos1[i].GetName()}_ratio"))
        histos_ratio[-1].Divide(histos2[i])

        for j in range(histos_ratio[-1].GetNbinsX()):
            statunc = math.sqrt(abs(stat1[j] * stat1[j] - stat2[j] * stat2[j]))
            histos_ratio[-1].SetBinError(j+1, histos_ratio[-1].GetBinContent(j+1) * statunc)

    return histos_ratio

def divide_all_by_first_multovermb(histos):
    """
    Divides all histograms in the list by the first one in the list and returns the
    divided histograms in the same order
    """

    histos_ratio = []
    err = []
    for h in histos:
        histos_ratio.append(h.Clone(f"{h.GetName()}_ratio"))

        stat = []
        for j in range(h.GetNbinsX()):
            stat.append(h.GetBinError(j+1) / h.GetBinContent(j+1))
        err.append(stat)
        histos_ratio[-1].Divide(histos[0])

        for j in range(h.GetNbinsX()):
            statunc = math.sqrt(abs(err[-1][j] * err[-1][j] - err[0][j] * err[0][j]))
            histos_ratio[-1].SetBinError(j+1, histos_ratio[-1].GetBinContent(j+1) * statunc)

    return histos_ratio

def put_in_pad(pad, use_log_y, histos, title="", x_label="", y_label="", yrange=None, **kwargs):
    """
    Providing a TPad this plots all given histograms in that pad adjusting the X- and Y-ranges
    accordingly.
    """

    draw_options = kwargs.get("draw_options", None)

    min_x, max_x, min_y, max_y = find_axes_limits(histos, use_log_y)
    pad.SetLogy(use_log_y)
    pad.cd()
    scale_frame_y = (0.01, 100.) if use_log_y else (0.7, 1.2)
    if yrange is None:
        yrange = [min_y * scale_frame_y[0], max_y * scale_frame_y[1]]
    frame = pad.DrawFrame(min_x, yrange[0], max_x, yrange[1],
                          f"{title};{x_label};{y_label}")
    frame.GetYaxis().SetTitleOffset(1.2)
    pad.SetTicks()
    if draw_options is None:
        draw_options = ["" for _ in histos]
    for h, o in zip(histos, draw_options):
        h.Draw(f"same {o}")

#pylint: disable=too-many-statements
def plot_histograms(histos, use_log_y=False, ratio_=False, legend_titles=None, title="", x_label="",
                    y_label_up="", y_label_ratio="", save_path="./plot.eps", **kwargs):
    """
    Throws all given histograms into one canvas. If desired, a ratio plot will be added.
    """
    gStyle.SetOptStat(0)
    justratioplot = False
    yrange = None
    if isinstance(ratio_, list):
        ratio = ratio_[0]
        justratioplot = ratio_[1]
        yrange = ratio_[2]
    else:
        justratioplot = ratio_
        ratio = ratio_

    linestyles = kwargs.get("linestyles", None)
    markerstyles = kwargs.get("markerstyles", None)
    colors = kwargs.get("colors", None)
    draw_options = kwargs.get("draw_options", None)
    linewidths = kwargs.get("linewidths", None)
    fillstyles = kwargs.get("fillstyles", None)
    fillcolors = kwargs.get("fillcolors", None)
    canvas_name = kwargs.get("canvas_name", "Canvas")
    style_histograms(histos, linestyles, markerstyles, colors, linewidths, fillstyles, fillcolors)

    canvas = TCanvas('canvas', canvas_name, 800, 800)
    pad_up_start = 0.4 if ratio else 0.

    pad_up = TPad("pad_up", "", 0., pad_up_start, 1., 1.)
    if ratio:
        pad_up.SetBottomMargin(0.)
    pad_up.Draw()

    x_label_up_tmp = x_label if not ratio else ""
    put_in_pad(pad_up, use_log_y, histos, title, x_label_up_tmp, y_label_up,
               yrange, draw_options=draw_options)

    pad_up.cd()
    legend = None
    if legend_titles is not None:
        if justratioplot:
            legend = TLegend(.2, .65, .6, .85)
        else:
            legend = TLegend(.45, .65, .85, .85)
        legend.SetBorderSize(0)
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.02)
        for h, l in zip(histos, legend_titles):
            if l is not None:
                legend.AddEntry(h, l)
        legend.Draw()

    canvas.cd()
    pad_ratio = None
    histos_ratio = None

    if ratio and justratioplot is False:
        histos_ratio = divide_all_by_first(histos)
        pad_ratio = TPad("pad_ratio", "", 0., 0.05, 1., pad_up_start)
        pad_ratio.SetTopMargin(0.)
        pad_ratio.SetBottomMargin(0.3)
        pad_ratio.Draw()

        put_in_pad(pad_ratio, False, histos_ratio, "", x_label, y_label_ratio)

    canvas.SaveAs(save_path)

    index = save_path.rfind(".")

    # Save also everything into a ROOT file
    root_save_path = save_path[:index] + ".root"
    root_file = TFile.Open(root_save_path, "RECREATE")
    for h in histos:
        h.Write()
    canvas.Write()
    root_file.Close()

    canvas.Close()

def save_histograms(histos, save_path="./plot.root"):
    """
    Save everything into a ROOT file for offline plotting
    """
    index = save_path.rfind(".")

    # Save also everything into a ROOT file
    root_save_path = save_path[:index] + ".root"
    root_file = TFile.Open(root_save_path, "RECREATE")
    for h in histos:
        h.Write()
    root_file.Close()

# pylint: disable=too-many-branches
def calc_systematic_multovermb(errnum_list, errden_list, n_bins, justfd=-99):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Ds(mult) / Ds(MB). Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list.errors.values())[0]) or \
     n_bins != len(list(errden_list.errors.values())[0]):
        get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                            n_bins, len(list(errnum_list.errors.values())[0]), \
                            len(list(errden_list.errors.values())[0]))

    listimpl = ["yield", "cut", "pid", "feeddown_mult", "feeddown_mult_spectra", "trigger", \
                "multiplicity_interval", "multiplicity_weights", "track", "ptshape", \
                "feeddown_NB", "sigmav0", "branching_ratio"]

    j = 0
    for (_, errnum), (_, errden) in zip(errnum_list.errors.items(), errden_list.errors.items()):
        for i in range(n_bins):

            if errnum_list.names[j] not in listimpl:
                get_logger().fatal("Unknown systematic name: %s", errnum_list.names[j])
            if errnum_list.names[j] != errden_list.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum.names[j], errden.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list.names[j] == "yield" and justfd is not True:
                    #Partially correlated, take largest
                    tot_list[i][nb] += max(errnum[i][nb], errden[i][nb]) \
                                        * max(errnum[i][nb], errden[i][nb])
                elif errnum_list.names[j] == "cut" and justfd is not True:
                    #Partially correlated, take largest
                    tot_list[i][nb] += max(errnum[i][nb], errden[i][nb]) \
                                        * max(errnum[i][nb], errden[i][nb])
                elif errnum_list.names[j] == "pid" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "feeddown_mult" and justfd is not False:
                    #Assign directly from multiplicity case, no syst for MB
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "feeddown_mult_spectra" and justfd is not False:
                    #Ratio here, skip spectra syst
                    pass
                elif errnum_list.names[j] == "trigger" and justfd is not True:
                    #Assign directly from multiplicity case, no syst for MB
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "multiplicity_interval" and justfd is not True:
                    #FD: estimated using 7TeV strategy directly for ratio
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "multiplicity_weights" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "track" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "ptshape" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "feeddown_NB" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "sigmav0" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list.names[j] == "branching_ratio" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

# pylint: disable=too-many-branches
def calc_systematic_mesonratio(errnum_list, errden_list, n_bins, justfd=-99):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Ds(MB or mult) / D0(MB or mult).
    Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list.errors.values())[0]) or \
     n_bins != len(list(errden_list.errors.values())[0]):
        get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                            n_bins, len(list(errnum_list.errors.values())[0]), \
                            len(list(errden_list.errors.values())[0]))

    listimpl = ["yield", "cut", "pid", "feeddown_mult", "feeddown_mult_spectra", "trigger", \
                "multiplicity_interval", "multiplicity_weights", "track", "ptshape", \
                "feeddown_NB", "sigmav0", "branching_ratio"]

    j = 0
    for (_, errnum), (_, errden) in zip(errnum_list.errors.items(), errden_list.errors.items()):
        for i in range(n_bins):

            if errnum_list.names[j] not in listimpl:
                get_logger().fatal("Unknown systematic name: %s", errnum_list.names[j])
            if errnum_list.names[j] != errden_list.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum_list.names[j], errden_list.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list.names[j] == "yield" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "cut" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "pid" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "feeddown_mult_spectra" and justfd is not False:
                    #Fully correlated
                    ynum = errnum_list.errors["feeddown_NB"][i][4]
                    yden = errden_list.errors["feeddown_NB"][i][4]
                    #Relative uncertainties stored, make absolute
                    ynuml = ynum - ynum * errnum[i][2]
                    ydenl = yden - yden * errden[i][2]
                    ynumh = ynum + ynum * errnum[i][3]
                    ydenh = yden + yden * errden[i][3]
                    rat = [ynuml / ydenl, ynum / yden, ynumh / ydenh]
                    minsys = min(rat)
                    maxsys = max(rat)
                    if nb == 2:
                        tot_list[i][nb] += (rat[1] - minsys) * (rat[1] - minsys) / (rat[1] * rat[1])
                    if nb == 3:
                        tot_list[i][nb] += (maxsys - rat[1]) * (maxsys - rat[1]) / (rat[1] * rat[1])
                elif errnum_list.names[j] == "feeddown_mult" and justfd is not False:
                    #Spectra here, skip ratio systematic
                    pass
                elif errnum_list.names[j] == "trigger" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "feeddown_NB" and justfd is not False:
                    #Fully correlated under assumption central Fc value stays within Nb syst
                    ynum = errnum[i][4]
                    yden = errden[i][4]
                    #Absolute uncertainties stored
                    ynuml = ynum - errnum[i][2]
                    ydenl = yden - errden[i][2]
                    ynumh = ynum + errnum[i][3]
                    ydenh = yden + errden[i][3]
                    rat = [ynuml / ydenl, ynum / yden, ynumh / ydenh]
                    minsys = min(rat)
                    maxsys = max(rat)
                    if nb == 2:
                        tot_list[i][nb] += (rat[1] - minsys) * (rat[1] - minsys) / (rat[1] * rat[1])
                    if nb == 3:
                        tot_list[i][nb] += (maxsys - rat[1]) * (maxsys - rat[1]) / (rat[1] * rat[1])
                elif errnum_list.names[j] == "multiplicity_weights" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "track" and justfd is not True:
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "ptshape" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "multiplicity_interval" and justfd is not True:
                    #NB: Assuming ratio: 3prongs over 2prongs here! 2prong part cancels
                    #We use 1/3 of systematic of numerator
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] / 9
                elif errnum_list.names[j] == "sigmav0" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list.names[j] == "branching_ratio" and justfd is not True:
                    #Uncorrelated, but usually not plotted in boxes, so pass
                    pass
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

def calc_systematic_mesondoubleratio(errnum_list1, errnum_list2, errden_list1, \
                                     errden_list2, n_bins, dropbins=None, justfd=-99):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Lc/D0_mult-i / Lc/D0_mult-j.
    Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list1.errors.values())[0]) or \
     n_bins != len(list(errden_list1.errors.values())[0]):
        if dropbins is None:
            get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                                n_bins, len(list(errnum_list1.errors.values())[0]), \
                                len(list(errden_list1.errors.values())[0]))

    listimpl = ["yield", "cut", "pid", "feeddown_mult", "feeddown_mult_spectra", "trigger", \
                "multiplicity_interval", "multiplicity_weights", "track", "ptshape", \
                "feeddown_NB", "sigmav0", "branching_ratio"]

    j = 0
    for (_, errnum1), (_, errnum2), (_, errden1), (_, errden2) in zip(errnum_list1.errors.items(), \
                                                                      errnum_list2.errors.items(), \
                                                                      errden_list1.errors.items(), \
                                                                      errden_list2.errors.items()):
        for i in range(n_bins):

            inum = i
            iden = i
            if dropbins is not None:
                inum = dropbins[0][i]
                iden = dropbins[1][i]

            if errnum_list1.names[j] not in listimpl:
                get_logger().fatal("Unknown systematic name: %s", errnum_list1.names[j])
            if errnum_list1.names[j] != errden_list2.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum_list1.names[j], errden_list2.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list1.names[j] == "yield" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum1[inum][nb] * errnum1[inum][nb] + \
                                       errnum2[inum][nb] * errnum2[inum][nb] + \
                                       errden1[iden][nb] * errden1[iden][nb] + \
                                       errden2[iden][nb] * errden2[iden][nb]
                elif errnum_list1.names[j] == "cut" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum1[inum][nb] * errnum1[inum][nb] + \
                                       errnum2[inum][nb] * errnum2[inum][nb] + \
                                       errden1[iden][nb] * errden1[iden][nb] + \
                                       errden2[iden][nb] * errden2[iden][nb]
                elif errnum_list1.names[j] == "pid" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "feeddown_mult_spectra" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "feeddown_mult" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "trigger" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "feeddown_NB" and justfd is not False:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "multiplicity_weights" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "track" and justfd is not True:
                    #Correlated, do nothing
                    pass
                elif errnum_list1.names[j] == "ptshape" and justfd is not True:
                    #Uncorrelated
                    tot_list[i][nb] += errnum1[inum][nb] * errnum1[inum][nb] + \
                                       errnum2[inum][nb] * errnum2[inum][nb] + \
                                       errden1[iden][nb] * errden1[iden][nb] + \
                                       errden2[iden][nb] * errden2[iden][nb]
                elif errnum_list1.names[j] == "multiplicity_interval" and justfd is not True:
                    #NB: Assuming ratio: 3prongs over 2prongs here! 2prong part cancels
                    #We use 1/3 of systematic of numerator
                    tot_list[i][nb] += errden1[iden][nb] * errden1[iden][nb] / 9
                elif errnum_list1.names[j] == "sigmav0" and justfd is not True:
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list1.names[j] == "branching_ratio" and justfd is not True:
                    #Uncorrelated, but usually not plotted in boxes, so pass
                    pass
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

# pylint: disable=too-many-nested-blocks
class Errors:
    """
    Errors corresponding to one histogram
    Relative errors are assumed
    """
    def __init__(self, n_bins):
        # A dictionary of lists, lists will contain 4-tuples
        self.errors = {}
        # Number of errors per bin
        self.n_bins = n_bins
        # Names of systematic in order as they appear in self.errors
        self.names = {}
        # The logger...
        self.logger = get_logger()

    @staticmethod
    def make_symm_y_errors(*args):
        return [[0, 0, a, a] for a in args]

    @staticmethod
    def make_asymm_y_errors(*args):
        if len(args) % 2 != 0:
            get_logger().fatal("Need an even number ==> ((low, up) * n_central) of errors")
        return [[0, 0, args[i], args[i+1]] for i in range(0, len(args), 2)]


    @staticmethod
    def make_root_asymm(histo_central, error_list, **kwargs):
        """
        This takes a list of 4-tuples and a central histogram assumed to have number of bins
        corresponding to length of error_list
        """
        n_bins = histo_central.GetNbinsX()
        if n_bins != len(error_list):
            get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i",
                               n_bins, len(error_list))
        rel_x = kwargs.get("rel_x", True)
        rel_y = kwargs.get("rel_y", True)
        const_x_err = kwargs.get("const_x_err", None)
        const_y_err = kwargs.get("const_y_err", None)

        x_low = None
        x_up = None
        y_low = None
        y_up = None
        # Make x up and down
        if const_x_err is not None:
            x_up = array("d", [const_x_err] * n_bins)
            x_low = array("d", [const_x_err] * n_bins)
        elif rel_x is True:
            x_up = array("d", [err[1] * histo_central.GetBinCenter(b + 1) \
                    for b, err in enumerate(error_list)])
            x_low = array("d", [err[0] * histo_central.GetBinCenter(b + 1) \
                    for b, err in enumerate(error_list)])
        else:
            x_up = array("d", [err[1] for err in error_list])
            x_low = array("d", [err[0] for err in error_list])

        # Make y up and down
        if const_y_err is not None:
            y_up = array("d", [const_y_err] * n_bins)
            y_low = array("d", [const_y_err] * n_bins)
        elif rel_y is True:
            y_up = array("d", [err[3] * histo_central.GetBinContent(b + 1) \
                    for b, err in enumerate(error_list)])
            y_low = array("d", [err[2] * histo_central.GetBinContent(b + 1) \
                    for b, err in enumerate(error_list)])
        else:
            y_up = array("d", [err[3] for err in error_list])
            y_low = array("d", [err[2] for err in error_list])

        bin_centers = array("d", [histo_central.GetBinCenter(b + 1) for b in range(n_bins)])
        bin_contents = array("d", [histo_central.GetBinContent(b + 1) for b in range(n_bins)])

        return TGraphAsymmErrors(n_bins, bin_centers, bin_contents, x_low, x_up, y_low, y_up)

    @staticmethod
    def make_root_asymm_dummy(histo_central):
        n_bins = histo_central.GetNbinsX()
        bin_centers = array("d", [histo_central.GetBinCenter(b + 1) for b in range(n_bins)])
        bin_contents = array("d", [histo_central.GetBinContent(b + 1) for b in range(n_bins)])
        y_up = array("d", [0.] * n_bins)
        y_low = array("d", [0.] * n_bins)
        x_up = array("d", [0.] * n_bins)
        x_low = array("d", [0.] * n_bins)

        return TGraphAsymmErrors(n_bins, bin_centers, bin_contents, x_low, x_up, y_low, y_up)

    def add_errors(self, name, err_list):
        """
        err_list assumed to be a list of 4-tuples
        """
        if name in self.errors:
            self.logger.fatal("Error %s already registered", name)
        if len(err_list) != self.n_bins:
            self.logger.fatal("%i errors required, you want to push %i", self.n_bins, len(err_list))

        self.errors[name] = err_list.copy()

    def read(self, yaml_errors, extra_errors=None):
        """
        Read everything from YAML
        """
        error_dict = parse_yaml(yaml_errors)
        for name, errors in error_dict.items():
            if name == "names":
                self.names = errors.copy()
            else:
                self.add_errors(name, errors)
        if extra_errors is not None:
            self.errors.update(extra_errors)
            for key in extra_errors:
                self.names.append(key)

    def write(self, yaml_path):
        """
        Write everything from YAML
        """
        dump_yaml_from_dict(self.errors, yaml_path)

    def define_correlations(self):
        """
        Not yet defined
        """
        self.logger.warning("Function \"define_correlations\' not yet defined")

    def divide(self):
        """
        Not yet defined
        """
        self.logger.warning("Function \"divide\" not yet defined")

    def get_total(self):
        """
        Returns a list of total errors
        For now only add in quadrature and take sqrt
        """
        tot_list = [[0., 0., 0., 0.] for _ in range(self.n_bins)]
        for _, errors in enumerate(self.errors.values()):
            for i in range(self.n_bins):
                for nb in range(len(tot_list[i])):
                    tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
        tot_list = np.sqrt(tot_list)
        return tot_list

    def get_total_for_spectra_plot(self, justfd=-99):
        """
        Returns a list of total errors
        For now only add in quadrature and take sqrt
        """
        tot_list = [[0., 0., 0., 0.] for _ in range(self.n_bins)]
        for j, errors in enumerate(self.errors.values()):
            for i in range(self.n_bins):
                for nb in range(len(tot_list[i])):
                    if self.names[j] != "branching_ratio" and self.names[j] != "sigmav0" \
                      and self.names[j] != "feeddown_mult":

                        if justfd == -99:
                            tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
                        elif justfd is True:
                            if self.names[j] == "feeddown_NB" \
                              or self.names[j] == "feeddown_mult_spectra":
                                tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
                        elif justfd is False:
                            if self.names[j] != "feeddown_NB" \
                              and self.names[j] != "feeddown_mult_spectra":
                                tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
                        else:
                            get_logger().fatal("Option for spectra systematic not valid")

        tot_list = np.sqrt(tot_list)
        return tot_list
