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
main script for doing data processing, machine learning and analysis
"""
from array import array
import multiprocessing as mp
from datetime import datetime
import pickle
import bz2
import gzip
import lzma
import os
import math
import numpy as np
import pandas as pd
import lz4
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
# pylint: disable=import-error, no-name-in-module
from ROOT import TH1F, TH2F, TFile, TH1, TGraphAsymmErrors
from ROOT import TPad, TCanvas, TLegend, kBlack, kGreen, kRed, kBlue, kWhite
from ROOT import Double
from machine_learning_hep.selectionutils import select_runs
from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict
from machine_learning_hep.logger import get_logger
def openfile(filename, attr):
    if filename.lower().endswith('.bz2'):
        return bz2.BZ2File(filename, attr)
    if filename.lower().endswith('.xz'):
        return lzma.open(filename, attr)
    if filename.lower().endswith('.gz'):
        return gzip.open(filename, attr)
    if filename.lower().endswith('.lz4'):
        return lz4.frame.open(filename, attr)
    return open(filename, attr)

def selectdfquery(dfr, selection):
    if selection is not None:
        dfr = dfr.query(selection)
    return dfr

def selectdfrunlist(dfr, runlist, runvar):
    if runlist is not None:
        isgoodrun = select_runs(runlist, dfr[runvar].values)
        dfr = dfr[np.array(isgoodrun, dtype=bool)]
    return dfr

def merge_method(listfiles, namemerged):
    dflist = []
    for myfilename in listfiles:
        myfile = openfile(myfilename, "rb")
        df = pickle.load(myfile)
        dflist.append(df)
    dftot = pd.concat(dflist)
    pickle.dump(dftot, openfile(namemerged, "wb"), protocol=4)

# pylint: disable=too-many-nested-blocks
def list_folders(main_dir, filenameinput, maxfiles):
    if not os.path.isdir(main_dir):
        print("the input directory =", main_dir, "doesnt exist")
    list_subdir0 = os.listdir(main_dir)
    listfolders = list()
    for subdir0 in list_subdir0:
        subdir0full = os.path.join(main_dir, subdir0)
        if os.path.isdir(subdir0full):
            list_subdir1 = os.listdir(subdir0full)
            for subdir1 in list_subdir1:
                subdir1full = os.path.join(subdir0full, subdir1)
                if os.path.isdir(subdir1full):
                    list_files_ = os.listdir(subdir1full)
                    for myfile in list_files_:
                        filefull = os.path.join(subdir1full, myfile)
                        if os.path.isfile(filefull) and \
                        myfile == filenameinput:
                            listfolders.append(os.path.join(subdir0, subdir1))
    if maxfiles is not -1:
        listfolders = listfolders[:maxfiles]
    return  listfolders

def create_folder_struc(maindir, listpath):
    for path in listpath:
        path = path.split("/")

        folder = os.path.join(maindir, path[0])
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(folder, path[1])
        if not os.path.exists(folder):
            os.makedirs(folder)

def checkdirlist(dirlist):
    exfolders = 0
    for _, mydir in enumerate(dirlist):
        if os.path.exists(mydir):
            print("rm -rf ", mydir)
            exfolders = exfolders - 1
    return exfolders

def checkdir(mydir):
    exfolders = 0
    if os.path.exists(mydir):
        print("rm -rf ", mydir)
        exfolders = -1
    return exfolders

def checkmakedirlist(dirlist):
    for _, mydir in enumerate(dirlist):
        print("creating folder ", mydir)
        os.makedirs(mydir)

def checkmakedir(mydir):
    print("creating folder ", mydir)
    os.makedirs(mydir)

def appendfiletolist(mylist, namefile):
    return [os.path.join(path, namefile) for path in mylist]

def appendmainfoldertolist(prefolder, mylist):
    return [os.path.join(prefolder, path) for path in mylist]

def createlist(prefolder, mylistfolder, namefile):
    listfiles = appendfiletolist(mylistfolder, namefile)
    listfiles = appendmainfoldertolist(prefolder, listfiles)
    return listfiles

def seldf_singlevar(dataframe, var, minval, maxval):
    dataframe = dataframe.loc[(dataframe[var] >= minval) & (dataframe[var] < maxval)]
    return dataframe

def split_df_sigbkg(dataframe_, var_signal_):
    dataframe_sig_ = dataframe_.loc[dataframe_[var_signal_] == 1]
    dataframe_bkg_ = dataframe_.loc[dataframe_[var_signal_] == 0]
    return dataframe_sig_, dataframe_bkg_

def createstringselection(var, low, high):
    string_selection = "dfselection_"+(("%s_%.1f_%.1f") % (var, low, high))
    return string_selection

def mergerootfiles(listfiles, mergedfile, tmp_dir):
    def divide_chunks(list_to_split, chunk_size):
        for i in range(0, len(list_to_split), chunk_size):
            yield list_to_split[i:i + chunk_size]

    tmp_files = []
    if len(listfiles) > 500:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        for i, split_list in enumerate(divide_chunks(listfiles, 500)):
            tmp_files.append(os.path.join(tmp_dir, f"hadd_tmp_merged{i}.root"))
            outstring = " ".join(split_list)
            os.system("hadd -f -j 30 %s  %s " % (tmp_files[-1], outstring))
    else:
        tmp_files = listfiles

    outstring = " ".join(tmp_files)
    os.system("hadd -f -j 30 %s  %s " % (mergedfile, outstring))

def createhisto(stringname, nbins, rmin, rmax):
    hden = TH1F("hden" + stringname, "hden" + stringname, nbins, rmin, rmax)
    hnum = TH1F("hnum" + stringname, "hnum" + stringname, nbins, rmin, rmax)
    hnum.Sumw2()
    hden.Sumw2()
    return hden, hnum

def makeff(dfevt, selnum, selden, stringname, nbins, rmin, rmax, variable):
    #loadstyle()
    hden, hnum = createhisto(stringname, nbins, rmin, rmax)
    dnum = dfevt
    dden = dfevt
    if selnum is not None:
        dnum = dfevt.query(selnum)
    if selden is not None:
        dden = dfevt.query(selden)
    fill_hist(hden, dden[variable])
    fill_hist(hnum, dnum[variable])
    return hden, hnum

def scatterplot(dfevt, nvar1, nvar2, nbins1, min1, max1, nbins2, min2, max2):
    hmult1_mult2 = TH2F(nvar1 + nvar2, nvar1 + nvar2, nbins1, min1, max1, nbins2, min2, max2)
    dfevt_rd = dfevt[[nvar1, nvar2]]
    arr2 = dfevt_rd.values
    fill_hist(hmult1_mult2, arr2)
    return hmult1_mult2

def z_calc(pt_1, phi_1, eta_1, pt_2, phi_2, eta_2):
    np_pt_1 = pt_1.values
    np_pt_2 = pt_2.values
    np_phi_1 = phi_1.values
    np_phi_2 = phi_2.values
    np_eta_1 = eta_1.values
    np_eta_2 = eta_2.values

    cos_phi_1 = np.cos(np_phi_1)
    cos_phi_2 =	np.cos(np_phi_2)
    sin_phi_1 =	np.sin(np_phi_1)
    sin_phi_2 = np.sin(np_phi_2)
    sinh_eta_1 = np.sinh(np_eta_1)
    sinh_eta_2 = np.sinh(np_eta_2)

    px_1 = np_pt_1*cos_phi_1
    px_2 = np_pt_2*cos_phi_2
    py_1 = np_pt_1*sin_phi_1
    py_2 = np_pt_2*sin_phi_2
    pz_1 = np_pt_1*sinh_eta_1
    pz_2 = np_pt_2*sinh_eta_2
    numerator = px_1*px_2+py_1*py_2+pz_1*pz_2
    denominator = px_1*px_1+py_1*py_1+pz_1*pz_1
    return numerator/denominator

def z_gen_calc(pt_1, phi_1, eta_1, pt_2, delta_phi, delta_eta):
    phi_2 = phi_1 + delta_phi
    eta_2 = eta_1 - delta_eta
    return z_calc(pt_1, phi_1, eta_1, pt_2, phi_2, eta_2)

def get_bins(axis):
    return np.array([axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins() + 2)])

def folding(h_input, response_matrix, h_output):
    h_folded = h_output.Clone("h_folded")
    for a in range(h_output.GetNbinsX()):
        for b in range(h_output.GetNbinsY()):
            val = 0.
            val_err = 0.
            for k in range(h_input.GetNbinsX()):
                for l in range(h_input.GetNbinsY()):
                    index_x_out = a + h_output.GetNbinsX() * b
                    index_x_in = k + h_input.GetNbinsX() * l
                    val += h_input.GetBinContent(k + 1, l + 1) * \
                        response_matrix(index_x_out, index_x_in)
                    val_err += h_input.GetBinError(k + 1, l + 1) * \
                        h_input.GetBinError(k + 1, l + 1) * \
                        response_matrix(index_x_out, index_x_in) * \
                        response_matrix(index_x_out, index_x_in)
            h_folded.SetBinContent(a + 1, b + 1, val)
            h_folded.SetBinError(a + 1, b + 1, math.sqrt(val_err))
    return h_folded

# Plotting stuff

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
            rebin = array('d',rebin2)
            histos2[i] = histos2[i].Rebin(len(rebin2)-1, f"{histos2[i].GetName()}_rebin", rebin)

        if scale is not None:
            histos1[i].Scale(1./scale[0])
            histos2[i].Scale(1./scale[1])

        histos_ratio.append(histos1[i].Clone(f"{histos1[i].GetName()}_ratio"))
        histos_ratio[-1].Divide(histos2[i])

    return histos_ratio


def divide_all_by_first_multovermb(histos):
    """
    Divides all histograms in the list by the first one in the list and returns the
    divided histograms in the same order
    """

    for h in histos:
        h.Scale(1./h.Integral())

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

# pylint: disable=too-many-branches
def calc_systematic_multovermb(errnum_list, errden_list, n_bins):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Ds(mult) / Ds(MB). Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list.errors.values())[0]) or \
     n_bins != len(list(errden_list.errors.values())[0]):
        get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                            n_bins, len(errnum_list.errors[0]), len(errden_list.errors[0]))

    j = 0
    for (_, errnum), (_, errden) in zip(errnum_list.errors.items(), errden_list.errors.items()):
        for i in range(n_bins):

            if errnum_list.names[j] != errden_list.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum.names[j], errden.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list.names[j] == "yield":
                    #Partially correlated, take largest
                    tot_list[i][nb] += max(errnum[i][nb], errden[i][nb]) \
                                        * max(errnum[i][nb], errden[i][nb])
                elif errnum_list.names[j] == "cut":
                    #Partially correlated, take largest
                    tot_list[i][nb] += max(errnum[i][nb], errden[i][nb]) \
                                        * max(errnum[i][nb], errden[i][nb])
                elif errnum_list.names[j] == "feeddown_mult":
                    #Assign directly from multiplicity case, no syst for MB
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "trigger":
                    #Assign directly from multiplicity case, no syst for MB
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "multiplicity_interval":
                    #FD: estimated using 7TeV strategy directly for ratio
                    #NB: At one point the strategy for spectra and Ds(mult)/Ds(MB) will change,
                    #    then there should be two keys
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb]
                elif errnum_list.names[j] == "multiplicity_weights":
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "track":
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "ptshape":
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "feeddown_NB":
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "sigmav0":
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list.names[j] == "branching_ratio":
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                else:
                    get_logger().fatal("Unknown systematic name: %s", errnum_list.names[j])
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

# pylint: disable=too-many-branches
def calc_systematic_mesonratio(errnum_list, errden_list, n_bins):
    """
    Returns a list of total errors taking into account the defined correlations
    Propagation uncertainties defined for Ds(MB or mult) / D0(MB or mult).
    Check if applicable to your situation
    """
    tot_list = [[0., 0., 0., 0.] for _ in range(n_bins)]
    if n_bins != len(list(errnum_list.errors.values())[0]) or \
     n_bins != len(list(errden_list.errors.values())[0]):
        get_logger().fatal("Number of bins and number of errors mismatch, %i vs. %i vs. %i", \
                            n_bins, len(errnum_list.errors[0]), len(errden_list.errors[0]))

    j = 0
    for (_, errnum), (_, errden) in zip(errnum_list.errors.items(), errden_list.errors.items()):
        for i in range(n_bins):

            if errnum_list.names[j] != errden_list.names[j]:
                get_logger().fatal("Names not in same order: %s vs %s", \
                                   errnum_list.names[j], errden_list.names[j])

            for nb in range(len(tot_list[i])):
                if errnum_list.names[j] == "yield":
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "cut":
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "feeddown_mult":
                    #Correlated, WHAT DO WE DO?? Now assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                    print("Warning: strategy for feeddown_mult to be refined")
                elif errnum_list.names[j] == "trigger":
                    #Correlated, do nothing
                    pass
                elif errnum_list.names[j] == "feeddown_NB":
                    #Fully correlated under assumption central Fc value stays within Nb syst
                    ynum = errnum[i][4]
                    yden = errden[i][4]
                    ynuml = ynum - errnum[i][2]
                    ydenl = yden - errden[i][2]
                    ynumh = ynum + errnum[i][3]
                    ydenh = yden + errden[i][3]
                    ratio = [ynuml / ydenl, ynum / yden, ynumh / ydenh]
                    minsyst = min(ratio)
                    maxsyst = max(ratio)
                    if nb == 2:
                        tot_list[i][nb] += (ratio[1] - minsyst) * (ratio[1] - minsyst)
                    if nb == 3:
                        tot_list[i][nb] += (maxsyst - ratio[1]) * (maxsyst - ratio[1])
                elif errnum_list.names[j] == "multiplicity_weights":
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "track":
                    #Correlated, assign difference
                    diff = abs(errnum[i][nb] - errden[i][nb])
                    tot_list[i][nb] += diff * diff
                elif errnum_list.names[j] == "ptshape":
                    #Uncorrelated
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] + errden[i][nb] * errden[i][nb]
                elif errnum_list.names[j] == "multiplicity_interval":
                    #NB: Assuming ratio: 3prongs over 2prongs here! 2prong part cancels
                    #We use 1/3 of systematic of numerator
                    tot_list[i][nb] += errnum[i][nb] * errnum[i][nb] / 9
                elif errnum_list.names[j] == "sigmav0":
                    #Correlated and usually not plotted in boxes, do nothing
                    pass
                elif errnum_list.names[j] == "branching_ratio":
                    #Uncorrelated, but usually not plotted in boxes, so pass
                    pass
                else:
                    get_logger().fatal("Unknown systematic name: %s", errnum_list.names[j])
        j = j + 1
    tot_list = np.sqrt(tot_list)
    return tot_list

def make_latex_table(column_names, row_names, rows, caption=None, save_path="./table.tex"):
    caption = caption if caption is not None else "Caption"
    with open(save_path, "w") as f:
        f.write("\\documentclass{article}\n\n")
        f.write("\\usepackage[margin=0.7in]{geometry}\n")
        f.write("\\usepackage[parfill]{parskip}\n")
        f.write("\\usepackage{rotating}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{sidewaystable}\n")
        f.write("\\centering\n")
        # As many columns as we need
        columns = "|".join(["c"] * (len(column_names) + 1))
        f.write("\\begin{tabular}{" + columns + "}\n")
        f.write("\\hline\n")
        columns = "&".join([""] + column_names)
        columns = columns.replace("_", "\\_")
        f.write(columns + "\\\\\n")
        f.write("\\hline\\hline\n")
        for rn, row in zip(row_names, rows):
            row_string = "&".join([rn] + row)
            row_string = row_string.replace("_", "\\_")
            f.write(row_string + "\\\\\n")
        f.write("\\end{tabular}\n")
        caption = caption.replace("_", "\\_")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\end{sidewaystable}\n")
        f.write("\\end{document}\n")

def parallelizer(function, argument_list, maxperchunk, max_n_procs=2):
    """
    A centralized version for quickly parallelizing basically identical to what can found in
    the Processer. It could also rely on this one.
    """
    chunks = [argument_list[x:x+maxperchunk] \
              for x in range(0, len(argument_list), maxperchunk)]
    for chunk in chunks:
        print("Processing new chunck size=", maxperchunk)
        pool = mp.Pool(max_n_procs)
        _ = [pool.apply_async(function, args=chunk[i]) for i in range(len(chunk))]
        pool.close()
        pool.join()

def get_timestamp_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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

    def get_total_for_spectra_plot(self):
        """
        Returns a list of total errors
        For now only add in quadrature and take sqrt
        """
        tot_list = [[0., 0., 0., 0.] for _ in range(self.n_bins)]
        for j, errors in enumerate(self.errors.values()):
            for i in range(self.n_bins):
                for nb in range(len(tot_list[i])):
                    if self.names[j] != "branching_ratio" or self.names[j] != "sigmav0":
                        tot_list[i][nb] += (errors[i][nb] * errors[i][nb])
                    if self.names[j] == "feeddown_mult":
                        print("Warning: strategy for feeddown_mult to be refined")
        tot_list = np.sqrt(tot_list)
        return tot_list
