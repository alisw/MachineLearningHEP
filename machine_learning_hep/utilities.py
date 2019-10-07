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
from ROOT import TH1F, TH2F
from ROOT import TPad, TCanvas, TLegend, kBlack, kGreen, kRed, kBlue
from machine_learning_hep.selectionutils import select_runs
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

def mergerootfiles(listfiles, mergedfile):
    outstring = ""
    for indexp, _ in enumerate(listfiles):
        outstring = outstring + listfiles[indexp] + " "
    os.system("hadd -f %s  %s " % (mergedfile, outstring))

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

def style_histograms(histos, linestyles=None, markerstyles=None, colors=None):
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

    for i, h in enumerate(histos):
        h.SetLineColor(colors[i % len(colors)])
        h.SetLineStyle(linestyles[i % len(linestyles)])
        h.SetMarkerStyle(markerstyles[i % len(markerstyles)])
        h.SetMarkerColor(colors[i % len(colors)])
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

def put_in_pad(pad, use_log_y, histos, title="", x_label="", y_label=""):
    """
    Providing a TPad this plots all given histograms in that pad adjusting the X- and Y-ranges
    accordingly.
    """
    min_x, max_x, min_y, max_y = find_axes_limits(histos, use_log_y)
    pad.SetLogy(use_log_y)
    pad.cd()
    scale_frame_y = (0.1, 10.) if use_log_y else (0.7, 1.2)
    pad.DrawFrame(min_x, min_y * scale_frame_y[0], max_x, max_y * scale_frame_y[1],
                  f"{title};{x_label};{y_label}")
    for h in histos:
        h.Draw("same")

def plot_histograms(histos, use_log_y=False, ratio=False, legend_titles=None, title="", x_label="",
                    y_label_up="", y_label_ratio="", save_path="./plot.eps", **kwargs):
    """
    Throws all given histograms into one canvas. If desired, a ratio plot will be added.
    """
    linestyles = kwargs.get("linestyles", None)
    markerstyles = kwargs.get("markerstyles", None)
    colors = kwargs.get("colors", None)
    canvas_name = kwargs.get("canvas_name", "Canvas")
    style_histograms(histos, linestyles, markerstyles, colors)

    canvas = TCanvas('canvas', canvas_name, 800, 800)
    pad_up_start = 0.4 if ratio else 0.

    pad_up = TPad("pad_up", "", 0., pad_up_start, 1., 1.)
    if ratio:
        pad_up.SetBottomMargin(0.)
    pad_up.Draw()

    put_in_pad(pad_up, use_log_y, histos, title, "", y_label_up)

    pad_up.cd()
    legend = None
    if legend_titles is not None:
        legend = TLegend(.45, .65, .85, .85)
        legend.SetBorderSize(0)
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.02)
        for h, l in zip(histos, legend_titles):
            legend.AddEntry(h, l)
        legend.Draw()

    canvas.cd()
    pad_ratio = None
    histos_ratio = None

    if ratio:
        histos_ratio = divide_all_by_first(histos)
        pad_ratio = TPad("pad_ratio", "", 0., 0.05, 1., pad_up_start)
        pad_ratio.SetTopMargin(0.)
        pad_ratio.SetBottomMargin(0.3)
        pad_ratio.Draw()

        put_in_pad(pad_ratio, False, histos_ratio, "", x_label, y_label_ratio)

    canvas.SaveAs(save_path)
    canvas.Close()

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
