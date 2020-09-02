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
Script containing all helper functions
e.g. processing files, creating objects, calculating physical quantities.
"""
import multiprocessing as mp
from datetime import datetime
import pickle
import bz2
import gzip
import lzma
import os
import shutil
import math
from array import array
import numpy as np
import pandas as pd
import lz4
from machine_learning_hep.selectionutils import select_runs
from ROOT import TObject, TCanvas, TLegend, TH1, TLatex, TGraph, TGraphAsymmErrors # pylint: disable=import-error, no-name-in-module
from ROOT import kBlack, kRed, kGreen, kBlue, kYellow, kOrange, kMagenta, kCyan, kGray # pylint: disable=import-error, no-name-in-module
from ROOT import kOpenCircle, kOpenSquare, kOpenDiamond, kOpenCross, kOpenStar, kOpenThreeTriangles # pylint: disable=import-error, no-name-in-module
from ROOT import kOpenFourTrianglesX, kOpenDoubleDiamond, kOpenFourTrianglesPlus, kOpenCrossX # pylint: disable=import-error, no-name-in-module
from ROOT import kFullCircle, kFullSquare, kFullDiamond, kFullCross, kFullStar, kFullThreeTriangles # pylint: disable=import-error, no-name-in-module
from ROOT import kFullFourTrianglesX, kFullDoubleDiamond, kFullFourTrianglesPlus, kFullCrossX # pylint: disable=import-error, no-name-in-module

def openfile(filename, attr):
    """
    Open file with different compression types
    """
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
    """
    Query on dataframe
    """
    if selection is not None:
        dfr = dfr.query(selection)
    return dfr

def selectdfrunlist(dfr, runlist, runvar):
    """
    Select smaller runlist on dataframe
    """
    if runlist is not None:
        runlist_np = np.asarray(runlist)
        array_run_np = np.asarray(dfr[runvar].values)
        issel = select_runs(runlist_np, array_run_np)
        dfr = dfr[issel]
    return dfr

def merge_method(listfiles, namemerged):
    """
    Merge list of dataframes into one
    """
    dflist = []
    for myfilename in listfiles:
        myfile = openfile(myfilename, "rb")
        df = pickle.load(myfile)
        dflist.append(df)
    dftot = pd.concat(dflist)
    pickle.dump(dftot, openfile(namemerged, "wb"), protocol=4)

def list_folders(main_dir, filenameinput, maxfiles, select=None):
    """
    List all files in a subdirectory structure
    """
    if not os.path.isdir(main_dir):
        print("the input directory =", main_dir, "doesnt exist")
    list_subdir0 = os.listdir(main_dir)
    listfolders = list()
    for subdir0 in list_subdir0: # pylint: disable=too-many-nested-blocks
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

    if select:
        # Select only folders with a matching sub-string in their paths
        list_folders_tmp = []
        for sel_sub_string in select:
            list_folders_tmp.extend([folder for folder in listfolders if sel_sub_string in folder])
        listfolders = list_folders_tmp

    if maxfiles is not -1:
        listfolders = listfolders[:maxfiles]
    return  listfolders

def create_folder_struc(maindir, listpath):
    """
    Reproduce the folder structure as input
    """
    for path in listpath:
        path = path.split("/")

        folder = os.path.join(maindir, path[0])
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(folder, path[1])
        if not os.path.exists(folder):
            os.makedirs(folder)

def checkdirlist(dirlist):
    """
    Checks if list of folder already exist, to not overwrite by accident
    """
    exfolders = 0
    for mydir in dirlist:
        if os.path.exists(mydir):
            print("rm -rf ", mydir)
            exfolders = exfolders - 1
    return exfolders

def checkdir(mydir):
    """
    Checks if folder already exist, to not overwrite by accident
    """
    exfolders = 0
    if os.path.exists(mydir):
        print("rm -rf ", mydir)
        exfolders = -1
    return exfolders

def checkmakedirlist(dirlist):
    """
    Makes directories from list using 'mkdir'
    """
    for mydir in dirlist:
        print("creating folder ", mydir)
        os.makedirs(mydir)

def checkmakedir(mydir):
    """
    Makes directory using 'mkdir'
    """
    print("creating folder ", mydir)
    os.makedirs(mydir)

def delete_dir(path: str):
    """
    Delete directory if it exists. Return True if success, False otherwise.
    """
    if not os.path.isdir(path):
        print("Directory %s does not exist." % path)
        return True
    print("Deleting directory %s" % path)
    try:
        shutil.rmtree(path)
    except OSError:
        print("Error: Failed to delete directory %s" % path)
        return False
    return True

def delete_dirlist(dirlist: str):
    """
    Delete directories from list. Return True if success, False otherwise.
    """
    for path in dirlist:
        if not delete_dir(path):
            return False
    return True

def appendfiletolist(mylist, namefile):
    """
    Append filename to list
    """
    return [os.path.join(path, namefile) for path in mylist]

def appendmainfoldertolist(prefolder, mylist):
    """
    Append base foldername to paths in list
    """
    return [os.path.join(prefolder, path) for path in mylist]

def createlist(prefolder, mylistfolder, namefile):
    """
    Appends base foldername + filename in list
    """
    listfiles = appendfiletolist(mylistfolder, namefile)
    listfiles = appendmainfoldertolist(prefolder, listfiles)
    return listfiles

def seldf_singlevar(dataframe, var, minval, maxval):
    """
    Make projection on variable using [X,Y), e.g. pT or multiplicity
    """
    dataframe = dataframe.loc[(dataframe[var] >= minval) & (dataframe[var] < maxval)]
    return dataframe

def seldf_singlevar_inclusive(dataframe, var, minval, maxval):
    """
    Make projection on variable using [X,Y), e.g. pT or multiplicity
    """
    dataframe = dataframe.loc[(dataframe[var] >= minval) & (dataframe[var] <= maxval)]
    return dataframe

def split_df_sigbkg(dataframe_, var_signal_):
    """
    Split dataframe in signal and background dataframes
    """
    dataframe_sig_ = dataframe_.loc[dataframe_[var_signal_] == 1]
    dataframe_bkg_ = dataframe_.loc[dataframe_[var_signal_] == 0]
    return dataframe_sig_, dataframe_bkg_

def createstringselection(var, low, high):
    """
    Create string of main dataframe selection (e.g. pT)
    Used as suffix for storing ML plots
    """
    string_selection = "dfselection_"+(("%s_%.1f_%.1f") % (var, low, high))
    return string_selection

def mergerootfiles(listfiles, mergedfile, tmp_dir):
    """
    Using ROOT's 'hadd' utility, to merge output rootfiles from analyses steps
    """
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
    """
    Get timestamp, used for temporary files (like the 'hadd' ones)
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + f"{os.getpid()}"

def make_latex_table(column_names, row_names, rows, caption=None, save_path="./table.tex"):
    """
    Store information in table format in tex file
    """
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

def make_file_path(directory, filename, extension, prefix=None, suffix=None):
    """
    Construct a common path+filename+suffix from args
    """
    if prefix is not None:
        filename = make_pre_suffix(prefix) + "_" + filename
    if suffix is not None:
        filename = filename + "_" + make_pre_suffix(suffix)
    extension = extension.replace(".", "")
    return os.path.join(directory, filename + "." + extension)

def make_pre_suffix(args):
    """
    Construct a common file suffix from args
    """
    try:
        _ = iter(args)
    except TypeError:
        args = [args]
    else:
        if isinstance(args, str):
            args = [args]
    return "_".join(args)

def make_message_notfound(name, location=None):
    """
    Return a formatted error message for not found or not properly loaded objects
    stating the name and optionally the location.
    """
    if location is not None:
        return "Error: Failed to get %s in %s" % (name, location)
    return "Error: Failed to get %s" % name

# Jet related functions, to comment

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
    """ Get a numpy array containing bin edges of a histogram axis (TAxis). """
    return np.array([axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins() + 2)])

def equal_axes(axis1, axis2):
    """ Compare the binning of two histogram axes. """
    if not np.array_equal(get_bins(axis1), get_bins(axis2)):
        return False
    return True

def equal_axis_list(axis1, list2, precision=10):
    """ Compare the binning of axis1 with list2. """
#    if not np.array_equal(get_bins(axis1), np.array(list2)):
#        return False
    bins = get_bins(axis1)
    if len(bins) != len(list2):
        return False
    for i, j in zip(bins, list2):
        if round(i, precision) != round(j, precision):
            return False
    return True

def equal_binning(his1, his2):
    """ Compare binning of axes of two histograms (derived from TH1). """
    if not equal_axes(his1.GetXaxis(), his2.GetXaxis()):
        return False
    if not equal_axes(his1.GetYaxis(), his2.GetYaxis()):
        return False
    if not equal_axes(his1.GetZaxis(), his2.GetZaxis()):
        return False
    return True

def equal_binning_lists(his, list_x=None, list_y=None, list_z=None):
    """ Compare binning of axes of a histogram with the respective lists. """
    if list_x is not None and not equal_axis_list(his.GetXaxis(), list_x):
        return False
    if list_y is not None and not equal_axis_list(his.GetYaxis(), list_y):
        return False
    if list_z is not None and not equal_axis_list(his.GetZaxis(), list_z):
        return False
    return True

def folding(h_input, response_matrix, h_output):
    h_folded = h_output.Clone("h_folded")
    for a in range(h_output.GetNbinsX()):
        for b in range(h_output.GetNbinsY()):
            val = 0.0
            val_err = 0.0
            for k in range(h_input.GetNbinsX()):
                for l in range(h_input.GetNbinsY()):
                    index_x_out = a+ h_output.GetNbinsX()*b
                    index_x_in = k + h_input.GetNbinsX()*l
                    val = val + h_input.GetBinContent(k+1, l+1) * \
                        response_matrix(index_x_out, index_x_in)
                    val_err = val_err + h_input.GetBinError(k+1, l+1) * \
                        h_input.GetBinError(k+1, l+1)* \
                        response_matrix(index_x_out, index_x_in) * \
                        response_matrix(index_x_out, index_x_in)
            h_folded.SetBinContent(a+1, b+1, val)
            h_folded.SetBinError(a+1, b+1, math.sqrt(val_err))
    return h_folded

def get_plot_range(val_min, val_max, margin_min, margin_max, logscale=False):
    '''Return the minimum and maximum of the plotting range so that there are margins
    expressed as fractions of the plotting range.'''
    k = 1 - margin_min - margin_max
    if k <= 0:
        return None, None
    k_min = margin_min / k
    k_max = margin_max / k
    if logscale:
        if val_min <= 0 or val_max <= 0:
            print("Error: Cannot plot non-positive values in logscale.")
            return None, None
        val_range = val_max / val_min
        val_min_plot = val_min / pow(val_range, k_min)
        val_max_plot = val_max * pow(val_range, k_max)
    else:
        val_range = val_max - val_min
        val_min_plot = val_min - k_min * val_range
        val_max_plot = val_max + k_max * val_range
    return val_min_plot, val_max_plot

def get_x_window_gr(l_gr: list, with_errors=True):
    '''Return the minimum and maximum x value so that all the points of the graphs in the list
    fit in the range (by default including the error bars).'''
    def err_low(graph):
        return graph.GetEXlow if isinstance(graph, TGraphAsymmErrors) else graph.GetEX
    def err_high(graph):
        return graph.GetEXhigh if isinstance(graph, TGraphAsymmErrors) else graph.GetEX

    if not isinstance(l_gr, list):
        l_gr = [l_gr]
    x_min = float("inf")
    x_max = float("-inf")
    for gr in l_gr:
        for i in range(gr.GetN()):
            x_min = min(x_min, (gr.GetX())[i] - ((err_low(gr)())[i] if with_errors else 0))
            x_max = max(x_max, (gr.GetX())[i] + ((err_high(gr)())[i] if with_errors else 0))
    return x_min, x_max

def get_x_window_his(l_his: list):
    '''Return the minimum and maximum x value so that all the bins of the histograms in the list
    fit in the range.'''
    if not isinstance(l_his, list):
        l_his = [l_his]
    x_min = float("inf")
    x_max = float("-inf")
    for his in l_his:
        x_min = min(x_min, his.GetXaxis().GetBinLowEdge(1))
        x_max = max(x_max, his.GetXaxis().GetBinUpEdge(his.GetNbinsX()))
    return x_min, x_max

def get_y_window_gr(l_gr: list, with_errors=True):
    '''Return the minimum and maximum y value so that all the points of the graphs in the list
    fit in the range (by default including the error bars).'''
    def err_low(graph):
        return graph.GetEYlow if isinstance(graph, TGraphAsymmErrors) else graph.GetEY
    def err_high(graph):
        return graph.GetEYhigh if isinstance(graph, TGraphAsymmErrors) else graph.GetEY

    if not isinstance(l_gr, list):
        l_gr = [l_gr]
    y_min = float("inf")
    y_max = float("-inf")
    for gr in l_gr:
        for i in range(gr.GetN()):
            y_min = min(y_min, (gr.GetY())[i] - ((err_low(gr)())[i] if with_errors else 0))
            y_max = max(y_max, (gr.GetY())[i] + ((err_high(gr)())[i] if with_errors else 0))
    return y_min, y_max

def get_y_window_his(l_his: list, with_errors=True):
    '''Return the minimum and maximum y value so that all the points of the histograms in the list
    fit in the range (by default including the error bars).'''
    if not isinstance(l_his, list):
        l_his = [l_his]
    y_min = float("inf")
    y_max = float("-inf")
    for his in l_his:
        for i in range(his.GetNbinsX()):
            cont = his.GetBinContent(i + 1)
            err = his.GetBinError(i + 1) if with_errors else 0
            y_min = min(y_min, cont - err)
            y_max = max(y_max, cont + err)
    return y_min, y_max

def get_colour(i: int, scheme=1):
    '''Return a colour from the list.'''
    colours = [kBlack, kBlue, kRed, kGreen + 1, kOrange + 1, kMagenta, kCyan + 1, kGray + 1, \
        kBlue + 2, kRed - 3, kGreen + 3, kYellow  + 1, kMagenta + 1, kCyan + 2, kRed + 3]
    colours_alice_point = [kBlack, kBlue + 1, kRed + 1, kGreen + 3, kMagenta + 2, kOrange + 4, \
        kCyan + 2, kYellow + 2]
    colours_alice_syst = [kGray + 1, kBlue - 7, kRed - 7, kGreen - 6, kMagenta - 4, kOrange - 3, \
        kCyan - 6, kYellow - 7]
    if scheme == 1:
        list_col = colours_alice_point
    elif scheme == 2:
        list_col = colours_alice_syst
    else:
        list_col = colours
    return list_col[i % len(list_col)]

def get_marker(i: int, option=0):
    '''Return a marker from the list.'''
    markers_open = [kOpenCircle, kOpenSquare, kOpenCross, kOpenDiamond, kOpenCrossX,
                    kOpenFourTrianglesPlus, kOpenStar,
                    kOpenThreeTriangles, kOpenFourTrianglesX, kOpenDoubleDiamond]
    markers_full = [kFullCircle, kFullSquare, kFullCross, kFullDiamond, kFullCrossX,
                    kFullFourTrianglesPlus, kFullStar,
                    kFullThreeTriangles, kFullFourTrianglesX, kFullDoubleDiamond]
    markers_thick = [88, 72, 75, 74, 76, 80, 82, 83, 84, 85]
    if option == 1:
        list_markers = markers_thick
    elif option == 2:
        list_markers = markers_full
    else:
        list_markers = markers_open
    return list_markers[i % len(list_markers)]

def get_markersize(marker: int, size_def=1.5):
    '''Return a marker size.'''
    markers_small = [kOpenCross, kOpenDiamond, kOpenStar, kOpenDoubleDiamond,
                     kOpenFourTrianglesPlus, kOpenCrossX,
                     kOpenThreeTriangles, kOpenFourTrianglesX,
                     kFullCross, kFullDiamond, kFullStar, kFullDoubleDiamond,
                     kFullFourTrianglesPlus, kFullCrossX,
                     kFullThreeTriangles, kFullFourTrianglesX,
                     75, 74, 76, 83, 84, 85]
    if marker in markers_small:
        return size_def * 4 / 3
    return size_def

def setup_histogram(hist, colour=1, markerstyle=kOpenCircle, size=1.5):
    hist.SetStats(0)
    hist.SetTitleSize(0.05, "X")
    hist.SetTitleOffset(1.0, "X")
    hist.SetTitleSize(0.05, "Y")
    hist.SetTitleOffset(1.0, "Y")
    hist.SetLineWidth(3)
    hist.SetLineColor(colour)
    hist.SetMarkerSize(size)
    hist.SetMarkerStyle(markerstyle)
    hist.SetMarkerColor(colour)

def setup_canvas(can):
    can.SetCanvasSize(1900, 1500)
    can.SetWindowSize(500, 500)
    can.SetFillColor(0)
    can.SetTicks(1, 1)
    can.SetBottomMargin(0.12)
    can.SetLeftMargin(0.12)
    can.SetTopMargin(0.1)
    can.SetRightMargin(0.02)
    can.cd()

def setup_legend(legend, textsize=0.03):
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(textsize)
    legend.SetTextFont(42)

def setup_tgraph(tg_, colour=1, markerstyle=kOpenCircle, size=1.5, alphastyle=0.8,
                 fillstyle=1001, textsize=0.05):
    tg_.GetXaxis().SetTitleSize(textsize)
    tg_.GetXaxis().SetTitleOffset(1.0)
    tg_.GetYaxis().SetTitleSize(textsize)
    tg_.GetYaxis().SetTitleOffset(1.0)
    tg_.SetFillColorAlpha(colour, alphastyle)
    tg_.SetLineWidth(0)
    tg_.SetLineColor(colour)
    tg_.SetFillStyle(fillstyle)
    tg_.SetMarkerSize(size)
    tg_.SetMarkerStyle(markerstyle)
    tg_.SetMarkerColor(colour)

def draw_latex(latex, colour=1, textsize=0.03):
    latex.SetNDC()
    latex.SetTextSize(textsize)
    latex.SetTextColor(colour)
    latex.SetTextFont(42)
    latex.Draw()

def make_plot(name, path=None, suffix="eps", title="", size=None, margins_c=None, # pylint: disable=too-many-arguments, too-many-branches, too-many-statements, too-many-locals
              list_obj=None, labels_obj=None,
              leg_pos=None, opt_leg_h="P", opt_leg_g="P", opt_plot_h="", opt_plot_g="P0",
              offsets_xy=None, maxdigits=3, colours=None, markers=None, sizes=None,
              range_x=None, range_y=None, margins_y=None, with_errors="xy", logscale=None):
    """
    Make a plot with objects from a list (list_obj).
    Returns a TCanvas and a list of other created ROOT objects.
    Minimum example:
        make_plot("canvas", list_obj=[histogram], path=".")
    To have access to the created object, do:
        canvas, list_can = make_plot("canvas", list_obj=[histogram])
    Features:
    - plotting of histograms (TH??), graphs (TGraph*), text fields (TLatex) and any other objects
        derived from TObject in any count and order
    - automatic calculation of plotting ranges (x, y) based on the data (histograms and graphs)
    - arbitrary x, y range
    - automatic style settings
    - optional plotting of the legend (enabled by providing the coordinates)
    - automatic adding of legend entries (in the plotting order)
    - logarithmic scale of x, y, z axes (logscale), (format: string containing any of x, y, z)
    - saving the canvas to a specified location (path) in a specified format (suffix)
    - access to created ROOT objects
    Adjustable parameters:
    - title and axis titles (title), (format: "title_plot;title_x;title_y")
    - canvas size (size), (format: [width, height])
    - plotting options for histograms and graphs (opt_plot_h, opt_plot_g),
        (format: see THistPainter and TGraphPainter, respectively)
    - legend position (leg_pos), (format: [x_min, y_min, x_max, y_max])
    - labels of legend entries (labels_obj)
    - styles of legend entries (opt_leg_h, opt_leg_g), (format: see TLegend::AddEntry)
    - colours, markers, sizes (colours, markers, sizes), (format: list of numbers or named values)
    - canvas margins (margins_c), (format: [bottom, left, top, right])
    - offsets of axis titles (offsets_xy), (format: [x, y])
    - maximum number of digits of the axis labels (maxdigits)
    - x range (range_x), (format: [x_min, x_max])
    - y range (range_y), (format: [y_min, y_max])
    - vertical margins between the horizontal axes and the data (margins_y), (format: [lower, upper]
        expressed as fractions of the total plotting range)
    - including the error bars in the range calculations (with_errors),
        (format: string containing any of x, y)
    """

    # HELPING FUNCTIONS

    def min0_gr(graph):
        """ Get the minimum positive y value in the graph. """
        list_pos = [y for y in graph.GetY() if y > 0]
        return min(list_pos) if list_pos else float("inf")

    def get_my_colour(i: int):
        if colours and isinstance(colours, list) and len(colours) > 0:
            return colours[i % len(colours)]
        return get_colour(i)

    def get_my_marker(i: int):
        if markers and isinstance(markers, list) and len(markers) > 0:
            return markers[i % len(markers)]
        return get_marker(i)

    def get_my_size(i: int):
        if sizes and isinstance(sizes, list) and len(sizes) > 0:
            return sizes[i % len(sizes)]
        return get_markersize(get_my_marker(i))

    def plot_graph(graph):
        setup_tgraph(graph, get_my_colour(counter_plot), get_my_marker(counter_plot),
                     get_my_size(counter_plot))
        graph.SetTitle(title)
        graph.GetXaxis().SetLimits(x_min_plot, x_max_plot)
        graph.GetYaxis().SetRangeUser(y_min_plot, y_max_plot)
        graph.GetXaxis().SetMaxDigits(maxdigits)
        graph.GetYaxis().SetMaxDigits(maxdigits)
        if offsets_xy:
            graph.GetXaxis().SetTitleOffset(offsets_xy[0])
            graph.GetYaxis().SetTitleOffset(offsets_xy[1])
        if leg and n_labels > counter_plot and len(labels_obj[counter_plot]) > 0:
            leg.AddEntry(graph, labels_obj[counter_plot], opt_leg_g)
        graph.Draw(opt_plot_g + "A" if counter_plot == 0 else opt_plot_g)

    def plot_histogram(histogram):
        # If nothing has been plotted yet, plot an empty graph to set the exact ranges.
        if counter_plot == 0:
            gr = TGraph(histogram)
            setup_tgraph(gr)
            gr.SetMarkerSize(0)
            gr.SetTitle(title)
            gr.GetXaxis().SetLimits(x_min_plot, x_max_plot)
            gr.GetYaxis().SetRangeUser(y_min_plot, y_max_plot)
            gr.GetXaxis().SetMaxDigits(maxdigits)
            gr.GetYaxis().SetMaxDigits(maxdigits)
            if offsets_xy:
                gr.GetXaxis().SetTitleOffset(offsets_xy[0])
                gr.GetYaxis().SetTitleOffset(offsets_xy[1])
            gr.Draw("AP")
            list_new.append(gr)
        setup_histogram(histogram, get_my_colour(counter_plot), get_my_marker(counter_plot),
                        get_my_size(counter_plot))
        if leg and n_labels > counter_plot and len(labels_obj[counter_plot]) > 0:
            leg.AddEntry(histogram, labels_obj[counter_plot], opt_leg_h)
        histogram.Draw(opt_plot_h)

    def plot_latex(latex):
        draw_latex(latex)

    def is_histogram(obj):
        return isinstance(obj, TH1)

    def is_graph(obj):
        return isinstance(obj, TGraph)

    def is_latex(obj):
        return isinstance(obj, TLatex)

    # BODY STARTS HERE

    if not (isinstance(list_obj, list) and len(list_obj) > 0):
        print("Error: Empty list of objects")
        return None, None

    list_new = [] # list of created objects that need to exist outside the function
    if not (isinstance(offsets_xy, list) and len(offsets_xy) == 2):
        offsets_xy = None
    if not isinstance(labels_obj, list):
        labels_obj = []
    n_labels = len(labels_obj)
    if margins_y is None:
        margins_y = [0.05, 0.05]

    # create and set canvas
    can = TCanvas(name, name)
    setup_canvas(can)
    if isinstance(size, list) and len(size) == 2:
        can.SetCanvasSize(*size)
    # set canvas margins
    if isinstance(margins_c, list) and len(margins_c) > 0:
        for setter, value in zip([can.SetBottomMargin, can.SetLeftMargin,
                                  can.SetTopMargin, can.SetRightMargin], margins_c):
            setter(value)
    # set logarithmic scale for selected axes
    log_y = False
    if isinstance(logscale, str) and len(logscale) > 0:
        for setter, axis in zip([can.SetLogx, can.SetLogy, can.SetLogz], ["x", "y", "z"]):
            if axis in logscale:
                setter()
                if axis == "y":
                    log_y = True

    # create and set legend
    leg = None
    if isinstance(leg_pos, list) and len(leg_pos) == 4:
        leg = TLegend(*leg_pos)
        setup_legend(leg)
        list_new.append(leg)

    # range calculation
    list_h = [] # list of histograms
    list_g = [] # list of graphs
    for obj in list_obj:
        if is_histogram(obj):
            list_h.append(obj)
        elif is_graph(obj):
            list_g.append(obj)

    # get x range of histograms
    x_min_h, x_max_h = float("inf"), float("-inf")
    if len(list_h) > 0:
        x_min_h, x_max_h = get_x_window_his(list_h)
    # get x range of graphs
    x_min_g, x_max_g = float("inf"), float("-inf")
    if len(list_g) > 0:
        x_min_g, x_max_g = get_x_window_gr(list_g, "x" in with_errors)
    # get total x range
    x_min = min(x_min_h, x_min_g)
    x_max = max(x_max_h, x_max_g)
    # get plotting x range
    x_min_plot, x_max_plot = x_min, x_max
    if isinstance(range_x, list) and len(range_x) == 2:
        x_min_plot, x_max_plot = range_x

    # get y range of histograms
    y_min_h, y_max_h = float("inf"), float("-inf")
    if len(list_h) > 0:
        y_min_h, y_max_h = get_y_window_his(list_h, "y" in with_errors)
        if log_y and y_min_h <= 0:
            y_min_h = min([h.GetMinimum(0) for h in list_h])
    # get y range of graphs
    y_min_g, y_max_g = float("inf"), float("-inf")
    if len(list_g) > 0:
        y_min_g, y_max_g = get_y_window_gr(list_g, "y" in with_errors)
        if log_y and y_min_g <= 0:
            y_min_g = min([min0_gr(g) for g in list_g])
    # get total y range
    y_min = min(y_min_h, y_min_g)
    y_max = max(y_max_h, y_max_g)
    # get plotting y range
    y_min_plot, y_max_plot = y_min, y_max
    if isinstance(margins_y, list) and len(margins_y) == 2:
        y_min_plot, y_max_plot = get_plot_range(y_min, y_max, *margins_y, log_y)
    if isinstance(range_y, list) and len(range_y) == 2:
        y_min_plot, y_max_plot = range_y

    # append "same" to the histogram plotting option if needed
    opt_plot_h = opt_plot_h.lower()
    opt_not_in = all(opt not in opt_plot_h for opt in ("same", "lego", "surf"))
    if opt_not_in:
        opt_plot_h += " same"

    # plot objects
    counter_plot = 0 # counter of plotted histograms and graphs
    for obj in list_obj:
        if is_histogram(obj):
            plot_histogram(obj)
            counter_plot += 1
        elif is_graph(obj):
            plot_graph(obj)
            counter_plot += 1
        elif is_latex(obj):
            plot_latex(obj)
        elif isinstance(obj, TObject):
            obj.Draw()
        else:
            continue

    # plot axes on top
    can.RedrawAxis()

    # plot legend
    if leg:
        leg.Draw()

    # save canvas if necessary info provided
    if path and name and suffix:
        can.SaveAs("%s/%s.%s" % (path, name, suffix))

    return can, list_new

def tg_sys(central, variations):
    shapebins_centres = []
    shapebins_contents = []
    shapebins_widths_up = []
    shapebins_widths_down = []
    shapebins_error_up = []
    shapebins_error_down = []

    for i in range(central.GetNbinsX()):
        shapebins_centres.append(central.GetBinCenter(i+1))
        shapebins_contents.append(central.GetBinContent(i+1))
        shapebins_widths_up.append(central.GetBinWidth(i+1)*0.5)
        shapebins_widths_down.append(central.GetBinWidth(i+1)*0.5)
        error_up = 0
        error_down = 0
        for j, _ in enumerate(variations):
            error = variations[j].GetBinContent(i+1)-central.GetBinContent(i+1)
            if error > 0 and error > error_up:
                error_up = error
            if error < 0 and abs(error) > error_down:
                error_down = abs(error)
        shapebins_error_up.append(error_up)
        shapebins_error_down.append(error_down)
    shapebins_centres_array = array('d', shapebins_centres)
    shapebins_contents_array = array('d', shapebins_contents)
    shapebins_widths_up_array = array('d', shapebins_widths_up)
    shapebins_widths_down_array = array('d', shapebins_widths_down)
    shapebins_error_up_array = array('d', shapebins_error_up)
    shapebins_error_down_array = array('d', shapebins_error_down)
    tg = TGraphAsymmErrors(central.GetNbinsX(), shapebins_centres_array,
                           shapebins_contents_array, shapebins_widths_down_array,
                           shapebins_widths_up_array, shapebins_error_down_array,
                           shapebins_error_up_array)
    return tg
