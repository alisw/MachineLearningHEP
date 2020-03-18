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
import math
from array import array
import numpy as np
import pandas as pd
import lz4
from ROOT import TGraphAsymmErrors  # pylint: disable=import-error, no-name-in-module

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
        return dfr[dfr[runvar].isin(runlist)]
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

# pylint: disable=too-many-nested-blocks
def list_folders(main_dir, filenameinput, maxfiles):
    """
    List all files in a subdirectory structure
    """
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
    for _, mydir in enumerate(dirlist):
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
    for _, mydir in enumerate(dirlist):
        print("creating folder ", mydir)
        os.makedirs(mydir)

def checkmakedir(mydir):
    """
    Makes directory using 'mkdir'
    """
    print("creating folder ", mydir)
    os.makedirs(mydir)

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
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

def equal_axis_list(axis1, list2):
    """ Compare the binning of axis1 with list2. """
    if not np.array_equal(get_bins(axis1), np.array(list2)):
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

def setup_histogram(hist, colour=1, markerstyle=25, size=1.5):
    hist.SetStats(0)
    hist.SetTitleSize(0.04, "X")
    hist.SetTitleOffset(1.0, "X")
    hist.SetTitleSize(0.04, "Y")
    hist.SetTitleOffset(1.0, "Y")
    hist.SetLineWidth(2)
    hist.SetLineColor(colour)
    hist.SetMarkerSize(size)
    hist.SetMarkerStyle(markerstyle)
    hist.SetMarkerColor(colour)

def setup_pad(pad):
    pad.SetFillColor(0)
    pad.SetMargin(0.15, 0.9, 0.15, 0.9)
    pad.Draw()
    pad.SetTicks(1, 1)
    pad.cd()

def setup_legend(legend):
    legend.SetBorderSize(0)
    legend.SetTextSize(0.03)
    legend.SetTextFont(42)

def setup_tgraph(tg_, colour=46, alphastyle=0.3, fillstyle=1001):
    tg_.GetXaxis().SetTitleSize(0.04)
    tg_.GetXaxis().SetTitleOffset(1.0)
    tg_.GetYaxis().SetTitleSize(0.04)
    tg_.GetYaxis().SetTitleOffset(1.0)
    tg_.SetFillColorAlpha(colour, alphastyle)
    tg_.SetLineWidth(2)
    tg_.SetLineColor(colour)
    tg_.SetFillStyle(fillstyle)
    tg_.SetMarkerSize(0)

def draw_latex(latex, colour=1, textsize=0.03):
    latex.SetNDC()
    latex.SetTextSize(textsize)
    latex.SetTextColor(colour)
    latex.SetTextFont(42)
    latex.Draw()

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
