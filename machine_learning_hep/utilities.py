#  © Copyright CERN 2024. All rights not expressly granted are reserved.  #
#                 Author: Gian.Michele.Innocenti@cern.ch                  #
# This program is free software: you can redistribute it and/or modify it #
#  under the terms of the GNU General Public License as published by the  #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version. This program is distributed in the hope that #
#  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  #
#     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#           See the GNU General Public License for more details.          #
#    You should have received a copy of the GNU General Public License    #
#   along with this program. if not, see <https://www.gnu.org/licenses/>. #

"""
Script containing all helper functions
e.g. processing files, creating objects, calculating physical quantities.
"""
import bz2
import gzip
import lzma
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from array import array
from datetime import datetime

import lz4  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
from ROOT import (TH1, TH1F, TCanvas, TGraph, TGraphAsymmErrors, TLatex, TLegend,
                  TObject, TRandom3, kBlack, kBlue, kCyan, kFullCircle, kFullCross,
                  kFullCrossX, kFullDiamond, kFullDoubleDiamond, kFullFourTrianglesPlus,
                  kFullFourTrianglesX, kFullSquare, kFullStar, kFullThreeTriangles,
                  kGray, kGreen, kMagenta, kOpenCircle, kOpenCross, kOpenCrossX,
                  kOpenDiamond, kOpenDoubleDiamond, kOpenFourTrianglesPlus,
                  kOpenFourTrianglesX, kOpenSquare, kOpenStar, kOpenThreeTriangles,
                  kOrange, kRed, kYellow,)

from machine_learning_hep.logger import get_logger
from machine_learning_hep.selectionutils import select_runs

# pylint: disable=too-many-lines

logger = get_logger()


def hist2array(hist):
    assert hist.GetDimension() == 1, "can only convert 1-d histogram"
    return [hist.GetBinContent(x) for x in range(hist.GetNbinsX())]


def array2hist(arr, hist):
    assert arr.ndim() == 1 and hist.GetDimension() == arr.ndim()
    for i, x in enumerate(arr):
        hist.SetBinContent(i + 1, x)


def fill_response(response, dfi: pd.DataFrame, dfmissed: pd.DataFrame = None, write=False):
    dfi.apply(lambda row: response.Fill(row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]), axis=1)
    if dfmissed:
        dfmissed.apply(lambda row: response.Miss(row.iloc[0], row.iloc[1]), axis=1)
    if write:
        response.Write()


def openfile(filename, attr):
    """
    Open file with different compression types
    """
    if filename.lower().endswith(".bz2"):
        return bz2.BZ2File(filename, attr)
    if filename.lower().endswith(".xz"):
        return lzma.open(filename, attr)
    if filename.lower().endswith(".gz"):
        return gzip.open(filename, attr)
    if filename.lower().endswith(".lz4"):
        return lz4.frame.open(filename, attr)
    return open(filename, attr, encoding="utf-8" if "b" not in attr else None)


def write_df(dfo, path):
    logger.debug("writing df to <%s>", path)
    if path.endswith(".parquet"):
        start = time.time()
        dfo.to_parquet(path)
        logger.debug("written to parquet in %.2f s", time.time() - start)
    else:
        start = time.time()
        with openfile(path, "wb") as file:
            pickle.dump(dfo, file, pickle.HIGHEST_PROTOCOL)
        logger.debug("written to pickle in %.2f s", time.time() - start)


def read_df(path, **kwargs):
    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path, **kwargs)
        else:
            df = pickle.load(openfile(path, "rb"))
    except Exception as e:  # pylint: disable=broad-except
        logger.critical("failed to open file <%s>: %s", path, str(e))
        sys.exit()
    return df


def mask_df(df_to_mask, mask_config):
    """
    Potentially mask some column values
    """

    def conv_none(value):
        if value is None:
            return np.nan
        return value

    mask_by_query_list = []
    mask_by_value_dict = {}
    for mc in mask_config:
        if mc["column"] not in df_to_mask:
            print(f"Nothing to mask for {mc['column']}, skip...")
            continue
        if "find_by_query" in mc:
            mask_by_query_list.append(mc)
            continue
        mask_by_value_dict[mc["column"]] = {mc["find_by_value"]: conv_none(mc["mask_with"])}

    # Mask all we can do by value
    df_to_mask.replace(mask_by_value_dict, inplace=True)

    # Now mask all which are searched by query
    for mc in mask_by_query_list:
        # find indices to mask which fulfil the query
        # This cast to float should be obsolete as soon as we move to a more recent pandas version
        df_to_mask[mc["column"]] = df_to_mask[mc["column"]].astype(float)
        mask_indices = df_to_mask.query(mc["find_by_query"]).index
        # Mask the at the column name with mask value
        df_to_mask.loc[mask_indices, [mc["column"]]] = conv_none(mc["mask_with"])


def dfquery(df, selection, **kwargs):  # pylint: disable=invalid-name
    return df.query(selection, **kwargs) if selection is not None else df


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


def count_df_length_pkl(*pkls):
    """
    Return total number of entries in all pkls
    """
    return sum(len(read_df(pkl)) for pkl in pkls)


def merge_method(listfiles, namemerged):
    """
    Merge dataframes from all files and write to single file
    """
    if len(listfiles) > 0:
        df_merged = pd.concat(read_df(filename) for filename in listfiles)
        write_df(df_merged, namemerged)


def seldf_singlevar(dataframe, var, minval, maxval):
    """
    Make projection on variable using [X,Y), e.g. pT or multiplicity
    """
    return dataframe.loc[(dataframe[var] >= minval) & (dataframe[var] < maxval)]


def seldf_singlevar_inclusive(dataframe, var, minval, maxval):
    """
    Make projection on variable using [X,Y), e.g. pT or multiplicity
    """
    return dataframe.loc[(dataframe[var] >= minval) & (dataframe[var] <= maxval)]


def split_df_classes(dataframe_, var_class_, output_labels_):
    """
    Split dataframe into dataframes for different output labels
    """
    return {label: dataframe_.loc[dataframe_[var_class_] == ind] for ind, label in enumerate(output_labels_)}


def createstringselection(var, low, high):
    """
    Create string of main dataframe selection (e.g. pT)
    Used as suffix for storing ML plots
    """
    return f"dfselection_{var}_{low:.1f}_{high:.1f}"


def mergerootfiles(listfiles, mergedfile, tmp_dir):
    """
    Using ROOT's 'hadd' utility, to merge output rootfiles from analyses steps
    """

    def divide_chunks(list_to_split, chunk_size):
        for i in range(0, len(list_to_split), chunk_size):
            yield list_to_split[i : i + chunk_size]

    tmp_files = []
    if len(listfiles) > 500:
        for i, split_list in enumerate(divide_chunks(listfiles, 500)):
            tmp_files.append(os.path.join(tmp_dir, f"hadd_tmp_merged{i}.root"))
            outstring = " ".join(split_list)
            os.system(f"hadd -f -j 30 {tmp_files[-1]} {outstring}")
    else:
        tmp_files = listfiles

    outstring = " ".join(tmp_files)
    os.system(f"hadd -f -j 30 {mergedfile} {outstring}")


def parallelizer(function, argument_list, maxperchunk, max_n_procs=2):
    """
    A centralized version for quickly parallelizing basically identical to what can found in
    the Processer. It could also rely on this one.
    """
    chunks = [argument_list[x : x + maxperchunk] for x in range(0, len(argument_list), maxperchunk)]
    for chunk in chunks:
        print("Processing new chunck size=", maxperchunk)
        with mp.Pool(max_n_procs) as pool:
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
    with open(save_path, "w", encoding="utf-8") as f:
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
        return f"Error: Failed to get {name} in {location}"
    return f"Error: Failed to get {name}"


# Jet related functions, to comment


def z_calc(pt_1, phi_1, eta_1, pt_2, phi_2, eta_2):
    np_pt_1 = pt_1.values
    np_pt_2 = pt_2.values
    np_phi_1 = phi_1.values
    np_phi_2 = phi_2.values
    np_eta_1 = eta_1.values
    np_eta_2 = eta_2.values

    cos_phi_1 = np.cos(np_phi_1)
    cos_phi_2 = np.cos(np_phi_2)
    sin_phi_1 = np.sin(np_phi_1)
    sin_phi_2 = np.sin(np_phi_2)
    sinh_eta_1 = np.sinh(np_eta_1)
    sinh_eta_2 = np.sinh(np_eta_2)

    px_1 = np_pt_1 * cos_phi_1
    px_2 = np_pt_2 * cos_phi_2
    py_1 = np_pt_1 * sin_phi_1
    py_2 = np_pt_2 * sin_phi_2
    pz_1 = np_pt_1 * sinh_eta_1
    pz_2 = np_pt_2 * sinh_eta_2
    numerator = px_1 * px_2 + py_1 * py_2 + pz_1 * pz_2
    denominator = px_1 * px_1 + py_1 * py_1 + pz_1 * pz_1
    return numerator / denominator


def z_gen_calc(pt_1, phi_1, eta_1, pt_2, delta_phi, delta_eta):
    phi_2 = phi_1 + delta_phi
    eta_2 = eta_1 - delta_eta
    return z_calc(pt_1, phi_1, eta_1, pt_2, phi_2, eta_2)


def get_bins(axis):
    """Get a numpy array containing bin edges of a histogram axis (TAxis)."""
    return np.array([axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins() + 2)])


def equal_axes(axis1, axis2):
    """Compare the binning of two histogram axes."""
    return np.array_equal(get_bins(axis1), get_bins(axis2))


def equal_axis_list(axis1, list2, precision=10):
    """Compare the binning of axis1 with list2."""
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
    """Compare binning of axes of two histograms (derived from TH1)."""
    return (
        equal_axes(his1.GetXaxis(), his2.GetXaxis())
        and equal_axes(his1.GetYaxis(), his2.GetYaxis())
        and equal_axes(his1.GetZaxis(), his2.GetZaxis())
    )


def equal_binning_lists(his, list_x=None, list_y=None, list_z=None):
    """Compare binning of axes of a histogram with the respective lists."""
    if list_x is not None and not equal_axis_list(his.GetXaxis(), list_x):
        return False
    if list_y is not None and not equal_axis_list(his.GetYaxis(), list_y):
        return False
    if list_z is not None and not equal_axis_list(his.GetZaxis(), list_z):
        return False
    return True


def folding(h_input, response_matrix, h_output):
    for a in range(h_output.GetNbinsX()):
        for b in range(h_output.GetNbinsY()):
            val = 0.0
            err = 0.0
            index_x_out = a + h_output.GetNbinsX() * b
            for k in range(h_input.GetNbinsX()):
                for m in range(h_input.GetNbinsY()):
                    index_x_in = k + h_input.GetNbinsX() * m
                    val += h_input.GetBinContent(k + 1, m + 1) * response_matrix(index_x_out, index_x_in)
                    err += h_input.GetBinError(k + 1, m + 1) ** 2 * response_matrix(index_x_out, index_x_in) ** 2
            h_output.SetBinContent(a + 1, b + 1, val)
            h_output.SetBinError(a + 1, b + 1, math.sqrt(err))
    return h_output


def get_plot_range(val_min, val_max, margin_min, margin_max, logscale=False):
    """Return the minimum and maximum of the plotting range so that there are margins
    expressed as fractions of the plotting range."""
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
    """Return the minimum and maximum x value so that all the points of the graphs in the list
    fit in the range (by default including the error bars)."""

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
    """Return the minimum and maximum x value so that all the bins of the histograms in the list
    fit in the range."""
    if not isinstance(l_his, list):
        l_his = [l_his]
    x_min = float("inf")
    x_max = float("-inf")
    for his in l_his:
        x_min = min(x_min, his.GetXaxis().GetBinLowEdge(1))
        x_max = max(x_max, his.GetXaxis().GetBinUpEdge(his.GetNbinsX()))
    return x_min, x_max


def get_y_window_gr(l_gr: list, with_errors=True, range_x=None):
    """Return the minimum and maximum y value so that all the points of the graphs in the list
    fit in the range (by default including the error bars).
    Consider only points within range_x if provided."""

    def err_low(graph):
        return graph.GetEYlow if isinstance(graph, TGraphAsymmErrors) else graph.GetEY

    def err_high(graph):
        return graph.GetEYhigh if isinstance(graph, TGraphAsymmErrors) else graph.GetEY

    if not isinstance(l_gr, list):
        l_gr = [l_gr]
    y_min = float("inf")
    y_max = float("-inf")
    consider_x = isinstance(range_x, list) and len(range_x) == 2
    for gr in l_gr:
        for i in range(gr.GetN()):
            x_i = (gr.GetX())[i]
            if consider_x and (x_i < range_x[0] or x_i > range_x[1]):
                continue
            y_min = min(y_min, (gr.GetY())[i] - ((err_low(gr)())[i] if with_errors else 0))
            y_max = max(y_max, (gr.GetY())[i] + ((err_high(gr)())[i] if with_errors else 0))
    return y_min, y_max


def get_y_window_his(l_his: list, with_errors=True, range_x=None):
    """Return the minimum and maximum y value so that all the points of the histograms in the list
    fit in the range (by default including the error bars).
    Consider only bins within range_x if provided."""
    if not isinstance(l_his, list):
        l_his = [l_his]
    y_min = float("inf")
    y_max = float("-inf")
    consider_x = isinstance(range_x, list) and len(range_x) == 2
    for his in l_his:
        for ibin in range(his.GetXaxis().GetFirst(), his.GetXaxis().GetLast() + 1):
            x_ibin = his.GetXaxis().GetBinCenter(ibin)
            if consider_x and (x_ibin < range_x[0] or x_ibin > range_x[1]):
                continue
            cont = his.GetBinContent(ibin)
            err = his.GetBinError(ibin) if with_errors else 0
            y_min = min(y_min, cont - err)
            y_max = max(y_max, cont + err)
    return y_min, y_max


colours = [
    kBlack,
    kBlue,
    kRed,
    kGreen + 1,
    kOrange + 1,
    kMagenta,
    kCyan + 1,
    kGray + 1,
    kBlue + 2,
    kRed - 3,
    kGreen + 3,
    kYellow + 1,
    kMagenta + 1,
    kCyan + 2,
    kRed + 3,
]
colours_alice_point = [kBlack, kBlue + 1, kRed + 1, kGreen + 3, kMagenta + 2, kOrange + 4, kCyan + 2, kYellow + 2]
colours_alice_syst = [kGray + 1, kBlue - 7, kRed - 7, kGreen - 6, kMagenta - 4, kOrange - 3, kCyan - 6, kYellow - 7]


def get_colour(i: int, scheme=1):
    """Return a colour from the list."""
    if scheme == 1:
        list_col = colours_alice_point
    elif scheme == 2:
        list_col = colours_alice_syst
    else:
        list_col = colours
    return list_col[i % len(list_col)]


def get_marker(i: int, option=0):
    """Return a marker from the list."""
    markers_open = [
        kOpenCircle,
        kOpenSquare,
        kOpenCross,
        kOpenDiamond,
        kOpenCrossX,
        kOpenFourTrianglesPlus,
        kOpenStar,
        kOpenThreeTriangles,
        kOpenFourTrianglesX,
        kOpenDoubleDiamond,
    ]
    markers_full = [
        kFullCircle,
        kFullSquare,
        kFullCross,
        kFullDiamond,
        kFullCrossX,
        kFullFourTrianglesPlus,
        kFullStar,
        kFullThreeTriangles,
        kFullFourTrianglesX,
        kFullDoubleDiamond,
    ]
    markers_thick = [88, 72, 75, 74, 76, 80, 82, 83, 84, 85]
    if option == 1:
        list_markers = markers_thick
    elif option == 2:
        list_markers = markers_full
    else:
        list_markers = markers_open
    return list_markers[i % len(list_markers)]


def get_markersize(marker: int, size_def=1.5):
    """Return a marker size."""
    markers_small = [
        kOpenCross,
        kOpenDiamond,
        kOpenStar,
        kOpenDoubleDiamond,
        kOpenFourTrianglesPlus,
        kOpenCrossX,
        kOpenThreeTriangles,
        kOpenFourTrianglesX,
        kFullCross,
        kFullDiamond,
        kFullStar,
        kFullDoubleDiamond,
        kFullFourTrianglesPlus,
        kFullCrossX,
        kFullThreeTriangles,
        kFullFourTrianglesX,
        75,
        74,
        76,
        83,
        84,
        85,
    ]
    if marker in markers_small:
        return size_def * 4 / 3
    return size_def


def setup_histogram(hist, colour=1, markerstyle=kOpenCircle, size=1.5, textsize=0.05, scale_title=1.3):
    hist.SetStats(0)
    hist.GetXaxis().SetLabelSize(textsize)
    hist.GetYaxis().SetLabelSize(textsize)
    hist.SetTitleSize(textsize * scale_title, "X")
    hist.SetTitleSize(textsize * scale_title, "Y")
    hist.SetTitleOffset(1.0, "X")
    hist.SetTitleOffset(1.0, "Y")
    hist.SetLineWidth(3)
    hist.SetLineColor(colour)
    hist.SetMarkerSize(size)
    hist.SetMarkerStyle(markerstyle)
    hist.SetMarkerColor(colour)


def setup_tgraph(tg_, colour=1, markerstyle=kOpenCircle, size=1.5, alphastyle=0.8, fillstyle=1001, textsize=0.05, scale_title=1.3):
    tg_.GetXaxis().SetLabelSize(textsize)
    tg_.GetYaxis().SetLabelSize(textsize)
    tg_.GetXaxis().SetTitleSize(textsize * scale_title)
    tg_.GetYaxis().SetTitleSize(textsize * scale_title)
    tg_.GetXaxis().SetTitleOffset(1.0)
    tg_.GetYaxis().SetTitleOffset(1.0)
    tg_.SetLineWidth(0)
    tg_.SetLineColor(colour)
    tg_.SetMarkerSize(size)
    tg_.SetMarkerStyle(markerstyle)
    tg_.SetMarkerColor(colour)
    if fillstyle:
        tg_.SetFillColorAlpha(colour, alphastyle)
        tg_.SetFillStyle(fillstyle)
        # If the marker colour is in the ALICE list, set a matching lighter fill colour.
        if colour in colours_alice_point:
            colour_fill = colours_alice_syst[colours_alice_point.index(colour)]
            tg_.SetFillColorAlpha(colour_fill, alphastyle)


def setup_canvas(can):
    can.SetCanvasSize(1970, 1500)
    can.SetWindowSize(570, 500)
    can.SetFillColor(0)
    can.SetTicks(1, 1)
    can.SetBottomMargin(0.12)
    can.SetLeftMargin(0.12)
    can.SetTopMargin(0.1)
    can.SetRightMargin(0.05)
    # can.cd()


def setup_legend(legend, textsize=0.035):
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(textsize)
    legend.SetTextFont(42)


def draw_latex(latex, colour=1, textsize=0.03):
    latex.SetNDC()
    latex.SetTextSize(textsize)
    latex.SetTextColor(colour)
    latex.SetTextFont(42)
    latex.Draw()


def draw_latex_lines(lines: "list[str]", x_start=0.18, y_start=0.85, y_step=0.05, font_size=0.04):
    list_latex = []
    for line in lines:
        latex = TLatex(x_start, y_start, line)
        list_latex.append(latex)
        draw_latex(latex, textsize=font_size, colour=1)
        y_start -= y_step
    return list_latex


def is_histogram(obj):
    return isinstance(obj, TH1)


def is_graph(obj):
    return isinstance(obj, TGraph)


def is_latex(obj):
    return isinstance(obj, TLatex)


def count_histograms(l_obj: list) -> int:
    """Get number of histograms in a list."""
    if not isinstance(l_obj, list):
        return 0
    return sum(is_histogram(o) for o in l_obj)


def count_graphs(l_obj: list) -> int:
    """Get number of graphs in a list."""
    if not isinstance(l_obj, list):
        return 0
    return sum(is_graph(o) for o in l_obj)


def make_plot(  # pylint: disable=too-many-arguments, too-many-branches, too-many-statements, too-many-locals
    name,
    can=None,
    pad=0,
    path=None,
    suffix="eps",
    title="",
    size=None,
    margins_c=None,
    list_obj=None,
    labels_obj=None,
    leg_pos=None,
    opt_leg_h="P",
    opt_leg_g="P",
    opt_plot_h="",
    opt_plot_g="P0",
    offsets_xy=None,
    maxdigits=3,
    colours=None,
    markers=None,
    sizes=None,
    range_x=None,
    range_y=None,
    margins_y=None,
    with_errors="xy",
    logscale=None,
    font_size=0.032,
    scale=1.,
    plot_order=None
):
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
    - font size relative to the full canvas height (font_size)
    - height of the pad relative to the full canvas height (scale)
    - order in which to plot objects (plot_order)
    """

    # HELPING FUNCTIONS

    def min0_gr(graph):
        """Get the minimum positive y value in the graph."""
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
        setup_tgraph(graph, get_my_colour(i_obj), get_my_marker(i_obj), get_my_size(i_obj), textsize=(font_size / scale))
        graph.SetTitle(title)
        graph.GetXaxis().SetLimits(x_min_plot, x_max_plot)
        graph.GetYaxis().SetRangeUser(y_min_plot, y_max_plot)
        graph.GetXaxis().SetMaxDigits(maxdigits)
        graph.GetYaxis().SetMaxDigits(maxdigits)
        graph.GetYaxis().SetTitleOffset(1.2)
        graph.GetXaxis().SetLabelOffset(0.005 / scale)
        graph.GetYaxis().SetLabelOffset(0.005)
        if offsets_xy:
            graph.GetXaxis().SetTitleOffset(offsets_xy[0])
            graph.GetYaxis().SetTitleOffset(offsets_xy[1] * scale)
        if leg and n_labels > i_obj and isinstance(labels_obj, list) and len(labels_obj[i_obj]) > 0:
            leg.AddEntry(graph, labels_obj[i_obj], opt_leg_g[counter_plot_g])
        graph.DrawClone(opt_plot_g[counter_plot_g] + "A" if counter_plot_obj == 0 else opt_plot_g[counter_plot_g])

    def plot_histogram(histogram):
        # If nothing has been plotted yet, plot an empty graph to set the exact ranges.
        if counter_plot_obj == 0:
            gr = TGraph(histogram)
            setup_tgraph(gr, textsize=(font_size / scale))
            gr.SetMarkerSize(0)
            gr.SetTitle(title)
            gr.GetXaxis().SetLimits(x_min_plot, x_max_plot)
            gr.GetYaxis().SetRangeUser(y_min_plot, y_max_plot)
            gr.GetXaxis().SetMaxDigits(maxdigits)
            gr.GetYaxis().SetMaxDigits(maxdigits)
            gr.GetYaxis().SetTitleOffset(1.2)
            gr.GetXaxis().SetLabelOffset(0.005 / scale)
            gr.GetYaxis().SetLabelOffset(0.005)
            if offsets_xy:
                gr.GetXaxis().SetTitleOffset(offsets_xy[0])
                gr.GetYaxis().SetTitleOffset(offsets_xy[1] * scale)
            gr.DrawClone("AP")
            list_new.append(gr)
        setup_histogram(histogram, get_my_colour(i_obj), get_my_marker(i_obj), get_my_size(i_obj), textsize=(font_size / scale))
        histogram.GetXaxis().SetLimits(x_min_plot, x_max_plot)
        histogram.GetXaxis().SetRangeUser(x_min_plot, x_max_plot)
        if leg and n_labels > i_obj and isinstance(labels_obj, list) and len(labels_obj[i_obj]) > 0:
            leg.AddEntry(histogram, labels_obj[i_obj], opt_leg_h[counter_plot_h])
        # print(f"Plotting {histogram.GetName()} with option {opt_plot_h[counter_plot_h]}")
        histogram.DrawCopy(opt_plot_h[counter_plot_h])

    def plot_latex(latex):
        draw_latex(latex)

    # BODY STARTS HERE

    if not (isinstance(list_obj, list) and len(list_obj) > 0):
        print("Error: Empty list of objects")
        return None, None

    list_new = []  # list of created objects that need to exist outside the function
    if not (isinstance(offsets_xy, list) and len(offsets_xy) == 2):
        offsets_xy = None
    if not isinstance(labels_obj, list):
        labels_obj = []
    n_labels = len(labels_obj)
    if margins_y is None:
        margins_y = [0.05, 0.05]
    if not isinstance(opt_leg_h, list):
        opt_leg_h = [opt_leg_h] * len(list_obj)
    if not isinstance(opt_leg_g, list):
        opt_leg_g = [opt_leg_g] * len(list_obj)
    if not isinstance(opt_plot_g, list):
        opt_plot_g = [opt_plot_g] * len(list_obj)
    if not isinstance(opt_plot_h, list):
        opt_plot_h = [opt_plot_h] * len(list_obj)

    # print(f"Objects {len(list_obj)}, options: {opt_leg_h}, {opt_leg_g}, {opt_plot_h}, {opt_plot_g}")

    # create and set canvas
    this_pad = None
    if can and pad > 0:
        print(f"Entering pad {pad} in canvas {can.GetName()}")
        this_pad = can.cd(pad)
    if not can:
        print(f"Creating new canvas with name {name}")
        can = TCanvas(name, name)
        this_pad = can
    setup_canvas(can)
    if isinstance(size, list) and len(size) == 2:
        can.SetCanvasSize(*size)
    # set canvas margins
    if isinstance(margins_c, list) and len(margins_c) > 0:
        for setter, value in zip(
            [can.SetBottomMargin, can.SetLeftMargin, can.SetTopMargin, can.SetRightMargin], margins_c
        ):
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
    list_h = []  # list of histograms
    list_g = []  # list of graphs
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

    # get y range of histograms within [x_min_plot, x_max_plot]
    y_min_h, y_max_h = float("inf"), float("-inf")
    if len(list_h) > 0:
        y_min_h, y_max_h = get_y_window_his(list_h, "y" in with_errors, [x_min_plot, x_max_plot])
        if log_y and y_min_h <= 0:
            y_min_h = min((h.GetMinimum(0) for h in list_h))
    # get y range of graphs within [x_min_plot, x_max_plot]
    y_min_g, y_max_g = float("inf"), float("-inf")
    if len(list_g) > 0:
        y_min_g, y_max_g = get_y_window_gr(list_g, "y" in with_errors, [x_min_plot, x_max_plot])
        if log_y and y_min_g <= 0:
            y_min_g = min((min0_gr(g) for g in list_g))
    # get total y range
    y_min = min(y_min_h, y_min_g)
    y_max = max(y_max_h, y_max_g)
    # get plotting y range
    y_min_plot, y_max_plot = y_min, y_max
    if isinstance(margins_y, list) and len(margins_y) == 2:
        y_min_plot, y_max_plot = get_plot_range(y_min, y_max, margins_y[0], margins_y[1], log_y)
    if isinstance(range_y, list) and len(range_y) == 2:
        y_min_plot, y_max_plot = range_y

    # append "same" to the histogram plotting option if needed
    for i, opt in enumerate(opt_plot_h):
        if not any(o in opt.lower() for o in ("same", "lego", "surf")):
            opt_plot_h[i] = opt + " same"

    # plot objects
    i_obj = -1  # index of the current object in the input lists
    counter_plot_obj = 0  # counter of plotted objects
    counter_plot_h = 0  # counter of plotted histograms
    counter_plot_g = 0  # counter of plotted graphs
    if not plot_order:
        plot_order = list(range(len(list_obj)))
    assert len(plot_order) == len(list_obj)
    assert list(dict.fromkeys(plot_order)) == plot_order  # Protect against duplicate indices.
    plot_index_order = sorted(plot_order)
    for i in plot_index_order:
        i_obj = plot_order.index(i)
        obj = list_obj[i_obj]
        if is_histogram(obj):
            plot_histogram(obj)
            counter_plot_h += 1
            counter_plot_obj += 1
        elif is_graph(obj):
            plot_graph(obj)
            counter_plot_g += 1
            counter_plot_obj += 1
        elif is_latex(obj):
            plot_latex(obj)
            counter_plot_obj += 1
        elif isinstance(obj, TObject):
            obj.Draw()
            counter_plot_obj += 1
        else:
            continue

    # plot axes on top
    assert this_pad is not None
    this_pad.RedrawAxis()

    # plot legend
    if leg:
        leg.Draw()

    # save canvas if necessary info provided
    if path and name and suffix:
        can.SaveAs(f"{path}/{name}.{suffix}")

    return can, list_new


def tg_sys(central, variations):
    shapebins_centres = []
    shapebins_contents = []
    shapebins_widths_up = []
    shapebins_widths_down = []
    shapebins_error_up = []
    shapebins_error_down = []

    for i in range(central.GetNbinsX()):
        shapebins_centres.append(central.GetBinCenter(i + 1))
        shapebins_contents.append(central.GetBinContent(i + 1))
        shapebins_widths_up.append(central.GetBinWidth(i + 1) * 0.5)
        shapebins_widths_down.append(central.GetBinWidth(i + 1) * 0.5)
        error_up = 0
        error_down = 0
        for j, _ in enumerate(variations):
            error = variations[j].GetBinContent(i + 1) - central.GetBinContent(i + 1)
            if error > 0 and error > error_up:
                error_up = error
            if error < 0 and abs(error) > error_down:
                error_down = abs(error)
        shapebins_error_up.append(error_up)
        shapebins_error_down.append(error_down)
    shapebins_centres_array = array("d", shapebins_centres)
    shapebins_contents_array = array("d", shapebins_contents)
    shapebins_widths_up_array = array("d", shapebins_widths_up)
    shapebins_widths_down_array = array("d", shapebins_widths_down)
    shapebins_error_up_array = array("d", shapebins_error_up)
    shapebins_error_down_array = array("d", shapebins_error_down)
    tg = TGraphAsymmErrors(
        central.GetNbinsX(),
        shapebins_centres_array,
        shapebins_contents_array,
        shapebins_widths_down_array,
        shapebins_widths_up_array,
        shapebins_error_down_array,
        shapebins_error_up_array,
    )
    return tg


def divide_histograms(h_num, h_den, errors_den=True):
    """Divide TGraphAsymmErrors"""
    n_bins = h_num.GetNbinsX()
    if h_den.GetNbinsX() != n_bins:
        print("Error: numbers of bins differ")
        return None
    h_rat = h_num.Clone(f"{h_num.GetName()}_ratio")
    for i in range(1, n_bins + 1):
        y_a = h_num.GetBinContent(i)
        e_a = h_num.GetBinError(i)
        y_b = h_den.GetBinContent(i)
        e_b = h_den.GetBinError(i)
        if not errors_den:
            e_b = 0.
        if abs(y_b) < 1.e-6:
            h_rat.SetBinContent(i, 0)
            h_rat.SetBinError(i, 0)
        else:
            r = y_a / y_b
            h_rat.SetBinContent(i, r)
            h_rat.SetBinError(i, math.sqrt(e_a * e_a + r * r * e_b * e_b) / y_b)
    return h_rat


def divide_graphs(gr_num, gr_den, errors_den=True):
    """Divide TGraphAsymmErrors"""
    n_points = gr_num.GetN()
    if gr_den.GetN() != n_points:
        print("Error: numbers of points differ")
        return None
    gr_rat = gr_num.Clone(f"{gr_num.GetName()}_ratio")
    for i in range(n_points):
        y_a = gr_num.GetPointY(i)
        e_a_plus = gr_num.GetErrorYhigh(i)
        e_a_minus = gr_num.GetErrorYlow(i)
        y_b = gr_den.GetPointY(i)
        e_b_plus = gr_den.GetErrorYhigh(i)
        e_b_minus = gr_den.GetErrorYlow(i)
        if not errors_den:
            e_b_plus = 0.
            e_b_minus = 0.
        if abs(y_b) < 1.e-6:
            gr_rat.SetPointY(i, 0)
            gr_rat.SetPointEYhigh(i, 0)
            gr_rat.SetPointEYlow(i, 0)
        else:
            r = y_a / y_b
            gr_rat.SetPointY(i, r)
            gr_rat.SetPointEYhigh(i, math.sqrt(e_a_plus * e_a_plus + r * r * e_b_minus * e_b_minus) / y_b)
            gr_rat.SetPointEYlow(i, math.sqrt(e_a_minus * e_a_minus + r * r * e_b_plus * e_b_plus) / y_b)
    return gr_rat


def make_ratios(l_his, l_gr, i_ref, errors_den=True):
    assert i_ref < len(l_his)
    l_ratio_h = [divide_histograms(h, l_his[i_ref], errors_den) for h in l_his]
    l_ratio_g = []
    if l_gr:
        assert i_ref < len(l_gr)
        l_ratio_g = [divide_graphs(g, l_gr[i_ref], errors_den) for g in l_gr]
    return l_ratio_h, l_ratio_g


def scale_graph(graph, number: float):
    """
    Scale TGraphAsymmErrors
    """
    if not graph:
        return
    for i in range(graph.GetN()):
        graph.SetPointY(i, graph.GetPointY(i) * number)
        graph.SetPointEYhigh(i, graph.GetErrorYlow(i) * number)
        graph.SetPointEYlow(i, graph.GetErrorYhigh(i) * number)


def reset_hist_outside_range(hist, x_min: float, x_max: float, val_reset=0.0):
    """Reset bins of histogram hist outside [x_min, x_max] to val_reset."""
    for i_bin in range(1, hist.GetNbinsX() + 1):
        if round(hist.GetBinLowEdge(i_bin + 1), 2) <= x_min or round(hist.GetBinLowEdge(i_bin), 2) >= x_max:
            hist.SetBinContent(i_bin, val_reset)
            hist.SetBinError(i_bin, 0.0)


def reset_graph_outside_range(graph, x_min: float, x_max: float, val_reset=0.0):
    """Reset points of graph outside [x_min, x_max] to val_reset."""
    for i_point in range(graph.GetN()):
        if (
            round(graph.GetPointX(i_point) + graph.GetErrorXhigh(i_point), 2) <= x_min
            or round(graph.GetPointX(i_point) - graph.GetErrorXlow(i_point), 2) >= x_max
        ):
            graph.SetPointY(i_point, val_reset)
            graph.SetPointEYlow(i_point, 0.0)
            graph.SetPointEYhigh(i_point, 0.0)


def sqrt_sum_sq(numbers):
    """Return sqrt(n1 * n1 + n2 * n2 + ...)"""
    return math.sqrt(sum(n * n for n in numbers))


def combine_graphs(graphs: list):
    """Combine uncertainties of TGraphAsymmErrors graphs in quadrature"""
    if not graphs:
        print("Error: No graphs given")
        sys.exit(1)
    gr_comb = graphs[0].Clone(f"{graphs[0].GetName()}_combined")
    if len(graphs) > 1:
        n_points = gr_comb.GetN()
        for gr in graphs:
            if gr.GetN() != n_points:
                print(f"Error: Different number of points: {gr.GetN()} vs {n_points}")
                sys.exit(1)
        for i in range(n_points):
            x_comb = gr_comb.GetPointX(i)
            y_comb = gr_comb.GetPointY(i)
            for gr in graphs:
                if gr.GetPointX(i) != x_comb:
                    print(f"Error: Different x-value of point {i}: {x_comb} vs {gr.GetPointX(i)}")
                    sys.exit(1)
                if gr.GetPointY(i) != y_comb:
                    print(f"Error: Different y-value of point {i}: {y_comb} vs {gr.GetPointY(i)}")
                    sys.exit(1)
            gr_comb.SetPointEYhigh(i, sqrt_sum_sq([gr.GetErrorYhigh(i) for gr in graphs]))
            gr_comb.SetPointEYlow(i, sqrt_sum_sq([gr.GetErrorYlow(i) for gr in graphs]))
    return gr_comb


def get_mean_hist(hist):
    """Get the mean x of a histogram."""
    sum_xy = 0
    sum_y = 0
    for i in range(hist.GetNbinsX()):
        sum_xy += hist.GetBinContent(i + 1) * hist.GetBinCenter(i + 1)
        sum_y += hist.GetBinContent(i + 1)
    return sum_xy / sum_y


def get_mean_graph(graph):
    """Get the mean x of a graph."""
    sum_xy = 0
    sum_y = 0
    for i in range(graph.GetN()):
        sum_xy += graph.GetPointY(i) * graph.GetPointX(i)
        sum_y += graph.GetPointY(i)
    return sum_xy / sum_y


def get_mean_uncertainty(his_stat, gr_syst=None, n_var=1000, combine=False):
    """Get the uncertainty of a distribution's mean by varying y-values within their uncertainties.
    his_stat is the histogram with stat. unc.
    gr_syst is the graphs with syst. unc.
    n_var is the number of variations
    combine is the switch for combining stat and syst uncertainties in quadrature
    Returns histograms with the distributions of means.
    """
    # combine stat. and syst. uncertainties in quadrature
    gr_stat = TGraphAsymmErrors(his_stat)
    gr_comb = None
    if gr_syst and combine:
        gr_comb = combine_graphs([gr_stat, gr_syst])
    n_points = gr_stat.GetN()
    # create a new empty histogram/graph for the variations
    hist_var_stat = his_stat.Clone(f"{his_stat.GetName()}_var_stat")
    hist_var_stat.Reset()
    hist_var_syst = his_stat.Clone(f"{his_stat.GetName()}_var_syst")
    hist_var_syst.Reset()
    hist_var_comb = his_stat.Clone(f"{his_stat.GetName()}_var_comb")
    hist_var_comb.Reset()
    rnd = TRandom3()
    # create a histograms for the distribution of means
    hist_means_stat = TH1F(
        f"{his_stat.GetName()}_means_stat",
        "hist_means_stat",
        1000,
        his_stat.GetXaxis().GetXmin(),
        his_stat.GetXaxis().GetXmax(),
    )
    hist_means_stat.Sumw2()
    hist_means_syst = TH1F(
        f"{his_stat.GetName()}_means_syst",
        "hist_means_syst",
        1000,
        his_stat.GetXaxis().GetXmin(),
        his_stat.GetXaxis().GetXmax(),
    )
    hist_means_syst.Sumw2()
    hist_means_comb = TH1F(
        f"{his_stat.GetName()}_means_comb",
        "hist_means_comb",
        1000,
        his_stat.GetXaxis().GetXmin(),
        his_stat.GetXaxis().GetXmax(),
    )
    hist_means_comb.Sumw2()
    for _ in range(n_var):
        for i in range(n_points):
            # vary y with Gaussian of mean = y_central, sigma = err_y_central
            y_var_stat = rnd.Gaus(gr_stat.GetPointY(i), (gr_stat.GetErrorYhigh(i) + gr_stat.GetErrorYlow(i)) / 2.0)
            hist_var_stat.SetBinContent(i + 1, y_var_stat)
            if gr_syst:
                y_var_syst = rnd.Gaus(gr_syst.GetPointY(i), (gr_syst.GetErrorYhigh(i) + gr_syst.GetErrorYlow(i)) / 2.0)
                hist_var_syst.SetBinContent(i + 1, y_var_syst)
                if gr_comb:
                    y_var_comb = rnd.Gaus(
                        gr_comb.GetPointY(i), (gr_comb.GetErrorYhigh(i) + gr_comb.GetErrorYlow(i)) / 2.0
                    )
                    hist_var_comb.SetBinContent(i + 1, y_var_comb)
        # calculate mean of the new histogram
        # fill the mean in a histogram of means
        hist_means_stat.Fill(get_mean_hist(hist_var_stat))
        if gr_syst:
            hist_means_syst.Fill(get_mean_hist(hist_var_syst))
            if gr_comb:
                hist_means_comb.Fill(get_mean_hist(hist_var_comb))
    return hist_means_stat, hist_means_syst, hist_means_comb


def format_number_prec(num, prec):
    """Format a number with a given decimal precision."""
    return f"{round(num, prec):.{max(0, prec)}f}"


def format_value_with_unc(y, e_stat=None, e_syst_plus=None, e_syst_minus=None, n_sig=2):  # pylint: disable=invalid-name
    """Format a value with uncertainties so that the main value is reported with a decimal precision
    given by the number of significant figures of the smallest uncertainty."""
    mag_y = math.floor(math.log10(y))
    mag_e_stat = mag_y
    mag_e_syst = mag_y
    if e_stat:
        mag_e_stat = math.floor(math.log10(e_stat))
    if e_syst_plus:
        if not e_syst_minus:
            e_syst_minus = e_syst_plus
        mag_e_syst = math.floor(math.log10(min(e_syst_plus, e_syst_minus)))
    mag_y = min(mag_y, mag_e_stat, mag_e_syst)
    # print(f"Mag stat {mag_e_stat}, sys {mag_e_syst}")
    prec_y = n_sig - 1 - mag_y
    prec_stat = n_sig - 1 - mag_e_stat
    prec_syst = n_sig - 1 - mag_e_syst
    str_value = format_number_prec(y, prec_y)
    if e_stat:
        str_value += f" ± {format_number_prec(e_stat, prec_stat)} (stat.)"
    if e_syst_plus:
        str_e_syst_plus = format_number_prec(e_syst_plus, prec_syst)
        str_e_syst_minus = format_number_prec(e_syst_minus, prec_syst)
        if str_e_syst_plus == str_e_syst_minus:
            str_value += f" ± {str_e_syst_plus} (syst.)"
        else:
            str_value += f" +{str_e_syst_plus} −{str_e_syst_minus} (syst.)"
    return str_value


def print_histogram(hist, verbose=True):
    if verbose:
        n_bins = hist.GetNbinsX()
        print(f"{hist.GetName()}, {n_bins} bins, {hist.GetXaxis().GetXmin()} to {hist.GetXaxis().GetXmax()}")
        print([hist.GetBinContent(i + 1) for i in range(n_bins)])
