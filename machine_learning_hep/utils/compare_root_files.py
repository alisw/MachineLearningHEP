#!/usr/bin/env python3

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

"""!
@brief Compare histogram-like objects between two ROOT files.

@author Vít Kučera <vit.kucera@cern.ch>
@date   2025-03-03
"""

# pylint: disable=too-many-return-statements,too-many-branches,too-many-statements

import argparse
import math
import sys
from enum import Enum

from ROOT import (
    TH1,
    TH2,
    TH3,
    RooUnfoldResponse,
    TAxis,
    TCanvas,
    TColor,
    TDirectoryFile,
    TFile,
    THnSparse,
    THnT,
    TLegend,
    gROOT,
)


class ObjectType(Enum):
    UNKNOWN = 0
    TH_1 = 1
    TH_2 = 2
    TH_3 = 3
    TH_N_T = 4
    TH_N_SPARSE = 5
    RESPONSE = 6


def msg_err(message: str):
    """Print error message."""
    print(f"Error: {message}")


def msg_fatal(message: str):
    """Print error message and exit."""
    print(f"Fatal: {message}")
    sys.exit(1)


def list_recursive(
    file: TDirectoryFile, objects: dict | None = None, path_dir: str = "", verbose: bool = False
) -> dict:
    """Recursively load objects from a ROOT file into a dictionary."""
    if objects is None:
        objects = {}
    for key in file.GetListOfKeys():
        name_obj = key.GetName()
        name_class = key.GetClassName()
        path_obj = f"{path_dir + '/' if path_dir else ''}{name_obj}"
        obj = file.Get(name_obj)
        if verbose:
            print(f"{path_obj}: {name_class}")
        if isinstance(obj, TDirectoryFile):
            list_recursive(obj, objects, path_obj, verbose)
        else:
            objects[path_obj] = obj
    return objects


def are_valid(*objects) -> bool:
    """Check whether objects exist."""
    result = True
    for i, o in enumerate(objects):
        if not o:
            msg_err(f"Bad object {i}")
            result = False
    return result


def are_same_values(val1, val2, mag_epsilon: None | int = None) -> bool:
    """Compare two values, even if they are NaN."""
    return (
        (val1 == val2)
        or (val1 is val2)
        or (math.isnan(val1) and math.isnan(val2))
        or (diff_rel(val1, val2) < 10**mag_epsilon if isinstance(mag_epsilon, int) else False)
    )


def diff_rel(val1, val2) -> float:
    """Calculate relative difference between two numbers."""
    if are_same_values(val1, val2):
        return 0.0
    return abs((val2 - val1) / (val1 + val2))


def compare_values(val1: float | tuple[float, float], val2: float | tuple[float, float]) -> str:
    """Format a string comparing two values (with optional uncertainties)
    and report also the relative difference(s).
    """
    unc1, unc2 = None, None
    if isinstance(val1, tuple):
        unc1 = val1[1]
        val1 = val1[0]
    if isinstance(val2, tuple):
        unc2 = val2[1]
        val2 = val2[0]
    string_val1 = f"{val1}{f' ± {unc1}' if unc1 is not None else ''}"
    string_val2 = f"{val2}{f' ± {unc2}' if unc2 is not None else ''}"
    string_diff = f"{diff_rel(val1, val2)}{'' if unc1 is None or unc2 is None else f' ± {diff_rel(unc1, unc2)}'}"
    return f"{string_val1} vs {string_val2}, rel. diff.: {string_diff}"


def are_same_axes(axis1, axis2) -> bool:
    """Tell whether two axes are same."""
    if not are_valid(axis1, axis2):
        msg_fatal("Bad input objects")
        return False
    # Check classes
    for i, o in enumerate((axis1, axis2)):
        if not isinstance(o, TAxis):
            msg_fatal(f"Object {i} is not an axis")
            return False
    # Check number of bins
    n_bins1, n_bins2 = axis1.GetNbins(), axis2.GetNbins()
    if n_bins1 != n_bins2:
        return False
    # Check bin arrays
    array1 = [axis1.GetBinLowEdge(i + 1) for i in range(n_bins1 + 1)]
    array2 = [axis2.GetBinLowEdge(i + 1) for i in range(n_bins2 + 1)]
    if array1 != array2:
        return False
    return True


def are_same_histograms(his1: TH1, his2: TH1, mag_epsilon: None | int = None) -> bool:
    """Tell whether two histograms are same."""
    if not are_valid(his1, his2):
        msg_fatal("Bad input objects")
        return False
    # Compare number of entries
    if not are_same_values(his1.GetEntries(), his2.GetEntries(), mag_epsilon):
        print(f"Different number of entries {compare_values(his1.GetEntries(), his2.GetEntries())}")
        return False
    # Compare axes
    for ax1, ax2 in zip(
        (his1.GetXaxis(), his1.GetYaxis(), his1.GetZaxis()), (his2.GetXaxis(), his2.GetYaxis(), his2.GetZaxis())
    ):
        if not are_same_axes(ax1, ax2):
            print("Different axes")
            return False
    # Compare bin counts and errors (include under/overflow bins)
    for bin_z in range(his1.GetNbinsZ() + 2):
        for bin_y in range(his1.GetNbinsY() + 2):
            for bin_x in range(his1.GetNbinsX() + 2):
                i_bin = his1.GetBin(bin_x, bin_y, bin_z)
                if not are_same_values(
                    his1.GetBinContent(i_bin), his2.GetBinContent(i_bin), mag_epsilon
                ) or not are_same_values(his1.GetBinError(i_bin), his2.GetBinError(i_bin), mag_epsilon):
                    print(
                        f"Different bin {i_bin} content: "
                        + compare_values(
                            (his1.GetBinContent(i_bin), his1.GetBinError(i_bin)),
                            (his2.GetBinContent(i_bin), his2.GetBinError(i_bin)),
                        )
                    )
                    return False
    return True


def are_same_thnspare(his1: THnSparse, his2: THnSparse, mag_epsilon: None | int = None) -> bool:
    """Tell whether two THnSparse objects are same."""
    if not are_valid(his1, his2):
        msg_fatal("Bad input objects")
        return False
    # Compare number of dimensions
    if his1.GetNdimensions() != his2.GetNdimensions():
        print(f"Different number of dimensions {his1.GetNdimensions()} vs {his2.GetNdimensions()}")
        return False
    # Compare number of entries
    if not are_same_values(his1.GetEntries(), his2.GetEntries(), mag_epsilon):
        print(f"Different number of entries {compare_values(his1.GetEntries(), his2.GetEntries())}")
        return False
    # Compare number of filled bins
    if his1.GetNbins() != his2.GetNbins():
        print(f"Different number of filled bins {his1.GetNbins()} vs {his2.GetNbins()}")
        return False
    # Compare axes
    for iAx in range(his1.GetNdimensions()):
        if not are_same_axes(his1.GetAxis(iAx), his2.GetAxis(iAx)):
            print("Different axes")
            return False
    # Compare bin content
    for i_bin in range(1, his1.GetNbins() + 1):
        if not are_same_values(
            his1.GetBinContent(i_bin), his2.GetBinContent(i_bin), mag_epsilon
        ) or not are_same_values(his1.GetBinError(i_bin), his2.GetBinError(i_bin), mag_epsilon):
            print(
                f"Different bin {i_bin} content: "
                + compare_values(
                    (his1.GetBinContent(i_bin), his1.GetBinError(i_bin)),
                    (his2.GetBinContent(i_bin), his2.GetBinError(i_bin)),
                )
            )
            return False
    return True


def are_same_response(his1: RooUnfoldResponse, his2: RooUnfoldResponse, mag_epsilon: None | int = None) -> bool:
    """Tell whether two RooUnfoldResponse objects are same."""
    if not are_valid(his1, his2):
        msg_fatal("Bad input objects")
        return False
    # Compare number of dimensions
    if (
        his1.GetDimensionMeasured() != his2.GetDimensionMeasured()
        or his1.GetDimensionTruth() != his2.GetDimensionTruth()
    ):
        return False
    # Compare number of bins
    if his1.GetNbinsMeasured() != his2.GetNbinsMeasured() or his1.GetNbinsTruth() != his2.GetNbinsTruth():
        return False
    # Compare axes and bin content
    if not are_same_histograms(his1.Hfakes(), his2.Hfakes(), mag_epsilon):
        return False
    if not are_same_histograms(his1.Hmeasured(), his2.Hmeasured(), mag_epsilon):
        return False
    if not are_same_histograms(his1.Htruth(), his2.Htruth(), mag_epsilon):
        return False
    if not are_same_histograms(his1.Hresponse(), his2.Hresponse(), mag_epsilon):
        return False
    return True


def get_object_type(obj) -> ObjectType:
    """Return histogram degree."""
    for num, tp in zip(
        (
            ObjectType.RESPONSE,
            ObjectType.TH_N_SPARSE,
            ObjectType.TH_N_T,
            ObjectType.TH_3,
            ObjectType.TH_2,
            ObjectType.TH_1,
        ),
        (RooUnfoldResponse, THnSparse, THnT(float), TH3, TH2, TH1),
    ):
        if isinstance(obj, tp):
            return num
    return ObjectType.UNKNOWN


def are_same_objects(obj1, obj2, mag_epsilon: None | int = None) -> bool:
    """Tell whether two histogram-like objects are same."""
    if not are_valid(obj1, obj2):
        msg_fatal("Bad input objects")
        return False
    # Compare types
    passed = True
    if type(obj1) is not type(obj2):
        print(f"Different types {obj1.ClassName()} vs {obj2.ClassName()}")
        passed = False
    # Get ROOT types
    list_type = [get_object_type(o) for o in (obj1, obj2)]
    # Compare ROOT types (is it not covered by type(obj)?)
    if list_type[0] is not list_type[1]:
        print(f"Different types {list_type[0]} vs {list_type[1]}")
        return False
    type_obj = list_type[0]
    # Compare supported ROOT objects
    if type_obj is ObjectType.RESPONSE:
        return are_same_response(obj1, obj2, mag_epsilon) and passed
    if type_obj in (ObjectType.TH_N_T, ObjectType.TH_N_SPARSE):
        return are_same_thnspare(obj1, obj2, mag_epsilon) and passed
    if type_obj in (ObjectType.TH_1, ObjectType.TH_2, ObjectType.TH_3):
        return are_same_histograms(obj1, obj2, mag_epsilon) and passed
    print(f"Objects have an unsupported type {type(obj1)}.")
    raise NotImplementedError


def are_same_files(
    dict_obj: dict, verbose: bool = False, diff_only: bool = False, mag_epsilon: None | int = None
) -> tuple[bool, bool, bool, bool, dict[str, bool]]:
    """Compare file contents.

    Contents are provided in a dictionary of objects.
    Returns a tuple of booleans: (same_structure, common_content, compared_all, same_content)
    and a dictionary with bool flags whether compared objects are same.
    """

    print("\nComparing file content")

    same_structure = True
    common_content = True
    compared_all = True
    same_content = True
    dict_results: dict[str, bool] = {}
    list_files = list(dict_obj.keys())
    if (n_files := len(list_files)) < 2:
        print(f"Got {n_files} files. Returning.")
        return (False, False, False, False, dict_results)
    name_file_1 = list_files[0]
    name_file_2 = list_files[1]
    dict_obj_1 = dict_obj[name_file_1]
    dict_obj_2 = dict_obj[name_file_2]

    # Compare lists of objects.
    list_names_1 = list(dict_obj_1.keys())
    list_names_2 = list(dict_obj_2.keys())
    list_names_common = sorted(set(list_names_1).intersection(list_names_2))
    list_names_only_1 = sorted(set(list_names_1).difference(list_names_2))
    list_names_only_2 = sorted(set(list_names_2).difference(list_names_1))
    for l_n, name_file in zip((list_names_only_1, list_names_only_2), (name_file_1, name_file_2)):
        if l_n:
            print(f"\nThese objects are only in file {name_file}:")
            print(l_n)
    if list_names_only_1 or list_names_only_2:
        same_structure = False

    # Numeric comparison
    print("\nComparing common objects.")
    common_content = bool(list_names_common)
    compared_all, same_content = common_content, common_content
    for key_obj in list_names_common:
        obj_1 = dict_obj_1[key_obj]
        obj_2 = dict_obj_2[key_obj]

        # Compare two objects.
        if verbose:
            print(f"Comparing {key_obj}")
        msg_base = f"Objects {key_obj}"
        try:
            if same_objects := are_same_objects(obj_1, obj_2, mag_epsilon):
                if not diff_only:
                    print(f"{msg_base} are same ({obj_1.GetEntries()} entries).")
            else:
                print(f"{msg_base} are different.")
                same_content = False
            dict_results[key_obj] = same_objects
        except NotImplementedError:
            compared_all = False
            print("Skipping")
    return (same_structure, common_content, compared_all, same_content, dict_results)


def make_plots(
    dict_obj: dict, dict_result: dict, verbose: bool = False, diff_only: bool = False, normalize: bool = False
):
    """Plot compared objects and their ratios."""

    print("\nPlotting")

    list_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    list_markers = [21, 20, 34]
    dict_colors: dict[str, int] = {}
    dict_markers: dict[str, int] = {}
    dict_list_canvas: dict[str, list] = {}

    # Drawing objects
    is_first_file = True
    key_file_first = ""
    for key_file, dict_file in dict_obj.items():
        i_file = len(dict_colors) + 1
        print("Entry", i_file, key_file)
        dict_colors[key_file] = TColor.GetColor(list_colors[len(dict_colors)])
        dict_markers[key_file] = list_markers[len(dict_markers)]
        if is_first_file:
            key_file_first = key_file
        for key_obj, obj in dict_file.items():
            if get_object_type(obj) not in (ObjectType.TH_1, ObjectType.TH_2):
                continue
            if diff_only and dict_result.get(key_obj, False):
                continue
            # Make the main canvas.
            opt = "LP"
            if key_obj not in dict_list_canvas:
                dict_list_canvas[key_obj] = [TCanvas(key_obj, key_obj), None]
            else:
                opt += "same"
            list_canvas = dict_list_canvas[key_obj]
            list_canvas[0].cd()
            # print(f'Drawing {obj.GetName()} with opt "{opt}" on canvas {gPad.GetName()}')
            obj.SetLineColor(dict_colors[key_file])
            obj.SetMarkerStyle(dict_markers[key_file])
            obj.SetMarkerColor(dict_colors[key_file])
            obj.SetBit(TH1.kNoTitle)
            obj.SetBit(TH1.kNoStats)
            obj.SetTitle(str(i_file))
            if normalize:
                obj_plot = obj.DrawNormalized(opt)
            else:
                obj_plot = obj.DrawClone(opt)
            list_canvas.append(obj_plot)
            # Make ratio.
            if not is_first_file and key_obj in dict_obj[key_file_first]:
                list_canvas[1] = TCanvas(f"{key_obj}_ratio", f"{key_obj}_ratio")
                list_canvas[1].cd()
                # print(f'Drawing {obj.GetName()} with opt "{opt}" on canvas {gPad.GetName()}')
                # line_1 = TLine(obj.GetXaxis().GetXmin(), 1, obj.GetXaxis().GetXmax(), 1)
                obj_ratio = obj.Clone(f"{obj.GetName()}_ratio")
                obj_ratio.SetTitle("ratio")
                obj_ratio.Divide(dict_obj[key_file_first][key_obj])
                list_canvas.append(obj_ratio.DrawClone(opt))
                # list_canvas.append(line_1.Draw())
        is_first_file = False
    # Make legends.
    for list_canvas in dict_list_canvas.values():
        for i in (0, 1):
            if not (can := list_canvas[i]):
                continue
            can.cd()
            leg = TLegend(0.1, 0.9, 0.7, 0.99, can.GetName())
            leg.SetNColumns(2)
            list_canvas.append(leg)
            for prim in can.GetListOfPrimitives():
                leg.AddEntry(prim)
            leg.Draw()

    # Save canvases in a file.
    first = True
    can_first = None
    for key_obj, list_canvas in sorted(dict_list_canvas.items()):
        can = list_canvas[0]
        if verbose:
            print(key_obj)
        if first:
            can_first = can
            can_first.SaveAs("Comparison.pdf[")
            first = False
        can.SaveAs("Comparison.pdf")
        if can_rat := list_canvas[1]:
            can_rat.SaveAs("Comparison.pdf")
    if can_first:
        can_first.SaveAs("Comparison.pdf]")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare histogram-like objects between two ROOT files.")
    parser.add_argument("file_1", type=str, help="first ROOT file")
    parser.add_argument("file_2", type=str, help="second ROOT file")
    parser.add_argument("-v", action="store_true", help="verbose mode")
    parser.add_argument("-p", action="store_true", help="plot objects")
    parser.add_argument("-d", action="store_true", help="report and plot only different objects")
    parser.add_argument(
        "-t", type=int, help="tolerance (order of magnitude of the maximum acceptable relative difference of values)"
    )

    args = parser.parse_args()
    path_file_1 = args.file_1
    path_file_2 = args.file_2
    verbose = args.v
    plot = args.p
    diff_only = args.d
    mag_epsilon = None if args.t is None else args.t

    gROOT.SetBatch(True)
    gROOT.ProcessLine("gErrorIgnoreLevel = 1001;")  # suppress INFO messages

    # Abort if paths are same.
    if path_file_1 == path_file_2:
        msg_fatal("File paths are same.")

    # Load objects.
    objects = {}
    with TFile(path_file_1) as file_1, TFile(path_file_2) as file_2:
        for i, (path_i, file_i) in enumerate(zip((path_file_1, path_file_2), (file_1, file_2))):
            key_i = path_i
            # For testing purposes, treat identical files as different.
            if path_file_1 == path_file_2:
                key_i += f"_{i + 1}"
            print(f"\nLoading objects from file {path_i}.")
            objects[key_i] = list_recursive(file_i, verbose=verbose)

        # Compare objects.
        same_structure, common_content, compared_all, same_content, dict_result = are_same_files(
            objects, verbose, diff_only, mag_epsilon
        )

        # Report results.
        string_tolerance = str(None) if mag_epsilon is None else f"1e{mag_epsilon}"
        print(f"\nSame structure:\t\t\t\t\t{same_structure}")
        print(f"Common content:\t\t\t\t\t{common_content}")
        print(f"Compared all common content:\t{compared_all}")
        print(f"Same compared content:\t\t\t{same_content} (tolerance {string_tolerance})")
        print(f"Files are same:\t\t\t\t\t{all((same_structure, common_content, compared_all, same_content))}")

        # Plot objects.
        if plot:
            make_plots(objects, dict_result, verbose, diff_only, normalize=False)


if __name__ == "__main__":
    main()
