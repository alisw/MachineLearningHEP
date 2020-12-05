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
Common utility functions for fitting.
Interfacing with
    1. OS / serialization of fitters
    2. user configuration database
Providing and storing fitters
"""
from os.path import join
from math import ceil
import inspect

# pylint: disable=import-error, no-name-in-module, unused-import
from ROOT import TFile

from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict, checkdir
from machine_learning_hep.logger import get_logger


def construct_rebinning(histo, rebin):

    try:
        iter(rebin)
        min_rebin = rebin[0]
        rebin_min_entries_per_bin = rebin[1]
        max_rebin = rebin[2]
        entries_per_bin = histo.Integral() / histo.GetNbinsX()
        rebin = rebin_min_entries_per_bin / entries_per_bin
        if rebin > max_rebin:
            return max_rebin
        if min_rebin and min_rebin < rebin:
            return min_rebin
        if rebin < 1:
            return None
        return ceil(rebin)
    except TypeError:
        return rebin


def save_fit(fit, save_dir, annotations=None):

    if not fit.has_attempt:
        get_logger().warning("Fit has not been done and will hence not be saved")
        return

    checkdir(save_dir)

    root_file_name = join(save_dir, "root_objects.root")
    root_file = TFile.Open(root_file_name, "RECREATE")
    root_file.cd()

    for name, root_object in fit.root_objects.items():
        if root_object:
            root_object.Write(name)
    fit.kernel.Write("kernel")
    root_file.Close()

    yaml_path = join(save_dir, "init_pars.yaml")
    dump_yaml_from_dict(fit.init_pars, yaml_path)

    yaml_path = join(save_dir, "fit_pars.yaml")
    dump_yaml_from_dict(fit.fit_pars, yaml_path)

    class_name = fit.__class__.__name__
    meta_info = {"fit_class": class_name,
                 "success": fit.success}
    if annotations:
        meta_info["annotations"] = annotations

    yaml_path = join(save_dir, "meta.yaml")
    dump_yaml_from_dict(meta_info, yaml_path)


def load_fit(save_dir):
    yaml_path = join(save_dir, "meta.yaml")
    meta_info = parse_yaml(yaml_path)

    yaml_path = join(save_dir, "init_pars.yaml")

    #pylint: disable=import-outside-toplevel
    import machine_learning_hep.fitting.fitters as search_module
    #pylint: enable=import-outside-toplevel
    fit_classes = {f[0]: getattr(search_module, f[0]) \
            for f in inspect.getmembers(search_module, inspect.isclass) \
            if f[1].__module__ == search_module.__name__}
    fit = None
    if meta_info["fit_class"] in fit_classes:
        fit = fit_classes[meta_info["fit_class"]](parse_yaml(yaml_path))
    else:
        get_logger().fatal("Fit class %s is invalid")

    yaml_path = join(save_dir, "fit_pars.yaml")
    fit.fit_pars = parse_yaml(yaml_path)

    root_file_name = join(save_dir, "root_objects.root")
    root_file = TFile.Open(root_file_name, "READ")

    keys = root_file.GetListOfKeys()

    root_objects = {}
    for k in keys:
        if k.GetName() == "kernel":
            fit.kernel = k.ReadObj()
            continue
        obj = k.ReadObj()
        obj.SetDirectory(0)
        root_objects[k.GetName()] = obj
    root_file.Close()

    fit.set_root_objects(root_objects)
    fit.success = meta_info["success"]
    fit.init_fit()

    if "annotations" not in meta_info:
        return fit
    return fit, meta_info["annotations"]
