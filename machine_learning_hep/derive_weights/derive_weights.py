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


import sys
from glob import glob
import multiprocessing as mp
import argparse
import pickle

import pandas as pd
import yaml
from lz4 import frame # pylint: disable=unused-import

from root_numpy import fill_hist # pylint: disable=import-error

from ROOT import TFile, TH1F, TH2F # pylint: disable=import-error

from machine_learning_hep.utilities import openfile
from machine_learning_hep.io import parse_yaml
from machine_learning_hep.do_variations import modify_dictionary


# Needed here for multiprocessing
INV_MASS = [None]
INV_MASS_WINDOW = [None]


def only_one_evt(df_in, dupl_cols):
    return df_in.drop_duplicates(dupl_cols)

def read_database(path, overwrite_path=None):
    data_param = None
    with open(path, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    case = list(data_param.keys())[0]
    data_param = data_param[case]
    if overwrite_path:
        overwrite_db = None
        with open(overwrite_path, 'r') as param_config:
            overwrite_db = yaml.load(param_config, Loader=yaml.FullLoader)
        modify_dictionary(data_param, overwrite_db)
    return case, data_param

def summary_histograms_and_write(file_out, histos, histo_names,
                                 histo_xtitles, histo_ytitles):

    histos_added = histos[0]
    for h_list in histos[1:]:
        for h_added, h in zip(histos_added, h_list):
            h_added.Add(h)

    for h_add, name, xtitle, ytitle \
            in zip(histos_added, histo_names, histo_xtitles, histo_ytitles):
        h_add.SetName(name)
        h_add.SetTitle(name)
        h_add.GetXaxis().SetTitle(xtitle)
        h_add.GetYaxis().SetTitle(ytitle)

        file_out.WriteTObject(h_add)



def derive_mc(periods, in_top_dirs,
              gen_file_name,
              required_columns,
              distribution_column, distribution_x_range,
              file_out_name,
              queries_periods=None, query_all=None, queries_slices=None):

    """

    make n_tracklets distributions for all events

    """

    queries_periods = [None] * len(periods) if not queries_periods else queries_periods

    # Prepare histogram parameters
    queries_slices = [None] if not queries_slices else queries_slices
    histo_names = [f"{distribution_column}_{i}" for i in range(len(queries_slices))]

    histo_params = [([distribution_column], distribution_x_range) for _ in histo_names]

    histo_xtitles = [distribution_column] * len(histo_params)
    histo_ytitles = ["entries"] * len(histo_params)


    file_out = TFile.Open(file_out_name, "RECREATE")

    for period, dir_applied, query_period in zip(periods, in_top_dirs, queries_periods):
        query_tmp = None
        if query_all:
            query_tmp = query_all
            if query_period:
                query_tmp += f" and {query_period}"
        elif query_period:
            query_tmp = query_period

        files_all_evt = glob(f"{dir_applied}/**/{gen_file_name}", recursive=True)

        args = [((f_evt,), histo_params, required_columns, \
                query_tmp, None, None, queries_slices, None) \
                for f_evt in files_all_evt]

        histos = multi_proc(fill_from_pickles, args, None, 100, 30)

        histo_names_period = [f"{name}_{period}" for name in histo_names]
        summary_histograms_and_write(file_out, histos, histo_names_period,
                                     histo_xtitles, histo_ytitles)

    file_out.Close()

def derive_data(periods, in_top_dirs, gen_file_name, required_columns, distribution_column, # pylint: disable=too-many-arguments, too-many-branches
                distribution_x_range, file_name_mlwp_map, file_out_name,
                queries_periods=None, query_all=None, queries_slices=None):

    """

    make n_tracklets distributions for all events

    """

    queries_periods = [None] * len(periods) if not queries_periods else queries_periods

    # Prepare histogram parameters
    queries_slices = [None] if not queries_slices else queries_slices
    histo_names = [f"{distribution_column}_{i}" for i in range(len(queries_slices))]

    histo_params = [([distribution_column], distribution_x_range) for _ in histo_names]

    histo_xtitles = [distribution_column] * len(histo_params)
    histo_ytitles = ["entries"] * len(histo_params)

    #print("queries slices", queries_slices)
    #sys.exit(0)
    #print(histo_names)
    #print(histo_params)
    #print(histo_xtitles)
    #print(histo_ytitles)

    file_out = TFile.Open(file_out_name, "RECREATE")

    merge_on = [required_columns[:3]]

    for period, dir_applied, query_period in zip(periods, in_top_dirs, queries_periods): # pylint: disable=too-many-nested-blocks
        query_tmp = None
        if query_all:
            query_tmp = query_all
            if query_period:
                query_tmp += f" and {query_period}"
        elif query_period:
            query_tmp = query_period

        if query_tmp:
            query_tmp += " and abs(inv_mass - @INV_MASS[0]) <= @INV_MASS_WINDOW[0]"
        else:
            query_tmp = "abs(inv_mass - @INV_MASS) <= @INV_MASS_WINDOW"

        files_all = glob(f"{dir_applied}/**/{gen_file_name}", recursive=True)

        if not file_name_mlwp_map:
            args = [((f_reco,), histo_params, required_columns, \
                    query_tmp, only_one_evt, merge_on[0], queries_slices, None) \
                    for f_reco in files_all]

        else:
            print(file_name_mlwp_map)
            args = []
            for file_name in files_all:
                found = False
                query_tmp_file = query_tmp
                for key, value in file_name_mlwp_map.items():
                    if key in file_name:
                        if query_tmp_file:
                            query_tmp_file += f" and {value}"
                        else:
                            query_tmp_file = value
                        found = True
                        break
                if not found:
                    print(f"ERROR: {file_name}")
                    sys.exit(0)
                args.append(((file_name,), histo_params, required_columns, \
                        query_tmp_file, only_one_evt, merge_on[0], queries_slices, None))


        histos = multi_proc(fill_from_pickles, args, None, 100, 30)

        histo_names_period = [f"{name}_{period}" for name in histo_names]
        summary_histograms_and_write(file_out, histos, histo_names_period,
                                     histo_xtitles, histo_ytitles)

    file_out.Close()




def make_distributions(args, inv_mass, inv_mass_window): # pylint: disable=too-many-statements

    config = parse_yaml(args.config)

    database_path = config["database"]
    data_or_mc = config["data_or_mc"]
    analysis_name = config["analysis"]
    distribution = config["distribution"]
    distribution_x_range = config["x_range"]
    out_file = config["out_file"]
    # whether or not to slice and derive weights in these slices
    period_cuts = config.get("period_cuts", None)
    slice_cuts = config.get("slice_cuts", None)
    required_columns = config.get("required_columns", None)

    # Now open database
    _, database = read_database(database_path)

    analysis_config = database["analysis"][analysis_name]
    inv_mass[0] = database["mass"]

    inv_mass_window[0] = config.get("mass_window", 0.02)

    # required column names
    column_names = ["ev_id", "ev_id_ext", "run_number"]
    column_names.append(distribution)

    # Add column names required by the user
    if required_columns:
        for rcn in required_columns:
            if rcn not in column_names:
                column_names.append(rcn)

    periods = database["multi"][data_or_mc]["period"]

    # is this ML or STD?
    is_ml = database["doml"]

    # No cuts for specific input file
    file_names_cut_map = None

    # Set where to read data from and set overall selection query
    if data_or_mc == "mc":
        query_all = None
        in_file_name_gen = database["files_names"]["namefile_evt"]
        in_file_names = [in_file_name_gen]
        in_top_dirs = database["multi"]["mc"]["pkl"]
    else:
        column_names.append("inv_mass")
        query_all = "is_ev_rej == 0"
        trigger_sel = analysis_config["triggersel"]["data"]
        in_top_dirs = database["mlapplication"]["data"]["pkl_skimmed_dec"]
        if trigger_sel:
            query_all += f" and {trigger_sel}"
        in_file_name_gen = database["files_names"]["namefile_reco"]
        in_file_name_gen = in_file_name_gen[:in_file_name_gen.find(".")]

        if is_ml:
            model_name = database["mlapplication"]["modelname"]
            ml_sel_column = f"y_test_prob{model_name}"
            column_names.append(ml_sel_column)
            ml_sel_pt = database["mlapplication"]["probcutoptimal"]
            pt_bins_low = database["sel_skim_binmin"]
            pt_bins_up = database["sel_skim_binmax"]
            in_file_names = [f"{in_file_name_gen}{ptl}_{ptu}" \
                    for ptl, ptu in zip(pt_bins_low, pt_bins_up)]
            pkl_extension = ""
            file_names_cut_map = {ifn: f"{ml_sel_column} > {cut}" \
                    for ifn, cut in zip(in_file_names, ml_sel_pt)}

        else:
            pkl_extension = "_std"

        in_file_name_gen = in_file_name_gen + "*"


        #file_extension = in_file_name_gen[in_file_name_gen.find("."):]
        #in_file_names = [f"{fn}{pkl_extension}{file_extension}" for fn in in_file_names]

        # Now make the directory path right
        in_top_dirs = [f"{itd}{pkl_extension}" for itd in in_top_dirs]



    if data_or_mc == "mc":
        derive_mc(periods, in_top_dirs,
                  in_file_name_gen,
                  column_names,
                  distribution, distribution_x_range,
                  out_file,
                  period_cuts, query_all, slice_cuts)

    else:
        derive_data(periods, in_top_dirs,
                    in_file_name_gen,
                    column_names,
                    distribution, distribution_x_range,
                    file_names_cut_map,
                    out_file,
                    period_cuts, query_all, slice_cuts)



def make_weights(args, *ignore): # pylint: disable=unused-argument
    file_data = TFile.Open(args.data, "READ")
    file_mc = TFile.Open(args.mc, "READ")

    keys_data = file_data.GetListOfKeys()
    keys_mc = file_mc.GetListOfKeys()

    out_file_name = f"weights_{args.data}"
    out_file = TFile.Open(out_file_name, "RECREATE")

    def get_mc_histo(histos, period):
        for h in histos:
            if period in h.GetName():
                return h
        sys.exit(1)
        return None

    mc_histos = [k.ReadObj() for k in keys_mc]
    data_histos = [k.ReadObj() for k in keys_data]

    # norm all
    for h in mc_histos:
        if h.GetEntries():
            h.Scale(1. / h.Integral())
    for h in data_histos:
        if h.GetEntries():
            h.Scale(1. / h.Integral())

    for dh in data_histos:
        name = dh.GetName()
        per_pos = name.rfind("_")

        period = name[per_pos:]
        mc_histo = get_mc_histo(mc_histos, period)

        dh.Divide(dh, mc_histo, 1., 1., "B")
        out_file.cd()
        dh.Write(f"{dh.GetName()}_weights")

    out_file.Close()
    file_data.Close()
    file_mc.Close()


#############
# FUNCTIONS #
#############

def _callback(err):
    print(err)

def multi_proc(function, argument_list, kw_argument_list, maxperchunk, max_n_procs=10):

    chunks_args = [argument_list[x:x+maxperchunk] \
            for x in range(0, len(argument_list), maxperchunk)]
    if not kw_argument_list:
        kw_argument_list = [{} for _ in argument_list]
    chunks_kwargs = [kw_argument_list[x:x+maxperchunk] \
            for x in range(0, len(kw_argument_list), maxperchunk)]
    res_all = []
    for chunk_args, chunk_kwargs in zip(chunks_args, chunks_kwargs):
        print("Processing new chunck size=", maxperchunk)
        pool = mp.Pool(max_n_procs)
        res = [pool.apply_async(function, args=args, kwds=kwds, error_callback=_callback) \
                for args, kwds in zip(chunk_args, chunk_kwargs)]
        pool.close()
        pool.join()
        res_all.extend(res)


    res_list = None
    try:
        res_list = [r.get() for r in res_all]
    except Exception as e: # pylint: disable=broad-except
        print("EXCEPTION")
        print(e)
    return res_list


def fill_from_pickles(file_paths, histo_params, cols=None, query=None, skim_func=None,
                      skim_func_args=None, queries=None, merge_on=None):

    print(f"Process files {file_paths}")

    dfs = [pickle.load(openfile(f, "rb")) for f in file_paths]
    df = dfs[0]
    if len(dfs) > 1:
        if merge_on and len(merge_on) != len(dfs) - 1:
            print(f"ERROR: merge_on must be {len(dfs) - 1} however found to be {len(merge_on)}")
            sys.exit(1)

        for df_, on in zip(dfs[1:], merge_on):
            # Recursively merge dataframes
            df = pd.merge(df, df_, on=on)

    if query:
        # Apply common query
        df = df.query(query)
    if cols:
        # Select already columns which are needed in the following
        df = df[cols]

    if skim_func:
        # Skim the dataframe according to user function
        df = skim_func(df, skim_func_args)


    histos = []
    if not queries:
        queries = [None] * len(histo_params)

    if len(queries) != len(histo_params):
        print("ERROR: Need as many queries as histogram parameters")
        sys.exit(1)

    for hp, qu in zip(histo_params, queries):
        n_cols = len(hp[0])
        if n_cols > 2:
            print(f"ERROR: Cannot handle plots with dimension > 2")
            sys.exit(1)
        histo_func = TH1F if n_cols == 1 else TH2F

        df_fill = df
        if qu:
            # If there is an additional query for this histogram apply it to dataframe
            df_fill = df.query(qu)

        # Arrange for 1D or 2D plotting
        fill_with = df_fill[hp[0][0]] if n_cols == 1 else df_fill[hp[0]].to_numpy()

        histo_name = "_".join(hp[0])
        histo = histo_func(histo_name, histo_name, *hp[1])

        weights = df_fill[hp[2]] if len(hp) == 3 else None
        fill_hist(histo, fill_with, weights=weights)
        histo.SetDirectory(0)
        histos.append(histo)

    return histos






def main():
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest="command", help="[distr, weights]")

    distr_parser = sub_parsers.add_parser("distr")
    distr_parser.add_argument("config", help="configuration to derive distributions")
    distr_parser.set_defaults(func=make_distributions)

    weights_parser = sub_parsers.add_parser("weights")
    weights_parser.add_argument("data", help="ROOT file with data distributions")
    weights_parser.add_argument("mc", help="ROOT file with MC distribution")
    weights_parser.set_defaults(func=make_weights)

    args_parsed = parser.parse_args()

    args_parsed.func(args_parsed, INV_MASS, INV_MASS_WINDOW)


if __name__ == "__main__":
    main()
