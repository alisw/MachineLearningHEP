# pylint: disable=missing-function-docstring, invalid-name
"""
file: run-mlhep-fitter-multitrial.py
brief: Prepare MLHEP database files for different fit configurations for multitrial systematics.
usage: python run-mlhep-fitter-multitrial.py database_lc data/data_run3 trial_configs_dir mlhep_results_dir_pattern
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""

import argparse
import re
import shutil

import yaml

SIGMA02="0.007, 0.007, 0.013"
SIGMA23="0.007, 0.007, 0.013"
SIGMA34="0.007, 0.007, 0.012"
SIGMA45="0.008, 0.008, 0.016"
SIGMA56="0.010, 0.010, 0.016"
SIGMA67="0.008, 0.008, 0.017"
SIGMA78="0.012, 0.012, 0.018"
SIGMA810="0.015, 0.012, 0.018"
SIGMA1012="0.010, 0.010, 0.022"
SIGMA1216="0.016, 0.016, 0.029"
SIGMA1624="0.016, 0.016, 0.029"
FREE_SIGMAS=[SIGMA02, SIGMA23, SIGMA34, SIGMA45, SIGMA56, SIGMA67, SIGMA78,
             SIGMA810, SIGMA1012, SIGMA1216, SIGMA1624]

CENTRAL_TRIAL=""

BASE_TRIALS = (
    ["alpha-15%", "alpha+15%"],
    ["n-15%", "n+15%"],
    ["rebin-1", "rebin+1"],
    ["free-sigma"],
    ["poly3"],
    ["narrow", "narrow2", "wide", "wide2"]
)

DIR_PATH = "/data8/majak/MLHEP"

def generate_trials(trial_classes):
    combinations = [""]
    for trial_class in trial_classes:
        class_comb = []
        for cur_comb in combinations:
            for trial in trial_class:
                class_comb.append(cur_comb + "_" + trial)
                #print(f"{cur_comb}_{trial}")
        combinations.extend(class_comb)
    return combinations

def replace_with_reval(var, in_str, frac):
    pattern = fr"{var}\[([0-9.]*), .*?\]"
    values = re.findall(pattern, in_str)
    new_val = round(float(values[0]) * frac, 3)
    return re.sub(pattern, f"{var}[{new_val}, {new_val}]", in_str)

def process_trial(trial, ana_cfg, data_cfg, mc_cfg):
    fit_cfg = ana_cfg["mass_roofit"]
    if "alpha-15%" in trial:
        print("Processing alpha-15%")
        for pt_cfg in mc_cfg:
            sig_fn = pt_cfg["components"]["sig"]["fn"]
            pt_cfg["components"]["sig"]["fn"] = replace_with_reval("alpha1", sig_fn, 0.85)
    elif "alpha+15%" in trial:
        print("Processing alpha+15%")
        for pt_cfg in mc_cfg:
            sig_fn = pt_cfg["components"]["sig"]["fn"]
            pt_cfg["components"]["sig"]["fn"] = replace_with_reval("alpha1", sig_fn, 1.15)
    elif "n-15%" in trial:
        print("Processing n-15%")
        for pt_cfg in mc_cfg:
            sig_fn = pt_cfg["components"]["sig"]["fn"]
            pt_cfg["components"]["sig"]["fn"] = replace_with_reval("n1", sig_fn, 0.85)
    elif "n+15%" in trial:
        print("Processing n+15%")
        for pt_cfg in mc_cfg:
            sig_fn = pt_cfg["components"]["sig"]["fn"]
            pt_cfg["components"]["sig"]["fn"] = replace_with_reval("n1", sig_fn, 1.15)
    elif "rebin-1" in trial:
        print("Processing rebin-1")
        ana_cfg["n_rebin"] = [rebin - 1 for rebin in ana_cfg["n_rebin"]]
    elif "rebin+1" in trial:
        print("Processing rebin+1")
        ana_cfg["n_rebin"] = [rebin + 1 for rebin in ana_cfg["n_rebin"]]
    elif "free-sigma" in trial:
        print("Processing free-sigma")
        for pt_cfg, free_sigma in zip(mc_cfg, FREE_SIGMAS, strict=False):
            sig_fn = pt_cfg["components"]["sig"]["fn"]
            pt_cfg["components"]["sig"]["fn"] = re.sub(r"sigma_g1\[(.*?)\]",
                                                       f"sigma_g1[{free_sigma}]", sig_fn)
    elif "poly3" in trial:
        print("Processing poly3")
        for pt_cfg in data_cfg:
            bkg_fn = pt_cfg["components"]["bkg"]["fn"]
            pt_cfg["components"]["bkg"]["fn"] = re.sub(r"a2\[(.*?)\]",
                                                       r"a2[\1], a3[-1e8, 1e8]", bkg_fn)
    elif "narrow2" in trial:
        print("Processing narrow2")
        for pt_cfg in fit_cfg:
            pt_cfg["range"] = [pt_cfg["range"][0] + 0.02, pt_cfg["range"][1] - 0.02]
    elif "narrow" in trial:
        print("Processing narrow")
        for pt_cfg in fit_cfg:
            pt_cfg["range"] = [pt_cfg["range"][0] + 0.01, pt_cfg["range"][1] - 0.01]
    elif "wide2" in trial:
        print("Processing wide2")
        for pt_cfg in fit_cfg:
            pt_cfg["range"] = [max(2.10, pt_cfg["range"][0] - 0.02),
                               min(2.47, pt_cfg["range"][1] + 0.02)]
    elif "wide" in trial:
        print("Processing wide")
        for pt_cfg in fit_cfg:
            pt_cfg["range"] = [max(2.10, pt_cfg["range"][0] - 0.01),
                               min(2.47, pt_cfg["range"][1] + 0.01)]


def main(db, db_dir, out_db_dir, resdir_pattern):
    db_ext=f"{db}.yml"
    db_path=f"{db_dir}/{db_ext}"
    combinations = generate_trials(BASE_TRIALS)

    for comb in combinations:
        print(comb)

        cur_cfg = f"{out_db_dir}/{db}{comb}.yml"
        shutil.copy2(db_path, cur_cfg)

        with open(cur_cfg, encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)

        ana_cfg = cfg["LcpKpi"]["analysis"]["Run3analysis"]
        fit_cfg = ana_cfg["mass_roofit"]
        mc_cfg = [fit_params for fit_params in fit_cfg \
                    if "level" in fit_params and fit_params["level"] == "mc"]
        data_cfg = [fit_params for fit_params in fit_cfg if "level" not in fit_params]

        resdir = f"{resdir_pattern}{comb}"
        respath = f"{DIR_PATH}/{resdir}/"
        ana_cfg["data"]["prefix_dir_res"] = respath
        ana_cfg["mc"]["prefix_dir_res"] = respath

        trials = comb.split("_")

        for trial in trials:
            process_trial(trial, ana_cfg, data_cfg, mc_cfg)

        with open(cur_cfg, "w", encoding="utf-8") as stream:
            yaml.dump(cfg, stream, sort_keys=False, width=10000, default_flow_style=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("db", help="MLHEP database without extension")
    parser.add_argument("db_dir", help="path to directory with MLHEP database")
    parser.add_argument("out_db_dir", help="path to output directory for generated MLHEP databases")
    parser.add_argument("resdir", help="MLHEP resdir pattern")
    args = parser.parse_args()

    main(args.db, args.db_dir, args.out_db_dir, args.resdir)
