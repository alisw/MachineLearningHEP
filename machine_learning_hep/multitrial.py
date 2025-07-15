# pylint: disable=missing-function-docstring, invalid-name
"""
file: multitrial.py
brief: Plot multitrial systematics based on multiple fit trials, one file per trial.
usage: python3 multitrial.py config_multitrial.json
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""
import argparse
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from ROOT import (  # pylint: disable=import-error,no-name-in-module
    TFile,
    gROOT,
)


def plot_text_box(ax, text):
    ax.text(0.98, 0.97, text,
            horizontalalignment="right", verticalalignment="top",
            fontsize=40, va="top", transform=ax.transAxes,
            bbox={"edgecolor": "black", "fill": False})


def get_yields(cfg):
    filenames = sorted(glob.glob(cfg["file_pattern"]),
                       key=lambda filename: re.split("/", filename)[-2])
    yields = {}
    yields_err = {}
    trials = {}
    chis = {}
    for pt_bin_min, pt_bin_max in zip(cfg["pt_bins_min"], cfg["pt_bins_max"]):
        yields[f"{pt_bin_min}_{pt_bin_max}"] = []
        yields_err[f"{pt_bin_min}_{pt_bin_max}"] = []
        trials[f"{pt_bin_min}_{pt_bin_max}"] = []
        chis[f"{pt_bin_min}_{pt_bin_max}"] = []
    for filename in filenames:
        print(f"Reading {filename}")
        with TFile.Open(filename) as fin:
            hist = fin.Get(cfg["histoname"])
            hist_sel = fin.Get(cfg["sel_histoname"])
            if hist.ClassName() != "TH1F":
                print(f"No hist in {filename}")
            if hist_sel.ClassName() != "TH1F":
                print(f"No hist sel in {filename}")
            dirname = re.split("/", filename)[4] # [-2] for D2H fitter
            trial_name = dirname.replace(cfg["dir_pattern"], "")
            for ind, (pt_bin_min, pt_bin_max) in enumerate(zip(cfg["pt_bins_min"],
                                                               cfg["pt_bins_max"])):
                if eval(cfg["selection"])(hist_sel.GetBinContent(ind + 1)) \
                        and hist.GetBinContent(ind + 1) > 1.0 :
                    yields[f"{pt_bin_min}_{pt_bin_max}"].append(hist.GetBinContent(ind + 1))
                    yields_err[f"{pt_bin_min}_{pt_bin_max}"].append(hist.GetBinError(ind + 1))
                    trials[f"{pt_bin_min}_{pt_bin_max}"].append(trial_name)
                    chis[f"{pt_bin_min}_{pt_bin_max}"].append(hist_sel.GetBinContent(ind + 1))
                else:
                    print(f"Rejected: {hist_sel.GetBinContent(ind + 1)} {trial_name} "\
                          f"pt: {pt_bin_min}, {pt_bin_max}")
                    if hist.GetBinContent(ind + 1) < 1.0:
                        print("Yield 0")
    return yields, yields_err, trials, chis


def prepare_figure(cfg, y_label, ticks):
    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel(cfg["x_axis"], fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(which="both", width=2.5, direction="in")
    ax.tick_params(which="major", labelsize=20, length=15)
    ax.tick_params(which="minor", length=7)
    ax.xaxis.set_major_locator(MultipleLocator(ticks))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    return fig, ax


def set_ax_limits(ax, pt_string, values, errs):
    ax.margins(0.01, 0.2)
    np_values = np.array(values, dtype="float32")
    np_errs = np.array(errs, dtype="float32")
    if ax.get_ylim()[1] - ax.get_ylim()[0] > 30.0 * np.std(np_values):
        ax.set_ylim(np.mean(np_values) - 10.0 * np.std(np_values),
                    np.mean(np_values) + 10.0 * np.std(np_values))
        print(f"{pt_string} narrowing down the axis to {ax.get_ylim()}")


def plot_trial_line(ax, central_trial_ind):
    axis_lim = ax.get_ylim()
    y_axis = np.linspace(*axis_lim, 100)
    ax.plot([central_trial_ind] * len(y_axis), y_axis, c="m", ls="--", linewidth=4.0)
    ax.set_ylim(*axis_lim)


def plot_yields_trials(yields, yields_err, trials, cfg, pt_string, plot_pt_string,
                       central_trial_ind, central_yield):
    fig, ax = prepare_figure(cfg, cfg["y_axis"], 100)
    x_axis = range(len(trials))
    ax.errorbar(x_axis, yields, yerr=yields_err,
                fmt="o", c="b", elinewidth=2.5, linewidth=4.0)
    set_ax_limits(ax, pt_string, yields, yields_err)
    central_line = np.array([central_yield] * len(x_axis), dtype="float32")
    ax.plot(x_axis, central_line, c="orange", ls="--", linewidth=4.0)
    central_err = np.array([yields_err[central_trial_ind]] * len(x_axis), dtype="float32")
    ax.fill_between(x_axis, central_line - central_err, central_line + central_err,
                    facecolor="orange", edgecolor="none", alpha=0.3)
    plot_trial_line(ax, central_trial_ind)
    plot_text_box(ax, plot_pt_string)
    fig.savefig(f'{cfg["outdir"]}/{cfg["outfile"]}_yields_trials_{pt_string}.png',
                bbox_inches='tight')
    plt.close()


def plot_chis(chis, cfg, pt_string, plot_pt_string):
    fig, ax = prepare_figure(cfg, "Chi2/ndf", 100)
    x_axis = range(len(chis))
    ax.scatter(x_axis, chis, c="b", marker="o")
    set_ax_limits(ax, pt_string, chis, [0.0] * len(chis))
    plot_text_box(ax, plot_pt_string)
    fig.savefig(f'{cfg["outdir"]}/{cfg["outfile"]}_chis_{pt_string}.png',
                bbox_inches='tight')
    plt.close()


def plot_yields_distr(yields, cfg, pt_string, plot_pt_string, central_trial_ind, central_yield):
    plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel("Ratio", fontsize=20)
    ax.tick_params(labelsize=20, length=7, width=2.5)
    ratios = [yield_ / central_yield for ind, yield_ in enumerate(yields) \
              if ind != central_trial_ind]
    ax.hist(ratios, color="b", linewidth=4.0)
    mean = np.mean(yields)
    std_dev = np.std(yields)
    diffs = [(yield_ - central_yield) / central_yield \
             for yield_ in yields[:central_trial_ind]]
    diffs.extend([(yield_ - central_yield) / central_yield \
                 for yield_ in yields[central_trial_ind+1:]])
    rmse = np.sqrt(np.mean(np.array(diffs, dtype="float32")**2))
    plot_text_box(ax, f"{plot_pt_string}\n"\
                      f"mean:    {mean:.0f}\n"\
                      f"std dev: {std_dev:.2f}\n"\
                      f"RMSE:    {rmse:.2f}\n"\
                      f"#trials: {len(yields)}")
    plt.savefig(f'{cfg["outdir"]}/{cfg["outfile"]}_distr_{pt_string}.png', bbox_inches='tight')
    plt.close()


def main():
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("config", help="JSON config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf8") as fil:
        cfg = json.load(fil)

        yields, yields_err, trials, chis = get_yields(cfg)

        for pt_bin_min, pt_bin_max in zip(cfg["pt_bins_min"], cfg["pt_bins_max"]):
            plot_pt_string = f"${pt_bin_min} < p_\\mathrm{{T}}/(\\mathrm{{GeV}}/c) < {pt_bin_max}$"
            pt_string = f"{pt_bin_min}_{pt_bin_max}"

            try:
                central_trial_ind = trials[pt_string].index(cfg["central_trial"])
                central_yield = yields[pt_string][central_trial_ind]

                plot_yields_trials(yields[pt_string], yields_err[pt_string], trials[pt_string], cfg,
                                   pt_string, plot_pt_string, central_trial_ind, central_yield)
                plot_yields_distr(yields[pt_string], cfg, pt_string, plot_pt_string,
                                  central_trial_ind, central_yield)
                plot_chis(chis[pt_string], cfg, pt_string, plot_pt_string)
            except:
                pass

            with open(f'{cfg["outdir"]}/{cfg["outfile"]}_trials_{pt_string}.txt',
                      "w", encoding="utf-8") as ftext:
                for trial in trials[pt_string]:
                    ftext.write(f"{trial}\n")


if __name__ == "__main__":
    main()
