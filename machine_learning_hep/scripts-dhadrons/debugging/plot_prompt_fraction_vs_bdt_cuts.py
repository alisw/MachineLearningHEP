"""
file:  plot_prompt_fraction_vs_fd_cuts.py
brief: Plot prompt fraction from cross section calculations for different non-prompt cuts
usage: python3 plot_prompt_fraction_vs_fd_cuts.py config_fraction_vs_fd_cuts.json
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""

import argparse
import glob
import json
import re

import matplotlib.pyplot as plt
from ROOT import (  # pylint: disable=import-error,no-name-in-module
    TFile,
    gROOT,
)


def get_fractions(cfg):
    filenames = sorted(glob.glob(cfg["file_pattern"]))
    fractions = {}
    fractions_err = {}
    fd_cuts = []
    for pt_bin_min, pt_bin_max in zip(cfg["pt_bins_min"], cfg["pt_bins_max"], strict=False):
        fractions[f"{pt_bin_min}_{pt_bin_max}"] = []
        fractions_err[f"{pt_bin_min}_{pt_bin_max}"] = []
    for filename in filenames:
        with TFile.Open(filename) as fin:
            hist = fin.Get(cfg["histoname"])
            dirname = re.search(cfg["dir_pattern"], filename).group(0)
            fd_cut = re.split("_", dirname)[-1]
            fd_cuts.append(fd_cut)
            for ind, (pt_bin_min, pt_bin_max) in enumerate(zip(cfg["pt_bins_min"], cfg["pt_bins_max"], strict=False)):
                fractions[f"{pt_bin_min}_{pt_bin_max}"].append(hist.GetPointY(ind + 1))
                fractions_err[f"{pt_bin_min}_{pt_bin_max}"].append(hist.GetErrorY(ind + 1))
    print(f"final fractions:\n{fractions}\nfd_cuts:\n{fd_cuts}\nfractions error:\n{fractions_err}")
    return fractions, fractions_err, fd_cuts


def main():
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("config", help="JSON config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf8") as fil:
        cfg = json.load(fil)

        fractions, fractions_err, fd_cuts = get_fractions(cfg)

        for pt_bin_min, pt_bin_max in zip(cfg["pt_bins_min"], cfg["pt_bins_max"], strict=False):
            plt.figure(figsize=(20, 15))
            ax = plt.subplot(1, 1, 1)
            ax.set_xlabel(cfg["x_axis"])
            ax.set_ylabel(cfg["y_axis"])
            ax.set_ylim([0.0, 1.0])
            ax.tick_params(labelsize=20)
            plt.grid(linestyle="-", linewidth=2)
            plt.errorbar(fd_cuts, fractions[f"{pt_bin_min}_{pt_bin_max}"],
                         yerr=fractions_err[f"{pt_bin_min}_{pt_bin_max}"],
                         c="b", elinewidth=2.5, linewidth=4.0)
            ax.set_xticks(ax.get_xticks()[::10])
            plt.savefig(f'{cfg["outdir"]}/{cfg["outfile"]}_{pt_bin_min}_{pt_bin_max}.png')


if __name__ == "__main__":
    main()
