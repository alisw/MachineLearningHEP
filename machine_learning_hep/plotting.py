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

import os
import glob
import pickle
import pandas as pd
from machine_learning_hep.utilities import openfile
from machine_learning_hep.logger import configure_logger, get_logger
from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict
import matplotlib.pyplot as plt
import numpy as np

class Histo1D:
    def __init__(self, name, bins, axis_range, label, xlabel, ylabel):
        self.name = name
        self.bins = bins
        self.axis_range = axis_range
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label

        self.hist, self.edges = np.histogram(np.array([], dtype="float64"), bins, axis_range, None, np.array([], dtype="float64"))

    def add_values(self, values, weights=None):
        values = [float(v) for v in values]
        if weights is not None:
            weights = [float(w) for w in weights]
        self.hist += np.histogram(values, self.bins, self.axis_range, None, weights)[0]

    def get_statistics(self):
        return [ ("bin_" + str(i + 1), self.edges[i], self.edges[i+1], self.hist[i]) for i in range(len(self.hist)) ]


def plot1D(histograms, plot_name, ax=None):

    logger = get_logger()
    try:
        it = iter(histograms)
    except TypeError as te:
        histograms = [histograms]

    fig = plt.figure(figsize=(60,25))
    if ax is None:
        ax = plt.gca()
    # Derive bin centers from edges of first histogram
    common_edges = histograms[0].edges
    common_xlabel = histograms[0].xlabel
    common_ylabel = histograms[0].ylabel
    bin_widths = [ 0.9 * (common_edges[i+1] - common_edges[i] ) for i in range(len(common_edges) - 1) ]
    bin_centers = [ (common_edges[i+1] + common_edges[i]) / 2 for i in range(len(common_edges) - 1) ]
    new_bottom = np.zeros(len(histograms[0].hist))
    for h in histograms:
        if not np.array_equal(h.edges,common_edges):
            logger_string = f"Incompatible edges found in histogram {h.name}"
            logger.fatal(logger_string)
        ax.bar(bin_centers, h.hist, width=bin_widths, alpha=0.5, label=h.label, bottom=new_bottom)
    ax.set_xticks(common_edges)
    ax.set_xticklabels(common_edges)
    ax.legend(fontsize=30)
    ax.set_xlabel(common_xlabel, fontsize=30)
    ax.set_ylabel(common_ylabel, fontsize=30)
    ax.tick_params(labelsize=30)
    return ax

def get_statistics(histograms):

    logger = get_logger()
    try:
        it = iter(histograms)
    except TypeError as te:
        histograms = [histograms]

    stats_list = []
    for h in histograms:
        stats_dict = {}
        stats = h.get_statistics()
        stats_dict["name"] = h.name
        stats_dict["edges"] = []
        stats_dict["total_count"] = 0
        total_stats = 0
        for s in stats:
            stats_dict["edges"].append([str(s[0]), float(s[1]), float(s[2]), float(s[3])])
            stats_dict["total_count"] += float(s[3])
        stats_list.append(stats_dict)

    return stats_list


def make_plots_stats(histograms, in_one_figure=False):

    figures = []
    stats = []
    names = []

    logger = get_logger()

    if in_one_figure:
        fig = plt.figure(figsize=(60, 25))
        gs = GridSpec(3, int(len(histograms)/3+1))
        axes = [fig.add_subplot(gs[i]) for i in range(len(histograms))]

        i = 0
        for h_name, histos in histograms.items():
            histo_list = [ v for v in histos.values() ] 
            names.append(h_name)
            plot1D(histo_list, h_name, ax=axes[i])
            stats.append(get_statistics(histo_list))
            i += 1
        return [fig], names, stats

    else:
        for h_name, histos in histograms.items():
            fig = plt.figure(figsize=(60, 25))
            ax = fig.subplots()
            histo_list = [ v for v in histos.values() ] 
            names.append(h_name)
            plot1D(histo_list, h_name, ax=ax)
            stats.append(get_statistics(histo_list))
            figures.append(fig)

    return figures, names, stats


def process_files(top_dir, file_signature, recursive, n_files, plot_config):
    top_dir = os.path.expanduser(top_dir)
    logger = get_logger()
    if not os.path.isdir(top_dir):
        logger_string = f"Directory {top_dir} does not exist."
        logger.fatal(logger_string)
    
    file_signature = os.path.join(top_dir, "**/", file_signature)
    logger_string = f"Globbing files with matching {file_signature}"
    logger.info(logger_string)

    root_files = glob.glob(file_signature, recursive=recursive)
    if not root_files:
        logger.fatal("No ROOT files selected")

    if len(root_files) < n_files:
        logger_string = f"Although {n_files} were requested, could only glob {len(root_files)}"
        logger.warning(logger_string)
    elif len(root_files) > n_files and n_files > 0:
        # Skim list of ROOT files to requested number
        root_files = root_files[:n_files]

    available_cuts = plot_config["cuts"]
    available_cuts[None] = None
    plots = plot_config["plots"]

    histograms = {}
    # Now, go though all files
    for i_file, f in enumerate(root_files):
        logger_string = f"Process {i_file}. file at {f}"
        logger.debug(logger_string)
        # Brute force for now
        for plot in plots:
            for o in plot["observables"]:
                legend_labels = o.get("legend_labels", o["cuts"])
                if len(legend_labels) != len(o["cuts"]):
                    logger_string = f"Different number of legend labels and cuts found for observable {o['name']}"
                    logger.fatal(logger_string)
                for cut, label in zip(o["cuts"], legend_labels):
                    if cut is None:
                        values = pickle.load(openfile(f, "rb"))[o["name"]]
                        if values.size == 0:
                            logger_string = f"Tree dataframe in file {f} seems to be empty"
                            logger.warning(logger_string)
                            continue
                        histo_name = o["name"] + "_no_cut"
                        if o["name"] not in histograms:
                            histograms[o["name"]] = {}
                        if histo_name not in histograms[o["name"]]:
                                histograms[o["name"]][histo_name] = Histo1D(histo_name, o["bins"], o["range"], str(label), o.get("xlabel", "xlabel"), o.get("ylabel", "ylabel"))
                        histograms[o["name"]][histo_name].add_values(values)
                    else:
                        values = pickle.load(openfile(f, "rb")).query(available_cuts[cut]["expression"])[o["name"]]
                        if values.size == 0:
                            logger_string = f"After cut nothing left in dataframe in file {f}"
                            logger.warning(logger_string)
                            continue
                        histo_name = o["name"] + "_" + cut
                        if o["name"] not in histograms:
                            histograms[o["name"]] = {}
                        if histo_name not in histograms[o["name"]]:
                            histograms[o["name"]][histo_name] = Histo1D(histo_name, o["bins"], o["range"], str(label), o.get("xlabel", "xlabel"), o.get("ylabel", "ylabel"))
                        histograms[o["name"]][histo_name].add_values(values)

    return make_plots_stats(histograms)

def process_dataframe(dfs, plot_config, output_dir):

    logger = get_logger()
    try:
        it = iter(dfs)
    except TypeError as te:
        dfs = [dfs]

    plots = plot_config["plots"]
    available_cuts = plot_config.get("cuts", {})
    available_cuts[None] = None

    for i_df, df in enumerate(dfs):
        logger_string = f"Process {i_file}. dataframe"
        logger.debug(logger_string)
        # Brute force for now
        for plot in plots:
            for o in plot["observables"]:
                for cut in o["cuts"]:
                    if cut is None:
                        values = df[o["name"]]
                        if values.size == 0:
                            logger_string = f"Dataframe seems to be empty"
                            logger.warning(logger_string)
                            continue
                        histo_name = o["name"] + "_no_cut"
                        if o["name"] not in histograms:
                            histograms[o["name"]] = {}
                        if histo_name not in histograms[o["name"]]:
                                histograms[o["name"]][histo_name] = Histo1D(histo_name, o["bins"], o["range"], "no_cut", o.get("xlabel", "xlabel"), o.get("ylabel", "ylabel"))
                        histograms[o["name"]][histo_name].add_values(values)
                    else:
                        values = df.query(available_cuts[cut]["expression"])[o["name"]]
                        if values.size == 0:
                            logger_string = f"After cut nothing left in dataframe"
                            logger.warning(logger_string)
                            continue
                        histo_name = o["name"] + "_" + cut
                        if o["name"] not in histograms:
                            histograms[o["name"]] = {}
                        if histo_name not in histograms[o["name"]]:
                            histograms[o["name"]][histo_name] = Histo1D(histo_name, o["bins"], o["range"], available_cuts[cut]["expression"], o.get("xlabel", "xlabel"), o.get("ylabel", "ylabel"))
                        histograms[o["name"]][histo_name].add_values(values)

    return make_plots_stats(histograms)


def plot_from_yamls(yamls):

    logger = get_logger()
    logger.info("Create histograms from YAML files")
    try:
        it = iter(yamls)
    except TypeError as te:
        yamls = [yamls]

    histograms = {}
    for y in yamls:
        y_dict = parse_yaml(y)
        y_dict["label"] = y_dict.get("label", "label")
        y_dict["xlabel"] = y_dict.get("xlabel", "xlabel")
        y_dict["ylabel"] = y_dict.get("ylabel", "ylabel")

        # Get lower edges from all bins and the very last one
        edges = [ e[1] for e in y_dict["edges"] ] + [y_dict["edges"][-1][2]]
        bin_centers = [ (edges[i+1] + edges[i]) / 2 for i in range(len(edges) - 1) ]
        histo = Histo1D(y_dict["name"], edges, None, y_dict["label"], y_dict["xlabel"], y_dict["ylabel"])
        histo.add_values(bin_centers, [e[3] for e in y_dict["edges"]])
        histograms[y_dict["name"]] = histo

    return make_plots_stats({"plot": histograms})


def save_plots(figures, names, stats, output_dir="./", close_figures=True):
    
    logger = get_logger()
    if not os.path.isdir(output_dir):
        if os.path.exists(output_dir):
            logger_string = f"{output_dir} exists but is not a directory"
            logger.fatal(logger_string)
        os.makedirs(output_dir)

    if len(figures) == 1 and len(names) > 1:
        # Apparently many histograms were put in one figure
        prefix = "_".join(names)
        output_path = os.path.join(output_dir, prefix + ".png")
        
        logger_string = f"Save plot {output_path}"
        logger.info(logger_string)
        fig[0].savefig(output_path)
        if close_figures:
            plt.close(fig[0])
    
    elif len(figures) != len(names):
        # Don't know how to handle this
        logger_string = f"{len(figures)} figures and {len(names)} given. Don't know how to handle that"
        logger.fatal(logger_string)

    else:
        for fig, name, stat in zip(figures, names, stats):
            output_path = os.path.join(output_dir, name + ".png")
            logger_string = f"Save plot {output_path}"
            logger.info(logger_string)
            fig.savefig(output_path)
            for s in stat:
                output_path = os.path.join(output_dir, s["name"] + ".yaml")
                dump_yaml_from_dict(s, output_path)
            if close_figures:
                plt.close(fig)

