#  © Copyright CERN 2018. All rights not expressly granted are reserved.  #
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
Script for plotting figures of the Run 3 HF-jet substructure analyses
Author: Vit Kucera <vit.kucera@cern.ch>
"""

# pylint: disable=too-many-lines, too-many-instance-attributes, too-many-statements, too-many-locals
# pylint: disable=too-many-nested-blocks, too-many-branches, consider-using-f-string
# pylint: disable=unused-variable

import argparse
import logging
import os
from pathlib import Path
from functools import reduce
import numpy as np

import yaml
from ROOT import TFile, gROOT, gStyle

from machine_learning_hep.logger import get_logger
from machine_learning_hep.analysis.analyzer_jets import string_range_ptjet, string_range_pthf

# HF specific imports
from machine_learning_hep.utilities import (
    make_plot,
    draw_latex_lines,
    get_colour,
    get_marker,
    make_message_notfound,
    reset_graph_outside_range,
    reset_hist_outside_range,
    get_mean_hist,
    get_mean_graph,
    get_mean_uncertainty,
)
from machine_learning_hep.utils.hist import get_axis, bin_array, project_hist, get_dim, get_bin_limits


def shrink_err_x(graph, width=0.1):
    for i in range(graph.GetN()):
        graph.SetPointEXlow(i, width)
        graph.SetPointEXhigh(i, width)


# final ranges
x_range = {
    "zg": [0.1, 0.5],
    "rg": [0.0, 0.4],
    "nsd": [-0.5, 5.5],
    "zpar": [0.3, 1.0],
}


class Plotter:
    def __init__(self, path_input_file: str, path_database_analysis: str, typean: str, var: str):
        self.logger = get_logger()
        self.typean = typean
        self.var = var
        self.method = "sidesub"
        self.logger.setLevel(logging.INFO)

        with open(path_database_analysis, "r", encoding="utf-8") as file_db:
            db_analysis = yaml.safe_load(file_db)
        case = list(db_analysis.keys())[0]
        self.datap = db_analysis[case]
        self.db_typean = self.datap["analysis"][self.typean]

        # directories with analysis results
        self.dir_result_mc = self.db_typean["mc"]["resultsallp"]
        self.dir_result_data = self.db_typean["data"]["resultsallp"]
        file_result_name = self.datap["files_names"]["resultfilename"]
        self.path_file_results = os.path.join(self.dir_result_data, file_result_name)

        # If input file path is not provided, take the analysis result.
        self.path_input_file = path_input_file if path_input_file else self.path_file_results
        self.dir_input = os.path.dirname(self.path_input_file)
        self.file_results = None

        print(f"Input file: {self.path_input_file}")

        # output directory for figures
        self.fig_formats = ["pdf", "png"]
        self.dir_out_figs = Path(f"{os.path.expandvars(self.dir_input)}/fig/plots")
        for fmt in self.fig_formats:
            (self.dir_out_figs / fmt).mkdir(parents=True, exist_ok=True)

        print(f"Plots will be saved in {self.dir_out_figs}")

        # plotting
        self.list_obj = []
        self.labels_obj = []
        self.list_latex = []
        self.list_colours = []
        self.list_markers = []
        self.list_new = []  # list to avoid loosing objects created in loops

        # LaTeX string
        self.latex_hadron = self.db_typean["latexnamehadron"]
        self.latex_ptjet = "#it{p}_{T}^{jet ch}"
        self.latex_pthf = "#it{p}_{T}^{%s} (GeV/#it{c})" % self.latex_hadron
        self.latex_obs = self.db_typean["observables"][self.var]["label"]
        self.latex_y = self.db_typean["observables"][self.var]["label_y"]

        # binning of hadron pt
        self.edges_pthf = np.asarray(self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:], 'd')
        self.n_bins_pthf = len(self.edges_pthf) - 1

        # binning of jet pt
        # reconstruction level
        self.edges_ptjet_rec = self.db_typean["bins_ptjet"]
        self.n_bins_ptjet_rec = len(self.edges_ptjet_rec) - 1
        self.ptjet_rec_min = self.edges_ptjet_rec[0]
        self.ptjet_rec_max = self.edges_ptjet_rec[-1]
        self.edges_ptjet_rec_min = self.edges_ptjet_rec[:-1]
        self.edges_ptjet_rec_max = self.edges_ptjet_rec[1:]
        # generator level
        self.edges_ptjet_gen = self.db_typean["bins_ptjet"]
        self.edges_ptjet_gen_eff = self.db_typean["bins_ptjet"]
        self.n_bins_ptjet_gen = len(self.edges_ptjet_gen) - 1
        self.ptjet_gen_min = self.edges_ptjet_gen[0]
        self.ptjet_gen_max = self.edges_ptjet_gen[-1]
        self.edges_ptjet_gen_min = self.edges_ptjet_gen[:-1]
        self.edges_ptjet_gen_max = self.edges_ptjet_gen[1:]

        # binning of observable (z, shape,...)
        # reconstruction level
        if binning := self.cfg(f'observables.{var}.bins_det_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{var}.bins_det_fix'):
            bins_tmp = bin_array(*binning)
        elif binning := self.cfg(f'observables.{var}.bins_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{var}.bins_fix'):
            bins_tmp = bin_array(*binning)
        else:
            self.logger.error('No binning specified for %s, using defaults', var)
            bins_tmp = bin_array(10, 0., 1.)
        self.edges_obs_rec = bins_tmp
        # self.n_bins_obs_rec = len(self.edges_obs_rec) - 1
        # self.obs_rec_min = float(self.edges_obs_rec[0])
        # self.obs_rec_max = float(self.edges_obs_rec[-1])

        # generator level
        if binning := self.cfg(f'observables.{var}.bins_gen_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{var}.bins_gen_fix'):
            bins_tmp = bin_array(*binning)
        elif binning := self.cfg(f'observables.{var}.bins_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{var}.bins_fix'):
            bins_tmp = bin_array(*binning)
        else:
            self.logger.error('No binning specified for %s, using defaults', var)
            bins_tmp = bin_array(10, 0., 1.)
        self.edges_obs_gen = bins_tmp
        # self.n_bins_obs_gen = len(self.edges_obs_gen) - 1
        # self.obs_gen_min = float(self.edges_obs_gen[0])
        # self.obs_gen_max = float(self.edges_obs_gen[-1])

        print("Rec obs edges:", self.edges_obs_rec, "Gen obs edges:", self.edges_obs_gen)
        print("Rec ptjet edges:", self.edges_ptjet_rec, "Gen ptjet edges:", self.edges_ptjet_gen)

        # official figures
        # self.size_can = [800, 800]
        self.offsets_axes = [0.8, 1.1]
        self.margins_can = [0.1, 0.13, 0.05, 0.03]
        self.fontsize = 0.035
        self.opt_leg_g = "FP"  # for systematic uncertainties in the legend
        self.opt_plot_g = "2"
        self.x_latex = 0.18
        self.y_latex_top = 0.88
        self.y_step = 0.05
        self.leg_pos = [.72, .75, .85, .85]
        # self.y_margin_up = 0.46
        self.y_margin_up = 0.05
        self.y_margin_down = 0.05
        # size_can_double = [800, 800]
        # offsets_axes_double = [0.8, 0.8]
        # margins_can_double = [0.1, 0.1, 0.1, 0.1]
        # margins_can_double = [0., 0., 0., 0.]
        # size_thg = 0.05
        # offset_thg = 0.85
        # fontsize_glob = 0.032 # font size relative to the canvas height
        # scale_title = 1.3 # scaling factor to increase the size of axis titles
        # tick_length = 0.02
        self.plot_errors_x = True  # plot horizontal error bars

        # axes titles
        self.title_x = self.latex_obs
        self.title_y = self.latex_y
        self.title_full = f";{self.title_x};{self.title_y}"
        self.title_full_default = self.title_full
        # self.title_full_ratio = ";%s;data/MC: ratio of %s" % (self.title_x, self.title_y)
        # text
        self.text_alice = "ALICE Preliminary, pp, #sqrt{#it{s}} = 13.6 TeV"  # preliminaries
        # self.text_alice = "#bf{ALICE}, pp, #sqrt{#it{s}} = 13.6 TeV"  # paper
        self.text_jets = "%s-tagged track jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.latex_hadron
        self.text_ptjet = "%g #leq %s (GeV/#it{c}) < %g, |#it{#eta}_{jet ch}| < 0.5"
        self.text_pth = "%g #leq #it{p}_{T}^{%s} (GeV/#it{c}) < %g, |#it{y}_{%s}| < 0.8"
        self.text_sd = "Soft drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"
        # self.text_acc_h = "|#it{y}| < 0.8"
        # self.text_powheg = "POWHEG + PYTHIA 6 + EvtGen"

    def cfg(self, param, default=None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
                      param.split("."), self.datap['analysis'][self.typean])

    def save_canvas(self, can, name=""):
        """Save canvas"""
        if not name:
            name = can.GetName()
        for fmt in self.fig_formats:
            can.SaveAs(f"{self.dir_out_figs}/{fmt}/{name}.{fmt}")

    def crop_histogram(self, hist, var: str):
        """Constrain x range of histogram and reset outside bins."""
        if var not in x_range:
            return
        # hist.GetXaxis().SetRangeUser(round(x_range[var][0], 2), round(x_range[var][1], 2))
        hist.GetXaxis().SetLimits(round(x_range[var][0], 2), round(x_range[var][1], 2))
        reset_hist_outside_range(hist, *x_range[var])

    def crop_graph(self, graph, var: str):
        """Constrain x range of graph and reset outside points."""
        if var not in x_range:
            return
        # graph.GetXaxis().SetRangeUser(round(x_range[var][0], 2), round(x_range[var][1], 2))
        graph.GetXaxis().SetLimits(round(x_range[var][0], 2), round(x_range[var][1], 2))
        reset_graph_outside_range(graph, *x_range[var])

    def get_object(self, name: str):
        if not (obj := self.file_results.Get(name)):
            self.logger.fatal(make_message_notfound(name))
        return obj

    def get_objects(self, *names: str):
        return [self.get_object(name) for name in names]

    def report_means(self, h_stat, h_syst, iptjet):
        mean_z_stat = get_mean_hist(h_stat)
        mean_z_syst = get_mean_graph(h_syst)
        hist_means_stat, hist_means_syst, hist_means_comb = get_mean_uncertainty(h_stat, h_syst, 100000)
        mean_z_var_comb = hist_means_comb.GetMean()
        sigma_z_var_comb = hist_means_comb.GetStdDev()
        mean_z_var_stat = hist_means_stat.GetMean()
        sigma_z_var_stat = hist_means_stat.GetStdDev()
        mean_z_var_syst = hist_means_syst.GetMean()
        sigma_z_var_syst = hist_means_syst.GetStdDev()
        make_plot(f"{self.var}_means_hf_comb_{iptjet}", list_obj=[hist_means_comb], suffix="pdf",
                  title=f"HF mean variations comb {iptjet};{self.latex_obs}")
        make_plot(f"{self.var}_means_hf_stat_{iptjet}", list_obj=[hist_means_stat], suffix="pdf",
                  title=f"HF mean variations stat {iptjet};{self.latex_obs}")
        make_plot(f"{self.var}_means_hf_syst_{iptjet}", list_obj=[hist_means_syst], suffix="pdf",
                  title=f"HF mean variations syst {iptjet};{self.latex_obs}")
        print(f"Mean HF {self.var} = stat {mean_z_stat} syst {mean_z_syst} ROOT stat {h_stat.GetMean()}")
        print(f"Mean HF {self.var} = var comb {mean_z_var_comb} +- {sigma_z_var_comb}")
        print(f"Mean HF {self.var} = var stat {mean_z_var_stat} +- {sigma_z_var_stat}")
        print(f"Mean HF {self.var} = var syst {mean_z_var_syst} +- {sigma_z_var_syst}")

    def get_text_range_ptjet(self, iptjet=-1):
        pt_min = self.edges_ptjet_gen[0] if iptjet < 0 else self.edges_ptjet_gen[iptjet]
        pt_max = self.edges_ptjet_gen[-1] if iptjet < 0 else self.edges_ptjet_gen[iptjet + 1]
        return self.text_ptjet % (pt_min, self.latex_ptjet, pt_max)

    def get_text_range_pthf(self, ipthf=-1, iptjet=-1):
        pt_min = self.edges_pthf[0] if ipthf < 0 else self.edges_pthf[ipthf]
        pt_max = self.edges_pthf[-1] if ipthf < 0 else self.edges_pthf[ipthf + 1]
        pt_max = min(pt_max, self.edges_ptjet_gen[iptjet + 1]) if iptjet > -1 else pt_max
        return self.text_pth % (pt_min, self.latex_hadron, pt_max, self.latex_hadron)

    def make_plot(self, name: str):
        """Wrapper method for calling make_plot and saving the canvas."""
        assert all(self.list_obj)
        n_obj = len(self.list_obj)
        assert len(self.labels_obj) == n_obj
        self.list_colours = [get_colour(i, 1) for i in range(n_obj)]
        self.list_markers = [get_marker(i) for i in range(n_obj)]
        y_margin_up_adj = self.y_margin_up + 1. - self.y_latex_top + self.y_step * (len(self.list_latex) - 1)
        can, new = make_plot(name,
                             list_obj=self.list_obj, labels_obj=self.labels_obj,
                             opt_leg_g=self.opt_leg_g, opt_plot_g=self.opt_plot_g,
                             offsets_xy=self.offsets_axes,
                             colours=self.list_colours, markers=self.list_markers, leg_pos=self.leg_pos,
                             margins_y=[self.y_margin_down, y_margin_up_adj], margins_c=self.margins_can,
                             title=self.title_full)
        self.list_new += new
        if self.list_latex:
            self.list_new += draw_latex_lines(self.list_latex,
                                              x_start=self.x_latex, y_start=self.y_latex_top,
                                              y_step=self.y_step, font_size=self.fontsize)
        gStyle.SetErrorX(0.5)  # reset default width
        if not self.plot_errors_x:
            gStyle.SetErrorX(0)  # do not plot horizontal error bars of histograms
        self.save_canvas(can)
        gStyle.SetErrorX(0.5)  # reset default width
        return can, new

    def plot(self):
        print("Observable:", self.var)

        with TFile.Open(self.path_input_file) as self.file_results:
            name_hist_unfold_2d = f"h_ptjet-{self.var}_{self.method}_unfolded_data_0"
            hist_unfold = self.get_object(name_hist_unfold_2d)
            axis_ptjet = get_axis(hist_unfold, 0)

            # Efficiency
            print("Plotting efficiency")
            self.list_obj = self.get_objects("h_pthf_effnew_pr", "h_pthf_effnew_np")
            self.labels_obj = ["prompt", "nonprompt"]
            self.title_full = f";{self.latex_pthf};efficiency"
            self.list_latex = [self.text_alice, self.text_jets, self.get_text_range_ptjet(), self.get_text_range_pthf()]
            self.make_plot(f"efficiency_{self.var}")

            bins_ptjet = (0, 1, 2, 3)
            self.list_obj = self.get_objects(*(f"h_ptjet-pthf_effnew_pr_"
                                               f"{string_range_ptjet(get_bin_limits(axis_ptjet, iptjet + 1))}"
                                               for iptjet in bins_ptjet))
            self.labels_obj = [self.get_text_range_ptjet(iptjet) for iptjet in bins_ptjet]
            self.make_plot(f"efficiency_ptjet_{self.var}")

            # TODO: efficiency (old vs new)

            # loop over jet pt
            for iptjet in (1, 2):
                range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                string_ptjet = string_range_ptjet(range_ptjet)

                # Sideband subtraction
                print("Plotting sideband subtraction")
                # loop over hadron pt
                for ipt in range(self.n_bins_pthf):
                    range_pthf = (self.edges_pthf[ipt], self.edges_pthf[ipt+1])
                    string_pthf = string_range_pthf(range_pthf)

                    self.list_obj = self.get_objects(f'h_ptjet-{self.var}_signal_{string_pthf}_data',
                                                     f'h_ptjet-{self.var}_sideband_{string_pthf}_data',
                                                     f'h_ptjet-{self.var}_subtracted_notscaled_{string_pthf}_data')
                    self.list_obj = [project_hist(h, [1], {0: (iptjet + 1, iptjet + 1)}) for h in self.list_obj]
                    self.labels_obj = ["signal region", "scaled sidebands", "after subtraction"]
                    self.title_full = f";{self.latex_obs};counts"
                    self.list_latex = [self.text_alice, self.text_jets, self.get_text_range_ptjet(iptjet),
                                       self.get_text_range_pthf(ipt, iptjet)]
                    if self.var in ("zg", "rg", "nsd"):
                        self.list_latex.append(self.text_sd)
                    self.make_plot(f"sidebands_{self.var}_{string_ptjet}_{string_pthf}")

                # Feed-down subtraction
                print("Plotting feed-down subtraction")
                self.list_obj = self.get_objects(f'h_ptjet-{self.var}_{self.method}_effscaled_data',
                                                 f'h_ptjet-{self.var}_feeddown_det_final',
                                                 f'h_ptjet-{self.var}_{self.method}_data')
                self.list_obj[0] = self.list_obj[0].Clone(f"{self.list_obj[0].GetName()}_fd_before_{iptjet}")
                self.list_obj[1] = self.list_obj[1].Clone(f"{self.list_obj[1].GetName()}_fd_{iptjet}")
                self.list_obj[2] = self.list_obj[2].Clone(f"{self.list_obj[2].GetName()}_fd_after_{iptjet}")
                axes = list(range(get_dim(self.list_obj[0])))
                self.list_obj = [project_hist(h, axes[1:], {0: (iptjet+1,)*2}) for h in self.list_obj]
                self.labels_obj = ["before subtraction", "feed-down", "after subtraction"]
                self.title_full = f";{self.latex_obs};counts"
                self.list_latex = [self.text_alice, self.text_jets, self.get_text_range_ptjet(iptjet),
                                   self.get_text_range_pthf(-1, iptjet)]
                if self.var in ("zg", "rg", "nsd"):
                    self.list_latex.append(self.text_sd)
                self.make_plot(f"feeddown_{self.var}_{string_ptjet}")

                # TODO: feed-down (after 2D, fraction)

                # Unfolding
                # print("Plotting unfolding")
                # TODO: unfolding (before/after, convergence)

                # Results
                print("Plotting results")
                self.list_obj = [self.get_object(f"h_{self.var}_{self.method}_unfolded_data_"
                                                 f"{string_ptjet}_sel_selfnorm")]
                # TODO: results (systematics)
                # nameobj = "%s_hf_data_%d_syst" % (self.var, iptjet)
                # hf_data_syst = self.get_object(nameobj, file_results)
                self.labels_obj = ["data"]
                self.title_full = self.title_full_default
                self.make_plot(f"results_{self.var}_{string_ptjet}")


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-analysis", "-d", dest="database_analysis",
                        help="analysis database to be used", required=True)
    parser.add_argument("--analysis", "-a", dest="type_ana",
                        help="choose type of analysis", required=True)
    parser.add_argument("--input", "-i", dest="input_file",
                        help="results input file")

    args = parser.parse_args()

    gROOT.SetBatch(True)

    # list_vars = ["zg", "nsd", "rg", "zpar"]
    list_vars = ["zpar"]
    for var in list_vars:
        print(f"Processing observable {var}")
        ploter = Plotter(args.input_file, args.database_analysis, args.type_ana, var)
        ploter.plot()


if __name__ == "__main__":
    main()
