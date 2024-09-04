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
Calculate and plot systematic uncertainties
"""

# pylint: disable=too-many-lines, too-many-instance-attributes, too-many-statements, too-many-locals
# pylint: disable=too-many-nested-blocks, too-many-branches, consider-using-f-string

import argparse
import logging
import os
from array import array
from math import sqrt
from pathlib import Path
from functools import reduce
import numpy as np

import yaml
from ROOT import TLegend  # , TLine
from ROOT import TH1F, TCanvas, TFile, TGraphAsymmErrors, TLatex, gROOT, gStyle

from machine_learning_hep.do_variations import (
    format_varlabel,
    format_varname,
    healthy_structure,
)
from machine_learning_hep.logger import get_logger
from machine_learning_hep.analysis.analyzer_jets import string_range_ptjet

# HF specific imports
from machine_learning_hep.utilities import (  # make_plot,
    combine_graphs,
    draw_latex,
    get_colour,
    get_marker,
    get_plot_range,
    get_y_window_gr,
    get_y_window_his,
    make_message_notfound,
    print_histogram,
    reset_graph_outside_range,
    reset_hist_outside_range,
    setup_canvas,
    setup_histogram,
    setup_legend,
    setup_tgraph,
)
from machine_learning_hep.utils.hist import get_axis, bin_array, get_bin_limits

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


class AnalyzerJetSystematics:
    def __init__(self, path_database_analysis: str, typean: str):
        self.logger = get_logger()
        self.typean = typean
        self.var = ""
        self.method = "sidesub"
        self.logger.setLevel(logging.INFO)
        self.verbose = False

        with open(path_database_analysis, "r", encoding="utf-8") as file_in:
            db_analysis = yaml.safe_load(file_in)
        case = list(db_analysis.keys())[0]
        self.datap = db_analysis[case]
        self.db_typean = self.datap["analysis"][self.typean]

        # plotting
        # LaTeX string
        self.latex_hadron = self.db_typean["latexnamehadron"]
        self.latex_ptjet = "#it{p}_{T}^{jet ch}"

        # binning of hadron pt
        self.edges_pthf_min = self.db_typean["sel_an_binmin"]
        self.edges_pthf_max = self.db_typean["sel_an_binmax"]
        # self.n_bins_pthf = len(self.edges_pthf_min)

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
        self.n_bins_ptjet_gen = len(self.edges_ptjet_gen) - 1
        self.ptjet_gen_min = self.edges_ptjet_gen[0]
        self.ptjet_gen_max = self.edges_ptjet_gen[-1]
        self.edges_ptjet_gen_min = self.edges_ptjet_gen[:-1]
        self.edges_ptjet_gen_max = self.edges_ptjet_gen[1:]

        # unfolding
        self.niter_unfolding = self.db_typean["unfolding_iterations"]
        self.choice_iter_unfolding = self.db_typean["unfolding_iterations_sel"]

        # import parameters of variations from the variation database

        path_database_variations = self.db_typean["variations_db"]
        if not path_database_variations:
            self.logger.critical(make_message_notfound("the variation database"))
        if "/" not in path_database_variations:
            path_database_variations = f"{os.path.dirname(path_database_analysis)}/{path_database_variations}"
        with open(path_database_variations, "r", encoding="utf-8") as file_sys:
            db_variations = yaml.safe_load(file_sys)

        if not healthy_structure(db_variations):
            self.logger.critical("Bad structure of the variation database.")
        db_variations = db_variations["categories"]
        self.systematic_catnames = [catname for catname, val in db_variations.items() if val["activate"]]
        self.n_sys_cat = len(self.systematic_catnames)
        self.systematic_catlabels = [""] * self.n_sys_cat
        self.systematic_catgroups = [""] * self.n_sys_cat
        self.systematic_varnames = [None] * self.n_sys_cat
        self.systematic_varlabels = [None] * self.n_sys_cat
        self.systematic_variations = [0] * self.n_sys_cat
        self.systematic_correlation = [None] * self.n_sys_cat
        self.systematic_rms = [False] * self.n_sys_cat
        self.systematic_symmetrise = [False] * self.n_sys_cat
        self.systematic_rms_both_sides = [False] * self.n_sys_cat
        self.powheg_nonprompt_varnames = []
        self.powheg_nonprompt_varlabels = []
        self.edges_ptjet_gen_sys = None
        self.edges_ptjet_rec_sys = None
        for c, catname in enumerate(self.systematic_catnames):
            self.systematic_catlabels[c] = db_variations[catname]["label"]
            self.systematic_catgroups[c] = db_variations[catname].get("group", self.systematic_catlabels[c])
            self.systematic_varnames[c] = []
            self.systematic_varlabels[c] = []
            for varname, val in db_variations[catname]["variations"].items():
                if catname == "binning" and varname == "pt_jet" and any(val["activate"]):
                    self.edges_ptjet_gen_sys = val["diffs"]["analysis"][self.typean]["bins_ptjet"]
                    self.edges_ptjet_rec_sys = val["diffs"]["analysis"][self.typean]["bins_ptjet"]
                n_var = len(val["activate"])
                for a, act in enumerate(val["activate"]):
                    if act:
                        varname_i = format_varname(varname, a, n_var)
                        varlabel_i = format_varlabel(val["label"], a, n_var)
                        self.systematic_varnames[c].append(varname_i)
                        self.systematic_varlabels[c].append(varlabel_i)
                        if catname == "feeddown":
                            self.powheg_nonprompt_varnames.append(varname_i)
                            self.powheg_nonprompt_varlabels.append(varlabel_i)
            self.systematic_variations[c] = len(self.systematic_varnames[c])
            self.systematic_correlation[c] = db_variations[catname]["correlation"]
            self.systematic_rms[c] = db_variations[catname]["rms"]
            self.systematic_symmetrise[c] = db_variations[catname]["symmetrise"]
            self.systematic_rms_both_sides[c] = db_variations[catname]["rms_both_sides"]
        self.systematic_catgroups_list = list(dict.fromkeys(self.systematic_catgroups))
        self.n_sys_gr = len(self.systematic_catgroups_list)

        # output directories with results
        self.dir_result_mc = self.db_typean["mc"]["resultsallp"]
        self.dir_result_data = self.db_typean["data"]["resultsallp"]

        self.string_default = "default/default"
        if self.string_default not in self.dir_result_data:
            self.logger.critical("Not a default database! Cannot run systematics.")

        # input files
        file_result_name = self.datap["files_names"]["resultfilename"]
        self.file_results = os.path.join(self.dir_result_data, file_result_name)
        self.file_efficiency = self.file_results

        # official figures
        self.fig_formats = ["pdf", "png"]
        self.size_can = [800, 800]
        self.offsets_axes = [0.8, 1.1]
        self.margins_can = [0.1, 0.13, 0.05, 0.03]
        self.fontsize = 0.035
        self.opt_leg_g = "FP"  # for systematic uncertanties in the legend
        self.opt_plot_g = "2"
        self.x_latex = 0.18
        self.y_latex_top = 0.88
        self.y_step = 0.05
        # text
        self.text_alice = "ALICE Preliminary, pp, #sqrt{#it{s}} = 13.6 TeV"
        # self.text_alice = "#bf{ALICE}, pp, #sqrt{#it{s}} = 13.6 TeV"
        self.text_jets = "%s-tagged charged jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.latex_hadron
        self.text_ptjet = "%g #leq %s < %g GeV/#it{c}, |#it{#eta}_{jet ch}| < 0.5"
        self.text_pth = "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, |#it{y}_{%s}| < 0.8"
        self.text_sd = "Soft drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"
        self.text_acc_h = "|#it{y}| < 0.8"
        self.text_powheg = "POWHEG + PYTHIA 6 + EvtGen"

        # output directory for figures
        self.dir_out_figs = Path(f"{os.path.expandvars(self.dir_result_data)}/fig/sys")
        for fmt in self.fig_formats:
            (self.dir_out_figs / fmt).mkdir(parents=True, exist_ok=True)

        # output file for histograms
        self.file_sys_out = TFile.Open(f"{self.dir_result_data}/systematics.root", "recreate")

        self.debug = True
        if self.debug:
            print("Categories: ", self.systematic_catnames)
            print("Category labels: ", self.systematic_catlabels)
            print("Category Groups: ", self.systematic_catgroups)
            print("Category Groups unique: ", self.systematic_catgroups_list, self.n_sys_gr)
            print("Numbers of variations: ", self.systematic_variations)
            print("Variations: ", self.systematic_varnames)
            print("Variation labels: ", self.systematic_varlabels)
            print("Correlation: ", self.systematic_correlation)
            print("RMS: ", self.systematic_rms)
            print("Symmetrisation: ", self.systematic_symmetrise)
            print("RMS both sides: ", self.systematic_rms_both_sides)
            print("Feed-down variations: ", self.powheg_nonprompt_varnames)
            print("Jet pT rec variations: ", self.edges_ptjet_rec_sys)
            print("Jet pT gen variations: ", self.edges_ptjet_gen_sys)

    def cfg(self, param, default=None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
                      param.split("."), self.db_typean)

    def save_canvas(self, can, name: str):
        """Save canvas"""
        for fmt in self.fig_formats:
            can.SaveAs(f"{self.dir_out_figs}/{fmt}/{name}.{fmt}")

    def crop_histogram(self, hist):
        """Constrain x range of histogram and reset outside bins."""
        if self.var not in x_range:
            return
        # hist.GetXaxis().SetRangeUser(round(x_range[self.var][0], 2), round(x_range[self.var][1], 2))
        hist.GetXaxis().SetLimits(round(x_range[self.var][0], 2), round(x_range[self.var][1], 2))
        reset_hist_outside_range(hist, *x_range[self.var])

    def crop_graph(self, graph):
        """Constrain x range of graph and reset outside points."""
        if self.var not in x_range:
            return
        # graph.GetXaxis().SetRangeUser(round(x_range[self.var][0], 2), round(x_range[self.var][1], 2))
        graph.GetXaxis().SetLimits(round(x_range[self.var][0], 2), round(x_range[self.var][1], 2))
        reset_graph_outside_range(graph, *x_range[self.var])

    def get_suffix_ptjet(self, iptjet: int):
        return string_range_ptjet((self.edges_ptjet_gen[iptjet], self.edges_ptjet_gen[iptjet + 1]))

    def process(self, list_vars: "list[str]"):
        """Do systematics for all variables"""
        for var in list_vars:
            # self.logger.info("Processing observable %s", var)
            print("Processing observable %s" % var)
            self.do_jet_systematics(var)

    def do_jet_systematics(self, var: str):
        """Do systematics for one variable"""
        self.var = var

        latex_obs = self.db_typean["observables"][self.var]["label"]
        latex_y = self.db_typean["observables"][self.var]["label_y"]
        # axis titles
        # title_x = latex_obs
        # title_y = latex_y
        # title_full = ";%s;%s" % (title_x, title_y)
        # title_full_ratio = ";%s;data/MC: ratio of %s" % (title_x, title_y)

        # binning of observable (z, shape,...)
        # reconstruction level
        if binning := self.cfg(f'observables.{self.var}.bins_det_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{self.var}.bins_det_fix'):
            bins_tmp = bin_array(*binning)
        elif binning := self.cfg(f'observables.{self.var}.bins_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{self.var}.bins_fix'):
            bins_tmp = bin_array(*binning)
        else:
            self.logger.error('No binning specified for %s, using defaults', self.var)
            bins_tmp = bin_array(10, 0., 1.)
        binning_obs_rec = bins_tmp
        # n_bins_obs_rec = len(binning_obs_rec) - 1
        # obs_rec_min = float(binning_obs_rec[0])
        # obs_rec_max = float(binning_obs_rec[-1])
        edges_obs_rec = binning_obs_rec

        # generator level
        if binning := self.cfg(f'observables.{self.var}.bins_gen_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{self.var}.bins_gen_fix'):
            bins_tmp = bin_array(*binning)
        elif binning := self.cfg(f'observables.{self.var}.bins_var'):
            bins_tmp = np.asarray(binning, 'd')
        elif binning := self.cfg(f'observables.{self.var}.bins_fix'):
            bins_tmp = bin_array(*binning)
        else:
            self.logger.error('No binning specified for %s, using defaults', self.var)
            bins_tmp = bin_array(10, 0., 1.)
        binning_obs_gen = bins_tmp
        n_bins_obs_gen = len(binning_obs_gen) - 1
        obs_gen_min = float(binning_obs_gen[0])
        obs_gen_max = float(binning_obs_gen[-1])
        edges_obs_gen = binning_obs_gen

        print("Rec obs edges:", edges_obs_rec, "Gen obs edges:", edges_obs_gen)
        print("Rec ptjet edges:", self.edges_ptjet_rec, "Gen ptjet edges:", self.edges_ptjet_gen)

        # Open input files for default results.
        path_def = self.file_results
        if not (input_file_default := TFile.Open(path_def)):
            self.logger.critical(make_message_notfound(path_def))
        path_eff = self.file_efficiency
        if not (eff_file_default := TFile.Open(path_eff)):
            self.logger.critical(make_message_notfound(path_eff))

        # get the default (central value) result histograms

        input_histograms_default = []
        eff_default = []
        for iptjet in range(self.n_bins_ptjet_gen):
            name_hist_unfold_2d = f"h_ptjet-{self.var}_{self.method}_unfolded_data_0"
            if not (hist_unfold := input_file_default.Get(name_hist_unfold_2d)):
                self.logger.critical(make_message_notfound(name_hist_unfold_2d, path_def))
            axis_ptjet = get_axis(hist_unfold, 0)
            range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
            name_his = f"h_{self.var}_{self.method}_unfolded_data_{string_range_ptjet(range_ptjet)}_sel_selfnorm"
            input_histograms_default.append(input_file_default.Get(name_his))
            if not input_histograms_default[iptjet]:
                self.logger.critical(make_message_notfound(name_his, path_def))
            self.crop_histogram(input_histograms_default[iptjet])
            print(f"Default histogram ({range_ptjet[0]} to {range_ptjet[1]})")
            print_histogram(input_histograms_default[iptjet], self.verbose)
            # name_eff = f"h_ptjet-pthf_effnew_pr_{string_range_ptjet(range_ptjet)}"
            name_eff = "h_pthf_effnew_pr"
            eff_default.append(eff_file_default.Get(name_eff))
            if not eff_default[iptjet]:
                self.logger.critical(make_message_notfound(name_eff, path_eff))
            print(f"Default efficiency ({range_ptjet[0]} to {range_ptjet[1]})")
            print_histogram(eff_default[iptjet], self.verbose)

        # get the files containing result variations

        path_input_files_sys = []
        input_files_sys = []
        input_files_eff = []
        for sys_cat in range(self.n_sys_cat):
            path_input_files_sysvar = []
            input_files_sysvar = []
            input_files_sysvar_eff = []
            for sys_var, varname in enumerate(self.systematic_varnames[sys_cat]):
                path = path_def.replace(self.string_default, self.systematic_catnames[sys_cat] + "/" + varname)
                path_input_files_sysvar.append(path)
                input_files_sysvar.append(TFile.Open(path))
                eff_file = path_eff.replace(self.string_default, self.systematic_catnames[sys_cat] + "/" + varname)
                input_files_sysvar_eff.append(TFile.Open(eff_file))
                if not input_files_sysvar[sys_var]:
                    self.logger.critical(make_message_notfound(path))
                if not input_files_sysvar_eff[sys_var]:
                    self.logger.critical(make_message_notfound(eff_file))
            path_input_files_sys.append(path_input_files_sysvar)
            input_files_sys.append(input_files_sysvar)
            input_files_eff.append(input_files_sysvar_eff)

        # get the variation result histograms

        input_histograms_sys = []
        input_histograms_sys_eff = []
        for iptjet in range(self.n_bins_ptjet_gen):
            input_histograms_syscat = []
            input_histograms_syscat_eff = []
            for sys_cat in range(self.n_sys_cat):
                input_histograms_syscatvar = []
                input_histograms_eff = []
                for sys_var in range(self.systematic_variations[sys_cat]):
                    print(
                        "Variation: %s, %s"
                        % (self.systematic_catnames[sys_cat], self.systematic_varnames[sys_cat][sys_var])
                    )
                    name_hist_unfold_2d = f"h_ptjet-{self.var}_{self.method}_unfolded_data_0"
                    if not (hist_unfold := input_files_sys[sys_cat][sys_var].Get(name_hist_unfold_2d)):
                        self.logger.critical(
                            make_message_notfound(name_hist_unfold_2d, path_input_files_sys[sys_cat][sys_var])
                        )
                    axis_ptjet = get_axis(hist_unfold, 0)
                    string_catvar = self.systematic_catnames[sys_cat] + "/" + self.systematic_varnames[sys_cat][sys_var]
                    range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                    name_his = f"h_{self.var}_{self.method}_unfolded_data_{string_range_ptjet(range_ptjet)}_sel_selfnorm"
                    sys_var_histo = input_files_sys[sys_cat][sys_var].Get(name_his)
                    path_file = path_def.replace(self.string_default, string_catvar)
                    if not sys_var_histo:
                        self.logger.critical(make_message_notfound(name_his, path_file))
                    # name_eff = f"h_ptjet-pthf_effnew_pr_{string_range_ptjet(range_ptjet)}"
                    name_eff = "h_pthf_effnew_pr"
                    sys_var_histo_eff = input_files_eff[sys_cat][sys_var].Get(name_eff)
                    path_eff_file = path_eff.replace(self.string_default, string_catvar)
                    if not sys_var_histo_eff:
                        self.logger.critical(make_message_notfound(name_eff, path_eff_file))
                    self.crop_histogram(sys_var_histo)
                    input_histograms_syscatvar.append(sys_var_histo)
                    input_histograms_eff.append(sys_var_histo_eff)
                    print_histogram(sys_var_histo_eff, self.verbose)
                    print_histogram(sys_var_histo, self.verbose)
                    if self.debug:
                        print(
                            "Variation: %s, %s: got histogram %s from file %s"
                            % (
                                self.systematic_catnames[sys_cat],
                                self.systematic_varnames[sys_cat][sys_var],
                                name_his,
                                path_file,
                            )
                        )
                        print(
                            "Variation: %s, %s: got efficiency histogram %s from file %s"
                            % (
                                self.systematic_catnames[sys_cat],
                                self.systematic_varnames[sys_cat][sys_var],
                                name_eff,
                                path_eff_file,
                            )
                        )
                input_histograms_syscat.append(input_histograms_syscatvar)
                input_histograms_syscat_eff.append(input_histograms_eff)
            input_histograms_sys.append(input_histograms_syscat)
            input_histograms_sys_eff.append(input_histograms_syscat_eff)

        # plot the variations

        print("Categories: %d", self.n_sys_cat)
        # self.logger.info("Categories: %d", self.n_sys_cat)

        for iptjet in range(self.n_bins_ptjet_gen):
            # plot all the variations together
            suffix = self.get_suffix_ptjet(iptjet)
            nsys = 0
            csysvar = TCanvas("csysvar_%s" % suffix, "systematic variations" + suffix)
            setup_canvas(csysvar)
            leg_sysvar = TLegend(0.75, 0.15, 0.95, 0.85, "variation")
            setup_legend(leg_sysvar)
            leg_sysvar.AddEntry(input_histograms_default[iptjet], "default", "P")
            setup_histogram(input_histograms_default[iptjet])
            l_his_all = []
            for l_cat in input_histograms_sys[iptjet]:
                for his_var in l_cat:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
            l_his_all.append(input_histograms_default[iptjet])
            y_min, y_max = get_y_window_his(l_his_all, False)
            y_margin_up = 0.15
            y_margin_down = 0.05
            input_histograms_default[iptjet].GetYaxis().SetRangeUser(
                *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
            )
            input_histograms_default[iptjet].SetTitle("")
            input_histograms_default[iptjet].SetXTitle(latex_obs)
            input_histograms_default[iptjet].SetYTitle(latex_y)
            input_histograms_default[iptjet].Draw()
            print_histogram(input_histograms_default[iptjet], self.verbose)

            self.logger.info("Categories: %d", self.n_sys_cat)
            print("Categories: %d" % self.n_sys_cat)

            for sys_cat in range(self.n_sys_cat):
                self.logger.info("Category: %s", self.systematic_catlabels[sys_cat])
                print("Category: %s" % self.systematic_catlabels[sys_cat])
                for sys_var in range(self.systematic_variations[sys_cat]):
                    self.logger.info("Variation: %s", self.systematic_varlabels[sys_cat][sys_var])
                    print("Variation: %s" % self.systematic_varlabels[sys_cat][sys_var])
                    leg_sysvar.AddEntry(
                        input_histograms_sys[iptjet][sys_cat][sys_var],
                        ("%s, %s" % (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var])),
                        "P",
                    )
                    self.logger.info(
                        "Adding label %s",
                        ("%s, %s" % (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var])),
                    )
                    print(
                        "Adding label %s"
                        % ("%s, %s" % (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var]))
                    )
                    setup_histogram(input_histograms_sys[iptjet][sys_cat][sys_var], get_colour(nsys + 1))
                    input_histograms_sys[iptjet][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1

            latex = TLatex(
                0.15,
                0.82,
                "%g #leq %s < %g GeV/#it{c}"
                % (self.edges_ptjet_gen_min[iptjet], self.latex_ptjet, self.edges_ptjet_gen_max[iptjet]),
            )
            draw_latex(latex)
            # leg_sysvar.Draw("same")
            self.save_canvas(csysvar, f"sys_var_{self.var}_{suffix}_all")

            # plot the variations for each category separately

            for sys_cat in range(self.n_sys_cat):
                suffix2 = self.systematic_catnames[sys_cat]
                nsys = 0
                csysvar_each = TCanvas("csysvar_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix)
                setup_canvas(csysvar_each)
                csysvar_each.SetRightMargin(0.25)
                leg_sysvar_each = TLegend(0.77, 0.2, 0.95, 0.85, self.systematic_catlabels[sys_cat])  # Rg
                setup_legend(leg_sysvar_each)
                leg_sysvar_each.AddEntry(input_histograms_default[iptjet], "default", "P")
                setup_histogram(input_histograms_default[iptjet])
                l_his_all = []
                for his_var in input_histograms_sys[iptjet][sys_cat]:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                l_his_all.append(input_histograms_default[iptjet])
                y_min, y_max = get_y_window_his(l_his_all, False)
                y_margin_up = 0.15
                y_margin_down = 0.05
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        input_histograms_default[iptjet].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        input_histograms_default[iptjet].SetTitle("")
                        input_histograms_default[iptjet].SetXTitle(latex_obs)
                        input_histograms_default[iptjet].SetYTitle(latex_y)
                        input_histograms_default[iptjet].Draw()
                    leg_sysvar_each.AddEntry(
                        input_histograms_sys[iptjet][sys_cat][sys_var], self.systematic_varlabels[sys_cat][sys_var], "P"
                    )
                    setup_histogram(
                        input_histograms_sys[iptjet][sys_cat][sys_var], get_colour(nsys + 1), get_marker(nsys + 1)
                    )
                    input_histograms_sys[iptjet][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.edges_ptjet_gen_min[iptjet], self.latex_ptjet, self.edges_ptjet_gen_max[iptjet]),
                )
                draw_latex(latex)
                leg_sysvar_each.Draw("same")
                self.save_canvas(csysvar_each, f"sys_var_{self.var}_{suffix}_{suffix2}")

                # plot ratios to the default

                nsys = 0
                csysvar_ratio = TCanvas(
                    "csysvar_ratio_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix
                )
                setup_canvas(csysvar_ratio)
                csysvar_ratio.SetRightMargin(0.25)
                leg_sysvar_ratio = TLegend(0.77, 0.2, 0.95, 0.85, self.systematic_catlabels[sys_cat])  # Rg
                setup_legend(leg_sysvar_ratio)
                histo_ratio = []
                for sys_var in range(self.systematic_variations[sys_cat]):
                    default_his = input_histograms_default[iptjet].Clone("default_his")
                    var_his = input_histograms_sys[iptjet][sys_cat][sys_var].Clone("var_his")
                    var_his.Divide(default_his)
                    histo_ratio.append(var_his)
                l_his_all = []
                for his_var in histo_ratio:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                y_min, y_max = get_y_window_his(l_his_all, False)
                y_margin_up = 0.15
                y_margin_down = 0.05

                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        histo_ratio[sys_var].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        histo_ratio[sys_var].SetTitle("")
                        histo_ratio[sys_var].SetXTitle(latex_obs)
                        histo_ratio[sys_var].SetYTitle("variation/default")
                        histo_ratio[sys_var].Draw()
                    leg_sysvar_ratio.AddEntry(histo_ratio[sys_var], self.systematic_varlabels[sys_cat][sys_var], "P")
                    setup_histogram(histo_ratio[sys_var], get_colour(nsys + 1), get_marker(nsys + 1))
                    histo_ratio[sys_var].Draw("same")
                    nsys = nsys + 1
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.edges_ptjet_gen_min[iptjet], self.latex_ptjet, self.edges_ptjet_gen_max[iptjet]),
                )
                draw_latex(latex)
                # line = TLine(obs_rec_min, 1, obs_rec_max, 1)
                # line.SetLineColor(1)
                # line.Draw()
                leg_sysvar_ratio.Draw("same")
                self.save_canvas(csysvar_ratio, f"sys_var_{self.var}_{suffix}_{suffix2}_ratio")

                # Plot efficiency variations

                csysvar_eff = TCanvas(
                    "csysvar_eff_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix
                )
                setup_canvas(csysvar_eff)
                csysvar_eff.SetRightMargin(0.25)
                leg_sysvar_eff = TLegend(
                    0.77,
                    0.2,
                    0.95,
                    0.85,
                    self.systematic_catlabels[sys_cat],
                )  # Rg
                setup_legend(leg_sysvar_eff)
                leg_sysvar_eff.AddEntry(eff_default[iptjet], "default", "P")
                setup_histogram(eff_default[iptjet])
                l_his_all = []
                for his_var in input_histograms_sys_eff[iptjet][sys_cat]:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                l_his_all.append(eff_default[iptjet])
                y_min, y_max = get_y_window_his(l_his_all)
                y_margin_up = 0.15
                y_margin_down = 0.05
                nsys = 0
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        eff_default[iptjet].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        eff_default[iptjet].SetTitle("")
                        eff_default[iptjet].SetXTitle("#it{p}_{T}^{%s} (GeV/#it{c})" % self.latex_hadron)
                        eff_default[iptjet].SetYTitle("prompt %s-jet efficiency" % self.latex_hadron)
                        eff_default[iptjet].Draw()
                    leg_sysvar_eff.AddEntry(
                        input_histograms_sys_eff[iptjet][sys_cat][sys_var],
                        self.systematic_varlabels[sys_cat][sys_var],
                        "P",
                    )
                    setup_histogram(
                        input_histograms_sys_eff[iptjet][sys_cat][sys_var],
                        get_colour(nsys + 1),
                        get_marker(nsys + 1),
                    )
                    input_histograms_sys_eff[iptjet][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.edges_ptjet_gen_min[iptjet], self.latex_ptjet, self.edges_ptjet_gen_max[iptjet]),
                )
                draw_latex(latex)
                leg_sysvar_eff.Draw("same")
                self.save_canvas(csysvar_eff, f"sys_var_{self.var}_{suffix}_{suffix2}_eff")

                nsys = 0

                # Plot ratios of efficiency variations to the default efficiency

                csysvar_eff_ratio = TCanvas(
                    "csysvar_eff_ratio_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix
                )
                setup_canvas(csysvar_eff_ratio)
                csysvar_eff_ratio.SetRightMargin(0.25)
                leg_sysvar_eff_ratio = TLegend(0.77, 0.2, 0.95, 0.85, self.systematic_catlabels[sys_cat])  # Rg
                setup_legend(leg_sysvar_eff_ratio)
                histo_ratio = []
                for sys_var in range(self.systematic_variations[sys_cat]):
                    default_his = eff_default[iptjet].Clone("default_his")
                    var_his = input_histograms_sys_eff[iptjet][sys_cat][sys_var].Clone("var_his")
                    var_his.Divide(default_his)
                    histo_ratio.append(var_his)
                l_his_all = []
                for his_var in histo_ratio:
                    if his_var.Integral() != 0:
                        l_his_all.append(his_var)
                y_min, y_max = get_y_window_his(l_his_all, False)
                y_margin_up = 0.05
                y_margin_down = 0.05
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        histo_ratio[sys_var].GetYaxis().SetRangeUser(
                            *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                        )
                        histo_ratio[sys_var].SetTitle("")
                        histo_ratio[sys_var].SetXTitle("#it{p}_{T}")
                        histo_ratio[sys_var].SetYTitle("efficiency (variation/default)")
                        histo_ratio[sys_var].Draw()
                    leg_sysvar_eff_ratio.AddEntry(
                        histo_ratio[sys_var], self.systematic_varlabels[sys_cat][sys_var], "P"
                    )
                    setup_histogram(histo_ratio[sys_var], get_colour(nsys + 1), get_marker(nsys + 1))
                    histo_ratio[sys_var].Draw("same")
                    nsys = nsys + 1
                # line = TLine(obs_rec_min, 1, obs_rec_max, 1)
                # line.SetLineColor(1)
                # line.Draw()
                latex = TLatex(
                    0.15,
                    0.82,
                    "%g #leq %s < %g GeV/#it{c}"
                    % (self.edges_ptjet_gen_min[iptjet], self.latex_ptjet, self.edges_ptjet_gen_max[iptjet]),
                )
                draw_latex(latex)
                leg_sysvar_eff_ratio.Draw("same")
                self.save_canvas(csysvar_eff_ratio, f"sys_var_{self.var}_{suffix}_{suffix2}_eff_ratio")

        # calculate the systematic uncertainties

        # list of absolute upward uncertainties for all categories, shape bins, pt_jet bins
        sys_up = []
        # list of absolute downward uncertainties for all categories, shape bins, pt_jet bins
        sys_down = []
        # list of combined absolute upward uncertainties for all shape bins, pt_jet bins
        sys_up_full = []
        # list of combined absolute downward uncertainties for all shape bins, pt_jet bins
        sys_down_full = []
        for iptjet in range(self.n_bins_ptjet_gen):
            # list of absolute upward uncertainties for all categories and shape bins in a given pt_jet bin
            sys_up_jetpt = []
            # list of absolute downward uncertainties for all categories and shape bins in a given pt_jet bin
            sys_down_jetpt = []
            # list of combined absolute upward uncertainties for all shape bins in a given pt_jet bin
            sys_up_z_full = []
            # list of combined absolute upward uncertainties for all shape bins in a given pt_jet bin
            sys_down_z_full = []
            for ibinshape in range(n_bins_obs_gen):
                # list of absolute upward uncertainties for all categories in a given (pt_jet, shape) bin
                sys_up_z = []
                # list of absolute downward uncertainties for all categories in a given (pt_jet, shape) bin
                sys_down_z = []
                # combined absolute upward uncertainty in a given (pt_jet, shape) bin
                error_full_up = 0
                # combined absolute downward uncertainty in a given (pt_jet, shape) bin
                error_full_down = 0
                for sys_cat in range(self.n_sys_cat):
                    # absolute upward uncertainty for a given category in a given (pt_jet, shape) bin
                    error_var_up = 0
                    # absolute downward uncertainty for a given category in a given (pt_jet, shape) bin
                    error_var_down = 0
                    count_sys_up = 0
                    count_sys_down = 0
                    error = 0
                    for sys_var in range(self.systematic_variations[sys_cat]):
                        out_sys = False
                        # FIXME exception for the untagged bin pylint: disable=fixme
                        bin_first = 1
                        # bin_first = 2 if "untagged" in self.systematic_varlabels[sys_cat][sys_var] else 1
                        # FIXME exception for the untagged bin pylint: disable=fixme
                        if input_histograms_sys[iptjet][sys_cat][sys_var].Integral() == 0:
                            error = 0
                            out_sys = True
                        else:
                            error = input_histograms_sys[iptjet][sys_cat][sys_var].GetBinContent(
                                ibinshape + bin_first
                            ) - input_histograms_default[iptjet].GetBinContent(ibinshape + 1)
                        if error >= 0:
                            if self.systematic_rms[sys_cat] is True:
                                error_var_up += error * error
                                if not out_sys:
                                    count_sys_up = count_sys_up + 1
                            else:
                                error_var_up = max(error_var_up, error)
                        else:
                            if self.systematic_rms[sys_cat] is True:
                                if self.systematic_rms_both_sides[sys_cat] is True:
                                    error_var_up += error * error
                                    if not out_sys:
                                        count_sys_up = count_sys_up + 1
                                else:
                                    error_var_down += error * error
                                    if not out_sys:
                                        count_sys_down = count_sys_down + 1
                            else:
                                error_var_down = max(error_var_down, abs(error))
                    if self.systematic_rms[sys_cat] is True:
                        if count_sys_up != 0:
                            error_var_up = error_var_up / count_sys_up
                        else:
                            error_var_up = 0.0
                        error_var_up = sqrt(error_var_up)
                        if count_sys_down != 0:
                            error_var_down = error_var_down / count_sys_down
                        else:
                            error_var_down = 0.0
                        if self.systematic_rms_both_sides[sys_cat] is True:
                            error_var_down = error_var_up
                        else:
                            error_var_down = sqrt(error_var_down)
                    if self.systematic_symmetrise[sys_cat] is True:
                        if error_var_up > error_var_down:
                            error_var_down = error_var_up
                        else:
                            error_var_up = error_var_down
                    error_full_up += error_var_up * error_var_up
                    error_full_down += error_var_down * error_var_down
                    sys_up_z.append(error_var_up)
                    sys_down_z.append(error_var_down)
                error_full_up = sqrt(error_full_up)
                sys_up_z_full.append(error_full_up)
                error_full_down = sqrt(error_full_down)
                sys_down_z_full.append(error_full_down)
                sys_up_jetpt.append(sys_up_z)
                sys_down_jetpt.append(sys_down_z)
            sys_up_full.append(sys_up_z_full)
            sys_down_full.append(sys_down_z_full)
            sys_up.append(sys_up_jetpt)
            sys_down.append(sys_down_jetpt)

        # create graphs to plot the uncertainties

        tgsys = []  # list of graphs with combined absolute uncertainties for all pt_jet bins
        tgsys_cat = []  # list of graphs with relative uncertainties for all categories, pt_jet bins
        full_unc_up = []  # list of relative uncertainties for all categories, pt_jet bins
        full_unc_down = []  # list of relative uncertainties for all categories, pt_jet bins
        for iptjet in range(self.n_bins_ptjet_gen):
            # combined uncertainties

            shapebins_centres = []
            shapebins_contents = []
            shapebins_widths_up = []
            shapebins_widths_down = []
            shapebins_error_up = []
            shapebins_error_down = []
            rel_unc_up = []
            rel_unc_down = []
            unc_rel_min = 100.0
            unc_rel_max = 0.0
            for ibinshape in range(n_bins_obs_gen):
                shapebins_centres.append(input_histograms_default[iptjet].GetBinCenter(ibinshape + 1))
                val = input_histograms_default[iptjet].GetBinContent(ibinshape + 1)
                shapebins_contents.append(val)
                shapebins_widths_up.append(input_histograms_default[iptjet].GetBinWidth(ibinshape + 1) * 0.5)
                shapebins_widths_down.append(input_histograms_default[iptjet].GetBinWidth(ibinshape + 1) * 0.5)
                err_up = sys_up_full[iptjet][ibinshape]
                err_down = sys_down_full[iptjet][ibinshape]
                shapebins_error_up.append(err_up)
                shapebins_error_down.append(err_down)
                if val > 0:
                    unc_rel_up = err_up / val
                    unc_rel_down = err_down / val
                    unc_rel_min = min(unc_rel_min, unc_rel_up, unc_rel_down)
                    unc_rel_max = max(unc_rel_max, unc_rel_up, unc_rel_down)
                    rel_unc_up.append(unc_rel_up)
                    rel_unc_down.append(unc_rel_down)
                    print(
                        "total rel. syst. unc.: ",
                        self.edges_ptjet_gen_min[iptjet],
                        " ",
                        self.edges_ptjet_gen_max[iptjet],
                        " ",
                        edges_obs_gen[ibinshape],
                        " ",
                        edges_obs_gen[ibinshape + 1],
                        " ",
                        unc_rel_up,
                        " ",
                        unc_rel_down,
                    )
                else:
                    rel_unc_up.append(0.0)
                    rel_unc_down.append(0.0)
            print(f"total rel. syst. unc. (%): min. {(100. * unc_rel_min):.2g}, max. {(100. * unc_rel_max):.2g}")
            shapebins_centres_array = array("d", shapebins_centres)
            shapebins_contents_array = array("d", shapebins_contents)
            shapebins_widths_up_array = array("d", shapebins_widths_up)
            shapebins_widths_down_array = array("d", shapebins_widths_down)
            shapebins_error_up_array = array("d", shapebins_error_up)
            shapebins_error_down_array = array("d", shapebins_error_down)
            tgsys.append(
                TGraphAsymmErrors(
                    n_bins_obs_gen,
                    shapebins_centres_array,
                    shapebins_contents_array,
                    shapebins_widths_down_array,
                    shapebins_widths_up_array,
                    shapebins_error_down_array,
                    shapebins_error_up_array,
                )
            )
            full_unc_up.append(rel_unc_up)
            full_unc_down.append(rel_unc_down)
            # relative uncertainties per category

            tgsys_cat_z = []  # list of graphs with relative uncertainties for all categories in a given pt_jet bin
            for sys_cat in range(self.n_sys_cat):
                shapebins_contents_cat = []
                shapebins_error_up_cat = []
                shapebins_error_down_cat = []
                for ibinshape in range(n_bins_obs_gen):
                    shapebins_contents_cat.append(0)
                    if abs(input_histograms_default[iptjet].GetBinContent(ibinshape + 1)) < 1.0e-7:
                        print("WARNING!!! Input histogram at bin", iptjet, " equal 0", suffix)
                        e_up = 0
                        e_down = 0
                    else:
                        e_up = sys_up[iptjet][ibinshape][sys_cat] / input_histograms_default[iptjet].GetBinContent(
                            ibinshape + 1
                        )
                        e_down = sys_down[iptjet][ibinshape][sys_cat] / input_histograms_default[iptjet].GetBinContent(
                            ibinshape + 1
                        )
                    shapebins_error_up_cat.append(e_up)
                    shapebins_error_down_cat.append(e_down)
                shapebins_contents_cat_array = array("d", shapebins_contents_cat)
                shapebins_error_up_cat_array = array("d", shapebins_error_up_cat)
                shapebins_error_down_cat_array = array("d", shapebins_error_down_cat)
                tgsys_cat_z.append(
                    TGraphAsymmErrors(
                        n_bins_obs_gen,
                        shapebins_centres_array,
                        shapebins_contents_cat_array,
                        shapebins_widths_down_array,
                        shapebins_widths_up_array,
                        shapebins_error_down_cat_array,
                        shapebins_error_up_cat_array,
                    )
                )
            tgsys_cat.append(tgsys_cat_z)

        # combine uncertainties from categories into groups

        tgsys_gr = []  # list of graphs with relative uncertainties for all groups, pt_jet bins
        for iptjet in range(self.n_bins_ptjet_gen):
            tgsys_gr_z = []  # list of graphs with relative uncertainties for all groups in a given pt_jet bin
            for gr in self.systematic_catgroups_list:
                # lists of graphs with relative uncertainties for categories in a given group in a given pt_jet bin
                tgsys_gr_z_cat = []
                for sys_cat, cat in enumerate(self.systematic_catlabels):
                    if self.systematic_catgroups[sys_cat] == gr:
                        print(f"Group {gr}: Adding category {cat}")
                        tgsys_gr_z_cat.append(tgsys_cat[iptjet][sys_cat])
                tgsys_gr_z.append(combine_graphs(tgsys_gr_z_cat))
            tgsys_gr.append(tgsys_gr_z)

        # write the combined systematic uncertainties in a file
        for iptjet in range(self.n_bins_ptjet_gen):
            suffix = self.get_suffix_ptjet(iptjet)
            self.file_sys_out.WriteObject(tgsys[iptjet], f"sys_{self.var}_{suffix}")
            unc_hist_up = TH1F(
                "unc_hist_up_%s" % suffix,
                "",
                n_bins_obs_gen,
                obs_gen_min,
                obs_gen_max,
            )
            unc_hist_down = TH1F(
                "unc_hist_down_%s" % suffix,
                "",
                n_bins_obs_gen,
                obs_gen_min,
                obs_gen_max,
            )
            for ibinshape in range(n_bins_obs_gen):
                unc_hist_up.SetBinContent(ibinshape + 1, full_unc_up[iptjet][ibinshape])
                unc_hist_down.SetBinContent(ibinshape + 1, full_unc_down[iptjet][ibinshape])
            self.file_sys_out.WriteObject(unc_hist_up, f"sys_{self.var}_{suffix}_rel_up")
            self.file_sys_out.WriteObject(unc_hist_down, f"sys_{self.var}_{suffix}_rel_down")

        # relative statistical uncertainty of the central values
        h_default_stat_err = []
        for iptjet in range(self.n_bins_ptjet_gen):
            suffix = self.get_suffix_ptjet(iptjet)
            h_default_stat_err.append(input_histograms_default[iptjet].Clone("h_default_stat_err" + suffix))
            for i in range(h_default_stat_err[iptjet].GetNbinsX()):
                if abs(input_histograms_default[iptjet].GetBinContent(i + 1)) < 1.0e-7:
                    print("WARNING!!! Input histogram at bin", iptjet, " equal 0", suffix)
                    h_default_stat_err[iptjet].SetBinContent(i + 1, 0)
                    h_default_stat_err[iptjet].SetBinError(i + 1, 0)
                else:
                    h_default_stat_err[iptjet].SetBinContent(i + 1, 0)
                    h_default_stat_err[iptjet].SetBinError(
                        i + 1,
                        input_histograms_default[iptjet].GetBinError(i + 1)
                        / input_histograms_default[iptjet].GetBinContent(i + 1),
                    )

        for iptjet in range(self.n_bins_ptjet_gen):
            # plot the results with systematic uncertainties
            suffix = self.get_suffix_ptjet(iptjet)
            cfinalwsys = TCanvas("cfinalwsys " + suffix, "final result with systematic uncertainties" + suffix)
            setup_canvas(cfinalwsys)
            leg_finalwsys = TLegend(0.7, 0.78, 0.85, 0.88)
            setup_legend(leg_finalwsys)
            leg_finalwsys.AddEntry(input_histograms_default[iptjet], "data", "P")
            self.crop_histogram(input_histograms_default[iptjet])
            self.crop_graph(tgsys[iptjet])
            setup_histogram(input_histograms_default[iptjet], get_colour(0, 0))
            y_min_g, y_max_g = get_y_window_gr([tgsys[iptjet]])
            y_min_h, y_max_h = get_y_window_his([input_histograms_default[iptjet]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.4
            y_margin_down = 0.05
            input_histograms_default[iptjet].GetYaxis().SetRangeUser(
                *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
            )
            # input_histograms_default[iptjet].GetXaxis().SetRangeUser(
            #     round(obs_gen_min, 2), round(obs_gen_max, 2)
            # )
            input_histograms_default[iptjet].GetXaxis().SetRangeUser(
                round(x_range[self.var][0], 2), round(x_range[self.var][1], 2)
            )
            input_histograms_default[iptjet].SetTitle("")
            input_histograms_default[iptjet].SetXTitle(latex_obs)
            input_histograms_default[iptjet].SetYTitle(latex_y)
            input_histograms_default[iptjet].Draw("AXIS")
            # input_histograms_default[iptjet].Draw("")
            setup_tgraph(tgsys[iptjet], get_colour(7, 0))
            tgsys[iptjet].Draw("5")
            input_histograms_default[iptjet].Draw("SAME")
            leg_finalwsys.AddEntry(tgsys[iptjet], "syst. unc.", "F")
            input_histograms_default[iptjet].Draw("AXISSAME")
            latex = TLatex(0.15, 0.82, self.text_alice)
            # latex = TLatex(0.15, 0.82, "pp, #sqrt{#it{s}} = 13.6 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.15, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.latex_hadron)
            draw_latex(latex1)
            latex2 = TLatex(
                0.15,
                0.72,
                "%g #leq %s < %g GeV/#it{c}, #left|#it{#eta}_{jet}#right| #leq 0.5"
                % (self.edges_ptjet_rec[iptjet], self.latex_ptjet, self.edges_ptjet_rec[iptjet + 1]),
            )
            draw_latex(latex2)
            latex3 = TLatex(
                0.15,
                0.67,
                "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, #left|#it{y}_{%s}#right| #leq 0.8"
                % (
                    self.edges_pthf_min[0],
                    self.latex_hadron,
                    min(self.edges_pthf_max[-1], self.edges_ptjet_rec[iptjet + 1]),
                    self.latex_hadron,
                ),
            )
            draw_latex(latex3)
            leg_finalwsys.Draw("same")
            if self.var != "zpar":
                latex_SD = TLatex(0.15, 0.62, self.text_sd)
                draw_latex(latex_SD)
            self.save_canvas(cfinalwsys, f"final_wsys_{self.var}_{suffix}")

            # plot the relative systematic uncertainties for all categories together

            # preliminary figure
            i_shape = 0 if self.var == "zg" else 1 if self.var == "rg" else 2

            crelativesys = TCanvas("crelativesys " + suffix, "relative systematic uncertainties" + suffix)
            gStyle.SetErrorX(0)
            setup_canvas(crelativesys)
            crelativesys.SetCanvasSize(900, 800)
            crelativesys.SetBottomMargin(self.margins_can[0])
            crelativesys.SetLeftMargin(self.margins_can[1])
            crelativesys.SetTopMargin(self.margins_can[2])
            crelativesys.SetRightMargin(self.margins_can[3])
            crelativesys.SetRightMargin(0.25)
            leg_relativesys = TLegend(0.77, 0.2, 0.95, 0.85)
            setup_legend(leg_relativesys, textsize=self.fontsize)
            for g in tgsys_cat[iptjet]:
                self.crop_graph(g)
            self.crop_histogram(h_default_stat_err[iptjet])
            y_min_g, y_max_g = get_y_window_gr(tgsys_cat[iptjet])
            y_min_h, y_max_h = get_y_window_his([h_default_stat_err[iptjet]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            list_y_margin_up = [0.2, 0.35, 0.2]
            y_margin_up = list_y_margin_up[i_shape]
            y_margin_down = 0.05
            setup_histogram(h_default_stat_err[iptjet])
            h_default_stat_err[iptjet].SetMarkerStyle(0)
            h_default_stat_err[iptjet].SetMarkerSize(0)
            leg_relativesys.AddEntry(h_default_stat_err[iptjet], "stat. unc.", "E")
            for sys_cat in range(self.n_sys_cat):
                setup_tgraph(tgsys_cat[iptjet][sys_cat], get_colour(sys_cat + 1, 0))
                tgsys_cat[iptjet][sys_cat].SetTitle("")
                tgsys_cat[iptjet][sys_cat].SetLineWidth(3)
                tgsys_cat[iptjet][sys_cat].SetFillStyle(0)
                tgsys_cat[iptjet][sys_cat].GetYaxis().SetRangeUser(
                    *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                )
                # tgsys_cat[iptjet][sys_cat].GetXaxis().SetLimits(
                #     round(edges_obs_gen[0 if self.var in ("nsd", "zpar") else 1], 2),
                #     round(obs_gen_max, 2),
                # )
                if self.var == "nsd":
                    tgsys_cat[iptjet][sys_cat].GetXaxis().SetNdivisions(5)
                    shrink_err_x(tgsys_cat[iptjet][sys_cat], 0.2)
                tgsys_cat[iptjet][sys_cat].GetXaxis().SetTitle(latex_obs)
                tgsys_cat[iptjet][sys_cat].GetYaxis().SetTitle("relative systematic uncertainty")
                tgsys_cat[iptjet][sys_cat].GetXaxis().SetTitleOffset(self.offsets_axes[0])
                tgsys_cat[iptjet][sys_cat].GetYaxis().SetTitleOffset(self.offsets_axes[1])
                leg_relativesys.AddEntry(tgsys_cat[iptjet][sys_cat], self.systematic_catlabels[sys_cat], "F")
                if sys_cat == 0:
                    tgsys_cat[iptjet][sys_cat].Draw("A2")
                else:
                    tgsys_cat[iptjet][sys_cat].Draw("2")
                unc_rel_min = 100.0
                unc_rel_max = 0.0
                for ibinshape in range(n_bins_obs_gen):
                    print(
                        "rel. syst. unc. ",
                        self.systematic_catlabels[sys_cat],
                        " ",
                        self.edges_ptjet_gen_min[iptjet],
                        " ",
                        self.edges_ptjet_gen_max[iptjet],
                        " ",
                        tgsys_cat[iptjet][sys_cat].GetErrorYhigh(ibinshape),
                        " ",
                        tgsys_cat[iptjet][sys_cat].GetErrorYlow(ibinshape),
                    )
                    unc_rel_min = min(
                        unc_rel_min,
                        tgsys_cat[iptjet][sys_cat].GetErrorYhigh(ibinshape),
                        tgsys_cat[iptjet][sys_cat].GetErrorYlow(ibinshape),
                    )
                    unc_rel_max = max(
                        unc_rel_max,
                        tgsys_cat[iptjet][sys_cat].GetErrorYhigh(ibinshape),
                        tgsys_cat[iptjet][sys_cat].GetErrorYlow(ibinshape),
                    )
                print(
                    f"rel. syst. unc. {self.systematic_catlabels[sys_cat]} (%): min. {(100. * unc_rel_min):.2g}, "
                    f"max. {(100. * unc_rel_max):.2g}"
                )
            h_default_stat_err[iptjet].Draw("same")
            h_default_stat_err[iptjet].Draw("axissame")
            # Draw LaTeX
            y_latex = self.y_latex_top
            list_latex = []
            text_ptjet_full = self.text_ptjet % (
                self.edges_ptjet_gen[iptjet],
                self.latex_ptjet,
                self.edges_ptjet_gen[iptjet + 1],
            )
            text_pth_full = self.text_pth % (
                self.edges_pthf_min[0],
                self.latex_hadron,
                min(self.edges_pthf_max[-1], self.edges_ptjet_gen[iptjet + 1]),
                self.latex_hadron,
            )
            for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
                latex = TLatex(self.x_latex, y_latex, text_latex)
                list_latex.append(latex)
                draw_latex(latex, textsize=self.fontsize)
                y_latex -= self.y_step
            leg_relativesys.Draw("same")
            self.save_canvas(crelativesys, f"sys_unc_{self.var}_{suffix}")
            gStyle.SetErrorX(0.5)

            # plot the relative systematic uncertainties for all categories together
            # same as above but categories combined into groups

            crelativesys_gr = TCanvas("crelativesys_gr " + suffix, "relative systematic uncertainties" + suffix)
            gStyle.SetErrorX(0)
            setup_canvas(crelativesys_gr)
            crelativesys_gr.SetCanvasSize(1000, 800)  # original width 900
            crelativesys_gr.SetBottomMargin(self.margins_can[0])
            crelativesys_gr.SetLeftMargin(9 / 10 * self.margins_can[1])  # scale for width 900 -> 1000
            crelativesys_gr.SetTopMargin(self.margins_can[2])
            # crelativesys_gr.SetRightMargin(0.25)
            crelativesys_gr.SetRightMargin(1 - 9 / 10 * (1 - 0.25))  # scale for width 900 -> 1000
            # leg_relativesys_gr = TLegend(.77, .2, 0.95, .85)
            leg_relativesys_gr = TLegend(0.77 * 9 / 10, 0.5, 0.95, 0.85)  # scale for width 900 -> 1000
            setup_legend(leg_relativesys_gr, textsize=self.fontsize)
            for g in tgsys_gr[iptjet]:
                self.crop_graph(g)
            self.crop_histogram(h_default_stat_err[iptjet])
            y_min_g, y_max_g = get_y_window_gr(tgsys_gr[iptjet])
            y_min_h, y_max_h = get_y_window_his([h_default_stat_err[iptjet]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            list_y_margin_up = [0.2, 0.35, 0.2]
            y_margin_up = list_y_margin_up[i_shape]
            y_margin_down = 0.05
            setup_histogram(h_default_stat_err[iptjet])
            h_default_stat_err[iptjet].SetMarkerStyle(0)
            h_default_stat_err[iptjet].SetMarkerSize(0)
            leg_relativesys_gr.AddEntry(h_default_stat_err[iptjet], "stat. unc.", "E")
            for sys_gr, gr in enumerate(self.systematic_catgroups_list):
                setup_tgraph(tgsys_gr[iptjet][sys_gr], get_colour(sys_gr + 1, 0))
                tgsys_gr[iptjet][sys_gr].SetTitle("")
                tgsys_gr[iptjet][sys_gr].SetLineWidth(3)
                tgsys_gr[iptjet][sys_gr].SetFillStyle(0)
                tgsys_gr[iptjet][sys_gr].GetYaxis().SetRangeUser(
                    *get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
                )
                # tgsys_gr[iptjet][sys_gr].GetXaxis().SetLimits(
                #     round(edges_obs_gen[0 if self.var in ("nsd", "zpar") else 1], 2),
                #     round(obs_gen_max, 2),
                # )
                if self.var == "nsd":
                    tgsys_gr[iptjet][sys_gr].GetXaxis().SetNdivisions(5)
                    shrink_err_x(tgsys_gr[iptjet][sys_gr], 0.2)
                tgsys_gr[iptjet][sys_gr].GetXaxis().SetTitle(latex_obs)
                tgsys_gr[iptjet][sys_gr].GetYaxis().SetTitle("relative systematic uncertainty")
                tgsys_gr[iptjet][sys_gr].GetXaxis().SetTitleOffset(self.offsets_axes[0])
                tgsys_gr[iptjet][sys_gr].GetYaxis().SetTitleOffset(self.offsets_axes[1])
                leg_relativesys_gr.AddEntry(tgsys_gr[iptjet][sys_gr], gr, "F")
                if sys_gr == 0:
                    tgsys_gr[iptjet][sys_gr].Draw("A2")
                else:
                    tgsys_gr[iptjet][sys_gr].Draw("2")
                unc_rel_min = 100.0
                unc_rel_max = 0.0
                for ibinshape in range(n_bins_obs_gen):
                    print(
                        "rel. syst. unc. ",
                        gr,
                        " ",
                        self.edges_ptjet_gen_min[iptjet],
                        " ",
                        self.edges_ptjet_gen_max[iptjet],
                        " ",
                        tgsys_gr[iptjet][sys_gr].GetErrorYhigh(ibinshape),
                        " ",
                        tgsys_gr[iptjet][sys_gr].GetErrorYlow(ibinshape),
                    )
                    unc_rel_min = min(
                        unc_rel_min,
                        tgsys_gr[iptjet][sys_gr].GetErrorYhigh(ibinshape),
                        tgsys_gr[iptjet][sys_gr].GetErrorYlow(ibinshape),
                    )
                    unc_rel_max = max(
                        unc_rel_max,
                        tgsys_gr[iptjet][sys_gr].GetErrorYhigh(ibinshape),
                        tgsys_gr[iptjet][sys_gr].GetErrorYlow(ibinshape),
                    )
                print(f"rel. syst. unc. {gr} (%): min. {(100. * unc_rel_min):.2g}, max. {(100. * unc_rel_max):.2g}")
            h_default_stat_err[iptjet].Draw("same")
            h_default_stat_err[iptjet].Draw("axissame")
            # Draw LaTeX
            y_latex = self.y_latex_top
            list_latex = []
            for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
                latex = TLatex(self.x_latex, y_latex, text_latex)
                list_latex.append(latex)
                draw_latex(latex, textsize=self.fontsize)
                y_latex -= self.y_step
            leg_relativesys_gr.Draw("same")
            self.save_canvas(crelativesys_gr, f"sys_unc_gr_{self.var}_{suffix}")
            gStyle.SetErrorX(0.5)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-analysis", "-d", dest="database_analysis", help="analysis database to be used", required=True
    )
    parser.add_argument("--analysis", "-a", dest="type_ana", help="choose type of analysis", required=True)
    args = parser.parse_args(args)

    gROOT.SetBatch(True)
    list_vars = ["zg", "nsd", "rg", "zpar"]
    # list_vars = ["zpar"]
    analyser = AnalyzerJetSystematics(args.database_analysis, args.type_ana)
    analyser.process(list_vars)


if __name__ == "__main__":
    main()
