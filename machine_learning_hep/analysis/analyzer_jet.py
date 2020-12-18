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
main script for doing final stage analysis
"""
# pylint: disable=too-many-lines, line-too-long
import os
from math import sqrt
from array import array
import numpy as np
import yaml
# pylint: disable=import-error, no-name-in-module
import uproot
from ROOT import TFile, TH1F, TH2F, TCanvas, TLatex, TGraphAsymmErrors, TLine, TGaxis
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils
from ROOT import TLegend
from ROOT import gROOT, gStyle
from ROOT import RooUnfoldBayes
# HF specific imports
from machine_learning_hep.utilities import folding, equal_binning_lists, make_message_notfound
from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.utilities import setup_histogram, setup_canvas, get_colour, get_marker, get_y_window_gr, get_y_window_his, get_plot_range
from machine_learning_hep.utilities import setup_legend, setup_tgraph, draw_latex, tg_sys, make_plot
from machine_learning_hep.do_variations import healthy_structure, format_varname, format_varlabel
from machine_learning_hep.utilities_plot import buildhisto, makefill2dhist, makefill3dhist
from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.utilities import seldf_singlevar

def shrink_err_x(graph, width=0.1):
    for i in range(graph.GetN()):
        graph.SetPointEXlow(i, width)
        graph.SetPointEXhigh(i, width)

# pylint: disable=too-many-instance-attributes, too-many-statements
class AnalyzerJet(Analyzer):
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

#        # normalisation
        self.p_nevents = 1 # number of selected events, taken from histonorm
        self.branching_ratio = 1
        #\ datap["analysis"][self.typean].get("branching_ratio", None)
        self.xsection_inel = \
            datap["analysis"][self.typean].get("xsection_inel", None)
        self.p_latexbin2var = datap["analysis"][self.typean]["latexbin2var"]
        self.v_varshape_latex = datap["analysis"][self.typean]["var_shape_latex"]

        # second variable (jet pt)
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"] # name
        self.lvar2_binmin_reco = datap["analysis"][self.typean].get("sel_binmin2_reco", None)
        self.lvar2_binmax_reco = datap["analysis"][self.typean].get("sel_binmax2_reco", None)
        self.p_nbin2_reco = len(self.lvar2_binmin_reco) # number of reco bins
        self.lvar2_binmin_gen = datap["analysis"][self.typean].get("sel_binmin2_gen", None)
        self.lvar2_binmax_gen = datap["analysis"][self.typean].get("sel_binmax2_gen", None)
        self.p_nbin2_gen = len(self.lvar2_binmin_gen) # number of gen bins
        self.var2ranges_reco = self.lvar2_binmin_reco.copy()
        self.var2ranges_reco.append(self.lvar2_binmax_reco[-1])
        self.var2binarray_reco = array("d", self.var2ranges_reco) # array of bin edges to use in histogram constructors
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        self.var2binarray_gen = array("d", self.var2ranges_gen) # array of bin edges to use in histogram constructors

        # observable (z, shape,...)
        self.v_varshape_binning = datap["analysis"][self.typean]["var_binningshape"] # name (reco)
        self.v_varshape_binning_gen = datap["analysis"][self.typean]["var_binningshape_gen"] # name (gen)
        self.lvarshape_binmin_reco = \
            datap["analysis"][self.typean].get("sel_binminshape_reco", None)
        self.lvarshape_binmax_reco = \
            datap["analysis"][self.typean].get("sel_binmaxshape_reco", None)
        self.p_nbinshape_reco = len(self.lvarshape_binmin_reco) # number of reco bins
        self.lvarshape_binmin_gen = \
            datap["analysis"][self.typean].get("sel_binminshape_gen", None)
        self.lvarshape_binmax_gen = \
            datap["analysis"][self.typean].get("sel_binmaxshape_gen", None)
        self.p_nbinshape_gen = len(self.lvarshape_binmin_gen) # number of gen bins
        self.varshaperanges_reco = self.lvarshape_binmin_reco.copy()
        self.varshaperanges_reco.append(self.lvarshape_binmax_reco[-1])
        self.varshapebinarray_reco = array("d", self.varshaperanges_reco) # array of bin edges to use in histogram constructors
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])
        self.varshapebinarray_gen = array("d", self.varshaperanges_gen) # array of bin edges to use in histogram constructors
        self.p_latexnhadron = ""
#        # systematics variations
#
#        # models to compare with
#        # POWHEG + PYTHIA 6
#        self.powheg_path_prompt = \
#            datap["analysis"][self.typean].get("powheg_path_prompt", None)
#        self.powheg_prompt_variations = \
#            datap["analysis"][self.typean].get("powheg_prompt_variations", None)
#        self.powheg_prompt_variations_path = \
#            datap["analysis"][self.typean].get("powheg_prompt_variations_path", None)
        # PYTHIA 8
        #self.pythia8_prompt_variations_path = \
            #datap["analysis"][self.typean].get("pythia8_prompt_variations_path", None)
        #self.pythia8_prompt_variations = \
            #datap["analysis"][self.typean].get("pythia8_prompt_variations", None)
        #self.pythia8_prompt_variations_legend = \
            #datap["analysis"][self.typean].get("pythia8_prompt_variations_legend", None)

        # unfolding
        self.niter_unfolding = \
            datap["analysis"][self.typean].get("niterunfolding", None)
        self.choice_iter_unfolding = \
            datap["analysis"][self.typean].get("niterunfoldingchosen", None)

#        # systematics
#        # import parameters of variations from the variation database
        path_sys_db = datap["analysis"][self.typean].get("variations_db", None)
        if not path_sys_db:
            self.logger.fatal(make_message_notfound("the variation database"))
        with open(path_sys_db, "r") as file_sys:
            db_sys = yaml.safe_load(file_sys)
        if not healthy_structure(db_sys):
            self.logger.fatal("Bad structure of the variation database.")
        db_sys = db_sys["categories"]
        self.systematic_catnames = [catname for catname, val in db_sys.items() if val["activate"]]
        self.n_sys_cat = len(self.systematic_catnames)
        self.systematic_catlabels = [""] * self.n_sys_cat
        self.systematic_varnames = [None] * self.n_sys_cat
        self.systematic_varlabels = [None] * self.n_sys_cat
        self.systematic_variations = [0] * self.n_sys_cat
        self.systematic_correlation = [None] * self.n_sys_cat
        self.systematic_rms = [False] * self.n_sys_cat
        self.systematic_symmetrise = [False] * self.n_sys_cat
        self.systematic_rms_both_sides = [False] * self.n_sys_cat
        self.powheg_nonprompt_varnames = []
        self.powheg_nonprompt_varlabels = []
        for c, catname in enumerate(self.systematic_catnames):
            self.systematic_catlabels[c] = db_sys[catname]["label"]
            self.systematic_varnames[c] = []
            self.systematic_varlabels[c] = []
            for varname, val in db_sys[catname]["variations"].items():
                n_var = len(val["activate"])
                for a, act in enumerate(val["activate"]):
                    if act:
                        varname_i = format_varname(varname, a, n_var)
                        varlabel_i = format_varlabel(val["label"], a, n_var)
                        self.systematic_varnames[c].append(varname_i)
                        self.systematic_varlabels[c].append(varlabel_i)
#                        if catname == "powheg":
#                            self.powheg_nonprompt_varnames.append(varname_i)
#                            self.powheg_nonprompt_varlabels.append(varlabel_i)
            self.systematic_variations[c] = len(self.systematic_varnames[c])
            self.systematic_correlation[c] = db_sys[catname]["correlation"]
            self.systematic_rms[c] = db_sys[catname]["rms"]
            self.systematic_symmetrise[c] = db_sys[catname]["symmetrise"]
            self.systematic_rms_both_sides[c] = db_sys[catname]["rms_both_sides"]

        # output directories
        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

        # input directories (processor output)
        self.d_resultsallpmc_proc = self.d_resultsallpmc
        self.d_resultsallpdata_proc = self.d_resultsallpdata
        # use a different processor output
        if "data_proc" in datap["analysis"][typean]:
            self.d_resultsallpdata_proc = datap["analysis"][typean]["data_proc"]["results"][period] \
                    if period is not None else datap["analysis"][typean]["data_proc"]["resultsallp"]
        if "mc_proc" in datap["analysis"][typean]:
            self.d_resultsallpmc_proc = datap["analysis"][typean]["mc_proc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc_proc"]["resultsallp"]

        # input files
        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata_proc, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc_proc, n_filemass_name)
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.n_fileresp = os.path.join(self.d_resultsallpmc_proc, self.n_fileresp)

        self.file_unfold = os.path.join(self.d_resultsallpdata, "unfolding_results.root")
        self.file_unfold_closure = os.path.join(self.d_resultsallpdata, "unfolding_closure.root")

        # official figures
        self.shape = typean[len("jet_"):]
        self.size_can = [800, 800]
        self.offsets_axes = [0.8, 1.1]
        self.margins_can = [0.1, 0.13, 0.05, 0.03]
        self.fontsize = 0.035
        self.opt_leg_g = "FP" # for systematic uncertanties in the legend
        self.opt_plot_g = "2"
        self.x_latex = 0.16
        self.y_latex_top = 0.88
        self.y_step = 0.055
        # axes titles
        self.title_x = self.v_varshape_latex
        self.title_y = "(1/#it{N}_{jet}) d#it{N}/d%s" % self.v_varshape_latex
        self.title_full = ";%s;%s" % (self.title_x, self.title_y)
        self.title_full_ratio = ";%s;data/MC: ratio of %s" % (self.title_x, self.title_y)
        # text
        self.text_alice = "#bf{ALICE} Preliminary, pp, #sqrt{#it{s}} = 13 TeV"
        self.text_jets = "inclusive charged jets, anti-#it{k}_{T}, #it{R} = 0.4"
        self.text_ptjet = "%g #leq %s < %g GeV/#it{c}, #left|#it{#eta}_{jet}#right| #leq 0.5"
        self.text_sd = "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"
        self.text_ptcut = "#it{p}_{T, incl. ch. jet}^{leading track} #geq 5.33 GeV/#it{c}"

    def unfolding(self):
        self.loadstyle()
        print("unfolding starts")

        fileouts = TFile.Open(self.file_unfold, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(self.file_unfold))

        # get the feed-down output
        unfolding_input_data_file = TFile.Open(self.n_filemass)
        if not unfolding_input_data_file:
            self.logger.fatal(make_message_notfound(self.n_filemass))
        input_data = unfolding_input_data_file.Get("h_jetptvsshape")
        if not input_data:
            self.logger.fatal(make_message_notfound("h_jetptvsshape", self.n_filemass))

        unfolding_input_file = TFile.Open(self.n_fileresp)
        if not unfolding_input_file:
            self.logger.fatal(make_message_notfound(self.n_fileresp))
        response_matrix = unfolding_input_file.Get("response_matrix")
        if not response_matrix:
            self.logger.fatal(make_message_notfound("response_matrix", self.n_fileresp))
        # rec. level cuts only applied
        hzvsjetpt_reco_nocuts = unfolding_input_file.Get("hzvsjetpt_reco_nocuts")
        if not hzvsjetpt_reco_nocuts:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco_nocuts", self.n_fileresp))
        # rec. level and gen. level cuts applied
        hzvsjetpt_reco_eff = unfolding_input_file.Get("hzvsjetpt_reco_cuts")
        if not hzvsjetpt_reco_eff:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco_cuts", self.n_fileresp))
        # closure test input
        input_mc_det = unfolding_input_file.Get("input_closure_reco")
        if not input_mc_det:
            self.logger.fatal(make_message_notfound("input_closure_reco", self.n_fileresp))

        stat_unfolding = input_data.Integral()
        stat_closure = input_mc_det.Integral()
        print("Unfolding: data statistics: %g, closure statistics: %g, ratio: %g" % (stat_unfolding, stat_closure, stat_unfolding/stat_closure))

        # Ignore the first bin for integration incase of untagged bin
        bin_int_first = 2 if self.lvarshape_binmin_reco[0] < 0 and "nsd" not in self.typean else 1

        # calculate rec. level kinematic efficiency and apply it to the unfolding input

        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)
        input_data.Multiply(hzvsjetpt_reco_eff)


        # gen. level cuts only applied
        hzvsjetpt_gen_nocuts = unfolding_input_file.Get("hzvsjetpt_gen_nocuts")
        if not hzvsjetpt_gen_nocuts:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen_nocuts", self.n_fileresp))
        # rec. level and gen. level cuts applied
        hzvsjetpt_gen_eff = unfolding_input_file.Get("hzvsjetpt_gen_cuts")
        if not hzvsjetpt_gen_eff:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen_cuts", self.n_fileresp))

        # calculate gen. level kinematic efficiency

        hzvsjetpt_gen_eff.Divide(hzvsjetpt_gen_nocuts)
#
#        # all gen. level jets
        input_mc_gen = unfolding_input_file.Get("hzvsjetpt_gen_unmatched")
        if not input_mc_gen:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen_unmatched", self.n_fileresp))
#        # rec. level cuts only applied
        mc_reco_matched = unfolding_input_file.Get("hzvsjetpt_reco")
        if not mc_reco_matched:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco", self.n_fileresp))
        # gen. level cuts only applied
        mc_gen_matched = unfolding_input_file.Get("hzvsjetpt_gen")
        if not mc_gen_matched:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen", self.n_fileresp))

        input_data_z = []

        input_mc_gen_z = []
        mc_reco_matched_z = []
        mc_gen_matched_z = []
        mc_reco_gen_matched_z_ratio = []
        hjetpt_fracdiff_list = []
        hz_fracdiff_list = []
        kinematic_eff = []
        hz_gen_nocuts = []

        hz_genvsreco_list = []
        hjetpt_genvsreco_list = []

        # reference for the jet pt refolding test
        input_data_jetpt = input_data.ProjectionY("input_data_jetpt", 1, self.p_nbinshape_reco, "e")
#
#        # get the central prompt POWHEG histogram
#        input_powheg = self.get_simulated_yields(self.powheg_path_prompt, 2, True)
#        if not input_powheg:
#            self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_prompt))
#        input_powheg.SetName("fh2_prompt_%s" % self.v_varshape_binning)
#        input_powheg_xsection = input_powheg.Clone(input_powheg.GetName() + "_xsec")
#
#        # Ensure correct binning: x - shape, y - jet pt
#        if not equal_binning_lists(input_powheg, list_x=self.varshaperanges_gen):
#            self.logger.fatal("Error: Incorrect binning in x.")
#        if not equal_binning_lists(input_powheg, list_y=self.var2ranges_gen):
#            self.logger.fatal("Error: Incorrect binning in y.")
#        # Ensure correct binning: x - shape, y - jet pt
#        if not equal_binning_lists(input_powheg_xsection, list_x=self.varshaperanges_gen):
#            self.logger.fatal("Error: Incorrect binning in x.")
#        if not equal_binning_lists(input_powheg_xsection, list_y=self.var2ranges_gen):
#            self.logger.fatal("Error: Incorrect binning in y.")
#
#        # get the prompt POWHEG variations
#
#        #input_powheg_sys = []
#        #input_powheg_xsection_sys = []
#        #for i_powheg in range(len(self.powheg_prompt_variations)):
#            #path = "%s%s.root" % (self.powheg_prompt_variations_path, self.powheg_prompt_variations[i_powheg])
#            #input_powheg_sys_i = self.get_simulated_yields(path, 2, True)
#            #if not input_powheg_sys_i:
#            #    self.logger.fatal(make_message_notfound("simulated yields", path))
#            #input_powheg_sys_i.SetName("fh2_prompt_%s_%d" % (self.v_varshape_binning, i_powheg))
#            #input_powheg_sys.append(input_powheg_sys_i)
#            #input_powheg_xsection_sys_i = input_powheg_sys_i.Clone(input_powheg_sys_i.GetName() + "_xsec")
#            #input_powheg_xsection_sys.append(input_powheg_xsection_sys_i)
#
#        input_powheg_z = []
#        input_powheg_xsection_z = []
#        #input_powheg_sys_z = []
#        #input_powheg_xsection_sys_z = []
#        #tg_powheg = []
#        #tg_powheg_xsection = []
#
#        # get simulated distributions from PYTHIA 6 and POWHEG and calculate their spread
#
#        for ibin2 in range(self.p_nbin2_gen):
#            suffix = "%s_%.2f_%.2f" % \
#                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
#            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
#            input_mc_gen_z[ibin2].Scale(1.0 / input_mc_gen_z[ibin2].Integral(bin_int_first, input_mc_gen_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])), "width")
#            input_powheg_z.append(input_powheg.ProjectionX("input_powheg_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
#            input_powheg_z[ibin2].Scale(1.0 / input_powheg_z[ibin2].Integral(bin_int_first, input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])), "width")
#            input_powheg_xsection_z.append(input_powheg_xsection.ProjectionX("input_powheg_xsection_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
#            input_powheg_xsection_z[ibin2].Scale(1.0, "width")
#            #input_powheg_sys_z_iter = []
#            #input_powheg_xsection_sys_z_iter = []
#            #for i_powheg in range(len(self.powheg_prompt_variations)):
#            #    input_powheg_sys_z_iter.append(input_powheg_sys[i_powheg].ProjectionX("input_powheg_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
#            #    input_powheg_sys_z_iter[i_powheg].Scale(1.0 / input_powheg_sys_z_iter[i_powheg].Integral(bin_int_first, input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[-1])), "width")
#            #    input_powheg_xsection_sys_z_iter.append(input_powheg_xsection_sys[i_powheg].ProjectionX("input_powheg_xsection_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
#            #    input_powheg_xsection_sys_z_iter[i_powheg].Scale(1.0, "width")
#            #input_powheg_sys_z.append(input_powheg_sys_z_iter)
#            #input_powheg_xsection_sys_z.append(input_powheg_xsection_sys_z_iter)
#            #tg_powheg.append(tg_sys(input_powheg_z[ibin2], input_powheg_sys_z[ibin2]))
#            #tg_powheg_xsection.append(tg_sys(input_powheg_xsection_z[ibin2], input_powheg_xsection_sys_z[ibin2]))
#
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])

            input_data_z.append(input_data.ProjectionX("input_data_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))

            # compare shapes of distributions of reconstructed jets that pass rec. vs. gen. level cuts

            mc_reco_matched_z.append(mc_reco_matched.ProjectionX("mc_reco_matched_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            mc_reco_matched_z[ibin2].Scale(1.0 / mc_reco_matched_z[ibin2].Integral(bin_int_first, mc_reco_matched_z[ibin2].GetNbinsX()))
            mc_gen_matched_z.append(mc_gen_matched.ProjectionX("mc_det_matched_z" + suffix, mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]), mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]), "e"))
            mc_gen_matched_z[ibin2].Scale(1.0 / mc_gen_matched_z[ibin2].Integral(bin_int_first, mc_gen_matched_z[ibin2].GetNbinsX()))
            mc_reco_gen_matched_z_ratio.append(mc_reco_matched_z[ibin2].Clone("input_mc_reco_gen_matched_z_ratio" + suffix))
            mc_reco_gen_matched_z_ratio[ibin2].Divide(mc_gen_matched_z[ibin2])

            c_mc_reco_gen_matched_z_ratio = TCanvas("c_mc_reco_gen_matched_z_ratio " + suffix, "Reco/Gen Ratio")
            setup_canvas(c_mc_reco_gen_matched_z_ratio)
            c_mc_reco_gen_matched_z_ratio.SetLeftMargin(0.13)
            setup_histogram(mc_reco_gen_matched_z_ratio[ibin2])
            y_min_h, y_max_h = get_y_window_his(mc_reco_gen_matched_z_ratio[ibin2])
            y_margin_up = 0.15
            y_margin_down = 0.05
            mc_reco_gen_matched_z_ratio[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            mc_reco_gen_matched_z_ratio[ibin2].SetTitle("")
            mc_reco_gen_matched_z_ratio[ibin2].SetXTitle(self.v_varshape_latex)
            mc_reco_gen_matched_z_ratio[ibin2].SetYTitle("reconstructed/generated")
            mc_reco_gen_matched_z_ratio[ibin2].SetTitleOffset(1.3, "Y")
            mc_reco_gen_matched_z_ratio[ibin2].Draw("same")
            latex = TLatex(0.2, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_mc_reco_gen_matched_z_ratio.SaveAs("%s/reco_gen_matched_%s_ratio_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

            c_mc_reco_gen_matched_z = TCanvas("c_mc_reco_gen_matched_z " + suffix, "Reco vs Gen")
            setup_canvas(c_mc_reco_gen_matched_z)
            c_mc_reco_gen_matched_z.SetLeftMargin(0.13)
            leg_mc_reco_gen_matched_z = TLegend(.6, .8, .8, .85, "")
            setup_legend(leg_mc_reco_gen_matched_z)
            setup_histogram(mc_reco_matched_z[ibin2], get_colour(1), get_marker(0))
            leg_mc_reco_gen_matched_z.AddEntry(mc_reco_matched_z[ibin2], "reconstructed", "P")
            y_min_h, y_max_h = get_y_window_his([mc_reco_matched_z[ibin2], mc_gen_matched_z[ibin2]])
            y_margin_up = 0.15
            y_margin_down = 0.05
            mc_reco_matched_z[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            mc_reco_matched_z[ibin2].SetTitle("")
            mc_reco_matched_z[ibin2].SetXTitle(self.v_varshape_latex)
            mc_reco_matched_z[ibin2].SetYTitle("self-normalised yield")
            mc_reco_matched_z[ibin2].SetTitleOffset(1.3, "Y")
            mc_reco_matched_z[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            mc_reco_matched_z[ibin2].Draw()
            setup_histogram(mc_gen_matched_z[ibin2], get_colour(2), get_marker(1))
            leg_mc_reco_gen_matched_z.AddEntry(mc_gen_matched_z[ibin2], "generated", "P")
            mc_gen_matched_z[ibin2].Draw("same")
            leg_mc_reco_gen_matched_z.Draw("same")
            latex = TLatex(0.2, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_mc_reco_gen_matched_z.SaveAs("%s/reco_gen_matched_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

            # plot the shape response matrix in jet pt bins

            hz_genvsreco_list.append(unfolding_input_file.Get("hz_genvsreco" + suffix))
            cz_genvsreco = TCanvas("cz_genvsreco_" + suffix, "response matrix 2D projection")
            setup_canvas(cz_genvsreco)
            cz_genvsreco.SetRightMargin(0.13)
            cz_genvsreco.SetLogz()
            setup_histogram(hz_genvsreco_list[ibin2])
            hz_genvsreco_list[ibin2].GetZaxis().SetRangeUser(hz_genvsreco_list[ibin2].GetMinimum(0), hz_genvsreco_list[ibin2].GetMaximum())
            hz_genvsreco_list[ibin2].SetTitle("%g #leq %s < %g GeV/#it{c}" % \
                (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            hz_genvsreco_list[ibin2].SetXTitle("%s^{gen}" % self.v_varshape_latex)
            hz_genvsreco_list[ibin2].SetYTitle("%s^{rec}" % self.v_varshape_latex)
            hz_genvsreco_list[ibin2].Draw("colz")
            cz_genvsreco.SaveAs("%s/response_pr_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

        # plot the jet pt response matrix in shape bins

        for ibinshape in range(self.p_nbinshape_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco_list.append(unfolding_input_file.Get("hjetpt_genvsreco" + suffix))

            cjetpt_genvsreco = TCanvas("cjetpt_genvsreco" + suffix, "response matrix 2D projection" + suffix)
            setup_canvas(cjetpt_genvsreco)
            cjetpt_genvsreco.SetRightMargin(0.13)
            cjetpt_genvsreco.SetLogz()
            setup_histogram(hjetpt_genvsreco_list[ibinshape])
            hjetpt_genvsreco_list[ibinshape].GetZaxis().SetRangeUser(hjetpt_genvsreco_list[ibinshape].GetMinimum(0), hjetpt_genvsreco_list[ibinshape].GetMaximum())
            hjetpt_genvsreco_list[ibinshape].SetTitle("%g #leq %s < %g" % \
                (self.lvarshape_binmin_reco[ibinshape], self.v_varshape_latex, self.lvarshape_binmax_reco[ibinshape]))
            hjetpt_genvsreco_list[ibinshape].SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
            hjetpt_genvsreco_list[ibinshape].SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
            hjetpt_genvsreco_list[ibinshape].Draw("colz")
            cjetpt_genvsreco.SaveAs("%s/response_pr_%s_%s.eps" % (self.d_resultsallpdata, self.v_var2_binning, suffix_plot))

        # plot the full response matrix

        his_response_pr = response_matrix.Hresponse() # linearised response matrix as a TH2
        cresponse_pr = TCanvas("cresponse_pr", "prompt response matrix")
        setup_canvas(cresponse_pr)
        cresponse_pr.SetLogz()
        cresponse_pr.SetRightMargin(0.13)
        setup_histogram(his_response_pr)
        his_response_pr.SetTitle("")
        his_response_pr.SetXTitle("(#it{p}_{T, jet}^{rec}, %s^{rec}) bin" % self.v_varshape_latex)
        his_response_pr.SetYTitle("(#it{p}_{T, jet}^{gen}, %s^{gen}) bin" % self.v_varshape_latex)
        his_response_pr.Draw("colz")
        cresponse_pr.SaveAs("%s/response_pr_matrix.eps" % self.d_resultsallpdata)

        hz_genvsreco_full = unfolding_input_file.Get("hz_genvsreco_full")
        if not hz_genvsreco_full:
            self.logger.fatal(make_message_notfound("hz_genvsreco_full", self.n_fileresp))
        hjetpt_genvsreco_full = unfolding_input_file.Get("hjetpt_genvsreco_full")
        if not hjetpt_genvsreco_full:
            self.logger.fatal(make_message_notfound("hjetpt_genvsreco_full", self.n_fileresp))

        hz_genvsreco_full.Scale(1. / hz_genvsreco_full.Integral())

        # preliminary figure
        text_ptjet_full = self.text_ptjet % (self.lvar2_binmin_reco[0], self.p_latexbin2var, self.lvar2_binmax_reco[-1])
        cz_genvsreco_full = TCanvas("cz_genvsreco_full", "response matrix 2D projection", 800, 800)
        setup_canvas(cz_genvsreco_full)
        cz_genvsreco_full.SetCanvasSize(900, 800)
        cz_genvsreco_full.SetLeftMargin(0.12)
        cz_genvsreco_full.SetRightMargin(0.18)
        cz_genvsreco_full.SetBottomMargin(0.12)
        cz_genvsreco_full.SetTopMargin(0.3)
        cz_genvsreco_full.SetLogz()
        setup_histogram(hz_genvsreco_full)
        hz_genvsreco_full.GetZaxis().SetRangeUser(hz_genvsreco_full.GetMinimum(0), hz_genvsreco_full.GetMaximum())
        hz_genvsreco_full.GetZaxis().SetTitleOffset(1.5)
        if self.shape == "nsd":
            hz_genvsreco_full.GetXaxis().SetNdivisions(5)
            hz_genvsreco_full.GetYaxis().SetNdivisions(5)
        hz_genvsreco_full.SetTitle(";%s^{gen};%s^{rec};self-normalised yield" % (self.v_varshape_latex, self.v_varshape_latex))
        hz_genvsreco_full.Draw("colz")
        y_latex = 0.95
        list_latex = []
        for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, self.text_ptcut, self.text_sd]:
            latex = TLatex(self.x_latex, y_latex, text_latex)
            list_latex.append(latex)
            draw_latex(latex, textsize=self.fontsize, colour=1)
            y_latex -= self.y_step
        cz_genvsreco_full.Update()
        cz_genvsreco_full.SaveAs("%s/response_pr_%s_full.pdf" % (self.d_resultsallpdata, self.v_varshape_binning))
        cz_genvsreco_full.SaveAs("%s/%s_resp_pr_full_incl.pdf" % (self.d_resultsallpdata, self.shape))

        cjetpt_genvsreco_full = TCanvas("cjetpt_genvsreco_full", "response matrix 2D projection")
        setup_canvas(cjetpt_genvsreco_full)
        cjetpt_genvsreco_full.SetRightMargin(0.13)
        cjetpt_genvsreco_full.SetLogz()
        setup_histogram(hjetpt_genvsreco_full)
        hjetpt_genvsreco_full.GetZaxis().SetRangeUser(hjetpt_genvsreco_full.GetMinimum(0), hjetpt_genvsreco_full.GetMaximum())
        hjetpt_genvsreco_full.SetTitle("")
        hjetpt_genvsreco_full.SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hjetpt_genvsreco_full.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hjetpt_genvsreco_full.Draw("colz")
        cjetpt_genvsreco_full.SaveAs("%s/response_pr_%s_full.eps" % (self.d_resultsallpdata, self.v_var2_binning))

        hz_genvsreco_full_real = unfolding_input_file.Get("hz_genvsreco_full_real")
        if not hz_genvsreco_full_real:
            self.logger.fatal(make_message_notfound("hz_genvsreco_full_real", self.n_fileresp))
        hjetpt_genvsreco_full_real = unfolding_input_file.Get("hjetpt_genvsreco_full_real")
        if not hjetpt_genvsreco_full_real:
            self.logger.fatal(make_message_notfound("hjetpt_genvsreco_full_real", self.n_fileresp))

        hz_genvsreco_full_real.Scale(1. / hz_genvsreco_full_real.Integral())

        # preliminary figure
        cz_genvsreco_full_real = TCanvas("cz_genvsreco_full_real", "response matrix 2D projection_real")
        setup_canvas(cz_genvsreco_full_real)
        cz_genvsreco_full_real.SetCanvasSize(900, 800)
        cz_genvsreco_full_real.SetLeftMargin(0.12)
        cz_genvsreco_full_real.SetRightMargin(0.18)
        cz_genvsreco_full_real.SetBottomMargin(0.12)
        cz_genvsreco_full_real.SetTopMargin(0.3)
        cz_genvsreco_full_real.SetLogz()
        setup_histogram(hz_genvsreco_full_real)
        hz_genvsreco_full_real.GetZaxis().SetRangeUser(hz_genvsreco_full_real.GetMinimum(0), hz_genvsreco_full_real.GetMaximum())
        hz_genvsreco_full_real.GetZaxis().SetTitleOffset(1.5)
        if self.shape == "nsd":
            hz_genvsreco_full_real.GetXaxis().SetNdivisions(5)
            hz_genvsreco_full_real.GetYaxis().SetNdivisions(5)
        hz_genvsreco_full_real.SetTitle(";%s^{gen};%s^{rec};self-normalised yield" % (self.v_varshape_latex, self.v_varshape_latex))
        hz_genvsreco_full_real.Draw("colz")
        y_latex = 0.95
        list_latex = []
        for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, self.text_ptcut, self.text_sd]:
            latex = TLatex(self.x_latex, y_latex, text_latex)
            list_latex.append(latex)
            draw_latex(latex, textsize=self.fontsize, colour=1)
            y_latex -= self.y_step
        cz_genvsreco_full_real.Update()
        cz_genvsreco_full_real.SaveAs("%s/response_pr_%s_full_real.pdf" % (self.d_resultsallpdata, self.v_varshape_binning))
        cz_genvsreco_full_real.SaveAs("%s/%s_resp_pr_full_real_incl.pdf" % (self.d_resultsallpdata, self.shape))

        cjetpt_genvsreco_full_real = TCanvas("cjetpt_genvsreco_full_real", "response matrix 2D projection_real")
        setup_canvas(cjetpt_genvsreco_full_real)
        cjetpt_genvsreco_full_real.SetRightMargin(0.13)
        cjetpt_genvsreco_full_real.SetLogz()
        setup_histogram(hjetpt_genvsreco_full_real)
        hjetpt_genvsreco_full_real.GetZaxis().SetRangeUser(hjetpt_genvsreco_full_real.GetMinimum(0), hjetpt_genvsreco_full_real.GetMaximum())
        hjetpt_genvsreco_full_real.SetTitle("")
        hjetpt_genvsreco_full_real.SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hjetpt_genvsreco_full_real.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hjetpt_genvsreco_full_real.Draw("colz")
        cjetpt_genvsreco_full_real.SaveAs("%s/response_pr_%s_full_real.eps" % (self.d_resultsallpdata, self.v_var2_binning))

        # plot gen. level kinematic efficiency for shape in jet pt bins

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            kinematic_eff.append(unfolding_input_file.Get("hz_gen_cuts" + suffix))
            hz_gen_nocuts.append(unfolding_input_file.Get("hz_gen_nocuts" + suffix))
            kinematic_eff[ibin2].Divide(hz_gen_nocuts[ibin2])
            ckinematic_eff = TCanvas("ckinematic_eff " + suffix, "Kinematic Eff" + suffix)
            setup_canvas(ckinematic_eff)
            ckinematic_eff.SetLeftMargin(0.13)
            setup_histogram(kinematic_eff[ibin2], get_colour(1))
            y_min_h, y_max_h = get_y_window_his(kinematic_eff[ibin2])
            y_margin_up = 0.15
            y_margin_down = 0.05
            kinematic_eff[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            kinematic_eff[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            kinematic_eff[ibin2].SetTitle("")
            kinematic_eff[ibin2].SetXTitle(self.v_varshape_latex)
            kinematic_eff[ibin2].SetYTitle("kinematic efficiency")
            kinematic_eff[ibin2].SetTitleOffset(1.5, "Y")
            kinematic_eff[ibin2].Draw()
            latex = TLatex(0.2, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            ckinematic_eff.SaveAs("%s/kineff_pr_gen_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

        # plot gen. level kinematic efficiency for jet pt

        kinematic_eff_jetpt = unfolding_input_file.Get("hjetpt_gen_cuts")
        hjetpt_gen_nocuts = unfolding_input_file.Get("hjetpt_gen_nocuts")
        kinematic_eff_jetpt.Divide(hjetpt_gen_nocuts)
        ckinematic_eff_jetpt = TCanvas("ckinematic_eff_jetpt", "Kinematic Eff_jetpt")
        setup_canvas(ckinematic_eff_jetpt)
        ckinematic_eff_jetpt.SetLeftMargin(0.13)
        setup_histogram(kinematic_eff_jetpt)
        y_min_h, y_max_h = get_y_window_his(kinematic_eff_jetpt)
        y_margin_up = 0.15
        y_margin_down = 0.05
        kinematic_eff_jetpt.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
        kinematic_eff_jetpt.GetXaxis().SetRangeUser(round(self.lvar2_binmin_reco[0], 2), round(self.lvar2_binmax_reco[-1], 2))
        kinematic_eff_jetpt.SetTitle("")
        kinematic_eff_jetpt.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
        kinematic_eff_jetpt.SetYTitle("kinematic efficiency")
        kinematic_eff_jetpt.SetTitleOffset(1.5, "Y")
        kinematic_eff_jetpt.Draw()
        latex = TLatex(0.2, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_reco[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_reco[-1], 2)))
        draw_latex(latex)
        ckinematic_eff_jetpt.SaveAs("%s/kineff_pr_gen_%s.eps" % (self.d_resultsallpdata, self.v_var2_binning))

        # plot gen. level kinematic efficiency

        cgen_eff = TCanvas("cgen_eff ", "gen efficiency applied to unfolding output")
        setup_canvas(cgen_eff)
        setup_histogram(hzvsjetpt_gen_eff)
        hzvsjetpt_gen_eff.SetTitle("")
        hzvsjetpt_gen_eff.SetXTitle("%s^{gen}" % self.v_varshape_latex)
        hzvsjetpt_gen_eff.SetYTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hzvsjetpt_gen_eff.Draw("text")
        cgen_eff.SaveAs("%s/kineff_pr_gen.eps" % self.d_resultsallpdata)

       # plot rec. level kinematic efficiency

        creco_eff = TCanvas("creco_eff ", "reco efficiency applied to input data")
        setup_canvas(creco_eff)
        setup_histogram(hzvsjetpt_reco_eff)
        hzvsjetpt_reco_eff.SetTitle("")
        hzvsjetpt_reco_eff.SetXTitle("%s^{rec}" % self.v_varshape_latex)
        hzvsjetpt_reco_eff.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hzvsjetpt_reco_eff.Draw("text")
        creco_eff.SaveAs("%s/kineff_pr_rec.eps" % self.d_resultsallpdata)

        # plot relative shift of jet pt

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff_list.append(unfolding_input_file.Get("hjetpt_fracdiff_prompt" + suffix))

        cjetpt_fracdiff = TCanvas("cjetpt_fracdiff", "prompt jetpt response fractional differences")
        setup_canvas(cjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        leg_jetpt_fracdiff = TLegend(.15, .5, .25, .8, "#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            setup_histogram(hjetpt_fracdiff_list[ibin2], get_colour(ibin2), get_marker(ibin2))
            leg_jetpt_fracdiff.AddEntry(hjetpt_fracdiff_list[ibin2], "%g#minus%g" % (self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2]), "P")
            if ibin2 == 0:
                hjetpt_fracdiff_list[ibin2].SetTitle("")
                hjetpt_fracdiff_list[ibin2].SetXTitle("(#it{p}_{T, jet}^{rec} #minus #it{p}_{T, jet}^{gen})/#it{p}_{T, jet}^{gen}")
                _, y_max_h = get_y_window_his(hjetpt_fracdiff_list)
                y_margin_up = 0.15
                y_margin_down = 0.05
                y_min_0 = min([h.GetMinimum(0) for h in hjetpt_fracdiff_list])
                hjetpt_fracdiff_list[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_0, y_max_h, y_margin_down, y_margin_up, True))
            hjetpt_fracdiff_list[ibin2].Draw("same")
        leg_jetpt_fracdiff.Draw("same")
        cjetpt_fracdiff.SaveAs("%s/response_pr_reldiff_%s.eps" % (self.d_resultsallpdata, self.v_var2_binning))

        # plot relative shift of shape

        for ibinshape in range(self.p_nbinshape_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_list.append(unfolding_input_file.Get("hz_fracdiff_prompt" + suffix))

        cz_fracdiff = TCanvas("cz_fracdiff", "prompt z response fractional differences")
        setup_canvas(cz_fracdiff)
        cz_fracdiff.SetLogy()
        leg_z_fracdiff = TLegend(.15, .5, .25, .8, self.v_varshape_latex)
        setup_legend(leg_z_fracdiff)
        for ibinshape in range(self.p_nbinshape_gen):
            setup_histogram(hz_fracdiff_list[ibinshape], get_colour(ibinshape), get_marker(ibinshape))
            leg_z_fracdiff.AddEntry(hz_fracdiff_list[ibinshape], "%g#minus%g" % (self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape]), "P")
            if ibinshape == 0:
                hz_fracdiff_list[ibinshape].SetTitle("")
                hz_fracdiff_list[ibinshape].SetXTitle("(%s^{rec} #minus %s^{gen})/%s^{gen}" % (self.v_varshape_latex, self.v_varshape_latex, self.v_varshape_latex))
                _, y_max_h = get_y_window_his(hz_fracdiff_list)
                y_min_0 = min([h.GetMinimum(0) for h in hz_fracdiff_list])
                y_margin_up = 0.15
                y_margin_down = 0.05
                hz_fracdiff_list[ibinshape].GetYaxis().SetRangeUser(*get_plot_range(y_min_0, y_max_h, y_margin_down, y_margin_up, True))
            hz_fracdiff_list[ibinshape].Draw("same")
        leg_z_fracdiff.Draw("same")
        cz_fracdiff.SaveAs("%s/response_pr_reldiff_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning))

        fileouts.cd()
        unfolded_z_scaled_list = []
        unfolded_z_xsection_list = []
        unfolded_jetpt_scaled_list = []
        refolding_test_list = []
        refolding_test_jetpt_list = []
        i_iter_choice = self.choice_iter_unfolding - 1 # list index of the chosen iteration
        for i in range(self.niter_unfolding):
            unfolded_z_scaled_list_iter = []
            unfolded_z_xsection_list_iter = []
            refolding_test_list_iter = []

            # unfold

            unfolding_object = RooUnfoldBayes(response_matrix, input_data, i + 1)
            unfolded_zvsjetpt = unfolding_object.Hreco(2)

            # plot the final unfolded shape vs jet pt

            if i == i_iter_choice:
                unfolded_zvsjetpt_final = unfolded_zvsjetpt.Clone("unfolded_zvsjetpt_final")
                # apply 2D gen. level kin. eff.
                unfolded_zvsjetpt_final.Divide(hzvsjetpt_gen_eff)
                cunfolded_output = TCanvas("cunfolded_output", "unfolded_output")
                setup_canvas(cunfolded_output)
                setup_histogram(unfolded_zvsjetpt_final, get_colour(1))
                unfolded_zvsjetpt_final.SetTitle("iteration %d" % (i + 1))
                unfolded_zvsjetpt_final.SetXTitle(self.v_varshape_latex)
                unfolded_zvsjetpt_final.SetYTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
                gStyle.SetPaintTextFormat(".0f")
                unfolded_zvsjetpt_final.Draw("texte")
                cunfolded_output.SaveAs("%s/unfolded_output.eps" % self.d_resultsallpdata)
                gStyle.SetPaintTextFormat("g")

            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                suffix_plot = "%s_%g_%g" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                unfolded_z = unfolded_zvsjetpt.ProjectionX("unfolded_z_proj_%d_%s" % (i + 1, suffix), ibin2 + 1, ibin2 + 1, "e")
                unfolded_z_scaled = unfolded_z.Clone("unfolded_z_scaled_%d_%s" % (i + 1, suffix))

                # apply gen. level kinematic efficiency

                unfolded_z_scaled.Divide(kinematic_eff[ibin2])

                # normalise by luminosity and divide by the branching ratio to get cross-section

                unfolded_z_xsection = unfolded_z_scaled.Clone("unfolded_z_xsection_%d_%s" % (i + 1, suffix))
                unfolded_z_xsection.Scale(self.xsection_inel / (self.p_nevents * self.branching_ratio), "width")

                # normalise by the number of jets

                unfolded_z_scaled.Scale(1.0 / unfolded_z_scaled.Integral(bin_int_first, unfolded_z_scaled.FindBin(self.lvarshape_binmin_reco[-1])), "width")

                unfolded_z_scaled.Write("unfolded_z_%d_%s" % (i + 1, suffix))
                unfolded_z_xsection.Write("unfolded_z_xsection_%d_%s" % (i + 1, suffix))
                unfolded_z_scaled_list_iter.append(unfolded_z_scaled)
                unfolded_z_xsection_list_iter.append(unfolded_z_xsection)
                cunfolded_z = TCanvas("cunfolded_z_%d_%s" % (i + 1, suffix), "1D output of unfolding" + suffix)
                setup_canvas(cunfolded_z)
                setup_histogram(unfolded_z_scaled, get_colour(1))
                y_min_h, y_max_h = get_y_window_his(unfolded_z_scaled)
                y_margin_up = 0.15
                y_margin_down = 0.05
                unfolded_z_scaled.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
                unfolded_z_scaled.GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
                unfolded_z_scaled.SetTitle("")
                unfolded_z_scaled.SetXTitle(self.v_varshape_latex)
                unfolded_z_scaled.SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
                unfolded_z_scaled.Draw()
                latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.5, 0.82, "iteration %d" % (i + 1))
                draw_latex(latex2)
                cunfolded_z.SaveAs("%s/unfolded_%s_%d_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, i + 1, suffix_plot))
                # Save the selected iteration under a special name.
                if i == i_iter_choice:
                    unfolded_z_scaled.Write("unfolded_z_sel_%s" % suffix)
                    unfolded_z_xsection.Write("unfolded_z_xsection_sel_%s" % suffix)
                    cunfolded_z.SaveAs("%s/unfolded_%s_sel_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

            unfolded_z_scaled_list.append(unfolded_z_scaled_list_iter)
            unfolded_z_xsection_list.append(unfolded_z_xsection_list_iter)

            # get unfolded jet pt

            unfolded_jetpt = unfolded_zvsjetpt.ProjectionY("unfolded_jetpt_%d" % (i + 1), 1, self.p_nbinshape_gen, "e")
            unfolded_jetpt_scaled = unfolded_jetpt.Clone("unfolded_jetpt_scaled_%d" % (i + 1))

            # apply gen. level kinematic efficiency

            unfolded_jetpt_scaled.Divide(kinematic_eff_jetpt)

            # normalise by number of jets

            unfolded_jetpt_scaled.Scale(1.0 / unfolded_jetpt_scaled.Integral(unfolded_jetpt_scaled.FindBin(self.lvar2_binmin_reco[0]), unfolded_jetpt_scaled.FindBin(self.lvar2_binmin_reco[-1])), "width")

            unfolded_jetpt_scaled.Write("unfolded_jetpt_%d" % (i + 1))
            unfolded_jetpt_scaled_list.append(unfolded_jetpt_scaled)
            cunfolded_jetpt = TCanvas("cunfolded_jetpt_%s" % (i + 1), "1D output of unfolding")
            setup_canvas(cunfolded_jetpt)
            cunfolded_jetpt.SetLogy()
            cunfolded_jetpt.SetLeftMargin(0.13)
            setup_histogram(unfolded_jetpt_scaled, get_colour(1))
            y_min_h, y_max_h = get_y_window_his(unfolded_jetpt_scaled)
            y_min_0 = unfolded_jetpt_scaled.GetMinimum(0)
            if y_min_h <= 0:
                y_min_h = y_min_0
            y_margin_up = 0.15
            y_margin_down = 0.05
            unfolded_jetpt_scaled.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up, True))
            unfolded_jetpt_scaled.GetXaxis().SetRangeUser(self.lvar2_binmin_reco[0], self.lvar2_binmax_reco[-1])
            unfolded_jetpt_scaled.SetTitle("")
            unfolded_jetpt_scaled.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
            unfolded_jetpt_scaled.SetYTitle("1/#it{N}_{jets} d#it{N}/d%s (#it{c}/GeV)" % self.p_latexbin2var)
            unfolded_jetpt_scaled.SetTitleOffset(1.5, "Y")
            unfolded_jetpt_scaled.Draw()
            latex = TLatex(0.2, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_reco[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_reco[-1], 2)))
            draw_latex(latex)
            latex2 = TLatex(0.55, 0.82, "iteration %d" % (i + 1))
            draw_latex(latex2)
            cunfolded_jetpt.SaveAs("%s/unfolded_%s_%d.eps" % (self.d_resultsallpdata, self.v_var2_binning, i + 1))

            # refolding test for shape in jet pt bins

            refolded = folding(unfolded_zvsjetpt, response_matrix, input_data)
            for ibin2 in range(self.p_nbin2_reco):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                suffix_plot = "%s_%g_%g" % \
                         (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                refolded_z = refolded.ProjectionX("refolded_z", ibin2 + 1, ibin2 + 1, "e")
                refolding_test = input_data_z[ibin2].Clone("refolding_test_%d_%s" % (i + 1, suffix))
                refolding_test.Divide(refolded_z)
                refolding_test_list_iter.append(refolding_test)
                cfolded_z = TCanvas("cfolded_z_%d_%s" % (i + 1, suffix), "1D output of folding" + suffix)
                setup_canvas(cfolded_z)
                setup_histogram(refolding_test, get_colour(1))
                line = TLine(round(self.lvarshape_binmin_reco[0], 2), 1, round(self.lvarshape_binmax_reco[-1], 2), 1)
                refolding_test.GetYaxis().SetRangeUser(0.5, 1.5)
                refolding_test.SetTitle("")
                refolding_test.SetXTitle(self.v_varshape_latex)
                refolding_test.SetYTitle("refolding test")
                refolding_test.Draw()
                line.Draw("same")
                latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.5, 0.82, "iteration %d" % (i + 1))
                draw_latex(latex2)
                cfolded_z.SaveAs("%s/refolding_%s_%d_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, i + 1, suffix_plot))
            refolding_test_list.append(refolding_test_list_iter)

            # refolding test for jet pt

            refolded_jetpt = refolded.ProjectionY("refolded_jetpt", 1, self.p_nbinshape_gen, "e")
            refolding_test_jetpt = input_data_jetpt.Clone("refolding_test_%d" % (i + 1))
            refolding_test_jetpt.Divide(refolded_jetpt)
            refolding_test_jetpt_list.append(refolding_test_jetpt)
            cfolded_jetpt = TCanvas("cfolded_jetpt_%d" % (i + 1), "1D output of folding")
            setup_canvas(cfolded_jetpt)
            setup_histogram(refolding_test_jetpt, get_colour(1))
            line = TLine(round(self.lvar2_binmin_gen[0], 2), 1, round(self.lvar2_binmax_gen[-1], 2), 1)
            refolding_test_jetpt.GetYaxis().SetRangeUser(0.5, 1.5)
            refolding_test_jetpt.SetTitle("")
            refolding_test_jetpt.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
            refolding_test_jetpt.SetYTitle("refolding test")
            refolding_test_jetpt.Draw()
            line.Draw("same")
            latex = TLatex(0.15, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_gen[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_gen[-1], 2)))
            draw_latex(latex)
            latex2 = TLatex(0.5, 0.82, "iteration %d" % (i + 1))
            draw_latex(latex2)
            cfolded_jetpt.SaveAs("%s/refolding_%s_%d.eps" % (self.d_resultsallpdata, self.v_var2_binning, i + 1))

        # end of iteration loop

        # plot the unfolded shape distributions for all iterations for each pt jet bin

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cconvergence_z = TCanvas("cconvergence_z " + suffix, "1D output of convergence")
            setup_canvas(cconvergence_z)
            cconvergence_z.SetRightMargin(0.2)
            leg_z = TLegend(.85, .45, 0.95, .85, "iteration")
            setup_legend(leg_z)
            l_his = [unfolded_z_scaled_list[i][ibin2] for i in range(self.niter_unfolding)]
            y_min_h, y_max_h = get_y_window_his(l_his)
            y_margin_up = 0.15
            y_margin_down = 0.05
            for i in range(self.niter_unfolding):
                setup_histogram(unfolded_z_scaled_list[i][ibin2], get_colour(i))
                leg_z.AddEntry(unfolded_z_scaled_list[i][ibin2], ("%d" % (i + 1)), "P")
                if i == 0:
                    unfolded_z_scaled_list[i][ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
                    unfolded_z_scaled_list[i][ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
                    unfolded_z_scaled_list[i][ibin2].SetTitle("")
                    unfolded_z_scaled_list[i][ibin2].SetXTitle(self.v_varshape_latex)
                    unfolded_z_scaled_list[i][ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
                unfolded_z_scaled_list[i][ibin2].Draw("same")
            leg_z.Draw("same")
            latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            cconvergence_z.SaveAs("%s/convergence_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

#            # plot the comparison of the final shape distribution, PYTHIA 6 and POWHEG
#
#            cinput_mc_gen_z = TCanvas("cinput_mc_gen_z " + suffix, "1D gen pythia z")
#            setup_canvas(cinput_mc_gen_z)
#            leg_input_mc_gen_z = TLegend(.15, .75, .45, .85)
#            setup_legend(leg_input_mc_gen_z)
#            setup_histogram(input_mc_gen_z[ibin2], get_colour(2), get_marker(1))
#            y_min_h, y_max_h = get_y_window_his([unfolded_z_scaled_list[i_iter_choice][ibin2], input_mc_gen_z[ibin2], input_powheg_z[ibin2]])
#            #y_min_g, y_max_g = get_y_window_gr([tg_powheg[ibin2]])
#            #y_min = min(y_min_g, y_min_h)
#            #y_max = max(y_max_g, y_max_h)
#            y_min = y_min_h
#            y_max = y_max_h
#            y_margin_up = 0.2
#            y_margin_down = 0.05
#            input_mc_gen_z[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
#            input_mc_gen_z[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
#            input_mc_gen_z[ibin2].SetTitle("")
#            input_mc_gen_z[ibin2].SetXTitle(self.v_varshape_latex)
#            input_mc_gen_z[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
#            input_mc_gen_z[ibin2].Draw()
#            setup_histogram(unfolded_z_scaled_list[i_iter_choice][ibin2], get_colour(1), get_marker(0))
#            leg_input_mc_gen_z.AddEntry(unfolded_z_scaled_list[i_iter_choice][ibin2], "unfolded data", "P")
#            unfolded_z_scaled_list[i_iter_choice][ibin2].Draw("same")
#            leg_input_mc_gen_z.AddEntry(input_mc_gen_z[ibin2], "PYTHIA 6", "P")
#            setup_histogram(input_powheg_z[ibin2], get_colour(3), get_marker(2))
#            leg_input_mc_gen_z.AddEntry(input_powheg_z[ibin2], "POWHEG + PYTHIA 6", "P")
#            input_powheg_z[ibin2].Draw("same")
#            #setup_tgraph(tg_powheg[ibin2], get_colour(3))
#            #tg_powheg[ibin2].Draw("5")
#            leg_input_mc_gen_z.Draw("same")
#            latex = TLatex(0.5, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
#            draw_latex(latex)
#            cinput_mc_gen_z.SaveAs("%s/unfolded_vs_mc_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))
#            #cinput_mc_gen_z.SaveAs("%s/unfolded_vs_mc_%s_%s.pdf" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))
#
#            # plot the comparison of the final cross-section and POWHEG
#
#            cinput_mc_gen_z_xsection = TCanvas("cinput_mc_gen_z_xsection " + suffix, "1D gen pythia z xsection")
#            setup_canvas(cinput_mc_gen_z_xsection)
#            leg_input_mc_gen_z_xsection = TLegend(.15, .75, .45, .85)
#            setup_legend(leg_input_mc_gen_z_xsection)
#            setup_histogram(unfolded_z_xsection_list[i_iter_choice][ibin2], get_colour(1), get_marker(0))
#            leg_input_mc_gen_z_xsection.AddEntry(unfolded_z_xsection_list[i_iter_choice][ibin2], "unfolded data", "P")
#            y_min_h, y_max_h = get_y_window_his([unfolded_z_xsection_list[i_iter_choice][ibin2], input_powheg_xsection_z[ibin2]])
#            #y_min_g, y_max_g = get_y_window_gr(tg_powheg_xsection[ibin2])
#            #y_min = min(y_min_g, y_min_h)
#            #y_max = max(y_max_g, y_max_h)
#            y_min = y_min_h
#            y_max = y_max_h
#            y_margin_up = 0.2
#            y_margin_down = 0.05
#            unfolded_z_xsection_list[i_iter_choice][ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
#            unfolded_z_xsection_list[i_iter_choice][ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
#            unfolded_z_xsection_list[i_iter_choice][ibin2].SetTitle("")
#            unfolded_z_xsection_list[i_iter_choice][ibin2].SetXTitle(self.v_varshape_latex)
#            unfolded_z_xsection_list[i_iter_choice][ibin2].SetYTitle("d#it{#sigma}/d%s (mb)" % self.v_varshape_latex)
#            unfolded_z_xsection_list[i_iter_choice][ibin2].GetYaxis().SetMaxDigits(3)
#            unfolded_z_xsection_list[i_iter_choice][ibin2].Draw()
#            setup_histogram(input_powheg_xsection_z[ibin2], get_colour(3), get_marker(2))
#            leg_input_mc_gen_z_xsection.AddEntry(input_powheg_xsection_z[ibin2], "POWHEG + PYTHIA 6", "P")
#            input_powheg_xsection_z[ibin2].Draw("same")
#            #setup_tgraph(tg_powheg_xsection[ibin2], get_colour(3))
#            #tg_powheg_xsection[ibin2].Draw("5")
#            latex = TLatex(0.5, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
#            draw_latex(latex)
#            leg_input_mc_gen_z_xsection.Draw("same")
#            cinput_mc_gen_z_xsection.SaveAs("%s/unfolded_vs_mc_%s_xsection_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))
#            #cinput_mc_gen_z_xsection.SaveAs("%s/unfolded_vs_mc_%s_xsection_%s.pdf" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])

            # convergence of the refolding test
            # plot the refolding test for all iterations together for each jet pt bin

            cconvergence_refolding_z = TCanvas("cconvergence_refolding_z " + suffix, "1D output of refolding convergence" + suffix)
            setup_canvas(cconvergence_refolding_z)
            cconvergence_refolding_z.SetRightMargin(0.2)
            leg_refolding_z = TLegend(.85, .45, 0.95, .85, "iteration")
            setup_legend(leg_refolding_z)
            l_his = [refolding_test_list[i][ibin2] for i in range(self.niter_unfolding)]
            y_min_h, y_max_h = get_y_window_his(l_his)
            y_margin_up = 0.15
            y_margin_down = 0.05
            for i in range(self.niter_unfolding):
                setup_histogram(refolding_test_list[i][ibin2], get_colour(i))
                leg_refolding_z.AddEntry(refolding_test_list[i][ibin2], ("%d" % (i + 1)), "P")
                refolding_test_list[i][ibin2].Draw("same")
                if i == 0:
                    refolding_test_list[i][ibin2].SetTitle("")
                    refolding_test_list[i][ibin2].SetXTitle(self.v_varshape_latex)
                    refolding_test_list[i][ibin2].SetYTitle("refolding test")
                    refolding_test_list[i][ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            leg_refolding_z.Draw("same")
            latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cconvergence_refolding_z.SaveAs("%s/convergence_refolding_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

            # compare the result before unfolding and after

            input_data_z_scaled = input_data_z[ibin2].Clone("input_data_z_scaled_%s" % suffix)
            input_data_z_scaled.Scale(1.0 / input_data_z_scaled.Integral(bin_int_first, input_data_z_scaled.GetNbinsX()), "width")
            cunfolded_not_z = TCanvas("cunfolded_not_z " + suffix, "Unfolded vs not Unfolded" + suffix)
            setup_canvas(cunfolded_not_z)
            leg_cunfolded_not_z = TLegend(.15, .75, .45, .85)
            setup_legend(leg_cunfolded_not_z)
            ibin_jetpt = input_mc_gen.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]) - 1
            setup_histogram(unfolded_z_scaled_list[i_iter_choice][ibin_jetpt], get_colour(1))
            leg_cunfolded_not_z.AddEntry(unfolded_z_scaled_list[i_iter_choice][ibin_jetpt], "unfolded", "P")
            y_min_h, y_max_h = get_y_window_his([unfolded_z_scaled_list[i_iter_choice][ibin_jetpt], input_data_z_scaled])
            y_margin_up = 0.2
            y_margin_down = 0.05
            unfolded_z_scaled_list[i_iter_choice][ibin_jetpt].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            unfolded_z_scaled_list[i_iter_choice][ibin_jetpt].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            unfolded_z_scaled_list[i_iter_choice][ibin_jetpt].SetTitle("")
            unfolded_z_scaled_list[i_iter_choice][ibin_jetpt].Draw()
            setup_histogram(input_data_z_scaled, get_colour(2), get_marker(1))
            leg_cunfolded_not_z.AddEntry(input_data_z_scaled, "not unfolded", "P")
            input_data_z_scaled.Draw("same")
            leg_cunfolded_not_z.Draw("same")
            latex = TLatex(0.5, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cunfolded_not_z.SaveAs("%s/unfolded_not_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

            # compare relative statistical uncertainties before unfolding and after

            h_unfolded_not_stat_error = TH1F("h_unfolded_not_stat_error" + suffix, "h_unfolded_not_stat_error" + suffix, self.p_nbinshape_reco, self.varshapebinarray_reco)
            for ibinshape in range(self.p_nbinshape_reco):
                error_on_unfolded = unfolded_z_scaled_list[i_iter_choice][ibin_jetpt].GetBinError(input_mc_gen.GetXaxis().FindBin(self.lvarshape_binmin_reco[ibinshape]))
                content_on_unfolded = unfolded_z_scaled_list[i_iter_choice][ibin_jetpt].GetBinContent(input_mc_gen.GetXaxis().FindBin(self.lvarshape_binmin_reco[ibinshape]))
                error_on_input_data = input_data_z_scaled.GetBinError(ibinshape + 1)
                content_on_input_data = input_data_z_scaled.GetBinContent(ibinshape + 1)
                if error_on_input_data != 0 and content_on_unfolded != 0:
                    h_unfolded_not_stat_error.SetBinContent(ibinshape + 1, (error_on_unfolded * content_on_input_data) / (content_on_unfolded * error_on_input_data))
                else:
                    h_unfolded_not_stat_error.SetBinContent(ibinshape + 1, 0.0)
            cunfolded_not_stat_error = TCanvas("cunfolded_not_stat_error " + suffix, "Ratio of stat error after to before unfolding" + suffix)
            setup_canvas(cunfolded_not_stat_error)
            cunfolded_not_stat_error.SetLeftMargin(0.13)
            setup_histogram(h_unfolded_not_stat_error, get_colour(1))
            h_unfolded_not_stat_error.SetTitle("Ratio of rel. stat. unc. after to before unfolding")
            h_unfolded_not_stat_error.SetXTitle(self.v_varshape_latex)
            h_unfolded_not_stat_error.SetYTitle("ratio")
            y_min_h = h_unfolded_not_stat_error.GetMinimum(0)
            y_max_h = h_unfolded_not_stat_error.GetMaximum()
            y_margin_up = 0.2
            y_margin_down = 0.05
            h_unfolded_not_stat_error.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            h_unfolded_not_stat_error.GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            h_unfolded_not_stat_error.SetTitleOffset(1.3, "Y")
            h_unfolded_not_stat_error.Draw()
            latex = TLatex(0.2, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cunfolded_not_stat_error.SaveAs("%s/unfolded_not_stat_error_%s.eps" % (self.d_resultsallpdata, suffix_plot))

        # plot the unfolded jet pt distributions for all iterations

        cconvergence_jetpt = TCanvas("cconvergence_jetpt ", "1D output of convergence")
        setup_canvas(cconvergence_jetpt)
        cconvergence_jetpt.SetLogy()
        cconvergence_jetpt.SetLeftMargin(0.13)
        cconvergence_jetpt.SetRightMargin(0.2)
        leg_jetpt = TLegend(.85, .45, 0.95, .85, "iteration")
        setup_legend(leg_jetpt)
        y_min_h, y_max_h = get_y_window_his(unfolded_jetpt_scaled_list)
        y_min_0 = min([h.GetMinimum(0) for h in unfolded_jetpt_scaled_list])
        if y_min_h <= 0:
            y_min_h = y_min_0
        y_margin_up = 0.15
        y_margin_down = 0.05
        for i in range(self.niter_unfolding):
            setup_histogram(unfolded_jetpt_scaled_list[i], get_colour(i))
            leg_jetpt.AddEntry(unfolded_jetpt_scaled_list[i], ("%d" % (i + 1)), "P")
            if i == 0:
                unfolded_jetpt_scaled_list[i].GetXaxis().SetRangeUser(round(self.lvar2_binmin_reco[0], 2), round(self.lvar2_binmax_reco[-1], 2))
                unfolded_jetpt_scaled_list[i].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up, True))
                unfolded_jetpt_scaled_list[i].SetTitle("")
                unfolded_jetpt_scaled_list[i].SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
                unfolded_jetpt_scaled_list[i].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s (#it{c}/GeV)" % self.p_latexbin2var)
                unfolded_jetpt_scaled_list[i].SetTitleOffset(1.5, "Y")
            unfolded_jetpt_scaled_list[i].Draw("same")
        leg_jetpt.Draw("same")
        latex = TLatex(0.2, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_gen[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_gen[-1], 2)))
        draw_latex(latex)
        cconvergence_jetpt.SaveAs("%s/convergence_%s.eps" % (self.d_resultsallpdata, self.v_var2_binning))

        # jet pt convergence refolding test
        # plot the refolding test of jet pt for all iterations

        cconvergence_refolding_jetpt = TCanvas("cconvergence_refolding_jetpt ", "1D output of refolding convergence")
        setup_canvas(cconvergence_refolding_jetpt)
        cconvergence_refolding_jetpt.SetRightMargin(0.2)
        leg_refolding_jetpt = TLegend(.85, .45, 0.95, .85, "iteration")
        setup_legend(leg_refolding_jetpt)
        y_min_h, y_max_h = get_y_window_his(refolding_test_jetpt_list)
        y_margin_up = 0.15
        y_margin_down = 0.05
        for i in range(self.niter_unfolding):
            setup_histogram(refolding_test_jetpt_list[i], get_colour(i))
            leg_refolding_jetpt.AddEntry(refolding_test_jetpt_list[i], ("%d" % (i + 1)), "P")
            if i == 0:
                refolding_test_jetpt_list[i].SetTitle("")
                refolding_test_jetpt_list[i].SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
                refolding_test_jetpt_list[i].SetYTitle("refolding test")
                refolding_test_jetpt_list[i].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
                refolding_test_jetpt_list[i].SetTitleOffset(1.5, "Y")
            refolding_test_jetpt_list[i].Draw("same")
        leg_refolding_jetpt.Draw("same")
        latex = TLatex(0.15, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_gen[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_gen[-1], 2)))
        draw_latex(latex)
        cconvergence_refolding_jetpt.SaveAs("%s/convergence_refolding_%s.eps" % (self.d_resultsallpdata, self.v_var2_binning))

    def unfolding_closure(self):
        self.loadstyle()
        fileouts = TFile.Open(self.file_unfold_closure, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(self.file_unfold_closure))
        unfolding_input_file = TFile.Open(self.n_fileresp)
        if not unfolding_input_file:
            self.logger.fatal(make_message_notfound(self.n_fileresp))
        response_matrix = unfolding_input_file.Get("response_matrix_closure")
        hzvsjetpt_reco_nocuts = unfolding_input_file.Get("hzvsjetpt_reco_nocuts_closure")
        hzvsjetpt_reco_eff = unfolding_input_file.Get("hzvsjetpt_reco_cuts_closure")
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)
        input_mc_det = unfolding_input_file.Get("input_closure_reco")
        input_mc_det.Multiply(hzvsjetpt_reco_eff)
        input_mc_gen = unfolding_input_file.Get("input_closure_gen")
        kinematic_eff = []
        hz_gen_nocuts = []
        input_mc_det_z = []
        input_mc_gen_z = []

        # Ignore the first bin for integration incase of untagged bin
        bin_int_first = 2 if self.lvarshape_binmin_reco[0] < 0 and "nsd" not in self.typean else 1

        kinematic_eff_jetpt = unfolding_input_file.Get("hjetpt_gen_cuts_closure")
        hjetpt_gen_nocuts = unfolding_input_file.Get("hjetpt_gen_nocuts_closure")
        kinematic_eff_jetpt.Divide(hjetpt_gen_nocuts)
        input_mc_gen_jetpt = input_mc_gen.ProjectionY("input_mc_gen_jetpt", 1, self.p_nbinshape_gen, "e")
        input_mc_gen_jetpt.Scale(1.0 / input_mc_gen_jetpt.Integral())

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            input_mc_det_z.append(input_mc_det.ProjectionX("input_mc_det_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_mc_det_z[ibin2].Scale(1.0 / input_mc_det_z[ibin2].Integral(bin_int_first, input_mc_det_z[ibin2].GetNbinsX()))

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_mc_gen_z[ibin2].Scale(1.0 / input_mc_gen_z[ibin2].Integral(bin_int_first, input_mc_gen_z[ibin2].GetNbinsX()))
            kinematic_eff.append(unfolding_input_file.Get("hz_gen_cuts_closure" + suffix))
            hz_gen_nocuts.append(unfolding_input_file.Get("hz_gen_nocuts_closure" + suffix))
            kinematic_eff[ibin2].Divide(hz_gen_nocuts[ibin2])

        unfolded_z_closure_list = []
        unfolded_jetpt_closure_list = []

        for i in range(self.niter_unfolding):
            unfolded_z_closure_list_iter = []

            # unfold

            unfolding_object = RooUnfoldBayes(response_matrix, input_mc_det, i + 1)
            unfolded_zvsjetpt = unfolding_object.Hreco(2)

            # plot closure test for shape for each iteration separately

            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                suffix_plot = "%s_%g_%g" % \
                         (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                unfolded_z = unfolded_zvsjetpt.ProjectionX("unfolded_z_%d_%s" % (i + 1, suffix), ibin2 + 1, ibin2 + 1, "e")
                unfolded_z.Divide(kinematic_eff[ibin2])
                unfolded_z.Scale(1.0 / unfolded_z.Integral(bin_int_first, unfolded_z.GetNbinsX()))
                unfolded_z.Divide(input_mc_gen_z[ibin2])
                fileouts.cd()
                unfolded_z.Write("closure_test_%d_%s" % (i + 1, suffix))
                unfolded_z_closure_list_iter.append(unfolded_z)

                cclosure_z = TCanvas("cclosure_z_%d_%s" % (i + 1, suffix), "1D output of closure" + suffix)
                setup_canvas(cclosure_z)
                line = TLine(round(self.lvarshape_binmin_reco[0], 2), 1, round(self.lvarshape_binmax_reco[-1], 2), 1)
                setup_histogram(unfolded_z, get_colour(1))
                unfolded_z.GetYaxis().SetRangeUser(0.5, 1.5)
                unfolded_z.GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
                unfolded_z.SetTitle("")
                unfolded_z.SetXTitle(self.v_varshape_latex)
                unfolded_z.SetYTitle("closure test")
                unfolded_z.Draw()
                line.Draw("same")
                latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.5, 0.82, "iteration %d" % (i + 1))
                draw_latex(latex2)
                cclosure_z.SaveAs("%s/closure_%s_%d_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, i + 1, suffix_plot))

            unfolded_z_closure_list.append(unfolded_z_closure_list_iter)

            unfolded_jetpt = unfolded_zvsjetpt.ProjectionY("unfolded_jetpt_%d" % (i + 1), 1, self.p_nbinshape_gen, "e")
            unfolded_jetpt.Divide(kinematic_eff_jetpt)
            unfolded_jetpt.Scale(1.0 / unfolded_jetpt.Integral())
            unfolded_jetpt.Divide(input_mc_gen_jetpt)
            fileouts.cd()
            unfolded_jetpt.Write("closure_test_jetpt_%d" % (i + 1))
            unfolded_jetpt_closure_list.append(unfolded_jetpt)

            # plot closure test for jet pt for each iteration separately

            cclosure_jetpt = TCanvas("cclosure_jetpt_%d" % (i + 1), "1D output of closure")
            setup_canvas(cclosure_jetpt)
            setup_histogram(unfolded_jetpt, get_colour(1))
            line = TLine(round(self.lvar2_binmin_gen[0], 2), 1, round(self.lvar2_binmax_gen[-1], 2), 1)
            unfolded_jetpt.GetYaxis().SetRangeUser(0.5, 1.5)
            unfolded_jetpt.SetTitle("")
            unfolded_jetpt.SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
            unfolded_jetpt.SetYTitle("closure test")
            unfolded_jetpt.Draw()
            line.Draw("same")
            latex = TLatex(0.15, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_gen[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_gen[-1], 2)))
            draw_latex(latex)
            latex2 = TLatex(0.5, 0.82, "iteration %d" % (i + 1))
            draw_latex(latex2)
            cclosure_jetpt.SaveAs("%s/closure_%s_%d.eps" % (self.d_resultsallpdata, self.v_var2_binning, i + 1))

        # plot closure test for shape for all iterations together

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cconvergence_closure_z = TCanvas("cconvergence_closure_z " + suffix, "1D output of closure convergence" + suffix)
            setup_canvas(cconvergence_closure_z)
            cconvergence_closure_z.SetRightMargin(0.2)
            leg_closure = TLegend(.85, .45, 0.95, .85, "iteration")
            setup_legend(leg_closure)
            l_his = [unfolded_z_closure_list[i][ibin2] for i in range(self.niter_unfolding)]
            y_min_h, y_max_h = get_y_window_his(l_his)
            y_margin_up = 0.15
            y_margin_down = 0.05
            for i in range(self.niter_unfolding):
                setup_histogram(unfolded_z_closure_list[i][ibin2], get_colour(i))
                leg_closure.AddEntry(unfolded_z_closure_list[i][ibin2], ("%d" % (i + 1)), "P")
                if i == 0:
                    unfolded_z_closure_list[i][ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
                    unfolded_z_closure_list[i][ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
                    unfolded_z_closure_list[i][ibin2].SetTitle("")
                    unfolded_z_closure_list[i][ibin2].SetXTitle(self.v_varshape_latex)
                    unfolded_z_closure_list[i][ibin2].SetYTitle("closure test")
                    unfolded_z_closure_list[i][ibin2].SetTitleOffset(1.2, "Y")
                unfolded_z_closure_list[i][ibin2].Draw("same")
            leg_closure.Draw("same")
            latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            cconvergence_closure_z.SaveAs("%s/convergence_closure_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix))

        # plot closure test for jet pt for all iterations together

        cconvergence_closure_jetpt = TCanvas("cconvergence_closure_jetpt ", "1D output of closure convergence")
        setup_canvas(cconvergence_closure_jetpt)
        cconvergence_closure_jetpt.SetRightMargin(0.2)
        leg_closure_jetpt = TLegend(.85, .45, 0.95, .85, "iteration")
        setup_legend(leg_closure_jetpt)
        y_min_h, y_max_h = get_y_window_his(unfolded_jetpt_closure_list)
        y_margin_up = 0.15
        y_margin_down = 0.05
        for i in range(self.niter_unfolding):
            setup_histogram(unfolded_jetpt_closure_list[i], get_colour(i))
            leg_closure_jetpt.AddEntry(unfolded_jetpt_closure_list[i], ("%d" % (i + 1)), "P")
            if i == 0:
                unfolded_jetpt_closure_list[i].GetXaxis().SetRangeUser(round(self.lvar2_binmin_gen[0], 2), round(self.lvar2_binmax_gen[-1], 2))
                unfolded_jetpt_closure_list[i].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
                unfolded_jetpt_closure_list[i].SetTitle("")
                unfolded_jetpt_closure_list[i].SetXTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
                unfolded_jetpt_closure_list[i].SetYTitle("closure test")
                unfolded_jetpt_closure_list[i].SetTitleOffset(1.2, "Y")
            unfolded_jetpt_closure_list[i].Draw("same")
        leg_closure_jetpt.Draw("same")
        latex = TLatex(0.15, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_gen[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_gen[-1], 2)))
        draw_latex(latex)
        cconvergence_closure_jetpt.SaveAs("%s/convergence_closure_%s.eps" % (self.d_resultsallpdata, self.v_var2_binning))












    def jetsystematics(self):
        self.loadstyle()
        string_default = "default/default"
        if string_default not in self.d_resultsallpdata:
            self.logger.fatal("Not a default database! Cannot run systematics.")

        debug = True
        if debug:
            print("Categories: ", self.systematic_catnames)
            print("Category labels: ", self.systematic_catlabels)
            print("Numbers of variations: ", self.systematic_variations)
            print("Variations: ", self.systematic_varnames)
            print("Variation labels: ", self.systematic_varlabels)
            print("Correlation: ", self.systematic_correlation)
            print("RMS: ", self.systematic_rms)
            print("Symmetrisation: ", self.systematic_symmetrise)
            print("RMS both sides: ", self.systematic_rms_both_sides)
            print("Feed-down variations: ", self.powheg_nonprompt_varnames)

        path_def = self.file_unfold
        input_file_default = TFile.Open(path_def)
        if not input_file_default:
            self.logger.fatal(make_message_notfound(path_def))

        # get the central prompt POWHEG histogram

        #input_powheg = self.get_simulated_yields(self.powheg_path_prompt, 2, True)
        #if not input_powheg:
            #self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_prompt))
        #input_powheg.SetName("fh2_prompt_%s" % self.v_varshape_binning)
        #input_powheg_xsection = input_powheg.Clone(input_powheg.GetName() + "_xsec")

        # get the prompt POWHEG variations and calculate their spread

        #input_powheg_sys = []
        #input_powheg_xsection_sys = []
        #for i_powheg in range(len(self.powheg_prompt_variations)):
            #path = "%s%s.root" % (self.powheg_prompt_variations_path, self.powheg_prompt_variations[i_powheg])
            #input_powheg_sys_i = self.get_simulated_yields(path, 2, True)
            #if not input_powheg_sys_i:
                #self.logger.fatal(make_message_notfound("simulated yields", path))
            #input_powheg_sys_i.SetName("fh2_prompt_%s_%d" % (self.v_varshape_binning, i_powheg))
            #input_powheg_sys.append(input_powheg_sys_i)
            #input_powheg_xsection_sys_i = input_powheg_sys_i.Clone(input_powheg_sys_i.GetName() + "_xsec")
            #input_powheg_xsection_sys.append(input_powheg_xsection_sys_i)
        #input_powheg_z = []
        #input_powheg_xsection_z = []
        #input_powheg_sys_z = []
        #input_powheg_xsection_sys_z = []
        #tg_powheg = []
        #tg_powheg_xsection = []
        #for ibin2 in range(self.p_nbin2_gen):
            #suffix = "%s_%.2f_%.2f" % \
                     #(self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            #input_powheg_z.append(input_powheg.ProjectionX("input_powheg_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            #input_powheg_z[ibin2].Scale(1.0 / input_powheg_z[ibin2].Integral(input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[0]), input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])), "width")
            #input_powheg_xsection_z.append(input_powheg_xsection.ProjectionX("input_powheg_xsection_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            #input_powheg_xsection_z[ibin2].Scale(1.0, "width")
            #input_powheg_sys_z_iter = []
            #input_powheg_xsection_sys_z_iter = []
            #for i_powheg in range(len(self.powheg_prompt_variations)):
                #input_powheg_sys_z_iter.append(input_powheg_sys[i_powheg].ProjectionX("input_powheg_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
                #input_powheg_sys_z_iter[i_powheg].Scale(1.0 / input_powheg_sys_z_iter[i_powheg].Integral(input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[0]), input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[-1])), "width")
                #input_powheg_xsection_sys_z_iter.append(input_powheg_xsection_sys[i_powheg].ProjectionX("input_powheg_xsection_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
            #input_powheg_sys_z.append(input_powheg_sys_z_iter)
            #input_powheg_xsection_sys_z.append(input_powheg_xsection_sys_z_iter)
            #tg_powheg.append(tg_sys(input_powheg_z[ibin2], input_powheg_sys_z[ibin2]))
            #tg_powheg_xsection.append(tg_sys(input_powheg_xsection_z[ibin2], input_powheg_xsection_sys_z[ibin2]))

        # get the default (central value) result histograms

        input_histograms_default = []
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            name_his = "unfolded_z_sel_%s" % suffix
            input_histograms_default.append(input_file_default.Get(name_his))
            if not input_histograms_default[ibin2]:
                self.logger.fatal(make_message_notfound(name_his, path_def))

        # get the files containing result variations

        input_files_sys = []
        for sys_cat in range(self.n_sys_cat):
            input_files_sysvar = []
            for sys_var, varname in enumerate(self.systematic_varnames[sys_cat]):
                path = path_def.replace(string_default, self.systematic_catnames[sys_cat] + "/" + varname)
                input_files_sysvar.append(TFile.Open(path))
                if not input_files_sysvar[sys_var]:
                    self.logger.fatal(make_message_notfound(path))
            input_files_sys.append(input_files_sysvar)

        # get the variation result histograms

        input_histograms_sys = []
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            name_his = "unfolded_z_sel_%s" % suffix
            input_histograms_syscat = []
            for sys_cat in range(self.n_sys_cat):
                input_histograms_syscatvar = []
                for sys_var in range(self.systematic_variations[sys_cat]):
                    string_catvar = self.systematic_catnames[sys_cat] + "/" + self.systematic_varnames[sys_cat][sys_var]
                    # FIXME exception for different jet pt binning pylint: disable=fixme
                    name_his_orig = name_his
                    if ibin2 == 0 and string_catvar == "binning/pt_jet_0":
                        name_his = "unfolded_z_sel_%s_%.2f_%.2f" % (self.v_var2_binning, 8, self.lvar2_binmax_gen[ibin2])
                    input_histograms_syscatvar.append(input_files_sys[sys_cat][sys_var].Get(name_his))
                    name_his = name_his_orig
                    path_file = path_def.replace(string_default, string_catvar)
                    if not input_histograms_syscatvar[sys_var]:
                        self.logger.fatal(make_message_notfound(name_his, path_file))
                    if debug:
                        print("Variation: %s, %s: got histogram %s from file %s" % (self.systematic_catnames[sys_cat], self.systematic_varnames[sys_cat][sys_var], name_his, path_file))
                    #input_histograms_syscatvar[sys_var].Scale(1.0, "width") #remove these later and put normalisation directly in systematics
                input_histograms_syscat.append(input_histograms_syscatvar)
            input_histograms_sys.append(input_histograms_syscat)

        # plot the variations

        for ibin2 in range(self.p_nbin2_gen):

            # plot all the variations together

            suffix = "%s_%g_%g" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            nsys = 0
            csysvar = TCanvas("csysvar_%s" % suffix, "systematic variations" + suffix)
            setup_canvas(csysvar)
            leg_sysvar = TLegend(.75, .15, 0.95, .85, "variation")
            setup_legend(leg_sysvar)
            leg_sysvar.AddEntry(input_histograms_default[ibin2], "default", "P")
            setup_histogram(input_histograms_default[ibin2])
            l_his_all = [his_var for l_cat in input_histograms_sys[ibin2] for his_var in l_cat] + [input_histograms_default[ibin2]]
            y_min, y_max = get_y_window_his(l_his_all)
            y_margin_up = 0.15
            y_margin_down = 0.05
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
            #input_histograms_default[ibin2].GetYaxis().SetRangeUser(0.0, input_histograms_default[ibin2].GetMaximum() * 1.5)
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw()
            for sys_cat in range(self.n_sys_cat):
                for sys_var in range(self.systematic_variations[sys_cat]):
                    leg_sysvar.AddEntry(input_histograms_sys[ibin2][sys_cat][sys_var], ("%s, %s" % \
                        (self.systematic_catlabels[sys_cat], self.systematic_varlabels[sys_cat][sys_var])), "P")
                    setup_histogram(input_histograms_sys[ibin2][sys_cat][sys_var], get_colour(nsys + 1))
                    input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1
            latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            #leg_sysvar.Draw("same")
            csysvar.SaveAs("%s/sys_var_all_%s.eps" % (self.d_resultsallpdata, suffix))

            # plot the variations for each category separately

            for sys_cat in range(self.n_sys_cat):
                suffix2 = self.systematic_catnames[sys_cat]
                nsys = 0
                csysvar_each = TCanvas("csysvar_%s_%s" % (suffix2, suffix), "systematic variations" + suffix2 + suffix)
                setup_canvas(csysvar_each)
                csysvar_each.SetRightMargin(0.25)
                leg_sysvar_each = TLegend(.77, .2, 0.95, .85, self.systematic_catlabels[sys_cat]) # Rg
                setup_legend(leg_sysvar_each)
                leg_sysvar_each.AddEntry(input_histograms_default[ibin2], "default", "P")
                setup_histogram(input_histograms_default[ibin2])
                y_min, y_max = get_y_window_his(input_histograms_sys[ibin2][sys_cat] + [input_histograms_default[ibin2]])
                y_margin_up = 0.15
                y_margin_down = 0.05
                for sys_var in range(self.systematic_variations[sys_cat]):
                    if sys_var == 0:
                        input_histograms_default[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
                        input_histograms_default[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
                        input_histograms_default[ibin2].SetTitle("")
                        input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
                        input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
                        input_histograms_default[ibin2].Draw()
                    leg_sysvar_each.AddEntry(input_histograms_sys[ibin2][sys_cat][sys_var], self.systematic_varlabels[sys_cat][sys_var], "P")
                    setup_histogram(input_histograms_sys[ibin2][sys_cat][sys_var], get_colour(nsys + 1), get_marker(nsys + 1))
                    input_histograms_sys[ibin2][sys_cat][sys_var].Draw("same")
                    nsys = nsys + 1
                latex = TLatex(0.15, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
                draw_latex(latex)
                leg_sysvar_each.Draw("same")
                csysvar_each.SaveAs("%s/sys_var_%s_%s.eps" % (self.d_resultsallpdata, suffix2, suffix))

        # calculate the systematic uncertainties

        sys_up = [] # list of absolute upward uncertainties for all categories, shape bins, pt_jet bins
        sys_down = [] # list of absolute downward uncertainties for all categories, shape bins, pt_jet bins
        sys_up_full = [] # list of combined absolute upward uncertainties for all shape bins, pt_jet bins
        sys_down_full = [] # list of combined absolute downward uncertainties for all shape bins, pt_jet bins
        for ibin2 in range(self.p_nbin2_gen): # pylint: disable=too-many-nested-blocks
            sys_up_jetpt = [] # list of absolute upward uncertainties for all categories and shape bins in a given pt_jet bin
            sys_down_jetpt = [] # list of absolute downward uncertainties for all categories and shape bins in a given pt_jet bin
            sys_up_z_full = [] # list of combined absolute upward uncertainties for all shape bins in a given pt_jet bin
            sys_down_z_full = [] # list of combined absolute upward uncertainties for all shape bins in a given pt_jet bin
            for ibinshape in range(self.p_nbinshape_gen):
                sys_up_z = [] # list of absolute upward uncertainties for all categories in a given (pt_jet, shape) bin
                sys_down_z = [] # list of absolute downward uncertainties for all categories in a given (pt_jet, shape) bin
                error_full_up = 0 # combined absolute upward uncertainty in a given (pt_jet, shape) bin
                error_full_down = 0 # combined absolute downward uncertainty in a given (pt_jet, shape) bin
                for sys_cat in range(self.n_sys_cat):
                    error_var_up = 0 # absolute upward uncertainty for a given category in a given (pt_jet, shape) bin
                    error_var_down = 0 # absolute downward uncertainty for a given category in a given (pt_jet, shape) bin
                    count_sys_up = 0
                    count_sys_down = 0
                    for sys_var in range(self.systematic_variations[sys_cat]):
                        # FIXME exception for the untagged bin pylint: disable=fixme
                        bin_first = 2 if "untagged" in self.systematic_varlabels[sys_cat][sys_var] else 1
                        error = input_histograms_sys[ibin2][sys_cat][sys_var].GetBinContent(ibinshape + bin_first) - input_histograms_default[ibin2].GetBinContent(ibinshape + 1)
                        if error >= 0:
                            if self.systematic_rms[sys_cat] is True:
                                error_var_up += error * error
                                count_sys_up = count_sys_up + 1
                            else:
                                if error > error_var_up:
                                    error_var_up = error
                        else:
                            if self.systematic_rms[sys_cat] is True:
                                if self.systematic_rms_both_sides[sys_cat] is True:
                                    error_var_up += error * error
                                    count_sys_up = count_sys_up + 1
                                else:
                                    error_var_down += error * error
                                    count_sys_down = count_sys_down + 1
                            else:
                                if abs(error) > error_var_down:
                                    error_var_down = abs(error)
                    if self.systematic_rms[sys_cat] is True:
                        if count_sys_up != 0:
                            error_var_up = error_var_up/count_sys_up
                        else:
                            error_var_up = 0.0
                        error_var_up = sqrt(error_var_up)
                        if count_sys_down != 0:
                            error_var_down = error_var_down/count_sys_down
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

        tgsys = [] # list of graphs with combined absolute uncertainties for all pt_jet bins
        tgsys_cat = [] # list of graphs with relative uncertainties for all categories, pt_jet bins
        for ibin2 in range(self.p_nbin2_gen):

            # combined uncertainties

            shapebins_centres = []
            shapebins_contents = []
            shapebins_widths_up = []
            shapebins_widths_down = []
            shapebins_error_up = []
            shapebins_error_down = []
            for ibinshape in range(self.p_nbinshape_gen):
                shapebins_centres.append(input_histograms_default[ibin2].GetBinCenter(ibinshape + 1))
                shapebins_contents.append(input_histograms_default[ibin2].GetBinContent(ibinshape + 1))
                shapebins_widths_up.append(input_histograms_default[ibin2].GetBinWidth(ibinshape + 1) * 0.5)
                shapebins_widths_down.append(input_histograms_default[ibin2].GetBinWidth(ibinshape + 1) * 0.5)
                shapebins_error_up.append(sys_up_full[ibin2][ibinshape])
                shapebins_error_down.append(sys_down_full[ibin2][ibinshape])
            shapebins_centres_array = array("d", shapebins_centres)
            shapebins_contents_array = array("d", shapebins_contents)
            shapebins_widths_up_array = array("d", shapebins_widths_up)
            shapebins_widths_down_array = array("d", shapebins_widths_down)
            shapebins_error_up_array = array("d", shapebins_error_up)
            shapebins_error_down_array = array("d", shapebins_error_down)
            tgsys.append(TGraphAsymmErrors(self.p_nbinshape_gen, \
                                           shapebins_centres_array, \
                                           shapebins_contents_array, \
                                           shapebins_widths_down_array, \
                                           shapebins_widths_up_array, \
                                           shapebins_error_down_array, \
                                           shapebins_error_up_array))

            # relative uncertainties per category

            tgsys_cat_z = [] # list of graphs with relative uncertainties for all categories in a given pt_jet bin
            for sys_cat in range(self.n_sys_cat):
                shapebins_contents_cat = []
                shapebins_error_up_cat = []
                shapebins_error_down_cat = []
                for ibinshape in range(self.p_nbinshape_gen):
                    shapebins_contents_cat.append(0)
                    shapebins_error_up_cat.append(sys_up[ibin2][ibinshape][sys_cat]/input_histograms_default[ibin2].GetBinContent(ibinshape + 1))
                    shapebins_error_down_cat.append(sys_down[ibin2][ibinshape][sys_cat]/input_histograms_default[ibin2].GetBinContent(ibinshape + 1))
                shapebins_contents_cat_array = array("d", shapebins_contents_cat)
                shapebins_error_up_cat_array = array("d", shapebins_error_up_cat)
                shapebins_error_down_cat_array = array("d", shapebins_error_down_cat)
                tgsys_cat_z.append(TGraphAsymmErrors(self.p_nbinshape_gen, \
                                                     shapebins_centres_array, \
                                                     shapebins_contents_cat_array, \
                                                     shapebins_widths_down_array, \
                                                     shapebins_widths_up_array, \
                                                     shapebins_error_down_cat_array, \
                                                     shapebins_error_up_cat_array))
            tgsys_cat.append(tgsys_cat_z)

        # relative statistical uncertainty of the central values

        h_default_stat_err = []
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            h_default_stat_err.append(input_histograms_default[ibin2].Clone("h_default_stat_err" + suffix))
            for i in range(h_default_stat_err[ibin2].GetNbinsX()):
                h_default_stat_err[ibin2].SetBinContent(i + 1, 0)
                h_default_stat_err[ibin2].SetBinError(i + 1, input_histograms_default[ibin2].GetBinError(i + 1) / input_histograms_default[ibin2].GetBinContent(i + 1))

        # get the prompt PYTHIA histograms

        #file_sim_out = TFile.Open("%s/simulations.root" % self.d_resultsallpdata, "recreate")
        #input_pythia8 = []
        #input_pythia8_xsection = []
        #input_pythia8_z = []
        #input_pythia8_xsection_z = []
        #for i_pythia8 in range(len(self.pythia8_prompt_variations)):
            #path = "%s%s.root" % (self.pythia8_prompt_variations_path, self.pythia8_prompt_variations[i_pythia8])
            #input_pythia8_i = self.get_simulated_yields(path, 2, True)
            #if not input_pythia8_i:
                #self.logger.fatal(make_message_notfound("simulated yields", path))
            #input_pythia8_i.SetName("fh2_pythia_prompt_%s_%d" % (self.v_varshape_binning, i_pythia8))
            #input_pythia8.append(input_pythia8_i)
            #input_pythia8_xsection_i = input_pythia8_i.Clone(input_pythia8_i.GetName() + "_xsec")
            #input_pythia8_xsection.append(input_pythia8_xsection_i)

            # Ensure correct binning: x - shape, y - jet pt
            #if not equal_binning_lists(input_pythia8[i_pythia8], list_x=self.varshaperanges_gen):
                #self.logger.fatal("Error: Incorrect binning in x.")
            #if not equal_binning_lists(input_pythia8[i_pythia8], list_y=self.var2ranges_gen):
                #self.logger.fatal("Error: Incorrect binning in y.")
            #if not equal_binning_lists(input_pythia8_xsection[i_pythia8], list_x=self.varshaperanges_gen):
                #self.logger.fatal("Error: Incorrect binning in x.")
            #if not equal_binning_lists(input_pythia8_xsection[i_pythia8], list_y=self.var2ranges_gen):
                #self.logger.fatal("Error: Incorrect binning in y.")

            #input_pythia8_z_jetpt = []
            #input_pythia8_xsection_z_jetpt = []
            #for ibin2 in range(self.p_nbin2_gen):
                #suffix = "%s_%.2f_%.2f" % \
                     #(self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                #input_pythia8_z_jetpt.append(input_pythia8[i_pythia8].ProjectionX("input_pythia8" + self.pythia8_prompt_variations[i_pythia8]+suffix, ibin2 + 1, ibin2 + 1, "e"))
                #input_pythia8_z_jetpt[ibin2].Scale(1.0 / input_pythia8_z_jetpt[ibin2].Integral(1, -1), "width")
                #pythia8_out = input_pythia8_z_jetpt[ibin2]
                #file_sim_out.cd()
                #pythia8_out.Write()
                #pythia8_out.SetDirectory(0)
                #input_pythia8_xsection_z_jetpt.append(input_pythia8_xsection[i_pythia8].ProjectionX("input_pythia8_xsection" + self.pythia8_prompt_variations[i_pythia8] + suffix, ibin2 + 1, ibin2 + 1, "e"))
            #input_pythia8_z.append(input_pythia8_z_jetpt)
            #input_pythia8_xsection_z.append(input_pythia8_xsection_z_jetpt)
        #file_sim_out.Close()

        file_sys_out = TFile.Open("%s/systematics_results.root" % self.d_resultsallpdata, "recreate")

        for ibin2 in range(self.p_nbin2_gen):

            # plot the results with systematic uncertainties

            suffix = "%s_%g_%g" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cfinalwsys = TCanvas("cfinalwsys " + suffix, "final result with systematic uncertainties" + suffix)
            setup_canvas(cfinalwsys)
            leg_finalwsys = TLegend(.7, .75, .85, .85)
            setup_legend(leg_finalwsys)
            leg_finalwsys.AddEntry(input_histograms_default[ibin2], "data", "P")
            setup_histogram(input_histograms_default[ibin2], get_colour(0))
            y_min_g, y_max_g = get_y_window_gr([tgsys[ibin2]])
            y_min_h, y_max_h = get_y_window_his([input_histograms_default[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.35
            y_margin_down = 0.05
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw("AXIS")
            #input_histograms_default[ibin2].Draw("")
            setup_tgraph(tgsys[ibin2], get_colour(0, 2))
            tgsys[ibin2].Draw("5")
            leg_finalwsys.AddEntry(tgsys[ibin2], "syst. unc.", "F")
            input_histograms_default[ibin2].Draw("SAME")
            input_histograms_default[ibin2].Draw("AXISSAME")
            #PREL latex = TLatex(0.15, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.15, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.15, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4, #left|#it{#eta}_{jet}#right| #leq 0.5" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.15, 0.72, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            #latex3 = TLatex(0.15, 0.67, "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2])))
            #draw_latex(latex3)
            leg_finalwsys.Draw("same")
            cfinalwsys.SaveAs("%s/final_wsys_%s.pdf" % (self.d_resultsallpdata, suffix))
            tgsys[ibin2].Write("tgsys_%s" % (suffix))

            # plot the results with systematic uncertainties and models

            cfinalwsys_wmodels = TCanvas("cfinalwsys_wmodels " + suffix, "final result with systematic uncertainties with models" + suffix)
            setup_canvas(cfinalwsys_wmodels)
            if self.typean == "jet_zg":
                leg_finalwsys_wmodels = TLegend(.55, .5, .65, .7)
            elif self.typean == "jet_rg":
                leg_finalwsys_wmodels = TLegend(.15, .45, .25, .65)
            else:
                leg_finalwsys_wmodels = TLegend(.55, .5, .65, .7)
            setup_legend(leg_finalwsys_wmodels)
            leg_finalwsys_wmodels.AddEntry(input_histograms_default[ibin2], "data", "P")
            setup_histogram(input_histograms_default[ibin2], get_colour(0))
            #y_min_g, y_max_g = get_y_window_gr([tgsys[ibin2], tg_powheg[ibin2]])
            y_min_h, y_max_h = get_y_window_his([input_histograms_default[ibin2]])
                #[input_pythia8_z[i][ibin2] for i in range(len(self.pythia8_prompt_variations))])
            y_min = y_min_h #min(tgsys[ibin2], input_histograms_default[ibin2])
            y_max = y_max_h #max(tgsys[ibin2], input_histograms_default[ibin2])
            y_margin_up = 0.35
            y_margin_down = 0.05
            y_plot_min, y_plot_max = get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(y_plot_min, y_plot_max)
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw()
            setup_tgraph(tgsys[ibin2], get_colour(0, 2))
            tgsys[ibin2].Draw("5")
            leg_finalwsys_wmodels.AddEntry(tgsys[ibin2], "syst. unc.", "F")
            #setup_histogram(input_powheg_z[ibin2], get_colour(1), get_marker(1))
            #leg_finalwsys_wmodels.AddEntry(input_powheg_z[ibin2], "POWHEG #plus PYTHIA 6", "P")
            #input_powheg_z[ibin2].Draw("same")
            #setup_tgraph(tg_powheg[ibin2], get_colour(1))
            #tg_powheg[ibin2].Draw("5")
            #for i_pythia8 in range(len(self.pythia8_prompt_variations)):
                #setup_histogram(input_pythia8_z[i_pythia8][ibin2], get_colour(i_pythia8 + 2), get_marker(i_pythia8 + 2), 2.)
                #leg_finalwsys_wmodels.AddEntry(input_pythia8_z[i_pythia8][ibin2], self.pythia8_prompt_variations_legend[i_pythia8], "P")
                #input_pythia8_z[i_pythia8][ibin2].Draw("same")
            input_histograms_default[ibin2].Draw("AXISSAME")
            #PREL latex = TLatex(0.15, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.15, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.15, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4, #left|#it{#eta}_{jet}#right| #leq 0.5" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.15, 0.72, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            #latex3 = TLatex(0.15, 0.7, "%g #leq %s < %g" % (round(self.lvarshape_binmin_reco[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_reco[-1], 2)))
            #latex3 = TLatex(0.15, 0.67, "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2])))
            #draw_latex(latex3)
            leg_finalwsys_wmodels.Draw("same")
            if self.typean == "jet_rg":
                # plot the theta_g axis
                axis_rg = input_histograms_default[ibin2].GetXaxis()
                rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
                rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
                radius_jet = 0.4
                thetag_min = rg_min / radius_jet
                thetag_max = rg_max / radius_jet
                y_axis = y_plot_max
                axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
                axis_thetag.SetTitle("#it{#theta}_{g} = #it{R}_{g}/#it{R}_{jet}")
                axis_thetag.SetTitleSize(0.037)
                axis_thetag.SetLabelSize(0.037)
                axis_thetag.SetTitleFont(42)
                axis_thetag.SetLabelFont(42)
                axis_thetag.SetLabelOffset(0)
                cfinalwsys_wmodels.SetTickx(0)
                axis_thetag.Draw("same")
            cfinalwsys_wmodels.SaveAs("%s/final_wsys_wmodels_%s.pdf" % (self.d_resultsallpdata, suffix))

            # plot the relative systematic uncertainties for all categories together

            text_ptjet_full = self.text_ptjet % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2])

            # preliminary figure
            crelativesys = TCanvas("crelativesys " + suffix, "relative systematic uncertainties" + suffix)
            gStyle.SetErrorX(0)
            setup_canvas(crelativesys)
            crelativesys.SetCanvasSize(900, 800)
            crelativesys.SetBottomMargin(self.margins_can[0])
            crelativesys.SetLeftMargin(self.margins_can[1])
            crelativesys.SetTopMargin(self.margins_can[2])
            crelativesys.SetRightMargin(self.margins_can[3])
            leg_relativesys = TLegend(.68, .65, .88, .91)
            setup_legend(leg_relativesys, textsize=self.fontsize)
            y_min_g, y_max_g = get_y_window_gr(tgsys_cat[ibin2])
            y_min_h, y_max_h = get_y_window_his([h_default_stat_err[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.38
            y_margin_down = 0.05
            setup_histogram(h_default_stat_err[ibin2])
            h_default_stat_err[ibin2].SetMarkerStyle(0)
            h_default_stat_err[ibin2].SetMarkerSize(0)
            leg_relativesys.AddEntry(h_default_stat_err[ibin2], "stat. unc.", "E")
            for sys_cat in range(self.n_sys_cat):
                setup_tgraph(tgsys_cat[ibin2][sys_cat], get_colour(sys_cat + 1, 0))
                tgsys_cat[ibin2][sys_cat].SetTitle("")
                tgsys_cat[ibin2][sys_cat].SetLineWidth(3)
                tgsys_cat[ibin2][sys_cat].SetFillStyle(0)
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetLimits(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
                if self.shape == "nsd":
                    tgsys_cat[ibin2][sys_cat].GetXaxis().SetNdivisions(5)
                    shrink_err_x(tgsys_cat[ibin2][sys_cat], 0.2)
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetTitle(self.v_varshape_latex)
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetTitle("relative systematic uncertainty")
                tgsys_cat[ibin2][sys_cat].GetXaxis().SetTitleOffset(self.offsets_axes[0])
                tgsys_cat[ibin2][sys_cat].GetYaxis().SetTitleOffset(self.offsets_axes[1])
                leg_relativesys.AddEntry(tgsys_cat[ibin2][sys_cat], self.systematic_catlabels[sys_cat], "F")
                if sys_cat == 0:
                    tgsys_cat[ibin2][sys_cat].Draw("A2")
                else:
                    tgsys_cat[ibin2][sys_cat].Draw("2")
            h_default_stat_err[ibin2].Draw("same")
            h_default_stat_err[ibin2].Draw("axissame")
            # Draw LaTeX
            y_latex = self.y_latex_top
            list_latex = []
            for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, self.text_ptcut, self.text_sd]:
                latex = TLatex(self.x_latex, y_latex, text_latex)
                list_latex.append(latex)
                draw_latex(latex, textsize=self.fontsize)
                y_latex -= self.y_step
            leg_relativesys.Draw("same")
            crelativesys.SaveAs("%s/sys_unc_%s.eps" % (self.d_resultsallpdata, suffix))
            if ibin2 == 1:
                crelativesys.SaveAs("%s/%s_sys_unc_%s_incl.pdf" % (self.d_resultsallpdata, self.shape, suffix))
            gStyle.SetErrorX(0.5)

        # plot the feed-down fraction with systematic uncertainties from POWHEG

        #file_feeddown = TFile.Open(self.file_feeddown)
        #if not file_feeddown:
            #self.logger.fatal(make_message_notfound(self.file_feeddown))
        #file_feeddown_variations = []
        #for i_powheg, varname in enumerate(self.powheg_nonprompt_varnames):
            #path = self.file_feeddown.replace(string_default, "powheg/" + varname)
            #file_feeddown_variations.append(TFile.Open(path))
            #if not file_feeddown_variations[i_powheg]:
                #self.logger.fatal(make_message_notfound(path))
        #h_feeddown_fraction = [] # list of the central feed-down fractions for all pt_jet bins
        #h_feeddown_fraction_variations = [] # list of feed-down fractions for all POWHEG variations and all pt_jet bins
        #tg_feeddown_fraction = [] # list of graphs with the spread of values for all pt_jet bins
        #for ibin2 in range(self.p_nbin2_reco):
            #suffix = "%s_%.2f_%.2f" % \
              #(self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            #suffix_plot = "%s_%g_%g" % \
              #(self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            #h_feeddown_fraction_variations_niter = [] # list of feed-down fractions for all POWHEG variations in a given pt_jet bin
            #h_feeddown_fraction.append(file_feeddown.Get("feeddown_fraction" + suffix))
            #for i_powheg in range(len(self.powheg_nonprompt_varnames)):
                #h_feeddown_fraction_variations_niter.append(file_feeddown_variations[i_powheg].Get("feeddown_fraction" + suffix))

            #h_feeddown_fraction_variations.append(h_feeddown_fraction_variations_niter)
            ## get the graph with the spread of values for all the POWHEG variations
            #tg_feeddown_fraction.append(tg_sys(h_feeddown_fraction[ibin2], h_feeddown_fraction_variations[ibin2]))

            #cfeeddown_fraction = TCanvas("cfeeddown_fraction " + suffix, "feeddown fraction" + suffix)
            #setup_canvas(cfeeddown_fraction)
            #cfeeddown_fraction.SetLeftMargin(0.13)
            #leg_fd = TLegend(.67, .6, .85, .85)
            #setup_legend(leg_fd, 0.025)
            #setup_histogram(h_feeddown_fraction[ibin2], get_colour(0))
            #y_min_g, y_max_g = get_y_window_gr([tg_feeddown_fraction[ibin2]])
            #y_min_h, y_max_h = get_y_window_his([h_feeddown_fraction[ibin2]])
            #y_min = min(y_min_g, y_min_h)
            #y_max = max(y_max_g, y_max_h)
            #y_margin_up = 0.45
            #y_margin_down = 0.05
            #h_feeddown_fraction[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
            #h_feeddown_fraction[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            #h_feeddown_fraction[ibin2].SetXTitle(self.v_varshape_latex)
            #h_feeddown_fraction[ibin2].SetYTitle("feed-down fraction")
            #h_feeddown_fraction[ibin2].SetTitleOffset(1.4, "Y")
            #h_feeddown_fraction[ibin2].SetTitle("")
            #h_feeddown_fraction[ibin2].Draw("same")
            ##tg_feeddown_fraction[ibin2].Draw("5")
            #leg_fd.AddEntry(h_feeddown_fraction[ibin2], "default", "P")
            #for i, his in enumerate(h_feeddown_fraction_variations_niter):
                #setup_histogram(his, get_colour(i + 2), 1)
                #leg_fd.AddEntry(his, self.powheg_nonprompt_varlabels[i], "L")
                #his.Draw("samehist")
            #setup_tgraph(tg_feeddown_fraction[ibin2], get_colour(1))
            #h_feeddown_fraction[ibin2].Draw("same")
            #h_feeddown_fraction[ibin2].Draw("axissame")
            #leg_fd.Draw("same")
            ##PREL latex = TLatex(0.18, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            #latex = TLatex(0.18, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
            #draw_latex(latex)
            #latex1 = TLatex(0.18, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4, #left|#it{#eta}_{jet}#right| #leq 0.5" % self.p_latexnhadron)
            #draw_latex(latex1)
            #latex2 = TLatex(0.18, 0.72, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            #draw_latex(latex2)
            ##latex3 = TLatex(0.18, 0.7, "%g #leq %s < %g" % (round(self.lvarshape_binmin_reco[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_reco[-1], 2)))
            #latex3 = TLatex(0.18, 0.67, "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2])))
            #draw_latex(latex3)
            #latex5 = TLatex(0.18, 0.62, "stat. unc. from data")
            #draw_latex(latex5)
            #latex6 = TLatex(0.18, 0.57, "syst. unc. from POWHEG #plus PYTHIA 6")
            #draw_latex(latex6)
            ##latex7 = TLatex(0.65, 0.75, "POWHEG based")
            ##draw_latex(latex7)
            #cfeeddown_fraction.SaveAs("%s/feeddown_fraction_var_%s.eps" % (self.d_resultsallpdata, suffix_plot))
            #cfeeddown_fraction.SaveAs("%s/feeddown_fraction_var_%s.pdf" % (self.d_resultsallpdata, suffix_plot))
