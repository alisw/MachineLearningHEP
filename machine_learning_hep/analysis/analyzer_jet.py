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
from machine_learning_hep.processerdhadrons_jet import adjust_nsd, adjust_z

def shrink_err_x(graph, width=0.1):
    for i in range(graph.GetN()):
        graph.SetPointEXlow(i, width)
        graph.SetPointEXhigh(i, width)

# pylint: disable=too-many-instance-attributes, too-many-statements
class AnalyzerJet(Analyzer):
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        # machine learning
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]

        # normalisation
        self.p_nevents = 1 # number of selected events, taken from histonorm
        self.branching_ratio = \
            datap["analysis"][self.typean].get("branching_ratio", None)
        self.xsection_inel = \
            datap["analysis"][self.typean].get("xsection_inel", None)

        # selection
        self.s_jetsel_sim = datap["analysis"][self.typean]["jetsel_sim"] # simulations

        # plotting
        # LaTeX string
        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]
        self.p_latexndecay = datap["analysis"][self.typean]["latexnamedecay"]
        self.p_latexbin2var = datap["analysis"][self.typean]["latexbin2var"]
        self.v_pth_latex = "#it{p}_{T}^{%s}" % self.p_latexnhadron
        self.v_varshape_latex = datap["analysis"][self.typean]["var_shape_latex"]

        # first variable (hadron pt)
        self.v_var_binning = datap["var_binning"] # name
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin) # number of bins
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.var1ranges = self.lpt_finbinmin.copy()
        self.var1ranges.append(self.lpt_finbinmax[-1])
        self.var1binarray = array("d", self.var1ranges) # array of bin edges to use in histogram constructors

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

        # fitting
        self.p_sgnfunc = datap["analysis"][self.typean]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"][self.typean]["bkgfunc"]
        self.p_masspeak = datap["analysis"][self.typean]["masspeak"]
        self.p_massmin = datap["analysis"][self.typean]["massmin"]
        self.p_massmax = datap["analysis"][self.typean]["massmax"]
        self.p_rebin = datap["analysis"][self.typean]["rebin"]
        self.p_fix_mean = datap["analysis"][self.typean]["fix_mean"]
        self.p_fix_sigma = datap["analysis"][self.typean]["fix_sigma"]
        self.p_sigmaarray = datap["analysis"][self.typean]["sigmaarray"]
        #self.p_masspeaksec = None
        self.p_fix_sigmasec = None
        self.p_sigmaarraysec = None
        if self.p_sgnfunc[0] == 1:
            #self.p_masspeaksec = datap["analysis"][self.typean]["masspeaksec"]
            self.p_fix_sigmasec = datap["analysis"][self.typean]["fix_sigmasec"]
            self.p_sigmaarraysec = datap["analysis"][self.typean]["sigmaarraysec"]

        # sideband subtraction
        self.signal_sigma = \
            datap["analysis"][self.typean].get("signal_sigma", None)
        self.sideband_sigma_1_left = \
            datap["analysis"][self.typean].get("sideband_sigma_1_left", None)
        self.sideband_sigma_1_right = \
            datap["analysis"][self.typean].get("sideband_sigma_1_right", None)
        self.sideband_sigma_2_left = \
            datap["analysis"][self.typean].get("sideband_sigma_2_left", None)
        self.sideband_sigma_2_right = \
            datap["analysis"][self.typean].get("sideband_sigma_2_right", None)
        self.sigma_scale = \
            datap["analysis"][self.typean].get("sigma_scale", None)
        self.sidebandleftonly = \
            datap["analysis"][self.typean].get("sidebandleftonly", None)

        # feed-down
        self.powheg_path_nonprompt = \
            datap["analysis"][self.typean].get("powheg_path_nonprompt", None)
        # systematics variations

        # models to compare with
        # POWHEG + PYTHIA 6
        self.powheg_path_prompt = \
            datap["analysis"][self.typean].get("powheg_path_prompt", None)
        self.powheg_prompt_variations = \
            datap["analysis"][self.typean].get("powheg_prompt_variations", None)
        self.powheg_prompt_variations_path = \
            datap["analysis"][self.typean].get("powheg_prompt_variations_path", None)
        # PYTHIA 8
        self.pythia8_prompt_variations_path = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations_path", None)
        self.pythia8_prompt_variations = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations", None)
        self.pythia8_prompt_variations_legend = \
            datap["analysis"][self.typean].get("pythia8_prompt_variations_legend", None)

        # unfolding
        self.niter_unfolding = \
            datap["analysis"][self.typean].get("niterunfolding", None)
        self.choice_iter_unfolding = \
            datap["analysis"][self.typean].get("niterunfoldingchosen", None)

        # systematics
        # import parameters of variations from the variation database
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
                        if catname == "feeddown":
                            self.powheg_nonprompt_varnames.append(varname_i)
                            self.powheg_nonprompt_varlabels.append(varlabel_i)
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
        # self.feeddown_db = datap["multi"]["feeddown_db"]
        self.feeddown_ratio = datap["multi"]["feeddown_ratio"]
        # if self.feeddown_db:
        #    self.d_resultsold =  datap["analysis"][typean]["data"]["resultsold"]
        if self.feeddown_ratio:
            self.d_resultslc =  datap["analysis"][typean]["data"]["resultslc"]


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
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_fileeff = os.path.join(self.d_resultsallpmc_proc, self.n_fileeff)
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.n_fileresp = os.path.join(self.d_resultsallpmc_proc, self.n_fileresp)

        # output files
        self.file_yields = os.path.join(self.d_resultsallpdata, "yields.root")
        self.file_efficiency = os.path.join(self.d_resultsallpmc, "efficiencies.root")
        self.file_sideband = os.path.join(self.d_resultsallpdata, "sideband_subtracted.root")
        self.file_feeddown = os.path.join(self.d_resultsallpdata, "feeddown.root")
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
        self.text_jets = "%s-tagged charged jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.p_latexnhadron
        self.text_ptjet = "%g #leq %s < %g GeV/#it{c}, #left|#it{#eta}_{jet}#right| #leq 0.5"
        self.text_pth = "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, #left|#it{y}_{%s}#right| #leq 0.8"
        self.text_sd = "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"
        self.text_acc_h = "#left|#it{y}_{%s}#right| #leq 0.8" % self.p_latexnhadron
        self.text_powheg = "POWHEG + PYTHIA 6 + EvtGen"


    #def makeratio_onedim(self, origin_histo, option, histo_to_compare, xtitle, ytitle):
    #    filename = self.d_resultsold + "/" + option + ".root"
    #    print("Open file with results to compare", filename)
    #    c_ratio = TCanvas("c_ratio", "histos ratio")
    #    setup_canvas(c_ratio)
    #    c_ratio.Divide(2,2)
    #    c_ratio.cd(1)
    #    myfild_old = TFile.Open(filename)
    #    first_histo = origin_histo.Clone("first_histo")
    #    second_histo = myfild_old.Get(histo_to_compare)
    #    if not second_histo:
    #        print("No old histo!", histo_to_compare)
    #    else:
    #        print(histo_to_compare)
    #        leg_ratio = TLegend(.6, .8, .8, .85)
    #        setup_legend(leg_ratio)
    #        setup_histogram(first_histo, get_colour(1), get_marker(0))
    #        setup_histogram(second_histo, get_colour(2), get_marker(1))
    #        leg_ratio.AddEntry(second_histo, "old_data %s" %option, "P")
    #        leg_ratio.AddEntry(first_histo, "new_data %s" %option, "P")
    #        #second_histo.SetYTitle(ytitle)
    #        second_histo.SetTitle(histo_to_compare)
    #        second_histo.Draw()
    #        first_histo.Draw("same")
    #        leg_ratio.Draw("same")
    #        scaled_1 = first_histo.Clone("scaled_1")
    #        scaled_2 = second_histo.Clone("scaled_2")
    #        c_ratio.cd(3)
    #        ratio = second_histo.Clone("ratio")
    #        ratio.Divide(first_histo)
    #        ratio.SetTitle("old data to new data ratio")
    #        setup_histogram(ratio, get_colour(0))
    #        ratio.Draw()
    #        if ((scaled_1.Integral()!=0) and (scaled_2.Integral()!=0)):
    #            scaled_2.Scale(1/scaled_2.Integral())
    #            scaled_1.Scale(1/scaled_1.Integral())
    #            #scaled_1.SetXTitle(xtitle)
    #            #scaled_2.SetYTitle(ytitle)
    #            c_ratio.cd(2)
    #            scaled_2.SetTitle("self normalized")
    #            scaled_2.Draw()
    #            scaled_1.Draw("same")
    #            leg_ratio.Draw("same")
    #            c_ratio.cd(4)
    #            norm_ratio = scaled_2.Clone("ratio")
    #            norm_ratio.Divide(scaled_1)
    #            norm_ratio.SetTitle("old data to new data (norm) ratio")
    #            setup_histogram(norm_ratio, get_colour(0))
    #            norm_ratio.Draw()
    #            c_ratio.SaveAs("%s/old_new_%s.png" % (self.d_resultsallpdata, histo_to_compare))
    #            #c_ratio.SaveAs("compare/old_new_%s.png" % (histo_to_compare))
    #        myfild_old.Close()

    #def makeratio_twodim(self, origin_histo, option, histo_to_compare):
    #    filename = self.d_resultsold + "/" + option + ".root"
    #    print("Open file with results to compare", filename)
    #    myfild_old = TFile.Open(filename)
    #    first_histo = origin_histo.Clone("first_histo")
    #    second_histo = myfild_old.Get(histo_to_compare)
    #    if not second_histo:
    #        print("No old histo!", histo_to_compare)
    #    else:
    #        print(histo_to_compare)
    #        c_ratio = TCanvas("c_ratio", "histos ratio")
    #        setup_canvas(c_ratio)
    #        leg_ratio = TLegend(.6, .8, .8, .85)
    #        setup_legend(leg_ratio)
    #        sub1 = first_histo.Clone("sub1")
    #        sub1.Add(second_histo, -1)
    #        setup_histogram(sub1)
    #        sub1.SetTitle("%s %s" % (histo_to_compare, option))
    #        sub1.Draw("text")
    #        c_ratio.SaveAs("%s/new-old_%s.png" % (self.d_resultsallpdata , histo_to_compare))
    #        #c_ratio.SaveAs("compare/new-old_diff_%s.png" % (histo_to_compare))
    #        second_histo.Scale(1/second_histo.Integral())
    #        first_histo.Scale(1/first_histo.Integral())
    #        second_histo.Divide(first_histo)
    #        second_histo.SetTitle("%s %s" % (histo_to_compare, option))
    #        second_histo.Draw("text")
    #        c_ratio.SaveAs("%s/old_new_ratio_%s.png" % (self.d_resultsallpdata, histo_to_compare))
    #        #c_ratio.SaveAs("compare/new_old_ratio_%s.png" % (histo_to_compare))
    #    myfild_old.Close()

    def makeratio(self, origin_histo, option, lc_histoname):
        D0_histo = origin_histo.Clone("D0_histo")
        filename = self.d_resultslc + "/" + option + ".root"
        print("Open file with Lc results", filename)
        myfilelc = TFile.Open(filename)
        lc_histo = myfilelc.Get(lc_histoname)
        Lc_histo = lc_histo.Clone("Lc_histo")
        c_ratio = TCanvas("c_ratio", "Lc to D0 ratio")
        setup_canvas(c_ratio)
        leg_ratio = TLegend(.6, .8, .8, .85)
        setup_legend(leg_ratio)
        D0_histo.SetTitle("")
        Lc_histo.SetXTitle("%s" % self.v_varshape_latex)
        Lc_histo.SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
        Lc_histo.SetTitle("")
        setup_histogram(D0_histo, get_colour(1), get_marker(0))
        setup_histogram(Lc_histo, get_colour(2), get_marker(1))
        D0_histo.Draw()
        Lc_histo.Draw("same")
        leg_ratio.AddEntry(Lc_histo, "Lc %s" %option, "P")
        leg_ratio.AddEntry(D0_histo, "D0 %s" %option, "P")
        leg_ratio.Draw("same")
        c_ratio.SaveAs("%s/Lc+D0_combined_plot_%s.eps" % (self.d_resultsallpdata , lc_histoname))
        c_ratio.SaveAs("Lc+D0_combined_plot_%s.png" % (lc_histoname))
        Lc_histo_sc = Lc_histo.Clone("Lc_histo_sc")
        D0_histo_sc = D0_histo.Clone("D0_histo_sc")
        Lc_histo_sc.Scale(1/Lc_histo_sc.Integral())
        D0_histo_sc.Scale(1/D0_histo_sc.Integral())
        Lc_histo_sc.Divide(D0_histo_sc)
        Lc_histo_sc.SetYTitle("{\Lambda}_{c} / {D}_{0}  ratio")
        Lc_histo_sc.Draw()
        c_ratio.SaveAs("%s/Lc_D0_ratio_%s.eps" % (self.d_resultsallpdata, lc_histoname))
        c_ratio.SaveAs("Lc_D0_ratio_%s.png" % (lc_histoname))
        del Lc_histo
        del D0_histo
        del lc_histo
        del Lc_histo_sc
        del D0_histo_sc
        myfilelc.Close()

    def fit(self):
        self.loadstyle()
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)
        fileout = TFile.Open(self.file_yields, "recreate")
        if not fileout:
            self.logger.fatal(make_message_notfound(self.file_yields))
        myfilemc = TFile.Open(self.n_filemass_mc)
        if not myfilemc:
            self.logger.fatal(make_message_notfound(self.n_filemass_mc))
        myfile = TFile.Open(self.n_filemass)
        if not myfile:
            self.logger.fatal(make_message_notfound(self.n_filemass))

        # Get the number of selected events.
        histonorm = myfile.Get("histonorm")
        if not histonorm:
            self.logger.fatal(make_message_notfound("histonorm", self.n_filemass))
        self.p_nevents = histonorm.GetBinContent(1)
        print("Number of selected event: %g" % self.p_nevents)
        #if self.feeddown_db:
        #    option = "masshisto"
        #    histo_to_compare = ("histonorm")
        #    print("Making ratio for", option, histo_to_compare)
        #    xtitle = ""
        #    ytitle = ""
        #    self.makeratio_onedim(histonorm, option, histo_to_compare, xtitle, ytitle)

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            for ibin2 in range(self.p_nbin2_reco):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                          self.lvar2_binmax_reco[ibin2])
                suffix_plot = "%s_%g_%g_%s_%g_%g" % \
                    (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2],
                     self.v_var_binning, self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                histomassmc = myfilemc.Get("hmass_sig" + suffix)
                if not histomassmc:
                    self.logger.fatal(make_message_notfound("hmass_sig" + suffix, self.n_filemass_mc))
                histomassmc_reb = AliVertexingHFUtils.RebinHisto(histomassmc, \
                                            self.p_rebin[ipt], -1)
                histomassmc_reb_f = TH1F()
                histomassmc_reb.Copy(histomassmc_reb_f)
                fittermc = AliHFInvMassFitter(histomassmc_reb_f, \
                    self.p_massmin[ipt], self.p_massmax[ipt], self.p_bkgfunc[ipt], 0)
                fittermc.SetInitialGaussianMean(self.p_masspeak)
                out = fittermc.MassFitter(0)
                print("I have made MC fit for sigma initialization, status: %d" % out)
                histomass = myfile.Get("hmass" + suffix)

                #if self.feeddown_db:
                #    option = "masshisto"
                #    histo_to_compare = ("hmass%s" % (suffix))
                #    print("Making ratio for", option, histo_to_compare)
                #    xtitle = "Inv mass, GeV/c"
                #    ytitle = ""
                #    self.makeratio_onedim(histomass, option, histo_to_compare, xtitle, ytitle )

                if not histomass:
                    self.logger.fatal(make_message_notfound("hmass" + suffix, self.n_filemass))
                histomass_reb = AliVertexingHFUtils.RebinHisto(histomass, \
                                            self.p_rebin[ipt], -1)
                histomass_reb_f = TH1F()
                histomass_reb.Copy(histomass_reb_f)
                fitter = AliHFInvMassFitter(histomass_reb_f, self.p_massmin[ipt], \
                    self.p_massmax[ipt], self.p_bkgfunc[ipt], self.p_sgnfunc[ipt])
                fitter.SetInitialGaussianSigma(fittermc.GetSigma())
                fitter.SetInitialGaussianMean(fittermc.GetMean())
                if self.p_fix_sigma[ipt] is True:
                    fitter.SetFixGaussianSigma(fittermc.GetSigma())
                if self.p_sgnfunc[ipt] == 1:
                    if self.p_fix_sigmasec[ipt] is True:
                        fitter.SetFixSecondGaussianSigma(self.p_sigmaarraysec[ipt])
                out = fitter.MassFitter(0)
                fit_dir = fileout.mkdir(suffix)
                fit_dir.WriteObject(fitter, "fitter%d" % (ipt))
                bkg_func = fitter.GetBackgroundRecalcFunc()
                sgn_func = fitter.GetMassFunc()

                c_fitted_result = TCanvas("c_fitted_result " + suffix, "Fitted Result")
                setup_canvas(c_fitted_result)
                setup_histogram(histomass_reb, get_colour(0), get_marker(0))
                histomass_reb.SetTitle("")
                histomass_reb.SetXTitle("invariant mass (GeV/#it{c}^{2})")
                histomass_reb.SetYTitle("counts")
                histomass_reb.SetTitleOffset(1.2, "Y")
                histomass_reb.GetYaxis().SetMaxDigits(3)
                y_min_h, y_max_h = get_y_window_his(histomass_reb)
                y_margin_up = 0.15
                y_margin_down = 0.05
                histomass_reb.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
                histomass_reb.Draw("same")
                if out == 1:
                    bkg_func.SetLineColor(get_colour(1))
                    sgn_func.SetLineColor(get_colour(2))
                    sgn_func.Draw("same")
                    bkg_func.Draw("same")
                latex = TLatex(0.2, 0.83, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.5, 0.83, "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" % \
                    (self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt], self.lvar2_binmax_reco[ibin2])))
                draw_latex(latex2)
                c_fitted_result.SaveAs("%s/fit_%s.eps" % (self.d_resultsallpdata, suffix_plot))
        myfilemc.Close()
        myfile.Close()
        fileout.Close()
        gROOT.SetBatch(tmp_is_root_batch)

    def efficiency(self): # pylint: disable=too-many-branches, too-many-locals
        self.loadstyle()
        lfileeff = TFile.Open(self.n_fileeff)
        if not lfileeff:
            self.logger.fatal(make_message_notfound(self.n_fileeff))
        lfileresp = TFile.Open(self.n_fileresp)
        if not lfileresp:
            self.logger.fatal(make_message_notfound(self.n_fileresp))
        fileouteff = TFile.Open(self.file_efficiency, "recreate")
        if not fileouteff:
            self.logger.fatal(make_message_notfound(self.file_efficiency))

        # FIXME pylint: disable=fixme
        # calculate prompt and non-prompt efficiency at rec. level using rec. jet pt and shape in numerator and denominator
        # calculate prompt and non-prompt efficiency at gen. level using gen. jet pt and shape in numerator and denominator
        # restrict the shape range at both levels

        string_shaperange = "%g #leq %s < %g" % (self.lvarshape_binmin_reco[0], self.v_varshape_latex, self.lvarshape_binmax_reco[-1])

        # PROMPT EFFICIENCY

        # rec. level cuts only applied
        hzvsjetpt_reco_nocuts = lfileresp.Get("hzvsjetpt_reco_nocuts")
        if not hzvsjetpt_reco_nocuts:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco_nocuts", self.n_fileresp))
        # rec. level and gen. level cuts applied
        hzvsjetpt_reco_eff = lfileresp.Get("hzvsjetpt_reco_cuts")
        if not hzvsjetpt_reco_eff:
            self.logger.fatal(make_message_notfound("hzvsjetpt_reco_cuts", self.n_fileresp))
        # calculate rec. level kinematic efficiency and apply it to the unfolding input
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)

        # gen. level cuts only applied
        hzvsjetpt_gen_nocuts = lfileresp.Get("hzvsjetpt_gen_nocuts")
        if not hzvsjetpt_gen_nocuts:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen_nocuts", self.n_fileresp))
        # rec. level and gen. level cuts applied
        hzvsjetpt_gen_eff = lfileresp.Get("hzvsjetpt_gen_cuts")
        if not hzvsjetpt_gen_eff:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen_cuts", self.n_fileresp))
        # calculate gen. level kinematic efficiency
        hzvsjetpt_gen_eff.Divide(hzvsjetpt_gen_nocuts)

        # get pt_cand vs pt_jet histogram of matched rec level jets (from processor output)
        # with overflow entries
        hisname_overflow = "h2_ptcand_ptjet_rec_overflow"
        h2_ptcand_ptjet_rec_overflow = lfileeff.Get(hisname_overflow)
        if not h2_ptcand_ptjet_rec_overflow:
            self.logger.fatal(make_message_notfound(hisname_overflow, self.n_fileeff))
        # without overflow entries
        hisname = "h2_ptcand_ptjet_rec"
        h2_ptcand_ptjet_rec = lfileeff.Get(hisname)
        if not h2_ptcand_ptjet_rec:
            self.logger.fatal(make_message_notfound(hisname, self.n_fileeff))

        # pt_cand vs pt_gen_jet of matched gen level jets
        # with overflow entries
        hisname = "h2_ptcand_ptjet_genmatched_overflow"
        h2_ptcand_ptjet_genmatched_overflow = lfileeff.Get(hisname)
        if not h2_ptcand_ptjet_genmatched_overflow:
            self.logger.fatal(make_message_notfound(hisname, self.n_fileeff))
        # without overflow entries
        hisname = "h2_ptcand_ptjet_genmatched"
        h2_ptcand_ptjet_genmatched = lfileeff.Get(hisname)
        if not h2_ptcand_ptjet_genmatched:
            self.logger.fatal(make_message_notfound(hisname, self.n_fileeff))

        # get shape_gen vs pt_gen_jet vs pt_cand histogram of selected gen level jets (from processor output)
        # with overflow entries
        hisname_overflow = "hzvsjetptvscandpt_gen_prompt"
        h3_z_ptjet_ptcand_gen = lfileresp.Get(hisname_overflow)
        if not h3_z_ptjet_ptcand_gen:
            self.logger.fatal(make_message_notfound(hisname_overflow, self.n_fileresp))
        # without overflow entries
        #hisname = "h3_shape_ptjet_ptcand_gen"
        #h3_z_ptjet_ptcand_gen = lfileeff.Get(hisname)
        #if not h3_z_ptjet_ptcand_gen:
        #    self.logger.fatal(make_message_notfound(hisname, self.n_fileeff))

        # make an empty pt_cand vs pt_jet histogram for folded gen level jets = gen_folded
        h2_ptcand_ptjet_gen_overflow_folded = buildhisto("h2_ptcand_ptjet_gen_overflow_folded", "h2_ptcand_ptjet_gen_overflow_folded", self.var1binarray, self.var2binarray_reco)
        h2_ptcand_ptjet_gen_folded = buildhisto("h2_ptcand_ptjet_gen_folded", "h2_ptcand_ptjet_gen_folded", self.var1binarray, self.var2binarray_reco)

        # get prompt 4D response matrix shape vs pt_jet (from processor output)
        hisname = "response_matrix"
        response_matrix = lfileresp.Get(hisname)
        if not response_matrix:
            self.logger.fatal(make_message_notfound(hisname, self.n_fileresp))

        # create template for the folded gen
        output_template = buildhisto("h2_z_jetpt", "h2_z_jetpt", self.varshapebinarray_reco, self.var2binarray_reco)

        list_ptcand_genmatched_new_overflow = []
        list_ptcand_genmatched_new = []
        list_ptcand_rec_new_overflow = []
        list_ptcand_rec_new = []
        list_ptcand_gen_new_overflow = []
        list_ptcand_gen_new_overflow_folded = []
        list_ptcand_gen_new = []
        list_ptcand_gen_new_folded = []
        list_ptcand_eff_new_overflow = []
        list_ptcand_eff_new_overflow_folded = []
        list_ptcand_eff_new = []
        list_ptcand_eff_new_folded = []
        list_ptcand_effgen_new_overflow = []
        list_ptcand_effgen_new = []

        # calculate gen, rec, eff using the new method without folding
        for ibin2 in range(self.p_nbin2_reco):
            # restrict pt_jet range
            h3_z_ptjet_ptcand_gen.GetYaxis().SetRange(ibin2 + 1, ibin2 + 1)

            # shape overflow projection
            h1_ptcand_rec_overflow = h2_ptcand_ptjet_rec_overflow.ProjectionX("h1_ptcand_rec_overflow_%d" % ibin2, ibin2 + 1, ibin2 + 1)
            list_ptcand_rec_new_overflow.append(h1_ptcand_rec_overflow)
            h1_ptcand_gen_overflow = h3_z_ptjet_ptcand_gen.Project3D("h1_ptcand_gen_%d_overflow_ze" % ibin2)
            list_ptcand_gen_new_overflow.append(h1_ptcand_gen_overflow)
            # rec level efficiency
            h1_ptcand_eff_overflow = h1_ptcand_rec_overflow.Clone("h1_ptcand_eff_overflow_%d" % ibin2)
            h1_ptcand_eff_overflow.Divide(h1_ptcand_gen_overflow)
            list_ptcand_eff_new_overflow.append(h1_ptcand_eff_overflow)
            # gen level efficiency
            h1_ptcand_genmatched_overflow = h2_ptcand_ptjet_genmatched_overflow.ProjectionX("h1_ptcand_genmatched_overflow_%d" % ibin2, ibin2 + 1, ibin2 + 1)
            list_ptcand_genmatched_new_overflow.append(h1_ptcand_genmatched_overflow)
            h1_ptcand_effgen_overflow = h1_ptcand_genmatched_overflow.Clone("h1_ptcand_effgen_overflow_%d" % ibin2)
            h1_ptcand_effgen_overflow.Divide(h1_ptcand_gen_overflow)
            list_ptcand_effgen_new_overflow.append(h1_ptcand_effgen_overflow)

            # restrict shape range
            h1_ptcand_rec = h2_ptcand_ptjet_rec.ProjectionX("h1_ptcand_rec_%d" % ibin2, ibin2 + 1, ibin2 + 1)
            list_ptcand_rec_new.append(h1_ptcand_rec)
            h3_z_ptjet_ptcand_gen.GetXaxis().SetRange(1, h3_z_ptjet_ptcand_gen.GetXaxis().GetNbins())
            h1_ptcand_gen = h3_z_ptjet_ptcand_gen.Project3D("h1_ptcand_gen_%d_ze" % ibin2)
            list_ptcand_gen_new.append(h1_ptcand_gen)
            # rec level efficiency
            h1_ptcand_eff = h1_ptcand_rec.Clone("h1_ptcand_eff_%d" % ibin2)
            h1_ptcand_eff.Divide(h1_ptcand_gen)
            list_ptcand_eff_new.append(h1_ptcand_eff)
            # gen level efficiency
            h1_ptcand_genmatched = h2_ptcand_ptjet_genmatched.ProjectionX("h1_ptcand_genmatched_%d" % ibin2, ibin2 + 1, ibin2 + 1)
            list_ptcand_genmatched_new.append(h1_ptcand_genmatched)
            h1_ptcand_effgen = h1_ptcand_genmatched.Clone("h1_ptcand_effgen_%d" % ibin2)
            h1_ptcand_effgen.Divide(h1_ptcand_gen)
            list_ptcand_effgen_new.append(h1_ptcand_effgen)

            # reset ranges
            h3_z_ptjet_ptcand_gen.GetYaxis().SetRange() # reset full pt_jet range
            h3_z_ptjet_ptcand_gen.GetXaxis().SetRange() # reset full shape range

        # for each pt_cand bin: fold the gen. level (shape, pt_jet) distribution and fill the (pt_cand, pt_jet) rec. level distribution
        for ipt in range(self.p_nptfinbins):
            # pt_jet projection of efficiency numerator
            # overflow projection
            h1_ptjet_rec_overflow = h2_ptcand_ptjet_rec_overflow.ProjectionY("h1_ptjet_rec_overflow_%d" % ipt, ipt + 1, ipt + 1, "e")
            # restricted projection
            h1_ptjet_rec = h2_ptcand_ptjet_rec.ProjectionY("h1_ptjet_rec_%d" % ipt, ipt + 1, ipt + 1, "e")

            # get the shape_gen vs pt_gen_jet projection for the given pt_cand bin
            h3_z_ptjet_ptcand_gen.GetZaxis().SetRange(ipt + 1, ipt + 1)
            # overflow projection
            h2_z_ptjet_gen_overflow_orig = h3_z_ptjet_ptcand_gen.Project3D("h2_shape_jetpt_ptcand_%d_gen_overflow_yxe" % ipt)
            h2_z_ptjet_gen_overflow = h2_z_ptjet_gen_overflow_orig.Clone(h2_z_ptjet_gen_overflow_orig.GetName() + "_clone")
            # restricted projection
            h3_z_ptjet_ptcand_gen.GetXaxis().SetRange(1, h3_z_ptjet_ptcand_gen.GetXaxis().GetNbins()) # restrict shape range
            h2_z_ptjet_gen_orig = h3_z_ptjet_ptcand_gen.Project3D("h2_shape_jetpt_ptcand_%d_gen_yxe" % ipt)
            h2_z_ptjet_gen = h2_z_ptjet_gen_orig.Clone(h2_z_ptjet_gen_orig.GetName() + "_clone")
            h3_z_ptjet_ptcand_gen.GetXaxis().SetRange() # reset shape range

            # apply gen. level kinematic efficiency
            h2_z_ptjet_gen_overflow.Multiply(hzvsjetpt_gen_eff)
            h2_z_ptjet_gen.Multiply(hzvsjetpt_gen_eff)

            # fold shape_gen vs pt_gen_jet with the prompt response matrix to get corresponding rec level shape vs pt_jet distribution
            h2_z_ptjet_gen_overflow_folded = folding(h2_z_ptjet_gen_overflow, response_matrix, output_template)
            h2_z_ptjet_gen_folded = folding(h2_z_ptjet_gen, response_matrix, output_template)

            # apply rec. level kinematic efficiency
            h2_z_ptjet_gen_overflow_folded.Divide(hzvsjetpt_reco_eff)
            h2_z_ptjet_gen_folded.Divide(hzvsjetpt_reco_eff)

            # sum up folded shape_rec bins and get pt_jet_rec projection
            # get original and folded pt_jet distribution of gen level jets
            # overflow projection
            h1_ptjet_gen_overflow_orig = h2_z_ptjet_gen_overflow_orig.ProjectionY("h1_ptjet_gen_overflow_orig_%d" % ipt, 0, self.p_nbinshape_gen + 1, "e")
            h1_ptjet_gen_overflow_folded = h2_z_ptjet_gen_overflow_folded.ProjectionY("h1_ptjet_gen_overflow_folded_%d" % ipt, 0, self.p_nbinshape_reco + 1, "e")
            # restricted projection
            h1_ptjet_gen_orig = h2_z_ptjet_gen_orig.ProjectionY("h1_ptjet_gen_orig_%d" % ipt, 1, self.p_nbinshape_gen, "e")
            h1_ptjet_gen_folded = h2_z_ptjet_gen_folded.ProjectionY("h1_ptjet_gen_folded_%d" % ipt, 1, self.p_nbinshape_reco, "e")

            # compare pt_jet projections
            latex = TLatex(0.13, 0.85, "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" % (self.lpt_finbinmin[ipt], self.p_latexnhadron, self.lpt_finbinmax[ipt]))
            latex_shaperange = TLatex(0.13, 0.8, string_shaperange)
            list_obj = [h1_ptjet_rec_overflow, h1_ptjet_rec, h1_ptjet_gen_overflow_orig, h1_ptjet_gen_overflow_folded, h1_ptjet_gen_orig, h1_ptjet_gen_folded, \
                latex, latex_shaperange]
            labels_obj = [ \
                "rec %s overflow" % self.v_varshape_latex, \
                "rec", \
                "gen %s overflow" % self.v_varshape_latex, \
                "gen %s overflow folded" % self.v_varshape_latex, \
                "gen", \
                "gen folded"]
            make_plot("efficiency_pr_ptcand_%d" % ipt, path=self.d_resultsallpdata, list_obj=list_obj, labels_obj=labels_obj, \
                title="new efficiency calculation;%s (GeV/#it{c});count" % self.p_latexbin2var, \
                leg_pos=[0.55, 0.72, 0.8, 0.86], logscale="y", margins_y=[0.05, 0.25], \
                colours=[get_colour(i) for i in (1, 3, 1, 2, 3, 4)], markers=[get_marker(i) for i in (0, 0, 1, 2, 3, 4)])

            # fill the (pt_cand, pt_jet) bins of gen_folded
            for ibin2 in range(self.p_nbin2_reco):
                # overflow folded
                h2_ptcand_ptjet_gen_overflow_folded.SetBinContent(ipt + 1, ibin2 + 1, h1_ptjet_gen_overflow_folded.GetBinContent(ibin2 + 1))
                h2_ptcand_ptjet_gen_overflow_folded.SetBinError(ipt + 1, ibin2 + 1, h1_ptjet_gen_overflow_folded.GetBinError(ibin2 + 1))
                # restricted folded
                h2_ptcand_ptjet_gen_folded.SetBinContent(ipt + 1, ibin2 + 1, h1_ptjet_gen_folded.GetBinContent(ibin2 + 1))
                h2_ptcand_ptjet_gen_folded.SetBinError(ipt + 1, ibin2 + 1, h1_ptjet_gen_folded.GetBinError(ibin2 + 1))

        # calculate the efficieny (pt_cand, pt_jet) as eff = rec_matched/gen_folded
        # overflow folded
        h2_ptcand_ptjet_eff_overflow_folded = h2_ptcand_ptjet_rec_overflow.Clone("h2_ptcand_ptjet_eff_overflow_folded")
        h2_ptcand_ptjet_eff_overflow_folded.Divide(h2_ptcand_ptjet_gen_overflow_folded)
        # restricted folded
        h2_ptcand_ptjet_eff_folded = h2_ptcand_ptjet_rec.Clone("h2_ptcand_ptjet_eff_folded")
        h2_ptcand_ptjet_eff_folded.Divide(h2_ptcand_ptjet_gen_folded)

        # make pt_cand projections of folded gen and eff
        for ibin2 in range(self.p_nbin2_reco):
            # overflow folded
            h1_ptcand_gen_overflow_folded = h2_ptcand_ptjet_gen_overflow_folded.ProjectionX("h1_ptcand_gen_overflow_folded_%d" % ibin2, ibin2 + 1, ibin2 + 1, "e")
            list_ptcand_gen_new_overflow_folded.append(h1_ptcand_gen_overflow_folded)
            h1_ptcand_eff_overflow_folded = h2_ptcand_ptjet_eff_overflow_folded.ProjectionX("h1_ptcand_eff_overflow_folded_%d" % ibin2, ibin2 + 1, ibin2 + 1, "e")
            list_ptcand_eff_new_overflow_folded.append(h1_ptcand_eff_overflow_folded)
            # restricted folded
            h1_ptcand_gen_folded = h2_ptcand_ptjet_gen_folded.ProjectionX("h1_ptcand_gen_folded_%d" % ibin2, ibin2 + 1, ibin2 + 1, "e")
            list_ptcand_gen_new_folded.append(h1_ptcand_gen_folded)
            h1_ptcand_eff_folded = h2_ptcand_ptjet_eff_folded.ProjectionX("h1_ptcand_eff_folded_%d" % ibin2, ibin2 + 1, ibin2 + 1, "e")
            list_ptcand_eff_new_folded.append(h1_ptcand_eff_folded)

        # The old (wrong) method

        cEff = TCanvas("cEff", "The Fit Canvas")
        setup_canvas(cEff)
        legeff = TLegend(.13, .65, .5, .88)
        setup_legend(legeff)
        list_his = []
        list_ptcand_gen_old = []
        list_ptcand_rec_old = []
        list_ptcand_eff_old = []
        for ibin2 in range(self.p_nbin2_reco):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                            self.lvar2_binmin_reco[ibin2], \
                                            self.lvar2_binmax_reco[ibin2])
            h_gen_pr = lfileeff.Get("h_gen_pr" + stringbin2)
            h_gen_pr_old = h_gen_pr.Clone(h_gen_pr.GetName() + "_old")
            list_ptcand_gen_old.append(h_gen_pr_old)

            h_sel_pr = lfileeff.Get("h_sel_pr" + stringbin2)
            h_sel_pr_old = h_sel_pr.Clone(h_sel_pr.GetName() + "_old")
            list_ptcand_rec_old.append(h_sel_pr_old)

            h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
            list_his.append(h_sel_pr)
            h_eff_pr_old = h_sel_pr.Clone("h_eff_pr" + stringbin2 + "_old")
            list_ptcand_eff_old.append(h_eff_pr_old)
        y_min_h, y_max_h = get_y_window_his(list_his)
        y_min_h = 0
        y_margin_up = 0.35
        y_margin_down = 0
        y_min, y_max = get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up)
        for ibin2 in range(self.p_nbin2_reco):
            h_sel_pr = list_his[ibin2]
            setup_histogram(h_sel_pr, get_colour(ibin2), 1)
            h_sel_pr.Draw("same")
            fileouteff.cd()
            h_sel_pr.SetName("eff_mult%d" % ibin2)
            h_sel_pr.Write()
            legeffstring = "%g #leq %s < %g GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var,
                     self.lvar2_binmax_reco[ibin2])
            legeff.AddEntry(h_sel_pr, legeffstring, "LE")
            h_sel_pr.SetTitle("")
            h_sel_pr.SetXTitle("#it{p}_{T}^{%s} (GeV/#it{c})" % self.p_latexnhadron)
            h_sel_pr.SetYTitle("prompt %s-jet efficiency" % self.p_latexnhadron)
            h_sel_pr.GetYaxis().SetRangeUser(y_min, y_max)
            h_sel_pr.SetTitleOffset(1.2, "Y")
            h_sel_pr.SetTitleOffset(1.1, "X")
        legeff.Draw()
        cEff.SaveAs("%s/efficiency_pr.eps" % self.d_resultsallpdata)

        # compare the old and the new method
        list_eff_diff = []
        list_eff_diff_labels = []
        list_effgen_diff = []
        list_effgen_diff_labels = []
        latex_shaperange = TLatex(0.13, 0.8, string_shaperange)
        for ibin2 in range(self.p_nbin2_reco):
            string_ptjet = "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2])
            latex = TLatex(.13, .85, string_ptjet)
            # compare rec for old, new x (w/ overflow, w/o overflow)
            make_plot("efficiency_pr_rec_%d" % ibin2, path=self.d_resultsallpdata, \
                colours=[get_colour(i) for i in (0, 1, 3, 5, 6)], markers=[get_marker(i) for i in (0, 1, 3, 5, 6)], \
                list_obj=[ \
                    list_ptcand_rec_old[ibin2], \
                    list_ptcand_rec_new_overflow[ibin2], \
                    list_ptcand_rec_new[ibin2], \
                    list_ptcand_genmatched_new_overflow[ibin2], \
                    list_ptcand_genmatched_new[ibin2], \
                    latex, \
                    latex_shaperange], \
                labels_obj=[ \
                    "rec old", \
                    "rec new %s overflow" % self.v_varshape_latex, \
                    "rec new", \
                    "gen %s overflow" % self.v_varshape_latex, \
                    "gen" \
                ], title="matched;#it{p}_{T}^{%s} (GeV/#it{c});counts" % self.p_latexnhadron, leg_pos=[0.55, 0.72, 0.8, 0.86], margins_y=[0.05, 0.25], logscale="y")
            # compare gen for old, new x (w/ overflow, w/o overflow) x (w/o folding, w/ folding)
            make_plot("efficiency_pr_gen_%d" % ibin2, path=self.d_resultsallpdata, \
                list_obj=[ \
                    list_ptcand_gen_old[ibin2], \
                    list_ptcand_gen_new_overflow[ibin2], \
                    list_ptcand_gen_new_overflow_folded[ibin2], \
                    list_ptcand_gen_new[ibin2], \
                    list_ptcand_gen_new_folded[ibin2], \
                    latex, \
                    latex_shaperange], \
                labels_obj=[ \
                    "old", \
                    "new %s overflow" % self.v_varshape_latex, \
                    "new %s overflow folded" % self.v_varshape_latex, \
                    "new", \
                    "new folded" \
                ], title="generator level;#it{p}_{T}^{%s} (GeV/#it{c});counts" % self.p_latexnhadron, leg_pos=[0.55, 0.72, 0.8, 0.86], margins_y=[0.05, 0.25], logscale="y")
            # compare eff for old, new x (w/ overflow, w/o overflow) x (w/o folding, w/ folding)
            make_plot("efficiency_pr_eff_%d" % ibin2, path=self.d_resultsallpdata, \
                list_obj=[ \
                    list_ptcand_eff_old[ibin2], \
                    list_ptcand_eff_new_overflow[ibin2], \
                    list_ptcand_eff_new_overflow_folded[ibin2], \
                    list_ptcand_eff_new[ibin2], \
                    list_ptcand_eff_new_folded[ibin2], \
                    latex, \
                    latex_shaperange], \
                labels_obj=[ \
                    "old", \
                    "new %s overflow" % self.v_varshape_latex, \
                    "new %s overflow folded" % self.v_varshape_latex, \
                    "new", \
                    "new folded" \
                ], title="reconstruction level efficiency;#it{p}_{T}^{%s} (GeV/#it{c});efficiency" % self.p_latexnhadron, leg_pos=[0.55, 0.72, 0.8, 0.86], margins_y=[0, 0.25])
            make_plot("efficiency_pr_effgen_%d" % ibin2, path=self.d_resultsallpdata, \
                colours=[get_colour(i) for i in (0, 5, 6)], markers=[get_marker(i) for i in (0, 5, 6)], \
                list_obj=[ \
                    list_ptcand_eff_old[ibin2], \
                    list_ptcand_effgen_new_overflow[ibin2], \
                    list_ptcand_effgen_new[ibin2], \
                    latex, \
                    latex_shaperange], \
                labels_obj=[ \
                    "old", \
                    "new %s overflow" % self.v_varshape_latex, \
                    "new", \
                ], title="generator level efficiency;#it{p}_{T}^{%s} (GeV/#it{c});efficiency" % self.p_latexnhadron, leg_pos=[0.55, 0.72, 0.8, 0.86], margins_y=[0, 0.25])
            # plot the error of old w.r.t. new folded
            eff_new = list_ptcand_eff_new_folded[ibin2].Clone("eff_pr_new_%d" % ibin2)
            eff_diff = list_ptcand_eff_old[ibin2].Clone("eff_pr_diff_%d" % ibin2)
            eff_diff.Add(eff_new, -1)
            eff_diff.Divide(eff_new)
            eff_diff.Scale(100)
            list_eff_diff.append(eff_diff)
            list_eff_diff_labels.append(string_ptjet)
            # plot the error of old w.r.t. new gen level
            effgen_new = list_ptcand_effgen_new[ibin2].Clone("effgen_pr_new_%d" % ibin2)
            effgen_diff = list_ptcand_eff_old[ibin2].Clone("effgen_pr_diff_%d" % ibin2)
            effgen_diff.Add(effgen_new, -1)
            effgen_diff.Divide(effgen_new)
            effgen_diff.Scale(100)
            list_effgen_diff.append(effgen_diff)
            list_effgen_diff_labels.append(string_ptjet)
        list_eff_diff.append(latex_shaperange)
        list_effgen_diff.append(latex_shaperange)
        line_0 = TLine(self.lpt_finbinmin[0], 0, self.lpt_finbinmax[-1], 0)
        list_eff_diff.append(line_0)
        list_effgen_diff.append(line_0)
        make_plot("efficiency_pr_eff_diff", path=self.d_resultsallpdata, list_obj=list_eff_diff, labels_obj=list_eff_diff_labels, \
            title="correction of efficiency calculation (rec. level);#it{p}_{T}^{%s} (GeV/#it{c});error = (old #minus new)/new (%%)" % self.p_latexnhadron, \
            leg_pos=[0.1, 0.62, 0.4, 0.77], margins_y=[0.05, 0.05])
        make_plot("efficiency_pr_effgen_diff", path=self.d_resultsallpdata, list_obj=list_effgen_diff, labels_obj=list_effgen_diff_labels, \
            title="correction of efficiency calculation (gen. level);#it{p}_{T}^{%s} (GeV/#it{c});error = (old #minus new)/new (%%)" % self.p_latexnhadron, \
            leg_pos=[0.55, 0.15, 0.85, 0.3], margins_y=[0.05, 0.05])

        # NON-PROMPT EFFICIENCY

        # The old (wrong) method

        cEffFD = TCanvas("cEffFD", "The Fit Canvas")
        setup_canvas(cEffFD)
        legeffFD = TLegend(.13, .65, .5, .88)
        setup_legend(legeffFD)
        list_his = []
        for ibin2 in range(self.p_nbin2_reco):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                            self.lvar2_binmin_gen[ibin2], \
                                            self.lvar2_binmax_gen[ibin2])
            h_gen_fd = lfileeff.Get("h_gen_fd" + stringbin2)
            h_sel_fd = lfileeff.Get("h_sel_fd" + stringbin2)
            h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
            list_his.append(h_sel_fd)
        y_min_h, y_max_h = get_y_window_his(list_his)
        y_min_h = 0
        y_margin_up = 0.35
        y_margin_down = 0
        y_min, y_max = get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up)
        for ibin2 in range(self.p_nbin2_reco):
            h_sel_fd = list_his[ibin2]
            setup_histogram(h_sel_fd, get_colour(ibin2), 1)
            h_sel_fd.Draw("same")
            fileouteff.cd()
            h_sel_fd.SetName("eff_fd_mult%d" % ibin2)
            h_sel_fd.Write()
            legeffFDstring = "%g #leq %s < %g GeV/#it{c}" % \
                    (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var,
                     self.lvar2_binmax_gen[ibin2])
            legeffFD.AddEntry(h_sel_fd, legeffFDstring, "LE")
            h_sel_fd.SetTitle("")
            h_sel_fd.SetXTitle("#it{p}_{T}^{%s} (GeV/#it{c})" % self.p_latexnhadron)
            h_sel_fd.SetYTitle("non-prompt %s-jet efficiency" % self.p_latexnhadron)
            h_sel_fd.GetYaxis().SetRangeUser(y_min, y_max)
            h_sel_fd.SetTitleOffset(1.2, "Y")
            h_sel_fd.SetTitleOffset(1.1, "X")
        legeffFD.Draw()
        cEffFD.SaveAs("%s/efficiency_fd.eps" % self.d_resultsallpdata)

    # pylint: disable=too-many-locals, too-many-branches
    def sideband_sub(self):
        #This function perform sideband subtraction of the histograms.
        #The input files for this function are coming from:
        #    - root file containing the histograms of mass vs z called here
        #     "hzvsmass". There is one for each bin of HF pt and jet pt.
        #    - fit function performed in the fit function above fit() called in
        #    this function "func_file"
        #    - several histograms coming from the efficiency ROOT file

        self.loadstyle()
        lfile = TFile.Open(self.n_filemass)
        if not lfile:
            self.logger.fatal(make_message_notfound(self.n_filemass))
        func_file = TFile.Open(self.file_yields)
        if not func_file:
            self.logger.fatal(make_message_notfound(self.file_yields))
        eff_file = TFile.Open(self.file_efficiency)
        if not eff_file:
            self.logger.fatal(make_message_notfound(self.file_efficiency))
        fileouts = TFile.Open(self.file_sideband, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(self.file_sideband))
        fileouts.cd()

        # hzvsjetpt is going to be the sideband subtracted histogram of z vs
        # jet that is going to be filled after subtraction

        hzvsjetpt = TH2F("hzvsjetpt", "", self.p_nbinshape_reco, self.varshapebinarray_reco,
                         self.p_nbin2_reco, self.var2binarray_reco)
        hzvsjetpt.Sumw2()

        # This is a loop over jet pt and over HF candidate pT

        for ibin2 in range(self.p_nbin2_reco):
            heff = eff_file.Get("eff_mult%d" % ibin2)
            hz = None
            first_fit = 0
            # shape vs. hadron pT histogram with relative contributions to the signal of
            # the pT-integrated shape bins after the sideband subtraction and efficiency correction
            hrelsig = buildhisto("hrelsig_%d" % ibin2, "hrelsig_%d" % ibin2, self.varshapebinarray_reco, self.var1binarray)
            # shape vs. hadron pT histogram with relative contributions to the stat. unc. of
            # the pT-integrated shape bins after the sideband subtraction and efficiency correction
            hrelunc = buildhisto("hrelunc_%d" % ibin2, "hrelunc_%d" % ibin2, self.varshapebinarray_reco, self.var1binarray)
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                suffix = "%s%g_%g_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                          self.lvar2_binmax_reco[ibin2])
                suffix_plot = "%s_%g_%g_%s_%g_%g" % \
                    (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2],
                     self.v_var_binning, self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

                # In this part of the code we extract for each bin of jet pt
                # and HF pT the fit function of the data fit to extract mean and
                # sigma. IF THERE IS NO GOOD FIT THE GIVEN BIN IS DISCARDED AND
                # WILL NOT ENTER THE FINAL RESULT

                load_dir = func_file.GetDirectory(suffix)
                mass_fitter = load_dir.Get("fitter%d" % (ipt))
                mean = mass_fitter.GetMean()
                sigma = mass_fitter.GetSigma()
                bkg_fit = mass_fitter.GetBackgroundRecalcFunc()

                # Here I define the boundaries for the sideband subtractions
                # based on the results of the fit. We get usually 4-9 sigma from
                # the mean in both sides to extract the sideband distributions

                hzvsmass = lfile.Get("hzvsmass" + suffix)
                binmasslow2sig = \
                    hzvsmass.GetXaxis().FindBin(mean - self.signal_sigma * sigma)
                masslow2sig = mean - self.signal_sigma*sigma
                binmasshigh2sig = \
                    hzvsmass.GetXaxis().FindBin(mean + self.signal_sigma * sigma)
                masshigh2sig = mean + self.signal_sigma*sigma
                binmasslow4sig = \
                    hzvsmass.GetXaxis().FindBin(mean - self.sideband_sigma_1_left * sigma)
                masslow4sig = \
                    mean - self.sideband_sigma_1_left * sigma
                binmasshigh4sig = \
                    hzvsmass.GetXaxis().FindBin(mean + self.sideband_sigma_1_right * sigma)
                masshigh4sig = \
                    mean + self.sideband_sigma_1_right * sigma
                binmasslow9sig = \
                    hzvsmass.GetXaxis().FindBin(mean - self.sideband_sigma_2_left * sigma)
                masslow9sig = \
                    mean - self.sideband_sigma_2_left * sigma
                binmasshigh9sig = \
                    hzvsmass.GetXaxis().FindBin(mean + self.sideband_sigma_2_right * sigma)
                masshigh9sig = \
                    mean + self.sideband_sigma_2_right * sigma

                # here we project over the z-axis the 2d distributions in the
                # three regions = signal region, left and right sideband

                hzsig = hzvsmass.ProjectionY("hzsig" + suffix, \
                             binmasslow2sig, binmasshigh2sig, "e")
                hzbkgleft = hzvsmass.ProjectionY("hzbkgleft" + suffix, \
                             binmasslow9sig, binmasslow4sig, "e")
                hzbkgright = hzvsmass.ProjectionY("hzbkgright" + suffix, \
                             binmasshigh4sig, binmasshigh9sig, "e")

                # the background histogram is made by adding the left and
                # right sideband in general. self.sidebandleftonly = True is
                # just made for systematic studies

                # Below a list of histograms are defined:
                #    - hzsig is as discussed before the distribution of z in
                #      the signal region not background subtracted
                #    - hzsub is the z-distribution after background subtraction
                #      using sidebands, efficiency corrected.
                #    - hzsub_noteffscaled is the z-distribution after background
                #      subtraction not corrected for efficiency
                #    - hzbkg_scaled is the bkg distribution scaled for the
                #      factor used to perform the background subtraction

                hzbkg = hzbkgleft.Clone("hzbkg" + suffix)
                if self.sidebandleftonly is False:
                    hzbkg.Add(hzbkgright)
                hzbkg_scaled = hzbkg.Clone("hzbkg_scaled" + suffix)

                area_scale_denominator = -1
                if not bkg_fit: # if there is no background fit it continues
                    continue
                area_scale_denominator = bkg_fit.Integral(masslow9sig, masslow4sig)
                if self.sidebandleftonly is False:
                    area_scale_denominator += bkg_fit.Integral(masshigh4sig, masshigh9sig)
                if area_scale_denominator == 0:
                    continue
                area_scale = bkg_fit.Integral(masslow2sig, masshigh2sig) / area_scale_denominator
                hzsub = hzsig.Clone("hzsub" + suffix)

                # subtract the scaled sideband yields

                hzsub.Add(hzbkg, -1 * area_scale)

                # set negative yields to zero

                for ibinz in range(hzsub.GetNbinsX()):
                    binz = ibinz + 1
                    if hzsub.GetBinContent(binz) <= 0:
                        hzsub.SetBinContent(binz, 0)
                        hzsub.SetBinError(binz, 0)

                hzsub_noteffscaled = hzsub.Clone("hzsub_noteffscaled" + suffix)
                hzbkg_scaled.Scale(area_scale)

                # correct for the efficiency

                eff = heff.GetBinContent(ipt + 1)
                if eff > 0.0:
                    hzsub.Scale(1.0 / (eff * self.sigma_scale))

                # add the corrected yield to the sum

                if first_fit == 0:
                    hz = hzsub.Clone("hz")
                    first_fit = 1
                else:
                    hz.Add(hzsub)

                for ishape in range(self.p_nbinshape_reco):
                    sig = hzsub.GetBinContent(ishape + 1)
                    unc = hzsub.GetBinError(ishape + 1)
                    hrelsig.SetBinContent(ishape + 1, ipt + 1, sig)
                    hrelunc.SetBinContent(ishape + 1, ipt + 1, unc * unc)

                fileouts.cd()
                hzsig.Write("hzsig" + suffix)
                hzbkgleft.Write("hzbkgleft" + suffix)
                hzbkgright.Write("hzbkgright" + suffix)
                hzbkg.Write("hzbkg" + suffix)
                hzbkg_scaled.Write()
                hzsub_noteffscaled.Write()
                hzsub.Write("hzsub" + suffix)

                #if self.feeddown_db:
                #    option = "sideband_subtracted"
                #    histo_to_compare = ("hzsub_noteffscaled%s" % (suffix))
                #    print("Making ratio for", option, histo_to_compare )
                #    xtitle = self.v_var_binning
                #    ytitle = "yield not effcor"
                #    self.makeratio_onedim(hzsub_noteffscaled, option, histo_to_compare, xtitle, ytitle)

                csblr = TCanvas("csblr" + suffix, "The Sideband Left-Right Canvas" + suffix)
                setup_canvas(csblr)
                csblr.Divide(1, 2, 0, 0)
                psblr_1 = csblr.cd(1)
                psblr_2 = csblr.cd(2)
                psblr_1.SetPad(0, 0.4, 0.95, 0.95)
                psblr_1.SetTopMargin(0.08)
                psblr_1.SetRightMargin(0.05)
                psblr_2.SetPad(0, 0, 0.95, 0.4)
                psblr_2.SetBottomMargin(0.2)
                psblr_2.SetRightMargin(0.05)
                psblr_1.SetFillColor(0)
                psblr_1.SetTicks(1, 1)
                psblr_2.SetFillColor(0)
                psblr_2.SetTicks(1, 1)
                csblr.cd(1)
                legsigbkgsblr = TLegend(.18, .7, .45, .85)
                setup_legend(legsigbkgsblr, 0.06)
                setup_histogram(hzbkgleft, get_colour(1), get_marker(0))
                legsigbkgsblr.AddEntry(hzbkgleft, "left sideband region", "P")
                setup_histogram(hzbkgright, get_colour(2), get_marker(1))
                legsigbkgsblr.AddEntry(hzbkgright, "right sideband region", "P")
                #y_min = min(hzbkgleft.GetMinimum(0), hzbkgright.GetMinimum(0))
                #y_max = max(hzbkgleft.GetMaximum(), hzbkgright.GetMaximum())
                y_min_h, y_max_h = get_y_window_his([hzbkgleft, hzbkgright])
                margin_up = 0.3
                margin_down = 0.05
                hzbkgleft.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, margin_down, margin_up))
                hzbkgleft.SetTitle("")
                hzbkgleft.SetXTitle(self.v_varshape_latex)
                hzbkgleft.SetYTitle("yield")
                hzbkgleft.SetLabelSize(0.06, "Y")
                hzbkgleft.SetTitleSize(0.06, "Y")
                hzbkgleft.SetTitleOffset(0.8, "Y")
                hzbkgleft.GetYaxis().SetMaxDigits(3)
                hzbkgleft.Draw()
                hzbkgright.Draw("same")
                legsigbkgsblr.Draw("same")
                latex = TLatex(0.6, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
                draw_latex(latex, 1, 0.06)
                latex2 = TLatex(0.6, 0.72,
                                "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" \
                                % (self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt], self.lvar2_binmax_reco[ibin2])))
                draw_latex(latex2, 1, 0.06)
                csblr.cd(2)
                hzbkgratio = hzbkgright.Clone("hzbkgratio_n" + suffix)
                hzbkgratio_d = hzbkgleft.Clone("hzbkgratio_d" + suffix)
                int_n = hzbkgratio.Integral()
                int_d = hzbkgratio_d.Integral()
                if int_n > 0 and int_d > 0:
                    hzbkgratio.Scale(1./int_n)
                    hzbkgratio_d.Scale(1./int_d)
                    hzbkgratio.Divide(hzbkgratio_d)
                    line = TLine(round(self.lvarshape_binmin_reco[0], 2), 1, round(self.lvarshape_binmax_reco[-1], 2), 1)
                    setup_histogram(hzbkgratio, get_colour(0), get_marker(0))
                    y_min_h = min(1, hzbkgratio.GetMinimum(0))
                    y_max_h = max(1, hzbkgratio.GetMaximum())
                    margin_up = 0.1
                    margin_down = 0.1
                    hzbkgratio.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, margin_down, margin_up))
                    hzbkgratio.SetYTitle("ratio of self-normalised yields")
                    hzbkgratio.SetXTitle(self.v_varshape_latex)
                    hzbkgratio.SetLabelSize(0.07, "Y")
                    hzbkgratio.SetTitleSize(0.07, "Y")
                    hzbkgratio.SetTitleOffset(0.7, "Y")
                    hzbkgratio.SetLabelSize(0.07, "X")
                    hzbkgratio.SetTitleSize(0.07, "X")
                    hzbkgratio.SetTitleOffset(1.2, "X")
                    hzbkgratio.Draw()
                    line.Draw("same")
                csblr.SaveAs("%s/sideband_left_right_%s.eps" % (self.d_resultsallpdata, suffix_plot))

                # This canvas will contain the distributions of the sideband
                # subtracted z-distributions in bin of the reco jet pt
                # variable, corrected for HF candidate efficiency

                csubz = TCanvas("csubz" + suffix, "The Side-Band Sub Canvas" + suffix)
                setup_canvas(csubz)
                setup_histogram(hzsub, get_colour(1), get_marker(0))
                y_min_h, y_max_h = get_y_window_his(hzsub)
                y_margin_up = 0.15
                y_margin_down = 0.05
                hzsub.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
                hzsub.SetTitle("Signal yield, bg-subtracted, efficiency-corrected")
                hzsub.SetXTitle(self.v_varshape_latex)
                hzsub.SetYTitle("yield")
                hzsub.SetTitleOffset(1.2, "Y")
                hzsub.GetYaxis().SetMaxDigits(3)
                hzsub.Draw()
                latex = TLatex(0.2, 0.83, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
                draw_latex(latex)
                latex2 = TLatex(0.5, 0.83,
                                "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" \
                                % (self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt], self.lvar2_binmax_reco[ibin2])))
                draw_latex(latex2)
                csubz.SaveAs("%s/sideband_sub_effcorr_%s.eps" % (self.d_resultsallpdata, suffix_plot))

                #if self.feeddown_db:
                #    histo_to_compare = ("hzsub%s" % (suffix))
                #    print("Making ratio for", option, histo_to_compare )
                #    xtitle = self.v_var_binning
                #    ytitle = "yield effcor"
                #    self.makeratio_onedim(hzsub_noteffscaled, option, histo_to_compare, xtitle, ytitle)

                # csigbkgsubz
                # This canvas contains the hzsig distributions of z in the signal
                # region (signal+bkg), the hzbkg_scaled distribution of
                # background rescaled, hzsub_noteffscaled the signal subtracted
                # distribution without efficiency corrections.

                csigbkgsubz = TCanvas("csigbkgsubz" + suffix, "The Side-Band Canvas" + suffix)
                setup_canvas(csigbkgsubz)
                legsigbkgsubz = TLegend(.15, .70, .35, .85)
                setup_legend(legsigbkgsubz)
                setup_histogram(hzsig, get_colour(1), get_marker(0))
                legsigbkgsubz.AddEntry(hzsig, "signal region", "P")
                logscale = True
                y_min_h, y_max_h = get_y_window_his([hzsig, hzbkg_scaled, hzsub_noteffscaled])
                y_margin_up = 0.35
                y_margin_down = 0.05
                y_min_0 = min([h.GetMinimum(0) for h in [hzsig, hzbkg_scaled, hzsub_noteffscaled]])
                if logscale and y_min_h <= 0:
                    y_min_h = y_min_0
                    if y_max_h <= 0:
                        logscale = False
                hzsig.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up, logscale))
                hzsig.GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), \
                                              round(self.lvarshape_binmax_reco[-1], 2))
                hzsig.SetTitle("")
                hzsig.SetXTitle(self.v_varshape_latex)
                hzsig.SetYTitle("yield")
                hzsig.SetTitleOffset(1.2, "Y")
                hzsig.GetYaxis().SetMaxDigits(3)
                hzsig.Draw()
                setup_histogram(hzbkg_scaled, get_colour(2), get_marker(1))
                legsigbkgsubz.AddEntry(hzbkg_scaled, "sideband region", "P")
                hzbkg_scaled.Draw("same")
                setup_histogram(hzsub_noteffscaled, get_colour(3), get_marker(2))
                legsigbkgsubz.AddEntry(hzsub_noteffscaled, "subtracted", "P")
                hzsub_noteffscaled.Draw("same")
                legsigbkgsubz.Draw("same")
                #PREL latex = TLatex(0.42, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
                latex = TLatex(0.42, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
                draw_latex(latex)
                latex1 = TLatex(0.42, 0.77, ("charged jets, anti-#it{k}_{T}, "
                                             "#it{R} = 0.4, #left|#it{#eta}_{jet}#right| #leq 0.5"))
                draw_latex(latex1)
                latex2 = TLatex(0.42, 0.72, "%g #leq %s < %g GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
                draw_latex(latex2)
                latex3 = TLatex(0.42, 0.67, "with %s, %g #leq #it{p}_{T, %s} < %g GeV/#it{c}" % \
                    (self.p_latexnhadron, self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt], self.lvar2_binmax_reco[ibin2])))
                draw_latex(latex3)
                if logscale:
                    csigbkgsubz.SetLogy()
                csigbkgsubz.SaveAs("%s/sideband_sub_%s.eps" % \
                    (self.d_resultsallpdata, suffix_plot))

                # Canvas to compare the shape of the left and right sidebands.
                # preliminary figure
                if ibin2 in [1] and ipt in [4, 5]:
                    text_ptjet_full = self.text_ptjet % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2])
                    text_pth_full = self.text_pth % (self.lpt_finbinmin[ipt], self.p_latexnhadron, min(self.lpt_finbinmax[ipt], self.lvar2_binmax_reco[ibin2]), self.p_latexnhadron)
                    if self.shape == "zg":
                        leg_pos = [.15, .15, .30, .30]
                    elif self.shape == "rg":
                        leg_pos = [.65, .12, .85, .27]
                    elif self.shape == "nsd":
                        leg_pos = [.35, .12, .55, .27]
                    else:
                        leg_pos = [.68, .72, .85, .85]
                    list_obj = [hzsig, hzbkg_scaled, hzsub_noteffscaled]
                    labels_obj = ["signal region", "sideband region", "after subtraction"]
                    colours = [get_colour(i) for i in [2, 3, 1]]
                    markers = [get_marker(i) for i in [0, 1, 2]]
                    y_margin_up = 0.4
                    y_margin_down = 0.05
                    c_sb_sub, list_obj_new = make_plot("c_sb_sub_" + suffix, size=self.size_can, \
                        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=self.opt_leg_g, opt_plot_g=self.opt_plot_g, offsets_xy=self.offsets_axes, \
                        colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=self.margins_can, \
                        title=";%s;yield" % self.title_x, logscale="y")
                    list_obj_new[0].SetTextSize(self.fontsize)
                    if self.shape == "nsd":
                        list_obj_new[1].GetXaxis().SetNdivisions(5)
                    # Draw LaTeX
                    y_latex = self.y_latex_top
                    list_latex = []
                    for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
                        latex = TLatex(self.x_latex, y_latex, text_latex)
                        list_latex.append(latex)
                        draw_latex(latex, textsize=self.fontsize)
                        y_latex -= self.y_step
                    c_sb_sub.Update()
                    c_sb_sub.SaveAs("%s/%s_sb_sub_%s.pdf" % (self.d_resultsallpdata, self.shape, suffix_plot))

            suffix = "%s_%g_%g" % \
                         (self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                          self.lvar2_binmax_reco[ibin2])
            if first_fit == 0:
                print("No successful fits for: %s" % suffix)
                continue

            # We are now outside of the loop of HF candidate pt. We are going now
            # to plot the "hz" histogram, which contains the Add of all the
            # bkg-subtracted efficiency corrected distributions of all the HF
            # candidate pt bins put together. Each "hz" distribution made for each
            # jet pt is normalized by its own area. We also fill a 2D histogram
            # called "hzvsjetpt" that contains all the z distributions of all jet pt.

            cz = TCanvas("cz" + suffix,
                         "The Efficiency Corrected Signal Yield Canvas" + suffix)
            setup_canvas(cz)
            setup_histogram(hz, get_colour(1), get_marker(0))
            y_min_h, y_max_h = get_y_window_his(hz)
            y_margin_up = 0.15
            y_margin_down = 0.05
            hz.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            hz.SetTitle("Signal yield, bg-subtracted, efficiency-corrected")
            hz.SetXTitle(self.v_varshape_latex)
            hz.SetYTitle("yield")
            hz.SetTitleOffset(1.2, "Y")
            hz.GetYaxis().SetMaxDigits(3)
            hz.Draw()
            latex = TLatex(0.2, 0.83, "%g #leq %s < %g GeV/#it{c}" % \
                           (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cz.SaveAs("%s/sideband_sub_effcorr_ptint_%s.eps" % \
                      (self.d_resultsallpdata, suffix))

            # fill the 2D histogram shape vs jet pt

            for zbins in range(self.p_nbinshape_reco):
                hzvsjetpt.SetBinContent(zbins + 1, ibin2 + 1, hz.GetBinContent(zbins + 1))
                hzvsjetpt.SetBinError(zbins + 1, ibin2 + 1, hz.GetBinError(zbins + 1))

            # Normalise hrelsig and hrelunc and make other test plots.

            # effect of including the bin relative to not including it
            himpincl = buildhisto("himpincl_%d" % ibin2, "himpincl_%d" % ibin2, self.varshapebinarray_reco, self.var1binarray)
            # effect of excluding the bin relative to keeping it
            himpexcl = buildhisto("himpexcl_%d" % ibin2, "himpexcl_%d" % ibin2, self.varshapebinarray_reco, self.var1binarray)
            # test condition
            himptest = buildhisto("himptest_%d" % ibin2, "himptest_%d" % ibin2, self.varshapebinarray_reco, self.var1binarray)
            for ishape in range(self.p_nbinshape_reco):
                sig_tot = hz.GetBinContent(ishape + 1)
                unc_tot = hz.GetBinError(ishape + 1)
                if sig_tot <= 0 or unc_tot <= 0:
                    continue
                rel_unc_tot = unc_tot / sig_tot
                for ipt in range(self.p_nptfinbins):
                    sig = hrelsig.GetBinContent(ishape + 1, ipt + 1)
                    if sig <= 0:
                        continue
                    unc_sq = hrelunc.GetBinContent(ishape + 1, ipt + 1)
                    frac_sig = sig / sig_tot
                    frac_unc = unc_sq / (unc_tot * unc_tot)
                    hrelsig.SetBinContent(ishape + 1, ipt + 1, 100 * frac_sig)
                    hrelunc.SetBinContent(ishape + 1, ipt + 1, 100 * frac_unc)
                    # How much does the bin improve the total relative uncertainty?
                    if frac_sig < 1 - 1e-7 and frac_unc < 1 - 1e-7: # avoid numerical errors
                        improve_test = frac_sig + frac_unc / frac_sig # The bin improves the rel. stat. unc. if improve_test < 2.
                        himptest.SetBinContent(ishape + 1, ipt + 1, 1 if improve_test < 2 else 2)
                        sig_tot_without = sig_tot - sig
                        unc_tot_without = sqrt(unc_tot * unc_tot - unc_sq)
                        rel_unc_tot_without = unc_tot_without / sig_tot_without
                        rel_unc_diff_rel_incl = (rel_unc_tot - rel_unc_tot_without) / rel_unc_tot_without # effect of including the bin relative to not including it
                        rel_unc_diff_rel_excl = (rel_unc_tot_without - rel_unc_tot) / rel_unc_tot # effect of excluding the bin relative to keeping it
                        # rel_unc_diff_rel_excl = 1 / (1 + rel_unc_diff_rel_incl) - 1
                        himpincl.SetBinContent(ishape + 1, ipt + 1, 100 * rel_unc_diff_rel_incl)
                        himpexcl.SetBinContent(ishape + 1, ipt + 1, 100 * rel_unc_diff_rel_excl)
            hworth = hrelsig.Clone("hworth_%d" % ibin2)
            hworth.Divide(hrelunc)

            latex = TLatex(0.15, 0.02, "%g #leq %s < %g GeV/#it{c}" % \
                           (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))

            crelsig = TCanvas("crelsig_" + suffix, "crelsig_" + suffix)
            setup_canvas(crelsig)
            crelsig.SetRightMargin(0.18)
            setup_histogram(hrelsig)
            hrelsig.SetTitle("relative contribution to the signal;%s;%s;fraction #it{f}_{#it{S}} (%%)" % (self.v_varshape_latex, "%s (GeV/#it{c})" % self.v_pth_latex))
            hrelsig.SetTitleSize(0.05, "Z")
            hrelsig.SetTitleOffset(1.0, "Z")
            hrelsig.GetZaxis().SetRangeUser(hrelsig.GetMinimum(0), hrelsig.GetMaximum())
            hrelsig.Draw("colz")
            gStyle.SetPaintTextFormat(".2f")
            hrelsig.Draw("text same")
            draw_latex(latex, textsize=0.04)
            crelsig.SaveAs("%s/sideband_relsig_%s.eps" % (self.d_resultsallpdata, suffix))

            crelunc = TCanvas("crelunc_" + suffix, "crelunc_" + suffix)
            setup_canvas(crelunc)
            crelunc.SetRightMargin(0.18)
            setup_histogram(hrelunc)
            hrelunc.SetTitle("relative contribution to the stat. unc. squared;%s;%s;fraction #it{f}_{#it{#sigma}^{2}} (%%)" % (self.v_varshape_latex, "%s (GeV/#it{c})" % self.v_pth_latex))
            hrelunc.SetTitleSize(0.05, "Z")
            hrelunc.SetTitleOffset(1.0, "Z")
            hrelunc.GetZaxis().SetRangeUser(hrelunc.GetMinimum(0), hrelunc.GetMaximum())
            hrelunc.Draw("colz")
            gStyle.SetPaintTextFormat(".2f")
            hrelunc.Draw("text same")
            draw_latex(latex, textsize=0.04)
            crelunc.SaveAs("%s/sideband_relunc_%s.eps" % (self.d_resultsallpdata, suffix))

            cworth = TCanvas("cworth_" + suffix, "cworth_" + suffix)
            setup_canvas(cworth)
            cworth.SetRightMargin(0.18)
            setup_histogram(hworth)
            hworth.SetTitle("worth = (rel. signal contrib.)/(rel. unc. contrib.);%s;%s;worth #it{w} = #it{f}_{#it{S}}/#it{f}_{#it{#sigma}^{2}}" % (self.v_varshape_latex, "%s (GeV/#it{c})" % self.v_pth_latex))
            hworth.SetTitleSize(0.05, "Z")
            hworth.SetTitleOffset(1.0, "Z")
            hworth.GetZaxis().SetRangeUser(hworth.GetMinimum(0), hworth.GetMaximum())
            hworth.Draw("colz")
            gStyle.SetPaintTextFormat(".2f")
            hworth.Draw("text same")
            draw_latex(latex, textsize=0.04)
            cworth.SaveAs("%s/sideband_worth_%s.eps" % (self.d_resultsallpdata, suffix))

            cimpincl = TCanvas("cimpincl_" + suffix, "cimpincl_" + suffix)
            setup_canvas(cimpincl)
            cimpincl.SetRightMargin(0.18)
            setup_histogram(himpincl)
            himpincl.SetTitle("inclusion impact on the rel. stat. unc.;%s;%s;relative effect (%%)" % (self.v_varshape_latex, "%s (GeV/#it{c})" % self.v_pth_latex))
            himpincl.SetTitleSize(0.05, "Z")
            himpincl.SetTitleOffset(1.0, "Z")
            himpincl.GetZaxis().SetRangeUser(himpincl.GetMinimum(), himpincl.GetMaximum())
            himpincl.GetZaxis().SetMaxDigits(3)
            himpincl.Draw("colz")
            gStyle.SetPaintTextFormat(".2f")
            himpincl.Draw("text same")
            draw_latex(latex, textsize=0.04)
            cimpincl.SaveAs("%s/sideband_impincl_%s.eps" % (self.d_resultsallpdata, suffix))

            cimpexcl = TCanvas("cimpexcl_" + suffix, "cimpexcl_" + suffix)
            setup_canvas(cimpexcl)
            cimpexcl.SetRightMargin(0.18)
            setup_histogram(himpexcl)
            himpexcl.SetTitle("exclusion impact on the rel. stat. unc.;%s;%s;relative effect (%%)" % (self.v_varshape_latex, "%s (GeV/#it{c})" % self.v_pth_latex))
            himpexcl.SetTitleSize(0.05, "Z")
            himpexcl.SetTitleOffset(1.0, "Z")
            himpexcl.GetZaxis().SetRangeUser(himpexcl.GetMinimum(), himpexcl.GetMaximum())
            himpexcl.GetZaxis().SetMaxDigits(3)
            himpexcl.Draw("colz")
            gStyle.SetPaintTextFormat(".2f")
            himpexcl.Draw("text same")
            draw_latex(latex, textsize=0.04)
            cimpexcl.SaveAs("%s/sideband_impexcl_%s.eps" % (self.d_resultsallpdata, suffix))

            cimptest = TCanvas("cimptest_" + suffix, "cimptest_" + suffix)
            setup_canvas(cimptest)
            cimptest.SetRightMargin(0.18)
            setup_histogram(himptest)
            himptest.SetTitle("Bin improves the rel. stat. unc. if (test value = 1).;%s;%s;test value" % (self.v_varshape_latex, "%s (GeV/#it{c})" % self.v_pth_latex))
            himptest.SetTitleSize(0.05, "Z")
            himptest.SetTitleOffset(1.0, "Z")
            himptest.GetZaxis().SetRangeUser(1, 2)
            himptest.SetContour(2)
            himptest.Draw("colz")
            gStyle.SetPaintTextFormat("g")
            himptest.Draw("text same")
            draw_latex(latex, textsize=0.04)
            cimptest.SaveAs("%s/sideband_imptest_%s.eps" % (self.d_resultsallpdata, suffix))

            gStyle.SetPaintTextFormat("g")

            # Normalise pT-integrated yields.

            hz.Scale(1.0 / hz.Integral())
            fileouts.cd()
            hz.Write("hz" + suffix)

        fileouts.cd()
        hzvsjetpt.Write("hzvsjetpt")

        czvsjetpt = TCanvas("czvsjetpt", "output of sideband subtraction")
        setup_canvas(czvsjetpt)
        setup_histogram(hzvsjetpt)
        hzvsjetpt.SetTitle("")
        hzvsjetpt.SetXTitle(self.v_varshape_latex)
        hzvsjetpt.SetYTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
        hzvsjetpt.Draw("text")
        czvsjetpt.SaveAs("%s/sideband_output.eps" % self.d_resultsallpdata)
        #if self.feeddown_db:
        #    option = "sideband_subtracted"
        #    histo_to_compare = ("hzvsjetpt")
        #    print("Making ratio for", option, histo_to_compare )
        #    self.makeratio_twodim(hzvsjetpt, option, histo_to_compare)
        fileouts.Close()

    def feeddown(self):

        #In this function we compute the feeddown fraction to be subtracted to
        #extract the prompt z distributions of HF tagged jets.

        #The ingredients are the efficiency file that contains prompt and
        #non-prompt efficiency for HF hadron reconstruction as a function of pT
        #in bins of jet pt (file_eff) and the output file of the jet processer that
        #contains all the response matrix and jet efficiencies (feeddown_input_file).


        self.loadstyle()
        feeddown_input_file = TFile.Open(self.n_fileresp)
        if not feeddown_input_file:
            self.logger.fatal(make_message_notfound(self.n_fileresp))
        file_eff = TFile.Open(self.file_efficiency)
        if not file_eff:
            self.logger.fatal(make_message_notfound(self.file_efficiency))
        fileouts = TFile.Open(self.file_feeddown, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(self.file_feeddown))

        response_matrix = feeddown_input_file.Get("response_matrix_nonprompt")

        # input_data is 3d histogram from powheg+pythia prediction that
        # contains z vs jet_pt vs HF pt.
        input_data = self.get_simulated_yields(self.powheg_path_nonprompt, 3, False)
        if not input_data:
            self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_nonprompt))
        input_data.SetName("fh3_feeddown_%s" % self.v_varshape_binning)

        # Ensure correct binning: x - shape, y - jet pt, z - pt hadron
        if not equal_binning_lists(input_data, list_x=self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(input_data, list_y=self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")
        if not equal_binning_lists(input_data, list_z=self.var1ranges):
            self.logger.fatal("Error: Incorrect binning in z.")

        fileouts.cd()
        input_data.Write()

        # output_template is the reco jet pt vs z for candidates in the reco
        # min-max region
        output_template = feeddown_input_file.Get("hzvsjetpt_reco")
        # Ensure correct binning: x - shape, y - jet pt
        if not equal_binning_lists(output_template, list_x=self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(output_template, list_y=self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")

        # hzvsjetpt_gen_nocuts_nonprompt is the 2d plot of gen z vs gen jet pt
        # for events in the gen min-max range
        hzvsjetpt_gen_nocuts = feeddown_input_file.Get("hzvsjetpt_gen_nocuts_nonprompt")
        # hzvsjetpt_gen_cuts_nonprompt is the 2d plot of gen z vs gen jet pt
        # for events in the gen and reco min-max range
        hzvsjetpt_gen_eff = feeddown_input_file.Get("hzvsjetpt_gen_cuts_nonprompt")
        hzvsjetpt_gen_eff.Divide(hzvsjetpt_gen_nocuts)

        # hzvsjetpt_reco_nocuts_nonprompt is the 2d plot of reco z vs reco jet pt
        # for events in the reco min-max range
        hzvsjetpt_reco_nocuts = feeddown_input_file.Get("hzvsjetpt_reco_nocuts_nonprompt")
        # hzvsjetpt_reco_cuts_nonprompt is the 2d plot of reco z vs reco jet pt
        # for events in the reco and gen min-max range
        hzvsjetpt_reco_eff = feeddown_input_file.Get("hzvsjetpt_reco_cuts_nonprompt")
        hzvsjetpt_reco_eff.Divide(hzvsjetpt_reco_nocuts)

        sideband_input_data_file = TFile.Open(self.file_sideband)
        if not sideband_input_data_file:
            self.logger.fatal(make_message_notfound(self.file_sideband))
        sideband_input_data = sideband_input_data_file.Get("hzvsjetpt")

        hz_genvsreco_list = []
        hjetpt_genvsreco_list = []

        hjetpt_fracdiff_list = []
        hz_fracdiff_list = []
        heff_pr_list = []
        heff_fd_list = []
        input_data_zvsjetpt_list = []
        input_data_scaled = TH2F()

        # cgen_eff is the efficiency that a candidate generated in the gen
        # limits has reco values in the reco limits

        cgen_eff = TCanvas("cgen_eff_nonprompt ", "gen efficiency applied to feedown")
        setup_canvas(cgen_eff)
        setup_histogram(hzvsjetpt_gen_eff)
        hzvsjetpt_gen_eff.SetTitle("")
        hzvsjetpt_gen_eff.SetXTitle("%s^{gen}" % self.v_varshape_latex)
        hzvsjetpt_gen_eff.SetYTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hzvsjetpt_gen_eff.Draw("text")
        cgen_eff.SaveAs("%s/kineff_fd_gen.eps" % self.d_resultsallpdata)

        # creco_eff is the kinematic efficiency that a candidate in reco limits
        # limits has gen values in the gen limits

        creco_eff = TCanvas("creco_eff_nonprompt ", "reco efficiency applied to feedown")
        setup_canvas(creco_eff)
        setup_histogram(hzvsjetpt_reco_eff)
        hzvsjetpt_reco_eff.SetTitle("")
        hzvsjetpt_reco_eff.SetXTitle("%s^{rec}" % self.v_varshape_latex)
        hzvsjetpt_reco_eff.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hzvsjetpt_reco_eff.Draw("text")
        creco_eff.SaveAs("%s/kineff_fd_rec.eps" % self.d_resultsallpdata)

        # plot the response

        his_response_fd = response_matrix.Hresponse() # linearised response matrix as a TH2
        cresponse_fd = TCanvas("cresponse_fd", "non-prompt response matrix")
        setup_canvas(cresponse_fd)
        cresponse_fd.SetLogz()
        cresponse_fd.SetRightMargin(0.13)
        setup_histogram(his_response_fd)
        his_response_fd.SetTitle("")
        his_response_fd.SetXTitle("(#it{p}_{T, jet}^{rec}, %s^{rec}) bin" % self.v_varshape_latex)
        his_response_fd.SetYTitle("(#it{p}_{T, jet}^{gen}, %s^{gen}) bin" % self.v_varshape_latex)
        his_response_fd.Draw("colz")
        cresponse_fd.SaveAs("%s/response_fd_matrix.eps" % self.d_resultsallpdata)

        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                      self.lvar2_binmax_reco[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                      self.lvar2_binmax_reco[ibin2])
            hz_genvsreco_list.append(feeddown_input_file.Get("hz_genvsreco_nonprompt" + suffix))

            cz_genvsreco = TCanvas("cz_genvsreco_nonprompt" + suffix,
                                   "response matrix 2D projection")
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
            cz_genvsreco.SaveAs("%s/response_fd_%s_%s.eps" % \
                (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

        for ibinshape in range(self.p_nbinshape_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco_list.append( \
                feeddown_input_file.Get("hjetpt_genvsreco_nonprompt" + suffix))

            cjetpt_genvsreco = TCanvas("cjetpt_genvsreco_nonprompt" + suffix,
                                       "response matrix 2D projection")
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
            cjetpt_genvsreco.SaveAs("%s/response_fd_%s_%s.eps" % \
                (self.d_resultsallpdata, self.v_var2_binning, suffix_plot))

        hz_genvsreco_full = feeddown_input_file.Get("hz_genvsreco_full_nonprompt")
        hjetpt_genvsreco_full = feeddown_input_file.Get("hjetpt_genvsreco_full_nonprompt")

        cz_genvsreco = TCanvas("cz_genvsreco_full_nonprompt", "response matrix 2D projection")
        setup_canvas(cz_genvsreco)
        cz_genvsreco.SetRightMargin(0.13)
        cz_genvsreco.SetLogz()
        setup_histogram(hz_genvsreco_full)
        hz_genvsreco_full.GetZaxis().SetRangeUser(hz_genvsreco_full.GetMinimum(0), hz_genvsreco_full.GetMaximum())
        hz_genvsreco_full.SetTitle("")
        hz_genvsreco_full.SetXTitle("%s^{gen}" % self.v_varshape_latex)
        hz_genvsreco_full.SetYTitle("%s^{rec}" % self.v_varshape_latex)
        hz_genvsreco_full.Draw("colz")
        cz_genvsreco.SaveAs("%s/response_fd_%s_full.eps" % \
                            (self.d_resultsallpdata, self.v_varshape_binning))

        cjetpt_genvsreco = TCanvas("cjetpt_genvsreco_full_nonprompt", "response matrix 2D projection")
        setup_canvas(cjetpt_genvsreco)
        cjetpt_genvsreco.SetRightMargin(0.13)
        cjetpt_genvsreco.SetLogz()
        setup_histogram(hjetpt_genvsreco_full)
        hjetpt_genvsreco_full.GetZaxis().SetRangeUser(hjetpt_genvsreco_full.GetMinimum(0), hjetpt_genvsreco_full.GetMaximum())
        hjetpt_genvsreco_full.SetTitle("")
        hjetpt_genvsreco_full.SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hjetpt_genvsreco_full.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hjetpt_genvsreco_full.Draw("colz")
        cjetpt_genvsreco.SaveAs("%s/response_fd_%s_full.eps" % \
            (self.d_resultsallpdata, self.v_var2_binning))

        hz_genvsreco_full_real = feeddown_input_file.Get("hz_genvsreco_full_nonprompt_real")
        hjetpt_genvsreco_full_real = feeddown_input_file.Get("hjetpt_genvsreco_full_nonprompt_real")

        cz_genvsreco_real = TCanvas("cz_genvsreco_full_nonprompt_real", "response matrix 2D projection_real")
        setup_canvas(cz_genvsreco_real)
        cz_genvsreco_real.SetRightMargin(0.13)
        cz_genvsreco_real.SetLogz()
        setup_histogram(hz_genvsreco_full_real)
        hz_genvsreco_full_real.GetZaxis().SetRangeUser(hz_genvsreco_full_real.GetMinimum(0), hz_genvsreco_full_real.GetMaximum())
        hz_genvsreco_full_real.SetTitle("")
        hz_genvsreco_full_real.SetXTitle("%s^{gen}" % self.v_varshape_latex)
        hz_genvsreco_full_real.SetYTitle("%s^{rec}" % self.v_varshape_latex)
        hz_genvsreco_full_real.Draw("colz")
        cz_genvsreco_real.SaveAs("%s/response_fd_%s_full_real.eps" % \
                            (self.d_resultsallpdata, self.v_varshape_binning))

        cjetpt_genvsreco_real = TCanvas("cjetpt_genvsreco_full_nonprompt_real", "response matrix 2D projection_real")
        setup_canvas(cjetpt_genvsreco_real)
        cjetpt_genvsreco_real.SetRightMargin(0.13)
        cjetpt_genvsreco_real.SetLogz()
        setup_histogram(hjetpt_genvsreco_full_real)
        hjetpt_genvsreco_full_real.GetZaxis().SetRangeUser(hjetpt_genvsreco_full_real.GetMinimum(0), hjetpt_genvsreco_full_real.GetMaximum())
        hjetpt_genvsreco_full_real.SetTitle("")
        hjetpt_genvsreco_full_real.SetXTitle("#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        hjetpt_genvsreco_full_real.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        hjetpt_genvsreco_full_real.Draw("colz")
        cjetpt_genvsreco_real.SaveAs("%s/response_fd_%s_full_real.eps" % \
            (self.d_resultsallpdata, self.v_var2_binning))

        # plot promp and non-prompt efficiencies

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2],
                      self.lvar2_binmax_gen[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2],
                      self.lvar2_binmax_gen[ibin2])

            hjetpt_fracdiff_list.append( \
                feeddown_input_file.Get("hjetpt_fracdiff_nonprompt" + suffix))

            heff_pr_list.append(file_eff.Get("eff_mult%d" % ibin2))
            heff_fd_list.append(file_eff.Get("eff_fd_mult%d" % ibin2))

            ceff = TCanvas("ceff " + suffix, "prompt and non-prompt efficiencies" + suffix)
            setup_canvas(ceff)
            leg_eff = TLegend(.7, .15, .85, .25, "")
            setup_legend(leg_eff)
            setup_histogram(heff_pr_list[ibin2], get_colour(1), get_marker(0))
            leg_eff.AddEntry(heff_pr_list[ibin2], "prompt", "P")
            y_min_h, y_max_h = get_y_window_his([heff_pr_list[ibin2], heff_fd_list[ibin2]])
            y_min_h = 0
            y_margin_up = 0.05
            y_margin_down = 0
            bin_pt_max = min(self.p_nptfinbins, heff_pr_list[ibin2].GetXaxis().FindBin(self.lvar2_binmax_gen[ibin2] - 0.01))
            heff_pr_list[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            heff_pr_list[ibin2].GetXaxis().SetRange(1, bin_pt_max)
            heff_pr_list[ibin2].SetTitle("")
            heff_pr_list[ibin2].SetXTitle("#it{p}_{T, %s} (GeV/#it{c})" % self.p_latexnhadron)
            heff_pr_list[ibin2].SetYTitle("Efficiency #times Acceptance")
            heff_pr_list[ibin2].SetTitleOffset(1.2, "Y")
            heff_pr_list[ibin2].SetTitle("")
            heff_pr_list[ibin2].Draw()
            setup_histogram(heff_fd_list[ibin2], get_colour(2), get_marker(1))
            leg_eff.AddEntry(heff_fd_list[ibin2], "non-prompt", "P")
            heff_fd_list[ibin2].Draw("same")
            leg_eff.Draw("same")
            #PREL latex = TLatex(0.15, 0.83, "ALICE Preliminary")
            #PREL draw_latex(latex)
            latex2 = TLatex(0.15, 0.78, "PYTHIA 6, pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex2)
            latex3 = TLatex(0.15, 0.73, ("%s #rightarrow %s (and charge conj.)" % (self.p_latexnhadron, self.p_latexndecay)))
            draw_latex(latex3)
            latex4 = TLatex(0.15, 0.68, "in charged jets, anti-#it{k}_{T}, #it{R} = 0.4")
            draw_latex(latex4)
            latex5 = TLatex(0.15, 0.63, "%g #leq %s < %g GeV/#it{c}" % \
                            (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex5)
            latex6 = TLatex(0.15, 0.58, "#left|#it{#eta}_{jet}#right| #leq 0.5")
            draw_latex(latex6)
            ceff.SaveAs("%s/efficiency_pr_fd_%s.eps" % (self.d_resultsallpdata, suffix_plot))

            # preliminary figure
            if ibin2 in [1]:
                leg_pos = [.7, .15, .85, .30]
                list_obj = [heff_pr_list[ibin2], heff_fd_list[ibin2]]
                labels_obj = ["prompt", "non-prompt"]
                colours = [get_colour(i) for i in [1, 2]]
                markers = [get_marker(i) for i in [0, 1]]
                y_margin_up = 0.3
                y_margin_down = 0.05
                c_eff_both, list_obj_new = make_plot("c_eff_both_" + suffix, size=self.size_can, \
                    list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=self.opt_leg_g, opt_plot_g=self.opt_plot_g, offsets_xy=[1, 1.1], \
                    colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=[0.12, 0.13, 0.05, 0.03], \
                    title=";#it{p}_{T}^{%s} (GeV/#it{c});reconstruction efficiency" % self.p_latexnhadron)
                list_obj_new[0].SetTextSize(self.fontsize)
                # Draw LaTeX
                y_latex = self.y_latex_top
                list_latex = []
                for text_latex in [self.text_alice, "%s #rightarrow %s (and charge conj.)" % (self.p_latexnhadron, self.p_latexndecay), self.text_acc_h]:
                    latex = TLatex(self.x_latex, y_latex, text_latex)
                    list_latex.append(latex)
                    draw_latex(latex, textsize=self.fontsize)
                    y_latex -= self.y_step
                c_eff_both.Update()
                c_eff_both.SaveAs("%s/%s_eff_pr_fd_%s.pdf" % (self.d_resultsallpdata, self.shape, suffix_plot))

        # plot relative jet pt shift

        cjetpt_fracdiff = TCanvas("cjetpt_fracdiff ", "non-prompt jetpt response fractional differences")
        setup_canvas(cjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        leg_jetpt_fracdiff = TLegend(.15, .5, .25, .8, "#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            setup_histogram(hjetpt_fracdiff_list[ibin2], get_colour(ibin2), get_marker(ibin2))
            leg_jetpt_fracdiff.AddEntry(hjetpt_fracdiff_list[ibin2], \
                "%g#minus%g" % (self.lvar2_binmin_gen[ibin2], \
                self.lvar2_binmax_gen[ibin2]), "P")
            if ibin2 == 0:
                hjetpt_fracdiff_list[ibin2].SetTitle("")
                hjetpt_fracdiff_list[ibin2].SetXTitle(\
                    "(#it{p}_{T, jet}^{rec} #minus #it{p}_{T, jet}^{gen})/#it{p}_{T, jet}^{gen}")
                _, y_max_h = get_y_window_his(hjetpt_fracdiff_list)
                y_margin_up = 0.15
                y_margin_down = 0.05
                y_min_0 = min([h.GetMinimum(0) for h in hjetpt_fracdiff_list])
                hjetpt_fracdiff_list[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_0, y_max_h, y_margin_down, y_margin_up, True))
            hjetpt_fracdiff_list[ibin2].Draw("same")
        leg_jetpt_fracdiff.Draw("same")
        cjetpt_fracdiff.SaveAs("%s/response_fd_reldiff_%s.eps" % (self.d_resultsallpdata, self.v_var2_binning))

        # plot relative shape shift

        for ibinshape in range(self.p_nbinshape_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_list.append(feeddown_input_file.Get("hz_fracdiff_nonprompt" + suffix))

        cz_fracdiff = TCanvas("cz_fracdiff ", "non-prompt z response fractional differences")
        setup_canvas(cz_fracdiff)
        cz_fracdiff.SetLogy()
        leg_z_fracdiff = TLegend(.15, .5, .25, .8, "%s^{gen}" % self.v_varshape_latex)
        setup_legend(leg_z_fracdiff)
        for ibinshape in range(self.p_nbinshape_gen):
            setup_histogram(hz_fracdiff_list[ibinshape], get_colour(ibinshape), get_marker(ibinshape))
            leg_z_fracdiff.AddEntry(hz_fracdiff_list[ibinshape], \
                "%g#minus%g" % (self.lvarshape_binmin_gen[ibinshape], \
                self.lvarshape_binmax_gen[ibinshape]), "P")
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
        cz_fracdiff.SaveAs("%s/response_fd_reldiff_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning))

        # scale with the ratio of non-prompt/prompt efficiencies

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            suffix = "%s%g_%g_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id])
            input_data.GetZaxis().SetRange(ipt + 1, ipt + 1)
            input_data_zvsjetpt_list.append( \
                input_data.Project3D("input_data_zvsjetpt" + suffix + "_yxe"))
            for ibin2 in range(self.p_nbin2_gen):
                for ibinshape in range(self.p_nbinshape_gen):
                    # set to zero if either efficiency is zero
                    if(heff_pr_list[ibin2].GetBinContent(ipt + 1) == 0 or \
                       heff_fd_list[ibin2].GetBinContent(ipt + 1) == 0):
                        input_data_zvsjetpt_list[ipt].SetBinContent(ibinshape + 1, ibin2 + 1, 0.0)
                    else:
                        input_data_zvsjetpt_list[ipt].SetBinContent(ibinshape + 1, ibin2 + 1, \
                            input_data_zvsjetpt_list[ipt].GetBinContent(ibinshape + 1, ibin2 + 1) * \
                            (heff_fd_list[ibin2].GetBinContent(ipt + 1)/ \
                             heff_pr_list[ibin2].GetBinContent(ipt + 1)))
            # sum up pt bins
            if ipt == 0:
                input_data_scaled = input_data_zvsjetpt_list[ipt].Clone("input_data_scaled")
            else:
                input_data_scaled.Add(input_data_zvsjetpt_list[ipt])

        # apply gen. level kinematic efficiency

        input_data_scaled.Multiply(hzvsjetpt_gen_eff)

        # scale with real luminosity and branching ratio

        input_data_scaled.Scale(self.p_nevents * self.branching_ratio / self.xsection_inel)

        # fold with the non-prompt response matrix

        folded = folding(input_data_scaled, response_matrix, output_template)

        # apply rec. level kinematic efficiency

        folded.Divide(hzvsjetpt_reco_eff)

        # plot shape projections in jet pt bins
        folded_z_list = []
        input_data_scaled_z_list = []
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            suffix_plot = "%s_%g_%g" % \
                (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])

            folded_z_list.append(folded.ProjectionX("folded_z_nonprompt_" + suffix,
                                                    ibin2 + 1, ibin2 + 1, "e"))
            input_data_scaled_z_list.append( \
                input_data_scaled.ProjectionX("Powheg_scaled_nonprompt_" + suffix, \
                    input_data_scaled.GetYaxis().FindBin(self.lvar2_binmin_gen[ibin2]), \
                    input_data_scaled.GetYaxis().FindBin(self.lvar2_binmin_gen[ibin2]), "e"))
            c_fd_fold = TCanvas("c_fd_fold " + suffix, "Powheg and folded" + suffix)
            setup_canvas(c_fd_fold)
            c_fd_fold.SetLeftMargin(0.13)
            leg_fd_fold = TLegend(.15, .75, .4, .85, "")
            setup_legend(leg_fd_fold)
            setup_histogram(input_data_scaled_z_list[ibin2], get_colour(1), get_marker(0))
            leg_fd_fold.AddEntry(input_data_scaled_z_list[ibin2], "POWHEG, eff. scaled", "P")
            y_min_h, y_max_h = get_y_window_his([input_data_scaled_z_list[ibin2], folded_z_list[ibin2]])
            y_margin_up = 0.2
            y_margin_down = 0.05
            input_data_scaled_z_list[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            input_data_scaled_z_list[ibin2].SetTitle("")
            input_data_scaled_z_list[ibin2].SetXTitle(self.v_varshape_latex)
            input_data_scaled_z_list[ibin2].SetYTitle("yield")
            input_data_scaled_z_list[ibin2].SetTitleOffset(1.5, "Y")
            input_data_scaled_z_list[ibin2].Draw()
            setup_histogram(folded_z_list[ibin2], get_colour(2), get_marker(1))
            leg_fd_fold.AddEntry(folded_z_list[ibin2], "POWHEG, eff. scaled, folded", "P")
            folded_z_list[ibin2].Draw("same")
            leg_fd_fold.Draw("same")
            latex = TLatex(0.6, 0.8, "%g #leq %s < %g GeV/#it{c}" % \
                    (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            c_fd_fold.SaveAs("%s/feeddown_folded_%s.eps" % (self.d_resultsallpdata, suffix_plot))

        # plot the final feed-down shape vs jet pt distribution that is subtracted

        cfeeddown_2d = TCanvas("cfeeddown_2d", "feedown_2d")
        setup_canvas(cfeeddown_2d)
        setup_histogram(folded)
        folded.SetTitle("")
        folded.SetXTitle("%s^{rec}" % self.v_varshape_latex)
        folded.SetYTitle("#it{p}_{T, jet}^{rec} (GeV/#it{c})")
        folded.Draw("text")
        cfeeddown_2d.SaveAs("%s/feeddown_folded.eps" % self.d_resultsallpdata)

        # subtract feed-down

        sideband_input_data_subtracted = \
            sideband_input_data.Clone("sideband_input_data_subtracted")
        sideband_input_data_subtracted.Add(folded, -1)

        # set negative yields to zero

        fileouts.cd()
        for ibin2 in range(self.p_nbin2_reco):
            for ibinshape in range(self.p_nbinshape_reco):
                if sideband_input_data_subtracted.GetBinContent( \
                sideband_input_data_subtracted.FindBin(self.lvarshape_binmin_reco[ibinshape], \
                self.lvar2_binmin_reco[ibin2])) < 0.0:
                    sideband_input_data_subtracted.SetBinContent( \
                        sideband_input_data_subtracted.FindBin( \
                            self.lvarshape_binmin_reco[ibinshape], \
                            self.lvar2_binmin_reco[ibin2]), 0.0)
        sideband_input_data_subtracted.Write()

        # plot the the yields before and after feed-down subtraction and the feed-down yields

        sideband_input_data_z = []
        sideband_input_data_subtracted_z = []
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                      self.lvar2_binmax_reco[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                      self.lvar2_binmax_reco[ibin2])
            sideband_input_data_z.append( \
                sideband_input_data.ProjectionX("sideband_input_data_z" + suffix,
                                                ibin2 + 1, ibin2 + 1, "e"))
            sideband_input_data_subtracted_z.append( \
                sideband_input_data_subtracted.ProjectionX( \
                "sideband_input_data_subtracted_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            cfeeddown = TCanvas("cfeeddown" + suffix, "cfeeddown" + suffix)
            setup_canvas(cfeeddown)
            logscale = True
            if logscale:
                cfeeddown.SetLogy()
            leg_feeddown = TLegend(.18, .70, .35, .85)
            setup_legend(leg_feeddown)
            setup_histogram(sideband_input_data_z[ibin2], get_colour(1), get_marker(0))
            leg_feeddown.AddEntry(sideband_input_data_z[ibin2], "prompt & non-prompt", "P")
            l_his = [sideband_input_data_z[ibin2], sideband_input_data_subtracted_z[ibin2], folded_z_list[ibin2]]
            y_min_h, y_max_h = get_y_window_his(l_his)
            y_min_0 = min([h.GetMinimum(0) for h in l_his])
            if logscale and y_min_h <= 0:
                y_min_h = y_min_0
                if y_max_h <= 0:
                    logscale = False
            y_margin_up = 0.27
            y_margin_down = 0.05
            sideband_input_data_z[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up, logscale))
            sideband_input_data_z[ibin2].SetTitle("")
            sideband_input_data_z[ibin2].SetXTitle(self.v_varshape_latex)
            sideband_input_data_z[ibin2].SetYTitle("yield")
            sideband_input_data_z[ibin2].Draw("")
            setup_histogram(folded_z_list[ibin2], get_colour(2), get_marker(1))
            leg_feeddown.AddEntry(folded_z_list[ibin2], "non-prompt (POWHEG)", "P")
            folded_z_list[ibin2].Draw("same")
            setup_histogram(sideband_input_data_subtracted_z[ibin2], get_colour(3), get_marker(2))
            leg_feeddown.AddEntry(sideband_input_data_subtracted_z[ibin2],
                                  "subtracted (prompt)", "P")
            sideband_input_data_subtracted_z[ibin2].Draw("same")
            # fileouts.cd()
            # sideband_input_data_subtracted_z[ibin2].Write()
            leg_feeddown.Draw("same")
            latex = TLatex(0.6, 0.8, "%g #leq %s < %g GeV/#it{c}" % \
                (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cfeeddown.SaveAs("%s/feeddown_subtraction_%s.eps" % \
                             (self.d_resultsallpdata, suffix_plot))
            cfeeddown.SaveAs("feeddown_subtraction_%s.png" % suffix_plot)
            #if self.feeddown_db:
            #    option = "feeddown"
            #    histo_to_compare = ("sideband_input_data_subtracted_z%s" % (suffix))
            #    xtitle = "feeddown subtracted"
            #    ytitle = self.v_var_binning
            #    self.makeratio_onedim(sideband_input_data_subtracted_z[ibin2], option, histo_to_compare, xtitle, ytitle )
            feeddown_fraction = folded_z_list[ibin2].Clone("feeddown_fraction" + suffix)
            feeddown_fraction_denominator = \
                sideband_input_data_z[ibin2].Clone("feeddown_denominator" + suffix)
            feeddown_fraction.Divide(feeddown_fraction_denominator)
            feeddown_fraction.Write()

            cfeeddown_fraction = TCanvas("cfeeddown_fraction" + suffix,
                                         "cfeeddown_fraction" + suffix)
            setup_canvas(cfeeddown_fraction)
            cfeeddown_fraction.SetLeftMargin(0.13)
            leg_fd_fraction = TLegend(.18, .8, .35, .85)
            setup_legend(leg_fd_fraction)
            setup_histogram(feeddown_fraction, get_colour(1), get_marker(0))
            y_min_h, y_max_h = get_y_window_his(feeddown_fraction)
            y_margin_up = 0.15
            y_margin_down = 0.05
            feeddown_fraction.GetYaxis().SetRangeUser(*get_plot_range(y_min_h, y_max_h, y_margin_down, y_margin_up))
            leg_fd_fraction.AddEntry(feeddown_fraction, "POWHEG based estimation", "P")
            feeddown_fraction.SetTitle("")
            feeddown_fraction.SetXTitle(self.v_varshape_latex)
            feeddown_fraction.SetYTitle("feed-down fraction")
            feeddown_fraction.SetTitleOffset(1.3, "Y")
            feeddown_fraction.Draw()
            leg_fd_fraction.Draw()
            latex = TLatex(0.6, 0.8, "%g #leq %s < %g GeV/#it{c}" % \
               (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex)
            cfeeddown_fraction.SaveAs("%s/feeddown_fraction_%s.eps" % \
                                      (self.d_resultsallpdata, suffix_plot))

        cfeeddown_output = TCanvas("cfeeddown_output", "cfeeddown_output")
        setup_canvas(cfeeddown_output)
        setup_histogram(sideband_input_data_subtracted)
        sideband_input_data_subtracted.SetTitle("")
        sideband_input_data_subtracted.SetXTitle(self.v_varshape_latex)
        sideband_input_data_subtracted.SetYTitle("%s (GeV/#it{c})" % self.p_latexbin2var)
        sideband_input_data_subtracted.Draw("text")
        cfeeddown_output.SaveAs("%s/feeddown_output.eps" % self.d_resultsallpdata)
        print("end of feed-down")

        #if self.feeddown_db:
        #    option = "feeddown"
        #    histo_to_compare = ("sideband_input_data_subtracted")
        #    print("Making ratio for", option, histo_to_compare )
        #    self.makeratio_twodim(sideband_input_data_subtracted, option, histo_to_compare)

    def append_histo(self, histo_list):
        histo = TH1F("Empty histo", "", 10, 1, 10)
        histo.FillRandom("gaus", 100)
        histo_list.append(histo)

    def unfolding(self):
        self.loadstyle()
        print("unfolding starts")

        fileouts = TFile.Open(self.file_unfold, "recreate")
        if not fileouts:
            self.logger.fatal(make_message_notfound(self.file_unfold))

        # get the feed-down output
        unfolding_input_data_file = TFile.Open(self.file_feeddown)
        if not unfolding_input_data_file:
            self.logger.fatal(make_message_notfound(self.file_feeddown))
        input_data = unfolding_input_data_file.Get("sideband_input_data_subtracted")
        if not input_data:
            self.logger.fatal(make_message_notfound("sideband_input_data_subtracted", self.file_feeddown))

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

        # Ignore the first bin for integration in case of untagged bin
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

        # all gen. level jets
        input_mc_gen = unfolding_input_file.Get("hzvsjetpt_gen_unmatched")
        if not input_mc_gen:
            self.logger.fatal(make_message_notfound("hzvsjetpt_gen_unmatched", self.n_fileresp))
        # rec. level cuts only applied
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

        # get the central prompt POWHEG histogram
        input_powheg = self.get_simulated_yields(self.powheg_path_prompt, 2, True)
        if not input_powheg:
            self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_prompt))
        input_powheg.SetName("fh2_prompt_%s" % self.v_varshape_binning)
        input_powheg_xsection = input_powheg.Clone(input_powheg.GetName() + "_xsec")

        # Ensure correct binning: x - shape, y - jet pt
        if not equal_binning_lists(input_powheg, list_x=self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(input_powheg, list_y=self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")
        # Ensure correct binning: x - shape, y - jet pt
        if not equal_binning_lists(input_powheg_xsection, list_x=self.varshaperanges_gen):
            self.logger.fatal("Error: Incorrect binning in x.")
        if not equal_binning_lists(input_powheg_xsection, list_y=self.var2ranges_gen):
            self.logger.fatal("Error: Incorrect binning in y.")

        # get the prompt POWHEG variations

        #input_powheg_sys = []
        #input_powheg_xsection_sys = []
        #for i_powheg in range(len(self.powheg_prompt_variations)):
            #path = "%s%s.root" % (self.powheg_prompt_variations_path, self.powheg_prompt_variations[i_powheg])
            #input_powheg_sys_i = self.get_simulated_yields(path, 2, True)
            #if not input_powheg_sys_i:
            #    self.logger.fatal(make_message_notfound("simulated yields", path))
            #input_powheg_sys_i.SetName("fh2_prompt_%s_%d" % (self.v_varshape_binning, i_powheg))
            #input_powheg_sys.append(input_powheg_sys_i)
            #input_powheg_xsection_sys_i = input_powheg_sys_i.Clone(input_powheg_sys_i.GetName() + "_xsec")
            #input_powheg_xsection_sys.append(input_powheg_xsection_sys_i)

        input_powheg_z = []
        input_powheg_xsection_z = []
        #input_powheg_sys_z = []
        #input_powheg_xsection_sys_z = []
        #tg_powheg = []
        #tg_powheg_xsection = []

        # get simulated distributions from PYTHIA 6 and POWHEG and calculate their spread

        for ibin2 in range(self.p_nbin2_gen):
            if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                self.append_histo(input_mc_gen_z)
                self.append_histo(input_powheg_z)
                self.append_histo(input_powheg_xsection_z)
                continue
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_mc_gen_z[ibin2].Scale(1.0 / input_mc_gen_z[ibin2].Integral(bin_int_first, input_mc_gen_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])), "width")
            input_powheg_z.append(input_powheg.ProjectionX("input_powheg_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_powheg_z[ibin2].Scale(1.0 / input_powheg_z[ibin2].Integral(bin_int_first, input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])), "width")
            input_powheg_xsection_z.append(input_powheg_xsection.ProjectionX("input_powheg_xsection_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_powheg_xsection_z[ibin2].Scale(1.0, "width")
            #input_powheg_sys_z_iter = []
            #input_powheg_xsection_sys_z_iter = []
            #for i_powheg in range(len(self.powheg_prompt_variations)):
            #    input_powheg_sys_z_iter.append(input_powheg_sys[i_powheg].ProjectionX("input_powheg_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
            #    input_powheg_sys_z_iter[i_powheg].Scale(1.0 / input_powheg_sys_z_iter[i_powheg].Integral(bin_int_first, input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[-1])), "width")
            #    input_powheg_xsection_sys_z_iter.append(input_powheg_xsection_sys[i_powheg].ProjectionX("input_powheg_xsection_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
            #    input_powheg_xsection_sys_z_iter[i_powheg].Scale(1.0, "width")
            #input_powheg_sys_z.append(input_powheg_sys_z_iter)
            #input_powheg_xsection_sys_z.append(input_powheg_xsection_sys_z_iter)
            #tg_powheg.append(tg_sys(input_powheg_z[ibin2], input_powheg_sys_z[ibin2]))
            #tg_powheg_xsection.append(tg_sys(input_powheg_xsection_z[ibin2], input_powheg_xsection_sys_z[ibin2]))

        for ibin2 in range(self.p_nbin2_reco):

            if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                self.append_histo(input_data_z)
                self.append_histo(mc_reco_matched_z)
                self.append_histo(mc_gen_matched_z)
                self.append_histo(mc_reco_gen_matched_z_ratio)
                self.append_histo(hz_genvsreco_list)
                continue
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            suffix_plot = "%s_%g_%g" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])

            input_data_z.append(input_data.ProjectionX("input_data_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))

            # compare shapes of distributions of reconstructed jets that pass rec. vs. gen. level cuts

            mc_reco_matched_z.append(mc_reco_matched.ProjectionX("mc_reco_matched_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            mc_reco_matched_z[ibin2].Scale(1.0 / mc_reco_matched_z[ibin2].Integral(bin_int_first, -1))
            mc_gen_matched_z.append(mc_gen_matched.ProjectionX("mc_det_matched_z" + suffix, mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]), mc_gen_matched.GetYaxis().FindBin(self.lvar2_binmin_reco[ibin2]), "e"))
            mc_gen_matched_z[ibin2].Scale(1.0 / mc_gen_matched_z[ibin2].Integral(bin_int_first, -1))
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
        text_pth_full = self.text_pth % (self.lpt_finbinmin[0], self.p_latexnhadron, self.lpt_finbinmax[-1], self.p_latexnhadron)
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
        for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
            latex = TLatex(self.x_latex, y_latex, text_latex)
            list_latex.append(latex)
            draw_latex(latex, textsize=self.fontsize, colour=1)
            y_latex -= self.y_step
        cz_genvsreco_full.Update()
        cz_genvsreco_full.SaveAs("%s/response_pr_%s_full.pdf" % (self.d_resultsallpdata, self.v_varshape_binning))
        cz_genvsreco_full.SaveAs("%s/%s_resp_pr_full.pdf" % (self.d_resultsallpdata, self.shape))

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
        for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
            latex = TLatex(self.x_latex, y_latex, text_latex)
            list_latex.append(latex)
            draw_latex(latex, textsize=self.fontsize, colour=1)
            y_latex -= self.y_step
        cz_genvsreco_full_real.Update()
        cz_genvsreco_full_real.SaveAs("%s/response_pr_%s_full_real.pdf" % (self.d_resultsallpdata, self.v_varshape_binning))
        cz_genvsreco_full_real.SaveAs("%s/%s_resp_pr_full_real.pdf" % (self.d_resultsallpdata, self.shape))

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
            if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                self.append_histo(kinematic_eff)
                self.append_histo(hz_gen_nocuts)
                continue
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
            if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                self.append_histo(hjetpt_fracdiff_list)
                continue
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff_list.append(unfolding_input_file.Get("hjetpt_fracdiff_prompt" + suffix))

        cjetpt_fracdiff = TCanvas("cjetpt_fracdiff", "prompt jetpt response fractional differences")
        setup_canvas(cjetpt_fracdiff)
        cjetpt_fracdiff.SetLogy()
        leg_jetpt_fracdiff = TLegend(.15, .5, .25, .8, "#it{p}_{T, jet}^{gen} (GeV/#it{c})")
        setup_legend(leg_jetpt_fracdiff)
        for ibin2 in range(self.p_nbin2_gen):
            if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                continue
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
                unfolded_zvsjetpt_final.Write()
                cunfolded_output.SaveAs("%s/unfolded_output.eps" % self.d_resultsallpdata)
                gStyle.SetPaintTextFormat("g")

                #if self.feeddown_db:
                #    option = "unfolding_results"
                #    histo_to_compare = ("unfolded_zvsjetpt_final")
                #    print("Making ratio for", option, histo_to_compare )
                #    self.makeratio_twodim(unfolded_zvsjetpt_final, option, histo_to_compare)

            for ibin2 in range(self.p_nbin2_gen):
                if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                    print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                    self.append_histo(unfolded_z_scaled_list_iter)
                    self.append_histo(unfolded_z_xsection_list_iter)
                    continue
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
            unfolded_jetpt_scaled.SetTitleOffset(1.2, "Y")
            unfolded_jetpt_scaled.Draw()
            latex = TLatex(0.2, 0.82, "%g #leq %s < %g" % (round(self.lvarshape_binmin_reco[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_reco[-1], 2)))
            draw_latex(latex)
            latex2 = TLatex(0.55, 0.82, "iteration %d" % (i + 1))
            draw_latex(latex2)
            cunfolded_jetpt.SaveAs("%s/unfolded_%s_%d.eps" % (self.d_resultsallpdata, self.v_var2_binning, i + 1))

            # refolding test for shape in jet pt bins

            refolded = folding(unfolded_zvsjetpt, response_matrix, input_data)
            for ibin2 in range(self.p_nbin2_reco):
                if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                    print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                    self.append_histo(refolding_test_list_iter)
                    continue
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
            if self.lpt_finbinmin[0] > self.lvar2_binmax_reco[ibin2]:
                print("Warning!!! HF_pt > jet_pt!!! Create random histo")
                self.append_histo(refolding_test_list)
                self.append_histo(refolding_test_jetpt_list)
                continue
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

            # plot the comparison of the final shape distribution, PYTHIA 6 and POWHEG

            cinput_mc_gen_z = TCanvas("cinput_mc_gen_z " + suffix, "1D gen pythia z")
            setup_canvas(cinput_mc_gen_z)
            leg_input_mc_gen_z = TLegend(.15, .75, .45, .85)
            setup_legend(leg_input_mc_gen_z)
            setup_histogram(input_mc_gen_z[ibin2], get_colour(2), get_marker(1))
            y_min_h, y_max_h = get_y_window_his([unfolded_z_scaled_list[i_iter_choice][ibin2], input_mc_gen_z[ibin2], input_powheg_z[ibin2]])
            #y_min_g, y_max_g = get_y_window_gr([tg_powheg[ibin2]])
            #y_min = min(y_min_g, y_min_h)
            #y_max = max(y_max_g, y_max_h)
            y_min = y_min_h
            y_max = y_max_h
            y_margin_up = 0.2
            y_margin_down = 0.05
            input_mc_gen_z[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
            input_mc_gen_z[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            input_mc_gen_z[ibin2].SetTitle("")
            input_mc_gen_z[ibin2].SetXTitle(self.v_varshape_latex)
            input_mc_gen_z[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_mc_gen_z[ibin2].Draw()
            setup_histogram(unfolded_z_scaled_list[i_iter_choice][ibin2], get_colour(1), get_marker(0))
            leg_input_mc_gen_z.AddEntry(unfolded_z_scaled_list[i_iter_choice][ibin2], "unfolded data", "P")
            unfolded_z_scaled_list[i_iter_choice][ibin2].Draw("same")
            leg_input_mc_gen_z.AddEntry(input_mc_gen_z[ibin2], "PYTHIA 6", "P")
            setup_histogram(input_powheg_z[ibin2], get_colour(3), get_marker(2))
            leg_input_mc_gen_z.AddEntry(input_powheg_z[ibin2], "POWHEG + PYTHIA 6", "P")
            input_powheg_z[ibin2].Draw("same")
            #setup_tgraph(tg_powheg[ibin2], get_colour(3))
            #tg_powheg[ibin2].Draw("5")
            leg_input_mc_gen_z.Draw("same")
            latex = TLatex(0.5, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            cinput_mc_gen_z.SaveAs("%s/unfolded_vs_mc_%s_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))
            #cinput_mc_gen_z.SaveAs("%s/unfolded_vs_mc_%s_%s.pdf" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

            # plot the comparison of the final cross-section and POWHEG

            cinput_mc_gen_z_xsection = TCanvas("cinput_mc_gen_z_xsection " + suffix, "1D gen pythia z xsection")
            setup_canvas(cinput_mc_gen_z_xsection)
            leg_input_mc_gen_z_xsection = TLegend(.15, .75, .45, .85)
            setup_legend(leg_input_mc_gen_z_xsection)
            setup_histogram(unfolded_z_xsection_list[i_iter_choice][ibin2], get_colour(1), get_marker(0))
            leg_input_mc_gen_z_xsection.AddEntry(unfolded_z_xsection_list[i_iter_choice][ibin2], "unfolded data", "P")
            y_min_h, y_max_h = get_y_window_his([unfolded_z_xsection_list[i_iter_choice][ibin2], input_powheg_xsection_z[ibin2]])
            #y_min_g, y_max_g = get_y_window_gr(tg_powheg_xsection[ibin2])
            #y_min = min(y_min_g, y_min_h)
            #y_max = max(y_max_g, y_max_h)
            y_min = y_min_h
            y_max = y_max_h
            y_margin_up = 0.2
            y_margin_down = 0.05
            unfolded_z_xsection_list[i_iter_choice][ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
            unfolded_z_xsection_list[i_iter_choice][ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            unfolded_z_xsection_list[i_iter_choice][ibin2].SetTitle("")
            unfolded_z_xsection_list[i_iter_choice][ibin2].SetXTitle(self.v_varshape_latex)
            unfolded_z_xsection_list[i_iter_choice][ibin2].SetYTitle("d#it{#sigma}/d%s (mb)" % self.v_varshape_latex)
            unfolded_z_xsection_list[i_iter_choice][ibin2].GetYaxis().SetMaxDigits(3)
            unfolded_z_xsection_list[i_iter_choice][ibin2].Draw()
            setup_histogram(input_powheg_xsection_z[ibin2], get_colour(3), get_marker(2))
            leg_input_mc_gen_z_xsection.AddEntry(input_powheg_xsection_z[ibin2], "POWHEG + PYTHIA 6", "P")
            input_powheg_xsection_z[ibin2].Draw("same")
            #setup_tgraph(tg_powheg_xsection[ibin2], get_colour(3))
            #tg_powheg_xsection[ibin2].Draw("5")
            latex = TLatex(0.5, 0.82, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_gen[ibin2], self.p_latexbin2var, self.lvar2_binmax_gen[ibin2]))
            draw_latex(latex)
            leg_input_mc_gen_z_xsection.Draw("same")
            cinput_mc_gen_z_xsection.SaveAs("%s/unfolded_vs_mc_%s_xsection_%s.eps" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))
            #cinput_mc_gen_z_xsection.SaveAs("%s/unfolded_vs_mc_%s_xsection_%s.pdf" % (self.d_resultsallpdata, self.v_varshape_binning, suffix_plot))

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
            input_data_z_scaled.Scale(1.0 / input_data_z_scaled.Integral(bin_int_first, -1), "width")
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
            cunfolded_not_z.SaveAs("unfolded_not_%s_%s.png" % (self.v_varshape_binning, suffix_plot))

          #  if self.feeddown_db:
          #      option = "unfolding_results"
          #      histo_to_compare = ("unfolded_z_%d_%s" % (i_iter_choice, suffix))
          #      print("Making ratio for", option, histo_to_compare)
          #      xtitle = ""
          #      ytitle = ""
          #      self.makeratio_onedim(unfolded_z_scaled_list[i_iter_choice][ibin_jetpt], option, histo_to_compare, xtitle, ytitle)
          #      #Lc tot D0 ratio

            if self.feeddown_ratio:
                option = "unfolding_results"
                lchistoname = ("unfolded_z_%d_%s" % (i_iter_choice, suffix))
                print("Making Lc to D0 ratio for", option, lchistoname)
                self.makeratio(unfolded_z_scaled_list[i_iter_choice][ibin_jetpt], option, lchistoname)
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
                unfolded_jetpt_scaled_list[i].SetTitleOffset(1.2, "Y")
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

        # Ignore the first bin for integration in case of untagged bin
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
            input_mc_det_z[ibin2].Scale(1.0 / input_mc_det_z[ibin2].Integral(bin_int_first, -1))

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_mc_gen_z.append(input_mc_gen.ProjectionX("input_mc_gen_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_mc_gen_z[ibin2].Scale(1.0 / input_mc_gen_z[ibin2].Integral(bin_int_first, -1))
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
                unfolded_z.Scale(1.0 / unfolded_z.Integral(bin_int_first, -1))
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

        input_powheg = self.get_simulated_yields(self.powheg_path_prompt, 2, True)
        if not input_powheg:
            self.logger.fatal(make_message_notfound("simulated yields", self.powheg_path_prompt))
        input_powheg.SetName("fh2_prompt_%s" % self.v_varshape_binning)
        input_powheg_xsection = input_powheg.Clone(input_powheg.GetName() + "_xsec")

        # get the prompt POWHEG variations and calculate their spread

        input_powheg_sys = []
        input_powheg_xsection_sys = []
        for i_powheg in range(len(self.powheg_prompt_variations)):
            path = "%s%s.root" % (self.powheg_prompt_variations_path, self.powheg_prompt_variations[i_powheg])
            input_powheg_sys_i = self.get_simulated_yields(path, 2, True)
            if not input_powheg_sys_i:
                self.logger.fatal(make_message_notfound("simulated yields", path))
            input_powheg_sys_i.SetName("fh2_prompt_%s_%d" % (self.v_varshape_binning, i_powheg))
            input_powheg_sys.append(input_powheg_sys_i)
            input_powheg_xsection_sys_i = input_powheg_sys_i.Clone(input_powheg_sys_i.GetName() + "_xsec")
            input_powheg_xsection_sys.append(input_powheg_xsection_sys_i)
        input_powheg_z = []
        input_powheg_xsection_z = []
        input_powheg_sys_z = []
        input_powheg_xsection_sys_z = []
        tg_powheg = []
        tg_powheg_xsection = []
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            input_powheg_z.append(input_powheg.ProjectionX("input_powheg_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_powheg_z[ibin2].Scale(1.0 / input_powheg_z[ibin2].Integral(input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[0]), input_powheg_z[ibin2].FindBin(self.lvarshape_binmin_reco[-1])), "width")
            input_powheg_xsection_z.append(input_powheg_xsection.ProjectionX("input_powheg_xsection_z" + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_powheg_xsection_z[ibin2].Scale(1.0, "width")
            input_powheg_sys_z_iter = []
            input_powheg_xsection_sys_z_iter = []
            for i_powheg in range(len(self.powheg_prompt_variations)):
                input_powheg_sys_z_iter.append(input_powheg_sys[i_powheg].ProjectionX("input_powheg_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
                input_powheg_sys_z_iter[i_powheg].Scale(1.0 / input_powheg_sys_z_iter[i_powheg].Integral(input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[0]), input_powheg_sys_z_iter[i_powheg].FindBin(self.lvarshape_binmin_reco[-1])), "width")
                input_powheg_xsection_sys_z_iter.append(input_powheg_xsection_sys[i_powheg].ProjectionX("input_powheg_xsection_sys_z"+self.powheg_prompt_variations[i_powheg]+suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_powheg_sys_z.append(input_powheg_sys_z_iter)
            input_powheg_xsection_sys_z.append(input_powheg_xsection_sys_z_iter)
            tg_powheg.append(tg_sys(input_powheg_z[ibin2], input_powheg_sys_z[ibin2]))
            tg_powheg_xsection.append(tg_sys(input_powheg_xsection_z[ibin2], input_powheg_xsection_sys_z[ibin2]))

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

        # write the combined systematic uncertainties in a file

        file_sys_out = TFile.Open("%s/systematics_results.root" % self.d_resultsallpdata, "recreate")
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            tgsys[ibin2].Write("tgsys_%s" % suffix)
        file_sys_out.Close()

        # relative statistical uncertainty of the central values

        h_default_stat_err = []
        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            h_default_stat_err.append(input_histograms_default[ibin2].Clone("h_default_stat_err" + suffix))
            for i in range(h_default_stat_err[ibin2].GetNbinsX()):
                h_default_stat_err[ibin2].SetBinContent(i + 1, 0)
                h_default_stat_err[ibin2].SetBinError(i + 1, input_histograms_default[ibin2].GetBinError(i + 1) / input_histograms_default[ibin2].GetBinContent(i + 1))

        # get the prompt PYTHIA histograms

        file_sim_out = TFile.Open("%s/simulations.root" % self.d_resultsallpdata, "recreate")
        input_pythia8 = []
        input_pythia8_xsection = []
        input_pythia8_z = []
        input_pythia8_xsection_z = []
        for i_pythia8 in range(len(self.pythia8_prompt_variations)):
            path = "%s%s.root" % (self.pythia8_prompt_variations_path, self.pythia8_prompt_variations[i_pythia8])
            input_pythia8_i = self.get_simulated_yields(path, 2, True)
            if not input_pythia8_i:
                self.logger.fatal(make_message_notfound("simulated yields", path))
            input_pythia8_i.SetName("fh2_pythia_prompt_%s_%d" % (self.v_varshape_binning, i_pythia8))
            input_pythia8.append(input_pythia8_i)
            input_pythia8_xsection_i = input_pythia8_i.Clone(input_pythia8_i.GetName() + "_xsec")
            input_pythia8_xsection.append(input_pythia8_xsection_i)

            # Ensure correct binning: x - shape, y - jet pt
            if not equal_binning_lists(input_pythia8[i_pythia8], list_x=self.varshaperanges_gen):
                self.logger.fatal("Error: Incorrect binning in x.")
            if not equal_binning_lists(input_pythia8[i_pythia8], list_y=self.var2ranges_gen):
                self.logger.fatal("Error: Incorrect binning in y.")
            if not equal_binning_lists(input_pythia8_xsection[i_pythia8], list_x=self.varshaperanges_gen):
                self.logger.fatal("Error: Incorrect binning in x.")
            if not equal_binning_lists(input_pythia8_xsection[i_pythia8], list_y=self.var2ranges_gen):
                self.logger.fatal("Error: Incorrect binning in y.")

            input_pythia8_z_jetpt = []
            input_pythia8_xsection_z_jetpt = []
            for ibin2 in range(self.p_nbin2_gen):
                suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                input_pythia8_z_jetpt.append(input_pythia8[i_pythia8].ProjectionX("input_pythia8" + self.pythia8_prompt_variations[i_pythia8]+suffix, ibin2 + 1, ibin2 + 1, "e"))
                input_pythia8_z_jetpt[ibin2].Scale(1.0 / input_pythia8_z_jetpt[ibin2].Integral(), "width")
                pythia8_out = input_pythia8_z_jetpt[ibin2]
                file_sim_out.cd()
                pythia8_out.Write()
                pythia8_out.SetDirectory(0)
                input_pythia8_xsection_z_jetpt.append(input_pythia8_xsection[i_pythia8].ProjectionX("input_pythia8_xsection" + self.pythia8_prompt_variations[i_pythia8] + suffix, ibin2 + 1, ibin2 + 1, "e"))
            input_pythia8_z.append(input_pythia8_z_jetpt)
            input_pythia8_xsection_z.append(input_pythia8_xsection_z_jetpt)
        file_sim_out.Close()

        for ibin2 in range(self.p_nbin2_gen):

            # plot the results with systematic uncertainties

            suffix = "%s_%g_%g" % (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            cfinalwsys = TCanvas("cfinalwsys " + suffix, "final result with systematic uncertainties" + suffix)
            setup_canvas(cfinalwsys)
            leg_finalwsys = TLegend(.7, .75, .85, .85)
            setup_legend(leg_finalwsys)
            leg_finalwsys.AddEntry(input_histograms_default[ibin2], "data", "P")
            setup_histogram(input_histograms_default[ibin2], get_colour(0, 0))
            y_min_g, y_max_g = get_y_window_gr([tgsys[ibin2]])
            y_min_h, y_max_h = get_y_window_his([input_histograms_default[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.4
            y_margin_down = 0.05
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw("")
            setup_tgraph(tgsys[ibin2], get_colour(7, 0))
            tgsys[ibin2].Draw("5")
            leg_finalwsys.AddEntry(tgsys[ibin2], "syst. unc.", "F")
            input_histograms_default[ibin2].Draw("AXISSAME")
            #PREL latex = TLatex(0.15, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.15, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.15, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.15, 0.72, "%g #leq %s < %g GeV/#it{c}, #left|#it{#eta}_{jet}#right| #leq 0.5" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            latex3 = TLatex(0.15, 0.67, "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, #left|#it{y}_{%s}#right| #leq 0.8" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2]), self.p_latexnhadron))
            draw_latex(latex3)
            leg_finalwsys.Draw("same")
            latex_SD = TLatex(0.15, 0.62, "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)")
            draw_latex(latex_SD)
            cfinalwsys.SaveAs("%s/final_wsys_%s.pdf" % (self.d_resultsallpdata, suffix))

            # plot the results with systematic uncertainties and models

            leg_pos = [.55, .65, .8, .85]

            cfinalwsys_wmodels = TCanvas("cfinalwsys_wmodels " + suffix, "final result with systematic uncertainties with models" + suffix)
            setup_canvas(cfinalwsys_wmodels)
#            if self.typean == "jet_zg":
            leg_finalwsys_wmodels = TLegend(*leg_pos)
#            elif self.typean == "jet_rg":
#                leg_finalwsys_wmodels = TLegend(.15, .45, .25, .65)
#            else:
#                leg_finalwsys_wmodels = TLegend(.55, .52, .65, .72)
            setup_legend(leg_finalwsys_wmodels)
            leg_finalwsys_wmodels.AddEntry(input_histograms_default[ibin2], "data", "P")
            setup_histogram(input_histograms_default[ibin2], get_colour(0))
            y_min_g, y_max_g = get_y_window_gr([tgsys[ibin2], tg_powheg[ibin2]])
            y_min_h, y_max_h = get_y_window_his([input_histograms_default[ibin2], input_powheg_z[ibin2]] + \
                [input_pythia8_z[i][ibin2] for i in range(len(self.pythia8_prompt_variations))])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.4
            y_margin_down = 0.05
            y_plot_min, y_plot_max = get_plot_range(y_min, y_max, y_margin_down, y_margin_up)
            input_histograms_default[ibin2].GetYaxis().SetRangeUser(y_plot_min, y_plot_max)
            input_histograms_default[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_gen[0], 2), round(self.lvarshape_binmax_gen[-1], 2))
            input_histograms_default[ibin2].SetTitle("")
            input_histograms_default[ibin2].SetXTitle(self.v_varshape_latex)
            input_histograms_default[ibin2].SetYTitle("1/#it{N}_{jets} d#it{N}/d%s" % self.v_varshape_latex)
            input_histograms_default[ibin2].Draw()
            setup_tgraph(tgsys[ibin2], get_colour(7, 0))
            tgsys[ibin2].Draw("5")
            leg_finalwsys_wmodels.AddEntry(tgsys[ibin2], "syst. unc.", "F")
            setup_histogram(input_powheg_z[ibin2], get_colour(1, 0), get_marker(1))
            leg_finalwsys_wmodels.AddEntry(input_powheg_z[ibin2], "POWHEG #plus PYTHIA 6", "P")
            input_powheg_z[ibin2].Draw("same")
            setup_tgraph(tg_powheg[ibin2], get_colour(1))
            tg_powheg[ibin2].Draw("5")
            for i_pythia8 in range(len(self.pythia8_prompt_variations)):
                setup_histogram(input_pythia8_z[i_pythia8][ibin2], get_colour(i_pythia8 + 2), get_marker(i_pythia8 + 2), 2.)
                leg_finalwsys_wmodels.AddEntry(input_pythia8_z[i_pythia8][ibin2], self.pythia8_prompt_variations_legend[i_pythia8], "P")
                input_pythia8_z[i_pythia8][ibin2].Draw("same")
            input_histograms_default[ibin2].Draw("AXISSAME")

            latex = TLatex(0.15, 0.82, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            #latex = TLatex(0.15, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.15, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.15, 0.72, "%g #leq %s < %g GeV/#it{c}, #left|#it{#eta}_{jet}#right| #leq 0.5" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            latex3 = TLatex(0.15, 0.67, "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, #left|#it{y}_{%s}#right| #leq 0.8" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2]), self.p_latexnhadron))
            draw_latex(latex3)
            latex_SD = TLatex(0.15, 0.62, "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)")
            draw_latex(latex_SD)
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

            text_ptjet_full = self.text_ptjet % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2])
            text_pth_full = self.text_pth % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2]), self.p_latexnhadron)

            list_obj = [tgsys[ibin2], tg_powheg[ibin2], input_histograms_default[ibin2], input_powheg_z[ibin2]]
            labels_obj = ["data", "POWHEG #plus PYTHIA 6", "", ""]
            colours = [get_colour(i, j) for i, j in zip((0, 1, 0, 1), (2, 2, 1, 1))]
            markers = [get_marker(i) for i in (0, 1, 0, 1)]
            for i_pythia8 in range(len(self.pythia8_prompt_variations)):
                list_obj.append(input_pythia8_z[i_pythia8][ibin2])
                labels_obj.append(self.pythia8_prompt_variations_legend[i_pythia8])
                colours.append(get_colour(i_pythia8 + 2))
                markers.append(get_marker(i_pythia8 + 2))
            cfinalwsys_wmodels_new, _ = make_plot("cfinalwsys_wmodels_new_" + suffix, \
                list_obj=list_obj, labels_obj=labels_obj, opt_leg_g="FP", opt_plot_g="2", \
                colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[0.05, 0.4], \
                title=";%s;1/#it{N}_{jets} d#it{N}/d%s" % (self.v_varshape_latex, self.v_varshape_latex))
            for gr, c in zip((tgsys[ibin2], tg_powheg[ibin2]), (0, 1)):
                gr.SetMarkerColor(get_colour(c))
            if self.typean == "jet_rg":
                cfinalwsys_wmodels_new.SetTickx(0)
                axis_thetag.Draw("same")
            # Draw LaTeX
            y_latex = 0.83
            list_text = [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full]
            if self.shape in ("zg", "rg", "nsd"):
                list_text.append(self.text_sd)
            list_latex = []
            for text_latex in list_text:
                latex = TLatex(self.x_latex, y_latex, text_latex)
                list_latex.append(latex)
                draw_latex(latex, textsize=0.03)
                y_latex -= self.y_step
            cfinalwsys_wmodels_new.SaveAs("%s/final_wsys_wmodels_%s_new.pdf" % (self.d_resultsallpdata, suffix))

            # plot the relative systematic uncertainties for all categories together

            # preliminary figure
            crelativesys = TCanvas("crelativesys " + suffix, "relative systematic uncertainties" + suffix)
            gStyle.SetErrorX(0)
            setup_canvas(crelativesys)
            crelativesys.SetCanvasSize(900, 800)
            crelativesys.SetBottomMargin(self.margins_can[0])
            crelativesys.SetLeftMargin(self.margins_can[1])
            crelativesys.SetTopMargin(self.margins_can[2])
            crelativesys.SetRightMargin(self.margins_can[3])
            leg_relativesys = TLegend(.68, .6, .88, .91)
            setup_legend(leg_relativesys, textsize=self.fontsize)
            y_min_g, y_max_g = get_y_window_gr(tgsys_cat[ibin2])
            y_min_h, y_max_h = get_y_window_his([h_default_stat_err[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.42
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
            for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd]:
                latex = TLatex(self.x_latex, y_latex, text_latex)
                list_latex.append(latex)
                draw_latex(latex, textsize=self.fontsize)
                y_latex -= self.y_step
            leg_relativesys.Draw("same")
            crelativesys.SaveAs("%s/sys_unc_%s.eps" % (self.d_resultsallpdata, suffix))
            if ibin2 == 1:
                crelativesys.SaveAs("%s/%s_sys_unc_%s.pdf" % (self.d_resultsallpdata, self.shape, suffix))
            gStyle.SetErrorX(0.5)

        # plot the feed-down fraction with systematic uncertainties from POWHEG

        file_feeddown = TFile.Open(self.file_feeddown)
        if not file_feeddown:
            self.logger.fatal(make_message_notfound(self.file_feeddown))
        file_feeddown_variations = []
        for i_powheg, varname in enumerate(self.powheg_nonprompt_varnames):
            path = self.file_feeddown.replace(string_default, "feeddown/" + varname)
            file_feeddown_variations.append(TFile.Open(path))
            if not file_feeddown_variations[i_powheg]:
                self.logger.fatal(make_message_notfound(path))
        h_feeddown_fraction = [] # list of the central feed-down fractions for all pt_jet bins
        h_feeddown_fraction_variations = [] # list of feed-down fractions for all POWHEG variations and all pt_jet bins
        tg_feeddown_fraction = [] # list of graphs with the spread of values for all pt_jet bins
        for ibin2 in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
              (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            suffix_plot = "%s_%g_%g" % \
              (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            h_feeddown_fraction_variations_niter = [] # list of feed-down fractions for all POWHEG variations in a given pt_jet bin
            h_feeddown_fraction.append(file_feeddown.Get("feeddown_fraction" + suffix))
            for i_powheg in range(len(self.powheg_nonprompt_varnames)):
                h_feeddown_fraction_variations_niter.append(file_feeddown_variations[i_powheg].Get("feeddown_fraction" + suffix))

            h_feeddown_fraction_variations.append(h_feeddown_fraction_variations_niter)
            # get the graph with the spread of values for all the POWHEG variations
            tg_feeddown_fraction.append(tg_sys(h_feeddown_fraction[ibin2], h_feeddown_fraction_variations[ibin2]))

            cfeeddown_fraction = TCanvas("cfeeddown_fraction " + suffix, "feeddown fraction" + suffix)
            setup_canvas(cfeeddown_fraction)
            cfeeddown_fraction.SetLeftMargin(0.13)
            leg_fd = TLegend(.67, .6, .85, .85)
            setup_legend(leg_fd, 0.025)
            setup_histogram(h_feeddown_fraction[ibin2], get_colour(0))
            y_min_g, y_max_g = get_y_window_gr([tg_feeddown_fraction[ibin2]])
            y_min_h, y_max_h = get_y_window_his([h_feeddown_fraction[ibin2]])
            y_min = min(y_min_g, y_min_h)
            y_max = max(y_max_g, y_max_h)
            y_margin_up = 0.45
            y_margin_down = 0.05
            h_feeddown_fraction[ibin2].GetYaxis().SetRangeUser(*get_plot_range(y_min, y_max, y_margin_down, y_margin_up))
            h_feeddown_fraction[ibin2].GetXaxis().SetRangeUser(round(self.lvarshape_binmin_reco[0], 2), round(self.lvarshape_binmax_reco[-1], 2))
            h_feeddown_fraction[ibin2].SetXTitle(self.v_varshape_latex)
            h_feeddown_fraction[ibin2].SetYTitle("feed-down fraction")
            h_feeddown_fraction[ibin2].SetTitleOffset(1.4, "Y")
            h_feeddown_fraction[ibin2].SetTitle("")
            h_feeddown_fraction[ibin2].Draw("same")
            #tg_feeddown_fraction[ibin2].Draw("5")
            leg_fd.AddEntry(h_feeddown_fraction[ibin2], "default", "P")
            for i, his in enumerate(h_feeddown_fraction_variations_niter):
                setup_histogram(his, get_colour(i + 2, 0), 1)
                leg_fd.AddEntry(his, self.powheg_nonprompt_varlabels[i], "L")
                his.Draw("samehist")
            setup_tgraph(tg_feeddown_fraction[ibin2], get_colour(1))
            h_feeddown_fraction[ibin2].Draw("same")
            h_feeddown_fraction[ibin2].Draw("axissame")
            leg_fd.Draw("same")
            #PREL latex = TLatex(0.18, 0.85, "ALICE Preliminary, pp, #sqrt{#it{s}} = 13 TeV")
            latex = TLatex(0.18, 0.82, "pp, #sqrt{#it{s}} = 13 TeV")
            draw_latex(latex)
            latex1 = TLatex(0.18, 0.77, "%s in charged jets, anti-#it{k}_{T}, #it{R} = 0.4, #left|#it{#eta}_{jet}#right| #leq 0.5" % self.p_latexnhadron)
            draw_latex(latex1)
            latex2 = TLatex(0.18, 0.72, "%g #leq %s < %g GeV/#it{c}" % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2]))
            draw_latex(latex2)
            #latex3 = TLatex(0.18, 0.7, "%g #leq %s < %g" % (round(self.lvarshape_binmin_reco[0], 2), self.v_varshape_latex, round(self.lvarshape_binmax_reco[-1], 2)))
            latex3 = TLatex(0.18, 0.67, "%g #leq #it{p}_{T, %s} < %g GeV/#it{c}" % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2])))
            draw_latex(latex3)
            latex5 = TLatex(0.18, 0.62, "stat. unc. from data")
            draw_latex(latex5)
            latex6 = TLatex(0.18, 0.57, "syst. unc. from POWHEG #plus PYTHIA 6")
            draw_latex(latex6)
            #latex7 = TLatex(0.65, 0.75, "POWHEG based")
            #draw_latex(latex7)
            cfeeddown_fraction.SaveAs("%s/feeddown_fraction_var_%s.eps" % (self.d_resultsallpdata, suffix_plot))
            cfeeddown_fraction.SaveAs("%s/feeddown_fraction_var_%s.pdf" % (self.d_resultsallpdata, suffix_plot))

            text_ptjet_full = self.text_ptjet % (self.lvar2_binmin_reco[ibin2], self.p_latexbin2var, self.lvar2_binmax_reco[ibin2])
            text_pth_full = self.text_pth % (self.lpt_finbinmin[0], self.p_latexnhadron, min(self.lpt_finbinmax[-1], self.lvar2_binmax_reco[ibin2]), self.p_latexnhadron)

            # preliminary figure
            if ibin2 == 1:
                gStyle.SetErrorX(0)
                leg_pos = [.16, .545, .31, .595]
                list_obj = [tg_feeddown_fraction[ibin2], h_feeddown_fraction[ibin2]]
                labels_obj = ["POWHEG uncertainty"]
                colours = [get_colour(i, j) for i, j in zip((1, 0), (2, 1))]
                markers = [get_marker(0)]
                y_margin_up = 0.5
                y_margin_down = 0.05
                c_fd_fr_sys, list_obj_data_new = make_plot("c_fd_fr_sys_" + suffix, size=self.size_can, \
                    list_obj=list_obj, labels_obj=labels_obj, opt_leg_g="F", opt_plot_g=self.opt_plot_g, offsets_xy=self.offsets_axes, \
                    colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=self.margins_can, \
                    title=";%s;feed-down fraction" % self.title_x)
                tg_feeddown_fraction[ibin2].SetMarkerColor(get_colour(1))
                tg_feeddown_fraction[ibin2].GetYaxis().SetTitleOffset(1.2)
                list_obj_data_new[0].SetTextSize(self.fontsize)
                if self.shape == "nsd":
                    tg_feeddown_fraction[ibin2].GetXaxis().SetNdivisions(5)
                    shrink_err_x(tg_feeddown_fraction[ibin2])
                c_fd_fr_sys.Update()
                # Draw LaTeX
                y_latex = self.y_latex_top
                list_latex = []
                for text_latex in [self.text_alice, self.text_jets, text_ptjet_full, text_pth_full, self.text_sd, self.text_powheg]:
                    latex = TLatex(self.x_latex, y_latex, text_latex)
                    list_latex.append(latex)
                    draw_latex(latex, textsize=self.fontsize)
                    y_latex -= self.y_step
                c_fd_fr_sys.Update()
                c_fd_fr_sys.SaveAs("%s/%s_fd_fr_sys_%s.pdf" % (self.d_resultsallpdata, self.shape, suffix_plot))
                gStyle.SetErrorX(0.5)

    def get_simulated_yields(self, file_path: str, dim: int, prompt: bool):
        """Create a histogram from a simulation tree.
        file_path - input file path
        dim - dimension of the output histogram: 2, 3
        prompt - prompt or non-prompt: True, False"""
        print("Starting the histogram extraction from an MC tree\nInput file: %s" % file_path)

        if dim not in (2, 3):
            self.logger.fatal("Error: %d is not a supported dimension.", dim)

        # Get the normalisation factor (inverse integrated luminosity).
        file_sim = TFile.Open(file_path)
        if not file_sim:
            self.logger.fatal(make_message_notfound(file_path))
        pr_xsec = file_sim.Get("fHistXsection")
        if not pr_xsec:
            self.logger.fatal(make_message_notfound("fHistXsection", file_path))
        scale_factor = pr_xsec.GetBinContent(1) / pr_xsec.GetEntries()
        file_sim.Close()

        # Load the tree.
        if "D0" in self.case:
            tree_name = "tree_D0"
            print("Loading the D0 tree")
        elif "Lc" in self.case:
            tree_name = "tree_Lc"
            print("Loading the Lc tree")
        else:
            self.logger.fatal(make_message_notfound("the particle name", self.case))
        tree_sim = uproot.open(file_path)[tree_name]
        if not tree_sim:
            self.logger.fatal(make_message_notfound(tree_name, file_path))

        print("Converting")
        # Convert it into a dataframe.
        list_branches = ["pt_cand", "eta_cand", "phi_cand", "y_cand", "pdg_parton", "pt_jet", \
            "eta_jet", "phi_jet", "delta_r_jet", "z", "n_const", "zg_jet", "rg_jet", "nsd_jet", \
            "k0_jet", "k1_jet", "k2_jet", "kT_jet"]
            #"Pt_splitting_jet", "Pt_mother_jet", \
        try:
            df_sim = tree_sim.pandas.df(branches=list_branches)
        except Exception: # pylint: disable=broad-except
            self.logger.fatal(make_message_notfound("variables", tree_name))

        # Adjust nSD values.
        df_sim = adjust_nsd(df_sim)
        # Adjust z values.
        df_sim = adjust_z(df_sim)

        print("Entries in the tree:", len(df_sim))
        print("Filtering %sprompt hadrons" % ("" if prompt else "non-"))
        # Apply the same cuts as in gen MC.
        # cut on jet pt
        df_sim = seldf_singlevar(df_sim, self.v_var2_binning, self.lvar2_binmin_gen[0], self.lvar2_binmax_gen[-1])
        # cut on hadron pt
        df_sim = seldf_singlevar(df_sim, self.v_var_binning, self.lpt_finbinmin[0], self.lpt_finbinmax[-1])
        # cut on shape
        df_sim = seldf_singlevar(df_sim, self.v_varshape_binning, self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1])
        # acceptance cut
        df_sim = df_sim.query(self.s_jetsel_sim)
        # pt-dependent rapidity cut
        sel_cand_array = selectfidacc(df_sim["pt_cand"].values, df_sim["y_cand"].values)
        df_sim = df_sim[np.array(sel_cand_array, dtype=bool)]
        # prompt vs. non-prompt selection
        pdg_parton_good = 4 if prompt else 5
        df_sim = df_sim[df_sim["pdg_parton"] == pdg_parton_good]

        print("Entries after filtering:", len(df_sim))
        # Create, fill and scale the histogram.
        print("Filling a %dD histogram" % dim)
        if dim == 2:
            # Binning: x - shape, y - jet pt
            his2 = makefill2dhist(df_sim, "h2_yield_sim", \
                self.varshapebinarray_gen, self.var2binarray_gen, \
                self.v_varshape_binning, self.v_var2_binning)
            print("Integral of the histogram:", his2.Integral())
            print("Scaling with:", scale_factor)
            his2.Scale(scale_factor)
            print("Entries in the histogram:", his2.GetEntries())
            print("Returning")
            return his2
        if dim == 3:
            # Binning: x - shape, y - jet pt, z - pt hadron
            his3 = makefill3dhist(df_sim, "h3_yield_sim", \
                self.varshapebinarray_gen, self.var2binarray_gen, self.var1binarray, \
                self.v_varshape_binning, self.v_var2_binning, self.v_var_binning)
            print("Integral of the histogram:", his3.Integral())
            print("Scaling with:", scale_factor)
            his3.Scale(scale_factor)
            print("Entries in the histogram:", his3.GetEntries())
            print("Returning")
            return his3
        return None
