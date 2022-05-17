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
import argparse
from array import array
from cmath import nan
import yaml
# pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TLatex, TLine, TGaxis, gROOT, gStyle, TCanvas, TGraphAsymmErrors, TGraphErrors, TGraph, TLegend
from machine_learning_hep.utilities import make_message_notfound
from machine_learning_hep.utilities import get_colour, get_marker, draw_latex
from machine_learning_hep.utilities import make_plot, get_y_window_his, get_y_window_gr, get_plot_range, divide_graphs, scale_graph, setup_legend
from machine_learning_hep.logger import get_logger


def main(): # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    """
    Main plotting function
    """
    gROOT.SetBatch(True)

    do_ivan = True

    # pylint: disable=unused-variable

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-analysis", "-d", dest="database_analysis",
                        help="analysis database to be used", required=True)
    parser.add_argument("--analysis", "-a", dest="type_ana",
                        help="choose type of analysis", required=True)
    parser.add_argument("--input", "-i", dest="input_file",
                        help="results input file", required=True)

    args = parser.parse_args()

    typean = args.type_ana
    shape = typean[len("jet_"):]
    print("Shape:", shape)
    if shape != "zg":
        do_ivan = False

    file_in = args.input_file
    with open(args.database_analysis, "r") as file_db:
        data_param = yaml.safe_load(file_db)
    case = list(data_param.keys())[0]
    datap = data_param[case]

    logger = get_logger()

    i_cut = file_in.rfind("/")
    rootpath = file_in[:i_cut]

    # plotting
    # LaTeX string
    p_latexnhadron = datap["analysis"][typean]["latexnamehadron"]
    p_latexbin2var = datap["analysis"][typean]["latexbin2var"]
    v_varshape_latex = datap["analysis"][typean]["var_shape_latex"]

    # first variable (hadron pt)
    lpt_finbinmin = datap["analysis"][typean]["sel_an_binmin"]
    lpt_finbinmax = datap["analysis"][typean]["sel_an_binmax"]
    var1ranges = lpt_finbinmin.copy()
    var1ranges.append(lpt_finbinmax[-1])

    # second variable (jet pt)
    v_var2_binning = datap["analysis"][typean]["var_binning2"] # name
    lvar2_binmin_reco = datap["analysis"][typean].get("sel_binmin2_reco", None)
    lvar2_binmax_reco = datap["analysis"][typean].get("sel_binmax2_reco", None)
    p_nbin2_reco = len(lvar2_binmin_reco) # number of reco bins
    lvar2_binmin_gen = datap["analysis"][typean].get("sel_binmin2_gen", None)
    lvar2_binmax_gen = datap["analysis"][typean].get("sel_binmax2_gen", None)
    p_nbin2_gen = len(lvar2_binmin_gen) # number of gen bins
    var2ranges_reco = lvar2_binmin_reco.copy()
    var2ranges_reco.append(lvar2_binmax_reco[-1])
    var2binarray_reco = array("d", var2ranges_reco) # array of bin edges to use in histogram constructors
    var2ranges_gen = lvar2_binmin_gen.copy()
    var2ranges_gen.append(lvar2_binmax_gen[-1])
    var2binarray_gen = array("d", var2ranges_gen) # array of bin edges to use in histogram constructors

    # observable (z, shape,...)
    v_varshape_binning = datap["analysis"][typean]["var_binningshape"] # name (reco)
    v_varshape_binning_gen = datap["analysis"][typean]["var_binningshape_gen"] # name (gen)
    lvarshape_binmin_reco = \
        datap["analysis"][typean].get("sel_binminshape_reco", None)
    lvarshape_binmax_reco = \
        datap["analysis"][typean].get("sel_binmaxshape_reco", None)
    p_nbinshape_reco = len(lvarshape_binmin_reco) # number of reco bins
    lvarshape_binmin_gen = \
        datap["analysis"][typean].get("sel_binminshape_gen", None)
    lvarshape_binmax_gen = \
        datap["analysis"][typean].get("sel_binmaxshape_gen", None)
    p_nbinshape_gen = len(lvarshape_binmin_gen) # number of gen bins
    varshaperanges_reco = lvarshape_binmin_reco.copy()
    varshaperanges_reco.append(lvarshape_binmax_reco[-1])
    varshapebinarray_reco = array("d", varshaperanges_reco) # array of bin edges to use in histogram constructors
    varshaperanges_gen = lvarshape_binmin_gen.copy()
    varshaperanges_gen.append(lvarshape_binmax_gen[-1])
    varshapebinarray_gen = array("d", varshaperanges_gen) # array of bin edges to use in histogram constructors

    file_results = TFile.Open(file_in)
    if not file_results:
        logger.fatal(make_message_notfound(file_in))

    ibin2 = 1

    suffix = "%s_%g_%g" % (v_var2_binning, lvar2_binmin_gen[ibin2], lvar2_binmax_gen[ibin2])

    # HF data
    nameobj = "%s_hf_data_%d_stat" % (shape, ibin2)
    hf_data_stat = file_results.Get(nameobj)
    if not hf_data_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    nameobj = "%s_hf_data_%d_syst" % (shape, ibin2)
    hf_data_syst = file_results.Get(nameobj)
    if not hf_data_syst:
        logger.fatal(make_message_notfound(nameobj, file_in))

    # HF PYTHIA
    nameobj = "%s_hf_pythia_%d_stat" % (shape, ibin2)
    hf_pythia_stat = file_results.Get(nameobj)
    if not hf_pythia_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    # create a graph with PYTHIA points and with zero syst. unc.
    hf_pythia_stat_zero = hf_pythia_stat.Clone(f"{hf_pythia_stat.GetName()}_zero")
    for i in range(hf_pythia_stat_zero.GetNbinsX()):
        hf_pythia_stat_zero.SetBinError(i + 1, 0)
    gStyle.SetErrorX(0.5) # we have to restore the histogram bin width to propagate it to graph
    hf_pythia_syst = TGraphAsymmErrors(hf_pythia_stat_zero) # convert histogram into a graph
    gStyle.SetErrorX(0) # set back the intended settings

    # HF POWHEG
    nameobj = "%s_hf_powheg_%d_stat" % (shape, ibin2)
    hf_powheg_stat = file_results.Get(nameobj)
    if not hf_powheg_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    nameobj = "%s_hf_powheg_%d_syst" % (shape, ibin2)
    hf_powheg_syst = file_results.Get(nameobj)
    if not hf_powheg_syst:
        logger.fatal(make_message_notfound(nameobj, file_in))

    if do_ivan:
        # HF Ivan
        nameobj = "%s_hf_ivan_stat" % shape
        hf_ivan_stat = file_results.Get(nameobj)
        if not hf_ivan_stat:
            logger.fatal(make_message_notfound(nameobj, file_in))
        nameobj = "%s_hf_ivan_syst" % shape
        hf_ivan_syst = file_results.Get(nameobj)
        if not hf_ivan_syst:
            logger.fatal(make_message_notfound(nameobj, file_in))

    # inclusive data
    nameobj = "%s_incl_data_%d_stat" % (shape, ibin2)
    incl_data_stat = file_results.Get(nameobj)
    if not incl_data_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    nameobj = "%s_incl_data_%d_syst" % (shape, ibin2)
    incl_data_syst = file_results.Get(nameobj)
    if not incl_data_syst:
        logger.fatal(make_message_notfound(nameobj, file_in))

    # inclusive PYTHIA
    nameobj = "%s_incl_pythia_%d_stat" % (shape, ibin2)
    incl_pythia_stat = file_results.Get(nameobj)
    if not incl_pythia_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    # create a graph with PYTHIA points and with zero syst. unc.
    incl_pythia_stat_zero = incl_pythia_stat.Clone(f"{incl_pythia_stat.GetName()}_zero")
    for i in range(incl_pythia_stat_zero.GetNbinsX()):
        incl_pythia_stat_zero.SetBinError(i + 1, 0)
    gStyle.SetErrorX(0.5) # we have to restore the histogram bin width to propagate it to graph
    incl_pythia_syst = TGraphAsymmErrors(incl_pythia_stat_zero) # convert histogram into a graph
    gStyle.SetErrorX(0) # set back the intended settings

    if do_ivan:
        # inclusive Ivan
        nameobj = "%s_incl_ivan_stat" % shape
        incl_ivan_stat = file_results.Get(nameobj)
        if not incl_ivan_stat:
            logger.fatal(make_message_notfound(nameobj, file_in))
        nameobj = "%s_incl_ivan_syst" % shape
        incl_ivan_syst = file_results.Get(nameobj)
        if not incl_ivan_syst:
            logger.fatal(make_message_notfound(nameobj, file_in))

    # plot the results with systematic uncertainties and models

    size_can = [800, 800]
    size_can_double = [800, 1200]
    offsets_axes = [0.8, 1.1]
    offsets_axes_double = [0.8, 0.8]
    margins_can = [0.1, 0.13, 0.1, 0.03]
    margins_can_double = [0.1, 0.1, 0.1, 0.1]
    margins_can_double = [0., 0., 0., 0.]
    size_thg = 0.05
    offset_thg = 0.85

    gStyle.SetErrorX(0) # do not plot horizontal error bars of histograms
    fontsize = 0.035
    opt_leg_g = "FP"
    opt_plot_g = "2"

    list_new = [] # list to avoid loosing objects created in loops

    # labels

    x_latex = 0.16
    y_latex_top = 0.83
    y_step = 0.062

    title_x = v_varshape_latex
    title_y = "(1/#it{N}_{jet}) d#it{N}/d%s" % v_varshape_latex
    title_full = ";%s;%s" % (title_x, title_y)
    title_full_ratio = ";%s;data/MC: ratio of %s" % (title_x, title_y)
    title_full_ratio_double = f";{title_x};MC/data"
    title_full_ratio_theory = f";{title_x};theory/data"

    text_alice = "#bf{ALICE}, pp, #sqrt{#it{s}} = 13 TeV"
    text_alice_sim = "#bf{ALICE} Simulation, pp, #sqrt{#it{s}} = 13 TeV"
    text_pythia = "PYTHIA 8 (Monash)"
    text_pythia_short = "PYTHIA 8"
    text_pythia_split = "#splitline{PYTHIA 8}{(Monash)}"
    text_powheg = "POWHEG #plus PYTHIA 6"
    text_ivan = "SCET"
    text_jets = "charged jets, anti-#it{k}_{T}, #it{R} = 0.4"
    text_ptjet = "%g #leq %s < %g GeV/#it{c}, |#it{#eta}_{jet}| #leq 0.5" % (lvar2_binmin_reco[ibin2], p_latexbin2var, lvar2_binmax_reco[ibin2])
    text_pth = "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, |#it{y}_{%s}| #leq 0.8" % (lpt_finbinmin[0], p_latexnhadron, min(lpt_finbinmax[-1], lvar2_binmax_reco[ibin2]), p_latexnhadron)
    text_ptcut = "#it{p}_{T, incl. ch. jet}^{leading track} #geq 5.33 GeV/#it{c}"
    text_ptcut_sim = "#it{p}_{T, incl. ch. jet}^{leading h^{#pm}} #geq 5.33 GeV/#it{c} (varied)"
    text_sd = "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"

    title_thetag = "#it{#theta}_{g} = #it{R}_{g}/#it{R}"
    radius_jet = 0.4

    # colour and marker indices
    c_hf_data = 0
    c_incl_data = 1
    c_hf_pythia = 2
    c_hf_powheg = 3
    c_incl_pythia = 6
    c_quark_pythia = 5
    c_gluon_pythia = 0
    c_hf_ivan = 4
    c_incl_ivan = 5

    # markers
    m_hf_data = get_marker(0)
    m_incl_data = get_marker(1)
    m_hf_pythia = get_marker(0, 2)
    m_hf_powheg = get_marker(4)
    m_incl_pythia = get_marker(1, 2)
    m_quark_pythia = get_marker(2)
    m_gluon_pythia = get_marker(3)
    m_hf_ivan = get_marker(2)
    m_incl_ivan = get_marker(3)

    # make the horizontal error bars smaller
    if shape == "nsd":
        for gr in [hf_data_syst, incl_data_syst, hf_powheg_syst, hf_pythia_syst, incl_pythia_syst]:
            for i in range(gr.GetN()):
                gr.SetPointEXlow(i, 0.1)
                gr.SetPointEXhigh(i, 0.1)

    # data, HF and inclusive

    hf_data_syst_cl = hf_data_syst.Clone()

    leg_pos = [.72, .75, .85, .85]
    list_obj = [hf_data_syst, incl_data_syst, hf_data_stat, incl_data_stat]
    labels_obj = ["%s-tagged" % p_latexnhadron, "inclusive", "", ""]
    colours = [get_colour(i, j) for i, j in zip((c_hf_data, c_incl_data, c_hf_data, c_incl_data), (2, 2, 1, 1))]
    markers = [m_hf_data, m_incl_data, m_hf_data, m_incl_data]
    y_margin_up = 0.46
    y_margin_down = 0.05
    cshape_data, list_obj_data_new = make_plot("cshape_data_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=margins_can, \
        title=title_full)
    for gr, c in zip((hf_data_syst, incl_data_syst), (c_hf_data, c_incl_data)):
        gr.SetMarkerColor(get_colour(c))
    list_obj_data_new[0].SetTextSize(fontsize)
    if shape == "nsd":
        hf_data_syst.GetXaxis().SetNdivisions(5)
    # Draw a line through the points.
    if shape == "nsd":
        for h in (hf_data_stat, incl_data_stat):
            h_line = h.Clone(h.GetName() + "_line")
            h_line.SetLineStyle(2)
            h_line.Draw("l hist same")
            list_new.append(h_line)
    cshape_data.Update()
    if shape == "rg":
        # plot the theta_g axis
        gr_frame = hf_data_syst
        axis_rg = gr_frame.GetXaxis()
        rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
        rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
        thetag_min = rg_min / radius_jet
        thetag_max = rg_max / radius_jet
        y_axis = cshape_data.GetUymax()
        axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
        axis_thetag.SetTitle(title_thetag)
        axis_thetag.SetTitleSize(size_thg)
        axis_thetag.SetLabelSize(0.036)
        axis_thetag.SetTitleFont(42)
        axis_thetag.SetLabelFont(42)
        axis_thetag.SetLabelOffset(0)
        axis_thetag.SetTitleOffset(offset_thg)
        cshape_data.SetTickx(0)
        axis_thetag.Draw("same")
    # Draw LaTeX
    y_latex = y_latex_top
    list_latex_data = []
    for text_latex in [text_alice, text_jets, text_ptjet, text_pth, text_ptcut, text_sd]:
        latex = TLatex(x_latex, y_latex, text_latex)
        list_latex_data.append(latex)
        draw_latex(latex, textsize=fontsize)
        y_latex -= y_step
    cshape_data.Update()
    cshape_data.SaveAs("%s/%s_data_%s.pdf" % (rootpath, shape, suffix))

    # data and PYTHIA, POWHEG, Ivan, HF

    leg_pos = [.72, .65, .85, .85]
    list_obj = [hf_data_syst_cl, hf_powheg_syst, hf_data_stat, hf_pythia_stat, hf_powheg_stat]
    labels_obj = ["data", text_powheg, "", text_pythia_split, "", ""]
    colours = [get_colour(i, j) for i, j in zip((c_hf_data, c_hf_powheg, c_hf_data, c_hf_pythia, c_hf_powheg), (2, 2, 1, 1, 1))]
    markers = [m_hf_data, m_hf_powheg, m_hf_data, m_hf_pythia, m_hf_powheg]
    y_margin_up = 0.4
    y_margin_down = 0.05
    cshape_data_mc_hf, list_obj_data_mc_hf_new = make_plot("cshape_data_mc_hf_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=margins_can, \
        title=title_full)
    for gr, c in zip([hf_data_syst_cl, hf_powheg_syst], [c_hf_data, c_hf_powheg]):
        gr.SetMarkerColor(get_colour(c))
    leg_data_mc_hf = list_obj_data_mc_hf_new[0]
    leg_data_mc_hf.SetHeader("%s-tagged" % p_latexnhadron)
    leg_data_mc_hf.SetTextSize(fontsize)
    if shape == "nsd":
        hf_data_syst_cl.GetXaxis().SetNdivisions(5)
        #axis_nsd = hf_data_syst_cl.GetHistogram().GetXaxis()
        #x1 = axis_nsd.GetBinLowEdge(1)
        #x2 = axis_nsd.GetBinUpEdge(axis_nsd.GetNbins())
        #axis_nsd.Set(5, x1, x2)
        #for ibin in range(axis_nsd.GetNbins()):
        #    axis_nsd.SetBinLabel(ibin + 1, "%d" % ibin)
        #axis_nsd.SetNdivisions(5)
    cshape_data_mc_hf.Update()
    if shape == "rg":
        # plot the theta_g axis
        axis_rg = hf_data_stat.GetXaxis()
        rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
        rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
        thetag_min = rg_min / radius_jet
        thetag_max = rg_max / radius_jet
        y_axis = cshape_data_mc_hf.GetUymax()
        axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
        axis_thetag.SetTitle(title_thetag)
        axis_thetag.SetTitleSize(size_thg)
        axis_thetag.SetLabelSize(0.036)
        axis_thetag.SetTitleFont(42)
        axis_thetag.SetLabelFont(42)
        axis_thetag.SetLabelOffset(0)
        axis_thetag.SetTitleOffset(offset_thg)
        cshape_data_mc_hf.SetTickx(0)
        axis_thetag.Draw("same")
    # Draw LaTeX
    y_latex = y_latex_top
    list_latex_data_mc_hf = []
    for text_latex in [text_alice, text_jets, text_ptjet, text_pth, text_sd]:
        latex = TLatex(x_latex, y_latex, text_latex)
        list_latex_data_mc_hf.append(latex)
        draw_latex(latex, textsize=fontsize)
        y_latex -= y_step
    cshape_data_mc_hf.Update()
    cshape_data_mc_hf.SaveAs("%s/%s_data_mc_hf_%s.pdf" % (rootpath, shape, suffix))

    # data and PYTHIA, inclusive

    #leg_pos = [.68, .65, .85, .85]
    list_obj = [incl_data_syst, incl_pythia_syst, incl_data_stat, incl_pythia_stat]
    labels_obj = ["data", text_pythia_split]
    colours = [get_colour(i, j) for i, j in zip((c_incl_data, c_incl_pythia, c_incl_data, c_incl_pythia), (2, 2, 1, 1))]
    markers = [m_incl_data, m_incl_pythia, m_incl_data, m_incl_pythia]
    y_margin_up = 0.4
    y_margin_down = 0.05
    cshape_data_mc_incl, list_obj_data_mc_incl_new = make_plot("cshape_data_mc_incl_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=margins_can, \
        title=title_full)
    for gr, c in zip([incl_data_syst, incl_pythia_syst], [c_incl_data, c_incl_pythia]):
        gr.SetMarkerColor(get_colour(c))
    leg_data_mc_incl = list_obj_data_mc_incl_new[0]
    leg_data_mc_incl.SetHeader("inclusive")
    leg_data_mc_incl.SetTextSize(fontsize)
    if shape == "nsd":
        incl_data_syst.GetXaxis().SetNdivisions(5)
    cshape_data_mc_incl.Update()
    if shape == "rg":
        # plot the theta_g axis
        axis_rg = incl_data_stat.GetXaxis()
        rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
        rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
        thetag_min = rg_min / radius_jet
        thetag_max = rg_max / radius_jet
        y_axis = cshape_data_mc_incl.GetUymax()
        axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
        axis_thetag.SetTitle(title_thetag)
        axis_thetag.SetTitleSize(size_thg)
        axis_thetag.SetLabelSize(0.036)
        axis_thetag.SetTitleFont(42)
        axis_thetag.SetLabelFont(42)
        axis_thetag.SetLabelOffset(0)
        axis_thetag.SetTitleOffset(offset_thg)
        cshape_data_mc_incl.SetTickx(0)
        axis_thetag.Draw("same")
    # Draw LaTeX
    y_latex = y_latex_top
    list_latex_data_mc_incl = []
    for text_latex in [text_alice, text_jets, text_ptjet, text_ptcut, text_sd]:
        latex = TLatex(x_latex, y_latex, text_latex)
        list_latex_data_mc_incl.append(latex)
        draw_latex(latex, textsize=fontsize)
        y_latex -= y_step
    cshape_data_mc_incl.Update()
    cshape_data_mc_incl.SaveAs("%s/%s_data_mc_incl_%s.pdf" % (rootpath, shape, suffix))

    # Ratios data/MC, HF and inclusive

    line_1 = TLine(lvarshape_binmin_reco[0], 1, lvarshape_binmax_reco[-1], 1)
    line_1.SetLineStyle(9)
    line_1.SetLineColor(1)
    line_1.SetLineWidth(3)

    # PYTHIA, HF, inclusive

    incl_pythia_syst_cl = incl_pythia_syst.Clone()

    y_min_h, y_max_h = get_y_window_his([hf_pythia_stat, incl_pythia_stat])
    y_min_g, y_max_g = get_y_window_gr([incl_pythia_syst])
    y_min = min(y_min_h, y_min_g)
    y_max = max(y_max_h, y_max_g)
    y_margin_up = 0.46
    y_margin_down = 0.05
    y_min_plot, y_max_plot = get_plot_range(y_min, y_max, y_margin_down, y_margin_up)

    #leg_pos = [.6, .65, .75, .85]
    leg_pos = [.72, .55, .85, .85]
    list_obj = [hf_pythia_syst, incl_pythia_syst, hf_pythia_stat, incl_pythia_stat]
    labels_obj = ["%s-tagged" % p_latexnhadron, "inclusive"]
    colours = [get_colour(i, j) for i, j in zip((c_hf_pythia, c_incl_pythia, c_hf_pythia, c_incl_pythia), (2, 2, 1, 1))]
    markers = [m_hf_pythia, m_incl_pythia, m_hf_pythia, m_incl_pythia]
    y_margin_up = 0.46
    y_margin_down = 0.05
    cshape_mc, list_obj_mc_new = make_plot("cshape_mc_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, range_y=[y_min_plot, y_max_plot], margins_c=margins_can, \
        title=title_full)
    cshape_mc.Update()
    for gr, c in zip((hf_pythia_syst, incl_pythia_syst), (c_hf_pythia, c_incl_pythia)):
        gr.SetMarkerColor(get_colour(c))
    leg_mc = list_obj_mc_new[0]
    leg_mc.SetTextSize(fontsize)
    leg_mc.SetHeader(text_pythia_split)
    if shape == "nsd":
        list_obj[0].GetXaxis().SetNdivisions(5)
    cshape_mc.Update()
    if shape == "rg":
        # plot the theta_g axis
        axis_rg = hf_pythia_stat.GetXaxis()
        rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
        rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
        thetag_min = rg_min / radius_jet
        thetag_max = rg_max / radius_jet
        y_axis = cshape_mc.GetUymax()
        axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
        axis_thetag.SetTitle(title_thetag)
        axis_thetag.SetTitleSize(size_thg)
        axis_thetag.SetLabelSize(0.036)
        axis_thetag.SetTitleFont(42)
        axis_thetag.SetLabelFont(42)
        axis_thetag.SetLabelOffset(0)
        axis_thetag.SetTitleOffset(offset_thg)
        cshape_mc.SetTickx(0)
        axis_thetag.Draw("same")
    # Draw LaTeX
    y_latex = y_latex_top
    list_latex_mc = []
    for text_latex in [text_alice_sim, text_jets, text_ptjet, text_pth, text_ptcut_sim, text_sd]:
        latex = TLatex(x_latex, y_latex, text_latex)
        list_latex_mc.append(latex)
        draw_latex(latex, textsize=fontsize)
        y_latex -= y_step
    cshape_mc.Update()
    cshape_mc.SaveAs("%s/%s_mc_%s.pdf" % (rootpath, shape, suffix))

    # PYTHIA, HF, inclusive

    #leg_pos = [.6, .65, .75, .85]
    leg_pos = [.72, .67, .85, .85]
    list_obj = [incl_pythia_syst_cl, incl_pythia_stat, hf_pythia_stat]
    labels_obj = ["inclusive", "", "%s-tagged" % p_latexnhadron]
    colours = [get_colour(i, j) for i, j in zip((c_incl_pythia, c_incl_pythia, c_hf_pythia), (2, 1, 1))]
    markers = [m_incl_pythia, m_incl_pythia, m_hf_pythia]
    y_margin_up = 0.46
    y_margin_down = 0.05
    cshape_mc, list_obj_mc_new = make_plot("cshape_mc_id_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, range_y=[y_min_plot, y_max_plot], margins_c=margins_can, \
        title=title_full)
    # Draw a line through the points.
    if shape == "nsd":
        for h in (incl_pythia_stat, hf_pythia_stat):
            h_line = h.Clone(h.GetName() + "_line")
            h_line.SetLineStyle(2)
            h_line.Draw("l hist same")
            list_new.append(h_line)
    cshape_mc.Update()
    incl_pythia_syst_cl.SetMarkerColor(get_colour(c_incl_pythia))
    leg_mc = list_obj_mc_new[0]
    leg_mc.SetTextSize(fontsize)
    leg_mc.SetHeader(text_pythia_split)
    if shape == "nsd":
        incl_pythia_syst_cl.GetXaxis().SetNdivisions(5)
    cshape_mc.Update()
    if shape == "rg":
        # plot the theta_g axis
        axis_rg = hf_pythia_stat.GetXaxis()
        rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
        rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
        thetag_min = rg_min / radius_jet
        thetag_max = rg_max / radius_jet
        y_axis = cshape_mc.GetUymax()
        axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
        axis_thetag.SetTitle(title_thetag)
        axis_thetag.SetTitleSize(size_thg)
        axis_thetag.SetLabelSize(0.036)
        axis_thetag.SetTitleFont(42)
        axis_thetag.SetLabelFont(42)
        axis_thetag.SetLabelOffset(0)
        axis_thetag.SetTitleOffset(offset_thg)
        cshape_mc.SetTickx(0)
        axis_thetag.Draw("same")
    # Draw LaTeX
    y_latex = y_latex_top
    list_latex_mc = []
    for text_latex in [text_alice_sim, text_jets, text_ptjet, text_pth, text_ptcut_sim, text_sd]:
        latex = TLatex(x_latex, y_latex, text_latex)
        list_latex_mc.append(latex)
        draw_latex(latex, textsize=fontsize)
        y_latex -= y_step
    cshape_mc.Update()
    cshape_mc.SaveAs("%s/%s_mc_id_%s.pdf" % (rootpath, shape, suffix))

    # data + MC/data, HF and inclusive

    # print relative syst. unc.
    for name, gr in zip(("HF", "inclusive"), (hf_data_syst, incl_data_syst)):
        print(f"Rel. syst. unc. for {name} {shape}")
        e_plus_min = float("inf")
        e_minus_min = float("inf")
        e_plus_max = 0.
        e_minus_max = 0.
        for i in range(gr.GetN()):
            # skip untagged bin for zg and rg
            if i == 0 and shape in ("zg", "rg"):
                continue
            y = gr.GetPointY(i)
            e_plus = 100 * gr.GetErrorYhigh(i)
            e_minus = 100 * gr.GetErrorYlow(i)
            e_plus /= y
            e_minus /= y
            e_plus_min = min(e_plus_min, e_plus)
            e_minus_min = min(e_minus_min, e_minus)
            e_plus_max = max(e_plus_max, e_plus)
            e_minus_max = max(e_minus_max, e_minus)
            # print(f"Point {i}, up {e_plus:.2g} %, down {e_minus:.2g} %")
        # print(f"Minima: up {e_plus_min:.2g} %, down {e_minus_min:.2g} %")
        # print(f"Maxima: up {e_plus_max:.2g} %, down {e_minus_max:.2g} %")
        print(f"Absolutes: min: {min(e_plus_min, e_minus_min):.2g} %, max {max(e_plus_max, e_minus_max):.2g} %")

    # explicit y ranges [zg, rg, nsd]
    list_range_y = [[0.01, 9], [0.01, 6.1], [0.001, 0.72]] # data
    list_range_y_rat = [[0.1, 5.9], [0, 2], [0, 2]] # mc/data ratios
    list_range_x = [[0.1, 0.5], [0, 0.4], [-0.5, 4.5]] # data and mc/data ratios
    i_shape = 0 if shape == "zg" else 1 if shape == "rg" else 2
    print(f"Index {i_shape}")

    # data
    leg_pos = [.7, .75, .82, .85]
    leg_pos = [.65, .7, .82, .85]
    fraction_untagged_hf = hf_data_stat.Integral(1, 1, "width")
    fraction_untagged_incl = incl_data_stat.Integral(1, 1, "width")
    # hard-coded to values to unify them across zg, rg, nsd
    fraction_untagged_hf_text = 0.22
    fraction_untagged_incl_text = 0.028
    print(f"Untagged fraction: HF {fraction_untagged_hf}, incl {fraction_untagged_incl}")
    list_obj = [hf_data_syst, incl_data_syst, hf_data_stat, incl_data_stat]
    labels_obj = ["%s-tagged" % p_latexnhadron, "inclusive", "", ""]
    colours = [get_colour(i, j) for i, j in zip((c_hf_data, c_incl_data, c_hf_data, c_incl_data), (2, 2, 1, 1))]
    markers = [m_hf_data, m_incl_data, m_hf_data, m_incl_data]
    y_margin_up = 0.42
    y_margin_down = 0.05
    cshape_datamc_all = TCanvas("cshape_datamc_" + suffix, "cshape_datamc_" + suffix)
    cshape_datamc_all.Divide(1, 3)
    pad1 = cshape_datamc_all.cd(1)
    pad2 = cshape_datamc_all.cd(2)
    pad3 = cshape_datamc_all.cd(3)
    pad1.SetPad(0., 0.45, 1, 1)
    pad2.SetPad(0., 0.25, 1, 0.45)
    pad3.SetPad(0., 0., 1, 0.25)
    pad1.SetBottomMargin(0.)
    pad2.SetBottomMargin(0.)
    pad3.SetBottomMargin(0.25)
    pad1.SetTopMargin(0.1)
    pad2.SetTopMargin(0.)
    pad3.SetTopMargin(0.)
    pad1.SetLeftMargin(0.12)
    pad2.SetLeftMargin(0.12)
    pad3.SetLeftMargin(0.12)
    pad1.SetTicks(1, 1)
    pad2.SetTicks(1, 1)
    pad3.SetTicks(1, 1)
    cshape_datamc_all, list_obj_data_new = make_plot("cshape_datamc_" + suffix, size=size_can_double, \
        can=cshape_datamc_all, pad=1, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=[0.8, 1.1], \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_c=margins_can_double, \
        range_x=list_range_x[i_shape], \
        # margins_y=[y_margin_down, y_margin_up], \
        range_y=list_range_y[i_shape], \
        title=title_full)
    for gr, c in zip((hf_data_syst, incl_data_syst), (c_hf_data, c_incl_data)):
        gr.SetMarkerColor(get_colour(c))
    list_obj_data_new[0].SetTextSize(fontsize)
    hf_data_syst.GetYaxis().SetLabelSize(0.1 * 3/7)
    #hf_data_syst.GetYaxis().SetTitleSize(0.1)
    if shape == "nsd":
        hf_data_syst.GetXaxis().SetNdivisions(5)
    # Draw a line through the points.
    if shape == "nsd":
        for h in (hf_data_stat, incl_data_stat):
            h_line = h.Clone(h.GetName() + "_line")
            h_line.SetLineStyle(2)
            h_line.Draw("l hist same")
            list_new.append(h_line)
    cshape_datamc_all.Update()
    if shape == "rg":
        # plot the theta_g axis
        gr_frame = hf_data_syst
        axis_rg = gr_frame.GetXaxis()
        rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
        rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
        thetag_min = rg_min / radius_jet
        thetag_max = rg_max / radius_jet
        y_axis = pad1.GetUymax()
        axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
        axis_thetag.SetTitle(title_thetag)
        axis_thetag.SetTitleSize(size_thg)
        axis_thetag.SetLabelSize(0.036)
        axis_thetag.SetTitleFont(42)
        axis_thetag.SetLabelFont(42)
        axis_thetag.SetLabelOffset(0)
        axis_thetag.SetTitleOffset(offset_thg)
        cshape_datamc_all.cd(1).SetTickx(0)
        axis_thetag.Draw("same")
    # Draw LaTeX
    y_latex = y_latex_top
    list_latex_data = []
    for text_latex in [text_alice, text_jets, text_ptjet, text_pth, text_ptcut, text_sd]:
        latex = TLatex(x_latex, y_latex, text_latex)
        list_latex_data.append(latex)
        draw_latex(latex, textsize=fontsize)
        y_latex -= y_step
    y_latex = y_latex_top - 3 * y_step
    if shape != "nsd":
        for text_latex in ["SD-untagged jets", f"{p_latexnhadron}-tagged: {100 * fraction_untagged_hf_text:.2g}%", f"inclusive: {100 * fraction_untagged_incl_text:.2g}%"]:
            latex = TLatex(x_latex + 0.45, y_latex, text_latex)
            list_latex_data.append(latex)
            draw_latex(latex, textsize=fontsize)
            y_latex -= y_step
    cshape_datamc_all.Update()

    # HF MC/data
    line_1 = TLine(list_range_x[i_shape][0], 1, list_range_x[i_shape][1], 1)
    line_1.SetLineStyle(9)
    line_1.SetLineColor(1)
    line_1.SetLineWidth(3)
    leg_pos = [.15, .7, .4, .95]
    hf_ratio_powheg_stat = hf_powheg_stat.Clone(f"{hf_powheg_stat.GetName()}_rat")
    hf_ratio_powheg_stat.Divide(hf_data_stat)
    hf_ratio_powheg_syst = divide_graphs(hf_powheg_syst, hf_data_syst)
    hf_ratio_pythia_stat = hf_pythia_stat.Clone(f"{hf_pythia_stat.GetName()}_rat")
    hf_ratio_pythia_stat.Divide(hf_data_stat)
    # hf_ratio_pythia_stat = hf_data_stat.Clone(f"{hf_data_stat.GetName()}_rat") # version data/MC
    # hf_ratio_pythia_stat.Divide(hf_pythia_stat) # version data/MC
    hf_ratio_pythia_syst = divide_graphs(hf_pythia_syst, hf_data_syst)
    # hf_ratio_pythia_syst = divide_graphs(hf_data_syst, hf_pythia_syst) # version data/MC
    incl_ratio_pythia_stat = incl_pythia_stat.Clone(f"{incl_pythia_stat.GetName()}_rat")
    incl_ratio_pythia_stat.Divide(incl_data_stat)
    incl_ratio_pythia_syst = divide_graphs(incl_pythia_syst, incl_data_syst)
    # incl_ratio_pythia_stat = incl_data_stat.Clone(f"{incl_data_stat.GetName()}_rat") # version data/MC
    # incl_ratio_pythia_stat.Divide(incl_pythia_stat) # version data/MC
    # incl_ratio_pythia_syst = divide_graphs(incl_data_syst, incl_pythia_syst) # version data/MC
    if shape != "nsd":
        for gr in (incl_ratio_pythia_syst, hf_ratio_pythia_syst, hf_ratio_powheg_syst):
            gr.SetPointY(0, 1.)
        for his in (incl_ratio_pythia_stat, hf_ratio_pythia_stat, hf_ratio_powheg_stat):
            his.SetBinContent(1, 1.)
    list_obj = [hf_ratio_powheg_syst, hf_ratio_pythia_syst, hf_ratio_powheg_stat, hf_ratio_pythia_stat, line_1]
    labels_obj = [f"{p_latexnhadron}-tagged {text_powheg}", f"{p_latexnhadron}-tagged {text_pythia_short}", "", ""]
    colours = [get_colour(i, j) for i, j in zip((c_hf_powheg, c_hf_pythia, c_hf_powheg, c_hf_pythia), (2, 2, 1, 1))]
    markers = [m_hf_powheg, m_hf_pythia, m_hf_powheg, m_hf_pythia]
    y_margin_up = 0.29
    y_margin_down = 0.05
    cshape_datamc_all, list_obj_data_mc_hf_new = make_plot("cshape_data_mc_hf_" + suffix, size=size_can_double, \
        can=cshape_datamc_all, pad=2, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=[1, 1.3 * 3/7], \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_c=margins_can_double, \
        range_x=list_range_x[i_shape], \
        margins_y=[y_margin_down, y_margin_up], \
        # range_y=list_range_y_rat[i_shape], \
        title=title_full_ratio_double)
    list_obj[0].GetXaxis().SetLabelSize(0.1)
    list_obj[0].GetXaxis().SetTitleSize(0.1)
    list_obj[0].GetYaxis().SetLabelSize(0.12)
    list_obj[0].GetYaxis().SetTitleSize(0.12)
    list_obj[0].GetYaxis().SetTitleOffset(0.5)
    for gr, c in zip([hf_ratio_powheg_syst, hf_ratio_pythia_syst], [c_hf_powheg, c_hf_pythia]):
        gr.SetMarkerColor(get_colour(c))
    leg_data_mc_hf = list_obj_data_mc_hf_new[0]
    #leg_data_mc_hf.SetHeader("%s-tagged" % p_latexnhadron)
    leg_data_mc_hf.SetTextSize(fontsize * 7/3)
    # leg_data_mc_hf.SetNColumns(2)
    if shape == "nsd":
        list_obj[0].GetXaxis().SetNdivisions(5)
    cshape_datamc_all.Update()

    # inclusive MC/data
    leg_pos = [.15, .8, .9, .95]
    list_obj = [incl_ratio_pythia_syst, incl_ratio_pythia_stat, line_1]
    labels_obj = [f"inclusive {text_pythia_short}", ""]
    colours = [get_colour(i, j) for i, j in zip((c_incl_pythia, c_incl_pythia), (2, 1))]
    markers = [m_incl_pythia, m_incl_pythia]
    y_margin_up = 0.3
    y_margin_down = 0.05
    cshape_datamc_all, list_obj_data_mc_hf_new_2 = make_plot("cshape_data_mc_hf_" + suffix, size=size_can_double, \
        can=cshape_datamc_all, pad=3, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=[1, 1.3 * 3/7], \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_c=margins_can_double, \
        range_x=list_range_x[i_shape], \
        margins_y=[y_margin_down, y_margin_up], \
        # range_y=list_range_y_rat[i_shape], \
        title=title_full_ratio_double)
    list_obj[0].GetXaxis().SetLabelSize(0.1)
    list_obj[0].GetXaxis().SetTitleSize(0.1)
    list_obj[0].GetYaxis().SetLabelSize(0.1)
    list_obj[0].GetYaxis().SetTitleSize(0.1)
    for gr, c in zip([incl_ratio_pythia_syst], [c_incl_pythia]):
        gr.SetMarkerColor(get_colour(c))
    leg_data_mc_hf = list_obj_data_mc_hf_new_2[0]
    #leg_data_mc_hf.SetHeader("%s-tagged" % p_latexnhadron)
    leg_data_mc_hf.SetTextSize(fontsize * 7/3)
    leg_data_mc_hf.SetNColumns(2)
    if shape == "nsd":
        list_obj[0].GetXaxis().SetNdivisions(5)
    cshape_datamc_all.Update()

    # Draw LaTeX
    #y_latex = y_latex_top
    #list_latex_data_mc_hf = []
    #for text_latex in [text_alice, text_jets, text_ptjet, text_pth, text_sd]:
    #    latex = TLatex(x_latex, y_latex, text_latex)
    #    list_latex_data_mc_hf.append(latex)
    #    draw_latex(latex, textsize=fontsize)
    #    y_latex -= y_step
    #cshape_datamc_all.Update()
    pad1.RedrawAxis()
    pad2.RedrawAxis()
    pad3.RedrawAxis()
    cshape_datamc_all.SaveAs("%s/%s_datamc_all_%s.pdf" % (rootpath, shape, suffix))

    # data + MC/data + Ivan, HF and inclusive

    if do_ivan:
        # Normalise ignoring the untagged bin
        if shape in ("zg", "rg"):
            int = 1 - fraction_untagged_hf
            hf_data_stat.Scale(1. / int)
            scale_graph(hf_data_syst, 1. / int)
            int = 1 - fraction_untagged_incl
            incl_data_stat.Scale(1. / int)
            scale_graph(incl_data_syst, 1. / int)
        # data
        leg_pos = [.65, .7, .82, .85]
        hf_ivan_syst_plot = hf_ivan_syst.Clone(f"{hf_ivan_syst.GetName()}_plot")
        hf_ivan_syst_plot.RemovePoint(0) # delete the untagged bin point
        incl_ivan_syst_plot = incl_ivan_syst.Clone(f"{incl_ivan_syst.GetName()}_plot")
        incl_ivan_syst_plot.RemovePoint(0) # delete the untagged bin point
        list_obj = [hf_data_syst, incl_data_syst, hf_ivan_syst_plot, incl_ivan_syst_plot, hf_data_stat, incl_data_stat, hf_ivan_stat, incl_ivan_stat]
        labels_obj = [f"{p_latexnhadron}-tagged", "inclusive", "", "", "", "", "", ""]
        colours = [get_colour(i, j) for i, j in zip((c_hf_data, c_incl_data, c_hf_ivan, c_incl_ivan, c_hf_data, c_incl_data, c_hf_ivan, c_incl_ivan), (2, 2, 2, 2, 1, 1, 1, 1))]
        markers = [m_hf_data, m_incl_data, m_hf_ivan, m_incl_ivan, m_hf_data, m_incl_data, m_hf_ivan, m_incl_ivan]
        y_margin_up = 0.45
        y_margin_down = 0.05
        cshape_datamc_ivan = TCanvas("cshape_datamc_ivan_" + suffix, "cshape_datamc_ivan_" + suffix)
        cshape_datamc_ivan.Divide(1, 3)
        pad1 = cshape_datamc_ivan.cd(1)
        pad2 = cshape_datamc_ivan.cd(2)
        pad3 = cshape_datamc_ivan.cd(3)
        pad1.SetPad(0., 0.45, 1, 1)
        pad2.SetPad(0., 0.25, 1, 0.45)
        pad3.SetPad(0., 0., 1, 0.25)
        pad1.SetBottomMargin(0.)
        pad2.SetBottomMargin(0.)
        pad3.SetBottomMargin(0.25)
        pad1.SetTopMargin(0.1)
        pad2.SetTopMargin(0.)
        pad3.SetTopMargin(0.)
        pad1.SetLeftMargin(0.12)
        pad2.SetLeftMargin(0.12)
        pad3.SetLeftMargin(0.12)
        pad1.SetTicks(1, 1)
        pad2.SetTicks(1, 1)
        pad3.SetTicks(1, 1)
        cshape_datamc_ivan, list_obj_data_new = make_plot("cshape_datamc_" + suffix, size=size_can_double, \
            can=cshape_datamc_ivan, pad=1, \
            list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=[opt_plot_g, opt_plot_g, "3", "3"], offsets_xy=[0.8, 1.1], \
            colours=colours, markers=markers, leg_pos=leg_pos, margins_c=margins_can_double, \
            range_x=list_range_x[i_shape], \
            margins_y=[y_margin_down, y_margin_up], \
            # range_y=list_range_y[i_shape], \
            title=title_full)
        for gr, c in zip((hf_data_syst, incl_data_syst, hf_ivan_syst_plot, incl_ivan_syst_plot), (c_hf_data, c_incl_data, c_hf_ivan, c_incl_ivan)):
            gr.SetMarkerColor(get_colour(c))
        leg_data_mc = list_obj_data_new[0]
        leg_data_mc.SetTextSize(fontsize)
        leg_data_mc.SetHeader("data")
        leg_data_mc_theory = TLegend(.65, .5, .82, .65)
        setup_legend(leg_data_mc_theory, fontsize)
        # leg_data_mc_theory.SetTextSize(fontsize)
        leg_data_mc_theory.SetHeader(text_ivan)
        leg_data_mc_theory.AddEntry(hf_ivan_syst_plot, f"{p_latexnhadron}-tagged")
        leg_data_mc_theory.AddEntry(incl_ivan_syst_plot, "inclusive")
        leg_data_mc_theory.Draw()
        hf_data_syst.GetYaxis().SetLabelSize(0.1 * 3/7)
        #hf_data_syst.GetYaxis().SetTitleSize(0.1)
        if shape == "nsd":
            hf_data_syst.GetXaxis().SetNdivisions(5)
        # Draw a line through the points.
        if shape == "nsd":
            for h in (hf_data_stat, incl_data_stat):
                h_line = h.Clone(h.GetName() + "_line")
                h_line.SetLineStyle(2)
                h_line.Draw("l hist same")
                list_new.append(h_line)
        cshape_datamc_ivan.Update()
        if shape == "rg":
            # plot the theta_g axis
            gr_frame = hf_data_syst
            axis_rg = gr_frame.GetXaxis()
            rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
            rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
            thetag_min = rg_min / radius_jet
            thetag_max = rg_max / radius_jet
            y_axis = pad1.GetUymax()
            axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
            axis_thetag.SetTitle(title_thetag)
            axis_thetag.SetTitleSize(size_thg)
            axis_thetag.SetLabelSize(0.036)
            axis_thetag.SetTitleFont(42)
            axis_thetag.SetLabelFont(42)
            axis_thetag.SetLabelOffset(0)
            axis_thetag.SetTitleOffset(offset_thg)
            cshape_datamc_ivan.cd(1).SetTickx(0)
            axis_thetag.Draw("same")
        # Draw LaTeX
        y_latex = y_latex_top
        list_latex_data = []
        for text_latex in [text_alice, text_jets, text_ptjet, text_pth, text_ptcut, text_sd]:
            latex = TLatex(x_latex, y_latex, text_latex)
            list_latex_data.append(latex)
            draw_latex(latex, textsize=fontsize)
            y_latex -= y_step
        y_latex = y_latex_top - 3 * y_step
        # for text_latex in ["SD-untagged jets", f"{p_latexnhadron}-tagged: {100 * fraction_untagged_hf:.2g}%", f"inclusive: {100 * fraction_untagged_incl:.2g}%"]:
        #     latex = TLatex(x_latex + 0.45, y_latex, text_latex)
        #     list_latex_data.append(latex)
        #     draw_latex(latex, textsize=fontsize)
        #     y_latex -= y_step
        cshape_datamc_ivan.Update()

        # HF theory/data
        line_1 = TLine(list_range_x[i_shape][0], 1, list_range_x[i_shape][1], 1)
        line_1.SetLineStyle(9)
        line_1.SetLineColor(1)
        line_1.SetLineWidth(3)
        hf_ratio_ivan_stat = hf_ivan_stat.Clone(f"{hf_ivan_stat.GetName()}_rat")
        hf_ratio_ivan_stat.Divide(hf_data_stat)
        hf_ratio_ivan_stat.SetBinContent(1, 1.)
        hf_ratio_ivan_syst = divide_graphs(hf_ivan_syst, hf_data_syst)
        hf_ratio_ivan_syst.RemovePoint(0) # delete the untagged bin point
        incl_ratio_ivan_stat = incl_ivan_stat.Clone(f"{incl_ivan_stat.GetName()}_rat")
        incl_ratio_ivan_stat.Divide(incl_data_stat)
        incl_ratio_ivan_stat.SetBinContent(1, 1.)
        incl_ratio_ivan_syst = divide_graphs(incl_ivan_syst, incl_data_syst)
        incl_ratio_ivan_syst.RemovePoint(0) # delete the untagged bin point
        leg_pos = [.15, .7, .4, .95]
        list_obj = [hf_ratio_ivan_syst, hf_ratio_ivan_stat, line_1]
        labels_obj = [f"{p_latexnhadron}-tagged {text_ivan}", "", ""]
        colours = [get_colour(i, j) for i, j in zip((c_hf_ivan, c_hf_ivan), (2, 1))]
        markers = [m_hf_ivan, m_hf_ivan]
        y_margin_up = 0.05
        y_margin_down = 0.05
        cshape_datamc_ivan, list_obj_data_mc_hf_new = make_plot("cshape_data_mc_hf_ivan_" + suffix, size=size_can_double, \
            can=cshape_datamc_ivan, pad=2, \
            list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g="3", offsets_xy=[1, 1.3 * 3/7], \
            colours=colours, markers=markers, leg_pos=None, margins_c=margins_can_double, \
            range_x=list_range_x[i_shape], \
            margins_y=[y_margin_down, y_margin_up], \
            # range_y=list_range_y_rat[i_shape], \
            title=title_full_ratio_theory)
        list_obj[0].GetXaxis().SetLabelSize(0.1)
        list_obj[0].GetXaxis().SetTitleSize(0.1)
        list_obj[0].GetYaxis().SetLabelSize(0.12)
        list_obj[0].GetYaxis().SetTitleSize(0.12)
        list_obj[0].GetYaxis().SetTitleOffset(0.5)
        list_obj[0].GetYaxis().SetNdivisions(4)
        for gr, c in zip([hf_ratio_ivan_syst], [c_hf_ivan]):
            gr.SetMarkerColor(get_colour(c))
        # leg_data_mc_hf = list_obj_data_mc_hf_new[0]
        #leg_data_mc_hf.SetHeader("%s-tagged" % p_latexnhadron)
        # leg_data_mc_hf.SetTextSize(fontsize * 7/3)
        # leg_data_mc_hf.SetNColumns(2)
        if shape == "nsd":
            list_obj[0].GetXaxis().SetNdivisions(5)
        cshape_datamc_ivan.Update()

        # inclusive theory/data
        leg_pos = [.15, .8, .9, .95]
        list_obj = [incl_ratio_ivan_syst, incl_ratio_ivan_stat, line_1]
        labels_obj = [f"inclusive {text_ivan}", ""]
        colours = [get_colour(i, j) for i, j in zip((c_incl_ivan, c_incl_ivan), (2, 1))]
        markers = [m_incl_ivan, m_incl_ivan]
        y_margin_up = 0.05
        y_margin_down = 0.05
        cshape_datamc_ivan, list_obj_data_mc_hf_new_2 = make_plot("cshape_data_mc_incl_ivan_" + suffix, size=size_can_double, \
            can=cshape_datamc_ivan, pad=3, \
            list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g="3", offsets_xy=[1, 1.3 * 3/7], \
            colours=colours, markers=markers, leg_pos=None, margins_c=margins_can_double, \
            range_x=list_range_x[i_shape], \
            margins_y=[y_margin_down, y_margin_up], \
            # range_y=list_range_y_rat[i_shape], \
            title=title_full_ratio_theory)
        list_obj[0].GetXaxis().SetLabelSize(0.1)
        list_obj[0].GetXaxis().SetTitleSize(0.1)
        list_obj[0].GetYaxis().SetLabelSize(0.1)
        list_obj[0].GetYaxis().SetTitleSize(0.1)
        list_obj[0].GetYaxis().SetNdivisions(4)
        for gr, c in zip([incl_ratio_ivan_syst], [c_incl_ivan]):
            gr.SetMarkerColor(get_colour(c))
        # leg_data_mc_hf = list_obj_data_mc_hf_new_2[0]
        #leg_data_mc_hf.SetHeader("%s-tagged" % p_latexnhadron)
        # leg_data_mc_hf.SetTextSize(fontsize * 7/3)
        # leg_data_mc_hf.SetNColumns(2)
        if shape == "nsd":
            list_obj[0].GetXaxis().SetNdivisions(5)
        cshape_datamc_ivan.Update()

        # Draw LaTeX
        #y_latex = y_latex_top
        #list_latex_data_mc_hf = []
        #for text_latex in [text_alice, text_jets, text_ptjet, text_pth, text_sd]:
        #    latex = TLatex(x_latex, y_latex, text_latex)
        #    list_latex_data_mc_hf.append(latex)
        #    draw_latex(latex, textsize=fontsize)
        #    y_latex -= y_step
        #cshape_datamc_ivan.Update()
        pad1.RedrawAxis()
        pad2.RedrawAxis()
        pad3.RedrawAxis()
        cshape_datamc_ivan.SaveAs("%s/%s_datamc_ivan_%s.pdf" % (rootpath, shape, suffix))

main()
