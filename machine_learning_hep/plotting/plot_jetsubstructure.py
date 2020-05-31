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
import yaml
# pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TLatex, TLine, TGaxis, gROOT, gStyle
from machine_learning_hep.utilities import make_message_notfound
from machine_learning_hep.utilities import get_colour, get_marker, draw_latex
from machine_learning_hep.utilities import make_plot, get_y_window_his, get_y_window_gr, get_plot_range
from machine_learning_hep.logger import get_logger

def main(): # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    """
    Main plotting function
    """
    gROOT.SetBatch(True)

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

    # HF ratio
    nameobj = "%s_hf_ratio_%d_stat" % (shape, ibin2)
    hf_ratio_stat = file_results.Get(nameobj)
    if not hf_ratio_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    nameobj = "%s_hf_ratio_%d_syst" % (shape, ibin2)
    hf_ratio_syst = file_results.Get(nameobj)
    if not hf_ratio_syst:
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
    nameobj = "%s_incl_pythia_%d_syst" % (shape, ibin2)
    incl_pythia_syst = file_results.Get(nameobj)
    if not incl_pythia_syst:
        logger.fatal(make_message_notfound(nameobj, file_in))

    # inclusive ratio
    nameobj = "%s_incl_ratio_%d_stat" % (shape, ibin2)
    incl_ratio_stat = file_results.Get(nameobj)
    if not incl_ratio_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    nameobj = "%s_incl_ratio_%d_syst" % (shape, ibin2)
    incl_ratio_syst = file_results.Get(nameobj)
    if not incl_ratio_syst:
        logger.fatal(make_message_notfound(nameobj, file_in))

    # quark PYTHIA
    nameobj = "%s_quark_pythia_%d_stat" % (shape, ibin2)
    quark_pythia_stat = file_results.Get(nameobj)
    if not quark_pythia_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    nameobj = "%s_quark_pythia_%d_syst" % (shape, ibin2)
    quark_pythia_syst = file_results.Get(nameobj)
    if not quark_pythia_syst:
        logger.fatal(make_message_notfound(nameobj, file_in))

    # gluon PYTHIA
    nameobj = "%s_gluon_pythia_%d_stat" % (shape, ibin2)
    gluon_pythia_stat = file_results.Get(nameobj)
    if not gluon_pythia_stat:
        logger.fatal(make_message_notfound(nameobj, file_in))
    nameobj = "%s_gluon_pythia_%d_syst" % (shape, ibin2)
    gluon_pythia_syst = file_results.Get(nameobj)
    if not gluon_pythia_syst:
        logger.fatal(make_message_notfound(nameobj, file_in))

    # plot the results with systematic uncertainties and models

    size_can = [800, 800]
    offsets_axes = [0.8, 1.1]
    margins_can = [0.1, 0.13, 0.1, 0.03]
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
    y_step = 0.055

    title_x = v_varshape_latex
    title_y = "(1/#it{N}_{jet}) d#it{N}/d%s" % v_varshape_latex
    title_full = ";%s;%s" % (title_x, title_y)
    title_full_ratio = ";%s;data/MC: ratio of %s" % (title_x, title_y)

    text_alice = "#bf{ALICE} Preliminary, pp, #sqrt{#it{s}} = 13 TeV"
    text_alice_sim = "#bf{ALICE} Simulation, pp, #sqrt{#it{s}} = 13 TeV"
    text_pythia = "PYTHIA 8 (Monash)"
    text_pythia_split = "#splitline{PYTHIA 8}{(Monash)}"
    text_jets = "charged jets, anti-#it{k}_{T}, #it{R} = 0.4"
    text_ptjet = "%g #leq %s < %g GeV/#it{c}, #left|#it{#eta}_{jet}#right| #leq 0.5" % (lvar2_binmin_reco[ibin2], p_latexbin2var, lvar2_binmax_reco[ibin2])
    text_pth = "%g #leq #it{p}_{T}^{%s} < %g GeV/#it{c}, #left|#it{y}_{%s}#right| #leq 0.8" % (lpt_finbinmin[0], p_latexnhadron, min(lpt_finbinmax[-1], lvar2_binmax_reco[ibin2]), p_latexnhadron)
    text_ptcut = "#it{p}_{T, incl. ch. jet}^{leading track} #geq 5.33 GeV/#it{c}"
    text_ptcut_sim = "#it{p}_{T, incl. ch. jet}^{leading h^{#pm}} #geq 5.33 GeV/#it{c} (varied)"
    text_sd = "Soft Drop (#it{z}_{cut} = 0.1, #it{#beta} = 0)"

    title_thetag = "#it{#theta}_{g} = #it{R}_{g}/#it{R}"
    radius_jet = 0.4

    # colour and marker indeces
    c_hf_data = 0
    c_incl_data = 1
    c_hf_mc = 2
    c_incl_mc = 6
    c_quark_mc = 5
    c_gluon_mc = 0

    # markers
    m_hf_data = get_marker(0)
    m_incl_data = get_marker(1)
    m_hf_mc = get_marker(0, 2)
    m_incl_mc = get_marker(1, 2)
    m_quark_mc = get_marker(2)
    m_gluon_mc = get_marker(3)

    # make the horizontal error bars smaller
    if shape == "nsd":
        for gr in [hf_data_syst, incl_data_syst, hf_ratio_syst, incl_ratio_syst, incl_pythia_syst, quark_pythia_syst, gluon_pythia_syst]:
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

    # data and PYTHIA, HF

    leg_pos = [.72, .65, .85, .85]
    list_obj = [hf_data_syst_cl, hf_data_stat, hf_pythia_stat]
    labels_obj = ["data", "", text_pythia_split]
    colours = [get_colour(i, j) for i, j in zip((c_hf_data, c_hf_data, c_hf_mc), (2, 1, 1))]
    markers = [m_hf_data, m_hf_data, m_hf_mc]
    y_margin_up = 0.4
    y_margin_down = 0.05
    cshape_data_mc_hf, list_obj_data_mc_hf_new = make_plot("cshape_data_mc_hf_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=margins_can, \
        title=title_full)
    for gr, c in zip([hf_data_syst_cl], [c_hf_data]):
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
    colours = [get_colour(i, j) for i, j in zip((c_incl_data, c_incl_mc, c_incl_data, c_incl_mc), (2, 2, 1, 1))]
    markers = [m_incl_data, m_incl_mc, m_incl_data, m_incl_mc]
    y_margin_up = 0.4
    y_margin_down = 0.05
    cshape_data_mc_incl, list_obj_data_mc_incl_new = make_plot("cshape_data_mc_incl_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=margins_can, \
        title=title_full)
    for gr, c in zip([incl_data_syst, incl_pythia_syst], [c_incl_data, c_incl_mc]):
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

    #leg_pos = [.72, .7, .85, .85] # with header
    leg_pos = [.72, .75, .85, .85] # without header
    list_obj = [hf_ratio_syst, line_1, incl_ratio_syst, hf_ratio_stat, incl_ratio_stat]
    labels_obj = ["%s-tagged" % p_latexnhadron, "inclusive"]
    colours = [get_colour(i, j) for i, j in zip((c_hf_data, c_incl_data, c_hf_data, c_incl_data), (2, 2, 1, 1))]
    markers = [m_hf_data, m_incl_data, m_hf_data, m_incl_data]
    y_margin_up = 0.52
    y_margin_down = 0.05
    if shape == "nsd":
        y_margin_up = 0.22
    cshape_ratio, list_obj_ratio_new = make_plot("cshape_ratio_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, margins_y=[y_margin_down, y_margin_up], margins_c=margins_can, \
        title=title_full_ratio)
    cshape_ratio.Update()
    for gr, c in zip((hf_ratio_syst, incl_ratio_syst), (c_hf_data, c_incl_data)):
        gr.SetMarkerColor(get_colour(c))
    leg_ratio = list_obj_ratio_new[0]
    leg_ratio.SetTextSize(fontsize)
    #leg_ratio.SetHeader("data/MC")
    if shape == "nsd":
        hf_ratio_syst.GetXaxis().SetNdivisions(5)
    cshape_ratio.Update()
    if shape == "rg":
        # plot the theta_g axis
        gr_frame = hf_ratio_syst
        axis_rg = gr_frame.GetXaxis()
        rg_min = axis_rg.GetBinLowEdge(axis_rg.GetFirst())
        rg_max = axis_rg.GetBinUpEdge(axis_rg.GetLast())
        thetag_min = rg_min / radius_jet
        thetag_max = rg_max / radius_jet
        y_axis = cshape_ratio.GetUymax()
        axis_thetag = TGaxis(rg_min, y_axis, rg_max, y_axis, thetag_min, thetag_max, 510, "-")
        axis_thetag.SetTitle(title_thetag)
        axis_thetag.SetTitleSize(size_thg)
        axis_thetag.SetLabelSize(0.036)
        axis_thetag.SetTitleFont(42)
        axis_thetag.SetLabelFont(42)
        axis_thetag.SetLabelOffset(0)
        axis_thetag.SetTitleOffset(offset_thg)
        cshape_ratio.SetTickx(0)
        axis_thetag.Draw("same")
    # Draw LaTeX
    y_latex = y_latex_top
    list_latex_ratio = []
    for text_latex in [text_alice, text_jets, text_ptjet, text_pth, text_ptcut, text_sd, text_pythia]:
        latex = TLatex(x_latex, y_latex, text_latex)
        list_latex_ratio.append(latex)
        draw_latex(latex, textsize=fontsize)
        y_latex -= y_step
    cshape_ratio.Update()
    cshape_ratio.SaveAs("%s/%s_ratio_%s.pdf" % (rootpath, shape, suffix))

    # PYTHIA, HF, inclusive, quark, gluon

    incl_pythia_syst_cl = incl_pythia_syst.Clone()

    y_min_h, y_max_h = get_y_window_his([hf_pythia_stat, incl_pythia_stat, quark_pythia_stat, gluon_pythia_stat])
    y_min_g, y_max_g = get_y_window_gr([incl_pythia_syst, quark_pythia_syst, gluon_pythia_syst])
    y_min = min(y_min_h, y_min_g)
    y_max = max(y_max_h, y_max_g)
    y_margin_up = 0.46
    y_margin_down = 0.05
    y_min_plot, y_max_plot = get_plot_range(y_min, y_max, y_margin_down, y_margin_up)

    #leg_pos = [.6, .65, .75, .85]
    leg_pos = [.72, .55, .85, .85]
    list_obj = [incl_pythia_syst, quark_pythia_syst, gluon_pythia_syst, hf_pythia_stat, incl_pythia_stat, quark_pythia_stat, gluon_pythia_stat]
    labels_obj = ["inclusive", "quark", "gluon", "%s-tagged" % p_latexnhadron]
    colours = [get_colour(i, j) for i, j in zip((c_incl_mc, c_quark_mc, c_gluon_mc, c_hf_mc, c_incl_mc, c_quark_mc, c_gluon_mc), (2, 2, 2, 1, 1, 1, 1))]
    markers = [m_incl_mc, m_quark_mc, m_gluon_mc, m_hf_mc, m_incl_mc, m_quark_mc, m_gluon_mc]
    y_margin_up = 0.46
    y_margin_down = 0.05
    cshape_mc, list_obj_mc_new = make_plot("cshape_mc_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, range_y=[y_min_plot, y_max_plot], margins_c=margins_can, \
        title=title_full)
    cshape_mc.Update()
    for gr, c in zip((incl_pythia_syst, quark_pythia_syst, gluon_pythia_syst), (c_incl_mc, c_quark_mc, c_gluon_mc)):
        gr.SetMarkerColor(get_colour(c))
    leg_mc = list_obj_mc_new[0]
    leg_mc.SetTextSize(fontsize)
    leg_mc.SetHeader(text_pythia_split)
    if shape == "nsd":
        incl_pythia_syst.GetXaxis().SetNdivisions(5)
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

    # PYTHIA, HF, quark, gluon

    #leg_pos = [.6, .65, .75, .85]
    leg_pos = [.72, .61, .85, .85]
    list_obj = [quark_pythia_syst, gluon_pythia_syst, hf_pythia_stat, quark_pythia_stat, gluon_pythia_stat]
    labels_obj = ["quark", "gluon", "%s-tagged" % p_latexnhadron]
    colours = [get_colour(i, j) for i, j in zip((c_quark_mc, c_gluon_mc, c_hf_mc, c_quark_mc, c_gluon_mc), (2, 2, 1, 1, 1))]
    markers = [m_quark_mc, m_gluon_mc, m_hf_mc, m_quark_mc, m_gluon_mc]
    y_margin_up = 0.46
    y_margin_down = 0.05
    cshape_mc, list_obj_mc_new = make_plot("cshape_mc_qgd_" + suffix, size=size_can, \
        list_obj=list_obj, labels_obj=labels_obj, opt_leg_g=opt_leg_g, opt_plot_g=opt_plot_g, offsets_xy=offsets_axes, \
        colours=colours, markers=markers, leg_pos=leg_pos, range_y=[y_min_plot, y_max_plot], margins_c=margins_can, \
        title=title_full)
    cshape_mc.Update()
    for gr, c in zip((quark_pythia_syst, gluon_pythia_syst), (c_quark_mc, c_gluon_mc)):
        gr.SetMarkerColor(get_colour(c))
    leg_mc = list_obj_mc_new[0]
    leg_mc.SetTextSize(fontsize)
    leg_mc.SetHeader(text_pythia_split)
    if shape == "nsd":
        quark_pythia_syst.GetXaxis().SetNdivisions(5)
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
    cshape_mc.SaveAs("%s/%s_mc_qgd_%s.pdf" % (rootpath, shape, suffix))

    # PYTHIA, HF, inclusive

    #leg_pos = [.6, .65, .75, .85]
    leg_pos = [.72, .67, .85, .85]
    list_obj = [incl_pythia_syst_cl, incl_pythia_stat, hf_pythia_stat]
    labels_obj = ["inclusive", "", "%s-tagged" % p_latexnhadron]
    colours = [get_colour(i, j) for i, j in zip((c_incl_mc, c_incl_mc, c_hf_mc), (2, 1, 1))]
    markers = [m_incl_mc, m_incl_mc, m_hf_mc]
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
    incl_pythia_syst_cl.SetMarkerColor(get_colour(c_incl_mc))
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

main()
