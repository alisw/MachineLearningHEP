# pylint: disable=missing-function-docstring, too-many-locals
"""
file:  compare_fractions.py
brief: Compare non-prompt corrected fractions for the systematic uncertainties analysis.
usage: python3 compare_fractions.py config_compare_fractions.json
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""

import argparse
import json
import math
import os
from array import array

from ROOT import (  # pylint: disable=import-error,no-name-in-module
    MakeNullPointer,
    TCanvas,
    TFile,
    TGraphAsymmErrors,
    TH1F,
    TLegend,
    TObject,
    TPaveText,
    TLine,
    gROOT,
    gStyle,
    kAzure,
    kBlack,
    kBlue,
    kCyan,
    kDashed,
    kGray,
    kGreen,
    kMagenta,
    kOrange,
    kRed,
    kTeal,
    kYellow
)

COLORS=[kBlack, kRed-3, kAzure-7, kGreen+2, kOrange-3, kBlue, kMagenta+2,
        kTeal+3, kGreen, kAzure+8,
        kYellow+3, kOrange-5, kMagenta+2, kBlue-6, kCyan+1, kGreen-6]
MODELS_COLORS=[kGray+1, kOrange-3, kCyan-2, kRed-9, kAzure-9, kBlue-6, kGreen-6, kOrange-5]
MODELS_STYLES=[3001, 3004, 3245, 3250, 3244, 3254, 3209, 3245, 3250, 3244, 3254, 3209]


def get_alice_text(cfg):
    if "alice_text" not in cfg:
        return None

    alice_text = TPaveText(0.17, 0.62, 0.50, 0.86, "brNDC")
    alice_text.SetTextFont(42)
    alice_text.SetTextSize(0.04)
    alice_text.SetBorderSize(0)
    alice_text.SetFillStyle(0)
    alice_text.SetTextAlign(11)

    alice_text_config = cfg["alice_text"]
    alice_text.AddText("#scale[1.35]{ALICE Preliminary}")
    alice_text.AddText("#scale[1.05]{pp,#kern[-0.05]{ #sqrt{#it{s}} = 13.6 TeV, |#it{y}| < 0.5}}")
    alice_text.AddText(f"#scale[1.20]{{{alice_text_config}}}")

    alice_text.Draw("same")

    return alice_text


def get_legend(x_1, y_1, x_2, y_2, num_hists, header=None):
    leg = TLegend(x_1, y_1, x_2, y_2)
    if num_hists > 4:
        leg.SetNColumns(2)
    if header:
        leg.SetHeader(header)
    leg.SetTextAlign(12)
    leg.SetTextSize(0.04)
    leg.SetMargin(0.3)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    return leg

def prepare_canvas(cname):
    canv = TCanvas(cname, "")
    canv.SetCanvasSize(900, 600)
    #canv.SetTickx()
    #canv.SetTicky()
    canv.SetLeftMargin(0.15)
    canv.SetBottomMargin(0.15)
    return canv


def save_canvas(canv, cfg, filename):
    for ext in ("png", "pdf"):
        canv.SaveAs(os.path.join(cfg["output"]["outdir"], f"{filename}.{ext}"))


def combine_syst_errors(syst_errors, value):
    err = 0.0
    err_perc = 0.0
    for syst in syst_errors:
        err += syst * syst
        err_perc += (100 * syst) * (100 * syst)
    err_perc = math.sqrt(err_perc)
    print(f"Combined percentage error: {err_perc:0.0f}")
    return math.sqrt(err) * value


def get_hist_limits(hist, graph_syst = None, miny = 0.0, maxy = 0.0):
    for binn in range(0, hist.GetNbinsX()):
        print(f"bin {binn + 1} [{hist.GetXaxis().GetBinLowEdge(binn + 1)}, "\
              f"{hist.GetXaxis().GetBinLowEdge(binn + 2)}) val {hist.GetBinContent(binn + 1)} "\
              f"err {hist.GetBinError(binn + 1)}")
        maxval = hist.GetBinContent(binn + 1) + hist.GetBinError(binn + 1)
        minval = hist.GetBinContent(binn + 1) - hist.GetBinError(binn + 1)
        if graph_syst:
            maxval = max(maxval, hist.GetBinContent(binn + 1) + graph_syst.GetErrorY(binn))
            minval = min(minval, hist.GetBinContent(binn + 1) - graph_syst.GetErrorY(binn))
        maxy = max(maxval, maxy)
        miny = min(minval, miny)
    return miny, maxy


def merge_fractions(inputdir, histname, filenames):
    with TFile.Open(os.path.join(inputdir, filenames[0])) as fin:
        reshist = fin.Get(histname).Clone()
        reshist.SetDirectory(0)

    for ind, file in enumerate(filenames[1:]):
        ind += 1
        with TFile.Open(os.path.join(inputdir, file)) as fin:
            hist = fin.Get(histname)
            reshist.SetBinContent(ind + 1, hist.GetBinContent(ind + 1))
            reshist.SetBinError(ind + 1, hist.GetBinError(ind + 1))

    return reshist


def set_hist_style(hist, color, y_axis, style=None):
    for axis in (hist.GetXaxis(), hist.GetYaxis()):
        axis.SetLabelFont(42)
        axis.SetLabelSize(0.05)
        axis.SetLabelOffset(0.02)
        axis.SetTitleFont(42)
        axis.SetTitleSize(0.06)
        axis.SetTitleOffset(1.3)
    hist.GetXaxis().SetTitle("#it{p}_{T}(GeV/#it{c})")
    hist.GetYaxis().SetTitle(y_axis)
    hist.GetYaxis().SetTitleSize(0.05)
    hist.GetXaxis().SetTitleOffset(1.1)

    hist.SetLineColor(color)
    hist.SetLineWidth(2)
    if style:
        hist.SetFillColor(color)
        hist.SetFillStyle(style)
        #hist.SetTitle("")
    else:
        hist.SetMarkerColor(color)
        hist.SetMarkerSize(1)
        hist.SetMarkerStyle(21)


def get_hist_for_label(label, color, cfg):
    if len(cfg["hists"][label]["file"]) == 1:
        with TFile.Open(os.path.join(cfg["inputdir"], cfg["hists"][label]["file"][0])) as fin:
            hist = fin.Get(cfg["histoname"])
            hist.SetDirectory(0)
    else:
        print(f"Merging histograms for {label}")
        hist = merge_fractions(cfg["inputdir"], cfg["histoname"], cfg["hists"][label]["file"])

    set_hist_style(hist, color, cfg["y_axis"])
    return hist


def get_graph_systematics(hist, label, color, cfg):
    if isinstance(cfg["hists"][label]["systematics"][0], str):
        with TFile.Open(os.path.join(cfg["inputdir"], \
                cfg["hists"][label]["systematics"][0])) as fin:
            graph_syst = fin.Get(cfg["hists"][label]["systematics"][1])
    else:
        graph_syst = TGraphAsymmErrors()
        graph_syst.SetName(f"graph_{label}_syst")
        for binn in range(hist.GetNbinsX()):
            syst_err = combine_syst_errors(cfg["hists"][label]["systematics"][binn],
                                               hist.GetBinContent(binn + 1))
            print(f"Syst error {label} bin {binn + 1} {syst_err}")
            x_point = hist.GetBinCenter(binn + 1)
            y_point = hist.GetBinContent(binn + 1)
            x_width = hist.GetBinWidth(binn + 1) / 4.0 # We want syst boxes to be of half-bin width
            if y_point != 0:
                graph_syst.SetPoint(binn, x_point, y_point)
                graph_syst.SetPointError(binn, x_width, x_width, syst_err, syst_err)
    set_hist_style(graph_syst, color, cfg["y_axis"])
    graph_syst.SetFillStyle(0)
    return graph_syst


def get_hist_model(label, color, style, cfg):
    with TFile.Open(os.path.join(cfg["inputdir"], cfg["models"][label]["file"])) as fin:
        hist = fin.Get(cfg["models"][label]["histoname"])
        hist.SetDirectory(0)

    set_hist_style(hist, color, cfg["y_axis"], style)
    #hist.SetTitle("")

    return hist


def plot_models(cfg, canv):
    maxy = 0.
    miny = 1000000.
    if cfg.get("models", None):
        hists_models = {}
        leg_models = get_legend(*cfg["legend_models"], len(cfg["models"]))
        leg_models.SetMargin(0.9)
        for ind, (label, color, style) in \
                enumerate(zip(cfg["models"], MODELS_COLORS, MODELS_STYLES)):
            hist = get_hist_model(label, color, style, cfg)
            print(f"hist model for {label}: {hist.GetName()}")
            miny, maxy = get_hist_limits(hist, None, miny, maxy)

            canv.cd()
            draw_opt = "sameE3" if ind != 0 else "E3"
            hist.Draw(draw_opt)
            leg_models.AddEntry(hist, label, "f")

            hists_models[label] = hist
    else:
        leg_models = None
        hists_models = None

    return canv, hists_models, leg_models, miny, maxy


def set_figs_limits(miny, maxy, hists, graphs, hists_models):
    margin = 0.1
    #k = 1.0 - 2 * margin
    #rangey = maxy - miny
    #miny = miny - margin / k * rangey
    #maxy = maxy + margin / k * rangey
    print(f"Hist maxy: {maxy} miny: {miny}")
    #miny = min(miny - margin * miny, 0)
    miny = miny - margin * miny
    if miny <= 0:
        miny = 0.006
    print(f"Recalculated hist maxy: {maxy + margin * maxy} miny: {miny}")
    if hists_models:
        for _, hist in hists_models.items():
            hist.GetYaxis().SetRangeUser(miny, maxy + margin * maxy)
    for _, hist in hists.items():
        hist.GetYaxis().SetRangeUser(miny, maxy + margin * maxy)
    if graphs:
        for graph_syst in graphs:
            graph_syst.GetYaxis().SetRangeUser(miny, maxy + margin * maxy)
    return hists, graphs, hists_models

def plot_compare(cfg):
    canv = prepare_canvas(f'c_{cfg["histoname"]}')
    if cfg.get("log_scale", False):
        canv.SetLogy()

    canv, hists_models, leg_models, miny, maxy = plot_models(cfg, canv)
    leg = get_legend(*cfg["legend"], len(cfg["hists"]))

    hists = {}
    central_graph = None
    graphs_syst = []
    for ind, (label, color) in enumerate(zip(cfg["hists"], COLORS)):
        hist = get_hist_for_label(label, color, cfg)
        print(label)
        miny, maxy = get_hist_limits(hist, None, miny, maxy)

        canv.cd()
        draw_opt = "sameE" if ind != 0 or hists_models else "E"
        hist.Draw(draw_opt)
        leg.AddEntry(hist, label, "p")

        hists[label] = hist

        if cfg["hists"][label].get("systematics", None):
            print("Plotting systematic")
            graph_syst = get_graph_systematics(hist, label, color, cfg)
            miny, maxy = get_hist_limits(hist, graph_syst, miny, maxy)
            graph_syst.Draw("sameE2")
            graphs_syst.append(graph_syst)
            if label == cfg["default"]:
                central_graph = graph_syst

    hists, graphs_syst, hists_models = set_figs_limits(miny, maxy, hists, graphs_syst, hists_models)

    leg.Draw()
    if leg_models:
        leg_models.Draw()

    alice_text = get_alice_text(cfg)

    return canv, hists, graphs_syst, hists_models, leg, leg_models, alice_text, central_graph


def get_average(hist, graph_syst):
    width_combined = hist.GetBinWidth(hist.GetNbinsX() -1) + hist.GetBinWidth(hist.GetNbinsX())
    val = ((hist.GetBinContent(hist.GetNbinsX() - 1) * hist.GetBinWidth(hist.GetNbinsX() - 1) +\
            hist.GetBinContent(hist.GetNbinsX()) * hist.GetBinWidth(hist.GetNbinsX())) /\
            width_combined)
    err = math.sqrt((hist.GetBinError(hist.GetNbinsX() - 1) *\
                     hist.GetBinWidth(hist.GetNbinsX() - 1) /\
                     width_combined) **2  +\
                    (hist.GetBinError(hist.GetNbinsX()) *\
                     hist.GetBinWidth(hist.GetNbinsX()) /\
                     width_combined) ** 2)
    syst_err = math.sqrt((graph_syst.GetErrorYlow(hist.GetNbinsX() - 2) *\
                          hist.GetBinWidth(hist.GetNbinsX() - 1) /\
                          width_combined) **2  +\
                         (graph_syst.GetErrorYlow(hist.GetNbinsX() - 1) *\
                          hist.GetBinWidth(hist.GetNbinsX()) /\
                          width_combined) ** 2)
    return val, err, syst_err


def hist_for_ratio(hist, graph, central_hist):
    hist2 = TH1F(hist.GetName(), "", central_hist.GetNbinsX(),
                 array('d', central_hist.GetXaxis().GetXbins()))
    graph2 = TGraphAsymmErrors()
    for binn in range(central_hist.GetNbinsX() - 1):
        hist2.SetBinContent(binn + 1, hist.GetBinContent(binn + 1))
        hist2.SetBinError(binn + 1, hist.GetBinError(binn + 1))
        graph2.SetPoint(binn, graph.GetPointX(binn), graph.GetPointY(binn))
        graph2.SetPointError(binn, graph.GetErrorX(binn), graph.GetErrorY(binn))
    val, err, syst_err = get_average(hist, graph)
    hist2.SetBinContent(hist2.GetNbinsX(), val)
    hist2.SetBinError(hist2.GetNbinsX(), err)
    graph2.SetPoint(hist2.GetNbinsX() - 1,
                    hist2.GetBinCenter(hist2.GetNbinsX()), val)
    graph2.SetPointError(hist2.GetNbinsX() - 1,
                         hist2.GetBinWidth(hist2.GetNbinsX()) / 4.0,
                         hist2.GetBinWidth(hist2.GetNbinsX()) / 4.0,
                         syst_err, syst_err)
    return hist2, graph2


def divide_syst_error(val, val1, val2, err1, err2):
    return val * math.sqrt((err1 / val1) **2 + (err2 / val2) **2)


def get_figs_ratio(central_graph, central_hist, hist_ratio, graph_ratio, label):
    histr = hist_ratio.Clone()
    histr.SetName(f"h_ratio_{label}")
    histr.Divide(hist_ratio, central_hist, 1., 1., "B")
    histr.GetXaxis().SetTitleOffset(1.10)
    for binn in range(1, histr.GetNbinsX() + 1):
        print(f"Ratio {binn}: {histr.GetBinContent(binn)}")

    graphr = None
    if central_graph:
        graphr = central_graph.Clone()
        graphr.SetName(f"g_ratio_{label}")
        for binn in range(1, central_hist.GetNbinsX() + 1):
            x_err = histr.GetBinWidth(binn) / 4.0
            y_low = divide_syst_error(histr.GetBinContent(binn),
                                      central_hist.GetBinContent(binn),
                                      hist_ratio.GetBinContent(binn),
                                      central_graph.GetErrorYlow(binn - 1),
                                      graph_ratio.GetErrorYlow(binn - 1))
            y_high = divide_syst_error(histr.GetBinContent(binn),
                                       central_hist.GetBinContent(binn),
                                       hist_ratio.GetBinContent(binn),
                                       central_graph.GetErrorYhigh(binn - 1),
                                       graph_ratio.GetErrorYhigh(binn - 1))
            graphr.SetPoint(binn - 1, histr.GetBinCenter(binn), histr.GetBinContent(binn))
            graphr.SetPointError(binn - 1, x_err, x_err, y_low, y_high)
            print(f"Central graph bin {binn-1} low {central_graph.GetErrorYlow(binn-1)} "\
                  f"{label} low: {graph_ratio.GetErrorYlow(binn-1)} "\
                  f"up {central_graph.GetErrorYhigh(binn-1)} "\
                  f"{label} up: {graph_ratio.GetErrorYhigh(binn-1)}")
    return histr, graphr


def plot_ratio_histos(canvr, legr, hists, graphs, central_hist,
                      central_label, central_graph, styles, y_axis):
    maxx = 0.0
    miny = 0.05
    maxy = 300
    histsr = []
    graphsr = []

    for ind, (label, color, style) in enumerate(zip(hists, COLORS, styles)):
        print(f"central hist bins: {central_hist.GetNbinsX()} "\
              f"{label} bins: {hists[label].GetNbinsX()}")
        if label != central_label and hists[label].GetNbinsX() == central_hist.GetNbinsX():
            graph = graphs[ind] if graphs else None
            #hist_ratio, graph_ratio = hist_for_ratio(hists[label], graph, central_hist)
            hist_ratio = hists[label]
            graph_ratio = graph

            histr, graphr = get_figs_ratio(central_graph, central_hist,
                                           hist_ratio, graph_ratio, label)
            #set_hist_style(histr, color, "Ratio to INEL > 0")
            histr.GetYaxis().SetTitle(y_axis)

            if style:
                set_hist_style(histr, color, y_axis, style)
                draw_opt = "sameE3" if ind != 0 else "E3"
            else:
                draw_opt = "sameE"
            histr.SetMaximum(maxy)
            histr.SetMinimum(miny)
            canvr.cd()
            histr.Draw(draw_opt)
            if style:
                histr2 = histr.Clone()
                histr2.SetFillStyle(0)
                histr2.SetFillColor(0)
                histr2.SetMarkerStyle(0)
                histr2.Draw("hist same L")
                histr2.SetFillStyle(style)
                histsr.append(histr2)
            histsr.append(histr)
            if graphr:
                set_hist_style(graphr, color, y_axis)
                graphr.Draw("sameE2")
                graphsr.append(graphr)
            if style and ind == 1:
                entry = legr.AddEntry(MakeNullPointer(TObject), "PYTHIA 8.243 Monash", "f")
                entry.SetFillColor(kBlack)
                entry.SetFillStyle(style)
            elif not style:
                legr.AddEntry(histr, label, "p")
            maxx = max(maxx, histr.GetBinLowEdge(histr.GetNbinsX() + 1))
    return canvr, legr, histsr, graphsr, maxx


def plot_ratio(cfg, hists, graphs_syst, central_graph, hists_models):
    canvr = prepare_canvas(f'c_ratio_{cfg["histoname"]}')
    canvr.SetLogy()

    if hists_models:
        leg_models = get_legend(*cfg["legend_ratio_models"], len(cfg["models"]))
        leg_models.SetMargin(0.5)
        central_hist = hists_models[cfg["model_default"]]
        canvr, leg_models, histsr_models, _, maxx =\
                plot_ratio_histos(canvr, leg_models, hists_models, None,
                                  central_hist, cfg["model_default"], None,
                                  [3001] * len(cfg["models"]), cfg["y_axis"])
        leg_models.Draw()
    else:
        histsr_models = []
        leg_models = None

    legr = get_legend(*cfg["legend_ratio"], len(cfg["hists"]),
                      "<d#it{N}_{ch}/d#it{#eta}>:")
    central_hist = hists[cfg["default"]]
    canvr, legr, histsr, graphsr, maxx =\
            plot_ratio_histos(canvr, legr, hists, graphs_syst,
                              central_hist, cfg["default"], central_graph,
                              [None] * len(cfg["hists"]), cfg["y_axis"])

    legr.Draw()

    line = TLine(histsr[0].GetBinLowEdge(1), 1.0, maxx, 1.0)
    line.SetLineColor(COLORS[len(histsr)])
    line.SetLineWidth(3)
    line.SetLineStyle(kDashed)
    line.Draw()

    alice_text = get_alice_text(cfg)

    return canvr, histsr, graphsr, histsr_models, legr, line, alice_text, leg_models


def calc_systematics(cfg, hists):
    syst_errors = []
    central_hist = hists[cfg["default"]]

    for binn in range(central_hist.GetNbinsX()):
        syst_err_bin = 0.00
        count = 0
        for label in hists:
            if label != cfg["default"] and hists[label].GetNbinsX() == central_hist.GetNbinsX():
                syst_err = float("inf") if central_hist.GetBinContent(binn + 1) == 0 else \
                           (hists[label].GetBinContent(binn + 1) - \
                             central_hist.GetBinContent(binn + 1)) / \
                             central_hist.GetBinContent(binn + 1)
                syst_err_bin += syst_err * syst_err
                count += 1
        if count == 0:
            return
        syst_err_bin = 100 * (math.sqrt(syst_err_bin / count))
        syst_errors.append(syst_err_bin)

    str_err = "Systematic errors:"
    for err in syst_errors:
        str_err = f"{str_err} {err:0.2f}"
    print(str_err)


def main():
    """
    Main function.
    """
    gROOT.SetBatch(True)

    gStyle.SetOptStat(0)
    gStyle.SetOptTitle(0)
    gStyle.SetFrameLineWidth(2)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("config", help="JSON config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf8") as fil:
        cfg = json.load(fil)

    with TFile(os.path.join(cfg["output"]["outdir"],
               f'{cfg["output"]["file"]}.root'), "recreate") as output:

        (canv, hists, graphs_syst, hists_models,
            leg, leg_models, alice_text, central_graph) = plot_compare(cfg) # pylint: disable=unused-variable
        output.cd()
        canv.Write()
        save_canvas(canv, cfg, cfg["output"]["file"])
        for _, hist in hists.items():
            hist.Write()
        if graphs_syst:
            for graph in graphs_syst:
                graph.Write()
        if hists_models:
            for _, hist in hists_models.items():
                hist.Write()

        canvr, histr, graphr, histr_models, legr, line, alice_text, leg_models =\
                plot_ratio(cfg, hists, graphs_syst, central_graph, hists_models) # pylint: disable=unused-variable
        output.cd()
        canvr.Write()
        save_canvas(canvr, cfg, f'{cfg["output"]["file"]}_ratio')
        for hist in histr:
            hist.Write()
        for graph in graphr:
            graph.Write()
        for hist in histr_models:
            hist.Write()

        calc_systematics(cfg, hists)


if __name__ == "__main__":
    main()
