import argparse
import json
import os

from ROOT import (  # pylint: disable=import-error,no-name-in-module
    TCanvas,
    TFile,
    TLegend,
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

from compare_fractions import get_legend, prepare_canvas, save_canvas, set_hist_style

COLORS=[kBlack, kRed-3, kAzure-7, kMagenta+1, kGreen+2, kOrange-3, kBlue, kTeal+3, kGreen, kAzure+8,
        kYellow+3, kOrange-5, kMagenta+2, kBlue-6, kCyan+1, kGreen-6]


def get_hist_limits(hist, miny = 0.0, maxy = 0.0):
    for binn in range(hist.GetN()):
        print(f"bin {binn} [{hist.GetPointX(binn)}, "\
              f"val {hist.GetPointY(binn)} "\
              f"err {hist.GetErrorYlow(binn)}, {hist.GetErrorYhigh(binn)}")
        maxval = hist.GetPointY(binn) + hist.GetErrorYhigh(binn)
        minval = hist.GetPointY(binn) - hist.GetErrorYlow(binn)
        maxy = max(maxval, maxy)
        miny = min(minval, miny)
    return miny, maxy


def main():
    gROOT.SetBatch(True)

    gStyle.SetOptStat(0)
    gStyle.SetFrameLineWidth(2)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("config", help="JSON config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf8") as fil:
        cfg = json.load(fil)

    with TFile(os.path.join(cfg["output"]["outdir"],
               f'{cfg["output"]["file"]}.root'), "recreate") as output:

        canv = prepare_canvas(f'c_{cfg["histoname"]}')
        leg = get_legend(*cfg["legend"], len(cfg["hists"]))

        maxy = 0.
        miny = 1.
        hists = []
        for ind, (label, color) in enumerate(zip(cfg["hists"], COLORS)):
            with TFile.Open(os.path.join(cfg["inputdir"], cfg["hists"][label]["file"][0])) as fin:
                hist = fin.Get(cfg["histoname"])
                print(f'hist {cfg["histoname"]}: {hist}')
            set_hist_style(hist, color, cfg["y_axis"])
            print(label)
            miny, maxy = get_hist_limits(hist, miny, maxy)

            canv.cd()
            draw_opt = "same" if ind != 0 else ""
            hist.Draw(draw_opt)
            leg.AddEntry(hist, label, "p")

            hists.append(hist)

        margin = 0.1
        print(f"Hist maxy: {maxy} miny: {miny}")
        for hist in hists:
            #hist.GetYaxis().SetRangeUser(miny - margin * miny, maxy + margin * maxy)
            hist.GetYaxis().SetRangeUser(0.0, 0.7)
            #hist.GetYaxis().SetRangeUser(0.5, 1.0)
            hist.GetXaxis().SetRangeUser(0.0, 25.0)

        leg.Draw()

        output.cd()
        canv.Write()
        save_canvas(canv, cfg, cfg["output"]["file"])
        for hist in hists:
            hist.Write()


if __name__ == "__main__":
    main()
