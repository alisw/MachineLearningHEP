#!/usr/bin/env python
"""
file:  merge_fractions.py
brief: Merge points from non-prompt fraction plots from different cutvar scans into a single plot
usage: ./merge_fractions.py my-plot.png file1.root file2.root file3.root
author: Maja Karwowska <mkarwowska@cern.ch>, CERN / Warsaw University of Technology
"""

import argparse

from ROOT import (  # pylint: disable=import-error,no-name-in-module
    TCanvas,
    TFile,
    TH1,
    gROOT,
    kOrange
)

HISTNAME = "hCorrFracNonPrompt"

def main():
    """
    Main function.
    """
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("outname", help="output filename")
    parser.add_argument("oldname", help="old results filename")
    parser.add_argument("files", nargs='+', help="input ROOT files")
    args = parser.parse_args()

    canv = TCanvas(f"c_{HISTNAME}", "")
    canv.SetCanvasSize(800, 600)

    rfile = TFile(args.files[0])
    hist = rfile.Get(HISTNAME)
    reshist = hist.Clone()

    for ind, file in enumerate(args.files):
        rfile2 = TFile(file)
        hist = rfile2.Get(HISTNAME)
        print(f"{ind + 1} bin content {hist.GetBinContent(ind + 1)}")
        reshist.SetBinContent(ind + 1, hist.GetBinContent(ind + 1))
        reshist.SetBinError(ind + 1, hist.GetBinError(ind + 1))
    for ind in range(2):
        reshist.SetBinContent(ind + 1 + len(args.files), 0.0)
        reshist.SetBinError(ind + 1 + len(args.files), 0.0)

    reshist.SetMaximum(0.25)
    reshist.SetMinimum(0.0)
    reshist.Draw()

    oldfile = TFile(args.oldname)
    oldhistt = oldfile.Get(HISTNAME)
    oldhist = oldhistt.Clone()
    oldhist.SetMarkerColor(kOrange)
    oldhist.SetLineColor(kOrange)
    for ind in range(2):
        oldhist.SetBinContent(oldhist.GetNbinsX() - ind, 0.0)
        oldhist.SetBinError(oldhist.GetNbinsX() - ind, 0.0)
    oldhist.Draw("same")

    canv.SaveAs(args.outname)


if __name__ == "__main__":
    main()
