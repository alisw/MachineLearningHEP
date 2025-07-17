# pylint: disable=missing-function-docstring
"""
file:  add_pt_bins.py
brief: Add 0-1 and 24-25 dummy pT bins to extend x-range of input histogram.
usage: python3 add_pt_bins.py file.root my_histo file_out.root
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""

import argparse
import math
from array import array

from ROOT import (  # pylint: disable=import-error,no-name-in-module
    gROOT,
    TFile,
    TH1F
)


def main():
    """
    Main function.
    """
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("filename", help="input file with histogram")
    parser.add_argument("histname", help="histogram name")
    parser.add_argument("outname", help="output file for the new histogram")
    args = parser.parse_args()

    with TFile(args.filename) as fin, TFile(args.outname, "recreate") as fout:
        hist = fin.Get(args.histname)
        hist.SetDirectory(0)
        first_bin = 1
        #last_bin = hist.GetXaxis().FindBin(12.0)
        last_bin = hist.GetNbinsX()
        bins = [0.0]
        #bins = []
        empty_bins = len(bins)
        for binn in range(first_bin, last_bin + 1):
            bins.append(hist.GetBinLowEdge(binn))
        #last_bins = [24.0, 25.0]
        last_bins = [24.0]
        bins += last_bins
        print(f"Hist bins {bins}")
        hist2 = TH1F(args.histname, "", len(bins) - 1, array('d', bins))
        for binn in range(empty_bins, last_bin + 1):
            hist2.SetBinContent(binn + 1, hist.GetBinContent(binn + 1 - empty_bins))
            hist2.SetBinError(binn + 1, hist.GetBinError(binn + 1 - empty_bins))
            print(f"Setting bin {binn + 1} low edge {hist2.GetBinLowEdge(binn + 1)} up edge {hist2.GetXaxis().GetBinUpEdge(binn + 1)} content to content from bin {binn + 1 - empty_bins}: {hist2.GetBinContent(binn + 1)}")
        #last_bin = hist2.GetNbinsX()
        #width_combined = hist.GetBinWidth(hist.GetNbinsX() -1) + hist.GetBinWidth(hist.GetNbinsX())
        #hist2.SetBinContent(last_bin,
        #                    ((hist.GetBinContent(hist.GetNbinsX() - 1) * hist.GetBinWidth(hist.GetNbinsX() - 1) +\
        #                     hist.GetBinContent(hist.GetNbinsX()) * hist.GetBinWidth(hist.GetNbinsX())) /\
        #                    width_combined))
        #hist2.SetBinError(last_bin,
        #                  math.sqrt((hist.GetBinError(hist.GetNbinsX() - 1) * hist.GetBinWidth(hist.GetNbinsX() - 1) /\
        #                            width_combined) **2  +\
        #                            (hist.GetBinError(hist.GetNbinsX()) * hist.GetBinWidth(hist.GetNbinsX()) /\
        #                            width_combined) ** 2))
        #print(f"Setting bin {last_bin} low edge {hist2.GetBinLowEdge(last_bin)} up edge {hist2.GetXaxis().GetBinUpEdge(last_bin)} content to content from bins {hist.GetNbinsX()-1}, {hist.GetNbinsX()}: {hist2.GetBinContent(last_bin)}")
        hist2.SetMarkerSize(hist.GetMarkerSize())
        hist2.SetMarkerColor(hist.GetMarkerColor())
        hist2.SetMarkerStyle(hist.GetMarkerStyle())
        hist2.SetLineWidth(hist.GetLineWidth())
        hist2.SetLineColor(hist.GetLineColor())
        hist2.SetLineStyle(hist.GetLineStyle())
        fout.cd()
        hist2.Write()


if __name__ == "__main__":
    main()
