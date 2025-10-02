# pylint: disable=missing-function-docstring
"""
file:  remove_high_pt.py
brief: Remove bins with pT > max_pt in all histograms matching my_histos_pattern in the input file.root.
usage: python3 remove_high_pt.py file.root my_histos_pattern file_out.root max_pt
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""

import argparse
from array import array

from ROOT import TH1F, TFile, gROOT  # pylint: disable=import-error,no-name-in-module


def main():
    """
    Main function.
    """
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("filename", help="input file with histogram")
    parser.add_argument("histname", help="histogram name pattern")
    parser.add_argument("outname", help="output file for the new histogram")
    parser.add_argument("maxval", type=float, help="maxval in histogram")
    args = parser.parse_args()

    with TFile(args.filename) as fin, TFile(args.outname, "recreate") as fout:
        objnames = fin.GetListOfKeys()
        histnames = [key.GetName() for key in fin.GetListOfKeys() if args.histname in key.GetName()]
        for histname in histnames:
            hist = fin.Get(histname)
            hist.SetDirectory(0)
            last_bin = hist.GetXaxis().FindBin(args.maxval)
            bins = []
            for binn in range(1, last_bin + 1):
                bins.append(hist.GetBinLowEdge(binn))
            hist2 = TH1F(histname, "", len(bins) - 1, array('d', bins))
            for binn in range(1, last_bin + 1):
                hist2.SetBinContent(binn, hist.GetBinContent(binn))
                hist2.SetBinError(binn, hist.GetBinError(binn))
                print(f"Setting bin {binn} low edge {hist2.GetBinLowEdge(binn)} " \
                      f"up edge {hist2.GetXaxis().GetBinUpEdge(binn)} content to content " \
                      f"from bin {binn}: {hist2.GetBinContent(binn)}")
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
