# pylint: disable=missing-function-docstring
"""
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
    parser.add_argument("histname", help="histogram name pattern")
    parser.add_argument("outname", help="output file for the new histogram")
    parser.add_argument("maxval", type=float, help="maxval in histogram")
    args = parser.parse_args()

    with TFile(args.filename) as fin, TFile(args.outname, "recreate") as fout:
        objnames = fin.GetListOfKeys()
        print(f"objnames : {objnames}")
        histnames = [key.GetName() for key in fin.GetListOfKeys() if args.histname in key.GetName()]
        print(f"histnames: {histnames}")
        for histname in histnames:
            hist = fin.Get(histname)
            hist.SetDirectory(0)
            last_bin = hist.GetXaxis().FindBin(args.maxval)
            bins = []
            for binn in range(1, last_bin + 1):
                bins.append(hist.GetBinLowEdge(binn))
            print(f"Hist bins {bins}")
            hist2 = TH1F(histname, "", len(bins) - 1, array('d', bins))
            for binn in range(1, last_bin + 1):
                hist2.SetBinContent(binn + 1, hist.GetBinContent(binn + 1))
                hist2.SetBinError(binn + 1, hist.GetBinError(binn + 1))
                #print(f"Setting bin {binn + 1} low edge {hist2.GetBinLowEdge(binn + 1)} up edge {hist2.GetXaxis().GetBinUpEdge(binn + 1)} content to content from bin {binn + 1}: {hist2.GetBinContent(binn + 1)}")
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
