# pylint: disable=missing-function-docstring
"""
file:  modify_input.py
brief: Perform adjustments on input histogram.
usage: python3 modify_input.py file.root my_histo file_out.root
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""

import argparse
from array import array
import math

from ROOT import (  # pylint: disable=import-error,no-name-in-module
    gROOT,
    TFile,
    TH1F
)

OUTPUT_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 24]
BR = 0.0623

def main():
    """
    Main function.
    """
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser(description="Arguments to pass")
    parser.add_argument("filename", help="input file with histogram")
    parser.add_argument("histname", help="histogram name")
    parser.add_argument("outhistname", help="outhistogram name")
    parser.add_argument("outname", help="output file for the new histogram")
    args = parser.parse_args()

    with TFile(args.filename) as fin, TFile(args.outname, "recreate") as fout:
        hist = fin.Get(args.histname)
        hist.SetDirectory(0)
        #hist.Scale(0.000000001)
        hist.Scale(1./BR)
        hist2 = TH1F(args.outhistname, "", len(OUTPUT_BINS) - 1, array('d', OUTPUT_BINS))
        merge_bins = [20] # dummy number so as not to merge [7, 9]
        ind = 0
        for binn in range(1, hist.GetNbinsX() + 1):
            print(f"Old hist bin {binn} low edge {hist.GetBinLowEdge(binn)} "\
                  f"up edge {hist.GetXaxis().GetBinUpEdge(binn)} "\
                  f"content: {hist.GetBinContent(binn)} +/- {hist.GetBinError(binn)}")
        for binn in range(1, hist2.GetNbinsX() + 1):
            if binn < merge_bins[0]:
                hist2.SetBinContent(binn, hist.GetBinContent(binn))
                hist2.SetBinError(binn, hist.GetBinError(binn))
            elif ind >= len(merge_bins) or binn > merge_bins[0] + len(merge_bins) / 2:
                hist2.SetBinContent(binn, hist.GetBinContent(binn + ind))
                hist2.SetBinError(binn, hist.GetBinError(binn + ind))
            else:
                bin1 = merge_bins[ind]
                bin2 = merge_bins[ind] + 1
                weight_sum = hist.GetBinWidth(bin1) + hist.GetBinWidth(bin2)
                average = hist.GetBinContent(bin1) * hist.GetBinWidth(bin1) + hist.GetBinContent(bin2) * hist.GetBinWidth(bin2)
                print(f"bin {bin1} width {hist.GetBinWidth(bin1)} bin2 {bin2} width {hist.GetBinWidth(bin2)}")
                print(f"weight sum: {weight_sum} average: {hist.GetBinContent(bin1) * hist.GetBinWidth(bin1)} + "
                      f"{hist.GetBinContent(bin2) + hist.GetBinWidth(bin2)} average: {average}")
                hist2.SetBinContent(binn,
                        (hist.GetBinContent(bin1) * hist.GetBinWidth(bin1) +\
                         hist.GetBinContent(bin2) * hist.GetBinWidth(bin2)) /\
                        weight_sum)
                print(f"bin {bin1} error {hist.GetBinError(bin1)} bin2 {hist.GetBinError(bin2)}\n" 
                        f"scaled: {hist.GetBinWidth(bin1) * hist.GetBinError(bin1)}, "\
                        f"{hist.GetBinWidth(bin2) * hist.GetBinError(bin2)}\n"\
                        f"divided: {(hist.GetBinWidth(bin1) * hist.GetBinError(bin1)) / weight_sum}, "\
                        f"{(hist.GetBinWidth(bin2) * hist.GetBinError(bin2)) / weight_sum}\n"\
                        f"power: {((hist.GetBinWidth(bin1) * hist.GetBinError(bin1)) / weight_sum)**2.}, "\
                        f"{((hist.GetBinWidth(bin2) * hist.GetBinError(bin2)) / weight_sum)**2.}\n"\
                        f"sum: {((hist.GetBinWidth(bin1) * hist.GetBinError(bin1)) / weight_sum)**2. + ((hist.GetBinWidth(bin2) * hist.GetBinError(bin2)) / weight_sum)**2.}\n"\
                        f"sqrt: {math.sqrt(((hist.GetBinWidth(bin1) * hist.GetBinError(bin1)) / weight_sum)**2. + ((hist.GetBinWidth(bin2) * hist.GetBinError(bin2)) / weight_sum)**2.)}\n")
                hist2.SetBinError(binn, math.sqrt(((hist.GetBinWidth(bin1) * hist.GetBinError(bin1)) / weight_sum) ** 2. +\
                                                  ((hist.GetBinWidth(bin2) * hist.GetBinError(bin2)) / weight_sum) ** 2.))
                ind += 1
            print(f"New bin {binn} low edge {hist2.GetBinLowEdge(binn)} "\
                  f"up edge {hist2.GetXaxis().GetBinUpEdge(binn)} "\
                  f"content: {hist2.GetBinContent(binn)} +/- {hist2.GetBinError(binn)} ind {ind}")
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
