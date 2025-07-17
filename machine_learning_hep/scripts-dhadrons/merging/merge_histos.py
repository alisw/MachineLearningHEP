"""
Merge histograms from different ROOT files. One file per pt bin.
A single histogram contains all pt bins.
"""

import argparse

from ROOT import TFile, gROOT # pylint: disable=import-error

def main():
    """
    Main
    """
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--histname", action="append", type=str,
                        help="name of histograms to merge")
    parser.add_argument("-o", "--outfile", action="append", type=str, help="Output file")
    parser.add_argument("-i", "--infile", action="append", type=str, help="Input file")
    args = parser.parse_args()

    if len(args.outfile) != 1:
        raise ValueError("Provide exactly 1 output file")

    print(f"filenames {args.infile}")

    with TFile(args.outfile[0], "RECREATE") as fout:
        fins = [TFile(filename) for filename in args.infile]

        histname = args.histname
        if args.histname is None:
            histname = [key.GetName() for key in fins[0].GetListOfKeys()]

        print(f"histnames {histname}")

        def get_hist(fin, histname):
            fin.cd()
            return fin.Get(histname)

        for name in histname:
            hist_list = [get_hist(fin, name) for fin in fins]
            print(f"{name} hist list length: {len(hist_list)}")
            if any(cls in hist_list[0].ClassName() for cls in ("TH1", "TGraph")):
                hist = hist_list[-1].Clone()
                for ind, hist_tmp in enumerate(hist_list):
                    print(f"hist {name} bin {ind+1} pt [{hist.GetBinLowEdge(ind + 1)}, {hist.GetBinLowEdge(ind + 2)}) " \
                          f"content {hist_tmp.GetBinContent(ind + 1)}")
                    hist.SetBinContent(ind+1, hist_tmp.GetBinContent(ind+1))
                    hist.SetBinError(ind+1, hist_tmp.GetBinError(ind+1))
                fout.cd()
                hist.Write()

if __name__ == "__main__":
    main()
