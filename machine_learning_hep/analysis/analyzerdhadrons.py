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
# pylint: disable=too-many-lines
import os
# pylint: disable=unused-wildcard-import, wildcard-import
# pylint: disable=import-error, no-name-in-module, unused-import
from root_numpy import hist2array, array2hist
from ROOT import TFile, TH1F, TH2F, TCanvas, TPad, TF1, TH1D
from ROOT import AliHFInvMassFitter, AliVertexingHFUtils, AliHFInvMassMultiTrialFit
from ROOT import gStyle, TLegend, TLine, TText, TPaveText, TArrow
from ROOT import gROOT, TDirectory, TPaveLabel
from ROOT import TStyle, kBlue, kGreen, kBlack, kRed, kOrange
from ROOT import TLatex
from ROOT import gInterpreter, gPad
# HF specific imports
from machine_learning_hep.fitting.helpers import MLFitter
from machine_learning_hep.io import dump_yaml_from_dict
from machine_learning_hep.utilities import folding, get_bins, make_latex_table, parallelizer
from machine_learning_hep.utilities_plot import plot_histograms
from machine_learning_hep.analysis.analyzer import Analyzer

# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements, fixme
class AnalyzerDhadrons(Analyzer):
    species = "analyzer"
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)


        # Differential binning
        self.v_var_binning = datap["var_binning"]


        # Directories
        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc, n_filemass_name)

        self.n_fileff = datap["files_names"]["efffilename"]
        self.n_fileff = os.path.join(self.d_resultsallpmc, self.n_fileff)

        self.yields_filename = "yields"
        self.fits_dirname = "fits"

        # Fitting
        self.fitter = None


        # Systematics
        self.p_indexhpt = datap["analysis"]["indexhptspectrum"]
        self.p_fd_method = datap["analysis"]["fd_method"]
        self.p_cctype = datap["analysis"]["cctype"]
        self.p_sigmav0 = datap["analysis"]["sigmav0"]
        self.p_inputfonllpred = datap["analysis"]["inputfonllpred"]

        # Names, labels
        self.p_latexnhadron = datap["analysis"][self.typean]["latexnamehadron"]

    def fit(self):
        # Enable ROOT batch mode and reset in the end
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        self.fitter = MLFitter(self.processer_helper, self.n_filemass, self.n_filemass_mc)
        self.fitter.perform_pre_fits()
        self.fitter.perform_central_fits()
        fileout_name = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                           None, [self.case, self.typean])
        fileout = TFile(fileout_name, "RECREATE")
        self.fitter.draw_fits(self.d_resultsallpdata, fileout)
        fileout.Close()
        fileout_name = os.path.join(self.d_resultsallpdata,
                                    f"{self.fits_dirname}_{self.case}_{self.typean}")
        self.fitter.save_fits(fileout_name)
        # Reset to former mode
        gROOT.SetBatch(tmp_is_root_batch)

    def efficiency(self):
        self.loadstyle()
        tmp_is_root_batch = gROOT.IsBatch()
        gROOT.SetBatch(True)

        lfileeff = TFile.Open(self.n_fileff)
        fileouteff = TFile.Open("%s/efficiencies%s%s.root" % (self.d_resultsallpmc, \
                                 self.case, self.typean), "recreate")
        cEff = TCanvas('cEff', 'The Fit Canvas')
        cEff.SetCanvasSize(1900, 1500)
        cEff.SetWindowSize(500, 500)

        legeff = TLegend(.5, .65, .7, .85)
        legeff.SetBorderSize(0)
        legeff.SetFillColor(0)
        legeff.SetFillStyle(0)
        legeff.SetTextFont(42)
        legeff.SetTextSize(0.035)

        h_gen_pr = lfileeff.Get("h_gen_pr")
        h_sel_pr = lfileeff.Get("h_sel_pr")
        h_sel_pr.Divide(h_sel_pr, h_gen_pr, 1.0, 1.0, "B")
        h_sel_pr.SetMinimum(0.)
        h_sel_pr.SetMaximum(1.5)
        fileouteff.cd()
        h_sel_pr.SetName("eff")
        h_sel_pr.Write()
        h_sel_pr.Draw("same")
        legeff.AddEntry(h_sel_pr, "prompt efficiency", "LEP")
        h_sel_pr.GetXaxis().SetTitle("#it{p}_{T} (GeV/#it{c})")
        h_sel_pr.GetYaxis().SetTitle("Acc x efficiency (prompt) %s %s (1/GeV)" \
                % (self.p_latexnhadron, self.typean))

        h_gen_fd = lfileeff.Get("h_gen_fd")
        h_sel_fd = lfileeff.Get("h_sel_fd")
        h_sel_fd.Divide(h_sel_fd, h_gen_fd, 1.0, 1.0, "B")
        fileouteff.cd()
        h_sel_fd.SetMinimum(0.)
        h_sel_fd.SetMaximum(1.5)
        h_sel_fd.SetName("eff_fd")
        h_sel_fd.Write()
        legeff.AddEntry(h_sel_pr, "feeddown efficiency", "LEP")
        h_sel_pr.Draw("same")
        legeff.Draw()
        cEff.SaveAs("%s/Eff%s%s.eps" % (self.d_resultsallpmc,
                                        self.case, self.typean))
        print("Efficiency finished")
        fileouteff.Close()
        gROOT.SetBatch(tmp_is_root_batch)

    # pylint: disable=import-outside-toplevel
    def makenormyields(self):
        gROOT.SetBatch(True)
        self.loadstyle()
        print("making yields")
        fileouteff = "%s/efficiencies%s%s.root" % \
                      (self.d_resultsallpmc, self.case, self.typean)
        yield_filename = self.make_file_path(self.d_resultsallpdata, self.yields_filename, "root",
                                             None, [self.case, self.typean])
        gROOT.LoadMacro("HFPtSpectrum.C")
        from ROOT import HFPtSpectrum, HFPtSpectrum2
        namehistoeffprompt = "eff"
        namehistoefffeed = "eff_fd"
        nameyield = "hyields"
        fileoutcross = "%s/finalcross%s%s.root" % \
                   (self.d_resultsallpdata, self.case, self.typean)
        norm = -1
        lfile = TFile.Open(self.n_filemass)
        hNorm = lfile.Get("hEvForNorm")
        normfromhisto = hNorm.GetBinContent(1)

        HFPtSpectrum(self.p_indexhpt, self.p_inputfonllpred, \
        fileouteff, namehistoeffprompt, namehistoefffeed, yield_filename, nameyield, \
        fileoutcross, norm, self.p_sigmav0 * 1e12, self.p_fd_method, self.p_cctype)

        cCross = TCanvas('cCross', 'The Fit Canvas')
        cCross.SetCanvasSize(1900, 1500)
        cCross.SetWindowSize(500, 500)
        cCross.SetLogy()

        legcross = TLegend(.5, .65, .7, .85)
        legcross.SetBorderSize(0)
        legcross.SetFillColor(0)
        legcross.SetFillStyle(0)
        legcross.SetTextFont(42)
        legcross.SetTextSize(0.035)

        myfile = TFile.Open(fileoutcross, "read")
        hcross = myfile.Get("histoSigmaCorr")
        hcross.GetXaxis().SetTitle("#it{p}_{T} %s (GeV/#it{c})" % self.p_latexnhadron)
        hcross.GetYaxis().SetTitle("d#sigma/d#it{p}_{T} (%s) %s" %
                                   (self.p_latexnhadron, self.typean))
        legcross.AddEntry(hcross, "cross section", "LEP")
        cCross.SaveAs("%s/Cross%s%sVs%s.eps" % (self.d_resultsallpdata,
                                                self.case, self.typean, self.v_var_binning))
