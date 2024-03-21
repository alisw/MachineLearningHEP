#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
##                                                                         ##
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

import os
import munch # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TCanvas, TF1, gStyle # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.analysis.analyzer import Analyzer

class AnalyzerJets(Analyzer): # pylint: disable=too-many-instance-attributes
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        self.cfg = munch.munchify(datap)
        self.cfg.ana = munch.munchify(datap).analysis[typean]

        # output directories
        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

        # input directories (processor output)
        self.d_resultsallpmc_proc = self.d_resultsallpmc
        self.d_resultsallpdata_proc = self.d_resultsallpdata

        # input files
        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata_proc, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc_proc, n_filemass_name)
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_fileeff = os.path.join(self.d_resultsallpmc_proc, self.n_fileeff)
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.n_fileresp = os.path.join(self.d_resultsallpmc_proc, self.n_fileresp)

        self.fitSigma = {'mc': 7 * [0], 'data': 7 * [0]}
        self.fitMean = {'mc': 7 * [0], 'data': 7 * [0]}
        self.fitBkgFunc = {'mc': [], 'data': []}


    def fit(self):
        gStyle.SetOptFit(1111)
        self.logger.info("Running fitter")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for ipt in range(7):
                    c = TCanvas("Candidate mass")
                    h_invmass = rfile.Get(f'hmass_{ipt}')
                    funcSignal = TF1("funcsignal", "gaus(0)", 1.67, 2.1)
                    funcBkg = TF1("funcBkg", "expo(0)", 1.67, 2.1)
                    funcTotal = TF1("funcTotal", "gaus(0)+expo(3)", 1.67, 2.1)
                    funcTotal.SetParameter(0, h_invmass.GetMaximum())
                    funcTotal.SetParameter(1, 1.86)
                    funcTotal.SetParLimits(2,0.0,0.1)
                    fitResult = h_invmass.Fit(funcTotal, "S", "", 1.67, 2.1)
                    self.fitSigma[mcordata][ipt] = fitResult.Parameter(2)
                    self.fitMean[mcordata][ipt] = fitResult.Parameter(1)
                    funcBkg.SetParameter(0, fitResult.Parameter(3))
                    funcBkg.SetParameter(1, fitResult.Parameter(4))
                    self.fitBkgFunc[mcordata].append(funcBkg)
                    h_invmass.Draw()
                    c.SaveAs(f'hmass_fitted_{ipt}_{mcordata}.png')
        self.sidebandsub()


    def sidebandsub(self):
        self.logger.info("Running sideband subtraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for ipt in range(7):
                    h2_invmass_zg = rfile.Get(f'h2jet_invmass_zg_{ipt}')
                    c = TCanvas("h2jet_invmass_zg")
                    h2_invmass_zg.Draw("colz")
                    c.SaveAs(f'h2jet_invmass_zg_{ipt}_{mcordata}.png')
                    signalA = self.fitMean[mcordata][ipt] - 2 * self.fitSigma[mcordata][ipt]
                    signalB = self.fitMean[mcordata][ipt] + 2 * self.fitSigma[mcordata][ipt]
                    sidebandLeftA = self.fitMean[mcordata][ipt] - 7 * self.fitSigma[mcordata][ipt]
                    sidebandLeftB = self.fitMean[mcordata][ipt] - 4 * self.fitSigma[mcordata][ipt]
                    sidebandRightA = self.fitMean[mcordata][ipt] + 4 * self.fitSigma[mcordata][ipt]
                    sidebandRightB = self.fitMean[mcordata][ipt] + 7 * self.fitSigma[mcordata][ipt]
                    print (h2_invmass_zg.GetXaxis().FindBin(signalA),h2_invmass_zg.GetXaxis().FindBin(signalB) )
                    fh_signal = h2_invmass_zg.ProjectionY(f'h2jet_zg_signal_{ipt}_{mcordata}', h2_invmass_zg.GetXaxis().FindBin(signalA), h2_invmass_zg.GetXaxis().FindBin(signalB),"e")
                    signalArea = self.fitBkgFunc[mcordata][ipt].Integral(signalA,signalB)
                    fh_sidebandleft = h2_invmass_zg.ProjectionY(f'h2jet_zg_sidebandleft_{ipt}_{mcordata}', h2_invmass_zg.GetXaxis().FindBin(sidebandLeftA), h2_invmass_zg.GetXaxis().FindBin(sidebandLeftB),"e")
                    sidebandLeftlArea = self.fitBkgFunc[mcordata][ipt].Integral(sidebandLeftA,sidebandLeftB)
                    fh_sidebandright = h2_invmass_zg.ProjectionY(f'h2jet_zg_sidebandright_{ipt}_{mcordata}', h2_invmass_zg.GetXaxis().FindBin(sidebandRightA), h2_invmass_zg.GetXaxis().FindBin(sidebandRightB),"e")
                    sidebandRightArea = self.fitBkgFunc[mcordata][ipt].Integral(sidebandRightA,sidebandRightB)
                    c = TCanvas("signal")
                    fh_signal.Draw()
                    c.SaveAs(f'hjet_zg_signal_{ipt}_{mcordata}.png')
                    c = TCanvas("sideband left")
                    fh_sidebandleft.Draw()
                    c.SaveAs(f'h2jet_zg_sidebandleft_{ipt}_{mcordata}.png')
                    c = TCanvas("sideband right")
                    fh_sidebandright.Draw()
                    c.SaveAs(f'h2jet_zg_sidebandright_{ipt}_{mcordata}.png')
                    areaNormFactor = signalArea / (sidebandLeftlArea + sidebandRightArea)
                    fh_sideband = fh_sidebandleft.Clone(f'h_sideband_{ipt}_{mcordata}')
                    fh_sideband.Add(fh_sidebandright, 1.0)
                    c = TCanvas("sideband")
                    fh_sideband.Draw()
                    c.SaveAs(f'hjet_zg_sideband_{ipt}_{mcordata}.png')
                    fh_subtracted = fh_signal.Clone(f'h_subtracted_{ipt}_{mcordata}')
                    fh_subtracted.Add(fh_sideband, -1.0 * areaNormFactor)
                    fh_subtracted.Scale(1.0 / 0.954)
                    c = TCanvas("subtracted")
                    fh_subtracted.Draw()
                    c.SaveAs(f'hjet_zg_subtracted_{ipt}_{mcordata}.png')

                    
                    





    def qa(self): # pylint: disable=too-many-branches, too-many-locals, invalid-name
        self.logger.info("Running D0 jet qa")

        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                histonorm = rfile.Get("histonorm")
                if not histonorm:
                    self.logger.critical('histonorm not found')
                p_nevents = histonorm.GetBinContent(1)
                self.logger.debug('Number of selected event: %d', p_nevents)

                for ipt in range(7):
                    c = TCanvas("Candidate mass")
                    h_invmass = rfile.Get(f'hmass_{ipt}')
                    if not h_invmass:
                        self.logger.critical('hmass not found')
                    h_invmass.Print()
                    h_invmass.Draw()
                    c.SaveAs(f'hmass_{ipt}_{mcordata}.png')

                    c = TCanvas("Candidate pt")
                    h_candpt = rfile.Get(f'hcandpt_{ipt}')
                    if not h_candpt:
                        self.logger.critical('hcandpt not found')
                    h_candpt.Print()
                    h_candpt.Draw()
                    c.SaveAs(f'hcandpt_{ipt}_{mcordata}.png')

                    c = TCanvas("Jet pt")
                    h_jetpt = rfile.Get(f'hjetpt_{ipt}')
                    if not h_jetpt:
                        self.logger.critical('hjetpt not found')
                    h_jetpt.Print()
                    h_jetpt.Draw()
                    c.SaveAs(f'hjetpt_{ipt}_{mcordata}.png')

                    c = TCanvas("Jet zg")
                    h_jetzg = rfile.Get(f'hjetzg_{ipt}')
                    if not h_jetzg:
                        self.logger.critical('hjetzg not found')
                    h_jetzg.Print()
                    h_jetzg.Draw()
                    c.SaveAs(f'hjetzg_{ipt}_{mcordata}.png')
