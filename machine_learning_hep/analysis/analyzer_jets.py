#############################################################################
##  © Copyright CERN 2024. All rights not expressly granted are reserved.  ##
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
from ROOT import TFile, TCanvas, TF1, TH1F, gStyle # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.analysis.analyzer import Analyzer

class AnalyzerJets(Analyzer): # pylint: disable=too-many-instance-attributes
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        self.config = munch.munchify(datap)
        self.config.ana = munch.munchify(datap).analysis[typean]

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

        self.fit_sigma = {'mc': 7 * [0], 'data': 7 * [0]}
        self.fit_mean = {'mc': 7 * [0], 'data': 7 * [0]}
        self.fit_func_bkg = {'mc': [], 'data': []}

    def _fit_mass(self, hist):
        fit_range = self.cfg('mass_fit.range')
        func_sig = TF1('funcSig', self.cfg('mass_fit.func_sig'))
        func_bkg = TF1('funcBkg', self.cfg('mass_fit.func_bkg'))
        func_tot = TF1('funcTot', f"{self.cfg('mass_fit.func_sig')} + {self.cfg('mass_fit.func_bkg')}")
        func_tot.SetParameter(0, hist.GetMaximum())
        for par, value in self.cfg('mass_fit.par_start', {}).items():
            self.logger.info('Setting par %i to %g', par, value)
            func_tot.SetParameter(par, value)
        for par, value in self.cfg('mass_fit.par_constrain', {}).items():
            self.logger.info('Constraining par %i to (%g, %g)', par, value[0], value[1])
            func_tot.SetParLimits(par, value[0], value[1])
        for par, value in self.cfg('mass_fit.par_fix', {}).items():
            self.logger.info('Fixing par %i to %g', par, value)
            func_tot.FixParameter(par, value)
        fit_res = hist.Fit(func_tot, "S", "", fit_range[0], fit_range[1])
        func_sig.SetParameters(func_tot.GetParameters())
        func_bkg.SetParameters(func_tot.GetParameters())
        return (fit_res, func_sig, func_bkg)

    def fit(self):
        self.logger.info("Running fitter")
        gStyle.SetOptFit(1111)
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for ipt in range(7):
                    c = TCanvas("Candidate mass")
                    h_invmass = rfile.Get(f'hmass_{ipt}')
                    fit_res, _, func_bkg = self._fit_mass(h_invmass)
                    self.fit_sigma[mcordata][ipt] = fit_res.Parameter(2)
                    self.fit_mean[mcordata][ipt] = fit_res.Parameter(1)
                    self.fit_func_bkg[mcordata].append(func_bkg)
                    h_invmass.Draw()
                    c.SaveAs(f'hmass_fitted_{ipt}_{mcordata}.png')

    def subtract_sidebands(self):
        self.logger.info("Running sideband subtraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for ipt in range(7):
                    h2_invmass_zg = rfile.Get(f'h2jet_invmass_zg_{ipt}')
                    c = TCanvas("h2jet_invmass_zg")
                    h2_invmass_zg.Draw("colz")
                    c.SaveAs(f'h2jet_invmass_zg_{ipt}_{mcordata}.png')
                    signalA = self.fit_mean[mcordata][ipt] - 2 * self.fit_sigma[mcordata][ipt]
                    signalB = self.fit_mean[mcordata][ipt] + 2 * self.fit_sigma[mcordata][ipt]
                    sidebandLeftA = self.fit_mean[mcordata][ipt] - 7 * self.fit_sigma[mcordata][ipt]
                    sidebandLeftB = self.fit_mean[mcordata][ipt] - 4 * self.fit_sigma[mcordata][ipt]
                    sidebandRightA = self.fit_mean[mcordata][ipt] + 4 * self.fit_sigma[mcordata][ipt]
                    sidebandRightB = self.fit_mean[mcordata][ipt] + 7 * self.fit_sigma[mcordata][ipt]
                    fh_signal = h2_invmass_zg.ProjectionY(f'h2jet_zg_signal_{ipt}_{mcordata}',
                                                          h2_invmass_zg.GetXaxis().FindBin(signalA),
                                                          h2_invmass_zg.GetXaxis().FindBin(signalB),
                                                          "e")
                    signalArea = self.fit_func_bkg[mcordata][ipt].Integral(signalA,signalB)
                    fh_sidebandleft = h2_invmass_zg.ProjectionY(f'h2jet_zg_sidebandleft_{ipt}_{mcordata}',
                                                                h2_invmass_zg.GetXaxis().FindBin(sidebandLeftA),
                                                                h2_invmass_zg.GetXaxis().FindBin(sidebandLeftB),
                                                                "e")
                    sidebandLeftlArea = self.fit_func_bkg[mcordata][ipt].Integral(sidebandLeftA,sidebandLeftB)
                    fh_sidebandright = h2_invmass_zg.ProjectionY(f'h2jet_zg_sidebandright_{ipt}_{mcordata}',
                                                                 h2_invmass_zg.GetXaxis().FindBin(sidebandRightA),
                                                                 h2_invmass_zg.GetXaxis().FindBin(sidebandRightB),
                                                                 "e")
                    sidebandRightArea = self.fit_func_bkg[mcordata][ipt].Integral(sidebandRightA,sidebandRightB)
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

    def extract_signal(self):
        self.logger.info("Running signal extraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for ipt in range(7):
                    h_zg = TH1F(
                    f'hjetzg_{ipt}', "", 10, 0.0, 1.0)
                    h_zg.SetBinContent(1, 0.0)
                    for i in range(1,5):
                        h_invmass = rfile.Get(f'hmass_zg_{ipt}_{i}')
                        c = TCanvas("Candidate mass")
                        _, func_sig, _ = self._fit_mass(h_invmass)
                        h_invmass.Draw()
                        c.SaveAs(f'hmass_zg_fitted_{ipt}_{i}_{mcordata}.png')
                        h_zg.SetBinContent(i + 1, func_sig.Integral(1.67, 2.1)*(1.0/h_invmass.GetBinWidth(1)))
                    h_zg.Draw()
                    c.SaveAs(f'zg_signalextracted_{ipt}_{mcordata}.png')

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
