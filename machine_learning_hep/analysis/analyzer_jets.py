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
import ROOT # pylint: disable=import-error

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

    def _save_canvas(self, canvas, filename, mcordata): # pylint: disable=unused-argument
        # folder = self.d_resultsallpmc if mcordata == 'mc' else self.d_resultsallpdata
        canvas.SaveAs(f'fig/{self.case}/{self.typean}_{filename}')

    def _save_hist(self, hist, filename, mcordata):
        if not hist:
            self.logger.error('no histogram for <%s>', filename)
            # TODO: remove file if it exists?
            return
        c = TCanvas()
        hist.Draw()
        self._save_canvas(c, filename, mcordata)

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
                    h_invmass = rfile.Get(f'hmass_{ipt}')
                    fit_res, _, func_bkg = self._fit_mass(h_invmass)
                    if fit_res is not None:
                        self.fit_sigma[mcordata][ipt] = fit_res.Parameter(2)
                        self.fit_mean[mcordata][ipt] = fit_res.Parameter(1)
                        self.fit_func_bkg[mcordata].append(func_bkg)
                    self._save_hist(h_invmass, f'hmass_fitted_{ipt}_{mcordata}.png', mcordata)

    def _subtract_sideband(self, hist, var, mcordata, ipt):
        self._save_hist(hist, f'h2jet_invmass_{var}_{ipt}_{mcordata}.png', mcordata)

        mean = self.fit_mean[mcordata][ipt]
        sigma = self.fit_sigma[mcordata][ipt]
        region_signal = (mean - 2 * sigma, mean + 2 * sigma)
        region_sideband_left = (mean - 7 * sigma, mean - 4 * sigma)
        region_sideband_right = (mean + 4 * sigma, mean + 7 * sigma)

        axis = hist.GetXaxis()
        bins_signal = tuple(map(axis.FindBin, region_signal))
        bins_sideband_left = tuple(map(axis.FindBin, region_sideband_left))
        bins_sideband_right = tuple(map(axis.FindBin, region_sideband_right))

        fh_signal = hist.ProjectionY(f'h2jet_{var}_signal_{ipt}_{mcordata}', bins_signal[0], bins_signal[1], "e")
        signalArea = self.fit_func_bkg[mcordata][ipt].Integral(region_signal[0], region_signal[1])

        fh_sidebandleft = hist.ProjectionY(f'h2jet_{var}_sidebandleft_{ipt}_{mcordata}',
                                           bins_sideband_left[0], bins_sideband_left[1], "e")
        sidebandLeftlArea = self.fit_func_bkg[mcordata][ipt].Integral(region_sideband_left[0], region_sideband_left[1])

        fh_sidebandright = hist.ProjectionY(f'h2jet_{var}_sidebandright_{ipt}_{mcordata}',
                                            bins_sideband_right[0], bins_sideband_right[1], "e")
        sidebandRightArea = self.fit_func_bkg[mcordata][ipt].Integral(
            region_sideband_right[0], region_sideband_right[1])

        self._save_hist(fh_signal, f'hjet_{var}_signal_{ipt}_{mcordata}.png', mcordata)
        self._save_hist(fh_sidebandleft, f'hjet_{var}_sidebandleft_{ipt}_{mcordata}.png', mcordata)
        self._save_hist(fh_sidebandright, f'hjet_{var}_sidebandright_{ipt}_{mcordata}.png', mcordata)

        areaNormFactor = signalArea / (sidebandLeftlArea + sidebandRightArea)
        fh_sideband = fh_sidebandleft.Clone(f'h_sideband_{ipt}_{mcordata}')
        fh_sideband.Add(fh_sidebandright, 1.0)
        self._save_hist(fh_sideband, f'hjet_{var}_sideband_{ipt}_{mcordata}.png', mcordata)

        fh_subtracted = fh_signal.Clone(f'h_subtracted_{ipt}_{mcordata}')
        fh_subtracted.Sumw2()
        fh_subtracted.Add(fh_sideband, -1.0 * areaNormFactor)
        fh_subtracted.Scale(1.0 / 0.954)
        self._save_hist(fh_subtracted, f'hjet_{var}_subtracted_{ipt}_{mcordata}.png', mcordata)

        c = TCanvas()
        fh_signal.SetLineColor(ROOT.kRed) # pylint: disable=no-member
        fh_signal.Draw()
        fh_sideband.Scale(areaNormFactor)
        fh_sideband.SetLineColor(ROOT.kBlue) # pylint: disable=no-member
        fh_sideband.Draw("same")
        fh_subtracted.SetLineColor(ROOT.kOrange) # pylint: disable=no-member
        fh_subtracted.Draw("same")
        self._save_canvas(c, f'hjet_{var}_overview_{ipt}_{mcordata}.png', mcordata)
        print(self.hcandeff.GetBinContent(ipt + 1))
        fh_subtracted.Scale(1.0 / self.hcandeff.GetBinContent(ipt + 1))
        
        self._save_hist(fh_subtracted, f'hjet_{var}_subtracted_effscaled_{ipt}_{mcordata}.png', mcordata)
        return fh_subtracted

    def subtract_sidebands(self):
        self.logger.info("Running sideband subtraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                fh_subtracted_zg = []
                fh_subtracted_rg = []
                fh_subtracted_nsd = []
                fh_subtracted_dr = []
                fh_subtracted_zpar=[]
                for ipt in range(7):
                    fh_subtracted_zg.append(self._subtract_sideband(rfile.Get(f'h2jet_invmass_zg_{ipt}'), 'zg', mcordata, ipt))
                    fh_subtracted_rg.append(self._subtract_sideband(rfile.Get(f'h2jet_invmass_rg_{ipt}'), 'rg', mcordata, ipt))
                    fh_subtracted_nsd.append(self._subtract_sideband(rfile.Get(f'h2jet_invmass_nsd_{ipt}'), 'nsd', mcordata, ipt))
                    fh_subtracted_dr.append(self._subtract_sideband(rfile.Get(f'h2jet_invmass_dr_{ipt}'), 'dr', mcordata, ipt))
                    fh_subtracted_zpar.append(self._subtract_sideband(rfile.Get(f'h2jet_invmass_zpar_{ipt}'), 'zpar', mcordata, ipt))
                for i in range(1, 7):
                    fh_subtracted_zg[0].Add(fh_subtracted_zg[i])
                    fh_subtracted_rg[0].Add(fh_subtracted_zg[i])
                    fh_subtracted_nsd[0].Add(fh_subtracted_zg[i])
                    fh_subtracted_dr[0].Add(fh_subtracted_zg[i])
                    fh_subtracted_zpar[0].Add(fh_subtracted_zg[i])
                self._save_hist(fh_subtracted_zg[0], f'hjet_zg_subtracted_effscaled.png', mcordata)
                self._save_hist(fh_subtracted_rg[0], f'hjet_rg_subtracted_effscaled.png', mcordata)
                self._save_hist(fh_subtracted_nsd[0], f'hjet_nsd_subtracted_effscaled.png', mcordata)
                self._save_hist(fh_subtracted_dr[0], f'hjet_dr_subtracted_effscaled.png', mcordata)
                self._save_hist(fh_subtracted_zpar[0], f'hjet_zpar_subtracted_effscaled.png', mcordata)
                    

    def extract_signals(self):
        self.logger.info("Running signal extraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in ['zg', 'rg', 'nsd', 'zpar', 'dr']:
                    for ipt in range(7):
                        hmass2 = rfile.Get(f'h2jet_invmass_{var}_{ipt}')
                        nbins = hmass2.GetNbinsY()
                        hrange = (hmass2.GetYaxis().GetXmin(), hmass2.GetYaxis().GetXmax())
                        hist = TH1F(f'hjet{var}_{ipt}', "", nbins, hrange[0], hrange[1])
                        hist.Sumw2()
                        hist.GetXaxis().SetTitle(hmass2.GetYaxis().GetTitle())
                        # hist.SetBinContent(1, 0.0)
                        for i in range(nbins):
                            hmass = hmass2.ProjectionX(f'h_invmass_zg_{ipt}_proj_{i}', i+1, i+2, "e")
                            _, func_sig, _ = self._fit_mass(hmass)
                            self._save_hist(hmass, f'hmass_{var}_fitted_{ipt}_{i}_{mcordata}.png', mcordata)
                            hist.SetBinContent(i + 1, func_sig.Integral(1.67, 2.1)*(1.0/hmass.GetBinWidth(1)))
                        self._save_hist(hist, f'{var}_signalextracted_{ipt}_{mcordata}.png', mcordata)
                        hist.Scale(1.0/self.hcandeff.GetBinContent(ipt + 1))
                        self._save_hist(hist, f'{var}_signalextracted_eff_scaled_{ipt}_{mcordata}.png', mcordata)
                        if ipt == 0:
                            hist_effscaled = hist.Clone(f'hjet{var}')
                        else:
                            hist_effscaled.Add(hist)
                    self._save_hist(hist_effscaled, f'{var}_signalextracted_eff_scaled_{mcordata}.png', mcordata)
                            


    def efficiency(self):
        self.logger.info("Running efficiency")
        rfilename = self.n_fileeff
        with TFile(rfilename) as rfile:
            h_gen = rfile.Get('hjetgen')
            h_det = rfile.Get('hjetdet')
            heff = h_det.Clone('hjeteff')
            heff.Sumw2()
            heff.Divide(h_gen)
            h_match = rfile.Get('hjetmatch')
            heff_match = h_match.Clone('hjeteff_match')
            heff_match.Sumw2()
            heff_match.Divide(h_gen)
            self.hcandeff = heff_match.Clone("hcand_efficiency")
            self.hcandeff.SetDirectory(0)
            self._save_hist(h_gen, 'hjet_gen.png', 'mc')
            self._save_hist(h_det, 'hjet_det.png', 'mc')
            self._save_hist(heff, 'hjet_efficiency.png', 'mc')
            self._save_hist(heff_match, 'hjet_matched_efficiency.png', 'mc')

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
                    self._save_hist(rfile.Get(f'hmass_{ipt}'), f'hmass_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'hcandpt_{ipt}'), f'hcandpt_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'hjetpt_{ipt}'), f'hjetpt_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'hjetzg_{ipt}'), f'hjetzg_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'hjetrg_{ipt}'), f'hjetrg_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'hjetnsd_{ipt}'), f'hjetnsd_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'hjetzpar_{ipt}'), f'hjetzpar_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'hjetdr_{ipt}'), f'hjetdr_{ipt}_{mcordata}.png', mcordata)
