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
from pathlib import Path

import numpy as np
import pandas as pd
from ROOT import TFile, TCanvas, TF1, TH1F, TH2F, TH3F, gStyle # pylint: disable=import-error, no-name-in-module
import ROOT # pylint: disable=import-error

from machine_learning_hep.utilities import folding
from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.utilities_hist import create_hist, fill_hist, scale_bin

class AnalyzerJets(Analyzer): # pylint: disable=too-many-instance-attributes
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

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

        self.observables = {
            'qa': ['zg', 'rg', 'nsd', 'zpar', 'dr', 'lntheta', 'lnkt'],
            'sideband': ['zg', 'rg', 'nsd', 'zpar', 'dr', 'lntheta-lnkt'],
            'signal': ['zg', 'rg', 'nsd', 'zpar', 'dr'],
            'fd': ['zg']
        }

        self.bins_candpt = np.asarray(self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:], 'd')
        self.nbins = len(self.bins_candpt) - 1

        self.fit_sigma = {}
        self.fit_mean = {}
        self.fit_func_bkg = {}
        self.hcandeff = None
        self.n_events = {}

        self.path_fig = Path(f'fig/{self.case}/{self.typean}')
        for folder in ['qa', 'fit', 'sideband', 'signalextr', 'fd']:
            (self.path_fig / folder).mkdir(parents=True, exist_ok=True)

    #region helpers
    def _save_canvas(self, canvas, filename):
        # folder = self.d_resultsallpmc if mcordata == 'mc' else self.d_resultsallpdata
        canvas.SaveAs(f'fig/{self.case}/{self.typean}/{filename}')

    def _save_hist(self, hist, filename, option = ''):
        if not hist:
            self.logger.error('no histogram for <%s>', filename)
            # TODO: remove file if it exists?
            return
        c = TCanvas()
        hist.Draw(option)
        self._save_canvas(c, filename)

    def _sum_histos(self, histos, name = None):
        """
        Return histogram with sum of all histograms from iterable
        """
        hist = None
        for h in histos:
            if h is None:
                continue
            if hist is None:
                hist = h.Clone(name or (h.GetName() + '_cloned'))
            else:
                hist.Add(h)
        return hist

    def init(self):
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                histonorm = rfile.Get("histonorm")
                if not histonorm:
                    self.logger.critical('histonorm not found')
                self.n_events[mcordata] = histonorm.GetBinContent(1)
                self.logger.debug('Number of selected events for %s: %d', mcordata, self.n_events[mcordata])

    #region basic qa
    def qa(self): # TODO: Put in init so we always have number of events
        self.logger.info("Running D0 jet qa")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for ipt in range(self.nbins):
                    self._save_hist(rfile.Get(f'h_mass_{ipt}'), f'qa/h_mass_{ipt}_{mcordata}.png')
                    self._save_hist(rfile.Get(f'h_ptcand_{ipt}'), f'qa/h_ptcand_{ipt}_{mcordata}.png')
                    self._save_hist(rfile.Get(f'h_ptjet_{ipt}'), f'qa/h_ptjet_{ipt}_{mcordata}.png')
                    for var in self.observables['qa']:
                        self._save_hist(rfile.Get(f'h_{var}_{ipt}'), f'qa/h_{var}_{ipt}_{mcordata}.png')

    #region efficiency
    def efficiency(self):
        self.logger.info("Running efficiency")
        cats = {'prompt', 'nonprompt'}
        rfilename = self.n_fileeff
        with TFile(rfilename) as rfile:
            h_gen = {cat: rfile.Get(f'h_pthf_{cat}_gen') for cat in cats}
            h_det = {cat: rfile.Get(f'h_pthf_{cat}_det').Clone(f'h_eff_{cat}') for cat in cats}
            h_match = {cat: rfile.Get(f'h_pthf_{cat}_match').Clone(f'h_eff_{cat}_matched') for cat in cats}

            for cat in cats:
                self._save_hist(h_gen[cat], f'qa/h_pthf_{cat}_gen.png')
                self._save_hist(h_det[cat], f'qa/h_pthf_{cat}_det.png')
                h_det[cat].Sumw2()
                h_det[cat].Divide(h_gen[cat])
                self._save_hist(h_det[cat], f'h_eff_{cat}.png')

                h_match[cat].Sumw2()
                h_match[cat].Divide(h_gen[cat])
                self._save_hist(h_match[cat], f'h_eff_{cat}_matched.png')

            self.hcandeff = h_match['prompt'] if self.cfg('efficiency.store_matched') else h_det['prompt']
            self.hcandeff.SetDirectory(0)
            self.hcandeff_np = h_match['nonprompt'] if self.cfg('efficiency.store_matched') else h_det['nonprompt']
            self.hcandeff_np.SetDirectory(0)

    #region fitting
    def _fit_mass(self, hist, filename = None):
        if hist.GetEntries() == 0:
            raise UserWarning('Cannot fit histogram with no entries')
        fit_range = self.cfg('mass_fit.range')
        func_sig = TF1('funcSig', self.cfg('mass_fit.func_sig'), *fit_range)
        func_bkg = TF1('funcBkg', self.cfg('mass_fit.func_bkg'), *fit_range)
        func_tot = TF1('funcTot', f"{self.cfg('mass_fit.func_sig')} + {self.cfg('mass_fit.func_bkg')}")
        func_tot.SetParameter(0, hist.GetMaximum())
        for par, value in self.cfg('mass_fit.par_start', {}).items():
            self.logger.debug('Setting par %i to %g', par, value)
            func_tot.SetParameter(par, value)
        for par, value in self.cfg('mass_fit.par_constrain', {}).items():
            self.logger.debug('Constraining par %i to (%g, %g)', par, value[0], value[1])
            func_tot.SetParLimits(par, value[0], value[1])
        for par, value in self.cfg('mass_fit.par_fix', {}).items():
            self.logger.debug('Fixing par %i to %g', par, value)
            func_tot.FixParameter(par, value)
        fit_res = hist.Fit(func_tot, "SQL", "", fit_range[0], fit_range[1])
        if fit_res and fit_res.Get() and fit_res.IsValid():
            # TODO: how to avoid error messages for excess parameters?
            func_sig.SetParameters(func_tot.GetParameters())
            func_bkg.SetParameters(func_tot.GetParameters())
        else:
            self.logger.warning('Invalid fit result for %s', hist.GetName())
            func_tot.Print('v')

        if filename:
            c = TCanvas()
            hist.Draw()
            func_sig.SetLineColor(ROOT.kBlue)
            func_sig.Draw('lsame')
            func_bkg.SetLineColor(ROOT.kCyan)
            func_bkg.Draw('lsame')
            self._save_canvas(c, filename)

        return (fit_res, func_sig, func_bkg)

    def fit(self):
        self.logger.info("Running fitter")
        gStyle.SetOptFit(1111)
        for mcordata in ['mc', 'data']:
            self.fit_mean[mcordata] = [None] * self.nbins
            self.fit_sigma[mcordata] = [None] * self.nbins
            self.fit_func_bkg[mcordata] = [None] * self.nbins
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for ipt in range(self.nbins):
                    if h_invmass := rfile.Get(f'h_mass_{ipt}'):
                        if h_invmass.GetEntries() == 0:
                            continue
                        fit_res, _, func_bkg = self._fit_mass(
                            h_invmass, f'fit/h_mass_fitted_{ipt}_{mcordata}.png')
                        if fit_res and fit_res.Get() and fit_res.IsValid():
                            self.fit_sigma[mcordata][ipt] = fit_res.Parameter(2)
                            self.fit_mean[mcordata][ipt] = fit_res.Parameter(1)
                            self.fit_func_bkg[mcordata][ipt] = func_bkg

    #region sidebands
    def _subtract_sideband(self, hist, var, mcordata, ipt):
        """
        Subtract sideband distributions, assuming mass on first axis
        """
        if not hist:
            self.logger.warning('no histogram for %s bin %d', var, ipt)
            return None
        assert hist.GetDimension() in [2, 3], 'only 2- and 3-dimensional histograms are supported'
        self._save_hist(hist, f'sideband/h_mass-{var}_{ipt}_{mcordata}.png')

        mean = self.fit_mean[mcordata][ipt]
        sigma = self.fit_sigma[mcordata][ipt]
        if mean is None or sigma is None:
            self.logger.warning('no fit parameters for bin %d', ipt)
            return None

        regions = {
            'signal': (mean - 2 * sigma, mean + 2 * sigma),
            'sideband_left': (mean - 7 * sigma, mean - 4 * sigma),
            'sideband_right': (mean + 4 * sigma, mean + 7 * sigma)
        }

        axis = hist.GetXaxis()
        bins = {key: tuple(map(axis.FindBin, region)) for key, region in regions.items()}
        limits = {key: (axis.GetBinLowEdge(bins[key][0]), axis.GetBinUpEdge(bins[key][1]))
                  for key in regions}
        self.logger.info('actual sideband regions %s', limits)

        fh = {}
        area = {}
        for region in regions:
            if hist.GetDimension() == 2:
                fh[region] = hist.ProjectionY(f'h_{var}_{region}_{ipt}_{mcordata}', *bins[region], "e")
            elif hist.GetDimension() == 3:
                hist.GetXaxis().SetRange(*bins[region])
                fh[region] = hist.Project3D('zye').Clone(f'h_{var}_{region}_{ipt}_{mcordata}')
                hist.GetXaxis().SetRange(0, hist.GetXaxis().GetNbins() + 1)
            area[region] = self.fit_func_bkg[mcordata][ipt].Integral(*limits[region])
            self._save_hist(fh[region], f'sideband/h_{var}_{region}_{ipt}_{mcordata}.png')

        areaNormFactor = area['signal'] / (area['sideband_left'] + area['sideband_right'])

        fh_sideband = self._sum_histos(
            [fh['sideband_left'], fh['sideband_right']], f'h_{var}_sideband_{ipt}_{mcordata}')
        self._save_hist(fh_sideband, f'sideband/h_{var}_sideband_{ipt}_{mcordata}.png')

        fh_subtracted = fh['signal'].Clone(f'h_{var}_subtracted_{ipt}_{mcordata}')
        fh_subtracted.Sumw2()
        fh_subtracted.Add(fh_sideband, -areaNormFactor)
        fh_subtracted.Scale(1.0 / 0.954) # TODO: calculate from region
        self._save_hist(fh_subtracted, f'sideband/h_{var}_subtracted_{ipt}_{mcordata}.png')

        if hist.GetDimension() == 2:
            c = TCanvas()
            fh['signal'].SetLineColor(ROOT.kRed) # pylint: disable=no-member
            fh['signal'].Draw()
            fh_sideband.Scale(areaNormFactor)
            fh_sideband.SetLineColor(ROOT.kCyan) # pylint: disable=no-member
            fh_sideband.Draw("same")
            fh_subtracted.Draw("same")
            fh_subtracted.GetYaxis().SetRangeUser(
                0., max(fh_subtracted.GetMaximum(), fh['signal'].GetMaximum(), fh_sideband.GetMaximum()))
            self._save_canvas(c, f'sideband/h_{var}_overview_{ipt}_{mcordata}.png')

        if self.hcandeff:
            fh_subtracted.Scale(1.0 / self.hcandeff.GetBinContent(ipt + 1))
            self._save_hist(fh_subtracted, f'sideband/h_{var}_subtracted_effscaled_{ipt}_{mcordata}.png')
        else:
            self.logger.error('no efficiency correction because of missing efficiency')

        return fh_subtracted

    def subtract_sidebands(self):
        self.logger.info("Running sideband subtraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in self.observables['sideband']:
                    fh_sub = [self._subtract_sideband(rfile.Get(f'h_mass-{var}_{ipt}'), var, mcordata, ipt)
                              for ipt in range(self.nbins)]
                    fh_sum = self._sum_histos(fh_sub)
                    self._save_hist(fh_sum, f'h_{var}_subtracted_effscaled_{mcordata}.png')

    #region signal extraction
    def _extract_signal(self, hmass2, var, mcordata, ipt): # pylint: disable=too-many-branches
        """
        Extract signal through inv. mass fit in bins of observable
        """
        if not hmass2:
            self.logger.warning('no histogram for %s bin %d', var, ipt)
            return None
        self._save_hist(hmass2, f'signalextr/h_mass-{var}_{ipt}_{mcordata}.png')
        nbins = hmass2.GetNbinsY()
        hrange = (hmass2.GetYaxis().GetXmin(), hmass2.GetYaxis().GetXmax())
        if hmass2.GetDimension() == 3:
            nbins2 = hmass2.GetNbinsZ()
            hrange2 = (hmass2.GetZaxis().GetXmin(), hmass2.GetZaxis().GetXmax())
            hist = TH2F(f'h_{var}_{ipt}', "", nbins, *hrange, nbins2, *hrange2)
            hist.GetYaxis().SetTitle(hmass2.GetZaxis().GetTitle())
        else:
            hist = TH1F(f'h_{var}_{ipt}', "", nbins, *hrange)
        hist.GetXaxis().SetTitle(hmass2.GetYaxis().GetTitle())

        if self.fit_mean[mcordata][ipt] is None or self.fit_sigma[mcordata][ipt] is None:
            self.logger.warning('no fit parameters for %s bin %d', var, ipt)
            return None # TODO: should we continue nonetheless?

        range_int = (self.fit_mean[mcordata][ipt] - 3 * self.fit_sigma[mcordata][ipt],
                     self.fit_mean[mcordata][ipt] + 3 * self.fit_sigma[mcordata][ipt])
        for i in range(nbins):
            if hmass2.GetDimension() == 3:
                hmass2.GetYaxis().SetRange(i+1, i+1)
                for j in range(nbins2):
                    hmass2.GetZaxis().SetRange(j+1, j+1)
                    hmass = hmass2.Project3D('x')
                    if hmass.GetEntries() > 10:
                        fit_res, func_sig, _ = self._fit_mass(
                            hmass, f'signalextr/h_mass-{var}_fitted_{ipt}_{i}_{j}_{mcordata}.png')
                        if fit_res and fit_res.Get() and fit_res.IsValid():
                            hist.SetBinContent(i + 1, j +  1, func_sig.Integral(*range_int) / hmass.GetBinWidth(1))
            else:
                hmass = hmass2.ProjectionX(f'h_mass-{var}_{ipt}_proj_{i}', i+1, i+1, "e")
                if hmass.GetEntries() > 10:
                    fit_res, func_sig, _ = self._fit_mass(
                        hmass, f'signalextr/h_mass-{var}_fitted_{ipt}_{i}_{mcordata}.png')
                    if fit_res and fit_res.Get() and fit_res.IsValid():
                        hist.SetBinContent(i + 1, func_sig.Integral(*range_int) / hmass.GetBinWidth(1))
        if hmass2.GetDimension() == 3:
            hmass2.GetYaxis().SetRange(0, hmass2.GetYaxis().GetNbins() + 1)
            hmass2.GetZaxis().SetRange(0, hmass2.GetYaxis().GetNbins() + 1)
        self._save_hist(hist, f'signalextr/h_{var}_signalextracted_{ipt}_{mcordata}.png')

        hist.Sumw2()
        if self.hcandeff:
            hist.Scale(1.0/self.hcandeff.GetBinContent(ipt + 1))
            self._save_hist(hist, f'signalextr/h_{var}_signalextracted_effscaled_{ipt}_{mcordata}.png')
        else:
            self.logger.error('no efficiency correction because of missing efficiency')

        return hist

    def extract_signals(self):
        self.logger.info("Running signal extraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in self.observables['signal']:
                    hsignals = [self._extract_signal(rfile.Get(f'h_mass-{var}_{ipt}'), var, mcordata, ipt)
                                for ipt in range(self.nbins)]
                    hist_effscaled = self._sum_histos(hsignals)
                    self._save_hist(hist_effscaled, f'h_{var}_signalextracted_effscaled_{mcordata}.png')

    #region feeddown
    def correct_feeddown(self):
        self.logger.info('Running feeddown correction')

        with TFile('/data2/vkucera/powheg/trees_powheg_fd_F05_R05.root') as rfile:
            powheg_xsection = rfile.Get('fHistXsection')
            powheg_xsection_scale_factor = powheg_xsection.GetBinContent(1) / powheg_xsection.GetEntries()

        for var in self.observables['fd']:
            jetptbins_array = np.linspace(5, 55, 10+1, dtype='d') #Todo take form database
            shapebins_array = np.linspace(0., 1., 10+1, dtype='d')

            df = pd.read_parquet('/data2/jklein/powheg/trees_powheg_fd_F05_R05.parquet')

            h3feeddown_gen = create_hist(f'h3_feeddown_gen',
                                         f';p_{{T}}^{{cand}} (GeV/#it{{c}});p_{{T}}^{{jet}} (GeV/#it{{c}});{var}',
                                         self.bins_candpt, jetptbins_array, shapebins_array)
            fill_hist(h3feeddown_gen, df[['pt_cand', 'pt_jet', 'zg_jet']])
            h3feeddown_gen.GetXaxis().SetRange(1,10)

            hfeeddown_gen_noeff = h3feeddown_gen.Project3D("zy")
            self._save_hist(hfeeddown_gen_noeff, f'fd/h_ptjet-{var}_feeddown_gen_noeffscaling.png')

            for ipt, _ in enumerate(self.bins_candpt):
                eff_pr = self.hcandeff.GetBinContent(ipt+1)
                eff_np = self.hcandeff_np.GetBinContent(ipt+1)

                for ijetpt, _ in enumerate(jetptbins_array): #TODO add jet pT binning to database
                    for ishape, _ in enumerate(shapebins_array):
                        # TODO: Improve error propagation
                        scale_bin(h3feeddown_gen, eff_np/eff_pr, ipt+1, ijetpt+1, ishape+1)

            hfeeddown_gen = h3feeddown_gen.Project3D("zy")
            self._save_hist(hfeeddown_gen, f'fd/h_ptjet-{var}_feeddown_gen_effscaled.png')

            with TFile(self.n_fileeff) as rfile:
                hkinematiceff_np_gennodetcuts_zg = rfile.Get(f'hkinematiceff_np_gennodetcuts_zg')
                hkinematiceff_np_gendetcuts_zg = rfile.Get(f'hkinematiceff_np_gendetcuts_zg')
                hkinematiceff_np_gendetcuts_zg.Divide(hkinematiceff_np_gennodetcuts_zg)
                self._save_hist(hkinematiceff_np_gendetcuts_zg, f'fd/h_effkine-ptjet-{var}_np_gen.png', 'text')

                # ROOT bug?
                ha = hfeeddown_gen
                hb = hkinematiceff_np_gendetcuts_zg
                print(ha.GetXaxis().GetXbins().fN, ha.GetYaxis().GetXbins().fN,
                    hb.GetXaxis().GetXbins().fN, hb.GetYaxis().GetXbins().fN)

                hfeeddown_gen.Multiply(hkinematiceff_np_gendetcuts_zg)
                self._save_hist(hfeeddown_gen, f'fd/h_ptjet-{var}_feeddown_gen_kineeffscaled.png')

                response_matrix_np = rfile.Get(f'hkinematiceff_np_detnogencuts_zg_hkinematiceff_np_gennodetcuts_zg') # TODO: fix name
                self._save_hist(response_matrix_np, f'fd/h_ptjet-{var}_response_np.png')

                hfeeddown_det =  create_hist(f'hfeeddown_det', f';p_{{T}}^{{jet}} (GeV/#it{{c}});{var}', 10, 5,55, 10,0,1)
                hfeeddown_det.Sumw2()
                hfeeddown_det = folding(hfeeddown_gen, response_matrix_np, hfeeddown_det)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det.png')

                hkinematiceff_np_detnogencuts_zg = rfile.Get(f'hkinematiceff_np_detnogencuts_zg')
                hkinematiceff_np_detgencuts_zg = rfile.Get(f'hkinematiceff_np_detgencuts_zg')
                hkinematiceff_np_detgencuts_zg.Divide(hkinematiceff_np_detnogencuts_zg)

                self._save_hist(hkinematiceff_np_detgencuts_zg, f'fd/h_effkine-ptjet-{var}_np_det.png','text')
                hfeeddown_det.Divide(hkinematiceff_np_detgencuts_zg)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_kineeffscaled.png')

                hfeeddown_det.Scale(self.cfg('branching_ratio'))
                hfeeddown_det.Scale(self.n_events['data'] * powheg_xsection_scale_factor / self.cfg('xsection_inel'))
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_final.png')
