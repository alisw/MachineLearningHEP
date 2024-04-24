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

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT
from ROOT import TF1, TCanvas, TFile, gStyle

from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.utilities import folding
from machine_learning_hep.utilities_hist import (bin_spec, create_hist, get_dim, fill_hist, get_axis,
                                                 scale_bin, sum_hists, project_hist)


class AnalyzerJets(Analyzer): # pylint: disable=too-many-instance-attributes
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        # output directories
        self.d_resultsallpmc = (datap["analysis"][typean]["mc"]["results"][period]
                                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"])
        self.d_resultsallpdata = (datap["analysis"][typean]["data"]["results"][period]
                                  if period is not None else datap["analysis"][typean]["data"]["resultsallp"])

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
            'qa': ['zg', 'rg', 'nsd', 'zpar', 'dr', 'lntheta', 'lnkt', 'lntheta-lnkt'],
            'sideband': ['zg'], #, 'rg', 'nsd', 'zpar', 'dr', 'lntheta-lnkt'],
            'signal': ['zg'], #, 'rg', 'nsd', 'zpar', 'dr'],
            'fd': ['zg'],
            'all': [var for var, spec in self.cfg('observables', {}).items()
                    if '-' not in var and 'arraycols' not in spec],
        }

        self.bins_candpt = np.asarray(self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:], 'd')
        self.nbins = len(self.bins_candpt) - 1

        self.fit_sigma = {}
        self.fit_mean = {}
        self.fit_func_bkg = {}
        self.hcandeff = None
        self.hcandeff_np = None
        self.hfeeddown_det = {}
        self.n_events = {}

        self.path_fig = Path(f'fig/{self.case}/{self.typean}')
        for folder in ['qa', 'fit', 'sideband', 'signalextr', 'fd', 'uf']:
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
        if isinstance(hist, ROOT.TH1) and get_dim(hist) == 2 and 'text' not in option:
            option += 'text'
        hist.Draw(option)
        self._save_canvas(c, filename)


    #region fundamentals
    def init(self):
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                histonorm = rfile.Get("histonorm")
                if not histonorm:
                    self.logger.critical('histonorm not found')
                self.n_events[mcordata] = histonorm.GetBinContent(1)
                self.logger.debug('Number of selected events for %s: %d', mcordata, self.n_events[mcordata])


    def qa(self): # pylint: disable=invalid-name
        self.logger.info("Running D0 jet qa")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                h = rfile.Get('h_mass-ptjet-pthf')
                self._save_hist(project_hist(h, [0], {}), f'qa/h_mass_{mcordata}.png')
                self._save_hist(project_hist(h, [1], {}), f'qa/h_ptjet_{mcordata}.png')
                self._save_hist(project_hist(h, [2], {}), f'qa/h_ptcand_{mcordata}.png')

                for var in self.observables['qa']:
                    if h := rfile.Get(f'h_mass-ptjet-pthf-{var}'):
                        axes = list(range(get_dim(h)))
                        hproj = project_hist(h, axes[3:], {})
                        self._save_hist(hproj, f'qa/h_{var}_{mcordata}.png')


    #region efficiency
    def calculate_efficiencies(self):
        self.logger.info("Calculating efficiencies")
        cats = {'pr', 'np'}
        rfilename = self.n_fileeff
        with TFile(rfilename) as rfile:
            h_gen = {cat: rfile.Get(f'h_pthf_{cat}_gen') for cat in cats}
            h_det = {cat: rfile.Get(f'h_pthf_{cat}_det').Clone(f'h_eff_{cat}') for cat in cats}

            for cat in cats:
                self._save_hist(h_gen[cat], f'qa/h_pthf_{cat}_gen.png')
                self._save_hist(h_det[cat], f'qa/h_pthf_{cat}_det.png')
                h_det[cat].Sumw2()
                h_det[cat].Divide(h_gen[cat]) # TODO: check uncertainties
                self._save_hist(h_det[cat], f'h_eff_{cat}.png')

            self.hcandeff = h_det['pr']
            self.hcandeff_np = h_det['np']


    def _correct_efficiency(self, hist, ipt):
        if not hist:
            return

        if not self.hcandeff:
            self.logger.error('no efficiency available for %s', hist.GetName())
            return

        if np.isclose(self.hcandeff.GetBinContent(ipt + 1), 0):
            if hist.GetEntries() > 0:
                # TODO: how should we handle this?
                self.logger.error('Efficiency 0 for %s ipt %d, no correction possible',
                                  hist.GetName(), ipt)
            return

        hist.Scale(1.0 / self.hcandeff.GetBinContent(ipt + 1))


    #region fitting
    def _fit_mass(self, hist, filename = None):
        if hist.GetEntries() == 0:
            raise UserWarning('Cannot fit histogram with no entries')
        fit_range = self.cfg('mass_fit.range')
        func_sig = TF1('funcSig', self.cfg('mass_fit.func_sig'), *fit_range)
        func_bkg = TF1('funcBkg', self.cfg('mass_fit.func_bkg'), *fit_range)
        par_offset = func_sig.GetNpar()
        func_tot = TF1('funcTot', f"{self.cfg('mass_fit.func_sig')} + {self.cfg('mass_fit.func_bkg')}({par_offset})")
        func_tot.SetParameter(0, hist.GetMaximum()) # TODO: better seeding?
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
            # TODO: generalize
            par = func_tot.GetParameters()
            idx = 0
            for i in range(func_sig.GetNpar()):
                func_sig.SetParameter(i, par[idx])
                idx += 1
            for i in range(func_bkg.GetNpar()):
                func_bkg.SetParameter(i, par[idx])
                idx += 1
            # func_tot.Print('v')
            # func_sig.Print('v')
            # func_bkg.Print('v')
        else:
            self.logger.warning('Invalid fit result for %s', hist.GetName())
            filename = filename.replace('.png', '_invalid.png')
            # TODO: how to deal with this
            # func_tot.Print('v')

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
        self.logger.info("Fitting inclusive mass distributions")
        gStyle.SetOptFit(1111)
        for mcordata in ['mc', 'data']:
            self.fit_mean[mcordata] = [None] * self.nbins
            self.fit_sigma[mcordata] = [None] * self.nbins
            self.fit_func_bkg[mcordata] = [None] * self.nbins
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                h = rfile.Get('h_mass-ptjet-pthf')
                for ipt in range(get_axis(h, 2).GetNbins()):
                    h_invmass = project_hist(h, [0], {2: (ipt+1, ipt+1)})
                    if h_invmass.GetEntries() < 10: # TODO: adjust threshold
                        self.logger.error('Not enough entries to fit for %s bin %d', mcordata, ipt)
                        continue
                    fit_res, _, func_bkg = self._fit_mass( h_invmass, f'fit/h_mass_fitted_{ipt}_{mcordata}.png')
                    if fit_res and fit_res.Get() and fit_res.IsValid():
                        self.fit_sigma[mcordata][ipt] = fit_res.Parameter(2)
                        self.fit_mean[mcordata][ipt] = fit_res.Parameter(1)
                        self.fit_func_bkg[mcordata][ipt] = func_bkg
                    else:
                        self.logger.error('Fit failed for %s bin %d', mcordata, ipt)


    #region sidebands
    def _subtract_sideband(self, hist, var, mcordata, ipt):
        """
        Subtract sideband distributions, assuming mass on first axis
        """
        if not hist:
            self.logger.error('no histogram for %s bin %d', var, ipt)
            return None
        self._save_hist(hist, f'sideband/h_mass-{var}_{ipt}_{mcordata}.png')

        mean = self.fit_mean[mcordata][ipt]
        sigma = self.fit_sigma[mcordata][ipt]
        if mean is None or sigma is None:
            self.logger.error('no fit parameters for %s bin %d', hist.GetName(), ipt)
            return None

        regions = {
            'signal': (mean - 2 * sigma, mean + 2 * sigma),
            'sideband_left': (mean - 7 * sigma, mean - 4 * sigma),
            'sideband_right': (mean + 4 * sigma, mean + 7 * sigma)
        }

        axis = get_axis(hist, 0)
        bins = {key: tuple(map(axis.FindBin, region)) for key, region in regions.items()}
        limits = {key: (axis.GetBinLowEdge(bins[key][0]), axis.GetBinUpEdge(bins[key][1]))
                  for key in regions}
        self.logger.debug('actual sideband regions %s', limits)

        fh = {}
        area = {}
        for region in regions:
            # project out the mass regions (first axis)
            axes = list(range(get_dim(hist)))[1:]
            fh[region] = project_hist(hist, axes, {0: bins[region]})
            if get_dim(fh[region]) < 4:
                self._save_hist(fh[region], f'sideband/h_{var}_{region}_{ipt}_{mcordata}.png')
            area[region] = self.fit_func_bkg[mcordata][ipt].Integral(*limits[region])

        areaNormFactor = area['signal'] / (area['sideband_left'] + area['sideband_right'])

        fh_sideband = sum_hists(
            [fh['sideband_left'], fh['sideband_right']], f'h_{var}_sideband_{ipt}_{mcordata}')
        self._save_hist(fh_sideband, f'sideband/h_{var}_sideband_{ipt}_{mcordata}.png')

        fh_subtracted = fh['signal'].Clone(f'h_{var}_subtracted_{ipt}_{mcordata}')
        fh_subtracted.Sumw2()
        fh_subtracted.Add(fh_sideband, -areaNormFactor)
        fh_subtracted.Scale(1.0 / 0.954) # TODO: calculate from region
        self._save_hist(fh_subtracted, f'sideband/h_{var}_subtracted_{ipt}_{mcordata}.png')

        if get_dim(hist) == 2: # TODO: extract 1d distribution also in case of higher dimension
            c = TCanvas()
            fh['signal'].SetLineColor(ROOT.kRed)
            fh['signal'].Draw()
            fh_sideband.Scale(areaNormFactor)
            fh_sideband.SetLineColor(ROOT.kCyan)
            fh_sideband.Draw("same")
            fh_subtracted.Draw("same")
            fh_subtracted.GetYaxis().SetRangeUser(
                0., max(fh_subtracted.GetMaximum(), fh['signal'].GetMaximum(), fh_sideband.GetMaximum()))
            self._save_canvas(c, f'sideband/h_{var}_overview_{ipt}_{mcordata}.png')

        return fh_subtracted


    def subtract_sidebands(self):
        self.logger.info("Running sideband subtraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in self.observables['all']:
                    self.logger.debug('looking for %s', f'h_mass-ptjet-pthf-{var}')
                    if fh := rfile.Get(f'h_mass-ptjet-pthf-{var}'):
                        fh_sub = []
                        for ipt in range(self.nbins):
                            h = project_hist(fh, [0, 1, 3], {2: (ipt+1, ipt+1)})
                            h = self._subtract_sideband(h, var, mcordata, ipt)
                            self._correct_efficiency(h, ipt)
                            fh_sub.append(h)
                        fh_sum = sum_hists(fh_sub)
                        self._save_hist(fh_sum, f'h_{var}_subtracted_effscaled_{mcordata}.png')

                        self._subtract_feeddown(fh_sum, var, mcordata)
                        self._save_hist(fh_sum, f'h_{var}_subtracted_fdcorr_{mcordata}.png')

                        fh_unfolded = self._unfold(fh_sum, var, mcordata)
                        for i, h in enumerate(fh_unfolded):
                            self._save_hist(h, f'h_{var}_subtracted_unfolded_{mcordata}_{i}.png')


    #region signal extraction
    def _extract_signal(self, hist, var, mcordata, ipt):
        """
        Extract signal through inv. mass fit (first axis) in bins of other axes
        """
        if not hist:
            self.logger.warning('no histogram for %s bin %d', var, ipt)
            return None
        self._save_hist(hist, f'signalextr/h_mass-{var}_{ipt}_{mcordata}.png')

        if self.fit_mean[mcordata][ipt] is None or self.fit_sigma[mcordata][ipt] is None:
            self.logger.warning('no fit parameters for %s bin %d', var, ipt)
            return None # TODO: should we continue nonetheless?

        axes = list(range(get_dim(hist)))
        hres = project_hist(hist, axes[1:], {}) # TODO: check if we can project without content
        hres.Reset()

        range_int = (self.fit_mean[mcordata][ipt] - 3 * self.fit_sigma[mcordata][ipt],
                     self.fit_mean[mcordata][ipt] + 3 * self.fit_sigma[mcordata][ipt])

        nbins = [list(range(1, get_axis(hres, i).GetNbins() + 1)) for i in range(get_dim(hres))]
        for binid in itertools.product(*nbins):
            label = f'{binid[0]}'
            for i in range(1, len(binid)):
                label += f'_{binid[i]}'
            limits = {i + 1: (j, j) for i, j in enumerate(binid)}
            hmass = project_hist(hist, [0], limits)
            if hmass.GetEntries() > 100:
                fit_res, func_sig, _ = self._fit_mass(
                    hmass, f'signalextr/h_mass-{var}_fitted_{ipt}_{label}_{mcordata}.png')
                if fit_res and fit_res.Get() and fit_res.IsValid():
                    hres.SetBinContent(*binid, func_sig.Integral(*range_int) / hmass.GetBinWidth(1))
        self._save_hist(hres, f'signalextr/h_{var}_signalextracted_{ipt}_{label}_{mcordata}.png')
        # hres.Sumw2() # TODO: check if we should do this here
        return hres


    def extract_signals(self):
        self.logger.info("Running signal extraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in self.observables['all']:
                    self.logger.debug('looking for %s', f'h_mass-ptjet-pthf-{var}')
                    fh = rfile.Get(f'h_mass-ptjet-pthf-{var}')
                    if fh:
                        fh_sig = []
                        for ipt in range(self.nbins):
                            h = project_hist(fh, [0, 1, 3], {2: (ipt+1, ipt+1)})
                            hres = self._extract_signal(h, var, mcordata, ipt)
                            self._correct_efficiency(hres, ipt)
                            fh_sig.append(hres)
                        fh_sum = sum_hists(fh_sig)
                        self._save_hist(fh_sum, f'h_{var}_sigextr_effscaled_{mcordata}.png')

                        self._subtract_feeddown(fh_sum, var, mcordata)
                        self._save_hist(fh_sum, f'h_{var}_sigextr_fdcorr_{mcordata}.png')

                        fh_unfolded = self._unfold(fh_sum, var, mcordata)
                        for i, h in enumerate(fh_unfolded):
                            self._save_hist(h, f'h_{var}_sigextr_unfolded_{mcordata}_{i}.png')


    #region feeddown
    # pylint: disable=too-many-statements
    def estimate_feeddown(self):
        self.logger.info('Estimating feeddown')

        with TFile('/data2/vkucera/powheg/trees_powheg_fd_F05_R05.root') as rfile:
            powheg_xsection = rfile.Get('fHistXsection')
            powheg_xsection_scale_factor = powheg_xsection.GetBinContent(1) / powheg_xsection.GetEntries()

        for var in self.observables['all']:
            bins_ptjet = np.asarray(self.cfg('bins_ptjet'), 'd')
            bins_obs = {var: bin_spec(*self.cfg(f'observables.{var}.bins_fix')) for var in self.observables['all']}

            df = pd.read_parquet('/data2/jklein/powheg/trees_powheg_fd_F05_R05.parquet') # TODO: read once
            col_mapping = {'dr': 'delta_r_jet', 'zpar': 'z'} # TODO: check mapping
            colname = col_mapping.get(var, f'{var}_jet')
            if f'{colname}' not in df:
                self.logger.error('No feeddown information for %s (%s), cannot estimate feeddown', var, colname)
                continue

            # TODO: derive histogram
            # TODO: speed up histogram filling (bottleneck)
            h3feeddown_gen = create_hist('h3_feeddown_gen',
                                         f';p_{{T}}^{{cand}} (GeV/#it{{c}});p_{{T}}^{{jet}} (GeV/#it{{c}});{var}',
                                         self.bins_candpt, bins_ptjet, bins_obs[var])
            fill_hist(h3feeddown_gen, df[['pt_cand', 'pt_jet', f'{colname}']])
            xaxis = h3feeddown_gen.GetXaxis()
            xaxis.SetRange(1, xaxis.GetNbins())
            self._save_hist(h3feeddown_gen.Project3D("zy"), f'fd/h_ptjet-{var}_feeddown_gen_noeffscaling.png')

            # TODO: last entry is edge of last bin, check other places
            for ipt, bins_candpt in enumerate(self.bins_candpt[:-1]):
                eff_pr = self.hcandeff.GetBinContent(ipt+1)
                eff_np = self.hcandeff_np.GetBinContent(ipt+1)
                if np.isclose(eff_pr, 0.):
                    self.logger.error('Efficiency zero (%s, %d: %s), continuing', var, ipt, bins_candpt)
                    continue # TODO: how should we handle this?

                for ijetpt, _ in enumerate(bins_ptjet):
                    for ishape, _ in enumerate(bins_obs[var]):
                        # TODO: Improve error propagation
                        scale_bin(h3feeddown_gen, eff_np/eff_pr, ipt+1, ijetpt+1, ishape+1)

            hfeeddown_gen = h3feeddown_gen.Project3D("zy")
            self._save_hist(hfeeddown_gen, f'fd/h_ptjet-{var}_feeddown_gen_effscaled.png')

            with TFile(self.n_fileeff) as rfile:
                hkinematiceff_np_gennodetcuts = rfile.Get(f'h_effkine_np_gen_nocuts_{var}')
                hkinematiceff_np_gendetcuts = rfile.Get(f'h_effkine_np_gen_cut_{var}')
                hkinematiceff_np_gendetcuts.Divide(hkinematiceff_np_gennodetcuts)
                self._save_hist(hkinematiceff_np_gendetcuts, f'fd/h_effkine-ptjet-{var}_np_gen.png', 'text')

                # ROOT complains about different bin limits because fN is 0 for the histogram from file, ROOT bug?
                hfeeddown_gen.Multiply(hkinematiceff_np_gendetcuts)
                self._save_hist(hfeeddown_gen, f'fd/h_ptjet-{var}_feeddown_gen_kineeffscaled.png')

                response_matrix_np = rfile.Get(f'h_effkine_np_det_nocuts_{var}_h_effkine_np_gen_nocuts_{var}')
                self._save_hist(response_matrix_np, f'fd/h_ptjet-{var}_response_np.png')

                hfeeddown_det = response_matrix_np.Hmeasured().Clone()
                hfeeddown_det.Sumw2()
                hfeeddown_det = folding(hfeeddown_gen, response_matrix_np, hfeeddown_det)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det.png')

                hkinematiceff_np_detnogencuts = rfile.Get(f'h_effkine_np_det_nocuts_{var}')
                hkinematiceff_np_detgencuts = rfile.Get(f'h_effkine_np_det_cut_{var}')
                hkinematiceff_np_detgencuts.Divide(hkinematiceff_np_detnogencuts)

                self._save_hist(hkinematiceff_np_detgencuts, f'fd/h_effkine-ptjet-{var}_np_det.png','text')
                hfeeddown_det.Divide(hkinematiceff_np_detgencuts)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_kineeffscaled.png')

                hfeeddown_det.Scale(self.cfg('branching_ratio'))
                print('number of events ', self.n_events['data'])
                print('powheg scale factor ', powheg_xsection_scale_factor)
                #TODO : We are artifically increasing by e4 because we dont have the correct number of events in data
                hfeeddown_det.Scale(self.n_events['data'] * 10000 * powheg_xsection_scale_factor /
                                    self.cfg('xsection_inel'))
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_final.png')
                self.hfeeddown_det[var] = hfeeddown_det


    def _subtract_feeddown(self, hist, var, mcordata):
    # TODO: store and retrieve for correct variable
        if mcordata == 'mc': # TODO: move
            return
        if var not in self.hfeeddown_det:
            self.logger.error('No feeddown information available for %s, cannot subtract', var)
            return
        if h_fd := self.hfeeddown_det[var]:
            if get_dim(hist) == 1:
                h_fd = project_hist(h_fd, [0], {})
            assert get_dim(h_fd) == get_dim(hist)
            hist.Add(h_fd, -1)
        else:
            self.logger.error('No feeddown estimation available for %s (%s)', var, mcordata)


    #region unfolding
    def _unfold(self, hist, var, _mcordata):
        self.logger.info('Unfolding for %s', var)
        with TFile(self.n_fileeff) as rfile:
            response_matrix_pr = rfile.Get(f'h_effkine_pr_det_nocuts_{var}_h_effkine_pr_gen_nocuts_{var}')
            if not response_matrix_pr:
                self.logger.error('Response matrix for %s not available, cannot unfold', var)
                return []

            h_effkine_pr_detnogencuts = rfile.Get(f'h_effkine_pr_det_nocuts_{var}')
            h_effkine_pr_detgencuts = rfile.Get(f'h_effkine_pr_det_cut_{var}')
            h_effkine_pr_detgencuts.Divide(h_effkine_pr_detnogencuts)
            self._save_hist(h_effkine_pr_detgencuts, f'uf/h_effkine-ptjet-{var}_pr_det.png', 'text')

            fh_unfolding_input = hist.Clone('fh_unfolding_input')
            if get_dim(fh_unfolding_input) != get_dim(h_effkine_pr_detgencuts):
                self.logger.error('histograms with different dimensions, cannot unfold')
                return []
            fh_unfolding_input.Multiply(h_effkine_pr_detgencuts)
            self._save_hist(response_matrix_pr, f'uf/h_ptjet-{var}_response_pr.png')

            h_effkine_pr_gennodetcuts = rfile.Get(f'h_effkine_pr_gen_nocuts_{var}')
            h_effkine_pr_gendetcuts = rfile.Get(f'h_effkine_pr_gen_cut_{var}')
            h_effkine_pr_gendetcuts.Divide(h_effkine_pr_gennodetcuts)
            self._save_hist(h_effkine_pr_gendetcuts, f'uf/h_effkine-ptjet-{var}_pr_gen.png', 'text')

            h_unfolding_output = []
            for n in range(1):
                unfolding_object = ROOT.RooUnfoldBayes(response_matrix_pr, fh_unfolding_input, n + 1)
                fh_unfolding_output = unfolding_object.Hreco(2)
                self._save_hist(fh_unfolding_output, f'uf/h_unfolded-{var}-{n}.png', 'text')
                fh_unfolding_output.Divide(h_effkine_pr_gendetcuts)
                self._save_hist(fh_unfolding_output, f'uf/h_unfolded_effcorrected-{var}-{n}.png', 'text')
                h_unfolding_output.append(fh_unfolding_output)

            return h_unfolding_output
