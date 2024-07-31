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
from machine_learning_hep.fitting.roofitter import RooFitter
from machine_learning_hep.utilities import folding
from machine_learning_hep.utils.hist import (bin_array, create_hist,
                                             fill_hist_fast, get_axis, get_dim,
                                             get_nbins, project_hist,
                                             scale_bin, sum_hists, ensure_sumw2)


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

        self.fit_levels = self.cfg('fit_levels', ['mc', 'data'])
        self.fit_sigma = {}
        self.fit_mean = {}
        self.fit_func_bkg = {}
        self.fit_range = {}
        self.hcandeff = None
        self.hcandeff_np = None
        self.hfeeddown_det = { 'mc': {}, 'data': {}}
        self.n_events = {}
        self.n_colls = {}

        self.path_fig = Path(f'fig/{self.case}/{self.typean}')
        for folder in ['qa', 'fit', 'roofit', 'sideband', 'signalextr', 'fd', 'uf']:
            (self.path_fig / folder).mkdir(parents=True, exist_ok=True)

        self.rfigfile = TFile(str(self.path_fig / 'output.root'), 'recreate')

        self.fitter = RooFitter()
        self.roo_ws = {}
        self.roows = {}

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
        if isinstance(hist, ROOT.TH1) and get_dim(hist) == 2 and 'texte' not in option:
            option += 'texte'
        hist.Draw(option)
        self._save_canvas(c, filename)
        rfilename = filename.split('/')[-1]
        rfilename = rfilename.removesuffix('.png')
        self.rfigfile.WriteObject(hist, rfilename)


    #region fundamentals
    def init(self):
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                histonorm = rfile.Get("histonorm")
                if not histonorm:
                    self.logger.critical('histonorm not found')
                self.n_events[mcordata] = histonorm.GetBinContent(1)
                self.n_colls[mcordata] = histonorm.GetBinContent(2)
                self.logger.info('Number of sampled collisions for %s: %g', mcordata, self.n_colls[mcordata])
                self.logger.debug('Number of selected events for %s: %d', mcordata, self.n_events[mcordata])


    def qa(self): # pylint: disable=invalid-name
        self.logger.info("Producing basic QA histograms")
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
                        hproj = project_hist(h, axes[3:], {1: [2,2]}) # temporary select higher jet pt bin
                        self._save_hist(hproj, f'qa/h_{var}_{mcordata}.png')

        with TFile(self.n_fileeff) as rfile:
            for var in self.observables['all']:
                if '-' in var:
                    continue
                h_gen = []
                h_ineff = {'pr': [], 'np': []}
                for cat in ('pr', 'np'):
                    if fh := rfile.Get(f'h_ptjet-pthf-{var}_{cat}_gen'):
                        h_gen.append(fh)
                        for ipt in range(self.nbins):
                            h = project_hist(fh, [0, 2], {1: (ipt+1, ipt+1)})
                            self._save_hist(h, f'qa/h_ptjet-{var}_{cat}_gen_ptbin{ipt}.png')
                            h = h.Clone()
                            h.Scale(self.hcandeff[ipt+1] if cat == 'pr' else self.hcandeff_np[ipt+1])
                            self._save_hist(h, f'qa/h_ptjet-{var}_{cat}_exp_ptbin{ipt}.png')
                            h_ineff[cat].append(h)
                    else:
                        self.logger.error('could not find %s', f'h_ptjet-pthf-{var}_{cat}_gen')
                        rfile.ls()

                h_sum = sum_hists(h_gen)
                for ipt in range(self.nbins):
                    h = project_hist(h_sum, [0, 2], {1: (ipt+1, ipt+1)})
                    self._save_hist(h, f'qa/h_ptjet-{var}_all_gen_ptbin{ipt}.png')

                    h_exp = sum_hists([h_ineff['pr'][ipt], h_ineff['np'][ipt]])
                    self._save_hist(h_exp, f'qa/h_ptjet-{var}_all_exp_ptbin{ipt}.png')

    #region efficiency
    def calculate_efficiencies(self):
        self.logger.info("Calculating efficiencies")
        cats = {'pr', 'np'}
        rfilename = self.n_fileeff
        with TFile(rfilename) as rfile:
            bins_ptjet = (1, 4)
            # TODO: fix projection range
            h_gen = {cat: project_hist(rfile.Get(f'h_ptjet-pthf_{cat}_gen'), [1], {0: bins_ptjet}) for cat in cats}
            h_det = {cat: project_hist(rfile.Get(f'h_ptjet-pthf_{cat}_det'), [1], {0: bins_ptjet}).Clone(f'h_eff_{cat}')
                     for cat in cats}

            for cat in cats:
                self._save_hist(h_gen[cat], f'qa/h_pthf_{cat}_gen.png')
                self._save_hist(h_det[cat], f'qa/h_pthf_{cat}_det.png')
                ensure_sumw2(h_det[cat])
                h_det[cat].Divide(h_gen[cat]) # TODO: check uncertainties
                self._save_hist(h_det[cat], f'h_eff_{cat}.png')

            self.hcandeff = h_det['pr']
            self.hcandeff_np = h_det['np']


    def _correct_efficiency(self, hist, ipt):
        if not hist:
            self.logger.error('no histogram to correct for efficiency')
            return

        if not self.hcandeff:
            self.logger.error('no efficiency available for %s', hist.GetName())
            return

        eff = self.hcandeff.GetBinContent(ipt + 1)
        if np.isclose(eff, 0):
            if hist.GetEntries() > 0:
                # TODO: how should we handle this?
                self.logger.error('Efficiency 0 for %s ipt %d, no correction possible',
                                  hist.GetName(), ipt)
            return

        self.logger.debug('scaling hist %s (ipt %i) with 1. / %g', hist.GetName(), ipt, eff)
        hist.Scale(1. / eff)


    #region fitting
    def _roofit_mass(self, hist, ipt, fitcfg, roows = None, filename = None):
        if fitcfg is None:
            return None, None
        res, ws, frame = self.fitter.fit_mass_new(hist, fitcfg, roows, True)
        frame.SetTitle(f'inv. mass for p_{{T}} {self.bins_candpt[ipt]} - {self.bins_candpt[ipt+1]} GeV/c')
        c = TCanvas()
        frame.Draw()
        if res.status() == 0:
            self._save_canvas(c, filename)
        else:
            self.logger.warning('Invalid fit result for %s', hist.GetName())
            # func_tot.Print('v')
            filename = filename.replace('.png', '_invalid.png')
            self._save_canvas(c, filename)
        return res, ws


    def _fit_mass(self, hist, filename = None):
        if hist.GetEntries() == 0:
            raise UserWarning('Cannot fit histogram with no entries')
        fit_range = self.cfg('mass_fit.range')
        func_sig = TF1('funcSig', self.cfg('mass_fit.func_sig'), *fit_range)
        func_bkg = TF1('funcBkg', self.cfg('mass_fit.func_bkg'), *fit_range)
        par_offset = func_sig.GetNpar()
        func_tot = TF1('funcTot', f"{self.cfg('mass_fit.func_sig')} + {self.cfg('mass_fit.func_bkg')}({par_offset})")
        func_tot.SetParameter(0, hist.GetMaximum()/3.) # TODO: better seeding?
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
            if filename:
                c = TCanvas()
                hist.Draw()
                func_sig.SetLineColor(ROOT.kBlue)
                func_sig.Draw('lsame')
                func_bkg.SetLineColor(ROOT.kCyan)
                func_bkg.Draw('lsame')
                self._save_canvas(c, filename)
        else:
            self.logger.warning('Invalid fit result for %s', hist.GetName())
            # func_tot.Print('v')
            filename = filename.replace('.png', '_invalid.png')
            self._save_hist(hist, filename)
            # TODO: how to deal with this

        return (fit_res, func_sig, func_bkg)


    # pylint: disable=too-many-branches,too-many-statements
    def fit(self):
        self.logger.info("Fitting inclusive mass distributions")
        gStyle.SetOptFit(1111)
        for level in self.fit_levels:
            self.fit_mean[level] = [None] * self.nbins
            self.fit_sigma[level] = [None] * self.nbins
            self.fit_func_bkg[level] = [None] * self.nbins
            self.fit_range[level] = [None] * self.nbins
            self.roo_ws[level] = [None] * self.nbins
            rfilename = self.n_filemass_mc if "mc" in level else self.n_filemass
            fitcfg = None
            with TFile(rfilename) as rfile:
                h = rfile.Get('h_mass-ptjet-pthf')
                for ipt in range(get_nbins(h, 2)):
                    self.logger.debug('fitting %s - %i', level, ipt)
                    roows = self.roows.get(ipt)
                    # TODO: add plots per jet pt bin
                    h_invmass = project_hist(h, [0], {2: (ipt+1, ipt+1)}) # TODO: under-/overflow for jets
                    if h_invmass.GetEntries() < 100: # TODO: reconsider criterion
                        self.logger.error('Not enough entries to fit for %s bin %d', level, ipt)
                        continue
                    ptrange = (self.bins_candpt[ipt], self.bins_candpt[ipt+1])
                    if self.cfg('mass_fit'):
                        fit_res, _, func_bkg = self._fit_mass(
                            h_invmass,
                            f'fit/h_mass_fitted_pthf-{ptrange[0]}-{ptrange[1]}_{level}.png')
                        if fit_res and fit_res.Get() and fit_res.IsValid():
                            self.fit_mean[level][ipt] = fit_res.Parameter(1)
                            self.fit_sigma[level][ipt] = fit_res.Parameter(2)
                            self.fit_func_bkg[level][ipt] = func_bkg
                        else:
                            self.logger.error('Fit failed for %s bin %d', level, ipt)
                    if self.cfg('mass_roofit'):
                        for entry in self.cfg('mass_roofit', []):
                            if lvl := entry.get('level'):
                                if lvl != level:
                                    continue
                            if ptspec := entry.get('ptrange'):
                                if ptspec[0] > ptrange[0] or ptspec[1] < ptrange[1]:
                                    continue
                            fitcfg = entry
                            break
                        self.logger.debug("Using fit config for %i: %s", ipt, fitcfg)
                        if datasel := fitcfg.get('datasel'):
                            h = rfile.Get(f'h_mass-ptjet-pthf_{datasel}')
                            h_invmass = project_hist(h, [0], {2: (ipt+1, ipt+1)}) # TODO: under-/overflow for jets
                        for fixpar in fitcfg.get('fix_params', []):
                            roows.var(fixpar).setConstant(True)
                        roo_res, roo_ws = self._roofit_mass(
                            h_invmass, ipt, fitcfg, roows,
                            f'roofit/h_mass_fitted_pthf-{ptrange[0]}-{ptrange[1]}_{level}.png')
                        # if level == 'mc':
                        #     roo_ws.Print()
                        # TODO: save snapshot per level
                        # roo_ws.saveSnapshot(level, None)
                        self.roo_ws[level][ipt] = roo_ws
                        self.roows[ipt] = roo_ws
                        if roo_res.status() == 0:
                            # TODO: take parameter names from DB
                            if level in ('data', 'mc_sig'):
                                self.fit_mean[level][ipt] = roo_ws.var('mean').getValV()
                                self.fit_sigma[level][ipt] = roo_ws.var('sigma_g1').getValV()
                            var_m = fitcfg.get('var', 'm')
                            if roo_ws.pdf("bkg"):
                                self.fit_func_bkg[level][ipt] = roo_ws.pdf("bkg").asTF(roo_ws.var(var_m))
                            self.fit_range[level][ipt] = (roo_ws.var(var_m).getMin('fit'),
                                                          roo_ws.var(var_m).getMax('fit'))
                            self.logger.info('fit range for %s-%i: %s', level, ipt, self.fit_range[level][ipt])
                        else:
                            self.logger.error('RooFit failed for %s bin %d', level, ipt)


    #region sidebands
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    def _subtract_sideband(self, hist, var, mcordata, ipt):
        """
        Subtract sideband distributions, assuming mass on first axis
        """
        if not hist:
            self.logger.error('no histogram for %s bin %d', var, ipt)
            return None
        label = f'-{var}' if var else ''
        ptrange = (self.bins_candpt[ipt], self.bins_candpt[ipt+1])
        self._save_hist(hist, f'sideband/h_mass-ptjet{label}_pthf-{ptrange[0]}-{ptrange[1]}_{mcordata}.png')

        mean = self.fit_mean[mcordata][ipt]
        # self.logger.info('means %g, %g', mean, self.roows[ipt].var('mean').getVal())
        sigma = self.fit_sigma[mcordata][ipt]
        # self.logger.info('sigmas %g, %g', sigma, self.roows[ipt].var('sigma_g1').getVal())
        fit_range = self.fit_range[mcordata][ipt]
        if mean is None or sigma is None or fit_range is None:
            self.logger.error('no fit parameters for %s bin %d', hist.GetName(), ipt)
            return None

        for entry in self.cfg('sidesub', []):
            if level := entry.get('level'):
                if level != mcordata:
                    continue
            if ptrange_sel := entry.get('ptrange'):
                if ptrange_sel[0] > self.bins_candpt[ipt] or ptrange_sel[1] < self.bins_candpt[ipt+1]:
                    continue
            regcfg = entry['regions']
            break
        regions = {
            'signal': (mean + regcfg['signal'][0] * sigma, mean + regcfg['signal'][1] * sigma),
            'sideband_left': (mean + regcfg['left'][0] * sigma, mean + regcfg['left'][1] * sigma),
            'sideband_right': (mean + regcfg['right'][0] * sigma, mean + regcfg['right'][1] * sigma)
        }
        if regions['sideband_left'][1] < fit_range[0] or regions['sideband_right'][0] > fit_range[1]:
            # TODO: change back to critical
            self.logger.error('sidebands %s not in fit range %s, fix regions!', regions, fit_range)
        for reg, lim in regions.items():
            if lim[0] < fit_range[0] or lim[1] > fit_range[1]:
                regions[reg] = (max(lim[0], fit_range[0]), min(lim[1], fit_range[1]))
                self.logger.warning('region %s for %s bin %d (%s) extends beyond fit range: %s, clipping to %s',
                                    reg, mcordata, ipt, ptrange, lim, regions[reg])
        axis = get_axis(hist, 0)
        bins = {key: tuple(map(axis.FindBin, region)) for key, region in regions.items()}
        limits = {key: (axis.GetBinLowEdge(bins[key][0]), axis.GetBinUpEdge(bins[key][1]))
                  for key in regions}
        self.logger.info('Using for %s-%i: %s, %s', mcordata, ipt, regions, limits)

        fh = {}
        area = {}
        var_m = self.roows[ipt].var("m")
        for region in regions:
            # project out the mass regions (first axis)
            axes = list(range(get_dim(hist)))[1:]
            fh[region] = project_hist(hist, axes, {0: bins[region]})
            self._save_hist(fh[region],
                            f'sideband/h_ptjet{label}_{region}_pthf-{ptrange[0]}-{ptrange[1]}_{mcordata}.png')
            f = self.roo_ws[mcordata][ipt].pdf("bkg").asTF(self.roo_ws[mcordata][ipt].var("m"))
            area[region] = f.Integral(*limits[region])

        self.logger.info('areas for %s-%s: %g, %g, %g',
                         mcordata, ipt, area['signal'], area['sideband_left'], area['sideband_right'])
        areaNormFactor = area['signal'] / (area['sideband_left'] + area['sideband_right'])

        fh_sideband = sum_hists(
            [fh['sideband_left'], fh['sideband_right']], f'h_ptjet{label}_sideband_{ipt}_{mcordata}')
        self._save_hist(fh_sideband, f'sideband/h_ptjet{label}_sideband_pthf-{ptrange[0]}-{ptrange[1]}_{mcordata}.png')

        fh_subtracted = fh['signal'].Clone(f'h_ptjet{label}_subtracted_{ipt}_{mcordata}')
        ensure_sumw2(fh_subtracted)
        if mcordata == 'data' or not self.cfg('closure.exclude_feeddown_det'):
            fh_subtracted.Add(fh_sideband, -areaNormFactor)

        roows = self.roows[ipt]
        roows.var('mean').setVal(self.fit_mean[mcordata][ipt])
        roows.var('sigma_g1').setVal(self.fit_sigma[mcordata][ipt])
        var_m.setRange('signal', *limits['signal'])
        var_m.setRange('sidel', *limits['sideband_left'])
        var_m.setRange('sider', *limits['sideband_right'])
        # correct for reflections
        if self.cfg('corr_refl') and (mcordata == 'data' or not self.cfg('closure.filter_reflections')):
            pdf_sig = self.roows[ipt].pdf('sig')
            pdf_refl = self.roows[ipt].pdf('refl')
            pdf_bkg = self.roows[ipt].pdf('bkg')
            frac_sig = roows.var('frac').getVal() if mcordata == 'data' else 1.
            frac_bkg = 1. - frac_sig
            fac_sig = frac_sig * (1. - roows.var('frac_refl').getVal())
            fac_refl = frac_sig * roows.var('frac_refl').getVal()
            fac_bkg = frac_bkg

            area_sig_sig = pdf_sig.createIntegral(var_m, ROOT.RooFit.NormSet(var_m),
                                                  ROOT.RooFit.Range('signal')).getVal() * fac_sig
            area_refl_sig = pdf_refl.createIntegral(var_m, ROOT.RooFit.NormSet(var_m),
                                                    ROOT.RooFit.Range('signal')).getVal() * fac_refl
            area_refl_sidel = pdf_refl.createIntegral(var_m, ROOT.RooFit.NormSet(var_m),
                                                      ROOT.RooFit.Range('sidel')).getVal() * fac_refl
            area_refl_sider = pdf_refl.createIntegral(var_m, ROOT.RooFit.NormSet(var_m),
                                                      ROOT.RooFit.Range('sider')).getVal() * fac_refl
            area_refl_side = area_refl_sidel + area_refl_sider
            area_bkg_sig = pdf_bkg.createIntegral(var_m, ROOT.RooFit.NormSet(var_m),
                                                  ROOT.RooFit.Range('signal')).getVal() * fac_bkg
            area_bkg_sidel = pdf_bkg.createIntegral(var_m, ROOT.RooFit.NormSet(var_m),
                                                    ROOT.RooFit.Range('sidel')).getVal() * fac_bkg
            area_bkg_sider = pdf_bkg.createIntegral(var_m, ROOT.RooFit.NormSet(var_m),
                                                    ROOT.RooFit.Range('sider')).getVal() * fac_bkg
            area_bkg_side = area_bkg_sidel + area_bkg_sider

            scale_bkg = area_bkg_sig / area_bkg_side if mcordata == 'data' else 1.
            corr = area_sig_sig / (area_sig_sig + area_refl_sig - area_refl_side * scale_bkg)
            self.logger.info('Correcting %s-%i for reflections with factor %g', mcordata, ipt, corr)
            fh_subtracted.Scale(corr)

        # clip negative values to 0
        for ibin in range(fh_subtracted.GetNcells()):
            if fh_subtracted.GetBinContent(ibin) < 0:
                fh_subtracted.SetBinContent(ibin, 0.)
                fh_subtracted.SetBinError(ibin, 0.)

        pdf_sig = self.roows[ipt].pdf('sig')
        frac_sig = pdf_sig.createIntegral(var_m, ROOT.RooFit.NormSet(var_m), ROOT.RooFit.Range('signal')).getVal()
        self.logger.info('correcting %s-%i for fractional signal area: %g', mcordata, ipt, frac_sig)

        fh_subtracted.Scale(1. / frac_sig)
        self._save_hist(fh_subtracted, f'sideband/h_ptjet{label}_subtracted_{ptrange[0]}-{ptrange[1]}_{mcordata}.png')

        if get_dim(hist) == 2: # TODO: extract 1d distribution also in case of higher dimension
            c = TCanvas()
            fh['signal'].SetLineColor(ROOT.kRed)
            fh['signal'].Draw()
            ensure_sumw2(fh_sideband)
            fh_sideband.Scale(areaNormFactor)
            fh_sideband.SetLineColor(ROOT.kCyan)
            fh_sideband.Draw("same")
            fh_subtracted.Draw("same")
            fh_subtracted.GetYaxis().SetRangeUser(
                0., max(fh_subtracted.GetMaximum(), fh['signal'].GetMaximum(), fh_sideband.GetMaximum()))
            self._save_canvas(c, f'sideband/h_ptjet{label}_overview_{ptrange[0]}-{ptrange[1]}_{mcordata}.png')

        return fh_subtracted


    # region analysis
    def _analyze(self, method = 'sidesub'):
        self.logger.info("Running sideband subtraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in [None] + self.observables['all']:
                    self.logger.info('Running analysis for %s using %s', var, method)
                    label = f'-{var}' if var else ''
                    self.logger.debug('looking for %s', f'h_mass-ptjet-pthf{label}')
                    if fh := rfile.Get(f'h_mass-ptjet-pthf{label}'):
                        axes_proj = list(range(get_dim(fh)))
                        axes_proj.remove(2)
                        fh_sub = []
                        for ipt in range(self.nbins):
                            h = project_hist(fh, axes_proj, {2: (ipt+1, ipt+1)})
                            ensure_sumw2(h)
                            if mcordata == 'mc' and self.cfg('closure.pure_signal'):
                                self.logger.info('assuming pure signal, projecting hist')
                                h = project_hist(h, axes_proj[1:], {})
                            elif method == 'sidesub':
                                h = self._subtract_sideband(h, var, mcordata, ipt)
                            elif method == 'sigextr':
                                h = self._extract_signal(h, var, mcordata, ipt)
                            else:
                                self.logger.critical('invalid method %s', method)
                            if mcordata == 'data' or not self.cfg('closure.use_matched'):
                                self.logger.info('correcting efficiency')
                                self._correct_efficiency(h, ipt)
                            fh_sub.append(h)
                        fh_sum = sum_hists(fh_sub)
                        self._save_hist(fh_sum, f'h_ptjet{label}_{method}_effscaled_{mcordata}.png')

                        if mcordata == 'data': # TODO: temporary
                            self._subtract_feeddown(fh_sum, var, mcordata)
                        self._save_hist(fh_sum, f'h_ptjet{label}_{method}_{mcordata}.png')

                        if not var:
                            continue
                        axis_jetpt = get_axis(fh_sum, 0)
                        for j in range(get_nbins(fh_sum, 0)):
                            # TODO: generalize to higher dimensions
                            hproj = project_hist(fh_sum, [1], {0: [j+1, j+1]})
                            jetptrange = (axis_jetpt.GetBinLowEdge(j+1), axis_jetpt.GetBinUpEdge(j+1))
                            self._save_hist(
                                hproj, f'uf/h_{var}_{method}_{mcordata}_jetpt-{jetptrange[0]}-{jetptrange[1]}.png')
                        fh_unfolded = self._unfold(fh_sum, var, mcordata)
                        for i, h in enumerate(fh_unfolded):
                            self._save_hist(h, f'h_{var}_{method}_unfolded_{mcordata}_{i}.png')
                            for j in range(get_nbins(h, 0)):
                                hproj = project_hist(h, [1], {0: [j+1, j+1]})
                                jetptrange = (axis_jetpt.GetBinLowEdge(j+1), axis_jetpt.GetBinUpEdge(j+1))
                                self._save_hist(
                                    hproj,
                                    f'uf/h_{var}_{method}_unfolded_{mcordata}_' +
                                    f'jetpt-{jetptrange[0]}-{jetptrange[1]}_{i}.png')
                                # TODO: also save all in one


    def analyze_with_sidesub(self):
        self._analyze('sidesub')


    def analyze_with_sigextr(self):
        self._analyze('sigextr')


    #region signal extraction
    def _extract_signal(self, hist, var, mcordata, ipt):
        """
        Extract signal through inv. mass fit (first axis) in bins of other axes
        """
        if not hist:
            self.logger.warning('no histogram for %s bin %d', var, ipt)
            return None
        ptrange = (self.bins_candpt[ipt], self.bins_candpt[ipt+1])
        self._save_hist(hist, f'signalextr/h_mass-{var}_pthf-{ptrange[0]}-{ptrange[1]}_{mcordata}.png')

        if self.fit_mean[mcordata][ipt] is None or self.fit_sigma[mcordata][ipt] is None:
            self.logger.warning('no fit parameters for %s bin %d', var, ipt)
            return None # TODO: should we continue nonetheless?

        axes = list(range(get_dim(hist)))
        hres = project_hist(hist, axes[1:], {}) # TODO: check if we can project without content
        hres.Reset()

        # TODO: take from DB, add scaling, or extend
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
                # TODO: change to RooFit
                fit_res, func_sig, _ = self._fit_mass(
                    hmass, f'signalextr/h_mass-{var}_fitted_pthf-{ptrange[0]}-{ptrange[1]}_{label}_{mcordata}.png')
                if fit_res and fit_res.Get() and fit_res.IsValid():
                    # TODO: consider adding scaling factor
                    hres.SetBinContent(*binid, func_sig.Integral(*range_int) / hmass.GetBinWidth(1))
                else:
                    self.logger.error("Could not extract signal for %s %s %i", var, mcordata, ipt)
        self._save_hist(
            hres,
            f'signalextr/h_{var}_signalextracted_pthf-{ptrange[0]}-{ptrange[1]}_{label}_{mcordata}.png')
        # hres.Sumw2() # TODO: check if we should do this here
        return hres


    #region feeddown
    # pylint: disable=too-many-statements
    def estimate_feeddown(self):
        self.logger.info('Estimating feeddown')

        with TFile(self.cfg('fd_root')) as rfile:
            powheg_xsection = rfile.Get('fHistXsection')
            powheg_xsection_scale_factor = powheg_xsection.GetBinContent(1) / powheg_xsection.GetEntries()
        self.logger.info('powheg scale factor %g', powheg_xsection_scale_factor)
        self.logger.info('number of collisions in data: %g', self.n_colls['data'])
        self.logger.info('number of collisions in MC: %g', self.n_colls['mc'])

        df = pd.read_parquet(self.cfg('fd_parquet'))
        col_mapping = {'dr': 'delta_r_jet', 'zpar': 'z'} # TODO: check mapping

        for var in self.observables['all']:
            bins_ptjet = np.asarray(self.cfg('bins_ptjet'), 'd')
            bins_obs = {var: bin_array(*self.cfg(f'observables.{var}.bins_fix')) for var in self.observables['all']}

            colname = col_mapping.get(var, f'{var}_jet')
            if f'{colname}' not in df:
                if var is not None:
                    self.logger.error('No feeddown information for %s (%s), cannot estimate feeddown', var, colname)
                continue

            # TODO: derive histogram
            # TODO: change order of axes to be consistent
            h3_fd_gen = create_hist('h3_feeddown_gen',
                                    f';p_{{T}}^{{cand}} (GeV/#it{{c}});p_{{T}}^{{jet}} (GeV/#it{{c}});{var}',
                                    self.bins_candpt, bins_ptjet, bins_obs[var])
            fill_hist_fast(h3_fd_gen, df[['pt_cand', 'pt_jet', f'{colname}']])
            self._save_hist(project_hist(h3_fd_gen, [1, 2], {}), f'fd/h_ptjet-{var}_feeddown_gen_noeffscaling.png')

            for ipt in range(get_nbins(h3_fd_gen, axis=0)):
                eff_pr = self.hcandeff.GetBinContent(ipt+1)
                eff_np = self.hcandeff_np.GetBinContent(ipt+1)
                if np.isclose(eff_pr, 0.):
                    self.logger.error('Efficiency zero for %s in pt bin %d, continuing', var, ipt)
                    continue # TODO: how should we handle this?

                for ijetpt in range(get_nbins(h3_fd_gen, axis=1)):
                    for ishape in range(get_nbins(h3_fd_gen, axis=2)):
                        # TODO: consider error propagation
                        scale_bin(h3_fd_gen, eff_np/eff_pr, ipt+1, ijetpt+1, ishape+1)

            h_fd_gen = project_hist(h3_fd_gen, [1, 2], {})
            self._save_hist(h_fd_gen, f'fd/h_ptjet-{var}_feeddown_gen_effscaled.png')

            with TFile(self.n_fileeff) as rfile:
                hkinematiceff_np_gennodetcuts = rfile.Get(f'h_effkine_np_gen_nocuts_{var}')
                hkinematiceff_np_gendetcuts = rfile.Get(f'h_effkine_np_gen_cut_{var}')
                ensure_sumw2(hkinematiceff_np_gendetcuts)
                hkinematiceff_np_gendetcuts.Divide(hkinematiceff_np_gennodetcuts)
                self._save_hist(hkinematiceff_np_gendetcuts, f'fd/h_effkine-ptjet-{var}_np_gen.png', 'text')

                # ROOT complains about different bin limits because fN is 0 for the histogram from file, ROOT bug?
                ensure_sumw2(h_fd_gen)
                h_fd_gen.Multiply(hkinematiceff_np_gendetcuts)
                self._save_hist(h_fd_gen, f'fd/h_ptjet-{var}_feeddown_gen_kineeffscaled.png')

                h_response = rfile.Get(f'h_response_np_{var}')
                response_matrix_np = ROOT.RooUnfoldResponse(
                    project_hist(h_response, [0, 1], {}), project_hist(h_response, [2, 3], {}))
                for hbin in itertools.product(
                    enumerate(get_axis(h_response, 0).GetXbins(), 1),
                    enumerate(get_axis(h_response, 1).GetXbins(), 1),
                    enumerate(get_axis(h_response, 2).GetXbins(), 1),
                    enumerate(get_axis(h_response, 3).GetXbins(), 1),
                    enumerate(get_axis(h_response, 4).GetXbins(), 1)):
                    n = h_response.GetBinContent(
                        np.asarray([hbin[0][0], hbin[1][0], hbin[2][0], hbin[3][0], hbin[4][0]], 'i'))
                    eff = self.hcandeff.GetBinContent(hbin[4][0])
                    for _ in range(int(n)):
                        response_matrix_np.Fill(hbin[0][1], hbin[1][1], hbin[2][1], hbin[3][1], 1./eff)
                # response_matrix_np.Mresponse().Print()

                # response_matrix_np = rfile.Get(f'h_effkine_np_det_nocuts_{var}_h_effkine_np_gen_nocuts_{var}')

                hfeeddown_det = response_matrix_np.Hmeasured().Clone()
                hfeeddown_det.Reset()
                ensure_sumw2(hfeeddown_det)
                hfeeddown_det = folding(h_fd_gen, response_matrix_np, hfeeddown_det)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det.png')

                hkinematiceff_np_detnogencuts = rfile.Get(f'h_effkine_np_det_nocuts_{var}')
                hkinematiceff_np_detgencuts = rfile.Get(f'h_effkine_np_det_cut_{var}')
                ensure_sumw2(hkinematiceff_np_detgencuts)
                hkinematiceff_np_detgencuts.Divide(hkinematiceff_np_detnogencuts)

                self._save_hist(hkinematiceff_np_detgencuts, f'fd/h_effkine-ptjet-{var}_np_det.png','text')
                hfeeddown_det.Divide(hkinematiceff_np_detgencuts)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_kineeffscaled.png')

                # TODO: check scaling
                hfeeddown_det.Scale(powheg_xsection_scale_factor * self.cfg('branching_ratio'))
                hfeeddown_det_mc = hfeeddown_det.Clone()
                hfeeddown_det_mc.SetName(hfeeddown_det_mc.GetName() + '_mc')
                hfeeddown_det.Scale(self.n_colls['data'] / self.cfg('xsection_inel'))
                hfeeddown_det_mc.Scale(self.n_colls['mc'] / self.cfg('xsection_inel_mc'))

                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_final.png')
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_final_mc.png')
                self.hfeeddown_det['data'][var] = hfeeddown_det
                self.hfeeddown_det['mc'][var] = hfeeddown_det_mc


    def _subtract_feeddown(self, hist, var, mcordata):
    # TODO: store and retrieve for correct variable
        if var not in self.hfeeddown_det[mcordata]:
            if var is not None:
                self.logger.error('No feeddown information available for %s, cannot subtract', var)
            return
        if h_fd := self.hfeeddown_det[mcordata][var]:
            if get_dim(hist) == 1:
                h_fd = project_hist(h_fd, [0], {})
            assert get_dim(h_fd) == get_dim(hist)
            hist.Add(h_fd, -1)
        else:
            self.logger.error('No feeddown estimation available for %s (%s)', var, mcordata)


    #region unfolding
    def _unfold(self, hist, var, mcordata):
        self.logger.debug('Unfolding for %s', var)
        suffix = '_frac' if mcordata == 'mc' else ''
        with TFile(self.n_fileeff) as rfile:
            h_response = rfile.Get(f'h_response_pr_{var}{suffix}')
            if not h_response:
                self.logger.error('Response matrix for %s not available, cannot unfold', var + suffix)
                return []
            response_matrix_pr = ROOT.RooUnfoldResponse(
                project_hist(h_response, [0, 1], {}), project_hist(h_response, [2, 3], {}))
            for hbin in itertools.product(
                enumerate(get_axis(h_response, 0).GetXbins(), 1),
                enumerate(get_axis(h_response, 1).GetXbins(), 1),
                enumerate(get_axis(h_response, 2).GetXbins(), 1),
                enumerate(get_axis(h_response, 3).GetXbins(), 1),
                enumerate(get_axis(h_response, 4).GetXbins(), 1)):
                n = h_response.GetBinContent(
                    np.asarray([hbin[0][0], hbin[1][0], hbin[2][0], hbin[3][0], hbin[4][0]], 'i'))
                eff = self.hcandeff.GetBinContent(hbin[4][0])
                for _ in range(int(n)):
                    response_matrix_pr.Fill(hbin[0][1], hbin[1][1], hbin[2][1], hbin[3][1],
                                            1./eff if mcordata == 'data' else 1.)

            # response_matrix_pr = rfile.Get(f'h_effkine_pr_det_nocuts_{var}_h_effkine_pr_gen_nocuts_{var}')

            h_effkine_pr_detnogencuts = rfile.Get(f'h_effkine_pr_det_nocuts_{var}{suffix}')
            h_effkine_pr_detgencuts = rfile.Get(f'h_effkine_pr_det_cut_{var}{suffix}')
            ensure_sumw2(h_effkine_pr_detgencuts)
            h_effkine_pr_detgencuts.Divide(h_effkine_pr_detnogencuts)
            self._save_hist(h_effkine_pr_detgencuts, f'uf/h_effkine-ptjet-{var}_pr_det_{mcordata}.png', 'text')

            fh_unfolding_input = hist.Clone('fh_unfolding_input')
            if get_dim(fh_unfolding_input) != get_dim(h_effkine_pr_detgencuts):
                self.logger.error('histograms with different dimensions, cannot unfold')
                return []
            ensure_sumw2(fh_unfolding_input)
            fh_unfolding_input.Multiply(h_effkine_pr_detgencuts)
            self._save_hist(response_matrix_pr, f'uf/h_ptjet-{var}_response_pr_{mcordata}.png')

            h_effkine_pr_gennodetcuts = rfile.Get(f'h_effkine_pr_gen_nocuts_{var}{suffix}')
            h_effkine_pr_gendetcuts = rfile.Get(f'h_effkine_pr_gen_cut_{var}{suffix}')
            ensure_sumw2(h_effkine_pr_gendetcuts)
            h_effkine_pr_gendetcuts.Divide(h_effkine_pr_gennodetcuts)
            self._save_hist(h_effkine_pr_gendetcuts, f'uf/h_effkine-ptjet-{var}_pr_gen_{mcordata}.png', 'text')

            # TODO: move, has nothing to do with unfolding
            if mcordata == 'mc':
                h_mctruth_pr = rfile.Get(f'h_ptjet-pthf-{var}_pr_gen')
                if h_mctruth_pr:
                    h_mctruth_pr = project_hist(h_mctruth_pr, [0, 2], {})
                    self._save_hist(h_mctruth_pr, f'h_ptjet-{var}_pr_mctruth.png', 'text')
                    h_mctruth_all = h_mctruth_pr.Clone()
                    h_mctruth_np = rfile.Get(f'h_ptjet-pthf-{var}_np_gen')
                    if h_mctruth_np:
                        h_mctruth_np = project_hist(h_mctruth_np, [0, 2], {})
                        self._save_hist(h_mctruth_np, f'h_ptjet-{var}_np_mctruth.png', 'text')
                        h_mctruth_all.Add(h_mctruth_np)
                        self._save_hist(h_mctruth_all, f'h_ptjet-{var}_all_mctruth.png', 'text')

            h_unfolding_output = []
            for n in range(self.cfg('unfolding_iterations', 8)):
                unfolding_object = ROOT.RooUnfoldBayes(response_matrix_pr, fh_unfolding_input, n + 1)
                fh_unfolding_output = unfolding_object.Hreco(2)
                self._save_hist(fh_unfolding_output, f'uf/h_ptjet-{var}_{mcordata}_unfold{n}.png', 'text')
                ensure_sumw2(fh_unfolding_output)
                fh_unfolding_output.Divide(h_effkine_pr_gendetcuts)
                self._save_hist(fh_unfolding_output, f'uf/h_ptjet-{var}_{mcordata}_unfoldeffcorr{n}.png', 'text')
                h_unfolding_output.append(fh_unfolding_output)

                if mcordata == 'mc':
                    if h_mctruth_pr:
                        h_mcunfolded = fh_unfolding_output.Clone()
                        h_mcunfolded.Divide(h_mctruth_pr)
                        self._save_hist(h_mcunfolded, f'uf/h_ptjet-{var}_{mcordata}_closure{n}.png', 'text')
                        for ibin in range(get_nbins(h_mcunfolded, 0)):
                            h = project_hist(h_mcunfolded, [1], {0: (ibin+1,ibin+1)})
                            self._save_hist(h, f'uf/h_{var}_{mcordata}_closure{n}_ptjet{ibin}.png', 'text')
                    else:
                        self.logger.error('Could not find histogram %s', f'h_mctruth_pr_{var}')
                        rfile.ls()

                h_refolding_input = fh_unfolding_output.Clone()
                h_refolding_input.Multiply(h_effkine_pr_gendetcuts)
                h_refolding_output = fh_unfolding_input.Clone()
                h_refolding_output.Reset()
                h_refolding_output = folding(h_refolding_input, response_matrix_pr, h_refolding_output)
                h_refolding_output.Divide(h_effkine_pr_detgencuts)
                self._save_hist(h_refolding_output, f'uf/h_ptjet-{var}_{mcordata}_refold{n}.png', 'text')

                h_refolding_output.Divide(fh_unfolding_input)
                self._save_hist(h_refolding_output, f'uf/h_ptjet-{var}_{mcordata}_refoldratio{n}.png', 'text')
                # TODO: save as 1d projections

            return h_unfolding_output
