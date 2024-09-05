#  © Copyright CERN 2024. All rights not expressly granted are reserved.  #
#                                                                         #
# This program is free software: you can redistribute it and/or modify it #
#  under the terms of the GNU General Public License as published by the  #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version. This program is distributed in the hope that #
#  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  #
#     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    #
#           See the GNU General Public License for more details.          #
#    You should have received a copy of the GNU General Public License    #
#   along with this program. if not, see <https://www.gnu.org/licenses/>. #

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import ROOT
from ROOT import TF1, TCanvas, TFile, gStyle

from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.fitting.roofitter import RooFitter
from machine_learning_hep.utilities import folding, make_message_notfound
from machine_learning_hep.utils.hist import (bin_array, create_hist, norm_response, fold_hist,
                                             fill_hist_fast, get_axis, get_dim, get_bin_limits,
                                             get_nbins, project_hist,
                                             scale_bin, sum_hists, ensure_sumw2)


def string_range_ptjet(range_pt):
    return f"ptjet-{range_pt[0]:g}-{range_pt[1]:g}"


def string_range_pthf(range_pt):
    return f"pthf-{range_pt[0]:g}-{range_pt[1]:g}"


class AnalyzerJets(Analyzer): # pylint: disable=too-many-instance-attributes,too-many-lines
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        # output directories
        self.d_resultsallpmc = (self.cfg(f"mc.results.{period}")
                                if period is not None else self.cfg("mc.resultsallp"))
        self.d_resultsallpdata = (self.cfg(f"data.results.{period}")
                                  if period is not None else self.cfg("data.resultsallp"))

        # input directories (processor output)
        self.d_resultsallpmc_proc = self.d_resultsallpmc
        self.d_resultsallpdata_proc = self.d_resultsallpdata
        # use a different processor output
        if "data_proc" in datap["analysis"][typean]:
            self.d_resultsallpdata_proc = self.cfg(f"data_proc.results.{period}") \
                if period is not None else self.cfg("data_proc.resultsallp")
        if "mc_proc" in datap["analysis"][typean]:
            self.d_resultsallpmc_proc = self.cfg(f"mc_proc.results.{period}") \
                if period is not None else self.cfg("mc_proc.resultsallp")

        # input files
        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata_proc, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc_proc, n_filemass_name)
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_fileeff = os.path.join(self.d_resultsallpmc_proc, self.n_fileeff)
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.n_fileresp = os.path.join(self.d_resultsallpmc_proc, self.n_fileresp)
        file_result_name = datap["files_names"]["resultfilename"]
        self.n_fileresult = os.path.join(self.d_resultsallpdata, file_result_name)

        self.observables = {
            'qa': ['zg', 'rg', 'nsd', 'zpar', 'dr', 'lntheta', 'lnkt', 'lntheta-lnkt'],
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
        self.hcandeff = {'pr': None, 'np': None}
        self.hcandeff_gen = {}
        self.hcandeff_det = {}
        self.h_eff_ptjet_pthf = {}
        self.h_effnew_ptjet_pthf = {'pr': None, 'np': None}
        self.h_effnew_pthf = {'pr': None, 'np': None}
        self.hfeeddown_det = {'mc': {}, 'data': {}}
        self.h_reflcorr = create_hist('h_reflcorr', ';p_{T}^{HF} (GeV/#it{c})', self.bins_candpt)
        self.n_events = {}
        self.n_colls = {}

        self.path_fig = Path(f'{os.path.expandvars(self.d_resultsallpdata)}/fig')
        for folder in ['qa', 'fit', 'roofit', 'sideband', 'signalextr', 'sidesub', 'sigextr', 'fd', 'uf', 'eff']:
            (self.path_fig / folder).mkdir(parents=True, exist_ok=True)

        self.file_out_histo = TFile(self.n_fileresult, 'recreate')

        self.fitter = RooFitter()
        self.roo_ws = {}
        self.roo_ws_ptjet = {}
        self.roows = {}
        self.roows_ptjet = {}

    #region helpers
    def _save_canvas(self, canvas, filename):
        canvas.SaveAs(f'{self.path_fig}/{filename}')


    def _save_hist(self, hist, filename, option = '', logy = False):
        if not hist:
            self.logger.error('No histogram for <%s>', filename)
            # TODO: remove file if it exists?
            return
        c = TCanvas()
        if isinstance(hist, ROOT.TH1) and get_dim(hist) == 2 and len(option) == 0:
            option += 'texte'
        hist.Draw(option)
        c.SetLogy(logy)
        self._save_canvas(c, filename)
        rfilename = filename.split('/')[-1]
        rfilename = rfilename.removesuffix('.png')
        self.file_out_histo.WriteObject(hist, rfilename)


    def _clip_neg(self, hist):
        for ibin in range(hist.GetNcells()):
            if hist.GetBinContent(ibin) < 0:
                hist.SetBinContent(ibin, 0.)
                hist.SetBinError(ibin, 0.)

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

                if h := rfile.Get('h_ncand'):
                    self._save_hist(h, f'qa/h_ncand_{mcordata}.png', logy = True)

                for var in self.observables['qa']:
                    if h := rfile.Get(f'h_mass-ptjet-pthf-{var}'):
                        axes = list(range(get_dim(h)))
                        hproj = project_hist(h, axes[3:], {})
                        self._save_hist(hproj, f'qa/h_{var}_{mcordata}.png')

        with TFile(self.n_fileeff) as rfile:
            for var in self.observables['all']:
                if '-' in var:
                    continue
                for cat in ('pr', 'np'):
                    h_response = rfile.Get(f'h_response_{cat}_{var}')
                    h_response_ptjet = project_hist(h_response, [0, 2], {})
                    h_response_shape = project_hist(h_response, [1, 3], {})
                    self._save_hist(h_response_ptjet, f'qa/h_ptjet-{var}_responsematrix-ptjet_{cat}.png', 'colz')
                    self._save_hist(h_response_shape, f'qa/h_ptjet-{var}_responsematrix-shape_{cat}.png', 'colz')


    #region efficiency
    # pylint: disable=too-many-statements
    def calculate_efficiencies(self):
        self.logger.info("Calculating efficiencies")
        cats = {'pr', 'np'}
        with TFile(self.n_fileeff) as rfile:
            h_gen = {cat: rfile.Get(f'h_ptjet-pthf_{cat}_gen') for cat in cats}
            h_det = {cat: rfile.Get(f'h_ptjet-pthf_{cat}_det') for cat in cats}
            h_genmatch = {cat: rfile.Get(f'h_ptjet-pthf_{cat}_genmatch') for cat in cats}
            h_detmatch = {cat: rfile.Get(f'h_ptjet-pthf_{cat}_detmatch') for cat in cats}
            h_detmatch_gencuts = {cat: rfile.Get(f'h_ptjet-pthf_{cat}_detmatch_gencuts') for cat in cats}

            # Run 2 efficiencies (only use ptjet bins used for analysis)
            bins_ptjet_ana = self.cfg('bins_ptjet', [])
            bins_ptjet = (get_axis(h_gen['pr'], 0).FindBin(min(bins_ptjet_ana)),
                get_axis(h_gen['pr'], 0).FindBin(max(bins_ptjet_ana) - .001))
            self.logger.info('derived ptjet bins: %i - %i', bins_ptjet[0], bins_ptjet[1])
            h_gen_proj = {cat: project_hist(h_gen[cat], [1], {0: bins_ptjet}) for cat in cats}
            h_det_proj = {cat: project_hist(h_detmatch_gencuts[cat], [1], {0: bins_ptjet}) for cat in cats}

            for cat in cats:
                self._save_hist(h_gen_proj[cat], f'eff/h_pthf_{cat}_gen.png')
                self._save_hist(h_det_proj[cat], f'eff/h_pthf_{cat}_det.png')
                ensure_sumw2(h_det_proj[cat])
                self.hcandeff[cat] = h_det_proj[cat].Clone(f'h_eff_{cat}')
                self.hcandeff[cat].Divide(h_gen_proj[cat])
                self._save_hist(self.hcandeff[cat], f'eff/h_eff_{cat}.png')

                # extract efficiencies in bins of jet pt
                ensure_sumw2(h_det[cat])
                self.h_eff_ptjet_pthf[cat] = h_detmatch_gencuts[cat].Clone()
                self.h_eff_ptjet_pthf[cat].Divide(h_gen[cat])
                self._save_hist(self.h_eff_ptjet_pthf[cat], f'eff/h_ptjet-pthf_eff_{cat}.png')
                c = TCanvas()
                c.cd()
                for i, iptjet in enumerate(range(*bins_ptjet)):
                    h = project_hist(self.h_eff_ptjet_pthf[cat], [1], {0: (iptjet, iptjet)})
                    h.DrawCopy('' if i == 0 else 'same')
                    h.SetLineColor(i)
                self._save_canvas(c, f'eff/h_ptjet-pthf_eff_{cat}_ptjet.png')

            # Run 3 efficiencies
            for cat in cats:
                # gen-level efficiency for feeddown estimation
                h_eff_gen = h_genmatch[cat].Clone()
                h_eff_gen.Divide(h_gen[cat])
                self._save_hist(h_eff_gen, f'eff/h_effgen_{cat}.png')
                self.hcandeff_gen[cat] = h_eff_gen

                # matching loss
                h_eff_match = h_detmatch[cat].Clone()
                h_eff_match.Divide(h_det[cat])
                self._save_hist(h_eff_match, f'eff/h_effmatch_{cat}.png')

                if not (h_response := rfile.Get(f'h_response_{cat}_fPt')):
                    self.logger.critical(make_message_notfound(f'h_response_{cat}_fPt', self.n_fileeff))
                h_response_ptjet = project_hist(h_response, [0, 2], {})
                h_response_pthf = project_hist(h_response, [1, 3], {})
                self._save_hist(h_response_ptjet, f'eff/h_ptjet-pthf_responsematrix-ptjet_{cat}.png', 'colz')
                self._save_hist(h_response_pthf, f'eff/h_ptjet-pthf_responsematrix-pthf_{cat}.png', 'colz')
                rm = self._build_response_matrix(h_response, self.hcandeff['pr'])
                h_effkine_gen = self._build_effkine(
                    rfile.Get(f'h_effkine_{cat}_gen_nocuts_fPt'),
                    rfile.Get(f'h_effkine_{cat}_gen_cut_fPt'))
                self._save_hist(h_effkine_gen, f'eff/h_effkine-ptjet-pthf_{cat}_gen.png', 'text')
                h_effkine_det = self._build_effkine(
                    rfile.Get(f'h_effkine_{cat}_det_nocuts_fPt'),
                    rfile.Get(f'h_effkine_{cat}_det_cut_fPt'))
                self._save_hist(h_effkine_det, f'eff/h_effkine-ptjet-pthf_{cat}_det.png', 'text')

                h_in = h_gen[cat].Clone()
                self._save_hist(project_hist(h_in, [1], {}), f'eff/h_pthf_{cat}_gen.png')
                h_in.Multiply(h_effkine_gen)
                h_out = h_in.Clone() # should derive this from the response matrix instead
                h_out = folding(h_in, rm, h_out)
                h_out.Divide(h_effkine_det)
                self._save_hist(project_hist(h_out, [1], {}), f'eff/h_pthf_{cat}_gen_folded.png')

                eff = h_det[cat].Clone(f'h_effnew_{cat}')
                ensure_sumw2(eff)
                eff.Divide(h_out)
                self._save_hist(eff, f'eff/h_ptjet-pthf_effnew_{cat}.png')
                self.h_effnew_ptjet_pthf[cat] = eff

                eff_avg = project_hist(h_det[cat], [1], {0: bins_ptjet})
                ensure_sumw2(eff_avg)
                eff_avg.Divide(project_hist(h_out, [1], {0: bins_ptjet}))
                self._save_hist(eff_avg, f'eff/h_pthf_effnew_{cat}.png')
                self.h_effnew_pthf[cat] = eff_avg

                c = TCanvas()
                c.cd()
                hc_eff = self.hcandeff[cat].DrawCopy()
                hc_eff.SetLineColor(ROOT.kViolet)
                hc_eff.SetLineWidth(3)
                hc_eff_avg = eff_avg.DrawCopy("same")
                hc_eff_avg.SetLineColor(ROOT.kGreen)
                hc_eff_avg.SetLineWidth(10)
                amax = hc_eff.GetMaximum()
                axis_ptjet = get_axis(eff, 0)
                for iptjet in reversed(range(1, get_nbins(eff, 0) - 1)):
                    h = project_hist(eff, [1], {0: (iptjet+1, iptjet+1)})
                    h.SetName(h.GetName() + f'_ptjet{iptjet}')
                    h.Draw('same')
                    h.SetLineColor(iptjet)
                    range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                    self._save_hist(h, f'h_ptjet-pthf_effnew_{cat}_{string_range_ptjet(range_ptjet)}.png')
                    amax = max(amax, h.GetMaximum())
                hc_eff.GetYaxis().SetRangeUser(0., 1.1 * amax)
                self._save_canvas(c, f'eff/h_ptjet-pthf_effnew_{cat}_ptjet.png')


    def _correct_efficiency(self, hist, ipt):
        if not hist:
            self.logger.error('no histogram to correct for efficiency')
            return

        if self.cfg('efficiency.correction_method') == 'run3':
            eff = self.h_effnew_pthf['pr'].GetBinContent(ipt + 1)
            eff_old = self.hcandeff['pr'].GetBinContent(ipt + 1)
            self.logger.info('Using Run 3 efficiency %g instead of %g', eff, eff_old)
            hist.Scale(1. / eff)
        elif self.cfg('efficiency.correction_method') == 'run2_2d':
            self.logger.info('using Run 2 efficiencies per jet pt bin')
            if not self.h_eff_ptjet_pthf['pr']:
                self.logger.error('no efficiency available for %s', hist.GetName())
                return

            for iptjet in range(get_nbins(hist, 0)):
                eff = self.h_eff_ptjet_pthf['pr'].GetBinContent(iptjet+1, ipt+1)
                if np.isclose(eff, 0):
                    self.logger.error('Efficiency 0 for %s ipt %d iptjet %d, no correction possible',
                                      hist.GetName(), ipt, iptjet)
                    continue
                for ivar in range(get_nbins(hist, 1)):
                    scale_bin(hist, 1./eff, iptjet+1, ivar+1)
        else:
            self.logger.info('Correcting with Run 2 efficiencies')
            if not self.hcandeff['pr']:
                self.logger.error('no efficiency available for %s', hist.GetName())
                return

            eff = self.hcandeff['pr'].GetBinContent(ipt + 1)
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
        if res.status() != 0:
            self.logger.warning('Invalid fit result for %s', hist.GetName())
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
            self.roo_ws_ptjet[level] = [[None] * self.nbins] * 10
            rfilename = self.n_filemass_mc if "mc" in level else self.n_filemass
            fitcfg = None
            self.logger.debug("Opening file %s.", rfilename)
            with TFile(rfilename) as rfile:
                if not rfile:
                    self.logger.critical("File %s not found.", rfilename)
                name_histo = "h_mass-ptjet-pthf"
                self.logger.debug("Opening histogram %s.", name_histo)
                if not (h := rfile.Get(name_histo)):
                    self.logger.critical("Histogram %s not found.", name_histo)
                for iptjet, ipt in itertools.product(itertools.chain((None,), range(get_nbins(h, 1))),
                                                     range(get_nbins(h, 2))):
                    self.logger.debug('fitting %s: %s, %i', level, iptjet, ipt)
                    axis_ptjet = get_axis(h, 1)
                    cuts_proj = {2: (ipt+1, ipt+1)}
                    if iptjet is not None:
                        cuts_proj.update({1: (iptjet+1, iptjet+1)})
                        jetptlabel = f'_{string_range_ptjet(get_bin_limits(axis_ptjet, iptjet + 1))}'
                    else:
                        jetptlabel = ''
                    h_invmass = project_hist(h, [0], cuts_proj)
                    # Rebin
                    if (n_rebin := self.cfg("n_rebin", 1)) != 1:
                        h_invmass.Rebin(n_rebin)
                    range_pthf = (self.bins_candpt[ipt], self.bins_candpt[ipt+1])
                    if self.cfg('mass_fit') and iptjet is None:
                        if h_invmass.GetEntries() < 100: # TODO: reconsider criterion
                            self.logger.error('Not enough entries to fit %s iptjet %s ipt %d',
                                              level, iptjet, ipt)
                            continue
                        fit_res, _, func_bkg = self._fit_mass(
                            h_invmass,
                            f'fit/h_mass_fitted_{string_range_pthf(range_pthf)}_{level}.png')
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
                                if ptspec[0] > range_pthf[0] or ptspec[1] < range_pthf[1]:
                                    continue
                            fitcfg = entry
                            break
                        self.logger.debug("Using fit config for %i: %s", ipt, fitcfg)
                        if iptjet is not None and not fitcfg.get('per_ptjet'):
                            continue
                        # TODO: link datasel to fit stage
                        if datasel := fitcfg.get('datasel'):
                            hist_name = f'h_mass-ptjet-pthf_{datasel}'
                            if not (hsel := rfile.Get(hist_name)):
                                self.logger.critical("Failed to get histogram %s", hist_name)
                            h_invmass = project_hist(hsel, [0], cuts_proj)
                        if h_invmass.GetEntries() < 100: # TODO: reconsider criterion
                            self.logger.error('Not enough entries to fit %s iptjet %s ipt %d',
                                                level, iptjet, ipt)
                            continue
                        roows = self.roows.get(ipt) if iptjet is None else self.roows_ptjet.get((iptjet, ipt))
                        if roows is None and level != self.fit_levels[0]:
                            self.logger.warning('missing previous fit result, skipping %s iptjet %s ipt %d',
                                                level, iptjet, ipt)
                            continue
                        for par in fitcfg.get('fix_params', []):
                            if var := roows.var(par):
                                var.setConstant(True)
                        for par in fitcfg.get('free_params', []):
                            if var := roows.var(par):
                                var.setConstant(False)
                        if iptjet is not None:
                            for par in fitcfg.get('fix_params_ptjet', []):
                                if var := roows.var(par):
                                    var.setConstant(True)
                        roo_res, roo_ws = self._roofit_mass(
                            h_invmass, ipt, fitcfg, roows,
                            f'roofit/h_mass_fitted{jetptlabel}_{string_range_pthf(range_pthf)}_{level}.png')
                        if roo_res.status() != 0:
                            self.logger.error('RooFit failed for %s iptjet %s ipt %d', level, iptjet, ipt)
                        # if level == 'mc':
                        #     roo_ws.Print()
                        # TODO: save snapshot per level
                        # roo_ws.saveSnapshot(level, None)
                        if iptjet is not None:
                            self.roows_ptjet[(iptjet, ipt)] = roo_ws
                            self.roo_ws_ptjet[level][iptjet][ipt] = roo_ws
                        else:
                            self.roows[ipt] = roo_ws
                            self.roo_ws[level][ipt] = roo_ws
                            # TODO: take parameter names from DB
                            if level in ('data', 'mc'):
                                varname_mean = fitcfg.get('var_mean', 'mean')
                                varname_sigma = fitcfg.get('var_sigma', 'sigma_g1')
                                self.fit_mean[level][ipt] = roo_ws.var(varname_mean).getValV()
                                self.fit_sigma[level][ipt] = roo_ws.var(varname_sigma).getValV()
                            varname_m = fitcfg.get('var', 'm')
                            if roo_ws.pdf("bkg"):
                                self.fit_func_bkg[level][ipt] = roo_ws.pdf("bkg").asTF(roo_ws.var(varname_m))
                            self.fit_range[level][ipt] = (roo_ws.var(varname_m).getMin('fit'),
                                                          roo_ws.var(varname_m).getMax('fit'))
                            self.logger.debug('fit range for %s-%i: %s', level, ipt, self.fit_range[level][ipt])

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
        range_pthf = (self.bins_candpt[ipt], self.bins_candpt[ipt+1])
        self._save_hist(hist, f'sideband/h_mass-ptjet{label}_{string_range_pthf(range_pthf)}_{mcordata}.png')

        mean = self.fit_mean[mcordata][ipt]
        # self.logger.info('means %g, %g', mean, self.roows[ipt].var('mean').getVal())
        sigma = self.fit_sigma[mcordata][ipt]
        # self.logger.info('sigmas %g, %g', sigma, self.roows[ipt].var('sigma_g1').getVal())
        fit_range = self.fit_range[mcordata][ipt]
        if mean is None or sigma is None or fit_range is None:
            self.logger.error('no fit parameters for %s bin %s-%d', var or 'none', mcordata, ipt)
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
            self.logger.critical('sidebands %s for %s-%i not in fit range %s, fix regions in DB!',
                                 regions, mcordata, ipt, fit_range)
        for reg, lim in regions.items():
            if lim[0] < fit_range[0] or lim[1] > fit_range[1]:
                regions[reg] = (max(lim[0], fit_range[0]), min(lim[1], fit_range[1]))
                self.logger.warning('region %s for %s bin %d (%s) extends beyond fit range: %s, clipping to %s',
                                    reg, mcordata, ipt, range_pthf, lim, regions[reg])
                if regions[reg][1] < regions[reg][0]:
                    self.logger.error('region limits inverted, reducing to zero width')
                    regions[reg] = (regions[reg][0], regions[reg][0])
        axis = get_axis(hist, 0)
        bins = {key: (axis.FindBin(region[0]), axis.FindBin(region[1]) - 1) for key, region in regions.items()}
        limits = {key: (axis.GetBinLowEdge(bins[key][0]), axis.GetBinUpEdge(bins[key][1])) for key in bins}
        self.logger.debug('Using for %s-%i: %s, %s', mcordata, ipt, regions, limits)

        fh = {}
        area = {}
        var_m = self.roows[ipt].var("m")
        for region in regions:
            # project out the mass regions (first axis)
            axes = list(range(get_dim(hist)))[1:]
            fh[region] = project_hist(hist, axes, {0: bins[region]})
            self.logger.info("Projecting %s to %s in %s: %g entries", hist, axes, bins[region], fh[region].GetEntries())
            self._save_hist(fh[region],
                            f'sideband/h_ptjet{label}_{region}_{string_range_pthf(range_pthf)}_{mcordata}.png')

        fh_subtracted = fh['signal'].Clone(f'h_ptjet{label}_subtracted_{ipt}_{mcordata}')
        ensure_sumw2(fh_subtracted)

        fh_sideband = sum_hists(
            [fh['sideband_left'], fh['sideband_right']], f'h_ptjet{label}_sideband_{ipt}_{mcordata}')
        ensure_sumw2(fh_sideband)

        subtract_sidebands = False
        if mcordata == 'data' and self.cfg('sidesub_per_ptjet'):
            self.logger.info('Subtracting sidebands in pt jet bins')
            for iptjet in range(1, get_nbins(fh_subtracted, 0)):
                if rws := self.roo_ws_ptjet[mcordata][iptjet][ipt]:
                    f = rws.pdf("bkg").asTF(self.roo_ws[mcordata][ipt].var("m"))
                else:
                    self.logger.error('Could not retrieve roows for %s-%i-%i', mcordata, iptjet, ipt)
                    continue
                area = {region: f.Integral(*limits[region]) for region in regions}
                self.logger.info('areas for %s-%s: %g, %g, %g',
                                 mcordata, ipt, area['signal'], area['sideband_left'], area['sideband_right'])
                if (area['sideband_left'] + area['sideband_right']) > 0.:
                    subtract_sidebands = True
                    areaNormFactor = area['signal'] / (area['sideband_left'] + area['sideband_right'])
                    # TODO: extend to higher dimensions
                    for ibin in range(get_nbins(fh_subtracted, 1)):
                        scale_bin(fh_sideband, areaNormFactor, iptjet + 1, ibin + 1)
        else:
            for region in regions:
                f = self.roo_ws[mcordata][ipt].pdf("bkg").asTF(self.roo_ws[mcordata][ipt].var("m"))
                area[region] = f.Integral(*limits[region])

            self.logger.info('areas for %s-%s: %g, %g, %g',
                             mcordata, ipt, area['signal'], area['sideband_left'], area['sideband_right'])

            if (area['sideband_left'] + area['sideband_right']) > 0.:
                subtract_sidebands = True
                areaNormFactor = area['signal'] / (area['sideband_left'] + area['sideband_right'])
                fh_sideband.Scale(areaNormFactor)

        if subtract_sidebands:
            self._save_hist(fh_sideband,
                            f'sideband/h_ptjet{label}_sideband_{string_range_pthf(range_pthf)}_{mcordata}.png')
            fh_subtracted.Add(fh_sideband, -1.)

        self._clip_neg(fh_subtracted)

        self._save_hist(fh_subtracted, f'sideband/h_ptjet{label}_subtracted_notscaled_'
                        f'{string_range_pthf(range_pthf)}_{mcordata}.png')

        # plot subtraction before applying multiplicative corrections
        if get_dim(hist) == 2:
            c = TCanvas()
            fh['signal'].SetLineColor(ROOT.kRed)
            fh['signal'].Draw()
            fh_sideband.SetLineColor(ROOT.kCyan)
            fh_sideband.Draw("same")
            fh_subtracted.Draw("same")
            fh_subtracted.GetYaxis().SetRangeUser(
                0., max(fh_subtracted.GetMaximum(), fh['signal'].GetMaximum(), fh_sideband.GetMaximum()))
            self._save_canvas(c, f'sideband/h_ptjet{label}_overview_{string_range_pthf(range_pthf)}_{mcordata}.png')
        else:
            axis_ptjet = get_axis(hist, 1)
            hists = [fh['signal'], fh_sideband, fh_subtracted]
            cmap = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+3]
            for iptjet in range(get_nbins(hist, 1)):
                c = TCanvas()
                hcs = []
                for i, h in enumerate(map(lambda h, ibin=iptjet+1: project_hist(h, [1], {0: (ibin, ibin)}), hists)):
                    hcs.append(h.DrawCopy('same' if i > 0 else ''))
                    hcs[-1].SetLineColor(cmap[i])
                hcs[0].GetYaxis().SetRangeUser(0., 1.1 * max(map(lambda h: h.GetMaximum(), hcs)))
                range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                filename = (f'sideband/h_{label[1:]}_overview_ptjet-pthf_{string_range_ptjet(range_ptjet)}' +
                            f'_{string_range_pthf(range_pthf)}_{mcordata}.png')
                self._save_canvas(c, filename)

        # TODO: calculate per ptjet bin
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
            self.logger.info('areas: %g, %g, %g, %g; bkgscale: %g',
                             area_sig_sig, area_refl_sig, area_refl_sidel, area_refl_sider, scale_bkg)
            self.h_reflcorr.SetBinContent(ipt + 1, corr)
            fh_subtracted.Scale(corr)

        pdf_sig = self.roows[ipt].pdf('sig')
        frac_sig = pdf_sig.createIntegral(var_m, ROOT.RooFit.NormSet(var_m), ROOT.RooFit.Range('signal')).getVal()
        if pdf_peak := self.roows[ipt].pdf('peak'):
            frac_peak = pdf_peak.createIntegral(var_m, ROOT.RooFit.NormSet(var_m), ROOT.RooFit.Range('signal')).getVal()
            self.logger.info('correcting %s-%i for fractional signal area: %g (Gaussian: %g)',
                             mcordata, ipt, frac_sig, frac_peak)

        fh_subtracted.Scale(1. / frac_sig)
        self._save_hist(fh_subtracted, f'sideband/h_ptjet{label}_subtracted_'
                        f'{string_range_pthf(range_pthf)}_{mcordata}.png')

        return fh_subtracted


    # region analysis
    def _analyze(self, method = 'sidesub'):
        self.logger.info("Running analysis")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in [None] + self.observables['all']:
                    self.logger.info('Running analysis for %s using %s', var, method)
                    label = f'-{var}' if var else ''
                    self.logger.debug('looking for %s', f'h_mass-ptjet-pthf{label}')
                    if fh := rfile.Get(f'h_mass-ptjet-pthf{label}'):  # TODO: add sanity check
                        axes_proj = list(range(get_dim(fh)))
                        axes_proj.remove(2)
                        fh_sub = []
                        self.h_reflcorr.Reset()
                        for ipt in range(self.nbins):
                            h_in = project_hist(fh, axes_proj, {2: (ipt+1, ipt+1)})
                            ensure_sumw2(h_in)
                            # Signal extraction
                            if method == 'sidesub':
                                h = self._subtract_sideband(h_in, var, mcordata, ipt)
                            elif method == 'sigextr':
                                h = self._extract_signal(h_in, var, mcordata, ipt)
                            else:
                                self.logger.critical('invalid method %s', method)
                            self._save_hist(h, f'h_ptjet{label}_{method}_noeff_{mcordata}_pt{ipt}.png')
                            if mcordata == 'mc':
                                h_proj = project_hist(h_in, axes_proj[1:], {})
                                h_proj_lim = project_hist(h_in, axes_proj[1:], {0: (1, get_nbins(h_in, 0))})
                                self._save_hist(h_proj, f'h_ptjet{label}_proj_noeff_{mcordata}_pt{ipt}.png')
                                if h and h_proj:
                                    self.logger.debug('signal loss %s-%i: %g, fraction in under-/overflow: %g',
                                                      mcordata, ipt,
                                                      1. - h.Integral()/h_proj.Integral(),
                                                      1. - h_proj_lim.Integral()/h_proj.Integral())
                                if self.cfg('closure.pure_signal'):
                                    self.logger.debug('assuming pure signal, using projection')
                                    h = h_proj
                            # Efficiency correction
                            if mcordata == 'data' or not self.cfg('closure.use_matched'):
                                self.logger.info('correcting efficiency')
                                self._correct_efficiency(h, ipt)
                            fh_sub.append(h)
                        fh_sum = sum_hists(fh_sub)
                        self._save_hist(self.h_reflcorr, f'h_reflcorr-pthf{label}_reflcorr_{mcordata}.png')
                        self._save_hist(fh_sum, f'h_ptjet{label}_{method}_effscaled_{mcordata}.png')

                        if get_dim(fh_sum) > 1:
                            axes = list(range(get_dim(fh_sum)))
                            axis_ptjet = get_axis(fh_sum, 0)
                            for iptjet in range(get_nbins(fh_sum, 0)):
                                c = TCanvas()
                                h_sig = project_hist(fh_sum, axes[1:], {0: (iptjet+1, iptjet+1)})
                                h_sig.Draw()
                                range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                                filename = (f'{method}/h_{label[1:]}_{method}_effscaled' +
                                            f'_{string_range_ptjet(range_ptjet)}.png')
                                self._save_canvas(c, filename)

                        fh_sum_fdsub = fh_sum.Clone()
                        # Feed-down subtraction
                        if mcordata == 'data' or not self.cfg('closure.exclude_feeddown_det'):
                            self._subtract_feeddown(fh_sum_fdsub, var, mcordata)
                        self._clip_neg(fh_sum_fdsub)
                        self._save_hist(fh_sum_fdsub, f'h_ptjet{label}_{method}_{mcordata}.png')

                        if get_dim(fh_sum) > 1:
                            axes = list(range(get_dim(fh_sum)))
                            axis_ptjet = get_axis(fh_sum, 0)
                            for iptjet in range(get_nbins(fh_sum, 0)):
                                c = TCanvas()
                                c.cd()
                                h_sig = project_hist(fh_sum, axes[1:], {0: (iptjet+1,)*2}).Clone('hsig')
                                h_sig.Draw("same")
                                h_sig.SetLineColor(ROOT.kRed)
                                ymax = h_sig.GetMaximum()
                                if var in self.hfeeddown_det[mcordata]:
                                    h_fd = self.hfeeddown_det[mcordata][var]
                                    h_fd = project_hist(h_fd, axes[1:], {0: (iptjet+1,)*2})
                                    h_fd.DrawCopy('same')
                                    h_fd.SetLineColor(ROOT.kCyan)
                                    ymax = max(ymax, h_fd.GetMaximum())
                                h_fdsub = project_hist(fh_sum_fdsub, axes[1:], {0: (iptjet+1,)*2}).Clone('hfdsub')
                                h_fdsub.Draw('same')
                                h_fdsub.SetLineColor(ROOT.kMagenta)
                                ymax = max(ymax, h_fdsub.GetMaximum())
                                h_sig.GetYaxis().SetRangeUser(0., 1.1 * ymax)
                                range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                                filename = (f'{method}/h_{label[1:]}_{method}_fdsub' +
                                            f'_{string_range_ptjet(range_ptjet)}.png')
                                self._save_canvas(c, filename)

                        if not var:
                            continue
                        axis_ptjet = get_axis(fh_sum_fdsub, 0)
                        for j in range(get_nbins(fh_sum_fdsub, 0)):
                            # TODO: generalize to higher dimensions
                            hproj = project_hist(fh_sum_fdsub, [1], {0: [j+1, j+1]})
                            range_ptjet = get_bin_limits(axis_ptjet, j + 1)
                            self._save_hist(
                                hproj, f'uf/h_{var}_{method}_{mcordata}_{string_range_ptjet(range_ptjet)}.png')
                        # Unfolding
                        fh_unfolded = self._unfold(fh_sum_fdsub, var, mcordata)
                        for i, h in enumerate(fh_unfolded):
                            self._save_hist(h, f'h_ptjet-{var}_{method}_unfolded_{mcordata}_{i}.png')
                        for j in range(get_nbins(h, 0)):
                            range_ptjet = get_bin_limits(axis_ptjet, j + 1)
                            c = TCanvas()
                            for i, h in enumerate(fh_unfolded):
                                hproj = project_hist(h, [1], {0: (j+1, j+1)})
                                self._save_hist(
                                    hproj,
                                    f'uf/h_{var}_{method}_unfolded_{mcordata}_' +
                                    f'{string_range_ptjet(range_ptjet)}_{i}.png')
                                # Save the default unfolding iteration separately.
                                if i == self.cfg("unfolding_iterations_sel") - 1:
                                    self._save_hist(
                                        hproj,
                                        f'uf/h_{var}_{method}_unfolded_{mcordata}_' +
                                        f'{string_range_ptjet(range_ptjet)}_sel.png')
                                    # Save also the self-normalised version.
                                    hproj_sel = hproj.Clone(f"{hproj.GetName()}_selfnorm")
                                    hproj_sel.Scale(1. / hproj_sel.Integral(), "width")
                                    self.logger.debug("Final histogram: %s, jet pT %g to %g",
                                                      var, range_ptjet[0], range_ptjet[1])
                                    # self.logger.debug(print_histogram(hproj_sel))
                                    self._save_hist(
                                        hproj_sel,
                                        f'uf/h_{var}_{method}_unfolded_{mcordata}_' +
                                        f'{string_range_ptjet(range_ptjet)}_sel_selfnorm.png')
                                c.cd()
                                hcopy = hproj.DrawCopy('same' if i > 0 else '')
                                hcopy.SetLineColor(i+1)
                            self._save_canvas(c,
                                              f'uf/h_{var}_{method}_convergence_{mcordata}_' +
                                              f'{string_range_ptjet(range_ptjet)}.png')


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
        range_pthf = (self.bins_candpt[ipt], self.bins_candpt[ipt+1])
        self._save_hist(hist, f'signalextr/h_mass-{var}_{string_range_pthf(range_pthf)}_{mcordata}.png')

        if self.fit_mean[mcordata][ipt] is None or self.fit_sigma[mcordata][ipt] is None:
            self.logger.warning('no fit parameters for %s bin %s-%d', var, mcordata, ipt)
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
                    hmass, f'signalextr/h_mass-{var}_fitted_{string_range_pthf(range_pthf)}_{label}_{mcordata}.png')
                if fit_res and fit_res.Get() and fit_res.IsValid():
                    # TODO: consider adding scaling factor
                    hres.SetBinContent(*binid, func_sig.Integral(*range_int) / hmass.GetBinWidth(1))
                else:
                    self.logger.error("Could not extract signal for %s %s %i", var, mcordata, ipt)
        self._save_hist(
            hres,
            f'signalextr/h_{var}_signalextracted_{string_range_pthf(range_pthf)}_{label}_{mcordata}.png')
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
            # TODO: generalize or derive from histogram?
            bins_obs = {}
            if binning := self.cfg(f'observables.{var}.bins_gen_var'):
                bins_tmp = np.asarray(binning, 'd')
            elif binning := self.cfg(f'observables.{var}.bins_gen_fix'):
                bins_tmp = bin_array(*binning)
            elif binning := self.cfg(f'observables.{var}.bins_var'):
                bins_tmp = np.asarray(binning, 'd')
            elif binning := self.cfg(f'observables.{var}.bins_fix'):
                bins_tmp = bin_array(*binning)
            else:
                self.logger.error('no binning specified for %s, using defaults', var)
                bins_tmp = bin_array(10, 0., 1.)
            bins_obs[var] = bins_tmp

            colname = col_mapping.get(var, f'{var}_jet')
            if f'{colname}' not in df:
                if var is not None:
                    self.logger.error('No feeddown information for %s (%s), cannot estimate feeddown', var, colname)
                continue

            # TODO: derive histogram
            h3_fd_gen_orig = create_hist('h3_feeddown_gen',
                                         f';p_{{T}}^{{jet}} (GeV/#it{{c}});p_{{T}}^{{HF}} (GeV/#it{{c}});{var}',
                                         bins_ptjet, self.bins_candpt, bins_obs[var])
            fill_hist_fast(h3_fd_gen_orig, df[['pt_jet', 'pt_cand', f'{colname}']])
            self._save_hist(project_hist(h3_fd_gen_orig, [0, 2], {}), f'fd/h_ptjet-{var}_feeddown_gen_noeffscaling.png')

            # new method
            h3_fd_gen = h3_fd_gen_orig.Clone()
            ensure_sumw2(h3_fd_gen)
            self._save_hist(project_hist(h3_fd_gen, [0, 2], {}), f'fd/h_ptjet-{var}_fdnew_gen.png')
            # apply np efficiency
            for ipt in range(get_nbins(h3_fd_gen, 1)):
                eff_np = self.hcandeff_gen['np'].GetBinContent(ipt+1)
                for iptjet, ishape in itertools.product(
                        range(get_nbins(h3_fd_gen, 0)), range(get_nbins(h3_fd_gen, 2))):
                    scale_bin(h3_fd_gen, eff_np, iptjet+1, ipt+1, ishape+1)
            self._save_hist(project_hist(h3_fd_gen, [0, 2], {}), f'fd/h_ptjet-{var}_fdnew_gen_geneff.png')

            # 3d folding incl. kinematic efficiencies
            with TFile(self.n_fileeff) as rfile:
                h_effkine_gen = self._build_effkine(
                    rfile.Get(f'h_effkine_fd_gen_nocuts_{var}'),
                    rfile.Get(f'h_effkine_fd_gen_cut_{var}'))
                h_effkine_det = self._build_effkine(
                    rfile.Get(f'h_effkine_fd_det_nocuts_{var}'),
                    rfile.Get(f'h_effkine_fd_det_cut_{var}'))
                h_response = rfile.Get(f'h_response_fd_{var}')
                h_response_norm = norm_response(h_response, 3)
                h3_fd_gen.Multiply(h_effkine_gen)
                self._save_hist(project_hist(h3_fd_gen, [0, 2], {}), f'fd/h_ptjet-{var}_fdnew_gen_genkine.png')
                h3_fd_det = fold_hist(h3_fd_gen, h_response_norm)
                self._save_hist(project_hist(h3_fd_det, [0, 2], {}), f'fd/h_ptjet-{var}_fdnew_det.png')
                h3_fd_det.Divide(h_effkine_det)
                self._save_hist(project_hist(h3_fd_det, [0, 2], {}), f'fd/h_ptjet-{var}_fdnew_det_detkine.png')

            # undo prompt efficiency
            for ipt in range(get_nbins(h3_fd_det, 1)):
                eff_pr = self.h_effnew_pthf['pr'].GetBinContent(ipt+1)
                if np.isclose(eff_pr, 0.):
                    self.logger.error('Efficiency zero for %s in pt bin %d, continuing', var, ipt)
                    continue # TODO: how should we handle this?
                for iptjet, ishape in itertools.product(
                        range(get_nbins(h3_fd_det, 0)), range(get_nbins(h3_fd_det, 2))):
                    scale_bin(h3_fd_det, 1./eff_pr, iptjet+1, ipt+1, ishape+1)
            self._save_hist(project_hist(h3_fd_det, [0, 2], {}), f'fd/h_ptjet-{var}_fdnew_det_deteff.png')

            # project to 2d (ptjet-shape)
            h_fd_det = project_hist(h3_fd_det, [0, 2], {})

            # old method
            h3_fd_gen = h3_fd_gen_orig.Clone()
            ensure_sumw2(h3_fd_gen)
            for ipt in range(get_nbins(h3_fd_gen, 1)):
                eff_pr = self.hcandeff['pr'].GetBinContent(ipt+1)
                eff_np = self.hcandeff['np'].GetBinContent(ipt+1)
                if np.isclose(eff_pr, 0.):
                    self.logger.error('Efficiency zero for %s in pt bin %d, continuing', var, ipt)
                    continue # TODO: how should we handle this?
                for iptjet, ishape in itertools.product(
                        range(get_nbins(h3_fd_gen, 0)), range(get_nbins(h3_fd_gen, 2))):
                    scale_bin(h3_fd_gen, eff_np/eff_pr, iptjet+1, ipt+1, ishape+1)

            h_fd_gen = project_hist(h3_fd_gen, [0, 2], {})
            self._save_hist(h_fd_gen, f'fd/h_ptjet-{var}_feeddown_gen_effscaled.png')

            with TFile(self.n_fileeff) as rfile:
                h_effkine_gen = self._build_effkine(
                    rfile.Get(f'h_effkine_np_gen_nocuts_{var}'),
                    rfile.Get(f'h_effkine_np_gen_cut_{var}'))
                self._save_hist(h_effkine_gen, f'fd/h_effkine-ptjet-{var}_np_gen.png', 'text')

                # ROOT complains about different bin limits because fN is 0 for the histogram from file, ROOT bug?
                ensure_sumw2(h_fd_gen)
                h_fd_gen.Multiply(h_effkine_gen)
                self._save_hist(h_fd_gen, f'fd/h_ptjet-{var}_feeddown_gen_kineeffscaled.png')

                h_response = rfile.Get(f'h_response_np_{var}')
                response_matrix_np  = self._build_response_matrix(h_response, self.hcandeff['pr'])
                self._save_hist(response_matrix_np.Hresponse(), f'fd/h_ptjet-{var}_responsematrix_np_lin.png', 'colz')

                hfeeddown_det = response_matrix_np.Hmeasured().Clone()
                hfeeddown_det.Reset()
                ensure_sumw2(hfeeddown_det)
                hfeeddown_det = folding(h_fd_gen, response_matrix_np, hfeeddown_det)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det.png')

                h_effkine_det = self._build_effkine(
                    rfile.Get(f'h_effkine_np_det_nocuts_{var}'),
                    rfile.Get(f'h_effkine_np_det_cut_{var}'))
                self._save_hist(h_effkine_det, f'fd/h_effkine-ptjet-{var}_np_det.png','text')
                hfeeddown_det.Divide(h_effkine_det)
                self._save_hist(hfeeddown_det, f'fd/h_ptjet-{var}_feeddown_det_kineeffscaled.png')

            if self.cfg('fd_folding_method') == '3d':
                self.logger.info('using 3d folding for feeddown estimation for %s', var)
                hfeeddown_det = h_fd_det
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


    def _build_effkine(self, h_nocuts, h_cuts):
        h_cuts = h_cuts.Clone()
        ensure_sumw2(h_cuts)
        h_cuts.Divide(h_nocuts)
        return h_cuts

    def _build_response_matrix(self, h_response, h_eff = None):
        rm = ROOT.RooUnfoldResponse(
            project_hist(h_response, [0, 1], {}), project_hist(h_response, [2, 3], {}))
        for hbin in itertools.product(
            enumerate(get_axis(h_response, 0).GetXbins(), 1),
            enumerate(get_axis(h_response, 1).GetXbins(), 1),
            enumerate(get_axis(h_response, 2).GetXbins(), 1),
            enumerate(get_axis(h_response, 3).GetXbins(), 1),
            enumerate(get_axis(h_response, 4).GetXbins(), 1)):
            n = h_response.GetBinContent(
                np.asarray([hbin[0][0], hbin[1][0], hbin[2][0], hbin[3][0], hbin[4][0]], 'i'))
            eff = h_eff.GetBinContent(hbin[4][0]) if h_eff else 1.
            if np.isclose(eff, 0.):
                self.logger.error('efficiency 0 for %s', hbin[4])
                continue
            for _ in range(int(n)):
                rm.Fill(hbin[0][1], hbin[1][1], hbin[2][1], hbin[3][1], 1./eff)
        # rm.Mresponse().Print()
        return rm


    def _subtract_feeddown(self, hist, var, mcordata):
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
            response_matrix_pr = self._build_response_matrix(
                h_response, self.hcandeff['pr'] if mcordata == 'data' else None)
            self._save_hist(response_matrix_pr.Hresponse(),
                            f'uf/h_ptjet-{var}-responsematrix_pr_lin_{mcordata}.png', 'colz')

            h_effkine_det = self._build_effkine(
                rfile.Get(f'h_effkine_pr_det_nocuts_{var}{suffix}'),
                rfile.Get(f'h_effkine_pr_det_cut_{var}{suffix}'))
            self._save_hist(h_effkine_det, f'uf/h_effkine-ptjet-{var}_pr_det_{mcordata}.png', 'text')

            fh_unfolding_input = hist.Clone('fh_unfolding_input')
            if get_dim(fh_unfolding_input) != get_dim(h_effkine_det):
                self.logger.error('histograms with different dimensions, cannot unfold')
                return []
            ensure_sumw2(fh_unfolding_input)
            fh_unfolding_input.Multiply(h_effkine_det)

            h_effkine_gen = self._build_effkine(
                rfile.Get(f'h_effkine_pr_gen_nocuts_{var}{suffix}'),
                rfile.Get(f'h_effkine_pr_gen_cut_{var}{suffix}'))
            self._save_hist(h_effkine_gen, f'uf/h_effkine-ptjet-{var}_pr_gen_{mcordata}.png', 'text')

            # TODO: move, has nothing to do with unfolding
            if mcordata == 'mc':
                h_mctruth_pr = rfile.Get(f'h_ptjet-pthf-{var}_pr_gen')
                if h_mctruth_pr:
                    h_mctruth_pr = project_hist(h_mctruth_pr, [0, 2], {})
                    self._save_hist(h_mctruth_pr, f'h_ptjet-{var}_pr_mctruth.png', 'texte')
                    h_mctruth_all = h_mctruth_pr.Clone()
                    h_mctruth_np = rfile.Get(f'h_ptjet-pthf-{var}_np_gen')
                    if h_mctruth_np:
                        h_mctruth_np = project_hist(h_mctruth_np, [0, 2], {})
                        self._save_hist(h_mctruth_np, f'h_ptjet-{var}_np_mctruth.png', 'texte')
                        h_mctruth_all.Add(h_mctruth_np)
                        self._save_hist(h_mctruth_all, f'h_ptjet-{var}_all_mctruth.png', 'texte')

            h_unfolding_output = []
            for n in range(self.cfg('unfolding_iterations', 8)):
                unfolding_object = ROOT.RooUnfoldBayes(response_matrix_pr, fh_unfolding_input, n + 1)
                fh_unfolding_output = unfolding_object.Hreco(2)
                self._save_hist(fh_unfolding_output, f'uf/h_ptjet-{var}_{mcordata}_unfold{n}.png', 'texte')
                ensure_sumw2(fh_unfolding_output)
                fh_unfolding_output.Divide(h_effkine_gen)
                self._save_hist(fh_unfolding_output, f'uf/h_ptjet-{var}_{mcordata}_unfoldeffcorr{n}.png', 'texte')
                h_unfolding_output.append(fh_unfolding_output)

                if mcordata == 'mc':
                    if h_mctruth_pr:
                        h_mcunfolded = fh_unfolding_output.Clone()
                        h_mcunfolded.Divide(h_mctruth_pr)
                        self._save_hist(h_mcunfolded, f'uf/h_ptjet-{var}_{mcordata}_closure{n}.png', 'texte')
                        axis_ptjet = get_axis(h_mcunfolded, 0)
                        for iptjet in range(get_nbins(h_mcunfolded, 0)):
                            h = project_hist(h_mcunfolded, [1], {0: (iptjet+1,iptjet+1)})
                            range_ptjet = get_bin_limits(axis_ptjet, iptjet + 1)
                            self._save_hist(h, f'uf/h_{var}_{mcordata}_closure{n}' +
                                            f'_{string_range_ptjet(range_ptjet)}.png', 'texte')
                    else:
                        self.logger.error('Could not find histogram %s', f'h_mctruth_pr_{var}')
                        rfile.ls()

                h_refolding_input = fh_unfolding_output.Clone()
                h_refolding_input.Multiply(h_effkine_gen)
                h_refolding_output = fh_unfolding_input.Clone()
                h_refolding_output.Reset()
                h_refolding_output = folding(h_refolding_input, response_matrix_pr, h_refolding_output)
                h_refolding_output.Divide(h_effkine_det)
                self._save_hist(h_refolding_output, f'uf/h_ptjet-{var}_{mcordata}_refold{n}.png', 'texte')

                h_refolding_output.Divide(fh_unfolding_input)
                self._save_hist(h_refolding_output, f'uf/h_ptjet-{var}_{mcordata}_refoldratio{n}.png', 'texte')
                # TODO: save as 1d projections

            return h_unfolding_output
