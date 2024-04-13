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

from pathlib import Path
import os
import munch # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TCanvas, TF1, TH1F, TH2F, gStyle # pylint: disable=import-error, no-name-in-module
import ROOT # pylint: disable=import-error
ROOT.TDirectory.AddDirectory(False)

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

        self.observables = ['zg', 'rg', 'nsd', 'zpar', 'dr', 'zg-rg']
        self.bins_candpt = self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:]
        self.nbins = len(self.bins_candpt) - 1

        self.fit_sigma = {}
        self.fit_mean = {}
        self.fit_func_bkg = {}
        self.hcandeff = None

        self.path_fig = Path(f'fig/{self.case}/{self.typean}')
        for folder in ['qa', 'fit', 'sideband', 'signalextr']:
            (self.path_fig / folder).mkdir(parents=True, exist_ok=True)

    #region helpers
    def _save_canvas(self, canvas, filename, mcordata): # pylint: disable=unused-argument
        # folder = self.d_resultsallpmc if mcordata == 'mc' else self.d_resultsallpdata
        canvas.SaveAs(f'fig/{self.case}/{self.typean}/{filename}')

    def _save_hist(self, hist, filename, mcordata):
        if not hist:
            self.logger.error('no histogram for <%s>', filename)
            # TODO: remove file if it exists?
            return
        c = TCanvas()
        hist.Draw()
        self._save_canvas(c, filename, mcordata)

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

    #region basic qa
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

                for ipt in range(self.nbins):
                    self._save_hist(rfile.Get(f'h_mass_{ipt}'), f'qa/h_mass_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'h_ptcand_{ipt}'), f'qa/h_ptcand_{ipt}_{mcordata}.png', mcordata)
                    self._save_hist(rfile.Get(f'h_ptjet_{ipt}'), f'qa/h_ptjet_{ipt}_{mcordata}.png', mcordata)
                    for var in self.observables:
                        if '-' in var: # only plot 1-d distributions
                            continue
                        self._save_hist(rfile.Get(f'h_{var}_{ipt}'), f'qa/h_{var}_{ipt}_{mcordata}.png', mcordata)

    #region efficiency
    def efficiency(self):
        self.logger.info("Running efficiency")
        rfilename = self.n_fileeff
        with TFile(rfilename) as rfile:
            h_gen = rfile.Get('h_pthf_gen')
            h_det = rfile.Get('h_pthf_det')
            heff = h_det.Clone('h_eff')
            heff.Sumw2()
            heff.Divide(h_gen)
            h_match = rfile.Get('h_pthf_match')
            heff_match = h_match.Clone('h_eff_match')
            heff_match.Sumw2()
            heff_match.Divide(h_gen)
            self._save_hist(h_gen, 'qa/h_pthf_gen.png', 'mc')
            self._save_hist(h_det, 'qa/h_pthf_det.png', 'mc')
            self._save_hist(heff, 'h_eff.png', 'mc')
            self._save_hist(heff_match, 'h_eff_matched.png', 'mc')
            self.hcandeff = heff_match if self.cfg('efficiency.store_matched') else heff
            self.hcandeff.SetDirectory(0)

    #region fitting
    def _fit_mass(self, hist):
        # TODO: check for empty histogram?
        fit_range = self.cfg('mass_fit.range')
        func_sig = TF1('funcSig', self.cfg('mass_fit.func_sig'))
        func_bkg = TF1('funcBkg', self.cfg('mass_fit.func_bkg'))
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
        fit_res = hist.Fit(func_tot, "S", "", fit_range[0], fit_range[1])
        func_sig.SetParameters(func_tot.GetParameters())
        func_bkg.SetParameters(func_tot.GetParameters())
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
                        fit_res, _, func_bkg = self._fit_mass(h_invmass)
                        if fit_res is not None:
                            self.fit_sigma[mcordata][ipt] = fit_res.Parameter(2)
                            self.fit_mean[mcordata][ipt] = fit_res.Parameter(1)
                            self.fit_func_bkg[mcordata][ipt] = func_bkg
                        self._save_hist(h_invmass, f'fit/h_mass_fitted_{ipt}_{mcordata}.png', mcordata)

    #region sidebands
    def _subtract_sideband(self, hist, var, mcordata, ipt):
        """
        Subtract sideband distributions, assuming mass on first axis
        """
        if not hist:
            return None
        assert hist.GetDimension() in [2, 3], 'only 2- and 3-dimensional histograms are supported'
        self._save_hist(hist, f'sideband/h_mass-{var}_{ipt}_{mcordata}.png', mcordata)

        mean = self.fit_mean[mcordata][ipt]
        sigma = self.fit_sigma[mcordata][ipt]

        regions = {
            'signal': (mean - 2 * sigma, mean + 2 * sigma),
            'sideband_left': (mean - 7 * sigma, mean - 4 * sigma),
            'sideband_right': (mean + 4 * sigma, mean + 7 * sigma)
        }

        axis = hist.GetXaxis()
        bins = {key: tuple(map(axis.FindBin, regions[key])) for key in regions}
        limits = {key: (axis.GetBinLowEdge(bins[key][0]), axis.GetBinUpEdge(bins[key][1]))
                  for key in regions}
        self.logger.info('actual sideband regions %s', limits)

        fh = {}
        area = {}
        for region in regions:
            if hist.GetDimension() == 2:
                fh[region] = hist.ProjectionY(f'h_{var}_{region}_{ipt}_{mcordata}', bins[region][0], bins[region][1], "e")
            elif hist.GetDimension() == 3:
                hist.GetXaxis().SetRange(bins[region][0], bins[region][1])
                fh[region] = hist.Project3D('zye').Clone(f'h_{var}_{region}_{ipt}_{mcordata}')
                hist.GetXaxis().SetRange(0, hist.GetXaxis().GetNbins() + 1)
            area[region] = self.fit_func_bkg[mcordata][ipt].Integral(regions[region][0], regions[region][1])
            self._save_hist(fh[region], f'sideband/h_{var}_{region}_{ipt}_{mcordata}.png', mcordata)

        areaNormFactor = area['signal'] / (area['sideband_left'] + area['sideband_right'])

        fh_sideband = self._sum_histos([fh['sideband_left'], fh['sideband_right']], f'h_{var}_sideband_{ipt}_{mcordata}')
        self._save_hist(fh_sideband, f'sideband/h_{var}_sideband_{ipt}_{mcordata}.png', mcordata)

        fh_subtracted = fh['signal'].Clone(f'h_{var}_subtracted_{ipt}_{mcordata}')
        fh_subtracted.Sumw2()
        fh_subtracted.Add(fh_sideband, -areaNormFactor)
        fh_subtracted.Scale(1.0 / 0.954) # TODO: calculate from region
        self._save_hist(fh_subtracted, f'sideband/h_{var}_subtracted_{ipt}_{mcordata}.png', mcordata)

        c = TCanvas()
        fh['signal'].SetLineColor(ROOT.kRed) # pylint: disable=no-member
        fh['signal'].Draw()
        fh_sideband.Scale(areaNormFactor)
        fh_sideband.SetLineColor(ROOT.kCyan) # pylint: disable=no-member
        fh_sideband.Draw("same")
        fh_subtracted.Draw("same")
        fh_subtracted.GetYaxis().SetRangeUser(0., max(fh_subtracted.GetMaximum(), fh['signal'].GetMaximum(), fh_sideband.GetMaximum()))
        self._save_canvas(c, f'sideband/h_{var}_overview_{ipt}_{mcordata}.png', mcordata)

        if self.hcandeff:
            fh_subtracted.Scale(1.0 / self.hcandeff.GetBinContent(ipt + 1))
            self._save_hist(fh_subtracted, f'sideband/h_{var}_subtracted_effscaled_{ipt}_{mcordata}.png', mcordata)
        else:
            self.logger.error('no efficiency correction because of missing efficiency')

        return fh_subtracted

    def subtract_sidebands(self):
        self.logger.info("Running sideband subtraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in self.observables:
                    fh_sub = [self._subtract_sideband(rfile.Get(f'h_mass-{var}_{ipt}'), var, mcordata, ipt)
                              for ipt in range(self.nbins)]
                    fh_sum = self._sum_histos(fh_sub)
                    self._save_hist(fh_sum, f'h_{var}_subtracted_effscaled_{mcordata}.png', mcordata)

    #region signal extraction
    def _extract_signal(self, hmass2, var, mcordata, ipt):
        """
        Extract signal through inv. mass fit in bins of observable
        """
        if not hmass2:
            return None
        self._save_hist(hmass2, f'signalextr/h_mass-{var}_{ipt}_{mcordata}.png', mcordata)
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
        hist.Sumw2()

        range_int = (self.fit_mean[mcordata][ipt] - 3 * self.fit_sigma[mcordata][ipt],
                     self.fit_mean[mcordata][ipt] + 3 * self.fit_sigma[mcordata][ipt])
        for i in range(nbins):
            if hmass2.GetDimension() == 3:
                hmass2.GetYaxis().SetRange(i+1, i+1)
                for j in range(nbins2):
                    hmass2.GetZaxis().SetRange(j+1, j+1)
                    hmass = hmass2.Project3D('x')
                    _, func_sig, _ = self._fit_mass(hmass)
                    self._save_hist(hmass, f'signalextr/h_mass-{var}_fitted_{ipt}_{i}_{j}_{mcordata}.png', mcordata)
                    hist.SetBinContent(i + 1, j +  1, func_sig.Integral(*range_int) / hmass.GetBinWidth(1))
            else:
                hmass = hmass2.ProjectionX(f'h_mass-{var}_{ipt}_proj_{i}', i+1, i+1, "e")
                _, func_sig, _ = self._fit_mass(hmass)
                self._save_hist(hmass, f'signalextr/h_mass-{var}_fitted_{ipt}_{i}_{mcordata}.png', mcordata)
                hist.SetBinContent(i + 1, func_sig.Integral(*range_int) / hmass.GetBinWidth(1))
        if hmass2.GetDimension() == 3:
            hmass2.GetYaxis().SetRange(0, hmass2.GetYaxis().GetNbins() + 1)
            hmass2.GetZaxis().SetRange(0, hmass2.GetYaxis().GetNbins() + 1)
        self._save_hist(hist, f'signalextr/h_{var}_signalextracted_{ipt}_{mcordata}.png', mcordata)

        if self.hcandeff:
            hist.Scale(1.0/self.hcandeff.GetBinContent(ipt + 1))
            self._save_hist(hist, f'signalextr/h_{var}_signalextracted_effscaled_{ipt}_{mcordata}.png', mcordata)
        else:
            self.logger.error('no efficiency correction because of missing efficiency')

        return hist

    def extract_signals(self):
        self.logger.info("Running signal extraction")
        for mcordata in ['mc', 'data']:
            rfilename = self.n_filemass_mc if mcordata == "mc" else self.n_filemass
            with TFile(rfilename) as rfile:
                for var in self.observables:
                    hsignals = [self._extract_signal(rfile.Get(f'h_mass-{var}_{ipt}'), var, mcordata, ipt)
                                for ipt in range(self.nbins)]
                    hist_effscaled = self._sum_histos(hsignals)
                    self._save_hist(hist_effscaled, f'h_{var}_signalextracted_effscaled_{mcordata}.png', mcordata)
