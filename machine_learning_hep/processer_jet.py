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
import math
import time

import numpy as np
import pandas as pd
import ROOT
from ROOT import TH1F, TFile

from machine_learning_hep.processer import Processer
from machine_learning_hep.utilities import dfquery, fill_response, read_df
from machine_learning_hep.utilities_hist import bin_spec, create_hist, fill_hist


class ProcesserJets(Processer):
    species = "processer"

    def __init__(self, case, datap, run_param, mcordata, p_maxfiles, # pylint: disable=too-many-arguments
                d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                d_results, typean, runlisttrigger, d_mcreweights):
        super().__init__(case, datap, run_param, mcordata, p_maxfiles,
                        d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                        p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                        p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                        d_results, typean, runlisttrigger, d_mcreweights)
        self.logger.info("initialized processer for HF jets")

        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]

        self.bins_skimming = list(zip(self.lpt_anbinmin, self.lpt_anbinmax))
        self.bins_analysis = np.array(list(zip(self.lpt_finbinmin, self.lpt_finbinmax)))

        self.p_bin_width = datap["analysis"][self.typean]["bin_width"]
        self.p_mass_fit_lim = datap["analysis"][self.typean]["mass_fit_lim"]
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / self.p_bin_width))

    #region observables
    def calculate_zg(self, df): # pylint: disable=invalid-name
        """
        Explicit implementation, for reference/validation only
        """
        start = time.time()
        df['zg_array'] = np.array(.5 - abs(df.fPtSubLeading / (df.fPtLeading + df.fPtSubLeading) - .5))
        df['zg_fast'] = df['zg_array'].apply((lambda ar: next((zg for zg in ar if zg >= .1), -1.)))
        df['rg_fast'] = df[['zg_array', 'fTheta']].apply(
            (lambda ar: next((rg for (zg, rg) in zip(ar.zg_array, ar.fTheta) if zg >= .1), -1.)), axis=1)
        df['nsd_fast'] = df['zg_array'].apply((lambda ar: len([zg for zg in ar if zg >= .1])))
        self.logger.debug('fast done in %.2g s', time.time() - start)

        start = time.time()
        df['rg'] = -1.0
        df['nsd'] = -1.0
        df['zg'] = -1.0
        for idx, row in df.iterrows():
            isSoftDropped = False
            nsd = 0
            for zg, theta in zip(row['zg_array'], row['fTheta']):
                if zg >= self.cfg('zcut', .1):
                    if not isSoftDropped:
                        df.loc[idx, 'zg'] = zg
                        df.loc[idx, 'rg'] = theta
                        isSoftDropped = True
                    nsd += 1
            df.loc[idx, 'nsd'] = nsd
        self.logger.debug('slow done in %.2g s', time.time() - start)
        if np.allclose(df.nsd, df.nsd_fast):
            self.logger.info('nsd all close')
        else:
            self.logger.error('nsd not all close')
        if np.allclose(df.zg, df.zg_fast):
            self.logger.info('zg all close')
        else:
            self.logger.error('zg not all close')
        if np.allclose(df.rg, df.rg_fast):
            self.logger.info('rg all close')
        else:
            self.logger.error('rg not all close')


    def _calculate_variables(self, df): # pylint: disable=invalid-name
        self.logger.info('calculating variables')
        df['dr'] = np.sqrt((df.fJetEta - df.fEta)**2 + ((df.fJetPhi - df.fPhi + math.pi) % math.tau - math.pi)**2)
        df['jetPx'] = df.fJetPt * np.cos(df.fJetPhi)
        df['jetPy'] = df.fJetPt * np.sin(df.fJetPhi)
        df['jetPz'] = df.fJetPt * np.sinh(df.fJetEta)
        df['hfPx'] = df.fPt * np.cos(df.fPhi)
        df['hfPy'] = df.fPt * np.sin(df.fPhi)
        df['hfPz'] = df.fPt * np.sinh(df.fEta)
        df['zpar_num'] = df.jetPx * df.hfPx + df.jetPy * df.hfPy + df.jetPz * df.hfPz
        df['zpar_den'] = df.jetPx * df.jetPx + df.jetPy * df.jetPy + df.jetPz * df.jetPz
        df['zpar'] = df.zpar_num / df.zpar_den

        self.logger.debug('zg')
        df['zg_array'] = np.array(.5 - abs(df.fPtSubLeading / (df.fPtLeading + df.fPtSubLeading) - .5))
        zcut = self.cfg('zcut', .1)
        df['zg'] = df['zg_array'].apply((lambda ar: next((zg for zg in ar if zg >= zcut), -1.)))
        df['rg'] = df[['zg_array', 'fTheta']].apply(
            (lambda ar: next((rg for (zg, rg) in zip(ar.zg_array, ar.fTheta) if zg >= zcut), -1.)), axis=1)
        df['nsd'] = df['zg_array'].apply((lambda ar: len([zg for zg in ar if zg >= zcut])))

        self.logger.debug('Lund')
        df['lnkt'] = df[['fPtSubLeading', 'fTheta']].apply(
            (lambda ar: np.log(ar.fPtSubLeading * np.sin(ar.fTheta))), axis=1)
        df['lntheta'] = df['fTheta'].apply(lambda x: -np.log(x))
        # df['lntheta'] = np.array(-np.log(df.fTheta))
        self.logger.debug('done')
        return df


    #region histomass
    def process_histomass_single(self, index): # pylint: disable=too-many-statements
        self.logger.info('Processing (histomass) %s', self.l_evtorig[index])

        with TFile.Open(self.l_histomass[index], "recreate") as _:
            dfevtorig = read_df(self.l_evtorig[index])
            histonorm = TH1F("histonorm", "histonorm", 1, 0, 1)
            histonorm.SetBinContent(1, len(dfevtorig.query(self.s_evtsel)))
            histonorm.Write()

            bins_skim = [iskim for iskim, ptrange in enumerate(self.bins_skimming)
                         if ptrange[0] < max(self.bins_analysis[:,1]) and ptrange[1] > min(self.bins_analysis[:,0])]
            self.logger.info('Using skimming bins: %s', bins_skim)
            bins_ptjet = np.asarray(self.cfg('bins_ptjet'), 'd')
            bins_mass = bin_spec(self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            bins_ana = np.asarray(self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:], 'd')

            # read all skimmed bins which overlap with the analysis range
            df = pd.concat(read_df(self.mptfiles_recosk[bin][index]) for bin in bins_skim)
            df = df.loc[(df.fJetPt >= min(bins_ptjet)) & (df.fJetPt < max(bins_ptjet))]
            df = df.loc[(df.fPt >= min(self.bins_analysis[:,0])) & (df.fPt < max(self.bins_analysis[:,1]))]
            self._calculate_variables(df)

            h = create_hist(
                'h_mass-ptjet-pthf',
                ';M (GeV/#it{c}^{2});p_{T}^{jet} (GeV/#it{c});p_{T}^{HF} (GeV/#it{c})',
                bins_mass, bins_ptjet, bins_ana)
            fill_hist(h, df[['fM', 'fJetPt', 'fPt']], write=True)

            for var, spec in self.cfg('observables', {}).items():
                self.logger.info('preparing histograms for %s', var)
                if '-' in var or 'arraycols' in spec:
                    self.logger.error('Writing for %s not yet available', var)
                    continue
                if binning := spec.get('bins_fix'):
                    bins_obs = bin_spec(*binning)
                h = create_hist(
                    f'h_mass-ptjet-pthf-{var}',
                    f';M (GeV/#it{{c}}^{{2}});p_{{T}}^{{jet}} (GeV/#it{{c}});p_{{T}}^{{HF}} (GeV/#it{{c}});{var}',
                    bins_mass, bins_ptjet, bins_ana, bins_obs)
                h.GetAxis(3).SetTitle(spec.get('label', var)) # TODO: why is this not derived from the title string?
                fill_hist(h, df[['fM', 'fJetPt', 'fPt', var]], write=True)


    #region efficiency
    def process_efficiency_single(self, index): # pylint: disable=too-many-branches,too-many-statements
        self.logger.info('Processing (efficiency) %s', self.l_evtorig[index])

        with TFile.Open(self.l_histoeff[index], "recreate") as _:
            ptbins = np.asarray(self.lpt_finbinmin + [self.lpt_finbinmax[-1]], 'd')

            cats = ['pr', 'np']
            levels = ['gen', 'det']
            cuts = ['nocuts', 'cut']
            observables = [var for var, spec in self.cfg('observables', {}).items()
                           if '-' not in var and 'arraycols' not in spec]
            self.logger.info('Using observables %s', observables)
            bins_skim = [iskim for iskim, ptrange in enumerate(self.bins_skimming)
                         if ptrange[0] < max(self.bins_analysis[:,1]) and ptrange[1] > min(self.bins_analysis[:,0])]
            self.logger.info('Using skimming bins: %s', bins_skim)
            bins_ptjet = np.asarray(self.cfg('bins_ptjet'), 'd')
            bins_obs = { var: bin_spec(*self.cfg(f'observables.{var}.bins_fix')) for var in observables}

            h_eff = {(cat, level): TH1F(f'h_pthf_{cat}_{level}', ";p_{T} (GeV/#it{c})", len(ptbins)-1, ptbins)
                     for cat in cats for level in levels}
            h_effkine = {(cat, level, cut, var):
                         create_hist(f'h_effkine_{cat}_{level}_{cut}_{var}',
                                     f";p_{{T}}^{{jet}} (GeV/#it{{c}});{var}",
                                     bins_ptjet, bins_obs[var])
                         for var, level, cat, cut in itertools.product(observables, levels, cats, cuts)}
            response_matrix = {
                (cat, var): ROOT.RooUnfoldResponse(h_effkine[(cat, 'det', 'nocuts', var)],
                                                   h_effkine[(cat, 'gen', 'nocuts', var)])
                for (cat, var) in itertools.product(cats, observables)}

            # read all skimmed bins which overlap with the analysis range
            cols = ['ismcprompt', 'fPt', 'fEta', 'fPhi', 'fJetPt', 'fJetEta', 'fJetPhi',
                    'fPtLeading', 'fPtSubLeading', 'fTheta']
            df = pd.concat(read_df(self.mptfiles_gensk[bin][index], columns=cols) for bin in bins_skim)
            df = df.loc[(df.fJetPt >= min(bins_ptjet)) & (df.fJetPt < max(bins_ptjet))]
            df = df.loc[(df.fPt >= min(self.bins_analysis[:,0])) & (df.fPt < max(self.bins_analysis[:,1]))]
            self._calculate_variables(df)
            df.rename(lambda name: name + '_gen', axis=1, inplace=True)
            dfgen = {'pr': df.loc[df.ismcprompt_gen == 1], 'np': df.loc[df.ismcprompt_gen == 0]}

            cols.extend(self.cfg('efficiency.extra_cols', []))
            if idx := self.cfg('efficiency.index_match'):
                cols.append(idx)
            df = pd.concat(read_df(self.mptfiles_recosk[bin][index], columns=cols) for bin in bins_skim)
            df = df.loc[(df.fJetPt >= min(bins_ptjet)) & (df.fJetPt < max(bins_ptjet))]
            df = df.loc[(df.fPt >= min(self.bins_analysis[:,0])) & (df.fPt < max(self.bins_analysis[:,1]))]
            dfquery(df, self.cfg('efficiency.filter_det'), inplace=True)
            df = self._calculate_variables(df)
            dfdet = {'pr': df.loc[df.ismcprompt == 1], 'np': df.loc[df.ismcprompt == 0]}

            if idx := self.cfg('efficiency.index_match'):
                for cat in cats:
                    dfdet[cat]['idx_match'] = dfdet[cat][idx].apply(lambda ar: ar[0] if len(ar) > 0 else -1)
                dfmatch = {cat: pd.merge(dfdet[cat], dfgen[cat], left_on=['df', 'idx_match'], right_index=True)
                           for cat in cats}
            else:
                self.logger.warning('No matching criterion specified, cannot calculate matched efficiency')
                dfmatch = {cat: None for cat in cats}

            for cat in cats:
                fill_hist(h_eff[(cat, 'gen')], dfgen[cat]['fPt_gen'])
                if dfmatch[cat] is not None:
                    fill_hist(h_eff[(cat, 'det')], dfmatch[cat]['fPt_gen'])
                else:
                    self.logger.error('No matching, using unmatched detector level for efficiency')
                    fill_hist(h_eff[(cat, 'det')], dfdet[cat]['fPt'])

            ptjet_min = min(bins_ptjet)
            ptjet_max = max(bins_ptjet)
            if dfmatch[cat] is not None:
                for var, cat in itertools.product(observables, cats):
                    var_min = min(bins_obs[var])
                    var_max = max(bins_obs[var])

                    df = dfmatch[cat]
                    df = df.loc[(df.fJetPt >= ptjet_min) & (df.fJetPt < ptjet_max) &
                                (df[var] > var_min) & (df[var] < var_max)]
                    fill_hist(h_effkine[(cat, 'det', 'nocuts', var)], df[['fJetPt', var]])
                    df = df.loc[(df.fJetPt_gen >= ptjet_min) & (df.fJetPt_gen < ptjet_max) &
                                (df[f'{var}_gen'] > var_min) & (df[f'{var}_gen'] < var_max)]
                    fill_hist(h_effkine[(cat, 'det', 'cut', var)], df[['fJetPt', var]])

                    fill_response(response_matrix[(cat, var)], df[['fJetPt', f'{var}', 'fJetPt_gen', f'{var}_gen']])

                    df = dfmatch[cat]
                    df = df.loc[(df.fJetPt_gen >= ptjet_min) & (df.fJetPt_gen < ptjet_max) &
                                (df[f'{var}_gen'] > var_min) & (df[f'{var}_gen'] < var_max)]
                    fill_hist(h_effkine[(cat, 'gen', 'nocuts', var)], df[['fJetPt_gen', f'{var}_gen']])
                    df = df.loc[(df.fJetPt >= ptjet_min) & (df.fJetPt < ptjet_max) &
                                (df[f'{var}'] > var_min) & (df[f'{var}'] < var_max)]
                    fill_hist(h_effkine[(cat, 'gen', 'cut', var)], df[['fJetPt_gen', f'{var}_gen']])

            for name, obj in itertools.chain(h_eff.items(), h_effkine.items(), response_matrix.items()):
                try:
                    obj.Write()
                except Exception as ex: # pylint: disable=broad-exception-caught
                    self.logger.error('Writing of <%s> (%s) failed: %s', name, str(obj), str(ex))
