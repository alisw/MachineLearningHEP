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

import copy
import functools
import itertools
import math

import numpy as np
import pandas as pd
from ROOT import TH1F, TFile

from machine_learning_hep.processer import Processer
from machine_learning_hep.utilities import dfquery, read_df
from machine_learning_hep.utils.hist import bin_array, create_hist, fill_hist, get_axis


# pylint: disable=too-many-instance-attributes
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

        # bins: 2d array [[low, high], ...]
        self.bins_skimming = np.array(list(zip(self.lpt_anbinmin, self.lpt_anbinmax)), 'd') # TODO: replace with cfg
        self.bins_analysis = np.array(list(zip(self.lpt_finbinmin, self.lpt_finbinmax)), 'd')

        # skimming bins in overlap with the analysis range
        self.active_bins_skim = [
            iskim for iskim, ptrange in enumerate(self.bins_skimming)
            if ptrange[0] < max(self.bins_analysis[:,1]) and ptrange[1] > min(self.bins_analysis[:,0])]
        self.logger.info('Using skimming bins: %s', self.active_bins_skim)

        # binarray: array of bin edges as double (passable to ROOT)
        limits_mass = datap["analysis"][self.typean]["mass_fit_lim"]
        binwidth_mass = datap["analysis"][self.typean]["bin_width"]
        nbins_mass = int(round((limits_mass[1] - limits_mass[0]) / binwidth_mass))
        self.binarray_mass = bin_array(nbins_mass, limits_mass[0], limits_mass[1])
        self.binarray_ptjet = np.asarray(self.cfg('bins_ptjet'), 'd')
        self.binarray_pthf = np.asarray(self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:], 'd')
        self.binarrays_obs = {}
        for obs in self.cfg('observables'):
            var = obs.split('-')
            for v in var:
                if v in self.binarrays_obs:
                    continue
                if binning := self.cfg(f'observables.{v}.bins_var'):
                    self.binarrays_obs[v] = np.asarray(binning, 'd')
                elif binning := self.cfg(f'observables.{v}.bins_fix'):
                    self.binarrays_obs[v] = bin_array(*binning)
                else:
                    self.logger.error('no binning specified for %s, using defaults', v)
                    self.binarrays_obs[v] = bin_array(10, 0., 1.)
        self.binarrays_obs['fPt'] = self.binarray_pthf
        self.binarrays_obs['fPtJet'] = self.binarray_ptjet


    # region observables
    # pylint: disable=invalid-name
    def _verify_variables(self, dfi):
        """
        Explicit (slow) implementation, use for reference/validation only
        """
        df = dfi.copy(deep=True)
        df['rg'] = -.1
        df['nsd'] = -1.0
        df['zg'] = -.1
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
        for var in ['zg', 'nsd', 'rg']:
            if np.allclose(dfi[var], df[var]):
                self.logger.info('%s check ok', var)
            else:
                self.logger.error('%s check failed', var)
                mask = np.isclose(dfi[var], df[var])
                print(df[~mask][var], flush=True)
                print(dfi[~mask][var], flush=True)


    def _calculate_variables(self, df, verify=False): # pylint: disable=invalid-name
        self.logger.info('calculating variables')
        if len(df) == 0:
            return df
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
        df[df['zpar'] == 1.]['zpar'] = .99999 # move 1 to last bin
        df['nsub21'] = df.fNSub2 / df.fNSub1

        self.logger.debug('zg')
        df['zg_array'] = np.array(.5 - abs(df.fPtSubLeading / (df.fPtLeading + df.fPtSubLeading) - .5))
        zcut = self.cfg('zcut', .1)
        df['zg'] = df['zg_array'].apply((lambda ar: next((zg for zg in ar if zg >= zcut), -.1)))
        df['rg'] = df[['zg_array', 'fTheta']].apply(
            (lambda ar: next((rg for (zg, rg) in zip(ar.zg_array, ar.fTheta) if zg >= zcut), -.1)), axis=1)
        df['nsd'] = df['zg_array'].apply((lambda ar: len([zg for zg in ar if zg >= zcut])))

        self.logger.debug('Lund')
        df['lnkt'] = df[['fPtSubLeading', 'fTheta']].apply(
            (lambda ar: np.log(ar.fPtSubLeading * np.sin(ar.fTheta))), axis=1)
        df['lntheta'] = df['fTheta'].apply(lambda x: -np.log(x))
        # df['lntheta'] = np.array(-np.log(df.fTheta))
        self.logger.debug('done')
        if verify:
            self._verify_variables(df)
        return df


    def split_df(self, dfi, frac):
        '''split data frame based on df number'''
        # dfa = dfi.split(frac=frac, random_state=1234)
        # return dfa, dfi.drop(dfa.index)
        mask = (dfi.index.get_level_values(0) % 100) < frac * 100
        return dfi[mask], dfi[~mask]


    # region histomass
    def process_histomass_single(self, index):
        self.logger.info('Processing (histomass) %s', self.l_evtorig[index])

        with TFile.Open(self.l_histomass[index], "recreate") as _:
            dfevtorig = read_df(self.l_evtorig[index])
            histonorm = TH1F("histonorm", "histonorm", 2, 0, 2)
            histonorm.SetBinContent(1, len(dfquery(dfevtorig, self.s_evtsel)))
            dfcollcnt = read_df(self.l_collcnt[index])
            collcnt = functools.reduce(
                lambda x,y: float(x)+float(y), (ar[0] for ar in dfcollcnt['fReadSelectedCounts']))
            self.logger.info('sampled %g collisions', collcnt)
            histonorm.SetBinContent(2, collcnt)
            get_axis(histonorm, 0).SetBinLabel(1, 'N_{evt}')
            get_axis(histonorm, 0).SetBinLabel(2, 'N_{coll}')
            histonorm.Write()

            df = pd.concat(read_df(self.mptfiles_recosk[bin][index]) for bin in self.active_bins_skim)

            h = create_hist(
                'h_mass-ptjet-pthf',
                ';M (GeV/#it{c}^{2});p_{T}^{jet} (GeV/#it{c});p_{T}^{HF} (GeV/#it{c})',
                self.binarray_mass, self.binarray_ptjet, self.binarray_pthf)
            fill_hist(h, df[['fM', 'fJetPt', 'fPt']], write=True)

            for sel_name, sel_spec in self.cfg('data_selections', {}).items():
                if sel_spec['level'] == self.mcordata:
                    df_sel = dfquery(df, sel_spec['query'])
                    h = create_hist(
                        f'h_mass-ptjet-pthf_{sel_name}',
                        ';M (GeV/#it{c}^{2});p_{T}^{jet} (GeV/#it{c});p_{T}^{HF} (GeV/#it{c})',
                        self.binarray_mass, self.binarray_ptjet, self.binarray_pthf)
                    fill_hist(h, df_sel[['fM', 'fJetPt', 'fPt']], write=True)

            if self.mcordata == 'mc':
                df, _ = self.split_df(df, self.cfg('frac_mcana', .2))
                if len(df) == 0:
                    return
                self.logger.debug('MC det: %s', df.index.get_level_values(0).unique())
                if f := self.cfg('closure.exclude_feeddown_det'):
                    dfquery(df, f, inplace=True)
                if f := self.cfg('closure.filter_reflections'):
                    dfquery(df, f, inplace=True)
                if self.cfg('closure.use_matched'):
                    if idx := self.cfg('efficiency.index_match'):
                        df['idx_match'] = df[idx].apply(lambda ar: ar[0] if len(ar) > 0 else -1)
                        dfquery(df, 'idx_match >= 0', inplace=True)

            self._calculate_variables(df)

            for obs, spec in self.cfg('observables', {}).items():
                self.logger.debug('preparing histograms for %s', obs)
                var = obs.split('-')
                if not all(v in df for v in var):
                    self.logger.error('dataframe does not contain %s', var)
                    continue
                h = create_hist(
                    f'h_mass-ptjet-pthf-{obs}',
                    f';M (GeV/#it{{c}}^{{2}});p_{{T}}^{{jet}} (GeV/#it{{c}});p_{{T}}^{{HF}} (GeV/#it{{c}});{obs}',
                    self.binarray_mass, self.binarray_ptjet, self.binarray_pthf, *[self.binarrays_obs[v] for v in var])
                for i, v in enumerate(var):
                    get_axis(h, 3+i).SetTitle(self.cfg(f'observables.{v}.label', v))

                fill_hist(h, df[['fM', 'fJetPt', 'fPt', *var]], arraycols=spec.get('arraycols', None), write=True)

    # TODO:
    # - binning variations (separate ranges for MC and data)
    # - priors (reweight response matrix)

    # region efficiency
    # pylint: disable=too-many-branches,too-many-statements
    def process_efficiency_single(self, index):
        self.logger.info('Processing (efficiency) %s', self.l_evtorig[index])

        cats = ['pr', 'np']
        levels = ['gen', 'det']
        cuts = ['nocuts', 'cut']
        observables = self.cfg('observables', [])
        observables.update({'fPt': {}})
        h_eff = {(cat, level): create_hist(f'h_ptjet-pthf_{cat}_{level}',
                                           ';p_{T}^{jet} (GeV/#it{c});p_{T}^{HF} (GeV/#it{c})',
                                           self.binarray_ptjet, self.binarray_pthf)
                                           for cat in cats for level in (levels + ['det_gencuts'])}
        # TODO: extend to multi-dimensional observables
        # TODO: allow different binnings for gen and det
        h_response = {
            (cat, var): create_hist(
                f'h_response_{cat}_{var}',
                f";p_{{T}}^{{jet}} (GeV/#it{{c}});{var};p_{{T}}^{{jet}} (GeV/#it{{c}});{var};p_{{T}} (GeV/#it{{c}})",
                self.binarray_ptjet, self.binarrays_obs[var],
                self.binarray_ptjet, self.binarrays_obs[var],
                self.binarray_pthf)
            for (cat, var) in itertools.product(cats, observables)
            if not '-' in var}
        # TODO: derive bins from response histogram
        h_effkine = {(cat, level, cut, var):
                        create_hist(f'h_effkine_{cat}_{level}_{cut}_{var}',
                                    f";p_{{T}}^{{jet}} (GeV/#it{{c}});{var}",
                                    self.binarray_ptjet, self.binarrays_obs[var])
                        for var, level, cat, cut in itertools.product(observables, levels, cats, cuts)
                        if not '-' in var}
        h_mctruth = {
            (cat, var): create_hist(
                f'h_ptjet-pthf-{var}_{cat}_gen',
                f";p_{{T}}^{{jet}} (GeV/#it{{c}});p_{{T}}^{{HF}} (GeV/#it{{c}});{var}",
                self.binarray_ptjet, self.binarray_pthf, self.binarrays_obs[var])
            for (cat, var) in itertools.product(cats, observables)
            if not '-' in var}

        # create partial versions for closure testing
        h_effkine_frac = copy.deepcopy(h_effkine)
        h_response_frac = copy.deepcopy(h_response)
        for hist in itertools.chain(h_effkine_frac.values(), h_response_frac.values()):
            hist.SetName(hist.GetName() + '_frac')

        with TFile.Open(self.l_histoeff[index], "recreate") as rfile:
            # TODO: avoid hard-coding values here (check if restriction is needed at all)
            cols = ['ismcprompt', 'ismcsignal', 'ismcfd', 'fPt', 'fEta', 'fPhi', 'fJetPt', 'fJetEta', 'fJetPhi',
                    'fPtLeading', 'fPtSubLeading', 'fTheta', 'fNSub2DR', 'fNSub1', 'fNSub2']

            # read generator level
            dfgen_orig = pd.concat(read_df(self.mptfiles_gensk[bin][index], columns=cols)
                                   for bin in self.active_bins_skim)
            df = self._calculate_variables(dfgen_orig)
            df = df.rename(lambda name: name + '_gen', axis=1)
            dfgen = {'pr': df.loc[(df.ismcsignal_gen == 1) & (df.ismcprompt_gen == 1)],
                     'np': df.loc[(df.ismcsignal_gen == 1) & (df.ismcfd_gen == 1)]}

            # read detector level
            cols.extend(self.cfg('efficiency.extra_cols', []))
            if idx := self.cfg('efficiency.index_match'):
                cols.append(idx)
            df = pd.concat(read_df(self.mptfiles_recosk[bin][index], columns=cols)
                           for bin in self.active_bins_skim)
            dfquery(df, self.cfg('efficiency.filter_det'), inplace=True)
            if idx := self.cfg('efficiency.index_match'):
                df['idx_match'] = df[idx].apply(lambda ar: ar[0] if len(ar) > 0 else -1)
            else:
                self.logger.warning('No matching criterion specified, cannot match det and gen')
            df = self._calculate_variables(df)
            dfdet = {'pr': df.loc[(df.ismcsignal == 1) & (df.ismcprompt == 1)],
                     'np': df.loc[(df.ismcsignal == 1) & (df.ismcfd == 1)]}

            dfmatch = {cat: pd.merge(dfdet[cat], dfgen[cat], left_on=['df', 'idx_match'], right_index=True)
                        for cat in cats if 'idx_match' in dfdet[cat]}

            for cat in cats:
                fill_hist(h_eff[(cat, 'gen')], dfgen[cat][['fJetPt_gen', 'fPt_gen']])
                # TODO: eventually use the full det-level sample (requires extension of gen-level folding)
                # fill_hist(h_eff[(cat, 'det')], dfdet[cat][['fJetPt', 'fPt']])
                if cat in dfmatch and dfmatch[cat] is not None:
                    df = dfmatch[cat]
                    fill_hist(h_eff[(cat, 'det')], df[['fJetPt', 'fPt']])
                    df = df.loc[(df.fJetPt_gen >= min(self.binarray_ptjet)) &
                                (df.fJetPt_gen < max(self.binarray_ptjet))]
                    df = df.loc[(df.fPt_gen >= min(self.binarray_pthf)) &
                                (df.fPt_gen < max(self.binarray_pthf))]
                    fill_hist(h_eff[(cat, 'det_gencuts')], df[['fJetPt', 'fPt']])
                else:
                    self.logger.error('No matching, could not fill matched detector-level histograms')

            for var, cat in itertools.product(observables, cats):
                # TODO: add support for more complex observables
                if '-' in var or self.cfg(f'observables.{var}.arraycols'):
                    continue
                if self.cfg('closure.use_matched'):
                    self.logger.info('using matched for truth')
                    df_mcana, _ = self.split_df(dfmatch[cat], self.cfg('frac_mcana', .2))
                else:
                    df_mcana, _ = self.split_df(dfgen[cat], self.cfg('frac_mcana', .2))
                if f := self.cfg('closure.exclude_feeddown_gen'):
                    self.logger.info('excluding feeddown gen')
                    dfquery(df_mcana, f, inplace=True)
                fill_hist(h_mctruth[(cat, var)], df_mcana[['fJetPt_gen', 'fPt_gen', f'{var}_gen']])

                if cat in dfmatch and dfmatch[cat] is not None:
                    self._prepare_response(dfmatch[cat], h_effkine, h_response, cat, var)
                    f = self.cfg('frac_mcana', .2)
                    _, df_mccorr = self.split_df(dfmatch[cat], f if f < 1. else 0.)
                    self._prepare_response(df_mccorr, h_effkine_frac, h_response_frac, cat, var)

            for name, obj in itertools.chain(h_eff.items(), h_effkine.items(), h_response.items(),
                                             h_effkine_frac.items(), h_response_frac.items(), h_mctruth.items()):
                try:
                    rfile.WriteObject(obj, obj.GetName())
                except Exception as ex: # pylint: disable=broad-exception-caught
                    self.logger.error('Writing of <%s> (%s) failed: %s', name, str(obj), str(ex))

    def _prepare_response(self, dfi, h_effkine, h_response, cat, var):
        ptjet_min = min(self.binarray_ptjet)
        ptjet_max = max(self.binarray_ptjet)
        var_min = min(self.binarrays_obs[var])
        var_max = max(self.binarrays_obs[var])

        df = dfi
        # TODO: derive cuts from definition of response histogram
        # TODO: is the first cut really needed? taken care of by under-/overflow bins
        df = df.loc[(df.fJetPt >= ptjet_min) & (df.fJetPt < ptjet_max) &
                    (df[var] >= var_min) & (df[var] < var_max)]
        fill_hist(h_effkine[(cat, 'det', 'nocuts', var)], df[['fJetPt', var]])
        df = df.loc[(df.fJetPt_gen >= ptjet_min) & (df.fJetPt_gen < ptjet_max) &
                    (df[f'{var}_gen'] >= var_min) & (df[f'{var}_gen'] < var_max)]
        fill_hist(h_effkine[(cat, 'det', 'cut', var)], df[['fJetPt', var]])

        fill_hist(h_response[(cat, var)], df[['fJetPt', f'{var}', 'fJetPt_gen', f'{var}_gen', 'fPt']])

        df = dfi
        df = df.loc[(df.fJetPt_gen >= ptjet_min) & (df.fJetPt_gen < ptjet_max) &
                    (df[f'{var}_gen'] >= var_min) & (df[f'{var}_gen'] < var_max)]
        fill_hist(h_effkine[(cat, 'gen', 'nocuts', var)], df[['fJetPt_gen', f'{var}_gen']])
        df = df.loc[(df.fJetPt >= ptjet_min) & (df.fJetPt < ptjet_max) &
                    (df[f'{var}'] >= var_min) & (df[f'{var}'] < var_max)]
        fill_hist(h_effkine[(cat, 'gen', 'cut', var)], df[['fJetPt_gen', f'{var}_gen']])
