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

from array import array
import itertools
import math # pylint: disable=unused-import
import time
import numpy as np
import pandas as pd
import ROOT
from ROOT import TFile, TH1F, TH2F, TH3F
from machine_learning_hep.processer import Processer
from machine_learning_hep.utilities import dfquery, read_df, fill_response
from machine_learning_hep.utilities_hist import create_hist, fill_hist

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
        self.bins_analysis = list(zip(self.lpt_finbinmin, self.lpt_finbinmax))

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

    def process_calculate_variables(self, df): # pylint: disable=invalid-name
        df.eval('dr = sqrt((fJetEta - fEta)**2 + ((fJetPhi - fPhi + @math.pi) % @math.tau - @math.pi)**2)',
                inplace=True)
        df.eval('jetPx = fJetPt * cos(fJetPhi)', inplace=True)
        df.eval('jetPy = fJetPt * sin(fJetPhi)', inplace=True)
        df.eval('jetPz = fJetPt * sinh(fJetEta)', inplace=True)
        df.eval('hfPx = fPt * cos(fPhi)', inplace=True)
        df.eval('hfPy = fPt * sin(fPhi)', inplace=True)
        df.eval('hfPz = fPt * sinh(fEta)', inplace=True)
        df.eval('zpar_num = jetPx * hfPx + jetPy * hfPy + jetPz * hfPz', inplace=True)
        df.eval('zpar_den = jetPx * jetPx + jetPy * jetPy + jetPz * jetPz', inplace=True)
        df.eval('zpar = zpar_num / zpar_den', inplace=True)
        df['zg_array'] = np.array(.5 - abs(df.fPtSubLeading / (df.fPtLeading + df.fPtSubLeading) - .5))
        zcut = self.cfg('zcut', .1)
        df['zg'] = df['zg_array'].apply((lambda ar: next((zg for zg in ar if zg >= zcut), -1.)))
        df['rg'] = df[['zg_array', 'fTheta']].apply(
            (lambda ar: next((rg for (zg, rg) in zip(ar.zg_array, ar.fTheta) if zg >= zcut), -1.)), axis=1)
        df['nsd'] = df['zg_array'].apply((lambda ar: len([zg for zg in ar if zg >= zcut])))
        df['lnkt'] = df[['fPtSubLeading', 'fTheta']].apply(
            (lambda ar: np.log(ar.fPtSubLeading * np.sin(ar.fTheta))), axis=1)
        df['lntheta'] = df[['fTheta']].apply((lambda ar: -np.log(ar.fTheta)), axis=1)
        # self.calculate_zg(df)
        return df

    #region histomass
    def process_histomass_single(self, index): # pylint: disable=too-many-statements
        self.logger.info('processing histomass single')

        myfile = TFile.Open(self.l_histomass[index], "recreate")
        myfile.cd()

        dfevtorig = read_df(self.l_evtorig[index])
        dfevtevtsel = dfevtorig.query(self.s_evtsel)
        neventsafterevtsel = len(dfevtevtsel)
        histonorm = TH1F("histonorm", "histonorm", 1, 0, 1)
        histonorm.SetBinContent(1, neventsafterevtsel)
        histonorm.Write()

        mass_binning = (self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])

        for ipt, (pt_min, pt_max) in enumerate(self.bins_analysis):
            # identify skimming bins which overlap with analysis interval
            bins = [iskim for iskim, ptrange in enumerate(self.bins_skimming)
                    if ptrange[0] < pt_max and ptrange[1] > pt_min]

            df = pd.concat(read_df(self.mptfiles_recosk[bin_id][index]) for bin_id in bins)
            df.query(f'fPt >= {pt_min} and fPt < {pt_max}', inplace=True)
            df.query('fJetPt > 7. and fJetPt < 15.', inplace=True) # TODO: take from DB
            if df.empty:
                self.logger.warning('No data for bin {ipt}')
                continue
            df = self.process_calculate_variables(df)

            self.logger.info('preparing histograms for bin %d', ipt)
            h_invmass_all = TH1F(f'h_mass_{ipt}', "Inv. mass;M (GeV/#it{c}^{2})", *mass_binning)
            fill_hist(h_invmass_all, df.fM, write=True)

            h_candpt_all = TH1F(f'h_ptcand_{ipt}', ";p_{T} (GeV/#it{c})",
                                self.p_num_bins, 0., 50.)
            fill_hist(h_candpt_all, df.fPt, write=True)

            h_jetpt_all = TH1F(f'h_ptjet_{ipt}', ";p_{T} (GeV/#it{c})",
                               self.p_num_bins, 0., 50.)
            fill_hist(h_jetpt_all, df.fJetPt, write=True)

            ## substructure
            h_zg = TH1F(f'h_zg_{ipt}', ";z_{g}",
                        10, 0.0, 1.0)
            fill_hist(h_zg, df.zg, write=True)

            h_nsd = TH1F(f'h_nsd_{ipt}', ";N_{sd}",
                         10, 0.0, 10.0)
            fill_hist(h_nsd, df.nsd, write=True)

            h_rg = TH1F(f'h_rg_{ipt}', ";R_{g}",
                        100, 0.0, 1.0)
            fill_hist(h_rg, df.rg, write=True)

            h_zpar = TH1F(f'h_zpar_{ipt}', ";z_{#parallel}",
                          100, 0.0, 1.0)
            fill_hist(h_zpar, df.zpar, write=True)

            h_dr = TH1F(f'h_dr_{ipt}', ";#Deltar",
                        10, 0.0, 1.0)
            fill_hist(h_dr, df.dr, write=True)

            h = TH1F(f'h_lntheta_{ipt}', ";-ln(#theta)",
                     100, 0.0, 5.0)
            fill_hist(h, df.lntheta, arraycols=True, write=True)

            h = TH1F(f'h_lnkt_{ipt}', ";ln k_{T}",
                     100, -8., 2.)
            fill_hist(h, df.lnkt, arraycols=True, write=True)

            h = TH2F(f'h_mass-zg_{ipt}', ";M (GeV/#it{c}^{2});z_{g}",
                     *mass_binning, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'zg']], write=True)

            h = TH2F(f'h_mass-nsd_{ipt}', ";M (GeV/#it{c}^{2});N_{sd}",
                     *mass_binning, 10, 0.0, 10.0)
            fill_hist(h, df[['fM', 'nsd']], write=True)

            h = TH2F(f'h_mass-rg_{ipt}', ";M (GeV/#it{c}^{2});R_{g}",
                     *mass_binning, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'rg']], write=True)

            h = TH2F(f'h_mass-zpar_{ipt}', ";M (GeV/#it{c}^{2});z_{#parallel}",
                     *mass_binning, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'zpar']], write=True)

            h = TH2F(f'h_mass-dr_{ipt}', ";M (GeV/#it{c}^{2});#Deltar",
                     *mass_binning, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'dr']], write=True)

            h = TH3F(f'h_mass-zg-rg_{ipt}', ";M (GeV/#it{c}^{2});z_{g};R_{g}",
                     *mass_binning, 10, 0., 1., 10, 0., 1.)
            fill_hist(h, df[['fM', 'zg', 'rg']], write=True)

            h = TH3F(f'h_mass-lntheta-lnkt_{ipt}', ";",
                     *mass_binning, 10, 0., 5., 10, -8., 2.)
            fill_hist(h, df[['fM', 'lntheta', 'lnkt']], arraycols=True, write=True)

            #invariant mass with candidatePT intervals (done)
            #invariant mass with jetPT and candidatePT intervals
            #invariant mass with jetPT and candidatePT and shape intervals

    #region efficiency
    def process_efficiency_single(self, index): # pylint: disable=too-many-branches,too-many-statements
        self.logger.info('Running efficiency')
        myfile = TFile.Open(self.l_histoeff[index], "recreate")
        myfile.cd()
        ptbins = array('f', self.lpt_finbinmin + [self.lpt_finbinmax[-1]])

        cats = ['prompt', 'nonprompt']
        levels = ['gen', 'det', 'match']
        h = {(cat, level): TH1F(f'h_pthf_{cat}_{level}', ";p_{T} (GeV/#it{c})", len(ptbins)-1, ptbins)
             for cat in cats for level in levels}

        observables = ['zg']
        h_effkine = {}
        response_matrix = {}
        for var in observables:
            h_effkine[('np', 'gen', 'nocuts', var)] = create_hist(
                f'hkinematiceff_np_gennodetcuts_{var}', f";p_{{T}}^{{jet}} (GeV/#it{{c}});{var}", 10, 5,55,10,0,1)
            h_effkine[('np', 'gen', 'detcuts', var)] = create_hist(
                f'hkinematiceff_np_gendetcuts_{var}', f";p_{{T}}^{{jet}} (GeV/#it{{c}});{var}", 10, 5,55,10,0,1)
            h_effkine[('np', 'det', 'nocuts', var)] = create_hist(
                f'hkinematiceff_np_detnogencuts_{var}', f";p_{{T}}^{{jet}} (GeV/#it{{c}});{var}", 10, 5,55,10,0,1)
            h_effkine[('np', 'det', 'gencuts', var)] = create_hist(
                f'hkinematiceff_np_detgencuts_{var}', f";p_{{T}}^{{jet}} (GeV/#it{{c}});{var}", 10, 5,55,10,0,1)
            response_matrix[('np', var)] = (
                ROOT.RooUnfoldResponse(h_effkine[('np', 'det', 'nocuts', var)],
                                       h_effkine[('np', 'gen', 'nocuts', var)]))

        for ipt in range(self.p_nptbins):
            cols = ['ismcprompt', 'fPt', 'fEta', 'fPhi',
                    'fJetPt', 'fJetEta', 'fJetPhi', 'fPtLeading', 'fPtSubLeading', 'fTheta']
            df = read_df(self.mptfiles_gensk[ipt][index], columns=cols)
            df.query('fJetPt > 5', inplace = True) #TODO: should be removed, just for a speedup check
            if df.empty:
                continue
            df = self.process_calculate_variables(df)
            df.rename(lambda name: name + '_gen', axis=1, inplace=True)
            dfgen = {'prompt': df.loc[df.ismcprompt_gen == 1], 'nonprompt': df.loc[df.ismcprompt_gen == 0]}

            cols.extend(self.cfg('efficiency.extra_cols', []))
            if idx := self.cfg('efficiency.index_match'):
                cols.append(idx)
            df = read_df(self.mptfiles_recosk[ipt][index], columns=cols)
            df.query('fJetPt > 5.', inplace=True) # TODO: check
            dfquery(df, self.cfg('efficiency.filter_det'), inplace=True)
            # dfdet = dfdet.loc[(dfdet.isd0 & dfdet.seld0) | (dfdet.isd0bar & dfdet.seld0bar)]
            if df.empty:
                continue
            df = self.process_calculate_variables(df)
            dfdet = {'prompt': df.loc[df.ismcprompt == 1], 'nonprompt': df.loc[df.ismcprompt == 0]}

            if idx := self.cfg('efficiency.index_match'):
                for cat in cats:
                    dfdet[cat]['idx_match'] = dfdet[cat][idx].apply(lambda ar: ar[0] if len(ar) > 0 else -1)
                dfmatch = {cat: pd.merge(dfdet[cat], dfgen[cat], left_on=['df', 'idx_match'], right_index=True)
                           for cat in cats}
            else:
                self.logger.warning('No matching criterion specified, cannot calculate matched efficiency')
                dfmatch = {cat: None for cat in cats}

            for cat in cats:
                fill_hist(h[(cat, 'gen')], dfgen[cat]['fPt_gen'])
                fill_hist(h[(cat, 'det')], dfdet[cat]['fPt'])
                if dfmatch[cat] is not None:
                    fill_hist(h[(cat, 'match')], dfmatch[cat]['fPt_gen'])

            if dfmatch[cat] is None:
                continue

            for var in observables:
                dfmatch_np_eff_gen = dfmatch['nonprompt'].query(
                    f'fJetPt_gen >= 5 and fJetPt_gen < 55 and {var}_gen > 0.1 and {var}_gen < 1 ')
                fill_hist(h_effkine[('np', 'gen', 'nocuts', var)], dfmatch_np_eff_gen[['fJetPt_gen', f'{var}_gen']])
                dfmatch_np_eff_gen.query(
                    f'fJetPt >= 5 and fJetPt < 55 and {var} > 0.1 and {var} < 1 ', inplace = True)
                fill_hist(h_effkine[('np', 'gen', 'detcuts', var)], dfmatch_np_eff_gen[['fJetPt_gen', f'{var}_gen']])
                fill_response(response_matrix[('np', var)],
                              dfmatch_np_eff_gen[['fJetPt', f'{var}', 'fJetPt_gen', f'{var}_gen']])

                dfmatch_np_eff_det = dfmatch['nonprompt'].query(
                    f'fJetPt >= 5 and fJetPt < 55 and {var} > 0.1 and {var} < 1 ')
                fill_hist(h_effkine[('np', 'det', 'nocuts', var)], dfmatch_np_eff_det[['fJetPt', f'{var}']])
                dfmatch_np_eff_det.query(
                    f'fJetPt_gen >= 5 and fJetPt_gen < 55 and {var}_gen > 0.1 and {var}_gen < 1 ', inplace = True)
                fill_hist(h_effkine[('np', 'det', 'gencuts', var)], dfmatch_np_eff_det[['fJetPt', f'{var}']])

        for obj in itertools.chain(h.values(), h_effkine.values(), response_matrix.values()):
            obj.Write()
