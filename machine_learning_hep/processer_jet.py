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
import math # pylint: disable=unused-import
import time
import numpy as np
import pandas as pd
from ROOT import TFile, TH1F, TH2F, TH3F # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.processer import Processer
from machine_learning_hep.utilities import fill_hist, read_df

class ProcesserJets(Processer): # pylint: disable=invalid-name, too-many-instance-attributes
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
        self.logger.info("initialized processer for D0 jets")

        # selection (temporary)
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_jetsel_gen = datap["analysis"][self.typean].get("jetsel_gen", None)
        self.s_jetsel_reco = datap["analysis"][self.typean].get("jetsel_reco", None)
        self.s_jetsel_gen_matched_reco = \
            datap["analysis"][self.typean].get("jetsel_gen_matched_reco", None)
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger

        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.p_bin_width = datap["analysis"][self.typean]["bin_width"]
        self.p_mass_fit_lim = datap["analysis"][self.typean]["mass_fit_lim"]
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) /
                                    self.p_bin_width))

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
                if zg >= 0.1:  # TODO: make this configurable
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
        zcut = .1
        df['zg'] = df['zg_array'].apply((lambda ar: next((zg for zg in ar if zg >= zcut), -1.)))
        df['rg'] = df[['zg_array', 'fTheta']].apply(
            (lambda ar: next((rg for (zg, rg) in zip(ar.zg_array, ar.fTheta) if zg >= zcut), -1.)), axis=1)
        df['nsd'] = df['zg_array'].apply((lambda ar: len([zg for zg in ar if zg >= zcut])))
        # self.calculate_zg(df)
        return df

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

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            pt_min = self.lpt_finbinmin[ipt]
            pt_max = self.lpt_finbinmax[ipt]

            df = read_df(self.mptfiles_recosk[bin_id][index])
            df.query(f'fPt >= {pt_min} and fPt < {pt_max}', inplace=True)
            if df.empty:
                continue
            df = self.process_calculate_variables(df)

            self.logger.info('preparing histograms for bin %d', ipt)
            h_invmass_all = TH1F(f'h_mass_{ipt}', "Inv. mass;M (GeV/#it{c}^{2})",
                                 self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
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

            h = TH2F(f'h_mass-zg_{ipt}', ";M (GeV/#it{c}^{2});z_{g}",
                     2000, 1.0, 3.0, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'zg']], write=True)

            h = TH2F(f'h_mass-nsd_{ipt}', ";M (GeV/#it{c}^{2});N_{sd}",
                     2000, 1.0, 3.0, 10, 0.0, 10.0)
            fill_hist(h, df[['fM', 'nsd']], write=True)

            h = TH2F(f'h_mass-rg_{ipt}', ";M (GeV/#it{c}^{2});R_{g}",
                     2000, 1.0, 3.0, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'rg']], write=True)

            h = TH2F(f'h_mass-zpar_{ipt}', ";M (GeV/#it{c}^{2});z_{#parallel}",
                     2000, 1.0, 3.0, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'zpar']], write=True)

            h = TH2F(f'h_mass-dr_{ipt}', ";M (GeV/#it{c}^{2});#Deltar",
                     2000, 1.0, 3.0, 10, 0.0, 1.0)
            fill_hist(h, df[['fM', 'dr']], write=True)

            h = TH3F(f'h_mass-zg-rg_{ipt}', ";M (GeV/#it{c}^{2});z_{g};R_{g}",
                     2000, 1., 3., 10, 0., 1., 10, 0., 1.)
            fill_hist(h, df[['fM', 'zg', 'rg']], write=True)

            #invariant mass with candidatePT intervals (done)
            #invariant mass with jetPT and candidatePT intervals
            #invariant mass with jetPT and candidatePT and shape intervals

    def process_efficiency_single(self, index):
        self.logger.info('Running efficiency')
        myfile = TFile.Open(self.l_histoeff[index], "recreate")
        myfile.cd()
        ptbins = array('f', self.lpt_finbinmin + [self.lpt_finbinmax[-1]])
        h_gen = TH1F('h_pthf_gen', ";p_{T} (GeV/#it{c})", len(ptbins)-1, ptbins)
        h_det = TH1F('h_pthf_det', ";p_{T} (GeV/#it{c})", len(ptbins)-1, ptbins)
        h_match = TH1F('h_pthf_match', ";p_{T} (GeV/#it{c})", len(ptbins)-1, ptbins)
        for ipt in range(self.p_nptbins):
            cols = ['ismcprompt', 'fPt']
            dfgen = read_df(self.mptfiles_gensk[ipt][index], filters=[('ismcprompt', '==', 1)], columns=cols)
            cols.append(self.cfg('index_match'))
            # cols.extend(['isd0', 'isd0bar', 'seld0', 'seld0bar'])
            dfdet = read_df(self.mptfiles_recosk[ipt][index], filters=[('ismcprompt', '==', 1)], columns=cols)
            dfgen = dfgen.loc[dfgen.ismcprompt == 1]
            # dfdet = dfdet.loc[(dfdet.isd0 & dfdet.seld0) | (dfdet.isd0bar & dfdet.seld0bar)] # TODO: generalize
            dfdet = dfdet.loc[dfdet.ismcprompt == 1]
            fill_hist(h_gen, dfgen['fPt'])
            fill_hist(h_det, dfdet['fPt'])
            if (idx := self.cfg('index_match')) is not None:
                dfdet['idx_match'] = dfdet[idx].apply(lambda ar: ar[0] if len(ar) > 0 else -1)
                dfmatch = pd.merge(dfdet, dfgen[['ismcprompt']],
                                   left_on=['df', 'idx_match'], right_index=True)
                fill_hist(h_match, dfmatch['fPt']) # TODO: fill generated pt?
        h_gen.Write()
        h_det.Write()
        h_match.Write()
