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

import pickle
import numba
import numpy as np
import pandas as pd
from ROOT import TFile, TH1F, TH2F # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.processer import Processer
from machine_learning_hep.utilities import fill_hist, openfile

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

    def calculate_zg(self, df):
        df['zg'] = -1.0
        df['rg'] = -1.0
        df['nsd'] = -1.0
        df['zg_array'] = np.array(df.fPtSubLeading / (df.fPtLeading + df.fPtSubLeading))
        # TODO: check for zg > 0.5
        # TODO: check for soft drop
        # TODO: can we optimize this loop?
        for idx, row in df.iterrows():
            isSoftDropped = False
            nsd = 0
            for zg, theta in zip(row['zg_array'], row['fTheta']):
                if zg > 0.5:
                    zg = 1.0 - 0.5
                if zg >= 0.1:  # TODO: make this configurable
                    if not isSoftDropped:
                        df.loc[idx, 'zg'] = zg
                        df.loc[idx, 'rg'] = theta
                        isSoftDropped = True
                    nsd += 1
            df.loc[idx, 'nsd'] = nsd

    def process_calculate_variables(self, df): # pylint: disable=invalid-name
        # TODO: make process step instead of internally called method
        df.eval('radial_distance = sqrt((fJetEta - fEta)**2 + (fJetPhi - fPhi)**2)', inplace=True) # TODO: consider periodic phi
        df.eval('jetPx = fJetPt * cos(fJetPhi)', inplace=True)
        df.eval('jetPy = fJetPt * sin(fJetPhi)', inplace=True)
        df.eval('jetPz = fJetPt * sinh(fJetEta)', inplace=True)
        df.eval('hfPx = fPt * cos(fPhi)', inplace=True)
        df.eval('hfPy = fPt * sin(fPhi)', inplace=True)
        df.eval('hfPz = fPt * sinh(fEta)', inplace=True)
        df.eval('zpar_num = jetPx * hfPx + jetPy * hfPy + jetPz * hfPz', inplace=True)
        df.eval('zpar_den = jetPx * jetPx + jetPy * jetPy + jetPz * jetPz', inplace=True)
        df.eval('z_parallel = zpar_num / zpar_den', inplace=True)
        self.calculate_zg(df)
        return df

    def process_histomass_single(self, index): # pylint: disable=too-many-statements
        self.logger.info('processing histomass single')

        myfile = TFile.Open(self.l_histomass[index], "recreate")
        myfile.cd()

        dfevtorig = pickle.load(openfile(self.l_evtorig[index], "rb"))
        dfevtevtsel = dfevtorig.query(self.s_evtsel)
        neventsafterevtsel = len(dfevtevtsel)
        histonorm = TH1F("histonorm", "histonorm", 1, 0, 1)
        histonorm.SetBinContent(1, neventsafterevtsel)
        histonorm.Write()

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            pt_min = self.lpt_finbinmin[ipt]
            pt_max = self.lpt_finbinmax[ipt]
            with openfile(self.mptfiles_recosk[bin_id][index], "rb") as file:
                df = pickle.load(file)
                df.query(f'fPt >= {pt_min} and fPt < {pt_max}', inplace=True)
                self.logger.info('calculating derived variables')
                df = self.process_calculate_variables(df)

                self.logger.info('preparing histograms')
                h_invmass_all = TH1F(f'hmass_{ipt}', "",
                                     self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                fill_hist(h_invmass_all, df.fM, write=True)

                h_candpt_all = TH1F(f'hcandpt_{ipt}', "", self.p_num_bins, 0., 50.)
                fill_hist(h_candpt_all, df.fPt, write=True)

                h_jetpt_all = TH1F(f'hjetpt_{ipt}', "", self.p_num_bins, 0., 50.)
                fill_hist(h_jetpt_all, df.fJetPt, write=True)

                ## substructure
                h_zg = TH1F(f'hjetzg_{ipt}', "", 10, 0.0, 1.0)
                fill_hist(h_zg, df.zg, write=True)

                h_nsd = TH1F(f'hjetnsd_{ipt}', "", 10, 0.0, 10.0)
                fill_hist(h_nsd, df.nsd, write=True)

                h_rg = TH1F(f'hjetrg_{ipt}', "", 100, 0.0, 1.0)
                fill_hist(h_rg, df.rg, write=True)

                h_zpar = TH1F(f'hjetzpar_{ipt}', "", 100, 0.0, 1.0)
                fill_hist(h_zpar, df.z_parallel, write=True)

                h_dr = TH1F(f'hjetdr_{ipt}', "", 10, 0.0, 1.0)
                fill_hist(h_dr, df.radial_distance, write=True)

                h = TH2F(f'h2jet_invmass_zg_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                fill_hist(h, df[['fM', 'zg']], write=True)

                h = TH2F(f'h2jet_invmass_nsd_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 10.0)
                fill_hist(h, df[['fM', 'nsd']], write=True)

                h = TH2F(f'h2jet_invmass_rg_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                fill_hist(h, df[['fM', 'rg']], write=True)

                h = TH2F(f'h2jet_invmass_zpar_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                fill_hist(h, df[['fM', 'z_parallel']], write=True)

                h = TH2F(f'h2jet_invmass_dr_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                fill_hist(h, df[['fM', 'radial_distance']], write=True)

                #invariant mass with candidatePT intervals (done)
                #invariant mass with jetPT and candidatePT intervals
                #invariant mass with jetPT and candidatePT and shape intervals

    def process_efficiency_single(self, index):
        self.logger.info('Running efficiency')
        myfile = TFile.Open(self.l_histoeff[index], "recreate")
        myfile.cd()
        h_gen = TH1F(f'hjetgen', "", self.p_nptfinbins, self.lpt_finbinmin[0], self.lpt_finbinmax[-1])
        h_det = TH1F(f'hjetdet', "", self.p_nptfinbins, self.lpt_finbinmin[0], self.lpt_finbinmax[-1])
        for ipt in range(self.p_nptbins):
            with (openfile(self.mptfiles_recosk[ipt][index], "rb") as file,
              openfile(self.mptfiles_gensk[ipt][index], "rb") as mcfile):
                dfdet = pickle.load(file)
                fill_hist(h_det, dfdet['fPt'])
                dfgen = pickle.load(mcfile)
                fill_hist(h_gen, dfgen['fPt'])
        h_gen.Write()
        h_det.Write()
