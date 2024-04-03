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
import math
import numpy as np
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

    def process_calculate_variables(self, df): # pylint: disable=invalid-name
        # TODO: make process step instead of internally called method
        df['zg'] = -1.0
        df['rg'] = -1.0
        df['nsd'] = -1.0
        df['z_parallel'] = -1.0
        df['radial_distance'] = -1.0
        for idx, row in df.iterrows():
            isSoftDropped = False
            nsd = 0
            for ptLeading, ptSubLeading, theta in zip(row['fPtLeading'], row['fPtSubLeading'], row['fTheta']):
                zg = ptSubLeading / (ptLeading + ptSubLeading)
                if zg > 0.5:
                    zg = 1.0 - 0.5
                if zg >= 0.1:  # TODO: make this configurable
                    if not isSoftDropped:
                        df.loc[idx, 'zg'] = zg
                        df.loc[idx, 'rg'] = theta
                        isSoftDropped = True
                    nsd += 1
            df.loc[idx, 'nsd'] = nsd
            pxJet = row['fJetPt'] * math.cos(row['fJetPhi'])
            pyJet = row['fJetPt'] * math.sin(row['fJetPhi'])
            pzJet = row['fJetPt'] * math.sinh(row['fJetEta'])
            pxHF = row['fPt'] * math.cos(row['fPhi'])
            pyHF = row['fPt'] * math.sin(row['fPhi'])
            pzHF = row['fPt'] * math.sinh(row['fEta'])
            z_parallel_numerator = pxJet * pxHF + pyJet * pyHF + pzJet * pzHF
            z_parallel_denominator = pxJet * pxJet + pyJet * pyJet + pzJet * pzJet
            df.loc[idx, 'z_parallel'] = z_parallel_numerator / z_parallel_denominator
            df.loc[idx, 'radial_distance'] = math.sqrt(
                (row['fJetEta'] - row['fEta'])**2 + (row['fJetPhi'] - row['fPhi'])**2) #TODO change eta to y
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
                df = self.process_calculate_variables(df)
                df.query(f'fPt >= {pt_min} and fPt < {pt_max}', inplace=True)

                h_invmass_all = TH1F(f'hmass_{ipt}', "",
                                     self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                fill_hist(h_invmass_all, df.fM)
                h_invmass_all.Write()

                h_candpt_all = TH1F(f'hcandpt_{ipt}', "", self.p_num_bins, 0., 50.)
                fill_hist(h_candpt_all, df.fPt)
                h_candpt_all.Write()

                h_jetpt_all = TH1F(f'hjetpt_{ipt}', "", self.p_num_bins, 0., 50.)
                fill_hist(h_jetpt_all, df.fJetPt)
                h_jetpt_all.Write()

                ## substructure
                h_zg = TH1F(f'hjetzg_{ipt}', "", 10, 0.0, 1.0)
                fill_hist(h_zg, df.zg)
                h_zg.Write()

                h_nsd = TH1F(f'hjetnsd_{ipt}', "", 10, 0.0, 10.0)
                fill_hist(h_nsd, df.nsd)
                h_nsd.Write()

                h_rg = TH1F(f'hjetrg_{ipt}', "", 100, 0.0, 1.0)
                fill_hist(h_rg, df.rg)
                h_rg.Write()

                h_zpar = TH1F(f'hjetzpar_{ipt}', "", 100, 0.0, 1.0)
                fill_hist(h_zpar, df.z_parallel)
                h_zpar.Write()

                h_dr = TH1F(f'hjetdr_{ipt}', "", 10, 0.0, 1.0)
                fill_hist(h_dr, df.radial_distance)
                h_dr.Write()

                h = TH2F(f'h2jet_invmass_zg_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                if len(df.fM) > 0:
                    h.FillN(len(df.fM), np.float64(df.fM), np.float64(df.zg), np.float64(len(df.fM)*[1.]))
                h.Write()

                h = TH2F(f'h2jet_invmass_nsd_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 10.0)
                if len(df.fM) > 0:
                    h.FillN(len(df.fM), np.float64(df.fM), np.float64(df.nsd), np.float64(len(df.fM)*[1.]))
                h.Write()

                h = TH2F(f'h2jet_invmass_rg_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                if len(df.fM) > 0:
                    h.FillN(len(df.fM), np.float64(df.fM), np.float64(df.rg), np.float64(len(df.fM)*[1.]))
                h.Write()

                h = TH2F(f'h2jet_invmass_zpar_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                if len(df.fM) > 0:
                    h.FillN(len(df.fM), np.float64(df.fM), np.float64(df.z_parallel), np.float64(len(df.fM)*[1.]))
                h.Write()

                h = TH2F(f'h2jet_invmass_dr_{ipt}', "", 2000, 1.0, 3.0, 10, 0.0, 1.0)
                if len(df.fM) > 0:
                    h.FillN(len(df.fM), np.float64(df.fM), np.float64(df.radial_distance), np.float64(len(df.fM)*[1.]))
                h.Write()

                # TODO: wouldn't it be better to project in the analyzer?
                for i in range(5):
                    df_zg = df.query(f'zg >= {i*0.1} and zg < {i*0.1+0.1}')
                    h_invmass_zg = TH1F(
                        f'hmass_zg_{ipt}_{i}', "", self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    fill_hist(h_invmass_zg, df_zg.fM)
                    h_invmass_zg.Write()

                #invariant mass with candidatePT intervals (done)
                #invariant mass with jetPT and candidatePT intervals
                #invariant mass with jetPT and candidatePT and shape intervals
