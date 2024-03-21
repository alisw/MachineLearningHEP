#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
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
from ROOT import TFile, TH1F, TH2F # pylint: disable=import-error, no-name-in-module
import numpy as np
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

    def process_histomass_single(self, index):
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
                df.query(f'fPt > {pt_min} and fPt < {pt_max}', inplace=True)
                df['zg'] = 1.
                for idx, row in df.iterrows():
                    for ptLeading, ptSubLeading in zip(row['fPtLeading'], row['fPtSubLeading']):
                        zg = ptSubLeading / (ptLeading + ptSubLeading)
                        if zg > 0.5:
                            zg = 1.0 - 0.5
                        if zg >= 0.1:
                            df.loc[idx,'zg'] = zg
                            break



                h_invmass_all = TH1F(
                    f'hmass_{ipt}', "",
                    self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                fill_hist(h_invmass_all, df.fM)
                h_invmass_all.Write()

                h_candpt_all = TH1F(
                    f'hcandpt_{ipt}', "",
                    self.p_num_bins, 0., 50.)
                fill_hist(h_candpt_all, df.fPt)
                h_candpt_all.Write()

                h_jetpt_all = TH1F(
                    f'hjetpt_{ipt}', "",
                    self.p_num_bins, 0., 50.)
                fill_hist(h_jetpt_all, df.fJetPt)
                h_jetpt_all.Write()

                h_zg = TH1F(
                    f'hjetzg_{ipt}', "",
                    10, 0.0, 1.0)
                fill_hist(h_zg, df.zg)
                h_zg.Write()

                h2_invmass_zg=TH2F(
                    f'h2jet_invmass_zg_{ipt}', "",
                    2000, 1.0, 3.0,
                    10, 0.0, 1.0)
                print(len(df.fM), np.float64(df.fM), np.float64(df.zg))
                h2_invmass_zg.FillN(len(df.fM), np.float64(df.fM), np.float64(df.zg),
                                    np.float64(len(df.fM)*[1.]))
                h2_invmass_zg.Write()
        print("end of processor")


                #invariant mass with candidatePT intervals (done)
                #invariant mass with jetPT and candidatePT intervals
                #invariant mass with jetPT and candidatePT and shape intervals
