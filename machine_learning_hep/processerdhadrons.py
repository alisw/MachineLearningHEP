#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
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

# pylint: disable=import-error, no-name-in-module, consider-using-f-string

"""
main script for doing data processing, machine learning and analysis
"""
import math
import array
import numpy as np
import pandas as pd
from ROOT import TFile, TH1F
from machine_learning_hep.utilities import seldf_singlevar, read_df
from machine_learning_hep.processer import Processer, dfquery
from machine_learning_hep.utils.hist import bin_array, create_hist, fill_hist

class ProcesserDhadrons(Processer): # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, typean, runlisttrigger, d_mcreweights):
        super().__init__(case, datap, run_param, mcordata, p_maxfiles,
                         d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                         p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                         p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                         d_results, typean, runlisttrigger, d_mcreweights)

        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']
        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        limits_mass = datap["analysis"][self.typean]["mass_fit_lim"]
        nbins_mass = int(round((limits_mass[1] - limits_mass[0]) / self.p_bin_width))
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        self.s_presel_gen_eff = datap["analysis"][self.typean]['presel_gen_eff']


        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.v_invmass = datap["variables"].get("var_inv_mass", "fM")
        self.binarray_mass = bin_array(nbins_mass, limits_mass[0], limits_mass[1])
        self.binarray_pthf = np.asarray(self.cfg('sel_an_binmin', []) + self.cfg('sel_an_binmax', [])[-1:], 'd')

    # pylint: disable=too-many-branches
    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        dfevtorig = read_df(self.l_evtorig[index])
        neventsorig = len(dfevtorig)
        if self.s_evtsel is not None:
            dfevtevtsel = dfevtorig.query(self.s_evtsel)
        else:
            dfevtevtsel = dfevtorig
        neventsafterevtsel = len(dfevtevtsel)

        #validation plot for event selection
        histonorm = TH1F("histonorm", "histonorm", 10, 0, 10)
        histonorm.SetBinContent(1, neventsorig)
        histonorm.GetXaxis().SetBinLabel(1, "tot events")
        histonorm.SetBinContent(2, neventsafterevtsel)
        histonorm.GetXaxis().SetBinLabel(2, "tot events after evt sel")
        histonorm.Write()

        myfile.cd()
        hEvents = TH1F('all_events', 'all_events', 1, -0.5, 0.5)
        hSelEvents = TH1F('sel_events', 'sel_events', 1, -0.5, 0.5)
        hEvents.SetBinContent(1, len(dfevtorig))
        hSelEvents.SetBinContent(1, len(dfevtevtsel))

        hEvents.Write()
        hSelEvents.Write()

        df_ptmerged = pd.DataFrame()

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = read_df(self.mptfiles_recoskmldec[bin_id][index])
            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)

            if self.doml is True:
                df = df.query(self.l_selml[bin_id])
            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

            if self.do_custom_analysis_cuts:
                df = self.apply_cuts_ptbin(df, ipt)

            df_ptmerged = pd.concat([df_ptmerged, df], ignore_index=True)

            if self.mltype == "MultiClassification":
                suffix = "%s%d_%d_%.2f%.2f%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt][0],
                          self.lpt_probcutfin[ipt][1], self.lpt_probcutfin[ipt][2])
            else:
                suffix = "%s%d_%d_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[ipt])

            h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                             self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])

            fill_hist(h_invmass, df[self.v_invmass])
            myfile.cd()
            h_invmass.Write()

            if self.mcordata == "mc":
                df_sig = df[df[self.v_ismcsignal] == 1]
                df_bkg = df[df[self.v_ismcbkg] == 1]
                h_invmass_sig = TH1F("hmass_sig" + suffix, "", self.p_num_bins,
                                     self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                h_invmass_bkg = TH1F("hmass_bkg" + suffix, "", self.p_num_bins,
                                     self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])

                fill_hist(h_invmass_sig, df_sig[self.v_invmass])
                fill_hist(h_invmass_bkg, df_bkg[self.v_invmass])

                myfile.cd()
                h_invmass_sig.Write()
                h_invmass_bkg.Write()

        for sel_name, sel_spec in self.cfg('data_selections', {}).items():
            if sel_spec['level'] == self.mcordata:
                df_sel = dfquery(df_ptmerged, sel_spec['query'])
                h = create_hist(
                    f'h_mass-pthf_{sel_name}',
                    ';M (GeV/#it{c}^{2});p_{T}^{HF} (GeV/#it{c})',
                    self.binarray_mass, self.binarray_pthf)
                fill_hist(h, df_sel[['fM', 'fPt']], write=True)

    # pylint: disable=line-too-long
    def process_efficiency_single(self, index):
        #TO UPDATE TO DHADRON_MULT VERSION
        out_file = TFile.Open(self.l_histoeff[index], "recreate")
        n_bins = len(self.lpt_finbinmin)
        analysis_bin_lims_temp = self.lpt_finbinmin.copy()
        analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
        analysis_bin_lims = array.array('f', analysis_bin_lims_temp)
        h_gen_pr = TH1F("h_gen_pr", "Prompt Generated in acceptance |y|<0.5", \
                        n_bins, analysis_bin_lims)
        h_presel_pr = TH1F("h_presel_pr", "Prompt Reco in acc |#eta|<0.8 and sel", \
                           n_bins, analysis_bin_lims)
        h_sel_pr = TH1F("h_sel_pr", "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                        n_bins, analysis_bin_lims)
        h_gen_fd = TH1F("h_gen_fd", "FD Generated in acceptance |y|<0.5", \
                        n_bins, analysis_bin_lims)
        h_presel_fd = TH1F("h_presel_fd", "FD Reco in acc |#eta|<0.8 and sel", \
                           n_bins, analysis_bin_lims)
        h_sel_fd = TH1F("h_sel_fd", "FD Reco and sel in acc |#eta|<0.8 and sel", \
                        n_bins, analysis_bin_lims)

        bincounter = 0
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df_mc_reco = read_df(self.mptfiles_recoskmldec[bin_id][index])
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            df_mc_gen = read_df(self.mptfiles_gensk[bin_id][index])
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df_gen_sel_pr = df_mc_gen.loc[(df_mc_gen.ismcprompt == 1) & (df_mc_gen.ismcsignal == 1)]
            df_reco_presel_pr = df_mc_reco.loc[(df_mc_reco.ismcprompt == 1) & (df_mc_reco.ismcsignal == 1)]
            df_reco_sel_pr = None
            if self.doml is True:
                df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[bin_id])
            else:
                df_reco_sel_pr = df_reco_presel_pr.copy()
            df_gen_sel_fd = df_mc_gen.loc[(df_mc_gen.ismcfd == 1) & (df_mc_gen.ismcsignal == 1)]
            df_reco_presel_fd = df_mc_reco.loc[(df_mc_reco.ismcfd == 1) & (df_mc_reco.ismcsignal == 1)]
            df_reco_sel_fd = None
            if self.doml is True:
                df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[bin_id])
            else:
                df_reco_sel_fd = df_reco_presel_fd.copy()

            if self.do_custom_analysis_cuts:
                df_reco_sel_pr = self.apply_cuts_ptbin(df_reco_sel_pr, ipt)
                df_reco_sel_fd = self.apply_cuts_ptbin(df_reco_sel_fd, ipt)

            val = len(df_gen_sel_pr)
            err = math.sqrt(val)
            h_gen_pr.SetBinContent(bincounter + 1, val)
            h_gen_pr.SetBinError(bincounter + 1, err)
            val = len(df_reco_presel_pr)
            err = math.sqrt(val)
            h_presel_pr.SetBinContent(bincounter + 1, val)
            h_presel_pr.SetBinError(bincounter + 1, err)
            val = len(df_reco_sel_pr)
            err = math.sqrt(val)
            h_sel_pr.SetBinContent(bincounter + 1, val)
            h_sel_pr.SetBinError(bincounter + 1, err)

            val = len(df_gen_sel_fd)
            err = math.sqrt(val)
            h_gen_fd.SetBinContent(bincounter + 1, val)
            h_gen_fd.SetBinError(bincounter + 1, err)
            val = len(df_reco_presel_fd)
            err = math.sqrt(val)
            h_presel_fd.SetBinContent(bincounter + 1, val)
            h_presel_fd.SetBinError(bincounter + 1, err)
            val = len(df_reco_sel_fd)
            err = math.sqrt(val)
            h_sel_fd.SetBinContent(bincounter + 1, val)
            h_sel_fd.SetBinError(bincounter + 1, err)
            bincounter = bincounter + 1

        out_file.cd()
        h_gen_pr.Write()
        h_presel_pr.Write()
        h_sel_pr.Write()
        h_gen_fd.Write()
        h_presel_fd.Write()
        h_sel_fd.Write()
