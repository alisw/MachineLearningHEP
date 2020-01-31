#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
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

"""
main script for doing data processing, machine learning and analysis
"""
import math
import array
import pickle
import numpy as np
from root_numpy import fill_hist  # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.bitwise import tag_bit_df
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.utilities import seldf_singlevar, openfile
from machine_learning_hep.selectionutils import getnormforselevt
from machine_learning_hep.processer import Processer

class ProcesserDhadrons(Processer): # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, d_val, typean, runlisttrigger, d_mcreweights):
        super().__init__(case, datap, run_param, mcordata, p_maxfiles,
                         d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                         p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                         p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                         d_results, d_val, typean, runlisttrigger, d_mcreweights)

        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']
        self.p_bin_width = datap["analysis"][self.typean]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        self.l_selml = ["y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[ipt]) \
                       for ipt in range(self.p_nptbins)]
        self.s_presel_gen_eff = datap["analysis"][self.typean]['presel_gen_eff']


        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger


    # pylint: disable=too-many-branches
    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        dfevtorig = pickle.load(openfile(self.l_evtorig[index], "rb"))
        if self.s_trigger is not None:
            dfevtorig = dfevtorig.query(self.s_trigger)
        dfevtorig = selectdfrunlist(dfevtorig, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
        hNorm = TH1F("hEvForNorm", "hEvForNorm", 2, 0.5, 2.5)
        hNorm.GetXaxis().SetBinLabel(1, "normsalisation factor")
        hNorm.GetXaxis().SetBinLabel(2, "selected events")
        nselevt = 0
        norm = 0
        if not dfevtorig.empty:
            nselevt = len(dfevtorig.query("is_ev_rej==0"))
            norm = getnormforselevt(dfevtorig)
        hNorm.SetBinContent(1, norm)
        hNorm.SetBinContent(2, nselevt)
        hNorm.Write()
        dfevtorig = dfevtorig.query("is_ev_rej==0")
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            if self.doml is True:
                df = df.query(self.l_selml[bin_id])
            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger is not None:
                df = df.query(self.s_trigger)
            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            suffix = "%s%d_%d" % \
                     (self.v_var_binning, self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                             self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            df = selectdfrunlist(df, \
                     self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            fill_hist(h_invmass, df.inv_mass)
            myfile.cd()
            h_invmass.Write()
            if self.mcordata == "mc":
                df[self.v_ismcrefl] = np.array(tag_bit_df(df, self.v_bitvar,
                                                          self.b_mcrefl), dtype=int)
                df_sig = df[df[self.v_ismcsignal] == 1]
                df_refl = df[df[self.v_ismcrefl] == 1]
                h_invmass_sig = TH1F("hmass_sig" + suffix, "", self.p_num_bins,
                                     self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                h_invmass_refl = TH1F("hmass_refl" + suffix, "", self.p_num_bins,
                                      self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                fill_hist(h_invmass_sig, df_sig.inv_mass)
                fill_hist(h_invmass_refl, df_refl.inv_mass)
                myfile.cd()
                h_invmass_sig.Write()
                h_invmass_refl.Write()
            print("FINISHED")

    # pylint: disable=line-too-long
    def process_efficiency_single(self, index):
        print("step1")
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
        print("step2")

        bincounter = 0
        for ipt in range(self.p_nptfinbins):
            print("step2a")
            bin_id = self.bin_matching[ipt]
            df_mc_reco = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_trigger is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger)
            df_mc_reco = selectdfrunlist(df_mc_reco, \
                     self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[bin_id][index], "rb"))
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
            print("step2b")
            df_mc_gen = selectdfrunlist(df_mc_gen, \
                     self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
            df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
            df_reco_sel_pr = None
            if self.doml is True:
                df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[bin_id])
            else:
                df_reco_sel_pr = df_reco_presel_pr.copy()
            df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
            df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
            df_reco_sel_fd = None
            print("step2d")
            if self.doml is True:
                df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[bin_id])
            else:
                df_reco_sel_fd = df_reco_presel_fd.copy()

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
            print("step2e")

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
            print("step2f")

        out_file.cd()
        h_gen_pr.Write()
        h_presel_pr.Write()
        h_sel_pr.Write()
        h_gen_fd.Write()
        h_presel_fd.Write()
        h_sel_fd.Write()
        print("FINALISED")
