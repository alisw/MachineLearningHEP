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
#import array
#import multiprocessing as mp
import pickle
#import os
#import random as rd
#import uproot
import pandas as pd
import numpy as np
from root_numpy import fill_hist #, evaluate # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F, TH2F #, TH3F, RooUnfoldResponse # pylint: disable=import-error, no-name-in-module
#from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import tag_bit_df
from machine_learning_hep.utilities import selectdfrunlist
#from machine_learning_hep.utilities import list_folders, createlist, appendmainfoldertolist
from machine_learning_hep.utilities import  seldf_singlevar, openfile
#from machine_learning_hep.utilities import mergerootfiles, z_calc, z_gen_calc
#from machine_learning_hep.utilities import get_timestamp_string
#from machine_learning_hep.utilities_plot import scatterplotroot
#from machine_learning_hep.models import apply # pylint: disable=import-error
#from machine_learning_hep.globalfitter import fitter
from machine_learning_hep.selectionutils import getnormforselevt
from machine_learning_hep.processer import Processer

def new_df_pairs(dfreco, mc_case):
    df = []
    for _, group in dfreco.groupby(["run_number", "ev_id"], sort=False):
        i = 0
        for row_number_1, row_1 in group.iterrows():
            for row_number_2, row_2 in group.iterrows():
                if row_number_2 > row_number_1:
                    df_tmp = pd.DataFrame()
                    df_tmp.loc[i, "run_number"] = row_2["run_number"]
                    df_tmp.loc[i, "ev_id"] = row_2["ev_id"]
                    df_tmp.loc[i, "pt_cand1"] = row_1["pt_cand"]
                    df_tmp.loc[i, "pt_cand2"] = row_2["pt_cand"]
                    df_tmp.loc[i, "inv_cand_1"] = row_1["inv_mass"]
                    df_tmp.loc[i, "inv_cand_2"] = row_2["inv_mass"]
                    delta_phi = row_1["phi_cand"] - row_2["phi_cand"]
                    if delta_phi < 0:
                        df_tmp.loc[i, "delta_phi"] = delta_phi + 2*np.pi
                    else:
                        df_tmp.loc[i, "delta_phi"] = (delta_phi)
                    if mc_case:
                        df_tmp.loc[i, "ismcsignal_1"] = row_1["ismcsignal"]
                        df_tmp.loc[i, "ismcsignal_2"] = row_2["ismcsignal"]
                    df.append(df_tmp)
                    i += 1
    dfreco = pd.concat(df)
    dfreco.reset_index()
    return dfreco

def split_df(dfreco, num_part):
    split_indices = (dfreco.shape[0] // num_part) * np.arange(1, num_part, dtype=np.int)
    for i in range(0, num_part-1):
        while (dfreco.iloc[split_indices[i]][["run_number", "ev_id"]] ==
               dfreco.iloc[split_indices[i]-1][["run_number", "ev_id"]]).all():
            split_indices[i] += 1
    dfreco_split = np.split(dfreco, split_indices)
    return dfreco_split


class ProcesserDhadrons_hfcorr(Processer):
    # pylint: disable=invalid-name
    # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, typean, runlisttrigger, d_mcreweights):
        super().__init__(case, datap, run_param, mcordata, p_maxfiles,
                         d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                         p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                         p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                         d_results, typean, runlisttrigger, d_mcreweights)

        self.p_mass_fit_lim = datap["analysis"][self.typean]['mass_fit_lim']
        self.p_delta_phi_lim = datap["analysis"][self.typean]['delta_phi_lim']
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

    def histos(self, name1, name2, dfreco):
        dfreco = dfreco.groupby(["run_number", "ev_id"]).filter(lambda x: len(x) > 1)
        dfreco = dfreco[dfreco["delta_phi"] != 0]
        inv_mass_1 = dfreco["inv_cand_1"].tolist()
        inv_mass_2 = dfreco["inv_cand_2"].tolist()
        d_phi = dfreco["delta_phi"].tolist()
        print("before phi_plot")
        h_delta_phi = TH1F("delta_phi DD pair" + name1 + name2, "", self.p_num_bins,
                           self.p_delta_phi_lim[0], self.p_delta_phi_lim[1])
        print("after phi_plot")
        h_invmass_cand_1 = TH1F("hmass_cand_1" + name1 + name2, "", self.p_num_bins,
                                self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
        h_invmass_cand_2 = TH1F("hmass_cand_2" + name2 + name1, "", self.p_num_bins,
                                self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
        h_DDbar_mass_tot = TH2F("hmass DDbar" + name1 + name2, "", self.p_num_bins,
                                self.p_mass_fit_lim[0], self.p_mass_fit_lim[1],
                                self.p_num_bins, self.p_mass_fit_lim[0],
                                self.p_mass_fit_lim[1])
        for i in range(0, len(inv_mass_1)-1):
            h_invmass_cand_1.Fill(inv_mass_1[i])
            h_invmass_cand_2.Fill(inv_mass_2[i])
            h_DDbar_mass_tot.Fill(inv_mass_1[i], inv_mass_2[i])
            h_delta_phi.Fill(d_phi[i])
        print("after fill")
        h_DDbar_mass_tot.SetOption("lego2z")
        h_delta_phi.Write()
        h_invmass_cand_1.Write()
        h_invmass_cand_2.Write()
        h_DDbar_mass_tot.Write()
        print("histos function - done")

    # pylint: disable=too-many-branches
    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        dfevtorig = pickle.load(openfile(self.l_evtorig[index], "rb"))
        if self.s_trigger is not None:
            dfevtorig = dfevtorig.query(self.s_trigger)
        if self.runlistrigger is not None:
            dfevtorig = selectdfrunlist(dfevtorig, \
                             self.run_param[self.runlistrigger], "run_number")
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
        df_tot = pd.DataFrame()
        for ipt in range(self.p_nptfinbins):
            print("ipt iteration", ipt)
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            print("df loaded", df.shape)
            df_no_cut = df[df["pt_cand">4]]
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
            if self.runlistrigger is not None:
                df = selectdfrunlist(df, \
                         self.run_param[self.runlistrigger], "run_number")
            h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins, self.p_mass_fit_lim[0],
                             self.p_mass_fit_lim[1])
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
            #df_tot = df_tot.append(df)
            df_tot = df_tot.append(df_no_cut)
        df_tot = df_tot.reset_index(drop=True)
        df_group = df_tot
        grouped = df_group.groupby(["run_number","ev_id"])
        grouplen = pd.array(grouped.size())
        gmin = grouplen.min()
        gmax = grouplen.max()
        g_bins = gmax - gmin
        h_grouplen = TH1F("group_length" , "", int(g_bins), gmin, gmax)
        fill_hist(h_grouplen, grouplen)
        h_grouplen.Draw()
        h_grouplen.Write()
        df_tot = df_tot[df_tot["pt_cand">4]]
        # df_tot = df_tot.sample(n=5000)
        mc_case = False
        if self.mcordata == "mc":
            df_work = df_tot[["run_number", "ev_id", "pt_cand", "inv_mass",
                              "phi_cand", "eta_cand", "ismcsignal"]]
            mc_case = True
        else:
            df_work = df_tot[["run_number", "ev_id", "pt_cand", "inv_mass",
                              "phi_cand", "eta_cand"]]
            mc_case = False
        if df_work.shape[0] > 10000:
            split_const = int(df_work.shape[0]/10000)
            df_work.sort_values(["run_number", "ev_id"], inplace=True)
            df_tmp = []
            for i, working_df in enumerate(split_df(df_work, split_const)):
                print("processing working df", i, split_const, "total size of df",
                      df_work.shape)
                #df_tmp.append(filter_phi(working_df))
                df_tmp.append(new_df_pairs(working_df, mc_case))
            df_filt = pd.concat(df_tmp)
        else:
            #df_filt = filter_phi(df_work)
            df_filt = new_df_pairs(df_work, mc_case)
        nameoffile = self.d_results + "/" +  self.l_path[index] + "/dframe.pkl"
        df_filt.to_pickle(nameoffile)
        print("file", nameoffile, "created")
        name1 = " Full data cand 1"
        name2 = " Full data cand 2"
        self.histos(name1, name2, df_filt)
        # sig_fake studies for mc_df
        if self.mcordata == "mc":
            name1 = " signal"
            name2 = " background"
            sig_1 = df_filt[df_filt["ismcsignal_1"] == 1]
            h_invmass_sig1 = TH1F("inv mass sig 1", "", self.p_num_bins,
                                  self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            fill_hist(h_invmass_sig1, sig_1.inv_cand_1)
            h_invmass_sig1.Write()
            sig_2 = df_filt[df_filt["ismcsignal_2"] == 1]
            h_invmass_sig2 = TH1F("inv mass sig 2", "", self.p_num_bins,
                                  self.p_mass_fit_lim[0],
                                  self.p_mass_fit_lim[1])
            fill_hist(h_invmass_sig2, sig_2.inv_cand_2)
            h_invmass_sig2.Write()
            sig_sig = sig_1[sig_1["ismcsignal_2"] == 1]
            sig_bkg = sig_1[sig_1["ismcsignal_2"] == 0]
            bkg_1 = df_filt[df_filt["ismcsignal_1"] == 0]
            h_invmass_bkg1 = TH1F("inv mass bkg 1", "", self.p_num_bins, self.p_mass_fit_lim[0],
                                  self.p_mass_fit_lim[1])
            fill_hist(h_invmass_bkg1, bkg_1.inv_cand_1)
            h_invmass_bkg1.Write()
            bkg_2 = df_filt[df_filt["ismcsignal_2"] == 0]
            h_invmass_bkg2 = TH1F("inv mass bkg 2", "", self.p_num_bins, self.p_mass_fit_lim[0],
                                  self.p_mass_fit_lim[1])
            fill_hist(h_invmass_bkg2, bkg_2.inv_cand_2)
            h_invmass_bkg2.Write()
            bkg_sig = bkg_1[bkg_1["ismcsignal_2"] == 1]
            bkg_bkg = bkg_1[bkg_1["ismcsignal_2"] == 0]
            self.histos(name1, name1, sig_sig)
            self.histos(name1, name2, sig_bkg)
            self.histos(name2, name1, bkg_sig)
            self.histos(name2, name2, bkg_bkg)

#
    # pylint: disable=line-too-long
#    def process_efficiency_single(self, index):
#        print("step1")
#        out_file = TFile.Open(self.l_histoeff[index], "recreate")
#        n_bins = len(self.lpt_finbinmin)
#        analysis_bin_lims_temp = self.lpt_finbinmin.copy()
#        analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
#        analysis_bin_lims = array.array('f', analysis_bin_lims_temp)
#        h_gen_pr = TH1F("h_gen_pr", "Prompt Generated in acceptance |y|<0.5", \
#                        n_bins, analysis_bin_lims)
#        h_presel_pr = TH1F("h_presel_pr", "Prompt Reco in acc |#eta|<0.8 and sel", \
#                           n_bins, analysis_bin_lims)
#        h_sel_pr = TH1F("h_sel_pr", "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
#                        n_bins, analysis_bin_lims)
#        h_gen_fd = TH1F("h_gen_fd", "FD Generated in acceptance |y|<0.5", \
#                        n_bins, analysis_bin_lims)
#        h_presel_fd = TH1F("h_presel_fd", "FD Reco in acc |#eta|<0.8 and sel", \
#                           n_bins, analysis_bin_lims)
#        h_sel_fd = TH1F("h_sel_fd", "FD Reco and sel in acc |#eta|<0.8 and sel", \
#                        n_bins, analysis_bin_lims)
#        print("step2")
#
#        bincounter = 0
#        for ipt in range(self.p_nptfinbins):
#            print("step2a")
#            bin_id = self.bin_matching[ipt]
#            df_mc_reco = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
#            if self.s_evtsel is not None:
#                df_mc_reco = df_mc_reco.query(self.s_evtsel)
#            if self.s_trigger is not None:
#                df_mc_reco = df_mc_reco.query(self.s_trigger)
#            df_mc_reco = selectdfrunlist(df_mc_reco, \
#                     self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
#            df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[bin_id][index], "rb"))
#            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
#            print("step2b")
#            df_mc_gen = selectdfrunlist(df_mc_gen, \
#                     self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
#            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
#                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
#            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
#                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
#            df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
#            df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
#            df_reco_sel_pr = None
#            if self.doml is True:
#                df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[bin_id])
#            else:
#                df_reco_sel_pr = df_reco_presel_pr.copy()
#            df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
#            df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
#            df_reco_sel_fd = None
#            print("step2d")
#            if self.doml is True:
#                df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[bin_id])
#            else:
#                df_reco_sel_fd = df_reco_presel_fd.copy()
#
#            val = len(df_gen_sel_pr)
#            err = math.sqrt(val)
#            h_gen_pr.SetBinContent(bincounter + 1, val)
#            h_gen_pr.SetBinError(bincounter + 1, err)
#            val = len(df_reco_presel_pr)
#            err = math.sqrt(val)
#            h_presel_pr.SetBinContent(bincounter + 1, val)
#            h_presel_pr.SetBinError(bincounter + 1, err)
#            val = len(df_reco_sel_pr)
#            err = math.sqrt(val)
#            h_sel_pr.SetBinContent(bincounter + 1, val)
#            h_sel_pr.SetBinError(bincounter + 1, err)
#            print("step2e")
#
#            val = len(df_gen_sel_fd)
#            err = math.sqrt(val)
#            h_gen_fd.SetBinContent(bincounter + 1, val)
#            h_gen_fd.SetBinError(bincounter + 1, err)
#            val = len(df_reco_presel_fd)
#            err = math.sqrt(val)
#            h_presel_fd.SetBinContent(bincounter + 1, val)
#            h_presel_fd.SetBinError(bincounter + 1, err)
#            val = len(df_reco_sel_fd)
#            err = math.sqrt(val)
#            h_sel_fd.SetBinContent(bincounter + 1, val)
#            h_sel_fd.SetBinError(bincounter + 1, err)
#            bincounter = bincounter + 1
#            print("step2f")
#
#        out_file.cd()
#        h_gen_pr.Write()
#        h_presel_pr.Write()
#        h_sel_pr.Write()
#        h_gen_fd.Write()
#        h_presel_fd.Write()
#        h_sel_fd.Write()
#        print("FINALISED")
