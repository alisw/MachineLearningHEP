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
import multiprocessing as mp
import pickle
import os
import random as rd
import uproot
import pandas as pd
import numpy as np
from root_numpy import fill_hist, evaluate # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F, TH2F, TH3F, RooUnfoldResponse # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import filter_bit_df, tag_bit_df
from machine_learning_hep.utilities import selectdfquery, selectdfrunlist, merge_method
from machine_learning_hep.utilities import list_folders, createlist, appendmainfoldertolist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from machine_learning_hep.utilities import mergerootfiles, z_calc, z_gen_calc
from machine_learning_hep.utilities import get_timestamp_string
from machine_learning_hep.utilities_plot import scatterplotroot
from machine_learning_hep.models import apply # pylint: disable=import-error
#from machine_learning_hep.globalfitter import fitter
from machine_learning_hep.selectionutils import getnormforselevt
from machine_learning_hep.processer import Processer

class ProcesserDhadrons_jet(Processer): # pylint: disable=too-many-instance-attributes
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

        self.lvar2_binmin = datap["analysis"][self.typean]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"][self.typean]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]
        self.v_var2_binning_gen = datap["analysis"][self.typean]["var_binning2_gen"]
        self.corr_eff_mult = datap["analysis"][self.typean]["corrEffMult"]

        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        #self.sel_final_fineptbins = datap["analysis"][self.typean]["sel_final_fineptbins"]
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
        for ibin2 in range(len(self.lvar2_binmin)):
            mybindfevtorig = seldf_singlevar(dfevtorig, self.v_var2_binning_gen, \
                                        self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
            hNorm = TH1F("hEvForNorm_mult%d" % ibin2, "hEvForNorm_mult%d" % ibin2, 2, 0.5, 2.5)
            hNorm.GetXaxis().SetBinLabel(1, "normsalisation factor")
            hNorm.GetXaxis().SetBinLabel(2, "selected events")
            nselevt = 0
            norm = 0
            if not mybindfevtorig.empty:
                nselevt = len(mybindfevtorig.query("is_ev_rej==0"))
                norm = getnormforselevt(mybindfevtorig)
            hNorm.SetBinContent(1, norm)
            hNorm.SetBinContent(2, nselevt)
            hNorm.Write()
#            histmultevt = TH1F("hmultevtmult%d" % ibin2,
#                               "hmultevtmult%d"  % ibin2, 100, 0, 100)
            mybindfevtorig = mybindfevtorig.query("is_ev_rej==0")
#            fill_hist(histmultevt, mybindfevtorig.n_tracklets_corr)
#            histmultevt.Write()
#            h_v0m_ntracklets = TH2F("h_v0m_ntracklets%d" % ibin2,
#                                    "h_v0m_ntracklets%d" % ibin2,
#                                    200, 0, 200, 200, -0.5, 1999.5)
#            v_v0m_ntracklets = np.vstack((mybindfevtorig.n_tracklets_corr,
#                                          mybindfevtorig.v0m_corr)).T
#            fill_hist(h_v0m_ntracklets, v_v0m_ntracklets)
#            h_v0m_ntracklets.Write()

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
            for ibin2 in range(len(self.lvar2_binmin)):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                h_invmass_weight = TH1F("h_invmass_weight" + suffix, "", self.p_num_bins,
                                        self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                df_bin = seldf_singlevar(df, self.v_var2_binning,
                                         self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_bin = selectdfrunlist(df_bin, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
                fill_hist(h_invmass, df_bin.inv_mass)
                if "INT7" not in self.triggerbit and self.mcordata == "data":
                    fileweight_name = "%s/correctionsweights.root" % self.d_val
                    fileweight = TFile.Open(fileweight_name, "read")
                    namefunction = "funcnorm_%s_%s" % (self.triggerbit, self.v_var2_binning_gen)
                    funcweighttrig = fileweight.Get(namefunction)
                    if funcweighttrig:
                        weights = evaluate(funcweighttrig, df_bin[self.v_var2_binning])
                        weightsinv = [1./weight for weight in weights]
                        fill_hist(h_invmass_weight, df_bin.inv_mass, weights=weightsinv)
                myfile.cd()
                h_invmass.Write()
                h_invmass_weight.Write()
                histmult = TH1F("hmultpt%dmult%d" % (ipt, ibin2),
                                "hmultpt%dmult%d"  % (ipt, ibin2), 1000, 0, 1000)
                fill_hist(histmult, df_bin.n_tracklets_corr)
                histmult.Write()
                h_v0m_ntrackletsD = TH2F("h_v0m_ntrackletsD%d%d" % (ibin2, ipt),
                                         "h_v0m_ntrackletsD%d%d" % (ibin2, ipt),
                                         200, 0, 200, 200, -0.5, 1999.5)
                v_v0m_ntrackletsD = np.vstack((df_bin.n_tracklets_corr,
                                               df_bin.v0m_corr)).T
                fill_hist(h_v0m_ntrackletsD, v_v0m_ntrackletsD)
                h_v0m_ntrackletsD.Write()
                if "pt_jet" in df_bin.columns:
                    zarray = z_calc(df_bin.pt_jet, df_bin.phi_jet, df_bin.eta_jet,
                                    df_bin.pt_cand, df_bin.phi_cand, df_bin.eta_cand)
                    h_zvsinvmass = TH2F("hzvsmass" + suffix, "", 5000, 1.00, 6.00, 2000, -0.5, 1.5)
                    zvsinvmass = np.vstack((df_bin.inv_mass, zarray)).T
                    fill_hist(h_zvsinvmass, zvsinvmass)
                    h_zvsinvmass.Write()

                if self.mcordata == "mc":
                    df_bin[self.v_ismcrefl] = np.array(tag_bit_df(df_bin, self.v_bitvar,
                                                                  self.b_mcrefl), dtype=int)
                    df_bin_sig = df_bin[df_bin[self.v_ismcsignal] == 1]
                    df_bin_refl = df_bin[df_bin[self.v_ismcrefl] == 1]
                    h_invmass_sig = TH1F("hmass_sig" + suffix, "", self.p_num_bins,
                                         self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    h_invmass_refl = TH1F("hmass_refl" + suffix, "", self.p_num_bins,
                                          self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    fill_hist(h_invmass_sig, df_bin_sig.inv_mass)
                    fill_hist(h_invmass_refl, df_bin_refl.inv_mass)
                    myfile.cd()
                    h_invmass_sig.Write()
                    h_invmass_refl.Write()
                    print("FINISHED")


    # pylint: disable=line-too-long
    def process_efficiency_single(self, index):
        out_file = TFile.Open(self.l_histoeff[index], "recreate")
        for ibin2 in range(len(self.lvar2_binmin)):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning_gen, \
                                        self.lvar2_binmin[ibin2], \
                                        self.lvar2_binmax[ibin2])
            n_bins = len(self.lpt_finbinmin)
            analysis_bin_lims_temp = self.lpt_finbinmin.copy()
            analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
            analysis_bin_lims = array.array('f', analysis_bin_lims_temp)
            h_gen_pr = TH1F("h_gen_pr" + stringbin2, "Prompt Generated in acceptance |y|<0.5", \
                            n_bins, analysis_bin_lims)
            h_presel_pr = TH1F("h_presel_pr" + stringbin2, "Prompt Reco in acc |#eta|<0.8 and sel", \
                               n_bins, analysis_bin_lims)
            h_sel_pr = TH1F("h_sel_pr" + stringbin2, "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                            n_bins, analysis_bin_lims)
            h_gen_fd = TH1F("h_gen_fd" + stringbin2, "FD Generated in acceptance |y|<0.5", \
                            n_bins, analysis_bin_lims)
            h_presel_fd = TH1F("h_presel_fd" + stringbin2, "FD Reco in acc |#eta|<0.8 and sel", \
                               n_bins, analysis_bin_lims)
            h_sel_fd = TH1F("h_sel_fd" + stringbin2, "FD Reco and sel in acc |#eta|<0.8 and sel", \
                            n_bins, analysis_bin_lims)

            bincounter = 0
            for ipt in range(self.p_nptfinbins):
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
                df_mc_gen = selectdfrunlist(df_mc_gen, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning_gen, \
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning_gen, \
                                            self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
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
                if self.doml is True:
                    df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[bin_id])
                else:
                    df_reco_sel_fd = df_reco_presel_fd.copy()

                if self.corr_eff_mult[ibin2] is True:
                    val, err = self.get_reweighted_count(df_gen_sel_pr)
                    h_gen_pr.SetBinContent(bincounter + 1, val)
                    h_gen_pr.SetBinError(bincounter + 1, err)
                    val, err = self.get_reweighted_count(df_reco_presel_pr)
                    h_presel_pr.SetBinContent(bincounter + 1, val)
                    h_presel_pr.SetBinError(bincounter + 1, err)
                    val, err = self.get_reweighted_count(df_reco_sel_pr)
                    h_sel_pr.SetBinContent(bincounter + 1, val)
                    h_sel_pr.SetBinError(bincounter + 1, err)
                    #print("prompt efficiency tot ptbin=", bincounter, ", value = ",
                    #      len(df_reco_sel_pr)/len(df_gen_sel_pr))

                    val, err = self.get_reweighted_count(df_gen_sel_fd)
                    h_gen_fd.SetBinContent(bincounter + 1, val)
                    h_gen_fd.SetBinError(bincounter + 1, err)
                    val, err = self.get_reweighted_count(df_reco_presel_fd)
                    h_presel_fd.SetBinContent(bincounter + 1, val)
                    h_presel_fd.SetBinError(bincounter + 1, err)
                    val, err = self.get_reweighted_count(df_reco_sel_fd)
                    h_sel_fd.SetBinContent(bincounter + 1, val)
                    h_sel_fd.SetBinError(bincounter + 1, err)
                    #print("fd efficiency tot ptbin=", bincounter, ", value = ",
                    #      len(df_reco_sel_fd)/len(df_gen_sel_fd))
                else:
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


    def process_response(self):
        list_df_mc_reco = []
        list_df_mc_gen = []
        for iptskim, _ in enumerate(self.lpt_anbinmin):
            df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged[iptskim], "rb"))
            if "pt_jet" not in df_mc_reco.columns:
                print("Jet variables not found in the dataframe. Skipping process_response.")
                return
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_trigger is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger)
            df_mc_reco = selectdfrunlist(df_mc_reco, \
                  self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            if self.doml is True:
                df_mc_reco = df_mc_reco.query(self.l_selml[iptskim])
            else:
                print("Doing std analysis")
            list_df_mc_reco.append(df_mc_reco)
            df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[iptskim], "rb"))
            df_mc_gen = selectdfrunlist(df_mc_gen, \
                    self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
            list_df_mc_gen.append(df_mc_gen)
        df_rec = pd.concat(list_df_mc_reco)
        df_gen = pd.concat(list_df_mc_gen)
        his_njets = TH1F("his_njets_gen", "Number of MC jets", 1, 0, 1)
        his_njets.SetBinContent(1, len(df_gen.index)) # total number of generated & selected jets for normalisation
        df_rec = df_rec[df_rec.ismcfd == 1] # reconstructed & selected non-prompt jets
        df_gen = df_gen[df_gen.ismcfd == 1] # generated & selected non-prompt jets
        out_file = TFile.Open(self.n_fileeff, "update")

        # Bin arrays
        # pt_cand
        n_bins_ptc = len(self.lpt_finbinmin)
        bins_ptc_temp = self.lpt_finbinmin.copy()
        bins_ptc_temp.append(self.lpt_finbinmax[n_bins_ptc - 1])
        bins_ptc = array.array('d', bins_ptc_temp)
        # pt_jet
        n_bins_ptjet = len(self.lvar2_binmin)
        bins_ptjet_temp = self.lvar2_binmin.copy()
        bins_ptjet_temp.append(self.lvar2_binmax[n_bins_ptjet - 1])
        bins_ptjet = array.array('d', bins_ptjet_temp)
        # z
        bins_z_temp = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        n_bins_z = len(bins_z_temp) - 1
        bins_z = array.array('d', bins_z_temp)

        # Detector response matrix of pt_jet of non-prompt jets
        df_resp_jet_fd = df_rec.loc[:, ["pt_gen_jet", "pt_jet"]]
        his_resp_jet_fd = TH2F("his_resp_jet_fd", \
            "Response matrix of #it{p}_{T}^{jet, ch} of non-prompt jets;#it{p}_{T}^{jet, ch, gen.} (GeV/#it{c});#it{p}_{T}^{jet, ch, rec.} (GeV/#it{c})", \
            100, 0, 100, 100, 0, 100)
        fill_hist(his_resp_jet_fd, df_resp_jet_fd)

        # Simulated pt_cand vs. pt_jet of non-prompt jets
        df_ptc_ptjet_fd = df_gen.loc[:, ["pt_cand", "pt_jet"]]
        his_ptc_ptjet_fd = TH2F("his_ptc_ptjet_fd", \
            "Simulated #it{p}_{T}^{cand.} vs. #it{p}_{T}^{jet} of non-prompt jets;#it{p}_{T}^{cand., gen.} (GeV/#it{c});#it{p}_{T}^{jet, ch, gen.} (GeV/#it{c})", \
            n_bins_ptc, bins_ptc, 100, 0, 100)
        fill_hist(his_ptc_ptjet_fd, df_ptc_ptjet_fd)

        # z_gen of reconstructed feed-down jets (for response)
        arr_z_gen_resp = z_gen_calc(df_rec.pt_gen_jet, df_rec.phi_gen_jet, df_rec.eta_gen_jet,
                                    df_rec.pt_gen_cand, df_rec.delta_phi_gen_jet, df_rec.delta_eta_gen_jet)
        # z_rec of reconstructed feed-down jets (for response)
        arr_z_rec_resp = z_calc(df_rec.pt_jet, df_rec.phi_jet, df_rec.eta_jet,
                                df_rec.pt_cand, df_rec.phi_cand, df_rec.eta_cand)
        # z_gen of simulated feed-down jets
        arr_z_gen_sim = z_calc(df_gen.pt_jet, df_gen.phi_jet, df_gen.eta_jet,
                               df_gen.pt_cand, df_gen.phi_cand, df_gen.eta_cand)
        df_rec["z_gen"] = arr_z_gen_resp
        df_rec["z"] = arr_z_rec_resp
        df_gen["z"] = arr_z_gen_sim

        # Simulated pt_cand vs. pt_jet vs z of non-prompt jets
        df_ptc_ptjet_z_fd = df_gen.loc[:, ["pt_cand", "pt_jet", "z"]]
        his_ptc_ptjet_z_fd = TH3F("his_ptc_ptjet_z_fd", \
            "Simulated #it{p}_{T}^{cand.} vs. #it{p}_{T}^{jet} vs. #it{z} of non-prompt jets;"
            "#it{p}_{T}^{cand., gen.} (GeV/#it{c});"
            "#it{p}_{T}^{jet, ch, gen.} (GeV/#it{c});"
            "#it{z}", \
            n_bins_ptc, bins_ptc, n_bins_ptjet, bins_ptjet, n_bins_z, bins_z)
        fill_hist(his_ptc_ptjet_z_fd, df_ptc_ptjet_z_fd)

        # Create response matrix for feed-down smearing
        # x axis = z, y axis = pt_jet
        his_resp_rec = TH2F("his_resp_rec", "his_resp_rec", n_bins_z, bins_z, n_bins_ptjet, bins_ptjet)
        his_resp_gen = TH2F("his_resp_gen", "his_resp_gen", n_bins_z, bins_z, n_bins_ptjet, bins_ptjet)
        resp_z = RooUnfoldResponse(his_resp_rec, his_resp_gen)
        for row in df_rec.itertuples():
            resp_z.Fill(row.z, row.pt_jet, row.z_gen, row.pt_gen_jet)

        out_file.cd()
        his_resp_jet_fd.Write()
        his_ptc_ptjet_fd.Write()
        his_ptc_ptjet_z_fd.Write()
        his_njets.Write()
        resp_z.Write("resp_z")
        out_file.Close()

