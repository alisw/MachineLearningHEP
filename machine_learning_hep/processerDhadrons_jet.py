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
from ROOT import TFile, TH1F, TH2F, TH3F, RooUnfold, RooUnfoldResponse, RooUnfoldBayes, TRandom3 # pylint: disable=import-error, no-name-in-module
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

        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]

        self.lvar2_binmin_reco = datap["analysis"][self.typean].get("sel_binmin2_reco", None)
        self.lvar2_binmax_reco = datap["analysis"][self.typean].get("sel_binmax2_reco", None)
        self.p_nbin2_reco = len(self.lvar2_binmin_reco)

        self.lvar2_binmin_gen = datap["analysis"][self.typean].get("sel_binmin2_gen", None)
        self.lvar2_binmax_gen = datap["analysis"][self.typean].get("sel_binmax2_gen", None)
        self.p_nbin2_gen = len(self.lvar2_binmin_gen)

        self.lvarshape_binmin_reco = datap["analysis"][self.typean].get("sel_binminshape_reco", None)
        self.lvarshape_binmax_reco = datap["analysis"][self.typean].get("sel_binmaxshape_reco", None)
        self.p_nbinshape_reco = len(self.lvarshape_binmin_reco)

        self.lvarshape_binmin_gen = datap["analysis"][self.typean].get("sel_binminshape_gen", None)
        self.lvarshape_binmax_gen = datap["analysis"][self.typean].get("sel_binmaxshape_gen", None)
        self.p_nbinshape_gen = len(self.lvarshape_binmin_gen)

        self.closure_frac = datap["analysis"][self.typean].get("sel_closure_frac", None)

        self.var2ranges_reco = self.lvar2_binmin_reco.copy()
        self.var2ranges_reco.append(self.lvar2_binmax_reco[-1])
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        self.varshaperanges_reco = self.lvarshape_binmin_reco.copy()
        self.varshaperanges_reco.append(self.lvarshape_binmax_reco[-1])
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])

        self.doprior = datap["analysis"][self.typean]["doprior"]

        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        #self.sel_final_fineptbins = datap["analysis"][self.typean]["sel_final_fineptbins"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_jetsel_gen = datap["analysis"][self.typean]["jetsel_gen"]
        self.s_jetsel_reco = datap["analysis"][self.typean]["jetsel_reco"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger


    # pylint: disable=too-many-branches
    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            if self.doml is True:
                df = df.query(self.l_selml[bin_id])
            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_jetsel_reco is not None:
                df = df.query(self.s_jetsel_reco)
            if self.s_trigger is not None:
                df = df.query(self.s_trigger)

            h_invmass_all = TH1F("hmass_%d" % ipt, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
            fill_hist(h_invmass_all, df.inv_mass)
            myfile.cd()
            h_invmass_all.Write()

            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            for ibin2 in range(len(self.lvar2_binmin_reco)):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                df_bin = seldf_singlevar(df, self.v_var2_binning,
                                         self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                df_bin = selectdfrunlist(df_bin, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
                fill_hist(h_invmass, df_bin.inv_mass)
                myfile.cd()
                h_invmass.Write()
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
                    #print("FINISHED")


    # pylint: disable=line-too-long
    def process_efficiency_single(self, index):
        out_file = TFile.Open(self.l_histoeff[index], "recreate")
        for ibin2 in range(len(self.lvar2_binmin_reco)):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                        self.lvar2_binmin_reco[ibin2], \
                                        self.lvar2_binmax_reco[ibin2])
            n_bins = self.p_nptfinbins
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
                if self.s_jetsel_reco is not None:
                    df_mc_reco = df_mc_reco.query(self.s_jetsel_reco)
                if self.s_trigger is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger)
                df_mc_reco = selectdfrunlist(df_mc_reco, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
                df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[bin_id][index], "rb"))
                df_mc_gen = df_mc_gen.query(self.s_jetsel_gen)
                df_mc_gen = selectdfrunlist(df_mc_gen, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning, \
                                             self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning, \
                                            self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
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
        out_file = TFile.Open(self.n_fileeff, "update")
        list_df_mc_reco = []
        list_df_mc_gen = []
        for iptskim, _ in enumerate(self.lpt_anbinmin):

            df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[iptskim], "rb"))
            df_mc_gen = selectdfrunlist(df_mc_gen, \
                    self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_mc_gen = df_mc_gen.query(self.s_jetsel_gen)
            list_df_mc_gen.append(df_mc_gen)

            df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged[iptskim], "rb"))
            if "pt_jet" not in df_mc_reco.columns:
                print("Jet variables not found in the dataframe. Skipping process_response.")
                return
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_jetsel_reco is not None:
                df_mc_reco = df_mc_reco.query(self.s_jetsel_reco)
            if self.s_trigger is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger)
            df_mc_reco = df_mc_reco.query(self.l_selml[iptskim])
            list_df_mc_reco.append(df_mc_reco)

        zbin_reco=[]
        nzbin_reco=self.p_nbinshape_reco
        zbin_reco =self.varshaperanges_reco
        zbinarray_reco=array.array('d',zbin_reco)

        zbin_gen =[]
        nzbin_gen=self.p_nbinshape_gen
        zbin_gen = self.varshaperanges_gen
        zbinarray_gen=array.array('d',zbin_gen)

        jetptbin_reco =[]
        njetptbin_reco=self.p_nbin2_reco
        jetptbin_reco = self.var2ranges_reco
        jetptbinarray_reco=array.array('d',jetptbin_reco)

        jetptbin_gen =[]
        njetptbin_gen=self.p_nbin2_gen
        jetptbin_gen = self.var2ranges_gen
        jetptbinarray_gen=array.array('d',jetptbin_gen)

        candptbin=[]
        ncandptbin=self.p_nptfinbins
        candptbin=self.lpt_finbinmin.copy()
        candptbin.append(self.lpt_finbinmax[-1])
        candptbinarray=array.array('d',candptbin)

        df_gen = pd.concat(list_df_mc_gen)
        df_gen_nonprompt = df_gen[df_gen.ismcfd == 1] # reconstructed & selected non-prompt jets


        z_array_gen_unmatched = z_calc(df_gen_nonprompt.pt_jet, df_gen_nonprompt.phi_jet, df_gen_nonprompt.eta_jet,
                               df_gen_nonprompt.pt_cand, df_gen_nonprompt.phi_cand, df_gen_nonprompt.eta_cand)
        df_gen_nonprompt["z"] = z_array_gen_unmatched
        df_zvsjetpt_gen_unmatched = df_gen_nonprompt.loc[:, ["z", "pt_jet"]]
        hzvsjetptvscandpt_gen_nonprompt = TH3F("hzvsjetptvscandpt_gen_nonprompt", "hzvsjetptvscandpt_gen_nonprompt",nzbin_gen,zbinarray_gen,njetptbin_gen,jetptbinarray_gen,ncandptbin,candptbinarray)
        for row in df_gen_nonprompt.itertuples():
            hzvsjetptvscandpt_gen_nonprompt.Fill(row.z,row.pt_jet,row.pt_cand)


        df_mc_reco_merged = pd.concat(list_df_mc_reco)


        df_mc_reco_merged_nonprompt = df_mc_reco_merged[df_mc_reco_merged.ismcfd == 1] # reconstructed & selected non-prompt jets

        zarray_reco = z_calc(df_mc_reco_merged_nonprompt.pt_jet, df_mc_reco_merged_nonprompt.phi_jet, df_mc_reco_merged_nonprompt.eta_jet,
                                    df_mc_reco_merged_nonprompt.pt_cand, df_mc_reco_merged_nonprompt.phi_cand, df_mc_reco_merged_nonprompt.eta_cand)

        zarray_gen = z_gen_calc(df_mc_reco_merged_nonprompt.pt_gen_jet, df_mc_reco_merged_nonprompt.phi_gen_jet, df_mc_reco_merged_nonprompt.eta_gen_jet,
                                    df_mc_reco_merged_nonprompt.pt_gen_cand, df_mc_reco_merged_nonprompt.delta_phi_gen_jet, df_mc_reco_merged_nonprompt.delta_eta_gen_jet)

        df_mc_reco_merged_nonprompt['z_reco'] = zarray_reco
        df_mc_reco_merged_nonprompt['z_gen'] = zarray_gen


        hzvsjetpt_reco=TH2F("hzvsjetpt_reco_nonprompt","hzvsjetpt_reco_nonprompt",nzbin_reco,zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco.Sumw2()
        hzvsjetpt_gen=TH2F("hzvsjetpt_genv","hzvsjetpt_gen_nonprompt",nzbin_gen,zbinarray_gen,njetptbin_gen,jetptbinarray_gen)
        hzvsjetpt_gen.Sumw2()

        response_matrix = RooUnfoldResponse(hzvsjetpt_reco, hzvsjetpt_gen)


        hz_gen_nocuts_list=[]
        hz_gen_cuts_list=[]
        for ibin2 in range(len(self.lvar2_binmin_gen)):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hz_gen_nocuts=TH1F("hz_gen_nocuts_nonprompt" + suffix,"hz_gen_nocuts_nonprompt" + suffix,nzbin_gen, zbinarray_gen)
            hz_gen_nocuts.Sumw2()
            hz_gen_nocuts_list.append(hz_gen_nocuts)
            hz_gen_cuts=TH1F("hz_gen_cuts_nonprompt" + suffix,"hz_gen_cuts_nonprompt" + suffix,nzbin_gen,zbinarray_gen)
            hz_gen_cuts.Sumw2()
            hz_gen_cuts_list.append(hz_gen_cuts)




        hjetpt_genvsreco_list=[]
        hz_genvsreco_list=[]
        hjetpt_genvsreco_full=TH2F("hjetpt_genvsreco_full_nonprompt","hjetpt_genvsreco_full_nonprompt",njetptbin_gen*100,self.lvar2_binmin_gen[0],self.lvar2_binmax_gen[-1],njetptbin_reco*100,self.lvar2_binmin_reco[0],self.lvar2_binmax_reco[-1])
        hz_genvsreco_full=TH2F("hz_genvsreco_full_nonprompt","hz_genvsreco_full_nonprompt",nzbin_gen*100,self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1],nzbin_reco*100,self.lvarshape_binmin_reco[0],self.lvarshape_binmax_reco[-1])

        for ibinshape in range(len(self.lvarshape_binmin_reco)):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco=TH2F("hjetpt_genvsreco_nonprompt"+suffix,"hjetpt_genvsreco_nonprompt"+suffix,njetptbin_gen*100,self.lvar2_binmin_gen[0],self.lvar2_binmax_gen[-1],njetptbin_reco*100,self.lvar2_binmin_reco[0],self.lvar2_binmax_reco[-1])
            hjetpt_genvsreco_list.append(hjetpt_genvsreco)



        for ibin2 in range(len(self.lvar2_binmin_reco)):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            hz_genvsreco=TH2F("hz_genvsreco_nonprompt"+suffix,"hz_genvsreco_nonprompt"+suffix,nzbin_gen*100,self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1],nzbin_reco*100,self.lvarshape_binmin_reco[0],self.lvarshape_binmax_reco[-1])
            hz_genvsreco_list.append(hz_genvsreco)




        hjetpt_fracdiff_list=[]
        hz_fracdiff_list=[]

        for ibin2 in range(len(self.lvar2_binmin_gen)):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff = TH1F("hjetpt_fracdiff_nonprompt" +suffix,"hjetpt_fracdiff_nonprompt" +suffix,100,-2,2)
            hjetpt_fracdiff_list.append(hjetpt_fracdiff)

        for ibinshape in range(len(self.lvarshape_binmin_gen)):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff = TH1F("hz_fracdiff_nonprompt" +suffix,"hz_fracdiff_nonprompt" +suffix,100,-2,2)
            hz_fracdiff_list.append(hz_fracdiff)

        hzvsjetpt_reco_nocuts=TH2F("hzvsjetpt_reco_nocuts_nonprompt","hzvsjetpt_reco_nocuts_nonprompt",nzbin_reco, zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco_nocuts.Sumw2()
        hzvsjetpt_reco_cuts=TH2F("hzvsjetpt_reco_cuts_nonprompt","hzvsjetpt_reco_cuts_nonprompt",nzbin_reco, zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco_cuts.Sumw2()

        hzvsjetpt_gen_nocuts=TH2F("hzvsjetpt_gen_nocuts_nonprompt","hzvsjetpt_gen_nocuts_nonprompt",nzbin_gen, zbinarray_gen,njetptbin_gen,jetptbinarray_gen)
        hzvsjetpt_gen_nocuts.Sumw2()
        hzvsjetpt_gen_cuts=TH2F("hzvsjetpt_gen_cuts_nonprompt","hzvsjetpt_gen_cuts_nonprompt",nzbin_gen, zbinarray_gen,njetptbin_gen,jetptbinarray_gen)
        hzvsjetpt_gen_cuts.Sumw2()

        for row in df_mc_reco_merged_nonprompt.itertuples():

            if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1]:
                hzvsjetpt_reco.Fill(row.z_reco,row.pt_jet)
                hzvsjetpt_reco_nocuts.Fill(row.z_reco,row.pt_jet)
                if row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1]:
                    hzvsjetpt_reco_cuts.Fill(row.z_reco,row.pt_jet)

            if row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1]:
                hzvsjetpt_gen.Fill(row.z_gen,row.pt_gen_jet)
                hzvsjetpt_gen_nocuts.Fill(row.z_gen,row.pt_gen_jet)
                if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1]:
                    hzvsjetpt_gen_cuts.Fill(row.z_gen,row.pt_gen_jet)

            if row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1]:
                if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1]:
                    response_matrix.Fill(row.z_reco,row.pt_jet,row.z_gen,row.pt_gen_jet)
                    hjetpt_genvsreco_full.Fill(row.pt_gen_jet,row.pt_jet)
                    hz_genvsreco_full.Fill(row.z_gen,row.z_reco)
                    for ibin2 in range(len(self.lvar2_binmin_reco)):
                        if row.pt_jet >= self.lvar2_binmin_reco[ibin2] and row.pt_jet < self.lvar2_binmax_reco[ibin2]:
                            hz_genvsreco_list[ibin2].Fill(row.z_gen,row.z_reco)
                    for ibinshape in range(len(self.lvarshape_binmin_reco)):
                        if row.z_reco >= self.lvarshape_binmin_reco[ibinshape] and row.z_reco < self.lvarshape_binmax_reco[ibinshape]:
                            hjetpt_genvsreco_list[ibinshape].Fill(row.pt_gen_jet,row.pt_jet)

            for ibin2 in range(len(self.lvar2_binmin_gen)):
                if row.pt_gen_jet >= self.lvar2_binmin_gen[ibin2] and row.pt_gen_jet < self.lvar2_binmax_gen[ibin2]:
                    hjetpt_fracdiff_list[ibin2].Fill((row.pt_jet-row.pt_gen_jet)/row.pt_gen_jet)

            for ibinshape in range(len(self.lvarshape_binmin_gen)):
                if row.z_gen >= self.lvarshape_binmin_gen[ibinshape] and row.z_gen < self.lvarshape_binmax_gen[ibinshape]:
                    hz_fracdiff_list[ibinshape].Fill((row.z_reco-row.z_gen)/row.z_gen)

            for ibin2 in range(len(self.lvar2_binmin_gen)):
                if row.pt_gen_jet >= self.lvar2_binmin_gen[ibin2] and row.pt_gen_jet < self.lvar2_binmax_gen[ibin2] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1] :
                    hz_gen_nocuts_list[ibin2].Fill(row.z_gen)
                    if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1] :
                        hz_gen_cuts_list[ibin2].Fill(row.z_gen)

        for ibin2 in range(len(self.lvar2_binmin_gen)):
            hz_gen_nocuts_list[ibin2].Write()
            hz_gen_cuts_list[ibin2].Write()
            hjetpt_fracdiff_list[ibin2].Scale(1.0/hjetpt_fracdiff_list[ibin2].Integral(1,-1))
            hjetpt_fracdiff_list[ibin2].Write()
        for ibinshape in range(len(self.lvarshape_binmin_gen)):
            hz_fracdiff_list[ibinshape].Scale(1.0/hz_fracdiff_list[ibinshape].Integral(1,-1))
            hz_fracdiff_list[ibinshape].Write()
        for ibin2 in range(len(self.lvar2_binmin_reco)):
            hz_genvsreco_list[ibin2].Scale(1.0/hz_genvsreco_list[ibin2].Integral(1,-1,1,-1))
            hz_genvsreco_list[ibin2].Write()
        for ibinshape in range(len(self.lvarshape_binmin_reco)):
            hjetpt_genvsreco_list[ibinshape].Scale(1.0/hjetpt_genvsreco_list[ibinshape].Integral(1,-1,1,-1))
            hjetpt_genvsreco_list[ibinshape].Write()
        hz_genvsreco_full.Scale(1.0/hz_genvsreco_full.Integral(1,-1,1,-1))
        hz_genvsreco_full.Write()
        hjetpt_genvsreco_full.Scale(1.0/hjetpt_genvsreco_full.Integral(1,-1,1,-1))
        hjetpt_genvsreco_full.Write()
        hzvsjetpt_reco.Write()
        hzvsjetpt_gen.Write()
        hzvsjetptvscandpt_gen_nonprompt.Write()
        response_matrix.Write("response_matrix_nonprompt")
        hzvsjetpt_reco_nocuts.Write()
        hzvsjetpt_reco_cuts.Write()
        hzvsjetpt_gen_nocuts.Write()
        hzvsjetpt_gen_cuts.Write()
        out_file.Close()

    # pylint: disable=too-many-locals
    def process_unfolding(self):
        out_file = TFile.Open(self.n_fileeff, "update")
        list_df_mc_reco = []
        list_df_mc_gen = []
        for iptskim, _ in enumerate(self.lpt_anbinmin):

            df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[iptskim], "rb"))
            df_mc_gen = selectdfrunlist(df_mc_gen, \
                    self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_mc_gen = df_mc_gen.query(self.s_jetsel_gen)
            list_df_mc_gen.append(df_mc_gen)

            df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged[iptskim], "rb"))
            if "pt_jet" not in df_mc_reco.columns:
                print("Jet variables not found in the dataframe. Skipping process_response.")
                return
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_jetsel_reco is not None:
                df_mc_reco = df_mc_reco.query(self.s_jetsel_reco)
            if self.s_trigger is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger)
            df_mc_reco = df_mc_reco.query(self.l_selml[iptskim])
            list_df_mc_reco.append(df_mc_reco)

        zbin_reco=[]
        nzbin_reco=self.p_nbinshape_reco
        zbin_reco =self.varshaperanges_reco
        zbinarray_reco=array.array('d',zbin_reco)

        zbin_gen =[]
        nzbin_gen=self.p_nbinshape_gen
        zbin_gen = self.varshaperanges_gen
        zbinarray_gen=array.array('d',zbin_gen)

        jetptbin_reco =[]
        njetptbin_reco=self.p_nbin2_reco
        jetptbin_reco = self.var2ranges_reco
        jetptbinarray_reco=array.array('d',jetptbin_reco)

        jetptbin_gen =[]
        njetptbin_gen=self.p_nbin2_gen
        jetptbin_gen = self.var2ranges_gen
        jetptbinarray_gen=array.array('d',jetptbin_gen)


        df_gen = pd.concat(list_df_mc_gen)
        df_gen_prompt = df_gen[df_gen.ismcprompt == 1] # reconstructed & selected non-prompt jets


        z_array_gen_unmatched = z_calc(df_gen_prompt.pt_jet, df_gen_prompt.phi_jet, df_gen_prompt.eta_jet,
                               df_gen_prompt.pt_cand, df_gen_prompt.phi_cand, df_gen_prompt.eta_cand)
        df_gen_prompt["z_gen"] = z_array_gen_unmatched
        df_zvsjetpt_gen_unmatched = df_gen_prompt.loc[:, ["z_gen", "pt_jet"]]
        hzvsjetpt_gen_unmatched = TH2F("hzvsjetpt_gen_unmatched", "hzvsjetpt_gen_unmatched",nzbin_gen,zbinarray_gen,njetptbin_gen,jetptbinarray_gen)
        fill_hist(hzvsjetpt_gen_unmatched, df_zvsjetpt_gen_unmatched)



        df_mc_reco_merged = pd.concat(list_df_mc_reco)


        df_mc_reco_merged_prompt = df_mc_reco_merged[df_mc_reco_merged.ismcprompt == 1] # reconstructed & selected non-prompt jets

        zarray_reco = z_calc(df_mc_reco_merged_prompt.pt_jet, df_mc_reco_merged_prompt.phi_jet, df_mc_reco_merged_prompt.eta_jet,
                                    df_mc_reco_merged_prompt.pt_cand, df_mc_reco_merged_prompt.phi_cand, df_mc_reco_merged_prompt.eta_cand)

        zarray_gen = z_gen_calc(df_mc_reco_merged_prompt.pt_gen_jet, df_mc_reco_merged_prompt.phi_gen_jet, df_mc_reco_merged_prompt.eta_gen_jet,
                                    df_mc_reco_merged_prompt.pt_gen_cand, df_mc_reco_merged_prompt.delta_phi_gen_jet, df_mc_reco_merged_prompt.delta_eta_gen_jet)

        df_mc_reco_merged_prompt['z_reco'] = zarray_reco
        df_mc_reco_merged_prompt['z_gen'] = zarray_gen


        hzvsjetpt_reco_closure=TH2F("hzvsjetpt_reco_closure","hzvsjetpt_reco_closure",nzbin_reco,zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco_closure.Sumw2()
        hzvsjetpt_gen_closure=TH2F("hzvsjetpt_gen_closure","hzvsjetpt_gen_closure",nzbin_gen,zbinarray_gen,njetptbin_gen,jetptbinarray_gen)
        hzvsjetpt_gen_closure.Sumw2()

        hzvsjetpt_reco=TH2F("hzvsjetpt_reco","hzvsjetpt_reco",nzbin_reco,zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco.Sumw2()
        hzvsjetpt_gen=TH2F("hzvsjetpt_gen","hzvsjetpt_gen",nzbin_gen,zbinarray_gen,njetptbin_gen,jetptbinarray_gen)
        hzvsjetpt_gen.Sumw2()

        response_matrix = RooUnfoldResponse(hzvsjetpt_reco, hzvsjetpt_gen)
        response_matrix_closure = RooUnfoldResponse(hzvsjetpt_reco, hzvsjetpt_gen)


        hz_gen_nocuts_list=[]
        hz_gen_cuts_list=[]
        hz_gen_nocuts_list_closure=[]
        hz_gen_cuts_list_closure=[]

        for ibin2 in range(len(self.lvar2_binmin_gen)):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hz_gen_nocuts=TH1F("hz_gen_nocuts" + suffix,"hz_gen_nocuts" + suffix,nzbin_gen, zbinarray_gen)
            hz_gen_nocuts.Sumw2()
            hz_gen_nocuts_list.append(hz_gen_nocuts)
            hz_gen_cuts=TH1F("hz_gen_cuts" + suffix,"hz_gen_cuts" + suffix,nzbin_gen,zbinarray_gen)
            hz_gen_cuts.Sumw2()
            hz_gen_cuts_list.append(hz_gen_cuts)

            hz_gen_nocuts_closure=TH1F("hz_gen_nocuts_closure" + suffix,"hz_gen_nocuts_closure" + suffix,nzbin_gen, zbinarray_gen)
            hz_gen_nocuts_closure.Sumw2()
            hz_gen_nocuts_list_closure.append(hz_gen_nocuts_closure)
            hz_gen_cuts_closure=TH1F("hz_gen_cuts_closure" + suffix,"hz_gen_cuts_closure" + suffix,nzbin_gen,zbinarray_gen)
            hz_gen_cuts_closure.Sumw2()
            hz_gen_cuts_list_closure.append(hz_gen_cuts_closure)



        hjetpt_gen_nocuts=TH1F("hjetpt_gen_nocuts","hjetpt_gen_nocuts",njetptbin_gen, jetptbinarray_gen)
        hjetpt_gen_nocuts.Sumw2()
        hjetpt_gen_cuts=TH1F("hjetpt_gen_cuts","hjetpt_gen_cuts",njetptbin_gen,jetptbinarray_gen)
        hjetpt_gen_cuts.Sumw2()

        hjetpt_gen_nocuts_closure=TH1F("hjetpt_gen_nocuts_closure","hjetpt_gen_nocuts_closure",njetptbin_gen, jetptbinarray_gen)
        hjetpt_gen_nocuts_closure.Sumw2()
        hjetpt_gen_cuts_closure=TH1F("hjetpt_gen_cuts_closure","hjetpt_gen_cuts_closure",njetptbin_gen,jetptbinarray_gen)
        hjetpt_gen_cuts_closure.Sumw2()





        hzvsjetpt_reco_nocuts=TH2F("hzvsjetpt_reco_nocuts","hzvsjetpt_reco_nocuts",nzbin_reco, zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco_nocuts.Sumw2()
        hzvsjetpt_reco_cuts=TH2F("hzvsjetpt_reco_cuts","hzvsjetpt_reco_cuts",nzbin_reco, zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco_cuts.Sumw2()

        hzvsjetpt_reco_nocuts_closure=TH2F("hzvsjetpt_reco_nocuts_closure","hzvsjetpt_reco_nocuts_closure",nzbin_reco, zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco_nocuts_closure.Sumw2()
        hzvsjetpt_reco_cuts_closure=TH2F("hzvsjetpt_reco_cuts_closure","hzvsjetpt_reco_cuts_closure",nzbin_reco, zbinarray_reco,njetptbin_reco,jetptbinarray_reco)
        hzvsjetpt_reco_cuts_closure.Sumw2()



        hjetpt_genvsreco_list=[]
        hz_genvsreco_list=[]
        hjetpt_genvsreco_full=TH2F("hjetpt_genvsreco_full","hjetpt_genvsreco_full",njetptbin_gen*100,self.lvar2_binmin_gen[0],self.lvar2_binmax_gen[-1],njetptbin_reco*100,self.lvar2_binmin_reco[0],self.lvar2_binmax_reco[-1])
        hz_genvsreco_full=TH2F("hz_genvsreco_full","hz_genvsreco_full",nzbin_gen*100,self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1],nzbin_reco*100,self.lvarshape_binmin_reco[0],self.lvarshape_binmax_reco[-1])

        for ibinshape in range(len(self.lvarshape_binmin_reco)):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco=TH2F("hjetpt_genvsreco"+suffix,"hjetpt_genvsreco"+suffix,njetptbin_gen*100,self.lvar2_binmin_gen[0],self.lvar2_binmax_gen[-1],njetptbin_reco*100,self.lvar2_binmin_reco[0],self.lvar2_binmax_reco[-1])
            hjetpt_genvsreco_list.append(hjetpt_genvsreco)



        for ibin2 in range(len(self.lvar2_binmin_reco)):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            hz_genvsreco=TH2F("hz_genvsreco"+suffix,"hz_genvsreco"+suffix,nzbin_gen*100,self.lvarshape_binmin_gen[0],self.lvarshape_binmax_gen[-1],nzbin_reco*100,self.lvarshape_binmin_reco[0],self.lvarshape_binmax_reco[-1])
            hz_genvsreco_list.append(hz_genvsreco)


        hjetpt_fracdiff_list=[]
        hz_fracdiff_list=[]

        for ibin2 in range(len(self.lvar2_binmin_gen)):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff = TH1F("hjetpt_fracdiff_prompt" +suffix,"hjetpt_fracdiff_prompt" +suffix,100,-2,2)
            hjetpt_fracdiff_list.append(hjetpt_fracdiff)

        for ibinshape in range(len(self.lvarshape_binmin_gen)):
            suffix = "z_%.2f_%.2f" % \
                     (self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff = TH1F("hz_fracdiff_prompt" +suffix,"hz_fracdiff_prompt" +suffix,100,-2,2)
            hz_fracdiff_list.append(hz_fracdiff)



        hzvsjetpt_prior_weights=TH2F("hzvsjetpt_prior_weights","hzvsjetpt_prior_weights",nzbin_gen,zbinarray_gen,njetptbin_gen,jetptbinarray_gen)
        hzvsjetpt_prior_weights.Sumw2()
        if self.doprior is True:
            for row in df_mc_reco_merged_prompt.itertuples():
                if row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1]:
                    if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1]:
                        hzvsjetpt_prior_weights.Fill(row.z_gen,row.pt_gen_jet)

        random_number = TRandom3(0)
        random_number_result=0.0
        response_matrix_weight = 1.0
        for row in df_mc_reco_merged_prompt.itertuples():

            random_number_result=random_number.Rndm()
            random_number_result_weights=random_number.Rndm()
            if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1]:
                hzvsjetpt_reco.Fill(row.z_reco,row.pt_jet)
                hzvsjetpt_reco_nocuts.Fill(row.z_reco,row.pt_jet)
                if random_number_result < self.closure_frac :
                    hzvsjetpt_reco_nocuts_closure.Fill(row.z_reco,row.pt_jet)
                if row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1]:
                    hzvsjetpt_reco_cuts.Fill(row.z_reco,row.pt_jet)
                    if random_number_result < self.closure_frac :
                        hzvsjetpt_reco_cuts_closure.Fill(row.z_reco,row.pt_jet)
            if row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1]:
                hzvsjetpt_gen.Fill(row.z_gen,row.pt_gen_jet)
            if row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1]:
                if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1]:
                    response_matrix_weight=1.0
                    if self.doprior is True:
                        if hzvsjetpt_prior_weights.GetBinContent(hzvsjetpt_prior_weights.GetXaxis().FindBin(row.z_gen),hzvsjetpt_prior_weights.GetYaxis().FindBin(row.pt_gen_jet)) > 0.0 :
                            response_matrix_weight=1.0/hzvsjetpt_prior_weights.GetBinContent(hzvsjetpt_prior_weights.GetXaxis().FindBin(row.z_gen),hzvsjetpt_prior_weights.GetYaxis().FindBin(row.pt_gen_jet))
                    response_matrix.Fill(row.z_reco,row.pt_jet,row.z_gen,row.pt_gen_jet,response_matrix_weight)
                    hjetpt_genvsreco_full.Fill(row.pt_gen_jet,row.pt_jet)
                    hz_genvsreco_full.Fill(row.z_gen,row.z_reco)
                    for ibin2 in range(len(self.lvar2_binmin_reco)):
                        if row.pt_jet >= self.lvar2_binmin_reco[ibin2] and row.pt_jet < self.lvar2_binmax_reco[ibin2]:
                            hz_genvsreco_list[ibin2].Fill(row.z_gen,row.z_reco)
                    for ibinshape in range(len(self.lvarshape_binmin_reco)):
                        if row.z_reco >= self.lvarshape_binmin_reco[ibinshape] and row.z_reco < self.lvarshape_binmax_reco[ibinshape]:
                            hjetpt_genvsreco_list[ibinshape].Fill(row.pt_gen_jet,row.pt_jet)


            for ibin2 in range(len(self.lvar2_binmin_gen)):
                if row.pt_gen_jet >= self.lvar2_binmin_gen[ibin2] and row.pt_gen_jet < self.lvar2_binmax_gen[ibin2]:
                    hjetpt_fracdiff_list[ibin2].Fill((row.pt_jet-row.pt_gen_jet)/row.pt_gen_jet)

            for ibinshape in range(len(self.lvarshape_binmin_gen)):
                if row.z_gen >= self.lvarshape_binmin_gen[ibinshape] and row.z_gen < self.lvarshape_binmax_gen[ibinshape]:
                    hz_fracdiff_list[ibinshape].Fill((row.z_reco-row.z_gen)/row.z_gen)

            for ibin2 in range(len(self.lvar2_binmin_gen)):
                if row.pt_gen_jet >= self.lvar2_binmin_gen[ibin2] and row.pt_gen_jet < self.lvar2_binmax_gen[ibin2] and row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1] :
                    hz_gen_nocuts_list[ibin2].Fill(row.z_gen)
                    if random_number_result < self.closure_frac :
                        hz_gen_nocuts_list_closure[ibin2].Fill(row.z_gen)
                    if row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] and row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1] :
                        hz_gen_cuts_list[ibin2].Fill(row.z_gen)
                        if random_number_result < self.closure_frac :
                            hz_gen_cuts_list_closure[ibin2].Fill(row.z_gen)

            if row.z_gen >= self.lvarshape_binmin_gen[0] and row.z_gen < self.lvarshape_binmax_gen[-1] and row.pt_gen_jet >= self.lvar2_binmin_gen[0] and row.pt_gen_jet < self.lvar2_binmax_gen[-1] :
                hjetpt_gen_nocuts.Fill(row.pt_gen_jet)
                if random_number_result < self.closure_frac :
                    hjetpt_gen_nocuts_closure.Fill(row.pt_gen_jet)
                if row.z_reco >= self.lvarshape_binmin_reco[0] and row.z_reco < self.lvarshape_binmax_reco[-1] and row.pt_jet >= self.lvar2_binmin_reco[0] and row.pt_jet < self.lvar2_binmax_reco[-1] :
                    hjetpt_gen_cuts.Fill(row.pt_gen_jet)
                    if random_number_result < self.closure_frac :
                        hjetpt_gen_cuts_closure.Fill(row.pt_gen_jet)

            if random_number_result < self.closure_frac :
                hzvsjetpt_reco_closure.Fill(row.z_reco,row.pt_jet)
                hzvsjetpt_gen_closure.Fill(row.z_gen,row.pt_gen_jet)
            else:
                response_matrix_weight=1.0
                if self.doprior is True:
                    if hzvsjetpt_prior_weights.GetBinContent(hzvsjetpt_prior_weights.GetXaxis().FindBin(row.z_gen),hzvsjetpt_prior_weights.GetYaxis().FindBin(row.pt_gen_jet)) > 0.0 :
                        response_matrix_weight=1.0/hzvsjetpt_prior_weights.GetBinContent(hzvsjetpt_prior_weights.GetXaxis().FindBin(row.z_gen),hzvsjetpt_prior_weights.GetYaxis().FindBin(row.pt_gen_jet))
                response_matrix_closure.Fill(row.z_reco,row.pt_jet,row.z_gen,row.pt_gen_jet,response_matrix_weight)

        for ibin2 in range(len(self.lvar2_binmin_gen)):
            hz_gen_nocuts_list[ibin2].Write()
            hz_gen_cuts_list[ibin2].Write()
            hz_gen_nocuts_list_closure[ibin2].Write()
            hz_gen_cuts_list_closure[ibin2].Write()
            hjetpt_fracdiff_list[ibin2].Scale(1.0/hjetpt_fracdiff_list[ibin2].Integral(1,-1))
            hjetpt_fracdiff_list[ibin2].Write()
        for ibinshape in range(len(self.lvarshape_binmin_gen)):
            hz_fracdiff_list[ibinshape].Scale(1.0/hz_fracdiff_list[ibinshape].Integral(1,-1))
            hz_fracdiff_list[ibinshape].Write()
        for ibin2 in range(len(self.lvar2_binmin_reco)):
            hz_genvsreco_list[ibin2].Scale(1.0/hz_genvsreco_list[ibin2].Integral(1,-1,1,-1))
            hz_genvsreco_list[ibin2].Write()
        for ibinshape in range(len(self.lvarshape_binmin_reco)):
            hjetpt_genvsreco_list[ibin2].Scale(1.0/hjetpt_genvsreco_list[ibinshape].Integral(1,-1,1,-1))
            hjetpt_genvsreco_list[ibinshape].Write()
        hjetpt_gen_nocuts.Write()
        hjetpt_gen_cuts.Write()
        hjetpt_gen_nocuts_closure.Write()
        hjetpt_gen_cuts_closure.Write()
        hz_genvsreco_full.Scale(1.0/hz_genvsreco_full.Integral(1,-1,1,-1))
        hz_genvsreco_full.Write()
        hjetpt_genvsreco_full.Scale(1.0/hjetpt_genvsreco_full.Integral(1,-1,1,-1))
        hjetpt_genvsreco_full.Write()
        hzvsjetpt_reco.Write()
        hzvsjetpt_gen.Write()
        hzvsjetpt_gen_unmatched.Write()
        hzvsjetpt_reco_nocuts.Write()
        hzvsjetpt_reco_cuts.Write()
        hzvsjetpt_reco_nocuts_closure.Write()
        hzvsjetpt_reco_cuts_closure.Write()
        response_matrix.Write("response_matrix")
        response_matrix_closure.Write("response_matrix_closure")
        hzvsjetpt_reco_closure.Write("input_closure_reco")
        hzvsjetpt_gen_closure.Write("input_closure_gen")
        out_file.Close()
