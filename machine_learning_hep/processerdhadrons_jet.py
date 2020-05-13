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
from array import array
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F, TH2F, RooUnfoldResponse # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.bitwise import tag_bit_df
from machine_learning_hep.utilities import selectdfrunlist, seldf_singlevar, openfile
from machine_learning_hep.utilities import create_folder_struc, mergerootfiles, get_timestamp_string
from machine_learning_hep.utilities import z_calc, z_gen_calc
from machine_learning_hep.utilities_plot import buildhisto, build2dhisto, fill2dhist, makefill3dhist
from machine_learning_hep.processer import Processer
#from machine_learning_hep.selectionutils import selectpid_dzerotokpi

def apply_cut_selpid(df_):
    df_["is_selpid"] = np.array(tag_bit_df(df_, "cand_type", [[7], []]), dtype=int)
    df_sel = df_[df_["is_selpid"] == 1]
    return df_sel

def apply_cut_selstd(df_):
    df_["is_selstd"] = np.array(tag_bit_df(df_, "cand_type", [[0], []]), dtype=int)
    df_sel = df_[df_["is_selstd"] == 1]
    return df_sel

class ProcesserDhadrons_jet(Processer): # pylint: disable=invalid-name, too-many-instance-attributes
    # Class Attribute
    species = "processer"

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments, line-too-long
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

        self.mass = datap["mass"]
        self.p_mass_fit_lim = datap["analysis"][self.typean]["mass_fit_lim"]
        self.p_bin_width = datap["analysis"][self.typean]["bin_width"]
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        self.l_selml = ["y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[ipt]) \
                       for ipt in range(self.p_nptbins)]

        # first variable (hadron pt)
        self.v_var_binning = datap["var_binning"] # name
        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin) # number of bins
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.var1ranges = self.lpt_finbinmin.copy()
        self.var1ranges.append(self.lpt_finbinmax[-1])
        self.var1binarray = array("d", self.var1ranges) # array of bin edges to use in histogram constructors

        # second variable (jet pt)
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"] # name
        self.v_var2_binning_gen = datap["analysis"][self.typean]["var_binning2_gen"] # name
        self.lvar2_binmin_reco = datap["analysis"][self.typean].get("sel_binmin2_reco", None)
        self.lvar2_binmax_reco = datap["analysis"][self.typean].get("sel_binmax2_reco", None)
        self.p_nbin2_reco = len(self.lvar2_binmin_reco) # number of reco bins
        self.lvar2_binmin_gen = datap["analysis"][self.typean].get("sel_binmin2_gen", None)
        self.lvar2_binmax_gen = datap["analysis"][self.typean].get("sel_binmax2_gen", None)
        self.p_nbin2_gen = len(self.lvar2_binmin_gen) # number of gen bins
        self.var2ranges_reco = self.lvar2_binmin_reco.copy()
        self.var2ranges_reco.append(self.lvar2_binmax_reco[-1])
        self.var2binarray_reco = array("d", self.var2ranges_reco) # array of bin edges to use in histogram constructors
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        self.var2binarray_gen = array("d", self.var2ranges_gen) # array of bin edges to use in histogram constructors

        # observable (z, shape,...)
        self.v_varshape_binning = datap["analysis"][self.typean]["var_binningshape"] # name (reco)
        self.v_varshape_binning_gen = datap["analysis"][self.typean]["var_binningshape_gen"] # name (gen)
        self.lvarshape_binmin_reco = \
            datap["analysis"][self.typean].get("sel_binminshape_reco", None)
        self.lvarshape_binmax_reco = \
            datap["analysis"][self.typean].get("sel_binmaxshape_reco", None)
        self.p_nbinshape_reco = len(self.lvarshape_binmin_reco) # number of reco bins
        self.lvarshape_binmin_gen = \
            datap["analysis"][self.typean].get("sel_binminshape_gen", None)
        self.lvarshape_binmax_gen = \
            datap["analysis"][self.typean].get("sel_binmaxshape_gen", None)
        self.p_nbinshape_gen = len(self.lvarshape_binmin_gen) # number of gen bins
        self.varshaperanges_reco = self.lvarshape_binmin_reco.copy()
        self.varshaperanges_reco.append(self.lvarshape_binmax_reco[-1])
        self.varshapebinarray_reco = array("d", self.varshaperanges_reco) # array of bin edges to use in histogram constructors
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])
        self.varshapebinarray_gen = array("d", self.varshaperanges_gen) # array of bin edges to use in histogram constructors

        self.closure_frac = datap["analysis"][self.typean].get("sel_closure_frac", None)
        self.doprior = datap["analysis"][self.typean]["doprior"]

        # selection
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_jetsel_gen = datap["analysis"][self.typean]["jetsel_gen"]
        self.s_jetsel_reco = datap["analysis"][self.typean]["jetsel_reco"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger

        self.p_usejetptbinned_deff = \
            datap["analysis"][self.typean].get("usejetptbinned_deff", False)
        print("use jet binned efficiency", self.p_usejetptbinned_deff)
    # pylint: disable=too-many-branches
    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")

        # Get number of selected events and save it in the first bin of the histonorm histogram.

        dfevtorig = pickle.load(openfile(self.l_evtorig[index], "rb"))
        if self.s_trigger is not None:
            dfevtorig = dfevtorig.query(self.s_trigger)
        if self.runlistrigger is not None:
            dfevtorig = selectdfrunlist(dfevtorig, self.run_param[self.runlistrigger], "run_number")
        dfevtevtsel = dfevtorig.query(self.s_evtsel)
        neventsafterevtsel = len(dfevtevtsel)
        histonorm = TH1F("histonorm", "histonorm", 1, 0, 1)
        histonorm.SetBinContent(1, neventsafterevtsel)
        myfile.cd()
        histonorm.Write()

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            if self.do_custom_analysis_cuts:
                df = pickle.load(openfile(self.mptfiles_recosk[bin_id][index], "rb"))
                df["imp_par_prod"] = df["imp_par_prod"].astype(float) # allow negative cut values
                #df["nsigTOF_Pi_0"] = df["nsigTOF_Pi_0"].astype(float) # allow negative cut values
                #df["nsigTOF_K_0"] = df["nsigTOF_K_0"].astype(float) # allow negative cut values
                #df["nsigTOF_Pi_1"] = df["nsigTOF_Pi_1"].astype(float) # allow negative cut values
                #df["nsigTOF_K_1"] = df["nsigTOF_K_1"].astype(float) # allow negative cut values
            else:
                df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            if self.doml is True:
                df = df.query(self.l_selml[bin_id])
            # custom cuts
            elif self.do_custom_analysis_cuts:
                entries_all = len(df)
                print("Entries before any cuts:", len(df))
                # PID cut
                df = apply_cut_selpid(df)
                entries_pid = len(df)
                print("Entries after PID cut:", len(df))
                # Custom cuts
                df = self.apply_cuts_ptbin(df, ipt)
                entries_cuts = len(df)
                print("Entries after custom cuts:", len(df))
                # Standard selection
                df = apply_cut_selstd(df)
                entries_std = len(df)
                print("Entries after STD sel cut:", len(df))
                print("All: %g, PID: %g, Custom: %g, Std: %g" % (entries_all, entries_pid, entries_cuts, entries_std))
                if entries_cuts != entries_std:
                    print("Cut effective: %g" % (entries_cuts / entries_std))
                #df["is_selpid"] = np.array(tag_bit_df(df, "cand_type", [[7], []]), dtype=int)
                #df = df[df["is_selpid"] == 1]
                #isselpid = selectpid_dzerotokpi(df["nsigTPC_Pi_0"].values, df["nsigTPC_K_0"].values, \
                #    df["nsigTOF_Pi_0"].values, df["nsigTOF_K_0"].values, \
                #    df["nsigTPC_Pi_1"].values, df["nsigTPC_K_1"].values, \
                #    df["nsigTOF_Pi_1"].values, df["nsigTOF_K_1"].values, 3) # FIXME pylint: disable=fixme
                #df = df[np.array(isselpid, dtype=bool)]
                #print("Entries after PID cut:", len(df))
                #string_sel_pid = "((abs(nsigTPC_Pi_0) < 3 and abs(nsigTOF_Pi_0) < 3 and nsigTOF_Pi_0 > -999)"
                #string_sel_pid += " or (abs(nsigTPC_K_0) < 3 and abs(nsigTOF_K_0) < 3 and nsigTOF_K_0 > -999))"
                #string_sel_pid += " and ((abs(nsigTPC_Pi_1) < 3 and abs(nsigTOF_Pi_1) < 3 and nsigTOF_Pi_1 > -999)"
                #string_sel_pid += " or (abs(nsigTPC_K_1) < 3 and abs(nsigTOF_K_1) < 3 and nsigTOF_K_1 > -999))"
                #df = df.query(string_sel_pid)
                #print("Entries after manual PID cut:", len(df))
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
            for ibin2 in range(self.p_nbin2_reco):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[ibin2],
                          self.lvar2_binmax_reco[ibin2])
                df_bin = seldf_singlevar(df, self.v_var2_binning,
                                         self.lvar2_binmin_reco[ibin2],
                                         self.lvar2_binmax_reco[ibin2])
                if self.runlistrigger is not None:
                    df_bin = selectdfrunlist(df_bin, \
                             self.run_param[self.runlistrigger], "run_number")

                # add the z column
                df_bin["z"] = z_calc(df_bin.pt_jet, df_bin.phi_jet, df_bin.eta_jet,
                                     df_bin.pt_cand, df_bin.phi_cand, df_bin.eta_cand)

                # Reject single-constituent jets, should be "ntracks_jet > 1" if available
                jetsel_string = "z < 1" # FIXME pylint: disable=fixme
                df_bin = df_bin.query(jetsel_string)

                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                fill_hist(h_invmass, df_bin.inv_mass)
                myfile.cd()
                h_invmass.Write()

                massarray = [1.0 + i * (5.0 / 5000.0) for i in range(5001)] # 5000 bins in range 1.0-6.0
                massarray_reco = array("d", massarray)
                h_zvsinvmass = TH2F("hzvsmass" + suffix, "", \
                    5000, massarray_reco, self.p_nbinshape_reco, self.varshapebinarray_reco)
                h_zvsinvmass.Sumw2()
                fill2dhist(df_bin, h_zvsinvmass, "inv_mass", self.v_varshape_binning)
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


    # pylint: disable=line-too-long
    def process_efficiency_single(self, index):
        out_file = TFile.Open(self.l_histoeff[index], "recreate")

        # matched rec level jets
        h2_ptcand_ptjet_rec_overflow = build2dhisto("h2_ptcand_ptjet_rec_overflow", self.var1binarray, self.var2binarray_reco)
        h2_ptcand_ptjet_rec = build2dhisto("h2_ptcand_ptjet_rec", self.var1binarray, self.var2binarray_reco)
        # matched gen level jets
        h2_ptcand_ptjet_genmatched_overflow = build2dhisto("h2_ptcand_ptjet_genmatched_overflow", self.var1binarray, self.var2binarray_gen)
        h2_ptcand_ptjet_genmatched = build2dhisto("h2_ptcand_ptjet_genmatched", self.var1binarray, self.var2binarray_gen)

        # selected gen level jets
        h3_shape_ptjet_ptcand_gen = buildhisto("h3_shape_ptjet_ptcand_gen", "h3_shape_ptjet_ptcand_gen", \
            self.varshapebinarray_gen, self.var2binarray_gen, self.var1binarray)

        for ibinshape in range(self.p_nbinshape_gen):
            for ibin2 in range(self.p_nbin2_gen):
                for ipt in range(self.p_nptfinbins):
                    bin_id = self.bin_matching[ipt]
                    df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[bin_id][index], "rb"))
                    df_mc_gen = df_mc_gen.query(self.s_jetsel_gen)
                    if self.runlistrigger is not None:
                        df_mc_gen = selectdfrunlist(df_mc_gen, \
                                 self.run_param[self.runlistrigger], "run_number")
                    df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                                self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                    df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning, \
                                                self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
                    df_mc_gen = seldf_singlevar(df_mc_gen, self.v_varshape_binning, \
                                                self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])

                    # Reject single-constituent jets, should be "ntracks_jet > 1" if available
                    df_mc_gen["z"] = z_calc(df_mc_gen.pt_jet, df_mc_gen.phi_jet, df_mc_gen.eta_jet,
                                            df_mc_gen.pt_cand, df_mc_gen.phi_cand, df_mc_gen.eta_cand) # FIXME pylint: disable=fixme
                    jetsel_string = "z < 1"
                    df_mc_gen = df_mc_gen.query(jetsel_string)

                    df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]

                    val = len(df_gen_sel_pr)
                    h3_shape_ptjet_ptcand_gen.SetBinContent(ibinshape + 1, ibin2 + 1, ipt + 1, val)
                    h3_shape_ptjet_ptcand_gen.SetBinError(ibinshape + 1, ibin2 + 1, ipt + 1, 0)
                    #for _ in range(val):
                    #    h3_shape_ptjet_ptcand_gen.Fill(self.lvarshape_binmin_gen[ibinshape],
                    #                                    self.lvar2_binmin_gen[ibin2],
                    #                                    self.lpt_finbinmin[ipt])

        out_file.cd()
        print("h3_shape_ptjet_ptcand_gen Integral:", h3_shape_ptjet_ptcand_gen.Integral())
        h3_shape_ptjet_ptcand_gen.Write()

        for ibin2 in range(self.p_nbin2_reco): # FIXME Do gen and rec loops separately pylint: disable=fixme
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                        self.lvar2_binmin_reco[ibin2], \
                                        self.lvar2_binmax_reco[ibin2])

            h_gen_pr = TH1F("h_gen_pr" + stringbin2, "Prompt Generated in acceptance |y|<0.5", \
                            self.p_nptfinbins, self.var1binarray)
            h_presel_pr = TH1F("h_presel_pr" + stringbin2, "Prompt Reco in acc |#eta|<0.8 and sel", \
                               self.p_nptfinbins, self.var1binarray)
            h_sel_pr = TH1F("h_sel_pr" + stringbin2, "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                            self.p_nptfinbins, self.var1binarray)
            h_gen_fd = TH1F("h_gen_fd" + stringbin2, "FD Generated in acceptance |y|<0.5", \
                            self.p_nptfinbins, self.var1binarray)
            h_presel_fd = TH1F("h_presel_fd" + stringbin2, "FD Reco in acc |#eta|<0.8 and sel", \
                               self.p_nptfinbins, self.var1binarray)
            h_sel_fd = TH1F("h_sel_fd" + stringbin2, "FD Reco and sel in acc |#eta|<0.8 and sel", \
                            self.p_nptfinbins, self.var1binarray)

            bincounter = 0
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                if self.do_custom_analysis_cuts:
                    df_mc_reco = pickle.load(openfile(self.mptfiles_recosk[bin_id][index], "rb"))
                    df_mc_reco["imp_par_prod"] = df_mc_reco["imp_par_prod"].astype(float) # allow negative cut values
                else:
                    df_mc_reco = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
                if self.s_evtsel is not None:
                    df_mc_reco = df_mc_reco.query(self.s_evtsel)
                if self.s_jetsel_reco is not None:
                    if self.p_usejetptbinned_deff is True:
                        df_mc_reco = df_mc_reco.query(self.s_jetsel_reco)
                if self.s_trigger is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger)
                if self.runlistrigger is not None:
                    df_mc_reco = selectdfrunlist(df_mc_reco, \
                             self.run_param[self.runlistrigger], "run_number")
                df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[bin_id][index], "rb"))
                if self.p_usejetptbinned_deff is True:
                    df_mc_gen = df_mc_gen.query(self.s_jetsel_gen)
                if self.runlistrigger is not None:
                    df_mc_gen = selectdfrunlist(df_mc_gen, \
                             self.run_param[self.runlistrigger], "run_number")
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                if self.p_usejetptbinned_deff is True:
                    df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning, \
                        self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                    df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning, \
                        self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2]) # FIXME use lvar2_binmin_gen pylint: disable=fixme

                df_mc_gen["z"] = z_calc(df_mc_gen.pt_jet, df_mc_gen.phi_jet, df_mc_gen.eta_jet,
                                        df_mc_gen.pt_cand, df_mc_gen.phi_cand, df_mc_gen.eta_cand)

                df_mc_reco["z"] = z_calc(df_mc_reco.pt_jet, df_mc_reco.phi_jet, df_mc_reco.eta_jet,
                                         df_mc_reco.pt_cand, df_mc_reco.phi_cand, df_mc_reco.eta_cand)

                # Reject single-constituent jets, should be "ntracks_jet > 1" if available
                jetsel_string = "z < 1" # FIXME pylint: disable=fixme
                if self.p_usejetptbinned_deff is True:
                    df_mc_gen = df_mc_gen.query(jetsel_string)
                    df_mc_reco = df_mc_reco.query(jetsel_string)

                # prompt
                df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
                df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
                df_reco_sel_pr = None
                if self.doml is True:
                    df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[bin_id])
                # custom cuts
                elif self.do_custom_analysis_cuts:
                    df_reco_sel_pr = self.apply_cuts_ptbin(df_reco_presel_pr, ipt)
                    df_reco_sel_pr = apply_cut_selpid(df_reco_sel_pr)
                else:
                    df_reco_sel_pr = df_reco_presel_pr.copy()
                # restrict shape range
                df_reco_sel_pr_no_overflow = seldf_singlevar(df_reco_sel_pr, \
                    self.v_varshape_binning, self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])
                # non-prompt
                df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
                df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
                df_reco_sel_fd = None
                if self.doml is True:
                    df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[bin_id])
                # custom cuts
                elif self.do_custom_analysis_cuts:
                    df_reco_sel_fd = self.apply_cuts_ptbin(df_reco_presel_fd, ipt)
                    df_reco_sel_fd = apply_cut_selpid(df_reco_sel_fd)
                else:
                    df_reco_sel_fd = df_reco_presel_fd.copy()

                val = len(df_gen_sel_pr)
                err = math.sqrt(val)
                h_gen_pr.SetBinContent(bincounter + 1, val)
                h_gen_pr.SetBinError(bincounter + 1, err) # FIXME Why do we set error for gen? pylint: disable=fixme
                val = len(df_reco_presel_pr)
                err = math.sqrt(val)
                h_presel_pr.SetBinContent(bincounter + 1, val)
                h_presel_pr.SetBinError(bincounter + 1, err)
                val = len(df_reco_sel_pr)
                err = math.sqrt(val)
                h_sel_pr.SetBinContent(bincounter + 1, val)
                h_sel_pr.SetBinError(bincounter + 1, err)
                h2_ptcand_ptjet_rec_overflow.SetBinContent(ipt + 1, ibin2 + 1, val)
                h2_ptcand_ptjet_rec_overflow.SetBinError(ipt + 1, ibin2 + 1, err)
                val = len(df_reco_sel_pr_no_overflow)
                err = math.sqrt(val)
                h2_ptcand_ptjet_rec.SetBinContent(ipt + 1, ibin2 + 1, val)
                h2_ptcand_ptjet_rec.SetBinError(ipt + 1, ibin2 + 1, err)

                val = len(df_gen_sel_fd)
                err = math.sqrt(val)
                h_gen_fd.SetBinContent(bincounter + 1, val)
                h_gen_fd.SetBinError(bincounter + 1, err) # FIXME Why do we set error for gen? pylint: disable=fixme
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

        # loop over gen bins
        for ibin2 in range(self.p_nbin2_gen):
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                if self.do_custom_analysis_cuts:
                    df_mc_reco = pickle.load(openfile(self.mptfiles_recosk[bin_id][index], "rb"))
                    df_mc_reco["imp_par_prod"] = df_mc_reco["imp_par_prod"].astype(float) # allow negative cut values
                else:
                    df_mc_reco = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
                if self.s_evtsel is not None:
                    df_mc_reco = df_mc_reco.query(self.s_evtsel)
                if self.s_jetsel_reco is not None:
                    df_mc_reco = df_mc_reco.query(self.s_jetsel_reco) # FIXME should it be gen here? pylint: disable=fixme
                if self.s_trigger is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger)
                if self.runlistrigger is not None:
                    df_mc_reco = selectdfrunlist(df_mc_reco, \
                    self.run_param[self.runlistrigger], "run_number")
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                    self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                if self.doml is True:
                    df_mc_reco = df_mc_reco.query(self.l_selml[bin_id])
                # custom cuts
                elif self.do_custom_analysis_cuts:
                    df_mc_reco = self.apply_cuts_ptbin(df_mc_reco, ipt)
                    df_mc_reco = apply_cut_selpid(df_mc_reco)
                # select pt_gen_jet bin
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning_gen, \
                    self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])

                df_mc_reco["z"] = z_calc(df_mc_reco.pt_jet, df_mc_reco.phi_jet, df_mc_reco.eta_jet,
                                         df_mc_reco.pt_cand, df_mc_reco.phi_cand, df_mc_reco.eta_cand)

                # Reject single-constituent jets, should be "ntracks_jet > 1" if available
                jetsel_string = "z < 1" # FIXME pylint: disable=fixme
                df_mc_reco = df_mc_reco.query(jetsel_string)

                # restrict gen shape range
                df_reco_no_overflow = seldf_singlevar(df_mc_reco, \
                    self.v_varshape_binning_gen, self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1])

                # prompt
                df_reco_pr_overflow = df_mc_reco[df_mc_reco.ismcprompt == 1]
                df_reco_pr = df_reco_no_overflow[df_reco_no_overflow.ismcprompt == 1]
                # non-prompt
                #df_reco_fd_overflow = df_mc_reco[df_mc_reco.ismcfd == 1]
                #df_reco_fd = df_reco_no_overflow[df_reco_no_overflow.ismcfd == 1]

                val = len(df_reco_pr_overflow)
                err = math.sqrt(val)
                h2_ptcand_ptjet_genmatched_overflow.SetBinContent(ipt + 1, ibin2 + 1, val)
                h2_ptcand_ptjet_genmatched_overflow.SetBinError(ipt + 1, ibin2 + 1, err)
                val = len(df_reco_pr)
                err = math.sqrt(val)
                h2_ptcand_ptjet_genmatched.SetBinContent(ipt + 1, ibin2 + 1, val)
                h2_ptcand_ptjet_genmatched.SetBinError(ipt + 1, ibin2 + 1, err)

        out_file.cd()
        h2_ptcand_ptjet_rec_overflow.Write()
        h2_ptcand_ptjet_rec.Write()
        h2_ptcand_ptjet_genmatched_overflow.Write()
        h2_ptcand_ptjet_genmatched.Write()

    def create_df_closure(self, df_):
        df_tmp_selgen = df_.copy()
        df_tmp_selgen = seldf_singlevar(df_tmp_selgen, self.v_varshape_binning_gen, \
            self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1])
        df_tmp_selgen = seldf_singlevar(df_tmp_selgen, "pt_gen_jet", \
            self.lvar2_binmin_gen[0], self.lvar2_binmax_gen[-1])

        df_tmp_selreco = df_.copy()
        df_tmp_selreco = seldf_singlevar(df_tmp_selreco, "pt_jet", \
            self.lvar2_binmin_reco[0], self.lvar2_binmax_reco[-1])
        df_tmp_selreco = seldf_singlevar(df_tmp_selreco, self.v_varshape_binning, \
            self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])

        df_tmp_selrecogen = df_tmp_selgen.copy()
        df_tmp_selrecogen = seldf_singlevar(df_tmp_selrecogen, "pt_jet", \
            self.lvar2_binmin_reco[0], self.lvar2_binmax_reco[-1])
        df_tmp_selrecogen = seldf_singlevar(df_tmp_selrecogen, self.v_varshape_binning, \
            self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])

        return df_tmp_selgen, df_tmp_selreco, df_tmp_selrecogen

    def process_response(self):
        print("Doing response", self.mcordata, self.period)
        print("Using run selection for resp histo", \
               self.runlistrigger, "for period", self.period)
        if self.doml is True:
            print("Doing ml analysis")
        elif self.do_custom_analysis_cuts:
            print("Using custom cuts")
        else:
            print("No extra selection needed since we are doing std analysis")

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_response_single, arguments, self.p_chunksizeunp)
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/historesp_{self.period}/{get_timestamp_string()}/" # pylint: disable=line-too-long
        mergerootfiles(self.l_historesp, self.n_fileresp, tmp_merged)

    def process_response_single(self, index): # pylint: disable=too-many-locals
        """
        First of all, we load all the mc gen and reco files that are skimmed
        in bins of HF candidate ptand we apply the standard selection to all
        of them. After this, we merged them all to create a single file of gen
        and reco monte carlo sample with all the HF candidate pt. In particular
        gen jets are selected according to run trigger, runlist, and gen jet
        zbin_recoand pseudorapidity. Reco candidates according to evt selection, eta
        jets, trigger and ml probability of the HF hadron
        """

        out_file = TFile.Open(self.l_historesp[index], "recreate")
        list_df_mc_reco = []
        list_df_mc_gen = []

        for iptskim in range(self.p_nptbins):

            df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[iptskim][index], "rb"))
            if self.runlistrigger is not None:
                df_mc_gen = selectdfrunlist(df_mc_gen, \
                        self.run_param[self.runlistrigger], "run_number")
            df_mc_gen = df_mc_gen.query(self.s_jetsel_gen)
            list_df_mc_gen.append(df_mc_gen)

            if self.do_custom_analysis_cuts:
                df_mc_reco = pickle.load(openfile(self.mptfiles_recosk[iptskim][index], "rb"))
                df_mc_reco["imp_par_prod"] = df_mc_reco["imp_par_prod"].astype(float) # allow negative cut values
            else:
                df_mc_reco = pickle.load(openfile(self.mptfiles_recoskmldec[iptskim][index], "rb"))
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_jetsel_reco is not None:
                df_mc_reco = df_mc_reco.query(self.s_jetsel_reco)
            if self.s_trigger is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger)
            if self.doml is True:
                df_mc_reco = df_mc_reco.query(self.l_selml[iptskim])
            elif self.do_custom_analysis_cuts: # custom cuts, pt bin
                list_df_mc_reco_ipt = []
                for ipt in range(self.p_nptfinbins):
                    df_mc_reco_ipt = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                        self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                    df_mc_reco_ipt = self.apply_cuts_ptbin(df_mc_reco_ipt, ipt)
                    list_df_mc_reco_ipt.append(df_mc_reco_ipt)
                df_mc_reco = pd.concat(list_df_mc_reco_ipt)
                df_mc_reco = apply_cut_selpid(df_mc_reco)
            list_df_mc_reco.append(df_mc_reco)

        # Here we can merge the dataframes corresponding to different HF pt in a
        # single one. In addition we are here selecting only non prompt HF

        df_gen = pd.concat(list_df_mc_gen)
        df_mc_reco = pd.concat(list_df_mc_reco)

        # add the z columns
        df_gen["z"] = z_calc(df_gen.pt_jet, df_gen.phi_jet, df_gen.eta_jet,
                             df_gen.pt_cand, df_gen.phi_cand, df_gen.eta_cand)

        df_mc_reco["z"] = z_calc(df_mc_reco.pt_jet, df_mc_reco.phi_jet, df_mc_reco.eta_jet,
                                 df_mc_reco.pt_cand, df_mc_reco.phi_cand, df_mc_reco.eta_cand)

        df_mc_reco["z_gen"] = z_gen_calc(df_mc_reco.pt_gen_jet, df_mc_reco.phi_gen_jet,
                                         df_mc_reco.eta_gen_jet, df_mc_reco.pt_gen_cand,
                                         df_mc_reco.delta_phi_gen_jet, df_mc_reco.delta_eta_gen_jet)

        # Reject single-constituent jets, should be "ntracks_jet > 1" if available
        jetsel_string = "z < 1" # FIXME pylint: disable=fixme
        df_gen = df_gen.query(jetsel_string)
        df_mc_reco = df_mc_reco.query(jetsel_string)

        df_gen_nonprompt = df_gen[df_gen.ismcfd == 1]
        df_gen_prompt = df_gen[df_gen.ismcprompt == 1]
        df_mc_reco_merged_nonprompt = df_mc_reco[df_mc_reco.ismcfd == 1]
        df_mc_reco_merged_prompt = df_mc_reco[df_mc_reco.ismcprompt == 1]

        # The following plots are 3d plots all at generated level of z,
        # pt_jet and pt_cand. This was used in the first version of the feeddown
        # subtraction, currently is obsolete

        hzvsjetpt_gen_unmatched = TH2F("hzvsjetpt_gen_unmatched", "hzvsjetpt_gen_unmatched", \
            self.p_nbinshape_gen, self.varshapebinarray_gen, self.p_nbin2_gen, self.var2binarray_gen)
        df_zvsjetpt_gen_unmatched = df_gen_prompt.loc[:, [self.v_varshape_binning, "pt_jet"]]
        fill_hist(hzvsjetpt_gen_unmatched, df_zvsjetpt_gen_unmatched)
        hzvsjetpt_gen_unmatched.Write()

        # 3D histogram for gen non-prompt
        titlehist = "hzvsjetptvscandpt_gen_nonprompt"
        hzvsjetptvscandpt_gen_nonprompt = makefill3dhist(df_gen_nonprompt, titlehist, \
            self.varshapebinarray_gen, self.var2binarray_gen, self.var1binarray, self.v_varshape_binning, "pt_jet", "pt_cand")
        hzvsjetptvscandpt_gen_nonprompt.Write()

        # 3D histogram for gen prompt
        titlehist = "hzvsjetptvscandpt_gen_prompt"
        hzvsjetptvscandpt_gen_prompt = makefill3dhist(df_gen_prompt, titlehist, \
            self.varshapebinarray_gen, self.var2binarray_gen, self.var1binarray, \
            self.v_varshape_binning, "pt_jet", "pt_cand") # FIXME Why do we set error for gen? pylint: disable=fixme
        print("hzvsjetptvscandpt_gen_prompt Integral:", hzvsjetptvscandpt_gen_prompt.Integral())
        hzvsjetptvscandpt_gen_prompt.Write()

        # hz_gen_nocuts is the distribution of generated z values in b in
        # bins of gen_jet pt before the reco z and jetpt selection. hz_gen_cuts
        # also includes cut on z reco and jet pt reco. These are used for overall
        # efficiency correction to estimate the fraction of candidates that are
        # in the reco range but outside the gen range and viceversa

        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hz_gen_nocuts = TH1F("hz_gen_nocuts_nonprompt" + suffix, \
                "hz_gen_nocuts_nonprompt" + suffix, self.p_nbinshape_gen, self.varshapebinarray_gen)
            hz_gen_nocuts.Sumw2()
            hz_gen_cuts = TH1F("hz_gen_cuts_nonprompt" + suffix,
                               "hz_gen_cuts_nonprompt" + suffix, self.p_nbinshape_gen, self.varshapebinarray_gen)
            hz_gen_cuts.Sumw2()

            df_tmp = seldf_singlevar(df_mc_reco_merged_nonprompt, "pt_gen_jet", \
                                     self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            df_tmp = seldf_singlevar(df_tmp, self.v_varshape_binning_gen, \
                                     self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1])
            fill_hist(hz_gen_nocuts, df_tmp[self.v_varshape_binning_gen])
            df_tmp = seldf_singlevar(df_tmp, "pt_jet",
                                     self.lvar2_binmin_reco[0], self.lvar2_binmax_reco[-1])
            df_tmp = seldf_singlevar(df_tmp, self.v_varshape_binning,
                                     self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])
            fill_hist(hz_gen_cuts, df_tmp[self.v_varshape_binning_gen])
            hz_gen_cuts.Write()
            hz_gen_nocuts.Write()

            # Addendum for unfolding
            hz_gen_nocuts_pr = TH1F("hz_gen_nocuts" + suffix, \
                "hz_gen_nocuts" + suffix, self.p_nbinshape_gen, self.varshapebinarray_gen)
            hz_gen_nocuts_pr.Sumw2()
            hz_gen_cuts_pr = TH1F("hz_gen_cuts" + suffix,
                                  "hz_gen_cuts" + suffix, self.p_nbinshape_gen, self.varshapebinarray_gen)
            hz_gen_cuts_pr.Sumw2()
            df_tmp_pr = seldf_singlevar(df_mc_reco_merged_prompt, "pt_gen_jet", \
                                     self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            df_tmp_pr = seldf_singlevar(df_tmp_pr, self.v_varshape_binning_gen, \
                                     self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1])
            fill_hist(hz_gen_nocuts_pr, df_tmp_pr[self.v_varshape_binning_gen])
            df_tmp_pr = seldf_singlevar(df_tmp_pr, "pt_jet",
                                        self.lvar2_binmin_reco[0], self.lvar2_binmax_reco[-1])
            df_tmp_pr = seldf_singlevar(df_tmp_pr, self.v_varshape_binning,
                                        self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])
            fill_hist(hz_gen_cuts_pr, df_tmp_pr[self.v_varshape_binning_gen])
            hz_gen_cuts_pr.Write()
            hz_gen_nocuts_pr.Write()
            # End addendum for unfolding


        df_tmp_selgen, df_tmp_selreco, df_tmp_selrecogen = \
                self.create_df_closure(df_mc_reco_merged_nonprompt)

        df_tmp_selgen_pr, df_tmp_selreco_pr, df_tmp_selrecogen_pr = \
                self.create_df_closure(df_mc_reco_merged_prompt)

        # histograms for response of feeddown
        hzvsjetpt_reco_nocuts = \
            build2dhisto("hzvsjetpt_reco_nocuts_nonprompt", self.varshapebinarray_reco, self.var2binarray_reco)
        hzvsjetpt_reco_cuts = \
            build2dhisto("hzvsjetpt_reco_cuts_nonprompt", self.varshapebinarray_reco, self.var2binarray_reco)
        hzvsjetpt_gen_nocuts = \
            build2dhisto("hzvsjetpt_gen_nocuts_nonprompt", self.varshapebinarray_gen, self.var2binarray_gen)
        hzvsjetpt_gen_cuts = \
            build2dhisto("hzvsjetpt_gen_cuts_nonprompt", self.varshapebinarray_gen, self.var2binarray_gen)

        hzvsjetpt_reco = hzvsjetpt_reco_nocuts.Clone("hzvsjetpt_reco_nonprompt")
        hzvsjetpt_gen = hzvsjetpt_gen_nocuts.Clone("hzvsjetpt_genv")
        response_matrix = RooUnfoldResponse(hzvsjetpt_reco, hzvsjetpt_gen)

        fill2dhist(df_tmp_selreco, hzvsjetpt_reco_nocuts, self.v_varshape_binning, "pt_jet")
        fill2dhist(df_tmp_selgen, hzvsjetpt_gen_nocuts, self.v_varshape_binning_gen, "pt_gen_jet")
        fill2dhist(df_tmp_selrecogen, hzvsjetpt_reco_cuts, self.v_varshape_binning, "pt_jet")
        fill2dhist(df_tmp_selrecogen, hzvsjetpt_gen_cuts, self.v_varshape_binning_gen, "pt_gen_jet")

        hzvsjetpt_reco_nocuts.Write()
        hzvsjetpt_gen_nocuts.Write()
        hzvsjetpt_reco_cuts.Write()
        hzvsjetpt_gen_cuts.Write()

        # histograms for unfolding
        hzvsjetpt_reco_nocuts_pr = \
            build2dhisto("hzvsjetpt_reco_nocuts", self.varshapebinarray_reco, self.var2binarray_reco)
        hzvsjetpt_reco_cuts_pr = \
            build2dhisto("hzvsjetpt_reco_cuts", self.varshapebinarray_reco, self.var2binarray_reco)
        hzvsjetpt_gen_nocuts_pr = \
            build2dhisto("hzvsjetpt_gen_nocuts", self.varshapebinarray_gen, self.var2binarray_gen)
        hzvsjetpt_gen_cuts_pr = \
            build2dhisto("hzvsjetpt_gen_cuts", self.varshapebinarray_gen, self.var2binarray_gen)

        fill2dhist(df_tmp_selreco_pr, hzvsjetpt_reco_nocuts_pr, self.v_varshape_binning, "pt_jet")
        fill2dhist(df_tmp_selgen_pr, hzvsjetpt_gen_nocuts_pr, self.v_varshape_binning_gen, "pt_gen_jet")
        fill2dhist(df_tmp_selrecogen_pr, hzvsjetpt_reco_cuts_pr, self.v_varshape_binning, "pt_jet")
        fill2dhist(df_tmp_selrecogen_pr, hzvsjetpt_gen_cuts_pr, self.v_varshape_binning_gen, "pt_gen_jet")
        hzvsjetpt_reco_nocuts_pr.Write()
        hzvsjetpt_gen_nocuts_pr.Write()
        hzvsjetpt_reco_cuts_pr.Write()
        hzvsjetpt_gen_cuts_pr.Write()

        hzvsjetpt_reco_closure_pr = \
            build2dhisto("hzvsjetpt_reco_closure", self.varshapebinarray_reco, self.var2binarray_reco)
        hzvsjetpt_gen_closure_pr = \
            build2dhisto("hzvsjetpt_gen_closure", self.varshapebinarray_reco, self.var2binarray_reco) # FIXME use varshapebinarray_gen pylint: disable=fixme
        hzvsjetpt_reco_pr = \
            build2dhisto("hzvsjetpt_reco", self.varshapebinarray_reco, self.var2binarray_reco)
        hzvsjetpt_gen_pr = \
            build2dhisto("hzvsjetpt_gen", self.varshapebinarray_gen, self.var2binarray_gen)
        response_matrix_pr = RooUnfoldResponse(hzvsjetpt_reco_pr, hzvsjetpt_gen_pr)
        response_matrix_closure_pr = RooUnfoldResponse(hzvsjetpt_reco_pr, hzvsjetpt_gen_pr)

        fill2dhist(df_tmp_selreco_pr, hzvsjetpt_reco_pr, self.v_varshape_binning, "pt_jet")
        fill2dhist(df_tmp_selgen_pr, hzvsjetpt_gen_pr, self.v_varshape_binning_gen, "pt_gen_jet")
        hzvsjetpt_reco_pr.Write()
        hzvsjetpt_gen_pr.Write()

        hjetpt_gen_nocuts_pr = TH1F("hjetpt_gen_nocuts", \
            "hjetpt_gen_nocuts", self.p_nbin2_gen, self.var2binarray_gen)
        hjetpt_gen_cuts_pr = TH1F("hjetpt_gen_cuts", \
            "hjetpt_gen_cuts", self.p_nbin2_gen, self.var2binarray_gen)
        hjetpt_gen_nocuts_closure = TH1F("hjetpt_gen_nocuts_closure", \
            "hjetpt_gen_nocuts_closure", self.p_nbin2_gen, self.var2binarray_gen)
        hjetpt_gen_cuts_closure = TH1F("hjetpt_gen_cuts_closure", \
            "hjetpt_gen_cuts_closure", self.p_nbin2_gen, self.var2binarray_gen)
        hjetpt_gen_nocuts_pr.Sumw2()
        hjetpt_gen_cuts_pr.Sumw2()
        hjetpt_gen_nocuts_closure.Sumw2()
        hjetpt_gen_cuts_closure.Sumw2()

        fill_hist(hjetpt_gen_nocuts_pr, df_tmp_selgen_pr["pt_gen_jet"])
        fill_hist(hjetpt_gen_cuts_pr, df_tmp_selrecogen_pr["pt_gen_jet"])
        hjetpt_gen_nocuts_pr.Write()
        hjetpt_gen_cuts_pr.Write()
        # end of histograms for unfolding

        hjetpt_genvsreco_full = \
            TH2F("hjetpt_genvsreco_full_nonprompt", "hjetpt_genvsreco_full_nonprompt", \
            self.p_nbin2_gen * 100, self.lvar2_binmin_gen[0], self.lvar2_binmax_gen[-1], \
            self.p_nbin2_reco * 100, self.lvar2_binmin_reco[0], self.lvar2_binmax_reco[-1])

        hz_genvsreco_full = \
            TH2F("hz_genvsreco_full_nonprompt", "hz_genvsreco_full_nonprompt", \
                 self.p_nbinshape_gen * 100, self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1],
                 self.p_nbinshape_reco * 100, self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])

        fill2dhist(df_tmp_selrecogen, hjetpt_genvsreco_full, "pt_gen_jet", "pt_jet")
        hjetpt_genvsreco_full.Scale(1.0 / hjetpt_genvsreco_full.Integral(1, -1, 1, -1))
        hjetpt_genvsreco_full.Write()
        fill2dhist(df_tmp_selrecogen, hz_genvsreco_full, self.v_varshape_binning_gen, self.v_varshape_binning)
        hz_genvsreco_full.Scale(1.0 / hz_genvsreco_full.Integral(1, -1, 1, -1))
        hz_genvsreco_full.Write()
        for row in df_tmp_selrecogen.itertuples():
            response_matrix.Fill(getattr(row, self.v_varshape_binning), row.pt_jet, getattr(row, self.v_varshape_binning_gen), row.pt_gen_jet)
        response_matrix.Write("response_matrix_nonprompt")

        # histograms for unfolding
        hjetpt_genvsreco_full_pr = \
            TH2F("hjetpt_genvsreco_full", "hjetpt_genvsreco_full", \
            self.p_nbin2_gen * 100, self.lvar2_binmin_gen[0], self.lvar2_binmax_gen[-1], \
            self.p_nbin2_reco * 100, self.lvar2_binmin_reco[0], self.lvar2_binmax_reco[-1])

        hz_genvsreco_full_pr = \
            TH2F("hz_genvsreco_full", "hz_genvsreco_full", \
                 self.p_nbinshape_gen * 100, self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1],
                 self.p_nbinshape_reco * 100, self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])
        fill2dhist(df_tmp_selrecogen_pr, hjetpt_genvsreco_full_pr, "pt_gen_jet", "pt_jet")
        hjetpt_genvsreco_full_pr.Scale(1.0 / hjetpt_genvsreco_full_pr.Integral(1, -1, 1, -1))
        hjetpt_genvsreco_full_pr.Write()
        fill2dhist(df_tmp_selrecogen_pr, hz_genvsreco_full_pr, self.v_varshape_binning_gen, self.v_varshape_binning)
        hz_genvsreco_full_pr.Scale(1.0 / hz_genvsreco_full_pr.Integral(1, -1, 1, -1))
        hz_genvsreco_full_pr.Write()


        hzvsjetpt_prior_weights = build2dhisto("hzvsjetpt_prior_weights", \
            self.varshapebinarray_gen, self.var2binarray_gen)
        fill2dhist(df_tmp_selrecogen_pr, hzvsjetpt_prior_weights, self.v_varshape_binning_gen, "pt_gen_jet")
        # end of histograms for unfolding

        for ibin2 in range(self.p_nbin2_reco):
            df_tmp_selrecogen_jetbin = seldf_singlevar(df_tmp_selrecogen, "pt_jet", \
                self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, \
                self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            hz_genvsreco = TH2F("hz_genvsreco_nonprompt" + suffix, "hz_genvsreco_nonprompt" + suffix, \
                self.p_nbinshape_gen * 100, self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1], \
                self.p_nbinshape_reco*100, self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])
            fill2dhist(df_tmp_selrecogen_jetbin, hz_genvsreco, self.v_varshape_binning_gen, self.v_varshape_binning)
            norm = hz_genvsreco.Integral(1, -1, 1, -1)
            if norm > 0:
                hz_genvsreco.Scale(1.0/norm)
            hz_genvsreco.Write()

            df_tmp_selrecogen_pr_jetbin = seldf_singlevar(df_tmp_selrecogen_pr, "pt_jet", \
                self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning, \
                self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
            hz_genvsreco_pr = TH2F("hz_genvsreco" + suffix, "hz_genvsreco" + suffix, \
                self.p_nbinshape_gen * 100, self.lvarshape_binmin_gen[0], self.lvarshape_binmax_gen[-1], \
                self.p_nbinshape_reco*100, self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])
            fill2dhist(df_tmp_selrecogen_pr_jetbin, hz_genvsreco_pr, self.v_varshape_binning_gen, self.v_varshape_binning)
            norm_pr = hz_genvsreco_pr.Integral(1, -1, 1, -1)
            if norm_pr > 0:
                hz_genvsreco_pr.Scale(1.0/norm_pr)
            hz_genvsreco_pr.Write()

        for ibinshape in range(len(self.lvarshape_binmin_reco)):
            df_tmp_selrecogen_zbin = seldf_singlevar(df_tmp_selrecogen, self.v_varshape_binning, \
                self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            suffix = "%s_%.2f_%.2f" % \
                (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco = TH2F("hjetpt_genvsreco_nonprompt" + suffix, \
                "hjetpt_genvsreco_nonprompt" + suffix, self.p_nbin2_gen * 100, self.lvar2_binmin_gen[0], \
                self.lvar2_binmax_gen[-1], self.p_nbin2_reco * 100, self.lvar2_binmin_reco[0], \
                self.lvar2_binmax_reco[-1])
            fill2dhist(df_tmp_selrecogen_zbin, hjetpt_genvsreco, "pt_gen_jet", "pt_jet")
            norm = hjetpt_genvsreco.Integral(1, -1, 1, -1)
            if norm > 0:
                hjetpt_genvsreco.Scale(1.0/norm)
            hjetpt_genvsreco.Write()

            df_tmp_selrecogen_pr_zbin = seldf_singlevar(df_tmp_selrecogen_pr, self.v_varshape_binning, \
                self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            suffix = "%s_%.2f_%.2f" % \
                (self.v_varshape_binning, self.lvarshape_binmin_reco[ibinshape], self.lvarshape_binmax_reco[ibinshape])
            hjetpt_genvsreco_pr = TH2F("hjetpt_genvsreco" + suffix, \
                "hjetpt_genvsreco" + suffix, self.p_nbin2_gen * 100, self.lvar2_binmin_gen[0], \
                self.lvar2_binmax_gen[-1], self.p_nbin2_reco * 100, self.lvar2_binmin_reco[0], \
                self.lvar2_binmax_reco[-1])
            fill2dhist(df_tmp_selrecogen_pr_zbin, hjetpt_genvsreco_pr, "pt_gen_jet", "pt_jet")
            norm_pr = hjetpt_genvsreco_pr.Integral(1, -1, 1, -1)
            if norm_pr > 0:
                hjetpt_genvsreco_pr.Scale(1.0/norm_pr)
            hjetpt_genvsreco_pr.Write()

        for ibinshape in range(len(self.lvarshape_binmin_gen)):
            dtmp_nonprompt_zgen = seldf_singlevar(df_mc_reco_merged_nonprompt, \
                self.v_varshape_binning_gen, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff = TH1F("hz_fracdiff_nonprompt" + suffix,
                               "hz_fracdiff_nonprompt" + suffix, 100, -2, 2)
            fill_hist(hz_fracdiff, (dtmp_nonprompt_zgen[self.v_varshape_binning] - \
                    dtmp_nonprompt_zgen[self.v_varshape_binning_gen])/dtmp_nonprompt_zgen[self.v_varshape_binning_gen])
            norm = hz_fracdiff.Integral(1, -1)
            if norm:
                hz_fracdiff.Scale(1.0 / norm)
            hz_fracdiff.Write()

            dtmp_prompt_zgen = seldf_singlevar(df_mc_reco_merged_prompt, \
                self.v_varshape_binning_gen, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_varshape_binning, self.lvarshape_binmin_gen[ibinshape], self.lvarshape_binmax_gen[ibinshape])
            hz_fracdiff_pr = TH1F("hz_fracdiff_prompt" + suffix,
                                  "hz_fracdiff_prompt" + suffix, 100, -2, 2)
            fill_hist(hz_fracdiff_pr, (dtmp_prompt_zgen[self.v_varshape_binning] - \
                    dtmp_prompt_zgen[self.v_varshape_binning_gen])/dtmp_prompt_zgen[self.v_varshape_binning_gen])
            norm_pr = hz_fracdiff_pr.Integral(1, -1)
            if norm_pr:
                hz_fracdiff_pr.Scale(1.0 / norm_pr)
            hz_fracdiff_pr.Write()

        for ibin2 in range(self.p_nbin2_gen):
            dtmp_nonprompt_jetptgen = seldf_singlevar(df_mc_reco_merged_nonprompt, \
                "pt_gen_jet", self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning,
                                       self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff = TH1F("hjetpt_fracdiff_nonprompt" + suffix,
                                   "hjetpt_fracdiff_nonprompt" + suffix, 100, -2, 2)
            fill_hist(hjetpt_fracdiff, (dtmp_nonprompt_jetptgen["pt_jet"] - \
                dtmp_nonprompt_jetptgen["pt_gen_jet"])/dtmp_nonprompt_jetptgen["pt_gen_jet"])
            norm = hjetpt_fracdiff.Integral(1, -1)
            if norm:
                hjetpt_fracdiff.Scale(1.0 / norm)
            hjetpt_fracdiff.Write()

            dtmp_prompt_jetptgen = seldf_singlevar(df_mc_reco_merged_prompt, \
                "pt_gen_jet", self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            suffix = "%s_%.2f_%.2f" % (self.v_var2_binning,
                                       self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hjetpt_fracdiff_pr = TH1F("hjetpt_fracdiff_prompt" + suffix,
                                      "hjetpt_fracdiff_prompt" + suffix, 100, -2, 2)
            fill_hist(hjetpt_fracdiff_pr, (dtmp_prompt_jetptgen["pt_jet"] - \
                dtmp_prompt_jetptgen["pt_gen_jet"])/dtmp_prompt_jetptgen["pt_gen_jet"])
            norm_pr = hjetpt_fracdiff_pr.Integral(1, -1)
            if norm_pr:
                hjetpt_fracdiff_pr.Scale(1.0 / norm_pr)
            hjetpt_fracdiff_pr.Write()

        df_mc_reco_merged_prompt_train, df_mc_reco_merged_prompt_test = \
                train_test_split(df_mc_reco_merged_prompt, test_size=self.closure_frac)
        df_tmp_selgen_pr_test, df_tmp_selreco_pr_test, df_tmp_selrecogen_pr_test = \
                self.create_df_closure(df_mc_reco_merged_prompt_test)
        _, _, df_tmp_selrecogen_pr_train = \
                self.create_df_closure(df_mc_reco_merged_prompt_train)

        fill2dhist(df_tmp_selreco_pr_test, hzvsjetpt_reco_closure_pr, self.v_varshape_binning, "pt_jet")
        fill2dhist(df_tmp_selgen_pr_test, hzvsjetpt_gen_closure_pr, self.v_varshape_binning_gen, "pt_gen_jet")
        hzvsjetpt_reco_closure_pr.Write("input_closure_reco")
        hzvsjetpt_gen_closure_pr.Write("input_closure_gen")


        for ibin2 in range(self.p_nbin2_gen):
            suffix = "%s_%.2f_%.2f" % \
                (self.v_var2_binning, self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            hz_gen_nocuts_closure = TH1F("hz_gen_nocuts_closure" + suffix,
                                         "hz_gen_nocuts_closure" + suffix,
                                         self.p_nbinshape_gen, self.varshapebinarray_gen)
            hz_gen_nocuts_closure.Sumw2()
            hz_gen_cuts_closure = TH1F("hz_gen_cuts_closure" + suffix,
                                       "hz_gen_cuts_closure" + suffix,
                                       self.p_nbinshape_gen, self.varshapebinarray_gen)
            hz_gen_cuts_closure.Sumw2()
            df_tmp_selgen_pr_test_bin = seldf_singlevar(df_tmp_selgen_pr_test, \
                "pt_gen_jet", self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            df_tmp_selrecogen_pr_test_bin = seldf_singlevar(df_tmp_selrecogen_pr_test, \
                "pt_gen_jet", self.lvar2_binmin_gen[ibin2], self.lvar2_binmax_gen[ibin2])
            fill_hist(hz_gen_nocuts_closure, df_tmp_selgen_pr_test_bin[self.v_varshape_binning_gen])
            fill_hist(hz_gen_cuts_closure, df_tmp_selrecogen_pr_test_bin[self.v_varshape_binning_gen])
            hz_gen_cuts_closure.Write()
            hz_gen_nocuts_closure.Write()

        fill_hist(hjetpt_gen_nocuts_closure, df_tmp_selgen_pr_test["pt_gen_jet"])
        fill_hist(hjetpt_gen_cuts_closure, df_tmp_selrecogen_pr_test["pt_gen_jet"])
        hjetpt_gen_nocuts_closure.Write()
        hjetpt_gen_cuts_closure.Write()

        hzvsjetpt_reco_nocuts_closure = TH2F("hzvsjetpt_reco_nocuts_closure",
                                             "hzvsjetpt_reco_nocuts_closure",
                                             self.p_nbinshape_reco, self.varshapebinarray_reco,
                                             self.p_nbin2_reco, self.var2binarray_reco)
        hzvsjetpt_reco_nocuts_closure.Sumw2()
        hzvsjetpt_reco_cuts_closure = TH2F("hzvsjetpt_reco_cuts_closure",
                                           "hzvsjetpt_reco_cuts_closure",
                                           self.p_nbinshape_reco, self.varshapebinarray_reco,
                                           self.p_nbin2_reco, self.var2binarray_reco)
        hzvsjetpt_reco_cuts_closure.Sumw2()

        fill2dhist(df_tmp_selreco_pr_test, hzvsjetpt_reco_nocuts_closure, self.v_varshape_binning, "pt_jet")
        fill2dhist(df_tmp_selrecogen_pr_test, hzvsjetpt_reco_cuts_closure, self.v_varshape_binning, "pt_jet")
        hzvsjetpt_reco_nocuts_closure.Write()
        hzvsjetpt_reco_cuts_closure.Write()

        for row in df_tmp_selrecogen_pr.itertuples():
            response_matrix_weight = 1.0
            if self.doprior is True:
                binx = hzvsjetpt_prior_weights.GetXaxis().FindBin(getattr(row, self.v_varshape_binning_gen))
                biny = hzvsjetpt_prior_weights.GetYaxis().FindBin(row.pt_gen_jet)
                weight = hzvsjetpt_prior_weights.GetBinContent(binx, biny)

                if weight > 0.0:
                    response_matrix_weight = 1.0/weight
            response_matrix_pr.Fill(getattr(row, self.v_varshape_binning), row.pt_jet,\
                getattr(row, self.v_varshape_binning_gen), row.pt_gen_jet, response_matrix_weight)
        for row in df_tmp_selrecogen_pr_train.itertuples():
            response_matrix_weight = 1.0
            if self.doprior is True:
                binx = hzvsjetpt_prior_weights.GetXaxis().FindBin(getattr(row, self.v_varshape_binning_gen))
                biny = hzvsjetpt_prior_weights.GetYaxis().FindBin(row.pt_gen_jet)
                weight = hzvsjetpt_prior_weights.GetBinContent(binx, biny)

                if weight > 0.0:
                    response_matrix_weight = 1.0/weight
            response_matrix_closure_pr.Fill(getattr(row, self.v_varshape_binning), row.pt_jet,\
                getattr(row, self.v_varshape_binning_gen), row.pt_gen_jet, response_matrix_weight)
        response_matrix_pr.Write("response_matrix")
        response_matrix_closure_pr.Write("response_matrix_closure")

        out_file.Close()
