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
import os
import numpy as np
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.bitwise import tag_bit_df
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, \
        seldf_singlevar_inclusive, openfile
from machine_learning_hep.utilities import mergerootfiles
from machine_learning_hep.utilities import get_timestamp_string
#from machine_learning_hep.globalfitter import fitter
from machine_learning_hep.selectionutils import gethistonormforselevt_mult
from machine_learning_hep.processer import Processer

class ProcesserDhadrons_mult(Processer): # pylint: disable=too-many-instance-attributes, invalid-name
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
        if self.runlistrigger is not None:
            dfevtorig = selectdfrunlist(dfevtorig, \
                             self.run_param[self.runlistrigger], "run_number")
        # Select periods within year if required by the user
        if self.periods_runlist_merged:
            dfevtorig = selectdfrunlist(dfevtorig, self.periods_runlist_merged, "run_number")

        dfevtevtsel = dfevtorig.query("is_ev_rej==0")

        labeltrigger = "hbit%svs%s" % (self.triggerbit, self.v_var2_binning_gen)

        myfile.cd()
        hsel, hnovtxmult, hvtxoutmult = \
            gethistonormforselevt_mult(dfevtorig, dfevtevtsel,
                                       labeltrigger, self.v_var2_binning_gen)
        hsel.Write()
        hnovtxmult.Write()
        hvtxoutmult.Write()

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.mptfiles_recoskmldec[bin_id][index], "rb"))
            if self.doml is True:
                df = df.query(self.l_selml[bin_id])
            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger is not None:
                df = df.query(self.s_trigger)

            # Select periods within year if required by the user
            if self.periods_runlist_merged:
                df = selectdfrunlist(df, self.periods_runlist_merged, "run_number")

            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            for ibin2 in range(len(self.lvar2_binmin)):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                df_bin = seldf_singlevar_inclusive(df, self.v_var2_binning, \
                                         self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                if self.runlistrigger is not None:
                    df_bin = selectdfrunlist(df_bin, \
                             self.run_param[self.runlistrigger], "run_number")
                fill_hist(h_invmass, df_bin.inv_mass)
                myfile.cd()
                h_invmass.Write()

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

    def process_histomass(self):
        print("Doing masshisto", self.mcordata, self.period)
        print("Using run selection for mass histo", \
               self.runlistrigger, "for period", self.period)
        if self.doml is True:
            print("Doing ml analysis")
        else:
            print("No extra selection needed since we are doing std analysis")

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_histomass_single, arguments, self.p_chunksizeunp)
        tmp_merged = \
        f"/data/tmp/hadd/{self.case}_{self.typean}/mass_{self.period}/{get_timestamp_string()}/"
        mergerootfiles(self.l_histomass, self.n_filemass, tmp_merged)


    def get_reweighted_count(self, dfsel):
        filename = os.path.join(self.d_mcreweights, self.n_mcreweights)
        weight_file = TFile.Open(filename, "read")
        weights = weight_file.Get("Weights0")
        w = [weights.GetBinContent(weights.FindBin(v)) for v in
             dfsel[self.v_var2_binning_gen]]
        val = sum(w)
        err = math.sqrt(sum(map(lambda i: i * i, w)))
        #print('reweighting sum: {:.1f} +- {:.1f} -> {:.1f} +- {:.1f} (zeroes: {})' \
        #      .format(len(dfsel), math.sqrt(len(dfsel)), val, err, w.count(0.)))
        return val, err

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
                if self.runlistrigger is not None:
                    df_mc_reco = selectdfrunlist(df_mc_reco, \
                         self.run_param[self.runlistrigger], "run_number")
                df_mc_gen = pickle.load(openfile(self.mptfiles_gensk[bin_id][index], "rb"))
                df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
                if self.runlistrigger is not None:
                    df_mc_gen = selectdfrunlist(df_mc_gen, \
                             self.run_param[self.runlistrigger], "run_number")
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_reco = seldf_singlevar_inclusive(df_mc_reco, self.v_var2_binning_gen, \
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_mc_gen = seldf_singlevar_inclusive(df_mc_gen, self.v_var2_binning_gen, \
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

    def process_efficiency(self):
        print("Doing efficiencies", self.mcordata, self.period)
        print("Using run selection for eff histo", \
               self.runlistrigger, "for period", self.period)
        if self.doml is True:
            print("Doing ml analysis")
        else:
            print("No extra selection needed since we are doing std analysis")
        for ibin2 in range(len(self.lvar2_binmin)):
            if self.corr_eff_mult[ibin2] is True:
                print("Reweighting efficiencies for bin", ibin2)
            else:
                print("Not reweighting efficiencies for bin", ibin2)

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_efficiency_single, arguments, self.p_chunksizeunp)
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/histoeff_{self.period}/{get_timestamp_string()}/"
        mergerootfiles(self.l_histoeff, self.n_fileeff, tmp_merged)
