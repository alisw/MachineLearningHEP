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
main script for doing systematics
"""
import math
import array
import pickle
import os
import uproot
import pandas as pd
import numpy as np
from root_numpy import fill_hist, evaluate # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F, TH2F, TH3F # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.utilities import selectdfrunlist
from machine_learning_hep.utilities import seldf_singlevar, openfile

# pylint: disable=too-few-public-methods, too-many-instance-attributes, too-many-statements
class Systematics:
    species = "systematics"
    def __init__(self, datap, typean, run_param, d_pkl_decmerged_mc, d_pkl_decmerged_data, d_results, d_val, p_period, runlisttrigger):

        self.typean = typean
        self.run_param = run_param
        self.d_pkl_decmerged_mc = d_pkl_decmerged_mc
        self.d_pkl_decmerged_data = d_pkl_decmerged_data
        self.d_results = d_results

        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])
        self.v_var_binning = datap["var_binning"]

        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        self.lpt_probcutpre_mc = datap["mlapplication"]["probcutpresel"]["mc"]
        self.lpt_probcutpre_data = datap["mlapplication"]["probcutpresel"]["data"]

        self.n_reco = datap["files_names"]["namefile_reco"]
        self.lpt_recodec_mc = [self.n_reco.replace(".pkl", "%d_%d_%.2f.pkl" % \
                              (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                               self.lpt_probcutpre_mc[i])) for i in range(self.p_nptbins)]
        self.lpt_recodec_data = [self.n_reco.replace(".pkl", "%d_%d_%.2f.pkl" % \
                                (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                                 self.lpt_probcutpre_data[i])) for i in range(self.p_nptbins)]

        self.lpt_recodecmerged_mc = [os.path.join(self.d_pkl_decmerged_mc, self.lpt_recodec_mc[ipt])
                                     for ipt in range(self.p_nptbins)]
        self.lpt_recodecmerged_data = [os.path.join(self.d_pkl_decmerged_data, self.lpt_recodec_data[ipt])
                                       for ipt in range(self.p_nptbins)]

        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_trigger_mc = datap["analysis"][self.typean]["triggersel"]["mc"]
        self.s_trigger_data = datap["analysis"][self.typean]["triggersel"]["data"]

        self.n_gen = datap["files_names"]["namefile_gen"]
        self.lpt_gensk = [self.n_gen.replace(".pkl", "_%s%d_%d.pkl" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_gendecmerged = [os.path.join(self.d_pkl_decmerged_mc, self.lpt_gensk[ipt])
                                 for ipt in range(self.p_nptbins)]
        
        self.s_presel_gen_eff = datap["analysis"][self.typean]["presel_gen_eff"]

        self.lvar2_binmin = datap["analysis"][self.typean]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"][self.typean]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"][self.typean]["var_binning2"]
        self.use_var2_bin = 0
        
        self.p_modelname = datap["mlapplication"]["modelname"]
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]

        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_filemass_cutvar = self.n_filemass.replace(".root", "_cutvar.root")
        self.n_fileeff_cutvar = self.n_fileeff.replace(".root", "_cutvar.root")
        self.n_filemass_cutvar = os.path.join(self.d_results, self.n_filemass_cutvar)
        self.n_fileeff_cutvar = os.path.join(self.d_results, self.n_fileeff_cutvar)

        self.p_mass_fit_lim = datap["analysis"][self.typean]["mass_fit_lim"]
        self.p_bin_width = datap["analysis"][self.typean]["bin_width"]
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))

        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger
        self.d_val = d_val
        self.period = p_period
        
        #To add in databases
        self.p_cutvar_minrange = [0.88, 0.84, 0.7, 0.7, 0.3]
        self.p_cutvar_maxrange = [0.97, 0.94, 0.9, 0.9, 0.7]
        self.p_ncutvar = 10
        self.p_maxperccutvar = 0.2
        #To remove from databases
        #self.p_prob_range = datap["systematics"]["probvariation"]["prob_range"]

    @staticmethod
    def loadstyle():
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(0)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)

    def probvariation(self):
        self.loadstyle()
        print(self.p_prob_range)

    def define_cutvariation_limits(self):

        min_cv_cut = []
        max_cv_cut = []
        for ipt in range(self.p_nptfinbins):
        
            print("Systematics pt-bin: ", ipt)

            bin_id = self.bin_matching[ipt]
            df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged_mc[bin_id], "rb"))
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_trigger_mc is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger_mc)

            df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)

            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])

            df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning, \
                                         self.lvar2_binmin[self.use_var2_bin], \
                                         self.lvar2_binmax[self.use_var2_bin])
            df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning, \
                                        self.lvar2_binmin[self.use_var2_bin], \
                                        self.lvar2_binmax[self.use_var2_bin])

            df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
            df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]

            selml_cent = "y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[bin_id])
            df_reco_sel_pr = df_reco_presel_pr.query(selml_cent)
            len_gen_pr = len(df_gen_sel_pr)
            eff_cent = len(df_reco_sel_pr)/len_gen_pr
            print("Central efficiency pt-bin", ipt, ": ", eff_cent)
            
            stepsmin = (self.lpt_probcutfin[bin_id] - self.p_cutvar_minrange[bin_id]) / self.p_ncutvar
            min_cv_cut.append(self.lpt_probcutfin[bin_id])
            df_reco_cvmin_pr = df_reco_presel_pr
            for icv in range(self.p_ncutvar):
                min_cv_cut[ipt] = self.p_cutvar_minrange[bin_id] + icv * stepsmin
                selml_min = "y_test_prob%s>%s" % (self.p_modelname, min_cv_cut[ipt])
                df_reco_cvmin_pr = df_reco_cvmin_pr.query(selml_min)
                eff_min = len(df_reco_cvmin_pr)/len_gen_pr
                if eff_min / eff_cent < 1 + self.p_maxperccutvar:
                    break

            eff_min = len(df_reco_cvmin_pr)/len_gen_pr
            print("Minimal efficiency pt-bin", ipt, ": ", eff_min, "(", eff_min / eff_cent, ")")
 
            stepsmax = (self.p_cutvar_maxrange[bin_id] - self.lpt_probcutfin[bin_id]) / self.p_ncutvar
            max_cv_cut.append(self.lpt_probcutfin[bin_id])
            df_reco_cvmax_pr = df_reco_sel_pr
            for icv in range(self.p_ncutvar):
                max_cv_cut[ipt] = self.lpt_probcutfin[bin_id] + icv * stepsmax
                selml_max = "y_test_prob%s>%s" % (self.p_modelname, max_cv_cut[ipt])
                df_reco_cvmax_pr = df_reco_cvmax_pr.query(selml_max)
                eff_max = len(df_reco_cvmax_pr)/len_gen_pr
                if eff_max / eff_cent < 1 - self.p_maxperccutvar:
                    break
            
            eff_max = len(df_reco_cvmax_pr)/len_gen_pr
            print("Maximal efficiency pt-bin", ipt, ": ", eff_max, "(", eff_max / eff_cent, ")")

        return min_cv_cut, max_cv_cut

    def cutvariation_masshistos(self, min_cv_cut, max_cv_cut):
        myfile = TFile.Open(self.n_filemass_cutvar, "recreate")

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.lpt_recodecmerged_data[bin_id], "rb"))

            stepsmin = (self.lpt_probcutfin[bin_id] - min_cv_cut[ipt]) / self.p_ncutvar
            stepsmax = (max_cv_cut[ipt] - self.lpt_probcutfin[bin_id]) / self.p_ncutvar
            ntrials = 2 * self.p_ncutvar + 1
            icvmax = 1
            
            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger_data is not None:
                df = df.query(self.s_trigger_data)
            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            print("Using run selection for mass histo", self.runlistrigger[self.triggerbit], \
                  "for period", self.period)
            df = selectdfrunlist(df, self.run_param[self.runlistrigger[self.triggerbit]], \
                                 "run_number")

            for icv in range(ntrials):
                if icv < self.p_ncutvar:
                    selml_cvval = min_cv_cut[ipt] + icv * stepsmin
                elif icv == self.p_ncutvar:
                    selml_cvval = self.lpt_probcutfin[bin_id]
                else:
                    selml_cvval = self.lpt_probcutfin[bin_id] + icvmax * stepsmin
                    icvmax = icvmax + 1
                selml_cv = "y_test_prob%s>%s" % (self.p_modelname, selml_cvval)

                print("Cutting on: ", selml_cv)
                df = df.query(selml_cv)

                for ibin2 in range(len(self.lvar2_binmin)):
                    suffix = "%s%d_%d_%d_%s%.2f_%.2f" % \
                             (self.v_var_binning, self.lpt_finbinmin[ipt],
                              self.lpt_finbinmax[ipt], icv,
                              self.v_var2_binning, self.lvar2_binmin[ibin2],
                              self.lvar2_binmax[ibin2])
                    h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                     self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                    h_invmass_weight = TH1F("h_invmass_weight" + suffix, "", self.p_num_bins,
                                            self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])

                    df_bin = seldf_singlevar(df, self.v_var2_binning,
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                      
                    fill_hist(h_invmass, df_bin.inv_mass)
                      
                    if "INT7" not in self.triggerbit:
                        fileweight_name = "%s/correctionsweights.root" % self.d_val
                        fileweight = TFile.Open(fileweight_name, "read")
                        namefunction = "funcnorm_%s_%s" % (self.triggerbit, self.v_var2_binning)
                        funcweighttrig = fileweight.Get(namefunction)
                        if funcweighttrig:
                            weights = evaluate(funcweighttrig, df_bin[self.v_var2_binning])
                            weightsinv = [1./weight for weight in weights]
                            fill_hist(h_invmass_weight, df_bin.inv_mass, weights=weightsinv)
                    myfile.cd()
                    h_invmass.Write()
                    h_invmass_weight.Write()

    # pylint: disable=line-too-long
    def cutvariation_efficiencies(self, min_cv_cut, max_cv_cut):
        myfile = TFile.Open(self.n_fileeff_cutvar, "recreate")

        h_gen_pr = []
        h_presel_pr = []
        h_sel_pr = []
        h_gen_fd = []
        h_presel_fd = []
        h_sel_fd = []

        idx = 0
        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.lpt_recodecmerged_mc[bin_id], "rb"))

            if self.s_evtsel is not None:
                df = df.query(self.s_evtsel)
            if self.s_trigger_mc is not None:
                df = df.query(self.s_trigger_mc)
            print("Using run selection for eff histo", self.runlistrigger[self.triggerbit], \
                  "for period", self.period)
            df = selectdfrunlist(df, \
                                 self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df = seldf_singlevar(df, self.v_var_binning, self.lpt_finbinmin[ipt], \
                                 self.lpt_finbinmax[ipt])

            df_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
            df_gen = df_gen.query(self.s_presel_gen_eff)
            df_gen = selectdfrunlist(df_gen, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_gen = seldf_singlevar(df_gen, self.v_var_binning, self.lpt_finbinmin[ipt], \
                                     self.lpt_finbinmax[ipt])

            stepsmin = (self.lpt_probcutfin[bin_id] - min_cv_cut[ipt]) / self.p_ncutvar
            stepsmax = (max_cv_cut[ipt] - self.lpt_probcutfin[bin_id]) / self.p_ncutvar
            ntrials = 2 * self.p_ncutvar + 1
            icvmax = 1

            idx = 0
            for icv in range(ntrials):
                if icv < self.p_ncutvar:
                    selml_cvval = min_cv_cut[ipt] + icv * stepsmin
                elif icv == self.p_ncutvar:
                    selml_cvval = self.lpt_probcutfin[bin_id]
                else:
                    selml_cvval = self.lpt_probcutfin[bin_id] + icvmax * stepsmin
                    icvmax = icvmax + 1
                selml_cv = "y_test_prob%s>%s" % (self.p_modelname, selml_cvval)

                print("Cutting on: ", selml_cv)
                df = df.query(selml_cv)

                for ibin2 in range(len(self.lvar2_binmin)):
                    stringbin2 = "_%d_%s_%.2f_%.2f" % (icv, \
                                                self.v_var2_binning, \
                                                self.lvar2_binmin[ibin2], \
                                                self.lvar2_binmax[ibin2])
            
                    if ipt == 0:
                        n_bins = len(self.lpt_finbinmin)
                        analysis_bin_lims_temp = self.lpt_finbinmin.copy()
                        analysis_bin_lims_temp.append(self.lpt_finbinmax[n_bins-1])
                        analysis_bin_lims = array.array('f', analysis_bin_lims_temp)
                        h_gen_pr.append(TH1F("h_gen_pr" + stringbin2, "Prompt Generated in acceptance |y|<0.5", \
                                        n_bins, analysis_bin_lims))
                        h_sel_pr.append(TH1F("h_sel_pr" + stringbin2, "Prompt Reco and sel in acc |#eta|<0.8 and sel", \
                                        n_bins, analysis_bin_lims))
                        h_gen_fd.append(TH1F("h_gen_fd" + stringbin2, "FD Generated in acceptance |y|<0.5", \
                                        n_bins, analysis_bin_lims))
                        h_sel_fd.append(TH1F("h_sel_fd" + stringbin2, "FD Reco and sel in acc |#eta|<0.8 and sel", \
                                        n_bins, analysis_bin_lims))

                    df_bin = seldf_singlevar(df, self.v_var2_binning, self.lvar2_binmin[ibin2], \
                                         self.lvar2_binmax[ibin2])
                    df_gen_bin = seldf_singlevar(df_gen, self.v_var2_binning, self.lvar2_binmin[ibin2], \
                                             self.lvar2_binmax[ibin2])

                    df_sel_pr = df_bin[df_bin.ismcprompt == 1]
                    df_gen_pr = df_gen_bin[df_gen_bin.ismcprompt == 1]
                    df_sel_fd = df_bin[df_bin.ismcfd == 1]
                    df_gen_fd = df_gen_bin[df_gen_bin.ismcfd == 1]

                    h_gen_pr[idx].SetBinContent(ipt + 1, len(df_gen_pr))
                    h_gen_pr[idx].SetBinError(ipt + 1, math.sqrt(len(df_gen_pr)))
                    h_sel_pr[idx].SetBinContent(ipt + 1, len(df_sel_pr))
                    h_sel_pr[idx].SetBinError(ipt + 1, math.sqrt(len(df_sel_pr)))

                    h_gen_fd[idx].SetBinContent(ipt + 1, len(df_gen_fd))
                    h_gen_fd[idx].SetBinError(ipt + 1, math.sqrt(len(df_gen_fd)))
                    h_sel_fd[idx].SetBinContent(ipt + 1, len(df_sel_fd))
                    h_sel_fd[idx].SetBinError(ipt + 1, math.sqrt(len(df_sel_fd)))
                    idx = idx + 1

        myfile.cd()
        for i in range(idx):
            h_gen_pr[i].Write()
            h_sel_pr[i].Write()
            h_gen_fd[i].Write()
            h_sel_fd[i].Write()
