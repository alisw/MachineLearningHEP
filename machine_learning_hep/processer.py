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
from machine_learning_hep.utilities import mergerootfiles, z_calc, z_gen_calc, scatterplot
from machine_learning_hep.models import apply # pylint: disable=import-error
#from machine_learning_hep.globalfitter import fitter
from machine_learning_hep.selectionutils import getnormforselevt

class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, d_val, typean):
        self.case = case
        self.typean = typean
        #directories
        self.d_root = d_root
        self.d_pkl = d_pkl
        self.d_pklsk = d_pklsk
        self.d_pkl_ml = d_pkl_ml
        self.d_results = d_results
        self.d_val = d_val
        self.datap = datap
        self.mcordata = mcordata
        self.p_frac_merge = p_frac_merge
        self.p_rd_merge = p_rd_merge
        self.period = p_period
        self.runlist = run_param[self.period]

        self.p_maxfiles = p_maxfiles
        self.p_chunksizeunp = p_chunksizeunp
        self.p_chunksizeskim = p_chunksizeskim

        #parameter names
        self.p_maxprocess = p_maxprocess
        self.indexsample = None
        self.p_dofullevtmerge = datap["dofullevtmerge"]

        #namefile root
        self.n_root = datap["files_names"]["namefile_unmerged_tree"]
        #troot trees names
        self.n_treereco = datap["files_names"]["treeoriginreco"]
        self.n_treegen = datap["files_names"]["treeorigingen"]
        self.n_treeevt = datap["files_names"]["treeoriginevt"]

        #namefiles pkl
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_evtvalroot = datap["files_names"]["namefile_evtvalroot"]
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        
        #selections
        self.s_reco_unp = datap["sel_reco_unp"]
        self.s_good_evt_unp = datap["sel_good_evt_unp"]
        self.s_cen_unp = datap["sel_cen_unp"]
        self.s_gen_unp = datap["sel_gen_unp"]
        self.s_reco_skim = datap["sel_reco_skim"]
        self.s_gen_skim = datap["sel_gen_skim"]

        #bitmap
        self.b_trackcuts = datap["sel_reco_singletrac_unp"]
        self.b_std = datap["bitmap_sel"]["isstd"]
        self.b_mcsig = datap["bitmap_sel"]["ismcsignal"]
        self.b_mcsigprompt = datap["bitmap_sel"]["ismcprompt"]
        self.b_mcsigfd = datap["bitmap_sel"]["ismcfd"]
        self.b_mcbkg = datap["bitmap_sel"]["ismcbkg"]
        self.b_mcrefl = datap["bitmap_sel"]["ismcrefl"]

        #variables name
        self.v_all = datap["variables"]["var_all"]
        self.v_train = datap["variables"]["var_training"]
        self.v_evt = datap["variables"]["var_evt"][self.mcordata]
        self.v_gen = datap["variables"]["var_gen"]
        self.v_evtmatch = datap["variables"]["var_evt_match"]
        self.v_bitvar = datap["bitmap_sel"]["var_name"]
        self.v_isstd = datap["bitmap_sel"]["var_isstd"]
        self.v_ismcsignal = datap["bitmap_sel"]["var_ismcsignal"]
        self.v_ismcprompt = datap["bitmap_sel"]["var_ismcprompt"]
        self.v_ismcfd = datap["bitmap_sel"]["var_ismcfd"]
        self.v_ismcbkg = datap["bitmap_sel"]["var_ismcbkg"]
        self.v_ismcrefl = datap["bitmap_sel"]["var_ismcrefl"]
        self.v_var_binning = datap["var_binning"]
        #list of files names

        self.l_path = None
        if os.path.isdir(self.d_root):
            self.l_path = list_folders(self.d_root, self.n_root, self.p_maxfiles)
        else:
            self.l_path = list_folders(self.d_pkl, self.n_reco, self.p_maxfiles)

        self.l_root = createlist(self.d_root, self.l_path, self.n_root)
        self.l_reco = createlist(self.d_pkl, self.l_path, self.n_reco)
        self.l_evt = createlist(self.d_pkl, self.l_path, self.n_evt)
        self.l_evtorig = createlist(self.d_pkl, self.l_path, self.n_evtorig)
        self.l_evtvalroot = createlist(self.d_val, self.l_path, self.n_evtvalroot)


        if self.mcordata == "mc":
            self.l_gen = createlist(self.d_pkl, self.l_path, self.n_gen)

        self.f_totevt = os.path.join(self.d_pkl, self.n_evt)
        self.f_totevtorig = os.path.join(self.d_pkl, self.n_evtorig)
        self.f_totevtvalroot = os.path.join(self.d_val, self.n_evtvalroot)

        self.p_modelname = datap["mlapplication"]["modelname"]
        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])
        self.lpt_model = datap["mlapplication"]["modelsperptbin"]
        self.dirmodel = datap["ml"]["mlout"]
        self.lpt_model = appendmainfoldertolist(self.dirmodel, self.lpt_model)
        self.lpt_probcutpre = datap["mlapplication"]["probcutpresel"][self.mcordata]
        self.lpt_probcutfin = datap["mlapplication"]["probcutoptimal"]

        if self.lpt_probcutfin < self.lpt_probcutpre:
            print("FATAL error: probability cut final must be tighter!")

        self.d_pkl_dec = d_pkl_dec
        self.mptfiles_recosk = []
        self.mptfiles_gensk = []

        self.d_pkl_decmerged = d_pkl_decmerged
        self.n_filemass = os.path.join(self.d_results, self.n_filemass)
        self.n_fileeff = os.path.join(self.d_results, self.n_fileeff)

        self.lpt_recosk = [self.n_reco.replace(".pkl", "_%s%d_%d.pkl" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_gensk = [self.n_gen.replace(".pkl", "_%s%d_%d.pkl" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_reco_ml = [os.path.join(self.d_pkl_ml, self.lpt_recosk[ipt]) \
                             for ipt in range(self.p_nptbins)]
        self.lpt_gen_ml = [os.path.join(self.d_pkl_ml, self.lpt_gensk[ipt]) \
                            for ipt in range(self.p_nptbins)]
        self.f_evt_ml = os.path.join(self.d_pkl_ml, self.n_evt)
        self.f_evtorig_ml = os.path.join(self.d_pkl_ml, self.n_evtorig)

        self.lpt_recodec = [self.n_reco.replace(".pkl", "%d_%d_%.2f.pkl" % \
                           (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                            self.lpt_probcutpre[i])) for i in range(self.p_nptbins)]
        self.mptfiles_recosk = [createlist(self.d_pklsk, self.l_path, \
                                self.lpt_recosk[ipt]) for ipt in range(self.p_nptbins)]
        self.mptfiles_recoskmldec = [createlist(self.d_pkl_dec, self.l_path, \
                                   self.lpt_recodec[ipt]) for ipt in range(self.p_nptbins)]
        self.lpt_recodecmerged = [os.path.join(self.d_pkl_decmerged, self.lpt_recodec[ipt])
                                  for ipt in range(self.p_nptbins)]
        if self.mcordata == "mc":
            self.mptfiles_gensk = [createlist(self.d_pklsk, self.l_path, \
                                    self.lpt_gensk[ipt]) for ipt in range(self.p_nptbins)]
            self.lpt_gendecmerged = [os.path.join(self.d_pkl_decmerged, self.lpt_gensk[ipt])
                                     for ipt in range(self.p_nptbins)]
        self.lpt_filemass = [self.n_filemass.replace(".root", "%d_%d_%.2f.root" % \
                (self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt], \
                 self.lpt_probcutfin[ipt])) for ipt in range(self.p_nptbins)]

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

        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        #self.sel_final_fineptbins = datap["analysis"][self.typean]["sel_final_fineptbins"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]


    def unpack(self, file_index):
        treeevtorig = uproot.open(self.l_root[file_index])[self.n_treeevt]
        dfevtorig = treeevtorig.pandas.df(branches=self.v_evt)
        dfevtorig = selectdfrunlist(dfevtorig, self.runlist, "run_number")
        dfevtorig = selectdfquery(dfevtorig, self.s_cen_unp)
        dfevtorig = dfevtorig.reset_index(drop=True)
        pickle.dump(dfevtorig, openfile(self.l_evtorig[file_index], "wb"), protocol=4)
        dfevt = selectdfquery(dfevtorig, self.s_good_evt_unp)
        dfevt = dfevt.reset_index(drop=True)
        pickle.dump(dfevt, openfile(self.l_evt[file_index], "wb"), protocol=4)


        treereco = uproot.open(self.l_root[file_index])[self.n_treereco]
        dfreco = treereco.pandas.df(branches=self.v_all)
        dfreco = selectdfrunlist(dfreco, self.runlist, "run_number")
        dfreco = selectdfquery(dfreco, self.s_reco_unp)
        dfreco = pd.merge(dfreco, dfevt, on=self.v_evtmatch)
        isselacc = selectfidacc(dfreco.pt_cand.values, dfreco.y_cand.values)
        dfreco = dfreco[np.array(isselacc, dtype=bool)]
        if self.b_trackcuts is not None:
            dfreco = filter_bit_df(dfreco, self.v_bitvar, self.b_trackcuts)
        dfreco[self.v_isstd] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                   self.b_std), dtype=int)
        dfreco = dfreco.reset_index(drop=True)
        if self.mcordata == "mc":
            dfreco[self.v_ismcsignal] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                            self.b_mcsig), dtype=int)
            dfreco[self.v_ismcprompt] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                            self.b_mcsigprompt), dtype=int)
            dfreco[self.v_ismcfd] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                        self.b_mcsigfd), dtype=int)
            dfreco[self.v_ismcbkg] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                         self.b_mcbkg), dtype=int)
        pickle.dump(dfreco, openfile(self.l_reco[file_index], "wb"), protocol=4)

        if self.mcordata == "mc":
            treegen = uproot.open(self.l_root[file_index])[self.n_treegen]
            dfgen = treegen.pandas.df(branches=self.v_gen)
            dfgen = selectdfrunlist(dfgen, self.runlist, "run_number")
            dfgen = pd.merge(dfgen, dfevtorig, on=self.v_evtmatch)
            dfgen = selectdfquery(dfgen, self.s_gen_unp)
            dfgen[self.v_isstd] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                      self.b_std), dtype=int)
            dfgen[self.v_ismcsignal] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                           self.b_mcsig), dtype=int)
            dfgen[self.v_ismcprompt] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                           self.b_mcsigprompt), dtype=int)
            dfgen[self.v_ismcfd] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                       self.b_mcsigfd), dtype=int)
            dfgen[self.v_ismcbkg] = np.array(tag_bit_df(dfgen, self.v_bitvar,
                                                        self.b_mcbkg), dtype=int)
            dfgen = dfgen.reset_index(drop=True)
            pickle.dump(dfgen, openfile(self.l_gen[file_index], "wb"), protocol=4)

    def skim(self, file_index):
        try:
            dfreco = pickle.load(openfile(self.l_reco[file_index], "rb"))
        except Exception as e: # pylint: disable=broad-except
            print('failed to open file', self.l_reco[file_index], str(e))
        for ipt in range(self.p_nptbins):
            dfrecosk = seldf_singlevar(dfreco, self.v_var_binning,
                                       self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt])
            dfrecosk = selectdfquery(dfrecosk, self.s_reco_skim[ipt])
            dfrecosk = dfrecosk.reset_index(drop=True)
            f = openfile(self.mptfiles_recosk[ipt][file_index], "wb")
            pickle.dump(dfrecosk, f, protocol=4)
            f.close()
            if self.mcordata == "mc":
                try:
                    dfgen = pickle.load(openfile(self.l_gen[file_index], "rb"))
                except Exception as e: # pylint: disable=broad-except
                    print('failed to open MC file', self.l_gen[file_index], str(e))
                dfgensk = seldf_singlevar(dfgen, self.v_var_binning,
                                          self.lpt_anbinmin[ipt], self.lpt_anbinmax[ipt])
                dfgensk = selectdfquery(dfgensk, self.s_gen_skim[ipt])
                dfgensk = dfgensk.reset_index(drop=True)
                pickle.dump(dfgensk, openfile(self.mptfiles_gensk[ipt][file_index], "wb"),
                            protocol=4)

    def applymodel(self, file_index):
        for ipt in range(self.p_nptbins):
            if os.path.exists(self.mptfiles_recoskmldec[ipt][file_index]):
                if os.stat(self.mptfiles_recoskmldec[ipt][file_index]).st_size != 0:
                    continue
            dfrecosk = pickle.load(openfile(self.mptfiles_recosk[ipt][file_index], "rb"))
            if os.path.isfile(self.lpt_model[ipt]) is False:
                print("Model file not present in bin %d" % ipt)
            mod = pickle.load(openfile(self.lpt_model[ipt], 'rb'))
            dfrecoskml = apply("BinaryClassification", [self.p_modelname], [mod],
                               dfrecosk, self.v_train)
            probvar = "y_test_prob" + self.p_modelname
            dfrecoskml = dfrecoskml.loc[dfrecoskml[probvar] > self.lpt_probcutpre[ipt]]
            pickle.dump(dfrecoskml, openfile(self.mptfiles_recoskmldec[ipt][file_index], "wb"),
                        protocol=4)

    def parallelizer(self, function, argument_list, maxperchunk):
        chunks = [argument_list[x:x+maxperchunk] \
                  for x in range(0, len(argument_list), maxperchunk)]
        for chunk in chunks:
            print("Processing new chunck size=", maxperchunk)
            pool = mp.Pool(self.p_maxprocess)
            _ = [pool.apply_async(function, args=chunk[i]) for i in range(len(chunk))]
            pool.close()
            pool.join()

    def process_unpack_par(self):
        print("doing unpacking", self.mcordata, self.period)
        create_folder_struc(self.d_pkl, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.unpack, arguments, self.p_chunksizeunp)

    def process_skim_par(self):
        print("doing skimming", self.mcordata, self.period)
        create_folder_struc(self.d_pklsk, self.l_path)
        arguments = [(i,) for i in range(len(self.l_reco))]
        self.parallelizer(self.skim, arguments, self.p_chunksizeskim)
        if self.p_dofullevtmerge is True:
            merge_method(self.l_evt, self.f_totevt)
            merge_method(self.l_evtorig, self.f_totevtorig)

    def process_applymodel_par(self):
        print("doing apply model", self.mcordata, self.period)
        create_folder_struc(self.d_pkl_dec, self.l_path)
        arguments = [(i,) for i in range(len(self.mptfiles_recosk[0]))]
        self.parallelizer(self.applymodel, arguments, self.p_chunksizeskim)

    def process_mergeforml(self):
        nfiles = len(self.mptfiles_recosk[0])
        if nfiles == 0:
            print("increase the fraction of merged files or the total number")
            print(" of files you process")
        ntomerge = (int)(nfiles * self.p_frac_merge)
        rd.seed(self.p_rd_merge)
        filesel = rd.sample(range(0, nfiles), ntomerge)
        for ipt in range(self.p_nptbins):
            list_sel_recosk = [self.mptfiles_recosk[ipt][j] for j in filesel]
            merge_method(list_sel_recosk, self.lpt_reco_ml[ipt])
            if self.mcordata == "mc":
                list_sel_gensk = [self.mptfiles_gensk[ipt][j] for j in filesel]
                merge_method(list_sel_gensk, self.lpt_gen_ml[ipt])

        list_sel_evt = [self.l_evt[j] for j in filesel]
        list_sel_evtorig = [self.l_evtorig[j] for j in filesel]
        merge_method(list_sel_evt, self.f_evt_ml)
        merge_method(list_sel_evtorig, self.f_evtorig_ml)

    def process_mergedec(self):
        for ipt in range(self.p_nptbins):
            merge_method(self.mptfiles_recoskmldec[ipt], self.lpt_recodecmerged[ipt])
            if self.mcordata == "mc":
                merge_method(self.mptfiles_gensk[ipt], self.lpt_gendecmerged[ipt])

    def process_histomass(self):
        myfile = TFile.Open(self.n_filemass, "recreate")

        for ipt in range(self.p_nptfinbins):
            bin_id = self.bin_matching[ipt]
            df = pickle.load(openfile(self.lpt_recodecmerged[bin_id], "rb"))
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
                fill_hist(h_invmass, df_bin.inv_mass)
                triggerbit = self.datap["analysis"][self.typean]["triggerbit"]
                if "INT7" not in triggerbit and self.mcordata == "data":
                    fileweight_name = "%s/correctionsweights.root" % self.d_val
                    fileweight = TFile.Open(fileweight_name, "read")
                    namefunction = "funcnorm_%s_%s" % (self.triggerbit, self.v_var2_binning)
                    funcweighttrig = fileweight.Get(namefunction)
                    weights = evaluate(funcweighttrig, df_bin[self.v_var2_binning])
                    weightsinv = [1./weight for weight in weights]
                    fill_hist(h_invmass_weight, df_bin.inv_mass, weights=weightsinv)
                myfile.cd()
                h_invmass.Write()
                h_invmass_weight.Write()
                if "pt_jet" in df_bin.columns:
                    zarray = z_calc(df_bin.pt_jet, df_bin.phi_jet, df_bin.eta_jet,
                                    df_bin.pt_cand, df_bin.phi_cand, df_bin.eta_cand)
                    h_zvsinvmass = TH2F("hzvsmass" + suffix, "", 5000, 1.00, 6.00, 2000, -0.5, 1.5)
                    h_zvsinvmass.Sumw2()
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

    # pylint: disable=line-too-long
    def process_efficiency(self):
        out_file = TFile.Open(self.n_fileeff, "recreate")
        for ibin2 in range(len(self.lvar2_binmin)):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                        self.lvar2_binmin[ibin2], \
                                        self.lvar2_binmax[ibin2])
            print(stringbin2)
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
                df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged[bin_id], "rb"))
                if self.s_evtsel is not None:
                    df_mc_reco = df_mc_reco.query(self.s_evtsel)
                if self.s_trigger is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger)
                df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
                df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var_binning, \
                                     self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
                df_mc_reco = seldf_singlevar(df_mc_reco, self.v_var2_binning, \
                                             self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_mc_gen = seldf_singlevar(df_mc_gen, self.v_var2_binning, \
                                            self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                df_gen_sel_pr = df_mc_gen[df_mc_gen.ismcprompt == 1]
                df_reco_presel_pr = df_mc_reco[df_mc_reco.ismcprompt == 1]
                df_reco_sel_pr = df_reco_presel_pr.query(self.l_selml[bin_id])
                df_gen_sel_fd = df_mc_gen[df_mc_gen.ismcfd == 1]
                df_reco_presel_fd = df_mc_reco[df_mc_reco.ismcfd == 1]
                df_reco_sel_fd = df_reco_presel_fd.query(self.l_selml[bin_id])

                h_gen_pr.SetBinContent(bincounter + 1, len(df_gen_sel_pr))
                h_gen_pr.SetBinError(bincounter + 1, math.sqrt(len(df_gen_sel_pr)))
                h_presel_pr.SetBinContent(bincounter + 1, len(df_reco_presel_pr))
                h_presel_pr.SetBinError(bincounter + 1, math.sqrt(len(df_reco_presel_pr)))
                h_sel_pr.SetBinContent(bincounter + 1, len(df_reco_sel_pr))
                h_sel_pr.SetBinError(bincounter + 1, math.sqrt(len(df_reco_sel_pr)))
                #print("prompt efficiency tot ptbin=", bincounter, ", value = ",
                #      len(df_reco_sel_pr)/len(df_gen_sel_pr))

                h_gen_fd.SetBinContent(bincounter + 1, len(df_gen_sel_fd))
                h_gen_fd.SetBinError(bincounter + 1, math.sqrt(len(df_gen_sel_fd)))
                h_presel_fd.SetBinContent(bincounter + 1, len(df_reco_presel_fd))
                h_presel_fd.SetBinError(bincounter + 1, math.sqrt(len(df_reco_presel_fd)))
                h_sel_fd.SetBinContent(bincounter + 1, len(df_reco_sel_fd))
                h_sel_fd.SetBinError(bincounter + 1, math.sqrt(len(df_reco_sel_fd)))
                #print("fd efficiency tot ptbin=", bincounter, ", value = ",
                #      len(df_reco_sel_fd)/len(df_gen_sel_fd))
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
            df_mc_reco = df_mc_reco.query(self.l_selml[iptskim])
            list_df_mc_reco.append(df_mc_reco)
            df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[iptskim], "rb"))
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

        out_file.cd()
        his_resp_jet_fd.Write()
        his_ptc_ptjet_fd.Write()
        his_ptc_ptjet_z_fd.Write()
        his_njets.Write()
        out_file.Close()

    # pylint: disable=too-many-locals
    def process_unfolding(self):
        out_file = TFile.Open(self.n_fileeff, "update")
        list_df_mc_reco = []
        for iptskim, _ in enumerate(self.lpt_anbinmin):
            df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged[iptskim], "rb"))
            if "pt_jet" not in df_mc_reco.columns:
                print("Jet variables not found in the dataframe. Skipping process_response.")
                return
            if self.s_evtsel is not None:
                df_mc_reco = df_mc_reco.query(self.s_evtsel)
            if self.s_trigger is not None:
                df_mc_reco = df_mc_reco.query(self.s_trigger)
            df_mc_reco = df_mc_reco.query(self.l_selml[iptskim])
            list_df_mc_reco.append(df_mc_reco)
        df_mc_reco_merged = pd.concat(list_df_mc_reco)
        df_mc_reco_merged_prompt = df_mc_reco_merged[df_mc_reco_merged.ismcprompt == 1] # reconstructed & selected non-prompt jets

        zarray_reco = z_calc(df_mc_reco_merged_prompt.pt_jet, df_mc_reco_merged_prompt.phi_jet, df_mc_reco_merged_prompt.eta_jet,
                                    df_mc_reco_merged_prompt.pt_cand, df_mc_reco_merged_prompt.phi_cand, df_mc_reco_merged_prompt.eta_cand)

        zarray_gen = z_gen_calc(df_mc_reco_merged_prompt.pt_gen_jet, df_mc_reco_merged_prompt.phi_gen_jet, df_mc_reco_merged_prompt.eta_gen_jet,
                                    df_mc_reco_merged_prompt.pt_gen_cand, df_mc_reco_merged_prompt.delta_phi_gen_jet, df_mc_reco_merged_prompt.delta_eta_gen_jet)

        df_mc_reco_merged_prompt['z_reco'] = zarray_reco
        df_mc_reco_merged_prompt['z_gen'] = zarray_gen

        zbin =[]
        for zbin_i in range(21) :
            zbin.append(-0.5+zbin_i*0.1)
        zbinarray=array.array("d",zbin)
        jetptbin = [0.0,5.0,15.0,35.0]
        jetptbinarray=array.array("d",jetptbin)
        
        hz_gen_nocuts=TH1F("hz_gen_nocuts","",20, zbinarray)
        hz_gen_cuts=TH1F("hz_gen_cuts","",20,zbinarray)

        hzvsjetpt_reco_closure=TH2F("hzvsjetpt_reco_closure","",20,zbinarray,3, jetptbinarray)
        hzvsjetpt_gen_closure=TH2F("hzvsjetpt_gen_closure","",20,zbinarray,3, jetptbinarray)

        hzvsjetpt_reco=TH2F("hzvsjetpt_reco","",20,zbinarray,3,jetptbinarray)
        hzvsjetpt_gen=TH2F("hzvsjetpt_gen","",20,zbinarray,3,jetptbinarray)

        response_matrix = RooUnfoldResponse(hzvsjetpt_reco, hzvsjetpt_gen)
        response_matrix_closure = RooUnfoldResponse(hzvsjetpt_reco, hzvsjetpt_gen)

        random_number = TRandom3(0)
        for index, row in df_mc_reco_merged_prompt.iterrows():

            hzvsjetpt_reco.Fill(row['z_reco'],row['pt_jet'])
            hzvsjetpt_gen.Fill(row['z_gen'],row['pt_gen_jet'])

            response_matrix.Fill(row['z_reco'],row['pt_jet'],row['z_gen'],row['pt_gen_jet'])

            hz_gen_nocuts.Fill(row['z_gen'])
            if row['pt_jet'] > 5 and row['pt_jet'] <15 :
                hz_gen_cuts.Fill(row['z_gen'])

            if random_number.Rndm() < 0.1 :
                hzvsjetpt_reco_closure.Fill(row['z_reco'],row['pt_jet'])
                hzvsjetpt_gen_closure.Fill(row['z_gen'],row['pt_gen_jet'])
            else:
                response_matrix_closure.Fill(row['z_reco'],row['pt_jet'],row['z_gen'],row['pt_gen_jet'])
        kine_eff = hz_gen_nocuts.Clone("kine_eff")
        kine_eff.Divide(hz_gen_cuts)

        hzvsjetpt_reco.Write()
        hzvsjetpt_gen.Write()
        response_matrix.Write("response_matrix")
        response_matrix_closure.Write("response_matrix_closure")
        hzvsjetpt_reco_closure.Write("input_closure_reco")
        hzvsjetpt_gen_closure.Write("input_closure_gen")
        kine_eff.Write("kin_eff") 



        
        
    def process_valevents(self, file_index):
        dfevt = pickle.load(openfile(self.l_evtorig[file_index], "rb"))
        dfevt = dfevt.query("is_ev_rej==0")
        triggerlist = ["HighMultV0", "HighMultSPD", "HighMultV0", "INT7"]
        varlist = ["v0m_corr", "n_tracklets_corr", "perc_v0m"]
        nbinsvar = [100, 200, 200]
        minrvar = [0, 0, 0]
        maxrvar = [1500, 200, .5]
        fileevtroot = TFile.Open(self.l_evtvalroot[file_index], "recreate")
        hv0mvsperc = scatterplot(dfevt, "perc_v0m", "v0m_corr", 50000, 0, 100, 200, 0., 2000.)
        hv0mvsperc.SetName("hv0mvsperc")
        hv0mvsperc.Write()
        for ivar, var in enumerate(varlist):
            label = "hbitINT7vs%s" % (var)
            histoMB = TH1F(label, label, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
            fill_hist(histoMB, dfevt.query("trigger_hasbit_INT7==1")[var])
            histoMB.Sumw2()
            histoMB.Write()
            for trigger in triggerlist:
                triggerbit = "trigger_hasbit_%s==1" % trigger
                labeltriggerANDMB = "hbit%sANDINT7vs%s" % (trigger, var)
                labeltrigger = "hbit%svs%s" % (trigger, var)
                histotrigANDMB = TH1F(labeltriggerANDMB, labeltriggerANDMB, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
                histotrig = TH1F(labeltrigger, labeltrigger, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
                fill_hist(histotrigANDMB, dfevt.query(triggerbit + " and trigger_hasbit_INT7==1")[var])
                fill_hist(histotrig, dfevt.query(triggerbit)[var])
                histotrigANDMB.Sumw2()
                histotrig.Sumw2()
                histotrigANDMB.Write()
                histotrig.Write()
        dfevtnorm = pickle.load(openfile(self.l_evtorig[file_index], "rb"))
        hNorm = TH1F("hEvForNorm", ";;Normalisation", 2, 0.5, 2.5)
        hNorm.GetXaxis().SetBinLabel(1, "normsalisation factor")
        hNorm.GetXaxis().SetBinLabel(2, "selected events")
        nselevt = 0
        norm = 0
        if not dfevtnorm.empty:
            nselevt = len(dfevtnorm.query("is_ev_rej==0"))
            norm = getnormforselevt(dfevtnorm)
        hNorm.SetBinContent(1, norm)
        hNorm.SetBinContent(2, nselevt)
        hNorm.Write()
        fileevtroot.Close()

    def process_valevents_par(self):
        print("doing event validation", self.mcordata, self.period)
        create_folder_struc(self.d_val, self.l_path)
        arguments = [(i,) for i in range(len(self.l_evtorig))]
        self.parallelizer(self.process_valevents, arguments, self.p_chunksizeskim)
        mergerootfiles(self.l_evtvalroot, self.f_totevtvalroot)
