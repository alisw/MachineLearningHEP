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
import multiprocessing as mp
import pickle
import os
import random as rd
import uproot
import pandas as pd
import numpy as np
from root_numpy import fill_hist # pylint: disable=import-error
from ROOT import TFile, TH1F# pylint: disable=import-error, no-name-in-module
from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import filter_bit_df, tag_bit_df
from machine_learning_hep.utilities import selectdfquery, selectdfrunlist, merge_method
from machine_learning_hep.utilities import list_folders, createlist, appendmainfoldertolist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from machine_learning_hep.utilities import mergerootfiles
from machine_learning_hep.utilities import get_timestamp_string
from machine_learning_hep.models import apply # pylint: disable=import-error
from machine_learning_hep.utilities_plot import scatterplotroot

class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                 d_results, d_val, typean, runlisttrigger, d_mcreweights):
        self.nprongs = datap["nprongs"]
        self.doml = datap["doml"]
        self.case = case
        self.typean = typean
        #directories
        self.d_root = d_root
        self.d_pkl = d_pkl
        self.d_pklsk = d_pklsk
        self.d_pkl_ml = d_pkl_ml
        self.d_results = d_results
        self.d_val = d_val
        self.d_mcreweights = d_mcreweights
        self.datap = datap
        self.mcordata = mcordata
        self.p_frac_merge = p_frac_merge
        self.p_rd_merge = p_rd_merge
        self.period = p_period
        self.runlist = run_param[self.period]
        self.run_param = run_param
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
        self.n_mcreweights = datap["files_names"]["namefile_mcweights"]

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
        self.l_histomass = createlist(self.d_results, self.l_path, self.n_filemass)
        self.l_histoeff = createlist(self.d_results, self.l_path, self.n_fileeff)


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
        self.lpt_recodec = None
        if self.doml is True:
            self.lpt_recodec = [self.n_reco.replace(".pkl", "%d_%d_%.2f.pkl" % \
                               (self.lpt_anbinmin[i], self.lpt_anbinmax[i], \
                                self.lpt_probcutpre[i])) for i in range(self.p_nptbins)]
        else:
            self.lpt_recodec = [self.n_reco.replace(".pkl", "%d_%d_std.pkl" % \
                               (self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                                                    for i in range(self.p_nptbins)]

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
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger


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
        arraysub = [0 for ival in range(len(dfreco))]
        n_tracklets_corr = dfreco["n_tracklets_corr"].values
        n_tracklets_corr_sub = None
        for iprong in range(self.nprongs):
            spdhits_thisprong = dfreco["spdhits_prong%s" % iprong].values
            ntrackletsthisprong = [1 if spdhits_thisprong[index] == 3 else 0 \
                                   for index in range(len(dfreco))]
            arraysub = np.add(ntrackletsthisprong, arraysub)
        n_tracklets_corr_sub = np.subtract(n_tracklets_corr, arraysub)

        dfreco["n_tracklets_corr_sub"] = n_tracklets_corr_sub
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
            if self.doml is True:
                if os.path.isfile(self.lpt_model[ipt]) is False:
                    print("Model file not present in bin %d" % ipt)
                mod = pickle.load(openfile(self.lpt_model[ipt], 'rb'))
                dfrecoskml = apply("BinaryClassification", [self.p_modelname], [mod],
                                   dfrecosk, self.v_train[ipt])
                probvar = "y_test_prob" + self.p_modelname
                dfrecoskml = dfrecoskml.loc[dfrecoskml[probvar] > self.lpt_probcutpre[ipt]]
            else:
                dfrecoskml = dfrecosk.query("isstd == 1")
            pickle.dump(dfrecoskml, openfile(self.mptfiles_recoskmldec[ipt][file_index], "wb"),
                        protocol=4)
    @staticmethod
    def callback(ex):
        print(ex)


    def parallelizer(self, function, argument_list, maxperchunk):
        chunks = [argument_list[x:x+maxperchunk] \
                  for x in range(0, len(argument_list), maxperchunk)]
        for chunk in chunks:
            print("Processing new chunck size=", maxperchunk)
            pool = mp.Pool(self.p_maxprocess)
            _ = [pool.apply_async(function, args=chunk[i],
                                  error_callback=self.callback) for i in range(len(chunk))]
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
    # pylint: disable=no-member
    def process_histomass(self):
        print("Doing masshisto", self.mcordata, self.period)
        print("Using run selection for mass histo", \
               self.runlistrigger[self.triggerbit], "for period", self.period)
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
    # pylint: disable=no-member
    def process_efficiency(self):
        print("Doing efficiencies", self.mcordata, self.period)
        print("Using run selection for eff histo", \
               self.runlistrigger[self.triggerbit], "for period", self.period)
        if self.doml is True:
            print("Doing ml analysis")
        else:
            print("No extra selection needed since we are doing std analysis")

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_efficiency_single, arguments, self.p_chunksizeunp)
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/histoeff_{self.period}/{get_timestamp_string()}/" # pylint: disable=line-too-long
        mergerootfiles(self.l_histoeff, self.n_fileeff, tmp_merged)

    # pylint: disable=too-many-locals
    def process_valevents(self, file_index):
        #dfevt = pickle.load(openfile(self.l_evtorig[file_index], "rb"))
        dfreco = pickle.load(openfile(self.l_reco[file_index], "rb"))
        fileevtroot = TFile.Open(self.l_evtvalroot[file_index], "recreate")
        dfreco = dfreco.query("is_ev_rej == 0")
        h_n_tracklets_corr = TH1F("h_n_tracklets_corr", "h_n_tracklets_corr", 100, -0.5, 99.5)
        h_run = TH1F("h_run", "h_run", 100000, 200000, 300000)
        h_trigg = TH1F("h_trigg", "h_trigg", 2, -0.5, 1.5)
        fill_hist(h_n_tracklets_corr, dfreco["n_tracklets_corr"])
        fill_hist(h_run, dfreco["run_number"])
        hmultvsrun = scatterplotroot(dfreco, "n_tracklets_corr",
                                     "run_number", 100, -0.5, 99.5, 100000,
                                     200000.5, 300000.5)
        hmultvsrun.SetName("hmultvsrun")
        fill_hist(h_trigg, dfreco["is_ev_rej_INT7"])
        hmultvsrun.Write()
        h_n_tracklets_corr.Write()
        hmultvsrun.Write()
        h_trigg.Write()
        h_run.Write()
        prof = hmultvsrun.ProfileY()
        prof.SetName("prof")
        prof.Write()
        fileevtroot.Close()

    def process_valevents_par(self):
        print("doing event validation", self.mcordata, self.period)
        create_folder_struc(self.d_val, self.l_path)
        tmp_merged = \
            f"/data/tmp/hadd/{self.case}_{self.typean}/val_{self.period}/{get_timestamp_string()}/"
        arguments = [(i,) for i in range(len(self.l_evtorig))]
        self.parallelizer(self.process_valevents, arguments, self.p_chunksizeskim)
        mergerootfiles(self.l_evtvalroot, self.f_totevtvalroot, tmp_merged)
