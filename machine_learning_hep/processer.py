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
from root_numpy import fill_hist # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TH1F # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import filter_bit_df, tag_bit_df
from machine_learning_hep.utilities import selectdfquery, selectdfrunlist, merge_method
from machine_learning_hep.utilities import list_folders, createlist, appendmainfoldertolist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from machine_learning_hep.utilities import mergerootfiles
from machine_learning_hep.models import apply # pylint: disable=import-error
#from machine_learning_hep.globalfitter import fitter
from machine_learning_hep.selectionutils import getnormforselevt

class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged, d_results):

        #directories
        self.d_root = d_root
        self.d_pkl = d_pkl
        self.d_pklsk = d_pklsk
        self.d_pkl_ml = d_pkl_ml
        self.d_results = d_results
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
        #parameter fitter
        self.p_sgnfunc = datap["analysis"]["sgnfunc"]
        self.p_bkgfunc = datap["analysis"]["bkgfunc"]
        self.p_masspeak = datap["analysis"]["masspeak"]
        self.p_massmin = datap["analysis"]["massmin"]
        self.p_massmax = datap["analysis"]["massmax"]
        self.p_rebin = datap["analysis"]["rebin"]
        self.p_includesecpeak = datap["analysis"]["includesecpeak"]
        self.p_masssecpeak = datap["analysis"]["masssecpeak"]
        self.p_fixedmean = datap["analysis"]["FixedMean"]
        self.p_fixingaussigma = datap["analysis"]["SetFixGaussianSigma"]
        self.p_fixingausmean = datap["analysis"]["SetInitialGaussianMean"]
        self.p_dolike = datap["analysis"]["dolikelihood"]
        self.p_sigmaarray = datap["analysis"]["sigmaarray"]
        self.p_fixedsigma = datap["analysis"]["FixedSigma"]
        self.p_casefit = datap["analysis"]["fitcase"]
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
        self.n_evtorigroot = datap["files_names"]["namefile_evtorigroot"]
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
        self.l_evtorigroot = createlist(self.d_pkl, self.l_path, self.n_evtorigroot)


        if self.mcordata == "mc":
            self.l_gen = createlist(self.d_pkl, self.l_path, self.n_gen)

        self.f_totevt = os.path.join(self.d_pkl, self.n_evt)
        self.f_totevtorig = os.path.join(self.d_pkl, self.n_evtorig)
        self.f_totevtorigroot = os.path.join(self.d_pkl, self.n_evtorigroot)

        self.usefit = datap["analysis"]["usefit"]
        self.p_modelname = datap["analysis"]["modelname"]
        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])
        self.lpt_model = datap["analysis"]["modelsperptbin"]
        self.dirmodel = datap["ml"]["mlout"]
        self.lpt_model = appendmainfoldertolist(self.dirmodel, self.lpt_model)
        self.lpt_probcutpre = datap["analysis"]["probcutpresel"][self.mcordata]
        self.lpt_probcutfin = datap["analysis"]["probcutoptimal"]

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

        self.p_mass_fit_lim = datap["analysis"]['mass_fit_lim']
        self.p_bin_width = datap["analysis"]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                    self.p_bin_width))
        self.l_selml = ["y_test_prob%s>%s" % (self.p_modelname, self.lpt_probcutfin[ipt]) \
                       for ipt in range(self.p_nptbins)]
        self.s_presel_gen_eff = datap["analysis"]['presel_gen_eff']

        self.lvar2_binmin = datap["analysis"]["sel_binmin2"]
        self.lvar2_binmax = datap["analysis"]["sel_binmax2"]
        self.v_var2_binning = datap["analysis"]["var_binning2"]

        self.lpt_finbinmin = datap["analysis"]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"]["sel_an_binmax"]
        self.bin_matching = datap["analysis"]["binning_matching"]
        self.p_nptfinbins = len(self.lpt_finbinmin)

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

        fileevtroot = TFile.Open(self.l_evtorigroot[file_index], "recreate")
        hNorm = TH1F("hEvForNorm", ";;Normalisation", 2, 0.5, 2.5)
        hNorm.GetXaxis().SetBinLabel(1, "normsalisation factor")
        hNorm.GetXaxis().SetBinLabel(2, "selected events")
        nselevt = 0
        norm = 0
        # Handle silent weird behaviour of Pandas if dataframe is empty
        # Otherwise, if it is empty it might just silently return from this frunction for some
        # reason and everything what follows would just be skipped.
        if not dfevt.empty:
            nselevt = len(dfevt.query("is_ev_rej==0"))
            norm = getnormforselevt(dfevt)
        hNorm.SetBinContent(1, norm)
        hNorm.SetBinContent(2, nselevt)
        hNorm.Write()
        fileevtroot.Close()

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
        mergerootfiles(self.l_evtorigroot, self.f_totevtorigroot)
        if self.p_dofullevtmerge is True:
            merge_method(self.l_evt, self.f_totevt)
            merge_method(self.l_evtorig, self.f_totevtorig)

    def process_applymodel_par(self):
        print("doing apply model", self.mcordata, self.period)
        create_folder_struc(self.d_pkl_dec, self.l_path)
        for ipt in range(self.p_nptbins):
            arguments = [(i,) for i in range(len(self.mptfiles_recosk[ipt]))]
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
            df = seldf_singlevar(df, self.v_var_binning, \
                                 self.lpt_finbinmin[ipt], self.lpt_finbinmax[ipt])
            for ibin2 in range(len(self.lvar2_binmin)):
                suffix = "%s%d_%d_%.2f%s_%d_%d" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                df_bin = seldf_singlevar(df, self.v_var2_binning,
                                         self.lvar2_binmin[ibin2], self.lvar2_binmax[ibin2])
                fill_hist(h_invmass, df_bin.inv_mass)
                myfile.cd()
                h_invmass.Write()
    # pylint: disable=line-too-long
    def process_efficiency(self):
        out_file = TFile.Open(self.n_fileeff, "recreate")
        for ibin2 in range(len(self.lvar2_binmin)):
            stringbin2 = "_%s_%d_%d" % (self.v_var2_binning,
                                        self.lvar2_binmin[ibin2],
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
