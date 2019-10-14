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
                 d_results, d_val, typean, runlisttrigger):
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

        self.lvar2_binmin_reco = datap["analysis"][self.typean]["sel_binmin2_reco"]
        self.lvar2_binmax_reco = datap["analysis"][self.typean]["sel_binmax2_reco"]
        self.p_nbin2_reco = len(self.lvar2_binmin_reco)

        self.lvar2_binmin_gen = datap["analysis"][self.typean]["sel_binmin2_gen"]
        self.lvar2_binmax_gen = datap["analysis"][self.typean]["sel_binmax2_gen"]
        self.p_nbin2_gen = len(self.lvar2_binmin_gen)

        self.lvarshape_binmin_reco = datap["analysis"][self.typean]["sel_binminshape_reco"]
        self.lvarshape_binmax_reco = datap["analysis"][self.typean]["sel_binmaxshape_reco"]
        self.p_nbinshape_reco = len(self.lvarshape_binmin_reco)

        self.lvarshape_binmin_gen = datap["analysis"][self.typean]["sel_binminshape_gen"]
        self.lvarshape_binmax_gen = datap["analysis"][self.typean]["sel_binmaxshape_gen"]
        self.p_nbinshape_gen = len(self.lvarshape_binmin_gen)
        
        self.closure_frac = datap["analysis"][self.typean]["sel_closure_frac"]

        self.var2ranges = self.lvar2_binmin.copy()
        self.var2ranges.append(self.lvar2_binmax[-1])
        self.var2ranges_reco = self.lvar2_binmin_reco.copy()
        self.var2ranges_reco.append(self.lvar2_binmax_reco[-1])
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        self.varshaperanges_reco = self.lvarshape_binmin_reco.copy()
        self.varshaperanges_reco.append(self.lvarshape_binmax_reco[-1])
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])

        self.lpt_finbinmin = datap["analysis"][self.typean]["sel_an_binmin"]
        self.lpt_finbinmax = datap["analysis"][self.typean]["sel_an_binmax"]
        self.p_nptfinbins = len(self.lpt_finbinmin)
        self.bin_matching = datap["analysis"][self.typean]["binning_matching"]
        #self.sel_final_fineptbins = datap["analysis"][self.typean]["sel_final_fineptbins"]
        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
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
            for ibin2 in range(len(self.lvar2_binmin_reco)):
                suffix = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                         (self.v_var_binning, self.lpt_finbinmin[ipt],
                          self.lpt_finbinmax[ipt], self.lpt_probcutfin[bin_id],
                          self.v_var2_binning, self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                h_invmass = TH1F("hmass" + suffix, "", self.p_num_bins,
                                 self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                h_invmass_weight = TH1F("h_invmass_weight" + suffix, "", self.p_num_bins,
                                        self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
                df_bin = seldf_singlevar(df, self.v_var2_binning,
                                         self.lvar2_binmin_reco[ibin2], self.lvar2_binmax_reco[ibin2])
                print("Using run selection for mass histo",
                      self.runlistrigger[self.triggerbit], "for period", self.period)
                df_bin = selectdfrunlist(df_bin, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
                fill_hist(h_invmass, df_bin.inv_mass)
                if "INT7" not in self.triggerbit and self.mcordata == "data":
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
                if "pt_jet" in df_bin.columns:
                    zarray = z_calc(df_bin.pt_jet, df_bin.phi_jet, df_bin.eta_jet,
                                    df_bin.pt_cand, df_bin.phi_jet, df_bin.eta_jet)
                    h_zvsinvmass = TH2F("hzvsmass" + suffix, "", 5000, 1.00, 6.00, self.p_nbinshape_reco, self.lvarshape_binmin_reco[0], self.lvarshape_binmax_reco[-1])
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
        for ibin2 in range(len(self.lvar2_binmin_reco)):
            stringbin2 = "_%s_%.2f_%.2f" % (self.v_var2_binning, \
                                        self.lvar2_binmin_reco[ibin2], \
                                        self.lvar2_binmax_reco[ibin2])
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

            bincounter = 0
            for ipt in range(self.p_nptfinbins):
                bin_id = self.bin_matching[ipt]
                df_mc_reco = pickle.load(openfile(self.lpt_recodecmerged[bin_id], "rb"))
                if self.s_evtsel is not None:
                    df_mc_reco = df_mc_reco.query(self.s_evtsel)
                if self.s_trigger is not None:
                    df_mc_reco = df_mc_reco.query(self.s_trigger)
                print("Using run selection for eff histo",
                      self.runlistrigger[self.triggerbit], "for period", self.period)
                df_mc_reco = selectdfrunlist(df_mc_reco, \
                         self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
                df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[bin_id], "rb"))
                df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
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
   
        
        out_file = TFile.Open(self.n_fileeff, "update")
        list_df_mc_reco = []
        list_df_mc_gen = []
        for iptskim, _ in enumerate(self.lpt_anbinmin):

            df_mc_gen = pickle.load(openfile(self.lpt_gendecmerged[iptskim], "rb"))
            df_mc_gen = selectdfrunlist(df_mc_gen, \
                    self.run_param[self.runlistrigger[self.triggerbit]], "run_number")
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
            list_df_mc_gen.append(df_mc_gen)
            
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
            df_mc_gen = df_mc_gen.query(self.s_presel_gen_eff)
            list_df_mc_gen.append(df_mc_gen)
            
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

        random_number = TRandom3(0)
        random_number_result=0.0
        for row in df_mc_reco_merged_prompt.itertuples():

            random_number_result=random_number.Rndm()
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
                response_matrix_closure.Fill(row.z_reco,row.pt_jet,row.z_gen,row.pt_gen_jet)

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


        
        
    def process_valevents(self, file_index):
        dfevt = pickle.load(openfile(self.l_evtorig[file_index], "rb"))
        dfevt = dfevt.query("is_ev_rej==0")
        dfevtmb = pickle.load(openfile(self.l_evtorig[file_index], "rb"))
        dfevtmb = dfevtmb.query("is_ev_rej==0")
        myrunlisttrigmb = self.runlistrigger["INT7"]
        dfevtselmb = selectdfrunlist(dfevtmb, self.run_param[myrunlisttrigmb], "run_number")
        triggerlist = ["INT7", "HighMultV0", "HighMultSPD"]
        varlist = ["v0m_corr", "n_tracklets_corr", "perc_v0m"]
        nbinsvar = [100, 200, 200]
        minrvar = [0, 0, 0]
        maxrvar = [1500, 200, .5]
        fileevtroot = TFile.Open(self.l_evtvalroot[file_index], "recreate")
        hv0mvsperc = scatterplot(dfevt, "perc_v0m", "v0m_corr", 50000, 0, 100, 200, 0., 2000.)
        hv0mvsperc.SetName("hv0mvsperc")
        hv0mvsperc.Write()
        dfevtnorm = pickle.load(openfile(self.l_evtorig[file_index], "rb"))
        hntrklsperc = scatterplot(dfevt, "perc_v0m", "n_tracklets_corr", 50000, 0, 100, 200, 0., 2000.)
        hntrklsperc.SetName("hntrklsperc")
        hntrklsperc.Write()
        for ivar, var in enumerate(varlist):
            label = "hbitINT7vs%s" % (var)
            histoMB = TH1F(label, label, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
            fill_hist(histoMB, dfevtselmb.query("trigger_hasbit_INT7==1")[var])
            histoMB.Sumw2()
            histoMB.Write()
            for trigger in triggerlist:
                triggerbit = "trigger_hasbit_%s==1" % trigger
                labeltriggerANDMB = "hbit%sANDINT7vs%s" % (trigger, var)
                labeltrigger = "hbit%svs%s" % (trigger, var)
                histotrigANDMB = TH1F(labeltriggerANDMB, labeltriggerANDMB, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
                histotrig = TH1F(labeltrigger, labeltrigger, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
                myrunlisttrig = self.runlistrigger[trigger]
                ev = len(dfevt)
                dfevtsel = selectdfrunlist(dfevt, self.run_param[myrunlisttrig], "run_number")
                if len(dfevtsel) < ev:
                    print("Reduced number of events in trigger", trigger)
                    print(ev, len(dfevtsel))
                fill_hist(histotrigANDMB, dfevtsel.query(triggerbit + " and trigger_hasbit_INT7==1")[var])
                fill_hist(histotrig, dfevtsel.query(triggerbit)[var])
                histotrigANDMB.Sumw2()
                histotrig.Sumw2()
                histotrigANDMB.Write()
                histotrig.Write()
                hSelMult = TH1F('sel_' + labeltrigger, 'sel_' + labeltrigger, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
                hNoVtxMult = TH1F('novtx_' + labeltrigger, 'novtx_' + labeltrigger, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])
                hVtxOutMult = TH1F('vtxout_' + labeltrigger, 'vtxout_' + labeltrigger, nbinsvar[ivar], minrvar[ivar], maxrvar[ivar])

                # multiplicity dependent normalisation
                dftrg = dfevtnorm.query(triggerbit)
                dfsel = dftrg.query('is_ev_rej == 0')
                df_to_keep = filter_bit_df(dftrg, 'is_ev_rej', [[], [0, 5, 6, 10, 11]])
                # events with reco vtx after previous selection
                tag_vtx = tag_bit_df(df_to_keep, 'is_ev_rej', [[], [1, 2, 7, 12]])
                df_no_vtx = df_to_keep[~tag_vtx.values]
                # events with reco zvtx > 10 cm after previous selection
                df_bit_zvtx_gr10 = filter_bit_df(df_to_keep, 'is_ev_rej', [[3], [1, 2, 7, 12]])

                fill_hist(hSelMult, dfsel[var])
                fill_hist(hNoVtxMult, df_no_vtx[var])
                fill_hist(hVtxOutMult, df_bit_zvtx_gr10[var])

                hSelMult.Write()
                hNoVtxMult.Write()
                hVtxOutMult.Write()

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
