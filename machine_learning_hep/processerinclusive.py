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
from array import array
import sys
from copy import deepcopy
import multiprocessing as mp
import pickle
import os
import random as rd
import uproot
import pandas as pd
import numpy as np
from ROOT import TFile, TH1F, TH2F
from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import filter_bit_df, tag_bit_df
from machine_learning_hep.utilities import selectdfquery, merge_method
from machine_learning_hep.utilities import list_folders, createlist, appendmainfoldertolist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from machine_learning_hep.utilities import mergerootfiles
from machine_learning_hep.utilities import get_timestamp_string
from machine_learning_hep.models import apply # pylint: disable=import-error
from machine_learning_hep.utilities_plot import buildhisto, build2dhisto, fill2dhist, makefill3dhist
#from machine_learning_hep.logger import get_logger

class ProcesserInclusive: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processerinclusive'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, case, datap, mcordata, p_maxfiles,
                 d_root, d_pkl, p_period,
                 p_chunksizeunp, p_maxprocess, typean, runlisttrigger, d_results):
        #self.logger = get_logger()
        self.case = case
        self.typean = typean
        #directories
        self.d_root = d_root
        self.d_pkl = d_pkl
        self.datap = datap
        self.mcordata = mcordata
        self.period = p_period
        self.p_maxfiles = p_maxfiles
        self.p_chunksizeunp = p_chunksizeunp
        self.d_results = d_results

        #parameter names
        self.p_maxprocess = p_maxprocess
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
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileresp = datap["files_names"]["respfilename"]

        #selections
        self.s_reco_unp = datap["sel_reco_unp"]
        self.s_good_evt_unp = datap["sel_good_evt_unp"]
        self.s_cen_unp = datap["sel_cen_unp"]
        self.s_gen_unp = datap["sel_gen_unp"]

        #variables name
        self.v_all = datap["variables"]["var_all"]
        self.v_evt = datap["variables"]["var_evt"][self.mcordata]
        self.v_gen = datap["variables"]["var_gen"]
        self.v_evtmatch = datap["variables"]["var_evt_match"]

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
        self.l_histomass = createlist(self.d_results, self.l_path, self.n_filemass)
        self.l_historesp = createlist(self.d_results, self.l_path, self.n_fileresp)

        if self.mcordata == "mc":
            self.l_gen = createlist(self.d_pkl, self.l_path, self.n_gen)
        self.n_filemass = os.path.join(self.d_results, self.n_filemass)
        self.n_fileresp = os.path.join(self.d_results, self.n_fileresp)

        self.s_evtsel = datap["analysis"][self.typean]["evtsel"]
        self.s_jetsel_gen = datap["analysis"][self.typean]["jetsel_gen"]
        self.s_jetsel_reco = datap["analysis"][self.typean]["jetsel_reco"]
        self.s_trigger = datap["analysis"][self.typean]["triggersel"][self.mcordata]
        self.triggerbit = datap["analysis"][self.typean]["triggerbit"]
        self.runlistrigger = runlisttrigger

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
        # array of bin edges to use in histogram constructors
        self.var2binarray_reco = array("d", self.var2ranges_reco)
        self.var2ranges_gen = self.lvar2_binmin_gen.copy()
        self.var2ranges_gen.append(self.lvar2_binmax_gen[-1])
        # array of bin edges to use in histogram constructors
        self.var2binarray_gen = array("d", self.var2ranges_gen)

        # observable (z, shape,...)
        self.v_varshape_binning = datap["analysis"][self.typean]["var_binningshape"]
        self.v_varshape_binning_gen = datap["analysis"][self.typean]["var_binningshape_gen"]
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
        # array of bin edges to use in histogram constructors
        self.varshapebinarray_reco = array("d", self.varshaperanges_reco)
        self.varshaperanges_gen = self.lvarshape_binmin_gen.copy()
        self.varshaperanges_gen.append(self.lvarshape_binmax_gen[-1])
        # array of bin edges to use in histogram constructors
        self.varshapebinarray_gen = array("d", self.varshaperanges_gen)

    def unpack(self, file_index):
        treeevtorig = uproot.open(self.l_root[file_index])[self.n_treeevt]
        try:
            dfevtorig = treeevtorig.pandas.df(branches=self.v_evt)
        except Exception as e: # pylint: disable=broad-except
            print('Missing variable in the event root tree', str(e))
            print('Missing variable in the candidate root tree')
            print('I am sorry, I am dying ...\n \n \n')
            sys.exit()

        dfevtorig = selectdfquery(dfevtorig, self.s_cen_unp)
        dfevtorig = dfevtorig.reset_index(drop=True)
        pickle.dump(dfevtorig, openfile(self.l_evtorig[file_index], "wb"), protocol=4)
        dfevt = selectdfquery(dfevtorig, self.s_good_evt_unp)
        dfevt = dfevt.reset_index(drop=True)
        pickle.dump(dfevt, openfile(self.l_evt[file_index], "wb"), protocol=4)


        treereco = uproot.open(self.l_root[file_index])[self.n_treereco]
        try:
            dfreco = treereco.pandas.df(branches=self.v_all)
        except Exception as e: # pylint: disable=broad-except
            print('Missing variable in the event root tree', str(e))
            print('Missing variable in the candidate root tree')
            print('I am sorry, I am dying ...\n \n \n')
            sys.exit()
        dfreco = selectdfquery(dfreco, self.s_reco_unp)
        dfreco = pd.merge(dfreco, dfevt, on=self.v_evtmatch)
        dfreco = dfreco.reset_index(drop=True)
        pickle.dump(dfreco, openfile(self.l_reco[file_index], "wb"), protocol=4)

        if self.mcordata == "mc":
            treegen = uproot.open(self.l_root[file_index])[self.n_treegen]
            dfgen = treegen.pandas.df(branches=self.v_gen)
            dfgen = pd.merge(dfgen, dfevtorig, on=self.v_evtmatch)
            dfgen = selectdfquery(dfgen, self.s_gen_unp)
            dfgen = dfgen.reset_index(drop=True)
            pickle.dump(dfgen, openfile(self.l_gen[file_index], "wb"), protocol=4)

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

    def process_histomass_single(self, index):
        myfile = TFile.Open(self.l_histomass[index], "recreate")
        df = pickle.load(openfile(self.l_reco[index], "rb"))
        if self.s_evtsel is not None:
            df = df.query(self.s_evtsel)
        if self.s_jetsel_reco is not None:
            df = df.query(self.s_jetsel_reco)
        if self.s_trigger is not None:
            df = df.query(self.s_trigger)
        if self.runlistrigger is not None:
            df = selectdfrunlist(df, \
                self.run_param[self.runlistrigger], "run_number")
        h_jetptvsshape = TH2F("h_jetptvsshape", "", \
            self.p_nbinshape_reco, self.varshapebinarray_reco,
            len(self.lvar2_binmin_reco), self.var2binarray_reco)
        fill2dhist(df, h_jetptvsshape, self.v_varshape_binning, self.v_var2_binning)
        h_jetptvsshape.Write()

        for ijet in range(self.p_nbin2_reco):
            suffix = "%s_%.2f_%.2f" % \
                     (self.v_var2_binning, self.lvar2_binmin_reco[ijet],
                      self.lvar2_binmax_reco[ijet])
            df_bin = seldf_singlevar(df, self.v_var2_binning,
                                         self.lvar2_binmin_reco[ijet],
                                         self.lvar2_binmax_reco[ijet])
            h_jetptvsshape = TH2F("h_jetptvsshape" + suffix, "", \
                self.p_nbinshape_reco, self.varshapebinarray_reco,
                len(self.lvar2_binmin_reco), self.var2binarray_reco)
            fill2dhist(df_bin, h_jetptvsshape, self.v_varshape_binning, self.v_var2_binning)
            h_jetptvsshape.Write()

    def process_histomass(self):
        print("Doing masshisto", self.mcordata, self.period)
        print("Using run selection for mass histo", \
               self.runlistrigger, "for period", self.period)

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_histomass_single, arguments, self.p_chunksizeunp) # pylint: disable=no-member
        tmp_merged = \
            f"/data/tmp/hadd/{self.case}_{self.typean}/mass_{self.period}/{get_timestamp_string()}/"
        mergerootfiles(self.l_histomass, self.n_filemass, tmp_merged)

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
        print("ECCOMI")
        out_file = TFile.Open(self.l_historesp[index], "recreate")
        list_df_mc_reco = []
        list_df_mc_gen = []
        h = TH1F("h", "h", 100, 0, 100)
        h.Write()
    def process_response(self):
        print("Doing response", self.mcordata, self.period)
        print("Using run selection for resp histo", \
               self.runlistrigger, "for period", self.period)

        create_folder_struc(self.d_results, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.process_response_single, arguments, self.p_chunksizeunp)
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/historesp_{self.period}/{get_timestamp_string()}/" # pylint: disable=line-too-long
        mergerootfiles(self.l_historesp, self.n_fileresp, tmp_merged)
