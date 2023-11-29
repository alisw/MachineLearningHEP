#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
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
import os
from machine_learning_hep.processer import Processer # pylint: disable=unused-import
from machine_learning_hep.utilities import merge_method, mergerootfiles, get_timestamp_string
from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict
from machine_learning_hep.logger import get_logger

class MultiProcesser: # pylint: disable=too-many-instance-attributes, too-many-statements, consider-using-f-string, too-many-branches
    species = "multiprocesser"
    logger = get_logger()

    def __init__(self, case, proc_class, datap, typean, run_param, mcordata):
        self.case = case
        self.datap = datap
        self.typean = typean
        self.run_param = run_param
        self.mcordata = mcordata
        self.prodnumber = len(datap["multi"][self.mcordata]["unmerged_tree_dir"])
        self.p_period = datap["multi"][self.mcordata]["period"]
        self.select_period = datap["multi"][self.mcordata]["select_period"]
        self.p_seedmerge = datap["multi"][self.mcordata]["seedmerge"]
        self.p_fracmerge = datap["multi"][self.mcordata]["fracmerge"]
        self.p_maxfiles = datap["multi"][self.mcordata]["maxfiles"]
        self.p_chunksizeunp = datap["multi"][self.mcordata]["chunksizeunp"]
        self.p_chunksizeskim = datap["multi"][self.mcordata]["chunksizeskim"]
        self.p_nparall = datap["multi"][self.mcordata]["nprocessesparallel"]
        self.lpt_anbinmin = datap["sel_skim_binmin"]
        self.lpt_anbinmax = datap["sel_skim_binmax"]
        self.p_nptbins = len(datap["sel_skim_binmax"])
        self.p_dofullevtmerge = datap["dofullevtmerge"]

        #directories
        self.dlper_root = []
        self.dlper_pkl = []
        self.dlper_pklsk = []
        self.dlper_pklml = []
        self.d_prefix = datap["multi"][self.mcordata].get("prefix_dir", "")
        self.d_prefix_app = datap["mlapplication"][self.mcordata].get("prefix_dir_app", "")
        self.d_prefix_res = datap["analysis"][self.typean][self.mcordata].get("prefix_dir_res", "")

        dp = datap["multi"][self.mcordata]
        self.dlper_root = [self.d_prefix + p for p in dp["unmerged_tree_dir"]]
        self.dlper_pkl = [self.d_prefix + p for p in dp["pkl"]]
        self.dlper_pklsk = [self.d_prefix + p for p in dp["pkl_skimmed"]]
        self.dlper_pklml = [self.d_prefix + p for p in dp["pkl_skimmed_merge_for_ml"]]
        self.d_pklml_mergedallp = self.d_prefix + dp["pkl_skimmed_merge_for_ml_all"]
        self.d_pklevt_mergedallp = self.d_prefix + dp["pkl_evtcounter_all"]
        self.dlper_mcreweights = datap["multi"][self.mcordata]["mcreweights"]

        #namefiles pkl
        self.v_var_binning = datap["var_binning"]
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_evt_count_ml = datap["files_names"].get("namefile_evt_count", "evtcount.yaml")
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.n_mcreweights = datap["files_names"]["namefile_mcweights"]
        self.lpt_recosk = [self.n_reco.replace(".pkl", "_%s%d_%d.pkl" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lpt_gensk = [self.n_gen.replace(".pkl", "_%s%d_%d.pkl" % \
                          (self.v_var_binning, self.lpt_anbinmin[i], self.lpt_anbinmax[i])) \
                          for i in range(self.p_nptbins)]
        self.lptper_recoml = [[os.path.join(direc, self.lpt_recosk[ipt]) \
                               for direc in self.dlper_pklml] \
                               for ipt in range(self.p_nptbins)]
        self.lper_evt_count_ml = [os.path.join(direc, self.n_evt_count_ml) \
                for direc in self.dlper_pklml]
        self.lptper_genml = [[os.path.join(direc, self.lpt_gensk[ipt]) \
                              for direc in self.dlper_pklml] \
                              for ipt in range(self.p_nptbins)]
        self.lpt_recoml_mergedallp = \
                [os.path.join(self.d_pklml_mergedallp, self.lpt_recosk[ipt]) \
                 for ipt in range(self.p_nptbins)]
        self.lpt_genml_mergedallp = \
                [os.path.join(self.d_pklml_mergedallp, self.lpt_gensk[ipt]) \
                 for ipt in range(self.p_nptbins)]
        self.f_evtml_count = \
                 os.path.join(self.d_pklml_mergedallp, self.n_evt_count_ml)
        self.lper_evt = [os.path.join(direc, self.n_evt) for direc in self.dlper_pkl]
        self.lper_evtorig = \
                [os.path.join(direc, self.n_evtorig) for direc in self.dlper_pkl]

        dp = datap["mlapplication"][self.mcordata]
        self.dlper_reco_modapp = [self.d_prefix_app + p for p in dp["pkl_skimmed_dec"]]
        self.dlper_reco_modappmerged = [self.d_prefix_app + p for p in dp["pkl_skimmed_decmerged"]]

        dp = datap["analysis"][self.typean][self.mcordata]
        self.d_results = [self.d_prefix_res + p for p in dp["results"]]
        self.d_resultsallp = self.d_prefix_res + dp["resultsallp"]

        self.f_evt_mergedallp = os.path.join(self.d_pklevt_mergedallp, self.n_evt)
        self.f_evtorig_mergedallp = \
                 os.path.join(self.d_pklevt_mergedallp, self.n_evtorig)

        self.lper_runlistrigger = datap["analysis"][self.typean][self.mcordata]["runselection"]

        self.lper_mcreweights = None
        if self.mcordata == "mc":
            self.lper_mcreweights = [os.path.join(direc, self.n_mcreweights)
                                     for direc in self.dlper_mcreweights]

        self.process_listsample = []
        for indexp in range(self.prodnumber):
            if self.select_period[indexp]>0:
                myprocess = proc_class(self.case, self.datap, self.run_param, self.mcordata,
                                       self.p_maxfiles[indexp], self.dlper_root[indexp],
                                       self.dlper_pkl[indexp], self.dlper_pklsk[indexp],
                                       self.dlper_pklml[indexp],
                                       self.p_period[indexp], indexp, self.p_chunksizeunp[indexp],
                                       self.p_chunksizeskim[indexp], self.p_nparall,
                                       self.p_fracmerge[indexp], self.p_seedmerge[indexp],
                                       self.dlper_reco_modapp[indexp],
                                       self.dlper_reco_modappmerged[indexp],
                                       self.d_results[indexp], self.typean,
                                       self.lper_runlistrigger[indexp], \
                                       self.dlper_mcreweights[indexp])
                self.process_listsample.append(myprocess)
            else:
                self.logger.info('Period [%s] excluded from the analysis', self.p_period[indexp])
                continue

        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.filemass_mergedall = os.path.join(self.d_resultsallp, self.n_filemass)
        self.fileeff_mergedall = os.path.join(self.d_resultsallp, self.n_fileeff)
        self.fileresp_mergedall = os.path.join(self.d_resultsallp, self.n_fileresp)

        self.p_useperiod = datap["analysis"][self.typean]["useperiod"]
        self.lper_filemass = []
        self.lper_fileeff = []
        self.lper_fileresp = []
        self.lper_normfiles = []
        for i, direc in enumerate(self.d_results):
            if self.p_useperiod[i] == 1:
                self.lper_filemass.append(os.path.join(direc, self.n_filemass))
                self.lper_fileeff.append(os.path.join(direc, self.n_fileeff))
                self.lper_fileresp.append(os.path.join(direc, self.n_fileresp))

    def multi_unpack_allperiods(self):
        for indexp, _ in enumerate(self.process_listsample):
            self.process_listsample[indexp].process_unpack_par()

    def multi_skim_allperiods(self):
        for indexp, _ in enumerate(self.process_listsample):
            self.process_listsample[indexp].process_skim_par()
        if self.p_dofullevtmerge is True:
            merge_method(self.lper_evt, self.f_evt_mergedallp)
            merge_method(self.lper_evtorig, self.f_evtorig_mergedallp)

    def multi_mergeml_allperiods(self):
        for indexp, _ in enumerate(self.process_listsample):
            self.process_listsample[indexp].process_mergeforml()

    def multi_mergeml_allinone(self):
        for ipt in range(self.p_nptbins):
            for indexp in range(self.prodnumber):
                if self.select_period[indexp] == 0:
                    self.lptper_recoml[ipt].remove(self.lptper_recoml[ipt][indexp])
                    if self.mcordata == "mc":
                        self.lptper_genml[ipt].remove(self.lptper_genml[ipt][indexp])
            merge_method(self.lptper_recoml[ipt], self.lpt_recoml_mergedallp[ipt])
            if self.mcordata == "mc":
                merge_method(self.lptper_genml[ipt], self.lpt_genml_mergedallp[ipt])

        count_evt = 0
        count_evtorig = 0
        for indexp in range(self.prodnumber):
            if self.select_period[indexp] == 0:
                self.lper_evt_count_ml.remove(self.lper_evt_count_ml[indexp])
        for evt_count_file in self.lper_evt_count_ml:
            count_dict = parse_yaml(evt_count_file)
            count_evt += count_dict["evt"]
            count_evtorig += count_dict["evtorig"]

        dump_yaml_from_dict({"evt": count_evt, "evtorig": count_evtorig}, self.f_evtml_count)

    def multi_apply_allperiods(self):
        for indexp, _ in enumerate(self.process_listsample):
            self.process_listsample[indexp].process_applymodel_par()

    def multi_mergeapply_allperiods(self):
        for indexp, _ in enumerate(self.process_listsample):
            self.process_listsample[indexp].process_mergedec()

    def multi_histomass(self):
        for indexp, _ in enumerate(self.process_listsample):
            if self.p_useperiod[indexp] == 1:
                self.process_listsample[indexp].process_histomass()
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/mass/{get_timestamp_string()}/"
        self.logger.debug('merging all')
        mergerootfiles(self.lper_filemass, self.filemass_mergedall, tmp_merged)

    def multi_efficiency(self):
        for indexp, _ in enumerate(self.process_listsample):
            if self.p_useperiod[indexp] == 1:
                self.process_listsample[indexp].process_efficiency()
        tmp_merged = \
                f"/data/tmp/hadd/{self.case}_{self.typean}/efficiency/{get_timestamp_string()}/"
        mergerootfiles(self.lper_fileeff, self.fileeff_mergedall, tmp_merged)

    def multi_response(self):
        resp_exists = False
        for indexp, _ in enumerate(self.process_listsample):
            if self.p_useperiod[indexp] == 1:
                if hasattr(self.process_listsample[indexp], "process_response"):
                    resp_exists = True
                    self.process_listsample[indexp].process_response()
        if resp_exists:
            tmp_merged = \
                    f"/data/tmp/hadd/{self.case}_{self.typean}/response/{get_timestamp_string()}/"
            mergerootfiles(self.lper_fileresp, self.fileresp_mergedall, tmp_merged)

    def multi_scancuts(self):
        for indexp, _ in enumerate(self.process_listsample):
            self.process_listsample[indexp].process_scancuts()
