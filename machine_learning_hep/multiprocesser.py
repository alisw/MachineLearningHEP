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
import os
from machine_learning_hep.processer import Processer # pylint: disable=unused-import
from machine_learning_hep.utilities import merge_method, mergerootfiles, get_timestamp_string
class MultiProcesser: # pylint: disable=too-many-instance-attributes, too-many-statements
    species = "multiprocesser"
    def __init__(self, case, proc_class, datap, typean, run_param, mcordata):
        self.case = case
        self.datap = datap
        self.typean = typean
        self.run_param = run_param
        self.mcordata = mcordata
        self.prodnumber = len(datap["multi"][self.mcordata]["unmerged_tree_dir"])
        self.p_period = datap["multi"][self.mcordata]["period"]
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
        self.dlper_root = datap["multi"][self.mcordata]["unmerged_tree_dir"]
        self.dlper_pkl = datap["multi"][self.mcordata]["pkl"]
        self.dlper_pklsk = datap["multi"][self.mcordata]["pkl_skimmed"]
        self.dlper_pklml = datap["multi"][self.mcordata]["pkl_skimmed_merge_for_ml"]
        self.d_pklml_mergedallp = datap["multi"][self.mcordata]["pkl_skimmed_merge_for_ml_all"]
        self.d_pklevt_mergedallp = datap["multi"][self.mcordata]["pkl_evtcounter_all"]

        self.dlper_mcreweights = datap["multi"][self.mcordata]["mcreweights"]

        #namefiles pkl
        self.v_var_binning = datap["var_binning"]
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
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
        self.lper_evtml = [os.path.join(direc, self.n_evt) for direc in self.dlper_pklml]
        self.lper_evtorigml = [os.path.join(direc, self.n_evtorig) for direc in self.dlper_pklml]
        self.lptper_genml = [[os.path.join(direc, self.lpt_gensk[ipt]) \
                              for direc in self.dlper_pklml] \
                              for ipt in range(self.p_nptbins)]
        self.lpt_recoml_mergedallp = [os.path.join(self.d_pklml_mergedallp, self.lpt_recosk[ipt]) \
                                    for ipt in range(self.p_nptbins)]
        self.lpt_genml_mergedallp = [os.path.join(self.d_pklml_mergedallp, self.lpt_gensk[ipt]) \
                                    for ipt in range(self.p_nptbins)]
        self.f_evtml_mergedallp = os.path.join(self.d_pklml_mergedallp, self.n_evt)
        self.f_evtorigml_mergedallp = os.path.join(self.d_pklml_mergedallp, self.n_evtorig)
        self.lper_evt = [os.path.join(direc, self.n_evt) for direc in self.dlper_pkl]
        self.lper_evtorig = [os.path.join(direc, self.n_evtorig) for direc in self.dlper_pkl]

        self.dlper_reco_modapp = datap["mlapplication"][self.mcordata]["pkl_skimmed_dec"]
        self.dlper_reco_modappmerged = \
                datap["mlapplication"][self.mcordata]["pkl_skimmed_decmerged"]
        self.d_results = datap["analysis"][self.typean][self.mcordata]["results"]
        self.d_resulsallp = datap["analysis"][self.typean][self.mcordata]["resultsallp"]
        self.lpt_probcutpre = datap["mlapplication"]["probcutpresel"]
        self.lpt_probcut = datap["mlapplication"]["probcutoptimal"]
        self.f_evt_mergedallp = os.path.join(self.d_pklevt_mergedallp, self.n_evt)
        self.f_evtorig_mergedallp = os.path.join(self.d_pklevt_mergedallp, self.n_evtorig)

        self.lper_runlistrigger = datap["analysis"][self.typean][self.mcordata]["runselection"]

        self.lper_mcreweights = None
        if self.mcordata == "mc":
            self.lper_mcreweights = [os.path.join(direc, self.n_mcreweights)
                                     for direc in self.dlper_mcreweights]

        self.process_listsample = []
        for indexp in range(self.prodnumber):
            myprocess = proc_class(self.case, self.datap, self.run_param, self.mcordata,
                                   self.p_maxfiles[indexp], self.dlper_root[indexp],
                                   self.dlper_pkl[indexp], self.dlper_pklsk[indexp],
                                   self.dlper_pklml[indexp],
                                   self.p_period[indexp], self.p_chunksizeunp[indexp],
                                   self.p_chunksizeskim[indexp], self.p_nparall,
                                   self.p_fracmerge[indexp], self.p_seedmerge[indexp],
                                   self.dlper_reco_modapp[indexp],
                                   self.dlper_reco_modappmerged[indexp],
                                   self.d_results[indexp], self.typean,
                                   self.lper_runlistrigger[indexp], \
                                   self.dlper_mcreweights[indexp])
            self.process_listsample.append(myprocess)

        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.filemass_mergedall = os.path.join(self.d_resulsallp, self.n_filemass)
        self.fileeff_mergedall = os.path.join(self.d_resulsallp, self.n_fileeff)
        self.fileresp_mergedall = os.path.join(self.d_resulsallp, self.n_fileresp)

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
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_unpack_par()

    def multi_skim_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_skim_par()
        if self.p_dofullevtmerge is True:
            merge_method(self.lper_evt, self.f_evt_mergedallp)
            merge_method(self.lper_evtorig, self.f_evtorig_mergedallp)

    def multi_mergeml_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_mergeforml()

    def multi_mergeml_allinone(self):
        for ipt in range(self.p_nptbins):
            merge_method(self.lptper_recoml[ipt], self.lpt_recoml_mergedallp[ipt])
            if self.mcordata == "mc":
                merge_method(self.lptper_genml[ipt], self.lpt_genml_mergedallp[ipt])
        merge_method(self.lper_evtml, self.f_evtml_mergedallp)
        merge_method(self.lper_evtorigml, self.f_evtorigml_mergedallp)

    def multi_apply_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_applymodel_par()

    def multi_mergeapply_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_mergedec()

    def multi_histomass(self):
        for indexp in range(self.prodnumber):
            if self.p_useperiod[indexp] == 1:
                self.process_listsample[indexp].process_histomass()
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/mass/{get_timestamp_string()}/"
        mergerootfiles(self.lper_filemass, self.filemass_mergedall, tmp_merged)

    def multi_efficiency(self):
        for indexp in range(self.prodnumber):
            if self.p_useperiod[indexp] == 1:
                self.process_listsample[indexp].process_efficiency()
        tmp_merged = \
                f"/data/tmp/hadd/{self.case}_{self.typean}/efficiency/{get_timestamp_string()}/"
        mergerootfiles(self.lper_fileeff, self.fileeff_mergedall, tmp_merged)

    def multi_response(self):
        resp_exists = False
        for indexp in range(self.prodnumber):
            if self.p_useperiod[indexp] == 1:
                if hasattr(self.process_listsample[indexp], "process_response"):
                    resp_exists = True
                    self.process_listsample[indexp].process_response()
        if resp_exists:
            tmp_merged = \
                    f"/data/tmp/hadd/{self.case}_{self.typean}/response/{get_timestamp_string()}/"
            mergerootfiles(self.lper_fileresp, self.fileresp_mergedall, tmp_merged)

    def multi_scancuts(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_scancuts()
