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
from machine_learning_hep.processerinclusive import ProcesserInclusive # pylint: disable=unused-import
from machine_learning_hep.utilities import merge_method, mergerootfiles, get_timestamp_string
class MultiProcesserInclusive: # pylint: disable=too-many-instance-attributes, too-many-statements
    species = "multiprocesserinclusive"
    def __init__(self, case, proc_class, datap, typean, mcordata):
        self.case = case
        self.datap = datap
        self.typean = typean
        self.mcordata = mcordata
        self.prodnumber = len(datap["multi"][self.mcordata]["unmerged_tree_dir"])
        self.p_period = datap["multi"][self.mcordata]["period"]
        self.p_maxfiles = datap["multi"][self.mcordata]["maxfiles"]
        self.p_chunksizeunp = datap["multi"][self.mcordata]["chunksizeunp"]
        self.p_nparall = datap["multi"][self.mcordata]["nprocessesparallel"]

        #directories
        self.dlper_root = datap["multi"][self.mcordata]["unmerged_tree_dir"]
        self.dlper_pkl = datap["multi"][self.mcordata]["pkl"]
        self.d_results = datap["analysis"][self.typean][self.mcordata]["results"]
        self.d_resulsallp = datap["analysis"][self.typean][self.mcordata]["resultsallp"]

        #namefiles pkl
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_gen = datap["files_names"]["namefile_gen"]

        self.lper_runlistrigger = datap["analysis"][self.typean][self.mcordata]["runselection"]

        self.process_listsample = []
        for indexp in range(self.prodnumber):
            print(self.case, self.mcordata, self.p_maxfiles[indexp],
                  self.dlper_root[indexp], self.p_period[indexp],
                  self.p_chunksizeunp[indexp], self.p_nparall,
                  self.lper_runlistrigger[indexp])
            myprocess = ProcesserInclusive(self.case, self.datap, self.mcordata,
                                   self.p_maxfiles[indexp], self.dlper_root[indexp],
                                   self.dlper_pkl[indexp], self.p_period[indexp],
                                   self.p_chunksizeunp[indexp], self.p_nparall,
                                   self.typean, self.lper_runlistrigger[indexp],
                                   self.d_results[indexp])
            self.process_listsample.append(myprocess)
        self.n_filemass = datap["files_names"]["histofilename"]
        self.filemass_mergedall = os.path.join(self.d_resulsallp, self.n_filemass)
        self.p_useperiod = datap["analysis"][self.typean]["useperiod"]
        self.lper_filemass = []
        for i, direc in enumerate(self.d_results):
            if self.p_useperiod[i] == 1:
                self.lper_filemass.append(os.path.join(direc, self.n_filemass))

    def multi_unpack_allperiods(self):
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].process_unpack_par()

    def multi_histomass(self):
        for indexp in range(self.prodnumber):
            if self.p_useperiod[indexp] == 1:
                self.process_listsample[indexp].process_histomass()
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/mass/{get_timestamp_string()}/"
        mergerootfiles(self.lper_filemass, self.filemass_mergedall, tmp_merged)
