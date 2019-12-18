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
from machine_learning_hep.analysis.systematics import Systematics
from machine_learning_hep.utilities import mergerootfiles, get_timestamp_string
from machine_learning_hep.logger import get_logger
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
class MultiSystematics:
    species = "multisystematic"
    def __init__(self, case, datap, typean, run_param):

        #General
        self.logger = get_logger()
        self.case = case
        self.datap = datap
        self.typean = typean
        self.run_param = run_param

        #Multi Data
        self.prodnumber = len(datap["multi"]["data"]["unmerged_tree_dir"])
        self.p_period = datap["multi"]["data"]["period"]
        self.lper_runlistrigger = datap["validation"]["runlisttrigger"]
        self.dlper_valevtroot = datap["validation"]["data"]["dir"]
        self.dlper_valevtroottot = datap["validation"]["data"]["dirmerged"]

        #MLapplication
        self.dlper_reco_modappmerged_mc = datap["mlapplication"]["mc"]["pkl_skimmed_decmerged"]
        self.dlper_reco_modappmerged_data = datap["mlapplication"]["data"]["pkl_skimmed_decmerged"]

        #Analysis
        self.d_results_cv = []
        self.d_results = datap["analysis"][self.typean]["data"]["results"]
        self.d_resultsallp = datap["analysis"][self.typean]["data"]["resultsallp"]
        for i, direc in enumerate(self.d_results):
            self.d_results_cv.append(self.d_results[i] + "/cutvar")
            if not os.path.exists(self.d_results_cv[i]):
                print("creating folder ", self.d_results_cv[i])
                os.makedirs(self.d_results_cv[i])
        self.d_resultsallp_cv = self.d_resultsallp + "/cutvar"
        if not os.path.exists(self.d_resultsallp_cv):
            print("creating folder ", self.d_resultsallp_cv)
            os.makedirs(self.d_resultsallp_cv)
        self.p_useperiodforlimits = datap["systematics"]["probvariation"]["useperiod"]
        self.p_useperiod = datap["analysis"][self.typean]["useperiod"]

        #File names
        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_filemass_cutvar = self.n_filemass.replace(".root", "_cutvar.root")
        self.n_fileeff_cutvar = self.n_fileeff.replace(".root", "_cutvar.root")
        self.n_fileeff_mcptshape = self.n_fileeff.replace(".root", "_ptshape.root")
        self.filemass_cutvar_mergedall = os.path.join(self.d_resultsallp_cv, self.n_filemass_cutvar)
        self.fileeff_cutvar_mergedall = os.path.join(self.d_resultsallp_cv, self.n_fileeff_cutvar)
        self.fileeff_ptshape_mergedall = os.path.join(self.d_resultsallp, self.n_fileeff_mcptshape)

        self.lper_filemass_cutvar = []
        self.lper_fileeff_cutvar = []
        for i, direc in enumerate(self.d_results_cv):
            if self.p_useperiod[i] == 1:
                self.lper_filemass_cutvar.append(os.path.join(direc, self.n_filemass_cutvar))
                self.lper_fileeff_cutvar.append(os.path.join(direc, self.n_fileeff_cutvar))
        self.lper_fileeff_mcptshape = []
        for i, direc in enumerate(self.d_results):
            if self.p_useperiod[i] == 1:
                self.lper_fileeff_mcptshape.append(os.path.join(direc, self.n_fileeff_mcptshape))

        self.process_listsample = []
        for indexp in range(self.prodnumber):
            myprocess = Systematics(self.case, self.datap, self.typean, self.run_param,
                                    self.dlper_reco_modappmerged_mc[indexp],
                                    self.dlper_reco_modappmerged_data[indexp],
                                    self.d_results[indexp], self.dlper_valevtroot[indexp],
                                    self.p_period[indexp],
                                    self.lper_runlistrigger[self.p_period[indexp]])
            self.process_listsample.append(myprocess)
        self.myprocesstot = Systematics(self.case, self.datap, self.typean, self.run_param,
                                        "", "", self.d_resultsallp, self.dlper_valevtroottot,
                                        None, None)

    def multi_cutvariation(self, domass, doeff, dofit, docross):
        """
        Goes through the full cut-variation systematic chain, based on what
        is set in databases.
        """
        self.logger.info("Processing cut variation systematic")
        if domass is True or doeff is True or dofit is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiodforlimits[indexp] == 1:
                    self.logger.info("Defining systematic cut variations for period: %s", \
                                     self.p_period[indexp])
                    min_cv_cut, max_cv_cut = \
                          self.process_listsample[indexp].define_cutvariation_limits()

        if domass is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].cutvariation_masshistos(min_cv_cut, max_cv_cut)
            tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/cutvar_mass/" \
                         f"{get_timestamp_string()}/"
            mergerootfiles(self.lper_filemass_cutvar, self.filemass_cutvar_mergedall, tmp_merged)

        if doeff is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].cutvariation_efficiencies(min_cv_cut, \
                                                                              max_cv_cut)
            tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/cutvar_eff/" \
                         f"{get_timestamp_string()}/"
            mergerootfiles(self.lper_fileeff_cutvar, self.fileeff_cutvar_mergedall, tmp_merged)

        if dofit is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].cutvariation_fitter(min_cv_cut, max_cv_cut)
                    self.process_listsample[indexp].cutvariation_efficiency()
            self.myprocesstot.cutvariation_fitter(min_cv_cut, max_cv_cut)
            self.myprocesstot.cutvariation_efficiency()

        if docross is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].cutvariation_makenormyields()
            self.myprocesstot.cutvariation_makenormyields()

            histname = ["histoSigmaCorr", "hDirectEffpt", "hFeedDownEffpt", "hRECpt"]
            for name in histname:
                for indexp in range(self.prodnumber):
                    if self.p_useperiod[indexp] == 1:
                        if domass is True or doeff is True or dofit is True:
                            self.process_listsample[indexp].cutvariation_makeplots(name, \
                                                                                   min_cv_cut, \
                                                                                   max_cv_cut)
                        else:
                            self.process_listsample[indexp].cutvariation_makeplots(name, None, None)
                if domass is True or doeff is True or dofit is True:
                    self.myprocesstot.cutvariation_makeplots(name, min_cv_cut, max_cv_cut)
                else:
                    self.myprocesstot.cutvariation_makeplots(name, None, None)

    def multimcptshape(self):
        """
        Goes through the full MC pT-shape systematic chain, based on what
        is set in databases.
        """
        self.logger.info("Processing MC pT shape systematic")
        for indexp in range(self.prodnumber):
            self.process_listsample[indexp].mcptshape_get_generated()
            self.process_listsample[indexp].mcptshape_build_efficiencies()
            self.process_listsample[indexp].mcptshape_efficiency()
            self.process_listsample[indexp].mcptshape_makeplots()
        tmp_merged = f"/data/tmp/hadd/{self.case}_{self.typean}/mcptshape_eff/" \
                      f"{get_timestamp_string()}/"
        mergerootfiles(self.lper_fileeff_mcptshape, self.fileeff_ptshape_mergedall, tmp_merged)
        self.myprocesstot.mcptshape_efficiency()
        self.myprocesstot.mcptshape_makeplots()
