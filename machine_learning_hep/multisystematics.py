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
from machine_learning_hep.systematics import Systematics
from machine_learning_hep.utilities import mergerootfiles
class MultiSystematics: # pylint: disable=too-many-instance-attributes, too-many-statements, too-few-public-methods
    species = "multisystematic"
    def __init__(self, case, datap, typean, run_param):

        #General
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
        self.d_results = datap["analysis"][self.typean]["data"]["results"]
        self.d_resultsallp = datap["analysis"][self.typean]["data"]["resultsallp"]

        #File names
        self.n_filemass = datap["files_names"]["histofilename"]
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_filemass_cutvar = self.n_filemass.replace(".root", "_cutvar.root")
        self.n_fileeff_cutvar = self.n_fileeff.replace(".root", "_cutvar.root")
        self.filemass_cutvar_mergedall = os.path.join(self.d_resultsallp, self.n_filemass_cutvar)
        self.fileeff_cutvar_mergedall = os.path.join(self.d_resultsallp, self.n_fileeff_cutvar)

        self.p_useperiodforlimits = datap["systematics"]["probvariation"]["useperiod"]
        self.p_useperiod = datap["analysis"][self.typean]["useperiod"]

        self.lper_filemass_cutvar = []
        self.lper_fileeff_cutvar = []
        for i, direc in enumerate(self.d_results):
            if self.p_useperiod[i] == 1:
                self.lper_filemass_cutvar.append(os.path.join(direc, self.n_filemass_cutvar))
                self.lper_fileeff_cutvar.append(os.path.join(direc, self.n_fileeff_cutvar))

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

    #pylint: disable=too-many-branches
    def multi_cutvariation(self, domass, doeff, dofit, docross):

        if domass is True or doeff is True or dofit is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiodforlimits[indexp] == 1:
                    print("Processing systematics period: ", indexp)
                    min_cv_cut, max_cv_cut = \
                          self.process_listsample[indexp].define_cutvariation_limits()

        if domass is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].cutvariation_masshistos(min_cv_cut, max_cv_cut)
            mergerootfiles(self.lper_filemass_cutvar, self.filemass_cutvar_mergedall)

        if doeff is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].cutvariation_efficiencies(min_cv_cut, \
                                                                              max_cv_cut)
            mergerootfiles(self.lper_fileeff_cutvar, self.fileeff_cutvar_mergedall)

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
                        self.process_listsample[indexp].cutvariation_makeplots(name)
                self.myprocesstot.cutvariation_makeplots(name)
