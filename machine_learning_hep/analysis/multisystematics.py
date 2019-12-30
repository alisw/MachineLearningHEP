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
from machine_learning_hep.utilities import mergerootfiles, get_timestamp_string
from machine_learning_hep.logger import get_logger
class MultiSystematics: # pylint: disable=too-many-instance-attributes, too-many-statements
    species = "multisystematics"
    def __init__(self, syst_class, datap, case, typean, doperiodbyperiod):
        self.logger = get_logger()
        self.datap = datap
        self.typean = typean
        self.case = case
        self.d_resultsallpmc = datap["analysis"][self.typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][self.typean]["data"]["resultsallp"]
        self.d_resultsmc = datap["analysis"][self.typean]["mc"]["results"]
        self.d_resultsdata = datap["analysis"][self.typean]["data"]["results"]
        self.d_valevtdata = datap["validation"]["data"]["dir"]
        self.d_valevtmc = datap["validation"]["mc"]["dir"]
        self.d_valevtallpdata = datap["validation"]["data"]["dirmerged"]
        self.d_valevtallpmc = datap["validation"]["mc"]["dirmerged"]
        self.n_evtvalroot = datap["files_names"]["namefile_evtvalroot"]
        self.prodnumber = len(self.d_resultsmc)
        self.syst_listsample = []
        self.p_useperiod = datap["analysis"][self.typean]["useperiod"]
        self.doperiodbyperiod = doperiodbyperiod
        for indexp in range(self.prodnumber):
            mysyst = syst_class(self.datap, case, typean,
                                        self.d_resultsdata[indexp],
                                        self.d_resultsmc[indexp],
                                        self.d_valevtdata[indexp],
                                        self.d_valevtmc[indexp])
            self.syst_listsample.append(mysyst)

        self.mysysttot = syst_class(self.datap, case, typean,
                                            self.d_resultsallpdata,
                                            self.d_resultsallpmc,
                                            self.d_valevtallpdata,
                                            self.d_valevtallpmc)

    def multi_cutvariation(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.syst_listsample[indexp].cutvariation()
        self.mysysttot.cutvariation()
