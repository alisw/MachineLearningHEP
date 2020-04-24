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

# HF specific imports
from os.path import exists, join
from os import makedirs

from machine_learning_hep.workflow.workflow_base import WorkflowBase
from machine_learning_hep.io import dump_yaml_from_dict


class Analyzer(WorkflowBase):
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        # The only thing here is to dump the database in the data analysis directory
        results_dir_data = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]
        # create otput directories in case they do not exist
        if not exists(results_dir_data):
            makedirs(results_dir_data)
        dump_yaml_from_dict({case: datap}, join(results_dir_data, f"database_{case}_{typean}.yml"))


class AnalyzerAfterBurner(WorkflowBase):
    def __init__(self, datap, case, typean):
        super().__init__(datap, case, typean, None)

        self.analyzers = None
        self.analyzer_merged = None
