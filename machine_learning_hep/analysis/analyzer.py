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
from machine_learning_hep.workflow.workflow_base import WorkflowBase


class Analyzer(WorkflowBase):
    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)



class AnalyzerAfterBurner(WorkflowBase):
    def __init__(self, datap, case, typean):
        super().__init__(datap, case, typean, None)

        self.analyzers = None
