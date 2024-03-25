#############################################################################
##  © Copyright CERN 2024. All rights not expressly granted are reserved.  ##
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

from functools import reduce
from os.path import join
# pylint: disable=import-error, no-name-in-module
from ROOT import gStyle
# HF specific imports
from machine_learning_hep.logger import get_logger

# pylint: disable=too-few-public-methods
class WorkflowBase:
    """
    Base class for all workflows related classes including systematics
    """
    species = "workflow_base"
    def __init__(self, datap, case, typean, period=None):

        self.logger = get_logger()
        self.datap = datap
        self.case = case
        self.typean = typean
        self.period = period

    def cfg(self, param, default = None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
                      param.split("."), self.datap['analysis'][self.typean])

    @staticmethod
    def loadstyle():
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(1)
        gStyle.SetNumberContours(100)
        gStyle.SetCanvasColor(0)
        gStyle.SetFrameFillColor(0)


    @staticmethod
    def make_pre_suffix(args):
        """
        Construct a common file suffix from args
        """
        try:
            _ = iter(args)
        except TypeError:
            args = [args]
        else:
            if isinstance(args, str):
                args = [args]
        args = [str(a) for a in args]
        return "_".join(args)


    @staticmethod
    def make_file_path(directory, filename, extension, prefix=None, suffix=None):
        if prefix is not None:
            filename = WorkflowBase.make_pre_suffix(prefix) + "_" + filename
        if suffix is not None:
            filename = filename + "_" + WorkflowBase.make_pre_suffix(suffix)
        extension = extension.replace(".", "")
        return join(directory, filename + "." + extension)


    def step(self, step: str):
        """
        Given a workflow steps as string, find the corresponding method and call it.
        Args:
            step: workflow step as string
        Returns:
            True if the step was found and executed, False otherwise
        """
        if not hasattr(self, step):
            self.logger.error("Could not run workflow step %s for workflow %s", step,
                              self.__class__.__name__)
            return False
        self.logger.info("Run workflow step %s for workflow %s", step, self.__class__.__name__)
        getattr(self, step)()
        return True


    def get_after_burner(self):
        """
        Return an after-burner object to be run after per-period workflow steps, OPTIONAL
        Can be overwritten by deriving class
        """
        return None
