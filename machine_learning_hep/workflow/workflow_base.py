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

from os.path import join
# HF specific imports
from machine_learning_hep.logger import get_logger
# pylint: disable=import-error, no-name-in-module
from ROOT import gStyle

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


    @staticmethod
    def loadstyle():
        gStyle.SetOptStat(0)
        gStyle.SetOptStat(0000)
        gStyle.SetPalette(1)
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
            return False
        getattr(self, step)()
        return True


    # pylint: disable=no-self-use
    def get_after_burner(self):
        """
        Return an after-burner object to be run after per-period workflow steps, OPTIONAL
        Can be overwritten by deriving class
        """
        return None
