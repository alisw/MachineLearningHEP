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

from machine_learning_hep.logger import get_logger

# pylint: disable=too-many-instance-attributes
class AnalyzerManager:
    """
    Manager class handling analysis and systematic objects
    """

    def __init__(self, ana_class, database, case, typean, doperiodbyperiod, *args):

        self.ana_class = ana_class
        self.database = database
        self.case = case
        self.typean = typean
        self.doperiodbyperiod = doperiodbyperiod

        # Additional arguments to be forwarded to the analyzers
        self.add_args = args

        self.logger = get_logger()

        self.analyzers = []
        self.after_burner = None

        self.is_initialized = False


    def get_analyzers(self, none_for_unused_period=True):
        self.initialize()
        if not none_for_unused_period:
            return self.analyzers

        useperiod = self.database["analysis"][self.typean]["useperiod"]
        analyzers = [None] * (len(useperiod) + 1)
        for a in self.analyzers:
            if a.period is not None:
                analyzers[a.period] = a
        analyzers[-1] = self.analyzers[-1]
        return analyzers


    def initialize(self):
        """
        Collect all analyzer objects required in a list and initialises the after_burner if present
        """

        if self.is_initialized:
            return

        self.logger.info("Initialize analyzer manager for analyzer %s", self.ana_class.__name__)

        useperiod = self.database["analysis"][self.typean]["useperiod"]

        for ip, period in enumerate(useperiod):
            if self.doperiodbyperiod and period:
                self.analyzers.append(self.ana_class(self.database, self.case, self.typean, ip,
                                                     *self.add_args))
        self.analyzers.append(self.ana_class(self.database, self.case, self.typean, None,
                                             *self.add_args))

        if self.doperiodbyperiod:
            # get after-burner, if any
            self.after_burner = self.analyzers[-1].get_after_burner()
            if self.after_burner:
                self.after_burner.analyzers = self.analyzers[:-1]
                self.after_burner.analyzer_merged = self.analyzers[-1]

        self.is_initialized = True


    def analyze(self, *ana_steps):
        """
        Gives a list of analyzers and analysis steps do each step for each analyzer
        Args:
            ana_steps: list of analysis steps as strings
        """

        if not ana_steps:
            self.logger.info("No analysis steps to be done for Analyzer class %s. Return...",
                             self.ana_class.__name__)
            return

        self.initialize()

        self.logger.info("Run all registered analyzers of type %s for following analysis steps",
                         self.ana_class.__name__)
        for step in ana_steps:
            print(f"  -> {step}")

        # Collect potentially failed systematic steps
        failed_steps = []
        failed_steps_after_burner = []
        for step in ana_steps:
            if self.doperiodbyperiod:
                for analyzer in self.analyzers[:-1]:
                    if not analyzer.step(step):
                        failed_steps.append((analyzer.__class__.__name__, step))
                        # If analysis step could not be found here,
                        # we don't need to go on trying this steps since all analyzers are of the
                        # same class
                        break

                # Run after-burner if one was provided by the analyzer object
                if self.after_burner and not self.after_burner.step(step):
                    failed_steps_after_burner.append((self.after_burner.__class__.__name__, step))

            # Do analysis step for period-merged analyzer
            self.analyzers[-1].step(step)

        if failed_steps:
            self.logger.error("Following analysis steps could not be found:")
            for fs in failed_steps:
                print(f"Analyzer class: {fs[0]}, anqalysis step: {fs[1]}")
