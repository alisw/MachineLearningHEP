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
from machine_learning_hep.analyzer import Analyzer
class MultiAnalyzer: # pylint: disable=too-many-instance-attributes, too-many-statements
    species = "multianalyzer"
    def __init__(self, datap, case, typean, doperiodbyperiod):
        self.datap = datap
        self.typean = typean
        self.d_resultsallpmc = datap["analysis"][self.typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][self.typean]["data"]["resultsallp"]
        self.d_valevtallpdata = datap["validation"]["data"]["dirmerged"]
        self.d_valevtallpmc = datap["validation"]["mc"]["dirmerged"]
        self.d_resultsmc = datap["analysis"][self.typean]["mc"]["results"]
        self.d_resultsdata = datap["analysis"][self.typean]["data"]["results"]
        self.d_valevtdata = datap["validation"]["data"]["dir"]
        self.d_valevtmc = datap["validation"]["mc"]["dir"]
        self.prodnumber = len(self.d_resultsmc)
        self.process_listsample = []
        self.doperiodbyperiod = doperiodbyperiod
        for indexp in range(self.prodnumber):
            myanalyzer = Analyzer(self.datap, case, typean,
                                  self.d_resultsdata[indexp], self.d_resultsmc[indexp],
                                  self.d_valevtdata[indexp],
                                  self.d_valevtmc[indexp])
            self.process_listsample.append(myanalyzer)

        self.myanalyzertot = Analyzer(self.datap, case, typean,
                                      self.d_resultsallpdata, self.d_resultsallpmc,
                                      self.d_valevtallpdata, self.d_valevtallpmc)

    def multi_fitter(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].fitter()
        self.myanalyzertot.fitter()

    def multi_yield_syst(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].yield_syst()
        self.myanalyzertot.yield_syst()

    def multi_efficiency(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].efficiency()
        self.myanalyzertot.efficiency()

    def multi_feeddown(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].feeddown()
        self.myanalyzertot.feeddown()

    def multi_side_band_sub(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].side_band_sub()
        self.myanalyzertot.side_band_sub()

    def multi_plotter(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].plotter()
        self.myanalyzertot.plotter()

    def multi_studyevents(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].studyevents()
        self.myanalyzertot.studyevents()
