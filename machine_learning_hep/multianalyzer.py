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
from machine_learning_hep.analyzer import Analyzer
from machine_learning_hep.utilities import mergerootfiles
class MultiAnalyzer: # pylint: disable=too-many-instance-attributes, too-many-statements
    species = "multianalyzer"
    def __init__(self, datap, case, typean, doperiodbyperiod):
        self.datap = datap
        self.typean = typean
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
        self.process_listsample = []
        self.p_useperiod = datap["analysis"][self.typean]["useperiod"]
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

        self.lper_normfilesorig = []
        self.lper_normfiles = []
        self.dlper_valevtroot = datap["validation"]["data"]["dir"]
        for i, _ in enumerate(self.d_resultsdata):
            self.lper_normfilesorig.append(os.path.join(self.dlper_valevtroot[i], \
                                                   "correctionsweights.root"))
            self.lper_normfiles.append(os.path.join(self.d_resultsdata[i], \
                                                   "correctionsweights.root"))
        self.f_normmerged = os.path.join(self.d_resultsallpdata, \
                                                    "correctionsweights.root")

    def multi_fitter(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
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
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].efficiency()
        self.myanalyzertot.efficiency()

    def multi_feeddown(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].feeddown()
        self.myanalyzertot.feeddown()

    def multi_unfolding(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].unfolding()
        self.myanalyzertot.unfolding()

    def multi_unfolding_closure(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].unfolding_closure()
        self.myanalyzertot.unfolding_closure()

    def multi_side_band_sub(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].side_band_sub()
        self.myanalyzertot.side_band_sub()

    def multi_plotter(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].plotter()
        self.myanalyzertot.plotter()

    def multi_plotternormyields(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].plotternormyields()
        self.myanalyzertot.plotternormyields()

    def multi_makenormyields(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    self.process_listsample[indexp].makenormyields()
        self.myanalyzertot.makenormyields()

    def multi_preparenorm(self):
        listempty = []
        for indexp in range(self.prodnumber):
            mergerootfiles([self.lper_normfilesorig[indexp]], self.lper_normfiles[indexp])
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                if self.p_useperiod[indexp] == 1:
                    listempty.append(self.lper_normfiles[indexp])
        mergerootfiles(listempty, self.f_normmerged)

    def multi_studyevents(self):
        if self.doperiodbyperiod is True:
            for indexp in range(self.prodnumber):
                self.process_listsample[indexp].studyevents()
        self.myanalyzertot.studyevents()
