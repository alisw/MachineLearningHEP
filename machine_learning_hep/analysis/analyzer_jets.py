#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
##                                                                         ##
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

import os
import munch # pylint: disable=import-error, no-name-in-module
from ROOT import TFile, TCanvas # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.analysis.analyzer import Analyzer

class AnalyzerJets(Analyzer): # pylint: disable=too-many-instance-attributes
    species = "analyzer"

    def __init__(self, datap, case, typean, period):
        super().__init__(datap, case, typean, period)

        self.cfg = munch.munchify(datap)
        self.cfg.ana = munch.munchify(datap).analysis[typean]

        # output directories
        self.d_resultsallpmc = datap["analysis"][typean]["mc"]["results"][period] \
                if period is not None else datap["analysis"][typean]["mc"]["resultsallp"]
        self.d_resultsallpdata = datap["analysis"][typean]["data"]["results"][period] \
                if period is not None else datap["analysis"][typean]["data"]["resultsallp"]

        # input directories (processor output)
        self.d_resultsallpmc_proc = self.d_resultsallpmc
        self.d_resultsallpdata_proc = self.d_resultsallpdata

        # input files
        n_filemass_name = datap["files_names"]["histofilename"]
        self.n_filemass = os.path.join(self.d_resultsallpdata_proc, n_filemass_name)
        self.n_filemass_mc = os.path.join(self.d_resultsallpmc_proc, n_filemass_name)
        self.n_fileeff = datap["files_names"]["efffilename"]
        self.n_fileeff = os.path.join(self.d_resultsallpmc_proc, self.n_fileeff)
        self.n_fileresp = datap["files_names"]["respfilename"]
        self.n_fileresp = os.path.join(self.d_resultsallpmc_proc, self.n_fileresp)

    def qa(self): # pylint: disable=too-many-branches, too-many-locals, invalid-name
        self.logger.info("Running D0 jet qa")

        print(self.n_filemass)
        with TFile(self.n_filemass) as rfile:
            histonorm = rfile.Get("histonorm")
            if not histonorm:
                self.logger.critical('histonorm not found')
            p_nevents = histonorm.GetBinContent(1)
            self.logger.debug('Number of selected event: %d', p_nevents)

            for ipt in range(7):
                c = TCanvas("Candidate mass")
                h_invmass = rfile.Get(f'hmass_{ipt}')
                if not h_invmass:
                    self.logger.critical('hmass not found')
                h_invmass.Print()
                h_invmass.Draw()
                c.SaveAs(f'hmass_{ipt}.png')

                c = TCanvas("Candidate pt")
                h_candpt = rfile.Get(f'hcandpt_{ipt}')
                if not h_candpt:
                    self.logger.critical('hcandpt not found')
                h_candpt.Print()
                h_candpt.Draw()
                c.SaveAs(f'hcandpt_{ipt}.png')

                c = TCanvas("Jet pt")
                h_jetpt = rfile.Get(f'hjetpt_{ipt}')
                if not h_jetpt:
                    self.logger.critical('hjetpt not found')
                h_jetpt.Print()
                h_jetpt.Draw()
                c.SaveAs(f'hjetpt_{ipt}.png')
