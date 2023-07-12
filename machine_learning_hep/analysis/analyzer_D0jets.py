from machine_learning_hep.analysis.analyzer import Analyzer

import munch
from ROOT import TFile, TH1F
import os

class AnalyzerD0jets(Analyzer):
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

    def qa(self): # pylint: disable=too-many-branches, too-many-locals
        self.logger.info("Running D0 jet qa")

        with TFile(self.n_filemass) as rfile:
            histonorm = rfile.Get("histonorm")
            if not histonorm:
                self.logger.critical('histonorm not found')
            self.p_nevents = histonorm.GetBinContent(1)
            self.logger.debug("Number of selected event: %g" % self.p_nevents)
