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
main macro for charm analysis with python
"""
#import argparse
#import sys
#import os.path

# pylint: disable=import-error
import time
#from machine_learning_hep.general import get_database_ml_parameters # pylint: disable=import-error
#from machine_learning_hep.general import get_database_ml_analysis
from machine_learning_hep.listfiles import list_files_dir_lev2
from machine_learning_hep.skimming import create_inv_mass
from machine_learning_hep.doskimming import merge
#from ROOT import TFile # pylint: disable=import-error, no-name-in-module
#from machine_learning_hep.skimming import plothisto
#from machine_learning_hep.fit import fitmass, plot_graph_yield

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doanalysis(data_config, data, case, useml):

    var_pt = data[case]["variables"]["var_binning"]
    fileinputdir = data[case]["output_folders"]["pkl_skimmed"]["data"]
    namefilereco = data[case]["files_names"]["namefile_reco_skim"]
    outputdirhisto = data[case]["output_folders"]["histoanalysis"]

    maxfiles = data_config["analysis"]["maxfiles"]
    nmaxchunks = data_config["analysis"]["nmaxchunks"]
    doinvmassspectra = data_config["analysis"]["doinvmassspectra"]
    binmin = data_config["analysis"]["binmin"]
    binmax = data_config["analysis"]["binmax"]
    models = data_config["analysis"]["models"]
    probcut = data_config["analysis"]["probcut"]
    models = data_config["analysis"]["models"]
    modelname = data_config["analysis"]["modelname"]
    skimmeddf = "skimmedLc.pkl"

    #yield_signal = []
    #yield_signal_err = []

    tstart = time.time()
    if doinvmassspectra == 1:
        index = 0
        for imin, imax in zip(binmin, binmax):
            namefilehist = ("histo%s_ptmin%s_%s_useml%d_0%d.root" % \
                            (case, imin, imax, useml, 1000*probcut[index]))
            listdf, listhisto = list_files_dir_lev2(fileinputdir, outputdirhisto,
                                                    namefilereco, namefilehist)
            listdf, listdfout = list_files_dir_lev2(fileinputdir, outputdirhisto,
                                                    namefilereco, skimmeddf)
            print(listdf)
            if maxfiles is not -1:
                listdf = listdf[:maxfiles]
                listhisto = listhisto[:maxfiles]

            chunksdf = [listdf[x:x+nmaxchunks] for x in range(0, len(listdf), nmaxchunks)]
            chunkshisto = [listhisto[x:x+nmaxchunks] \
                           for x in range(0, len(listhisto), nmaxchunks)]

            chunksoutdf = [listdfout[x:x+nmaxchunks] \
                           for x in range(0, len(listdfout), nmaxchunks)]

            for idf, _ in enumerate(chunksdf):
                print("chunk number=", idf)
                _ = create_inv_mass(data, chunksdf[idf], chunkshisto[idf],
                                    chunksoutdf[idf], var_pt, imin, imax,
                                    useml, modelname, models[index],
                                    probcut[index], case)
            index = index + 1
        merge(listdfout, skimmeddf)
    timestop = time.time()
    print("total time of filling histo=", tstart - timestop)
