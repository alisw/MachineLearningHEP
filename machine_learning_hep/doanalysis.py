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
#from machine_learning_hep.skimming import plothisto
#from machine_learning_hep.fit import fitmass, plot_graph_yield

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doanalysis(data_config, data, case, useml):

    var_pt = data[case]["variables"]["var_binning"]
    fileinputdir = data[case]["output_folders"]["pkl_out"]["data"]
    namefilereco = data[case]["files_names"]["namefile_reco"]
    outputdirhisto = data[case]["output_folders"]["histoanalysis"]

    maxfiles = data_config["analysis"]["maxfiles"]
    nmaxchunks = data_config["analysis"]["nmaxchunks"]
    doinvmassspectra = data_config["analysis"]["doinvmassspectra"]
    binmin = data_config["analysis"]["binmin"]
    binmax = data_config["analysis"]["binmax"]
    models = data_config["analysis"]["models"]
    probcut = data_config["analysis"]["probcut"]


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

            if maxfiles is not -1:
                listdf = listdf[:maxfiles]
                listhisto = listhisto[:maxfiles]

            chunksdf = [listdf[x:x+nmaxchunks] for x in range(0, len(listdf), nmaxchunks)]
            chunkshisto = [listhisto[x:x+nmaxchunks] \
                           for x in range(0, len(listhisto), nmaxchunks)]

            for chunk, chunkhisto in zip(chunksdf, chunkshisto):
                print("new chunck")
                _ = create_inv_mass(data, chunk, chunkhisto, var_pt, imin, imax,
                                    useml, models[index], probcut[index], case)
            index = index + 1

    timestop = time.time()
    print("total time of filling histo=", tstart - timestop)
#    print("TOTAL TIME ALL BINS,", timestop - tstart)
#    if dofit == 1:
#        for imin, imax in zip(binmin, binmax):
#            signal, err_signal = fitmass(histomass)
#            yield_signal.append(signal)
#            yield_signal_err.append(err_signal)
#            plot_graph_yield(yield_signal, yield_signal_err, binmin, binmax))
