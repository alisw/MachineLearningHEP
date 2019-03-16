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
from machine_learning_hep.general import get_database_ml_parameters # pylint: disable=import-error
from machine_learning_hep.general import get_database_ml_analysis
from machine_learning_hep.listfiles import list_files_dir_lev1
from machine_learning_hep.skimming import create_inv_mass
from machine_learning_hep.skimming import plothisto
from machine_learning_hep.fit import fitmass, plot_graph_yield
from machine_learning_hep.io import checkdir

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doanalysis():

    case = "LctopK0s"
    dataset = "LHC18r"
    #dataset = "LHC17j4d2"
    data = get_database_ml_parameters()
    data_analysis = get_database_ml_analysis()

    var_pt = data[case]["var_binning"]

    fileinputdir = data_analysis[case][dataset]["inputdirdata"]
    print(fileinputdir)
    namefilereco = data_analysis[case]["namefile_in_pkl"]
    fileoutputdir = data_analysis[case]["outputdir"]
    outputdirhisto = data_analysis[case]["outputdirhisto"]
    probcut = data_analysis[case]["probcut"]
    models = data_analysis[case][dataset]["models"]
    binmin = data_analysis[case]["binmin"]
    binmax = data_analysis[case]["binmax"]

    print("starting my analysis")

    useml = 0
    yield_signal = []
    yield_signal_err = []

    doinvmassspectra = 1
    dofit = 0

    checkdir(fileoutputdir)
    checkdir(outputdirhisto)
    checkdir("plots")

    tstart = time.time()
    if doinvmassspectra == 1:
        index = 0
        for imin, imax in zip(binmin, binmax):
            namefilehist = ("histo%s_ptmin%s_%s_useml%d_0%d.root" % \
                            (case, imin, imax, useml, 1000*probcut[index]))
            namefileplot = ("plots/histotot%s_ptmin%s_%s_useml%d_0%d.pdf" % \
                            (case, imin, imax, useml, 1000*probcut[index]))
            listdf, listhisto = list_files_dir_lev1(fileinputdir, fileoutputdir,
                                                    namefilereco, namefilehist)
            histomass = create_inv_mass(listdf, listhisto, var_pt, imin, imax,
                                        useml, models[index], probcut[index], case)
            plothisto(histomass, namefileplot)
            index = index + 1

    timestop = time.time()
    print("TOTAL TIME ALL BINS,", timestop - tstart)
    if dofit == 1:
        for imin, imax in zip(binmin, binmax):
            signal, err_signal = fitmass(histomass)
            yield_signal.append(signal)
            yield_signal_err.append(err_signal)
            plot_graph_yield(yield_signal, yield_signal_err, binmin, binmax)

doanalysis()
