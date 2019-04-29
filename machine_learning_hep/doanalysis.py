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
import os.path

# pylint: disable=import-error
import time
import yaml
#from machine_learning_hep.general import get_database_ml_parameters # pylint: disable=import-error
#from machine_learning_hep.general import get_database_ml_analysis
from machine_learning_hep.listfiles import list_files_dir_lev2, list_files_lev2
from machine_learning_hep.doskimming import merge
#from ROOT import TFile # pylint: disable=import-error, no-name-in-module
from machine_learning_hep.skimming import selectcandidatesall
#from machine_learning_hep.fit import fitmass, plot_graph_yield

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doanalysis(data_config, data, case, useml, mcordata, index):

    var_pt = data[case]["variables"]["var_binning"]
    fileinputdir = data[case]["output_folders"]["pkl_skimmed"][mcordata][index]
    outputdirfin = data[case]["output_folders"]["pkl_final"][mcordata][index]
    namefilereco_ml = data[case]["files_names"]["namefile_reco_skim_ml"]
    namefilereco_std = data[case]["files_names"]["namefile_reco_skim_std"]
    namefile_reco_skim = data[case]["files_names"]["namefile_reco_skim"]
    namefile_gen_skim = data[case]["files_names"]["namefile_gen_skim"]
    namefilereco_ml_tot = data[case]["files_names"]["namefile_reco_skim_ml_tot"]
    namefilereco_std_tot = data[case]["files_names"]["namefile_reco_skim_std_tot"]
    namefile_gen_skim_tot = data[case]["files_names"]["namefile_gen_skim_tot"]

    maxfiles = data_config["analysis"][mcordata]["maxfiles"]
    nmaxchunks = data_config["analysis"][mcordata]["nmaxchunks"]
    doinvmassspectra = data_config["analysis"]["doinvmassspectra"]
    binmin = data_config["analysis"]["binmin"]
    binmax = data_config["analysis"]["binmax"]
    models = data_config["analysis"]["models"]
    probcut = data_config["analysis"]["probcut"]
    models = data_config["analysis"]["models"]
    modelname = data_config["analysis"]["modelname"]

    cuts_map = None
    if useml == 0:
        usecustomsel = data[case]["custom_std_sel"]["use"]
        if usecustomsel:
            cuts_config_filename = data[case]["custom_std_sel"]["cuts_config_file"]
            with open(cuts_config_filename, 'r') as cuts_config:
                cuts_map = yaml.load(cuts_config)
            #NB: in case of custom linear selections it overrides pT bins of default_complete
            binmin = cuts_map["var_binning"]["min"]
            binmax = cuts_map["var_binning"]["max"]

    tstart = time.time()
    if doinvmassspectra == 1:
        index = 0
        for imin, imax, ibin in zip(binmin, binmax, enumerate(binmin)):
            namefilereco_ml_in = namefilereco_ml.replace(".pkl", "%d_%d.pkl" % (imin, imax))
            namefilereco_std_in = namefilereco_std.replace(".pkl", "%d_%d.pkl" % (imin, imax))
            listdf, listdfout_ml = list_files_dir_lev2(fileinputdir, outputdirfin,
                                                       namefile_reco_skim, namefilereco_ml_in)
            listdf, listdfout_std = list_files_dir_lev2(fileinputdir, outputdirfin,
                                                        namefile_reco_skim, namefilereco_std_in)
            if maxfiles is not -1:
                listdf = listdf[:maxfiles]
                listdfout_ml = listdfout_ml[:maxfiles]
                listdfout_std = listdfout_std[:maxfiles]
            chunksdf = [listdf[x:x+nmaxchunks] \
                        for x in range(0, len(listdf), nmaxchunks)]
            chunksdfout_ml = [listdfout_ml[x:x+nmaxchunks] \
                           for x in range(0, len(listdfout_ml), nmaxchunks)]
            chunksdfout_std = [listdfout_std[x:x+nmaxchunks] \
                           for x in range(0, len(listdfout_std), nmaxchunks)]

            for idf, _ in enumerate(chunksdf):
                print("chunk number=", idf)
                selectcandidatesall(data, chunksdf[idf], chunksdfout_ml[idf],
                                    chunksdfout_std[idf], var_pt, imin, imax,
                                    useml, modelname, models[index],
                                    probcut[index], case, cuts_map, ibin[0])
            if useml == 1:
                namefilereco_ml_tot = os.path.join(outputdirfin, namefilereco_ml_tot)
                namefilereco_ml_tot = \
                    namefilereco_ml_tot.replace(".pkl", "%d_%d.pkl" % (imin, imax))
                merge(listdfout_ml, namefilereco_ml_tot)
            if useml == 0:
                namefilereco_std_tot = os.path.join(outputdirfin, namefilereco_std_tot)
                namefilereco_std_tot = \
                    namefilereco_std_tot.replace(".pkl", "%d_%d.pkl" % (imin, imax))
                merge(listdfout_std, namefilereco_std_tot)
            index = index + 1

        if mcordata == "mc":
            namefile_gen_skim_tot = os.path.join(outputdirfin, namefile_gen_skim_tot)
            listgen, _ = list_files_lev2(fileinputdir, "", namefile_gen_skim, "")
            merge(listgen, namefile_gen_skim_tot)

    timestop = time.time()
    print("total time of filling histo=", tstart - timestop)
