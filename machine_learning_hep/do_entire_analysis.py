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
import yaml
from machine_learning_hep.doskimming import conversion, merging
from machine_learning_hep.doclassification_regression import doclassification_regression
from machine_learning_hep.doanalysis import doanalysis

def do_entire_analysis(): # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    case = "LctopK0sPbPbCen010"

    with open("default_complete.yaml", 'r') as run_config:
        data_config = yaml.load(run_config)

    with open("data/database_ml_parameters.yml", 'r') as param_config:
        data_param = yaml.load(param_config)

    binminarray = data_config["ml_study"]["binmin"]
    binmaxarray = data_config["ml_study"]["binmax"]
    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    domergingmc = data_config["merging"]["mc"]["activate"]
    domergingdata = data_config["merging"]["data"]["activate"]
    doml = data_config["ml_study"]["activate"]
    doanalyml = data_config["analysis"]["ml"]["activate"]
    doanalystd = data_config["analysis"]["std"]["activate"]
    #binminarrayan = data_config["analysis"]["binmin"]
    #binmaxarrayan = data_config["analysis"]["binmax"]
    #models = data_config["analysis"]["models"]

    if doconversionmc is True:
        pkl_mc = data_param[case]["output_folders"]["pkl_out"]["mc"]
        if os.path.exists(pkl_mc):
            print("output mc pkl exists")
            print("rm -rf ", pkl_mc)
        else:
            print("creating dir mc pkl")
            os.makedirs(pkl_mc)
            conversion(data_config, data_param, "mc")

    if doconversiondata is True:
        pkl_data = data_param[case]["output_folders"]["pkl_out"]["data"]
        if os.path.exists(pkl_data):
            print("output data pkl exists")
            print("rm -rf ", pkl_data)
        else:
            print("creating dir data pkl")
            os.makedirs(pkl_data)
            conversion(data_config, data_param, "data")

    if domergingdata is True:
        pkl_merged_data = data_param[case]["output_folders"]["pkl_merged"]["data"]
        if os.path.exists(pkl_merged_data):
            print("output data merged pkl exists")
            print("rm -rf ", pkl_merged_data)
        else:
            print("creating dir data merged pkl")
            os.makedirs(pkl_merged_data)
            merging(data_config, data_param, "data")

    if domergingmc is True:
        pkl_merged_mc = data_param[case]["output_folders"]["pkl_merged"]["mc"]
        if os.path.exists(pkl_merged_mc):
            print("output mc merged pkl exists")
            print("rm -rf ", pkl_merged_mc)
        else:
            print("creating dir merged mc pkl")
            os.makedirs(pkl_merged_mc)
            merging(data_config, data_param, "mc")

    if doml is True:
        mlout = data_param[case]["output_folders"]["mlout"]
        mlplot = data_param[case]["output_folders"]["mlplot"]
        if os.path.exists(mlout) or os.path.exists(mlplot):
            print("ml folders exist")
            print("rm -rf ", mlout)
            print("rm -rf ", mlplot)
        else:
            print("creating ml folder dir")
            os.makedirs(mlout)
            os.makedirs(mlplot)
            for binmin, binmax in zip(binminarray, binmaxarray):
                print(binmin, binmax)
                doclassification_regression(data_config["ml_study"],
                                            data_param, case, binmin, binmax)
    if doanalyml is True or doanalystd is True:
        pltanaldir = data_param[case]["output_folders"]["plotsanalysis"]
        histoanaldir = data_param[case]["output_folders"]["histoanalysis"]

        if os.path.exists(pltanaldir) or os.path.exists(histoanaldir):
            print("ml folders exist")
            print("rm -rf ", pltanaldir)
            print("rm -rf ", histoanaldir)
        else:
            print("creating analysis dir")
            os.makedirs(pltanaldir)
            os.makedirs(histoanaldir)
            if doanalystd is True:
                doanalysis(data_config, data_param, case, 0)

do_entire_analysis()

#def doskimming(data_config, data_param):
