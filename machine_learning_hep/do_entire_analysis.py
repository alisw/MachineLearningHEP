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
from machine_learning_hep.doskimming import conversion, merging, skim
from machine_learning_hep.doclassification_regression import doclassification_regression
from machine_learning_hep.doanalysis import doanalysis
from machine_learning_hep.extractmasshisto import extractmasshisto
from machine_learning_hep.efficiency import extract_eff_histo

def do_entire_analysis(): # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    case = "Dspp5TeV"

    with open("default_complete.yaml", 'r') as run_config:
        data_config = yaml.load(run_config)

    with open("data/database_ml_parameters.yml", 'r') as param_config:
        data_param = yaml.load(param_config)

    with open("data/config_model_parameters.yml", 'r') as mod_config:
        data_model = yaml.load(mod_config)

    case = data_config["case"]
    binminarray = data_config["ml_study"]["binmin"]
    binmaxarray = data_config["ml_study"]["binmax"]
    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    domergingmc = data_config["merging"]["mc"]["activate"]
    domergingdata = data_config["merging"]["data"]["activate"]
    doskimmingmc = data_config["skimming"]["mc"]["activate"]
    doskimmingdata = data_config["skimming"]["data"]["activate"]
    doml = data_config["ml_study"]["activate"]
    mltype = data_config["ml_study"]["mltype"]
    doapplymldata = data_config["analysis"]["data"]["ml"]["doapply"]
    doapplystddata = data_config["analysis"]["data"]["std"]["doapply"]
    doapplymlmc = data_config["analysis"]["mc"]["ml"]["doapply"]
    doapplystdmc = data_config["analysis"]["mc"]["std"]["doapply"]
    domassmldata = data_config["analysis"]["data"]["ml"]["domass"]
    domassstddata = data_config["analysis"]["data"]["std"]["domass"]
    domassmlmc = data_config["analysis"]["mc"]["ml"]["domass"]
    domassstdmc = data_config["analysis"]["mc"]["std"]["domass"]
    doeffhistml = data_config["analysis"]["mc"]["ml"]["doeffhist"]
    doeffhiststd = data_config["analysis"]["mc"]["std"]["doeffhist"]

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

    if doskimmingdata is True:
        pkl_skimmed_data = data_param[case]["output_folders"]["pkl_skimmed"]["data"]
        if os.path.exists(pkl_skimmed_data):
            print("output data skimmed pkl exists")
            print("rm -rf ", pkl_skimmed_data)
        else:
            print("creating dir data skimmed pkl")
            os.makedirs(pkl_skimmed_data)
            skim(data_config, data_param, "data")

    if doskimmingmc is True:
        pkl_skimmed_mc = data_param[case]["output_folders"]["pkl_skimmed"]["mc"]
        if os.path.exists(pkl_skimmed_mc):
            print("output mc skimmed pkl exists")
            print("rm -rf ", pkl_skimmed_mc)
        else:
            print("creating dir mc skimmed pkl")
            os.makedirs(pkl_skimmed_mc)
            skim(data_config, data_param, "mc")

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
        print("DOING ML optimisation")
        for binmin, binmax in zip(binminarray, binmaxarray):
            print(binmin, binmax)
            doclassification_regression(data_config["ml_study"],
                                        data_param, data_model[mltype], case, binmin, binmax)
    if doapplymldata is True:
        print("applying ml to data")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["data"])
        useml = 1
        doanalysis(data_config, data_param, case, useml, "data")

    if doapplystddata is True:
        print("applying std to data")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["data"])
        useml = 0
        doanalysis(data_config, data_param, case, useml, "data")

    if doapplymlmc is True:
        print("applying ml to mc")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["mc"])
        useml = 1
        doanalysis(data_config, data_param, case, useml, "mc")

    if doapplystdmc is True:
        print("applying std to mc")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["mc"])
        useml = 0
        doanalysis(data_config, data_param, case, useml, "mc")

    if domassmldata is True:
        print("extracting mass histo ml data")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["data"])
        useml = 1
        extractmasshisto(data_config, data_param, case, useml, "data")

    if domassstddata is True:
        print("extracting mass histo std data")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["data"])
        useml = 0
        extractmasshisto(data_config, data_param, case, useml, "data")

    if domassmlmc is True:
        print("extracting mass histo std mc")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["mc"])
        useml = 1
        extractmasshisto(data_config, data_param, case, useml, "mc")

    if domassstdmc is True:
        print("extracting mass histo std mc")
        print("Writing output to", data_param[case]["output_folders"]["pkl_final"]["mc"])
        useml = 0
        extractmasshisto(data_config, data_param, case, useml, "mc")

    if doeffhistml:
        print("extracting eff x acc histo ml")
        extract_eff_histo(data_config, data_param, case, 'ml')

    if doeffhiststd:
        print("extracting eff x acc histo std")
        extract_eff_histo(data_config, data_param, case, 'std')

do_entire_analysis()
