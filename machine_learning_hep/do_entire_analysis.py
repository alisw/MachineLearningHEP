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
from machine_learning_hep.doskimming import conversion, merging, merging_period, skim
from machine_learning_hep.doclassification_regression import doclassification_regression
from machine_learning_hep.doanalysis import doanalysis
from machine_learning_hep.extractmasshisto import extractmasshisto
#from machine_learning_hep.efficiency import extract_eff_histo

def do_entire_analysis(): # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    with open("default_complete.yaml", 'r') as run_config:
        data_config = yaml.load(run_config)

    with open("data/database_ml_parameters.yml", 'r') as param_config:
        data_param = yaml.load(param_config)

    with open("data/config_model_parameters.yml", 'r') as mod_config:
        data_model = yaml.load(mod_config)

    with open("data/database_run_list.yml", 'r') as runlist_config:
        run_param = yaml.load(runlist_config)

    case = data_config["case"]
    binminarray = data_config["ml_study"]["binmin"]
    binmaxarray = data_config["ml_study"]["binmax"]
    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    domergingmc = data_config["merging"]["mc"]["activate"]
    domergingdata = data_config["merging"]["data"]["activate"]
    doskimmingmc = data_config["skimming"]["mc"]["activate"]
    doskimmingdata = data_config["skimming"]["data"]["activate"]
    domergingperiodsmc = data_config["mergingperiods"]["mc"]["activate"]
    domergingperiodsdata = data_config["mergingperiods"]["data"]["activate"]
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
    #doeffhistml = data_config["analysis"]["mc"]["ml"]["doeffhist"]
    #doeffhiststd = data_config["analysis"]["mc"]["std"]["doeffhist"]

    pkl_mc_list = data_param[case]["output_folders"]["pkl_out"]["mc"]
    pkl_data_list = data_param[case]["output_folders"]["pkl_out"]["data"]
    pkl_skimmed_mc_list = data_param[case]["output_folders"]["pkl_skimmed"]["mc"]
    pkl_skimmed_data_list = data_param[case]["output_folders"]["pkl_skimmed"]["data"]
    pkl_merged_data_list = data_param[case]["output_folders"]["pkl_merged"]["data"]
    pkl_merged_mc_list = data_param[case]["output_folders"]["pkl_merged"]["mc"]
    pkl_merged_all_data = data_param[case]["output_folders"]["pkl_merged_all"]["data"]
    pkl_merged_all_mc = data_param[case]["output_folders"]["pkl_merged_all"]["mc"]
    pkl_final_mc_list = data_param[case]["output_folders"]["pkl_final"]["mc"]
    pkl_final_data_list = data_param[case]["output_folders"]["pkl_final"]["data"]
    pkl_analysis_data_list = data_param[case]["output_folders"]["plotsanalysis"]["data"]
    pkl_analysis_mc_list = data_param[case]["output_folders"]["plotsanalysis"]["mc"]

    #nperiodsmc = data_param[case]["nperiodsmc"]
    #nperiodsdata = data_param[case]["nperiodsdata"]

    if doconversionmc is True:
        for index, pkl_mc in enumerate(pkl_mc_list):
            if os.path.exists(pkl_mc):
                print("output mc pkl exists")
                print("rm -rf ", pkl_mc)
            else:
                print("creating dir mc pkl period=", index)
                os.makedirs(pkl_mc)
                conversion(data_config, data_param, "mc", index)

    if doconversiondata is True:
        for index, pkl_data in enumerate(pkl_data_list):
            if os.path.exists(pkl_data):
                print("output data pkl exists")
                print("rm -rf ", pkl_data)
            else:
                print("creating dir data pkl period=", index)
                os.makedirs(pkl_data)
                conversion(data_config, data_param, "data", index)

    if doskimmingmc is True:
        print(pkl_skimmed_mc_list)
        for index, pkl_skimmed_mc in enumerate(pkl_skimmed_mc_list):
            if os.path.exists(pkl_skimmed_mc):
                print("output mc skimmed pkl exists")
                print("rm -rf ", pkl_skimmed_mc)
            else:
                print("creating dir mc skimmed pkl period=", index)
                os.makedirs(pkl_skimmed_mc)
                skim(data_config, data_param, "mc", run_param, index)

    if doskimmingdata is True:
        for index, pkl_skimmed_data in enumerate(pkl_skimmed_data_list):
            if os.path.exists(pkl_skimmed_data):
                print("output data skimmed pkl exists")
                print("rm -rf ", pkl_skimmed_data)
            else:
                print("creating dir data skimmed pkl period=", pkl_skimmed_data)
                os.makedirs(pkl_skimmed_data)
                skim(data_config, data_param, "data", run_param, index)

    if domergingdata is True:
        for index, pkl_merged_data in enumerate(pkl_merged_data_list):
            if os.path.exists(pkl_merged_data):
                print("output data merged pkl exists")
                print("rm -rf ", pkl_merged_data)
            else:
                print("creating dir data merged pkl period=", index)
                os.makedirs(pkl_merged_data)
                merging(data_config, data_param, "data", index)

    if domergingmc is True:
        for index, pkl_merged_mc in enumerate(pkl_merged_mc_list):
            if os.path.exists(pkl_merged_mc):
                print("output mc merged pkl exists")
                print("rm -rf ", pkl_merged_mc)
            else:
                print("creating dir mc merged pkl period=", index)
                os.makedirs(pkl_merged_mc)
                merging(data_config, data_param, "mc", index)

    if domergingperiodsdata is True:
        if os.path.exists(pkl_merged_all_data):
            print("output data merged all periods pkl exists")
            print("rm -rf ", pkl_merged_all_data)
        else:
            print("creating dir data merged all periods pkl")
            os.makedirs(pkl_merged_all_data)
            merging_period(data_config, data_param, "data")

    if domergingperiodsmc is True:
        if os.path.exists(pkl_merged_all_mc):
            print("output mc merged all periods pkl exists")
            print("rm -rf ", pkl_merged_all_mc)
        else:
            print("creating dir mc merged all periods pkl")
            os.makedirs(pkl_merged_all_mc)
            merging_period(data_config, data_param, "mc")

    if doml is True:
        print("DOING ML optimisation")
        for binmin, binmax in zip(binminarray, binmaxarray):
            print(binmin, binmax)
            doclassification_regression(data_config["ml_study"],
                                        data_param, data_model[mltype], case, binmin, binmax)

    for index, pkl_final_data in enumerate(pkl_final_data_list):
        if os.path.exists(pkl_final_data) is not True:
            os.makedirs(pkl_final_data)
    for index, pkl_final_mc in enumerate(pkl_final_mc_list):
        if os.path.exists(pkl_final_mc) is not True:
            os.makedirs(pkl_final_mc)
    for index, pkl_analysis_data in enumerate(pkl_analysis_data_list):
        if os.path.exists(pkl_analysis_data) is not True:
            os.makedirs(pkl_analysis_data)
    for index, pkl_analysis_mc in enumerate(pkl_analysis_mc_list):
        if os.path.exists(pkl_analysis_mc) is not True:
            os.makedirs(pkl_analysis_mc)

    if doapplymldata is True:
        for index, pkl_final_data in enumerate(pkl_final_data_list):
            print("applying ml to data")
            print("Writing output to", pkl_final_data)
            useml = 1
            doanalysis(data_config, data_param, case, useml, "data", index)

    if doapplymlmc is True:
        for index, pkl_final_mc in enumerate(pkl_final_mc_list):
            print("applying ml to mc")
            print("Writing output to", pkl_final_mc)
            useml = 1
            doanalysis(data_config, data_param, case, useml, "mc", index)

    if doapplystddata is True:
        for index, pkl_final_data in enumerate(pkl_final_data_list):
            print("applying std to data")
            print("Writing output to", pkl_final_data)
            useml = 0
            doanalysis(data_config, data_param, case, useml, "data", index)

    if doapplystdmc is True:
        for index, pkl_final_mc in enumerate(pkl_final_mc_list):
            print("applying std to mc")
            print("Writing output to", pkl_final_mc)
            useml = 0
            doanalysis(data_config, data_param, case, useml, "mc", index)

    if domassmldata is True:
        for index, pkl_analysis_data in enumerate(pkl_analysis_data_list):
            print("extracting mass histo ml data")
            print("Writing output to", pkl_analysis_data)
            useml = 1
            extractmasshisto(data_config, data_param, case, useml, "data", index)

    if domassmlmc is True:
        for index, pkl_analysis_mc in enumerate(pkl_analysis_mc_list):
            print("extracting mass histo ml mc")
            print("Writing output to", pkl_analysis_mc)
            useml = 1
            extractmasshisto(data_config, data_param, case, useml, "mc", index)

    if domassstddata is True:
        for index, pkl_analysis_data in enumerate(pkl_analysis_data_list):
            print("extracting mass histo std data")
            print("Writing output to", pkl_analysis_data)
            useml = 0
            extractmasshisto(data_config, data_param, case, useml, "data", index)

    if domassstdmc is True:
        for index, pkl_analysis_mc in enumerate(pkl_analysis_mc_list):
            print("extracting mass histo std mc")
            print("Writing output to", pkl_analysis_mc)
            useml = 0
            extractmasshisto(data_config, data_param, case, useml, "mc", index)
#
#    if doeffhistml:
#        pkl_final_mc_list = data_param[case]["output_folders"]["pkl_final"]["mc"]
#        index = 0
#        for pkl_final_data in pkl_final_data_list:
#            print("extracting eff x acc histo ml")
#            extract_eff_histo(index, data_config, data_param, case, 'ml')
#
#    if doeffhiststd:
#        pkl_final_mc_list = data_param[case]["output_folders"]["pkl_final"]["mc"]
#        index = 0
#        for pkl_final_data in pkl_final_data_list:
#            print("extracting eff x acc histo std")
#            extract_eff_histo(index, data_config, data_param, case, 'std')

do_entire_analysis()
