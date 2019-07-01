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

#import os
import yaml
from multiprocesser import MultiProcesser  # pylint: disable=import-error
#from machine_learning_hep.doskimming import conversion, merging, merging_period, skim
#from machine_learning_hep.doclassification_regression import doclassification_regression
#from machine_learning_hep.doanalysis import doanalysis
#from machine_learning_hep.extractmasshisto import extractmasshisto
#from machine_learning_hep.efficiencyan import analysis_eff
from  machine_learning_hep.utilities import checkdirlist, checkdir
from optimiser import Optimiser

def do_entire_analysis(): # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    with open("default_complete.yaml", 'r') as run_config:
        data_config = yaml.load(run_config, Loader=yaml.FullLoader)
    case = data_config["case"]

    with open("data/database_ml_parameters_%s.yml" % case, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)

    with open("data/config_model_parameters.yml", 'r') as mod_config:
        data_model = yaml.load(mod_config, Loader=yaml.FullLoader)

    with open("data/database_run_list.yml", 'r') as runlist_config:
        run_param = yaml.load(runlist_config, Loader=yaml.FullLoader)

    with open("data/database_ml_gridsearch.yml", 'r') as grid_config:
        grid_param = yaml.load(grid_config, Loader=yaml.FullLoader)

    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    domergingmc = data_config["merging"]["mc"]["activate"]
    domergingdata = data_config["merging"]["data"]["activate"]
    doskimmingmc = data_config["skimming"]["mc"]["activate"]
    doskimmingdata = data_config["skimming"]["data"]["activate"]
    domergingperiodsmc = data_config["mergingperiods"]["mc"]["activate"]
    domergingperiodsdata = data_config["mergingperiods"]["data"]["activate"]
    doml = data_config["ml_study"]["activate"]
    docorrelation = data_config["ml_study"]['docorrelation']
    dotraining = data_config["ml_study"]['dotraining']
    dotesting = data_config["ml_study"]['dotesting']
    doapplytodatamc = data_config["ml_study"]['applytodatamc']
    docrossvalidation = data_config["ml_study"]['docrossvalidation']
    dolearningcurve = data_config["ml_study"]['dolearningcurve']
    doroc = data_config["ml_study"]['doroc']
    doboundary = data_config["ml_study"]['doboundary']
    doimportance = data_config["ml_study"]['doimportance']
    dogridsearch = data_config["ml_study"]['dogridsearch']
    dosignifopt = data_config["ml_study"]['dosignifopt']
    doscancuts = data_config["ml_study"]["doscancuts"]
    #doefficiency = run_config['doefficiency']
    doapplydata = data_config["analysis"]["data"]["doapply"]
    doapplymc = data_config["analysis"]["mc"]["doapply"]
    domergeapplydata = data_config["analysis"]["data"]["domergeapply"]
    domergeapplymc = data_config["analysis"]["mc"]["domergeapply"]
    dohistomassmc = data_config["analysis"]["mc"]["histomass"]
    dohistomassdata = data_config["analysis"]["data"]["histomass"]
    doefficiency = data_config["analysis"]["mc"]["efficiency"]

    dirpklmc = data_param[case]["multi"]["mc"]["pkl"]
    dirpklevtcounter_allmc = data_param[case]["multi"]["mc"]["pkl_evtcounter_all"]
    dirpklskmc = data_param[case]["multi"]["mc"]["pkl_skimmed"]
    dirpklmlmc = data_param[case]["multi"]["mc"]["pkl_skimmed_merge_for_ml"]
    dirpklmltotmc = data_param[case]["multi"]["mc"]["pkl_skimmed_merge_for_ml_all"]
    dirpkldata = data_param[case]["multi"]["data"]["pkl"]
    dirpklevtcounter_alldata = data_param[case]["multi"]["data"]["pkl_evtcounter_all"]
    dirpklskdata = data_param[case]["multi"]["data"]["pkl_skimmed"]
    dirpklmldata = data_param[case]["multi"]["data"]["pkl_skimmed_merge_for_ml"]
    dirpklmltotdata = data_param[case]["multi"]["data"]["pkl_skimmed_merge_for_ml_all"]
    dirpklskdecmc = data_param[case]["analysis"]["mc"]["pkl_skimmed_dec"]
    dirpklskdec_mergedmc = data_param[case]["analysis"]["mc"]["pkl_skimmed_decmerged"]
    dirpklskdecdata = data_param[case]["analysis"]["data"]["pkl_skimmed_dec"]
    dirpklskdec_mergeddata = data_param[case]["analysis"]["data"]["pkl_skimmed_decmerged"]

    dirresultsdata = data_param[case]["analysis"]["data"]["results"]
    dirresultsmc = data_param[case]["analysis"]["mc"]["results"]

    binminarray = data_param[case]["ml"]["binmin"]
    binmaxarray = data_param[case]["ml"]["binmax"]
    raahp = data_param[case]["ml"]["opt"]["raahp"]
    mltype = data_param[case]["ml"]["mltype"]

    mlout = data_param[case]["ml"]["mlout"]
    mlplot = data_param[case]["ml"]["mlplot"]


    mymultiprocessmc = MultiProcesser(data_param[case], run_param, "mc")
    mymultiprocessdata = MultiProcesser(data_param[case], run_param, "data")

    #creating folder if not present
    if doconversionmc is True:
        if checkdirlist(dirpklmc) is True:
            exit()

    if doconversiondata is True:
        if checkdirlist(dirpkldata) is True:
            exit()

    if doskimmingmc is True:
        if checkdirlist(dirpklskmc) or checkdir(dirpklevtcounter_allmc) is True:
            exit()

    if doskimmingdata is True:
        if checkdirlist(dirpklskdata) or checkdir(dirpklevtcounter_alldata) is True:
            exit()

    if domergingmc is True:
        if checkdirlist(dirpklmlmc) is True:
            exit()

    if domergingdata is True:
        if checkdirlist(dirpklmldata) is True:
            exit()

    if domergingperiodsmc is True:
        if checkdir(dirpklmltotmc) is True:
            exit()

    if domergingperiodsdata is True:
        if checkdir(dirpklmltotdata) is True:
            exit()

    if doml is True:
        if checkdir(mlout) or checkdir(mlplot) is True:
            print("check mlout and mlplot")

    if doapplymc is True:
        if checkdirlist(dirpklskdecmc) is True:
            exit()

    if doapplydata is True:
        if checkdirlist(dirpklskdecdata) is True:
            exit()

    if domergeapplymc is True:
        if checkdirlist(dirpklskdec_mergedmc) is True:
            exit()

    if domergeapplydata is True:
        if checkdirlist(dirpklskdec_mergeddata) is True:
            exit()

    if dohistomassmc is True:
        if checkdirlist(dirresultsmc) is True:
            exit()

    if dohistomassdata is True:
        if checkdirlist(dirresultsdata) is True:
            print("folder exists")

    #perform the analysis flow
    if doconversionmc == 1:
        mymultiprocessmc.multi_unpack_allperiods()

    if doconversiondata == 1:
        mymultiprocessdata.multi_unpack_allperiods()

    if doskimmingmc == 1:
        mymultiprocessmc.multi_skim_allperiods()

    if doskimmingdata == 1:
        mymultiprocessdata.multi_skim_allperiods()

    if domergingmc == 1:
        mymultiprocessmc.multi_mergeml_allperiods()

    if domergingdata == 1:
        mymultiprocessdata.multi_mergeml_allperiods()

    if domergingperiodsmc == 1:
        mymultiprocessmc.multi_mergeml_allinone()

    if domergingperiodsdata == 1:
        mymultiprocessdata.multi_mergeml_allinone()

    if doml is True:
        index = 0
        for binmin, binmax in zip(binminarray, binmaxarray):
            myopt = Optimiser(data_param[case], case,
                              data_model[mltype], grid_param, binmin, binmax,
                              raahp[index])
            if docorrelation is True:
                myopt.do_corr()
            if dotraining is True:
                myopt.do_train()
            if dotesting is True:
                myopt.do_test()
            if doapplytodatamc is True:
                myopt.do_apply()
            if docrossvalidation is True:
                myopt.do_crossval()
            if dolearningcurve is True:
                myopt.do_learningcurve()
            if doroc is True:
                myopt.do_roc()
            if doimportance is True:
                myopt.do_importance()
            if dogridsearch is True:
                myopt.do_grid()
            if doboundary is True:
                myopt.do_boundary()
            if dosignifopt is True:
                myopt.do_significance()
            if doscancuts is True:
                myopt.do_scancuts()
            index = index + 1

    if doapplydata is True:
        mymultiprocessdata.multi_apply_allperiods()
    if doapplymc is True:
        mymultiprocessmc.multi_apply_allperiods()
    if domergeapplydata is True:
        mymultiprocessdata.multi_mergeapply_allperiods()
    if domergeapplymc is True:
        mymultiprocessmc.multi_mergeapply_allperiods()
    if dohistomassmc is True:
        mymultiprocessmc.multi_histomass()
    if dohistomassdata is True:
        mymultiprocessdata.multi_histomass()
    if doefficiency is True:
        mymultiprocessmc.multi_efficiency()

do_entire_analysis()
