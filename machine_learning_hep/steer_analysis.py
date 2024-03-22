#############################################################################
##  © Copyright CERN 2023. All rights not expressly granted are reserved.  ##
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

import argparse
import importlib
import os
from os.path import exists
import subprocess
import sys

import yaml
# unclear why shap needs to be imported from here,
# segfaults when imported from within other modules
import shap # pylint: disable=unused-import

from .analysis.analyzer_manager import AnalyzerManager
from .config import update_config
from .logger import configure_logger, get_logger
from .utilities_files import checkmakedirlist, checkmakedir, checkdirlist, checkdir, delete_dirlist

def do_entire_analysis(data_config: dict, data_param: dict, data_param_overwrite: dict, # pylint: disable=too-many-locals, too-many-statements, too-many-branches
                       data_model: dict, run_param: dict, clean: bool):

    logger = get_logger()
    logger.info("Do analysis chain")

    # If we are here we are interested in the very first key in the parameters database
    case = list(data_param.keys())[0]

    # Update database accordingly if needed
    update_config(data_param, data_config, data_param_overwrite)

    dodownloadalice = data_config["download"]["alice"]["activate"]
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
    doapplytodatamc = data_config["ml_study"]['doapplytodatamc']
    docrossvalidation = data_config["ml_study"]['docrossvalidation']
    dolearningcurve = data_config["ml_study"]['dolearningcurve']
    doroc = data_config["ml_study"]['doroc']
    doroctraintest = data_config["ml_study"]['doroctraintest']
    doboundary = data_config["ml_study"]['doboundary']
    doimportance = data_config["ml_study"]['doimportance']
    doimportanceshap = data_config["ml_study"]['doimportanceshap']
    dogridsearch = data_config["ml_study"]['dogridsearch']
    dobayesianopt = data_config["ml_study"]['dobayesianopt']
    doefficiencyml = data_config["ml_study"]['doefficiency']
    dosignifopt = data_config["ml_study"]['dosignifopt']
    doscancuts = data_config["ml_study"]["doscancuts"]
    doplotdistr = data_config["ml_study"]["doplotdistr"]
    doapplydata = data_config["mlapplication"]["data"]["doapply"]
    doapplymc = data_config["mlapplication"]["mc"]["doapply"]
    domergeapplydata = data_config["mlapplication"]["data"]["domergeapply"]
    domergeapplymc = data_config["mlapplication"]["mc"]["domergeapply"]
    docontinueapplydata = data_config["mlapplication"]["data"]["docontinueafterstop"]
    docontinueapplymc = data_config["mlapplication"]["mc"]["docontinueafterstop"]
    dohistomassmc = data_config["analysis"]["mc"]["histomass"]
    dohistomassdata = data_config["analysis"]["data"]["histomass"]
    doefficiency = data_config["analysis"]["mc"]["efficiency"]
    efficiency_resp = data_config["analysis"]["mc"].get("efficiency_resp", False)
    doresponse = data_config["analysis"]["mc"]["response"]
    dofeeddown = data_config["analysis"]["mc"]["feeddown"]
    dounfolding = data_config["analysis"]["mc"]["dounfolding"]
    dojetsystematics = data_config["analysis"]["data"]["dojetsystematics"]
    doqa = data_config["analysis"]["doqa"]
    dofit = data_config["analysis"]["dofit"]
    dosidebandsub = data_config["analysis"].get("dosidebandsub", False)
    doeff = data_config["analysis"]["doeff"]
    docross = data_config["analysis"]["docross"]
    doplotsval = data_config["analysis"]["doplotsval"]
    doplots = data_config["analysis"]["doplots"]
    dosyst = data_config["analysis"]["dosyst"]
    do_syst_ml = data_config["systematics"]["cutvar"]["activate"]
    do_syst_ml_only_analysis = data_config["systematics"]["cutvar"]["do_only_analysis"]
    do_syst_ml_resume = data_config["systematics"]["cutvar"]["resume"]
    doanaperperiod = data_config["analysis"]["doperperiod"]
    typean = data_config["analysis"]["type"]

    dojetstudies = data_config["analysis"]["dojetstudies"]

    dp = data_param[case]["multi"]["mc"]
    dirprefixmc = dp.get("prefix_dir", "")
    dirpklmc = [dirprefixmc + os.path.expandvars(p) for p in dp["pkl"]]
    dirpklskmc = [dirprefixmc + os.path.expandvars(p) for p in dp["pkl_skimmed"]]
    dirpklmlmc = [dirprefixmc + os.path.expandvars(p) for p in dp["pkl_skimmed_merge_for_ml"]]
    dirpklevtcounter_allmc = dirprefixmc + os.path.expandvars(dp["pkl_evtcounter_all"])
    dirpklmltotmc = dirprefixmc + os.path.expandvars(dp["pkl_skimmed_merge_for_ml_all"])

    dp = data_param[case]["multi"]["data"]
    dirprefixdata = dp.get("prefix_dir", "")
    dirpkldata = [dirprefixdata + os.path.expandvars(p) for p in dp["pkl"]]
    dirpklskdata = [dirprefixdata + os.path.expandvars(p) for p in dp["pkl_skimmed"]]
    dirpklmldata = [dirprefixdata + os.path.expandvars(p) for p in dp["pkl_skimmed_merge_for_ml"]]
    dirpklevtcounter_alldata = dirprefixdata + os.path.expandvars(dp["pkl_evtcounter_all"])
    dirpklmltotdata = dirprefixdata + os.path.expandvars(dp["pkl_skimmed_merge_for_ml_all"])

    dp = data_param[case]["mlapplication"]["mc"]
    dirprefixmcapp = dp.get("prefix_dir_app", "")
    dirpklskdecmc = [dirprefixmcapp + p for p in dp["pkl_skimmed_dec"]]
    dirpklskdec_mergedmc = [dirprefixmcapp + p for p in dp["pkl_skimmed_decmerged"]]

    dp = data_param[case]["mlapplication"]["data"]
    dirprefixdataapp = dp.get("prefix_dir_app", "")
    dirpklskdecdata = [dirprefixdataapp + p for p in dp["pkl_skimmed_dec"]]
    dirpklskdec_mergeddata = [dirprefixdataapp + p for p in dp["pkl_skimmed_decmerged"]]

    dp = data_param[case]["analysis"][typean]["data"]
    dirprefixdatares = dp.get("prefix_dir_res", "")
    dirresultsdata = [dirprefixdatares + os.path.expandvars(p) for p in dp["results"]]
    dirresultsdatatot = dirprefixdatares + os.path.expandvars(dp["resultsallp"])

    dp = data_param[case]["analysis"][typean]["mc"]
    dirprefixmcres = dp.get("prefix_dir_res", "")
    dirresultsmc = [dirprefixmcres + os.path.expandvars(p) for p in dp["results"]]
    dirresultsmctot = dirprefixmcres + os.path.expandvars(dp["resultsallp"])

    binminarray = data_param[case]["ml"]["binmin"]
    binmaxarray = data_param[case]["ml"]["binmax"]
    multbkg = data_param[case]["ml"]["mult_bkg"]
    raahp = data_param[case]["ml"]["opt"]["raahp"]
    mltype = data_param[case]["ml"]["mltype"]
    training_vars = data_param[case]["variables"]["var_training"]

    dirprefixml = data_param[case]["ml"].get("prefix_dir_ml", "")
    mlout = dirprefixml + data_param[case]["ml"]["mlout"]
    mlplot = dirprefixml + data_param[case]["ml"]["mlplot"]

    proc_type = data_param[case]["analysis"][typean]["proc_type"]

    counter = 0
    if doconversionmc:
        counter = counter + checkdirlist(dirpklmc)

    if doconversiondata:
        counter = counter + checkdirlist(dirpkldata)

    if doskimmingmc:
        checkdirlist(dirpklskmc)
        counter = counter + checkdir(dirpklevtcounter_allmc)

    if doskimmingdata:
        counter = counter + checkdirlist(dirpklskdata)
        counter = counter + checkdir(dirpklevtcounter_alldata)

    if domergingmc:
        counter = counter + checkdirlist(dirpklmlmc)

    if domergingdata:
        counter = counter + checkdirlist(dirpklmldata)

    if domergingperiodsmc:
        counter = counter + checkdir(dirpklmltotmc)

    if domergingperiodsdata:
        counter = counter + checkdir(dirpklmltotdata)

    if not docontinueapplymc:
        if doapplymc:
            counter = counter + checkdirlist(dirpklskdecmc)

        if domergeapplymc:
            counter = counter + checkdirlist(dirpklskdec_mergedmc)

    if not docontinueapplydata:
        if doapplydata:
            counter = counter + checkdirlist(dirpklskdecdata)

        if domergeapplydata:
            counter = counter + checkdirlist(dirpklskdec_mergeddata)

    if dohistomassmc:
        counter = counter + checkdirlist(dirresultsmc)
        counter = counter + checkdir(dirresultsmctot)

    if dohistomassdata:
        counter = counter + checkdirlist(dirresultsdata)
        counter = counter + checkdir(dirresultsdatatot)

    if counter < 0:
        sys.exit()
    # check and create directories

    if doconversionmc:
        checkmakedirlist(dirpklmc)

    if doconversiondata:
        checkmakedirlist(dirpkldata)

    if doskimmingmc:
        checkmakedirlist(dirpklskmc)
        checkmakedir(dirpklevtcounter_allmc)

    if doskimmingdata:
        checkmakedirlist(dirpklskdata)
        checkmakedir(dirpklevtcounter_alldata)

    if domergingmc:
        checkmakedirlist(dirpklmlmc)

    if domergingdata:
        checkmakedirlist(dirpklmldata)

    if domergingperiodsmc:
        checkmakedir(dirpklmltotmc)

    if domergingperiodsdata:
        checkmakedir(dirpklmltotdata)

    if doml:
        checkmakedir(mlout)
        checkmakedir(mlplot)

    if not docontinueapplymc:
        if doapplymc:
            checkmakedirlist(dirpklskdecmc)

        if domergeapplymc:
            checkmakedirlist(dirpklskdec_mergedmc)

    if not docontinueapplydata:
        if doapplydata:
            checkmakedirlist(dirpklskdecdata)

        if domergeapplydata:
            checkmakedirlist(dirpklskdec_mergeddata)

    if dohistomassmc:
        checkmakedirlist(dirresultsmc)
        checkmakedir(dirresultsmctot)

    if dohistomassdata:
        checkmakedirlist(dirresultsdata)
        checkmakedir(dirresultsdatatot)

    def mlhepmod(name):
        return importlib.import_module(f"..{name}", __name__)

    from machine_learning_hep.multiprocesser import MultiProcesser # pylint: disable=import-outside-toplevel
    syst_class = mlhepmod('analysis.systematics').SystematicsMLWP
    if proc_type == "Dhadrons":
        proc_class = mlhepmod('processerdhadrons').ProcesserDhadrons
        ana_class = mlhepmod('analysis.analyzerdhadrons').AnalyzerDhadrons
    elif proc_type == "Dhadrons_mult":
        proc_class = mlhepmod('processerdhadrons_mult').ProcesserDhadrons_mult
        ana_class = mlhepmod('analysis.analyzerdhadrons_mult').AnalyzerDhadrons_mult
    elif proc_type == "Dhadrons_jet":
        proc_class = mlhepmod('processerdhadrons_jet').ProcesserDhadrons_jet
        ana_class = mlhepmod('analysis.analyzer_jet').AnalyzerJet
    elif proc_type == "Jets":
        proc_class = mlhepmod("processer_jet").ProcesserJets
        ana_class = mlhepmod("analysis.analyzer_jets").AnalyzerJets
    else:
        proc_class = mlhepmod('processer').Processer
        ana_class = mlhepmod('analysis.analyzer').Analyzer

    mymultiprocessmc = MultiProcesser(
        case, proc_class, data_param[case], typean, run_param, "mc")
    mymultiprocessdata = MultiProcesser(
        case, proc_class, data_param[case], typean, run_param, "data")

    ana_mgr = AnalyzerManager(ana_class, data_param[case], case, typean, doanaperperiod)

    analyzers = ana_mgr.get_analyzers()
    # For ML WP systematics
    if mltype == "MultiClassification":
        syst_ml_pt_cl0 = syst_class(data_param[case], case, typean, analyzers,
                                    mymultiprocessmc, mymultiprocessdata, 0)
        syst_ml_pt_cl1 = syst_class(data_param[case], case, typean, analyzers,
                                    mymultiprocessmc, mymultiprocessdata, 1)
    else:
        syst_ml_pt = syst_class(data_param[case], case, typean, analyzers,
                                mymultiprocessmc, mymultiprocessdata)

    #perform the analysis flow
    if dodownloadalice:
        subprocess.call("../cplusutilities/Download.sh")

    if doconversionmc:
        mymultiprocessmc.multi_unpack_allperiods()

    if doconversiondata:
        mymultiprocessdata.multi_unpack_allperiods()

    if doskimmingmc:
        mymultiprocessmc.multi_skim_allperiods()

    if doskimmingdata:
        mymultiprocessdata.multi_skim_allperiods()

    if domergingmc:
        mymultiprocessmc.multi_mergeml_allperiods()

    if domergingdata:
        mymultiprocessdata.multi_mergeml_allperiods()

    if domergingperiodsmc:
        mymultiprocessmc.multi_mergeml_allinone()

    if domergingperiodsdata:
        mymultiprocessdata.multi_mergeml_allinone()

    if doml:
        from machine_learning_hep.optimiser import Optimiser # pylint: disable=import-outside-toplevel
        for index, (binmin, binmax) in enumerate(zip(binminarray, binmaxarray)):
            myopt = Optimiser(data_param[case], case, typean,
                              data_model[mltype], binmin, binmax, multbkg[index],
                              raahp[index], training_vars[index], index)
            if docorrelation:
                myopt.do_corr()
            if dotraining:
                myopt.do_train()
            if dotesting:
                myopt.do_test()
            if doapplytodatamc:
                myopt.do_apply()
            if docrossvalidation:
                myopt.do_crossval()
            if dolearningcurve:
                myopt.do_learningcurve()
            if doroc:
                myopt.do_roc()
            if doroctraintest:
                myopt.do_roc_train_test()
            if doplotdistr:
                myopt.do_plot_model_pred()
            if doimportance:
                myopt.do_importance()
            if doimportanceshap:
                myopt.do_importance_shap()
            if dogridsearch:
                myopt.do_grid()
            if dobayesianopt:
                myopt.do_bayesian_opt()
            if doboundary:
                myopt.do_boundary()
            if doefficiencyml:
                myopt.do_efficiency()
            if dosignifopt:
                myopt.do_significance()
            if doscancuts:
                myopt.do_scancuts()

    if doapplydata:
        mymultiprocessdata.multi_apply_allperiods()
    if doapplymc:
        mymultiprocessmc.multi_apply_allperiods()
    if domergeapplydata:
        mymultiprocessdata.multi_mergeapply_allperiods()
    if domergeapplymc:
        mymultiprocessmc.multi_mergeapply_allperiods()

    if dohistomassmc:
        mymultiprocessmc.multi_histomass()
    if dohistomassdata:
        # After-burner in case of a mult analysis to obtain "correctionsweight.root"
        # for merged-period data
        # pylint: disable=fixme
        # FIXME Can only be run here because result directories are constructed when histomass
        #       is run. If this step was independent, histomass would always complain that the
        #       result directory already exists.
        mymultiprocessdata.multi_histomass()
    if doefficiency:
        mymultiprocessmc.multi_efficiency()
    analyze_steps = []
    if efficiency_resp:
        analyze_steps.append("efficiency_inclusive")
        ana_mgr.analyze(analyze_steps)
    if doresponse:
        mymultiprocessmc.multi_response()

    # Collect all desired analysis steps
    analyze_steps = []
    if doqa:
        analyze_steps.append("qa")
    if dofit:
        analyze_steps.append("fit")
    if dosidebandsub:
        analyze_steps.append("sidebandsub")
    if dosyst:
        analyze_steps.append("yield_syst")
    if doeff:
        analyze_steps.append("efficiency")
    if dojetstudies:
        if not dofit:
            analyze_steps.append("fit")
        if not doeff:
            analyze_steps.append("efficiency")
        analyze_steps.append("sideband_sub")
    if dofeeddown:
        analyze_steps.append("feeddown")
    if dounfolding:
        analyze_steps.append("unfolding")
        analyze_steps.append("unfolding_closure")
    if dojetsystematics:
        analyze_steps.append("jetsystematics")
    if docross:
        analyze_steps.append("makenormyields")
    if doplots:
        analyze_steps.append("plotternormyields")
    if doplotsval:
        analyze_steps.append("plottervalidation")

    # Now do the analysis
    ana_mgr.analyze(analyze_steps)

    if do_syst_ml:
        if mltype == "MultiClassification":
            syst_ml_pt_cl0.ml_systematics(do_syst_ml_only_analysis, do_syst_ml_resume)
            syst_ml_pt_cl1.ml_systematics(do_syst_ml_only_analysis, do_syst_ml_resume)
        else:
            syst_ml_pt.ml_systematics(do_syst_ml_only_analysis, do_syst_ml_resume)

    # Delete per-period results.
    if clean:
        print("Cleaning")
        if doanaperperiod:
            print("Per-period analysis enabled. Skipping.")
        else:
            if not delete_dirlist(dirresultsmc + dirresultsdata):
                print("Error: Failed to complete cleaning.")

    logger.info("Done")

def load_config(user_path: str, default_path=None) -> dict:
    """
    Quickly extract either configuration given by user and fall back to package default if no user
    config given.
    Args:
        user_path: path to YAML file
        default_path: tuple were to find the resource and name of resource
    Returns:
        dictionary built from YAML
    """
    if not user_path and not default_path:
        return None

    if user_path:
        if not exists(user_path):
            get_logger().fatal("The file %s does not exist", user_path)
            sys.exit(-1)
        with open(user_path, 'r', encoding='utf-8') as stream:
            cfg = yaml.safe_load(stream)
    else:
        res = importlib.resources.files(default_path[0]).joinpath(default_path[1]).read_bytes()
        cfg = yaml.safe_load(res)
    return cfg

def main(args=None):
    """
    This is used as the entry point for ml-analysis.
    Read optional command line arguments and launch the analysis.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="activate debug log level")
    parser.add_argument("--quiet", '-q', action="store_true", help="quiet logging")
    parser.add_argument("--log-file", dest="log_file", help="file to print the log to")
    parser.add_argument("--run-config", "-r", dest="run_config",
                        help="the run configuration to be used")
    parser.add_argument("--database-analysis", "-d", dest="database_analysis",
                        help="analysis database to be used", required=True)
    parser.add_argument("--database-overwrite", dest="database_overwrite",
                        help="overwrite fields in analysis database")
    parser.add_argument("--database-ml-models", dest="database_ml_models",
                        help="ml model database to be used")
    parser.add_argument("--database-run-list", dest="database_run_list",
                        help="run list database to be used")
    parser.add_argument("--analysis", "-a", dest="type_ana",
                        help="choose type of analysis")
    parser.add_argument("--clean", "-c", action="store_true",
                        help="delete per-period results at the end")

    args = parser.parse_args(args)

    configure_logger(args.debug, args.log_file, args.quiet)

    # Extract which database and run config to be used
    pkg_data = "machine_learning_hep.data"
    pkg_data_run_config = "machine_learning_hep.submission"
    run_config = load_config(args.run_config, (pkg_data_run_config, "default_complete.yml"))
    if args.type_ana is not None:
        run_config["analysis"]["type"] = args.type_ana

    db_analysis = load_config(args.database_analysis)
    db_analysis_overwrite = load_config(args.database_overwrite)
    db_ml_models = load_config(args.database_ml_models, (pkg_data, "config_model_parameters.yml"))
    db_run_list = load_config(args.database_run_list, (pkg_data, "database_run_list.yml"))

    # Run the chain
    do_entire_analysis(run_config, db_analysis, db_analysis_overwrite, db_ml_models, db_run_list,
                       args.clean)

if __name__ == '__main__':
    main()
