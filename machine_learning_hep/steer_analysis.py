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
import sys
import subprocess
import argparse
from os.path import exists
import yaml
from pkg_resources import resource_stream

# To set batch mode immediately
from ROOT import gROOT # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.multiprocesserinclusive import MultiProcesserInclusive
from machine_learning_hep.processerinclusive import ProcesserInclusive
#from machine_learning_hep.doskimming import conversion, merging, merging_period, skim
#from machine_learning_hep.doclassification_regression import doclassification_regression
#from machine_learning_hep.doanalysis import doanalysis
#from machine_learning_hep.extractmasshisto import extractmasshisto
#from machine_learning_hep.efficiencyan import analysis_eff
from machine_learning_hep.config import update_config
from  machine_learning_hep.utilities import checkmakedirlist, checkmakedir
from  machine_learning_hep.utilities import checkdirlist, checkdir, delete_dirlist
from  machine_learning_hep.logger import configure_logger, get_logger

from machine_learning_hep.analysis.analyzer_manager import AnalyzerManager
from machine_learning_hep.analysis.analyzer import Analyzer
from machine_learning_hep.analysis.analyzer_jet import AnalyzerJet

from machine_learning_hep.analysis.systematics import Systematics

try:
# FIXME(https://github.com/abseil/abseil-py/issues/99) # pylint: disable=fixme
# FIXME(https://github.com/abseil/abseil-py/issues/102) #pylint: disable=fixme
# Unfortunately, many libraries that include absl (including Tensorflow)
# will get bitten by double-logging due to absl's incorrect use of
# the python logging library:
#   2019-07-19 23:47:38,829 my_logger   779 : test
#   I0719 23:47:38.829330 139904865122112 foo.py:63] test
#   2019-07-19 23:47:38,829 my_logger   779 : test
#   I0719 23:47:38.829469 139904865122112 foo.py:63] test
# The code below fixes this double-logging.  FMI see:
#   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
    import logging
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler) # pylint: disable=protected-access
    absl.logging._warn_preinit_stderr = False # pylint: disable=protected-access
except Exception as e: # pylint: disable=broad-except
    print("##############################")
    print("Failed to fix absl logging bug", e)
    print("##############################")


def do_entire_analysis(data_config: dict, data_param: dict, data_param_overwrite: dict, # pylint: disable=too-many-locals, too-many-statements, too-many-branches
                       data_model: dict, run_param: dict, clean: bool):

    # Disable any graphical stuff. No TCanvases opened and shown by default
    gROOT.SetBatch(True)
    print("IAM TRYING")
    logger = get_logger()
    logger.info("Do analysis chain")

    # If we are here we are interested in the very first key in the parameters database
    case = list(data_param.keys())[0]

    # Update database accordingly if needed
    update_config(data_param, data_config, data_param_overwrite)

    dodownloadalice = data_config["download"]["alice"]["activate"]
    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    doanaperperiod = data_config["analysis"]["doperperiod"]
    typean = data_config["analysis"]["type"]
    dohistomassmc = data_config["analysis"]["mc"]["histomass"]
    dohistomassdata = data_config["analysis"]["data"]["histomass"]
    doresponse = data_config["analysis"]["mc"]["response"]
    dounfolding = data_config["analysis"]["mc"]["dounfolding"]
    dojetsystematics = data_config["analysis"]["data"]["dojetsystematics"]
    dirpklmc = data_param[case]["multi"]["mc"]["pkl"]
    dirpkldata = data_param[case]["multi"]["data"]["pkl"]
    dirresultsdata = data_param[case]["analysis"][typean]["data"]["results"]
    dirresultsmc = data_param[case]["analysis"][typean]["mc"]["results"]
    dirresultsdatatot = data_param[case]["analysis"][typean]["data"]["resultsallp"]
    dirresultsmctot = data_param[case]["analysis"][typean]["mc"]["resultsallp"]
    proc_type = data_param[case]["analysis"][typean]["proc_type"]

    #creating folder if not present
    counter = 0
    if doconversionmc is True:
        counter = counter + checkdirlist(dirpklmc)

    if doconversiondata is True:
        counter = counter + checkdirlist(dirpkldata)

    if dohistomassmc is True:
        counter = counter + checkdirlist(dirresultsmc)
        counter = counter + checkdir(dirresultsmctot)

    if dohistomassdata is True:
        counter = counter + checkdirlist(dirresultsdata)
        counter = counter + checkdir(dirresultsdatatot)

    if counter < 0:
        sys.exit()
    # check and create directories

    if doconversionmc is True:
        checkmakedirlist(dirpklmc)

    if doconversiondata is True:
        checkmakedirlist(dirpkldata)

    if dohistomassmc is True:
        checkmakedirlist(dirresultsmc)
        checkmakedir(dirresultsmctot)

    if dohistomassdata is True:
        checkmakedirlist(dirresultsdata)
        checkmakedir(dirresultsdatatot)
    proc_class = ProcesserInclusive
    mymultiprocessmc = MultiProcesserInclusive(case, proc_class, data_param[case], typean, "mc")
    mymultiprocessdata = MultiProcesserInclusive(case, proc_class, data_param[case], typean, "data")
    ana_mgr = AnalyzerManager(AnalyzerJet, data_param[case], case, typean, doanaperperiod)
    if dodownloadalice == 1:
       subprocess.call("../cplusutilities/Download.sh")

    if doconversionmc == 1:
        mymultiprocessmc.multi_unpack_allperiods()

    if doconversiondata == 1:
        mymultiprocessdata.multi_unpack_allperiods()

    if dohistomassmc is True:
        mymultiprocessmc.multi_histomass()

    if dohistomassdata is True:
        mymultiprocessdata.multi_histomass()

    if doresponse is True:
        mymultiprocessmc.multi_response()
    analyze_steps = []
    if dounfolding is True:
        analyze_steps.append("unfolding")
        analyze_steps.append("unfolding_closure")
    if dojetsystematics is True:
        analyze_steps.append("jetsystematics")
    print("Done")
    # Now do the analysis
    ana_mgr.analyze(*analyze_steps)

    # Delete per-period results.
    if clean:
        print("Cleaning")
        if doanaperperiod:
            print("Per-period analysis enabled. Skipping.")
        else:
            if not delete_dirlist(dirresultsmc + dirresultsdata):
                print("Error: Failed to complete cleaning.")

    print("Done")

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

    stream = None
    if user_path:
        if not exists(user_path):
            get_logger().fatal("The file %s does not exist", user_path)
        stream = open(user_path)
    else:
        stream = resource_stream(default_path[0], default_path[1])
    return yaml.safe_load(stream)

def main():
    """
    This is used as the entry point for ml-analysis.
    Read optional command line arguments and launch the analysis.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="activate debug log level")
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

    args = parser.parse_args()

    configure_logger(args.debug, args.log_file)
    # Extract which database and run config to be used
    pkg_data = "machine_learning_hep.data"
    pkg_data_run_config = "machine_learning_hep.submission"
    run_config = load_config(args.run_config, (pkg_data_run_config,
                                               "default_complete.yml"))
    if args.type_ana is not None:
        run_config["analysis"]["type"] = args.type_ana

    db_analysis = load_config(args.database_analysis)
    db_analysis_overwrite = load_config(args.database_overwrite)
    db_ml_models = load_config(args.database_ml_models, (pkg_data, "config_model_parameters.yml"))
    db_run_list = load_config(args.database_run_list, (pkg_data, "database_run_list.yml"))

    # Run the chain
    do_entire_analysis(run_config, db_analysis, db_analysis_overwrite, db_ml_models, db_run_list,
                       args.clean)
