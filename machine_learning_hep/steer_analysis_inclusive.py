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

from machine_learning_hep.config import update_config
from  machine_learning_hep.utilities import checkmakedirlist, checkmakedir
from  machine_learning_hep.utilities import checkdirlist, checkdir, delete_dirlist
from  machine_learning_hep.logger import configure_logger, get_logger
from machine_learning_hep.multiprocesserinclusive import MultiProcesserInclusive
from machine_learning_hep.processerinclusive import ProcesserInclusive

try:
    import logging
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler) # pylint: disable=protected-access
    absl.logging._warn_preinit_stderr = False # pylint: disable=protected-access
except Exception as e: # pylint: disable=broad-except
    print("##############################")
    print("Failed to fix absl logging bug", e)
    print("##############################")

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def do_entire_analysis():

    data_config = yaml.safe_load(open("submission/default_complete_inclusive.yml", 'r'))
    data_param = yaml.safe_load(open("data/data_prod_20200304/database_jetinclusive.yml", 'r'))
    # Disable any graphical stuff. No TCanvases opened and shown by default
    gROOT.SetBatch(True)

    logger = get_logger()
    logger.info("Do analysis chain")

    # If we are here we are interested in the very first key in the parameters database
    case = "inclusivejets"
    proc_class = "light"
    typean = "jet_zg"

    dodownloadalice = data_config["download"]["alice"]["activate"]
    doconversionmc = data_config["conversion"]["mc"]["activate"]
    doconversiondata = data_config["conversion"]["data"]["activate"]
    doanaperperiod = data_config["analysis"]["doperperiod"]
    dohistomassmc = data_config["analysis"]["mc"]["histomass"]
    dohistomassdata = data_config["analysis"]["data"]["histomass"]

    dirpklmc = data_param[case]["multi"]["mc"]["pkl"]
    dirpkldata = data_param[case]["multi"]["data"]["pkl"]
    dirresultsdata = data_param[case]["analysis"][typean]["data"]["results"]
    dirresultsmc = data_param[case]["analysis"][typean]["mc"]["results"]
    dirresultsdatatot = data_param[case]["analysis"][typean]["data"]["resultsallp"]
    dirresultsmctot = data_param[case]["analysis"][typean]["mc"]["resultsallp"]

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

    mymultiprocessmc = MultiProcesserInclusive(case, proc_class, data_param[case], typean, "mc")
    mymultiprocessdata = MultiProcesserInclusive(case, proc_class, data_param[case], typean, "data")
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
    print("Done")

do_entire_analysis()
