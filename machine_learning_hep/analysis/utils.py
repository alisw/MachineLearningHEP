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

from os.path import join, expanduser

from machine_learning_hep.utilities import mergerootfiles, get_timestamp_string
from machine_learning_hep.logger import get_logger

def multi_preparenorm(database, case, typean, doperiodbyperiod):

    logger = get_logger()

    lper_normfilesorig = []
    lper_normfiles = []
    dlper_valevtroot = database["validation"]["data"]["dir"]
    resultsdata = database["analysis"][typean]["data"]["results"]

    for res_path, lper_val in zip(resultsdata, dlper_valevtroot):
        lper_normfilesorig.append(join(lper_val, "correctionsweights.root"))
        lper_normfiles.append(join(res_path, "correctionsweights.root"))

    f_normmerged = join(database["analysis"][typean]["data"]["resultsallp"],
                        "correctionsweights.root")

    listempty = []
    tmp_merged = expanduser(f"~/tmp/hadd/{case}_{typean}/norm_analyzer/{get_timestamp_string()}/")
    useperiod = database["analysis"][typean]["useperiod"]

    for indexp in range(len(resultsdata)):
        logger.info("Origin path: %s, target path: %s", lper_normfilesorig[indexp],
                    lper_normfiles[indexp])
        mergerootfiles([lper_normfilesorig[indexp]], lper_normfiles[indexp], tmp_merged)
        if doperiodbyperiod and useperiod[indexp]:
            listempty.append(lper_normfiles[indexp])

    mergerootfiles(listempty, f_normmerged, tmp_merged)
