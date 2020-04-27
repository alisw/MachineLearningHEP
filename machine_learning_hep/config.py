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
Methods to: update/assert database and run configuration
"""

from itertools import product
from machine_learning_hep.logger import get_logger
from machine_learning_hep.do_variations import modify_dictionary


# disable pylint unused-argument because this is done already in view of updating the
# database depending on info in there
def update_config(database: dict, run_config: dict, database_overwrite=None): # pylint: disable=unused-argument
    """Update database before usage

    1. overwrite with potential additional user configuration
    2. adjust paths
    This adjusts database inline ==> no return value

    Args:
        database: dict
            input database as read from YAML
        run_config: dict
            input run configuration as read from default_<stage>.yaml
        database_overwrite: dict (optional)
            substructured corresponding to database used to overwrite
            corresponding fields in database
    """

    logger = get_logger()

    # Extract the case
    case = list(database.keys())[0]
    database = database[case]

    # First overwrite as required by the user
    # To be implemented
    if database_overwrite:
        logger.info("Updating database fields with custom user input")
        modify_dictionary(database, database_overwrite)

    # If not an ML analysis...
    if not database["doml"]:
        logger.info("Not an ML analysis, adjust paths and settings accordingly")
        # ...append "_std" to paths where necessary
        data_mc = ("data", "mc")
        pkl_keys = ("pkl_skimmed_dec", "pkl_skimmed_decmerged")
        for keys in product(data_mc, pkl_keys):
            database["mlapplication"][keys[0]][keys[1]][:] = \
                    [f"{path}_std" for path in database["mlapplication"][keys[0]][keys[1]]]
        # ...set the ML working point all to 0
        for k in data_mc:
            database["mlapplication"]["probcutpresel"][k][:] = \
                    [0] * len(database["mlapplication"]["probcutpresel"][k])
        database["mlapplication"]["probcutoptimal"][:] \
                = [0] * len(database["mlapplication"]["probcutoptimal"])
