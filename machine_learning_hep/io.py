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
Methods to: manage input/output
"""

import os
import yaml

def parse_yaml(filepath):
    """
    Parse a YAML file and return dictionary
    Args:
        filepath: Path to the YAML file to be parsed.
    """
    if not os.path.isfile(filepath):
        print("YAML file %s does not exist." % filepath)
        exit(1)
    with open(filepath) as f:
        return yaml.safe_load(f)


def checkdir(path):
    """
    Check for existence of directory and create if not existing
    """
    if not os.path.exists(path):
        os.makedirs(path)
