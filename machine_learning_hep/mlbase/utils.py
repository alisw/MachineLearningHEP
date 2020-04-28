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
Additional utils for ML
"""

import sys
from os.path import isfile, expanduser
import yaml


def parse_yaml(filepath):
    """
    Parse a YAML file and return dictionary
    Args:
        filepath: Path to the YAML file to be parsed.
    """
    if not isfile(filepath):
        print(f"YAML file {filepath} does not exist. Exit...")
        sys.exit(1)
    with open(filepath) as f:
        return yaml.safe_load(f)


def dump_yaml_from_dict(to_yaml, path):
    path = expanduser(path)
    with open(path, "w") as stream:
        yaml.safe_dump(to_yaml, stream, default_flow_style=False, allow_unicode=False)
