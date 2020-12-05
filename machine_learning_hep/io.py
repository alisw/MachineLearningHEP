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
from numbers import Number
from inspect import isclass
import yaml
from machine_learning_hep.logger import get_logger

def dict_yamlable(params):
    """make dictionary ready for yaml.safe_dump
    Args:
        params: dict
            dictionary to modify
    Returns:
        dict: modified dictionary which can be used with yaml.safe_dump

    """
    params_seri = {}
    for k, v in params.items():
        if isinstance(v, dict):
            params_seri[k] = dict_yamlable(v)
        else:
            if isinstance(v, (Number, str, list, tuple)):
                # This we can handle with standard PyYAML
                params_seri[k] = v
            elif isclass(v):
                params_seri[k] = f"custom:{v.__name__}"
            else:
                params_seri[k] = f"custom:{v.__class__.__name__}"
    return params_seri


def parse_yaml(filepath):
    """
    Parse a YAML file and return dictionary
    Args:
        filepath: Path to the YAML file to be parsed.
    """
    if not os.path.isfile(filepath):
        get_logger().critical("YAML file %s does not exist.", filepath)
    with open(filepath) as f:
        return yaml.safe_load(f)


def dump_yaml_from_dict(to_yaml, path):
    path = os.path.expanduser(path)
    with open(path, "w") as stream:
        yaml.safe_dump(to_yaml, stream, default_flow_style=False, allow_unicode=False)


def checkdir(path):
    """
    Check for existence of directory and create if not existing
    """
    if not os.path.exists(path):
        os.makedirs(path)

def print_dict(to_be_printed, indent=0, skip=None):
    for key, value in to_be_printed.items():
        if isinstance(skip, list) and key in skip:
            continue
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
