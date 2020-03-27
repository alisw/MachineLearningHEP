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
Script to run the analysis with variations of the database parameters
Author: Vit Kucera <vit.kucera@cern.ch>
"""

import argparse
import subprocess
import shlex
from copy import deepcopy
import datetime
import yaml

def msg_err(message: str):
    '''Print an error message.'''
    print("\x1b[1;31mError: %s\x1b[0m" % message)

def modify_paths(dic: dict, old: str, new: str):
    '''Modify the paths of the results directories'''
    if "analysis" not in dic:
        msg_err("key \"analysis\" not found.")
        return
    for key_a, val_a in dic["analysis"].items():
        if not isinstance(val_a, dict):
            continue
        dic_ana = dic["analysis"][key_a]
        for data in ["data", "mc"]:
            for key_d, val_d in dic_ana[data].items():
                if isinstance(val_d, list):
                    dic_ana[data][key_d] = [v.replace(old, new) \
                        if isinstance(v, str) else v for v in val_d]
                else:
                    dic_ana[data][key_d] = val_d.replace(old, new) \
                        if isinstance(val_d, str) else val_d

def format_value(old, new):
    '''Format the new value based on the format of the old one
    in case they are not of the same type.'''
    if type(old) is type(new):
        return new
    if isinstance(old, list):
        return [new] * len(old) # Return a list of the same length filled with the new values.
    return None

def modify_dictionary(dic: dict, diff: dict):
    '''Modify the dic dictionary using the diff dictionary.'''
    for key, value in diff.items():
        if key in dic: # Do not add keys that are not already in the original dictionary.
            if isinstance(value, dict):
                modify_dictionary(dic[key], value)
            else:
                dic[key] = format_value(dic[key], value)
        else:
            print("\x1b[1;36mWarning:\x1b[0m Key %s was not found and will be ignored." % key)

def healthy_structure(dic_diff: dict): # pylint: disable=too-many-return-statements, too-many-branches
    '''Check correct structure of the variation dictionary.'''
    if not isinstance(dic_diff, dict):
        msg_err("No dictionary found.")
        return False
    if "categories" not in dic_diff:
        msg_err("key \"categories\" not found.")
        return False
    dic_cats = dic_diff["categories"]
    if not isinstance(dic_cats, dict):
        msg_err("\"categories\" is not a dictionary.")
        return False
    # Categories
    for cat in dic_cats:
        dic_cat_single = dic_cats[cat]
        if not isinstance(dic_cat_single, dict):
            msg_err("%s is not a dictionary." % cat)
            return False
        good = True
        for key in ["activate", "label", "variations"]:
            if key not in dic_cat_single:
                msg_err("key \"%s\" not found in %s." % (key, cat))
                good = False
        if not good:
            return False
        dic_vars = dic_cat_single["variations"]
        if not isinstance(dic_vars, dict):
            msg_err("\"variations\" in category %s is not a dictionary." % cat)
            return False
        # Variations
        for var in dic_vars:
            dic_var_single = dic_vars[var]
            if not isinstance(dic_var_single, dict):
                msg_err("%s in %s is not a dictionary." % (var, cat))
                return False
            good = True
            for key in ["activate", "label", "diffs"]:
                if key not in dic_var_single:
                    msg_err("key \"%s\" not found in %s/%s." % (key, cat, var))
                    good = False
            if not good:
                return False
            if not isinstance(dic_var_single["diffs"], dict):
                msg_err("\"diffs\" in %s/%s is not a dictionary." % (cat, var))
                return False
    return True

def main(yaml_in, yaml_diff, analysis): # pylint: disable=too-many-locals
    '''Main function'''
    with open(yaml_in, 'r') as file_in:
        dic_in = yaml.safe_load(file_in)
    with open(yaml_diff, 'r') as file_diff:
        dic_diff = yaml.safe_load(file_diff)

    if not healthy_structure(dic_diff):
        msg_err("Bad structure.")
        return

    #print(yaml.safe_dump(dic_in, default_flow_style=False))

    # Save the original database in the same format as the output for debugging.
    i_dot = yaml_in.rfind(".") # Find the position of the suffix.
    yaml_out = yaml_in[:i_dot] + "_orig" + yaml_in[i_dot:]
    print("\nSaving the original database to %s" % yaml_out)
    with open(yaml_out, 'w') as file_out:
        yaml.safe_dump(dic_in, file_out, default_flow_style=False)

    dic_cats = dic_diff["categories"]
    # Loop over categories.
    for cat in dic_cats:
        dic_cat_single = dic_cats[cat]
        label_cat = dic_cat_single["label"]
        if not dic_cat_single["activate"]:
            print("\nSkipping category %s (%s)" % (cat, label_cat))
            continue
        print("\nProcessing category %s (\x1b[1;34m%s\x1b[0m)" % (cat, label_cat))
        dic_vars = dic_cat_single["variations"]
        # Loop over variations.
        for var in dic_vars:
            dic_var_single = dic_vars[var]
            label_var = dic_var_single["label"]
            if not dic_var_single["activate"]:
                print("\nSkipping variation %s/%s (%s: %s)" % \
                    (cat, var, label_cat, label_var))
                continue
            print("\nProcessing variation %s/%s (\x1b[1;33m%s: %s\x1b[0m)" % \
                (cat, var, label_cat, label_var))

            dic_db = deepcopy(dic_in)
            # Get the database from the first top-level key.
            for k in dic_db:
                dic_new = dic_db[k]
                break

            # Modify the database.
            if not dic_var_single["diffs"]:
                print("\x1b[1;36mWarning:\x1b[0m Empty diffs. No changes to make.")
            modify_dictionary(dic_new, dic_var_single["diffs"])
            modify_paths(dic_new, "default/default", "%s/%s" % (cat, var))

            #print(yaml.safe_dump(dic_db, default_flow_style=False))

            # Save the new database.
            i_dot = yaml_in.rfind(".") # Find the position of the suffix.
            yaml_out = yaml_in[:i_dot] + "_" + cat + "_" + var + yaml_in[i_dot:]
            print("Saving the new database to %s" % yaml_out)
            with open(yaml_out, 'w') as file_out:
                yaml.safe_dump(dic_db, file_out, default_flow_style=False)

            # Start the analysis.
            if analysis:
                print("Starting the analysis \x1b[1;32m%s\x1b[0m for the variation " \
                    "\x1b[1;32m%s: %s\x1b[0m" % (analysis, label_cat, label_var))
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                logfile = "stdouterr_%s_%s_%s_%s.log" % (analysis, cat, var, timestamp)
                print("Logfile: %s" % logfile)
                with open(logfile, "w") as ana_out:
                    subprocess.Popen(shlex.split("python do_entire_analysis.py " \
                        "-r submission/default_complete.yml " \
                        "-d %s -a %s" % (yaml_out, analysis)), \
                        stdout=ana_out, stderr=ana_out, universal_newlines=True)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Run the analysis with " \
                                                 "variations of parameters.")
    PARSER.add_argument("input", help="database with default parameters")
    PARSER.add_argument("diff", help="database with variations")
    PARSER.add_argument("-a", dest="analysis", help="analysis type")
    ARGS = PARSER.parse_args()
    main(ARGS.input, ARGS.diff, ARGS.analysis)
