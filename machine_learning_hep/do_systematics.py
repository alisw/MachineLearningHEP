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

import os
import shutil
import argparse
import subprocess
import shlex
from copy import deepcopy
import datetime
import yaml

def msg_err(message: str):
    '''Print an error message.'''
    print("\x1b[1;31mError: %s\x1b[0m" % message)

def msg_warn(message: str):
    '''Print a warning message.'''
    print("\x1b[1;36mWarning:\x1b[0m %s" % message)

def replace_strings(obj, old: str, new: str, strict=False):
    '''Replace old with new in every string in obj.
    Return None if strict is True and old is not found in every string.'''
    if isinstance(obj, str):
        if strict and old not in obj:
            return None
        return obj.replace(old, new)
    if isinstance(obj, list):
        new_obj = [replace_strings(o, old, new, strict) for o in obj]
        if strict and None in new_obj:
            return None
        return new_obj
    return obj

def modify_paths(dic: dict, old: str, new: str, do_proc: bool):
    '''Modify the paths of the results directories.
    If do_proc is True, modify also paths of the input directories.'''
    strict = True # If True, require old to be in every string.
    if "analysis" not in dic:
        msg_err("key \"analysis\" not found.")
        return False
    for key_a, val_a in dic["analysis"].items():
        if not isinstance(val_a, dict):
            continue
        # Skip non-jet analyses.
        if "jet" not in key_a:
            continue
        dic_ana = dic["analysis"][key_a]
        dirs = ["data", "mc"]
        dirs_proc = ["data_proc", "mc_proc"]
        if do_proc:
            dirs += dirs_proc
        for data in dirs:
            if data in dirs_proc and data not in dic_ana:
                continue
            for key_d, val_d in dic_ana[data].items():
                if "result" not in key_d:
                    continue
                new_val_d = replace_strings(val_d, old, new, strict)
                if new_val_d is None and val_d is not None:
                    msg_err("\"%s\" not found in %s/%s/%s" % (old, key_a, data, key_d))
                    return False
                dic_ana[data][key_d] = new_val_d
    return True

def ask_delete_dir(path: str):
    '''Check whether the directory exists and delete it if the user approves.'''
    if isinstance(path, list):
        for p in path:
            if not ask_delete_dir(p):
                return False
        return True
    if not isinstance(path, str):
        msg_err("Not a string input: %s." % path)
        return False
    if not os.path.isdir(path):
        return True
    msg_warn("This directory already exists:\n%s\nDo you wish to delete it?" % path)
    answers_yes = ["Y", "y", "Yes", "yes"]
    answers_no = ["N", "n", "No", "no"]
    answers_allowed = answers_yes + answers_no
    print("Allowed answers:", *answers_allowed)
    answer = input("Your answer: ")
    while answer not in answers_allowed:
        print("Not a valid answer.")
        answer = input("Your answer: ")
    if answer in answers_yes:
        print("Deleting the directory %s" % path)
        try:
            shutil.rmtree(path)
        except OSError:
            msg_err("Failed to delete the directory %s" % path)
            return False
    return True

def delete_output_dirs(dic: dict, ana: str):
    '''Check whether the output directories exist and delete them if the user approves.'''
    if "analysis" not in dic:
        msg_err("key \"analysis\" not found.")
        return False
    if ana not in dic["analysis"]:
        msg_err("Analysis \"%s\" not found." % ana)
        return False
    dic_ana = dic["analysis"][ana]
    if not isinstance(dic_ana, dict):
        msg_err("key \"%s\" is not a dictionary." % ana)
        return False
    dirs = ["data", "mc"]
    results = ["results", "resultsallp"]
    for data in dirs:
        for res in results:
            if not ask_delete_dir(dic_ana[data][res]):
                return False
    return True

def format_value(old, new):
    '''Format the new value based on the format of the old one.
    If old is a list and new is not, replace all elements in old with new, recursively.
    Keep the old value if new is the special character.'''
    spec_char = "#" # Special character: Value will not be changed.
    if new == spec_char:
        return old
    if type(old) is type(new):
        return new
    if isinstance(old, list):
        # Return a list of the same structure, filled with new.
        return [format_value(old_i, new) for old_i in old]
    msg_warn("Change of type: %s -> %s\n\t%s -> %s" % (type(old), type(new), old, new))
    return new

def format_varname(varname: str, index: int, n_var: int):
    '''Format the name of a variation in a variation group. Used in paths of output directories.'''
    return "%s_%d" % (varname, index) if n_var > 1 else varname

def format_varlabel(varlabel: list, index: int, n_var: int):
    '''Format the label of a variation in a variation group.'''
    return "%s: %d" % (varlabel[0], index) if len(varlabel) != n_var else varlabel[index]

def modify_dictionary(dic: dict, diff: dict):
    '''Modify the dic dictionary using the diff dictionary.'''
    for key, value in diff.items():
        if key in dic: # Do not add keys that are not already in the original dictionary.
            if isinstance(value, dict):
                modify_dictionary(dic[key], value)
            else:
                dic[key] = format_value(dic[key], value)
        else:
            msg_warn("Key %s was not found and will be ignored." % key)

def good_list_length(obj, length: int, name=None):
    '''Check whether all the values are lists of the correct length.'''
    result = True
    #if name:
    #    print(name)
    if isinstance(obj, dict):
        for key in obj:
            newname = "%s/%s" % (name, key) if name else None
            result_this = good_list_length(obj[key], length, newname)
            result = result and result_this
    elif isinstance(obj, list):
        l_obj = len(obj)
        result = bool(l_obj == length)
        if not result:
            msg_err("List%s does not have correct length: %d (expected: %d)." % \
                (" %s" % name if name else "", l_obj, length))
    else:
        msg_err("Object%s is neither a dictionary nor a list." % (" %s" % name if name else ""))
        result = False
    return result

def slice_dic(dic: dict, index: int):
    '''Replace every list in the dictionary with its i-th element'''
    for key, val in dic.items():
        if isinstance(val, list):
            dic[key] = val[index]
        elif isinstance(val, dict):
            slice_dic(val, index)

def healthy_structure(dic_diff: dict): # pylint: disable=too-many-return-statements, too-many-branches
    '''Check correct structure of the variation dictionary.'''
    if not isinstance(dic_diff, dict):
        msg_err("No dictionary found.")
        return False
    if "categories" not in dic_diff:
        msg_err("Key \"categories\" not found.")
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
        for key in ["activate", "label", "variations", "processor"]:
            if key not in dic_cat_single:
                msg_err("Key \"%s\" not found in category %s." % (key, cat))
                good = False
        if not good:
            return False
        if not isinstance(dic_cat_single["activate"], bool):
            msg_err("\"activate\" in category %s is not a boolean." % cat)
            return False
        if not isinstance(dic_cat_single["processor"], bool):
            msg_err("\"processor\" in category %s is not a boolean." % cat)
            return False
        if not isinstance(dic_cat_single["label"], str):
            msg_err("\"label\" in category %s is not a string." % cat)
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
                    msg_err("Key \"%s\" not found in variation group %s/%s." % (key, cat, var))
                    good = False
            if not good:
                return False
            # Activate
            if not isinstance(dic_var_single["activate"], list):
                msg_err("\"activate\" in %s/%s is not a list." % (cat, var))
                return False
            for i, act in enumerate(dic_var_single["activate"]):
                if not isinstance(act, bool):
                    msg_err("Element %d of \"activate\" in %s/%s is not a boolean." % (i, cat, var))
                    return False
            length = len(dic_var_single["activate"])
            # Label
            if not isinstance(dic_var_single["label"], list):
                msg_err("\"label\" in %s/%s is not a list." % (cat, var))
                return False
            len_lab = len(dic_var_single["label"])
            if len_lab not in (length, 1) or len_lab == 0:
                msg_err("\"label\" in %s/%s does not have correct length: %d (expected: 1%s)." % \
                    (cat, var, len_lab, " or %d" % length if length > 1 else ""))
                return False
            for i, lab in enumerate(dic_var_single["label"]):
                if not isinstance(lab, str):
                    msg_err("Element %d of \"label\" in %s/%s is not a string." % (i, cat, var))
                    return False
            # Diffs
            if not isinstance(dic_var_single["diffs"], dict):
                msg_err("\"diffs\" in %s/%s is not a dictionary." % (cat, var))
                return False
            if not good_list_length(dic_var_single["diffs"], length, "diffs"):
                msg_err("\"diffs\" in %s/%s does not contain lists of correct length (%d)." % \
                    (cat, var, length))
                return False
    return True

def main(yaml_in, yaml_diff, analysis, clean, proc): # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    '''Main function'''
    with open(yaml_in, 'r') as file_in:
        dic_in = yaml.safe_load(file_in)
    with open(yaml_diff, 'r') as file_diff:
        dic_diff = yaml.safe_load(file_diff)

    if not healthy_structure(dic_diff):
        msg_err("Bad structure.")
        return

    #print(yaml.safe_dump(dic_in, default_flow_style=False))

    new_files_db = [] # List of created database files.

    # Save the original database in the same format as the output for debugging.
    i_dot = yaml_in.rfind(".") # Find the position of the suffix.
    yaml_out = yaml_in[:i_dot] + "_orig" + yaml_in[i_dot:]
    print("\nSaving the original database to %s" % yaml_out)
    with open(yaml_out, 'w') as file_out:
        yaml.safe_dump(dic_in, file_out, default_flow_style=False)
    new_files_db.append(yaml_out)

    if proc is not None:
        msg_warn("Only categories that DO%s require running the processor will be processed." % \
            ("" if proc else " NOT"))
    dic_cats = dic_diff["categories"]
    # Loop over categories.
    for cat in dic_cats:
        dic_cat_single = dic_cats[cat]
        label_cat = dic_cat_single["label"]
        run = True
        do_processor = dic_cat_single["processor"]
        if proc is not None and bool(proc) is not do_processor:
            run = False
        if not run or not dic_cat_single["activate"]:
            print("\nSkipping category %s (%s)" % (cat, label_cat))
            continue
        print("\nProcessing category %s (\x1b[1;34m%s\x1b[0m)" % (cat, label_cat))
        dic_vars = dic_cat_single["variations"]
        # Loop over variation groups.
        for var in dic_vars:
            dic_var_single = dic_vars[var]
            label_var = dic_var_single["label"]
            n_var = len(dic_var_single["activate"])
            if not n_var:
                print("\nSkipping empty variation group %s/%s (%s: %s)" % \
                    (cat, var, label_cat, label_var[0]))
                continue
            print("\nProcessing variation group %s/%s (%s: %s)" % \
                (cat, var, label_cat, label_var[0] if len(label_var) == 1 else var))
            # Loop over list items.
            for index in range(n_var):

                if not dic_var_single["activate"][index]:
                    print("\nSkipping variation %s/%s (%s: %s)" % \
                        (cat, format_varname(var, index, n_var), \
                        label_cat, format_varlabel(label_var, index, n_var)))
                    continue
                print("\nProcessing variation %s/%s (\x1b[1;33m%s: %s\x1b[0m)" % \
                    (cat, format_varname(var, index, n_var), \
                    label_cat, format_varlabel(label_var, index, n_var)))

                dic_db = deepcopy(dic_in) # Avoid ovewriting the original database.
                # Get the database from the first top-level key.
                for k in dic_db:
                    dic_new = dic_db[k]
                    break

                # Get a slice of diffs at the index.
                # Avoid ovewriting the variation database.
                dic_var_single_slice = deepcopy(dic_var_single["diffs"])
                slice_dic(dic_var_single_slice, index)

                # Modify the database.
                if not modify_paths(dic_new, "default/default", "%s/%s" % \
                    (cat, format_varname(var, index, n_var)), do_processor):
                    return
                if not dic_var_single_slice:
                    msg_warn("Empty diffs. No changes to make.")
                modify_dictionary(dic_new, dic_var_single_slice)

                #print(yaml.safe_dump(dic_db, default_flow_style=False))

                # Save the new database.
                i_dot = yaml_in.rfind(".") # Find the position of the suffix.
                yaml_out = yaml_in[:i_dot] + "_%s_%s" % \
                    (cat, format_varname(var, index, n_var)) + yaml_in[i_dot:]
                print("Saving the new database to %s" % yaml_out)
                with open(yaml_out, 'w') as file_out:
                    yaml.safe_dump(dic_db, file_out, default_flow_style=False)
                new_files_db.append(yaml_out)

                # Start the analysis.
                if analysis:
                    if do_processor and not delete_output_dirs(dic_new, analysis):
                        return
                    mode = "complete" if do_processor else "analyzer"
                    config = "submission/default_%s.yml" % mode
                    print("Starting the analysis \x1b[1;32m%s\x1b[0m for the variation " \
                        "\x1b[1;32m%s: %s\x1b[0m" % \
                        (analysis, label_cat, format_varlabel(label_var, index, n_var)))
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    logfile = "stdouterr_%s_%s_%s_%s.log" % \
                        (analysis, cat, format_varname(var, index, n_var), timestamp)
                    print("Logfile: %s" % logfile)
                    with open(logfile, "w") as ana_out:
                        subprocess.Popen(shlex.split("python do_entire_analysis.py " \
                            "-r %s -d %s -a %s" % (config, yaml_out, analysis)), \
                            stdout=ana_out, stderr=ana_out, universal_newlines=True)

    # Delete the created database files.
    if clean:
        if analysis:
            print("\nSkipping deleting right after starting the analysis.")
        else:
            print("\nDeleting database files:")
            for fil in new_files_db:
                print(fil)
                os.remove(fil)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Run the analysis with " \
                                                 "variations of parameters.")
    PARSER.add_argument("input", help="database with default parameters")
    PARSER.add_argument("diff", help="database with variations")
    PARSER.add_argument("-a", dest="analysis", help="analysis type " \
        "(If provided, the analysis will be started for all activated variations.)")
    PARSER.add_argument("-c", "--clean", action="store_true", \
        help="Delete the created database files at the end.")
    PARSER.add_argument("-p", type=int, choices=[1, 0], dest="proc", help="If 1/0, process only " \
        "categories that do/don't require running the processor.")
    ARGS = PARSER.parse_args()
    main(ARGS.input, ARGS.diff, ARGS.analysis, ARGS.clean, ARGS.proc)
