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

import multiprocessing as mp
from glob import glob
import pickle
from lz4 import frame # pylint: disable=unused-import

import yaml

from machine_learning_hep.utilities import openfile
from machine_learning_hep.io import dump_yaml_from_dict
from machine_learning_hep.do_variations import modify_dictionary


def read_database(path, overwrite_path=None):
    data_param = None
    with open(path, 'r') as param_config:
        data_param = yaml.load(param_config, Loader=yaml.FullLoader)
    case = list(data_param.keys())[0]
    data_param = data_param[case]
    if overwrite_path:
        overwrite_db = None
        with open(overwrite_path, 'r') as param_config:
            overwrite_db = yaml.load(param_config, Loader=yaml.FullLoader)
        modify_dictionary(data_param, overwrite_db)
    return case, data_param

def _callback(exept_msg):
    print(exept_msg)

def multi_proc(function, argument_list, kw_argument_list, maxperchunk, max_n_procs=10):
    chunks_args = [argument_list[x:x+maxperchunk] \
              for x in range(0, len(argument_list), maxperchunk)]
    if not kw_argument_list:
        kw_argument_list = [{} for _ in argument_list]
    chunks_kwargs = [kw_argument_list[x:x+maxperchunk] \
              for x in range(0, len(kw_argument_list), maxperchunk)]
    res_list = []
    for chunk_args, chunk_kwargs in zip(chunks_args, chunks_kwargs):
        print("Processing new chunck size=", maxperchunk)
        pool = mp.Pool(max_n_procs)
        res = [pool.apply_async(function, args=args, kwds=kwds, error_callback=_callback) \
                for args, kwds in zip(chunk_args, chunk_kwargs)]
        pool.close()
        pool.join()
        res_list.extend(res)

    try:
        res_list = [r.get() for r in res_list]
    except Exception as e: # pylint: disable=broad-except
        print("EXCEPTION")
        print(e)
    return res_list


def check_duplicates(file_path, cols):
    """Open dataframe and check for duplicates
    """

    df = pickle.load(openfile(file_path, "rb"))[cols]
    len_orig = len(df)
    df_dupl = df[df.duplicated()]
    len_dupl = len(df_dupl)

    return len_orig, len_dupl, df_dupl

###########################
#          MAIN           #
###########################

# BASICALLY THESE HAVE TO BE ADJUSTED
DATABASE_PATH = "/home/bvolkel/HF/MachineLearningHEP/machine_learning_hep/data/data_prod_20200304/database_ml_parameters_LcpK0spp_0304.yml" # pylint: disable=line-too-long

# Summary YAML will be written to this one
# Check "has_duplicates" to find all files with duplictates and the dupl/all ratio
SUMMARY_FILE = "duplicates_summary.yaml"

# To actually extract the duplicated ev_id, ev_id_ext and run_number columns
EXTRACT_DUPL_INFO = True

# Columns in which the dataframe SHOULD be unique. You can add any other event variable which
# present in the event pkl AnalysisResultsEvtOrig
UNIQUE_COLS = ["ev_id", "ev_id_ext", "run_number"]

# Run over mc and/or data, like this automatically over data and MC
DATA_MC = ("mc",) # "data") # ("mc",)  ("data",)


#################################
# Nothing to be done below that #
#################################


_, DATABASE = read_database(DATABASE_PATH)
FILE_NAME = DATABASE["files_names"]["namefile_evtorig"]



DUPLICATES_SUMMARY = {}

for dm in DATA_MC:
    DUPLICATES_SUMMARY[dm] = {}
    for period, dir_applied in zip(DATABASE["multi"][dm]["period"],
                                   DATABASE["multi"][dm]["pkl"]):
        print(f"Process {dm} of period {period}")
        DUPLICATES_SUMMARY[dm][period] = {}
        files_all = glob(f"{dir_applied}/**/{FILE_NAME}", recursive=True)
        children = []
        for d in files_all:
            pos_child = d.find("child_")
            pos_end_child = d.find("/", pos_child)
            child = d[pos_child:pos_end_child]
            if child not in children:
                children.append(child)

        for child in children:
            files_child = [f for f in files_all if f"/{child}/" in f]
            args = []
            for f in  files_child:
                args.append((f, UNIQUE_COLS))

            duplicates = multi_proc(check_duplicates, args, None, 500, 40)
            duplicates_ratio = [d[1] / d[0] * 100 if d[0] > 0 else 0. for d in duplicates]

            if EXTRACT_DUPL_INFO:
                duplicates_cols = []
                for d in duplicates:
                    duplicates_cols_this_df = []
                    for _, row in d[2].iterrows():
                        duplicates_cols_this_df.append([float(row[col_name]) \
                                for col_name in UNIQUE_COLS])
                    duplicates_cols.append(duplicates_cols_this_df)
            else:
                duplicates_cols = [None] * len(duplicates)

            has_duplicates = [dr > 0. for dr in duplicates_ratio]
            DUPLICATES_SUMMARY[dm][period][child] = \
                    [{"file": df, "dupl_ratio": dr, "has_duplicates": hd, "duplicates": dc} \
                    for df, dr, hd, dc \
                    in zip(files_child, duplicates_ratio, has_duplicates, duplicates_cols)]


dump_yaml_from_dict(DUPLICATES_SUMMARY, SUMMARY_FILE)
