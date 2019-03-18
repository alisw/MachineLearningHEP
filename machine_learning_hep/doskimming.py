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
main macro for charm analysis with python
"""
import os
import multiprocessing as mp
import time
import pickle
import uproot
import pandas as pd
#from machine_learning_hep.logger import get_logger
#from machine_learning_hep.general import get_database_ml_parameters, get_database_ml_analysis
from machine_learning_hep.listfiles import list_files_dir_lev2, list_files_lev2

def writelist_tofile(fileout, mylist):
    with open(fileout, 'w') as f:
        for item in mylist:
            f.write("%s\n" % item)

def flattenroot_to_pandas(filein, fileout, treenamein, var_all, skimming_sel):
    tree = uproot.open(filein)[treenamein]
    df = tree.pandas.df(branches=var_all, flatten=True)
    df = df.query(skimming_sel)
    df.to_pickle(fileout)

def convert_to_pandas(filein, fileout, treenamein, var_all, skimming_sel):
    tree = uproot.open(filein)[treenamein]
    df = tree.pandas.df(branches=var_all)
    df = df.query(skimming_sel)
    df.to_pickle(fileout)

def flattenallpickle(chunk, chunkout, treenamein, var_all, skimming_sel):
    processes = [mp.Process(target=flattenroot_to_pandas, args=(filein, chunkout[index], \
                                                         treenamein, var_all, skimming_sel))
                 for index, filein in enumerate(chunk)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def convertallpickle(chunk, chunkout, treenamein, var_all, skimming_sel):
    processes = [mp.Process(target=convert_to_pandas, args=(filein, chunkout[index], \
                                                         treenamein, var_all, skimming_sel))
                 for index, filein in enumerate(chunk)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def merge(chunk, namemerged):
    dfList = []
    for myfilename in chunk:
        myfile = open(myfilename, "rb")
        df = pickle.load(myfile)
        dfList.append(df)
    dftot = pd.concat(dfList)
    dftot.to_pickle(namemerged)

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def conversion(data_config, data_param, mcordata):

    case = data_config["case"]

    inputdir = data_param[case]["inputs"][mcordata]["unmerged_tree_dir"]
    namefile_unmerged_tree = data_param[case]["files_names"]["namefile_unmerged_tree"]

    namefile_reco = data_param[case]["files_names"]["namefile_reco"]
    namefile_evt = data_param[case]["files_names"]["namefile_evt"]
    namefile_gen = data_param[case]["files_names"]["namefile_gen"]
    treeoriginreco = data_param[case]["files_names"]["treeoriginreco"]
    treeorigingen = data_param[case]["files_names"]["treeorigingen"]
    treeoriginevt = data_param[case]["files_names"]["treeoriginevt"]

    var_all = data_param[case]["variables"]["var_all"]
    var_gen = data_param[case]["variables"]["var_gen"]
    var_evt = data_param[case]["variables"]["var_evt"][mcordata]

    skimming_sel = data_param[case]["selections"]["skim"]["skimming_sel"]
    skimming_sel_gen = data_param[case]["selections"]["skim"]["skimming_sel_gen"]
    skimming_sel_evt = data_param[case]["selections"]["skim"]["skimming_sel_evt"]

    outputdir = data_param[case]["output_folders"]["pkl_out"][mcordata]
    maxfiles = data_config["conversion"][mcordata]["maxfiles"]
    nmaxconvers = data_config["conversion"][mcordata]["nmaxconvers"]

    listfilespath, listfilespathout = list_files_dir_lev2(inputdir, outputdir, \
                                                     namefile_unmerged_tree, namefile_reco)
    listfilespathgen, listfilespathoutgen = list_files_dir_lev2(inputdir, outputdir, \
                                                     namefile_unmerged_tree, namefile_gen)
    listfilespathevt, listfilespathoutevt = list_files_dir_lev2(inputdir, outputdir, \
                                                     namefile_unmerged_tree, namefile_evt)
    print(listfilespathout)
    print(inputdir)
    print(namefile_unmerged_tree)

    if maxfiles is not -1:
        listfilespath = listfilespath[:maxfiles]
        listfilespathout = listfilespathout[:maxfiles]
        listfilespathgen = listfilespathgen[:maxfiles]
        listfilespathoutgen = listfilespathoutgen[:maxfiles]
        listfilespathevt = listfilespathevt[:maxfiles]
        listfilespathoutevt = listfilespathoutevt[:maxfiles]

    tstart = time.time()
    print("I am extracting flat trees")
    chunks = [listfilespath[x:x+nmaxconvers] for x in range(0, len(listfilespath), nmaxconvers)]
    chunksout = [listfilespathout[x:x+nmaxconvers] \
                 for x in range(0, len(listfilespathout), nmaxconvers)]
    i = 0
    for chunk, chunkout in zip(chunks, chunksout):
        print("Processing chunk number=", i, "with n=", len(chunk))
        flattenallpickle(chunk, chunkout, treeoriginreco, var_all, skimming_sel)
        if mcordata == "mc":
            chunkoutgen = [i.replace(namefile_reco, namefile_gen) \
                           for i in chunkout]
            flattenallpickle(chunk, chunkoutgen, treeorigingen,
                             var_gen, skimming_sel_gen)
        chunkoutevt = [i.replace(namefile_reco, namefile_evt) for i in chunkout]
        convertallpickle(chunk, chunkoutevt, treeoriginevt, var_evt,
                         skimming_sel_evt)
        i = i+1
    print("Total time elapsed", time.time()-tstart)

def merging(data_config, data_param, mcordata):

    case = data_config["case"]
    maxfilestomerge = data_config["merging"][mcordata]["maxfilestomerge"]

    print("I am merging flat trees")
    namefile_reco = data_param[case]["files_names"]["namefile_reco"]
    namefile_gen = data_param[case]["files_names"]["namefile_gen"]
    namefile_evt = data_param[case]["files_names"]["namefile_evt"]
    namefile_reco_merged = data_param[case]["files_names"]["namefile_reco_merged"]
    namefile_evt_merged = data_param[case]["files_names"]["namefile_evt_merged"]
    namefile_gen_merged = data_param[case]["files_names"]["namefile_gen_merged"]

    outputdir = data_param[case]["output_folders"]["pkl_out"][mcordata]
    outputdirmerged = data_param[case]["output_folders"]["pkl_merged"][mcordata]

    listfilespathtomerge, _ = list_files_lev2(outputdir, "", namefile_reco, "")
    listfilespathgentomerge, _ = list_files_lev2(outputdir, "", namefile_gen, "")
    listfilespathevttomerge, _ = list_files_lev2(outputdir, "", namefile_evt, "")
    if maxfilestomerge is not -1:
        listfilespathtomerge = listfilespathtomerge[:maxfilestomerge]
        listfilespathgentomerge = listfilespathgentomerge[:maxfilestomerge]
        listfilespathevttomerge = listfilespathevttomerge[:maxfilestomerge]
    print(listfilespathtomerge)
    print(listfilespathevttomerge)
    print(listfilespathgentomerge)

    merge(listfilespathtomerge, os.path.join(outputdirmerged, namefile_reco_merged))
    merge(listfilespathevttomerge, os.path.join(outputdirmerged, namefile_evt_merged))
    if mcordata == "mc":
        merge(listfilespathgentomerge, os.path.join(outputdirmerged, namefile_gen_merged))
