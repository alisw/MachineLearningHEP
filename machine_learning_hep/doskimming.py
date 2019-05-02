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
import random
import uproot
import pandas as pd
import numpy as np
#from machine_learning_hep.logger import get_logger
#from machine_learning_hep.general import get_database_ml_parameters, get_database_ml_analysis

from machine_learning_hep.listfiles import list_files_dir_lev2, list_files_lev2
from machine_learning_hep.general import filter_df_cand
from machine_learning_hep.selectionutils import selectfidacc, select_runs

def flattenroot_to_pandas(filein, fileout, treenamein, var_all, skimming_sel, runlist):
    tree = uproot.open(filein)[treenamein]
    df = tree.pandas.df(branches=var_all)
    if skimming_sel is not None:
        df = df.query(skimming_sel)
    if runlist is not None:
        array_run = df.run_number.values
        isgoodrun = select_runs(runlist, array_run)
        df = df[np.array(isgoodrun, dtype=bool)]
    df.to_pickle(fileout)


def skimmer(filein, filevt, fileout, skimming_sel, var_evt_match,
            param_case, presel_reco, sel_cent, skimming2_dotrackpid):
    df = pickle.load(open(filein, "rb"))
    dfevt = pickle.load(open(filevt, "rb"))
    if "Evt" not in filein:
        df = pd.merge(df, dfevt, on=var_evt_match)
    if skimming_sel is not None:
        df = df.query(skimming_sel)
    if "Reco" in filein:
        if skimming2_dotrackpid is True:
            df = filter_df_cand(df, param_case, 'presel_track_pid')
        if presel_reco is not None:
            df = df.query(presel_reco)
        array_pt = df.pt_cand.values
        array_y = df.y_cand.values
        isselacc = selectfidacc(array_pt, array_y)
        df = df[np.array(isselacc, dtype=bool)]
    if sel_cent is not None:
        df = df.query(sel_cent)
    df.to_pickle(fileout)

def flattenallpickle(chunk, chunkout, treenamein, var_all, skimming_sel,
                     runlist):
    processes = [mp.Process(target=flattenroot_to_pandas, args=(filein, chunkout[index], \
                                                         treenamein, var_all, \
                                                         skimming_sel,  runlist))
                 for index, filein in enumerate(chunk)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

def skimall(chunk, chunkevt, chunkout, skimming_sel, var_evt_match,
            param_case, presel_reco, sel_cent, skimming2_dotrackpid):
    processes = [mp.Process(target=skimmer, args=(filein, chunkevt[index],
                                                  chunkout[index],
                                                  skimming_sel, var_evt_match,
                                                  param_case, presel_reco,
                                                  sel_cent,
                                                  skimming2_dotrackpid))
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


def list_create_dir(inputdir, outputdir, namea, nameb, namec,
                    nameaout, namebout, namecout, maxfiles):
    lista, listaout = list_files_dir_lev2(inputdir, outputdir, namea, nameaout)
    listb, listbout = list_files_dir_lev2(inputdir, outputdir, nameb, namebout)
    listc, listcout = list_files_dir_lev2(inputdir, outputdir, namec, namecout)

    if maxfiles is not -1:
        lista = lista[:maxfiles]
        listb = listb[:maxfiles]
        listc = listc[:maxfiles]
        listaout = listaout[:maxfiles]
        listbout = listbout[:maxfiles]
        listcout = listcout[:maxfiles]
    return lista, listb, listc, listaout, listbout, listcout

def createchunks(listin, listout, maxperchunk):
    chunks = [listin[x:x+maxperchunk]  for x in range(0, len(listin), maxperchunk)]
    chunksout = [listout[x:x+maxperchunk] for x in range(0, len(listout), maxperchunk)]
    return chunks, chunksout

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def conversion(data_config, data_param, run_param, mcordata, indexp):

    case = data_config["case"]

    inputdir = data_param[case]["inputs"][mcordata]["unmerged_tree_dir"][indexp]
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

    skimming_sel = data_param[case]["skimming_sel"]
    skimming_sel_gen = data_param[case]["skimming_sel_gen"]
    skimming_sel_evt = data_param[case]["skimming_sel_evt"]

    outputdir = data_param[case]["output_folders"]["pkl_out"][mcordata][indexp]
    maxfiles = data_config["conversion"][mcordata]["maxfiles"]
    nmaxconvers = data_config["conversion"][mcordata]["nmaxconvers"]

    listfilespath, listfilespathevt, listfilespathgen, \
    listfilespathout, listfilespathoutevt, listfilespathoutgen = \
        list_create_dir(inputdir, outputdir, \
                        namefile_unmerged_tree, namefile_unmerged_tree, namefile_unmerged_tree, \
                        namefile_reco, namefile_evt, namefile_gen, maxfiles)
    prod = data_param[case]["inputs"][mcordata]["production"][indexp]
    runlist = run_param[prod]
    tstart = time.time()
    print("I am extracting flat trees")

    chunks, chunksout = createchunks(listfilespath, listfilespathout, nmaxconvers)
    chunksgen, chunksoutgen = createchunks(listfilespathgen, listfilespathoutgen, nmaxconvers)
    chunksevt, chunksoutevt = createchunks(listfilespathevt, listfilespathoutevt, nmaxconvers)
    for index, _ in enumerate(chunks):
        print("Processing chunk number=", index)
        flattenallpickle(chunks[index], chunksout[index], treeoriginreco,
                         var_all, skimming_sel, runlist)
        flattenallpickle(chunksevt[index], chunksoutevt[index], treeoriginevt, \
                         var_evt, skimming_sel_evt, runlist)
        if mcordata == "mc":
            flattenallpickle(chunksgen[index], chunksoutgen[index], treeorigingen, \
                             var_gen, skimming_sel_gen, runlist)
        time.sleep(10)
    print("Total time elapsed", time.time()-tstart)

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def skim(data_config, data_param, mcordata, indexp):

    case = data_config["case"]
    param_case = data_param[case]

    namefile_reco = data_param[case]["files_names"]["namefile_reco"]
    namefile_evt = data_param[case]["files_names"]["namefile_evt"]
    namefile_gen = data_param[case]["files_names"]["namefile_gen"]
    namefile_reco_skim = data_param[case]["files_names"]["namefile_reco_skim"]
    namefile_evt_skim = data_param[case]["files_names"]["namefile_evt_skim"]
    namefile_gen_skim = data_param[case]["files_names"]["namefile_gen_skim"]

    var_evt_match = data_param[case]["variables"]["var_evt_match"]
    skimming_sel = data_param[case]["skimming2_sel"]
    skimming_sel_gen = data_param[case]["skimming2_sel_gen"]
    skimming_sel_evt = data_param[case]["skimming2_sel_evt"]
    presel_reco = data_param[case]["presel_reco"]
    sel_cent = data_param[case]["sel_cent"]
    skimming2_dotrackpid = data_param[case]["skimming2_dotrackpid"]

    inputdir = data_param[case]["output_folders"]["pkl_out"][mcordata][indexp]
    outputdir = data_param[case]["output_folders"]["pkl_skimmed"][mcordata][indexp]
    maxfiles = data_config["skimming"][mcordata]["maxfiles"]
    nmaxconvers = data_config["skimming"][mcordata]["nmaxconvers"]

    listfilespath, listfilespathevt, listfilespathgen, \
    listfilespathout, listfilespathoutevt, listfilespathoutgen = \
        list_create_dir(inputdir, outputdir, namefile_reco, \
                        namefile_evt, namefile_gen, \
                        namefile_reco_skim, namefile_evt_skim, namefile_gen_skim, maxfiles)
    tstart = time.time()
    print("I am skimming")

    chunks, chunksout = createchunks(listfilespath, listfilespathout, nmaxconvers)
    chunksgen, chunksoutgen = createchunks(listfilespathgen, listfilespathoutgen, nmaxconvers)
    chunksevt, chunksoutevt = createchunks(listfilespathevt, listfilespathoutevt, nmaxconvers)

    for index, _ in enumerate(chunks):
        print("Processing chunk number=", index)
        skimall(chunks[index], chunksevt[index], chunksout[index],
                skimming_sel, var_evt_match, param_case, presel_reco, sel_cent,
                skimming2_dotrackpid)
        skimall(chunksevt[index], chunksevt[index], chunksoutevt[index], \
                skimming_sel_evt, var_evt_match, param_case, presel_reco, \
                sel_cent, skimming2_dotrackpid)
        if mcordata == "mc":
            skimall(chunksgen[index], chunksevt[index], chunksoutgen[index], \
                    skimming_sel_gen, var_evt_match, param_case, presel_reco, \
                    sel_cent, skimming2_dotrackpid)
    print("Total time elapsed", time.time()-tstart)
def merging(data_config, data_param, mcordata, indexp):

    case = data_config["case"]
    maxfilestomerge = data_config["merging"][mcordata]["maxfilestomerge"]
    rnd_seed = data_config["merging"][mcordata]["rnd_seed"]

    print("I am merging flat trees")
    namefile_reco = data_param[case]["files_names"]["namefile_reco_skim"]
    namefile_gen = data_param[case]["files_names"]["namefile_gen_skim"]
    namefile_evt = data_param[case]["files_names"]["namefile_evt_skim"]
    namefile_evt_skim_tot = data_param[case]["files_names"]["namefile_evt_skim_tot"]
    namefile_reco_merged = data_param[case]["files_names"]["namefile_reco_merged"]
    namefile_evt_merged = data_param[case]["files_names"]["namefile_evt_merged"]
    namefile_gen_merged = data_param[case]["files_names"]["namefile_gen_merged"]

    outputdir = data_param[case]["output_folders"]["pkl_skimmed"][mcordata][indexp]
    outputdirmerged = data_param[case]["output_folders"]["pkl_merged"][mcordata][indexp]

    listfilespathtomerge, _ = list_files_lev2(outputdir, "", namefile_reco, "")
    listfilespathgentomerge, _ = list_files_lev2(outputdir, "", namefile_gen, "")
    listfilespathevttomerge, _ = list_files_lev2(outputdir, "", namefile_evt, "")
    if mcordata == "mc":
        list_zip = list(zip(listfilespathtomerge, listfilespathgentomerge, listfilespathevttomerge))
        random.seed(rnd_seed)
        random.shuffle(list_zip)
        listfilespathtomerge, listfilespathgentomerge, listfilespathevttomerge = zip(*list_zip)
    else:
        list_zip = list(zip(listfilespathtomerge, listfilespathevttomerge))
        random.seed(rnd_seed)
        random.shuffle(list_zip)
        listfilespathtomerge, listfilespathevttomerge = zip(*list_zip)

    merge(listfilespathevttomerge, os.path.join(outputdirmerged, namefile_evt_skim_tot))
    if maxfilestomerge is not -1:
        listfilespathtomerge = listfilespathtomerge[:maxfilestomerge]
        listfilespathgentomerge = listfilespathgentomerge[:maxfilestomerge]
        listfilespathevttomerge = listfilespathevttomerge[:maxfilestomerge]

    merge(listfilespathtomerge, os.path.join(outputdirmerged, namefile_reco_merged))
    merge(listfilespathevttomerge, os.path.join(outputdirmerged, namefile_evt_merged))
    if mcordata == "mc":
        merge(listfilespathgentomerge, os.path.join(outputdirmerged, namefile_gen_merged))

def merging_period(data_config, data_param, mcordata):
    case = data_config["case"]

    print("I am merging flat trees all periods")

    namefile_reco_merged = data_param[case]["files_names"]["namefile_reco_merged"]
    namefile_evt_merged = data_param[case]["files_names"]["namefile_evt_merged"]
    namefile_gen_merged = data_param[case]["files_names"]["namefile_gen_merged"]
    namefile_evt_skim_tot = data_param[case]["files_names"]["namefile_evt_skim_tot"]

    outputindir_list = data_param[case]["output_folders"]["pkl_merged"][mcordata]
    outputoutdir = data_param[case]["output_folders"]["pkl_merged_all"][mcordata]

    listfilesreco = [os.path.join(dirin, namefile_reco_merged) for dirin in outputindir_list]
    listfilesgen = [os.path.join(dirin, namefile_gen_merged) for dirin in outputindir_list]
    listfilesevt = [os.path.join(dirin, namefile_evt_merged) for dirin in outputindir_list]
    listfilesevttot = [os.path.join(dirin, namefile_evt_skim_tot) for dirin in outputindir_list]

    merge(listfilesreco, os.path.join(outputoutdir, namefile_reco_merged))
    merge(listfilesevt, os.path.join(outputoutdir, namefile_evt_merged))
    merge(listfilesevttot, os.path.join(outputoutdir, namefile_evt_skim_tot))
    if mcordata == "mc":
        merge(listfilesgen, os.path.join(outputoutdir, namefile_gen_merged))
