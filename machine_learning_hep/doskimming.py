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
#import argparse
#import sys
#import os.path

# pylint: disable=import-error
import sys
import os.path

import multiprocessing as mp
import time
import pickle
import uproot
import pandas as pd
from machine_learning_hep.general import get_database_ml_parameters, get_database_ml_analysis
from machine_learning_hep.listfiles import list_files_dir_lev2, create_subdir_list_lev1

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

def mergeall(chunksmerge, listmerged):
    processes = [mp.Process(target=merge, args=(chunk, merged))
                 for chunk, merged in zip(chunksmerge, listmerged)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doskimming(case, dataset):

    skimconfig = get_database_ml_analysis()
    inputdir = skimconfig[case][dataset]["inputdir_unmerged"]
    mergeddir = skimconfig[case][dataset]["outputdir_skimmer"]
    namefileinput = skimconfig[case][dataset]["namefile_train"]
    mcordata = skimconfig[case][dataset]["data_or_mc"]
    doconversion = skimconfig[case][dataset]["doconversion"]
    domerge = skimconfig[case][dataset]["domerge"]
    nmaxchunks = skimconfig[case][dataset]["nmaxchunks"]
    nmaxfiles = skimconfig[case][dataset]["nmaxfiles"]
    nmaxmerge = skimconfig[case][dataset]["nmaxmerge"]

    namefileinputpklreco = 'AnalysisResultsReco%s.pkl' % case
    namefileinputpklgen = 'AnalysisResultsGen%s.pkl' % case
    namefileinputpklevt = 'AnalysisResultsEvt%s.pkl' % case
    data = get_database_ml_parameters()
    var_all = data[case]["var_all"]
    var_gen = data[case]["var_gen"]
    var_evt = data[case]["var_evt"][mcordata]
    treeoriginreco = data[case]["treeoriginreco"]
    treeorigingen = data[case]["treeorigingen"]
    treeoriginevt = data[case]["treeoriginevt"]
    skimming_sel = data[case]["skimming_sel"]
    skimming_sel_gen = data[case]["skimming_sel_gen"]
    skimming_sel_evt = data[case]["skimming_sel_evt"]
    
    if os.path.exists(mergeddir):
        print("ERROR: merged dir already exists! Change it to avoid deletion of existing files")
        exit()
    else:
        os.mkdir(mergeddir)

    listfilespath, listfilespathout = list_files_dir_lev2(inputdir, mergeddir, \
                                                     namefileinput, namefileinputpklreco)
    listfilespathgen, listfilespathoutgen = list_files_dir_lev2(inputdir, mergeddir, \
                                                     namefileinput, namefileinputpklgen)
    listfilespathevt, listfilespathoutevt = list_files_dir_lev2(inputdir, mergeddir, \
                                                     namefileinput, namefileinputpklevt)

    listfilespath = listfilespath[:nmaxfiles]
    listfilespathout = listfilespathout[:nmaxfiles]
    listfilespathgen = listfilespathgen[:nmaxfiles]
    listfilespathoutgen = listfilespathoutgen[:nmaxfiles]
    listfilespathevt = listfilespathevt[:nmaxfiles]
    listfilespathoutevt = listfilespathoutevt[:nmaxfiles]

    tstart = time.time()
    if doconversion == 1:
        print("I am extracting flat trees")
        chunks = [listfilespath[x:x+nmaxchunks] for x in range(0, len(listfilespath), nmaxchunks)]
        chunksout = [listfilespathout[x:x+nmaxchunks] \
                     for x in range(0, len(listfilespathout), nmaxchunks)]
        i = 0
        for chunk, chunkout in zip(chunks, chunksout):
            print("Processing chunk number=", i, "with n=", len(chunk))
            flattenallpickle(chunk, chunkout, treeoriginreco, var_all, skimming_sel)
            if mcordata == "MC":
                chunkoutgen = [i.replace(namefileinputpklreco, namefileinputpklgen) \
                               for i in chunkout]
                flattenallpickle(chunk, chunkoutgen, treeorigingen,
                                 var_gen, skimming_sel_gen)
            chunkoutevt = [i.replace(namefileinputpklreco, namefileinputpklevt) for i in chunkout]
            convertallpickle(chunk, chunkoutevt, treeoriginevt, var_evt,
                             skimming_sel_evt)
            i = i+1
            print("elapsed time=", time.time()-tstart)
    tstopconv = time.time()
    print("total coversion time", tstopconv - tstart)

    if domerge == 1:
        print("I am merging")
        chunksmerged = [listfilespathout[x:x+nmaxmerge] \
                   for x in range(0, len(listfilespathout), nmaxmerge)]
        chunksmergedevt = [listfilespathoutevt[x:x+nmaxmerge] \
                   for x in range(0, len(listfilespathoutevt), nmaxmerge)]
        nameReco = "AnalysisResults%sMergedReco.pkl" % (case)
        nameEvt = "AnalysisResults%sMergedEvt.pkl" % (case)

        listmerged = create_subdir_list_lev1(mergeddir, len(chunksmerged), nameReco)
        listmergedevt = create_subdir_list_lev1(mergeddir, len(chunksmergedevt), nameEvt)
        print(listmergedevt)
        mergeall(chunksmerged, listmerged)
        mergeall(chunksmergedevt, listmergedevt)
        if mcordata == "MC":
            nameGen = "AnalysisResults%sMergedGen.pkl" % (case)
            listmergedgen = create_subdir_list_lev1(mergeddir, len(chunksmergedgen), nameGen)
            chunksmergedgen = [listfilespathoutgen[x:x+nmaxmerge] \
                for x in range(0, len(listfilespathoutgen), nmaxmerge)]
            mergeall(chunksmergedgen, listmergedgen)

        timemerge = time.time() - tstopconv
        print("total merging time", timemerge)
    print("Total time elapsed", time.time()-tstart)


RUNCASE = sys.argv[1]
RUNDATASET = sys.argv[2]
doskimming(RUNCASE, RUNDATASET)
