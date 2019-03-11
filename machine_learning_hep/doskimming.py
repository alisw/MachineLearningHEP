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

import multiprocessing as mp
import time
import pickle
import uproot
import pandas as pd
from machine_learning_hep.general import get_database_ml_parameters
from machine_learning_hep.listfiles import list_files_dir

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

def merge(chunk, mergeddir, index, case):
    dfList = []
    for myfilename in chunk:
        myfile = open(myfilename, "rb")
        df = pickle.load(myfile)
        dfList.append(df)
    dftot = pd.concat(dfList)
    namemerged = "%s/AnalysisResults%sMergedIndex%d.pkl" % (mergeddir, case, index)
    dftot.to_pickle(namemerged)

def mergeall(chunksmerge, mergeddir, case):
    processes = [mp.Process(target=merge, args=(chunk, mergeddir, index, case))
                 for index, chunk in enumerate(chunksmerge)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doskimming(case):

    namefileinput = 'AnalysisResults.root'
    namefileinputpklreco = 'AnalysisResultsReco%s.pkl' % case
    namefileinputpklgen = 'AnalysisResultsGen%s.pkl' % case
    namefileinputpklevt = 'AnalysisResultsEvt%s.pkl' % case
    data = get_database_ml_parameters()
    var_all_unflat = data[case]["var_all_unflat"]
    var_gen_unflat = data[case]["var_gen_unflat"]
    var_evt_unflat = data[case]["var_evt_unflat"]
    treeoriginreco = data[case]["treeoriginreco"]
    treeorigingen = data[case]["treeorigingen"]
    treeoriginevt = data[case]["treeoriginevt"]
    skimming_sel = data[case]["skimming_sel"]
    skimming_sel_gen = data[case]["skimming_sel_gen"]
    skimming_sel_evt = data[case]["skimming_sel_evt"]
    nmaxchunks = 1
    nmaxfiles = 10
    nmaxmerge = 10

    doconversion = 1
    domerge = 0
    isMC = 0

    #inputdir = "/home/ginnocen/LearningPythonML/inputs"
    inputdir = "/data/HeavyFlavour/DmesonsLc_pp_5TeV/Data_LHC17pq/12-02-2019/340_20190211-2126/unmerged" # pylint: disable=line-too-long
    mergeddir = "/data/HeavyFlavour/DmesonsLc_pp_5TeV/Data_LHC17pq/12-02-2019/340_20190211-2126/unmergedpkl" # pylint: disable=line-too-long

    listfilespath, listfilespathout = list_files_dir(inputdir, outdir=mergeddir, filenameinput=namefileinput, filenameoutput=namefileinputpklreco)

    listfilespath = listfilespath[:nmaxfiles]
    listfilespathout = listfilespathout[:nmaxfiles]

    tstart = time.time()
    if doconversion == 1:
        print("I am extracting flat trees")
        chunks = [listfilespath[x:x+nmaxchunks] for x in range(0, len(listfilespath), nmaxchunks)]
        chunksout = [listfilespathout[x:x+nmaxchunks] \
                     for x in range(0, len(listfilespathout), nmaxchunks)]
        i = 0
        for chunk, chunkout in zip(chunks, chunksout):
            print("Processing chunk number=", i, "with n=", len(chunk))
            flattenallpickle(chunk, chunkout, treeoriginreco, var_all_unflat, skimming_sel)
            if isMC == 1:
                chunkoutgen = [i.replace(namefileinputpklreco, namefileinputpklgen) for i in chunkout]
                flattenallpickle(chunk, chunkoutgen, treeorigingen,
                                 var_gen_unflat, skimming_sel_gen)
            chunkoutevt = [i.replace(namefileinputpklreco, namefileinputpklevt) for i in chunkout]
            convertallpickle(chunk, chunkoutevt, treeoriginevt, var_evt_unflat,
                             skimming_sel_evt)
            i = i+1
            print("elapsed time=", time.time()-tstart)
        tstopconv = time.time()
        print("total coversion time", tstopconv - tstart)

    if domerge == 1:
        print("I am merging")
        chunksmerge = [listfilespathout[x:x+nmaxmerge] \
                   for x in range(0, len(listfilespathout), nmaxmerge)]
        mergeall(chunksmerge, mergeddir, case)
        timemerge = time.time() - tstopconv
        print("total merging time", timemerge)
    print("Total time elapsed", time.time()-tstart)
runcase = "Dzero" # pylint: disable=invalid-name
doskimming(runcase)
