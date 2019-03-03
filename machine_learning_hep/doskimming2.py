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

import os
import time
import multiprocessing as mp
import pandas as pd
import uproot
from machine_learning_hep.general import get_database_ml_parameters
#from machine_learning_hep.root import write_tree
from machine_learning_hep.listfiles import list_files

def flattenroot_to_pandas(namefileinput, treenamein, var_all, skimming_sel):
    tree = uproot.open(namefileinput)[treenamein]
    df = tree.pandas.df(branches=var_all, flatten=True)
    df = df.query(skimming_sel)
    return df

def writelist_tofile(fileout, mylist):
    with open(fileout, 'w') as f:
        for item in mylist:
            f.write("%s\n" % item)

def merger_skimmer(listfiles, index, directory, case, treenamein, \
                   var_all, skimming_sel, nmaxfile):
    #dfList = []
    df = flattenroot_to_pandas(listfiles, treenamein, var_all, skimming_sel)
    fileoutpkl = "%s/AnalysisResults%d.pkl" % (directory, index)
    df.to_pickle(fileoutpkl)
    #for namefileinput in listfiles:
    #    df = flattenroot_to_pandas(namefileinput, treenamein, var_all, skimming_sel)
    #    df.to_pickle(namefileinput.replace(".root", ".pkl"))
    #filemergedpkl = "%s/AnalysisResultsMerged%sIndex%d_Nfiles%d.pkl" % \
    #                (directory, case, index, nmaxfile)
    #filelist = "%s/mergedfilesMerged%sIndex%d_Nfiles%d.txt" % (directory, case, index, nmaxfile)
    #writelist_tofile(filelist, listfiles)
    #dfTot = pd.concat(dfList)
    #dfTot.to_pickle(filemergedpkl)

def mergefiles(chunks, mergeddir, case, treenamein, \
               var_all_unflat, skimming_sel, nmaxfile):
    processes = [mp.Process(target=merger_skimmer, args=(mylist, index, mergeddir, case, \
                                                         treenamein, var_all_unflat, \
                                                         skimming_sel, nmaxfile))
                 for index, mylist in enumerate(chunks)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doskimming2():

    case = "Dzero"
    namefileinput = 'AnalysisResults.root'
    treenamein = 'PWGHF_TreeCreator/tree_D0'
    data = get_database_ml_parameters()
    var_all_unflat = data[case]["var_all_unflat"]
    skimming_sel = "n_cand> 0 & pt_cand>3"
    nmaxfile = 50
    nmaxfilestoprocess = 500
    tstart = time.time()

    inputdir = "/home/ginnocen/LearningPythonML/inputs"
    inputdir = "/data/HeavyFlavour/DmesonsLc_pp_5TeV/Data_LHC17pq/12-02-2019/340_20190211-2126/unmerged" # pylint: disable=line-too-long
    mergeddir = "/data/HeavyFlavour/DmesonsLc_pp_5TeV/Data_LHC17pq/12-02-2019/340_20190211-2126/mergedPython" # pylint: disable=line-too-long

    listfilespath, _ = list_files(inputdir, outdir="", \
                                  filenameinput=namefileinput, filenameoutput="")
    listfilespath = listfilespath[:nmaxfilestoprocess]
    print(len(listfilespath))
    chunks = [listfilespath[x:x+nmaxfile] for x in range(0, len(listfilespath), nmaxfile)]
    for i, chunk in enumerate(chunks):
        print("chunk number =", i)
        mergefiles(chunk, mergeddir, case, treenamein, var_all_unflat, skimming_sel, nmaxfile)
    tstop = time.time()
    print("total time =", tstop - tstart)
doskimming2()
