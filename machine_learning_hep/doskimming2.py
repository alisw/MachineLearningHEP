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
import uproot
from machine_learning_hep.general import get_database_ml_parameters
from machine_learning_hep.listfiles import list_files_dir

def writelist_tofile(fileout, mylist):
    with open(fileout, 'w') as f:
        for item in mylist:
            f.write("%s\n" % item)

def flattenroot_to_pandas(filein, fileout, treenamein, var_all, skimming_sel):
    print(filein)
    tree = uproot.open(filein)[treenamein]
    df = tree.pandas.df(branches=var_all, flatten=True)
    df = df.query(skimming_sel)
    df.to_pickle(fileout)

def flattenallpickle(chunck, chunckout, treenamein, var_all, skimming_sel):
    processes = [mp.Process(target=flattenroot_to_pandas, args=(filein, chunckout[index], \
                                                         treenamein, var_all, skimming_sel))
                 for index, filein in enumerate(chunck)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def doskimming2():

    case = "Dzero"
    namefileinput = 'AnalysisResults.root'
    namefileinputpkl = 'AnalysisResults%s.pkl' % case
    treenamein = 'PWGHF_TreeCreator/tree_D0'
    data = get_database_ml_parameters()
    var_all_unflat = data[case]["var_all_unflat"]
    skimming_sel = "n_cand> 0 & pt_cand>3"
    nmaxchunks = 50
    nmaxfiles = 50

    #inputdir = "/home/ginnocen/LearningPythonML/inputs"
    inputdir = "/data/HeavyFlavour/DmesonsLc_pp_5TeV/Data_LHC17pq/12-02-2019/340_20190211-2126/unmerged" # pylint: disable=line-too-long
    mergeddir = \
    "/data/HeavyFlavour/DmesonsLc_pp_5TeV/Data_LHC17pq/12-02-2019/340_20190211-2126/unmergedpkl" # pylint: disable=line-too-long

    listfilespath, listfilespathout = list_files_dir(inputdir, outdir=mergeddir, \
                                  filenameinput=namefileinput, filenameoutput=namefileinputpkl)
    listfilespath = listfilespath[:nmaxfiles]
    listfilespathout = listfilespathout[:nmaxfiles]
    print(len(listfilespath))
    print(len(listfilespathout))
    chunks = [listfilespath[x:x+nmaxchunks] for x in range(0, len(listfilespath), nmaxchunks)]
    chunksout = [listfilespathout[x:x+nmaxchunks] for x in range(0, len(listfilespathout), nmaxchunks)]
    for chunck, chunckout in zip(chunks, chunksout):
        flattenallpickle(chunck, chunckout, treenamein, var_all_unflat, skimming_sel)

doskimming2()
