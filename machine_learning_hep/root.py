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
Methods to: read and write a ROOT TNtuple
"""

import array
import ast
import numpy as np
from ROOT import TNtuple, TFile # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.logger import get_logger


def read_ntuple(ntuple, variables):
    """
      Return a numpy array with the values from TNtuple.
        ntuple : input TNtuple
        variables : list of ntuple variables to read
    """
    logger = get_logger()
    code_list = []
    for v in variables:
        code_list += [compile("i.%s" % v, '<string>', 'eval')]
    nentries = ntuple.GetEntries()
    nvars = len(variables)
    myarray = np.zeros((nentries, nvars))
    for n, _ in enumerate(ntuple):
        for m, v in enumerate(code_list):
            myarray[n][m] = ast.literal_eval(v)
        if n % 100000 == 0:
            logger.info("%d/%d", n, nentries)
    return myarray


def read_ntuple_ml(ntuple, variablesfeatures, variablesothers, variabley):
    """
      Return a numpy array with the values from TNtuple.
        ntuple : input TNtuple
        variables : list of ntuple variables to read
    """
    logger = get_logger()
    code_listfeatures = []
    code_listothers = []
    for v in variablesfeatures:
        code_listfeatures += [compile("i.%s" % v, '<string>', 'eval')]
    for v in variablesothers:
        code_listothers += [compile("i.%s" % v, '<string>', 'eval')]
    codevariabley = compile("i.%s" % variabley, '<string>', 'eval')
    nentries = ntuple.GetEntries()
    nvars = len(variablesfeatures)
    nvarsothers = len(variablesothers)
    arrayfeatures = np.zeros((nentries, nvars))
    arrayothers = np.zeros((nentries, nvarsothers))
    arrayy = np.zeros(nentries)
    for n, _ in enumerate(ntuple):
        for m, v in enumerate(code_listfeatures):
            arrayfeatures[n][m] = ast.literal_eval(v)
        for m, v in enumerate(code_listothers):
            arrayothers[n][m] = ast.literal_eval(v)
        arrayy[n] = ast.literal_eval(codevariabley)
        if n % 100000 == 0:
            logger.info("%d/%d", n, nentries)
    return arrayfeatures, arrayothers, arrayy


def fill_ntuple(tupname, data, names):
    """
      Create and fill ROOT NTuple with the data sample.
        tupname : name of the NTuple
        data : data sample
        names : names of the NTuple variables
    """
    variables = ""
    for n in names:
        variables += "%s:" % n
    variables = variables[:-1]
    values = len(names)*[0.]
    avalues = array.array('f', values)
    nt = TNtuple(tupname, "", variables)
    for d in data:
        for i in range(len(names)):
            avalues[i] = d[i]
        nt.Fill(avalues)
    nt.Write()


def write_tree(filename, treename, dataframe):
    listvar = list(dataframe)
    values = dataframe.values
    fout = TFile.Open(filename, "recreate")
    fout.cd()
    fill_ntuple(treename, values, listvar)
