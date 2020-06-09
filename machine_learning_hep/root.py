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
from ROOT import gROOT, TNtuple, TFile # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.logger import get_logger


META_INFO = "struct MLHEPMetaInfo { \
               Float_t firstLow; \
               Float_t firstUp; \
               Float_t secondLow; \
               Float_t secondUp; \
               Float_t MLWorkingPoint; \
               std::string firstBinName; \
               std::string secondBinName; \
             };"
gROOT.ProcessLine(META_INFO)

META_INFO_NAME = "MLHEPMetaInfo"

from ROOT import MLHEPMetaInfo # pylint: disable=wrong-import-position, import-error, no-name-in-module, ungrouped-imports

def create_meta_info(first_name, first_low, first_up, second_name, second_low, second_up, ml_wp):
    """Fill MLHEPMetaInfo struct

    Custom MLHEP ROOT struct to store meta info

    Args:
        first_name: str
            name of first binning variable
        first_low: float
            low bin value of first variable
        first_up: float
            up bin value of first variable
        second_name: str
            name of second binning variable
        second_low: float
            low bin value of second variable
        second_up: float
            up bin value of second variable
        ml_wp: float
            ML working point used to cut

    Returns:
        MLHEPMetaInfo
    """

    meta_info = MLHEPMetaInfo()
    meta_info.firstBinName = first_name
    meta_info.firstLow = first_low
    meta_info.firstUp = first_up
    meta_info.secondBinName = second_name
    meta_info.secondLow = second_low
    meta_info.secondUp = second_up
    meta_info.MLWorkingPoint = ml_wp
    return meta_info


def write_meta_info(root_dir, meta_info):
    """Write MLHEPMetaInfo to ROOT directory

    Args:
        root_dir: inheriting from TDirectory
            ROOT directory where to write
        meta_info: MLHEPMetaInfo
            the meta info to be written
    """
    root_dir.WriteObject(meta_info, META_INFO_NAME)


def read_meta_info(root_dir, fail_not_found=True):
    """Read MLHEPMetaInfo

    Args:
        root_dir: inheriting from TDirectory
            ROOT directory where to read from
        fail_not_found: bool
            if True fail if not found
    Returns:
        MLHEPMetaInfo
    """


    meta_info = root_dir.Get(META_INFO_NAME)
    if not meta_info and fail_not_found:
        get_logger().fatal("Cannot find %s in directory %s", META_INFO_NAME, root_dir.GetName())
    return meta_info


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


def save_root_object(obj, path, name=None, extension="pdf"):
    """
    Function to save a root object in path with a defined extension
    If no name is give, the name of the object is taken as output.
        obj : object to save
        path : path to save the object in
        name : name of the output file
        extension : extension of the output file (e.g. pdf, png, eps)
    """
    name = name if name is not None else obj.GetName()
    obj.SaveAs(f"{path}/{name}.{extension}")
