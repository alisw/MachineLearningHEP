###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to: load and write data to ROOT files
            filter and manipulate pandas DataFrames
"""

import os
import uproot
import yaml
from pkg_resources import resource_stream
from ROOT import TFile  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.root import fill_ntuple

def get_database_ml_parameters():
    stream = resource_stream("machine_learning_hep.data", "database_ml_parameters.yml")
    return yaml.safe_load(stream)

def split_df_sigbkg(dataframe_, var_signal_):
    dataframe_sig_ = dataframe_.loc[dataframe_[var_signal_] == 1]
    dataframe_bkg_ = dataframe_.loc[dataframe_[var_signal_] == 0]
    return dataframe_sig_, dataframe_bkg_

def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preparestringforuproot(myarray):
    arrayfinal = []
    for itm in myarray:
        arrayfinal.append(itm+"*")
    return arrayfinal

def select_df_ontag(df_tosel, tag, value):
    df_tosel = df_tosel.loc[(df_tosel[tag] == value)]
    return df_tosel

def getdataframe(filename, treename, variables):
    file = uproot.open(filename)
    tree = file[treename]
    dataframe = tree.pandas.df(preparestringforuproot(variables))
    return dataframe

def filterdataframe(dataframe_, var_list, minlist_, maxlist_):
    dataframe_sel = dataframe_
    for var, low, high in zip(var_list, minlist_, maxlist_):
        dataframe_sel = dataframe_sel.loc[(dataframe_sel[var] > low) & (dataframe_sel[var] < high)]
    return dataframe_sel

def createstringselection(var_skimming_, minlist_, maxlist_):
    string_selection = "dfselection_"
    for var, low, high in zip(var_skimming_, minlist_, maxlist_):
        string_selection = string_selection+(("%s_%.1f_%.1f") % (var, low, high))
    return string_selection

def write_tree(filename, treename, dataframe):
    listvar = list(dataframe)
    values = dataframe.values
    TFile.Open(filename, "recreate")
    fill_ntuple(treename, values, listvar)
