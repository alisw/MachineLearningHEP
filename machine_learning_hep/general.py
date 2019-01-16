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
Methods to: load and write data to ROOT files
            filter and manipulate pandas DataFrames
"""

import uproot
import yaml
from pkg_resources import resource_stream


def get_database_ml_parameters():
    stream = resource_stream("machine_learning_hep.data", "database_ml_parameters.yml")
    return yaml.safe_load(stream)


def get_database_ml_gridsearch():
    stream = resource_stream("machine_learning_hep.data", "database_ml_gridsearch.yml")
    return yaml.safe_load(stream)


def split_df_sigbkg(dataframe_, var_signal_):
    dataframe_sig_ = dataframe_.loc[dataframe_[var_signal_] == 1]
    dataframe_bkg_ = dataframe_.loc[dataframe_[var_signal_] == 0]
    return dataframe_sig_, dataframe_bkg_


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

def filterdataframe_singlevar(dataframe, var, minval, maxval):
    dataframe = dataframe.loc[(dataframe[var] > minval) & (dataframe[var] < maxval)]
    return dataframe


def filterdataframe(dataframe_, var_list, minlist_, maxlist_):
    dataframe_sel = dataframe_
    for var, low, high in zip(var_list, minlist_, maxlist_):
        dataframe_sel = dataframe_sel.loc[(dataframe_sel[var] > low) & (dataframe_sel[var] < high)]
    return dataframe_sel


def createstringselection(var, low, high):
    string_selection = "dfselection_"+(("%s_%.1f_%.1f") % (var, low, high))
    return string_selection
