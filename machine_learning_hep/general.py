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
from machine_learning_hep.logger import get_logger
from machine_learning_hep.bitwise import filter_bit_df


def get_database_ml_analysis():
    stream = resource_stream("machine_learning_hep.data", "database_ml_analysis.yml")
    return yaml.safe_load(stream)

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


# pylint: disable=too-many-statements, too-many-branches
def filter_df_cand(dataframe, main_dict, sel_opt, mc_gen=False):
    '''Filter a dataframe looking at the type of candidate.

    It works both for bitmap and old selection method.
    In 'database_ml_parameters.yml' only one between old_sel and bitmap_sel must have 'use: True'

    Implemented selection options:
        - 'mc_signal' -> select MC signal
        - 'mc_signal_prompt' -> select only prompt MC signal
        - 'mc_signal_FD' -> select only feed-down MC signal
        - 'mc_bkg' -> select MC background (reco MC only)
        - 'presel_track_pid' -> select candidates satisfing PID and track pre-selections
                                (reco MC only)
        - 'sel_std_analysis' -> select candidates fulfilling the std analysis selections
                                (reco MC only)

    Args:
        dataframe: pandas dataframe to filter
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml'
        sel_opt: selection option (string)
        mc_gen: flag to distinguish reconstructed and generated MC

    Return:
        df_selected: filtered pandas dataframe
    '''
    logger = get_logger()

    bitmap_dict = main_dict['bitmapsel'] # var name to change
    old_dict = main_dict['old_sel']
    use_bitmap = bitmap_dict['use']
    use_old = old_dict['use']

    if use_bitmap == use_old:
        logger.critical("One and only one of the selection method have to be used, i.e. with "
                        "'use' flag set to True")

    if use_bitmap:
        logger.debug("Using bitmap selection")

        if mc_gen:
            var_name = main_dict['bitselvariable_gen'] # to change
        else:
            var_name = main_dict['bitselvariable'] # to change

        if sel_opt == 'mc_signal':
            sel_bits = bitmap_dict['mcsignal_on_off']
        elif sel_opt == 'mc_signal_prompt':
            sel_bits = bitmap_dict['mcsignal_prompt_on_off']
        elif sel_opt == 'mc_signal_FD':
            sel_bits = bitmap_dict['mcsignal_feed_on_off']
        elif sel_opt == 'mc_bkg' and not mc_gen:
            sel_bits = bitmap_dict['mcbkg_on_off']
        elif sel_opt == 'presel_track_pid' and not mc_gen:
            sel_bits = bitmap_dict['preseltrack_pid_on_off']
        elif sel_opt == 'sel_std_analysis' and not mc_gen:
            sel_bits = bitmap_dict['std_analysis_on_off']
        else:
            logger.critical("Wrong selection option!")

        df_selected = filter_bit_df(dataframe, var_name, sel_bits)

    if use_old:
        logger.debug("Using old selection")

        if sel_opt == 'mc_signal':
            if mc_gen:
                sel_string = old_dict['mc_gen_signal']
            else:
                sel_string = old_dict['mc_signal']
        elif sel_opt == 'mc_signal_prompt':
            if mc_gen:
                sel_string = old_dict['mc_gen_signal_prompt']
            else:
                sel_string = old_dict['mc_signal_prompt']
        elif sel_opt == 'mc_signal_FD':
            if mc_gen:
                sel_string = old_dict['mc_gen_signal_FD']
            else:
                sel_string = old_dict['mc_signal_FD']
        elif sel_opt == 'mc_bkg' and not mc_gen:
            sel_string = old_dict['mc_bkg']
        elif sel_opt == 'presel_track_pid'and not mc_gen:
            sel_string = old_dict['presel_track_pid']
        elif sel_opt == 'sel_std_analysis' and not mc_gen:
            sel_string = old_dict['sel_std_analysis']
        else:
            logger.critical("Wrong selection option!")

        df_selected = dataframe.query(sel_string)

    return df_selected
