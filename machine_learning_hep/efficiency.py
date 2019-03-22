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
Methods to: study expected efficiency
"""
import numpy as np
from machine_learning_hep.logger import get_logger
from machine_learning_hep.general import filter_df_cand

def calc_eff(df_to_sel, sel_opt, main_dict, name, num_step, do_std=False):
    """Calculate the selection efficiency as a function of the treshold on the ML model output.

    It works also for standard selections, setting do_std to True. In this case the same value is
    repeated in the output array to easily plot the sandard selection efficiency together to the
    model one.

    Args:
        df_to_sel: pandas dataframe
        sel_opt: string, according to filter_df_cand options
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml'
        name: name of the ML model
        num_step: number of divisions on the model prediction output
        do_std: True for standard selections

    Return:
        eff_array: array of efficiency as a function of the threshold on the model output
        eff_err_array: array of uncertainties
        x_axis: array of threshold values
    """
    df_sig = filter_df_cand(df_to_sel, main_dict, sel_opt)
    x_axis = np.linspace(0, 1.00, num_step)
    num_tot_cand = len(df_sig)
    eff_array = []
    eff_err_array = []

    if not do_std:
        for thr in x_axis:
            num_sel_cand = len(df_sig[df_sig['y_test_prob' + name].values >= thr])
            eff = num_sel_cand / num_tot_cand
            eff_err = np.sqrt(eff * (1 - eff) / num_tot_cand)
            eff_array.append(eff)
            eff_err_array.append(eff_err)
    else:
        num_sel_cand = len(filter_df_cand(df_sig, main_dict, 'sel_std_analysis'))
        eff = num_sel_cand / num_tot_cand
        eff_err = np.sqrt(eff * (1 - eff) / num_tot_cand)
        eff_array = [eff] * num_step
        eff_err_array = [eff_err] * num_step

    return eff_array, eff_err_array, x_axis


def calc_eff_acc(df_mc_gen, df_mc_reco, sel_opt, main_dict):
    """Calculate the efficiency times acceptance before the ML model selections.

     Args:
        df_mc_gen: pandas dataframe, generated MC
        df_mc_gen: pandas dataframe, reconstructed MC
        sel_opt: string, according to filter_df_cand options
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml'

    Return:
        eff_acc: efficiency times acceptance
        err_eff_acc: uncertainty
    """
    logger = get_logger()
    df_mc_gen_sel = filter_df_cand(df_mc_gen, main_dict, sel_opt)
    df_mc_reco_sel = filter_df_cand(df_mc_reco, main_dict, sel_opt)
    if df_mc_gen_sel.empty:
        logger.error("In division denominator is empty")
        return 0.
    num_tot_cand = len(df_mc_gen_sel)
    eff_acc = len(df_mc_reco_sel) / num_tot_cand
    err_eff_acc = np.sqrt(eff_acc * (1 - eff_acc) / num_tot_cand)
    logger.debug("Pre-selection efficiency times acceptance: %f +/- %f", eff_acc, err_eff_acc)

    return eff_acc, err_eff_acc
