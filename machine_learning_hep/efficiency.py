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
import matplotlib.pyplot as plt
from machine_learning_hep.logger import get_logger
from machine_learning_hep.general import filter_df_cand, get_database_ml_parameters

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


def calc_eff_fixed(df_to_sel, sel_opt, main_dict, name, thr_value, do_std=False):
    """Calculate the selection efficiency of a ML model for a fixed threshold.

    It works also for standard selections, setting do_std to True.

    Args:
        df_to_sel: pandas dataframe
        sel_opt: string, according to filter_df_cand options
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml'
        name: name of the ML model
        thr_value: threshold value on ML model output
        do_std: True for standard selections

    Return:
        eff: efficiency
        eff_err: uncertainty
    """
    df_sig = filter_df_cand(df_to_sel, main_dict, sel_opt)
    num_tot_cand = len(df_sig)

    if do_std:
        num_sel_cand = len(filter_df_cand(df_sig, main_dict, 'sel_std_analysis'))
    else:
        num_sel_cand = len(df_sig[df_sig['y_test_prob' + name].values >= thr_value])

    eff = num_sel_cand / num_tot_cand
    eff_err = np.sqrt(eff * (1 - eff) / num_tot_cand)

    return eff, eff_err


def calc_eff_acc(df_mc_gen, df_mc_reco, sel_opt, main_dict):
    """Calculate the efficiency times acceptance before the ML model selections.

    Args:
        df_mc_gen: pandas dataframe, generated MC
        df_mc_gen: pandas dataframe, reconstructed MC
        sel_opt: string, according to filter_df_cand options
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml'

    Return:
        eff_acc: efficiency times acceptance
        eff_acc_err: uncertainty
    """
    logger = get_logger()
    df_mc_gen_sel = filter_df_cand(df_mc_gen, main_dict, sel_opt)
    df_mc_reco_sel = filter_df_cand(df_mc_reco, main_dict, sel_opt)
    if df_mc_gen_sel.empty:
        logger.error("In division denominator is empty")
        return 0.
    num_tot_cand = len(df_mc_gen_sel)
    eff_acc = len(df_mc_reco_sel) / num_tot_cand
    eff_acc_err = np.sqrt(eff_acc * (1 - eff_acc) / num_tot_cand)
    logger.debug("Pre-selection efficiency times acceptance: %f +/- %f", eff_acc, eff_acc_err)

    return eff_acc, eff_acc_err


def study_eff(case, names, suffix, plot_dir, df_ml_test):
    """Plot the model selection efficiency as function of the threshold value on the model output.

       A comparison with standard selections is also present. It is possible to have the efficieny
       of prompt and feed_down candidates separately, setting split_prompt_FD to True in
       database_ml_parameters.yml.

    Args:
        case: analysis particle
        names: list of ML model names
        suffix: string with run informations
        plot_dir: directory for plots
        df_ml_test: pandas dataframe cointaining test set candidates
    """
    gen_dict = get_database_ml_parameters()[case]
    split_prompt_fd = gen_dict['efficiency']['split_prompt_FD']
    num_step = gen_dict['efficiency']['num_steps']

    fig_eff = plt.figure(figsize=(20, 15))
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('Model Efficiency', fontsize=20)
    plt.title("Efficiency vs Threshold", fontsize=20)

    for name in names:

        if split_prompt_fd:
            eff_prompt, eff_err_prompt, x_axis = calc_eff(df_ml_test, 'mc_signal_prompt', gen_dict,
                                                          name, num_step)
            plt.figure(fig_eff.number)
            plt.errorbar(x_axis, eff_prompt, yerr=eff_err_prompt, alpha=0.3,
                         label=f'{name} - Prompt', elinewidth=2.5, linewidth=4.0)

            eff_fd, eff_err_fd, _ = calc_eff(df_ml_test, 'mc_signal_FD', gen_dict, name, num_step)
            plt.figure(fig_eff.number)
            plt.errorbar(x_axis, eff_fd, yerr=eff_err_fd, alpha=0.3,
                         label=f'{name} - FD', elinewidth=2.5, linewidth=4.0)
        else:
            eff, eff_err, x_axis = calc_eff(df_ml_test, 'mc_signal', gen_dict, name, num_step)
            plt.figure(fig_eff.number)
            plt.errorbar(x_axis, eff, yerr=eff_err, alpha=0.3, label=f'{name}', elinewidth=2.5,
                         linewidth=4.0)

    if split_prompt_fd:
        eff_prompt_std, eff_err_prompt_std, x_axis_std = calc_eff(df_ml_test, 'mc_signal_prompt',
                                                                  gen_dict, '', num_step, True)
        plt.figure(fig_eff.number)
        plt.errorbar(x_axis_std, eff_prompt_std, yerr=eff_err_prompt_std, alpha=0.3,
                     label='ALICE Standard - Prompt', elinewidth=2.5, linewidth=4.0)

        eff_fd_std, eff_err_fd_std, _ = calc_eff(df_ml_test, 'mc_signal_FD', gen_dict, '',
                                                 num_step, True)
        plt.figure(fig_eff.number)
        plt.errorbar(x_axis_std, eff_fd_std, yerr=eff_err_fd_std, alpha=0.3,
                     label='ALICE Standard - FD', elinewidth=2.5, linewidth=4.0)
    else:
        eff_std, eff_err_std, x_axis_std = calc_eff(df_ml_test, 'mc_signal', gen_dict, '',
                                                    num_step, True)
        plt.figure(fig_eff.number)
        plt.errorbar(x_axis_std, eff_std, yerr=eff_err_std, alpha=0.3, label='ALICE Standard',
                     elinewidth=2.5, linewidth=4.0)

    plt.figure(fig_eff.number)
    plt.legend(loc="lower left", prop={'size': 18})
    plt.savefig(plot_dir + '/Efficiency%sSignal.png' % suffix)
