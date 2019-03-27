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
import os.path
import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ROOT import TFile, TH1F, gROOT # pylint: disable=import-error,no-name-in-module
from ROOT import kRed, kBlue, kFALSE # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.logger import get_logger
from machine_learning_hep.general import filter_df_cand, get_database_ml_parameters
from machine_learning_hep.general import filterdataframe_singlevar

def calc_eff(df_to_sel, sel_opt, main_dict, name, num_step, do_std=False):
    """Calculate the selection efficiency as a function of the treshold on the ML model output.

    It works also for standard selections, setting do_std to True. In this case the same value is
    repeated in the output array to easily plot the sandard selection efficiency together to the
    model one.

    Args:
        df_to_sel: pandas dataframe
        sel_opt: string, according to filter_df_cand options
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml' with case
                   already selected
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
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml' with case
                   already selected
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
        main_dict: dictionary of parameters loaded from 'database_ml_parameters.yml' with case
                   already selected

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

    # efficiency of std selections estimated on the same candidates as ML models
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


# pylint: disable=too-many-statements, too-many-locals
def extract_eff_histo(run_config, data_dict, case, sel_type='ml'):
    """Build a histogram with the ML model selection efficiency for the different transverse
    momentum of the analysis.

    A threshold on the model output is necessary for each p_t bin. The pre-selection efficiency
    times acceptance is taken into account.

    Args:
        run_config: configuration dictionary loaded from 'default_complete.yml'
        data_dict: dictionary of parameters loaded from 'database_ml_parameters.yml'
        case: analysis particle
        sel_type: string to differentiate between 'ml' and 'std' selections
    """
    logger = get_logger()
    gROOT.SetBatch(True)
    gROOT.ProcessLine('gErrorIgnoreLevel = kWarning;')

    bin_min = run_config['analysis']['binmin']
    bin_max = run_config['analysis']['binmax']
    model_name = run_config['analysis']['modelname']
    test_df_list = run_config['analysis']['test_df_list']
    cuts = run_config['analysis']['probcutoptimal']

    data_dict = data_dict[case]
    var_bin = data_dict['variables']['var_binning']
    folder_mc = data_dict['output_folders']['pkl_merged']['mc']
    test_df_dir = data_dict['output_folders']['mlout']
    file_mc_reco = data_dict['files_names']['namefile_reco_merged']
    file_mc_gen = data_dict['files_names']['namefile_gen_merged']

    if len(bin_min) != len(bin_max):
        logger.Critical('Wrong bin limits in default file')
    n_bins = len(bin_min)
    bin_min.append(bin_max[n_bins-1])
    bin_lims = array.array('f', bin_min)

    h_eff_model_prompt = TH1F('hEff_Prompt_Model', ';#it{p}_{T} (GeV/#it{c});Model Efficiency',
                              n_bins, bin_lims)
    h_eff_model_fd = TH1F('hEff_FD_Model', ';#it{p}_{T} (GeV/#it{c});Model Efficiency', n_bins,
                          bin_lims)
    h_effacc_prompt = TH1F('hEffAcc_Prompt_Presel',
                           ';#it{p}_{T} (GeV/#it{c});Pre-Sel Efficiency x Acceptance', n_bins,
                           bin_lims)
    h_effacc_fd = TH1F('hEffAcc_Prompt_FD',
                       ';#it{p}_{T} (GeV/#it{c});Pre-Sel Efficiency x Acceptance', n_bins,
                       bin_lims)

    if sel_type == 'std':
        h_eff_model_prompt.GetYaxis().SetTitle('Std Efficiency')
        h_eff_model_fd.GetYaxis().SetTitle('Std Efficiency')

    df_mc_reco = pd.read_pickle(os.path.join(folder_mc, file_mc_reco))
    df_mc_reco = df_mc_reco.query(data_dict['presel_reco']) #probably not necessary any more
    df_mc_gen = pd.read_pickle(os.path.join(folder_mc, file_mc_gen))
    df_mc_gen = df_mc_gen.query(data_dict['presel_gen'])

    i = 1
    for b_min, b_max in zip(bin_min, bin_max):
        #pre-sel eff x acc
        df_reco_sel = filterdataframe_singlevar(df_mc_reco, var_bin, b_min, b_max)
        df_gen_sel = filterdataframe_singlevar(df_mc_gen, var_bin, b_min, b_max)
        ea_prompt, ea_prompt_err = calc_eff_acc(df_gen_sel, df_reco_sel, 'mc_signal_prompt',
                                                data_dict)
        h_effacc_prompt.SetBinContent(i, ea_prompt)
        h_effacc_prompt.SetBinError(i, ea_prompt_err)
        ea_fd, ea_fd_err = calc_eff_acc(df_gen_sel, df_reco_sel, 'mc_signal_FD', data_dict)
        h_effacc_fd.SetBinContent(i, ea_fd)
        h_effacc_fd.SetBinError(i, ea_fd_err)

        #std selection efficiency
        if sel_type == 'std':
            eff_prompt_std, eff_prompt_std_err = calc_eff_fixed(df_reco_sel, 'mc_signal_prompt',
                                                                data_dict, '', None, True)
            h_eff_model_prompt.SetBinContent(i, eff_prompt_std)
            h_eff_model_prompt.SetBinError(i, eff_prompt_std_err)
            eff_fd_std, eff_fd_std_err = calc_eff_fixed(df_reco_sel, 'mc_signal_FD',
                                                        data_dict, '', None, True)
            h_eff_model_fd.SetBinContent(i, eff_fd_std)
            h_eff_model_fd.SetBinError(i, eff_fd_std_err)

        i += 1

    #model efficiency
    if sel_type == 'ml':
        i = 1
        for thr_value, test_df in zip(cuts, test_df_list):
            df_test_appl = pd.read_pickle(os.path.join(test_df_dir, test_df))
            eff_prompt_ml, eff_prompt_ml_err = calc_eff_fixed(df_test_appl, 'mc_signal_prompt',
                                                              data_dict, model_name, thr_value)
            h_eff_model_prompt.SetBinContent(i, eff_prompt_ml)
            h_eff_model_prompt.SetBinError(i, eff_prompt_ml_err)
            eff_fd_ml, eff_fd_ml_err = calc_eff_fixed(df_test_appl, 'mc_signal_FD',
                                                      data_dict, model_name, thr_value)
            h_eff_model_fd.SetBinContent(i, eff_fd_ml)
            h_eff_model_fd.SetBinError(i, eff_fd_ml_err)
            i += 1

    h_tot_prompt = TH1F('hEffAccPrompt', ';#it{p}_{T} (GeV/#it{c});  Efficiency x Acceptance',
                        n_bins, bin_lims)
    h_effacc_prompt.Sumw2()
    h_eff_model_prompt.Sumw2()
    h_tot_prompt.Multiply(h_effacc_prompt, h_eff_model_prompt, 1., 1.)
    h_tot_fd = TH1F('hEffAccFD', ';#it{p}_{T} (GeV/#it{c});  Efficiency x Acceptance',
                    n_bins, bin_lims)
    h_effacc_fd.Sumw2()
    h_eff_model_fd.Sumw2()
    h_tot_fd.Multiply(h_effacc_fd, h_eff_model_fd, 1., 1.)

    #cosmetics
    h_tot_prompt.SetStats(kFALSE)
    h_tot_prompt.SetMarkerSize(1.)
    h_tot_prompt.SetMarkerStyle(20)
    h_tot_prompt.SetLineWidth(2)
    h_tot_prompt.SetMarkerColor(kRed)
    h_tot_prompt.SetLineColor(kRed)
    h_tot_fd.SetStats(kFALSE)
    h_tot_fd.SetMarkerSize(1.)
    h_tot_fd.SetMarkerStyle(20)
    h_tot_fd.SetLineWidth(2)
    h_tot_fd.SetMarkerColor(kBlue)
    h_tot_fd.SetLineColor(kBlue)
    h_eff_model_prompt.SetStats(kFALSE)
    h_eff_model_prompt.SetMarkerSize(1.)
    h_eff_model_prompt.SetMarkerStyle(20)
    h_eff_model_prompt.SetLineWidth(2)
    h_eff_model_prompt.SetMarkerColor(kRed-7)
    h_eff_model_prompt.SetLineColor(kRed-7)
    h_eff_model_fd.SetStats(kFALSE)
    h_eff_model_fd.SetMarkerSize(1.)
    h_eff_model_fd.SetMarkerStyle(20)
    h_eff_model_fd.SetLineWidth(2)
    h_eff_model_fd.SetMarkerColor(kBlue-7)
    h_eff_model_fd.SetLineColor(kBlue-7)
    h_effacc_prompt.SetStats(kFALSE)
    h_effacc_prompt.SetMarkerSize(1.)
    h_effacc_prompt.SetMarkerStyle(20)
    h_effacc_prompt.SetLineWidth(2)
    h_effacc_prompt.SetMarkerColor(kRed+2)
    h_effacc_prompt.SetLineColor(kRed+2)
    h_effacc_fd.SetStats(kFALSE)
    h_effacc_fd.SetMarkerSize(1.)
    h_effacc_fd.SetMarkerStyle(20)
    h_effacc_fd.SetLineWidth(2)
    h_effacc_fd.SetMarkerColor(kBlue+2)
    h_effacc_fd.SetLineColor(kBlue+2)

    out_file = TFile.Open(f'efficiencies_{case}_{sel_type}.root', 'recreate')
    out_file.cd()
    h_tot_prompt.Write()
    h_tot_fd.Write()
    h_eff_model_prompt.Write()
    h_eff_model_fd.Write()
    h_effacc_prompt.Write()
    h_effacc_fd.Write()
