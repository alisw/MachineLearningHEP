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
Methods to: study selection efficiency and expected significance
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ROOT import TH1F, TF1, gROOT  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.logger import get_logger
from machine_learning_hep.general import getdataframe, filterdataframe_singlevar
from machine_learning_hep.general import get_database_ml_parameters

def calc_efficiency(df_to_sel, sel_signal, name, num_step, sel_signal_std=None):
    """
    Calculate the ML selection efficiency as a function of the treshold on the
    ML model output.
    """
    df_to_sel = df_to_sel.query(sel_signal)
    x_axis = np.linspace(0, 1.00, num_step)
    num_tot_cand = len(df_to_sel)
    eff_array = []
    eff_err_array = []

    if sel_signal_std is None:
        for thr in x_axis:
            num_sel_cand = len(df_to_sel[df_to_sel['y_test_prob' + name].values >= thr])
            eff = num_sel_cand / num_tot_cand
            eff_err = np.sqrt(eff * (1 - eff) / num_tot_cand)
            eff_array.append(eff)
            eff_err_array.append(eff_err)
    else:
        num_sel_cand = len(df_to_sel.query(sel_signal_std))
        eff = num_sel_cand / num_tot_cand
        eff_err = np.sqrt(eff * (1 - eff) / num_tot_cand)
        eff_array = [eff] * num_step
        eff_err_array = [eff_err] * num_step

    return eff_array, eff_err_array, x_axis

def calc_bkg(df_bkg, name, num_step, fit_region, bin_width, sig_region):
    """
    Estimate the number of background candidates under the signal peak. This is obtained
    from real data with a fit of the sidebands of the invariant mass distribution.
    """
    logger = get_logger()
    x_axis = np.linspace(0, 1.00, num_step)
    bkg_array = []
    bkg_err_array = []
    num_bins = (fit_region[1] - fit_region[0]) / bin_width
    num_bins = int(round(num_bins))
    bin_width = (fit_region[1] - fit_region[0]) / num_bins

    logger.debug("To fit the bkg an exponential function is used")
    for thr in x_axis:
        bkg = 0.
        bkg_err = 0.
        hmass = TH1F('hmass', '', num_bins, fit_region[0], fit_region[1])
        bkg_sel_mask = df_bkg['y_test_prob' + name].values >= thr
        sel_mass_array = df_bkg[bkg_sel_mask]['inv_mass_ML'].values

        if len(sel_mass_array) > 5:
            for mass_value in np.nditer(sel_mass_array):
                hmass.Fill(mass_value)

            fit = hmass.Fit('expo', 'Q', '', fit_region[0], fit_region[1])
            if int(fit) == 0:
                fit_func = hmass.GetFunction('expo')
                bkg = fit_func.Integral(sig_region[0], sig_region[1]) / bin_width
                bkg_err = fit_func.IntegralError(sig_region[0], sig_region[1]) / bin_width
                del fit_func

        bkg_array.append(bkg)
        bkg_err_array.append(bkg_err)
        del hmass

    return bkg_array, bkg_err_array, x_axis


def calc_peak_sigma(df_mc_reco, sel_signal, mass, fit_region, bin_width):
    """
    Estimate the width of the signal peak from MC.
    """
    logger = get_logger()
    df_signal = df_mc_reco.query(sel_signal)
    num_bins = (fit_region[1] - fit_region[0]) / bin_width
    num_bins = int(round(num_bins))

    hmass = TH1F('hmass', '', num_bins, fit_region[0], fit_region[1])
    mass_array = df_signal['inv_mass_ML'].values
    for mass_value in np.nditer(mass_array):
        hmass.Fill(mass_value)

    gaus_fit = TF1("gaus_fit", "gaus", fit_region[0], fit_region[1])
    gaus_fit.SetParameters(0, hmass.Integral())
    gaus_fit.SetParameters(1, mass)
    gaus_fit.SetParameters(2, 0.02)
    logger.debug("To fit the signal a gaussian function is used")
    fit = hmass.Fit("gaus_fit", "RQ")

    if int(fit) != 0:
        logger.error("Problem in signal peak fit")
        sigma = 0.
        return sigma

    sigma = gaus_fit.GetParameter(2)
    logger.debug("Mean of the gaussian: %f", gaus_fit.GetParameter(1))
    logger.debug("Sigma of the gaussian: %f", sigma)
    del hmass
    del gaus_fit

    return sigma


def calc_signif(sig_array, sig_err_array, bkg_array, bkg_err_array):
    """
    Calculate the expected signal significance as a function of the treshold on the
    ML model output.
    """
    signif_array = []
    signif_err_array = []

    for sig, bkg, sig_err, bkg_err in zip(sig_array, bkg_array, sig_err_array, bkg_err_array):
        signif = 0.
        signif_err = 0.

        if sig > 0 and (sig + bkg) > 0:
            signif = sig / np.sqrt(sig + bkg)
            signif_err = signif * np.sqrt((sig_err**2 + bkg_err**2) / (4 * (sig + bkg)**2) + \
                         (bkg / (sig + bkg)) * sig_err**2 / sig**2)

        signif_array.append(signif)
        signif_err_array.append(signif_err)

    return signif_array, signif_err_array


def plot_fonll(filename, fonll_pred, frag_frac, part_label, suffix, plot_dir):
    """
    Plot the FONLL prediction for the current particle.
    """
    df = pd.read_csv(filename)
    plt.figure(figsize=(20, 15))
    plt.subplot(111)
    plt.plot(df['pt'], df[fonll_pred] * frag_frac, linewidth=4.0)
    plt.xlabel('P_t [GeV/c]', fontsize=20)
    plt.ylabel('Cross Section [pb/GeV]', fontsize=20)
    plt.title("FONLL cross section " + part_label, fontsize=20)
    plt.semilogy()
    plot_name = plot_dir + '/FONLL curve %s.png' % (suffix)
    plt.savefig(plot_name)


def calc_eff_acc(df_mc_gen, df_mc_reco, sel_signal_reco, sel_signal_gen):
    """
    Calculate the efficiency times acceptance before the ML model selections.
    """
    logger = get_logger()
    df_mc_gen = df_mc_gen.query(sel_signal_gen)
    df_mc_reco = df_mc_reco.query(sel_signal_reco)
    if df_mc_gen.empty:
        logger.error("In division denominator is empty")
        return 0.
    eff_acc = len(df_mc_reco) / len(df_mc_gen)
    logger.debug("Pre-selection efficiency times acceptance: %f", eff_acc)

    return eff_acc


def calc_sig_dmeson(filename, fonll_pred, frag_frac, branch_ratio, sigma_mb, f_prompt,
                    ptmin, ptmax, eff_acc, n_events):
    """
    Estimate the expected signal yield before the ML model selections,
    this approach is valid for all D-meson with the proper parameter configuration.
    """
    logger = get_logger()
    df = pd.read_csv(filename)
    df_in_pt = df.query('(pt >= @ptmin) and (pt < @ptmax)')[fonll_pred]
    prod_cross = df_in_pt.sum() * frag_frac * 1e-12 / len(df_in_pt)
    delta_pt = ptmax - ptmin
    signal_yield = 2. * prod_cross * delta_pt * branch_ratio * eff_acc * n_events \
                   / (sigma_mb * f_prompt)
    logger.debug("Expected signal yield: %f", signal_yield)

    return signal_yield


def study_signif(case, names, bin_lim, file_mc, file_data, df_mc_reco, df_ml_test,
                 df_data_dec, suffix, plotdir):
    """
    Study the efficiency and the expected signal significance as a function of
    the threshold value on a ML model output.
    """
    logger = get_logger()
    gROOT.SetBatch(True)
    gROOT.ProcessLine("gErrorIgnoreLevel = kWarning;")

    gen_dict = get_database_ml_parameters()[case]
    mass = gen_dict["mass"]

    sopt_dict = gen_dict['signif_opt']
    mass_fit_lim = sopt_dict['mass_fit_lim']
    bin_width = sopt_dict['bin_width']
    bkg_fract = sopt_dict['bkg_data_fraction']

    df_mc_gen = getdataframe(file_mc, gen_dict['treename_gen'], gen_dict['var_gen'])
    df_mc_gen = df_mc_gen.query(gen_dict['presel_gen'])
    df_mc_gen = filterdataframe_singlevar(df_mc_gen, gen_dict['ptgen'], bin_lim[0], bin_lim[1])
    df_evt = getdataframe(file_data, sopt_dict['treename_event'], sopt_dict['var_event'])
    n_events = len(df_evt.query(sopt_dict['sel_event']))
    logger.debug("Number of events: %d", n_events)

    # The uncertainty on the pre-selection efficiency times acceptance is neglected as
    # that on the expected signal yield
    eff_acc = calc_eff_acc(df_mc_gen, df_mc_reco, sopt_dict['sel_signal_reco_sopt'],
                           sopt_dict['sel_signal_gen_sopt'])
    exp_signal = calc_sig_dmeson(sopt_dict['filename_fonll'], sopt_dict['fonll_pred'],
                                 sopt_dict['FF'], sopt_dict['BR'], sopt_dict['sigma_MB'],
                                 sopt_dict['f_prompt'], bin_lim[0], bin_lim[1], eff_acc, n_events)
    plot_fonll(sopt_dict['filename_fonll'], sopt_dict['fonll_pred'],
               sopt_dict['FF'], case, suffix, plotdir)

    fig_eff = plt.figure(figsize=(20, 15))
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Efficiency', fontsize=20)
    plt.title("Efficiency Signal", fontsize=20)

    fig_signif = plt.figure(figsize=(20, 15))
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Significance (A.U.)', fontsize=20)
    plt.title("Significance vs probability ", fontsize=20)

    df_data_dec = df_data_dec.tail(round(len(df_data_dec) * bkg_fract))
    sigma = calc_peak_sigma(df_mc_reco, sopt_dict['sel_signal_reco_sopt'], mass,
                            mass_fit_lim, bin_width)
    sig_region = [mass - 3 * sigma, mass + 3 * sigma]

    for name in names:

        eff_array, eff_err_array, x_axis = calc_efficiency(df_ml_test,
                                                           sopt_dict['sel_signal_reco_sopt'],
                                                           name, sopt_dict['num_steps'])
        plt.figure(fig_eff.number)
        plt.errorbar(x_axis, eff_array, yerr=eff_err_array, alpha=0.3, label=f'{name}',
                     elinewidth=2.5, linewidth=4.0)

        sig_array = [eff * exp_signal for eff in eff_array]
        sig_err_array = [eff_err * exp_signal for eff_err in eff_err_array]
        bkg_array, bkg_err_array, _ = calc_bkg(df_data_dec, name, sopt_dict['num_steps'],
                                               mass_fit_lim, bin_width, sig_region)
        bkg_array = [bkg / bkg_fract for bkg in bkg_array]
        bkg_err_array = [bkg_err / bkg_fract for bkg_err in bkg_err_array]
        signif_array, signif_err_array = calc_signif(sig_array, sig_err_array, bkg_array,
                                                     bkg_err_array)
        plt.figure(fig_signif.number)
        plt.errorbar(x_axis, signif_array, yerr=signif_err_array, alpha=0.3, label=f'{name}',
                     elinewidth=2.5, linewidth=4.0)

    eff_arr_std, eff_er_arr_std, x_axis_std = calc_efficiency(df_ml_test,
                                                              sopt_dict['sel_signal_reco_sopt'],
                                                              'Standard ALICE', num_steps,
                                                              sopt_dict['sel_signal_reco_std'])
    plt.figure(fig_eff.number)
    plt.errorbar(x_axis_std, eff_arr_std, yerr=eff_er_arr_std, alpha=0.3, label=f'ALICE Standard',
                 elinewidth=2.5, linewidth=4.0)

    plt.figure(fig_eff.number)
    plt.legend(loc="lower left", prop={'size': 18})
    plt.savefig(plotdir + '/Efficiency%sSignal.png' % suffix)

    plt.figure(fig_signif.number)
    plt.legend(loc="lower left", prop={'size': 18})
    plt.savefig(plotdir + '/Significance%s.png' % suffix)
