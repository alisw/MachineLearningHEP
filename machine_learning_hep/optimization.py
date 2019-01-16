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

def calc_efficiency(df_to_sel, sel_signal, name, num_step):
    df_to_sel = df_to_sel.query(sel_signal)
    x_axis = np.linspace(0, 1.00, num_step)
    num_tot_cand = len(df_to_sel)
    eff_array = []

    for thr in x_axis:
        num_sel_cand = len(df_to_sel[df_to_sel['y_test_prob' + name].values >= thr])
        eff_array.append(num_sel_cand / num_tot_cand)

    return eff_array, x_axis


def calc_bkg(df_bkg, name, num_step, fit_region, bin_width, sig_region):
    x_axis = np.linspace(0, 1.00, num_step)
    bkg_array = []
    num_bins = (fit_region[1] - fit_region[0]) / bin_width
    num_bins = int(round(num_bins))
    bin_width = (fit_region[1] - fit_region[0]) / num_bins

    for thr in x_axis:
        bkg = 0.
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
                del fit_func

        bkg_array.append(bkg)
        del hmass

    return bkg_array, x_axis

def calc_peak_sigma(df_mc_reco, sel_signal, mass, fit_region, bin_width):
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
    fit = hmass.Fit("gaus_fit", "RQ")

    if int(fit) != 0:
        logger.error("Problem in signal peak fit")
        sigma = 0.
        return sigma

    sigma = gaus_fit.GetParameter(2)
    del hmass
    del gaus_fit

    return sigma


def calc_signif(sig_array, bkg_array):
    signif_array = []
    for sig, bkg in zip(sig_array, bkg_array):
        signif = 0
        if sig > 0 and bkg > 0:
            signif = sig / np.sqrt(sig + bkg)
        signif_array.append(signif)

    return signif_array


def plot_fonll(filename, fonll_pred, frag_frac, part_label, suffix, plot_dir):
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


def calculate_eff_acc(df_mc_gen, df_mc_reco, sel_signal_reco, sel_signal_gen):
    logger = get_logger()
    df_mc_gen = df_mc_gen.query(sel_signal_gen)
    df_mc_reco = df_mc_reco.query(sel_signal_reco)
    if df_mc_gen.empty:
        logger.error("In division denominator is empty")
    eff_acc = len(df_mc_reco) / len(df_mc_gen)

    return eff_acc

def calc_sig_dmeson(filename, fonll_pred, frag_frac, branch_ratio, sigma_mb, f_prompt,
                    ptmin, ptmax, eff_acc, n_events):
    df = pd.read_csv(filename)
    df_in_pt = df.query('(pt >= @ptmin) and (pt < @ptmax)')[fonll_pred]
    prod_cross = df_in_pt.sum() * frag_frac * 1e-12 / len(df_in_pt)
    delta_pt = ptmax - ptmin
    signal_yield = 2. * prod_cross * delta_pt * branch_ratio * eff_acc * n_events \
                   / (sigma_mb * f_prompt)

    return signal_yield

def study_signif(case, names, bin_lim, file_mc, file_data, df_mc_reco, df_ml_test,
                 df_data_dec, suffix, plotdir):
    gROOT.SetBatch(True)
    gROOT.ProcessLine("gErrorIgnoreLevel = 2000;")

    gen_dict = get_database_ml_parameters()[case]
    mass = gen_dict["mass"]

    sopt_dict = gen_dict['signif_opt']
    sel_signal_reco = sopt_dict['sel_signal_reco_sopt']
    sel_signal_gen = sopt_dict['sel_signal_gen_sopt']
    filename_fonll = sopt_dict['filename_fonll']
    fonll_pred = sopt_dict['fonll_pred']
    frag_frac = sopt_dict['FF']
    mass_fit_lim = sopt_dict['mass_fit_lim']
    bin_width = sopt_dict['bin_width']
    bkg_fract = sopt_dict['bkg_data_fraction']

    df_mc_gen = getdataframe(file_mc, gen_dict['treename_gen'], gen_dict['var_gen'])
    df_mc_gen = df_mc_gen.query(gen_dict['presel_gen'])
    df_mc_gen = filterdataframe_singlevar(df_mc_gen, gen_dict['ptgen'], bin_lim[0], bin_lim[1])
    df_evt = getdataframe(file_data, sopt_dict['treename_event'], sopt_dict['var_event'])
    n_events = len(df_evt.query(sopt_dict['sel_event']))

    sigma = calc_peak_sigma(df_mc_reco, sel_signal_reco, mass, mass_fit_lim, bin_width)
    sig_region = [mass - 3 * sigma, mass + 3 * sigma]
    plot_fonll(filename_fonll, fonll_pred, frag_frac, case, suffix, plotdir)
    eff_acc = calculate_eff_acc(df_mc_gen, df_mc_reco, sel_signal_reco, sel_signal_gen)
    exp_signal = calc_sig_dmeson(filename_fonll, fonll_pred, frag_frac, sopt_dict['BR'],
                                 sopt_dict['sigma_MB'], sopt_dict['f_prompt'], bin_lim[0],
                                 bin_lim[1], eff_acc, n_events)

    fig_eff = plt.figure(figsize=(20, 15))
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Efficiency', fontsize=20)
    plt.title("Efficiency Signal", fontsize=20)

    fig_signif = plt.figure(figsize=(20, 15))
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Significance (A.U.)', fontsize=20)
    plt.title("Significance vs probability ", fontsize=20)

    df_data_dec = df_data_dec.tail(round(len(df_data_dec) * bkg_fract))
    num_steps = 101

    for name in names:

        eff_array, x_axis = calc_efficiency(df_ml_test, sel_signal_reco, name, num_steps)
        plt.figure(fig_eff.number)
        plt.plot(x_axis, eff_array, alpha=0.3, label='%s' % name, linewidth=4.0)

        sig_array = [eff * exp_signal for eff in eff_array]
        bkg_array, _ = calc_bkg(df_data_dec, name, num_steps,
                                mass_fit_lim, bin_width, sig_region)
        bkg_array = [bkg / bkg_fract for bkg in bkg_array]
        signif_array = calc_signif(sig_array, bkg_array)
        plt.figure(fig_signif.number)
        plt.plot(x_axis, signif_array, alpha=0.3, label='%s' % name, linewidth=4.0)

        plt.figure(fig_eff.number)
        plt.legend(loc="lower left", prop={'size': 18})
        plt.savefig(plotdir + '/Efficiency%sSignal.png' % suffix)

        plt.figure(fig_signif.number)
        plt.legend(loc="lower left", prop={'size': 18})
        plt.savefig(plotdir + '/Significance%s.png' % suffix)
