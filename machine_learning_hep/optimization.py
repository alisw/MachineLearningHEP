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
from ROOT import TH1F, gROOT  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.logger import get_logger
from machine_learning_hep.general import get_database_signifopt

def calc_efficiency(df_to_sel, sel_signal, name, num_step):
    df_to_sel = df_to_sel.query(sel_signal)
    x_axis = np.linspace(0, 1.00, num_step)
    num_tot_cand = len(df_to_sel)
    eff_array = []

    for thr in x_axis:
        num_sel_cand = len(df_to_sel[df_to_sel['y_test_prob' + name].values >= thr])
        eff_array.append(num_sel_cand/num_tot_cand)

    return eff_array, x_axis


def calc_bkg(df_bkg, name, num_step, mass_cuts, fit_region, bin_width, sig_region):
    x_axis = np.linspace(0, 1.00, num_step)
    bkg_array = []
    num_bins = (fit_region[1] - fit_region[0]) / bin_width
    num_bins = int(round(num_bins))
    bin_width = (fit_region[1] - fit_region[0]) / num_bins
    bkg_mass_mask = (df_bkg['inv_mass_ML'].values <= mass_cuts[0]) | (
        df_bkg['inv_mass_ML'].values >= mass_cuts[1])
    df_mass = df_bkg[bkg_mass_mask]

    for thr in x_axis:
        bkg = 0.
        hmass = TH1F('hmass', '', num_bins, fit_region[0], fit_region[1])
        bkg_sel_mask = df_mass['y_test_prob' + name].values >= thr
        sel_mass_array = df_mass[bkg_sel_mask]['inv_mass_ML'].values

        if len(sel_mass_array) > 5:
            for mass_value in np.nditer(sel_mass_array):
                hmass.Fill(mass_value)

            fit = hmass.Fit('expo', 'Q', '', fit_region[0], fit_region[1])
            if int(fit) == 0:
                fit_func = hmass.GetFunction('expo')
                bkg = fit_func.Integral(
                    sig_region[0], sig_region[1]) / bin_width
                del fit_func

        bkg_array.append(bkg)
        del hmass

    return bkg_array, x_axis


def calc_signif(sig_array, bkg_array):
    signif_array = []

    for sig, bkg in zip(sig_array, bkg_array):
        signif = 0
        if sig > 0 and bkg > 0:
            signif = sig/np.sqrt(sig + bkg)
        signif_array.append(signif)

    return signif_array


def plot_fonll(common_dict, part_label, suffix, plot_dir):
    df = pd.read_csv(common_dict['filename'])
    plt.figure(figsize=(20, 15))
    plt.subplot(111)
    plt.plot(df['pt'], df['central'] * common_dict['FF'], linewidth=4.0)
    plt.xlabel('P_t [GeV/c]', fontsize=20)
    plt.ylabel('Cross Section [pb/GeV]', fontsize=20)
    plt.title("FONLL cross section " + part_label, fontsize=20)
    plt.semilogy()
    plot_name = plot_dir + '/FONLL curve %s.png' % (suffix)
    plt.savefig(plot_name)


def countevents(df_evt, sel_evt_counter):
    df_evt = df_evt.query(sel_evt_counter)
    nevents = len(df_evt)

    return nevents


def calculate_eff_acc(df_mc_gen, df_mc_reco, sel_signal_gen):
    logger = get_logger()
    df_mc_gen = df_mc_gen.query(sel_signal_gen)
    df_mc_reco = df_mc_reco.query(sel_signal_gen)
    if df_mc_gen.empty:
        logger.error("In division denominator is empty")
    eff_acc = len(df_mc_reco)/len(df_mc_gen)

    return eff_acc

def calc_sig_dmeson(common_dict, ptmin, ptmax, eff_acc, n_events):
    df = pd.read_csv(common_dict['filename'])
    df_in_pt = df.query('(pt >= @ptmin) and (pt < @ptmax)')['central']
    prod_cross = df_in_pt.sum() * common_dict['FF'] * 1e-12 / len(df_in_pt)
    delta_pt = ptmax - ptmin
    signal_yield = 2. * prod_cross * delta_pt * common_dict['BR'] * eff_acc * n_events \
                   / (common_dict['sigma_MB'] * common_dict['f_prompt'])
    return signal_yield

# pylint: disable=too-many-arguments
def study_signif(case, names, binmin, binmax, df_mc_gen, df_mc_reco, df_ml_test, df_data_dec,
                 n_events, sel_signal_gen, mass_cut, suffix, plotdir):
    gROOT.SetBatch(True)
    gROOT.ProcessLine("gErrorIgnoreLevel = 2000;")

    data_signifopt = get_database_signifopt()[case]
    common_dict = data_signifopt['common']
    mass_fit_lim = common_dict['mass_fit_lim']
    bin_width = common_dict['bin_width']
    sigma = common_dict['sigma']
    mass = common_dict["mass"]
    sig_region = [mass - 3 * sigma, mass + 3 * sigma]

    plot_fonll(common_dict, case, suffix, plotdir)
    eff_acc = calculate_eff_acc(df_mc_gen, df_mc_reco, sel_signal_gen)
    exp_signal = calc_sig_dmeson(common_dict, binmin, binmax, eff_acc, n_events)

    fig_eff = plt.figure(figsize=(20, 15))
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Efficiency', fontsize=20)
    plt.title("Efficiency Signal", fontsize=20)

    fig_signif = plt.figure(figsize=(20, 15))
    plt.xlabel('Probability', fontsize=20)
    plt.ylabel('Significance (A.U.)', fontsize=20)
    plt.title("Significance vs probability ", fontsize=20)

    num_steps = 101

    for name in names:

        eff_array, x_axis = calc_efficiency(df_ml_test, sel_signal_gen, name, num_steps)
        plt.figure(fig_eff.number)
        plt.plot(x_axis, eff_array, alpha=0.3, label='%s' % name, linewidth=4.0)

        sig_array = [eff * exp_signal for eff in eff_array]
        bkg_array, _ = calc_bkg(df_data_dec, name, num_steps, mass_cut,
                                mass_fit_lim, bin_width, sig_region)
        signif_array = calc_signif(sig_array, bkg_array)
        plt.figure(fig_signif.number)
        plt.plot(x_axis, signif_array, alpha=0.3, label='%s' % name, linewidth=4.0)

        plt.figure(fig_eff.number)
        plt.legend(loc="lower left", prop={'size': 18})
        plt.savefig(plotdir + '/Efficiency%sSignal.png' % suffix)

        plt.figure(fig_signif.number)
        plt.legend(loc="lower left", prop={'size': 18})
        plt.savefig(plotdir + '/Significance%s.png' % suffix)
