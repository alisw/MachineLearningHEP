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
Methods to: utility methods to conpute efficiency and study expected significance
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from ROOT import TH1F, TFile  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.logger import get_logger

def select_by_threshold(df, label, thr, name):
    # Changed from >= to > since we use that atm for the nominal selection
    # See processer.py self.l_selml
    if label == "bkg":
        return df[df[f'y_test_prob{name}{label}'].values <= thr]
    if label == "":
        return df[df[f'y_test_prob{name}{label}'].values > thr]
    return df[df[f'y_test_prob{name}{label}'].values >= thr]

def get_x_axis(num_steps, class_label):
    ns_left = int(num_steps / 10) - 1
    ns_right = num_steps - ns_left
    if class_label == "bkg":
        ns_left, ns_right = ns_right, ns_left
    x_axis_left = np.linspace(0., 0.49, ns_left)
    x_axis_right = np.linspace(0.5, 1.0, ns_right)
    x_axis = np.concatenate((x_axis_left, x_axis_right))
    return x_axis

def calc_bkg(df_bkg, name, num_steps, fit_region, bkg_func, bin_width, sig_region, save_fit, #pylint: disable=too-many-arguments
             out_dir, pt_lims, invmassvar, mltype):
    """
    Estimate the number of background candidates under the signal peak. This is obtained
    from real data with a fit of the sidebands of the invariant mass distribution.
    """
    logger = get_logger()
    class_label = "bkg" if mltype == "MultiClassification" else ""
    x_axis = get_x_axis(num_steps, class_label)
    bkg_array = []
    bkg_err_array = []
    num_bins = (fit_region[1] - fit_region[0]) / bin_width
    num_bins = int(round(num_bins))
    bin_width = (fit_region[1] - fit_region[0]) / num_bins

    if save_fit:
        logger.debug("Saving bkg fits to file")
        pt_min = pt_lims[0]
        pt_max = pt_lims[1]
        out_file = TFile(f'{out_dir}/bkg_fits_{name}_pt{pt_min:.1f}_{pt_max:.1f}.root', 'recreate')
        out_file.cd()

    logger.debug("To fit the bkg a %s function is used", bkg_func)
    for thr in x_axis:
        bkg = 0.
        bkg_err = 0.
        hmass = TH1F(f'hmass_{thr:.5f}', '', num_bins, fit_region[0], fit_region[1])
        df_bkg_sel = select_by_threshold(df_bkg, class_label, thr, name)
        sel_mass_array = df_bkg_sel[invmassvar].values

        if len(sel_mass_array) > 5:
            for mass_value in np.nditer(sel_mass_array):
                hmass.Fill(mass_value)
            fit = hmass.Fit(bkg_func, 'Q', '', fit_region[0], fit_region[1])
            if save_fit:
                hmass.Write()
            if int(fit) == 0:
                fit_func = hmass.GetFunction(bkg_func)
                bkg = fit_func.Integral(sig_region[0], sig_region[1]) / bin_width
                bkg_err = fit_func.IntegralError(sig_region[0], sig_region[1]) / bin_width
                del fit_func
        elif save_fit:
            hmass.Write()

        bkg_array.append(bkg)
        bkg_err_array.append(bkg_err)
        del hmass

    out_file.Close()
    return bkg_array, bkg_err_array, x_axis



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

def calc_eff(num, den):
    eff = num / den
    eff_err = np.sqrt(eff * (1 - eff) / den)

    return eff, eff_err

def calc_sigeff_steps(num_steps, df_sig, name, mltype):
    logger = get_logger()
    class_label = "bkg" if mltype == "MultiClassification" else ""
    x_axis = get_x_axis(num_steps, class_label)
    if df_sig.empty:
        logger.error("In division denominator is empty")
        eff_array = [0] * num_steps
        eff_err_array = [0] * num_steps
        return eff_array, eff_err_array, x_axis
    num_tot_cand = len(df_sig)
    eff_array = []
    eff_err_array = []
    for thr in x_axis:
        num_sel_cand = len(select_by_threshold(df_sig, class_label, thr, name))
        eff, err_eff = calc_eff(num_sel_cand, num_tot_cand)
        eff_array.append(eff)
        eff_err_array.append(err_eff)

    return eff_array, eff_err_array, x_axis

def prepare_eff_signif_figure(y_label, mltype):
    class_label = "Bkg" if mltype == "MultiClassification" else "Prompt"
    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel(f"{class_label} threshold", fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=20)
    return fig
