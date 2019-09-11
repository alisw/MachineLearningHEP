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
Methods to: study expected significance
"""
import numpy as np
from ROOT import TH1F, TFile  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.logger import get_logger

def calc_bkg(df_bkg, name, num_steps, fit_region, bkg_func, bin_width, sig_region, save_fit,
             out_dir, pt_lims):
    """
    Estimate the number of background candidates under the signal peak. This is obtained
    from real data with a fit of the sidebands of the invariant mass distribution.
    """
    logger = get_logger()
    ns_left = int(num_steps / 10) - 1
    ns_right = num_steps - ns_left
    x_axis_left = np.linspace(0., 0.49, ns_left)
    x_axis_right = np.linspace(0.5, 1.0, ns_right)
    x_axis = np.concatenate((x_axis_left, x_axis_right))
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
        bkg_sel_mask = df_bkg['y_test_prob' + name].values >= thr
        sel_mass_array = df_bkg[bkg_sel_mask]['inv_mass'].values

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
