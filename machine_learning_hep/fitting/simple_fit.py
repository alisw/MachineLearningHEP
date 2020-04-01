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
Script only used for fitting
"""

from os.path import exists, join
from os import makedirs
import argparse

from ROOT import TFile, TCanvas # pylint: disable=import-error, no-name-in-module

from machine_learning_hep.logger import configure_logger #, get_logger
from machine_learning_hep.io import parse_yaml
from machine_learning_hep.fitting.fitters import FitAliHF, FitROOTGauss
from machine_learning_hep.fitting.utils import save_fit

#############################################################################
#                                                                           #
# This script is used for stand-alone fitting.                              #
#                                                                           #
# It can be used from the command-line. To see available arguments either   #
# read the code at the bottom concerning the argument parsing or just type  #
#                                                                           #
# $> python fitting/simple_fit.py -- -h                                     #
#                                                                           #
# Giving the path to a database, the corresponding masshisto.root is read   #
# from the result path (period can be specified).                           #
#                                                                           #
# This script assumes an initial fit on MC should be used to extract an     #
# initial value for the Gaussian width to be used to initialise the data    #
# fit with.                                                                 #
#                                                                           #
# It is looped over all pT and multiplicity bins and single parameters can  #
# be easiliy exchanged in the dictionary called "fit_pars" which can be     #
# found below in the double loop. At the moment the parameters are as well  #
# taken from the database specified in the beginning.                       #
#                                                                           #
# All fits are saved and drawn to an output directory which can be          #
# specified as a command line argument as well.                             #
#                                                                           #
# NOTE: This script can also be used by calling the function                #
#                                                                           #
# do_simple_fit(...)                                                        #
#                                                                           #
# directly from any other script.                                           #
#                                                                           #
#############################################################################

def draw(fitter, save_name, **kwargs):
    """Draw helper function

        This can safely be ignored in view of understanding this script
        and it doesn't do anything but drawing a fit. It won't change
        any number.
    """
    c = TCanvas("canvas", "", 500, 500)
    try:
        fitter.draw(c, title="Fit", x_axis_label="", y_axis_label="", **kwargs)
    # NOTE The broad-except is only used to make this script running under
    #      any circumstances and ignore any reason for which a fit could not
    #      be drawn.
    except Exception as e: # pylint: disable=broad-except
        print(f"Could not draw fit")
        print(fitter)
        print(e)
    c.SaveAs(save_name)
    c.Close()


# pylint: disable=too-many-locals, too-many-statements
def do_simple_fit(database, type_ana, period_number=-1, output_dir="simple_fit"):
    """Doing the fit

    1) Extracting parameters from the database
    2) Looping over all pT bins once to do a fit on MC invariant mass distribution
       assuming a pure Gaussian
    3) Looping over all pT and multiplicity bins. Fits in per pT bin are initialised
       with the same sigma extracted from the MC fit.

    Args:
        database: entire analysis/processing database below the case field
        type_ana: analysis name to extract the fit parameters from
        period_number:  -1 --> all periods merged (default)
                         0 --> 2016
                         1 --> 2017
                         2 --> 2018
        output_dir: directory where all result plots are saved along with the fits
    """
    # Map fit function names to numbers complying with AliHFInvMassFitter framework
    sig_func_map = {"kGaus": 0, "k2Gaus": 1, "kGausSigmaRatioPar": 2}
    bkg_func_map = {"kExpo": 0, "kLin": 1, "Pol2": 2, "kNoBk": 3, "kPow": 4, "kPowEx": 5}


    # Extract the analysis parameters
    fit_pars = database["analysis"][type_ana]

    ####################################
    # BEGIN reading all fit parameters #
    ####################################

    # 1) Binning settings

    # Name of first binning variable is the only thing that cannot be extracted from the
    # analysis section of the database
    bin1_name = database["var_binning"]
    bins1_edges_low = fit_pars["sel_an_binmin"]
    bins1_edges_up = fit_pars["sel_an_binmax"]
    n_bins1 = len(bins1_edges_low)
    bin2_name = fit_pars["var_binning2"]
    bin2_gen_name = fit_pars["var_binning2_gen"]
    bins2_edges_low = fit_pars["sel_binmin2"]
    bins2_edges_up = fit_pars["sel_binmax2"]
    n_bins2 = len(bins2_edges_low)
    bin_matching = fit_pars["binning_matching"]

    bineff = fit_pars["usesinglebineff"]
    bins2_int_bin = bineff if bineff is not None else 0
    # Fit method flags
    sig_func_name = fit_pars["sgnfunc"]
    bkg_func_name = fit_pars["bkgfunc"]
    fit_range_low = fit_pars["massmin"]
    fit_range_up = fit_pars["massmax"]
    likelihood = fit_pars["dolikelihood"]
    rebin = fit_pars["rebin"]
    try:
        iter(rebin[0])
    except TypeError:
        rebin = [rebin for _ in range(n_bins2)]

    # Initial fit parameters
    mean = fit_pars["masspeak"]
    fix_mean = fit_pars["FixedMean"]
    sigma = fit_pars["sigmaarray"]
    fix_sigma = fit_pars["SetFixGaussianSigma"]
    n_sigma_sideband = fit_pars["exclude_nsigma_sideband"]
    n_sigma_signal = fit_pars["nsigma_signal"]
    rel_sigma_bound = fit_pars["MaxPercSigmaDeviation"]

    # Second peak flags
    include_sec_peak = fit_pars.get("includesecpeak", [False] * n_bins1)
    try:
        iter(include_sec_peak[0])
    except TypeError:
        include_sec_peak = [include_sec_peak for _ in range(n_bins2)]

    sec_mean = fit_pars["masssecpeak"] if include_sec_peak else None
    fix_sec_mean = fit_pars.get("fix_masssecpeak", [False] * n_bins1)
    try:
        iter(fix_sec_mean[0])
    except TypeError:
        fix_sec_mean = [fix_sec_mean for _ in range(n_bins2)]
    sec_sigma = fit_pars["widthsecpeak"] if include_sec_peak else None
    fix_sec_sigma = fit_pars["fix_widthsecpeak"] if include_sec_peak else None

    # Reflections flag
    include_reflections = fit_pars.get("include_reflection", False)

    # Is this a trigger weighted histogram?
    apply_weights = fit_pars["triggersel"]["weighttrig"]

    # 4) Misc
    # ML WP is needed to build the suffix for extracting the mass histogram
    prob_cut_fin = database["mlapplication"]["probcutoptimal"]

    ##################################
    # END reading all fit parameters #
    ##################################


    # Where the histomass.root is read from
    input_dir_mc = fit_pars["mc"]["results"][period_number] \
            if period_number > -1 else fit_pars["mc"]["resultsallp"]
    input_dir_data = fit_pars["data"]["results"][period_number] \
            if period_number > -1 else fit_pars["data"]["resultsallp"]

    # Otherwise the output directory might not exist, hence create
    if not exists(output_dir):
        makedirs(output_dir)

    # Read the files with extracted mass histograms
    histo_file_mc = TFile.Open(join(input_dir_mc, "masshisto.root"), "READ")
    histo_file_data = TFile.Open(join(input_dir_data, "masshisto.root"), "READ")

    ##############################################
    # Pre-fit on MC multiplicity integrated bins #
    ##############################################
    mc_fitters = []
    for ipt in range(n_bins1):

        # Always have the MC histogram for mult. integrated
        bin_id_match = bin_matching[ipt]
        suffix_mc_int = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                        (bin1_name, bins1_edges_low[ipt],
                         bins1_edges_up[ipt], prob_cut_fin[bin_id_match],
                         bin2_gen_name, bins2_edges_low[bins2_int_bin],
                         bins2_edges_up[bins2_int_bin])
        # Get always the one for the multiplicity integrated
        histo_mc_int = histo_file_mc.Get("hmass_sig" + suffix_mc_int)
        histo_mc_int.SetDirectory(0)

        fit_pars_mc = {"mean": mean,
                       "sigma": sigma[ipt],
                       "rebin": rebin[bins2_int_bin][ipt],
                       "use_user_fit_range": False,
                       "fit_range_low": fit_range_low[ipt],
                       "fit_range_up": fit_range_up[ipt],
                       "likelihood": likelihood}

        fitter_mc = FitROOTGauss(fit_pars_mc, histo=histo_mc_int)
        mc_fitters.append(fitter_mc)

        # Fit, draw and save
        fitter_mc.fit()
        save_name = f"fit_MC_bin1_{ipt}.eps"
        save_name = join(output_dir, save_name)
        draw(fitter_mc, save_name)

        save_fit(fitter_mc, join(output_dir, f"fit_MC_ipt_{ipt}"))
        if not fitter_mc.success:
            print(f"Fit in (ipt,) = ({ipt},) failed for MC fit.")

    ########################
    # Perform central fits #
    ########################

    # Fitters are collected due to possible memory issues observed
    # for AliHFInvMassFitter (probably issues related to destructor)
    data_fitters = []
    for imult in range(n_bins2):
        for ipt in range(n_bins1):

            # We only perform fit where the fit on M was successful
            mc_fit = mc_fitters[ipt]
            if not mc_fit.success:
                print(f"Pre-fit on MC in pT bin {ipt} not successful. Continue...")
                continue

            bin_id_match = bin_matching[ipt]

            suffix_data = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                          (bin1_name, bins1_edges_low[ipt],
                           bins1_edges_up[ipt], prob_cut_fin[bin_id_match],
                           bin2_name, bins2_edges_low[imult],
                           bins2_edges_up[imult])
            # There might be a different name for the MC histogram due to a potential
            # difference in the multiplicity binning variable
            suffix_mc = "%s%d_%d_%.2f%s_%.2f_%.2f" % \
                        (bin1_name, bins1_edges_low[ipt],
                         bins1_edges_up[ipt], prob_cut_fin[bin_id_match],
                         bin2_gen_name, bins2_edges_low[imult],
                         bins2_edges_up[imult])


            # Get all histograms which might be required
            # Are we using weighted or unweighted histograms?
            histo_data_name = "h_invmass_weight" if apply_weights else "hmass"
            histo_data = histo_file_data.Get(histo_data_name + suffix_data)
            histo_data.SetDirectory(0)
            histo_mc = histo_file_mc.Get("hmass_sig" + suffix_mc)
            histo_mc.SetDirectory(0)
            histo_refl = histo_file_mc.Get("hmass_refl" + suffix_mc)
            histo_refl.SetDirectory(0)

            ##################################
            # All fit parameters from the DB #
            ##################################
            fit_pars = {"mean": mean,
                        "fix_mean": fix_mean,
                        "sigma": mc_fit.fit_pars["sigma"],
                        "fix_sigma": fix_sigma[ipt],
                        "include_sec_peak": include_sec_peak[imult][ipt],
                        "sec_mean": None,
                        "fix_sec_mean": False,
                        "sec_sigma": None,
                        "fix_sec_sigma": False,
                        "use_sec_peak_rel_sigma": True,
                        "include_reflections": include_reflections,
                        "fix_reflections_s_over_b": True,
                        "rebin": rebin[imult][ipt],
                        "fit_range_low": fit_range_low[ipt],
                        "fit_range_up": fit_range_up[ipt],
                        "likelihood": likelihood,
                        "n_sigma_sideband": n_sigma_sideband,
                        "rel_sigma_bound": rel_sigma_bound,
                        "sig_func_name": sig_func_map[sig_func_name[ipt]],
                        "bkg_func_name": bkg_func_map[bkg_func_name[ipt]]}

            # Include second peak if required
            if fit_pars["include_sec_peak"]:
                fit_pars["sec_mean"] = sec_mean
                fit_pars["fix_sec_mean"] = fix_sec_mean[imult][ipt]
                fit_pars["sec_sigma"] = sec_sigma
                fit_pars["fix_sec_sigma"] = fix_sec_sigma
                fit_pars["use_sec_peak_rel_sigma"] = True

            ################################
            # END fit parameter extraction #
            ################################

            # Construct fitter and add to list
            fitter = FitAliHF(fit_pars, histo=histo_data, histo_mc=histo_mc,
                              histo_reflections=histo_refl)
            data_fitters.append(fitter)

            # Fit, draw and save
            fitter.fit()
            save_name = f"fit_bin1_{ipt}_ibin2_{imult}.eps"
            save_name = join(output_dir, save_name)
            draw(fitter, save_name, sigma_signal=n_sigma_signal)

            save_fit(fitter, join(output_dir, f"fit_ipt_{ipt}_imult_{imult}"))

            if not fitter.success:
                print(f"Fit in (ipt, imult) = ({ipt}, {imult}) failed. Try to draw and save " \
                      f"anyway.")


def main():
    """
    This is used as the entry point for fitting.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-analysis", "-d", dest="database_analysis",
                        help="analysis database to be used", required=True)
    parser.add_argument("--analysis", "-a", dest="type_ana",
                        help="choose type of analysis")
    parser.add_argument("--period-number", "-p", dest="period_number", type=int,
                        help="choose type of analysis (0: 2016, 1: 2017, 2: 2018, " \
                             "-1: all merged (default))", default=-1)
    parser.add_argument("--output", "-o", default="simple_fit",
                        help="result output directory")


    args = parser.parse_args()

    configure_logger(False, None)

    # Extract database as dictionary
    data = parse_yaml(args.database_analysis)
    data = data[list(data.keys())[0]]
    # Run the chain
    do_simple_fit(data, args.type_ana, args.period_number, args.output)


if __name__ == "__main__":
    main()
