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
utilities functions for do_classification_regression
"""
from sklearn.utils import shuffle
# from sklearn.metrics import make_scorer, accuracy_score
from machine_learning_hep.general import filterdataframe, split_df_sigbkg
from machine_learning_hep.preparesamples import prep_mlsamples


def create_mlsamples(df_sig, df_bkg, sel_signal, sel_bkg, rnd_shuffle,  # pylint: disable=too-many-arguments
                     var_skimming, varmin, varmax, var_signal, var_training,
                     nevt_sig, nevt_bkg, test_frac, rnd_splt):
    df_sig = filterdataframe(df_sig, var_skimming, varmin, varmax)
    df_bkg = filterdataframe(df_bkg, var_skimming, varmin, varmax)
    df_sig = df_sig.query(sel_signal)
    df_bkg = df_bkg.query(sel_bkg)
    df_sig = shuffle(df_sig, random_state=rnd_shuffle)
    df_bkg = shuffle(df_bkg, random_state=rnd_shuffle)
    df_ml_train, df_ml_test = prep_mlsamples(df_sig, df_bkg, var_signal, nevt_sig,
                                             nevt_bkg, test_frac, rnd_splt)
    df_sig_train, df_bkg_train = split_df_sigbkg(df_ml_train, var_signal)
    df_sig_test, df_bkg_test = split_df_sigbkg(df_ml_test, var_signal)
    print("events for ml train %d and test %d" % (len(df_ml_train), len(df_ml_test)))
    print("events for signal train %d and test %d" % (len(df_sig_train), len(df_sig_test)))
    print("events for bkg train %d and test %d" % (len(df_bkg_train), len(df_bkg_test)))
    x_train = df_ml_train[var_training]
    y_train = df_ml_train[var_signal]
    x_test = df_ml_test[var_training]
    y_test = df_ml_test[var_signal]

    return df_ml_train, df_ml_test, df_sig_train, df_bkg_train, df_sig_test, df_bkg_test, \
        x_train, y_train, x_test, y_test


