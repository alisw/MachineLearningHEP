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
main macro for running the study
"""
from sklearn.utils import shuffle
from machine_learning_hep.general import get_database_ml_parameters, getdataframe
from machine_learning_hep.general import filterdataframe, split_df_sigbkg
from machine_learning_hep.preparesamples import prep_mlsamples

def test():
    data = get_database_ml_parameters()
    nevt_sig = 10000
    nevt_bkg = 10000
    mltype = "BinaryClassification"
    mlsubtype = "HFmeson"
    case = "Lc"
    var_skimming = ["pt_cand_ML"]
    varmin = [2]
    varmax = [4]
    test_frac = 0.2
    rnd_splt = 12
    rnd_shuffle = 12

    print(nevt_sig, nevt_bkg, mltype, mlsubtype, case)
    var_all = data[case]["var_all"]
    var_signal = data[case]["var_signal"]
    sel_signal = data[case]["sel_signal"]
    sel_bkg = data[case]["sel_bkg"]

#     var_training = data[case]["var_training"]
#     var_target = data[case]["var_target"]
#     var_corr_x, var_corr_y = data[case]["var_correlation"]

    loadsampleoption = 1

    if loadsampleoption == 1:
        filesig, filebkg = data[case]["sig_bkg_files"]
        trename = data[case]["tree_name"]
        df_sig = getdataframe(filesig, trename, var_all)
        df_bkg = getdataframe(filebkg, trename, var_all)
        df_sig = filterdataframe(df_sig, var_skimming, varmin, varmax)
        df_bkg = filterdataframe(df_bkg, var_skimming, varmin, varmax)
        df_sig = df_sig.query(sel_signal, random_state=rnd_shuffle)
        df_bkg = df_bkg.query(sel_bkg, random_state=rnd_shuffle)
        df_sig = shuffle(df_sig)
        df_bkg = shuffle(df_bkg)
        df_ml_train, df_ml_test = \
            prep_mlsamples(df_sig, df_bkg, var_signal, nevt_sig, nevt_bkg, test_frac, rnd_splt)
        df_sig_train, df_bkg_train = split_df_sigbkg(df_ml_train, var_signal)
        df_sig_test, df_bkg_test = split_df_sigbkg(df_ml_test, var_signal)
        print("events for ml train %d and test %d" % (len(df_ml_train), len(df_ml_test)))
        print("events for signal train %d and test %d" % (len(df_sig_train), len(df_sig_test)))
        print("events for bkg train %d and test %d" % (len(df_bkg_train), len(df_bkg_test)))

test()
