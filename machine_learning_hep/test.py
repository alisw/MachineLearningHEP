###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
main macro for running the study
"""
from sklearn.utils import shuffle
from machine_learning_hep.general import get_database_ml_parameters, getdataframe
from machine_learning_hep.general import checkdir
from machine_learning_hep.general import filterdataframe, split_df_sigbkg, createstringselection
from machine_learning_hep.preparesamples import prep_mlsamples
from machine_learning_hep.correlations import vardistplot, scatterplot, correlationmatrix
from machine_learning_hep.pca import getdataframe_standardised, get_pcadataframe_pca
from machine_learning_hep.pca import plotvariance_pca
from machine_learning_hep.models import getclf_scikit, getclf_xgboost, getclf_keras
from machine_learning_hep.models import fit


def test():  # pylint: disable=too-many-locals, too-many-statements
    data = get_database_ml_parameters()
    nevt_sig = 1000
    nevt_bkg = 1000
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
    var_training = data[case]["var_training"]
#     var_target = data[case]["var_target"]
    var_corr_x, var_corr_y = data[case]["var_correlation"]

    loadsampleoption = 1
    docorrelation = 1
    dostandard = 1
    dopca = 1
    activate_scikit = 1
    activate_xgboost = 1
    activate_keras = 0
    dotraining = 1

    string_selection = createstringselection(var_skimming, varmin, varmax)
    suffix = "nevt_sig%d_nevt_sig%d_%s%s_%s" % \
        (nevt_bkg, nevt_bkg, mltype, case, string_selection)

    dataframe = "dataframes_%s" % (suffix)
    plotdir = "plots_%s" % (suffix)
    output = "output_%s" % (suffix)
    checkdir(dataframe)
    checkdir(plotdir)
    checkdir(output)

    classifiers = []
    names = []

    trainedmodels = []

    x_train = []
    y_train = []

    if loadsampleoption == 1:
        filesig, filebkg = data[case]["sig_bkg_files"]
        trename = data[case]["tree_name"]
        df_sig = getdataframe(filesig, trename, var_all)
        df_bkg = getdataframe(filebkg, trename, var_all)
        df_sig = filterdataframe(df_sig, var_skimming, varmin, varmax)
        df_bkg = filterdataframe(df_bkg, var_skimming, varmin, varmax)
        df_sig = df_sig.query(sel_signal)
        df_bkg = df_bkg.query(sel_bkg)
        df_sig = shuffle(df_sig, random_state=rnd_shuffle)
        df_bkg = shuffle(df_bkg, random_state=rnd_shuffle)
        df_ml_train, df_ml_test = \
            prep_mlsamples(df_sig, df_bkg, var_signal, nevt_sig, nevt_bkg, test_frac, rnd_splt)
        df_sig_train, df_bkg_train = split_df_sigbkg(df_ml_train, var_signal)
        df_sig_test, df_bkg_test = split_df_sigbkg(df_ml_test, var_signal)
        print("events for ml train %d and test %d" % (len(df_ml_train), len(df_ml_test)))
        print("events for signal train %d and test %d" % (len(df_sig_train), len(df_sig_test)))
        print("events for bkg train %d and test %d" % (len(df_bkg_train), len(df_bkg_test)))
        x_train = df_ml_train[var_training]
        y_train = df_ml_train[var_signal]

    if docorrelation == 1:
        vardistplot(df_sig_train, df_bkg_train, var_all, plotdir)
        scatterplot(df_sig_train, df_bkg_train, var_corr_x, var_corr_y, plotdir)
        correlationmatrix(df_sig_train, plotdir, "signal")
        correlationmatrix(df_bkg_train, plotdir, "background")

    if dostandard == 1:
        x_train = getdataframe_standardised(x_train)

    if dopca == 1:
        n_pca = 9
        x_train, pca = get_pcadataframe_pca(x_train, n_pca)
        plotvariance_pca(pca, plotdir)

    if activate_scikit == 1:
        classifiers_scikit, names_scikit = getclf_scikit(mltype)
        classifiers = classifiers+classifiers_scikit
        names = names+names_scikit

    if activate_xgboost == 1:
        classifiers_xgboost, names_xgboost = getclf_xgboost(mltype)
        classifiers = classifiers+classifiers_xgboost
        names = names+names_xgboost

    if activate_keras == 1:
        classifiers_keras, names_keras = getclf_keras(mltype, len(x_train.columns))
        classifiers = classifiers+classifiers_keras
        names = names+names_keras

    if dotraining == 1:
        trainedmodels = fit(names, classifiers, x_train, y_train)
        print(trainedmodels)
#         savemodels(names, trainedmodels, var_training, var_signal, output, suffix)


test()
