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
from machine_learning_hep.general import checkdir, write_tree
from machine_learning_hep.general import filterdataframe, split_df_sigbkg, createstringselection
from machine_learning_hep.preparesamples import prep_mlsamples
from machine_learning_hep.correlations import vardistplot, scatterplot, correlationmatrix
from machine_learning_hep.pca import getdataframe_standardised, get_pcadataframe_pca
from machine_learning_hep.pca import plotvariance_pca
from machine_learning_hep.models import getclf_scikit, getclf_xgboost, getclf_keras
from machine_learning_hep.models import fit, savemodels, test, apply, decisionboundaries
from machine_learning_hep.models import importanceplotall
from machine_learning_hep.mlperformance import cross_validation_mse, cross_validation_mse_continuous
from machine_learning_hep.mlperformance import plot_cross_validation_mse, plot_learning_curves
from machine_learning_hep.mlperformance import plotdistributiontarget, plotscattertarget
# from machine_learning_hep.mlperformance import confusion
from machine_learning_hep.mlperformance import precision_recall


def doclassification_regression():  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
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
    nkfolds = 5
    ncores = -1

    print(nevt_sig, nevt_bkg, mltype, mlsubtype, case)
    var_all = data[case]["var_all"]
    var_signal = data[case]["var_signal"]
    sel_signal = data[case]["sel_signal"]
    sel_bkg = data[case]["sel_bkg"]
    var_training = data[case]["var_training"]
    var_target = data[case]["var_target"]
    var_corr_x, var_corr_y = data[case]["var_correlation"]

    loadsampleoption = 1
    docorrelation = 0
    dostandard = 0
    dopca = 0
    activate_scikit = 0
    activate_xgboost = 1
    activate_keras = 0
    dotraining = 1
    dotesting = 1
    applytodatamc = 1
    docrossvalidation = 1
    dolearningcurve = 1
    doROC = 1
    doboundary = 1
    doimportance = 1
    dopltregressionxy = 0

    if mltype == "Regression":
        print("these tests cannot be performed for regression:")
        print("- doROCcurve")
        print("- doOptimisation")
        print("- doBinarySearch")
        print("- doBoundary")
        print("- doImportance")
        doROC = 0
        doboundary = 0
        activate_keras = 0
        doimportance = 0

    if mltype == "BinaryClassification":
        print("these tests cannot be performed for classification:")
        print("- plotdistributiontargetregression")
        dopltregressionxy = 0

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
    classifiers_scikit = []
    classifiers_xgboost = []
    classifiers_keras = []

    names = []
    names_scikit = []
    names_xgboost = []
    names_keras = []

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
        x_test = df_ml_test[var_training]
        y_test = df_ml_test[var_signal]

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
        savemodels(names, trainedmodels, var_training, var_signal, output, suffix)

    if dotesting == 1:
        df_ml_test_dec = test(mltype, names, trainedmodels, df_ml_test, var_training, var_signal)
        df_ml_test_dec_to_df = output+"/testsample_%s_mldecision.pkl" % (suffix)
        df_ml_test_dec_to_root = output+"/testsample_%s_mldecision.root" % (suffix)
        df_ml_test_dec.to_pickle(df_ml_test_dec_to_df)
        write_tree(df_ml_test_dec_to_root, trename, df_ml_test_dec)

    if applytodatamc == 1:
        filedata, filemc = data[case]["data_mc_files"]
        trename = data[case]["tree_name"]
        df_data = getdataframe(filedata, trename, var_all)
        df_mc = getdataframe(filemc, trename, var_all)
        df_data = filterdataframe(df_data, var_skimming, varmin, varmax)
        df_mc = filterdataframe(df_mc, var_skimming, varmin, varmax)
        df_data_dec = apply(mltype, names, trainedmodels, df_data, var_training)
        df_mc_dec = apply(mltype, names, trainedmodels, df_mc, var_training)
        df_data_dec_to_root = output+"/data_%s_mldecision.root" % (suffix)
        df_mc_dec_to_root = output+"/mc_%s_mldecision.root" % (suffix)
        write_tree(df_data_dec_to_root, trename, df_data_dec)
        write_tree(df_mc_dec_to_root, trename, df_mc_dec)

    if docrossvalidation == 1:
        df_scores = []
        if mltype == "Regression":
            df_scores = cross_validation_mse_continuous(
                names, classifiers, x_train, y_train, nkfolds, ncores)
        if mltype == "BinaryClassification":
            df_scores = cross_validation_mse(
                names, classifiers, x_train, y_train, nkfolds, ncores)
        plot_cross_validation_mse(names, df_scores, suffix, plotdir)

    if dolearningcurve == 1:
        #         confusion(names, classifiers, suffix, x_train, y_train, nkfolds, plotdir)
        npoints = 10
        plot_learning_curves(names, classifiers, suffix, plotdir, x_train, y_train, npoints)

    if doROC == 1:
        precision_recall(names, classifiers, suffix, x_train, y_train, nkfolds, plotdir)

    if doboundary == 1:
        classifiers_scikit_2var, names_2var = getclf_scikit(mltype)
        classifiers_keras_2var, names_keras_2var = getclf_keras(mltype, 2)
        classifiers_2var = classifiers_scikit_2var+classifiers_keras_2var
        names_2var = names_2var+names_keras_2var
        x_test_boundary = x_test[data[case]["var_boundaries"]]
        trainedmodels_2var = fit(names_2var, classifiers_2var, x_test_boundary, y_test)
        decisionboundaries(
            names_2var, trainedmodels_2var, suffix+"2var", x_test_boundary, y_test, plotdir)

    if doimportance == 1:
        importanceplotall(var_training, names_scikit+names_xgboost,
                          classifiers_scikit+classifiers_xgboost, suffix, plotdir)

    if dopltregressionxy == 1:
        plotdistributiontarget(names, df_ml_test, var_target, suffix, plotdir)
        plotscattertarget(names, df_ml_test, var_target, suffix, plotdir)


doclassification_regression()
