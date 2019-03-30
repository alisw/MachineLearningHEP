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
import os.path
import pandas as pd

from sklearn.utils import shuffle
from machine_learning_hep.general import createstringselection, filterdataframe_singlevar
from machine_learning_hep.general import get_database_ml_gridsearch, filter_df_cand
from machine_learning_hep.root import write_tree
from machine_learning_hep.functions import create_mlsamples, do_correlation
from machine_learning_hep.pca import getdataframe_standardised, get_pcadataframe_pca
from machine_learning_hep.pca import plotvariance_pca
from machine_learning_hep.models import getclf_scikit, getclf_xgboost, getclf_keras
from machine_learning_hep.models import fit, savemodels, test, apply, decisionboundaries
from machine_learning_hep.models import importanceplotall
from machine_learning_hep.mlperformance import cross_validation_mse, cross_validation_mse_continuous
from machine_learning_hep.mlperformance import plot_cross_validation_mse, plot_learning_curves
# from machine_learning_hep.mlperformance import confusion, plot_overtraining
from machine_learning_hep.mlperformance import precision_recall
from machine_learning_hep.grid_search import do_gridsearch, read_grid_dict, perform_plot_gridsearch
from machine_learning_hep.logger import get_logger
from machine_learning_hep.optimization import study_signif
from machine_learning_hep.efficiency import study_eff
DATA_PREFIX = os.path.expanduser("~/.machine_learning_hep")


def doclassification_regression(run_config, data, model_config, case, binmin, binmax):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    logger = get_logger()
    logger.info("Start classification_regression run")

    mltype = run_config['mltype']
    mlsubtype = run_config['mlsubtype']
    loadsampleoption = run_config['loadsampleoption']
    rnd_shuffle = run_config['rnd_shuffle']
    nevt_sig = run_config['nevt_sig']
    nevt_bkg = run_config['nevt_bkg']
    test_frac = run_config['test_frac']
    rnd_splt = run_config['rnd_splt']
    docorrelation = run_config['docorrelation']
    dostandard = run_config['dostandard']
    dopca = run_config['dopca']
    dotraining = run_config['dotraining']
    dotesting = run_config['dotesting']
    applytodatamc = run_config['applytodatamc']
    docrossvalidation = run_config['docrossvalidation']
    dolearningcurve = run_config['dolearningcurve']
    doROC = run_config['doROC']
    doboundary = run_config['doboundary']
    doimportance = run_config['doimportance']
    dogridsearch = run_config['dogridsearch']
    dosignifopt = run_config['dosignifopt']
    doefficiency = run_config['doefficiency']
    nkfolds = run_config['nkfolds']
    ncores = run_config['ncores']

    foldermc = data[case]["output_folders"]["pkl_merged"]["mc"]
    folderdata = data[case]["output_folders"]["pkl_merged"]["data"]
    #folder_evt_tot = data[case]["output_folders"]["pkl_final"]["data"]

    filedata = data[case]["files_names"]["namefile_reco_merged"]
    filemc = data[case]["files_names"]["namefile_reco_merged"]
    file_data_evt_ml = data[case]["files_names"]["namefile_evt_merged"]
    filemc_gen = data[case]["files_names"]["namefile_gen_merged"]
    file_data_evt_tot = data[case]["files_names"]["namefile_evt_skim_tot"]

    filedata = os.path.join(folderdata, filedata)
    filemc = os.path.join(foldermc, filemc)
    file_data_evt_ml = os.path.join(folderdata, file_data_evt_ml)
    filemc_gen = os.path.join(foldermc, filemc_gen)
    file_data_evt_tot = os.path.join(folderdata, file_data_evt_tot)

    arrayname = [filedata, filemc]

    sig_tag = data[case]["ml_study"]["prepare"]["sig_tag"]
    bkg_tag = data[case]["ml_study"]["prepare"]["bkg_tag"]
    filesig, filebkg = arrayname[sig_tag], arrayname[bkg_tag]

    sel_bkg = data[case]["ml_study"]["prepare"]["sel_bkg"]

    tree_name = data[case]["files_names"]["treeoutput"]
    var_all = data[case]["variables"]["var_all"]
    var_signal = data[case]["variables"]["var_signal"]
    var_training = data[case]["variables"]["var_training"]
    var_corr_x, var_corr_y = data[case]["variables"]["var_correlation"]
    var_boundaries = data[case]["variables"]["var_boundaries"]
    var_binning = data[case]["variables"]['var_binning']

    mlplot = data[case]["output_folders"]["mlplot"]
    mlout = data[case]["output_folders"]["mlout"]

    summary_string = f"#sig events: {nevt_sig}\n#bkg events: {nevt_bkg}\nmltype: {mltype}\n" \
                     f"mlsubtype: {mlsubtype}\ncase: {case}"
    logger.debug(summary_string)

    string_selection = createstringselection(var_binning, binmin, binmax)
    suffix = f"nevt_sig{nevt_sig}_nevt_bkg{nevt_bkg}_" \
             f"{mltype}{case}_{string_selection}"

    classifiers = []
    classifiers_scikit = []
    classifiers_xgboost = []
    classifiers_keras = []

    names = []
    names_scikit = []
    names_xgboost = []
    names_keras = []

#    filesig = os.path.join(DATA_PREFIX, filesig)
#    filebkg = os.path.join(DATA_PREFIX, filebkg)
#    filedata = os.path.join(DATA_PREFIX, filedata)
#    filemc = os.path.join(DATA_PREFIX, filemc)
#    filemc_gen = os.path.join(DATA_PREFIX, filemc_gen)
#    filedata_evt = os.path.join(DATA_PREFIX, filedata_evt)

    trainedmodels = []

    if loadsampleoption == 1:
        df_sig = pd.read_pickle(filesig)
        df_bkg = pd.read_pickle(filebkg)
        df_sig = filterdataframe_singlevar(df_sig, var_binning, binmin, binmax)
        df_bkg = filterdataframe_singlevar(df_bkg, var_binning, binmin, binmax)

        df_sig = filter_df_cand(df_sig, data[case], 'mc_signal')

        df_mc = pd.read_pickle(filemc)
        df_data = pd.read_pickle(filedata)
        df_data = filterdataframe_singlevar(df_data, var_binning, binmin, binmax)
        df_mc = filterdataframe_singlevar(df_mc, var_binning, binmin, binmax)

        _, df_ml_test, df_sig_train, df_bkg_train, _, _, \
        x_train, y_train, x_test, y_test = \
            create_mlsamples(df_sig, df_bkg, 'mc_signal', data[case], sel_bkg, rnd_shuffle,
                             var_signal, var_training, nevt_sig, nevt_bkg, test_frac, rnd_splt)

    if docorrelation == 1:
        do_correlation(df_sig_train, df_bkg_train, var_all, var_corr_x, var_corr_y, mlplot)

    if dostandard == 1:
        x_train = getdataframe_standardised(x_train)

    if dopca == 1:
        n_pca = 9
        x_train, pca = get_pcadataframe_pca(x_train, n_pca)
        plotvariance_pca(pca, mlplot)


    classifiers_scikit, names_scikit = getclf_scikit(model_config)

    classifiers_xgboost, names_xgboost = getclf_xgboost(model_config)

    classifiers_keras, names_keras = getclf_keras(model_config, len(x_train.columns))

    classifiers = classifiers_scikit+classifiers_xgboost+classifiers_keras
    names = names_scikit+names_xgboost+names_keras
    print(names)
    if dotraining == 1:
        trainedmodels = fit(names, classifiers, x_train, y_train)
        savemodels(names, trainedmodels, mlout, suffix)

    if dotesting == 1:
        # The model predictions are added to the test dataframe
        df_ml_test = test(mltype, names, trainedmodels, df_ml_test, var_training, var_signal)
        df_ml_test_to_df = mlout+"/testsample_%s_mldecision.pkl" % (suffix)
        df_ml_test_to_root = mlout+"/testsample_%s_mldecision.root" % (suffix)
        df_ml_test.to_pickle(df_ml_test_to_df)
        write_tree(df_ml_test_to_root, tree_name, df_ml_test)
        #plot_overtraining(names, classifiers, suffix, mlplot, x_train, y_train, x_test, y_test)

    if applytodatamc == 1:
        # The model predictions are added to the dataframes of data and MC
        df_data = apply(mltype, names, trainedmodels, df_data, var_training)
        df_mc = apply(mltype, names, trainedmodels, df_mc, var_training)
        df_data_to_root = mlout+"/data_%s_mldecision.root" % (suffix)
        df_mc_to_root = mlout+"/mc_%s_mldecision.root" % (suffix)
        write_tree(df_data_to_root, tree_name, df_data)
        write_tree(df_mc_to_root, tree_name, df_mc)

    if docrossvalidation == 1:
        df_scores = []
        if mltype == "Regression":
            df_scores = cross_validation_mse_continuous(
                names, classifiers, x_train, y_train, nkfolds, ncores)
        if mltype == "BinaryClassification":
            df_scores = cross_validation_mse(names, classifiers, x_train, y_train,
                                             nkfolds, ncores)
        plot_cross_validation_mse(names, df_scores, suffix, mlplot)

    if dolearningcurve == 1:
        #confusion(names, classifiers, suffix, x_train, y_train, nkfolds, mlplot)
        npoints = 10
        plot_learning_curves(names, classifiers, suffix, mlplot, x_train, y_train, npoints)

    if doROC == 1:
        precision_recall(names, classifiers, suffix, x_train, y_train, nkfolds, mlplot)

    if doboundary == 1:
        classifiers_scikit_2var, names_2var = getclf_scikit(mltype)
        classifiers_keras_2var, names_keras_2var = getclf_keras(data["ml_study"], 2)
        classifiers_2var = classifiers_scikit_2var+classifiers_keras_2var
        names_2var = names_2var+names_keras_2var
        x_test_boundary = x_test[var_boundaries]
        trainedmodels_2var = fit(names_2var, classifiers_2var, x_test_boundary, y_test)
        decisionboundaries(
            names_2var, trainedmodels_2var, suffix+"2var", x_test_boundary, y_test, mlplot)

    if doimportance == 1:
        importanceplotall(var_training, names_scikit+names_xgboost,
                          classifiers_scikit+classifiers_xgboost, suffix, mlplot)

    if dogridsearch == 1:
        datasearch = get_database_ml_gridsearch()
        analysisdb = datasearch[mltype]
        names_cv, clf_cv, par_grid_cv, refit_cv, var_param, \
            par_grid_cv_keys = read_grid_dict(analysisdb)
        _, _, dfscore = do_gridsearch(
            names_cv, clf_cv, par_grid_cv, refit_cv, x_train, y_train, nkfolds,
            ncores)
        perform_plot_gridsearch(
            names_cv, dfscore, par_grid_cv, par_grid_cv_keys, var_param, mlplot, suffix, 0.1)

    if dosignifopt == 1:
        logger.info("Doing significance optimization")
        if dotraining and dotesting and applytodatamc:
            if mlsubtype == "HFmeson":
                df_data_opt = df_data.query(sel_bkg)
                df_data_opt = shuffle(df_data_opt, random_state=rnd_shuffle)
                study_signif(case, names, [binmin, binmax], filemc_gen, file_data_evt_ml,
                             file_data_evt_tot, df_mc, df_ml_test, df_data_opt, suffix, mlplot)
            else:
                logger.error("Optimisation is not implemented for this classification problem.")
        else:
            logger.error("Training, testing and applytodata flags must be set to True")

    if doefficiency == 1:
        logger.info("Doing selection efficiency of ML models")
        if dotraining and dotesting:
            study_eff(case, names, suffix, mlplot, df_ml_test)
        else:
            logger.error("Training and testing flags must be set to True")
