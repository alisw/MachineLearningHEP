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
import argparse
import sys
import os.path

import pickle
from sklearn.utils import shuffle
# from sklearn.metrics import make_scorer, accuracy_score
from machine_learning_hep.general import get_database_ml_parameters, getdataframe
from machine_learning_hep.general import createstringselection, filterdataframe_singlevar
from machine_learning_hep.general import get_database_ml_gridsearch, filter_df_cand
from machine_learning_hep.root import write_tree
from machine_learning_hep.functions import create_mlsamples, do_correlation
from machine_learning_hep.io import checkdir
from machine_learning_hep.config import Configuration
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
from machine_learning_hep.grid_search import do_gridsearch, read_grid_dict, perform_plot_gridsearch
from machine_learning_hep.logger import configure_logger, get_logger
from machine_learning_hep.optimization import study_signif
DATA_PREFIX = os.path.expanduser("~/.machine_learning_hep")


def doclassification_regression(conf):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    logger = get_logger()
    #logger.info(f"Start classification_regression run")

    run_config = conf.get_run_config()
    model_config = conf.get_model_config()

    mltype = run_config['mltype']
    mlsubtype = run_config['mlsubtype']
    case = run_config['case']
    loadsampleoption = run_config['loadsampleoption']
    usesampleoption = run_config['usesampleoption']
    binmin = run_config['binmin']
    binmax = run_config['binmax']
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
    dopltregressionxy = run_config['dopltregressionxy']
    dogridsearch = run_config['dogridsearch']
    dosignifopt = run_config['dosignifopt']
    nkfolds = run_config['nkfolds']
    ncores = run_config['ncores']
    usefileserver = run_config['usefileserver']

    data = get_database_ml_parameters()

    filesig, filebkg = data[case]["sig_bkg_files"]
    filedata, filemc = data[case]["data_mc_files"]
    if usefileserver is True:
        filesig, filebkg = data[case]["sig_bkg_files_server"]
        filedata, filemc = data[case]["data_mc_files_server"]
    trename = data[case]["tree_name"]
    var_all = data[case]["var_all"]
    var_signal = data[case]["var_signal"]
    sel_bkg = data[case]["sel_bkg"]
    var_training = data[case]["var_training"]
    var_target = data[case]["var_target"]
    var_corr_x, var_corr_y = data[case]["var_correlation"]
    var_boundaries = data[case]["var_boundaries"]
    var_binning = data[case]['var_binning']
    presel_reco = data[case]["presel_reco"]
    summary_string = f"#sig events: {nevt_sig}\n#bkg events: {nevt_bkg}\nmltype: {mltype}\n" \
                     f"mlsubtype: {mlsubtype}\ncase: {case}"
    logger.debug(summary_string)

    string_selection = createstringselection(var_binning, binmin, binmax)
    suffix = f"nevt_sig{nevt_sig}_nevt_bkg{nevt_bkg}_" \
             f"{mltype}{case}_{string_selection}"
    dataframe = f"dataframepkl"
    plotdir = f"plots_{suffix}"
    output = f"output_{suffix}"

    dataframesig = f"{dataframe}/dfsignal.pkl"
    dataframebkg = f"{dataframe}/dfbackground.pkl"
    dataframedata = f"{dataframe}/dfdata.pkl"
    dataframemc = f"{dataframe}/dfmc.pkl"

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

    filesig = os.path.join(DATA_PREFIX, filesig)
    filebkg = os.path.join(DATA_PREFIX, filebkg)
    filedata = os.path.join(DATA_PREFIX, filedata)
    filemc = os.path.join(DATA_PREFIX, filemc)

    trainedmodels = []

    if loadsampleoption == 1:
        df_sig = getdataframe(filesig, trename, var_all)
        df_bkg = getdataframe(filebkg, trename, var_all)
        df_sig = filterdataframe_singlevar(df_sig, var_binning, binmin, binmax)
        df_bkg = filterdataframe_singlevar(df_bkg, var_binning, binmin, binmax)

        if presel_reco is not None:
            df_sig = df_sig.query(presel_reco)
            df_bkg = df_bkg.query(presel_reco)
        df_sig = filter_df_cand(df_sig, data[case], 'presel_track_pid')
        df_bkg = filter_df_cand(df_bkg, data[case], 'presel_track_pid')
        df_sig = filter_df_cand(df_sig, data[case], 'mc_signal')

        df_data = getdataframe(filedata, trename, var_all)
        df_mc = getdataframe(filemc, trename, var_all)
        df_data = filterdataframe_singlevar(df_data, var_binning, binmin, binmax)
        df_mc = filterdataframe_singlevar(df_mc, var_binning, binmin, binmax)

        if presel_reco is not None:
            df_mc = df_mc.query(presel_reco)
            df_data = df_data.query(presel_reco)
        df_mc = filter_df_cand(df_mc, data[case], 'presel_track_pid')
        df_data = filter_df_cand(df_data, data[case], 'presel_track_pid')

        df_sig.to_pickle(dataframesig)
        df_bkg.to_pickle(dataframebkg)
        df_mc.to_pickle(dataframemc)
        df_data.to_pickle(dataframedata)

    if usesampleoption == 1:
        filesaved_sig = open(dataframesig, "rb")
        filesaved_bkg = open(dataframebkg, "rb")
        df_sig = pickle.load(filesaved_sig)
        df_bkg = pickle.load(filesaved_bkg)
        _, df_ml_test, df_sig_train, df_bkg_train, _, _, \
        x_train, y_train, x_test, y_test = \
            create_mlsamples(df_sig, df_bkg, 'mc_signal', data[case], sel_bkg, rnd_shuffle,
                             var_signal, var_training, nevt_sig, nevt_bkg, test_frac, rnd_splt)

    if docorrelation == 1:
        do_correlation(df_sig_train, df_bkg_train, var_all, var_corr_x, var_corr_y, plotdir)

    if dostandard == 1:
        x_train = getdataframe_standardised(x_train)

    if dopca == 1:
        n_pca = 9
        x_train, pca = get_pcadataframe_pca(x_train, n_pca)
        plotvariance_pca(pca, plotdir)



    classifiers_scikit, names_scikit = getclf_scikit(model_config)

    classifiers_xgboost, names_xgboost = getclf_xgboost(model_config)

    classifiers_keras, names_keras = getclf_keras(model_config, len(x_train.columns))

    classifiers = classifiers_scikit+classifiers_xgboost+classifiers_keras
    names = names_scikit+names_xgboost+names_keras


    if dotraining == 1:
        trainedmodels = fit(names, classifiers, x_train, y_train)
        savemodels(names, trainedmodels, output, suffix)

    if dotesting == 1:
        # The model predictions are added to the test dataframe
        df_ml_test = test(mltype, names, trainedmodels, df_ml_test, var_training, var_signal)
        df_ml_test_to_df = output+"/testsample_%s_mldecision.pkl" % (suffix)
        df_ml_test_to_root = output+"/testsample_%s_mldecision.root" % (suffix)
        df_ml_test.to_pickle(df_ml_test_to_df)
        write_tree(df_ml_test_to_root, trename, df_ml_test)

    if applytodatamc == 1:
        # The model predictions are added to the dataframes of data and MC
        filesaved_mc = open(dataframemc, "rb")
        filesaved_data = open(dataframedata, "rb")
        df_data = pickle.load(filesaved_data)
        df_mc = pickle.load(filesaved_mc)
        df_data = apply(mltype, names, trainedmodels, df_data, var_training)
        df_mc = apply(mltype, names, trainedmodels, df_mc, var_training)
        df_data_to_root = output+"/data_%s_mldecision.root" % (suffix)
        df_mc_to_root = output+"/mc_%s_mldecision.root" % (suffix)
        write_tree(df_data_to_root, trename, df_data)
        write_tree(df_mc_to_root, trename, df_mc)

    if docrossvalidation == 1:
        df_scores = []
        if mltype == "Regression":
            df_scores = cross_validation_mse_continuous(
                names, classifiers, x_train, y_train, nkfolds, ncores)
        if mltype == "BinaryClassification":
            df_scores = cross_validation_mse(names, classifiers, x_train, y_train,
                                             nkfolds, ncores)
        plot_cross_validation_mse(names, df_scores, suffix, plotdir)

    if dolearningcurve == 1:
        #         confusion(names, classifiers, suffix, x_train, y_train, nkfolds, plotdir)
        npoints = 10
        plot_learning_curves(names, classifiers, suffix, plotdir, x_train, y_train, npoints)

    if doROC == 1:
        precision_recall(names, classifiers, suffix, x_train, y_train, nkfolds, plotdir)

    if doboundary == 1:
        classifiers_scikit_2var, names_2var = getclf_scikit(mltype)
        classifiers_keras_2var, names_keras_2var = getclf_keras(model_config, 2)
        classifiers_2var = classifiers_scikit_2var+classifiers_keras_2var
        names_2var = names_2var+names_keras_2var
        x_test_boundary = x_test[var_boundaries]
        trainedmodels_2var = fit(names_2var, classifiers_2var, x_test_boundary, y_test)
        decisionboundaries(
            names_2var, trainedmodels_2var, suffix+"2var", x_test_boundary, y_test, plotdir)

    if doimportance == 1:
        importanceplotall(var_training, names_scikit+names_xgboost,
                          classifiers_scikit+classifiers_xgboost, suffix, plotdir)

    if dopltregressionxy == 1:
        plotdistributiontarget(names, df_ml_test, var_target, suffix, plotdir)
        plotscattertarget(names, df_ml_test, var_target, suffix, plotdir)

    if dogridsearch == 1:
        datasearch = get_database_ml_gridsearch()
        analysisdb = datasearch[mltype]
        names_cv, clf_cv, par_grid_cv, refit_cv, var_param, \
            par_grid_cv_keys = read_grid_dict(analysisdb)
        _, _, dfscore = do_gridsearch(
            names_cv, clf_cv, par_grid_cv, refit_cv, x_train, y_train, nkfolds,
            ncores)
        perform_plot_gridsearch(
            names_cv, dfscore, par_grid_cv, par_grid_cv_keys, var_param, plotdir, suffix, 0.1)

    if dosignifopt == 1:
        logger.info("Doing significance optimization")
        if dotraining and dotesting and applytodatamc:
            if mlsubtype == "HFmeson":
                df_data_opt = df_data.query(sel_bkg)
                df_data_opt = shuffle(df_data_opt, random_state=rnd_shuffle)
                study_signif(case, names, [binmin, binmax], filemc, filedata, df_mc, df_ml_test,
                             df_data_opt, suffix, plotdir)
            else:
                logger.error("Optimisation is not implemented for this classification problem.")
        else:
            logger.error("Training, testing and applytodata flags must be set to 1")

def main():
    """
    Parse and handle arguments and dispatch to central function doclassification_regression.
    This includes following steps:
        1) Configure the logger
        2) Dump default configuration YAMLs if requested (exit afterwards assuming the user wants
                                                         to edit and use them later)
        3) Steer doclassification_regression with extracted configuration
    """
    parser = argparse.ArgumentParser()
    # Require a config file with some plotting info
    parser.add_argument("--dump-default-config", dest="dump_default_config",
                        help="get default run parameters as YAML config file")
    parser.add_argument("--dump-default-models", dest="dump_default_models",
                        help="get default model parameters as YAML config file")
    parser.add_argument("-c", "--load-run-config",
                        help="specify YAML file with run configuration to be loaded")
    parser.add_argument("-m", "--load-model-config",
                        help="specify YAML file with model configuration to be loaded")
    parser.add_argument("--debug", action="store_true", help="turn in debug information")
    parser.add_argument("--logfile", help="specify path to log file")

    args = parser.parse_args()

    configure_logger(args.debug, args.logfile)

    immediate_exit = False
    for k, v in {"run": args.dump_default_config, "models": args.dump_default_models}.items():
        if v is not None:
            Configuration.dump_default_config(k, v)
            immediate_exit = True
    if immediate_exit:
        sys.exit(0)

    #model_config = assert_model_config(args.load_model_config, run_config)

    conf = Configuration(args.load_run_config, args.load_model_config)
    conf.configure()

    conf.print_configuration()


    # Pass config dictionary
    doclassification_regression(conf)
