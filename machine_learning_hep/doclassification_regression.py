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

# from sklearn.utils import shuffle
# from sklearn.metrics import make_scorer, accuracy_score
from machine_learning_hep.general import get_database_ml_parameters, getdataframe
from machine_learning_hep.general import createstringselection, filterdataframe_singlevar
from machine_learning_hep.general import get_database_ml_gridsearch
from machine_learning_hep.root import write_tree
from machine_learning_hep.functions import create_mlsamples, do_correlation
from machine_learning_hep.io import parse_yaml, checkdir
from machine_learning_hep.config import assert_config, dump_default_config
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
from machine_learning_hep.optimization import countevents, study_signif

DATA_PREFIX = os.path.expanduser("~/.machine_learning_hep")

def doclassification_regression(config):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    logger = get_logger()
    logger.info(f"Start classification_regression run")

    mltype = config['mltype']
    mlsubtype = config['mlsubtype']
    case = config['case']
    loadsampleoption = config['loadsampleoption']
    binmin = config['binmin']
    binmax = config['binmax']
    rnd_shuffle = config['rnd_shuffle']
    nevt_sig = config['nevt_sig']
    nevt_bkg = config['nevt_bkg']
    test_frac = config['test_frac']
    rnd_splt = config['rnd_splt']
    docorrelation = config['docorrelation']
    dostandard = config['dostandard']
    dopca = config['dopca']
    activate_scikit = config['activate_scikit']
    activate_xgboost = config['activate_xgboost']
    activate_keras = config['activate_keras']
    dotraining = config['dotraining']
    dotesting = config['dotesting']
    applytodatamc = config['applytodatamc']
    docrossvalidation = config['docrossvalidation']
    dolearningcurve = config['dolearningcurve']
    doROC = config['doROC']
    doboundary = config['doboundary']
    doimportance = config['doimportance']
    dopltregressionxy = config['dopltregressionxy']
    dogridsearch = config['dogridsearch']
    dosignifopt = config['dosignifopt']
    nkfolds = config['nkfolds']
    ncores = config['ncores']

    data = get_database_ml_parameters()
    filesig, filebkg = data[case]["sig_bkg_files"]
    filedata, filemc = data[case]["data_mc_files"]
    trename = data[case]["tree_name"]
    var_all = data[case]["var_all"]
    var_signal = data[case]["var_signal"]
    sel_signal = data[case]["sel_signal"]
    sel_bkg = data[case]["sel_bkg"]
    var_training = data[case]["var_training"]
    var_target = data[case]["var_target"]
    var_corr_x, var_corr_y = data[case]["var_correlation"]
    var_boundaries = data[case]["var_boundaries"]
    var_binning = data[case]['var_binning']
    presel_gen = data[case]['presel_gen']
    presel_reco = data[case]["presel_reco"]
    mass_cut = data[case]["mass_cut"]
    var_gen = data[case]["var_gen"]
    ptgen = data[case]["ptgen"]
    treename_evt = data[case]["treename_evt"]
    treename_gen = data[case]["treename_gen"]
    var_evt = data[case]["var_evt"]
    sel_evt_counter = data[case]["sel_evt_counter"]
    sel_signal_signifopt = data[case]["sel_signal_signifopt"]
    sel_signal_signifopt_gen = data[case]["sel_signal_signifopt_gen"]

    summary_string = f"#sig events: {nevt_sig}\n#bkg events: {nevt_bkg}\nmltype: {mltype}\n" \
                     f"mlsubtype: {mlsubtype}\ncase: {case}"
    logger.debug(summary_string)

    string_selection = createstringselection(var_binning, binmin, binmax)
    suffix = f"nevt_sig{nevt_sig}_nevt_bkg{nevt_bkg}_" \
             f"{mltype}{case}_{string_selection}"
    dataframe = f"dataframes_{suffix}"
    plotdir = f"plots_{suffix}"
    output = f"output_{suffix}"
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

    filedata = os.path.join(DATA_PREFIX, filedata)
    filemc = os.path.join(DATA_PREFIX, filemc)

    trainedmodels = []

    if loadsampleoption == 1:
        df_sig = getdataframe(filesig, trename, var_all)
        df_bkg = getdataframe(filebkg, trename, var_all)
        if presel_reco is not None:
            df_sig = df_sig.query(presel_reco)
            df_bkg = df_bkg.query(presel_reco)
        df_sig = filterdataframe_singlevar(df_sig, var_binning, binmin, binmax)
        df_bkg = filterdataframe_singlevar(df_bkg, var_binning, binmin, binmax)
        _, df_ml_test, df_sig_train, df_bkg_train, _, _, \
        x_train, y_train, x_test, y_test = \
            create_mlsamples(df_sig, df_bkg, sel_signal, sel_bkg, rnd_shuffle,
                             var_signal, var_training,
                             nevt_sig, nevt_bkg, test_frac, rnd_splt)

    if docorrelation == 1:
        do_correlation(df_sig_train, df_bkg_train, var_all, var_corr_x, var_corr_y, plotdir)

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
        savemodels(names, trainedmodels, output, suffix)

    if dotesting == 1:
        # The model predictions are added to the test dataframe
        df_ml_test = test(mltype, names, trainedmodels, df_ml_test, var_training, var_signal)
        df_ml_test_to_df = output+"/testsample_%s_mldecision.pkl" % (suffix)
        df_ml_test_to_root = output+"/testsample_%s_mldecision.root" % (suffix)
        df_ml_test.to_pickle(df_ml_test_to_df)
        write_tree(df_ml_test_to_root, trename, df_ml_test)

    if applytodatamc == 1:
        df_data = getdataframe(filedata, trename, var_all)
        df_mc = getdataframe(filemc, trename, var_all)
        if presel_reco is not None:
            df_mc = df_mc.query(presel_reco)
            df_data = df_data.query(presel_reco)
        df_data = filterdataframe_singlevar(df_data, var_binning, binmin, binmax)
        df_mc = filterdataframe_singlevar(df_mc, var_binning, binmin, binmax)
        # The model predictions are added to the dataframes of data and MC
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
        classifiers_keras_2var, names_keras_2var = getclf_keras(mltype, 2)
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
            if (mlsubtype == "HFmeson") and (case == "Dsnew"):
                df_mc_gen = getdataframe(filemc, treename_gen, var_gen)
                df_mc_gen = df_mc_gen.query(presel_gen)
                df_mc_gen = filterdataframe_singlevar(df_mc_gen, ptgen, binmin, binmax)
                df_data_evt = getdataframe(filedata, treename_evt, var_evt)
                n_events = countevents(df_data_evt, sel_evt_counter)
                study_signif(case, names, binmin, binmax, df_mc_gen, df_mc, df_ml_test, df_data, \
                             n_events, sel_signal_signifopt, sel_signal_signifopt_gen, mass_cut, \
                             suffix, plotdir)
            else:
                logger.error("Optimisation is not implemented for this classification problem.")
        else:
            logger.error("Training, testing and applytodata flags must be set to 1")

def main():
    parser = argparse.ArgumentParser()
    # Require a config file with some plotting info
    parser.add_argument("-c", "--config", help="config YAML file, do -d <path> to get " \
                                               "YAML file with defaults at <path>")
    parser.add_argument("--dump-default-config", help="get default YAML config file")
    parser.add_argument("--debug", action="store_true", help="turn in debug information")
    parser.add_argument("--logfile", help="specify path to log file")

    args = parser.parse_args()

    configure_logger(args.debug, args.logfile)

    if args.dump_default_config is not None:
        dump_default_config(args.dump_default_config)
        sys.exit(0)

    user_config = {}
    if args.config is not None:
        user_config = parse_yaml(args.config)

    # Pass config dictionary
    doclassification_regression(assert_config(user_config))
