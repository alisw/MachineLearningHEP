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
main script for doing ml optisation
"""
import os
import time
from math import sqrt
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from ROOT import TFile, TCanvas, TH1F, TF1, gROOT  # pylint: disable=import-error,no-name-in-module
from machine_learning_hep.utilities import seldf_singlevar, split_df_sigbkg, createstringselection
from machine_learning_hep.utilities import openfile, selectdfquery
from machine_learning_hep.correlations import vardistplot, scatterplot, correlationmatrix
from machine_learning_hep.models import getclf_scikit, getclf_xgboost, getclf_keras
from machine_learning_hep.models import fit, savemodels, test, apply, decisionboundaries
from machine_learning_hep.root import write_tree
from machine_learning_hep.mlperformance import cross_validation_mse, plot_cross_validation_mse
from machine_learning_hep.mlperformance import plot_learning_curves, precision_recall
from machine_learning_hep.mlperformance import roc_train_test, plot_overtraining
from machine_learning_hep.optimisation.grid_search import do_gridsearch, perform_plot_gridsearch
from machine_learning_hep.models import importanceplotall, shap_study
from machine_learning_hep.logger import get_logger
from machine_learning_hep.optimization import calc_bkg, calc_signif, calc_eff, calc_sigeff_steps
from machine_learning_hep.correlations import vardistplot_probscan, efficiency_cutscan
from machine_learning_hep.utilities import checkdirlist, checkmakedirlist

# pylint: disable=too-many-instance-attributes, too-many-statements
class Optimiser:
    #Class Attribute
    species = "optimiser"

    def __init__(self, data_param, case, typean, model_config, binmin,
                 binmax, raahp, training_var):

        self.logger = get_logger()

        dirmcml = data_param["multi"]["mc"]["pkl_skimmed_merge_for_ml_all"]
        dirdataml = data_param["multi"]["data"]["pkl_skimmed_merge_for_ml_all"]
        dirdatatotsample = data_param["multi"]["data"]["pkl_evtcounter_all"]
        self.v_bin = data_param["var_binning"]
        #directory
        self.dirmlout = data_param["ml"]["mlout"]
        self.dirmlplot = data_param["ml"]["mlplot"]
        #ml file names
        self.n_reco = data_param["files_names"]["namefile_reco"]
        self.n_reco = self.n_reco.replace(".pkl", "_%s%d_%d.pkl" % (self.v_bin, binmin, binmax))
        self.n_evt = data_param["files_names"]["namefile_evt"]
        self.n_gen = data_param["files_names"]["namefile_gen"]
        self.n_gen = self.n_gen.replace(".pkl", "_%s%d_%d.pkl" % (self.v_bin, binmin, binmax))
        self.n_treetest = data_param["files_names"]["treeoutput"]
        self.n_reco_applieddata = data_param["files_names"]["namefile_reco_applieddata"]
        self.n_reco_appliedmc = data_param["files_names"]["namefile_reco_appliedmc"]
        # ml files
        self.f_gen_mc = os.path.join(dirmcml, self.n_gen)
        self.f_reco_mc = os.path.join(dirmcml, self.n_reco)
        self.f_evt_mc = os.path.join(dirmcml, self.n_evt)
        self.f_reco_data = os.path.join(dirdataml, self.n_reco)
        self.f_evt_data = os.path.join(dirdataml, self.n_evt)
        self.f_evttotsample_data = os.path.join(dirdatatotsample, self.n_evt)
        self.f_reco_applieddata = os.path.join(self.dirmlout, self.n_reco_applieddata)
        self.f_reco_appliedmc = os.path.join(self.dirmlout, self.n_reco_appliedmc)
        #variables
        self.v_all = data_param["variables"]["var_all"]
        self.v_train = training_var
        self.v_bound = data_param["variables"]["var_boundaries"]
        self.v_sig = data_param["variables"]["var_signal"]
        self.v_invmass = data_param["variables"]["var_inv_mass"]
        self.v_cuts = data_param["variables"].get("var_cuts", [])
        self.v_corrx = data_param["variables"]["var_correlation"][0]
        self.v_corry = data_param["variables"]["var_correlation"][1]
        self.v_isstd = data_param["bitmap_sel"]["var_isstd"]
        self.v_ismcsignal = data_param["bitmap_sel"]["var_ismcsignal"]
        self.v_ismcprompt = data_param["bitmap_sel"]["var_ismcprompt"]
        self.v_ismcfd = data_param["bitmap_sel"]["var_ismcfd"]
        self.v_ismcbkg = data_param["bitmap_sel"]["var_ismcbkg"]
        #parameters
        self.p_case = case
        self.p_typean = typean
        self.p_nbkg = data_param["ml"]["nbkg"]
        self.p_nsig = data_param["ml"]["nsig"]
        self.p_tagsig = data_param["ml"]["sampletagforsignal"]
        self.p_tagbkg = data_param["ml"]["sampletagforbkg"]
        self.p_binmin = binmin
        self.p_binmax = binmax
        self.p_npca = None
        self.p_mltype = data_param["ml"]["mltype"]
        self.p_nkfolds = data_param["ml"]["nkfolds"]
        self.p_ncorescross = data_param["ml"]["ncorescrossval"]
        self.rnd_shuffle = data_param["ml"]["rnd_shuffle"]
        self.rnd_splt = data_param["ml"]["rnd_splt"]
        self.test_frac = data_param["ml"]["test_frac"]
        self.p_plot_options = data_param["variables"].get("plot_options", {})
        self.p_dofullevtmerge = data_param["dofullevtmerge"]

        self.p_evtsel = data_param["ml"]["evtsel"]
        self.p_triggersel_mc = data_param["ml"]["triggersel"]["mc"]
        self.p_triggersel_data = data_param["ml"]["triggersel"]["data"]

        #dataframes
        self.df_mc = None
        self.df_mcgen = None
        self.df_data = None
        self.df_sig = None
        self.df_bkg = None
        self.df_ml = None
        self.df_mltest = None
        self.df_mltrain = None
        self.df_sigtrain = None
        self.df_sigtest = None
        self.df_bkgtrain = None
        self.df_bktest = None
        self.df_xtrain = None
        self.df_ytrain = None
        self.df_xtest = None
        self.df_ytest = None
        #selections
        self.s_selbkgml = data_param["ml"]["sel_bkgml"]
        self.s_selsigml = data_param["ml"]["sel_sigml"]
        #model param
        self.db_model = model_config
        self.p_class = None
        self.p_classname = None
        self.p_trainedmod = None
        self.s_suffix = None

        #significance
        self.is_fonll_from_root = data_param["ml"]["opt"]["isFONLLfromROOT"]
        self.f_fonll = data_param["ml"]["opt"]["filename_fonll"]
        if self.is_fonll_from_root and "fonll_particle" not in data_param["ml"]["opt"]:
            self.logger.fatal("Attempt to read FONLL from ROOT file but field " \
                    "\"fonll_particle\" not provided in database")
        self.p_fonllparticle = data_param["ml"]["opt"].get("fonll_particle", "")
        self.p_fonllband = data_param["ml"]["opt"]["fonll_pred"]
        self.p_fragf = data_param["ml"]["opt"]["FF"]
        self.p_sigmamb = data_param["ml"]["opt"]["sigma_MB"]
        self.p_taa = data_param["ml"]["opt"]["Taa"]
        self.p_br = data_param["ml"]["opt"]["BR"]
        self.p_fprompt = data_param["ml"]["opt"]["f_prompt"]
        self.p_bkgfracopt = data_param["ml"]["opt"]["bkg_data_fraction"]
        self.p_nstepsign = data_param["ml"]["opt"]["num_steps"]
        self.p_bkg_func = data_param["ml"]["opt"]["bkg_function"]
        self.p_savefit = data_param["ml"]["opt"]["save_fit"]
        self.p_nevtml = None
        self.p_nevttot = None
        self.p_presel_gen_eff = data_param["ml"]["opt"]["presel_gen_eff"]
        self.p_mass_fit_lim = data_param["analysis"][self.p_typean]['mass_fit_lim']
        self.p_bin_width = data_param["analysis"][self.p_typean]['bin_width']
        self.p_num_bins = int(round((self.p_mass_fit_lim[1] - self.p_mass_fit_lim[0]) / \
                                     self.p_bin_width))
        self.p_mass = data_param["mass"]
        self.p_raahp = raahp
        self.preparesample()
        self.loadmodels()
        self.create_suffix()
        self.df_evt_data = None
        self.df_evttotsample_data = None

        self.f_reco_applieddata = \
                self.f_reco_applieddata.replace(".pkl", "%s.pkl" % self.s_suffix)
        self.f_reco_appliedmc = \
                self.f_reco_appliedmc.replace(".pkl", "%s.pkl" % self.s_suffix)


    def create_suffix(self):
        string_selection = createstringselection(self.v_bin, self.p_binmin, self.p_binmax)
        self.s_suffix = f"{self.p_case}_{string_selection}"

    def preparesample(self):
        self.logger.info("Prepare Sample")
        self.df_data = pickle.load(openfile(self.f_reco_data, "rb"))
        self.df_mc = pickle.load(openfile(self.f_reco_mc, "rb"))
        self.df_mcgen = pickle.load(openfile(self.f_gen_mc, "rb"))
        self.df_data = selectdfquery(self.df_data, self.p_evtsel)
        self.df_mc = selectdfquery(self.df_mc, self.p_evtsel)
        self.df_mcgen = selectdfquery(self.df_mcgen, self.p_evtsel)

        self.df_data = selectdfquery(self.df_data, self.p_triggersel_data)
        self.df_mc = selectdfquery(self.df_mc, self.p_triggersel_mc)
        self.df_mcgen = selectdfquery(self.df_mcgen, self.p_triggersel_mc)

        self.df_mcgen = self.df_mcgen.query(self.p_presel_gen_eff)
        arraydf = [self.df_data, self.df_mc]
        self.df_mc = seldf_singlevar(self.df_mc, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_mcgen = seldf_singlevar(self.df_mcgen, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_data = seldf_singlevar(self.df_data, self.v_bin, self.p_binmin, self.p_binmax)


        self.df_sig, self.df_bkg = arraydf[self.p_tagsig], arraydf[self.p_tagbkg]
        self.df_sig = seldf_singlevar(self.df_sig, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_bkg = seldf_singlevar(self.df_bkg, self.v_bin, self.p_binmin, self.p_binmax)
        self.df_sig = self.df_sig.query(self.s_selsigml)
        self.df_bkg = self.df_bkg.query(self.s_selbkgml)
        self.df_bkg["ismcsignal"] = 0
        self.df_bkg["ismcprompt"] = 0
        self.df_bkg["ismcfd"] = 0
        self.df_bkg["ismcbkg"] = 0


        if self.p_nsig > len(self.df_sig):
            self.logger.warning("There are not enough signal events")
        if self.p_nbkg > len(self.df_bkg):
            self.logger.warning("There are not enough background events")

        self.p_nsig = min(len(self.df_sig), self.p_nsig)
        self.p_nbkg = min(len(self.df_bkg), self.p_nbkg)

        self.logger.info("Used number of signal events is %d", self.p_nsig)
        self.logger.info("Used number of background events is %d", self.p_nbkg)

        self.df_ml = pd.DataFrame()
        self.df_sig = shuffle(self.df_sig, random_state=self.rnd_shuffle)
        self.df_bkg = shuffle(self.df_bkg, random_state=self.rnd_shuffle)
        self.df_sig = self.df_sig[:self.p_nsig]
        self.df_bkg = self.df_bkg[:self.p_nbkg]
        self.df_sig[self.v_sig] = 1
        self.df_bkg[self.v_sig] = 0
        self.df_ml = pd.concat([self.df_sig, self.df_bkg])
        self.df_mltrain, self.df_mltest = train_test_split(self.df_ml, \
                                           test_size=self.test_frac, random_state=self.rnd_splt)
        self.df_mltrain = self.df_mltrain.reset_index(drop=True)
        self.df_mltest = self.df_mltest.reset_index(drop=True)
        self.df_sigtrain, self.df_bkgtrain = split_df_sigbkg(self.df_mltrain, self.v_sig)
        self.df_sigtest, self.df_bkgtest = split_df_sigbkg(self.df_mltest, self.v_sig)
        self.logger.info("Total number of candidates: train %d and test %d", len(self.df_mltrain),
                         len(self.df_mltest))
        self.logger.info("Number of signal candidates: train %d and test %d",
                         len(self.df_sigtrain), len(self.df_sigtest))
        self.logger.info("Number of bkg candidates: %d and test %d", len(self.df_bkgtrain),
                         len(self.df_bkgtest))

        self.df_xtrain = self.df_mltrain[self.v_train]
        self.df_ytrain = self.df_mltrain[self.v_sig]
        self.df_xtest = self.df_mltest[self.v_train]
        self.df_ytest = self.df_mltest[self.v_sig]

    def do_corr(self):
        imageIO_vardist_all = vardistplot(self.df_sigtrain, self.df_bkgtrain,
                                          self.v_all, self.dirmlplot,
                                          self.p_binmin, self.p_binmax)
        imageIO_vardist_train = vardistplot(self.df_sigtrain, self.df_bkgtrain,
                                            self.v_train, self.dirmlplot,
                                            self.p_binmin, self.p_binmax)
        imageIO_scatterplot = scatterplot(self.df_sigtrain, self.df_bkgtrain,
                                          self.v_corrx, self.v_corry,
                                          self.dirmlplot, self.p_binmin, self.p_binmax)
        imageIO_corr_sig_all = correlationmatrix(self.df_sigtrain, self.v_all, "Signal",
                                                 self.dirmlplot, self.p_binmin, self.p_binmax)
        imageIO_corr_bkg_all = correlationmatrix(self.df_bkgtrain, self.v_all, "Background",
                                                 self.dirmlplot, self.p_binmin, self.p_binmax)
        imageIO_corr_sig_train = correlationmatrix(self.df_sigtrain, self.v_train, "Signal",
                                                   self.dirmlplot, self.p_binmin, self.p_binmax)
        imageIO_corr_bkg_train = correlationmatrix(self.df_bkgtrain, self.v_train, "Background",
                                                   self.dirmlplot, self.p_binmin, self.p_binmax)
        return imageIO_vardist_all, imageIO_vardist_train, imageIO_scatterplot, \
                  imageIO_corr_sig_all, imageIO_corr_bkg_all, imageIO_corr_sig_train, \
                  imageIO_corr_bkg_train

    def loadmodels(self):
        classifiers_scikit, names_scikit, _, _ = getclf_scikit(self.db_model)
        classifiers_xgboost, names_xgboost, _, _ = getclf_xgboost(self.db_model)
        classifiers_keras, names_keras, _, _ = getclf_keras(self.db_model,
                                                            len(self.df_xtrain.columns))
        self.p_class = classifiers_scikit+classifiers_xgboost+classifiers_keras
        self.p_classname = names_scikit+names_xgboost+names_keras

    def do_train(self):
        self.logger.info("Training")
        t0 = time.time()
        self.p_trainedmod = fit(self.p_classname, self.p_class, self.df_xtrain, self.df_ytrain)
        savemodels(self.p_classname, self.p_trainedmod, self.dirmlout, self.s_suffix)
        self.logger.info("Training over")
        self.logger.info("Time elapsed = %.3f", time.time() - t0)

    def do_test(self):
        df_ml_test = test(self.p_mltype, self.p_classname, self.p_trainedmod,
                          self.df_mltest, self.v_train, self.v_sig)
        df_ml_test_to_df = self.dirmlout+"/testsample_%s_mldecision.pkl" % (self.s_suffix)
        df_ml_test_to_root = self.dirmlout+"/testsample_%s_mldecision.root" % (self.s_suffix)
        pickle.dump(df_ml_test, openfile(df_ml_test_to_df, "wb"), protocol=4)
        write_tree(df_ml_test_to_root, self.n_treetest, df_ml_test)

    def do_apply(self):
        df_data = apply(self.p_mltype, self.p_classname, self.p_trainedmod,
                        self.df_data, self.v_train)
        df_mc = apply(self.p_mltype, self.p_classname, self.p_trainedmod,
                      self.df_mc, self.v_train)
        pickle.dump(df_data, openfile(self.f_reco_applieddata, "wb"), protocol=4)
        pickle.dump(df_mc, openfile(self.f_reco_appliedmc, "wb"), protocol=4)

    def do_crossval(self):
        df_scores = cross_validation_mse(self.p_classname, self.p_class,
                                         self.df_xtrain, self.df_ytrain,
                                         self.p_nkfolds, self.p_ncorescross)
        plot_cross_validation_mse(self.p_classname, df_scores, self.s_suffix, self.dirmlplot)

    def do_learningcurve(self):
        npoints = 10
        plot_learning_curves(self.p_classname, self.p_class, self.s_suffix,
                             self.dirmlplot, self.df_xtrain, self.df_ytrain, npoints)

    def do_roc(self):
        precision_recall(self.p_classname, self.p_class, self.s_suffix,
                         self.df_xtrain, self.df_ytrain, self.p_nkfolds, self.dirmlplot)

    def do_roc_train_test(self):
        roc_train_test(self.p_classname, self.p_class, self.df_xtrain, self.df_ytrain,
                       self.df_xtest, self.df_ytest, self.s_suffix, self.dirmlplot)

    def do_plot_model_pred(self):
        plot_overtraining(self.p_classname, self.p_class, self.s_suffix, self.dirmlplot,
                          self.df_xtrain, self.df_ytrain, self.df_xtest, self.df_ytest)

    def do_importance(self):
        importanceplotall(self.v_train, self.p_classname, self.p_class,
                          self.s_suffix, self.dirmlplot)

    def do_importance_shap(self):
        shap_study(self.p_classname, self.p_class, self.df_xtrain, self.s_suffix, self.dirmlplot)

    def do_bayesian_opt(self):
        self.logger.info("Do Bayesian optimisation for all classifiers")
        _, names_scikit, _, bayes_opt_scikit = getclf_scikit(self.db_model)
        _, names_xgboost, _, bayes_opt_xgboost = getclf_xgboost(self.db_model)
        _, names_keras, _, bayes_opt_keras = getclf_keras(self.db_model,
                                                          len(self.df_xtrain.columns))
        clfs_all = bayes_opt_scikit + bayes_opt_xgboost + bayes_opt_keras
        clfs_names_all = names_scikit + names_xgboost + names_keras


        clfs_names_all = [name for name, clf in zip(clfs_names_all, clfs_all) if clf]
        clfs_all = [clf for clf in clfs_all if clf]

        out_dirs = [os.path.join(self.dirmlplot, "bayesian_opt", name, f"{name}{self.s_suffix}") \
                for name in clfs_names_all]
        if checkdirlist(out_dirs):
            # Only draw results if any can be found
            self.logger.warning("Not overwriting anything, just plotting if possible " \
                    "Please remove corresponding directories if you are certain you want to do " \
                    "grid search again")
            return
        checkmakedirlist(out_dirs)

        # Now, do it
        for opt, out_dir in zip(clfs_all, out_dirs):
            opt.x_train = self.df_xtrain
            opt.y_train = self.df_ytrain

            opt.optimise(ncores=self.p_ncorescross)
            opt.save(out_dir)
            opt.plot(out_dir)


    def do_grid(self):
        self.logger.info("Do grid search")
        clfs_scikit, names_scikit, grid_params_scikit, _ = getclf_scikit(self.db_model)
        clfs_xgboost, names_xgboost, grid_params_xgboost, _ = getclf_xgboost(self.db_model)
        clfs_keras, names_keras, grid_params_keras, _ = getclf_keras(self.db_model,
                                                                     len(self.df_xtrain.columns))
        clfs_grid_params_all = grid_params_scikit + grid_params_xgboost + grid_params_keras
        clfs_all = clfs_scikit + clfs_xgboost + clfs_keras
        clfs_names_all = names_scikit + names_xgboost + names_keras

        clfs_all = [clf for clf, gps in zip(clfs_all, clfs_grid_params_all) if gps]
        clfs_names_all = [name for name, gps in zip(clfs_names_all, clfs_grid_params_all) if gps]
        clfs_grid_params_all = [gps for gps in clfs_grid_params_all if gps]

        out_dirs = [os.path.join(self.dirmlplot, "grid_search", name, f"{name}{self.s_suffix}") \
                for name in clfs_names_all]
        if checkdirlist(out_dirs):
            # Only draw results if any can be found
            self.logger.warning("Not overwriting anything, just plotting again what was done " \
                    "before and returning. Please remove corresponding directories " \
                    "if you are certain you want do do grid search again")
            perform_plot_gridsearch(clfs_names_all, out_dirs)
            return
        checkmakedirlist(out_dirs)

        do_gridsearch(clfs_names_all, clfs_all, clfs_grid_params_all, self.df_xtrain,
                      self.df_ytrain, self.p_nkfolds, out_dirs, self.p_ncorescross)
        perform_plot_gridsearch(clfs_names_all, out_dirs)

    def do_boundary(self):
        classifiers_scikit_2var, names_2var = getclf_scikit(self.db_model)
        classifiers_keras_2var, names_keras_2var = getclf_keras(self.db_model, 2)
        classifiers_2var = classifiers_scikit_2var+classifiers_keras_2var
        names_2var = names_2var+names_keras_2var
        x_test_boundary = self.df_xtest[self.v_bound]
        trainedmodels_2var = fit(names_2var, classifiers_2var, x_test_boundary, self.df_ytest)
        decisionboundaries(
            names_2var, trainedmodels_2var, self.s_suffix+"2var", x_test_boundary,
            self.df_ytest, self.dirmlplot)

    def do_efficiency(self):
        self.logger.info("Doing efficiency estimation")
        fig_eff = plt.figure(figsize=(20, 15))
        plt.xlabel('Threshold', fontsize=20)
        plt.ylabel('Model Efficiency', fontsize=20)
        plt.title("Efficiency vs Threshold", fontsize=20)
        df_sig = self.df_mltest[self.df_mltest["ismcprompt"] == 1]
        for name in self.p_classname:
            eff_array, eff_err_array, x_axis = calc_sigeff_steps(self.p_nstepsign, df_sig, name)
            plt.figure(fig_eff.number)
            plt.errorbar(x_axis, eff_array, yerr=eff_err_array, alpha=0.3, label=f'{name}',
                         elinewidth=2.5, linewidth=4.0)
        plt.figure(fig_eff.number)
        plt.legend(loc="lower left", prop={'size': 18})
        plt.savefig(f'{self.dirmlplot}/Efficiency_{self.s_suffix}.png')
        with open(f'{self.dirmlplot}/Efficiency_{self.s_suffix}.pickle', 'wb') as out:
            pickle.dump(fig_eff, out)

    #pylint: disable=too-many-locals
    def do_significance(self):
        self.logger.info("Doing significance optimization")
        gROOT.SetBatch(True)
        gROOT.ProcessLine("gErrorIgnoreLevel = kWarning;")
        #first extract the number of data events in the ml sample
        self.df_evt_data = pickle.load(openfile(self.f_evt_data, 'rb'))
        if self.p_dofullevtmerge is True:
            self.df_evttotsample_data = pickle.load(openfile(self.f_evttotsample_data, 'rb'))
        else:
            self.logger.warning("The total merged event dataframe was not merged for space limits")
            self.df_evttotsample_data = pickle.load(openfile(self.f_evt_data, 'rb'))
        #and the total number of events
        self.p_nevttot = len(self.df_evttotsample_data)
        self.p_nevtml = len(self.df_evt_data)
        self.logger.debug("Number of data events used for ML: %d", self.p_nevtml)
        self.logger.debug("Total number of data events: %d", self.p_nevttot)
        #calculate acceptance correction. we use in this case all
        #the signal from the mc sample, without limiting to the n. signal
        #events used for training
        denacc = len(self.df_mcgen[self.df_mcgen["ismcprompt"] == 1])
        numacc = len(self.df_mc[self.df_mc["ismcprompt"] == 1])
        acc, acc_err = calc_eff(numacc, denacc)
        self.logger.debug("Acceptance: %.3e +/- %.3e", acc, acc_err)
        #calculation of the expected fonll signals
        ptmin = self.p_binmin
        ptmax = self.p_binmax
        delta_pt = ptmax - ptmin
        if self.is_fonll_from_root:
            df_fonll = TFile.Open(self.f_fonll)
            df_fonll_Lc = df_fonll.Get(self.p_fonllparticle+"pred_"+self.p_fonllband)
            prod_cross = df_fonll_Lc.Integral(ptmin*20, ptmax*20)* self.p_fragf * 1e-12 / delta_pt
            signal_yield = 2. * prod_cross * delta_pt * acc * self.p_taa \
                           / (self.p_sigmamb * self.p_fprompt)
            #now we plot the fonll expectation
            cFONLL = TCanvas('cFONLL', 'The FONLL expectation')
            df_fonll_Lc.GetXaxis().SetRangeUser(0, 16)
            df_fonll_Lc.Draw("")
            cFONLL.SaveAs("%s/FONLL_curve_%s.png" % (self.dirmlplot, self.s_suffix))
        else:
            df_fonll = pd.read_csv(self.f_fonll)
            df_fonll_in_pt = df_fonll.query('(pt >= @ptmin) and (pt < @ptmax)')[self.p_fonllband]
            prod_cross = df_fonll_in_pt.sum() * self.p_fragf * 1e-12 / delta_pt
            signal_yield = 2. * prod_cross * delta_pt * self.p_br * acc * self.p_taa \
                           / (self.p_sigmamb * self.p_fprompt)
            #now we plot the fonll expectation
            plt.figure(figsize=(20, 15))
            plt.subplot(111)
            plt.plot(df_fonll['pt'], df_fonll[self.p_fonllband] * self.p_fragf, linewidth=4.0)
            plt.xlabel('P_t [GeV/c]', fontsize=20)
            plt.ylabel('Cross Section [pb/GeV]', fontsize=20)
            plt.title("FONLL cross section " + self.p_case, fontsize=20)
            plt.semilogy()
            plt.savefig(f'{self.dirmlplot}/FONLL_curve_{self.s_suffix}.png')

        self.logger.debug("Expected signal yield: %.3e", signal_yield)
        signal_yield = self.p_raahp * signal_yield
        self.logger.debug("Expected signal yield x RAA hp: %.3e", signal_yield)

        df_data_sideband = self.df_data.query(self.s_selbkgml)
        df_data_sideband = shuffle(df_data_sideband, random_state=self.rnd_shuffle)
        df_data_sideband = df_data_sideband.tail(round(len(df_data_sideband) * self.p_bkgfracopt))
        hmass = TH1F('hmass', '', self.p_num_bins, self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
        df_mc_signal = self.df_mc[self.df_mc["ismcsignal"] == 1]
        mass_array = df_mc_signal[self.v_invmass].values
        for mass_value in np.nditer(mass_array):
            hmass.Fill(mass_value)

        gaus_fit = TF1("gaus_fit", "gaus", self.p_mass_fit_lim[0], self.p_mass_fit_lim[1])
        gaus_fit.SetParameters(0, hmass.Integral())
        gaus_fit.SetParameters(1, self.p_mass)
        gaus_fit.SetParameters(2, 0.02)
        self.logger.debug("To fit the signal a gaussian function is used")
        fitsucc = hmass.Fit("gaus_fit", "RQ")

        if int(fitsucc) != 0:
            self.logger.warning("Problem in signal peak fit")
            sigma = 0.

        sigma = gaus_fit.GetParameter(2)
        self.logger.debug("Mean of the gaussian: %.3e", gaus_fit.GetParameter(1))
        self.logger.debug("Sigma of the gaussian: %.3e", sigma)
        sig_region = [self.p_mass - 3 * sigma, self.p_mass + 3 * sigma]
        fig_signif_pevt = plt.figure(figsize=(20, 15))
        plt.xlabel('Threshold', fontsize=20)
        plt.ylabel(r'Significance Per Event ($3 \sigma$)', fontsize=20)
        plt.title("Significance Per Event vs Threshold", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        fig_signif = plt.figure(figsize=(20, 15))
        plt.xlabel('Threshold', fontsize=20)
        plt.ylabel(r'Significance ($3 \sigma$)', fontsize=20)
        plt.title("Significance vs Threshold", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        df_sig = self.df_mltest[self.df_mltest["ismcprompt"] == 1]

        for name in self.p_classname:
            eff_array, eff_err_array, x_axis = calc_sigeff_steps(self.p_nstepsign, df_sig, name)
            bkg_array, bkg_err_array, _ = calc_bkg(df_data_sideband, name, self.p_nstepsign,
                                                   self.p_mass_fit_lim, self.p_bkg_func,
                                                   self.p_bin_width, sig_region, self.p_savefit,
                                                   self.dirmlplot, [self.p_binmin, self.p_binmax])
            sig_array = [eff * signal_yield for eff in eff_array]
            sig_err_array = [eff_err * signal_yield for eff_err in eff_err_array]
            bkg_array = [bkg / (self.p_bkgfracopt * self.p_nevtml) for bkg in bkg_array]
            bkg_err_array = [bkg_err / (self.p_bkgfracopt * self.p_nevtml) \
                             for bkg_err in bkg_err_array]
            signif_array, signif_err_array = calc_signif(sig_array, sig_err_array, bkg_array,
                                                         bkg_err_array)
            plt.figure(fig_signif_pevt.number)
            plt.errorbar(x_axis, signif_array, yerr=signif_err_array, label=f'{name}',
                         elinewidth=2.5, linewidth=5.0)
            signif_array_ml = [sig * sqrt(self.p_nevtml) for sig in signif_array]
            signif_err_array_ml = [sig_err * sqrt(self.p_nevtml) for sig_err in signif_err_array]
            plt.figure(fig_signif.number)
            plt.errorbar(x_axis, signif_array_ml, yerr=signif_err_array_ml,
                         label=f'{name}_ML_dataset', elinewidth=2.5, linewidth=5.0)
            signif_array_tot = [sig * sqrt(self.p_nevttot) for sig in signif_array]
            signif_err_array_tot = [sig_err * sqrt(self.p_nevttot) for sig_err in signif_err_array]
            plt.figure(fig_signif.number)
            plt.errorbar(x_axis, signif_array_tot, yerr=signif_err_array_tot,
                         label=f'{name}_Tot', elinewidth=2.5, linewidth=5.0)
            plt.figure(fig_signif_pevt.number)
            plt.legend(loc="upper left", prop={'size': 30})
            plt.savefig(f'{self.dirmlplot}/Significance_PerEvent_{self.s_suffix}.png')
            plt.figure(fig_signif.number)
            plt.legend(loc="upper left", prop={'size': 30})
            plt.savefig(f'{self.dirmlplot}/Significance_{self.s_suffix}.png')
            with open(f'{self.dirmlplot}/Significance_{self.s_suffix}.pickle', 'wb') as out:
                pickle.dump(fig_signif, out)

    def do_scancuts(self):
        self.logger.info("Scanning cuts")

        prob_array = [0.0, 0.2, 0.6, 0.9]
        dfdata = pickle.load(openfile(self.f_reco_applieddata, "rb"))
        dfmc = pickle.load(openfile(self.f_reco_appliedmc, "rb"))
        vardistplot_probscan(dfmc, self.v_all, "xgboost_classifier",
                             prob_array, self.dirmlplot, "mc" + self.s_suffix,
                             0, self.p_plot_options)
        vardistplot_probscan(dfmc, self.v_all, "xgboost_classifier",
                             prob_array, self.dirmlplot, "mc" + self.s_suffix,
                             1, self.p_plot_options)
        vardistplot_probscan(dfdata, self.v_all, "xgboost_classifier",
                             prob_array, self.dirmlplot, "data" + self.s_suffix,
                             0, self.p_plot_options)
        vardistplot_probscan(dfdata, self.v_all, "xgboost_classifier",
                             prob_array, self.dirmlplot, "data" + self.s_suffix,
                             1, self.p_plot_options)
        if not self.v_cuts:
            self.logger.warning("No variables for cut efficiency scan. Will be skipped")
            return
        efficiency_cutscan(dfmc, self.v_cuts, "xgboost_classifier", 0.0,
                           self.dirmlplot, "mc" + self.s_suffix, self.p_plot_options)
        efficiency_cutscan(dfmc, self.v_cuts, "xgboost_classifier", 0.5,
                           self.dirmlplot, "mc" + self.s_suffix, self.p_plot_options)
        efficiency_cutscan(dfdata, self.v_cuts, "xgboost_classifier", 0.0,
                           self.dirmlplot, "data" + self.s_suffix, self.p_plot_options)
        efficiency_cutscan(dfdata, self.v_cuts, "xgboost_classifier", 0.5,
                           self.dirmlplot, "data" + self.s_suffix, self.p_plot_options)
