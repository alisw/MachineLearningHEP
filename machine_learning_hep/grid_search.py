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
Methods to do hyper-parameters optimization
"""
from io import BytesIO
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier  # pylint: disable=unused-import
from xgboost import XGBClassifier # pylint: disable=unused-import
from machine_learning_hep.logger import get_logger


def do_gridsearch(names, classifiers, param_grid, refit_arr, x_train, y_train_, cv_, ncores):
    logger = get_logger()
    grid_search_models_ = []
    grid_search_bests_ = []
    list_scores_ = []
    for _, clf, param_cv, refit in zip(names, classifiers, param_grid, refit_arr):
        grid_search = GridSearchCV(clf, param_cv, cv=cv_, refit=refit,
                                   scoring='neg_mean_squared_error', n_jobs=ncores)
        grid_search_model = grid_search.fit(x_train, y_train_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            logger.info(np.sqrt(-mean_score), params)
        list_scores_.append(pd.DataFrame(cvres))
        grid_search_best = grid_search.best_estimator_.fit(x_train, y_train_)
        grid_search_models_.append(grid_search_model)
        grid_search_bests_.append(grid_search_best)
    return grid_search_models_, grid_search_bests_, list_scores_


def read_grid_dict(grid_dict):
    names_cv = []
    clf_cv = []
    par_grid_cv = []
    par_grid_cv_keys = []
    refit_cv = []
    var_param_cv = []

    for keymodels, _ in grid_dict.items():
        names_cv.append(grid_dict[keymodels]["name"])
        clf_cv.append(eval(grid_dict[keymodels]["clf"]))  # pylint: disable=eval-used
        par_grid_cv.append([grid_dict[keymodels]["param_grid"]])
        refit_cv.append(grid_dict[keymodels]["refit_grid"])
        var_param_cv.append(grid_dict[keymodels]["var_param"])
        par_grid_cv_keys.append(
            ["param_{}".format(key) for key in grid_dict[keymodels]["param_grid"].keys()])
    return names_cv, clf_cv, par_grid_cv, refit_cv, var_param_cv, par_grid_cv_keys


def perform_plot_gridsearch(names, scores, par_grid, keys, changeparameter, output_,
                            suffix_, alpha):
    '''
    Function for grid scores plotting (working with scikit 0.20)
    '''
    logger = get_logger()
    fig = plt.figure(figsize=(35, 15))
    for name, score_obj, pars, key, change in zip(names, scores, par_grid,
                                                  keys, changeparameter):
        change_par = "param_" + change
        if len(key) == 1:
            logger.warning("Add at least 1 parameter (even just 1 value)")
            continue

        dict_par = pars[0].copy()
        dict_par.pop(change)
        lst_par_values = list(dict_par.values())
        listcomb = []
        for comb in itertools.product(*lst_par_values):
            listcomb.append(comb)

        # plotting a graph for every combination of paramater different from
        # change (e.g.: n_estimator in random_forest): score vs. change
        pad = fig.add_subplot(1, len(names), names.index(name)+1)
        pad.set_title(name, fontsize=20)
        plt.ylim(-0.6, 0)
        plt.xlabel(change_par, fontsize=15)
        plt.ylabel('neg_mean_squared_error', fontsize=15)
        pad.get_xaxis().set_tick_params(labelsize=15)
        pad.get_yaxis().set_tick_params(labelsize=15)
        key.remove(change_par)
        for case in listcomb:
            df_case = score_obj.copy()
            lab = ""
            for i_case, i_key in zip(case, key):
                df_case = df_case.loc[df_case[i_key] == float(i_case)]
                lab = lab + "{0}: {1} \n".format(i_key, i_case)
            df_case.plot(x=change_par, y='mean_test_score', ax=pad, label=lab,
                         marker='o', style="-")
            sample_x = list(df_case[change_par])
            sample_score_mean = list(df_case["mean_test_score"])
            sample_score_std = list(df_case["std_test_score"])
            lst_up = [mean+std for mean, std in zip(sample_score_mean,
                                                    sample_score_std)]
            lst_down = [mean-std for mean, std in zip(sample_score_mean,
                                                      sample_score_std)]
            plt.fill_between(sample_x, lst_down, lst_up, alpha=alpha)
        pad.legend(fontsize=10)
    plotname = output_ + "/GridSearchResults" + suffix_ + ".png"
    plt.savefig(plotname)
    img_gridsearch = BytesIO()
    plt.savefig(img_gridsearch, format='png')
    img_gridsearch.seek(0)
    return img_gridsearch



#CODE TO BE USED LATER ON.
# New Implementation of Grid Search which relies on AUC and Precision Recall metrics to perform hyperparam tuning
        # Maximize F1-score or ROC-AUC
#        metric = 'f1' # just remember to replace f1 with roc_auc if you want to tune w.r.t roc-auc
#        cv_params = {'max_depth': [1,2,3,4,5,6], 'min_child_weight': [1,2,3,4]}    # parameters to be tried in the grid search
#        fix_params = {'learning_rate': 0.2, 'n_estimators': 100, 'objective': 'binary:logistic'}   #other parameters, fixed for the moment 
#        csv = GridSearchCV(xgb.XGBClassifier(**fix_params, nthread = -1), cv_params, scoring = metric, cv = 5)
#        csv.fit(self.df_xtrain.values, self.df_ytrain.values)
#        print(csv.cv_results_)
#        print(csv.best_params_)
#        cv_params = {'subsample': [0.8,0.9,1], 'max_delta_step': [0,1,2,4,8]}
#        fix_params = {'learning_rate': 0.2, 'n_estimators': 100, 'objective': 'binary:logistic', 'max_depth': 5, 'min_child_weight':3}
#        csv = GridSearchCV(xgb.XGBClassifier(**fix_params, nthread = -1), cv_params, scoring = metric, cv = 5) 
#        csv.fit(self.df_xtrain.values, self.df_ytrain.values)
#        print(csv.best_params_)
#        print(csv.cv_results_)
        #cv_params = {'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]}
        #fix_params['max_delta_step'] = 0
        #fix_params['subsample'] = 0.9
        #csv = GridSearchCV(xgb.XGBClassifier(**fix_params, nthread=-1), cv_params, scoring = metric, cv = 5) 
        #csv.fit(self.df_xtrain.values, self.df_ytrain.values)
        #print(csv.cv_results_)
        #print(csv.best_params_)
        #fix_params['learning_rate'] = 0.25
        #params_final =  fix_params
        #print(params_final)
        #xgdmat_train = xgb.DMatrix(self.df_xtrain.values, self.df_ytrain.values)
        #xgdmat_test = xgb.DMatrix(self.df_xtest.values, self.df_ytest.values)
        #xgb_final = xgb.train(params_final, xgdmat_train, num_boost_round = 100)
        #y_pred = xgb_final.predict(xgdmat_test)
        #thresh = 0.08
        #y_pred [y_pred > thresh] = 1
        #y_pred [y_pred <= thresh] = 0
        #cm = confusion_matrix(self.df_ytest.values, y_pred)
        #pc.plot_confusion_matrix(cm, ['0', '1'], )
        #pr, tpr, fpr = pc.show_data(cm, print_res = 1);
'''
        par = params_final
        plot_roc(X_, y_, par, 'max_depth', [1,2,3,4,5,7,10,15])
        par['max_depth'] = 5
        plot_roc(X_, y_, par, 'learning_rate', [0.05,0.1,0.15,0.2,0.25,0.3])
        par['learning_rate'] = 0.2

        xgdmat_train = xgb.DMatrix(X_, y_)
        xgdmat_test = xgb.DMatrix(X_test, y_test)
        xgb_final = xgb.train(par, xgdmat_train, num_boost_round = 100)
        y_pred = xgb_final.predict(xgdmat_test)
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred)
        prec, rec, thresholds_pr = precision_recall_curve(y_test, y_pred)

        mean_fpr = np.linspace(0, 1, 100000)
        mean_rec = np.linspace(0, 1, 1000)

       prec = list(reversed(prec)) #reverse, otherwise the interp doesn not work
       rec = list(reversed(rec))
       mean_tpr = np.interp(mean_fpr, fpr, tpr)
       mean_prec = np.interp(mean_rec, rec, prec)
       mean_fpr2, mean_tpr2, mean_prec2, mean_rec2 = gen_curves(X_, y_, par)   #the averaged curves from the training set as comparison

       f, (ax1, ax2) = plt.subplots(1, 2, figsize = (18,7));
       ax1.plot(mean_fpr, mean_tpr, label = 'on testing set');
       ax1.plot(mean_fpr2, mean_tpr2, label = 'on training set');

       ax2.plot(mean_rec, mean_prec, label = 'on testing set');
       ax2.plot(mean_rec2, mean_prec2, label = 'on training set');

      ax1.set_xlim([0, 0.0005])
      ax1.set_ylim([0.5, 0.95])
      ax1.axvline(2e-4, color='b', linestyle='dashed', linewidth=2)
      ax1.set_xlabel('FPR/Fallout')
      ax1.set_ylabel('TPR/Recall')
      ax2.set_xlabel('Recall')
      ax2.set_ylabel('Precision')
      ax1.set_title('ROC')
      ax2.set_title('PR')
      ax2.set_xlim([0.5, 1])
      ax1.legend(loc="lower right")
      ax2.legend(loc = 'lower left')
      plt.show()
'''
 #    y_final = np.copy(y_pred)
 #    thresh = 0.08
 #    y_final [y_final > thresh] = 1
 #    y_final [y_final <= thresh] = 0
 #    cm = confusion_matrix(y_test, y_final)
 #    pc.plot_confusion_matrix(cm, ['0', '1'], )
 #    pr, tpr, fpr = pc.show_data(cm, print_res = 1);
