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

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier  # pylint: disable=unused-import
from sklearn_evaluation import plot


def do_gridsearch(names, classifiers, param_grid, refit_arr, x_train, y_train_, cv_, ncores):
    grid_search_models_ = []
    grid_search_bests_ = []
    list_scores_ = []
    for _, clf, param_cv, refit in zip(names, classifiers, param_grid, refit_arr):
        grid_search = GridSearchCV(clf, param_cv, cv=cv_, refit=refit,
                                   scoring='neg_mean_squared_error', n_jobs=ncores)
        grid_search_model = grid_search.fit(x_train, y_train_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        list_scores_.append(pd.DataFrame(cvres))
        grid_search_best = grid_search.best_estimator_.fit(x_train, y_train_)
        grid_search_models_.append(grid_search_model)
        grid_search_bests_.append(grid_search_best)
    return grid_search_models_, grid_search_bests_, list_scores_


def plot_gridsearch(names, change_, grid_search_models_, output_, suffix_):

    for nameCV, change, gs_clf in zip(names, change_, grid_search_models_):
        figure = plt.figure(figsize=(10, 10))  # pylint: disable=unused-variable
        plot.grid_search(gs_clf.grid_scores_, change=change, kind='bar')
        plt.title('Grid search results ' + nameCV, fontsize=17)
        plt.ylim(-0.8, 0)
        plt.ylabel('negative mean squared error', fontsize=17)
        plt.xlabel(change, fontsize=17)
        plotname = output_+"/GridSearchResults"+nameCV+suffix_+".png"
        plt.savefig(plotname)


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
    fig = plt.figure(figsize=(35, 15))
    for name, score_obj, pars, key, change in zip(names, scores, par_grid,
                                                  keys, changeparameter):
        change_par = "param_" + change
        if len(key) == 1:
            print("### ADD AT LEAST 1 PARAMETER (even just 1 value) ###")
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
    # plt.show()
    plt.savefig(plotname)
