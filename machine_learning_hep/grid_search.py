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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier  # pylint: disable=unused-import
from sklearn_evaluation import plot


def do_gridsearch(names, classifiers, param_grid, refit_arr, x_train, y_train_, cv_, ncores):
    grid_search_models_ = []
    grid_search_bests_ = []
    for _, clf, param_cv, refit in zip(names, classifiers, param_grid, refit_arr):
        grid_search = GridSearchCV(clf, param_cv, cv=cv_, refit=refit,
                                   scoring='neg_mean_squared_error', n_jobs=ncores)
        grid_search_model = grid_search.fit(x_train, y_train_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        grid_search_best = grid_search.best_estimator_.fit(x_train, y_train_)
        grid_search_models_.append(grid_search_model)
        grid_search_bests_.append(grid_search_best)
    return grid_search_models_, grid_search_bests_


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
    refit_cv = []
    var_param_cv = []

    for keymodels, _ in grid_dict.items():
        names_cv.append(grid_dict[keymodels]["name"])
        clf_cv.append(eval(grid_dict[keymodels]["clf"]))  # pylint: disable=eval-used
        par_grid_cv.append([grid_dict[keymodels]["param_grid"]])
        refit_cv.append(grid_dict[keymodels]["refit_grid"])
        var_param_cv.append(grid_dict[keymodels]["var_param"])
    return names_cv, clf_cv, par_grid_cv, refit_cv, var_param_cv
