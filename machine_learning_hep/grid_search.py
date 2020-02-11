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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier  # pylint: disable=unused-import
from xgboost import XGBClassifier # pylint: disable=unused-import
from machine_learning_hep.logger import get_logger
from machine_learning_hep.io import print_dict


def get_scorers(score_names):
    """Construct dictionary of scorers

    Args:
        score_names: tuple of names. Available names see below
    Returns:
        dictionary mapping scorers to names
    """

    scorers = {}
    for sn in score_names:
        if sn == "AUC":
            scorers["AUC"] = make_scorer(roc_auc_score, needs_threshold=True)
        elif sn == "Accuracy":
            scorers["Accuracy"] = make_scorer(accuracy_score)

    return scorers


def do_gridsearch(names, classifiers, grid_params, x_train, y_train, nkfolds, ncores):
    """Hyperparameter grid search for a list of classifiers

    Given a list of classifiers, do a hyperparameter grid search based on a corresponding
    set of parameters

    Args:
        names: iteratable of classifier names
        classifiers: iterable of classifiers
        grid_params: iterable of parameters used to perform the grid search
        x_train: feature dataframe
        y_train: targets dataframe
        nkfolds: int, cross-validation generator or an iterable
        ncores: number of cores to distribute jobs to
    Returns:
        lists of grid search models, the best model and scoring dataframes
    """

    logger = get_logger()
    grid_search_models_ = []
    grid_search_bests_ = []
    list_scores_ = []

    # As this may take a long time, accept a keyboard interrupt
    try:
        for clf_name, clf, gps in zip(names, classifiers, grid_params):
            if not gps:
                logger.info("Nothing to be done for grid search of model %s", clf_name)
                continue
            logger.info("Grid search for model %s with following parameters:", clf_name)
            print_dict(gps)

            # To work for probabilities. This will call model.decision_function or
            # model.predict_proba as it is done for the nominal ROC curves as well to decide on the
            # performance
            scoring = get_scorers(gps["scoring"])

            grid_search = GridSearchCV(clf, gps["params"], cv=nkfolds, refit=gps["refit"],
                                       scoring=scoring, n_jobs=ncores, verbose=2,
                                       return_train_score=True)
            grid_search.fit(x_train, y_train)
            cvres = grid_search.cv_results_
            #for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            #    logger.info(mean_score, params)

            # Add test AUC if test data is available
            # Tuple of grid search dataframe and test AUC
            list_scores_.append(pd.DataFrame(cvres))
            grid_search_models_.append(grid_search)
            grid_search_bests_.append(grid_search.best_estimator_)
    except KeyboardInterrupt:
        # Could be of different length depending on where the KeyboardInterrupt was issued
        # ==> Trim to the same length
        logger.warning("Grid search interrupted. Returning what was done.")
        min_length = min(len(grid_search_models_), len(grid_search_bests_), len(list_scores_))
        grid_search_models_ = grid_search_models_[:min_length]
        grid_search_bests_ = grid_search_bests_[:min_length]
        list_scores_ = list_scores_[:min_length]
    return grid_search_models_, grid_search_bests_, list_scores_


# pylint: disable=too-many-locals, too-many-statements
def perform_plot_gridsearch(names, scores, grid_params, out_dirs, suffix_):
    '''
    Function for grid scores plotting (working with scikit 0.20)
    '''
    logger = get_logger()

    for name, score_obj, gps, out_dir in zip(names, scores, grid_params, out_dirs):
        param_keys = [f"param_{key}" for key in gps["params"].keys()]
        if not param_keys:
            logger.warning("Add at least 1 parameter (even just 1 value)")
            continue

        # Re-arrange scoring such that the refitted one is always on top
        score_names = gps["scoring"]
        refit_score = gps["refit"]
        del score_names[score_names.index(refit_score)]
        score_names.insert(0, refit_score)

        # Extract scores
        x_labels = []
        y_values = {}
        y_errors = {}

        for sn in score_names:
            y_values[sn] = {"train": [], "test": []}
            y_errors[sn] = {"train": [], "test": []}

        # Get indices of values to put on x-axis and identify parameter combination
        values_indices = [range(len(values)) for values in gps["params"].values()]

        y_axis_mins = {sn: 9999 for sn in score_names}
        y_axis_maxs = {sn: -9999 for sn in score_names}
        for indices, case in zip(itertools.product(*values_indices),
                                 itertools.product(*list(gps["params"].values()))):
            df_case = score_obj.copy()
            for i_case, i_key in zip(case, param_keys):
                df_case = df_case.loc[df_case[i_key] == df_case[i_key].dtype.type(i_case)]

            x_labels.append(",".join([str(i) for i in indices]))
            # As we just nailed it down to one value
            for sn in score_names:
                for tt in ("train", "test"):
                    y_values[sn][tt].append(df_case[f"mean_{tt}_{sn}"].values[0])
                    y_errors[sn][tt].append(df_case[f"std_{tt}_{sn}"].values[0])
                    y_axis_mins[sn] = min(y_axis_mins[sn], y_values[sn][tt][-1])
                    y_axis_maxs[sn] = max(y_axis_maxs[sn], y_values[sn][tt][-1])

        # Prepare text for parameters
        text_parameters = "\n".join([f"{key}: {values}" for key, values in gps["params"].items()])

        # To determine fontsizes later
        figsize = (35, 18 * len(score_names))
        fig, axes = plt.subplots(len(score_names), 1, sharex=True, gridspec_kw={"hspace": 0.05},
                                 figsize=figsize)
        ax_plot = dict(zip(score_names, axes))

        # The axes to put the parameter list
        ax_main = axes[-1]
        # The axes with the title being on top
        ax_top = axes[0]

        points_per_inch = 72
        markerstyles = ["o", "+"]
        markersize = 20

        for sn in score_names:
            ax = ax_plot[sn]
            ax_min = y_axis_mins[sn] - (y_axis_maxs[sn] - y_axis_mins[sn]) / 10.
            ax_max = y_axis_maxs[sn] + (y_axis_maxs[sn] - y_axis_mins[sn]) / 10.
            ax.set_ylim(ax_min, ax_max)
            ax.set_ylabel(f"mean {sn}", fontsize=20)
            ax.get_yaxis().set_tick_params(labelsize=20)

            for j, tt in enumerate(("train", "test")):
                markerstyle = markerstyles[j % len(markerstyles)]

                ax.errorbar(range(len(x_labels)), y_values[sn][tt], yerr=y_errors[sn][tt],
                            ls="", marker=markerstyle, markersize=markersize, label=f"{sn} ({tt})")

                # Add values to points
                ylim = ax.get_ylim()
                plot_labels_offset = (ylim[1] - ylim[0]) / 40
                for x, y in enumerate(y_values[sn][tt]):
                    ax.text(x, y - plot_labels_offset, f"{y:.4f}", fontsize=20)

        ax_main.set_xlabel("parameter indices", fontsize=20)
        ax_top.set_title(f"Grid search {name}", fontsize=30)
        ax_main.get_xaxis().set_tick_params(labelsize=20)
        ax_main.set_xticks(range(len(x_labels)))
        ax_main.set_xticklabels(x_labels, rotation=45)

        text_point_size = int(4 * fig.dpi / points_per_inch * figsize[1] / len(gps["params"]))
        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()

        xlow = xlim[0] + (xlim[1] - xlim[0]) / 100
        ylow = ylim[0] + (ylim[1] - ylim[0]) / 3
        ax_main.text(xlow, ylow, text_parameters, fontsize=text_point_size)

        for ax in ax_plot.values():
            ax.legend(loc="center right", fontsize=20)
        plotname = f"{out_dir}/GridSearchResults_{suffix_}_{name}.png"
        plt.savefig(plotname)
        plt.close(fig)
