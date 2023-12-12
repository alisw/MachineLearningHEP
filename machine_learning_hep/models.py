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
Methods to: choose, train and apply ML models
            load and save ML models
            obtain control plots
"""
from os.path import exists
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from sklearn.feature_extraction import DictVectorizer

import shap

from machine_learning_hep.logger import get_logger
from machine_learning_hep import templates_keras, templates_xgboost, templates_scikit
pd.options.mode.chained_assignment = None

def getclf_scikit(model_config):

    logger = get_logger()
    logger.debug("Load scikit models")

    if "scikit" not in model_config:
        logger.debug("No scikit models found")
        return [], []

    classifiers = []
    names = []
    grid_search_params = []
    bayesian_opt = []

    for c in model_config["scikit"]:
        if model_config["scikit"][c]["activate"]:
            try:
                model = getattr(templates_scikit, c)(model_config["scikit"][c]["central_params"])
                c_bayesian = f"{c}_bayesian_opt"
                bayes_opt = None
                if hasattr(templates_scikit, c_bayesian):
                    bayes_opt = getattr(templates_scikit, c_bayesian) \
                            (model_config["scikit"][c]["central_params"])
                bayesian_opt.append(bayes_opt)
                classifiers.append(model)
                names.append(c)
                grid_search_params.append(model_config["scikit"][c].get("grid_search", {}))
                logger.info("Added scikit model %s", c)
            except AttributeError:
                logger.critical("Could not load scikit model %s", c)

    return classifiers, names, grid_search_params, bayesian_opt


def getclf_xgboost(model_config):

    logger = get_logger()
    logger.debug("Load xgboost models")

    if "xgboost" not in model_config:
        logger.debug("No xgboost models found")
        return [], []

    classifiers = []
    names = []
    grid_search_params = []
    bayesian_opt = []

    for c in model_config["xgboost"]:
        if model_config["xgboost"][c]["activate"]:
            try:
                model = getattr(templates_xgboost, c)(model_config["xgboost"][c]["central_params"])
                c_bayesian = f"{c}_bayesian_opt"
                bayes_opt = None
                if hasattr(templates_xgboost, c_bayesian):
                    bayes_opt = getattr(templates_xgboost, c_bayesian) \
                            (model_config["xgboost"][c]["central_params"])
                bayesian_opt.append(bayes_opt)
                classifiers.append(model)
                names.append(c)
                grid_search_params.append(model_config["xgboost"][c].get("grid_search", {}))
                logger.info("Added xgboost model %s", c)
            except AttributeError:
                logger.critical("Could not load xgboost model %s", c)

    return classifiers, names, grid_search_params, bayesian_opt


def getclf_keras(model_config, length_input):

    logger = get_logger()
    logger.debug("Load keras models")

    if "keras" not in model_config:
        logger.debug("No keras models found")
        return [], []

    classifiers = []
    names = []
    bayesian_opt = []

    for c in model_config["keras"]:
        if model_config["keras"][c]["activate"]:
            try:
                model = getattr(templates_keras, c)(model_config["keras"][c]["central_params"],
                                                    length_input)
                classifiers.append(model)
                c_bayesian = f"{c}_bayesian_opt"
                bayes_opt = None
                if hasattr(templates_keras, c_bayesian):
                    bayes_opt = getattr(templates_keras, c_bayesian) \
                            (model_config["keras"][c]["central_params"], length_input)
                bayesian_opt.append(bayes_opt)
                names.append(c)
                logger.info("Added keras model %s", c)
            except AttributeError:
                logger.critical("Could not load keras model %s", c)

    #logger.critical("Some reason")
    return classifiers, names, [], bayesian_opt



def fit(names_, classifiers_, x_train_, y_train_):
    trainedmodels_ = []
    for _, clf in zip(names_, classifiers_):
        clf.fit(x_train_, y_train_)
        trainedmodels_.append(clf)
    return trainedmodels_


def apply(ml_type, names_, trainedmodels_, test_set_, mylistvariables_, labels_=None):
    logger = get_logger()

    x_values = test_set_[mylistvariables_]
    for name, model in zip(names_, trainedmodels_):
        y_test_prediction = model.predict(x_values)
        test_set_[f"y_test_prediction{name}"] = pd.Series(y_test_prediction, index=test_set_.index)

        y_test_prob = model.predict_proba(x_values)
        if ml_type == "BinaryClassification":
            test_set_[f"y_test_prob{name}"] = pd.Series(y_test_prob[:, 1], index=test_set_.index)
        elif ml_type == "MultiClassification" and labels_ is not None:
            for pred, lab in enumerate(labels_):
                test_set_[f"y_test_prob{name}{lab}"] = pd.Series(y_test_prob[:, pred],
                                                                 index=test_set_.index)
        else:
            logger.fatal("Incorrect settings for chosen mltype")
    return test_set_


def savemodels(names_, trainedmodels_, folder_, suffix_):
    for name, model in zip(names_, trainedmodels_):
        if "keras" in name:
            architecture_file = f"{folder_}/{name}{suffix_}_architecture.json"
            weights_file = f"{folder_}/{name}{suffix_}_weights.h5"
            arch_json = model.model.to_json()
            with open(architecture_file, 'w', encoding='utf-8') as json_file:
                json_file.write(arch_json)
            model.model.save_weights(weights_file)
        if "scikit" in name:
            fileoutmodel = f"{folder_}/{name}{suffix_}.sav"
            with open(fileoutmodel, 'wb') as out_file:
                pickle.dump(model, out_file, protocol=4)
        if "xgboost" in name:
            fileoutmodel = f"{folder_}/{name}{suffix_}.sav"
            with open(fileoutmodel, 'wb') as out_file:
                pickle.dump(model, out_file, protocol=4)
            fileoutmodel = fileoutmodel.replace(".sav", ".model")
            model.save_model(fileoutmodel)

def readmodels(names_, folder_, suffix_):
    trainedmodels_ = []
    for name in names_:
        fileinput = folder_+"/"+name+suffix_+".sav"
        if not exists(fileinput):
            return None
        with open(fileinput, 'rb') as input_file:
            model = pickle.load(input_file)
        trainedmodels_.append(model)
    return trainedmodels_


def importanceplotall(mylistvariables_, names_, trainedmodels_, suffix_, folder):
    names_models = [(name, model) for name, model in zip(names_, trainedmodels_) \
            if not any(mname in name for mname in ("SVC", "Logistic", "Keras"))]
    if len(names_models) == 1:
        figure = plt.figure(figsize=(18, 15))
        nrows, ncols = (1, 1)
    else:
        figure = plt.figure(figsize=(25, 15))
        nrows, ncols = (2, (len(names_models) + 1) / 2)
    for ind, (name, model) in enumerate(names_models, start=1):
        ax = plt.subplot(nrows, ncols, ind)
        #plt.subplots_adjust(left=0.3, right=0.9)
        feature_importances_ = model.feature_importances_
        y_pos = np.arange(len(mylistvariables_))
        ax.barh(y_pos, feature_importances_, align='center', color='green')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(mylistvariables_, fontsize=17)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("Importance", fontsize=17)
        ax.set_title(f"Importance features {name}", fontsize=17)
        ax.xaxis.set_tick_params(labelsize=17)
        plt.xlim(0, 0.7)
    if len(names_models) > 1:
        plt.subplots_adjust(wspace=0.5)
    plotname = f"{folder}/importanceplotall{suffix_}.png"
    figure.savefig(plotname, bbox_inches='tight')
    plt.close()

def shap_study(names_, trainedmodels_, x_train_, suffix_, folder, plot_options_):
    """Importance via SHAP

    Args:
        names_: list
            Names of models to do study for
        models_: list
            Models to be studied
        x_train_: pandas.DataFrame
            Dataframe with training samples
        suffix_: str
        folder: str
            Where to be saved
    """
    mpl.rcParams.update({"text.usetex": True})
    plot_type_name = "prob_cut_scan"
    plot_options = plot_options_.get(plot_type_name, {}) \
            if isinstance(plot_options_, dict) else {}
    feature_names = []
    for fn in x_train_.columns:
        if fn in plot_options and "xlabel" in plot_options[fn]:
            feature_names.append("$" + plot_options[fn]["xlabel"] + "$")
        else:
            feature_names.append(fn.replace("_", ":"))

    # Rely on name to exclude certain models at the moment
    names_models = [(name, model) for name, model in zip(names_, trainedmodels_) \
            if not any(mname in name for mname in ("SVC", "Logistic", "Keras"))]
    if len(names_models) == 1:
        figure = plt.figure(figsize=(18, 15))
        nrows, ncols = (1, 1)
    else:
        figure = plt.figure(figsize=(25, 15))
        nrows, ncols = (2, (len(names_models) + 1) / 2)
    for ind, (name, model) in enumerate(names_models, start=1):
        ax = figure.add_subplot(nrows, ncols, ind)
        plt.sca(ax)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train_)
        shap.summary_plot(shap_values, x_train_, show=False, feature_names=feature_names)
    plotname = f"{folder}/importanceplotall_shap_{suffix_}.png"
    figure.tight_layout()
    figure.savefig(plotname, bbox_inches='tight')
    mpl.rcParams.update({"text.usetex": False})
    plt.close(figure)


def decisionboundaries(names_, trainedmodels_, suffix_, x_train_, y_train_, folder):
    mylistvariables_ = x_train_.columns.tolist()
    dictionary_train = x_train_.to_dict(orient='records')
    vec = DictVectorizer()
    x_train_array_ = vec.fit_transform(dictionary_train).toarray()

    figure = plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)
    height = .10
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = x_train_array_[:, 0].min() - .5, x_train_array_[:, 0].max() + .5
    y_min, y_max = x_train_array_[:, 1].min() - .5, x_train_array_[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, height), np.arange(y_min, y_max, height))

    i = 1
    for name, model in zip(names_, trainedmodels_):
        if hasattr(model, "decision_function"):
            z_contour = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            z_contour = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        ax = plt.subplot(2, (len(names_)+1)/2, i)

        z_contour = z_contour.reshape(xx.shape)
        ax.contourf(xx, yy, z_contour, cmap=cm, alpha=.8)
        # Plot also the training points
        ax.scatter(x_train_array_[:, 0], x_train_array_[:, 1],
                   c=y_train_, cmap=cm_bright, edgecolors='k', alpha=0.3)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        score = model.score(x_train_, y_train_)
        ax.text(xx.max() - .3, yy.min() + .3, (f"accuracy={score:.2f}").lstrip('0'),
                size=15, horizontalalignment='right', verticalalignment='center')
        ax.set_title(name, fontsize=17)
        ax.set_ylabel(mylistvariables_[1], fontsize=17)
        ax.set_xlabel(mylistvariables_[0], fontsize=17)
        figure.subplots_adjust(hspace=.5)
        i += 1
    plotname = f"{folder}/decisionboundaries{suffix_}.png"
    figure.savefig(plotname, bbox_inches='tight')
    plt.close(figure)
