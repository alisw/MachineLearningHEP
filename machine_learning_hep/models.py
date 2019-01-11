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
from io import BytesIO
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pkg_resources import resource_stream

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_extraction import DictVectorizer

from keras.layers import Input, Dense
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier
from machine_learning_hep.logger import get_logger
from machine_learning_hep.io import parse_yaml
import machine_learning_hep.templates_keras as templates_keras
import machine_learning_hep.templates_xgboost as templates_xgboost
import machine_learning_hep.templates_scikit as templates_scikit


def getclf_scikit(model_config):

    logger = get_logger()
    logger.debug("Load scikit models")

    if "scikit" not in model_config:
        logger.debug("No scikit models found")
        return [], []

    classifiers = []
    names = []

    for c in model_config["scikit"]:
        try:
            model = getattr(templates_scikit, c)(model_config["scikit"][c])
            classifiers.append(model)
            names.append(c)
            logger.info("Added scikit model %s", c)
        except AttributeError:
            logger.critical("Could not load scikit model %s", c)

    return classifiers, names


def getclf_xgboost(model_config):

    logger = get_logger()
    logger.debug("Load xgboost models")

    if "xgboost" not in model_config:
        logger.debug("No xgboost models found")
        return [], []

    classifiers = []
    names = []

    for c in model_config["xgboost"]:
        try:
            model = getattr(templates_xgboost, c)(model_config["xgboost"][c])
            classifiers.append(model)
            names.append(c)
            logger.info("Added xgboost model %s", c)
        except AttributeError:
            logger.critical("Could not load xgboost model %s", c)

    return classifiers, names


def getclf_keras(model_config, length_input):

    logger = get_logger()
    logger.debug("Load keras models")

    if "keras" not in model_config:
        logger.debug("No keras models found")
        return [], []

    classifiers = []
    names = []

    for c in model_config["keras"]:
        try:
            def get_model():
                return getattr(templates_keras, c)(model_config["keras"][c], length_input)
            classifiers.append(KerasClassifier(build_fn=get_model,
                                               epochs=model_config["keras"][c]["epochs"],
                                               batch_size=model_config["keras"][c]["batch_size"],
                                               verbose=0))
            names.append(c)
            logger.info("Added keras model %s", c)
        except AttributeError:
            logger.critical("Could not load keras model %s", c)

    #logger.critical("Some reason")
    return classifiers, names



def fit(names_, classifiers_, x_train_, y_train_):
    trainedmodels_ = []
    for _, clf in zip(names_, classifiers_):
        clf.fit(x_train_, y_train_)
        trainedmodels_.append(clf)
    return trainedmodels_


def test(ml_type, names_, trainedmodels_, test_set_, mylistvariables_, myvariablesy_):
    x_test_ = test_set_[mylistvariables_]
    y_test_ = test_set_[myvariablesy_].values.reshape(len(x_test_),)
    test_set_[myvariablesy_] = pd.Series(y_test_, index=test_set_.index)
    for name, model in zip(names_, trainedmodels_):
        y_test_prediction = []
        y_test_prob = []
        y_test_prediction = model.predict(x_test_)
        y_test_prediction = y_test_prediction.reshape(len(y_test_prediction),)
        test_set_['y_test_prediction'+name] = pd.Series(y_test_prediction, index=test_set_.index)

        if ml_type == "BinaryClassification":
            y_test_prob = model.predict_proba(x_test_)[:, 1]
            test_set_['y_test_prob'+name] = pd.Series(y_test_prob, index=test_set_.index)
    return test_set_


def apply(ml_type, names_, trainedmodels_, test_set_, mylistvariablestraining_):
    x_values = test_set_[mylistvariablestraining_]
    for name, model in zip(names_, trainedmodels_):
        y_test_prediction = []
        y_test_prob = []
        y_test_prediction = model.predict(x_values)
        y_test_prediction = y_test_prediction.reshape(len(y_test_prediction),)
        test_set_['y_test_prediction'+name] = pd.Series(y_test_prediction, index=test_set_.index)

        if ml_type == "BinaryClassification":
            y_test_prob = model.predict_proba(x_values)[:, 1]
            test_set_['y_test_prob'+name] = pd.Series(y_test_prob, index=test_set_.index)
    return test_set_


def savemodels(names_, trainedmodels_, folder_, suffix_):
    for name, model in zip(names_, trainedmodels_):
        if "Keras" in name:
            architecture_file = folder_+"/"+name+suffix_+"_architecture.json"
            weights_file = folder_+"/"+name+suffix_+"_weights.h5"
            arch_json = model.model.to_json()
            with open(architecture_file, 'w') as json_file:
                json_file.write(arch_json)
            model.model.save_weights(weights_file)
        if "Scikit" in name:
            fileoutmodel = folder_+"/"+name+suffix_+".sav"
            pickle.dump(model, open(fileoutmodel, 'wb'))


def readmodels(names_, folder_, suffix_):
    trainedmodels_ = []
    for name in names_:
        fileinput = folder_+"/"+name+suffix_+".sav"
        model = pickle.load(open(fileinput, 'rb'))
        trainedmodels_.append(model)
    return trainedmodels_


def importanceplotall(mylistvariables_, names_, trainedmodels_, suffix_, folder):
    plt.figure(figsize=(25, 15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

    i = 1
    for name, model in zip(names_, trainedmodels_):
        if "SVC" in name:
            continue
        if "Logistic" in name:
            continue
        ax1 = plt.subplot(2, (len(names_)+1)/2, i)
        #plt.subplots_adjust(left=0.3, right=0.9)
        feature_importances_ = model.feature_importances_
        y_pos = np.arange(len(mylistvariables_))
        ax1.barh(y_pos, feature_importances_, align='center', color='green')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(mylistvariables_, fontsize=17)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel('Importance', fontsize=17)
        ax1.set_title('Importance features '+name, fontsize=17)
        ax1.xaxis.set_tick_params(labelsize=17)
        plt.xlim(0, 0.7)
        i += 1
    plt.subplots_adjust(wspace=0.5)
    plotname = folder+'/importanceplotall%s.png' % (suffix_)
    plt.savefig(plotname)
    img_import = BytesIO()
    plt.savefig(img_import, format='png')
    img_import.seek(0)
    return img_import


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
        ax.text(xx.max() - .3, yy.min() + .3, ('accuracy=%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right', verticalalignment='center')
        ax.set_title(name, fontsize=17)
        ax.set_ylabel(mylistvariables_[1], fontsize=17)
        ax.set_xlabel(mylistvariables_[0], fontsize=17)
        figure.subplots_adjust(hspace=.5)
        i += 1
    plotname = folder+'/decisionboundaries%s.png' % (suffix_)
    plt.savefig(plotname)
    img_boundary = BytesIO()
    plt.savefig(img_boundary, format='png')
    img_boundary.seek(0)
    return img_boundary
