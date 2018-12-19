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

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_extraction import DictVectorizer

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier


def getclf_scikit(ml_type):
    classifiers = []
    names = []
    if ml_type == "BinaryClassification":
        classifiers = [
            #GradientBoostingClassifier(learning_rate=0.01, n_estimators=2500, max_depth=1),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            DecisionTreeClassifier(max_depth=5)
            # LinearSVC(C=1, loss="hinge"),
            # SVC(kernel="rbf", gamma=5, C=0.001), LogisticRegression()
        ]

        names = [
            # "ScikitTreeGradientBoostingClassifier",
            "ScikitTreeRandom_Forest", "ScikitTreeAdaBoost", "ScikitTreeDecision_Tree"
            #       "ScikitLinearSVC", "ScikitSVC_rbf","ScikitLogisticRegression"
        ]

    if ml_type == "Regression":
        classifiers = [
            LinearRegression(), Ridge(alpha=1, solver="cholesky"), Lasso(alpha=0.1)
        ]

        names = [
            "Scikit_linear_regression", "Scikit_Ridge_regression", "Scikit_Lasso_regression"

        ]
    return classifiers, names


def getclf_xgboost(ml_type):
    classifiers = []
    names = []

    if ml_type == "BinaryClassification":
        classifiers = [XGBClassifier()]
        names = ["XGBoostXGBClassifier"]

    if ml_type == "Regression":
        print("No XGBoost models implemented for Regression")
    return classifiers, names


def getclf_keras(ml_type, length_input):
    classifiers = []
    names = []

    if ml_type == "BinaryClassification":
        def create_model_sequential():
            model = Sequential()
            model.add(Dense(12, input_dim=length_input, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        classifiers = [KerasClassifier(build_fn=create_model_sequential,
                                       epochs=1000, batch_size=50, verbose=0)]
        names = ["KerasSequential"]

    if ml_type == "Regression":
        print("No Keras models implemented for Regression")
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
