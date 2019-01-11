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

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def random_forest_classifier(model_config):
    return RandomForestClassifier(max_depth=model_config["max_depth"],
                                  n_estimators=model_config["n_estimators"],
                                  max_features=model_config["max_features"])


def adaboost_classifier(model_config): # pylint: disable=W0613
    return AdaBoostClassifier()


def decision_tree_classifier(model_config):
    return DecisionTreeClassifier(max_depth=model_config["max_depth"])


def linear_regression(model_config): # pylint: disable=W0613
    return LinearRegression()


def ridge(model_config):
    return Ridge(alpha=model_config["alpha"], solver=model_config["solver"])


def lasso(model_config):
    return Lasso(alpha=model_config["alpha"])
