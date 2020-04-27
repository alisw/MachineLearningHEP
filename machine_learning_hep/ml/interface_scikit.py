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


########################
# BinaryClassification #
########################

#################
# Random forest #
#################

def random_forest(**kwargs):
    model_config = kwargs["model_config"]
    return RandomForestClassifier(max_depth=model_config["max_depth"],
                                  n_estimators=model_config["n_estimators"],
                                  max_features=model_config["max_features"])


def random_forest_config_nominal():
    return {"max_depth": 5,
            "n_estimators": 10,
            "max_features": 1}


############
# Adaboost #
############

def adaboost(**kwargs): # pylint: disable=unused-argument
    return AdaBoostClassifier()


def adaboost_config_nominal():
    return {}


#################
# decision tree #
#################

def decision_tree(**kwargs):
    return DecisionTreeClassifier(max_depth=kwargs["model_config"]["max_depth"])


def decision_tree_config_nominal():
    return {"max_depth": 5}


##############
# Regression #
##############

#####################
# linear regression #
#####################

def linear(**kwargs): # pylint: disable=unused-argument
    return LinearRegression()


def linear_config_nominal():
    return {}


####################
# ridge regression #
####################

def ridge(**kwargs):
    model_config = kwargs["model_config"]
    return Ridge(alpha=model_config["alpha"], solver=model_config["solver"])


def ridge_config_nominal():
    return {"alpha": 1,
            "solver": "cholesky"}


####################
# lasso regression #
####################

def lasso(**kwargs):
    return Lasso(alpha=kwargs["model_config"]["alpha"])


def lasso_config_nominal():
    return {"alpha": 0.1}


##################################
# Final collection of all models #
##################################


MODELS = {"BinaryClassification": {"scikit_random_forest": {"model_nominal": random_forest,
                                                            "config_nominal": \
                                                                    random_forest_config_nominal,
                                                            # This is just left here as an example
                                                            "bayesian_opt": None,
                                                            # This is just left here as an example
                                                            "bayesian_space": None},
                                   "scikit_adaboost": {"model_nominal": adaboost,
                                                       "config_nominal": adaboost_config_nominal},
                                   "scikit_decision_tree": {"model_nominal": decision_tree,
                                                            "config_nominal": \
                                                                    decision_tree_config_nominal}},
          "Regression": {"scikit_lasso": {"model_nominal": lasso,
                                          "config_nominal": lasso_config_nominal},
                         "scikit_linear": {"model_nominal": linear,
                                           "config_nominal": linear_config_nominal},
                         "scikit_ridge": {"model_nominal": ridge,
                                          "config_nominal": ridge_config_nominal}}}
