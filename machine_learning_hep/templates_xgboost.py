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

from os.path import join

import pickle

from xgboost import XGBClassifier
from hyperopt import hp

from machine_learning_hep.optimisation.bayesian_opt import BayesianOpt
from machine_learning_hep.optimisation.metrics import get_scorers

def xgboost_classifier(model_config): # pylint: disable=W0613
    return XGBClassifier(verbosity=1,
                         n_gpus=0,
                         **model_config)


def xgboost_classifier_bayesian_space():
    return {"max_depth": hp.quniform("x_max_depth", 1, 6, 1),
            "n_estimators": hp.quniform("x_n_estimators", 600, 1000, 1),
            "min_child_weight": hp.quniform("x_min_child", 1, 4, 1),
            "subsample": hp.uniform("x_subsample", 0.5, 0.9),
            "gamma": hp.uniform("x_gamma", 0.0, 0.2),
            "colsample_bytree": hp.uniform("x_colsample_bytree", 0.5, 0.9),
            "reg_lambda": hp.uniform("x_reg_lambda", 0, 1),
            "reg_alpha": hp.uniform("x_reg_alpha", 0, 1),
            "learning_rate": hp.uniform("x_learning_rate", 0.05, 0.35),
            "max_delta_step": hp.quniform("x_max_delta_step", 0, 8, 2)}


class XGBoostClassifierBayesianOpt(BayesianOpt):


    def yield_model_(self, model_config, space):
        config = self.next_params(space)
        config["early_stopping_rounds"] = 10
        # NOTE If that's not really an integer, it will crash!
        if "n_estimators" in config:
            config["n_estimators"] = int(config["n_estimators"])
        if "max_depth" in config:
            config["max_depth"] = int(config["max_depth"])
        return xgboost_classifier(config), config


    def save_model_(self, model, out_dir):
        out_filename = join(out_dir, "xgboost_classifier.sav")
        pickle.dump(model, open(out_filename, 'wb'), protocol=4)
        out_filename = join(out_dir, "xgboost_classifier.model")
        model.save_model(out_filename)


def xgboost_classifier_bayesian_opt(model_config):
    bayesian_opt = XGBoostClassifierBayesianOpt(model_config, xgboost_classifier_bayesian_space())
    bayesian_opt.nkfolds = 3
    bayesian_opt.scoring = get_scorers(["AUC", "Accuracy"])
    bayesian_opt.scoring_opt = "AUC"
    bayesian_opt.low_is_better = False
    bayesian_opt.n_trials = 100
    return bayesian_opt
