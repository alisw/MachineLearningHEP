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
import numpy as np

from sklearn.model_selection import cross_validate

from hyperopt import fmin, tpe, STATUS_OK

from machine_learning_hep.logger import get_logger
from machine_learning_hep.io import dump_yaml_from_dict


class BayesianOpt: #pylint: disable=too-many-instance-attributes

    def __init__(self, model_config, space):

        # Train samples
        self.x_train = None
        self.y_train = None

        # Nominal model configuration dict
        self.model_config = model_config

        # Space to draw parameter values for Bayesian optimisation
        self.space = space

        # KFolds for CV
        self.nkfolds = 1

        # Number of trials
        self.n_trials = 100

        # Scorers
        self.scoring = None
        # Optimise with this score
        self.scoring_opt = None

        # Min- or maximise?
        self.low_is_better = True
        # Current minimum score
        self.min_score = None

        # Lambda to yield a custom classifier on the fly
        self.yield_clf_custom = None
        self.save_clf_custom = None

        # Collect...

        # ...CV results
        self.results = []

        # ...parameters of trial
        self.params = []

        # ...best classifier and index to find score value/parameters etc.
        self.best_index = None
        self.best = None

        # Number of parallel jobs
        self.ncores = 20

        # In order to have proper logging output
        self.logger = get_logger()

    def reset(self):
        """Reset to default
        """

        self.min_score = None
        self.results = []
        self.params = []
        self.best_index = None
        self.best = None


    def yield_clf(self, space): # pylint: disable=unused-argument, useless-return
        """Yield next classifier

        Next classifier constructed from space. To be overwritten for concrete implementation

        Args:
            space: dict of sampled parameters

        Returns: classifier

        """
        self.logger.error("Not implemented...")
        return None, None


    def next_params(self, space):
        """Yield next set of parameters

        This is a helper function which can be used to construct the next classifier

        """
        config = {}
        for key, value in space.items():
            config[key] = value
        if self.model_config:
            for key, value in self.model_config.items():
                if key not in config:
                    config[key] = value
        return config


    def trial(self, space):
        """One trial

        Doing one trial with a next configured classifier

        Args:
            clf: classifier

        Returns: dict of score and status
        """

        print("Next trial")

        clf = None
        params = None
        # Yield classifier and parameters on the fly or with class method
        if self.yield_clf_custom:
            clf, params = self.yield_clf_custom(self.model_config, space)
        else:
            clf, params = self.yield_clf(space) # pylint: disable=assignment-from-none

        # Collect parameters
        self.params.append(params)

        # Do cross validation for this classifier
        res = cross_validate(clf, self.x_train, self.y_train, cv=self.nkfolds,
                             scoring=self.scoring, n_jobs=self.ncores, return_train_score=True)

        # Collect results
        res_tmp = {f"test_{name}":  float(np.mean(res[f"test_{name}"])) for name in self.scoring} # pylint: disable=not-an-iterable
        self.results.append(res_tmp)

        # Extract mean score from CV
        score = np.mean(res[f"test_{self.scoring_opt}"])

        # Because we minimise always, needs to be
        if not self.low_is_better:
            score = -score

        if self.min_score is None or score < self.min_score:
            self.min_score = score
            self.best = clf
            self.best_index = len(self.params) - 1

        return {"loss": score, "status": STATUS_OK}


    def optimise(self, yield_clf=None, save_clf=None, ncores=None):
        """Do Bayesian optimisation

        Parent function using

        """

        if self.params:
            self.logger.warning("Already optimised, call reset() to run again")

        self.logger.info("Do Bayesian optimisation")

        if ncores:
            self.ncores = ncores

        # Potentially yield a custom classifier on the fly
        self.yield_clf_custom = yield_clf
        self.save_clf_custom = save_clf
        if yield_clf and save_clf is None:
            self.logger.fatal("Classifier is created on the fly but no save method was provided")

        _ = fmin(fn=self.trial, space=self.space, algo=tpe.suggest, max_evals=self.n_trials)

        self.ncores = 20

        # Now, train the best model on the full dataset
        self.best.fit(self.x_train, self.y_train)


    def make_results(self):
        return {"cv": self.results,
                "params": self.params,
                "best_index": self.best_index}


    def save_classifier_(self, clf, out_dir): # pylint: disable=unused-argument
        """Save a model

        Routine to save a model, to be implemented for concrete model

        """

        self.logger.error("Not implemented...")


    def save(self, out_dir):
        dump_yaml_from_dict(self.make_results(), join(out_dir, "results.yaml"))
        self.logger.info("Save best classifier from Bayesian opt at %s", out_dir)
        if self.yield_clf_custom and self.save_clf_custom:
            self.save_clf_custom(self.best, out_dir)
        else:
            self.save_classifier_(self.best, out_dir)


    def plot(self, out_dir):
        pass
