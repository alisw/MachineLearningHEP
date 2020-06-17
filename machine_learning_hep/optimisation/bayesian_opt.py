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

import sys
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

from yaml.representer import RepresenterError

from sklearn.model_selection import cross_validate

from hyperopt import fmin, tpe, STATUS_OK

from machine_learning_hep.io import dump_yaml_from_dict, parse_yaml

# Change to that backend to not have problems with saving fgures
# when X11 connection got lost
matplotlib.use("agg")


class BayesianOpt: #pylint: disable=too-many-instance-attributes
    """Base/utilitiy class for Bayesian model optimisation

        This class utilises the hyperopt package to perform Bayesian model optimisation independent
        of the concrete ML model.
        The central method is "optimise" which soleyly relies on getting a model configured with
        the new parameters. A method method to obtain a new model can either be implemented by
        deriving this class and overwrite "yield_model_" or by passing a lambda as the
        "yield_model" argument when calling "optimise".
        Additionally, the best model is automatically saved when either "save_model_" is
        overwritten or a lambda is passed to the "save_model" argument in optimise.

        Optimisation is done "self.n_trials" times and for each trial a Cross Validation is done
        with "self.nkfolds" folds.

        Scoring functions can be freely defined in contained in the dictionary "self.scoring" and
        the optimisation is done according to the scoring function with key "self.scoring_opt".
        Note, that the underlying optimisation procedure is a minimisation. Hence, when a maximum
        score is the best one, "self.low_is_better" must be set to False.

        All parameters and scores can be written to a YAML file and the field "best_index"
        specifies the best model wrt the best test score.
    """

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

        # Lambda to yield a custom model on the fly
        self.yield_model_custom = None
        self.save_model_custom = None

        # Collect...

        # ...CV results
        self.results = []

        # ...parameters of trial
        self.params = []

        # ...models
        self.models = []

        # ...best model and index to find score value/parameters etc.
        self.best_index = None
        self.best = None
        self.best_params = None
        self.best_scores = None

        # Number of parallel jobs
        self.ncores = 20

        self.fit_pool = []


    def reset(self):
        """Reset to default
        """

        self.min_score = None
        self.results = []
        self.params = []
        self.models = []
        self.best_index = None
        self.best = None
        self.best_params = None
        self.best_scores = None


    def yield_model_(self, model_config, space): # pylint: disable=unused-argument, useless-return, no-self-use
        """Yield next model

        Next model constructed from space. To be overwritten for concrete implementation

        Args:
            space: dict of sampled parameters

        Returns: model

        """
        print("yield_model_ not implemented...")
        return None, None


    def next_params(self, space_drawn):
        """Yield next set of parameters

        Helper function which can be used to extract parameters for next model

        """
        config = {}
        for key, value in space_drawn.items():
            config[key] = value
        if self.model_config:
            for key, value in self.model_config.items():
                if key not in config:
                    config[key] = value
        return config


    def trial_(self, space_drawn):
        """Default single trial

        Args:
            space_drawn: dict
                sampled new parameters
        Returns:
            res: dict
                dictionary with CV results
            model: model used in this trial
            params: dict
                parameters used in this trial
        """
        model = None
        params = None
        # Yield model and parameters on the fly or with class method
        if self.yield_model_custom:
            model, params = self.yield_model_custom(self.model_config, space_drawn)
        else:
            model, params = self.yield_model_(self.model_config, space_drawn) # pylint: disable=assignment-from-none

        # Collect parameters
        self.params.append(params)

        # Do cross validation for this model
        res = cross_validate(model, self.x_train, self.y_train, cv=self.nkfolds,
                             scoring=self.scoring, n_jobs=self.ncores, return_train_score=True)

        return res, model, params


    def trial(self, space_drawn):
        """One trial

        Doing one trial with a next configured model

        Args:
            model: model

        Returns: dict of score and status
        """

        res, model, params = self.trial_(space_drawn)

        # Collect results
        res_tmp = {}
        for t in ("train", "test"):
            for sc in self.scoring: # pylint: disable=not-an-iterable
                res_tmp[f"{t}_{sc}"] = float(np.mean(res[f"{t}_{sc}"]))
                res_tmp[f"{t}_{sc}_std"] = float(np.std(res[f"{t}_{sc}"]))
        self.results.append(res_tmp)
        self.models.append(model)
        self.params.append(params)

        # Extract mean score from CV
        score = np.mean(res[f"test_{self.scoring_opt}"])

        # Because we minimise always, needs to be
        if not self.low_is_better:
            score = -score

        if self.min_score is None or score < self.min_score:
            self.min_score = score
            self.best = model
            self.best_index = len(self.params) - 1
            self.best_params = params
            self.best_scores = res_tmp

        return {"loss": score, "status": STATUS_OK}


    def finalise(self):
        """Finalising...
        """

        # Reset number of cores
        self.ncores = 20

        # Now, train the best model on the full dataset
        if self.best:
            print("Fit best model to whole dataset")
            self.best.fit(self.x_train, self.y_train)


    def optimise(self, yield_model=None, save_model=None, space=None, ncores=None):
        """Do Bayesian optimisation

        Central function to be called for the optimisation. Takes care of running a CV for all
        trials.

        Args:
            yield_model: lambda(space) (optional)
                Hyperopt parameter space to draw parameters from.
                If not passed, it is assumed that this is called from a derived class implementing
                self.yield_model_
            save_model: lambda(model, out_dir) (optional)
                Procedure to save a model. Since this class does not know the details, it
                cannot know how to save a model.
                If not passed, it is assumed that this is called from a derived class implementing
                self.save_model_
            space: hyperopt space (optional)
                On the fly set the hyperopt space for this optimisation to draw parameters
                from.
            ncores: int
                number of cores to be used
        """

        if self.params:
            print("Already optimised, call reset() to run again")
            return

        print("Do Bayesian optimisation")

        if ncores:
            self.ncores = ncores

        if space:
            self.space = space

        # Potentially yield a custom model on the fly
        self.yield_model_custom = yield_model
        self.save_model_custom = save_model
        if yield_model and save_model is None:
            print("Model is created on the fly but no save method was provided")
            sys.exit(1)

        try:
            _ = fmin(fn=self.trial, space=self.space, algo=tpe.suggest, max_evals=self.n_trials)
        except KeyboardInterrupt:
            self.finalise()
        else:
            self.finalise()


    def make_results(self):
        """Helper function to make dictionary of parameters and results
        """

        return {"cv": self.results,
                "params": self.params,
                "best_index": self.best_index,
                "best_params": self.best_params,
                "best_scores": self.best_scores,
                "score_names": list(self.scoring.keys()),
                "score_opt_name": self.scoring_opt}


    def save_model_(self, model, out_dir): # pylint: disable=unused-argument, no-self-use
        """Save a model

        Routine to save a model, to be implemented for concrete model

        """
        print("save_model_  not implemented")


    def save(self, out_dir, best_only=True):
        """Save paramaters/results and best model
        """

        results = self.make_results()
        try:
            dump_yaml_from_dict(results, join(out_dir, "results.yaml"))
        except RepresenterError:
            print("Cannot save optimisation results as YAML")

        try:
            pickle.dump(results, open(join(out_dir, "results.pkl"), "wb"))
        except Exception: #pylint: disable=broad-except
            print("Cannot pickle optimisation results")


        save_func = self.save_model_
        print(f"Save best model from Bayesian opt at {out_dir}")
        if self.yield_model_custom and self.save_model_custom:
            save_func = self.save_model_custom
        save_func(self.best, out_dir)

        if not best_only:
            # Save all models
            for i, m in enumerate(self.models):
                out_dir_model = join(out_dir, f"model_{i}")
                save_func(m, out_dir_model)


    def plot(self, out_dir, from_yaml=None, from_pickle=None): # pylint: disable=unused-argument, too-many-statements
        """Plot results

        Results are plotted to out_dir/results.png

        Args:
            out_dir: str
                output directory where results.png will be saved
            from_yaml: str
                path to YAML file to read and plot results from

        """

        results_tmp = self.results
        scores_tmp = list(self.scoring.keys())
        score_opt_tmp = self.scoring_opt

        if from_yaml:
            read_yaml = parse_yaml(from_yaml)
            results_tmp = read_yaml["cv"]
            scores_tmp = read_yaml["score_names"]
            score_opt_tmp = read_yaml["score_opt_name"]
        elif from_pickle:
            read_yaml = pickle.load(open(from_pickle, "rb"))
            results_tmp = read_yaml["cv"]
            scores_tmp = read_yaml["score_names"]
            score_opt_tmp = read_yaml["score_opt_name"]


        # Re-arrange such that always the optimisation score is on top
        score_names = list(scores_tmp)
        del score_names[score_names.index(score_opt_tmp)]
        score_names.insert(0, score_opt_tmp)

        # Prepare figrue and axes
        figsize = (35, 18 * len(score_names))
        fig, axes = plt.subplots(len(score_names), 1, sharex=True, gridspec_kw={"hspace": 0.05},
                                 figsize=figsize)

        # If only one score is given, need to make it iterable
        try:
            iter(axes)
        except TypeError:
            axes = [axes]

        markerstyles = ["o", "+"]
        markersize = 20
        for axi, (sn, ax) in enumerate(zip(score_names, axes)):
            ax.set_ylabel(f"CV mean {sn}", fontsize=20)
            ax.get_yaxis().set_tick_params(labelsize=20)

            # Get means of scores and plot with their std
            means = {}
            for i, tt in enumerate(("train", "test")):
                markerstyle = markerstyles[i % len(markerstyles)]
                means[tt] = [r[f"{tt}_{sn}"] for r in results_tmp]
                stds = [r[f"{tt}_{sn}_std"] for r in results_tmp]
                ax.errorbar(range(len(means[tt])), means[tt], yerr=stds, ls="",
                            marker=markerstyle, markersize=markersize, label=f"{sn} ({tt})")

            # Relative deviations between test and train
            index_high_score = means["test"].index(max(means["test"]))
            dev_high_score = \
                    abs(means["test"][index_high_score] - means["train"][index_high_score]) \
                    / means["test"][index_high_score]
            index_low_score = means["test"].index(min(means["test"]))
            dev_low_score = \
                    abs(means["test"][index_low_score] - means["train"][index_low_score]) \
                    / means["test"][index_low_score]
            dev_min = [abs(test - train) / test \
                    for train, test in zip(means["train"], means["test"])]
            index_min = dev_min.index(min(dev_min))
            dev_min = min(dev_min)

            ax.axvline(index_high_score, color="red")
            y_coord = (means["test"][index_high_score] + means["train"][index_high_score]) / 2
            ax.text(index_high_score, y_coord, f"{dev_high_score:.4f}", color="red", fontsize=20)
            ax.axvline(index_low_score, color="blue")
            y_coord = (means["test"][index_low_score] + means["train"][index_low_score]) / 2
            ax.text(index_low_score, y_coord, f"{dev_low_score:.4f}", color="blue", fontsize=20)
            ax.axvline(index_min, color="green")
            y_coord = (means["test"][index_min] + means["train"][index_min]) / 2
            ax.text(index_min, y_coord, f"{dev_min:.4f}", color="green", fontsize=20)

            leg = ax.legend(loc="upper right", fontsize=20)
            if axi == 0:
                # Add another legend for highest, lowest score and min. rel. deviation between
                # test and train score
                handles = [Line2D([0], [0], color="red"),
                           Line2D([0], [0], color="blue"),
                           Line2D([0], [0], color="green")]
                labels = ["highest test score", "lowest test score", "min. rel deviation"]
                ax.legend(handles, labels, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                          ncol=3, mode="expand", borderaxespad=0., fontsize=20)
                # Add back first legend
                ax.add_artist(leg)

        axes[-1].set_xticks(range(len(results_tmp)))
        axes[-1].set_xticklabels(range(len(results_tmp)), fontsize=20)
        axes[-1].set_xlabel("# trial", fontsize=20)
        fig.suptitle("Bayesian model optimisation", fontsize=35)

        fig.tight_layout()
        out_file = join(out_dir, "results.png")
        fig.savefig(out_file)
        plt.close(fig)
