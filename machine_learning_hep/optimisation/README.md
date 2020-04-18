# Machine Learning and optimisation


## Basic Machine Learning

ML and optimisation `Python` source files to be moved here...


## Bayesian optimisation

Bayesian optimiastion can be used instead of a brute-force grid search to optimise the hyperparameters of a model. It might superior in the sense that it does not try all possible combinations of parameters to be varied. Instead, it takes previous performance and parameter settings, to decide on setting for a next **trial**.

This package uses Bayesian optimisation for its models via the [hyperopt package](https://github.com/hyperopt/hyperopt). A parameter space is defined to draw the values from which is already done in `templates_xgboost.py` where one can find the corresponding implementation:

```python

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
```

In this case all parameters are uniformly distributed (there are more ways to do that, see [hyperopt's wiki](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions)). For each trial, a new set is drawn and used for fitting the next model.

More explanation is coming soon...


## How to use it

PREREQUISITES: before running the bayesian optimisation routine please repeat the installation of the required package by running the usual command (see below). By doing this you will make sure that the package hyperopt, used for the optimisation, is included in your virtual environment 
```python
pip3 install -e .
```

In order to use Bayesian optimisation for a model, you need to do the following (`templates_xgboost.py` is taken here as an example):

1. Derive a class from `BayesianOpt` (if you are interested, you can find it in `optimisation/bayesian_opt.py` in this package).
2. Implement the method `yield_model_(self, model_config, space)` which must return a model constructed from the configuration parameters passed via `model_config` (parameters for central model) and `space` (drawn from the parameter space as explained above). The user still has the freedom to combine/overwrite parameters of the central configuration. Of course, it would be overhead to draw from many parameters but only use a few... **NOTE** that all parameters in `space` are floating point numbers and it is you responsibility to cast that to an integer if needed (as it would be e.g. necessary for XGBoost's `n_estimators`).
3. Implement the method `save_model_(self, model, out_dir)` where `out_dir` is the directory to save the `model` in. This has to be done by the user because `BayesianOpt` has inherently no idea about the actual model implementation; hence, it cannot know how to save it.

These first three steps are in principle independent of the package and one could use this class easily somewhere else to use this type of optimisation. At some point, it has to be made sure that further members of `BayesianOpt` are set, such as the training data and other parameters:

```python

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

```

Afterwards, it can be run like (here you see which members **must** be set:


```python
bayes_opt.x_train = x_train # must be set
bayes_opt.y_train = y_train # must be set
bayes_opt.nkfolds = 5 # can be changed, default is 1
bayes_opt.scoring = {"AUC": auc_scorer, "Accuracy": accuracy_scorer} # needs to be a dictionary mapping a scoring function to its name, all metrics are evaluated
bayes_opt.scoring_opt = "AUC" # must be one key from the above, this is used for optimisation
bayes_opt.low_is_better = False # indicate if metric needs to be minimised or maximised
bayes_opt.n_trials = 100 # can be changed, default is indeed 100

bayes_opt.optimise(ncores=ncores) # optimisation, number of cores can be set on-the-fly
bayes_opt.save(out_dir) # save results and model in output_dir
bayes_opt.plot(out_dir) # plot results in output_dir

```

The **package specific** part is to provide a function `<full_model_name>_bayesian_opt(model_config)` in the `templates_<model_class>.py` whose only task is to return an instance of your derived `BayesianOpt` class. `model_config` are again the central model parameters which - in the MLHEP package - are defined in `data/config_model_parameters.yml`. This can just be forwarded as the first argument in the constructor while the second one should be the space needed by `hyperopt`. For XGBoost models in the package, this is constructed and returned by `xgboost_classifier_bayesian_opt_space()` as already mentioned above.

That's it and as you can see, you need roughly only 30 lines to put a full blown Bayesian optimisation in place.
