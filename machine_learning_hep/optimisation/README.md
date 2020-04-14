# Machine Learning and optimisation


## Basic Machine Learning

Files to be moved here...


## Bayesian optimisation

Bayesian optimiastion can be used instead of a brute-force grid search to optimise the hyperparameters of that model. It might superior in the sense that it does not try all possible combinations of parameters to be varied. Instead, it takes previous performance and parameter settings, to decide on setting for a next trial.

This package uses Bayesian optimisation for its models via the [hyperopt package](https://github.com/hyperopt/hyperopt). In the easiest case a parameter space us defined to draw the values from. As an example, this can be seen in `templates_xgboost.py` where one can find the corresponding implementation

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

In this case all parameters are uniformly distributed, but there are more ways to do that (see [hyperopt's wiki](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions)). For each **trial**, a new set is drawn and used for fitting the next model.

More explanation is coming soon...


## How to use it

In order to use Bayesian optimisation for a model, you need to do the following (`templates_xgboost.py` is taken here as an example):

1. derive a class from `BayesianOpt` (if you are interested, you can find it in `optimisation/bayesian_opt.py` in this package)
2. implement the method `yield_model_(model_config, space)` which needs to return your classifier constructed from the configuration parameters passed via `model_config` (parameters for central model) and `space` (drawn from the parameter space as explained above). The user has to decide how to combine/overwrite parameters of the central configuration. Of course, it would be overhead to draw from many parameters but only use a very few... **NOTE** that all parameters in `space` are floating point numbers so in case your model would crash if you pass such a thing instead of - say - a proper integer (as it would be the case for XGBoost's `n_estimators`), it's your fault. So cast it if needed.
3. implement the method `save_model_(model, out_dir)` where `out_dir` is the directory to save the model in. This has to be done by the user because `BayesianOpt` has inherently no idea about the actual model implementation; hence, it cannot know how to save it
4. provide a function `<full_model_name>_bayesian_opt(model_config)` whose only task is to return an instance of your derived `BayesianOpt` class. This function will be called from inside the package later and will pass the central parameters via `model_config`. This can just be forwarded as the first argument in the constructor while the second one should be the space needed by `hyperopt`. For XGBoost models in the package, this is constructed and returned by `xgboost_classifier_bayesian_opt_space()`.

That's it and as you can see, you need roughly only 30 lines to put a full blown Bayesian optimisation in place.
