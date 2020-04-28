# Different (scikit) wrappers for models
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBClassifier

# MLHEP specific
from machine_learning_hep.ml.interface import get_model, get_bayesian_opt
from machine_learning_hep.mlbase.bayesian_opt import BayesianOpt

# Pretending we had no knowledge about the models implemented in the
# MLHEP interface
MODELS_ = {"BinaryClassification": [("keras_classifier", KerasClassifier),
                                    ("scikit_random_forest", RandomForestClassifier),
                                    ("scikit_adaboost", AdaBoostClassifier),
                                    ("scikit_decision_tree", DecisionTreeClassifier),
                                    ("xgboost_classifier", XGBClassifier)],
           "Regression": [("scikit_lasso", Lasso),
                          ("scikit_linear", LinearRegression),
                          ("scikit_ridge", Ridge)]}


BAYESIANS_ = {"BinaryClassification": [("keras_classifier", BayesianOpt),
                                       ("xgboost_classifier", BayesianOpt)]}


def test_load_models():
    """Check loading models

    Loading with dummy number of features
    """

    for model_type, models in MODELS_.items():
        for model_name, model_class in models:
            assert isinstance(get_model(model_type, model_name, 2), model_class)


def test_load_bayesian():
    """Check loading Bayesian optimisers

    Loading with dummy number of features
    """

    for model_type, models in BAYESIANS_.items():
        for model_name, model_class in models:
            assert isinstance(get_bayesian_opt(model_type, model_name, 2), model_class)
