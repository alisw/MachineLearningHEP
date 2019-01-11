from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso


def random_forest_classifier(model_config):
    return RandomForestClassifier(max_depth=model_config["max_depth"],
                                  n_estimators=model_config["n_estimators"],
                                  max_features=model_config["max_features"])


def adaboost_classifier(model_config):
    return AdaBoostClassifier()


def decision_tree_classifier(model_config):
    return DecisionTreeClassifier(max_depth=model_config["max_depth"])


def linear_regression(model_config):
    return LinearRegression()


def ridge(model_config):
    return Ridge(alpha=model_config["alpha"], solver=model_config["solver"])


def lasso(model_config):
    return Lasso(alpha=model_config["alpha"])
