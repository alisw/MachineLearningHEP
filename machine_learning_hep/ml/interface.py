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
Methods to: Construct, save and load ML models and their optimisation routines
"""


import pickle

from ..logger import get_logger
from .interface_keras import MODELS as models_keras
from .interface_scikit import MODELS as models_scikit
from .interface_xgboost import MODELS as models_xgboost

_MODEL_TYPES = ("BinaryClassification", "Regression")


def get_model_any(model_type, model_name, selection_lambda):
    """Return model parameters based on selection_lambda

    Args:
        model_type: str
            One of the valid model types
        model_name: str
            name of model below model_type
        selction_lambda: lambda(model_paramas)
            where model_params is a dictionary mapping
            construction functions to corresponding
            keys

    Returns:
        selection_lambda(model_params)
    """
    if model_type not in _MODEL_TYPES:
        get_logger().fatal("Unknown model_type %s. Available ones are %s", model_type,
                           ", ".join(_MODEL_TYPES))
    for model_dicts in (models_keras, models_scikit, models_xgboost):
        for model_type_, models in model_dicts.items():
            if model_type_ != model_type:
                continue
            for model_name_, model_params in models.items():
                if model_name_ != model_name:
                    continue
                return selection_lambda(model_params)


def get_model(model_type, model_name, n_features, **kwargs):
    """Get a model according to its type and name

    Args:
        model_type: str
            One of the valid model types
        model_name: str
            name of model below model_type
        n_features: int
            number of expected features
        kwargs: dict
            used to overwrite model configuration or bayesian space

    Returns:
        lambda(model_params) according to model_type and model_name
    """
    return get_model_any(model_type, model_name,
                         lambda x: x["model_nominal"](\
                                 model_config=kwargs.get("config_nominal", x["config_nominal"]()),
                                 n_features=n_features))


def get_bayesian_opt(model_type, model_name, n_features, **kwargs):
    """Get a Bayesian optimiser according to its type and name

    Args:
        model_type: str
            One of the valid model types
        model_name: str
            name of model below model_type
        n_features: int
            number of expected features
        kwargs: dict
            used to overwrite model configuration or bayesian space

    Returns:
        construct_bayesian(model_params) according to model_type and model_name
    """
    def construct_bayesian(params):
        if "bayesian_opt" not in params or not params["bayesian_opt"]:
            return None
        return params["bayesian_opt"](\
                model_config=kwargs.get("config_nominal", params["config_nominal"]()),
                space=kwargs.get("space", params["bayesian_space"]()),
                n_features=n_features)

    return get_model_any(model_type, model_name, construct_bayesian)


def savemodels(names_, trainedmodels_, folder_, suffix_):
    for name, model in zip(names_, trainedmodels_):
        if "keras" in name:
            architecture_file = folder_+"/"+name+suffix_+"_architecture.json"
            weights_file = folder_+"/"+name+suffix_+"_weights.h5"
            arch_json = model.model.to_json()
            with open(architecture_file, 'w') as json_file:
                json_file.write(arch_json)
            model.model.save_weights(weights_file)
        if "scikit" in name:
            fileoutmodel = folder_+"/"+name+suffix_+".sav"
            pickle.dump(model, open(fileoutmodel, 'wb'), protocol=4)
        if "xgboost" in name:
            fileoutmodel = folder_+"/"+name+suffix_+".sav"
            pickle.dump(model, open(fileoutmodel, 'wb'), protocol=4)
            fileoutmodel = fileoutmodel.replace(".sav", ".model")
            model.save_model(fileoutmodel)


def readmodels(names_, folder_, suffix_):
    trainedmodels_ = []
    for name in names_:
        fileinput = folder_+"/"+name+suffix_+".sav"
        model = pickle.load(open(fileinput, 'rb'))
        trainedmodels_.append(model)
    return trainedmodels_
