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

from copy import deepcopy

from keras.layers import Input, Dense
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier

from hyperopt import hp

from machine_learning_hep.mlbase.bayesian_opt import BayesianOpt
from machine_learning_hep.mlbase.metrics import get_scorers


def keras_classifier_(model_config, input_length):
    """
    NN for binary classification with 1 hidden layers
    """
    # Create layers
    inputs = Input(shape=(input_length,))
    layer = Dense(model_config["layers"][0]["n_nodes"],
                  activation=model_config["layers"][0]["activation"])(inputs)
    predictions = Dense(1, activation='sigmoid')(layer)
    # Build model from layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=model_config["loss"], optimizer=model_config["optimizer"],
                  metrics=['accuracy'])
    return model


def keras_classifier(**kwargs):
    model_config = kwargs["model_config"]
    input_length = kwargs["n_features"]
    return KerasClassifier(build_fn=lambda: \
                    keras_classifier_(model_config, input_length), \
                                      epochs=model_config["epochs"], \
                                      batch_size=model_config["batch_size"], \
                                      verbose=1)


def keras_classifier_config_nominal():
    return {"layers": [{"n_nodes": 12, "activation": "relu"}],
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "epochs": 30,
            "batch_size": 50}


def keras_classifier_bayesian_space():
    return {"n_nodes": hp.choice("x_n_nodes", [[12, 64], [12], [12, 64, 16]]),
            "activation_0": hp.choice("x_activation_0", ["relu", "sigmoid"]),
            "activation_1": hp.choice("x_activation_1", ["relu", "sigmoid"]),
            "epochs": hp.quniform("x_epochs", 50, 100, 1),
            "batch_size": hp.quniform("x_batch_size", 28, 256, 1)}


class KerasClassifierBayesianOpt(BayesianOpt): # pylint: disable=too-many-instance-attributes


    def __init__(self, model_config, space, input_length):
        super().__init__(model_config, space)
        self.input_length = input_length
        # Cache drawn space and model config to build the model several times in
        # self.get_scikit_model (should have these available but cannot take arguments
        self.model_config_tmp = None
        self.space_tmp = None


    def get_scikit_model(self):
        """Just a helper funtion

        KerasClassifier needs something callable to obtain the model

        """
        inputs = Input(shape=(self.input_length,))
        layer = Dense(self.space_tmp["n_nodes"][0],
                      activation=self.space_tmp["activation_0"])(inputs)
        for i, n_nodes in enumerate(self.space_tmp["n_nodes"][1:]):
            layer = Dense(n_nodes,
                          activation=self.space_tmp[f"activation_{(i+1)%2}"])(layer)
        predictions = Dense(1, activation='sigmoid')(layer)
        # Build model from layers
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss=self.model_config_tmp["loss"],
                      optimizer=self.model_config_tmp["optimizer"],
                      metrics=['accuracy'])
        return model


    def yield_model_(self, model_config, space):

        self.space_tmp = deepcopy(space)
        self.model_config_tmp = deepcopy(model_config)

        return KerasClassifier(build_fn=self.get_scikit_model, epochs=int(space["epochs"]),
                               batch_size=int(space["batch_size"]), verbose=1), space


    def save_model_(self, model, out_dir):
        """Not implemented yet
        """


def keras_classifier_bayesian_opt(**kwargs):
    bayesian_opt = KerasClassifierBayesianOpt(kwargs["model_config"], kwargs["space"],
                                              kwargs["n_features"])
    bayesian_opt.nkfolds = 3
    bayesian_opt.scoring = get_scorers(["AUC", "Accuracy"])
    bayesian_opt.scoring_opt = "AUC"
    bayesian_opt.low_is_better = False
    bayesian_opt.n_trials = 30
    return bayesian_opt


MODELS = {"BinaryClassification": {"keras_classifier": {"model_nominal": keras_classifier,
                                                        "config_nominal": \
                                                                keras_classifier_config_nominal,
                                                        "bayesian_space": \
                                                                keras_classifier_bayesian_space,
                                                        "bayesian_opt": \
                                                                keras_classifier_bayesian_opt}}}
