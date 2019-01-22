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

from keras.layers import Input, Dense
from keras.models import Model

def keras_simple_one_layer_binary_classifier(model_config, length_input):
    """
    NN for binary classification with 1 hidden layers
    """
    # Create layers
    inputs = Input(shape=(length_input,))
    layer = Dense(model_config["layers"][0]["n_nodes"],
                  activation=model_config["layers"][0]["activation"])(inputs)
    predictions = Dense(1, activation='sigmoid')(layer)
    # Build model from layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=model_config["loss"], optimizer=model_config["optimizer"],
                  metrics=['accuracy'])
    return model

def keras_simple_two_layer_binary_classifier(model_config, length_input):
    """
    NN for binary classification with 2 hidden layers
    """
    # Create layers
    inputs = Input(shape=(length_input,))
    layer = Dense(model_config["layers"][0]["n_nodes"],
                  activation=model_config["layers"][0]["activation"])(inputs)
    layer = Dense(model_config["layers"][1]["n_nodes"],
                  activation=model_config["layers"][1]["activation"])(layer)
    predictions = Dense(1, activation='sigmoid')(layer)
    # Build model from layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=model_config["loss"], optimizer=model_config["optimizer"],
                  metrics=['accuracy'])
    return model
