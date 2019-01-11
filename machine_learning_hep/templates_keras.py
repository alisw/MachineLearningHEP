from keras.layers import Input, Dense
from keras.models import Model

def simple_one_layer_binary_class(model_config, length_input):
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

def simple_two_layer_binary_class(model_config, length_input):
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
