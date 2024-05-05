# pylint: disable=E0401
"""
This script is used to train the model. It loads the tokenized data and the
parameters from the dvc.yaml file.
"""

import pickle
import os
import dvc.api

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

PARAMS = dvc.api.params_show()

def create_model(char_index_size, num_categories):
    """ Create the model to be used for training.

    Args:
        char_index_size (int): Size of the character index.
        num_categories (int): Number of categories in the output layer.

    Returns:
        Sequential: The constructed Keras model.
    """

    model = Sequential()
    model.add(Embedding(char_index_size + 1, 50))
    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    # Additional layers can be configured similarly
    # For brevity, other layers are not shown here

    model.add(Flatten())
    model.add(Dense(num_categories, activation='sigmoid'))

    return model

def load_pickle(path):
    """ Load the pickle file from the given path """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

if __name__ == "__main__":
    x_train = load_pickle(os.path.join(PARAMS["tokenized_folder"], "x_train.pickle"))
    y_train = load_pickle(os.path.join(PARAMS["tokenized_folder"], "y_train.pickle"))
    x_val = load_pickle(os.path.join(PARAMS["tokenized_folder"], "x_val.pickle"))
    y_val = load_pickle(os.path.join(PARAMS["tokenized_folder"], "y_val.pickle"))

    if not os.path.exists(PARAMS["trained_folder"]):
        os.makedirs(PARAMS["trained_folder"])

    char_index = load_pickle(os.path.join(PARAMS["tokenized_folder"], "char_index.pickle"))
    training_model = create_model(len(char_index), len(PARAMS['categories']) - 1)

    training_model.compile(loss=PARAMS['loss_function'],
                           optimizer=PARAMS['optimizer'], metrics=['accuracy'])
    training_model.fit(x_train, y_train, batch_size=PARAMS['batch_size'],
                       epochs=PARAMS['epochs'], shuffle=True,
                       validation_data=(x_val, y_val))

    training_model.save(os.path.join(PARAMS["trained_folder"], "model.keras"))
