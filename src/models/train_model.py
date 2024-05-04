import pickle
import os
import dvc.api

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout


INPUT_DIR = "tokenized/"
OUPUT_DIR = "trained/"

def create_model(params):
    """ Create the model to be used for training

    Args:
        params (dict): parameters for the model

    Returns:
        Sequential: model to be used for training
    """

    with open(INPUT_DIR + "char_index.pickle", "rb") as f:
        char_index = pickle.load(f)

    model = Sequential()
    voc_size = len(char_index.keys())
    print(f"voc_size: {voc_size}")
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params['categories'])-1, activation='sigmoid'))

    return model


def load_pickle(path):
    """ Load the pickle file from the given path """    
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


if __name__ == "__main__":
    params = dvc.api.params_show()
    
    x_train = load_pickle(INPUT_DIR + "x_train.pickle")
    y_train = load_pickle(INPUT_DIR + "y_train.pickle")
    x_val = load_pickle(INPUT_DIR + "x_val.pickle")
    y_val = load_pickle(INPUT_DIR + "y_val.pickle")

    if not os.path.exists(OUPUT_DIR):
        os.makedirs(OUPUT_DIR)

    model = create_model(params)
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])
    model.fit(x_train, y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        shuffle=True,
        validation_data=(x_val, y_val)
        )
    model.save(OUPUT_DIR + "model.keras")
